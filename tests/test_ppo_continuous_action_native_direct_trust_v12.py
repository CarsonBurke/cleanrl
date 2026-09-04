import copy
import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Normal

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "native_pg" / "ppo_continuous_action_native_direct_trust_v12.py"
SPEC = importlib.util.spec_from_file_location("native_direct_trust_v12", SCRIPT)
v12 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v12)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


class OneStepTruncationEnv(gym.Env):
    observation_space = gym.spaces.Box(-100.0, 100.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([1.0, 2.0], dtype=np.float32), {}

    def step(self, action):
        del action
        return np.array([8.0, 9.0], dtype=np.float32), 1.0, False, True, {}


def actor_parameters(agent):
    return tuple(parameter for name, parameter in agent.named_parameters() if name.startswith("actor_"))


def test_defaults_keep_the_successful_fresh_native_geometry():
    args = v12.Args()

    assert args.env_id == "HalfCheetah-v4"
    assert args.learning_rate == 3e-4
    assert args.num_envs == 16
    assert args.num_steps == 128
    assert args.num_envs * args.num_steps == args.target_actor_batch_size == 2048
    assert args.sigma_mode == "state"
    assert args.max_mean_kl == 0.0
    assert args.max_grad_norm == 0.5
    assert args.update_epochs == 10
    assert args.num_minibatches == 32
    assert not args.anneal_lr
    assert not hasattr(args, "norm_adv")
    assert not hasattr(args, "reward_ema_decay")
    assert not hasattr(args, "actor_weighting")


def test_state_sigma_is_sac_bounded_state_dependent_and_initialized():
    torch.manual_seed(3)
    agent = v12.Agent(DummyVectorEnv(), sigma_mode="state")
    observations = torch.tensor([[0.0, 0.0, 0.0], [4.0, -3.0, 2.0]])

    _, logstd = agent.get_policy_parameters(observations)

    assert logstd.shape == (2, 2)
    assert torch.all(logstd >= v12.LOG_STD_MIN)
    assert torch.all(logstd <= v12.LOG_STD_MAX)
    assert torch.allclose(logstd[0], torch.full((2,), v12.INITIAL_LOG_STD))
    assert not torch.equal(logstd[0], logstd[1])


def test_global_sigma_is_per_action_but_constant_across_states():
    torch.manual_seed(5)
    agent = v12.Agent(DummyVectorEnv(), sigma_mode="global")
    observations = torch.randn(11, 3)

    _, logstd = agent.get_policy_parameters(observations)

    assert agent.actor_logstd_head is None
    assert agent.actor_logstd_param.shape == (1, 2)
    assert torch.allclose(logstd, torch.full((11, 2), v12.INITIAL_LOG_STD))
    with torch.no_grad():
        agent.actor_logstd_param[0, 0] += 0.2
    _, changed = agent.get_policy_parameters(observations)
    assert torch.all(changed[:, 0] == changed[0, 0])
    assert not torch.equal(changed[:, 0], logstd[:, 0])


def test_exact_tanh_logprob_replay_is_finite_at_saturation():
    agent = v12.Agent(DummyVectorEnv(), sigma_mode="state")
    with torch.no_grad():
        agent.actor_mean.weight.zero_()
        agent.actor_mean.bias.fill_(20.0)
    observations = torch.zeros(4, 3)

    actions, raw_actions, sampled_logprob, _ = agent.sample_action_and_value(observations)
    gaussian_logprob, replayed_logprob, mean, logstd = agent.action_logprobs(observations, raw_actions)
    expected_gaussian = Normal(mean, logstd.exp()).log_prob(raw_actions).sum(dim=-1)

    assert torch.all(actions == 1.0)
    assert torch.all(torch.isfinite(sampled_logprob))
    assert torch.equal(sampled_logprob, replayed_logprob)
    assert torch.equal(gaussian_logprob, expected_gaussian)

    try:
        agent.get_action_and_value(observations, action=actions)
    except ValueError as error:
        assert "raw_action" in str(error)
    else:
        raise AssertionError("stored tanh actions require their raw preimage")


def test_direct_loss_is_full_raw_policy_gradient_not_half_weighted():
    torch.manual_seed(7)
    agent = v12.Agent(DummyVectorEnv(), sigma_mode="state")
    observations = torch.randn(8, 3)
    raw_actions = torch.randn(8, 2)
    advantages = torch.linspace(-2.0, 2.0, 8)
    parameters = actor_parameters(agent)

    gaussian_logprob, _, _, _ = agent.action_logprobs(observations, raw_actions)
    direct_loss = -(advantages * gaussian_logprob).mean()
    direct_gradients = torch.autograd.grad(direct_loss, parameters)

    mean, logstd = agent.get_policy_parameters(observations)
    reference_logprob = Normal(mean, logstd.exp()).log_prob(raw_actions).sum(dim=-1)
    reference_loss = -(advantages * reference_logprob).mean()
    reference_gradients = torch.autograd.grad(reference_loss, parameters)

    for direct, reference in zip(direct_gradients, reference_gradients, strict=True):
        assert torch.allclose(direct, reference, atol=1e-6)


def test_interpolation_scales_one_adam_proposal_without_another_step():
    old = torch.nn.Parameter(torch.tensor([0.5, -1.0]))
    candidate = torch.nn.Parameter(old.detach().clone())
    scaled = torch.nn.Parameter(old.detach().clone())
    candidate_optimizer = torch.optim.Adam([candidate], lr=3e-4, eps=1e-5)
    scaled_optimizer = torch.optim.Adam([scaled], lr=0.25 * 3e-4, eps=1e-5)
    gradient = torch.tensor([2.0, -0.25])

    candidate.grad = gradient.clone()
    candidate_optimizer.step()
    candidate_value = candidate.detach().clone()
    v12.interpolate_parameters([candidate], [old.detach().clone()], [candidate_value], 0.25)

    scaled.grad = gradient.clone()
    scaled_optimizer.step()

    assert torch.allclose(candidate, scaled)
    candidate_state = candidate_optimizer.state[candidate]
    scaled_state = scaled_optimizer.state[scaled]
    assert candidate_state["step"] == scaled_state["step"] == 1
    assert torch.equal(candidate_state["exp_avg"], scaled_state["exp_avg"])
    assert torch.equal(candidate_state["exp_avg_sq"], scaled_state["exp_avg_sq"])


def test_backtrack_accepts_a_fraction_without_another_optimizer_step():
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    old = [parameter.detach().clone()]
    candidate = [torch.tensor([2.0])]
    parameter.data.copy_(candidate[0])
    evaluations = 0

    def evaluate():
        nonlocal evaluations
        evaluations += 1
        kl = parameter.square()
        return (kl,)

    accepted, fraction, backtracks, policy_change, proposal_kl = v12.backtrack_parameters(
        [parameter], old, candidate, evaluate, 0.25, 0.5, 4
    )

    assert accepted
    assert fraction == 0.25
    assert backtracks == 2
    assert evaluations == 3
    assert torch.equal(parameter, torch.tensor([0.5]))
    assert policy_change[-1].item() == 0.25
    assert proposal_kl.item() == 4.0


def test_nonfinite_rejection_exactly_restores_parameters_and_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
    optimizer = torch.optim.Adam([parameter], lr=3e-4)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    old = [parameter.detach().clone()]
    parameter.grad = torch.tensor([float("nan"), 1.0])
    optimizer.step()
    candidate = [parameter.detach().clone()]

    def evaluate():
        return (parameter.square(),)

    accepted, fraction, _, _, _ = v12.backtrack_parameters([parameter], old, candidate, evaluate, 0.01, 0.5, 3)
    if not accepted:
        optimizer.load_state_dict(optimizer_state)

    assert not accepted
    assert fraction == 0.0
    assert torch.equal(parameter, old[0])
    assert len(optimizer.state) == 0


def test_disabled_trust_region_still_rejects_a_nonfinite_proposal():
    parameter = torch.nn.Parameter(torch.tensor([float("inf")]))
    old = [torch.tensor([1.0])]
    candidate = [parameter.detach().clone()]

    def evaluate():
        return (parameter.square(),)

    accepted, fraction, _, policy_change, proposal_kl = v12.backtrack_parameters(
        [parameter], old, candidate, evaluate, 0.0, 0.5, 0
    )

    assert not accepted
    assert fraction == 0.0
    assert torch.equal(parameter, old[0])
    assert torch.isfinite(policy_change[-1]).all()
    assert not torch.isfinite(proposal_kl).all()


def test_source_has_one_actor_backward_and_no_reward_or_advantage_normalization():
    source = SCRIPT.read_text()

    assert "NormalizeReward" not in source
    assert "TransformReward" not in source
    assert "actor_loss = -(b_advantages * gaussian_logprob).mean()" in source
    assert source.count("actor_loss.backward()") == 1
    assert source.count("actor_optimizer.step()") == 1
    assert "interpolate_parameters(" in source
    assert "backtrack_parameters(" in source
    assert "kl_divergence(" in source
    assert "advantages - advantages.mean" not in source
    assert "b_advantages /" not in source
    assert "0.5 * b_advantages" not in source
    assert "critic_loss = 0.5 * nn.functional.mse_loss(" in source
    assert "v_loss_clipped" not in source


BASE_SCRIPT = Path(__file__).parents[1] / "cleanrl" / "ppo_continuous_action.py"
BASE_SPEC = importlib.util.spec_from_file_location("base_ppo_continuous_action", BASE_SCRIPT)
base_ppo = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base_ppo)


def test_bootstrap_observations_use_final_state_only_for_truncations():
    next_obs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    infos = {
        "final_observation": np.array(
            [
                np.array([8.0, 9.0], dtype=np.float32),
                np.array([10.0, 11.0], dtype=np.float32),
                None,
            ],
            dtype=object,
        ),
        "_final_observation": np.array([True, True, False]),
    }
    truncations = np.array([True, False, False])

    for module in (v12, base_ppo):
        result = module.bootstrap_observations(next_obs, truncations, infos)
        np.testing.assert_array_equal(result[0], np.array([8.0, 9.0]))
        np.testing.assert_array_equal(result[1:], next_obs[1:])


def test_missing_final_observation_for_truncation_is_an_error():
    next_obs = np.zeros((2, 3), dtype=np.float32)
    truncations = np.array([False, True])

    for module in (v12, base_ppo):
        try:
            module.bootstrap_observations(next_obs, truncations, {})
        except RuntimeError as error:
            assert "final_observation" in str(error)
        else:
            raise AssertionError("a truncation must never bootstrap from autoreset state")


def test_sync_vector_env_exposes_wrapped_final_observation_for_bootstrap():
    def make_wrapped_env():
        env = OneStepTruncationEnv()
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        return gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

    envs = gym.vector.SyncVectorEnv([make_wrapped_env])
    try:
        envs.reset(seed=1)
        autoreset_obs, _, terminations, truncations, infos = envs.step(np.zeros((1, 1), dtype=np.float32))
        assert not terminations[0]
        assert truncations[0]
        assert infos["_final_observation"][0]
        assert not np.array_equal(autoreset_obs[0], infos["final_observation"][0])

        for module in (v12, base_ppo):
            bootstrap_obs = module.bootstrap_observations(autoreset_obs, truncations, infos)
            np.testing.assert_allclose(bootstrap_obs[0], infos["final_observation"][0], rtol=1e-6)
    finally:
        envs.close()


def test_gae_bootstraps_truncations_but_cuts_both_reset_boundaries():
    rewards = torch.ones(3, 3)
    values = torch.zeros_like(rewards)
    terminations = torch.zeros_like(rewards)
    truncations = torch.zeros_like(rewards)
    terminations[1, 2] = 1.0
    truncations[1, 1] = 1.0
    truncation_values = torch.zeros_like(rewards)
    truncation_values[1, 1] = 10.0
    truncation_values[1, 2] = 1000.0
    rollout_tail_value = torch.full((3,), 4.0)
    expected_advantages = torch.tensor([[7.0, 12.0, 2.0], [6.0, 11.0, 1.0], [5.0, 5.0, 5.0]])

    for module in (v12, base_ppo):
        advantages, returns = module.compute_gae(
            rewards,
            values,
            terminations,
            truncations,
            truncation_values,
            rollout_tail_value,
            gamma=1.0,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(advantages, expected_advantages)
        torch.testing.assert_close(returns, expected_advantages)


def test_rollout_tail_truncation_uses_final_value_not_autoreset_tail_value():
    rewards = torch.tensor([[2.0, 2.0, 2.0]])
    values = torch.tensor([[3.0, 3.0, 3.0]])
    terminations = torch.tensor([[0.0, 1.0, 1.0]])
    truncations = torch.tensor([[1.0, 0.0, 1.0]])
    truncation_values = torch.tensor([[5.0, 500.0, 500.0]])
    autoreset_tail_values = torch.tensor([1000.0, 1000.0, 1000.0])
    expected = torch.tensor([[3.5, -1.0, -1.0]])

    for module in (v12, base_ppo):
        advantages, _ = module.compute_gae(
            rewards,
            values,
            terminations,
            truncations,
            truncation_values,
            autoreset_tail_values,
            gamma=0.9,
            gae_lambda=0.95,
        )
        torch.testing.assert_close(advantages, expected)


def test_no_boundary_gae_matches_the_previous_recursion():
    generator = torch.Generator().manual_seed(19)
    rewards = torch.randn(7, 4, generator=generator)
    values = torch.randn(7, 4, generator=generator)
    tail_value = torch.randn(4, generator=generator)
    zeros = torch.zeros_like(rewards)
    stale_truncation_values = torch.full_like(rewards, torch.nan)
    gamma = 0.97
    gae_lambda = 0.91

    expected = torch.zeros_like(rewards)
    last_advantage = torch.zeros(4)
    for t in reversed(range(rewards.shape[0])):
        next_value = tail_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last_advantage = delta + gamma * gae_lambda * last_advantage
        expected[t] = last_advantage

    for module in (v12, base_ppo):
        advantages, returns = module.compute_gae(
            rewards,
            values,
            zeros,
            zeros,
            stale_truncation_values,
            tail_value,
            gamma,
            gae_lambda,
        )
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(returns, expected + values)

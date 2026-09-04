import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl" / "delightful" / "ppo" / "ppo_continuous_action_delightful_ppo_tanh_boundedreward_statesigma_v8.py"
)
SPEC = importlib.util.spec_from_file_location(
    "delightful_ppo_tanh_boundedreward_statesigma_v8", SCRIPT
)
dg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dg)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(3,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def test_args_encode_requested_scale_choices():
    args = dg.Args()
    assert not hasattr(args, "norm_adv")
    assert args.num_envs == 16
    assert args.num_steps == 128
    assert args.num_envs * args.num_steps == args.target_actor_batch_size == 2048
    assert args.num_minibatches == 32
    assert args.target_actor_batch_size // args.num_minibatches == 64
    assert not args.anneal_lr
    assert args.learning_rate == 3e-4
    assert not hasattr(args, "reward_ema_decay")
    assert not hasattr(args, "reward_norm_eps")
    assert not hasattr(args, "scale_gae_kernel")
    assert args.delight_temperature == 1.0
    assert args.surprisal_clip == 10.0


def test_reward_projection_matches_control_suite_cheetah_run_tolerance():
    normalized = dg.control_suite_cheetah_reward(
        np.array([-5.0, 0.0, 2.5, 5.0, 10.0, 20.0], dtype=np.float32)
    )

    assert np.array_equal(
        normalized, np.array([0.0, 0.0, 0.25, 0.5, 1.0, 1.0])
    )
    assert normalized.min() == dg.REWARD_LOWER_BOUND
    assert normalized.max() == dg.REWARD_UPPER_BOUND
    assert np.all(normalized >= dg.REWARD_LOWER_BOUND)
    assert np.all(normalized <= dg.REWARD_UPPER_BOUND)


def test_forward_speed_extraction_handles_vector_autoreset_info():
    regular_infos = {
        "x_velocity": np.array([1.5, 2.5], dtype=np.float32),
        "_x_velocity": np.array([True, True]),
    }
    assert np.array_equal(
        dg.extract_forward_speeds(regular_infos, 2),
        np.array([1.5, 2.5], dtype=np.float32),
    )

    mixed_infos = {
        "x_velocity": np.array([1.5, 0.0], dtype=np.float32),
        "_x_velocity": np.array([True, False]),
        "final_info": np.array(
            [None, {"x_velocity": 3.5}], dtype=object
        ),
        "_final_info": np.array([False, True]),
    }
    assert np.array_equal(
        dg.extract_forward_speeds(mixed_infos, 2),
        np.array([1.5, 3.5], dtype=np.float32),
    )

    all_final_infos = {
        "final_info": np.array(
            [{"x_velocity": 4.5}, {"x_velocity": 5.5}], dtype=object
        ),
        "_final_info": np.array([True, True]),
    }
    assert np.array_equal(
        dg.extract_forward_speeds(all_final_infos, 2),
        np.array([4.5, 5.5], dtype=np.float32),
    )


def test_delightful_gate_matches_continuous_action_algorithm_without_whitening():
    advantages = torch.tensor([2.0, -3.0, 1.0])
    logprobs = torch.tensor([-20.0, -2.0, 4.0], requires_grad=True)

    gate, surprisal, delight = dg.delightful_gate(advantages, logprobs)

    expected_surprisal = torch.tensor([10.0, 2.0, -4.0])
    expected_delight = advantages * expected_surprisal
    assert torch.equal(surprisal, expected_surprisal)
    assert torch.equal(delight, expected_delight)
    assert torch.allclose(gate, torch.sigmoid(expected_delight))
    assert not gate.requires_grad


def test_gate_weights_the_standard_joint_ratio_ppo_surrogate():
    advantages = torch.tensor([2.0, -2.0])
    ratios = torch.tensor([1.5, 0.5])
    gate = torch.tensor([0.75, 0.25])

    loss = dg.delightful_ppo_loss(advantages, ratios, gate, clip_coef=0.2)

    expected_per_sample = torch.tensor([-2.4, 1.6]) * gate
    assert torch.allclose(loss, expected_per_sample.mean())


def test_policy_log_standard_deviation_is_state_dependent_and_bounded():
    torch.manual_seed(7)
    agent = dg.Agent(DummyVectorEnv())
    observations = torch.tensor([[0.0, 0.0, 0.0], [4.0, -3.0, 2.0]])

    _, logstd = agent.get_action_distribution(observations)

    assert logstd.shape == (2, 2)
    assert not torch.equal(logstd[0], logstd[1])
    assert torch.all(logstd >= dg.LOG_STD_MIN)
    assert torch.all(logstd <= dg.LOG_STD_MAX)
    assert torch.allclose(logstd[0], torch.full((2,), dg.INITIAL_LOG_STD))
    logstd.sum().backward()
    assert agent.actor_logstd.weight.grad is not None


def test_sac_midpoint_initialization_exposes_negative_density_surprisal():
    agent = dg.Agent(DummyVectorEnv())
    observations = torch.zeros((1, 3))
    distribution, logstd = agent.get_action_distribution(observations)
    raw_action = distribution.mean

    _, logprob = agent._action_and_logprob_from_raw(distribution, raw_action)

    assert torch.allclose(logstd, torch.full_like(logstd, dg.INITIAL_LOG_STD))
    assert (-logprob).item() < 0.0


def test_squashed_action_and_replayed_logprob_match_sac_change_of_variables():
    torch.manual_seed(11)
    agent = dg.Agent(DummyVectorEnv())
    observations = torch.tensor([[0.5, -0.25, 1.0], [-1.0, 0.75, 0.25]])

    actions, raw_actions, sampled_logprob, _, _ = agent.sample_action_and_value(
        observations
    )
    _, replayed_logprob, _, _ = agent.get_action_and_value(
        observations, actions, raw_action=raw_actions
    )
    base_distribution, _ = agent.get_action_distribution(observations)
    log_tanh_jacobian = 2.0 * (
        np.log(2.0) - raw_actions - torch.nn.functional.softplus(-2.0 * raw_actions)
    )
    expected_logprob = (
        base_distribution.log_prob(raw_actions)
        - log_tanh_jacobian
        - torch.log(agent.action_scale)
    ).sum(1)

    assert torch.all(actions > -1.0)
    assert torch.all(actions < 1.0)
    assert torch.allclose(sampled_logprob, replayed_logprob, atol=1e-5)
    assert torch.allclose(sampled_logprob, expected_logprob, atol=1e-5)


def test_saturated_tanh_action_preserves_exact_fresh_ratio_via_raw_latent():
    agent = dg.Agent(DummyVectorEnv())
    with torch.no_grad():
        agent.actor_mean.weight.zero_()
        agent.actor_mean.bias.fill_(20.0)
    observations = torch.zeros((2, 3))

    actions, raw_actions, rollout_logprob, _, _ = agent.sample_action_and_value(
        observations
    )
    _, replayed_logprob, _, _ = agent.get_action_and_value(
        observations, actions, raw_action=raw_actions
    )

    assert torch.all(actions == 1.0)
    assert torch.all(torch.isfinite(rollout_logprob))
    assert torch.equal(rollout_logprob, replayed_logprob)
    assert torch.equal((replayed_logprob - rollout_logprob).exp(), torch.ones(2))

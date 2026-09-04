import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Normal


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl" / "delightful" / "ppo" / "ppo_continuous_action_delightful_ppo_native_tail_meanonly_statesigma_v10.py"
)
SPEC = importlib.util.spec_from_file_location(
    "delightful_ppo_native_tail_meanonly_statesigma_v10", SCRIPT
)
dg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dg)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(3,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def actor_parameters(agent):
    return tuple(agent.actor_trunk.parameters()) + tuple(
        agent.actor_mean.parameters()
    ) + tuple(agent.actor_logstd.parameters())


def test_defaults_encode_the_native_fresh_batch_design():
    args = dg.Args()

    assert args.env_id == "HalfCheetah-v4"
    assert args.actor_weighting == "tail-dg"
    assert args.learning_rate == 3e-4
    assert not args.anneal_lr
    assert args.num_envs == 16
    assert args.num_steps == 128
    assert args.num_envs * args.num_steps == args.target_actor_batch_size == 2048
    assert args.update_epochs == 10
    assert args.num_minibatches == 32
    assert args.max_grad_norm == 0.5
    assert not hasattr(args, "norm_adv")
    assert not hasattr(args, "reward_ema_decay")
    assert not hasattr(args, "delight_temperature")
    assert not hasattr(args, "surprisal_clip")
    assert not hasattr(args, "clip_vloss")


def test_tail_surprisal_matches_exact_six_dimensional_survival():
    standardized = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -2.0, 0.5, -0.25, 1.5, -0.75],
        ]
    )
    mean = torch.tensor(
        [[2.0, -1.0, 0.5, 3.0, -4.0, 1.0], [-1.0, 2.0, 4.0, -3.0, 1.0, 0.0]]
    )
    std = torch.tensor(
        [[0.5, 2.0, 1.5, 0.25, 3.0, 0.75], [2.0, 0.5, 0.25, 4.0, 1.5, 3.0]]
    )
    raw_action = mean + std * standardized

    q, rho = dg.latent_tail_statistics(raw_action, mean, std)

    expected_q = standardized.square().sum(dim=-1)
    x = expected_q / 2.0
    expected_rho = x - torch.log1p(x + 0.5 * x.square())
    assert torch.allclose(q, expected_q)
    assert torch.allclose(rho, expected_rho, atol=1e-6)
    assert not q.requires_grad
    assert not rho.requires_grad


def test_tail_surprisal_is_affine_scale_invariant_and_dimension_calibrated():
    torch.manual_seed(19)
    standardized = torch.randn(64, 3)
    mean_a = torch.zeros_like(standardized)
    std_a = torch.ones_like(standardized)
    mean_b = torch.randn_like(standardized)
    std_b = torch.rand_like(standardized) * 2.0 + 0.1

    q_a, rho_a = dg.latent_tail_statistics(standardized, mean_a, std_a)
    q_b, rho_b = dg.latent_tail_statistics(
        mean_b + std_b * standardized, mean_b, std_b
    )

    expected_rho = -torch.log(
        torch.special.gammaincc(
            torch.full_like(q_a.double(), 1.5), 0.5 * q_a.double()
        )
    ).float()
    assert torch.allclose(q_a, q_b, atol=1e-5)
    assert torch.allclose(rho_a, rho_b, atol=1e-5)
    assert torch.allclose(rho_a, expected_rho, atol=1e-6)


def test_tail_gate_is_reward_unit_invariant_but_outer_weight_remains_raw():
    advantages = torch.tensor([2.0, -1.0, 0.5, -4.0])
    rho = torch.tensor([0.25, 0.75, 1.5, 3.0])

    outputs = dg.native_score_weights(advantages, rho, "tail-dg")
    scaled_outputs = dg.native_score_weights(7.0 * advantages, rho, "tail-dg")
    mean_weight, scale_weight, gate, candidate_gate, delight, eta, logits = outputs
    (
        scaled_mean_weight,
        scaled_scale_weight,
        scaled_gate,
        scaled_candidate_gate,
        scaled_delight,
        scaled_eta,
        scaled_logits,
    ) = scaled_outputs

    assert torch.allclose(gate, scaled_gate)
    assert torch.allclose(candidate_gate, scaled_candidate_gate)
    assert torch.allclose(logits, scaled_logits)
    assert torch.allclose(scaled_eta, 7.0 * eta)
    assert torch.allclose(scaled_delight, 7.0 * delight)
    assert torch.allclose(scaled_mean_weight, 7.0 * mean_weight)
    assert torch.allclose(scaled_scale_weight, 7.0 * scale_weight)
    assert torch.allclose(logits.abs().mean(), torch.tensor(1.0))
    assert not gate.requires_grad
    assert not mean_weight.requires_grad


def test_zero_delight_and_neutral_control_apply_exact_half_gate():
    advantages = torch.tensor([0.0, 0.0])
    rho = torch.tensor([1.0, 3.0])
    outputs = dg.native_score_weights(advantages, rho, "tail-dg")
    assert outputs[5].item() == 0.0
    assert torch.equal(outputs[2], torch.full((2,), 0.5))
    assert torch.equal(outputs[6], torch.zeros(2))

    advantages = torch.tensor([2.0, -4.0])
    outputs = dg.native_score_weights(advantages, rho, "neutral")
    mean_weight, scale_weight, applied_gate, candidate_gate = outputs[:4]
    assert torch.equal(applied_gate, torch.full((2,), 0.5))
    assert torch.equal(mean_weight, 0.5 * advantages)
    assert torch.equal(scale_weight, 0.5 * advantages)
    assert not torch.equal(candidate_gate, applied_gate)


def test_neutral_decomposed_score_equals_half_the_full_gaussian_score():
    torch.manual_seed(23)
    agent = dg.Agent(DummyVectorEnv())
    observations = torch.tensor([[0.5, -0.25, 1.0], [-1.0, 0.75, 0.25]])
    raw_actions = torch.tensor([[0.3, -0.7], [1.1, -0.2]])
    advantages = torch.tensor([1.25, -0.75])
    parameters = actor_parameters(agent)

    mean_logprob, scale_logprob, _, _ = agent.decomposed_score_logprobs(
        observations, raw_actions
    )
    decomposed_loss = -(
        0.5 * advantages * mean_logprob + 0.5 * advantages * scale_logprob
    ).mean()
    decomposed_gradients = torch.autograd.grad(decomposed_loss, parameters)

    distribution, _ = agent.get_action_distribution(observations)
    full_logprob = distribution.log_prob(raw_actions).sum(dim=-1)
    full_loss = -(0.5 * advantages * full_logprob).mean()
    full_gradients = torch.autograd.grad(full_loss, parameters)

    for decomposed, full in zip(decomposed_gradients, full_gradients, strict=True):
        assert torch.allclose(decomposed, full, atol=1e-6)


def test_gate_changes_mean_head_gradient_but_not_logstd_head_gradient():
    torch.manual_seed(29)
    agent = dg.Agent(DummyVectorEnv())
    observations = torch.tensor([[0.5, -0.25, 1.0], [-1.0, 0.75, 0.25]])
    raw_actions = torch.tensor([[0.3, -0.7], [1.1, -0.2]])
    advantages = torch.tensor([1.25, -0.75])

    def head_gradients(mean_gate):
        mean_logprob, scale_logprob, _, _ = agent.decomposed_score_logprobs(
            observations, raw_actions
        )
        loss = -(
            advantages * mean_gate * mean_logprob
            + 0.5 * advantages * scale_logprob
        ).mean()
        return torch.autograd.grad(
            loss,
            (agent.actor_mean.weight, agent.actor_logstd.weight),
        )

    neutral_mean_grad, neutral_scale_grad = head_gradients(torch.full((2,), 0.5))
    gated_mean_grad, gated_scale_grad = head_gradients(torch.tensor([0.9, 0.1]))

    assert not torch.allclose(neutral_mean_grad, gated_mean_grad)
    assert torch.allclose(neutral_scale_grad, gated_scale_grad, atol=1e-7)


def test_policy_logstd_is_state_dependent_sac_bounded_and_initialized():
    torch.manual_seed(31)
    agent = dg.Agent(DummyVectorEnv())
    observations = torch.tensor([[0.0, 0.0, 0.0], [4.0, -3.0, 2.0]])

    _, logstd = agent.get_action_distribution(observations)

    assert logstd.shape == (2, 2)
    assert torch.all(logstd >= dg.LOG_STD_MIN)
    assert torch.all(logstd <= dg.LOG_STD_MAX)
    assert torch.allclose(logstd[0], torch.full((2,), dg.INITIAL_LOG_STD))
    assert not torch.equal(logstd[0], logstd[1])


def test_tanh_action_logprob_replay_is_exact_even_at_saturation():
    agent = dg.Agent(DummyVectorEnv())
    with torch.no_grad():
        agent.actor_mean.weight.zero_()
        agent.actor_mean.bias.fill_(20.0)
    observations = torch.zeros((2, 3))

    actions, raw_actions, sampled_logprob, _, _ = agent.sample_action_and_value(
        observations
    )
    replayed_logprob, mean, logstd = agent.exact_action_logprob(
        observations, raw_actions
    )
    distribution = Normal(mean, logstd.exp())
    _, expected_logprob, _ = agent._action_and_logprob_from_raw(
        distribution, raw_actions
    )

    assert torch.all(actions == 1.0)
    assert torch.all(torch.isfinite(sampled_logprob))
    assert torch.equal(sampled_logprob, replayed_logprob)
    assert torch.equal(replayed_logprob, expected_logprob)


def test_training_source_uses_native_reward_direct_score_and_unclipped_value_mse():
    source = SCRIPT.read_text()

    assert "NormalizeReward" not in source
    assert "control_suite_cheetah_reward" not in source
    assert "rewards[step] = torch.as_tensor(" in source
    assert "reward, dtype=torch.float32, device=device" in source
    assert "mean_loss = -(mean_weight * mean_score_logprob).mean()" in source
    assert "scale_loss = -(scale_weight * scale_score_logprob).mean()" in source
    assert "old_mean = old_mean.detach().clone()" in source
    assert "old_logstd = old_logstd.detach().clone()" in source
    assert "post_logstd = post_logstd.clone()" in source
    assert "critic_loss = 0.5 * torch.nn.functional.mse_loss(" in source
    assert "v_loss_clipped" not in source
    assert "newvalue - b_values" not in source

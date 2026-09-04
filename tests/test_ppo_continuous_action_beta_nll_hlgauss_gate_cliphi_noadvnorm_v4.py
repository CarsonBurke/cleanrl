import gymnasium as gym
import numpy as np
import torch

from cleanrl.beta_policy.ppo_continuous_action_beta_nll_hlgauss_gate_cliphi_noadvnorm_v4 import (
    Agent,
    Args,
    SAMPLE_EPS,
    beta_nll_to_gate,
    clipped_policy_loss,
    hl_value_loss,
    hlgauss_coord_bounds,
    make_hlgauss_support,
    normalized_return_abs_bound,
)


class DummyVecEnv:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
    single_action_space = gym.spaces.Box(
        np.array([-2.0, -0.5], dtype=np.float32),
        np.array([3.0, 1.5], dtype=np.float32),
        dtype=np.float32,
    )


def test_beta_policy_samples_native_z_and_maps_to_action_bounds():
    torch.manual_seed(0)
    agent = Agent(DummyVecEnv(), Args(num_bins=31))
    obs = torch.randn(32, 5)

    action, z, logprob, entropy, value, beta_nll, value_logits = agent.get_beta_action_and_value(obs)

    low = torch.as_tensor(DummyVecEnv.single_action_space.low)
    high = torch.as_tensor(DummyVecEnv.single_action_space.high)
    assert action.shape == z.shape == (32, 2)
    assert torch.all(z > 0.0) and torch.all(z < 1.0)
    assert torch.all(action >= low - 1e-6)
    assert torch.all(action <= high + 1e-6)
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(beta_nll).all()
    assert value.shape == (32, 1)
    assert value_logits.shape == (32, 31)


def test_v4_defaults_use_requested_ablations_hlgauss_and_bounded_gate():
    args = Args()
    assert args.norm_adv is False
    assert args.clip_coef == 0.2
    assert args.clip_coef_high == 0.28
    assert args.beta_nll_gate_coef == 1.0
    assert args.beta_nll_gate_adv_clip == 5.0
    assert args.beta_nll_gate_surprisal_clip == 5.0
    assert args.num_bins == 511
    assert args.value_sigma_to_bin_ratio == 0.75


def test_replaying_stored_z_recomputes_same_logprob():
    torch.manual_seed(1)
    agent = Agent(DummyVecEnv(), Args(num_bins=31))
    obs = torch.randn(16, 5)

    _, z, logprob, _, _, _, _ = agent.get_beta_action_and_value(obs)
    _, replay_z, replay_logprob, _, _, _, _ = agent.get_beta_action_and_value(obs, z)

    assert torch.allclose(replay_z, z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS))
    assert torch.allclose(replay_logprob, logprob, atol=1e-6)


def test_standard_action_api_keeps_cleanrl_eval_contract():
    torch.manual_seed(1)
    agent = Agent(DummyVecEnv(), Args(num_bins=31))
    obs = torch.randn(16, 5)

    action, logprob, entropy, value = agent.get_action_and_value(obs)
    replay_action, replay_logprob, _, _ = agent.get_action_and_value(obs, action)

    assert action.shape == replay_action.shape == (16, 2)
    assert torch.allclose(replay_action, action, atol=1e-6)
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()
    assert torch.allclose(replay_logprob, logprob, atol=1e-6)


def test_beta_nll_gate_is_finite_detached_bounded_and_sign_aware():
    advantages = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    beta_nll = torch.ones(4)

    clipped_nll, gate, adv_z, surprisal = beta_nll_to_gate(
        advantages,
        beta_nll,
        clip=10.0,
        gate_coef=1.0,
        adv_clip=5.0,
        surprisal_clip=5.0,
    )

    assert torch.allclose(clipped_nll, beta_nll)
    assert torch.isfinite(gate).all()
    assert torch.isfinite(adv_z).all()
    assert torch.isfinite(surprisal).all()
    assert gate.requires_grad is False
    assert torch.all(gate > 0.0)
    assert torch.all(gate < 2.0)
    assert torch.all(gate[:2] < 1.0)
    assert torch.all(gate[2:] > 1.0)


def test_beta_nll_gate_uses_surprisal_to_strengthen_rare_successes_and_failures():
    advantages = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    beta_nll = torch.tensor([0.1, 2.0, 0.1, 2.0])

    _, gate, _, surprisal = beta_nll_to_gate(
        advantages,
        beta_nll,
        clip=10.0,
        gate_coef=1.0,
        adv_clip=5.0,
        surprisal_clip=5.0,
    )

    assert surprisal[1] > surprisal[0]
    assert surprisal[3] > surprisal[2]
    assert gate[1] < gate[0] < 1.0
    assert gate[3] > gate[2] > 1.0


def test_beta_nll_weights_legacy_helper_remains_finite_detached_and_mean_normalized():
    torch.manual_seed(2)
    agent = Agent(DummyVecEnv(), Args(num_bins=31))
    obs = torch.randn(64, 5)
    dist = agent._dist(obs)
    z = torch.linspace(0.05, 0.95, steps=64).unsqueeze(1).expand(-1, 2)

    beta_nll, weights = agent.beta_nll_weights(dist, z, clip=10.0, weight_min=0.0, weight_max=100.0)

    assert torch.isfinite(beta_nll).all()
    assert torch.isfinite(weights).all()
    assert torch.all(beta_nll >= 0.0)
    assert weights.requires_grad is False
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-5)


def test_zero_beta_nll_falls_back_to_unit_weights():
    agent = Agent(DummyVecEnv(), Args(num_bins=31))
    obs = torch.randn(8, 5)
    dist = agent._dist(obs)
    alpha = dist.concentration1
    beta = dist.concentration0
    mode = ((alpha - 1.0) / (alpha + beta - 2.0).clamp_min(SAMPLE_EPS)).clamp(
        SAMPLE_EPS, 1.0 - SAMPLE_EPS
    )

    _, weights = agent.beta_nll_weights(dist, mode, clip=10.0, weight_min=0.25, weight_max=4.0)

    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)


def test_update_gate_uses_stored_behavior_nll_not_live_policy():
    advantages = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    stored_nll = torch.tensor([0.5, 1.0, 1.5, 2.0])
    live_policy_nll = torch.tensor([9.0, 1.0, 1.0, 1.0])

    _, behavior_gate, _, _ = beta_nll_to_gate(
        advantages,
        stored_nll,
        clip=10.0,
        gate_coef=1.0,
        adv_clip=5.0,
        surprisal_clip=5.0,
    )
    _, live_gate, _, _ = beta_nll_to_gate(
        advantages,
        live_policy_nll,
        clip=10.0,
        gate_coef=1.0,
        adv_clip=5.0,
        surprisal_clip=5.0,
    )

    assert not torch.allclose(behavior_gate, live_gate)
    assert behavior_gate.requires_grad is False


def test_policy_loss_uses_asymmetric_clip_higher_bound():
    adv = torch.tensor([1.0, 1.0, 1.0, -1.0])
    ratio = torch.tensor([0.75, 1.25, 1.35, 0.70], requires_grad=True)

    loss = clipped_policy_loss(adv, ratio, clip_coef=0.2, clip_coef_high=0.28)
    expected = torch.max(
        -adv * ratio,
        -adv * torch.tensor([0.8, 1.25, 1.28, 0.8]),
    ).mean()

    assert torch.allclose(loss, expected)


def test_hlgauss_default_support_matches_normalized_clipped_return_range():
    args = Args(gamma=0.99, reward_clip=10.0, return_abs_bound=0.0, num_bins=31)

    assert np.isclose(normalized_return_abs_bound(args), 1000.0)

    coord_min, coord_max = hlgauss_coord_bounds(args)
    assert np.isclose(coord_min, -coord_max)
    assert np.isclose(np.expm1(coord_max), 1000.0, rtol=1e-6)

    support = make_hlgauss_support(args, torch.device("cpu"))
    assert support.support.shape == (31,)
    assert torch.isclose(support.support[0], torch.tensor(-1000.0), atol=1e-3)
    assert torch.isclose(support.support[-1], torch.tensor(1000.0), atol=1e-3)
    assert torch.isclose(support.support[15], torch.tensor(0.0), atol=1e-6)


def test_unnormalized_reward_requires_explicit_return_range():
    args = Args(norm_reward=False, return_abs_bound=0.0)

    try:
        normalized_return_abs_bound(args)
    except ValueError as exc:
        assert "return_abs_bound" in str(exc)
    else:
        raise AssertionError("expected explicit return range guard")

    args.return_abs_bound = 20000.0
    assert normalized_return_abs_bound(args) == 20000.0


def test_hlgauss_value_loss_is_finite_and_trains_critic_head():
    torch.manual_seed(3)
    args = Args(num_bins=31)
    agent = Agent(DummyVecEnv(), args)
    obs = torch.randn(16, 5)
    returns = torch.linspace(-50.0, 50.0, steps=16)

    value_logits = agent.get_value_logits(obs)
    loss = hl_value_loss(value_logits, returns, agent.value_support(obs.device))
    loss.backward()

    assert torch.isfinite(loss)
    assert agent.critic_head.weight.grad is not None
    assert torch.isfinite(agent.critic_head.weight.grad).all()

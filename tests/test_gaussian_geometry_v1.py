"""Numerical contracts for joint Gaussian geometry, host sampling, and epoch projection."""

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Independent, Normal, kl_divergence

from cleanrl import ppo_continuous_action_gaussian_geometry_v1 as geometry


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def deterministic_runtime():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(71)
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def tensor(values):
    return torch.tensor(values, device="cuda", dtype=torch.float64)


def distribution(mean, log_std):
    return Independent(Normal(mean, log_std.exp()), 1)


def make_agent(**kwargs):
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            low=np.array([-3.0, 0.5, -0.2, -7.0], dtype=np.float32),
            high=np.array([1.0, 5.5, 0.8, -1.0], dtype=np.float32),
        ),
    )
    agent = geometry.Agent(envs, geometry.Args(**kwargs)).cuda()
    # Deterministic, genuinely state-dependent outputs through both tanh layers.
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor[0].weight[:3].copy_(torch.eye(3, device="cuda"))
        agent.actor[2].weight[:3, :3].copy_(1.3 * torch.eye(3, device="cuda"))
        agent.actor[4].weight[:, :3].copy_(torch.tensor(
            [[0.7, -0.2, 0.1], [-0.4, 0.6, 0.2], [0.1, 0.3, -0.5], [0.2, -0.1, 0.4],
             [0.15, -0.1, 0.05], [-0.1, 0.2, 0.1], [0.05, 0.1, -0.15], [0.1, -0.05, 0.2]],
            device="cuda",
        ))
        agent.actor[4].bias.copy_(torch.tensor(
            [0.1, -0.2, 0.05, 0.3, -0.1, 0.05, 0.1, -0.05], device="cuda",
        ))
    return agent


def test_forward_kl_decomposes_against_analytic_unequal_gaussians():
    old_mean = tensor([[0.3, -1.2, 2.1], [-0.7, 0.2, 1.3]])
    mean = tensor([[-0.2, -0.8, 1.0], [0.1, -1.0, 1.5]])
    old_log_std = tensor([[0.4, 1.2, 0.08], [1.7, 0.03, 0.8]]).log()
    log_std = tensor([[1.1, 0.2, 0.7], [0.4, 0.09, 1.5]]).log()
    mean_kl, scale_kl = geometry.gaussian_kl_parts(old_mean, old_log_std, mean, log_std)
    expected_mean = kl_divergence(distribution(old_mean, log_std), distribution(mean, log_std))
    expected_scale = kl_divergence(distribution(old_mean, old_log_std), distribution(old_mean, log_std))
    expected_joint = kl_divergence(distribution(old_mean, old_log_std), distribution(mean, log_std))
    torch.testing.assert_close(mean_kl, expected_mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(scale_kl, expected_scale, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(mean_kl + scale_kl, expected_joint, rtol=1e-12, atol=1e-12)


def test_joint_ratio_and_physical_jacobian_cancel_even_at_tanh_saturation():
    agent = make_agent().double()
    native = tensor([[0.2, -0.5, 1.0, -1.3], [40.0, -45.0, 50.0, -55.0]])
    old_mean = native - tensor([0.2, -0.3, 0.1, -0.4])
    mean = old_mean + tensor([0.15, -0.1, 0.2, -0.05])
    old_log_std = tensor([0.4, 0.7, 1.1, 0.5]).log().expand_as(native)
    log_std = tensor([0.6, 0.5, 0.8, 0.7]).log().expand_as(native)
    expected_old = distribution(old_mean, old_log_std).log_prob(native)
    expected_new = distribution(mean, log_std).log_prob(native)
    old_logprob = agent.action_logprob(old_mean, old_log_std, native)
    new_logprob = agent.action_logprob(mean, log_std, native)
    expected_ratio = (expected_new - expected_old).exp()
    torch.testing.assert_close(old_logprob, expected_old, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(new_logprob, expected_new, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close((new_logprob - old_logprob).exp(), expected_ratio, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(expected_ratio, expected_ratio.pow(1 / agent.action_dim))

    # abs/log1p is independent of the trainer's signed softplus identity and
    # remains finite when tanh(native) rounds to exactly +/-1.
    log_jacobian = 2 * (math.log(2) - native.abs() - torch.log1p(torch.exp(-2 * native.abs())))
    log_physical_scale = agent.action_scale.log()
    physical_old = agent.physical_logprob(old_mean, old_log_std, native)
    physical_new = agent.physical_logprob(mean, log_std, native)
    assert bool((native[1].tanh().abs() == 1).all())
    assert bool(torch.isfinite(physical_old).all() & torch.isfinite(physical_new).all())
    torch.testing.assert_close(physical_old, expected_old - (log_jacobian + log_physical_scale).sum(-1),
                               rtol=0, atol=1e-7)
    torch.testing.assert_close((physical_new - physical_old).exp(), expected_ratio, rtol=1e-11, atol=1e-12)


def test_noise_rescaling_preserves_whitened_geometry_but_amplifies_raw_mean_kl():
    full_noise = make_agent(noise_exponent=0.0, whiten_mean=False).double()
    low_noise_raw = make_agent(noise_exponent=1.0, whiten_mean=False).double()
    low_noise_white = make_agent(noise_exponent=1.0, whiten_mean=True).double()
    observations = tensor([[0.1, -0.7, 1.2], [-0.6, 0.9, -0.2], [1.4, -0.1, 0.5]])
    mean, log_std = full_noise.policy_parameters(observations)
    raw_mean, raw_log_std = low_noise_raw.policy_parameters(observations)
    white_mean, white_log_std = low_noise_white.policy_parameters(observations)
    scale = 1 / full_noise.action_dim
    torch.testing.assert_close(raw_mean, mean)
    torch.testing.assert_close(white_mean, mean * scale)
    torch.testing.assert_close(raw_log_std.exp(), log_std.exp() * scale)
    torch.testing.assert_close(white_log_std.exp(), log_std.exp() * scale)
    old_mean, old_log_std = torch.zeros_like(mean), torch.zeros_like(log_std)
    base_parts = geometry.gaussian_kl_parts(old_mean, old_log_std, mean, log_std)
    raw_parts = geometry.gaussian_kl_parts(old_mean, old_log_std + math.log(scale), raw_mean, raw_log_std)
    white_parts = geometry.gaussian_kl_parts(old_mean, old_log_std + math.log(scale), white_mean, white_log_std)
    torch.testing.assert_close(raw_parts[0], base_parts[0] / scale**2)
    torch.testing.assert_close(raw_parts[1], base_parts[1])
    for actual, expected in zip(white_parts, base_parts):
        torch.testing.assert_close(actual, expected)
    native = tensor([[0.2, -0.5, 0.7, -0.1], [-0.4, 0.3, -0.2, 0.6], [0.5, -0.1, 0.4, -0.7]])
    base_ratio = (geometry.gaussian_logprob(native, mean, log_std)
                  - geometry.gaussian_logprob(native, old_mean, old_log_std)).exp()
    scaled_ratio = (geometry.gaussian_logprob(native * scale, white_mean, white_log_std)
                    - geometry.gaussian_logprob(native * scale, old_mean, old_log_std + math.log(scale))).exp()
    torch.testing.assert_close(scaled_ratio, base_ratio)


def test_host_sampling_matches_device_small_state_dependent_noise_and_preserves_native_buffer():
    agent = make_agent(noise_sigma=0.03, noise_exponent=1.0, whiten_mean=True)
    observations = torch.tensor([[0.2, -0.5, 1.0], [-0.9, 0.4, -0.2], [1.1, 0.7, -0.8]], device="cuda")
    sampler = geometry.HostGaussianSampler(agent, len(observations))
    with torch.no_grad():
        logits = agent.actor(observations).cpu().numpy()
        mean, log_std = agent.policy_parameters(observations)
        assert bool((log_std.exp() < 0.02).all())
        assert not torch.allclose(mean[0], mean[1])
        assert not torch.allclose(log_std[0], log_std[1])
        draws = np.random.default_rng(812).standard_normal(mean.shape, dtype=np.float32)
        expected_native = mean + log_std.exp() * torch.from_numpy(draws).cuda()
        expected_action = expected_native.tanh() * agent.action_scale + agent.action_bias
        native, action = sampler(logits, np.random.default_rng(812))
        np.testing.assert_allclose(native, expected_native.cpu().numpy(), rtol=3e-5, atol=2e-8)
        np.testing.assert_allclose(action, expected_action.cpu().numpy(), rtol=2e-6, atol=5e-7)
        assert not np.shares_memory(native, action)
        # The retained behavior law must describe the samples, not a second
        # device evaluation whose small-sigma rounding changes the denominator.
        np.testing.assert_allclose((native - sampler.mean) / np.exp(sampler.log_std), draws,
                                   rtol=3e-5, atol=2e-6)
        host_logprob = geometry.gaussian_logprob(torch.from_numpy(native).cuda(),
                                                torch.from_numpy(sampler.mean).cuda(),
                                                torch.from_numpy(sampler.log_std).cuda())
        torch.testing.assert_close(host_logprob, distribution(mean, log_std).log_prob(expected_native),
                                   rtol=2e-5, atol=2e-5)
        saved_native, saved_action = native.copy(), action.copy()
        assert not np.allclose(saved_native, saved_action)
        assert np.all(action >= agent.action_low.cpu().numpy())
        assert np.all(action <= agent.action_high.cpu().numpy())
        # Retained arrays are reusable staging buffers, not immutable samples;
        # both must update without applying the action transform to native data.
        next_draws = np.random.default_rng(913).standard_normal(mean.shape, dtype=np.float32)
        next_expected_native = mean + log_std.exp() * torch.from_numpy(next_draws).cuda()
        next_native, next_action = sampler(logits, np.random.default_rng(913))
        assert next_native is native and next_action is action
        np.testing.assert_allclose(next_native, next_expected_native.cpu().numpy(), rtol=3e-5, atol=2e-8)
        np.testing.assert_allclose(next_action, (next_expected_native.tanh() * agent.action_scale + agent.action_bias).cpu().numpy(),
                                   rtol=2e-6, atol=5e-7)
        assert not np.allclose(native, saved_native)
        assert not np.allclose(action, saved_action)


def test_nonlinear_epoch_projection_enforces_full_rollout_forward_kl():
    agent = make_agent(noise_exponent=0.5).double()
    observations = tensor([[0.0, 0.0, 0.0], [0.01, -0.02, 0.03], [1.0, -2.0, 0.5],
                           [-3.0, 0.7, 2.0], [0.4, 1.2, -1.8], [2.0, 2.0, 2.0]])
    with torch.no_grad():
        old_mean, old_log_std = agent.policy_parameters(observations)
        before = tuple(parameter.clone() for parameter in agent.actor.parameters())
        agent.actor[0].weight.mul_(4)
        agent.actor[2].weight.mul_(3)
        agent.actor[4].weight[:agent.action_dim].mul_(6)
        agent.actor[4].bias[agent.action_dim:].sub_(1.5)

        def full_rollout_kl():
            mean, log_std = agent.policy_parameters(observations)
            mean_kl, scale_kl = geometry.gaussian_kl_parts(old_mean, old_log_std, mean, log_std)
            return (mean_kl + scale_kl).mean()

        limit = 0.02
        assert full_rollout_kl().item() > 100 * limit
        fraction = geometry.constrain_epoch(agent.actor, before, full_rollout_kl, limit)
        mean, log_std = agent.policy_parameters(observations)
        actual_kl = kl_divergence(distribution(old_mean, old_log_std), distribution(mean, log_std)).mean()
        assert 0 < fraction < 1
        assert 0 < actual_kl.item() <= limit
        # Zero-state-only checks miss the nonlinear hidden-layer displacement.
        per_state_kl = kl_divergence(distribution(old_mean, old_log_std), distribution(mean, log_std))
        assert per_state_kl.max().item() > per_state_kl.min().item()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_epoch_proposal_restores_previous_policy(invalid):
    agent = make_agent().double()
    observations = tensor([[0.2, -0.7, 1.0], [-0.4, 1.2, 0.3]])
    with torch.no_grad():
        old_mean, old_log_std = agent.policy_parameters(observations)
        before = tuple(parameter.clone() for parameter in agent.actor.parameters())
        agent.actor[0].weight.add_(0.5)
        agent.actor[4].bias[0] = invalid

        def full_rollout_kl():
            mean, log_std = agent.policy_parameters(observations)
            parts = geometry.gaussian_kl_parts(old_mean, old_log_std, mean, log_std)
            return (parts[0] + parts[1]).mean()

        assert not bool(torch.isfinite(full_rollout_kl()))
        fraction = geometry.constrain_epoch(agent.actor, before, full_rollout_kl, 0.02)
        assert fraction == 0
        for parameter, expected in zip(agent.actor.parameters(), before):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        mean, log_std = agent.policy_parameters(observations)
        torch.testing.assert_close(mean, old_mean, rtol=0, atol=0)
        torch.testing.assert_close(log_std, old_log_std, rtol=0, atol=0)
        assert full_rollout_kl().item() == 0

        # A rollback is not a successful constraint result when its reference
        # already violates the rollout behavior policy's KL budget.
        agent.actor[4].bias[0].add_(3)
        infeasible_before = tuple(parameter.clone() for parameter in agent.actor.parameters())
        assert full_rollout_kl().item() > 0.02
        agent.actor[4].bias[0] = invalid
        with pytest.raises(RuntimeError):
            geometry.constrain_epoch(agent.actor, infeasible_before, full_rollout_kl, 0.02)
        for parameter, expected in zip(agent.actor.parameters(), infeasible_before):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

"""Behavioral contracts for matched SiTU distribution controls (CUDA learner)."""

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Beta, Independent, Normal, kl_divergence

from cleanrl import ppo_continuous_action_stiglu_distribution_v2 as control


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


def make_agent(policy):
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            low=np.array([-3.0, 0.5, -0.2], dtype=np.float32),
            high=np.array([1.0, 5.5, 0.8], dtype=np.float32),
        ),
    )
    return control.Agent(envs, control.Args(policy=policy)).cuda()


def test_initial_policies_match_physical_variance_and_seeded_critic_and_trunk():
    torch.manual_seed(1)
    beta = make_agent("beta")
    torch.manual_seed(1)
    gaussian = make_agent("gaussian")
    assert control.state_hash(beta.critic) == control.state_hash(gaussian.critic)
    assert control.state_hash(beta.actor[0]) == control.state_hash(gaussian.actor[0])
    observations = torch.tensor([[0.2, -1.0, 0.7], [-0.6, 0.4, 1.3]], device="cuda")
    with torch.no_grad():
        alpha, concentration = beta.policy_parameters(observations)
        mean, log_std = gaussian.policy_parameters(observations)
        beta_distribution = Beta(alpha, concentration)
        beta_mean = beta.action_low + beta.action_span * beta_distribution.mean
        beta_variance = beta.action_span.square() * beta_distribution.variance
        torch.testing.assert_close(beta_mean, gaussian.action_bias.expand_as(mean))
        torch.testing.assert_close(mean, torch.zeros_like(mean), atol=0, rtol=0)
        # Independent, higher-order quadrature checks calibration, not the root
        # solver's own quadrature evaluation or a noisy sample-variance estimate.
        gaussian_normalized_variance = control.tanh_gaussian_variance(float(log_std[0, 0].exp()), 256)
        gaussian_variance = gaussian.action_scale.square() * gaussian_normalized_variance
        torch.testing.assert_close(beta_variance, gaussian_variance.expand_as(beta_variance), rtol=2e-6, atol=1e-7)
        assert gaussian_normalized_variance == pytest.approx(control.MATCHED_ACTION_VARIANCE, rel=2e-6)
        torch.testing.assert_close(beta.get_value(observations), gaussian.get_value(observations), rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["beta", "gaussian"])
def test_true_joint_likelihood_physical_transform_and_exact_kl(policy):
    agent = make_agent(policy).double()
    first = torch.tensor([[1.2, 2.5, 3.1], [0.8, 1.1, 2.0]], device="cuda", dtype=torch.float64)
    second = torch.tensor([[2.2, 1.3, 0.9], [1.7, 3.0, 2.4]], device="cuda", dtype=torch.float64)
    if policy == "beta":
        native = torch.tensor([[0.2, 0.6, 0.95], [0.8, 0.3, 0.1]], device="cuda", dtype=torch.float64)
        old_distribution = Independent(Beta(first, second), 1)
        new_first, new_second = first + 0.3, second * 0.7
        new_distribution = Independent(Beta(new_first, new_second), 1)
        expected_jacobian = agent.action_span.log().sum()
    else:
        first = first - 1.5
        second = second * 0.1 - 1.3
        # Large native values exercise a stable tanh Jacobian beyond where
        # calculating log(1 - tanh(z)^2) would produce infinities.
        native = torch.tensor([[0.2, -0.4, 25.0], [-21.0, 0.3, 0.1]], device="cuda", dtype=torch.float64)
        old_distribution = Independent(Normal(first, second.exp()), 1)
        new_first, new_second = first + 0.3, second + 0.2
        new_distribution = Independent(Normal(new_first, new_second.exp()), 1)
        expected_jacobian = (2 * (math.log(2) - torch.logaddexp(native, -native)) + agent.action_scale.log()).sum(-1)
    compiled_kl = torch.compile(agent.joint_kl, fullgraph=True)
    torch.testing.assert_close(compiled_kl(first, second, new_first, new_second),
                               kl_divergence(old_distribution, new_distribution), rtol=2e-10, atol=1e-10)
    old_logprob = agent.action_logprob(first, second, native)
    new_logprob = agent.action_logprob(new_first, new_second, native)
    torch.testing.assert_close(old_logprob, old_distribution.log_prob(native), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(new_logprob - old_logprob,
                               new_distribution.log_prob(native) - old_distribution.log_prob(native), rtol=1e-12, atol=1e-12)
    # Agent buffers originate in FP32, so compare the physical log scale to
    # that precision rather than treating .double() as a new calibration.
    torch.testing.assert_close(agent.physical_logprob(first, second, native),
                               old_logprob - expected_jacobian, rtol=2e-8, atol=1e-7)
    torch.testing.assert_close(agent.joint_kl(first, second, new_first, new_second),
                               kl_divergence(old_distribution, new_distribution), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("policy", ["beta", "gaussian"])
def test_host_behavior_is_retained_when_device_replay_changes(policy):
    agent = make_agent(policy)
    observations = torch.tensor([[0.2, -0.5, 1.0], [-0.9, 0.4, -0.2]], device="cuda")
    with torch.no_grad():
        agent.actor[-1].weight.normal_(std=0.4)
        logits = agent.actor(observations).cpu().numpy()
        sampler_class = control.HostBetaSampler if policy == "beta" else control.HostGaussianSampler
        sampler = sampler_class(agent, len(observations))
        native_np, action_np = sampler(logits, np.random.default_rng(812))
        native = torch.from_numpy(native_np.copy()).cuda()
        old_first = torch.from_numpy(sampler.first.copy()).cuda()
        old_second = torch.from_numpy(sampler.second.copy()).cuda()
        first, second = agent.policy_parameters(observations)
        torch.testing.assert_close(old_first, first, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(old_second, second, rtol=2e-6, atol=2e-7)
        if policy == "beta":
            expected = Independent(Beta(old_first, old_second), 1).log_prob(native)
            physical = agent.action_low + agent.action_span * native
        else:
            expected = Independent(Normal(old_first, old_second.exp()), 1).log_prob(native)
            physical = agent.action_bias + agent.action_scale * native.tanh()
        np.testing.assert_allclose(action_np, physical.cpu().numpy(), rtol=2e-6, atol=2e-7)
        agent.actor[-1].bias.add_(0.6)
        _, scored_behavior, drift = control.rollout_statistics(agent, observations, native, old_first, old_second)
        torch.testing.assert_close(scored_behavior, expected, rtol=2e-6, atol=2e-6)
        replay_first, replay_second = agent.policy_parameters(observations)
        replay_logprob = agent.action_logprob(replay_first, replay_second, native)
        assert not torch.allclose(scored_behavior, replay_logprob)
        assert drift[0].item() > 0.01
        torch.testing.assert_close(drift[1], (replay_logprob - expected).abs().max(), rtol=2e-5, atol=2e-6)


def test_gaussian_host_head_uses_sac_bounds_without_mean_or_dimension_rescaling():
    agent = make_agent("gaussian")
    logits = np.array([[0.4, -1.2, 0.7, -100.0, 0.0, 100.0]], dtype=np.float32)
    sampler = control.HostGaussianSampler(agent, 1)
    native, action = sampler(logits, np.random.default_rng(819))
    np.testing.assert_array_equal(sampler.first, logits[:, :3])
    expected_log_std = np.array([[-5.0, math.log(control.matched_gaussian_std()), 2.0]], dtype=np.float32)
    np.testing.assert_allclose(sampler.second, expected_log_std, rtol=2e-6, atol=2e-7)
    draws = np.random.default_rng(819).standard_normal((1, 3), dtype=np.float32)
    np.testing.assert_allclose(native, logits[:, :3] + np.exp(expected_log_std) * draws, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(action, agent.action_bias.cpu().numpy() + agent.action_scale.cpu().numpy() * np.tanh(native),
                               rtol=2e-6, atol=2e-7)
    assert not np.shares_memory(native, action)

"""Angular CFM, finite-candidate E-step and actual CUDA/host lifetime contracts."""
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from cleanrl import ppo_continuous_action_stiglu_conditional_fm_v6 as flow
from cleanrl import ppo_continuous_action_stiglu_distribution_v2 as control
from cleanrl.shared.rollout_transfer import RolloutTransfer

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


def environments():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            low=np.array([-3.0, 0.5, -0.2], dtype=np.float32),
            high=np.array([1.0, 5.5, 0.8], dtype=np.float32)),
    )


def make_agent(policy="flow"):
    return flow.Agent(environments(), flow.Args(policy=policy)).cuda()


def test_angular_gaussian_stationarity_and_endpoint_direction():
    # Tensor-product Gaussian quadrature integrates all quadratic moments here
    # exactly. For the joint Gaussian (z,v), zero covariance implies E[v|z]=0.
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    data = torch.tensor(np.repeat(nodes * math.sqrt(2), 12), device="cuda", dtype=torch.float64)[:, None]
    noise = torch.tensor(np.tile(nodes * math.sqrt(2), 12), device="cuda", dtype=torch.float64)[:, None]
    probability = torch.tensor(np.outer(weights, weights).reshape(-1) / math.pi, device="cuda", dtype=torch.float64)[:, None]
    for time in (0.0, 0.125, 0.5, 0.875, 1.0):
        times = torch.full_like(data, time)
        state, velocity = flow.flow_interpolant(data, noise, times)
        torch.testing.assert_close((probability * state.square()).sum(), torch.ones((), device="cuda", dtype=torch.float64))
        torch.testing.assert_close((probability * state * velocity).sum(), torch.zeros((), device="cuda", dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close((probability * velocity.square()).sum(), torch.tensor(math.pi**2 / 4, device="cuda", dtype=torch.float64))
    at_zero, velocity_zero = flow.flow_interpolant(data, noise, torch.zeros_like(data))
    at_one, velocity_one = flow.flow_interpolant(data, noise, torch.ones_like(data))
    torch.testing.assert_close(at_zero, data)
    torch.testing.assert_close(at_one, noise)
    torch.testing.assert_close(velocity_zero, math.pi / 2 * noise)
    torch.testing.assert_close(velocity_one, -math.pi / 2 * data)


def test_state_weights_preserve_mass_offsets_global_scale_and_kl_with_ties():
    scores = torch.tensor([[-5.0, -1.0, 0.0, 3.0], [20.0, 20.0, 20.0, 20.0],
                           [-3.0, -3.0, 2.0, 2.0]], device="cuda", dtype=torch.float64, requires_grad=True)
    weights, metrics = flow.candidate_weights(scores, 0.02)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, device="cuda", dtype=torch.float64))
    torch.testing.assert_close(weights[1], torch.full((4,), 0.25, device="cuda", dtype=torch.float64))
    torch.testing.assert_close(metrics["estep/conditional_kl"], torch.tensor(0.02, device="cuda", dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(weights[2, 2], weights[2, 3], rtol=0, atol=0)
    assert not weights.requires_grad
    offsets = torch.tensor([[100.0], [-200.0], [700.0]], device="cuda", dtype=torch.float64)
    for changed in (scores.detach() + offsets, scores.detach() * 1e-20, scores.detach() * 1e20):
        actual, _ = flow.candidate_weights(changed, 0.02)
        torch.testing.assert_close(actual, weights, rtol=1e-12, atol=1e-12)
    # A tiny fitted score still spends the KL budget: do not confuse this with
    # evidence the score contains learnable action signal.
    tiny, tiny_metrics = flow.candidate_weights(scores.detach() * 1e-20, 0.02)
    torch.testing.assert_close(tiny, weights)
    torch.testing.assert_close(tiny_metrics["estep/temperature"], metrics["estep/temperature"] * 1e-20)
    # Reachable KL can be below budget because all but one row have complete ties.
    tied = torch.zeros((100, 4), device="cuda", dtype=torch.float64)
    tied[0, :2] = 1
    limit_weights, limit_metrics = flow.candidate_weights(tied, 0.02)
    torch.testing.assert_close(limit_metrics["estep/achievable_kl"], torch.tensor(math.log(2) / 100, device="cuda", dtype=torch.float64))
    assert 0 < limit_metrics["estep/temperature"]
    assert limit_metrics["estep/conditional_kl"] < 0.02
    torch.testing.assert_close(limit_weights[0], torch.tensor([0.5, 0.5, 0, 0], device="cuda", dtype=torch.float64))
    flat, flat_metrics = flow.candidate_weights(torch.full((3, 16), 1e30, device="cuda"), 0.02)
    torch.testing.assert_close(flat, torch.full_like(flat, 1 / 16), atol=0, rtol=0)
    assert flat_metrics["estep/temperature"] > 0
    assert flat_metrics["estep/conditional_kl"] == 0


@pytest.mark.parametrize("policy", ["flow", "gaussian"])
def test_local_projection_detaches_candidates_weights_critic_and_interpolant(policy):
    agent = make_agent(policy)
    regressor = flow.AdvantageRegressor(3, 3).cuda()
    observations = torch.randn((4, 3), device="cuda", requires_grad=True)
    candidates = torch.randn((4, 16, 3), device="cuda", requires_grad=True)
    state_features = regressor.state_features(observations)
    scores = (regressor.coefficients(state_features)[:, None] * candidates).sum(-1)
    weights, _ = flow.candidate_weights(scores, 0.02)
    target = flow.select_targets(candidates, weights, torch.Generator(device="cuda").manual_seed(1))
    noise = torch.randn((4, 3), device="cuda", requires_grad=True)
    times = torch.rand((4, 1), device="cuda", requires_grad=True)
    loss = flow.projection_loss(agent, observations, target, noise, times)
    loss.backward()
    assert agent.actor[-1].bias.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in agent.critic.parameters())
    assert all(parameter.grad is None for parameter in regressor.parameters())
    assert candidates.grad is None and observations.grad is None
    assert noise.grad is None and times.grad is None
    assert not target.requires_grad and not weights.requires_grad


@pytest.mark.parametrize("action_order", ["linear", "quadratic"])
def test_captured_frame_uses_same_units_for_observed_candidates_and_detaches(action_order):
    regressor = flow.AdvantageRegressor(3, 3, action_order).cuda()
    mean = torch.tensor([[100.0, -20.0, 30.0]], device="cuda", requires_grad=True)
    std = torch.tensor([[0.125, 2.0, 0.5]], device="cuda", requires_grad=True)
    standardized = torch.tensor([[-1.0, 0.5, 2.0]], device="cuda")
    native = (mean + std * standardized).detach().requires_grad_()
    feature_mean = torch.zeros((1, regressor.feature_dim), device="cuda", requires_grad=True)
    observed = regressor.features(native, mean, std, feature_mean)
    candidates = regressor.features(native[:, None].expand(-1, 16, -1), mean[:, None], std[:, None], feature_mean[:, None])
    torch.testing.assert_close(observed, regressor.basis(standardized))
    torch.testing.assert_close(candidates, observed[:, None].expand(-1, 16, -1))
    assert not observed.requires_grad and not candidates.requires_grad
    coefficient_features = regressor.state_features(torch.zeros((1, 3), device="cuda"))
    # A real regression update has the direction needed to explain a positive
    # residual, and it cannot train a frame or the frozen state encoder.
    prediction = regressor.action_prediction(coefficient_features, observed)
    (prediction - 1).square().mean().backward()
    assert regressor.coefficients.bias.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in regressor.baseline.parameters())
    assert mean.grad is None and std.grad is None and native.grad is None and feature_mean.grad is None


def test_actual_host_heun_borrowed_output_and_transfer_snapshot_lifetime():
    # Use a real representable Linear velocity, not a mocked host mirror.
    actor = nn.Sequential(nn.Linear(3, 1)).cuda()
    with torch.no_grad():
        actor[0].weight.copy_(torch.tensor([[0.2, 0.5, 0.25]], device="cuda"))
        actor[0].bias.fill_(-0.1)
    agent = SimpleNamespace(policy="flow", obs_dim=1, action_dim=1, initial_std=flow.INITIAL_STD,
                            actor=actor, action_scale=torch.tensor([2.0], device="cuda"),
                            action_bias=torch.tensor([-0.5], device="cuda"))
    observations = np.array([[0.75], [-1.25]], dtype=np.float32)
    noise = np.array([[1.5], [-0.5]], dtype=np.float32)
    untouched_noise = noise.copy()
    a, b = 0.5, 0.25
    c = 0.2 * observations.astype(np.float64) - 0.1
    exact = (noise + b / a + c / a + b / a**2) * math.exp(-a) - c / a - b / a**2
    errors = []
    for steps in (4, 8, 16):
        sampler = flow.HostSampler(agent, 2, steps)
        native, action = sampler(observations, None, noise=noise)
        h = -1.0 / steps
        factor = 1 + a * h + (a * h)**2 / 2
        expected = noise.astype(np.float64)
        for step in range(steps):
            expected = factor * expected + h * (1 + a * h / 2) * (b * (1 - step / steps) + c) + h**2 * b / 2
        np.testing.assert_allclose(native, expected, rtol=2e-6, atol=3e-7)
        np.testing.assert_allclose(action, -0.5 + 2 * np.tanh(flow.INITIAL_STD * native), rtol=2e-6, atol=2e-7)
        errors.append(np.max(np.abs(native - exact)))
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.7 < coarse / fine < 4.6
    np.testing.assert_array_equal(noise, untouched_noise)
    staged_native, staged_action = native.copy(), action.copy()
    transfer = RolloutTransfer(2, 2, (1,), torch.device("cuda"),
                               fields={"observations": (1,), "native_actions": (1,)})
    try:
        transfer.push(0, np.zeros(2), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool),
                      observations=observations, native_actions=native)
        next_native, next_action = sampler(observations, None, noise=-noise)
        assert np.shares_memory(native, next_native) and np.shares_memory(action, next_action)
        assert not np.shares_memory(next_native, next_action)
        transfer.push(1, np.zeros(2), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool),
                      observations=observations, native_actions=next_native)
        batch = transfer.upload()
        np.testing.assert_array_equal(batch.fields["native_actions"][0].cpu().numpy(), staged_native)
        np.testing.assert_array_equal(batch.fields["native_actions"][1].cpu().numpy(), next_native)
        np.testing.assert_allclose(staged_action, -0.5 + 2 * np.tanh(flow.INITIAL_STD * staged_native), rtol=2e-6, atol=2e-7)
    finally:
        transfer.close()


def test_low_noise_initialization_matched_controls_and_learned_host_device_parity():
    torch.manual_seed(1)
    reference = control.Agent(environments(), control.Args(policy="gaussian")).cuda()
    torch.manual_seed(1)
    agent = make_agent("flow")
    torch.manual_seed(1)
    gaussian = make_agent("gaussian")
    observations = torch.tensor([[0.2, -0.5, 1.0], [-0.9, 0.4, -0.2]], device="cuda")
    observations_np = observations.cpu().numpy()
    noise = np.array([[1.5, -0.7, 0.2], [-0.3, 1.1, -2.0]], dtype=np.float32)
    with torch.no_grad():
        for candidate in (agent, gaussian):
            torch.testing.assert_close(candidate.get_value(observations), reference.get_value(observations), atol=0, rtol=0)
            sampler = flow.HostSampler(candidate, 2, 8)
            native, action = sampler(observations_np, None, noise=noise)
            np.testing.assert_array_equal(native, noise)
            expected = candidate.action_bias.cpu().numpy() + candidate.action_scale.cpu().numpy() * np.tanh(np.float32(flow.INITIAL_STD) * noise)
            np.testing.assert_allclose(action, expected, atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(flow.sample_native(candidate, observations, torch.from_numpy(noise).cuda(), 8), torch.from_numpy(noise).cuda(), atol=0, rtol=0)
        head = agent.actor[-1]
        head.weight.copy_(torch.linspace(-0.2, 0.3, head.weight.numel(), device="cuda").reshape_as(head.weight))
        head.bias.copy_(torch.tensor([0.1, -0.2, 0.3], device="cuda"))
        sampler = flow.HostSampler(agent, 2, 8)
        host_native, _ = sampler(observations_np, None, noise=noise)
        device_native = flow.sample_native(agent, observations, torch.from_numpy(noise).cuda(), 8)
        np.testing.assert_allclose(host_native, device_native.cpu().numpy(), rtol=3e-5, atol=4e-6)
        # Refresh after another real head change, rather than checking only init.
        head.bias.add_(0.2)
        sampler.refresh()
        host_native, _ = sampler(observations_np, None, noise=noise)
        device_native = flow.sample_native(agent, observations, torch.from_numpy(noise).cuda(), 8)
        np.testing.assert_allclose(host_native, device_native.cpu().numpy(), rtol=3e-5, atol=4e-6)


def test_separate_candidate_and_reference_streams_and_frozen_cache():
    agent = make_agent()
    regressor = flow.AdvantageRegressor(3, 3).cuda()
    args = flow.Args(minibatch_size=2, frame_samples=32)
    observations = torch.zeros((4, 3), device="cuda")
    candidate_generator = torch.Generator(device="cuda").manual_seed(11)
    frame_generator = torch.Generator(device="cuda").manual_seed(12)
    candidates, means, stds, feature_means = flow.capture_candidates(
        agent, regressor, observations, args, candidate_generator, frame_generator,
        lambda obs, noise: flow.sample_native(agent, obs, noise, args.flow_steps))
    expected_generator = torch.Generator(device="cuda").manual_seed(11)
    expected = torch.cat([torch.randn((2 * 16, 3), device="cuda", generator=expected_generator).reshape(2, 16, 3) for _ in range(2)])
    torch.testing.assert_close(candidates, expected, atol=0, rtol=0)
    original = candidates.clone()
    with torch.no_grad():
        agent.actor[-1].bias.add_(1)
    torch.testing.assert_close(candidates, original, atol=0, rtol=0)
    assert not candidates.requires_grad and not means.requires_grad and not stds.requires_grad
    # The independent frame should not exactly center the candidate bank itself.
    assert ((candidates - means[:, None]) / stds[:, None]).mean(1).abs().max() > 0.01
    torch.testing.assert_close(feature_means, torch.zeros_like(feature_means), atol=2e-7, rtol=0)


def test_action_calibration_distinguishes_signal_from_shuffled_residuals():
    residual = torch.linspace(-2, 2, 128, device="cuda")
    prediction = residual * 0.5
    actual = flow.action_calibration(prediction, residual)
    torch.testing.assert_close(actual["critic/heldout_action_correlation"], torch.ones((), device="cuda"))
    torch.testing.assert_close(actual["critic/heldout_action_calibration_slope"], torch.tensor(2.0, device="cuda"))
    zero = flow.action_calibration(torch.zeros_like(residual), residual)
    assert zero["critic/heldout_calibration_valid"] == 0
    # Per-env, within-window permutations preserve the actual control samples.
    samples = torch.arange(64 * 2, device="cuda", dtype=torch.float32).reshape(128, 1)
    shuffled = flow.shuffled_action_residuals(samples, 64, 2, torch.Generator(device="cuda").manual_seed(73))
    original_windows = samples.reshape(2, 32, 2, 1)
    shuffled_windows = shuffled.reshape(2, 32, 2, 1)
    torch.testing.assert_close(shuffled_windows.sort(dim=1).values, original_windows)
    assert not torch.equal(samples, shuffled)

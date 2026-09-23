"""Math and host/device contracts for advantage-weighted flow matching (CUDA)."""

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from cleanrl import ppo_continuous_action_stiglu_distribution_v2 as control
from cleanrl import ppo_continuous_action_stiglu_flow_matching_v3 as flow


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
            high=np.array([1.0, 5.5, 0.8], dtype=np.float32),
        ),
    )


def make_agent(policy="flow"):
    return flow.Agent(environments(), flow.Args(policy=policy)).cuda()


def test_paper_interpolant_has_data_at_zero_noise_at_one_and_no_target_graph():
    targets = torch.tensor([[2.0, -3.0], [4.0, 1.0], [-2.0, 3.0]], device="cuda", requires_grad=True)
    noise = torch.tensor([[-1.0, 2.0], [0.0, -1.0], [4.0, -3.0]], device="cuda", requires_grad=True)
    times = torch.tensor([[0.0], [0.25], [1.0]], device="cuda", requires_grad=True)
    states, velocity = flow.flow_interpolant(targets, noise, times)
    torch.testing.assert_close(states, torch.tensor([[2.0, -3.0], [3.0, 0.5], [4.0, -3.0]], device="cuda"))
    torch.testing.assert_close(velocity, noise.detach() - targets.detach())
    assert not states.requires_grad
    assert not velocity.requires_grad


def test_rollout_weights_are_stable_affine_invariant_and_preserve_partitioned_objective():
    advantages = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device="cuda", requires_grad=True)
    weights, ess = flow.advantage_weights(advantages, 0.7)
    reference = torch.softmax(advantages.detach().double() / advantages.detach().double().std(unbiased=False) / 0.7, 0) * 5
    torch.testing.assert_close(weights.double(), reference, rtol=2e-6, atol=1e-7)
    assert (weights > 0).all()
    assert not weights.requires_grad and not ess.requires_grad
    torch.testing.assert_close(ess, weights.sum().square() / (5 * weights.square().sum()))
    for transformed in (advantages.detach() * 8 + 100, advantages.detach() * 1e30):
        actual, actual_ess = flow.advantage_weights(transformed, 0.7)
        torch.testing.assert_close(actual, weights, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(actual_ess, ess, rtol=2e-6, atol=1e-7)
    equal, equal_ess = flow.advantage_weights(torch.full((5,), 1e30, device="cuda"), 0.7)
    torch.testing.assert_close(equal, torch.ones_like(equal), rtol=0, atol=0)
    torch.testing.assert_close(equal_ess, torch.ones_like(equal_ess), rtol=0, atol=0)
    # A diagnostic subset can exclude every globally non-underflowed weight.
    # Renormalize its original logits, not the already rounded global weights.
    separated = torch.tensor([-3.0, -2.0, -1.0, 100.0], device="cuda")
    global_weights, _ = flow.advantage_weights(separated, 0.001)
    assert global_weights[:3].sum() == 0
    local_weights = torch.softmax(flow.advantage_logits(separated, 0.001)[:3], 0)
    errors = torch.tensor([7.0, 5.0, 3.0], device="cuda")
    torch.testing.assert_close((local_weights * errors).sum(), errors[-1], rtol=1e-6, atol=1e-6)

    agent = make_agent("gaussian_awr")
    args = flow.Args(policy="gaussian_awr", clip_vloss=False)
    observations = torch.arange(15, device="cuda", dtype=torch.float32).reshape(5, 3) / 10
    native = torch.tensor([[-2.0, 0.1, 0.2], [0.3, -0.1, 0.5], [1.0, 0.2, -0.4],
                           [0.5, 0.1, 0.2], [2.0, -1.0, 0.4]], device="cuda")
    returns = torch.linspace(-1, 1, 5, device="cuda")
    old_values = torch.zeros_like(returns)
    full, full_metrics = flow.policy_loss(agent, observations, native, weights, returns, old_values, args)
    pieces, piece_metrics = [], []
    for indices in (torch.tensor([0, 3], device="cuda"), torch.tensor([1, 2, 4], device="cuda")):
        loss, metrics = flow.policy_loss(agent, observations[indices], native[indices], weights[indices],
                                         returns[indices], old_values[indices], args)
        pieces.append(loss * (len(indices) / 5))
        piece_metrics.append(metrics * (len(indices) / 5))
    torch.testing.assert_close(sum(pieces), full, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(sum(piece_metrics), full_metrics, rtol=2e-6, atol=1e-6)


def test_initial_sampling_matches_v2_and_refreshed_learned_velocity_matches_device():
    torch.manual_seed(1)
    gaussian = control.Agent(environments(), control.Args(policy="gaussian")).cuda()
    torch.manual_seed(1)
    agent = make_agent()
    torch.manual_seed(1)
    awr = make_agent("gaussian_awr")
    observations = torch.tensor([[0.2, -0.5, 1.0], [-0.9, 0.4, -0.2]], device="cuda")
    observations_np = observations.cpu().numpy()
    with torch.no_grad():
        for candidate in (agent, awr):
            for actual, expected in zip(candidate.critic.parameters(), gaussian.critic.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(candidate.get_value(observations), gaussian.get_value(observations), rtol=0, atol=0)
        for actual, expected in zip(awr.actor.parameters(), gaussian.actor.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        sampler = flow.HostSampler(agent, 2, 8)
        awr_sampler = flow.HostSampler(awr, 2, 8)
        gaussian_sampler = control.HostGaussianSampler(gaussian, 2)
        native, action = sampler(observations_np, np.random.default_rng(812))
        awr_native, awr_action = awr_sampler(observations_np, np.random.default_rng(812))
        gaussian_native, gaussian_action = gaussian_sampler(gaussian.actor(observations).cpu().numpy(), np.random.default_rng(812))
        draws = np.random.default_rng(812).standard_normal((2, 3), dtype=np.float32)
        np.testing.assert_array_equal(native, draws)
        np.testing.assert_allclose(native * flow.MATCHED_STD, gaussian_native, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(action, gaussian_action, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(awr_native, native, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(awr_action, action, rtol=2e-6, atol=2e-7)
        # Change the actual trained head after the mirror was created. Reusing
        # a zero/stale mirror would pass initial-policy parity but fail here.
        head = agent.actor[-1]
        head.weight.copy_(torch.linspace(-0.2, 0.3, head.weight.numel(), device="cuda").reshape_as(head.weight))
        head.bias.copy_(torch.tensor([0.1, -0.2, 0.3], device="cuda"))
        sampler.refresh()
        states = torch.tensor([[0.4, -1.2, 0.7], [-0.3, 0.6, 1.1]], device="cuda")
        times = torch.tensor([[0.0], [0.75]], device="cuda")
        inputs = torch.cat((observations, states, times), -1).cpu().numpy()
        np.testing.assert_allclose(sampler.actor(inputs), agent.velocity(observations, states, times).cpu().numpy(),
                                   rtol=2e-5, atol=3e-6)


def test_real_heun_sampler_integrates_backward_with_second_order_accuracy_and_borrowed_buffers():
    # x' = a*x + b*t + c(obs), with a representable CUDA Linear actor.
    # The time term makes using the first-stage time at both stages incorrect.
    actor = nn.Sequential(nn.Linear(3, 1)).cuda()
    with torch.no_grad():
        actor[0].weight.copy_(torch.tensor([[0.2, 0.5, 0.25]], device="cuda"))
        actor[0].bias.fill_(-0.1)
    agent = SimpleNamespace(policy="flow", obs_dim=1, action_dim=1, std_bias=0.0, actor=actor,
                            action_scale=torch.tensor([2.0], device="cuda"),
                            action_bias=torch.tensor([-0.5], device="cuda"))
    observations = np.array([[0.75], [-1.25]], dtype=np.float32)
    noise = np.array([[1.5], [-0.5]], dtype=np.float32)
    original_noise = noise.copy()
    a, b = 0.5, 0.25
    c = 0.2 * observations.astype(np.float64) - 0.1
    # Exact solution with boundary x(1)=noise, evaluated at t=0.
    exact = (noise + b / a + c / a + b / a**2) * math.exp(-a) - c / a - b / a**2
    errors = []
    for steps in (4, 8, 16):
        sampler = flow.HostSampler(agent, 2, steps)
        native, action = sampler(observations, None, noise=noise)
        # Independent closed-form Heun recurrence for this affine ODE checks
        # sign, both endpoint times, and a complete interval, not just order.
        h = -1.0 / steps
        factor = 1 + a * h + 0.5 * (a * h)**2
        expected = noise.astype(np.float64)
        for step in range(steps):
            expected = factor * expected + h * (1 + a * h / 2) * (b * (1 - step / steps) + c) + h**2 * b / 2
        np.testing.assert_allclose(native, expected, rtol=2e-6, atol=3e-7)
        errors.append(float(np.max(np.abs(native - exact))))
        np.testing.assert_allclose(action, -0.5 + 2 * np.tanh(flow.MATCHED_STD * native), rtol=2e-6, atol=2e-7)
        assert not np.shares_memory(native, action)
        assert not np.shares_memory(native, noise)
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.7 < coarse / fine < 4.6
    np.testing.assert_array_equal(noise, original_noise)
    staged_native, staged_action = native.copy(), action.copy()
    next_native, next_action = sampler(observations, None, noise=-noise)
    assert np.shares_memory(native, next_native)
    assert np.shares_memory(action, next_action)
    # A rollout that stages a copy retains the old sample across buffer reuse.
    np.testing.assert_allclose(staged_native, expected, rtol=2e-6, atol=3e-7)
    np.testing.assert_allclose(staged_action, -0.5 + 2 * np.tanh(flow.MATCHED_STD * staged_native), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(next_native - staged_native, -2 * noise * factor**steps, rtol=2e-6, atol=5e-7)


def test_flow_gradient_moves_samples_toward_positive_weight_targets_without_target_gradients():
    agent = make_agent()
    args = flow.Args(vf_coef=0.0, clip_vloss=False)
    observations = torch.zeros((2, 3), device="cuda")
    rng_state = torch.cuda.get_rng_state()
    paired_noise = torch.randn_like(observations)
    displacement = torch.tensor([[-3.0, -1.0, -2.0], [1.0, 2.0, 3.0]], device="cuda")
    native = (paired_noise + displacement).requires_grad_()
    weights = torch.tensor([0.1, 1.9], device="cuda", requires_grad=True)
    torch.cuda.set_rng_state(rng_state)
    loss, _ = flow.policy_loss(agent, observations, native, weights, torch.zeros(2, device="cuda"),
                               torch.zeros(2, device="cuda"), args)
    loss.backward()
    expected_gradient = 2 * (weights.detach()[:, None] * displacement).mean(0)
    torch.testing.assert_close(agent.actor[-1].bias.grad, expected_gradient, rtol=2e-6, atol=1e-6)
    assert native.grad is None
    assert weights.grad is None
    # Update only the constant velocity: backward sampling must move in the
    # opposite direction to velocity, toward the positively weighted targets.
    with torch.no_grad():
        agent.actor[-1].bias.add_(agent.actor[-1].bias.grad, alpha=-0.05)
    sampler = flow.HostSampler(agent, 2, 8)
    draws = paired_noise.cpu().numpy()
    sampled, _ = sampler(observations.cpu().numpy(), None, noise=draws)
    np.testing.assert_allclose(sampled - draws, np.broadcast_to(0.05 * expected_gradient.cpu().numpy(), draws.shape),
                               rtol=3e-6, atol=6e-7)


def test_gaussian_awr_density_gradients_match_v2_and_critic_is_unweighted():
    agent = make_agent("gaussian_awr")
    reference = control.Agent(environments(), control.Args(policy="gaussian")).cuda()
    with torch.no_grad():
        head = agent.actor[-1]
        head.weight.copy_(torch.linspace(-0.03, 0.04, head.weight.numel(), device="cuda").reshape_as(head.weight))
        head.bias.copy_(torch.tensor([0.3, -0.4, 0.2, 0.05, -0.1, 0.15], device="cuda"))
    reference.actor.load_state_dict(agent.actor.state_dict())
    reference.critic.load_state_dict(agent.critic.state_dict())
    args = flow.Args(policy="gaussian_awr", clip_vloss=False, vf_coef=0.7)
    observations = torch.tensor([[0.2, -0.5, 1.0], [-0.9, 0.4, -0.2]], device="cuda")
    native = torch.tensor([[1.5, -0.7, 0.2], [-0.3, 1.1, -2.0]], device="cuda", requires_grad=True)
    weights = torch.tensor([0.2, 1.8], device="cuda", requires_grad=True)
    returns = torch.tensor([1.2, -0.8], device="cuda")
    loss, metrics = flow.policy_loss(agent, observations, native, weights, returns, torch.zeros_like(returns), args)
    mean, log_std = reference.policy_parameters(observations)
    expected_actor = -(weights.detach() * reference.action_logprob(mean, log_std, native.detach() * flow.MATCHED_STD)).mean()
    expected_value = 0.5 * (reference.get_value(observations).flatten() - returns).square().mean()
    expected_loss = expected_actor + args.vf_coef * expected_value
    torch.testing.assert_close(metrics[:2], torch.stack((expected_actor, expected_value)), rtol=2e-6, atol=1e-6)
    loss.backward()
    expected_loss.backward()
    for actual, expected in zip(agent.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-6)
    assert native.grad is None
    assert weights.grad is None

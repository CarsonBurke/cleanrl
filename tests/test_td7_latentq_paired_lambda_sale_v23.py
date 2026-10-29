"""Pure tensor/probability and metadata contracts; no models or simulations."""
import copy
import importlib.util
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl/td7/latentq/td7_latentq_paired_lambda_sale_v23.py"
SPEC = importlib.util.spec_from_file_location("paired_lambda_sale_v23", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_paired_squared_loss_decomposes_into_mean_and_quarter_contrast():
    center = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    odd = torch.tensor([[0.5, -0.25], [2.0, -1.0]])
    even = torch.tensor([[-0.5, 0.25], [1.0, 0.5]])
    plus, minus = torch.tensor([[4.0], [9.0]]), torch.tensor([[1.0], [3.0]])
    loss = MODULE.paired_hermite_mse(center + odd + even, center - odd + even, plus, minus)
    mean_loss = (center + even - 0.5 * (plus + minus)).square().sum(-1).mean()
    contrast_loss = (2 * odd - (plus - minus)).square().sum(-1).mean()
    torch.testing.assert_close(loss, mean_loss + 0.25 * contrast_loss)


def test_uniform_direction_projection_recovers_clipped_gaussian_mean_derivative():
    # Numerical probability integration, not a neural model. For G=clip(U),
    # d_mu E[G] is the probability that the raw Gaussian is inside its bounds.
    eps = torch.linspace(-9.0, 9.0, 40001, dtype=torch.float64)
    density = torch.exp(-0.5 * eps.square()) / np.sqrt(2 * np.pi)
    mu, sigma = 0.8, 0.4
    plus, minus = (mu + sigma * eps).clamp(-1, 1), (mu - sigma * eps).clamp(-1, 1)
    contrast = plus - minus
    phi = lambda x: 0.5 * (1 + torch.erf(torch.as_tensor(x, dtype=torch.float64) / np.sqrt(2)))
    expected_derivative = phi((1 - mu) / sigma) - phi((-1 - mu) / sigma)
    moment = torch.trapezoid(eps * contrast * density, eps) / (2 * sigma)
    torch.testing.assert_close(moment, expected_derivative, atol=2e-8, rtol=2e-8)
    g = expected_derivative.detach().requires_grad_()
    objective = torch.trapezoid((2 * sigma * eps * g - contrast).square() * density, eps)
    gradient = torch.autograd.grad(objective, g)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-8, rtol=0)
    # Actual clipped displacements are not valid substitutes for raw sigma*eps.
    assert not torch.allclose(plus - minus, 2 * sigma * eps)


def test_lambda_stops_at_truncation_but_bootstraps_final_observation():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[100.0]], [[4.0]]])
    next_values = torch.tensor([[[10.0]], [[20.0]], [[30.0]], [[40.0]]])
    mask = torch.ones_like(rewards)
    boundary = torch.tensor([[[0.0]], [[1.0]], [[0.0]], [[0.0]]])
    target = MODULE.lambda_returns(rewards, mask, boundary, next_values, 0.9, 0.5, 4)
    torch.testing.assert_close(target[:2], torch.tensor([[[14.5]], [[20.0]]]))
    rewards[2:] += 1000
    changed = MODULE.lambda_returns(rewards, mask, boundary, next_values, 0.9, 0.5, 4)
    torch.testing.assert_close(changed[:2], target[:2])
    mask[1] = 0
    terminal = MODULE.lambda_returns(rewards, mask, boundary, next_values, 0.9, 0.5, 4)
    torch.testing.assert_close(terminal[1], torch.tensor([[2.0]]))


def test_trace_segments_do_not_recurse_into_the_next_segment():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[100.0]], [[200.0]]])
    values = torch.full_like(rewards, 10)
    target = MODULE.lambda_returns(rewards, torch.ones_like(rewards), torch.zeros_like(rewards), values, 0.9, 1.0, 2)
    torch.testing.assert_close(target[:2], torch.tensor([[[10.9]], [[11.0]]]))


def test_snapshot_metadata_copies_native_age_and_rng_without_aliasing():
    def fake(count, seed):
        return SimpleNamespace(
            _episode_lengths=np.arange(count, dtype=np.int32) + 9,
            _episode_returns=np.arange(count, dtype=np.float32) + 13,
            _episode_starts=np.arange(count, dtype=np.float32) + 17,
            _limits=[SimpleNamespace(_elapsed_steps=9 + i) for i in range(count)],
            _bases=[SimpleNamespace(np_random=np.random.default_rng(seed + i)) for i in range(count)],
            _statistics=[SimpleNamespace(return_queue=deque([3.0]), length_queue=deque([7]), episode_count=2) for _ in range(count)],
            observations=np.arange(count * 3, dtype=np.float64).reshape(count, 3),
        )
    source, destination = fake(1, 3), fake(2, 90)
    initial_rng = copy.deepcopy(source._bases[0].np_random.bit_generator.state)
    MODULE.copy_episode_metadata(source, destination, 0, 1, source.observations[0])
    assert destination._episode_lengths[1] == source._episode_lengths[0]
    assert destination._limits[1]._elapsed_steps == source._limits[0]._elapsed_steps
    assert destination._episode_returns[1] == source._episode_returns[0]
    np.testing.assert_array_equal(destination.observations[1], source.observations[0])
    assert destination._bases[1].np_random.bit_generator.state == initial_rng
    destination._bases[1].np_random.normal()
    assert source._bases[0].np_random.bit_generator.state == initial_rng
    destination._statistics[1].return_queue.append(100)
    assert list(source._statistics[0].return_queue) == [3.0]
    destination.observations[1] = -999
    assert source.observations[0, 0] == 0


def test_budget_counts_every_shadow_step_including_masked_post_boundary_rows():
    anchors = MODULE.segment_anchor_steps(1024, 32)
    np.testing.assert_array_equal(anchors, np.arange(8) * 128)
    cost = MODULE.rollout_physics_budget(1024, 16, 32)
    assert cost == 16384 + 4096
    warmup_transitions = 10000
    iterations = (8000000 - warmup_transitions) // cost
    assert warmup_transitions + iterations * cost <= 8000000
    assert warmup_transitions + (iterations + 1) * cost > 8000000


def test_odd_even_projection_identifies_clipped_mean_and_scale_moments():
    eps = torch.linspace(-9.0, 9.0, 40001, dtype=torch.float64)
    density = torch.exp(-0.5 * eps.square()) / np.sqrt(2 * np.pi)
    mu, sigma = 0.8, 0.4
    alpha, beta = (-1 - mu) / sigma, (1 - mu) / sigma
    cdf = lambda x: 0.5 * (1 + torch.erf(torch.as_tensor(x, dtype=torch.float64) / np.sqrt(2)))
    pdf = lambda x: torch.exp(-0.5 * torch.as_tensor(x, dtype=torch.float64).square()) / np.sqrt(2 * np.pi)
    probability = cdf(beta) - cdf(alpha)
    true_center = mu * probability + sigma * (pdf(alpha) - pdf(beta)) + 1 - cdf(beta) - cdf(alpha)
    center = true_center.detach().requires_grad_()
    mean_derivative = probability.detach().requires_grad_()
    scale_derivative = (pdf(alpha) - pdf(beta)).detach().requires_grad_()
    odd = sigma * eps * mean_derivative
    even = 0.5 * sigma * (eps.square() - 1) * scale_derivative
    plus, minus = (mu + sigma * eps).clamp(-1, 1), (mu - sigma * eps).clamp(-1, 1)
    objective = torch.trapezoid(0.5 * ((center + odd + even - plus).square() +
                                    (center - odd + even - minus).square()) * density, eps)
    grads = torch.autograd.grad(objective, (center, mean_derivative, scale_derivative))
    for gradient in grads:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-8, rtol=0)


def test_zero_norm_statistics_are_undefined_not_measured_orthogonality():
    ratios = MODULE.defined_stat_ratio(torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 2.0]))
    assert torch.isnan(ratios[:2]).all()
    torch.testing.assert_close(ratios[2], torch.tensor(0.0))


def test_snapshot_observation_uses_rollout_state_when_native_cache_is_stale():
    source = SimpleNamespace(
        _episode_lengths=np.array([0], dtype=np.int32),
        _episode_returns=np.zeros(1, dtype=np.float32),
        _episode_starts=np.zeros(1, dtype=np.float32),
        _limits=[SimpleNamespace(_elapsed_steps=0)],
        _bases=[SimpleNamespace(np_random=np.random.default_rng(1))],
        _statistics=[SimpleNamespace()],
        observations=np.array([[100.0, 200.0, 300.0]]),
    )
    destination = SimpleNamespace(
        _episode_lengths=np.array([999], dtype=np.int32),
        _episode_returns=np.zeros(1, dtype=np.float32),
        _episode_starts=np.zeros(1, dtype=np.float32),
        _limits=[SimpleNamespace(_elapsed_steps=999)],
        _bases=[SimpleNamespace(np_random=np.random.default_rng(2))],
        _statistics=[SimpleNamespace()],
        observations=np.zeros((1, 3)),
    )
    rollout_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    MODULE.copy_episode_metadata(source, destination, 0, 0, rollout_obs)
    np.testing.assert_array_equal(destination.observations[0], rollout_obs)
    np.testing.assert_array_equal(source.observations[0], [100.0, 200.0, 300.0])
    assert destination._episode_lengths[0] == destination._limits[0]._elapsed_steps == 0
    rollout_obs[:] = -50
    np.testing.assert_array_equal(destination.observations[0], [1.0, 2.0, 3.0])

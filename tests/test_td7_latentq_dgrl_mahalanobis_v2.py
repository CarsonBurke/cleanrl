"""DGRL Mahalanobis v2: DBU softmax-average targets, the matching variance, and SDN rank selection."""
import math

import pytest
import torch

from cleanrl.td7.dgrl import td7_latentq_dgrl_mahalanobis_v2 as trainer


def cloud(samples, batch, action_dim, *, sigma=0.1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.rand(batch, action_dim, generator=generator, dtype=torch.float64).mul(0.6).sub(0.3)
    eps = torch.randn(samples // 2, batch, action_dim, generator=generator, dtype=torch.float64)
    eps = torch.cat([eps, -eps])
    return mu, torch.full_like(mu, sigma), mu + sigma * eps


def score(actions, gradient, curvature=0.0, center=None):
    offset = actions - (actions.new_zeros(()) if center is None else center)
    return (offset * gradient).sum(-1) - 0.5 * curvature * offset.square().sum(-1)


def test_flat_critic_returns_the_mode_and_the_policy_variance():
    mu, sigma, draws = cloud(64, 4096, 6)
    candidates = torch.cat([draws, mu[None]])
    target, variance, diagnostics = trainer.dbu_targets(
        candidates, torch.zeros(len(candidates), mu.shape[0], dtype=mu.dtype), len(draws), 1.0, True)
    torch.testing.assert_close(target, mu)
    assert variance.mean().item() == pytest.approx(sigma[0, 0].item() ** 2, rel=0.02)
    assert diagnostics[1].item() == pytest.approx(1.0)


def test_linear_critic_steps_along_the_gradient_without_changing_variance():
    mu, sigma, draws = cloud(32, 8192, 6)
    gradient = torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0, -1.0], dtype=mu.dtype)
    target, variance, _ = trainer.dbu_targets(draws, score(draws, gradient), len(draws), 1.0, False)
    step = (target - mu).mean(0)
    cosine = (step @ gradient) / (step.norm() * gradient.norm())
    assert cosine.item() > 0.99
    assert (variance / sigma.square()).mean().item() == pytest.approx(1.0, abs=0.04)


def test_concave_peak_at_the_mode_narrows_and_convex_widens():
    mu, sigma, draws = cloud(32, 4096, 4)
    gradient = torch.zeros(4, dtype=mu.dtype)
    ratios = []
    for curvature in (60.0, -60.0):
        q = score(draws, gradient, curvature, center=mu)
        _, variance, _ = trainer.dbu_targets(draws, q, len(draws), 1.0, False)
        ratios.append((variance / sigma.square()).mean().item())
    assert ratios[0] < 0.9 < 1.1 < ratios[1]


def test_target_stays_in_the_box_tanh_reaches():
    mu, sigma, draws = cloud(16, 512, 3, sigma=0.8)
    q = score(draws.clamp(-1, 1), torch.ones(3, dtype=mu.dtype))
    target, _, _ = trainer.dbu_targets(draws, q, len(draws), 0.1, False)
    assert target.abs().max().item() <= trainer.TARGET_BOUND


def test_clipping_does_not_pull_a_saturated_mode_inward():
    mu, sigma, draws = cloud(32, 2048, 1, sigma=0.3)
    mu = torch.full_like(mu, 0.95)
    draws = mu + (draws - draws.mean(0))
    candidates = torch.cat([draws, mu[None]])
    flat = torch.zeros(len(candidates), mu.shape[0], dtype=mu.dtype)
    target, _, _ = trainer.dbu_targets(candidates, flat, len(draws), 1.0, True)
    torch.testing.assert_close(target, mu)
    rising = score(candidates.clamp(-1, 1), torch.ones(1, dtype=mu.dtype))
    target, _, _ = trainer.dbu_targets(candidates, rising, len(draws), 1.0, True)
    assert (target > mu).all()


def test_one_dominant_state_cannot_underflow_the_draw_weights():
    mu, sigma, draws = cloud(32, 256, 2)
    candidates = torch.cat([draws, mu[None]])
    q = torch.zeros(len(candidates), mu.shape[0], dtype=mu.dtype)
    q[-1, 0] = 1e6  # the mode wins one state by far more than every other deviation
    _, variance, _ = trainer.dbu_targets(candidates, q, len(draws), 0.5, True)
    assert torch.isfinite(variance).all()


def test_losses_regress_the_mean_and_fit_variance_without_jensen_bias():
    mu = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[math.log(0.1), math.log(0.3)]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.25, -0.2]], dtype=torch.float64)
    variance = torch.tensor([[0.02, 0.09]], dtype=torch.float64)
    location, scale = trainer.dbu_losses(mu, log_std, target, variance)
    d_mu, = torch.autograd.grad(location, mu)
    d_log_std, = torch.autograd.grad(scale, log_std)
    torch.testing.assert_close(d_mu, (mu.detach() - target) / (2 * log_std.detach()).exp())
    torch.testing.assert_close(d_log_std, 0.5 * (1 - variance / (2 * log_std.detach()).exp()))


def test_rank_selection_is_a_decreasing_distribution():
    probability = trainer.rank_selection_probabilities(16, 0.8)
    assert probability.sum().item() == pytest.approx(1.0)
    assert (probability[1:] < probability[:-1]).all()
    assert probability[1].item() / probability[0].item() == pytest.approx(0.8)

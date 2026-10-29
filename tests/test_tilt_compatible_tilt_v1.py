"""Compatible-tilt contracts; run CUDA work exclusively through mlq."""
import torch
from torch.distributions import Beta, kl_divergence

from cleanrl.tilt.ppo_continuous_action_sepclip_compatible_tilt_v1 import (
    MIN_CONCENTRATION, beta_cross_entropy, beta_fisher_cholesky, beta_mean_statistics,
    natural_step, tilt_target, whitened_score,
)


def concentrations(batch=256, dims=6, seed=1):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    draw = torch.rand((2, batch, dims), generator=generator, device="cuda", dtype=torch.float64)
    return (1.0 + 30.0 * draw ** 3).unbind(0)


def test_whitened_score_is_centered_with_identity_covariance():
    alpha, beta = (x[:4] for x in concentrations())
    torch.manual_seed(1)
    actions = Beta(alpha, beta).sample((400_000,))
    first, second = whitened_score(alpha, beta, actions).chunk(2, dim=-1)
    assert max(first.mean(0).abs().max(), second.mean(0).abs().max()) < 0.02
    for moment, expected in (((first * first).mean(0), 1.0), ((second * second).mean(0), 1.0),
                             ((first * second).mean(0), 0.0)):
        assert (moment - expected).abs().max() < 0.03


def test_cholesky_reconstructs_the_autograd_fisher():
    alpha, beta = concentrations()
    alpha.requires_grad_(True), beta.requires_grad_(True)
    log_normalizer = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    grad_alpha, grad_beta = torch.autograd.grad(log_normalizer.sum(), (alpha, beta), create_graph=True)
    f_aa, f_ab = torch.autograd.grad(grad_alpha.sum(), (alpha, beta), retain_graph=True)
    f_bb = torch.autograd.grad(grad_beta.sum(), beta)[0]
    l11, l21, l22 = beta_fisher_cholesky(alpha.detach(), beta.detach())
    torch.testing.assert_close(l11 * l11, f_aa)
    torch.testing.assert_close(l11 * l21, f_ab)
    torch.testing.assert_close(l21 * l21 + l22 * l22, f_bb)


def test_natural_step_tilts_the_log_density_by_slope_dot_score():
    alpha, beta = concentrations()
    slope = torch.randn(alpha.shape[0], 2 * alpha.shape[1], device="cuda", dtype=torch.float64)
    actions = Beta(alpha, beta).sample()
    step_alpha, step_beta = natural_step(alpha, beta, slope)
    mean_log, mean_log1m = beta_mean_statistics(alpha, beta).chunk(2, dim=-1)
    tilt = step_alpha * (actions.log() - mean_log) + step_beta * (torch.log1p(-actions) - mean_log1m)
    torch.testing.assert_close(tilt, sum((slope * whitened_score(alpha, beta, actions)).chunk(2, dim=-1)))


def test_least_squares_slope_is_the_whitened_policy_gradient():
    alpha, beta = (x[:1, :1] for x in concentrations())
    torch.manual_seed(1)
    actions = Beta(alpha, beta).sample((400_000,)).reshape(-1, 1)
    score = whitened_score(alpha, beta, actions)
    advantage = torch.sin(6.0 * actions[:, 0]) + 0.5 * torch.randn_like(actions[:, 0])
    fitted = torch.linalg.lstsq(score, advantage[:, None]).solution[:, 0]
    torch.testing.assert_close(fitted, (advantage[:, None] * score).mean(0), atol=0.01, rtol=0.0)


def test_tilt_target_meets_the_kl_budget_and_reports_exact_statistics():
    alpha, beta = concentrations()
    slope = 0.2 * torch.randn(alpha.shape[0], 2 * alpha.shape[1], device="cuda", dtype=torch.float64)
    budget = torch.tensor(0.02, dtype=torch.float64, device="cuda")
    statistics, entropy, (step, kl, clamped, slope_square) = tilt_target(alpha, beta, slope, budget)
    step_alpha, step_beta = natural_step(alpha, beta, slope)
    target = Beta((alpha + step * step_alpha).clamp_min(MIN_CONCENTRATION),
                  (beta + step * step_beta).clamp_min(MIN_CONCENTRATION))
    torch.testing.assert_close(kl, budget, rtol=1e-6, atol=0.0)
    torch.testing.assert_close(kl_divergence(target, Beta(alpha, beta)).sum(-1).mean(), kl)
    torch.testing.assert_close(entropy, target.entropy().sum(-1))
    torch.testing.assert_close(statistics, beta_mean_statistics(target.concentration1, target.concentration0))
    torch.testing.assert_close(slope_square, slope.square().sum(-1).mean())
    assert 0.0 < clamped < 1.0
    # Second-order check: with no clamp in play the step is sqrt(2 kl / |u|^2).
    small = tilt_target(alpha + 5.0, beta + 5.0, 1e-3 * slope, 1e-8 * budget)[2]
    torch.testing.assert_close(small[0], (2e-8 * budget / small[3]).sqrt(), rtol=1e-2, atol=0.0)
    assert small[2] == 0.0


def test_cross_entropy_gradient_is_the_mean_statistic_mismatch():
    alpha, beta = concentrations()
    target_alpha, target_beta = concentrations(seed=2)
    statistics = beta_mean_statistics(target_alpha, target_beta)
    alpha.requires_grad_(True), beta.requires_grad_(True)
    cross_entropy = beta_cross_entropy(alpha, beta, statistics)
    entropy = beta_cross_entropy(target_alpha, target_beta, statistics)
    torch.testing.assert_close(cross_entropy - entropy,
                               kl_divergence(Beta(target_alpha, target_beta), Beta(alpha, beta)).sum(-1))
    gradient = torch.cat(torch.autograd.grad(cross_entropy.sum(), (alpha, beta)), dim=-1)
    torch.testing.assert_close(gradient, beta_mean_statistics(alpha, beta).detach() - statistics)

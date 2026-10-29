"""Pure Gaussian and score-gradient checks; no networks, environments or training."""

import itertools
import importlib
import math

import pytest
import torch

from cleanrl.td7.delight.td7_latentq_delight_cubature_v1 import (
    delight_score_loss,
    gaussian_cubature,
    gaussian_log_probability,
)


DTYPE = torch.float64


@pytest.fixture(autouse=True, params=["td7_latentq_delight_cubature_v1", "td7_latentq_delight_nofloor_v2", "td7_latentq_delight_fast_preaction_v3", "td7_latentq_value_baseline_v4", "td7_latentq_value_floor_v5", "td7_latentq_delight_hermite_v6", "td7_latentq_delight_hermite_ablation_v7", "td7_latentq_delight_hermite_integrated_v8", "td7_latentq_delight_cubature_raw_v9"])
def score_implementation(request, monkeypatch):
    module = importlib.import_module(f"cleanrl.td7.delight.{request.param}")
    for name in ("delight_score_loss", "gaussian_cubature", "gaussian_log_probability"):
        monkeypatch.setitem(globals(), name, getattr(module, name))


@pytest.mark.parametrize("dimension", [1, 3, 6])
def test_cubature_integrates_all_gaussian_moments_through_degree_three(dimension):
    nodes = gaussian_cubature(dimension, dtype=DTYPE)
    assert nodes.shape == (2 * dimension, dimension)
    mu = torch.linspace(-0.7, 0.9, dimension, dtype=DTYPE)
    sigma = torch.linspace(0.2, 1.3, dimension, dtype=DTYPE)
    actions = mu + sigma * nodes
    for degree in range(1, 4):
        for indices in itertools.combinations_with_replacement(range(dimension), degree):
            expected = torch.ones((), dtype=DTYPE)
            for coordinate in set(indices):
                power = indices.count(coordinate)
                m, s = mu[coordinate], sigma[coordinate]
                expected *= {1: m, 2: m.square() + s.square(),
                             3: m.pow(3) + 3 * m * s.square()}[power]
            actual = actions[:, list(indices)].prod(-1).mean()
            torch.testing.assert_close(actual, expected)


def test_fixed_preaction_log_density_and_score_include_clipped_tail():
    mu = torch.tensor([[0.8, -0.2]], dtype=DTYPE, requires_grad=True)
    log_std = torch.tensor([[math.log(0.4), math.log(0.3)]], dtype=DTYPE, requires_grad=True)
    # Intentionally construct through the parameters: the helper must hold u fixed.
    epsilon = torch.tensor([[[3.0, -2.0]], [[-0.5, 0.7]]], dtype=DTYPE)
    preaction = mu + log_std.exp() * epsilon
    logp = gaussian_log_probability(preaction, mu, log_std)
    reference = torch.distributions.Normal(mu.detach(), log_std.detach().exp()).log_prob(preaction.detach()).sum(-1)
    torch.testing.assert_close(logp, reference)
    assert preaction[0, 0, 0] > 1
    gradients = torch.autograd.grad(logp.sum(), (mu, log_std))
    torch.testing.assert_close(gradients[0], (epsilon / log_std.detach().exp()).sum(0))
    torch.testing.assert_close(gradients[1], (epsilon.square() - 1).sum(0))
    clipped_logp = gaussian_log_probability(preaction.clamp(-1, 1), mu, log_std)
    assert not torch.allclose(logp, clipped_logp)


@pytest.mark.parametrize("surprisal", ["relative", "density"])
def test_gate_and_advantages_are_detached_from_score_gradient(surprisal):
    mu = torch.tensor([[0.2, -0.3]], dtype=DTYPE, requires_grad=True)
    log_std = torch.tensor([[-0.4, -0.7]], dtype=DTYPE, requires_grad=True)
    u = torch.tensor([[[1.4, -0.7]], [[-0.2, 0.8]]], dtype=DTYPE, requires_grad=True)
    advantage = torch.tensor([[2.0], [-1.0]], dtype=DTYPE, requires_grad=True)
    loss, diagnostics = delight_score_loss(u, mu, log_std, advantage, surprisal=surprisal, scope="all")
    assert not diagnostics.requires_grad
    logp = gaussian_log_probability(u, mu, log_std)
    with torch.no_grad():
        epsilon = (u - mu) / log_std.exp()
        rarity = (0.5 * epsilon.square().sum(-1)).clamp(max=10) if surprisal == "relative" else (-logp).clamp(-10, 10)
        weight = 2 * torch.sigmoid(advantage / advantage.square().mean().sqrt() * rarity)
        coefficient = weight * advantage / advantage.numel()
    gradients = torch.autograd.grad(loss, (mu, log_std, u, advantage), allow_unused=True)
    torch.testing.assert_close(gradients[0], -(coefficient[..., None] * epsilon / log_std.detach().exp()).sum(0))
    torch.testing.assert_close(gradients[1], -(coefficient[..., None] * (epsilon.square() - 1)).sum(0))
    assert gradients[2] is None
    assert gradients[3] is None


def test_mean_scope_gates_location_but_preserves_the_pg_scale_gradient():
    mu = torch.tensor([[0.2, -0.3]], dtype=DTYPE, requires_grad=True)
    log_std = torch.tensor([[-0.4, -0.7]], dtype=DTYPE, requires_grad=True)
    u = torch.tensor([[[1.4, -0.7]], [[-0.2, 0.8]]], dtype=DTYPE)
    advantage = torch.tensor([[2.0], [-1.0]], dtype=DTYPE)
    pg_loss, _ = delight_score_loss(u, mu, log_std, advantage, objective="pg")
    all_loss, _ = delight_score_loss(u, mu, log_std, advantage, scope="all")
    mean_loss, _ = delight_score_loss(u, mu, log_std, advantage, scope="mean")
    pg_gradient = torch.autograd.grad(pg_loss, (mu, log_std))
    all_gradient = torch.autograd.grad(all_loss, (mu, log_std))
    mean_gradient = torch.autograd.grad(mean_loss, (mu, log_std))
    torch.testing.assert_close(mean_gradient[0], all_gradient[0])
    torch.testing.assert_close(mean_gradient[1], pg_gradient[1])
    # This fixture would detect accidental all-parameter gating in either scope.
    assert not torch.allclose(all_gradient[1], pg_gradient[1])
    assert not torch.allclose(mean_gradient[0], pg_gradient[0])


def test_relative_gate_is_invariant_to_affine_coordinate_scale():
    mu = torch.tensor([[0.2, -0.3]], dtype=DTYPE)
    log_std = torch.tensor([[-0.4, -0.7]], dtype=DTYPE)
    u = torch.tensor([[[1.4, -0.7]], [[-0.2, 0.8]]], dtype=DTYPE)
    advantage = torch.tensor([[2.0], [-1.0]], dtype=DTYPE)
    coordinate_scale = torch.tensor([3.0, 0.2], dtype=DTYPE)
    offset = torch.tensor([-4.0, 1.0], dtype=DTYPE)
    loss, diagnostics = delight_score_loss(u, mu, log_std, advantage)
    transformed_loss, transformed = delight_score_loss(
        u * coordinate_scale + offset, mu * coordinate_scale + offset,
        log_std + coordinate_scale.log(), advantage,
    )
    # A coordinate Jacobian changes the scalar surrogate by a constant; gate and
    # rarity stay invariant. The clipping diagnostic is intentionally physical.
    torch.testing.assert_close(diagnostics[:7], transformed[:7])
    logp = gaussian_log_probability(u, mu, log_std)
    epsilon = (u - mu) / log_std.exp()
    weight = 2 * torch.sigmoid(advantage / advantage.square().mean().sqrt() * (0.5 * epsilon.square().sum(-1)))
    expected_difference = (weight * advantage).mean() * coordinate_scale.log().sum()
    torch.testing.assert_close(transformed_loss - loss, expected_difference)
    assert torch.isfinite(logp).all()


def test_reward_offset_cancels_when_subtracting_cubature_baseline():
    nodes = gaussian_cubature(2, dtype=DTYPE)
    u = torch.tensor([[[0.4, 0.7]], [[-0.2, -0.9]]], dtype=DTYPE)
    mu = torch.zeros(1, 2, dtype=DTYPE)
    log_std = torch.zeros_like(mu)
    q_samples = (2 * u[..., 0] - u[..., 1].square())
    q_baseline = (2 * nodes[:, 0] - nodes[:, 1].square()).mean()
    original = delight_score_loss(u, mu, log_std, q_samples - q_baseline)
    shifted = delight_score_loss(u, mu, log_std, (q_samples + 17) - (q_baseline + 17))
    for actual, expected in zip(shifted, original):
        torch.testing.assert_close(actual, expected)


def test_zero_advantage_is_finite_and_has_zero_policy_gradient():
    mu = torch.zeros(2, 3, dtype=DTYPE, requires_grad=True)
    log_std = torch.zeros_like(mu, requires_grad=True)
    u = torch.full((4, 2, 3), 2.0, dtype=DTYPE)
    loss, diagnostics = delight_score_loss(u, mu, log_std, torch.zeros(4, 2, dtype=DTYPE))
    assert loss == 0
    assert torch.isfinite(diagnostics).all()
    assert diagnostics[0] == 1  # 2*sigmoid(0), restores PG units.
    for gradient in torch.autograd.grad(loss, (mu, log_std)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_pg_control_and_large_temperature_restore_ungated_score():
    mu = torch.tensor([[0.2]], dtype=DTYPE, requires_grad=True)
    log_std = torch.tensor([[-0.7]], dtype=DTYPE, requires_grad=True)
    u = torch.tensor([[[1.2]], [[-0.4]]], dtype=DTYPE)
    advantage = torch.tensor([[3.0], [-2.0]], dtype=DTYPE)
    pg_loss, diagnostics = delight_score_loss(u, mu, log_std, advantage, objective="pg")
    expected = -(advantage * gaussian_log_probability(u, mu, log_std)).mean()
    torch.testing.assert_close(pg_loss, expected)
    assert diagnostics[0] == 1
    limit_loss, _ = delight_score_loss(u, mu, log_std, advantage, temperature=1e16)
    torch.testing.assert_close(limit_loss, pg_loss)
    pg_grad = torch.autograd.grad(pg_loss, (mu, log_std), retain_graph=True)
    limit_grad = torch.autograd.grad(limit_loss, (mu, log_std))
    for actual, reference in zip(limit_grad, pg_grad):
        torch.testing.assert_close(actual, reference)


def test_antithetic_pg_score_recovers_linear_action_value_gradient():
    mu = torch.tensor([[0.2, -0.3, 0.4]], dtype=DTYPE, requires_grad=True)
    log_std = torch.tensor([[-0.4, -0.7, 0.2]], dtype=DTYPE, requires_grad=True)
    slope = torch.tensor([1.2, -0.7, 2.3], dtype=DTYPE)
    epsilon = gaussian_cubature(3, dtype=DTYPE).unsqueeze(1)
    u = mu + log_std.exp() * epsilon
    advantage = ((u - mu) * slope).sum(-1)
    loss, _ = delight_score_loss(u, mu, log_std, advantage, objective="pg")
    mu_gradient, scale_gradient = torch.autograd.grad(loss, (mu, log_std))
    torch.testing.assert_close(mu_gradient, -slope.unsqueeze(0))
    torch.testing.assert_close(scale_gradient, torch.zeros_like(log_std))

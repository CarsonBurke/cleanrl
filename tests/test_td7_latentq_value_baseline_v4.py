"""Learned-V baseline identities on tensors, without models or training."""

import importlib

import pytest
import torch

from cleanrl.td7.delight.td7_latentq_value_baseline_v4 import (
    delight_score_loss,
    value_baseline_advantage,
)


@pytest.fixture(autouse=True, params=["td7_latentq_value_baseline_v4", "td7_latentq_value_floor_v5", "td7_latentq_delight_hermite_v6", "td7_latentq_delight_hermite_ablation_v7", "td7_latentq_delight_hermite_integrated_v8"])
def value_implementation(request, monkeypatch):
    module = importlib.import_module(f"cleanrl.td7.delight.{request.param}")
    for name in ("delight_score_loss", "value_baseline_advantage"):
        monkeypatch.setitem(globals(), name, getattr(module, name))


@pytest.mark.parametrize("baseline", ["latent", "cubature"])
def test_linear_q_pg_mean_gradient_is_independent_of_value_baseline(baseline):
    # Antithetic standard-normal quadrature with exact first/second moments.
    # Even deliberately inaccurate, state-dependent V must not alter the mean
    # score gradient for linear Q. Neither Q nor V may receive actor gradients.
    mu = torch.tensor([[0.2], [-0.3]], dtype=torch.float64, requires_grad=True)
    log_std = torch.full_like(mu, -1.0, requires_grad=True)
    eps = torch.tensor([[[1.], [1.]], [[-1.], [-1.]]], dtype=torch.float64)
    pre_action = (mu + log_std.exp() * eps).detach()
    slope = torch.tensor([[2.], [-3.]], dtype=torch.float64)
    sampled_q = (pre_action.squeeze(-1) * slope.squeeze(-1)).requires_grad_()
    latent_v = torch.tensor([5., -7.], dtype=torch.float64, requires_grad=True)
    cubature_v = (mu.detach() * slope).squeeze(-1).requires_grad_()
    advantages = value_baseline_advantage(sampled_q, latent_v, cubature_v, baseline)
    loss, _ = delight_score_loss(pre_action, mu, log_std, advantages, objective="pg")
    grads = torch.autograd.grad(loss, (mu, sampled_q, latent_v, cubature_v), allow_unused=True)
    torch.testing.assert_close(grads[0], -slope / len(mu))
    assert all(gradient is None for gradient in grads[1:])


def test_changing_value_baseline_can_change_dg_gate_without_critic_gradients():
    # Unlike ungated PG, DG is sensitive to baseline error through its nonlinear
    # gate. This is why the paired baseline control and sign diagnostics matter.
    mu = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    log_std = torch.zeros_like(mu, requires_grad=True)
    pre_action = torch.tensor([[[1.]], [[-1.]]], dtype=torch.float64)
    sampled_q = pre_action.squeeze(-1).clone().requires_grad_()
    latent_v = torch.tensor([3.], dtype=torch.float64, requires_grad=True)
    cubature_v = torch.zeros_like(latent_v, requires_grad=True)
    gradients = []
    for baseline in ("latent", "cubature"):
        advantage = value_baseline_advantage(sampled_q, latent_v, cubature_v, baseline)
        loss, _ = delight_score_loss(pre_action, mu, log_std, advantage, objective="delight")
        grads = torch.autograd.grad(loss, (mu, sampled_q, latent_v, cubature_v), allow_unused=True)
        gradients.append(grads[0])
        assert all(gradient is None for gradient in grads[1:])
    assert not torch.allclose(*gradients)

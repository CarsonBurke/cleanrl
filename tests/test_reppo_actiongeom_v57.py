"""Mathematical contracts for v57's trust geometry; no networks or environments."""
import torch

from cleanrl.reppo.ppo_continuous_action_reppo_actiongeom_v57 import (
    action_trust_distance, beta_concentrations, beta_kl,
)


def test_mode_budget_is_independent_of_concentration():
    old_mode = torch.tensor([[0.2, -0.4]])
    mode = (old_mode + 0.01).requires_grad_()
    distances, gradients = [], []
    for log_nu in (1.0, 6.0):
        precision = torch.full_like(mode, log_nu)
        distance, _, _, _, _ = action_trust_distance(old_mode, precision, mode, precision, 0.02, 0.03)
        distances.append(distance)
        gradients.append(torch.autograd.grad(distance.sum(), mode)[0])
    torch.testing.assert_close(distances[0], distances[1], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(gradients[0], gradients[1])
    assert gradients[0].abs().sum() > 0


def test_precision_update_retains_original_kl_and_gradient():
    mode = torch.tensor([[0.2, -0.4]])
    old_precision = torch.tensor([[2.0, 3.0]])
    precision = (old_precision + 0.15).requires_grad_()
    distance, _, _, _, _ = action_trust_distance(mode, old_precision, mode, precision, 0.02, 0.03)
    original = beta_kl(*beta_concentrations(mode, old_precision), *beta_concentrations(mode, precision))
    torch.testing.assert_close(distance, original)
    gradient = torch.autograd.grad(distance.sum(), precision, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(original.sum(), precision)[0])
    assert gradient.abs().sum() > 0


def test_physical_mode_step_consumes_one_budget():
    mode_step, bound = 0.02, 0.03
    old_mode = torch.zeros(1, 6)
    mode = 2 * torch.atanh(torch.full_like(old_mode, mode_step))
    precision = torch.full_like(mode, 4.0)
    distance, squared, _, _, _ = action_trust_distance(old_mode, precision, mode, precision, mode_step, bound)
    torch.testing.assert_close(distance, torch.full_like(distance, bound))
    torch.testing.assert_close(squared, torch.full_like(squared, mode_step**2))

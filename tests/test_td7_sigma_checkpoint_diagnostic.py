"""Pure distribution math; no networks or training are run by these tests."""
import math

import numpy as np
import pytest
import torch

from benchmarks.td7_delight_cubature_v1.diagnose_sigma_checkpoint import clipped_gaussian_log_probability


@pytest.mark.parametrize("mu_value,sigma", [(0., .1), (.9, .4), (-.8, .7), (0., 1.)])
def test_clipped_distribution_score_expectation_is_zero(mu_value, sigma):
    # Integrate the Lebesgue interior and explicitly add the two boundary atoms.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    actions = torch.tensor(np.r_[nodes, -1., 1.], dtype=torch.float64).reshape(-1, 1, 1)
    mu = torch.tensor([[mu_value]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[math.log(sigma)]], dtype=torch.float64, requires_grad=True)
    log_prob = clipped_gaussian_log_probability(actions, mu, log_std)
    quadrature = torch.tensor(np.r_[weights, 1., 1.], dtype=torch.float64).reshape(-1, 1)
    probabilities = log_prob.detach().exp() * quadrature
    assert probabilities.sum().item() == pytest.approx(1., abs=1e-12)
    gradients = torch.autograd.grad((log_prob * probabilities).sum(), (mu, log_std))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


def test_boundary_score_and_rarity_do_not_depend_on_latent_overshoot():
    mu = torch.tensor([[.8]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[-.4]], dtype=torch.float64, requires_grad=True)
    near = clipped_gaussian_log_probability(torch.tensor([[[1.01]]]), mu, log_std)
    far = clipped_gaussian_log_probability(torch.tensor([[[3.0]]]), mu, log_std)
    torch.testing.assert_close(near, far, atol=0, rtol=0)
    a = torch.autograd.grad(near.sum(), (mu, log_std), retain_graph=True)
    b = torch.autograd.grad(far.sum(), (mu, log_std))
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.parametrize("normalizer", [0., -2.767])
def test_joint_dg_inflates_reward_flat_coordinate_with_exact_mean_reference(normalizer):
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    epsilon = np.sqrt(2) * nodes
    probability = weights / np.sqrt(np.pi)
    first, second = np.meshgrid(epsilon, epsilon, indexing="ij")
    advantage = 1 - first ** 2  # Q = -eps_1^2, exact Gaussian mean Q = -1.
    surprise = np.clip(.5 * (first ** 2 + second ** 2) + normalizer, -10, 10)
    gate = 2 * torch.sigmoid(torch.from_numpy(advantage * surprise)).numpy()
    p = probability[:, None] * probability[None, :]
    pg = [np.sum(p * advantage * (noise ** 2 - 1)) for noise in [first, second]]
    dg = [np.sum(p * gate * advantage * (noise ** 2 - 1)) for noise in [first, second]]
    assert pg == pytest.approx([-2., 0.], abs=1e-12)
    assert dg[0] < -.4
    assert dg[1] > .1

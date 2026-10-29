"""Analytical tests of action integration and trace boundaries; no network runs."""
import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl/td7/latentq/td7_latentq_action_integral_fresh_v11.py"
SPEC = importlib.util.spec_from_file_location("action_integral_v11", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_antithetic_affine_expectation_and_gradients():
    mu = torch.tensor([[0.1, -0.2]], requires_grad=True)
    sigma = torch.tensor([[0.15, 0.2]], requires_grad=True)
    noise = torch.tensor([[[1.0, -0.5]], [[-0.25, 1.0]]])
    weights = torch.tensor([2.0, -3.0])
    actions = MODULE.antithetic_actions(mu, sigma, noise)
    expected = (actions * weights).sum(-1).mean()
    torch.testing.assert_close(expected, (mu * weights).sum())
    expected.backward()
    torch.testing.assert_close(mu.grad, weights.unsqueeze(0))
    torch.testing.assert_close(sigma.grad, torch.zeros_like(sigma))


def test_quadratic_expectation_and_variance_gradient():
    # Symmetric unit-second-moment quadrature makes the Gaussian quadratic
    # expectation exact, exposing both the mean and std pathwise derivatives.
    mu = torch.tensor([[0.1, -0.15]], requires_grad=True)
    sigma = torch.tensor([[0.2, 0.3]], requires_grad=True)
    noise = torch.tensor([[[1.0, -1.0]], [[-1.0, 1.0]]])
    weights = torch.tensor([2.0, 3.0])
    expected = (MODULE.antithetic_actions(mu, sigma, noise).square() * weights).sum(-1).mean()
    torch.testing.assert_close(expected, ((mu.square() + sigma.square()) * weights).sum())
    expected.backward()
    torch.testing.assert_close(mu.grad, 2 * mu.detach() * weights)
    torch.testing.assert_close(sigma.grad, 2 * sigma.detach() * weights)


def test_clipped_action_expectation_has_saturation_derivatives():
    mu = torch.tensor([[0.9]], requires_grad=True)
    sigma = torch.tensor([[0.5]], requires_grad=True)
    noise = torch.ones(1, 1, 1)
    actions = MODULE.antithetic_actions(mu, sigma, noise)
    torch.testing.assert_close(actions, torch.tensor([[[1.0]], [[0.4]]]))
    actions.mean().backward()
    torch.testing.assert_close(mu.grad, torch.tensor([[0.5]]))
    torch.testing.assert_close(sigma.grad, torch.tensor([[-0.5]]))


def test_lambda_truncation_bootstraps_without_crossing_autoreset():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[100.0]]])
    values = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
    not_done = torch.ones_like(rewards)
    boundaries = torch.tensor([[[0.0]], [[1.0]], [[0.0]]])
    targets = MODULE.lambda_returns(rewards, not_done, boundaries, values, 0.9, 0.5)
    # Last row: 100 + .9*30. Truncation: 2 + .9*20; row0 mixes 10 and 20.
    torch.testing.assert_close(targets, torch.tensor([[[14.5]], [[20.0]], [[127.0]]]))
    rewards[-1] = -999
    changed = MODULE.lambda_returns(rewards, not_done, boundaries, values, 0.9, 0.5)
    torch.testing.assert_close(targets[:2], changed[:2])


def test_lambda_termination_and_parallel_environment_independence():
    rewards = torch.tensor([[[1.0], [5.0]], [[2.0], [6.0]]])
    values = torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]])
    not_done = torch.tensor([[[0.0], [1.0]], [[1.0], [1.0]]])
    boundaries = 1 - not_done
    targets = MODULE.lambda_returns(rewards, not_done, boundaries, values, 0.9, 1.0)
    torch.testing.assert_close(targets, torch.tensor([[[1.0], [42.8]], [[29.0], [42.0]]]))
    one_step = MODULE.lambda_returns(rewards, not_done, boundaries, values, 0.9, 0.0)
    torch.testing.assert_close(one_step, rewards + 0.9 * not_done * values)


def test_raw_gaussian_score_gradient_matches_quadratic_pathwise_gradient():
    # Three-node Gaussian quadrature integrates moments through degree five.
    # Actions stay inside the clip; stored raw draws and return baseline must be
    # detached when computing the likelihood-ratio gradient.
    mu = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[0.2]], dtype=torch.float64).log().requires_grad_()
    noise = torch.tensor([-(3.0 ** 0.5), 0.0, 3.0 ** 0.5], dtype=torch.float64).reshape(3, 1)
    weights = torch.tensor([1 / 6, 2 / 3, 1 / 6], dtype=torch.float64)
    actions = mu + log_std.exp() * noise
    path_loss = -(actions.square().squeeze(-1) * weights).sum()
    path_grads = torch.autograd.grad(path_loss, (mu, log_std))
    stored_actions = actions.detach()
    baseline = (mu.square() + log_std.exp().square()).detach()
    advantage = (stored_actions.square() - baseline).squeeze(-1)
    score_loss = -(advantage * MODULE.gaussian_log_prob(mu, log_std, stored_actions) * weights).sum()
    score_grads = torch.autograd.grad(score_loss, (mu, log_std))
    torch.testing.assert_close(path_grads[0], torch.tensor([[-0.2]], dtype=torch.float64))
    torch.testing.assert_close(path_grads[1], torch.tensor([[-0.08]], dtype=torch.float64))
    for path_grad, score_grad in zip(path_grads, score_grads):
        torch.testing.assert_close(path_grad, score_grad)

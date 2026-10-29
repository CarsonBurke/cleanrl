"""Pure tensor checks for physical integration and moving one-step TD labels."""
import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl/td7/latentq/td7_latentq_action_integral_td_v16.py"
SPEC = importlib.util.spec_from_file_location("action_integral_td_v16", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_quadratic_integral_has_mean_and_scale_pathwise_gradients():
    mu = torch.tensor([[0.1, -0.15]], requires_grad=True)
    sigma = torch.tensor([[0.2, 0.3]], requires_grad=True)
    noise = torch.tensor([[[1.0, -1.0]], [[-1.0, 1.0]]])
    weights = torch.tensor([2.0, 3.0])
    expected = (MODULE.antithetic_actions(mu, sigma, noise).square() * weights).sum(-1).mean()
    torch.testing.assert_close(expected, ((mu.square() + sigma.square()) * weights).sum())
    expected.backward()
    torch.testing.assert_close(mu.grad, 2 * mu.detach() * weights)
    torch.testing.assert_close(sigma.grad, 2 * sigma.detach() * weights)


def test_target_integral_retains_fixed_point_two_std_and_clipped_actions():
    mu = torch.tensor([[0.1]])
    noise = torch.ones(4, 1, 1)
    target_actions = MODULE.target_smoothing_actions(mu, noise, 0.2)
    torch.testing.assert_close(target_actions.square().mean(), torch.tensor(0.05))
    behavior_actions = MODULE.antithetic_actions(mu, torch.full_like(mu, 0.6), noise)
    torch.testing.assert_close(behavior_actions.square().mean(), torch.tensor(0.37))
    saturated = MODULE.target_smoothing_actions(torch.tensor([[0.95]]), noise, 0.2)
    torch.testing.assert_close(saturated[:4], torch.ones(4, 1, 1))
    torch.testing.assert_close(saturated[4:], torch.full((4, 1, 1), 0.75))


def test_td_target_bounds_integrated_twins_and_masks_only_termination():
    q = torch.tensor([[[10.0, 1.0], [2.0, 4.0], [-5.0, -2.0]],
                      [[1.0, 10.0], [6.0, 3.0], [-7.0, -8.0]]])
    reward = torch.tensor([[5.0], [7.0], [9.0]])
    # Row0 is terminal. Row1 is time-limit truncated with its final observation;
    # row2 is an ordinary continuation. Both latter rows bootstrap.
    not_done = torch.tensor([[0.0], [1.0], [1.0]])
    result = MODULE.one_step_targets(reward, not_done, q, 0.9, -4.0, 2.0)
    torch.testing.assert_close(result, torch.tensor([[5.0], [8.8], [5.4]]))


def test_one_step_label_moves_with_target_critic_without_reward_recursion():
    q = torch.tensor([[[3.0, 5.0]], [[5.0, 7.0]]])
    reward, not_done = torch.tensor([[2.0]]), torch.ones(1, 1)
    old = MODULE.one_step_targets(reward, not_done, q, 0.9, -100.0, 100.0)
    new = MODULE.one_step_targets(reward, not_done, q + 1, 0.9, -100.0, 100.0)
    torch.testing.assert_close(new - old, torch.tensor([[0.9]]))


def test_crossing_twins_take_min_after_integration_for_v9_fairness():
    q = torch.tensor([[[10.0, 1.0]], [[1.0, 10.0]]])
    result = MODULE.one_step_targets(torch.zeros(1, 1), torch.ones(1, 1), q, 1.0, -100.0, 100.0)
    # Each twin integrates to 5.5. Taking a minimum per sampled action instead
    # gives 1.0 and introduces an extra pessimistic bias absent from v9.
    torch.testing.assert_close(result, torch.tensor([[5.5]]))

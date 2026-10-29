"""Analytical bootstrap propagation tests; no models or GPU execution."""
import pytest
import torch

from cleanrl.td7.latentq.td7_latentq_preaction_lambda_refresh_cv_v20 import lambda_returns


def targets(values, not_done=None, boundaries=None):
    reward = torch.arange(1, len(values) + 1, dtype=torch.float64).reshape(-1, 1, 1)
    values = torch.as_tensor(values, dtype=torch.float64).reshape_as(reward)
    not_done = torch.ones_like(reward) if not_done is None else torch.tensor(not_done).reshape_as(reward)
    boundaries = torch.zeros_like(reward) if boundaries is None else torch.tensor(boundaries).reshape_as(reward)
    return lambda_returns(reward, not_done, boundaries, values, 0.99, 0.95, 32).flatten()


def test_endpoint_bootstrap_refresh_keeps_thirty_two_step_signal():
    before = targets([0] * 32)
    values = [0] * 32
    values[-1] = 10
    delta = targets(values) - before
    expected = torch.tensor([10 * 0.99 * (0.99 * 0.95)**(31 - t) for t in range(32)], dtype=torch.float64)
    torch.testing.assert_close(delta, expected)
    assert delta[0] > 0


def test_intermediate_bootstrap_refresh_has_lambda_mixture_weight():
    values = [0] * 32
    values[8] = 10
    delta = targets(values) - targets([0] * 32)
    expected = torch.zeros(32, dtype=torch.float64)
    for t in range(9):
        expected[t] = 10 * 0.99 * (1 - 0.95) * (0.99 * 0.95)**(8 - t)
    torch.testing.assert_close(delta, expected)


@pytest.mark.parametrize('not_done,expected', [([0, 1, 1], 0.0), ([1, 1, 1], 9.9)])
def test_refresh_distinguishes_termination_and_truncation_without_cross_reset(not_done, expected):
    delta = targets([10, 0, 0], not_done=not_done, boundaries=[1, 0, 0]) - targets([0, 0, 0], not_done=not_done, boundaries=[1, 0, 0])
    torch.testing.assert_close(delta, torch.tensor([expected, 0, 0], dtype=torch.float64))
    # Later episode's changed bootstrap cannot cross the boundary.
    later = targets([0, 10, 10], not_done=not_done, boundaries=[1, 0, 0])
    assert later[0] == targets([0, 0, 0], not_done=not_done, boundaries=[1, 0, 0])[0]


def test_fixed_controls_cancel_in_refresh_increment():
    before, after = targets([0] * 32), targets([10] * 32)
    fixed_control = torch.linspace(-2, 3, 32, dtype=torch.float64)
    torch.testing.assert_close((after - fixed_control) - (before - fixed_control), after - before)
    torch.testing.assert_close(targets([10] * 64)[:32], after)

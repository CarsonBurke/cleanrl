"""Analytical control-variate tensor tests; no ML models or GPU execution."""
import math

import pytest
import torch

from cleanrl.td7.latentq.td7_latentq_preaction_lambda_cv_v18 import (
    clean_labels,
    future_controls,
    lambda_returns,
    paired_residual_metrics,
)


def controls(noise, gamma=1.0, trace_lambda=1.0, horizon=32, not_done=None, boundaries=None):
    noise = torch.tensor(noise, dtype=torch.float64).reshape(-1, 1, 1)
    mask = torch.ones_like(noise) if not_done is None else torch.tensor(not_done).reshape_as(noise)
    cut = torch.zeros_like(noise) if boundaries is None else torch.tensor(boundaries).reshape_as(noise)
    return future_controls(noise, mask, cut, gamma, trace_lambda, horizon).flatten()


def test_current_innovation_never_subtracted_and_final_control_zero():
    torch.testing.assert_close(controls([100.0, 3.0, 5.0]), torch.tensor([8.0, 5.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(controls([-10000.0, 3.0, 5.0]), controls([100.0, 3.0, 5.0]))


def test_future_control_has_gamma_lambda_powers():
    actual = controls([100.0, 3.0, 5.0], gamma=0.8, trace_lambda=0.5)
    torch.testing.assert_close(actual, torch.tensor([0.4 * 3 + 0.4**2 * 5, 0.4 * 5, 0], dtype=torch.float64))


def test_thirty_two_step_cuts_and_short_final_segment():
    actual = controls([1.0] * 65)
    expected = torch.cat([torch.arange(31, -1, -1, dtype=torch.float64).repeat(2), torch.zeros(1)])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("not_done", [[0, 1, 1], [1, 1, 1]])
def test_terminal_or_truncation_prevents_control_crossing_reset(not_done):
    actual = controls([10.0, 1000.0, 3.0], not_done=not_done, boundaries=[1, 0, 0])
    torch.testing.assert_close(actual, torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64))


def test_zero_strength_reproduces_exact_v17_labels():
    reward = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    values = torch.full_like(reward, 10.0)
    raw = lambda_returns(reward, torch.ones_like(reward), torch.zeros_like(reward), values, 0.99, 0.95)
    arbitrary_controls = torch.full((3, 1, 2), 1e20)
    clean = clean_labels(raw, arbitrary_controls, 0.0)
    assert torch.equal(clean, raw)
    assert clean is raw


def test_gaussian_quadrature_preserves_conditional_label_mean_and_first_action_signal():
    # Three-node Gaussian quadrature weights 1/6, 4/6, 1/6, represented as repeats.
    # Exact for linear/quadratic Hermite means and their fourth-degree variance.
    nodes = torch.tensor([-math.sqrt(3), 0, 0, 0, 0, math.sqrt(3)], dtype=torch.float64)
    first, later = torch.meshgrid(nodes, nodes, indexing="ij")
    first, later = first.flatten(), later.flatten()
    noise = torch.stack([first, later]).unsqueeze(-1)
    innovations = torch.cat([noise, 0.5 * (noise.square() - 1)], dim=-1)
    mask = torch.ones(2, 36, 1, dtype=torch.float64)
    control = future_controls(innovations, mask, torch.zeros_like(mask), 0.9, 0.95)
    first_signal = 2 * first + first.square()
    raw = first_signal[:, None] + control[0].sum(-1, keepdim=True)
    clean = clean_labels(raw, control[0], 1.0)
    torch.testing.assert_close(clean[:, 0], first_signal)
    raw_conditional = raw.reshape(6, 6).mean(1)
    clean_conditional = clean.reshape(6, 6).mean(1)
    torch.testing.assert_close(raw_conditional, clean_conditional)
    torch.testing.assert_close(clean_conditional, 2 * nodes + nodes.square())
    torch.testing.assert_close(innovations[1].mean(0), torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=0)
    metrics = paired_residual_metrics(raw, clean, torch.zeros_like(raw), control[0])
    removed_variance = (0.9 * 0.95) ** 2 * (1 + 0.5)
    assert metrics["raw_residual_variance"] - metrics["clean_residual_variance"] == pytest.approx(removed_variance)
    assert metrics["residual_control_covariance"] == pytest.approx(removed_variance)
    assert metrics["variance_ratio"] < 1


def test_variance_proxy_reports_increase_and_undefined_zero_baseline():
    raw = torch.tensor([[-1.0], [1.0]])
    control = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    clean = clean_labels(raw, control, 1.0)
    metrics = paired_residual_metrics(raw, clean, torch.zeros_like(raw), control)
    assert metrics["variance_ratio"] == 4
    assert metrics["variance_reduction_fraction"] == -3
    zero = torch.zeros_like(raw)
    metrics = paired_residual_metrics(zero, zero, zero, torch.zeros_like(control))
    assert torch.isnan(metrics["variance_ratio"])
    assert torch.isnan(metrics["variance_reduction_fraction"])

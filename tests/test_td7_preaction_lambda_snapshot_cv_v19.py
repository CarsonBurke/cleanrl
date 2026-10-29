"""Analytical covariance tests only; no models or GPU evaluation."""
import math

import torch

from cleanrl.td7.latentq.td7_latentq_preaction_lambda_snapshot_cv_v19 import (
    calibrate_controls,
    clean_labels,
    future_controls,
    paired_residual_metrics,
)


def panel():
    nodes = torch.tensor([-math.sqrt(3), 0, 0, 0, 0, math.sqrt(3)], dtype=torch.float64)
    a, b = torch.meshgrid(nodes, nodes, indexing="ij")
    return torch.stack([a.flatten(), b.flatten()], dim=1)


def test_joint_calibration_recovers_correlated_signed_orders():
    independent = panel()
    controls = torch.stack([2 * independent[:, 0], independent[:, 0] + independent[:, 1]], dim=1)
    expected = torch.tensor([0.25, -0.5], dtype=torch.float64)
    center = torch.arange(len(controls), dtype=torch.float64).unsqueeze(1)
    raw = center + 7 + (controls @ expected).unsqueeze(1)
    actual = calibrate_controls(raw, center, controls)
    torch.testing.assert_close(actual, expected)
    clean = clean_labels(raw, controls * actual, 1)
    torch.testing.assert_close(clean, center + 7)


def test_absent_or_collinear_controls_have_minimum_norm_solution():
    x = panel()[:, 0]
    controls = torch.stack([x, 2 * x], dim=1)
    raw = (3 * x).unsqueeze(1)
    coefficient = calibrate_controls(raw, torch.zeros_like(raw), controls)
    torch.testing.assert_close(coefficient, torch.tensor([0.6, 1.2], dtype=torch.float64))
    empty = calibrate_controls(raw, torch.zeros_like(raw), torch.zeros_like(controls))
    torch.testing.assert_close(empty, torch.zeros(2, dtype=torch.float64))


def test_calibration_is_invariant_to_constant_control_and_label_offsets():
    controls = panel()
    raw = (controls @ torch.tensor([0.3, 0.8], dtype=torch.float64)).unsqueeze(1)
    original = calibrate_controls(raw, torch.zeros_like(raw), controls)
    shifted = calibrate_controls(raw + 17, torch.zeros_like(raw), controls + 9)
    torch.testing.assert_close(original, shifted)


def test_same_snapshot_scale_matters_for_future_variance():
    # Deliberately apply a coefficient to different slopes: mean preservation
    # survives, while calibration and variance reduction do not.
    controls = panel()
    raw = (controls[:, 0] * 0.1).unsqueeze(1)
    coefficient = calibrate_controls(raw, torch.zeros_like(raw), controls)
    paired = clean_labels(raw, controls * coefficient, 1)
    stale = clean_labels(raw, controls * 30 * coefficient, 1)
    assert paired.var() < raw.var()
    assert stale.var() > raw.var()
    torch.testing.assert_close(stale.mean(), raw.mean(), atol=1e-15, rtol=0)


def test_previous_coefficients_preserve_next_conditional_mean_and_current_action():
    calibration_controls = panel()
    calibration_raw = (calibration_controls @ torch.tensor([0.4, -0.2], dtype=torch.float64)).unsqueeze(1)
    beta = calibrate_controls(calibration_raw, torch.zeros_like(calibration_raw), calibration_controls)
    # A separate quadrature panel represents next-collection action draws.
    nodes = torch.tensor([-math.sqrt(3), 0, 0, 0, 0, math.sqrt(3)], dtype=torch.float64)
    current, later = torch.meshgrid(nodes, nodes, indexing="ij")
    eps = torch.stack([current.flatten(), later.flatten()]).unsqueeze(-1)
    innovations = torch.cat([eps, 0.5 * (eps.square() - 1)], dim=-1)
    mask = torch.ones(2, 36, 1, dtype=torch.float64)
    controls = future_controls(innovations, mask, torch.zeros_like(mask), 0.99, 0.95)
    raw = (3 * current.flatten()).unsqueeze(1) + (controls[0] @ beta).unsqueeze(1)
    clean = clean_labels(raw, controls[0] * beta, 1)
    torch.testing.assert_close(raw.reshape(6, 6).mean(1), clean.reshape(6, 6).mean(1))
    torch.testing.assert_close(clean[:, 0], 3 * current.flatten())
    torch.testing.assert_close(controls[1], torch.zeros_like(controls[1]))
    assert paired_residual_metrics(raw, clean, torch.zeros_like(raw), controls[0] * beta)["variance_ratio"] < 1


def test_zero_initial_coefficients_leave_labels_exactly_unchanged():
    raw = torch.arange(36, dtype=torch.float64).unsqueeze(1)
    controls = panel()
    clean = clean_labels(raw, controls * torch.zeros(2), 1)
    assert torch.equal(clean, raw)

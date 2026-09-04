"""Exactness checks for the spline coupling flow in eflow_v1.

The M-step and trust region rely on the flow's log-density being exact, so
these compare the analytic log-determinants against autograd Jacobians,
verify forward/inverse round trips, and confirm the identity initialisation.
"""
import importlib

import pytest
import torch

eflow = importlib.import_module(
    "cleanrl.vmpo.ppo_continuous_action_iterthink_v24_eflow_v1_vmpo_exact_spline_coupling_flow"
)

torch.manual_seed(0)
DEVICE = torch.device("cuda")


def make_flow(randomize, dtype=torch.float64):
    """Random splines of moderate steepness; extreme knots are covered by the
    single-spline test, where the inverse's conditioning is under control."""
    flow = eflow.SplineCouplingFlow(6, 16, layers=3, bins=8, hidden=32).to(DEVICE, dtype)
    if randomize:
        with torch.no_grad():
            for conditioner in flow.conditioners:
                conditioner[-1].weight.normal_(std=0.05)
                conditioner[-1].bias.normal_(std=0.5)
    return flow


def test_identity_at_init():
    flow = make_flow(randomize=False)
    base = torch.rand(256, 6, device=DEVICE, dtype=torch.float64)
    context = torch.randn(256, 16, device=DEVICE, dtype=torch.float64)
    action, log_abs_det = flow(base, context)
    assert torch.allclose(action, base, atol=1e-6)
    assert torch.allclose(log_abs_det, torch.zeros_like(log_abs_det), atol=1e-6)


def test_round_trip_and_log_det_sign():
    flow = make_flow(randomize=True)
    base = torch.rand(512, 6, device=DEVICE, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
    context = torch.randn(512, 16, device=DEVICE, dtype=torch.float64)
    action, forward_log_abs_det = flow(base, context)
    assert (action > 0).all() and (action < 1).all()
    recovered, inverse_log_abs_det = flow.inverse(action, context)
    # Inverse inputs are clamped to [SAMPLE_EPS, 1 - SAMPLE_EPS], exactly as
    # sampled actions are, so only interior actions round-trip exactly.
    interior = ((action > 1e-5) & (action < 1 - 1e-5)).all(dim=-1)
    assert interior.sum() > 400
    assert torch.allclose(recovered[interior], base[interior], atol=1e-10)
    assert torch.allclose(
        forward_log_abs_det[interior], -inverse_log_abs_det[interior], atol=1e-8
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_log_det_matches_autograd(inverse):
    flow = make_flow(randomize=True)
    context = torch.randn(8, 16, device=DEVICE, dtype=torch.float64)
    points = torch.rand(8, 6, device=DEVICE, dtype=torch.float64).clamp(0.05, 0.95)
    for row in range(8):
        def mapping(x):
            fn = flow.inverse if inverse else flow.forward
            return fn(x.unsqueeze(0), context[row : row + 1])[0].squeeze(0)

        jacobian = torch.autograd.functional.jacobian(mapping, points[row])
        analytic = (flow.inverse if inverse else flow.forward)(
            points[row : row + 1], context[row : row + 1]
        )[1].squeeze(0)
        numeric = torch.linalg.slogdet(jacobian)[1]
        assert torch.allclose(analytic, numeric, atol=1e-8), (analytic, numeric)


def test_single_spline_monotone_and_exact_inverse():
    torch.manual_seed(3)
    grid = torch.linspace(1e-4, 1 - 1e-4, 4000, device=DEVICE, dtype=torch.float64)
    for scale in (0.1, 1.0, 3.0):
        raw = [
            torch.randn(1, size, device=DEVICE, dtype=torch.float64).expand(4000, -1) * scale
            for size in (8, 8, 9)
        ]
        outputs, log_derivative = eflow.rational_quadratic_spline(grid, *raw, False)
        assert (outputs[1:] - outputs[:-1] > 0).all()
        assert torch.isfinite(log_derivative).all()
        recovered, inverse_log_derivative = eflow.rational_quadratic_spline(
            outputs, *raw, True
        )
        interior = (outputs > 1e-5) & (outputs < 1 - 1e-5)
        assert torch.allclose(recovered[interior], grid[interior], atol=1e-9)
        assert torch.allclose(
            log_derivative[interior], -inverse_log_derivative[interior], atol=1e-8
        )

"""Curvature transport contracts; execute CUDA checks only through mlq.

These tests establish algebra, causality and checkpoint behavior, not benchmark
performance. No training result is inferred from this small deterministic input.
"""

import copy

import pytest
import torch

from cleanrl.plasticity.predictive_transport_optimizer_v1 import (
    PredictiveTransport,
    TransportState,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_diagonal_transport_recovers_path_independent_evidence():
    anchor = torch.tensor([[0.3, -0.2]], device="cuda")
    weights = [anchor.clone(), anchor.clone()]
    states = [TransportState(w, prior_density=1.0) for w in weights]
    teacher = torch.tensor([[1.2, 0.7]], device="cuda")
    h = torch.tensor([[0.25, 2.0]], device="cuda")
    variance = torch.ones((1, 1), device="cuda")
    for step in range(1, 25):
        # Different iterates of the SAME diagonal quadratic observation must
        # yield identical transported evidence, not merely similar predictions.
        weights[1].copy_(anchor + step * torch.tensor([[0.2, -0.1]], device="cuda"))
        for weight, state in zip(weights, states):
            state.step(weight, h * (weight - teacher), h, variance)
    torch.testing.assert_close(states[0].score, states[1].score, rtol=2e-5, atol=2e-5)
    expected_score = 24 * h * (teacher - anchor)
    torch.testing.assert_close(states[0].score, expected_score)
    # Dense prior gives Gaussian posterior with Q=24*h and A=24*h.
    expected = anchor + (teacher - anchor) * (24 * h) / (1 + 24 * h)
    torch.testing.assert_close(weights[0], expected)
    torch.testing.assert_close(weights[1], expected, rtol=2e-5, atol=2e-5)


def test_unobserved_and_noiseless_coordinates_do_not_corrupt_weights():
    weight = torch.zeros((2, 3), device="cuda")
    density = torch.tensor([[1.0], [0.01]], device="cuda")
    state = TransportState(weight, prior_density=density)
    compiled = torch.compile(state.step, fullgraph=True, options={"triton.cudagraphs": False})
    curvature = torch.tensor([[0.0, 1.0, 2.0]], device="cuda")
    for _ in range(3):
        compiled(weight, torch.zeros_like(weight), curvature, torch.zeros((2, 1), device="cuda"))
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
    assert torch.isfinite(state.inclusion).all()
    # A truly unobserved coordinate retains its prior, not a fictitious sample.
    torch.testing.assert_close(state.inclusion[:, :1], density)


def test_checkpoint_resume_preserves_next_prediction_and_update():
    parameter = torch.nn.Parameter(torch.tensor([[0.2, -0.1]], device="cuda"))
    optimizer = PredictiveTransport([parameter], prior_scale=0.7, prior_density=0.2)
    x = torch.tensor([[1.0, -0.5]], device="cuda")
    for target in (0.1, -0.3, 0.8):
        residual = (parameter * x).sum(-1, keepdim=True) - target
        parameter.grad = (residual * x).detach()
        optimizer.step(curvatures=[x.square()], residual_squares=[residual.detach().square()])
    restored = torch.nn.Parameter(parameter.detach().clone())
    resumed = PredictiveTransport([restored])
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for target in (-0.4, 0.6):
        for p, opt in ((parameter, optimizer), (restored, resumed)):
            residual = (p * x).sum(-1, keepdim=True) - target
            p.grad = (residual * x).detach()
            opt.step(curvatures=[x.square()], residual_squares=[residual.detach().square()])
        torch.testing.assert_close(parameter, restored, rtol=0, atol=0)


def test_opposed_evidence_retracts_a_previously_supported_displacement():
    weight = torch.zeros((1, 1), device="cuda")
    state = TransportState(weight, prior_density=0.01)
    curvature = torch.ones_like(weight)
    variance = torch.ones_like(weight)
    for _ in range(100):
        state.step(weight, weight - 1, curvature, variance)
    assert weight.item() > 0.98
    for _ in range(100):
        state.step(weight, weight + 1, curvature, variance)
    assert abs(weight.item()) < 1e-6
    assert state.inclusion.item() < 0.001


def test_large_finite_curvature_does_not_overflow_posterior():
    weight = torch.zeros((1, 1), device="cuda")
    state = TransportState(weight, prior_density=0.5)
    state.step(weight, torch.full_like(weight, -1),
               torch.full_like(weight, 1e20), torch.full_like(weight, 1e-20))
    # Q=1, b=1, A=1e20: almost certain exclusion, but a finite posterior.
    assert torch.isfinite(weight).all()
    assert torch.isfinite(state.inclusion).all()
    assert weight.item() >= 0


def test_added_parameter_group_receives_its_own_evidence():
    original = torch.nn.Parameter(torch.zeros((1, 1), device="cuda"))
    added = torch.nn.Parameter(torch.zeros((1, 1), device="cuda"))
    optimizer = PredictiveTransport([original], prior_density=1.0)
    optimizer.add_param_group({"params": [added]})
    added.grad = torch.full_like(added, -1)
    optimizer.step(curvatures=[None, torch.ones_like(added)],
                   residual_squares=[None, torch.ones_like(added)])
    torch.testing.assert_close(added, torch.full_like(added, 0.5))
    torch.testing.assert_close(original, torch.zeros_like(original))

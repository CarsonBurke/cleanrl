"""Deterministic learning-action contracts; execute CUDA checks through mlq."""

import copy
import math

import pytest
import torch

from cleanrl.plasticity.prospective_memory_optimizer_v1 import (
    ProspectiveMemory,
    ProspectiveMemoryState,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_scalar_frozen_controller_sensitivities_match_autograd():
    weight = torch.zeros((1, 1), device="cuda")
    state = ProspectiveMemoryState(weight, initial_lr=0.4, retain_rate=0.2, meta_lr=0)
    q = torch.tensor(math.log(0.4), dtype=torch.float64, device="cuda", requires_grad=True)
    z = torch.tensor(math.log(4.0), dtype=torch.float64, device="cuda", requires_grad=True)
    reference = q * 0
    for x, y in ((1., 0.5), (0., 2.), (2., -0.3), (0.5, 1.0)):
        grad = (weight * x - y) * x
        state.step(weight, grad, torch.full_like(weight, x * x))
        reference = (z.sigmoid() if x != 0 else 1.0) * reference - q.exp() * (reference * x - y) * x / (1 + q.exp() * x * x)
    dq, dz = torch.autograd.grad(reference, (q, z))
    torch.testing.assert_close(weight.squeeze().double(), reference.detach(), rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(state.write_trace.squeeze().double(), dq, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(state.retain_trace.squeeze().double(), dz, rtol=2e-5, atol=1e-7)


def test_current_action_cannot_use_its_own_credit_to_change_rate():
    weights = [torch.zeros((1, 1), device="cuda") for _ in range(2)]
    states = [ProspectiveMemoryState(w, initial_lr=0.1, meta_lr=0.1, control="write") for w in weights]
    states[1].write_trace.fill_(1)
    grad = torch.ones_like(weights[0])
    curvature = torch.ones_like(grad)
    for w, state in zip(weights, states):
        state.step(w, grad, curvature)
    torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
    assert states[1].log_write.item() < states[0].log_write.item()
    for w, state in zip(weights, states):
        state.step(w, grad, curvature)
    assert weights[1].item() > weights[0].item()


def test_latest_action_credit_includes_other_parameter_effects():
    weight = torch.zeros((1, 2), device="cuda")
    state = ProspectiveMemoryState(weight, initial_lr=1., meta_lr=0.01, control="write")
    state.step(weight, -torch.ones_like(weight), torch.ones_like(weight))
    # First action writes (1/3,1/3). Future x=(1,3),y=1/3 gives g=(1,3).
    # Exact dL/dq1=-1/9, while a diagonal-only rule wrongly gives +2/9.
    state.step(weight, torch.tensor([[1., 3.]], device="cuda"),
               torch.tensor([[1., 9.]], device="cuda"))
    assert state.log_write[0, 0].item() > 0
    assert state.log_write[0, 1].item() < 0


def test_cross_credit_for_an_inactive_coordinate_is_normalized():
    weight = torch.zeros((1, 2), device="cuda")
    state = ProspectiveMemoryState(weight, initial_lr=1., meta_lr=0.01, control="write")
    state.step(weight, -torch.ones_like(weight), torch.ones_like(weight))
    # Previous q1 changed w2 through the common denominator. Its future credit
    # is nonzero even though feature1 is NOW absent; a current-h-only RMS fails.
    state.step(weight, torch.tensor([[0., 1. / 3]], device="cuda"),
               torch.tensor([[0., 1.]], device="cuda"))
    assert 0 < state.log_write[0, 0].item() < 0.1
    assert torch.isfinite(weight).all()


@pytest.mark.parametrize("control,factor", [("full", 1.0), ("wall", 0.8)])
def test_inactive_clock_follows_declared_control_without_inventing_credit(control, factor):
    weight = torch.zeros((1, 1), device="cuda")
    state = ProspectiveMemoryState(weight, retain_rate=0.2, meta_lr=0.1, control=control)
    weight.fill_(1)
    old_q, old_z = state.log_write.clone(), state.retain_logit.clone()
    for _ in range(5):
        state.step(weight, torch.zeros_like(weight), torch.zeros_like(weight))
    torch.testing.assert_close(weight, torch.full_like(weight, factor ** 5))
    torch.testing.assert_close(state.log_write, old_q, rtol=0, atol=0)
    torch.testing.assert_close(state.retain_logit, old_z, rtol=0, atol=0)


def test_compiled_extreme_write_keeps_zero_axes_finite_and_derivative_alive():
    weight = torch.zeros((1, 2), device="cuda")
    state = ProspectiveMemoryState(weight, initial_lr=1., meta_lr=0., control="write")
    state.log_write.fill_(30.)
    compiled = torch.compile(state.step, fullgraph=True, options={"triton.cudagraphs": False})
    compiled(weight, torch.tensor([[1., 0.]], device="cuda"),
             torch.tensor([[1., 0.]], device="cuda"))
    torch.testing.assert_close(weight, torch.tensor([[-1., 0.]], device="cuda"))
    assert state.write_trace[0, 0].item() < 0  # not rounded to zero at saturation
    for buffer in state.buffers():
        assert torch.isfinite(buffer).all()


def test_checkpoint_resume_and_zero_meta_control_preserve_predictions():
    weight = torch.nn.Parameter(torch.zeros((1, 2), device="cuda"))
    optimizer = ProspectiveMemory([weight], initial_lr=0.1, meta_lr=0.01)
    x = torch.tensor([[1., -0.5]], device="cuda")
    for target in (0.4, -0.2, 0.8):
        weight.grad = (((weight * x).sum() - target) * x).detach()
        optimizer.step(curvatures=[x.square()])
    resumed_weight = torch.nn.Parameter(weight.detach().clone())
    resumed = ProspectiveMemory([resumed_weight])
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for target in (-0.3, 1.1):
        for w, opt in ((weight, optimizer), (resumed_weight, resumed)):
            w.grad = (((w * x).sum() - target) * x).detach()
            opt.step(curvatures=[x.square()])
        torch.testing.assert_close(weight, resumed_weight, rtol=0, atol=0)

    frozen_weights = [torch.zeros((1, 2), device="cuda") for _ in range(2)]
    frozen_states = [ProspectiveMemoryState(w, initial_lr=0.1, meta_lr=0., control=control)
                     for w, control in zip(frozen_weights, ("full", "fixed"))]
    for target in (0.4, -0.2, 0.8):
        for w, state in zip(frozen_weights, frozen_states):
            state.step(w, ((w * x).sum() - target) * x, x.square())
        torch.testing.assert_close(frozen_weights[0], frozen_weights[1], rtol=0, atol=0)

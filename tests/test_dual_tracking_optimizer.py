import copy
import math

import pytest
import torch

from cleanrl.plasticity.dual_tracking_optimizer_v1 import DualTracking, DualTrackingState


@pytest.mark.parametrize("control", ["dual", "mirror", "innovation"])
def test_discounted_updates_match_scalar_reference(control):
    weight = torch.tensor([0.25], device="cuda")
    state = DualTrackingState(weight, lr=0.02, memory=10, control=control)
    gsum, qsum, mass, expected = 0.0, 0.0, 0.0, 0.25
    for gradient in (0.0, 2.0, 1.0, -3.0, 0.0, 0.5):
        surprise = gradient - (gsum / mass if mass else 0) if control == "innovation" else gradient
        gsum = math.exp(-0.1) * gsum + gradient
        qsum = math.exp(-0.1) * qsum + surprise * surprise
        mass = math.exp(-0.1) * mass + 1
        if qsum:
            expected = expected - 0.02 * gradient / math.sqrt(qsum) if control == "mirror" else 0.25 - 0.02 * gsum / math.sqrt(qsum)
        state.step(weight, torch.full_like(weight, gradient))
        torch.testing.assert_close(weight, torch.full_like(weight, expected), rtol=2e-6, atol=2e-7)


def test_later_contradiction_erases_dual_position_not_banked_mirror_steps():
    weights = [torch.zeros(1, device="cuda") for _ in range(2)]
    states = [DualTrackingState(weight, lr=0.1, control=control)
              for weight, control in zip(weights, ("dual", "mirror"))]
    for gradient in (1.0, -1.0):
        for state, weight in zip(states, weights):
            state.step(weight, torch.full_like(weight, gradient))
    torch.testing.assert_close(weights[0], torch.zeros_like(weights[0]), rtol=0, atol=0)
    torch.testing.assert_close(weights[1], torch.full_like(weights[1], -0.1 + 0.1 / math.sqrt(2)))


def test_compiled_zero_capture_and_batched_memory_settings_are_finite():
    weight = torch.zeros(2, 2, device="cuda")
    state = DualTrackingState(weight, lr=torch.tensor([[0.1], [0.2]], device="cuda"),
                              memory=torch.tensor([[0.0], [10.0]], device="cuda"))
    step = torch.compile(state.step, fullgraph=True)
    step(weight, torch.zeros_like(weight))
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
    step(weight, torch.tensor([[1.0, 0.0], [1.0, 0.0]], device="cuda"))
    step(weight, torch.zeros_like(weight))
    expected = torch.tensor([[-0.1, 0.0], [-0.2 * math.exp(-0.05), 0.0]], device="cuda")
    torch.testing.assert_close(weight, expected)


def test_checkpoint_resume_and_learning_rate_change_affect_next_position():
    parameter = torch.nn.Parameter(torch.tensor([0.1, -0.2], device="cuda"))
    optimizer = DualTracking([parameter], lr=0.02, memory=1000, control="innovation")
    for values in ((1.0, -0.2), (-0.3, 0.8)):
        parameter.grad = torch.tensor(values, device="cuda")
        optimizer.step()
    snapshot = copy.deepcopy(optimizer.state_dict())
    restored = torch.nn.Parameter(parameter.detach().clone())
    resumed = DualTracking([restored], lr=1.0)
    resumed.load_state_dict(snapshot)
    for values in ((0.2, 0.3), (-0.4, -0.1)):
        for weight, opt in ((parameter, optimizer), (restored, resumed)):
            opt.param_groups[0]["lr"] = 0.03
            weight.grad = torch.tensor(values, device="cuda")
            opt.step()
        torch.testing.assert_close(restored, parameter, rtol=0, atol=0)
    state = optimizer.state[parameter]
    torch.testing.assert_close(parameter, state["anchor"] - 0.03 * state["gradient_sum"] / state["square_sum"].sqrt())

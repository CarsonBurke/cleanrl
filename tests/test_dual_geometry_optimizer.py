import math

import pytest
import torch

from cleanrl.plasticity.dual_geometry_optimizer_v2 import DualGeometryState
from cleanrl.plasticity.dual_tracking_optimizer_v1 import DualTrackingState


@pytest.mark.parametrize("control,power", [("shared", 0.0), ("rms", 0.5), ("variance", 1.0), ("strong", 1.5)])
def test_metric_changes_coordinate_learning_without_changing_loss_units(control, power):
    weights = [torch.zeros(1, 3, device="cuda") for _ in range(2)]
    states = [DualGeometryState(weight, lr=0.1, control=control) for weight in weights]
    grad = torch.tensor([[1.0, 2.0, 0.0]], device="cuda")
    for weight, state, scale in zip(weights, states, (1.0, 1000.0)):
        state.step(weight, grad * scale)
    q = 5 / 3
    expected = torch.tensor([[-0.1 / (math.sqrt(q) * (1/q)**power),
                             -0.2 / (math.sqrt(q) * (4/q)**power), 0.0]], device="cuda")
    torch.testing.assert_close(weights[0], expected)
    torch.testing.assert_close(weights[1], expected)


def test_rms_recovers_lifetime_dual_trajectory():
    weights = [torch.zeros(2, 3, device="cuda") for _ in range(2)]
    geometry = DualGeometryState(weights[0], lr=0.01, control="rms")
    previous = DualTrackingState(weights[1], lr=0.01, memory=0, control="dual")
    for values in ((1, -2, 0), (-3, 0, 1), (0, 0, 0), (2, 1, -1)):
        grad = torch.tensor([values, values[::-1]], dtype=torch.float32, device="cuda")
        geometry.step(weights[0], grad)
        previous.step(weights[1], grad)
        torch.testing.assert_close(weights[0], weights[1], rtol=3e-6, atol=1e-8)


def test_compiled_zero_start_and_never_observed_coordinate_remain_at_anchor():
    weight = torch.tensor([[0.25, -0.3, 0.2]], device="cuda")
    state = DualGeometryState(weight, control="strong")
    step = torch.compile(state.step, fullgraph=True)
    step(weight, torch.zeros_like(weight))
    torch.testing.assert_close(weight, state.anchor, rtol=0, atol=0)
    step(weight, torch.tensor([[1.0, -2.0, 0.0]], device="cuda"))
    torch.testing.assert_close(weight[:, 2], state.anchor[:, 2], rtol=0, atol=0)
    step(weight, torch.tensor([[-1.0, 2.0, 0.0]], device="cuda"))
    torch.testing.assert_close(weight, state.anchor, rtol=0, atol=0)

import pytest
import torch

from cleanrl.plasticity.ppo_continuous_action_sphere_prospective_v1 import (
    ProspectiveUtilityStepper,
    UtilityForecaster,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_args():
    class Args:
        utility_lr = 1e-3
        utility_beta = 0.97
        utility_warmup = 32
        utility_floor = 0.25
        utility_ceiling = 2.0
        utility_temperature = 0.1
        utility_gain = 1.0
        utility_clip = 1.0

    return Args()


def test_forecaster_is_neutral_at_initialization():
    forecaster = UtilityForecaster().cuda()
    mean, logvar = forecaster(torch.randn(16, 6, device="cuda"))
    assert torch.equal(mean, torch.zeros_like(mean))
    assert torch.equal(logvar, torch.zeros_like(logvar))


def test_delayed_utility_sign_matches_future_gradient():
    layer = torch.nn.Linear(3, 2, device="cuda")
    stepper = ProspectiveUtilityStepper([("layer", layer)], make_args(), torch.device("cuda"))

    layer.weight.grad = torch.ones_like(layer.weight)
    layer.bias.grad = torch.ones_like(layer.bias)
    stepper.observe()
    stepper.stash()
    with torch.no_grad():
        layer.weight.add_(1.0)
        layer.bias.add_(1.0)
    stepper.apply_and_remember()

    layer.weight.grad = -torch.ones_like(layer.weight)
    layer.bias.grad = -torch.ones_like(layer.bias)
    stepper.observe()

    assert stepper.last_label_mean.item() > 0.99


def test_post_adam_gate_scales_realized_row_update():
    layer = torch.nn.Linear(3, 2, device="cuda")
    stepper = ProspectiveUtilityStepper([("layer", layer)], make_args(), torch.device("cuda"))
    layer.weight.grad = torch.ones_like(layer.weight)
    layer.bias.grad = torch.ones_like(layer.bias)
    stepper.observe()
    stepper.stash()
    with torch.no_grad():
        layer.weight.add_(1.0)
        layer.bias.add_(1.0)
    stepper.gates = [torch.full((2,), 0.5, device="cuda")]
    stepper.apply_and_remember()

    assert torch.equal(layer.weight, stepper.snapshots[0] + 0.5)
    bias_snapshot = stepper.bias_snapshots[0]
    assert bias_snapshot is not None
    assert torch.equal(layer.bias, bias_snapshot + 0.5)

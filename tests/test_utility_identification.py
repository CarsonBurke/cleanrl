"""CUDA contracts; execute through mlq, not as bare local GPU work."""

import pytest
import torch

from cleanrl.plasticity.utility_identification_v1 import Collector, CHUNK, SIZES, INPUTS, forward, linear_forward
from cleanrl.plasticity.utility_predictor_v1 import UtilityModel
from cleanrl.shared.runtime import configure_runtime


def test_finite_utility_stochastic_pairing_and_block_interaction():
    from cleanrl.plasticity.utility_target_audit_v1 import run_audits
    run_audits()


def test_historical_credit_requires_exact_sensitivity_transport():
    from cleanrl.plasticity.utility_credit_audit_v1 import run_audits
    run_audits()


@pytest.mark.parametrize("nonlinear", (False, True))
@torch.no_grad()
def test_compiled_collector_preserves_counterfactual_origin_and_next_example(nonlinear):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    rng = torch.Generator(device="cuda").manual_seed(1)
    p = sum(SIZES) if nonlinear else 16
    initial = torch.randn((2, p), device="cuda", generator=rng) * 0.05
    rates = torch.tensor([0.001, 0.002], device="cuda")
    collected = Collector(initial, rates, nonlinear)
    eager = Collector(initial, rates, nonlinear)
    collected.capture()
    inputs = torch.randn((CHUNK, INPUTS if nonlinear else p), generator=rng, device="cuda")
    targets = torch.randn((CHUNK, 2), generator=rng, device="cuda")
    collected.advance(inputs, targets)
    eager.inputs[0].copy_(inputs)
    eager.inputs[1].copy_(targets)
    predict = forward if nonlinear else linear_forward
    batched = torch.vmap(predict, in_dims=(0, None))
    explicit = []
    for offset in range(CHUNK):
        # Independent same-example loss calculation before updating persistent
        # snapshots. Wrong-origin, current-label, or mutation-alias bugs fail.
        before = batched(eager.previous, inputs[offset]).double()
        after = batched(eager.previous + eager.delta, inputs[offset]).double()
        target = targets[offset].double()
        explicit.append(0.5 * ((before - target).square() - (after - target).square()))
        eager.invoke(eager.transition, offset)
    torch.testing.assert_close(collected.output[1][:, :, 2], torch.stack(explicit).float(), rtol=1e-3, atol=2e-5)
    for actual, expected in zip(collected.output, eager.output):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=3e-5)
    torch.testing.assert_close(collected.previous, eager.previous, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(collected.weight, eager.weight, rtol=1e-4, atol=1e-6)


def test_compiled_utility_derivative_supervision_matches_same_function():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    model = UtilityModel(7).cuda()
    context = torch.randn(8, 7, device="cuda")
    actions = torch.rand(15, 4, device="cuda") * 2
    values, derivative = model.values_and_derivatives(context, actions)
    torch.testing.assert_close(model(context, torch.zeros(8, 4, device="cuda")),
                               torch.zeros(8, device="cuda"), rtol=0, atol=0)
    epsilon = 0.002
    central = []
    for axis in range(4):
        perturb = torch.zeros((8, 4), device="cuda")
        perturb[:, axis] = epsilon
        central.append((model(context, 1 + perturb) - model(context, 1 - perturb)) / (2 * epsilon))
    torch.testing.assert_close(derivative, torch.stack(central, 1), rtol=2e-3, atol=2e-5)

    def objective(x):
        utility, slope = model.values_and_derivatives(x, actions)
        return (utility - 1).square().mean() + (slope + 1).square().mean()

    expected_loss = objective(context)
    expected = torch.autograd.grad(expected_loss, tuple(model.parameters()))
    compiled = torch.compile(objective, fullgraph=True, dynamic=False)
    actual_loss = compiled(context)
    actual = torch.autograd.grad(actual_loss, tuple(model.parameters()))
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=2e-4, atol=1e-6)

"""CUDA numerical contracts; run through the machine-wide mlq daemon."""
import math

import torch

from cleanrl.plasticity.contextual_precision_v1 import ContextualPrecisionState
from cleanrl.shared.runtime import configure_runtime


@torch.no_grad()
def test_same_error_has_state_conditioned_effect_before_current_statistics_change():
    updates = []
    for context in (-1.0, 1.0):
        weight = torch.zeros((1, 1), device="cuda")
        state = ContextualPrecisionState(weight)
        state.logvar_bias.fill_(math.log(4))
        state.logvar_context.fill_(math.log(2))
        one = torch.ones_like(weight)
        state.step(weight, 2 * one, one, context * one, one[:, 0] * 2)
        updates.append(weight.item())
    # Variances2 and8 imply gains1/3 and1/9 for this scalar prediction.
    torch.testing.assert_close(torch.tensor(updates), torch.tensor([-2 / 3, -2 / 9]), rtol=1e-6, atol=1e-7)


@torch.no_grad()
def test_systematic_error_is_not_automatically_counted_as_noise():
    predicted = []
    for control in ("conditional", "uncentered_noise"):
        weight = torch.zeros((1, 1), device="cuda")
        state = ContextualPrecisionState(weight, control=control)
        state.mean_bias.fill_(3)
        one = torch.ones_like(weight)
        state.step(weight, 3 * one, one, 0 * one, one[:, 0] * 3)
        predicted.append(float(state.predictive(0 * one)[3]))
        # Both observations were scored against the PRIOR variance1.
        torch.testing.assert_close(state.predicted_variance_sum, torch.ones(1, device="cuda", dtype=torch.float64))
    assert predicted[0] < 1 < predicted[1]


@torch.no_grad()
def test_current_outlier_cannot_retroactively_inflate_its_own_variance():
    updates = []
    for residual in (2.0, 20.0):
        weight = torch.zeros((1, 1), device="cuda")
        state = ContextualPrecisionState(weight)
        one = torch.ones_like(weight)
        state.step(weight, residual * one, one, 0 * one, one[:, 0] * residual)
        updates.append(weight.clone())
    torch.testing.assert_close(updates[1], 10 * updates[0], rtol=0, atol=0)


@torch.no_grad()
def test_absent_participation_retains_learning_evidence_and_future_response():
    weights = [torch.zeros((1, 1), device="cuda") for _ in range(2)]
    states = [ContextualPrecisionState(weight) for weight in weights]
    one = torch.ones_like(weights[0])
    zero = torch.zeros_like(one)
    for state, weight in zip(states, weights):
        state.step(weight, one, one, one * 0.3, one[:, 0])
    for _ in range(100):
        states[1].step(weights[1], zero, zero, one * -0.7, one[:, 0] * 10)
    for state, weight in zip(states, weights):
        residual = weight[:, 0] - 1
        state.step(weight, residual[:, None], one, one * 0.3, residual)
    torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
    for name, value in states[0].buffers().items():
        torch.testing.assert_close(value, states[1].buffers()[name], rtol=0, atol=0)


@torch.no_grad()
def test_zero_residual_still_supplies_predictive_calibration_evidence():
    weight = torch.zeros((1, 1), device="cuda")
    state = ContextualPrecisionState(weight)
    zero = torch.zeros_like(weight)
    state.step(weight, zero, torch.ones_like(weight), zero, zero[:, 0])
    torch.testing.assert_close(weight, zero, rtol=0, atol=0)
    assert state.logvar_bias.item() < 0
    assert state.observation_count.item() == 1


def test_local_trace_matches_complete_scalar_linear_logstep_derivative():
    samples = ((2.0, 1.0), (-0.5, -2.0), (3.0, 0.3), (0.0, 10.0))
    log_alpha = torch.tensor(math.log(0.16), device="cuda", dtype=torch.float64, requires_grad=True)
    exact_weight = log_alpha.new_tensor(0.7)
    for x, y in samples:
        gain = log_alpha.exp() / (1 + log_alpha.exp() * x * x)
        exact_weight = exact_weight - gain * (exact_weight * x - y) * x
    exact_sensitivity = torch.autograd.grad(exact_weight, log_alpha)[0]
    weight = torch.full((1, 1), 0.7, device="cuda")
    state = ContextualPrecisionState(weight, control="no_precision", reference=weight.new_tensor(0.4), meta_step=0)
    for x, y in samples:
        residual = weight[:, 0] * x - y
        state.step(weight, residual[:, None] * x, torch.full_like(weight, x), torch.zeros_like(weight), residual)
    torch.testing.assert_close(weight.squeeze().double(), exact_weight.detach(), rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(state.sensitivity.squeeze().double(), exact_sensitivity, rtol=2e-6, atol=1e-7)


@torch.no_grad()
def test_linear_cuda_graph_matches_eager_updates_and_prequential_calibration():
    from cleanrl.plasticity.contextual_precision_benchmark_v1 import Args, Runner, configurations, CHUNK
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    args = Args(task="sparse")
    streams = [{"kind": "signal", "alpha": 1.0}, {"kind": "pure_noise", "alpha": 0.0}]
    configs, groups = configurations(streams)
    eager = Runner(args, 16, streams, 20000, 0, configs, groups)
    compiled = Runner(args, 16, streams, 20000, 0, configs, groups)
    compiled.capture()
    rng = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.rand((CHUNK, 16), generator=rng, device="cuda") < 0.2).float()
    y = torch.randn((CHUNK, 2), generator=rng, device="cuda")
    clean = torch.zeros_like(y)
    compiled.advance(x, y, clean)
    for offset in range(CHUNK):
        eager.eager_step(x[offset], y[offset], clean[offset])
    candidate = slice(groups["conditional"].start, None)
    torch.testing.assert_close(compiled.weight[candidate], eager.weight[candidate], rtol=5e-4, atol=2e-5)
    for name in compiled.controllers:
        torch.testing.assert_close(compiled.controllers[name].nll_sum, eager.controllers[name].nll_sum, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(compiled.controllers[name].predicted_variance_sum,
                                   eager.controllers[name].predicted_variance_sum, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_nonlinear_matched_state_transitions_preserve_next_prediction():
    from cleanrl.plasticity.contextual_precision_nonlinear_v1 import (
        Args, Experiment, PARAMETERS, batched_forward,
    )
    from cleanrl.plasticity.contextual_precision_benchmark_v1 import configurations
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    args = Args()
    rng = torch.Generator(device="cuda").manual_seed(1)
    initial = torch.randn(PARAMETERS, generator=rng, device="cuda") * 0.05
    reference = torch.full_like(initial, 1 / math.sqrt(32))
    streams = [{"kind": "signal", "alpha": 1.0}, {"kind": "pure_noise", "alpha": 0.0}]
    configs, groups = configurations(streams)
    eager = Experiment(args, initial, reference, configs, groups)
    compiled = Experiment(args, initial, reference, configs, groups)
    candidate = slice(groups["conditional"].start, None)
    for _ in range(100):
        x = torch.randn(32, generator=rng, device="cuda")
        target = torch.randn(2, generator=rng, device="cuda")
        clean = torch.zeros_like(target)
        for actual, expected in zip(eager.states, compiled.states):
            actual.copy_(expected)
        eager.eager_step(x, target, clean)
        compiled.compiled(x, target, clean)
        next_x = torch.randn(32, generator=rng, device="cuda")
        expected = batched_forward(eager.weight[candidate], next_x)
        actual = batched_forward(compiled.weight[candidate], next_x)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
        for name in compiled.controllers:
            torch.testing.assert_close(compiled.controllers[name].nll_sum,
                                       eager.controllers[name].nll_sum, rtol=1e-5, atol=1e-5)

"""Observable FP32 optimizer contracts; CUDA execution must be queued via mlq."""

import math
from types import SimpleNamespace

import pytest
import torch

from cleanrl.plasticity import predictive_transport_stream_v2 as transport
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def learner(method="predictive", grid=((0.007, 0.9), (0.025, 0.99999))):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    a = SimpleNamespace(graph_steps=2)
    initial = [torch.tensor(value, device="cuda") for value in (
        [[0.7, -0.2, 0.1], [-0.3, 0.6, -0.2]],
        [[0.8, -0.4, -0.1], [0.2, 0.5, 0.3]],
        [[0.4, -0.5, 0.2]])]
    xs = torch.tensor([[-1.2, 0.3], [0.4, -0.8], [1.1, 0.7], [-0.2, -0.6],
                       [0.8, 0.1], [-0.7, 1.0], [0.3, 0.9], [-0.4, 0.2]], device="cuda")
    ys = torch.tensor([0.7, -0.4, 1.2, 0.9, -0.8, 0.5, -0.7, 0.3], device="cuda")
    return transport.Learner(method, grid, initial, a, xs, ys, ys * 0.6, torch.ones_like(ys))


def flatten(tensors):
    return torch.cat([tensor.flatten(1) for tensor in tensors], -1)


def network(theta, x):
    w1, w2, w3 = theta[:6].reshape(2, 3), theta[6:12].reshape(2, 3), theta[12:]
    h1 = torch.tanh(w1[:, :2] @ x + w1[:, 2])
    h2 = torch.tanh(w2[:, :2] @ h1 + w2[:, 2])
    return w3[:2] @ h2 + w3[2]


def state(theta, x):
    with torch.enable_grad():
        return network(theta, x), torch.autograd.functional.jacobian(lambda w: network(w, x), theta)


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_fp32_trajectory_matches_independent_autograd_and_dense_proximal_solve(method):
    model = learner(method)
    theta = flatten(model.weights).double()
    previous = theta.clone()
    m, v = torch.zeros_like(theta), torch.zeros_like(theta)
    error = torch.zeros(len(theta), device="cuda", dtype=torch.float64)
    for t in range(1, len(model.xs) + 1):
        before = theta.clone()
        for k, (lr, beta) in enumerate(model.grid):
            # Match the actually represented FP32 hyperparameter, not its decimal spelling.
            beta = float(torch.tensor(beta, dtype=torch.float32))
            x, y = model.xs[t - 1].double(), model.ys[t - 1].double()
            f, j = state(theta[k], x)
            previous_f = network(previous[k], x)
            residual = f - y
            if method == "robust":
                score = residual / torch.sqrt(1 + residual.square())
                old_residual = previous_f - y
                old_score = old_residual / torch.sqrt(1 + old_residual.square())
                g, c = score * j, (score - old_score) * j
            else:
                g = residual * j
                change = j @ (theta[k] - previous[k]) if method == "tangent" else f - previous_f
                c = torch.zeros_like(j) if method == "adam" else change * j
            old_mass = -math.expm1(math.log(beta) * (t - 1)) if beta else 0.0
            mass = -math.expm1(math.log(beta) * t) if beta else 1.0
            m[k] = beta * m[k] + (1 - beta) * g + beta * old_mass * c
            v[k] = 0.999 * v[k] + 0.001 * g.square()
            denom = (v[k] / (-math.expm1(math.log(0.999) * t))).sqrt() + 1e-8
            if method == "implicit":
                # An independent tiny dense solve verifies the vector-only production formula.
                hessian = torch.diag(denom / lr) + torch.outer(j, j)
                theta[k] += torch.linalg.solve(hessian, -m[k] / mass)
            else:
                theta[k] -= lr * (m[k] / mass) / denom
            error[k] += (f - model.clean[t - 1].double()).square()
        previous = before
        model.update()
        for actual, expected in ((flatten(model.weights), theta), (flatten(model.m), m),
                                 (flatten(model.v), v), (model.error, error)):
            torch.testing.assert_close(actual, expected.float(), rtol=3e-4, atol=8e-6)
        if method != "adam":
            torch.testing.assert_close(flatten(model.previous_weights), previous.float(), rtol=3e-4, atol=8e-6)
        assert model.index.item() == model.steps.item() == t
        assert all(tensor.dtype == torch.float32 for tensor in model.mutable if tensor.is_floating_point())


@pytest.mark.parametrize("method", ["predictive", "tangent"])
@pytest.mark.parametrize("beta", [0.0, 0.99999])
@torch.no_grad()
def test_transport_first_observation_and_zero_beta_reduce_to_adam(method, beta):
    actual, adam = learner(method, ((0.01, beta),)), learner("adam", ((0.01, beta),))
    for _ in range(len(actual.xs) if beta == 0 else 1):
        actual.update()
        adam.update()
        for left, right in zip([*actual.weights, *actual.m, *actual.v], [*adam.weights, *adam.m, *adam.v]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_reporting_oracles_and_future_observations_do_not_affect_current_learning(method):
    actual, changed = learner(method), learner(method)
    changed.clean.add_(100)
    changed.xs[1:].mul_(-3)
    changed.ys[1:].add_(100)
    # The constructor must not retain conditional noise as an optimizer input.
    noisy = transport.Learner(method, actual.grid, [w[0].clone() for w in actual.weights], actual.a,
                              actual.xs.clone(), actual.ys.clone(), actual.clean.clone(),
                              torch.full_like(actual.ys, 10000))
    for model in (actual, changed, noisy):
        model.update()
    for model in (changed, noisy):
        for left, right in zip([*actual.weights, *actual.m, *actual.v], [*model.weights, *model.m, *model.v]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not torch.equal(actual.error, changed.error)


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_nonfinite_candidate_cannot_change_neighbor_trajectory(method):
    batch = learner(method)
    single = learner(method, (batch.grid[1],))
    batch.weights[0][0, 0, 0] = float("nan")
    for _ in range(3):
        batch.update()
        single.update()
    assert batch.finite_candidates.tolist() == [False, True]
    for left, right in zip([*batch.weights, *batch.m, *batch.v], [*single.weights, *single.m, *single.v]):
        torch.testing.assert_close(left[1:], right, rtol=0, atol=0)


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_capture_preserves_nonzero_state_and_replays_owned_weight_history(method):
    torch.compiler.reset()
    captured, eager = learner(method), learner(method)
    captured.update()
    captured.update()
    eager.update()
    eager.update()
    before = captured.snapshot()
    graph, parity = captured.capture()
    try:
        assert math.isfinite(parity)
        for actual, expected in zip(captured.mutable, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for _ in range(3):
            graph.replay()
            for _ in range(captured.capture_steps):
                eager.update()
        torch.cuda.synchronize()
        for actual, expected in zip(captured.mutable, eager.mutable):
            torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-5)
        final = captured.snapshot()
        captured.restore(before)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(captured.mutable, final):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert captured.index.item() == len(captured.xs)
    finally:
        del graph
        torch.compiler.reset()


@torch.no_grad()
def test_capture_restores_nonfinite_candidates_without_excluding_finite_neighbors():
    torch.compiler.reset()
    model = learner("implicit")
    model.weights[0][0, 0, 0] = float("nan")
    before = model.snapshot()
    graph, parity = model.capture()
    try:
        assert math.isfinite(parity)
        assert model.capture_failed_candidates == [True, False]
        for actual, expected in zip(model.mutable, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        graph.replay()
        torch.cuda.synchronize()
        assert model.finite_candidates.tolist() == [False, True]
    finally:
        del graph
        torch.compiler.reset()


@pytest.mark.parametrize("failure_side", ["compiled", "eager"])
@torch.no_grad()
def test_capture_rejects_one_sided_nonfinite_transition_and_restores_state(monkeypatch, failure_side):
    torch.compiler.reset()
    model = learner("predictive")
    before = model.snapshot()
    real_compile, real_update = torch.compile, model.update
    calls = 0

    def compile_with_fault(fn, **kwargs):
        compiled = real_compile(fn, **kwargs)

        def run():
            nonlocal calls
            output = compiled()
            calls += 1
            if calls == 3 and failure_side == "compiled":
                output[0][0, 0, 0] = float("nan")
                model.finite_candidates[0] = False
            return output

        return run

    def eager_with_fault():
        real_update()
        if failure_side == "eager":
            model.weights[0][0, 0, 0] = float("nan")
            model.finite_candidates[0] = False

    monkeypatch.setattr(torch, "compile", compile_with_fault)
    monkeypatch.setattr(model, "update", eager_with_fault)
    try:
        with pytest.raises(AssertionError):
            model.capture()
        for actual, expected in zip(model.mutable, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert model.capture_failed_candidates == [False, False]
    finally:
        torch.compiler.reset()

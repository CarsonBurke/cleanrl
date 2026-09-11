"""Behavioral CUDA contracts for matrix-free transport; run only through mlq."""

import pytest
import torch

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import predictive_transport_stream_v1 as transport
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def learner(method="uncertainty", grid=((0.007, 0.9), (0.025, 0.99))):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    a = reference.Args(input_dim=2, hidden=2, samples=8, graph_steps=2, noise_rate=0.2)
    initial = [torch.tensor(value, device="cuda") for value in (
        [[0.7, -0.2, 0.1], [-0.3, 0.6, -0.2]],
        [[0.8, -0.4, -0.1], [0.2, 0.5, 0.3]],
        [[0.4, -0.5, 0.2]])]
    xs = torch.tensor([[-1.2, 0.3], [0.4, -0.8], [1.1, 0.7], [-0.2, -0.6],
                       [0.8, 0.1], [-0.7, 1.0], [0.3, 0.9], [-0.4, 0.2]], device="cuda")
    ys = torch.tensor([0.7, -0.4, 1.2, 0.9, -0.8, 0.5, -0.7, 0.3], device="cuda")
    clean = ys * 0.6
    noise = torch.linspace(0.2, 1.0, len(ys), device="cuda")
    return transport.Learner(method, grid, initial, a, xs, ys, clean, noise)


def flatten(tensors):
    return torch.cat([tensor.flatten(1) for tensor in tensors], -1)


def independent_network(theta, x):
    w1, w2, w3 = theta[:6].reshape(2, 3), theta[6:12].reshape(2, 3), theta[12:]
    h1 = torch.tanh(w1[:, :2] @ x + w1[:, 2])
    h2 = torch.tanh(w2[:, :2] @ h1 + w2[:, 2])
    return w3[:2] @ h2 + w3[2]


def independent_state(theta, x):
    with torch.enable_grad():
        prediction = independent_network(theta, x)
        jacobian = torch.autograd.functional.jacobian(lambda value: independent_network(value, x), theta)
    return prediction, jacobian


def layer_rows(flat):
    return (flat[:6].reshape(2, 3), flat[6:12].reshape(2, 3), flat[12:].reshape(1, 3))


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_nonlinear_trajectory_matches_independent_autograd_adam(method):
    model = learner(method)
    theta = flatten(model.weights).double()
    previous = theta.clone()
    first, second = torch.zeros_like(theta), torch.zeros_like(theta)
    history = [torch.ones(len(theta), rows, device="cuda", dtype=torch.float64) for rows in (2, 2, 1)]
    error = torch.zeros(len(theta), device="cuda", dtype=torch.float64)
    for t in range(1, len(model.xs) + 1):
        before = theta.clone()
        expected_q, expected_applied = [[] for _ in range(3)], [[] for _ in range(3)]
        for k, (lr, beta) in enumerate(model.grid):
            x, y = model.xs[t - 1].double(), model.ys[t - 1].double()
            f, j = independent_state(theta[k], x)
            old_f, old_j = independent_state(previous[k], x)
            # Directly verify all exact manual Jacobians as well as the trajectory.
            manual_f, inputs, sensitivities = reference.sample_state(
                [w[k:k + 1] for w in model.weights], model.xs[t - 1])
            manual_j = torch.cat([(inp[:, None] * sensitivity[..., None]).flatten()
                                  for inp, sensitivity in zip(inputs, sensitivities)])
            torch.testing.assert_close(manual_f[0], f.float(), rtol=2e-4, atol=5e-6)
            torch.testing.assert_close(manual_j, j.float(), rtol=2e-4, atol=5e-6)
            g = (f - y) * j
            g_previous = (old_f - y) * old_j
            corrections = []
            for layer, (row_j, row_old_j) in enumerate(zip(layer_rows(j), layer_rows(old_j))):
                energy = row_j.square().sum(-1)
                denominator = energy + (row_j - row_old_j).square().sum(-1)
                q = torch.where(denominator == 0, torch.ones_like(denominator), energy / denominator)
                if method == "adam":
                    q = torch.ones_like(q)
                applied = (q if method == "uncertainty" else q.mean().expand_as(q) if method == "shared"
                           else history[layer][k].clone() if method == "history" else torch.ones_like(q))
                expected_q[layer].append(q)
                expected_applied[layer].append(applied)
                base = (layer_rows(g - g_previous)[layer] if method == "full"
                        else (f - old_f) * row_j)
                corrections.append((base * applied[:, None]).flatten())
                history[layer][k] += model.a.noise_rate * (q - history[layer][k])
            correction = torch.zeros_like(g) if method == "adam" else torch.cat(corrections)
            first[k] = beta * first[k] + (1 - beta) * g + beta * (1 - beta ** (t - 1)) * correction
            second[k] = 0.999 * second[k] + 0.001 * g.square()
            theta[k] -= lr * (first[k] / (1 - beta ** t)) / ((second[k] / (1 - 0.999 ** t)).sqrt() + 1e-8)
            error[k] += (f - model.clean[t - 1].double()).square()
        previous = before
        model.update()
        for actual, expected in ((flatten(model.weights), theta), (flatten(model.previous_weights), previous),
                                 (flatten(model.m), first), (flatten(model.v), second), (model.error, error)):
            torch.testing.assert_close(actual, expected.float(), rtol=3e-4, atol=8e-6)
        for layer in range(3):
            for actual, expected in ((model.q_current[layer], torch.stack(expected_q[layer])),
                                     (model.q_applied[layer], torch.stack(expected_applied[layer])),
                                     (model.q_history[layer], history[layer])):
                torch.testing.assert_close(actual, expected.float(), rtol=3e-4, atol=8e-6)
        assert model.index.item() == model.steps.item() == t


@pytest.mark.parametrize("beta", [0.0, 0.99])
@torch.no_grad()
def test_zero_beta_and_first_observation_are_adam(beta):
    models = [learner(method, ((0.01, beta),)) for method in transport.METHODS]
    for _ in range(len(models[0].xs) if beta == 0 else 1):
        for model in models:
            model.update()
        for model in models[1:]:
            for actual, expected in zip(model.weights + model.m + model.v,
                                        models[0].weights + models[0].m + models[0].v):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("beta", [0.9, 0.999, 0.99999])
@torch.no_grad()
def test_exact_gradient_transport_preserves_biased_momentum_mass(beta):
    w = torch.zeros(1, 1, 2, device="cuda")
    m, v = torch.zeros_like(w), torch.zeros_like(w)
    previous = torch.zeros_like(w)
    second = torch.zeros_like(w)
    b = torch.full((1, 1, 1), beta, device="cuda")
    for t, values in enumerate(([1.0, -0.5], [-0.7, 1.1], [0.2, 0.8], [-1.3, -0.2]), 1):
        gradient = torch.tensor(values, device="cuda").reshape_as(w)
        transport.transported_adam_update(w, m, v, gradient, gradient - previous, 0.01, b,
                                           torch.tensor(float(t), device="cuda"))
        second = 0.999 * second + 0.001 * gradient.square()
        mass = (-torch.expm1(b.double().log() * t)).float()
        torch.testing.assert_close(m / mass, gradient, rtol=4e-5, atol=4e-5)
        torch.testing.assert_close(v, second, rtol=1e-6, atol=1e-8)
        previous = gradient


@torch.no_grad()
def test_fixed_jacobian_linear_transport_variants_cancel_same_sample_noise():
    j = torch.tensor([[[0.4, -0.7, 1.0]]], device="cuda")
    f, old_f = torch.tensor([0.7], device="cuda"), torch.tensor([-0.3], device="cuda")
    history = torch.ones(1, 1, device="cuda")
    results = []
    for method in transport.METHODS[1:]:
        for y in (torch.tensor([0.2], device="cuda"), torch.tensor([4.0], device="cuda")):
            correction, q, applied, _, difference = transport.transport_terms(method, f, old_f, y, j, j, history)
            torch.testing.assert_close(correction, (f - old_f)[:, None, None] * j, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(q, torch.ones_like(q), rtol=0, atol=0)
            torch.testing.assert_close(applied, q, rtol=0, atol=0)
            torch.testing.assert_close(difference, torch.zeros_like(difference), rtol=0, atol=0)
            results.append(correction)
    for correction in results[1:]:
        torch.testing.assert_close(correction, results[0], rtol=1e-6, atol=1e-6)
    states = [[torch.zeros_like(j) for _ in range(4)] for _ in transport.METHODS[1:]]
    beta = torch.full((1, 1, 1), 0.99, device="cuda")
    for t, target in enumerate((0.5, -0.7, 1.3, -0.2), 1):
        y = torch.tensor([target], device="cuda")
        for method, (weight, previous, m, v) in zip(transport.METHODS[1:], states):
            f, old_f = (weight * j).sum((-1, -2)), (previous * j).sum((-1, -2))
            correction, _, _, _, _ = transport.transport_terms(method, f, old_f, y, j, j, history)
            previous.copy_(weight)
            transport.transported_adam_update(weight, m, v, (f - y)[:, None, None] * j,
                                               correction, 0.01, beta, torch.tensor(float(t), device="cuda"))
        for state in states[1:]:
            for actual, expected in zip(state, states[0]):
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("method", ["predictive", "uncertainty", "shared", "history"])
@torch.no_grad()
def test_current_label_cannot_change_prediction_transport_or_stability(method):
    left, right = learner(method), learner(method)
    left.update()
    right.update()
    right.ys[1] += 11
    left.update()
    right.update()
    for a, b in zip(left.q_current + left.q_history + left.q_applied + [left.correction_norm_sum],
                    right.q_current + right.q_history + right.q_applied + [right.correction_norm_sum]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(flatten(left.weights), flatten(right.weights))


@torch.no_grad()
def test_history_uses_prior_level_and_shared_uses_exact_layer_row_mean():
    history, shared = learner("history"), learner("shared")
    for _ in range(3):
        prior = [q.clone() for q in history.q_history]
        history.update()
        shared.update()
        for layer in range(3):
            torch.testing.assert_close(history.q_applied[layer], prior[layer], rtol=0, atol=0)
            expected_next = prior[layer] + history.a.noise_rate * (history.q_current[layer] - prior[layer])
            torch.testing.assert_close(history.q_history[layer], expected_next)
            expected_shared = shared.q_current[layer].mean(-1, keepdim=True).expand_as(shared.q_current[layer])
            torch.testing.assert_close(shared.q_applied[layer], expected_shared, rtol=0, atol=0)
    assert any(not torch.equal(q, applied) for q, applied in zip(history.q_current, history.q_applied))
    assert shared.q_current[0].std(-1, unbiased=False).max().item() > 0


@torch.no_grad()
def test_zero_jacobian_row_is_stable_and_shared_mean_counts_it():
    j = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], device="cuda")
    old = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]]], device="cuda")
    f, old_f, y = [torch.tensor([v], device="cuda") for v in (1.0, 0.0, 3.0)]
    _, q, applied, _, _ = transport.transport_terms("shared", f, old_f, y, j, old, torch.ones(1, 3, device="cuda"))
    torch.testing.assert_close(q, torch.tensor([[1.0, 0.5, 0.5]], device="cuda"), rtol=0, atol=0)
    torch.testing.assert_close(applied, torch.full_like(q, 2 / 3), rtol=0, atol=0)


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_clean_labels_noise_oracle_and_future_inputs_do_not_enter_updates(method):
    original, changed = learner(method), learner(method)
    changed.clean.add_(19)
    changed.xs[3:].mul_(-4)
    changed.ys[3:].add_(7)
    # The constructor must not even consult this privileged tensor.
    rebuilt = transport.Learner(method, original.grid, [w[0] for w in original.weights], original.a,
                                 original.xs, original.ys, original.clean,
                                 torch.full_like(original.ys, float("nan")))
    for _ in range(3):
        original.update()
        changed.update()
        rebuilt.update()
    for other in (changed, rebuilt):
        for a, b in zip(original.weights + original.m + original.v + original.q_history,
                        other.weights + other.m + other.v + other.q_history):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(original.error, changed.error)


@torch.no_grad()
def test_nonfinite_candidate_cannot_contaminate_independent_finite_config():
    batch, single = learner(), learner(grid=((0.025, 0.99),))
    batch.weights[0][0, 0, 0] = float("nan")
    for _ in range(3):
        batch.update()
        single.update()
        assert batch.finite_candidates.tolist() == [False, True]
        for actual, expected in zip(batch.weights + batch.m + batch.v + batch.q_history,
                                    single.weights + single.m + single.v + single.q_history):
            torch.testing.assert_close(actual[1], expected[0], rtol=2e-5, atol=2e-6)
    assert batch.diagnostics()["correction_l2_mean"][0] is None


@torch.no_grad()
def test_capture_restores_all_nonzero_state_and_replays_without_consuming_warmup():
    torch.compiler.reset()
    captured, eager = learner("history"), learner("history")
    for _ in range(2):
        captured.update()
        eager.update()
    before = captured.snapshot()
    graph, _ = captured.capture()
    try:
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
        assert captured.index.item() == captured.steps.item() == len(captured.xs)
    finally:
        del graph
        torch.compiler.reset()


@torch.no_grad()
def test_capture_failure_is_candidate_local_and_still_rolls_back():
    torch.compiler.reset()
    model = learner("uncertainty")
    model.weights[0][0, 0, 0] = float("nan")
    before = model.snapshot()
    graph, parity = model.capture()
    try:
        assert model.capture_failed_candidates == [True, False]
        assert 0 <= parity < float("inf")
        for actual, expected in zip(model.mutable, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        graph.replay()
        torch.cuda.synchronize()
        assert model.finite_candidates.tolist() == [False, True]
    finally:
        del graph
        torch.compiler.reset()

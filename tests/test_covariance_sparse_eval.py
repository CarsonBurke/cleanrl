"""CUDA contract checks, not reduced-horizon learning evidence. Execute via mlq."""

from itertools import product

import pytest
import torch

from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
from cleanrl.plasticity import network_bayes_stream_v2 as v2
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def fixture_stream(method="network", *, diffusion=0.0, noise_rate=0.0):
    a = sparse.Args(input_dim=3, steps=4, graph_steps=3, noise_rate=noise_rate)
    x = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]], device="cuda", dtype=torch.bool)
    y = torch.tensor([1.2, -.7, .4, 1.8], device="cuda")
    learner = sparse.LinearLearner(method, [1.0 if method == "network" else .01], a, x, y,
                                   diffusion=diffusion)
    return learner, x, y


@torch.no_grad()
def test_linear_posterior_matches_closed_form_gaussian_at_every_observation():
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    learner, x, y = fixture_stream()
    precision = 3 * torch.eye(3, dtype=torch.float64, device="cuda")
    information = torch.zeros(3, dtype=torch.float64, device="cuda")
    previous_mean = information.clone()
    for xi, yi in zip(x.double(), y.double()):
        expected_prediction = xi @ previous_mean
        precision += xi[:, None] * xi[None, :]
        information += xi * yi
        posterior = torch.linalg.inv(precision)
        previous_mean = torch.linalg.solve(precision, information)
        learner.update()
        torch.testing.assert_close(learner.prediction[0].double(), expected_prediction, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(learner.weights[0][0, 0].double(), previous_mean, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(learner.cov[0].double(), posterior, rtol=2e-6, atol=2e-7)


@torch.no_grad()
def test_linear_specialization_matches_frozen_v2_with_process_noise():
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    learner, x, y = fixture_stream(diffusion=1e-5, noise_rate=.001)
    a = v2.Args(diffusion=1e-5, noise_rate=.001)
    reference = v2.Learner("network", [1.0], [torch.zeros((1, 3), device="cuda")], a,
                           x, y, torch.zeros_like(y), torch.ones_like(y))
    for xi, yi in zip(x.float(), y):
        predicted = (reference.weights[0][0, 0] * xi).sum().reshape(1)
        residual = predicted - yi
        reference.network_update((xi.reshape(1, -1),), (torch.ones((1, 1), device="cuda"),),
                                 residual, reference.noise)
        reference.noise.lerp_(residual.square(), a.noise_rate)
        learner.update()
        torch.testing.assert_close(learner.weights[0], reference.weights[0], rtol=0, atol=0)
        torch.testing.assert_close(learner.cov, reference.cov, rtol=0, atol=0)
        torch.testing.assert_close(learner.noise, reference.noise, rtol=0, atol=0)


@torch.no_grad()
def test_null_has_no_teacher_signal_or_extra_noise_and_move_is_exact():
    a = sparse.Args(input_dim=4, steps=20, noise_sigma=1.0, feature_prob=.5)
    x, noise = sparse.draw_stream(a, torch.device("cuda"))
    stationary, stationary_clean = sparse.teacher_labels(x, noise, "stationary")
    null, null_clean = sparse.teacher_labels(x, noise, "null")
    changed, changed_clean = sparse.teacher_labels(x, noise, "change")
    torch.testing.assert_close(null_clean, torch.zeros_like(noise), rtol=0, atol=0)
    torch.testing.assert_close(null, noise, rtol=0, atol=0)
    torch.testing.assert_close(stationary_clean, x[:, 0].float(), rtol=0, atol=0)
    torch.testing.assert_close(changed_clean[:10], x[:10, 0].float(), rtol=0, atol=0)
    torch.testing.assert_close(changed_clean[10:], x[10:, 1].float(), rtol=0, atol=0)
    torch.testing.assert_close(stationary - stationary_clean, null, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(changed - changed_clean, null, rtol=1e-6, atol=1e-7)
    # The null teacher cannot inject spikes or a hidden coefficient, even when
    # the caller supplies no observation noise.
    silent, _ = sparse.teacher_labels(x, torch.zeros_like(noise), "null")
    torch.testing.assert_close(silent, torch.zeros_like(noise), rtol=0, atol=0)


@pytest.mark.parametrize("support_index", [None, 1])
@torch.no_grad()
def test_exact_risk_matches_exhaustive_uncentered_bernoulli_decomposition(support_index):
    p = .2
    x = torch.tensor(list(product((0., 1.), repeat=3)), device="cuda", dtype=torch.float64)
    mass = torch.where(x.bool(), torch.full_like(x, p), torch.full_like(x, 1 - p)).prod(-1)
    weight = torch.tensor([[.7, .4, -.5], [-.3, 1.5, .2]], device="cuda", dtype=torch.float64)
    support = torch.zeros(3, device="cuda", dtype=torch.float64)
    if support_index is not None:
        support[support_index] = 1
    risk = sparse.exact_risk(weight, support, p, stale_index=0)
    signal = x @ ((weight - support) * (support != 0)).T
    junk = x @ (weight * (support == 0)).T
    stale = x[:, :1] * weight[:, 0]
    remaining = junk - stale
    error = x @ (weight - support).T
    average = lambda values: (mass[:, None] * values).sum(0)
    expected = {
        "clean_mse": average(error.square()),
        "signal_reconstruction_mse": average(signal.square()),
        "distractor_leakage_mse": average(junk.square()),
        "distractor_mean": average(junk),
        "distractor_variance": average((junk - average(junk)).square()),
        "signal_distractor_cross": average(2 * signal * junk),
        "stale_support_leakage_mse": average(stale.square()),
        "remaining_distractor_leakage_mse": average(remaining.square()),
        "stale_remaining_cross": average(2 * stale * remaining),
        "prediction_energy": average((x @ weight.T).square()),
    }
    for name, value in expected.items():
        torch.testing.assert_close(risk[name], value, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(risk["clean_mse"], risk["signal_reconstruction_mse"]
                               + risk["distractor_variance"] + risk["distractor_mean_squared"]
                               + risk["signal_distractor_cross"], rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(risk["distractor_leakage_mse"], risk["stale_support_leakage_mse"]
                               + risk["remaining_distractor_leakage_mse"] + risk["stale_remaining_cross"],
                               rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("method", ["network", "adam", "softhinge"])
@torch.no_grad()
def test_current_label_cannot_change_reported_prediction(method):
    first, _, _ = fixture_stream(method)
    other, _, _ = fixture_stream(method)
    first.weights[0].fill_(.2)
    other.weights[0].fill_(.2)
    other.ys = other.ys.clone()
    other.ys[0] = -100
    first.update()
    other.update()
    expected = torch.tensor([.4], device="cuda")
    torch.testing.assert_close(first.prediction, expected, rtol=0, atol=0)
    torch.testing.assert_close(other.prediction, expected, rtol=0, atol=0)
    torch.testing.assert_close(first.moments[:, 1], other.moments[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(first.moments[:, 2], (1.2 * expected).double(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(other.moments[:, 2], (-100 * expected).double(), rtol=0, atol=0)


@torch.no_grad()
def test_softhinge_uses_corrected_signed_past_evidence_formula():
    learner, x, y = fixture_stream("softhinge")
    learner.running_sum.fill_(8)
    learner.running_sq.fill_(2)
    certainty = torch.nn.functional.softplus(torch.tensor(24 * (1 - 16 / 32), device="cuda")) / 24
    gradient = -y[0] * x[0].float()
    expected = -.01 * gradient / (gradient.abs() + 1e-5) * certainty
    learner.update()
    torch.testing.assert_close(learner.weights[0][0, 0], expected, rtol=3e-5, atol=1e-8)


@pytest.mark.parametrize("method", ["network", "adam", "softhinge"])
@torch.no_grad()
def test_compiled_capture_resets_all_state_and_exact_tail_matches_eager(method):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    learner, _, _ = fixture_stream(method, noise_rate=.001)
    reference, _, _ = fixture_stream(method, noise_rate=.001)
    initial = learner.snapshot()
    graphs = learner.capture()
    for actual, expected in zip(learner.mutable, initial):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    blocks, tail = divmod(4, learner.capture_steps)
    for _ in range(blocks):
        graphs[learner.capture_steps].replay()
    for _ in range(tail):
        graphs[1].replay()
    for _ in range(4):
        reference.update()
    torch.cuda.synchronize()
    assert learner.index.item() == 4
    for actual, expected in zip(learner.mutable, reference.mutable):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)

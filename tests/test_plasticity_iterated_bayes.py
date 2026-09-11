"""CUDA iterated-EKF contracts; execute only through the mlq queue."""

import pytest
import torch

from cleanrl.plasticity import iterated_bayes_stream_v4 as iterated
from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def learner(iterations=2, known_noise=True, frozen=False):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    a = reference.Args(input_dim=2, hidden=2, samples=6, graph_steps=2,
                       diffusion=0.025, noise_rate=0.2, known_noise=known_noise)
    initial = [torch.tensor(value, device="cuda") for value in (
        [[0.7, -0.2, 0.1], [-0.3, 0.6, -0.2]],
        [[0.8, -0.4, -0.1], [0.2, 0.5, 0.3]],
        [[0.4, -0.5, 0.2]])]
    xs = torch.tensor([[-1.2, 0.3], [0.4, -0.8], [1.1, 0.7],
                       [-0.2, -0.6], [0.8, 0.1], [-0.7, 1.0]], device="cuda")
    ys = torch.tensor([0.7, -0.4, 1.2, 0.9, -0.8, 0.5], device="cuda")
    clean = torch.tensor([0.5, -0.1, 0.8, 0.6, -0.3, 0.2], device="cuda")
    noise = torch.tensor([0.4, 1.2, 0.7, 0.9, 0.6, 0.3], device="cuda")
    constructor, first = (reference.Learner, "network") if frozen else (iterated.IteratedLearner, iterations)
    model = constructor(first, (0.2, 0.7), initial, a, xs, ys, clean, noise)
    # Correlated, nonuniform priors and Q expose diagonal and isotropic shortcuts.
    coordinate = torch.linspace(0.4, 1.7, 15, device="cuda")
    model.cov.diagonal(dim1=-2, dim2=-1).mul_(coordinate)
    model.cov.add_(0.01 * model.scale[:, None, None] * coordinate[None, :, None] * coordinate[None, None, :])
    model.process.mul_(coordinate.flip(0))
    return model


def flat_weights(model):
    return torch.cat([w.flatten(1) for w in model.weights], -1)


def independent_network(theta, x):
    """Independent float64 network, differentiated by autograd, not sample_state."""
    w1, w2, w3 = theta[:6].reshape(2, 3), theta[6:12].reshape(2, 3), theta[12:]
    h1 = torch.tanh(w1[:, :2] @ x + w1[:, 2])
    h2 = torch.tanh(w2[:, :2] @ h1 + w2[:, 2])
    return w3[:2] @ h2 + w3[2]


def independent_state(theta, x):
    with torch.enable_grad():
        prediction = independent_network(theta, x)
        jacobian = torch.autograd.functional.jacobian(lambda t: independent_network(t, x), theta)
    return prediction, jacobian


def row_leverage(jacobian, direction, denominator):
    return torch.cat(((jacobian[:6] * direction[:6]).reshape(2, 3).sum(-1),
                      (jacobian[6:12] * direction[6:12]).reshape(2, 3).sum(-1),
                      (jacobian[12:] * direction[12:]).sum().reshape(1))) / denominator


@pytest.mark.parametrize("iterations", [1, 2, 4])
@pytest.mark.parametrize("known_noise", [False, True])
@torch.no_grad()
def test_nonlinear_stream_matches_independent_fixed_prior_oracle(iterations, known_noise):
    model = learner(iterations, known_noise)
    theta, covariance = flat_weights(model).double(), model.cov.double().clone()
    noise = model.noise.double().clone()
    error, gain_sum, variance = [torch.zeros_like(noise) for _ in range(3)]
    for t in range(len(model.xs)):
        x, y = model.xs[t].double(), model.ys[t].double()
        for k in range(len(theta)):
            theta0 = theta[k].clone()
            prediction0 = independent_network(theta0, x)
            prior = covariance[k] + torch.diag(model.process[k].double())
            observation = model.noise_var[t].double() if known_noise else noise[k].clone()
            for _ in range(iterations):
                prediction, j = independent_state(theta[k], x)
                pj = prior @ j
                denominator = observation + j @ pj
                innovation = prediction - y - j @ (theta[k] - theta0)
                theta[k] = theta0 - pj * innovation / denominator
            covariance[k] = prior - torch.outer(pj, pj) / denominator
            credit = row_leverage(j, pj, denominator)
            gain_sum[k] += credit.sum()
            variance[k] += credit.var(unbiased=False)
            noise[k].lerp_((prediction0 - y).square(), model.a.noise_rate)
            error[k] += (prediction0 - model.clean[t].double()).square()
        model.update()
        for actual, expected in ((flat_weights(model), theta), (model.cov, covariance),
                                 (model.noise, noise), (model.error, error),
                                 (model.gain_sum, gain_sum), (model.unit_variance, variance)):
            torch.testing.assert_close(actual, expected.float(), rtol=1e-4, atol=5e-6)
        assert model.index.item() == model.steps.item() == t + 1


@torch.no_grad()
def test_one_iteration_matches_frozen_full_ekf_over_real_stream():
    actual, frozen = learner(1, False), learner(1, False, frozen=True)
    for _ in range(len(actual.xs)):
        actual.update()
        frozen.update()
        for observed, expected in zip(actual.mutable, frozen.mutable):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def linear_learner(iterations, known_noise=True):
    model = learner(iterations, known_noise)
    # Freeze both hidden layers by removing their covariance, not replacing the
    # network/Jacobian. The three output parameters have a correlated Gaussian.
    output_covariance = model.cov[:, -3:, -3:].clone()
    model.cov.zero_()
    model.cov[:, -3:, -3:].copy_(output_covariance)
    model.process.zero_()
    return model


@pytest.mark.parametrize("iterations", [1, 2, 4])
@torch.no_grad()
def test_linear_gaussian_posterior_counts_each_observation_once(iterations):
    model = linear_learner(iterations)
    initial = [w.double().clone() for w in model.weights]
    theta0 = initial[-1][:, 0]
    precision0 = torch.linalg.inv(model.cov[:, -3:, -3:].double())
    information0 = (precision0 @ theta0.unsqueeze(-1)).squeeze(-1)
    h1 = torch.tanh(model.xs.double() @ initial[0][0, :, :2].T + initial[0][0, :, 2])
    h2 = torch.tanh(h1 @ initial[1][0, :, :2].T + initial[1][0, :, 2])
    design = torch.cat((h2, torch.ones_like(h2[:, :1])), -1)
    for n in range(1, len(model.xs) + 1):
        model.update()
        x, r = design[:n], model.noise_var[:n].double()
        precision = precision0 + x.T @ (x / r[:, None])
        covariance = torch.linalg.inv(precision)
        information = information0 + x.T @ (model.ys[:n].double() / r)
        mean = torch.linalg.solve(precision, information.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(model.weights[-1][:, 0], mean.float(), rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(model.cov[:, -3:, -3:], covariance.float(), rtol=3e-5, atol=3e-6)
        for actual, fixed in zip(model.weights[:2], initial[:2]):
            torch.testing.assert_close(actual, fixed.float(), rtol=0, atol=0)
        assert torch.linalg.eigvalsh(model.cov.double()).min().item() >= -1e-8


@pytest.mark.parametrize("iterations", [2, 4])
@torch.no_grad()
def test_current_target_affects_mean_but_only_next_observation_noise(iterations):
    original, changed = linear_learner(iterations, False), linear_learner(iterations, False)
    changed.ys[0] += 3
    prediction = torch.stack([independent_network(theta, original.xs[0])
                              for theta in flat_weights(original)])
    original.update()
    changed.update()
    # The linear model keeps j target-independent, so identical covariance
    # directly witnesses use of the same historical R despite different targets.
    torch.testing.assert_close(original.cov, changed.cov, rtol=2e-5, atol=2e-6)
    assert not torch.equal(flat_weights(original), flat_weights(changed))
    for model in (original, changed):
        expected = 1 + model.a.noise_rate * ((prediction - model.ys[0]).square() - 1)
        torch.testing.assert_close(model.noise, expected, rtol=2e-5, atol=2e-6)
    assert not torch.equal(original.noise, changed.noise)
    for w, other in zip(changed.weights, original.weights):
        w.copy_(other)
    original.update()
    changed.update()
    assert (original.cov - changed.cov).abs().max().item() > 1e-4


@pytest.mark.parametrize("iterations", [2, 4])
@torch.no_grad()
def test_clean_labels_and_future_stream_cannot_affect_learning(iterations):
    original, changed = learner(iterations, False), learner(iterations, False)
    changed.clean.add_(19)
    changed.xs[3:].mul_(-3)
    changed.ys[3:].add_(10)
    changed.noise_var.mul_(7)
    for _ in range(3):
        original.update()
        changed.update()
    for left, right in zip(original.weights + [original.cov, original.noise, original.gain_sum],
                           changed.weights + [changed.cov, changed.noise, changed.gain_sum]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not torch.equal(original.error, changed.error)
    assert not torch.equal(original.null_error, changed.null_error)
    original.update()
    changed.update()
    assert not torch.equal(flat_weights(original), flat_weights(changed))


@pytest.mark.parametrize("iterations", [2, 4])
@torch.no_grad()
def test_capture_restores_nonzero_state_and_replays_remaining_stream(iterations):
    torch.compiler.reset()
    captured, eager = learner(iterations, False), learner(iterations, False)
    captured.update()
    eager.update()
    before = captured.snapshot()
    graph, _ = captured.capture()
    try:
        for actual, expected in zip(captured.mutable, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for _ in range(1, len(captured.xs)):
            graph.replay()
            eager.update()
            torch.cuda.synchronize()
            for actual, expected in zip(captured.mutable, eager.mutable):
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-5)
        assert captured.index.item() == captured.steps.item() == len(captured.xs)
    finally:
        del graph
        torch.compiler.reset()

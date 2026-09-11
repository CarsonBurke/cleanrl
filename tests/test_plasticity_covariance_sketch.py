"""CUDA covariance-sketch contracts; execute only through the mlq queue."""

import pytest
import torch

from cleanrl.plasticity import covariance_sketch_stream_v3 as sketch

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def learner(method="sketch", rank=2, known_noise=True):
    a = sketch.Args(input_dim=1, hidden=1, samples=8, buffer=4,
                    graph_steps=2, diffusion=0.03, noise_rate=0.2,
                    known_noise=known_noise)
    initial = [torch.tensor(value, device="cuda") for value in (
        [[1.1, 0.2]], [[0.7, -0.3]], [[0.25, -0.1]])]
    xs = torch.tensor([[-1.2], [0.4], [1.1], [-0.2], [0.8], [-0.7], [0.1], [1.4]], device="cuda")
    ys = torch.tensor([1.7, -0.4, 2.2, 0.9, -0.8, 0.5, 1.1, -0.2], device="cuda")
    noise = torch.tensor([0.4, 1.2, 0.7, 0.9, 0.6, 0.3, 0.8, 1.1], device="cuda")
    return sketch.SketchLearner(method, rank, (0.3, 1.2), initial, a, xs, ys,
                               ys.clone(), noise)


def flat_weights(model):
    return torch.cat([w.flatten(1) for w in model.weights], -1)


def dense_covariance(model):
    return torch.diag_embed(model.D) - model.U @ model.U.transpose(1, 2)


def scalar_network(theta, x):
    """Independent tiny network, rather than the implementation's Jacobian helper."""
    h1 = torch.tanh(theta[0] * x + theta[1])
    h2 = torch.tanh(theta[2] * h1 + theta[3])
    return theta[4] * h2 + theta[5]


def independent_state(theta, x):
    prediction = scalar_network(theta, x)
    jacobian = torch.autograd.functional.jacobian(lambda t: scalar_network(t, x), theta)
    return prediction, jacobian


@pytest.mark.parametrize("method", ["sketch", "sketch_scalar", "diagonal"])
@pytest.mark.parametrize("known_noise", [False, True])
@torch.no_grad()
def test_buffered_recurrence_matches_independent_dense_ekf(method, known_noise):
    model = learner(method, known_noise=known_noise)
    # Nonuniform covariance and process detect broadcast/isotropic approximations.
    model.D.mul_(torch.linspace(0.4, 1.9, 6, device="cuda"))
    model.process.mul_(torch.linspace(0.7, 1.6, 6, device="cuda"))
    theta = flat_weights(model).double()
    covariance = dense_covariance(model).double()
    noise = model.noise.double().clone()
    error = torch.zeros_like(noise)
    gain_sum = torch.zeros_like(noise)
    unit_variance = torch.zeros_like(noise)
    for t in range(model.capture_steps):
        for k in range(len(theta)):
            prediction, j = independent_state(theta[k], model.xs[t, 0].double())
            covariance[k] += torch.diag(model.process[k].double())
            pj = covariance[k] @ j
            leverage = j @ pj
            denominator = (model.noise_var[t].double() if known_noise else noise[k]) + leverage
            residual = prediction - model.ys[t].double()
            applied = j * (leverage / (j @ j)) if method == "sketch_scalar" else pj
            gains = (j * applied).reshape(3, 2).sum(-1) / denominator
            gain_sum[k] += gains.sum()
            unit_variance[k] += gains.var(unbiased=False)
            theta[k] -= applied * residual / denominator
            covariance[k] -= torch.outer(pj, pj) / denominator
            if method == "diagonal":
                covariance[k] = torch.diag(covariance[k].diagonal())
            noise[k].lerp_(residual.square(), model.a.noise_rate)
            error[k] += (prediction - model.clean[t].double()).square()
        model.update()
        torch.testing.assert_close(flat_weights(model), theta.float(), rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(dense_covariance(model), covariance.float(), rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(model.noise, noise.float(), rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(model.error, error.float(), rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(model.gain_sum, gain_sum.float(), rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(model.unit_variance, unit_variance.float(), rtol=3e-5, atol=3e-6)
        assert model.offset.item() == model.index.item() == model.steps.item() == t + 1


@pytest.mark.parametrize("rank", [0, 1, 8])
@torch.no_grad()
def test_compression_restores_uncertainty_in_psd_order_including_zero_modes(rank):
    model = learner(rank=rank)
    gen = torch.Generator(device="cuda").manual_seed(17)
    model.D.copy_(torch.rand(model.D.shape, generator=gen, device="cuda") + 0.2)
    normalized = torch.randn(model.U.shape, generator=gen, device="cuda", dtype=torch.float64)
    normalized *= 0.8 / torch.linalg.matrix_norm(normalized, ord=2)[:, None, None]
    model.U.copy_(normalized.float() * model.D.sqrt().unsqueeze(-1))
    before = dense_covariance(model).double()
    old_d = model.D.clone()
    model.offset.fill_(model.capture_steps)
    model.compress()
    after = dense_covariance(model).double()
    assert torch.linalg.eigvalsh(before).min().item() > 0
    assert torch.linalg.eigvalsh(after).min().item() > 0
    assert torch.linalg.eigvalsh(after - before).min().item() >= -2e-6
    assert torch.linalg.eigvalsh(torch.diag_embed(model.D.double()) - after).min().item() >= -2e-6
    torch.testing.assert_close(model.D, old_d, rtol=0, atol=0)
    if rank == 0:
        torch.testing.assert_close(after, torch.diag_embed(model.D.double()), rtol=0, atol=0)
    if rank >= model.D.shape[-1]:
        torch.testing.assert_close(after, before, rtol=2e-5, atol=2e-6)
    assert model.offset.item() == 0
    assert torch.count_nonzero(model.U[:, :, rank:]).item() == 0


@torch.no_grad()
def test_compression_selects_whitened_mode_not_largest_raw_downdate():
    model = learner(rank=1)
    model.D.fill_(1)
    model.D[:, 0] = 100
    model.U.zero_()
    # Raw energies 20 > 0.8, whitened energies 0.2 < 0.8.
    model.U[:, 0, 0] = 20 ** 0.5
    model.U[:, 1, 1] = 0.8 ** 0.5
    model.compress()
    expected = torch.diag_embed(model.D)
    expected[:, 1, 1] -= 0.8
    torch.testing.assert_close(dense_covariance(model), expected, rtol=1e-6, atol=1e-6)
    before_probe = model.snapshot()
    diagnostics = model.geometry(model.xs[:3])
    torch.testing.assert_close(torch.tensor(diagnostics["maximum_whitened_eigenvalue"]),
                               torch.full((2,), 0.8), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.tensor(diagnostics["covariance_downdate_trace"]),
                               torch.full((2,), 0.8), rtol=1e-6, atol=1e-6)
    for actual, expected in zip(model.mutable, before_probe):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_scalar_control_preserves_instantaneous_gain_but_erases_direction():
    full, scalar = learner(), learner("sketch_scalar")
    for model in (full, scalar):
        model.D.mul_(torch.linspace(0.2, 2.0, 6, device="cuda"))
        model.U[:, [0, 3, 5], 0] = torch.tensor([0.03, -0.1, 0.2], device="cuda")
    before = flat_weights(full)
    jacobian = torch.stack([independent_state(theta.double(), full.xs[0, 0].double())[1]
                            for theta in before]).float()
    full.update()
    scalar.update()
    delta_full = flat_weights(full) - before
    delta_scalar = flat_weights(scalar) - before
    torch.testing.assert_close((jacobian * delta_full).sum(-1),
                               (jacobian * delta_scalar).sum(-1), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(dense_covariance(full), dense_covariance(scalar), rtol=0, atol=0)
    ratio = (jacobian * delta_scalar).sum(-1) / jacobian.square().sum(-1)
    torch.testing.assert_close(delta_scalar, jacobian * ratio[:, None], rtol=2e-5, atol=2e-6)
    assert (delta_full - delta_scalar).abs().max().item() > 1e-3
    full.update()
    scalar.update()
    assert (dense_covariance(full) - dense_covariance(scalar)).abs().max().item() > 1e-4


@torch.no_grad()
def test_cross_neuron_covariance_moves_zero_local_jacobian_parameter():
    model = learner()
    model.process.zero_()
    model.D.fill_(1)
    model.U.zero_()
    model.weights[-1][:, 0, 0] = 0
    # P[first-layer bias, output bias] = +0.4 despite local hidden j = 0.
    model.U[:, 1, 0] = 0.4 ** 0.5
    model.U[:, -1, 0] = -(0.4 ** 0.5)
    before = model.weights[0][:, 0, -1].clone()
    theta = flat_weights(model)
    predictions, jacobians = zip(*(independent_state(t.double(), model.xs[0, 0].double()) for t in theta))
    j = torch.stack(jacobians).float()
    assert torch.count_nonzero(j[:, :4]).item() == 0
    pj = (dense_covariance(model) @ j.unsqueeze(-1)).squeeze(-1)
    denominator = model.noise_var[0] + (j * pj).sum(-1)
    expected = before - pj[:, 1] * (torch.stack(predictions).float() - model.ys[0]) / denominator
    model.update()
    torch.testing.assert_close(model.weights[0][:, 0, -1], expected)
    assert (model.weights[0][:, 0, -1] - before).abs().min().item() > 0.1


@pytest.mark.parametrize("method", ["sketch", "sketch_scalar", "diagonal"])
@torch.no_grad()
def test_clean_targets_and_future_observations_cannot_affect_learning(method):
    original, changed = learner(method, known_noise=False), learner(method, known_noise=False)
    changed.clean.add_(19)
    changed.xs[4:].mul_(-3)
    changed.ys[4:].add_(100)
    changed.noise_var.mul_(7)  # Unknown-noise learner must ignore privileged variance.
    for _ in range(4):
        original.update()
        changed.update()
    original.compress()
    changed.compress()
    for left, right in zip(original.weights + [original.D, original.U, original.noise],
                           changed.weights + [changed.D, changed.U, changed.noise]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not torch.equal(original.error, changed.error)
    assert not torch.equal(original.null_error, changed.null_error)
    original.update()
    changed.update()
    assert not torch.equal(flat_weights(original), flat_weights(changed))


@torch.no_grad()
def test_current_residual_updates_only_next_inferred_noise():
    original, changed = learner(known_noise=False), learner(known_noise=False)
    changed.ys[0] += 8
    original.update()
    changed.update()
    torch.testing.assert_close(dense_covariance(original), dense_covariance(changed), rtol=0, atol=0)
    torch.testing.assert_close(original.gain_sum, changed.gain_sum, rtol=0, atol=0)
    assert not torch.equal(original.noise, changed.noise)
    # Reset only mean to isolate the next observation's different noise scale.
    for w, other in zip(changed.weights, original.weights):
        w.copy_(other)
    original.update()
    changed.update()
    assert not torch.equal(dense_covariance(original), dense_covariance(changed))


@pytest.mark.parametrize("method", ["sketch", "sketch_scalar", "diagonal"])
@torch.no_grad()
def test_capture_restores_state_and_replays_exact_macrocycles(method):
    captured, reference = learner(method), learner(method)
    # Capture at a nonzero boundary to detect hardcoded reset-to-zero rollback.
    for model in (captured, reference):
        for _ in range(model.capture_steps):
            model.update()
        model.compress()
    before = captured.snapshot()
    replayer, parity = captured.capture()
    assert parity >= 0
    for actual, expected in zip(captured.mutable, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    replayer.replay()
    for _ in range(reference.capture_steps):
        reference.update()
    reference.compress()
    torch.cuda.synchronize()
    torch.testing.assert_close(flat_weights(captured), flat_weights(reference), rtol=3e-3, atol=3e-5)
    torch.testing.assert_close(dense_covariance(captured), dense_covariance(reference), rtol=3e-3, atol=3e-5)
    for actual, expected in ((captured.error, reference.error), (captured.noise, reference.noise),
                             (captured.gain_sum, reference.gain_sum),
                             (captured.unit_variance, reference.unit_variance)):
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-5)
    assert captured.index.item() == captured.steps.item() == len(captured.xs)
    assert captured.offset.item() == 0

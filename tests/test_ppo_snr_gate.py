"""Run CUDA uncertainty-gated Adam regressions through mlq; no training smoke runs."""

import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_snr_gate_v1 import SNRGatedAdam, uncertainty_gate
from cleanrl.ppo_continuous_action_snr_gate_momentum_v2 import (
    SNRGatedAdam as MomentumGatedAdam,
    ema_mean_variance_factor,
    uncertainty_gate as momentum_uncertainty_gate,
)
from test_ppo_normres_twohot import device


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    return device


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(params=[(SNRGatedAdam, False), (MomentumGatedAdam, True)], ids=["gradient", "momentum"])
def optimizer_spec(request):
    return request.param


def oracle_update(weight, history, lr, betas, eps, tau, gate_eps, mean_uncertainty=False):
    # Recompute normalized geometric averages from observations, rather than
    # duplicating the optimizer's recursive state or bias-correction clocks.
    gradients = np.stack(history).astype(np.float64)
    ages = np.arange(len(history) - 1, -1, -1)
    mean_weights = betas[0] ** ages
    adam_weights = betas[1] ** ages
    mean_weights /= mean_weights.sum()
    adam_weights /= adam_weights.sum()
    mean = np.tensordot(mean_weights, gradients, axes=1)
    gate_second_moment = np.tensordot(mean_weights, gradients ** 2, axes=1)
    adam_second_moment = np.tensordot(adam_weights, gradients ** 2, axes=1)
    variance = np.maximum(gate_second_moment - mean ** 2, 0)
    if mean_uncertainty:
        variance *= np.sum(mean_weights ** 2)
    gate = mean ** 2 / (mean ** 2 + tau * (variance + gate_eps))
    return weight - np.float32(lr) * gate * mean / (np.sqrt(adam_second_moment) + eps)


def test_compiled_updates_match_matched_moment_oracle_with_live_lr(optimizer_spec):
    from torch._dynamo.backends.registry import lookup_backend

    compilations = 0
    inductor = lookup_backend("inductor")

    def backend(graph, inputs, **kwargs):
        nonlocal compilations
        compilations += 1
        return inductor(graph, inputs, config_patches=kwargs.pop("options", {}), **kwargs)

    real_compile = torch.compile
    optimizer_class, mean_uncertainty = optimizer_spec
    initial = np.array([[0.2, -0.8, 1.3], [-0.4, 0.7, 0.1]], dtype=np.float32)
    parameter = torch.nn.Parameter(torch.tensor(initial, device="cuda"))
    betas, eps, tau, gate_eps = (0.75, 0.99), 1e-8, 2.75, 1e-12
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "compile", lambda fn, **kwargs: real_compile(fn, backend=backend, **kwargs))
        optimizer = optimizer_class([parameter], lr=0.02, betas=betas, eps=eps, gate_tau=tau, gate_eps=gate_eps)
    weight = initial.astype(np.float64)
    history = []
    with torch._dynamo.config.patch(error_on_recompile=True):
        for index in range(24):
            # Sustained spikes after low energy make beta2's second moment
            # incompatible with beta1's mean: their difference can be negative.
            spike = 15.0 if 12 <= index < 18 else 0.03
            gradient = np.array([[0.3, (-1.0) ** index, 1e-7], [0.0, spike, -0.1]], dtype=np.float32)
            lr = 0.02 * (1 - index / 24) if index != 16 else 0.0
            parameter.grad = torch.tensor(gradient, device="cuda")
            optimizer.set_lr(lr)
            previous = torch.cuda.get_sync_debug_mode()
            try:
                if index:
                    torch.cuda.set_sync_debug_mode("error")
                optimizer.step()
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            history.append(gradient)
            weight = oracle_update(weight, history, lr, betas, eps, tau, gate_eps, mean_uncertainty)
            np.testing.assert_allclose(parameter.detach().cpu().numpy(), weight, rtol=3e-5, atol=3e-6)
            np.testing.assert_array_equal(parameter.grad.cpu().numpy(), gradient)
    assert compilations == 1


def test_gate_distinguishes_directional_consistency_and_bounds_roundoff():
    tau, eps = 2.5, 1e-12
    gradients = np.stack([np.ones(12), (-1.0) ** np.arange(12)], axis=1)
    weights = 0.75 ** np.arange(11, -1, -1)
    weights /= weights.sum()
    mean = weights @ gradients
    second_moment = weights @ (gradients ** 2)
    expected = mean ** 2 / (mean ** 2 + tau * (second_moment - mean ** 2 + eps))
    actual = uncertainty_gate(
        torch.tensor(mean, dtype=torch.float32, device="cuda"),
        torch.tensor(second_moment, dtype=torch.float32, device="cuda"),
        tau,
        eps,
    ).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)
    assert actual[0] > 0.99
    assert actual[1] < 0.05

    # These are valid moments before conversion. Independent float32 rounding
    # can put q just below m*m (notably for 1.00000007); the gate must not exceed 1.
    exact_mean = np.array([0.0, 1e-8, 1.00000007, 100.000005])
    rounded_mean = exact_mean.astype(np.float32)
    rounded_second = (exact_mean ** 2 * (1 + 1e-14)).astype(np.float32)
    squared_mean = rounded_mean ** 2
    variance = np.maximum(rounded_second - squared_mean, 0)
    expected = squared_mean / (squared_mean + tau * (variance + eps))
    actual = uncertainty_gate(
        torch.tensor(rounded_mean, device="cuda"),
        torch.tensor(rounded_second, device="cuda"),
        tau,
        eps,
    ).cpu().numpy()
    assert np.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)


def test_missing_gradients_preserve_future_updates_but_zero_is_an_observation(optimizer_spec):
    initial = np.array([0.5, -0.25], dtype=np.float32)
    optimizer_class, mean_uncertainty = optimizer_spec
    parameters = [torch.nn.Parameter(torch.tensor(initial, device="cuda")) for _ in range(2)]
    betas, eps, tau, gate_eps = (0.8, 0.99), 1e-8, 1.7, 1e-12
    lr = 0.02
    optimizer = optimizer_class(parameters, lr=lr, betas=betas, eps=eps, gate_tau=tau, gate_eps=gate_eps, compile=False)
    first = np.array([0.2, -0.3], dtype=np.float32)
    zero = np.zeros(2, dtype=np.float32)
    last = np.array([-0.7, 0.1], dtype=np.float32)
    schedule = [
        (first, None),
        (None, None),
        (None, None),
        (None, first),
        (zero, None),
        (None, None),
        (None, zero),
        (last, last),
    ]
    expected = [initial.astype(np.float64) for _ in parameters]
    histories = [[], []]
    for observations in schedule:
        before = [parameter.detach().cpu().numpy().copy() for parameter in parameters]
        for parameter, gradient in zip(parameters, observations):
            parameter.grad = None if gradient is None else torch.tensor(gradient, device="cuda")
        optimizer.step()
        for index, (parameter, gradient) in enumerate(zip(parameters, observations)):
            actual = parameter.detach().cpu().numpy()
            if gradient is None:
                np.testing.assert_array_equal(actual, before[index])
            else:
                histories[index].append(gradient)
                expected[index] = oracle_update(expected[index], histories[index], lr, betas, eps, tau, gate_eps, mean_uncertainty)
                if not gradient.any():
                    assert np.all(actual != before[index])
            np.testing.assert_allclose(actual, expected[index], rtol=3e-5, atol=3e-6)
    # Identical observation histories produce identical future weights despite
    # different numbers and placements of globally executed optimizer steps.
    torch.testing.assert_close(parameters[0], parameters[1], rtol=0, atol=0)


def test_momentum_gate_uses_finite_history_weights_without_scaling_epsilon():
    # Startup is not the steady-state 1/19 factor: compare against explicit
    # normalized history weights, including no momentum and long-memory EMAs.
    counts = [1, 2, 7, 100, 10000]
    for beta in (0.0, 0.9, 0.999):
        expected_factors = []
        for count in counts:
            weights = beta ** np.arange(count, dtype=np.float64)
            weights /= weights.sum()
            expected_factors.append(np.sum(weights ** 2))
        corrections = 1.0 - beta ** torch.tensor(counts, device="cuda", dtype=torch.float64)
        factors = ema_mean_variance_factor(beta, corrections)
        np.testing.assert_allclose(factors.cpu().numpy(), expected_factors, rtol=1e-12, atol=1e-12)
        means = torch.tensor([1e-7, 0.3, -0.3], device="cuda", dtype=torch.float64)
        seconds = means.square() + 0.5 * means.square()
        gate = momentum_uncertainty_gate(means, seconds, 1.7, 1e-12, factors[:, None])
        host_mean = means.cpu().numpy()
        host_variance = 0.5 * host_mean ** 2
        expected = host_mean ** 2 / (
            host_mean ** 2 + 1.7 * (np.asarray(expected_factors)[:, None] * host_variance + 1e-12)
        )
        np.testing.assert_allclose(gate.cpu().numpy(), expected, rtol=1e-12, atol=1e-12)

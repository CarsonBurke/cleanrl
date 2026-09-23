"""Queue these fresh tangent-value CUDA contracts through mlq."""

from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from cleanrl.ppo_continuous_action_tangent_value_policy_v6 import Agent, TangentValueAdam
from test_ppo_normres_twohot import device


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

(
    ACTUAL_GAIN, MODEL_GAIN, SCALE, ACCEPTED, FALLBACK,
    SLOPE, CURVATURE, ORIGIN_LOSS, SELECTED_LOSS, NONFINITE,
) = range(10)


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    torch._dynamo.reset()
    yield device
    torch._dynamo.reset()


class BiasCritic(nn.Module):
    def __init__(self, device, *, value=0.0, center=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([value], device=device))
        self.register_buffer("center", torch.tensor([center], device=device))

    def forward(self, observations):
        return (self.bias - self.center).expand_as(observations)


class ExponentialCritic(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, device=device, dtype=torch.float64))

    def forward(self, observations):
        return (observations * self.weight).exp()


class LogarithmicCritic(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.1], device=device, dtype=torch.float64))

    def forward(self, observations):
        return (observations * self.weight).log()


def linear_critic(device, weights, *, dtype=torch.float64):
    critic = nn.Linear(len(weights), 1, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        critic.weight.copy_(critic.weight.new_tensor([weights]))
    return critic


def values(critic, observations):
    with torch.no_grad():
        return critic(observations).flatten()


def paired_gain(old, new, targets, vf_coef):
    # Compute the observed objective change independently, without subtracting
    # two large aggregate losses or trusting the optimizer's diagnostics.
    delta = old.double() - new.double()
    return vf_coef * (delta * (old.double() - targets.double() - 0.5 * delta)).mean()


def assert_first_observation(optimizer, parameter, gradient, betas):
    history = optimizer.state[parameter]
    assert history["step"].item() == 1
    torch.testing.assert_close(history["exp_avg"], (1 - betas[0]) * gradient, rtol=2e-6, atol=1e-12)
    torch.testing.assert_close(history["exp_avg_sq"], (1 - betas[1]) * gradient.square(),
                               rtol=2e-6, atol=1e-12)


def test_linear_critic_selects_weighted_gauss_newton_ray_optimum(device):
    critic = linear_critic(device, [2.0, -1.0, 0.25])
    observations = critic.weight.new_tensor([
        [3.0, 0.0, 1.0], [-1.0, 2.0, 0.5], [0.0, -3.0, 2.0],
        [1.0, 1.0, -1.0], [2.0, -0.5, 4.0], [-2.0, -1.0, 0.3],
    ])
    targets = observations.new_tensor([-0.2, 1.3, 0.7, -1.0, 2.2, -0.6])
    # Nonunit vf_coef and a material epsilon expose both an unweighted Adam
    # gradient and an incorrectly weighted GN denominator, without scale rows.
    vf_coef, betas, eps = 0.17, (0.6, 0.8), 0.2
    optimizer = TangentValueAdam(critic, vf_coef=vf_coef, betas=betas, eps=eps, compile=False)
    origin = critic.weight.detach().clone()
    old = values(critic, observations)
    residual = old - targets
    gradient = vf_coef * (residual[:, None] * observations).mean(0, keepdim=True)
    direction = gradient / (gradient.abs() + eps)
    tangent = (observations @ direction.T).flatten()
    slope = vf_coef * (residual * tangent).mean()
    curvature = vf_coef * tangent.square().mean()
    optimum = slope / curvature
    prediction = optimum * slope - 0.5 * optimum.square() * curvature

    result = optimizer.step(observations, targets)
    new = values(critic, observations)

    assert result[ACCEPTED].item() == 1
    assert result[FALLBACK].item() == 0
    assert result[NONFINITE].item() == 0
    torch.testing.assert_close(critic.weight, origin - optimum * direction, rtol=3e-6, atol=1e-8)
    torch.testing.assert_close(result[SCALE], optimum, rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(result[SLOPE], slope, rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(result[CURVATURE], curvature, rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(result[MODEL_GAIN], prediction, rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(result[ACTUAL_GAIN], paired_gain(old, new, targets, vf_coef),
                               rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(result[ACTUAL_GAIN], prediction, rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(result[ORIGIN_LOSS], 0.5 * residual.square().mean())
    torch.testing.assert_close(result[SELECTED_LOSS], 0.5 * (new - targets).square().mean())
    assert_first_observation(optimizer, critic.weight, gradient, betas)


def test_nonlinear_trials_select_best_real_gain_not_first_positive_trial(device):
    critic = ExponentialCritic(device)
    observations = critic.weight.new_ones((8, 1))
    targets = observations.new_full((8,), 5.0)
    vf_coef, betas, eps = 0.5, (0.6, 0.8), 1e-5
    optimizer = TangentValueAdam(critic, vf_coef=vf_coef, betas=betas, eps=eps, compile=False)
    origin = critic.weight.detach().clone()
    old = values(critic, observations)
    gradient = (vf_coef * (old - targets) * old).mean().reshape_as(origin)
    direction = gradient / (gradient.abs() + eps)
    tangent = old * direction.squeeze()
    slope = vf_coef * ((old - targets) * tangent).mean()
    curvature = vf_coef * tangent.square().mean()
    optimum = slope / curvature
    fractions = observations.new_tensor([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    with torch.no_grad():
        trials = [functional_call(critic, {"weight": origin - optimum * fraction * direction},
                                  (observations,)).flatten() for fraction in fractions]
    gains = torch.stack([paired_gain(old, trial, targets, vf_coef) for trial in trials])
    # Newton's full step gives exp(4), the first positive check exp(2), but
    # exp(1) is better. Acceptance must maximize actual, not predicted, gain.
    assert gains[0].item() < 0 < gains[1].item() < gains[2].item()
    best = int(gains.argmax().item())
    assert best == 2

    result = optimizer.step(observations, targets)
    new = values(critic, observations)
    selected_scale = optimum * fractions[best]

    assert result[ACCEPTED].item() == 1
    torch.testing.assert_close(new, trials[best], rtol=3e-6, atol=1e-8)
    torch.testing.assert_close(result[SCALE], selected_scale, rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(result[ACTUAL_GAIN], gains[best], rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(result[ACTUAL_GAIN], paired_gain(old, new, targets, vf_coef),
                               rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(result[MODEL_GAIN],
                               selected_scale * slope - 0.5 * selected_scale.square() * curvature,
                               rtol=3e-6, atol=1e-9)
    assert paired_gain(old, new, targets, vf_coef).item() > 0
    assert_first_observation(optimizer, critic.weight, gradient, betas)


def test_paired_gain_accepts_real_improvement_hidden_by_aggregate_mse_roundoff(device):
    critic = BiasCritic(device)
    observations = torch.ones((16, 1), device=device)
    # Both signs and the small mean are exactly representable. The residual
    # variance dominates FP32 MSE, but not the per-example paired gain.
    targets = observations.new_tensor([-2**14 + 0.125, 2**14 + 0.125]).repeat(8)
    vf_coef = 0.5
    # Make the first direction exactly -1 in FP32, isolating loss-subtraction
    # cancellation from a different issue: cancellation in a nonunit JVP slope.
    optimizer = TangentValueAdam(critic, vf_coef=vf_coef, betas=(0.0, 0.0), eps=1e-12, compile=False)
    old = values(critic, observations)
    before = 0.5 * (old - targets).square().mean()

    result = optimizer.step(observations, targets)
    new = values(critic, observations)
    after = 0.5 * (new - targets).square().mean()
    expected_gain = 0.5 * vf_coef * targets.double().mean().square()

    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert result[ACCEPTED].item() == 1
    torch.testing.assert_close(critic.bias, targets.mean().reshape_as(critic.bias), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(paired_gain(old, new, targets, vf_coef), expected_gain, rtol=2e-5, atol=1e-9)
    torch.testing.assert_close(result[ACTUAL_GAIN].double(), expected_gain, rtol=2e-5, atol=1e-9)
    assert result[ACTUAL_GAIN].item() > 0
    torch.testing.assert_close(result[ORIGIN_LOSS], result[SELECTED_LOSS], rtol=0, atol=0)


@pytest.mark.parametrize("unrepresentable", [False, True], ids=["zero_gradient", "sub_ulp_displacement"])
def test_zero_or_unrepresentable_displacement_cannot_be_accepted(device, unrepresentable):
    center = float(2**24) if unrepresentable else 0.0
    critic = BiasCritic(device, value=center, center=center)
    observations = torch.ones((16, 1), device=device)
    targets = observations.new_full((16,), 0.125 if unrepresentable else 0.0)
    optimizer = TangentValueAdam(critic, compile=False)
    origin = critic.bias.detach().clone()
    old = values(critic, observations)

    result = optimizer.step(observations, targets)

    torch.testing.assert_close(critic.bias, origin, rtol=0, atol=0)
    torch.testing.assert_close(values(critic, observations), old, rtol=0, atol=0)
    assert result[ACCEPTED].item() == 0
    assert result[ACTUAL_GAIN].item() == 0
    assert result[MODEL_GAIN].item() == 0
    assert result[SCALE].item() == 0
    if unrepresentable:
        # A positive tangent model alone cannot justify a rounded-away trial.
        assert result[SLOPE].item() > 0 and result[CURVATURE].item() > 0
    else:
        assert result[SLOPE].item() == 0 and result[CURVATURE].item() == 0


@pytest.mark.parametrize("old_target_weight", [0.0, 1e-20], ids=["zero_gradient", "tiny_gradient"])
def test_fresh_targets_escape_zero_or_tiny_previous_step_history(device, old_target_weight):
    critic = linear_critic(device, [0.0])
    observations = critic.weight.new_tensor([[1e6], [2e6], [-1e6], [0.5e6]])
    old_targets = observations.flatten() * old_target_weight
    optimizer = TangentValueAdam(critic, compile=False)
    # The tiny-gradient case has a real but extremely short ray scale; the
    # next minibatch must recompute its geometry, not inherit that step size.
    initial = optimizer.step(observations, old_targets)
    if old_target_weight:
        assert initial[ACCEPTED].item() == 1
        assert 0 < initial[SCALE].item() < 1e-12
    else:
        assert initial[ACCEPTED].item() == 0
    optimizer.step(observations, old_targets)
    origin = critic.weight.detach().clone()
    targets = observations.flatten() * 1.75
    old = values(critic, observations)

    result = optimizer.step(observations, targets)
    new = values(critic, observations)

    assert result[ACCEPTED].item() == 1
    assert (critic.weight - origin).abs().item() > 1.0
    torch.testing.assert_close(critic.weight, critic.weight.new_full((1, 1), 1.75), rtol=3e-6, atol=1e-8)
    assert paired_gain(old, new, targets, 0.5).item() > 1e11
    assert result[SELECTED_LOSS].item() < 1e-10 * result[ORIGIN_LOSS].item()
    assert optimizer.state[critic.weight]["step"].item() == 3


def test_opposed_adam_momentum_falls_back_to_current_preconditioned_gradient(device):
    critic = linear_critic(device, [0.0])
    observations = critic.weight.new_ones((8, 1))
    vf_coef, betas, eps = 0.5, (0.9, 0.999), 1e-5
    optimizer = TangentValueAdam(critic, vf_coef=vf_coef, betas=betas, eps=eps, compile=False)
    optimizer.step(observations, observations.new_ones(8))
    old = values(critic, observations)
    targets = observations.new_full((8,), 0.5)
    gradient = (vf_coef * (old - targets).mean()).reshape_as(critic.weight)
    previous_moment = optimizer.state[critic.weight]["exp_avg"].clone()
    previous_second = optimizer.state[critic.weight]["exp_avg_sq"].clone()
    moment = betas[0] * previous_moment + (1 - betas[0]) * gradient
    second = betas[1] * previous_second + (1 - betas[1]) * gradient.square()
    assert (gradient * moment).sum().item() < 0
    direction = gradient / ((second / (1 - betas[1] ** 2)).sqrt() + eps)
    tangent = (observations @ direction.T).flatten()

    result = optimizer.step(observations, targets)

    assert result[FALLBACK].item() == 1
    assert result[ACCEPTED].item() == 1
    torch.testing.assert_close(critic.weight, critic.weight.new_full((1, 1), 0.5), rtol=3e-6, atol=1e-8)
    torch.testing.assert_close(result[SLOPE], vf_coef * ((old - targets) * tangent).mean(),
                               rtol=3e-5, atol=1e-9)
    torch.testing.assert_close(result[CURVATURE], vf_coef * tangent.square().mean(), rtol=3e-5, atol=1e-9)
    torch.testing.assert_close(optimizer.state[critic.weight]["exp_avg"], moment)
    torch.testing.assert_close(optimizer.state[critic.weight]["exp_avg_sq"], second)
    assert optimizer.state[critic.weight]["step"].item() == 2


def test_all_nonfinite_log_domain_trials_preserve_exact_origin_and_observe_once(device):
    critic = LogarithmicCritic(device)
    observations = critic.weight.new_tensor([[1.0], [2.0], [0.5], [4.0]])
    old = values(critic, observations)
    targets = old - 64.0
    vf_coef, betas = 0.5, (0.6, 0.8)
    optimizer = TangentValueAdam(critic, vf_coef=vf_coef, betas=betas, compile=False)
    origin = critic.weight.detach().clone()
    gradient = (vf_coef * (old - targets).mean() / origin).reshape_as(origin)
    # The GN displacement is 6.4, even its 1/32 trial crosses log's domain.
    with torch.no_grad():
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = functional_call(critic, {"weight": origin - 6.4 * fraction}, (observations,))
            assert torch.isnan(trial).all()

    result = optimizer.step(observations, targets)

    torch.testing.assert_close(critic.weight, origin, rtol=0, atol=0)
    torch.testing.assert_close(values(critic, observations), old, rtol=0, atol=0)
    assert result[ACCEPTED].item() == 0
    assert result[ACTUAL_GAIN].item() == 0
    assert result[MODEL_GAIN].item() == 0
    assert result[SCALE].item() == 0
    assert result[NONFINITE].item() == 1
    torch.testing.assert_close(result[ORIGIN_LOSS], old.new_tensor(2048.0))
    torch.testing.assert_close(result[SELECTED_LOSS], result[ORIGIN_LOSS], rtol=0, atol=0)
    assert_first_observation(optimizer, critic.weight, gradient, betas)


def actual_critic(device):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
    )
    with torch.device(device):
        return Agent(spaces).critic


def test_compiled_actual_critic_cycles_remain_feasible_without_postwarmup_sync_or_recompile(device, monkeypatch):
    from torch._inductor.compile_fx import compile_fx

    compilations = 0

    def backend(graph, inputs, **kwargs):
        nonlocal compilations
        compilations += 1
        return compile_fx(graph, inputs, config_patches=kwargs.pop("options", {}), **kwargs)

    critic = actual_critic(device)
    eager_critic = deepcopy(critic)
    real_compile = torch.compile
    with monkeypatch.context() as patch:
        patch.setattr(torch, "compile", lambda function, **kwargs: real_compile(function, backend=backend, **kwargs))
        optimizer = TangentValueAdam(critic)
    eager = TangentValueAdam(eager_critic, compile=False)
    observations = torch.randn((64, 17), device=device)
    signal = 0.7 * observations[:, 0].sin() + 0.3 * observations[:, 1]
    shifts = (0.75, -0.5, 1.25, -1.0, 0.25, 0.0, None, -0.75)

    for iteration, shift in enumerate(shifts):
        current_observations = observations * (1 + 0.02 * iteration)
        old = values(critic, current_observations)
        eager_old = values(eager_critic, current_observations)
        # Near representational precision, either branch can legitimately win;
        # compare feasibility, not exact acceptance or selected-fraction bits.
        targets = old + 1e-7 if shift is None else signal + shift
        eager_targets = eager_old + 1e-7 if shift is None else targets
        expected = eager.step(current_observations, eager_targets)
        previous_sync_mode = torch.cuda.get_sync_debug_mode()
        try:
            if iteration >= 2:
                torch.cuda.set_sync_debug_mode("error")
                with torch._dynamo.config.patch(error_on_recompile=True):
                    result = optimizer.step(current_observations, targets)
            else:
                result = optimizer.step(current_observations, targets)
        finally:
            torch.cuda.set_sync_debug_mode(previous_sync_mode)

        if iteration == 1:
            warm_compilations = compilations
            assert warm_compilations > 0
        elif iteration >= 2:
            assert compilations == warm_compilations
        new = values(critic, current_observations)
        eager_new = values(eager_critic, current_observations)
        for checked, before, after, labels in (
            (result, old, new, targets), (expected, eager_old, eager_new, eager_targets),
        ):
            assert checked.is_cuda
            assert torch.isfinite(checked).all()
            assert checked[ACTUAL_GAIN].item() >= 0
            gain = paired_gain(before, after, labels, 0.5)
            # The independent check uses eager FP32 forwards; allow their
            # roundoff, not a meaningful increase in the regression objective.
            tolerance = 2e-5 * max(1.0, (before.double() - labels.double()).square().mean().item())
            assert gain.item() >= -tolerance
            torch.testing.assert_close(checked[ACTUAL_GAIN].double(), gain, rtol=3e-3, atol=tolerance)
            torch.testing.assert_close(checked[ORIGIN_LOSS], 0.5 * (before - labels).square().mean(),
                                       rtol=3e-4, atol=2e-6)
            torch.testing.assert_close(checked[SELECTED_LOSS], 0.5 * (after - labels).square().mean(),
                                       rtol=3e-4, atol=2e-6)
            if shift is not None:
                assert gain.item() > 0
        if shift is not None:
            torch.testing.assert_close(result[SELECTED_LOSS], expected[SELECTED_LOSS], rtol=1e-2, atol=2e-5)

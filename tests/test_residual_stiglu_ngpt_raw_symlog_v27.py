"""CUDA eager/compiled contracts for value clipping in raw-return units."""
import math
from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_raw_symlog_v27 import (
    Agent, Args, ppo_loss, scalar_value_loss, symlog, symexp, value_loss,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.fail("These contracts require CUDA; run them in the exclusive mlq test job.")
    configure_runtime(matmul_precision="highest", allow_tf32=False)


@pytest.fixture(params=["eager", "compiled"])
def api(request):
    functions = {
        "symlog": symlog,
        "symexp": symexp,
        "scalar_value_loss": scalar_value_loss,
        "value_loss": value_loss,
        "ppo_loss": ppo_loss,
    }
    if request.param == "eager":
        yield SimpleNamespace(**functions)
        return
    # Specializations are process-global; isolate each contract's modes/shapes.
    torch._dynamo.reset()
    try:
        yield SimpleNamespace(**{
            name: torch.compile(function, fullgraph=True, options={"triton.cudagraphs": False})
            for name, function in functions.items()
        })
    finally:
        torch._dynamo.reset()


def _coordinate(raw, mode):
    return raw if mode == "mse" else math.copysign(math.log1p(abs(raw)), raw)


def _element_loss(coordinate, target, mode):
    """Independent scalar oracle, never used as a model or compiled objective."""
    if mode == "mse":
        return 0.5 * (coordinate - target) ** 2
    if mode == "symlog_mse":
        return 0.5 * (coordinate - _coordinate(target, mode)) ** 2
    magnitude = abs(coordinate)
    target_magnitude = abs(target)
    return (
        math.expm1(magnitude) - magnitude - target * coordinate
        + (target_magnitude + 1.0) * math.log1p(target_magnitude) - target_magnitude
    )


def _element_gradient(coordinate, target, mode):
    if mode == "symlog_mean":
        return math.copysign(math.expm1(abs(coordinate)), coordinate) - target
    return coordinate - _coordinate(target, mode)


def test_signed_transforms_are_inverses_with_unit_derivative_at_origin(api):
    raw = torch.tensor(
        [-1e6, -20.0, -1e-7, 0.0, 1e-7, 20.0, 1e6],
        device="cuda", requires_grad=True,
    )
    coordinates = api.symlog(raw)
    gradient, = torch.autograd.grad(coordinates.sum(), raw)
    expected_coordinates = raw.detach().sign() * torch.log1p(raw.detach().abs())
    torch.testing.assert_close(coordinates, expected_coordinates, rtol=2e-6, atol=1e-12)
    torch.testing.assert_close(gradient, 1.0 / (1.0 + raw.detach().abs()), rtol=2e-6, atol=1e-12)
    assert gradient[3].item() == 1.0

    z = coordinates.detach().requires_grad_()
    decoded = api.symexp(z)
    inverse_gradient, = torch.autograd.grad(decoded.sum(), z)
    torch.testing.assert_close(decoded, raw.detach(), rtol=2e-6, atol=1e-12)
    torch.testing.assert_close(inverse_gradient, z.detach().abs().exp(), rtol=2e-6, atol=1e-12)
    assert inverse_gradient[3].item() == 1.0

    for original, roundtrip in (
        (raw, api.symexp(api.symlog(raw))),
        (z, api.symlog(api.symexp(z))),
    ):
        roundtrip_gradient, = torch.autograd.grad(roundtrip.sum(), original)
        torch.testing.assert_close(roundtrip, original, rtol=2e-6, atol=1e-12)
        torch.testing.assert_close(roundtrip_gradient, torch.ones_like(original), rtol=2e-6, atol=1e-7)


def test_mean_objective_gradient_remains_decoded_residual_including_origin(api):
    prediction = torch.tensor(
        [-5.0, -0.02, 0.0, 0.0, 0.0, 0.02, 5.0],
        device="cuda", dtype=torch.float64, requires_grad=True,
    )
    targets = torch.tensor(
        [-12.0, 3.0, 0.0, 2.0, -2.0, -3.0, 12.0],
        device="cuda", dtype=torch.float64, requires_grad=True,
    )
    loss = api.scalar_value_loss(prediction, targets, "symlog_mean")
    loss.backward()
    decoded = prediction.detach().sign() * torch.expm1(prediction.detach().abs())
    torch.testing.assert_close(
        prediction.grad, (decoded - targets.detach()) / targets.numel(), rtol=1e-12, atol=1e-12,
    )
    assert targets.grad is None


def test_conditional_optimum_is_arithmetic_mean_not_mean_symlog(api):
    targets = torch.tensor(
        [[0.0, 0.0, 0.0, 100.0], [-100.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 30.0]],
        device="cuda", dtype=torch.float64,
    )
    means = targets.mean(-1)
    optimum = api.symlog(means).detach().requires_grad_()
    loss = api.scalar_value_loss(optimum[:, None].expand_as(targets), targets, "symlog_mean")
    gradient, = torch.autograd.grad(loss, optimum)
    torch.testing.assert_close(gradient, torch.zeros_like(optimum), rtol=0, atol=1e-12)
    torch.testing.assert_close(api.symexp(optimum), means, rtol=1e-12, atol=1e-12)
    for displacement in (-0.05, 0.05):
        perturbed = api.scalar_value_loss(
            (optimum.detach() + displacement)[:, None].expand_as(targets), targets, "symlog_mean",
        )
        assert perturbed > loss.detach()

    log_optimum = api.symlog(targets).mean(-1).detach().requires_grad_()
    log_loss = api.scalar_value_loss(log_optimum[:, None].expand_as(targets), targets, "symlog_mse")
    log_gradient, = torch.autograd.grad(log_loss, log_optimum)
    torch.testing.assert_close(log_gradient, torch.zeros_like(log_optimum), rtol=0, atol=1e-12)
    assert torch.all((api.symexp(log_optimum) - means).abs() > 4.0)
    assert api.scalar_value_loss(
        log_optimum[:, None].expand_as(targets), targets, "symlog_mean",
    ) > loss.detach()


@pytest.mark.parametrize("mode", ["symlog_mean", "symlog_mse"])
def test_value_clip_radius_is_raw_units_at_large_positive_and_negative_values(api, mode):
    args = Args(critic_loss=mode, clip_vloss=True, clip_coef=0.2, clip_coef_upper=0.6)
    prediction = torch.tensor(
        [_coordinate(1100.0, mode), _coordinate(-1100.0, mode)],
        device="cuda", requires_grad=True,
    )
    old_values = torch.tensor([1000.0, -1000.0], device="cuda", requires_grad=True)
    targets = torch.tensor([2000.0, -2000.0], device="cuda", requires_grad=True)
    loss = api.value_loss(prediction, targets, old_values, args)
    expected = _element_loss(_coordinate(1000.2, mode), 2000.0, mode)
    torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=3e-5, atol=2e-6)
    # The actual change is 100 raw units, but less than .2 symlog units.
    # Coordinate-space clipping would incorrectly leave the prediction alone.
    wrong_coordinate_loss = _element_loss(_coordinate(1100.0, mode), 2000.0, mode)
    assert loss > loss.new_tensor(wrong_coordinate_loss * 1.1)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), rtol=0, atol=0)
    assert old_values.grad is None
    assert targets.grad is None


@pytest.mark.parametrize("mode", ["symlog_mean", "symlog_mse"])
def test_clipping_uses_pointwise_max_with_plateau_and_harmful_update_gradients(api, mode):
    args = Args(critic_loss=mode, clip_vloss=True)
    raw_predictions = [1.0, 1.0, -1.0, -1.0]
    raw_targets = [2.0, -2.0, -2.0, 2.0]
    coordinates = [_coordinate(value, mode) for value in raw_predictions]
    prediction = torch.tensor(coordinates, device="cuda", requires_grad=True)
    targets = torch.tensor(raw_targets, device="cuda", requires_grad=True)
    old_values = torch.zeros(4, device="cuda", requires_grad=True)
    loss = api.value_loss(prediction, targets, old_values, args)
    unclipped = [_element_loss(z, target, mode) for z, target in zip(coordinates, raw_targets)]
    clipped = [
        _element_loss(_coordinate(math.copysign(0.2, value), mode), target, mode)
        for value, target in zip(raw_predictions, raw_targets)
    ]
    expected = sum(max(original, limited) for original, limited in zip(unclipped, clipped)) / 4.0
    torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=2e-6, atol=2e-6)
    assert loss > loss.new_tensor(max(sum(unclipped), sum(clipped)) / 4.0 + 0.01)
    loss.backward()
    expected_gradient = prediction.new_tensor([
        0.0, _element_gradient(coordinates[1], raw_targets[1], mode) / 4.0,
        0.0, _element_gradient(coordinates[3], raw_targets[3], mode) / 4.0,
    ])
    torch.testing.assert_close(prediction.grad, expected_gradient, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(prediction.grad[[0, 2]], torch.zeros(2, device="cuda"), rtol=0, atol=0)
    assert old_values.grad is None
    assert targets.grad is None


@pytest.mark.parametrize("mode", ["mse", "symlog_mean", "symlog_mse"])
def test_inactive_clipping_preserves_unclipped_fp32_loss_and_gradient_at_zero(api, mode):
    args = Args(critic_loss=mode, clip_vloss=True)
    coordinates = [_coordinate(value, mode) for value in (-1000.0, -1.0, -0.04, 0.0, 0.04, 1.0, 1000.0)]
    prediction = torch.tensor(coordinates, device="cuda", requires_grad=True)
    decoded = prediction.detach() if mode == "mse" else prediction.detach().sign() * prediction.detach().abs().expm1()
    old_values = (decoded + prediction.new_tensor([0.05, -0.04, 0.03, 0.15, -0.03, 0.04, -0.05])).requires_grad_()
    targets = prediction.new_tensor([-1250.0, 2.0, -1.0, 3.0, 1.0, -2.0, 1250.0]).requires_grad_()
    reference_prediction = prediction.detach().clone().requires_grad_()
    reference = api.scalar_value_loss(reference_prediction, targets, mode)
    reference_gradient, = torch.autograd.grad(reference, reference_prediction)
    actual = api.value_loss(prediction, targets, old_values, args)
    actual.backward()
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(prediction.grad, reference_gradient, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(
        prediction.grad[3], prediction.new_tensor(_element_gradient(0.0, 3.0, mode) / 7.0),
        rtol=2e-6, atol=1e-7,
    )
    assert old_values.grad is None
    assert targets.grad is None


@pytest.mark.parametrize("mode", ["mse", "symlog_mean", "symlog_mse"])
@pytest.mark.parametrize("clip_vloss", [False, True])
def test_masks_exclude_nonfinite_operands_before_arithmetic_and_empty_masks_are_zero(api, mode, clip_vloss):
    args = Args(critic_loss=mode, clip_vloss=clip_vloss)
    positive = _coordinate(1.0, mode)
    negative = _coordinate(-1.0, mode)
    prediction = torch.tensor(
        [positive, float("nan"), negative, float("inf"), -float("inf"), 1000.0],
        device="cuda", requires_grad=True,
    )
    targets = torch.tensor(
        [2.0, float("nan"), 2.0, float("inf"), -float("inf"), float("inf")],
        device="cuda", requires_grad=True,
    )
    old_values = torch.tensor(
        [0.0, float("inf"), 0.0, float("nan"), -float("inf"), float("inf")],
        device="cuda", requires_grad=True,
    )
    mask = torch.tensor([True, False, True, False, False, False], device="cuda")
    loss = api.value_loss(prediction, targets, old_values, args, mask=mask)
    selected_positive = _coordinate(0.2, mode) if clip_vloss else positive
    expected = (_element_loss(selected_positive, 2.0, mode) + _element_loss(negative, 2.0, mode)) / 2.0
    torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=2e-6, atol=2e-6)
    loss.backward()
    expected_gradient = prediction.new_tensor([
        0.0 if clip_vloss else _element_gradient(positive, 2.0, mode) / 2.0,
        0.0, _element_gradient(negative, 2.0, mode) / 2.0, 0.0, 0.0, 0.0,
    ])
    torch.testing.assert_close(prediction.grad, expected_gradient, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(prediction.grad[~mask], torch.zeros(4, device="cuda"), rtol=0, atol=0)
    assert old_values.grad is None
    assert targets.grad is None

    prediction.grad = None
    empty = api.value_loss(prediction, targets, old_values, args, mask=torch.zeros_like(mask))
    torch.testing.assert_close(empty, torch.zeros_like(empty), rtol=0, atol=0)
    empty.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), rtol=0, atol=0)
    assert old_values.grad is None
    assert targets.grad is None


def test_clipped_mse_matches_original_v7_equation_and_gradients(api):
    args = Args(critic_loss="mse", clip_vloss=True, clip_coef=0.2, clip_coef_upper=0.6)
    prediction = torch.tensor([-5.0, -0.1, 0.0, 0.1, 4.0, 7.0], device="cuda", requires_grad=True)
    targets = torch.tensor([-9.0, 1.0, 2.0, -3.0, 8.0, 1.0], device="cuda", requires_grad=True)
    old_values = torch.tensor([-2.0, 0.0, 0.5, 0.0, 2.0, 9.0], device="cuda", requires_grad=True)
    reference_prediction = prediction.detach().clone().requires_grad_()
    clipped = old_values.detach() + (reference_prediction - old_values.detach()).clamp(-0.2, 0.2)
    reference = 0.5 * torch.maximum(
        (reference_prediction - targets.detach()).square(), (clipped - targets.detach()).square(),
    ).mean()
    reference_gradient, = torch.autograd.grad(reference, reference_prediction)
    actual = api.value_loss(prediction, targets, old_values, args)
    actual.backward()
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(prediction.grad, reference_gradient, rtol=2e-6, atol=2e-6)
    assert targets.grad is None
    assert old_values.grad is None


@pytest.mark.parametrize("mode", ["symlog_mean", "symlog_mse", "mse"])
def test_real_agent_ppo_integrates_value_clipping_without_changing_actor_gradients(api, mode):
    torch.manual_seed(1)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    base_args = Args(critic_loss=mode, norm_adv=False, ent_coef=0.01)
    # Exercise no-Args construction as a consumer: its public values must decode
    # exactly like the default symlog_mean trainer, not the historical MSE arm.
    agent = (Agent(spaces) if mode == "symlog_mean" else Agent(spaces, base_args)).cuda()
    agent.normalize_matrices()
    observations = torch.randn(8, 17, device="cuda")
    native_actions = torch.linspace(0.15, 0.85, 48, device="cuda").reshape(8, 6)
    ratios = torch.tensor([0.6, 0.79, 0.9, 1.0, 1.1, 1.21, 1.4, 1.5], device="cuda")
    advantages = torch.tensor([2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], device="cuda")
    with torch.no_grad():
        coordinate = agent.critic(observations).flatten()
        decoded = coordinate if mode == "mse" else coordinate.sign() * coordinate.abs().expm1()
        alpha, beta, public_values = agent.get_policy_and_value(observations)
        torch.testing.assert_close(public_values.flatten(), decoded, rtol=2e-6, atol=2e-6)
        old_logprobs = agent.action_logprob(alpha, beta, native_actions) - ratios.log()
        raw_targets = decoded + decoded.new_tensor([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0])
    mask = torch.tensor([True, True, False, True, True, False, True, True], device="cuda")
    returns = torch.where(mask, raw_targets, float("nan")).requires_grad_()
    old_values = torch.where(mask, decoded - 1.0, float("inf")).requires_grad_()
    critic_parameters = tuple(agent.critic.parameters())
    reference_actor_gradients = None
    reference_critic_gradients = None
    reference_value = None
    for enabled in (False, True):
        args = replace(base_args, clip_vloss=enabled)
        agent.zero_grad(set_to_none=True)
        expected_value = api.value_loss(
            agent.critic(observations).flatten(), returns, old_values, args, mask=mask,
        )
        expected_critic_gradients = torch.autograd.grad(args.vf_coef * expected_value, critic_parameters)
        loss, metrics = api.ppo_loss(
            agent, observations, native_actions, old_logprobs, advantages,
            returns, old_values, args, critic_mask=mask,
        )
        torch.testing.assert_close(metrics[1], expected_value.detach(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            loss, metrics[0] - args.ent_coef * metrics[2] + args.vf_coef * expected_value.detach(),
            rtol=2e-5, atol=2e-6,
        )
        expected_pg = torch.maximum(
            -advantages * ratios, -advantages * ratios.clamp(0.8, 1.2),
        ).mean()
        torch.testing.assert_close(metrics[0], expected_pg, rtol=2e-5, atol=2e-6)
        loss.backward()
        assert returns.grad is None
        assert old_values.grad is None
        for parameter, expected in zip(critic_parameters, expected_critic_gradients, strict=True):
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-6)
        actor_gradients = [parameter.grad.detach().clone() for parameter in agent.actor.parameters()]
        critic_gradients = [parameter.grad.detach().clone() for parameter in critic_parameters]
        assert all(torch.isfinite(gradient).all() for gradient in (*actor_gradients, *critic_gradients))
        if not enabled:
            assert any(torch.count_nonzero(gradient) > 0 for gradient in actor_gradients)
            reference_actor_gradients = actor_gradients
            reference_critic_gradients = critic_gradients
            reference_value = metrics[1].clone()
        else:
            assert metrics[1] > reference_value
            assert any(
                not torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)
                for actual, expected in zip(critic_gradients, reference_critic_gradients, strict=True)
            )
            # Deliberately compare before joint gradient clipping: the changed
            # critic norm is allowed to change the later shared clipping factor.
            for actual, expected in zip(actor_gradients, reference_actor_gradients, strict=True):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

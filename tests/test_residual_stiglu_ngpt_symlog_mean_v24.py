"""CUDA contracts for arithmetic-mean symlog regression and frozen v7 fidelity."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_residual_stiglu_ngpt_scaled_asymclip_v7 as baseline
from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_symlog_mean_v24 import (
    Agent, Args, apply_gradient_clipping, ppo_loss, scalar_value_loss, symlog, symexp,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.fail("These contracts require CUDA; run them in the exclusive mlq test job.")
    configure_runtime(matmul_precision="highest", allow_tf32=False)


@pytest.fixture(scope="module", params=["eager", "compiled"])
def value_loss(request):
    if request.param == "compiled":
        return torch.compile(scalar_value_loss, fullgraph=True,
                             options={"triton.cudagraphs": False})
    return scalar_value_loss


@pytest.fixture
def spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )


def test_symlog_roundtrip_covers_signs_zero_and_small_values():
    values = torch.tensor(
        [-1e6, -100.0, -1.0, -1e-7, 0.0, 1e-7, 1.0, 100.0, 1e6], device="cuda",
    )
    coordinates = symlog(values)
    torch.testing.assert_close(symexp(coordinates), values, rtol=2e-6, atol=1e-12)
    torch.testing.assert_close(coordinates.sign(), values.sign(), rtol=0, atol=0)
    torch.testing.assert_close(symlog(symexp(coordinates)), coordinates, rtol=2e-6, atol=1e-12)
    assert coordinates[4].item() == 0.0


def test_mean_loss_gradient_is_decoded_residual_including_origin(value_loss):
    prediction = torch.tensor(
        [-5.0, -0.02, 0.0, 0.0, 0.0, 0.02, 5.0],
        device="cuda", dtype=torch.float64, requires_grad=True,
    )
    targets = torch.tensor(
        [-12.0, 3.0, 0.0, 2.0, -2.0, -3.0, 12.0],
        device="cuda", dtype=torch.float64, requires_grad=True,
    )
    loss = value_loss(prediction, targets, "symlog_mean")
    loss.backward()
    decoded = prediction.detach().sign() * torch.expm1(prediction.detach().abs())
    torch.testing.assert_close(
        prediction.grad, (decoded - targets.detach()) / targets.numel(), rtol=1e-12, atol=1e-12,
    )
    assert targets.grad is None
    # The target-only conjugate term makes perfect individual predictions zero,
    # not an arbitrary negative loss with the same training gradient.
    perfect_loss = value_loss(symlog(targets.detach()), targets, "symlog_mean")
    torch.testing.assert_close(perfect_loss, torch.zeros_like(perfect_loss), rtol=0, atol=1e-12)


def test_conditional_optimum_is_arithmetic_mean_not_mean_symlog(value_loss):
    targets = torch.tensor(
        [[0.0, 0.0, 0.0, 100.0], [-100.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 30.0]],
        device="cuda", dtype=torch.float64,
    )
    means = targets.mean(-1)
    optimum = symlog(means).detach().requires_grad_()
    loss = value_loss(optimum[:, None].expand_as(targets), targets, "symlog_mean")
    gradient, = torch.autograd.grad(loss, optimum)
    torch.testing.assert_close(gradient, torch.zeros_like(optimum), rtol=0, atol=1e-12)
    torch.testing.assert_close(symexp(optimum), means, rtol=1e-12, atol=1e-12)
    for displacement in (-0.05, 0.05):
        perturbed = value_loss(
            (optimum.detach() + displacement)[:, None].expand_as(targets), targets, "symlog_mean",
        )
        assert perturbed > loss.detach()

    naive_optimum = symlog(targets).mean(-1).detach().requires_grad_()
    naive_loss = value_loss(naive_optimum[:, None].expand_as(targets), targets, "symlog_mse")
    naive_gradient, = torch.autograd.grad(naive_loss, naive_optimum)
    torch.testing.assert_close(naive_gradient, torch.zeros_like(naive_optimum), rtol=0, atol=1e-12)
    # Positive, negative, and mixed-sign conditional targets all expose the bias.
    assert torch.all((symexp(naive_optimum) - means).abs() > 4.0)
    assert value_loss(
        naive_optimum[:, None].expand_as(targets), targets, "symlog_mean",
    ) > loss.detach()


def test_mean_loss_and_gradient_stay_finite_over_wide_return_range(value_loss):
    values = torch.tensor(
        [-1e6, -1e3, -10.0, -1e-4, 0.0, 1e-4, 10.0, 1e3, 1e6], device="cuda",
    )
    prediction = symlog(values).repeat_interleave(values.numel()).detach().requires_grad_()
    targets = values.repeat(values.numel()).requires_grad_()
    loss = value_loss(prediction, targets, "symlog_mean")
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    decoded = prediction.detach().sign() * torch.expm1(prediction.detach().abs())
    torch.testing.assert_close(
        prediction.grad, (decoded - targets.detach()) / targets.numel(), rtol=3e-6, atol=1e-3,
    )
    assert targets.grad is None


def test_no_clipping_preserves_gradients_and_global_clipping_bounds_joint_norm():
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0], device="cuda")),
        torch.nn.Parameter(torch.tensor([3.0], device="cuda")),
        torch.nn.Parameter(torch.tensor([4.0], device="cuda")),
    ]
    parameters[0].grad = torch.tensor([3.0, 4.0], device="cuda")
    parameters[1].grad = torch.tensor([12.0], device="cuda")
    original = [parameter.grad.clone() for parameter in parameters[:2]]
    apply_gradient_clipping(iter(parameters), 0.5, "none")
    for parameter, expected in zip(parameters[:2], original, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    assert parameters[2].grad is None

    apply_gradient_clipping(iter(parameters), 0.5, "global")
    joint_gradient = torch.cat([parameter.grad for parameter in parameters[:2]])
    assert joint_gradient.norm() <= 0.5 + 1e-6
    # Independent per-parameter clipping would fail both this direction and norm.
    torch.testing.assert_close(joint_gradient, torch.cat(original) * (0.5 / 13.0))
    assert parameters[2].grad is None


@pytest.mark.parametrize("mode", ["mse", "symlog_mean", "symlog_mse"])
def test_public_agent_values_use_decoded_return_units(spaces, mode):
    torch.manual_seed(1)
    agent = Agent(spaces, Args(critic_loss=mode)).cuda()
    agent.normalize_matrices()
    observations = torch.randn(8, 17, device="cuda")
    actions = torch.zeros(8, 6, device="cuda")
    options = {"triton.cudagraphs": False}
    critic = torch.compile(agent.critic, fullgraph=True, options=options)
    get_value = torch.compile(agent.get_value, fullgraph=True, options=options)
    policy = torch.compile(agent.get_policy_and_value, fullgraph=True, options=options)
    action_and_value = torch.compile(agent.get_action_and_value, fullgraph=True, options=options)
    with torch.no_grad():
        raw = critic(observations)
        expected = raw if mode == "mse" else raw.sign() * torch.expm1(raw.abs())
        torch.testing.assert_close(get_value(observations), expected)
        torch.testing.assert_close(policy(observations)[2], expected)
        torch.testing.assert_close(action_and_value(observations, actions)[3], expected)
        if mode != "mse":
            assert (expected - raw).abs().max() > 1e-3


def test_mse_global_matches_frozen_v7_initialization_ppo_and_projected_adam_step(spaces):
    args = Args()
    historical_args = baseline.Args(
        total_timesteps=50_000_000, norm_adv=False, clip_vloss=False, clip_coef_upper=0.2,
    )
    torch.manual_seed(1)
    historical = baseline.Agent(spaces).cuda()
    torch.manual_seed(1)
    current = Agent(spaces, args).cuda()
    observations = torch.randn(8, 17, device="cuda")
    options = {"triton.cudagraphs": False}
    historical_policy = torch.compile(historical.get_policy_and_value, fullgraph=True, options=options)
    current_policy = torch.compile(current.get_policy_and_value, fullgraph=True, options=options)
    with torch.no_grad():
        # Check both fresh initialization and the matrix projection used by PPO.
        for expected, actual in zip(historical_policy(observations), current_policy(observations), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        historical.normalize_matrices()
        current.normalize_matrices()
        for expected, actual in zip(historical_policy(observations), current_policy(observations), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        alpha, beta, _ = historical_policy(observations)
        native_actions = torch.linspace(0.15, 0.85, 48, device="cuda").reshape(8, 6)
        ratios = torch.tensor([0.6, 0.79, 0.9, 1.0, 1.1, 1.21, 1.4, 1.5], device="cuda")
        old_logprobs = historical.action_logprob(alpha, beta, native_actions) - ratios.log()
    advantages = torch.tensor([2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], device="cuda")
    returns = torch.tensor([-3.0, 4.0, -5.0, 7.0, -9.0, 11.0, -13.0, 17.0], device="cuda")
    old_values = returns + 100.0
    batch = (observations, native_actions, old_logprobs, advantages, returns, old_values)
    historical_loss_fn = torch.compile(baseline.ppo_loss, fullgraph=True, options=options)
    current_loss_fn = torch.compile(ppo_loss, fullgraph=True, options=options)
    expected_loss, expected_metrics = historical_loss_fn(historical, *batch, historical_args)
    actual_loss, actual_metrics = current_loss_fn(current, *batch, args)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_metrics[:6], expected_metrics, rtol=2e-6, atol=2e-6)
    expected_loss.backward()
    actual_loss.backward()
    for expected, actual in zip(historical.parameters(), current.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
    torch.nn.utils.clip_grad_norm_(historical.parameters(), historical_args.max_grad_norm)
    apply_gradient_clipping(current.parameters(), args.max_grad_norm, args.grad_clip)
    torch.optim.Adam(historical.parameters(), lr=historical_args.learning_rate, eps=1e-5, fused=True).step()
    torch.optim.Adam(current.parameters(), lr=args.learning_rate, eps=1e-5, fused=True).step()
    historical.normalize_matrices()
    current.normalize_matrices()
    with torch.no_grad():
        for expected, actual in zip(historical_policy(observations), current_policy(observations), strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

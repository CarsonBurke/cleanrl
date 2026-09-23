"""CUDA contracts for the raw-reward scalar-MSE control plus PopArt.

Run only in the queued CUDA validation job. No CPU model fallback is permitted.
"""
import io
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Beta

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_popart_v28 import (
    Agent, Args, ppo_loss,
)
from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_raw_reward_v26 import (
    Agent as ControlAgent,
    Args as ControlArgs,
)
from cleanrl.shared.ppo_loop import TruncationBootstrapCache, get_gae_fn
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.fail("These contracts require CUDA; run them in the exclusive mlq test job.")
    configure_runtime(matmul_precision="highest", allow_tf32=False)


@pytest.fixture(params=["eager", "compiled"])
def execution(request):
    compiled = request.param == "compiled"
    if compiled:
        torch._dynamo.reset()

    def wrap(function, *, dynamic=False):
        if not compiled:
            return function
        return torch.compile(
            function, fullgraph=True, dynamic=dynamic,
            options={"triton.cudagraphs": False},
        )

    try:
        yield SimpleNamespace(wrap=wrap, compiled=compiled)
    finally:
        if compiled:
            torch._dynamo.reset()


def _spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )


def _agent(args=None):
    torch.manual_seed(1)
    agent = Agent(_spaces(), args).cuda()
    agent.normalize_matrices()
    return agent


def _ema(mean, square_mean, returns, rate, lower, upper):
    """VMPO's per-rollout EMA, independent of any model or projection code."""
    mean = torch.lerp(mean, returns.detach().mean(), rate)
    square_mean = torch.lerp(square_mean, returns.detach().square().mean(), rate)
    std = (square_mean - mean.square()).clamp_min(0.0).sqrt().clamp(lower, upper)
    return mean, square_mean, std


def _assert_frame(agent, mean, square_mean, std, targets, normalized):
    torch.testing.assert_close(agent.popart_mean, mean)
    torch.testing.assert_close(agent.popart_sq_mean, square_mean)
    torch.testing.assert_close(agent.popart_std, std)
    torch.testing.assert_close(normalized, (targets.detach() - mean) / std)
    assert not normalized.requires_grad
    assert targets.grad is None


def test_seeded_initial_policy_and_raw_values_match_the_exact_scalar_control(execution):
    control_args = ControlArgs(
        critic_loss="mse", critic_target="gae", norm_reward=False,
        norm_adv=False, grad_clip="none", clip_vloss=False,
    )
    torch.manual_seed(1)
    control = ControlAgent(_spaces(), control_args).cuda()
    torch.manual_seed(1)
    agent = Agent(_spaces()).cuda()
    # Extra learned translation or a different head changes the control's model.
    assert sum(p.numel() for p in agent.parameters()) == sum(p.numel() for p in control.parameters())
    observations = torch.randn(19, 17, device="cuda")
    policy = execution.wrap(agent.get_policy_and_value)
    control_policy = execution.wrap(control.get_policy_and_value)
    project = execution.wrap(agent.normalize_matrices)
    control_project = execution.wrap(control.normalize_matrices)
    with torch.no_grad():
        for projected in (False, True):
            if projected:
                project()
                control_project()
            for actual, expected in zip(policy(observations), control_policy(observations), strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                agent.get_normalized_value(observations), control.get_value(observations),
                rtol=0, atol=0,
            )


def test_frame_updates_preserve_signed_and_zero_gain_values_through_unit_projection(execution):
    agent = _agent()
    observations = torch.randn(23, 17, device="cuda") * 2.0
    policy = execution.wrap(agent.get_policy_and_value)
    update = execution.wrap(agent.update_popart)
    project = execution.wrap(agent.normalize_matrices)
    mean = torch.zeros((), device="cuda")
    square_mean = torch.ones((), device="cuda")
    targets = (
        torch.linspace(-20.0, 80.0, 23, device="cuda"),
        torch.linspace(-90.0, -10.0, 23, device="cuda"),
        torch.linspace(5.0, 17.0, 23, device="cuda"),
    )
    for gain in (1.75, -1.25, 0.0):
        with torch.no_grad():
            agent.critic.readout_gain.fill_(gain)
            agent.critic.head[0].bias.fill_(0.375)
            before = tuple(value.clone() for value in policy(observations))
        for raw_targets, rate in zip(targets, (1e-4, 0.4, 0.7), strict=True):
            raw_targets = raw_targets.detach().requires_grad_()
            mean, square_mean, std = _ema(mean, square_mean, raw_targets, rate, 1e-2, 1e6)
            normalized = update(raw_targets, rate, 1e-2, 1e6)
            _assert_frame(agent, mean, square_mean, std, raw_targets, normalized)
            with torch.no_grad():
                # Rescaling the unit head matrix would pass the first comparison
                # but fail the second, because projection erases that rescaling.
                for projected in (False, True):
                    if projected:
                        project()
                    alpha, beta, value = policy(observations)
                    torch.testing.assert_close(alpha, before[0], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(beta, before[1], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(value, before[2], rtol=3e-5, atol=2e-5)
                    normalized_value = agent.get_normalized_value(observations)
                    torch.testing.assert_close(
                        normalized_value, (before[2] - mean) / std, rtol=3e-5, atol=2e-5,
                    )
                    assert torch.isfinite(value).all()
                    assert torch.isfinite(normalized_value).all()


def test_vmpo_statistics_guard_degenerate_variance_without_replacing_second_moment(execution):
    agent = _agent()
    update = execution.wrap(agent.update_popart)
    value_fn = execution.wrap(agent.get_value)
    observations = torch.randn(11, 17, device="cuda")
    with torch.no_grad():
        before = value_fn(observations).clone()
    mean = torch.zeros((), device="cuda")
    square_mean = torch.ones((), device="cuda")
    cases = (
        # A singleton constant return must not use sample variance (NaN).
        (torch.tensor([13.0], device="cuda"), 1.0, 1e-2, 1e6),
        # Clipping std must not overwrite the actual EMA second moment.
        (torch.tensor([-100.0, 100.0], device="cuda"), 1.0, 1e-2, 4.0),
        (torch.tensor([-2.0, 2.0], device="cuda"), 0.25, 1e-2, 1e6),
    )
    for targets, rate, lower, upper in cases:
        targets.requires_grad_()
        mean, square_mean, std = _ema(mean, square_mean, targets, rate, lower, upper)
        normalized = update(targets, rate, lower, upper)
        _assert_frame(agent, mean, square_mean, std, targets, normalized)
        with torch.no_grad():
            torch.testing.assert_close(value_fn(observations), before, rtol=3e-5, atol=3e-5)
        assert torch.isfinite(normalized).all()


@pytest.mark.parametrize("gain", [-1.25, 0.0])
def test_ppo_preserves_raw_actor_gradients_and_transforms_normalized_head_gradients(execution, gain):
    args = Args(ent_coef=0.03)
    agent = _agent(args)
    with torch.no_grad():
        agent.critic.readout_gain.fill_(gain)
        agent.critic.head[0].bias.fill_(0.375)
    observations = torch.randn(8, 17, device="cuda")
    native_actions = torch.linspace(0.15, 0.85, 48, device="cuda").reshape(8, 6)
    ratios = torch.tensor([0.6, 0.79, 0.9, 1.0, 1.1, 1.21, 1.4, 1.5], device="cuda")
    advantages = torch.tensor([2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], device="cuda")
    offsets = torch.tensor([4.0, -2.0, 6.0, -3.0, 8.0, -5.0, 10.0, -6.0], device="cuda")
    with torch.no_grad():
        alpha, beta, initial_value = agent.get_policy_and_value(observations)
        initial_value = initial_value.flatten().clone()
        old_logprobs = agent.action_logprob(alpha, beta, native_actions) - ratios.log()
    returns = (initial_value + offsets).requires_grad_()
    # With no value clipping this argument is genuinely unused, not zero-weighted.
    old_values = torch.full_like(returns, float("nan"), requires_grad=True)
    actor_parameters = tuple(agent.actor.parameters())
    critic_parameters = tuple(agent.critic.parameters())
    head_parameters = (
        agent.critic.head[0].weight, agent.critic.head[0].bias, agent.critic.readout_gain,
    )
    alpha, beta, _ = agent.get_policy_and_value(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    logprobs = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    live_ratio = (logprobs - old_logprobs).exp()
    actor_loss = torch.maximum(-advantages * live_ratio, -advantages * live_ratio.clamp(0.8, 1.2)).mean()
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1).mean()
    expected_actor_gradients = torch.autograd.grad(actor_loss - args.ent_coef * entropy, actor_parameters)
    expected_policy_loss = torch.maximum(-advantages * ratios, -advantages * ratios.clamp(0.8, 1.2)).mean()
    loss_fn = execution.wrap(ppo_loss)
    update = execution.wrap(agent.update_popart)
    value_fn = execution.wrap(agent.get_value)
    initial_actor_gradients = None
    initial_head_gradients = None
    initial_std = agent.popart_std.clone()
    for frame in range(3):
        if frame:
            frame_returns = torch.linspace(-12.0, 28.0, 8, device="cuda") + frame * 7.0
            update(frame_returns, 0.4, 1e-2, 1e6)
        agent.zero_grad(set_to_none=True)
        # The oracle stays in raw units and differentiates through the public
        # decoder. It catches normalized targets subtracted from raw predictions,
        # double decoding, a missing half factor, and an attached return target.
        raw_prediction = agent.get_value(observations).flatten()
        oracle = 0.5 * ((raw_prediction - returns.detach()) / agent.popart_std).square().mean()
        expected_critic_gradients = torch.autograd.grad(args.vf_coef * oracle, critic_parameters)
        loss, metrics = loss_fn(
            agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args,
        )
        torch.testing.assert_close(metrics[0], expected_policy_loss, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(metrics[1], oracle.detach(), rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(
            loss, metrics[0] - args.ent_coef * metrics[2] + args.vf_coef * oracle.detach(),
            rtol=3e-5, atol=3e-6,
        )
        loss.backward()
        assert returns.grad is None
        assert old_values.grad is None
        for parameter, expected in zip(critic_parameters, expected_critic_gradients, strict=True):
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected, rtol=3e-4, atol=3e-6)
        for parameter, expected in zip(actor_parameters, expected_actor_gradients, strict=True):
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected, rtol=3e-4, atol=3e-6)
        actor_gradients = [parameter.grad.clone() for parameter in actor_parameters]
        head_gradients = [parameter.grad.clone() for parameter in head_parameters]
        if frame == 0:
            initial_actor_gradients = actor_gradients
            initial_head_gradients = head_gradients
            assert any(torch.count_nonzero(gradient) for gradient in actor_gradients)
            assert torch.count_nonzero(head_gradients[2])
        else:
            for actual, expected in zip(actor_gradients, initial_actor_gradients, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            scale = initial_std / agent.popart_std
            for actual, expected in zip(head_gradients[:2], initial_head_gradients[:2], strict=True):
                torch.testing.assert_close(actual, expected * scale.square(), rtol=3e-4, atol=3e-6)
            torch.testing.assert_close(
                head_gradients[2], initial_head_gradients[2] * scale, rtol=3e-4, atol=3e-6,
            )
        if gain == 0.0:
            for parameter in critic_parameters:
                if parameter is not agent.critic.readout_gain:
                    torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(value_fn(observations).flatten(), initial_value, rtol=3e-5, atol=2e-5)

    # Frame changes preserve predictions; actual learned updates must not be
    # hidden by a stale compiled value graph, and zero gain must remain learnable.
    before_loss = metrics[1].clone()
    torch.optim.SGD(critic_parameters, lr=1e-3).step()
    with torch.no_grad():
        after = value_fn(observations).flatten()
        assert not torch.allclose(after, initial_value, rtol=0, atol=1e-7)
        _, learned_metrics = loss_fn(
            agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args,
        )
        assert learned_metrics[1] < before_loss


def test_public_values_and_truncation_bootstraps_keep_gae_in_raw_reward_units(execution):
    agent = _agent()
    observations = torch.randn(4, 2, 17, device="cuda")
    tail = torch.randn(2, 17, device="cuda")
    finals = np.random.default_rng(7).normal(size=(2, 2, 17)).astype(np.float32)
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], device="cuda")
    terms = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], device="cuda")
    truncs = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], device="cuda")
    cache = TruncationBootstrapCache(4, 2, (17,))
    cache.push_normalized(0, np.array([False, True]), finals[0])
    cache.push_normalized(2, np.array([True, False]), finals[1])
    value_fn = execution.wrap(agent.get_value, dynamic=True)
    policy_fn = execution.wrap(agent.get_policy_and_value)
    update = execution.wrap(agent.update_popart)
    with torch.no_grad():
        raw_values = value_fn(observations.flatten(0, 1)).reshape(4, 2).clone()
        raw_tail = value_fn(tail).flatten().clone()
        raw_finals = value_fn(torch.as_tensor(finals, device="cuda").flatten(0, 1)).reshape(2, 2).clone()
        update(torch.linspace(100.0, 260.0, 8, device="cuda"), 0.6, 1e-2, 1e6)
        alpha, beta, public_values = policy_fn(observations.flatten(0, 1))
        torch.testing.assert_close(public_values.reshape(4, 2), raw_values, rtol=4e-5, atol=3e-5)
        physical_actions = torch.full((8, 6), 0.25, device="cuda")
        action, logprob, _, action_values = agent.get_action_and_value(observations.flatten(0, 1), physical_actions)
        torch.testing.assert_close(action_values.reshape(4, 2), raw_values, rtol=4e-5, atol=3e-5)
        torch.testing.assert_close(action, physical_actions)
        native = (physical_actions - agent.action_low) / agent.action_scale
        torch.testing.assert_close(logprob, agent.action_logprob(alpha, beta, native), rtol=2e-5, atol=2e-6)
        truncation_values = cache.resolve(value_fn, "cuda")
        expected_finals = torch.zeros_like(rewards)
        expected_finals[0, 1] = raw_finals[0, 1]
        expected_finals[2, 0] = raw_finals[1, 0]
        torch.testing.assert_close(truncation_values, expected_finals, rtol=4e-5, atol=3e-5)
        gamma, lam = 0.9, 0.8
        # Hand-expanded paths include termination, truncation, continuing trace,
        # and rollout tail; no reward from an autoreset episode crosses a boundary.
        a10 = rewards[1, 0] - raw_values[1, 0]
        a31 = rewards[3, 1] - raw_values[3, 1]
        a21 = rewards[2, 1] + gamma * raw_values[3, 1] - raw_values[2, 1] + gamma * lam * a31
        a11 = rewards[1, 1] + gamma * raw_values[2, 1] - raw_values[1, 1] + gamma * lam * a21
        expected_advantages = torch.stack((
            torch.stack((rewards[0, 0] + gamma * raw_values[1, 0] - raw_values[0, 0] + gamma * lam * a10,
                         rewards[0, 1] + gamma * raw_finals[0, 1] - raw_values[0, 1])),
            torch.stack((a10, a11)),
            torch.stack((rewards[2, 0] + gamma * raw_finals[1, 0] - raw_values[2, 0], a21)),
            torch.stack((rewards[3, 0] + gamma * raw_tail[0] - raw_values[3, 0], a31)),
        ))
        gae = get_gae_fn(compiled=execution.compiled, mode="default")
        advantages, returns = gae(
            rewards, public_values.reshape(4, 2), terms, truncs, truncation_values,
            value_fn(tail).flatten(), gamma, lam,
        )
        torch.testing.assert_close(advantages, expected_advantages, rtol=4e-5, atol=4e-5)
        torch.testing.assert_close(returns, expected_advantages + raw_values, rtol=4e-5, atol=4e-5)
        frozen_advantages, frozen_returns = advantages.clone(), returns.clone()
        update(frozen_returns, 0.5, 1e-2, 1e6)
        torch.testing.assert_close(frozen_advantages, expected_advantages, rtol=4e-5, atol=4e-5)
        torch.testing.assert_close(frozen_returns, expected_advantages + raw_values, rtol=4e-5, atol=4e-5)


def test_checkpoint_restores_the_entire_frame_and_frame_changes_leave_adam_moments_untouched(execution):
    args = Args()
    agent = _agent(args)
    observations = torch.randn(8, 17, device="cuda")
    actions = torch.linspace(0.2, 0.8, 48, device="cuda").reshape(8, 6)
    advantages = torch.tensor([2.0, -1.0, 3.0, -2.0, 4.0, -3.0, 5.0, -4.0], device="cuda")
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, actions)
        returns = values.flatten() + torch.linspace(1.0, 4.0, 8, device="cuda")
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3, eps=1e-5, fused=True)
    loss_fn = execution.wrap(ppo_loss)
    loss, _ = loss_fn(agent, observations, actions, old_logprobs, advantages, returns, values.flatten(), args)
    loss.backward()
    optimizer.step()
    agent.normalize_matrices()
    assert torch.count_nonzero(optimizer.state[agent.critic.readout_gain]["exp_avg"])
    moments = {
        parameter: {name: value.clone() for name, value in state.items()}
        for parameter, state in optimizer.state.items()
    }
    update = execution.wrap(agent.update_popart)
    policy = execution.wrap(agent.get_policy_and_value)
    with torch.no_grad():
        before = tuple(value.clone() for value in policy(observations))
        update(torch.linspace(-20.0, 80.0, 8, device="cuda"), 0.4, 1e-2, 1e6)
        update(torch.linspace(5.0, 17.0, 8, device="cuda"), 0.3, 1e-2, 1e6)
        for actual, expected in zip(policy(observations), before, strict=True):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    # This promises unchanged Adam state, not coordinate-invariant Adam updates.
    for parameter, state in optimizer.state.items():
        for name, actual in state.items():
            torch.testing.assert_close(actual, moments[parameter][name], rtol=0, atol=0)

    checkpoint = io.BytesIO()
    torch.save(agent.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = Agent(_spaces()).cuda()
    restored.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    restored_policy = execution.wrap(restored.get_policy_and_value)
    restored_update = execution.wrap(restored.update_popart)
    with torch.no_grad():
        for actual, expected in zip(restored_policy(observations), policy(observations), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            restored.get_normalized_value(observations), agent.get_normalized_value(observations), rtol=0, atol=0,
        )
        # Another frame update exercises saved sq_mean as well as mean/std/shift;
        # comparing only predictions immediately after loading would miss it.
        targets = torch.linspace(-70.0, 90.0, 8, device="cuda")
        expected_targets = update(targets, 0.2, 1e-2, 1e6)
        restored_targets = restored_update(targets, 0.2, 1e-2, 1e6)
        torch.testing.assert_close(restored_targets, expected_targets, rtol=0, atol=0)
        agent.normalize_matrices()
        restored.normalize_matrices()
        for actual, expected, original in zip(restored_policy(observations), policy(observations), before, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, original, rtol=3e-5, atol=2e-5)

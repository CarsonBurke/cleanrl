"""Scalar Peri target/cadence regressions; CUDA execution must be queued via mlq."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_scalar_peri_no_final_v8 as baseline
from cleanrl import ppo_continuous_action_peri_critic_targets_v19 as trainer
from cleanrl.shared.ppo_loop import get_gae_fn
from test_ppo_normres_twohot import device

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required"),
]


def _agent(module, device):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-3.0, 1.0], np.float32), np.array([2.0, 5.0], np.float32)
        ),
    )
    torch.manual_seed(1)
    with torch.device(device):
        return module.Agent(spaces)


def _rollout(device, steps=8):
    time = torch.arange(steps, device=device, dtype=torch.float32).unsqueeze(1)
    environment = torch.arange(3, device=device, dtype=torch.float32).unsqueeze(0)
    rewards = 0.3 + 0.2 * time - 0.1 * environment
    values = -0.7 + 0.13 * time + 0.4 * environment
    next_values = 2.1 - 0.07 * time + 0.6 * environment
    terminations = torch.zeros_like(rewards)
    truncations = torch.zeros_like(rewards)
    terminations[1, 0] = 1
    truncations[3, 1] = 1
    terminations[5, 2] = truncations[5, 2] = 1
    # Final transition values deliberately differ from the following reset value.
    next_values[3, 1] = 17.0
    values[4, 1] = -11.0
    next_values[1, 0] = 91.0
    return rewards, values, terminations, truncations, next_values


def _reference_targets(rewards, values, terminations, truncations, next_values, gamma, gae_lambda, trace_steps):
    """Independent finite sum of discounted TD errors, not the reverse recurrence."""
    r, v, terminal, truncated, nv = [
        tensor.detach().cpu().double().tolist()
        for tensor in (rewards, values, terminations, truncations, next_values)
    ]
    steps, environments = rewards.shape
    advantages = [[0.0] * environments for _ in range(steps)]
    for start in range(steps):
        for environment in range(environments):
            discount = 1.0
            for end in range(start, steps):
                bootstrap = 0.0 if terminal[end][environment] else gamma * nv[end][environment]
                advantages[start][environment] += discount * (
                    r[end][environment] + bootstrap - v[end][environment]
                )
                synthetic_cut = trace_steps > 0 and (end + 1) % trace_steps == 0
                if terminal[end][environment] or truncated[end][environment] or synthetic_cut:
                    break
                discount *= gamma * gae_lambda
    advantages = torch.tensor(advantages, device=rewards.device, dtype=rewards.dtype)
    return advantages, advantages + values


@pytest.mark.parametrize(("steps", "trace_steps"), [(8, 0), (8, 3), (80, 39)])
def test_critic_trace_matches_independent_sum_with_real_transition_bootstraps(device, steps, trace_steps):
    inputs = _rollout(device, steps)
    rewards, values, terminations, truncations, next_values = inputs
    untouched = [tensor.clone() for tensor in inputs]
    boundaries = trainer.critic_trace_boundaries(truncations, trace_steps)
    actual = get_gae_fn(explicit_next_values=True)(
        rewards, values, terminations, boundaries, next_values, 0.99, 0.95
    )
    expected = _reference_targets(*inputs, 0.99, 0.95, trace_steps)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    # A termination suppresses bootstrap; a timeout uses its final, not reset, state.
    torch.testing.assert_close(actual[1][1, 0], rewards[1, 0])
    torch.testing.assert_close(actual[1][3, 1], rewards[3, 1] + 0.99 * next_values[3, 1])
    for tensor, original in zip(inputs, untouched):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.parametrize("clip_vloss", [False, True])
def test_scalar_value_loss_matches_clipped_mse_and_detaches_targets(device, clip_vloss):
    prediction = torch.tensor([0.8, -0.8, 0.3, -0.3], device=device, requires_grad=True)
    targets = torch.tensor([1.0, -1.0, -1.0, 1.0], device=device, requires_grad=True)
    old_values = torch.zeros_like(prediction)
    actual = trainer.scalar_value_loss(prediction, targets, old_values, 0.2, clip_vloss)
    residual_squares = [0.64, 0.64, 1.69, 1.69] if clip_vloss else [0.04, 0.04, 1.69, 1.69]
    expected = torch.tensor(0.5 * sum(residual_squares) / 4, device=device)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    gradient = [0.0, 0.0, 1.3 / 4, -1.3 / 4] if clip_vloss else [-0.2 / 4, 0.2 / 4, 1.3 / 4, -1.3 / 4]
    torch.testing.assert_close(prediction.grad, torch.tensor(gradient, device=device))
    assert targets.grad is None


def _policy_batch(agent, device):
    observations = torch.linspace(-1.4, 1.7, 24 * 17, device=device).reshape(24, 17)
    native_actions = torch.linspace(0.05, 0.95, 48, device=device).reshape(24, 2)
    with torch.no_grad():
        alpha, beta, old_values = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native_actions)
        old_logprobs -= torch.tensor([0.0, 0.6, -0.6], device=device).repeat(8)
        old_values = old_values.flatten().clone()
    advantages = torch.tensor([2.0, -3.0, 7.0, -1.0], device=device).repeat(6)
    targets = old_values + torch.linspace(-2.0, 3.0, 24, device=device)
    return observations, native_actions, old_logprobs, advantages, targets, old_values


def test_baseline_scalar_ppo_loss_and_parameter_gradients_match_v8(device):
    original = _agent(baseline, device)
    agent = _agent(trainer, device)
    agent.load_state_dict(original.state_dict())
    batch = _policy_batch(original, device)
    args = trainer.Args()
    expected_loss, expected_metrics = baseline.ppo_loss(original, *batch, args)
    actual_loss, actual_metrics = trainer.ppo_loss(agent, *batch, args)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual_metrics, expected_metrics, rtol=0, atol=0)
    expected_loss.backward()
    actual_loss.backward()
    for (name, parameter), (expected_name, expected_parameter) in zip(
        agent.named_parameters(), original.named_parameters(), strict=True
    ):
        assert name == expected_name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)


def test_joint_step_keeps_full_actor_loss_but_selects_critic_samples(device):
    agent = _agent(trainer, device)
    control = _agent(baseline, device)
    control.load_state_dict(agent.state_dict())
    batch = _policy_batch(agent, device)
    observations, actions, logprobs, advantages, targets, old_values = batch
    indices = torch.tensor([1, 7, 8, 19, 23], device=device)
    args = trainer.Args()
    actual_loss, _ = trainer.ppo_loss(agent, *batch, args, critic_indices=indices)
    # The original actor loss remains a full-batch objective.
    actor_args = SimpleNamespace(**vars(args))
    actor_args.vf_coef = 0.0
    actor_loss, _ = baseline.ppo_loss(control, *batch, actor_args)
    selected_prediction = control.get_value(observations[indices]).flatten()
    selected_old = old_values[indices]
    selected_targets = targets[indices]
    clipped = selected_old + (selected_prediction - selected_old).clamp(-args.clip_coef, args.clip_coef)
    critic_loss = 0.5 * torch.maximum(
        (selected_prediction - selected_targets).square(), (clipped - selected_targets).square()
    ).mean()
    expected_loss = actor_loss + args.vf_coef * critic_loss
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
    actual_loss.backward()
    expected_loss.backward()
    for parameter, expected in zip(agent.parameters(), control.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize(("steps", "trace_steps"), [(8, 3), (2048, 39)])
def test_compiled_rollout_targets_own_full_actor_gae_across_short_calls(device, steps, trace_steps):
    inputs = _rollout(device, steps=steps)
    expected_actor, expected_long_returns = _reference_targets(*inputs, 0.99, 0.95, 0)
    _, expected_short_returns = _reference_targets(*inputs, 0.99, 0.95, trace_steps)
    assert not torch.allclose(expected_long_returns, expected_short_returns)
    torch.compiler.reset()
    try:
        gae = get_gae_fn(compiled=True, explicit_next_values=True)
        for _ in range(4):
            torch.compiler.cudagraph_mark_step_begin()
            actor_advantages, long_returns, critic_targets = trainer.compute_rollout_targets(
                gae, *inputs, 0.99, 0.95, trace_steps
            )
            torch.testing.assert_close(actor_advantages, expected_actor, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(long_returns, expected_long_returns, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(critic_targets, expected_short_returns, rtol=2e-5, atol=2e-5)
            # Replay the same graph with different targets before consuming the saved actor data.
            altered_inputs = (inputs[0] + 13.0, *inputs[1:])
            torch.compiler.cudagraph_mark_step_begin()
            trainer.compute_rollout_targets(gae, *altered_inputs, 0.99, 0.95, trace_steps)
            torch.testing.assert_close(actor_advantages, expected_actor, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(long_returns, expected_long_returns, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(critic_targets, expected_short_returns, rtol=2e-5, atol=2e-5)
        full_actor, full_returns, full_critic = trainer.compute_rollout_targets(
            gae, *inputs, 0.99, 0.95, 0
        )
        torch.testing.assert_close(full_actor, expected_actor, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(full_returns, expected_long_returns, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(full_critic, expected_long_returns, rtol=2e-5, atol=2e-5)
    finally:
        torch.compiler.reset()


def test_compiled_fresh_critic_minibatches_leave_populated_actor_adam_unchanged(device):
    agent = _agent(trainer, device)
    args = trainer.Args()
    args.critic_minibatches = 13
    args.critic_trace_steps = 3
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    rewards, _, terminations, truncations, _ = _rollout(device, steps=13)
    observations = torch.linspace(-1.4, 1.7, 39 * 17, device=device).reshape(39, 17)
    transition_observations = observations.roll(-3, dims=0) + 0.37
    # The actual timeout transition must not be replaced with the next/reset row.
    transition_observations[10] = 2.5
    native_actions = torch.linspace(0.05, 0.95, 78, device=device).reshape(39, 2)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native_actions).clone()
    permutation = torch.randperm(39, device=device)
    chunks = torch.tensor_split(permutation, args.critic_minibatches)
    boundaries = trainer.critic_trace_boundaries(truncations, args.critic_trace_steps)

    def statistics(obs, next_obs):
        return trainer.critic_statistics(agent, obs, next_obs)

    def critic_loss(obs, targets, old_values):
        return args.vf_coef * trainer.scalar_value_loss(
            agent.get_value(obs).flatten(), targets, old_values, args.clip_coef, args.clip_vloss
        )

    torch.compiler.reset()
    try:
        compiled_statistics = torch.compile(statistics, fullgraph=True, mode="reduce-overhead")
        compiled_loss = torch.compile(critic_loss, fullgraph=True, mode="reduce-overhead")
        gae = get_gae_fn(compiled=True, explicit_next_values=True)
        previous_targets = None
        actor_snapshot = None
        actor_advantages = long_returns = None
        frozen_actor_advantages = frozen_long_returns = None
        for substep, indices in enumerate(chunks):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                flat_values, flat_next_values = compiled_statistics(observations, transition_observations)
                values = flat_values.reshape_as(rewards).clone()
                next_values = flat_next_values.reshape_as(rewards).clone()
                torch.testing.assert_close(
                    values.flatten(), agent.get_value(observations).flatten(), rtol=2e-4, atol=2e-5
                )
                torch.testing.assert_close(
                    next_values.flatten(), agent.get_value(transition_observations).flatten(),
                    rtol=2e-4, atol=2e-5,
                )
                inputs = rewards, values, terminations, truncations, next_values
                if substep == 0:
                    actor_advantages, long_returns, targets = trainer.compute_rollout_targets(
                        gae, *inputs, args.gamma, args.gae_lambda, args.critic_trace_steps
                    )
                    frozen_actor_advantages = actor_advantages.clone()
                    frozen_long_returns = long_returns.clone()
                else:
                    _, generated_targets = gae(
                        rewards, values, terminations, boundaries, next_values, args.gamma, args.gae_lambda
                    )
                    targets = generated_targets.clone()
                _, expected_targets = _reference_targets(
                    *inputs, args.gamma, args.gae_lambda, args.critic_trace_steps
                )
                torch.testing.assert_close(targets, expected_targets, rtol=2e-5, atol=2e-5)
                if previous_targets is not None:
                    assert not torch.allclose(targets, previous_targets, rtol=1e-5, atol=1e-6)
                previous_targets = targets.clone()
                prediction_before = agent.get_value(observations).clone()

            assert actor_advantages is not None and long_returns is not None
            assert frozen_actor_advantages is not None and frozen_long_returns is not None
            if substep == 0:
                loss, _ = trainer.ppo_loss(
                    agent, observations, native_actions, old_logprobs, actor_advantages.flatten(),
                    targets.flatten(), values.flatten(), args, critic_indices=indices,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                actor_snapshot = [
                    (parameter.detach().clone(), {
                        key: value.clone() for key, value in optimizer.state[parameter].items()
                    })
                    for parameter in agent.actor.parameters()
                ]
                assert any(state["exp_avg"].abs().max() > 0 for _, state in actor_snapshot)
            else:
                assert actor_snapshot is not None
                trainer.critic_step(
                    agent, optimizer, observations[indices], targets.flatten()[indices],
                    values.flatten()[indices], args, loss_model=compiled_loss,
                )
                for parameter, (saved_parameter, saved_state) in zip(
                    agent.actor.parameters(), actor_snapshot, strict=True
                ):
                    torch.testing.assert_close(parameter, saved_parameter, rtol=0, atol=0)
                    for key, saved in saved_state.items():
                        torch.testing.assert_close(optimizer.state[parameter][key], saved, rtol=0, atol=0)
            with torch.no_grad():
                assert not torch.allclose(agent.get_value(observations), prediction_before)
            torch.testing.assert_close(actor_advantages, frozen_actor_advantages, rtol=0, atol=0)
            torch.testing.assert_close(long_returns, frozen_long_returns, rtol=0, atol=0)
            for parameter in agent.actor.parameters():
                assert optimizer.state[parameter]["step"].item() == 1
            for parameter in agent.critic.parameters():
                assert optimizer.state[parameter]["step"].item() == substep + 1
    finally:
        torch.compiler.reset()

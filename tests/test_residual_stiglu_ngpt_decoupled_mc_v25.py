"""CUDA contracts for decoupled actor GAE and boundary-aware critic targets."""
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_decoupled_mc_v25 import (
    Agent,
    Args,
    compute_training_targets,
    ppo_loss,
    scalar_value_loss,
)
from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.fail("These contracts require CUDA; run them in the exclusive mlq test job.")
    configure_runtime(matmul_precision="highest", allow_tf32=False)


@pytest.fixture(scope="module")
def gae_fn():
    return get_gae_fn(compiled=True, mode="default")


@pytest.fixture(scope="module")
def compiled_ppo_loss():
    return torch.compile(ppo_loss, fullgraph=True, options={"triton.cudagraphs": False})


@pytest.fixture(scope="module")
def compiled_value_loss():
    return torch.compile(scalar_value_loss, fullgraph=True, options={"triton.cudagraphs": False})


@pytest.fixture
def rollout():
    rewards = torch.arange(1, 25, device="cuda", dtype=torch.float32).reshape(6, 4)
    terminations = torch.zeros_like(rewards)
    truncations = torch.zeros_like(rewards)
    # Env 0: termination, then truncation, then an unfinished reset episode.
    terminations[1, 0] = 1
    truncations[4, 0] = 1
    # Env 1: truncation followed by a termination at the rollout's last step.
    truncations[2, 1] = 1
    terminations[5, 1] = 1
    # Env 2 never ends; env 3 has simultaneous flags, then an unfinished tail.
    terminations[0, 3] = 1
    truncations[0, 3] = 1
    truncation_values = torch.full_like(rewards, 999.0)
    truncation_values[4, 0] = 20.0
    truncation_values[2, 1] = 30.0
    truncation_values[0, 3] = 1000.0
    return SimpleNamespace(
        rewards=rewards,
        values=(rewards - 8.0) / 3.0,
        terminations=terminations,
        truncations=truncations,
        truncation_values=truncation_values,
        tail_value=torch.tensor([40.0, 50.0, 60.0, 70.0], device="cuda"),
        gamma=0.5,
        gae_lambda=0.95,
    )


def _targets(gae_fn, rollout, mode):
    return compute_training_targets(
        gae_fn,
        rollout.rewards,
        rollout.values,
        rollout.terminations,
        rollout.truncations,
        rollout.truncation_values,
        rollout.tail_value,
        rollout.gamma,
        rollout.gae_lambda,
        mode,
    )


def test_every_critic_mode_keeps_the_original_actor_gae_exactly(gae_fn, rollout):
    expected_advantages, expected_returns = gae_fn(
        rollout.rewards,
        rollout.values,
        rollout.terminations,
        rollout.truncations,
        rollout.truncation_values,
        rollout.tail_value,
        rollout.gamma,
        rollout.gae_lambda,
    )
    for mode in ("gae", "vapo", "episode_mc"):
        advantages, returns, mask = _targets(gae_fn, rollout, mode)
        torch.testing.assert_close(advantages, expected_advantages, rtol=0, atol=0)
        if mode == "gae":
            torch.testing.assert_close(returns, expected_returns, rtol=0, atol=0)
        if mode != "episode_mc":
            torch.testing.assert_close(mask, torch.ones_like(rollout.rewards, dtype=torch.bool))


def test_vapo_lambda_one_uses_only_the_correct_boundary_bootstrap(gae_fn, rollout):
    _, returns, _ = _targets(gae_fn, rollout, "vapo")
    # Hand-computed discounted rewards, gamma=.5. For example env 0, t=2:
    # 9 + .5*13 + .25*17 + .125*20 = 22.25. Reset rewards never cross a boundary.
    # The env-3 simultaneous termination/truncation must ignore its bootstrap.
    expected = torch.tensor(
        [
            [3.5, 11.25, 13.96875, 4.0],
            [5.0, 18.5, 21.9375, 24.1875],
            [22.25, 25.0, 29.875, 32.375],
            [26.5, 28.5, 37.75, 40.75],
            [27.0, 29.0, 45.5, 49.5],
            [41.0, 22.0, 53.0, 59.0],
        ],
        device="cuda",
    )
    torch.testing.assert_close(returns, expected, rtol=2e-6, atol=2e-6)


def test_episode_mc_uses_full_observed_reward_suffixes_without_value_bootstrap(gae_fn, rollout):
    _, returns, mask = _targets(gae_fn, rollout, "episode_mc")
    expected_mask = torch.tensor(
        [
            [True, True, False, True],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [False, True, False, False],
        ],
        device="cuda",
    )
    # Zero placeholders are deliberately not assertions about invalid targets.
    # Episode prefixes before the rollout are irrelevant to these future sums.
    expected = torch.tensor(
        [
            [3.5, 7.5, 0.0, 4.0],
            [5.0, 11.0, 0.0, 0.0],
            [19.75, 10.0, 0.0, 0.0],
            [21.5, 28.5, 0.0, 0.0],
            [17.0, 29.0, 0.0, 0.0],
            [0.0, 22.0, 0.0, 0.0],
        ],
        device="cuda",
    )
    torch.testing.assert_close(mask, expected_mask)
    torch.testing.assert_close(returns[mask], expected[mask], rtol=0, atol=0)

    different_values = SimpleNamespace(**vars(rollout))
    different_values.values = -17.0 * rollout.values + 83.0
    different_values.truncation_values = -12.0 * rollout.truncation_values + 37.0
    different_values.tail_value = -7.0 * rollout.tail_value + 123.0
    _, changed_returns, changed_mask = _targets(gae_fn, different_values, "episode_mc")
    torch.testing.assert_close(changed_returns, returns, rtol=0, atol=0)
    torch.testing.assert_close(changed_mask, mask)


def test_episode_mc_never_includes_rewards_from_the_autoreset_episode(gae_fn, rollout):
    _, before, before_mask = _targets(gae_fn, rollout, "episode_mc")
    changed = SimpleNamespace(**vars(rollout))
    changed.rewards = rollout.rewards.clone()
    changed.rewards[2, 0] += 10000.0
    changed.rewards[3, 1] -= 10000.0
    changed.rewards[1, 3] += 10000.0
    _, after, after_mask = _targets(gae_fn, changed, "episode_mc")
    torch.testing.assert_close(after_mask, before_mask)
    torch.testing.assert_close(after[:2, 0], before[:2, 0], rtol=0, atol=0)
    torch.testing.assert_close(after[:3, 1], before[:3, 1], rtol=0, atol=0)
    torch.testing.assert_close(after[0, 3], before[0, 3], rtol=0, atol=0)
    torch.testing.assert_close(after[2, 0], before[2, 0] + 10000.0, rtol=0, atol=0)
    torch.testing.assert_close(after[3, 1], before[3, 1] - 10000.0, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["mse", "symlog_mean", "symlog_mse"])
def test_masked_scalar_loss_averages_only_valid_samples_and_handles_empty_masks(compiled_value_loss, mode):
    prediction = torch.tensor([0.3, 8.0, -0.4, -7.0], device="cuda", requires_grad=True)
    targets = torch.tensor([2.0, 1000.0, -3.0, -1000.0], device="cuda", requires_grad=True)
    mask = torch.tensor([True, False, True, False], device="cuda")
    if mode == "mse":
        residuals = (0.3 - 2.0, -0.4 + 3.0)
        expected_loss = sum(residual**2 for residual in residuals) / 4.0
    elif mode == "symlog_mse":
        residuals = (0.3 - math.log1p(2.0), -0.4 + math.log1p(3.0))
        expected_loss = sum(residual**2 for residual in residuals) / 4.0
    else:
        residuals = (math.expm1(0.3) - 2.0, -math.expm1(0.4) + 3.0)
        expected_loss = sum(
            math.expm1(abs(z)) - abs(z) - target * z
            + (abs(target) + 1.0) * math.log1p(abs(target)) - abs(target)
            for z, target in ((0.3, 2.0), (-0.4, -3.0))
        ) / 2.0
    loss = compiled_value_loss(prediction, targets, mode, mask=mask)
    torch.testing.assert_close(loss, prediction.new_tensor(expected_loss), rtol=2e-6, atol=2e-6)
    loss.backward()
    expected_gradient = prediction.new_tensor([residuals[0] / 2.0, 0.0, residuals[1] / 2.0, 0.0])
    torch.testing.assert_close(prediction.grad, expected_gradient, rtol=2e-6, atol=2e-6)
    assert targets.grad is None

    prediction.grad = None
    empty_loss = compiled_value_loss(prediction, targets, mode, mask=torch.zeros_like(mask))
    torch.testing.assert_close(empty_loss, torch.zeros_like(empty_loss), rtol=0, atol=0)
    empty_loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), rtol=0, atol=0)


@pytest.fixture
def policy_batch(rollout):
    args = Args(critic_loss="mse", clip_vloss=False)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    torch.manual_seed(1)
    agent = Agent(spaces, args).cuda()
    agent.normalize_matrices()
    batch_size = rollout.rewards.numel()
    observations = torch.randn(batch_size, 17, device="cuda")
    native_actions = torch.linspace(0.15, 0.85, batch_size * 6, device="cuda").reshape(batch_size, 6)
    policy = torch.compile(agent.get_policy_and_value, fullgraph=True, options={"triton.cudagraphs": False})
    with torch.no_grad():
        alpha, beta, values = policy(observations)
        ratios = torch.linspace(0.7, 1.3, batch_size, device="cuda")
        old_logprobs = agent.action_logprob(alpha, beta, native_actions) - ratios.log()
    return SimpleNamespace(
        agent=agent,
        args=args,
        observations=observations,
        native_actions=native_actions,
        old_logprobs=old_logprobs,
        predictions=values.reshape(-1),
    )


def test_critic_targets_change_only_critic_gradients_before_joint_clipping(
    gae_fn, compiled_ppo_loss, rollout, policy_batch,
):
    # This does not assert equal actor updates after joint global clipping:
    # different critic gradient norms can change the shared clipping factor.
    batch = policy_batch
    reference_actor = None
    reference_critic = None
    for mode in ("gae", "vapo", "episode_mc"):
        advantages, returns, mask = _targets(gae_fn, rollout, mode)
        batch.agent.zero_grad(set_to_none=True)
        loss, metrics = compiled_ppo_loss(
            batch.agent,
            batch.observations,
            batch.native_actions,
            batch.old_logprobs,
            advantages.flatten(),
            returns.flatten(),
            rollout.values.flatten(),
            batch.args,
            critic_mask=mask.flatten(),
        )
        expected_value_loss = 0.5 * ((batch.predictions - returns.flatten())[mask.flatten()] ** 2).mean()
        torch.testing.assert_close(metrics[1], expected_value_loss, rtol=2e-5, atol=2e-5)
        loss.backward()
        actor_gradients = [parameter.grad.detach().clone() for parameter in batch.agent.actor.parameters()]
        critic_gradients = [parameter.grad.detach().clone() for parameter in batch.agent.critic.parameters()]
        assert all(torch.isfinite(gradient).all() for gradient in (*actor_gradients, *critic_gradients))
        if reference_actor is None:
            assert any(torch.count_nonzero(gradient) > 0 for gradient in actor_gradients)
            reference_actor = actor_gradients
            reference_critic = critic_gradients
        else:
            for actual, expected in zip(actor_gradients, reference_actor, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert any(
                not torch.equal(actual, expected)
                for actual, expected in zip(critic_gradients, reference_critic, strict=True)
            )


def test_no_observed_episode_end_still_trains_actor_with_zero_critic_loss(
    gae_fn, compiled_ppo_loss, rollout, policy_batch,
):
    rollout.terminations.zero_()
    rollout.truncations.zero_()
    advantages, returns, mask = _targets(gae_fn, rollout, "episode_mc")
    torch.testing.assert_close(mask, torch.zeros_like(rollout.rewards, dtype=torch.bool))
    batch = policy_batch
    loss, metrics = compiled_ppo_loss(
        batch.agent,
        batch.observations,
        batch.native_actions,
        batch.old_logprobs,
        advantages.flatten(),
        returns.flatten(),
        rollout.values.flatten(),
        batch.args,
        critic_mask=mask.flatten(),
    )
    torch.testing.assert_close(metrics[1], torch.zeros_like(metrics[1]), rtol=0, atol=0)
    assert torch.isfinite(loss)
    loss.backward()
    actor_gradients = [parameter.grad for parameter in batch.agent.actor.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in actor_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in actor_gradients)
    for parameter in batch.agent.critic.parameters():
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)

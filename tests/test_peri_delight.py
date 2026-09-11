"""Physical-density truegate contracts; CUDA model checks run only through mlq."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_scalar_peri_abs_delight_v12 as trainer
from test_ppo_normres_twohot import device

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required"),
]


def _agent(device, widths=(0.5, 2.0)):
    widths = np.asarray(widths, dtype=np.float32)
    low = np.arange(len(widths), dtype=np.float32) - 3.0
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (5,), np.float32),
        single_action_space=gym.spaces.Box(low, low + widths),
    )
    with torch.device(device):
        return trainer.Agent(spaces)


def test_truegate_sign_zero_and_symmetric_surprisal_clipping(device):
    # Cover all four sign quadrants, both zero boundaries, and both saturated
    # surprisal tails. Clipping the product instead of surprisal fails the tails.
    advantages = torch.tensor([2.0, -2.0, 2.0, -2.0, 0.0, 7.0, 0.2, -0.2], device=device)
    logprobs = torch.tensor([-1.0, -1.0, 1.0, 1.0, -3.0, 0.0, -100.0, 100.0], device=device)
    weights = trainer.truegate_weights(advantages, logprobs, eta=1.0, surprisal_clip=10.0)
    expected_logits = torch.tensor([2.0, -2.0, -2.0, 2.0, 0.0, 0.0, 2.0, 2.0], device=device)
    torch.testing.assert_close(weights, 1.0 / (1.0 + torch.exp(-expected_logits)))
    # The gate is absolute per-sample density, not a batch rank or centered score.
    singleton = trainer.truegate_weights(advantages[:1], logprobs[:1], 1.0, 10.0)
    torch.testing.assert_close(singleton, weights[:1], rtol=0, atol=0)


def test_physical_density_jacobian_is_applied_exactly_once(device):
    # Beta(2,2) has native density 1.5 at its midpoint. Physical widths on
    # opposite sides of 1.5 cross density=1 without changing the native sample.
    advantages = torch.tensor([1.0, -1.0], device=device)
    for width, physical_density in [(0.5, 3.0), (1.5, 1.0), (2.0, 0.75)]:
        agent = _agent(device, widths=(width,))
        concentrations = torch.full((2, 1), 2.0, device=device)
        native_actions = torch.full((2, 1), 0.5, device=device)
        physical_logprobs = agent.action_logprob(concentrations, concentrations, native_actions)
        expected_logprob = torch.tensor(physical_density, device=device).log().expand(2)
        torch.testing.assert_close(physical_logprobs, expected_logprob, atol=2e-7, rtol=2e-6)
        weights = trainer.truegate_weights(advantages, physical_logprobs, 1.0, 10.0)
        # For U=+1, sigmoid(-log p)=1/(1+p); U=-1 is its complement.
        expected = torch.tensor(
            [1.0 / (1.0 + physical_density), physical_density / (1.0 + physical_density)],
            device=device,
        )
        torch.testing.assert_close(weights, expected)


@pytest.mark.parametrize("clip_vloss", [False, True], ids=["scalar-mse", "scalar-clipped-mse"])
def test_fixed_gate_changes_only_policy_gradient_and_preserves_value_targets(device, clip_vloss):
    agent = _agent(device)
    observations = torch.linspace(-1.0, 1.0, 40, device=device).reshape(8, 5)
    native_actions = torch.linspace(0.1, 0.9, 16, device=device).reshape(8, 2)
    with torch.no_grad():
        alpha, beta, initial_values = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native_actions)
        # Exercise both PPO clipping boundaries as well as the unclipped region.
        old_logprobs = old_logprobs - observations.new_tensor([0.0, 0.6, -0.6, 0.0] * 2)
        initial_values = initial_values.flatten()
    advantages = observations.new_tensor([2.0, 7.0, -3.0, -5.0, 1.0, -2.0, 4.0, -1.0])
    # Give the gate differentiable inputs deliberately: even outside the rollout's
    # no_grad context it must never backpropagate into GAE or the old policy.
    advantages.requires_grad_()
    old_logprobs.requires_grad_()
    weights = trainer.truegate_weights(advantages, old_logprobs, 1.0, 10.0)
    assert not weights.requires_grad
    gated_advantages = advantages.detach() * weights
    fixed_logprobs = old_logprobs.detach().clone()
    direction = observations.new_tensor([1.0, -1.0] * 4)
    old_values = initial_values + direction
    # Half the samples choose the clipped value error, half the unclipped error.
    targets = initial_values + observations.new_tensor([-2.0, -3.0] * 4)
    saved_targets = targets.clone()
    args = trainer.Args(clip_vloss=clip_vloss, ent_coef=0.03, vf_coef=0.7)
    parameters = tuple(agent.actor.parameters()) + tuple(agent.critic.parameters())
    actor_count = len(tuple(agent.actor.parameters()))
    raw_loss, raw_metrics = trainer.ppo_loss(
        agent, observations, native_actions, fixed_logprobs, advantages.detach(), targets, old_values, args
    )
    raw_gradients = torch.autograd.grad(raw_loss, parameters)
    gated_loss, gated_metrics = trainer.ppo_loss(
        agent, observations, native_actions, fixed_logprobs, gated_advantages, targets, old_values, args
    )
    gated_gradients = torch.autograd.grad(gated_loss, parameters)
    torch.testing.assert_close(gated_metrics[1:], raw_metrics[1:], rtol=0, atol=0)
    torch.testing.assert_close(gated_gradients[actor_count:], raw_gradients[actor_count:], rtol=0, atol=0)
    assert any(not torch.allclose(a, b) for a, b in zip(gated_gradients[:actor_count], raw_gradients[:actor_count]))
    assert sum(gradient.square().sum() for gradient in gated_gradients[actor_count:]) > 0
    errors = (initial_values - targets).square()
    if clip_vloss:
        clipped_values = old_values + (initial_values - old_values).clamp(-args.clip_coef, args.clip_coef)
        errors = torch.maximum(errors, (clipped_values - targets).square())
    torch.testing.assert_close(gated_metrics[1], 0.5 * errors.mean())
    with torch.no_grad():
        current_alpha, current_beta, _ = agent.get_policy_and_value(observations)
        ratio = (agent.action_logprob(current_alpha, current_beta, native_actions) - fixed_logprobs).exp()
        expected_policy_loss = -torch.minimum(
            gated_advantages * ratio,
            gated_advantages * ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef),
        ).mean()
    torch.testing.assert_close(gated_metrics[0], expected_policy_loss)
    torch.testing.assert_close(targets, saved_targets, rtol=0, atol=0)
    assert advantages.grad is None and old_logprobs.grad is None

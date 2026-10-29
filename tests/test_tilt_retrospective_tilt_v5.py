"""Retrospective-tilt contracts; run CUDA work exclusively through mlq."""
from types import SimpleNamespace

import torch

from cleanrl.tilt.ppo_continuous_action_sepclip_retrospective_tilt_v5 import (
    critic_loss, retrospective_sensitivity, value_and_gradient,
)


def test_retrospective_sensitivity_is_the_gradient_of_the_realized_discounted_outcome():
    torch.manual_seed(1)
    steps, envs, state_dim, action_dim, gamma = 7, 3, 4, 2, 0.9
    shape = lambda *dims: 0.5 * torch.randn(steps, envs, *dims, device="cuda", dtype=torch.float64)
    state_map, action_map, reaction = shape(state_dim, state_dim), shape(state_dim, action_dim), shape(action_dim, state_dim)
    reward_state, reward_action = shape(state_dim), shape(action_dim)
    tail = torch.randn(state_dim, device="cuda", dtype=torch.float64)
    start = torch.randn(envs, state_dim, device="cuda", dtype=torch.float64, requires_grad=True)
    nudges = [torch.zeros(envs, action_dim, device="cuda", dtype=torch.float64, requires_grad=True) for _ in range(steps)]

    def outcome(first):
        """Discounted reward from `first` on, with the policy reacting linearly to the state."""
        state, total = start if first == 0 else states[first].detach().requires_grad_(True), 0.0
        origin = state
        for step in range(first, steps):
            action = (reaction[step] @ state[..., None]).squeeze(-1) + nudges[step]
            total = total + gamma ** (step - first) * ((reward_state[step] * state).sum(-1) + (reward_action[step] * action).sum(-1))
            state = (state_map[step] @ state[..., None]).squeeze(-1) + (action_map[step] @ action[..., None]).squeeze(-1)
        return total + gamma ** (steps - first) * (state * tail).sum(-1), origin

    states = [start]
    with torch.no_grad():
        for step in range(steps):
            action = (reaction[step] @ states[-1][..., None]).squeeze(-1)
            states.append((state_map[step] @ states[-1][..., None]).squeeze(-1) + (action_map[step] @ action[..., None]).squeeze(-1))
    by_observation = torch.cat((state_map, reward_state[:, :, None, :]), dim=2)
    by_action = torch.cat((action_map, reward_action[:, :, None, :]), dim=2)
    continuing = torch.ones(steps, envs, device="cuda", dtype=torch.float64)
    # mix = 1 follows the realized trajectory to the rollout's end, where the critic's gradient takes over.
    state_sensitivity, action_sensitivity, bounded = retrospective_sensitivity(
        by_observation, by_action, reaction, tail.expand(steps, envs, state_dim), continuing, gamma, 1.0, 1e9)
    assert bounded == 0.0
    for first in (0, 3, steps - 1):
        total, origin = outcome(first)
        expected_state, expected_action = torch.autograd.grad(total.sum(), (origin, nudges[first]))
        torch.testing.assert_close(state_sensitivity[first], expected_state)
        torch.testing.assert_close(action_sensitivity[first], expected_action)
    # An episode boundary cuts the continuation: only the immediate reward remains.
    continuing[2] = 0.0
    cut = retrospective_sensitivity(by_observation, by_action, reaction, tail.expand(steps, envs, state_dim),
                                    continuing, gamma, 1.0, 1e9)
    torch.testing.assert_close(cut[1][2], reward_action[2])
    torch.testing.assert_close(cut[0][2], reward_state[2] + (reaction[2].transpose(-1, -2) @ reward_action[2][..., None]).squeeze(-1))
    # The bound caps every sensitivity's norm.
    capped = retrospective_sensitivity(by_observation, by_action, reaction, tail.expand(steps, envs, state_dim),
                                       torch.ones_like(continuing), gamma, 1.0, 0.1)
    assert torch.linalg.vector_norm(capped[0], dim=-1).max() <= 0.1 + 1e-9 and capped[2] > 0.0


def test_critic_loss_matches_v4_without_the_slope_term_and_trains_the_input_gradient_with_it():
    torch.manual_seed(1)
    critic = torch.nn.Sequential(torch.nn.Linear(5, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).cuda()
    agent = SimpleNamespace(critic=critic)
    observations, returns = torch.randn(64, 5, device="cuda"), torch.randn(64, device="cuda")
    sensitivity = torch.randn(64, 5, device="cuda")
    value, gradient = value_and_gradient(agent, observations)
    plain = critic_loss(agent, observations, returns, sensitivity, SimpleNamespace(vf_coef=0.5, adjoint_coef=0.0))
    torch.testing.assert_close(plain[0], 0.25 * ((value - returns) ** 2).mean())
    torch.testing.assert_close(plain[2], ((gradient - sensitivity) ** 2).sum(-1).mean() / sensitivity.square().sum(-1).mean())
    loss = critic_loss(agent, observations, returns, sensitivity, SimpleNamespace(vf_coef=0.5, adjoint_coef=1.0))[0]
    torch.testing.assert_close(loss, plain[0] + 0.25 * returns.var() * plain[2])
    # Both objectives are differentiable in the critic: the value-only path keeps its graph after the
    # input gradient is taken, and the slope path backpropagates through that gradient.
    value_only = torch.autograd.grad(plain[0], critic[0].weight)[0]
    with_slope = torch.autograd.grad(loss, critic[0].weight)[0]
    assert torch.isfinite(value_only).all() and not torch.allclose(with_slope, value_only)

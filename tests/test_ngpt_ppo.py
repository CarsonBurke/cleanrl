"""Projected critic steps must not perturb the behavior policy or its Adam state."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_ngpt_spherical_v20 as trainer
from test_ppo_normres_twohot import device

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_critic_projection_preserves_actor_with_populated_adam_state(device):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
    )
    with torch.device(device):
        agent = trainer.Agent(spaces)
    observations = torch.randn(64, 17, device=device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3, eps=1e-5, fused=True)
    # Populate real actor moments before the critic-only update.
    (agent.actor(observations).square().mean() + agent.get_value(observations).square().mean()).backward()
    optimizer.step()
    trainer.project_ngpt_weights(agent)
    actor_before = {name: parameter.detach().clone() for name, parameter in agent.actor.named_parameters()}
    actor_state = {parameter: {key: value.clone() for key, value in optimizer.state[parameter].items()}
                   for parameter in agent.actor.parameters()}
    with torch.no_grad():
        old_values = agent.get_value(observations).flatten().clone()
        targets = old_values + 0.1
        policy_before = agent.actor(observations).clone()
    args = trainer.Args()
    trainer.critic_step(agent, optimizer, observations, targets, old_values, args)
    with torch.no_grad():
        new_values = agent.get_value(observations).flatten()
        assert (new_values - targets).square().mean() < (old_values - targets).square().mean()
        torch.testing.assert_close(agent.actor(observations), policy_before, rtol=0, atol=0)
    for name, parameter in agent.actor.named_parameters():
        torch.testing.assert_close(parameter, actor_before[name], rtol=0, atol=0)
        assert parameter.grad is None
        for key, value in optimizer.state[parameter].items():
            torch.testing.assert_close(value, actor_state[parameter][key], rtol=0, atol=0)
    diagnostics = trainer.geometry_metrics(agent)
    assert diagnostics["geometry/critic_weight_norm_error"] < 3e-7

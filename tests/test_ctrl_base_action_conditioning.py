"""Behavioral contracts for ctrl-base action-conditioned PPO variants."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_action_feedback_v1 as feedback
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_beta_critic_v1 as beta_critic


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-1, -1, -1, -1, -1, -1], np.float32),
            np.array([1, 1, 1, 1, 1, 1], np.float32),
        ),
    )


def _feedback_agent():
    torch.manual_seed(1)
    return feedback.Agent(_spaces(), norm_kind="rms", activation="stiglu").cuda()


def _beta_critic_agent():
    torch.manual_seed(1)
    return beta_critic.Agent(_spaces(), norm_kind="rms", activation="stiglu").cuda()


def test_ctrl_base_actor_uses_previous_native_action():
    agent = _feedback_agent()
    observations = torch.randn((8, 17), device="cuda")
    previous_a = torch.full((8, 6), 0.2, device="cuda")
    previous_b = torch.full((8, 6), 0.8, device="cuda")

    alpha_a, beta_a, _ = agent.get_policy_and_value(observations, torch.cat((observations, previous_a), dim=-1))
    alpha_b, beta_b, _ = agent.get_policy_and_value(observations, torch.cat((observations, previous_b), dim=-1))

    assert agent.actor_input_dim == 23
    assert not torch.equal(alpha_a, alpha_b)
    assert not torch.equal(beta_a, beta_b)


def test_ctrl_base_critic_conditions_on_beta_distribution_without_actor_value_gradients():
    agent = _beta_critic_agent()
    observations = torch.randn((8, 17), device="cuda")
    alpha, beta, value = agent.get_policy_and_value(observations)
    critic_input = torch.cat((observations, alpha.detach(), beta.detach()), dim=-1)
    changed_input = torch.cat((observations, (alpha + 0.5).detach(), beta.detach()), dim=-1)

    changed_value = agent.critic(changed_input)
    assert agent.critic[0].in_dim == 29
    assert not torch.equal(value, changed_value)

    value.sum().backward()
    assert all(parameter.grad is None for parameter in agent.actor.parameters())

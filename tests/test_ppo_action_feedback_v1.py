"""Contracts for the one-step action-feedback PPO policy."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_action_feedback_v1 as trainer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (5,), np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-2, -1, 0], np.float32),
            np.array([2, 3, 4], np.float32),
        ),
    )


def _agent():
    torch.manual_seed(1)
    return trainer.Agent(_spaces(), trainer.validate_args(trainer.Args(compile=False))).cuda()


def test_actor_input_includes_previous_native_action():
    agent = _agent()
    observations = torch.randn((4, 5), device="cuda")
    previous_a = torch.full((4, 3), 0.2, device="cuda")
    previous_b = torch.full((4, 3), 0.8, device="cuda")
    actor_a = torch.cat((observations, previous_a), dim=-1)
    actor_b = torch.cat((observations, previous_b), dim=-1)

    alpha_a, beta_a, _ = agent.get_policy_and_readout(observations, actor_a)
    alpha_b, beta_b, _ = agent.get_policy_and_readout(observations, actor_b)

    assert agent.actor_input_dim == 8
    assert not torch.equal(alpha_a, alpha_b)
    assert not torch.equal(beta_a, beta_b)


def test_action_feedback_ppo_loss_reuses_conditional_policy_inputs():
    agent = _agent()
    count = 8
    observations = torch.randn((count, 5), device="cuda")
    previous = torch.rand((count, 3), device="cuda").clamp(0.05, 0.95)
    actor_inputs = torch.cat((observations, previous), dim=-1)
    native = torch.rand((count, 3), device="cuda").clamp(0.05, 0.95)
    alpha, beta, readout = agent.get_policy_and_readout(observations, actor_inputs)
    old_logprobs = agent.action_logprob(alpha, beta, native).detach()
    old_values = agent.decode(readout).flatten().detach()

    loss, metrics = trainer.scalar_ppo_loss(
        agent,
        actor_inputs,
        observations,
        native,
        old_logprobs,
        torch.ones(count, device="cuda"),
        torch.zeros(count, device="cuda"),
        old_values,
        trainer.validate_args(trainer.Args(compile=False)),
    )

    assert torch.isfinite(loss)
    assert metrics.shape == (7,)
    assert torch.equal(metrics[3], torch.zeros((), device="cuda"))

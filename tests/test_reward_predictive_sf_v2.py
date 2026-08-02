import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_reward_predictive_sf_v2.py"
)
SPEC = importlib.util.spec_from_file_location("reward_predictive_sf_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(-10.0, 10.0, shape=(3,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))


def test_td0_target_distinguishes_termination_truncation_and_missing_final_state():
    cumulants = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ]
    )
    next_psi = torch.full_like(cumulants, 10.0)
    terminations = torch.tensor([[1.0], [0.0], [0.0]])
    valids = torch.tensor([[0.0], [1.0], [0.0]])

    targets, target_valid = MODULE.vector_td0_target(
        cumulants, next_psi, terminations, valids, gamma=0.9
    )

    torch.testing.assert_close(targets[0], cumulants[0])
    torch.testing.assert_close(targets[1], cumulants[1] + 0.9 * next_psi[1])
    torch.testing.assert_close(targets[2], torch.zeros_like(targets[2]))
    torch.testing.assert_close(
        target_valid, torch.tensor([[True], [True], [False]])
    )


def test_canonical_vector_td0_projects_to_exact_scalar_reward_td0():
    torch.manual_seed(3)
    feature_dim = 7
    reward_vector = torch.full((feature_dim,), feature_dim**-0.5)
    raw_phi = torch.randn(23, feature_dim)
    rewards = torch.randn(23)
    next_psi = torch.randn(23, feature_dim)
    terminations = torch.zeros(23)
    valids = torch.ones(23)
    phi = MODULE.canonicalize_reward_feature(raw_phi, rewards, reward_vector)

    target, valid = MODULE.vector_td0_target(
        phi, next_psi, terminations, valids, gamma=0.97
    )

    expected = rewards + 0.97 * (next_psi @ reward_vector)
    torch.testing.assert_close(target[valid] @ reward_vector, expected[valid])


def test_scalar_gae_censors_missing_truncation_and_keeps_true_terminal():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.6], [0.7]])
    next_values = torch.tensor([[4.0], [5.0], [6.0]])
    terminations = torch.tensor([[0.0], [0.0], [1.0]])
    boundaries = torch.tensor([[1.0], [1.0], [1.0]])
    valids = torch.tensor([[1.0], [0.0], [0.0]])

    advantages, returns = MODULE.scalar_gae(
        rewards,
        values,
        next_values,
        terminations,
        boundaries,
        valids,
        gamma=0.9,
        gae_lambda=0.8,
    )

    torch.testing.assert_close(
        advantages[0], rewards[0] + 0.9 * next_values[0] - values[0]
    )
    torch.testing.assert_close(advantages[1], torch.zeros(1))
    torch.testing.assert_close(returns[1], values[1])
    torch.testing.assert_close(advantages[2], rewards[2] - values[2])


def test_actor_advantage_path_is_exactly_true_reward_gae():
    torch.manual_seed(2)
    shape = (7, 3)
    rewards = torch.randn(shape)
    values = torch.randn(shape)
    next_values = torch.randn(shape)
    terminations = torch.zeros(shape)
    boundaries = torch.zeros(shape)
    valids = torch.ones(shape)

    actor_advantage, actor_return = MODULE.actor_advantages_from_rewards(
        rewards,
        values,
        next_values,
        terminations,
        boundaries,
        valids,
        gamma=0.99,
        gae_lambda=0.95,
    )
    expected_advantage, expected_return = MODULE.scalar_gae(
        rewards,
        values,
        next_values,
        terminations,
        boundaries,
        valids,
        gamma=0.99,
        gae_lambda=0.95,
    )

    torch.testing.assert_close(actor_advantage, expected_advantage)
    torch.testing.assert_close(actor_return, expected_return)


def test_fixed_unit_readout_and_canonicalization_match_physical_reward_exactly():
    args = MODULE.Args(
        hidden=8, k_blocks=1, n_experts=2, feature_dim=5, feature_hidden=12
    )
    agent = MODULE.Agent(DummyVectorEnv(), args)
    reward_vector = agent.reward_vector
    rewards = torch.tensor([-8.0, -1.5, 0.0, 3.0, 11.0])
    raw_phi = torch.randn(5, 5, requires_grad=True)
    phi = MODULE.canonicalize_reward_feature(raw_phi, rewards, reward_vector)

    torch.testing.assert_close(reward_vector.norm(), torch.ones(()))
    torch.testing.assert_close(agent.predict_reward(phi), rewards)
    assert "_reward_vector" in dict(agent.named_buffers())
    assert "_reward_vector" not in dict(agent.named_parameters())

    phi.square().mean().backward()
    expected_reward_direction_gradient = raw_phi.grad @ reward_vector
    torch.testing.assert_close(
        expected_reward_direction_gradient,
        torch.zeros_like(expected_reward_direction_gradient),
        atol=1e-6,
        rtol=0.0,
    )


def test_valid_advantage_normalization_keeps_censored_rows_zero():
    advantages = torch.tensor([1.0, 1000.0, 3.0, -1000.0])
    valid = torch.tensor([True, False, True, False])

    normalized = MODULE.normalize_valid_advantages(advantages, valid)

    torch.testing.assert_close(normalized, torch.tensor([-1.0, 0.0, 1.0, 0.0]))


def test_anchor_and_closure_targets_are_detached_from_old_feature_frame():
    args = MODULE.Args(
        hidden=8, k_blocks=1, n_experts=2, feature_dim=5, feature_hidden=12
    )
    agent = MODULE.Agent(DummyVectorEnv(), args)
    phi_current = torch.randn(16, 5, requires_grad=True)
    phi_next_old = torch.randn(16, 5, requires_grad=True)
    phi_old_same_state = torch.randn(16, 5, requires_grad=True)
    next_action = torch.randn(16, 2)
    next_reward = torch.randn(16)

    closure_loss, future_reward_loss, _ = agent.closure_objectives(
        phi_current, phi_next_old, next_action, next_reward
    )
    anchor_loss = MODULE.feature_anchor_loss(phi_current, phi_old_same_state)
    (closure_loss + future_reward_loss + anchor_loss).backward()

    assert phi_current.grad is not None and phi_current.grad.abs().sum() > 0
    assert phi_next_old.grad is None
    assert phi_old_same_state.grad is None


def test_future_reward_closure_uses_the_same_global_readout():
    args = MODULE.Args(
        hidden=8, k_blocks=1, n_experts=2, feature_dim=4, feature_hidden=12
    )
    agent = MODULE.Agent(DummyVectorEnv(), args)
    with torch.no_grad():
        agent.closure.base.copy_(torch.eye(4))
        agent.closure.hyper[-1].weight.zero_()
        agent.closure.hyper[-1].bias.zero_()
    phi = torch.randn(20, 4)
    next_reward = phi @ agent.reward_vector
    closure_loss, future_reward_loss, prediction = agent.closure_objectives(
        phi, phi.detach().clone(), torch.randn(20, 2), next_reward
    )

    torch.testing.assert_close(prediction, phi)
    torch.testing.assert_close(closure_loss, torch.zeros(()))
    torch.testing.assert_close(future_reward_loss, torch.zeros(()))


def test_closure_pair_sampling_never_crosses_boundaries_or_returns_empty_batch():
    boundaries = torch.tensor(
        [[False, False], [True, False], [False, True], [False, False]]
    )
    expected = torch.tensor([[True, True], [False, True], [True, False]])
    torch.testing.assert_close(MODULE.closure_pair_mask(boundaries), expected)
    assert MODULE.sample_closure_indices(0, 32) is None
    sampled = MODULE.sample_closure_indices(3, 32)
    assert sampled.shape == (32,)
    assert np.logical_and(sampled >= 0, sampled < 3).all()


def test_actor_backward_is_isolated_from_feature_and_closure_parameters():
    torch.manual_seed(11)
    args = MODULE.Args(
        hidden=8, k_blocks=1, n_experts=2, feature_dim=5, feature_hidden=12
    )
    agent = MODULE.Agent(DummyVectorEnv(), args)
    _, _, logprob, entropy, _ = agent.get_action_and_value(torch.randn(32, 3))

    (-(logprob + 0.01 * entropy).mean()).backward()

    assert all(parameter.grad is None for parameter in agent.feature_parameters())
    assert all(parameter.grad is not None for parameter in agent.actor_parameters())

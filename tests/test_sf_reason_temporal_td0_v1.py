import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_reason_temporal_td0_v1.py"
)
SPEC = importlib.util.spec_from_file_location("sf_reason_temporal_td0_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))


class DummyRewardEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
    healthy_reward = 0.75

    def control_cost(self, action):
        return 0.1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 2.0, False, False, {}


class DummyTerminatingRewardEnv(DummyRewardEnv):
    def step(self, action):
        return np.zeros(3, dtype=np.float32), 2.0, True, False, {}


def test_temporal_reward_basis_has_exact_fixed_task_readout():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    reward_components = torch.tensor([[3.0, -0.5, 0.0], [1.0, -2.0, 0.0]])

    phi = agent.transition_features(reward_components)

    expected_reward_features = reward_components.repeat(1, len(agent.temporal_discounts))
    torch.testing.assert_close(phi[:, : agent.reward_feature_dim], expected_reward_features)
    torch.testing.assert_close(agent.scalar_value(phi), reward_components.sum(dim=-1))
    assert agent.sf_dim == 3 * len(agent.temporal_discounts)


def test_reward_component_wrapper_reconstructs_an_exact_sum():
    env = MODULE.RecordRewardComponents(DummyRewardEnv())
    _, reward, _, _, info = env.step(torch.zeros(2).numpy())

    components = info[MODULE.RecordRewardComponents.info_key]
    torch.testing.assert_close(torch.tensor(components).sum(), torch.tensor(reward))
    torch.testing.assert_close(torch.tensor(components), torch.tensor([1.35, -0.1, 0.75]))


def test_reward_components_survive_vector_env_autoreset():
    envs = gym.vector.SyncVectorEnv(
        [lambda: MODULE.RecordRewardComponents(DummyTerminatingRewardEnv()) for _ in range(2)]
    )
    try:
        envs.reset(seed=1)
        _, rewards, terminations, _, infos = envs.step(np.zeros((2, 2), dtype=np.float32))
        components = MODULE.reward_components_from_infos(infos, num_envs=2)

        assert terminations.all()
        np.testing.assert_allclose(components.sum(axis=-1), rewards, rtol=1e-6, atol=1e-6)
    finally:
        envs.close()


def test_vector_td_target_has_exact_scalar_bellman_readout():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    components = torch.tensor([[3.0, -0.5, 0.25], [1.0, -2.0, 0.75]])
    next_psi = torch.randn(2, agent.sf_dim)
    target = agent.transition_features(components) + agent.sf_discounts * next_psi

    expected = components.sum(dim=-1) + args.gamma * agent.scalar_value(next_psi)
    torch.testing.assert_close(agent.scalar_value(target), expected)


def test_identity_tolerance_accounts_for_large_opposing_reason_terms():
    terms = torch.tensor([[1_000.0, -999.99, 0.0]])
    scalar_total = terms.sum(dim=-1)
    simulated_associativity_error = torch.tensor([1.2e-3])
    tolerance = MODULE.cancellation_aware_tolerance(terms)

    assert tolerance > simulated_associativity_error
    assert tolerance > 1e-5 + 1e-5 * scalar_total.abs()


def test_beta_action_basis_is_centered_orthonormal_across_policy_concentrations():
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(1234)
    try:
        for alpha_value, beta_value in (
            (2.0, 2.0),
            (1.2, 5.0),
            (10.0, 1.1),
            (10_000.0, 10_000.0),
            (5_000.0, 10_000.0),
        ):
            alpha = torch.tensor([alpha_value])
            beta = torch.tensor([beta_value])
            samples = torch.distributions.Beta(alpha, beta).sample((250_000,))
            alpha_batch = alpha.expand_as(samples)
            beta_batch = beta.expand_as(samples)
            basis = MODULE.Agent._orthogonal_beta_basis(samples, alpha_batch, beta_batch)
            basis = basis.squeeze(-1)

            torch.testing.assert_close(
                basis.mean(dim=0), torch.zeros(2), atol=1.5e-2, rtol=0.0
            )
            gram = basis.T @ basis / basis.shape[0]
            torch.testing.assert_close(gram, torch.eye(2), atol=2.5e-2, rtol=0.0)
    finally:
        torch.random.set_rng_state(generator_state)


def test_gae_censors_missing_truncation_target_but_keeps_true_terminal():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.6], [0.7]])
    next_values = torch.tensor([[4.0], [5.0], [6.0]])
    terminations = torch.tensor([[0.0], [0.0], [1.0]])
    boundaries = torch.tensor([[0.0], [1.0], [1.0]])
    # t=1 is an invalid time-limit transition; t=2 is a valid true terminal
    # even though no successor observation is available.
    bootstrap_valids = torch.tensor([[1.0], [0.0], [0.0]])

    advantages, returns = MODULE.compute_scalar_gae(
        rewards,
        values,
        next_values,
        terminations,
        boundaries,
        bootstrap_valids,
        gamma=0.9,
        gae_lambda=0.8,
    )

    torch.testing.assert_close(advantages[1], torch.zeros(1))
    torch.testing.assert_close(returns[1], values[1])
    torch.testing.assert_close(advantages[2], rewards[2] - values[2])
    expected_t0 = rewards[0] + 0.9 * next_values[0] - values[0]
    torch.testing.assert_close(advantages[0], expected_t0)


def test_reasonwise_gae_sums_exactly_to_scalar_gae():
    torch.manual_seed(12)
    reason_rewards = torch.randn(7, 4, 3)
    reason_values = torch.randn(7, 4, 3)
    next_reason_values = torch.randn(7, 4, 3)
    terminations = torch.zeros(7, 4)
    boundaries = torch.zeros(7, 4)
    bootstrap_valids = torch.ones(7, 4)
    terminations[4, 1] = 1.0
    boundaries[4, 1] = 1.0
    boundaries[5, 2] = 1.0
    bootstrap_valids[5, 2] = 0.0

    _, reason_returns = MODULE.compute_scalar_gae(
        reason_rewards,
        reason_values,
        next_reason_values,
        terminations,
        boundaries,
        bootstrap_valids,
        gamma=0.99,
        gae_lambda=0.95,
    )
    _, scalar_returns = MODULE.compute_scalar_gae(
        reason_rewards.sum(dim=-1),
        reason_values.sum(dim=-1),
        next_reason_values.sum(dim=-1),
        terminations,
        boundaries,
        bootstrap_valids,
        gamma=0.99,
        gae_lambda=0.95,
    )

    torch.testing.assert_close(reason_returns.sum(dim=-1), scalar_returns, atol=1e-5, rtol=1e-5)


def test_full_degree_two_action_basis_is_orthonormal_for_factorized_beta():
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(4321)
    try:
        alpha = torch.tensor([1.5, 8.0])
        beta = torch.tensor([4.0, 2.0])
        samples = torch.distributions.Beta(alpha, beta).sample((250_000,))
        basis = MODULE.Agent._orthogonal_beta_basis(
            samples, alpha.expand_as(samples), beta.expand_as(samples)
        )

        assert basis.shape[1] == 5  # 2 linear + 2 quadratic + 1 cross term
        torch.testing.assert_close(basis.mean(dim=0), torch.zeros(5), atol=1.5e-2, rtol=0.0)
        gram = basis.T @ basis / basis.shape[0]
        torch.testing.assert_close(gram, torch.eye(5), atol=3e-2, rtol=0.0)
    finally:
        torch.random.set_rng_state(generator_state)


def test_action_loss_has_no_trunk_or_value_head_gradient():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    observations = torch.randn(32, 3)
    basis = torch.randn(32, agent.action_basis_dim)
    _, action_advantage = agent.get_psi_v_action_advantage(observations, basis)

    action_advantage.square().mean().backward()

    for parameter in agent.value_critic_parameters():
        assert parameter.grad is None
    assert all(parameter.grad is not None for parameter in agent.action_critic_parameters())

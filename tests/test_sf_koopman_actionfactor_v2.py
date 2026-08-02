import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_koopman_actionfactor_v2.py"
)
SPEC = importlib.util.spec_from_file_location("sf_koopman_actionfactor_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(-10.0, 10.0, shape=(3,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))


class DummyRewardEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
    observation_space = gym.spaces.Box(-10.0, 10.0, shape=(3,))
    healthy_reward = 0.75

    def __init__(self, terminate=False):
        super().__init__()
        self.terminate = terminate
        self.state = np.zeros(3, dtype=np.float32)

    def control_cost(self, action):
        return 0.1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(3, dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        self.state += 1.0
        return self.state.copy(), 2.0, self.terminate, False, {}


def test_resolvent_fit_recovers_column_orientation_and_discounted_value():
    torch.manual_seed(4)
    n, dim, reasons = 20_000, 5, 3
    x = torch.randn(n, dim)
    x[:, 0] = 1.0
    koopman = torch.tensor(
        [
            [0.96, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.7, -0.2, 0.0, 0.0],
            [0.0, 0.1, 0.5, 0.1, 0.0],
            [0.0, 0.0, 0.2, 0.3, 0.1],
            [0.0, 0.0, 0.0, -0.1, 0.2],
        ]
    )
    reward_map = torch.randn(reasons, dim)
    y = x @ koopman.T
    rewards = x @ reward_map.T

    fit_k, fit_b, fit_g, condition, _ = MODULE.fit_koopman_resolvent(
        x, y, rewards, gamma=0.93, ridge=1e-10
    )
    expected_g = torch.linalg.solve(
        (torch.eye(dim) - 0.93 * koopman).T, reward_map.T
    ).T

    torch.testing.assert_close(fit_k, koopman, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fit_b, reward_map, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fit_g, expected_g, atol=3e-5, rtol=3e-5)
    assert torch.isfinite(condition)
    bellman_error = x @ fit_g.T - (rewards + 0.93 * (y @ fit_g.T))
    torch.testing.assert_close(bellman_error, torch.zeros_like(bellman_error), atol=1e-4, rtol=0.0)


def test_operator_projection_is_contractive_in_empirical_l2_metric():
    metric = torch.tensor(
        [[2.0, 0.4, 0.0], [0.4, 1.0, 0.2], [0.0, 0.2, 0.5]],
        dtype=torch.float64,
    )
    unstable = torch.tensor(
        [[1.3, 1.1, 0.0], [0.0, 0.9, 0.7], [0.2, 0.0, 1.2]],
        dtype=torch.float64,
    )
    projected = MODULE.project_l2_contractive(unstable, metric)
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    metric_sqrt = (
        eigenvectors * eigenvalues.sqrt().unsqueeze(0)
    ) @ eigenvectors.T
    metric_inv_sqrt = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.T
    whitened = metric_inv_sqrt @ projected @ metric_sqrt

    assert torch.linalg.svdvals(whitened).max() <= 1.0 + 1e-10
    assert torch.linalg.eigvals(projected).abs().max() <= 1.0 + 1e-10


def test_terminal_zero_targets_and_truncation_bootstraps_fit_distinctly():
    x = torch.eye(4).repeat(4, 1)
    next_x = torch.roll(x, shifts=1, dims=0)
    terminations = torch.tensor([1.0, 0.0, 0.0, 0.0] * 4).unsqueeze(-1)
    bootstrap_valid = torch.tensor([0.0, 1.0, 1.0, 1.0] * 4).unsqueeze(-1)
    y = (1.0 - terminations) * bootstrap_valid * next_x
    rewards = torch.randn(len(x), 3)
    fit_k, _, _, _, _ = MODULE.fit_koopman_resolvent(
        x, y, rewards, gamma=0.9, ridge=1e-6
    )
    prediction = x @ fit_k.T

    torch.testing.assert_close(prediction, y, atol=1e-5, rtol=1e-5)
    assert prediction[0].abs().max() < 1e-5
    assert prediction[1].abs().sum() > 0.5


def test_gae_censors_missing_truncation_but_keeps_true_terminal():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.6], [0.7]])
    next_values = torch.tensor([[4.0], [5.0], [6.0]])
    terminations = torch.tensor([[0.0], [0.0], [1.0]])
    boundaries = torch.tensor([[0.0], [1.0], [1.0]])
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
    torch.testing.assert_close(
        advantages[0], rewards[0] + 0.9 * next_values[0] - values[0]
    )


def test_missing_final_raw_observation_marks_truncation_invalid():
    infos = {
        "raw_observation": np.zeros((1, 3), dtype=np.float32),
        "_raw_observation": np.array([True]),
        "final_info": np.array([{}], dtype=object),
        "_final_info": np.array([True]),
    }
    _, transition, valid = MODULE.raw_observations_from_infos(
        infos, 1, boundaries=np.array([True])
    )

    np.testing.assert_allclose(transition, np.zeros((1, 3), dtype=np.float32))
    assert not valid[0]


def test_reward_and_raw_observation_survive_vector_autoreset_final_info():
    def make_wrapped():
        env = MODULE.RecordRewardComponents(DummyRewardEnv(terminate=True))
        return MODULE.RecordRawObservation(env)

    envs = gym.vector.SyncVectorEnv([make_wrapped, make_wrapped])
    try:
        _, reset_infos = envs.reset(seed=1)
        emitted, _, _ = MODULE.raw_observations_from_infos(reset_infos, 2)
        np.testing.assert_allclose(emitted, np.zeros((2, 3), dtype=np.float32))
        _, rewards, terminations, _, infos = envs.step(
            np.zeros((2, 2), dtype=np.float32)
        )
        components = MODULE.reward_components_from_infos(infos, 2)
        _, transition_raw, transition_valid = MODULE.raw_observations_from_infos(
            infos, 2, terminations
        )

        np.testing.assert_allclose(components.sum(axis=-1), rewards, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(transition_raw, np.ones((2, 3), dtype=np.float32))
        assert transition_valid.all()
    finally:
        envs.close()


def test_whitening_penalty_rejects_collapse_and_accepts_whitened_latent():
    torch.manual_seed(8)
    collapsed = torch.zeros(4096, 7)
    samples = torch.randn(4096, 7)
    whitened = (samples - samples.mean(dim=0)) @ torch.linalg.inv(
        torch.linalg.cholesky(torch.cov(samples.T))
    ).T

    collapsed_loss = MODULE.whitening_loss(collapsed)
    whitened_loss = MODULE.whitening_loss(whitened)

    assert collapsed_loss > 0.1
    assert whitened_loss < 2e-3


def test_centered_canonical_monomials_have_zero_beta_expectation():
    torch.manual_seed(4321)
    alpha = torch.tensor([1.5, 8.0])
    beta = torch.tensor([4.0, 2.0])
    samples = torch.distributions.Beta(alpha, beta).sample((250_000,))
    basis = MODULE.Agent._centered_beta_monomials(
        samples, alpha.expand_as(samples), beta.expand_as(samples)
    )

    assert basis.shape[1] == 5
    torch.testing.assert_close(basis.mean(dim=0), torch.zeros(5), atol=2e-3, rtol=0.0)


def test_centered_monomial_action_differences_are_policy_invariant():
    z = torch.tensor([[0.15, 0.8], [0.7, 0.25], [0.4, 0.6]])
    z_reference = torch.tensor([[0.55, 0.35], [0.1, 0.9], [0.8, 0.2]])
    alpha_one = torch.tensor([[1.2, 7.0]]).expand_as(z)
    beta_one = torch.tensor([[5.0, 2.0]]).expand_as(z)
    alpha_two = torch.tensor([[12.0, 1.1]]).expand_as(z)
    beta_two = torch.tensor([[1.5, 9.0]]).expand_as(z)

    difference_one = MODULE.Agent._centered_beta_monomials(
        z, alpha_one, beta_one
    ) - MODULE.Agent._centered_beta_monomials(
        z_reference, alpha_one, beta_one
    )
    difference_two = MODULE.Agent._centered_beta_monomials(
        z, alpha_two, beta_two
    ) - MODULE.Agent._centered_beta_monomials(
        z_reference, alpha_two, beta_two
    )

    torch.testing.assert_close(
        difference_one, difference_two, atol=1e-7, rtol=1e-7
    )


def test_action_loss_isolated_from_encoder_and_operator_snapshot():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, model_hidden=8, model_dim=6)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    raw_obs = torch.randn(64, 3)
    latent = agent.encode(raw_obs)
    basis = torch.randn(64, agent.action_basis_dim)
    operator_before = agent.koopman.clone()
    reward_effect, dynamics_effect, prediction = agent.action_effects(latent, basis)

    (prediction.square().mean() + reward_effect.mean() + dynamics_effect.mean()).backward()

    assert all(parameter.grad is None for parameter in agent.model_parameters())
    assert all(parameter.grad is not None for parameter in agent.action_critic_parameters())
    torch.testing.assert_close(agent.koopman, operator_before)


def test_action_factor_composition_matches_q_minus_v_identity():
    torch.manual_seed(17)
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, model_hidden=8, model_dim=6)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    latent = torch.randn(13, agent.latent_dim)
    basis = torch.randn(13, agent.action_basis_dim)
    value_map = torch.randn(3, agent.model_dim)
    with torch.no_grad():
        agent.reward_action_head.weight.normal_()
        agent.reward_action_head.bias.normal_()
        agent.dynamics_action_head.weight.normal_()
        agent.dynamics_action_head.bias.normal_()

    reward_effect, dynamics_effect, reason_advantage = agent.action_effects(
        latent, basis, value_map
    )
    expected = reward_effect + args.gamma * (dynamics_effect @ value_map.T)

    torch.testing.assert_close(reason_advantage, expected)
    torch.testing.assert_close(
        reason_advantage.sum(dim=-1),
        reward_effect.sum(dim=-1)
        + args.gamma * (dynamics_effect @ value_map.T).sum(dim=-1),
    )


def test_operator_fit_projects_empirical_dynamics_to_contraction():
    torch.manual_seed(22)
    raw = torch.randn(6, 6)
    metric = raw @ raw.T + 0.1 * torch.eye(6)
    unstable = torch.randn(6, 6) * 2.0
    koopman = MODULE.project_l2_contractive(unstable.double(), metric.double()).float()
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    metric_sqrt = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
    metric_inv_sqrt = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.T
    empirical_whitened = metric_inv_sqrt @ koopman @ metric_sqrt

    assert torch.linalg.svdvals(empirical_whitened).max() <= 1.0 + 1e-5
    assert 0.99 * torch.linalg.eigvals(koopman).abs().max() < 1.0


def test_snapshot_changes_value_only_when_explicitly_applied():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, model_hidden=8, model_dim=6)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    state = agent.model_state(torch.randn(16, 3))
    old_value = agent.scalar_value(state)
    koopman = torch.eye(agent.model_dim) * 0.5
    reward_map = torch.randn(3, agent.model_dim)
    value_map = torch.linalg.solve(
        (torch.eye(agent.model_dim) - args.gamma * koopman).T, reward_map.T
    ).T

    torch.testing.assert_close(old_value, torch.zeros_like(old_value))
    agent.snapshot_operator(koopman, reward_map, value_map)
    new_value = agent.scalar_value(state)
    assert not torch.equal(old_value, new_value)
    assert agent.operator_initialized.item()


def test_observability_spectrum_detects_reward_invisible_modes():
    koopman = torch.diag(torch.tensor([0.9, 0.7, 0.4, 0.2]))
    reward_map = torch.zeros(3, 4)
    reward_map[0, 0] = 1.0
    reward_map[1, 1] = 1.0
    spectrum = MODULE.observability_spectrum(koopman, reward_map)

    assert (spectrum > 1e-6).sum() == 2

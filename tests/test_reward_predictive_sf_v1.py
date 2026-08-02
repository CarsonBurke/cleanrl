import importlib.util
from pathlib import Path

import gymnasium as gym
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_reward_predictive_sf_v1.py"
)
SPEC = importlib.util.spec_from_file_location("reward_predictive_sf_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(-10.0, 10.0, shape=(3,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))


def test_action_conditioned_closure_shapes_and_gradients_reach_both_feature_sides():
    torch.manual_seed(3)
    feature_net = MODULE.RewardPredictiveFeatures(3, 2, 16, 8)
    closure = MODULE.ActionConditionedClosure(2, 16, 8)
    current_phi = feature_net(torch.randn(32, 3), torch.randn(32, 2))
    next_phi = feature_net(torch.randn(32, 3), torch.randn(32, 2))
    current_phi.retain_grad()
    next_phi.retain_grad()
    prediction = closure(current_phi, torch.randn(32, 2))

    assert prediction.shape == (32, 8)
    (prediction - next_phi).square().mean().backward()

    assert current_phi.grad is not None and current_phi.grad.abs().sum() > 0
    assert next_phi.grad is not None and next_phi.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in closure.parameters())
    assert all(parameter.grad is not None for parameter in feature_net.parameters())


def test_feature_rmsnorm_fixes_per_sample_scale_without_batch_statistics():
    torch.manual_seed(5)
    feature_net = MODULE.RewardPredictiveFeatures(3, 2, 16, 32)
    phi = feature_net(torch.randn(64, 3) * 10.0, torch.randn(64, 2))
    per_sample_rms = phi.square().mean(dim=-1).sqrt()

    torch.testing.assert_close(per_sample_rms, torch.ones_like(per_sample_rms), atol=2e-4, rtol=0.0)
    assert phi.std(dim=0).mean() > 0.05


def test_global_reward_readout_is_a_plain_dot_product():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, feature_dim=7, feature_hidden=12)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    phi = torch.randn(19, 7)
    with torch.no_grad():
        agent.reward_vector.copy_(torch.randn(7))

    torch.testing.assert_close(agent.predict_reward(phi), phi @ agent.reward_vector)


def test_projected_vector_gae_equals_scalar_gae_when_reward_is_phi_dot_w():
    torch.manual_seed(7)
    steps, envs, dim = 9, 4, 6
    phi = torch.randn(steps, envs, dim)
    psi = torch.randn(steps, envs, dim)
    next_psi = torch.randn(steps, envs, dim)
    w = torch.randn(dim)
    terminations = torch.zeros(steps, envs)
    boundaries = torch.zeros(steps, envs)
    valids = torch.ones(steps, envs)
    terminations[4, 1] = boundaries[4, 1] = 1.0
    boundaries[6, 2] = 1.0  # valid time-limit bootstrap

    vector_advantage, _ = MODULE.vector_gae(
        phi, psi, next_psi, terminations, boundaries, valids, 0.99, 0.95
    )
    scalar_advantage, _ = MODULE.scalar_gae(
        phi @ w,
        psi @ w,
        next_psi @ w,
        terminations,
        boundaries,
        valids,
        0.99,
        0.95,
    )

    torch.testing.assert_close(vector_advantage @ w, scalar_advantage, atol=2e-5, rtol=2e-5)


def test_boundary_masks_exclude_closure_pairs_and_preserve_bootstrap_semantics():
    boundaries = torch.tensor(
        [[False, False], [True, False], [False, True], [False, False]]
    )
    expected_pairs = torch.tensor([[True, True], [False, True], [True, False]])
    torch.testing.assert_close(MODULE.closure_pair_mask(boundaries), expected_pairs)

    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.6]])
    next_values = torch.tensor([[4.0], [5.0]])
    terminations = torch.tensor([[0.0], [1.0]])
    episode_boundaries = torch.ones(2, 1)
    valids = torch.tensor([[1.0], [0.0]])
    advantages, _ = MODULE.scalar_gae(
        rewards, values, next_values, terminations, episode_boundaries, valids, 0.9, 0.8
    )

    torch.testing.assert_close(advantages[0], rewards[0] + 0.9 * next_values[0] - values[0])
    torch.testing.assert_close(advantages[1], rewards[1] - values[1])


def test_actor_backward_is_isolated_from_features_closure_and_reward_readout():
    torch.manual_seed(11)
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, feature_dim=7, feature_hidden=12)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    obs = torch.randn(32, 3)
    _, _, logprob, entropy, _ = agent.get_action_and_value(obs)

    (-(logprob + 0.01 * entropy).mean()).backward()

    assert all(parameter.grad is None for parameter in agent.feature_parameters())
    assert all(parameter.grad is not None for parameter in agent.actor_parameters())

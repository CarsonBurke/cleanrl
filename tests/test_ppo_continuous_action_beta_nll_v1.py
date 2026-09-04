import gymnasium as gym
import numpy as np
import torch

from cleanrl.beta_policy.ppo_continuous_action_beta_nll_v1 import Agent, SAMPLE_EPS, beta_nll_to_weights


class DummyVecEnv:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
    single_action_space = gym.spaces.Box(
        np.array([-2.0, -0.5], dtype=np.float32),
        np.array([3.0, 1.5], dtype=np.float32),
        dtype=np.float32,
    )


def test_beta_policy_samples_native_z_and_maps_to_action_bounds():
    torch.manual_seed(0)
    agent = Agent(DummyVecEnv())
    obs = torch.randn(32, 5)

    action, z, logprob, entropy, value, beta_nll = agent.get_beta_action_and_value(obs)

    low = torch.as_tensor(DummyVecEnv.single_action_space.low)
    high = torch.as_tensor(DummyVecEnv.single_action_space.high)
    assert action.shape == z.shape == (32, 2)
    assert torch.all(z > 0.0) and torch.all(z < 1.0)
    assert torch.all(action >= low - 1e-6)
    assert torch.all(action <= high + 1e-6)
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(beta_nll).all()
    assert value.shape == (32, 1)


def test_replaying_stored_z_recomputes_same_logprob():
    torch.manual_seed(1)
    agent = Agent(DummyVecEnv())
    obs = torch.randn(16, 5)

    _, z, logprob, _, _, _ = agent.get_beta_action_and_value(obs)
    _, replay_z, replay_logprob, _, _, _ = agent.get_beta_action_and_value(obs, z)

    assert torch.allclose(replay_z, z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS))
    assert torch.allclose(replay_logprob, logprob, atol=1e-6)


def test_standard_action_api_keeps_cleanrl_eval_contract():
    torch.manual_seed(1)
    agent = Agent(DummyVecEnv())
    obs = torch.randn(16, 5)

    action, logprob, entropy, value = agent.get_action_and_value(obs)
    replay_action, replay_logprob, _, _ = agent.get_action_and_value(obs, action)

    assert action.shape == replay_action.shape == (16, 2)
    assert torch.allclose(replay_action, action, atol=1e-6)
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()
    assert torch.allclose(replay_logprob, logprob, atol=1e-6)


def test_beta_nll_weights_are_finite_detached_and_mean_normalized():
    torch.manual_seed(2)
    agent = Agent(DummyVecEnv())
    obs = torch.randn(64, 5)
    dist = agent._dist(obs)
    z = torch.linspace(0.05, 0.95, steps=64).unsqueeze(1).expand(-1, 2)

    beta_nll, weights = agent.beta_nll_weights(dist, z, clip=10.0, weight_min=0.0, weight_max=100.0)

    assert torch.isfinite(beta_nll).all()
    assert torch.isfinite(weights).all()
    assert torch.all(beta_nll >= 0.0)
    assert weights.requires_grad is False
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-5)


def test_zero_beta_nll_falls_back_to_unit_weights():
    agent = Agent(DummyVecEnv())
    obs = torch.randn(8, 5)
    dist = agent._dist(obs)
    alpha = dist.concentration1
    beta = dist.concentration0
    mode = ((alpha - 1.0) / (alpha + beta - 2.0).clamp_min(SAMPLE_EPS)).clamp(
        SAMPLE_EPS, 1.0 - SAMPLE_EPS
    )

    _, weights = agent.beta_nll_weights(dist, mode, clip=10.0, weight_min=0.25, weight_max=4.0)

    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)


def test_update_weights_use_stored_behavior_nll_not_live_policy():
    stored_nll = torch.tensor([0.5, 1.0, 1.5, 2.0])
    live_policy_nll = torch.tensor([9.0, 1.0, 1.0, 1.0])

    _, behavior_weights = beta_nll_to_weights(stored_nll, clip=10.0, weight_min=0.0, weight_max=100.0)
    _, live_weights = beta_nll_to_weights(live_policy_nll, clip=10.0, weight_min=0.0, weight_max=100.0)

    assert torch.allclose(behavior_weights, torch.tensor([0.4, 0.8, 1.2, 1.6]))
    assert not torch.allclose(behavior_weights, live_weights)
    assert behavior_weights.requires_grad is False

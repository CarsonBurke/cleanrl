import gymnasium as gym
import torch

from cleanrl.ppo_continuous_action_panlejepa_direct_v1 import (
    Agent,
    Args,
    sample_pan_hindsight_indices,
    select_pan_elite_indices,
)


class _DummyVecEnv:
    single_observation_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(17,), dtype=float
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=float
    )


def _agent():
    args = Args(
        hidden=16,
        k_blocks=1,
        n_experts=2,
        lewm_dim=16,
        lewm_encoder_layers=1,
        lewm_predictor_layers=1,
        lewm_heads=4,
        lewm_kv_heads=2,
        lewm_ffn_mult=2,
        lewm_sigreg_num_proj=8,
    )
    return Agent(_DummyVecEnv(), args)


def test_elite_and_hindsight_indices_never_cross_episode_boundaries():
    rewards = torch.zeros(20, 1)
    rewards[5] = 10.0
    rewards[15] = 20.0
    boundaries = torch.zeros_like(rewards)
    boundaries[9] = 1.0

    elite, elite_valid, _, _, episode_ids = select_pan_elite_indices(
        rewards, boundaries, min_offset=2, horizon=5, quality_window=2
    )
    generator = torch.Generator().manual_seed(7)
    hindsight, hindsight_valid, hindsight_episode_ids = sample_pan_hindsight_indices(
        boundaries, min_offset=2, horizon=5, generator=generator
    )

    assert torch.equal(episode_ids, hindsight_episode_ids)
    for indices, valid in ((elite, elite_valid), (hindsight, hindsight_valid)):
        source_episode = episode_ids[:, 0]
        selected_episode = episode_ids[indices[:, 0], 0]
        assert torch.equal(source_episode[valid[:, 0]], selected_episode[valid[:, 0]])
        assert torch.all(indices[valid] >= torch.arange(20).view(-1, 1)[valid] + 2)
    assert not elite_valid[8, 0]
    assert not hindsight_valid[8, 0]


def test_zero_initialized_goal_branch_preserves_base_beta_policy():
    torch.manual_seed(3)
    agent = _agent()
    agent_input = torch.randn(4, agent.agent_input_dim)
    explore_code = torch.randn(4, agent.explore_code_dim)

    composite_alpha, composite_beta = agent.get_beta_params_from_agent_input(
        agent_input, explore_code
    )
    actor_feat = agent._actor_feat_from_agent_input(agent_input)
    base_dist, _, _ = agent._actor_dist(actor_feat, explore_code)

    torch.testing.assert_close(composite_alpha, base_dist.concentration1)
    torch.testing.assert_close(composite_beta, base_dist.concentration0)


def test_goal_shapes_and_frozen_replay_are_exact():
    torch.manual_seed(4)
    agent = _agent()
    batch = 3
    agent_input = torch.randn(batch, agent.agent_input_dim)
    obs = torch.randn(batch, agent.obs_dim)
    explore_code = torch.randn(batch, agent.explore_code_dim)
    current, elite, goal = agent.propose_aspirational_goal(agent_input)
    assert current.shape == elite.shape == goal.shape == (batch, agent.obs_dim, 16)

    # Exercise exact replay with a genuinely nonzero composite goal policy, not
    # only the zero-initialized compatibility state.
    with torch.no_grad():
        agent.goal_alpha_residual.weight.normal_(std=0.05)
        agent.goal_beta_residual.weight.normal_(std=0.05)

    first = agent.get_action_and_value_from_agent_input(
        obs, agent_input, explore_code=explore_code
    )
    replay = agent.get_action_and_value_from_agent_input(
        obs, agent_input, z=first[1], explore_code=explore_code
    )
    torch.testing.assert_close(first[2], replay[2], rtol=0.0, atol=0.0)


def test_hindsight_nll_cannot_update_base_actor():
    torch.manual_seed(5)
    agent = _agent()
    batch = 4
    agent_input = torch.randn(batch, agent.agent_input_dim)
    desired_next = torch.randn(batch, agent.obs_dim, 16)
    native_action = torch.full((batch, 6), 0.5)

    agent.pan_hindsight_nll(agent_input, desired_next, native_action).mean().backward()

    base_modules = (
        agent.actor_readout,
        agent.actor_trunk,
        agent.actor_alpha_head,
        agent.actor_beta_head,
        agent.actor_explore_basis,
    )
    assert all(
        parameter.grad is None
        for module in base_modules
        for parameter in module.parameters()
    )
    assert agent.goal_alpha_residual.weight.grad is not None
    assert agent.goal_beta_residual.weight.grad is not None

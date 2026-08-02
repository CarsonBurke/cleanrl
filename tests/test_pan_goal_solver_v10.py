import copy

import numpy as np
import torch

from cleanrl.ppo_continuous_action_pan_goal_solver_v10 import (
    Args,
    DirectGoalFollower,
    PanGoalSolver,
    VectorReplayBuffer,
    train_follower_step,
    train_goal_step,
)


def _args():
    return Args(
        latent_dim=16,
        encoder_layers=1,
        temporal_layers=1,
        heads=4,
        ffn_mult=2,
        history=8,
        batch_size=16,
        follower_hidden=32,
        sigreg_projections=8,
    )


def _agent():
    torch.manual_seed(3)
    return PanGoalSolver(17, 6, _args())


def _batch(batch_size=16, history=8):
    rng = np.random.default_rng(4)
    return {
        "obs": rng.normal(size=(batch_size, history, 17)).astype(np.float32),
        "incoming": rng.normal(size=(batch_size, history, 6)).astype(np.float32),
        "next_obs": rng.normal(size=(batch_size, history, 17)).astype(np.float32),
        "future_obs": rng.normal(size=(batch_size, history, 17)).astype(np.float32),
        "occupancy_obs": rng.normal(size=(batch_size, history, 17)).astype(np.float32),
        "occupancy_incoming": rng.normal(size=(batch_size, history, 6)).astype(np.float32),
        "occupancy_future_obs": rng.normal(size=(batch_size, history, 17)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(batch_size, 6)).astype(np.float32),
        "reward": rng.normal(size=batch_size).astype(np.float32),
    }


def test_goal_predictor_maps_each_detached_belief_to_one_point():
    agent = _agent()
    belief = torch.randn(5, 16, requires_grad=True)
    goal = agent.goal_predictor(belief.detach())
    assert goal.shape == (5, 16)
    goal.sum().backward()
    assert belief.grad is None


def test_reward_prediction_is_invariant_to_goal_distance():
    agent = _agent()
    goal = torch.randn(5, 16)
    with torch.no_grad():
        near = agent.reward_predictor(goal)
        far = agent.reward_predictor(100.0 * goal)
    torch.testing.assert_close(near, far, rtol=1e-5, atol=1e-5)


def test_goal_and_follower_belief_are_observation_only():
    agent = _agent()
    obs = torch.randn(5, 8, 17)
    incoming_a = torch.randn(5, 8, 6)
    incoming_b = torch.randn(5, 8, 6)
    with torch.no_grad():
        belief_a, goal_a, _, world_belief_a = agent.follower_state(obs, incoming_a)
        belief_b, goal_b, _, world_belief_b = agent.follower_state(obs, incoming_b)
    torch.testing.assert_close(goal_a, goal_b, rtol=0, atol=0)
    torch.testing.assert_close(belief_a, belief_b, rtol=0, atol=0)
    torch.testing.assert_close(world_belief_a, world_belief_b, rtol=0, atol=0)


def test_each_action_recomputes_one_goal_and_calls_one_follower():
    agent = _agent()
    follower_calls = 0
    goal_predictor_calls = 0
    original_forward = agent.follower.forward
    original_goal_predictor = agent.goal_predictor.forward

    def counted_forward(*args, **kwargs):
        nonlocal follower_calls
        follower_calls += 1
        return original_forward(*args, **kwargs)

    def counted_goal_predictor(*args, **kwargs):
        nonlocal goal_predictor_calls
        goal_predictor_calls += 1
        return original_goal_predictor(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("training-only module was called during action inference")

    agent.goal_predictor.forward = counted_goal_predictor
    agent.reward_predictor.forward = forbidden
    agent.world_model.transition.forward = forbidden
    agent.follower.forward = counted_forward
    with torch.no_grad():
        action, current_goal, goal = agent.act(
            torch.randn(4, 8, 17), torch.randn(4, 8, 6)
        )
    assert action.shape == (4, 6)
    assert current_goal.shape == (4, 16)
    assert goal.shape == (4, 16)
    assert follower_calls == 1
    assert goal_predictor_calls == 1


def test_optimizer_parameter_sets_are_disjoint():
    groups = _agent().parameter_groups()
    ids = [{id(parameter) for parameter in group} for group in groups]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]


def test_follower_and_goal_losses_respect_gradient_firewalls():
    args = _args()
    agent = _agent()
    batch = _batch()
    world, follower, goal = agent.parameter_groups()
    follower_optimizer = torch.optim.AdamW(follower, lr=1e-3)
    goal_optimizer = torch.optim.AdamW(agent.goal_predictor.parameters(), lr=1e-3)
    reward_optimizer = torch.optim.AdamW(agent.reward_predictor.parameters(), lr=1e-3)

    for parameter in agent.parameters():
        parameter.grad = None
    train_follower_step(
        agent,
        follower_optimizer,
        batch,
        args,
        torch.device("cpu"),
    )
    assert all(parameter.grad is None for parameter in world)
    assert all(parameter.grad is None for parameter in goal)

    for parameter in agent.parameters():
        parameter.grad = None
    train_goal_step(
        agent,
        goal_optimizer,
        reward_optimizer,
        batch,
        args,
        torch.device("cpu"),
    )
    assert all(parameter.grad is None for parameter in world)
    assert all(parameter.grad is None for parameter in follower)


def test_reward_changes_only_goal_parameters():
    args = _args()
    first = _agent()
    second = copy.deepcopy(first)
    batch = _batch()
    first_goal = first.parameter_groups()[2]
    second_goal = second.parameter_groups()[2]
    first_goal_optimizer = torch.optim.AdamW(first.goal_predictor.parameters(), lr=1e-3)
    second_goal_optimizer = torch.optim.AdamW(second.goal_predictor.parameters(), lr=1e-3)
    first_reward_optimizer = torch.optim.AdamW(first.reward_predictor.parameters(), lr=1e-3)
    second_reward_optimizer = torch.optim.AdamW(second.reward_predictor.parameters(), lr=1e-3)
    first_batch = copy.deepcopy(batch)
    second_batch = copy.deepcopy(batch)
    second_batch["reward"] = first_batch["reward"][::-1].copy()
    train_goal_step(
        first,
        first_goal_optimizer,
        first_reward_optimizer,
        first_batch,
        args,
        torch.device("cpu"),
    )
    train_goal_step(
        second,
        second_goal_optimizer,
        second_reward_optimizer,
        second_batch,
        args,
        torch.device("cpu"),
    )
    assert any(not torch.equal(a, b) for a, b in zip(first_goal, second_goal, strict=True))
    for first_group, second_group in zip(first.parameter_groups()[:2], second.parameter_groups()[:2], strict=True):
        for first_parameter, second_parameter in zip(first_group, second_group, strict=True):
            torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def test_hindsight_and_occupancy_update_only_the_direct_follower():
    args = _args()
    agent = _agent()
    world, follower, goal = agent.parameter_groups()
    optimizer = torch.optim.AdamW(follower, lr=1e-3)
    before = [parameter.detach().clone() for parameter in follower]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("follower training evaluated an imagined action or reward head")

    agent.world_model.transition.forward = forbidden
    agent.goal_predictor.forward = forbidden
    agent.reward_predictor.forward = forbidden
    train_follower_step(
        agent,
        optimizer,
        _batch(),
        args,
        torch.device("cpu"),
    )
    assert any(
        not torch.equal(old, parameter)
        for old, parameter in zip(before, follower, strict=True)
    )
    assert all(parameter.grad is None for parameter in world)
    assert all(parameter.grad is None for parameter in goal)


def test_replay_hindsight_does_not_cross_episodes():
    rng = np.random.default_rng(7)
    replay = VectorReplayBuffer(10_000, 2, 3, 2)
    episode = np.zeros(2, dtype=np.float32)
    for step in range(300):
        done = np.array([step % 41 == 40, step % 53 == 52])
        observation = np.repeat(episode[:, None], 3, axis=1)
        replay.add(
            observation,
            np.zeros((2, 2), np.float32),
            np.zeros((2, 2), np.float32),
            episode.copy(),
            done,
        )
        episode += done

    hindsight = replay.sample_hindsight(
        128,
        history=8,
        max_action_offset=16,
        max_occupancy_offset=64,
        discount=0.98,
        rng=rng,
    )
    for key in ("obs", "next_obs", "future_obs"):
        values = hindsight[key][..., 0]
        assert np.all(values == values[:, :1])
    assert np.all(
        hindsight["occupancy_obs"][:, -1, 0]
        == hindsight["occupancy_future_obs"][:, -1, 0]
    )


def test_replay_training_matches_reset_left_padding():
    replay = VectorReplayBuffer(1_000, 1, 2, 1)
    for step in range(12):
        replay.add(
            np.full((1, 2), step, np.float32),
            np.full((1, 1), 0 if step == 5 else step, np.float32),
            np.zeros((1, 1), np.float32),
            np.zeros(1, np.float32),
            np.array([step == 4]),
        )
    obs = replay._padded_history(replay.obs, np.array([0]), np.array([5]), 4)
    incoming = replay._padded_history(
        replay.incoming_action, np.array([0]), np.array([5]), 4, zero_pad=True
    )
    np.testing.assert_array_equal(obs[0, :, 0], np.array([5, 5, 5, 5]))
    np.testing.assert_array_equal(incoming[0, :, 0], np.zeros(4))


def test_follower_can_use_goal_to_disambiguate_same_belief_actions():
    torch.manual_seed(11)
    follower = DirectGoalFollower(
        observation_tokens=2, belief_dim=4, goal_dim=4, act_dim=1, hidden=32
    )
    optimizer = torch.optim.Adam(follower.parameters(), lr=2e-2)
    belief = torch.ones(2, 4)
    current = torch.zeros(2, 4)
    goals = torch.tensor([[2.0, 0, 0, 0], [-2.0, 0, 0, 0]])
    actions = torch.tensor([[0.8], [-0.8]])
    for _ in range(200):
        predicted = follower(belief, current, goals)
        loss = torch.nn.functional.mse_loss(predicted, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    predicted = follower(belief, current, goals)
    assert predicted[0, 0] - predicted[1, 0] > 1.0


def test_follower_has_no_belief_only_action_path():
    follower = DirectGoalFollower(
        observation_tokens=2, belief_dim=4, goal_dim=4, act_dim=2, hidden=16
    )
    belief = torch.randn(7, 4)
    current = torch.randn(7, 4)
    torch.testing.assert_close(
        follower(belief, current, current), torch.zeros(7, 2), rtol=0, atol=0
    )


def test_goal_reward_ascent_freezes_reward_head():
    args = _args()
    args.reward_prediction_coef = 0.0
    args.goal_reward_coef = 1.0
    agent = _agent()
    goal_optimizer = torch.optim.AdamW(
        agent.goal_predictor.parameters(), lr=1e-3, weight_decay=0.0
    )
    reward_optimizer = torch.optim.AdamW(
        agent.reward_predictor.parameters(), lr=1e-3, weight_decay=0.0
    )
    goal_before = [parameter.detach().clone() for parameter in agent.goal_predictor.parameters()]
    reward_before = [parameter.detach().clone() for parameter in agent.reward_predictor.parameters()]
    train_goal_step(
        agent,
        goal_optimizer,
        reward_optimizer,
        _batch(),
        args,
        torch.device("cpu"),
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(goal_before, agent.goal_predictor.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(reward_before, agent.reward_predictor.parameters(), strict=True)
    )

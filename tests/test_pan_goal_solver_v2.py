import copy

import numpy as np
import torch

from cleanrl.pan_goal_solver.ppo_continuous_action_pan_goal_solver_v2 import (
    Args,
    DirectGoalFollower,
    PanGoalSolver,
    VectorReplayBuffer,
    nearest_anchor_mse,
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
        "alternate_future_obs": rng.normal(size=(batch_size, history, 17)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(batch_size, 6)).astype(np.float32),
    }


def test_global_proposer_has_no_current_state_input_and_returns_one_point():
    agent = _agent()
    reward = torch.tensor([25.0])
    goal = agent.proposer(reward)
    assert goal.shape == (1, 16)
    coordinate = torch.asinh(reward / agent.proposer.reward_scale) - torch.asinh(
        torch.tensor(agent.proposer.anchor_reward / agent.proposer.reward_scale)
    )
    expected = agent.proposer.mu + coordinate[:, None] * agent.proposer.direction
    torch.testing.assert_close(goal, expected)


def test_goal_and_follower_belief_are_observation_only():
    agent = _agent()
    obs = torch.randn(5, 8, 17)
    incoming_a = torch.randn(5, 8, 6)
    incoming_b = torch.randn(5, 8, 6)
    with torch.no_grad():
        belief_a, goal_a, _ = agent.follower_state(obs, incoming_a)
        belief_b, goal_b, _ = agent.follower_state(obs, incoming_b)
    torch.testing.assert_close(goal_a, goal_b, rtol=0, atol=0)
    torch.testing.assert_close(belief_a, belief_b, rtol=0, atol=0)


def test_direct_inference_does_not_call_proposer_utility_or_transition():
    agent = _agent()
    follower_calls = 0
    original_forward = agent.follower.forward

    def counted_forward(*args, **kwargs):
        nonlocal follower_calls
        follower_calls += 1
        return original_forward(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("training-only module was called during action inference")

    agent.proposer.forward = forbidden
    agent.utility.forward = forbidden
    agent.world_model.transition.forward = forbidden
    agent.follower.forward = counted_forward
    with torch.no_grad():
        action, current_goal, desired_next = agent.direct_action(
            torch.randn(4, 8, 17), torch.randn(4, 8, 6), torch.randn(16)
        )
    assert action.shape == (4, 6)
    assert current_goal.shape == desired_next.shape == (4, 16)
    assert follower_calls == 1


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
    goal_optimizer = torch.optim.AdamW(goal, lr=1e-3)

    for parameter in agent.parameters():
        parameter.grad = None
    train_follower_step(
        agent,
        follower_optimizer,
        batch,
        args,
        torch.device("cpu"),
        np.random.default_rng(5),
    )
    assert all(parameter.grad is None for parameter in world)
    assert all(parameter.grad is None for parameter in goal)

    for parameter in agent.parameters():
        parameter.grad = None
    reward_rate = np.linspace(-1, 2, 16, dtype=np.float32)
    goal_scale_before = agent.follower.goal_scale.clone()
    train_goal_step(
        agent,
        goal_optimizer,
        batch["obs"],
        reward_rate,
        args,
        torch.device("cpu"),
    )
    assert all(parameter.grad is None for parameter in world)
    assert all(parameter.grad is None for parameter in follower)
    torch.testing.assert_close(agent.follower.goal_scale, goal_scale_before, rtol=0, atol=0)


def test_reward_changes_only_goal_parameters():
    args = _args()
    first = _agent()
    second = copy.deepcopy(first)
    batch = _batch()
    first_goal = first.parameter_groups()[2]
    second_goal = second.parameter_groups()[2]
    first_optimizer = torch.optim.AdamW(first_goal, lr=1e-3)
    second_optimizer = torch.optim.AdamW(second_goal, lr=1e-3)
    reward = np.linspace(-2, 3, 16, dtype=np.float32)
    train_goal_step(first, first_optimizer, batch["obs"], reward, args, torch.device("cpu"))
    train_goal_step(second, second_optimizer, batch["obs"], reward[::-1].copy(), args, torch.device("cpu"))
    assert any(not torch.equal(a, b) for a, b in zip(first_goal, second_goal, strict=True))
    for first_group, second_group in zip(first.parameter_groups()[:2], second.parameter_groups()[:2], strict=True):
        for first_parameter, second_parameter in zip(first_group, second_group, strict=True):
            torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def test_replay_hindsight_and_reward_windows_do_not_cross_episodes():
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
        128, history=8, max_offset=64, discount=0.98, rng=rng
    )
    for key in ("obs", "next_obs", "future_obs", "alternate_future_obs"):
        values = hindsight[key][..., 0]
        assert np.all(values == values[:, :1])
    goal_obs, reward = replay.sample_reward_goals(
        128, history=8, past=4, future=8, rng=rng
    )
    assert np.all(goal_obs[..., 0] == goal_obs[:, :1, 0])
    np.testing.assert_allclose(reward, goal_obs[:, -1, 0])


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
    belief = torch.zeros(2, 4)
    current = torch.zeros(2, 4)
    goals = torch.tensor([[2.0, 0, 0, 0], [-2.0, 0, 0, 0]])
    actions = torch.tensor([[0.8], [-0.8]])
    for _ in range(200):
        _, desired_next = follower(belief, current, goals)
        loss = follower.action_nll(belief, desired_next, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    predicted, _ = follower(belief, current, goals)
    assert predicted[0, 0] - predicted[1, 0] > 1.0


def test_nearest_anchor_diagnostic_returns_scalar_tensor():
    anchors = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    goal = torch.tensor([2.5, 0.0])
    nearest = nearest_anchor_mse(anchors, goal)
    assert nearest.ndim == 0
    torch.testing.assert_close(nearest, torch.tensor(0.125))

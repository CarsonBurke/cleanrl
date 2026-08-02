import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_adaptive_navigator_v1.py"
)
SPEC = importlib.util.spec_from_file_location("embopt_adaptive_navigator_v1", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_factual_pair_targets_use_executed_reward_and_reject_resets():
    rewards = torch.tensor([[1.0], [2.0], [9.0], [4.0]])
    prefix = torch.cat([torch.zeros(1, 1), rewards.cumsum(0)], 0)
    episode_id = torch.tensor([[0], [0], [0], [1], [1]])
    start = torch.tensor([0, 1, 2])
    end = torch.tensor([2, 3, 4])
    env = torch.zeros(3, dtype=torch.long)

    rate, log_steps, valid = MODULE.factual_pair_targets(prefix, episode_id, start, end, env)

    torch.testing.assert_close(rate, torch.tensor([1.5, 5.5, 6.5]))
    torch.testing.assert_close(log_steps.exp(), torch.full((3,), 2.0))
    torch.testing.assert_close(valid, torch.tensor([1.0, 0.0, 0.0]))


def test_adaptive_switch_has_no_age_or_fixed_horizon_trigger():
    current_rate = torch.tensor([2.0, 2.0, 2.0, 2.0])
    candidate_rate = torch.tensor([2.05, 2.2, 100.0, 2.2])
    current_log_steps = torch.tensor([5.0, 5.0, 0.0, 5.0]).exp().log()
    valid = torch.tensor([True, True, True, False])

    switch, arrived, challenger = MODULE.adaptive_goal_switch(
        current_rate, candidate_rate, current_log_steps, valid, margin=0.1, arrival_threshold=1.5
    )

    torch.testing.assert_close(switch, torch.tensor([False, True, True, True]))
    torch.testing.assert_close(arrived, torch.tensor([False, False, True, False]))
    torch.testing.assert_close(challenger, torch.tensor([False, True, False, False]))


def test_navigator_best_of_many_does_not_regress_to_mode_average():
    candidates = torch.tensor(
        [
            [[1.0, 0.0], [-1.0, 0.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
        ]
    )
    target = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    valid = torch.ones(2)

    loss, fit, winner, usage = MODULE.navigator_loss(candidates, target, valid, balance_coef=0.05)

    assert loss.isfinite()
    torch.testing.assert_close(fit, torch.tensor(0.0))
    torch.testing.assert_close(winner, torch.tensor([0, 1]))
    torch.testing.assert_close(usage, torch.tensor([0.5, 0.5]), atol=0.02, rtol=0)


def test_command_targets_ground_exact_goal_until_next_switch():
    reward = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    done = torch.zeros(5, 1)
    switched = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    rate, duration = MODULE.command_segment_targets(reward, done, switched)

    torch.testing.assert_close(rate.squeeze(-1), torch.tensor([1.5, 2.0, 3.5, 4.0]))
    torch.testing.assert_close(duration.squeeze(-1), torch.tensor([2.0, 1.0, 2.0, 1.0]))


def test_goal_losses_cannot_update_lejepa_encoder():
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, (5,), dtype=float),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=float),
    )
    args = SimpleNamespace(latent_dim=8, goal_dim=4, hidden_dim=16, navigator_modes=2)
    agent = MODULE.Agent(env, args)
    obs = torch.randn(7, 5)

    z = agent.encode(obs)
    y = agent.goal_encode(z)
    goal = agent.propose(y)
    rate, arrival = agent.value(y, goal)
    (rate.mean() + arrival.mean()).backward()

    assert all(parameter.grad is None for parameter in agent.encoder.parameters())
    assert any(parameter.grad is not None for parameter in agent.goal_projector.parameters())

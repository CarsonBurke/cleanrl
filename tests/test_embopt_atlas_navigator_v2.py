import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_atlas_navigator_v2.py"
)
SPEC = importlib.util.spec_from_file_location("embopt_atlas_navigator_v2", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_final_observation_is_used_for_terminal_transition():
    next_observation = np.asarray([[10.0, 11.0], [20.0, 21.0]])
    infos = {
        "final_observation": np.asarray(
            [np.asarray([1.0, 2.0]), None], dtype=object
        ),
        "_final_observation": np.asarray([True, False]),
    }

    factual = MODULE.factual_transition_observations(next_observation, infos)

    np.testing.assert_allclose(factual[0], [1.0, 2.0])
    np.testing.assert_allclose(factual[1], [20.0, 21.0])


def test_ordered_pair_targets_span_arbitrary_factual_duration_and_reject_reset():
    rewards = torch.tensor([[1.0], [2.0], [9.0], [4.0]])
    prefix = torch.cat([torch.zeros(1, 1), rewards.cumsum(0)], 0)
    episode = torch.tensor([[0], [0], [1], [1]])
    start = torch.tensor([0, 0, 2])
    end = torch.tensor([1, 3, 3])
    env = torch.zeros(3, dtype=torch.long)

    rate, log_duration, valid = MODULE.ordered_pair_targets(
        prefix, episode, start, end, env
    )

    torch.testing.assert_close(rate, torch.tensor([1.5, 4.0, 6.5]))
    torch.testing.assert_close(log_duration, torch.tensor([2.0, 4.0, 2.0]).log())
    torch.testing.assert_close(valid, torch.tensor([1.0, 0.0, 1.0]))


def test_nearby_atlas_transports_only_factual_motif_displacements():
    current = torch.tensor([[10.0, 0.0], [-10.0, 0.0]])
    starts = torch.tensor([[9.0, 0.0], [-9.0, 0.0], [0.0, 100.0]])
    ends = torch.tensor([[11.0, 1.0], [-12.0, 2.0], [100.0, 100.0]])
    valid_pool = torch.tensor([True, True, False])

    goals, nearest, valid, _ = MODULE.select_nearby_motifs(
        current, starts, ends, valid_pool, candidate_count=1
    )

    torch.testing.assert_close(nearest.squeeze(1), torch.tensor([0, 1]))
    torch.testing.assert_close(goals.squeeze(1), torch.tensor([[12.0, 1.0], [-13.0, 2.0]]))
    assert valid.all()


def test_atlas_endpoint_is_realized_rate_optimal_stopping_not_a_horizon_bin():
    rewards = torch.tensor([[1.0], [4.0], [-10.0], [2.0]])
    prefix = torch.cat([torch.zeros(1, 1), rewards.cumsum(0)], 0)
    episode = torch.zeros(4, 1, dtype=torch.long)
    start = torch.tensor([0, 2])
    env = torch.zeros(2, dtype=torch.long)

    end, rate, valid = MODULE.realized_rate_optimal_endpoints(
        prefix, episode, start, env
    )

    # Start 0 prefers ending after reward 4 (mean 2.5); start 2 prefers extending
    # through reward 2 (mean -4) over stopping immediately at -10.
    torch.testing.assert_close(end, torch.tensor([1, 3]))
    torch.testing.assert_close(rate, torch.tensor([2.5, -4.0]))
    assert valid.all()


def test_pessimistic_score_rejects_unsupported_high_rate_candidate():
    rates = torch.tensor([[[2.0, 2.0, 2.0], [100.0, 100.0, 100.0]]])
    supports = torch.tensor([[[3.0, 3.0, 3.0], [-3.0, -3.0, -3.0]]])

    score, credible, probability = MODULE.pessimistic_pair_score(
        rates,
        supports,
        uncertainty_coef=1.0,
        support_score_coef=0.1,
        minimum_support=0.55,
    )

    assert credible.tolist() == [[True, False]]
    assert torch.isfinite(score[0, 0])
    assert torch.isneginf(score[0, 1])
    assert probability[0, 0] > probability[0, 1]


def test_terminal_edge_disambiguates_pair_features_with_same_goal_point():
    current = torch.tensor([[0.0, 0.0]])
    goal = torch.tensor([[2.0, 1.0]])
    forward_edge = torch.tensor([[1.0, 0.0]])
    backward_edge = -forward_edge

    forward = MODULE.Agent.pair_features(current, goal, forward_edge)
    backward = MODULE.Agent.pair_features(current, goal, backward_edge)

    torch.testing.assert_close(forward[..., :4], backward[..., :4])
    assert not torch.equal(forward, backward)


def test_switching_has_only_arrival_or_credible_rate_causes():
    current_rate = torch.tensor([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    candidate_rate = torch.tensor([[2.01, 2.01], [3.0, 3.0], [3.0, 3.0]])
    current_log_duration = torch.tensor(
        [[math.log(8.0)] * 2, [math.log(8.0)] * 2, [math.log(1.0)] * 2]
    )
    supported = torch.full_like(current_rate, 3.0)
    valid = torch.ones(3, dtype=torch.bool)

    switch, arrived, challenger = MODULE.endogenous_goal_switch(
        current_rate,
        current_log_duration,
        supported,
        candidate_rate,
        supported,
        valid,
        uncertainty_coef=1.0,
        minimum_support=0.55,
        switch_margin=0.05,
        arrival_threshold=1.5,
    )

    torch.testing.assert_close(switch, torch.tensor([False, True, True]))
    torch.testing.assert_close(arrived, torch.tensor([False, False, True]))
    torch.testing.assert_close(challenger, torch.tensor([False, True, False]))


def test_arrival_ends_control_even_without_a_credible_replacement():
    requested = torch.tensor([True, True, False])
    arrived = torch.tensor([True, False, False])
    credible = torch.tensor([False, True, False])
    valid = torch.tensor([True, True, True])

    install, event, next_valid = MODULE.resolve_goal_install(
        requested, arrived, credible, valid
    )

    torch.testing.assert_close(install, torch.tensor([False, True, False]))
    torch.testing.assert_close(event, torch.tensor([True, True, False]))
    torch.testing.assert_close(next_valid, torch.tensor([False, True, True]))


def test_evidence_gate_requires_every_head_and_all_components():
    assert not MODULE.complete_evidence_gate(100, [10, 10, 9], 20, 100, 10, 20)
    assert not MODULE.complete_evidence_gate(99, [10, 10, 10], 20, 100, 10, 20)
    assert not MODULE.complete_evidence_gate(100, [10, 10, 10], 19, 100, 10, 20)
    assert MODULE.complete_evidence_gate(100, [10, 10, 10], 20, 100, 10, 20)


def test_delivered_goal_rate_censors_unfinished_right_edge_commands():
    reward = torch.tensor([[1.0], [3.0], [2.0], [4.0]])
    done = torch.zeros_like(reward)
    switched = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    rate, valid = MODULE.completed_goal_suffix_rates(reward, done, switched)

    torch.testing.assert_close(rate.squeeze(1), torch.tensor([2.0, 3.0, 3.0, 4.0]))
    torch.testing.assert_close(valid.squeeze(1), torch.tensor([1.0, 1.0, 0.0, 0.0]))

    target_rate, log_duration, target_valid = MODULE.completed_goal_suffix_targets(
        reward, done, switched
    )
    torch.testing.assert_close(target_rate, rate)
    torch.testing.assert_close(
        log_duration.squeeze(1).exp(), torch.tensor([2.0, 1.0, 2.0, 1.0])
    )
    torch.testing.assert_close(target_valid, valid)


def test_goal_losses_do_not_update_online_lejepa_encoder():
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, (5,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
    )
    args = SimpleNamespace(
        latent_dim=8,
        goal_dim=4,
        hidden_dim=16,
        value_heads=3,
        navigator_modes=2,
    )
    agent = MODULE.Agent(env, args)
    obs = torch.randn(7, 5)
    z = agent.encode(obs)
    y = agent.goal_encode(z)
    goal = y.roll(1, 0)
    rates, duration, support = agent.pair_values(y, goal, goal - y)

    (rates.mean() + duration.mean() + support.mean()).backward()

    assert all(parameter.grad is None for parameter in agent.encoder.parameters())
    assert any(parameter.grad is not None for parameter in agent.pair_heads.parameters())


def test_no_duration_exponentiation_needed_for_arrival_with_extreme_predictions():
    current_rate = torch.zeros(2, 3)
    candidate_rate = torch.ones(2, 3)
    duration = torch.tensor([[1e30] * 3, [-1e30] * 3])
    support = torch.full((2, 3), 3.0)

    switch, arrived, _ = MODULE.endogenous_goal_switch(
        current_rate,
        duration,
        support,
        candidate_rate,
        support,
        torch.ones(2, dtype=torch.bool),
        uncertainty_coef=1.0,
        minimum_support=0.5,
        switch_margin=2.0,
        arrival_threshold=1.5,
    )

    assert torch.isfinite(duration).all()
    torch.testing.assert_close(arrived, torch.tensor([False, True]))
    torch.testing.assert_close(switch, torch.tensor([False, True]))

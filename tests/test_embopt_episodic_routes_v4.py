import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_episodic_routes_v4.py"
)
SPEC = importlib.util.spec_from_file_location("embopt_episodic_routes_v4", MODULE_PATH)
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


def test_full_replay_ring_slots_remain_chronological_after_wrap():
    before_wrap = MODULE.chronological_replay_slots(3, 3, 5)
    after_wrap = MODULE.chronological_replay_slots(5, 2, 5)

    torch.testing.assert_close(before_wrap, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(after_wrap, torch.tensor([2, 3, 4, 0, 1]))


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


def test_nearby_routes_return_absolute_factual_endpoints_without_transport():
    current = torch.tensor([[10.0, 0.0], [-10.0, 0.0]])
    starts = torch.tensor([[9.0, 0.0], [-9.0, 0.0], [0.0, 100.0]])
    ends = torch.tensor([[11.0, 1.0], [-12.0, 2.0], [100.0, 100.0]])
    valid_pool = torch.tensor([True, True, False])

    goals, nearest, valid, _ = MODULE.retrieve_absolute_routes(
        current, starts, ends, valid_pool, candidate_count=1
    )

    torch.testing.assert_close(nearest.squeeze(1), torch.tensor([0, 1]))
    torch.testing.assert_close(
        goals.squeeze(1), torch.tensor([[11.0, 1.0], [-12.0, 2.0]])
    )
    assert valid.all()


def test_stability_score_rejects_noisy_one_step_spike_for_stable_route():
    rewards = torch.tensor([[10.0], [8.0], [8.0], [-10.0]])
    prefix = torch.cat([torch.zeros(1, 1), rewards.cumsum(0)], 0)
    square_prefix = torch.cat(
        [torch.zeros(1, 1), rewards.square().cumsum(0)], 0
    )
    episode = torch.zeros(4, 1, dtype=torch.long)
    start = torch.tensor([0])
    env = torch.zeros(1, dtype=torch.long)

    end, mean, score, duration, valid = MODULE.stability_adjusted_optimal_endpoints(
        prefix,
        square_prefix,
        episode,
        start,
        env,
        stability_coef=1.5,
        variance_prior=1.0,
    )

    assert end.item() > 0
    assert duration.item() > 1
    assert mean.item() < rewards[0].item()
    assert torch.isfinite(score).all()
    assert valid.all()


def test_stability_score_can_still_choose_genuinely_stable_best_one_step():
    rewards = torch.tensor([[5.0], [4.9], [4.9], [4.9]])
    prefix = torch.cat([torch.zeros(1, 1), rewards.cumsum(0)], 0)
    square_prefix = torch.cat(
        [torch.zeros(1, 1), rewards.square().cumsum(0)], 0
    )
    episode = torch.zeros(4, 1, dtype=torch.long)

    end, mean, _, duration, valid = MODULE.stability_adjusted_optimal_endpoints(
        prefix,
        square_prefix,
        episode,
        torch.tensor([0]),
        torch.tensor([0]),
        stability_coef=1.5,
        variance_prior=1.0,
    )

    torch.testing.assert_close(end, torch.tensor([0]))
    torch.testing.assert_close(duration, torch.tensor([1.0]))
    torch.testing.assert_close(mean, torch.tensor([5.0]))
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


def test_factual_source_rate_directly_drives_atlas_score():
    source = torch.tensor([[8.0, 1.0]])
    equal_head_rate = torch.tensor([[2.0, 2.0]])
    supported = torch.tensor([[3.0, 3.0]])

    score, credible, _ = MODULE.factual_thompson_score(
        source,
        equal_head_rate,
        supported,
        applicability_coef=0.35,
        support_score_coef=0.1,
        minimum_support=0.55,
    )

    assert credible.all()
    assert score[0, 0] > score[0, 1]


def test_thompson_heads_diversify_candidates_and_are_excluded_from_evaluation():
    rates = torch.tensor([[[8.0, 1.0, 4.0], [1.0, 8.0, 4.0]]])
    first_head = torch.tensor([[0, 0]])
    second_head = torch.tensor([[1, 1]])

    first_scores = MODULE.gather_head(rates, first_head)
    second_scores = MODULE.gather_head(rates, second_head)
    first_evaluators = MODULE.leave_one_out(rates, first_head)

    assert first_scores.argmax(-1).item() == 0
    assert second_scores.argmax(-1).item() == 1
    torch.testing.assert_close(first_evaluators[0, 0], torch.tensor([1.0, 4.0]))


def test_terminal_edge_disambiguates_pair_features_with_same_goal_point():
    current = torch.tensor([[0.0, 0.0]])
    goal = torch.tensor([[2.0, 1.0]])
    first_edge = torch.tensor([[0.3, 0.2]])
    forward_terminal_edge = torch.tensor([[1.0, 0.0]])
    backward_terminal_edge = -forward_terminal_edge

    forward = MODULE.Agent.pair_features(
        current, goal, forward_terminal_edge, first_edge
    )
    backward = MODULE.Agent.pair_features(
        current, goal, backward_terminal_edge, first_edge
    )

    torch.testing.assert_close(forward[..., :4], backward[..., :4])
    assert not torch.equal(forward, backward)


def test_first_edge_disambiguates_pair_features_for_same_endpoint_route():
    current = torch.tensor([[0.0, 0.0]])
    goal = torch.tensor([[2.0, 1.0]])
    terminal_edge = torch.tensor([[1.0, 0.0]])
    forward_first_edge = torch.tensor([[0.3, 0.2]])
    backward_first_edge = -forward_first_edge

    forward = MODULE.Agent.pair_features(
        current, goal, terminal_edge, forward_first_edge
    )
    backward = MODULE.Agent.pair_features(
        current, goal, terminal_edge, backward_first_edge
    )

    torch.testing.assert_close(forward[..., :6], backward[..., :6])
    assert not torch.equal(forward, backward)


def test_elite_archive_removes_exact_duplicate_starts_before_truncation():
    starts = torch.tensor([[0.0], [0.0], [1.0], [2.0]])
    scores = torch.tensor([1.0, 4.0, 3.0, 2.0])
    valid = torch.tensor([True, True, True, True])

    selected = MODULE.deduplicated_elite_indices(starts, scores, valid, capacity=3)

    assert selected.tolist() == [1, 2, 3]


def test_evidence_gate_requires_every_head_and_all_components():
    assert not MODULE.complete_evidence_gate(100, [10, 10, 9], 20, 100, 10, 20)
    assert not MODULE.complete_evidence_gate(99, [10, 10, 10], 20, 100, 10, 20)
    assert not MODULE.complete_evidence_gate(100, [10, 10, 10], 19, 100, 10, 20)
    assert MODULE.complete_evidence_gate(100, [10, 10, 10], 20, 100, 10, 20)


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
    first_edge = y.roll(-1, 0) - y
    rates, support = agent.pair_values(y, goal, goal - y, first_edge)

    (rates.mean() + support.mean()).backward()

    assert all(parameter.grad is None for parameter in agent.encoder.parameters())
    assert any(parameter.grad is not None for parameter in agent.pair_heads.parameters())


def test_controller_has_no_learned_navigator_or_duration_head():
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
    assert not hasattr(agent, "navigator")
    assert not hasattr(agent, "state_bridge")
    assert agent.pair_heads[0][-1].out_features == 2


def test_training_source_has_no_commitment_or_navigator_action_path():
    source = MODULE_PATH.read_text()

    assert "agent.navigate(" not in source
    assert "endogenous_goal_switch" not in source
    assert "completed_goal_suffix" not in source
    assert "arrival_threshold" not in source
    assert "route_archive_capacity" in source
    assert "chronological_replay_slots" in source


def test_exact_first_edge_is_local_and_does_not_use_current_query():
    source_start = torch.tensor([[10.0, -3.0]])
    source_next = torch.tensor([[11.5, -2.0]])
    unrelated_current = torch.tensor([[-100.0, 50.0]])

    desired = MODULE.factual_first_edge(source_start, source_next)

    torch.testing.assert_close(desired, torch.tensor([[1.5, 1.0]]))
    assert not torch.equal(desired, source_next - unrelated_current)


def test_waypoint_quality_downranks_failed_transfer():
    desired = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    actual = torch.tensor([[0.9, 0.1], [-1.0, 0.0]])

    quality, relative_error, cosine = MODULE.waypoint_delivery_quality(
        desired, actual
    )

    assert quality[0] > quality[1]
    assert relative_error[0] < relative_error[1]
    assert cosine[0] > cosine[1]

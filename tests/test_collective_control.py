import json

import numpy as np
import pytest
import torch

from cleanrl.collective_control.control import CollectiveController, ObservationAdapter
from cleanrl.collective_control.genome import Genome, Node, Source


def test_genome_round_trip_and_mutation_preserve_interface():
    rng = np.random.default_rng(1)
    genome = Genome.random(rng, observation_dim=5, action_dim=3, node_count=4, evolved_authority=True)
    genome.mutate(rng, 5, 3, max_nodes=8, events=4.0, length_probability=1.0, evolved_authority=True)
    genome.validate(action_dim=3, max_nodes=8, evolved_authority=True)
    restored = Genome.from_json(json.loads(json.dumps(genome.to_json())))
    restored.validate(action_dim=3, max_nodes=8, evolved_authority=True)
    assert restored.outputs == genome.outputs
    assert restored.authority == genome.authority
    assert len(restored.nodes) == len(genome.nodes)


@pytest.mark.cuda
def test_controller_emits_bounded_action_for_generic_dimensions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by the controller")
    rng = np.random.default_rng(2)
    genomes = [Genome.random(rng, 5, 3, 4, evolved_authority=False) for _ in range(3)]
    adapter = ObservationAdapter(np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32))
    controller = CollectiveController(
        genomes,
        adapter,
        np.full(3, -1.0, dtype=np.float32),
        np.full(3, 1.0, dtype=np.float32),
        max_nodes=8,
        authority="uniform",
        device=torch.device("cuda"),
    )
    action = controller.action(np.arange(5, dtype=np.float32))
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


@pytest.mark.cuda
def test_half_cheetah_step_accepts_collective_action():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by the controller")
    import gymnasium as gym

    env = gym.make("HalfCheetah-v4")
    observation, _ = env.reset(seed=1)
    rng = np.random.default_rng(3)
    genomes = [
        Genome.random(
            rng,
            int(np.prod(env.observation_space.shape)),
            int(np.prod(env.action_space.shape)),
            8,
            evolved_authority=False,
        )
        for _ in range(2)
    ]
    adapter = ObservationAdapter(np.zeros(observation.shape, dtype=np.float32), np.ones(observation.shape, dtype=np.float32))
    controller = CollectiveController(
        genomes,
        adapter,
        env.action_space.low,
        env.action_space.high,
        max_nodes=16,
        authority="uniform",
        device=torch.device("cuda"),
    )
    action = controller.action(observation)
    next_observation, reward, terminated, truncated, _ = env.step(action)
    env.close()
    assert next_observation.shape == observation.shape
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated


def _recurrent_teams():
    teams = []
    for team_index in range(2):
        team = []
        for resident in range(2):
            shift = 0.05 * (team_index * 2 + resident)
            nodes = [
                Node(10, (Source(0, 1), Source(1, 20)), np.array([0.1, 0.8, 0.4, 0.9]), 0.7, 0.2 + shift),
                Node(20, (Source(2, 2), Source(1, 10)), np.array([0.8, 0.2, 0.6, 0.3]), 0.4, 0.7 - shift),
                Node(30, (Source(3), Source(4)), np.array([0.9, 0.7, 0.2, 0.1]), 0.9, 0.4),
            ]
            if resident:
                nodes.append(Node(50, (Source(1, 999), Source(0, 4)), np.array([0.2, 0.9, 0.3, 0.6]), 0.6, 0.8))
            # Missing output/authority/source references are legitimate after
            # node deletion and must resolve to half, not a padded slot's state.
            team.append(Genome(nodes, [20, 50 if resident else 30, 10 if resident else 999], 51, 20 if resident else 999))
        teams.append(team)
    return teams


def _reference_reset(teams, mapping, action_dim):
    state = [[{node.node_id: node.initial for node in genome.nodes} for genome in teams[index]] for index in mapping]
    previous = [[np.full(action_dim, 0.5) for _ in teams[index]] for index in mapping]
    return state, previous


def _reference_step(teams, mapping, observations, adapter, low, high, authority, state, previous):
    """Scalar NumPy recurrence independent of fixed-slot tensor compilation."""
    result = []
    for environment, team_index in enumerate(mapping):
        encoded = 0.5 + 0.5 * np.tanh((observations[environment].astype(np.float64) - adapter.mean) / adapter.scale)
        actions, weights = [], []
        for resident, genome in enumerate(teams[team_index]):
            old = state[environment][resident]

            def source_value(source):
                if source.kind == 0:
                    return encoded[np.clip(source.index, 0, len(encoded) - 1)]
                if source.kind == 1:
                    return old.get(source.index, 0.5)
                if source.kind == 2:
                    return previous[environment][resident][np.clip(source.index, 0, len(low) - 1)]
                return 0.0 if source.kind == 3 else 0.5

            updated = {}
            for node in genome.nodes:
                left, right = (source_value(source) for source in node.sources)
                q00, q01, q10, q11 = node.q
                mixed = (1 - left) * (1 - right) * q00 + (1 - left) * right * q01 + left * (1 - right) * q10 + left * right * q11
                updated[node.node_id] = (1 - node.update_rate) * old[node.node_id] + node.update_rate * mixed
            normalized = np.array([updated.get(node_id, 0.5) for node_id in genome.outputs])
            actions.append(low + normalized * (high - low))
            weights.append(updated.get(genome.authority, 0.5) if authority == "evolved" else 1.0)
            state[environment][resident] = updated
            previous[environment][resident] = normalized
        result.append(np.clip(np.average(actions, axis=0, weights=weights), low, high))
    return np.asarray(result)


@pytest.mark.cuda
@pytest.mark.parametrize("authority", ["uniform", "evolved"])
def test_batched_recurrence_reset_remapping_and_genome_reload(authority):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by the controller")
    teams = _recurrent_teams()
    adapter = ObservationAdapter(
        np.linspace(-0.3, 0.3, 5, dtype=np.float32),
        np.linspace(0.5, 1.5, 5, dtype=np.float32),
    )
    low = np.array([-2.0, 0.2, -0.5], dtype=np.float32)
    high = np.array([0.7, 2.0, 3.0], dtype=np.float32)
    controller = CollectiveController(teams, adapter, low, high, 8, authority, torch.device("cuda"))
    rng = np.random.default_rng(4)

    def check_sequence(mapping):
        controller.reset(len(mapping), mapping)
        state, previous = _reference_reset(teams, mapping, len(low))
        retained = []
        for _ in range(12):
            observations = rng.normal(size=(len(mapping), 5)).astype(np.float32)
            expected = _reference_step(teams, mapping, observations, adapter, low, high, authority, state, previous)
            actual = controller.action(observations)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            retained.append((actual, actual.copy()))
        for actual, snapshot in retained:
            np.testing.assert_array_equal(actual, snapshot)

    check_sequence([1, 0, 1])
    check_sequence([0, 1, 0])  # Same shape, different mapping and reset state.
    check_sequence([1, 0, 0, 1])  # Different number of environments.

    # Grow beyond the original compiled width, shrink another resident, and
    # change outputs/authority while retaining the team/resident interface.
    teams = [[genome.clone() for genome in team] for team in teams]
    teams[0][0].nodes.extend([
        Node(60, (Source(2, 1), Source(1, 20)), np.array([0.1, 0.3, 0.9, 0.4]), 0.8, 0.1),
        Node(70, (Source(1, 60), Source(0, 0)), np.array([0.9, 0.1, 0.3, 0.8]), 0.5, 0.9),
    ])
    teams[0][0].outputs = [60, 70, 10]
    teams[0][0].authority = 70
    teams[0][0].next_id = 71
    teams[1][1].nodes = teams[1][1].nodes[:1]
    teams[1][1].outputs = [10, 999, 10]
    teams[1][1].authority = 10
    controller.reload(teams)
    check_sequence([1, 0, 0, 1])
    check_sequence([0, 1, 0])


def _install_fake_evolution(monkeypatch, objective):
    from cleanrl.collective_control import evolve

    calls = []

    class ObjectiveEvaluator:
        def __init__(self, *args):
            self.evaluation_transitions = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def evaluate(self, teams, episode_seeds, horizon):
            calls.append(([[genome.clone() for genome in team] for team in teams], list(episode_seeds), horizon))
            self.evaluation_transitions += len(teams) * len(episode_seeds) * horizon
            return np.asarray(objective(teams, episode_seeds, horizon), dtype=np.float64)

    def blank_genome(*args):
        return Genome([Node(0, (Source(3), Source(3)), np.zeros(4), 1.0, 0.5)], [0], 1)

    def mutate(genome, *args):
        genome.nodes[0].q[0] = 1.0

    monkeypatch.setattr(evolve.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evolve, "environment_dimensions", lambda *args: (1, 1))
    monkeypatch.setattr(evolve, "calibrate_observations",
                        lambda *args: ObservationAdapter(np.zeros(1), np.ones(1)))
    monkeypatch.setattr(evolve, "TeamEvaluator", ObjectiveEvaluator)
    monkeypatch.setattr(Genome, "random", blank_genome)
    monkeypatch.setattr(Genome, "mutate", mutate)
    return evolve, calls


def _fake_config(evolve, path, **overrides):
    values = dict(run_dir=str(path), residents=1, candidates=1, shortlist=1,
                  initial_nodes=1, max_nodes=1, generations=1,
                  proposal_episodes=1, confirmation_episodes=1, development_episodes=1)
    return evolve.Config(**(values | overrides))


def _constant_objective(teams, episode_seeds, horizon):
    return [[sum(genome.nodes[0].q[0] for genome in team)] * len(episode_seeds) for team in teams]


def test_confirmation_accepts_complementary_candidates_sequentially(monkeypatch, tmp_path):
    def complementary_objective(teams, episode_seeds, horizon):
        returns = []
        for team in teams:
            x, y = (genome.nodes[0].q[0] for genome in team)
            # y is harmful alone, but beneficial after x is accepted.
            returns.append([2 * x + (2 * x - 1) * y] * len(episode_seeds))
        return returns

    evolve, calls = _install_fake_evolution(monkeypatch, complementary_objective)

    class BirthOrder:
        def __init__(self):
            self.draws = iter([0, 1])

        def integers(self, *args):
            return next(self.draws)

        def random(self):
            return 0.5

        def permutation(self, count):
            return np.arange(count)

    monkeypatch.setattr(evolve.np.random, "default_rng", lambda seed: BirthOrder())
    config = _fake_config(evolve, tmp_path / "sequential", residents=2, candidates=2)
    run_dir = evolve.train(config)
    checkpoint = json.loads((run_dir / "latest.json").read_text())
    assert checkpoint["champion"]["mean_return"] == 3.0
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[1]["accepted"] == 2
    assert rows[1]["evaluation_transition_budget"] == 72000
    assert checkpoint["cumulative_evaluation_transitions"] == 73000
    assert rows[1]["operators"]["mutation"]["accepted_gain_sum"] == 3.0
    decisions = rows[1]["decisions"]
    assert [decision["screening_gain"] for decision in decisions] == [2.0, 1.0]
    held_out_suites = [set(decision[key]) for decision in decisions for key in ("screening_seeds", "validation_seeds")]
    for index, suite in enumerate(held_out_suites):
        assert all(suite.isdisjoint(other) for other in held_out_suites[index + 1:])
    assert calls[0][1] == calls[-1][1] == evolve.seeds(config.seed, 31, 0, 1)


def test_paired_validation_cancels_common_episode_noise():
    from cleanrl.collective_control.evolve import paired_validation

    incumbent = np.array([-10000, 50000, -30000, 90000], dtype=np.float64)
    result = paired_validation(incumbent, incumbent + 2, 3.0)
    assert result["accepted"]
    assert result["paired_mean_gain"] == 2.0
    assert result["paired_sem"] == 0.0
    assert result["paired_lower_bound"] == 2.0


@pytest.mark.parametrize("differences", [
    np.full(16, -1.0),
    np.array([-7.0, 9.0] * 8),
    np.zeros(16),
])
def test_paired_validation_rejects_harm_uncertainty_and_exact_ties(differences):
    from cleanrl.collective_control.evolve import paired_validation

    result = paired_validation(np.zeros(16), differences, 1.7530503556925547)
    assert not result["accepted"]
    assert result["paired_lower_bound"] <= 0.0
    assert result["rejection"] == "nonpositive_paired_lower_bound"


def test_screen_winner_cannot_accept_on_screening_luck(monkeypatch, tmp_path):
    from cleanrl.collective_control import evolve

    validation_suite = evolve.seeds(1, 1001, 1, 16)

    def objective(teams, episode_seeds, horizon):
        sign = -1 if list(episode_seeds) == validation_suite else 10
        return [[sign * team[0].nodes[0].q[0]] * len(episode_seeds) for team in teams]

    evolve, _ = _install_fake_evolution(monkeypatch, objective)
    run_dir = evolve.train(_fake_config(evolve, tmp_path / "screen"))
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    decision = rows[1]["decisions"][0]
    assert decision["screening_gain"] == 10
    assert decision["paired_mean_gain"] == -1
    assert rows[1]["accepted"] == 0
    checkpoint = json.loads((run_dir / "latest.json").read_text())
    assert checkpoint["champion"]["generation"] == 0
    assert checkpoint["champion"]["mean_return"] == 0


def test_transplant_copies_distinct_parent_without_mutation(monkeypatch, tmp_path):
    evolve, calls = _install_fake_evolution(monkeypatch, _constant_objective)
    initial_values = iter([0.2, 0.8])

    def genome(*args):
        return Genome([Node(0, (Source(3), Source(3)), np.full(4, next(initial_values)), 1.0, 0.5)], [0], 1)

    def mutation_forbidden(*args):
        pytest.fail("transplant must not also mutate")

    class BirthOrder:
        def integers(self, *args):
            return 0

        def random(self):
            return 0.0

        def permutation(self, count):
            return np.arange(count)

    monkeypatch.setattr(Genome, "random", genome)
    monkeypatch.setattr(Genome, "mutate", mutation_forbidden)
    monkeypatch.setattr(evolve.np.random, "default_rng", lambda seed: BirthOrder())
    run_dir = evolve.train(_fake_config(evolve, tmp_path / "transplant", residents=2, transplant_probability=1.0))
    assert [genome.nodes[0].q[0] for genome in calls[1][0][1]] == [0.8, 0.8]
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[1]["operators"]["transplant"]["accepted"] == 1
    assert rows[1]["operators"]["mutation"]["proposed"] == 0


def test_singleton_mutates_and_budget_stop_scores_final_population(monkeypatch, tmp_path):
    evolve, calls = _install_fake_evolution(monkeypatch, _constant_objective)
    run_dir = evolve.train(_fake_config(
        evolve, tmp_path / "budget", generations=10, total_transitions=2000,
        development_every=5, transplant_probability=1.0,
    ))
    checkpoint = json.loads((run_dir / "latest.json").read_text())
    assert checkpoint["generation"] == 1
    assert checkpoint["stop_reason"]["kind"] == "transition_budget"
    assert checkpoint["champion"]["generation"] == 1
    assert checkpoint["champion"]["mean_return"] == 1.0
    assert checkpoint["cumulative_evaluation_transitions"] == 38000
    assert calls[-1][0][0][0].nodes[0].q[0] == 1.0
    assert calls[-1][1] == calls[0][1]
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["operators"]["mutation"]["accepted"] == 1
    assert rows[-1]["development_transitions"] == 1000


def test_plateau_stop_is_successful_scored_checkpoint(monkeypatch, tmp_path):
    evolve, _ = _install_fake_evolution(monkeypatch, lambda teams, suite, horizon: np.zeros((len(teams), len(suite))))
    run_dir = evolve.train(_fake_config(
        evolve, tmp_path / "plateau", generations=20, plateau_warmup_evaluations=2, plateau_patience=2,
    ))
    checkpoint = json.loads((run_dir / "latest.json").read_text())
    assert checkpoint["generation"] == 3
    assert checkpoint["stop_reason"]["kind"] == "development_plateau"
    assert checkpoint["cull_state"]["evaluations"] == 4
    assert checkpoint["cull_state"]["stale_evaluations"] == 2
    assert checkpoint["champion"]["generation"] == 0
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["development_mean_return"] == 0.0
    assert all(row["accepted"] == 0 for row in rows)


def test_raw_or_ema_material_progress_resets_plateau_patience():
    from cleanrl.collective_control.evolve import Config, CullState

    config = Config(plateau_warmup_evaluations=0, plateau_patience=2, plateau_decay=0.8, plateau_material_delta=0.1)
    state = CullState()
    assert not state.update(0.0, config)
    assert not state.update(0.0, config)
    assert not state.update(1.0, config)  # Raw progress resets the accumulated stale evaluation.
    assert state.stale_evaluations == 0
    assert not state.update(1.0, config)  # No new raw best, but EMA materially improves.
    assert state.stale_evaluations == 0
    assert not state.update(-1.0, config)
    assert state.update(-1.0, config)


def test_checkpoint_evaluation_suites_ignore_generation_and_singleton_diagnostic(monkeypatch, tmp_path):
    evolve, _ = _install_fake_evolution(monkeypatch, _constant_objective)
    run_dir = evolve.train(_fake_config(evolve, tmp_path / "checkpoints", generations=0))
    first = run_dir / "latest.json"
    value = json.loads(first.read_text())
    value["generation"] = 999
    value["schema"] = 2
    second = tmp_path / "legacy.json"
    second.write_text(json.dumps(value))
    evaluated_suites = []

    def evaluate(env_id, population, adapter, suite, *args):
        assert population, "singleton diagnostics must never evaluate an empty team"
        evaluated_suites.append(list(suite))
        return np.asarray(suite, dtype=np.float64) % 17

    monkeypatch.setattr(evolve, "evaluate", evaluate)
    first_result = evolve.evaluate_checkpoint(first, 4, 1000, None)
    second_result = evolve.evaluate_checkpoint(second, 4, 1000, None)
    assert first_result["seeds"] == second_result["seeds"]
    assert first_result["returns"] == second_result["returns"]
    assert not first_result["legacy_reinterpreted"]
    assert second_result["legacy_reinterpreted"]
    first_diagnostic = evolve.diagnose_checkpoint(first, 4, 1000)
    second_diagnostic = evolve.diagnose_checkpoint(second, 4, 1000)
    assert first_diagnostic["seeds"] == second_diagnostic["seeds"]
    assert first_diagnostic["leave_one_out_gains"] is None
    assert first_diagnostic["leave_one_out_unavailable_reason"]
    assert set(first_result["seeds"]).isdisjoint(first_diagnostic["seeds"])

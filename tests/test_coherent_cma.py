"""Distribution and orchestration contracts; no substitute MuJoCo mini-training."""
import json

import numpy as np
import pytest

from cleanrl import coherent_cma_v10 as trainer
from cleanrl.collective_control.control import ObservationAdapter
from cleanrl.collective_control.evolve import seeds


class RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, generation):
        self.scalars[name] = value

    def add_text(self, *args):
        pass

    def flush(self):
        pass


class SuiteEvaluator:
    """Explicit fake at the rollout boundary, not a policy or training benchmark."""
    observation_dim = 1
    action_dim = 1
    parameter_dim = 2
    action_low = np.array([-2.0], dtype=np.float32)
    action_high = np.array([3.0], dtype=np.float32)

    def __init__(self, config):
        self.evaluation_transitions = 0
        self.development_seeds = seeds(config.seed, 31, 0, config.development_episodes)
        self.calls = []
        self.development_calls = 0

    def evaluate(self, parameters, suite, horizon, *, blind=False):
        self.calls.append((parameters.copy(), list(suite), horizon, blind))
        self.evaluation_transitions += len(parameters) * len(suite) * horizon
        if list(suite) == self.development_seeds:
            self.development_calls += 1
            # Later rotating train winners must not replace the baseline champion
            # when their scores on the fixed development suite regress.
            score = np.full(len(parameters), 100.0 if self.development_calls == 1 else -10.0)
        else:
            score = 10000.0 - np.square(parameters - 0.4).sum(axis=1)
            if blind:
                score -= 7.0
        return score[:, None] + np.arange(len(suite))[None, :] * 0.25


def setup_loop(tmp_path, **overrides):
    values = dict(population=4, generations=2, total_transitions=0, horizon=5,
                  train_episodes=2, development_episodes=3, development_every=10,
                  final_episodes=4, plateau_patience=0)
    values.update(overrides)
    config = trainer.Config(**values)
    evaluator = SuiteEvaluator(config)
    adapter = ObservationAdapter(np.array([2.0], dtype=np.float32), np.array([3.0], dtype=np.float32))
    writer = RecordingWriter()
    return config, evaluator, adapter, writer


def test_real_cma_optimizes_rotated_anisotropic_objective():
    config = trainer.Config(population=16, sigma=0.5, seed=17)
    optimizer = trainer.make_optimizer(config, 4)
    rotation, _ = np.linalg.qr(np.random.default_rng(19).normal(size=(4, 4)))
    target = np.array([1.0, -2.0, 0.5, 1.5])

    def objective(points):
        residual = (np.asarray(points) - target) @ rotation
        return (residual ** 2 * np.array([1.0, 4.0, 20.0, 100.0])).sum(axis=-1)

    initial = objective(optimizer.mean)
    for _ in range(180):
        population = optimizer.ask()
        optimizer.tell(population, objective(population).tolist())
        if optimizer.stop():
            break
    # A winner-only fixed/isotropic search or incorrect fitness sign will not
    # reliably learn this narrow, rotated basin with the same sample budget.
    assert objective(optimizer.mean) < initial * 1e-7
    np.testing.assert_allclose(optimizer.mean, target, atol=1e-3)


def test_zero_seed_is_reproducible_without_sharing_global_numpy_state():
    config = trainer.Config(population=8, seed=0)
    left = trainer.make_optimizer(config, 4)
    right = trainer.make_optimizer(config, 4)
    global_state = np.random.get_state()
    try:
        for _ in range(4):
            left_population = left.ask()
            np.random.standard_normal(100)
            right_population = right.ask()
            np.testing.assert_array_equal(left_population, right_population)
            losses = np.square(np.asarray(left_population) - 1).sum(axis=1).tolist()
            left.tell(left_population, losses)
            right.tell(right_population, losses)
    finally:
        np.random.set_state(global_state)


def test_champion_changes_only_for_fixed_suite_improvement():
    params = np.array([[1.0, 2.0], [3.0, 4.0]])
    champion = trainer.select_champion(None, params, np.array([[5.0, 7.0], [2.0, 3.0]]),
                                      10, ["cma_mean", "population_best"])
    regressed = trainer.select_champion(champion, params + 10, np.array([[0.0, 1.0], [5.0, 7.0]]),
                                       20, ["cma_mean", "population_best"])
    assert regressed is champion
    improved = trainer.select_champion(champion, params, np.array([[2.0, 3.0], [8.0, 9.0]]),
                                      30, ["cma_mean", "population_best"])
    params[:] = -999
    assert improved.source == "population_best"
    np.testing.assert_array_equal(improved.parameters, [3.0, 4.0])
    np.testing.assert_array_equal(champion.parameters, [1.0, 2.0])


@pytest.mark.parametrize("generations", [0, 2])
def test_generation_stop_scores_final_mean_and_population_on_fixed_suite(tmp_path, generations):
    config, evaluator, adapter, writer = setup_loop(tmp_path, generations=generations)
    champion, summary = trainer.run_evolution(config, evaluator, adapter, tmp_path, writer)
    assert summary["stop_reason"]["kind"] == "generation_limit"
    assert summary["generation"] == generations
    dev_calls = [call for call in evaluator.calls if call[1] == evaluator.development_seeds]
    assert len(dev_calls) == (1 if generations == 0 else 2)
    assert len(dev_calls[-1][0]) == (1 if generations == 0 else 2)
    assert evaluator.calls[-1][1] == evaluator.development_seeds
    assert champion.generation == 0
    np.testing.assert_array_equal(champion.parameters, np.zeros(2))
    train_calls = [call for call in evaluator.calls if call[1] != evaluator.development_seeds]
    for generation, call in enumerate(train_calls, 1):
        assert call[1] == seeds(config.seed, 101, generation, config.train_episodes)
        assert len(call[0]) == config.population
    if len(train_calls) > 1:
        assert train_calls[0][1] != train_calls[1][1]
    final_suite = set(seeds(config.seed, 701, 0, config.final_episodes))
    assert not any(final_suite.intersection(call[1]) for call in evaluator.calls)


def test_budget_preserves_generation_and_forced_development_overshoot(tmp_path):
    config, evaluator, adapter, writer = setup_loop(tmp_path, generations=100, total_transitions=70)
    _, summary = trainer.run_evolution(config, evaluator, adapter, tmp_path, writer)
    # 15 baseline-dev + two 40-step populations + 30 final-dev transitions.
    assert summary["generation"] == 2
    assert summary["stop_reason"]["kind"] == "transition_budget"
    assert summary["cumulative_evaluation_transitions"] == 125
    assert summary["transition_budget_overshoot"] == 55
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["cumulative_evaluation_transitions"] == 125
    assert latest["stop_reason"] == summary["stop_reason"]


def test_library_stop_forces_development_without_retry(tmp_path, monkeypatch):
    config, evaluator, adapter, writer = setup_loop(tmp_path, generations=100)
    optimizer = trainer.make_optimizer(config, evaluator.parameter_dim)
    monkeypatch.setattr(optimizer, "stop", lambda: {"tolconditioncov": 1e14} if optimizer.countiter else {})
    monkeypatch.setattr(trainer, "make_optimizer", lambda *_: optimizer)
    _, summary = trainer.run_evolution(config, evaluator, adapter, tmp_path, writer)
    assert summary["generation"] == 1
    assert summary["stop_reason"] == {"kind": "cma_stop", "library_reasons": {"tolconditioncov": 1e14}}
    assert evaluator.calls[-1][1] == evaluator.development_seeds
    assert len(evaluator.calls[-1][0]) == 2
    assert json.loads((tmp_path / "latest.json").read_text())["stop_reason"] == summary["stop_reason"]


def test_time_limit_between_cadences_forces_development(tmp_path, monkeypatch):
    config, evaluator, adapter, writer = setup_loop(tmp_path, generations=100, time_limit_seconds=1)
    monkeypatch.setattr(
        trainer.time, "monotonic",
        lambda: 2.0 if any(call[1] != evaluator.development_seeds for call in evaluator.calls) else 0.0,
    )
    _, summary = trainer.run_evolution(config, evaluator, adapter, tmp_path, writer)
    assert summary["generation"] == 1
    assert summary["stop_reason"]["kind"] == "time_limit"
    assert evaluator.calls[-1][1] == evaluator.development_seeds
    assert len(evaluator.calls[-1][0]) == 2


def test_plateau_checkpoints_champion_before_successful_autocull(tmp_path, capsys):
    config, evaluator, adapter, writer = setup_loop(tmp_path, generations=100, development_every=1,
                                                  plateau_patience=1, plateau_warmup_evaluations=0)
    champion, summary = trainer.run_evolution(config, evaluator, adapter, tmp_path, writer)
    assert summary["stop_reason"]["kind"] == "development_plateau"
    assert summary["generation"] == 1
    assert champion.generation == 0
    assert '"event": "AUTOCULL"' in capsys.readouterr().out
    state = json.loads((tmp_path / "champion.json").read_text())
    assert state["stop_reason"] == summary["stop_reason"]
    assert state["cull_state"]["stale_evaluations"] == 1


def test_checkpoint_roundtrip_and_final_suite_do_not_mutate_champion(tmp_path):
    config, evaluator, adapter, writer = setup_loop(tmp_path)
    champion, summary = trainer.run_evolution(config, evaluator, adapter, tmp_path, writer)
    checkpoint = tmp_path / "champion.json"
    saved = checkpoint.read_bytes()
    restored_config, restored_adapter, restored_parameters, state = trainer.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(restored_adapter.mean, adapter.mean)
    np.testing.assert_array_equal(restored_adapter.scale, adapter.scale)
    np.testing.assert_array_equal(restored_parameters, champion.parameters)
    baseline = evaluator.evaluate(champion.parameters[None], evaluator.development_seeds, config.horizon)
    restored = evaluator.evaluate(restored_parameters[None], evaluator.development_seeds, config.horizon)
    np.testing.assert_array_equal(restored, baseline)
    assert restored_config == config
    assert state["action_low"] == [-2.0] and state["action_high"] == [3.0]
    training_seeds = {seed for call in evaluator.calls for seed in call[1]}
    result = trainer.evaluate_final(evaluator, restored_parameters, config.seed, config.final_episodes, config.horizon)
    assert not training_seeds.intersection(result["seeds"])
    assert evaluator.calls[-2][1] == evaluator.calls[-1][1] == result["seeds"]
    assert evaluator.calls[-2][3] is False and evaluator.calls[-1][3] is True
    np.testing.assert_allclose(np.asarray(result["returns"]) - result["blind_returns"], 7.0)
    assert result["blind_paired_gain"] == 7.0
    assert result["blind_paired_sem"] == 0.0
    assert result["final_evaluation_transitions"] == 2 * config.final_episodes * config.horizon
    assert checkpoint.read_bytes() == saved
    assert champion.generation == 0 and summary["generation"] == 2


def test_checkpoint_refuses_different_policy_contract(tmp_path):
    config, evaluator, adapter, writer = setup_loop(tmp_path, generations=0)
    trainer.run_evolution(config, evaluator, adapter, tmp_path, writer)
    checkpoint = tmp_path / "champion.json"
    value = json.loads(checkpoint.read_text())
    value["policy_contract"]["observation"] = "tanh(z)"
    checkpoint.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="policy contract"):
        trainer.load_checkpoint(checkpoint)

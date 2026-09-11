"""Causal targets, prefix-only locks, provenance and graph/export contracts."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from cleanrl.plasticity import panel_hd
from cleanrl.plasticity import panel_return_hierarchy_eval_v7 as evaluation
from cleanrl.plasticity.panel_predictive_memory_v4 import CausalState
from cleanrl.plasticity.panel_return_hierarchy_v7 import Config, GROUPS, Learner, configurations
from cleanrl.plasticity.panel_return_representation_v5 import Config as NeuralConfig, Learner as NeuralLearner

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; queue via mlq")


def close_fixture():
    changes = np.random.default_rng(27).normal(0., .004, (96, 5))
    changes[55, 2] = .15
    return (100 * np.exp(changes.cumsum(0))).astype(np.float32)


def prefix_raw(configs, bars=25):
    count = len(configs) + len(evaluation.REFERENCE_NAMES)
    raw = np.zeros((bars, count, len(evaluation.METRICS)), dtype=np.float64)
    raw[:, :, 0] = np.arange(count)[None] + 1
    raw[:, :, 3:5] = 4
    raw[:, :, 5] = np.arange(count)[::-1][None] + 1
    raw[:, :, 8:10] = 4
    return raw


def test_masks_unclipped_target_rms_and_raw_zero_are_aligned():
    close = close_fixture()
    close[9, 1:] = np.nan
    close[45, 2] = np.nan
    close[65, 3] *= 2
    args = evaluation.Args(min_coverage=.5)
    series = evaluation.return_series(args, close)
    z, _, _, _, observed = panel_hd.series(args, close, np.ones_like(close))
    mask = observed & np.roll(observed, -1, axis=0)
    mask[:33], mask[-1] = False, False
    np.testing.assert_array_equal(series["valid"], mask)
    t = 53
    np.testing.assert_array_equal(series["normalized_target"][t],
        series["raw_return"][t + 1, 1:] / np.maximum(series["scale"][t, 1:], 1e-6))
    assert series["normalized_target"][t, 1] > 10 and z[t + 1, 2] == 10
    y, raw, rms, valid = [torch.from_numpy(series[name][t]) for name in
                         ("normalized_target", "raw_target", "denominator", "valid")]
    row = evaluation.metric_row(torch.zeros((1, 4)), y, raw, rms, valid)[0]
    assert row[0] == y[valid].square().sum()
    assert row[5] == raw[valid].square().sum()
    assert row[6] == row[7] == 0
    # Raw forecasts use the known RMS; original FP32 target division can round.
    perfect = evaluation.metric_row(y[None], y, raw, rms, valid)[0]
    assert perfect[0] == 0
    raw_forecast = y.double() * rms.double()
    assert perfect[5] == (raw_forecast[valid] - raw.double()[valid]).square().sum()
    torch.testing.assert_close(raw_forecast[valid], raw.double()[valid], rtol=1e-7, atol=0.)


def test_future_labels_do_not_change_signed_features_and_null_is_fresh_paired_prefix_stable():
    close = close_fixture()
    changed = close.copy()
    changed[61:] *= np.linspace(.8, 1.2, 35)[:, None]
    left, right = [evaluation.return_series(evaluation.Args(), c) for c in (close, changed)]
    for name in ("scale", "observed_signed", "observed_valid"):
        np.testing.assert_array_equal(left[name][:61], right[name][:61])
    ridges = [evaluation.reference.RidgeReference(s["observed_signed"], s["observed_valid"], "cpu") for s in (left, right)]
    torch.testing.assert_close(ridges[0].features(torch.tensor(60)), ridges[1].features(torch.tensor(60)), rtol=0, atol=0)
    assert not np.array_equal(left["normalized_target"][60], right["normalized_target"][60])
    y = np.arange(400, dtype=np.float64).reshape(100, 4) / 37
    raw = y * .004
    signed, raw_signed, identity = evaluation.randomized_labels(y, raw)
    prefix, raw_prefix, _ = evaluation.randomized_labels(y[:61], raw[:61])
    np.testing.assert_array_equal(signed[:61], prefix)
    np.testing.assert_array_equal(raw_signed[:61], raw_prefix)
    np.testing.assert_array_equal(signed ** 2, y ** 2)
    np.testing.assert_array_equal(raw_signed, signed * .004)
    assert identity == evaluation.randomized_labels(y, raw)[2]
    assert not np.array_equal(signed, evaluation.reference.randomized_labels(y, raw)[0])


def test_only_normalized_selection_window_can_choose_new_locks(tmp_path):
    configs = configurations()
    raw = prefix_raw(configs, 20)
    raw[0, 0, 0] = np.inf
    health = np.ones(len(configs), dtype=bool)
    health[1] = False
    locks = evaluation.select_locks(raw, configs, 10, 20, 30, 30, health)
    assert locks[GROUPS[0]]["column"] == 2
    changed = raw.copy()
    changed[:, :, 5:] *= -1e12
    changed[:10, 2:, 0] *= 1e12
    assert evaluation.select_locks(changed, configs, 10, 20, 30, 30, health) == locks
    with pytest.raises(ValueError, match="prefix"):
        evaluation.select_locks(np.concatenate((raw, raw[:1])), configs, 10, 20, 30, 31)
    for group in GROUPS:
        columns = [i for i, c in enumerate(configs) if c.family == group]
        failed = health.copy()
        failed[columns] = False
        assert evaluation.select_locks(raw, configs, 10, 20, 30, 30, failed)[group]["column"] is None


def reference_fixture(tmp_path):
    configs = configurations()
    raw = prefix_raw(configs)
    identity = {"stream_start": 10, "selection_start": 20, "cut": 30, "stream_stop": 35,
                "preprocessing": {"rank_offset": 0}}
    neural = {"sha256": "frozen-neural-result", "config": {"family": "memory_adam", "lr": .001, "beta2": .99, "aux_weight": 0.}}
    result = {"version": 7, "stage": "development", "view": "real", "status": "completed",
              "configs": [asdict(c) for c in configs], "data_identity": identity, "source_sha256": {"model": "fixed"},
              "neural_reference": neural, "prefix_health": [True] * len(configs), "consumed_until_exclusive": 30,
              "locks": evaluation.select_locks(raw[:20], configs, 10, 20, 30, 30), "source_and_stream_unchanged": True,
              "raw_metric_bounds": [10, 35]}
    evaluation.persist_prefix_locks(tmp_path, raw[:20], result)
    result["consumed_until_exclusive"] = 35
    result["artifacts"] = {"raw_metrics": "raw_metrics.npz"}
    return configs, raw, result


def save_reference(tmp_path, raw, result):
    np.savez(tmp_path / "raw_metrics.npz", bar_index=np.arange(10, 35),
             **{key: raw[:, :, i] for i, key in enumerate(evaluation.METRICS)})
    result["artifact_sha256"] = {"raw_metrics": evaluation.frozen.file_sha256(tmp_path / "raw_metrics.npz")}
    evaluation.frozen.save_json(tmp_path / "results.json", result)


def test_authenticated_locks_reconstruct_prefix_and_reject_data_source_neural_and_export_tampering(tmp_path):
    configs, raw, result = reference_fixture(tmp_path)
    save_reference(tmp_path, raw, result)
    path = tmp_path / "results.json"
    load = lambda: evaluation.load_reference(path, result["source_sha256"], configs, result["neural_reference"])[0]
    original = load()
    raw[20:, :, 0] = np.arange(raw.shape[1])[::-1] * 1e8
    save_reference(tmp_path, raw, result)
    assert load() == original
    with pytest.raises(ValueError, match="source"):
        evaluation.load_reference(path, {"model": "changed"}, configs, result["neural_reference"])
    with pytest.raises(ValueError, match="neural"):
        evaluation.load_reference(path, result["source_sha256"], configs, {"sha256": "other"})
    result["raw_metric_bounds"] = [10, 34]
    save_reference(tmp_path, raw, result)
    with pytest.raises(ValueError, match="bounds"):
        load()
    result["raw_metric_bounds"] = [10, 35]
    result["data_identity"]["cut"] = 31
    save_reference(tmp_path, raw, result)
    with pytest.raises(ValueError, match="provenance"):
        load()
    result["data_identity"]["cut"] = 30
    raw[0, 0, 0] += 1
    save_reference(tmp_path, raw, result)
    with pytest.raises(ValueError, match="provenance"):
        load()


def test_neural_choice_is_authenticated_from_actual_v6_prefix_not_suffix(tmp_path, monkeypatch):
    ref = evaluation.reference
    configs = ref.configurations()
    raw = np.zeros((25, len(configs) + 2, 10), dtype=np.float64)
    raw[:, :, 0] = np.arange(raw.shape[1]) + 1
    raw[:, :, 3:5] = 4
    raw[:, :, 8:10] = 4
    hashes = {"frozen": "identity"}
    monkeypatch.setattr(ref, "source_hashes", lambda: hashes)
    identity = {"stream_start": 10, "selection_start": 20, "cut": 30, "stream_stop": 35}
    result = {"version": 6, "stage": "development", "view": "real", "status": "completed",
              "configs": [asdict(c) for c in configs], "data_identity": identity, "source_sha256": hashes,
              "prefix_health": [True] * len(configs), "consumed_until_exclusive": 30,
              "locks": ref.select_locks(raw[:20], configs, 10, 20, 30, 30), "source_and_stream_unchanged": True,
              "source_sha256_after": hashes, "stream_sha256_before": {"actual": "bytes"}, "stream_sha256_after": {"actual": "bytes"}}
    ref.persist_prefix_locks(tmp_path, raw[:20], result)
    result["consumed_until_exclusive"] = 35
    result["artifacts"] = {"raw_metrics": "raw_metrics.npz"}
    save_reference(tmp_path, raw, result)
    config, _ = evaluation.load_neural_reference(tmp_path / "results.json")
    assert asdict(config) == result["locks"]["memory_adam"]["config"]
    raw[20:, :, 0] = np.arange(raw.shape[1])[::-1] * 1e9
    save_reference(tmp_path, raw, result)
    assert evaluation.load_neural_reference(tmp_path / "results.json")[0] == config
    raw[10, result["locks"]["memory_adam"]["column"], 0] += 100
    save_reference(tmp_path, raw, result)
    with pytest.raises(ValueError, match="provenance"):
        evaluation.load_neural_reference(tmp_path / "results.json")


def test_cohort_cache_routing_protects_real_data_and_uses_own_suffix(tmp_path):
    for stage, rank in (("development", 0), ("replay", 400), ("confirmation", 600)):
        args = evaluation.Args(output_dir="unused", stage=stage, rank_offset=rank,
                               reference_result="real/results.json" if rank else "")
        evaluation.validate_args(args)
        assert args.cache == evaluation.CACHE_PATHS[rank]
    with pytest.raises(ValueError, match="reserved cache"):
        evaluation.validate_args(evaluation.Args(output_dir="unused", stage="confirmation", rank_offset=600,
                                  reference_result="real/results.json", cache=evaluation.CACHE_PATHS[400]))
    path = tmp_path / "cache.npz"
    np.savez(path, rank_offset=400, n_stocks=200, sentinel=np.arange(4))
    digest = evaluation.frozen.file_sha256(path)
    with pytest.raises(ValueError, match="another cohort"):
        evaluation.protect_cache(evaluation.Args(cache=str(path), rank_offset=600))
    assert evaluation.frozen.file_sha256(path) == digest
    bank = type("Bank", (), {"T": 96, "N": 4, "L": 32, "F": 257, "cut": 57})()
    args = evaluation.Args(cache=str(path), rank_offset=600)
    close = close_fixture()
    series = evaluation.return_series(args, close)
    identity = evaluation.data_identity(args, bank, close, ["A", "B", "C", "D"], np.arange(96), series)
    assert identity["cut"] == 57
    assert identity["calendar_boundaries"]["cut"] == {"bar": 57, "timestamp": "57"}


def test_durable_bounds_never_advertise_unwritten_prediction_pages(tmp_path):
    journal = np.lib.format.open_memmap(tmp_path / "metrics.npy", mode="w+", dtype=np.float64, shape=(5, 2, 10))
    predictions = np.lib.format.open_memmap(tmp_path / "predictions.npy", mode="w+", dtype=np.float64, shape=(5, 2, 4))
    journal[:3] = np.arange(60).reshape(3, 2, 10)
    predictions[:2] = .25
    result = {"status": "interrupted", "checkpoint_raw_bounds": [10, 13], "prediction_valid_until_exclusive": 12}
    evaluation.persist_progress(tmp_path, result, journal, predictions)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["checkpoint_raw_bounds"] == [10, 13]
    assert saved["prediction_valid_until_exclusive"] == 12
    np.testing.assert_array_equal(np.load(tmp_path / "metrics.npy")[:3], journal[:3])
    np.testing.assert_array_equal(np.load(tmp_path / "predictions.npy")[:2], np.full((2, 2, 4), .25))


@cuda
@torch.no_grad()
def test_graph_restore_label_causality_and_shared_offline_ridge_fit(monkeypatch):
    close = close_fixture()
    args = evaluation.Args()
    series = evaluation.return_series(args, close)
    z, vs, cs, acs, valid = panel_hd.series(args, close, np.ones_like(close))
    bank = panel_hd.Bank.__new__(panel_hd.Bank)
    bank.T, bank.N, bank.L, bank.F, bank.cut = 96, 4, 32, 257, 57
    bank.dev = torch.device("cuda")
    bank.zt, bank.vst, bank.cst, bank.acst = [torch.tensor(a, device=bank.dev) for a in (z, vs, cs, acs)]
    bank.mu = torch.tensor(0., device=bank.dev)
    bank.ones = torch.ones((bank.N, 1), device=bank.dev)
    bank.lag_idx = torch.arange(1, 33, device=bank.dev)
    state = CausalState(bank, torch.tensor(valid, device=bank.dev))
    state.own.zero_()
    state.cross.zero_()
    for t in range(33):
        state.observe(torch.tensor(t, device=bank.dev))
    configs = (Config(GROUPS[0], 2.), Config(GROUPS[1], 2.), Config(GROUPS[2], 2., 3.))
    model = Learner(bank.N, 13, configs)
    neural = NeuralLearner(bank.F, 128, (NeuralConfig("memory_adam", .001),), bank.dev, num_samples=bank.N)
    runner = evaluation.Runner(bank, model, neural, series, state, 32, 33, 95)
    initial = [t.clone() for t in runner.mutable]
    graphs = runner.capture()
    for actual, expected in zip(runner.mutable, initial):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graphs[1].replay()
    torch.cuda.synchronize()
    prediction = runner.prediction_ring[0].clone()
    for actual, expected in zip(runner.mutable, initial):
        actual.copy_(expected)
    runner.target[33] *= -10
    runner.raw_target[33] *= -10
    graphs[1].replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(runner.prediction_ring[0], prediction, rtol=0, atol=0)
    runner.fit_reference()
    x = runner.ridge.features(torch.tensor(33, device="cuda"))
    mask = runner.valid[33]
    expected = torch.linalg.solve(x[mask].T @ x[mask] + torch.eye(13, device="cuda", dtype=torch.float64),
                                  x[mask].T @ runner.target[33, mask])
    torch.testing.assert_close(runner.ridge.coefficient, expected, rtol=1e-9, atol=1e-10)
    before_failure = [t.clone() for t in runner.mutable]

    def fail_capture(*args, **kwargs):
        raise RuntimeError("injected graph failure")

    monkeypatch.setattr(torch.cuda, "graph", fail_capture)
    with pytest.raises(RuntimeError, match="injected graph failure"):
        runner.capture()
    for actual, expected in zip(runner.mutable, before_failure):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

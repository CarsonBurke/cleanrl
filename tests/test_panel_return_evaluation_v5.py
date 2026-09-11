"""Small behavioral contracts; CUDA checks must be queued through mlq."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from cleanrl.plasticity import panel_hd
from cleanrl.plasticity import panel_return_representation_eval_v5 as evaluation
from cleanrl.plasticity.panel_predictive_memory_v4 import CausalState
from cleanrl.plasticity.panel_return_representation_v5 import Config, GROUPS, Learner, configurations

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contracts; queue with mlq")


def close_fixture():
    rng = np.random.default_rng(27)
    changes = rng.normal(0., .004, (96, 5))
    changes[55, 2] = .15  # deliberately beyond ten previous-RMS units, below raw filter
    return (100 * np.exp(changes.cumsum(0))).astype(np.float32)


def test_target_scale_alignment_raw_zero_and_bank_mask_agreement():
    close = close_fixture()
    close[9, 1:] = np.nan  # dropped coverage bar, so returns span the gap
    close[45, 2] = np.nan  # retain this bar with a missing stock
    close[65, 3] *= 2  # abs>.2 invalidates both adjacent returns
    args = evaluation.Args(min_coverage=.5)
    series = evaluation.return_series(args, close)
    z, _, _, _, observed = panel_hd.series(args, close, np.ones_like(close))
    expected_mask = observed & np.roll(observed, -1, axis=0)
    expected_mask[:args.lags + 1] = False
    expected_mask[-1] = False
    np.testing.assert_array_equal(series["valid"], expected_mask)
    np.testing.assert_array_equal(series["observed_valid"], observed)
    t = 53  # original bar54 predicts the deliberately large return at55
    expected = series["raw_return"][t + 1, 1:] / np.maximum(series["scale"][t, 1:], 1e-6)
    np.testing.assert_array_equal(series["normalized_target"][t], expected)
    assert series["normalized_target"][t, 1] > 10
    assert z[t + 1, 2] == 10  # feature clipping unchanged, label not clipped
    assert np.isfinite(series["scale"]).all()
    y = torch.from_numpy(series["normalized_target"][t])
    raw = torch.from_numpy(series["raw_target"][t])
    denominator = torch.from_numpy(series["denominator"][t])
    mask = torch.from_numpy(series["valid"][t])
    row = evaluation.metric_row(torch.zeros((1, 4)), y, raw, denominator, mask)[0]
    assert row[5].item() == torch.where(mask, raw, 0.).square().sum().item()
    assert row[6].item() == row[7].item() == 0.
    assert row[0].item() == torch.where(mask, y, 0.).square().sum().item()


def test_future_perturbation_preserves_scale_features_and_current_forecast_inputs():
    close = close_fixture()
    args = evaluation.Args()
    changed = close.copy()
    changed[61:] *= np.linspace(.8, 1.2, len(changed) - 61)[:, None]
    first = evaluation.return_series(args, close)
    second = evaluation.return_series(args, changed)
    for name in ("raw_return", "scale", "observed_signed", "observed_valid"):
        np.testing.assert_array_equal(first[name][:61], second[name][:61])
    np.testing.assert_array_equal(first["normalized_target"][:60], second["normalized_target"][:60])
    old_features = panel_hd.series(args, close, np.ones_like(close))
    new_features = panel_hd.series(args, changed, np.ones_like(close))
    for first_feature, second_feature in zip(old_features, new_features):
        np.testing.assert_array_equal(first_feature[:61], second_feature[:61])
    assert not np.array_equal(first["normalized_target"][60], second["normalized_target"][60])
    ridge_a = evaluation.RidgeReference(first["observed_signed"], first["observed_valid"], "cpu")
    ridge_b = evaluation.RidgeReference(second["observed_signed"], second["observed_valid"], "cpu")
    torch.testing.assert_close(ridge_a.features(torch.tensor(60)), ridge_b.features(torch.tensor(60)), rtol=0, atol=0)


def test_rademacher_pairs_labels_preserves_volatility_and_is_prefix_stable():
    normalized = np.arange(400, dtype=np.float64).reshape(100, 4) / 37
    raw = normalized * .004
    signed, raw_signed, fingerprint = evaluation.randomized_labels(normalized, raw)
    prefix, raw_prefix, _ = evaluation.randomized_labels(normalized[:61], raw[:61])
    np.testing.assert_array_equal(signed[:61], prefix)
    np.testing.assert_array_equal(raw_signed[:61], raw_prefix)
    np.testing.assert_array_equal(np.square(signed), np.square(normalized))
    np.testing.assert_array_equal(raw_signed, signed * .004)
    assert np.any(signed < 0) and np.any(signed > 0)
    assert fingerprint == evaluation.randomized_labels(normalized, raw)[2]


def test_fixed_ridge_solve_and_suffix_labels_do_not_change_forecasts():
    signed = np.arange(96 * 5, dtype=np.float64).reshape(96, 5) / 200
    valid = np.ones((96, 4), dtype=bool)
    ridge = evaluation.RidgeReference(signed, valid, "cpu")
    t = torch.tensor(40)
    x = ridge.features(t)
    expected_own = np.stack([signed[41 - h:41, 1:].sum(0) for h in evaluation.HORIZONS], axis=1)
    np.testing.assert_allclose(x[:, :4], expected_own)
    y = torch.tensor([.2, .5, float("nan"), -.3], dtype=torch.float64)
    mask = torch.tensor([True, True, False, True])
    assert torch.equal(ridge.step(t, y, mask, torch.tensor(True)), torch.zeros((1, 4), dtype=torch.float64))
    ridge.fit()
    expected = torch.linalg.solve(x[mask].T @ x[mask] + torch.eye(13, dtype=torch.float64), x[mask].T @ y[mask])
    torch.testing.assert_close(ridge.coefficient, expected)
    coefficient, gram, rhs = ridge.coefficient.clone(), ridge.gram.clone(), ridge.rhs.clone()
    a = ridge.step(t, y, mask, torch.tensor(False))
    b = ridge.step(t, y * 100, mask, torch.tensor(False))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a, (x @ expected)[None])
    for actual, original in ((ridge.coefficient, coefficient), (ridge.gram, gram), (ridge.rhs, rhs)):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def prefix_raw(configs, bars=20):
    count = len(configs) + len(evaluation.REFERENCE_NAMES)
    raw = np.zeros((bars, count, len(evaluation.METRICS)), dtype=np.float64)
    raw[:, :, 0] = np.arange(count)[None] + 1
    raw[:, :, 3:5] = 4
    raw[:, :, 5] = np.arange(count)[::-1][None] + 1
    raw[:, :, 8:10] = 4
    return raw


def test_locks_only_use_prefix_normalized_error_and_exclude_failed_candidates(tmp_path):
    configs = configurations()
    raw = prefix_raw(configs)
    raw[0, 0, 0] = np.inf
    health = np.ones(len(configs), dtype=bool)
    health[1] = False
    locks = evaluation.select_locks(raw, configs, 10, 20, 30, 30, health)
    assert locks[GROUPS[0]]["column"] == 2
    assert locks[GROUPS[0]]["config"] == asdict(configs[2])
    changed = raw.copy()
    changed[:, :, 5] *= -1e12
    assert evaluation.select_locks(changed, configs, 10, 20, 30, 30, health) == locks
    with pytest.raises(ValueError, match="prefix"):
        evaluation.select_locks(np.concatenate((raw, raw[:1])), configs, 10, 20, 30, 31)
    result = {"stage": "development", "view": "real", "data_identity": {"cut": 30, "stream_start": 10},
              "consumed_until_exclusive": 30, "source_sha256": {}, "prefix_health": health.tolist(), "locks": locks}
    evaluation.persist_prefix_locks(tmp_path, raw, result)
    with pytest.raises(FileExistsError):
        evaluation.persist_prefix_locks(tmp_path, raw, result)
    result["consumed_until_exclusive"] = 31
    with pytest.raises(ValueError, match="suffix"):
        evaluation.persist_prefix_locks(tmp_path, raw, result)


def test_reference_proof_authentication_cannot_select_from_suffix(tmp_path):
    configs = configurations()
    raw = prefix_raw(configs, 25)
    identity = {"stream_start": 10, "selection_start": 20, "cut": 30}
    result = {"version": 5, "stage": "development", "view": "real", "status": "completed",
              "configs": [asdict(c) for c in configs], "data_identity": identity, "source_sha256": {"model": "fixed"},
              "prefix_health": [True] * len(configs), "consumed_until_exclusive": 30,
              "locks": evaluation.select_locks(raw[:20], configs, 10, 20, 30, 30)}
    evaluation.persist_prefix_locks(tmp_path, raw[:20], result)
    result["consumed_until_exclusive"] = 35
    result["artifacts"] = {"raw_metrics": "raw_metrics.npz"}

    def save():
        np.savez(tmp_path / "raw_metrics.npz", bar_index=np.arange(10, 35),
                 **{key: raw[:, :, i] for i, key in enumerate(evaluation.METRICS)})
        result["artifact_sha256"] = {"raw_metrics": evaluation.frozen.file_sha256(tmp_path / "raw_metrics.npz")}
        evaluation.frozen.save_json(tmp_path / "results.json", result)

    save()
    original, _ = evaluation.load_reference(tmp_path / "results.json", result["source_sha256"], configs)
    raw[20:, :, 0] = np.arange(raw.shape[1])[::-1] * 1e8
    save()
    changed = evaluation.load_reference(tmp_path / "results.json", result["source_sha256"], configs)[0]
    assert original == changed
    raw[0, 0, 0] += 1
    save()
    with pytest.raises(ValueError, match="provenance"):
        evaluation.load_reference(tmp_path / "results.json", result["source_sha256"], configs)


def test_confirmation_requires_external_locks_and_authenticated_historical_boundary():
    with pytest.raises(ValueError, match="reference-result"):
        evaluation.validate_args(evaluation.Args(output_dir="x", stage="confirmation", rank_offset=400))
    with pytest.raises(ValueError, match="reference-result"):
        evaluation.validate_args(evaluation.Args(output_dir="x", view="rademacher"))
    with pytest.raises(ValueError, match="historical-result"):
        evaluation.validate_args(evaluation.Args(output_dir="x", stage="confirmation", rank_offset=400, reference_result="locked"))
    with pytest.raises(ValueError, match="historical"):
        evaluation.validate_args(evaluation.Args(output_dir="x", stage="confirmation", rank_offset=400,
            reference_result="locked", historical_result="historical", min_coverage=.8))
    root = evaluation.Path(__file__).resolve().parents[1]
    historical = json.loads((root / "benchmarks/plasticity/panel_frontier_v3_evidence.json").read_text())
    identity = copy.deepcopy(historical["data_identity"])
    identity["preprocessing"]["target"] = "ret"
    identity.pop("mu", None)
    assert evaluation.memory_eval.authenticate_confirmation(identity, historical) == 90145
    identity["cache_sha256"] = "different"
    with pytest.raises(ValueError, match="cache_sha256"):
        evaluation.memory_eval.authenticate_confirmation(identity, historical)


def test_durable_checkpoint_bounds_exclude_unwritten_prediction_pages(tmp_path):
    journal = np.lib.format.open_memmap(tmp_path / "metrics.npy", mode="w+", dtype=np.float64, shape=(5, 2, 10))
    predictions = np.lib.format.open_memmap(tmp_path / "predictions.npy", mode="w+", dtype=np.float64, shape=(5, 2, 4))
    first = np.arange(60, dtype=np.float64).reshape(3, 2, 10)
    journal[:3] = first
    predictions[:2] = .25
    result = {"status": "running", "checkpoint_raw_bounds": [10, 13], "prediction_valid_until_exclusive": 12}
    evaluation.persist_progress(tmp_path, result, journal, predictions)
    np.testing.assert_array_equal(np.load(tmp_path / "metrics.npy", mmap_mode="r")[:3], first)
    np.testing.assert_array_equal(np.load(tmp_path / "predictions.npy", mmap_mode="r")[:2], np.full((2, 2, 4), .25))
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["checkpoint_raw_bounds"] == [10, 13]
    assert saved["prediction_valid_until_exclusive"] == 12
    with pytest.raises(KeyboardInterrupt, match="received signal"):
        evaluation.interrupt_run(15, None)
    result["status"] = "interrupted"
    evaluation.persist_progress(tmp_path, result, journal, predictions)
    assert json.loads((tmp_path / "results.json").read_text())["status"] == "interrupted"


@cuda
@torch.no_grad()
def test_graph_capture_restores_all_state_and_current_forecasts_ignore_current_labels(monkeypatch):
    close = close_fixture()
    args = evaluation.Args()
    series = evaluation.return_series(args, close)
    z, vs, cs, acs, valid = panel_hd.series(args, close, np.ones_like(close))
    bank = panel_hd.Bank.__new__(panel_hd.Bank)
    bank.T, bank.N, bank.L, bank.F, bank.cut = len(close), 4, 32, 257, int(.6 * len(close))
    bank.dev = torch.device("cuda")
    bank.zt, bank.vst, bank.cst, bank.acst = [torch.tensor(a, device=bank.dev) for a in (z, vs, cs, acs)]
    bank.mu = torch.tensor(0., device=bank.dev)
    bank.ones = torch.ones((bank.N, 1), device=bank.dev)
    bank.lag_idx = torch.arange(1, 33, device=bank.dev)
    state = CausalState(bank, torch.tensor(valid, device=bank.dev))
    for t in range(33):
        state.observe(torch.tensor(t, device=bank.dev))
    configs = tuple(Config(g, .001, aux_weight=1. if "aux" in g else 0.) for g in GROUPS)
    model = Learner(bank.F, 128, configs, bank.dev, num_samples=bank.N, bins=5)
    runner = evaluation.Runner(bank, model, series, state, 32, 33, bank.T - 1)
    initial = [t.clone() for t in runner.mutable]
    graphs = runner.capture()
    for actual, expected in zip(runner.mutable, initial):
        assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
    graphs[1].replay()
    torch.cuda.synchronize()
    prediction = runner.prediction_ring[0].clone()
    assert torch.equal(runner.raw[0, :, 4], torch.full_like(runner.raw[0, :, 4], 4.))
    for actual, expected in zip(runner.mutable, initial):
        actual.copy_(expected)
    runner.target[33] *= -10
    runner.raw_target[33] *= -10
    runner.vol_target[33] *= 3
    graphs[1].replay()
    torch.cuda.synchronize()
    assert torch.equal(runner.prediction_ring[0], prediction)
    before_failure = [t.clone() for t in runner.mutable]

    def fail_capture(*args, **kwargs):
        raise RuntimeError("injected graph failure")

    monkeypatch.setattr(torch.cuda, "graph", fail_capture)
    with pytest.raises(RuntimeError, match="injected graph failure"):
        runner.capture()
    for actual, expected in zip(runner.mutable, before_failure):
        assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))

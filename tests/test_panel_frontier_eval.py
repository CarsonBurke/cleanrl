"""Host protocol contracts and queued CUDA graph contracts; no reduced learning runs."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cleanrl.plasticity import panel_frontier_eval_v3 as panel
from cleanrl.plasticity.panel_frontier_model_v3 import Config, FAMILIES, Learner
from cleanrl.shared import runtime


def grid():
    return tuple(Config(family, lr) for family in FAMILIES for lr in (1e-4, 1e-3, 1e-2))


def raw_metrics(bars=20):
    # Two unit targets per bar; predictions .2, .5, .8 for each family's LRs.
    predictions = np.tile([0.2, 0.5, 0.8], len(FAMILIES))
    row = np.stack((2 * (predictions - 1) ** 2, 2 * predictions ** 2,
                    2 * predictions, np.full(len(predictions), 2.0), np.full(len(predictions), 2.0)), axis=1)
    return np.repeat(row[None], bars, axis=0)


def test_confirmation_rejects_wrong_rank_grid_and_reference_overwrite(tmp_path):
    args = panel.Args(output_dir="unused")
    args.rank_offset = 200
    with pytest.raises(ValueError, match="rank-offset 400"):
        panel.validate_args(args)
    args.rank_offset = 400
    args.lrs = (1e-4, 1e-3)
    with pytest.raises(ValueError, match="preregistered"):
        panel.validate_args(args)
    args.lrs = panel.Args().lrs
    for reference in args.reference_caches:
        args.cache = reference
        with pytest.raises(ValueError, match="reference cache"):
            panel.validate_args(args)
    # Hard-linked aliases must also not overwrite either original cache.
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"original")
    alias = tmp_path / "alias.npz"
    alias.hardlink_to(reference)
    args.reference_caches = (str(reference), str(tmp_path / "second.npz"))
    args.cache = str(alias)
    with pytest.raises(ValueError, match="reference cache"):
        panel.validate_args(args)


def make_cohorts(tmp_path):
    paths = [tmp_path / f"rank{rank}.npz" for rank in (0, 200, 400)]
    symbols = [[f"R{rank}S{i}" for i in range(200)] for rank in (0, 200, 400)]
    for path, rank, names in zip(paths, (0, 200, 400), symbols):
        np.savez(path, symbols=names, n_stocks=200, rank_offset=rank)
    args = panel.Args(output_dir="unused", cache=str(paths[2]),
                      reference_caches=tuple(str(path) for path in paths[:2]))
    return args, paths, symbols


def test_cohort_identity_authenticates_both_actual_symbol_sets(tmp_path):
    args, paths, symbols = make_cohorts(tmp_path)
    bank = SimpleNamespace(T=100, N=200, F=257, cut=60, L=32, mu=torch.tensor(1.0))
    identity = panel.data_identity(args, bank)
    cohort = identity["cohort"]
    panel.require_disjoint(cohort)
    assert cohort["symbols"] == symbols[2]
    for row, path, names in zip(cohort["references"], paths, symbols):
        assert row["symbols"] == names
        assert row["cache_sha256"] == panel.file_sha256(path)
        assert row["overlap_symbols"] == []
    assert identity["configs"] == [asdict(Config(f, lr)) for f in FAMILIES for lr in args.lrs]


@pytest.mark.parametrize("reference_index", [0, 1])
def test_cohort_rejects_actual_overlap_despite_rank_metadata(tmp_path, reference_index):
    args, paths, symbols = make_cohorts(tmp_path)
    old_hash = panel.file_sha256(paths[2])
    symbols[2][0] = symbols[reference_index][0]
    np.savez(paths[2], symbols=symbols[2], n_stocks=200, rank_offset=400)
    cohort = panel.cohort_provenance(args)
    assert cohort["references"][reference_index]["overlap_symbols"] == [symbols[2][0]]
    assert panel.file_sha256(paths[2]) != old_hash
    with pytest.raises(ValueError, match="zero stock-symbol overlap"):
        panel.require_disjoint(cohort)


def test_cohort_fails_closed_on_missing_reference_and_duplicate_symbols(tmp_path):
    args, paths, symbols = make_cohorts(tmp_path)
    paths[0].unlink()
    with pytest.raises(FileNotFoundError):
        panel.cohort_provenance(args)
    symbols[2][0] = symbols[2][1]
    np.savez(paths[2], symbols=symbols[2], n_stocks=200, rank_offset=400)
    with pytest.raises(ValueError, match="200 unique stock symbols"):
        panel.cohort_provenance(args)


def test_primary_pairing_includes_loss_and_capacity_controls():
    raw = raw_metrics()
    configs = grid()
    primary = next(i for i, config in enumerate(configs) if config.family == "categorical_ce_js")
    raw[4:12, primary, 0] = 0.01
    raw[12:, primary, 0] = 0.02
    locks = panel.select_locks(raw[:12], configs, 2, 6, 14, 14)
    summary = panel.phase_summary(raw, 2, 22, 6, 14, 22, [f"{c.family}_{c.lr:g}" for c in configs])
    paired = panel.paired_comparisons(summary, locks)["forward"]
    assert set(paired["selected"]) == set(FAMILIES)
    assert set(paired["primary_minus_baseline"]) == set(FAMILIES) - {"categorical_ce_js"}
    for family in ("scalar_js", "scalar_budget_js", "categorical_mse_js"):
        assert paired["primary_minus_baseline"][family]["mse"] == pytest.approx(
            0.01 - paired["selected"][family]["mse"])


def test_prefix_locks_are_independent_and_cannot_consult_forward_rows():
    configs = grid()
    raw = raw_metrics()
    # Validation winner differs per family. Training and suffix prefer a
    # different LR, ensuring neither can enter the selection reduction.
    expected = [0, 1, 2, 1]
    raw[:4, :, 0] = np.tile([1e9, 0, 0], len(FAMILIES))
    raw[12:, :, 0] = np.tile([0, 0, 1e9], len(FAMILIES))
    for family, offset in enumerate(expected):
        raw[4:12, family * 3:family * 3 + 3, 0] = 9
        raw[4:12, family * 3 + offset, 0] = 1
    locks = panel.select_locks(raw[:12], configs, 2, 6, 14, 14)
    for family, offset in zip(FAMILIES, expected):
        assert locks[family]["lr"] == configs[3 * FAMILIES.index(family) + offset].lr
        assert locks[family]["endpoint_winner"] == (offset in (0, 2))
        assert locks[family]["candidates"][offset]["count"] == 16
    with pytest.raises(ValueError, match="exact prefix"):
        panel.select_locks(raw, configs, 2, 6, 14, 14)
    with pytest.raises(ValueError, match="exact prefix"):
        panel.select_locks(raw[:12], configs, 2, 6, 14, 15)


def test_selection_uses_sample_weighted_mse_and_first_lr_ties():
    raw = raw_metrics(12)
    # Equal candidate means per bar would prefer col0; true pooled valid-sample
    # MSE prefers col1. This also defends fixed-N graphs with variable masks.
    raw[4:, :, 4] = np.array([1, 100, 1, 100, 1, 100, 1, 100])[:, None]
    raw[4:, 0, 0] = [0, 200, 0, 200, 0, 200, 0, 200]
    raw[4:, 1, 0] = [3, 100, 3, 100, 3, 100, 3, 100]
    raw[4:, 2, 0] = raw[4:, 1, 0]
    locks = panel.select_locks(raw, grid(), 2, 6, 14, 14)
    assert locks["scalar_js"]["column"] == 1
    assert locks["scalar_js"]["candidates"][1]["mse"] == pytest.approx(412 / 404)


def test_nonfinite_candidates_remain_raw_and_cannot_win():
    raw = raw_metrics(12)
    raw[2, 2, 0] = np.nan  # Failure before validation still disqualifies it.
    raw[7, 1, 0] = np.inf
    original = raw.copy()
    locks = panel.select_locks(raw, grid(), 2, 6, 14, 14)
    assert locks["scalar_js"]["column"] == 0
    assert not locks["scalar_js"]["candidates"][2]["eligible"]
    assert locks["scalar_js"]["candidates"][1]["status"] == "nonfinite"
    assert panel.finite_json(locks)["scalar_js"]["candidates"][1]["mse"] is None
    np.testing.assert_array_equal(raw, original)
    state_finite = np.ones(len(grid()), dtype=bool)
    state_finite[0] = False
    locks = panel.select_locks(raw, grid(), 2, 6, 14, 14, state_finite)
    assert locks["scalar_js"]["column"] is None


def test_time_blocks_count_only_consumed_masked_samples():
    raw = raw_metrics(17)
    raw[:, :, 4] = (np.arange(17) % 3)[:, None]
    names = [f"{c.family}_{c.lr:g}" for c in grid()]
    summary = panel.phase_summary(raw, 2, 19, 6, 14, 30, names)
    assert summary["forward"]["observed_range"] == [14, 19]
    assert summary["forward"]["sample_count"] == int(raw[12:, 0, 4].sum())
    blocks = [summary[f"forward_block_{i}"] for i in range(1, 9)]
    assert sum(block["bars"] for block in blocks) == 5
    assert sum(block["sample_count"] for block in blocks) == summary["forward"]["sample_count"]
    assert blocks[2]["observed_range"] == [18, 19]
    assert not blocks[2]["complete"]
    assert blocks[3]["bars"] == 0
    assert blocks[3]["sample_count"] == 0
    with pytest.raises(ValueError, match="consumed rows"):
        panel.phase_summary(raw, 2, 18, 6, 14, 30, names)


def test_null_inherits_real_prefix_locks_and_rejects_mismatched_provenance(tmp_path):
    raw = raw_metrics(20)
    configs = grid()
    identity = {"stream_start": 2, "selection_start": 6, "cut": 14,
                "cache_sha256": "fresh-panel", "preprocessing": {"rank_offset": 400},
                "configs": [asdict(config) for config in configs]}
    locks = panel.select_locks(raw[:12], configs, 2, 6, 14, 14)
    raw[12:, :, 0] = np.tile([0, 1e9, 1e9], len(FAMILIES))
    artifact = tmp_path / "raw_metrics.npz"
    np.savez(artifact, bar_index=np.arange(2, 22),
             **{key: raw[:, :, i] for i, key in enumerate(panel.METRICS)})
    source = {"view": "real", "status": "pruned", "data_identity": identity,
              "source_sha256": {"model": "frozen"}, "consumed_until_exclusive": 22,
              "locks": panel.finite_json(locks), "artifacts": {"raw_metrics": artifact.name},
              "artifact_sha256": {"raw_metrics": panel.file_sha256(artifact)}}
    source["prefix_state_finite"] = [True] * len(configs)
    panel.persist_prefix_locks(tmp_path, raw[:12], 14, source)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(source))
    inherited, provenance = panel.load_real_locks(path, identity, {"model": "frozen"}, configs)
    assert inherited == locks
    assert provenance["results_sha256"] == panel.file_sha256(path)
    assert set(inherited) == set(FAMILIES)
    with pytest.raises(ValueError, match="matching real"):
        panel.load_real_locks(path, identity, {"model": "changed"}, configs)
    with pytest.raises(ValueError, match="matching real"):
        panel.load_real_locks(path, {**identity, "cache_sha256": "old-panel"}, {"model": "frozen"}, configs)
    with pytest.raises(ValueError, match="matching real"):
        panel.load_real_locks(path, {**identity, "mu": 2}, {"model": "frozen"}, configs)
    original_raw = artifact.read_bytes()
    artifact.write_bytes(original_raw + b"tampered")
    with pytest.raises(ValueError, match="raw metrics digest"):
        panel.load_real_locks(path, identity, {"model": "frozen"}, configs)
    artifact.write_bytes(original_raw)
    proof_path = tmp_path / "prefix_locks.json"
    original_proof = proof_path.read_text()
    proof_path.write_text(original_proof + " ")
    with pytest.raises(ValueError, match="prefix lock artifact digest"):
        panel.load_real_locks(path, identity, {"model": "frozen"}, configs)
    proof_path.write_text(original_proof)
    source["locks"]["scalar_js"]["column"] = 0
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="provenance"):
        panel.load_real_locks(path, identity, {"model": "frozen"}, configs)


def test_artifacts_preserve_failed_values_and_consumed_checkpoint(tmp_path):
    configs = grid()
    state = (torch.tensor(12, dtype=torch.int64), torch.tensor([float("nan")]))
    model = SimpleNamespace(configs=configs, output_names=[f"{c.family}_{c.lr:g}" for c in configs],
                            state_tensors=lambda: state)
    runner = SimpleNamespace(start=2, stop=30, model=model, index=torch.tensor(14))
    raw = raw_metrics(12)
    raw[0, 0, 0] = np.nan
    raw[1, 1, 0] = np.inf
    result = {"locks": {}, "data_identity": {"selection_start": 6, "cut": 14}, "status": "failed"}
    panel.save_artifacts(tmp_path, runner, raw, 14, None, result)
    with np.load(tmp_path / "raw_metrics.npz") as stored:
        assert np.isnan(stored["residual_sse"][0, 0])
        assert np.isposinf(stored["residual_sse"][1, 1])
        np.testing.assert_array_equal(stored["bar_index"], np.arange(2, 14))
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert checkpoint["consumed_until_exclusive"] == 14
    assert torch.isnan(checkpoint["model_state"][1]).all()
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["phase_metrics"]["all_consumed"]["metrics"][0]["status"] == "nonfinite"
    assert saved["phase_metrics"]["forward"]["bars"] == 0


def test_failed_checkpoint_recovers_all_group_clocks_not_stale_host_rows():
    raw = torch.tensor(raw_metrics(20))
    clocks = tuple((torch.tensor(13), torch.tensor(11)) for _ in FAMILIES)
    runner = SimpleNamespace(start=2, stop=22, index=torch.tensor(15),
                            model=SimpleNamespace(clocks=clocks), raw=raw)
    consumed, recovered = panel.recover_consumed(runner)
    assert consumed == 15
    np.testing.assert_array_equal(recovered, raw[:13].numpy())
    for pair in clocks:
        for clock in pair:
            original = clock.item()
            clock.sub_(1)
            with pytest.raises(RuntimeError, match="coherent checkpoint"):
                panel.recover_consumed(runner)
            clock.fill_(original)


def test_prefix_proof_rejects_late_rows_and_cannot_be_rewritten(tmp_path):
    raw = raw_metrics(12)
    configs = grid()
    result = {"view": "real", "data_identity": {"stream_start": 2, "cut": 14},
              "source_sha256": {"model": "frozen"}, "prefix_state_finite": [True] * len(configs),
              "locks": panel.select_locks(raw, configs, 2, 6, 14, 14)}
    with pytest.raises(ValueError, match="exact real prefix"):
        panel.persist_prefix_locks(tmp_path, raw, 15, result)
    panel.persist_prefix_locks(tmp_path, raw, 14, result)
    with pytest.raises(ValueError, match="immutable"):
        panel.persist_prefix_locks(tmp_path, raw, 14, result)


class TinyBank:
    """Only for graph causality/restore contracts, never a learning experiment."""

    def __init__(self):
        self.L, self.T, self.N, self.F = 1, 40, 4, 3
        self.dev = torch.device("cuda")
        self.x = torch.tensor([[0.2, -0.4, 1], [0.3, 0.8, 1],
                               [-0.2, 0.4, 1], [-0.7, 0.3, 1]], device=self.dev)

    def feats(self, index):
        return self.x + index.float().reshape(1, 1) * 0.001


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; queue with mlq")
@torch.no_grad()
def test_graph_labels_change_future_not_current_forecasts_and_capture_restores(monkeypatch):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    bank = TinyBank()
    configs = tuple(Config(family, 0.01) for family in FAMILIES)
    target = torch.full((bank.T, bank.N), -1.0, device=bank.dev)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, -1] = False
    target[:, -1] = float("nan")  # Must not enter squared-error sums or gradients.
    # Frozen Bank sanitizes features before training; only masked labels may be NaN.
    model = Learner(bank.F, 8, 1.0, configs, bank.dev, bins=5, num_samples=bank.N)
    # Parameter budgets are scientific controls, not dense-width assumptions.
    report = panel.capacity_report(model)
    counts = dict(zip(FAMILIES, report["parameter_count_effective_by_config"]))
    assert counts == {"scalar_js": 113, "scalar_budget_js": 161,
                      "categorical_mse_js": 149, "categorical_ce_js": 149}
    assert dict(zip(FAMILIES, report["widths_by_config"])) == {
        "scalar_js": 8, "scalar_budget_js": 10, "categorical_mse_js": 8, "categorical_ce_js": 8}
    assert report["parameter_count_allocated"] == sum(t.numel() for t in model.parameters)
    runner = panel.Runner(bank, model, target, valid)
    initial = [tensor.clone() for tensor in runner.mutable]
    graphs = runner.capture()
    for actual, expected in zip(runner.mutable, initial):
        assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
    graphs[1].replay()
    torch.cuda.synchronize()
    first = model.prediction.clone()
    p = first[:, :3].double()
    expected = torch.stack(((p + 1).square().sum(1), p.square().sum(1), -p.sum(1),
                            torch.full_like(p[:, 0], 3), torch.full_like(p[:, 0], 3)), dim=1)
    torch.testing.assert_close(runner.raw[0], expected, rtol=0, atol=0)
    assert torch.equal(runner.raw[0, :, 4], torch.full((len(FAMILIES),), 3.0, dtype=torch.float64, device=bank.dev))
    # Evidence gates intentionally begin closed. Check downstream learning after
    # repeated observations, not an immediate update that the algorithm forbids.
    graphs[8].replay()
    torch.cuda.synchronize()
    next_original = model.prediction.clone()
    for actual, expected in zip(runner.mutable, initial):
        actual.copy_(expected)
    target[runner.start, :3] = 4.0
    graphs[1].replay()
    torch.cuda.synchronize()
    assert torch.equal(model.prediction, first)
    graphs[8].replay()
    torch.cuda.synchronize()
    assert (model.prediction[:, :3] != next_original[:, :3]).any(1).all()
    assert runner.index.item() == runner.start + 9
    assert model.steps.item() == 9
    panel.check_clocks(runner, runner.start + 9)

    # Failure after compile warmup must restore the already consumed state too.
    before_failure = [tensor.clone() for tensor in runner.mutable]

    def reject_capture(*args, **kwargs):
        raise RuntimeError("injected CUDA capture failure")

    monkeypatch.setattr(torch.cuda, "graph", reject_capture)
    with pytest.raises(RuntimeError, match="injected CUDA capture failure"):
        runner.capture()
    for actual, expected in zip(runner.mutable, before_failure):
        assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))

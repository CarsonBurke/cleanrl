"""Host protocol contracts and queued CUDA graph contracts; no reduced learning runs."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cleanrl.plasticity import panel_specialist_eval_v2 as panel
from cleanrl.plasticity.panel_specialist_model_v2 import Config, FAMILIES, Learner
from cleanrl.shared import runtime


def grid():
    return tuple(Config(family, lr) for family in FAMILIES for lr in (1e-4, 1e-3, 1e-2))


def raw_metrics(bars=20):
    # Two unit targets per bar; predictions .2, .5, .8 for each family's LRs.
    predictions = np.tile([0.2, 0.5, 0.8], len(FAMILIES))
    row = np.stack((2 * (predictions - 1) ** 2, 2 * predictions ** 2,
                    2 * predictions, np.full(len(predictions), 2.0), np.full(len(predictions), 2.0)), axis=1)
    return np.repeat(row[None], bars, axis=0)


def test_confirmation_protocol_and_secondary_panel_guard():
    args = panel.Args(output_dir="unused")
    panel.validate_args(args)
    assert (args.n_stocks, args.rank_offset, args.width, args.bins, args.seed) == (200, 200, 256, 33, 1)
    assert args.cache == "/tmp/panel_distributional_rank200.npz"
    assert len(FAMILIES) * len(args.lrs) == 42
    assert args.lrs == (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    args.rank_offset = 0
    with pytest.raises(ValueError, match="secondary"):
        panel.validate_args(args)
    args.dataset_role = "secondary"
    panel.validate_args(args)
    args.cache = args.reference_cache
    with pytest.raises(ValueError, match="reference cache"):
        panel.validate_args(args)


def test_cohort_identity_fingerprints_membership_and_all_seven_families(tmp_path):
    cache, reference = tmp_path / "fresh.npz", tmp_path / "original.npz"
    original = [f"OLD{i}" for i in range(200)]
    current = [f"NEW{i}" for i in range(200)]
    np.savez(reference, symbols=original, n_stocks=200, rank_offset=0)
    np.savez(cache, symbols=current, n_stocks=200, rank_offset=200)
    args = panel.Args(output_dir="unused", cache=str(cache), reference_cache=str(reference))
    bank = SimpleNamespace(T=100, N=200, F=257, cut=60, L=32, mu=torch.tensor(1.0))
    identity = panel.data_identity(args, bank)
    cohort = identity["cohort"]
    assert cohort["symbols"] == current
    assert cohort["reference_symbols"] == original
    assert cohort["disjoint_from_reference"] is True
    assert cohort["reference_cache_sha256"] == panel.file_sha256(reference)
    assert identity["configs"] == [asdict(Config(f, lr)) for f in FAMILIES for lr in args.lrs]
    current[0] = original[0]  # Offset alone does not establish disjointness.
    np.savez(cache, symbols=current, n_stocks=200, rank_offset=200)
    changed = panel.data_identity(args, bank)
    assert changed["cache_sha256"] != identity["cache_sha256"]
    assert changed["cohort"]["symbols_sha256"] != cohort["symbols_sha256"]
    assert changed["cohort"]["disjoint_from_reference"] is False
    assert changed["cohort"]["overlap_symbols"] == [original[0]]
    args.reference_cache = str(tmp_path / "missing.npz")
    assert panel.cohort_provenance(args)["disjoint_from_reference"] is False


def test_primary_pairing_includes_uniform_and_learned_constant_controls():
    raw = raw_metrics()
    configs = grid()
    primary = next(i for i, config in enumerate(configs) if config.family == "state_moe_js")
    raw[4:12, primary, 0] = 0.01
    raw[12:, primary, 0] = 0.02
    locks = panel.select_locks(raw[:12], configs, 2, 6, 14, 14)
    summary = panel.phase_summary(raw, 2, 22, 6, 14, 22, [f"{c.family}_{c.lr:g}" for c in configs])
    paired = panel.paired_comparisons(summary, locks)["forward"]
    assert set(paired["selected"]) == set(FAMILIES)
    assert set(paired["primary_minus_baseline"]) == set(FAMILIES) - {"state_moe_js"}
    for family in ("scalar_js", "uniform_moe_js", "constant_moe_js", "state_moe_adam"):
        assert paired["primary_minus_baseline"][family]["mse"] == pytest.approx(
            0.01 - paired["selected"][family]["mse"])


def test_prefix_locks_are_independent_and_cannot_consult_forward_rows():
    configs = grid()
    raw = raw_metrics()
    # Validation winner differs per family. Training and suffix prefer a
    # different LR, ensuring neither can enter the selection reduction.
    expected = [0, 1, 2, 1, 0, 2, 1]
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
    assert locks["scalar_adam"]["column"] == 1
    assert locks["scalar_adam"]["candidates"][1]["mse"] == pytest.approx(412 / 404)


def test_nonfinite_candidates_remain_raw_and_cannot_win():
    raw = raw_metrics(12)
    raw[2, 2, 0] = np.nan  # Failure before validation still disqualifies it.
    raw[7, 1, 0] = np.inf
    original = raw.copy()
    locks = panel.select_locks(raw, grid(), 2, 6, 14, 14)
    assert locks["scalar_adam"]["column"] == 0
    assert not locks["scalar_adam"]["candidates"][2]["eligible"]
    assert locks["scalar_adam"]["candidates"][1]["status"] == "nonfinite"
    assert panel.finite_json(locks)["scalar_adam"]["candidates"][1]["mse"] is None
    np.testing.assert_array_equal(raw, original)
    state_finite = np.ones(len(grid()), dtype=bool)
    state_finite[0] = False
    locks = panel.select_locks(raw, grid(), 2, 6, 14, 14, state_finite)
    assert locks["scalar_adam"]["column"] is None


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
                "cache_sha256": "fresh-panel", "preprocessing": {"rank_offset": 200},
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
    source["locks"]["scalar_adam"]["column"] = 0
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


def test_failed_checkpoint_recovers_device_boundary_not_stale_host_rows():
    raw = torch.tensor(raw_metrics(20))
    runner = SimpleNamespace(start=2, stop=22, index=torch.tensor(15),
                            model=SimpleNamespace(steps=torch.tensor(13), moe_steps=torch.tensor(13),
                                                  adam_steps=torch.tensor(13), moe_adam_steps=torch.tensor(13)), raw=raw)
    consumed, recovered = panel.recover_consumed(runner)
    assert consumed == 15
    np.testing.assert_array_equal(recovered, raw[:13].numpy())
    for name in ("steps", "moe_steps", "adam_steps", "moe_adam_steps"):
        clock = getattr(runner.model, name)
        clock.fill_(12)
        with pytest.raises(RuntimeError, match="coherent checkpoint"):
            panel.recover_consumed(runner)
        clock.fill_(13)


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
    bank.x[-1] = float("nan")
    model = Learner(bank.F, 8, 1.0, configs, bank.dev, bins=5, num_samples=bank.N)
    # Parameter budgets are scientific controls, not dense-width assumptions.
    counts = dict(zip(FAMILIES, model.effective_parameter_counts))
    assert counts == {
        "scalar_adam": 113, "scalar_js": 113, "categorical_ce_js": 149,
        "uniform_moe_js": 68, "constant_moe_js": 72,
        "state_moe_adam": 84, "state_moe_js": 84,
    }
    assert model.allocated_parameter_count == sum(t.numel() for t in model.parameters)
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
    assert model.moe_steps.item() == 9
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

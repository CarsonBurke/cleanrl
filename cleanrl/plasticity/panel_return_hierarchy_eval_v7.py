"""Matched structural return priors; full causal streams and authenticated locks.

Rank400 is explicitly reused. Rank600 is an additional stock cohort, not pristine
market dates or an iid replication. No profitability or market-model claim.
"""

import hashlib
import json
import math
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import panel_hd
from cleanrl.plasticity import panel_return_refinement_eval_v6 as reference
from cleanrl.plasticity.panel_predictive_memory_v4 import CausalState
from cleanrl.plasticity.panel_return_representation_v5 import Config as NeuralConfig, Learner as NeuralLearner
from cleanrl.plasticity.panel_return_hierarchy_v7 import Config, GROUPS, Learner, configurations
from cleanrl.plasticity.panel_stream import build_panel
from cleanrl.shared import runtime

frozen = reference.frozen
frontier = reference.frontier
METRICS = reference.METRICS
REFERENCE_NAMES = ("memory_adam", "zero_return", "ridge_signed_1")
PRIMARY = "hierarchical_ridge"
SIGN_NAMESPACE = "panel_return_hierarchy_v7_signed_label_rademacher_seed1"
SOURCE_FILES = (*reference.SOURCE_FILES, "cleanrl/plasticity/panel_return_hierarchy_v7.py",
                "cleanrl/plasticity/panel_return_hierarchy_eval_v7.py")
CACHE_PATHS = {0: "/tmp/panel_cache.npz", 400: "/tmp/panel_frontier_rank400.npz", 600: "/tmp/panel_frontier_rank600.npz"}


@dataclass
class Args(panel_hd.Args):
    output_dir: str = ""
    stage: Literal["development", "replay", "confirmation"] = "development"
    reference_result: str = ""
    neural_reference: str = "runs/panel_return_refinement_v6_development/results.json"
    cache: str = ""
    width: int = 128
    bins: int = 33
    seed: int = 1
    view: Literal["real", "rademacher"] = "real"
    log_every: int = 4096


def validate_args(a):
    if not a.output_dir.strip():
        raise ValueError("--output-dir required; no resume")
    if (a.target, a.n_stocks, a.lags, a.width, a.bins, a.seed) != ("ret", 200, 32, 128, 33, 1):
        raise ValueError("fixed dimensions, direct return target and seed1 required")
    if (a.min_coverage, a.vol_span, a.first_session_before) != (.9, 200, "2018-01-01"):
        raise ValueError("historical coverage, RMS and session preprocessing required")
    if a.rank_offset != {"development": 0, "replay": 400, "confirmation": 600}[a.stage]:
        raise ValueError("development rank0; explicit replay rank400; additional-cohort confirmation rank600")
    if bool(a.reference_result) != (a.stage != "development" or a.view == "rademacher"):
        raise ValueError("replay/confirmation/null require --reference-result; real development cannot inherit locks")
    if a.view != "real" and a.stage != "development":
        raise ValueError("null requires rank0 real development locks")
    if not a.neural_reference or a.log_every < 16:
        raise ValueError("neural reference and log-every>=16 required")
    if not a.cache:
        a.cache = CACHE_PATHS[a.rank_offset]
    for rank, path in CACHE_PATHS.items():
        if Path(a.cache).resolve() == Path(path).resolve() and rank != a.rank_offset:
            raise ValueError("refusing another cohort's reserved cache path")


def protect_cache(a):
    if Path(a.cache).exists():
        with np.load(a.cache, allow_pickle=False) as cache:
            if int(cache["rank_offset"]) != a.rank_offset or int(cache["n_stocks"]) != a.n_stocks:
                raise ValueError("refusing to overwrite another cohort cache")


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: frozen.file_sha256(root / name) for name in SOURCE_FILES}


def load_neural_reference(path):
    """Authenticate the actual v6 development prefix, never confirmation results."""
    configs = reference.configurations()
    locks, provenance = reference.load_reference(path, reference.source_hashes(), configs)
    source = json.loads(Path(path).read_text())
    if (not source.get("source_and_stream_unchanged")
            or source["consumed_until_exclusive"] != source["data_identity"]["stream_stop"]
            or source.get("source_sha256_after") != source["source_sha256"]
            or source.get("stream_sha256_before") != source.get("stream_sha256_after")):
        raise ValueError("neural development stream/source is incomplete or changed")
    lock = locks["memory_adam"]
    if lock["status"] != "locked" or lock["config"] is None:
        raise ValueError("frozen memory Adam has no eligible prefix lock")
    config = NeuralConfig(**lock["config"])
    if config.family != "memory_adam" or config.aux_weight != 0:
        raise ValueError("expected frozen direct-return memory Adam")
    return config, {**provenance, "config": asdict(config), "prefix_lock": lock}


return_series = reference.return_series
metric_row = reference.metric_row
reduce_metrics = reference.reduce_metrics
persist_progress = reference.persist_progress
interrupt_run = reference.interrupt_run
stream_identity = reference.stream_identity


def randomized_labels(normalized, raw):
    seed = int.from_bytes(hashlib.sha256(SIGN_NAMESPACE.encode()).digest()[:8], "little")
    signs = np.random.Generator(np.random.PCG64(seed)).integers(0, 2, size=normalized.shape, dtype=np.int8) * 2 - 1
    return normalized * signs, raw * signs, reference.array_identity(signs)


def data_identity(a, bank, close, symbols, ts, series):
    names = list(map(str, symbols))
    if len(names) != bank.N or len(set(names)) != bank.N or "SPY" in names:
        raise ValueError("expected unique stocks and separate market column")
    kept_ts = ts[series["keep"]]
    boundaries = {"stream_start": bank.L + 1, "selection_start": int(.4 * bank.T), "cut": bank.cut,
                  "stream_stop": bank.T - 1, "last_forecast": bank.T - 2}
    return {"cache_sha256": frozen.file_sha256(a.cache),
        "preprocessing": {k: getattr(a, k) for k in ("rank_offset", "n_stocks", "lags", "vol_span", "min_coverage", "first_session_before", "target")},
        "bars": bank.T, "stocks": bank.N, "features": bank.F, **boundaries,
        "width": a.width, "bins": a.bins, "seed": a.seed, "configs": [asdict(c) for c in configurations()],
        "symbols": names, "shared_market_symbol": "SPY",
        "calendar_boundaries": {name: {"bar": i, "timestamp": str(kept_ts[i])} for name, i in boundaries.items()},
        "retained_timestamp_sha256": hashlib.sha256(np.ascontiguousarray(kept_ts).tobytes()).hexdigest(),
        "return_arrays": {name: reference.array_identity(series[name]) for name in ("raw_return", "scale")}}


class Runner(frozen.Runner):
    """Reuse the frozen fullgraph capture/reset protocol, including all new state."""

    def __init__(self, bank, model, neural, series, state, ring_size, score_start, score_stop):
        self.bank, self.model, self.neural, self.state = bank, model, neural, state
        self.start, self.stop = bank.L + 1, bank.T - 1
        self.score_start, self.score_stop, self.ring_size = score_start, score_stop, ring_size
        self.index = torch.tensor(self.start, dtype=torch.int64, device=bank.dev)
        self.target = torch.tensor(series["normalized_target"], dtype=torch.float64, device=bank.dev)
        self.raw_target = torch.tensor(series["raw_target"], dtype=torch.float64, device=bank.dev)
        self.denominator = torch.tensor(series["denominator"], dtype=torch.float64, device=bank.dev)
        self.valid = torch.tensor(series["valid"], dtype=torch.bool, device=bank.dev)
        self.vol_target = torch.roll(bank.zt[:, 1:], -1, 0).square().clamp_max(25.)
        self.ridge = reference.RidgeReference(series["observed_signed"], series["observed_valid"], bank.dev)
        self.output_names = (*model.output_names, *REFERENCE_NAMES)
        self.raw = torch.zeros((self.stop - self.start, len(self.output_names), len(METRICS)), dtype=torch.float64, device=bank.dev)
        self.prediction_ring = torch.zeros((ring_size, len(self.output_names), bank.N), dtype=torch.float64, device=bank.dev)
        self.mutable = (*model.state_tensors(), *neural.state_tensors(), *state.state_tensors(),
                        *self.ridge.state_tensors(), self.index, self.raw, self.prediction_ring)
        self.compile_seconds = self.capture_seconds = 0.

    @torch.no_grad()
    def step(self):
        index = self.index.reshape(1)
        self.state.observe(self.index)
        y = self.target.index_select(0, index).squeeze(0)
        mask = self.valid.index_select(0, index).squeeze(0)
        x = self.ridge.features(self.index)
        learned = self.model.step(x, y, mask)
        neural = self.neural.step(self.state.frames(self.index), y.float(), self.vol_target.index_select(0, index).squeeze(0), mask)
        ridge = (x @ self.ridge.coefficient)[None]
        prediction = torch.cat((learned, neural.double(), torch.zeros_like(y[None]), ridge))
        visible = (self.index >= self.score_start) & (self.index < self.score_stop)
        row = metric_row(prediction, y, self.raw_target.index_select(0, index).squeeze(0),
                         self.denominator.index_select(0, index).squeeze(0), mask & visible)
        self.raw.index_copy_(0, (self.index - self.start).reshape(1), row[None])
        self.prediction_ring.index_copy_(0, ((self.index - self.start) % self.ring_size).reshape(1),
                                         torch.where(visible, prediction, 0.)[None])
        self.index.add_(1)

    @torch.no_grad()
    def fit_reference(self):
        # The pooled sufficient statistics already contain exactly the first60%.
        # Do not compute a second identical Gram/RHS per bar for offline ridge1.
        self.ridge.gram.copy_(self.model.gram)
        self.ridge.rhs.copy_(self.model.rhs)
        self.ridge.fit()


def select_locks(raw, configs, start, selection_start, cut, consumed, health=None):
    if consumed != cut or len(raw) != cut - start or not start <= selection_start < cut:
        raise ValueError("locks require exactly prefix rows, no suffix")
    if raw.shape[1:] != (len(configs) + len(REFERENCE_NAMES), len(METRICS)):
        raise ValueError("candidate/reference metric columns disagree")
    scores = frozen.reduce_metrics(raw[selection_start - start:, :, :5])
    alive = np.isfinite(raw[:, :, :5]).all(axis=(0, 2))
    if health is not None:
        if np.asarray(health).shape != (len(configs),):
            raise ValueError("prefix health dimensions disagree")
        alive[:len(configs)] &= np.asarray(health, dtype=bool)
    locks = {}
    for group in GROUPS:
        candidates = [{"column": i, **asdict(c), **scores[i],
                       "eligible": bool(alive[i] and scores[i]["count"] and math.isfinite(scores[i]["mse"]))}
                      for i, c in enumerate(configs) if c.family == group]
        eligible = [c for c in candidates if c["eligible"]]
        winner = min(eligible, key=lambda c: c["mse"]) if eligible else None
        locks[group] = {"column": winner["column"] if winner else None,
                        "config": asdict(configs[winner["column"]]) if winner else None,
                        "status": "locked" if winner else "failed_nonfinite_or_empty", "candidates": candidates,
                        "selection_start_inclusive": selection_start, "selection_stop_exclusive": cut, "locked_before_bar": cut}
    return locks


def persist_prefix_locks(root, raw, result):
    identity = result["data_identity"]
    if ((result["stage"], result["view"]) != ("development", "real")
            or len(raw) != identity["cut"] - identity["stream_start"] or result["consumed_until_exclusive"] != identity["cut"]):
        raise ValueError("prefix locks cannot use suffix, null or confirmation data")
    proof = {"version": 7, "source_sha256": result["source_sha256"], "data_identity": identity,
             "neural_reference": result["neural_reference"], "raw_prefix_sha256": frontier.raw_prefix_hash(raw),
             "health": result["prefix_health"], "locks": result["locks"], "locked_before_bar": identity["cut"]}
    path = root / "prefix_locks.json"
    with path.open("x") as stream:
        json.dump(frozen.finite_json(proof), stream, indent=2, allow_nan=False)
    result["prefix_lock_artifact"] = {"path": path.name, "sha256": frozen.file_sha256(path)}


def load_reference(path, hashes, configs, neural_reference):
    path = Path(path)
    source = json.loads(path.read_text())
    if (source.get("version"), source.get("stage"), source.get("view"), source.get("status")) != (7, "development", "real", "completed"):
        raise ValueError("reference must be completed real v7 development")
    if source.get("source_sha256") != hashes or source.get("configs") != [asdict(c) for c in configs]:
        raise ValueError("reference source hashes/configs differ")
    if source.get("neural_reference") != neural_reference:
        raise ValueError("reference frozen neural lock differs")
    identity = source["data_identity"]
    if (identity["preprocessing"]["rank_offset"] != 0 or source.get("consumed_until_exclusive") != identity["stream_stop"]
            or source.get("raw_metric_bounds") != [identity["stream_start"], identity["stream_stop"]]
            or not source.get("source_and_stream_unchanged")):
        raise ValueError("reference data provenance or complete export bounds differ")
    artifact = source["prefix_lock_artifact"]
    proof_path = path.parent / artifact["path"]
    if frozen.file_sha256(proof_path) != artifact["sha256"]:
        raise ValueError("prefix proof digest mismatch")
    proof = json.loads(proof_path.read_text())
    raw_path = path.parent / source["artifacts"]["raw_metrics"]
    if frozen.file_sha256(raw_path) != source["artifact_sha256"]["raw_metrics"]:
        raise ValueError("reference raw metric digest mismatch")
    count = identity["cut"] - identity["stream_start"]
    with np.load(raw_path, allow_pickle=False) as stored:
        if not np.array_equal(stored["bar_index"], np.arange(identity["stream_start"], identity["stream_stop"])):
            raise ValueError("reference full bar indices differ")
        raw = np.stack([stored[key][:count] for key in METRICS], axis=-1)
    if (proof.get("version") != 7 or proof.get("source_sha256") != hashes or proof.get("data_identity") != identity
            or proof.get("neural_reference") != neural_reference or proof.get("locked_before_bar") != identity["cut"]
            or proof.get("raw_prefix_sha256") != frontier.raw_prefix_hash(raw) or proof.get("locks") != source["locks"]
            or proof.get("health") != source["prefix_health"]):
        raise ValueError("reference prefix provenance mismatch")
    locks = select_locks(raw, configs, identity["stream_start"], identity["selection_start"], identity["cut"], identity["cut"], proof["health"])
    if frozen.finite_json(locks) != source["locks"]:
        raise ValueError("locks disagree with prefix-only normalized MSE selection")
    return locks, {"path": str(path.resolve()), "sha256": frozen.file_sha256(path), "prefix_sha256": artifact["sha256"],
                   "development_identity": identity, "development_consumed_until_exclusive": source["consumed_until_exclusive"]}


def selected_columns(locks, learned_count):
    selected = {name: row["column"] for name, row in locks.items() if row["column"] is not None}
    selected.update({name: learned_count + i for i, name in enumerate(REFERENCE_NAMES)})
    return selected


def summary(raw, start, low, high, locks, learned_count):
    selected = selected_columns(locks, learned_count)
    rows = reduce_metrics(raw[low - start:high - start])
    result = {"start_inclusive": low, "stop_exclusive": high, "all_candidates": rows,
              "selected": {g: rows[i] for g, i in selected.items()}, "paired_blocks": []}
    for block in range(low, high, 4096):
        end = min(high, block + 4096)
        metrics = reduce_metrics(raw[block - start:end - start])
        comparisons = {group: {unit: {other: {
            "primary_sse": metrics[selected[group]][unit]["residual_sse"],
            "comparator_sse": metrics[i][unit]["residual_sse"], "count": metrics[i][unit]["count"],
            "mse_difference": metrics[selected[group]][unit]["mse"] - metrics[i][unit]["mse"]}
            for other, i in selected.items() if other != group} for unit in ("normalized", "raw")}
            for group in GROUPS if group in selected}
        result["paired_blocks"].append({"start_inclusive": block, "stop_exclusive": end, "comparisons": comparisons})
    return result


def main():
    a = tyro.cli(Args)
    validate_args(a)
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("output must be empty; no resume")
    result = {"version": 7, "stage": a.stage, "view": a.view, "status": "initializing", "args": asdict(a),
        "source_sha256": source_hashes(), "locks": {}, "primary": PRIMARY,
        "protocol": {
            "target": "unclipped uncentered r[t+1]/max(RMS[t],1e-6); raw forecast=mean*known RMS denominator",
            "features": "frozen SignedRidge 12 own/SPY/cross signed cumulative horizons1/4/16/32 plus intercept",
            "model": "exact FP64 static hierarchical unit-noise Gaussian posterior means, positive Schur information updates",
            "controls": "64 unique configurations each pooled/independent/hierarchical; shared local states and pooled Gram/RHS",
            "neural": "one authenticated frozen v6 prefix-selected memory Adam, trained in this same stream",
            "ridge": "frozen offline pooled ridge1; prefix zero; fit first60% including penalized intercept",
            "selection": "three group locks on40-60% normalized MSE only; raw errors reporting only; no suffix selection",
            "null": f"fresh paired raw/normalized Rademacher signs {SIGN_NAMESPACE}; real locks; unchanged features",
            "scope": "rank400 explicitly reused; rank600 additional cohort, not globally pristine dates or iid replication; no profitability claim",
            "confirmation": "train full stream, score cohort-specific last40%; no inherited historical bar boundary",
            "runtime": "seed1 CUDA FP64 inference/metrics, frozen FP32 neural; TF32 off; fullgraph graphs1/8/16; no eager fallback"}}
    writer = SummaryWriter(str(root))
    runner = predictions = journal = None
    started = time.perf_counter()
    previous_terminate = signal.signal(signal.SIGTERM, interrupt_run)
    try:
        neural_config, result["neural_reference"] = load_neural_reference(a.neural_reference)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
        protect_cache(a)
        bank = panel_hd.Bank(a)
        close, volume, symbols, ts = build_panel(a)
        series = return_series(a, close)
        identity = data_identity(a, bank, close, symbols, ts, series)
        result["data_identity"] = identity
        if not np.array_equal(series["valid"], bank.valid.cpu().numpy()):
            raise ValueError("direct-return mask differs from frozen Bank")
        # Authenticate the rank0 data against the neural development stream too.
        neural_identity = result["neural_reference"]["development_identity"]
        if a.rank_offset == 0 and any(identity[key] != neural_identity[key] for key in
                ("cache_sha256", "preprocessing", "symbols", "retained_timestamp_sha256", "return_arrays", "stream_start", "stream_stop", "cut")):
            raise ValueError("rank0 data differ from frozen neural development")
        np.savez(root / "return_arrays.npz", raw_return=series["raw_return"], scale=series["scale"],
                 denominator=series["denominator"], retained_timestamps=ts[series["keep"]])
        result["artifacts"] = {"return_arrays": "return_arrays.npz", "raw_metrics": "raw_metrics.npz"}
        configs = configurations()
        result["configs"] = [asdict(c) for c in configs]
        if a.reference_result:
            result["locks"], result["lock_source"] = load_reference(a.reference_result, result["source_sha256"], configs, result["neural_reference"])
            previous = result["lock_source"]["development_identity"]
            if a.stage == "development" and previous != identity:
                raise ValueError("null must match real data identity")
            if a.stage != "development":
                overlap = sorted(set(previous["symbols"]) & set(identity["symbols"]))
                if overlap:
                    raise ValueError("additional cohort overlaps rank0 stocks")
                result["cohort_overlap_with_development"] = overlap
                result["cohort_role"] = "explicitly_reused_rank400" if a.stage == "replay" else "additional_rank600_not_pristine_dates"
            with (root / "inherited_locks.json").open("x") as stream:
                json.dump(frozen.finite_json({"locks": result["locks"], "source": result["lock_source"],
                    "neural_reference": result["neural_reference"], "target_identity": identity}), stream, indent=2, allow_nan=False)
            result["inherited_lock_sha256"] = frozen.file_sha256(root / "inherited_locks.json")
        if a.view == "rademacher":
            series["normalized_target"], series["raw_target"], result["rademacher_signs"] = randomized_labels(series["normalized_target"], series["raw_target"])
        state = CausalState(bank, torch.tensor(series["observed_valid"], device=bank.dev))
        state.own.zero_()
        state.cross.zero_()
        for t in range(bank.L + 1):
            state.observe(torch.tensor(t, device=bank.dev))
        del close, volume
        model = Learner(bank.N, 13, configs, bank.dev)
        neural = NeuralLearner(bank.F, a.width, (neural_config,), bank.dev, bins=a.bins, seed=a.seed, num_samples=bank.N)
        score_start, score_stop = (bank.L + 1 if a.stage == "development" else bank.cut), bank.T - 1
        runner = Runner(bank, model, neural, series, state, a.log_every, score_start, score_stop)
        observed_masks = series["valid"][runner.start:runner.stop]
        return_updates = np.concatenate(([0], np.cumsum(observed_masks.any(1), dtype=np.int64)))
        result.update(output_names=list(runner.output_names), metric_names=list(METRICS), costs=model.costs,
            neural_costs=neural.costs, mutable_bytes=sum(t.numel() * t.element_size() for t in runner.mutable),
            consumed_until_exclusive=runner.start, score_start_inclusive=score_start, score_stop_exclusive=score_stop,
            prediction_valid_until_exclusive=None)
        result["stream_sha256_before"] = stream_identity(bank, runner)
        journal = np.lib.format.open_memmap(root / "raw_metric_journal.npy", mode="w+", dtype=np.float64,
            shape=(score_stop - score_start, len(runner.output_names), len(METRICS)))
        result["artifacts"]["raw_metric_journal"] = "raw_metric_journal.npy"
        result["checkpoint_raw_bounds"] = [score_start, score_start]
        persist_progress(root, result, journal, None)
        graphs = runner.capture()
        result.update(status="running", compile_seconds=runner.compile_seconds, capture_seconds=runner.capture_seconds,
                      full_state_capture_verified=True, replay_seconds=0.)
        host_raw = np.zeros(tuple(runner.raw.shape), dtype=np.float64)
        prediction_start = bank.cut
        prediction_columns = None

        def open_selected_predictions():
            columns = sorted(set(selected_columns(result["locks"], len(configs)).values()))
            result.update(prediction_columns=columns, prediction_output_names=[runner.output_names[i] for i in columns],
                prediction_start_inclusive=prediction_start, prediction_valid_until_exclusive=prediction_start,
                prediction_units="normalized signed posterior mean; multiply denominator[bar] for raw forecast")
            result["artifacts"]["selected_predictions"] = "selected_predictions.npy"
            storage = np.lib.format.open_memmap(root / "selected_predictions.npy", mode="w+", dtype=np.float64,
                shape=(score_stop - prediction_start, len(columns), bank.N))
            return storage, torch.tensor(columns, device=bank.dev)

        if result["locks"]:
            predictions, prediction_columns = open_selected_predictions()
        boundaries = sorted(set(range(runner.start + a.log_every, runner.stop, a.log_every)) | {bank.cut, score_stop})
        consumed = runner.start
        health = model.candidate_finite().cpu().numpy()
        observation_counts = np.zeros(bank.N, dtype=np.int64)
        for end in boundaries:
            if end <= consumed:
                continue
            before = consumed
            tick = time.perf_counter()
            blocks, remain = divmod(end - consumed, 16)
            for _ in range(blocks):
                graphs[16].replay()
            if remain >= 8:
                graphs[8].replay()
                remain -= 8
            for _ in range(remain):
                graphs[1].replay()
            torch.cuda.synchronize()
            result["replay_seconds"] += time.perf_counter() - tick
            consumed = int(runner.index.cpu())
            updates = consumed - runner.start
            observation_counts += observed_masks[before - runner.start:updates].sum(0, dtype=np.int64)
            if (consumed != end or int(state.observed_steps.cpu()) != consumed or int(model.steps.cpu()) != updates
                    or int(neural.groups[0].adam_steps[0].cpu()) != return_updates[updates]
                    or not np.array_equal(model.observations.cpu().numpy(), observation_counts)):
                raise RuntimeError("causal state/solver/neural observation clocks disagree")
            result["consumed_until_exclusive"] = consumed
            low, high = max(before, score_start), min(end, score_stop)
            if low < high:
                host_raw[low - runner.start:high - runner.start] = runner.raw[low - runner.start:high - runner.start].cpu().numpy()
                journal[low - score_start:high - score_start] = host_raw[low - runner.start:high - runner.start]
                export_low = max(low, prediction_start)
                if predictions is not None and export_low < high:
                    indices = (torch.arange(export_low, high, device=bank.dev) - runner.start) % a.log_every
                    selected = runner.prediction_ring.index_select(0, indices).index_select(1, prediction_columns)
                    predictions[export_low - prediction_start:high - prediction_start] = selected.cpu().numpy()
                    result["prediction_valid_until_exclusive"] = high
            health = model.candidate_finite().cpu().numpy()
            result["candidate_health"] = health.tolist()
            result["solver_failures"] = model.solver_failures.cpu().tolist()
            result["neural_health"] = bool(neural.candidate_finite()[0].cpu())
            if end == bank.cut:
                if a.stage == "development" and a.view == "real":
                    prefix = host_raw[:bank.cut - runner.start]
                    result["prefix_health"] = health.tolist()
                    result["locks"] = select_locks(prefix, configs, runner.start, identity["selection_start"], bank.cut, end, health)
                    persist_prefix_locks(root, prefix, result)
                    predictions, prediction_columns = open_selected_predictions()
                runner.fit_reference()
                result["ridge_reference"] = {"penalty": 1., "intercept_penalized": True,
                    "fit_start_inclusive": runner.start, "fit_stop_exclusive": bank.cut,
                    "horizons": reference.HORIZONS, "coefficient": runner.ridge.coefficient.cpu().tolist(),
                    "prefix_predictions": "zero", "suffix_coefficients": "frozen"}
            if low < high:
                interval = reduce_metrics(host_raw[low - runner.start:high - runner.start])
                for name, row in zip(runner.output_names, interval):
                    for unit in ("normalized", "raw"):
                        writer.add_scalar(f"{a.view}/{name}/{unit}_mse", row[unit]["mse"], high)
            writer.add_scalar("progress/consumed_until_exclusive", consumed, consumed)
            writer.add_scalar("health/failed_structural_candidates", int((~health).sum()), consumed)
            writer.add_scalar("health/neural_finite", int(result["neural_health"]), consumed)
            writer.flush()
            result["checkpoint_raw_bounds"] = [score_start, max(score_start, min(consumed, score_stop))]
            persist_progress(root, result, journal, predictions)
            print(json.dumps({"status": "running", "next_bar": consumed, "score_stop": score_stop}), flush=True)
        regions = {"confirmation": (score_start, score_stop)} if a.stage != "development" else {
            "fit": (runner.start, identity["selection_start"]), "selection": (identity["selection_start"], bank.cut),
            "prefix": (runner.start, bank.cut), "suffix": (bank.cut, score_stop), "full": (runner.start, score_stop)}
        result["phase_metrics"] = {name: summary(host_raw, runner.start, low, high, result["locks"], len(configs)) for name, (low, high) in regions.items()}
        result["posterior_audits"] = {group: model.audit(configs[lock["column"]]) if lock["column"] is not None
                                      else {"status": "failed_no_prefix_candidate"}
                                      for group, lock in result["locks"].items()}
        result["posterior_audit_bounds"] = [runner.start, consumed]
        result["posterior_audit_all_agree"] = all(row["status"] == "agreement" for row in result["posterior_audits"].values())
        result["locked_candidate_status"] = {g: "failed_no_prefix_candidate" if row["column"] is None else
            ("finite" if health[row["column"]] and np.isfinite(host_raw[:, row["column"]]).all() else "nonfinite")
            for g, row in result["locks"].items()}
        result["locked_candidate_status"]["memory_adam"] = "finite" if result["neural_health"] else "nonfinite"
        result["stream_sha256_after"] = stream_identity(bank, runner)
        result["source_sha256_after"] = source_hashes()
        result["cache_sha256_after"] = frozen.file_sha256(a.cache)
        if (result["stream_sha256_before"] != result["stream_sha256_after"] or result["source_sha256_after"] != result["source_sha256"]
                or result["cache_sha256_after"] != identity["cache_sha256"]
                or load_neural_reference(a.neural_reference)[1] != result["neural_reference"]):
            raise RuntimeError("source/data/stream/neural lock bytes changed")
        if a.reference_result and load_reference(a.reference_result, result["source_sha256"], configs, result["neural_reference"])[1] != result["lock_source"]:
            raise RuntimeError("inherited lock bytes changed")
        lock_file = root / ("inherited_locks.json" if a.reference_result else "prefix_locks.json")
        lock_digest = result["inherited_lock_sha256"] if a.reference_result else result["prefix_lock_artifact"]["sha256"]
        if frozen.file_sha256(lock_file) != lock_digest:
            raise RuntimeError("local durable lock proof bytes changed")
        if (result["prediction_valid_until_exclusive"] != score_stop or consumed != runner.stop
                or result["checkpoint_raw_bounds"] != [score_start, score_stop]):
            raise RuntimeError("final evidence does not reach the last forecast")
        result.update(status="completed", source_and_stream_unchanged=True)
    except KeyboardInterrupt as error:
        result.update(status="interrupted", failure={"type": type(error).__name__, "message": str(error)})
        raise
    except Exception as error:
        result.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        try:
            if runner is not None:
                consumed = int(runner.index.cpu())
                result["consumed_until_exclusive"] = consumed
                result["candidate_health"] = runner.model.candidate_finite().cpu().tolist()
                result["solver_failures"] = runner.model.solver_failures.cpu().tolist()
                result["neural_health"] = bool(runner.neural.candidate_finite()[0].cpu())
                low, high = runner.score_start, max(runner.score_start, min(consumed, runner.score_stop))
                np.savez_compressed(root / "posterior_state.npz",
                    **{name: getattr(runner.model, name).detach().cpu().numpy() for name in
                       ("P", "m", "B", "q", "gram", "rhs", "stock_gram", "stock_rhs",
                        "local_precision", "hierarchy_precision", "healthy", "solver_failures", "steps", "observations")},
                    consumed_bounds=np.asarray([runner.start, consumed], dtype=np.int64))
                result["artifacts"]["posterior_state"] = "posterior_state.npz"
                result["posterior_state_bounds"] = [runner.start, consumed]
                raw = runner.raw[low - runner.start:high - runner.start].cpu().numpy()
                np.savez_compressed(root / "raw_metrics.npz", bar_index=np.arange(low, high), output_names=np.asarray(runner.output_names),
                                    **{name: raw[:, :, i] for i, name in enumerate(METRICS)})
                result["raw_metric_bounds"] = [low, high]
                if journal is not None:
                    journal[:high - low] = raw
                    result["checkpoint_raw_bounds"] = [low, high]
                    persist_progress(root, result, journal, predictions)
                # Unexported ring pages are never advertised after interruption.
                result["artifact_sha256"] = {key: frozen.file_sha256(root / filename) for key, filename in result.get("artifacts", {}).items()
                                             if (root / filename).is_file()}
        finally:
            frozen.save_json(root / "results.json", result)
            writer.close()
            signal.signal(signal.SIGTERM, previous_terminate)


if __name__ == "__main__":
    main()

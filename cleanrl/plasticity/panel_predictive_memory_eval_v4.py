"""Full-horizon CUDA experiment with immutable prefix locks and sealed metrics.

Development trains rank0 through the entire frozen stream, selects on 40--60%,
and exposes scores only through 85%. Confirmation trains rank400 from scratch
with development-frozen LRs, scoring only [90145,T-1). No resume is implemented.
The cohort dates overlap: neither split is globally pristine market evidence.
"""

import hashlib
import json
import math
import time
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import panel_distributional_eval_v1 as frozen
from cleanrl.plasticity import panel_frontier_eval_v3 as frontier
from cleanrl.plasticity import panel_hd
from cleanrl.plasticity.panel_predictive_memory_v4 import CausalState, Config, GROUPS, LRS, Learner, REFERENCE_NAMES, configurations
from cleanrl.plasticity.panel_stream import build_panel
from cleanrl.shared import runtime

METRICS = frozen.METRICS
PRIMARY = "memory_categorical_ce_js"
SOURCE_FILES = tuple(dict.fromkeys((*frozen.SOURCE_FILES, "cleanrl/plasticity/panel_frontier_eval_v3.py",
    "cleanrl/plasticity/panel_frontier_model_v3.py", "cleanrl/plasticity/panel_predictive_memory_v4.py",
    "cleanrl/plasticity/panel_predictive_memory_eval_v4.py", "benchmarks/plasticity/panel_frontier_v3_evidence.json")))


@dataclass
class Args(panel_hd.Args):
    output_dir: str = ""
    stage: Literal["development", "confirmation"] = "development"
    reference_result: str = ""
    confirmation_family: str = PRIMARY
    rank_offset: int = 0
    cache: str = "/tmp/panel_cache.npz"
    target: str = "vol"
    width: int = 256
    bins: int = 33
    seed: int = 1
    lrs: tuple[float, ...] = LRS
    view: Literal["real", "permuted"] = "real"
    log_every: int = 4096


def validate_args(a):
    if not a.output_dir.strip():
        raise ValueError("--output-dir required; no resume is supported")
    if (a.target, a.n_stocks, a.lags, a.width, a.bins, a.seed, tuple(a.lrs)) != ("vol", 200, 32, 256, 33, 1, LRS):
        raise ValueError("frozen target, dimensions, seed1 and six-LR grid required")
    if a.log_every < 16:
        raise ValueError("log-every must fit the 16-bar capture")
    if a.rank_offset != (0 if a.stage == "development" else 400):
        raise ValueError("development rank0; confirmation rank400")
    if bool(a.reference_result) != (a.stage == "confirmation" or a.view == "permuted"):
        raise ValueError("confirmation and null require --reference-result; development real cannot inherit locks")
    if a.stage == "confirmation" and a.view != "real":
        raise ValueError("null control is development-only and inherits real-prefix locks")
    if a.confirmation_family not in GROUPS:
        raise ValueError("unknown confirmation family")
    if a.stage == "confirmation" and (a.min_coverage, a.vol_span, a.first_session_before) != (.9, 200, "2018-01-01"):
        raise ValueError("confirmation must preserve the historical rank400 preprocessing")


def authenticate_confirmation(identity, historical):
    """Authenticate the old observation index, independent of the new forecast target."""
    previous = historical["data_identity"]
    for key in ("cache_sha256", "bars", "stocks", "features", "selection_start", "cut", "stream_start", "stream_stop"):
        if identity.get(key) != previous.get(key):
            raise ValueError(f"historical confirmation identity mismatch: {key}")
    for key in ("rank_offset", "n_stocks", "lags", "vol_span", "min_coverage", "first_session_before"):
        if identity["preprocessing"].get(key) != previous["preprocessing"].get(key):
            raise ValueError(f"historical confirmation preprocessing mismatch: {key}")
    root = Path(__file__).resolve().parents[2]
    for path in ("cleanrl/plasticity/panel_hd.py", "cleanrl/plasticity/panel_stream.py", "cleanrl/plasticity/stock_stream.py"):
        if frozen.file_sha256(root / path) != historical["source_sha256"][path]:
            raise ValueError(f"historical preprocessing source changed: {path}")
    boundary = historical["views"]["real"]["forward_range"][1]
    if boundary != 90145 or boundary >= identity["stream_stop"]:
        raise ValueError("historical confirmation boundary does not match the declared protocol")
    return boundary


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {p: frozen.file_sha256(root / p) for p in SOURCE_FILES}


def data_identity(a, bank, close, symbols, ts):
    identity = frozen.data_identity(a, bank)
    identity["configs"] = [asdict(c) for c in configurations(a.lrs)]
    kept_ts = ts[np.isfinite(close[:, 1:]).mean(1) >= a.min_coverage]
    identity["symbols"] = [str(s) for s in symbols]
    if len(symbols) != bank.N or len(set(identity["symbols"])) != bank.N or "SPY" in identity["symbols"]:
        raise ValueError("expected 200 unique stock symbols; SPY is the separate shared column0")
    identity["shared_market_symbol"] = "SPY"
    boundaries = {"stream_start": bank.L + 1, "selection_start": int(.4 * bank.T), "cut": bank.cut,
                  "development_stop": bank.T - 1, "confirmation_start": 90145, "last_forecast": bank.T - 2}
    identity["calendar_boundaries"] = {name: {"bar": i, "timestamp": str(kept_ts[i])}
                                       for name, i in boundaries.items() if i < bank.T}
    identity["retained_timestamp_sha256"] = hashlib.sha256(np.ascontiguousarray(kept_ts).tobytes()).hexdigest()
    return identity


def select_locks(raw, configs, start, selection_start, cut, consumed, health=None):
    if consumed != cut or len(raw) != cut - start or not start <= selection_start < cut:
        raise ValueError("locks require exactly the prefix, with no suffix rows")
    if raw.shape[1] != len(configs) + len(REFERENCE_NAMES):
        raise ValueError("candidate/reference columns disagree")
    scores = frozen.reduce_metrics(raw[selection_start - start:])
    alive = np.isfinite(raw).all(axis=(0, 2))
    if health is not None:
        alive &= np.asarray(health, dtype=bool)
    groups = {g: [i for i, c in enumerate(configs) if c.family == g] for g in GROUPS}
    groups.update(constant_mu=[len(configs)], own_ewma=list(range(len(configs) + 1, len(configs) + 4)),
                  cross_ewma=list(range(len(configs) + 4, len(configs) + 7)), ridge_ewma_1=[len(configs) + 7])
    locks = {}
    for group, columns in groups.items():
        candidates = [{"column": i, "lr": configs[i].lr if i < len(configs) else None,
                       "reference": REFERENCE_NAMES[i - len(configs)] if i >= len(configs) else None,
                       "eligible": bool(alive[i] and scores[i]["count"] and math.isfinite(scores[i]["mse"])),
                       **scores[i]} for i in columns]
        eligible = [c for c in candidates if c["eligible"]]
        winner = min(eligible, key=lambda c: c["mse"]) if eligible else None
        locks[group] = {"column": winner["column"] if winner else None, "lr": winner["lr"] if winner else None,
                        "reference": winner["reference"] if winner else None,
                        "status": "locked" if winner else "failed_nonfinite_or_empty", "candidates": candidates,
                        "selection_start_inclusive": selection_start, "selection_stop_exclusive": cut,
                        "locked_before_bar": cut}
    return locks


def persist_prefix_locks(root, raw, result):
    identity = result["data_identity"]
    if len(raw) != identity["cut"] - identity["stream_start"] or result["consumed_until_exclusive"] != identity["cut"]:
        raise ValueError("prefix locks cannot use suffix rows")
    proof = {"version": 4, "stage": "development", "view": "real", "source_sha256": result["source_sha256"],
             "data_identity": identity, "raw_prefix_sha256": frontier.raw_prefix_hash(raw),
             "health": result["prefix_health"], "locks": result["locks"], "locked_before_bar": identity["cut"]}
    path = root / "prefix_locks.json"
    with path.open("x") as f:
        json.dump(frozen.finite_json(proof), f, indent=2, allow_nan=False)
    result["prefix_lock_artifact"] = {"path": path.name, "sha256": frozen.file_sha256(path)}


def load_reference(path, hashes, configs):
    path = Path(path)
    source = json.loads(path.read_text())
    if (source.get("version"), source.get("stage"), source.get("view"), source.get("status")) != (4, "development", "real", "completed"):
        raise ValueError("reference must be a completed real development result")
    if source.get("source_sha256") != hashes:
        raise ValueError("reference source hashes differ")
    identity = source["data_identity"]
    artifact = source["prefix_lock_artifact"]
    proof_path = path.parent / artifact["path"]
    if frozen.file_sha256(proof_path) != artifact["sha256"]:
        raise ValueError("prefix proof digest mismatch")
    proof = json.loads(proof_path.read_text())
    raw_path = path.parent / source["artifacts"]["raw_metrics"]
    if frozen.file_sha256(raw_path) != source["artifact_sha256"]["raw_metrics"]:
        raise ValueError("reference raw metric digest mismatch")
    count = identity["cut"] - identity["stream_start"]
    with np.load(raw_path, allow_pickle=False) as f:
        if not np.array_equal(f["bar_index"][:count], np.arange(identity["stream_start"], identity["cut"])):
            raise ValueError("reference prefix bar indices differ")
        raw = np.stack([f[key][:count] for key in METRICS], axis=-1)
    if (proof.get("version") != 4 or proof.get("source_sha256") != hashes or proof.get("data_identity") != identity
            or proof.get("locked_before_bar") != identity["cut"] or proof.get("raw_prefix_sha256") != frontier.raw_prefix_hash(raw)
            or proof.get("locks") != source["locks"] or proof.get("health") != source["prefix_health"]):
        raise ValueError("reference prefix provenance mismatch")
    locks = select_locks(raw, configs, identity["stream_start"], identity["selection_start"], identity["cut"], identity["cut"], proof["health"])
    if frozen.finite_json(locks) != source["locks"]:
        raise ValueError("locks disagree with prefix-only selection")
    return locks, {"path": str(path.resolve()), "sha256": frozen.file_sha256(path), "prefix_sha256": artifact["sha256"],
                   "development_identity": identity, "development_consumed_until_exclusive": source["consumed_until_exclusive"]}


class Runner(frozen.Runner):
    """Frozen fullgraph capture/restoration with causal state and bounded prediction I/O."""

    def __init__(self, bank, model, target, valid, state, ring_size, score_start, score_stop):
        super().__init__(bank, model, target, valid)
        self.state, self.score_start, self.score_stop = state, score_start, score_stop
        self.raw = torch.zeros((self.stop - self.start, len(model.output_names), len(METRICS)), dtype=torch.float64, device=bank.dev)
        self.prediction_ring = torch.zeros((ring_size, len(model.output_names), bank.N), device=bank.dev)
        self.ring_size = ring_size
        from cleanrl.plasticity.panel_predictive_memory_v4 import RidgeReference
        self.ridge = RidgeReference(bank.dev)
        self.mutable = (*model.state_tensors(), *state.state_tensors(), *self.ridge.state_tensors(), self.index, self.raw, self.prediction_ring)

    @torch.no_grad()
    def step(self):
        index = self.index.reshape(1)
        self.state.observe(self.index)
        frames = self.state.frames(self.index)
        y = self.target.index_select(0, index).squeeze(0)
        mask = self.valid.index_select(0, index).squeeze(0)
        references = self.state.references()
        ridge_prediction = self.ridge.step(references, y, mask, self.index < self.bank.cut)
        prediction = self.model.step(frames, y, mask, torch.cat((references, ridge_prediction), dim=0))
        visible = (self.index >= self.score_start) & (self.index < self.score_stop)
        scoring_mask = mask & visible
        p = torch.where(scoring_mask[None], prediction.double(), 0.0)
        target = torch.where(scoring_mask, y.double(), 0.0)
        n = len(self.model.output_names)
        row = torch.stack(((p - target[None]).square().sum(1), p.square().sum(1), (p * target[None]).sum(1),
                           target.square().sum().expand(n), scoring_mask.sum().double().expand(n)), dim=1)
        self.raw.index_copy_(0, (self.index - self.start).reshape(1), row[None])
        self.prediction_ring.index_copy_(0, ((self.index - self.start) % self.ring_size).reshape(1),
                                         torch.where(visible, prediction, 0.0)[None])
        self.index.add_(1)


def summary(raw, start, low, high, locks, primary):
    selected = {g: row["column"] for g, row in locks.items() if row["column"] is not None}
    region = raw[max(0, low - start):max(0, high - start)]
    rows = frozen.reduce_metrics(region)
    output = {"start_inclusive": low, "stop_exclusive": high, "bars": len(region),
              "selected": {g: rows[i] for g, i in selected.items()}, "paired_blocks": []}
    if primary not in selected:
        return output
    # Shared chronological blocks, never stock-iid confidence intervals.
    for block in range(low, high, 4096):
        end = min(high, block + 4096)
        metrics = frozen.reduce_metrics(raw[block - start:end - start])
        p = metrics[selected[primary]]
        output["paired_blocks"].append({"start_inclusive": block, "stop_exclusive": end,
            "primary": primary, "comparisons": {g: {"primary_sse": p["residual_sse"],
            "comparator_sse": metrics[i]["residual_sse"], "count": p["count"],
            "mse_difference": p["mse"] - metrics[i]["mse"]} for g, i in selected.items() if g != primary}})
    return output


def persist_progress(root, result, journal, predictions):
    """Publish bounds only after the corresponding metric/prediction pages flush."""
    journal.flush()
    if predictions is not None:
        predictions.flush()
    frozen.save_json(root / "results.json", result)


def interrupt_run(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


def main():
    a = tyro.cli(Args)
    validate_args(a)
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("output must be empty; no resume/checkpoint continuation implemented")
    result = {"version": 4, "stage": a.stage, "view": a.view, "status": "initializing", "args": asdict(a),
        "source_sha256": source_hashes(), "locks": {}, "primary": PRIMARY if a.stage == "development" else a.confirmation_family,
        "protocol": {"target": "Frozen min(z[t+1]^2,25)-mu; forecast after t; no return-direction or profitability claim",
        "normalization": "Frozen mu uses first60% labels including selection; not strictly online prefix normalization",
        "information": "Old t-1..t-32; latest t..t-31; memory adds 3x8 EWMAs through t, zero initialized before bar0",
        "reference": "mu initialized, own/current-valid and cross/current-valid EWMA clipped z[t]^2, half-lives16/128/1024; same40-60% selection",
        "attribution": "Extra history/information and objective factors, not an optimizer or biological advance; joint timing not equalcompute",
        "objective": "Exp raw mean; MSE vs log(mean)+raw_y/mean QLIKE, identical exp parameters, no overflow floors; not Gamma likelihood",
        "null": "Named panel_target_permutation namespace, seed1; joint target/mask time permutation; noise-fit control, not market significance",
        "selection": "48 candidates, eight groups, six LRs each; prefix MSE locks before consuming bar60%; references independently prefix locked",
        "development": "Full-horizon training and reporting; prefix-only locks. Historical development dates are already research-consumed.",
        "confirmation": "Rank400 train from scratch with rank0 development locks before training; scores only[90145,end); overlapping market dates, prior v3 consumed through90145",
        "rng": {"frozen_cpu_trunk": 1, "frozen_cuda_categorical_head": 1, "panel_target_permutation": 1},
        "runtime": "CUDA FP32, TF32 off, fullgraph compile/capture1/8/16; no CPU/eager fallback, no resume"}}
    writer = SummaryWriter(str(root))
    runner = None
    predictions = None
    journal = None
    started = time.perf_counter()
    previous_terminate = signal.signal(signal.SIGTERM, interrupt_run)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
        if Path(a.cache).exists():
            with np.load(a.cache, allow_pickle=False) as cached:
                if int(cached["rank_offset"]) != a.rank_offset or int(cached["n_stocks"]) != a.n_stocks:
                    raise ValueError("refusing to overwrite a different frozen cohort cache")
        bank = panel_hd.Bank(a)
        close, volume, symbols, ts = build_panel(a)
        identity = data_identity(a, bank, close, symbols, ts)
        result["data_identity"] = identity
        configs = configurations(a.lrs)
        if a.reference_result:
            result["locks"], result["lock_source"] = load_reference(a.reference_result, result["source_sha256"], configs)
            previous = result["lock_source"]["development_identity"]
            if a.stage == "development" and previous != identity:
                raise ValueError("null must match real data identity exactly")
            if a.stage == "confirmation":
                history_path = Path(__file__).resolve().parents[2] / "benchmarks/plasticity/panel_frontier_v3_evidence.json"
                historical = json.loads(history_path.read_text())
                result["historical_confirmation_boundary"] = authenticate_confirmation(identity, historical)
                result["historical_evidence_sha256"] = frozen.file_sha256(history_path)
                overlap = sorted(set(previous["symbols"]) & set(identity["symbols"]))
                result["cohort_overlap_with_development"] = overlap
                if overlap or bank.T - 1 <= 90145:
                    raise ValueError("confirmation needs disjoint stocks and data beyond90145")
            # Persist the entire imported choice BEFORE confirmation/null training.
            with (root / "inherited_locks.json").open("x") as f:
                json.dump(frozen.finite_json({"locks": result["locks"], "source": result["lock_source"],
                    "primary": result["primary"], "target_identity": identity}), f, indent=2, allow_nan=False)
            result["inherited_lock_sha256"] = frozen.file_sha256(root / "inherited_locks.json")
        _, _, _, _, observed_valid = panel_hd.series(a, close, volume)
        del close, volume
        state = CausalState(bank, torch.tensor(observed_valid, device=bank.dev))
        # Feature-only causal burn-in, NOT model execution and no label reads.
        for t in range(bank.L + 1):
            state.observe(torch.tensor(t, device=bank.dev))
        target, valid = bank.y, bank.valid
        if a.view == "permuted":
            generator = torch.Generator(device=bank.dev).manual_seed(a.seed)
            permutation = torch.randperm(bank.T, generator=generator, device=bank.dev)
            target, valid = target[permutation], valid[permutation]
        result["stream_sha256_before"] = frontier.stream_identity(bank, target, valid)
        model = Learner(bank.F, a.width, bank.mu, configs, bank.dev, bins=a.bins, seed=a.seed, num_samples=bank.N)
        score_start = bank.L + 1 if a.stage == "development" else 90145
        score_stop = bank.T - 1
        runner = Runner(bank, model, target, valid, state, a.log_every, score_start, score_stop)
        result.update(output_names=list(model.output_names), configs=[asdict(c) for c in configs], costs=model.costs,
            memory_state_coordinates=24 * bank.N, reference_state_coordinates=3 * bank.N + 3,
            parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters),
            mutable_bytes=sum(t.numel() * t.element_size() for t in runner.mutable),
            consumed_until_exclusive=runner.start, score_start_inclusive=score_start, score_stop_exclusive=score_stop)
        result["artifacts"] = {"raw_metrics": "raw_metrics.npz"}
        result["prediction_valid_until_exclusive"] = None
        graphs = runner.capture()
        result.update(status="running", compile_seconds=runner.compile_seconds, capture_seconds=runner.capture_seconds,
                      full_state_capture_verified=True, replay_seconds=0.0)
        frozen.save_json(root / "results.json", result)
        host_raw = np.zeros(tuple(runner.raw.shape), dtype=np.float64)
        journal = np.lib.format.open_memmap(root / "raw_metric_journal.npy", mode="w+", dtype=np.float64,
            shape=(score_stop - score_start, len(model.output_names), len(METRICS)))
        result["artifacts"]["raw_metric_journal"] = "raw_metric_journal.npy"
        result["checkpoint_raw_bounds"] = [score_start, score_start]
        prediction_columns = None
        prediction_start = bank.cut if a.stage == "development" else score_start

        def open_selected_predictions():
            columns = sorted({row["column"] for row in result["locks"].values() if row["column"] is not None})
            result["prediction_columns"] = columns
            result["prediction_output_names"] = [model.output_names[i] for i in columns]
            result["prediction_start_inclusive"] = prediction_start
            result["prediction_valid_until_exclusive"] = prediction_start
            result["artifacts"]["raw_predictions"] = "raw_predictions.npy"
            storage = np.lib.format.open_memmap(root / "raw_predictions.npy", mode="w+", dtype=np.float32,
                shape=(score_stop - prediction_start, len(columns), bank.N))
            return storage, torch.tensor(columns, device=bank.dev)

        if result["locks"]:
            predictions, prediction_columns = open_selected_predictions()
        boundaries = sorted(set(range(runner.start + a.log_every, runner.stop, a.log_every)) |
                            {bank.cut, score_start, score_stop, runner.stop})
        consumed = runner.start
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
            if consumed != end or int(state.observed_steps.cpu()) != consumed:
                raise RuntimeError("causal state or graph index clock disagreement")
            expected_updates = int(valid[runner.start:consumed].any(1).sum().cpu())
            for group in model.groups:
                if int(group.steps.cpu()) != consumed - runner.start or int(group.adam_steps.cpu()) != expected_updates:
                    raise RuntimeError("learner consumption clocks disagree")
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
            if end <= score_stop:
                health = model.candidate_finite().cpu().numpy()
                result["candidate_health"] = health.tolist()
            if end == bank.cut and a.stage == "development" and a.view == "real":
                prefix = host_raw[:bank.cut - runner.start]
                result["prefix_health"] = health.tolist()
                result["locks"] = select_locks(prefix, configs, runner.start, identity["selection_start"], bank.cut, end, health)
                persist_prefix_locks(root, prefix, result)
                predictions, prediction_columns = open_selected_predictions()
            if low < high:
                interval = frozen.reduce_metrics(host_raw[low - runner.start:high - runner.start])
                for name, row in zip(model.output_names, interval):
                    writer.add_scalar(f"{a.view}/{name}/mse", row["mse"], high)
                writer.flush()
            if end == bank.cut:
                runner.ridge.fit()
                result["ridge_reference"] = {"penalty": 1.0, "intercept_penalized": True,
                    "fit_start_inclusive": runner.start, "fit_stop_exclusive": bank.cut,
                    "coefficient": runner.ridge.coefficient.cpu().tolist(),
                    "note": "Fixed ridge1; centered EWMA inputs and centered labels; prefix predictions constant mu, only frozen suffix scores interpret calibration."}
            # Failures remain recorded per candidate. One diverged QLIKE LR must
            # not discard evidence from seven independent groups; no replacement
            # or suffix reselection occurs for a failed locked candidate.
            result["checkpoint_raw_bounds"] = [score_start, max(score_start, min(consumed, score_stop))]
            persist_progress(root, result, journal, predictions)
            print(json.dumps({"status": "running", "next_bar": consumed, "score_stop": score_stop}), flush=True)
        result["phase_metrics"] = summary(host_raw, runner.start,
            bank.cut if a.stage == "development" else score_start, score_stop, result["locks"], result["primary"])
        result["locked_candidate_status"] = {g: "failed_no_prefix_candidate" if row["column"] is None else
            ("finite" if health[row["column"]] and np.isfinite(host_raw[:, row["column"]]).all() else "nonfinite")
            for g, row in result["locks"].items()}
        result["stream_sha256_after"] = frontier.stream_identity(bank, target, valid)
        if result["stream_sha256_before"] != result["stream_sha256_after"] or source_hashes() != result["source_sha256"]:
            raise RuntimeError("source or stream bytes changed")
        if frozen.file_sha256(a.cache) != identity["cache_sha256"]:
            raise RuntimeError("data cache changed")
        result.update(status="completed", source_and_stream_unchanged=True)
    except KeyboardInterrupt as error:
        result.update(status="interrupted", failure={"type": type(error).__name__, "message": str(error)})
        raise
    except Exception as error:
        result.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        if runner is not None:
            # Recover the actual device index, including a failed replay. Metrics
            # outside the declared exposed interval have already been zeroed.
            consumed = int(runner.index.cpu())
            result["consumed_until_exclusive"] = consumed
            low, high = runner.score_start, min(consumed, runner.score_stop)
            raw = runner.raw[low - runner.start:max(low, high) - runner.start].cpu().numpy()
            np.savez_compressed(root / "raw_metrics.npz", bar_index=np.arange(low, max(low, high)),
                **{key: raw[:, :, j] for j, key in enumerate(METRICS)})
            result["raw_metric_bounds"] = [low, max(low, high)]
            if predictions is not None:
                predictions.flush()
            result["artifact_sha256"] = {key: frozen.file_sha256(root / filename) for key, filename in result.get("artifacts", {}).items()
                                         if (root / filename).is_file()}
        frozen.save_json(root / "results.json", result)
        writer.close()
        signal.signal(signal.SIGTERM, previous_terminate)


if __name__ == "__main__":
    main()

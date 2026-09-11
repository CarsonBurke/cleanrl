"""Fresh-panel confirmation of state-dependent expert credit, CUDA/full stream only.

Run ``python -m cleanrl.plasticity.panel_specialist_eval_v2 --output-dir ...``
through mlq. The default rank-200 panel is distinct from the exploratory rank-0
panel: compare arms within this run, never their scores numerically across panels.
Permuted runs require the real results.json and inherit its authenticated locks.
Exit 75 is a checkpointed suffix prune; exit 1 is failure. No resume CLI.
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import panel_hd
from cleanrl.plasticity import panel_distributional_eval_v1 as frozen
from cleanrl.plasticity.panel_distributional_eval_v1 import (
    METRICS, Runner, file_sha256, finite_json, phase_summary, reduce_metrics, save_json,
)
from cleanrl.plasticity.panel_specialist_model_v2 import Config, FAMILIES, Learner
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull

PRIMARY = "state_moe_js"
SOURCE_FILES = (
    "cleanrl/plasticity/panel_specialist_eval_v2.py",
    "cleanrl/plasticity/panel_specialist_model_v2.py",
    *frozen.SOURCE_FILES,
)


@dataclass
class Args(frozen.Args):
    rank_offset: int = 200
    cache: str = "/tmp/panel_distributional_rank200.npz"
    reference_cache: str = "/tmp/panel_cache.npz"
    dataset_role: Literal["confirmation", "secondary"] = "confirmation"


def validate_args(args):
    frozen.validate_args(args)
    if args.rank_offset not in (0, 200):
        raise ValueError("protocol permits rank-offset 200 confirmation or rank-offset 0 secondary only")
    if args.rank_offset == 0 and args.dataset_role != "secondary":
        raise ValueError("rank-offset 0 requires --dataset-role secondary; not fresh confirmation")
    if Path(args.cache).resolve() == Path(args.reference_cache).resolve():
        raise ValueError("experiment cache must not overwrite the reference cache")


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


def cohort_provenance(args):
    """Read stock membership only; SPY is a shared feature, not a stock entry."""
    def members(path, rank):
        with np.load(path, allow_pickle=False) as stored:
            symbols = stored["symbols"].tolist()
            if (int(stored["n_stocks"]), int(stored["rank_offset"])) != (200, rank):
                raise ValueError("cohort cache rank/count metadata disagree")
        if len(symbols) != 200 or len(set(symbols)) != 200 or not all(isinstance(s, str) for s in symbols):
            raise ValueError("cohort cache requires 200 unique stock symbols")
        return symbols

    def symbol_hash(symbols):
        return hashlib.sha256(json.dumps(symbols, separators=(",", ":")).encode()).hexdigest()

    current = members(args.cache, args.rank_offset)
    reference_path = Path(args.reference_cache)
    reference = members(reference_path, 0) if reference_path.is_file() else None
    overlap = sorted(set(current) & set(reference)) if reference is not None else None
    return {
        "symbols": current, "symbols_sha256": symbol_hash(current),
        "reference_symbols": reference,
        "reference_symbols_sha256": symbol_hash(reference) if reference is not None else None,
        "reference_cache_sha256": file_sha256(reference_path) if reference is not None else None,
        "overlap_symbols": overlap, "disjoint_from_reference": overlap == [],
        "market_feature": "SPY shared across panels; not included in the stock-symbol lists",
    }


def data_identity(args, bank):
    identity = frozen.data_identity(args, bank)
    identity.update(configs=[asdict(Config(family, lr)) for family in FAMILIES for lr in args.lrs],
                    dataset_role=args.dataset_role,
                    model_protocol="panel_specialist_v2_four_independent_tanh_experts",
                    cohort=cohort_provenance(args))
    return identity


def select_locks(raw, configs, stream_start, selection_start, cut, consumed, candidate_finite=None):
    """Call only at cut, with exactly consumed rows; cannot inspect suffix labels."""
    if consumed != cut or len(raw) != cut - stream_start or not stream_start <= selection_start < cut:
        raise ValueError("locks require the exact prefix boundary and no forward rows")
    if raw.shape[1] != len(configs):
        raise ValueError("config columns disagree")
    scores = reduce_metrics(raw[selection_start - stream_start:])
    alive = np.isfinite(raw).all(axis=(0, 2))
    if candidate_finite is not None:
        alive &= np.asarray(candidate_finite, dtype=bool)
    locks = {}
    for family in FAMILIES:
        columns = [i for i, config in enumerate(configs) if config.family == family]
        candidates = [{"column": i, **asdict(configs[i]), **scores[i],
                       "eligible": bool(alive[i] and scores[i]["count"] and math.isfinite(scores[i]["mse"]))}
                      for i in columns]
        eligible = [row for row in candidates if row["eligible"]]
        winner = min(eligible, key=lambda row: row["mse"]) if eligible else None
        locks[family] = {
            "status": "locked" if winner else "failed_nonfinite_or_empty",
            "column": winner["column"] if winner else None,
            "lr": winner["lr"] if winner else None,
            "endpoint_winner": bool(winner and winner["column"] in (columns[0], columns[-1])),
            "selection_start_inclusive": selection_start, "selection_stop_exclusive": cut,
            "locked_before_bar": cut, "candidates": candidates,
        }
    return locks


def load_real_locks(path, identity, hashes, configs):
    """Authenticate real locks against raw prefix metrics, never the null or suffix."""
    path = Path(path)
    source = json.loads(path.read_text())
    if (source.get("view") != "real" or source.get("status") not in ("completed", "pruned")
            or source.get("data_identity") != identity or source.get("source_sha256") != hashes):
        raise ValueError("real lock source must be a completed/pruned matching real experiment")
    if source.get("consumed_until_exclusive", 0) < identity["cut"]:
        raise ValueError("real source never reached the lock boundary")
    raw_path = path.parent / source["artifacts"]["raw_metrics"]
    if file_sha256(raw_path) != source["artifact_sha256"]["raw_metrics"]:
        raise ValueError("real raw metrics digest mismatch")
    with np.load(raw_path, allow_pickle=False) as stored:
        count = identity["cut"] - identity["stream_start"]
        expected_bars = np.arange(identity["stream_start"], identity["cut"])
        if not np.array_equal(stored["bar_index"][:count], expected_bars):
            raise ValueError("real prefix bar indices disagree")
        raw = np.stack([stored[key][:count] for key in METRICS], axis=-1)
    expected = select_locks(raw, configs, identity["stream_start"], identity["selection_start"],
                            identity["cut"], identity["cut"], source.get("prefix_state_finite"))
    if finite_json(expected) != source.get("locks") or any(row["column"] is None for row in expected.values()):
        raise ValueError("real lock provenance disagrees with prefix scores or lacks a family")
    return expected, {"results_path": str(path.resolve()), "results_sha256": file_sha256(path),
                      "raw_metrics_sha256": file_sha256(raw_path)}

def paired_comparisons(summary, locks):
    primary = locks.get(PRIMARY, {}).get("column")
    if primary is None:
        return {}
    result = {}
    for phase, record in summary.items():
        if not phase.startswith("forward"):
            continue
        rows = record["metrics"]
        selected = {family: rows[lock["column"]] for family, lock in locks.items() if lock["column"] is not None}
        result[phase] = {"bars": record["bars"], "sample_count": record["sample_count"],
                         "complete": record["complete"], "selected": selected,
                         "primary_minus_baseline": {
                             family: {"mse": rows[primary]["mse"] - row["mse"],
                                      "error_ratio": rows[primary]["error_ratio"] - row["error_ratio"]}
                             for family, row in selected.items() if family != PRIMARY}}
    return result


def save_artifacts(root, runner, raw, consumed, policy, result):
    if len(raw) != consumed - runner.start:
        raise ValueError("artifact rows must exactly match the consumed checkpoint")
    temporary = root / "raw_metrics.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, bar_index=np.arange(runner.start, consumed, dtype=np.int64),
                 output_names=np.asarray(runner.model.output_names),
                 **{key: raw[:, :, index] for index, key in enumerate(METRICS)})
    temporary.replace(root / "raw_metrics.npz")
    checkpoint = root / "checkpoint.pt.tmp"
    torch.save({"version": 2, "consumed_until_exclusive": consumed,
                "stream_start": runner.start, "runner_index": runner.index.cpu(),
                "model_state": [tensor.cpu() for tensor in runner.model.state_tensors()],
                "model_state_shapes": [list(tensor.shape) for tensor in runner.model.state_tensors()],
                "model_configs": [asdict(config) for config in runner.model.configs],
                "locks": finite_json(result["locks"]), "autocull": policy.state_dict() if policy else None,
                "metrics_artifact": "raw_metrics.npz", "metric_keys": METRICS,
                "restore_note": "Construct identical learner/runner; copy model_state in state_tensors order, copy runner_index, zero runner.raw then restore consumed metric rows from NPZ. No resume CLI."}, checkpoint)
    checkpoint.replace(root / "checkpoint.pt")
    identity = result["data_identity"]
    summary = phase_summary(raw, runner.start, consumed, identity["selection_start"],
                            identity["cut"], runner.stop, runner.model.output_names)
    result.update(consumed_until_exclusive=consumed, consumed_bars=len(raw),
                  phase_metrics=summary, paired_comparisons=paired_comparisons(summary, result["locks"]),
                  autocull_state=policy.state_dict() if policy else None,
                  artifacts={"raw_metrics": "raw_metrics.npz", "checkpoint": "checkpoint.pt",
                             "results": "results.json", "tensorboard": "."},
                  artifact_sha256={"raw_metrics": file_sha256(root / "raw_metrics.npz"),
                                   "checkpoint": file_sha256(root / "checkpoint.pt")})
    save_json(root / "results.json", result)


def check_clocks(runner, consumed):
    """Host-boundary validation of both independently updated learner groups."""
    expected = consumed - runner.start
    if any(int(clock.cpu()) != expected for clock in (runner.model.steps, runner.model.moe_steps)):
        raise RuntimeError("runner/dense/MoE clocks do not describe a coherent checkpoint")
    adam_steps = int(runner.model.adam_steps.cpu())
    if not 0 <= adam_steps <= expected or int(runner.model.moe_adam_steps.cpu()) != adam_steps:
        raise RuntimeError("dense/MoE optimizer clocks do not describe a coherent checkpoint")


def recover_consumed(runner):
    consumed = int(runner.index.cpu())
    if not runner.start <= consumed <= runner.stop:
        raise RuntimeError("failed runner index is outside the stream")
    check_clocks(runner, consumed)
    return consumed, runner.raw[:consumed - runner.start].cpu().numpy().copy()


def main():
    args = tyro.cli(Args)
    validate_args(args)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("output directory must be empty; existing experiment artifacts are immutable")
    started = time.perf_counter()
    result = {"version": 2, "status": "initializing", "view": args.view,
              "args": asdict(args), "primary_family": PRIMARY, "locks": {}, "curves": [],
              "source_sha256": source_hashes(), "tensorboard_dir": str(root),
              "protocol": {
                  "normalization": "Unchanged Bank.mu uses masked targets from first 60% of the full panel, including the selection window. Shared fixed prefix normalization; NOT a claim of streaming-only first-60% label causality.",
                  "selection": "Independent family LR locks by masked sample-weighted prequential MSE on [floor(.4T),floor(.6T)); tie breaks at the first LR. All seven locks persisted before label cut.",
                  "forward": "Primary family state_moe_js is fixed a priori. Continuous predict-before-update through T-2, no reset, no suffix LR/family selection. Every grid candidate retained, even nonfinite.",
                  "null": "Existing Bank PERM shuffles target and validity together with its fixed seed0 permutation; features unchanged. Inherit all real locks. A permutation baseline, not financial significance.",
                  "culling": "Suffix only, locked state_moe_js error_ratio min_delta .001; ProxyCull bar-index units (one observation is one cross-sectional bar, NOT 200 stocks), median3, EMA half-life8192 bars, warmup16384 bars, stride4096 bars, patience3. Pruned exit75; numerical failure exit1.",
                  "precision": "CUDA-only FP32 model, FP64 metric reductions, highest matmul precision, TF32 off, fullgraph compile, verified CUDA graphs1/8/16; no eager fallback.",
                  "comparison": "Rank-offset 200 is a fresh confirmation panel, not numerically comparable to exploratory rank-offset 0 scores. Only within-panel paired arm differences are valid. Rank-offset 0 is secondary only.",
                  "model": "Four independent F-64-64-1 tanh experts (width256), uniform versus learned constant (bias-only) versus state-dependent softmax router, ordinary MSE. State-dependent exact prediction-error credit; no entropy, balancing or other auxiliary losses. All expert arms share initial expert weights; router starts uniform. Dense scalar and categorical CE-JS controls delegate frozen v1 unchanged.",
                  "cache": "Frozen build_panel cache behavior retained; cached universe validity checks are its n_stocks/rank_offset checks. Actual cache bytes and preprocessing args fingerprinted.",
              }}
    writer = SummaryWriter(str(root))
    runner, raw, consumed, policy = None, None, None, None
    exit_code = 0
    try:
        save_json(root / "results.json", result)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
        bank = panel_hd.Bank(args)
        if (bank.N, bank.F) != (200, 257):
            raise ValueError("frozen panel dimensions differ from 200 stocks / 257 features")
        identity = data_identity(args, bank)
        result["data_identity"] = identity
        if args.dataset_role == "confirmation" and not identity["cohort"]["disjoint_from_reference"]:
            raise ValueError("confirmation requires an existing rank-0 reference cache and zero stock-symbol overlap")
        configs = tuple(Config(family, lr) for family in FAMILIES for lr in args.lrs)
        if args.view == "permuted":
            result["locks"], result["lock_source"] = load_real_locks(args.real_result, identity, result["source_sha256"], configs)
        streams = dict((name, (y, mask)) for name, y, mask in bank.streams())
        target, valid = streams["REAL" if args.view == "real" else "PERM"]
        model = Learner(bank.F, args.width, bank.mu, configs, bank.dev, bins=args.bins,
                        seed=args.seed, kappa=1.0, num_samples=bank.N)
        runner = Runner(bank, model, target, valid)
        consumed = runner.start
        raw = np.empty((0, len(configs), len(METRICS)), dtype=np.float64)
        result.update(output_names=list(model.output_names), configs=[asdict(config) for config in configs],
                      support={"spacing": "uniform", "raw_min": 0.0, "raw_max": 25.0,
                               "bins": args.bins, "centered_by": identity["mu"]},
                      parameter_count_allocated=model.allocated_parameter_count,
                      parameter_count_effective_by_config=list(model.effective_parameter_counts), maximum_bars=runner.stop - runner.start)
        policy = ProxyCull(1, {"error_ratio": 1e-3}) if args.autocull else None
        writer.add_text("protocol", json.dumps(result["protocol"], indent=2))
        graphs = runner.capture()
        result.update(status="running", compile_seconds=runner.compile_seconds,
                      capture_seconds=runner.capture_seconds, capture_steps=[1, 8, 16],
                      capture_complete_state_verified=True, replay_seconds=0.0)
        save_artifacts(root, runner, raw, consumed, policy, result)
        boundaries = sorted(set(range(runner.start + args.log_every, runner.stop, args.log_every))
                            | {identity["selection_start"], bank.cut, runner.stop})
        host_raw = np.empty(tuple(runner.raw.shape), dtype=np.float64)
        for end in boundaries:
            if end <= consumed:
                continue
            previous = consumed
            replay_start = time.perf_counter()
            blocks, remainder = divmod(end - previous, 16)
            for _ in range(blocks):
                graphs[16].replay()
            if remainder >= 8:
                graphs[8].replay()
                remainder -= 8
            for _ in range(remainder):
                graphs[1].replay()
            torch.cuda.synchronize()
            result["replay_seconds"] += time.perf_counter() - replay_start
            consumed = int(runner.index.cpu())
            if consumed != end:
                raise RuntimeError("runner index disagrees with consumed bar boundary")
            check_clocks(runner, consumed)
            chunk = runner.raw[previous - runner.start:consumed - runner.start].cpu().numpy()
            host_raw[previous - runner.start:consumed - runner.start] = chunk
            raw = host_raw[:consumed - runner.start]
            interval = reduce_metrics(chunk)
            candidate_state = model.candidate_finite().cpu().numpy()
            result["candidate_health"] = [
                {"name": name, "status": "finite" if candidate_state[i] and np.isfinite(raw[:, i]).all() else "nonfinite"}
                for i, name in enumerate(model.output_names)]
            if end == bank.cut and args.view == "real":
                result["prefix_state_finite"] = candidate_state.tolist()
                result["locks"] = select_locks(raw, configs, runner.start, identity["selection_start"],
                                               bank.cut, consumed, candidate_state)
                # This durable checkpoint occurs before any graph can consume bar cut.
                save_artifacts(root, runner, raw, consumed, policy, result)
            selected = [lock["column"] for lock in result["locks"].values()]
            if end >= bank.cut and (len(selected) != len(FAMILIES) or None in selected):
                raise FloatingPointError("a family has no finite nonempty prefix candidate")
            if end >= bank.cut and any(not candidate_state[i] or not np.isfinite(raw[:, i]).all() for i in selected):
                raise FloatingPointError("a locked family became nonfinite; retained raw failed candidates")
            decision = None
            if previous >= bank.cut and policy:
                primary_column = result["locks"][PRIMARY]["column"]
                score = interval[primary_column]["error_ratio"]
                if not math.isfinite(score):
                    raise FloatingPointError("locked primary suffix metric is undefined/nonfinite")
                decision = policy.observe(consumed, {"error_ratio": [score]}, phase="forward", phase_start=bank.cut)
            result["curves"].append({"start_inclusive": previous, "stop_exclusive": consumed,
                                     "metrics": interval, "autocull": decision})
            for name, row in zip(model.output_names, interval):
                for metric in ("mse", "error_ratio", "prediction_energy_ratio", "signed_cross_term_ratio"):
                    writer.add_scalar(f"{args.view}/{name}/{metric}", row[metric], consumed)
            result["wall_seconds"] = time.perf_counter() - started
            if decision:
                result.update(status="pruned", prune=decision, exit_code=PRUNED_EXIT_CODE)
                save_json(root / "autocull.json", decision)
                print("AUTOCULL " + json.dumps(finite_json(decision), allow_nan=False), flush=True)
                exit_code = PRUNED_EXIT_CODE
            save_artifacts(root, runner, raw, consumed, policy, result)
            writer.flush()
            print(json.dumps({"status": result["status"], "next_bar": consumed,
                              "maximum_bar_exclusive": runner.stop,
                              "locked_primary_lr": result["locks"].get(PRIMARY, {}).get("lr")}), flush=True)
            if decision:
                break
        if not exit_code:
            result.update(status="completed", exit_code=0)
    except Exception as error:
        result.update(status="failed", exit_code=1,
                      failure={"type": type(error).__name__, "message": str(error)})
        exit_code = 1
        # Do not hide errors or turn a failed compilation/graph into an eager run.
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        try:
            if runner is not None and raw is not None:
                if result["status"] == "failed":
                    try:
                        consumed, raw = recover_consumed(runner)
                        save_artifacts(root, runner, raw, consumed, policy, result)
                    except Exception as artifact_error:
                        result["artifact_failure"] = {
                            "type": type(artifact_error).__name__, "message": str(artifact_error),
                            "note": "Failure checkpoint could not be recovered; any prior artifacts are not claimed to represent the final device state.",
                        }
                        save_json(root / "results.json", result)
                else:
                    save_artifacts(root, runner, raw, consumed, policy, result)
            else:
                save_json(root / "results.json", result)
        finally:
            writer.close()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

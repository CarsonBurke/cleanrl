"""Third-cohort confirmation of categorical CE-JS superiority, CUDA/full stream only.

Run ``python -m cleanrl.plasticity.panel_frontier_eval_v3 --output-dir ...``
through mlq. The default rank-400 panel is disjoint from the rank-0 and rank-200
panels: compare arms within this run, never their scores numerically across panels.
Permuted runs require the real results.json and inherit its authenticated locks.
Exit 75 is a checkpointed suffix prune; exit 1 is failure. No resume CLI.
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import panel_hd
from cleanrl.plasticity import panel_distributional_eval_v1 as frozen
from cleanrl.plasticity.panel_distributional_eval_v1 import (
    METRICS, Runner, file_sha256, finite_json, phase_summary, reduce_metrics, save_json,
)
from cleanrl.plasticity.panel_frontier_model_v3 import Config, FAMILIES, Learner
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull

PRIMARY = "categorical_ce_js"
SOURCE_FILES = (
    "cleanrl/plasticity/panel_frontier_eval_v3.py",
    "cleanrl/plasticity/panel_frontier_model_v3.py",
    *frozen.SOURCE_FILES,
)


@dataclass
class Args(frozen.Args):
    rank_offset: int = 400
    cache: str = "/tmp/panel_frontier_rank400.npz"
    reference_caches: tuple[str, ...] = ("/tmp/panel_cache.npz", "/tmp/panel_distributional_rank200.npz")


def validate_args(args):
    frozen.validate_args(args)
    if args.rank_offset != 400:
        raise ValueError("third-cohort confirmation fixes rank-offset 400")
    if args.bins != 33 or args.lrs != Args.__dataclass_fields__["lrs"].default:
        raise ValueError("confirmation fixes bins=33 and the six preregistered learning rates")
    references = tuple(Path(path).resolve() for path in args.reference_caches)
    if len(references) != 2 or len(set(references)) != 2:
        raise ValueError("two distinct reference caches are required (rank 0, rank 200)")
    current = Path(args.cache).resolve()
    if any(current == path or (current.exists() and path.exists() and current.samefile(path))
           for path in references):
        raise ValueError("experiment cache must not overwrite either reference cache")


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


def cohort_provenance(args):
    """Verify actual stock membership; SPY is a shared feature, not a stock entry."""
    def members(path, rank):
        with np.load(path, allow_pickle=False) as stored:
            symbols = stored["symbols"].tolist()
            if (int(stored["n_stocks"]), int(stored["rank_offset"])) != (200, rank):
                raise ValueError("cohort cache rank/count metadata disagree")
        if len(symbols) != 200 or not all(isinstance(s, str) and s for s in symbols) or len(set(symbols)) != 200:
            raise ValueError("cohort cache requires 200 unique stock symbols")
        return symbols

    def symbol_hash(symbols):
        return hashlib.sha256(json.dumps(symbols, separators=(",", ":")).encode()).hexdigest()

    current = members(args.cache, 400)
    references = []
    for path, rank in zip(args.reference_caches, (0, 200), strict=True):
        reference = members(path, rank)  # Missing references fail closed, never create them.
        overlap = sorted(set(current) & set(reference))
        references.append({"rank_offset": rank, "symbols": reference,
                           "symbols_sha256": symbol_hash(reference),
                           "cache_sha256": file_sha256(path), "overlap_symbols": overlap,
                           "disjoint": not overlap})
    return {"symbols": current, "symbols_sha256": symbol_hash(current),
            "references": references,
            "disjoint_from_references": all(row["disjoint"] for row in references),
            "market_feature": "SPY shared across panels; not included in the stock-symbol lists"}


def require_disjoint(cohort):
    if not cohort["disjoint_from_references"]:
        overlaps = {row["rank_offset"]: row["overlap_symbols"] for row in cohort["references"] if not row["disjoint"]}
        raise ValueError(f"third-cohort confirmation requires zero stock-symbol overlap: {overlaps}")


def data_identity(args, bank):
    identity = frozen.data_identity(args, bank)
    identity.update(configs=[asdict(Config(family, lr)) for family in FAMILIES for lr in args.lrs],
                    dataset_role="confirmation",
                    model_protocol="panel_frontier_v3_categorical_loss_and_scalar_capacity_controls",
                    cohort=cohort_provenance(args))
    return identity


def stream_identity(bank, target, valid):
    """Byte fingerprints of the shared frozen feature state and selected label view."""
    tensors = {name: getattr(bank, name) for name in
               ("zt", "vst", "cst", "acst", "lag_idx", "ones", "mu", "y", "valid", "perm_t")}
    tensors.update(selected_target=target, selected_valid=valid)
    return {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                   "sha256": hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name, tensor in tensors.items()}


def capacity_report(model):
    return {"parameter_count_allocated": model.allocated_parameter_count,
            "parameter_count_effective_by_config": list(model.effective_parameter_counts),
            "widths_by_config": list(model.widths), "group_metadata": list(model.group_metadata)}


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


def raw_prefix_hash(raw):
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(raw.shape), "dtype": str(raw.dtype)}, sort_keys=True).encode())
    digest.update(memoryview(np.ascontiguousarray(raw)).cast("B"))
    return digest.hexdigest()


def persist_prefix_locks(root, raw, consumed, result):
    """Write a once-only lock manifest before any graph can consume the suffix."""
    identity = result["data_identity"]
    if result["view"] != "real" or consumed != identity["cut"] or len(raw) != consumed - identity["stream_start"]:
        raise ValueError("prefix proof requires the exact real prefix boundary")
    path = root / "prefix_locks.json"
    if path.exists():
        raise ValueError("prefix lock artifact is immutable")
    proof = {"version": 3, "view": "real", "primary_family": PRIMARY,
             "locked_before_bar": consumed, "data_identity": identity,
             "source_sha256": result["source_sha256"],
             "raw_prefix_sha256": raw_prefix_hash(raw),
             "prefix_state_finite": result["prefix_state_finite"], "locks": result["locks"]}
    save_json(path, proof)
    result["prefix_lock_artifact"] = {"path": path.name, "sha256": file_sha256(path)}


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
    artifact = source.get("prefix_lock_artifact", {})
    proof_path = path.parent / artifact.get("path", "prefix_locks.json")
    if not proof_path.is_file() or file_sha256(proof_path) != artifact.get("sha256"):
        raise ValueError("real prefix lock artifact digest mismatch")
    proof = json.loads(proof_path.read_text())
    if (proof.get("version") != 3 or proof.get("view") != "real" or proof.get("primary_family") != PRIMARY
            or proof.get("locked_before_bar") != identity["cut"]
            or proof.get("data_identity") != identity or proof.get("source_sha256") != hashes
            or proof.get("raw_prefix_sha256") != raw_prefix_hash(raw)
            or proof.get("locks") != source.get("locks")
            or proof.get("prefix_state_finite") != source.get("prefix_state_finite")):
        raise ValueError("real lock provenance disagrees with immutable prefix proof")
    expected = select_locks(raw, configs, identity["stream_start"], identity["selection_start"],
                            identity["cut"], identity["cut"], proof["prefix_state_finite"])
    if finite_json(expected) != source.get("locks") or any(row["column"] is None for row in expected.values()):
        raise ValueError("real lock provenance disagrees with prefix scores or lacks a family")
    return expected, {"results_path": str(path.resolve()), "results_sha256": file_sha256(path),
                      "raw_metrics_sha256": file_sha256(raw_path),
                      "prefix_locks_sha256": file_sha256(proof_path),
                      "raw_prefix_sha256": proof["raw_prefix_sha256"]}

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
    torch.save({"version": 3, "consumed_until_exclusive": consumed,
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
    """All independent groups must consume the same bars and nonempty-mask updates."""
    expected = consumed - runner.start
    clocks = [(int(steps.cpu()), int(adam.cpu())) for steps, adam in runner.model.clocks]
    if not clocks or any(steps != expected for steps, _ in clocks):
        raise RuntimeError("runner/group clocks do not describe a coherent checkpoint")
    if any(not 0 <= adam <= expected or adam != clocks[0][1] for _, adam in clocks):
        raise RuntimeError("group optimizer clocks do not describe a coherent checkpoint")
    return [{"consumed_steps": steps, "optimizer_steps": adam} for steps, adam in clocks]


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
    result = {"version": 3, "status": "initializing", "view": args.view,
              "args": asdict(args), "primary_family": PRIMARY, "locks": {}, "curves": [],
              "source_sha256": source_hashes(), "tensorboard_dir": str(root),
              "protocol": {
                  "normalization": "Unchanged Bank.mu uses masked targets from first 60% of the full panel, including the selection window. Shared fixed prefix normalization; NOT a claim of streaming-only first-60% label causality.",
                  "selection": "Independent family LR locks by masked sample-weighted prequential MSE on [floor(.4T),floor(.6T)); tie breaks at the first LR. All four locks persisted before label cut.",
                  "forward": "Primary family categorical_ce_js is fixed a priori. Continuous predict-before-update through T-2, no reset, no suffix LR/family selection. Every grid candidate retained, even nonfinite.",
                  "null": "Existing Bank PERM shuffles target and validity together with its fixed seed0 permutation; features unchanged. Inherit all real locks. A permutation baseline, not financial significance.",
                  "culling": "Suffix only, locked categorical_ce_js error_ratio min_delta .001; ProxyCull bar-index units (one observation is one cross-sectional bar, NOT 200 stocks), median3, EMA half-life8192 bars, warmup16384 bars, stride4096 bars, patience3. Pruned exit75; numerical failure exit1.",
                  "precision": "CUDA-only FP32 model, FP64 metric reductions, highest matmul precision, TF32 off, fullgraph compile, verified CUDA graphs1/8/16; no eager fallback.",
                  "comparison": "Rank-offset 400 must have 200 unique actual stocks disjoint from both rank-0 and rank-200 reference caches. Only within-panel paired arm differences are valid; no cross-cohort numerical ranking.",
                  "model": "Frozen v1 JS learning rule in four independent groups: scalar width256; scalar smallest hidden width matching or exceeding categorical parameter count; categorical expected-value MSE+JS; categorical CE+JS. Categorical groups have identical initialization, support, head, gate and parameters, differing only in loss. No extra LR, regularizer, clipping or target transform.",
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
        require_disjoint(identity["cohort"])
        configs = tuple(Config(family, lr) for family in FAMILIES for lr in args.lrs)
        if args.view == "permuted":
            result["locks"], result["lock_source"] = load_real_locks(args.real_result, identity, result["source_sha256"], configs)
        streams = dict((name, (y, mask)) for name, y, mask in bank.streams())
        target, valid = streams["REAL" if args.view == "real" else "PERM"]
        result["stream_sha256_before"] = stream_identity(bank, target, valid)
        model = Learner(bank.F, args.width, bank.mu, configs, bank.dev, bins=args.bins,
                        seed=args.seed, kappa=1.0, num_samples=bank.N)
        runner = Runner(bank, model, target, valid)
        consumed = runner.start
        raw = np.empty((0, len(configs), len(METRICS)), dtype=np.float64)
        result.update(output_names=list(model.output_names), configs=[asdict(config) for config in configs],
                      support={"spacing": "uniform", "raw_min": 0.0, "raw_max": 25.0,
                               "bins": args.bins, "centered_by": identity["mu"]},
                      **capacity_report(model), maximum_bars=runner.stop - runner.start)
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
            result["group_clocks"] = check_clocks(runner, consumed)
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
                persist_prefix_locks(root, raw, consumed, result)
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
        result["stream_sha256_after"] = stream_identity(bank, target, valid)
        result["stream_unchanged"] = result["stream_sha256_before"] == result["stream_sha256_after"]
        result["source_unchanged"] = source_hashes() == result["source_sha256"]
        result["data_unchanged"] = data_identity(args, bank) == identity
        if not all(result[key] for key in ("stream_unchanged", "source_unchanged", "data_unchanged")):
            raise RuntimeError("frozen source, cohort data or shared stream changed during evaluation")
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

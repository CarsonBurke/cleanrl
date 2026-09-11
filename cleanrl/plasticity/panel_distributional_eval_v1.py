"""Frozen-panel, five-family prequential comparison; CUDA and full stream only.

Run with ``python -m cleanrl.plasticity.panel_distributional_eval_v1
--output-dir runs/panel_distributional_real`` through mlq. A permuted run requires
``--view permuted --real-result <real-output>/results.json``. Exit 75 is an
intentional, checkpointed suffix prune; exit 1 is a failure. No resume CLI.
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
from cleanrl.plasticity.panel_distributional_model_v1 import Config, FAMILIES, Learner
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull

PRIMARY = "categorical_ce"
METRICS = ("residual_sse", "prediction_energy", "cross_sum", "target_energy", "count")
SOURCE_FILES = (
    "cleanrl/plasticity/panel_distributional_eval_v1.py",
    "cleanrl/plasticity/panel_distributional_model_v1.py",
    "cleanrl/plasticity/panel_hd.py",
    "cleanrl/plasticity/panel_hd_gate.py",
    "cleanrl/plasticity/panel_stream.py",
    "cleanrl/plasticity/stock_stream.py",
    "cleanrl/shared/runtime.py",
    "cleanrl/shared/autocull.py",
    "cleanrl/shared/two_hot.py",
)


@dataclass
class Args(panel_hd.Args):
    output_dir: str = ""
    target: str = "vol"
    width: int = 256
    bins: int = 33
    seed: int = 1
    lrs: tuple[float, ...] = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    view: Literal["real", "permuted"] = "real"
    real_result: str = ""
    log_every: int = 4096
    autocull: bool = True


def validate_args(args):
    if not args.output_dir.strip():
        raise ValueError("--output-dir is required")
    if (args.target, args.n_stocks, args.lags, args.width, args.seed) != ("vol", 200, 32, 256, 1):
        raise ValueError("protocol fixes target=vol, stocks=200, lags=32, width=256, seed=1")
    if args.bins < 2 or args.log_every < 1:
        raise ValueError("bins must be >=2 and log-every positive")
    if not args.lrs or any(not math.isfinite(lr) or lr <= 0 for lr in args.lrs):
        raise ValueError("learning rates must be positive and finite")
    if tuple(sorted(set(args.lrs))) != tuple(args.lrs):
        raise ValueError("learning rates must be unique and increasing")
    if (args.view == "permuted") != bool(args.real_result):
        raise ValueError("only the permuted view requires --real-result")


def finite_json(value):
    """JSON reports null for nonfinite numbers; raw NPZ deliberately keeps them."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


def data_identity(args, bank):
    # Hash the actual frozen input cache, including values, timestamps and symbols,
    # not its pathname/mtime. Bank itself is unchanged and owns preprocessing.
    return {
        "cache_sha256": file_sha256(args.cache),
        "preprocessing": {key: getattr(args, key) for key in (
            "rank_offset", "n_stocks", "lags", "vol_span", "min_coverage",
            "first_session_before", "target")},
        "bars": bank.T, "stocks": bank.N, "features": bank.F,
        "selection_start": int(0.4 * bank.T), "cut": bank.cut,
        "stream_start": bank.L + 1, "stream_stop": bank.T - 1,
        "mu": float(bank.mu.cpu()),
        "width": args.width, "bins": args.bins, "seed": args.seed,
        "kappa": 1.0, "configs": [asdict(Config(family, lr)) for family in FAMILIES for lr in args.lrs],
    }


class Runner:
    """One fixed-cross-section update; index always denotes the next unconsumed bar."""

    def __init__(self, bank, model, target, valid):
        self.bank, self.model, self.target, self.valid = bank, model, target, valid
        self.start, self.stop = bank.L + 1, bank.T - 1
        self.index = torch.tensor(self.start, dtype=torch.int64, device=bank.dev)
        self.raw = torch.zeros((self.stop - self.start, len(model.configs), len(METRICS)),
                               dtype=torch.float64, device=bank.dev)
        self.mutable = (*model.state_tensors(), self.index, self.raw)
        self.compile_seconds = 0.0
        self.capture_seconds = 0.0

    @torch.no_grad()
    def step(self):
        index = self.index.reshape(1)
        x = self.bank.feats(index)
        y = self.target.index_select(0, index).squeeze(0)
        mask = self.valid.index_select(0, index).squeeze(0)
        prediction = self.model.step(x, y, mask)
        # FP64 reductions of the actual FP32 pre-update forecasts. where, rather
        # than multiply-by-zero, gives invalid samples exactly zero contribution.
        p = torch.where(mask[None], prediction.double(), 0.0)
        target = torch.where(mask, y.double(), 0.0)
        residual = p - target[None]
        energy = target.square().sum().expand(len(self.model.configs))
        count = mask.sum().double().expand(len(self.model.configs))
        row = torch.stack((residual.square().sum(1), p.square().sum(1),
                           (p * target[None]).sum(1), energy, count), dim=-1)
        self.raw.index_copy_(0, (self.index - self.start).reshape(1), row.unsqueeze(0))
        self.index.add_(1)

    @torch.no_grad()
    def capture(self):
        """Compile once, capture 1/8/16 steps, verify and restore ALL mutable state.

        Warmup/capture never consumes real labels: each attempt restores clocks,
        evidence, Adam moments, weights, prediction and all metric storage. CUDA
        failures propagate, including a failed bitwise replay check; no fallback.
        """
        if self.stop - self.start < 16:
            raise ValueError("stream must fit the largest capture")
        initial = [tensor.clone() for tensor in self.mutable]

        def reset():
            for tensor, saved in zip(self.mutable, initial):
                tensor.copy_(saved)

        compiled = torch.compile(self.step, fullgraph=True, mode="max-autotune-no-cudagraphs")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        started = time.perf_counter()
        try:
            with torch.cuda.stream(stream):
                compiled()
                reset()
                compiled()
            stream.synchronize()
            reset()
            torch.cuda.synchronize()
            self.compile_seconds = time.perf_counter() - started
            started = time.perf_counter()
            for size in (1, 8, 16):
                for _ in range(size):
                    compiled()
                expected = [tensor.clone() for tensor in self.mutable]
                reset()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(size):
                        compiled()
                reset()
                graph.replay()
                torch.cuda.synchronize()
                for actual, reference in zip(self.mutable, expected):
                    # Byte equality also covers clocks and signed zeros.
                    if not torch.equal(actual.reshape(-1).view(torch.uint8),
                                       reference.reshape(-1).view(torch.uint8)):
                        raise RuntimeError(f"capture replay differs bitwise for block {size}")
                reset()
                graphs[size] = graph
                del expected
            self.capture_seconds = time.perf_counter() - started
            return graphs
        finally:
            stream.synchronize()
            reset()
            torch.cuda.synchronize()


def reduce_metrics(raw):
    """Per-config sums, preserving numerical failure instead of silently dropping rows."""
    if raw.ndim != 3 or raw.shape[2] != len(METRICS):
        raise ValueError("expected [bars, configs, five metrics]")
    sums = raw.sum(axis=0, dtype=np.float64)
    rows = []
    for column, values in enumerate(sums):
        row = dict(zip(METRICS, map(float, values)))
        finite = bool(np.isfinite(raw[:, column]).all() and np.isfinite(values).all())
        row["count"] = int(row["count"]) if math.isfinite(row["count"]) else None
        denominator = row["target_energy"]
        row.update(status="finite" if finite else "nonfinite", bars=len(raw),
                   mse=row["residual_sse"] / row["count"] if row["count"] else math.nan,
                   error_ratio=row["residual_sse"] / denominator if denominator > 0 else math.nan,
                   prediction_energy_ratio=row["prediction_energy"] / denominator if denominator > 0 else math.nan,
                   signed_cross_term_ratio=-2 * row["cross_sum"] / denominator if denominator > 0 else math.nan)
        rows.append(row)
    return rows


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


def phase_summary(raw, stream_start, consumed, selection_start, cut, stop, names):
    if len(raw) != consumed - stream_start or not stream_start <= consumed <= stop:
        raise ValueError("summary must contain consumed rows only")
    ranges = {"training": (stream_start, selection_start), "selection": (selection_start, cut),
              "forward": (cut, stop), "all_consumed": (stream_start, stop)}
    edges = np.linspace(cut, stop, 9, dtype=np.int64)
    ranges.update({f"forward_block_{i + 1}": (int(edges[i]), int(edges[i + 1])) for i in range(8)})
    result = {}
    for phase, (begin, end) in ranges.items():
        observed_end = max(begin, min(consumed, end))
        rows = raw[begin - stream_start:observed_end - stream_start] if consumed > begin else raw[:0]
        metrics = reduce_metrics(rows)
        result[phase] = {"planned_range": [begin, end], "observed_range": [begin, observed_end],
                         "bars": len(rows), "complete": consumed >= end,
                         "sample_count": metrics[0]["count"],
                         "metrics": [{"name": name, **row} for name, row in zip(names, metrics)]}
    return result


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
    torch.save({"version": 1, "consumed_until_exclusive": consumed,
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


def recover_consumed(runner):
    """Recover the actual device boundary after a failed host checkpoint step."""
    consumed = int(runner.index.cpu())
    if not runner.start <= consumed <= runner.stop:
        raise RuntimeError("failed runner index is outside the stream")
    if int(runner.model.steps.cpu()) != consumed - runner.start:
        raise RuntimeError("failed runner/model clocks do not describe a coherent checkpoint")
    return consumed, runner.raw[:consumed - runner.start].cpu().numpy().copy()


def main():
    args = tyro.cli(Args)
    validate_args(args)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("output directory must be empty; existing experiment artifacts are immutable")
    started = time.perf_counter()
    result = {"version": 1, "status": "initializing", "view": args.view,
              "args": asdict(args), "primary_family": PRIMARY, "locks": {}, "curves": [],
              "source_sha256": source_hashes(), "tensorboard_dir": str(root),
              "protocol": {
                  "normalization": "Unchanged Bank.mu uses masked targets from first 60% of the full panel, including the selection window. Shared fixed prefix normalization; NOT a claim of streaming-only first-60% label causality.",
                  "selection": "Independent family LR locks by masked sample-weighted prequential MSE on [floor(.4T),floor(.6T)); tie breaks at the first LR. All five locks persisted before label cut.",
                  "forward": "Primary family categorical_ce is fixed a priori. Continuous predict-before-update through T-2, no reset, no suffix LR/family selection. Every grid candidate retained, even nonfinite.",
                  "null": "Existing Bank PERM shuffles target and validity together with its fixed seed0 permutation; features unchanged. Inherit all real locks. A permutation baseline, not financial significance.",
                  "culling": "Suffix only, locked categorical_ce error_ratio min_delta .001; ProxyCull bar-index units (one observation is one cross-sectional bar, NOT 200 stocks), median3, EMA half-life8192 bars, warmup16384 bars, stride4096 bars, patience3. Pruned exit75; numerical failure exit1.",
                  "precision": "CUDA-only FP32 model, FP64 metric reductions, highest matmul precision, TF32 off, fullgraph compile, verified CUDA graphs1/8/16; no eager fallback.",
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
                      parameter_count_allocated=sum(t.numel() for t in (*model.weights, *model.biases)),
                      parameter_count_effective_by_config=[bank.F * args.width + args.width * args.width
                          + 2 * args.width + (1 if c.family.startswith("scalar_") else args.bins) * (args.width + 1)
                          for c in configs], maximum_bars=runner.stop - runner.start)
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
            if consumed != end or int(model.steps.cpu()) != consumed - runner.start:
                raise RuntimeError("runner/model clocks disagree with consumed bar boundary")
            chunk = runner.raw[previous - runner.start:consumed - runner.start].cpu().numpy()
            host_raw[previous - runner.start:consumed - runner.start] = chunk
            raw = host_raw[:consumed - runner.start]
            interval = reduce_metrics(chunk)
            candidate_state = torch.stack([
                torch.isfinite(tensor).reshape(len(configs), -1).all(1)
                for tensor in model.state_tensors()
                if tensor.ndim and tensor.shape[0] == len(configs)
            ]).all(0).cpu().numpy()
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

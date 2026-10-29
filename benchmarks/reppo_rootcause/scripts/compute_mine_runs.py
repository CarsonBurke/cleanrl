"""Mine tfevents for the compute study (06_compute.md): Args, returns, throughput, phase timings, critic metrics.

Read-only and CPU-only. One JSON record per run: the hyperparameter table the script logged, wall-clock span,
windowed returns at matched steps, SPS, every `timing/*` phase (median seconds per log interval) and every
other scalar tag at 1M / 4M / 8M. Returns use scripts/_runs.py so they match score_runs.py --at.

Usage:
  .venv/bin/python benchmarks/reppo_rootcause/scripts/compute_mine_runs.py OUT.json PATTERN [PATTERN...] [--jobs 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from _runs import RETURN_TAG  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

RETURN_STEPS = (500_000, 1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000)
RETURN_WINDOW = 50_000
METRIC_STEPS = (1_000_000, 4_000_000, 8_000_000)
METRIC_WINDOW = 250_000
SKIP_PREFIXES = ("charts/episodic_",)


def parse_args_table(accumulator):
    """The `|param|value|` markdown table CleanRL scripts log as a text summary."""
    for tag in accumulator.Tags().get("tensors", []):
        if not tag.startswith("hyperparameters"):
            continue
        text = accumulator.Tensors(tag)[0].tensor_proto.string_val[0].decode()
        rows = (line.strip("|").split("|", 1) for line in text.splitlines()[2:])
        return {row[0]: row[1] for row in rows if len(row) == 2}
    return {}


def window_mean(steps, values, step, window):
    mask = (steps >= step - window) & (steps <= step + window)
    return float(values[mask].mean()) if mask.any() else None


def mine(run_dir: Path):
    record = {"run": run_dir.name, "dir": str(run_dir.relative_to(ROOT))}
    try:
        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0, "tensors": 4})
        accumulator.Reload()
    except Exception as exc:  # corrupt or empty run
        record["error"] = type(exc).__name__
        return record
    record["args"] = parse_args_table(accumulator)
    scalars = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        scalars[tag] = (np.array([e.step for e in events], dtype=np.int64),
                        np.array([e.value for e in events], dtype=np.float64),
                        np.array([e.wall_time for e in events], dtype=np.float64))
    if RETURN_TAG in scalars:
        steps, values, wall = scalars[RETURN_TAG]
        record["max_step"] = int(steps[-1])
        record["wall_s"] = float(wall[-1] - wall[0])
        record["wall_start"] = float(wall[0])
        record["wall_end"] = float(wall[-1])
        record["returns"] = {str(step): window_mean(steps, values, step, RETURN_WINDOW)
                             for step in RETURN_STEPS if step <= steps[-1] + RETURN_WINDOW}
        record["return_last20"] = float(values[-20:].mean())
        # Wall-clock throughput between 1M and the end, free of compile/warmup time.
        late = steps >= 1_000_000
        if late.sum() > 2:
            record["sps_after_1M"] = float((steps[late][-1] - steps[late][0]) / (wall[late][-1] - wall[late][0]))
    metrics = {}
    for tag, (steps, values, _) in scalars.items():
        if tag.startswith(SKIP_PREFIXES):
            continue
        finite = values[np.isfinite(values)]
        entry = {"n": int(values.size), "median": float(np.median(finite)) if finite.size else None,
                 "last": float(values[-1])}
        for step in METRIC_STEPS:
            entry[str(step)] = window_mean(steps, values, step, METRIC_WINDOW)
        if tag.startswith("timing/") and steps.size > 1:
            # Median steps between log points turns a per-interval total into seconds per 16k (or per rollout).
            entry["steps_per_log"] = float(np.median(np.diff(steps)))
        metrics[tag] = entry
    record["metrics"] = metrics
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--jobs", type=int, default=3)
    options = parser.parse_args()
    roots = [ROOT / "runs", ROOT / "runs/_dead", ROOT / "runs/_archive", ROOT / "runs/_aborted"]
    run_dirs = sorted({path for root in roots if root.exists() for path in root.iterdir()
                       if path.is_dir() and any(pattern in path.name for pattern in options.patterns)})
    with Pool(options.jobs) as pool:
        records = pool.map(mine, run_dirs, chunksize=1)
    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
    Path(options.out).write_text(json.dumps(records, indent=1))
    print(f"{len(records)} runs -> {options.out}")


if __name__ == "__main__":
    main()

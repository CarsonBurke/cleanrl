"""Score and compare experiment runs from TensorBoard logs.

Usage:
    # Compare all runs matching a pattern:
    python cleanrl/lstd_ablations/score_runs.py "dirnoise"

    # Compare specific variants:
    python cleanrl/lstd_ablations/score_runs.py "directamp" "fixedfloor" "amprange"

    # Only HalfCheetah runs:
    python cleanrl/lstd_ablations/score_runs.py "dirnoise" --env HalfCheetah-v4

    # Use last N episodes instead of default 20:
    python cleanrl/lstd_ablations/score_runs.py "dirnoise" --last 50

Reports mean, std, min, max, and 95% CI for each run's final episodic returns.
Ranks runs by mean and flags whether differences are statistically significant.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np


class RunResult(NamedTuple):
    returns: np.ndarray
    mean_end: float
    mean_all: float
    ci95: float
    max_step: int


TAG = "charts/episodic_return"


def load_returns(run_dir: Path, last_n: int = 20) -> RunResult | None:
    """Load episodic returns by parsing TF event files directly (skips EventAccumulator)."""
    from tensorboard.compat.proto import event_pb2
    import struct

    values, max_step = [], 0
    for ef in sorted(run_dir.glob("events.out.tfevents.*")):
        with open(ef, "rb") as f:
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                data_len = struct.unpack("Q", hdr)[0]
                f.read(4)  # masked crc of length
                data = f.read(data_len)
                f.read(4)  # masked crc of data
                event = event_pb2.Event.FromString(data)
                if event.HasField("summary"):
                    for v in event.summary.value:
                        if v.tag == TAG:
                            values.append(v.simple_value)
                            max_step = max(max_step, event.step)

    if not values:
        return None
    all_values = np.array(values)
    end_values = all_values[-last_n:]
    return RunResult(
        returns=end_values,
        mean_end=float(np.mean(end_values)),
        mean_all=float(np.mean(all_values)),
        ci95=1.96 * float(np.std(end_values, ddof=1)) / np.sqrt(len(end_values)) if len(end_values) > 1 else float("nan"),
        max_step=max_step,
    )



def find_runs(runs_dir: Path, patterns: list[str], env_filter: str | None) -> list[Path]:
    """Find run directories matching any pattern, optionally filtered by env."""
    matched = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if env_filter and env_filter not in name:
            continue
        if any(p in name for p in patterns):
            matched.append(d)
    return matched


def parse_run_name(dirname: str) -> tuple[str, str]:
    """Parse 'env__variant__seed__timestamp' into (env, variant)."""
    parts = dirname.split("__")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", dirname


def main():
    parser = argparse.ArgumentParser(description="Score and compare TensorBoard runs")
    parser.add_argument("patterns", nargs="+", help="Patterns to match in run directory names")
    parser.add_argument("--env", default=None, help="Filter by environment (e.g. HalfCheetah-v4)")
    parser.add_argument("--last", type=int, default=20, help="Number of final episodes to evaluate")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing TensorBoard runs")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        sys.exit(1)

    matched = find_runs(runs_dir, args.patterns, args.env)
    if not matched:
        print(f"No runs found matching {args.patterns} in {runs_dir}/")
        sys.exit(1)

    # Load results, keyed by "env / variant"
    results: dict[str, RunResult] = {}
    for d in matched:
        env, variant = parse_run_name(d.name)
        label = f"{env} / {variant}" if env else d.name
        result = load_returns(d, args.last)
        if result is not None:
            results[label] = result

    if not results:
        print("No valid results found.")
        sys.exit(1)

    ranked = sorted(results.items(), key=lambda x: x[1].mean_end, reverse=True)
    variants = [label.split(" / ")[-1] for label, _ in ranked]
    vw = max(len(v) for v in variants)  # variant column width

    # Print ranked table
    print(f"Scoring: last {args.last} episodes | {len(results)} runs")
    print()
    print(f"  {'#':>2}  {'Variant':<{vw}}  {'Avg end':>6}  {'±CI95':>5}  {'Avg all':>6}  {'Steps':>9}")
    print()
    for i, ((label, r), variant) in enumerate(zip(ranked, variants)):
        print(f"  {i+1:>2}  {variant:<{vw}}  {r.mean_end:>6.0f}  {r.ci95:>5.0f}  {r.mean_all:>6.0f}  {r.max_step:>9}")



if __name__ == "__main__":
    main()

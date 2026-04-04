"""Score and compare experiment runs from TensorBoard logs.

Usage:
    python cleanrl/score_runs.py "dirnoise"
    python cleanrl/score_runs.py "directamp" "fixedfloor" "amprange"
    python cleanrl/score_runs.py "dirnoise" --env HalfCheetah-v4
    python cleanrl/score_runs.py "dirnoise" --last 50
    python cleanrl/score_runs.py "hlgauss" --runs-dir hl-gauss-ablations
    python cleanrl/score_runs.py "hlgauss" --runs-dir runs hl-gauss-ablations

Reports mean, std, 95% CI for each run's final episodic returns.
Ranks runs by mean. Groups by environment when multiple envs present.
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
    """Load episodic returns by parsing TF event files directly."""
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
                f.read(4)
                data = f.read(data_len)
                f.read(4)
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


def find_runs(runs_dirs: list[Path], patterns: list[str], env_filter: str | None) -> list[Path]:
    """Find run directories matching any pattern across multiple directories."""
    matched = []
    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            continue
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


def print_group(env: str, runs: list[tuple[str, RunResult]], last_n: int):
    """Print a ranked table for one environment group."""
    ranked = sorted(runs, key=lambda x: x[1].mean_end, reverse=True)
    variants = [label for label, _ in ranked]
    vw = max(len(v) for v in variants)

    print(f"  {env}  ({len(ranked)} runs, last {last_n} eps)")
    print(f"  {'#':>2}  {'Variant':<{vw}}  {'Mean':>7}  {'±CI95':>6}  {'Avg all':>7}  {'Steps':>9}")
    for i, (variant, r) in enumerate(ranked):
        print(f"  {i+1:>2}  {variant:<{vw}}  {r.mean_end:>7.1f}  {r.ci95:>6.1f}  {r.mean_all:>7.1f}  {r.max_step:>9}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Score and compare TensorBoard runs")
    parser.add_argument("patterns", nargs="+", help="Patterns to match in run directory names")
    parser.add_argument("--env", default=None, help="Filter by environment (e.g. HalfCheetah-v4)")
    parser.add_argument("--last", type=int, default=20, help="Number of final episodes to evaluate")
    parser.add_argument("--runs-dir", nargs="*", default=["runs"], help="Directories containing TensorBoard runs")
    args = parser.parse_args()

    runs_dirs = [Path(d) for d in args.runs_dir]
    matched = find_runs(runs_dirs, args.patterns, args.env)
    if not matched:
        dirs_str = ", ".join(str(d) for d in runs_dirs)
        print(f"No runs found matching {args.patterns} in {dirs_str}")
        sys.exit(1)

    # Load results grouped by env
    by_env: dict[str, list[tuple[str, RunResult]]] = {}
    for d in matched:
        env, variant = parse_run_name(d.name)
        result = load_returns(d, args.last)
        if result is not None:
            by_env.setdefault(env, []).append((variant, result))

    if not by_env:
        print("No valid results found.")
        sys.exit(1)

    total = sum(len(v) for v in by_env.values())
    print(f"\n  {total} runs across {len(by_env)} env(s)\n")

    for env in sorted(by_env.keys()):
        print_group(env, by_env[env], args.last)


if __name__ == "__main__":
    main()

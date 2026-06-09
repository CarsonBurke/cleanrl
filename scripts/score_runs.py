"""Score and compare experiment runs from TensorBoard logs.

Usage:
    python scripts/score_runs.py "dirnoise"
    python scripts/score_runs.py "directamp" "fixedfloor" "amprange"
    python scripts/score_runs.py "dirnoise" --env HalfCheetah-v4
    python scripts/score_runs.py "dirnoise" --last 50
    python scripts/score_runs.py "dirnoise" --before 1780625004
    python scripts/score_runs.py "dirnoise" --before 2026-06-04T18:00:00
    python scripts/score_runs.py "hlgauss" --runs-dir hl-gauss-ablations
    python scripts/score_runs.py "hlgauss" --runs-dir runs hl-gauss-ablations
    python scripts/score_runs.py "dsrg_v5" --at 500k,1M,2M       # matched-step returns
    python scripts/score_runs.py "dsrg_v5" --metrics losses/clipfrac,losses/explained_variance

Reports mean, std, 95% CI for each run's final episodic returns.
Ranks runs by mean. Groups by environment when multiple envs present.

With --at, additionally prints windowed-mean episodic_return at each given
global step (so variants are compared at matched steps, not just at the end).
With --metrics, appends the latest value of extra scalar tags as columns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Shared helpers (run-dir globbing, scalar loading, step parsing, window-mean).
from _runs import (
    RETURN_TAG,
    RunScalars,
    find_runs,
    fmt_step,
    parse_cutoff,
    parse_run_name,
    parse_steps,
)

TAG = RETURN_TAG


class RunResult(NamedTuple):
    returns: np.ndarray
    mean_end: float
    mean_all: float
    ci95: float
    max_step: int
    scalars: RunScalars  # underlying accessor, for --at / --metrics columns


def load_returns(run_dir: Path, last_n: int = 20) -> RunResult | None:
    """Load episodic returns and stats for a run dir via EventAccumulator."""
    sc = RunScalars(run_dir)
    _, values = sc.series(TAG)
    if not values.size:
        return None
    all_values = np.asarray(values)
    end_values = all_values[-last_n:]
    return RunResult(
        returns=end_values,
        mean_end=float(np.mean(end_values)),
        mean_all=float(np.mean(all_values)),
        ci95=1.96 * float(np.std(end_values, ddof=1)) / np.sqrt(len(end_values)) if len(end_values) > 1 else float("nan"),
        max_step=sc.max_step(TAG),
        scalars=sc,
    )


def _fmt_metric(v: float | None) -> str:
    """Compact formatting for appended metric/at columns."""
    if v is None:
        return "--"
    a = abs(v)
    if a != 0 and (a >= 100000 or a < 1e-3):
        return f"{v:.2e}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.4g}"


def print_group(
    env: str,
    runs: list[tuple[str, RunResult]],
    last_n: int,
    at_steps: list[int],
    metrics: list[str],
    window: int,
):
    """Print a ranked table for one environment group.

    Base columns (rank, variant, mean, CI, avg-all, steps) are unchanged for
    backward compatibility. Optional matched-step return columns (--at) and
    extra latest-value metric columns (--metrics) are appended on the right.
    """
    ranked = sorted(runs, key=lambda x: x[1].mean_end, reverse=True)
    variants = [label for label, _ in ranked]
    vw = max(len(v) for v in variants)

    at_labels = [f"@{fmt_step(s)}" for s in at_steps]
    metric_labels = [m.split("/", 1)[-1] for m in metrics]

    # Precompute appended cells so columns can be width-sized.
    at_cells = {
        (label, s): _fmt_metric(r.scalars.window_mean(TAG, s, window))
        for label, r in ranked
        for s in at_steps
    }
    metric_cells = {
        (label, m): _fmt_metric(r.scalars.latest(m))
        for label, r in ranked
        for m in metrics
    }
    at_w = [max([len(lbl)] + [len(at_cells[(v, s)]) for v in variants]) for s, lbl in zip(at_steps, at_labels)]
    m_w = [max([len(lbl)] + [len(metric_cells[(v, m)]) for v in variants]) for m, lbl in zip(metrics, metric_labels)]

    print(f"  {env}  ({len(ranked)} runs, last {last_n} eps)")
    header = f"  {'#':>2}  {'Variant':<{vw}}  {'Mean':>7}  {'±CI95':>6}  {'Avg all':>7}  {'Steps':>9}"
    header += "".join(f"  {lbl:>{w}}" for lbl, w in zip(at_labels, at_w))
    header += "".join(f"  {lbl:>{w}}" for lbl, w in zip(metric_labels, m_w))
    print(header)
    for i, (variant, r) in enumerate(ranked):
        row = f"  {i+1:>2}  {variant:<{vw}}  {r.mean_end:>7.1f}  {r.ci95:>6.1f}  {r.mean_all:>7.1f}  {r.max_step:>9}"
        row += "".join(f"  {at_cells[(variant, s)]:>{w}}" for s, w in zip(at_steps, at_w))
        row += "".join(f"  {metric_cells[(variant, m)]:>{w}}" for m, w in zip(metrics, m_w))
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Score and compare TensorBoard runs")
    parser.add_argument("patterns", nargs="+", help="Patterns to match in run directory names")
    parser.add_argument("--env", default=None, help="Filter by environment (e.g. HalfCheetah-v4)")
    parser.add_argument("--last", type=int, default=20, help="Number of final episodes to evaluate")
    parser.add_argument(
        "--before",
        type=parse_cutoff,
        default=None,
        help=(
            "Only include runs whose trailing directory timestamp is before this "
            "cutoff. Accepts epoch seconds or ISO date/datetime."
        ),
    )
    parser.add_argument("--runs-dir", nargs="*", default=["runs"], help="Directories containing TensorBoard runs")
    parser.add_argument(
        "--at",
        default=None,
        help="Comma-separated steps (e.g. 500k,1M,2M) to report windowed-mean episodic_return at matched steps",
    )
    parser.add_argument(
        "--at-window",
        type=int,
        default=50_000,
        help="±window (steps) used for the --at windowed mean",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated extra scalar tags to append as latest-value columns",
    )
    args = parser.parse_args()

    at_steps = parse_steps(args.at) if args.at else []
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else []

    runs_dirs = [Path(d) for d in args.runs_dir]
    matched = find_runs(runs_dirs, args.patterns, args.env, args.before)
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
        print_group(env, by_env[env], args.last, at_steps, metrics, args.at_window)


if __name__ == "__main__":
    main()

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
    python scripts/score_runs.py "comparison_v3" --jobs 4933 4935 4936 4937
    python scripts/score_runs.py "dsrg_v5" --at 500k,1M,2M       # matched-step returns
    python scripts/score_runs.py "dsrg_v5" --metrics losses/clipfrac,losses/explained_variance

Reports mean, std, 95% CI for each run's final episodic returns.
Ranks runs by mean. Groups by environment when multiple envs present.

With --at, additionally prints windowed-mean episodic_return at each given
global step (so variants are compared at matched steps, not just at the end).
With --metrics, appends extra scalar tags AT A MATCHED STEP (not their latest
value -- see below).

READ THIS BEFORE COMPARING TWO RUNS OF DIFFERENT LENGTHS.
  Mean / +-CI95 / Avg all are END-OF-RUN statistics. A run stopped at 2M and a
  run finished at 8M are at different points on their own learning curves, so
  those columns do not compare them -- they compare "wherever each happened to
  stop". Only the --at columns are matched-step. When run lengths differ
  materially this script now prints a warning saying exactly that.

  --metrics is matched-step for the same reason. It reports each tag at the last
  --at step (or, with no --at, at the shortest run's final step) and puts that
  step in the column header. Asking for losses/approx_kl across a 2.4M run and a
  finished 8M run used to silently contrast a mid-training value with a
  converged one, which reads as a large effect and is an artifact. Pass
  --metrics-latest to get the old last-value behaviour back, explicitly.

  Steps beyond a run's data are NEVER extrapolated: they print `--`. A value
  prefixed `~` means the run ended inside the averaging window, so the mean
  covers less data than the others.
"""
from __future__ import annotations

import argparse
import json
import subprocess
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


def print_job_states(job_ids: list[int]):
    """Report queue outcomes independently of any matching TensorBoard scores.

    Query each distinct job once. A missing daemon, unknown job or malformed
    response is an error, rather than an excuse to infer its state from logs.
    """
    jobs = []
    for job_id in dict.fromkeys(job_ids):
        command = ["mlq", "show", str(job_id), "--json"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(f"Could not query mlq job {job_id}: {detail}") from error
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Could not query mlq job {job_id}: {error}") from error
        try:
            job = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Could not query mlq job {job_id}: invalid JSON response") from error
        if (not isinstance(job, dict) or job.get("id") != job_id
                or not isinstance(job.get("state"), str) or not job["state"]):
            raise RuntimeError(f"Could not query mlq job {job_id}: response lacks the requested job ID/state")
        if job.get("stateReason") is not None and not isinstance(job["stateReason"], str):
            raise RuntimeError(f"Could not query mlq job {job_id}: invalid stateReason in response")
        jobs.append(job)

    print("\n  ML queue states (independent of TensorBoard scores):")
    for job in jobs:
        name = f" [{job['name']}]" if job.get("name") else ""
        reason = f" — {job['stateReason']}" if job.get("stateReason") else ""
        print(f"  job {job['id']}{name}: {job['state']}{reason}")
    unsuccessful = [str(job["id"]) for job in jobs if job["state"] != "succeeded"]
    if unsuccessful:
        print(f"  ! Jobs not reported as succeeded: {', '.join(unsuccessful)}. Available scores may be partial.")
    print("  Scores do not establish job completion or successful training.\n")


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


THIN_WINDOW = 10


def _cell(r: RunResult, tag: str, step: int, window: int) -> str:
    """Matched-step cell, with out-of-range made VISIBLE rather than extrapolated.

    RunScalars.window_mean falls back to the nearest sample when the requested
    step lies past the end of a run, which formats an extrapolation exactly like
    a measurement -- the single easiest way to misread this table. Here a step
    the run never reached prints `--`, and a step it only partially covers is
    prefixed `~`.

    A cell averaging fewer than THIN_WINDOW samples is suffixed `!`: a value
    backed by three episodes must not render identically to one backed by a
    hundred, or a trend gets read off sampling noise.
    """
    if step - window > r.max_step:
        return "--"
    stats = r.scalars.window_stats(tag, step, window)
    if stats is None:
        return "--"
    mean, _ci, n = stats
    text = _fmt_metric(mean)
    if n < THIN_WINDOW:
        text += "!"
    return f"~{text}" if step > r.max_step else text


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
    metrics_latest: bool,
):
    """Print a ranked table for one environment group.

    Base columns (rank, variant, mean, CI, avg-all, steps) are unchanged for
    backward compatibility. Optional matched-step return columns (--at) and
    extra latest-value metric columns (--metrics) are appended on the right.
    """
    ranked = sorted(runs, key=lambda x: x[1].mean_end, reverse=True)
    variants = [label for label, _ in ranked]
    vw = max(len(v) for v in variants)

    # Metrics are read at a MATCHED step: the last --at step, else the shortest
    # run's end (the latest point every run in the group actually reached).
    steps_seen = [r.max_step for _, r in ranked]
    metric_step = at_steps[-1] if at_steps else min(steps_seen)

    at_labels = [f"@{fmt_step(s)}" for s in at_steps]
    metric_labels = [
        m.split("/", 1)[-1] + ("(last)" if metrics_latest else f"@{fmt_step(metric_step)}")
        for m in metrics
    ]

    # Precompute appended cells so columns can be width-sized.
    at_cells = {
        (i, s): _cell(r, TAG, s, window)
        for i, (_, r) in enumerate(ranked)
        for s in at_steps
    }
    metric_cells = {
        (i, m): (
            _fmt_metric(r.scalars.latest(m))
            if metrics_latest
            else _cell(r, m, metric_step, window)
        )
        for i, (_, r) in enumerate(ranked)
        for m in metrics
    }
    rows = range(len(ranked))
    at_w = [max([len(lbl)] + [len(at_cells[(i, s)]) for i in rows]) for s, lbl in zip(at_steps, at_labels)]
    m_w = [max([len(lbl)] + [len(metric_cells[(i, m)]) for i in rows]) for m, lbl in zip(metrics, metric_labels)]

    print(f"  {env}  ({len(ranked)} runs, last {last_n} eps)")
    # The trap this warning exists for: end-of-run columns silently compare runs
    # at different points on their own learning curves.
    dupes = {v for v in variants if variants.count(v) > 1}
    if dupes:
        print(
            f"  ! {len(dupes)} variant name(s) appear more than once "
            f"({', '.join(sorted(dupes))}) -- rows are distinct RUNS, often a cancelled "
            f"job and its relaunch, which may not be the same code."
        )
    if steps_seen and max(steps_seen) > 1.1 * min(steps_seen):
        print(
            f"  ! run lengths differ ({fmt_step(min(steps_seen))}..{fmt_step(max(steps_seen))}):"
            f" Mean / +-CI95 / Avg all are END-OF-RUN and NOT comparable here."
        )
        if not at_steps:
            print("    add --at 500k,1M,2M for a matched-step comparison.")
        if metrics and metrics_latest:
            print("    --metrics-latest is also end-of-run; drop it for matched-step values.")
    header = f"  {'#':>2}  {'Variant':<{vw}}  {'Mean':>7}  {'±CI95':>6}  {'Avg all':>7}  {'Steps':>9}"
    header += "".join(f"  {lbl:>{w}}" for lbl, w in zip(at_labels, at_w))
    header += "".join(f"  {lbl:>{w}}" for lbl, w in zip(metric_labels, m_w))
    print(header)
    for i, (variant, r) in enumerate(ranked):
        row = f"  {i+1:>2}  {variant:<{vw}}  {r.mean_end:>7.1f}  {r.ci95:>6.1f}  {r.mean_all:>7.1f}  {r.max_step:>9}"
        row += "".join(f"  {at_cells[(i, s)]:>{w}}" for s, w in zip(at_steps, at_w))
        row += "".join(f"  {metric_cells[(i, m)]:>{w}}" for m, w in zip(metrics, m_w))
        print(row)
    _shown = {**at_cells, **metric_cells}.values()
    if any(c.startswith(("~", "-")) or c.endswith("!") for c in _shown):
        print("    (-- = run never reached this step; ~ = run ended inside the averaging window;"
              f" ! = fewer than {THIN_WINDOW} episodes in the window)")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score and compare TensorBoard runs")
    parser.add_argument("patterns", nargs="+", help="Patterns to match in run directory names")
    parser.add_argument("--env", default=None, help="Filter by environment (e.g. HalfCheetah-v4)")
    parser.add_argument("--last", type=int, default=20, help="Number of final episodes to evaluate")
    parser.add_argument(
        "--jobs", nargs="+", type=int, default=None,
        help="Also report these mlq job states/reasons; fail if a job state cannot be queried",
    )
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
        help=(
            "Comma-separated extra scalar tags, reported AT A MATCHED STEP (the last "
            "--at step, else the shortest run's end). The step appears in the header."
        ),
    )
    parser.add_argument(
        "--metrics-latest",
        action="store_true",
        help=(
            "Report --metrics at each run's LAST logged step instead of a matched one. "
            "Only meaningful when every run is the same length; across runs of "
            "different lengths this contrasts mid-training with converged values."
        ),
    )
    args = parser.parse_args(argv)
    if args.jobs is not None:
        if any(job_id <= 0 for job_id in args.jobs):
            parser.error("--jobs requires positive integer job IDs")
        try:
            print_job_states(args.jobs)
        except RuntimeError as error:
            parser.exit(status=1, message=f"{error}\n")

    at_steps = parse_steps(args.at) if args.at else []
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else []

    runs_dirs = [Path(d) for d in args.runs_dir]
    matched = find_runs(runs_dirs, args.patterns, args.env, args.before)
    for pattern in dict.fromkeys(args.patterns):
        if not any(pattern in run_dir.name for run_dir in matched):
            print(f"  ! No run directories matched requested pattern {pattern!r} within the selected directories/filters.")
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
        else:
            print(f"  ! Run {str(d)!r} has no {TAG} samples; omitted from the score table.")

    if not by_env:
        print("No valid results found.")
        sys.exit(1)

    total = sum(len(v) for v in by_env.values())
    print(f"\n  {total} runs across {len(by_env)} env(s)\n")

    for env in sorted(by_env.keys()):
        print_group(
            env, by_env[env], args.last, at_steps, metrics, args.at_window,
            args.metrics_latest,
        )


if __name__ == "__main__":
    main()

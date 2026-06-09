"""Watch a run's live progress, or block until it reaches a step / stalls.

Usage:
    python scripts/watch_run.py <pattern> [--env ENV]                 # one-shot status
    python scripts/watch_run.py <pattern> --until 2M [--env ENV]      # block until >=2M or stalled
    python scripts/watch_run.py <pattern> --until 2M --poll 60 --stall 600

Liveness is decided from the tfevents file mtime (see `_runs.last_active`), NOT
from tracking PIDs, so it is robust to the launching shell having exited and to
transient `pgrep` misses. Intended to be launched as a *tracked* background task
(a blocking call, no `nohup`/`&`): with --until it blocks until the target step
is reached or the run stalls, then prints a final status line and exits 0, so
the harness fires its completion notification at the right moment.

Without --until it prints a single status line and exits immediately.

When a pattern matches several run dirs (seeds/timestamps), the most recently
active one is chosen.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from _runs import (
    RETURN_TAG,
    RunScalars,
    find_runs,
    fmt_step,
    last_active,
    parse_step,
)

SPS_TAG = "charts/SPS"


def resolve_run(patterns: list[str], env: str | None, runs_dirs: list[Path]) -> Path | None:
    """Pick the most recently active run dir matching the pattern(s)."""
    matched = find_runs(runs_dirs, patterns, env)
    if not matched:
        return None
    return max(matched, key=last_active)


def status_line(run_dir: Path, last_n: int = 20) -> tuple[int, float, str]:
    """Return (max_step, age_secs, formatted-status-line) for a run dir.

    A fresh RunScalars is built each call so newly-flushed events are seen.
    """
    sc = RunScalars(run_dir)
    step = sc.max_step(RETURN_TAG)
    _, vals = sc.series(RETURN_TAG)
    ret = float(np.mean(vals[-last_n:])) if vals.size else float("nan")
    sps = sc.latest(SPS_TAG)
    age = time.time() - last_active(run_dir)
    sps_str = f"{sps:.0f}" if sps is not None else "--"
    line = (
        f"{sc.variant}  step={step} ({fmt_step(step)})  "
        f"ret(last{last_n})={ret:7.1f}  sps={sps_str}  age={age:.0f}s"
    )
    return step, age, line


def main():
    p = argparse.ArgumentParser(description="Watch a run or block until it hits a step / stalls")
    p.add_argument("patterns", nargs="+", help="Substring pattern(s) matching a run dir name")
    p.add_argument("--env", default=None, help="Filter by environment (e.g. HalfCheetah-v4)")
    p.add_argument("--runs-dir", nargs="*", default=["runs"], help="Directories containing runs")
    p.add_argument("--until", default=None, help="Block until run reaches this step (e.g. 2M)")
    p.add_argument("--poll", type=int, default=60, help="Seconds between polls when blocking")
    p.add_argument("--stall", type=int, default=600, help="Seconds of no tfevent activity => stalled/dead")
    p.add_argument("--last", type=int, default=20, help="Episodes for the trailing-mean return")
    args = p.parse_args()

    runs_dirs = [Path(d) for d in args.runs_dir]
    run_dir = resolve_run(args.patterns, args.env, runs_dirs)

    if args.until is None:
        if run_dir is None:
            print(f"No run found matching {args.patterns}" + (f" (env {args.env})" if args.env else ""))
            sys.exit(1)
        _, age, line = status_line(run_dir, args.last)
        state = "RUNNING" if age <= args.stall else "STALLED"
        print(f"{line}  [{state}]")
        return

    target = parse_step(args.until)
    # A just-launched run may not have written its tfevents dir yet, so the first
    # resolve can miss. Poll for it to appear (bounded by --stall) instead of bailing.
    waited = 0
    while run_dir is None:
        if waited > args.stall:
            print(f"No run found matching {args.patterns}" + (f" (env {args.env})" if args.env else "") + f" after {waited}s")
            sys.exit(1)
        time.sleep(args.poll)
        waited += args.poll
        run_dir = resolve_run(args.patterns, args.env, runs_dirs)

    while True:
        step, age, line = status_line(run_dir, args.last)
        if step >= target:
            print(f"REACHED {fmt_step(target)}: {line}")
            return
        if age > args.stall:
            print(f"STALLED (no events for {age:.0f}s, target {fmt_step(target)} not reached): {line}")
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()

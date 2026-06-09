"""tb.py — quick TensorBoard scalar queries for CleanRL runs.

Subcommands:
  live    Dashboard: liveness + current return + trajectory milestones. No
          patterns = auto-discover every run active in the last --since seconds.
  snap    Snapshot a preset of scalar tags (latest, optionally first) per run.
  traj    Compare runs at matched global steps (windowed-mean of a tag).
  status  Per-run progress: max step, latest SPS, recent episodic_return.

Examples:
  python scripts/tb.py live                       # all currently-active runs, one shot
  python scripts/tb.py live --watch 30            # refresh every 30s
  python scripts/tb.py live simba dsrg_v18        # only matching variants
  python scripts/tb.py snap dsrg_v5_pathway
  python scripts/tb.py traj dsrg_v5_pathway dsrg_v2_hc8m --at 500k,1M,2M
  python scripts/tb.py status dsrg_v5_pathway
  python scripts/tb.py snap dsrg_v5_pathway --tags charts/SPS,losses/clipfrac --first

The `live` milestone columns show `·` for steps a run has not yet reached (unlike
`traj`, which repeats the final value past a run's progress).

Run dirs follow runs/{env_id}__{exp_name}__{seed}__{timestamp}/.
By default only HalfCheetah-v4 runs are shown; pass --env '' for all envs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _runs import (
    RETURN_TAG,
    RunScalars,
    find_runs,
    fmt_step,
    last_active,
    latest_per_variant,
    load_run,
    parse_cutoff,
    parse_steps,
)

# Default preset for `snap`: training-health + routing + circuit signals.
# Older runs lack many of these; missing tags render as "--".
DEFAULT_SNAP_TAGS = [
    "charts/episodic_return",
    "charts/SPS",
    "losses/clipfrac",
    "losses/approx_kl",
    "losses/explained_variance",
    "losses/entropy",
    "losses/value_loss",
    "losses/policy_loss",
    "losses/actor_grad_norm",
    "losses/critic_grad_norm",
    "losses/route_grad_norm",
    "debug/route_active_frac",
    "debug/route_gate_entropy",
    "debug/sigma_mean",
    "circuit/conditionality",
    "circuit/tick_dynamism",
    "circuit/diversity",
    "circuit/reuse_cv",
    "circuit/live_frac",
]


def _short_tag(tag: str) -> str:
    """Drop the group prefix for column headers (charts/SPS -> SPS)."""
    return tag.split("/", 1)[-1]


def _fmt_val(v: float | None) -> str:
    if v is None:
        return "--"
    a = abs(v)
    if a != 0 and (a >= 100000 or a < 1e-3):
        return f"{v:.2e}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.4g}"


def _resolve_runs(args) -> list[RunScalars]:
    """Discover, dedupe-to-latest, and load runs for the given patterns."""
    runs_dirs = [Path(d) for d in args.runs_dir]
    before = parse_cutoff(args.before) if args.before else None
    env = args.env or None
    matched = find_runs(runs_dirs, args.patterns, env, before)
    if not matched:
        dirs_str = ", ".join(str(d) for d in runs_dirs)
        print(f"No runs found matching {args.patterns} in {dirs_str}")
        sys.exit(1)
    matched = latest_per_variant(matched)
    return [load_run(d) for d in matched]


# --------------------------------------------------------------------------- #
# snap
# --------------------------------------------------------------------------- #
def cmd_snap(args):
    runs = _resolve_runs(args)
    tags = args.tags.split(",") if args.tags else DEFAULT_SNAP_TAGS

    vw = max(len(r.variant) for r in runs)
    headers = [_short_tag(t) for t in tags]
    # Per-column width: at least the header, accommodate formatted values.
    cells = {}
    for r in runs:
        for t in tags:
            cells[(r.variant, t, "latest")] = _fmt_val(r.latest(t))
            if args.first:
                cells[(r.variant, t, "first")] = _fmt_val(r.first(t))
    widths = []
    for t, h in zip(tags, headers):
        w = len(h)
        for r in runs:
            w = max(w, len(cells[(r.variant, t, "latest")]))
            if args.first:
                w = max(w, len(cells[(r.variant, t, "first")]))
        widths.append(w)

    print(f"\n  snapshot — {len(runs)} run(s), latest values" + (" (first/latest)" if args.first else ""))
    hdr = f"  {'Variant':<{vw}}  " + "  ".join(f"{h:>{w}}" for h, w in zip(headers, widths))
    print(hdr)
    for r in runs:
        row = f"  {r.variant:<{vw}}  " + "  ".join(
            f"{cells[(r.variant, t, 'latest')]:>{w}}" for t, w in zip(tags, widths)
        )
        print(row)
        if args.first:
            frow = f"  {'  (first)':<{vw}}  " + "  ".join(
                f"{cells[(r.variant, t, 'first')]:>{w}}" for t, w in zip(tags, widths)
            )
            print(frow)
    print()


# --------------------------------------------------------------------------- #
# traj
# --------------------------------------------------------------------------- #
def cmd_traj(args):
    runs = _resolve_runs(args)
    steps = parse_steps(args.at)
    tag = args.tag

    vw = max(len(r.variant) for r in runs)
    col_labels = [fmt_step(s) for s in steps]
    # Compute all cells first to size columns.
    grid = {}
    for r in runs:
        for s in steps:
            grid[(r.variant, s)] = r.window_mean(tag, s, args.window)
    widths = []
    for s, lbl in zip(steps, col_labels):
        w = len(lbl)
        for r in runs:
            w = max(w, len(_fmt_val(grid[(r.variant, s)])))
        widths.append(w)

    print(f"\n  trajectory — {tag}  (windowed mean ±{fmt_step(args.window)})")
    hdr = f"  {'Variant':<{vw}}  " + "  ".join(f"{lbl:>{w}}" for lbl, w in zip(col_labels, widths))
    print(hdr)
    for r in runs:
        row = f"  {r.variant:<{vw}}  " + "  ".join(
            f"{_fmt_val(grid[(r.variant, s)]):>{w}}" for s, w in zip(steps, widths)
        )
        print(row)
    print()


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def cmd_status(args):
    runs = _resolve_runs(args)
    vw = max(len(r.variant) for r in runs)
    n_recent = args.last

    print(f"\n  status — {len(runs)} run(s)")
    print(f"  {'Variant':<{vw}}  {'MaxStep':>9}  {'SPS':>6}  {'Return(last)':>12}  Recent returns")
    for r in runs:
        max_step = r.max_step(RETURN_TAG)
        sps = r.latest("charts/SPS")
        sps_s = f"{sps:.0f}" if sps is not None else "--"
        _, vals = r.series(RETURN_TAG)
        last = f"{vals[-1]:.1f}" if vals.size else "--"
        recent = " ".join(f"{v:.0f}" for v in vals[-n_recent:]) if vals.size else "--"
        print(f"  {r.variant:<{vw}}  {max_step:>9}  {sps_s:>6}  {last:>12}  {recent}")
    print()


# --------------------------------------------------------------------------- #
# live — one-shot dashboard: liveness + current return + trajectory milestones
# --------------------------------------------------------------------------- #
def _live_runs(args) -> list[RunScalars]:
    """Resolve runs for `live`. With patterns: latest-per-variant matches. Without:
    auto-discover every run whose newest event file was touched within --since."""
    runs_dirs = [Path(d) for d in args.runs_dir]
    env = args.env or None
    patterns = args.patterns or [""]            # "" substring matches every run dir
    matched = find_runs(runs_dirs, patterns, env, None)
    if not args.patterns:                        # auto-discover: keep only recent runs
        import time
        now = time.time()
        matched = [d for d in matched if now - last_active(d) <= args.since]
    matched = latest_per_variant(matched)
    return [load_run(d) for d in matched]


def _render_live(args):
    import time
    runs = _live_runs(args)
    if not runs:
        print(f"\n  live — no runs active within {fmt_step(args.since)}s "
              f"(env={args.env or 'any'}). Pass patterns or raise --since.\n")
        return
    steps = parse_steps(args.at)
    now = time.time()
    # Most-recently-active first.
    runs.sort(key=lambda r: last_active(r.run_dir), reverse=True)
    vw = max(len(r.variant) for r in runs)
    mcols = [fmt_step(s) for s in steps]
    mw = max(7, *(len(c) for c in mcols))

    hdr = (f"  {'':1}  {'Variant':<{vw}}  {'Step':>7}  {'SPS':>5}  {'Return':>8}   "
           + "  ".join(f"{c:>{mw}}" for c in mcols))
    print(f"\n  live — {len(runs)} run(s)   ● training  ○ idle   "
          f"(milestones: windowed mean ±{fmt_step(args.window)}, · = not yet reached)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in runs:
        age = now - last_active(r.run_dir)
        live = age <= args.active
        mark = "●" if live else "○"
        max_step = r.max_step(RETURN_TAG)
        sps = r.latest("charts/SPS")
        sps_s = f"{sps:.0f}" if sps is not None else "--"
        ret = r.window_mean(RETURN_TAG, max_step, args.window)
        ret_s = f"{ret:.0f}" if ret is not None else "--"
        cells = []
        for s in steps:
            # Only show a milestone the run has actually reached — avoids the
            # `traj` trap of repeating the final value for unreached steps.
            v = r.window_mean(RETURN_TAG, s, args.window) if s <= max_step else None
            cells.append(f"{v:.0f}" if v is not None else "·")
        if live:
            age_s = ""
        elif age < 90:
            age_s = f"  ({int(age)}s ago)"
        elif age < 5400:
            age_s = f"  ({age / 60:.0f}m ago)"
        else:
            age_s = f"  ({age / 3600:.1f}h ago)"
        print(f"  {mark}  {r.variant:<{vw}}  {fmt_step(max_step):>7}  {sps_s:>5}  "
              f"{ret_s:>8}   " + "  ".join(f"{c:>{mw}}" for c in cells) + age_s)
    print()


def cmd_live(args):
    if args.watch:
        import os
        import time
        try:
            while True:
                os.system("clear")
                print(f"  (refreshing every {args.watch}s — Ctrl-C to stop)")
                _render_live(args)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        _render_live(args)


def main():
    p = argparse.ArgumentParser(description="Quick TensorBoard scalar queries for CleanRL runs")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("patterns", nargs="+", help="Substring patterns to match run dir names")
        sp.add_argument("--env", default="HalfCheetah-v4", help="Env filter substring; pass '' for all envs")
        sp.add_argument("--runs-dir", nargs="*", default=["runs"], help="Directories containing runs")
        sp.add_argument("--before", default=None, help="Only runs before this epoch/ISO timestamp")

    sp = sub.add_parser("snap", help="Snapshot latest (and optionally first) scalar values")
    common(sp)
    sp.add_argument("--tags", default=None, help="Comma-separated tags (default: health+routing+circuit preset)")
    sp.add_argument("--first", action="store_true", help="Also show first value of each tag")
    sp.set_defaults(func=cmd_snap)

    sp = sub.add_parser("traj", help="Compare runs at matched global steps")
    common(sp)
    sp.add_argument("--at", default="500k,1M,2M,4M,8M", help="Comma-separated steps (e.g. 500k,1M,2M)")
    sp.add_argument("--tag", default=RETURN_TAG, help=f"Scalar tag to compare (default {RETURN_TAG})")
    sp.add_argument("--window", type=int, default=50_000, help="±window (steps) for the mean")
    sp.set_defaults(func=cmd_traj)

    sp = sub.add_parser("status", help="Per-run progress (max step, SPS, recent returns)")
    common(sp)
    sp.add_argument("--last", type=int, default=5, help="How many recent returns to show")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("live", help="Dashboard: liveness + current return + trajectory milestones. "
                                     "No patterns = auto-discover active runs.")
    sp.add_argument("patterns", nargs="*", help="Optional substring patterns (default: all recently-active runs)")
    sp.add_argument("--env", default="HalfCheetah-v4", help="Env filter substring; pass '' for all envs")
    sp.add_argument("--runs-dir", nargs="*", default=["runs"], help="Directories containing runs")
    sp.add_argument("--at", default="500k,1M,2M,4M,8M", help="Milestone steps to show (default 500k,1M,2M,4M,8M)")
    sp.add_argument("--window", type=int, default=50_000, help="±window (steps) for the windowed mean")
    sp.add_argument("--since", type=int, default=1800, help="Auto-discover runs active within this many seconds (default 1800)")
    sp.add_argument("--active", type=int, default=150, help="Mark ● training if event file touched within this many seconds (default 150)")
    sp.add_argument("--watch", type=int, default=0, help="Refresh every N seconds (0 = one-shot)")
    sp.set_defaults(func=cmd_live)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Shared helpers for TensorBoard run analysis tools (score_runs.py, tb.py).

Centralizes the bits that were previously hand-written over and over:
  - run-dir discovery / globbing from a name pattern
  - loading scalar series via EventAccumulator (cached per dir)
  - human-readable step parsing ("500k", "1M", "1000000")
  - windowed-mean of a scalar tag around a target step

Run dirs follow the CleanRL convention:
    runs/{env_id}__{exp_name}__{seed}__{timestamp}/
each containing a tfevents file with scalar tags.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Default episodic-return tag, shared everywhere.
RETURN_TAG = "charts/episodic_return"


# --------------------------------------------------------------------------- #
# Run-dir discovery
# --------------------------------------------------------------------------- #
def run_timestamp(run_dir: Path) -> float | None:
    """Return trailing run timestamp from 'env__variant__seed__timestamp'."""
    try:
        return float(run_dir.name.rsplit("__", 1)[1])
    except (IndexError, ValueError):
        return None


def last_active(run_dir: Path) -> float:
    """Wall-clock mtime (epoch secs) of the newest tfevents file in a run dir.

    A run still being written has a recent mtime; this is how `live` decides
    liveness without tracking PIDs. Returns 0.0 if no event files are found.

    Tolerates files that disappear between the listing and the stat: archiving a run
    (`mv runs/<x> runs/_dead/`) while another process walks `runs/` is routine here, and the
    raised FileNotFoundError propagated all the way out of autocull's supervisor loop.
    """
    files = list(run_dir.glob("events.out.tfevents.*")) or [
        f for f in run_dir.rglob("*") if f.is_file()
    ]
    mtimes = []
    for f in files:
        try:
            mtimes.append(f.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes, default=0.0)


def parse_run_name(dirname: str) -> tuple[str, str]:
    """Parse 'env__variant__seed__timestamp' into (env, variant)."""
    parts = dirname.split("__")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", dirname


def find_runs(
    runs_dirs: list[Path],
    patterns: list[str],
    env_filter: str | None = None,
    before: float | None = None,
) -> list[Path]:
    """Find run directories matching any substring pattern across dirs.

    A pattern may match MULTIPLE run dirs (different timestamps/seeds); all
    matches are returned sorted by directory name (callers may dedupe).
    """
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
            if before is not None:
                started = run_timestamp(d)
                if started is None or started >= before:
                    continue
            if any(p in name for p in patterns):
                matched.append(d)
    return matched


def latest_per_variant(run_dirs: list[Path]) -> list[Path]:
    """Collapse multiple matching dirs to the latest run per (env, variant).

    When a pattern matches several timestamps for the same variant, keep the
    one with the largest trailing timestamp (most recent run).
    """
    best: dict[tuple[str, str], tuple[float, Path]] = {}
    for d in run_dirs:
        key = parse_run_name(d.name)
        ts = run_timestamp(d) or 0.0
        if key not in best or ts > best[key][0]:
            best[key] = (ts, d)
    # Preserve a stable, name-sorted order.
    return [d for _, d in sorted((v for v in best.values()), key=lambda x: x[1].name)]


# --------------------------------------------------------------------------- #
# Scalar loading (cached EventAccumulator)
# --------------------------------------------------------------------------- #
class RunScalars:
    """Lazily-loaded scalar accessor for a single run dir.

    Wraps an EventAccumulator and exposes (steps, values) numpy arrays per tag,
    caching results so repeated lookups (e.g. many tags) are cheap.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.env, self.variant = parse_run_name(run_dir.name)
        self._ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        self._ea.Reload()
        self._tags = set(self._ea.Tags().get("scalars", []))
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def tags(self) -> set[str]:
        return self._tags

    def series(self, tag: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (steps, values) arrays for a tag, empty arrays if absent."""
        if tag in self._cache:
            return self._cache[tag]
        if tag not in self._tags:
            out = (np.array([], dtype=np.int64), np.array([], dtype=np.float64))
            self._cache[tag] = out
            return out
        events = self._ea.Scalars(tag)
        steps = np.fromiter((e.step for e in events), dtype=np.int64, count=len(events))
        vals = np.fromiter((e.value for e in events), dtype=np.float64, count=len(events))
        out = (steps, vals)
        self._cache[tag] = out
        return out

    def max_step(self, tag: str = RETURN_TAG) -> int:
        steps, _ = self.series(tag)
        return int(steps[-1]) if steps.size else 0

    def latest(self, tag: str) -> float | None:
        """Latest (last-by-step) value of a tag, or None if absent/empty."""
        steps, vals = self.series(tag)
        return float(vals[-1]) if vals.size else None

    def first(self, tag: str) -> float | None:
        """First value of a tag, or None if absent/empty."""
        _, vals = self.series(tag)
        return float(vals[0]) if vals.size else None

    def value_near(self, tag: str, step: int) -> float | None:
        """Latest value of a tag at-or-before `step`; nearest if all later."""
        steps, vals = self.series(tag)
        if not steps.size:
            return None
        idx = np.searchsorted(steps, step, side="right") - 1
        if idx < 0:
            idx = 0  # all recorded steps are after the target; use the first.
        return float(vals[idx])

    def window_mean(self, tag: str, step: int, window: int) -> float | None:
        """Mean of a tag over [step-window, step+window].

        Falls back to the single value nearest `step` if no samples land in the
        window (e.g. sparse logging or a step beyond the run's data).
        """
        steps, vals = self.series(tag)
        if not steps.size:
            return None
        lo, hi = step - window, step + window
        mask = (steps >= lo) & (steps <= hi)
        if mask.any():
            return float(np.mean(vals[mask]))
        return self.value_near(tag, step)

    def window_stats(self, tag: str, step: int, window: int) -> tuple[float, float, int] | None:
        """(mean, ci95, n) of a tag over [step-window, step+window].

        Exposes the SAMPLE COUNT so a cell backed by three episodes cannot look
        identical to one backed by a hundred. ``n == 0`` means the value came
        from the nearest-sample fallback and is a point read, not an average.
        """
        steps, vals = self.series(tag)
        if not steps.size:
            return None
        mask = (steps >= step - window) & (steps <= step + window)
        n = int(mask.sum())
        if n == 0:
            near = self.value_near(tag, step)
            return None if near is None else (near, float("nan"), 0)
        sel = vals[mask]
        ci = (1.96 * float(np.std(sel, ddof=1)) / np.sqrt(n)) if n > 1 else float("nan")
        return float(np.mean(sel)), ci, n


def load_run(run_dir: Path) -> RunScalars:
    return RunScalars(run_dir)


# --------------------------------------------------------------------------- #
# Step-string parsing
# --------------------------------------------------------------------------- #
_SUFFIX = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def parse_step(text: str) -> int:
    """Parse a human-readable step like '500k', '1M', '2.5m', '1000000'."""
    s = str(text).strip().lower().replace("_", "").replace(",", "")
    if not s:
        raise ValueError("empty step string")
    if s[-1] in _SUFFIX:
        return int(round(float(s[:-1]) * _SUFFIX[s[-1]]))
    return int(round(float(s)))


def parse_steps(text: str) -> list[int]:
    """Parse a comma-separated list of step strings into ints."""
    return [parse_step(p) for p in text.split(",") if p.strip()]


def fmt_step(step: int) -> str:
    """Compact human label for a step count (e.g. 1500000 -> '1.5M')."""
    for suffix, scale in (("B", 1_000_000_000), ("M", 1_000_000), ("k", 1_000)):
        if abs(step) >= scale:
            v = step / scale
            return f"{v:.0f}{suffix}" if v == int(v) else f"{v:.1f}{suffix}"
    return str(step)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def parse_cutoff(value: str) -> float:
    """Parse a --before cutoff as epoch seconds or ISO local date/datetime."""
    try:
        return float(value)
    except ValueError:
        pass
    text = value.strip()
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            "cutoff must be an epoch timestamp or ISO date/datetime "
            "(for example 1780625004 or 2026-06-04T18:00:00)"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
    return parsed.timestamp()

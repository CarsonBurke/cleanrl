"""Dump matched-step windowed returns (and selected diagnostics) for runs matching patterns.

Read-only over tfevents. Uses scripts/_runs.py so numbers match score_runs.py --at (window +-50k).

Usage: .venv/bin/python benchmarks/reppo_rootcause/scripts/dump_runs.py OUT.tsv PATTERN [PATTERN...] [--tags a,b]
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from _runs import RETURN_TAG, RunScalars  # noqa: E402

STEPS = [500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 6_000_000, 8_000_000, 12_000_000, 16_000_000]
WINDOW = 50_000
EXTRA: list[str] = []


def one(run_dir: Path) -> str:
    try:
        rs = RunScalars(run_dir, {RETURN_TAG, *EXTRA})
        steps, vals = rs.series(RETURN_TAG)
    except Exception as exc:  # corrupt / empty run
        return f"{run_dir.name}\tERR {type(exc).__name__}"
    if steps.size == 0:
        return f"{run_dir.name}\tEMPTY"
    cols = [run_dir.name, str(int(steps[-1])), str(int(steps[0]))]
    for s in STEPS:
        # window_mean falls back to the nearest sample beyond the data; never extrapolate.
        m = rs.window_mean(RETURN_TAG, s, WINDOW) if steps[0] - WINDOW <= s <= steps[-1] + WINDOW else None
        cols.append("" if m is None else f"{m:.0f}")
    cols.append(f"{np.mean(vals[-20:]):.0f}")
    for tag in EXTRA:
        if tag not in rs.tags:
            cols += [""] * 4
            continue
        for s in (1_000_000, 2_000_000, 4_000_000, 8_000_000):
            m = rs.window_mean(tag, s, 250_000) if steps[0] <= s <= steps[-1] + WINDOW else None
            cols.append("" if m is None else f"{m:.4g}")
    return "\t".join(cols)


def main() -> None:
    global EXTRA
    args = sys.argv[1:]
    if "--tags" in args:
        i = args.index("--tags")
        EXTRA = args[i + 1].split(",")
        del args[i : i + 2]
    out, patterns = Path(args[0]), args[1:]
    runs = sorted(
        d for d in (ROOT / "runs").iterdir()
        if d.is_dir() and d.name.startswith("HalfCheetah-v4__") and any(p in d.name for p in patterns)
        and "snapshot" not in d.name
    )
    header = ["run", "last_step", "first_step"] + [f"@{s // 1000}k" for s in STEPS] + ["last20"]
    for tag in EXTRA:
        header += [f"{tag.split('/')[-1]}@{s}" for s in ("1M", "2M", "4M", "8M")]
    with Pool(3, initializer=_init, initargs=(EXTRA,)) as pool:
        rows = pool.map(one, runs, chunksize=4)
    out.write_text("\t".join(header) + "\n" + "\n".join(rows) + "\n")
    print(f"{len(rows)} runs -> {out}")


def _init(extra: list[str]) -> None:
    global EXTRA
    EXTRA = extra


if __name__ == "__main__":
    main()

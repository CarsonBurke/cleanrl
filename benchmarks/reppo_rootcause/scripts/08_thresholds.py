"""Red-team 08: steps-to-threshold and return gained per 16,384 env steps in fixed step windows.

Smooths episodic return with a +-50k-step window on a 25k grid, then reports the first step at which the smoothed
curve reaches each threshold (sample-efficiency ratio is read across runs at a matched RETURN, not a matched step),
and the windowed gain per 16,384 steps, plus exploration scale (entropy / std tags where logged).
"""
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

THR = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
EDGES = [0.25e6, 0.5e6, 1e6, 2e6, 3e6, 4e6, 6e6, 7.9e6]
AT = [0.5e6, 1e6, 2e6, 4e6, 6e6, 7.5e6]
EXTRA = ["losses/entropy", "policy/entropy", "policy/std", "losses/approx_kl", "policy/kl", "losses/clipfrac"]


def series(acc, tag):
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev], float), np.array([e.value for e in ev], float)


def smooth(s, v, grid, half=50_000):
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        m = (s >= g - half) & (s <= g + half)
        if m.any():
            out[i] = np.nanmean(v[m])
    return out


rows_thr, rows_gain, rows_extra = [], [], []
for name in sys.argv[1:]:
    dirs = sorted(Path("runs").glob(f"HalfCheetah-v4__{name}__1__*"))
    if not dirs:
        print(f"NOT FOUND {name}"); continue
    d = max(dirs, key=lambda p: sum(f.stat().st_size for f in p.glob("events*")))
    acc = EventAccumulator(str(d), size_guidance={"scalars": 0}); acc.Reload()
    s, v = series(acc, "charts/episodic_return")
    grid = np.arange(50_000, min(s.max(), 8.0e6) + 1, 25_000)
    sm = smooth(s, v, grid)
    thr = []
    for t in THR:
        idx = np.where(sm >= t)[0]
        thr.append(f"{grid[idx[0]]/1e6:6.2f}" if len(idx) else "     -")
    rows_thr.append(name[:44].ljust(45) + " ".join(thr))
    at = lambda x: sm[np.argmin(np.abs(grid - x))] if grid[-1] >= x - 60_000 else np.nan
    gains = [(at(b) - at(a)) / ((b - a) / 16384) for a, b in zip(EDGES[:-1], EDGES[1:])]
    rows_gain.append(name[:44].ljust(45) + " ".join(f"{g:7.1f}" for g in gains))
    have = set(acc.Tags()["scalars"])
    for t in EXTRA:
        if t in have:
            es, ev = series(acc, t)
            vals = [np.nanmean(ev[(es >= a - 50_000) & (es <= a + 50_000)]) if ((es >= a - 50_000) & (es <= a + 50_000)).any() else np.nan for a in AT]
            rows_extra.append(name[:34].ljust(35) + t[:18].ljust(19) + " ".join(f"{x:8.4f}" for x in vals))

print("STEPS (M) TO REACH RETURN".ljust(45) + " ".join(f"{t/1000:5.0f}k" for t in THR)); print("\n".join(rows_thr))
print("\nRETURN GAINED PER 16,384 STEPS in window".ljust(46) + " ".join(f"{a/1e6:.2f}-{b/1e6:.1f}"[:7].rjust(7) for a, b in zip(EDGES[:-1], EDGES[1:]))); print("\n".join(rows_gain))
print("\nEXPLORATION / STEP SCALE at".ljust(55) + " ".join(f"{a/1e6:7.1f}M" for a in AT)); print("\n".join(rows_extra))

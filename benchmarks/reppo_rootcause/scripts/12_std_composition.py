"""12: is the logged per-iteration `policy/std` jitter a batch-composition effect of crashed episodes?

`policy/std` is the mean pre-tanh std over the rollout's rows. On the oracle chunks the policy's std on inverted rows is
0.86-1.08 against 0.47-0.50 on upright rows, so each crashed episode (~500 inverted rows of 16,384 = 3%) should add
~0.015 to the batch mean. Regress the detrended logged std on the number of crashed episodes (return < 0.85 x running
median) that finish during the rollout it was computed on and during the next one (episodes straddle two rollouts).
"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
common = import_module("09_common")
jitter = import_module("12_sigma_jitter")
runs = {run.exp: run for run in common.load_runs()}
ITER = 16384
for exp in ("reppo_noisefork_v60_base", "reppo_latentroll_v27_h8_50M"):
    run = runs[exp]
    steps, std = run.series("policy/std")
    keep = steps >= 3.5e6
    steps, std = steps[keep], std[keep].astype(np.float64)
    entropy = run.series("policy/entropy")[1][keep].astype(np.float64)
    ep_steps, ep = run.series(common.RETURN)
    crashes = np.zeros((len(steps), 3))
    for i, step in enumerate(steps):
        recent = ep[(ep_steps > step - 20 * ITER) & (ep_steps <= step + 20 * ITER)]
        threshold = 0.85 * np.median(recent)
        for j, (lo, hi) in enumerate(((step - ITER, step), (step, step + ITER), (step + ITER, step + 2 * ITER))):
            crashes[i, j] = np.sum(ep[(ep_steps > lo) & (ep_steps <= hi)] < threshold)
    print(f"\n{exp}: crashed episodes per rollout mean {crashes[:, 0].mean():.2f} (sd {crashes[:, 0].std():.2f}) of ~16")
    for name, series in (("std", std), ("entropy", entropy)):
        y = jitter.detrend(series)
        design = np.column_stack((np.ones(len(y)), crashes[:, :2] - crashes[:, :2].mean(0)))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        r2 = 1 - (y - design @ coef).var() / y.var()
        future = np.corrcoef(y, crashes[:, 2])[0, 1]
        print(f"  {name:8s}: detrended sd {y.std():.4f}; per crashed episode finishing in the same rollout {coef[1]:+.4f}, in the next {coef[2]:+.4f}; "
              f"R^2 {r2:.2f}; residual sd {(y - design @ coef).std():.4f}; corr with crashes in episodes wholly AFTER the log point {future:+.3f}")

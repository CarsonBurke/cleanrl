"""12: is the +-3-4% per-iteration sigma jitter a real +-85 swing in training return?

Per iteration after 3.5M: policy/std, its first difference, and the mean / median of the episodes that finished in
the following rollouts. Detrend both with a 33-iteration running median and regress return on std at several lags.
Predicted slope if noise cost = 1400 (sigma / 0.5)^2: d return / d sigma = -5600 per unit sigma at sigma 0.5.
"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
common = import_module("09_common")
runs = {run.exp: run for run in common.load_runs()}
ITER = 16384


def detrend(values, half=16):
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = values[i] - np.median(values[max(0, i - half):i + half + 1])
    return out


for exp in sys.argv[1:]:
    run = runs[exp]
    std_steps, std = run.series("policy/std")
    keep = std_steps >= 3.5e6
    std_steps, std = std_steps[keep], std[keep].astype(np.float64)
    ent = run.series("policy/entropy")[1][keep].astype(np.float64)
    gaps = np.diff(std_steps)
    print(f"\n{exp}: {len(std)} std points after 3.5M, median step gap {np.median(gaps):.0f}")
    diff = np.diff(std)
    lag1 = np.corrcoef(diff[:-1], diff[1:])[0, 1]
    print(f"  std mean {std.mean():.3f}; diff sd {diff.std():.4f} ({100 * diff.std() / std.mean():.1f}% of mean); "
          f"median |diff| {np.median(np.abs(diff)):.4f} ({100 * np.median(np.abs(diff)) / std.mean():.1f}%); lag-1 of diffs {lag1:+.2f}")
    print(f"  detrended std sd {detrend(std).std():.4f} ({100 * detrend(std).std() / std.mean():.1f}%); entropy diff sd {np.diff(ent).std():.3f}")
    ep_steps, ep = run.series(common.RETURN)
    # the policy written at log step s (after update) generates the rollout ending at s + ITER
    mean_ret, med_ret, n_ep = [], [], []
    for step in std_steps:
        mask = (ep_steps > step) & (ep_steps <= step + ITER)
        n_ep.append(mask.sum())
        mean_ret.append(ep[mask].mean() if mask.any() else np.nan)
        med_ret.append(np.median(ep[mask]) if mask.any() else np.nan)
    mean_ret, med_ret = np.array(mean_ret), np.array(med_ret)
    ok = np.isfinite(mean_ret)
    print(f"  episodes per iteration: median {np.median(n_ep):.0f}; per-iteration return sd after detrending: "
          f"mean {np.nanstd(detrend(mean_ret[ok])):.0f}, median {np.nanstd(detrend(med_ret[ok])):.0f}; "
          f"iteration-to-iteration diff sd: mean {np.nanstd(np.diff(mean_ret[ok])):.0f}, median {np.nanstd(np.diff(med_ret[ok])):.0f}")
    x = detrend(std[ok])
    for name, series in (("mean", mean_ret[ok]), ("median", med_ret[ok])):
        y = detrend(series)
        for lag in (-1, 0, 1):
            xs, ys = (x[:len(x) - lag], y[lag:]) if lag >= 0 else (x[-lag:], y[:lag])
            slope = (xs * ys).sum() / (xs * xs).sum()
            corr = np.corrcoef(xs, ys)[0, 1]
            se = np.sqrt((1 - corr ** 2) / (len(xs) - 2)) * ys.std() / xs.std()
            print(f"    {name:6s} return on std, lag {lag:+d}: slope {slope:8.0f} +- {se:5.0f} per unit sigma, corr {corr:+.3f}, "
                  f"implied swing per 1 sd of std jitter {slope * x.std():+6.0f}")

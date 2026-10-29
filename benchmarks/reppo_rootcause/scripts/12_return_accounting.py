"""12: how much of the deterministic-minus-stochastic return gap, and of the slope change, is crashed episodes?

Per window of the training episode-return record: mean, median, crash share (episode < 0.85 x window median),
tumble cost = mean(non-crashed) - mean(all), ordinary noise cost = deterministic eval - mean(non-crashed).
"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
common = import_module("09_common")
runs = {run.exp: run for run in common.load_runs()}
exp, width = sys.argv[1], float(sys.argv[2]) * 1e6
edges = [float(x) * 1e6 for x in sys.argv[3:]]
run = runs[exp]
steps, returns = run.series(common.RETURN)
det_steps, det = run.series("eval/deterministic_return")
print(exp)
print("  M    n_ep   mean  median  crash%  tumble_cost  det   det-mean  noise_cost(non-crash)  std")
for centre in edges:
    mask = (steps >= centre - width / 2) & (steps < centre + width / 2)
    window = returns[mask]
    if len(window) < 20:
        continue
    median = np.median(window)
    crashed = window < 0.85 * median
    clean_mean = window[~crashed].mean()
    det_mask = (det_steps >= centre - width / 2) & (det_steps < centre + width / 2)
    det_mean = det[det_mask].mean() if det_mask.any() else float("nan")
    std = run.window("policy/std", centre - width / 2, centre + width / 2)
    print(f" {centre / 1e6:5.2f} {len(window):5d} {window.mean():7.0f} {median:7.0f} {100 * crashed.mean():6.1f} "
          f"{clean_mean - window.mean():9.0f} {det_mean:9.0f} {det_mean - window.mean():7.0f} {det_mean - clean_mean:9.0f} {std:12.3f}")

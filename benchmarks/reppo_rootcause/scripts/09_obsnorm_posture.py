"""Posture from saved observation statistics, without simulation. VectorObsNorm keeps a cumulative per-env running
mean / variance of raw observations; obs[1] is the root pitch (rad). For each snapshot: cumulative mean and sd of
pitch (averaged over envs) and the implied share of all steps so far spent inverted, assuming a two-point mixture
(upright pitch u, inverted pitch 3.2):  f = (mean - u) / (3.2 - u), reported for u = 0.1 and u = 0.4, with the
mixture's predicted sd as a consistency check. Consecutive snapshots of one run give the interval mean pitch.
Output: stdout (tee to data/09/obsnorm_posture.txt)."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]


def main():
    files = sorted(REPO.glob("runs/HalfCheetah-v4__reppo*/snapshot_*.pt")) + \
        sorted(REPO.glob("runs/HalfCheetah-v4__reppo*/checkpoint_*.pt")) + sorted(REPO.glob("runs/HalfCheetah-v4__reppo*/final.pt"))
    by_run = defaultdict(list)
    for path in files:
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if "obs_means" not in snapshot:
            continue
        means, variances = np.asarray(snapshot["obs_means"]), np.asarray(snapshot["obs_variances"])
        second = (variances[:, 1] + means[:, 1] ** 2).mean()
        by_run[path.parent.name.split("__")[1].replace("reppo_", "")].append(
            (int(snapshot["global_step"]), float(means[:, 1].mean()), float(np.sqrt(variances[:, 1]).mean()), float(second),
             float(means[:, 8].mean())))
    print(f"{'run':34s} {'step M':>7s} {'pitch mean':>10s} {'sd':>5s} {'f(u=.1)':>8s} {'f(u=.4)':>8s} {'mix sd':>7s} {'interval pitch':>15s} {'interval rms':>13s} {'cum velocity':>13s}")
    for run, rows in by_run.items():
        previous = None
        for step, mean, sd, second, velocity in sorted(rows):
            f_low, f_high = (mean - 0.1) / 3.1, max(0.0, (mean - 0.4) / 2.8)
            mixture_sd = 3.1 * np.sqrt(max(f_low * (1 - f_low), 0.0))
            interval = rms = float("nan")
            if previous and step > previous[0]:
                interval = (mean * step - previous[1] * previous[0]) / (step - previous[0])
                rms = np.sqrt(max((second * step - previous[2] * previous[0]) / (step - previous[0]), 0.0))
            print(f"{run[:33]:34s} {step / 1e6:7.2f} {mean:10.2f} {sd:5.2f} {f_low:8.2f} {f_high:8.2f} {mixture_sd:7.2f} {interval:15.2f} {rms:13.2f} {velocity:13.2f}")
            previous = (step, mean, second)


if __name__ == "__main__":
    main()

"""Shared loaders for the 09 flip / regime-change analysis. Read-only on data/03_runs (scalar dumps of every
runs/HalfCheetah-v4__reppo_* tfevents run, written by 03_dump_reppo_runs.py)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DUMPS = ROOT / "data/03_runs"
OUT = ROOT / "data/09"
RETURN = "charts/episodic_return"


class Run:
    def __init__(self, path):
        self.path = path
        self.name = path.stem
        self.exp = self.name.split("__")[1]
        self.stamp = int(self.name.split("__")[-1])
        self._data = np.load(path, allow_pickle=True)
        self.tags = sorted(key[2:] for key in self._data.files if key.startswith("s|"))
        self.hyper = {}
        for line in str(self._data["hyper"]).splitlines()[2:]:
            parts = line.strip("|").split("|")
            if len(parts) >= 2:
                self.hyper[parts[0]] = "|".join(parts[1:])

    def series(self, tag):
        if tag not in self.tags:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        return self._data[f"s|{tag}"], self._data[f"v|{tag}"]

    @property
    def last_step(self):
        steps, _ = self.series(RETURN)
        return int(steps[-1]) if len(steps) else 0

    def window(self, tag, lo, hi, reduce=np.mean):
        steps, values = self.series(tag)
        mask = (steps >= lo) & (steps < hi) & np.isfinite(values)
        return float(reduce(values[mask])) if mask.any() else float("nan")

    def at(self, tag, step, half=50_000):
        return self.window(tag, step - half, step + half)


def load_runs():
    runs = [Run(path) for path in sorted(DUMPS.glob("*.npz"))]
    return [run for run in runs if len(run.series(RETURN)[0])]


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)

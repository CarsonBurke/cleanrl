"""B (premise check). Per-sample advantage signal share of the trainer's lambda-return target, recomputed.

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/07_B2_signal_share.py

signal = true within-state variance of Q over the 8 fresh actions (cross product of rollout halves: noise-free);
noise  = Var over oracle states of (target - Q_true(s, a0)), the oracle's own rollout SE removed.
"""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("common07", Path(__file__).resolve().parent / "07_common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
rng = np.random.default_rng(0)
for name in ("v23_4M", "v29h8_8M", "v29h8_12M"):
    d = common.load(name, u_samples=2)
    ret = d.returns[300]
    fresh = ret[:, d.layout["fresh"][0]:d.layout["fresh"][1]]
    a0 = ret[:, d.layout["a0_probes"][0]]
    k = fresh.shape[1]
    ca, cb = fresh[..., ::2].mean(-1), fresh[..., 1::2].mean(-1)
    ca, cb = ca - ca.mean(1, keepdims=True), cb - cb.mean(1, keepdims=True)
    err = d.target[d.index] - a0.mean(-1)
    se2 = a0.var(-1, ddof=1) / a0.shape[-1]

    def share(idx):
        signal = (ca[idx] * cb[idx]).sum(1).mean() / (k - 1)
        noise = err[idx].var(ddof=1) - se2[idx].mean()
        return signal, noise, signal / (signal + noise)
    point = share(np.arange(len(err)))
    draws = np.array([share(rng.integers(0, len(err), len(err))) for _ in range(2000)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    up = d.upright[d.index]
    pu = share(np.flatnonzero(up))
    print(f"{name}: signal var {point[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}], target noise var {point[1]:.2f} [{lo[1]:.2f}, {hi[1]:.2f}], "
          f"share {point[2]:.3f} [{lo[2]:.3f}, {hi[2]:.3f}] | upright only ({int(up.sum())}): share {pu[2]:.3f}")

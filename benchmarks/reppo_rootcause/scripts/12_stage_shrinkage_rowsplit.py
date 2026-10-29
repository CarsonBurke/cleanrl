"""12: (a) per-stage test of the 'optimal shrinkage' reading; (b) upright / inverted row split of the on-policy chunk.

(a) Within-state advantages over the 8 fresh policy actions at the 512 oracle states. Truth = mean soft return over
    32 CRN rollouts (H = 100), split into two independent halves so its noise drops out of every product.
    For a posterior mean: slope(truth on critic) = 1 and amplitude ratio = correlation. slope < 1 = over-dispersed
    (noise fitted into the action dependence), slope > 1 = under-fit. Bootstrap over states.
(b) Rows of the 16 x 1024 chunk tagged by root pitch (state[:, 2]): inverted when cos(pitch) < 0.
"""
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
rng = np.random.default_rng(0)
FRESH = slice(50, 58)


def stats(critic, half_a, half_b, index):
    c, a, b = critic[index], half_a[index], half_b[index]
    var_c = (c * c).mean()
    var_t = (a * b).mean()
    cov = 0.5 * ((c * a).mean() + (c * b).mean())
    return cov / var_c, var_c, var_t, cov


def report(label, critic, half_a, half_b, keep):
    index = np.flatnonzero(keep)
    slope, var_c, var_t, cov = stats(critic, half_a, half_b, index)
    draws = np.array([stats(critic, half_a, half_b, rng.choice(index, len(index)))[:1] for _ in range(1000)])[:, 0]
    corr = cov / np.sqrt(var_c * var_t) if var_t > 0 else float("nan")
    amp = np.sqrt(var_c / var_t) if var_t > 0 else float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"    {label:26s} n {len(index):3d}  corr {corr:+.3f}  amp {amp:5.2f}  amp/corr {amp / corr:6.2f}  "
          f"slope truth-on-critic {slope:+.2f} [{lo:+.2f}, {hi:+.2f}]  P(slope<1) {np.mean(draws < 1):.2f}  sd_true {np.sqrt(max(var_t, 0)):.3f}")


for tag in sys.argv[1:]:
    d = np.load(DATA / f"03_chunk_{tag}.npz")
    print(f"\n===== {tag}")
    returns = d["ret_100"][:, FRESH].astype(np.float64)
    centre = lambda x: x - x.mean(1, keepdims=True)
    half_a, half_b = centre(returns[..., :16].mean(-1)), centre(returns[..., 16:].mean(-1))
    critic = centre(d["s_critic_q"][:, FRESH])
    energy = (critic ** 2).sum(1)
    top = np.sort(energy)[::-1]
    print(f"  (a) top-1 / top-5 states hold {top[0] / top.sum():.2f} / {top[:5].sum() / top.sum():.2f} of critic within-state energy")
    everything = np.ones(512, bool)
    report("all states", critic, half_a, half_b, everything)
    report("drop top 1% critic energy", critic, half_a, half_b, energy < np.quantile(energy, 0.99))
    report("drop top 5% critic energy", critic, half_a, half_b, energy < np.quantile(energy, 0.95))
    report("upright states only", critic, half_a, half_b, d["upright"])
    report("upright, drop top 5%", critic, half_a, half_b, d["upright"] & (energy < np.quantile(energy, 0.95)))
    # per-state normalised (each state unit critic energy): what a per-state trust region consumes
    scale = np.sqrt(energy / 8)[:, None] + 1e-12
    per_state = ((critic / scale) * 0.5 * (half_a + half_b)).mean(1)
    print(f"    per-state gain along unit critic contrast: mean {per_state.mean():+.4f} +- {per_state.std() / np.sqrt(512):.4f}, "
          f"median {np.median(per_state):+.4f}, frac > 0 {np.mean(per_state > 0):.2f}")

    pitch = d["state"][:, 2]
    inverted = np.cos(pitch) < 0
    wound = np.abs(pitch) > np.pi / 2
    error = (d["critic_q"] - d["target"]).astype(np.float64) ** 2
    std = d["std"].mean(1)
    lanes = d["row"] % 16 if "row" in d.files else None
    print(f"  (b) rows inverted (cos pitch < 0) {inverted.mean():.4f}; |pitch| > pi/2 (incl. wound-up upright) {wound.mean():.4f}; "
          f"oracle states not upright {1 - d['upright'].mean():.3f}")
    for name, mask in (("upright", ~wound), ("inverted or wound", wound)):
        if mask.sum() == 0:
            continue
        print(f"    {name:18s} rows {mask.mean():.4f}  pre-tanh std {std[mask].mean():.3f}  mean|tanh mu| {np.abs(np.tanh(d['mean'][mask])).mean():.3f}  "
              f"q_mse {error[mask].mean():8.3f}  share of q_mse {error[mask].sum() / error.sum():.3f}  target mean {d['target'][mask].mean():7.2f}  critic mean {d['critic_q'][mask].mean():7.2f}")
    print(f"    all rows: pre-tanh std {std.mean():.3f}; upright-rows-only std {std[~wound].mean():.3f}; q_mse {error.mean():.3f}; "
          f"top 1% of rows hold {np.sort(error)[::-1][:164].sum() / error.sum():.2f} of q_mse")

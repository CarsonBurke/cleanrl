"""What is the trainer's logged advantage_corr_heldout worth? On each oracle chunk (held-out for the snapshot critic):
  logged-style  = corr(Q(s,a0) - b, G_lambda - b)          (b = critic E_pi Q; what the trainer logs)
  truth         = corr(Q(s,a0) - b, A_true(a0))             (A_true(a0) = Q_true(a0) - mean_k Q_true(a_k))
  ceiling       = corr(A_true(a0), G_lambda - b)            (what a PERFECT critic advantage would log)
A_true(a0) is noisy (32 rollouts, 8-action baseline); corr with it is disattenuated by its split-half reliability."""
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
RNG = np.random.default_rng(0)


def stats(c, g, a_m):
    a = a_m.mean(1)
    rel = np.corrcoef(a_m[:, ::2].mean(1), a_m[:, 1::2].mean(1))[0, 1]
    rel = 2 * rel / (1 + rel)  # Spearman-Brown to the full 32 rollouts (rollout noise only; the 8-action baseline error is left in)
    r = np.corrcoef
    return np.array([r(c, g)[0, 1], r(c, a)[0, 1] / np.sqrt(rel), r(a, g)[0, 1] / np.sqrt(rel), rel, c.std() / g.std(), a.std() * np.sqrt(rel) / g.std()])


for name in ["v23_4M", "v23_12M", "v23_30M", "v29h8_8M", "v29h8_12M"]:
    d = np.load(DATA / f"03_chunk_{name}.npz")
    idx, b = d["s_index"], d["s_critic_baseline"]
    ret = d["ret_300"].astype(np.float64)
    a_m = ret[:, 25] - ret[:, 50:58].mean(1)
    c, g = d["s_critic_q"][:, 25] - b, d["target"][idx] - b
    for label, keep in (("all", np.ones(len(b), bool)), ("cruising", d["cruising"])):
        point = stats(c[keep], g[keep], a_m[keep])
        n = keep.sum()
        draws = np.array([stats(c[keep][i], g[keep][i], a_m[keep][i]) for i in RNG.integers(0, n, (1000, n))])
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        cell = lambda i: f"{point[i]:+.3f} [{lo[i]:+.3f},{hi[i]:+.3f}]"  # noqa: E731
        print(f"{name:10s} {label:8s} lam={float(d['lam'])}: logged-style {cell(0)} | critic vs truth {cell(1)} | ceiling (perfect critic) {cell(2)} "
              f"| A_true(a0) reliability {point[3]:.2f} | std ratio critic/G {point[4]:.3f}, true/G {point[5]:.3f}")

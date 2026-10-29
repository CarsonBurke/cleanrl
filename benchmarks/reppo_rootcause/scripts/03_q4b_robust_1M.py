"""Heavy-tail robustness of the v60 @1M critic-oracle numbers, and the horizon structure of the true advantage."""
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
z = np.load(REPO / "runs/reppo_critic_oracle/v60_base_1015424.npz")
RNG = np.random.default_rng(0)
q = z["q"][0].astype(np.float64)
critic, residual, h, valid, projected = z["critic"], z["residual"], z["h"], z["valid"], z["projected"]
checks, cps = z["checks_soft"].astype(np.float64), z["checkpoints"]
n, A, M = q.shape
K, J = 8, 6
fresh = slice(1, 9)


def ci(fn, *arrays, boots=2000):
    count = len(arrays[0])
    point = fn(*arrays)
    draws = np.array([fn(*(a[i] for a in arrays)) for i in RNG.integers(0, count, (boots, count))])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return f"{point:+.3f} [{lo:+.3f},{hi:+.3f}]"


cc = critic[:, fresh] - critic[:, fresh].mean(1, keepdims=True)
dev_m = q[:, fresh] - q[:, fresh].mean(1, keepdims=True)
dev = dev_m.mean(2)
c_energy = (cc ** 2).sum(1)
order = np.argsort(c_energy)
print(f"critic within-state energy: top 1% states {c_energy[order[-5:]].sum() / c_energy.sum():.2f}, top 5% {c_energy[order[-26:]].sum() / c_energy.sum():.2f}; "
      f"critic within sd: all {np.sqrt(c_energy.mean() / 7):.3f}, median state {np.sqrt(np.median(c_energy) / 7):.3f}; "
      f"true (raw incl. noise) median state {np.sqrt(np.median((dev ** 2).sum(1)) / 7):.3f}")
top = order[-5:]
print("  the 5 highest-energy states: |a0| max", np.round(np.abs(z["a0"][top]).max(1), 3), "age", z["age"][top], "critic V", np.round(critic[top, 0], 1),
      "true V", np.round(q[top, 0].mean(1), 1))
keep = np.ones(n, bool); keep[order[-26:]] = False


def pooled(c, t_m):
    t = t_m.mean(2)
    noise = (t_m.var(2, ddof=1) / t_m.shape[2]).sum()
    return (c * t).sum() / np.sqrt((c ** 2).sum() * max((t ** 2).sum() - noise, 1e-30))
print("within-state corr (noise-corrected), all states:", ci(pooled, cc, dev_m), "| critic-energy top 5% removed:", ci(pooled, cc[keep], dev_m[keep]))
spear = np.array([stats.spearmanr(critic[i, fresh], q[i, fresh].mean(1))[0] for i in range(n)])
print(f"per-state Spearman (raw truth): mean {spear.mean():+.3f} +- {spear.std() / np.sqrt(n):.3f}; trimmed-set mean {spear[keep].mean():+.3f}")
# what is the best achievable per-state Spearman given truth noise? simulate: truth_A vs truth_B halves
sp_half = np.array([stats.spearmanr(q[i, fresh][:, ::2].mean(1), q[i, fresh][:, 1::2].mean(1))[0] for i in range(n)])
print(f"  split-half truth-vs-truth per-state Spearman (32 vs 32 rollouts): mean {sp_half.mean():+.3f} -> a perfect critic would score ~{np.sqrt(max(2 * sp_half.mean() / (1 + sp_half.mean()), 0)):.2f} against the 64-rollout truth")

# self-consistency, robust
res_c = residual[:, fresh] - residual[:, fresh].mean(1, keepdims=True)
look = res_c + cc
def corr(a, b):
    return (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())
print("self-consistency corr(Q, r + gamma V'(s')) within state: all", ci(corr, cc, look), "| critic-energy top 5% removed", ci(corr, cc[keep], look[keep]))
ps = np.array([np.corrcoef(cc[i], look[i])[0, 1] for i in range(n)])
print(f"  per-state Pearson: mean {ps.mean():+.3f} +- {ps.std() / np.sqrt(n):.3f}, median {np.median(ps):+.3f}")
print("  one-step lookahead vs truth (noise-corrected): all", ci(pooled, look, dev_m), "| trimmed", ci(pooled, look[keep], dev_m[keep]))
print(f"  rms within-state: critic adv {np.sqrt((cc[keep] ** 2).mean()):.3f}, lookahead {np.sqrt((look[keep] ** 2).mean()):.3f} (trimmed set)")

# gradient, robust: trim by critic gradient energy
pairs = q[:, 1 + K:-1].reshape(n, 2, 2, J, 2, M)
deriv = (pairs[:, :, :, :, 0] - pairs[:, :, :, :, 1]) / (2 * h[..., None])
ge = (projected[:, 1] ** 2).sum(1)
gkeep = ge < np.quantile(ge, 0.95)
def slope(c, t_m):
    return (t_m.mean(2) * c).sum() / (c ** 2).sum()
def unit_slope(c, t_m):  # every state weighted equally: true rate along the critic's unit direction over its rms norm
    u = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-30)
    return (t_m.mean(2) * u).sum(1).mean()
for i, step in enumerate((0.1, 0.3)):
    d = deriv[:, 1, i]
    print(f"pre-space h={step}: slope all {ci(slope, projected[:, 1], d)} | critic-grad-energy top 5% removed {ci(slope, projected[gkeep, 1], d[gkeep])} "
          f"| pooled corr corrected trimmed {ci(pooled, projected[gkeep, 1], d[gkeep])}")
    print(f"     true rate of improvement along the critic's unit gradient (equal state weight): all {ci(unit_slope, projected[:, 1], d)} | trimmed {ci(unit_slope, projected[gkeep, 1], d[gkeep])}"
          f" ; critic's claimed rate (median |grad|) {np.median(np.linalg.norm(projected[:, 1], axis=1)):.3f}, mean {np.linalg.norm(projected[:, 1], axis=1).mean():.3f}")
    t = d.mean(2); noise = (d.var(2, ddof=1) / M).sum(1)
    true_norm2 = (t ** 2).sum(1) - noise
    print(f"     true |grad|: rms (noise-corrected) {np.sqrt(max(true_norm2.mean(), 0)):.3f}; => max achievable equal-weight rate ~ E|g| <= {np.sqrt(max(true_norm2.mean(), 0)):.3f}")

# horizon structure of the truth: split-half disattenuated corr of t-step within-state advantage with the H=400 one
print("horizon structure (within-state, fresh actions): corr(truth_t, truth_400) disattenuated by split halves; sd of the difference")
def dm(x):
    return x - x.mean(1, keepdims=True)
fa, fb = dm(q[:, fresh][:, :, ::2].mean(2)), dm(q[:, fresh][:, :, 1::2].mean(2))
for j, t in enumerate(cps):
    ta, tb = dm(checks[j][:, fresh][:, :, ::2].mean(2)), dm(checks[j][:, fresh][:, :, 1::2].mean(2))
    cross = 0.5 * ((ta * fb).sum() + (tb * fa).sum())
    r = cross / np.sqrt(max((ta * tb).sum(), 1e-30) * (fa * fb).sum())
    print(f"  t={int(t):3d}: corr {r:+.3f}; signal var t-step {(ta * tb).sum() / (n * 7):.4f} vs H400 {(fa * fb).sum() / (n * 7):.4f}; cross-cov {cross / (n * 7):.4f}")

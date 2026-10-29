"""v60 @1M: does the critic's own V', composed with the TRUE one-step dynamics (r + gamma V'(f(s, a)), the t = 1
checkpoint), rank actions better than its direct Q(s, a) readout? Robust statistics (trim levels, per-state rank
correlations) for the t-step lookaheads against the H = 400 truth; also the pre-tanh gradient version."""
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
z = np.load(REPO / "runs/reppo_critic_oracle/v60_base_1015424.npz")
RNG = np.random.default_rng(0)
q = z["q"][0].astype(np.float64)
critic, h, projected = z["critic"], z["h"], z["projected"]
checks, cps = z["checks_soft"].astype(np.float64), z["checkpoints"]
n, A, M = q.shape
fresh = slice(1, 9)
dm = lambda x: x - x.mean(1, keepdims=True)  # noqa: E731
truth_m = dm(q[:, fresh])
truth = truth_m.mean(2)
cc = dm(critic[:, fresh])


def pooled(c, t_m):
    t = t_m.mean(2)
    noise = (t_m.var(2, ddof=1) / t_m.shape[2]).sum()
    return (c * t).sum() / np.sqrt((c ** 2).sum() * max((t ** 2).sum() - noise, 1e-30))


def ci(fn, *arrays, boots=1000):
    count = len(arrays[0])
    point = fn(*arrays)
    draws = np.array([fn(*(a[i] for a in arrays)) for i in RNG.integers(0, count, (boots, count))])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return f"{point:+.3f} [{lo:+.3f},{hi:+.3f}]"


preds = {"Q(s,a) direct": cc}
for j, t in enumerate(cps[:4]):
    # predictor = mean over rollouts of the t-step lookahead; for t=1 this is deterministic dynamics + V' samples
    preds[f"lookahead t={int(t)}"] = dm(checks[j][:, fresh].mean(2))
print("predictor            | pooled corr (noise-corrected) by trim of the predictor's own top-energy states            | per-state Spearman mean+-se | rms within")
for name, p in preds.items():
    e = (p ** 2).sum(1)
    cells = []
    for trim in (0.0, 0.01, 0.05, 0.10):
        keep = e <= np.quantile(e, 1 - trim) if trim else np.ones(n, bool)
        # for t>1 lookaheads the predictor shares rollout noise with the truth; use split halves: pred from even, truth from odd
        if name.startswith("lookahead") and not name.endswith("t=1"):
            j = list(preds).index(name) - 1
            pa = dm(checks[j][:, fresh][:, :, ::2].mean(2))
            cells.append(f"{trim:.2f}: " + ci(pooled, pa[keep], truth_m[keep][:, :, 1::2]))
        else:
            cells.append(f"{trim:.2f}: " + ci(pooled, p[keep], truth_m[keep]))
    sp = np.array([stats.spearmanr(p[i], truth[i])[0] for i in range(n)])
    print(f"{name:20s} | " + " | ".join(cells) + f" | {sp.mean():+.3f}+-{sp.std() / np.sqrt(n):.3f} | {np.sqrt(np.median(e) / 7):.3f} (median state)")
sp_half = np.array([stats.spearmanr(truth_m[i][:, ::2].mean(1), truth_m[i][:, 1::2].mean(1))[0] for i in range(n)])
print(f"truth split-half per-state Spearman {sp_half.mean():+.3f} (ceiling for a perfect predictor vs 64-rollout truth ~ {np.sqrt(2 * sp_half.mean() / (1 + sp_half.mean())):.2f})")
print("NB t=1 lookahead noise: V' averaged over 16 policy samples x 64 keys -> independent of the truth's rollout noise except through CRN first-step sharing (none: first step is deterministic).")

# gradient version in pre-tanh space
K, J = 8, 6
def pair_derivs(values):
    pairs = values[:, 1 + K:-1].reshape(n, 2, 2, J, 2, *values.shape[2:])
    return (pairs[:, :, :, :, 0] - pairs[:, :, :, :, 1]) / (2 * h.reshape(h.shape + (1,) * (values.ndim - 2)))
with np.errstate(invalid="ignore", divide="ignore"):
    truth_d = pair_derivs(q)
    look_d = pair_derivs(checks[0].mean(2))
print("pre-tanh gradient: predictor vs true CRN central differences; pooled corr noise-corrected, by trim of predictor energy; equal-weight unit-direction true rate")
for i, step in enumerate((0.1, 0.3)):
    t_m = truth_d[:, 1, i]
    for name, p in (("critic autograd", projected[:, 1]), ("one-step lookahead FD", look_d[:, 1, i])):
        e = (p ** 2).sum(1)
        cells = []
        for trim in (0.0, 0.05):
            keep = e <= np.quantile(e, 1 - trim) if trim else np.ones(n, bool)
            cells.append(f"trim {trim:.2f}: corr {ci(pooled, p[keep], t_m[keep])}")
        unit = lambda c, tm: ((tm.mean(2) * c).sum(1) / np.maximum(np.linalg.norm(c, axis=1), 1e-30)).mean()  # noqa: E731
        print(f"  h={step} {name:22s} " + " | ".join(cells) + f" | true rate along unit direction {ci(unit, p, t_m)}")

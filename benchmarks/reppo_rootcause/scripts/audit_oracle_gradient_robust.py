"""Oracle gradient statistics with the outlier state(s) removed and with state-bootstrap SEs.

slope_true_on_critic = sum(d_true . g_critic) / sum(g_critic^2) weights each state by |g_critic|^2, so a single
off-distribution state with a huge critic gradient can own the statistic. Same for the pooled within-state corr."""
import sys
import numpy as np

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "runs/reppo_critic_oracle/v60_base_1015424.npz")
K, steps = 8, (0.1, 0.3)
q = d["q"].astype(np.float64)[0]  # soft truth (n, A, M)
check = {int(t): d["checks_soft"].astype(np.float64)[i] for i, t in enumerate(d["checkpoints"])}
critic, projected, h, valid = d["critic"], d["projected"], d["h"], d["valid"]
n = q.shape[0]
rng = np.random.default_rng(0)


def pair_derivatives(values):
    rest = values.shape[2:]
    pairs = values[:, 1 + K:-1].reshape(n, 2, len(steps), -1, 2, *rest)
    step = h.reshape(h.shape + (1,) * len(rest))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pairs[:, :, :, :, 0] - pairs[:, :, :, :, 1]) / (2.0 * step)


def slope(truth, pred, idx):
    return (truth[idx] * pred[idx]).sum() / (pred[idx] ** 2).sum()


def boot(truth, pred, idx, reps=400):
    vals = [slope(truth, pred, rng.choice(idx, len(idx))) for _ in range(reps)]
    return np.std(vals)


gnorm = np.linalg.norm(d["grad"], axis=1)
order = np.argsort(gnorm)[::-1]
print("critic |dQ/da| quantiles 50/90/99/max:", np.quantile(gnorm, [0.5, 0.9, 0.99, 1.0]).round(3),
      "| share of sum |g|^2 in top 1/5 states:", (gnorm[order[:1]] ** 2).sum() / (gnorm ** 2).sum(),
      (gnorm[order[:5]] ** 2).sum() / (gnorm ** 2).sum(), "| top states", order[:5], "their Q0", critic[order[:5], 0].round(1))
truth = pair_derivatives(q).mean(-1)  # (n, 2, |h|, J)
model1 = pair_derivatives(check[1]).mean(-1)
model5 = pair_derivatives(check[5]).mean(-1)
critic_fd = pair_derivatives(critic)
ok = valid.all(-1)
q0 = critic[:, 0]
inlier = q0 > np.quantile(q0, 0.02)  # drop the 2% lowest-value (off-distribution) states
print(f"inliers: {inlier.sum()}/{n} (Q0 > {np.quantile(q0, 0.02):.1f})")
for s, space in enumerate(("action", "pre")):
    for i, step in enumerate(steps):
        for name, keep in (("all", ok[:, s, i]), ("inlier", ok[:, s, i] & inlier)):
            idx = np.flatnonzero(keep)
            t, c = truth[:, s, i], projected[:, s]
            row = [f"{space:6s} h={step} {name:6s} n={len(idx):3d}"]
            for label, pred in (("critic", c), ("critic_fd", critic_fd[:, s, i]), ("look1", model1[:, s, i]), ("look5", model5[:, s, i])):
                row.append(f"true-on-{label} {slope(t, pred, idx):+.3f}+-{boot(t, pred, idx):.3f}")
            row.append(f"look1-on-critic {slope(model1[:, s, i], c, idx):+.3f}")
            row.append(f"rms critic {np.sqrt((c[idx] ** 2).mean()):.3f} look1 {np.sqrt((model1[idx, s, i] ** 2).mean()):.3f} "
                       f"look5 {np.sqrt((model5[idx, s, i] ** 2).mean()):.3f} truth(raw, noisy) {np.sqrt((t[idx] ** 2).mean()):.3f}")
            cos = (model1[idx, s, i] * c[idx]).sum(-1) / np.sqrt((model1[idx, s, i] ** 2).sum(-1) * (c[idx] ** 2).sum(-1))
            row.append(f"cos(look1, critic) median {np.median(cos):+.2f}")
            print(" | ".join(row))

# within-state (K fresh actions), inliers, with state bootstrap
fresh_c = critic[:, 1:1 + K] - critic[:, 1:1 + K].mean(1, keepdims=True)
fresh_t = q[:, 1:1 + K]
dev = fresh_t - fresh_t.mean(1, keepdims=True)
tm = dev.mean(-1)
noise = dev.var(-1, ddof=1) / q.shape[-1]


def within(idx):
    c, t = fresh_c[idx], tm[idx]
    signal = (t ** 2).sum() - noise[idx].sum()
    return ((c * t).sum() / np.sqrt((c ** 2).sum() * signal), np.sqrt((c ** 2).sum() / signal),
            (c * t).sum() / (c ** 2).sum(), (c * t).sum() / signal)


for name, idx in (("all", np.arange(n)), ("inlier", np.flatnonzero(inlier))):
    base = within(idx)
    bs = np.array([within(rng.choice(idx, len(idx))) for _ in range(400)]).std(0)
    print(f"within-state {name}: noise-corrected corr {base[0]:+.3f}+-{bs[0]:.3f}, sd ratio critic/true {base[1]:.3f}+-{bs[1]:.3f}, "
          f"slope true-on-critic {base[2]:+.3f}+-{bs[2]:.3f}, slope critic-on-true {base[3]:+.3f}+-{bs[3]:.3f}")

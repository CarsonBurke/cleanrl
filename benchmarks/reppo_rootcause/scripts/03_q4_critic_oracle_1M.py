"""Q4 (critic oracle, v60-base @1.015M): re-derive the published numbers from runs/reppo_critic_oracle/*.npz with
bootstrap CIs (states; whole env lanes), split-half disattenuation, and a variance decomposition incl. an
approximate lambda-return built from the saved n-step checkpoints. numpy only."""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
z = np.load(REPO / "runs/reppo_critic_oracle/v60_base_1015424.npz")
report = json.load(open(REPO / "runs/reppo_critic_oracle/v60_base.json"))[0]
RNG = np.random.default_rng(0)
BOOT = 2000
q = z["q"][0].astype(np.float64)  # soft, through the limit: (512, 58, 64)
critic, residual, row = z["critic"], z["residual"], z["row"]
n, A, M = q.shape
K, J = 8, 6
fresh = slice(1, 1 + K)
h, valid, projected = z["h"], z["valid"], z["projected"]
checks = z["checks_soft"].astype(np.float64)  # (8, 512, 58, 64)
cps = z["checkpoints"]


def boot(fn, *arrays, clusters=None, boots=BOOT):
    count = len(arrays[0])
    point = np.atleast_1d(fn(*arrays))
    draws = []
    ids = None if clusters is None else np.unique(clusters)
    for _ in range(boots):
        idx = RNG.integers(0, count, count) if clusters is None else np.concatenate(
            [np.flatnonzero(clusters == c) for c in RNG.choice(ids, len(ids))])
        draws.append(np.atleast_1d(fn(*(a[idx] for a in arrays))))
    lo, hi = np.nanpercentile(np.array(draws), [2.5, 97.5], axis=0)
    return point, lo, hi


def show(label, triple, names):
    point, lo, hi = triple
    print(f"{label}: " + "; ".join(f"{nm} {p:+.3f} [{l:+.3f},{u:+.3f}]" for nm, p, l, u in zip(names, point, lo, hi)))


# ---------------- level
def level(c, s):
    truth = s.mean(1)
    noise = (s.var(1, ddof=1) / s.shape[1]).mean()
    err = c - truth
    return np.array([np.corrcoef(c, truth)[0, 1], np.sqrt(max(err.var() - noise, 0)), np.sqrt(max(truth.var() - noise, 0)), err.mean()])
print("== LEVEL (published: corr 0.93, err sd 0.90, between sd 2.3)")
show("  state boot", boot(level, critic[:, 0], q[:, 0]), ["corr", "err_sd", "between_sd", "bias"])
show("  lane boot ", boot(level, critic[:, 0], q[:, 0], clusters=row), ["corr", "err_sd", "between_sd", "bias"])


# ---------------- within
def within(c, s):
    m = s.shape[2]
    dev_m = s - s.mean(1, keepdims=True)
    dev = dev_m.mean(2)
    noise = (dev_m.var(2, ddof=1) / m).sum()
    ours = c - c.mean(1, keepdims=True)
    tt = (dev ** 2).sum()
    corr = (ours * dev).sum() / np.sqrt((ours ** 2).sum() * tt)
    rel = 1 - noise / tt
    # split-half (independent rollouts) disattenuation
    a, b = dev_m[:, :, ::2].mean(2), dev_m[:, :, 1::2].mean(2)
    sh = 0.5 * ((ours * a).sum() + (ours * b).sum()) / np.sqrt((ours ** 2).sum() * max((a * b).sum(), 1e-30))
    true_sd = np.sqrt(max(tt - noise, 0) / (len(c) * (c.shape[1] - 1)))
    crit_sd = np.sqrt((ours ** 2).sum() / (len(c) * (c.shape[1] - 1)))
    return np.array([corr, corr / np.sqrt(rel) if rel > 0 else np.nan, sh, rel, true_sd, crit_sd, (ours * dev).sum() / (ours ** 2).sum()])
names = ["corr_raw", "corr_corrected", "corr_splithalf", "reliability", "true_sd", "critic_sd", "slope_true_on_critic"]
print("== WITHIN-STATE over 8 fresh actions (published: corr 0.11, 0.15 corrected, true sd 0.22, critic sd 0.21)")
show("  state boot", boot(within, critic[:, fresh], q[:, fresh]), names)
show("  lane boot ", boot(within, critic[:, fresh], q[:, fresh], clusters=row), names)
# heavy-tail check: drop the 5% of states with the largest true within-state energy
dev = (q[:, fresh] - q[:, fresh].mean(1, keepdims=True)).mean(2)
energy = (dev ** 2).sum(1)
keep = energy < np.quantile(energy, 0.95)
print(f"  top 5% states hold {energy[~keep].sum() / energy.sum():.2f} of the raw within-state energy; top 1% {np.sort(energy)[-5:].sum() / energy.sum():.2f}")
show("  trimmed 5%", boot(within, critic[keep][:, fresh], q[keep][:, fresh]), names)
per_state = np.array([np.corrcoef(critic[i, fresh], q[i, fresh].mean(1))[0, 1] for i in range(n)])
print(f"  per-state Pearson: mean {per_state.mean():+.3f} +- {per_state.std() / np.sqrt(n):.3f}, median {np.median(per_state):+.3f}, frac>0 {np.mean(per_state > 0):.3f}")


# ---------------- gradients
def pair_derivs(values):
    pairs = values[:, 1 + K:-1].reshape(n, 2, 2, J, 2, *values.shape[2:])
    step = h.reshape(h.shape + (1,) * (values.ndim - 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pairs[:, :, :, :, 0] - pairs[:, :, :, :, 1]) / (2 * step)
deriv = pair_derivs(q)  # (n, space, step, J, M)
ok = valid.all(-1)


def grad(samples, c):
    m = samples.shape[2]
    truth = samples.mean(2)
    noise = (samples.var(2, ddof=1) / m).sum()
    tt, pp, dot = (truth ** 2).sum(), (c ** 2).sum(), (truth * c).sum()
    norms = np.maximum(np.linalg.norm(c, axis=1), 1e-30)
    a, b = samples[:, :, ::2].mean(2), samples[:, :, 1::2].mean(2)
    sig_sh = (a * b).sum()
    return np.array([dot / pp, ((truth * c).sum(1) / norms).sum() / norms.sum(), dot / np.sqrt(pp * tt),
                     dot / np.sqrt(pp * (tt - noise)) if tt > noise else np.nan,
                     dot / np.sqrt(pp * sig_sh) if sig_sh > 0 else np.nan, 1 - noise / tt])
gn = ["slope_true_on_critic", "ascent_ratio", "pooled_corr_raw", "pooled_corr_corrected", "pooled_corr_splithalf", "truth_signal_frac"]
print("== ACTION GRADIENT (published, pre-tanh: slope 0.37+-0.12 @h0.1, 0.21+-0.05 @h0.3)")
for s, space in enumerate(("action", "pre")):
    for i, step in enumerate((0.1, 0.3)):
        k = ok[:, s, i]
        show(f"  {space:6s} h={step} n={k.sum():3d} state boot", boot(grad, deriv[k, s, i], projected[k, s]), gn)
        if space == "pre":
            show(f"  {space:6s} h={step}         lane boot ", boot(grad, deriv[k, s, i], projected[k, s], clusters=row[k]), gn)
# anomaly check: pre-space gradient restricted to the states that have valid action-space pairs, and the complement
for i, step in enumerate((0.1, 0.3)):
    k = ok[:, 0, i]
    show(f"  pre h={step} on states WITH valid action pairs (n={k.sum()})", boot(grad, deriv[k, 1, i], projected[k, 1]), gn)
    show(f"  pre h={step} on states WITHOUT (saturated; n={(~k).sum()})   ", boot(grad, deriv[~k, 1, i], projected[~k, 1]), gn)
sat = (np.abs(z["a0"]) > 0.95).any(1)
print(f"  states with any |a0|>0.95: {sat.mean():.3f}; mean |a0| {np.abs(z['a0']).mean():.3f}; frac of action coords |a0|>0.95: {(np.abs(z['a0']) > 0.95).mean():.3f}")
# energy concentration of the critic's projected gradient (what the slope is weighted by)
e = (projected[:, 1] ** 2).sum(1)
print(f"  critic pre-grad energy: top 5% states hold {np.sort(e)[-26:].sum() / e.sum():.2f}; slope contribution by state is weighted by this")


# ---------------- self-consistency
def selfcons(res, c, s):
    dev_m = s - s.mean(1, keepdims=True)
    adv_err = (c - c.mean(1, keepdims=True)) - dev_m.mean(2)
    cen = res - res.mean(1, keepdims=True)
    cc = c - c.mean(1, keepdims=True)
    lookahead = cen + cc  # r + gamma V'(s') centered within state
    return np.array([(cen * -adv_err).sum() / np.sqrt((cen ** 2).sum() * (adv_err ** 2).sum()),
                     (cc * lookahead).sum() / np.sqrt((cc ** 2).sum() * (lookahead ** 2).sum()),
                     (lookahead * dev_m.mean(2)).sum() / np.sqrt((lookahead ** 2).sum() * (dev_m.mean(2) ** 2).sum()),
                     np.sqrt((lookahead ** 2).mean()), np.sqrt((cc ** 2).mean()), np.sqrt((cen ** 2).mean())])
print("== SELF-CONSISTENCY within state (published: corr of Q with its own r + gamma V'(f(s,a)) = -0.33)")
show("  state boot", boot(selfcons, residual[:, fresh], critic[:, fresh], q[:, fresh]),
     ["corr(residual, -adv_err)", "corr(Q, r+gV')", "corr(r+gV', truth) raw", "rms lookahead", "rms critic adv", "rms residual"])

# ---------------- horizon sweep of the within-state truth (critic tail after t steps)
print("== HORIZON SWEEP: within-state sd of the t-step truth (critic bootstrap after t) and critic corr with it")
for j, t in enumerate(cps):
    w = within(critic[:, fresh], checks[j][:, fresh])
    print(f"  t={int(t):3d}: true within sd {w[4]:.3f} (reliability {w[3]:.2f}); critic corr raw {w[0]:+.3f} corrected {w[1]:+.3f}; slope true on critic {w[6]:+.3f}")

# ---------------- variance decomposition incl. approximate lambda-returns
print("== VARIANCE DECOMPOSITION @1M (scaled^2)")
qa0 = q[:, 0]
single = qa0.var(1, ddof=1).mean()
lev = level(critic[:, 0], qa0)
w = within(critic[:, fresh], q[:, fresh])
print(f"  between-state var {lev[2] ** 2:.3f}; within-state true var {w[4] ** 2:.4f}; critic error var {lev[1] ** 2:.3f}; "
      f"MC(H=400) single-rollout noise var {single:.3f}")
gamma = 0.99
for lam in (0.8, 0.95):
    weights = np.zeros(len(cps))
    edges = [1] + [int(np.ceil(np.sqrt(cps[i] * cps[i + 1]))) for i in range(len(cps) - 1)] + [10 ** 9]
    for i in range(len(cps)):
        lo_n, hi_n = edges[i], edges[i + 1]
        weights[i] = lam ** (lo_n - 1) - (lam ** (hi_n - 1) if hi_n < 10 ** 8 else 0.0)
    g_lam = np.tensordot(weights, checks, axes=(0, 0))  # (512, 58, 64) approx lambda-return samples
    noise = g_lam[:, 0].var(1, ddof=1).mean()
    dev_m = g_lam[:, fresh] - g_lam[:, fresh].mean(1, keepdims=True)
    sig = ((dev_m.mean(2) ** 2).sum() - (dev_m.var(2, ddof=1) / M).sum()) / (n * (K - 1))
    # single-sample advantage noise without CRN: independent rollouts for each action = single-rollout noise
    c = gamma * lam
    print(f"  lambda {lam}: checkpoint weights {np.round(weights, 3)}; lambda-return sampling-noise var {noise:.4f}; "
          f"within-state var of E[G^lam] {sig:.4f}; signal share {sig / (sig + noise):.3f}; "
          f"analytic sigma_A^2 c^2/(1-c^2) = {w[4] ** 2 * c * c / (1 - c * c):.4f}")

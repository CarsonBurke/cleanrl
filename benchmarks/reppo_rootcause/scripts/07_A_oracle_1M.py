"""A. Red-team of the "one flipped state / heavy tail" re-reading of the 1M critic oracle.

    .venv/bin/python benchmarks/reppo_rootcause/scripts/07_A_oracle_1M.py > benchmarks/reppo_rootcause/data/07_A_log.txt

Raw arrays only (runs/reppo_critic_oracle/v60_base_1015424.npz). Layout per state: [a0, 8 fresh, 48 FD probes
(2 spaces x 2 steps x 6 directions x +-), a0 duplicate]; q[0] is the soft truth (512, 58, 64 CRN rollouts, H 400);
checks_soft[j] is "t real steps then the critic's V'" for t in checkpoints.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
z = np.load("runs/reppo_critic_oracle/v60_base_1015424.npz")
q, checks, critic, projected, h, valid = z["q"][0].astype(np.float64), z["checks_soft"].astype(np.float64), z["critic"], z["projected"], z["h"], z["valid"]
cps = list(z["checkpoints"])
n, A, M = q.shape
K = 8
fresh = slice(1, 1 + K)
print(f"states {n} actions {A} rollouts {M} checkpoints {cps}; FD steps pre {np.unique(h[:, 1])}; CRN dup err {np.abs(q[:, -1] - q[:, 0]).max():.1e}")


def pair_fd(values, space, step):
    """(n, 6, ...) central differences for a space (0 action, 1 pre-tanh) and step index."""
    pairs = values[:, 1 + K:-1].reshape(n, 2, 2, 6, 2, *values.shape[2:])
    hh = h[:, space, step].reshape(n, 6, *([1] * (values.ndim - 2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pairs[:, space, step, :, 0] - pairs[:, space, step, :, 1]) / (2 * hh)


def boot(fn, keep, draws=2000):
    idx = np.flatnonzero(keep)
    out = [fn(rng.choice(idx, len(idx))) for _ in range(draws)]
    return np.nanpercentile(out, [2.5, 97.5])


def centered(x):
    return x - x.mean(1, keepdims=True)


# ---------------------------------------------------------------- 1. reproduce the pooled numbers
qf = q[:, fresh]
truth_f = centered(qf.mean(-1)) * K / (K - 1)
dev_m = qf - qf.mean(1, keepdims=True)
noise_f = dev_m.var(-1, ddof=1) / M  # (n, K) rollout-noise variance of the centered truth
crit_f = centered(critic[:, fresh])
look1_f = centered(checks[0][:, fresh].mean(-1))
print("\n== 1. pooled reproduction (all 512 states)")
pooled = lambda x, y, sel=slice(None): (x[sel] * y[sel]).sum() / np.sqrt((x[sel] ** 2).sum() * (y[sel] ** 2).sum())
rel_all = 1 - noise_f.sum() / (centered(qf.mean(-1)) ** 2).sum()
print(f"within corr critic vs truth {pooled(crit_f, truth_f):+.3f} (corrected {pooled(crit_f, truth_f) / np.sqrt(rel_all):+.3f}, reliability {rel_all:.3f})")
print(f"self-consistency corr critic vs own 1-step lookahead {pooled(crit_f, look1_f):+.3f}")
true_sd = np.sqrt(((centered(qf.mean(-1)) ** 2).sum(1) - noise_f.sum(1)).mean() / (K - 1))
print(f"true within sd {true_sd:.3f}; critic within sd pooled {np.sqrt((crit_f ** 2).sum(1).mean() / (K - 1)):.3f}, median state {np.median(np.sqrt((crit_f ** 2).sum(1) / (K - 1))):.3f}")
for step, hv in enumerate((0.1, 0.3)):
    d = pair_fd(q, 1, step)
    g = projected[:, 1]
    print(f"pre-tanh h={hv}: slope true-on-critic {(d.mean(-1) * g).sum() / (g ** 2).sum():+.3f}")

# ---------------------------------------------------------------- 2. who owns the pooled statistics
print("\n== 2. influence")
e_within = (crit_f ** 2).sum(1)
e_grad = (projected[:, 1] ** 2).sum(1)
order = np.argsort(-e_grad)
print("top-5 states by critic pre-tanh gradient energy:", order[:5].tolist(), "shares", np.round(e_grad[order[:5]] / e_grad.sum(), 3).tolist())
print("top-5 states by critic within-state energy:", np.argsort(-e_within)[:5].tolist(), "shares", np.round(np.sort(e_within)[::-1][:5] / e_within.sum(), 3).tolist())
top = order[0]
print(f"state {top}: age {z['age'][top]}, critic Q(a0) {critic[top, 0]:.2f}, true Q(a0) {q[top, 0].mean():.2f} +- {q[top, 0].std(ddof=1) / 8:.2f}; "
      f"rest: critic Q median {np.median(critic[:, 0]):.2f}, true median {np.median(q[:, 0].mean(-1)):.2f}")
print(f"  its within-state: critic sd {np.sqrt(e_within[top] / (K - 1)):.3f}, true sd (noise-corrected) "
      f"{np.sqrt(max((centered(qf.mean(-1))[top] ** 2).sum() - noise_f[top].sum(), 0) / (K - 1)):.3f}, "
      f"corr critic-truth {np.corrcoef(crit_f[top], truth_f[top])[0, 1]:+.3f}, corr critic-lookahead1 {np.corrcoef(crit_f[top], look1_f[top])[0, 1]:+.3f}")
for step, hv in enumerate((0.1, 0.3)):
    d = pair_fd(q, 1, step)[top]  # (6, M)
    g = projected[top, 1]
    proj = (d * g[:, None]).sum(0) / np.linalg.norm(g)
    d1 = pair_fd(checks[0], 1, step)[top].mean(-1)
    d5 = pair_fd(checks[1], 1, step)[top].mean(-1)
    cosv = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    print(f"  h={hv}: |g_critic| {np.linalg.norm(g):.2f}; true rate along critic's unit gradient {proj.mean():+.3f} +- {proj.std(ddof=1) / np.sqrt(M):.3f} "
          f"(|true grad| {np.linalg.norm(d.mean(-1)):.2f}, noise rms/dim {np.sqrt((d.var(-1, ddof=1) / M).mean()):.2f}); "
          f"cos(critic, 1-step lookahead FD) {cosv(g, d1):+.3f}, cos(critic, 5-step) {cosv(g, d5):+.3f}, cos(1-step, truth) {cosv(d1, d.mean(-1)):+.3f}, cos(5-step, truth) {cosv(d5, d.mean(-1)):+.3f}")

# ---------------------------------------------------------------- 3. what the actor experiences
print("\n== 3. statistics by how the field is consumed (pre-tanh space, the reparameterized path)")
print("   field step delta_a(s) = w(s) g_c(s); true first-order gain = sum_s w(s) g_c(s).g_true(s); reported as")
print("   G = mean_s(w g_c.g_true) / sqrt(mean_s |w g_c|^2)  (true rate per unit rms step; a perfect field scores rms|g_true|)")
masks = {"all": np.ones(n, bool), "drop top state": np.arange(n) != top, "drop top 1% |g|": e_grad < np.quantile(e_grad, 0.99),
         "drop top 5% |g|": e_grad < np.quantile(e_grad, 0.95)}
for step, hv in enumerate((0.1, 0.3)):
    d_m = pair_fd(q, 1, step)
    d = d_m.mean(-1)
    g = projected[:, 1]
    noise = d_m.var(-1, ddof=1).sum(1) / M
    sig_rms = np.sqrt(max(((d ** 2).sum(1) - noise).mean(), 0))
    gn = np.linalg.norm(g, axis=1)
    print(f" h={hv}: true gradient rms (noise-corrected, per state) {sig_rms:.3f}")
    weights = {"raw (w=1): the unconstrained sum the actor loss differentiates": np.ones(n),
               "clip |g| at p99": np.minimum(1, np.quantile(gn, 0.99) / gn), "clip |g| at p90": np.minimum(1, np.quantile(gn, 0.90) / gn),
               "unit (w=1/|g|): a per-state trust region": 1 / gn}
    for wname, w in weights.items():
        for mname, keep in masks.items():
            if wname.startswith("unit") and mname not in ("all", "drop top state"):
                continue
            fn = lambda idx: (w[idx] * (g[idx] * d[idx]).sum(1)).mean() / np.sqrt(((w[idx] * gn[idx]) ** 2).mean())
            val = fn(np.flatnonzero(keep))
            lo, hi = boot(fn, keep, 1000)
            print(f"   {wname:66s} | {mname:16s} G {val:+.3f} [{lo:+.3f}, {hi:+.3f}]  = {val / sig_rms:+.3f} of a perfect field's rms")

# ---------------------------------------------------------------- 4. which regression says what
print("\n== 4. regression directions (fresh-action within-state advantages; truth noise removed where it biases)")
print("   slope(truth on critic) = cov/var(critic): unbiased under truth noise; equals 1 for ANY conditional-mean predictor,")
print("   including an optimally shrunk noisy one. slope(critic on truth) = cov/var(truth signal). r^2 = product.")
for mname, keep in masks.items():
    def fn(idx):
        c, t = crit_f[idx], truth_f[idx]
        sig = ((centered(qf[idx].mean(-1)) ** 2).sum() - noise_f[idx].sum()) * (K / (K - 1)) ** 2
        cov = (c * t).sum()
        return cov / (c ** 2).sum(), cov / sig, cov ** 2 / ((c ** 2).sum() * sig), np.sqrt((c ** 2).sum() / sig)
    val = fn(np.flatnonzero(keep))
    idxs = np.flatnonzero(keep)
    draws = np.array([fn(rng.choice(idxs, len(idxs))) for _ in range(2000)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    print(f"   {mname:16s} truth-on-critic {val[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}] | critic-on-truth {val[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}] "
          f"| r^2 {val[2]:.3f} [{lo[2]:.3f}, {hi[2]:.3f}] | sd ratio critic/true {val[3]:.2f} [{lo[3]:.2f}, {hi[3]:.2f}]")
print("   gradient (pre-tanh FD):")
for step, hv in enumerate((0.1, 0.3)):
    d_m = pair_fd(q, 1, step)
    d, g = d_m.mean(-1), projected[:, 1]
    nz = d_m.var(-1, ddof=1).sum(1) / M
    for mname, keep in masks.items():
        def fn(idx):
            cov, sig = (g[idx] * d[idx]).sum(), (d[idx] ** 2).sum() - nz[idx].sum()
            return cov / (g[idx] ** 2).sum(), cov / sig, cov ** 2 / ((g[idx] ** 2).sum() * sig) * np.sign(cov), np.sqrt((g[idx] ** 2).sum() / sig)
        val = fn(np.flatnonzero(keep))
        idxs = np.flatnonzero(keep)
        draws = np.array([fn(rng.choice(idxs, len(idxs))) for _ in range(2000)])
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        print(f"   h={hv} {mname:16s} truth-on-critic {val[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}] | critic-on-truth {val[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}] "
              f"| signed r^2 {val[2]:+.3f} [{lo[2]:+.3f}, {hi[2]:+.3f}] | norm ratio {val[3]:.2f} [{lo[3]:.2f}, {hi[3]:.2f}]")

# ---------------------------------------------------------------- 5. per-state distributions with split-half ceilings
print("\n== 5. per-state distributions (every state weighted equally; no trimming)")
qs = lambda x: "p10 %+.2f p25 %+.2f p50 %+.2f p75 %+.2f p90 %+.2f mean %+.3f +- %.3f frac>0 %.2f" % (
    *np.nanpercentile(x, [10, 25, 50, 75, 90]), np.nanmean(x), np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x))), np.nanmean(x > 0))
rowcorr = lambda a, b: np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)])
rowrank = lambda a, b: np.array([stats.spearmanr(x, y)[0] for x, y in zip(a, b)])
half_a, half_b = centered(qf[..., ::2].mean(-1)), centered(qf[..., 1::2].mean(-1))
r_hh = rowcorr(half_a, half_b)
for label, truth_like, note in (("H=400 truth", centered(qf.mean(-1)), ""),
                                ("10-step truth (t=10 then critic V'; 2.7x less noise, critic-contaminated)", centered(checks[cps.index(10)][:, fresh].mean(-1)), "")):
    print(f" reference: {label}")
    print("   critic  Pearson :", qs(rowcorr(crit_f, truth_like)))
    print("   critic  Spearman:", qs(rowrank(crit_f, truth_like)))
    for t in (1, 5):
        look = centered(checks[cps.index(t)][:, fresh].mean(-1))
        print(f"   {t}-step lookahead Pearson :", qs(rowcorr(look, truth_like)))
print("   truth split-half (32 vs 32 rollouts) Pearson:", qs(r_hh))
sb = 2 * np.nanmean(r_hh) / (1 + np.nanmean(r_hh))
print(f"   => mean per-state reliability of the 64-rollout truth {sb:.3f}; a perfect predictor's mean per-state Pearson ~ {np.sqrt(sb):.3f}")
print(f"   => critic's disattenuated mean per-state Pearson ~ {np.nanmean(rowcorr(crit_f, centered(qf.mean(-1)))) / np.sqrt(sb):+.3f}")
ten = centered(checks[cps.index(10)][:, fresh])
r10 = rowcorr(centered(ten[..., ::2].mean(-1)), centered(ten[..., 1::2].mean(-1)))
sb10 = 2 * np.nanmean(r10) / (1 + np.nanmean(r10))
print(f"   10-step truth: split-half mean {np.nanmean(r10):.3f}, reliability {sb10:.3f}; critic disattenuated mean per-state Pearson "
      f"{np.nanmean(rowcorr(crit_f, centered(checks[cps.index(10)][:, fresh].mean(-1)))) / np.sqrt(sb10):+.3f}")
cosr = lambda a, b: (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
print(" gradient cosines per state (6-D, pre-tanh):")
for step, hv in enumerate((0.1, 0.3)):
    for label, src in (("H=400", q), ("10-step", checks[cps.index(10)]), ("1-step", checks[0])):
        d_m = pair_fd(src, 1, step)
        g = projected[:, 1]
        c = cosr(g, d_m.mean(-1))
        sh = cosr(d_m[..., ::2].mean(-1), d_m[..., 1::2].mean(-1))
        rel = 2 * np.mean(sh) / (1 + np.mean(sh))
        print(f"   h={hv} vs {label:7s}: critic cos", qs(c), f"| truth split-half cos mean {np.mean(sh):+.3f} -> ceiling ~{np.sqrt(max(rel, 0)):.2f} -> disattenuated {np.mean(c) / np.sqrt(max(rel, 1e-9)):+.3f}")
print(" expected cosine of a random 6-D direction: 0 +- 0.41 per state")

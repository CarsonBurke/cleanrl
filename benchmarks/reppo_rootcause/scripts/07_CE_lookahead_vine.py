"""C + E. The transition channel and vine economics, from the 1M oracle arrays only.

    .venv/bin/python benchmarks/reppo_rootcause/scripts/07_CE_lookahead_vine.py > benchmarks/reppo_rootcause/data/07_CE_log.txt

checks_soft[j][s, a, m] = sum of the first t real rewards (+ entropy bonuses) + gamma^t V'_critic(s_t), rollout m
(future policy noise shared by all actions of a state: common random numbers). t in {1, 5, 10, 25, 50, 100, 200, 400}.
The arrays do NOT contain the reward-only partial sums, the first reward, the states or the fresh actions, so the split
"rewards alone vs the critic's V' tail" cannot be recomputed for t > 1, and horizons 2-4 and 6-9 do not exist.

Independence: rollouts are split into halves A (m < 32) and B (m >= 32). Any product of a label and a truth uses
different halves, so no label shares a rollout with its truth; variances of means are cross-half products.
"""
import numpy as np

rng = np.random.default_rng(0)
z = np.load("runs/reppo_critic_oracle/v60_base_1015424.npz")
q = z["q"][0].astype(np.float64)
checks = z["checks_soft"].astype(np.float64)
critic = z["critic"]
cps = list(z["checkpoints"])
n, A, M = q.shape
K = 8
half = M // 2
fresh = slice(1, 1 + K)
center = lambda x, axis=1: x - x.mean(axis, keepdims=True)
e_within = (center(critic[:, fresh]) ** 2).sum(1)
keeps = {"all 512 states": np.ones(n, bool), "drop top 1% critic-energy states (6 states)": e_within < np.quantile(e_within, 0.99)}
t_a, t_b = center(q[:, fresh, :half].mean(-1)), center(q[:, fresh, half:].mean(-1))
lab_a = [center(c[:, fresh, :half].mean(-1)) for c in checks]
lab_b = [center(c[:, fresh, half:].mean(-1)) for c in checks]
dot = lambda x, y, idx: (x[idx] * y[idx]).sum()


def ci(fn, keep, draws=1000):
    idx = np.flatnonzero(keep)
    vals = [fn(rng.choice(idx, len(idx))) for _ in range(draws)]
    return np.nanpercentile(vals, [2.5, 97.5])


def mean_corr(j):
    """corr(E[n-step label], truth) with rollout noise removed from both sides."""
    def fn(idx):
        num = 0.5 * (dot(lab_a[j], t_b, idx) + dot(lab_b[j], t_a, idx))
        den = dot(lab_a[j], lab_b[j], idx) * dot(t_a, t_b, idx)
        return num / np.sqrt(den) if den > 0 else np.nan
    return fn


print("== C(ii). corr(E[n real steps + critic V'], H=400 truth), within-state over 8 fresh actions, noise removed from both sides")
rho = {}
for name, keep in keeps.items():
    idx = np.flatnonzero(keep)
    print(f" {name}: true within-state variance {dot(t_a, t_b, idx) / len(idx) / (K - 1):.4f}, truth(32 rollouts) reliability {dot(t_a, t_b, idx) / np.sqrt(dot(t_a, t_a, idx) * dot(t_b, t_b, idx)):.3f}")
    cq = center(critic[:, fresh])
    fn_q = lambda ix: 0.5 * (dot(cq, t_a, ix) + dot(cq, t_b, ix)) / np.sqrt(dot(cq, cq, ix) * dot(t_a, t_b, ix))
    lo, hi = ci(fn_q, keep)
    print(f"   Q_critic(s, a) direct: {fn_q(idx):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for j, t in enumerate(cps):
        fn = mean_corr(j)
        lo, hi = ci(fn, keep)
        amp = np.sqrt(max(dot(lab_a[j], lab_b[j], idx), 0) / dot(t_a, t_b, idx))
        rho[name, t] = fn(idx)
        print(f"   n={t:3d}: {fn(idx):+.3f} [{lo:+.3f}, {hi:+.3f}]   sd(E label) / sd(truth) {amp:.2f}")

print("\n== C(i)/(iii): rewards alone vs critic V' alone. NOT separable from the arrays for n >= 5 (no reward-only partial sums).")
print("   n = 1 is separable in the action-gradient sense:")
print("   generator JSON model_based.*.reward_share (FD energy of r(s,a) alone / FD energy of r + gamma V'(f(s,a))) = 0.0011-0.0015,")
print("   i.e. <= 4% of the 1-step label's action-gradient amplitude is reward; the exactly-known control-cost part, recomputed here:")
pre0, dirs = z["pre0"].astype(np.float64), z["directions"].astype(np.float64)
for step, hv in enumerate((0.1, 0.3)):
    plus, minus = np.tanh(pre0[:, None] + hv * dirs), np.tanh(pre0[:, None] - hv * dirs)
    ctrl_fd = -0.1 * 0.1 * ((plus ** 2).sum(-1) - (minus ** 2).sum(-1)) / (2 * hv)  # reward_scale * ctrl_cost_weight
    pairs = checks[0][:, 1 + K:-1].mean(-1).reshape(n, 2, 2, 6, 2)
    model_fd = (pairs[:, 1, step, :, 0] - pairs[:, 1, step, :, 1]) / (2 * hv)
    ratio = (ctrl_fd ** 2).sum(1) / (model_fd ** 2).sum(1)
    print(f"   pre-tanh h={hv}: control-cost FD energy / 1-step-label FD energy: pooled {(ctrl_fd ** 2).sum() / (model_fd ** 2).sum():.5f}, "
          f"per-state median {np.median(ratio):.4f}, p90 {np.quantile(ratio, 0.9):.4f}")

print("\n== C(CRN) + E. single-rollout labels (what one branch delivers), correlation with the true advantage (truth-side noise removed)")
print("   CRN  : the K actions of a state share one rollout index; label_k = G_n(k, m) - mean_j G_n(j, m)")
print("   noCRN: every action uses a different rollout index (independent future noise), same centring")
print("   r_n  : reliability of one CRN-centred label = var(E label) / (var(E label) + branch noise)")


def single(j, k_actions, crn, keep, reps=6):
    idx = np.flatnonzero(keep)
    vals = []
    for _ in range(reps):
        sub = rng.permuted(np.tile(np.arange(K), (len(idx), 1)), axis=1)[:, :k_actions]
        if crn:
            m = np.repeat(rng.integers(0, half, (len(idx), 1)), k_actions, 1)
        else:
            m = rng.permuted(np.tile(np.arange(half), (len(idx), 1)), axis=1)[:, :k_actions]
        lab = center(checks[j][idx[:, None], 1 + sub, m])
        ta = center(q[idx[:, None], 1 + sub, :half].mean(-1))
        tb = center(q[idx[:, None], 1 + sub, half:].mean(-1))
        # truth-side disattenuation: var(truth signal) = <ta, tb>
        vals.append(((lab * tb).sum(), (lab ** 2).sum(), (ta * tb).sum()))
    num, lab_sq, sig = np.mean(vals, 0)
    return num / np.sqrt(lab_sq * sig)


table = {}
for name, keep in keeps.items():
    idx = np.flatnonzero(keep)
    print(f" {name}")
    print("   n     CRN K=8  K=4    K=2   | noCRN K=8 | branch-noise var: CRN-centred / uncentred | r_n   | ceiling rho_n")
    for j, t in enumerate(cps):
        row = [single(j, k, True, keep) for k in (8, 4, 2)]
        nocrn = single(j, 8, False, keep)
        lab = checks[j][keep][:, fresh]
        noise_crn = center(lab, 1).var(-1, ddof=1).mean() * K / (K - 1)
        noise_raw = lab.var(-1, ddof=1).mean()
        sig = dot(lab_a[j], lab_b[j], idx) / len(idx) / (K - 1)
        r_n = sig / (sig + noise_crn)
        table[name, t] = (row, nocrn, noise_crn, noise_raw, r_n, sig)
        print(f"   {t:3d}   {row[0]:+.3f}  {row[1]:+.3f}  {row[2]:+.3f} | {nocrn:+.3f}    | {noise_crn:.4f} / {noise_raw:.4f}                         | {r_n:.3f} | {rho[name, t]:+.3f}")

print("\n   the ordinary on-policy sample at the same snapshot (one action a0 per state, one rollout, no branching):")
grid = np.arange(1, 401)


def lambda_weights(lam):
    """Weights on the 8 saved horizons for the lambda-return, horizons between checkpoints linearly interpolated."""
    w_n = (1 - lam) * lam ** (grid - 1)
    w_n[-1] += lam ** 400
    w = np.zeros(len(cps))
    for horizon, weight in zip(grid, w_n):
        j = np.searchsorted(cps, horizon)
        if cps[j] == horizon:
            w[j] += weight
        else:
            frac = (horizon - cps[j - 1]) / (cps[j] - cps[j - 1])
            w[j - 1] += weight * (1 - frac)
            w[j] += weight * frac
    return w


ordinary = {}
v_critic = critic[:, fresh].mean(1)
for name, keep in keeps.items():
    idx = np.flatnonzero(keep)
    a_a = q[:, 0, :half].mean(-1) - q[:, fresh, :half].mean((1, 2))
    a_b = q[:, 0, half:].mean(-1) - q[:, fresh, half:].mean((1, 2))
    ca, cb = a_a[idx] - a_a[idx].mean(), a_b[idx] - a_b[idx].mean()
    sig = (ca * cb).mean()
    print(f" {name}: Var A_true(s, a0) = {sig:.4f}")
    for lam in (0.8, 0.95):
        w = lambda_weights(lam)
        lab = np.tensordot(w, checks[:, :, 0, :], axes=1)  # (n, M) lambda-return samples of (s, a0)
        noise = lab[idx].var(-1, ddof=1).mean()
        # correlation of a single-rollout label (critic baseline subtracted, as the trainer's diagnostics do) with A_true
        x = lab[idx][:, :half] - v_critic[idx, None]
        corr = np.mean([np.corrcoef(x[:, m], cb)[0, 1] for m in range(half)]) / np.sqrt(sig / cb.var())
        mean_lab = lab[idx][:, :half].mean(-1) - v_critic[idx]
        ordinary[name, lam] = sig / (sig + noise)
        print(f"   lambda={lam}: per-(s,a) sampling variance {noise:.4f} -> sampling-only signal share {sig / (sig + noise):.3f}; "
              f"single label minus critic V vs A_true corr {corr:+.3f}")

print("\n== E. information per env step.")
print("   Estimating the label's mean field from N labels proceeds at rate r/(1-r) per label (r = label reliability); the field it converges to")
print("   has correlation rho_n with the true advantage (the ceiling). Vine: K actions x n steps = K n env steps buy K-1 contrasts.")
print("   Ordinary sample: one label per env step with share s; rate s/(1-s). 'speed' below = vine rate per env step / ordinary rate.")
for name in keeps:
    print(f" {name}")
    refs = {"1M, lambda 0.8, sampling noise only": ordinary[name, 0.8], "1M, lambda 0.95, sampling noise only": ordinary[name, 0.95],
            "4M-8M measured (bootstrap error included)": 0.05, "same, upper": 0.07}
    for rname, share in refs.items():
        base = share / (1 - share)
        print(f"   ordinary share {share:.3f} ({rname}), rate {base:.3f}/step")
        for t in cps[:6]:
            _, _, _, _, r_n, _ = table[name, t]
            rate = r_n / max(1 - r_n, 1e-9)
            print(f"      n={t:3d}: ceiling {rho[name, t]:+.2f}, r_n {r_n:.3f}; speed K=2 {0.5 / t * rate / base:7.2f}x  K=4 {0.75 / t * rate / base:7.2f}x  K=8 {0.875 / t * rate / base:7.2f}x")

"""Re-derive the 'Q vs own one-step lookahead' within-state statistic from the oracle arrays, robustly."""
import sys
import numpy as np

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "runs/reppo_critic_oracle/v60_base_1015424.npz")
K = 8
critic = d["critic"][:, 1:1 + K]
checks = d["checks_soft"].astype(np.float64)[:, :, 1:1 + K].mean(-1)  # (T, n, K)
truth = d["q"].astype(np.float64)[0][:, 1:1 + K].mean(-1)
cps = d["checkpoints"]
c = critic - critic.mean(1, keepdims=True)


def pooled(a, b):
    return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())


def per_state(a, b):
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


print("t | pooled corr(critic, check_t) | per-state median | frac states >0 | within sd check_t | corr(check_t, truth)")
for i, t in enumerate(cps):
    m = checks[i] - checks[i].mean(1, keepdims=True)
    tr = truth - truth.mean(1, keepdims=True)
    ps = per_state(c, m)
    print(f"{t:4d} | {pooled(c, m):+.3f} | {np.median(ps):+.3f} | {(ps > 0).mean():.2f} | {np.sqrt((m ** 2).mean()):.3f} "
          f"| pooled {pooled(m, tr):+.3f} median {np.median(per_state(m, tr)):+.3f}")
m1 = checks[0] - checks[0].mean(1, keepdims=True)
energy = (m1 ** 2).sum(1)
order = np.argsort(energy)[::-1]
print("share of check_1 within-state energy in top 1/5/10% states:",
      [round(float(energy[order[:int(len(order) * f)]].sum() / energy.sum()), 3) for f in (0.01, 0.05, 0.1)])
cross = (c * m1).sum(1)
print("share of the (negative) cross term from top 5% energy states:", cross[order[:25]].sum() / cross.sum(),
      "cross total", cross.sum())
for lo, hi in ((0, 25), (25, 128), (128, 512)):
    idx = order[lo:hi]
    print(f"states ranked {lo}-{hi} by check_1 energy: pooled corr {pooled(c[idx], m1[idx]):+.3f}, "
          f"check sd {np.sqrt((m1[idx] ** 2).mean()):.3f}, critic sd {np.sqrt((c[idx] ** 2).mean()):.3f}")
a0 = d["a0"]
sat = (np.abs(a0) > 0.99).any(1)
print("saturated a0 states:", sat.mean(), "corr sat:", pooled(c[sat], m1[sat]), "unsat:", pooled(c[~sat], m1[~sat]))
# residual between-state: Q(s,a0) vs t-step lookahead
q0 = d["critic"][:, 0]
for i, t in enumerate(cps):
    look = d["checks_soft"].astype(np.float64)[i, :, 0].mean(-1)
    print(f"a0 residual t={t}: mean {np.mean(look - q0):+.3f} sd {np.std(look - q0):.3f}")
age = d["age"]
res1 = d["checks_soft"].astype(np.float64)[0, :, 0].mean(-1) - q0
print("corr(residual_1, age):", np.corrcoef(res1, age)[0, 1], " sd by age<50:", res1[age < 50].std() if (age < 50).any() else None,
      "age>=50:", res1[age >= 50].std(), "n young", (age < 50).sum())

"""Q1 continued: nonparametric (local-linear kNN) advantage estimators on the on-policy chunk, their
learning curve in N, and how linear the TRUE advantage is in the policy noise.

Inputs: data/03_chunk_*.npz. numpy only.

1. kNN local-linear: for each oracle state, the k nearest chunk states (normalized-obs Euclidean; the state's own
   env lane excluded, so its own trajectory never contributes), OLS of the lambda-return target on [1, z]:
   the slope is the local z-space advantage gradient. k sweeps the bias/variance trade: small k = local but
   noisy, k = N = the global constant gradient. Scored like the critic (within-state corr over 8 fresh actions).
2. Learning curve: the same with a random N/4, N/2 of the chunk (does accuracy still climb with data?).
3. Residual variant: the same regression on target - Q_critic(s, a) (what the chunk knows that the snapshot
   critic does not); critic + correction scored.
4. Linearity of the truth: split the 32 rollouts in halves A/B (independent noise); the true FD gradient at the mean
   action from half A predicts the fresh actions' true advantages of half B. Disattenuated by the split-half
   reliabilities of both sides. Tells how much of the within-state advantage a perfect local gradient explains.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "benchmarks/reppo_rootcause/data"
NAMES = ["v23_4M", "v23_12M", "v23_30M", "v29h8_8M", "v29h8_12M"]
RNG = np.random.default_rng(0)
BOOT = 1000


def adv_stats(pred, true):
    m = true.shape[2]
    centered = true - true.mean(1, keepdims=True)
    t = centered.mean(2)
    c = pred - pred.mean(1, keepdims=True)
    cov, var_c, var_t = (c * t).sum(), (c * c).sum(), (t * t).sum()
    signal = var_t - centered.var(2, ddof=1).sum() / m
    return np.array([cov / np.sqrt(var_c * var_t), cov / np.sqrt(var_c * signal) if signal > 0 else np.nan])


def boot(fn, *arrays):
    n = len(arrays[0])
    point = fn(*arrays)
    draws = np.array([fn(*(a[idx] for a in arrays)) for idx in RNG.integers(0, n, (BOOT, n))])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return point, lo, hi


def local_linear(order, z, y, k):
    """order (n, >=k) neighbor indices; returns (n, 6) local slopes of y on z with an intercept."""
    out = np.empty((len(order), z.shape[1]))
    for i, nb in enumerate(order[:, :k]):
        X = np.concatenate([np.ones((k, 1)), z[nb]], 1)
        out[i] = np.linalg.lstsq(X, y[nb], rcond=None)[0][1:]
    return out


def analyze(name, out):
    d = np.load(DATA / f"03_chunk_{name}.npz")
    obs, act, mean, std, y = (d[k].astype(np.float64) for k in ("obs", "action", "mean", "std", "target"))
    row, cq = d["row"], d["critic_q"].astype(np.float64)
    z = (np.arctanh(np.clip(act, -1 + 1e-7, 1 - 1e-7)) - mean) / std
    ret = d["ret_300"].astype(np.float64)
    m = ret.shape[2]
    fresh = slice(50, 58)
    s_obs, s_mean, s_std, s_act, s_row = (d[k].astype(np.float64) if k != "s_row" else d[k]
                                          for k in ("s_obs", "s_mean", "s_std", "s_actions", "s_row"))
    true_fresh = ret[:, fresh]
    z_fresh = (np.arctanh(np.clip(s_act[:, fresh], -1 + 1e-7, 1 - 1e-7)) - s_mean[:, None]) / s_std[:, None]
    crit_fresh = d["s_critic_q"][:, fresh]
    res = {}
    print(f"\n================ {name}")
    crit = boot(adv_stats, crit_fresh, true_fresh)
    print(f"critic                      adv corr corrected {crit[0][1]:+.3f} [{crit[1][1]:+.3f},{crit[2][1]:+.3f}]")
    res["critic"] = [float(crit[0][1]), float(crit[1][1]), float(crit[2][1])]

    # squared distances oracle states -> chunk states; own lane pushed to infinity
    d2 = (s_obs ** 2).sum(1)[:, None] + (obs ** 2).sum(1)[None] - 2.0 * s_obs @ obs.T
    d2[s_row[:, None] == row[None]] = np.inf
    order_full = np.argsort(d2, axis=1)[:, :8192]

    def run(order, target, label, ks):
        best = None
        for k in ks:
            g = local_linear(order, z, target, k)
            a_hat = np.einsum("nkd,nd->nk", z_fresh, g)
            point, lo, hi = boot(adv_stats, a_hat, true_fresh)
            res[f"{label}_k{k}"] = [float(point[1]), float(lo[1]), float(hi[1])]
            print(f"{label:22s} k={k:5d} adv corr corrected {point[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]  raw {point[0]:+.3f}")
            if best is None or point[1] > best[0]:
                best = (point[1], k, a_hat)
        return best

    ks = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    best = run(order_full, y, "kNN target", ks)
    # 3. residual on the critic, and critic + correction
    best_r = run(order_full, y - cq, "kNN target-critic", [128, 512, 2048, 8192])
    for k in (512, 2048):
        g = local_linear(order_full, z, y - cq, k)
        a_hat = crit_fresh + np.einsum("nkd,nd->nk", z_fresh, g)
        point, lo, hi = boot(adv_stats, a_hat, true_fresh)
        diff = boot(lambda a, c, t: adv_stats(a, t) - adv_stats(c, t), a_hat, crit_fresh, true_fresh)
        res[f"critic_plus_corr_k{k}"] = [float(point[1]), float(lo[1]), float(hi[1])]
        res[f"critic_plus_corr_minus_critic_k{k}"] = [float(diff[0][1]), float(diff[1][1]), float(diff[2][1])]
        print(f"critic + kNN correction k={k:5d} adv corr corrected {point[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]; "
              f"minus critic {diff[0][1]:+.3f} [{diff[1][1]:+.3f},{diff[2][1]:+.3f}]")
    # 2. learning curve at the best k fraction (k scales with N to hold the neighborhood radius)
    k_best = best[1]
    for frac in (0.25, 0.5):
        vals = []
        for rep in range(4):
            keep = np.zeros(len(y), dtype=bool)
            keep[RNG.choice(len(y), int(frac * len(y)), replace=False)] = True
            d2s = np.where(keep[None], d2, np.inf)
            kk = max(int(k_best * frac), 8)
            order = np.argpartition(d2s, kk, axis=1)[:, :kk]
            g = local_linear(order, z, y, kk)
            vals.append(adv_stats(np.einsum("nkd,nd->nk", z_fresh, g), true_fresh)[1])
        res[f"learning_curve_{frac}"] = [float(np.mean(vals)), float(np.std(vals))]
        print(f"learning curve N*{frac}: k={kk} (same radius) adv corr corrected {np.mean(vals):+.3f} (sd over 4 subsamples {np.std(vals):.3f})")
    print(f"learning curve N*1.0 : k={k_best} adv corr corrected {best[0]:+.3f}")

    # 4. linearity of the truth: FD gradient (half A) -> fresh advantage (half B)
    star = np.tanh(s_mean)
    jac = (1.0 - star ** 2) * s_std
    span = d["span_star"].astype(np.float64)
    halves = [np.arange(m) % 2 == 0, np.arange(m) % 2 == 1]
    for i, h in enumerate(d["fd_steps"]):
        plus, minus = ret[:, 1 + 12 * i:7 + 12 * i], ret[:, 7 + 12 * i:13 + 12 * i]
        fd = (plus - minus) / span[i][:, :, None] * jac[:, :, None]
        preds = [np.einsum("nkd,nd->nk", z_fresh, fd[:, :, hm].mean(2)) for hm in halves]
        preds = [p - p.mean(1, keepdims=True) for p in preds]
        truths = [true_fresh[:, :, hm].mean(2) for hm in halves]
        truths = [t - t.mean(1, keepdims=True) for t in truths]

        def stat(pa, pb, ta, tb):
            cross = 0.5 * ((pa * tb).sum() + (pb * ta).sum())
            rel_p, rel_t = (pa * pb).sum(), (ta * tb).sum()
            return np.array([cross / np.sqrt(rel_p * rel_t) if rel_p > 0 and rel_t > 0 else np.nan,
                             rel_p / np.sqrt((pa * pa).sum() * (pb * pb).sum()),
                             rel_t / np.sqrt((ta * ta).sum() * (tb * tb).sum()), cross / rel_p if rel_p > 0 else np.nan])
        point, lo, hi = boot(stat, preds[0], preds[1], truths[0], truths[1])
        res[f"linearity_h{h}"] = {"corr_disattenuated": [float(point[0]), float(lo[0]), float(hi[0])],
                                  "splithalf_rel_pred": float(point[1]), "splithalf_rel_truth": float(point[2]),
                                  "slope_truth_on_linear": [float(point[3]), float(lo[3]), float(hi[3])]}
        print(f"truth linearity h={h}: corr(true FD gradient . z, true fresh advantage) disattenuated {point[0]:+.3f} "
              f"[{lo[0]:+.3f},{hi[0]:+.3f}]; split-half reliabilities pred {point[1]:.3f} truth {point[2]:.3f}; "
              f"slope truth on linear prediction {point[3]:+.3f} [{lo[3]:+.3f},{hi[3]:+.3f}]")
    out[name] = res


def main():
    out = {}
    for name in (sys.argv[1:] or NAMES):
        analyze(name, out)
    path = DATA / "03_q1b_results.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    prior.update(out)
    path.write_text(json.dumps(prior, indent=1))


if __name__ == "__main__":
    main()

"""Red-team 08: is there a collinearity leak in the REAL critic? (oracle chunks, CPU, no training)

04's RC2 says the critic's action slope holds the cross-state association between "what the policy does here"
and "how good here is". Signature proposed there: the critic's within-state contrasts should line up with the
contrast predicted by the ACROSS-state regression of value on action. Test on held-out oracle states:

  assoc(s, a)  = w . (a - mean_fresh a),   w = across-state OLS slope of the critic's Q on the action
                 (two versions: raw, and after partialling linear obs out of both sides = Frisch-Waugh)
  critic(s, a) = Q(s, a) - mean_fresh Q      true(s, a) = oracle return - mean_fresh
  -> within-state correlations, and the multiple regression  critic ~ b_true * true + b_assoc * assoc.
A leak predicts b_assoc > 0 with corr(critic, assoc) well above corr(true, assoc). Also prints sd(mu)/sigma, the
ratio 04's leak time constant depends on. States are trimmed by critic within-state energy (top 5%) because one
flipped state otherwise owns every pooled statistic (02 C1 / 03 Q4).
"""
import numpy as np

rng = np.random.default_rng(0)
for tag in ("v23_4M", "v29h8_8M", "v29h8_12M", "v23_30M"):
    z = np.load(f"benchmarks/reppo_rootcause/data/03_chunk_{tag}.npz")
    obs, act, q, mean, std = (z[k].astype(np.float64) for k in ("obs", "action", "critic_q", "mean", "std"))
    ratio_pre = mean.std(0) / std.mean(0)
    print(f"\n=== {tag}: pre-tanh sigma mean {std.mean():.3f}; between-state sd(mu)/sigma per dim {np.round(ratio_pre, 2)} (mean {ratio_pre.mean():.2f});"
          f" action-space between/within sd {np.tanh(mean).std(0).mean():.2f}/{np.sqrt(((act - np.tanh(mean))**2).mean()):.2f}")
    X = np.c_[np.ones(len(obs)), obs]
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    w_raw = np.linalg.lstsq(np.c_[np.ones(len(act)), act], q, rcond=None)[0][1:]
    w_fw = np.linalg.lstsq(res(act), res(q), rcond=None)[0]           # action slope with linear state partialled out
    w_mu = np.linalg.lstsq(np.c_[np.ones(len(act)), np.tanh(mean)], q, rcond=None)[0][1:]  # pure association: Q on tanh(mu(s))
    fresh = slice(50, 58)
    a_f = z["s_actions"][:, fresh].astype(np.float64)                  # 512 x 8 x 6
    c = z["s_critic_q"][:, fresh]; c = c - c.mean(1, keepdims=True)
    ret = z["ret_300"][:, fresh].astype(np.float64)                    # 512 x 8 x 32
    t = ret.mean(-1); t = t - t.mean(1, keepdims=True)
    h1, h2 = ret[..., ::2].mean(-1), ret[..., 1::2].mean(-1)
    h1, h2 = h1 - h1.mean(1, keepdims=True), h2 - h2.mean(1, keepdims=True)
    keep = (c ** 2).sum(1) <= np.quantile((c ** 2).sum(1), 0.95)
    da = a_f - a_f.mean(1, keepdims=True)
    rel = (h1[keep] * h2[keep]).sum() / np.sqrt((h1[keep] ** 2).sum() * (h2[keep] ** 2).sum())  # split-half reliability of truth
    corr = lambda x, y: float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()))
    print(f"    truth split-half corr {rel:.2f};  critic-vs-truth within corr {corr(c[keep], t[keep]):+.3f} (raw, 5% trim);"
          f" critic sd {c[keep].std():.3f} true sd (noise-corrected) {np.sqrt(max((h1[keep]*h2[keep]).mean(), 0)):.3f}")
    for name, w in (("raw across-state slope of Q on a", w_raw), ("Frisch-Waugh (linear obs partialled)", w_fw), ("slope of Q on tanh(mu(s)) only", w_mu)):
        assoc = da @ w
        ck, tk, ak = c[keep].ravel(), t[keep].ravel(), assoc[keep].ravel()
        B = np.linalg.lstsq(np.c_[tk, ak], ck, rcond=None)[0]
        # bootstrap over states for the partial coefficient on assoc
        idx_keep = np.where(keep)[0]; bs = []
        for _ in range(400):
            ii = rng.choice(idx_keep, len(idx_keep))
            bs.append(np.linalg.lstsq(np.c_[t[ii].ravel(), assoc[ii].ravel()], c[ii].ravel(), rcond=None)[0][1])
        print(f"    {name:38s} |w| {np.linalg.norm(w):6.2f}  sd(assoc contrast) {ak.std():.3f}  corr(critic, assoc) {corr(ck, ak):+.3f}"
              f"  corr(true, assoc) {corr(tk, ak):+.3f}  partial b_assoc {B[1]:+.3f} [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]")

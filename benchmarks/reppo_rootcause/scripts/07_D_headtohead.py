"""D. Head-to-head from the SAME on-policy chunk: REPPO's critic action gradient vs PPO's score-function estimator,
each against the oracle's branching truth, aggregated in policy-relevant low-dimensional spaces.

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/07_D_headtohead.py [dataset ...]

Everything is a field g(s) in R^6: the gradient of E_z Q_soft(s, tanh(mu + sigma z)) with respect to the
pre-tanh mean mu(s), or an estimate of it at one on-policy sample (s_i, a_i, z_i):

    critic       dQ_c/da (s_i, a_i) * (1 - a_i^2)                 (REPPO's pathwise term at the sampled action)
    critic_star  dQ_c/da (s_i, a*)  * (1 - a*^2),  a* = tanh(mu)  (deterministic-policy-gradient variant)
    sf_L         A_hat_GAE(L)(s_i, a_i) * z_i / sigma_i           (PPO's score function, soft rewards, baseline U_c)
    sf_target    (trainer lambda-return target - U_c) * z_i / sigma_i
truth at the 512 oracle states (32 CRN rollouts each, horizon H):
    fd_a0        central difference of Q_true around a0 (h = 0.2, per coordinate) * (1 - a0^2)
    fresh        (1/8) sum_k (Q_true(s, a_k) - leave-one-out mean) z_k / sigma   (exact SF identity, 8 fresh actions)
    both         their average (independent action sets; both unbiased for the same field up to FD smoothing)

Aggregate V = mean_s g(s) (x) F(s) for feature maps F: const (6), linear in obs (108), the actor's last hidden
layer (= exact gradient of the mean head's weights, 3078), and the full actor parameter gradient J^T g.
The truth aggregate from 512 states is noisy, so nothing is divided by its norm directly: |V*|^2 is the
U-statistic over pairs of DIFFERENT oracle states (independent noise, unbiased), and the estimator aggregate
is built from chunk rows that are NOT oracle states. Reported per estimator:
    gain   = <V_est, V_true> / |V_est|          true first-order improvement per unit step along the estimate
    cos    = gain / sqrt(|V*|^2)                realized cosine with the true aggregate (disattenuated)
    ratio  = gain_critic / gain_sf              free of |V*|; the head-to-head
Bootstrap: oracle states (512) and chunk blocks (16 lanes x 16 time blocks) resampled independently, 2000 draws.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("common07", HERE / "07_common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

BOOTS = 2000
P3_GROUPS = 64  # oracle states are pooled into this many random groups for the full-parameter projection
TIME_BLOCKS = 16


def truth_fields(d, horizon):
    """Per-rollout truth fields at the oracle states: {name: (n, 6, M)} in pre-tanh-mean coordinates."""
    ret = d.returns[horizon]  # (n, 58, M)
    lay = d.layout
    a0 = d.s_actions[:, lay["a0_probes"][0]]
    star = d.s_actions[:, lay["star_probes"][0]]
    out = {}
    for which, h in enumerate(d.fd_steps):
        for key, lo, centre, span in (("a0", lay["a0_probes"][0], a0, d.span_a0), ("star", lay["star_probes"][0], star, d.span_star)):
            block = ret[:, lo:lo + 25]
            plus = 1 + 12 * which
            fd = (block[:, plus:plus + 6] - block[:, plus + 6:plus + 12]) / span[which][:, :, None]
            out[f"fd_{key}_h{h:g}"] = fd * (1.0 - centre ** 2)[:, :, None]
    fresh_actions = d.s_actions[:, lay["fresh"][0]:lay["fresh"][1]]  # (n, K, 6)
    mean, std = d.mean[d.index], d.std[d.index]
    z = (np.arctanh(np.clip(fresh_actions, -1 + 1e-7, 1 - 1e-7)) - mean[:, None]) / std[:, None]
    q = ret[:, lay["fresh"][0]:lay["fresh"][1]]  # (n, K, M)
    k = q.shape[1]
    loo = (q.sum(1, keepdims=True) - q) / (k - 1)
    adv = q - loo
    # The leave-one-out baseline of action k is independent of z_k, so E[adv_k z_k / sigma] = g exactly.
    out["fresh"] = np.einsum("nkm,nkd->ndm", adv, z / std[:, None]) / k
    out["both"] = 0.5 * (out["fd_a0_h0.2"] + out["fresh"])
    return out, z


def estimator_fields(d):
    jac = 1.0 - d.action ** 2
    star = np.tanh(d.mean)
    _, grad_star = common.critic_grad(d.critic, d.obs, torch.as_tensor(star, dtype=torch.float32))
    score = d.z / d.std
    fields = {"critic": d.grad * jac, "critic_star": grad_star.double().numpy() * (1.0 - star ** 2)}
    for lam in (0.0, 0.8, 0.95, 0.99):
        fields[f"sf_gae{lam:g}"] = common.soft_gae(d, lam)[:, None] * score
    fields["sf_target"] = (d.target - d.u_c)[:, None] * score
    fields["sf_criticA"] = (d.q - d.u_c)[:, None] * score  # the critic's own advantage pushed through the score function
    return fields


def feature_maps(d):
    with torch.no_grad():
        hidden = d.actor.net[:-1](d.obs).double().numpy()
    ones = np.ones((len(hidden), 1))
    return {"const(6)": ones, "linear(108)": np.concatenate([ones, d.obs.double().numpy()], 1),
            "lastlayer(3078)": np.concatenate([ones, hidden], 1)}


def contributions(field, features, groups, count):
    """Group sums of g(s) (x) F(s): (count, 6 P)."""
    per = np.einsum("nd,np->ndp", field, features).reshape(len(field), -1)
    out = np.zeros((count, per.shape[1]))
    np.add.at(out, groups, per)
    return out


def param_contributions(actor, obs, field, groups, count):
    """Group sums of J_mu(s)^T g(s) over all actor-net parameters: (count, n_params) float32."""
    params = [p for p in actor.net.parameters()]
    for p in params:
        p.requires_grad_(True)
    rows = []
    field_t = torch.as_tensor(field, dtype=torch.float32)
    groups_t = torch.as_tensor(groups)
    for g in range(count):
        sel = torch.nonzero(groups_t == g).flatten()
        if sel.numel() == 0:
            rows.append(torch.zeros(sum(p.numel() for p in params)))
            continue
        mean, _ = actor(obs[sel])
        grads = torch.autograd.grad((mean * field_t[sel]).sum(), params, allow_unused=True)
        rows.append(torch.cat([(g_ if g_ is not None else torch.zeros_like(p)).flatten() for g_, p in zip(grads, params)]))
    for p in params:
        p.requires_grad_(False)
    return torch.stack(rows).numpy()


def stats_from_grams(k_tt, k_et, k_ee, rng, boots=BOOTS):
    """gain, cos, own reliability for one estimator against one truth; returns point and bootstrap arrays."""
    n, b = k_tt.shape[0], k_ee.shape[0]

    def compute(w, v):
        sw, sv = w.sum(), v.sum()
        s_true = (w @ k_tt @ w - (w ** 2 * np.diag(k_tt)).sum()) / (sw ** 2 - (w ** 2).sum())
        cross = v @ k_et @ w / (sv * sw)
        norm_e = np.sqrt(max(v @ k_ee @ v, 0.0)) / sv
        s_est = (v @ k_ee @ v - (v ** 2 * np.diag(k_ee)).sum()) / (sv ** 2 - (v ** 2).sum())
        return cross / norm_e, s_true, s_est / norm_e ** 2, cross
    point = compute(np.ones(n), np.ones(b))
    draws = np.array([compute(rng.multinomial(n, np.ones(n) / n).astype(float), rng.multinomial(b, np.ones(b) / b).astype(float))
                      for _ in range(boots)])
    return point, draws


def fmt(value, lo, hi, spec="+.3f"):
    return f"{value:{spec}} [{lo:{spec}}, {hi:{spec}}]"


def run(name, log):
    d = common.load(name)
    print(f"\n===== {name}: step {d.meta['global_step']} lam {d.lam} T {d.temperature:.4f} | recon {d.recon}", file=log, flush=True)
    est = estimator_fields(d)
    maps = feature_maps(d)
    rng = np.random.default_rng(0)
    is_oracle = np.zeros(len(d.q), bool)
    is_oracle[d.index] = True
    lane, step = d.row, np.arange(len(d.q)) // common.NUM_ENVS
    block = lane * TIME_BLOCKS + step * TIME_BLOCKS // common.NUM_STEPS
    n_blocks = common.NUM_ENVS * TIME_BLOCKS
    results = {}
    subsets = {"all": np.ones(len(d.q), bool)}
    if not d.upright.all():
        subsets["upright"] = d.upright
    # Sanity: sizes of the fields (rms per coordinate, pre-tanh-mean units).
    print("field rms: " + ", ".join(f"{k} {np.sqrt((v ** 2).mean()):.3f}" for k, v in est.items()), file=log)
    est_cache = {}
    for horizon in (300, 100):
        truth, _ = truth_fields(d, horizon)
        m = next(iter(truth.values())).shape[-1]
        print(f"\n--- truth horizon {horizon}: per-state signal share (rollout split-half) "
              + ", ".join(f"{k} {np.mean(v[..., ::2].mean(-1) * v[..., 1::2].mean(-1)) / np.mean(v.mean(-1) ** 2):.3f}"
                          for k, v in truth.items()), file=log)
        for sub_name, sub in subsets.items():
            o_keep = sub[d.index]
            o_idx = d.index[o_keep]
            chunk_keep = sub & ~is_oracle
            for map_name in list(maps) + ["actor_params(full)"]:
                grams_truth, grams_est = {}, {}
                if map_name.startswith("actor_params"):
                    perm = np.random.default_rng(1).permutation(len(o_idx)) % P3_GROUPS
                    c_truth = {k: param_contributions(d.actor, d.obs[o_idx], v[o_keep].mean(-1), perm, P3_GROUPS)
                               for k, v in truth.items() if k in ("fd_a0_h0.2", "fresh", "both")}
                    if (sub_name, map_name) not in est_cache:
                        est_cache[sub_name, map_name] = {
                            k: param_contributions(d.actor, d.obs[chunk_keep], v[chunk_keep], block[chunk_keep], n_blocks)
                            for k, v in est.items() if k in ("critic", "critic_star", "sf_gae0.95", "sf_gae0.8", "sf_target")}
                    c_est = est_cache[sub_name, map_name]
                else:
                    feats = maps[map_name]
                    c_truth = {k: contributions(v[o_keep].mean(-1), feats[o_idx], np.arange(len(o_idx)), len(o_idx))
                               for k, v in truth.items()}
                    if (sub_name, map_name) not in est_cache:
                        est_cache[sub_name, map_name] = {k: contributions(v[chunk_keep], feats[chunk_keep], block[chunk_keep], n_blocks)
                                                         for k, v in est.items()}
                    c_est = est_cache[sub_name, map_name]
                print(f"\n[{name} H{horizon} {sub_name} | {map_name}] oracle units {len(next(iter(c_truth.values())))}, "
                      f"chunk rows {int(chunk_keep.sum())}", file=log)
                for t_name in ("both", "fd_a0_h0.2", "fresh"):
                    ct = c_truth[t_name]
                    k_tt = ct @ ct.T
                    cell = {}
                    draws_all = {}
                    for e_name, ce in c_est.items():
                        point, draws = stats_from_grams(k_tt, ce @ ct.T, ce @ ce.T, np.random.default_rng(42))
                        cell[e_name] = (point, draws)
                        draws_all[e_name] = draws
                    s_true = next(iter(cell.values()))[0][1]
                    s_draws = next(iter(draws_all.values()))[:, 1]
                    realized = np.trace(k_tt) / len(ct) ** 2  # E|V_true|^2 diagonal part: noise + state sampling
                    print(f"  truth={t_name}: |V*|^2 U-stat {s_true:.3e} [{np.percentile(s_draws, 2.5):.2e}, {np.percentile(s_draws, 97.5):.2e}] "
                          f"P(>0) {np.mean(s_draws > 0):.3f}; truth-aggregate reliability {s_true / (s_true + realized):.3f}", file=log)
                    for e_name, (point, draws) in cell.items():
                        gain, _, rel, _ = point
                        cos = gain / np.sqrt(s_true) if s_true > 0 else float("nan")
                        cos_draws = np.where(draws[:, 1] > 0, draws[:, 0] / np.sqrt(np.abs(draws[:, 1])), np.nan)
                        print(f"    {e_name:12s} gain {fmt(gain, *np.percentile(draws[:, 0], [2.5, 97.5]), '+.4f')} P(>0) {np.mean(draws[:, 0] > 0):.3f}"
                              f" | cos {fmt(cos, *np.nanpercentile(cos_draws, [2.5, 97.5]))} | own reliability {rel:+.3f}", file=log)
                        results[f"{horizon}|{sub_name}|{map_name}|{t_name}|{e_name}"] = {
                            "gain": gain, "gain_ci": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(), "cos": cos,
                            "cos_ci": np.nanpercentile(cos_draws, [2.5, 97.5]).tolist(), "s_true": s_true, "own_reliability": rel}
                    for ref in ("sf_gae0.95", "sf_target"):
                        if ref in cell and "critic" in cell:
                            diff = cell["critic"][1][:, 0] - cell[ref][1][:, 0]
                            ratio = cell["critic"][1][:, 0] / cell[ref][1][:, 0]
                            print(f"    critic - {ref}: gain diff {fmt(cell['critic'][0][0] - cell[ref][0][0], *np.percentile(diff, [2.5, 97.5]), '+.4f')}"
                                  f" P(critic > {ref}) {np.mean(diff > 0):.3f}; ratio {cell['critic'][0][0] / cell[ref][0][0]:+.2f} "
                                  f"(median {np.median(ratio):+.2f}, IQR {np.percentile(ratio, 25):+.2f}..{np.percentile(ratio, 75):+.2f})", file=log)
                            results[f"{horizon}|{sub_name}|{map_name}|{t_name}|critic_minus_{ref}"] = {
                                "diff": cell["critic"][0][0] - cell[ref][0][0], "ci": np.percentile(diff, [2.5, 97.5]).tolist(),
                                "p_critic_better": float(np.mean(diff > 0))}
                log.flush()
    return results


def main():
    names = sys.argv[1:] or list(common.DATASETS)
    out = {}
    with open(common.OUT / "07_D_log.txt", "a") as log:
        for name in names:
            out[name] = run(name, log)
            with open(common.OUT / f"07_D_results_{name}.json", "w") as handle:
                json.dump(out[name], handle, indent=1)


if __name__ == "__main__":
    main()

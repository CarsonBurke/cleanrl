"""Gate 1 (bet B1): does mixing the pathwise critic gradient with the score-function gradient, or re-weighting the
critic gradient per state, give a better estimate of the true projected policy gradient than the critic gradient alone?

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/14_gate1_hybrid.py [dataset ...]

Framework: 07_D3_combined.py (two unbiased truth views O = oracle branching, S = near-Monte-Carlo score function; every
product with the truth taken between different env lanes; bootstrap over chunk blocks and oracle units). Added here:

    (a) critic            dQ_c/da (1 - a^2) at the sampled action               (REPPO's pathwise term)
    (b) sf_gae0.95        A_hat_GAE(0.95) z / sigma                             (PPO's estimator)
    (c) unit_sum          V_a / |V_a| + V_b / |V_b|, norms of the realized full-chunk aggregates
        unit_sum_mb       the same with the norms a 2048-row minibatch would have (B1 normalises per actor step)
    (d) opt_half          w_c V_a + w_s V_b, w = argmax predicted cosine (non-negative), fitted on 8 lanes and scored on the
                          other 8 against the other 8 lanes' truth only; compared with the critic on the same 8 lanes
        opt_insample      the same fitted and scored on all lanes (optimistic upper bound)
    (e) crit_norm_*       every state's gradient rescaled to the batch rms norm (pre: the pre-tanh 6-vector the actor
                          consumes; act: dQ/da rescaled, then multiplied by 1 - a^2)
    (f) crit_clip_*       per-state norms clipped at their 99th percentile (same two coordinate systems)
        crit_drop1        top 1% of states by pre-tanh norm removed
    controls              critic rows permuted; score-function rows permuted
Error correlation between (a) and (b):
    rho_noise  correlation of their same-chunk fluctuations = (G_ab - X_ab) / sqrt((G_aa - X_aa)(G_bb - X_bb)),
               G realized (same rows) and X cross-lane products; needs no truth.
    rho_error  partial correlation given the truth = (cos_ab - cos_a cos_b) / sqrt((1 - cos_a^2)(1 - cos_b^2)); needs |V*|^2.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load("common07", "07_common.py")
d1 = _load("d1", "07_D_headtohead.py")
d2 = _load("d2", "07_D2_chunk_reference.py")
BOOTS, HORIZON, SPLITS, MINIBATCH = 2000, 100, 8, 2048
SF = "sf_gae0.95"
LINEAR = ("critic", SF, "sf_gae0.8", "crit_norm_pre", "crit_clip_pre", "crit_norm_act", "crit_clip_act", "crit_drop1",
          "critic_perm", "sf_perm")
PAIRS = (("critic", SF), ("critic", "sf_gae0.8"), ("crit_norm_pre", SF), ("crit_clip_pre", SF), ("critic_perm", SF),
         ("critic", "sf_perm"))
REFS = ("sf_gae0.99", "sf_gae0.95")


def reweighted(field, mode):
    norm = np.linalg.norm(field, axis=1)
    if mode == "norm":
        scale = np.sqrt(np.mean(norm ** 2)) / np.maximum(norm, 1e-12)
    elif mode == "clip":
        scale = np.minimum(1.0, np.quantile(norm, 0.99) / np.maximum(norm, 1e-12))
    else:
        scale = (norm <= np.quantile(norm, 0.99)).astype(float)
    return field * scale[:, None]


def fields(d):
    est = d2.fields(d)
    jac = 1.0 - d.action ** 2
    rng = np.random.default_rng(0)
    est["crit_norm_pre"], est["crit_clip_pre"] = reweighted(est["critic"], "norm"), reweighted(est["critic"], "clip")
    est["crit_norm_act"], est["crit_clip_act"] = reweighted(d.grad, "norm") * jac, reweighted(d.grad, "clip") * jac
    est["crit_drop1"] = reweighted(est["critic"], "drop")
    est["critic_perm"] = est["critic"][rng.permutation(len(jac))]
    est["sf_perm"] = est[SF][rng.permutation(len(jac))]
    return est


def quad(left, kernel, right):
    return ((left @ kernel) * right).sum(1)


class Space:
    """Bootstrap-ready kernels of one projection space."""

    def __init__(self, c_chunk, n_chunk, lane_chunk, c_oracle, n_oracle, lane_oracle):
        self.c, self.n, self.lane = c_chunk, n_chunk, lane_chunk
        self.co, self.no, self.lane_o = c_oracle, n_oracle, lane_oracle
        self.diff_cc = (lane_chunk[:, None] != lane_chunk[None, :]).astype(float)
        self.diff_co = (lane_chunk[:, None] != lane_oracle[None, :]).astype(float)
        self.diff_oo = (lane_oracle[:, None] != lane_oracle[None, :]).astype(float)
        self.k_oo, self.k_sym = c_oracle @ c_oracle.T * self.diff_oo, {}
        self.k_eo = {k: c_chunk[k] @ c_oracle.T * self.diff_co for k in LINEAR + REFS}
        self.k_gram = {(x, y): c_chunk[x] @ c_chunk[y].T for x, y in tuple((k, k) for k in LINEAR) + PAIRS}

    def sym(self, x, y):
        if (x, y) not in self.k_sym:
            kernel = self.c[x] @ self.c[y].T
            self.k_sym[x, y] = 0.5 * (kernel + kernel.T) * self.diff_cc
        return self.k_sym[x, y]

    def moments(self, v, w, ref, weights=None, names=LINEAR, naive=False):
        """Per-draw truth norm b, alignments a[name], realized Grams g[(x, y)], cross-lane products x[(x, y)].

        v (draws, blocks), w (draws, oracle units): bootstrap weights, already masked to a lane subset if wanted.
        `weights` fixes the O / S mixing weights (so that halves reuse the full-sample choice).
        Realized Grams: a resampled block set carries its own resampling noise on top of the sampling noise, which
        doubles the diagonal (same-block) part of v K v (E v_i^2 = 2 - 1/n) and so inflates the norm of noisy
        estimators in every draw (07 did this: `naive`). The default replaces v_i^2 by v_i on the diagonal, which
        keeps the draw's expected norm at the one-rollout value while the cross-block part still varies."""
        vn, wn = v * self.n, w * self.no
        den_cc, den_co, den_oo = quad(vn, self.diff_cc, vn), quad(vn, self.diff_co, wn), quad(wn, self.diff_oo, wn)
        total = vn.sum(1)
        used = {} if weights is None else weights

        def mix(key, parts):
            if key not in used:
                var = np.array([np.var(p[1:]) for p in parts])
                used[key] = (1.0 / var) / (1.0 / var).sum()
            return sum(wi * p for wi, p in zip(used[key], parts))
        b = mix("b", [quad(w, self.k_oo, w) / den_oo, quad(v, self.k_eo[ref], w) / den_co, quad(v, self.sym(ref, ref), v) / den_cc])
        a = {k: mix(k, [quad(v, self.k_eo[k], w) / den_co, quad(v, self.sym(k, ref), v) / den_cc]) for k in names}
        excess = np.zeros_like(v) if naive else v ** 2 - v
        g = {key: (quad(v, kernel, v) - excess @ np.diag(kernel)) / total ** 2
             for key, kernel in self.k_gram.items() if key[0] in names and key[1] in names}
        x = {key: quad(v, self.sym(*key), v) / den_cc for key in (("critic", "critic"), (SF, SF), ("critic", SF))}
        return b, a, g, x, used


def best_weights(g_cc, g_ss, g_cs, a_c, a_s):
    """Non-negative weights maximising a.w / sqrt(w G w), per draw."""
    det = g_cc * g_ss - g_cs ** 2
    w_c, w_s = (g_ss * a_c - g_cs * a_s) / det, (g_cc * a_s - g_cs * a_c) / det
    feasible = (w_c >= 0) & (w_s >= 0) & (det > 0)
    critic_better = a_c / np.sqrt(g_cc) >= a_s / np.sqrt(g_ss)
    w_c = np.where(feasible, w_c, np.where(critic_better, 1.0, 0.0))
    w_s = np.where(feasible, w_s, np.where(critic_better, 0.0, 1.0))
    return w_c, w_s


def mixed(w_c, w_s, a, g, pair):
    """Gain of w_c V_x + w_s V_y."""
    x, y = pair
    return (w_c * a[x] + w_s * a[y]) / np.sqrt(w_c ** 2 * g[x, x] + 2 * w_c * w_s * g[x, y] + w_s ** 2 * g[y, y])


def summary(values, reference=None):
    out = {"point": float(values[0]), "ci": np.nanpercentile(values[1:], [2.5, 97.5]).tolist(),
           "p_pos": float(np.nanmean(values[1:] > 0))}
    if reference is not None:
        diff = values - reference
        out |= {"diff": float(diff[0]), "diff_ci": np.nanpercentile(diff[1:], [2.5, 97.5]).tolist(),
                "p_better": float(np.nanmean(diff[1:] > 0))}
    return out


def analyse(space, ref, log, tag, results, naive=False):
    rng = np.random.default_rng(11)
    nb, no = len(space.lane), len(space.lane_o)
    v = np.concatenate([np.ones((1, nb)), rng.multinomial(nb, np.ones(nb) / nb, size=BOOTS)]).astype(float)
    w = np.concatenate([np.ones((1, no)), rng.multinomial(no, np.ones(no) / no, size=BOOTS)]).astype(float)
    b, a, g, x, used = space.moments(v, w, ref, naive=naive)
    root_b = np.where(b > 0, np.sqrt(np.abs(b)), np.nan)
    gains = {k: a[k] / np.sqrt(g[k, k]) for k in LINEAR}
    for label, pair in (("unit_sum", ("critic", SF)), ("unit_sum_gae0.8", ("critic", "sf_gae0.8")),
                        ("unit_sum_normpre", ("crit_norm_pre", SF)), ("unit_sum_clippre", ("crit_clip_pre", SF)),
                        ("unit_sum_critperm", ("critic_perm", SF)), ("unit_sum_sfperm", ("critic", "sf_perm"))):
        gains[label] = mixed(1 / np.sqrt(g[pair[0], pair[0]]), 1 / np.sqrt(g[pair[1], pair[1]]), a, g, pair)
    # B1 normalises each 2048-row minibatch gradient: noise energy is rows / MINIBATCH times the full-chunk value.
    inflate = space.n.sum() / MINIBATCH
    mb_norm = {k: np.sqrt(np.maximum(x[k, k] + inflate * (g[k, k] - x[k, k]), 1e-30)) for k in ("critic", SF)}
    gains["unit_sum_mb"] = mixed(1 / mb_norm["critic"], 1 / mb_norm[SF], a, g, ("critic", SF))
    w_c, w_s = best_weights(g["critic", "critic"], g[SF, SF], g["critic", SF], a["critic"], a[SF])
    gains["opt_insample"] = mixed(w_c, w_s, a, g, ("critic", SF))
    # (d) honest version: weights from one half of the lanes, scored on the other half against that half's truth.
    split_rng = np.random.default_rng(3)
    half = {k: np.zeros(BOOTS + 1) for k in ("opt_half", "critic_half", "sf_half", "unit_sum_half")}
    share = np.zeros(BOOTS + 1)
    for _ in range(SPLITS):
        first = np.isin(np.arange(common.NUM_ENVS), split_rng.permutation(common.NUM_ENVS)[:common.NUM_ENVS // 2])
        for fit in (first, ~first):
            _, a_f, g_f, _, _ = space.moments(v * fit[space.lane], w * fit[space.lane_o], ref, used, ("critic", SF), naive)
            _, a_s, g_s, _, _ = space.moments(v * ~fit[space.lane], w * ~fit[space.lane_o], ref, used, ("critic", SF), naive)
            h_c, h_s = best_weights(g_f["critic", "critic"], g_f[SF, SF], g_f["critic", SF], a_f["critic"], a_f[SF])
            half["opt_half"] += mixed(h_c, h_s, a_s, g_s, ("critic", SF))
            half["critic_half"] += a_s["critic"] / np.sqrt(g_s["critic", "critic"])
            half["sf_half"] += a_s[SF] / np.sqrt(g_s[SF, SF])
            half["unit_sum_half"] += mixed(1 / np.sqrt(g_s["critic", "critic"]), 1 / np.sqrt(g_s[SF, SF]), a_s, g_s, ("critic", SF))
            unit_c, unit_s = h_c * np.sqrt(g_f["critic", "critic"]), h_s * np.sqrt(g_f[SF, SF])
            share += unit_s / (unit_c + unit_s)
    half = {k: val / (2 * SPLITS) for k, val in half.items()}
    share /= 2 * SPLITS

    print(f"\n[{tag} | ref {ref}] |V*|^2 {b[0]:.3e} [{np.percentile(b[1:], 2.5):.2e}, {np.percentile(b[1:], 97.5):.2e}] P(>0) {np.mean(b[1:] > 0):.3f}", file=log)
    for name, value in gains.items():
        s = summary(value, gains["critic"])
        cos = summary(value / root_b)
        results[f"{tag}|{ref}|{name}"] = {"gain": s, "cos": cos}
        print(f"    {name:18s} gain {s['point']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] P(>0) {s['p_pos']:.3f} | cos {cos['point']:+.3f} "
              f"[{cos['ci'][0]:+.3f}, {cos['ci'][1]:+.3f}] | vs critic {s['diff']:+.4f} [{s['diff_ci'][0]:+.4f}, {s['diff_ci'][1]:+.4f}] "
              f"P(better) {s['p_better']:.3f}", file=log)
    for name in ("critic_half", "sf_half", "unit_sum_half", "opt_half"):
        s = summary(half[name], half["critic_half"])
        results[f"{tag}|{ref}|{name}"] = {"gain": s}
        print(f"    {name:18s} gain {s['point']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] P(>0) {s['p_pos']:.3f} | 8-lane halves "
              f"| vs critic_half {s['diff']:+.4f} [{s['diff_ci'][0]:+.4f}, {s['diff_ci'][1]:+.4f}] P(better) {s['p_better']:.3f}", file=log)
    print(f"    opt_half: score-function share of the unit-norm-scaled weights {share[0]:.2f} [{np.percentile(share[1:], 2.5):.2f}, "
          f"{np.percentile(share[1:], 97.5):.2f}]; in-sample {(w_s * np.sqrt(g[SF, SF]) / (w_c * np.sqrt(g['critic', 'critic']) + w_s * np.sqrt(g[SF, SF])))[0]:.2f}", file=log)

    noise = {k: g[k, k] - x[k, k] for k in ("critic", SF)}
    rho_noise = (g["critic", SF] - x["critic", SF]) / np.sqrt(np.where((noise["critic"] > 0) & (noise[SF] > 0), noise["critic"] * noise[SF], np.nan))
    cos_c, cos_s = gains["critic"] / root_b, gains[SF] / root_b
    cos_cs = g["critic", SF] / np.sqrt(g["critic", "critic"] * g[SF, SF])
    valid = (cos_c ** 2 < 1) & (cos_s ** 2 < 1)
    rho_error = np.where(valid, (cos_cs - cos_c * cos_s) / np.sqrt(np.where(valid, (1 - cos_c ** 2) * (1 - cos_s ** 2), np.nan)), np.nan)
    independent = (cos_c + cos_s) / np.sqrt(2 + 2 * cos_c * cos_s)  # unit-sum cosine if the two errors were uncorrelated
    bias_cos = x["critic", SF] / np.sqrt(np.where((x["critic", "critic"] > 0) & (x[SF, SF] > 0), x["critic", "critic"] * x[SF, SF], np.nan))
    rows = {"cos_realized_critic_sf": cos_cs, "noise_share_critic": noise["critic"] / g["critic", "critic"],
            "noise_share_sf": noise[SF] / g[SF, SF], "rho_noise": rho_noise, "rho_error": rho_error,
            "expectation_cos_critic_sf": bias_cos, "unit_sum_cos_if_independent": independent,
            "unit_sum_cos_measured": gains["unit_sum"] / root_b}
    for name, value in rows.items():
        s = summary(value)
        results[f"{tag}|{ref}|{name}"] = s
        print(f"    {name:28s} {s['point']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  (valid draws {np.mean(np.isfinite(value[1:])):.2f})", file=log)
    log.flush()


def run(name, log):
    d = common.load(name, threads=3)
    est = fields(d)
    maps = d1.feature_maps(d)
    truth = d1.truth_fields(d, HORIZON)[0]["both"].mean(-1)
    lane, step = d.row, np.arange(len(d.q)) // common.NUM_ENVS
    norm_pre, norm_act = np.linalg.norm(est["critic"], axis=1), np.linalg.norm(d.grad, axis=1)
    print(f"\n===== {name}: step {d.meta['global_step']} | per-state |g| pre-tanh: top-1% energy share "
          f"{np.sort(norm_pre ** 2)[-len(norm_pre) // 100:].sum() / (norm_pre ** 2).sum():.3f}, post-tanh {np.sort(norm_act ** 2)[-len(norm_act) // 100:].sum() / (norm_act ** 2).sum():.3f}; "
          f"p50 / p99 / max pre {np.median(norm_pre):.3f} / {np.quantile(norm_pre, 0.99):.3f} / {norm_pre.max():.3f}", file=log, flush=True)
    results = {}
    o_idx = d.index
    for map_name in ("linear(108)", "lastlayer(3078)", "actor_params(full)"):
        full = map_name.startswith("actor_params")
        tb = 4 if full else 16
        block = lane * tb + step * tb // common.NUM_STEPS
        lane_chunk = np.repeat(np.arange(common.NUM_ENVS), tb)
        n_chunk = np.bincount(block, minlength=len(lane_chunk)).astype(float)
        names = LINEAR + tuple(r for r in REFS if r not in LINEAR)
        if full:
            rng = np.random.default_rng(5)
            group = lane[o_idx] * 4 + rng.integers(0, 4, len(o_idx))
            lane_oracle, n_oracle = np.repeat(np.arange(common.NUM_ENVS), 4), np.bincount(group, minlength=64).astype(float)
            c_oracle = d1.param_contributions(d.actor, d.obs[o_idx], truth, group, 64).astype(np.float64)
            c_chunk = {k: d1.param_contributions(d.actor, d.obs, est[k], block, len(lane_chunk)).astype(np.float64) for k in names}
        else:
            feats = maps[map_name]
            lane_oracle, n_oracle = lane[o_idx], np.ones(len(o_idx))
            c_oracle = d1.contributions(truth, feats[o_idx], np.arange(len(o_idx)), len(o_idx))
            c_chunk = {k: d1.contributions(est[k], feats, block, len(lane_chunk)) for k in names}
        space = Space(c_chunk, n_chunk, lane_chunk, c_oracle, n_oracle, lane_oracle)
        for ref in REFS:
            analyse(space, ref, log, f"{name}|{map_name}", results)
            analyse(space, ref, log, f"{name}|{map_name}|naive-norm", results, naive=True)
    return results


def main():
    names = sys.argv[1:] or list(common.DATASETS)
    for name in names:
        with open(common.OUT / "14_gate1_log.txt", "a") as log:
            result = run(name, log)
        with open(common.OUT / f"14_gate1_results_{name}.json", "w") as handle:
            json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()

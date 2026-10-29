"""D (final arm). Critic gradient vs PPO score function against a COMBINED unbiased truth aggregate.

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/07_D3_combined.py [dataset ...]

Two unbiased views of the true aggregate V* = E_s[g_true(s) (x) F(s)] exist per dataset:
    O  the oracle branching truth at 512 states (H = 100, mean of the a0 finite difference and the fresh-action SF),
    S  the chunk's score-function estimator with a near-Monte-Carlo return (GAE lam_ref = 0.99; 0.95 and 1 as checks).
Every inner product below is taken ONLY between units in different env lanes (independent trajectories), so no
product shares state sampling, action noise or return noise:
    b = |V*|^2          from X(O,O), X(O,S), X(S,S)      (inverse-variance weighted; each is unbiased alone)
    a_E = <E[V_E], V*>  from X(E,O), X(E,S)
    cos_E = a_E / (|V_E| sqrt(b))     |V_E| = realized norm of the full-chunk aggregate (what one rollout delivers)
    kappa = cos_critic * cos_sf = a_critic / (|V_critic| |V_sf|) when sf is unbiased: free of b
Bootstrap: oracle units and chunk blocks resampled independently (2000 draws), lane labels kept.
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
BOOTS = 2000
ESTIMATORS = ("critic", "critic_star", "sf_gae0.8", "sf_gae0.95", "sf_gae0.99", "sf_target", "cv_combo")
HORIZON = 100


def cross(weights_a, kernel, weights_b, counts_a, counts_b, different):
    """Mean product over pairs in different lanes, per-row units: (boots,)."""
    num = np.einsum("na,ab,nb->n", weights_a, kernel * different, weights_b)
    den = np.einsum("na,ab,nb->n", weights_a * counts_a, different.astype(float), weights_b * counts_b)
    return num / den


def combine(estimates):
    """Inverse-variance weighted mean of unbiased estimates (list of (boots + 1,) arrays; row 0 = point)."""
    var = np.array([np.var(e[1:]) for e in estimates])
    w = (1.0 / var) / (1.0 / var).sum()
    return sum(wi * e for wi, e in zip(w, estimates)), w


def analyse(c_chunk, n_chunk, lane_chunk, c_oracle, n_oracle, lane_oracle, ref, log, tag, results):
    rng = np.random.default_rng(11)
    nb, no = len(lane_chunk), len(lane_oracle)
    v = np.concatenate([np.ones((1, nb)), rng.multinomial(nb, np.ones(nb) / nb, size=BOOTS)]).astype(float)
    w = np.concatenate([np.ones((1, no)), rng.multinomial(no, np.ones(no) / no, size=BOOTS)]).astype(float)
    diff_cc = lane_chunk[:, None] != lane_chunk[None, :]
    diff_co = lane_chunk[:, None] != lane_oracle[None, :]
    diff_oo = lane_oracle[:, None] != lane_oracle[None, :]
    s = c_chunk[ref]
    b_parts = [cross(w, c_oracle @ c_oracle.T, w, n_oracle, n_oracle, diff_oo),
               cross(v, s @ c_oracle.T, w, n_chunk, n_oracle, diff_co),
               cross(v, s @ s.T, v, n_chunk, n_chunk, diff_cc)]
    b, b_w = combine(b_parts)
    total = (v * n_chunk).sum(1)
    print(f"[{tag} | ref {ref}] |V*|^2: OO {b_parts[0][0]:.3e}, OS {b_parts[1][0]:.3e}, SS {b_parts[2][0]:.3e} -> combined {b[0]:.3e} "
          f"[{np.percentile(b[1:], 2.5):.2e}, {np.percentile(b[1:], 97.5):.2e}] P(>0) {np.mean(b[1:] > 0):.3f} weights {np.round(b_w, 2).tolist()}", file=log)
    cos, gains, norms = {}, {}, {}
    for name in ESTIMATORS:
        e = c_chunk[name]
        a_parts = [cross(v, e @ c_oracle.T, w, n_chunk, n_oracle, diff_co),
                   cross(v, 0.5 * (e @ s.T + s @ e.T), v, n_chunk, n_chunk, diff_cc)]
        a, a_w = combine(a_parts)
        norm = np.sqrt(np.einsum("na,ab,nb->n", v, e @ e.T, v)) / total
        gains[name], norms[name] = a / norm, norm
        cos[name] = np.where(b > 0, a / (norm * np.sqrt(np.abs(b))), np.nan)
        own = cross(v, e @ e.T, v, n_chunk, n_chunk, diff_cc)
        print(f"    {name:12s} cos {cos[name][0]:+.3f} [{np.nanpercentile(cos[name][1:], 2.5):+.3f}, {np.nanpercentile(cos[name][1:], 97.5):+.3f}] "
              f"| gain {gains[name][0]:+.4f} [{np.percentile(gains[name][1:], 2.5):+.4f}, {np.percentile(gains[name][1:], 97.5):+.4f}] P(>0) {np.mean(gains[name][1:] > 0):.3f} "
              f"| a via O {a_parts[0][0] / norm[0]:+.4f}, via S {a_parts[1][0] / norm[0]:+.4f} | |V| {norm[0]:.4f} own-rel {own[0] / norm[0] ** 2:+.3f}", file=log)
        results[f"{tag}|{ref}|{name}"] = {
            "cos": float(cos[name][0]), "cos_ci": np.nanpercentile(cos[name][1:], [2.5, 97.5]).tolist(), "gain": float(gains[name][0]),
            "gain_ci": np.percentile(gains[name][1:], [2.5, 97.5]).tolist(), "p_gain_pos": float(np.mean(gains[name][1:] > 0)),
            "norm": float(norm[0]), "own_reliability": float(own[0] / norm[0] ** 2)}
    results[f"{tag}|{ref}|b"] = {"b": float(b[0]), "ci": np.percentile(b[1:], [2.5, 97.5]).tolist(), "p_pos": float(np.mean(b[1:] > 0))}
    for other in ("sf_gae0.95", "sf_gae0.8", "sf_target"):
        diff = gains["critic"] - gains[other]
        kappa = gains["critic"] / norms[other]
        # SF-sample equivalent of the critic: N with cos_sf(N) = cos_critic, from SF noise = |V_sf|^2 - b at 16384 rows.
        noise = norms[other][0] ** 2 - b[0]
        c2 = cos["critic"][0] ** 2
        n_eq = 16384 * c2 / (1 - c2) * noise / b[0] if (b[0] > 0 and 0 < c2 < 1 and noise > 0) else float("nan")
        print(f"    critic vs {other}: gain diff {diff[0]:+.4f} [{np.percentile(diff[1:], 2.5):+.4f}, {np.percentile(diff[1:], 97.5):+.4f}] "
              f"P(critic better) {np.mean(diff[1:] > 0):.3f} | kappa=cos_c*cos_sf {kappa[0]:+.3f} [{np.percentile(kappa[1:], 2.5):+.3f}, "
              f"{np.percentile(kappa[1:], 97.5):+.3f}] | critic ~ {n_eq:.0f} SF samples", file=log)
        results[f"{tag}|{ref}|critic_vs_{other}"] = {
            "gain_diff": float(diff[0]), "ci": np.percentile(diff[1:], [2.5, 97.5]).tolist(), "p_critic_better": float(np.mean(diff[1:] > 0)),
            "kappa": float(kappa[0]), "kappa_ci": np.percentile(kappa[1:], [2.5, 97.5]).tolist(), "sf_sample_equivalent": float(n_eq)}
    log.flush()


def run(name, log):
    d = common.load(name)
    est = d2.fields(d)
    maps = d1.feature_maps(d)
    truth = d1.truth_fields(d, HORIZON)[0]["both"].mean(-1)
    lane, step = d.row, np.arange(len(d.q)) // common.NUM_ENVS
    print(f"\n===== {name}: step {d.meta['global_step']} lam {d.lam} upright {d.upright.mean():.4f} (oracle {d.upright[d.index].mean():.4f})", file=log, flush=True)
    results = {}
    subsets = {"all": np.ones(len(d.q), bool)}
    if not d.upright.all():
        subsets["upright"] = d.upright
    for sub_name, sub in subsets.items():
        o_keep = sub[d.index]
        o_idx = d.index[o_keep]
        for map_name in ["linear(108)", "lastlayer(3078)", "actor_params(full)"]:
            full = map_name.startswith("actor_params")
            tb = 4 if full else 16
            block = lane * tb + step * tb // common.NUM_STEPS
            lane_chunk = np.repeat(np.arange(common.NUM_ENVS), tb)
            n_chunk = np.bincount(block[sub], minlength=len(lane_chunk)).astype(float)
            if full:
                # oracle units: 4 random groups inside each lane (64 units), so every unit has one lane
                rng = np.random.default_rng(5)
                o_lane = lane[o_idx]
                group = o_lane * 4 + rng.integers(0, 4, len(o_idx))
                lane_oracle = np.repeat(np.arange(common.NUM_ENVS), 4)
                n_oracle = np.bincount(group, minlength=64).astype(float)
                c_oracle = d1.param_contributions(d.actor, d.obs[o_idx], truth[o_keep], group, 64).astype(np.float64)
                c_chunk = {k: d1.param_contributions(d.actor, d.obs[sub], est[k][sub], block[sub], len(lane_chunk)).astype(np.float64)
                           for k in ESTIMATORS}
            else:
                feats = maps[map_name]
                lane_oracle, n_oracle = lane[o_idx], np.ones(len(o_idx))
                c_oracle = d1.contributions(truth[o_keep], feats[o_idx], np.arange(len(o_idx)), len(o_idx))
                c_chunk = {k: d1.contributions(est[k][sub], feats[sub], block[sub], len(lane_chunk)) for k in ESTIMATORS}
            for ref in ("sf_gae0.99", "sf_gae0.95"):
                analyse(c_chunk, n_chunk, lane_chunk, c_oracle, n_oracle, lane_oracle, ref, log, f"{name}|{sub_name}|{map_name}", results)
    return results


def main():
    names = sys.argv[1:] or list(common.DATASETS)
    with open(common.OUT / "07_D3_log.txt", "a") as log:
        for name in names:
            result = run(name, log)
            with open(common.OUT / f"07_D3_results_{name}.json", "w") as handle:
                json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()

"""D (second arm). The same head-to-head, with the truth aggregate supplied by the chunk itself.

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/07_D2_chunk_reference.py [dataset ...]

The oracle's 512 states give an unbiased but low-powered truth aggregate (07_D_headtohead.py). A second unbiased
reference exists in the chunk: the score-function estimator with a (near-)Monte-Carlo return,
    R_i = A_hat_GAE(lam_ref)(s_i, a_i) z_i / sigma_i,  lam_ref in {0.95, 0.99, 1},
whose expectation is the true mean-gradient field for any baseline when lam_ref = 1 (the bootstrap at the chunk end
is a state value 0..1023 steps away). Lanes are independent trajectories, so for any estimator E
    X(E, R) = mean over pairs of blocks in DIFFERENT lanes of <C_E,b, C_R,b'>     is unbiased for <E[V_E], V*>,
    X(R, R)                                                                      is unbiased for |V*|^2,
and  cos(E, truth) = X(E, R) / (|V_E| sqrt(X(R, R))),  |V_E| the realized norm of the full-chunk aggregate (its noise
included: this is the cosine of the vector an optimizer would actually receive from one rollout). For E = R this is
sqrt(reliability). No oracle rollout is used; the oracle arm is the independent check.

Estimators as in 07_D_headtohead.py plus
    cv_combo = critic + sf_gae0.95 - sf_criticA     (critic gradient as a Stein control variate; unbiased like SF)
Bootstrap over 256 blocks (16 lanes x 16 time blocks; 64 for the full-parameter map), 2000 draws.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load("common07", "07_common.py")
d1 = _load("d1", "07_D_headtohead.py")
BOOTS = 2000
REFS = ("sf_gae0.95", "sf_gae0.99", "sf_gae1")
SHOW = ("critic", "critic_star", "sf_gae0.8", "sf_gae0.95", "sf_gae0.99", "sf_gae1", "sf_target", "cv_combo")


def fields(d):
    est = d1.estimator_fields(d)
    est["sf_gae1"] = common.soft_gae(d, 1.0)[:, None] * (d.z / d.std)
    est["cv_combo"] = est["critic"] + est["sf_gae0.95"] - est["sf_criticA"]
    return est


def analyse(contrib, lanes, rng, log, tag, results):
    """contrib: {name: (blocks, dim)}; lanes: (blocks,) lane of each block."""
    names = [n for n in SHOW if n in contrib]
    blocks = len(lanes)
    same = lanes[:, None] == lanes[None, :]
    weights = rng.multinomial(blocks, np.ones(blocks) / blocks, size=BOOTS).astype(float)
    weights = np.concatenate([np.ones((1, blocks)), weights])  # row 0 is the point estimate
    cross_norm = np.einsum("nb,bc,nc->n", weights, (~same).astype(float), weights)
    total = weights.sum(1)

    def quad(kernel):
        return np.einsum("nb,bc,nc->n", weights, kernel, weights)
    norm = {n: np.sqrt(np.maximum(quad(contrib[n] @ contrib[n].T), 0.0)) / total for n in names}
    out = {}
    for ref in REFS:
        s_ref = quad((contrib[ref] @ contrib[ref].T) * ~same) / cross_norm
        print(f"  ref={ref}: |V*|^2 = {s_ref[0]:.3e} [{np.percentile(s_ref[1:], 2.5):.2e}, {np.percentile(s_ref[1:], 97.5):.2e}] "
              f"P(>0) {np.mean(s_ref[1:] > 0):.3f}", file=log)
        cos = {}
        for n in names:
            kernel = contrib[n] @ contrib[ref].T
            x = quad(0.5 * (kernel + kernel.T) * ~same) / cross_norm
            gain = x / norm[n]
            cos[n] = np.where(s_ref > 0, gain / np.sqrt(np.abs(s_ref)), np.nan)
            signal = quad((contrib[n] @ contrib[n].T) * ~same) / cross_norm  # the estimator's own |E V_E|^2
            cos_signal = np.where((s_ref > 0) & (signal > 0), x / np.sqrt(np.abs(s_ref * signal)), np.nan)
            print(f"    {n:12s} cos {cos[n][0]:+.3f} [{np.nanpercentile(cos[n][1:], 2.5):+.3f}, {np.nanpercentile(cos[n][1:], 97.5):+.3f}]"
                  f" | gain {gain[0]:+.4f} [{np.percentile(gain[1:], 2.5):+.4f}, {np.percentile(gain[1:], 97.5):+.4f}] P(>0) {np.mean(gain[1:] > 0):.3f}"
                  f" | own reliability {signal[0] / norm[n][0] ** 2:+.3f} | bias-only cos {cos_signal[0]:+.3f} "
                  f"[{np.nanpercentile(cos_signal[1:], 2.5):+.3f}, {np.nanpercentile(cos_signal[1:], 97.5):+.3f}]", file=log)
            out[f"{tag}|{ref}|{n}"] = {"cos": float(cos[n][0]), "cos_ci": np.nanpercentile(cos[n][1:], [2.5, 97.5]).tolist(),
                                       "gain": float(gain[0]), "gain_ci": np.percentile(gain[1:], [2.5, 97.5]).tolist(),
                                       "own_reliability": float(signal[0] / norm[n][0] ** 2), "bias_only_cos": float(cos_signal[0]),
                                       "bias_only_ci": np.nanpercentile(cos_signal[1:], [2.5, 97.5]).tolist()}
        for other in ("sf_gae0.95", "sf_gae0.8"):
            diff = cos["critic"] - cos[other]
            print(f"    critic - {other}: cos diff {diff[0]:+.3f} [{np.nanpercentile(diff[1:], 2.5):+.3f}, {np.nanpercentile(diff[1:], 97.5):+.3f}] "
                  f"P(critic better) {np.nanmean(diff[1:] > 0):.3f}", file=log)
            out[f"{tag}|{ref}|critic_minus_{other}"] = {"diff": float(diff[0]), "ci": np.nanpercentile(diff[1:], [2.5, 97.5]).tolist(),
                                                         "p_critic_better": float(np.nanmean(diff[1:] > 0))}
    results.update(out)


def run(name, log):
    d = common.load(name)
    est = fields(d)
    maps = d1.feature_maps(d)
    lane, step = d.row, np.arange(len(d.q)) // common.NUM_ENVS
    print(f"\n===== {name}: step {d.meta['global_step']} lam {d.lam} | upright frac {d.upright.mean():.4f}", file=log, flush=True)
    results = {}
    subsets = {"all": np.ones(len(d.q), bool)}
    if not d.upright.all():
        subsets["upright"] = d.upright
    for sub_name, sub in subsets.items():
        for map_name in list(maps) + ["actor_params(full)"]:
            tb = 4 if map_name.startswith("actor_params") else 16
            block = lane * tb + step * tb // common.NUM_STEPS
            lanes = np.repeat(np.arange(common.NUM_ENVS), tb)
            if map_name.startswith("actor_params"):
                contrib = {k: d1.param_contributions(d.actor, d.obs[sub], v[sub], block[sub], len(lanes)) for k, v in est.items() if k in SHOW}
                contrib = {k: v.astype(np.float64) for k, v in contrib.items()}
            else:
                contrib = {k: d1.contributions(v[sub], maps[map_name][sub], block[sub], len(lanes)) for k, v in est.items() if k in SHOW}
            print(f"\n[{name} {sub_name} | {map_name}] rows {int(sub.sum())}", file=log)
            analyse(contrib, lanes, np.random.default_rng(3), log, f"{sub_name}|{map_name}", results)
            log.flush()
    return results


def main():
    names = sys.argv[1:] or list(common.DATASETS)
    with open(common.OUT / "07_D2_log.txt", "a") as log:
        for name in names:
            result = run(name, log)
            with open(common.OUT / f"07_D2_results_{name}.json", "w") as handle:
                json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()

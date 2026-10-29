"""Does curvature rescue the chunk-only estimators? Ridge on [z, z_d z_e - delta_de] x phi(s) (27 action features),
phi = const (27 dof) / [1, obs] (486 dof) / RF-128 (3942 dof); leave-env-rows-out; scored like the critic.
Also reports the cruising subset (upright, V >= 0.9 median) to check heavy-tail sensitivity."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("q1", HERE / "03_q1_estimation_ceiling.py")
q1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q1)
RNG = np.random.default_rng(2)
IU = np.triu_indices(6)


def action_features(z):
    quad = z[..., :, None] * z[..., None, :] - np.eye(6)
    return np.concatenate([z, quad[..., IU[0], IU[1]]], -1)


for name in (sys.argv[1:] or q1.NAMES):
    d = np.load(q1.DATA / f"03_chunk_{name}.npz")
    obs, act, mean, std, y = (d[k].astype(np.float64) for k in ("obs", "action", "mean", "std", "target"))
    row, s_row, s_obs = d["row"], d["s_row"], d["s_obs"].astype(np.float64)
    z = (np.arctanh(np.clip(act, -1 + 1e-7, 1 - 1e-7)) - mean) / std
    s_act, s_mean, s_std = (d[k].astype(np.float64) for k in ("s_actions", "s_mean", "s_std"))
    z_fresh = (np.arctanh(np.clip(s_act[:, 50:58], -1 + 1e-7, 1 - 1e-7)) - s_mean[:, None]) / s_std[:, None]
    true_fresh = d["ret_300"].astype(np.float64)[:, 50:58]
    crit = d["s_critic_q"][:, 50:58]
    cruising = d["cruising"]
    proj = (RNG.normal(size=(17, 128)) / np.sqrt(17), RNG.normal(size=128))
    proj_b = (RNG.normal(size=(17, 768)) / np.sqrt(17), RNG.normal(size=768))
    f, f_fresh = action_features(z), action_features(z_fresh)
    preds = {}
    for kind in ("const", "linear", "rf"):
        for label, cols in (("lin", slice(0, 6)), ("lin+quad", slice(0, 27))):
            a_hat = np.zeros((len(s_obs), 8))
            for fold in np.array_split(np.arange(16), 4):
                train = ~np.isin(row, fold)
                phi_b = q1.state_features(obs, "rf", proj_b)
                mu_b, sd_b = phi_b[train].mean(0), phi_b[train].std(0) + 1e-8
                phi_b = (phi_b - mu_b) / sd_b
                phi_b[:, 0] = 1.0
                resid = y - phi_b @ q1.cv_ridge(phi_b[train], y[train], row[train])[0]
                phi, phi_s = q1.state_features(obs, kind, proj), q1.state_features(s_obs, kind, proj)
                if kind != "const":
                    mu, sd = phi[train].mean(0), phi[train].std(0) + 1e-8
                    phi, phi_s = (phi - mu) / sd, (phi_s - mu) / sd
                    phi[:, 0], phi_s[:, 0] = 1.0, 1.0
                X = q1.z_design(f[:, cols], phi)
                w = q1.cv_ridge(X[train], resid[train], row[train], lambdas=np.array([1e-1, 10.0, 1e3, 1e5, 1e7]))[0]
                coef = phi_s @ w.reshape(f[:, cols].shape[1], -1).T  # (n, F)
                test = np.isin(s_row, fold)
                a_hat[test] = np.einsum("nkf,nf->nk", f_fresh[:, :, cols], coef)[test]
            preds[f"{kind}/{label}"] = a_hat
    print(f"\n=== {name}  (cruising states: {int(cruising.sum())}/512)")
    for key, a_hat in {"critic": crit, **preds}.items():
        full = q1.boot(q1.adv_stats, a_hat, true_fresh)
        sub = q1.boot(q1.adv_stats, a_hat[cruising], true_fresh[cruising])
        print(f"   {key:16s} all: {q1.fmt(full, 1):26s} cruising: {q1.fmt(sub, 1)}", flush=True)

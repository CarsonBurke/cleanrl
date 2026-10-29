"""Positive control for 03_q1_estimation_ceiling.py: plant a KNOWN advantage of the measured within-state variance
into the real lambda-return targets of the chunk and ask the same estimators to recover it (leave-env-rows-out).

Planted fields A_p(s, z) = z . g(s), var(A_p) = the oracle's noise-corrected within-state variance:
  const  : g constant                         (6 dof)
  linear : g = W [1, obs]                     (108 dof)
  smooth : g = random 2-layer tanh net of obs (smooth, not in any estimator's class)
  rough  : g = sum of 6 x 2000 random Fourier features of obs at 4x the frequency (many dof)
Recovery corr is exact (the planted truth is noise-free). If the real advantage had that structure and size, the
estimators would score this; their real score (0.02-0.12) then bounds how much of the real advantage is of that kind.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("q1", HERE / "03_q1_estimation_ceiling.py")
q1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q1)
DATA = q1.DATA
RNG = np.random.default_rng(1)


def corr_within(pred, truth):
    c, t = pred - pred.mean(1, keepdims=True), truth - truth.mean(1, keepdims=True)
    return (c * t).sum() / np.sqrt((c * c).sum() * (t * t).sum())


def main():
    for name in (sys.argv[1:] or ["v23_4M", "v29h8_8M"]):
        d = np.load(DATA / f"03_chunk_{name}.npz")
        obs, act, mean, std, y = (d[k].astype(np.float64) for k in ("obs", "action", "mean", "std", "target"))
        row, s_row, s_obs = d["row"], d["s_row"], d["s_obs"].astype(np.float64)
        z = (np.arctanh(np.clip(act, -1 + 1e-7, 1 - 1e-7)) - mean) / std
        s_act, s_mean, s_std = (d[k].astype(np.float64) for k in ("s_actions", "s_mean", "s_std"))
        z_fresh = (np.arctanh(np.clip(s_act[:, 50:58], -1 + 1e-7, 1 - 1e-7)) - s_mean[:, None]) / s_std[:, None]
        ret = d["ret_300"].astype(np.float64)[:, 50:58]
        cen = ret - ret.mean(1, keepdims=True)
        sigma2 = (cen.mean(2) ** 2).sum(1).mean() / 7 - cen.var(2, ddof=1).sum(1).mean() / (ret.shape[2] * 7)
        w1, b1, w2 = RNG.normal(size=(17, 64)) / np.sqrt(17), RNG.normal(size=64), RNG.normal(size=(64, 6)) / 8
        wf, bf, cf = RNG.normal(size=(17, 2000)) * 4 / np.sqrt(17), RNG.uniform(0, 2 * np.pi, 2000), RNG.normal(size=(2000, 6))
        wl = RNG.normal(size=(18, 6))
        fields = {
            "const": lambda o: np.tile(wl[0], (len(o), 1)),
            "linear": lambda o: np.concatenate([np.ones((len(o), 1)), o], 1) @ wl,
            "smooth": lambda o: np.tanh(o @ w1 + b1) @ w2,
            "rough": lambda o: np.cos(o @ wf + bf) @ cf,
        }
        proj = (RNG.normal(size=(17, 128)) / np.sqrt(17), RNG.normal(size=128))
        proj_b = (RNG.normal(size=(17, 768)) / np.sqrt(17), RNG.normal(size=768))
        print(f"\n=== {name}: planted variance {sigma2:.4f}, real target variance after baseline ~ see q1 log")
        d2 = (s_obs ** 2).sum(1)[:, None] + (obs ** 2).sum(1)[None] - 2.0 * s_obs @ obs.T
        d2[s_row[:, None] == row[None]] = np.inf
        order = np.argsort(d2, axis=1)[:, :2048]
        for label, field in fields.items():
            g_chunk, g_states = field(obs), field(s_obs)
            scale = np.sqrt(sigma2 / (z * g_chunk).sum(1).var())
            planted_y = y + scale * (z * g_chunk).sum(1)
            truth = scale * np.einsum("nkd,nd->nk", z_fresh, g_states)
            line = f"   planted {label:7s}"
            for kind in ("const", "linear", "rf"):
                g_hat = np.zeros((len(s_obs), 6))
                for fold in np.array_split(np.arange(16), 4):
                    train = ~np.isin(row, fold)
                    phi_b = q1.state_features(obs, "rf", proj_b)
                    mu_b, sd_b = phi_b[train].mean(0), phi_b[train].std(0) + 1e-8
                    phi_b = (phi_b - mu_b) / sd_b
                    phi_b[:, 0] = 1.0
                    resid = planted_y - phi_b @ q1.cv_ridge(phi_b[train], planted_y[train], row[train])[0]
                    phi, phi_s = q1.state_features(obs, kind, proj), q1.state_features(s_obs, kind, proj)
                    if kind != "const":
                        mu, sd = phi[train].mean(0), phi[train].std(0) + 1e-8
                        phi, phi_s = (phi - mu) / sd, (phi_s - mu) / sd
                        phi[:, 0], phi_s[:, 0] = 1.0, 1.0
                    X = q1.z_design(z, phi)
                    w = q1.ridge(X[train], resid[train], 0.0) if kind == "const" else q1.cv_ridge(X[train], resid[train], row[train])[0]
                    test = np.isin(s_row, fold)
                    g_hat[test] = (phi_s @ w.reshape(6, -1).T)[test]
                line += f" | ridge-{kind} {corr_within(np.einsum('nkd,nd->nk', z_fresh, g_hat), truth):+.3f}"
            for k in (128, 512, 2048):
                g_hat = np.empty((len(s_obs), 6))
                for i, nb in enumerate(order[:, :k]):
                    g_hat[i] = np.linalg.lstsq(np.concatenate([np.ones((k, 1)), z[nb]], 1), planted_y[nb], rcond=None)[0][1:]
                line += f" | kNN{k} {corr_within(np.einsum('nkd,nd->nk', z_fresh, g_hat), truth):+.3f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()

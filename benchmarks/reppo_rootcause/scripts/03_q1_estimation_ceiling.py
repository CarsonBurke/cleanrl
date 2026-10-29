"""Q1: is the critic's within-state ranking limited by ESTIMATION or by OPTIMIZATION/REPRESENTATION?

Inputs: benchmarks/reppo_rootcause/data/03_chunk_*.npz (from 03_prep_oracle_chunks.py). numpy only.

Per oracle dataset (one frozen snapshot):
  A. variance decomposition of the critic's regression problem, measured at the 512 oracle states
     (truth = H-step MuJoCo branching returns, 32 CRN rollouts, rollout noise removed);
  B. the decisive test: closed-form ridge estimators of the advantage, fit ONLY on the 16384-sample on-policy
     chunk with the trainer's own lambda-return targets (the critic's exact training signal for one
     iteration; no simulator truth enters the fit), scored against the same truth as the snapshot critic.
       A_hat(s, a) = sum_d z_d * (phi(s) . W_d),  z = (atanh(a) - mu(s)) / sigma(s)  (the policy's own noise)
     with phi = 1 (6 parameters), [1, obs] (108), or [1, obs, 128 random ReLU features] (876).
     The baseline b(s) is a random-feature ridge of the target on obs (z is independent of s, so any b is valid).
     Scored (i) leave-env-rows-out (4 folds of 4 lanes: the model never saw the scored states' trajectories)
     and (ii) fit on all 16 lanes (the scored fresh actions/probes are still unseen).
  C. autocorrelation / effective sample size of the regression noise.
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
LAMBDAS = np.array([1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])


def ridge(X, y, lam):
    gram = X.T @ X
    gram[np.diag_indices_from(gram)] += lam
    return np.linalg.solve(gram, X.T @ y)


def cv_ridge(X, y, groups, lambdas=LAMBDAS):
    """Ridge with lambda chosen by leave-group-out MSE on the (noisy) targets only."""
    ids = np.unique(groups)
    folds = np.array_split(ids, min(4, len(ids)))
    scores = np.zeros(len(lambdas))
    for fold in folds:
        test = np.isin(groups, fold)
        gram, rhs = X[~test].T @ X[~test], X[~test].T @ y[~test]
        for i, lam in enumerate(lambdas):
            w = np.linalg.solve(gram + lam * np.eye(len(gram)), rhs)
            scores[i] += ((X[test] @ w - y[test]) ** 2).sum()
    lam = lambdas[scores.argmin()]
    return ridge(X, y, lam), lam


def state_features(obs, kind, proj):
    one = np.ones((len(obs), 1))
    if kind == "const":
        return one
    if kind == "linear":
        return np.concatenate([one, obs], 1)
    rf = np.maximum(obs @ proj[0] + proj[1], 0.0)
    return np.concatenate([one, obs, rf], 1)


def z_design(z, phi):
    return (z[:, :, None] * phi[:, None, :]).reshape(len(z), -1)


def adv_stats(pred, true):
    """pred (n, K), true (n, K, M): pooled within-state corr, raw and rollout-noise-corrected; slope truth on pred."""
    k, m = true.shape[1], true.shape[2]
    centered = true - true.mean(1, keepdims=True)
    t = centered.mean(2)
    c = pred - pred.mean(1, keepdims=True)
    cov, var_c, var_t = (c * t).sum() , (c * c).sum(), (t * t).sum()
    noise = (centered.var(2, ddof=1).sum() / m)
    signal = var_t - noise
    return np.array([cov / np.sqrt(var_c * var_t), cov / np.sqrt(var_c * signal) if signal > 0 else np.nan,
                     cov / var_c, np.sqrt(var_c / signal) if signal > 0 else np.nan])


def grad_stats(pred, true):
    """pred (n, D), true (n, D, M) directional derivatives: slope of truth on pred, pooled uncentered corr raw /
    noise-corrected, ascent ratio (true rate along unit pred over claimed rate)."""
    m = true.shape[2]
    t = true.mean(2)
    noise = (true.var(2, ddof=1) / m).sum()
    dot, pp, tt = (t * pred).sum(), (pred * pred).sum(), (t * t).sum()
    signal = tt - noise
    norms = np.maximum(np.linalg.norm(pred, axis=1), 1e-30)
    ascent = ((t * pred).sum(1) / norms).sum() / norms.sum()
    return np.array([dot / pp, dot / np.sqrt(pp * tt), dot / np.sqrt(pp * signal) if signal > 0 else np.nan, ascent,
                     signal / tt])


def boot(fn, *arrays, clusters=None):
    """Point estimate and 95% CI resampling states (or whole env rows when clusters is given)."""
    n = len(arrays[0])
    point = fn(*arrays)
    draws = []
    for _ in range(BOOT):
        if clusters is None:
            idx = RNG.integers(0, n, n)
        else:
            ids = np.unique(clusters)
            idx = np.concatenate([np.flatnonzero(clusters == c) for c in RNG.choice(ids, len(ids))])
        draws.append(fn(*(a[idx] for a in arrays)))
    lo, hi = np.nanpercentile(np.array(draws), [2.5, 97.5], axis=0)
    return point, lo, hi


def fmt(triple, i):
    point, lo, hi = triple
    return f"{point[i]:+.3f} [{lo[i]:+.3f},{hi[i]:+.3f}]"


def autocorr_ess(values, steps, envs, max_lag=200):
    """ESS factor 1 / (1 + 2 sum_k rho_k) of a per-(step, env) series; truncated at the first non-positive rho."""
    x = values.reshape(steps, envs)
    x = x - x.mean()
    var = (x * x).mean()
    rhos = []
    for lag in range(1, max_lag):
        rho = (x[lag:] * x[:-lag]).mean() / var
        if rho <= 0.0:
            break
        rhos.append(rho)
    return 1.0 / (1.0 + 2.0 * sum(rhos)), (rhos[0] if rhos else 0.0), len(rhos)


def analyze(name, out):
    d = np.load(DATA / f"03_chunk_{name}.npz")
    obs, act, mean, std, y = d["obs"].astype(np.float64), d["action"].astype(np.float64), d["mean"].astype(np.float64), \
        d["std"].astype(np.float64), d["target"].astype(np.float64)
    row, idx = d["row"], d["s_index"]
    n_total, steps, envs = len(y), 1024, 16
    z = (np.arctanh(np.clip(act, -1 + 1e-7, 1 - 1e-7)) - mean) / std
    horizon = "300"
    ret = d[f"ret_{horizon}"].astype(np.float64)  # (512, 58, 32)
    m = ret.shape[2]
    a0_slot, fresh = 25, slice(50, 58)
    s_mean, s_std, s_act = d["s_mean"].astype(np.float64), d["s_std"].astype(np.float64), d["s_actions"].astype(np.float64)
    star = np.tanh(s_mean)
    res = {"global_step": int(d["global_step"]), "lam": float(d["lam"]), "temperature": float(d["temperature"])}
    print(f"\n================ {name}: step {res['global_step']}, lambda {res['lam']}, T {res['temperature']:.4f}")
    print(f"z reconstruction: mean {z.mean():+.4f} var {z.var():.4f} max|z| {np.abs(z).max():.2f} "
          f"(|a|>0.999 frac {np.mean(np.abs(act) > 0.999):.4f}); policy std mean {std.mean():.3f}")

    # ---------------- A. variance decomposition at the oracle states
    q_a0 = ret[:, a0_slot]
    q_a0_mean, q_a0_noise = q_a0.mean(1), (q_a0.var(1, ddof=1) / m).mean()
    fr = ret[:, fresh]
    v_true = fr.mean((1, 2))
    centered = fr - fr.mean(1, keepdims=True)
    within_noise = centered.var(2, ddof=1).sum(1).mean() / (m * 7)
    within_var = (centered.mean(2) ** 2).sum(1).mean() / 7 - within_noise
    v_noise = (fr.mean(1).var(1, ddof=1) / m).mean() + max(within_var, 0) / 8
    between_var = v_true.var() - v_noise
    single_rollout_var = q_a0.var(1, ddof=1).mean()
    # CRN-paired single-rollout advantage noise (what one MC return tells about A): var over m of Q_k - mean_j Q_j
    target_err = y[idx] - q_a0_mean
    target_noise_var = target_err.var() - q_a0_noise
    critic_a0 = d["s_critic_q"][:, a0_slot]
    critic_err = critic_a0 - q_a0_mean
    critic_err_var = critic_err.var() - q_a0_noise
    crit_fresh = d["s_critic_q"][:, fresh]
    critic_within_var = ((crit_fresh - crit_fresh.mean(1, keepdims=True)) ** 2).sum(1).mean() / 7
    cq = d["critic_q"].astype(np.float64)
    res["decomp"] = {
        "target_total_var_chunk": float(y.var()), "between_state_var": float(between_var),
        "within_state_true_var": float(within_var), "within_state_true_sd": float(np.sqrt(max(within_var, 0))),
        "target_noise_var": float(target_noise_var), "target_bias": float(target_err.mean()),
        "mc_single_rollout_var_H300": float(single_rollout_var),
        "critic_err_var": float(critic_err_var), "critic_bias": float(critic_err.mean()),
        "critic_within_var": float(critic_within_var),
        "chunk_resid_var_target_minus_critic": float((y - cq).var()),
        "oracle_q_se2": float(q_a0_noise), "oracle_within_noise_share": float(within_noise / (within_var + within_noise)),
    }
    dd = res["decomp"]
    print("A. variance decomposition (scaled units^2; sd in parentheses)")
    for key in ["target_total_var_chunk", "between_state_var", "within_state_true_var", "target_noise_var",
                "mc_single_rollout_var_H300", "critic_err_var", "critic_within_var", "chunk_resid_var_target_minus_critic"]:
        print(f"   {key:38s} {dd[key]:9.4f} ({np.sqrt(max(dd[key], 0)):.3f})")
    print(f"   target bias {dd['target_bias']:+.3f}, critic bias {dd['critic_bias']:+.3f}, oracle Q se^2 {q_a0_noise:.4f}, "
          f"oracle within-state noise share {dd['oracle_within_noise_share']:.2f}")
    # Analytic lambda-return noise from future sampled actions: var = sigma_A^2 c^2/(1-c^2), c = gamma*lam
    c = float(d["gamma"]) * res["lam"]
    print(f"   analytic future-action noise: sigma_A^2 * c^2/(1-c^2) = {within_var:.4f} * {c * c / (1 - c * c):.2f} "
          f"= {within_var * c * c / (1 - c * c):.4f}  (c = gamma*lambda = {c:.3f})")
    snr = within_var / (within_var + target_noise_var)
    print(f"   per-sample advantage signal share  sigma_A^2 / (sigma_A^2 + target noise) = {snr:.3f}"
          f"  -> single-sample corr ceiling {np.sqrt(max(snr, 0)):.3f}")

    # ---------------- oracle-side quantities for scoring
    true_fresh = ret[:, fresh]
    z_fresh = (np.arctanh(np.clip(s_act[:, fresh], -1 + 1e-7, 1 - 1e-7)) - s_mean[:, None]) / s_std[:, None]
    span = d["span_star"].astype(np.float64)  # (2, n, 6)
    jac = (1.0 - star ** 2) * s_std  # d a / d z at the mean action
    fd = {}
    for i, h in enumerate(d["fd_steps"]):
        plus, minus = ret[:, 1 + 12 * i:7 + 12 * i], ret[:, 7 + 12 * i:13 + 12 * i]
        fd[float(h)] = (plus - minus) / span[i][:, :, None] * jac[:, :, None]  # z-space derivative samples (n, 6, M)
    critic_gz = d["s_critic_grad_star"].astype(np.float64) * jac
    s_row = d["s_row"]

    scored = {"critic": (crit_fresh, critic_gz)}

    # ---------------- B. ridge estimators on the chunk
    proj = (RNG.normal(size=(17, 128)) / np.sqrt(17), RNG.normal(size=128))
    proj_b = (RNG.normal(size=(17, 768)) / np.sqrt(17), RNG.normal(size=768))

    def fit_predict(train, kinds=("const", "linear", "rf")):
        phi_b = state_features(obs, "rf", proj_b)
        mu_b, sd_b = phi_b[train].mean(0), phi_b[train].std(0) + 1e-8
        phi_b = (phi_b - mu_b) / sd_b
        phi_b[:, 0] = 1.0
        w_b, lam_b = cv_ridge(phi_b[train], y[train], row[train])
        resid = y - phi_b @ w_b
        preds, info = {}, {"baseline_lambda": float(lam_b), "resid_var_train": float(resid[train].var()),
                           "resid_var_heldout": float(resid[~train].var()) if (~train).any() else None}
        for kind in kinds:
            phi = state_features(obs, kind, proj)
            mu, sd = phi[train].mean(0), phi[train].std(0) + 1e-8
            if kind != "const":
                phi = (phi - mu) / sd
                phi[:, 0] = 1.0
            X = z_design(z, phi)
            if kind == "const":
                w, lam = ridge(X[train], resid[train], 0.0), 0.0
            else:
                w, lam = cv_ridge(X[train], resid[train], row[train])
            W = w.reshape(6, -1)
            phi_s = state_features(d["s_obs"].astype(np.float64), kind, proj)
            if kind != "const":
                phi_s = (phi_s - mu) / sd
                phi_s[:, 0] = 1.0
            g = phi_s @ W.T  # (n, 6) z-space gradient field at the oracle states
            preds[kind] = g
            info[f"lambda_{kind}"] = float(lam)
            info[f"params_{kind}"] = int(W.size)
        return preds, info, resid

    # (i) leave-env-rows-out
    g_cv = {}
    for fold in np.array_split(np.arange(envs), 4):
        train = ~np.isin(row, fold)
        preds, info, _ = fit_predict(train)
        test_states = np.isin(s_row, fold)
        for kind, g in preds.items():
            g_cv.setdefault(kind, np.zeros_like(g))[test_states] = g[test_states]
    # (ii) all lanes
    preds_all, info_all, resid_all = fit_predict(np.ones(n_total, dtype=bool))
    res["fit_info_all"] = info_all
    print(f"B. ridge fit info (all lanes): {json.dumps(info_all)}")
    for kind in ("const", "linear", "rf"):
        scored[f"{kind}_cv"] = (np.einsum("nkd,nd->nk", z_fresh, g_cv[kind]), g_cv[kind])
        scored[f"{kind}_all"] = (np.einsum("nkd,nd->nk", z_fresh, preds_all[kind]), preds_all[kind])
    # blends: does the estimator add information to the critic? (sum of standardized predictions)
    def unit_scale(a):
        return a / np.sqrt(((a - a.mean(1, keepdims=True)) ** 2).mean())
    for kind in ("linear", "rf"):
        a_e, g_e = scored[f"{kind}_cv"]
        scored[f"critic+{kind}_cv"] = (unit_scale(crit_fresh) + unit_scale(a_e),
                                       critic_gz / np.sqrt((critic_gz ** 2).mean()) + g_e / np.sqrt((g_e ** 2).mean()))

    res["scores"] = {}
    h_main = float(d["fd_steps"][0])
    print(f"   scoring vs truth (H={horizon}, {m} CRN rollouts). adv = pooled within-state corr over 8 fresh actions; "
          f"grad = z-space FD at mean action, h={h_main}")
    print(f"   {'estimator':18s} {'adv corr raw':26s} {'adv corr noise-corrected':26s} {'grad slope true-on-est':26s} "
          f"{'grad corr noise-corrected':26s} {'ascent ratio':24s}")
    for key, (a_hat, g_hat) in scored.items():
        adv = boot(adv_stats, a_hat, true_fresh)
        adv_row = boot(adv_stats, a_hat, true_fresh, clusters=s_row)
        grd = boot(grad_stats, g_hat, fd[h_main])
        res["scores"][key] = {
            "adv_corr_raw": [float(v[0]) for v in adv], "adv_corr_corrected": [float(v[1]) for v in adv],
            "adv_corr_corrected_rowboot": [float(v[1]) for v in adv_row],
            "adv_slope_true_on_est": [float(v[2]) for v in adv], "adv_sd_ratio": [float(v[3]) for v in adv],
            "grad_slope": [float(v[0]) for v in grd], "grad_corr_raw": [float(v[1]) for v in grd],
            "grad_corr_corrected": [float(v[2]) for v in grd], "grad_ascent_ratio": [float(v[3]) for v in grd],
            "grad_truth_signal_frac": float(grd[0][4]),
        }
        print(f"   {key:18s} {fmt(adv, 0):26s} {fmt(adv, 1):26s} {fmt(grd, 0):26s} {fmt(grd, 2):26s} {fmt(grd, 3):24s}")
    # paired differences estimator - critic (state bootstrap) on the noise-corrected advantage corr
    for kind in ("const_cv", "linear_cv", "rf_cv", "linear_all"):
        a_hat = scored[kind][0]
        diff = lambda a, c, t: adv_stats(a, t)[:2] - adv_stats(c, t)[:2]  # noqa: E731
        point, lo, hi = boot(diff, a_hat, crit_fresh, true_fresh)
        res["scores"][kind]["adv_corr_corrected_minus_critic"] = [float(point[1]), float(lo[1]), float(hi[1])]
        print(f"   paired {kind:12s} - critic, adv corr corrected: {point[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]")
    # gradient truth quality and structure
    for h, samples in fd.items():
        t = samples.mean(2)
        noise = (samples.var(2, ddof=1) / m).sum()
        tt = (t * t).sum()
        gbar = t.mean(0)
        gbar_noise = (samples.mean(0).var(1, ddof=1) / m).sum()  # noise of the across-state mean, per dim, summed
        energy = (tt - noise) / len(t)
        const_share = ((gbar ** 2).sum() - gbar_noise) / energy if energy > 0 else np.nan
        print(f"   truth z-grad h={h}: E|g_z|^2 noise-corrected {energy:.4f} (signal frac {1 - noise / tt:.2f}); "
              f"state-independent share |E g|^2/E|g|^2 = {const_share:.3f}; mean g_z = {np.round(gbar, 3)}")
        res.setdefault("grad_truth", {})[str(h)] = {"energy": float(energy), "signal_frac": float(1 - noise / tt),
                                                     "const_share": float(const_share)}

    # ---------------- C. autocorrelation / ESS
    ess_y, rho1_y, lags_y = autocorr_ess(resid_all, steps, envs)
    prod = resid_all[:, None] * z
    ess_p = [autocorr_ess(prod[:, k], steps, envs)[0] for k in range(6)]
    td = y - cq
    ess_td, rho1_td, _ = autocorr_ess(td, steps, envs)
    ess_obs = np.mean([autocorr_ess(obs[:, k], steps, envs)[0] for k in range(17)])
    res["ess"] = {"resid_factor": float(ess_y), "resid_rho1": float(rho1_y), "score_product_factor": float(np.mean(ess_p)),
                  "target_minus_critic_factor": float(ess_td), "target_minus_critic_rho1": float(rho1_td),
                  "obs_factor": float(ess_obs)}
    print(f"C. ESS factor (1 = iid): target residual y-b {ess_y:.3f} (rho1 {rho1_y:.2f}, {lags_y} positive lags); "
          f"target-critic {ess_td:.3f} (rho1 {rho1_td:.2f}); score products (y-b)*z_d {np.mean(ess_p):.3f}; obs dims {ess_obs:.3f}")
    # analytic estimation ceiling for a P-feature linear gradient field
    sigma_r2 = resid_all.var()
    S = res["grad_truth"][str(h_main)]["energy"]
    per_param = sigma_r2 / n_total
    print(f"   analytic: residual var {sigma_r2:.3f}; error energy per (dim x feature) = {per_param:.2e}; signal energy S = {S:.4f}")
    for target_corr in (0.9, 0.7, 0.5, 0.1):
        P = S * (1 / target_corr ** 2 - 1) / (6 * per_param) if S > 0 else np.nan
        print(f"      corr {target_corr}: reached while the gradient field needs <= {P:,.0f} free features per action dim")
    res["analytic"] = {"sigma_r2": float(sigma_r2), "signal_energy": float(S)}
    out[name] = res


def main():
    out = {}
    for name in (sys.argv[1:] or NAMES):
        analyze(name, out)
    (DATA / "03_q1_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

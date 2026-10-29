"""Per-channel reliability of the per-sample natural-gradient policy target (cleanrl/awr/OPTIMAL_CE.md s4-5).

    python scripts/awr_channel_reliability.py SNAPSHOT.pt [SNAPSHOT.pt ...] [--out runs/awr_channel_reliability.json]
    python scripts/awr_channel_reliability.py --self-test      # CPU checks of the math, no env or GPU

For each snapshot (scripts/awr_capture_policy.py format), frozen-policy rollouts are collected with the
snapshot's own trainer module through scripts/awr_update_bench.collect (trainer-identical GAE, per chunk).
Per sample i and action dim d, with the old Beta(alpha, beta) in coordinates (mu, l = log kappa):

    s  = d log Beta(a; mu kappa, (1 - mu) kappa) / d(mu, l)          (closed form, digamma)
    F  = J^T F_ab J,  F_ab = [[psi1(a) - psi1(k), -psi1(k)], [-psi1(k), psi1(b) - psi1(k)]]
    t  = F^-1 s z      (z: per-chunk z-scored GAE advantage, as the trainers use)

Channels: loc_d = t[0], conc_d = t[1]; loc_raw/conc_raw use the raw advantage (z's shift is removed
only by z-scoring, so raw differs by more than scale); pg_loc/pg_conc are the vanilla output-space
gradient s z (no F^-1); z and adv_raw are the scalar forces themselves (state-predictability reference).

Statistics per channel:
  top1   share of sum t^2 carried by the 1% of samples with largest |z| (z^2 itself is the baseline);
  rho    split-half reliability: ridge fits on envs 0-7 vs envs 8-15 of the training chunks, both
         predicted on held-out chunks, rho = corr(pred_A, pred_B), sb = 2 rho / (1 + rho);
  slope  calibration: the half-A fit predicted on half B's own samples, slope of B's raw targets on it
         (Cov(t_B, pred_A) / Var(pred_A)), averaged with B->A; ~1 = calibrated, ~0 = fitted noise.
         slope_huber is the same slope from a Huber (IRLS) regression, robust to the tail samples;
         slope_ols uses near-unpenalized half fits, so it is the fitted function's attenuation factor
         (~ signal / (signal + estimation noise), the MSE-optimal shrinkage) on a common [0, 1] scale;
  rho_null  rho_split with shuffled targets: the floor from feature anisotropy alone;
  r2     out-of-sample R2 of the all-envs fit on the held-out chunks.
Also per dim: corr(loc_d, conc_d) of the raw per-sample targets (noise shared through the scalar A).
Ridge penalties are chosen per channel by inner cross-validation over env-pair folds (no temporal
leakage). Two feature sets: the actor's last hidden activations (input of the Beta head) and
[old mu, log kappa, obs].
"""

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import awr_update_bench as bench  # noqa: E402

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
SUMMARY_GROUPS = ("loc", "conc", "loc_raw", "conc_raw", "pg_loc", "pg_conc")


# ---------------------------------------------------------------- Beta geometry in (mu, log kappa)

def beta_score_fisher(alpha, beta, a):
    """Closed-form score s (..., 2) and Fisher F (..., 2, 2) of Beta(a; alpha, beta) in (mu, log kappa)."""
    kappa = alpha + beta
    log_a, log_1ma = torch.log(a), torch.log1p(-a)
    dg_k = torch.digamma(kappa)
    s_alpha = log_a - torch.digamma(alpha) + dg_k
    s_beta = log_1ma - torch.digamma(beta) + dg_k
    # alpha = mu kappa, beta = (1 - mu) kappa: J = d(alpha, beta)/d(mu, l) = [[kappa, alpha], [-kappa, beta]].
    s = torch.stack((kappa * (s_alpha - s_beta), alpha * s_alpha + beta * s_beta), -1)
    t1_k = torch.polygamma(1, kappa)
    f_aa = torch.polygamma(1, alpha) - t1_k
    f_bb = torch.polygamma(1, beta) - t1_k
    f_ab = -t1_k
    # F = J^T F_ab J, written out.
    f_mm = kappa ** 2 * (f_aa - 2 * f_ab + f_bb)
    f_ml = kappa * (alpha * f_aa + (beta - alpha) * f_ab - beta * f_bb)
    f_ll = alpha ** 2 * f_aa + 2 * alpha * beta * f_ab + beta ** 2 * f_bb
    fisher = torch.stack((torch.stack((f_mm, f_ml), -1), torch.stack((f_ml, f_ll), -1)), -2)
    return s, fisher


def natural_direction(s, fisher):
    """F^-1 s for batched 2x2 F (explicit inverse; F is SPD)."""
    f_mm, f_ml, f_ll = fisher[..., 0, 0], fisher[..., 0, 1], fisher[..., 1, 1]
    det = f_mm * f_ll - f_ml ** 2
    return torch.stack(((f_ll * s[..., 0] - f_ml * s[..., 1]) / det,
                        (f_mm * s[..., 1] - f_ml * s[..., 0]) / det), -1)


# ---------------------------------------------------------------- statistics

def top_share(values_sq, z, frac=0.01):
    """Share of sum(values_sq) (n, c) carried by the frac of samples with largest |z|."""
    k = max(1, int(round(frac * z.shape[0])))
    top = torch.topk(z.abs(), k).indices
    return values_sq[top].sum(0) / values_sq.sum(0)


def standardize(x, mean, std):
    return (x - mean) / std


def ridge_cv(x, y, folds, lambdas=LAMBDAS):
    """Ridge of y (n, c) on x (n, p) with intercept; lambda per column by grouped CV.

    Features are standardized with the fitting set's statistics and the penalty is lambda * n on the
    standardized weights (intercept unpenalized). Returns a predictor and the chosen lambdas.
    """
    lam = torch.tensor(lambdas, dtype=x.dtype, device=x.device)
    sse = torch.zeros(len(lambdas), y.shape[1], dtype=x.dtype, device=x.device)
    for fold in folds.unique():
        train, test = folds != fold, folds == fold
        solve = _ridge_solutions(x[train], y[train], lam)
        for j, predict in enumerate(solve):
            sse[j] += ((predict(x[test]) - y[test]) ** 2).sum(0)
    best = sse.argmin(0)
    solutions = _ridge_solutions(x, y, lam)
    chosen = lam[best]

    def predict(x_new):
        out = torch.stack([solution(x_new) for solution in solutions])  # (L, n, c)
        return out.gather(0, best.view(1, 1, -1).expand(1, x_new.shape[0], -1)).squeeze(0)

    return predict, chosen


def _ridge_solutions(x, y, lam):
    n = x.shape[0]
    mean = x.mean(0)
    std = x.std(0)
    std = torch.where(std > 1e-8 * std.max().clamp_min(1e-30), std, torch.ones_like(std))
    xs = standardize(x, mean, std)
    y_mean = y.mean(0)
    gram = xs.T @ xs / n
    cross = xs.T @ (y - y_mean) / n
    evals, evecs = torch.linalg.eigh(gram)
    rotated = evecs.T @ cross
    solutions = []
    for value in lam:
        w = evecs @ (rotated / (evals + value).unsqueeze(1))

        def predict(x_new, w=w):
            return standardize(x_new, mean, std) @ w + y_mean

        solutions.append(predict)
    return solutions


def corr_cols(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    denom = (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-300)
    return (a * b).sum(0) / denom


def r2_cols(pred, y):
    return 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def reliability(x_train, y_train, env_train, x_hold, y_hold, num_envs):
    """Split-half (by env) reliability, Spearman-Brown, held-out R2 of the full fit, chosen lambdas."""
    half = num_envs // 2
    in_a = env_train < half
    out = {}
    preds, predictors = [], []
    for name, mask in (("A", in_a), ("B", ~in_a)):
        folds = env_train[mask] // 2  # env pairs within the half
        predict, lam = ridge_cv(x_train[mask], y_train[mask], folds)
        predictors.append(predict)
        preds.append(predict(x_hold))
        out[f"lambda_{name}"] = lam
    rho = corr_cols(*preds)
    out["rho_split"] = rho
    out["sb"] = 2 * rho / (1 + rho)  # meaningful only for rho > 0; compare rho against rho_null
    # Calibration: one half's fitted function against the other half's own raw targets.
    slopes, robust = [], []
    for predict, mask in ((predictors[0], ~in_a), (predictors[1], in_a)):
        slopes.append(slope_cols(predict(x_train[mask]), y_train[mask]))
        robust.append(huber_slope_cols(predict(x_train[mask]), y_train[mask]))
    out["slope_AB"], out["slope_BA"] = slopes
    out["slope"] = (slopes[0] + slopes[1]) / 2
    out["slope_huber"] = (robust[0] + robust[1]) / 2
    # Unshrunk (near-OLS) half fits: their cross-half slope is the attenuation factor signal / (signal +
    # estimation noise) of the fitted function, i.e. the MSE-optimal shrinkage rho, on a common scale.
    tiny = torch.tensor([1e-6], dtype=x_train.dtype, device=x_train.device)
    ols = [_ridge_solutions(x_train[mask], y_train[mask], tiny)[0] for mask in (in_a, ~in_a)]
    out["slope_ols"] = (slope_cols(ols[0](x_train[~in_a]), y_train[~in_a])
                        + slope_cols(ols[1](x_train[in_a]), y_train[in_a])) / 2
    out["slope_ols_huber"] = (huber_slope_cols(ols[0](x_train[~in_a]), y_train[~in_a])
                              + huber_slope_cols(ols[1](x_train[in_a]), y_train[in_a])) / 2
    # Null for rho: shuffled targets keep the feature anisotropy, which alone makes two shrunk noise
    # fits correlate (both load on the top principal directions).
    generator = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(y_train.shape[0], generator=generator).to(y_train.device)
    null_preds = [ridge_cv(x_train[mask], y_train[perm][mask], env_train[mask] // 2)[0](x_hold)
                  for mask in (in_a, ~in_a)]
    out["rho_null"] = corr_cols(*null_preds)
    predict, lam = ridge_cv(x_train, y_train, env_train // 2)
    out["lambda_full"] = lam
    pred_hold = predict(x_hold)
    out["r2"] = r2_cols(pred_hold, y_hold)
    out["slope_full_heldout"] = slope_cols(pred_hold, y_hold)
    return out


def slope_cols(pred, y):
    """Calibration slope Cov(y, pred) / Var(pred) per column."""
    pc = pred - pred.mean(0)
    return (pc * (y - y.mean(0))).sum(0) / (pc ** 2).sum(0).clamp_min(1e-300)


def huber_slope_cols(pred, y, c=1.345, iters=30):
    """Slope of the Huber regression y ~ a + b pred per column (IRLS, MAD residual scale).

    Unlike trimming rows by |y|, which truncates on the response and biases the slope toward 0,
    Huber downweights large residuals and stays unbiased for symmetric noise.
    """
    weights = torch.ones_like(y)
    for _ in range(iters):
        count = weights.sum(0)
        p_mean, y_mean = (weights * pred).sum(0) / count, (weights * y).sum(0) / count
        pc = pred - p_mean
        slope = (weights * pc * (y - y_mean)).sum(0) / (weights * pc ** 2).sum(0).clamp_min(1e-300)
        resid = y - y_mean - slope * pc
        scale = 1.4826 * (resid - resid.median(0).values).abs().median(0).values
        weights = torch.clamp(c * scale / resid.abs().clamp_min(1e-300), max=1.0)
    return slope


# ---------------------------------------------------------------- per-snapshot analysis

def build_targets(batch, action_dim):
    """All channels (n, C) plus names; z-scoring is per chunk, as in the trainers."""
    alpha, beta = batch["alpha"].double(), batch["beta"].double()
    a = batch["native"].double()
    s, fisher = beta_score_fisher(alpha, beta, a)
    nat = natural_direction(s, fisher)  # (n, d, 2)
    adv = batch["advantages"].double()
    z = batch["z"].double()
    cols, names = [], []
    for group, base, force in (("loc", nat[..., 0], z), ("conc", nat[..., 1], z),
                               ("loc_raw", nat[..., 0], adv), ("conc_raw", nat[..., 1], adv),
                               ("pg_loc", s[..., 0], z), ("pg_conc", s[..., 1], z)):
        cols.append(base * force.unsqueeze(1))
        names += [f"{group}_{d}" for d in range(action_dim)]
    cols += [z.unsqueeze(1), adv.unsqueeze(1)]
    names += ["z", "adv_raw"]
    return torch.cat(cols, 1), names


def hidden_features(agent, obs, chunk=65536):
    """Input of the actor's Beta head (the last hidden activations), captured by a pre-hook."""
    captured = []
    handle = agent.actor.head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    try:
        with torch.no_grad():
            for start in range(0, obs.shape[0], chunk):
                agent.actor(obs[start:start + chunk])
    finally:
        handle.remove()
    return torch.cat(captured).double()


def analyze_snapshot(path, cli, device):
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    module = importlib.import_module(snapshot["trainer_module"])
    module.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    args = bench.snapshot_args(snapshot)
    if args.gae_lambda != 0.95:
        raise ValueError(f"{path}: expected gae_lambda=0.95, got {args.gae_lambda}")
    envs = module.make_training_env(args, "channel_probe")
    agent = module.Agent(envs).to(device)
    envs.close()
    agent.load_state_dict(snapshot["model"])
    agent.eval()
    num_envs, num_steps, action_dim = args.num_envs, args.num_steps, agent.action_dim

    start = time.perf_counter()
    batches, returns = bench.collect(module, agent, snapshot, cli.train_chunks + cli.heldout_chunks,
                                     device, cli.seed)
    collect_s = time.perf_counter() - start
    env_index = torch.arange(num_steps * num_envs, device=device) % num_envs  # rows are (step, env) flattened
    for batch in batches:
        batch["z"] = bench.zscore(batch["advantages"])
        batch["env"] = env_index
        batch["alpha"], batch["beta"] = batch["alpha"].float(), batch["beta"].float()
    train = bench.concatenate(batches[:cli.train_chunks])
    hold = bench.concatenate(batches[cli.train_chunks:])
    everything = bench.concatenate(batches)

    y_train, names = build_targets(train, action_dim)
    y_hold, _ = build_targets(hold, action_dim)
    y_all, _ = build_targets(everything, action_dim)

    # (a) energy share of the top 1% |z| samples, pooled over every chunk.
    shares = top_share(y_all ** 2, everything["z"].double(), cli.top_frac)
    z_share = float(shares[names.index("z")])
    cross = {}
    for loc, conc in (("loc", "conc"), ("pg_loc", "pg_conc")):
        per_dim = [float(corr_cols(y_all[:, [names.index(f"{loc}_{d}")]],
                                   y_all[:, [names.index(f"{conc}_{d}")]])) for d in range(action_dim)]
        cross[f"corr_{loc}_{conc}"] = {"per_dim": per_dim, "mean": float(np.mean(per_dim))}

    def stats_features(part):
        alpha, beta = part["alpha"].double(), part["beta"].double()
        return torch.cat((alpha / (alpha + beta), torch.log(alpha + beta), part["obs"].double()), 1)

    feature_sets = {
        "hidden": (hidden_features(agent, train["obs"]), hidden_features(agent, hold["obs"])),
        "stats_obs": (stats_features(train), stats_features(hold)),
    }
    results = {}
    for fname, (x_train, x_hold) in feature_sets.items():
        results[fname] = reliability(x_train, y_train, train["env"], x_hold, y_hold, num_envs)

    channels = {}
    for c, name in enumerate(names):
        entry = {"top1_share": float(shares[c])}
        for fname, res in results.items():
            entry[fname] = {key: float(value[c]) for key, value in res.items()}
        channels[name] = entry

    def mean_over_dims(group):
        rows = [channels[f"{group}_{d}"] for d in range(action_dim)]
        out = {"top1_share": float(np.mean([row["top1_share"] for row in rows]))}
        for fname in feature_sets:
            out[fname] = {key: float(np.mean([row[fname][key] for row in rows])) for key in rows[0][fname]}
        return out

    summary = {group: mean_over_dims(group) for group in SUMMARY_GROUPS}
    summary["z"] = channels["z"]
    summary["adv_raw"] = channels["adv_raw"]
    alpha, beta = everything["alpha"].double(), everything["beta"].double()
    header = {
        "snapshot": str(path), "trainer": snapshot["trainer_module"], "global_step": snapshot["global_step"],
        "collect_s": collect_s, "train_samples": int(y_train.shape[0]), "heldout_samples": int(y_hold.shape[0]),
        "episode_return_mean": float(np.mean(returns)) if returns else None,
        "z_top1_share": z_share, "mean_kappa": float((alpha + beta).mean()),
        "hidden_dim": int(feature_sets["hidden"][0].shape[1]),
        "loc_conc_corr": cross,
    }
    print(json.dumps(header), flush=True)
    return {"header": header, "summary": summary, "channels": channels}


# ---------------------------------------------------------------- report

def print_table(report):
    cols = ("top1", "rho_hid", "null_hid", "r2_hid", "ols_hid", "olsH_hid", "rho_sto", "null_sto", "r2_sto",
            "ols_sto", "lam_hid")
    print("\n" + f"{'snapshot':28s} {'channel':9s} " + " ".join(f"{c:>8s}" for c in cols))
    for snap in report:
        head = snap["header"]
        label = f"{head['trainer'].split('continuous_action_')[-1][-14:]}@{head['global_step'] / 1e6:.1f}M"
        for group in (*SUMMARY_GROUPS, "z"):
            row = snap["summary"][group]
            hid, sto = row["hidden"], row["stats_obs"]
            values = (row["top1_share"], hid["rho_split"], hid["rho_null"], hid["r2"], hid["slope_ols"],
                      hid["slope_ols_huber"], sto["rho_split"], sto["rho_null"], sto["r2"], sto["slope_ols"],
                      hid["lambda_full"])
            print(f"{label:28s} {group:9s} " + " ".join(f"{v:8.3g}" for v in values))
        cross = head["loc_conc_corr"]
        print(f"{'':28s} z^2 top1 baseline {head['z_top1_share']:.3f}; mean kappa {head['mean_kappa']:.1f}; "
              f"corr(loc,conc) {cross['corr_loc_conc']['mean']:+.3f}; "
              f"corr(pg_loc,pg_conc) {cross['corr_pg_loc_pg_conc']['mean']:+.3f}")
    print("\nper-dim (hidden features): top1 / rho_split / r2 / slope_ols")
    for snap in report:
        head = snap["header"]
        print(f"  {head['trainer'].rsplit('.', 1)[-1]} @ {head['global_step']}")
        for group in ("loc", "conc", "pg_loc", "pg_conc"):
            cells = []
            for name, entry in snap["channels"].items():
                if name.rsplit("_", 1)[0] == group:
                    h = entry["hidden"]
                    cells.append(f"{entry['top1_share']:.2f}/{h['rho_split']:.2f}/{h['r2']:.4f}/"
                                 f"{h['slope_ols']:.2f}")
            print(f"    {group:8s} " + "  ".join(cells))


# ---------------------------------------------------------------- self test (CPU)

def self_test():
    torch.manual_seed(0)
    dt = torch.float64
    params = torch.tensor([[1.3, 2.7], [5.0, 4.0], [20.0, 3.0], [1.05, 1.9], [60.0, 45.0]], dtype=dt)
    # 1. Closed-form score vs autograd in (mu, log kappa).
    for alpha0, beta0 in params:
        a = torch.distributions.Beta(alpha0, beta0).sample((64,))
        mu = (alpha0 / (alpha0 + beta0)).clone().requires_grad_(True)
        log_k = torch.log(alpha0 + beta0).clone().requires_grad_(True)
        k = log_k.exp()
        lp = torch.distributions.Beta(mu * k, (1 - mu) * k).log_prob(a)
        auto = torch.stack([torch.stack(torch.autograd.grad(v, (mu, log_k), retain_graph=True)) for v in lp])
        closed, _ = beta_score_fisher(alpha0.expand(64), beta0.expand(64), a)
        err = (auto - closed).abs().max().item()
        assert err < 1e-9 * (1 + auto.abs().max().item()), f"score mismatch {err}"
    print("score vs autograd: ok")
    # 2. Fisher = E[s s^T] and E[s] = 0 by Monte Carlo; also vs autograd Hessian of the expected NLL.
    n = 2_000_000
    for alpha0, beta0 in params:
        a = torch.distributions.Beta(alpha0, beta0).sample((n,)).clamp(1e-12, 1 - 1e-12)
        s, fisher = beta_score_fisher(alpha0.expand(n), beta0.expand(n), a)
        mc = s.T @ s / n
        f = fisher[0]
        se = (s.unsqueeze(2) * s.unsqueeze(1)).reshape(n, 4).std(0).reshape(2, 2) / math.sqrt(n)
        z_err = ((mc - f) / se).abs().max().item()
        mean_z = (s.mean(0) / (s.std(0) / math.sqrt(n))).abs().max().item()
        assert z_err < 5 and mean_z < 5, (alpha0, beta0, mc, f, z_err, mean_z)
        # Analytic check: Fisher = Hessian of KL(old || new) at new = old, via autograd.
        theta = torch.stack((alpha0 / (alpha0 + beta0), torch.log(alpha0 + beta0))).requires_grad_(True)

        def kl(th):
            k = th[1].exp()
            return bench.beta_kl(alpha0, beta0, th[0] * k, (1 - th[0]) * k)

        hess = torch.autograd.functional.hessian(kl, theta)
        assert (hess - f).abs().max() < 1e-8 * (1 + f.abs().max()), (hess, f)
        nat = natural_direction(s[:4], fisher[:4])
        assert torch.allclose(nat, torch.linalg.solve(fisher[:4], s[:4].unsqueeze(-1)).squeeze(-1))
    print("Fisher vs MC E[ss^T], KL Hessian, F^-1 s solve: ok")
    # 3. Ridge/reliability on synthetic data with a known answer.
    n_steps, num_envs, p = 2048, 16, 20
    for true_r2 in (0.0, 0.02, 0.3):
        x = torch.randn(n_steps * num_envs, p, dtype=dt)
        x_hold = torch.randn(40000, p, dtype=dt)
        w = torch.randn(p, 1, dtype=dt)
        w = w / w.norm()
        noise_sd = math.sqrt((1 - true_r2) / true_r2) if true_r2 > 0 else 1.0
        scale = 1.0 if true_r2 > 0 else 0.0
        y = scale * x @ w + noise_sd * torch.randn(x.shape[0], 1, dtype=dt) + 3.0
        y_hold = scale * x_hold @ w + noise_sd * torch.randn(x_hold.shape[0], 1, dtype=dt) + 3.0
        env = torch.arange(x.shape[0]) % num_envs
        res = reliability(x, y, env, x_hold, y_hold, num_envs)
        r2, sb = res["r2"].item(), res["sb"].item()
        # Expected full-sample reliability of an OLS fit: signal var / (signal var + p noise var / n).
        expect_sb = true_r2 / (true_r2 + (1 - true_r2) * p / x.shape[0]) if true_r2 > 0 else 0.0
        print(f"  synthetic R2={true_r2}: r2={r2:.4f} sb={sb:.3f} (OLS-expected ~{expect_sb:.3f}) "
              f"lambda={res['lambda_full'].item():g}")
        assert abs(r2 - true_r2) < 0.02, r2
        if true_r2 >= 0.02:
            assert abs(sb - expect_sb) < 0.1, (sb, expect_sb)
            # A half fit's calibration slope on the other half attenuates by that fit's own reliability
            # (ridge shrinkage can only push it back toward 1).
            half_rel = true_r2 / (true_r2 + (1 - true_r2) * p / (x.shape[0] / 2))
            slope, slope_h = res["slope"].item(), res["slope_huber"].item()
            print(f"    slope={slope:.3f} huber={slope_h:.3f} (OLS half-fit reliability ~{half_rel:.3f})")
            assert half_rel - 0.1 < slope < 1.1 and half_rel - 0.1 < slope_h < 1.1, (slope, slope_h)
    # 4. top-share: uniform energy gives ~frac; energy only on the top-|z| sample gives 1.
    z = torch.randn(10000, dtype=dt)
    assert abs(top_share(torch.ones(10000, 1, dtype=dt), z).item() - 0.01) < 1e-12
    only = torch.zeros(10000, 1, dtype=dt)
    only[z.abs().argmax()] = 1.0
    assert top_share(only, z).item() == 1.0
    print("ridge reliability and top-share: ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshots", nargs="*", type=Path)
    parser.add_argument("--train-chunks", type=int, default=16, help="16x1024 rollouts used for fitting")
    parser.add_argument("--heldout-chunks", type=int, default=4, help="later chunks, prediction only")
    parser.add_argument("--top-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("runs/awr_channel_reliability.json"))
    parser.add_argument("--self-test", action="store_true")
    cli = parser.parse_args()
    if cli.self_test:
        self_test()
        return
    if not cli.snapshots:
        parser.error("no snapshots given")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(cli.seed)
    device = torch.device("cuda")
    report = [analyze_snapshot(path, cli, device) for path in cli.snapshots]
    print_table(report)
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {cli.out}")


if __name__ == "__main__":
    main()

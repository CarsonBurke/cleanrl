"""Does a persistent learned advantage model A_phi(s, a) carry more signal than one batch's A_hat?

    python scripts/awr_advantage_model_bench.py DATA.pt [DATA.pt ...] [--out results.json]

DATA files come from scripts/awr_oracle_advantage.py --save-data: a frozen snapshot policy's EVAL states
with oracle A_true (sampled action + K fresh actions, per-action SE from the rollout spread) and, disjoint
in time, model chunks (trainer-identical 16 x 1024 batches with their GAE inputs).

In deterministic MuJoCo, A_hat's noise is not irreducible: it is critic error plus later actions' true
advantages, both functions of states and actions a model can learn across batches. The bench fits A_phi
on the model chunks and scores it on EVAL against A_true:
  (a) R2sig on the sampled action: corr(A_phi, A_true)^2 / reliability, where reliability =
      1 - mean(SE^2) / Var(A_true) removes the oracle's own rollout noise;
  (b) within-state R2sig over the K fresh actions after per-state demeaning (the quantity a K-sample
      CE / MPO target depends on), noise-corrected the same way.
Raw A_hat_lambda's (a) on the same EVAL states is the baseline.

Models, on (obs, u) with u = (native - Beta mean) / Beta std per dim, always action-centred under pi_old:
  free: A = f(s, u) - mean_j f(s, u'_j) over fixed fresh draws u'_j ~ pi_old (2 x 256 SiLU);
  quad: A = g(s).u + 0.5 sum_i h_i(s) (u_i^2 - 1), exactly zero-mean under pi_old (E u = 0, E u^2 = 1).
Fit by Adam on MSE to A_hat_lambda (lambda in 0, 0.5, 0.8, 0.95), early-stopped on held-out model chunks,
at 1 / 4 / 16 / all training chunks: does persistence across batches pay?
M2 (iterated future-action correction): refit on GAE_lambda over delta_t - A_phi(s_t, a_t), plus
A_phi(s_t, a_t) added back. That strips later actions' fitted advantages from each trace while leaving
the estimator's expectation unchanged when A_phi is centred. Cross-fitted over two chunk folds, so no
sample's own noise enters its correction. Scored as the two folds' ensemble, per round.
Run it through mlq (CUDA).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.ppo_loop import compute_gae  # noqa: E402
from cleanrl.shared.runtime import configure_runtime  # noqa: E402

SAMPLE_EPS = 1e-6
TARGET_LAMBDAS = (0.0, 0.5, 0.8, 0.95)


# ---------------------------------------------------------------- models

def standardize(native, alpha, beta):
    """Per-dim action offset in Beta standard deviations; alpha/beta broadcast over the action axis."""
    total = alpha + beta
    mean = alpha / total
    std = torch.sqrt(alpha * beta / (total * total * (total + 1.0)))
    return (native - mean) / std


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                         nn.Linear(width, outputs))


class FreeModel(nn.Module):
    """A = f(s, u) - mean_j f(s, u'_j): any function of the action, centred by the model itself."""

    def __init__(self, obs_dim, act_dim, width):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, width)

    def forward(self, obs, u, base):
        def f(actions):
            return self.net(torch.cat([obs[:, None].expand(-1, actions.shape[1], -1), actions], -1))[..., 0]

        return f(u) - f(base).mean(1, keepdim=True)


class QuadModel(nn.Module):
    """A = g(s).u + 0.5 sum h(s) (u^2 - 1): a local quadratic, exactly centred under pi_old."""

    def __init__(self, obs_dim, act_dim, width):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, width)

    def forward(self, obs, u, base):
        g, h = self.net(obs)[:, None].chunk(2, -1)
        return (g * u).sum(-1) + 0.5 * (h * (u * u - 1.0)).sum(-1)


MODELS = {"free": FreeModel, "quad": QuadModel}


def reinitialize(model, seed):
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()


# ---------------------------------------------------------------- data

class Samples:
    """Flat samples on the GPU: obs, u of the scored actions (n, A, dim), fixed baseline draws (n, Kc, dim)."""

    def __init__(self, obs, native, alpha, beta, base_draws):
        self.obs = obs
        self.u = standardize(native, alpha[:, None], beta[:, None])
        draws = Beta(alpha, beta).sample((base_draws,)).transpose(0, 1).clamp(SAMPLE_EPS, 1 - SAMPLE_EPS)
        self.base = standardize(draws, alpha[:, None], beta[:, None])

    def __len__(self):
        return self.obs.shape[0]

    def take(self, index):
        return self.obs[index], self.u[index], self.base[index]


def chunk_samples(chunks, device, base_draws):
    cat = lambda key: torch.cat([chunk[key] for chunk in chunks]).to(device)  # noqa: E731
    return Samples(cat("obs"), cat("native")[:, None], cat("alpha"), cat("beta"), base_draws)


def corrected_targets(chunks, phis, gamma, lam, device):
    """Per chunk: GAE_lambda over delta_t - phi_t, plus phi_t (phi flat (step, env) per chunk)."""
    out = []
    for chunk, phi in zip(chunks, phis):
        trace = {key: value.to(device) for key, value in chunk["trace"].items()}
        phi = phi.view_as(trace["rewards"])
        advantages, _ = compute_gae(trace["rewards"] - phi, trace["values"], torch.zeros_like(trace["rewards"]),
                                    trace["truncations"], trace["truncation_values"], trace["tail"], gamma, lam)
        out.append((advantages + phi).flatten())
    return torch.cat(out)


def check_trace(chunks, gamma, device):
    """The saved GAE inputs must reproduce the saved advantages exactly (the M2 rebuild's premise)."""
    for chunk in chunks[:2]:
        zero = [torch.zeros(chunk["obs"].shape[0], device=device)]
        for lam in TARGET_LAMBDAS:
            rebuilt = corrected_targets([chunk], zero, gamma, lam, device)
            assert torch.allclose(rebuilt, chunk[f"advantages_{lam:g}"].to(device), atol=1e-5), lam


# ---------------------------------------------------------------- fitting

class Fitter:
    """One model instance per kind, re-initialized per fit, so each kind compiles one training graph."""

    def __init__(self, obs_dim, act_dim, cli, device):
        self.cli, self.device = cli, device
        self.models = {name: cls(obs_dim, act_dim, cli.width).to(device) for name, cls in MODELS.items()}
        self.steps = {name: torch.compile(self._make_step(model), fullgraph=False, dynamic=False)
                      for name, model in self.models.items()}

    @staticmethod
    def _make_step(model):
        def loss(obs, u, base, target):
            return F.mse_loss(model(obs, u, base)[:, 0], target)
        return loss

    @torch.no_grad()
    def predict(self, name, samples, chunk=8192):
        model = self.models[name]
        return torch.cat([model(*samples.take(slice(start, start + chunk)))
                          for start in range(0, len(samples), chunk)])

    def fit(self, name, train, target, val, val_target, seed):
        """Adam on standardized targets, best-of-validation weights; predictions come back in target units."""
        cli, model, step = self.cli, self.models[name], self.steps[name]
        reinitialize(model, seed)
        scale = target.std()
        target, val_target = target / scale, val_target / scale
        optimizer = torch.optim.Adam(model.parameters(), lr=cli.lr, fused=True)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        best, best_state, stale, steps = float("inf"), None, 0, 0
        start = time.perf_counter()
        while steps < cli.max_steps and stale < cli.patience:
            for _ in range(cli.eval_every):
                index = torch.randint(len(train), (cli.batch,), device=self.device, generator=generator)
                optimizer.zero_grad(set_to_none=True)
                step(*train.take(index), target[index]).backward()
                optimizer.step()
            steps += cli.eval_every
            loss = float(F.mse_loss(self.predict(name, val)[:, 0], val_target))
            if loss < best - 1e-5:
                best, stale = loss, 0
                best_state = {key: value.clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        model.load_state_dict(best_state)
        return scale, {"steps": steps, "val_mse": best, "fit_s": time.perf_counter() - start}


# ---------------------------------------------------------------- scoring

def r2_signal(prediction, truth, se):
    """Noise-corrected squared correlation: corr^2 over the oracle's reliability."""
    reliability = 1.0 - np.mean(se ** 2) / truth.var()
    corr = np.corrcoef(prediction, truth)[0, 1]
    return float(corr ** 2 / reliability), float(corr), float(reliability)


def score(prediction, evaluation, horizons):
    """(a) sampled action and (b) within-state over the fresh actions, per oracle horizon."""
    out = {}
    for h in horizons:
        truth, se = evaluation[f"a_true_{h}"], evaluation[f"a_true_se_{h}"]
        cell = {}
        cell["sampled"], cell["corr_sampled"], cell["rel_sampled"] = r2_signal(prediction[:, 0], truth[:, 0], se[:, 0])
        if prediction.shape[1] > 1:
            fresh = prediction[:, 1:] - prediction[:, 1:].mean(1, keepdims=True)
            fresh_truth = truth[:, 1:] - truth[:, 1:].mean(1, keepdims=True)
            cell["within"], cell["corr_within"], cell["rel_within"] = r2_signal(fresh.ravel(), fresh_truth.ravel(),
                                                                                se[:, 1:].ravel())
            # Calibration: the least-squares scale that maps the model's within-state A onto A_true.
            cell["slope_within"] = float((fresh * fresh_truth).sum() / (fresh * fresh).sum())
        out[f"{h}"] = cell
    return out


def row_text(row, horizons):
    cells = " ".join(f"H{h}: {row['score'][f'{h}']['sampled']:.3f}"
                     + (f"/{row['score'][f'{h}']['within']:.3f}" if "within" in row["score"][f"{h}"] else "")
                     for h in horizons)
    timing = f" [{row['fit_s']:.0f}s, {row['steps']} steps]" if "fit_s" in row else ""
    return f"  {row['model']:8s} {row['target']:10s} n={row['n_chunks']:<3} {cells}{timing}"


# ---------------------------------------------------------------- driver

def bench_file(path, cli, device, fitter):
    data = torch.load(path, map_location="cpu", weights_only=False)
    meta, evaluation_cpu = data["meta"], data["eval"]
    gamma = meta["gamma"]
    horizons = [h for h in cli.horizons if f"a_true_{h}" in evaluation_cpu]
    chunks = data["model_chunks"]
    val_chunks, pool = chunks[-cli.val_chunks:], chunks[:-cli.val_chunks]
    check_trace(pool, gamma, device)
    print(json.dumps({"data": str(path), "snapshot": meta["snapshot"], "global_step": meta["global_step"],
                      "pool_chunks": len(pool), "val_chunks": len(val_chunks),
                      "eval_states": int(evaluation_cpu["obs"].shape[0]), "K": meta["K"], "M": meta["M"],
                      **{f"precision_{h}": evaluation_cpu[f"precision_{h}"] for h in horizons}}), flush=True)
    evaluation = {key: value.double().numpy() for key, value in evaluation_cpu.items()
                  if key.startswith("a_true")}
    eval_samples = Samples(evaluation_cpu["obs"].to(device), evaluation_cpu["native_all"].to(device),
                           evaluation_cpu["alpha"].to(device), evaluation_cpu["beta"].to(device), cli.eval_draws)
    rows = []

    def record(row):
        rows.append(row)
        print(row_text(row, horizons), flush=True)

    for lam in (*TARGET_LAMBDAS, 1.0):
        key = f"advantages_{lam:g}"
        if key in evaluation_cpu:
            raw = evaluation_cpu[key].double().numpy()[:, None]
            record({"model": "raw", "target": f"A_{lam:g}", "n_chunks": 1, "score": score(raw, evaluation, horizons)})

    def fit_and_score(name, train_chunks, targets, val_targets, seed, label, n_chunks):
        train = chunk_samples(train_chunks, device, cli.train_draws)
        scale, info = fitter.fit(name, train, targets, val, val_targets, seed)
        prediction = (fitter.predict(name, eval_samples) * scale).double().cpu().numpy()
        row = {"model": name, "target": label, "n_chunks": n_chunks, **info,
               "score": score(prediction, evaluation, horizons)}
        record(row)
        return scale

    val = chunk_samples(val_chunks, device, cli.train_draws)
    sizes = sorted({min(size, len(pool)) for size in cli.chunk_counts})
    print("M1: (a) sampled / (b) within-state R2sig per oracle horizon", flush=True)
    for name in cli.models:
        for lam in TARGET_LAMBDAS:
            key = f"advantages_{lam:g}"
            val_targets = torch.cat([chunk[key] for chunk in val_chunks]).to(device)
            for size in sizes:
                targets = torch.cat([chunk[key] for chunk in pool[:size]]).to(device)
                fit_and_score(name, pool[:size], targets, val_targets, cli.seed + size, f"A_{lam:g}", size)

    # M2: iterated, cross-fitted future-action correction on the whole pool.
    folds = [pool[0::2], pool[1::2]]
    fold_samples = [chunk_samples(fold, device, cli.train_draws) for fold in folds]
    eval_chunk_samples = chunk_samples(data["eval_chunks"], device, cli.train_draws)
    eval_index = evaluation_cpu["index"].to(device)

    def phis(name, fitted, samples_list, chunk_lists, owners):
        """Per chunk, phi from the fold model(s) not trained on it; owner None averages both folds."""
        out = []
        for samples, chunk_list, owner in zip(samples_list, chunk_lists, owners):
            predictions = []
            for fold, (state, scale) in enumerate(fitted):
                if fold != owner:
                    fitter.models[name].load_state_dict(state)
                    predictions.append(fitter.predict(name, samples)[:, 0] * scale)
            out.append(list(torch.stack(predictions).mean(0).split(chunk_list[0]["obs"].shape[0])))
        return out

    def fit_folds(name, targets, val_targets, round_):
        fitted, start = [], time.perf_counter()
        for fold in (0, 1):
            scale, _ = fitter.fit(name, fold_samples[fold], targets[fold], val, val_targets, cli.seed + 100 * round_ + fold)
            fitted.append(({k: v.clone() for k, v in fitter.models[name].state_dict().items()}, scale))
        return fitted, time.perf_counter() - start

    @torch.no_grad()
    def ensemble_eval(name, fitted):
        predictions = []
        for state, scale in fitted:
            fitter.models[name].load_state_dict(state)
            predictions.append(fitter.predict(name, eval_samples) * scale)
        return torch.stack(predictions).mean(0).double().cpu().numpy()

    print("M2: iterated cross-fitted correction (fold ensemble); '>tgt' rows score the corrected A_hat itself",
          flush=True)
    key = "advantages_0.8"
    for name in cli.models:
        # Round 0 (shared by every lambda): fold models on A_hat_0.8.
        base_targets = [torch.cat([chunk[key] for chunk in fold]).to(device) for fold in folds]
        base_val = torch.cat([chunk[key] for chunk in val_chunks]).to(device)
        round0, seconds = fit_folds(name, base_targets, base_val, 0)
        record({"model": name, "target": "A_0.8", "n_chunks": len(pool), "round": 0, "fit_s": seconds, "steps": 0,
                "score": score(ensemble_eval(name, round0), evaluation, horizons)})
        for lam in cli.m2_lambdas:
            fitted = round0
            for round_ in range(1, cli.m2_rounds + 1):
                # Rebuild every target with the other fold's model; validation and EVAL chunks use the ensemble.
                fold_phis = phis(name, fitted, fold_samples, folds, (0, 1))
                targets = [corrected_targets(folds[fold], fold_phis[fold], gamma, lam, device) for fold in (0, 1)]
                val_phi, eval_phi = phis(name, fitted, [val, eval_chunk_samples], [val_chunks, data["eval_chunks"]],
                                         (None, None))
                val_targets = corrected_targets(val_chunks, val_phi, gamma, lam, device)
                corrected_eval = corrected_targets(data["eval_chunks"], eval_phi, gamma, lam, device)[eval_index]
                record({"model": f"{name}>tgt", "target": f"M2_{lam:g}", "n_chunks": len(pool), "round": round_,
                        "score": score(corrected_eval.double().cpu().numpy()[:, None], evaluation, horizons)})
                fitted, seconds = fit_folds(name, targets, val_targets, round_)
                record({"model": name, "target": f"M2_{lam:g}", "n_chunks": len(pool), "round": round_,
                        "fit_s": seconds, "steps": 0, "score": score(ensemble_eval(name, fitted), evaluation, horizons)})
    return {"meta": meta, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="+", type=Path)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--chunk-counts", type=int, nargs="+", default=[1, 4, 16, 1000], help="capped at the pool")
    parser.add_argument("--val-chunks", type=int, default=4)
    parser.add_argument("--horizons", type=int, nargs="+", default=[25, 200])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--train-draws", type=int, default=8, help="fixed baseline draws per training sample (free)")
    parser.add_argument("--eval-draws", type=int, default=64, help="baseline draws per EVAL state (free)")
    parser.add_argument("--m2-lambdas", type=float, nargs="+", default=[0.8, 0.95])
    parser.add_argument("--m2-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(matmul_precision="high")
    torch.manual_seed(cli.seed)
    device = torch.device("cuda")
    report = []
    fitter = None
    for path in cli.data:
        if fitter is None:
            probe = torch.load(path, map_location="cpu", weights_only=False)["eval"]
            fitter = Fitter(probe["obs"].shape[1], probe["native"].shape[1], cli, device)
            del probe
        report.append(bench_file(path, cli, device, fitter))
        if cli.out is not None:
            cli.out.parent.mkdir(parents=True, exist_ok=True)
            cli.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

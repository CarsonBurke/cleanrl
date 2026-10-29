"""B. What can an estimator that POOLS iterations reach, and where does the critic sit?

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/07_B_ceiling.py > benchmarks/reppo_rootcause/data/07_B_log.txt

Oracle-regression bound for the mean-gradient field G(s) (pre-tanh-mean coordinates), per feature space Phi with
WHITENED features (E[F F^T] = I on the chunk):
    b      = |E_s[G(s) (x) F(s)]|^2      the true field's energy inside span(Phi)   (combined unbiased estimate, 07_D3)
    E|G|^2 = total field energy          cross product of two independent truths at the same state:
                                         FD at a0 (rollout half A) x fresh-action SF truth (rollout half B)
    f      = b / E|G|^2                  share of the true field that Phi can represent at all
    tau    = |V_sf|^2 - b                noise energy of ONE chunk's unbiased score-function projection (GAE 0.95)
An unbiased projection estimator pooling M chunks of a static field has aggregate cosine sqrt(M b / (M b + tau)) and
per-state field correlation sqrt(f) times that (Cramer-Rao for the linear model: tau is the per-chunk Fisher noise).
The critic is placed on the same axes: its per-state field correlation E[g_c . G] / sqrt(E|g_c|^2 E|G|^2).

Second part: how much of its action-gradient field does the critic rewrite in ONE iteration? Consecutive snapshots of
the v55-dual run (iterations 244/245 and 482/483), evaluated on the oracle chunks' states (a different run: off-run
states, on-distribution in step count only).
"""
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

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
d2 = _load("d2", "07_D2_chunk_reference.py")
d3 = _load("d3", "07_D3_combined.py")
rng = np.random.default_rng(0)


def whiten(features, floor=1e-6):
    cov = features.T @ features / len(features)
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > floor * vals.max()
    return features @ (vecs[:, keep] / np.sqrt(vals[keep]))


def field_energy(d, horizon=100):
    """E_s |G(s)|^2 from two independent truths at each oracle state, and the critic's per-state field correlation."""
    truth, _ = d1.truth_fields(d, horizon)
    fd, fresh = truth["fd_a0_h0.2"], truth["fresh"]
    per_state = 0.5 * ((fd[..., ::2].mean(-1) * fresh[..., 1::2].mean(-1)).sum(1) + (fd[..., 1::2].mean(-1) * fresh[..., ::2].mean(-1)).sum(1))
    g_c = (d.grad * (1 - d.action ** 2))[d.index]
    cross = (g_c * fresh.mean(-1)).sum(1)  # fresh truth is independent of a0: unbiased for G_c(s) . G(s)
    own = (g_c ** 2).sum(1)

    def stats(idx):
        e = per_state[idx].mean()
        return e, cross[idx].mean() / np.sqrt(max(own[idx].mean() * e, 1e-12)) if e > 0 else np.nan
    point = stats(np.arange(len(per_state)))
    draws = np.array([stats(rng.integers(0, len(per_state), len(per_state))) for _ in range(2000)])
    return point, np.nanpercentile(draws, [2.5, 97.5], axis=0), np.mean(draws[:, 0] > 0)


def bound(name):
    d = common.load(name)
    (energy, critic_corr), ci, p_pos = field_energy(d)
    print(f"\n===== {name}: E|G|^2 = {energy:.3f} [{ci[0, 0]:.3f}, {ci[1, 0]:.3f}] P(>0) {p_pos:.3f} (rms per state {np.sqrt(max(energy, 0)):.3f}); "
          f"critic per-state field corr {critic_corr:+.3f} [{ci[0, 1]:+.3f}, {ci[1, 1]:+.3f}]; critic field rms {np.sqrt(((d.grad * (1 - d.action ** 2)) ** 2).sum(1).mean()):.3f}")
    est = d2.fields(d)
    raw = d1.feature_maps(d)
    truth = d1.truth_fields(d, 100)[0]["both"].mean(-1)
    lane, step = d.row, np.arange(len(d.q)) // common.NUM_ENVS
    block = lane * 16 + step * 16 // common.NUM_STEPS
    lane_chunk = np.repeat(np.arange(common.NUM_ENVS), 16)
    n_chunk = np.bincount(block, minlength=256).astype(float)
    spaces = {"const (6 dof)": raw["const(6)"], "linear (108 dof)": whiten(raw["linear(108)"])}
    hidden = whiten(raw["lastlayer(3078)"])
    for k in (32, 128, hidden.shape[1]):
        # leading principal directions of the actor's last hidden layer (whitening keeps the eigen-order ascending)
        spaces[f"actor last layer, top {k} PCs ({6 * k} dof)"] = hidden[:, -k:]
    print("   feature space                         | b (field energy in span)        | f = b/E|G|^2 | tau/b (1 chunk) | aggregate cos M=1 / 10 / 30 | per-state field corr ceiling M=1 / 10 / 30 / inf")
    for label, feats in spaces.items():
        c_oracle = d1.contributions(truth, feats[d.index], np.arange(len(d.index)), len(d.index))
        c_chunk = {k: d1.contributions(est[k], feats, block, 256) for k in d3.ESTIMATORS}
        results = {}
        d3.analyse(c_chunk, n_chunk, lane_chunk, c_oracle, np.ones(len(d.index)), lane[d.index], "sf_gae0.95", io.StringIO(), "x", results)
        b = results["x|sf_gae0.95|b"]
        norm_sf = results["x|sf_gae0.95|sf_gae0.95"]["norm"]
        bb = b["b"]
        if bb <= 0:
            print(f"   {label:38s}| {bb:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] | not resolved (b <= 0)")
            continue
        tau = max(norm_sf ** 2 - bb, 0.0)
        f = bb / energy
        agg = [np.sqrt(m * bb / (m * bb + tau)) for m in (1, 10, 30)]
        per = [np.sqrt(min(f, 1.0)) * a for a in agg] + [np.sqrt(min(f, 1.0))]
        crit = results["x|sf_gae0.95|critic"]
        print(f"   {label:38s}| {bb:.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] | {f:.3f} [{b['ci'][0] / energy:+.3f}, {b['ci'][1] / energy:+.3f}] | {tau / bb:6.2f} "
              f"| {agg[0]:.2f} / {agg[1]:.2f} / {agg[2]:.2f} | {per[0]:.2f} / {per[1]:.2f} / {per[2]:.2f} / {per[3]:.2f} | critic aggregate cos {crit['cos']:+.2f} [{crit['cos_ci'][0]:+.2f}, {crit['cos_ci'][1]:+.2f}]")
    return d


def rewrite(d, pair_dir, steps, trainer_file):
    """Within-state agreement of two consecutive critics on the same states and actions."""
    root = common.REPO / pair_dir
    nets = []
    for step in steps:
        snapshot = torch.load(root / f"snapshot_{step}.pt", map_location="cpu", weights_only=False)
        trainer = common.load_module(trainer_file)
        args = SimpleNamespace(**snapshot["args"])
        actor = trainer.Actor(17, 6, args)
        actor.net.load_state_dict(snapshot["actor_net"])
        critic = trainer.Critic(17, 6, args, torch.device("cpu"))
        critic.load_state_dict(snapshot["critic"])
        nets.append((snapshot, actor.eval(), critic.eval()))
    # Re-normalize the chunk's raw observations with the first snapshot's own per-lane statistics.
    data = torch.load(common.REPO / common.DATASETS[d.name][0], map_location="cpu", weights_only=False)
    raw = data["chunk"]["raw_obs"].double()
    rows = torch.as_tensor(d.row)
    snap = nets[0][0]
    mean_o, scale = torch.as_tensor(snap["obs_means"]), torch.sqrt(torch.as_tensor(snap["obs_variances"]) + snap["obs_epsilon"])
    obs = ((raw - mean_o[rows]) / scale[rows]).clamp(-10, 10).float()
    sel = torch.as_tensor(np.random.default_rng(1).choice(len(obs), 4096, replace=False))
    obs = obs[sel]
    with torch.no_grad():
        mean, std = nets[0][1](obs)
        noise = torch.randn((len(obs), 8, 6), generator=torch.Generator().manual_seed(0))
        actions = torch.tanh(mean[:, None] + std[:, None] * noise)
        qs = [net[2].value(obs[:, None].expand(-1, 8, -1).reshape(-1, 17), actions.reshape(-1, 6)).view(-1, 8).double().numpy() for net in nets]
        mean2, std2 = nets[1][1](obs)
        kl = (torch.log(std2 / std) + (std ** 2 + (mean - mean2) ** 2) / (2 * std2 ** 2) - 0.5).sum(-1).mean()
    grads = []
    for net in nets:
        _, g = common.critic_grad(net[2], obs, actions[:, 0])
        grads.append((g * (1 - actions[:, 0] ** 2)).double().numpy())
    adv = [x - x.mean(1, keepdims=True) for x in qs]
    energy = (grads[0] ** 2).sum(1)
    for label, keep in (("all", np.ones(len(obs), bool)), ("drop top 1% gradient energy", energy < np.quantile(energy, 0.99))):
        pooled = lambda x, y: (x[keep] * y[keep]).sum() / np.sqrt((x[keep] ** 2).sum() * (y[keep] ** 2).sum())
        per_state = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(adv[0][keep], adv[1][keep])])
        cos = (grads[0][keep] * grads[1][keep]).sum(1) / np.sqrt((grads[0][keep] ** 2).sum(1) * (grads[1][keep] ** 2).sum(1))
        delta = grads[1][keep] - grads[0][keep]
        print(f"   {label}: within-state advantage corr pooled {pooled(adv[0], adv[1]):+.3f}, per-state median {np.median(per_state):+.3f} (p10 {np.quantile(per_state, 0.1):+.2f}); "
              f"gradient field pooled cos {pooled(grads[0], grads[1]):+.3f}, per-state cos median {np.median(cos):+.3f} (p10 {np.quantile(cos, 0.1):+.2f}); "
              f"rms |g_k| {np.sqrt((grads[0][keep] ** 2).sum(1).mean()):.3f}, rms |g_k+1 - g_k| {np.sqrt((delta ** 2).sum(1).mean()):.3f}")
    print(f"   policy KL(k || k+1) on these states {float(kl):.4f}; critic level change sd {np.std(qs[1].mean(1) - qs[0].mean(1)):.3f} (mean {np.mean(qs[1].mean(1) - qs[0].mean(1)):+.3f})")


def main():
    loaded = {name: bound(name) for name in ("v23_4M", "v29h8_8M")}
    pair = "runs/HalfCheetah-v4__reppo_landscape_ref_v55dual_pairs__1__1790712519"
    trainer = "cleanrl/reppo/ppo_continuous_action_reppo_regretprobe_v55.py"
    print("\n===== one-iteration rewrite of the critic's action dependence (v55-dual consecutive snapshots; 4096 off-run states, 8 policy actions each)")
    for name, steps in (("v23_4M", (4013696, 4030080)), ("v29h8_8M", (7913088, 7929472))):
        print(f" snapshots {steps} on the states of {name}")
        rewrite(loaded[name], pair, steps, trainer)


if __name__ == "__main__":
    main()

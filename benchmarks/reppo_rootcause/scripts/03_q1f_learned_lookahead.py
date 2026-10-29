"""Channel B test on the later-stage oracle chunks: can a one-step action-effect model fit on ONE on-policy chunk,
composed with the snapshot critic's own V', rank actions better than the critic's direct Q(s, a) readout?

  s'(s, a_k) = s'_observed(s, a0) + (z_k - z_0) x phi(s) . W      (ridge on the chunk's deterministic transitions;
                                                                   phi = [1, obs]; W fit leave-env-rows-out)
  r(s, a_k)  = r_observed + same form
  score_k    = r_k + gamma * V'(s'_k),  V'(s') = mean_j [Q(s', a'_j) - T log pi(a'_j | s')], 16 common-noise draws

CPU only (torch for the frozen actor/critic forward passes). Scored like the critic: within-state corr over the
8 fresh actions against the H=300 branching truth, noise-corrected, bootstrap over states.
"""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("prep", HERE / "03_prep_oracle_chunks.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
spec = importlib.util.spec_from_file_location("q1", HERE / "03_q1_estimation_ceiling.py")
q1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q1)


@torch.no_grad()
def main():
    for name in (sys.argv[1:] or ["v23_4M", "v29h8_8M", "v29h8_12M", "v23_30M"]):
        path, trainer_path = prep.DATASETS[name]
        data = torch.load(REPO / path, map_location="cpu", weights_only=False)
        meta = data["meta"]
        snap_path = Path(meta["snapshot"])
        snapshot = torch.load(snap_path if snap_path.is_absolute() else REPO / snap_path, map_location="cpu", weights_only=False)
        trainer_path = Path(trainer_path or snapshot.get("trainer_file") or "cleanrl/reppo/ppo_continuous_action_reppo_latentroll_v27.py")
        trainer = prep.load_module(trainer_path if trainer_path.is_absolute() else REPO / trainer_path)
        args = SimpleNamespace(**snapshot["args"])
        actor = trainer.Actor(17, 6, args)
        actor.net.load_state_dict(snapshot["actor_net"])
        critic = trainer.Critic(17, 6, args, torch.device("cpu"))
        critic.load_state_dict(snapshot["critic"])
        actor.eval(), critic.eval()
        d = np.load(q1.DATA / f"03_chunk_{name}.npz")
        chunk = data["chunk"]
        rows = torch.as_tensor(d["row"])
        mean_o = data["norm"]["obs_means"].double()
        scale = torch.sqrt(data["norm"]["obs_variances"].double() + data["norm"]["obs_epsilon"])
        nxt = ((chunk["raw_next_obs"].double() - mean_o[rows]) / scale[rows]).clamp(-meta["obs_clip"], meta["obs_clip"]).numpy()
        obs, act, mean, std = (d[k].astype(np.float64) for k in ("obs", "action", "mean", "std"))
        reward, row, idx, s_row = d["reward"].astype(np.float64), d["row"], d["s_index"], d["s_row"]
        z = (np.arctanh(np.clip(act, -1 + 1e-7, 1 - 1e-7)) - mean) / std
        s_obs, s_act, s_mean, s_std = (d[k].astype(np.float64) for k in ("s_obs", "s_actions", "s_mean", "s_std"))
        z_fresh = (np.arctanh(np.clip(s_act[:, 50:58], -1 + 1e-7, 1 - 1e-7)) - s_mean[:, None]) / s_std[:, None]
        z0 = z[idx]
        true_fresh = d["ret_300"].astype(np.float64)[:, 50:58]
        crit_fresh = d["s_critic_q"][:, 50:58]
        rng = np.random.default_rng(3)
        proj_b = (rng.normal(size=(17, 768)) / np.sqrt(17), rng.normal(size=768))
        target = np.concatenate([nxt - obs, reward[:, None]], 1)  # (N, 18): next-state delta and reward
        pred = np.zeros((512, 8, 18))
        r2_num, r2_den = np.zeros(18), np.zeros(18)
        for kind in ("linear",):
            for fold in np.array_split(np.arange(16), 4):
                train = ~np.isin(row, fold)
                phi_b = q1.state_features(obs, "rf", proj_b)
                mu_b, sd_b = phi_b[train].mean(0), phi_b[train].std(0) + 1e-8
                phi_b = (phi_b - mu_b) / sd_b
                phi_b[:, 0] = 1.0
                resid = target - phi_b @ q1.ridge(phi_b[train], target[train], 1.0)  # state-only part removed
                phi, phi_s = q1.state_features(obs, kind, None), q1.state_features(s_obs, kind, None)
                mu, sd = phi[train].mean(0), phi[train].std(0) + 1e-8
                phi, phi_s = (phi - mu) / sd, (phi_s - mu) / sd
                phi[:, 0], phi_s[:, 0] = 1.0, 1.0
                X = q1.z_design(z, phi)
                W = q1.ridge(X[train], resid[train], 1.0)  # (108, 18)
                held = ~train
                r2_num += ((resid[held] - X[held] @ W) ** 2).sum(0)
                r2_den += (resid[held] ** 2).sum(0)
                test = np.isin(s_row, fold)
                dz = z_fresh - z0[:, None]  # (512, 8, 6)
                Xs = (dz[:, :, :, None] * phi_s[:, None, None, :]).reshape(512, 8, -1)
                pred[test] = (Xs @ W)[test]
        r2 = 1 - r2_num / r2_den
        s_next = torch.as_tensor(nxt[idx][:, None] + pred[:, :, :17], dtype=torch.float32)  # (512, 8, 17)
        r_k = reward[idx][:, None] + pred[:, :, 17]
        samples = 16
        noise = torch.randn((512, 1, samples, 6), generator=torch.Generator().manual_seed(0)).expand(-1, 8, -1, -1)
        m, s = actor(s_next)
        pre = m[:, :, None] + s[:, :, None] * noise
        log_det = 2.0 * (math.log(2.0) - pre - torch.nn.functional.softplus(-2.0 * pre))
        logp = (-0.5 * noise.square() - s[:, :, None].log() - 0.5 * math.log(2 * math.pi) - log_det).sum(-1)
        q = critic.value(s_next[:, :, None].expand(-1, -1, samples, -1).reshape(-1, 17), torch.tanh(pre).reshape(-1, 6)).view(512, 8, samples)
        v_next = (q - float(d["temperature"]) * logp).mean(2).double().numpy()
        gamma = float(d["gamma"])
        scores = {"critic Q(s,a) direct": crit_fresh, "learned 1-step lookahead r+gV'(s')": r_k + gamma * v_next,
                  "  reward part only": r_k, "  gamma V'(s') part only": gamma * v_next}
        print(f"\n=== {name}: action-effect model held-out R^2 on the state-residual: next-obs dims median {np.median(r2[:17]):.3f} "
              f"(min {r2[:17].min():.3f}, max {r2[:17].max():.3f}), reward {r2[17]:.3f}")
        for label, a_hat in scores.items():
            full = q1.boot(q1.adv_stats, a_hat, true_fresh)
            sub = q1.boot(q1.adv_stats, a_hat[d["cruising"]], true_fresh[d["cruising"]])
            print(f"   {label:36s} adv corr corrected all {q1.fmt(full, 1)} | cruising {q1.fmt(sub, 1)} | within rms {np.sqrt(((a_hat - a_hat.mean(1, keepdims=True)) ** 2).mean()):.3f}", flush=True)
        both = q1.unit_scale(crit_fresh) if hasattr(q1, "unit_scale") else None


if __name__ == "__main__":
    main()

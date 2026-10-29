"""Red-team 08: the score-function update-gain instrument evaluated on REAL data, for a REAL REPPO update.

Data (no simulation, CPU): the v29-H8 snapshots at 8.01M / 12.01M hold the actor before and after one full
iteration's update (`actor_net_before_update`, `actor_net`) and both critics; the oracle file holds a fresh
16 x 1024 on-policy rollout of the POST-update policy with trainer-exact soft lambda-0.8 targets. Those rows
were seen by neither the update nor either critic, so this is the held-out "backward" form:

    gain_hat = - mean_i  A_new(s_i, a_i) * [log pi_old(u_i | s_i) - log pi_new(u_i | s_i)],   (s_i, a_i) ~ pi_new
    E gain_hat = E_{s ~ d_new, a ~ pi_new} A (1 - pi_old/pi_new) + O(KL^1.5) ~ first-order J(new) - J(old), per step

in the units of a first-action advantage (scaled soft return). A_new = G_lambda - mean_k Q(s, a_k).
Compared with the critic's own predicted gain on the same rows, E_s[Q(s, a_new) - Q(s, a_old)] (common noise).
Standard errors: iid rows, and a block bootstrap over the 16 env lanes.

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/08_sf_instrument_real.py
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
torch.set_num_threads(3)
torch.manual_seed(0)
BASE = "runs/reppo_latentroll_oracle/HalfCheetah-v4__reppo_diagnose_v29_h8_reward_16M__1__1790481916_{}.pt"
LOG_SQRT_2PI = 0.5 * float(np.log(2 * np.pi))


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"trainer_{Path(path).stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gauss_logp(u, mean, std):
    return (-0.5 * ((u - mean) / std) ** 2 - std.log() - LOG_SQRT_2PI).sum(-1)


def lane_boot(x, lanes, n=4000, rng=np.random.default_rng(0)):
    sums = np.array([x[lanes == l].sum() for l in range(16)]); cnt = np.array([(lanes == l).sum() for l in range(16)])
    idx = rng.integers(0, 16, size=(n, 16))
    return float((sums[idx].sum(1) / cnt[idx].sum(1)).std())


def winsor(a, q):
    lo, hi = np.quantile(a, [q, 1 - q])
    return np.clip(a, lo, hi)


@torch.no_grad()
def main():
    for step in (8011392, 12009088):
        data = torch.load(REPO / BASE.format(step), map_location="cpu", weights_only=False)
        meta = data["meta"]
        snap = torch.load(meta["snapshot"], map_location="cpu", weights_only=False)
        trainer = load_module(snap["trainer_file"])
        args = SimpleNamespace(**snap["args"])

        def actor_from(sd):
            a = trainer.Actor(17, 6, args); a.net.load_state_dict(sd); return a.eval()

        def critic_from(sd):
            c = trainer.Critic(17, 6, args, torch.device("cpu")); c.load_state_dict(sd); return c.eval()

        new, old = actor_from(snap["actor_net"]), actor_from(snap["actor_net_before_update"])
        critic_new, critic_old = critic_from(snap["critic"]), critic_from(snap["critic_before_update"])
        T = float(meta["temperature"])

        chunk = data["chunk"]
        rows = torch.as_tensor(chunk["row"])
        lanes = (rows % 16).numpy()
        mean_o = data["norm"]["obs_means"].double()
        scale = torch.sqrt(data["norm"]["obs_variances"].double() + data["norm"]["obs_epsilon"])
        obs = ((chunk["raw_obs"].double() - mean_o[rows]) / scale[rows]).clamp(-meta["obs_clip"], meta["obs_clip"]).float()
        act = chunk["actions"].float()
        u = torch.atanh(act.double().clamp(-1 + 1e-7, 1 - 1e-7))
        m_new, s_new = (x.double() for x in new(obs)); m_old, s_old = (x.double() for x in old(obs))
        eps = (u - m_new) / s_new
        ok = (act.abs() < 0.9999).all(-1)  # rows whose pre-tanh action is recoverable in float32
        logratio = gauss_logp(u, m_old, s_old) - gauss_logp(u, m_new, s_new)  # log pi_old - log pi_new
        kl_new_old = (torch.log(s_old / s_new) + (s_new ** 2 + (m_new - m_old) ** 2) / (2 * s_old ** 2) - 0.5).sum(-1)
        dmu = (m_old - m_new) / s_new
        lin_mean = (dmu * eps).sum(-1)  # linearised mean part of the log ratio
        print(f"\n=== v29 H8 @ {step/1e6:.2f}M  lambda {meta['lambda']}  T {T:.4f}  rows {len(obs)}  recoverable-u rows {int(ok.sum())}")
        print(f"eps reconstruction: mean {eps[ok].mean():+.4f} sd {eps[ok].std():.4f} (expect 0, 1);  "
              f"KL(new||old) mean {kl_new_old.mean():.4f} median {kl_new_old.median():.4f};  "
              f"rms |dmu|/sigma {dmu.pow(2).sum(-1).mean().sqrt():.3f};  sd(logratio) {logratio[ok].std():.3f}")

        # what the update did to the noise scale and to the squashed entropy (the actor maximises Q - T log pi)
        dlogs = (s_new / s_old).log().sum(-1)
        n_h = torch.randn(16, len(obs), 6, dtype=torch.float64)
        def squashed_entropy(mean, std):
            x = mean + std * n_h
            return (std.log().sum(-1) + 6 * (0.5 + LOG_SQRT_2PI) + (1 - torch.tanh(x) ** 2).clamp_min(1e-12).log().sum(-1).mean(0))
        dH = squashed_entropy(m_new, s_new) - squashed_entropy(m_old, s_old)
        print(f"update: mean sum_d dlog sigma {dlogs.mean():+.4f} (sd over states {dlogs.std():.3f});  squashed entropy change {dH.mean():+.4f} nats"
              f" -> T x dH = {T * dH.mean():+.5f} (entropy part of the soft objective, same units as the gains below)")
        # baselines: V(s) = mean_k Q(s, a_k), a_k ~ pi_new, K = 8, for both critics
        K = 8
        noise = torch.randn(K, len(obs), 6)

        def q_at(critic, mean, std, n):
            a = torch.tanh(mean.float() + std.float() * n)
            return torch.cat([critic.value(obs[i:i + 4096], a[i:i + 4096]) for i in range(0, len(obs), 4096)]).double()

        target = chunk["targets"].double()
        for cname, critic in (("post-update critic (trained through this iteration)", critic_new),
                              ("pre-update critic (the one the actor started climbing)", critic_old)):
            q_new = torch.stack([q_at(critic, m_new, s_new, noise[k]) for k in range(K)])
            q_old = torch.stack([q_at(critic, m_old, s_old, noise[k]) for k in range(K)])
            v_new = q_new.mean(0)
            adv = (target - v_new).numpy()
            pred_rows = (q_new - q_old).mean(0).numpy()  # critic-predicted first-action gain, common noise
            okn = ok.numpy()
            lr, lm = logratio.numpy(), lin_mean.numpy()
            print(f"--- baseline / predictor: {cname}")
            print(f"    sd(A) {adv.std():.3f}  (1%/99% {np.quantile(adv, .01):+.2f}/{np.quantile(adv, .99):+.2f}, max|A| {np.abs(adv).max():.1f})"
                  f"   critic-predicted gain {pred_rows.mean():+.5f} (iid se {pred_rows.std()/128:.5f}, lane se {lane_boot(pred_rows, lanes):.5f})")
            for label, a in (("raw A", adv), ("A winsorised 0.5%", winsor(adv, .005)), ("A winsorised 2.5%", winsor(adv, .025))):
                a = a - a[okn].mean()
                for form, w in (("full log-ratio", lr), ("linear mean part", lm)):
                    x = -(a * w)[okn]
                    lag1 = np.corrcoef(x[:-16], x[16:])[0, 1]  # rows are (step, env) flattened: lag 16 = next step same lane
                    print(f"    realised-gain estimate [{label:18s} | {form:16s}]: {x.mean():+.5f}  iid se {x.std()/np.sqrt(len(x)):.5f}"
                          f"  lane-boot se {lane_boot(x, lanes[okn]):.5f}  per-row sd {x.std():.3f}  lag-1 autocorr {lag1:+.3f}")
        # reference scale: training-curve gain per iteration converts as dR(return)/1e4 (reward scale 0.1, 1000-step episodes)
        print("    scale: +10 return per iteration == +0.00100 in these units")


if __name__ == "__main__":
    main()

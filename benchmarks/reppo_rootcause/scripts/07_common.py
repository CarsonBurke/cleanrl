"""Shared loader for the 07 red-team scripts. CPU only; no simulation.

Rebuilds, for one saved oracle dataset (runs/reppo_oracle, runs/reppo_latentroll_oracle), everything the
head-to-head needs from raw arrays plus the snapshot networks:

chunk (16 lanes x 1024 on-policy steps, flat index = step * 16 + lane):
    obs, a (tanh action), mean/std (pre-tanh), z = (atanh(a) - mean) / std, reward, logprob (of the action
    taken), critic Q(s, a), dQ/da at a (autograd), U_c(s) = E_pi Q_c(s, .) from `u_samples` fresh draws,
    trainer lambda-return target, upright flag.
oracle (512 chunk rows): index, actions (58), per-rollout soft returns at H = 100/200/300, spans.

Nothing here is taken from the first-round reports; reconstruction errors against the saved oracle values
are returned so callers can print them.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "benchmarks/reppo_rootcause/data"
V23 = "cleanrl/reppo/ppo_continuous_action_reppo_probe_v23.py"
V27 = "cleanrl/reppo/ppo_continuous_action_reppo_latentroll_v27.py"
DATASETS = {
    "v23_4M": ("runs/reppo_oracle/HalfCheetah-v4__reppo_probe_v23_50M__1__1790369616_4013696.pt", V23),
    "v29h8_8M": ("runs/reppo_latentroll_oracle/HalfCheetah-v4__reppo_diagnose_v29_h8_reward_16M__1__1790481916_8011392.pt", None),
    "v23_12M": ("runs/reppo_oracle/HalfCheetah-v4__reppo_probe_v23_50M__1__1790369616_12009088.pt", V23),
    "v29h8_12M": ("runs/reppo_latentroll_oracle/HalfCheetah-v4__reppo_diagnose_v29_h8_reward_16M__1__1790481916_12009088.pt", None),
    "v23_30M": ("runs/reppo_oracle/HalfCheetah-v4__reppo_probe_v23_50M__1__1790369616_30015104.pt", V23),
}
NUM_ENVS, NUM_STEPS, DIMS = 16, 1024, 6


def load_module(path):
    path = Path(path)
    path = path if path.is_absolute() else REPO / path
    name = f"trainer_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_networks(snapshot_path, trainer_path=None):
    snapshot_path = Path(snapshot_path)
    snapshot = torch.load(snapshot_path if snapshot_path.is_absolute() else REPO / snapshot_path, map_location="cpu",
                          weights_only=False)
    trainer = load_module(trainer_path or snapshot.get("trainer_file") or V27)
    args = SimpleNamespace(**snapshot["args"])
    actor = trainer.Actor(17, 6, args)
    actor.net.load_state_dict(snapshot["actor_net"])
    critic = trainer.Critic(17, 6, args, torch.device("cpu"))
    critic.load_state_dict(snapshot["critic"])
    actor.eval(), critic.eval()
    for p in list(actor.parameters()) + list(critic.parameters()):
        p.requires_grad_(False)
    return snapshot, actor, critic


def critic_grad(critic, obs, action, chunk=4096):
    """Q_c(s, a) and dQ_c/da by autograd (action space)."""
    qs, grads = [], []
    for i in range(0, len(obs), chunk):
        a = action[i:i + chunk].clone().requires_grad_(True)
        q = critic.value(obs[i:i + chunk], a)
        grads.append(torch.autograd.grad(q.sum(), a)[0])
        qs.append(q.detach())
    return torch.cat(qs), torch.cat(grads)


@torch.no_grad()
def expected_q(critic, obs, mean, std, samples, seed, chunk=4096):
    """U_c(s) = E_pi Q_c(s, .), `samples` fresh tanh-Gaussian draws per state."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.zeros(len(obs), dtype=torch.float64)
    for _ in range(samples):
        noise = torch.randn(mean.shape, generator=gen)
        action = torch.tanh(mean + std * noise)
        out += torch.cat([critic.value(obs[i:i + chunk], action[i:i + chunk]) for i in range(0, len(obs), chunk)]).double()
    return out / samples


def load(name, u_samples=16, threads=3):
    torch.set_num_threads(threads)
    path, trainer_path = DATASETS[name]
    data = torch.load(REPO / path, map_location="cpu", weights_only=False)
    meta, chunk, states = data["meta"], data["chunk"], data["states"]
    snapshot, actor, critic = load_networks(meta["snapshot"], trainer_path)
    rows = torch.as_tensor(np.asarray(chunk["row"]))
    mean_o = data["norm"]["obs_means"].double()
    scale = torch.sqrt(data["norm"]["obs_variances"].double() + data["norm"]["obs_epsilon"])
    clip = meta["obs_clip"]
    obs = ((chunk["raw_obs"].double() - mean_o[rows]) / scale[rows]).clamp(-clip, clip).float()
    next_obs = ((chunk["raw_next_obs"].double() - mean_o[rows]) / scale[rows]).clamp(-clip, clip).float()
    action = chunk["actions"].float()
    with torch.no_grad():
        mean, std = actor(obs)
        next_mean, next_std = actor(next_obs)
    pre = torch.atanh(action.double().clamp(-1 + 1e-7, 1 - 1e-7))
    z = ((pre - mean.double()) / std.double())
    q, grad = critic_grad(critic, obs, action)
    u_c = expected_q(critic, obs, mean, std, u_samples, 7)
    u_next = expected_q(critic, next_obs, next_mean, next_std, u_samples, 8)
    index = states["index"].long()
    a0_slot = meta["action_layout"]["a0_probes"][0]
    recon = {
        "obs": float((obs[index] - states["obs"]).abs().max()),
        "a0": float((action[index] - states["actions"][:, a0_slot]).abs().max()),
        "critic_q": float((q[index].double() - states["critic_q"][:, a0_slot]).abs().max()),
        "critic_grad_a0": float((grad[index].double() - states["critic_grad_a0"].double()).abs().max()),
        "u_c_vs_saved_baseline_rms": float((u_c[index] - states["critic_baseline"]).pow(2).mean().sqrt()),
        "star": float((torch.tanh(mean[index]) - states["actions"][:, 0]).abs().max()),
    }
    state = np.asarray(chunk["state"])
    out = SimpleNamespace(
        name=name, meta=meta, actor=actor, critic=critic, snapshot=snapshot, recon=recon,
        gamma=float(meta["gamma"]), lam=float(meta["lambda"]), temperature=float(meta["temperature"]),
        obs=obs, next_obs=next_obs, action=action.double().numpy(), mean=mean.double().numpy(), std=std.double().numpy(),
        z=z.numpy(), reward=chunk["rewards"].double().numpy(), logprob=chunk["logprob"].double().numpy(),
        next_logprob=chunk["next_logprob"].double().numpy(), next_q=chunk["next_q"].double().numpy(),
        target=chunk["targets"].double().numpy(), trunc=chunk["truncations"].double().numpy(),
        ends=chunk["ends"].double().numpy(), row=rows.numpy(), age=np.asarray(chunk["age"]),
        q=q.double().numpy(), grad=grad.double().numpy(), u_c=u_c.numpy(), u_next=u_next.numpy(),
        upright=np.cos(state[:, 2]) > 0.0,
        index=index.numpy(), s_actions=states["actions"].double().numpy(), s_critic_q=states["critic_q"].double().numpy(),
        s_grad_star=states["critic_grad_star"].double().numpy(), s_grad_a0=states["critic_grad_a0"].double().numpy(),
        span_star=states["span_star"].double().numpy(), span_a0=states["span_a0"].double().numpy(),
        returns={int(h): v.double().numpy() for h, v in data["returns"].items()},
        layout=meta["action_layout"], fd_steps=list(meta["fd_steps"]),
    )
    return out


def soft_gae(d, lam, baseline=None, baseline_next=None):
    """GAE(lam) advantage of the taken action for the SOFT objective the oracle measures.

    Q_soft(s_t, a_t) = E[r_t + gamma (Q_soft(s_{t+1}, a_{t+1}) - T log pi(a_{t+1} | s_{t+1}))], so with
    U(s) = E_pi Q_soft(s, .) and the realized next action's log-prob,
        delta_t = r_t - gamma T log pi(a_{t+1} | s_{t+1}) + gamma U(s_{t+1}) - U(s_t),   E[delta_t | s_t, a_t] = A(s_t, a_t)
        A_hat_t = sum_k (gamma lam)^k delta_{t+k}.
    At a truncation or the chunk end there is no realized next action: the entropy term uses the chunk's
    fresh-draw next_logprob (same expectation) and the trace is cut. U defaults to the snapshot critic's.
    """
    shape = (NUM_STEPS, NUM_ENVS)
    u = (d.u_c if baseline is None else baseline).reshape(shape)
    u_next = (d.u_next if baseline_next is None else baseline_next).reshape(shape)
    reward, logprob = d.reward.reshape(shape), d.logprob.reshape(shape)
    fresh_next_logprob = d.next_logprob.reshape(shape)
    boundary = np.maximum(d.trunc, d.ends).reshape(shape) > 0
    next_logprob = fresh_next_logprob.copy()
    next_logprob[:-1] = np.where(boundary[:-1], fresh_next_logprob[:-1], logprob[1:])
    delta = reward - d.gamma * d.temperature * next_logprob + d.gamma * u_next - u
    adv = np.zeros(shape)
    running = np.zeros(NUM_ENVS)
    for t in range(NUM_STEPS - 1, -1, -1):
        running = delta[t] + d.gamma * lam * np.where(boundary[t], 0.0, running)
        adv[t] = running
    return adv.reshape(-1)


def boot_ci(values, axis=0, lo=2.5, hi=97.5):
    values = np.asarray(values)
    return np.nanpercentile(values, lo, axis=axis), np.nanpercentile(values, hi, axis=axis)

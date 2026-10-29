"""How heavy-tailed is the actor's pathwise signal over on-policy states? CPU, gymnasium, no training.

Rolls the checkpoint's stochastic policy (16 lanes x 1000 steps, each lane with its own frozen normalization row),
then measures per-state dQ/du (u pre-tanh, the reparameterized path) and how much of the actor's parameter
gradient of mean(-Q) comes from the few states with the largest critic slopes."""
import importlib.util
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

torch.set_num_threads(3)
REPO = "/home/marvin/Documents/repositories/cleanrl"
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("v60", f"{REPO}/cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py")
v60 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v60)

for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**ck["args"])
    lanes, obs_dim = ck["obs_means"].shape
    act_dim = len(ck["temperature"])
    critic = v60.Critic(obs_dim, act_dim, args, "cpu")
    critic.load_state_dict(ck["critic"])
    critic.requires_grad_(False)
    actor = v60.Actor(obs_dim, act_dim, args)
    actor.load_state_dict(ck["actor_state"])
    mean_np = torch.as_tensor(ck["obs_means"], dtype=torch.float32)
    inv = torch.as_tensor(1.0 / np.sqrt(ck["obs_variances"] + ck["obs_epsilon"]), dtype=torch.float32)
    envs = [gym.make("HalfCheetah-v4") for _ in range(lanes)]
    raw = np.stack([env.reset(seed=777 + i)[0] for i, env in enumerate(envs)])
    gen = torch.Generator().manual_seed(0)
    O, E, R = [], [], []
    for t in range(1000):
        obs = ((torch.as_tensor(raw, dtype=torch.float32) - mean_np) * inv).clamp(-args.obs_clip, args.obs_clip)
        with torch.no_grad():
            mu, sd = actor(obs)
        eps = torch.randn(mu.shape, generator=gen)
        act = torch.tanh(mu + sd * eps).numpy()
        rew = np.zeros(lanes)
        for i, env in enumerate(envs):
            raw[i], rew[i], _, _, _ = env.step(act[i])
        O.append(obs), E.append(eps), R.append(rew)
    obs, eps = torch.cat(O), torch.cat(E)
    rewards = np.stack(R)
    n = obs.shape[0]
    mu, sd = actor(obs)
    pre = (mu + sd * eps)
    pre.retain_grad()
    q = critic.value(obs, torch.tanh(pre))
    (-q.mean()).backward()
    full = torch.cat([p.grad.flatten() for p in actor.net.parameters()]).clone()
    g_u = -pre.grad * n  # per-state dQ/du
    slope = (g_u * sd.detach()).norm(dim=-1)  # in units of Q per unit of policy noise
    order = slope.argsort(descending=True)
    qd = q.detach()
    clipped = (obs.abs() >= args.obs_clip).any(-1).float().mean()
    print(f"\n{path.split('/')[-1]} step {ck['global_step']}: episode return {rewards.sum(0).mean():.0f}; Q quantiles 0/0.5%/5%/50%/100%: "
          f"{np.quantile(qd.numpy(), [0, 0.005, 0.05, 0.5, 1]).round(1).tolist()}; rows with a clipped obs dim {clipped:.4f}; "
          f"policy sd mean {sd.mean():.3f}")
    print(f"  |sigma dQ/du| quantiles 50/90/99/99.9/max: {np.quantile(slope.numpy(), [0.5, 0.9, 0.99, 0.999, 1]).round(3).tolist()}; "
          f"share of sum |.|^2 from top 0.1/0.5/1/5% states: "
          + ", ".join(f"{(slope[order[:max(1, int(n * f))]] ** 2).sum() / (slope ** 2).sum():.2f}" for f in (0.001, 0.005, 0.01, 0.05)))
    low = qd < np.quantile(qd.numpy(), 0.02)
    print(f"  median slope in the lowest-2%-Q states {slope[low].median():.3f} vs elsewhere {slope[~low].median():.3f}; "
          f"lowest-2%-Q states hold {(slope[low] ** 2).sum() / (slope ** 2).sum():.2f} of sum |.|^2")

    def param_grad(index):
        actor.zero_grad()
        m, s = actor(obs[index])
        (-critic.value(obs[index], torch.tanh(m + s * eps[index])).sum() / n).backward()
        return torch.cat([p.grad.flatten() for p in actor.net.parameters()]).clone()

    for frac in (0.005, 0.02):
        top = order[: int(n * frac)]
        rest = order[int(n * frac):]
        gt, gr = param_grad(top), param_grad(rest)
        cos = torch.nn.functional.cosine_similarity
        print(f"  actor parameter gradient of mean(-Q): |full| {full.norm():.3f}; top {frac:.1%} states alone |g| {gt.norm():.3f}, "
              f"other {1 - frac:.1%} |g| {gr.norm():.3f}; cos(full, top) {cos(full, gt, dim=0):+.2f}, cos(full, rest) {cos(full, gr, dim=0):+.2f}, "
              f"cos(top, rest) {cos(gt, gr, dim=0):+.2f}")
    # minibatch view: how often does a 2048-row minibatch gradient point along its top-0.5% rows?
    gen2 = torch.Generator().manual_seed(1)
    vals = []
    for _ in range(6):
        rows = torch.randint(0, n, (2048,), generator=gen2)
        s_rows = slope[rows]
        top = rows[s_rows.argsort(descending=True)[:10]]
        rest = rows[s_rows.argsort(descending=True)[10:]]
        gt, gr = param_grad(top), param_grad(rest)
        vals.append((float(gt.norm() / (gt + gr).norm()), float(torch.nn.functional.cosine_similarity(gt + gr, gr, dim=0))))
    print("  2048-row minibatches, top-10 rows by slope: |g_top10| / |g_batch| and cos(g_batch, g_without_top10):",
          [(round(a, 2), round(b, 2)) for a, b in vals])

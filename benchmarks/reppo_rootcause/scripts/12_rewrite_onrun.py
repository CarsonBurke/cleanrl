"""12: one-iteration rewrite of the critic's action-gradient field on ON-RUN states, and which states the KL gate holds.

07 B measured the rewrite on 4096 states of a different run (v23 / v29 chunks) pushed through the v55 lane statistics.
Here: 3 stochastic episodes (3000 simulator steps, ~2 s) from the first snapshot of each consecutive pair give
on-distribution states; both critics and both actors are evaluated on them. CPU only, no training.
"""
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
common = import_module("07_common")
torch.set_num_threads(2)
PAIR = common.REPO / "runs/HalfCheetah-v4__reppo_landscape_ref_v55dual_pairs__1__1790712519"
TRAINER = common.load_module("cleanrl/reppo/ppo_continuous_action_reppo_regretprobe_v55.py")


def load(step):
    snapshot = torch.load(PAIR / f"snapshot_{step}.pt", map_location="cpu", weights_only=False)
    args = SimpleNamespace(**snapshot["args"])
    actor = TRAINER.Actor(17, 6, args)
    actor.net.load_state_dict(snapshot["actor_net"])
    critic = TRAINER.Critic(17, 6, args, torch.device("cpu"))
    critic.load_state_dict(snapshot["critic"])
    return snapshot, args, actor.eval(), critic.eval()


def rollout(snapshot, args, actor, episodes=3):
    env = gym.make(args.env_id)
    mean = np.asarray(snapshot["obs_means"])[0]
    inv_std = 1.0 / np.sqrt(np.asarray(snapshot["obs_variances"])[0] + snapshot["obs_epsilon"])
    rng = np.random.default_rng(12)
    observations, pitches, returns = [], [], []
    for index in range(episodes):
        raw, _ = env.reset(seed=41_000 + index)
        done, total = False, 0.0
        while not done:
            obs = np.clip((raw - mean) * inv_std, -args.obs_clip, args.obs_clip).astype(np.float32)
            with torch.no_grad():
                mu, std = actor(torch.from_numpy(obs)[None])
            raw, reward, terminated, truncated, _ = env.step(np.tanh(mu[0].numpy() + std[0].numpy() * rng.standard_normal(6)))
            observations.append(obs)
            pitches.append(float(env.unwrapped.data.qpos[2]))
            total += float(reward)
            done = terminated or truncated
        returns.append(round(total))
    env.close()
    return torch.as_tensor(np.array(observations)), np.array(pitches), returns


def field(critic, obs, actions):
    _, grad = common.critic_grad(critic, obs, actions)
    return (grad * (1 - actions ** 2)).double().numpy()


for steps in ((1015424, 1031808), (4013696, 4030080), (7913088, 7929472)):
    (snap0, args, actor0, critic0), (_, _, actor1, critic1) = load(steps[0]), load(steps[1])
    obs, pitch, returns = rollout(snap0, args, actor0)
    upright = np.cos(pitch) > 0
    with torch.no_grad():
        mean0, std0 = actor0(obs)
        mean1, std1 = actor1(obs)
        kl = (torch.log(std1 / std0) + (std0 ** 2 + (mean0 - mean1) ** 2) / (2 * std1 ** 2) - 0.5).sum(-1).numpy()
        noise = torch.randn((len(obs), 6), generator=torch.Generator().manual_seed(0))
        actions = torch.tanh(mean0 + std0 * noise)
    g0, g1 = field(critic0, obs, actions), field(critic1, obs, actions)
    sigma = std0.double().numpy()
    print(f"\n===== snapshots {steps}: {len(obs)} on-run states, episode returns {returns}, upright rows {upright.mean():.3f}")
    energy = (g0 ** 2).sum(1)
    order = np.argsort(energy)
    print(f"  top 1% / 5% of states hold {energy[order[-len(order) // 100:]].sum() / energy.sum():.2f} / {energy[order[-len(order) // 20:]].sum() / energy.sum():.2f} of gradient energy")
    masks = [("all", np.ones(len(obs), bool)), ("upright only", upright), ("drop top 1% energy", energy < np.quantile(energy, 0.99)),
             ("bottom half by |g|", energy <= np.median(energy)), ("50-90%", (energy > np.median(energy)) & (energy <= np.quantile(energy, 0.9))),
             ("90-99%", (energy > np.quantile(energy, 0.9)) & (energy <= np.quantile(energy, 0.99))), ("top 1%", energy > np.quantile(energy, 0.99))]
    for label, keep in masks:
        a, b = g0[keep], g1[keep]
        cos = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
        pooled = (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())
        change = np.sqrt(((b - a) ** 2).sum(1).mean()) / np.sqrt((a ** 2).sum(1).mean())
        # AR(1) reading: Var(delta) = 2 (1 - rho) Var  =>  rho = 1 - change^2 / 2 (equal-variance fields)
        print(f"   {label:20s} n {keep.sum():4d}  per-state cos median {np.median(cos):+.3f} (p10 {np.quantile(cos, 0.1):+.2f})  pooled cos {pooled:+.3f}  "
              f"rms change / rms field {change:.2f}  rms ratio k+1/k {np.sqrt((b ** 2).sum() / (a ** 2).sum()):.2f}")
    # KL gate: which states end the iteration outside the bound?
    bound = args.kl_bound
    outside = kl > bound
    step_energy = ((g0 * sigma) ** 2).sum(1)  # squared first-order pathwise pull in noise units
    ranks = lambda x: np.argsort(np.argsort(x))
    print(f"  KL(k -> k+1): mean {kl.mean():.4f}, median {np.median(kl):.4f}, frac > {bound} {outside.mean():.3f}, frac > 2x bound {np.mean(kl > 2 * bound):.3f}; "
          f"upright rows frac outside {outside[upright].mean():.3f}" + (f", inverted rows {outside[~upright].mean():.3f}" if (~upright).any() else ""))
    print(f"  Spearman(KL, |sigma g|) {np.corrcoef(ranks(kl), ranks(step_energy))[0, 1]:+.3f}; share of |sigma g|^2 in states outside the bound {step_energy[outside].sum() / step_energy.sum():.3f}; "
          f"outside-fraction by |sigma g| quintile {[round(float(outside[q].mean()), 2) for q in np.array_split(np.argsort(step_energy), 5)]}")
    move = ((mean1 - mean0) / std0).double().numpy()
    align = (move * g0 * sigma).sum(1) / np.sqrt((move ** 2).sum(1) * ((g0 * sigma) ** 2).sum(1))
    print(f"  cos(actual mean move, critic_k pathwise direction) per state: median {np.median(align):+.3f}, inside {np.median(align[~outside]):+.3f}, outside {np.median(align[outside]):+.3f}; "
          f"with critic_k+1: median {np.median((move * g1 * sigma).sum(1) / np.sqrt((move ** 2).sum(1) * ((g1 * sigma) ** 2).sum(1))):+.3f}")

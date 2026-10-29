"""Posture of REPPO snapshot policies on HalfCheetah: is a low-return policy running flipped onto its back?

For each snapshot (saved by a REPPO trainer's --snapshot-steps), rebuild the actor from the trainer file,
normalize observations with environment 0's saved statistics (as the trainer's evaluate() does), and roll
out mean-action and sampled-action episodes. Reports return, mean forward velocity, and the root pitch
qpos[2] (radians; |pitch| > pi/2 means the torso is upside down).

    python scripts/reppo_posture.py --trainer cleanrl/reppo/<file>.py --snapshots runs/<run>/snapshot_*.pt
"""
import argparse
import importlib.util
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("reppo_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rollout(actor, snapshot, episodes, stochastic, seed):
    args = SimpleNamespace(**snapshot["args"])
    mean = np.asarray(snapshot["obs_means"])[0]
    inv_std = 1.0 / np.sqrt(np.asarray(snapshot["obs_variances"])[0] + snapshot["obs_epsilon"])
    env = gym.make(args.env_id)
    rng = np.random.default_rng(seed)
    results = []
    for episode in range(episodes):
        raw, _ = env.reset(seed=seed + episode)
        total, pitches, velocities, done = 0.0, [], [], False
        while not done:
            obs = np.clip((raw - mean) * inv_std, -args.obs_clip, args.obs_clip).astype(np.float32)
            with torch.no_grad():
                mu, std = actor(torch.from_numpy(obs)[None])
            pre = mu[0].numpy()
            if stochastic:
                pre = pre + std[0].numpy() * rng.standard_normal(pre.shape)
            raw, reward, terminated, truncated, _ = env.step(np.tanh(pre))
            data = env.unwrapped.data
            pitches.append(float(data.qpos[2]))
            velocities.append(float(data.qvel[0]))
            total += float(reward)
            done = terminated or truncated
        pitches = np.asarray(pitches)
        results.append({"return": total, "mean_velocity": float(np.mean(velocities)),
                        "mean_pitch": float(pitches.mean()),
                        "upside_down_frac": float((np.abs(pitches) > np.pi / 2).mean())})
    env.close()
    return {key: float(np.mean([r[key] for r in results])) for key in results[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--snapshots", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_000)
    cli = parser.parse_args()
    trainer = load_trainer(cli.trainer)
    for path in cli.snapshots:
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        args = SimpleNamespace(**snapshot["args"])
        env = gym.make(args.env_id)
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        env.close()
        actor = trainer.Actor(observation_dim, action_dim, args)
        actor.net.load_state_dict(snapshot["actor_net"])
        actor.eval()
        report = {"snapshot": path, "global_step": snapshot["global_step"],
                  "mean_action": rollout(actor, snapshot, cli.episodes, False, cli.seed),
                  "sampled_action": rollout(actor, snapshot, cli.episodes, True, cli.seed)}
        print(json.dumps(report))


if __name__ == "__main__":
    main()

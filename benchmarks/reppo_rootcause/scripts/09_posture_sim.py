"""Short CPU posture rollouts of existing REPPO snapshots (a few thousand simulator steps each; no training).

Per episode: return, first step at which the torso is inverted (|root pitch| > pi/2), inverted fraction, number of
full pitch revolutions, action saturation, and mean pre-tanh std. Normalisation follows scripts/reppo_posture.py
(environment 0's saved statistics). Usage:
    09_posture_sim.py <trainer.py> <snapshot.pt> [stochastic_episodes] [deterministic_episodes] [noise_scale]
noise_scale multiplies the policy's pre-tanh std in the stochastic episodes (1.0 = the training behaviour)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
torch.set_num_threads(2)


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("reppo_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode(env, actor, mean, inv_std, clip, stochastic, seed, noise_scale=1.0):
    rng = np.random.default_rng(seed)
    raw, _ = env.reset(seed=seed)
    record = {"return": 0.0, "pitch": [], "saturated": [], "mode_saturated": [], "std": [], "velocity": []}
    done = False
    while not done:
        obs = np.clip((raw - mean) * inv_std, -clip, clip).astype(np.float32)
        with torch.no_grad():
            mu, std = actor(torch.from_numpy(obs)[None])
        mu, std = mu[0].numpy(), std[0].numpy()
        pre = mu + noise_scale * std * rng.standard_normal(mu.shape) if stochastic else mu
        action = np.tanh(pre)
        raw, reward, terminated, truncated, _ = env.step(action)
        data = env.unwrapped.data
        record["return"] += float(reward)
        record["pitch"].append(float(data.qpos[2]))
        record["velocity"].append(float(data.qvel[0]))
        record["saturated"].append(float((np.abs(action) > 0.99).mean()))
        record["mode_saturated"].append(float((np.abs(np.tanh(mu)) > 0.99).mean()))
        record["std"].append(float(std.mean()))
        done = terminated or truncated
    pitch = np.asarray(record["pitch"])
    inverted = np.abs(np.mod(pitch + np.pi, 2 * np.pi) - np.pi) > np.pi / 2
    first = int(np.argmax(inverted)) if inverted.any() else -1
    velocity = np.asarray(record["velocity"])
    return {"return": round(record["return"]), "first_inverted_step": first,
            "inverted_frac": round(float(inverted.mean()), 3),
            "revolutions": round(float(pitch[-1] / (2 * np.pi)), 2),
            "pitch_min_max": [round(float(pitch.min()), 2), round(float(pitch.max()), 2)],
            "velocity_upright": round(float(velocity[~inverted].mean()), 2) if (~inverted).any() else None,
            "velocity_inverted": round(float(velocity[inverted].mean()), 2) if inverted.any() else None,
            "action_saturation": round(float(np.mean(record["saturated"])), 3),
            "mode_saturation": round(float(np.mean(record["mode_saturated"])), 3),
            "pre_tanh_std": round(float(np.mean(record["std"])), 3)}


def main():
    trainer = load_trainer(sys.argv[1])
    snapshot = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    stochastic_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    deterministic_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    noise_scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    args = SimpleNamespace(**snapshot["args"])
    env = gym.make(args.env_id)
    actor = trainer.Actor(env.observation_space.shape[0], env.action_space.shape[0], args)
    actor.net.load_state_dict(snapshot["actor_net"])
    actor.eval()
    mean = np.asarray(snapshot["obs_means"])[0]
    inv_std = 1.0 / np.sqrt(np.asarray(snapshot["obs_variances"])[0] + snapshot["obs_epsilon"])
    print(json.dumps({"snapshot": str(Path(sys.argv[2]).relative_to(REPO)) if Path(sys.argv[2]).is_absolute() else sys.argv[2],
                      "global_step": snapshot["global_step"], "noise_scale": noise_scale}))
    for mode, count in (("deterministic", deterministic_episodes), ("stochastic", stochastic_episodes)):
        for index in range(count):
            result = episode(env, actor, mean, inv_std, args.obs_clip, mode == "stochastic", 30_000 + index, noise_scale)
            print(json.dumps({"mode": mode, **result}))
    env.close()


if __name__ == "__main__":
    main()

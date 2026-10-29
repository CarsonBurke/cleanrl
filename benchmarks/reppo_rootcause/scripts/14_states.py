"""On-distribution states for the v60 checkpoints (none are stored with them).

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/14_states.py

Five stochastic episodes per checkpoint (15 x 1000 simulator steps in total, single thread), each under a different
lane's observation statistics, as the trainer's lanes run. Saves raw observations, the action noise, simulator pitch and
floor contacts to data/14_states_<step>.npz.
"""
import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("common14", Path(__file__).resolve().parent / "14_common.py")
c14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c14)
torch.set_num_threads(1)

for step in c14.STEPS:
    ck = c14.load_checkpoint(step)
    env = gym.make(ck.args.env_id)
    model, data = env.unwrapped.model, env.unwrapped.data
    geoms = [model.geom(i).name for i in range(model.ngeom)]
    floor = geoms.index("floor")
    rng = np.random.default_rng(14)
    out = {k: [] for k in ("raw", "noise", "action", "lane", "t", "reward", "pitch", "contact")}
    returns = []
    for episode, lane in enumerate(c14.LANES):
        raw, _ = env.reset(seed=14_000 + episode)
        done, total, t = False, 0.0, 0
        while not done:
            touching = np.zeros(len(geoms), bool)
            for contact in data.contact[:data.ncon]:
                if floor in (contact.geom1, contact.geom2):
                    touching[contact.geom1 + contact.geom2 - floor] = True
            obs = c14.normalise(raw[None], np.array([lane]), ck)
            with torch.no_grad():
                mean, std = ck.actor(torch.from_numpy(obs))
            noise = rng.standard_normal(6).astype(np.float32)
            action = np.tanh(mean[0].numpy() + std[0].numpy() * noise)
            for key, value in (("raw", raw.copy()), ("noise", noise), ("action", action), ("lane", lane), ("t", t),
                               ("pitch", float(data.qpos[2])), ("contact", touching)):
                out[key].append(value)
            raw, reward, terminated, truncated, _ = env.step(action)
            out["reward"].append(float(reward))
            total, t, done = total + float(reward), t + 1, terminated or truncated
        returns.append(round(total))
    env.close()
    arrays = {k: np.asarray(v) for k, v in out.items()}
    np.savez(c14.OUT / f"14_states_{step}.npz", geoms=np.array(geoms), **arrays)
    print(f"{step}: {len(arrays['raw'])} states, episode returns {returns}, inverted rows {np.mean(np.cos(arrays['pitch']) < 0):.3f}, "
          f"raw obs[1] == qpos pitch: {np.abs(arrays['raw'][:, 1] - arrays['pitch']).max():.1e}")

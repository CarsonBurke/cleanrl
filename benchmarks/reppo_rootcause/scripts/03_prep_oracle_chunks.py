"""Flatten the saved REPPO oracle datasets (runs/reppo_oracle, runs/reppo_latentroll_oracle) into numpy.

CPU only. For each dataset: the full on-policy 16 x 1024 chunk the oracle collected under the frozen snapshot
policy (normalized obs, actions, trainer-exact lambda-return targets), the snapshot actor's (mean, std) on it,
the snapshot critic's Q on it, and the 512 oracle states' branching returns. No simulation is run.

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/03_prep_oracle_chunks.py
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "benchmarks/reppo_rootcause/data"
torch.set_num_threads(2)

DATASETS = {
    "v23_4M": ("runs/reppo_oracle/HalfCheetah-v4__reppo_probe_v23_50M__1__1790369616_4013696.pt",
               "cleanrl/reppo/ppo_continuous_action_reppo_probe_v23.py"),
    "v23_12M": ("runs/reppo_oracle/HalfCheetah-v4__reppo_probe_v23_50M__1__1790369616_12009088.pt",
                "cleanrl/reppo/ppo_continuous_action_reppo_probe_v23.py"),
    "v23_30M": ("runs/reppo_oracle/HalfCheetah-v4__reppo_probe_v23_50M__1__1790369616_30015104.pt",
                "cleanrl/reppo/ppo_continuous_action_reppo_probe_v23.py"),
    "v29h8_8M": ("runs/reppo_latentroll_oracle/HalfCheetah-v4__reppo_diagnose_v29_h8_reward_16M__1__1790481916_8011392.pt",
                 None),
    "v29h8_12M": ("runs/reppo_latentroll_oracle/HalfCheetah-v4__reppo_diagnose_v29_h8_reward_16M__1__1790481916_12009088.pt",
                  None),
}


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"trainer_{Path(path).stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (path, trainer_path) in DATASETS.items():
        data = torch.load(REPO / path, map_location="cpu", weights_only=False)
        meta = data["meta"]
        snap_path = Path(meta["snapshot"])
        snapshot = torch.load(snap_path if snap_path.is_absolute() else REPO / snap_path, map_location="cpu",
                              weights_only=False)
        if trainer_path is None:
            trainer_path = snapshot.get("trainer_file") or "cleanrl/reppo/ppo_continuous_action_reppo_latentroll_v27.py"
        trainer_path = Path(trainer_path)
        trainer = load_module(trainer_path if trainer_path.is_absolute() else REPO / trainer_path)
        args = SimpleNamespace(**snapshot["args"])
        actor = trainer.Actor(17, 6, args)
        actor.net.load_state_dict(snapshot["actor_net"])
        critic = trainer.Critic(17, 6, args, torch.device("cpu"))
        critic.load_state_dict(snapshot["critic"])
        actor.eval(), critic.eval()

        chunk, states = data["chunk"], data["states"]
        rows = torch.as_tensor(chunk["row"])
        mean_o = data["norm"]["obs_means"].double()
        scale = torch.sqrt(data["norm"]["obs_variances"].double() + data["norm"]["obs_epsilon"])
        obs = ((chunk["raw_obs"].double() - mean_o[rows]) / scale[rows]).clamp(-meta["obs_clip"], meta["obs_clip"]).float()
        mean, std = actor(obs)
        action = chunk["actions"].float()
        critic_q = torch.cat([critic.value(obs[i:i + 4096], action[i:i + 4096]) for i in range(0, len(obs), 4096)])
        index = states["index"]
        # Reconstruction checks: the oracle states' obs and critic values must match the saved ones.
        obs_err = float((obs[index] - states["obs"]).abs().max())
        a0_slot = meta["action_layout"]["a0_probes"][0]
        q_err = float((critic_q[index].double() - states["critic_q"][:, a0_slot]).abs().max())
        a_err = float((action[index] - states["actions"][:, a0_slot]).abs().max())
        s_mean, s_std = actor(states["obs"])
        star_err = float((torch.tanh(s_mean) - states["actions"][:, 0]).abs().max())
        print(f"{name}: step {meta['global_step']} lam {meta['lambda']} T {meta['temperature']:.4f} "
              f"| recon err obs {obs_err:.1e} a0 {a_err:.1e} critic {q_err:.1e} star {star_err:.1e} trainer {trainer_path.name}",
              flush=True)
        np.savez_compressed(
            OUT / f"03_chunk_{name}.npz",
            obs=obs.numpy(), action=action.numpy(), mean=mean.numpy(), std=std.numpy(),
            target=chunk["targets"].numpy(), reward=chunk["rewards"].numpy(), next_q=chunk["next_q"].numpy(),
            next_logprob=chunk["next_logprob"].numpy(), logprob=chunk["logprob"].numpy(),
            truncations=chunk["truncations"].numpy(), ends=chunk["ends"].numpy(), row=chunk["row"].numpy() if hasattr(chunk["row"], "numpy") else np.asarray(chunk["row"]),
            age=chunk["age"].numpy() if hasattr(chunk["age"], "numpy") else np.asarray(chunk["age"]),
            state=chunk["state"].numpy() if hasattr(chunk["state"], "numpy") else np.asarray(chunk["state"]),
            critic_q=critic_q.numpy(),
            s_index=index.numpy(), s_obs=states["obs"].numpy(), s_actions=states["actions"].numpy(),
            s_mean=s_mean.numpy(), s_std=s_std.numpy(), s_row=states["row"].numpy(), s_age=states["age"].numpy(),
            s_critic_q=states["critic_q"].numpy(), s_critic_baseline=states["critic_baseline"].numpy(),
            s_critic_grad_star=states["critic_grad_star"].numpy(), s_critic_grad_a0=states["critic_grad_a0"].numpy(),
            span_star=states["span_star"].numpy(), span_a0=states["span_a0"].numpy(),
            upright=states["subset_upright"].numpy(), cruising=states["subset_cruising"].numpy(),
            **{f"ret_{h}": value.numpy().astype(np.float32) for h, value in data["returns"].items()},
            gamma=meta["gamma"], lam=meta["lambda"], temperature=meta["temperature"], fd_steps=np.asarray(meta["fd_steps"]),
            global_step=meta["global_step"])


if __name__ == "__main__":
    main()

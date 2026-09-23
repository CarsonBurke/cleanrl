"""Frozen-policy, same-data raw-vs-JEPA probes; run only through mlq.

Fits diagnostic probes, never fine-tunes the policy. Held-out environments and
finite-horizon factual returns avoid comparing critics on their own TD targets.
Per-coordinate input scaling is fitted on training environments only. Better probe
fit is evidence about readout difficulty, not a proof of information sufficiency.
"""
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.ppo_continuous_action_jepa_shared_ffn_ablation_v6 import Agent, Args as TrainerArgs, HostPolicy, TaskFFN
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.vector_norm import VectorObsNorm


@dataclass
class Args:
    checkpoint: Path
    exp_name: str = "jepa_representation_audit"
    collect_steps: int = 8192
    epochs: int = 100
    batch_size: int = 1024
    seed: int = 1
    env_threads: int = 2


class FrozenObsNorm(VectorObsNorm):
    """Reuse boundary/warmup semantics with checkpoint moments held fixed."""
    def normalize(self, obs, rows=None, out_dtype=np.float32):
        means = self.means if rows is None else self.means[rows]
        variances = self.variances if rows is None else self.variances[rows]
        result = (np.asarray(obs) - means) / np.sqrt(variances + self.epsilon)
        np.clip(result, -self.clip, self.clip, out=result)
        return result.astype(out_dtype, copy=False)


def factual_targets(rewards, boundaries, horizons, gamma):
    """No bootstrap, no reset crossing; all horizons use identical start rows."""
    longest = max(horizons)
    length = rewards.shape[0] - longest + 1
    if length <= 0:
        raise ValueError("collection must exceed the longest prediction horizon")
    source = rewards.T[:, None, :]
    targets = []
    for horizon in horizons:
        kernel = gamma ** torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
        targets.append(F.conv1d(source, kernel[None, None, :])[:, 0, :length].T)
    # Conservatively exclude windows ending at a boundary as well as crossing it.
    count = F.conv1d(boundaries.T[:, None, :].float(), rewards.new_ones(1, 1, longest))
    valid = count[:, 0, :].T == 0
    return torch.stack(targets, -1), valid


def r2(predicted, target):
    error = (predicted - target).square().mean(0)
    variance = target.var(0, correction=0)
    return torch.where(variance > 0, 1 - error / variance, torch.full_like(variance, float("nan")))


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit this audit through mlq")
    if min(args.collect_steps, args.epochs, args.batch_size) <= 0:
        raise ValueError("collection, epochs and batch size must be positive")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    trainer_args = TrainerArgs(**checkpoint["args"])
    n = trainer_args.num_envs
    if n < 4 or trainer_args.jepa_mode == "none":
        raise ValueError("audit requires an encoder and at least four checkpoint environments")
    run_name = f"{trainer_args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    output = Path("runs") / run_name
    writer = SummaryWriter(str(output))
    writer.add_text("hyperparameters", json.dumps({**vars(args), "checkpoint": str(args.checkpoint)}, default=str))
    envs = make_mujoco_vector_env(trainer_args.env_id, n, num_threads=args.env_threads)
    transfer = None
    try:
        agent = Agent(envs, trainer_args).to(device)
        agent.load_state_dict(checkpoint["model"])
        agent.eval().requires_grad_(False)
        obs_shape = envs.single_observation_space.shape
        obs_dim = int(np.prod(obs_shape))
        if obs_dim > 64:
            raise ValueError("equal-width raw observation control supports at most64 coordinates")
        norm_state = checkpoint["obs_norm"]
        norm = FrozenObsNorm(n, obs_shape, epsilon=norm_state["epsilon"], clip=norm_state["clip"])
        for key in ("means", "variances", "counts"):
            getattr(norm, key)[:] = norm_state[key].numpy()
        host = HostPolicy(agent, n)
        sampler = make_beta_sampler(n, agent.action_dim, agent.action_low.cpu().numpy(), agent.action_high.cpu().numpy())
        rng = np.random.default_rng(args.seed)

        def act(observations):
            _, action = sampler(host(observations), rng)
            if not np.isfinite(action).all():
                raise FloatingPointError("nonfinite frozen-policy actions")
            return action.reshape((n,) + agent.action_shape)

        horizon = episode_horizon(trainer_args.env_id)
        warm = run_phase_warmup(envs, obs_norm=norm, rew_norm=None, act_fn=act,
                                horizon=horizon, phase_offsets=compute_phase_offsets(n, horizon, args.seed), seed=args.seed)
        obs = warm.next_obs
        transfer = RolloutTransfer(args.collect_steps, n, obs_shape, device, fields={"observations": obs_shape})
        for step in range(args.collect_steps):
            action = act(obs)
            raw_next, raw_reward, terms, truncs, _ = envs.step(action)
            transfer.push(step, raw_reward, terms, truncs, observations=obs)
            obs = norm.normalize(raw_next)
        batch = transfer.upload()
        horizons = (1, 32, 128, 512)
        targets, valid = factual_targets(batch.rewards, torch.maximum(batch.terminations, batch.truncations), horizons, trainer_args.gamma)
        length = targets.shape[0]
        observations = batch.fields["observations"][:length].flatten(0, 1)
        targets = targets.flatten(0, 1)
        valid = valid.flatten()
        env_indices = torch.arange(length * n, device=device) % n
        train = torch.nonzero(valid & (env_indices < n // 2), as_tuple=True)[0]
        held = torch.nonzero(valid & (env_indices >= n // 2), as_tuple=True)[0]
        if train.numel() < args.batch_size or held.numel() < args.batch_size:
            raise RuntimeError("insufficient complete within-episode train/held-out windows")
        encoder = torch.compile(agent.encoder, fullgraph=True, options={"triton.cudagraphs": False})
        with torch.no_grad():
            latent = encoder(observations)
            raw = F.pad(observations, (0, 64 - obs_dim))
            features = []
            for values in (raw, latent):
                mean = values[train].mean(0)
                scale = values[train].std(0, correction=0)
                scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
                features.append((values - mean) / scale)
            target_mean = targets[train].mean(0)
            target_scale = targets[train].std(0, correction=0)
            if bool((target_scale <= 0).any()):
                raise RuntimeError("constant return target prevents an interpretable probe")
            returns = (targets - target_mean) / target_scale
            state_mean = observations[train].mean(0)
            state_scale = observations[train].std(0, correction=0)
            state_scale = torch.where(state_scale > 1e-6, state_scale, torch.ones_like(state_scale))
            states = (observations - state_mean) / state_scale
        models = torch.nn.ModuleList([TaskFFN(64, len(horizons), 1.0), TaskFFN(64, len(horizons), 1.0),
                                     TaskFFN(64, obs_dim, 1.0), TaskFFN(64, obs_dim, 1.0)]).to(device)
        models[1].load_state_dict(copy.deepcopy(models[0].state_dict()))
        models[3].load_state_dict(copy.deepcopy(models[2].state_dict()))
        optimizer = torch.optim.Adam(models.parameters(), lr=3e-4, eps=1e-5)
        groups = tuple(tuple(model.parameters()) for model in models)

        def loss_fn(raw_x, latent_x, return_y, state_y):
            return torch.stack((F.mse_loss(models[0](raw_x), return_y), F.mse_loss(models[1](latent_x), return_y),
                                F.mse_loss(models[2](raw_x), state_y), F.mse_loss(models[3](latent_x), state_y)))

        def evaluation_fn(raw_x, latent_x, return_y, state_y):
            return (r2(models[0](raw_x), return_y), r2(models[1](latent_x), return_y),
                    r2(models[2](raw_x), state_y), r2(models[3](latent_x), state_y))

        loss_model = torch.compile(loss_fn, fullgraph=True, options={"triton.cudagraphs": False})
        evaluate = torch.compile(evaluation_fn, fullgraph=True, options={"triton.cudagraphs": False})
        generator = torch.Generator(device=device).manual_seed(args.seed)
        history = []
        for epoch in range(args.epochs + 1):
            if epoch:
                order = train[torch.randperm(train.numel(), device=device, generator=generator)]
                # Fixed shapes; remainder rotates with each epoch's independent shuffle.
                for indices in order[:(order.numel() // args.batch_size) * args.batch_size].view(-1, args.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss_model(features[0][indices], features[1][indices], returns[indices], states[indices]).sum().backward()
                    for parameters in groups:
                        torch.nn.utils.clip_grad_norm_(parameters, 0.5)
                    optimizer.step()
            if epoch % 10 == 0 or epoch == args.epochs:
                with torch.no_grad():
                    scores = evaluate(features[0][held], features[1][held], returns[held], states[held])
                    packed = torch.cat(scores).cpu().tolist()
                record = {"epoch": epoch, "raw_return_r2": packed[:4], "latent_return_r2": packed[4:8],
                          "raw_state_r2": packed[8:8 + obs_dim], "latent_state_r2": packed[8 + obs_dim:]}
                history.append(record)
                for name in ("raw", "latent"):
                    for horizon, value in zip(horizons, record[f"{name}_return_r2"]):
                        writer.add_scalar(f"probe/{name}_return_h{horizon}_r2", value, epoch)
                    writer.add_scalar(f"probe/{name}_state_mean_r2", np.mean(record[f"{name}_state_r2"]), epoch)
                print(json.dumps(record), flush=True)
        result = {"checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["global_step"],
                  "train_environments": list(range(n // 2)), "held_out_environments": list(range(n // 2, n)),
                  "train_samples": train.numel(), "held_out_samples": held.numel(), "horizons": horizons,
                  "gamma": trainer_args.gamma, "raw_reward_targets": True, "frozen_normalization": True,
                  "finite_returns_without_bootstrap": True, "history": history}
        (output / "probe_results.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if transfer is not None:
            transfer.close()
        envs.close()
        writer.close()


if __name__ == "__main__":
    main()

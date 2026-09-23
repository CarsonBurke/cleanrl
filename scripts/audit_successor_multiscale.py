"""Fresh-rollout diagnostics for frozen v15 critics; execute only through mlq.

No policy/critic fitting. Compare each discount against long factual returns,
report the small endpoint bootstrap separately, and measure contrast redundancy.
Different policies induce different data distributions: this is NOT a matched-data
cross-policy critic ranking or an estimate of true conditional advantage error.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl import ppo_continuous_action_successor_multiscale_v15 as model
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.vector_norm import VectorObsNorm


@dataclass
class Args:
    checkpoint: Path
    exp_name: str = "successor_multiscale_v15_audit"
    collect_steps: int = 8192
    return_horizon: int = 768
    seed: int = 1
    env_threads: int = 2


class FrozenObsNorm(VectorObsNorm):
    """Checkpoint coordinate frames, with inherited factual-boundary handling."""
    def normalize(self, obs, rows=None, out_dtype=np.float32):
        means = self.means if rows is None else self.means[rows]
        variances = self.variances if rows is None else self.variances[rows]
        result = (np.asarray(obs) - means) / np.sqrt(variances + self.epsilon)
        if self.clip is not None:
            np.clip(result, -self.clip, self.clip, out=result)
        return result.astype(out_dtype, copy=False)


def finite_returns(factors, boundaries, horizon, discount):
    """Factual factor sums; windows crossing OR ending at reset are excluded."""
    kernel = discount ** torch.arange(horizon, device=factors.device, dtype=factors.dtype)
    source = factors.permute(1, 2, 0)
    target = F.conv1d(source, kernel.expand(2, 1, -1), groups=2).permute(2, 0, 1)
    count = F.conv1d(boundaries.T[:, None].float(), kernel.new_ones(1, 1, horizon))
    return target, count[:, 0].T == 0


def regression_metrics(prediction, target):
    mse = (prediction - target).square().mean()
    variance = target.var(correction=0)
    return {"mse": mse, "bias": (prediction - target).mean(),
            "target_variance": variance, "r2": 1 - mse / variance.clamp_min(1e-12),
            "residual_variance": (prediction - target).var(correction=0)}


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; submit through mlq")
    if args.collect_steps <= args.return_horizon or args.return_horizon <= 0:
        raise ValueError("collection must exceed a positive return horizon")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = model.Args(**checkpoint["args"])
    n = config.num_envs
    if n < 4 or args.return_horizon >= episode_horizon(config.env_id):
        raise ValueError("need at least four environments and within-episode return windows")
    output = Path("runs") / f"{config.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(str(output))
    writer.add_text("hyperparameters", json.dumps(vars(args), default=str))
    envs = make_mujoco_vector_env(config.env_id, n, num_threads=args.env_threads)
    transfer = None
    try:
        agent = model.Agent(envs, config).cuda().eval().requires_grad_(False)
        agent.load_state_dict(checkpoint["model"])
        shape = envs.single_observation_space.shape
        state = checkpoint["obs_norm"]
        norm = FrozenObsNorm(n, shape, epsilon=state["epsilon"], clip=state["clip"])
        for key in ("means", "variances", "counts"):
            getattr(norm, key)[:] = state[key].numpy()
        reward_state = checkpoint["reward_norm"]
        reward_std = np.sqrt(reward_state["variances"].numpy() + reward_state["epsilon"])
        host = model.HostPolicy(agent, n)
        sampler = make_beta_sampler(n, agent.action_dim, agent.action_low.cpu().numpy(), agent.action_high.cpu().numpy())
        rng = np.random.default_rng(args.seed)

        def act(observations):
            action = sampler(host(observations), rng)[1]
            if not np.isfinite(action).all():
                raise FloatingPointError("nonfinite frozen policy action")
            return action.reshape((n,) + agent.action_shape)

        episode_steps = episode_horizon(config.env_id)
        warm = run_phase_warmup(envs, obs_norm=norm, rew_norm=None, act_fn=act,
                                horizon=episode_steps, phase_offsets=compute_phase_offsets(n, episode_steps, args.seed),
                                seed=args.seed)
        obs = warm.next_obs
        suppress = warm.suppress_mask.copy()
        transfer = RolloutTransfer(args.collect_steps, n, shape, "cuda", fields={
            "observations": shape, "next_observations": shape, "reward_factors": (2,),
        })
        episodic_returns = []
        for step in range(args.collect_steps):
            raw_next, raw_reward, terms, truncs, infos = envs.step(act(obs))
            raw_factors = model.observed_reward_factors(raw_reward, terms, truncs, infos)
            divisor = reward_std
            if reward_state["clip"] is not None:
                divisor = np.maximum(divisor, np.abs(raw_reward) / reward_state["clip"])
            factors = raw_factors / divisor[:, None]
            next_obs, factual = norm.normalize_step(raw_next, terms, truncs, infos)
            transfer.push(step, factors.sum(-1), terms, truncs, observations=obs,
                          next_observations=factual, reward_factors=factors)
            obs = next_obs
            for index, info in enumerate(infos.get("final_info", ())):
                if info and "episode" in info:
                    if suppress[index]:
                        suppress[index] = False
                        continue
                    episodic_returns.append(float(info["episode"]["r"]))
        batch = transfer.upload()
        predict = torch.compile(agent.get_successor, fullgraph=True, options={"triton.cudagraphs": False})
        finite = torch.compile(finite_returns, fullgraph=True, options={"triton.cudagraphs": False})
        gae = get_gae_fn(compiled=True, explicit_next_values=True)
        with torch.no_grad():
            predictions = predict(batch.fields["observations"].flatten(0, 1)).reshape(args.collect_steps, n, -1)
            following = predict(batch.fields["next_observations"].flatten(0, 1)).reshape_as(predictions)
            length = args.collect_steps - args.return_horizon + 1
            env_indices = torch.arange(n, device="cuda").expand(length, n)
            metrics = {}
            for index, discount in enumerate(agent.discounts):
                targets, valid = finite(batch.fields["reward_factors"],
                                         torch.maximum(batch.terminations, batch.truncations),
                                         args.return_horizon, discount)
                prefix = f"discount_{discount:g}"
                factor_prediction = predictions[..., 2 * index:2 * index + 2]
                factor_next = following[..., 2 * index:2 * index + 2]
                values = factor_prediction.sum(-1)
                next_values = factor_next.sum(-1)
                observed = targets.sum(-1)
                endpoint = discount ** args.return_horizon * next_values[args.return_horizon - 1:]
                start_values = values[:length]
                for label, estimate, target in (
                    ("finite_no_bootstrap", start_values[valid], observed[valid]),
                    ("endpoint_corrected", (start_values - endpoint)[valid], observed[valid]),
                ):
                    for key, value in regression_metrics(estimate, target).items():
                        metrics[f"{prefix}/{label}/{key}"] = value
                metrics[f"{prefix}/endpoint_rms"] = endpoint[valid].square().mean().sqrt()
                metrics[f"{prefix}/valid_windows"] = valid.sum()
                metrics[f"{prefix}/finite_return_rms"] = observed[valid].square().mean().sqrt()
                # Describe redundancy using train-env-fitted linear regression;
                # held-out environments, no critic/policy parameters are fitted.
                contrast = (targets[..., 0] - targets[..., 1]) / np.sqrt(2)
                train, held = valid & (env_indices < n // 2), valid & (env_indices >= n // 2)
                x = torch.stack((torch.ones_like(observed), observed), -1).double()
                coefficients = torch.linalg.lstsq(x[train], contrast[train].double()).solution
                held_residual = contrast[held].double() - x[held] @ coefficients
                metrics[f"{prefix}/contrast_linear_conditional_variance_fraction"] = (
                    held_residual.var(correction=0) / contrast[held].double().var(correction=0).clamp_min(1e-12))
                metrics[f"{prefix}/contrast_linear_heldout_r2"] = (
                    1 - held_residual.square().mean() / contrast[held].double().var(correction=0).clamp_min(1e-12))
                metrics[f"{prefix}/contrast_centered_to_uncentered_energy"] = (
                    contrast[valid].var(correction=0) / contrast[valid].square().mean().clamp_min(1e-12))
                if agent.successor_dim != 1:
                    predicted_contrast = (factor_prediction[..., 0] - factor_prediction[..., 1]) / np.sqrt(2)
                    end_contrast = (factor_next[..., 0] - factor_next[..., 1]) / np.sqrt(2)
                    corrected = predicted_contrast[:length] - discount ** args.return_horizon * end_contrast[args.return_horizon - 1:]
                    for key, value in regression_metrics(corrected[valid], contrast[valid]).items():
                        metrics[f"{prefix}/contrast_endpoint_corrected/{key}"] = value
                if index == 0:
                    advantage, _ = gae(batch.rewards, values, batch.terminations, batch.truncations,
                                       next_values, discount, config.gae_lambda)
                    long_advantage = (observed + endpoint - start_values)[valid]
                    gae_advantage = advantage[:length][valid]
                    metrics["advantage/gae_vs_long_sample_mse"] = (gae_advantage - long_advantage).square().mean()
                    metrics["advantage/long_sample_variance"] = long_advantage.var(correction=0)
                    metrics["advantage/gae_variance"] = gae_advantage.var(correction=0)
                    centered_gae = gae_advantage - gae_advantage.mean()
                    centered_long = long_advantage - long_advantage.mean()
                    metrics["advantage/gae_long_sample_correlation"] = (centered_gae * centered_long).mean() / (
                        centered_gae.square().mean() * centered_long.square().mean()).sqrt().clamp_min(1e-12)
            scalar_metrics = {name: float(value.cpu()) for name, value in metrics.items()}
        if not all(np.isfinite(value) for value in scalar_metrics.values()):
            raise FloatingPointError("nonfinite critic diagnostics or insufficient valid return windows")
        result = {
            "checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["global_step"],
            "critic_mode": config.critic_mode, "discounts": agent.discounts,
            "transitions": args.collect_steps * n, "return_horizon": args.return_horizon,
            "fresh_rollout_episode_return_mean": float(np.mean(episodic_returns)),
            "metrics": scalar_metrics,
            "limitations": [
                "Single seed; overlapping return windows are not independent samples.",
                "Each critic is evaluated under its own frozen stochastic policy, not matched cross-policy data.",
                "Frozen per-environment observation/reward scales differ from training's slowly moving moments.",
                "Endpoint-corrected error uses the critic only at a discount-suppressed endpoint; unbootstrapped error is also reported.",
                "Long sampled returns include stochastic policy noise; this is not true conditional advantage error.",
                "Linear conditional variance is a heldout linear redundancy diagnostic, not full conditional variance.",
            ],
        }
        for name, value in scalar_metrics.items():
            writer.add_scalar(f"audit/{name}", value, args.collect_steps * n)
        writer.add_scalar("charts/episodic_return", result["fresh_rollout_episode_return_mean"], args.collect_steps * n)
        (output / "critic_audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if transfer is not None:
            transfer.close()
        envs.close()
        writer.close()


if __name__ == "__main__":
    main()

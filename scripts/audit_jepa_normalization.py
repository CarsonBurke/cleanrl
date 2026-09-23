"""Measure normalization-induced coordinate variation on one frozen policy via mlq.

No policy or world-model fitting. Mature checkpoint moments and a cold-statistics
counterfactual see the SAME physical transitions. Cold statistics are not a claim
about the original policy's early training distribution.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.ppo_continuous_action_jepa_geometry_drift_v7 import Agent, Args as TrainerArgs, HostPolicy
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


@dataclass
class Args:
    checkpoint: Path
    collect_steps: int = 2048
    seed: int = 1
    env_threads: int = 2
    exp_name: str = "jepa_normalization_audit"


class FrozenPolicyNorm(VectorObsNorm):
    def normalize(self, obs, rows=None, out_dtype=np.float32):
        if not hasattr(self, "raw"):
            self.raw = np.zeros_like(self.means)
        if rows is None:
            self.raw[:] = obs
            means, variances = self.means, self.variances
        else:
            self.raw[rows] = obs
            means, variances = self.means[rows], self.variances[rows]
        return np.clip((obs - means) / np.sqrt(variances + self.epsilon),
                       -self.clip, self.clip).astype(out_dtype)


def load_moments(norm, state):
    for name in ("means", "variances", "counts"):
        getattr(norm, name)[:] = state[name].numpy()
    if hasattr(norm, "returns"):
        norm.returns[:] = state["returns"].numpy()


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; submit through mlq")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = TrainerArgs(**checkpoint["args"])
    n = config.num_envs
    output = Path("runs") / f"{config.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(str(output))
    writer.add_text("hyperparameters", json.dumps(vars(args), default=str))
    envs = make_mujoco_vector_env(config.env_id, n, num_threads=args.env_threads)
    try:
        agent = Agent(envs, config).cuda().eval().requires_grad_(False)
        agent.load_state_dict(checkpoint["model"])
        shape = envs.single_observation_space.shape
        policy_norm = FrozenPolicyNorm(n, shape)
        load_moments(policy_norm, checkpoint["obs_norm"])
        host = HostPolicy(agent, n)
        sampler = make_beta_sampler(n, agent.action_dim, agent.action_low.cpu().numpy(), agent.action_high.cpu().numpy())
        rng = np.random.default_rng(args.seed)

        def act(observations):
            return sampler(host(observations), rng)[1].reshape((n,) + agent.action_shape)

        horizon = episode_horizon(config.env_id)
        warm = run_phase_warmup(envs, obs_norm=policy_norm, rew_norm=None, act_fn=act,
                                horizon=horizon, phase_offsets=compute_phase_offsets(n, horizon, args.seed), seed=args.seed)
        policy_obs = warm.next_obs
        raw_current = policy_norm.raw.copy()
        modes = {}
        for name in ("mature", "cold_counterfactual"):
            obs_norm = VectorObsNorm(n, shape)
            reward_norm = VectorRewardNorm(n, config.gamma)
            if name == "mature":
                load_moments(obs_norm, checkpoint["obs_norm"])
                load_moments(reward_norm, checkpoint["reward_norm"])
            modes[name] = {"obs_norm": obs_norm, "reward_norm": reward_norm,
                           "current": obs_norm.normalize(raw_current), "current_rows": [],
                           "target_rows": [], "same_chart_rows": [], "reward_scales": [],
                           "clipped_count": 0, "element_count": 0,
                           "reward_clipped_count": 0, "reward_count": 0,
                           "initial_mean": obs_norm.means.copy(), "initial_scale": np.sqrt(obs_norm.variances + obs_norm.epsilon)}
        raw_samples = []
        for step in range(args.collect_steps):
            actions = act(policy_obs)
            raw_next, raw_reward, terms, truncs, infos = envs.step(actions)
            factual_next = np.array(raw_next, copy=True)
            for index in np.flatnonzero(terms | truncs):
                factual_next[index] = infos["final_observation"][index]
            raw_samples.append(raw_current.copy())
            for mode in modes.values():
                norm = mode["obs_norm"]
                same_chart = np.clip((factual_next - norm.means) / np.sqrt(norm.variances + norm.epsilon), -norm.clip, norm.clip)
                next_obs, target = norm.normalize_step(raw_next, terms, truncs, infos)
                mode["current_rows"].append(mode["current"])
                mode["target_rows"].append(target)
                mode["same_chart_rows"].append(same_chart.astype(np.float32))
                mode["clipped_count"] += int(np.count_nonzero(np.abs(target) >= norm.clip))
                mode["element_count"] += target.size
                normalized_reward = mode["reward_norm"].normalize(raw_reward, terms)
                mode["reward_scales"].append(np.sqrt(mode["reward_norm"].variances + mode["reward_norm"].epsilon).copy())
                mode["reward_clipped_count"] += int(np.count_nonzero(np.abs(normalized_reward) >= mode["reward_norm"].clip))
                mode["reward_count"] += n
                mode["current"] = next_obs
            policy_obs, _ = policy_norm.normalize_step(raw_next, terms, truncs, infos)
            raw_current = policy_norm.raw.copy()
        project = torch.compile(lambda x: agent.ssl.projector(agent.encoder(x)), fullgraph=True,
                                options={"triton.cudagraphs": False})
        result = {"checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["global_step"],
                  "transitions": args.collect_steps * n, "policy_frozen": True,
                  "policy_normalization_frozen": True, "modes": {},
                  "limitations": ["No predictor fitting; coordinate effects are not a measured return penalty.",
                                  "Cold statistics use a mature policy, not original early-training data.",
                                  "Reward running-return state initially comes from a different checkpoint trajectory; late-half diagnostics reduce that transient."]}
        shared_raw = np.asarray(raw_samples)[::max(1, args.collect_steps // 128), 0]
        for name, mode in modes.items():
            current, target, same_chart = (np.asarray(mode[key]) for key in ("current_rows", "target_rows", "same_chart_rows"))
            record = {}
            for window_name, selection in (("all", slice(None)), ("late_half", slice(args.collect_steps // 2, None))):
                physical = same_chart[selection] - current[selection]
                coordinate = target[selection] - same_chart[selection]
                record[window_name + "/chart_rms"] = float(np.sqrt(np.mean(coordinate ** 2)))
                record[window_name + "/physical_step_rms"] = float(np.sqrt(np.mean(physical ** 2)))
                record[window_name + "/chart_to_physical_rms"] = float(np.sqrt(np.mean(coordinate ** 2) / max(np.mean(physical ** 2), 1e-20)))
                scales = np.asarray(mode["reward_scales"])[selection]
                record[window_name + "/reward_scale_cross_env_cv"] = float(np.mean(scales.std(1) / scales.mean(1)))
            with torch.no_grad():
                projected = [project(torch.as_tensor(values.reshape(-1, shape[0]), device="cuda", dtype=torch.float32)).clone()
                             for values in (current, target, same_chart)]
                p0, p1, psame = projected
                record["projected_chart_mse"] = float((p1 - psame).square().mean().cpu())
                record["projected_physical_step_mse"] = float((psame - p0).square().mean().cpu())
            norm = mode["obs_norm"]
            scale = np.sqrt(norm.variances + norm.epsilon)
            aliases = np.clip((shared_raw[:, None] - norm.means[None]) / scale[None], -norm.clip, norm.clip)
            record["same_raw_state_cross_env_coordinate_rms"] = float(np.sqrt(aliases.var(axis=1).mean()))
            record["per_coordinate_same_raw_state_cross_env_rms"] = np.sqrt(aliases.var(axis=1).mean(0)).tolist()
            record["obs_clip_fraction"] = mode["clipped_count"] / mode["element_count"]
            record["reward_clip_fraction"] = mode["reward_clipped_count"] / mode["reward_count"]
            record["mean_shift_in_initial_std_rms"] = float(np.sqrt(np.mean(((norm.means - mode["initial_mean"]) / mode["initial_scale"]) ** 2)))
            record["scale_relative_change_rms"] = float(np.sqrt(np.mean((scale / mode["initial_scale"] - 1) ** 2)))
            result["modes"][name] = record
            for key, value in record.items():
                if isinstance(value, (float, int)):
                    writer.add_scalar(f"normalization/{name}/{key}", value, args.collect_steps * n)
        (output / "normalization_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        envs.close()
        writer.close()


if __name__ == "__main__":
    main()

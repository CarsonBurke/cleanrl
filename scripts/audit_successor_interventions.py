"""Measure action-conditioned successor credit by real MuJoCo interventions via mlq.

Frozen v16 policy; no fitting of its actor, critic or control network. Exact native
mjData copies retain solver warm starts, and replay is checked against a factual
collector transition. Future Beta quantiles share uniforms across first-action
branches (common random numbers), with independent repeat streams. A per-state
compatible-control fit uses disjoint first actions AND future seeds from evaluation.
This finite-data oracle is a diagnostic, not a certified variance lower bound.
"""
import ctypes
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.special import betaincinv
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl import ppo_continuous_action_successor_score_control_v16 as model
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.vector_norm import VectorObsNorm


@dataclass
class Args:
    checkpoint: Path
    exp_name: str = "successor_intervention_v16"
    anchors: int = 16
    actions: int = 64
    repeats: int = 8
    horizons: tuple[int, ...] = (1, 8, 32, 128, 512)
    anchor_spacing: int = 64
    seed: int = 1
    env_threads: int = 2


class FrozenObsNorm(VectorObsNorm):
    def normalize(self, obs, rows=None, out_dtype=np.float32):
        mean = self.means if rows is None else self.means[rows]
        variance = self.variances if rows is None else self.variances[rows]
        normalized = (np.asarray(obs) - mean) / np.sqrt(variance + self.epsilon)
        if self.clip is not None:
            np.clip(normalized, -self.clip, self.clip, out=normalized)
        return normalized.astype(out_dtype, copy=False)


def copy_function(env):
    # MuJoCo2.3.3 lacks Python mj_copyData/mj_getState. The native backend already
    # links this public C ABI and owns the same live Python model/data pointers.
    function = env._native.mj_copyData
    function.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    function.restype = ctypes.c_void_p
    return function


def restore_branches(env, copy_data, snapshot, age):
    for base in env._bases:
        if not copy_data(base.data._address, base.model._address, snapshot._address):
            raise RuntimeError("MuJoCo full-data copy failed")
    env._episode_lengths.fill(age)
    env._episode_returns.fill(0)
    for limit in env._limits:
        limit._elapsed_steps = age


def analyze_branches(returns, native, alpha, beta, coefficients, gram):
    """One horizon: returns[S,R,A,2], independent train/held actions and seeds."""
    total = returns.sum(-1)
    states, repeats, actions = total.shape
    train_r, train_a = repeats // 2, actions // 2
    # CRN induces cross-action covariance. Remove each repeat's common component
    # before estimating the finite-repeat contamination of between-action means.
    centered = total - total.mean(-1, keepdim=True)
    noise_of_action_means = centered.var(1, correction=1).mean(-1) / repeats
    signal = (centered.mean(1).square().mean(-1) - noise_of_action_means) * actions / (actions - 1)
    future_noise = total.var(1, correction=1).mean(-1)
    score = model.beta_score(native, alpha[:, None, :], beta[:, None, :])
    jacobian = model.beta_logit_jacobian(alpha, beta)
    dimensions = score.shape[-1]
    identity = torch.eye(dimensions, dtype=score.dtype, device=score.device)
    fisher = model.beta_fisher_product(alpha[:, None, :], beta[:, None, :],
                                       identity.expand(states, dimensions, dimensions))
    # D maps [b0,b] to the zero-mean logit-gradient correction.
    design = torch.cat((score.unsqueeze(-1), score.unsqueeze(-1) * score.unsqueeze(-2)
                        - fisher[:, None, :, :]), dim=-1) * jacobian[:, None, :, None]
    training = total[:, :train_r, :train_a]
    mean = training.mean((1, 2))
    scale = training.flatten(1).std(-1, correction=0).clamp_min(1e-12)
    targets = (total - mean[:, None, None]) / scale[:, None, None]
    base = jacobian[:, None, None, :] * score[:, None, :, :] * targets.unsqueeze(-1)
    train_design = design[:, :train_a]
    weighted_design = gram[:, None] @ train_design
    normal = (train_design.transpose(-1, -2) @ weighted_design).mean(1)
    rhs = (weighted_design.transpose(-1, -2) @ base[:, :train_r, :train_a].mean(1).unsqueeze(-1)).mean(1)
    # Numerical minimum-norm solve, not a tuned regularizer or a trusted oracle.
    fitted = (torch.linalg.pinv(normal, hermitian=True, rtol=1e-10) @ rhs).squeeze(-1)
    held_base = base[:, train_r:, train_a:]
    held_corrected = held_base - (design[:, train_a:] @ fitted[:, None, :, None]).squeeze(-1)[:, None]
    held_metric = gram[:, None, None]
    base_power = (held_base.unsqueeze(-2) @ held_metric @ held_base.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    corrected_power = (held_corrected.unsqueeze(-2) @ held_metric @ held_corrected.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    train_base = base[:, :train_r, :train_a]
    train_corrected = train_base - (train_design @ fitted[:, None, :, None]).squeeze(-1)[:, None]
    train_base_power = (train_base.unsqueeze(-2) @ held_metric @ train_base.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    train_corrected_power = (train_corrected.unsqueeze(-2) @ held_metric @ train_corrected.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    current = model.control_sample(coefficients[:, None, :], score)
    current = current[:, train_a:] - current[:, train_a:].mean(-1, keepdim=True)
    held_action_return = total[:, train_r:, train_a:].mean(1)
    held_action_return = held_action_return - held_action_return.mean(-1, keepdim=True)
    correlation = (current * held_action_return).mean() / (
        current.square().mean() * held_action_return.square().mean()).sqrt().clamp_min(1e-12)
    singular = torch.linalg.eigvalsh(normal)
    return {
        "action_signal_variance_estimate": signal.mean(),
        "future_sampling_variance": future_noise.mean(),
        "action_signal_fraction_estimate": signal.mean() / (signal.mean() + future_noise.mean()).clamp_min(1e-12),
        "action_signal_standard_error_across_anchors": signal.std(correction=1) / states ** 0.5,
        "negative_action_signal_estimate_fraction": (signal < 0).double().mean(),
        "oracle_train_gradient_second_moment_ratio": train_corrected_power.mean() / train_base_power.mean().clamp_min(1e-12),
        "oracle_heldout_gradient_second_moment_ratio": corrected_power.mean() / base_power.mean().clamp_min(1e-12),
        "oracle_heldout_ratio_median_across_anchors": (corrected_power.mean((1, 2)) / base_power.mean((1, 2)).clamp_min(1e-12)).median(),
        "oracle_normal_condition_median": (singular[:, -1] / singular[:, 0].clamp_min(1e-20)).median(),
        "trained_control_action_credit_correlation": correlation,
        "first_factor_mean": returns[..., 0].mean(),
        "second_factor_mean": returns[..., 1].mean(),
    }


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; submit this experiment through mlq")
    if args.anchors < 2 or args.actions < 4 or args.repeats < 4 or args.actions % 2 or args.repeats % 2:
        raise ValueError("need multiple anchors and even action/repeat counts for held-out splits")
    if not args.horizons or min(args.horizons) <= 0 or len(set(args.horizons)) != len(args.horizons):
        raise ValueError("horizons must be distinct and positive")
    if args.anchor_spacing <= 0:
        raise ValueError("anchor_spacing must be positive")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = model.Args(**checkpoint["args"])
    horizon = max(args.horizons)
    episode_steps = episode_horizon(config.env_id)
    if horizon >= episode_steps:
        raise ValueError("factual branches must fit strictly inside an episode")
    output = Path("runs") / f"{config.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(str(output))
    writer.add_text("hyperparameters", json.dumps(vars(args), default=str))
    collector = make_mujoco_vector_env(config.env_id, config.num_envs, num_threads=args.env_threads)
    branches = None
    try:
        agent = model.Agent(collector, config).cuda().eval().requires_grad_(False)
        agent.load_state_dict(checkpoint["model"])
        state = checkpoint["obs_norm"]
        norm = FrozenObsNorm(config.num_envs, collector.single_observation_space.shape,
                             epsilon=state["epsilon"], clip=state["clip"])
        for name in ("means", "variances", "counts"):
            getattr(norm, name)[:] = state[name].numpy()
        host = model.HostPolicy(agent, config.num_envs)
        low, high = agent.action_low.cpu().numpy(), agent.action_high.cpu().numpy()
        sampler = make_beta_sampler(config.num_envs, agent.action_dim, low, high)
        rng = np.random.default_rng(args.seed)

        def act(observations):
            return sampler(host(observations), rng)[1]

        warm = run_phase_warmup(collector, obs_norm=norm, rew_norm=None, act_fn=act,
                                horizon=episode_steps, phase_offsets=compute_phase_offsets(config.num_envs, episode_steps, args.seed),
                                seed=args.seed)
        obs = warm.next_obs
        anchors = []
        copy_data = copy_function(collector)
        collection_steps = 0
        while len(anchors) < args.anchors:
            actions = act(obs)
            record = None
            if collection_steps % args.anchor_spacing == 0:
                eligible = np.flatnonzero(collector._episode_lengths < episode_steps - horizon)
                if eligible.size:
                    index = int(eligible[len(anchors) % eligible.size])
                    base = collector._bases[index]
                    snapshot = mujoco.MjData(base.model)
                    if not copy_data(snapshot._address, base.model._address, base.data._address):
                        raise RuntimeError("MuJoCo anchor snapshot failed")
                    record = {"snapshot": snapshot, "index": index, "age": int(collector._episode_lengths[index]),
                              "observation": obs[index].copy(), "reference_action": actions[index].copy()}
            raw_next, reward, terms, truncs, _ = collector.step(actions)
            if record is not None:
                index = record["index"]
                if terms[index] or truncs[index]:
                    raise RuntimeError("eligible anchor unexpectedly crossed an episode boundary")
                record["reference_next"] = raw_next[index].copy()
                record["reference_reward"] = float(reward[index])
                anchors.append(record)
            obs = norm.normalize(raw_next)
            collection_steps += 1
            if collection_steps > args.anchors * (episode_steps + args.anchor_spacing):
                raise RuntimeError("could not collect sufficient within-episode anchor states")
        branches = make_mujoco_vector_env(config.env_id, args.actions, num_threads=args.env_threads)
        branches.reset(seed=args.seed)
        branch_host = model.HostPolicy(agent, args.actions)
        branch_sampler = make_beta_sampler(args.actions, agent.action_dim, low, high)
        copy_branch = copy_function(branches)
        rewards_state = checkpoint["reward_norm"]
        stds = np.sqrt(rewards_state["variances"].numpy() + rewards_state["epsilon"])
        records = np.empty((args.anchors, args.repeats, args.actions, len(args.horizons), 2), dtype=np.float64)
        natives = np.empty((args.anchors, args.actions, agent.action_dim), dtype=np.float32)
        replay_error = 0.0
        reconstructed_reward_error = 0.0
        branch_steps = 0
        for anchor_index, anchor in enumerate(anchors):
            restore_branches(branches, copy_branch, anchor["snapshot"], anchor["age"])
            replay_obs, replay_reward, _, _, _ = branches.step(np.repeat(anchor["reference_action"][None], args.actions, axis=0))
            error = max(float(np.max(np.abs(replay_obs - anchor["reference_next"]))),
                        float(np.max(np.abs(replay_reward - anchor["reference_reward"]))))
            replay_error = max(replay_error, error)
            if error > 1e-10:
                raise RuntimeError(f"full MuJoCo state replay failed: max error {error}")
            branch_steps += args.actions
            initial = np.repeat(anchor["observation"][None], args.actions, axis=0)
            first_native, first_action = branch_sampler(branch_host(initial), rng)
            natives[anchor_index] = first_native.copy()
            first_action = first_action.copy()
            source_row = anchor["index"]
            for repeat in range(args.repeats):
                restore_branches(branches, copy_branch, anchor["snapshot"], anchor["age"])
                branch_obs = initial
                total = np.zeros((args.actions, 2), dtype=np.float64)
                # Each repeat is independent; uniforms within a repeat couple
                # branch continuations without changing their Beta marginals.
                uniforms = rng.random((horizon - 1, agent.action_dim))
                for step in range(horizon):
                    if step == 0:
                        action = first_action
                    else:
                        concentrations = np.logaddexp(0, branch_host(branch_obs)) + 1
                        alpha, beta = np.split(concentrations, 2, axis=-1)
                        native = betaincinv(alpha, beta, uniforms[step - 1][None]).astype(np.float32)
                        np.clip(native, model.SAMPLE_EPS, 1 - model.SAMPLE_EPS, out=native)
                        action = low + (high - low) * native
                    raw_next, raw_reward, terms, truncs, infos = branches.step(action)
                    if np.any(terms | truncs):
                        raise RuntimeError("branch crossed a boundary despite anchor-age eligibility")
                    factors = np.stack((infos["reward_run"], infos["reward_ctrl"]), axis=-1)
                    reconstructed_reward_error = max(reconstructed_reward_error,
                        float(np.max(np.abs(factors.sum(-1) - raw_reward))))
                    divisor = np.full(args.actions, stds[source_row])
                    if rewards_state["clip"] is not None:
                        divisor = np.maximum(divisor, np.abs(raw_reward) / rewards_state["clip"])
                    total += config.gamma ** step * factors / divisor[:, None]
                    if step + 1 in args.horizons:
                        records[anchor_index, repeat, :, args.horizons.index(step + 1)] = total
                    branch_obs = norm.normalize(raw_next, rows=np.full(args.actions, source_row))
                    branch_steps += args.actions
            print(f"anchor={anchor_index + 1}/{args.anchors}, branch_transitions={branch_steps}", flush=True)
        if 1 in args.horizons:
            first = records[..., args.horizons.index(1), :]
            if float(np.max(np.ptp(first, axis=1))) > 1e-12:
                raise RuntimeError("identical first action/state changed reward across repeats")
        observations = torch.as_tensor(np.stack([anchor["observation"] for anchor in anchors]), device="cuda")

        def predict(values):
            alpha, beta, _ = agent.get_policy_and_value(values)
            return alpha, beta, agent.control(values), model.actor_jacobian_gram(agent.actor, values)

        predict = torch.compile(predict, fullgraph=True, options={"triton.cudagraphs": False})
        analyze = torch.compile(analyze_branches, fullgraph=True, options={"triton.cudagraphs": False})
        results = {}
        with torch.no_grad():
            alpha, beta, coefficients, gram = (value.double() for value in predict(observations))
            native_tensor = torch.as_tensor(natives, device="cuda", dtype=torch.float64)
            for index, steps in enumerate(args.horizons):
                returned = torch.as_tensor(records[..., index, :].copy(), device="cuda", dtype=torch.float64)
                metrics = analyze(returned, native_tensor, alpha, beta, coefficients, gram)
                results[str(steps)] = {name: float(value.cpu()) for name, value in metrics.items()}
                if not all(np.isfinite(value) for value in results[str(steps)].values()):
                    raise FloatingPointError("nonfinite intervention diagnostic")
                for name, value in results[str(steps)].items():
                    writer.add_scalar(f"intervention/h{steps}/{name}", value, branch_steps)
        result = {
            "checkpoint": str(args.checkpoint), "mode": config.control_mode,
            "checkpoint_step": checkpoint["global_step"], "anchors": args.anchors,
            "actions_per_anchor": args.actions, "future_repeats": args.repeats,
            "branch_transitions": branch_steps, "collection_transitions": collection_steps * config.num_envs,
            "warmup_transitions": warm.transitions, "factual_replay_max_error": replay_error,
            "reward_factor_reconstruction_max_error": reconstructed_reward_error, "horizons": results,
            "limitations": [
                "Frozen policy and checkpoint normalization; no training or benchmark return improvement from these extra evaluation transitions.",
                "Finite-horizon actual rewards, no critic bootstrap; these are not exact infinite-horizon Q values.",
                "Shared future uniforms reduce contrast-estimation noise but make action branches correlated within each repeat.",
                "Oracle fit uses disjoint actions and repeat seeds; finite-data capacity diagnostic, not a certified population optimum.",
                "Signed noise-corrected action-variance estimates may be negative; do not silently clamp or interpret as negative true variance.",
                "Control correlation compares action ordering, not calibrated return units; control was trained on normalized GAE, not these finite returns.",
                "One trained policy seed and a small correlated set of anchor states; anchor standard error is descriptive, not seed uncertainty.",
            ],
        }
        (output / "intervention_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if branches is not None:
            branches.close()
        collector.close()
        writer.close()


if __name__ == "__main__":
    main()

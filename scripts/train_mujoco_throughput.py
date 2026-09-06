"""Execution-only v30 proxy integrating the shared throughput infrastructure.

The model, loss and algorithm arguments come from the frozen reference. This
adapter changes execution/storage only, keeps explicit batched next values and
v30's promotion rule, and exposes independent switches for controlled ablation.
Validate fixed-work parity before any full seed-1 8M job through mlq. Fused
projection/temperature candidates failed the optimizer parity gate and remain
benchmark-only. Manual learner capture is excluded because compiled-peer
integration is unsafe; there is no GPU-physics substitution in this runner.
"""

import copy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.collector import OnPolicyCollector
from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import gather_metrics, get_gae_fn
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.sampling import sample_beta_actions
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm
from cleanrl_utils.reference_loss import load_reference_loss

REFERENCE = "cleanrl.vmpo.ppo_continuous_action_iterthink_v24_beta_vmpo_v30_dreamer_bucket_moment_hlgauss_reward_norm"
reference = importlib.import_module(REFERENCE)


@dataclass
class Args(reference.Args):
    exp_name: str = "v30_shared_pipeline_v2"
    env_backend: str = "native"
    env_threads: int = 4
    non_blocking_transfers: bool = True


METRICS = (
    "losses/policy_loss", "losses/value_loss", "losses/temperature_loss",
    "vmpo/mean_kl", "vmpo/concentration_kl", "vmpo/full_beta_kl",
    "vmpo/weight_ess_fraction", "vmpo/top_advantage_min", "debug/advantage_mean",
    "debug/advantage_std", "vmpo/e_step_kl", "vmpo/eta_stationarity",
    "vmpo/weight_perplexity_fraction", "vmpo/max_weight", "vmpo/weight_ess",
    "vmpo/mean_kl_residual", "vmpo/concentration_kl_residual", "debug/value_rmse",
    "debug/value_explained_variance", "critic/target_outside_support",
    "critic/target_edge_mass", "critic/prediction_edge_mass",
    "debug/policy_concentration", "debug/policy_native_variance", "vmpo/eta",
)


def validate(args):
    if args.track:
        raise ValueError("throughput comparisons log locally; tracking is not supported")
    if not args.cuda or not args.bf16 or not args.compile:
        raise ValueError("the v30 proxy requires CUDA, BF16 and compilation")
    if args.num_envs <= 0 or args.num_steps != 39:
        raise ValueError("v30 requires positive num_envs and 39-step unrolls")
    if args.log_interval <= 0 or args.target_update_period <= 0 or args.target_update_period % args.log_interval:
        raise ValueError("target_update_period must be positive and divisible by log_interval")
    args.batch_size = args.num_envs * args.num_steps
    args.topk_size = int(args.batch_size * args.topk_fraction)
    if not 0 < args.topk_size <= args.batch_size or not 0 <= args.gae_lambda <= 1:
        raise ValueError("invalid top-k or GAE lambda")
    if not 0 <= args.return_percentile_low < args.return_percentile_high <= 1 or args.return_percentile_floor <= 0:
        raise ValueError("invalid return percentile settings")
    if args.num_value_bins < 3 or args.num_value_bins % 2 == 0 or args.value_support_limit <= 0 or args.value_sigma_to_bin_ratio <= 0:
        raise ValueError("invalid value support settings")
    args.initial_phase_warmup_steps = episode_horizon(args.env_id)
    args.num_iterations = (args.total_timesteps - args.num_envs * args.initial_phase_warmup_steps) // args.batch_size
    if args.num_iterations < 1:
        raise ValueError("budget must contain phase warmup and a full rollout")


def main():
    args = tyro.cli(Args)
    validate(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 is required")
    configure_runtime(cudnn_deterministic=args.torch_deterministic)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{name}")
    writer.add_text("hyperparameters", json.dumps(vars(args), indent=2))
    envs = make_mujoco_vector_env(args.env_id, args.num_envs, backend=args.env_backend,
                                  num_threads=args.env_threads, capture_video=args.capture_video, run_name=name)
    collector = None
    try:
        agent = reference.Agent(envs, args).to(device)
        target = copy.deepcopy(agent).requires_grad_(False)
        duals = torch.nn.Parameter(torch.tensor([args.initial_alpha_mean, args.initial_alpha_concentration], device=device))
        limit = float(np.log1p(args.value_support_limit))
        support = Dreamer3BucketHLGaussSupport(args.num_value_bins, -limit, limit, args.value_sigma_to_bin_ratio, device)
        optimizer = torch.optim.Adam([*agent.parameters(), duals], lr=args.learning_rate,
                                      betas=(0.9, 0.999), eps=1e-8, fused=True)
        raw_loss, source_hash = load_reference_loss(reference, dict(args=args, agent=agent,
            duals=duals, hl_support=support, autocast_dtype=torch.bfloat16))
        writer.add_text("benchmark/reference_sha256", source_hash)
        writer.add_text("benchmark/reference", REFERENCE)

        def rollout_model(obs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                alpha, beta = target.policy(obs)
                logits = agent.value_logits(obs)
            return alpha.float(), beta.float(), support.to_scalar(logits.float())

        def value_model(obs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = agent.value_logits(obs)
            return support.to_scalar(logits.float())

        rollout = graph_compile(rollout_model)
        gae = get_gae_fn(compiled=True, mode=args.compile_mode, explicit_next_values=True)

        def rollout_gae(transitions, rewards, current_values, terms, truncs):
            # Keep next-critic decoding and GAE in one graph, as frozen v30 does.
            next_values = value_model(transitions.flatten(0, 1)).view(args.num_steps, args.num_envs)
            return gae(rewards, current_values, terms, truncs, next_values, args.gamma, args.gae_lambda)

        gae_model = torch.compile(rollout_gae, mode=args.compile_mode, fullgraph=True, dynamic=False)

        def sample(obs):
            alpha, beta, value = rollout(obs)
            native, action = sample_beta_actions(alpha, beta, target.action_low, target.action_high)
            return dict(action=action, native_action=native, alpha=alpha, beta=beta, value=value)

        suppress = np.zeros(args.num_envs, dtype=bool)

        def log_episodes(infos, step):
            for i, info in enumerate(infos.get("final_info", ())):
                if info and "episode" in info:
                    if suppress[i]:
                        suppress[i] = False
                        continue
                    for label, key in (("return", "r"), ("length", "l")):
                        writer.add_scalar(f"charts/episodic_{label}", float(info["episode"][key]), step)

        obs_norm = VectorObsNorm(args.num_envs, envs.single_observation_space.shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        collector = OnPolicyCollector(envs, args.num_steps, sample, obs_norm, rew_norm,
                                      non_blocking=args.non_blocking_transfers,
                                      episode_callback=log_episodes)
        start = time.perf_counter()
        offsets = compute_phase_offsets(args.num_envs, args.initial_phase_warmup_steps, args.seed)
        writer.add_text("initial_phase_offsets", ",".join(map(str, offsets)))

        def warmup_action(obs):
            action = collector.graph.step(obs)
            if not np.isfinite(action).all():
                raise FloatingPointError("nonfinite warmup action")
            return action

        warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm, act_fn=warmup_action,
                                horizon=args.initial_phase_warmup_steps, phase_offsets=offsets, seed=args.seed)
        suppress[:] = warm.suppress_mask
        collector.set_observation(warm.next_obs, total_steps=warm.transitions)
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start, collector.total_steps)
        compiled_loss = torch.compile(raw_loss, mode=args.compile_mode, fullgraph=True, dynamic=False)
        timer = PhaseTimer()
        levels = torch.tensor([args.return_percentile_low, args.return_percentile_high], device=device)
        age = promotions = 0
        interval_start, interval_step = time.perf_counter(), collector.total_steps
        for iteration in range(1, args.num_iterations + 1):
            batch = collector.collect()
            rollout_age = age
            with timer.span("gae"), torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                advantages, returns = gae_model(batch.transitions.transition_observations,
                                                batch.transitions.rewards, batch.policy["value"],
                                                batch.transitions.terminations, batch.transitions.truncations)
                advantages, targets = advantages.flatten().clone(), returns.flatten().clone()
                quantiles = torch.quantile(targets, levels)
                scale = (quantiles[1] - quantiles[0]).clamp_min(args.return_percentile_floor)
                advantages.div_(scale)
            inputs = (batch.observations.flatten(0, 1), batch.policy["native_action"].flatten(0, 1),
                      batch.policy["alpha"].flatten(0, 1), batch.policy["beta"].flatten(0, 1), advantages, targets)
            should_log = iteration % args.log_interval == 0 or iteration == 1
            before = duals.detach().clone() if should_log else None
            with timer.span("update"):
                torch.compiler.cudagraph_mark_step_begin()
                loss, metrics = compiled_loss(*inputs)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    duals.clamp_(min=reference.DUAL_FLOOR)
            age += 1
            if should_log:
                log = gather_metrics(dict(zip(METRICS, metrics)) | {
                    "vmpo/alpha_mean": duals[0], "vmpo/alpha_concentration": duals[1],
                    "vmpo/alpha_mean_delta": duals[0] - before[0],
                    "vmpo/alpha_concentration_delta": duals[1] - before[1],
                    "debug/return_percentile_scale": scale,
                })
                if not all(np.isfinite(value) for value in log.values()):
                    raise FloatingPointError("nonfinite learner metric at an existing logging synchronization")
                mean_trigger = iteration % args.log_interval == 0 and log["vmpo/mean_kl"] >= args.epsilon_alpha_mean
                age_trigger = age >= args.target_update_period
                if mean_trigger or age_trigger:
                    target.load_state_dict(agent.state_dict())
                    age = 0
                    promotions += 1
                for name_, value in log.items():
                    writer.add_scalar(name_, value, collector.total_steps)
                for name_, value in {"target_age_batches": rollout_age, "target_age_transitions": rollout_age * args.batch_size,
                                     "target_promoted": mean_trigger or age_trigger, "target_promoted_for_mean_kl": mean_trigger,
                                     "target_promoted_for_max_age": age_trigger, "target_promotions": promotions,
                                     "learner_updates": iteration}.items():
                    writer.add_scalar(f"vmpo/{name_}", value, collector.total_steps)
                for phases in (collector.timer.summary(), timer.summary()):
                    for phase, measured in phases.items():
                        writer.add_scalar(f"timing/{phase}_s", measured["total_s"], collector.total_steps)
                end = time.perf_counter()
                rate = (collector.total_steps - interval_step) / (end - interval_start)
                writer.add_scalar("charts/interval_SPS", rate, collector.total_steps)
                writer.add_scalar("charts/SPS", collector.total_steps / (end - start), collector.total_steps)
                writer.add_scalar("charts/learning_rate", args.learning_rate, collector.total_steps)
                writer.flush()
                print(f"step={collector.total_steps} interval_SPS={rate:.0f} value_loss={log['losses/value_loss']:.5g}", flush=True)
                collector.timer.reset()
                timer.reset()
                interval_start, interval_step = time.perf_counter(), collector.total_steps
        writer.add_scalar("benchmark/complete", 1, collector.total_steps)
    finally:
        try:
            (collector.close if collector is not None else envs.close)()
        finally:
            writer.close()


if __name__ == "__main__":
    main()

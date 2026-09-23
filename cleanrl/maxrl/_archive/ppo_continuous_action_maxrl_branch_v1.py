# MaxRL-branch v1: pass@k in continuous control via MuJoCo state cloning. Critic-free.
#
# The problem with every other MaxRL port. MaxRL's mechanism is CROSS-PROMPT reweighting:
# w_T(p) = (1-(1-p)^T)/p spends gradient on prompts whose pass rate is low. That needs
# (a) many prompts of differing difficulty and (b) N i.i.d. rollouts of EACH prompt, to
# estimate its p. MuJoCo offers neither by default: v4 initial states are a narrow
# perturbation of one pose, so there is effectively a single prompt, and you cannot re-roll
# from a state, so there is no group. maxrl_pure_v1 accepts both losses and degenerates
# into success-vs-failure imitation within one group.
#
# What this file does instead. MuJoCo state IS clonable. The native backend steps the real
# gym envs' mjData in place, so writing qpos/qvel into a sibling env teleports it: verified
# that a cloned group then produces bit-identical observations and rewards. So:
#   * split num_envs into groups of group_size; env 0 of each group is its LEADER.
#   * every segment_length steps, clone the leader's state into its group-mates. All N
#     members now continue from one state x -- a genuine prompt with a genuine group.
#     Their normalized observation is the leader's by construction, so nothing is
#     renormalized and the running statistics are not double-counted.
#   * the outcome is the discounted return over the segment; success is 1{return > bar}.
#     With N continuations of one state this is literally pass@k for control.
#   * the group's MaxRL advantage is broadcast to that member's segment timesteps, exactly
#     as the reference broadcasts a sequence advantage over a response's tokens.
# No critic, no GAE, no bootstrap: the outcome carries all of the signal.
#
# The bar must be GLOBAL, not per-group. A per-group quantile fixes the success count and
# so fixes 1/p, which erases the very cross-prompt variation this construction exists to
# create. An EMA'd global quantile lets easy states pass often and hard states rarely,
# which is what makes w(p) do anything.
#
# Hypothesis: with real per-state pass rates, MaxRL's upweighting of low-p states
# concentrates learning on the parts of state space the policy handles badly, and the
# forward-KL, mode-covering form preserves behavioural diversity (the paper's Fig. 14
# broad pass-rate distribution) where PPO's mean-seeking gradient collapses it.
#
# Costs, stated plainly. Cloning discards group_size-1 states per segment, so state
# coverage drops to num_groups independent trajectories -- bought back by raising num_envs.
# The segment return is truncated at segment_length with no bootstrap, making the objective
# myopic (gamma^125 leaves 71% of the discounted mass inside the window). Only LEADER
# episodes are reported as charts/episodic_return: a branch env is teleported mid-episode,
# so its accumulated return is a splice of unrelated trajectories and is not a valid score.
# The reward normalizer's per-env discounted accumulator is also left un-cloned; it only
# feeds a running scale estimate, so the drift is second order, but it is a real deviation.
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import device_minibatches, gather_metrics
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import episode_horizon
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
GROUP_EPS = 1e-6
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
ESTIMATORS = ("maxrl", "maclaurin", "grpo", "rloo")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    capture_video: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 64
    """more environments buy back the state coverage that cloning spends"""
    num_steps: int = 250
    group_size: int = 8
    """N: continuations sampled from each cloned state, so pass@k is measured at k <= N"""
    segment_length: int = 125
    """how long each branch runs before the next clone; must divide num_steps and the horizon"""
    anneal_lr: bool = True
    gamma: float = 0.99
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    maxrl_estimator: Literal["maxrl", "maclaurin", "grpo", "rloo"] = "maxrl"
    maxrl_order: int = 8
    maxrl_quantile: float = 0.6
    """global success bar; low enough that few groups land on a zero-gradient C=0"""
    maxrl_tau_ema: float = 0.95

    env_backend: str = "native"
    """cloning reaches into the native backend's mjData, so the backend is not negotiable"""
    env_threads: int = 4
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    non_blocking_transfers: bool = False

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    num_groups: int = 0
    segments_per_rollout: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor only: outcome advantages need no value function."""

    def __init__(self, envs):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta PPO requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )

    def policy(self, x):
        return (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)


def maclaurin_weights(successes, group_size, order):
    """Exact (w_succ, w_fail) for sum_{k=1..T} (1/k) grad pass@k; see maxrl/maclaurin.py.

    r_k = C(f-1,k-1)/C(N-1,k-1) is the chance that k-1 of the OTHER samples all failed, so
    w_fail carries the higher-order failure terms that REINFORCE (T=1) discards.
    """
    failures = (group_size - successes).to(torch.float64)
    w_succ = torch.full_like(failures, 1.0 / group_size)
    if order < 2 or group_size < 2:
        return w_succ, torch.zeros_like(w_succ)
    limit = min(order, group_size)
    ratio = (failures - 1.0) / float(group_size - 1)
    total = torch.zeros_like(w_succ)
    for k in range(2, limit + 1):
        total = total + torch.where(failures >= k, ratio, torch.zeros_like(ratio))
        if k < limit:
            ratio = ratio * (failures - float(k)) / float(group_size - k)
    return w_succ, -(total / float(group_size))


def group_advantages(segment_returns, bar, args):
    """MaxRL advantage per (segment, group, member), from per-state pass rates.

    ``segment_returns`` is (segments, groups, group_size): one outcome per branch. The
    pass rate is taken WITHIN a group -- that is the point of cloning, since every member
    of a group continues from the same state -- while the bar is global, so pass rates
    genuinely differ between groups.
    """
    rewards = (segment_returns > bar).float()
    pass_rate = rewards.mean(dim=-1, keepdim=True)

    if args.maxrl_estimator == "maxrl":
        advantages = (rewards - pass_rate) / (pass_rate + GROUP_EPS)
    elif args.maxrl_estimator == "grpo":
        advantages = (rewards - pass_rate) / (rewards.std(dim=-1, keepdim=True, unbiased=False) + GROUP_EPS)
    elif args.maxrl_estimator == "rloo":
        advantages = rewards - pass_rate
    elif args.maxrl_estimator == "maclaurin":
        w_succ, w_fail = maclaurin_weights(rewards.sum(dim=-1), args.group_size, args.maxrl_order)
        advantages = (w_succ.unsqueeze(-1) * rewards
                      + w_fail.unsqueeze(-1) * (1.0 - rewards)).to(segment_returns.dtype)
    else:
        raise ValueError(f"unknown estimator {args.maxrl_estimator}")

    # A group whose branches all failed, or all passed, carries no information about which
    # continuation to prefer. The paper drops those prompts rather than learning from them.
    informative = (rewards.sum(dim=-1, keepdim=True) > 0) & (rewards.sum(dim=-1, keepdim=True) < args.group_size)
    advantages = torch.where(informative, advantages, torch.zeros_like(advantages))

    diagnostics = torch.stack((
        bar, pass_rate.mean(), pass_rate.std(),
        (pass_rate.squeeze(-1) == 0.0).float().mean(),
        (pass_rate.squeeze(-1) == 1.0).float().mean(),
        informative.float().mean(), advantages.abs().mean(),
        segment_returns.mean(), segment_returns.std(),
    ))
    return advantages, diagnostics


def clone_group_states(bases, normalized_obs, group_size):
    """Teleport every group-mate onto its leader's state, making one prompt per group.

    The leader (member 0) is never written to, so its trajectory stays a genuine on-policy
    episode and is the only one whose episodic return is reported. Members inherit the
    leader's normalized observation verbatim: they are in the identical physical state, so
    recomputing it would only risk double-counting the observation normalizer.
    """
    for start in range(0, len(bases), group_size):
        leader = bases[start]
        qpos, qvel = leader.data.qpos.copy(), leader.data.qvel.copy()
        for member in range(start + 1, start + group_size):
            bases[member].set_state(qpos, qvel)
            normalized_obs[member] = normalized_obs[start]


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, args):
    """Clipped surrogate, no value term."""
    alpha, beta = agent.policy(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss
    metrics = torch.stack((pg_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac))
    return loss, metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.group_size < 2:
        raise ValueError("a pass rate needs at least two continuations per state")
    if args.num_envs % args.group_size:
        raise ValueError("num_envs must divide into whole groups")
    if args.num_steps % args.segment_length:
        raise ValueError("num_steps must be a whole number of segments")
    horizon = episode_horizon(args.env_id)
    if horizon:
        # A segment straddling an autoreset would mix rewards from two episodes into one
        # outcome, and a rollout straddling one would misalign the segment grid.
        if horizon % args.segment_length or horizon % args.num_steps:
            raise ValueError(f"segment_length and num_steps must divide the {args.env_id} "
                             f"horizon of {horizon}")
    if args.env_backend != "native":
        raise ValueError("state cloning requires the native backend's in-place mjData")
    args.num_groups = args.num_envs // args.group_size
    args.segments_per_rollout = args.num_steps // args.segment_length
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    if args.maxrl_order < 1:
        raise ValueError("maxrl_order is a truncation level T >= 1")
    if not 0.0 < args.maxrl_quantile < 1.0:
        raise ValueError("maxrl_quantile must lie strictly inside (0, 1)")
    return args


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.env_id not in NATIVE_TASKS or gym.__version__ != "0.29.1":
        raise ValueError("state cloning is verified only for the native v4 MuJoCo tasks")
    configure_runtime(cudnn_deterministic=args.torch_deterministic,
                      matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover a full rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("maxrl", f"critic-free per-state groups via mjData cloning; "
                                 f"estimator={args.maxrl_estimator}; T={args.maxrl_order}; "
                                 f"{args.num_groups} groups of N={args.group_size}; "
                                 f"segment={args.segment_length}; global bar q={args.maxrl_quantile}")
        envs = make_mujoco_vector_env(
            args.env_id, args.num_envs, backend="native",
            num_threads=min(args.env_threads, args.num_envs),
            capture_video=args.capture_video, run_name=run_name,
        )
        resources.callback(envs.close)
        bases = getattr(envs, "_bases", None)
        if bases is None or len(bases) != args.num_envs or not hasattr(bases[0], "set_state"):
            raise RuntimeError("native backend did not expose settable per-environment mjData")
        leaders = np.arange(args.num_groups) * args.group_size

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)

        def rollout_statistics(observations, native):
            alpha, beta = agent.policy(observations)
            return agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        obs_shape = envs.single_observation_space.shape
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 5), device=device)
        success_bar = torch.zeros((), device=device)
        bar_initialised = False
        discounts = (args.gamma ** torch.arange(args.segment_length, device=device, dtype=torch.float32))
        timer = PhaseTimer()
        start_time = time.perf_counter()
        raw_obs, _ = envs.reset(seed=args.seed)
        next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span("clone", use_cuda=False):
                    if step % args.segment_length == 0:
                        clone_group_states(bases, next_obs_np, args.group_size)
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, _ = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native)
                global_step += args.num_envs
                final_infos = infos.get("final_info", ())
                for index in leaders:
                    info = final_infos[index] if index < len(final_infos) else None
                    if info and "episode" in info:
                        # Leaders only: a branch env was teleported mid-episode, so its
                        # accumulated return splices unrelated trajectories together.
                        episode_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_logprobs = rollout_statistics(b_obs, b_native)
                # (steps, envs) -> (segments, length, groups, members) -> one outcome each.
                shaped = batch.rewards.view(args.segments_per_rollout, args.segment_length,
                                            args.num_groups, args.group_size)
                segment_returns = (shaped * discounts[None, :, None, None]).sum(dim=1)
                batch_bar = torch.quantile(segment_returns.flatten(), args.maxrl_quantile)
                if bar_initialised:
                    success_bar.mul_(args.maxrl_tau_ema).add_(batch_bar, alpha=1.0 - args.maxrl_tau_ema)
                else:
                    success_bar.copy_(batch_bar)
                    bar_initialised = True
                advantages, maxrl_diagnostics = group_advantages(segment_returns, success_bar, args)
                # Broadcast each branch's outcome back over its own segment timesteps.
                b_advantages = advantages.unsqueeze(1).expand(
                    -1, args.segment_length, -1, -1
                ).reshape(args.num_steps, args.num_envs).flatten().clone()

            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices], b_advantages[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    if args.target_kl is not None and update_metrics[updates - 1, 3] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/entropy": last[1],
                "losses/old_approx_kl": last[2], "losses/approx_kl": last[3],
                "losses/clipfrac": update_metrics[:updates, 4].mean(),
                **dict(zip((
                    "maxrl/success_bar", "maxrl/pass_rate_mean", "maxrl/pass_rate_std",
                    "maxrl/groups_all_failed", "maxrl/groups_all_passed",
                    "maxrl/informative_groups", "maxrl/advantage_scale",
                    "maxrl/segment_return_mean", "maxrl/segment_return_std",
                ), maxrl_diagnostics.unbind())),
            })
            if any(not np.isfinite(value) for value in logged.values()):
                raise FloatingPointError("nonfinite MaxRL learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step
    finally:
        resources.close()


if __name__ == "__main__":
    main()

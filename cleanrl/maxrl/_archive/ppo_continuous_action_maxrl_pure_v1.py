# MaxRL-pure v1: the reference MaxRL estimator, critic-free, on episode groups.
#
# This is MaxRL as the authors implement it (verl fork, core_algos.py), not a
# reinterpretation. Their registered estimator is an OUTCOME advantage over a group of
# N rollouts of one prompt, with no value network anywhere:
#     maxrl:      A_i = (r_i - mean) / (mean + eps)      <- Algorithm 1
#     maclaurin:  A_i = w_succ if r_i else w_fail, the exact hypergeometric estimator of
#                 sum_{k=1..T} (1/k) grad pass@k (maclaurin.py)
#     grpo:       A_i = (r_i - mean) / (std + eps)
#     rloo:       A_i = r_i - mean
# The sequence-level advantage is broadcast to every token of the response. Here the
# "prompt" is the initial state, the "response" is a whole episode, and the advantage is
# broadcast to every timestep of that episode.
#
# What this is for. It is the faithfulness control for ppo_continuous_action_maxrl_v1,
# which keeps the critic and reweights per state. Only one factor differs from the PPO
# baseline: the advantage. The clipped surrogate, optimizer, epochs, minibatching and
# normalization are untouched, so a gap is attributable to the outcome advantage alone.
#
# Honest prediction. MuJoCo has effectively ONE prompt: v4 initial states are a narrow
# perturbation of a fixed pose, so every group is drawn from the same x and MaxRL's
# cross-prompt reweighting -- the mechanism the paper's Fig. 1 is about -- has nothing to
# discriminate. What survives within a group is success-vs-failure imitation. Credit also
# collapses from 16k dense per-state advantages per iteration to num_envs bits. This is
# expected to lose badly to PPO; the point is to measure WHERE MaxRL's assumptions break
# when they meet dense-reward control, and to make maxrl_branch_v1's per-state groups
# interpretable by contrast.
#
# The bar. Binary success needs a verifier. --maxrl-bar-mode group_quantile marks the top
# (1-q) of each group successful, which pins the pass rate away from the degenerate ends:
# the paper drops groups with C=0, and with a single group per iteration C=0 would mean a
# zero gradient for that whole iteration. global_ema instead ratchets an absolute bar and
# lets the pass rate move, which is closer to a fixed verifier.
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
ESTIMATORS = ("maxrl", "maclaurin", "grpo", "rloo", "dense_maxrl")


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
    num_envs: int = 16
    """group size N: every environment contributes one episode per iteration"""
    num_steps: int = 1000
    """must equal the episode horizon so a rollout segment is exactly one episode"""
    anneal_lr: bool = True
    gamma: float = 0.99
    """discount for the outcome return; 1.0 recovers the undiscounted episode return"""
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # MaxRL outcome advantage.
    maxrl_estimator: Literal["maxrl", "maclaurin", "grpo", "rloo", "dense_maxrl"] = "maxrl"
    """maxrl is Algorithm 1; maclaurin is the exact pass@k series; the rest are its controls"""
    maxrl_order: int = 8
    """Maclaurin truncation order T; unused by the closed-form estimators"""
    maxrl_quantile: float = 0.75
    """success bar: the top (1 - q) of the group, or the q-quantile ratchet under global_ema"""
    maxrl_bar_mode: Literal["group_quantile", "global_ema"] = "group_quantile"
    maxrl_tau_ema: float = 0.95

    env_backend: str = "auto"
    env_threads: int = 4
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    non_blocking_transfers: bool = False

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor only. Outcome advantages need no value function, so there is no critic."""

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

    def get_action_and_logprob(self, x, action=None):
        alpha, beta = self.policy(x)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((x.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(x.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy


def maclaurin_weights(successes, group_size, order):
    """Exact (w_succ, w_fail) for sum_{k=1..T} (1/k) grad pass@k, following maclaurin.py.

    r_k = C(f-1, k-1)/C(N-1, k-1) is the chance that k-1 of the OTHER samples all failed,
    so w_fail accumulates the higher-order failure terms that REINFORCE (T=1) discards.
    Built with a running ratio rather than factorials, which overflow well before N=64.
    """
    failures = group_size - successes
    w_succ = torch.full_like(successes, 1.0 / group_size, dtype=torch.float64)
    if order < 2 or group_size < 2:
        return w_succ, torch.zeros_like(w_succ)

    failures = failures.to(torch.float64)
    limit = min(order, group_size)
    ratio = (failures - 1.0) / float(group_size - 1)
    total = torch.zeros_like(w_succ)
    for k in range(2, limit + 1):
        total = total + torch.where(failures >= k, ratio, torch.zeros_like(ratio))
        if k < limit:
            ratio = ratio * (failures - float(k)) / float(group_size - k)
    return w_succ, -(total / float(group_size))


def outcome_advantages(segment_returns, bar, args):
    """One scalar advantage per episode, from the group of num_envs episodes.

    The group is the whole batch: MuJoCo's initial-state distribution is narrow enough
    that all episodes share one effective prompt.
    """
    group_size = segment_returns.numel()
    if args.maxrl_estimator == "dense_maxrl":
        # Appendix M.4: log E[r] for non-negative rewards. Shift into the positive orthant
        # about the group minimum, since MuJoCo returns are signed and E[r] must exceed 0.
        rewards = segment_returns - segment_returns.min() + 1.0
    else:
        rewards = (segment_returns > bar).float()

    mean = rewards.mean()
    if args.maxrl_estimator in ("maxrl", "dense_maxrl"):
        advantages = (rewards - mean) / (mean + GROUP_EPS)
    elif args.maxrl_estimator == "grpo":
        advantages = (rewards - mean) / (rewards.std(unbiased=False) + GROUP_EPS)
    elif args.maxrl_estimator == "rloo":
        advantages = rewards - mean
    elif args.maxrl_estimator == "maclaurin":
        successes = rewards.sum()
        w_succ, w_fail = maclaurin_weights(successes, group_size, args.maxrl_order)
        advantages = (w_succ * rewards + w_fail * (1.0 - rewards)).to(segment_returns.dtype)
    else:
        raise ValueError(f"unknown estimator {args.maxrl_estimator}")

    diagnostics = torch.stack((
        bar, mean, rewards.sum(), advantages.std(),
        segment_returns.mean(), segment_returns.std(),
    ))
    return advantages, diagnostics


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, args):
    """Clipped surrogate with no value term; the outcome advantage carries all the signal."""
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
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
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
    if args.num_envs < 2:
        raise ValueError("an outcome advantage needs a group, so num_envs must exceed 1")
    horizon = episode_horizon(args.env_id)
    if horizon and args.num_steps != horizon:
        # A segment that is not one episode would give the group a truncated outcome and
        # silently break the "one response per prompt" correspondence the estimator needs.
        raise ValueError(f"num_steps must equal the {args.env_id} horizon of {horizon}")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id, args.num_envs, backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video, run_name=run_name,
    )


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=args.torch_deterministic,
                      matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    # Episodes must stay aligned across the group, so environments start together: there
    # is no staggering here, unlike the critic-based versions.
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
        writer.add_text("maxrl", f"critic-free outcome advantage; estimator={args.maxrl_estimator}; "
                                 f"T={args.maxrl_order}; group N={args.num_envs}; "
                                 f"bar={args.maxrl_bar_mode} q={args.maxrl_quantile}")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
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
        # gamma^t, so the outcome is the discounted return the policy actually optimizes.
        discounts = (args.gamma ** torch.arange(args.num_steps, device=device, dtype=torch.float32))
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
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        episode_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_logprobs = rollout_statistics(b_obs, b_native)
                # One outcome per environment: the discounted return of its episode.
                segment_returns = (batch.rewards * discounts[:, None]).sum(dim=0)
                if args.maxrl_bar_mode == "group_quantile":
                    bar = torch.quantile(segment_returns, args.maxrl_quantile)
                else:
                    batch_bar = torch.quantile(segment_returns, args.maxrl_quantile)
                    if bar_initialised:
                        success_bar.mul_(args.maxrl_tau_ema).add_(batch_bar, alpha=1.0 - args.maxrl_tau_ema)
                    else:
                        success_bar.copy_(batch_bar)
                        bar_initialised = True
                    bar = success_bar
                episode_advantages, maxrl_diagnostics = outcome_advantages(segment_returns, bar, args)
                # Broadcast the episode's advantage to each of its timesteps, exactly as the
                # reference broadcasts a sequence advantage across a response's tokens.
                b_advantages = episode_advantages.expand(args.num_steps, -1).flatten().clone()

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
                    "maxrl/success_bar", "maxrl/pass_rate", "maxrl/successes",
                    "maxrl/advantage_std", "maxrl/segment_return_mean", "maxrl/segment_return_std",
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

# State-dependent tanh-Gaussian PPO on the base PPO architecture/runtime.
# Hypothesis: shrinking noise alone stiffens the mean Fisher metric. Expressing
# the mean in the same fixed noise units removes that artificial scale mismatch.
# Compare noise_exponent=0, 0.5, 1 with/without whiten_mean at identical initial
# policies. This normalizes independent noise energy; it does NOT correlate it.
# Keep the true joint likelihood. Report exact post-update forward KL, separated
# into mean displacement and scale change; never divide the PPO ratio by width.
import json
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    save_model: bool = False
    noise_exponent: float = 0.5
    """component std = state-dependent sigma / action_dim**noise_exponent"""
    noise_sigma: float = 1.0
    """initial sigma before dimensional scaling; SAC's zero head gives exp(-1.5)"""
    whiten_mean: bool = True
    """express learned mean in initial component-noise units, not raw action units"""
    exact_kl_limit: float | None = None
    """optional full-rollout joint forward-KL constraint after each optimizer epoch"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads; two balances latency with concurrent N16 runs"""
    compile: bool = True
    """compile deterministic policy statistics, PPO loss and GAE"""
    compile_mode: str = "reduce-overhead"
    """PyTorch compilation mode for fixed-shape paths"""
    non_blocking_transfers: bool = False
    """opt into event-protected asynchronous pinned transfers"""
    staggered_starts: bool = True
    """stagger parallel environments; warmup counts toward total_timesteps"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def gaussian_logprob(sample, mean, log_std):
    return (-0.5 * ((sample - mean) * (-log_std).exp()).square()
            - log_std - 0.5 * LOG_TWO_PI).sum(-1)


def gaussian_kl_parts(old_mean, old_log_std, mean, log_std):
    """Exact KL(old || new), also exact after the shared invertible tanh map."""
    log_scale_ratio = old_log_std - log_std
    mean_kl = 0.5 * ((mean - old_mean) * (-log_std).exp()).square().sum(-1)
    scale_kl = (0.5 * torch.expm1(2.0 * log_scale_ratio) - log_scale_ratio).sum(-1)
    return mean_kl, scale_kl


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Gaussian PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Gaussian PPO requires finite, ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.noise_log_scale = -args.noise_exponent * math.log(self.action_dim)
        self.mean_scale = args.noise_sigma * math.exp(self.noise_log_scale) if args.whiten_mean else 1.0
        self.std_bias = math.atanh((math.log(args.noise_sigma) + 1.5) / 3.5)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            # Zero outputs make the initial policy identical between raw and
            # whitened parameterizations, including state-independent std.
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def policy_parameters(self, x):
        mean, raw_std = self.actor(x).chunk(2, dim=-1)
        log_std = -1.5 + 3.5 * torch.tanh(raw_std + self.std_bias) + self.noise_log_scale
        return mean * self.mean_scale, log_std

    def get_policy_and_value(self, x):
        mean, log_std = self.policy_parameters(x)
        return mean, log_std, self.critic(x)

    def action_logprob(self, mean, log_std, native):
        # Training retains pre-tanh samples. The common Jacobian cancels from
        # old/new ratios, so omit it on BOTH sides to avoid cancellation error.
        return gaussian_logprob(native, mean, log_std)

    def physical_logprob(self, mean, log_std, native):
        log_jacobian = 2.0 * (math.log(2.0) - native - F.softplus(-2.0 * native))
        return gaussian_logprob(native, mean, log_std) - (log_jacobian + self.log_action_scale).sum(-1)


class HostGaussianSampler:
    """Reusable FP32 native/action buffers; stage both before the next call."""
    def __init__(self, agent, num_envs):
        self.mean_scale = np.float32(agent.mean_scale)
        self.std_bias = np.float32(agent.std_bias)
        self.noise_log_scale = np.float32(agent.noise_log_scale)
        self.scale = agent.action_scale.cpu().numpy().copy()
        self.bias = agent.action_bias.cpu().numpy().copy()
        shape = (num_envs, agent.action_dim)
        self.std = np.empty(shape, dtype=np.float32)
        self.mean = np.empty(shape, dtype=np.float32)
        self.log_std = np.empty(shape, dtype=np.float32)
        self.native = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)

    def __call__(self, logits, rng):
        mean, raw_std = np.split(logits, 2, axis=-1)
        np.add(raw_std, self.std_bias, out=self.log_std)
        np.tanh(self.log_std, out=self.log_std)
        self.log_std *= np.float32(3.5)
        self.log_std += np.float32(-1.5)
        self.log_std += self.noise_log_scale
        np.exp(self.log_std, out=self.std)
        rng.standard_normal(self.native.shape, dtype=np.float32, out=self.native)
        self.native *= self.std
        np.multiply(mean, self.mean_scale, out=self.mean)
        self.native += self.mean
        np.tanh(self.native, out=self.action)
        self.action *= self.scale
        self.action += self.bias
        return self.native, self.action


@torch.no_grad()
def constrain_epoch(actor, before, constraint, limit):
    """Backtrack an Adam epoch proposal against exact full-rollout joint KL.

    This is a trust-region acceptance rule, not a claim that PPO clipping bounds
    KL. The previous epoch is feasible; evaluate the nonlinear interpolated
    network rather than assuming KL is quadratic. Adam moments retain the
    proposal's gradient history (as in projected optimization).
    """
    proposed = tuple(p.detach().clone() for p in actor.parameters())
    fraction = 1.0
    for _ in range(20):
        kl = constraint()
        # One deliberate host sync per acceptance check, outside optimizer steps.
        if bool(torch.isfinite(kl) & (kl <= limit)):
            return fraction
        fraction *= 0.5
        for p, start, end in zip(actor.parameters(), before, proposed):
            p.copy_(torch.lerp(start, end, fraction))
    for p, start in zip(actor.parameters(), before):
        p.copy_(start)
    restored_kl = constraint()
    if not bool(torch.isfinite(restored_kl) & (restored_kl <= limit)):
        raise RuntimeError("pre-epoch actor is not feasible under the actual behavior KL constraint")
    return 0.0


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args):
    """Clipped PPO using the true joint Gaussian ratio in pre-tanh coordinates."""
    mean, log_std, newvalue = agent.get_policy_and_value(observations)
    newlogprob = agent.action_logprob(mean, log_std, native_actions)
    gaussian_entropy = (log_std + 0.5 * (1.0 + LOG_TWO_PI)).sum(-1)
    if args.ent_coef:
        sample = mean + log_std.exp() * torch.randn_like(mean)
        entropy = -agent.physical_logprob(mean, log_std, sample)
    else:
        entropy = gaussian_entropy
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
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), gaussian_entropy.mean().detach(),
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
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    if args.noise_exponent not in (0.0, 0.5, 1.0):
        raise ValueError("noise_exponent must be 0, 0.5, or 1")
    if not math.exp(LOG_STD_MIN) < args.noise_sigma < math.exp(LOG_STD_MAX):
        raise ValueError("noise_sigma must be strictly inside SAC's std bounds")
    if args.exact_kl_limit is not None and (not math.isfinite(args.exact_kl_limit) or args.exact_kl_limit <= 0):
        raise ValueError("exact_kl_limit must be finite and positive")
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
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}"
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    metric_file = resources.enter_context(open(f"runs/{run_name}/geometry.jsonl", "w"))
    with open(f"runs/{run_name}/config.json", "w") as config:
        json.dump(vars(args), config, indent=2)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "Tanh Gaussian; state-dependent SAC logstd; scaled noise; true joint ratios; FP32 host rollout")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations, native, old_mean, old_log_std):
            """Score the ACTUAL host behavior; measure replay drift without erasing it."""
            mean, log_std, value = agent.get_policy_and_value(observations)
            old_logprob = agent.action_logprob(old_mean, old_log_std, native)
            replay_logprob = agent.action_logprob(mean, log_std, native)
            mean_kl, scale_kl = gaussian_kl_parts(old_mean, old_log_std, mean, log_std)
            drift = torch.stack(((mean_kl + scale_kl).mean(),
                                 (replay_logprob - old_logprob).abs().max()))
            return value.flatten(), old_logprob, drift

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        def geometry_model(observations, old_mean, old_log_std):
            mean, log_std = agent.policy_parameters(observations)
            mean_kl, scale_kl = gaussian_kl_parts(old_mean, old_log_std, mean, log_std)
            joint = mean_kl + scale_kl
            return torch.stack((joint.mean(), joint.max(), mean_kl.mean(), scale_kl.mean(),
                                log_std.exp().mean(), log_std.exp().square().sum(-1).sqrt().mean(),
                                (mean - old_mean).square().sum(-1).sqrt().mean(),
                                (log_std - old_log_std).mean()))

        if args.compile:
            geometry_model = torch.compile(geometry_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        sampler = np.random.default_rng(args.seed)
        sample_actions = HostGaussianSampler(agent, args.num_envs)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,),
                                           "old_mean": (agent.action_dim,), "old_log_std": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=warmup_action, horizon=horizon,
                                    phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
            bootstraps.reset()
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native,
                                  old_mean=sample_actions.mean, old_log_std=sample_actions.log_std)
                global_step += args.num_envs
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_mean = batch.fields["old_mean"].flatten(0, 1)
                b_log_std = batch.fields["old_log_std"].flatten(0, 1)
                b_values, b_logprobs, replay_drift = rollout_statistics(b_obs, b_native, b_mean, b_log_std)
                # CUDA graph outputs are borrowed; freeze statistics until the next rollout.
                b_values, b_logprobs = b_values.clone(), b_logprobs.clone()
                replay_drift = replay_drift.clone()
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
            updates = 0
            accepted_fraction = 1.0

            def full_rollout_kl():
                return geometry_model(b_obs, b_mean, b_log_std)[0]

            with timer.span("update"):
                for _ in range(args.update_epochs):
                    before = (tuple(p.detach().clone() for p in agent.actor.parameters())
                              if args.exact_kl_limit is not None else ())
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    if args.exact_kl_limit is not None:
                        accepted_fraction = constrain_epoch(agent.actor, before, full_rollout_kl, args.exact_kl_limit)
                        if accepted_fraction < 1.0:
                            break
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            with torch.no_grad():
                geometry = geometry_model(b_obs, b_mean, b_log_std)
            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/gaussian_entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "geometry/joint_kl": geometry[0], "geometry/max_state_kl": geometry[1],
                "geometry/mean_kl": geometry[2], "geometry/scale_kl": geometry[3],
                "geometry/component_std": geometry[4], "geometry/noise_rms_norm": geometry[5],
                "geometry/mean_displacement": geometry[6], "geometry/log_std_change": geometry[7],
                "geometry/replay_kl": replay_drift[0],
                "geometry/replay_max_abs_logratio": replay_drift[1],
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            logged["geometry/accepted_fraction"] = accepted_fraction
            logged["geometry/optimizer_steps"] = updates
            metric_file.write(json.dumps({"step": global_step, **logged}) + "\n")
            metric_file.flush()
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

        transfer.close()
        if args.save_model:
            torch.save({"state_dict": agent.state_dict(), "args": vars(args)},
                       f"runs/{run_name}/{args.exp_name}.cleanrl_model")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

# SiTU-GLU tanh-Gaussian PPO: calibrated low noise and optional exact epoch trust.
import hashlib
import json
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.host_actor import make_situ_sphere_trunk
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
    initial_std: float = 0.610376541075546
    """initial pre-tanh component std, strictly within exp([-5, 2])"""
    whiten_mean: bool = False
    """express mean logits in fixed initial_std units; never dimension-scale noise"""
    exact_kl_limit: float = 0.0
    """full-rollout mean KL(old || new) cap; zero disables epoch backtracking"""

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


def state_hash(module):
    """Stable state fingerprint, including names, dtypes, shapes and FP32 bytes."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


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
        self.register_buffer("action_span", self.action_high - self.action_low)
        if not torch.isfinite(self.action_span).all() or not (self.action_span > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("action_scale", self.action_span / 2.0)
        self.register_buffer("action_bias", self.action_low + self.action_scale)
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.mean_scale = args.initial_std if args.whiten_mean else 1.0
        self.std_bias = math.atanh((math.log(args.initial_std) + 1.5) / 3.5)
        # Construct critic FIRST, identically to the frozen v2 Gaussian control.
        self.critic = nn.Sequential(
            make_situ_sphere_trunk(observation_dim, 64, n_blocks=3),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            make_situ_sphere_trunk(observation_dim, 64, n_blocks=3),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def policy_parameters(self, x):
        logits = self.actor(x)
        mean, raw_std = logits.chunk(2, dim=-1)
        return mean * self.mean_scale, -1.5 + 3.5 * torch.tanh(raw_std + self.std_bias)

    def get_policy_and_value(self, x):
        first, second = self.policy_parameters(x)
        return first, second, self.critic(x)

    def action_logprob(self, first, second, native):
        # Score in native coordinates on BOTH sides: the common invertible
        # action Jacobian cancels exactly from PPO ratios and joint KL.
        return gaussian_logprob(native, first, second)

    def physical_logprob(self, first, second, native):
        log_jacobian = 2.0 * (math.log(2.0) - native - F.softplus(-2.0 * native))
        return self.action_logprob(first, second, native) - (log_jacobian + self.log_action_scale).sum(-1)

    def joint_kl(self, old_first, old_second, first, second):
        mean_kl, scale_kl = gaussian_kl_parts(old_first, old_second, first, second)
        return mean_kl + scale_kl

    def native_entropy(self, first, second):
        return (second + 0.5 * (1.0 + LOG_TWO_PI)).sum(-1)


class HostGaussianSampler:
    """Reusable FP32 law/native/action buffers; stage before the next sample."""
    def __init__(self, agent, num_envs):
        self.mean_scale = np.float32(agent.mean_scale)
        self.std_bias = np.float32(agent.std_bias)
        self.scale = agent.action_scale.cpu().numpy().copy()
        self.bias = agent.action_bias.cpu().numpy().copy()
        shape = (num_envs, agent.action_dim)
        self.std = np.empty(shape, dtype=np.float32)
        self.first = np.empty(shape, dtype=np.float32)
        self.second = np.empty(shape, dtype=np.float32)
        self.native = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)

    def __call__(self, logits, rng):
        mean, raw_std = np.split(logits, 2, axis=-1)
        np.multiply(mean, self.mean_scale, out=self.first)
        np.add(raw_std, self.std_bias, out=self.second)
        np.tanh(self.second, out=self.second)
        self.second *= np.float32(3.5)
        self.second += np.float32(-1.5)
        np.exp(self.second, out=self.std)
        rng.standard_normal(self.native.shape, dtype=np.float32, out=self.native)
        self.native *= self.std
        self.native += self.first
        np.tanh(self.native, out=self.action)
        self.action *= self.scale
        self.action += self.bias
        return self.native, self.action


def rollout_statistics(agent, observations, native, old_first, old_second):
    """Score the ACTUAL host law; replay is only a drift diagnostic, never a refresh."""
    first, second, value = agent.get_policy_and_value(observations)
    old_logprob = agent.action_logprob(old_first, old_second, native)
    logratio = agent.action_logprob(first, second, native) - old_logprob
    joint = agent.joint_kl(old_first, old_second, first, second)
    drift = torch.stack((joint.mean(), logratio.abs().max(), logratio.mean(), logratio.square().mean().sqrt()))
    return value.flatten(), old_logprob, drift


@torch.no_grad()
def snapshot_actor_epoch(actor, optimizer):
    """Save only actor parameters and Adam state; critic updates are never undone."""
    parameters = tuple(p.detach().clone() for p in actor.parameters())
    states = tuple({key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in optimizer.state.get(p, {}).items()}
                   for p in actor.parameters())
    return parameters, states


@torch.no_grad()
def constrain_epoch(actor, optimizer, before, constraint, limit):
    """Backtrack against exact KL from the captured full-rollout behavior law.

    The constraint is a mean over rollout states, not a per-state KL bound.
    Check the nonlinear network at every interpolation; clipping is not a bound.
    For a full proposal, keep Adam state. For partial acceptance or rejection,
    restore pre-epoch actor moments AND step counters (including absent lazy
    state): gradients of the abandoned proposal do not influence the next epoch.
    A partial parameter step with pre-epoch moments is deliberately not an Adam
    step. Critic parameters and its optimizer history always retain their update.
    Synchronization occurs only at epoch acceptance checks, never minibatches.
    """
    parameters, states = before
    proposed = tuple(p.detach().clone() for p in actor.parameters())
    fraction = 1.0
    for _ in range(20):
        kl = constraint()
        if bool(torch.isfinite(kl) & (kl <= limit)):
            break
        fraction *= 0.5
        for p, start, end in zip(actor.parameters(), parameters, proposed):
            p.copy_(torch.lerp(start, end, fraction))
    else:
        fraction = 0.0
        for p, start in zip(actor.parameters(), parameters):
            p.copy_(start)
    if fraction < 1.0:
        for p, state in zip(actor.parameters(), states):
            if state:
                current = optimizer.state[p]
                for key, value in state.items():
                    if torch.is_tensor(value):
                        current[key].copy_(value)
                    else:
                        current[key] = value
            else:
                optimizer.state.pop(p, None)
    if fraction == 0.0:
        restored_kl = constraint()
        if not bool(torch.isfinite(restored_kl) & (restored_kl <= limit)):
            raise RuntimeError("pre-epoch actor is infeasible under actual behavior KL")
    return fraction


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args):
    """True joint-ratio PPO; entropy disabled for the controlled Gaussian comparison."""
    first, second, newvalue = agent.get_policy_and_value(observations)
    newlogprob = agent.action_logprob(first, second, native_actions)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        native_entropy = agent.native_entropy(first, second).mean()
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
    loss = pg_loss + v_loss * args.vf_coef
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), native_entropy,
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
    if not args.compile:
        raise ValueError("the low-noise experiment requires compiled training")
    if not math.exp(LOG_STD_MIN) < args.initial_std < math.exp(LOG_STD_MAX):
        raise ValueError("initial_std must be strictly inside exp([-5, 2])")
    if not math.isfinite(args.exact_kl_limit) or args.exact_kl_limit < 0:
        raise ValueError("exact_kl_limit must be finite and nonnegative")
    if args.ent_coef != 0.0:
        raise ValueError("the low-noise experiment requires ent_coef=0")
    return args


def provenance(agent, args):
    return {
        "config": vars(args).copy(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "critic_sha256": state_hash(agent.critic),
        "critic_trunk_sha256": state_hash(agent.critic[0]),
        "actor_trunk_sha256": state_hash(agent.actor[0]),
        "actor_sha256": state_hash(agent.actor),
        "actor_parameter_count": sum(p.numel() for p in agent.actor.parameters()),
        "critic_parameter_count": sum(p.numel() for p in agent.critic.parameters()),
        "architecture": {"trunk": "SiTUSphereTrunk", "width": 64, "n_blocks": 3,
                         "critic_first": True, "actor_head_outputs": 2 * agent.action_dim,
                         "actor_head_gain": 0.0, "critic_head_gain": 1.0},
        "calibration": {"initial_std": args.initial_std, "mean_scale": agent.mean_scale,
                        "std_bias": agent.std_bias, "log_std_bounds": [LOG_STD_MIN, LOG_STD_MAX],
                        "dimension_scaling": False},
        "likelihood": "true joint native density; common physical Jacobian cancels in PPO ratio",
        "exact_kl": "full-rollout mean continuous old-to-new KL, not rounded atoms or a per-state bound",
        "trust_optimizer": "partial/rejected epochs restore actor Adam moments and counters; retain critic updates",
        "entropy": "ent_coef=0 required; logged entropy is in pre-tanh coordinates",
        "runtime": {"learner": "CUDA FP32 compiled", "host_mirror": "FP32",
                    "matmul_precision": "highest", "allow_tf32": False},
    }


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
    metric_file = resources.enter_context(open(f"runs/{run_name}/metrics.jsonl", "w"))
    with open(f"runs/{run_name}/config.json", "w") as config:
        json.dump(vars(args), config, indent=2)
    Path(f"runs/{run_name}/source.py").write_bytes(Path(__file__).read_bytes())
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "Gaussian; SiTU sphere; calibrated noise; true joint ratios; FP32 host rollout")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args)
        with open(f"runs/{run_name}/provenance.json", "w") as output:
            initial_provenance = provenance(agent, args)
            json.dump(initial_provenance, output, indent=2)
        agent = agent.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def statistics_model(observations, native, old_first, old_second):
            return rollout_statistics(agent, observations, native, old_first, old_second)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        def geometry_model(observations, old_first, old_second):
            first, second = agent.policy_parameters(observations)
            mean_kl, scale_kl = gaussian_kl_parts(old_first, old_second, first, second)
            joint = mean_kl + scale_kl
            return torch.stack((joint.mean(), joint.max(), mean_kl.mean(), scale_kl.mean(),
                                second.exp().mean(), second.exp().square().sum(-1).sqrt().mean(),
                                (first - old_first).square().sum(-1).sqrt().mean(),
                                (second - old_second).mean()))

        if args.compile:
            geometry_model = torch.compile(geometry_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            statistics_model = graph_compile(statistics_model)
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
                                           "old_first": (agent.action_dim,), "old_second": (agent.action_dim,)})
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
            episode_returns, episode_lengths = [], []
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
                                  old_first=sample_actions.first, old_second=sample_actions.second)
                global_step += args.num_envs
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        episode_returns.append(episode_return)
                        episode_lengths.append(float(info["episode"]["l"]))
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_first = batch.fields["old_first"].flatten(0, 1)
                b_second = batch.fields["old_second"].flatten(0, 1)
                b_values, b_logprobs, replay_drift = statistics_model(b_obs, b_native, b_first, b_second)
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

            @torch.no_grad()
            def full_rollout_kl():
                return geometry_model(b_obs, b_first, b_second)[0]

            with timer.span("update"):
                for _ in range(args.update_epochs):
                    before = snapshot_actor_epoch(agent.actor, optimizer) if args.exact_kl_limit else None
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
                    if args.exact_kl_limit:
                        accepted_fraction = constrain_epoch(
                            agent.actor, optimizer, before, full_rollout_kl, args.exact_kl_limit)
                        if accepted_fraction < 1.0:
                            break

            with torch.no_grad():
                geometry = geometry_model(b_obs, b_first, b_second)
            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/native_entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "geometry/joint_kl": geometry[0], "geometry/max_state_kl": geometry[1],
                "geometry/mean_kl": geometry[2], "geometry/scale_kl": geometry[3],
                "geometry/component_std": geometry[4], "geometry/noise_rms_norm": geometry[5],
                "geometry/mean_displacement": geometry[6], "geometry/log_std_change": geometry[7],
                "geometry/replay_kl": replay_drift[0],
                "geometry/replay_max_abs_logratio": replay_drift[1],
                "geometry/replay_mean_logratio": replay_drift[2],
                "geometry/replay_rms_logratio": replay_drift[3],
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            logged["geometry/accepted_fraction"] = accepted_fraction
            logged["geometry/optimizer_steps"] = updates
            now = time.perf_counter()
            logged["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
            logged["charts/SPS"] = int(global_step / (now - start_time))
            logged["charts/interval_SPS"] = (global_step - interval_step) / (now - interval_start)
            for phase, timing in timer.summary().items():
                logged[f"timing/{phase}_s"] = timing["total_s"]
            metric_file.write(json.dumps({"step": global_step, "iteration": iteration,
                                         "policy": "gaussian", "episode_returns": episode_returns,
                                         "episode_lengths": episode_lengths, **logged}) + "\n")
            metric_file.flush()
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        transfer.close()
        if args.save_model:
            # Normalizers expose arrays, not state_dict(). Preserve every row;
            # restoring their arrays must copy in-place (native pointers own them).
            torch.save({"state_dict": agent.state_dict(), "args": vars(args),
                        "optimizer": optimizer.state_dict(), "global_step": global_step,
                        "provenance": initial_provenance,
                        "obs_norm": {"means": torch.from_numpy(obs_norm.means.copy()),
                                     "variances": torch.from_numpy(obs_norm.variances.copy()),
                                     "counts": torch.from_numpy(obs_norm.counts.copy()),
                                     "epsilon": obs_norm.epsilon, "clip": obs_norm.clip},
                        "rew_norm": {"means": torch.from_numpy(rew_norm.means.copy()),
                                     "variances": torch.from_numpy(rew_norm.variances.copy()),
                                     "counts": torch.from_numpy(rew_norm.counts.copy()),
                                     "returns": torch.from_numpy(rew_norm.returns.copy()),
                                     "gamma": rew_norm.gamma, "epsilon": rew_norm.epsilon,
                                     "clip": rew_norm.clip}},
                       f"runs/{run_name}/{args.exp_name}.cleanrl_model")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

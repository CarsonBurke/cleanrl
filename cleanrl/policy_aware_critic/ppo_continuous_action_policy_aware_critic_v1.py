# Policy-aware critic PPO v1. Base: ppo_continuous_action_normres_indclip_v4_gaussian_50M_v2.
# The critic is V(s, c): c is a frozen, rollout-time descriptor of the policy at
# s, "what I believed and would have done here": the actor's RMS-normalized
# trunk latent plus tanh(mean) and log_std of its Gaussian. Because c labels the
# policy that generated each trajectory, the last K rollouts all stay on-policy
# for this critic: their lambda-returns are recomputed every iteration with the
# current critic evaluated at each state's own (s, c), with no importance weights
# and no lag bias. Within one policy c = f(s) adds nothing; the value of c comes
# from generalizing across policies, so a fresh rollout's c (after an actor
# update) predicts its value instead of the stale average. Stored observations
# and rewards are re-expressed in the current normalizer statistics each iteration.
# Hypothesis: lower value lag (higher pre-fit EV on fresh data) and K-times more
# critic data give better advantages -> higher return than V(s) with or without
# the same replay. Controls: --context none (same replay), --context none
# --replay-rollouts 1 (the base).
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    device_minibatches,
    explained_variance,
    gather_metrics,
    get_gae_fn,
)
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets,
    episode_horizon,
    run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
LOG_TWO_PI = math.log(2.0 * math.pi)
MATCHED_ACTION_VARIANCE = 1.0 / (2.0 * (1.0 + math.log(2.0)) + 1.0)
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
TRUNK_WIDTH = 64


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """Gaussian-control learning rate"""
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
    """actor minibatches per epoch; the critic takes as many, each K times larger"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """normalize advantages as in the Gaussian control"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """toggle PPO value-loss clipping independently of gradient clipping"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum gradient norm, applied independently to actor and critic"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    reward_norm: bool = True
    """normalize/clip rewards before GAE; never normalize GAE return targets"""

    context: Literal["latent_dist", "dist", "latent", "none"] = "latent_dist"
    """policy descriptor fed to the critic; none is the V(s) control"""
    replay_rollouts: int = 4
    """K most recent rollouts the critic regresses on, targets recomputed each iteration"""

    placement: Literal["pre", "post"] = "pre"
    """normalize branch inputs with an identity stream, or residual outputs"""
    norm_kind: Literal["layer", "rms"] = "rms"
    """non-affine centered LayerNorm or uncentered RMSNorm, epsilon 1e-5"""
    activation: Literal["lrelusq", "stiglu"] = "stiglu"
    """squared leaky-ReLU pair or parameter-matched SiTU-GLU branch"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads, capped at num_envs"""
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


@lru_cache(maxsize=1)
def matched_gaussian_std():
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    nodes *= math.sqrt(2.0)
    weights /= math.sqrt(math.pi)
    low, high = math.exp(LOG_STD_MIN), math.exp(LOG_STD_MAX)
    for _ in range(80):
        middle = (low + high) / 2.0
        if np.dot(weights, np.tanh(middle * nodes) ** 2) < MATCHED_ACTION_VARIANCE:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def context_dim(kind, action_dim):
    return {"latent_dist": TRUNK_WIDTH + 2 * action_dim, "dist": 2 * action_dim,
            "latent": TRUNK_WIDTH, "none": 0}[kind]


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    action_bias: torch.Tensor
    log_action_scale: torch.Tensor

    def __init__(self, envs, *, context="latent_dist", placement="pre", norm_kind="rms", activation="stiglu"):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Gaussian PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Gaussian PPO requires finite, strictly ordered action bounds")
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
        matched_std = matched_gaussian_std()
        self.std_bias = math.atanh((math.log(matched_std) + 1.5) / 3.5)
        self.context = context
        self.context_dim = context_dim(context, self.action_dim)
        self.critic = nn.Sequential(
            make_norm_residual_trunk(
                observation_dim + self.context_dim, TRUNK_WIDTH,
                placement=placement, norm_kind=norm_kind, activation=activation,
            ),
            layer_init(nn.Linear(TRUNK_WIDTH, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            make_norm_residual_trunk(
                observation_dim, TRUNK_WIDTH, placement=placement, norm_kind=norm_kind, activation=activation
            ),
            layer_init(nn.Linear(TRUNK_WIDTH, 2 * self.action_dim), std=0.0),
        )

    def policy(self, x):
        """Gaussian parameters and the critic's policy descriptor (always detached)."""
        latent = self.actor[0](x)
        mean, raw_std = self.actor[1](latent).chunk(2, dim=-1)
        log_std = -1.5 + 3.5 * torch.tanh(raw_std + self.std_bias)
        parts = []
        if self.context in ("latent_dist", "latent"):
            # The pre-RMS residual stream's scale drifts; its direction is the belief.
            parts.append(F.rms_norm(latent, (TRUNK_WIDTH,), eps=1e-5))
        if self.context in ("latent_dist", "dist"):
            # tanh(mean) is the physical-unit intent; log_std its commitment.
            parts += [torch.tanh(mean), log_std]
        context = torch.cat(parts, dim=-1).detach() if parts else x[..., :0]
        return mean, log_std, context

    def value(self, x, context):
        if self.context_dim:
            x = torch.cat((x, context), dim=-1)
        return self.critic(x).flatten()

    def action_logprob(self, mean, log_std, native_action):
        return gaussian_logprob(native_action, mean, log_std)


class HostGaussianSampler:
    """Reusable FP32 Gaussian/native/action buffers for the host rollout."""
    def __init__(self, agent, num_envs):
        self.std_bias = np.float32(agent.std_bias)
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
        np.copyto(self.mean, mean)
        np.add(raw_std, self.std_bias, out=self.log_std)
        np.tanh(self.log_std, out=self.log_std)
        self.log_std *= np.float32(3.5)
        self.log_std += np.float32(-1.5)
        np.exp(self.log_std, out=self.std)
        rng.standard_normal(self.native.shape, dtype=np.float32, out=self.native)
        self.native *= self.std
        self.native += self.mean
        np.tanh(self.native, out=self.action)
        self.action *= self.scale
        self.action += self.bias
        return self.native, self.action


class PolicyReplay:
    """Ring of the last K rollouts, laid out (T, K, N, ...) so GAE sees (T, K*N).

    Slots keep raw observations and rewards, so ``refresh`` normalizes every
    slot with the current per-env statistics (the current slot included), and
    the frozen policy contexts computed from the as-acted normalized inputs.
    """

    def __init__(self, k, num_steps, num_envs, obs_dim, ctx_dim, device):
        self.k = k
        shape = (num_steps, k, num_envs)
        self.raw_obs = torch.zeros(shape + (obs_dim,), device=device)
        self.raw_next_obs = torch.zeros_like(self.raw_obs)
        self.ctx = torch.zeros(shape + (ctx_dim,), device=device)
        self.next_ctx = torch.zeros_like(self.ctx)
        self.raw_rewards = torch.zeros(shape, device=device)
        self.terminations = torch.zeros(shape, device=device)
        self.truncations = torch.zeros(shape, device=device)
        self.cursor = 0
        self.filled = False

    def insert(self, raw_obs, raw_next_obs, ctx, next_ctx, raw_rewards, terminations, truncations):
        """Write one (T, N, ...) rollout; the first one fills every slot."""
        slots = range(self.k) if not self.filled else (self.cursor,)
        for slot in slots:
            self.raw_obs[:, slot].copy_(raw_obs)
            self.raw_next_obs[:, slot].copy_(raw_next_obs)
            self.ctx[:, slot].copy_(ctx)
            self.next_ctx[:, slot].copy_(next_ctx)
            self.raw_rewards[:, slot].copy_(raw_rewards)
            self.terminations[:, slot].copy_(terminations)
            self.truncations[:, slot].copy_(truncations)
        current = self.cursor
        self.cursor = (self.cursor + 1) % self.k
        self.filled = True
        return current

    def refresh(self, obs_mean, obs_std, reward_std, obs_clip, reward_clip):
        """Every slot's observations and rewards in the current normalizer units.

        Stats are per env (N, ...), broadcast over time and slots, matching the
        VectorObsNorm / VectorRewardNorm output formulas.
        """
        obs = ((self.raw_obs - obs_mean) / obs_std).clamp_(-obs_clip, obs_clip)
        next_obs = ((self.raw_next_obs - obs_mean) / obs_std).clamp_(-obs_clip, obs_clip)
        rewards = (self.raw_rewards / reward_std).clamp_(-reward_clip, reward_clip)
        return obs, next_obs, rewards


def clip_gradients(actor_parameters, critic_parameters, max_grad_norm):
    """Clip each network independently after the weighted joint loss backward."""
    actor_preclip_norm = nn.utils.clip_grad_norm_(actor_parameters, max_grad_norm, foreach=True)
    critic_preclip_norm = nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm, foreach=True)
    return actor_preclip_norm, critic_preclip_norm


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages,
             critic_obs, critic_ctx, targets, old_values, args):
    """Clipped PPO on the current rollout; clipped value MSE on the policy replay."""
    mean, log_std, _ = agent.policy(observations)
    newlogprob = agent.action_logprob(mean, log_std, native_actions)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        entropy = (log_std + 0.5 * (1.0 + LOG_TWO_PI)).sum(-1).mean()
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    newvalue = agent.value(critic_obs, critic_ctx)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - targets) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - targets) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - targets) ** 2).mean()
    loss = pg_loss + v_loss * args.vf_coef
    metrics = torch.stack(
        (pg_loss.detach(), v_loss.detach(), entropy, old_approx_kl, approx_kl, clipfrac)
    )
    return loss, metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs, args.replay_rollouts) <= 0:
        raise ValueError("environment, rollout, minibatch, epoch and replay counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0 or args.batch_size % args.minibatch_size:
        raise ValueError("num_minibatches must divide batch_size")
    if args.norm_adv and args.minibatch_size < 2:
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id,
        args.num_envs,
        backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video,
        run_name=run_name,
    )


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
        )
        writer.add_text(
            "policy",
            f"Matched tanh-Gaussian host actor; policy-aware critic V(s, c={args.context}) over the last "
            f"{args.replay_rollouts} rollouts with per-iteration lambda-return recomputation",
        )
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, context=args.context, placement=args.placement, norm_kind=args.norm_kind,
                      activation=args.activation).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        actor_parameters = tuple(agent.actor.parameters())
        critic_parameters = tuple(agent.critic.parameters())
        obs_shape = envs.single_observation_space.shape
        obs_dim = int(np.prod(obs_shape))
        replay = PolicyReplay(args.replay_rollouts, args.num_steps, args.num_envs, obs_dim,
                              agent.context_dim, device)
        replay_size = args.replay_rollouts * args.batch_size
        critic_minibatch_size = args.replay_rollouts * args.minibatch_size

        def rollout_statistics(observations, next_observations, native, old_mean, old_log_std):
            """Behavior log-probabilities and frozen policy contexts, one actor forward."""
            _, _, context = agent.policy(torch.cat((observations, next_observations)))
            old_logprob = agent.action_logprob(old_mean, old_log_std, native)
            return old_logprob, *context.chunk(2)

        def replay_values(observations, contexts):
            return agent.value(observations, contexts)

        def context_shift(observations, contexts):
            """V(s, c_now(s)) - V(s, c_stored): the critic's predicted gain of the current policy."""
            _, _, now = agent.policy(observations)
            return agent.value(observations, now) - agent.value(observations, contexts)

        def loss_model(observations, native, old_logprobs, advantages, critic_obs, critic_ctx, targets, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages,
                            critic_obs, critic_ctx, targets, old_values, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            replay_values = graph_compile(replay_values)
            context_shift = graph_compile(context_shift)
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode, explicit_next_values=True)
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

        transfer = RolloutTransfer(
            args.num_steps,
            args.num_envs,
            obs_shape,
            device,
            store_transition_observations=True,
            non_blocking=args.non_blocking_transfers,
            fields={
                "observations": obs_shape,
                "native_actions": (agent.action_dim,),
                "old_mean": (agent.action_dim,),
                "old_log_std": (agent.action_dim,),
                "raw_observations": obs_shape,
                "raw_transitions": obs_shape,
                "raw_rewards": (1,),
            },
        )
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma) if args.reward_norm else None

        obs_clip = math.inf if obs_norm.clip is None else obs_norm.clip
        reward_clip = math.inf if rew_norm is None or rew_norm.clip is None else rew_norm.clip
        # Per-env normalizer statistics, staged in pinned memory and copied
        # asynchronously; the next overwrite follows a full rollout and a sync.
        stats_host = torch.empty((3, args.num_envs, obs_dim), pin_memory=True)
        stats_device = torch.empty_like(stats_host, device=device)
        stats_view = stats_host.numpy()

        def normalizer_state():
            """Current per-env (obs mean, obs std, reward std) on device."""
            stats_view[0] = obs_norm.means.reshape(args.num_envs, obs_dim)
            np.sqrt(obs_norm.variances.reshape(args.num_envs, obs_dim) + obs_norm.epsilon, out=stats_view[1],
                    casting="same_kind")
            stats_view[2] = 1.0 if rew_norm is None else np.sqrt(rew_norm.variances + rew_norm.epsilon)[:, None]
            stats_device.copy_(stats_host, non_blocking=True)
            return stats_device[0], stats_device[1], stats_device[2, :, 0]

        def raw_transition(raw_next_obs, terms, truncs, infos):
            """Raw s_{t+1}: the pre-reset final observation at episode boundaries."""
            if not (np.count_nonzero(terms) or np.count_nonzero(truncs)):
                return raw_next_obs
            transition = np.array(raw_next_obs, dtype=np.float32, copy=True)
            finals = infos["final_observation"]
            for index in np.flatnonzero(np.logical_or(terms, truncs)):
                transition[index] = finals[index]
            return transition

        old_slot_index = [
            torch.tensor([slot for slot in range(args.replay_rollouts) if slot != current] or [current],
                         device=device)
            for current in range(args.replay_rollouts)
        ]

        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * args.num_minibatches
        update_metrics = torch.empty((max_updates, 6), device=device)
        grad_norms = torch.empty((max_updates, 2), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(
                envs,
                obs_norm=obs_norm,
                rew_norm=rew_norm,
                act_fn=warmup_action,
                horizon=horizon,
                phase_offsets=phases,
                seed=args.seed,
            )
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        # Warmup hides raw observations; its last normalization used the
        # current statistics, so inverting it is exact up to clipping.
        raw_obs_step = (next_obs_np * np.sqrt(obs_norm.variances + obs_norm.epsilon)
                        + obs_norm.means).astype(np.float32)
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step
        rollout_shape = (args.num_steps, args.num_envs)

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
                    reward = rew_norm.normalize(raw_reward, terms) if rew_norm is not None else raw_reward
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(
                        step, reward, terms, truncs, transition_obs, observations=obs_step, native_actions=native,
                        old_mean=sample_actions.mean, old_log_std=sample_actions.log_std,
                        raw_observations=raw_obs_step,
                        raw_transitions=raw_transition(raw_obs, terms, truncs, infos),
                        raw_rewards=np.reshape(raw_reward, (args.num_envs, 1)),
                    )
                    # The env may reuse its output buffer; keep our own copy.
                    np.copyto(raw_obs_step, raw_obs)
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
                b_next_obs = batch.transition_observations.flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_logprobs, b_ctx, b_next_ctx = rollout_statistics(
                    b_obs, b_next_obs, b_native,
                    batch.fields["old_mean"].flatten(0, 1), batch.fields["old_log_std"].flatten(0, 1),
                )
                obs_mean, obs_std, reward_std = normalizer_state()
                current = replay.insert(
                    batch.fields["raw_observations"], batch.fields["raw_transitions"],
                    b_ctx.view(*rollout_shape, -1), b_next_ctx.view(*rollout_shape, -1),
                    batch.fields["raw_rewards"].view(rollout_shape), batch.terminations, batch.truncations,
                )
                # Every slot is on-policy for V(., c): re-derive its lambda-returns
                # with the current critic in the current normalizer units.
                r_obs, r_next_obs, r_rewards = replay.refresh(obs_mean, obs_std, reward_std, obs_clip, reward_clip)
                r_obs = r_obs.flatten(0, 2)
                r_ctx = replay.ctx.flatten(0, 2)
                r_values = replay_values(r_obs, r_ctx)
                r_next_values = replay_values(r_next_obs.flatten(0, 2), replay.next_ctx.flatten(0, 2))
                wide = (args.num_steps, args.replay_rollouts * args.num_envs)
                advantages, returns = gae_fn(
                    r_rewards.view(wide),
                    r_values.view(wide),
                    replay.terminations.view(wide),
                    replay.truncations.view(wide),
                    r_next_values.view(wide),
                    args.gamma,
                    args.gae_lambda,
                )
                # reduce-overhead GAE outputs live in graph memory the update reuses.
                advantages, returns = advantages.clone(), returns.clone()
                r_returns = returns.flatten()
                by_slot = (args.num_steps, args.replay_rollouts, args.num_envs)
                b_advantages = advantages.view(by_slot)[:, current].flatten()
                b_values = r_values.view(by_slot)[:, current].flatten()
                b_returns = returns.view(by_slot)[:, current].flatten()
                old_slots = old_slot_index[current]
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    actor_batches = device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator)
                    critic_batches = device_minibatches(replay_size, critic_minibatch_size, device, shuffle_generator)
                    for indices, critic_indices in zip(actor_batches, critic_batches):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices],
                            b_native[indices],
                            b_logprobs[indices],
                            b_advantages[indices],
                            r_obs[critic_indices],
                            r_ctx[critic_indices],
                            r_returns[critic_indices],
                            r_values[critic_indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        actor_preclip_norm, critic_preclip_norm = clip_gradients(
                            actor_parameters,
                            critic_parameters,
                            args.max_grad_norm,
                        )
                        grad_norms[updates, 0].copy_(actor_preclip_norm)
                        grad_norms[updates, 1].copy_(critic_preclip_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            with timer.span("diagnostics"), torch.no_grad():
                shift = context_shift(r_obs, r_ctx).view(by_slot)
                current_shift = shift[:, current]
                # With K=1 the "old" slot is the current one (fresh-data EV again).
                old_ev = explained_variance(r_values.view(by_slot)[:, old_slots].flatten(),
                                            returns.view(by_slot)[:, old_slots].flatten())

            last = update_metrics[updates - 1]
            performed_grad_norms = grad_norms[:updates]
            mean_grad_norms = performed_grad_norms.mean(dim=0)
            grad_clip_fractions = (performed_grad_norms > args.max_grad_norm).float().mean(dim=0)
            logged = gather_metrics(
                {
                    "losses/policy_loss": last[0],
                    "losses/value_loss": last[1],
                    "losses/entropy": last[2],
                    "losses/old_approx_kl": last[3],
                    "losses/approx_kl": last[4],
                    "losses/clipfrac": update_metrics[:updates, 5].mean(),
                    # Pre-fit values on a fresh rollout: the value-lag measure.
                    "losses/explained_variance": explained_variance(b_values, b_returns),
                    "value/replay_explained_variance": old_ev,
                    "value/context_shift_current_mean": current_shift.mean(),
                    "value/context_shift_current_std": current_shift.std(),
                    "value/context_shift_replay_mean": shift[:, old_slots].mean(),
                    "value/return_std": b_returns.std(),
                    "grad/actor_preclip_norm": mean_grad_norms[0],
                    "grad/critic_preclip_norm": mean_grad_norms[1],
                    "grad/actor_clip_fraction": grad_clip_fractions[0],
                    "grad/critic_clip_fraction": grad_clip_fractions[1],
                }
            )
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name not in ("losses/explained_variance", "value/replay_explained_variance")):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar(
                "charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step
            )
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        transfer.close()
        envs.close()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(
                {
                    "model": agent.state_dict(),
                    "args": vars(args),
                    "obs_norm": {
                        "means": torch.from_numpy(obs_norm.means.copy()),
                        "variances": torch.from_numpy(obs_norm.variances.copy()),
                        "counts": torch.from_numpy(obs_norm.counts.copy()),
                        "epsilon": obs_norm.epsilon,
                        "clip": obs_norm.clip,
                    },
                },
                model_path,
            )
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

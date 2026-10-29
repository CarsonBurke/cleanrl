# FPO (Flow Policy Optimization, arXiv:2507.21053) on the sepclip control
# (awr/ppo_continuous_action_ppo_sepclip_control_v1.py): same critic, optimizer,
# full-batch 10-epoch schedule, separate grad clipping and standard GAE; only
# the Beta actor becomes a flow policy trained with the FPO ratio
#   ratio = exp(mean_i l_old(tau_i, eps_i) - mean_i l_new(tau_i, eps_i)).
#
# Design (vs official code and our fpo_v1..v6):
# - Velocity-MSE CFM loss: its weighting is monotone in noise level (the OT
#   case of Kingma & Gao), unlike the official eps-MSE; it was also our best.
# - Stratified continuous tau per action (no dead tau where the loss is
#   theta-independent, lower MC variance than i.i.d. grid draws), N=16 pairs.
# - Old losses come from one batched forward over the uploaded rollout, so
#   they share the update's observation normalization (exact ratio=1 at start).
# - Shared observation trunk evaluated once; only a small residual velocity
#   head runs per Euler step and per MC pair. Prior std 0.6 matches the
#   control's initial Beta action spread after tanh.
# Hypothesis: a flow head can learn state-dependent, cross-joint coordinated
# exploration that a diagonal Beta cannot; logged spread/correlation tracks it.
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

from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch
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
WIDTH = 64
RESIDUAL_SCALE = 8.0  # sqrt(WIDTH): unit-norm features at unit per-coordinate RMS


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
    learning_rate: float = 0.0024
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.05
    """symmetric clipping radius of the FPO ratio"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping, applied per network"""

    # Flow policy
    flow_steps: int = 10
    """Euler steps of the rollout sampler (noise at tau=0, action at tau=1)"""
    mc_samples: int = 16
    """(tau, eps) pairs per action; taus are stratified over [0, 1)"""
    prior_std: float = 0.6
    """std of the Gaussian source; 0.6 matches the control's initial action spread"""
    time_embed_dim: int = 8
    """sinusoidal flow-time features"""
    diagnostic_states: int = 256
    """rollout states whose action distribution is probed each iteration"""
    diagnostic_samples: int = 32
    """flow samples per probed state"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile rollout statistics, the FPO loss and GAE"""
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


class ResidualPair(nn.Module):
    """The control's two-stage residual SiTU-GLU trunk: 8 * unit(h + branch)."""

    def __init__(self, in_dim):
        super().__init__()
        self.first = nn.Sequential(init_situglu_branch(SiTUGLUBranch(in_dim, WIDTH)))
        self.second = nn.Sequential(init_situglu_branch(SiTUGLUBranch(WIDTH, WIDTH)))

    def forward(self, x):
        h = F.normalize(self.first(x), p=2, dim=-1)
        branch = F.normalize(self.second(RESIDUAL_SCALE * h), p=2, dim=-1)
        return RESIDUAL_SCALE * F.normalize(h + branch, p=2, dim=-1)


class ResidualMLP(nn.Module):
    """Control critic, unchanged: residual pair, linear readout, learned gains."""

    def __init__(self, in_dim, output_dim, output_std):
        super().__init__()
        self.trunk = ResidualPair(in_dim)
        self.head = nn.Sequential(layer_init(nn.Linear(WIDTH, output_dim), std=output_std))
        self.readout_gain = nn.Parameter(torch.ones(output_dim))

    def forward(self, x):
        return self.readout_gain * self.head(self.trunk(x))


def time_features(times, dim):
    """(..., 1) flow times -> (..., dim) sinusoids at frequencies pi * 2^k."""
    frequencies = np.pi * 2.0 ** torch.arange(dim // 2, device=times.device, dtype=times.dtype)
    scaled = times * frequencies
    return torch.cat((scaled.cos(), scaled.sin()), dim=-1)


class FlowActor(nn.Module):
    """v(obs, x_tau, tau) = velocity head over [trunk(obs), x_tau, emb(tau)]."""

    def __init__(self, observation_dim, action_dim, time_embed_dim):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.trunk = ResidualPair(observation_dim)
        self.velocity_trunk = ResidualPair(WIDTH + action_dim + time_embed_dim)
        # normalize_matrices rescales head rows to unit norm, so the near-zero
        # initial velocity (initial policy = tanh(prior noise)) lives in the gain.
        self.head = nn.Sequential(layer_init(nn.Linear(WIDTH, action_dim)))
        self.readout_gain = nn.Parameter(torch.full((action_dim,), 0.01))

    def velocity(self, features, states, times):
        inputs = torch.cat((features, states, time_features(times, self.time_embed_dim)), dim=-1)
        return self.readout_gain * self.head(self.velocity_trunk(inputs))


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("flow PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("flow PPO requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        self.obs_dim = int(np.prod(envs.single_observation_space.shape))
        self.flow_steps = args.flow_steps
        self.prior_std = args.prior_std
        self.register_buffer("action_bias", torch.as_tensor((high + low).reshape(-1) / 2.0, dtype=torch.float32))
        self.register_buffer("action_scale", torch.as_tensor((high - low).reshape(-1) / 2.0, dtype=torch.float32))
        self.critic = ResidualMLP(self.obs_dim, 1, output_std=1.0)
        self.actor = FlowActor(self.obs_dim, self.action_dim, args.time_embed_dim)

    @torch.no_grad()
    def normalize_matrices(self):
        """nGPT row/column normalization of every residual stage and readout."""
        for pair in (self.critic.trunk, self.actor.trunk, self.actor.velocity_trunk):
            for stage in (pair.first, pair.second):
                branch = stage[0]
                for weight, dim in ((branch.gate.weight, 1), (branch.up.weight, 1), (branch.down.weight, 0)):
                    weight.div_(torch.linalg.vector_norm(weight, dim=dim, keepdim=True))
        for weight in (self.critic.head[0].weight, self.actor.head[0].weight):
            weight.div_(torch.linalg.vector_norm(weight, dim=1, keepdim=True))

    def get_value(self, x):
        return self.critic(x)

    @torch.no_grad()
    def sample(self, observations, noise):
        """Device Euler sampler (diagnostics); noise is (B, K, A) standard normal."""
        features = self.actor.trunk(observations)[:, None, :].expand(-1, noise.shape[1], -1)
        native = self.prior_std * noise
        times = native.new_empty(native.shape[:-1] + (1,))
        for step in range(self.flow_steps):
            times.fill_(step / self.flow_steps)
            native = native + self.actor.velocity(features, native, times) / self.flow_steps
        return native


def cfm_loss(agent, features, native, times, noise):
    """Per-action MC mean of velocity-MSE (mean over action dims, as official).

    x_tau = (1 - tau) * s * eps + tau * z and u = z - s * eps for the stored
    pre-tanh action z; (tau, eps) are frozen per rollout and shared by the old
    and new losses, so their difference is the FPO log-ratio.
    """
    native = native.detach()[:, None, :]
    source = agent.prior_std * noise
    states = (1.0 - times) * source + times * native
    expanded = features[:, None, :].expand(-1, noise.shape[1], -1)
    velocity = agent.actor.velocity(expanded, states, times)
    return (velocity - (native - source)).square().mean(dim=-1).mean(dim=-1)


def fpo_loss(agent, observations, native, times, noise, old_loss, advantages, returns, args):
    new_loss = cfm_loss(agent, agent.actor.trunk(observations), native, times, noise)
    logratio = old_loss - new_loss
    ratio = logratio.exp()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = torch.max(-advantages * ratio,
                        -advantages * ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef)).mean()
    value_loss = 0.5 * (agent.get_value(observations).view(-1) - returns).square().mean()
    loss = pg_loss + args.vf_coef * value_loss
    with torch.no_grad():
        # Divergence of the CFM proxy ratio, not of the policy itself.
        proxy_kl = ((ratio - 1.0) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
        metrics = torch.stack((pg_loss.detach(), value_loss.detach(), new_loss.mean().detach(),
                               proxy_kl, clipfrac, logratio.abs().mean()))
    return loss, metrics


class HostResidualPair:
    """FP32 host mirror of ResidualPair; returns a permanent (rows, WIDTH) buffer."""

    def __init__(self, pair, num_rows):
        self.first = make_host_mirror(pair.first, num_rows)
        self.second = make_host_mirror(pair.second, num_rows)
        self.hidden = np.empty((num_rows, WIDTH), dtype=np.float32)
        self.branch = np.empty_like(self.hidden)
        self.squared = np.empty_like(self.hidden)
        self.norm = np.empty((num_rows, 1), dtype=np.float32)

    def refresh(self):
        self.first.refresh()
        self.second.refresh()

    def _normalize(self, values):
        np.multiply(values, values, out=self.squared)
        np.sum(self.squared, axis=1, keepdims=True, out=self.norm)
        np.sqrt(self.norm, out=self.norm)
        # Match F.normalize's zero-vector definition, including its epsilon.
        np.maximum(self.norm, np.float32(1e-12), out=self.norm)
        np.divide(values, self.norm, out=values)

    def __call__(self, inputs):
        np.copyto(self.hidden, self.first(inputs))
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(RESIDUAL_SCALE), out=self.branch)
        np.copyto(self.branch, self.second(self.branch))
        self._normalize(self.branch)
        np.add(self.hidden, self.branch, out=self.hidden)
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(RESIDUAL_SCALE), out=self.hidden)
        return self.hidden


class HostFlowSampler:
    """Host Euler sampler: trunk once per env step, velocity head per Euler step.

    Returns permanent (native, action) buffers overwritten by the next call;
    stage them before sampling again. refresh() after every optimizer update.
    """

    def __init__(self, agent, num_rows):
        actor = agent.actor
        self.actor = actor
        self.steps = agent.flow_steps
        self.prior_std = np.float32(agent.prior_std)
        self.trunk = HostResidualPair(actor.trunk, num_rows)
        self.velocity_trunk = HostResidualPair(actor.velocity_trunk, num_rows)
        self.head = make_host_mirror(actor.head, num_rows)
        self.readout_gain = np.empty(agent.action_dim, dtype=np.float32)
        self.scale = agent.action_scale.cpu().numpy().copy()
        self.bias = agent.action_bias.cpu().numpy().copy()
        self.native = np.empty((num_rows, agent.action_dim), dtype=np.float32)
        self.action = np.empty_like(self.native)
        self.velocity = np.empty_like(self.native)
        self.inputs = np.empty((num_rows, WIDTH + agent.action_dim + actor.time_embed_dim), dtype=np.float32)
        grid = torch.arange(self.steps, dtype=torch.float32)[:, None] / self.steps
        self.time_rows = time_features(grid, actor.time_embed_dim).numpy()
        self.action_slice = slice(WIDTH, WIDTH + agent.action_dim)
        self.time_slice = slice(WIDTH + agent.action_dim, None)

    def refresh(self):
        self.trunk.refresh()
        self.velocity_trunk.refresh()
        self.head.refresh()
        np.copyto(self.readout_gain, self.actor.readout_gain.detach().cpu().numpy())

    def __call__(self, observations, rng):
        np.copyto(self.inputs[:, :WIDTH], self.trunk(observations))
        rng.standard_normal(self.native.shape, dtype=np.float32, out=self.native)
        self.native *= self.prior_std
        step_size = np.float32(1.0 / self.steps)
        for step in range(self.steps):
            self.inputs[:, self.action_slice] = self.native
            self.inputs[:, self.time_slice] = self.time_rows[step]
            np.multiply(self.head(self.velocity_trunk(self.inputs)), self.readout_gain, out=self.velocity)
            self.velocity *= step_size
            self.native += self.velocity
        np.tanh(self.native, out=self.action)
        self.action *= self.scale
        self.action += self.bias
        return self.native, self.action


class RolloutPairs:
    """Per-rollout frozen (tau, eps) pairs, old CFM losses and old values."""

    def __init__(self, args, action_dim, device):
        self.times = torch.empty((args.batch_size, args.mc_samples, 1), device=device)
        self.noise = torch.empty((args.batch_size, args.mc_samples, action_dim), device=device)
        self.strata = torch.arange(args.mc_samples, device=device, dtype=torch.float32)[None, :, None]
        self.mc_samples = args.mc_samples

    @torch.no_grad()
    def draw(self, generator):
        # One uniform draw inside each of the N equal-width tau strata.
        self.times.uniform_(generator=generator).add_(self.strata).div_(self.mc_samples)
        self.noise.normal_(generator=generator)


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs,
           args.flow_steps, args.mc_samples) <= 0:
        raise ValueError("environment, rollout, minibatch, epoch, flow and MC counts must be positive")
    if args.time_embed_dim <= 0 or args.time_embed_dim % 2:
        raise ValueError("time_embed_dim must be a positive even number")
    if not 0.0 < args.clip_coef < 1.0 or args.prior_std <= 0.0:
        raise ValueError("clip_coef must lie in (0, 1) and prior_std must be positive")
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
        args.env_id, args.num_envs, backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video, run_name=run_name,
    )


def spread_statistics(native):
    """(S, K, A) pre-tanh samples -> mean per-dim action std, mean |off-diagonal corr|."""
    actions = native.tanh()
    centered = actions - actions.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / (actions.shape[1] - 1)
    std = covariance.diagonal(dim1=1, dim2=2).clamp_min(1e-12).sqrt()
    correlation = covariance / (std[:, :, None] * std[:, None, :])
    off_diagonal = ~torch.eye(actions.shape[-1], dtype=torch.bool, device=actions.device)
    return std.mean(), correlation[:, off_diagonal].abs().mean()


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
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "FPO flow: tanh(Euler(v)) from N(0, prior_std^2); velocity-MSE ratio; stratified tau")
        writer.add_text("architecture", "control residual SiTU-GLU critic; actor = residual obs trunk + residual velocity head")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        actor_parameters, critic_parameters = tuple(agent.actor.parameters()), tuple(agent.critic.parameters())
        value_model = agent.get_value

        def rollout_statistics(observations, native, times, noise):
            """Old values and old CFM losses for the whole uploaded rollout in one forward."""
            return (agent.get_value(observations).flatten(),
                    cfm_loss(agent, agent.actor.trunk(observations), native, times, noise))

        def loss_model(observations, native, times, noise, old_loss, advantages, returns):
            return fpo_loss(agent, observations, native, times, noise, old_loss, advantages, returns, args)

        def probe(observations, noise):
            return spread_statistics(agent.sample(observations, noise))

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            probe = graph_compile(probe)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: Euler-integrate the flow on an FP32
        # host mirror, stage everything in pinned memory, upload once per rollout.
        host_sampler = HostFlowSampler(agent, args.num_envs)
        sampler = np.random.default_rng(args.seed)

        def act(observations):
            native, action = host_sampler(observations, sampler)
            if not (np.isfinite(native).all() and np.isfinite(action).all()):
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling and MC pairs must not consume the policy sampler's random streams.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        pair_generator = torch.Generator(device=device).manual_seed(args.seed + 991)
        probe_generator = torch.Generator(device=device).manual_seed(args.seed + 1999)
        pairs = RolloutPairs(args, agent.action_dim, device)
        probe_noise = torch.empty((args.diagnostic_states, args.diagnostic_samples, agent.action_dim), device=device)
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
            host_sampler.refresh()
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
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native)
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
                pairs.draw(pair_generator)
                b_values, b_old_loss = rollout_statistics(b_obs, b_native, pairs.times, pairs.noise)
                values = b_values.view(args.num_steps, args.num_envs)
                tail_value = value_model(transfer.observation(next_obs_np)).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
                # Probe the behavior policy that collected this rollout.
                probe_rows = torch.randint(args.batch_size, (args.diagnostic_states,), device=device,
                                           generator=probe_generator)
                probe_noise.normal_(generator=probe_generator)
                action_std, action_abs_corr = (value.clone() for value in probe(b_obs[probe_rows], probe_noise))
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], pairs.times[indices], pairs.noise[indices],
                            b_old_loss[indices], b_advantages[indices], b_returns[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norms[updates, 0] = nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm, foreach=True)
                        grad_norms[updates, 1] = nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
                        optimizer.step()
                        normalize_matrices()
                        update_metrics[updates].copy_(metrics)
                        updates += 1

            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/cfm_loss": last[2], "losses/proxy_kl": last[3],
                "losses/clipfrac": update_metrics[:updates, 4].mean(),
                "losses/abs_logratio": last[5],
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "grad/actor_preclip_norm": grad_norms[:updates, 0].mean(),
                "grad/critic_preclip_norm": grad_norms[:updates, 1].mean(),
                "policy/action_std": action_std,
                "policy/abs_corr": action_abs_corr,
                "policy/max_abs_native": b_native.abs().max(),
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite FPO learner metrics")
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

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            # Per-env observation statistics are needed to replay the policy.
            torch.save({"state_dict": agent.state_dict(), "args": vars(args),
                        "obs_norm": {"means": obs_norm.means, "variances": obs_norm.variances,
                                     "counts": obs_norm.counts}}, model_path)
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

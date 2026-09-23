"""Learned-world-model RFO adaptation of arXiv:2602.03501v1, Eqs.10--15/App.B.

Native MuJoCo is NOT differentiable. This is not the paper's simulator experiment:
we fit raw transition deltas/rewards/termination with a learned model, then really
backpropagate through that model AND every forward-Euler flow step. Horizon 3
(instead of paper 32) deliberately limits accumulated learned-model error, not
policy gradients. No action chunking, PPO ratio, likelihood, entropy or KL proxy.
Keep the baseline 64x64 Tanh networks, Adam(3e-4, eps=1e-5), clipping of gradient
norms, vector normalization and TD-lambda/GAE collector. Dual independent critics
average bootstrap values. One actor pass/optimizer step per rollout (Algorithm1).

Gaussian source t=0 -> latent data t=1, velocity=data-noise, Euler K=4, followed
by an affine tanh decoder. App.B CFM targets are pre-tanh; rollout latents are
stored, NEVER reconstructed by inverting saturated actions. Only CFM targets are
bounded at the FP32 endpoint-safe atanh limit (paper does not specify its bound).
Uniform CFM draws are uniform in bounded physical action space, not latent space.

World coordinates/scales are fixed from the first TRAIN split, permanently;
world supervision always uses raw states/rewards and pre-reset final observations.
Actor/critic vector-normalization snapshots are frozen within each rollout and
its update. The two-rollout CFM buffer stores raw states and detached latents,
re-normalized with the current snapshot. Imagined rewards use the same frozen
reward scaling/standard normalization clip as real GAE rewards. Learned survival
probabilities mask future rewards/bootstrap; time-limit truncations are not death.
Model holdout metrics measure prediction error, NOT gradient reliability or a
confidence gate. Model exploitation/partial observability remain real limitations.
"""
import hashlib
import json
import math
import os
import random
import time
from contextlib import ExitStack, contextmanager
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
from cleanrl.shared.ppo_loop import device_minibatches, explained_variance, gather_metrics, get_gae_fn
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
CFM_TARGET_BOUND = math.atanh(1.0 - np.finfo(np.float32).eps)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    """critic epochs only; the actor always gets one pass and one optimizer step"""
    world_epochs: int = 10
    world_holdout_fraction: float = 0.1
    imagination_horizon: int = 3
    """shorter than paper H=32 to reduce learned-model error accumulation"""
    flow_steps: int = 4
    c_past: float = 0.2
    c_uni: float = 0.2
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    env_backend: str = "auto"
    env_threads: int = 2
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    non_blocking_transfers: bool = False
    staggered_starts: bool = True
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(inputs, outputs, output_std):
    return nn.Sequential(layer_init(nn.Linear(inputs, 64)), nn.Tanh(),
                         layer_init(nn.Linear(64, 64)), nn.Tanh(),
                         layer_init(nn.Linear(64, outputs), std=output_std))


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        args = Args() if args is None else args
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("RFO requires a bounded Box action space")
        low, high = np.asarray(space.low).reshape(-1), np.asarray(space.high).reshape(-1)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and (high > low).all()):
            raise ValueError("RFO requires finite strictly ordered bounds")
        self.obs_dim = int(np.prod(envs.single_observation_space.shape))
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        self.flow_steps = args.flow_steps
        self.register_buffer("action_low", torch.tensor(low.copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(high.copy(), dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("FP32 action range must be finite and positive")
        self.actor = mlp(self.obs_dim + self.action_dim + 1, self.action_dim, 0.01)
        # Separate constructions: independently initialized, never deep-copied.
        self.critics = nn.ModuleList([mlp(self.obs_dim, 1, 1.0), mlp(self.obs_dim, 1, 1.0)])
        self.register_buffer("obs_mean", torch.zeros(args.num_envs, self.obs_dim))
        self.register_buffer("obs_std", torch.ones(args.num_envs, self.obs_dim))
        self.register_buffer("reward_std", torch.ones(args.num_envs))

    def velocity(self, observations, latent, times):
        return self.actor(torch.cat((observations, latent, times), dim=-1))

    def sample(self, observations, noise):
        latent = noise
        for step in range(self.flow_steps):
            times = torch.full_like(latent[:, :1], step / self.flow_steps)
            latent = latent + self.velocity(observations, latent, times) / self.flow_steps
        return latent, self.action_bias + self.action_scale * latent.tanh()

    def get_value(self, observations):
        return (self.critics[0](observations) + self.critics[1](observations)) * 0.5


def normalize_states(raw, mean, std):
    return ((raw - mean) / std).clamp(-10.0, 10.0)


def normalize_rewards(raw, std):
    return (raw / std).clamp(-10.0, 10.0)


class WorldModel(nn.Module):
    """Raw-coordinate delta/reward/termination model; fixed affine conditioning."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.network = mlp(obs_dim + action_dim, obs_dim + 2, 1.0)
        self.register_buffer("state_mean", torch.zeros(obs_dim))
        self.register_buffer("state_std", torch.ones(obs_dim))
        self.register_buffer("delta_std", torch.ones(obs_dim))
        self.register_buffer("reward_scale", torch.ones(()))
        self.register_buffer("action_bias", torch.zeros(action_dim))
        self.register_buffer("action_scale", torch.ones(action_dim))

    @torch.no_grad()
    def initialize_coordinates(self, raw, next_raw, rewards, agent):
        self.state_mean.copy_(raw.mean(0))
        self.state_std.copy_((raw.var(0, unbiased=False) + 1e-8).sqrt())
        self.delta_std.copy_(((next_raw - raw).square().mean(0) + 1e-8).sqrt())
        self.reward_scale.copy_((rewards.square().mean() + 1e-8).sqrt())
        self.action_bias.copy_(agent.action_bias)
        self.action_scale.copy_(agent.action_scale)

    def forward(self, raw_states, actions):
        # Do not clip raw world coordinates: preserve dynamics input gradients.
        inputs = torch.cat(((raw_states - self.state_mean) / self.state_std,
                            (actions - self.action_bias) / self.action_scale), dim=-1)
        prediction = self.network(inputs)
        next_raw = raw_states + prediction[:, :self.obs_dim] * self.delta_std
        raw_reward = prediction[:, self.obs_dim] * self.reward_scale
        termination_logits = prediction[:, self.obs_dim + 1]
        return next_raw, raw_reward, termination_logits


def world_loss(world, raw, actions, next_raw, raw_rewards, terminations):
    predicted_next, predicted_reward, logits = world(raw, actions)
    delta_error = (predicted_next - next_raw) / world.delta_std
    reward_error = (predicted_reward - raw_rewards) / world.reward_scale
    state_loss = delta_error.square().mean()
    reward_loss = reward_error.square().mean()
    termination_loss = F.binary_cross_entropy_with_logits(logits, terminations)
    metrics = torch.stack((state_loss.detach(), reward_loss.detach(), termination_loss.detach(),
                           (predicted_next - next_raw).square().mean().sqrt().detach(),
                           (predicted_reward - raw_rewards).square().mean().sqrt().detach(),
                           (logits.sigmoid() - terminations).square().mean().detach()))
    return state_loss + reward_loss + termination_loss, metrics


@contextmanager
def frozen_parameters(*modules):
    """Freeze weights, not computations: input Jacobians remain differentiable."""
    parameters = [parameter for module in modules for parameter in module.parameters()]
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.grad = None
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def imagined_return(agent, world, raw_states, mean, std, reward_std, noises, gamma):
    """Eq.10 with differentiable learned survival; NO detach inside the horizon."""
    states = raw_states
    survival = torch.ones_like(reward_std)
    total = torch.zeros_like(reward_std)
    discount = 1.0
    for noise in noises.unbind(0):
        _, action = agent.sample(normalize_states(states, mean, std), noise)
        states, reward, termination_logits = world(states, action)
        total = total + discount * survival * normalize_rewards(reward, reward_std)
        survival = survival * torch.sigmoid(-termination_logits)
        discount *= gamma
    terminal = agent.get_value(normalize_states(states, mean, std)).flatten()
    return total + discount * survival * terminal


def uniform_targets(agent, count, generator=None):
    # Midpoints of 2**23 equal-probability physical-action bins. Endpoints cannot
    # occur in FP32; inverse the generating affine coordinate, not rounded actions.
    bins = torch.randint(0, 2**23, (count, agent.action_dim), device=agent.action_low.device,
                         generator=generator)
    unit = (bins.to(torch.float32) + 0.5) / 2**23
    signed = 2.0 * unit - 1.0
    return agent.action_low + 2.0 * agent.action_scale * unit, torch.atanh(signed)


def cfm_loss(agent, observations, targets, noise, times):
    targets = targets.detach().clamp(-CFM_TARGET_BOUND, CFM_TARGET_BOUND)
    noise, times = noise.detach(), times.detach()
    state = (1.0 - times) * noise + times * targets
    return (agent.velocity(observations, state, times) - (targets - noise)).square().sum(-1).mean()


def actor_loss(agent, world, raw, mean, std, reward_std, recent_raw, recent_native,
               recent_mean, recent_std, noises, past_noise, past_times,
               uniform_native, uniform_noise, uniform_times, args):
    rpg = -imagined_return(agent, world, raw, mean, std, reward_std, noises, args.gamma).mean()
    past = cfm_loss(agent, normalize_states(recent_raw, recent_mean, recent_std),
                    recent_native, past_noise, past_times)
    uniform = cfm_loss(agent, normalize_states(raw, mean, std),
                       uniform_native, uniform_noise, uniform_times)
    loss = rpg + args.c_past * past + args.c_uni * uniform
    return loss, torch.stack((loss.detach(), rpg.detach(), past.detach(), uniform.detach()))


def critic_loss(agent, observations, returns, old_values, args):
    values = torch.cat([critic(observations) for critic in agent.critics], dim=-1)
    errors = (values - returns[:, None]).square()
    if args.clip_vloss:
        clipped = old_values + (values - old_values).clamp(-args.clip_coef, args.clip_coef)
        errors = torch.maximum(errors, (clipped - returns[:, None]).square())
    return args.vf_coef * 0.5 * errors.mean()


class RecentRollouts:
    """Exactly the current and immediately previous complete rollout, no replay."""
    def __init__(self):
        self.rollouts = []

    def push(self, raw_observations, native_actions):
        self.rollouts = self.rollouts[-1:] + [(raw_observations.detach().clone(), native_actions.detach().clone())]

    def minibatch(self, indices):
        return (torch.cat([raw[indices] for raw, _ in self.rollouts]),
                torch.cat([native[indices] for _, native in self.rollouts]))


def transition_targets(raw_next, rewards, terminations, truncations, infos):
    """Recover BOTH death/time-limit final states; only death labels termination."""
    next_states = np.asarray(raw_next, dtype=np.float32)
    boundaries = np.flatnonzero(np.logical_or(terminations, truncations))
    if boundaries.size:
        next_states = next_states.copy()
        finals, mask = infos.get("final_observation"), infos.get("_final_observation")
        for index in boundaries:
            if finals is None or finals[index] is None or (mask is not None and not mask[index]):
                raise RuntimeError(f"completed environment {index} missing final_observation")
            next_states[index] = finals[index]
    return next_states, np.asarray(rewards, dtype=np.float32), np.asarray(terminations, dtype=np.float32)


class RawTrackingObsNorm(VectorObsNorm):
    """Keep the raw counterpart of warmup's normalized result, never inverse clip."""
    def __init__(self, num_envs, obs_shape):
        super().__init__(num_envs, obs_shape)
        self.raw_observations = np.empty((num_envs,) + tuple(obs_shape), dtype=np.float32)

    def normalize(self, obs, rows=None, out_dtype=np.float32):
        if rows is None:
            self.raw_observations[...] = obs
        else:
            self.raw_observations[rows] = obs
        return super().normalize(obs, rows=rows, out_dtype=out_dtype)


class HostSampler:
    """Reusable NumPy Euler buffers and FP32 compiled host actor mirror."""
    def __init__(self, agent, num_envs):
        self.obs_dim = agent.obs_dim
        self.steps = agent.flow_steps
        self.actor = make_host_mirror(agent.actor, num_envs)
        self.scale = agent.action_scale.cpu().numpy().copy()
        self.bias = agent.action_bias.cpu().numpy().copy()
        self.native = np.empty((num_envs, agent.action_dim), dtype=np.float32)
        self.action = np.empty_like(self.native)
        self.work = np.empty_like(self.native)
        self.inputs = np.empty((num_envs, agent.obs_dim + agent.action_dim + 1), dtype=np.float32)

    def __call__(self, observations, rng):
        rng.standard_normal(self.native.shape, dtype=np.float32, out=self.native)
        self.inputs[:, :self.obs_dim] = observations
        for step in range(self.steps):
            self.inputs[:, self.obs_dim:-1] = self.native
            self.inputs[:, -1] = np.float32(step / self.steps)
            np.multiply(self.actor(self.inputs), np.float32(1.0 / self.steps), out=self.work)
            self.native += self.work
        np.tanh(self.native, out=self.action)
        self.action *= self.scale
        self.action += self.bias
        if not np.isfinite(self.native).all():
            raise FloatingPointError("nonfinite flow rollout")
        return self.native, self.action


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs,
           args.world_epochs, args.imagination_horizon, args.flow_steps, args.env_threads) <= 0:
        raise ValueError("rollout, epoch, horizon, integration and thread counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"}:
        raise ValueError("invalid environment backend")
    if not args.cuda or not args.compile:
        raise ValueError("this RFO trainer requires compiled CUDA updates")
    if not 0 < args.world_holdout_fraction < 1:
        raise ValueError("world holdout fraction must lie strictly inside (0,1)")
    for name in ("learning_rate", "max_grad_norm", "vf_coef"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not (0 <= args.gamma < 1 and 0 <= args.gae_lambda <= 1):
        raise ValueError("invalid discount or GAE lambda")
    if any(not math.isfinite(value) or value < 0 for value in (args.c_past, args.c_uni, args.clip_coef)):
        raise ValueError("regularization weights and value clip must be finite and nonnegative")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size < 1 or args.batch_size < 2:
        raise ValueError("need a nonempty minibatch and separate world train/holdout transitions")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(args.env_id, args.num_envs, backend=backend,
                                  num_threads=min(args.env_threads, args.num_envs),
                                  capture_video=args.capture_video, run_name=run_name)


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
        raise ValueError("total_timesteps must cover warmup and a complete rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    with ExitStack() as resources:
        writer = SummaryWriter(f"runs/{run_name}")
        resources.callback(writer.close)
        metric_file = resources.enter_context(open(f"runs/{run_name}/metrics.jsonl", "w"))
        with open(f"runs/{run_name}/config.json", "w") as handle:
            json.dump(vars(args), handle, indent=2)
        with open(__file__, "rb") as handle:
            source = handle.read()
        provenance = {"paper": "https://arxiv.org/pdf/2602.03501v1", "adaptation": "learned-world-model RFO",
                      "source_sha256": hashlib.sha256(source).hexdigest(), "args": vars(args),
                      "world_coordinates": "fixed first training split", "cfm_target_bound": CFM_TARGET_BOUND}
        with open(f"runs/{run_name}/provenance.json", "w") as handle:
            json.dump(provenance, handle, indent=2)
        with open(f"runs/{run_name}/source.py", "wb") as handle:
            handle.write(source)
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", __doc__)
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        with torch.device(device):
            agent = Agent(envs, args)
            world = WorldModel(agent.obs_dim, agent.action_dim)
        actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        critic_optimizer = optim.Adam(agent.critics.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        world_optimizer = optim.Adam(world.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        # Keep actor accumulation storage outside CUDA Graph ownership. A None
        # grad may alias a compiled backward output, invalidated by the next
        # minibatch's mark_step_begin(); autograd must add into persistent grads.
        for parameter in agent.actor.parameters():
            parameter.grad = torch.zeros_like(parameter)
        actor_objective = torch.compile(lambda *inputs: actor_loss(agent, world, *inputs, args),
                                        mode=args.compile_mode, fullgraph=True, dynamic=False)
        world_objective = torch.compile(lambda *inputs: world_loss(world, *inputs),
                                        mode=args.compile_mode, fullgraph=True, dynamic=False)
        critic_objective = torch.compile(lambda *inputs: critic_loss(agent, *inputs, args),
                                         mode=args.compile_mode, fullgraph=True, dynamic=False)
        # No CUDA graphs for persistent rollout statistics: outputs survive many
        # compiled world/actor calls before critic fitting consumes them.
        value_model = torch.compile(agent.get_value, fullgraph=True, dynamic=True,
                                    options={"triton.cudagraphs": False})
        gae_fn = get_gae_fn(compiled=True, mode=args.compile_mode, explicit_next_values=True)
        obs_shape = envs.single_observation_space.shape
        if len(obs_shape) != 1:
            raise ValueError("learned-world-model RFO requires vector observations")
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers, store_transition_observations=True,
                                   fields={"raw_observations": obs_shape, "native_actions": (agent.action_dim,),
                                           "actions": (agent.action_dim,), "raw_rewards": ()})
        resources.callback(transfer.close)
        host = HostSampler(agent, args.num_envs)
        rng = np.random.default_rng(args.seed)
        shuffle = torch.Generator(device=device).manual_seed(args.seed)
        obs_norm = RawTrackingObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        recent = RecentRollouts()
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)
        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=lambda obs: host(obs, rng)[1].reshape((args.num_envs,) + agent.action_shape),
                                    horizon=horizon, phase_offsets=phases, seed=args.seed)
            global_step, suppress = warm.transitions, warm.suppress_mask
        else:
            raw, _ = envs.reset(seed=args.seed)
            obs_norm.normalize(raw)
            global_step = 0
        raw_current = obs_norm.raw_observations.copy()
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step
        slots = torch.arange(args.batch_size, device=device) % args.num_envs
        heldout_count = min(args.batch_size - 1, max(1, int(args.batch_size * args.world_holdout_fraction)))
        actor_metrics = torch.zeros(4, device=device)
        for iteration in range(1, args.num_iterations + 1):
            for optimizer in (actor_optimizer, critic_optimizer, world_optimizer):
                if args.anneal_lr:
                    optimizer.param_groups[0]["lr"] = args.learning_rate * (1.0 - (iteration - 1) / args.num_iterations)
            # All real/imagined actor and critic coordinates use this snapshot.
            mean_np = obs_norm.means.astype(np.float32).copy()
            std_np = np.sqrt(obs_norm.variances + obs_norm.epsilon).astype(np.float32)
            reward_std_np = np.sqrt(rew_norm.variances + rew_norm.epsilon).astype(np.float32)
            agent.obs_mean.copy_(torch.from_numpy(mean_np).to(device))
            agent.obs_std.copy_(torch.from_numpy(std_np).to(device))
            agent.reward_std.copy_(torch.from_numpy(reward_std_np).to(device))
            host.actor.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    normalized = np.clip((raw_current - mean_np) / std_np, -10, 10)
                    native, action = host(normalized, rng)
                with timer.span("env", use_cuda=False):
                    raw_next, raw_reward, terms, truncs, infos = envs.step(
                        action.reshape((args.num_envs,) + agent.action_shape))
                with timer.span("normalize_transfer", use_cuda=False):
                    next_target, reward_target, term_target = transition_targets(raw_next, raw_reward, terms, truncs, infos)
                    reward = np.clip(reward_target / reward_std_np, -10, 10)
                    transfer.push(step, reward, term_target, truncs, next_target,
                                  raw_observations=raw_current, native_actions=native, actions=action,
                                  raw_rewards=reward_target)
                    # Update moments for NEXT rollout, never fit drifting deltas.
                    obs_norm.normalize_step(raw_next, terms, truncs, infos)
                    rew_norm.normalize(raw_reward, terms)
                    np.copyto(raw_current, raw_next)
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
                raw = batch.fields["raw_observations"].flatten(0, 1)
                next_raw = batch.transition_observations.flatten(0, 1)
                actions = batch.fields["actions"].flatten(0, 1)
                native = batch.fields["native_actions"].flatten(0, 1)
                raw_rewards = batch.fields["raw_rewards"].flatten()
                terms = batch.terminations.flatten()
                means, stds, reward_stds = agent.obs_mean[slots], agent.obs_std[slots], agent.reward_std[slots]
                observations = normalize_states(raw, means, stds)
                values = value_model(observations).flatten().clone()
                old_dual_values = torch.cat([critic(observations) for critic in agent.critics], -1).clone()
                next_values = value_model(normalize_states(next_raw, means, stds)).reshape(args.num_steps, args.num_envs).clone()
                _, returns = gae_fn(batch.rewards, values.reshape(args.num_steps, args.num_envs),
                                    batch.terminations, batch.truncations, next_values, args.gamma, args.gae_lambda)
                returns = returns.flatten().clone()
                recent.push(raw, native)
                permutation = torch.randperm(args.batch_size, device=device, generator=shuffle)
                holdout, training = permutation[:heldout_count], permutation[heldout_count:]
                if iteration == 1:
                    world.initialize_coordinates(raw[training], next_raw[training], raw_rewards[training], agent)
            with timer.span("world_update"):
                with torch.no_grad():
                    _, before = world_objective(raw[holdout], actions[holdout], next_raw[holdout], raw_rewards[holdout], terms[holdout])
                    before = before.clone()
                for _ in range(args.world_epochs):
                    for rows in device_minibatches(len(training), args.minibatch_size, device, shuffle):
                        indices = training[rows]
                        torch.compiler.cudagraph_mark_step_begin()
                        loss, _ = world_objective(raw[indices], actions[indices], next_raw[indices], raw_rewards[indices], terms[indices])
                        world_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(world.parameters(), args.max_grad_norm)
                        world_optimizer.step()
                with torch.no_grad():
                    _, after = world_objective(raw[holdout], actions[holdout], next_raw[holdout], raw_rewards[holdout], terms[holdout])
                    after = after.clone()
            with timer.span("actor_update"), frozen_parameters(world, agent.critics):
                actor_optimizer.zero_grad(set_to_none=False)
                actor_metrics.zero_()
                for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle):
                    count = len(indices)
                    recent_raw, recent_native = recent.minibatch(indices)
                    repeats = len(recent.rollouts)
                    noise = torch.randn(args.imagination_horizon, count, agent.action_dim, device=device)
                    past_noise = torch.randn_like(recent_native)
                    past_times = torch.rand(count * repeats, 1, device=device)
                    _, uniform_native = uniform_targets(agent, count)
                    uniform_noise = torch.randn_like(uniform_native)
                    uniform_times = torch.rand(count, 1, device=device)
                    torch.compiler.cudagraph_mark_step_begin()
                    loss, metrics = actor_objective(raw[indices], means[indices], stds[indices], reward_stds[indices],
                                                    recent_raw, recent_native, means[indices].repeat(repeats, 1),
                                                    stds[indices].repeat(repeats, 1), noise, past_noise, past_times,
                                                    uniform_native, uniform_noise, uniform_times)
                    weight = count / args.batch_size
                    (loss * weight).backward()
                    actor_metrics.add_(metrics, alpha=weight)
                actor_grad = nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                actor_optimizer.step()
            with timer.span("critic_update"):
                for _ in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle):
                        torch.compiler.cudagraph_mark_step_begin()
                        value_loss = critic_objective(observations[indices], returns[indices], old_dual_values[indices])
                        critic_optimizer.zero_grad(set_to_none=True)
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(agent.critics.parameters(), args.max_grad_norm)
                        critic_optimizer.step()
                value_loss = value_loss.detach().clone()
            metrics = {"losses/policy_loss": actor_metrics[0], "losses/rpg_loss": actor_metrics[1],
                       "losses/cfm_past": actor_metrics[2], "losses/cfm_uniform": actor_metrics[3],
                       "losses/value_loss": value_loss, "losses/actor_grad_norm": actor_grad,
                       "losses/explained_variance": explained_variance(values, returns),
                       "sampling/saturation_fraction": (native.tanh().abs() > 0.99).float().mean(),
                       "sampling/cfm_target_bound_fraction": (native.abs() > CFM_TARGET_BOUND).float().mean()}
            names = ("scaled_delta_mse", "scaled_reward_mse", "termination_bce", "state_rmse", "reward_rmse", "termination_brier")
            metrics.update({f"world/holdout_{phase}_{name}": result[index]
                            for phase, result in (("pre", before), ("post", after)) for index, name in enumerate(names)})
            logged = gather_metrics(metrics)
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite RFO learner metrics")
            now = time.perf_counter()
            logged.update({"optimizer/actor_steps": iteration, "buffer/recent_rollouts": len(recent.rollouts),
                           "sampling/nfe": args.flow_steps, "charts/learning_rate": actor_optimizer.param_groups[0]["lr"],
                           "charts/SPS": int(global_step / (now - start_time)),
                           "charts/interval_SPS": (global_step - interval_step) / (now - interval_start)})
            for phase, timing in timer.summary().items():
                logged[f"timing/{phase}_s"] = timing["total_s"]
            logged["timing/update_s"] = sum(
                logged[f"timing/{phase}_s"] for phase in ("world_update", "actor_update", "critic_update")
            )
            timer.reset()
            metric_file.write(json.dumps({"step": global_step, **logged}) + "\n")
            metric_file.flush()
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            print(f"SPS: {logged['charts/SPS']}")
            interval_start, interval_step = time.perf_counter(), global_step
        if args.save_model:
            torch.save({"state_dict": agent.state_dict(), "world_state_dict": world.state_dict(), "args": vars(args),
                        "actor_optimizer": actor_optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
                        "world_optimizer": world_optimizer.state_dict(), "global_step": global_step,
                        "observation_normalizer": {name: getattr(obs_norm, name).copy() for name in ("means", "variances", "counts")},
                        "reward_normalizer": {name: getattr(rew_norm, name).copy() for name in ("means", "variances", "counts", "returns")},
                        "recent_rollouts": recent.rollouts, "provenance": provenance},
                       f"runs/{run_name}/{args.exp_name}.cleanrl_model")


if __name__ == "__main__":
    main()

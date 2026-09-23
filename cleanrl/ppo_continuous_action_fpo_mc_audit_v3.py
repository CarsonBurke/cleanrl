"""FPO within-update Monte Carlo audit; algorithm unchanged from v2 tanh.

PPO critic, optimizer, GAE, clipping, and training settings remain unchanged.
Train with eight fixed CFM pairs. At five rollout checkpoints, compare the
training pairs with independent 8/32 pairs and two independent 128-pair sets
(combined as a 256-pair finite-MC reference), before and after every epoch.
Record signed clipping decisions, ratio errors and actor gradient agreement.
Private probe RNG and autograd.grad keep training RNG and .grad state intact.
This is a fresh full benchmark, not checkpoint finetuning or a shortened run.
"""

import hashlib
import json
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
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
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
    norm_adv: bool = True
    clip_coef: float = 0.05
    """FPO actor surrogate ratio clip; independent of the value clip."""
    value_clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    flow_steps: int = 10
    mc_samples: int = 8
    mse: Literal["epsilon", "velocity"] = "epsilon"
    action_transform: Literal["raw", "tanh"] = "tanh"
    """Raw reference actions; tanh exists only for the decoder ablation."""
    discrete_training_times: bool = True
    """Match official code's Euler-grid MC times; false uses Uniform[0,1)."""
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
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    action_bias: torch.Tensor
    time_frequencies: torch.Tensor

    def __init__(self, envs, args):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("FPO requires a continuous Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("finite, strictly ordered action bounds required")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        self.obs_dim = int(np.prod(envs.single_observation_space.shape))
        self.flow_steps = args.flow_steps
        self.action_transform = args.action_transform
        self.output_scale = 0.25
        self.register_buffer("time_frequencies", 2.0 ** torch.arange(4, dtype=torch.float32))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high / 2 - self.action_low / 2)
        self.register_buffer("action_bias", self.action_high / 2 + self.action_low / 2)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 half-range")
        # Preserve baseline critic dimensions and initialization order.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        layers = []
        widths = (self.obs_dim + self.action_dim + 8, 32, 32, 32, 32, self.action_dim)
        for index, (inputs, outputs) in enumerate(zip(widths[:-1], widths[1:])):
            linear = nn.Linear(inputs, outputs)
            # JAX LeCun uniform: fan_in variance 1/fan_in, including final head.
            nn.init.uniform_(linear.weight, -np.sqrt(3.0 / inputs), np.sqrt(3.0 / inputs))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
        self.actor = nn.Sequential(*layers)

    def get_value(self, observations):
        return self.critic(observations.reshape(-1, self.obs_dim))

    def velocity(self, observations, states, times):
        phases = times * self.time_frequencies
        embedding = torch.cat((phases.cos(), phases.sin()), dim=-1)
        return self.output_scale * self.actor(torch.cat((observations, states, embedding), dim=-1))

    def decode(self, native):
        if self.action_transform == "tanh":
            return self.action_bias + self.action_scale * native.tanh()
        return native

    @torch.no_grad()
    def sample(self, observations, noise):
        """CUDA inference/reference sampler; training rollouts use HostSampler."""
        observations = observations.reshape(-1, self.obs_dim)
        native = noise.detach().clone()
        times = native.new_empty((native.shape[0], 1))
        for step in range(self.flow_steps):
            times.fill_(1.0 - step / self.flow_steps)
            native = native - self.velocity(observations, native, times) * (1.0 / self.flow_steps)
        return native, self.decode(native)


def flow_interpolant(native, noise, times):
    """Eq.9 with frozen data/noise/time; inputs may include an MC dimension."""
    native, noise, times = native.detach(), noise.detach(), times.detach()
    return times * noise + (1.0 - times) * native, noise - native


def cfm_errors(agent, observations, native, times, noise, mse):
    """Return [action, MC] errors, mean over action coordinates (official code)."""
    times, noise = times.detach(), noise.detach()
    states, target_velocity = flow_interpolant(native.detach()[:, None, :], noise, times)
    rows, draws, _ = states.shape
    conditions = observations.detach().reshape(rows, 1, agent.obs_dim).expand(-1, draws, -1)
    velocity = agent.velocity(conditions, states, times)
    if mse == "epsilon":
        residual = states + (1.0 - times) * velocity - noise
    else:
        residual = velocity - target_velocity
    return residual.square().mean(dim=-1)


def cfm_loss(agent, observations, native, times, noise, mse):
    """One MC-averaged error PER ACTION, not per-pair surrogate ratios."""
    return cfm_errors(agent, observations, native, times, noise, mse).mean(dim=-1)


def clipped_surrogate(new_loss, old_loss, advantages, clip_coef):
    """Signed Eq.6/Algorithm1 surrogate; old statistics never receive gradients."""
    logratio = old_loss.detach() - new_loss
    ratio = logratio.exp()
    advantages = advantages.detach()
    loss = torch.maximum(-advantages * ratio,
                         -advantages * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)).mean()
    return loss, logratio, ratio


def fpo_loss(agent, observations, native, times, noise, old_loss, advantages, returns, old_values, args):
    new_loss = cfm_loss(agent, observations, native, times, noise, args.mse)
    advantages = advantages.detach()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss, logratio, ratio = clipped_surrogate(new_loss, old_loss, advantages, args.clip_coef)
    new_value = agent.get_value(observations).flatten()
    returns, old_values = returns.detach(), old_values.detach()
    squared_error = (new_value - returns).square()
    if args.clip_vloss:
        clipped = old_values + (new_value - old_values).clamp(-args.value_clip_coef, args.value_clip_coef)
        value_loss = 0.5 * torch.maximum(squared_error, (clipped - returns).square()).mean()
    else:
        value_loss = 0.5 * squared_error.mean()
    loss = policy_loss + args.vf_coef * value_loss
    with torch.no_grad():
        # Nonnegative proxy diagnostic ONLY: this is not policy KL divergence.
        divergence = (ratio - 1.0 - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
        metrics = torch.stack((policy_loss.detach(), value_loss.detach(), new_loss.mean().detach(),
                               divergence, clipfrac, ratio.mean()))
    return loss, metrics


class RolloutCFM:
    """Permanent device buffers; refresh ONCE per rollout before any optimizer step.

    Frozen old per-action errors/values and exact per-action MC pairs are indexed
    together for every minibatch of every epoch. No old-policy copy is needed.
    """
    def __init__(self, args, action_dim, device):
        self.args = args
        self.times = torch.empty((args.batch_size, args.mc_samples, 1), device=device)
        self.noise = torch.empty((args.batch_size, args.mc_samples, action_dim), device=device)
        self.old_loss = torch.empty(args.batch_size, device=device)
        self.old_values = torch.empty(args.batch_size, device=device)
        self.time_indices = torch.empty_like(self.times, dtype=torch.int64) if args.discrete_training_times else None

    @torch.no_grad()
    def refresh(self, statistics, observations, native, generator):
        args = self.args
        if self.time_indices is not None:
            torch.randint(args.flow_steps, self.time_indices.shape, out=self.time_indices, generator=generator)
            self.times.copy_(self.time_indices)
            self.times.div_(args.flow_steps)
            self.times.neg_().add_(1.0)
        else:
            self.times.uniform_(generator=generator)
        self.noise.normal_(generator=generator)
        # Chunk at the learner's minibatch shape: bounded activations, no scalar
        # transfers, and no full-rollout x MC autograd graph or repeated targets.
        for start in range(0, args.batch_size, args.minibatch_size):
            stop = start + args.minibatch_size
            values, errors = statistics(observations[start:stop], native[start:stop],
                                        self.times[start:stop], self.noise[start:stop])
            self.old_values[start:stop].copy_(values)
            self.old_loss[start:stop].copy_(errors)


class HostSampler:
    """Reverse Euler over permanent borrowed FP32 buffers, with no CUDA calls.

    refresh() is an iteration-boundary device-to-host synchronization; callers
    must stage the returned native sample before the next sampler invocation.
    """
    def __init__(self, agent, num_rows, steps):
        self.obs_dim = agent.obs_dim
        self.steps = steps
        self.output_scale = agent.output_scale
        self.action_transform = agent.action_transform
        self.actor = make_host_mirror(agent.actor, num_rows)
        self.scale = agent.action_scale.detach().cpu().numpy().copy()
        self.bias = agent.action_bias.detach().cpu().numpy().copy()
        self.native = np.empty((num_rows, agent.action_dim), dtype=np.float32)
        self.action = np.empty_like(self.native)
        self.work = np.empty_like(self.native)
        self.inputs = np.empty((num_rows, agent.obs_dim + agent.action_dim + 8), dtype=np.float32)
        times = (1.0 - np.arange(steps, dtype=np.float32) / steps)[:, None]
        phases = times * agent.time_frequencies.detach().cpu().numpy()[None, :]
        self.time_embeddings = np.concatenate((np.cos(phases), np.sin(phases)), axis=-1)

    def refresh(self):
        self.actor.refresh()

    def __call__(self, observations, rng, *, noise=None):
        if noise is None:
            rng.standard_normal(self.native.shape, dtype=np.float32, out=self.native)
        else:
            np.copyto(self.native, noise)
        self.inputs[:, :self.obs_dim] = observations.reshape(self.native.shape[0], self.obs_dim)
        step_size = np.float32(1.0 / self.steps)
        for step in range(self.steps):
            self.inputs[:, self.obs_dim:self.obs_dim + self.native.shape[-1]] = self.native
            self.inputs[:, -8:] = self.time_embeddings[step]
            np.multiply(self.actor(self.inputs), np.float32(self.output_scale), out=self.work)
            self.work *= step_size
            self.native -= self.work
        if self.action_transform == "tanh":
            np.tanh(self.native, out=self.action)
            self.action *= self.scale
            self.action += self.bias
        else:
            np.copyto(self.action, self.native)
        return self.native, self.action


class UpdateMCProbe:
    """Observe fixed states/old losses across epochs without mutating training.

    The 256-pair estimate is a finite-MC reference, not an exact likelihood.
    Independent 128-pair halves expose uncertainty in that reference. All draws
    use a private generator; autograd.grad leaves optimizer gradient buffers alone.
    """
    def __init__(self, agent, args, output, writer):
        self.agent, self.args, self.output, self.writer = agent, args, output, writer
        self.generator = torch.Generator(device='cuda').manual_seed(args.seed + 77123)
        self.errors = torch.compile(
            lambda obs, native, times, noise: cfm_loss(agent, obs, native, times, noise, args.mse),
            fullgraph=True, options={'triton.cudagraphs': False})
        self.parameters = tuple(agent.actor.parameters())

    @torch.no_grad()
    def begin(self, observations, native, advantages, cache, step):
        self.step = step
        rows = torch.randperm(len(observations), device=observations.device,
                             generator=self.generator)[:512]
        self.obs, self.native = observations[rows], native[rows]
        self.adv = advantages[rows].detach()
        # Identical advantage weights across all estimators and all epochs.
        if self.args.norm_adv:
            self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)
        self.pairs = {'train8': (cache.times[rows].clone(), cache.noise[rows].clone())}
        for name, draws in (('ind8', 8), ('ind32', 32), ('ref128a', 128), ('ref128b', 128)):
            times = torch.randint(self.args.flow_steps, (len(rows), draws, 1),
                                  device='cuda', generator=self.generator).float()
            times = 1.0 - times / self.args.flow_steps
            noise = torch.randn(len(rows), draws, self.agent.action_dim,
                                device='cuda', generator=self.generator)
            self.pairs[name] = times, noise
        a, b = self.pairs['ref128a'], self.pairs['ref128b']
        self.pairs['ref256'] = torch.cat((a[0], b[0]), 1), torch.cat((a[1], b[1]), 1)
        self.old = {}
        for name, (times, noise) in self.pairs.items():
            self.old[name] = torch.cat([
                self.errors(self.obs[i:i+128], self.native[i:i+128], times[i:i+128], noise[i:i+128]).clone()
                for i in range(0, len(rows), 128)])
        torch.testing.assert_close(self.old['train8'], cache.old_loss[rows], rtol=1e-4, atol=2e-6)
        self.old_snapshots = {name: value.clone() for name, value in self.old.items()}

    def measure(self, epoch):
        gradients, logratios, objectives = {}, {}, {}
        for name, (times, noise) in self.pairs.items():
            gradient = [torch.zeros_like(p) for p in self.parameters]
            ratios, losses = [], []
            for start in range(0, len(self.obs), 128):
                rows = slice(start, start + 128)
                error = self.errors(self.obs[rows], self.native[rows], times[rows], noise[rows])
                loss, logratio, _ = clipped_surrogate(error, self.old[name][rows], self.adv[rows], self.args.clip_coef)
                grads = torch.autograd.grad(loss, self.parameters)
                for total, g in zip(gradient, grads):
                    total.add_(g, alpha=128 / len(self.obs))
                ratios.append(logratio.detach().clone())
                losses.append(loss.detach().clone())
            gradients[name] = torch.cat([g.flatten() for g in gradient])
            logratios[name] = torch.cat(ratios)
            objectives[name] = -torch.stack(losses).mean()
            torch.testing.assert_close(self.old[name], self.old_snapshots[name], rtol=0, atol=0)
        if epoch == 0:
            for value in logratios.values():
                torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=2e-6)
        reference = logratios['ref256']
        reference_ratio = reference.exp()
        clip = self.args.clip_coef
        def blocked(ratio):
            return ((self.adv > 0) & (ratio > 1 + clip)) | ((self.adv < 0) & (ratio < 1 - clip))
        ref_blocked = blocked(reference_ratio)
        weight = self.adv.abs()
        metrics = {}
        for name, logratio in logratios.items():
            ratio = logratio.exp()
            stop = blocked(ratio)
            g, gref = gradients[name], gradients['ref256']
            prefix = f'audit/{name}/'
            metrics.update({
                prefix+'logratio_mean': logratio.mean(),
                prefix+'logratio_rmse': (logratio-reference).square().mean().sqrt(),
                prefix+'ratio_bias_vs_ref': (ratio-reference_ratio).mean(),
                prefix+'ratio_mae_vs_ref': (ratio-reference_ratio).abs().mean(),
                prefix+'clipfrac': ((ratio-1).abs() > clip).float().mean(),
                prefix+'blocked_fraction': stop.float().mean(),
                prefix+'blocked_disagreement': (stop != ref_blocked).float().mean(),
                prefix+'adv_weighted_blocked_disagreement': (weight*(stop != ref_blocked)).sum()/weight.sum(),
                prefix+'false_block_fraction': (stop & ~ref_blocked).float().mean(),
                prefix+'missed_block_fraction': (~stop & ref_blocked).float().mean(),
                prefix+'surrogate_gain': objectives[name],
                prefix+'grad_norm': g.norm(),
                prefix+'grad_cosine_vs_ref': torch.nn.functional.cosine_similarity(g, gref, dim=0),
                prefix+'grad_relative_error': (g-gref).norm()/gref.norm().clamp_min(1e-12),
            })
        result = gather_metrics(metrics)
        if not all(np.isfinite(value) for value in result.values()):
            raise FloatingPointError('nonfinite within-update MC diagnostic')
        self.output.write(json.dumps({'step': self.step, 'epoch': epoch, **result}) + '\n')
        self.output.flush()
        # Every rollout/epoch point has a distinct TB step; env step is retained
        # explicitly in the JSON evidence, not mislabeled as extra experience.
        probe_step = self.step * (self.args.update_epochs + 1) + epoch
        for name, value in result.items():
            self.writer.add_scalar(name, value, probe_step)
        print(f'MC_AUDIT step={self.step} epoch={epoch} '
              f'train8_cos={result["audit/train8/grad_cosine_vs_ref"]:.4f} '
              f'ind8_cos={result["audit/ind8/grad_cosine_vs_ref"]:.4f} '
              f'train8_block_disagree={result["audit/train8/blocked_disagreement"]:.4f}', flush=True)


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs,
           args.flow_steps, args.mc_samples) <= 0:
        raise ValueError("environment, rollout, minibatch, epoch, flow and MC counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.mse not in {"epsilon", "velocity"}:
        raise ValueError("mse must be epsilon or velocity")
    if args.action_transform not in {"raw", "tanh"}:
        raise ValueError("action_transform must be raw or tanh")
    if args.discrete_training_times and args.mse == "epsilon" and args.flow_steps == 1:
        raise ValueError("discrete epsilon training requires flow_steps >= 2; t=1 has zero actor gradient")
    if not 0 < args.clip_coef < 1 or not np.isfinite(args.value_clip_coef) or args.value_clip_coef < 0:
        raise ValueError("actor clip must be in (0,1) and value clip finite and nonnegative")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("FPO training requires CUDA")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(args.env_id, args.num_envs, backend=backend,
                                  num_threads=min(args.env_threads, args.num_envs),
                                  capture_video=args.capture_video, run_name=run_name)


def normalization_state(obs_norm, rew_norm):
    """Per-environment moments retained as tensors for safe checkpoint loading."""
    state = {}
    for name, normalizer in (("observation", obs_norm), ("reward", rew_norm)):
        state[name] = {field: torch.from_numpy(getattr(normalizer, field).copy())
                       for field in ("means", "variances", "counts")}
        state[name].update(epsilon=normalizer.epsilon, clip=normalizer.clip)
    state["reward"].update(gamma=rew_norm.gamma, returns=torch.from_numpy(rew_norm.returns.copy()))
    return state


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
    with ExitStack() as resources:
        writer = SummaryWriter(f"runs/{run_name}")
        resources.callback(writer.close)
        metric_file = resources.enter_context(open(f"runs/{run_name}/metrics.jsonl", "w"))
        with open(f"runs/{run_name}/config.json", "w") as handle:
            json.dump(vars(args), handle, indent=2)
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "FPO Algorithm1: frozen MC error ratios; signed clipped GAE; "
                        f"reference 4x32 SiLU; reverse Euler; {args.action_transform} actions; no likelihood/entropy/KL claims")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args)
        with open(__file__, "rb") as source:
            source_bytes = source.read()
        provenance = {
            "critic_hash": state_hash(agent.critic), "actor_hash": state_hash(agent.actor),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "actor_parameters": sum(p.numel() for p in agent.actor.parameters()),
            "critic_parameters": sum(p.numel() for p in agent.critic.parameters()),
            "paper": "https://arxiv.org/pdf/2507.21053v2",
            "reference_code": "https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/playground/src/flow_policy/fpo.py",
            "action_error_reduction": "mean", "mc_error_reduction": "mean_before_exp",
            "time_convention": "noise_at_1_data_at_0", "args": vars(args),
        }
        with open(f"runs/{run_name}/provenance.json", "w") as handle:
            json.dump(provenance, handle, indent=2)
        with open(f"runs/{run_name}/source.py", "wb") as handle:
            handle.write(source_bytes)
        agent = agent.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations, native, times, noise):
            return (agent.get_value(observations).flatten(),
                    cfm_loss(agent, observations, native, times, noise, args.mse))

        def loss_model(observations, native, times, noise, old_loss, advantages, returns, old_values):
            return fpo_loss(agent, observations, native, times, noise, old_loss,
                            advantages, returns, old_values, args)

        if args.compile:
            # Old stats are copied into owned buffers, never cudagraph-tree outputs.
            rollout_statistics = graph_compile(rollout_statistics)
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        sampler_rng = np.random.default_rng(args.seed)
        sample_actions = HostSampler(agent, args.num_envs, args.flow_steps)

        def act(observations):
            native, action = sample_actions(observations, sampler_rng)
            if not np.isfinite(native).all() or not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite samples")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        mc_generator = torch.Generator(device=device).manual_seed(args.seed + 991)
        cfm = RolloutCFM(args, agent.action_dim, device)
        audit_output = resources.enter_context(open(f"runs/{run_name}/mc_audit.jsonl", "w"))
        probe = UpdateMCProbe(agent, args, audit_output, writer)
        probe_iterations = {1, 31, 61, 122, args.num_iterations}
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)
        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=lambda observations: act(observations)[1], horizon=horizon,
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
            sample_actions.refresh()
            # All model inference, noise draws, normalization and staging below
            # are host-only. CUDA statistics/GAE begin after the full rollout.
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
                cfm.refresh(rollout_statistics, b_obs, b_native, mc_generator)
                b_values = cfm.old_values
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
            audit = iteration in probe_iterations
            if audit:
                with timer.span("mc_audit"):
                    probe.begin(b_obs, b_native, b_advantages, cfm, global_step)
                    probe.measure(0)
            updates = 0
            for epoch in range(args.update_epochs):
                with timer.span("update"):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], cfm.times[indices], cfm.noise[indices],
                            cfm.old_loss[indices], b_advantages[indices], b_returns[indices], b_values[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                if audit:
                    with timer.span("mc_audit"):
                        probe.measure(epoch + 1)
            with torch.no_grad():
                decoded = agent.decode(b_native)
                bounded = ((decoded.clamp(agent.action_low, agent.action_high) - agent.action_bias)
                           / agent.action_scale)
                logged = gather_metrics({
                    "losses/policy_loss": update_metrics[:updates, 0].mean(),
                    "losses/value_loss": update_metrics[:updates, 1].mean(),
                    "flow/mean_cfm_error": update_metrics[:updates, 2].mean(),
                    "flow/old_mean_cfm_error": cfm.old_loss.mean(),
                    "surrogate/divergence": update_metrics[:updates, 3].mean(),
                    "losses/clipfrac": update_metrics[:updates, 4].mean(),
                    "surrogate/ratio_mean": update_metrics[:updates, 5].mean(),
                    "losses/explained_variance": explained_variance(b_values, b_returns),
                    "sampling/saturation_fraction": (bounded.abs() > 0.99).float().mean(),
                    "sampling/rollout_action_std": bounded.std(dim=0, unbiased=False).mean(),
                    "sampling/rollout_native_std": b_native.std(dim=0, unbiased=False).mean(),
                })
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite FPO learner metrics")
            logged["sampling/nfe"] = args.flow_steps
            logged["optimizer/steps"] = updates
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

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save({"state_dict": agent.state_dict(), "args": vars(args), "provenance": provenance,
                        "normalization": normalization_state(obs_norm, rew_norm), "global_step": global_step}, model_path)
            print(f"model saved to {model_path}")


if __name__ == "__main__":
    main()

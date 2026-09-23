# Decoupled critic-target ablation of v24 / historical job8254: the actor always
# uses the original boundary-aware GAE with gae_lambda=.95 by default. Networks,
# initialization, reward normalization, fused Adam and global clipping are intact.
# vapo uses critic lambda=1 with final-observation/time-limit and rollout-tail
# bootstraps, preserving the baseline's continuing-task boundary semantics.
# episode_mc instead treats BOTH termination and truncation as terminal and
# fits discounted rewards with no bootstrap, only where the next episode boundary
# is observed inside this rollout. A rollout may start mid-episode: every sampled
# transition through the final observed boundary has a complete future suffix;
# the unfinished suffix after that boundary is excluded separately per environment.
# No missing episode prefix is needed, no rewards cross autoresets, and no
# cross-policy replay, forced reset or additional environment step is introduced.
# This strict arm changes the critic's objective to finite-episode returns, unlike
# the continuing baseline; actor GAE retains its original time-limit bootstraps.
# gae retains the original v24 critic targets for behavioral-fidelity comparisons.
import json
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
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
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """whether to use deterministic cuDNN behavior"""
    cuda: bool = True
    """CUDA is required; CPU model execution is unsupported"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """also export the agent state dict; a full final learning checkpoint is always saved"""

    # Historical job8254 PPO settings, not the frozen v7 file's CLI defaults.
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments, including staggered-start warmup"""
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
    """the actor GAE lambda; unchanged by critic_target"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization; off in every matched arm"""
    clip_coef: float = 0.2
    """lower policy ratio clipping coefficient, also the optional MSE value clipping radius"""
    clip_coef_upper: float = 0.2
    """upper policy ratio clipping coefficient"""
    clip_vloss: bool = False
    """optional v7 MSE value clipping; unsupported for either symlog objective"""
    critic_loss: Literal["mse", "symlog_mean", "symlog_mse"] = "mse"
    """scalar critic objective; all value APIs decode to normalized-return units"""
    critic_target: Literal["gae", "vapo", "episode_mc"] = "vapo"
    """original GAE, boundary-bootstrapped lambda=1, or complete episodic suffixes"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for global gradient clipping"""
    grad_clip: Literal["global", "none"] = "global"
    """global v7 clipping or completely unmodified gradients"""
    target_kl: float | None = None
    """the optional v7 last-minibatch KL threshold, checked after each epoch"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
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


def symlog(x):
    """Signed log1p in the input's dtype, with no clipping or saturation."""
    return x.sign() * torch.log1p(x.abs())


def symexp(z):
    """Decode scalar symlog coordinates with expm1, without range clipping."""
    return z.sign() * torch.expm1(z.abs())


def compute_training_targets(
    gae_fn, rewards, values, terminations, truncations, truncation_values,
    tail_value, gamma, gae_lambda, critic_target,
):
    """Separate actor GAE from critic returns without changing actor boundaries.

    The Boolean mask has shape (T, N). In episode_mc, it selects exactly the
    transitions with a next termination/truncation inside this rollout. Unfinished
    tails have placeholder partial returns but must never enter a critic reduction.
    """
    if critic_target not in {"gae", "vapo", "episode_mc"}:
        raise ValueError("critic_target must be gae, vapo, or episode_mc")
    actor_advantages, original_returns = gae_fn(
        rewards, values, terminations, truncations, truncation_values, tail_value,
        gamma, gae_lambda,
    )
    # Own actor targets across the next compiled GAE call and optimizer graphs.
    actor_advantages = actor_advantages.clone()
    if critic_target == "gae":
        return actor_advantages, original_returns, torch.ones_like(rewards, dtype=torch.bool)
    if critic_target == "vapo":
        _, critic_returns = gae_fn(
            rewards, values, terminations, truncations, truncation_values, tail_value,
            gamma, 1.0,
        )
        critic_mask = torch.ones_like(rewards, dtype=torch.bool)
    else:
        boundaries = terminations.bool() | truncations.bool()
        zero_values = torch.zeros_like(values)
        _, critic_returns = gae_fn(
            rewards, zero_values, boundaries.to(terminations.dtype), truncations,
            zero_values, torch.zeros_like(tail_value), gamma, 1.0,
        )
        critic_mask = boundaries.flip(0).cumsum(dim=0, dtype=torch.int32).flip(0) > 0
    return actor_advantages, critic_returns, critic_mask


def _masked_mean(values, mask=None):
    """Fixed-shape reduction; an empty selection has a finite zero mean."""
    if mask is None:
        return values.mean()
    return torch.where(mask, values, 0.0).sum() / mask.sum().clamp_min(1)


def _masked_variance(values, mask):
    mean = _masked_mean(values, mask)
    centered = torch.where(mask, values - mean, 0.0)
    return _masked_mean(centered.square(), mask)


def _masked_extrema(values, mask):
    present = mask.any()
    lower = torch.where(mask, values, float("inf")).min()
    upper = torch.where(mask, values, -float("inf")).max()
    return torch.where(present, lower, 0.0), torch.where(present, upper, 0.0)


def _masked_explained_variance(prediction, targets, mask):
    target_variance = _masked_variance(targets, mask)
    score = 1.0 - _masked_variance(targets - prediction, mask) / target_variance
    return torch.where(target_variance == 0, float("nan"), score)


def scalar_value_loss(prediction, targets, mode, mask=None):
    """Scalar objective averaged only over selected detached return targets.

    For symlog_mean, dL/dz = symexp(z) - y, divided by the selected count.
    Excluded coordinates/targets are removed before nonlinear arithmetic, so an
    empty mask produces differentiable finite zero even with invalid placeholders.
    """
    targets = targets.detach()
    if mask is not None:
        mask = mask.bool()
        prediction = torch.where(mask, prediction, 0.0)
        targets = torch.where(mask, targets, 0.0)
    if mode == "mse":
        return 0.5 * _masked_mean((prediction - targets) ** 2, mask)
    if mode == "symlog_mse":
        return 0.5 * _masked_mean((prediction - symlog(targets)) ** 2, mask)
    if mode == "symlog_mean":
        magnitude = prediction.abs()
        target_magnitude = targets.abs()
        potential = torch.expm1(magnitude) - magnitude
        conjugate = (target_magnitude + 1.0) * torch.log1p(target_magnitude) - target_magnitude
        return _masked_mean(potential - targets * prediction + conjugate, mask)
    raise ValueError("critic_loss must be mse, symlog_mean, or symlog_mse")


def apply_gradient_clipping(parameters, max_norm, mode):
    """Preserve v7 global clipping or leave every gradient untouched."""
    if mode == "global":
        return nn.utils.clip_grad_norm_(parameters, max_norm)
    if mode == "none":
        return None
    raise ValueError("grad_clip must be global or none")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualMLP(nn.Module):
    """Unit residual directions with variance-restored branches and readouts."""

    def __init__(self, observation_dim, output_dim, output_std):
        super().__init__()

        def stage(in_dim):
            return nn.Sequential(init_situglu_branch(SiTUGLUBranch(in_dim, 64)))

        self.first = stage(observation_dim)
        self.second = stage(64)
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))
        self.readout_gain = nn.Parameter(torch.ones(output_dim))

    def forward(self, x):
        h = F.normalize(self.first(x), p=2, dim=-1)
        branch = F.normalize(self.second(8.0 * h), p=2, dim=-1)
        return self.readout_gain * self.head(8.0 * F.normalize(h + branch, p=2, dim=-1))


class ResidualHostMirror:
    """Compose fused FP32 stage mirrors without changing shared implementations.

    Stage outputs are permanent buffers; copy the first before calling the
    second, and add into our own permanent buffer before evaluating the head.
    """

    def __init__(self, actor, num_envs):
        self.first = make_host_mirror(actor.first, num_envs)
        self.second = make_host_mirror(actor.second, num_envs)
        self.head = make_host_mirror(actor.head, num_envs)
        self.hidden = np.empty((num_envs, 64), dtype=np.float32)
        self.branch = np.empty_like(self.hidden)
        self.squared = np.empty_like(self.hidden)
        self.norm = np.empty((num_envs, 1), dtype=np.float32)
        self.actor = actor
        self.readout_gain = np.empty(actor.readout_gain.numel(), dtype=np.float32)
        self.refresh()

    def refresh(self):
        self.first.refresh()
        self.second.refresh()
        self.head.refresh()
        np.copyto(self.readout_gain, self.actor.readout_gain.detach().cpu().numpy())

    def _normalize(self, values):
        np.multiply(values, values, out=self.squared)
        np.sum(self.squared, axis=1, keepdims=True, out=self.norm)
        np.sqrt(self.norm, out=self.norm)
        # Match F.normalize's zero-vector definition, including its epsilon.
        np.maximum(self.norm, np.float32(1e-12), out=self.norm)
        np.divide(values, self.norm, out=values)

    def __call__(self, observations):
        np.copyto(self.hidden, self.first(observations))
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.branch)
        np.copyto(self.branch, self.second(self.branch))
        self._normalize(self.branch)
        np.add(self.hidden, self.branch, out=self.hidden)
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.hidden)
        output = self.head(self.hidden)
        np.multiply(output, self.readout_gain, out=output)
        return output


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        self.critic_loss = "mse" if args is None else args.critic_loss
        if self.critic_loss not in {"mse", "symlog_mean", "symlog_mse"}:
            raise ValueError("critic_loss must be mse, symlog_mean, or symlog_mse")
        if args is not None and args.clip_vloss and self.critic_loss != "mse":
            raise ValueError("clip_vloss is only supported with critic_loss=mse")
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
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        # Keep v7's RNG consumption, initialization order and readout gains in ALL arms.
        self.critic = ResidualMLP(observation_dim, 1, output_std=1.0)
        self.actor = ResidualMLP(observation_dim, 2 * self.action_dim, output_std=0.01)

    @torch.no_grad()
    def normalize_matrices(self):
        """Match nGPT's matrix axes without changing Adam moments or biases."""
        for trunk in (self.actor, self.critic):
            for stage in (trunk.first, trunk.second):
                branch = stage[0]
                for weight, dim in ((branch.gate.weight, 1),
                                    (branch.up.weight, 1),
                                    (branch.down.weight, 0)):
                    weight.div_(torch.linalg.vector_norm(weight, dim=dim, keepdim=True))
            weight = trunk.head[0].weight
            weight.div_(torch.linalg.vector_norm(weight, dim=1, keepdim=True))

    def get_value(self, x):
        """Return decoded V in normalized-reward discounted-return units."""
        prediction = self.critic(x)
        return prediction if self.critic_loss == "mse" else symexp(prediction)

    def get_policy_and_value(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.get_value(x)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, x, action=None):
        """Public API uses physical actions and decoded V; training keeps native samples."""
        alpha, beta, value = self.get_policy_and_value(x)
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
        return action, logprob, entropy, value


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args,
             critic_mask=None):
    """Original PPO actor loss; only critic reductions use the optional mask."""
    if args.clip_vloss and args.critic_loss != "mse":
        raise ValueError("clip_vloss is only supported with critic_loss=mse")
    alpha, beta = (F.softplus(agent.actor(observations)) + 1.0).chunk(2, dim=-1)
    newvalue = agent.critic(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio < 1 - args.clip_coef) | (ratio > 1 + args.clip_coef_upper)).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef_upper)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        if critic_mask is not None:
            critic_mask = critic_mask.bool()
            newvalue = torch.where(critic_mask, newvalue, 0.0)
            returns = torch.where(critic_mask, returns, 0.0)
            old_values = torch.where(critic_mask, old_values, 0.0)
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * _masked_mean(torch.max(v_loss_unclipped, (v_clipped - returns) ** 2), critic_mask)
    else:
        v_loss = scalar_value_loss(newvalue, returns, args.critic_loss, critic_mask)
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac))
    return loss, metrics


def validate_args(args):
    if args.critic_target not in {"gae", "vapo", "episode_mc"}:
        raise ValueError("critic_target must be gae, vapo, or episode_mc")
    if args.critic_loss not in {"mse", "symlog_mean", "symlog_mse"}:
        raise ValueError("critic_loss must be mse, symlog_mean, or symlog_mse")
    if args.grad_clip not in {"global", "none"}:
        raise ValueError("grad_clip must be global or none")
    if args.clip_vloss and args.critic_loss != "mse":
        raise ValueError("clip_vloss is only supported with critic_loss=mse")
    if args.grad_clip == "global" and (not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0):
        raise ValueError("global max_grad_norm must be finite and positive")
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
        metrics_file = resources.enter_context(open(f"runs/{run_name}/metrics.jsonl", "w", buffering=1))
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); FP32; native-action storage; host actor mirror")
        writer.add_text("architecture", "v7 residual SiTU-GLU; unit-L2 matrices and residual directions; sqrt(width) branch/readout scale; learned output gains; no learned mixing")
        writer.add_text("critic_objective", f"{args.critic_loss}; gradient clipping={args.grad_clip}; normalized rewards and decoded GAE/V; identical v7 initialization, not identical decoded initial values across coordinates")
        writer.add_text("critic_target", f"{args.critic_target}; actor GAE lambda={args.gae_lambda}; vapo retains continuing bootstraps, episode_mc fits finite-episode complete suffixes only")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        parameters = tuple(agent.parameters())
        optimizer = optim.Adam(parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        actor_parameters = tuple(agent.actor.parameters())
        critic_parameters = tuple(agent.critic.parameters())
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and decoded values for the entire uploaded rollout."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values, critic_mask):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args,
                            critic_mask=critic_mask)

        @torch.no_grad()
        def critic_statistics(observations, returns, critic_mask):
            """One boundary-only critic forward; no extra forward in any PPO loss."""
            coordinate = agent.critic(observations).flatten()
            decoded = coordinate if args.critic_loss == "mse" else symexp(coordinate)
            error = decoded - returns
            decoded_min, decoded_max = _masked_extrema(decoded, critic_mask)
            coordinate_min, coordinate_max = _masked_extrema(coordinate, critic_mask)
            return {
                "critic/decoded_mse": _masked_mean(error.square(), critic_mask),
                "critic/value_bias": _masked_mean(error, critic_mask),
                "critic/postupdate_ev": _masked_explained_variance(decoded, returns, critic_mask),
                "critic/decoded_mean": _masked_mean(decoded, critic_mask),
                "critic/decoded_min": decoded_min,
                "critic/decoded_max": decoded_max,
                "critic/decoded_finite_fraction": _masked_mean(torch.isfinite(decoded).float(), critic_mask),
                "critic/coordinate_mean": _masked_mean(coordinate, critic_mask),
                "critic/coordinate_min": coordinate_min,
                "critic/coordinate_max": coordinate_max,
                "critic/coordinate_abs_max": torch.maximum(coordinate_min.abs(), coordinate_max.abs()),
                "critic/coordinate_finite_fraction": _masked_mean(torch.isfinite(coordinate).float(), critic_mask),
            }

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            critic_statistics = torch.compile(critic_statistics, fullgraph=True,
                                              options={"triton.cudagraphs": False})
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = ResidualHostMirror(agent.actor, args.num_envs)
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
            episode_returns = []
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
                        episode_returns.append(episode_return)
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs = rollout_statistics(b_obs, b_native)
                # Every bootstrap and value passed to GAE is decoded. Reward
                # normalization is unchanged; symlog is never applied to GAE.
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns, critic_mask = compute_training_targets(
                    gae_fn, batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda, args.critic_target,
                )
                b_advantages = advantages.flatten()
                b_returns = returns.flatten().clone()
                b_critic_mask = critic_mask.flatten()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                            b_critic_mask[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        selected_preclip_norm = apply_gradient_clipping(parameters, args.max_grad_norm, args.grad_clip)
                        optimizer.step()
                        normalize_matrices()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve v7's optional last-minibatch KL check after an epoch.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            # Additional norms/critic diagnostics run once per logging boundary,
            # never synchronize or inspect optimizer scalars inside an update.
            with torch.no_grad():
                critic_logged = critic_statistics(b_obs, b_returns, b_critic_mask)
                target_min, target_max = _masked_extrema(b_returns, b_critic_mask)
                actor_grad_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in actor_parameters]))
                critic_grad_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in critic_parameters]))
                postclip_norm = torch.sqrt(actor_grad_norm.square() + critic_grad_norm.square())
                preclip_norm = selected_preclip_norm if args.grad_clip == "global" else postclip_norm
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": _masked_explained_variance(b_values, b_returns, b_critic_mask),
                "critic/preupdate_mse": _masked_mean((b_values - b_returns).square(), b_critic_mask),
                "critic/target_mean": _masked_mean(b_returns, b_critic_mask),
                "critic/target_std": _masked_variance(b_returns, b_critic_mask).sqrt(),
                "critic/target_min": target_min, "critic/target_max": target_max,
                "critic/target_finite_fraction": _masked_mean(torch.isfinite(b_returns).float(), b_critic_mask),
                "critic/target_fraction": b_critic_mask.float().mean(),
                "critic/target_count": b_critic_mask.sum(),
                "critic/advantage_mean": b_advantages.mean(),
                "critic/advantage_std": b_advantages.std(unbiased=False),
                "grad/total_preclip_norm": preclip_norm, "grad/total_postclip_norm": postclip_norm,
                "grad/actor_postclip_norm": actor_grad_norm, "grad/critic_postclip_norm": critic_grad_norm,
                **critic_logged,
            })
            now = time.perf_counter()
            logged.update({
                "charts/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "charts/SPS": float(global_step / (now - start_time)),
                "charts/interval_SPS": float((global_step - interval_step) / (now - interval_start)),
                "charts/completed_episodes": float(len(episode_returns)),
            })
            if episode_returns:
                logged["charts/rollout_episodic_return"] = sum(episode_returns) / len(episode_returns)
            for phase, timing in timer.summary().items():
                logged[f"timing/{phase}_s"] = float(timing["total_s"])
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            # Persist even a failing boundary. Undefined explained variance and
            # nonfinite diagnostics serialize as null, never nonstandard JSON NaN.
            record = {name: value if math.isfinite(value) else None for name, value in logged.items()}
            record.update(step=global_step, iteration=iteration, critic_target=args.critic_target)
            metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
            if any(not math.isfinite(value) for name, value in logged.items()
                   if name not in {"losses/explained_variance", "critic/postupdate_ev"}):
                raise FloatingPointError("nonfinite PPO learner metrics; see metrics.jsonl")
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        checkpoint_path = f"runs/{run_name}/final_learning_checkpoint.pt"
        temporary_path = checkpoint_path + ".tmp"
        # Learning/auditing state only: the shared vector environment does not
        # serialize live MuJoCo state, so this is NOT an exact environment resume.
        checkpoint = {
            "version": 1, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
            "steps": global_step, "iteration": args.num_iterations, "args": asdict(args),
            "exact_environment_resume": False,
            "resume_note": "Learning state only; live environment and episode-wrapper state are not serialized.",
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "policy_sampler_state": sampler.bit_generator.state,
            "shuffle_generator_state": shuffle_generator.get_state(),
            "observation_normalizer": {
                "means": obs_norm.means.copy(), "variances": obs_norm.variances.copy(),
                "counts": obs_norm.counts.copy(), "epsilon": obs_norm.epsilon, "clip": obs_norm.clip,
            },
            "reward_normalizer": {
                "means": rew_norm.means.copy(), "variances": rew_norm.variances.copy(),
                "counts": rew_norm.counts.copy(), "returns": rew_norm.returns.copy(),
                "gamma": rew_norm.gamma, "epsilon": rew_norm.epsilon, "clip": rew_norm.clip,
            },
            "next_observation": next_obs_np.copy(), "suppress_mask": suppress.copy(),
        }
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        print(f"learning checkpoint saved to {checkpoint_path}")
        if args.save_model:
            # Weights-only export needs the checkpoint's Args and normalizers for
            # inference; legacy ppo_eval would silently use the wrong coordinates.
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

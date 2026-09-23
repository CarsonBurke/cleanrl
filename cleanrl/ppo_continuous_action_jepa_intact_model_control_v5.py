"""INTACT-inspired model-driven direct control with a shared stochastic action law.

All four modes train the same world model and deploy the same zero-search actor.
Only the prescriber's actor objective and the critic's imagined regression differ.
The behavior state is [normalized physical observation, previous physical action].
Frozen v4 remains unchanged; this is an experimental model-based control ablation,
not a reproduction of INTACT or evidence that imagined returns imply real control.
"""
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
from torch.func import functional_call
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.lejepa import ActionEncoder, FeedForward, MLP, SIGReg
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
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
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 32
    num_steps: int = 1024
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    """optional epoch-level stopping for PPO-actor modes only"""

    control_mode: Literal["actor", "critic", "both", "none"] = "both"
    """model-driven actor, imagined critic extension, both, or matched PPO control"""
    imagination_horizon: int = 3
    imagination_batch_size: int = 128
    goal_horizon: int = 8
    local_nll_coef: float = 0.1
    goal_nll_coef: float = 0.05
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256
    ssl_learning_rate: float = 5e-5
    ssl_weight_decay: float = 1e-3

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


class ActionConditionedMLP(nn.Module):
    """The v4 AdaLN-zero feed-forward dynamics, without attention or tokens."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(64, 64)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(64, 3 * 64))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.output_norm = nn.LayerNorm(64)

    def forward(self, latent, condition):
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        hidden = self.norm(latent) * (1.0 + scale) + shift
        return self.output_norm(latent + gate * self.mlp(hidden))


def frozen_call(module, *inputs):
    """Freeze weights, not input derivatives; dictionaries trace into full graphs.

    Detached tensors still alias the live parameter storage. All backward work
    therefore finishes before any of the three optimizers mutates parameters.
    """
    parameters = {name: value.detach() for name, value in module.named_parameters()}
    buffers = {name: value.detach() for name, value in module.named_buffers()}
    return functional_call(module, (parameters, buffers), inputs)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta control requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta control requires finite, strictly ordered action bounds")
        if args.control_mode not in {"actor", "critic", "both", "none"}:
            raise ValueError("invalid control_mode")
        if min(args.sigreg_num_proj, args.sigreg_proj_chunk) <= 0:
            raise ValueError("SIGReg projection counts must be positive")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        self.observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.input_dim = self.observation_dim + self.action_dim
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(self.observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
        )
        self.action_encoder = ActionEncoder(self.action_dim, 64)
        self.dynamics = ActionConditionedMLP()
        # The AdaLN block ends in LayerNorm; the LeWM output projector restores
        # an unconstrained prediction in the encoder's unbounded coordinates.
        self.pred_proj = MLP(64, 64, 64)
        self.sigreg = SIGReg(knots=17, num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk)
        self.reward_head = nn.Sequential(
            layer_init(nn.Linear(64 + self.action_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.continuation_head = nn.Sequential(
            layer_init(nn.Linear(64 + self.action_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.previous_action_embedding = nn.Sequential(
            layer_init(nn.Linear(self.action_dim, 64)), nn.Tanh(),
        )
        self.action_law = nn.Sequential(
            layer_init(nn.Linear(4 * 64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        self.prescriber = nn.Sequential(
            layer_init(nn.Linear(64 + self.action_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64 + self.action_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def parameter_groups(self):
        """Three disjoint owners: world, prescriber only, and value FFN only."""
        world_modules = (self.encoder, self.action_encoder, self.dynamics, self.pred_proj,
                         self.reward_head, self.continuation_head,
                         self.previous_action_embedding, self.action_law)
        world = tuple(parameter for module in world_modules for parameter in module.parameters())
        return world, tuple(self.prescriber.parameters()), tuple(self.critic.parameters())

    def parameter_counts(self):
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in self.named_children()
        }
        counts["deployed_actor"] = sum(counts[name] for name in (
            "encoder", "prescriber", "previous_action_embedding", "action_law",
        ))
        counts["inference"] = counts["deployed_actor"] + counts["critic"]
        counts["total"] = sum(parameter.numel() for parameter in self.parameters())
        counts["training_only"] = counts["total"] - counts["inference"]
        return counts

    def encode(self, observations):
        return self.encoder(observations[..., :self.observation_dim])

    def law_from_intent(self, latent, intent, previous_action, *, frozen=False):
        call = frozen_call if frozen else lambda module, *inputs: module(*inputs)
        previous = call(self.previous_action_embedding, previous_action)
        logits = call(self.action_law, torch.cat((latent, intent, latent * intent, previous), dim=-1))
        return (F.softplus(logits) + 1.0).chunk(2, dim=-1)

    def policy_from_latent(self, latent, previous_action, *, frozen=True):
        intent = self.prescriber(torch.cat((latent, previous_action), dim=-1))
        return self.law_from_intent(latent, intent, previous_action, frozen=frozen)

    def predict_next(self, latent, physical_action, *, frozen=False):
        call = frozen_call if frozen else lambda module, *inputs: module(*inputs)
        centered_action = 2.0 * (physical_action - self.action_low) / self.action_scale - 1.0
        predicted = call(self.dynamics, latent, call(self.action_encoder, centered_action))
        return call(self.pred_proj, predicted)

    def predict_reward(self, latent, physical_action, *, frozen=False):
        features = torch.cat((latent, physical_action), dim=-1)
        output = frozen_call(self.reward_head, features) if frozen else self.reward_head(features)
        return output.squeeze(-1)

    def continuation_logits(self, latent, physical_action, *, frozen=False):
        features = torch.cat((latent, physical_action), dim=-1)
        output = frozen_call(self.continuation_head, features) if frozen else self.continuation_head(features)
        return output.squeeze(-1)

    def predict_continuation(self, latent, physical_action, *, frozen=False):
        return self.continuation_logits(latent, physical_action, frozen=frozen).sigmoid()

    def value_from_latent(self, latent, previous_action, *, frozen=False):
        features = torch.cat((latent, previous_action), dim=-1)
        return frozen_call(self.critic, features) if frozen else self.critic(features)

    def direct_policy(self, observations):
        return self.policy_from_latent(self.encode(observations).detach(),
                                       observations[..., self.observation_dim:])

    def get_value(self, observations):
        return self.value_from_latent(self.encode(observations).detach(),
                                      observations[..., self.observation_dim:])

    def get_policy_and_value(self, observations):
        latent = self.encode(observations).detach()
        previous = observations[..., self.observation_dim:]
        alpha, beta = self.policy_from_latent(latent, previous)
        return alpha, beta, self.value_from_latent(latent, previous)

    def action_logprob(self, alpha, beta, native_action):
        native_action = native_action.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, observations, action=None):
        """Physical-action API; rollout training stores the native Beta sample."""
        alpha, beta, value = self.get_policy_and_value(observations)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((observations.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(observations.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS,
            )
        entropy = (Beta(alpha, beta, validate_args=False).entropy() + self.log_action_scale).sum(-1)
        return action, self.action_logprob(alpha, beta, native), entropy, value


class HostIntactActor:
    """Compose four native mirrors with permanent contiguous NumPy staging.

    Returns Beta logits, not concentrations. No neural torch CPU forward, model
    rollout, candidate search, or per-environment-step CUDA synchronization.
    """

    def __init__(self, agent, num_rows):
        self.num_rows = num_rows
        self.in_features = agent.input_dim
        self.out_features = 2 * agent.action_dim
        self.observation_dim = agent.observation_dim
        self.encoder = make_host_mirror(agent.encoder, num_rows)
        self.prescriber = make_host_mirror(agent.prescriber, num_rows)
        self.previous_action_embedding = make_host_mirror(agent.previous_action_embedding, num_rows)
        self.action_law = make_host_mirror(agent.action_law, num_rows)
        self._mirrors = (self.encoder, self.prescriber, self.previous_action_embedding, self.action_law)
        if not all(mirror.fused for mirror in self._mirrors):
            raise RuntimeError("INTACT control requires native host graph mirrors")
        self.fused = True
        self._observations = np.empty((num_rows, agent.observation_dim), dtype=np.float32)
        self._previous = np.empty((num_rows, agent.action_dim), dtype=np.float32)
        self._context = np.empty((num_rows, 64 + agent.action_dim), dtype=np.float32)
        self._law_input = np.empty((num_rows, 4 * 64), dtype=np.float32)

    def refresh(self):
        for mirror in self._mirrors:
            mirror.refresh()

    def __call__(self, observations):
        if observations.shape != (self.num_rows, self.in_features) or observations.dtype != np.float32:
            raise ValueError(f"expected float32 augmented observations {(self.num_rows, self.in_features)}")
        self._observations[:] = observations[:, :self.observation_dim]
        self._previous[:] = observations[:, self.observation_dim:]
        latent = self.encoder(self._observations)
        self._context[:, :64] = latent
        self._context[:, 64:] = self._previous
        intent = self.prescriber(self._context)
        self._law_input[:, :64] = latent
        self._law_input[:, 64:128] = intent
        np.multiply(latent, intent, out=self._law_input[:, 128:192])
        self._law_input[:, 192:] = self.previous_action_embedding(self._previous)
        return self.action_law(self._law_input)


def world_loss(agent, observations, native_actions, next_observations, goal_observations,
               rewards, terminations, args):
    """Attached JEPA/local intent, detached goal endpoint, and grounded heads."""
    latent = agent.encode(observations)
    following = agent.encode(next_observations)
    with torch.no_grad():
        goal = agent.encode(goal_observations)
    previous = observations[..., agent.observation_dim:]
    physical = agent.action_low + agent.action_scale * native_actions
    prediction = F.mse_loss(agent.predict_next(latent, physical), following)
    regularization = agent.sigreg(torch.stack((latent, following)))
    local_alpha, local_beta = agent.law_from_intent(latent, following - latent, previous)
    goal_alpha, goal_beta = agent.law_from_intent(latent, goal - latent, previous)
    local_nll = -agent.action_logprob(local_alpha, local_beta, native_actions).mean()
    goal_nll = -agent.action_logprob(goal_alpha, goal_beta, native_actions).mean()
    reward_loss = F.mse_loss(agent.predict_reward(latent.detach(), physical), rewards.reshape(-1))
    continuation_loss = F.binary_cross_entropy_with_logits(
        agent.continuation_logits(latent.detach(), physical), 1.0 - terminations.reshape(-1).float(),
    )
    total = (prediction + args.sigreg_weight * regularization + args.local_nll_coef * local_nll
             + args.goal_nll_coef * goal_nll + reward_loss + continuation_loss)
    with torch.no_grad():
        metrics = {
            "world/prediction_loss": prediction.detach(),
            "world/sigreg_loss": regularization.detach(),
            "world/local_nll": local_nll.detach(),
            "world/goal_nll": goal_nll.detach(),
            "world/reward_loss": reward_loss.detach(),
            "world/continuation_loss": continuation_loss.detach(),
            "world/latent_std": following.std(dim=0, correction=0).mean(),
            "world/local_intent_norm": (following - latent).norm(dim=-1).mean(),
            "world/goal_intent_norm": (goal - latent).norm(dim=-1).mean(),
            "world/total_loss": total.detach(),
        }
    return total, metrics


def actor_objective(agent, observations, args):
    """Pathwise stochastic H-step return; only prescriber weights receive grads."""
    observations = observations[:args.imagination_batch_size]
    latent = agent.encode(observations).detach()
    previous = observations[..., agent.observation_dim:].detach()
    returns = latent.new_zeros(latent.shape[0])
    discount = torch.ones_like(returns)
    for _ in range(args.imagination_horizon):
        alpha, beta = agent.policy_from_latent(latent, previous)
        native = Beta(alpha, beta, validate_args=False).rsample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        physical = agent.action_low + agent.action_scale * native
        returns = returns + discount * agent.predict_reward(latent, physical, frozen=True)
        discount = discount * args.gamma * agent.predict_continuation(latent, physical, frozen=True)
        latent = agent.predict_next(latent, physical, frozen=True)
        previous = physical
    returns = returns + discount * agent.value_from_latent(latent, previous, frozen=True).squeeze(-1)
    return -returns.mean()


def imagined_critic_loss(agent, observations, args):
    """Current-policy imagined lambda regression; no encoder/policy target grads."""
    observations = observations[:args.imagination_batch_size]
    with torch.no_grad():
        latent = agent.encode(observations)
        previous = observations[..., agent.observation_dim:]
        states, previous_actions, rewards, discounts, next_values = [], [], [], [], []
        for _ in range(args.imagination_horizon):
            states.append(latent)
            previous_actions.append(previous)
            alpha, beta = agent.policy_from_latent(latent, previous)
            native = Beta(alpha, beta, validate_args=False).rsample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            physical = agent.action_low + agent.action_scale * native
            rewards.append(agent.predict_reward(latent, physical))
            discounts.append(args.gamma * agent.predict_continuation(latent, physical))
            latent = agent.predict_next(latent, physical)
            previous = physical
            next_values.append(agent.value_from_latent(latent, previous).squeeze(-1))
        target = next_values[-1]
        targets = []
        for step in range(args.imagination_horizon - 1, -1, -1):
            target = rewards[step] + discounts[step] * (
                (1.0 - args.gae_lambda) * next_values[step] + args.gae_lambda * target
            )
            targets.append(target)
        target_batch = torch.stack(targets[::-1]).flatten().detach()
        latent_batch = torch.stack(states).flatten(0, 1).detach()
        previous_batch = torch.stack(previous_actions).flatten(0, 1).detach()
    predictions = agent.value_from_latent(latent_batch, previous_batch).squeeze(-1)
    return 0.5 * F.mse_loss(predictions, target_batch)


def augment_transition_observations(next_obs, final_obs, physical_actions, terminations, truncations,
                                   *, next_out, final_out):
    """Fill separate policy/reset and factual successor buffers on the host.

    Observations are already normalized, physical actions are not. Factual
    terminal/time-limit states retain the executed action; reset contexts are
    zero. Both outputs must be independent preallocated augmented-state arrays.
    """
    rows = next_out.shape[0]
    observations = next_obs.reshape(rows, -1)
    observation_dim = observations.shape[1]
    physical = physical_actions.reshape(rows, -1)
    final_out[:, :observation_dim] = final_obs.reshape(rows, -1)
    final_out[:, observation_dim:] = physical
    next_out[:, :observation_dim] = observations
    next_out[:, observation_dim:] = physical
    next_out[np.logical_or(terminations, truncations), observation_dim:] = 0.0
    return next_out, final_out


def make_goal_observations(next_observations, terminations, truncations, horizon):
    """Furthest factual endpoint within H transitions, stopping at every reset.

    Boundary step itself is inclusive: its factual final observation is usable,
    but its reset observation and all following episode states are not.
    """
    if horizon <= 0:
        raise ValueError("goal horizon must be positive")
    if next_observations.ndim != 3 or next_observations.shape[0] == 0:
        raise ValueError("next_observations must have nonempty [time, env, augmented_state] shape")
    if terminations.shape != next_observations.shape[:2] or truncations.shape != terminations.shape:
        raise ValueError("boundary flags must have [time, env] shape")
    steps, environments = terminations.shape
    time_indices = torch.arange(steps, device=next_observations.device)[:, None]
    boundaries = torch.where(terminations.bool() | truncations.bool(), time_indices, steps - 1)
    nearest_boundary = boundaries.flip(0).cummin(dim=0).values.flip(0)
    endpoints = torch.minimum(time_indices + min(horizon, steps) - 1, nearest_boundary)
    environment_indices = torch.arange(environments, device=next_observations.device)[None, :]
    return next_observations[endpoints, environment_indices]


def training_loss(agent, observations, native_actions, old_logprobs, advantages, returns,
                  old_values, next_observations, goal_observations, rewards, terminations, args):
    """One backward graph, three owners; every mode keeps identical world losses."""
    alpha, beta, newvalue = agent.get_policy_and_value(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = agent.action_logprob(alpha, beta, native_actions)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1).mean()
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = torch.maximum(-advantages * ratio,
                            -advantages * ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef)).mean()
    newvalue = newvalue.flatten()
    if args.clip_vloss:
        clipped = old_values + (newvalue - old_values).clamp(-args.clip_coef, args.clip_coef)
        real_value_loss = 0.5 * torch.maximum((newvalue - returns).square(), (clipped - returns).square()).mean()
    else:
        real_value_loss = 0.5 * (newvalue - returns).square().mean()
    actor_loss = (actor_objective(agent, observations, args) if args.control_mode in {"actor", "both"}
                  else pg_loss - args.ent_coef * entropy)
    imagined_loss = (imagined_critic_loss(agent, observations, args) if args.control_mode in {"critic", "both"}
                     else observations.new_zeros(()))
    critic_loss = args.vf_coef * (real_value_loss + imagined_loss)
    model_loss, metrics = world_loss(agent, observations, native_actions, next_observations,
                                     goal_observations, rewards, terminations, args)
    with torch.no_grad():
        metrics.update({
            "losses/actor_objective": actor_loss.detach(),
            "losses/policy_loss": pg_loss.detach(),
            "losses/value_loss": real_value_loss.detach(),
            "losses/imagined_value_loss": imagined_loss.detach(),
            "losses/critic_objective": critic_loss.detach(),
            "losses/entropy": entropy.detach(),
            "losses/old_approx_kl": (-logratio).mean(),
            "losses/approx_kl": ((ratio - 1.0) - logratio).mean(),
            "losses/clipfrac": ((ratio - 1.0).abs() > args.clip_coef).float().mean(),
        })
        if args.control_mode in {"actor", "both"}:
            metrics["model/imagined_actor_return"] = -actor_loss.detach()
    return model_loss + actor_loss + critic_loss, metrics


def validate_args(args):
    if min(args.total_timesteps, args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("timestep, environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.control_mode not in {"actor", "critic", "both", "none"}:
        raise ValueError("invalid control_mode")
    if min(args.imagination_horizon, args.imagination_batch_size, args.goal_horizon,
           args.sigreg_num_proj, args.sigreg_proj_chunk) <= 0:
        raise ValueError("imagination, goal and SIGReg counts must be positive")
    for name in ("local_nll_coef", "goal_nll_coef", "sigreg_weight", "ssl_weight_decay",
                 "ent_coef", "vf_coef", "clip_coef"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("learning_rate", "ssl_learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("gamma", "gae_lambda"):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.target_kl is not None:
        if not np.isfinite(args.target_kl) or args.target_kl <= 0:
            raise ValueError("target_kl must be finite and positive")
        if args.control_mode in {"actor", "both"}:
            raise ValueError("target_kl is a PPO stopping rule, unavailable for a model-driven actor")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the model-control learner requires CUDA")
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
    run_name = f"{args.env_id}__{args.exp_name}_{args.control_mode}__{args.seed}__{int(time.time())}"
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
        writer.add_text("policy", "Beta direct control; FP32; augmented previous-action context; native host actor; zero candidates")
        writer.add_text("model_metrics", "Imagined returns and training errors are diagnostics, not control-quality evidence; real episodic returns are logged separately.")
        writer.add_scalar("control/deployed_candidate_budget", 0, 0)
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        world_parameters, actor_parameters, critic_parameters = agent.parameter_groups()
        world_optimizer = optim.AdamW(world_parameters, lr=args.ssl_learning_rate,
                                      weight_decay=args.ssl_weight_decay, fused=True)
        actor_optimizer = optim.Adam(actor_parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        critic_optimizer = optim.Adam(critic_parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        optimizers = (world_optimizer, actor_optimizer, critic_optimizer)
        parameter_groups = (world_parameters, actor_parameters, critic_parameters)
        counts = agent.parameter_counts()
        writer.add_text("parameter_counts", str(counts))
        print(f"parameter_counts={counts}")
        for name, count in counts.items():
            writer.add_scalar(f"parameters/{name}", count, 0)

        def rollout_statistics(observations, native, following, rewards, terminations):
            """Behavior likelihood/value and held-out-in-time error before fitting."""
            latent = agent.encode(observations)
            next_latent = agent.encode(following)
            previous = observations[..., agent.observation_dim:]
            alpha, beta = agent.policy_from_latent(latent, previous)
            physical = agent.action_low + agent.action_scale * native
            diagnostics = {
                "model/preupdate_prediction_mse": F.mse_loss(agent.predict_next(latent, physical), next_latent),
                "model/preupdate_persistence_mse": F.mse_loss(latent, next_latent),
                "model/preupdate_reward_mse": F.mse_loss(agent.predict_reward(latent, physical), rewards),
                "model/preupdate_continuation_bce": F.binary_cross_entropy_with_logits(
                    agent.continuation_logits(latent, physical), 1.0 - terminations.float(),
                ),
            }
            return (agent.value_from_latent(latent, previous).flatten(),
                    agent.action_logprob(alpha, beta, native), diagnostics)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values,
                       following, goals, rewards, terminations):
            return training_loss(agent, observations, native, old_logprobs, advantages, returns,
                                 old_values, following, goals, rewards, terminations, args)

        value_model = agent.get_value
        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        augmented_shape = (agent.input_dim,)
        host_actor = HostIntactActor(agent, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(
            args.num_steps, args.num_envs, augmented_shape, device,
            non_blocking=args.non_blocking_transfers,
            fields={"observations": augmented_shape, "native_actions": (agent.action_dim,),
                    "next_observations": augmented_shape},
        )
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, augmented_shape)
        resources.callback(bootstraps.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        gradient_norms = torch.empty((max_updates, 3), device=device)
        metric_sums = None
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)
        warmup_input = np.zeros((args.num_envs, agent.input_dim), dtype=np.float32)

        def warmup_action(observations):
            # Warmup is unrecorded. Recorded behavior starts from a zero context.
            warmup_input[:, :agent.observation_dim] = observations.reshape(args.num_envs, -1)
            return act(warmup_input)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=warmup_action, horizon=horizon,
                                    phase_offsets=phases, seed=args.seed)
            physical_obs, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            physical_obs, global_step = obs_norm.normalize(raw_obs), 0
        # Two policy-state buffers prevent overwriting the state being staged.
        next_obs_np = np.zeros((args.num_envs, agent.input_dim), dtype=np.float32)
        next_obs_np[:, :agent.observation_dim] = physical_obs.reshape(args.num_envs, -1)
        spare_obs = np.empty_like(next_obs_np)
        transition_obs = np.empty_like(next_obs_np)
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                learning_rate = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
                actor_optimizer.param_groups[0]["lr"] = learning_rate
                critic_optimizer.param_groups[0]["lr"] = learning_rate
            bootstraps.reset()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    policy_physical, factual_physical = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    augment_transition_observations(
                        policy_physical, factual_physical, host_action, terms, truncs,
                        next_out=spare_obs, final_out=transition_obs,
                    )
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    # Native sampler buffers are copied now, before the next act.
                    transfer.push(step, reward, terms, truncs, observations=obs_step,
                                  native_actions=native, next_observations=transition_obs)
                    next_obs_np, spare_obs = spare_obs, next_obs_np
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
                next_observations = batch.fields["next_observations"]
                b_next_obs = next_observations.flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_rewards = batch.rewards.flatten()
                b_terminations = batch.terminations.flatten()
                b_goals = make_goal_observations(next_observations, batch.terminations,
                                                batch.truncations, args.goal_horizon).flatten(0, 1)
                b_values, b_logprobs, preupdate_metrics = rollout_statistics(
                    b_obs, b_native, b_next_obs, b_rewards, b_terminations,
                )
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
            if metric_sums is not None:
                for value in metric_sums.values():
                    value.zero_()
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                            b_next_obs[indices], b_goals[indices], b_rewards[indices], b_terminations[indices],
                        )
                        # No optimizer may mutate aliased frozen weights before this.
                        loss.backward()
                        for column, parameters in enumerate(parameter_groups):
                            gradient_norms[updates, column] = nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                        for optimizer in optimizers:
                            optimizer.step()
                        if metric_sums is None:
                            metric_sums = {name: torch.zeros_like(value) for name, value in metrics.items()}
                        for name, value in metrics.items():
                            metric_sums[name].add_(value.detach())
                        last_kl = metrics["losses/approx_kl"].detach()
                        updates += 1
                        del loss, metrics
                    # Optional PPO-only synchronization, after the full epoch.
                    if args.target_kl is not None and last_kl > args.target_kl:
                        break

            with timer.span("host_refresh"):
                host_actor.refresh()
            metric_values = {name: value / updates for name, value in metric_sums.items()}
            metric_values.update(preupdate_metrics)
            metric_values["losses/explained_variance"] = explained_variance(b_values, b_returns)
            for column, owner in enumerate(("world", "actor", "critic")):
                metric_values[f"gradients/{owner}_norm"] = gradient_norms[:updates, column].mean()
            logged = gather_metrics(metric_values)
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite model-control learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/ssl_learning_rate", world_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save({
                "model": agent.state_dict(),
                "args": vars(args),
                "obs_norm": {
                    "means": torch.from_numpy(obs_norm.means.copy()),
                    "variances": torch.from_numpy(obs_norm.variances.copy()),
                    "counts": torch.from_numpy(obs_norm.counts.copy()),
                    "epsilon": obs_norm.epsilon,
                    "clip": obs_norm.clip,
                },
            }, model_path)
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

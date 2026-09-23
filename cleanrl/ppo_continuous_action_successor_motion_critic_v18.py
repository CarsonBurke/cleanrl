# Critic-only physical motion successor supervision for ordinary Beta PPO v18.
# The canonical actor and scalar critic retain their architecture, initialization,
# reward GAE, clipped losses and Adam owners. Two zero-initialized heads use only
# the critic's hidden state: state occupancy and an action-centered degree-two
# conditional approximation. No latent actor, auxiliary actor loss, action-model
# gradient, reward replacement or successor readout enters the policy objective.
# PPO fits the same successor heads with a detached critic trunk; one_step uses
# factual next-motion targets; successor uses normalized discounted occupancy.
# Frozen rollout coordinate RMS changes only the auxiliary loss metric, never
# physical feature/output/bootstrap units. All prefit teachers remain immutable.
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

from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import device_minibatches, explained_variance, gather_metrics, get_gae_fn
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
MOTION_DIM = 45
ACTION_BASIS_DIM = 27
# minibatch_objective returns a flat tensor in exactly this order.
MINIBATCH_METRIC_NAMES = (
    "losses/policy_loss", "losses/value_loss", "successor/state_loss",
    "successor/action_loss", "successor/auxiliary_loss", "losses/entropy",
    "losses/old_approx_kl", "losses/approx_kl", "losses/clipfrac",
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    save_model: bool = False
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
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
    aux_mode: Literal["ppo", "one_step", "successor"] = "successor"
    """PPO detaches the auxiliary trunk; other modes train only the critic trunk."""
    env_backend: Literal["native"] = "native"
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


class TaskFFN(nn.Module):
    """The unchanged canonical independent 64x64 Tanh task network."""

    def __init__(self, input_dim, output_dim, output_std):
        super().__init__()
        self.first = nn.Sequential(layer_init(nn.Linear(input_dim, 64)), nn.Tanh())
        self.second = nn.Sequential(layer_init(nn.Linear(64, 64)), nn.Tanh())
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))

    def forward(self, x):
        return self.head(self.second(self.first(x)))


class HostPolicy:
    """One fused native FP32 actor graph, refreshed once after learner updates."""

    def __init__(self, agent, num_rows):
        self.fused = make_host_mirror(nn.Sequential(
            *agent.actor.first, *agent.actor.second, *agent.actor.head,
        ), num_rows)

    def refresh(self):
        self.fused.refresh()

    def __call__(self, observations):
        return self.fused(observations)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        if args.aux_mode not in {"ppo", "one_step", "successor"}:
            raise ValueError("aux_mode must be ppo, one_step or successor")
        self.aux_mode = args.aux_mode
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta PPO requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        if self.action_dim != 6 or tuple(envs.single_observation_space.shape) != (17,):
            raise ValueError("motion critic requires HalfCheetah's 17 observations and 6 actions")
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        # Initialize both complete task networks BEFORE forked auxiliary RNG.
        self.actor = TaskFFN(observation_dim, 2 * self.action_dim, 0.01)
        self.critic = TaskFFN(observation_dim, 1, 1.0)
        with torch.random.fork_rng(devices=[]):
            self.state_head = nn.Linear(64, MOTION_DIM)
            self.action_head = nn.Linear(64, MOTION_DIM * ACTION_BASIS_DIM)
            nn.init.zeros_(self.state_head.weight)
            nn.init.zeros_(self.state_head.bias)
            nn.init.zeros_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)

    def parameter_groups(self):
        """Exclusive actor, original scalar critic, and auxiliary-head owners."""
        return (tuple(self.actor.parameters()), tuple(self.critic.parameters()),
                (*self.state_head.parameters(), *self.action_head.parameters()))

    def parameter_counts(self):
        actor, critic, auxiliary = (sum(parameter.numel() for parameter in group)
                                    for group in self.parameter_groups())
        return {"actor_ffn": actor, "critic_ffn": critic, "auxiliary_heads": auxiliary,
                "state_head": sum(parameter.numel() for parameter in self.state_head.parameters()),
                "action_head": sum(parameter.numel() for parameter in self.action_head.parameters()),
                "actor_inference": actor, "inference": actor + critic,
                "total": actor + critic + auxiliary}

    def _critic_hidden(self, observations):
        return self.critic.second(self.critic.first(observations.detach()))

    def _auxiliary_from_hidden(self, hidden, basis):
        if self.aux_mode == "ppo":
            hidden = hidden.detach()
        state = self.state_head(hidden)
        coefficients = self.action_head(hidden).unflatten(-1, (MOTION_DIM, ACTION_BASIS_DIM))
        action = (coefficients * basis.detach().unsqueeze(-2)).sum(-1)
        return state, action

    def auxiliary_predictions(self, observations, basis):
        """Physical state and action-residual predictions; neither inputs attach.

        PPO trains the same two heads/targets but cannot backpropagate their
        losses into the scalar critic. No mode accesses the actor here.
        """
        return self._auxiliary_from_hidden(self._critic_hidden(observations), basis)

    def get_value(self, observations):
        return self.critic(observations.detach())

    def get_policy_and_value(self, observations):
        logits = self.actor(observations.detach())
        alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.get_value(observations)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action.detach()) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, observations, action=None):
        """Public API uses physical actions; training stores unit-interval samples."""
        alpha, beta, value = self.get_policy_and_value(observations)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((observations.shape[0],) + self.action_shape)
        else:
            native = ((action.detach().reshape(observations.shape[0], -1) - self.action_low)
                      / self.action_scale).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, value


@torch.no_grad()
def motion_features(raw_next_observations):
    """45 physical features in the design's exact order, without normalization.

    Height; sin/cos of seven angles; nine velocities; then the 21 upper
    triangular products (including diagonal) of the six joint velocities.
    """
    if raw_next_observations.shape[-1] != 17:
        raise ValueError("motion features require raw HalfCheetah observations with 17 coordinates")
    observations = raw_next_observations.detach()
    angles = observations[..., 1:8]
    joints = observations[..., 11:17]
    rows, columns = torch.triu_indices(6, 6, device=observations.device)
    return torch.cat((observations[..., :1], angles.sin(), angles.cos(), observations[..., 8:17],
                      joints[..., rows] * joints[..., columns]), dim=-1)


@torch.no_grad()
def beta_action_basis(alpha, beta, native_actions):
    """Frozen orthonormal degree-two basis under the behavior product-Beta law.

    Columns are six standardized actions, six centered orthonormal quadratics,
    then 15 off-diagonal z_i*z_j in torch.triu_indices order. This is a compact
    conditional approximation, not arbitrary Q or log-score geometry. Sampling's
    finite-precision epsilon clipping only approximates the ideal-Beta moments.
    """
    if alpha.shape != beta.shape or alpha.shape != native_actions.shape or alpha.shape[-1] != 6:
        raise ValueError("alpha, beta and native actions must have matching (...,6) shapes")
    alpha, beta, native_actions = alpha.detach(), beta.detach(), native_actions.detach()
    total = alpha + beta
    product = alpha * beta
    variance = product / (total.square() * (total + 1))
    z = (native_actions - alpha / total) / variance.sqrt()
    skew = 2 * (beta - alpha) * (total + 1).sqrt() / ((total + 2) * product.sqrt())
    # Algebraically kurtosis - 1 - skew^2, without subtracting similar moments.
    quadratic_variance = (2 * ((alpha + 1) / alpha) * ((beta + 1) / beta)
                          * (total / (total + 2)).square() * (total / (total + 3)))
    quadratic = (z.square() - 1 - skew * z) / quadratic_variance.sqrt()
    rows, columns = torch.triu_indices(6, 6, offset=1, device=z.device)
    return torch.cat((z, quadratic, z[..., rows] * z[..., columns]), dim=-1).detach()


def factual_next_observations(raw_next_obs, terminations, truncations, infos):
    """Preserve raw physical finals, rejecting missing or invalid boundary data.

    The common no-boundary path returns the original array without copying.
    Boundary substitution never mutates the autoreset observation array or infos.
    Normalized bootstrap observations are computed separately by VectorObsNorm.
    """
    if raw_next_obs.ndim != 2 or raw_next_obs.shape[-1] != 17:
        raise ValueError("raw next observations must have shape (N,17)")
    count = raw_next_obs.shape[0]
    if np.shape(terminations) != (count,) or np.shape(truncations) != (count,):
        raise ValueError("termination and truncation masks must have shape (N,)")
    if not (np.count_nonzero(terminations) or np.count_nonzero(truncations)):
        return raw_next_obs
    boundaries = np.flatnonzero(np.logical_or(terminations, truncations))
    finals, masks = infos.get("final_observation"), infos.get("_final_observation")
    if finals is None or len(finals) != count:
        raise RuntimeError("completed transition missing final_observation")
    if masks is not None and np.shape(masks) != (count,):
        raise RuntimeError("invalid final_observation mask shape")
    factual = raw_next_obs.copy()
    for index in boundaries:
        if (masks is not None and not masks[index]) or finals[index] is None:
            raise RuntimeError(f"completed environment {index} has no final observation")
        final = np.asarray(finals[index])
        if final.shape != raw_next_obs.shape[1:]:
            raise RuntimeError(f"completed environment {index} has invalid final observation shape")
        if not np.isfinite(final).all():
            raise FloatingPointError(f"completed environment {index} has nonfinite final observation")
        factual[index] = final
    return factual


@torch.no_grad()
def successor_lambda_targets(features, predictions, next_predictions, terminations, truncations,
                             gamma, gae_lambda, gae_fn):
    """Normalized occupancy TD(lambda) through shared explicit-next vector GAE.

    Inputs have shape (T,N,45), masks (T,N). Immediate features are multiplied
    by (1-gamma); state predictions and factual next-state predictions remain in
    those same occupancy units. Terminations cut bootstrap and trace, truncations
    bootstrap factual finals and cut trace. The rollout tail bootstraps once.
    """
    if features.ndim != 3 or features.shape[-1] != MOTION_DIM:
        raise ValueError("successor features must have shape (T,N,45)")
    if predictions.shape != features.shape or next_predictions.shape != features.shape:
        raise ValueError("current and factual next predictions must match features")
    if terminations.shape != features.shape[:2] or truncations.shape != features.shape[:2]:
        raise ValueError("termination and truncation masks must have shape (T,N)")
    steps, environments, channels = features.shape
    terms = terminations.detach().to(features.dtype).unsqueeze(-1).expand(
        steps, environments, channels,
    ).reshape(steps, -1)
    truncs = truncations.detach().to(features.dtype).unsqueeze(-1).expand(
        steps, environments, channels,
    ).reshape(steps, -1)
    _, targets = gae_fn(
        ((1 - gamma) * features.detach()).reshape(steps, -1),
        predictions.detach().reshape(steps, -1), terms, truncs,
        next_predictions.detach().reshape(steps, -1), gamma, gae_lambda,
    )
    return targets.reshape_as(features).detach().clone()


@torch.no_grad()
def auxiliary_targets(features, predictions, next_predictions, terminations, truncations, args, gae_fn):
    """Return frozen state targets, old-state residual targets and separate RMS.

    One-step uses physical next features directly. PPO and successor share the
    exact same occupancy target construction; only trunk gradient routing differs.
    RMS values alter losses alone, never any head output or bootstrap coordinate.
    """
    if features.ndim != 3 or features.shape[-1] != MOTION_DIM:
        raise ValueError("auxiliary features must have shape (T,N,45)")
    if predictions.shape != features.shape or next_predictions.shape != features.shape:
        raise ValueError("current and factual next predictions must match features")
    if terminations.shape != features.shape[:2] or truncations.shape != features.shape[:2]:
        raise ValueError("termination and truncation masks must have shape (T,N)")
    if args.aux_mode == "one_step":
        state_targets = features.detach().clone()
    elif args.aux_mode in {"ppo", "successor"}:
        state_targets = successor_lambda_targets(
            features, predictions, next_predictions, terminations, truncations,
            args.gamma, args.gae_lambda, gae_fn,
        )
    else:
        raise ValueError("aux_mode must be ppo, one_step or successor")
    action_targets = (state_targets - predictions.detach()).detach().clone()
    state_scale = (state_targets.reshape(-1, MOTION_DIM).square().mean(0) + 1e-8).sqrt().detach().clone()
    action_scale = (action_targets.reshape(-1, MOTION_DIM).square().mean(0) + 1e-8).sqrt().detach().clone()
    return state_targets, action_targets, state_scale, action_scale


def critic_loss(predictions, returns, old_values, args):
    """The unchanged scalar clipped task-value objective."""
    values = predictions.squeeze(-1)
    returns, old_values = returns.detach(), old_values.detach()
    squared_error = (values - returns).square()
    if args.clip_vloss:
        clipped = old_values + (values - old_values).clamp(-args.clip_coef, args.clip_coef)
        squared_error = torch.maximum(squared_error, (clipped - returns).square())
    return 0.5 * squared_error.mean()


def minibatch_objective(agent, args, observations, native_actions, old_logprob, advantages, returns,
                        old_values, basis, state_targets, action_targets, state_scale, action_scale):
    """One backward, one critic-trunk forward, no policy auxiliary gradient.

    Returns (scalar total loss, flat detached tensor). Metric order is the public
    MINIBATCH_METRIC_NAMES constant. Every action, basis, scale and target teacher
    is detached even when callers accidentally provide gradient-bearing tensors.
    """
    logits = agent.actor(observations.detach())
    alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
    hidden = agent._critic_hidden(observations)
    predictions = agent.critic.head(hidden)
    state, action = agent._auxiliary_from_hidden(hidden, basis)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions.detach()) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1).mean()
    logratio = newlogprob - old_logprob.detach()
    ratio = logratio.exp()
    advantages = advantages.detach()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = torch.maximum(-advantages * ratio,
                            -advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    value_loss = critic_loss(predictions, returns, old_values, args)
    state_loss = 0.5 * ((state - state_targets.detach()) / state_scale.detach()).square().mean()
    action_loss = 0.5 * ((action - action_targets.detach()) / action_scale.detach()).square().mean()
    auxiliary_loss = 0.5 * (state_loss + action_loss)
    objective = pg_loss - args.ent_coef * entropy + args.vf_coef * (value_loss + auxiliary_loss)
    with torch.no_grad():
        metrics = torch.stack((
            pg_loss.detach(), value_loss.detach(), state_loss.detach(), action_loss.detach(),
            auxiliary_loss.detach(), entropy.detach(), (-logratio).mean(),
            ((ratio - 1) - logratio).mean(), ((ratio - 1).abs() > args.clip_coef).float().mean(),
        ))
    return objective, metrics


@torch.no_grad()
def rollout_statistics(agent, observations, native_actions, raw_next_observations):
    """Independent pre-update snapshots; no CUDA-graph output reuse or actor AD."""
    logits = agent.actor(observations.detach())
    alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
    basis = beta_action_basis(alpha, beta, native_actions)
    hidden = agent._critic_hidden(observations)
    state, action = agent._auxiliary_from_hidden(hidden, basis)
    data = {
        "values": agent.critic.head(hidden).squeeze(-1),
        "logprobs": agent.action_logprob(alpha, beta, native_actions),
        "alpha": alpha, "beta": beta, "basis": basis,
        "state_predictions": state, "action_predictions": action,
        "features": motion_features(raw_next_observations),
    }
    return {name: value.detach().clone() for name, value in data.items()}


@torch.no_grad()
def bootstrap_statistics(agent, observations):
    """Scalar V and state SF at normalized factual next observations only."""
    hidden = agent._critic_hidden(observations)
    return agent.critic.head(hidden).squeeze(-1).detach().clone(), agent.state_head(hidden).detach().clone()


@torch.no_grad()
def auxiliary_diagnostics(state_predictions, action_predictions, state_targets, action_targets,
                          state_scale, action_scale):
    """Fresh pre-fit errors, never post-fit or causal counterfactual guarantees.

    State-only and state+action errors share state-target RMS, making relative
    improvement a matched prediction comparison. The residual-fit error instead
    uses its own frozen action-target RMS, exactly as in the action-head loss.
    """
    state_error = ((state_predictions - state_targets) / state_scale).square().mean()
    joint_error = ((state_predictions + action_predictions - state_targets) / state_scale).square().mean()
    action_error = ((action_predictions - action_targets) / action_scale).square().mean()
    return {
        "successor/prefit_state_normalized_mse": state_error.detach(),
        "successor/prefit_state_action_normalized_mse": joint_error.detach(),
        "successor/prefit_relative_action_improvement": ((state_error - joint_error) / (state_error + 1e-8)).detach(),
        "successor/prefit_action_normalized_mse": action_error.detach(),
        "successor/state_scale_min": state_scale.min().detach(),
        "successor/state_scale_max": state_scale.max().detach(),
        "successor/action_scale_min": action_scale.min().detach(),
        "successor/action_scale_max": action_scale.max().detach(),
    }


def optimizer_step(optimizers, parameters, max_grad_norm, norms):
    """Clip each exclusive owner separately after the one combined backward."""
    for index, group in enumerate(parameters):
        norms[index].copy_(nn.utils.clip_grad_norm_(group, max_grad_norm))
    for optimizer in optimizers:
        optimizer.step()


def validate_args(args):
    if args.env_id != "HalfCheetah-v4" or args.env_backend != "native":
        raise ValueError("v18 matched fixed-task runs require native HalfCheetah-v4")
    if not args.cuda:
        raise ValueError("v18 requires CUDA learner execution")
    if args.track:
        raise ValueError("v18 matched runs use local TensorBoard, not WandB")
    if args.aux_mode not in {"ppo", "one_step", "successor"}:
        raise ValueError("invalid aux_mode")
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs, args.env_threads) <= 0:
        raise ValueError("environment, rollout, minibatch, epoch and thread counts must be positive")
    if not (0 < args.gamma < 1 and 0 <= args.gae_lambda <= 1):
        raise ValueError("gamma must be in (0,1), lambda in [0,1]")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not (0 < args.clip_coef < 1) or not np.isfinite(args.vf_coef) or args.vf_coef <= 0:
        raise ValueError("invalid clipping or critic coefficient")
    if not np.isfinite(args.ent_coef) or args.ent_coef < 0:
        raise ValueError("ent_coef must be finite and nonnegative")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches:
        raise ValueError("num_minibatches must divide batch_size exactly")
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size < 2 or args.total_timesteps < args.batch_size:
        raise ValueError("need at least two examples per minibatch and a complete rollout")
    return args


def make_training_env(args, run_name):
    return make_mujoco_vector_env(args.env_id, args.num_envs, backend=args.env_backend,
                                  num_threads=min(args.env_threads, args.num_envs),
                                  capture_video=False, run_name=run_name)


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
    with ExitStack() as resources:
        writer = SummaryWriter(f"runs/{run_name}")
        resources.callback(writer.close)
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        contract = {
            "version": 18, "method": "successor_motion_critic_v18", "aux_mode": args.aux_mode,
            "gamma": args.gamma, "gae_lambda": args.gae_lambda,
            "architecture": "canonical_independent_actor_and_scalar_critic_tanh64x64; "
                            "two_zero_linear_auxiliary_heads_on_critic_hidden_only; native_host_actor",
            "initialization": "canonical_v17_actor_scalar_critic_then_CPU_fork_rng_zero_auxiliary_heads",
            "actor": "ordinary_clipped_Beta_PPO; no_auxiliary_gradient_or_latent_input_or_action_model_gradient",
            "reward_advantage": "original_scalar_V_and_normalized_real_reward_explicit_next_GAE_only",
            "feature_dimension": MOTION_DIM, "action_basis_dimension": ACTION_BASIS_DIM,
            "feature_order": "raw_height[0],sin_angles[1:8],cos_angles[1:8],velocities[8:17],"
                             "joint_velocities[11:17]_upper_products_including_diagonal_triu_indices_order",
            "feature_source": "raw_factual_next_observation_including_validated_finals; never_reset_or_normalized",
            "basis_order": "z[0:6],(z^2-1-skew*z)/sqrt(kurtosis-1-skew^2)[0:6],"
                           "z_i*z_j_i<j_triu_indices_order; frozen_behavior_product_Beta_analytic_moments",
            "head_order": "state[45]; action_coefficients[45,27]_feature_major_contracted_with_frozen_basis",
            "target": "one_step:physical_next_features; ppo_and_successor:normalized_occupancy_TD(lambda)_"
                      "with_immediate_(1-gamma)*phi_and_frozen_state_SF_bootstraps",
            "boundary_semantics": "termination_cuts_bootstrap_and_trace; truncation_uses_factual_final_"
                                  "bootstrap_and_cuts_trace; rollout_tail_bootstraps_once",
            "action_target": "same_frozen_state_target_minus_frozen_old_state_SF",
            "loss_scales": "separate_frozen_coordinate_sqrt(mean(target^2)+1e-8); loss_only; "
                           "no_running_normalization_or_output_rescaling_or_bootstrap_coordinate_changes",
            "auxiliary_loss": "(0.5*mean((state_error/state_scale)^2)+"
                              "0.5*mean((action_error/action_scale)^2))/2",
            "objective": "standard_pg-ent_coef*entropy+vf_coef*(standard_clipped_value_loss+auxiliary_loss)",
            "ppo_control": "same_successor_targets_heads_and_compute; detach_critic_hidden_for_auxiliary_only",
            "optimizer_ownership": "exclusive_actor_scalar_critic_auxiliary_heads_fused_Adam; "
                                   "same_annealed_LR_and_eps1e-5; each_owner_clips_at_max_grad_norm",
            "teacher_freezing": "pre_update_alpha_beta_basis_values_logprobs_state_and_action_predictions_"
                                "factual_next_values_and_state_SF_reward_GAE_targets_and_scales",
            "diagnostics": "fresh_prefit_state_and_state_plus_action_normalized_target_MSE; "
                           "relative_action_prediction_improvement; separate_auxiliary_losses; scalar_EV",
            "metric_order": MINIBATCH_METRIC_NAMES,
            "limitations": "fixed_physical_feature_hypothesis_not_advantage_kernel_optimum; "
                           "degree_two_conditional_approximation_not_arbitrary_action_Q; "
                           "on_policy_prediction_not_causal_counterfactual_guarantee; "
                           "known_scalar_reward_does_not_need_multidimensional_SF_for_expressivity; "
                           "critic_gradient_conflict_and_history_dependent_normalization; "
                           "finite_precision_clipped_Beta_sampling_approximates_ideal_basis_orthogonality; "
                           "no_policy_family_latent_actor_or_finetuning_mechanism",
        }
        writer.add_text("motion_critic_contract", str(contract))
        parameters = agent.parameter_groups()
        optimizers = tuple(optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in parameters)
        counts = agent.parameter_counts()
        writer.add_text("parameter_counts", str(counts))
        print(f"parameter_counts={counts}")
        for name, count in counts.items():
            writer.add_scalar(f"parameters/{name}", count, 0)

        def statistics_model(observations, native, raw_next):
            return rollout_statistics(agent, observations, native, raw_next)

        def bootstrap_model(observations):
            return bootstrap_statistics(agent, observations)

        def objective_model(observations, native, old_logprob, advantages, returns, old_values,
                            basis, state_targets, action_targets, state_scale, action_scale):
            return minibatch_objective(agent, args, observations, native, old_logprob, advantages, returns,
                                       old_values, basis, state_targets, action_targets, state_scale, action_scale)

        diagnostics_model = auxiliary_diagnostics
        if args.compile:
            statistics_model = graph_compile(statistics_model)
            bootstrap_model = graph_compile(bootstrap_model)
            diagnostics_model = graph_compile(diagnostics_model)
            objective_model = torch.compile(objective_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode, explicit_next_values=True)
        obs_shape = envs.single_observation_space.shape
        host_actor = HostPolicy(agent, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        fields = {"observations": obs_shape, "native_actions": (agent.action_dim,),
                  "next_observations": obs_shape, "raw_next_observations": obs_shape}
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers, fields=fields)
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        updates_per_rollout = args.update_epochs * args.num_minibatches
        gradient_norms = torch.empty((updates_per_rollout, 3), device=device)
        metric_sums = torch.zeros(len(MINIBATCH_METRIC_NAMES), device=device)
        total_updates = 0
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=warmup_action, horizon=horizon, phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                for optimizer in optimizers:
                    optimizer.param_groups[0]["lr"] = (1 - (iteration - 1) / args.num_iterations) * args.learning_rate
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    raw_factual_next = factual_next_observations(raw_obs, terms, truncs, infos)
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, factual_next_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native,
                                  next_observations=factual_next_obs, raw_next_observations=raw_factual_next)
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
                # graph_compile disables CUDA-graph output reuse; both models own
                # cloned snapshots, immune to all subsequent optimizer updates.
                old_data = statistics_model(b_obs, b_native, batch.fields["raw_next_observations"].flatten(0, 1))
                next_values, next_state = bootstrap_model(batch.fields["next_observations"].flatten(0, 1))
                advantages, returns = gae_fn(
                    batch.rewards, old_data["values"].view(args.num_steps, args.num_envs),
                    batch.terminations, batch.truncations, next_values.view(args.num_steps, args.num_envs),
                    args.gamma, args.gae_lambda,
                )
                # Snapshot scalar GAE BEFORE invoking the same shared callable for
                # vector occupancy targets, including when compiled graphs reuse memory.
                b_advantages, b_returns = advantages.flatten().detach().clone(), returns.flatten().detach().clone()
                state_targets, action_targets, state_scale, action_scale = auxiliary_targets(
                    old_data["features"].view(args.num_steps, args.num_envs, MOTION_DIM),
                    old_data["state_predictions"].view(args.num_steps, args.num_envs, MOTION_DIM),
                    next_state.view(args.num_steps, args.num_envs, MOTION_DIM),
                    batch.terminations, batch.truncations, args, gae_fn,
                )
                b_state_targets, b_action_targets = state_targets.flatten(0, 1), action_targets.flatten(0, 1)
            with timer.span("diagnostics"):
                diagnostic_metrics = diagnostics_model(
                    old_data["state_predictions"], old_data["action_predictions"], b_state_targets,
                    b_action_targets, state_scale, action_scale,
                )
            metric_sums.zero_()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        torch.compiler.cudagraph_mark_step_begin()
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        objective, metrics = objective_model(
                            b_obs[indices], b_native[indices], old_data["logprobs"][indices],
                            b_advantages[indices], b_returns[indices], old_data["values"][indices],
                            old_data["basis"][indices], b_state_targets[indices], b_action_targets[indices],
                            state_scale, action_scale,
                        )
                        objective.backward()
                        optimizer_step(optimizers, parameters, args.max_grad_norm, gradient_norms[updates])
                        metric_sums.add_(metrics.detach())
                        updates += 1
                host_actor.refresh()
            total_updates += updates
            metric_values = dict(zip(MINIBATCH_METRIC_NAMES, (metric_sums / updates).unbind()))
            metric_values.update({
                "losses/explained_variance": explained_variance(old_data["values"], b_returns),
                "gradients/actor_norm": gradient_norms[:, 0].mean(),
                "gradients/critic_norm": gradient_norms[:, 1].mean(),
                "gradients/auxiliary_norm": gradient_norms[:, 2].mean(),
            })
            metric_values.update(diagnostic_metrics)
            logged = gather_metrics(metric_values)
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            for name, value in {
                "ppo_steps": updates, "critic_steps": updates, "auxiliary_steps": updates,
                "ppo_steps_total": total_updates, "critic_steps_total": total_updates,
                "auxiliary_steps_total": total_updates,
                "ppo_examples": updates * args.minibatch_size, "critic_examples": updates * args.minibatch_size,
                "auxiliary_examples": updates * args.minibatch_size,
                "ppo_examples_total": total_updates * args.minibatch_size,
                "critic_examples_total": total_updates * args.minibatch_size,
                "auxiliary_examples_total": total_updates * args.minibatch_size,
                "ppo_minibatch_size": args.minibatch_size, "epochs": args.update_epochs,
            }.items():
                writer.add_scalar(f"updates/{name}", value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizers[0].param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (now - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save({
                "model": agent.state_dict(), "args": vars(args), "global_step": global_step,
                "parameter_counts": counts, "aux_mode": args.aux_mode,
                "motion_critic_contract": contract,
                "state_scale": state_scale.detach().cpu(), "action_scale": action_scale.detach().cpu(),
                "optimizer_steps": total_updates,
                "optimizers": {name: optimizer.state_dict()
                               for name, optimizer in zip(("actor", "critic", "auxiliary_heads"), optimizers)},
                "reward_norm": {
                    "returns": torch.from_numpy(rew_norm.returns.copy()),
                    "means": torch.from_numpy(rew_norm.means.copy()),
                    "variances": torch.from_numpy(rew_norm.variances.copy()),
                    "counts": torch.from_numpy(rew_norm.counts.copy()),
                    "gamma": rew_norm.gamma, "epsilon": rew_norm.epsilon, "clip": rew_norm.clip,
                },
                "obs_norm": {
                    "means": torch.from_numpy(obs_norm.means.copy()),
                    "variances": torch.from_numpy(obs_norm.variances.copy()),
                    "counts": torch.from_numpy(obs_norm.counts.copy()),
                    "epsilon": obs_norm.epsilon, "clip": obs_norm.clip,
                },
            }, model_path)
            print(f"model saved to {model_path}")


if __name__ == "__main__":
    main()

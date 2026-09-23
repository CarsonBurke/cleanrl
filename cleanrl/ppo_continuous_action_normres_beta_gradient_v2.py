# Beta gradient-control PPO v2, descended only from normres_indclip_v4.
# Separate Bellman value and gradient-control critics. Centered, Fisher-whitened
# Beta sufficient statistics + a state baseline are trained on projected BLOCK
# gradient second moments, not return MSE. Random parameter JVPs preserve the
# full actor's Euclidean gradient geometry without per-sample parameter Jacobians.
# Lagged controls retain the analytic expectation correction. Both actor arms
# share exact-KL backtracking. A frozen-policy, disjoint-data gate precedes RL.
# Evidence: mlq 7405, seed=1 fixed-policy gate FAILED after 3,686,016 steps.
# Held-out block variance ratio 1.00772, one-sided bootstrap upper95 1.01264;
# 6/16 rollout wins. State-only ablation ratio 0.98857 (exploratory).
# Preserve as a negative result, not an endorsed improvement; no RL escalation.
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
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache,
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
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets,
    episode_horizon,
    run_phase_warmup,
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
    learning_rate: float = 9.6e-3
    """32x the 3e-4 PPO default; one Adam covers actor and critic"""
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
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """raw GAE in the surrogate; no minibatch standardization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """toggle PPO value-loss clipping independently of gradient clipping"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum gradient norm, applied independently to actor and critic"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    reward_norm: bool = True
    """normalize/clip rewards before GAE; never normalize GAE return targets"""

    placement: Literal["pre", "post"] = "pre"
    """normalize branch inputs with an identity stream, or residual outputs"""
    norm_kind: Literal["layer", "rms"] = "rms"
    """non-affine centered LayerNorm or uncentered RMSNorm, epsilon 1e-5"""
    activation: Literal["lrelusq", "stiglu"] = "stiglu"
    """squared leaky-ReLU pair or parameter-matched SiTU-GLU branch"""

    actor_update: Literal["ordinary", "corrected"] = "ordinary"
    """matched-KL ordinary or analytically corrected residual PPO"""
    control_learning_rate: float = 3e-4
    """independent supervised gradient-control optimizer, not the actor/value LR"""
    train_projections: int = 4
    """fresh Rademacher parameter directions per training rollout"""
    eval_projections: int = 16
    """independent directions for out-of-rollout gradient diagnostics"""
    gradient_block_steps: int = 256
    """contiguous per-environment gradient blocks; preserve temporal covariance"""
    gate_checkpoint: str | None = None
    """our v1 checkpoint for a fixed-policy mechanism experiment, never actor finetuning"""
    gate_calibration_rollouts: int = 32
    """re-estimate reward scaling and policy value; old checkpoint lacks reward RMS"""
    gate_fit_rollouts: int = 64
    """fit only the control after freezing value and reward normalization"""
    gate_holdout_rollouts: int = 16
    """fresh rollouts with every network and normalizer frozen"""
    trust_kl: float = 0.02
    """both actor arms: maximum mean exact KL(old policy || candidate)"""
    max_backtracks: int = 12
    """halve an actor proposal this many times before restoring it and its Adam state"""
    gradient_diagnostic_interval: int = 16
    """rollout interval for independent full-rollout projected diagnostics"""

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


def beta_score_reference(alpha, beta):
    """Per-action log-statistic means and 2x2 Fisher Cholesky factors.

    All five tensors have (..., action_dim) shape. Cholesky computation uses
    double precision to avoid subtractive cancellation at concentrated policies;
    the retained rollout features remain in the policy's dtype.
    """
    a, b = alpha.double(), beta.double()
    total = a + b
    mean_a, mean_b = a.digamma() - total.digamma(), b.digamma() - total.digamma()
    common = torch.polygamma(1, total)
    va, vb, covariance = torch.polygamma(1, a) - common, torch.polygamma(1, b) - common, -common
    l00 = va.sqrt()
    l10 = covariance / l00
    l11 = (vb - l10.square()).sqrt()
    return tuple(t.to(alpha.dtype) for t in (mean_a, mean_b, l00, l10, l11))


def whiten_log_statistics(log_a, log_b, reference):
    mean_a, mean_b, l00, l10, l11 = reference
    first = (log_a - mean_a) / l00
    second = (log_b - mean_b - l10 * first) / l11
    return torch.cat((first, second), dim=-1)


def beta_score_features(native_actions, reference):
    white = whiten_log_statistics(native_actions.log(), torch.log1p(-native_actions), reference)
    return torch.cat((torch.ones_like(white[..., :1]), white), dim=-1)


def beta_score_means(alpha, beta, reference):
    total_digamma = (alpha + beta).digamma()
    white = whiten_log_statistics(alpha.digamma() - total_digamma, beta.digamma() - total_digamma, reference)
    return torch.cat((torch.ones_like(white[..., :1]), white), dim=-1)


def beta_kl_reference(alpha, beta):
    total = alpha + beta
    log_partition = alpha.lgamma() + beta.lgamma() - total.lgamma()
    total_digamma = total.digamma()
    return alpha, beta, log_partition, alpha.digamma() - total_digamma, beta.digamma() - total_digamma


def beta_kl(alpha, beta, reference):
    old_alpha, old_beta, old_partition, old_da, old_db = reference
    partition = alpha.lgamma() + beta.lgamma() - (alpha + beta).lgamma()
    return partition - old_partition + (old_alpha - alpha) * old_da + (old_beta - beta) * old_db


def control_inputs(observations, alpha, beta):
    # Policy context makes the estimator's changing conditional distribution explicit.
    return torch.cat((observations, alpha.detach().log(), beta.detach().log()), dim=-1)


class ValueCritic(nn.Module):
    def __init__(self, observation_dim, *, placement, norm_kind, activation):
        super().__init__()
        self.trunk = make_norm_residual_trunk(
            observation_dim, 64, placement=placement, norm_kind=norm_kind, activation=activation
        )
        self.value_head = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, observations):
        return self.value_head(self.trunk(observations))


class Agent(nn.Module):
    def __init__(self, envs, *, placement="pre", norm_kind="rms", activation="stiglu"):
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
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.critic = ValueCritic(observation_dim, placement=placement, norm_kind=norm_kind, activation=activation)
        self.actor = nn.Sequential(
            make_norm_residual_trunk(observation_dim, 64, placement=placement, norm_kind=norm_kind, activation=activation),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        # Keep the base actor/value initialization, and never let the variance
        # objective distort the Bellman value representation.
        with torch.random.fork_rng(devices=[]):
            self.control = nn.Sequential(
                make_norm_residual_trunk(
                    observation_dim + 2 * self.action_dim, 64,
                    placement=placement, norm_kind=norm_kind, activation=activation,
                ),
                layer_init(nn.Linear(64, 1 + 2 * self.action_dim), std=0.0),
            )

    def get_value(self, observations):
        return self.critic(observations)

    def get_policy_and_value(self, observations):
        alpha, beta = (F.softplus(self.actor(observations)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(observations)

    def action_logprob(self, alpha, beta, native_action):
        return (Beta(alpha, beta, validate_args=False).log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, observations, action=None):
        alpha, beta, value = self.get_policy_and_value(observations)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((observations.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(observations.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        distribution = Beta(alpha, beta, validate_args=False)
        return action, self.action_logprob(alpha, beta, native), (distribution.entropy() + self.log_action_scale).sum(-1), value


def sample_parameter_directions(actor, count, generator):
    return {
        name: torch.empty((count,) + tuple(parameter.shape), device=parameter.device, dtype=parameter.dtype)
        .bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        for name, parameter in actor.named_parameters()
    }


def projected_beta_statistics(logits, tangent_logits, native_actions, reference):
    """Return S_z[N,K] and B_z[N,K,F] from actor-logit JVPs[K,N,2D]."""
    mean_a, mean_b, l00, l10, l11 = reference
    da, db = (tangent_logits * logits.sigmoid()).chunk(2, dim=-1)
    score = ((native_actions.log() - mean_a) * da + (torch.log1p(-native_actions) - mean_b) * db).sum(-1)
    va, covariance, vb = l00.square(), l00 * l10, l10.square() + l11.square()
    dmean_a = va * da + covariance * db
    dmean_b = covariance * da + vb * db
    first = dmean_a / l00
    second = (dmean_b - l10 * first) / l11
    derivative = torch.cat((torch.zeros_like(first[..., :1]), first, second), dim=-1)
    return score.transpose(0, 1), derivative.transpose(0, 1)


def make_projection_function(actor):
    parameters = dict(actor.named_parameters())

    def project(observations, native_actions, reference, directions):
        def forward(parameters):
            return torch.func.functional_call(actor, parameters, (observations,))

        def one(direction):
            return torch.func.jvp(forward, (parameters,), (direction,))[1]

        tangents = torch.func.vmap(one)(directions)
        return projected_beta_statistics(actor(observations), tangents, native_actions, reference)

    return project


def block_gradients(projected_gradients, num_envs, block_steps):
    # RolloutTransfer flattens [time,env]. Never use PPO's shuffled indices here.
    return projected_gradients.reshape(-1, block_steps, num_envs, projected_gradients.shape[-1]).mean(1).flatten(0, 1)


def projected_estimators(advantages, features, coefficients, scores, mean_derivatives):
    prediction = (features * coefficients).sum(-1)
    ordinary = scores * advantages[:, None]
    baseline_only = scores * (advantages - coefficients[:, 0])[:, None]
    corrected = scores * (advantages - prediction)[:, None] + (mean_derivatives * coefficients[:, None, :]).sum(-1)
    return ordinary, baseline_only, corrected


def control_loss(control, inputs, advantages, features, scores, mean_derivatives, normalizer, num_envs, block_steps):
    coefficients = control(inputs.detach())
    _, _, corrected = projected_estimators(
        advantages.detach(), features.detach(), coefficients, scores.detach(), mean_derivatives.detach()
    )
    blocks = block_gradients(corrected, num_envs, block_steps)
    return 0.5 * blocks.square().mean() / normalizer.detach()


def projection_diagnostics(advantages, features, coefficients, scores, mean_derivatives, num_envs, block_steps):
    ordinary, baseline_only, corrected = projected_estimators(advantages, features, coefficients, scores, mean_derivatives)
    ordinary_blocks = block_gradients(ordinary, num_envs, block_steps)
    baseline_blocks = block_gradients(baseline_only, num_envs, block_steps)
    corrected_blocks = block_gradients(corrected, num_envs, block_steps)
    return {
        "gradient/ordinary_variance_trace": ordinary.var(0, unbiased=False).mean(),
        "gradient/corrected_variance_trace": corrected.var(0, unbiased=False).mean(),
        "gradient/ordinary_block_variance": ordinary_blocks.var(0, unbiased=False).mean(),
        "gradient/baseline_only_block_variance": baseline_blocks.var(0, unbiased=False).mean(),
        "gradient/corrected_block_variance": corrected_blocks.var(0, unbiased=False).mean(),
        "gradient/ordinary_block_second_moment": ordinary_blocks.square().mean(),
        "gradient/corrected_block_second_moment": corrected_blocks.square().mean(),
        "gradient/mean_difference_norm2": (corrected.mean(0) - ordinary.mean(0)).square().mean(),
    }


def value_loss(agent, observations, targets, old_values, args):
    values = agent.critic(observations).flatten()
    error = (values - targets).square()
    if args.clip_vloss:
        clipped = old_values + (values - old_values).clamp(-args.clip_coef, args.clip_coef)
        error = torch.maximum(error, (clipped - targets).square())
    return 0.5 * error.mean()


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, targets, old_values,
             features, reference, old_coefficients, args):
    alpha, beta = (F.softplus(agent.actor(observations)) + 1.0).chunk(2, dim=-1)
    distribution = Beta(alpha, beta, validate_args=False)
    logratio = agent.action_logprob(alpha, beta, native_actions) - old_logprobs
    ratio = logratio.exp()
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1).mean()
    coefficients = old_coefficients.detach()
    if args.actor_update == "corrected":
        residual = advantages - (features.detach() * coefficients).sum(-1)
        frozen_reference = tuple(t.detach() for t in reference)
        current_means = beta_score_means(alpha, beta, frozen_reference)
        # Subtract old expectation (the state baseline) only to center logging;
        # it has no actor derivative. Nonconstant old feature means are zero.
        correction = ((current_means * coefficients).sum(-1) - coefficients[:, 0]).mean()
    else:
        residual, correction = advantages, ratio.new_zeros(())
    pg_loss = torch.maximum(-residual * ratio, -residual * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean() - correction
    v_loss = value_loss(agent, observations, targets, old_values, args)
    with torch.no_grad():
        metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy.detach(), (-logratio).mean(),
                               (ratio - 1 - logratio).mean(), ((ratio - 1).abs() > args.clip_coef).float().mean(),
                               correction.detach()))
    return pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss, metrics


class ActorTrustRegion:
    """Backtrack only actor parameters; rejected proposals restore actor Adam too.

    The critic has independent parameters and clipping and always keeps its step.
    Accepted fractional proposals keep Adam moments: line search scales the proposed
    displacement, not the gradient estimator. Exact KL checks intentionally sync
    only at proposal boundaries, outside the compiled loss/optimizer computation.
    """
    def __init__(self, actor, optimizer, *, budget, max_backtracks):
        self.parameters = tuple(actor.parameters())
        self.optimizer = optimizer
        self.budget = budget
        self.max_backtracks = max_backtracks
        self.before = tuple(torch.empty_like(p) for p in self.parameters)
        self.delta = tuple(torch.empty_like(p) for p in self.parameters)
        self.moment_copies = []
        self.had_state = []

    @torch.no_grad()
    def snapshot(self):
        self.had_state = []
        if not self.moment_copies:
            self.moment_copies = [dict() for _ in self.parameters]
        for parameter, before, saved in zip(self.parameters, self.before, self.moment_copies):
            before.copy_(parameter)
            state = self.optimizer.state.get(parameter, {})
            self.had_state.append(bool(state))
            for key, value in state.items():
                if key not in saved:
                    saved[key] = torch.empty_like(value)
                saved[key].copy_(value)

    @torch.no_grad()
    def accept(self, measure_kl):
        for delta, parameter, before in zip(self.delta, self.parameters, self.before):
            torch.sub(parameter, before, out=delta)
        fraction = 1.0
        for backtracks in range(self.max_backtracks + 1):
            kl = measure_kl()
            if torch.isfinite(kl) and kl <= self.budget:
                return kl, fraction, backtracks
            if backtracks < self.max_backtracks:
                fraction *= 0.5
                for parameter, before, delta in zip(self.parameters, self.before, self.delta):
                    parameter.copy_(before).add_(delta, alpha=fraction)
        for parameter, before, saved, had_state in zip(
            self.parameters, self.before, self.moment_copies, self.had_state
        ):
            parameter.copy_(before)
            if had_state:
                for key, value in saved.items():
                    self.optimizer.state[parameter][key].copy_(value)
            else:
                self.optimizer.state.pop(parameter, None)
        return measure_kl(), 0.0, self.max_backtracks + 1


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.update_epochs, args.train_projections, args.eval_projections) <= 0:
        raise ValueError("positive rollout sizes, epochs, and projection counts required")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.num_minibatches != 1 or args.norm_adv:
        raise ValueError("preserve one full-rollout PPO minibatch and unnormalized advantages")
    if args.gradient_block_steps <= 0 or args.num_steps % args.gradient_block_steps:
        raise ValueError("gradient blocks must partition each environment's rollout")
    if args.trust_kl <= 0 or args.max_backtracks < 0 or args.control_learning_rate <= 0:
        raise ValueError("invalid trust region or control learning rate")
    if args.gradient_diagnostic_interval < 0:
        raise ValueError("invalid diagnostic interval")
    if args.gate_checkpoint and (min(args.gate_calibration_rollouts, args.gate_fit_rollouts) <= 0 or args.gate_holdout_rollouts < 16):
        raise ValueError("gate needs calibration, fit data, and at least sixteen held-out rollouts")
    if not args.cuda:
        raise ValueError("CUDA is required")
    args.batch_size = args.num_steps * args.num_envs
    args.minibatch_size = args.batch_size
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


class FrozenObsNorm:
    """Read-only checkpoint preprocessing for the gate, not a training normalizer.

    Shared VectorObsNorm remains the training implementation. Ping-pong outputs
    preserve the old observation until RolloutTransfer stages the same transition.
    """
    def __init__(self, state, num_envs, obs_shape):
        self.means = state["means"].cpu().numpy().copy()
        self.variances = state["variances"].cpu().numpy().copy()
        self.counts = state["counts"].cpu().numpy().copy()
        self.epsilon, self.clip = state["epsilon"], state["clip"]
        if self.means.shape != (num_envs, int(np.prod(obs_shape))):
            raise ValueError("gate must use the checkpoint's environment count and observation shape")
        self.inverse_std = 1.0 / np.sqrt(self.variances + self.epsilon)
        self._buffers = [np.empty((num_envs,) + tuple(obs_shape), dtype=np.float32) for _ in range(2)]
        self._transition = np.empty_like(self._buffers[0])
        self._work = np.empty_like(self.means, dtype=np.float64)
        self._cursor = 0

    def normalize(self, obs, rows=None, out_dtype=np.float32):
        if rows is not None:
            return np.clip((obs - self.means[rows]) * self.inverse_std[rows], -self.clip, self.clip).astype(out_dtype)
        output = self._buffers[self._cursor]
        self._cursor ^= 1
        np.subtract(np.asarray(obs).reshape(self.means.shape), self.means, out=self._work)
        np.multiply(self._work, self.inverse_std, out=self._work)
        np.clip(self._work, -self.clip, self.clip, out=self._work)
        output[...] = self._work.reshape(output.shape)
        return output

    def normalize_step(self, raw_next_obs, terminations, truncations, infos):
        normalized = self.normalize(raw_next_obs)
        self._transition[...] = normalized
        for index in np.flatnonzero(truncations):
            finals = infos.get("final_observation")
            if finals is None or finals[index] is None:
                raise ValueError("truncation requires its final observation, never the autoreset observation")
            final = np.asarray(finals[index]).reshape(-1)
            self._transition[index] = np.clip(
                (final - self.means[index]) * self.inverse_std[index], -self.clip, self.clip
            ).reshape(self._transition[index].shape)
        return normalized, self._transition


class FrozenRewardNorm:
    """Freeze a calibrated shared VectorRewardNorm's scaling for fit and holdout."""
    def __init__(self, normalizer):
        self.means, self.variances, self.counts = (getattr(normalizer, name).copy() for name in ("means", "variances", "counts"))
        self.epsilon, self.clip = normalizer.epsilon, normalizer.clip
        self.inverse_std = 1.0 / np.sqrt(self.variances + self.epsilon)
        self._work = np.empty_like(self.variances)
        self._output = np.empty_like(self.variances, dtype=np.float32)

    def normalize(self, rewards, terminations, out_dtype=np.float32):
        np.multiply(rewards, self.inverse_std, out=self._work)
        np.clip(self._work, -self.clip, self.clip, out=self._work)
        self._output[...] = self._work
        return self._output


def summarize_gate(rows, seed=1):
    """Paired whole-rollout bootstrap; not a confidence interval across RL seeds."""
    if len(rows) < 16:
        raise ValueError("incomplete held-out gate")
    ordinary = np.asarray([row["gradient/ordinary_block_variance"] for row in rows])
    corrected = np.asarray([row["gradient/corrected_block_variance"] for row in rows])
    point_ordinary = np.asarray([row["gradient/ordinary_variance_trace"] for row in rows])
    point_corrected = np.asarray([row["gradient/corrected_variance_trace"] for row in rows])
    if not (np.isfinite(ordinary).all() and np.isfinite(corrected).all()) or ordinary.sum() <= 0:
        return {"passed": False, "reason": "nonfinite or zero baseline block variance", "holdout": rows}
    indices = np.random.default_rng(seed).integers(0, len(rows), size=(10000, len(rows)))
    boot_ratios = corrected[indices].sum(1) / ordinary[indices].sum(1)
    upper = float(np.quantile(boot_ratios, 0.95))
    wins = int((corrected < ordinary).sum())
    return {
        "passed": bool(upper < 0.95 and wins >= np.ceil(0.75 * len(rows))),
        "criterion": "one-sided 95% paired rollout-bootstrap upper ratio <0.95 and >=75% rollout wins",
        "block_variance_ratio": float(corrected.sum() / ordinary.sum()),
        "block_ratio_upper95": upper,
        "rollout_wins": wins,
        "holdout_rollouts": len(rows),
        "transition_variance_ratio": float(point_corrected.sum() / point_ordinary.sum()),
        "holdout": rows,
    }


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
    if args.gate_checkpoint:
        args.num_iterations = args.gate_calibration_rollouts + args.gate_fit_rollouts + args.gate_holdout_rollouts
        args.total_timesteps = horizon * args.num_envs + args.num_iterations * args.batch_size
    else:
        args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", f"base Beta actor/value; independent gradient-control critic; {args.actor_update}; matched KL={args.trust_kl}")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        obs_shape = envs.single_observation_space.shape
        agent = Agent(envs, placement=args.placement, norm_kind=args.norm_kind, activation=args.activation).to(device)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma) if args.reward_norm else None
        if args.gate_checkpoint:
            checkpoint = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=False)
            if checkpoint["args"]["env_id"] != args.env_id:
                raise ValueError("gate checkpoint/environment mismatch")
            agent.actor.load_state_dict({key[len("actor."):]: value for key, value in checkpoint["model"].items() if key.startswith("actor.")})
            agent.critic.load_state_dict({key[len("critic."):]: value for key, value in checkpoint["model"].items()
                                          if key.startswith("critic.trunk.") or key.startswith("critic.value_head.")})
            obs_norm = FrozenObsNorm(checkpoint["obs_norm"], args.num_envs, obs_shape)
            del checkpoint
        actor_parameters = tuple(agent.actor.parameters())
        critic_parameters = tuple(agent.critic.parameters())
        control_parameters = tuple(agent.control.parameters())
        actor_start = tuple(parameter.detach().clone() for parameter in actor_parameters) if args.gate_checkpoint else ()
        value_start, control_start = (), ()
        optimizer = optim.Adam(actor_parameters + critic_parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        control_optimizer = optim.Adam(control_parameters, lr=args.control_learning_rate, eps=1e-5, fused=True)
        trust_region = ActorTrustRegion(agent.actor, optimizer, budget=args.trust_kl, max_backtracks=args.max_backtracks)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            alpha, beta, value = agent.get_policy_and_value(observations)
            inputs = control_inputs(observations, alpha, beta)
            return value.flatten(), agent.action_logprob(alpha, beta, native), agent.control(inputs), inputs, alpha, beta

        def loss_model(observations, native, old_logprobs, advantages, targets, old_values, features, reference, coefficients):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, targets, old_values, features, reference, coefficients, args)

        def control_model(inputs, advantages, features, scores, derivatives, normalizer):
            return control_loss(agent.control, inputs, advantages, features, scores, derivatives, normalizer,
                                args.num_envs, args.gradient_block_steps)

        def value_fit_model(observations, targets, old_values):
            return value_loss(agent, observations, targets, old_values, args)

        def exact_policy_kl(observations, reference):
            alpha, beta = (F.softplus(agent.actor(observations)) + 1).chunk(2, -1)
            return beta_kl(alpha, beta, reference).sum(-1).mean()

        project = make_projection_function(agent.actor)
        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            control_model = torch.compile(control_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            value_fit_model = torch.compile(value_fit_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            exact_policy_kl = torch.compile(exact_policy_kl, mode=args.compile_mode, fullgraph=True, dynamic=False)
            project = torch.compile(project, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)
        sampler_rng = np.random.default_rng(args.seed)

        def act(observations):
            native, action = sampler(host_actor(observations), sampler_rng)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        fit_generator = torch.Generator(device=device).manual_seed(args.seed + 10001)
        eval_generator = torch.Generator(device=device).manual_seed(args.seed + 20001)
        update_metrics = torch.empty((args.update_epochs, 8), device=device)
        grad_norms = torch.empty((args.update_epochs, 3), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)
        gate_rows = []
        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm, act_fn=lambda obs: act(obs)[1],
                                    horizon=horizon, phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            stage = "train"
            if args.gate_checkpoint:
                if iteration <= args.gate_calibration_rollouts:
                    stage = "calibrate"
                elif iteration <= args.gate_calibration_rollouts + args.gate_fit_rollouts:
                    stage = "fit"
                else:
                    stage = "holdout"
                if iteration == args.gate_calibration_rollouts + 1:
                    if rew_norm is not None:
                        rew_norm = FrozenRewardNorm(rew_norm)
                    value_start = tuple(parameter.detach().clone() for parameter in critic_parameters)
                if iteration == args.gate_calibration_rollouts + args.gate_fit_rollouts + 1:
                    control_start = tuple(parameter.detach().clone() for parameter in control_parameters)
            train_actor = stage == "train"
            train_value = stage in {"train", "calibrate"}
            train_control = stage in {"train", "fit"}
            if args.anneal_lr and train_actor:
                optimizer.param_groups[0]["lr"] = (1 - (iteration - 1) / args.num_iterations) * args.learning_rate
            bootstraps.reset()
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
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                batch = transfer.upload()
                b_obs, b_native = batch.fields["observations"].flatten(0, 1), batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs, b_coefficients, b_inputs, b_alpha, b_beta = (
                    tensor.clone() for tensor in rollout_statistics(b_obs, b_native)
                )
                reference = beta_score_reference(b_alpha, b_beta)
                b_features = beta_score_features(b_native, reference)
                kl_reference = beta_kl_reference(b_alpha, b_beta)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(batch.rewards, b_values.view(args.num_steps, args.num_envs),
                                             batch.terminations, batch.truncations, truncation_values, tail_value,
                                             args.gamma, args.gae_lambda)
                b_advantages, b_returns = advantages.flatten().clone(), returns.flatten().clone()
            diagnostics = {"losses/explained_variance": explained_variance(b_values, b_returns)}
            if train_control:
                with timer.span("control_geometry"), torch.no_grad():
                    directions = sample_parameter_directions(agent.actor, args.train_projections, fit_generator)
                    scores, derivatives = (tensor.clone() for tensor in project(b_obs, b_native, reference, directions))
                    ordinary_blocks = block_gradients(scores * b_advantages[:, None], args.num_envs, args.gradient_block_steps)
                    second_moment = ordinary_blocks.square().mean()
                    normalizer = torch.where(second_moment > 0, second_moment, torch.ones_like(second_moment))
                    del directions
            measure = stage == "holdout" or (stage != "calibrate" and args.gradient_diagnostic_interval and (iteration - 1) % args.gradient_diagnostic_interval == 0)
            if measure:
                with timer.span("gradient_diagnostics"), torch.no_grad():
                    directions = sample_parameter_directions(agent.actor, args.eval_projections, eval_generator)
                    eval_scores, eval_derivatives = (tensor.clone() for tensor in project(b_obs, b_native, reference, directions))
                    diagnostics.update(projection_diagnostics(b_advantages, b_features, b_coefficients, eval_scores, eval_derivatives,
                                                              args.num_envs, args.gradient_block_steps))
                    diagnostics["control/prediction_rms"] = (b_features * b_coefficients).sum(-1).square().mean().sqrt()
                    diagnostics["control/state_baseline_rms"] = b_coefficients[:, 0].square().mean().sqrt()
                    del directions, eval_scores, eval_derivatives
            updates, accepted_updates, backtrack_count, fraction_sum = 0, 0, 0, 0.0
            if stage != "holdout":
                with timer.span("update"):
                    for epoch in range(args.update_epochs):
                        for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                            if args.compile:
                                torch.compiler.cudagraph_mark_step_begin()
                            update_metrics[updates].zero_()
                            grad_norms[updates].zero_()
                            if train_actor:
                                loss, metrics = loss_model(b_obs[indices], b_native[indices], b_logprobs[indices], b_advantages[indices],
                                                           b_returns[indices], b_values[indices], b_features[indices],
                                                           tuple(t[indices] for t in reference), b_coefficients[indices])
                                optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                update_metrics[updates, :7].copy_(metrics)
                                grad_norms[updates, 0].copy_(nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm, foreach=True))
                                grad_norms[updates, 1].copy_(nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True))
                                trust_region.snapshot()
                                optimizer.step()
                                _, fraction, backtracks = trust_region.accept(lambda: exact_policy_kl(b_obs, kl_reference))
                                del _, loss, metrics
                                accepted_updates += int(fraction > 0)
                                fraction_sum += fraction
                                backtrack_count += backtracks
                            elif train_value:
                                loss = value_fit_model(b_obs, b_returns, b_values)
                                optimizer.zero_grad(set_to_none=True)
                                (args.vf_coef * loss).backward()
                                update_metrics[updates, 1].copy_(loss.detach())
                                grad_norms[updates, 1].copy_(nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True))
                                optimizer.step()
                                del loss
                            if train_control:
                                # Keep trajectory order; actor minibatch permutation is irrelevant here.
                                loss = control_model(b_inputs, b_advantages, b_features, scores, derivatives, normalizer)
                                control_optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                update_metrics[updates, 7].copy_(loss.detach())
                                grad_norms[updates, 2].copy_(nn.utils.clip_grad_norm_(control_parameters, args.max_grad_norm, foreach=True))
                                control_optimizer.step()
                                del loss
                            updates += 1
                        if train_actor and args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                            break
            with torch.no_grad():
                diagnostics["losses/exact_kl"] = exact_policy_kl(b_obs, kl_reference).clone()
                if updates:
                    last = update_metrics[updates - 1]
                    diagnostics.update({"losses/policy_loss": last[0], "losses/value_loss": last[1], "losses/entropy": last[2],
                                        "losses/old_approx_kl": last[3], "losses/approx_kl": last[4],
                                        "losses/clipfrac": update_metrics[:updates, 5].mean(), "losses/analytic_correction": last[6],
                                        "losses/control_block_second_moment": last[7]})
                    for index, name in enumerate(("actor", "critic", "control")):
                        diagnostics[f"grad/{name}_preclip_norm"] = grad_norms[:updates, index].mean()
                        diagnostics[f"grad/{name}_clip_fraction"] = (grad_norms[:updates, index] > args.max_grad_norm).float().mean()
            logged = gather_metrics(diagnostics)
            if train_actor:
                logged.update({"trust/rejected_updates": updates - accepted_updates, "trust/backtracks": backtrack_count,
                               "trust/mean_accepted_fraction": fraction_sum / updates})
            if any(not np.isfinite(value) for name, value in logged.items() if not name.endswith("explained_variance")):
                raise FloatingPointError("nonfinite learner or gradient-control metrics")
            if stage == "holdout":
                gate_rows.append({"rollout": len(gate_rows) + 1, "step": global_step, **logged})
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            if measure:
                ratio = logged["gradient/corrected_block_variance"] / logged["gradient/ordinary_block_variance"]
                print(f"stage={stage} rollout={iteration} step={global_step} block_variance_ratio={ratio:.6f}")
            else:
                print(f"stage={stage} rollout={iteration} step={global_step} exact_kl={logged['losses/exact_kl']:.6f}")
            interval_start, interval_step = now, global_step
        if args.gate_checkpoint:
            if not all(torch.equal(p, q) for p, q in zip(actor_parameters, actor_start)):
                raise RuntimeError("gate actor was modified")
            if not all(torch.equal(p, q) for p, q in zip(critic_parameters, value_start)):
                raise RuntimeError("gate value changed after calibration")
            if not all(torch.equal(p, q) for p, q in zip(control_parameters, control_start)):
                raise RuntimeError("gate control changed during holdout")
            report = summarize_gate(gate_rows, args.seed)
            report.update({"checkpoint": args.gate_checkpoint, "actor_unchanged": True, "value_frozen_after_calibration": True,
                           "control_frozen_for_holdout": True, "seed": args.seed, "args": vars(args)})
            report_path = f"runs/{run_name}/gate.json"
            with open(report_path, "w") as output:
                json.dump(report, output, indent=2, allow_nan=False)
            writer.add_scalar("gate/passed", int(report["passed"]), global_step)
            print(f"GATE_RESULT={json.dumps({key: value for key, value in report.items() if key not in {'holdout', 'args'}})}")
            print(f"gate report saved to {report_path}")
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            reward_state = None if rew_norm is None else {
                name: torch.from_numpy(getattr(rew_norm, name).copy()) for name in ("means", "variances", "counts")
            }
            if reward_state is not None:
                reward_state.update({"epsilon": rew_norm.epsilon, "clip": rew_norm.clip, "gamma": args.gamma,
                                     "frozen": isinstance(rew_norm, FrozenRewardNorm),
                                     "returns": torch.from_numpy(rew_norm.returns.copy()) if hasattr(rew_norm, "returns") else None})
            torch.save({"model": agent.state_dict(), "args": vars(args), "reward_norm": reward_state,
                        "obs_norm": {"means": torch.from_numpy(obs_norm.means.copy()), "variances": torch.from_numpy(obs_norm.variances.copy()),
                                     "counts": torch.from_numpy(obs_norm.counts.copy()), "epsilon": obs_norm.epsilon, "clip": obs_norm.clip}}, model_path)
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

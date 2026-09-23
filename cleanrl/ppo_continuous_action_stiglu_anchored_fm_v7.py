"""Moment-anchored Flow-MPO: Gaussian projection plus genuine residual angular CFM.
Gaussian head fits the conditional location/scale of the improved candidate law.
An odd velocity field fits centered, variance-normalized candidate residuals;
it cannot move the conditional native mean. Physical initial noise stays1/sqrt1536.
This is a hybrid: weighted Gaussian NLL for moments, CFM for residual shape.
Finite-bank moments/projection/solver errors remain; no fitted-policy KL guarantee.
"""
import hashlib
import json
import math
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

from cleanrl.shared.host_actor import make_situ_sphere_trunk
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
INITIAL_STD = 0.02551551815399144  # 1/sqrt(1536), fixed requested native scale.
LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    policy: Literal["flow", "gaussian"] = "flow"
    initial_std: float = INITIAL_STD
    candidate_count: int = 16
    candidate_kl: float = 0.02
    candidate_improvement: bool = True
    """false is the uniform-weight projection null control, not a fallback"""
    frame_samples: int = 32
    """independent old-policy samples for coordinate moments, not E-step candidates"""
    critic_baseline_epochs: int = 5
    critic_action_epochs: int = 5
    critic_holdout_fraction: float = 0.25
    action_order: Literal["linear", "quadratic"] = "linear"
    flow_steps: int = 8
    shape_epochs: int = 10
    diagnostic_interval: int = 8
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    save_model: bool = True
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
    clip_coef: float = 0.2
    clip_vloss: bool = True
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


def flow_interpolant(data, noise, times):
    """Angular path: data at zero, independent Gaussian source at one.

    For data,noise iid N(0,I), Cov(z_t,v_t)=0 at every t, so E[v_t|z_t]=0.
    This is stationary in distribution, not a zero per-pair regression target.
    """
    data, noise, times = data.detach(), noise.detach(), times.detach()
    angle = times * (math.pi / 2.0)
    cosine, sine = angle.cos(), angle.sin()
    return cosine * data + sine * noise, (math.pi / 2.0) * (-sine * data + cosine * noise)


@torch.no_grad()
def candidate_weights(scores, kl_budget):
    """One scalar positive temperature; states retain equal total mass.

    FP64 centering and scale removal preserve per-state offsets/global positive
    scaling. A 64-step bisection retains the feasible upper endpoint. Exact
    maximizer ties bound achievable KL; inactive constraints approach eta=0+
    without pretending the constraint can be saturated. No weight clipping.
    """
    values = scores.detach().double()
    centered = values - values.amax(-1, keepdim=True)
    scale = centered.abs().amax()
    normalizer = torch.where(scale > 0, scale, torch.ones_like(scale))
    centered = centered / normalizer
    count = scores.shape[-1]
    ties = (centered == 0).sum(-1)
    achievable = (math.log(count) - ties.double().log()).mean()
    lower = torch.zeros((), device=scores.device, dtype=torch.float64)
    upper = torch.full_like(lower, max(1.0, 1.0 / math.sqrt(2.0 * kl_budget)))
    for _ in range(64):
        middle = (lower + upper) * 0.5
        log_weights = torch.log_softmax(centered / middle, dim=-1)
        kl = (log_weights.exp() * (log_weights + math.log(count))).sum(-1).mean()
        feasible = kl <= kl_budget
        upper = torch.where(feasible, middle, upper)
        lower = torch.where(feasible, lower, middle)
    temperature = torch.where(scale > 0, upper * normalizer, torch.ones_like(scale))
    weights = torch.softmax(centered / upper, dim=-1).to(scores.dtype)
    # Report the actual cached FP32 distribution, not only the FP64 solver.
    wd = weights.double()
    conditional_kl = (torch.special.xlogy(wd, wd * count)).sum(-1)
    return weights.detach(), {
        "estep/conditional_kl": conditional_kl.mean(),
        "estep/max_conditional_kl": conditional_kl.max(),
        "estep/temperature": temperature,
        "estep/achievable_kl": achievable,
        "estep/tied_max_fraction": (ties > 1).double().mean(),
        "estep/flat_state_fraction": (ties == count).double().mean(),
        "estep/ess": wd.square().sum(-1).reciprocal().mean(),
        "estep/max_weight": wd.max(),
        "estep/predicted_improvement": (wd * values).sum(-1).sub(values.mean(-1)).mean(),
        "estep/candidate_best_gap": values.max(-1).values.sub(values.mean(-1)).mean(),
    }


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Flow-MPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("finite ordered action bounds required")
        self.policy, self.initial_std = args.policy, args.initial_std
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        self.obs_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", self.action_low + self.action_scale)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have finite positive FP32 width")
        # Same SiTU value architecture and critic-first RNG order as v2/v3.
        self.critic = nn.Sequential(
            make_situ_sphere_trunk(self.obs_dim, 64, n_blocks=3),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            make_situ_sphere_trunk(self.obs_dim, 64, n_blocks=3),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.0),
        )
        self.shape_actor = None
        if self.policy == "flow":
            self.shape_actor = nn.Sequential(
                make_situ_sphere_trunk(self.obs_dim + self.action_dim + 1, 64, n_blocks=3),
                layer_init(nn.Linear(64, self.action_dim), std=0.0),
            )
            # The odd projection cancels this coefficient; host mirror still
            # requires its zero bias slot. Do not optimize an unidentifiable bias.
            self.shape_actor[-1].bias.requires_grad_(False)

    def get_value(self, observations):
        return self.critic(observations)

    def velocity(self, observations, states, times):
        positive = torch.cat((observations, states, times), dim=-1)
        negative = torch.cat((observations, -states, times), dim=-1)
        first, second = self.shape_actor(torch.cat((positive, negative), dim=0)).chunk(2, 0)
        return 0.5 * (first - second)

    def gaussian_parameters(self, observations):
        # Unit-native mean zero/std one initially; no artificial std bounds.
        return self.actor(observations).chunk(2, dim=-1)

    def decode(self, native):
        return self.action_bias + self.action_scale * torch.tanh(self.initial_std * native)


@torch.no_grad()
def sample_shape(agent, observations, noise, steps):
    """Odd Heun map preserves central symmetry of the Gaussian source."""
    observations, native = observations.detach(), noise.detach().clone()
    h = -1.0 / steps
    for step in range(steps):
        times = torch.full_like(native[:, :1], 1.0 - step / steps)
        first = agent.velocity(observations, native, times)
        predictor = native + h * first
        second = agent.velocity(observations, predictor, torch.full_like(times, 1.0 - (step + 1) / steps))
        native = native + (0.5 * h) * (first + second)
    return native.detach()


@torch.no_grad()
def sample_native(agent, observations, noise, steps):
    mean, log_std = agent.gaussian_parameters(observations.detach())
    residual = sample_shape(agent, observations, noise, steps) if agent.policy == "flow" else noise.detach()
    return (mean + log_std.exp() * residual).detach()


class HostSampler:
    """Affine Gaussian anchor plus paired-row odd residual Heun mirror."""
    def __init__(self, agent, num_rows, steps):
        self.policy, self.initial_std = agent.policy, np.float32(agent.initial_std)
        self.obs_dim, self.action_dim, self.steps, self.rows = agent.obs_dim, agent.action_dim, steps, num_rows
        self.scale = agent.action_scale.detach().cpu().numpy().copy()
        self.bias = agent.action_bias.detach().cpu().numpy().copy()
        self.actor = make_host_mirror(agent.actor, num_rows)
        self.shape_actor = make_host_mirror(agent.shape_actor, 2 * num_rows) if agent.policy == "flow" else None
        shape = (num_rows, agent.action_dim)
        self.native, self.action, self.noise = (np.empty(shape, dtype=np.float32) for _ in range(3))
        self.first_velocity, self.work, self.second_velocity = (np.empty(shape, dtype=np.float32) for _ in range(3))
        self.mean, self.std, self.residual = (np.empty(shape, dtype=np.float32) for _ in range(3))
        self.inputs = np.empty((2 * num_rows, agent.obs_dim + agent.action_dim + 1), dtype=np.float32)

    def refresh(self):
        self.actor.refresh()
        if self.shape_actor is not None:
            self.shape_actor.refresh()

    def velocity(self, observations, states, time, out):
        self.inputs[:self.rows, :self.obs_dim] = observations
        self.inputs[self.rows:, :self.obs_dim] = observations
        self.inputs[:self.rows, self.obs_dim:-1] = states
        np.negative(states, out=self.inputs[self.rows:, self.obs_dim:-1])
        self.inputs[:, -1] = np.float32(time)
        first, second = np.split(self.shape_actor(self.inputs), 2, axis=0)
        np.subtract(first, second, out=out)
        out *= np.float32(0.5)

    def __call__(self, observations, rng, *, noise=None):
        if noise is None:
            rng.standard_normal(self.noise.shape, dtype=np.float32, out=self.noise)
        else:
            np.copyto(self.noise, noise)
        observations = observations.reshape(self.rows, self.obs_dim)
        mean, log_std = np.split(self.actor(observations), 2, axis=-1)
        np.copyto(self.mean, mean)
        np.exp(log_std, out=self.std)
        np.copyto(self.residual, self.noise)
        if self.shape_actor is not None:
            h = np.float32(-1.0 / self.steps)
            for step in range(self.steps):
                self.velocity(observations, self.residual, 1.0 - step / self.steps, self.first_velocity)
                np.multiply(self.first_velocity, h, out=self.work)
                self.work += self.residual
                self.velocity(observations, self.work, 1.0 - (step + 1) / self.steps, self.second_velocity)
                np.add(self.first_velocity, self.second_velocity, out=self.work)
                self.work *= h * np.float32(0.5)
                self.residual += self.work
        np.multiply(self.residual, self.std, out=self.native)
        self.native += self.mean
        np.multiply(self.native, self.initial_std, out=self.action)
        np.tanh(self.action, out=self.action)
        self.action *= self.scale
        self.action += self.bias
        return self.native, self.action


def quadratic_features(standardized, rows, columns):
    products = standardized[..., rows] * standardized[..., columns]
    divisor = torch.where(rows == columns, math.sqrt(2.0), 1.0)
    return torch.cat((standardized, products / divisor), dim=-1)


class AdvantageRegressor(nn.Module):
    """Staged, rollout-fresh residual regression, never a differentiable Q actor.

    Baseline: SiTU sphere64 x3 -> scalar. Action coefficients: one affine head
    of [frozen baseline64 features, raw normalized observation]. Action basis:
    r_i, optionally r_i*r_j (diagonals /sqrt(2)), centered by reference moments.
    A separate reference bank supplies the frame for BOTH policy families;
    candidates and observed actions use the SAME captured frame independently
    of the reference bank. Better conditioning does not increase return SNR.
    Heldout MSE benefit, not fit loss, diagnoses action-critic learnability.
    """
    def __init__(self, observation_dim, action_dim, action_order="linear"):
        super().__init__()
        self.action_order = action_order
        rows, columns = torch.triu_indices(action_dim, action_dim)
        self.register_buffer("rows", rows)
        self.register_buffer("columns", columns)
        self.feature_dim = action_dim + (rows.numel() if action_order == "quadratic" else 0)
        self.baseline = nn.Sequential(
            make_situ_sphere_trunk(observation_dim, 64, n_blocks=3),
            layer_init(nn.Linear(64, 1), std=0.0),
        )
        self.coefficients = layer_init(nn.Linear(64 + observation_dim, self.feature_dim), std=0.0)

    def basis(self, standardized):
        if self.action_order == "linear":
            return standardized
        return quadratic_features(standardized, self.rows, self.columns)

    def features(self, native, mean, std, feature_mean):
        return (self.basis((native.detach() - mean.detach()) / std.detach())
                - feature_mean.detach()).detach()

    @torch.no_grad()
    def state_features(self, observations):
        return torch.cat((self.baseline[0](observations.detach()), observations.detach()), dim=-1).detach()

    def action_prediction(self, state_features, action_features):
        return (self.coefficients(state_features.detach()) * action_features.detach()).sum(-1)


@torch.no_grad()
def capture_candidates(agent, regressor, observations, args, candidate_generator, frame_generator, sampler):
    """Actor is pre-update throughout capture; never invoke per-state host loops."""
    batch, action_dim = observations.shape[0], agent.action_dim
    candidates = torch.empty((batch, args.candidate_count, action_dim), device=observations.device)
    means = torch.empty((batch, action_dim), device=observations.device)
    stds = torch.empty_like(means)
    feature_means = torch.empty((batch, regressor.feature_dim), device=observations.device)
    for start in range(0, batch, args.minibatch_size):
        stop = min(start + args.minibatch_size, batch)
        obs = observations[start:stop]
        expanded = obs.repeat_interleave(args.candidate_count, dim=0)
        noise = torch.randn((expanded.shape[0], action_dim), device=obs.device, generator=candidate_generator)
        candidates[start:stop].copy_(sampler(expanded, noise).reshape(-1, args.candidate_count, action_dim))
        expanded = obs.repeat_interleave(args.frame_samples, dim=0)
        noise = torch.randn((expanded.shape[0], action_dim), device=obs.device, generator=frame_generator)
        reference = sampler(expanded, noise).reshape(-1, args.frame_samples, action_dim)
        mean = reference.mean(1)
        std = reference.std(1, unbiased=False)
        means[start:stop].copy_(mean)
        stds[start:stop].copy_(std)
        reference_features = regressor.basis((reference - mean[:, None]) / std[:, None])
        feature_means[start:stop].copy_(reference_features.mean(1))
    if not bool(torch.isfinite(candidates).all() & torch.isfinite(stds).all() & (stds > 0).all()
                & torch.isfinite(feature_means).all()):
        raise FloatingPointError("nonfinite/collapsed behavior frame or candidate bank")
    return candidates, means, stds, feature_means


@torch.no_grad()
def shuffled_action_residuals(residuals, num_steps, num_envs, generator):
    """Negative control: permute nearby standardized actions within each env.

    Thirty-two-step windows reduce state/law mismatch versus global shuffling;
    this is a diagnostic, not an exact conditional randomization test.
    """
    shuffled = torch.empty_like(residuals)
    environments = torch.arange(num_envs, device=residuals.device)[None]
    for start in range(0, num_steps, 32):
        length = min(32, num_steps - start)
        order = torch.rand((length, num_envs), device=residuals.device, generator=generator).argsort(0)
        indices = (start + order) * num_envs + environments
        shuffled[start * num_envs:(start + length) * num_envs].copy_(residuals[indices.flatten()])
    return shuffled


@torch.no_grad()
def action_calibration(prediction, residual):
    centered_prediction = prediction - prediction.mean()
    centered_residual = residual - residual.mean()
    variance = centered_prediction.square().mean()
    covariance = (centered_prediction * centered_residual).mean()
    correlation_denominator = (variance * centered_residual.square().mean()).sqrt()
    # Zero-predictor calibration is undefined: expose validity, not fake signal.
    valid = correlation_denominator > 0
    return {
        "critic/heldout_calibration_valid": valid.float(),
        "critic/heldout_action_correlation": covariance / torch.where(valid, correlation_denominator, torch.ones_like(variance)),
        "critic/heldout_action_calibration_slope": covariance / torch.where(variance > 0, variance, torch.ones_like(variance)),
        "critic/heldout_action_calibration_intercept": residual.mean()
        - prediction.mean() * covariance / torch.where(variance > 0, variance, torch.ones_like(variance)),
    }


@torch.no_grad()
def select_targets(candidates, weights, generator):
    indices = torch.multinomial(weights.detach(), 1, generator=generator)
    return candidates.detach().gather(1, indices[..., None].expand(-1, 1, candidates.shape[-1])).squeeze(1)


@torch.no_grad()
def candidate_moments(candidates, weights):
    mean = (weights[..., None] * candidates).sum(1)
    variance = (weights[..., None] * (candidates - mean[:, None]).square()).sum(1)
    return mean.detach(), variance.sqrt().detach()


def projection_loss(agent, observations, native, noise, times):
    """Location/scale objective is identical for Gaussian and flow modes."""
    native, observations = native.detach(), observations.detach()
    mean, log_std = agent.gaussian_parameters(observations)
    return (0.5 * ((native - mean) * (-log_std).exp()).square() + log_std + 0.5 * LOG_TWO_PI).sum(-1).mean()


def shape_loss(agent, observations, native, mean, std, noise, times):
    # Bank moments are frozen projection targets, not gradients into the base
    # actor. Odd velocity fits the centrally symmetrized standardized residual.
    residual = (native.detach() - mean.detach()) / std.detach()
    states, target = flow_interpolant(residual, noise, times)
    return (agent.velocity(observations.detach(), states, times.detach()) - target).square().mean()


def value_loss(agent, observations, returns, old_values, args):
    values = agent.get_value(observations.detach()).flatten()
    errors = (values - returns.detach()).square()
    if args.clip_vloss:
        clipped = old_values.detach() + (values - old_values.detach()).clamp(-args.clip_coef, args.clip_coef)
        errors = torch.maximum(errors, (clipped - returns.detach()).square())
    return 0.5 * errors.mean()


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs,
           args.flow_steps, args.shape_epochs, args.diagnostic_interval, args.critic_baseline_epochs,
           args.critic_action_epochs) <= 0:
        raise ValueError("rollout, optimizer and solver counts must be positive")
    if args.candidate_count < 2 or args.frame_samples < 4:
        raise ValueError("need >=2 candidates and >=4 independent frame samples")
    if not math.isfinite(args.candidate_kl) or not 0 < args.candidate_kl < math.log(args.candidate_count):
        raise ValueError("candidate KL must be in (0, log(K))")
    if not math.isfinite(args.initial_std) or args.initial_std != INITIAL_STD:
        raise ValueError(f"v7 hard initial native std must be {INITIAL_STD}")
    if not 0 < args.critic_holdout_fraction < 1 or args.num_envs < 2:
        raise ValueError("critic needs a nonempty heldout environment subset")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size <= 0 or args.batch_size % args.num_minibatches:
        raise ValueError("batch must divide evenly into nonempty minibatches")
    if not args.cuda:
        raise ValueError("CUDA is required")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(args.env_id, args.num_envs, backend=backend,
                                  num_threads=min(args.env_threads, args.num_envs),
                                  capture_video=args.capture_video, run_name=run_name)


def normalization_state(obs_norm, rew_norm):
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
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args)
        # Independent initialization stream prevents family-specific actor sizes
        # from changing the auxiliary critic initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 1237)
            regressor = AdvantageRegressor(agent.obs_dim, agent.action_dim, args.action_order)
        with open(__file__, "rb") as handle:
            source_bytes = handle.read()
        provenance = {
            "args": vars(args), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "actor_hash": state_hash(agent.actor), "critic_hash": state_hash(agent.critic),
            "advantage_regressor_hash": state_hash(regressor),
            "actor_parameters": sum(p.numel() for p in agent.actor.parameters()),
            "shape_parameters": sum(p.numel() for p in agent.shape_actor.parameters()) if agent.shape_actor is not None else 0,
            "shape_hash": state_hash(agent.shape_actor) if agent.shape_actor is not None else None,
            "critic_parameters": sum(p.numel() for p in agent.critic.parameters()),
            "advantage_parameters": sum(p.numel() for p in regressor.parameters()),
            "initial_physical_native_std": args.initial_std,
            "initial_unit_native_std": 1.0, "decoder": "bias+scale*tanh(initial_std*unit_native)",
            "path": "angularCFM of centered/variance-normalized weightedbank residuals; odd field; source1,data0; backwardHeun",
            "anchor": "weightedGaussianNLL fits base moments; odd residual flow cannot shift native mean, not a fixed-noise-magnitude bound",
            "estep": "mean-state finite-candidate KL only; no fitted-policy KL or return guarantee",
            "critic": "fresh baseline SiTU64x3 then affine coefficients on frozen64+obs; linear action basis by default, optional quadratic",
            "frame": "independent old-policy reference sample mean/std/feature moments, same estimator for both families",
            "holdout": "entire random env subset; labels excluded from both regressor stages; policy projects all states",
            "target": "fixed GAE standardized with train-only mean/std; residual subtract frozen state baseline",
            "optimizer": f"separate fused Adam lr={args.learning_rate} eps1e-5; regressor state+moments reset each rollout",
            "schedule": "persistent policy/value/shape anneal; rollout-fresh auxiliary regressor keeps fixed learning rate to preserve its per-fit optimization budget",
            "torch": torch.__version__, "numpy": np.__version__, "gymnasium": gym.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device),
            "papers": ["https://arxiv.org/abs/1806.06920", "https://arxiv.org/abs/2210.02747"],
        }
        for name, payload in (("config", vars(args)), ("provenance", provenance)):
            with open(f"runs/{run_name}/{name}.json", "w") as handle:
                json.dump(payload, handle, indent=2)
        with open(f"runs/{run_name}/source.py", "wb") as handle:
            handle.write(source_bytes)
        writer.add_text("policy", json.dumps(provenance, indent=2))
        agent, regressor = agent.to(device), regressor.to(device)
        initial_regressor = {name: value.detach().clone() for name, value in regressor.state_dict().items()}
        actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_optimizer = optim.Adam(agent.critic.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        shape_optimizer = (optim.Adam(agent.shape_actor.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
                           if agent.shape_actor is not None else None)
        baseline_optimizer = optim.Adam(regressor.baseline.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        action_optimizer = optim.Adam(regressor.coefficients.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)

        def actor_objective(obs, target, noise, times):
            return projection_loss(agent, obs, target, noise, times)

        def shape_objective(obs, target, mean, std, noise, times):
            return shape_loss(agent, obs, target, mean, std, noise, times)

        def value_objective(obs, returns, old):
            return value_loss(agent, obs, returns, old, args)

        def baseline_objective(obs, target):
            return 0.5 * (regressor.baseline(obs.detach()).flatten() - target.detach()).square().mean()

        def action_objective(features, actions, target):
            return 0.5 * (regressor.action_prediction(features, actions) - target.detach()).square().mean()

        def device_sampler(obs, noise):
            return sample_native(agent, obs, noise, args.flow_steps)

        value_model = agent.get_value
        baseline_model = regressor.baseline
        state_feature_model = regressor.state_features
        if args.compile:
            # No cudagraph output lifetime ambiguity across staged learner phases.
            def compile_dynamic(function):
                return torch.compile(function, fullgraph=True, dynamic=True,
                                     options={"triton.cudagraphs": False})
            value_model = compile_dynamic(value_model)
            baseline_model = compile_dynamic(baseline_model)
            state_feature_model = compile_dynamic(state_feature_model)
            device_sampler = torch.compile(device_sampler, fullgraph=True, dynamic=False,
                                           options={"triton.cudagraphs": False})
            actor_objective = torch.compile(actor_objective, fullgraph=True, dynamic=False, mode=args.compile_mode)
            if args.policy == "flow":
                shape_objective = torch.compile(shape_objective, fullgraph=True, dynamic=False, mode=args.compile_mode)
            value_objective = torch.compile(value_objective, fullgraph=True, dynamic=False, mode=args.compile_mode)
            baseline_objective = torch.compile(baseline_objective, fullgraph=True, dynamic=False, mode=args.compile_mode)
            action_objective = torch.compile(action_objective, fullgraph=True, dynamic=False, mode=args.compile_mode)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        generators = {name: torch.Generator(device=device).manual_seed(args.seed + offset)
                      for name, offset in (("shuffle", 101), ("candidates", 211), ("frame", 307),
                                           ("selection", 401), ("cfm", 503), ("holdout", 601),
                                           ("diagnostics", 709), ("negative_control", 907),
                                           ("shape_shuffle", 1001), ("shape_noise", 1109), ("shape_selection", 1213))}
        host_rng = np.random.default_rng(args.seed)
        probe_rng = np.random.default_rng(args.seed + 811)
        sample_actions = HostSampler(agent, args.num_envs, args.flow_steps)
        probe_states, probe_samples = 8, 32
        probe_rows = probe_states * probe_samples
        probe_sampler = HostSampler(agent, probe_rows, args.flow_steps)
        drift_sampler = (HostSampler(agent, probe_rows, 8)
                         if args.policy == "flow" and args.flow_steps != 8 else probe_sampler)
        fine_sampler = HostSampler(agent, probe_rows, 16) if args.policy == "flow" else None
        obs_shape = envs.single_observation_space.shape
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm, rew_norm = VectorObsNorm(args.num_envs, obs_shape), VectorRewardNorm(args.num_envs, args.gamma)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def act(observations):
            native, action = sample_actions(observations, host_rng)
            if not np.isfinite(native).all() or not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite samples")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

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
            learning_rate = args.learning_rate * (1.0 - (iteration - 1.0) / args.num_iterations) if args.anneal_lr else args.learning_rate
            # The auxiliary critic is reinitialized each rollout. Annealing its
            # learning rate would progressively stop fitting each fresh model.
            optimizers = [actor_optimizer, value_optimizer]
            if shape_optimizer is not None:
                optimizers.append(shape_optimizer)
            for optimizer in optimizers:
                optimizer.param_groups[0]["lr"] = learning_rate
            bootstraps.reset()
            sample_actions.refresh()
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
                b_obs = batch.fields["observations"].reshape(args.batch_size, agent.obs_dim)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values = value_model(b_obs).flatten().clone()
                tail_value = value_model(transfer.observation(next_obs_np).reshape(args.num_envs, agent.obs_dim)).flatten().clone()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(batch.rewards, b_values.view(args.num_steps, args.num_envs),
                    batch.terminations, batch.truncations, truncation_values, tail_value, args.gamma, args.gae_lambda)
                b_advantages, b_returns = advantages.flatten().clone(), returns.flatten().clone()
            # The actor remains frozen until all candidates, frames and E-step
            # weights are cached. No copy of an actor is needed for this ordering.
            with timer.span("candidate_sampling"):
                candidates, frame_mean, frame_std, feature_mean = capture_candidates(
                    agent, regressor, b_obs, args, generators["candidates"], generators["frame"], device_sampler)
            with timer.span("critic_prepare"), torch.no_grad():
                regressor.load_state_dict(initial_regressor)
                baseline_optimizer.state.clear()
                action_optimizer.state.clear()
                heldout_count = min(args.num_envs - 1, max(1, round(args.num_envs * args.critic_holdout_fraction)))
                env_order = torch.randperm(args.num_envs, device=device, generator=generators["holdout"])
                env_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                env_mask[env_order[:heldout_count]] = True
                heldout_mask = env_mask.repeat(args.num_steps)
                train_indices = (~heldout_mask).nonzero().flatten()
                heldout_indices = heldout_mask.nonzero().flatten()
                target_mean = b_advantages[train_indices].mean()
                target_std = b_advantages[train_indices].std(unbiased=False)
                target_std = torch.where(target_std > 0, target_std, torch.ones_like(target_std))
                standardized_advantage = ((b_advantages - target_mean) / target_std).detach()
                observed_features = regressor.features(b_native, frame_mean, frame_std, feature_mean)
            with timer.span("critic_baseline"):
                baseline_losses = []
                for _ in range(args.critic_baseline_epochs):
                    for local in device_minibatches(train_indices.numel(), args.minibatch_size, device, generators["shuffle"]):
                        indices = train_indices[local]
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss = baseline_objective(b_obs[indices], standardized_advantage[indices])
                        baseline_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(regressor.baseline.parameters(), args.max_grad_norm)
                        baseline_optimizer.step()
                        baseline_losses.append(loss.detach().clone())
            with timer.span("critic_features"), torch.no_grad():
                baseline_prediction = baseline_model(b_obs).flatten().clone()
                state_features = state_feature_model(b_obs).clone()
                residual_target = (standardized_advantage - baseline_prediction).detach()
            with timer.span("critic_action"):
                action_losses = []
                for _ in range(args.critic_action_epochs):
                    for local in device_minibatches(train_indices.numel(), args.minibatch_size, device, generators["shuffle"]):
                        indices = train_indices[local]
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss = action_objective(state_features[indices], observed_features[indices], residual_target[indices])
                        action_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(regressor.coefficients.parameters(), args.max_grad_norm)
                        action_optimizer.step()
                        action_losses.append(loss.detach().clone())
            with timer.span("estep"), torch.no_grad():
                scores = torch.empty((args.batch_size, args.candidate_count), device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    stop = start + args.minibatch_size
                    features = regressor.features(candidates[start:stop], frame_mean[start:stop, None],
                                                   frame_std[start:stop, None], feature_mean[start:stop, None])
                    coefficients = regressor.coefficients(state_features[start:stop])
                    # State baseline cancels exactly from conditional softmax.
                    scores[start:stop].copy_((features * coefficients[:, None]).sum(-1))
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("nonfinite action critic scores")
                estep_scores = scores if args.candidate_improvement else torch.zeros_like(scores)
                weights, estep_metrics = candidate_weights(estep_scores, args.candidate_kl)
                projection_mean, projection_std = candidate_moments(candidates, weights)
                if not bool(torch.isfinite(projection_std).all() & (projection_std > 0).all()):
                    raise FloatingPointError("collapsed/nonfinite candidate projection moments")
                prediction = regressor.action_prediction(state_features, observed_features)
                base_errors = residual_target.square()
                full_errors = (residual_target - prediction).square()
                shuffled_residuals = shuffled_action_residuals(
                    (b_native - frame_mean) / frame_std, args.num_steps, args.num_envs, generators["negative_control"])
                shuffled_features = regressor.basis(shuffled_residuals) - feature_mean
                shuffled_prediction = regressor.action_prediction(state_features, shuffled_features)
                shuffled_errors = (residual_target - shuffled_prediction).square()
                score_spread = scores.std(-1, unbiased=False)
                critic_metrics = {
                    **action_calibration(prediction[heldout_indices], residual_target[heldout_indices]),
                    "critic/heldout_shuffled_full_mse": shuffled_errors[heldout_indices].mean(),
                    "critic/heldout_shuffled_action_benefit": (base_errors - shuffled_errors)[heldout_indices].mean(),
                    "critic/heldout_unshuffled_minus_shuffled_benefit": (shuffled_errors - full_errors)[heldout_indices].mean(),
                    "estep/candidate_score_std_mean": score_spread.mean(),
                    "estep/candidate_score_std_min": score_spread.min(),
                    "estep/candidate_score_std_max": score_spread.max(),
                    "estep/raw_candidate_score_std_mean": score_spread.mean() * target_std,
                    "critic/baseline_fit_loss": torch.stack(baseline_losses).mean(),
                    "critic/action_fit_loss": torch.stack(action_losses).mean(),
                    "critic/heldout_baseline_mse": base_errors[heldout_indices].mean(),
                    "critic/heldout_full_mse": full_errors[heldout_indices].mean(),
                    "critic/heldout_action_benefit": (base_errors - full_errors)[heldout_indices].mean(),
                    "critic/train_action_benefit": (base_errors - full_errors)[train_indices].mean(),
                    "critic/heldout_raw_action_benefit": (base_errors - full_errors)[heldout_indices].mean() * target_std.square(),
                    "critic/heldout_action_prediction_std": prediction[heldout_indices].std(unbiased=False),
                    "critic/heldout_residual_std": residual_target[heldout_indices].std(unbiased=False),
                    "critic/heldout_action_feature_mean_rms": observed_features[heldout_indices].mean(0).square().mean().sqrt(),
                    "critic/heldout_action_feature_second_moment": observed_features[heldout_indices].square().mean(),
                    "critic/target_std": target_std,
                    "sampling/behavior_frame_physical_std": frame_std.mean() * args.initial_std,
                }
                diagnose = iteration == 1 or iteration % args.diagnostic_interval == 0 or iteration == args.num_iterations
                if diagnose:
                    # Fresh heldout pairs at auxiliary-critic-heldout states.
                    # Actor SGD visits these states with other pairs: this is
                    # pair error, not heldout-state projection generalization.
                    fit_indices = heldout_indices[torch.linspace(0, heldout_indices.numel() - 1, probe_rows, device=device).long()]
                    fit_target = select_targets(candidates[fit_indices], weights[fit_indices], generators["diagnostics"])
                    fit_noise = torch.randn(fit_target.shape, device=device, generator=generators["diagnostics"])
                    fit_times = torch.rand((probe_rows, 1), device=device, generator=generators["diagnostics"])
                    before_fit = projection_loss(agent, b_obs[fit_indices], fit_target, fit_noise, fit_times).clone()
                    if args.policy == "flow":
                        before_shape = shape_loss(agent, b_obs[fit_indices], fit_target,
                                                  projection_mean[fit_indices], projection_std[fit_indices], fit_noise, fit_times).clone()
            actor_losses, value_losses = [], []
            with timer.span("value_update"):
                for _ in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, generators["shuffle"]):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss = value_objective(b_obs[indices], b_returns[indices], b_values[indices])
                        value_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                        value_optimizer.step()
                        value_losses.append(loss.detach().clone())
            with timer.span("projection_update"):
                for _ in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, generators["shuffle"]):
                        target = select_targets(candidates[indices], weights[indices], generators["selection"])
                        noise = torch.randn(target.shape, device=device, generator=generators["cfm"])
                        times = torch.rand((indices.numel(), 1), device=device, generator=generators["cfm"])
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss = actor_objective(b_obs[indices], target, noise, times)
                        actor_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                        actor_optimizer.step()
                        actor_losses.append(loss.detach().clone())
            shape_losses = []
            if args.policy == "flow":
                with timer.span("shape_projection"):
                    for _ in range(args.shape_epochs):
                        for indices in device_minibatches(args.batch_size, args.minibatch_size, device, generators["shape_shuffle"]):
                            target = select_targets(candidates[indices], weights[indices], generators["shape_selection"])
                            noise = torch.randn(target.shape, device=device, generator=generators["shape_noise"])
                            times = torch.rand((indices.numel(), 1), device=device, generator=generators["shape_noise"])
                            if args.compile:
                                torch.compiler.cudagraph_mark_step_begin()
                            loss = shape_objective(b_obs[indices], target, projection_mean[indices], projection_std[indices], noise, times)
                            shape_optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(agent.shape_actor.parameters(), args.max_grad_norm)
                            shape_optimizer.step()
                            shape_losses.append(loss.detach().clone())
            with timer.span("diagnostics"), torch.no_grad():
                anchor_mean, anchor_log_std = agent.gaussian_parameters(b_obs)
                tensor_metrics = {**estep_metrics, **critic_metrics,
                    "anchor/mean_target_error_noise_units": ((anchor_mean - projection_mean) / projection_std).square().mean().sqrt(),
                    "anchor/log_std_target_error": (anchor_log_std - projection_std.log()).square().mean().sqrt(),
                    "losses/actor_loss": torch.stack(actor_losses).mean(),
                    "losses/value_loss": torch.stack(value_losses).mean(),
                    "losses/explained_variance": explained_variance(b_values, b_returns)}
                if diagnose:
                    after_fit = projection_loss(agent, b_obs[fit_indices], fit_target, fit_noise, fit_times)
                    prefix = "gaussian_projection"
                    tensor_metrics.update({f"{prefix}/heldout_pair_error_before": before_fit,
                                           f"{prefix}/heldout_pair_error_after": after_fit,
                                           f"{prefix}/heldout_pair_improvement": before_fit - after_fit})
                if args.policy == "flow":
                    tensor_metrics["losses/shape_loss"] = torch.stack(shape_losses).mean()
                    if diagnose:
                        after_shape = shape_loss(agent, b_obs[fit_indices], fit_target,
                                                 projection_mean[fit_indices], projection_std[fit_indices], fit_noise, fit_times)
                        tensor_metrics.update({"flow/heldout_shape_error_before": before_shape,
                                               "flow/heldout_shape_error_after": after_shape,
                                               "flow/heldout_shape_improvement": before_shape - after_shape})
                logged = gather_metrics(tensor_metrics)
            if diagnose:
                with timer.span("host_diagnostics", use_cuda=False):
                    state_indices = torch.linspace(0, args.batch_size - 1, probe_states, device=device).long()
                    probe_observations = np.ascontiguousarray(np.repeat(b_obs[state_indices].cpu().numpy(), probe_samples, axis=0))
                    half_noise = probe_rng.standard_normal((probe_states, probe_samples // 2, agent.action_dim), dtype=np.float32)
                    probe_noise = np.concatenate((half_noise, -half_noise), axis=1).reshape(probe_rows, agent.action_dim)
                    probe_sampler.refresh()
                    probe_native, probe_action = probe_sampler(probe_observations, None, noise=probe_noise)
                    conditional = ((probe_action - probe_sampler.bias) / probe_sampler.scale).reshape(probe_states, probe_samples, agent.action_dim)
                    logged["sampling/effective_conditional_action_std"] = float(conditional.std(1, ddof=1).mean())
                    logged["sampling/effective_conditional_physical_native_std"] = float(probe_native.reshape(probe_states, probe_samples, agent.action_dim).std(1, ddof=1).mean() * args.initial_std)
                    conditional_native = probe_native.reshape(probe_states, probe_samples, agent.action_dim) * args.initial_std
                    logged["sampling/conditional_native_mean_rms"] = float(np.sqrt(np.mean(conditional_native.mean(1) ** 2)))
                    logged["sampling/conditional_action_mean_rms"] = float(np.sqrt(np.mean(conditional.mean(1) ** 2)))
                    logged["sampling/saturation_fraction"] = float((np.abs(conditional) > 0.99).mean())
                    if fine_sampler is not None:
                        if drift_sampler is probe_sampler:
                            coarse_native, coarse_action = probe_native, probe_action
                        else:
                            drift_sampler.refresh()
                            coarse_native, coarse_action = drift_sampler(probe_observations, None, noise=probe_noise)
                        fine_sampler.refresh()
                        fine_native, fine_action = fine_sampler(probe_observations, None, noise=probe_noise)
                        logged["solver/8vs16_action_rms"] = float(np.sqrt(np.mean(((coarse_action - fine_action) / probe_sampler.scale) ** 2)))
                        logged["solver/8vs16_unit_native_rms"] = float(np.sqrt(np.mean((coarse_native - fine_native) ** 2)))
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite Flow-MPO metrics")
            now = time.perf_counter()
            logged.update({"charts/learning_rate": learning_rate,
                           "charts/SPS": global_step / (now - start_time),
                           "charts/interval_SPS": (global_step - interval_step) / (now - interval_start),
                           "optimizer/actor_steps": len(actor_losses), "optimizer/value_steps": len(value_losses),
                           "sampling/nfe": 1 + 4 * args.flow_steps if args.policy == "flow" else 1,
                           "optimizer/shape_steps": len(shape_losses),
                           "estep/kl_budget": args.candidate_kl})
            if episode_returns:
                logged["charts/rollout_mean_return"] = float(np.mean(episode_returns))
            for phase, timing in timer.summary().items():
                logged[f"timing/{phase}_s"] = timing["total_s"]
            timer.reset()
            metric_file.write(json.dumps({"step": global_step, **logged}) + "\n")
            metric_file.flush()
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            print(f"SPS: {int(logged['charts/SPS'])}, heldout_action_benefit={logged['critic/heldout_action_benefit']:.6g}, candidate_KL={logged['estep/conditional_kl']:.6g}")
            interval_start, interval_step = time.perf_counter(), global_step
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save({"state_dict": agent.state_dict(), "advantage_regressor": regressor.state_dict(),
                        "args": vars(args), "provenance": provenance, "global_step": global_step,
                        "normalization": normalization_state(obs_norm, rew_norm),
                        "optimizers": {"actor": actor_optimizer.state_dict(), "value": value_optimizer.state_dict(),
                                       "shape": shape_optimizer.state_dict() if shape_optimizer is not None else None},
                        "rng": {name: generator.get_state() for name, generator in generators.items()},
                        "host_rng": host_rng.bit_generator.state, "probe_rng": probe_rng.bit_generator.state}, model_path)
            print(f"model saved to {model_path}")


if __name__ == "__main__":
    main()

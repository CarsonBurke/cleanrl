# Credit metric v18: fit the joint future-return embedding for policy credit.
# A full-rank component/utility metric and frozen Fisher score weights emphasize
# errors in alpha/beta credit. Means use the observed TD segment's fixed scale;
# joint characteristic moments retain distributional representation pressure.
# Actor, target snapshots and rollout protocol are unchanged from v17.
# No GAE, replay, contrastive learning, decoding, or Bellman target sweeps.
import json
import math
import random
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter
import tyro

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import gather_metrics
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

NATIVE_TASKS = frozenset(('HalfCheetah-v4',))


@dataclass
class Args:
    exp_name: str = 'vector_credit_metric_v18'
    seed: int = 1
    env_id: str = 'HalfCheetah-v4'
    total_timesteps: int = 8000000
    num_envs: int = 16
    num_steps: int = 2048
    td_steps: int = 32
    critic_width: int = 192
    critic_epochs: int = 18
    critic_learning_rate: float = .0003
    credit_score_weighting: bool = True
    max_grad_norm: float = .5
    trust_kl: float = .03
    cg_iterations: int = 100
    action_samples: int = 8
    cg_damping: float = .01
    cg_relative_tolerance: float = 1e-6
    line_search_steps: int = 16
    env_backend: str = 'auto'
    env_threads: int = 2
    compile: bool = True
    compile_mode: str = 'reduce-overhead'
    capture_video: bool = False
    torch_deterministic: bool = True
    non_blocking_transfers: bool = False
    save_model: bool = True



class RecordingObsNorm:
    """Use shared calibration while retaining its last unnormalized observation."""
    def __init__(self, normalizer):
        self.base = normalizer
        self.last_raw = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def normalize(self, observations, *args, **kwargs):
        self.last_raw = np.asarray(observations).copy()
        return self.base.normalize(observations, *args, **kwargs)

    def normalize_step(self, observations, *args, **kwargs):
        self.last_raw = np.asarray(observations).copy()
        return self.base.normalize_step(observations, *args, **kwargs)


class FixedAffineObsNorm:
    """Pool warmup statistics once; use one smooth coordinate system for all envs.

    No clipping: a clipped observation would lose physical derivative information.
    Ping-pong buffers retain the previous observation until transition staging.
    """
    def __init__(self, normalizer, num_envs, obs_shape):
        counts = np.asarray(normalizer.counts, dtype=np.float64).reshape(num_envs, -1)
        if counts.shape[1] != 1:
            raise ValueError('expected one observation count per environment')
        weights = counts / counts.sum()
        means = np.asarray(normalizer.means, dtype=np.float64)
        variances = np.asarray(normalizer.variances, dtype=np.float64)
        self.mean = (means * weights).sum(0)
        variance = ((variances + (means - self.mean) ** 2) * weights).sum(0)
        self.inverse_std = 1 / np.sqrt(variance + normalizer.epsilon)
        self._buffers = [np.empty((num_envs,) + tuple(obs_shape), np.float32) for _ in range(2)]
        self._transition = np.empty_like(self._buffers[0])
        self._cursor = 0

    def normalize(self, observations):
        output = self._buffers[self._cursor]
        self._cursor ^= 1
        output[:] = (np.asarray(observations) - self.mean) * self.inverse_std
        return output

    def normalize_step(self, observations, terminations, truncations, infos):
        normalized = self.normalize(observations)
        self._transition[:] = normalized
        for index in np.flatnonzero(terminations | truncations):
            final = infos.get('final_observation')
            if final is None or final[index] is None:
                raise ValueError('real terminal observation required for transition learning')
            self._transition[index] = (np.asarray(final[index]) - self.mean) * self.inverse_std
        return normalized, self._transition


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError('Beta actor requires a Box action space')
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError('finite ordered action bounds required')
        self.register_buffer('action_low', torch.as_tensor(low.reshape(-1).copy()))
        self.register_buffer('action_high', torch.as_tensor(high.reshape(-1).copy()))
        self.actor = nn.Sequential(
            make_norm_residual_trunk(int(np.prod(envs.single_observation_space.shape)), 64,
                                    placement='pre', norm_kind='rms', activation='stiglu'),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=.01))


def beta_geometry(logits, actions):
    """F=L L^T, whitened scores z=L^-1(T-E[T]), and policy parameters.

    Layout is [batch, action, alpha/beta]. Work in FP64 for trigamma subtraction,
    then return the input dtype. No Fisher damping changes the target geometry.
    """
    alpha, beta = (F.softplus(logits) + 1).double().chunk(2, -1)
    total = alpha + beta
    off = -torch.polygamma(1, total)
    l11 = (torch.polygamma(1, alpha) + off).sqrt()
    l21 = off / l11
    l22 = (torch.polygamma(1, beta) + off - l21.square()).sqrt()
    score_a = actions.double().log() - alpha.digamma() + total.digamma()
    score_b = torch.log1p(-actions.double()) - beta.digamma() + total.digamma()
    z1 = score_a / l11
    z2 = (score_b - l21 * z1) / l22
    latent = torch.stack((z1, z2), -1).to(logits.dtype)
    factor = torch.stack((l11, l21, l22), -1).to(logits.dtype)
    parameters = torch.stack((alpha, beta), -1).to(logits.dtype)
    return latent, factor, parameters


def beta_kl_reference(parameters):
    alpha, beta = parameters.double().unbind(-1)
    total = alpha + beta
    return torch.stack((alpha, beta, alpha.lgamma() + beta.lgamma() - total.lgamma(),
                        alpha.digamma() - total.digamma(), beta.digamma() - total.digamma()), -1)


def conjugate_gradient(operator, rhs, iterations, relative_tolerance=1e-6, diagonal=None):
    """GPU preconditioned CG; FP64 reductions and tensor-only stopping."""
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    diagonal = torch.ones_like(rhs) if diagonal is None else diagonal
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    dot = lambda x, y: (x.double() * y.double()).sum()
    squared = dot(residual, residual)
    threshold = squared * relative_tolerance ** 2
    product_rz = dot(residual, preconditioned)
    tiny = torch.finfo(torch.float64).tiny
    for _ in range(iterations):
        product = operator(direction)
        curvature = dot(direction, product)
        active = (squared > threshold) & (curvature > tiny) & torch.isfinite(curvature)
        step = torch.where(active, product_rz / curvature.clamp_min(tiny), 0.).to(rhs.dtype)
        solution = solution + step * direction
        residual = residual - step * torch.where(active, product, 0.)
        preconditioned = residual / diagonal
        next_rz = dot(residual, preconditioned)
        ratio = torch.where(active, next_rz / product_rz.clamp_min(tiny), 0.).to(rhs.dtype)
        direction = preconditioned + ratio * direction
        product_rz = next_rz
        squared = dot(residual, residual)
    return solution, squared.sqrt() / dot(rhs, rhs).sqrt().clamp_min(tiny)


class ActorNaturalGradient:
    """One full-batch, representable-actor natural step shared by both credits.

    The Fisher includes the complete alpha/beta block, softplus derivatives,
    and the actor Jacobian. Damping only regularizes CG; KL scaling uses the
    undamped Fisher. No actor optimizer state exists to bias either condition.
    """
    def __init__(self, actor, budget=.01, damping=.01, cg_iterations=10,
                 relative_tolerance=1e-6, line_search_steps=12, compile=False,
                 compile_mode='reduce-overhead'):
        self.actor = actor
        self.named = tuple(actor.named_parameters())
        self.budget, self.damping = budget, damping
        self.cg_iterations, self.relative_tolerance = cg_iterations, relative_tolerance
        self.line_search_steps = line_search_steps

        def forward(parameters, observation):
            return torch.func.functional_call(actor, parameters, (observation,))

        def gradient(parameters, observation, old_logits, credit, weights):
            _, backward = torch.func.vjp(lambda p: forward(p, observation), parameters)
            logit_credit = torch.cat((credit[..., 0], credit[..., 1]), -1) * old_logits.sigmoid()
            result, = backward(logit_credit * (weights / weights.sum())[:, None])
            return self.flatten(result)

        def fisher(parameters, observation, old_logits, factor, weights, vector):
            tangent = self.unflatten(vector)
            _, pushed = torch.func.jvp(lambda p: forward(p, observation), (parameters,), (tangent,))
            sigmoid = old_logits.sigmoid()
            first, second = (pushed * sigmoid).chunk(2, -1)
            l11, l21, l22 = factor.unbind(-1)
            # Apply F=L L^T in factored form: expanding nearly cancelling
            # trigamma terms loses the small concentration-direction curvature.
            first_transpose = l11 * first + l21 * second
            second_transpose = l22 * second
            alpha = l11 * first_transpose
            beta = l21 * first_transpose + l22 * second_transpose
            cotangent = torch.cat((alpha, beta), -1) * sigmoid * (weights / weights.sum())[:, None]
            _, backward = torch.func.vjp(lambda p: forward(p, observation), parameters)
            result, = backward(cotangent)
            return self.flatten(result)

        def diagonal_estimate(parameters, observation, old_logits, factor, weights, probes):
            _, backward = torch.func.vjp(lambda p: forward(p, observation), parameters)
            l11, l21, l22 = factor.unbind(-1)
            scale = (weights / weights.sum()).sqrt()[:, None]
            def pull(probe):
                first, second = probe.unbind(-1)
                cotangent = torch.cat((l11 * first, l21 * first + l22 * second), -1)
                result, = backward(cotangent * old_logits.sigmoid() * scale)
                return self.flatten(result)
            return torch.vmap(pull)(probes).square().mean(0)

        self.diagonal_estimate = diagonal_estimate
        self.gradient = gradient
        self.fisher = fisher
        if compile:
            self.diagonal_estimate = torch.compile(diagonal_estimate, fullgraph=True, mode=compile_mode)
            self.gradient = torch.compile(gradient, fullgraph=True, mode=compile_mode)
            self.fisher = torch.compile(fisher, fullgraph=True, mode=compile_mode)

    def flatten(self, parameters):
        return torch.cat(tuple(parameters[name].reshape(-1) for name, _ in self.named))

    def unflatten(self, vector):
        result = {}
        offset = 0
        for name, parameter in self.named:
            result[name] = vector[offset:offset + parameter.numel()].view_as(parameter)
            offset += parameter.numel()
        return result

    @torch.no_grad()
    def search(self, before, direction, initial_scale, measure):
        """Search the whole bounded ray, including interior reward maxima.

        Golden-section refinement is only a one-dimensional approximation. Keep
        the best actually measured positive-gain, exact-KL-feasible proposal.
        """
        delta = self.unflatten(direction)
        initial = float(initial_scale)
        if not math.isfinite(initial) or initial <= 0:
            gain, kl = measure()
            return gain, kl, 0., 0, 0
        best_scale, best_gain, evaluations = 0., 0., 0

        def evaluate(scale):
            nonlocal best_scale, best_gain, evaluations
            for name, parameter in self.named:
                parameter.copy_(before[name]).add_(delta[name], alpha=scale)
            gain, kl = measure()
            evaluations += 1
            feasible = bool(torch.isfinite(gain) & torch.isfinite(kl) &
                            (kl >= -1e-8) & (kl <= self.budget))
            value = float(gain) if feasible else -math.inf
            if value > best_gain:
                best_gain, best_scale = value, scale
            return value, feasible

        upper = initial
        bracketed = False
        for _ in range(min(3, self.line_search_steps - 2)):
            _, feasible = evaluate(upper)
            if not feasible:
                bracketed = True
                break
            upper *= 2
        lower = 0.
        ratio = (math.sqrt(5) - 1) / 2
        left = upper - ratio * (upper - lower)
        right = lower + ratio * (upper - lower)
        left_value, _ = evaluate(left)
        right_value, _ = evaluate(right)
        while evaluations < self.line_search_steps:
            if left_value >= right_value:
                upper, right, right_value = right, left, left_value
                left = upper - ratio * (upper - lower)
                left_value, _ = evaluate(left)
            else:
                lower, left, left_value = left, right, right_value
                right = lower + ratio * (upper - lower)
                right_value, _ = evaluate(right)
        for name, parameter in self.named:
            parameter.copy_(before[name])
            if best_scale > 0:
                parameter.add_(delta[name], alpha=best_scale)
        gain, kl = measure()
        return gain, kl, best_scale, evaluations, int(not bracketed)

    @torch.no_grad()
    def step(self, observations, logits, factor, credit, weights, measure):
        before = {name: p.detach().clone() for name, p in self.named}
        gradient = self.gradient(before, observations, logits, credit, weights).clone()
        # Normalize only the linear solve RHS: trust scaling cancels this factor.
        # This also keeps CG stopping behavior invariant to global reward units.
        rhs = gradient / gradient.norm().clamp_min(torch.finfo(gradient.dtype).tiny)

        def undamped(vector):
            return self.fisher(before, observations, logits, factor, weights, vector).clone()

        # Independent signs across probes, states, actuators and alpha/beta.
        probes = torch.empty((8, *credit.shape), device=credit.device, dtype=credit.dtype)
        probes.bernoulli_(.5).mul_(2).sub_(1)
        diagonal = self.diagonal_estimate(before, observations, logits, factor, weights, probes).clone()
        diagonal = diagonal + self.damping
        operator = lambda v: undamped(v) + self.damping * v
        direction, recursive_residual = conjugate_gradient(
            operator, rhs, self.cg_iterations, self.relative_tolerance, diagonal)
        residual = (rhs - operator(direction)).double().norm() / rhs.double().norm().clamp_min(1e-30)
        curvature = (direction.double() * undamped(direction).double()).sum()
        gain_direction = (gradient.double() * direction.double()).sum()
        safe = (curvature > 0) & (gain_direction > 0) & torch.isfinite(curvature) & torch.isfinite(gain_direction)
        scale = torch.where(safe, (2 * self.budget / curvature.clamp_min(torch.finfo(curvature.dtype).tiny)).sqrt(), 0.)
        gain, kl, accepted, evaluations, cap_hits = self.search(before, direction, scale, measure)
        return {'policy/accepted_gain': gain, 'policy/exact_joint_kl': kl,
                'policy/accepted_scale': accepted, 'policy/line_search_evaluations': evaluations,
                'policy/line_search_cap_hits': cap_hits, 'natural/cg_residual_ratio': residual,
                'natural/undamped_curvature': curvature, 'natural/g_dot_direction': gain_direction,
                'natural/gradient_norm': gradient.norm(),
                'natural/cg_recursive_residual_ratio': recursive_residual,
                'natural/preconditioner_min': diagonal.min(), 'natural/preconditioner_max': diagonal.max()}



def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def resolved_ages(ages, ends, horizon=1000):
    """Infer unknown warmup ages only from actual fixed-horizon episode ends."""
    steps, environments = ages.shape
    positions = torch.arange(steps, device=ages.device)[:, None].expand(-1, environments)
    ending = torch.where(ends, positions, steps).T.contiguous().flip(-1).cummin(-1).values.flip(-1).T
    resolved = torch.where((ages < 0) & (ending < steps), horizon - (ending - positions + 1), ages)
    return resolved.flatten()


def reward_components(rewards, physical_actions, control_weight):
    costs = control_weight * physical_actions.square()
    return torch.cat(((rewards + costs.sum(-1))[..., None], -costs), -1)


def td_segments(components, ends, td_steps=32):
    """Real component sums and endpoints for the unified Bellman target."""
    steps, environments, channels = components.shape
    positions = torch.arange(steps, device=components.device)[:, None].expand(-1, environments)
    ending = torch.where(ends, positions, steps).T.contiguous().flip(-1).cummin(-1).values.flip(-1).T
    consumed = torch.minimum(torch.full_like(positions, td_steps), steps - positions)
    consumed = torch.minimum(consumed, ending - positions + 1)
    last = positions + consumed - 1
    environment_index = torch.arange(environments, device=components.device)[None]
    endpoint_index = last * environments + environment_index
    endpoint_ends = ends.flatten()[endpoint_index.flatten()]
    # Separate FP64 time streams; short tiles avoid Inductor's long-scan bug.
    streams = components.double().permute(1, 2, 0).contiguous().flatten(0, 1)
    block = 128
    blocks = (steps + block - 1) // block
    tiled = F.pad(streams, (0, blocks * block - steps)).reshape(-1, block).cumsum(-1)
    tiled = tiled.view(streams.shape[0], blocks, block)
    totals = tiled[..., -1]
    offsets = totals.cumsum(-1) - totals
    prefix = (tiled + offsets[..., None]).flatten(1)[:, :steps]
    prefix = prefix.view(environments, channels, steps).permute(2, 0, 1)
    before = prefix[(positions - 1).clamp_min(0), environment_index] * (positions > 0)[..., None]
    observed = prefix[last, environment_index] - before
    return (observed.flatten(0, 1).to(components.dtype), endpoint_index.flatten(),
            consumed.flatten(), endpoint_ends)


def lift_returns(returns, frequencies, horizon=1000):
    """Fixed, decoder-free embedding of a realized signed vector return."""
    channels, count = returns.shape[-1], frequencies.shape[0]
    phase = (returns / horizon) @ frequencies.T
    return torch.cat((returns / (horizon * math.sqrt(channels)),
                      phase.cos() / math.sqrt(count), phase.sin() / math.sqrt(count)), -1)


def distributional_td_target(observed, future, ends, frequencies, horizon=1000):
    """Translate a continuation embedding by an actual observed vector segment.

    The fixed horizon normalization makes complex multiplication an exact
    composition rule. Average continuation embeddings, never exponentiate their
    mean returns. True ends have a point mass at the zero remaining return.
    """
    channels, count = observed.shape[-1], frequencies.shape[0]
    means, cosine, sine = future.detach().split((channels, count, count), -1)
    means = torch.where(ends[:, None], 0., means)
    cosine = torch.where(ends[:, None], 1 / math.sqrt(count), cosine)
    sine = torch.where(ends[:, None], 0., sine)
    phase = (observed / horizon) @ frequencies.T
    real, imag = phase.cos(), phase.sin()
    return torch.cat((observed / (horizon * math.sqrt(channels)) + means,
                      real * cosine - imag * sine, imag * cosine + real * sine), -1).detach()


class PredictiveCritic(nn.Module):
    """One latent predicts means and finite joint return-characteristic moments.

    Complex moments live in the unit disk. This necessary constraint does not
    guarantee a valid joint distribution or identify its complete law.
    """
    def __init__(self, state_dim, action_dim, width=192, horizon=1000):
        super().__init__()
        self.action_dim, self.components_count, self.horizon = action_dim, action_dim + 1, horizon
        channels = self.components_count
        generator = torch.Generator().manual_seed(1729)
        random_directions = torch.randn(max(0, 32 - channels - 1), channels, generator=generator)
        random_directions = F.normalize(random_directions, dim=-1)
        directions = torch.cat((torch.eye(channels), torch.ones(1, channels) / math.sqrt(channels),
                                random_directions), 0)
        frequencies = (directions[:, None, :] * torch.tensor([.5, 2., 8., 32.])[None, :, None]).flatten(0, 1)
        self.register_buffer('frequencies', frequencies)
        self.feature_dim = channels + 2 * frequencies.shape[0]
        self.trunk = make_norm_residual_trunk(state_dim + action_dim + 2, width,
                                            placement='pre', norm_kind='rms', activation='stiglu')
        self.head = nn.Linear(width, self.feature_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def representation(self, observations, age, actions):
        phase = age.to(observations.dtype)[:, None] / self.horizon
        inputs = torch.cat((observations, 2 * actions - 1, phase, 1 - phase), -1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=observations.dtype == torch.float32):
            hidden = self.trunk(inputs)
        return hidden.to(self.head.weight.dtype)

    def predict(self, observations, age, actions):
        hidden = self.representation(observations, age, actions)
        count = self.frequencies.shape[0]
        means, amplitude, phase = self.head(hidden).split((self.components_count, count, count), -1)
        amplitude = amplitude.sigmoid() / math.sqrt(count)
        means = torch.where((age < self.horizon)[:, None], means, 0.)
        cosine = torch.where((age < self.horizon)[:, None], amplitude * phase.cos(), 1 / math.sqrt(count))
        sine = torch.where((age < self.horizon)[:, None], amplitude * phase.sin(), 0.)
        return torch.cat((means, cosine, sine), -1)

    def components(self, observations, age, actions):
        hidden = self.representation(observations, age, actions)
        count = self.components_count
        means = F.linear(hidden, self.head.weight[:count], self.head.bias[:count])
        return torch.where((age < self.horizon)[:, None], means, 0.) * (self.horizon * math.sqrt(count))


def beta_log_prob(parameters, actions):
    alpha, beta = parameters.double().unbind(-1)
    alpha, beta = alpha[:, None], beta[:, None]
    action = actions.double()
    partition = alpha.lgamma() + beta.lgamma() - (alpha + beta).lgamma()
    return ((alpha - 1) * action.log() + (beta - 1) * torch.log1p(-action) - partition).sum(-1)


def credit_metric_loss(residual, score_energy, weights, channels, horizon, td_steps):
    """Full-rank vector TD loss in local policy-credit units.

    score_energy = ||L^-1 score(a)||^2 / (2 * action_dim), frozen under the
    collection policy. Its population expectation is one. M=(I+11^T)/2
    retains component errors and emphasizes the actual sum-return direction.
    The fixed TD segment length sets units, not a second prediction horizon.
    This weights sampled residual credit, not the complete corrected estimator.
    """
    means = residual[:, :channels] * (horizon / td_steps)
    mean_loss = .25 * score_energy * (means.square().sum(-1) + means.sum(-1).square())
    moment_loss = .5 * residual[:, channels:].square().sum(-1)
    return ((mean_loss + moment_loss) * weights).sum() / weights.sum()


def corrected_weights(model_components, observed_components, target_components):
    """Independent-action leave-one-out baselines preserve the correction identity."""
    count = model_components.shape[1]
    if count < 2:
        raise ValueError('leave-one-out integration requires at least two actions')
    model = model_components.double()
    baseline = (model.sum(1, keepdim=True) - model) / (count - 1)
    modeled = (model - baseline) / count
    observed = target_components.double() - observed_components.double()
    return torch.cat((modeled, observed[:, None]), 1).detach()


def corrected_credit(parameters, actions, vector_weights):
    alpha, beta = parameters.double().unbind(-1)
    total = alpha + beta
    first = actions.double().log() - (alpha.digamma() - total.digamma())[:, None]
    second = torch.log1p(-actions.double()) - (beta.digamma() - total.digamma())[:, None]
    scores = torch.stack((first, second), -1)
    return (scores * vector_weights.double().sum(-1)[..., None, None]).sum(1).to(parameters.dtype)


def corrected_gain_kl(logits, reference, actions, old_logprob, utility_weights, weights):
    """Frozen target surrogate; finite sample optimization still can overfit it."""
    # Same FP32 softplus-to-parameter map as beta_geometry, then FP64 statistics.
    alpha, beta = (F.softplus(logits) + 1).double().chunk(2, -1)
    parameters = torch.stack((alpha, beta), -1)
    ratio_delta = torch.expm1(beta_log_prob(parameters, actions) - old_logprob)
    gain = (ratio_delta * utility_weights.double()).sum(-1)
    old_alpha, old_beta, partition, old_da, old_db = reference.unbind(-1)
    total = alpha + beta
    kl = (alpha.lgamma() + beta.lgamma() - total.lgamma() - partition
          + (old_alpha - alpha) * old_da + (old_beta - beta) * old_db).sum(-1)
    denominator = weights.sum().clamp_min(1.)
    return (gain * weights).sum() / denominator, (kl * weights).sum() / denominator


def sample_actions(logits, count):
    alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
    result = Beta(alpha, beta, validate_args=False).sample((count,)).permute(1, 0, 2).contiguous()
    epsilon = torch.finfo(result.dtype).eps
    return result.clamp(epsilon, 1 - epsilon)


def main():
    args = tyro.cli(Args)
    if args.env_id != 'HalfCheetah-v4' or args.num_envs < 2 or args.num_steps < 1000:
        raise ValueError('requires fixed-horizon HalfCheetah and a rollout of at least one horizon')
    if min(args.total_timesteps, args.td_steps, args.critic_width, args.critic_epochs, args.critic_learning_rate,
           args.max_grad_norm, args.cg_iterations, args.cg_damping, args.trust_kl) <= 0 or args.line_search_steps < 5 or args.action_samples < 2:
        raise ValueError('positive settings and at least five line-search evaluations required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA and BF16 support required')
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision='highest', allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    batch_size = args.num_envs * args.num_steps
    run_name = f'{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}'
    run_dir = Path('runs') / run_name
    with ExitStack() as resources:
        writer = SummaryWriter(str(run_dir))
        resources.callback(writer.close)
        writer.add_text('hyperparameters', '|param|value|\n|-|-|\n' + '\n'.join(f'|{k}|{v}|' for k, v in vars(args).items()))
        writer.add_text('method', 'Unified joint characteristic embedding fitted with a full-rank component/utility credit metric and frozen score weights; pre-fit model integration plus observed vector TD correction; fresh full batches; no GAE/replay/decoding/contrastive loss/target sweeps.')
        probe = gym.make(args.env_id)
        try:
            control_weight = float(probe.unwrapped._ctrl_cost_weight)
        finally:
            probe.close()
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        horizon = episode_horizon(args.env_id)
        if horizon != 1000:
            raise ValueError('requires the 1000-step HalfCheetah episode horizon')
        obs_shape = envs.single_observation_space.shape
        agent = Agent(envs).to(device)
        critic = PredictiveCritic(int(np.prod(obs_shape)), agent.action_dim, args.critic_width, horizon).to(device)
        optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate, eps=1e-5, fused=True)
        trust = ActorNaturalGradient(agent.actor, args.trust_kl, args.cg_damping, args.cg_iterations,
                                     args.cg_relative_tolerance, args.line_search_steps, args.compile, args.compile_mode)
        obs_calibration = RecordingObsNorm(VectorObsNorm(args.num_envs, obs_shape))
        reward_norm = VectorRewardNorm(args.num_envs, 1.)
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        sampler = make_beta_sampler(args.num_envs, agent.action_dim,
                                   agent.action_low.cpu().numpy(), agent.action_high.cpu().numpy())
        sampler_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 7]))

        def act(observations):
            native, physical = sampler(host_actor(observations), sampler_rng)
            if not np.isfinite(physical).all():
                raise FloatingPointError('nonfinite actions')
            return native, physical.reshape((args.num_envs,) + agent.action_shape)

        started = time.perf_counter()
        warm = run_phase_warmup(envs, obs_norm=obs_calibration, rew_norm=reward_norm,
                                act_fn=lambda x: act(x)[1], horizon=horizon,
                                phase_offsets=compute_phase_offsets(args.num_envs, horizon, args.seed), seed=args.seed)
        obs_norm = FixedAffineObsNorm(obs_calibration.base, args.num_envs, obs_shape)
        next_obs_np = obs_norm.normalize(obs_calibration.last_raw)
        global_step = warm.transitions
        suppress = warm.suppress_mask.copy()
        current_age = np.full(args.num_envs, -1, dtype=np.int64)
        ages = np.empty((args.num_steps, args.num_envs), dtype=np.int64)
        ends = np.empty_like(ages, dtype=bool)
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                  non_blocking=args.non_blocking_transfers,
                                  fields={'observations': obs_shape, 'next_observations': obs_shape,
                                          'native_actions': (agent.action_dim,), 'raw_rewards': ()})
        resources.callback(transfer.close)

        prediction = critic.predict

        def critic_loss(obs, age, actions, target, score_energy, weights):
            residual = critic.predict(obs, age, actions) - target
            return credit_metric_loss(residual, score_energy, weights,
                                      critic.components_count, horizon, args.td_steps)

        def continuation(obs, age, actions):
            count = actions.shape[1]
            expanded_obs = obs[:, None].expand(-1, count, -1).flatten(0, 1)
            expanded_age = age[:, None].expand(-1, count).flatten()
            return critic.predict(expanded_obs, expanded_age, actions.flatten(0, 1)).view(
                obs.shape[0], count, -1).mean(1)

        def actor_components(obs, age, actions):
            count = actions.shape[1]
            expanded_obs = obs[:, None].expand(-1, count, -1).flatten(0, 1)
            expanded_age = age[:, None].expand(-1, count).flatten()
            return critic.components(expanded_obs, expanded_age, actions.flatten(0, 1)).view(
                obs.shape[0], count, -1)

        def measure_function(obs, reference, actions, old_logprob, utility_weights, weights):
            return corrected_gain_kl(agent.actor(obs), reference, actions, old_logprob, utility_weights, weights)

        policy_model = agent.actor.forward
        geometry = beta_geometry
        age_function = resolved_ages
        segment_function = td_segments
        target_function = distributional_td_target
        credit_function = corrected_credit
        if args.compile:
            policy_model = graph_compile(policy_model)
            prediction = torch.compile(prediction, fullgraph=True, mode=args.compile_mode)
            critic_loss = torch.compile(critic_loss, fullgraph=True, mode=args.compile_mode)
            continuation = torch.compile(continuation, fullgraph=True, mode=args.compile_mode)
            actor_components = torch.compile(actor_components, fullgraph=True, mode=args.compile_mode)
            measure_function = torch.compile(measure_function, fullgraph=True, mode=args.compile_mode)
            geometry = torch.compile(geometry, fullgraph=True, mode=args.compile_mode)
            age_function = torch.compile(age_function, fullgraph=True, mode=args.compile_mode)
            segment_function = torch.compile(segment_function, fullgraph=True, mode=args.compile_mode)
            target_function = torch.compile(target_function, fullgraph=True, mode=args.compile_mode)
            credit_function = torch.compile(credit_function, fullgraph=True, mode=args.compile_mode)

        timer = PhaseTimer()
        recent_returns = deque(maxlen=100)
        progress = []
        trained_transitions = actor_updates = critic_steps = 0
        interval_start, interval_step = time.perf_counter(), global_step
        iterations = math.ceil(args.total_timesteps / batch_size)
        import signal
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        previous_handler = signal.signal(signal.SIGTERM, request_stop)
        resources.callback(signal.signal, signal.SIGTERM, previous_handler)
        for iteration in range(1, iterations + 1):
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span('rollout', use_cuda=False):
                    obs_step = next_obs_np
                    native, physical = act(obs_step)
                    ages[step] = current_age
                with timer.span('env', use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(physical)
                with timer.span('normalize_transfer', use_cuda=False):
                    if terms.any():
                        raise ValueError('early termination is outside this finite-horizon experiment')
                    normalized_reward = reward_norm.normalize(raw_reward, terms)
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(step, normalized_reward, terms, truncs, observations=obs_step,
                                  next_observations=transition_obs, native_actions=native, raw_rewards=raw_reward)
                    ends[step] = terms | truncs
                    current_age = np.where(terms | truncs, 0, np.where(current_age >= 0, current_age + 1, -1))
                global_step += args.num_envs
                for index, info in enumerate(infos.get('final_info', ())):
                    if info and 'episode' in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        value = float(info['episode']['r'])
                        recent_returns.append(value)
                        writer.add_scalar('charts/episodic_return', value, global_step)
                        writer.add_scalar('charts/episodic_length', float(info['episode']['l']), global_step)
            diagnostics = {}
            with timer.span('targets'), torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                batch = transfer.upload()
                observations = batch.fields['observations'].flatten(0, 1)
                next_observations = batch.fields['next_observations'].flatten(0, 1)
                native_actions = batch.fields['native_actions'].flatten(0, 1)
                physical_actions = agent.action_low + (agent.action_high - agent.action_low) * native_actions
                components = reward_components(batch.fields['raw_rewards'].flatten(), physical_actions, control_weight)
                ends_tensor = torch.as_tensor(ends.copy(), device=device)
                age = age_function(torch.as_tensor(ages.copy(), device=device), ends_tensor).clone()
                weights = (age >= 0).float()
                if not bool(weights.sum() > 0):
                    raise RuntimeError('no resolved episode ages')
                age = age.clamp_min(0)
                observed, endpoint_index, consumed, endpoint_ends = (x.clone() for x in segment_function(
                    components.view(args.num_steps, args.num_envs, -1), ends_tensor, args.td_steps))
                endpoint_observations = next_observations[endpoint_index]
                # All real-segment continuations snapshot the same pre-fit critic.
                next_logits = policy_model(endpoint_observations).clone()
                endpoint_actions = sample_actions(next_logits, args.action_samples)
                future = continuation(endpoint_observations, age + consumed, endpoint_actions).clone()
                target = target_function(observed, future, endpoint_ends, critic.frequencies, horizon).clone()
                logits = policy_model(observations).clone()
                whitened_score, collection_factor, parameters = (x.clone() for x in geometry(logits, native_actions))
                score_energy = whitened_score.square().mean((-2, -1))
                diagnostics['critic/score_energy_mean'] = (score_energy * weights).sum() / weights.sum()
                diagnostics['critic/score_energy_max'] = torch.where(weights > 0, score_energy, 0.).max()
                if not args.credit_score_weighting:
                    score_energy = torch.ones_like(score_energy)
                reference = beta_kl_reference(parameters)
                sampled_actions = sample_actions(logits, args.action_samples)
                candidate_actions = torch.cat((sampled_actions, native_actions[:, None]), 1)
                # Materialize ALL critic-based actor weights before fitting this rollout.
                prefit_components = actor_components(observations, age, candidate_actions).clone()
                target_components = target[:, :critic.components_count] * (horizon * math.sqrt(critic.components_count))
                vector_weights = corrected_weights(prefit_components[:, :-1], prefit_components[:, -1], target_components)
                utility_weights = vector_weights.sum(-1)
                old_logprob = beta_log_prob(parameters, candidate_actions)
                credit = credit_function(parameters, candidate_actions, vector_weights).clone()
                model_credit = corrected_credit(parameters, candidate_actions[:, :-1], vector_weights[:, :-1])
                residual_credit = corrected_credit(parameters, candidate_actions[:, -1:], vector_weights[:, -1:])
                diagnostics['credit/alpha_beta_rms'] = credit.square().mean().sqrt()
                diagnostics['credit/model_alpha_beta_rms'] = model_credit.square().mean().sqrt()
                diagnostics['credit/residual_alpha_beta_rms'] = residual_credit.square().mean().sqrt()
                diagnostics['credit/model_residual_cosine'] = (model_credit * residual_credit).sum() / (
                    model_credit.norm() * residual_credit.norm()).clamp_min(1e-30)
                diagnostics['targets/component_rms'] = target_components.square().mean().sqrt()
                diagnostics['credit/model_vector_rms'] = vector_weights[:, :-1].square().mean().sqrt()
                diagnostics['credit/observed_residual_rms'] = vector_weights[:, -1].square().mean().sqrt()
                trained_transitions += int(weights.sum())
                diagnostics['data/valid_fraction'] = weights.mean()
                diagnostics['data/trained_transitions'] = trained_transitions
                diagnostics['targets/observed_steps_mean'] = consumed.float().mean()
                diagnostics['targets/observed_vector_rms'] = observed.square().mean().sqrt()
                diagnostics['targets/continuation_rms'] = future.square().mean().sqrt()
                diagnostics['targets/vector_rms'] = target.square().mean().sqrt()
                diagnostics['policy/concentration'] = (parameters.sum(-1).mean(-1) * weights).sum() / weights.sum()
                diagnostics['policy/entropy'] = (Beta(parameters[..., 0], parameters[..., 1], validate_args=False).entropy().sum(-1) * weights).sum() / weights.sum()

            with timer.span('update'):
                fitting_metrics = []
                for _ in range(args.critic_epochs):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    optimizer.zero_grad(set_to_none=True)
                    loss = critic_loss(observations, age, native_actions, target, score_energy, weights)
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm, foreach=True)
                    optimizer.step()
                    critic_steps += 1
                    fitting_metrics.append(torch.stack((loss.detach(), norm.detach())))
                    del loss
                optimizer.zero_grad(set_to_none=True)
                diagnostics['losses/vector_td'], diagnostics['grad/critic_preclip_norm'] = torch.stack(fitting_metrics).mean(0).unbind()
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    fitted = prediction(observations, age, native_actions).clone()
                    residual = fitted - target
                    diagnostics['critic/td_residual_rms'] = (residual.square().mean(-1) * weights).sum().div(weights.sum()).sqrt()
                    mean_residual = residual[:, :critic.components_count] * (horizon * math.sqrt(critic.components_count))
                    diagnostics['critic/component_td_rms'] = (mean_residual.square().mean(-1) * weights).sum().div(weights.sum()).sqrt()
                    diagnostics['critic/moment_td_rms'] = residual[:, critic.components_count:].square().mean().sqrt()
                    diagnostics['losses/credit_metric_postfit'] = credit_metric_loss(
                        residual, score_energy, weights, critic.components_count, horizon, args.td_steps)
                    diagnostics['critic/utility_td_rms'] = (mean_residual.sum(-1).square() * weights).sum().div(weights.sum()).sqrt()
                    diagnostics['losses/mean_block_postfit'] = .5 * (residual[:, :critic.components_count].square().sum(-1) * weights).sum() / weights.sum()
                    diagnostics['losses/moment_block_postfit'] = .5 * (residual[:, critic.components_count:].square().sum(-1) * weights).sum() / weights.sum()
                    diagnostics['losses/optimized_mean_block_postfit'] = (
                        diagnostics['losses/credit_metric_postfit'] - diagnostics['losses/moment_block_postfit'])

                def measure():
                    gain, kl = measure_function(observations, reference, candidate_actions, old_logprob, utility_weights, weights)
                    return gain.clone(), kl.clone()

                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    actor_metrics = trust.step(observations, logits, collection_factor, credit, weights, measure)
                    diagnostics.update(actor_metrics)
                    accepted_logits = policy_model(observations).clone()
                    alpha, beta = (F.softplus(accepted_logits) + 1).chunk(2, -1)
                    ratios = (beta_log_prob(torch.stack((alpha, beta), -1), candidate_actions) - old_logprob).exp()
                    ratio_mean = (ratios.mean(-1) * weights).sum() / weights.sum()
                    ratio_second = (ratios.square().mean(-1) * weights).sum() / weights.sum()
                    diagnostics['policy/importance_max'] = torch.where(weights[:, None] > 0, ratios, 0.).max()
                    diagnostics['policy/importance_mean'] = ratio_mean
                    diagnostics['policy/importance_ess_fraction'] = ratio_mean.square() / ratio_second

                    actor_updates += int(actor_metrics['policy/accepted_scale'] > 0)
            diagnostics.update({'updates/actor_attempted': iteration, 'updates/actor_accepted': actor_updates,
                                'updates/critic_optimizer': critic_steps, 'updates/target_snapshots': iteration,
                                'charts/episodic_return_mean_100': float(np.mean(recent_returns))})
            logged = gather_metrics({key: torch.as_tensor(value, device=device) for key, value in diagnostics.items()})
            if not all(np.isfinite(value) for value in logged.values()):
                raise FloatingPointError('nonfinite vector TD diagnostics')
            now = time.perf_counter()
            logged['charts/SPS'] = global_step / (now - started)
            logged['charts/interval_SPS'] = (global_step - interval_step) / (now - interval_start)
            for phase, timing in timer.summary().items():
                logged[f'timing/{phase}_s'] = timing['total_s']
            timer.reset()
            for key, value in logged.items():
                writer.add_scalar(key, value, global_step)
            writer.flush()
            progress.append(dict(step=global_step, iteration=iteration, **logged))
            atomic_json(run_dir / 'progress.json', progress)
            print(f'iteration={iteration}/{iterations} step={global_step} metrics={json.dumps(logged)}', flush=True)
            interval_start, interval_step = now, global_step
            if stop_requested:
                break

        report = dict(args=vars(args), status='cancelled' if stop_requested else 'completed',
                      fresh_initialization=True, checkpoint_loaded=False, gae_used=False, replay_used=False,
                      scalar_value_network_used=False, learned_dynamics_used=False, gamma=1., td_steps=args.td_steps,
                      horizon_query_heads=0, separate_value_network=False,
                      target_snapshots=len(progress), target_refreshes_per_rollout=1, transitions=global_step,
                      trained_transitions=trained_transitions, actor_updates_attempted=len(progress),
                      actor_updates_accepted=actor_updates, critic_optimizer_steps=critic_steps,
                      final_return_mean_100=float(np.mean(recent_returns)), final_returns=list(recent_returns),
                      control_weight=control_weight, progress=progress,
                      objective='Joint future vector distribution fitted with full-rank component/utility TD error metric and frozen policy-score weights; pre-fit observed-residual-corrected actor',
                      limitations='Single seed, training returns. Credit metric targets sampled residual credit, not the complete corrected gradient variance. Finite characteristic features and unit-disk constraints do not identify a complete joint distribution. Actor correction removes current-action model bias for fixed candidates, not continuation bias or finite-sample selection error. Unclipped importance ratios can have high variance. Natural-gradient solves are approximate.')
        atomic_json(run_dir / 'result.json', report)
        if args.save_model:
            torch.save(dict(actor=agent.actor.state_dict(), critic=critic.state_dict(),
                            normalization_mean=obs_norm.mean, normalization_inverse_std=obs_norm.inverse_std,
                            args=vars(args)), run_dir / f'{args.exp_name}.cleanrl_model')
        print('TRAINING_RESULT=' + json.dumps({key: value for key, value in report.items()
                                              if key not in ('args', 'progress', 'final_returns')}), flush=True)


if __name__ == '__main__':
    main()

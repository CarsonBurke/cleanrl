# Unified vector TD v16: one state-action prediction of the remaining future.
# A single latent coefficient tensor and shared trunk produce every component.
# The same function supplies observed-action predictions, real-segment TD
# continuations, and exact Beta policy improvement. No horizon-specific queries,
# separate value/response networks, GAE, replay, or repeated target sweeps.
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
    exp_name: str = 'vector_unified_td_v16'
    seed: int = 1
    env_id: str = 'HalfCheetah-v4'
    total_timesteps: int = 8000000
    num_envs: int = 16
    num_steps: int = 2048
    td_steps: int = 32
    critic_width: int = 192
    critic_epochs: int = 18
    critic_learning_rate: float = .0003
    max_grad_norm: float = .5
    trust_kl: float = .03
    cg_iterations: int = 50
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


def conjugate_gradient(operator, rhs, iterations, relative_tolerance=1e-6):
    """Fixed-budget GPU CG with tensor-only stopping and zero-RHS safeguards."""
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    squared = residual.dot(residual)
    threshold = squared * relative_tolerance ** 2
    tiny = torch.finfo(rhs.dtype).tiny
    for _ in range(iterations):
        product = operator(direction)
        curvature = direction.dot(product)
        active = (squared > threshold) & (curvature > tiny) & torch.isfinite(curvature)
        step = torch.where(active, squared / curvature.clamp_min(tiny), 0.)
        solution = solution + step * direction
        residual = residual - step * torch.where(active, product, 0.)
        new_squared = residual.dot(residual)
        ratio = torch.where(active, new_squared / squared.clamp_min(tiny), 0.)
        direction = residual + ratio * direction
        squared = new_squared
    return solution, squared.sqrt() / rhs.norm().clamp_min(tiny)


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

        self.gradient = gradient
        self.fisher = fisher
        if compile:
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

        direction, residual = conjugate_gradient(lambda v: undamped(v) + self.damping * v,
                                                rhs, self.cg_iterations, self.relative_tolerance)
        curvature = direction.dot(undamped(direction))
        gain_direction = gradient.dot(direction)
        safe = (curvature > 0) & (gain_direction > 0) & torch.isfinite(curvature) & torch.isfinite(gain_direction)
        scale = torch.where(safe, (2 * self.budget / curvature.clamp_min(torch.finfo(curvature.dtype).tiny)).sqrt(), 0.)
        gain, kl, accepted, evaluations, cap_hits = self.search(before, direction, scale, measure)
        return {'policy/accepted_gain': gain, 'policy/exact_joint_kl': kl,
                'policy/accepted_scale': accepted, 'policy/line_search_evaluations': evaluations,
                'policy/line_search_cap_hits': cap_hits, 'natural/cg_residual_ratio': residual,
                'natural/undamped_curvature': curvature, 'natural/g_dot_direction': gain_direction,
                'natural/gradient_norm': gradient.norm()}



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


def vector_td_target(observed, future, endpoint_ends):
    """One detached vector correction of the same prediction at a real endpoint."""
    return (observed + torch.where(endpoint_ends[:, None], 0., future.detach())).detach()


def basis_geometry(actions):
    """A policy-independent Beta(2,2) feature coordinate system."""
    alpha = torch.full_like(actions, 2., dtype=torch.float64)
    covariance = -torch.polygamma(1, 2 * alpha)
    diagonal = torch.polygamma(1, alpha) + covariance
    l11 = diagonal.sqrt()
    l21 = covariance / l11
    l22 = (diagonal - l21.square()).sqrt()
    first = (actions.double().log() + 5 / 6) / l11
    second = (torch.log1p(-actions.double()) + 5 / 6 - l21 * first) / l22
    return (torch.stack((first, second), -1).to(actions.dtype),
            torch.stack((l11, l21, l22), -1).to(actions.dtype),
            torch.stack((alpha, alpha), -1).to(actions.dtype))


def kernels_observed(actions, powers, indices=None):
    x = actions.double()[:, None, :] if indices is None else actions.double()[:, indices]
    p, q = powers.double().unbind(-1)
    return (x.log() * p + torch.log1p(-x) * q).sum(-1).exp().to(actions.dtype)


def kernel_statistics(parameters, powers, indices=None):
    """Return moments in FP64 so variance/centering do not cancel in FP32."""
    selected = parameters.double()[:, None] if indices is None else parameters.double()[:, indices]
    alpha, beta = selected.unbind(-1)
    p, q = powers.double().unbind(-1)
    normalizer = alpha.lgamma() + beta.lgamma() - (alpha + beta).lgamma()

    def moment(multiplier):
        ap, bq = alpha + multiplier * p, beta + multiplier * q
        return (ap.lgamma() + bq.lgamma() - (ap + bq).lgamma() - normalizer).sum(-1).exp()

    mean = moment(1)
    std = (moment(2) - mean.square()).clamp_min(1e-12).sqrt()
    return mean, std


def kernel_moments(parameters, powers, indices=None):
    """Analytic Beta mean and its alpha/beta derivatives, optionally sparse."""
    selected = parameters.double()[:, None] if indices is None else parameters.double()[:, indices]
    alpha, beta = selected.unbind(-1)
    p, q = powers.double().unbind(-1)
    total = alpha + beta
    combined = total + p + q
    mean = (torch.lgamma(alpha + p) + torch.lgamma(beta + q) - torch.lgamma(combined)
            - alpha.lgamma() - beta.lgamma() + total.lgamma()).sum(-1).exp()
    da = (alpha + p).digamma() - combined.digamma() - alpha.digamma() + total.digamma()
    db = (beta + q).digamma() - combined.digamma() - beta.digamma() + total.digamma()
    derivative = mean[..., None, None] * torch.stack((da, db), -1)
    if indices is not None:
        expanded = indices[None, :, :, None].expand(parameters.shape[0], -1, -1, 2)
        derivative = torch.zeros(parameters.shape[0], powers.shape[0], parameters.shape[1], 2,
                                 device=parameters.device, dtype=torch.float64).scatter_add(2, expanded, derivative)
    return mean.to(parameters.dtype), derivative.to(parameters.dtype)


class UnifiedCritic(nn.Module):
    """One learned state-action function for the remaining vector return.

    The constant basis term belongs to the same coefficient tensor as every
    action feature; there is no separately fitted state-only or horizon head.
    """
    def __init__(self, state_dim, action_dim, width=192, horizon=1000):
        super().__init__()
        self.action_dim, self.components, self.horizon = action_dim, action_dim + 1, horizon
        initial = []
        for actuator in range(action_dim):
            for side in range(2):
                power = torch.zeros(action_dim, 2)
                power[actuator, side] = 2.
                initial.append(power)
        for first in range(action_dim):
            for second in range(first + 1, action_dim):
                power = torch.zeros(action_dim, 2)
                power[first, 0] = power[second, 0] = 1.
                initial.append(power)
        if action_dim >= 3:
            for start in range(action_dim):
                power = torch.zeros(action_dim, 2)
                for offset in range(3):
                    power[(start + offset) % action_dim, (start + offset) % 2] = 1.
                initial.append(power)
        for side in range(2):
            power = torch.zeros(action_dim, 2)
            power[:, side] = 1.
            initial.append(power)
        initial = torch.stack(initial)
        self.register_buffer('power_mask', (initial > 0).float())
        fraction = torch.where(initial > 0, initial / 4, .5)
        self.features = 1 + 2 * action_dim + initial.shape[0]
        self.kernel_indices = None

        self.trunk = nn.Sequential(nn.Linear(state_dim + 2, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.coefficient_head = nn.Linear(width, self.components * self.features)
        self.power_head = nn.Linear(width, initial.numel())
        nn.init.normal_(self.power_head.weight, std=.005)
        with torch.no_grad():
            self.power_head.bias.copy_(torch.logit(fraction).flatten())
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)

    def representation(self, observations, age):
        phase = age.to(observations.dtype)[:, None] / self.horizon
        inputs = torch.cat((observations, phase, 1 - phase), -1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=observations.dtype == torch.float32):
            hidden = self.trunk(inputs)
        hidden = hidden.to(self.coefficient_head.weight.dtype)
        coefficients = self.coefficient_head(hidden).view(-1, self.components, self.features)
        coefficients = torch.where((age < self.horizon)[:, None, None], coefficients, 0.)
        powers = 4 * self.power_head(hidden).view(-1, *self.power_mask.shape).sigmoid() * self.power_mask
        return coefficients, powers

    def coefficients(self, observations, age):
        return self.representation(observations, age)[0]

    def powers(self, observations, age):
        return self.representation(observations, age)[1]

    def predict(self, observations, age, actions):
        coefficients, powers = self.representation(observations, age)
        scores, _, anchor = basis_geometry(actions)
        mean, std = kernel_statistics(anchor, powers)
        observed = kernels_observed(actions.double(), powers)
        features = torch.cat((torch.ones_like(actions[:, :1]), scores.flatten(1),
                              ((observed - mean) / std).to(actions.dtype)), -1)
        return torch.einsum('bcf,bf->bc', coefficients, features)

    def expected(self, observations, age, parameters):
        coefficients, powers = self.representation(observations, age)
        _, factor, anchor = basis_geometry(torch.full_like(parameters[..., 0], .5))
        basis_mean, basis_std = kernel_statistics(anchor, powers)
        current_mean, _ = kernel_statistics(parameters, powers)
        alpha, beta = parameters.double().unbind(-1)
        total = alpha + beta
        da, db = alpha.digamma() - total.digamma() + 5 / 6, beta.digamma() - total.digamma() + 5 / 6
        l11, l21, l22 = factor.double().unbind(-1)
        first = da / l11
        second = (db - l21 * first) / l22
        features = torch.cat((torch.ones_like(alpha[:, :1]), torch.stack((first, second), -1).flatten(1),
                              (current_mean - basis_mean) / basis_std), -1)
        return torch.einsum('bcf,bf->bc', coefficients, features.to(coefficients.dtype))


def response_credit(parameters, factor, coefficients, powers, indices, kernel_std):
    """Exact derivative of the integrated response at the collection policy."""
    action_dim = parameters.shape[1]
    first, second = coefficients[:, :2 * action_dim].reshape(-1, action_dim, 2).unbind(-1)
    l11, l21, l22 = factor.unbind(-1)
    # F_current L_basis^{-T} c; L_basis*c is valid only for a moving basis
    # whitened at the current policy, which would redefine a TD critic.
    log_a = first.double() / l11.double() - l21.double() * second.double() / (l11.double() * l22.double())
    log_b = second.double() / l22.double()
    alpha, beta = parameters.double().unbind(-1)
    covariance = -torch.polygamma(1, alpha + beta)
    fa, fb = torch.polygamma(1, alpha) + covariance, torch.polygamma(1, beta) + covariance
    current_l11 = fa.sqrt()
    current_l21 = covariance / current_l11
    current_l22 = (fb - current_l21.square()).sqrt()
    pushed_first = current_l11 * log_a + current_l21 * log_b
    pushed_second = current_l22 * log_b
    score_credit = torch.stack((current_l11 * pushed_first,
                                current_l21 * pushed_first + current_l22 * pushed_second), -1).to(first.dtype)
    _, derivative = kernel_moments(parameters.double(), powers, indices)
    kernel_credit = torch.einsum('bk,bkai->bai', coefficients[:, 2 * action_dim:].double() / kernel_std,
                                 derivative)
    return score_credit + kernel_credit.to(score_credit.dtype)


def integrated_gain_kl(logits, reference, factor, coefficients, powers, indices,
                       old_kernel_mean, old_kernel_std, weights):
    """Nonlinear modeled gain; frozen critic and collection-policy coordinates."""
    alpha, beta = (F.softplus(logits) + 1).double().chunk(2, -1)
    parameters = torch.stack((alpha, beta), -1)
    old_alpha, old_beta, old_partition, old_da, old_db = reference.unbind(-1)
    total = alpha + beta
    delta_a, delta_b = alpha.digamma() - total.digamma() - old_da, beta.digamma() - total.digamma() - old_db
    l11, l21, l22 = factor.double().unbind(-1)
    z1 = delta_a / l11
    z2 = (delta_b - l21 * z1) / l22
    kernel_mean, _ = kernel_statistics(parameters, powers, indices)
    expected = torch.cat((torch.stack((z1, z2), -1).flatten(1),
                          (kernel_mean - old_kernel_mean) / old_kernel_std), -1)
    gain = (expected * coefficients.double()).sum(-1)
    partition = alpha.lgamma() + beta.lgamma() - total.lgamma()
    kl = (partition - old_partition + (old_alpha - alpha) * old_da + (old_beta - beta) * old_db).sum(-1)
    denominator = weights.sum().clamp_min(1.)
    return (gain * weights).sum() / denominator, (kl * weights).sum() / denominator


def main():
    args = tyro.cli(Args)
    if args.env_id != 'HalfCheetah-v4' or args.num_envs < 2 or args.num_steps < 1000:
        raise ValueError('requires fixed-horizon HalfCheetah and a rollout of at least one horizon')
    if min(args.total_timesteps, args.td_steps, args.critic_width, args.critic_epochs, args.critic_learning_rate,
           args.max_grad_norm, args.cg_iterations, args.cg_damping, args.trust_kl) <= 0 or args.line_search_steps < 5:
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
        writer.add_text('method', 'Unified vector TD; one remaining-future function; fixed feature reference; learned joint-action powers; exact Beta expectations; fresh full batches; no GAE/replay/scalar value/imagined states/target sweeps.')
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
        critic = UnifiedCritic(int(np.prod(obs_shape)), agent.action_dim, args.critic_width, horizon).to(device)
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

        def critic_loss(obs, age, actions, target, weights, scale):
            residual = prediction(obs, age, actions) - target
            loss = .5 * (residual.square().mean(-1) * weights).sum() / (weights.sum() * scale)
            return loss

        def continuation(obs, age, logits):
            alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
            return critic.expected(obs, age, torch.stack((alpha, beta), -1))

        def actor_representation(obs, age):
            coefficients, powers = critic.representation(obs, age)
            # The constant coordinate cancels only in policy gain. TD uses all.
            return coefficients[:, :, 1:].sum(1), powers

        def measure_function(obs, reference, factor, coefficients, powers, mean, std, weights):
            return integrated_gain_kl(agent.actor(obs), reference, factor, coefficients, powers,
                                       None, mean, std, weights)

        policy_model = agent.actor.forward
        geometry, basis_function = beta_geometry, basis_geometry
        age_function = resolved_ages
        segment_function = td_segments
        target_function = vector_td_target
        statistics_function, credit_function = kernel_statistics, response_credit
        if args.compile:
            policy_model = graph_compile(policy_model)
            prediction = torch.compile(prediction, fullgraph=True, mode=args.compile_mode)
            critic_loss = torch.compile(critic_loss, fullgraph=True, mode=args.compile_mode)
            continuation = torch.compile(continuation, fullgraph=True, mode=args.compile_mode)
            actor_representation = torch.compile(actor_representation, fullgraph=True, mode=args.compile_mode)
            measure_function = torch.compile(measure_function, fullgraph=True, mode=args.compile_mode)
            geometry = torch.compile(geometry, fullgraph=True, mode=args.compile_mode)
            basis_function = torch.compile(basis_function, fullgraph=True, mode=args.compile_mode)
            age_function = torch.compile(age_function, fullgraph=True, mode=args.compile_mode)
            segment_function = torch.compile(segment_function, fullgraph=True, mode=args.compile_mode)
            target_function = torch.compile(target_function, fullgraph=True, mode=args.compile_mode)
            statistics_function = torch.compile(statistics_function, fullgraph=True, mode=args.compile_mode)
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
                future = continuation(endpoint_observations, age + consumed, next_logits).clone()
                target = target_function(observed, future, endpoint_ends).clone()
                # Reward-unit scaling cannot shrink TD pressure as predicted values grow.
                scale = ((components.square().mean(-1) * weights).sum() / weights.sum()).clamp_min(1.)
                logits = policy_model(observations).clone()
                _, collection_factor, parameters = (x.clone() for x in geometry(logits, native_actions))
                _, basis_factor, anchor = (x.clone() for x in basis_function(native_actions))
                reference = beta_kl_reference(parameters)
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
                    loss = critic_loss(observations, age, native_actions, target, weights, scale)
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
                    coefficients, powers = (x.clone() for x in actor_representation(observations, age))
                    kernel_mean, _ = (x.clone() for x in statistics_function(parameters, powers))
                    _, basis_std = (x.clone() for x in statistics_function(anchor, powers))
                    credit = credit_function(parameters, basis_factor, coefficients, powers, None, basis_std).clone()
                    diagnostics['credit/alpha_beta_rms'] = credit.square().mean().sqrt()
                    diagnostics['critic/response_coefficient_rms'] = coefficients.square().mean().sqrt()
                    diagnostics['critic/active_power_mean'] = powers.sum() / (critic.power_mask.sum() * batch_size)
                    diagnostics['critic/kernel_basis_std_min'] = basis_std.min()

                def measure():
                    gain, kl = measure_function(observations, reference, basis_factor, coefficients, powers,
                                                kernel_mean, basis_std, weights)
                    return gain.clone(), kl.clone()

                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    actor_metrics = trust.step(observations, logits, collection_factor, credit, weights, measure)
                    diagnostics.update(actor_metrics)
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
                      objective='Unified remaining-future vector action value, one shared function and one frozen real-segment TD target per fresh rollout',
                      limitations='Single seed, training returns. Vector TD has bootstrap bias and only one target refresh per rollout. Fixed-reference learned kernels restrict the action-response family. Exact model integration is not proof of environmental improvement. Natural-gradient solves are approximate.')
        atomic_json(run_dir / 'result.json', report)
        if args.save_model:
            torch.save(dict(actor=agent.actor.state_dict(), critic=critic.state_dict(),
                            normalization_mean=obs_norm.mean, normalization_inverse_std=obs_norm.inverse_std,
                            args=vars(args)), run_dir / f'{args.exp_name}.cleanrl_model')
        print('TRAINING_RESULT=' + json.dumps({key: value for key, value in report.items()
                                              if key not in ('args', 'progress', 'final_returns')}), flush=True)


if __name__ == '__main__':
    main()

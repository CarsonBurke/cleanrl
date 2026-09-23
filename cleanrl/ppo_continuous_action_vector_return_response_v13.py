# Vector return response v13: predict action-dependent future reward profiles.
# Separate state-only profile and latent response tensor; exact Beta expectations
# replace learned dynamics derivatives. Targets contain progress and every
# actuator cost across future intervals, from actual complete episode suffixes.
# Full-batch fitting; nonlinear KL-constrained actor improvement; no GAE,
# replay, scalar value head, imagined states, or Bellman target sweeps.
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
    exp_name: str = 'vector_return_response_v13'
    seed: int = 1
    env_id: str = 'HalfCheetah-v4'
    total_timesteps: int = 8000000
    num_envs: int = 16
    num_steps: int = 2048
    critic_width: int = 192
    baseline_epochs: int = 10
    response_epochs: int = 8
    baseline_learning_rate: float = .001
    response_learning_rate: float = .0003
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


PROFILE_EDGES = (0, 1, 8, 32, 128, 512, 1000)


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


def reward_profiles(rewards, physical_actions, ends, ages, control_weight,
                    edges=PROFILE_EDGES, horizon=1000):
    """Exact signed component sums on completed future suffixes, time-major.

    The left episode prefix may precede this policy; all future actions used in
    each target must belong to this rollout. Right-censored targets are masked.
    """
    steps, environments = rewards.shape
    positions = torch.arange(steps, device=rewards.device)[:, None].expand(-1, environments)
    ending = torch.where(ends, positions, steps).T.contiguous().flip(-1).cummin(-1).values.flip(-1).T
    valid = ending < steps
    resolved = torch.where((ages < 0) & valid, horizon - (ending - positions + 1), ages)
    cost = control_weight * physical_actions.double().square()
    components = torch.cat(((rewards.double() + cost.sum(-1))[..., None], -cost), -1)
    # Hierarchical parallel scan avoids Inductor's broken long SplitScan lowering.
    # This is target arithmetic over the full rollout, not minibatch fitting.
    streams = components.permute(1, 2, 0).contiguous().flatten(0, 1)
    block = 128
    blocks = (steps + block - 1) // block
    tiled = F.pad(streams, (0, blocks * block - steps)).reshape(-1, block).cumsum(-1)
    tiled = tiled.view(streams.shape[0], blocks, block)
    totals = tiled[..., -1]
    offsets = totals.cumsum(-1) - totals
    prefix = (tiled + offsets[..., None]).flatten(1)[:, :steps]
    prefix = prefix.view(environments, components.shape[-1], steps).permute(2, 0, 1)
    env_index = torch.arange(environments, device=rewards.device)[None, :]
    values, lengths = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        begin = torch.minimum(positions + lower, ending + 1).clamp_max(steps)
        stop = torch.minimum(positions + upper, ending + 1).clamp_max(steps)
        total = prefix[(stop - 1).clamp_min(0), env_index] * (stop > 0)[..., None]
        before = prefix[(begin - 1).clamp_min(0), env_index] * (begin > 0)[..., None]
        values.append((total - before) * valid[..., None])
        lengths.append((stop - begin) * valid)
    return (torch.stack(values, 2).flatten(0, 1).to(rewards.dtype), valid.flatten(), resolved.flatten(),
            torch.stack(lengths, 2).flatten(0, 1).to(rewards.dtype))


def band_lengths(ages, edges=PROFILE_EDGES, horizon=1000):
    remaining = (horizon - ages).clamp_min(0)
    return torch.stack([(remaining - lower).clamp(0, upper - lower)
                        for lower, upper in zip(edges[:-1], edges[1:])], -1)


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


class ResponseCritic(nn.Module):
    """State-conditioned profile-response tensor with learned integrable kernels.

    Baseline and response trunks have no shared parameters. The latent tensor
    retains temporal interval, physical reward component, and action feature.
    """
    def __init__(self, state_dim, action_dim, width=192, bands=6, horizon=1000):
        super().__init__()
        self.action_dim, self.components, self.bands, self.horizon = action_dim, action_dim + 1, bands, horizon
        indices, powers = [], []
        for actuator in range(action_dim):
            for side in range(2):
                indices.append((actuator, actuator))
                power = [[0., 0.], [0., 0.]]
                power[0][side] = 2.
                powers.append(power)
        for first in range(action_dim):
            for second in range(first + 1, action_dim):
                indices.append((first, second))
                powers.append([[1., 0.], [1., 0.]])
        initial = torch.tensor(powers)
        self.register_buffer('kernel_indices', torch.tensor(indices, dtype=torch.long))
        self.register_buffer('power_mask', (initial > 0).float())
        # A bounded broad family, not arbitrarily narrow unsupported kernels.
        fraction = torch.where(initial > 0, initial / 4, .5)
        self.power_logits = nn.Parameter(torch.logit(fraction))
        self.features = 2 * action_dim + len(indices)

        def trunk():
            return nn.Sequential(nn.Linear(state_dim + 2, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU())

        self.baseline_trunk, self.response_trunk = trunk(), trunk()
        self.baseline_head = nn.Linear(width, bands * self.components)
        self.response_head = nn.Linear(width, bands * self.components * self.features)
        for head in (self.baseline_head, self.response_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def inputs(self, observations, age):
        phase = age.to(observations.dtype)[:, None] / self.horizon
        return torch.cat((observations, phase, 1 - phase), -1)

    def baseline_rates(self, observations, age):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=observations.dtype == torch.float32):
            hidden = self.baseline_trunk(self.inputs(observations, age))
        with torch.autocast('cuda', enabled=False):
            return self.baseline_head(hidden.to(self.baseline_head.weight.dtype)).view(-1, self.bands, self.components)

    def coefficients(self, observations, age):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=observations.dtype == torch.float32):
            hidden = self.response_trunk(self.inputs(observations, age))
        with torch.autocast('cuda', enabled=False):
            return self.response_head(hidden.to(self.response_head.weight.dtype)).view(
                -1, self.bands, self.components, self.features)

    def powers(self):
        return 4 * self.power_logits.sigmoid() * self.power_mask

    def response_profile(self, observations, age, scores, actions, parameters, lengths):
        powers = self.powers()
        mean, std = kernel_statistics(parameters, powers, self.kernel_indices)
        observed = kernels_observed(actions.double(), powers, self.kernel_indices)
        features = torch.cat((scores.flatten(1), ((observed - mean) / std).to(scores.dtype)), -1)
        return torch.einsum('bhcf,bf->bhc', self.coefficients(observations, age), features) * lengths[..., None]


def response_credit(parameters, factor, coefficients, powers, indices, kernel_std):
    """Exact derivative of the integrated response at the collection policy."""
    action_dim = parameters.shape[1]
    first, second = coefficients[:, :2 * action_dim].reshape(-1, action_dim, 2).unbind(-1)
    l11, l21, l22 = factor.unbind(-1)
    score_credit = torch.stack((l11 * first, l21 * first + l22 * second), -1)
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
    if min(args.total_timesteps, args.critic_width, args.baseline_epochs, args.response_epochs,
           args.baseline_learning_rate, args.response_learning_rate, args.max_grad_norm,
           args.cg_iterations, args.cg_damping, args.trust_kl) <= 0 or args.line_search_steps < 5:
        raise ValueError('positive training settings and at least five line-search evaluations required')
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
        writer.add_text('method', 'Vector action-response profile; analytic Beta integration; fresh full batches; no GAE/replay/dynamics/scalar value/Bellman backup.')
        probe = gym.make(args.env_id)
        try:
            control_weight = float(probe.unwrapped._ctrl_cost_weight)
        finally:
            probe.close()
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        horizon = episode_horizon(args.env_id)
        if horizon != PROFILE_EDGES[-1]:
            raise ValueError('profile intervals must cover the actual episode horizon')
        obs_shape = envs.single_observation_space.shape
        agent = Agent(envs).to(device)
        critic = ResponseCritic(int(np.prod(obs_shape)), agent.action_dim, args.critic_width,
                                len(PROFILE_EDGES) - 1, horizon).to(device)
        baseline_parameters = [p for name, p in critic.named_parameters() if name.startswith('baseline_')]
        response_parameters = [p for name, p in critic.named_parameters() if not name.startswith('baseline_')]
        baseline_optimizer = torch.optim.Adam(baseline_parameters, lr=args.baseline_learning_rate, eps=1e-5, fused=True)
        response_optimizer = torch.optim.Adam(response_parameters, lr=args.response_learning_rate, eps=1e-5, fused=True)
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
                                  fields={'observations': obs_shape, 'native_actions': (agent.action_dim,), 'raw_rewards': ()})
        resources.callback(transfer.close)

        def baseline_prediction(obs, age, lengths):
            return critic.baseline_rates(obs, age) * lengths[..., None]

        def profile_loss(prediction, target, weights, scale):
            per_state = (prediction - target).square().mean((-1, -2))
            return .5 * (per_state * weights).sum() / (weights.sum() * scale)

        def baseline_loss(obs, age, lengths, target, weights, scale):
            return profile_loss(baseline_prediction(obs, age, lengths), target, weights, scale)

        def response_loss(obs, age, scores, actions, parameters, lengths, baseline, target, weights, scale):
            response = critic.response_profile(obs, age, scores, actions, parameters, lengths)
            return profile_loss(baseline + response, target, weights, scale)

        def actor_coefficients(obs, age, lengths):
            # Final objective contraction only; fitting preserves every profile channel.
            return (critic.coefficients(obs, age) * lengths[:, :, None, None]).sum((1, 2))

        def measure_function(obs, reference, factor, coefficients, powers, mean, std, weights):
            return integrated_gain_kl(agent.actor(obs), reference, factor, coefficients, powers,
                                       critic.kernel_indices, mean, std, weights)

        policy_model = agent.actor.forward
        geometry = beta_geometry
        target_function = reward_profiles
        lengths_function = band_lengths
        statistics_function = kernel_statistics
        credit_function = response_credit
        if args.compile:
            policy_model = graph_compile(policy_model)
            baseline_prediction = torch.compile(baseline_prediction, fullgraph=True, mode=args.compile_mode)
            baseline_loss = torch.compile(baseline_loss, fullgraph=True, mode=args.compile_mode)
            response_loss = torch.compile(response_loss, fullgraph=True, mode=args.compile_mode)
            actor_coefficients = torch.compile(actor_coefficients, fullgraph=True, mode=args.compile_mode)
            measure_function = torch.compile(measure_function, fullgraph=True, mode=args.compile_mode)
            geometry = torch.compile(geometry, fullgraph=True, mode=args.compile_mode)
            target_function = torch.compile(target_function, fullgraph=True, mode=args.compile_mode)
            lengths_function = torch.compile(lengths_function, fullgraph=True, mode=args.compile_mode)
            statistics_function = torch.compile(statistics_function, fullgraph=True, mode=args.compile_mode)
            credit_function = torch.compile(credit_function, fullgraph=True, mode=args.compile_mode)

        timer = PhaseTimer()
        recent_returns = deque(maxlen=100)
        progress = []
        trained_transitions = 0
        actor_updates = 0
        baseline_steps = 0
        response_steps = 0
        interval_start, interval_step = time.perf_counter(), global_step
        iterations = math.ceil(args.total_timesteps / batch_size)
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
                    next_obs_np, _ = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(step, normalized_reward, terms, truncs, observations=obs_step,
                                  native_actions=native, raw_rewards=raw_reward)
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
                native_actions = batch.fields['native_actions'].flatten(0, 1)
                physical_actions = agent.action_low + (agent.action_high - agent.action_low) * batch.fields['native_actions']
                target, valid, age, target_lengths = (x.clone() for x in target_function(
                    batch.fields['raw_rewards'], physical_actions, torch.as_tensor(ends.copy(), device=device),
                    torch.as_tensor(ages.copy(), device=device), control_weight))
                actor_weights = (age >= 0).float()
                weights = valid.float() * actor_weights
                if not bool(weights.sum() > 0):
                    raise RuntimeError('rollout has no complete future suffixes')
                age = age.clamp_min(0)
                # Censored states still have modeled future lengths for actor improvement.
                actor_lengths = lengths_function(age).clone().to(target.dtype)
                scale = ((target.square().mean((-1, -2)) * weights).sum() / weights.sum()).clamp_min(1.)
                logits = policy_model(observations).clone()
                scores, factor, parameters = (x.clone() for x in geometry(logits, native_actions))
                reference = beta_kl_reference(parameters)
                trained_transitions += int(weights.sum())
                diagnostics['data/complete_suffix_fraction'] = weights.mean()
                diagnostics['data/trained_transitions'] = trained_transitions
                diagnostics['targets/profile_rms'] = scale.sqrt()
                diagnostics['policy/concentration'] = (parameters.sum(-1).mean(-1) * actor_weights).sum() / actor_weights.sum()
                diagnostics['policy/entropy'] = (Beta(parameters[..., 0], parameters[..., 1], validate_args=False).entropy().sum(-1) * actor_weights).sum() / actor_weights.sum()

            with timer.span('update'):
                baseline_metrics = []
                for _ in range(args.baseline_epochs):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    baseline_optimizer.zero_grad(set_to_none=True)
                    loss = baseline_loss(observations, age, target_lengths, target, weights, scale)
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(baseline_parameters, args.max_grad_norm, foreach=True)
                    baseline_optimizer.step()
                    baseline_steps += 1
                    baseline_metrics.append(torch.stack((loss.detach(), norm.detach())))
                    del loss
                baseline_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    baseline = baseline_prediction(observations, age, target_lengths).clone()
                response_metrics = []
                for _ in range(args.response_epochs):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    response_optimizer.zero_grad(set_to_none=True)
                    loss = response_loss(observations, age, scores, native_actions, parameters, target_lengths,
                                         baseline, target, weights, scale)
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(response_parameters, args.max_grad_norm, foreach=True)
                    response_optimizer.step()
                    response_steps += 1
                    response_metrics.append(torch.stack((loss.detach(), norm.detach())))
                    del loss
                response_optimizer.zero_grad(set_to_none=True)
                diagnostics['losses/baseline_profile'], diagnostics['grad/baseline_preclip_norm'] = torch.stack(baseline_metrics).mean(0).unbind()
                diagnostics['losses/response_profile'], diagnostics['grad/response_preclip_norm'] = torch.stack(response_metrics).mean(0).unbind()
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    coefficients = actor_coefficients(observations, age, actor_lengths).clone()
                    powers = critic.powers().detach().clone()
                    kernel_mean, kernel_std = (x.clone() for x in statistics_function(parameters, powers, critic.kernel_indices))
                    credit = credit_function(parameters, factor, coefficients, powers, critic.kernel_indices, kernel_std).clone()
                    diagnostics['credit/alpha_beta_rms'] = credit.square().mean().sqrt()
                    diagnostics['critic/response_coefficient_rms'] = coefficients.square().mean().sqrt()
                    diagnostics['critic/active_power_mean'] = powers.sum() / critic.power_mask.sum()
                    diagnostics['critic/kernel_std_min'] = kernel_std.min()

                def measure():
                    gain, kl = measure_function(observations, reference, factor, coefficients, powers,
                                                kernel_mean, kernel_std, actor_weights)
                    return gain.clone(), kl.clone()

                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    actor_metrics = trust.step(observations, logits, factor, credit, actor_weights, measure)
                    diagnostics.update(actor_metrics)
                    actor_updates += int(actor_metrics['policy/accepted_scale'] > 0)
            diagnostics.update({'updates/actor_attempted': iteration, 'updates/actor_accepted': actor_updates,
                                'updates/baseline_optimizer': baseline_steps, 'updates/response_optimizer': response_steps,
                                'charts/episodic_return_mean_100': float(np.mean(recent_returns))})
            logged = gather_metrics({key: torch.as_tensor(value, device=device) for key, value in diagnostics.items()})
            if not all(np.isfinite(value) for value in logged.values()):
                raise FloatingPointError('nonfinite vector-response training diagnostics')
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

        report = dict(args=vars(args), fresh_initialization=True, checkpoint_loaded=False,
                      gae_used=False, replay_used=False, scalar_value_network_used=False,
                      learned_dynamics_used=False, bellman_backups=0, profile_edges=PROFILE_EDGES,
                      transitions=global_step, trained_transitions=trained_transitions,
                      actor_updates_attempted=iterations, actor_updates_accepted=actor_updates,
                      baseline_optimizer_steps=baseline_steps, response_optimizer_steps=response_steps,
                      final_return_mean_100=float(np.mean(recent_returns)), final_returns=list(recent_returns),
                      control_weight=control_weight, progress=progress,
                      objective='Uniform fresh-state expected signed future reward-profile improvement under an analytic action-response model',
                      limitations='Single seed, training returns. Right-censored targets excluded. Shared finite neural fitting does not guarantee conditional score orthogonality. Candidate action-response expectations are exact only for the learned model; future-policy and occupancy shift remain. Golden-section natural-gradient ray search is not global constrained optimization.')
        atomic_json(run_dir / 'result.json', report)
        if args.save_model:
            torch.save(dict(actor=agent.actor.state_dict(), critic=critic.state_dict(),
                            normalization_mean=obs_norm.mean, normalization_inverse_std=obs_norm.inverse_std,
                            args=vars(args)), run_dir / f'{args.exp_name}.cleanrl_model')
        print('TRAINING_RESULT=' + json.dumps({key: value for key, value in report.items()
                                              if key not in ('args', 'progress', 'final_returns')}), flush=True)


if __name__ == '__main__':
    main()

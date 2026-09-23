# Representable natural return-field policy improvement v9: fresh end-to-end return optimization.
# Critic outputs temporal x actuator x (alpha,beta) reward-score cross moments.
# Tensor PopArt preserves raw credit while conditioning the vector regression.
# Actor solves the representable natural gradient, then enforces one exact joint KL.
# No GAE, scalar value head, hindsight gate, generated outcome, or reward decoder.
# Complete observed episodes and leave-one-environment-out baselines provide targets.
import json
import math
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from typing import Literal
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import gather_metrics, device_minibatches
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
TEMPORAL_LOWER = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

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


def unwhiten(latent, factor):
    first = factor[..., 0] * latent[..., 0]
    second = factor[..., 1] * latent[..., 0] + factor[..., 2] * latent[..., 1]
    return torch.stack((first, second), -1)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, x):
        return x + self.second(F.silu(self.first(self.norm(x)))) / math.sqrt(2)


def complete_episode_layout(ages, terminations, truncations, episode_length):
    """Select whole fixed-length episodes, never action-dependent partial windows.

    Unknown warmup ages are -1 until the next reset. The first implementation
    deliberately rejects early termination: its selection proof is HalfCheetah-
    specific and must not silently extend to variable-length episodes.
    """
    if terminations.any():
        raise ValueError('this fixed-length gate cannot censor early terminations')
    steps, environments = ages.shape
    if np.any(truncations & (ages >= 0) & (ages != episode_length - 1)):
        raise ValueError('unexpected episode length in the fixed-horizon gate')
    ids = np.full_like(ages, -1, dtype=np.int64)
    episode_count = 0
    for env in range(environments):
        for end in np.flatnonzero(truncations[:, env]):
            start = end - episode_length + 1
            if start >= 0 and ages[start, env] == 0:
                if not np.array_equal(ages[start:end + 1, env], np.arange(episode_length)):
                    raise ValueError('episode age/order mismatch')
                ids[start:end + 1, env] = episode_count
                episode_count += 1
    return ids, episode_count


def discounted_episode_returns(rewards, episode_ids, gamma):
    """Exact observed finite-episode returns; invalid rows contribute zero."""
    # Parallel doubling scan: O(log T) batched operations, no timestep kernel loop.
    result = rewards * (episode_ids >= 0)
    offset = 1
    while offset < rewards.shape[0]:
        same = (episode_ids[:-offset] >= 0) & (episode_ids[:-offset] == episode_ids[offset:])
        result = torch.cat((result[:-offset] + gamma ** offset * result[offset:] * same, result[-offset:]), 0)
        offset *= 2
    return result


@dataclass
class Args:
    exp_name: str = 'return_field_natural_v9'
    seed: int = 1
    env_id: str = 'HalfCheetah-v4'
    total_timesteps: int = 8000000
    num_envs: int = 16
    num_steps: int = 2048
    gamma: float = 1.0
    credit_source: Literal['learned', 'sampled'] = 'learned'
    critic_width: int = 256
    critic_epochs: int = 10
    minibatch_size: int = 4096
    critic_learning_rate: float = 1e-3
    max_grad_norm: float = .5
    trust_kl: float = .01
    cg_iterations: int = 10
    cg_damping: float = .01
    cg_relative_tolerance: float = 1e-6
    line_search_steps: int = 12
    popart_rate: float = .05
    env_backend: str = 'auto'
    env_threads: int = 2
    compile: bool = True
    compile_mode: str = 'reduce-overhead'
    capture_video: bool = False
    torch_deterministic: bool = True
    non_blocking_transfers: bool = False
    staggered_starts: bool = True
    save_model: bool = True


def temporal_returns(episode_rewards, gamma):
    """Observed [episode,time,band] discounted sums; all bands telescope to G.

    FP64 avoids subtractive cancellation when a one-step band is formed from
    large undiscounted returns. The final tensor returns to the input dtype.
    """
    episodes, length = episode_rewards.shape
    ids = torch.arange(episodes, device=episode_rewards.device).expand(length, -1)
    returns = discounted_episode_returns(episode_rewards.double().T, ids, gamma).T
    starts = torch.arange(length, device=episode_rewards.device)
    values = []
    upper = TEMPORAL_LOWER[1:] + (length,)
    for lo, hi in zip(TEMPORAL_LOWER, upper):
        left, right = starts + lo, starts + hi
        first = returns[:, left.clamp_max(length - 1)] * (left < length) * gamma ** lo
        second = returns[:, right.clamp_max(length - 1)] * (right < length) * gamma ** hi
        values.append(first - second)
    return torch.stack(values, -1).to(episode_rewards.dtype)


def leave_environment_out_rewards(episode_rewards, episode_environment, num_envs):
    """Baseline uses other environments, excluding ALL source-env episodes.

    Shared normalization is per-environment, so previous source episodes may
    affect source statistics but cannot enter this baseline through those stats.
    Raw rewards are centered before any learned/output normalization.
    """
    sums = torch.zeros(num_envs, episode_rewards.shape[1], device=episode_rewards.device, dtype=episode_rewards.dtype)
    sums.index_add_(0, episode_environment, episode_rewards)
    counts = torch.zeros(num_envs, device=episode_rewards.device, dtype=episode_rewards.dtype)
    counts.index_add_(0, episode_environment, torch.ones_like(episode_environment, dtype=episode_rewards.dtype))
    other_count = episode_rewards.shape[0] - counts[episode_environment]
    baseline = (episode_rewards.sum(0) - sums[episode_environment]) / other_count[:, None]
    return episode_rewards - baseline


class ReturnField(nn.Module):
    """Conditional reward-score cross moments, with a tensor PopArt readout.

    Head axes: temporal band, actuator, whitened alpha/beta statistic. No scalar
    Q/return prediction precedes the head. Population means are E[R_band*z|state,
    age, policy context]; their sum supplies the local return-gain vector.
    """
    def __init__(self, observation_dim, action_dim, num_envs, width=256):
        super().__init__()
        self.shape = (len(TEMPORAL_LOWER), action_dim, 2)
        self.environment = nn.Embedding(num_envs, 8)
        self.trunk = nn.Sequential(nn.Linear(observation_dim + 2 * action_dim + 2 + 8, width),
                                   nn.SiLU(), ResidualBlock(width), ResidualBlock(width))
        self.head = layer_init(nn.Linear(width, math.prod(self.shape)), std=.01)
        self.register_buffer('mean', torch.zeros(self.shape))
        self.register_buffer('second', torch.ones(self.shape))
        self.register_buffer('scale', torch.ones(self.shape))
        self.initialized = False

    def normalized(self, observation, parameters, age, environment):
        phase = age.float()[:, None] / 1000
        context = torch.cat((observation, parameters.log().flatten(1), phase, 1 - phase,
                             self.environment(environment)), -1)
        hidden = self.trunk(context)
        # Preserve small raw cross moments when subtracting PopArt offsets;
        # BF16 quantization of the final head would be amplified by its scale.
        with torch.autocast('cuda', enabled=False):
            return self.head(hidden.float()).reshape(-1, *self.shape)

    def support(self, age):
        lower = torch.tensor(TEMPORAL_LOWER, device=age.device)
        return (age[:, None] + lower < 1000)[:, :, None, None]

    def forward(self, observation, parameters, age, environment):
        raw = self.normalized(observation, parameters, age, environment).float() * self.scale + self.mean
        # Exact episode support, not a learned decay or arbitrary attenuation.
        return raw * self.support(age)

    @torch.no_grad()
    def update_statistics(self, target, weights, optimizer, rate):
        """Exactly preserve raw outputs; update head Adam history in loss units.

        At unchanged raw residual, normalized-MSE head gradients scale by a=
        old_scale/new_scale. Thus head first/second moments scale by a/a².
        This does not make shared-trunk history, global clipping, or Adam itself
        invariant under the changing normalization objective.
        """
        normalizer = weights.sum()
        weights = weights[:, None, None, None]
        mean = (target * weights).sum(0) / normalizer
        second = (target.square() * weights).sum(0) / normalizer
        rate = rate if self.initialized else 1.
        new_mean = self.mean.lerp(mean, rate)
        new_second = self.second.lerp(second, rate)
        new_scale = (new_second - new_mean.square()).clamp_min(1e-8).sqrt()
        ratio = self.scale / new_scale
        self.head.weight.mul_(ratio.flatten()[:, None])
        self.head.bias.mul_(ratio.flatten()).add_(((self.mean - new_mean) / new_scale).flatten())
        for parameter, multiplier in ((self.head.weight, ratio.flatten()[:, None]), (self.head.bias, ratio.flatten())):
            state = optimizer.state.get(parameter, {})
            if 'exp_avg' in state:
                state['exp_avg'].mul_(multiplier)
                state['exp_avg_sq'].mul_(multiplier.square())
        self.mean.copy_(new_mean)
        self.second.copy_(new_second)
        self.scale.copy_(new_scale)
        self.initialized = True


def beta_kl_reference(parameters):
    alpha, beta = parameters.double().unbind(-1)
    total = alpha + beta
    return torch.stack((alpha, beta, alpha.lgamma() + beta.lgamma() - total.lgamma(),
                        alpha.digamma() - total.digamma(), beta.digamma() - total.digamma()), -1)


def policy_gain_kl(logits, old_parameters, reference, natural_credit, weights):
    alpha, beta = (F.softplus(logits) + 1).double().chunk(2, -1)
    new_parameters = torch.stack((alpha, beta), -1)
    old_alpha, old_beta, old_partition, old_da, old_db = reference.unbind(-1)
    partition = alpha.lgamma() + beta.lgamma() - (alpha + beta).lgamma()
    joint_kl = (partition - old_partition + (old_alpha - alpha) * old_da + (old_beta - beta) * old_db).sum(-1)
    gain = ((new_parameters - old_parameters) * natural_credit).sum((-1, -2))
    denominator = weights.sum().clamp_min(1e-12)
    return (gain * weights).sum() / denominator, (joint_kl * weights).sum() / denominator


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
        """Explore one ray, retaining the best actually measured feasible gain.

        Expand until finding an infeasible candidate, then refine that local
        bracket. Feasibility is checked at every point; global monotonicity is
        neither asserted nor required. Search does not certify a global maximum.
        """
        delta = self.unflatten(direction)
        scale = float(initial_scale)
        if not math.isfinite(scale) or scale <= 0:
            gain, kl = measure()
            return gain, kl, 0., 0, 0
        best_scale, best_gain = 0., 0.
        lower, upper = 0., None
        evaluations = 0
        for _ in range(self.line_search_steps):
            for name, parameter in self.named:
                parameter.copy_(before[name]).add_(delta[name], alpha=scale)
            gain, kl = measure()
            evaluations += 1
            # Intentional host synchronization at completed proposal boundaries.
            feasible = bool(torch.isfinite(gain) & torch.isfinite(kl) &
                            (gain > 0) & (kl >= -1e-8) & (kl <= self.budget))
            if feasible:
                if float(gain) > best_gain:
                    best_gain, best_scale = float(gain), scale
                lower = scale
                scale = scale * 2 if upper is None else (lower + upper) * .5
            else:
                upper = scale
                scale = (lower + upper) * .5
        for name, parameter in self.named:
            parameter.copy_(before[name])
            if best_scale > 0:
                parameter.add_(delta[name], alpha=best_scale)
        gain, kl = measure()
        # A cap hit denotes expansion still unbracketed at the finite search cap.
        return gain, kl, best_scale, evaluations, int(upper is None)

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


def validate_args(args):
    if args.env_id != 'HalfCheetah-v4' or args.num_envs < 2:
        raise ValueError('This first version requires fixed-length HalfCheetah and independent baseline environments')
    if not 0 < args.gamma <= 1 or not 0 < args.popart_rate <= 1:
        raise ValueError('invalid discount or normalization rate')
    if args.num_steps < 2000 or (args.num_steps * args.num_envs) % args.minibatch_size:
        raise ValueError('need two episode lengths and fixed-size minibatches partitioning the rollout')
    if min(args.cg_iterations, args.line_search_steps, args.critic_epochs, args.critic_width, args.minibatch_size, args.env_threads,
           args.cg_damping, args.cg_relative_tolerance, args.critic_learning_rate, args.max_grad_norm, args.trust_kl, args.total_timesteps) <= 0:
        raise ValueError('positive dimensions, optimizer settings, and training budget required')
    return args


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required')
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision='highest', allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    batch_size = args.num_steps * args.num_envs
    iterations = math.ceil(args.total_timesteps / batch_size)
    run_name = f'{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}'
    run_dir = f'runs/{run_name}'
    with ExitStack() as resources:
        writer = SummaryWriter(run_dir)
        resources.callback(writer.close)
        writer.add_text('hyperparameters', '|param|value|\n|-|-|\n' + '\n'.join(f'|{k}|{v}|' for k, v in vars(args).items()))
        writer.add_text('method', 'On-policy return-field improvement; temporal x actuator x alpha/beta tensor; one joint exact KL; no GAE or scalar value head. Fresh training, no held-out gate.')
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        length = episode_horizon(args.env_id)
        if length != 1000:
            raise ValueError('expected 1000-step HalfCheetah episodes')
        obs_shape = envs.single_observation_space.shape
        obs_dim = int(np.prod(obs_shape))
        agent = Agent(envs).to(device)
        field = ReturnField(obs_dim, agent.action_dim, args.num_envs, args.critic_width).to(device)
        field_optimizer = optim.Adam(field.parameters(), lr=args.critic_learning_rate, eps=1e-5, fused=True)
        trust = ActorNaturalGradient(agent.actor, args.trust_kl, args.cg_damping, args.cg_iterations,
                                     args.cg_relative_tolerance, args.line_search_steps, args.compile, args.compile_mode)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        # Required shared normalizer for rollout logging only. Raw rewards define
        # all learning targets; per-env normalization never reweights the task.
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)

        def field_loss(obs, parameters, age, environment, target, weights):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = field.normalized(obs, parameters, age, environment)
            normalized_target = (target - field.mean) / field.scale
            residual = ((prediction.float() - normalized_target).square() * field.support(age)).mean((1, 2, 3))
            return .5 * (residual * weights).sum() / weights.sum().clamp_min(1e-12)

        def field_predict(obs, parameters, age, environment):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return field(obs, parameters, age, environment)

        def actor_measure(obs, parameters, reference, natural_credit, weights):
            return policy_gain_kl(agent.actor(obs), parameters, reference, natural_credit, weights)

        policy_model = agent.actor.forward
        geometry_fn = beta_geometry
        bands_fn = temporal_returns
        if args.compile:
            policy_model = graph_compile(policy_model)
            geometry_fn = torch.compile(geometry_fn, fullgraph=True, mode=args.compile_mode)
            bands_fn = torch.compile(bands_fn, fullgraph=True, dynamic=True, options={'triton.cudagraphs': False})
            field_loss = torch.compile(field_loss, fullgraph=True, mode=args.compile_mode)
            field_predict = graph_compile(field_predict)
            actor_measure = graph_compile(actor_measure)
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        sampler = make_beta_sampler(args.num_envs, agent.action_dim, agent.action_low.detach().cpu().numpy(), agent.action_high.detach().cpu().numpy())
        sampler_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 7]))

        def act(observations):
            native, physical = sampler(host_actor(observations), sampler_rng)
            if not np.isfinite(physical).all():
                raise FloatingPointError('nonfinite actor action')
            return native, physical.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                  non_blocking=args.non_blocking_transfers,
                                  fields={'observations': obs_shape, 'native_actions': (agent.action_dim,), 'raw_rewards': ()})
        resources.callback(transfer.close)
        ages = np.empty((args.num_steps, args.num_envs), dtype=np.int64)
        terms_buffer = np.empty_like(ages, dtype=bool)
        truncs_buffer = np.empty_like(ages, dtype=bool)
        current_age = np.full(args.num_envs, -1, dtype=np.int64)
        environment_index = torch.arange(batch_size, device=device) % args.num_envs
        critic_shuffle = torch.Generator(device=device).manual_seed(args.seed + 7102)
        timer = PhaseTimer()
        started = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)
        if args.staggered_starts:
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=lambda obs: act(obs)[1], horizon=length,
                                    phase_offsets=compute_phase_offsets(args.num_envs, length, args.seed), seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
            current_age.fill(0)
        recent_returns = deque(maxlen=100)
        progress = []
        trained_transitions = 0
        interval_start, interval_step = time.perf_counter(), global_step
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
                    normalized_reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, _ = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(step, normalized_reward, terms, truncs, observations=obs_step,
                                  native_actions=native, raw_rewards=raw_reward)
                    terms_buffer[step], truncs_buffer[step] = terms, truncs
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
                actions = batch.fields['native_actions'].flatten(0, 1)
                raw_rewards = batch.fields['raw_rewards'].flatten()
                age = torch.as_tensor(ages.copy().flatten(), device=device).clamp_min(0)
                layout, episode_count = complete_episode_layout(ages, terms_buffer, truncs_buffer, length)
                if episode_count < args.num_envs:
                    raise ValueError('need a complete episode from every baseline environment')
                episode_environment = np.empty(episode_count, dtype=np.int64)
                for env in range(args.num_envs):
                    present = np.unique(layout[:, env])
                    episode_environment[present[present >= 0]] = env
                episode_environment = torch.as_tensor(episode_environment, device=device)
                ids = torch.as_tensor(layout.flatten(), device=device)
                valid = ids >= 0
                weights = args.gamma ** age.float() * valid
                episode_rewards = torch.zeros(episode_count, length, device=device)
                episode_rewards[ids[valid], age[valid]] = raw_rewards[valid]
                centered = leave_environment_out_rewards(episode_rewards, episode_environment, args.num_envs)
                band_returns = bands_fn(centered, args.gamma).clone()
                sampled_returns = band_returns[ids.clamp_min(0), age] * valid[:, None]
                logits = policy_model(observations).clone()
                z, factor, parameters = (x.clone() for x in geometry_fn(logits, actions))
                targets = sampled_returns[:, :, None, None] * z[:, None, :, :]
                reference = beta_kl_reference(parameters)
                trained_transitions += episode_count * length
                diagnostics['data/complete_episodes'] = float(episode_count)
                diagnostics['data/fraction_used'] = valid.float().mean()
                diagnostics['data/trained_transitions'] = float(trained_transitions)
                diagnostics['policy/concentration'] = (parameters.sum(-1).mean(-1) * weights).sum() / weights.sum()
                diagnostics['policy/entropy'] = (Beta(parameters[..., 0], parameters[..., 1], validate_args=False).entropy().sum(-1) * weights).sum() / weights.sum()
                # Spectrum of the coupled alpha/beta Fisher block, not just its diagonal.
                aa = factor[..., 0].square()
                bb = factor[..., 1].square() + factor[..., 2].square()
                ab = factor[..., 0] * factor[..., 1]
                gap = ((aa - bb).square() + 4 * ab.square()).sqrt()
                diagnostics['policy/fisher_min_eigenvalue'] = ((aa + bb - gap) / 2)[valid].min()
                diagnostics['policy/fisher_max_eigenvalue'] = ((aa + bb + gap) / 2)[valid].max()

            with timer.span('update'):
                if args.credit_source == 'learned':
                    field.update_statistics(targets, weights, field_optimizer, args.popart_rate)
                    metrics = []
                    for _ in range(args.critic_epochs):
                        for indices in device_minibatches(batch_size, args.minibatch_size, device, critic_shuffle):
                            if args.compile:
                                torch.compiler.cudagraph_mark_step_begin()
                            loss = field_loss(observations[indices], parameters[indices], age[indices], environment_index[indices],
                                              targets[indices], weights[indices])
                            field_optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            norm = nn.utils.clip_grad_norm_(field.parameters(), args.max_grad_norm, foreach=True)
                            field_optimizer.step()
                            metrics.append(torch.stack((loss.detach(), norm.detach())))
                            del loss
                    diagnostics['losses/field_normalized_mse_half'], diagnostics['grad/field_preclip_norm'] = torch.stack(metrics).mean(0).unbind()
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    if args.credit_source == 'learned':
                        temporal_credit = field_predict(observations, parameters, age, environment_index).clone()
                    else:
                        temporal_credit = targets
                    credit = temporal_credit.sum(1)
                    natural_credit = unwhiten(credit, factor).detach()
                    squared_norm = (credit.square().sum((-1, -2)) * weights).sum() / weights.sum()
                    diagnostics['credit/joint_field_norm'] = squared_norm.sqrt()
                    for band, lower in enumerate(TEMPORAL_LOWER):
                        diagnostics[f'credit/band_{lower}_norm'] = (temporal_credit[:, band].square().sum((-1, -2)) * weights).sum().div(weights.sum()).sqrt()
                def measure():
                    gain, kl = actor_measure(observations, parameters, reference, natural_credit, weights)
                    return gain.clone(), kl.clone()

                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    diagnostics.update(trust.step(observations, logits, factor, natural_credit, weights, measure))
                    updated_logits = policy_model(observations).clone()
                    alpha, beta = (F.softplus(updated_logits) + 1).chunk(2, -1)
                    new_parameters = torch.stack((alpha, beta), -1)
                    old_mean = parameters[..., 0] / parameters.sum(-1)
                    new_mean = alpha / (alpha + beta)
                    diagnostics['policy/mean_action_shift_rms'] = (((new_mean - old_mean).square().sum(-1) * weights).sum() / weights.sum()).sqrt()
                    diagnostics['policy/concentration_shift_rms'] = (((new_parameters.sum(-1) - parameters.sum(-1)).square().sum(-1) * weights).sum() / weights.sum()).sqrt()
            diagnostics['charts/episodic_return_mean_100'] = float(np.mean(recent_returns))
            logged = gather_metrics({name: torch.as_tensor(value, device=device) for name, value in diagnostics.items()})
            if not all(np.isfinite(value) for value in logged.values()):
                raise FloatingPointError('nonfinite training metrics')
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar('charts/SPS', global_step / (now - started), global_step)
            writer.add_scalar('charts/interval_SPS', (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f'timing/{phase}_s', timing['total_s'], global_step)
            timer.reset()
            writer.flush()
            progress.append(dict(step=global_step, iteration=iteration, **logged))
            with open(f'{run_dir}/progress.json', 'w') as output:
                json.dump(progress, output, allow_nan=False)
            print(f'iteration={iteration}/{iterations} step={global_step} metrics={json.dumps(logged)}', flush=True)
            interval_start, interval_step = now, global_step
        report = dict(args=vars(args), fresh_initialization=True, checkpoint_loaded=False, gae_used=False,
                      value_network_used=False, heldout_gate_used=False, transitions=global_step,
                      trained_transitions=trained_transitions, final_return_mean_100=float(np.mean(recent_returns)),
                      final_returns=list(recent_returns), progress=progress,
                      objective='Undiscounted 1000-step episodic reward' if args.gamma == 1 else 'Finite discounted episodic reward with outer age weighting',
                      limitations='One seed; training-episode returns, not a held-out score; fitted field and finite actor projection are approximate. Only complete episodes contribute targets. Observation normalization is adaptive per environment. Finite damped natural-gradient solve and bounded exact-KL line search; no actor optimizer state.')
        with open(f'{run_dir}/result.json', 'w') as output:
            json.dump(report, output, indent=2, allow_nan=False)
        if args.save_model:
            torch.save({'actor': agent.actor.state_dict(), 'return_field': field.state_dict(),
                        'obs_norm': {name: getattr(obs_norm, name) for name in ('means', 'variances', 'counts', 'epsilon', 'clip')},
                        'args': vars(args)}, f'{run_dir}/{args.exp_name}.cleanrl_model')
        print('TRAINING_RESULT=' + json.dumps({k: v for k, v in report.items() if k not in ('args', 'progress', 'final_returns')}), flush=True)


if __name__ == '__main__':
    main()

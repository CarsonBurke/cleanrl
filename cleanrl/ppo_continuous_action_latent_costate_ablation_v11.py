# Latent costate ablation v11: isolate discount, full-batch fitting, and target refreshes.
# Predict latent reward sensitivities, pull them back to fixed state coordinates,
# and propagate through learned one-step dynamics at real observed transitions.
# Beta implicit derivatives carry action credit into alpha/beta. No scalar value,
# GAE, contrastive branches, imagined trajectory, or learned latent reward decoder.
# Three independent fresh-run ablations of v10, with explicit optimizer/update counts.
# Discounted costates use uniform rollout-state actor weighting, not gamma**age.
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


@dataclass
class Args:
    exp_name: str = 'latent_costate_ablation_v11'
    seed: int = 1
    env_id: str = 'HalfCheetah-v4'
    total_timesteps: int = 8000000
    num_envs: int = 16
    num_steps: int = 2048
    model_width: int = 256
    critic_width: int = 192
    latent_features: int = 32
    model_epochs: int = 10
    critic_epochs: int = 8
    critic_target_refreshes: int = 8
    gamma: float = 1.
    full_batch: bool = False
    minibatch_size: int = 4096
    model_learning_rate: float = .001
    critic_learning_rate: float = .0003
    projection_weight: float = 0.
    max_grad_norm: float = .5
    trust_kl: float = .03
    cg_iterations: int = 50
    cg_damping: float = .01
    cg_relative_tolerance: float = 1e-6
    line_search_steps: int = 12
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


class Dynamics(nn.Module):
    """Predict fixed-coordinate state deltas and actual forward progress speed."""
    def __init__(self, state_dim, action_dim, width=256):
        super().__init__()
        self.state_dim = state_dim
        self.trunk = nn.Sequential(nn.Linear(state_dim + action_dim, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.head = nn.Linear(width, state_dim + 1)
        nn.init.orthogonal_(self.head.weight, gain=.01)
        nn.init.zeros_(self.head.bias)

    def delta_progress(self, state, action):
        hidden = self.trunk(torch.cat((state, action), -1))
        with torch.autocast('cuda', enabled=False):
            return self.head(hidden.to(self.head.weight.dtype))

    def forward(self, state, action):
        output = self.delta_progress(state, action)
        return torch.cat((state + output[:, :self.state_dim], output[:, self.state_dim:]), -1)


class LatentCostate(nn.Module):
    """Latent covector with physical pullback J_h^T c; there is no scalar head.

    Identity features anchor the coordinates. Learned features receive gradients
    through the pullback Jacobian as well as the covector readout. The actor has
    separate parameters, so critic learning cannot change behavior outside KL.
    """
    def __init__(self, state_dim, width=192, latent_features=32, horizon=1000):
        super().__init__()
        self.state_dim, self.horizon = state_dim, horizon
        self.encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.SiLU(),
                                     nn.Linear(64, latent_features), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(state_dim + latent_features + 2, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.head = nn.Linear(width, state_dim + latent_features)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def features(self, state):
        # Stable input Jacobian; the small encoder runs in FP32 rather than BF16.
        with torch.autocast('cuda', enabled=False):
            return torch.cat((state, self.encoder(state)), -1)

    def forward(self, state, age):
        features, pullback = torch.func.vjp(self.features, state)
        phase = age.to(state.dtype)[:, None] / self.horizon
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=state.dtype == torch.float32):
            hidden = self.trunk(torch.cat((features, phase, 1 - phase), -1))
        with torch.autocast('cuda', enabled=False):
            covector = self.head(hidden.to(self.head.weight.dtype))
            result, = pullback(covector)
        return result * (age < self.horizon)[:, None]


def observed_beta_jacobian(native_action, parameters, action_range):
    """Implicit Dirichlet transport derivative at the actual observed Beta draw.

    This is the same transport used by Beta.rsample's backward, evaluated from
    its sufficient inputs. No gradient is attached to a host RNG operation.
    """
    # Complement reconstruction and the transport kernel are sensitive in FP32.
    # Evaluate both coordinates from one FP64 scalar, then return actor precision.
    action64, parameters64 = native_action.double(), parameters.double()
    x = torch.stack((action64, 1 - action64), -1)
    total = parameters64.sum(-1, keepdim=True).expand_as(parameters64)
    transport = torch._dirichlet_grad(x, parameters64, total)
    derivatives = torch.stack(((1 - action64) * transport[..., 0],
                               -action64 * transport[..., 1]), -1)
    return (derivatives * action_range[None, :, None]).to(parameters.dtype)


def differential_targets(model, actor, observations, physical_actions, beta_jacobian,
                         next_costate, continuation, control_weight, gamma=1.):
    """Discount continuation once; immediate reward and progress stay undiscounted."""
    cotangent = torch.cat((next_costate.detach() * continuation[:, None] * gamma,
                           torch.ones_like(continuation[:, None])), -1)
    _, model_backward = torch.func.vjp(model, observations, physical_actions)
    state_credit, action_credit = model_backward(cotangent)
    action_credit = action_credit - 2 * control_weight * physical_actions
    eta_credit = action_credit[:, :, None] * beta_jacobian.detach()
    logits, actor_backward = torch.func.vjp(actor, observations)
    logit_credit = torch.cat((eta_credit[..., 0], eta_credit[..., 1]), -1) * logits.sigmoid()
    policy_state_credit, = actor_backward(logit_credit)
    return ((state_credit + policy_state_credit).detach(), action_credit.detach(), eta_credit.detach())


def action_jacobian(model, observations, physical_actions):
    """Forward-mode action Jacobian; do not materialize the larger state Jacobian."""
    values = []
    for coordinate in range(physical_actions.shape[1]):
        direction = torch.zeros_like(physical_actions)
        direction[:, coordinate] = 1
        _, tangent = torch.func.jvp(lambda actions: model(observations, actions),
                                    (physical_actions,), (direction,))
        values.append(tangent[:, :observations.shape[1]])
    return torch.stack(values, -1).detach()


def projected_residual(residual, incoming_jacobian, incoming_beta_jacobian):
    action_error = torch.einsum('bsa,bs->ba', incoming_jacobian, residual)
    return action_error[:, :, None] * incoming_beta_jacobian


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


def incoming_projection(jacobian, beta_jacobian, continuation, age, num_envs):
    """Align each state's residual with its real preceding transition, time-major."""
    mask = torch.roll(continuation, num_envs) * (age > 0)
    mask[:num_envs] = 0
    return (torch.roll(jacobian, num_envs, 0) * mask[:, None, None],
            torch.roll(beta_jacobian, num_envs, 0))


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def fitting_batches(batch_size, minibatch_size, device, generator, full_batch):
    """A full-batch step uses the entire rollout without shuffling or partitioning."""
    if full_batch:
        yield slice(None)
    else:
        yield from device_minibatches(batch_size, minibatch_size, device, generator)


def critic_fitting_schedule(epochs, target_refreshes):
    """Refresh at epoch boundaries, holding each detached target for equal passes."""
    if epochs <= 0 or target_refreshes <= 0 or epochs % target_refreshes:
        raise ValueError('positive target refresh count must divide critic fitting epochs')
    return tuple(epoch % (epochs // target_refreshes) == 0 for epoch in range(epochs))


def main():
    args = tyro.cli(Args)
    if args.env_id != 'HalfCheetah-v4' or args.num_envs < 2:
        raise ValueError('fixed-horizon HalfCheetah is the supported first experiment')
    batch_size = args.num_envs * args.num_steps
    if args.num_steps < 1000 or args.minibatch_size <= 0 or (not args.full_batch and batch_size % args.minibatch_size):
        raise ValueError('positive minibatches must divide a rollout of at least one horizon')
    critic_schedule = critic_fitting_schedule(args.critic_epochs, args.critic_target_refreshes)
    if not 0 <= args.gamma <= 1:
        raise ValueError('gamma must lie in [0, 1]')
    if min(args.total_timesteps, args.model_epochs, args.critic_epochs, args.model_learning_rate,
           args.critic_learning_rate, args.cg_iterations, args.cg_damping, args.trust_kl,
           args.model_width, args.critic_width, args.latent_features, args.max_grad_norm) <= 0:
        raise ValueError('positive training dimensions and optimizer settings required')
    if args.projection_weight < 0:
        raise ValueError('projection weight must be nonnegative')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA and BF16 support required')
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision='highest', allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    run_name = f'{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}'
    run_dir = Path('runs') / run_name
    with ExitStack() as resources:
        writer = SummaryWriter(str(run_dir))
        resources.callback(writer.close)
        writer.add_text('hyperparameters', '|param|value|\n|-|-|\n' + '\n'.join(f'|{k}|{v}|' for k, v in vars(args).items()))
        writer.add_text('method', 'Latent vector costate; differential Bellman; observed next states; fresh training; no GAE/scalar value/contrastive data.')
        probe = gym.make(args.env_id)
        try:
            control_weight = float(probe.unwrapped._ctrl_cost_weight)
            environment_dt = float(probe.unwrapped.dt)
        finally:
            probe.close()
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        horizon = episode_horizon(args.env_id)
        if horizon != 1000:
            raise ValueError('expected fixed 1000-step episodes')
        obs_shape = envs.single_observation_space.shape
        state_dim = int(np.prod(obs_shape))
        agent = Agent(envs).to(device)
        model = Dynamics(state_dim, agent.action_dim, args.model_width).to(device)
        critic = LatentCostate(state_dim, args.critic_width, args.latent_features, horizon).to(device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_learning_rate, eps=1e-5, fused=True)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate, eps=1e-5, fused=True)
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
        if obs_calibration.last_raw is None:
            raise RuntimeError('warmup did not expose its final physical observations')
        obs_norm = FixedAffineObsNorm(obs_calibration.base, args.num_envs, obs_shape)
        next_obs_np = obs_norm.normalize(obs_calibration.last_raw)
        global_step = warm.transitions
        suppress = warm.suppress_mask.copy()
        current_age = np.full(args.num_envs, -1, dtype=np.int64)
        ages = np.empty((args.num_steps, args.num_envs), dtype=np.int64)
        ends = np.empty_like(ages, dtype=bool)
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                  non_blocking=args.non_blocking_transfers,
                                  fields={'observations': obs_shape, 'next_states': obs_shape,
                                          'native_actions': (agent.action_dim,), 'raw_rewards': ()})
        resources.callback(transfer.close)
        model_shuffle = torch.Generator(device=device).manual_seed(args.seed + 10101)
        critic_shuffle = torch.Generator(device=device).manual_seed(args.seed + 10102)

        def model_loss(obs, action, target, scale):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model.delta_progress(obs, action)
            return .5 * ((prediction - target) / scale).square().mean()

        def critic_loss(obs, age, target, weights, incoming, beta_incoming, full_scale, projected_scale):
            error = critic(obs, age) - target
            full = error.square().mean(-1) / full_scale
            projected = projected_residual(error, incoming, beta_incoming).square().mean((-1, -2)) / projected_scale
            denominator = weights.sum().clamp_min(1.)
            return .5 * ((full + args.projection_weight * projected) * weights).sum() / denominator

        def target_function(obs, action, beta_jac, next_value, continuation):
            return differential_targets(model, agent.actor, obs, action, beta_jac, next_value, continuation, control_weight, args.gamma)

        def jacobian_function(obs, action):
            return action_jacobian(model, obs, action)

        def actor_measure(obs, parameters, reference, credit, weights):
            return policy_gain_kl(agent.actor(obs), parameters, reference, credit, weights)

        policy_model = agent.actor.forward
        critic_predict = critic.forward
        geometry = beta_geometry
        beta_jacobian = observed_beta_jacobian
        if args.compile:
            policy_model = graph_compile(policy_model)
            critic_predict = torch.compile(critic_predict, fullgraph=True, mode=args.compile_mode)
            model_loss = torch.compile(model_loss, fullgraph=True, mode=args.compile_mode)
            critic_loss = torch.compile(critic_loss, fullgraph=True, mode=args.compile_mode)
            target_function = torch.compile(target_function, fullgraph=True, mode=args.compile_mode)
            jacobian_function = torch.compile(jacobian_function, fullgraph=True, mode=args.compile_mode)
            geometry = torch.compile(geometry, fullgraph=True, mode=args.compile_mode)
            beta_jacobian = torch.compile(beta_jacobian, fullgraph=True, mode=args.compile_mode)
            actor_measure = graph_compile(actor_measure)

        timer = PhaseTimer()
        recent_returns = deque(maxlen=100)
        progress = []
        trained_transitions = 0
        actor_updates = 0
        model_optimizer_steps = 0
        critic_optimizer_steps = 0
        critic_target_refreshes = 0
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
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(step, normalized_reward, terms, truncs, observations=obs_step,
                                  next_states=transition_obs, native_actions=native, raw_rewards=raw_reward)
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
                next_observations = batch.fields['next_states'].flatten(0, 1)
                native_actions = batch.fields['native_actions'].flatten(0, 1)
                physical_actions = agent.action_low + (agent.action_high - agent.action_low) * native_actions
                raw_rewards = batch.fields['raw_rewards'].flatten()
                age = torch.as_tensor(ages.copy().flatten(), device=device)
                weights = (age >= 0).float()
                age = age.clamp_min(0)
                continuation = (~torch.as_tensor(ends.copy().flatten(), device=device)).float()
                progress_speed = raw_rewards + control_weight * physical_actions.square().sum(-1)
                dynamics_target = torch.cat((next_observations - observations, progress_speed[:, None]), -1)
                dynamics_scale = dynamics_target.square().mean(0).clamp_min(1e-4).sqrt()
                logits = policy_model(observations).clone()
                _, factor, parameters = (x.clone() for x in geometry(logits, native_actions))
                beta_jac = beta_jacobian(native_actions, parameters, agent.action_high - agent.action_low).clone()
                reference = beta_kl_reference(parameters)
                trained_transitions += int((ages >= 0).sum())
                diagnostics['data/critic_fraction_used'] = weights.mean()
                diagnostics['data/trained_transitions'] = trained_transitions
                diagnostics['policy/concentration'] = (parameters.sum(-1).mean(-1) * weights).sum() / weights.sum()
                diagnostics['policy/entropy'] = (Beta(parameters[..., 0], parameters[..., 1], validate_args=False).entropy().sum(-1) * weights).sum() / weights.sum()

            with timer.span('update'):
                model.requires_grad_(True)
                model_metrics = []
                for _ in range(args.model_epochs):
                    for indices in fitting_batches(batch_size, args.minibatch_size, device, model_shuffle, args.full_batch):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        model_optimizer.zero_grad(set_to_none=True)
                        loss = model_loss(observations[indices], physical_actions[indices], dynamics_target[indices], dynamics_scale)
                        loss.backward()
                        norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, foreach=True)
                        model_optimizer.step()
                        model_optimizer_steps += 1
                        model_metrics.append(torch.stack((loss.detach(), norm.detach())))
                        del loss
                diagnostics['losses/model_normalized_mse_half'], diagnostics['grad/model_preclip_norm'] = torch.stack(model_metrics).mean(0).unbind()
                model_optimizer.zero_grad(set_to_none=True)
                model.requires_grad_(False)

                with torch.no_grad():
                    jacobians = []
                    for start in range(0, batch_size, args.minibatch_size):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        jacobians.append(jacobian_function(observations[start:start + args.minibatch_size],
                                                          physical_actions[start:start + args.minibatch_size]).clone())
                    jacobian = torch.cat(jacobians)
                    incoming, beta_incoming = incoming_projection(jacobian, beta_jac, continuation, age, args.num_envs)
                    diagnostics['model/action_jacobian_rms'] = jacobian.square().mean().sqrt()
                critic_metrics = []
                for refresh_target in critic_schedule:
                    if refresh_target:
                        critic_target_refreshes += 1
                        with torch.no_grad():
                            if args.compile:
                                torch.compiler.cudagraph_mark_step_begin()
                            next_value = critic_predict(next_observations, age + 1).clone()
                            target, qa, eta_credit = (x.clone() for x in target_function(
                                observations, physical_actions, beta_jac, next_value, continuation))
                            full_scale = (target.square().mean(-1) * weights).sum().div(weights.sum()).clamp_min(1.)
                            projected_scale = (projected_residual(target, incoming, beta_incoming).square().mean((-1, -2)) * weights).sum().div(weights.sum()).clamp_min(1.)
                    for indices in fitting_batches(batch_size, args.minibatch_size, device, critic_shuffle, args.full_batch):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        critic_optimizer.zero_grad(set_to_none=True)
                        loss = critic_loss(observations[indices], age[indices], target[indices], weights[indices],
                                           incoming[indices], beta_incoming[indices], full_scale, projected_scale)
                        loss.backward()
                        norm = nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm, foreach=True)
                        critic_optimizer.step()
                        critic_optimizer_steps += 1
                        critic_metrics.append(torch.stack((loss.detach(), norm.detach())))
                        del loss
                diagnostics['losses/critic_vector_and_projection'], diagnostics['grad/critic_preclip_norm'] = torch.stack(critic_metrics).mean(0).unbind()
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    next_value = critic_predict(next_observations, age + 1).clone()
                    target, qa, eta_credit = (x.clone() for x in target_function(
                        observations, physical_actions, beta_jac, next_value, continuation))
                    diagnostics['credit/next_costate_rms'] = next_value.square().mean().sqrt()
                    diagnostics['credit/target_costate_rms'] = target.square().mean().sqrt()
                    diagnostics['credit/action_rms'] = qa.square().mean().sqrt()
                    diagnostics['credit/alpha_beta_rms'] = eta_credit.square().mean().sqrt()

                def measure():
                    gain, kl = actor_measure(observations, parameters, reference, eta_credit, weights)
                    return gain.clone(), kl.clone()

                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    actor_metrics = trust.step(observations, logits, factor, eta_credit, weights, measure)
                    diagnostics.update(actor_metrics)
                    actor_updates += int(actor_metrics['policy/accepted_scale'] > 0)
            diagnostics.update({'updates/actor_attempted': iteration,
                                'updates/actor_accepted': actor_updates,
                                'updates/model_optimizer': model_optimizer_steps,
                                'updates/critic_optimizer': critic_optimizer_steps,
                                'updates/critic_target_refreshes': critic_target_refreshes})
            diagnostics['charts/episodic_return_mean_100'] = float(np.mean(recent_returns))
            logged = gather_metrics({key: torch.as_tensor(value, device=device) for key, value in diagnostics.items()})
            if not all(np.isfinite(value) for value in logged.values()):
                raise FloatingPointError('nonfinite costate training diagnostics')
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
                      gae_used=False, scalar_value_network_used=False, contrastive_data_used=False,
                      imagined_next_state_bootstrap=False, transitions=global_step,
                      actor_updates_attempted=iterations, actor_updates_accepted=actor_updates,
                      model_optimizer_steps=model_optimizer_steps, critic_optimizer_steps=critic_optimizer_steps,
                      critic_target_refreshes=critic_target_refreshes,
                      trained_transitions=trained_transitions, final_return_mean_100=float(np.mean(recent_returns)),
                      final_returns=list(recent_returns), progress=progress,
                      control_weight=control_weight, environment_dt=environment_dt,
                      objective='Uniform rollout-state improvement using discounted finite-horizon costates; benchmark uses undiscounted episodic reward',
                      limitations='No episode-age discount in actor weighting: not an exact episode-start discounted gradient. Full-batch and minibatch arms match dataset passes, not optimizer steps. Single seed and training returns. Learned dynamics derivatives may be inaccurate. Differential Bellman recursion need not contract. Fixed affine normalization after stochastic warmup. State occupancy includes ongoing-episode policy changes. Natural actor solve is approximate.')
        atomic_json(run_dir / 'result.json', report)
        if args.save_model:
            torch.save(dict(actor=agent.actor.state_dict(), dynamics=model.state_dict(), critic=critic.state_dict(),
                            normalization_mean=obs_norm.mean, normalization_inverse_std=obs_norm.inverse_std,
                            args=vars(args)), run_dir / f'{args.exp_name}.cleanrl_model')
        print('TRAINING_RESULT=' + json.dumps({key: value for key, value in report.items()
                                              if key not in ('args', 'progress', 'final_returns')}), flush=True)


if __name__ == '__main__':
    main()

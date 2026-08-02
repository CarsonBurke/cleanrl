# Full Matrix-Normal-Wishart BPC with clamped actor/value outputs v2.
#
# Actor Beta logits and the scalar value are Gaussian output states in complete
# BPC hierarchies. Each transition creates a KL-bounded policy-logit pseudo-target
# and a one-step TD value target, clamps those outputs during latent inference,
# then conjugately updates every edge. There are no backpropagated network updates,
# rollout batches, GAE, or TD(lambda) traces. Discounted power-Bayes supplies the
# only synaptic memory; a recent-observation guard limits posterior function drift.
import copy
import os
import random
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter


SAMPLE_EPS = 1e-6


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    num_envs: int = 16
    gamma: float = 0.99
    hidden_size: int = 64
    num_hidden_layers: int = 6
    hidden_state_clip: float = 5.0

    # One-step actor/value target construction. Policy logits q parameterize
    # alpha,beta = 1 + softplus(q); the target follows normalized TD * dlogpi/dq.
    actor_target_step: float = 1.0
    actor_target_mean_kl: float = 1e-4
    actor_target_sample_kl: float = 5e-4
    target_bisection_steps: int = 12
    td_rms_decay: float = 0.999
    td_rms_min: float = 0.1
    td_norm_clip: float = 10.0
    critic_td_clip: float = 10.0

    # Hidden-state-only predictive-coding inference. A scalar curvature bound
    # preserves gradient coordinate ratios; rejected energy-increasing moves are
    # damped and eventually rolled back. The output state remains fixed.
    inference_steps: int = 5
    inference_lr: float = 1.0
    inference_damping: float = 0.5
    inference_backtracks: int = 3
    inference_tolerance: float = 0.0
    inference_curvature_floor: float = 1e-3
    inference_curvature_refresh_interval: int = 32
    """reuse scalar spectral curvature bounds between guarded posterior steps"""

    # eta <- eta0 + rho*(eta-eta0) + evidence_weight*stats.
    actor_posterior_discount: float = 0.97
    critic_posterior_discount: float = 0.99
    actor_evidence_weight: float = 0.25
    critic_evidence_weight: float = 1.0
    posterior_jitter: float = 1e-5
    prior_column_covariance: float = 0.1
    prior_expected_precision: float = 1.0
    prior_dof_offset: float = 2.0

    # Posterior guards cover the current transition and a recent observation
    # reservoir, preventing locally safe updates from silently breaking old states.
    anchor_capacity: int = 512
    anchor_sample_size: int = 128
    actor_posterior_max_kl: float = 5e-4
    actor_posterior_max_sample_kl: float = 2e-3
    critic_posterior_rms_limit: float = 0.02
    critic_posterior_max_abs: float = 0.1
    posterior_guard_trials: int = 3

    # Slowly moving target BPC critic for the one-step bootstrap.
    target_critic_tau: float = 0.005
    target_critic_interval: int = 32

    # Retained for queue/CLI compatibility. The natural-state Cholesky recovery
    # and guarded variable-fraction commits are intentionally eager in this arm.
    compile: bool = True
    compile_mode: Optional[str] = "reduce-overhead"
    torch_float32_matmul_precision: str = "high"
    log_interval: int = 100
    num_updates: int = 0


class NaturalParameters(NamedTuple):
    Lambda: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor
    xi: torch.Tensor


class MNWParameters(NamedTuple):
    M: torch.Tensor
    V: torch.Tensor
    Psi: torch.Tensor
    nu: torch.Tensor


class SufficientStatistics(NamedTuple):
    Sxx: torch.Tensor
    Syx: torch.Tensor
    Syy: torch.Tensor
    N: torch.Tensor


def symmetrize(matrix):
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def spd_inverse(matrix, jitter=1e-6):
    matrix = symmetrize(matrix)
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    current_jitter = 0.0
    for _ in range(7):
        chol, info = torch.linalg.cholesky_ex(matrix + current_jitter * identity)
        if not torch.any(info):
            return torch.cholesky_inverse(chol)
        current_jitter = jitter if current_jitter == 0.0 else 10.0 * current_jitter
    raise RuntimeError("Matrix-Normal-Wishart parameter left the SPD cone")


def mnw_to_natural(parameters: MNWParameters, jitter=1e-6):
    M, V, Psi, nu = parameters
    dy, dx = M.shape[-2:]
    Lambda = spd_inverse(V, jitter)
    Q = M @ Lambda
    R = spd_inverse(Psi, jitter) + M @ Lambda @ M.transpose(-1, -2)
    xi = torch.as_tensor(nu, dtype=M.dtype, device=M.device) - dy + dx - 1
    return NaturalParameters(symmetrize(Lambda), Q, symmetrize(R), xi)


def natural_to_mnw(natural: NaturalParameters, jitter=1e-6):
    Lambda, Q, R, xi = natural
    if not bool(torch.stack([torch.isfinite(value).all() for value in natural]).all()):
        raise ValueError("non-finite Matrix-Normal-Wishart natural parameter")
    dx, dy = Lambda.shape[-1], R.shape[-1]
    V = spd_inverse(Lambda, jitter)
    M = Q @ V
    Psi = spd_inverse(symmetrize(R - Q @ V @ Q.transpose(-1, -2)), jitter)
    nu = xi + dy - dx + 1
    recovered_finite = torch.stack([torch.isfinite(value).all() for value in (M, V, Psi, nu)]).all()
    if bool((torch.any(nu <= dy - 1) | ~recovered_finite).item()):
        raise ValueError("invalid Wishart degrees of freedom")
    return MNWParameters(M, symmetrize(V), symmetrize(Psi), nu)


def sufficient_statistics(x, y):
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be aligned rank-two batches")
    return SufficientStatistics(
        x.transpose(0, 1) @ x,
        y.transpose(0, 1) @ x,
        y.transpose(0, 1) @ y,
        x.new_tensor(float(x.shape[0])),
    )


def discounted_conjugate_update(prior, current, stats, rho, evidence_weight=1.0):
    if not 0.0 <= rho <= 1.0 or evidence_weight <= 0.0:
        raise ValueError("rho must be in [0,1] and evidence weight positive")
    return NaturalParameters(
        *(base + rho * (old - base) + evidence_weight * statistic for base, old, statistic in zip(prior, current, stats))
    )


def recover_natural_group(naturals, jitter):
    batched = NaturalParameters(*(torch.stack(values) for values in zip(*naturals)))
    posterior = natural_to_mnw(batched, jitter)
    return [MNWParameters(*(value[index] for value in posterior)) for index in range(len(naturals))]


class MatrixNormalWishartEdge(nn.Module):
    """Bias-augmented Gaussian edge with a full Matrix-Normal-Wishart posterior."""

    def __init__(self, input_dim, output_dim, args, first_edge=False, initial_std=np.sqrt(2)):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.first_edge, self.jitter = first_edge, args.posterior_jitter
        dx, dtype = input_dim + 1, torch.float64
        mean = nn.Linear(input_dim, output_dim)
        nn.init.orthogonal_(mean.weight, initial_std)
        nn.init.zeros_(mean.bias)
        initial_M = torch.cat((mean.weight.detach(), mean.bias.detach().unsqueeze(1)), dim=1).to(dtype)
        V = args.prior_column_covariance * torch.eye(dx, dtype=dtype)
        nu = torch.tensor(output_dim + args.prior_dof_offset, dtype=dtype)
        Psi = (args.prior_expected_precision / float(nu)) * torch.eye(output_dim, dtype=dtype)
        natural = mnw_to_natural(MNWParameters(initial_M, V, Psi, nu), args.posterior_jitter)
        for prefix in ("prior", "natural"):
            for name, value in zip(NaturalParameters._fields, natural):
                self.register_buffer(f"{prefix}_{name}", value.clone())
        self.register_buffer("cached_M", torch.empty((output_dim, dx), dtype=torch.float32))
        self.register_buffer("cached_V", torch.empty((dx, dx), dtype=torch.float32))
        self.register_buffer("cached_precision", torch.empty((output_dim, output_dim), dtype=torch.float32))
        self._refresh_cache()

    def _natural(self, prefix="natural"):
        return NaturalParameters(*(getattr(self, f"{prefix}_{name}") for name in NaturalParameters._fields))

    def posterior(self):
        return natural_to_mnw(self._natural(), self.jitter)

    @torch.no_grad()
    def _refresh_cache(self, posterior=None):
        posterior = self.posterior() if posterior is None else posterior
        self.cached_M.copy_(posterior.M.float())
        self.cached_V.copy_(posterior.V.float())
        self.cached_precision.copy_((posterior.nu * posterior.Psi).float())

    @torch.no_grad()
    def commit_natural_(self, natural, posterior=None):
        posterior = natural_to_mnw(natural, self.jitter) if posterior is None else posterior
        for name, value in zip(NaturalParameters._fields, natural):
            getattr(self, f"natural_{name}").copy_(value)
        self._refresh_cache(posterior)

    def features(self, state):
        activated = state if self.first_edge else torch.tanh(state)
        return torch.cat((activated, torch.ones_like(activated[:, :1])), dim=1)

    def feature_derivative(self, state):
        return torch.ones_like(state) if self.first_edge else 1.0 - torch.tanh(state).square()

    def predict(self, previous_state):
        return self.features(previous_state) @ self.cached_M.transpose(0, 1)

    def expected_energy(self, previous_state, state):
        x = self.features(previous_state)
        residual = state - x @ self.cached_M.transpose(0, 1)
        data = torch.einsum("bi,ij,bj->b", residual, self.cached_precision, residual)
        uncertainty = self.output_dim * torch.einsum("bi,ij,bj->b", x, self.cached_V, x)
        return 0.5 * (data + uncertainty)

    def incoming_gradient(self, previous_state, state):
        return (state - self.predict(previous_state)) @ self.cached_precision

    def previous_state_gradient(self, previous_state, state):
        x = self.features(previous_state)
        residual = state - x @ self.cached_M.transpose(0, 1)
        mean = -(residual @ self.cached_precision) @ self.cached_M[:, : self.input_dim]
        uncertainty = self.output_dim * (x @ self.cached_V[:, : self.input_dim])
        return (mean + uncertainty) * self.feature_derivative(previous_state)

    def input_curvature_bound(self):
        """Positive curvature proxy; residual tanh curvature is handled by backtracking."""
        mean = self.cached_M[:, : self.input_dim]
        mean_curvature = torch.linalg.matrix_norm(mean.transpose(0, 1) @ self.cached_precision @ mean, ord=2)
        uncertainty = self.output_dim * torch.linalg.matrix_norm(self.cached_V[: self.input_dim, : self.input_dim], ord=2)
        return mean_curvature + uncertainty

    def output_curvature_bound(self):
        return torch.linalg.matrix_norm(self.cached_precision, ord=2)

    def posterior_candidate(self, previous_state, state, rho, evidence_weight):
        stats = sufficient_statistics(self.features(previous_state).double(), state.double())
        return discounted_conjugate_update(self._natural("prior"), self._natural(), stats, rho, evidence_weight)


class BPCNetwork(nn.Module):
    """Gaussian hidden hierarchy plus an explicitly Bayesian Gaussian output edge."""

    def __init__(self, input_dim, output_dim, args, output_std):
        super().__init__()
        self.hidden_state_clip = args.hidden_state_clip
        edges = []
        for index in range(args.num_hidden_layers):
            edges.append(
                MatrixNormalWishartEdge(
                    input_dim if index == 0 else args.hidden_size,
                    args.hidden_size,
                    args,
                    first_edge=index == 0,
                )
            )
        edges.append(MatrixNormalWishartEdge(args.hidden_size, output_dim, args, initial_std=output_std))
        self.edges = nn.ModuleList(edges)
        self.register_buffer("cached_inference_curvatures", torch.ones(args.num_hidden_layers))
        self._curvature_initialized = False
        self._curvature_cache_age = args.inference_curvature_refresh_interval

    @property
    def output_edge(self):
        return self.edges[-1]

    @torch.no_grad()
    def forward_states(self, observation):
        states, previous = [], observation
        for edge in self.edges[:-1]:
            state = edge.predict(previous)
            if self.hidden_state_clip > 0:
                state.clamp_(-self.hidden_state_clip, self.hidden_state_clip)
            states.append(state)
            previous = state
        return states, self.output_edge.predict(states[-1])

    def local_energy(self, observation, states, output, state_index):
        previous = observation if state_index == 0 else states[state_index - 1]
        energy = self.edges[state_index].expected_energy(previous, states[state_index])
        following = output if state_index == len(states) - 1 else states[state_index + 1]
        energy = energy + self.edges[state_index + 1].expected_energy(states[state_index], following)
        return energy

    def total_energy(self, observation, states, output):
        activities = [observation, *states]
        targets = [*states, output]
        return sum(
            edge.expected_energy(previous, target)
            for edge, previous, target in zip(self.edges, activities, targets)
        )

    @torch.no_grad()
    def inference_curvatures(self, args):
        """Cache conservative scalar blocks; energy backtracking remains authoritative."""
        if self._curvature_initialized:
            self._curvature_cache_age += 1
            if self._curvature_cache_age < args.inference_curvature_refresh_interval:
                return self.cached_inference_curvatures
        curvatures = torch.stack(
            [
                self.edges[index].output_curvature_bound()
                + self.edges[index + 1].input_curvature_bound()
                for index in range(len(self.edges) - 1)
            ]
        ).clamp_min(args.inference_curvature_floor)
        self.cached_inference_curvatures.copy_(curvatures)
        self._curvature_initialized = True
        self._curvature_cache_age = 0
        return self.cached_inference_curvatures

    @torch.no_grad()
    def settle_clamped_output(self, observation, initial_states, fixed_output, args):
        """Top-down Gauss-Seidel settling; fixed_output is never modified."""
        states = [state.clone() for state in initial_states]
        # Posterior guards keep moment changes small. Spectral bounds are therefore
        # reused across updates, while local energy backtracking still rejects any
        # stale-bound overshoot exactly.
        curvatures = self.inference_curvatures(args)
        maximum_change = observation.new_zeros(())
        accepted, rejected = 0, 0
        steps_taken = 0
        for sweep in range(args.inference_steps):
            old_states = [state.clone() for state in states]
            old_energy = self.total_energy(observation, states, fixed_output).sum()
            scale, moved = 1.0, False
            for _ in range(args.inference_backtracks + 1):
                for state, old_state in zip(states, old_states):
                    state.copy_(old_state)
                for index in reversed(range(len(states))):
                    previous = observation if index == 0 else states[index - 1]
                    following = fixed_output if index == len(states) - 1 else states[index + 1]
                    gradient = self.edges[index].incoming_gradient(previous, states[index])
                    gradient.add_(self.edges[index + 1].previous_state_gradient(states[index], following))
                    change = scale * args.inference_lr * gradient / curvatures[index]
                    states[index].sub_(change)
                    if self.hidden_state_clip > 0:
                        states[index].clamp_(-self.hidden_state_clip, self.hidden_state_clip)
                new_energy = self.total_energy(observation, states, fixed_output).sum()
                if torch.isfinite(new_energy) and new_energy <= old_energy + 1e-6 * old_energy.abs().clamp_min(1.0):
                    moved = True
                    break
                scale *= args.inference_damping
            if moved:
                sweep_change = torch.stack(
                    [(state - old_state).abs().max() for state, old_state in zip(states, old_states)]
                ).max()
                accepted += len(states)
                if scale < 1.0:
                    # Backtracking is direct evidence that the cached proxy is no
                    # longer conservative enough; refresh before the next sample.
                    self._curvature_cache_age = args.inference_curvature_refresh_interval
            else:
                for state, old_state in zip(states, old_states):
                    state.copy_(old_state)
                sweep_change = observation.new_zeros(())
                rejected += len(states)
            maximum_change = sweep_change
            steps_taken = sweep + 1
            if args.inference_tolerance > 0 and maximum_change.item() < args.inference_tolerance:
                break
        return states, steps_taken, maximum_change, accepted, rejected

    def posterior_candidates(self, observation, states, output, rho, evidence_weight):
        activities = [observation, *states]
        targets = [*states, output]
        return [
            edge.posterior_candidate(previous, target, rho, evidence_weight)
            for edge, previous, target in zip(self.edges, activities, targets)
        ]

    @torch.no_grad()
    def commit_naturals_(self, naturals):
        if len(naturals) != len(self.edges):
            raise ValueError("posterior proposal does not match network")
        groups = {}
        for index, (edge, natural) in enumerate(zip(self.edges, naturals)):
            key = (natural.Lambda.shape, natural.Q.shape, natural.R.shape, edge.jitter)
            groups.setdefault(key, []).append(index)
        for indices in groups.values():
            posteriors = recover_natural_group([naturals[index] for index in indices], self.edges[indices[0]].jitter)
            for index, posterior in zip(indices, posteriors):
                self.edges[index].commit_natural_(naturals[index], posterior)

    def snapshot_natural(self):
        return [NaturalParameters(*(value.clone() for value in edge._natural())) for edge in self.edges]

    @torch.no_grad()
    def interpolate_natural_(self, start, end, fraction):
        interpolated = [
            NaturalParameters(*(old.lerp(new, fraction) for old, new in zip(initial, final)))
            for initial, final in zip(start, end)
        ]
        self.commit_naturals_(interpolated)

    @torch.no_grad()
    def soft_update_from_(self, source, fraction):
        start, end = self.snapshot_natural(), source.snapshot_natural()
        self.interpolate_natural_(start, end, fraction)


def logits_to_beta(logits):
    alpha_logits, beta_logits = logits.chunk(2, dim=-1)
    return 1.0 + F.softplus(alpha_logits), 1.0 + F.softplus(beta_logits)


def beta_log_prob(alpha, beta, z):
    z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    return (
        torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1.0) * z.log()
        + (beta - 1.0) * torch.log1p(-z)
    )


def beta_entropy(alpha, beta):
    log_beta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return (
        log_beta
        - (alpha - 1.0) * torch.digamma(alpha)
        - (beta - 1.0) * torch.digamma(beta)
        + (alpha + beta - 2.0) * torch.digamma(alpha + beta)
    )


def policy_kl(old_logits, new_logits):
    old_alpha, old_beta = logits_to_beta(old_logits)
    new_alpha, new_beta = logits_to_beta(new_logits)
    old_sum = old_alpha + old_beta
    new_sum = new_alpha + new_beta
    divergence = (
        torch.lgamma(new_alpha)
        + torch.lgamma(new_beta)
        - torch.lgamma(new_sum)
        - torch.lgamma(old_alpha)
        - torch.lgamma(old_beta)
        + torch.lgamma(old_sum)
        + (old_alpha - new_alpha) * torch.digamma(old_alpha)
        + (old_beta - new_beta) * torch.digamma(old_beta)
        + (new_sum - old_sum) * torch.digamma(old_sum)
    )
    return divergence.sum(-1)


def policy_logit_score(logits, action_z):
    """Exact d log pi(a|q)/dq; q alone requires autograd, never network weights."""
    with torch.enable_grad():
        q = logits.detach().requires_grad_(True)
        alpha, beta = logits_to_beta(q)
        log_probability = beta_log_prob(alpha, beta, action_z.detach()).sum()
        return torch.autograd.grad(log_probability, q)[0].detach()


@torch.no_grad()
def bounded_actor_pseudo_target(logits, action_z, normalized_delta, args):
    """Create q + step*delta*dlogpi/dq, bounded per sample and in mean Beta KL."""
    score = policy_logit_score(logits, action_z)
    # Trust-region decisions use fp64 and evaluate the eventual fp32 target. This
    # avoids falsely accepting extreme-logit targets due to fp32 KL cancellation.
    old_logits64 = logits.double()
    raw_change64 = (args.actor_target_step * normalized_delta.detach().unsqueeze(1) * score).double()

    def candidate(scale):
        return logits + (scale * raw_change64).to(logits.dtype)

    def candidate_kl(scale):
        return policy_kl(old_logits64, candidate(scale).double())

    # First cap each sample independently. This prevents a single boundary action
    # from consuming the entire minibatch trust region.
    low = torch.zeros((logits.shape[0], 1), dtype=torch.float64, device=logits.device)
    high = torch.ones_like(low)
    for _ in range(args.target_bisection_steps):
        middle = 0.5 * (low + high)
        divergences = candidate_kl(middle).unsqueeze(1)
        low = torch.where(divergences <= args.actor_target_sample_kl, middle, low)
        high = torch.where(divergences <= args.actor_target_sample_kl, high, middle)
    sample_scaled_change = low * raw_change64

    # A global scale then enforces the desired mean target KL while preserving
    # every coordinate ratio and the per-sample TD/score direction.
    global_low = logits.new_zeros((), dtype=torch.float64)
    global_high = logits.new_ones((), dtype=torch.float64)
    for _ in range(args.target_bisection_steps):
        middle = 0.5 * (global_low + global_high)
        proposed = logits + (middle * sample_scaled_change).to(logits.dtype)
        within = policy_kl(old_logits64, proposed.double()).mean() <= args.actor_target_mean_kl
        global_low = torch.where(within, middle, global_low)
        global_high = torch.where(within, global_high, middle)
    target = logits + (global_low * sample_scaled_change).to(logits.dtype)
    kl = policy_kl(old_logits64, target.double())
    # KL is locally monotone here, but softplus logits are not a global natural
    # parameterization. Explicit backoff makes the bound a checked contract.
    for _ in range(16):
        within = (kl.max() <= args.actor_target_sample_kl) & (kl.mean() <= args.actor_target_mean_kl)
        global_low = torch.where(within, global_low, 0.5 * global_low)
        target = logits + (global_low * sample_scaled_change).to(logits.dtype)
        kl = policy_kl(old_logits64, target.double())
    within = (kl.max() <= args.actor_target_sample_kl) & (kl.mean() <= args.actor_target_mean_kl)
    global_low = torch.where(within, global_low, torch.zeros_like(global_low))
    target = torch.where(within, target, logits)
    kl = torch.where(within, kl, torch.zeros_like(kl))
    return target, score, kl, low.squeeze(1), global_low


class RunningTDRMS:
    def __init__(self, device, decay, minimum):
        self.decay, self.minimum = decay, minimum
        self.mean_square = torch.ones((), device=device)
        self.initialized = False

    @torch.no_grad()
    def normalize(self, delta, clip):
        square = delta.square().mean()
        if self.initialized:
            self.mean_square.lerp_(square, 1.0 - self.decay)
        else:
            self.mean_square.copy_(square)
            self.initialized = True
        normalized = delta / self.mean_square.sqrt().clamp_min(self.minimum)
        return normalized.clamp(-clip, clip) if clip > 0 else normalized


class RecentObservationReservoir:
    """Device-resident ring retaining recent observations for posterior guards."""

    def __init__(self, capacity, observation_dim, device):
        self.capacity = capacity
        self.values = torch.empty((capacity, observation_dim), device=device)
        self.size, self.position = 0, 0

    @torch.no_grad()
    def add(self, observations):
        if self.capacity == 0:
            return
        observations = observations.detach()
        if observations.shape[0] >= self.capacity:
            self.values.copy_(observations[-self.capacity :])
            self.size, self.position = self.capacity, 0
            return
        count = observations.shape[0]
        first = min(count, self.capacity - self.position)
        self.values[self.position : self.position + first].copy_(observations[:first])
        if first < count:
            self.values[: count - first].copy_(observations[first:])
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, maximum):
        if self.size == 0 or maximum == 0:
            return self.values[:0]
        count = min(self.size, maximum)
        # Even spacing covers the recent window deterministically without a CPU
        # RNG synchronization or repeatedly favoring the newest mini-batch.
        indices = torch.linspace(0, self.size - 1, count, device=self.values.device).long()
        return self.values[: self.size].index_select(0, indices)


@torch.no_grad()
def guarded_actor_posterior_update(
    network, candidates, guard_observations, max_kl, corrective_trials=3, max_sample_kl=None
):
    start = network.snapshot_natural()
    old_logits = network.forward_states(guard_observations)[1]
    try:
        network.commit_naturals_(candidates)
    except (RuntimeError, ValueError):
        network.commit_naturals_(start)
        infinity = old_logits.new_tensor(float("inf"))
        return old_logits.new_zeros(()), infinity, 0.0, True, True
    proposal = network.snapshot_natural()

    def current_kl():
        new_logits = network.forward_states(guard_observations)[1]
        divergences = policy_kl(old_logits.double(), new_logits.double())
        return divergences.mean(), divergences.max()

    proposal_kl, proposal_sample_kl = current_kl()
    sample_limit_satisfied = max_sample_kl is None or proposal_sample_kl.item() <= max_sample_kl
    if torch.isfinite(proposal_kl) and torch.isfinite(proposal_sample_kl) and proposal_kl.item() <= max_kl and sample_limit_satisfied:
        return proposal_kl, proposal_kl, 1.0, False, False
    mean_scale = np.sqrt(max_kl / proposal_kl.item()) if torch.isfinite(proposal_kl) and proposal_kl.item() > 0 else 0.0
    sample_scale = (
        np.sqrt(max_sample_kl / proposal_sample_kl.item())
        if max_sample_kl is not None and torch.isfinite(proposal_sample_kl) and proposal_sample_kl.item() > 0
        else 1.0
    )
    scale = min(1.0, mean_scale, sample_scale)
    accepted_kl, accepted_scale = proposal_kl.new_zeros(()), 0.0
    for _ in range(min(corrective_trials, 3)):
        if scale <= 0 or not np.isfinite(scale):
            break
        try:
            network.interpolate_natural_(start, proposal, scale)
        except (RuntimeError, ValueError):
            network.commit_naturals_(start)
            return proposal_kl.new_zeros(()), proposal_kl, 0.0, True, True
        candidate_kl, candidate_sample_kl = current_kl()
        sample_limit_satisfied = max_sample_kl is None or candidate_sample_kl.item() <= max_sample_kl * (1.0 + 1e-6)
        if (
            torch.isfinite(candidate_kl)
            and torch.isfinite(candidate_sample_kl)
            and candidate_kl.item() <= max_kl * (1.0 + 1e-6)
            and sample_limit_satisfied
        ):
            accepted_kl, accepted_scale = candidate_kl, scale
            break
        mean_correction = np.sqrt(max_kl / candidate_kl.item()) if candidate_kl.item() > 0 else 0.0
        sample_correction = (
            np.sqrt(max_sample_kl / candidate_sample_kl.item())
            if max_sample_kl is not None and candidate_sample_kl.item() > 0
            else 1.0
        )
        scale *= 0.99 * min(1.0, mean_correction, sample_correction)
    rolled_back = accepted_scale == 0.0
    if rolled_back:
        network.commit_naturals_(start)
    return accepted_kl, proposal_kl, accepted_scale, True, rolled_back


@torch.no_grad()
def guarded_critic_posterior_update(network, candidates, guard_observations, rms_limit, max_abs_limit, corrective_trials=3):
    start = network.snapshot_natural()
    old_values = network.forward_states(guard_observations)[1].view(-1)
    try:
        network.commit_naturals_(candidates)
    except (RuntimeError, ValueError):
        network.commit_naturals_(start)
        zero, infinity = old_values.new_zeros(()), old_values.new_tensor(float("inf"))
        return zero, zero, infinity, infinity, 0.0, True, True
    proposal = network.snapshot_natural()

    def metrics():
        change = network.forward_states(guard_observations)[1].view(-1) - old_values
        return change.square().mean().sqrt(), change.abs().max()

    proposal_rms, proposal_max = metrics()
    finite = torch.isfinite(proposal_rms) & torch.isfinite(proposal_max)
    if finite and proposal_rms.item() <= rms_limit and proposal_max.item() <= max_abs_limit:
        return proposal_rms, proposal_max, proposal_rms, proposal_max, 1.0, False, False
    rms_scale = rms_limit / proposal_rms.item() if torch.isfinite(proposal_rms) and proposal_rms.item() > 0 else 0.0
    max_scale = max_abs_limit / proposal_max.item() if torch.isfinite(proposal_max) and proposal_max.item() > 0 else 0.0
    scale = min(1.0, rms_scale, max_scale)
    accepted_rms, accepted_max, accepted_scale = proposal_rms.new_zeros(()), proposal_max.new_zeros(()), 0.0
    for _ in range(min(corrective_trials, 3)):
        if scale <= 0 or not np.isfinite(scale):
            break
        try:
            network.interpolate_natural_(start, proposal, scale)
        except (RuntimeError, ValueError):
            network.commit_naturals_(start)
            return (
                proposal_rms.new_zeros(()),
                proposal_max.new_zeros(()),
                proposal_rms,
                proposal_max,
                0.0,
                True,
                True,
            )
        candidate_rms, candidate_max = metrics()
        if (
            torch.isfinite(candidate_rms)
            and torch.isfinite(candidate_max)
            and candidate_rms.item() <= rms_limit * (1.0 + 1e-6)
            and candidate_max.item() <= max_abs_limit * (1.0 + 1e-6)
        ):
            accepted_rms, accepted_max, accepted_scale = candidate_rms, candidate_max, scale
            break
        rms_correction = rms_limit / candidate_rms.item() if candidate_rms.item() > 0 else 0.0
        max_correction = max_abs_limit / candidate_max.item() if candidate_max.item() > 0 else 0.0
        scale *= 0.99 * min(1.0, rms_correction, max_correction)
    rolled_back = accepted_scale == 0.0
    if rolled_back:
        network.commit_naturals_(start)
    return accepted_rms, accepted_max, proposal_rms, proposal_max, accepted_scale, True, rolled_back


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.actor = BPCNetwork(observation_dim, 2 * action_dim, args, output_std=0.01)
        self.critic = BPCNetwork(observation_dim, 1, args, output_std=1.0)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def action_from_z(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    @torch.no_grad()
    def get_action_and_value(self, observation, action_z=None):
        actor_states, logits = self.actor.forward_states(observation)
        critic_states, value = self.critic.forward_states(observation)
        alpha, beta = logits_to_beta(logits)
        distribution = Beta(alpha, beta)
        if action_z is None:
            action_z = distribution.sample()
        action_z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        return (
            self.action_from_z(action_z),
            action_z,
            beta_log_prob(alpha, beta, action_z).sum(1),
            beta_entropy(alpha, beta).sum(1),
            value.view(-1),
            logits,
            actor_states,
            critic_states,
        )


def bootstrap_observations(next_obs, truncations, infos):
    bootstrap_obs = np.array(next_obs, copy=True)
    if not np.any(truncations):
        return bootstrap_obs
    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None:
        raise RuntimeError("truncated transition missing final_observation")
    for index in np.flatnonzero(truncations):
        if (final_mask is not None and not final_mask[index]) or final_observations[index] is None:
            raise RuntimeError(f"truncated environment {index} has no final observation")
        bootstrap_obs[index] = final_observations[index]
    return bootstrap_obs


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array") if capture_video and idx == 0 else gym.make(env_id)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def validate_args(args):
    if (
        args.inference_steps < 1
        or args.inference_lr <= 0
        or args.inference_backtracks < 0
        or args.inference_curvature_floor <= 0
        or args.inference_curvature_refresh_interval < 1
    ):
        raise ValueError("latent inference settings must be positive")
    if not 0 < args.inference_damping < 1:
        raise ValueError("inference damping must be in (0,1)")
    if not 0 <= args.actor_posterior_discount <= 1 or not 0 <= args.critic_posterior_discount <= 1:
        raise ValueError("posterior discounts must be in [0,1]")
    if args.actor_evidence_weight <= 0 or args.critic_evidence_weight <= 0:
        raise ValueError("evidence weights must be positive")
    if not 0 < args.actor_target_mean_kl <= args.actor_target_sample_kl:
        raise ValueError("actor target KLs must satisfy 0 < mean <= sample")
    if args.actor_target_step <= 0 or args.target_bisection_steps < 1:
        raise ValueError("actor pseudo-target step and bisection count must be positive")
    if not 0 <= args.td_rms_decay < 1 or args.td_rms_min <= 0 or args.td_norm_clip <= 0 or args.critic_td_clip <= 0:
        raise ValueError("TD normalization and clipping settings must be positive")
    if (
        args.actor_posterior_max_kl <= 0
        or args.actor_posterior_max_sample_kl < args.actor_posterior_max_kl
        or args.critic_posterior_rms_limit <= 0
        or args.critic_posterior_max_abs <= 0
    ):
        raise ValueError("posterior guard limits must be positive")
    if not 1 <= args.posterior_guard_trials <= 3:
        raise ValueError("posterior guard permits one to three corrective trials")
    if args.anchor_capacity < 0 or not 0 <= args.anchor_sample_size <= args.anchor_capacity:
        raise ValueError("anchor sample size must fit the reservoir")
    if not 0 < args.target_critic_tau <= 1 or args.target_critic_interval < 1:
        raise ValueError("target critic settings must be positive")
    if (
        args.prior_column_covariance <= 0
        or args.prior_expected_precision <= 0
        or args.posterior_jitter <= 0
        or args.prior_dof_offset <= -1
    ):
        raise ValueError("MNW scales must be positive and prior DOF offset greater than -1")


def main():
    args = tyro.cli(Args)
    validate_args(args)
    args.num_updates = args.total_timesteps // args.num_envs
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=run_name)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision(args.torch_float32_matmul_precision)
    assert args.cuda and torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, index, args.capture_video, run_name, args.gamma) for index in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous actions are supported"
    agent = Agent(envs, args).to(device)
    target_critic = copy.deepcopy(agent.critic).to(device)
    observation_dim = int(np.prod(envs.single_observation_space.shape))
    anchors = RecentObservationReservoir(args.anchor_capacity, observation_dim, device)
    td_rms = RunningTDRMS(device, args.td_rms_decay, args.td_rms_min)

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    for update in range(1, args.num_updates + 1):
        global_step += args.num_envs
        obs = next_obs
        with torch.no_grad():
            action, action_z, logprob, entropy, value, logits, actor_states, critic_states = agent.get_action_and_value(obs)
        next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())
        bootstrap_np = bootstrap_observations(next_obs_np, truncated_np, infos)
        bootstrap = torch.as_tensor(bootstrap_np, dtype=torch.float32, device=device)
        reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
        terminated = torch.as_tensor(terminated_np, dtype=torch.bool, device=device)
        with torch.no_grad():
            _, target_next_value = target_critic.forward_states(bootstrap)
            td_target = reward + args.gamma * (~terminated).float() * target_next_value.view(-1)
            td_error = td_target - value
            normalized_delta = td_rms.normalize(td_error, args.td_norm_clip)
            critic_target = value + td_error.clamp(-args.critic_td_clip, args.critic_td_clip)
            actor_target, actor_score, target_kl, sample_scales, global_target_scale = bounded_actor_pseudo_target(
                logits, action_z, normalized_delta, args
            )

        actor_states, actor_sweeps, actor_change, actor_inference_accepts, actor_inference_rejects = agent.actor.settle_clamped_output(
            obs, actor_states, actor_target, args
        )
        critic_states, critic_sweeps, critic_change, critic_inference_accepts, critic_inference_rejects = agent.critic.settle_clamped_output(
            obs, critic_states, critic_target.unsqueeze(1), args
        )

        actor_candidates = agent.actor.posterior_candidates(
            obs, actor_states, actor_target, args.actor_posterior_discount, args.actor_evidence_weight
        )
        critic_candidates = agent.critic.posterior_candidates(
            obs, critic_states, critic_target.unsqueeze(1), args.critic_posterior_discount, args.critic_evidence_weight
        )
        anchor_batch = anchors.sample(args.anchor_sample_size)
        actor_guard_observations = torch.cat((obs, anchor_batch), dim=0)
        critic_guard_observations = torch.cat((obs, bootstrap, anchor_batch), dim=0)
        actor_post_kl, actor_proposal_kl, actor_post_scale, actor_limited, actor_rollback = guarded_actor_posterior_update(
            agent.actor,
            actor_candidates,
            actor_guard_observations,
            args.actor_posterior_max_kl,
            args.posterior_guard_trials,
            args.actor_posterior_max_sample_kl,
        )
        (
            critic_post_rms,
            critic_post_max,
            critic_proposal_rms,
            critic_proposal_max,
            critic_post_scale,
            critic_limited,
            critic_rollback,
        ) = guarded_critic_posterior_update(
            agent.critic,
            critic_candidates,
            critic_guard_observations,
            args.critic_posterior_rms_limit,
            args.critic_posterior_max_abs,
            args.posterior_guard_trials,
        )
        anchors.add(obs)

        if update % args.target_critic_interval == 0:
            target_fraction = 1.0 - (1.0 - args.target_critic_tau) ** args.target_critic_interval
            target_critic.soft_update_from_(agent.critic, target_fraction)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_return = float(np.asarray(info["episode"]["r"]))
                    episodic_length = int(np.asarray(info["episode"]["l"]))
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        if update % args.log_interval == 0 or update == 1:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("losses/td_error_mean", td_error.mean().item(), global_step)
            writer.add_scalar("losses/td_error_rms", td_rms.mean_square.sqrt().item(), global_step)
            writer.add_scalar("policy/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("policy/pseudo_target_mean_kl", target_kl.mean().item(), global_step)
            writer.add_scalar("policy/pseudo_target_max_kl", target_kl.max().item(), global_step)
            writer.add_scalar("policy/pseudo_target_global_scale", global_target_scale, global_step)
            writer.add_scalar("policy/pseudo_target_sample_scale_mean", sample_scales.mean().item(), global_step)
            writer.add_scalar("policy/logit_score_rms", actor_score.square().mean().sqrt().item(), global_step)
            writer.add_scalar("policy/posterior_guard_kl", actor_post_kl.item(), global_step)
            writer.add_scalar("policy/posterior_proposal_kl", actor_proposal_kl.item(), global_step)
            writer.add_scalar("policy/posterior_step_scale", actor_post_scale, global_step)
            writer.add_scalar("policy/posterior_limited", float(actor_limited), global_step)
            writer.add_scalar("policy/posterior_rollback", float(actor_rollback), global_step)
            writer.add_scalar("critic/posterior_guard_rms", critic_post_rms.item(), global_step)
            writer.add_scalar("critic/posterior_guard_max_abs", critic_post_max.item(), global_step)
            writer.add_scalar("critic/posterior_proposal_rms", critic_proposal_rms.item(), global_step)
            writer.add_scalar("critic/posterior_proposal_max_abs", critic_proposal_max.item(), global_step)
            writer.add_scalar("critic/posterior_step_scale", critic_post_scale, global_step)
            writer.add_scalar("critic/posterior_limited", float(critic_limited), global_step)
            writer.add_scalar("critic/posterior_rollback", float(critic_rollback), global_step)
            writer.add_scalar("bpc/actor_settle_sweeps", actor_sweeps, global_step)
            writer.add_scalar("bpc/critic_settle_sweeps", critic_sweeps, global_step)
            writer.add_scalar("bpc/actor_maximum_change", actor_change.item(), global_step)
            writer.add_scalar("bpc/critic_maximum_change", critic_change.item(), global_step)
            writer.add_scalar("bpc/actor_inference_accepts", actor_inference_accepts, global_step)
            writer.add_scalar("bpc/actor_inference_rejects", actor_inference_rejects, global_step)
            writer.add_scalar("bpc/critic_inference_accepts", critic_inference_accepts, global_step)
            writer.add_scalar("bpc/critic_inference_rejects", critic_inference_rejects, global_step)
            writer.add_scalar("bpc/actor_posterior_discount", args.actor_posterior_discount, global_step)
            writer.add_scalar("bpc/critic_posterior_discount", args.critic_posterior_discount, global_step)
            writer.add_scalar("bpc/anchor_count", anchors.size, global_step)

    envs.close()
    writer.close()
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")


if __name__ == "__main__":
    main()

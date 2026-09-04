# Full Matrix-Normal-Wishart BPC hidden hierarchy + streaming TD(lambda) heads v1.
#
# Each Gaussian hidden edge carries the paper's complete natural posterior
# (Lambda, Q, R, xi), performs expected-free-energy latent inference including
# the weight-uncertainty correction, and receives one conjugate natural update
# from settled pre/post activity. Reward shapes settling once through the terminal
# actor/critic energy; it never multiplies the Bayesian sufficient statistics.
# Only the nonconjugate Beta and value heads use conventional parameter-level
# TD(lambda) traces. This is therefore an honest BPC-hidden RL hybrid, not an
# end-to-end Bayesian treatment of the control objective. Capped paper-style SVI
# and generalized discounted conjugate updates are both available.
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
from torch.distributions import Beta, kl_divergence
from torch.func import functional_call, grad, vmap
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
    learning_rate: float = 3e-3
    anneal_lr: bool = True
    gamma: float = 0.99
    trace_lambda: float = 0.95
    actor_trace_lambda: Optional[float] = None
    critic_trace_lambda: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    td_rms_decay: float = 0.999
    td_norm_clip: float = 10.0
    td_rms_min: float = 0.1
    critic_td_clip: float = 10.0
    actor_terminal_coef: float = 10.0
    """actor reward-terminal energy strength; does not scale head TD(lambda)"""

    hidden_size: int = 64
    num_hidden_layers: int = 6
    hidden_state_clip: float = 5.0
    inference_steps: int = 10
    inference_lr: float = 0.01
    inference_beta1: float = 0.9
    inference_beta2: float = 0.999
    inference_eps: float = 1e-8
    inference_tolerance: float = 0.0
    posterior_effective_samples: Optional[float] = None
    """optional online evidence tempering; None uses the exact batch sums"""
    posterior_update_mode: str = "discounted"
    """discounted (streaming generalized BPC) or paper_svi"""
    posterior_discount: float = 0.99
    """power-prior retention rho for discounted posterior updates"""
    posterior_kappa_exponent: float = 0.25
    posterior_kappa_max: float = 0.1
    """online-RL stability cap; paper's uncapped schedule starts at one"""
    posterior_kappa_min: float = 0.01
    posterior_jitter: float = 1e-5
    prior_column_covariance: float = 1e-3
    """RL-scaled V prior; use 10 to reproduce the paper experiments"""
    prior_mean_mode: str = "initial"
    """initial preserves unseen online features; zero reproduces the paper prior"""
    prior_expected_precision: float = 1.0
    """isotropic E[Sigma^-1] when prior_wishart_scale is None"""
    prior_wishart_scale: Optional[float] = None
    """direct Psi scale override; use 1000 with V=10 for paper priors"""
    prior_dof_offset: float = 2.0

    target_update_kl: float = 1e-4
    max_update_kl: float = 5e-4
    kl_bisection_steps: int = 10
    posterior_guard_trials: int = 2
    kl_adaptation_rate: float = 0.05
    kl_scale_min: float = 0.05
    kl_scale_max: float = 100.0
    critic_posterior_rms_fraction: float = 0.1
    """posterior-only value RMS limit as a fraction of running TD RMS"""
    critic_posterior_rms_floor: float = 1e-3
    """minimum posterior-only value RMS limit"""
    critic_posterior_max_abs: float = 0.1
    """hard maximum absolute posterior-induced value change"""

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
    """Cholesky inverse with geometrically increased numerical jitter."""
    matrix = symmetrize(matrix)
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    current_jitter = 0.0
    for _ in range(7):
        chol, info = torch.linalg.cholesky_ex(matrix + current_jitter * identity)
        if not torch.any(info):
            return torch.cholesky_inverse(chol)
        current_jitter = jitter if current_jitter == 0.0 else current_jitter * 10.0
    raise RuntimeError("Matrix-Normal-Wishart parameter left the SPD cone")


def mnw_to_natural(parameters: MNWParameters, jitter=1e-6):
    M, V, Psi, nu = parameters
    dy, dx = M.shape[-2:]
    if V.shape[-2:] != (dx, dx) or Psi.shape[-2:] != (dy, dy) or M.shape[:-2] != V.shape[:-2] or M.shape[:-2] != Psi.shape[:-2]:
        raise ValueError("incompatible Matrix-Normal-Wishart shapes")
    Lambda = spd_inverse(V, jitter)
    Q = M @ Lambda
    R = spd_inverse(Psi, jitter) + M @ Lambda @ M.transpose(-1, -2)
    xi = torch.as_tensor(nu, dtype=M.dtype, device=M.device) - dy + dx - 1
    return NaturalParameters(symmetrize(Lambda), Q, symmetrize(R), xi)


def natural_to_mnw(natural: NaturalParameters, jitter=1e-6):
    Lambda, Q, R, xi = natural
    dx = Lambda.shape[-1]
    dy = R.shape[-1]
    batch_shape = Lambda.shape[:-2]
    if (
        Lambda.shape[-2:] != (dx, dx)
        or Q.shape[-2:] != (dy, dx)
        or R.shape[-2:] != (dy, dy)
        or Q.shape[:-2] != batch_shape
        or R.shape[:-2] != batch_shape
    ):
        raise ValueError("incompatible Matrix-Normal-Wishart natural shapes")
    V = spd_inverse(Lambda, jitter)
    M = Q @ V
    psi_inverse = symmetrize(R - Q @ V @ Q.transpose(-1, -2))
    Psi = spd_inverse(psi_inverse, jitter)
    nu = xi + dy - dx + 1
    if bool(torch.any(nu <= dy - 1).item()):
        raise ValueError(f"Wishart degrees of freedom must exceed dy-1={dy - 1}")
    return MNWParameters(M, symmetrize(V), symmetrize(Psi), nu)


def recover_natural_group(naturals, jitter):
    """Recover equal-shaped MNW posteriors in one batched Cholesky path."""
    if not naturals:
        return []
    batched = NaturalParameters(*(torch.stack(values) for values in zip(*naturals)))
    posterior = natural_to_mnw(batched, jitter)
    return [MNWParameters(*(value[index] for value in posterior)) for index in range(len(naturals))]


def sufficient_statistics(x, y):
    """Statistics for y = W x + noise; Syx orientation matches W (dy, dx)."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be aligned rank-two batches")
    return SufficientStatistics(
        x.transpose(0, 1) @ x,
        y.transpose(0, 1) @ x,
        y.transpose(0, 1) @ y,
        x.new_tensor(float(x.shape[0])),
    )


def scale_statistics(stats, scale):
    return SufficientStatistics(*(value * scale for value in stats))


def conjugate_candidate(prior: NaturalParameters, stats: SufficientStatistics):
    return NaturalParameters(
        prior.Lambda + stats.Sxx,
        prior.Q + stats.Syx,
        prior.R + stats.Syy,
        prior.xi + stats.N,
    )


def add_statistics(natural: NaturalParameters, stats: SufficientStatistics):
    """Exact sequential conjugate update (distinct from paper-style SVI)."""
    return conjugate_candidate(natural, stats)


def stochastic_natural_update(current, candidate, kappa):
    if not 0.0 < kappa <= 1.0:
        raise ValueError("kappa must be in (0, 1]")
    return NaturalParameters(*(old.lerp(new, kappa) for old, new in zip(current, candidate)))


def discounted_conjugate_update(prior, current, stats, rho):
    """Power-Bayes update eta0 + rho*(eta-eta0) + sufficient statistics."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    discounted = NaturalParameters(*(base + rho * (old - base) for base, old in zip(prior, current)))
    return add_statistics(discounted, stats)


class MatrixNormalWishartEdge(nn.Module):
    """A full MNW posterior for one bias-augmented Gaussian BPC edge."""

    def __init__(self, input_dim, output_dim, args, first_edge=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.first_edge = first_edge
        self.jitter = args.posterior_jitter
        dx = input_dim + 1
        # Long online sequences make the Schur complement R-Q Lambda^-1 Q^T
        # fragile in fp32. Persistent natural state is fp64; inference uses cached
        # fp32 moments so this does not turn every settling matmul into fp64.
        dtype = torch.float64
        V_prior = args.prior_column_covariance * torch.eye(dx, dtype=dtype)
        nu_prior = torch.tensor(output_dim + args.prior_dof_offset, dtype=dtype)
        psi_scale = (
            args.prior_expected_precision / float(nu_prior)
            if args.prior_wishart_scale is None
            else args.prior_wishart_scale
        )
        Psi_prior = psi_scale * torch.eye(output_dim, dtype=dtype)
        # Batches here have 16 rows while most augmented edges have 65 columns.
        # Giving the actual prior the initialized mean preserves the feature map
        # in directions absent from a small batch. The paper used a zero-mean prior
        # with batches of 128; this is the explicit small-online-batch adaptation.
        mean = nn.Linear(input_dim, output_dim)
        nn.init.orthogonal_(mean.weight, np.sqrt(2))
        nn.init.zeros_(mean.bias)
        M_initial = torch.cat((mean.weight.detach(), mean.bias.detach().unsqueeze(1)), dim=1).to(dtype)
        prior_mean = M_initial if args.prior_mean_mode == "initial" else torch.zeros_like(M_initial)
        prior = mnw_to_natural(MNWParameters(prior_mean, V_prior, Psi_prior, nu_prior), self.jitter)
        current = mnw_to_natural(MNWParameters(M_initial, V_prior, Psi_prior, nu_prior), self.jitter)
        for prefix, natural in (("prior", prior), ("natural", current)):
            self.register_buffer(f"{prefix}_Lambda", natural.Lambda.clone())
            self.register_buffer(f"{prefix}_Q", natural.Q.clone())
            self.register_buffer(f"{prefix}_R", natural.R.clone())
            self.register_buffer(f"{prefix}_xi", natural.xi.clone())
        self.register_buffer("cached_M", torch.empty((output_dim, dx), dtype=torch.float32))
        self.register_buffer("cached_V", torch.empty((dx, dx), dtype=torch.float32))
        self.register_buffer("cached_precision", torch.empty((output_dim, output_dim), dtype=torch.float32))
        self._refresh_cache()

    def _natural(self, prefix="natural"):
        return NaturalParameters(*(getattr(self, f"{prefix}_{name}") for name in NaturalParameters._fields))

    def posterior(self):
        return natural_to_mnw(self._natural(), self.jitter)

    @torch.no_grad()
    def _refresh_cache(self):
        posterior = self.posterior()
        self._refresh_cache_from_posterior(posterior)

    @torch.no_grad()
    def _refresh_cache_from_posterior(self, posterior):
        self.cached_M.copy_(posterior.M.float())
        self.cached_V.copy_(posterior.V.float())
        self.cached_precision.copy_((posterior.nu * posterior.Psi).float())

    @torch.no_grad()
    def commit_natural_(self, natural, posterior=None):
        # Recovery is both the SPD validation and the cache source. Passing a
        # batched recovery avoids performing the same Cholesky work twice.
        posterior = natural_to_mnw(natural, self.jitter) if posterior is None else posterior
        for name, value in zip(NaturalParameters._fields, natural):
            getattr(self, f"natural_{name}").copy_(value)
        self._refresh_cache_from_posterior(posterior)

    def features(self, state):
        activated = state if self.first_edge else torch.tanh(state)
        return torch.cat((activated, torch.ones_like(activated[:, :1])), dim=1)

    def feature_derivative(self, state):
        return torch.ones_like(state) if self.first_edge else 1.0 - torch.tanh(state).square()

    def predict(self, previous_state):
        return self.features(previous_state) @ self.cached_M.transpose(0, 1)

    def expected_precision(self):
        return self.cached_precision

    def expected_energy(self, previous_state, state):
        """Per-example expected negative log likelihood, up to constants."""
        x = self.features(previous_state)
        residual = state - x @ self.cached_M.transpose(0, 1)
        data_term = torch.einsum("bi,ij,bj->b", residual, self.cached_precision, residual)
        uncertainty = self.output_dim * torch.einsum("bi,ij,bj->b", x, self.cached_V, x)
        return 0.5 * (data_term + uncertainty)

    def incoming_gradient(self, previous_state, state):
        residual = state - self.features(previous_state) @ self.cached_M.transpose(0, 1)
        return residual @ self.cached_precision

    def previous_state_gradient(self, previous_state, state):
        """Full d E_edge / d previous_state, including d_y D V x."""
        x = self.features(previous_state)
        residual = state - x @ self.cached_M.transpose(0, 1)
        mean_term = -(residual @ self.cached_precision) @ self.cached_M[:, : self.input_dim]
        covariance_term = self.output_dim * (x @ self.cached_V[:, : self.input_dim])
        return (mean_term + covariance_term) * self.feature_derivative(previous_state)

    @torch.no_grad()
    def update_from_activity(self, previous_state, state, effective_samples, mode, kappa, discount):
        updated = self.posterior_candidate(previous_state, state, effective_samples, mode, kappa, discount)
        self.commit_natural_(updated)

    @torch.no_grad()
    def posterior_candidate(self, previous_state, state, effective_samples, mode, kappa, discount):
        x = self.features(previous_state)
        stats = sufficient_statistics(x.double(), state.double())
        # None is the exact conjugate sum from the paper. A configured effective
        # sample count is an explicitly tempered online-RL option, not exact BPC.
        if effective_samples is not None:
            stats = scale_statistics(stats, effective_samples / float(x.shape[0]))
        prior, current = self._natural("prior"), self._natural()
        if mode == "paper_svi":
            updated = stochastic_natural_update(current, conjugate_candidate(prior, stats), kappa)
        elif mode == "discounted":
            updated = discounted_conjugate_update(prior, current, stats, discount)
        else:
            raise ValueError(f"unknown posterior update mode: {mode}")
        return updated


class BPCStack(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.hidden_state_clip = args.hidden_state_clip
        self.edges = nn.ModuleList()
        for index in range(args.num_hidden_layers):
            self.edges.append(
                MatrixNormalWishartEdge(
                    input_dim if index == 0 else args.hidden_size,
                    args.hidden_size,
                    args,
                    first_edge=index == 0,
                )
            )

    @torch.no_grad()
    def forward_states(self, observation):
        states = []
        previous = observation
        for edge in self.edges:
            state = edge.predict(previous)
            if self.hidden_state_clip > 0:
                state.clamp_(-self.hidden_state_clip, self.hidden_state_clip)
            states.append(state)
            previous = state
        return states

    def latent_gradients(self, observation, states, terminal_gradient):
        gradients = []
        for index, edge in enumerate(self.edges):
            previous = observation if index == 0 else states[index - 1]
            gradient = edge.incoming_gradient(previous, states[index])
            if index + 1 < len(self.edges):
                gradient = gradient + self.edges[index + 1].previous_state_gradient(states[index], states[index + 1])
            else:
                gradient = gradient + terminal_gradient
            gradients.append(gradient)
        return gradients

    @torch.no_grad()
    def update_posteriors(self, observation, states, effective_samples, mode, kappa, discount):
        candidates = []
        previous = observation
        for edge, state in zip(self.edges, states):
            candidates.append(edge.posterior_candidate(previous, state, effective_samples, mode, kappa, discount))
            previous = state
        self.commit_naturals_(candidates)

    @torch.no_grad()
    def commit_naturals_(self, naturals):
        if len(naturals) != len(self.edges):
            raise ValueError("posterior proposal does not match stack")
        groups = {}
        for index, (edge, natural) in enumerate(zip(self.edges, naturals)):
            key = (natural.Lambda.shape, natural.Q.shape, natural.R.shape, edge.jitter)
            groups.setdefault(key, []).append(index)
        for indices in groups.values():
            posteriors = recover_natural_group([naturals[index] for index in indices], self.edges[indices[0]].jitter)
            for index, posterior in zip(indices, posteriors):
                self.edges[index].commit_natural_(naturals[index], posterior)

    @torch.no_grad()
    def snapshot_natural(self):
        return [NaturalParameters(*(value.clone() for value in edge._natural())) for edge in self.edges]

    @torch.no_grad()
    def interpolate_natural_(self, start, end, fraction):
        if len(start) != len(self.edges) or len(end) != len(self.edges):
            raise ValueError("posterior snapshot does not match stack")
        interpolated = [
            NaturalParameters(*(old.lerp(new, fraction) for old, new in zip(initial, final)))
            for initial, final in zip(start, end)
        ]
        self.commit_naturals_(interpolated)


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


def layer_init(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class BetaHead(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.alpha = layer_init(nn.Linear(hidden_size, action_dim), 0.01)
        self.beta = layer_init(nn.Linear(hidden_size, action_dim), 0.01)

    def forward(self, state):
        return 1.0 + F.softplus(self.alpha(state)), 1.0 + F.softplus(self.beta(state))


class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value = layer_init(nn.Linear(hidden_size, 1), 1.0)

    def forward(self, state):
        return self.value(state)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor_bpc = BPCStack(observation_dim, args)
        self.actor_head = BetaHead(args.hidden_size, action_dim)
        self.critic_bpc = BPCStack(observation_dim, args)
        self.critic_head = ValueHead(args.hidden_size)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def action_from_z(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    @torch.no_grad()
    def get_action_and_value(self, observation, action_z=None):
        actor_states = self.actor_bpc.forward_states(observation)
        critic_states = self.critic_bpc.forward_states(observation)
        alpha, beta = self.actor_head(actor_states[-1])
        value = self.critic_head(critic_states[-1]).view(-1)
        distribution = Beta(alpha, beta)
        if action_z is None:
            action_z = distribution.sample()
        action_z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        return (
            self.action_from_z(action_z),
            action_z,
            beta_log_prob(alpha, beta, action_z).sum(1),
            beta_entropy(alpha, beta).sum(1),
            value,
            alpha,
            beta,
            actor_states,
            critic_states,
        )


class LatentAdam:
    """Fresh per-transition Adam state, matching the paper's hidden-state optimizer."""

    def __init__(self, states, beta1, beta2, epsilon):
        self.m = [torch.zeros_like(state) for state in states]
        self.v = [torch.zeros_like(state) for state in states]
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.step = 0

    @torch.no_grad()
    def update(self, states, gradients, learning_rate, state_clip):
        self.step += 1
        maximum_change = states[0].new_zeros(())
        for index, (state, gradient) in enumerate(zip(states, gradients)):
            self.m[index].mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            self.v[index].mul_(self.beta2).addcmul_(gradient, gradient, value=1.0 - self.beta2)
            m_hat = self.m[index] / (1.0 - self.beta1**self.step)
            v_hat = self.v[index] / (1.0 - self.beta2**self.step)
            change = learning_rate * m_hat / (v_hat.sqrt() + self.epsilon)
            state.sub_(change)
            if state_clip > 0:
                state.clamp_(-state_clip, state_clip)
            maximum_change = torch.maximum(maximum_change, change.abs().max())
        return maximum_change


def actor_terminal_gradient(head, top_state, action_z, actor_td, coefficient=1.0):
    with torch.enable_grad():
        state = top_state.detach().requires_grad_(True)
        alpha, beta = head(state)
        energy = -(coefficient * actor_td.detach() * beta_log_prob(alpha, beta, action_z.detach()).sum(1)).sum()
        return torch.autograd.grad(energy, state)[0].detach()


def critic_terminal_gradient(head, top_state, td_target, vf_coef):
    with torch.enable_grad():
        state = top_state.detach().requires_grad_(True)
        residual = head(state).view(-1) - td_target.detach()
        energy = 0.5 * vf_coef * residual.square().sum()
        return torch.autograd.grad(energy, state)[0].detach()


def settle_stack(stack, observation, states, terminal_gradient_fn, args):
    optimizer = LatentAdam(states, args.inference_beta1, args.inference_beta2, args.inference_eps)
    steps_taken = 0
    maximum_change = observation.new_tensor(float("inf"))
    for step in range(args.inference_steps):
        terminal_gradient = terminal_gradient_fn(states[-1])
        gradients = stack.latent_gradients(observation, states, terminal_gradient)
        maximum_change = optimizer.update(states, gradients, args.inference_lr, args.hidden_state_clip)
        steps_taken = step + 1
        if args.inference_tolerance > 0 and maximum_change.item() < args.inference_tolerance:
            break
    return states, steps_taken, maximum_change


class FlatParameterLayout:
    def __init__(self, module):
        self.names, self.parameters, self.slices = [], [], {}
        offset = 0
        for name, parameter in module.named_parameters():
            self.names.append(name)
            self.parameters.append(parameter)
            self.slices[name] = slice(offset, offset + parameter.numel())
            offset += parameter.numel()
        self.numel = offset

    def flatten_batched(self, gradients):
        batch = gradients[self.names[0]].shape[0]
        return torch.cat([gradients[name].reshape(batch, -1) for name in self.names], dim=1)

    def flat_parameters(self):
        return torch.cat([parameter.detach().reshape(-1) for parameter in self.parameters])

    @torch.no_grad()
    def add_flat_(self, direction, step_size):
        for name, parameter in zip(self.names, self.parameters):
            parameter.add_(direction[self.slices[name]].view_as(parameter), alpha=step_size)

    @torch.no_grad()
    def copy_flat_(self, flat):
        for name, parameter in zip(self.names, self.parameters):
            parameter.copy_(flat[self.slices[name]].view_as(parameter))


class ExactHeadJacobians:
    def __init__(self, actor_head, critic_head, args):
        self.actor_head, self.critic_head = actor_head, critic_head

        def actor_score(parameters, state, action_z):
            alpha, beta = functional_call(actor_head, parameters, (state.unsqueeze(0),))
            return beta_log_prob(alpha.squeeze(0), beta.squeeze(0), action_z).sum()

        def critic_value(parameters, state):
            return functional_call(critic_head, parameters, (state.unsqueeze(0),)).squeeze()

        def actor_entropy(parameters, state):
            alpha, beta = functional_call(actor_head, parameters, (state.unsqueeze(0),))
            return beta_entropy(alpha.squeeze(0), beta.squeeze(0)).sum()

        self._actor = vmap(grad(actor_score), in_dims=(None, 0, 0), randomness="error")
        self._critic = vmap(grad(critic_value), in_dims=(None, 0), randomness="error")
        self._entropy = vmap(grad(actor_entropy), in_dims=(None, 0), randomness="error")
        if args.compile:
            self._actor = torch.compile(self._actor, mode=args.compile_mode, dynamic=False)
            self._critic = torch.compile(self._critic, mode=args.compile_mode, dynamic=False)
            if args.ent_coef != 0.0:
                self._entropy = torch.compile(self._entropy, mode=args.compile_mode, dynamic=False)

    def actor(self, state, action_z):
        return self._actor(dict(self.actor_head.named_parameters()), state, action_z)

    def critic(self, state):
        return self._critic(dict(self.critic_head.named_parameters()), state)

    def entropy(self, state):
        return self._entropy(dict(self.actor_head.named_parameters()), state)


class EligibilityTrace:
    def __init__(self, num_envs, num_parameters, device):
        self.value = torch.zeros((num_envs, num_parameters), device=device)

    @torch.no_grad()
    def accumulate(self, instantaneous, decay):
        self.value.mul_(decay).add_(instantaneous)

    def modulated_mean(self, signal):
        return (signal.detach().unsqueeze(1) * self.value).mean(0)

    @torch.no_grad()
    def reset(self, done):
        self.value.masked_fill_(done.unsqueeze(1), 0.0)


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
        result = delta / self.mean_square.sqrt().clamp_min(self.minimum)
        return result.clamp(-clip, clip) if clip > 0 else result


def clip_update_direction(direction, max_norm):
    norm = direction.norm()
    if max_norm > 0:
        direction = direction * (max_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
    return direction, norm


class OnlineKLController:
    def __init__(self, args, device):
        self.target = args.target_update_kl
        self.rate = args.kl_adaptation_rate
        self.log_min = float(np.log(args.kl_scale_min))
        self.log_max = float(np.log(args.kl_scale_max))
        self.log_scale = torch.zeros((), device=device)

    @property
    def scale(self):
        return self.log_scale.exp()

    @torch.no_grad()
    def observe(self, kl):
        feedback = ((self.target - kl) / self.target).clamp(-2.0, 2.0)
        self.log_scale.add_(self.rate * feedback).clamp_(self.log_min, self.log_max)


@torch.no_grad()
def apply_actor_update_with_kl_limit(actor, layout, state, old_alpha, old_beta, direction, lr, scale, args):
    original = layout.flat_parameters()

    def apply(candidate_scale):
        layout.copy_flat_(original)
        layout.add_flat_(direction, lr * candidate_scale)

    def current_kl():
        alpha, beta = actor(state)
        return kl_divergence(Beta(old_alpha, old_beta), Beta(alpha, beta)).sum(1).mean()

    proposal = float(scale.item())
    apply(proposal)
    accepted_kl = current_kl()
    if accepted_kl.item() <= args.max_update_kl:
        return accepted_kl, proposal, False
    low, high = 0.0, proposal
    accepted_kl = torch.zeros_like(accepted_kl)
    for _ in range(args.kl_bisection_steps):
        candidate = 0.5 * (low + high)
        apply(candidate)
        candidate_kl = current_kl()
        if candidate_kl.item() <= args.max_update_kl:
            low, accepted_kl = candidate, candidate_kl
        else:
            high = candidate
    apply(low)
    return accepted_kl, low, True


@torch.no_grad()
def apply_actor_posterior_with_kl_limit(
    stack,
    head,
    observation,
    behavior_alpha,
    behavior_beta,
    posterior_update,
    max_kl,
    corrective_trials,
):
    """Commit a KL-safe posterior fraction using local quadratic KL scaling."""
    initial = stack.snapshot_natural()
    pre_states = stack.forward_states(observation)
    pre_alpha, pre_beta = head(pre_states[-1])
    posterior_update()
    proposed = stack.snapshot_natural()

    def policy_parameters():
        states = stack.forward_states(observation)
        return head(states[-1])

    def combined_kl():
        alpha, beta = policy_parameters()
        return kl_divergence(Beta(behavior_alpha, behavior_beta), Beta(alpha, beta)).sum(1).mean()

    proposal_kl = combined_kl()
    if torch.isfinite(proposal_kl) and proposal_kl.item() <= max_kl:
        final_alpha, final_beta = policy_parameters()
        posterior_kl = kl_divergence(Beta(pre_alpha, pre_beta), Beta(final_alpha, final_beta)).sum(1).mean()
        return proposal_kl, posterior_kl, 1.0, False, proposal_kl

    stack.interpolate_natural_(initial, proposed, 0.0)
    baseline_kl = combined_kl()
    if not torch.isfinite(baseline_kl) or baseline_kl.item() > max_kl * (1.0 + 1e-5):
        # The independently guarded head update must fit at fraction zero.
        raise RuntimeError("head-only actor update already exceeds combined KL limit")

    # KL is locally quadratic in parameter displacement. Account for KL already
    # spent by the head update, then allow at most two validated corrections.
    available = max(max_kl - baseline_kl.item(), 0.0)
    proposal_excess = max(proposal_kl.item() - baseline_kl.item(), 1e-30) if torch.isfinite(proposal_kl) else float("inf")
    scale = min(1.0, np.sqrt(available / proposal_excess)) if available > 0.0 else 0.0
    accepted_kl, accepted_scale = baseline_kl, 0.0
    for _ in range(min(int(corrective_trials), 2)):
        if scale <= 0.0 or not np.isfinite(scale):
            break
        stack.interpolate_natural_(initial, proposed, scale)
        candidate_kl = combined_kl()
        if torch.isfinite(candidate_kl) and candidate_kl.item() <= max_kl * (1.0 + 1e-6):
            accepted_kl, accepted_scale = candidate_kl, scale
            break
        current_excess = max(candidate_kl.item() - baseline_kl.item(), 1e-30) if torch.isfinite(candidate_kl) else float("inf")
        correction = np.sqrt(available / current_excess) if available > 0.0 else 0.0
        scale *= min(0.99, correction)

    # An accepted trial is already committed; only a failed search needs the
    # explicit rollback recovery.
    if accepted_scale == 0.0:
        stack.interpolate_natural_(initial, proposed, 0.0)
    final_alpha, final_beta = policy_parameters()
    posterior_kl = kl_divergence(Beta(pre_alpha, pre_beta), Beta(final_alpha, final_beta)).sum(1).mean()
    return accepted_kl, posterior_kl, accepted_scale, True, proposal_kl


@torch.no_grad()
def apply_critic_posterior_with_value_limit(
    stack,
    head,
    observations,
    posterior_update,
    rms_limit,
    max_abs_limit,
    corrective_trials=2,
):
    """Limit only the value change induced by the conjugate hidden update."""
    initial = stack.snapshot_natural()

    def values():
        return head(stack.forward_states(observations)[-1]).view(-1)

    baseline = values()
    posterior_update()
    proposed = stack.snapshot_natural()

    def change_metrics():
        change = values() - baseline
        return change.square().mean().sqrt(), change.abs().max()

    proposal_rms, proposal_max = change_metrics()
    finite = torch.isfinite(proposal_rms) & torch.isfinite(proposal_max)
    if finite and proposal_rms.item() <= rms_limit and proposal_max.item() <= max_abs_limit:
        return proposal_rms, proposal_max, 1.0, False, False, proposal_rms, proposal_max

    rms_scale = rms_limit / proposal_rms.item() if torch.isfinite(proposal_rms) and proposal_rms.item() > 0.0 else 0.0
    max_scale = max_abs_limit / proposal_max.item() if torch.isfinite(proposal_max) and proposal_max.item() > 0.0 else 0.0
    scale = min(1.0, rms_scale, max_scale)
    accepted_rms = proposal_rms.new_zeros(())
    accepted_max = proposal_max.new_zeros(())
    accepted_scale = 0.0
    for _ in range(min(int(corrective_trials), 2)):
        if scale <= 0.0 or not np.isfinite(scale):
            break
        stack.interpolate_natural_(initial, proposed, scale)
        candidate_rms, candidate_max = change_metrics()
        if (
            torch.isfinite(candidate_rms)
            and torch.isfinite(candidate_max)
            and candidate_rms.item() <= rms_limit * (1.0 + 1e-6)
            and candidate_max.item() <= max_abs_limit * (1.0 + 1e-6)
        ):
            accepted_rms, accepted_max, accepted_scale = candidate_rms, candidate_max, scale
            break
        rms_correction = rms_limit / candidate_rms.item() if torch.isfinite(candidate_rms) and candidate_rms.item() > 0.0 else 0.0
        max_correction = max_abs_limit / candidate_max.item() if torch.isfinite(candidate_max) and candidate_max.item() > 0.0 else 0.0
        scale *= min(0.99, rms_correction, max_correction)

    rolled_back = accepted_scale == 0.0
    if rolled_back:
        stack.interpolate_natural_(initial, proposed, 0.0)
    return accepted_rms, accepted_max, accepted_scale, True, rolled_back, proposal_rms, proposal_max


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


def main():
    args = tyro.cli(Args)
    args.num_updates = args.total_timesteps // args.num_envs
    actor_lambda = args.trace_lambda if args.actor_trace_lambda is None else args.actor_trace_lambda
    critic_lambda = args.trace_lambda if args.critic_trace_lambda is None else args.critic_trace_lambda
    if not 0 <= actor_lambda <= 1 or not 0 <= critic_lambda <= 1:
        raise ValueError("trace lambdas must be in [0, 1]")
    if args.posterior_update_mode not in ("discounted", "paper_svi"):
        raise ValueError("posterior_update_mode must be discounted or paper_svi")
    if not 0 <= args.posterior_discount <= 1:
        raise ValueError("posterior_discount must be in [0, 1]")
    if not 0 < args.posterior_kappa_min <= args.posterior_kappa_max <= 1:
        raise ValueError("posterior kappas must satisfy 0 < min <= max <= 1")
    if args.posterior_effective_samples is not None and args.posterior_effective_samples <= 0:
        raise ValueError("posterior_effective_samples must be positive")
    if args.prior_column_covariance <= 0 or args.prior_expected_precision <= 0:
        raise ValueError("MNW prior scales must be positive")
    if args.prior_mean_mode not in ("initial", "zero"):
        raise ValueError("prior_mean_mode must be initial or zero")
    if args.prior_wishart_scale is not None and args.prior_wishart_scale <= 0:
        raise ValueError("prior_wishart_scale must be positive")
    if args.inference_steps < 1 or args.inference_lr <= 0:
        raise ValueError("latent inference requires positive steps and learning rate")
    if args.max_update_kl <= 0 or args.kl_bisection_steps < 1:
        raise ValueError("KL guard requires positive max KL and bisection steps")
    if not 1 <= args.posterior_guard_trials <= 2:
        raise ValueError("posterior_guard_trials must be one or two")
    if args.target_update_kl <= 0 or not 0 < args.kl_scale_min <= args.kl_scale_max:
        raise ValueError("KL controller target and scale interval must be positive")
    if args.posterior_jitter <= 0 or args.prior_dof_offset <= 0:
        raise ValueError("posterior jitter and prior DOF offset must be positive")
    if args.actor_terminal_coef <= 0:
        raise ValueError("actor_terminal_coef must be positive")
    if args.critic_posterior_rms_fraction <= 0 or args.critic_posterior_rms_floor <= 0 or args.critic_posterior_max_abs <= 0:
        raise ValueError("critic posterior value limits must be positive")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=run_name)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision(args.torch_float32_matmul_precision)
    assert args.cuda and torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous actions are supported"
    agent = Agent(envs, args).to(device)
    actor_layout = FlatParameterLayout(agent.actor_head)
    critic_layout = FlatParameterLayout(agent.critic_head)
    jacobians = ExactHeadJacobians(agent.actor_head, agent.critic_head, args)
    actor_trace = EligibilityTrace(args.num_envs, actor_layout.numel, device)
    critic_trace = EligibilityTrace(args.num_envs, critic_layout.numel, device)
    td_rms = RunningTDRMS(device, args.td_rms_decay, args.td_rms_min)
    kl_controller = OnlineKLController(args, device)

    global_step, posterior_step = 0, 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    for update in range(1, args.num_updates + 1):
        global_step += args.num_envs
        frac = 1.0 - (update - 1.0) / args.num_updates if args.anneal_lr else 1.0
        obs = next_obs
        with torch.no_grad():
            action, action_z, logprob, entropy, value, old_alpha, old_beta, actor_states, critic_states = agent.get_action_and_value(obs)
            # The TD(lambda) heads remain a conventional on-policy control: their
            # score/value Jacobians use the behavior features that produced this
            # action and baseline. Settled features are exclusively BPC activity.
            behavior_actor_top = actor_states[-1].clone()
            behavior_critic_top = critic_states[-1].clone()
        next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())
        bootstrap_np = bootstrap_observations(next_obs_np, truncated_np, infos)
        bootstrap = torch.as_tensor(bootstrap_np, dtype=torch.float32, device=device)
        reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
        terminated = torch.as_tensor(terminated_np, dtype=torch.bool, device=device)
        truncated = torch.as_tensor(truncated_np, dtype=torch.bool, device=device)
        done = terminated | truncated
        with torch.no_grad():
            next_critic_states = agent.critic_bpc.forward_states(bootstrap)
            next_value = agent.critic_head(next_critic_states[-1]).view(-1)
            td_target = reward + args.gamma * (~terminated).float() * next_value
            td_error = td_target - value
            actor_td = td_rms.normalize(td_error, args.td_norm_clip)
            critic_td = td_error.clamp(-args.critic_td_clip, args.critic_td_clip) if args.critic_td_clip > 0 else td_error

        actor_states, actor_settle_steps, actor_change = settle_stack(
            agent.actor_bpc,
            obs,
            actor_states,
            lambda top: actor_terminal_gradient(agent.actor_head, top, action_z, actor_td, args.actor_terminal_coef),
            args,
        )
        critic_states, critic_settle_steps, critic_change = settle_stack(
            agent.critic_bpc,
            obs,
            critic_states,
            lambda top: critic_terminal_gradient(agent.critic_head, top, td_target, args.vf_coef),
            args,
        )

        # Conventional heads use the behavior features. Their eligibility is added
        # before the current TD error; terminal traces reset only after it contributes.
        actor_instant = actor_layout.flatten_batched(jacobians.actor(behavior_actor_top, action_z)).detach()
        critic_instant = critic_layout.flatten_batched(jacobians.critic(behavior_critic_top)).detach()
        actor_trace.accumulate(actor_instant, args.gamma * actor_lambda)
        critic_trace.accumulate(critic_instant, args.gamma * critic_lambda)
        actor_direction = actor_trace.modulated_mean(actor_td)
        if args.ent_coef != 0.0:
            entropy_direction = actor_layout.flatten_batched(jacobians.entropy(behavior_actor_top)).mean(0).detach()
            actor_direction = actor_direction + args.ent_coef * entropy_direction
        actor_direction, actor_norm = clip_update_direction(actor_direction, args.max_grad_norm)
        critic_direction, critic_norm = clip_update_direction(args.vf_coef * critic_trace.modulated_mean(critic_td), args.max_grad_norm)
        head_post_kl, accepted_scale, kl_limited = apply_actor_update_with_kl_limit(
            agent.actor_head,
            actor_layout,
            behavior_actor_top,
            old_alpha,
            old_beta,
            actor_direction,
            args.learning_rate * frac,
            kl_controller.scale,
            args,
        )
        critic_layout.add_flat_(critic_direction, args.learning_rate * frac)

        # The Bayesian edge update consumes settled activity only. Delta already
        # influenced the terminal energy; multiplying these stats by delta again
        # would be nonconjugate and would create an effective signed delta^2 path.
        posterior_step += 1
        # paper_svi uses the paper's t^-0.25 schedule with an explicit online-RL
        # cap. Discounted mode ignores kappa and uses its power-prior retention.
        kappa = max(
            args.posterior_kappa_min,
            min(args.posterior_kappa_max, posterior_step ** (-args.posterior_kappa_exponent)),
        )
        post_kl, posterior_policy_kl, posterior_scale, posterior_kl_limited, posterior_proposal_kl = apply_actor_posterior_with_kl_limit(
            agent.actor_bpc,
            agent.actor_head,
            obs,
            old_alpha,
            old_beta,
            lambda: agent.actor_bpc.update_posteriors(
                obs,
                actor_states,
                args.posterior_effective_samples,
                args.posterior_update_mode,
                kappa,
                args.posterior_discount,
            ),
            args.max_update_kl,
            args.posterior_guard_trials,
        )
        critic_rms_limit = max(
            args.critic_posterior_rms_floor,
            args.critic_posterior_rms_fraction * td_rms.mean_square.sqrt().item(),
        )
        (
            critic_posterior_rms,
            critic_posterior_max,
            critic_posterior_scale,
            critic_posterior_limited,
            critic_posterior_rollback,
            critic_posterior_proposal_rms,
            critic_posterior_proposal_max,
        ) = (
            apply_critic_posterior_with_value_limit(
                agent.critic_bpc,
                agent.critic_head,
                torch.cat((obs, bootstrap), dim=0),
                lambda: agent.critic_bpc.update_posteriors(
                    obs,
                    critic_states,
                    args.posterior_effective_samples,
                    args.posterior_update_mode,
                    kappa,
                    args.posterior_discount,
                ),
                critic_rms_limit,
                args.critic_posterior_max_abs,
                args.posterior_guard_trials,
            )
        )
        kl_controller.observe(post_kl)
        actor_trace.reset(done)
        critic_trace.reset(done)
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
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/head_learning_rate", args.learning_rate * frac, global_step)
            writer.add_scalar("losses/td_error_mean", td_error.mean().item(), global_step)
            writer.add_scalar("losses/td_error_rms", td_rms.mean_square.sqrt().item(), global_step)
            writer.add_scalar("policy/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("policy/post_update_kl", post_kl.item(), global_step)
            writer.add_scalar("policy/head_post_update_kl", head_post_kl.item(), global_step)
            writer.add_scalar("policy/posterior_policy_kl", posterior_policy_kl.item(), global_step)
            writer.add_scalar("policy/posterior_proposal_kl", posterior_proposal_kl.item(), global_step)
            writer.add_scalar("policy/kl_step_scale", kl_controller.scale.item(), global_step)
            writer.add_scalar("policy/accepted_kl_step_scale", accepted_scale, global_step)
            writer.add_scalar("policy/kl_limited", float(kl_limited), global_step)
            writer.add_scalar("policy/posterior_kl_step_scale", posterior_scale, global_step)
            writer.add_scalar("policy/posterior_kl_limited", float(posterior_kl_limited), global_step)
            writer.add_scalar("critic/posterior_value_rms", critic_posterior_rms.item(), global_step)
            writer.add_scalar("critic/posterior_value_max_abs", critic_posterior_max.item(), global_step)
            writer.add_scalar("critic/posterior_proposal_value_rms", critic_posterior_proposal_rms.item(), global_step)
            writer.add_scalar("critic/posterior_proposal_value_max_abs", critic_posterior_proposal_max.item(), global_step)
            writer.add_scalar("critic/posterior_value_rms_limit", critic_rms_limit, global_step)
            writer.add_scalar("critic/posterior_step_scale", critic_posterior_scale, global_step)
            writer.add_scalar("critic/posterior_limited", float(critic_posterior_limited), global_step)
            writer.add_scalar("critic/posterior_rollback", float(critic_posterior_rollback), global_step)
            writer.add_scalar("trace/actor_rms", actor_trace.value.square().mean().sqrt().item(), global_step)
            writer.add_scalar("trace/critic_rms", critic_trace.value.square().mean().sqrt().item(), global_step)
            writer.add_scalar("update/actor_unclipped_norm", actor_norm.item(), global_step)
            writer.add_scalar("update/critic_unclipped_norm", critic_norm.item(), global_step)
            writer.add_scalar("bpc/paper_svi_kappa", kappa, global_step)
            writer.add_scalar("bpc/posterior_discount", args.posterior_discount, global_step)
            writer.add_scalar("bpc/actor_terminal_coef", args.actor_terminal_coef, global_step)
            writer.add_scalar("bpc/actor_terminal_signal_rms", (args.actor_terminal_coef * actor_td).square().mean().sqrt().item(), global_step)
            writer.add_scalar("bpc/actor_terminal_zero_fraction", (actor_td == 0.0).float().mean().item(), global_step)
            writer.add_scalar("bpc/actor_settle_steps", actor_settle_steps, global_step)
            writer.add_scalar("bpc/critic_settle_steps", critic_settle_steps, global_step)
            writer.add_scalar("bpc/actor_max_change", actor_change.item(), global_step)
            writer.add_scalar("bpc/critic_max_change", critic_change.item(), global_step)
            for index, (actor_edge, critic_edge) in enumerate(zip(agent.actor_bpc.edges, agent.critic_bpc.edges), 1):
                writer.add_scalar(f"bpc_layers/actor_{index}_precision_trace", torch.trace(actor_edge.cached_precision).item(), global_step)
                writer.add_scalar(f"bpc_layers/critic_{index}_precision_trace", torch.trace(critic_edge.cached_precision).item(), global_step)
                writer.add_scalar(f"bpc_layers/actor_{index}_V_trace", torch.trace(actor_edge.cached_V).item(), global_step)
                writer.add_scalar(f"bpc_layers/critic_{index}_V_trace", torch.trace(critic_edge.cached_V).item(), global_step)
            posterior_gain = kappa if args.posterior_update_mode == "paper_svi" else args.posterior_discount
            print(f"update={update}, global_step={global_step}, SPS={sps}, td_rms={td_rms.mean_square.sqrt().item():.3f}, posterior_gain={posterior_gain:.3f}, kl={post_kl.item():.2e}")

    if args.save_model:
        path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), path)
        print(f"model saved to {path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

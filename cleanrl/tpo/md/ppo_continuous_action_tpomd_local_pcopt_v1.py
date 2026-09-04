# TPO-MD local predictive-coding optimizer v1.
#
# The probe-scored TPO target and the HL-Gauss TD(lambda) target are the only task
# evidence.  Separate actor and critic predictive-coding chains each contain four
# affine hidden edges and one affine output edge.  Ten detached reverse
# Gauss-Seidel inference sweeps turn terminal cross-entropy forces into stopped
# local activity targets.  Every edge then performs the exact local conditional
# natural/M-step residual correction from whole-rollout float64 sufficient
# statistics; all corrections are captured before one atomic mutation.
# The TPO target deliberately retains v5's reverse-KL one-sided-cap heuristic so
# the optimizer is the isolated intervention, but its noise scale is the current
# rollout TD RMS (a sufficient statistic), never an EMA.
#
# There is no Adam, optimizer state, global task backward, BPTT, auxiliary future
# latent prediction, teacher, frozen target network, EMA, update clipping, ridge,
# decay, or learning-rate schedule.  The hidden inference block uses a shared empirical
# Gauss-Newton preconditioner, not the more expensive per-example exact block.
# PC converges to backpropagation at equilibrium; this experiment instead tests
# whether finite local inference plus exact local least-squares geometry can replace
# both backpropagation and Adam.  The sweeps are spatial PC, not temporal TD.

import os
import random
import time
from dataclasses import dataclass
from math import log
from typing import NamedTuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport


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

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 16
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_reward: bool = False
    clip_reward: bool = False

    actor_dist: str = "beta"
    logvar_min: float = -8.0
    logvar_max: float = 8.0
    hidden: int = 64
    share_backbone: bool = False
    pc_hidden_layers: int = 4
    pc_inference_steps: int = 10
    pc_chunk_size: int = 512

    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 0.5

    tpo_k: int = 8
    tpo_sigma_scale_coef: float = 1.0
    tpo_eps: float = 0.03
    tpo_adaptive_eta: bool = True
    tpo_dyn_trust: bool = True
    tpo_eta_base: float = 1.0
    tpo_eta: float = 6.0
    tpo_kl_breaker: float = 0.09

    compile: bool = True
    compile_mode: str = "reduce-overhead"

    batch_size: int = 0
    num_iterations: int = 0


def rescale(tensor: torch.Tensor, old_range: tuple[float, float], new_range: tuple[float, float]):
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (tensor - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def value_support_bounds(args: Args) -> tuple[float, float]:
    return args.v_min, args.v_max


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(
    env_id: str,
    index: int,
    capture_video: bool,
    run_name: str,
    gamma: float,
    normalize_reward: bool,
    clip_reward: bool,
):
    def thunk():
        if capture_video and index == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        clipped_space = gym.spaces.Box(
            low=np.full(env.observation_space.shape, -10.0, dtype=env.observation_space.dtype),
            high=np.full(env.observation_space.shape, 10.0, dtype=env.observation_space.dtype),
            dtype=env.observation_space.dtype,
        )
        try:
            env = gym.wrappers.TransformObservation(
                env,
                lambda observation: np.clip(observation, -10, 10),
                observation_space=clipped_space,
            )
        except TypeError:
            env = gym.wrappers.TransformObservation(env, lambda observation: np.clip(observation, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    return None


def _require_finite(label: str, *values: torch.Tensor) -> None:
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError(f"non-finite {label}; refusing the PC projection")


class PCChain(nn.Module):
    """Affine prediction edges whose hidden activities are preactivations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        *,
        output_std: float,
        zero_output: bool,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("a PC chain needs at least one hidden edge")
        edges = [layer_init(nn.Linear(input_dim, hidden_dim))]
        edges.extend(
            layer_init(nn.Linear(hidden_dim, hidden_dim))
            for _ in range(hidden_layers - 1)
        )
        edges.append(layer_init(nn.Linear(hidden_dim, output_dim), std=output_std))
        self.edges = nn.ModuleList(edges)
        if zero_output:
            with torch.no_grad():
                self.edges[-1].weight.zero_()
                self.edges[-1].bias.zero_()

    @property
    def hidden_layers(self) -> int:
        return len(self.edges) - 1

    def edge_features(self, edge_index: int, parent: torch.Tensor) -> torch.Tensor:
        # Every hidden node is a preactivation.  Its outgoing affine edge sees
        # tanh(x), including the final hidden-to-raw-output edge.
        return parent if edge_index == 0 else torch.tanh(parent)

    def forward_activities(self, observations: torch.Tensor) -> tuple[torch.Tensor, ...]:
        activities = []
        parent = observations
        for edge_index, edge in enumerate(self.edges):
            child = edge(self.edge_features(edge_index, parent))
            activities.append(child)
            parent = child
        return tuple(activities)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.forward_activities(observations)[-1]

    def predictions(
        self, observations: torch.Tensor, activities: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        predictions = []
        for edge_index, edge in enumerate(self.edges):
            parent = observations if edge_index == 0 else activities[edge_index - 1]
            predictions.append(edge(self.edge_features(edge_index, parent)))
        return tuple(predictions)


class Agent(nn.Module):
    def __init__(self, envs, args: Args):
        super().__init__()
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor_dist = args.actor_dist
        self.logvar_min = args.logvar_min
        self.logvar_max = args.logvar_max
        self.num_bins = args.num_bins
        self.actor_chain = PCChain(
            observation_dim,
            args.hidden,
            args.pc_hidden_layers,
            2 * action_dim,
            output_std=0.01,
            zero_output=False,
        )
        self.critic_chain = PCChain(
            observation_dim,
            args.hidden,
            args.pc_hidden_layers,
            args.num_bins,
            output_std=0.1,
            zero_output=True,
        )
        self.register_buffer(
            "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        )
        # This implementation uses analytic local inference and sufficient
        # statistics exclusively.  Keeping task parameters non-autograd is an
        # executable guard against accidentally restoring a global BP path.
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def actor_distribution_from_raw(self, raw: torch.Tensor):
        first, second = raw.chunk(2, dim=-1)
        if self.actor_dist == "beta":
            distribution = Beta(
                1.0 + F.softplus(first),
                1.0 + F.softplus(second),
                validate_args=False,
            )
            return (
                distribution,
                lambda latent: self.action_low + (self.action_high - self.action_low) * latent,
                lambda latent: 0.0,
            )
        if self.actor_dist == "gaussian":
            log_variance = rescale(
                (second / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            distribution = Normal(first, (0.5 * log_variance).exp(), validate_args=False)
            return (
                distribution,
                torch.tanh,
                lambda latent: 2.0 * (log(2.0) - latent - F.softplus(-2.0 * latent)),
            )
        raise ValueError(f"unknown actor_dist {self.actor_dist}")

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic_chain(observations)

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        candidate_zs: Optional[torch.Tensor] = None,
        return_dist: bool = False,
    ):
        raw = self.actor_chain(observations)
        distribution, to_action, log_det = self.actor_distribution_from_raw(raw)
        if latent is None:
            latent = distribution.sample()
            if self.actor_dist == "beta":
                latent = latent.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(latent)
        log_probability = (distribution.log_prob(latent) - log_det(latent)).sum(1)
        if self.actor_dist == "gaussian":
            entropy_latent = distribution.rsample()
            entropy = (
                distribution.log_prob(entropy_latent) - log_det(entropy_latent)
            ).sum(1).neg()
        else:
            entropy = distribution.entropy().sum(1)
        output = (action, latent, log_probability, entropy, self.critic_chain(observations))
        if candidate_zs is not None:
            output += (actor_candidate_logits(raw, candidate_zs, self),)
        if return_dist:
            output += (distribution, to_action, log_det)
        return output

    def task_parameters(self) -> list[nn.Parameter]:
        return list(self.actor_chain.parameters()) + list(self.critic_chain.parameters())


def actor_candidate_logits(
    raw: torch.Tensor, candidate_zs: torch.Tensor, agent: Agent
) -> torch.Tensor:
    distribution, _, log_det = agent.actor_distribution_from_raw(raw)
    candidates = candidate_zs.transpose(0, 1)
    return (
        distribution.log_prob(candidates) - log_det(candidates)
    ).sum(-1).transpose(0, 1)


def actor_score_jacobian(
    raw: torch.Tensor, candidate_zs: torch.Tensor, agent: Agent
) -> torch.Tensor:
    """Analytic Jacobian of each restricted candidate log score wrt raw output."""

    first, second = raw.chunk(2, dim=-1)
    if agent.actor_dist == "beta":
        alpha = 1.0 + F.softplus(first)
        beta = 1.0 + F.softplus(second)
        candidates = candidate_zs.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        shared = torch.digamma(alpha + beta).unsqueeze(1)
        alpha_jacobian = torch.sigmoid(first).unsqueeze(1) * (
            candidates.log() - torch.digamma(alpha).unsqueeze(1) + shared
        )
        beta_jacobian = torch.sigmoid(second).unsqueeze(1) * (
            torch.log1p(-candidates) - torch.digamma(beta).unsqueeze(1) + shared
        )
        return torch.cat((alpha_jacobian, beta_jacobian), dim=-1)
    mean = first
    span = agent.logvar_max - agent.logvar_min
    bounded = (second / span).tanh()
    log_variance = rescale(
        bounded,
        (-1.0, 1.0),
        (agent.logvar_min, agent.logvar_max),
    )
    variance = log_variance.exp()
    difference = candidate_zs - mean.unsqueeze(1)
    mean_jacobian = difference / variance.unsqueeze(1)
    log_variance_derivative = 0.5 * (1.0 - bounded.square())
    variance_jacobian = log_variance_derivative.unsqueeze(1) * 0.5 * (
        difference.square() / variance.unsqueeze(1) - 1.0
    )
    return torch.cat((mean_jacobian, variance_jacobian), dim=-1)


def categorical_gn_from_score_jacobian(
    probabilities: torch.Tensor, score_jacobian: torch.Tensor
) -> torch.Tensor:
    mean_jacobian = (probabilities.unsqueeze(-1) * score_jacobian).sum(dim=1, keepdim=True)
    centered = score_jacobian - mean_jacobian
    return torch.einsum(
        "bk,bki,bkj->bij", probabilities, centered, centered
    )


def actor_boundary_gradient_and_gn(
    raw: torch.Tensor,
    candidate_zs: torch.Tensor,
    targets: torch.Tensor,
    agent: Agent,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = actor_candidate_logits(raw, candidate_zs, agent)
    probabilities = torch.softmax(logits, dim=-1)
    jacobian = actor_score_jacobian(raw, candidate_zs, agent)
    gradient = torch.einsum("bk,bkd->bd", probabilities - targets, jacobian)
    return gradient, categorical_gn_from_score_jacobian(probabilities, jacobian)


def solve_identity_plus_categorical_hessian(
    probabilities: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    """O(C) Sherman-Morrison solve for (I+diag(p)-p p^T)^-1 g."""

    inverse_diagonal_gradient = gradient / (1.0 + probabilities)
    inverse_diagonal_probability = probabilities / (1.0 + probabilities)
    numerator = (probabilities * inverse_diagonal_gradient).sum(-1, keepdim=True)
    denominator = 1.0 - (probabilities * inverse_diagonal_probability).sum(-1, keepdim=True)
    return inverse_diagonal_gradient + inverse_diagonal_probability * (numerator / denominator)


def _batched_spd_solve(matrix: torch.Tensor, rhs: torch.Tensor, label: str) -> torch.Tensor:
    del label
    cholesky = torch.linalg.cholesky_ex(matrix, check_errors=False)[0]
    return torch.cholesky_solve(rhs.unsqueeze(-1), cholesky).squeeze(-1)


def chain_energy(
    chain: PCChain,
    observations: torch.Tensor,
    activities: tuple[torch.Tensor, ...],
    boundary_energy: torch.Tensor,
) -> torch.Tensor:
    predictions = chain.predictions(observations, activities)
    prediction_energy = sum(
        0.5 * (activity - prediction).square().sum()
        for activity, prediction in zip(activities, predictions, strict=True)
    )
    return prediction_energy + boundary_energy.sum()


def actor_boundary_energy(
    raw: torch.Tensor, candidate_zs: torch.Tensor, targets: torch.Tensor, agent: Agent
) -> torch.Tensor:
    return -(targets * F.log_softmax(actor_candidate_logits(raw, candidate_zs, agent), dim=-1)).sum(-1)


def critic_boundary_energy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(-1)


def hidden_shared_gn(
    next_weight: torch.Tensor,
    hidden_activity: torch.Tensor,
) -> torch.Tensor:
    derivative = 1.0 - torch.tanh(hidden_activity).square()
    gram = next_weight.T @ next_weight
    derivative_second_moment = derivative.T @ derivative / derivative.shape[0]
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return identity + gram * derivative_second_moment


class SettleResult(NamedTuple):
    activities: tuple[torch.Tensor, ...]
    energies: torch.Tensor
    stationarity_rms: torch.Tensor


def _reject_systematic_energy_growth(label: str, energies: list[torch.Tensor]) -> None:
    if len(energies) < 4:
        return
    trace = torch.stack(energies)
    recent_growth = trace[-3:] > trace[-4:-1]
    if bool(recent_growth.all()) and bool(trace[-1] > trace[0]):
        raise RuntimeError(f"{label} PC energy rose systematically; rejecting the design step")


def _chain_hidden_reverse_sweep(
    chain: PCChain,
    observations: torch.Tensor,
    activities: list[torch.Tensor],
) -> None:
    for layer in reversed(range(chain.hidden_layers)):
        predictions = chain.predictions(observations, tuple(activities))
        errors = [
            activity - prediction
            for activity, prediction in zip(activities, predictions, strict=True)
        ]
        derivative = 1.0 - torch.tanh(activities[layer]).square()
        next_weight = chain.edges[layer + 1].weight.detach()
        gradient = errors[layer] - derivative * (errors[layer + 1] @ next_weight)
        preconditioner = hidden_shared_gn(
            next_weight,
            activities[layer],
        )
        cholesky = torch.linalg.cholesky_ex(preconditioner, check_errors=False)[0]
        step = torch.cholesky_solve(gradient.T, cholesky).T
        activities[layer] = (activities[layer] - step).detach()


def _chain_stationarity(
    chain: PCChain,
    observations: torch.Tensor,
    activities: tuple[torch.Tensor, ...],
    top_gradient: torch.Tensor,
) -> torch.Tensor:
    predictions = chain.predictions(observations, activities)
    errors = tuple(
        activity - prediction
        for activity, prediction in zip(activities, predictions, strict=True)
    )
    gradients = [errors[-1] + top_gradient]
    for layer in reversed(range(chain.hidden_layers)):
        derivative = 1.0 - torch.tanh(activities[layer]).square()
        gradients.append(
            errors[layer] - derivative * (errors[layer + 1] @ chain.edges[layer + 1].weight.detach())
        )
    return torch.cat([gradient.flatten() for gradient in gradients]).square().mean().sqrt()


def _settle_actor_chain_core(
    agent: Agent,
    observations: torch.Tensor,
    candidate_zs: torch.Tensor,
    targets: torch.Tensor,
    inference_steps: int,
) -> SettleResult:
    with torch.no_grad():
        activities = list(agent.actor_chain.forward_activities(observations))
        energies = [
            chain_energy(
                agent.actor_chain,
                observations,
                tuple(activities),
                actor_boundary_energy(activities[-1], candidate_zs, targets, agent),
            )
        ]
        for _ in range(inference_steps):
            prediction_error = activities[-1] - agent.actor_chain.predictions(
                observations, tuple(activities)
            )[-1]
            task_gradient, task_gn = actor_boundary_gradient_and_gn(
                activities[-1], candidate_zs, targets, agent
            )
            identity = torch.eye(
                task_gn.shape[-1], device=task_gn.device, dtype=task_gn.dtype
            ).expand_as(task_gn)
            top_step = _batched_spd_solve(
                identity + task_gn,
                prediction_error + task_gradient,
                "actor output GGN",
            )
            activities[-1] = (activities[-1] - top_step).detach()
            _chain_hidden_reverse_sweep(agent.actor_chain, observations, activities)
            energy = chain_energy(
                agent.actor_chain,
                observations,
                tuple(activities),
                actor_boundary_energy(activities[-1], candidate_zs, targets, agent),
            )
            energies.append(energy)
        task_gradient, _ = actor_boundary_gradient_and_gn(
            activities[-1], candidate_zs, targets, agent
        )
        stationarity = _chain_stationarity(
            agent.actor_chain, observations, tuple(activities), task_gradient
        )
    return SettleResult(
        tuple(activity.detach() for activity in activities),
        torch.stack(energies).detach(),
        stationarity.detach(),
    )


def _settle_critic_chain_core(
    agent: Agent,
    observations: torch.Tensor,
    targets: torch.Tensor,
    inference_steps: int,
) -> SettleResult:
    with torch.no_grad():
        activities = list(agent.critic_chain.forward_activities(observations))
        energies = [
            chain_energy(
                agent.critic_chain,
                observations,
                tuple(activities),
                critic_boundary_energy(activities[-1], targets),
            )
        ]
        for _ in range(inference_steps):
            predictions = agent.critic_chain.predictions(observations, tuple(activities))
            probabilities = torch.softmax(activities[-1], dim=-1)
            top_gradient = activities[-1] - predictions[-1] + probabilities - targets
            top_step = solve_identity_plus_categorical_hessian(probabilities, top_gradient)
            activities[-1] = (activities[-1] - top_step).detach()
            _chain_hidden_reverse_sweep(agent.critic_chain, observations, activities)
            energy = chain_energy(
                agent.critic_chain,
                observations,
                tuple(activities),
                critic_boundary_energy(activities[-1], targets),
            )
            energies.append(energy)
        task_gradient = torch.softmax(activities[-1], dim=-1) - targets
        stationarity = _chain_stationarity(
            agent.critic_chain, observations, tuple(activities), task_gradient
        )
    return SettleResult(
        tuple(activity.detach() for activity in activities),
        torch.stack(energies).detach(),
        stationarity.detach(),
    )


def validate_settle_result(label: str, result: SettleResult) -> SettleResult:
    """One post-kernel failure check; the hot sweeps contain no host syncs."""

    _require_finite(label, *result.activities, result.energies, result.stationarity_rms)
    _reject_systematic_energy_growth(label, list(result.energies.unbind()))
    return result


def settle_actor_chain(
    agent: Agent,
    observations: torch.Tensor,
    candidate_zs: torch.Tensor,
    targets: torch.Tensor,
    inference_steps: int,
) -> SettleResult:
    return validate_settle_result(
        "actor",
        _settle_actor_chain_core(
            agent, observations, candidate_zs, targets, inference_steps
        ),
    )


def settle_critic_chain(
    agent: Agent,
    observations: torch.Tensor,
    targets: torch.Tensor,
    inference_steps: int,
) -> SettleResult:
    return validate_settle_result(
        "critic",
        _settle_critic_chain_core(agent, observations, targets, inference_steps),
    )


class EdgeStatistics(NamedTuple):
    covariance: torch.Tensor
    residual_cross: torch.Tensor
    residual_sse: torch.Tensor
    rows: int


def empty_chain_statistics(chain: PCChain, device: torch.device) -> list[EdgeStatistics]:
    statistics = []
    for edge in chain.edges:
        feature_dim = edge.in_features + 1
        statistics.append(
            EdgeStatistics(
                torch.zeros(feature_dim, feature_dim, dtype=torch.float64, device=device),
                torch.zeros(edge.out_features, feature_dim, dtype=torch.float64, device=device),
                torch.zeros((), dtype=torch.float64, device=device),
                0,
            )
        )
    return statistics


def accumulate_chain_statistics(
    chain: PCChain,
    statistics: list[EdgeStatistics],
    observations: torch.Tensor,
    settled: tuple[torch.Tensor, ...],
) -> None:
    """Accumulate stopped sufficient statistics without mutating live weights."""

    with torch.no_grad():
        predictions = chain.predictions(observations, settled)
        for edge_index, (edge, child, prediction) in enumerate(
            zip(chain.edges, settled, predictions, strict=True)
        ):
            parent = observations if edge_index == 0 else settled[edge_index - 1]
            features = chain.edge_features(edge_index, parent)
            augmented = torch.cat(
                (features, torch.ones_like(features[:, :1])), dim=-1
            ).to(torch.float64)
            residual = (child - prediction).to(torch.float64)
            previous = statistics[edge_index]
            statistics[edge_index] = EdgeStatistics(
                previous.covariance + augmented.T @ augmented,
                previous.residual_cross + residual.T @ augmented,
                previous.residual_sse + residual.square().sum(),
                previous.rows + augmented.shape[0],
            )


def chain_m_step_deltas(
    chain: PCChain, statistics: list[EdgeStatistics]
) -> dict[str, torch.Tensor]:
    """Minimum-norm residual correction; never replaces the old affine map."""

    deltas = {}
    for edge_index, (edge, stats) in enumerate(zip(chain.edges, statistics, strict=True)):
        if stats.rows == 0:
            raise ValueError("cannot project empty PC statistics")
        count = stats.covariance[-1, -1]
        feature_sum = stats.covariance[:-1, -1]
        residual_sum = stats.residual_cross[:, -1]
        centered_covariance = (
            stats.covariance[:-1, :-1]
            - feature_sum[:, None] * feature_sum[None, :] / count
        )
        centered_cross = (
            stats.residual_cross[:, :-1]
            - residual_sum[:, None] * feature_sum[None, :] / count
        )
        weight_correction = centered_cross @ torch.linalg.pinv(
            centered_covariance, hermitian=True
        )
        bias_correction = residual_sum / count - weight_correction @ (feature_sum / count)
        correction = torch.cat((weight_correction, bias_correction[:, None]), dim=-1)
        _require_finite("local M-step", correction)
        exact_next_sse = (
            stats.residual_sse
            - 2.0 * (correction * stats.residual_cross).sum()
            + (correction @ stats.covariance * correction).sum()
        )
        exact_tolerance = 1e-9 * (1.0 + float(stats.residual_sse.abs()))
        if float(exact_next_sse - stats.residual_sse) > exact_tolerance:
            raise RuntimeError("float64 local M-step increased stopped factor SSE")
        deltas[f"edges.{edge_index}.weight"] = correction[:, :-1].to(edge.weight.dtype)
        deltas[f"edges.{edge_index}.bias"] = correction[:, -1].to(edge.bias.dtype)
    return deltas


def projected_chain_sse(
    chain: PCChain,
    statistics: list[EdgeStatistics],
    deltas: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    before, after = [], []
    for edge_index, (edge, stats) in enumerate(zip(chain.edges, statistics, strict=True)):
        weight_delta = deltas[f"edges.{edge_index}.weight"].to(torch.float64)
        bias_delta = deltas[f"edges.{edge_index}.bias"].to(torch.float64)
        correction = torch.cat((weight_delta, bias_delta[:, None]), dim=-1)
        next_sse = (
            stats.residual_sse
            - 2.0 * (correction * stats.residual_cross).sum()
            + (correction @ stats.covariance * correction).sum()
        )
        tolerance = 2e-6 * (1.0 + float(stats.residual_sse.abs()))
        if float(next_sse - stats.residual_sse) > tolerance:
            raise RuntimeError("local M-step increased its stopped factor SSE")
        before.append(stats.residual_sse)
        after.append(next_sse)
    return torch.stack(before), torch.stack(after)


def covariance_diagnostics(
    statistics: list[EdgeStatistics],
) -> tuple[torch.Tensor, torch.Tensor]:
    diagnostics = [edge_covariance_diagnostics(stats) for stats in statistics]
    rank_fractions = [item[0] for item in diagnostics]
    conditions = [item[2] for item in diagnostics]
    return torch.stack(rank_fractions).min(), torch.stack(conditions).max()


def edge_covariance_diagnostics(
    stats: EdgeStatistics,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = stats.covariance[-1, -1]
    feature_sum = stats.covariance[:-1, -1]
    centered_covariance = (
        stats.covariance[:-1, :-1]
        - feature_sum[:, None] * feature_sum[None, :] / count
    )
    eigenvalues = torch.linalg.eigvalsh(centered_covariance).clamp_min(0.0)
    threshold = (
        max(centered_covariance.shape)
        * torch.finfo(centered_covariance.dtype).eps
        * eigenvalues[-1]
    )
    retained = eigenvalues > threshold
    rank_fraction = retained.to(torch.float64).mean()
    smallest = torch.where(
        retained, eigenvalues, torch.full_like(eigenvalues, torch.inf)
    ).min()
    condition = eigenvalues[-1] / smallest
    return rank_fraction, smallest, condition


def apply_atomic_chain_deltas(
    agent: Agent,
    actor_deltas: Optional[dict[str, torch.Tensor]],
    critic_deltas: dict[str, torch.Tensor],
) -> None:
    named = dict(agent.named_parameters())
    updates = {
        f"critic_chain.{name}": delta for name, delta in critic_deltas.items()
    }
    if actor_deltas is not None:
        updates.update(
            {f"actor_chain.{name}": delta for name, delta in actor_deltas.items()}
        )
    if set(updates).difference(named):
        raise KeyError("PC update contains an unknown parameter")
    _require_finite("atomic parameter update", *updates.values())
    with torch.no_grad():
        for name, delta in updates.items():
            named[name].add_(delta)


def functional_chain_forward(
    chain: PCChain,
    observations: torch.Tensor,
    deltas: dict[str, torch.Tensor],
) -> torch.Tensor:
    parameters = {
        name: parameter + deltas.get(name, torch.zeros_like(parameter))
        for name, parameter in chain.named_parameters()
    }
    return torch.func.functional_call(chain, parameters, (observations,))


def proposed_actor_kl(
    agent: Agent,
    observations: torch.Tensor,
    latent_zs: torch.Tensor,
    old_logprobs: torch.Tensor,
    deltas: dict[str, torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    estimates = []
    for obs_chunk, latent_chunk, old_chunk in zip(
        observations.split(chunk_size),
        latent_zs.split(chunk_size),
        old_logprobs.split(chunk_size),
        strict=True,
    ):
        raw = functional_chain_forward(agent.actor_chain, obs_chunk, deltas)
        distribution, _, log_det = agent.actor_distribution_from_raw(raw)
        new_logprob = (distribution.log_prob(latent_chunk) - log_det(latent_chunk)).sum(-1)
        logratio = new_logprob - old_chunk
        estimates.append((logratio.exp() - 1.0) - logratio)
    return torch.cat(estimates).mean()


def boundary_loss_diagnostics(
    agent: Agent,
    observations: torch.Tensor,
    candidate_zs: torch.Tensor,
    tpo_targets: torch.Tensor,
    value_targets: torch.Tensor,
    actor_deltas: dict[str, torch.Tensor],
    critic_deltas: dict[str, torch.Tensor],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actor_before, actor_after, critic_before, critic_after = [], [], [], []
    for start in range(0, observations.shape[0], chunk_size):
        end = start + chunk_size
        obs = observations[start:end]
        candidates = candidate_zs[start:end]
        actor_target = tpo_targets[start:end]
        critic_target = value_targets[start:end]
        old_actor_raw = agent.actor_chain(obs)
        new_actor_raw = functional_chain_forward(agent.actor_chain, obs, actor_deltas)
        old_critic_logits = agent.critic_chain(obs)
        new_critic_logits = functional_chain_forward(agent.critic_chain, obs, critic_deltas)
        actor_before.append(actor_boundary_energy(old_actor_raw, candidates, actor_target, agent))
        actor_after.append(actor_boundary_energy(new_actor_raw, candidates, actor_target, agent))
        critic_before.append(critic_boundary_energy(old_critic_logits, critic_target))
        critic_after.append(critic_boundary_energy(new_critic_logits, critic_target))
    return tuple(
        torch.cat(values).mean()
        for values in (actor_before, actor_after, critic_before, critic_after)
    )


class TpoTarget(NamedTuple):
    probabilities: torch.Tensor
    eta: float
    achieved_reverse_kl: float
    base_reverse_kl: float
    cap_engaged: float
    score_std_mean: float
    score_std_p90: float
    score_scale: float


def build_tpo_target(
    anchor_logprobs: torch.Tensor,
    scores: torch.Tensor,
    current_td_rms: torch.Tensor,
    args: Args,
) -> TpoTarget:
    """v5 reverse-KL cap target, without v5's EMA or score clamp."""

    score_scale = max(
        args.tpo_sigma_scale_coef * float(current_td_rms.detach()), 1e-6
    )
    centered_scores = (scores - scores.mean(dim=-1, keepdim=True)) / score_scale
    anchor = F.log_softmax(anchor_logprobs, dim=-1)
    old_probabilities = anchor.exp()

    def reverse_kl(temperature: float) -> float:
        log_target = F.log_softmax(anchor + centered_scores / temperature, dim=-1)
        return float(
            (old_probabilities * (anchor - log_target)).sum(-1).mean()
        )

    base_kl = reverse_kl(args.tpo_eta_base)
    cap_engaged = 0.0
    if bool(centered_scores.abs().max() < 1e-8):
        eta = args.tpo_eta_base if args.tpo_dyn_trust else 1.0
    elif args.tpo_dyn_trust:
        if base_kl <= args.tpo_eps:
            eta = args.tpo_eta_base
        else:
            cap_engaged = 1.0
            log_low, log_high = float(np.log(args.tpo_eta_base)), float(np.log(1e4))
            if reverse_kl(float(np.exp(log_high))) > args.tpo_eps:
                eta = float(np.exp(log_high))
            else:
                for _ in range(40):
                    midpoint = 0.5 * (log_low + log_high)
                    if reverse_kl(float(np.exp(midpoint))) > args.tpo_eps:
                        log_low = midpoint
                    else:
                        log_high = midpoint
                eta = float(np.exp(0.5 * (log_low + log_high)))
    elif args.tpo_adaptive_eta:
        log_low, log_high = float(np.log(1e-2)), float(np.log(1e4))
        if reverse_kl(float(np.exp(log_low))) < args.tpo_eps:
            eta = float(np.exp(log_low))
        elif reverse_kl(float(np.exp(log_high))) > args.tpo_eps:
            eta = float(np.exp(log_high))
        else:
            for _ in range(40):
                midpoint = 0.5 * (log_low + log_high)
                if reverse_kl(float(np.exp(midpoint))) > args.tpo_eps:
                    log_low = midpoint
                else:
                    log_high = midpoint
            eta = float(np.exp(0.5 * (log_low + log_high)))
    else:
        eta = args.tpo_eta
    probabilities = torch.softmax(anchor + centered_scores / eta, dim=-1).detach()
    group_std = scores.std(dim=-1, unbiased=False)
    return TpoTarget(
        probabilities,
        eta,
        reverse_kl(eta),
        base_kl,
        cap_engaged,
        float(group_std.mean()),
        float(group_std.quantile(0.9)),
        score_scale,
    )


def gae_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminations: torch.Tensor,
    boundaries: torch.Tensor,
    valids: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TD(lambda): bootstrap truncations, stop lambda across every boundary."""

    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        bootstrap = (1.0 - terminations[step]) * valids[step]
        lambda_continue = 1.0 - boundaries[step]
        delta = rewards[step] + gamma * next_values[step] * bootstrap - values[step]
        accumulator = delta + gamma * gae_lambda * lambda_continue * accumulator
        advantages[step] = accumulator
    return advantages, advantages + values


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.actor_dist not in ("beta", "gaussian"):
        raise ValueError("actor_dist must be beta or gaussian")
    if args.share_backbone:
        raise ValueError("local PC v1 uses separate actor and critic chains")
    if args.pc_hidden_layers != 4:
        raise ValueError("v1 is the fixed four-hidden-edge intervention")
    if args.pc_inference_steps != 10:
        raise ValueError("v1 is the fixed ten-sweep intervention")
    if args.pc_chunk_size <= 0:
        raise ValueError("pc_chunk_size must be positive")
    if args.batch_size % args.pc_chunk_size:
        raise ValueError("compiled v1 requires batch_size divisible by pc_chunk_size")
    if args.tpo_k < 2:
        raise ValueError("TPO requires at least two candidates")
    if args.normalize_reward or args.clip_reward:
        raise ValueError("probe scores and critic targets must remain in raw reward units")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                index,
                args.capture_video,
                run_name,
                args.gamma,
                args.normalize_reward,
                args.clip_reward,
            )
            for index in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("only continuous action spaces are supported")

    agent = Agent(envs, args).to(device)
    if any(parameter.requires_grad for parameter in agent.task_parameters()):
        raise RuntimeError("task parameters must remain outside autograd")
    actor_settle_core = lambda observation, candidates, target: _settle_actor_chain_core(
        agent,
        observation,
        candidates,
        target,
        args.pc_inference_steps,
    )
    critic_settle_core = lambda observation, target: _settle_critic_chain_core(
        agent,
        observation,
        target,
        args.pc_inference_steps,
    )
    if args.compile:
        actor_settle_core = torch.compile(
            actor_settle_core,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )
        critic_settle_core = torch.compile(
            critic_settle_core,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )

    support_min, support_max = value_support_bounds(args)
    hl_support = HLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )

    def value_logits_to_scalar(logits: torch.Tensor) -> torch.Tensor:
        return hl_support.to_expected_scalar(logits)

    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    observations = torch.zeros((args.num_steps, args.num_envs) + observation_shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape, device=device)
    latent_zs = torch.zeros_like(actions)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros_like(logprobs)
    values = torch.zeros_like(logprobs)
    transition_terminations = torch.zeros_like(logprobs)
    transition_boundaries = torch.zeros_like(logprobs)
    transition_valids = torch.ones_like(logprobs)
    next_transition_values = torch.zeros_like(logprobs)

    candidate_zs = torch.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + action_shape,
        device=device,
    )
    candidate_logprobs = torch.zeros(
        (args.num_steps, args.num_envs, args.tpo_k), device=device
    )
    candidate_next_obs = np.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + observation_shape,
        dtype=np.float32,
    )
    candidate_rewards = np.zeros(
        (args.num_steps, args.num_envs, args.tpo_k), dtype=np.float32
    )
    candidate_terminations = np.zeros_like(candidate_rewards)

    probe_base_envs = [env.unwrapped for env in envs.envs]
    probe_observation_wrappers = [
        find_wrapper(env, gym.wrappers.NormalizeObservation) for env in envs.envs
    ]
    if any(wrapper is None for wrapper in probe_observation_wrappers):
        raise RuntimeError("NormalizeObservation wrapper not found")
    probe_action_low = envs.single_action_space.low
    probe_action_high = envs.single_action_space.high
    probe_cpu_rng_state = None
    probe_cuda_rng_state = None

    global_step = 0
    start_time = time.time()
    next_observation_np, _ = envs.reset(seed=args.seed)
    next_observation = torch.as_tensor(
        next_observation_np, dtype=torch.float32, device=device
    )

    for iteration in range(1, args.num_iterations + 1):
        probe_seconds = 0.0
        for step in range(args.num_steps):
            global_step += args.num_envs
            observations[step] = next_observation
            with torch.no_grad():
                (
                    action,
                    latent,
                    log_probability,
                    _,
                    value_logits,
                    rollout_distribution,
                    rollout_to_action,
                    rollout_log_det,
                ) = agent.get_action_and_value(next_observation, return_dist=True)
                values[step] = value_logits_to_scalar(value_logits)
            actions[step] = action
            latent_zs[step] = latent
            logprobs[step] = log_probability

            probe_start = time.time()
            with torch.no_grad():
                with torch.random.fork_rng(devices=[device]):
                    if probe_cpu_rng_state is None:
                        torch.manual_seed(args.seed + 1_000_003)
                    else:
                        torch.set_rng_state(probe_cpu_rng_state)
                        torch.cuda.set_rng_state(probe_cuda_rng_state, device)
                    sampled_candidates = rollout_distribution.sample(
                        torch.Size([args.tpo_k])
                    )
                    probe_cpu_rng_state = torch.get_rng_state()
                    probe_cuda_rng_state = torch.cuda.get_rng_state(device)
                sampled_candidates = sampled_candidates.permute(1, 0, 2).contiguous()
                if args.actor_dist == "beta":
                    sampled_candidates.clamp_(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                sampled_candidates[:, 0] = latent
                transposed_candidates = sampled_candidates.transpose(0, 1)
                sampled_logprobs = (
                    rollout_distribution.log_prob(transposed_candidates)
                    - rollout_log_det(transposed_candidates)
                ).sum(-1).transpose(0, 1)
                candidate_zs[step] = sampled_candidates
                candidate_logprobs[step] = sampled_logprobs
                candidate_actions_np = rollout_to_action(sampled_candidates).cpu().numpy()
            candidate_actions_np = np.clip(
                candidate_actions_np, probe_action_low, probe_action_high
            )
            for env_index in range(args.num_envs):
                base_env = probe_base_envs[env_index]
                observation_rms = probe_observation_wrappers[env_index].obs_rms
                saved_qpos = base_env.data.qpos.copy()
                saved_qvel = base_env.data.qvel.copy()
                saved_warmstart = base_env.data.qacc_warmstart.copy()
                saved_time = base_env.data.time
                for candidate_index in range(args.tpo_k):
                    base_env.data.qpos[:] = saved_qpos
                    base_env.data.qvel[:] = saved_qvel
                    base_env.data.qacc_warmstart[:] = saved_warmstart
                    base_env.data.time = saved_time
                    (
                        probe_observation,
                        probe_reward,
                        probe_terminated,
                        _,
                        _,
                    ) = base_env.step(candidate_actions_np[env_index, candidate_index])
                    normalized_observation = (
                        (probe_observation - observation_rms.mean)
                        / np.sqrt(observation_rms.var + 1e-8)
                    ).astype(np.float32)
                    candidate_next_obs[step, env_index, candidate_index] = np.clip(
                        normalized_observation, -10.0, 10.0
                    )
                    candidate_rewards[step, env_index, candidate_index] = probe_reward
                    candidate_terminations[step, env_index, candidate_index] = float(
                        probe_terminated
                    )
                base_env.data.qpos[:] = saved_qpos
                base_env.data.qvel[:] = saved_qvel
                base_env.data.qacc_warmstart[:] = saved_warmstart
                base_env.data.time = saved_time
            probe_seconds += time.time() - probe_start

            next_observation_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            boundary = np.logical_or(terminated, truncated)
            valid = (~boundary).astype(np.float32)
            final_observations = infos.get("final_observation")
            final_mask = infos.get("_final_observation")
            if final_observations is not None:
                transition_next_observation = np.array(next_observation_np, copy=True)
                if final_mask is None:
                    final_mask = [item is not None for item in final_observations]
                for env_index, has_final in enumerate(final_mask):
                    if has_final and final_observations[env_index] is not None:
                        transition_next_observation[env_index] = final_observations[env_index]
                        valid[env_index] = 1.0
                    elif boundary[env_index]:
                        valid[env_index] = 0.0
            else:
                transition_next_observation = next_observation_np
            transition_next_tensor = torch.as_tensor(
                transition_next_observation, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                next_transition_values[step] = value_logits_to_scalar(
                    agent.get_value(transition_next_tensor)
                )
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            transition_terminations[step] = torch.as_tensor(
                terminated, dtype=torch.float32, device=device
            )
            transition_boundaries[step] = torch.as_tensor(
                boundary, dtype=torch.float32, device=device
            )
            transition_valids[step] = torch.as_tensor(
                valid, dtype=torch.float32, device=device
            )
            next_observation = torch.as_tensor(
                next_observation_np, dtype=torch.float32, device=device
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_return = info["episode"]["r"]
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar(
                            "charts/episodic_return", episode_return, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        with torch.no_grad():
            advantages, returns = gae_lambda_returns(
                rewards,
                values,
                next_transition_values,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
                args.gae_lambda,
            )
            value_targets = hl_support.project(returns).reshape(-1, args.num_bins)
            td_residual = (
                rewards
                + args.gamma
                * next_transition_values
                * (1.0 - transition_terminations)
                * transition_valids
                - values
            )
            current_td_rms = td_residual.square().mean().sqrt()

            flat_probe_observations = torch.as_tensor(
                candidate_next_obs.reshape(-1, int(np.prod(observation_shape))),
                dtype=torch.float32,
                device=device,
            )
            probe_next_values = torch.cat(
                [
                    value_logits_to_scalar(agent.get_value(chunk))
                    for chunk in flat_probe_observations.split(65_536)
                ]
            ).reshape(args.batch_size, args.tpo_k)
            probe_rewards = torch.as_tensor(
                candidate_rewards.reshape(args.batch_size, args.tpo_k),
                dtype=torch.float32,
                device=device,
            )
            probe_terminations = torch.as_tensor(
                candidate_terminations.reshape(args.batch_size, args.tpo_k),
                dtype=torch.float32,
                device=device,
            )
            scores = probe_rewards + args.gamma * (
                1.0 - probe_terminations
            ) * probe_next_values
            tpo_target = build_tpo_target(
                candidate_logprobs.reshape(args.batch_size, args.tpo_k),
                scores,
                current_td_rms,
                args,
            )

        flat_observations = observations.reshape(
            (args.batch_size,) + observation_shape
        )
        flat_candidates = candidate_zs.reshape(
            (args.batch_size, args.tpo_k) + action_shape
        )
        flat_latents = latent_zs.reshape((args.batch_size,) + action_shape)
        flat_old_logprobs = logprobs.reshape(args.batch_size)
        actor_statistics = empty_chain_statistics(agent.actor_chain, device)
        critic_statistics = empty_chain_statistics(agent.critic_chain, device)
        actor_energy_per_row = torch.zeros(args.pc_inference_steps + 1, device=device)
        critic_energy_per_row = torch.zeros_like(actor_energy_per_row)
        actor_stationarity = torch.zeros((), device=device)
        critic_stationarity = torch.zeros((), device=device)
        settle_failed = torch.zeros((), dtype=torch.bool, device=device)

        for start in range(0, args.batch_size, args.pc_chunk_size):
            end = start + args.pc_chunk_size
            chunk_observations = flat_observations[start:end]
            actor_settled = actor_settle_core(
                chunk_observations,
                flat_candidates[start:end],
                tpo_target.probabilities[start:end],
            )
            accumulate_chain_statistics(
                agent.actor_chain,
                actor_statistics,
                chunk_observations,
                actor_settled.activities,
            )
            actor_finite = torch.stack(
                [
                    *(activity.isfinite().all() for activity in actor_settled.activities),
                    actor_settled.energies.isfinite().all(),
                    actor_settled.stationarity_rms.isfinite(),
                ]
            ).all()
            actor_growth = (
                (actor_settled.energies[-3:] > actor_settled.energies[-4:-1]).all()
                & (actor_settled.energies[-1] > actor_settled.energies[0])
            )
            settle_failed |= ~actor_finite | actor_growth
            # Consume actor graph outputs before replaying the independently
            # compiled critic graph, whose CUDA pool may reuse temporary storage.
            actor_energy_per_row += actor_settled.energies / args.batch_size
            actor_stationarity += (
                (end - start) / args.batch_size * actor_settled.stationarity_rms.square()
            )

            critic_settled = critic_settle_core(
                chunk_observations,
                value_targets[start:end],
            )
            accumulate_chain_statistics(
                agent.critic_chain,
                critic_statistics,
                chunk_observations,
                critic_settled.activities,
            )
            critic_finite = torch.stack(
                [
                    *(activity.isfinite().all() for activity in critic_settled.activities),
                    critic_settled.energies.isfinite().all(),
                    critic_settled.stationarity_rms.isfinite(),
                ]
            ).all()
            critic_growth = (
                (critic_settled.energies[-3:] > critic_settled.energies[-4:-1]).all()
                & (critic_settled.energies[-1] > critic_settled.energies[0])
            )
            settle_failed |= ~critic_finite | critic_growth
            row_fraction = (end - start) / args.batch_size
            critic_energy_per_row += critic_settled.energies / args.batch_size
            critic_stationarity += row_fraction * critic_settled.stationarity_rms.square()

        if bool(settle_failed):
            raise FloatingPointError(
                "non-finite or systematically rising whole-rollout PC settlement"
            )
        actor_stationarity = actor_stationarity.sqrt()
        critic_stationarity = critic_stationarity.sqrt()

        actor_deltas = chain_m_step_deltas(agent.actor_chain, actor_statistics)
        critic_deltas = chain_m_step_deltas(agent.critic_chain, critic_statistics)
        actor_sse_before, actor_sse_after = projected_chain_sse(
            agent.actor_chain, actor_statistics, actor_deltas
        )
        critic_sse_before, critic_sse_after = projected_chain_sse(
            agent.critic_chain, critic_statistics, critic_deltas
        )
        with torch.no_grad():
            actor_proposed_kl = proposed_actor_kl(
                agent,
                flat_observations,
                flat_latents,
                flat_old_logprobs,
                actor_deltas,
                args.pc_chunk_size,
            )
            (
                actor_ce_before,
                actor_ce_after,
                critic_ce_before,
                critic_ce_after,
            ) = boundary_loss_diagnostics(
                agent,
                flat_observations,
                flat_candidates,
                tpo_target.probabilities,
                value_targets,
                actor_deltas,
                critic_deltas,
                args.pc_chunk_size,
            )
        actor_rank_fraction, actor_condition = covariance_diagnostics(actor_statistics)
        critic_rank_fraction, critic_condition = covariance_diagnostics(critic_statistics)
        apply_atomic_chain_deltas(
            agent, actor_deltas, critic_deltas
        )

        returns_numpy = returns.cpu().numpy().reshape(-1)
        values_numpy = values.cpu().numpy().reshape(-1)
        target_variance = np.var(returns_numpy)
        explained_variance = (
            np.nan
            if target_variance == 0
            else 1.0 - np.var(returns_numpy - values_numpy) / target_variance
        )
        edge_mass = float(value_targets[:, [0, -1]].sum(-1).mean())
        delta_norm = torch.stack(
            [delta.square().sum() for delta in (*actor_deltas.values(), *critic_deltas.values())]
        ).sum().sqrt()

        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("debug/returns_mean", float(returns.mean()), global_step)
        writer.add_scalar("debug/returns_std", float(returns.std()), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        writer.add_scalar("debug/current_td_rms", float(current_td_rms), global_step)
        writer.add_scalar("debug/tpo_eta_solved", tpo_target.eta, global_step)
        writer.add_scalar(
            "debug/tpo_reverse_kl_achieved",
            tpo_target.achieved_reverse_kl,
            global_step,
        )
        writer.add_scalar("debug/tpo_reverse_kl_base", tpo_target.base_reverse_kl, global_step)
        writer.add_scalar("debug/tpo_cap_engaged", tpo_target.cap_engaged, global_step)
        writer.add_scalar("debug/tpo_score_std_mean", tpo_target.score_std_mean, global_step)
        writer.add_scalar("debug/tpo_score_std_p90", tpo_target.score_std_p90, global_step)
        writer.add_scalar("debug/tpo_score_scale", tpo_target.score_scale, global_step)
        writer.add_scalar("pc/actor_energy_free", float(actor_energy_per_row[0]), global_step)
        writer.add_scalar("pc/actor_energy_settled", float(actor_energy_per_row[-1]), global_step)
        writer.add_scalar("pc/critic_energy_free", float(critic_energy_per_row[0]), global_step)
        writer.add_scalar("pc/critic_energy_settled", float(critic_energy_per_row[-1]), global_step)
        writer.add_scalar("pc/actor_stationarity_rms", float(actor_stationarity), global_step)
        writer.add_scalar("pc/critic_stationarity_rms", float(critic_stationarity), global_step)
        writer.add_scalar("pc/actor_proposed_kl", float(actor_proposed_kl), global_step)
        writer.add_scalar(
            "pc/actor_kl_breaker_exceeded",
            float(actor_proposed_kl > args.tpo_kl_breaker),
            global_step,
        )
        writer.add_scalar("pc/actor_boundary_ce_before", float(actor_ce_before), global_step)
        writer.add_scalar("pc/actor_boundary_ce_after", float(actor_ce_after), global_step)
        writer.add_scalar("pc/critic_boundary_ce_before", float(critic_ce_before), global_step)
        writer.add_scalar("pc/critic_boundary_ce_after", float(critic_ce_after), global_step)
        writer.add_scalar(
            "pc/actor_local_sse_ratio",
            float(actor_sse_after.sum() / actor_sse_before.sum().clamp_min(1e-30)),
            global_step,
        )
        writer.add_scalar(
            "pc/critic_local_sse_ratio",
            float(critic_sse_after.sum() / critic_sse_before.sum().clamp_min(1e-30)),
            global_step,
        )
        writer.add_scalar("pc/parameter_delta_norm", float(delta_norm), global_step)
        writer.add_scalar("pc/actor_min_cov_rank_fraction", float(actor_rank_fraction), global_step)
        writer.add_scalar("pc/actor_max_cov_condition", float(actor_condition), global_step)
        writer.add_scalar("pc/critic_min_cov_rank_fraction", float(critic_rank_fraction), global_step)
        writer.add_scalar("pc/critic_max_cov_condition", float(critic_condition), global_step)
        for chain_name, statistics, deltas in (
            ("actor", actor_statistics, actor_deltas),
            ("critic", critic_statistics, critic_deltas),
        ):
            for edge_index, edge_statistics in enumerate(statistics):
                rank_fraction, min_eigenvalue, condition = edge_covariance_diagnostics(
                    edge_statistics
                )
                correction_norm = (
                    deltas[f"edges.{edge_index}.weight"].square().sum()
                    + deltas[f"edges.{edge_index}.bias"].square().sum()
                ).sqrt()
                writer.add_scalar(
                    f"pc_cov/{chain_name}_edge_{edge_index}_rank_fraction",
                    float(rank_fraction),
                    global_step,
                )
                writer.add_scalar(
                    f"pc_cov/{chain_name}_edge_{edge_index}_min_positive_eigenvalue",
                    float(min_eigenvalue),
                    global_step,
                )
                writer.add_scalar(
                    f"pc_cov/{chain_name}_edge_{edge_index}_condition",
                    float(condition),
                    global_step,
                )
                writer.add_scalar(
                    f"pc_cov/{chain_name}_edge_{edge_index}_correction_norm",
                    float(correction_norm),
                    global_step,
                )
        writer.add_scalar(
            "pc/actor_entry_residual_sse_per_row",
            float(actor_statistics[0].residual_sse / actor_statistics[0].rows),
            global_step,
        )
        writer.add_scalar(
            "pc/critic_entry_residual_sse_per_row",
            float(critic_statistics[0].residual_sse / critic_statistics[0].rows),
            global_step,
        )
        writer.add_scalar(
            "pc/actor_entry_delta_norm",
            float(
                actor_deltas["edges.0.weight"].square().sum()
                + actor_deltas["edges.0.bias"].square().sum()
            ) ** 0.5,
            global_step,
        )
        writer.add_scalar(
            "pc/critic_entry_delta_norm",
            float(
                critic_deltas["edges.0.weight"].square().sum()
                + critic_deltas["edges.0.bias"].square().sum()
            ) ** 0.5,
            global_step,
        )
        for sweep in range(args.pc_inference_steps + 1):
            writer.add_scalar(
                f"pc/actor_energy_sweep_{sweep}",
                float(actor_energy_per_row[sweep]),
                global_step,
            )
            writer.add_scalar(
                f"pc/critic_energy_sweep_{sweep}",
                float(critic_energy_per_row[sweep]),
                global_step,
            )
        writer.add_scalar("charts/probe_seconds", probe_seconds, global_step)
        steps_per_second = int(global_step / (time.time() - start_time))
        print("SPS:", steps_per_second)
        writer.add_scalar("charts/SPS", steps_per_second, global_step)

    envs.close()
    writer.close()

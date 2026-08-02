# PPO + counterfactual Sparse-K rewiring v5.
#
# Layer-2 dormant gradients are scored exactly with dense input/adjoint outer
# products while production inference remains sparse. Challenger identities are
# screened on one rollout, validated as environment-clustered observations over
# four later rollouts, and swapped only when a 99.9% benefit LCB beats the live
# edge's deletion-cost UCB. New effective weights start at zero; topology changes
# occur only after PPO and are bounded by exact policy-KL/value audits. The random
# arm commits the same accepted deletions but random dormant identities, isolating
# candidate selection from conservative low-rate churn.
import math
import os
import random
import time
from dataclasses import dataclass
from statistics import NormalDist
from collections import deque
from contextlib import contextmanager
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import kl_divergence
from torch.distributions.beta import Beta
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
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

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
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # Sparse-K
    width: int = 256
    """hidden width N (units per layer)"""
    k: int = 64
    """incoming connections per unit (dense-64 fan-in match)"""
    num_hidden_layers: int = 2
    """number of sparse hidden layers"""
    pool: str = "prev"
    """prev | prior — previous-layer only vs all-prior prefix pool"""
    weight_coordinate: str = "effective"
    """effective only; raw coordinates are intentionally unsupported"""
    rewire: str = "counterfactual"
    """none | random | counterfactual"""
    rewire_challengers: int = 8
    """dormant sources frozen per perceptron after each screening rollout"""
    rewire_validation_rollouts: int = 4
    """independent rollouts used to validate each frozen challenger cohort"""
    rewire_min_edge_age: int = 4
    """complete rollouts before a live edge may be deleted"""
    rewire_confidence_z: float = 3.29
    """normal critical value for benefit LCB and deletion-cost UCB"""
    rewire_fisher_damping: float = 0.01
    """layer-relative diagonal Fisher/Gauss-Newton damping"""
    rewire_score_stride: int = 4
    """temporal subsampling within each environment trajectory"""
    rewire_max_topology_kl: float = 1e-3
    """absolute cap on mean actor topology-only KL"""
    rewire_ppo_kl_fraction: float = 0.05
    """topology KL budget as a fraction of recent PPO KL"""
    rewire_topology_kl_p99_ratio: float = 10.0
    """p99 per-state topology KL cap relative to the mean budget"""
    rewire_critic_rms_fraction: float = 0.05
    """critic output-change RMS cap relative to current return residual RMS"""
    rewire_max_swaps_per_trunk: int = 8
    """topology trust-region cap per five-rollout cohort; never a forced quota"""
    zeta: float = 0.3
    """SET: fraction of lowest-|w| edges to rewire on episode end"""
    utility_threshold: float = 0.02
    """thresh: rewire edges with EMA utility below this (absolute)"""
    utility_ema: float = 0.9
    """EMA rate for edge utility (|pre| * |w|)"""
    learned_thresh_init: float = 0.1
    """learned: initial global threshold X = softplus(logit)"""
    learned_thresh_tau: float = 0.1
    """learned: soft gate temperature; gate = sigmoid((u - X) / tau); wider ⇒ less saturation"""
    meta_q: float = 0.1
    """meta: fraction of mature edges (lowest u) to rewire on episode end"""
    meta_age_min: int = 100
    """meta: optim-steps before an edge is eligible for rewire"""
    meta_h_decay: float = 0.9
    """meta: EMA decay for grad memory h"""
    meta_u_decay: float = 0.9
    """meta: EMA decay for usefulness u"""
    compile: bool = False
    compile_mode: str = "default"

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) = y for y > 0."""
    y = max(float(y), 1e-6)
    return math.log(math.expm1(y))


class DenseLayerScores(NamedTuple):
    """Per-environment dense local gradient and positive curvature estimates."""

    gradient: torch.Tensor
    curvature: torch.Tensor


class SwapProposal(NamedTuple):
    row: int
    slot: int
    source: int
    benefit_lcb: float
    gate_benefit_lcb: float
    deletion_ucb: float
    margin: float
    fitted_weight: float


class CommittedSwap(NamedTuple):
    row: int
    slot: int
    old_source: int
    new_source: int
    old_weight: float
    benefit_lcb: float
    deletion_ucb: float
    fitted_weight: float


@contextmanager
def temporary_swaps(
    layer: "SparseKLinear", swaps: List[CommittedSwap], fitted: bool = False
):
    """Apply topology changes transactionally without touching optimizer state."""

    old_indices = layer.indices.detach().clone()
    old_weights = layer.weight.detach().clone()
    try:
        with torch.no_grad():
            for swap in swaps:
                layer.indices[swap.row, swap.slot] = swap.new_source
                layer.weight[swap.row, swap.slot] = swap.fitted_weight if fitted else 0.0
        yield
    finally:
        with torch.no_grad():
            layer.indices.copy_(old_indices)
            layer.weight.copy_(old_weights)


@contextmanager
def capture_layer_call(layer: "SparseKLinear"):
    """Capture one eager layer call without changing values, gradients, or RNG."""

    capture: Dict[str, torch.Tensor] = {}

    def hook(_module, inputs, output):
        capture["input"] = inputs[0].detach()
        capture["output"] = output

    handle = layer.register_forward_hook(hook)
    try:
        yield capture
    finally:
        handle.remove()


def _cluster_view(tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Convert time-major flattened samples to independent environment clusters."""

    return tensor.reshape(-1, num_envs, *tensor.shape[1:]).transpose(0, 1)


def actor_dense_scores(
    agent: "Agent",
    layer: "SparseKLinear",
    obs: torch.Tensor,
    zs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    env_ids: torch.Tensor,
    stride: int,
    offset: int,
    clip_coef: float,
) -> DenseLayerScores:
    """Exact clipped-PPO dormant slopes and diagonal empirical Fisher by env."""

    sample_obs = obs[offset::stride, env_ids]
    sample_zs = zs[offset::stride, env_ids]
    sample_old_logprobs = old_logprobs[offset::stride, env_ids]
    sample_advantages = advantages[offset::stride, env_ids]
    time_count, env_count = sample_obs.shape[:2]
    flat_obs = sample_obs.reshape(time_count * env_count, -1)
    flat_zs = sample_zs.reshape(time_count * env_count, -1)
    with capture_layer_call(layer) as capture:
        dist = agent._dist(flat_obs)
        logprob = dist.log_prob(flat_zs.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)).sum(1)
    ratio = (logprob - sample_old_logprobs.reshape(-1)).exp()
    flat_advantages = sample_advantages.reshape(-1)
    pg_loss = torch.maximum(
        -flat_advantages * ratio,
        -flat_advantages * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef),
    )
    score = torch.autograd.grad(logprob.sum(), capture["output"], retain_graph=True)[0]
    q = torch.autograd.grad(pg_loss.sum(), capture["output"])[0]
    omega = ratio.detach().clamp(1.0 - clip_coef, 1.0 + clip_coef)
    clustered_q = _cluster_view(q, env_count)
    clustered_score = _cluster_view(score, env_count)
    clustered_x = _cluster_view(capture["input"], env_count)
    clustered_omega = _cluster_view(omega[:, None], env_count)
    gradient = torch.einsum("eto,eti->eoi", clustered_q, clustered_x) / time_count
    curvature = torch.einsum(
        "eto,eti->eoi",
        clustered_omega * clustered_score.square(),
        clustered_x.square(),
    ) / time_count
    return DenseLayerScores(gradient.detach(), curvature.detach())


def critic_dense_scores(
    agent: "Agent",
    layer: "SparseKLinear",
    obs: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    env_ids: torch.Tensor,
    stride: int,
    offset: int,
    vf_coef: float,
    clip_coef: float,
) -> DenseLayerScores:
    """Exact clipped-value dormant slopes with diagonal Gauss-Newton curvature."""

    sample_obs = obs[offset::stride, env_ids]
    sample_returns = returns[offset::stride, env_ids]
    sample_old_values = old_values[offset::stride, env_ids]
    time_count, env_count = sample_obs.shape[:2]
    flat_obs = sample_obs.reshape(time_count * env_count, -1)
    with capture_layer_call(layer) as capture:
        value = agent.get_value(flat_obs).flatten()
    flat_returns = sample_returns.reshape(-1)
    flat_old_values = sample_old_values.reshape(-1)
    value_score = torch.autograd.grad(value.sum(), capture["output"], retain_graph=True)[0]
    unclipped = (value - flat_returns).square()
    clipped_value = flat_old_values + (value - flat_old_values).clamp(-clip_coef, clip_coef)
    clipped = (clipped_value - flat_returns).square()
    value_loss = 0.5 * vf_coef * torch.maximum(unclipped, clipped)
    q = torch.autograd.grad(value_loss.sum(), capture["output"])[0]
    use_unclipped = unclipped.detach() >= clipped.detach()
    clipped_active = (value.detach() - flat_old_values).abs() <= clip_coef
    active = (use_unclipped | clipped_active).to(value_score.dtype)
    clustered_q = _cluster_view(q, env_count)
    clustered_score = _cluster_view(value_score, env_count)
    clustered_x = _cluster_view(capture["input"], env_count)
    gradient = torch.einsum("eto,eti->eoi", clustered_q, clustered_x) / time_count
    curvature = vf_coef * torch.einsum(
        "eto,eti->eoi",
        _cluster_view(active[:, None], env_count) * clustered_score.square(),
        clustered_x.square(),
    ) / time_count
    return DenseLayerScores(gradient.detach(), curvature.detach())


class ChallengerCohort:
    """Frozen candidate identities with rollout-clustered confidence statistics."""

    def __init__(self, layer: "SparseKLinear", challengers: int):
        self.layer = layer
        self.challengers = min(int(challengers), layer.in_features - layer.k)
        self.validation_rollouts = 0
        self.cluster_count = 0
        self.candidates: Optional[torch.Tensor] = None
        self.delete_slots: Optional[torch.Tensor] = None
        self.valid_rows: Optional[torch.Tensor] = None
        self.g_sum: Optional[torch.Tensor] = None
        self.g_sq_sum: Optional[torch.Tensor] = None
        self.h_sum: Optional[torch.Tensor] = None
        self.h_sq_sum: Optional[torch.Tensor] = None
        self.d_sum: Optional[torch.Tensor] = None
        self.d_sq_sum: Optional[torch.Tensor] = None
        self.rho_sum = 0.0
        self.rho_sq_sum = 0.0
        self.last_benefit: Optional[torch.Tensor] = None
        self.last_fitted_weight: Optional[torch.Tensor] = None
        self.last_deletion_ucb: Optional[torch.Tensor] = None
        self.last_safe_gradient_fraction = 0.0
        self.last_positive_margin_fraction = 0.0
        self.last_max_benefit = 0.0
        self.last_min_deletion_ucb = 0.0

    @property
    def screening(self) -> bool:
        return self.candidates is None

    def reset(self) -> None:
        self.__init__(self.layer, self.challengers)

    @torch.no_grad()
    def screen(
        self,
        scores: DenseLayerScores,
        edge_age: torch.Tensor,
        min_edge_age: int,
        damping: float,
    ) -> None:
        mean_g = scores.gradient.mean(0)
        mean_h = scores.curvature.mean(0)
        active = torch.zeros_like(mean_g, dtype=torch.bool)
        active.scatter_(1, self.layer.indices, True)
        live_h = mean_h.gather(1, self.layer.indices)
        positive_live_h = live_h[live_h > 0]
        median_h = positive_live_h.median() if positive_live_h.numel() else mean_h.new_tensor(0.0)
        rho = torch.clamp(median_h * damping, min=1e-8)
        benefit = mean_g.square() / (2.0 * (mean_h + rho))
        benefit.masked_fill_(active, -torch.inf)
        _, self.candidates = benefit.topk(self.challengers, dim=1)

        live_g = mean_g.gather(1, self.layer.indices)
        weight = self.layer.effective_weight().detach()
        deletion = -live_g * weight + 0.5 * (live_h + rho) * weight.square()
        eligible = edge_age >= int(min_edge_age)
        deletion.masked_fill_(~eligible, torch.inf)
        _, self.delete_slots = deletion.min(dim=1)
        self.valid_rows = eligible.any(dim=1) & torch.isfinite(
            benefit.gather(1, self.candidates[:, :1]).squeeze(1)
        )
        shape = self.candidates.shape
        self.g_sum = mean_g.new_zeros(shape)
        self.g_sq_sum = mean_g.new_zeros(shape)
        self.h_sum = mean_g.new_zeros(shape)
        self.h_sq_sum = mean_g.new_zeros(shape)
        self.d_sum = mean_g.new_zeros(self.layer.out_features)
        self.d_sq_sum = mean_g.new_zeros(self.layer.out_features)

    @torch.no_grad()
    def validate(self, scores: DenseLayerScores, damping: float) -> None:
        if self.screening:
            raise RuntimeError("cannot validate before screening")
        env_count = scores.gradient.shape[0]
        candidates = self.candidates.unsqueeze(0).expand(env_count, -1, -1)
        candidate_g = scores.gradient.gather(2, candidates)
        candidate_h = scores.curvature.gather(2, candidates)

        live_h = scores.curvature.gather(
            2, self.layer.indices.unsqueeze(0).expand(env_count, -1, -1)
        )
        positive_live_h = live_h[live_h > 0]
        median_h = positive_live_h.median() if positive_live_h.numel() else live_h.new_tensor(0.0)
        rho = float(torch.clamp(median_h * damping, min=1e-8))
        slots = self.delete_slots.view(1, -1, 1).expand(env_count, -1, -1)
        live_sources = self.layer.indices.unsqueeze(0).expand(env_count, -1, -1).gather(2, slots)
        delete_g = scores.gradient.gather(2, live_sources).squeeze(2)
        delete_h = scores.curvature.gather(2, live_sources).squeeze(2)
        weight = self.layer.effective_weight().gather(1, self.delete_slots[:, None]).squeeze(1)
        deletion = -delete_g * weight + 0.5 * (delete_h + rho) * weight.square()

        self.g_sum.add_(candidate_g.sum(0))
        self.g_sq_sum.add_(candidate_g.square().sum(0))
        self.h_sum.add_(candidate_h.sum(0))
        self.h_sq_sum.add_(candidate_h.square().sum(0))
        self.d_sum.add_(deletion.sum(0))
        self.d_sq_sum.add_(deletion.square().sum(0))
        self.rho_sum += rho * env_count
        self.rho_sq_sum += rho * rho * env_count
        self.cluster_count += env_count
        self.validation_rollouts += 1

    @torch.no_grad()
    def proposals(self, confidence_z: float) -> List[SwapProposal]:
        n = self.cluster_count
        if n < 2:
            return []
        mean_g = self.g_sum / n
        g_var = (self.g_sq_sum - self.g_sum.square() / n).clamp_min(0.0) / (n - 1)
        g_se = torch.sqrt(g_var / n)
        safe_g = (mean_g.abs() - confidence_z * g_se).clamp_min(0.0)
        mean_h = self.h_sum / n
        h_var = (self.h_sq_sum - self.h_sum.square() / n).clamp_min(0.0) / (n - 1)
        h_ucb = mean_h + confidence_z * torch.sqrt(h_var / n)
        mean_rho = self.rho_sum / n
        rho_var = max((self.rho_sq_sum - self.rho_sum**2 / n) / (n - 1), 0.0)
        rho_ucb = mean_rho + confidence_z * math.sqrt(rho_var / n)
        conservative_h = h_ucb + rho_ucb
        live_radius = self.layer.effective_weight().square().mean(1, keepdim=True).sqrt().clamp_min(1e-3)
        fitted_weight = (-mean_g.sign() * safe_g / conservative_h).clamp(
            min=-live_radius, max=live_radius
        )
        benefit = safe_g * fitted_weight.abs() - 0.5 * conservative_h * fitted_weight.square()

        mean_d = self.d_sum / n
        d_var = (self.d_sq_sum - self.d_sum.square() / n).clamp_min(0.0) / (n - 1)
        deletion_ucb = mean_d + confidence_z * torch.sqrt(d_var / n)
        best_benefit, best_column = benefit.max(dim=1)
        self.last_benefit = benefit
        self.last_fitted_weight = fitted_weight
        self.last_deletion_ucb = deletion_ucb
        guarded_deletion = deletion_ucb.clamp_min(0.0)
        margin = best_benefit - guarded_deletion
        qualified = self.valid_rows & torch.isfinite(margin) & (margin > 0)
        self.last_safe_gradient_fraction = float((safe_g > 0).float().mean())
        self.last_positive_margin_fraction = float(
            (self.valid_rows & torch.isfinite(margin) & (margin > 0)).float().mean()
        )
        self.last_max_benefit = float(best_benefit.max())
        self.last_min_deletion_ucb = float(deletion_ucb.min())
        proposals = []
        for row in qualified.nonzero(as_tuple=False).flatten().tolist():
            column = int(best_column[row])
            proposals.append(
                SwapProposal(
                    row=row,
                    slot=int(self.delete_slots[row]),
                    source=int(self.candidates[row, column]),
                    benefit_lcb=float(best_benefit[row]),
                    gate_benefit_lcb=float(best_benefit[row]),
                    deletion_ucb=float(deletion_ucb[row]),
                    margin=float(margin[row]),
                    fitted_weight=float(fitted_weight[row, column]),
                )
            )
        return sorted(proposals, key=lambda proposal: proposal.margin, reverse=True)


class SparseKLinear(nn.Module):
    """Linear layer with fixed fan-in K hard connections per output unit."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        zeta: float = 0.3,
        rewire_mode: str = "none",
        utility_threshold: float = 0.02,
        utility_ema: float = 0.9,
        learned_thresh_tau: float = 0.05,
        meta_q: float = 0.1,
        meta_age_min: int = 100,
        meta_h_decay: float = 0.9,
        meta_u_decay: float = 0.9,
        weight_coordinate: str = "effective",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.k = int(min(k, in_features))
        self.zeta = float(zeta)
        self.rewire_mode = rewire_mode
        self.utility_threshold = float(utility_threshold)
        self.utility_ema = float(utility_ema)
        self.learned_thresh_tau = float(learned_thresh_tau)
        self.meta_q = float(meta_q)
        self.meta_age_min = int(meta_age_min)
        self.meta_h_decay = float(meta_h_decay)
        self.meta_u_decay = float(meta_u_decay)
        if weight_coordinate not in ("effective", "raw"):
            raise ValueError(f"weight_coordinate must be effective|raw, got {weight_coordinate}")
        self.weight_coordinate = weight_coordinate
        self.fully_dense = self.k >= self.in_features
        # Optional shared Parameter set by Agent for rewire_mode == "learned"
        self.threshold_logit: Optional[nn.Parameter] = None

        self.weight = nn.Parameter(torch.empty(self.out_features, self.k))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.register_buffer("indices", torch.zeros(self.out_features, self.k, dtype=torch.long))
        self.register_buffer(
            "destinations",
            torch.arange(self.out_features).repeat_interleave(self.k),
            persistent=False,
        )
        # thresh: start high so cold edges are not mass-rewired before EMA has signal.
        # learned: start near X so soft gate is not saturated (needs ∇X early).
        if rewire_mode == "thresh":
            init_u = max(utility_threshold * 10.0, 1.0)
        elif rewire_mode == "learned":
            init_u = float(utility_threshold)
        else:
            init_u = 0.0
        self.register_buffer("utility", torch.full((self.out_features, self.k), init_u))
        # Meta (grad-agreement) state — only used when rewire_mode == "meta"
        self.register_buffer("meta_h", torch.zeros(self.out_features, self.k))
        self.register_buffer("meta_u", torch.zeros(self.out_features, self.k))
        self.register_buffer("age", torch.zeros(self.out_features, self.k))
        self.last_rewired = 0
        self._track_utility = rewire_mode in ("thresh", "learned", "set") and not self.fully_dense

        self.reset_parameters()
        self._init_indices()

    def reset_parameters(self):
        # Draw the same effective coefficients in both arms. The raw arm stores
        # an exactly mapped parameterization without consuming additional RNG.
        std = math.sqrt(2.0 / max(self.k, 1))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.weight_coordinate == "raw":
            with torch.no_grad():
                self.weight.mul_(math.sqrt(self.k))
        nn.init.zeros_(self.bias)

    def effective_weight(self) -> torch.Tensor:
        if self.weight_coordinate == "raw":
            return self.weight / math.sqrt(self.k)
        return self.weight

    def _init_indices(self):
        with torch.no_grad():
            if self.fully_dense:
                self.indices.copy_(
                    torch.arange(self.in_features, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(self.out_features, -1)
                )
            else:
                for i in range(self.out_features):
                    self.indices[i] = torch.randperm(self.in_features)[: self.k]

    def resolve_threshold(self) -> torch.Tensor:
        """Scalar threshold tensor (may require grad if learned)."""
        if self.rewire_mode == "learned":
            if self.threshold_logit is None:
                raise RuntimeError("learned rewire requires threshold_logit on layers")
            return F.softplus(self.threshold_logit)
        return self.weight.new_tensor(self.utility_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deliberately use the packed flat-edge kernel in both arms. The only
        # difference is the stored Adam coordinate.
        sources = self.indices.reshape(-1)
        gathered = x[:, sources]
        contrib = gathered * self.weight.reshape(-1)
        if self.weight_coordinate == "raw":
            contrib = contrib / math.sqrt(self.k)
        if self.rewire_mode == "learned" and not self.fully_dense:
            # Soft keep-gate so global X = softplus(logit) receives task gradients.
            # High utility relative to X → gate→1; below X → gate→0.
            # Centered gate (0.5 at u=X) keeps early gradients alive.
            X = self.resolve_threshold()
            tau = max(self.learned_thresh_tau, 1e-4)
            gate = torch.sigmoid((self.utility.detach() - X) / tau)
            contrib = contrib.view(x.shape[0], self.out_features, self.k) * gate
            contrib = contrib.reshape(x.shape[0], -1)
        y = x.new_zeros((x.shape[0], self.out_features))
        y.index_add_(1, self.destinations, contrib)
        y = y + self.bias
        if self._track_utility and self.training:
            with torch.no_grad():
                pre = gathered.detach().abs().mean(dim=0).view(self.out_features, self.k)
                u = pre * self.effective_weight().detach().abs()
                self.utility.mul_(self.utility_ema).add_(u, alpha=1.0 - self.utility_ema)
        return y

    @torch.no_grad()
    def update_meta_from_grad(self) -> None:
        """IDBD-flavored usefulness: h←EMA(-g), u←EMA((-g)·h). Call after backward."""
        if self.rewire_mode != "meta" or self.fully_dense:
            return
        g = self.weight.grad
        if g is None:
            return
        lh = self.meta_h_decay
        lu = self.meta_u_decay
        self.meta_h.mul_(lh).add_(-g, alpha=1.0 - lh)
        s = (-g) * self.meta_h
        self.meta_u.mul_(lu).add_(s, alpha=1.0 - lu)
        self.age.add_(1.0)

    def rewire(self, optimizer: Optional[optim.Optimizer] = None) -> int:
        if self.rewire_mode == "none" or self.fully_dense:
            self.last_rewired = 0
            return 0
        if self.rewire_mode == "set":
            n = self._rewire_set(optimizer)
        elif self.rewire_mode in ("thresh", "learned"):
            n = self._rewire_thresh(optimizer)
        elif self.rewire_mode == "meta":
            n = self._rewire_meta(optimizer)
        else:
            raise ValueError(f"unknown rewire_mode={self.rewire_mode}")
        self.last_rewired = n
        return n

    def _rewire_set(self, optimizer: Optional[optim.Optimizer]) -> int:
        flat_abs = self.weight.data.abs().reshape(-1)
        n = flat_abs.numel()
        n_drop = max(1, int(self.zeta * n))
        _, flat_idx = torch.topk(flat_abs, n_drop, largest=False)
        rows = torch.div(flat_idx, self.k, rounding_mode="floor")
        cols = flat_idx % self.k
        return self._replace_slots(rows, cols, optimizer)

    def _rewire_thresh(self, optimizer: Optional[optim.Optimizer]) -> int:
        with torch.no_grad():
            X = float(self.resolve_threshold().detach())
        mask = self.utility < X
        if not bool(mask.any()):
            return 0
        rows, cols = mask.nonzero(as_tuple=True)
        return self._replace_slots(rows, cols, optimizer)

    def _rewire_meta(self, optimizer: Optional[optim.Optimizer]) -> int:
        """Among mature edges (age ≥ T), rewire bottom meta_q by usefulness u."""
        mature = self.age >= float(self.meta_age_min)
        n_mature = int(mature.sum().item())
        if n_mature <= 0:
            return 0
        n_drop = max(1, int(self.meta_q * n_mature))
        n_drop = min(n_drop, n_mature)
        # Only rank mature edges (avoids picking +inf placeholders)
        mature_idx = mature.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        mature_u = self.meta_u.reshape(-1)[mature_idx]
        _, local = torch.topk(mature_u, n_drop, largest=False)
        flat_idx = mature_idx[local]
        rows = torch.div(flat_idx, self.k, rounding_mode="floor")
        cols = flat_idx % self.k
        return self._replace_slots(rows, cols, optimizer)

    def _replace_slots(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        optimizer: Optional[optim.Optimizer],
    ) -> int:
        n = int(rows.numel())
        if n == 0:
            return 0
        device = self.weight.device
        std = math.sqrt(2.0 / max(self.k, 1))
        if self.weight_coordinate == "raw":
            std *= math.sqrt(self.k)
        # Vectorized random candidates; resolve collisions per-row with a few retries
        for attempt in range(8):
            cand = torch.randint(0, self.in_features, (n,), device=device)
            # reject candidates already present in the same row (except the dying slot)
            for i in range(n):
                r = int(rows[i])
                c = int(cols[i])
                row_idx = self.indices[r]
                # allow cand if not in row, or equals the slot being replaced
                if (row_idx == cand[i]).any() and int(row_idx[c]) != int(cand[i]):
                    # resample this one
                    for _ in range(16):
                        alt = torch.randint(0, self.in_features, (1,), device=device).item()
                        if not (row_idx == alt).any() or int(row_idx[c]) == alt:
                            cand[i] = alt
                            break
            break

        self.indices[rows, cols] = cand
        self.weight.data[rows, cols] = torch.randn(n, device=device) * std
        self.utility[rows, cols] = 0.0
        self.meta_h[rows, cols] = 0.0
        self.meta_u[rows, cols] = 0.0
        self.age[rows, cols] = 0.0

        if optimizer is not None:
            state = optimizer.state.get(self.weight)
            if state is not None:
                for key, buf in state.items():
                    if torch.is_tensor(buf) and buf.shape == self.weight.shape:
                        buf[rows, cols] = 0
        return n


class SparseTrunk(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        width: int,
        k: int,
        num_layers: int,
        any_prior: bool,
        rewire_mode: str,
        zeta: float,
        utility_threshold: float,
        utility_ema: float,
        learned_thresh_tau: float = 0.05,
        meta_q: float = 0.1,
        meta_age_min: int = 100,
        meta_h_decay: float = 0.9,
        meta_u_decay: float = 0.9,
        weight_coordinate: str = "effective",
    ):
        super().__init__()
        self.any_prior = any_prior
        self.width = width
        self.layers = nn.ModuleList()
        layer_kw = dict(
            zeta=zeta,
            rewire_mode=rewire_mode,
            utility_threshold=utility_threshold,
            utility_ema=utility_ema,
            learned_thresh_tau=learned_thresh_tau,
            meta_q=meta_q,
            meta_age_min=meta_age_min,
            meta_h_decay=meta_h_decay,
            meta_u_decay=meta_u_decay,
            weight_coordinate=weight_coordinate,
        )
        # layer 0: obs -> width
        self.layers.append(SparseKLinear(obs_dim, width, k, **layer_kw))
        for i in range(1, num_layers):
            in_dim = obs_dim + i * width if any_prior else width
            self.layers.append(SparseKLinear(in_dim, width, k, **layer_kw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts: List[torch.Tensor] = [x]
        h = x
        for i, layer in enumerate(self.layers):
            if i == 0:
                inp = x
            elif self.any_prior:
                inp = torch.cat(acts, dim=-1)
            else:
                inp = h
            h = F.silu(layer(inp))
            acts.append(h)
        return h

    def rewire_all(self, optimizer: Optional[optim.Optimizer] = None) -> int:
        total = 0
        for layer in self.layers:
            total += layer.rewire(optimizer)
        return total

    def update_meta_from_grad(self) -> None:
        for layer in self.layers:
            layer.update_meta_from_grad()

    def sparse_layers(self) -> List[SparseKLinear]:
        return list(self.layers)


class Agent(nn.Module):
    def __init__(self, envs, args: Args):
        super().__init__()
        self.args = args
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        any_prior = args.pool == "prior"
        # For learned mode, init layer cold-utility relative to init X
        util_thresh = (
            args.learned_thresh_init if args.rewire == "learned" else args.utility_threshold
        )
        common = dict(
            width=args.width,
            k=args.k,
            num_layers=args.num_hidden_layers,
            any_prior=any_prior,
            # Counterfactual rewiring is controlled only between PPO updates;
            # the legacy forward-time utility/episode rewiring path stays off.
            rewire_mode="none",
            zeta=args.zeta,
            utility_threshold=util_thresh,
            utility_ema=args.utility_ema,
            learned_thresh_tau=args.learned_thresh_tau,
            meta_q=args.meta_q,
            meta_age_min=args.meta_age_min,
            meta_h_decay=args.meta_h_decay,
            meta_u_decay=args.meta_u_decay,
            weight_coordinate=args.weight_coordinate,
        )
        self.actor_trunk = SparseTrunk(obs_dim, **common)
        self.critic_trunk = SparseTrunk(obs_dim, **common)
        # One global threshold for all sparse layers (actor + critic)
        if args.rewire == "learned":
            self.threshold_logit = nn.Parameter(
                torch.tensor(_inv_softplus(args.learned_thresh_init), dtype=torch.float32)
            )
            for layer in self.actor_trunk.sparse_layers() + self.critic_trunk.sparse_layers():
                layer.threshold_logit = self.threshold_logit
        else:
            self.register_parameter("threshold_logit", None)

        self.actor_alpha = layer_init(nn.Linear(args.width, act_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.width, act_dim), std=0.01)
        self.critic_out = layer_init(nn.Linear(args.width, 1), std=1.0)
        self.register_buffer(
            "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        )
        self._rewire_frozen = False
        self.last_rewired_total = 0

    def freeze_rewire(self, frozen: bool):
        self._rewire_frozen = frozen

    def current_threshold(self) -> Optional[torch.Tensor]:
        if self.args.rewire == "learned" and self.threshold_logit is not None:
            return F.softplus(self.threshold_logit)
        if self.args.rewire == "thresh":
            return torch.tensor(self.args.utility_threshold)
        return None

    def rewire_on_episode_end(self, optimizer: Optional[optim.Optimizer] = None) -> int:
        if self._rewire_frozen or self.args.rewire == "none":
            return 0
        n = self.actor_trunk.rewire_all(optimizer) + self.critic_trunk.rewire_all(optimizer)
        self.last_rewired_total = n
        return n

    def update_meta_from_grad(self) -> None:
        if self.args.rewire != "meta":
            return
        self.actor_trunk.update_meta_from_grad()
        self.critic_trunk.update_meta_from_grad()

    def _dist(self, x: torch.Tensor) -> Beta:
        h = self.actor_trunk(x)
        alpha = 1.0 + F.softplus(self.actor_alpha(h))
        beta = 1.0 + F.softplus(self.actor_beta(h))
        return Beta(alpha, beta)

    def _z_to_action(self, z: torch.Tensor) -> torch.Tensor:
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action: torch.Tensor) -> torch.Tensor:
        z = (action - self.action_low) / (self.action_high - self.action_low)
        return z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_out(self.critic_trunk(x))

    def get_beta_action_and_value(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        dist = self._dist(x)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        else:
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self._z_to_action(z)
        logprob = dist.log_prob(z).sum(1)
        value = self.get_value(x)
        return action, z, logprob, dist.entropy().sum(1), value

    def get_action_and_value(self, x, action=None):
        z = None if action is None else self._action_to_z(action)
        action, _, logprob, entropy, value = self.get_beta_action_and_value(x, z)
        return action, logprob, entropy, value


class CounterfactualRewirer:
    """Screen, validate, audit, and commit layer-2 source swaps."""

    def __init__(self, agent: Agent, optimizer: optim.Adam, args: Args, device: torch.device):
        if args.num_hidden_layers < 2:
            raise ValueError("counterfactual rewiring requires at least two hidden layers")
        self.agent = agent
        self.optimizer = optimizer
        self.args = args
        self.device = device
        self.actor_layer = agent.actor_trunk.layers[1]
        self.critic_layer = agent.critic_trunk.layers[1]
        if self.actor_layer.fully_dense or self.critic_layer.fully_dense:
            raise ValueError("counterfactual layer must have dormant sources")
        self.actor_cohort = ChallengerCohort(self.actor_layer, args.rewire_challengers)
        self.critic_cohort = ChallengerCohort(self.critic_layer, args.rewire_challengers)
        self.actor_age = torch.full_like(self.actor_layer.weight, args.rewire_min_edge_age, dtype=torch.long)
        self.critic_age = torch.full_like(self.critic_layer.weight, args.rewire_min_edge_age, dtype=torch.long)
        self.recent_ppo_kl = deque(maxlen=4)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(args.seed + 0x5EED)
        self.audit_generator = torch.Generator(device=device)
        self.audit_generator.manual_seed(args.seed + 0xA0D17)
        family_tests = 2 * self.actor_layer.out_features * args.rewire_challengers
        family_z = NormalDist().inv_cdf(1.0 - 0.001 / (2.0 * family_tests))
        self.confidence_z = max(args.rewire_confidence_z, family_z)
        self.actor_total = 0
        self.critic_total = 0
        self.score_offset = 0
        self.pending_actor: List[Tuple[CommittedSwap, int]] = []
        self.pending_critic: List[Tuple[CommittedSwap, int]] = []

    def _sample(self, tensor: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        sampled = tensor[self.score_offset :: self.args.rewire_score_stride, env_ids]
        return sampled.reshape(-1, *sampled.shape[2:])

    @staticmethod
    def _audit_swaps(
        layer: SparseKLinear, proposals: List[SwapProposal]
    ) -> List[CommittedSwap]:
        return [
            CommittedSwap(
                proposal.row,
                proposal.slot,
                int(layer.indices[proposal.row, proposal.slot]),
                proposal.source,
                float(layer.weight[proposal.row, proposal.slot].detach()),
                proposal.benefit_lcb,
                proposal.deletion_ucb,
                proposal.fitted_weight,
            )
            for proposal in proposals
        ]

    def _log_lifecycle(self, writer: SummaryWriter, global_step: int) -> None:
        for name, layer, pending in (
            ("actor", self.actor_layer, self.pending_actor),
            ("critic", self.critic_layer, self.pending_critic),
        ):
            advanced = []
            grouped: Dict[int, List[CommittedSwap]] = {}
            for swap, age in pending:
                age += 1
                grouped.setdefault(age, []).append(swap)
                if age < 4:
                    advanced.append((swap, age))
            for age, swaps in grouped.items():
                if age not in (1, 2, 4):
                    continue
                weights = torch.stack(
                    [layer.weight[swap.row, swap.slot].detach() for swap in swaps]
                )
                fitted = weights.new_tensor([swap.fitted_weight for swap in swaps])
                writer.add_scalar(
                    f"rewire/lifecycle_{name}_weight_abs_age_{age}",
                    float(weights.abs().mean()),
                    global_step,
                )
                writer.add_scalar(
                    f"rewire/lifecycle_{name}_fit_fraction_age_{age}",
                    float((weights.abs() / fitted.abs().clamp_min(1e-8)).mean()),
                    global_step,
                )
                writer.add_scalar(
                    f"rewire/lifecycle_{name}_sign_agreement_age_{age}",
                    float((weights.sign() == fitted.sign()).float().mean()),
                    global_step,
                )
            if name == "actor":
                self.pending_actor = advanced
            else:
                self.pending_critic = advanced

    @torch.no_grad()
    def _randomize_sources(
        self,
        layer: SparseKLinear,
        cohort: ChallengerCohort,
        proposals: List[SwapProposal],
        force: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> List[SwapProposal]:
        if self.args.rewire != "random" and not force:
            return proposals
        generator = self.generator if generator is None else generator
        randomized = []
        for proposal in proposals:
            column = int(
                torch.randint(
                    cohort.challengers, (1,), generator=generator, device=self.device
                )
            )
            benefit = float(cohort.last_benefit[proposal.row, column])
            deletion = float(cohort.last_deletion_ucb[proposal.row])
            randomized.append(
                proposal._replace(
                    source=int(cohort.candidates[proposal.row, column]),
                    benefit_lcb=benefit,
                    margin=benefit - max(deletion, 0.0),
                    fitted_weight=float(cohort.last_fitted_weight[proposal.row, column]),
                )
            )
        return randomized

    @torch.no_grad()
    def _seed_adam_coordinate(self, layer: SparseKLinear, row: int, slot: int) -> None:
        state = self.optimizer.state.get(layer.weight)
        if not state:
            return
        first = state.get("exp_avg")
        second = state.get("exp_avg_sq")
        if first is not None:
            first[row, slot] = 0.0
        if second is not None:
            other = torch.cat((second[row, :slot], second[row, slot + 1 :]))
            seed = other.median() if other.numel() else second.new_tensor(1e-16)
            second[row, slot] = seed.clamp_min(1e-16)
        maximum = state.get("max_exp_avg_sq")
        if maximum is not None:
            other = torch.cat((maximum[row, :slot], maximum[row, slot + 1 :]))
            seed = other.median() if other.numel() else maximum.new_tensor(1e-16)
            maximum[row, slot] = seed.clamp_min(1e-16)

    @torch.no_grad()
    def _actor_gate(
        self,
        proposals: List[SwapProposal],
        obs: torch.Tensor,
        zs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> Tuple[List[CommittedSwap], Dict[str, float]]:
        old_weights = self.actor_layer.weight.detach().clone()
        try:
            return self._actor_gate_impl(
                proposals, obs, zs, old_logprobs, advantages, env_ids
            )
        finally:
            self.actor_layer.weight.copy_(old_weights)

    @torch.no_grad()
    def _actor_gate_impl(
        self,
        proposals: List[SwapProposal],
        obs: torch.Tensor,
        zs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> Tuple[List[CommittedSwap], Dict[str, float]]:
        if not proposals or not self.recent_ppo_kl:
            return [], {
                "attempted": 0.0,
                "rejected_kl_mean": 0.0,
                "rejected_kl_p99": 0.0,
                "rejected_loss": 0.0,
                "kl_mean": 0.0,
                "kl_p99": 0.0,
                "loss_delta": 0.0,
            }
        flat_obs = self._sample(obs, env_ids)
        flat_zs = self._sample(zs, env_ids)
        flat_old_logprobs = self._sample(old_logprobs, env_ids).flatten()
        flat_advantages = self._sample(advantages, env_ids).flatten()
        pre_dist = self.agent._dist(flat_obs)
        pre_alpha = pre_dist.concentration1.detach().clone()
        pre_beta = pre_dist.concentration0.detach().clone()
        pre_logprob = pre_dist.log_prob(flat_zs).sum(1)
        pre_ratio = (pre_logprob - flat_old_logprobs).exp()
        pre_loss = torch.maximum(
            -flat_advantages * pre_ratio,
            -flat_advantages
            * pre_ratio.clamp(1.0 - self.args.clip_coef, 1.0 + self.args.clip_coef),
        ).mean()
        old_dist = Beta(pre_alpha, pre_beta)
        kl_budget = min(
            self.args.rewire_max_topology_kl,
            self.args.rewire_ppo_kl_fraction * float(np.median(self.recent_ppo_kl)),
        )
        if kl_budget <= 0:
            return [], {
                "attempted": float(len(proposals)),
                "rejected_kl_mean": float(len(proposals)),
                "rejected_kl_p99": 0.0,
                "rejected_loss": 0.0,
                "kl_mean": 0.0,
                "kl_p99": 0.0,
                "loss_delta": 0.0,
                "kl_budget": kl_budget,
            }

        accepted: List[CommittedSwap] = []
        benefit_budget = 0.0
        final_kl = pre_loss.new_zeros(())
        final_p99 = pre_loss.new_zeros(())
        final_delta = pre_loss.new_zeros(())
        rejected_kl_mean = 0
        rejected_kl_p99 = 0
        rejected_loss = 0
        max_kl_mean_ratio = 0.0
        max_kl_p99_ratio = 0.0
        max_loss_excess = -float("inf")
        for proposal in proposals:
            row, slot = proposal.row, proposal.slot
            old_weight = float(self.actor_layer.weight[row, slot])
            old_source = int(self.actor_layer.indices[row, slot])
            self.actor_layer.weight[row, slot] = 0.0
            new_dist = self.agent._dist(flat_obs)
            new_logprob = new_dist.log_prob(flat_zs).sum(1)
            topology_kl = kl_divergence(old_dist, new_dist).sum(1)
            new_ratio = (new_logprob - flat_old_logprobs).exp()
            new_loss = torch.maximum(
                -flat_advantages * new_ratio,
                -flat_advantages
                * new_ratio.clamp(1.0 - self.args.clip_coef, 1.0 + self.args.clip_coef),
            ).mean()
            loss_delta = new_loss - pre_loss
            candidate_budget = benefit_budget + proposal.gate_benefit_lcb
            mean_ok = float(topology_kl.mean()) <= kl_budget
            p99_ok = (
                float(torch.quantile(topology_kl, 0.99))
                <= self.args.rewire_topology_kl_p99_ratio * kl_budget
            )
            loss_ok = float(loss_delta) <= candidate_budget
            max_kl_mean_ratio = max(
                max_kl_mean_ratio, float(topology_kl.mean()) / max(kl_budget, 1e-12)
            )
            max_kl_p99_ratio = max(
                max_kl_p99_ratio,
                float(torch.quantile(topology_kl, 0.99))
                / max(self.args.rewire_topology_kl_p99_ratio * kl_budget, 1e-12),
            )
            max_loss_excess = max(max_loss_excess, float(loss_delta) - candidate_budget)
            rejected_kl_mean += int(not mean_ok)
            rejected_kl_p99 += int(not p99_ok)
            rejected_loss += int(not loss_ok)
            passes = mean_ok and p99_ok and loss_ok
            if passes:
                benefit_budget = candidate_budget
                final_kl = topology_kl.mean()
                final_p99 = torch.quantile(topology_kl, 0.99)
                final_delta = loss_delta
                accepted.append(
                    CommittedSwap(
                        row,
                        slot,
                        old_source,
                        proposal.source,
                        old_weight,
                        proposal.benefit_lcb,
                        proposal.deletion_ucb,
                        proposal.fitted_weight,
                    )
                )
            else:
                self.actor_layer.weight[row, slot] = old_weight
        metrics = {
            "attempted": float(len(proposals)),
            "rejected_kl_mean": float(rejected_kl_mean),
            "rejected_kl_p99": float(rejected_kl_p99),
            "rejected_loss": float(rejected_loss),
            "max_kl_mean_budget_ratio": max_kl_mean_ratio,
            "max_kl_p99_budget_ratio": max_kl_p99_ratio,
            "max_loss_budget_excess": max_loss_excess,
            "kl_mean": float(final_kl),
            "kl_p99": float(final_p99),
            "loss_delta": float(final_delta),
            "kl_budget": kl_budget,
        }
        for swap in accepted:
            self.actor_layer.weight[swap.row, swap.slot] = swap.old_weight
        return accepted, metrics

    @torch.no_grad()
    def _critic_gate(
        self,
        proposals: List[SwapProposal],
        obs: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> Tuple[List[CommittedSwap], Dict[str, float]]:
        old_weights = self.critic_layer.weight.detach().clone()
        try:
            return self._critic_gate_impl(proposals, obs, returns, old_values, env_ids)
        finally:
            self.critic_layer.weight.copy_(old_weights)

    @torch.no_grad()
    def _critic_gate_impl(
        self,
        proposals: List[SwapProposal],
        obs: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> Tuple[List[CommittedSwap], Dict[str, float]]:
        if not proposals:
            return [], {
                "attempted": 0.0,
                "rejected_rms": 0.0,
                "rejected_loss": 0.0,
                "rms": 0.0,
                "loss_delta": 0.0,
            }
        flat_obs = self._sample(obs, env_ids)
        flat_returns = self._sample(returns, env_ids).flatten()
        flat_old_values = self._sample(old_values, env_ids).flatten()
        pre_value = self.agent.get_value(flat_obs).flatten()
        pre_unclipped = (pre_value - flat_returns).square()
        pre_clipped_value = flat_old_values + (pre_value - flat_old_values).clamp(
            -self.args.clip_coef, self.args.clip_coef
        )
        pre_clipped = (pre_clipped_value - flat_returns).square()
        pre_loss = 0.5 * self.args.vf_coef * torch.maximum(pre_unclipped, pre_clipped).mean()
        residual_rms = torch.sqrt((pre_value - flat_returns).square().mean())
        accepted: List[CommittedSwap] = []
        benefit_budget = 0.0
        final_rms = pre_loss.new_zeros(())
        final_delta = pre_loss.new_zeros(())
        rejected_rms = 0
        rejected_loss = 0
        max_rms_budget_ratio = 0.0
        max_loss_excess = -float("inf")
        for proposal in proposals:
            row, slot = proposal.row, proposal.slot
            old_weight = float(self.critic_layer.weight[row, slot])
            old_source = int(self.critic_layer.indices[row, slot])
            self.critic_layer.weight[row, slot] = 0.0
            new_value = self.agent.get_value(flat_obs).flatten()
            new_unclipped = (new_value - flat_returns).square()
            new_clipped_value = flat_old_values + (new_value - flat_old_values).clamp(
                -self.args.clip_coef, self.args.clip_coef
            )
            new_clipped = (new_clipped_value - flat_returns).square()
            new_loss = 0.5 * self.args.vf_coef * torch.maximum(
                new_unclipped, new_clipped
            ).mean()
            loss_delta = new_loss - pre_loss
            value_rms = torch.sqrt((new_value - pre_value).square().mean())
            candidate_budget = benefit_budget + proposal.gate_benefit_lcb
            loss_ok = float(loss_delta) <= candidate_budget
            rms_ok = float(value_rms) <= self.args.rewire_critic_rms_fraction * max(
                float(residual_rms), 1e-8
            )
            rms_budget = self.args.rewire_critic_rms_fraction * max(
                float(residual_rms), 1e-8
            )
            max_rms_budget_ratio = max(
                max_rms_budget_ratio, float(value_rms) / max(rms_budget, 1e-12)
            )
            max_loss_excess = max(max_loss_excess, float(loss_delta) - candidate_budget)
            rejected_loss += int(not loss_ok)
            rejected_rms += int(not rms_ok)
            passes = loss_ok and rms_ok
            if passes:
                benefit_budget = candidate_budget
                final_rms = value_rms
                final_delta = loss_delta
                accepted.append(
                    CommittedSwap(
                        row,
                        slot,
                        old_source,
                        proposal.source,
                        old_weight,
                        proposal.benefit_lcb,
                        proposal.deletion_ucb,
                        proposal.fitted_weight,
                    )
                )
            else:
                self.critic_layer.weight[row, slot] = old_weight
        metrics = {
            "attempted": float(len(proposals)),
            "rejected_rms": float(rejected_rms),
            "rejected_loss": float(rejected_loss),
            "max_rms_budget_ratio": max_rms_budget_ratio,
            "max_loss_budget_excess": max_loss_excess,
            "rms": float(final_rms),
            "loss_delta": float(final_delta),
        }
        for swap in accepted:
            self.critic_layer.weight[swap.row, swap.slot] = swap.old_weight
        return accepted, metrics

    @torch.no_grad()
    def _commit(
        self,
        layer: SparseKLinear,
        edge_age: torch.Tensor,
        accepted: List[CommittedSwap],
    ) -> None:
        for swap in accepted:
            layer.indices[swap.row, swap.slot] = swap.new_source
            layer.weight[swap.row, swap.slot] = 0.0
            edge_age[swap.row, swap.slot] = 0
            self._seed_adam_coordinate(layer, swap.row, swap.slot)

    @torch.no_grad()
    def _heldout_audit(
        self,
        actor_swaps: List[CommittedSwap],
        critic_swaps: List[CommittedSwap],
        obs: torch.Tensor,
        zs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        flat_obs = self._sample(obs, env_ids)
        if actor_swaps:
            flat_zs = self._sample(zs, env_ids)
            flat_old_logprobs = self._sample(old_logprobs, env_ids).flatten()
            flat_advantages = self._sample(advantages, env_ids).flatten()
            shadow = self.agent._dist(flat_obs)
            shadow_logprob = shadow.log_prob(flat_zs).sum(1)
            shadow_loss = torch.maximum(
                -flat_advantages * (shadow_logprob - flat_old_logprobs).exp(),
                -flat_advantages
                * (shadow_logprob - flat_old_logprobs)
                .exp()
                .clamp(1.0 - self.args.clip_coef, 1.0 + self.args.clip_coef),
            ).mean()
            with temporary_swaps(self.actor_layer, actor_swaps):
                deployed = self.agent._dist(flat_obs)
                deployed_logprob = deployed.log_prob(flat_zs).sum(1)
                deployed_loss = torch.maximum(
                    -flat_advantages * (deployed_logprob - flat_old_logprobs).exp(),
                    -flat_advantages
                    * (deployed_logprob - flat_old_logprobs)
                    .exp()
                    .clamp(1.0 - self.args.clip_coef, 1.0 + self.args.clip_coef),
                ).mean()
                topology_kl = kl_divergence(shadow, deployed).sum(1)
            metrics["actor_loss_delta"] = float(deployed_loss - shadow_loss)
            metrics["actor_kl_mean"] = float(topology_kl.mean())
            metrics["actor_kl_p99"] = float(torch.quantile(topology_kl, 0.99))
            with temporary_swaps(self.actor_layer, actor_swaps, fitted=True):
                fitted_dist = self.agent._dist(flat_obs)
                fitted_logprob = fitted_dist.log_prob(flat_zs).sum(1)
                fitted_ratio = (fitted_logprob - flat_old_logprobs).exp()
                fitted_loss = torch.maximum(
                    -flat_advantages * fitted_ratio,
                    -flat_advantages
                    * fitted_ratio.clamp(1.0 - self.args.clip_coef, 1.0 + self.args.clip_coef),
                ).mean()
            metrics["actor_fitted_improvement"] = float(shadow_loss - fitted_loss)
            metrics["actor_fitted_addition_improvement"] = float(deployed_loss - fitted_loss)
            metrics["actor_predicted_benefit"] = float(
                sum(swap.benefit_lcb for swap in actor_swaps)
            )
            metrics["actor_predicted_net_margin"] = float(
                sum(
                    swap.benefit_lcb - max(swap.deletion_ucb, 0.0)
                    for swap in actor_swaps
                )
            )
            actor_realized = []
            for swap in actor_swaps:
                with temporary_swaps(self.actor_layer, [swap], fitted=True):
                    fitted_dist = self.agent._dist(flat_obs)
                    fitted_logprob = fitted_dist.log_prob(flat_zs).sum(1)
                    fitted_ratio = (fitted_logprob - flat_old_logprobs).exp()
                    isolated_loss = torch.maximum(
                        -flat_advantages * fitted_ratio,
                        -flat_advantages
                        * fitted_ratio.clamp(
                            1.0 - self.args.clip_coef, 1.0 + self.args.clip_coef
                        ),
                    ).mean()
                actor_realized.append(float(shadow_loss - isolated_loss))
            metrics.update(self._calibration_metrics("actor", actor_swaps, actor_realized))
            metrics["actor_interaction_residual"] = float(
                (shadow_loss - fitted_loss) - sum(actor_realized)
            )
        if critic_swaps:
            flat_returns = self._sample(returns, env_ids).flatten()
            flat_old_values = self._sample(old_values, env_ids).flatten()
            shadow_value = self.agent.get_value(flat_obs).flatten()
            shadow_unclipped = (shadow_value - flat_returns).square()
            shadow_clipped_value = flat_old_values + (shadow_value - flat_old_values).clamp(
                -self.args.clip_coef, self.args.clip_coef
            )
            shadow_loss = 0.5 * self.args.vf_coef * torch.maximum(
                shadow_unclipped, (shadow_clipped_value - flat_returns).square()
            ).mean()
            with temporary_swaps(self.critic_layer, critic_swaps):
                deployed_value = self.agent.get_value(flat_obs).flatten()
                deployed_unclipped = (deployed_value - flat_returns).square()
                deployed_clipped_value = flat_old_values + (
                    deployed_value - flat_old_values
                ).clamp(-self.args.clip_coef, self.args.clip_coef)
                deployed_loss = 0.5 * self.args.vf_coef * torch.maximum(
                    deployed_unclipped, (deployed_clipped_value - flat_returns).square()
                ).mean()
            metrics["critic_loss_delta"] = float(deployed_loss - shadow_loss)
            metrics["critic_value_rms"] = float(
                torch.sqrt((deployed_value - shadow_value).square().mean())
            )
            with temporary_swaps(self.critic_layer, critic_swaps, fitted=True):
                fitted_value = self.agent.get_value(flat_obs).flatten()
                fitted_unclipped = (fitted_value - flat_returns).square()
                fitted_clipped_value = flat_old_values + (fitted_value - flat_old_values).clamp(
                    -self.args.clip_coef, self.args.clip_coef
                )
                fitted_loss = 0.5 * self.args.vf_coef * torch.maximum(
                    fitted_unclipped, (fitted_clipped_value - flat_returns).square()
                ).mean()
            metrics["critic_fitted_improvement"] = float(shadow_loss - fitted_loss)
            metrics["critic_fitted_addition_improvement"] = float(deployed_loss - fitted_loss)
            metrics["critic_predicted_benefit"] = float(
                sum(swap.benefit_lcb for swap in critic_swaps)
            )
            metrics["critic_predicted_net_margin"] = float(
                sum(
                    swap.benefit_lcb - max(swap.deletion_ucb, 0.0)
                    for swap in critic_swaps
                )
            )
            critic_realized = []
            for swap in critic_swaps:
                with temporary_swaps(self.critic_layer, [swap], fitted=True):
                    fitted_value = self.agent.get_value(flat_obs).flatten()
                    fitted_unclipped = (fitted_value - flat_returns).square()
                    fitted_clipped_value = flat_old_values + (
                        fitted_value - flat_old_values
                    ).clamp(-self.args.clip_coef, self.args.clip_coef)
                    isolated_loss = 0.5 * self.args.vf_coef * torch.maximum(
                        fitted_unclipped, (fitted_clipped_value - flat_returns).square()
                    ).mean()
                critic_realized.append(float(shadow_loss - isolated_loss))
            metrics.update(self._calibration_metrics("critic", critic_swaps, critic_realized))
            metrics["critic_interaction_residual"] = float(
                (shadow_loss - fitted_loss) - sum(critic_realized)
            )
        return metrics

    @staticmethod
    def _calibration_metrics(
        prefix: str, swaps: List[CommittedSwap], realized: List[float]
    ) -> Dict[str, float]:
        predicted = np.asarray(
            [swap.benefit_lcb - max(swap.deletion_ucb, 0.0) for swap in swaps],
            dtype=np.float64,
        )
        actual = np.asarray(realized, dtype=np.float64)
        finite = np.isfinite(predicted) & np.isfinite(actual)
        metrics = {
            f"{prefix}_fitted_positive_rate": (
                float(np.mean(actual[finite] > 0)) if finite.any() else float("nan")
            )
        }
        if finite.any():
            metrics[f"{prefix}_fitted_sign_agreement"] = float(
                np.mean(np.sign(predicted[finite]) == np.sign(actual[finite]))
            )
            predicted_positive = finite & (predicted > 0)
            metrics[f"{prefix}_fitted_positive_precision"] = (
                float(np.mean(actual[predicted_positive] > 0))
                if predicted_positive.any()
                else float("nan")
            )
        if finite.sum() >= 2 and np.std(predicted[finite]) > 0 and np.std(actual[finite]) > 0:
            predicted_rank = np.argsort(np.argsort(predicted[finite]))
            actual_rank = np.argsort(np.argsort(actual[finite]))
            metrics[f"{prefix}_fitted_rank_correlation"] = float(
                np.corrcoef(predicted_rank, actual_rank)[0, 1]
            )
            slope, intercept = np.polyfit(predicted[finite], actual[finite], 1)
            metrics[f"{prefix}_fitted_calibration_slope"] = float(slope)
            metrics[f"{prefix}_fitted_calibration_intercept"] = float(intercept)
        metrics[f"{prefix}_fitted_calibration_n"] = float(finite.sum())
        return metrics

    def step(
        self,
        obs: torch.Tensor,
        zs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        approx_kl: float,
        writer: SummaryWriter,
        global_step: int,
    ) -> None:
        self._log_lifecycle(writer, global_step)
        self.actor_age.add_(1)
        self.critic_age.add_(1)
        self.score_offset = (global_step // self.args.batch_size) % self.args.rewire_score_stride
        selection_end = self.args.num_envs // 2
        gate_end = 3 * self.args.num_envs // 4
        selection_envs = torch.arange(0, selection_end, device=self.device)
        gate_envs = torch.arange(selection_end, gate_end, device=self.device)
        audit_envs = torch.arange(gate_end, self.args.num_envs, device=self.device)
        with torch.no_grad():
            selection_obs = self._sample(obs, selection_envs)
            selection_zs = self._sample(zs, selection_envs)
            selection_old_logprobs = self._sample(old_logprobs, selection_envs).flatten()
            selection_logprobs = self.agent._dist(selection_obs).log_prob(selection_zs).sum(1)
            selection_logratio = selection_logprobs - selection_old_logprobs
            selection_ratio = selection_logratio.exp()
            selection_ppo_kl = ((selection_ratio - 1.0) - selection_logratio).mean()
        self.recent_ppo_kl.append(float(selection_ppo_kl))
        writer.add_scalar("rewire/selection_ppo_kl", float(selection_ppo_kl), global_step)
        writer.add_scalar("rewire/training_iteration_ppo_kl", approx_kl, global_step)
        selection_advantages = advantages[:, selection_envs]
        normalized_advantages = (advantages - selection_advantages.mean()) / (
            selection_advantages.std() + 1e-8
        )
        actor_scores = actor_dense_scores(
            self.agent,
            self.actor_layer,
            obs,
            zs,
            old_logprobs,
            normalized_advantages,
            selection_envs,
            self.args.rewire_score_stride,
            self.score_offset,
            self.args.clip_coef,
        )
        critic_scores = critic_dense_scores(
            self.agent,
            self.critic_layer,
            obs,
            returns,
            old_values,
            selection_envs,
            self.args.rewire_score_stride,
            self.score_offset,
            self.args.vf_coef,
            self.args.clip_coef,
        )
        if self.actor_cohort.screening:
            self.actor_cohort.screen(
                actor_scores,
                self.actor_age,
                self.args.rewire_min_edge_age,
                self.args.rewire_fisher_damping,
            )
            self.critic_cohort.screen(
                critic_scores,
                self.critic_age,
                self.args.rewire_min_edge_age,
                self.args.rewire_fisher_damping,
            )
            writer.add_scalar("rewire/cohort_phase", 0, global_step)
            return

        self.actor_cohort.validate(actor_scores, self.args.rewire_fisher_damping)
        self.critic_cohort.validate(critic_scores, self.args.rewire_fisher_damping)
        writer.add_scalar("rewire/cohort_phase", 1, global_step)
        writer.add_scalar(
            "rewire/validation_rollouts", self.actor_cohort.validation_rollouts, global_step
        )
        if self.actor_cohort.validation_rollouts < self.args.rewire_validation_rollouts:
            return

        actor_qualified = self.actor_cohort.proposals(self.confidence_z)
        critic_qualified = self.critic_cohort.proposals(self.confidence_z)
        actor_selected = actor_qualified[: self.args.rewire_max_swaps_per_trunk]
        critic_selected = critic_qualified[: self.args.rewire_max_swaps_per_trunk]
        actor_proposals = self._randomize_sources(
            self.actor_layer,
            self.actor_cohort,
            actor_selected,
        )
        critic_proposals = self._randomize_sources(
            self.critic_layer,
            self.critic_cohort,
            critic_selected,
        )
        actor_random_audit = (
            actor_proposals
            if self.args.rewire == "random"
            else self._randomize_sources(
                self.actor_layer,
                self.actor_cohort,
                actor_selected,
                force=True,
                generator=self.audit_generator,
            )
        )
        critic_random_audit = (
            critic_proposals
            if self.args.rewire == "random"
            else self._randomize_sources(
                self.critic_layer,
                self.critic_cohort,
                critic_selected,
                force=True,
                generator=self.audit_generator,
            )
        )
        actor_swaps, actor_gate = self._actor_gate(
            actor_proposals, obs, zs, old_logprobs, normalized_advantages, gate_envs
        )
        critic_swaps, critic_gate = self._critic_gate(
            critic_proposals, obs, returns, old_values, gate_envs
        )
        best_audit = self._heldout_audit(
            self._audit_swaps(self.actor_layer, actor_selected),
            self._audit_swaps(self.critic_layer, critic_selected),
            obs,
            zs,
            old_logprobs,
            normalized_advantages,
            returns,
            old_values,
            audit_envs,
        )
        random_audit = self._heldout_audit(
            self._audit_swaps(self.actor_layer, actor_random_audit),
            self._audit_swaps(self.critic_layer, critic_random_audit),
            obs,
            zs,
            old_logprobs,
            normalized_advantages,
            returns,
            old_values,
            audit_envs,
        )
        self._commit(self.actor_layer, self.actor_age, actor_swaps)
        self._commit(self.critic_layer, self.critic_age, critic_swaps)
        self.pending_actor.extend((swap, 0) for swap in actor_swaps)
        self.pending_critic.extend((swap, 0) for swap in critic_swaps)
        self.actor_total += len(actor_swaps)
        self.critic_total += len(critic_swaps)

        writer.add_scalar("rewire/actor_proposed", len(actor_proposals), global_step)
        writer.add_scalar("rewire/actor_qualified_precap", len(actor_qualified), global_step)
        writer.add_scalar("rewire/critic_qualified_precap", len(critic_qualified), global_step)
        for name, cohort in (
            ("actor", self.actor_cohort),
            ("critic", self.critic_cohort),
        ):
            writer.add_scalar(
                f"rewire/{name}_safe_gradient_fraction",
                cohort.last_safe_gradient_fraction,
                global_step,
            )
            writer.add_scalar(
                f"rewire/{name}_positive_margin_row_fraction",
                cohort.last_positive_margin_fraction,
                global_step,
            )
            writer.add_scalar(
                f"rewire/{name}_max_benefit_lcb", cohort.last_max_benefit, global_step
            )
            writer.add_scalar(
                f"rewire/{name}_min_deletion_ucb",
                cohort.last_min_deletion_ucb,
                global_step,
            )
        writer.add_scalar("rewire/familywise_confidence_z", self.confidence_z, global_step)
        writer.add_scalar("rewire/critic_proposed", len(critic_proposals), global_step)
        writer.add_scalar("rewire/actor_accepted", len(actor_swaps), global_step)
        writer.add_scalar("rewire/critic_accepted", len(critic_swaps), global_step)
        writer.add_scalar("rewire/actor_cumulative", self.actor_total, global_step)
        writer.add_scalar("rewire/critic_cumulative", self.critic_total, global_step)
        for name, value in actor_gate.items():
            writer.add_scalar(f"rewire/actor_gate_{name}", value, global_step)
        for name, value in critic_gate.items():
            writer.add_scalar(f"rewire/critic_gate_{name}", value, global_step)
        for name, value in best_audit.items():
            writer.add_scalar(f"rewire/audit_best_{name}", value, global_step)
        for name, value in random_audit.items():
            writer.add_scalar(f"rewire/audit_random_{name}", value, global_step)
        for trunk in ("actor", "critic"):
            key = f"{trunk}_fitted_improvement"
            if key in best_audit and key in random_audit:
                writer.add_scalar(
                    f"rewire/audit_{trunk}_selection_uplift",
                    best_audit[key] - random_audit[key],
                    global_step,
                )
        if actor_proposals:
            writer.add_scalar(
                "rewire/actor_margin_mean",
                float(np.mean([proposal.margin for proposal in actor_proposals])),
                global_step,
            )
        if critic_proposals:
            writer.add_scalar(
                "rewire/critic_margin_mean",
                float(np.mean([proposal.margin for proposal in critic_proposals])),
                global_step,
            )
        self.actor_cohort.reset()
        self.critic_cohort.reset()


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.pool in ("prev", "prior"), f"pool must be prev|prior, got {args.pool}"
    if args.rewire not in ("none", "random", "counterfactual"):
        raise ValueError("rewire must be none|random|counterfactual")
    if args.weight_coordinate != "effective":
        raise ValueError("counterfactual rewiring requires effective weight coordinates")
    if args.num_envs < 4 or args.num_envs % 4:
        raise ValueError("counterfactual scoring requires num_envs divisible by four")
    if args.rewire_score_stride < 1:
        raise ValueError("rewire_score_stride must be positive")
    if args.rewire_score_stride > args.num_steps:
        raise ValueError("rewire_score_stride cannot exceed num_steps")
    if args.rewire_validation_rollouts < 1:
        raise ValueError("rewire_validation_rollouts must be positive")
    if args.rewire_challengers < 1:
        raise ValueError("rewire_challengers must be positive")
    if args.rewire_max_swaps_per_trunk < 1:
        raise ValueError("rewire_max_swaps_per_trunk must be positive")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("packed SPK experiments require CUDA")

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.compile:
        agent = torch.compile(agent, mode=args.compile_mode)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    episode_rewire_events = 0
    episode_rewire_edges = 0

    # unwrap compile for rewire helpers
    def raw_agent():
        return agent._orig_mod if hasattr(agent, "_orig_mod") else agent

    rewirer = (
        CounterfactualRewirer(raw_agent(), optimizer, args, device)
        if args.rewire != "none"
        else None
    )

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        raw_agent().freeze_rewire(False)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, _, value = agent.get_beta_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            zs[step] = z
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done_np.astype(np.float32)).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Freeze connectivity during PPO update
        raw_agent().freeze_rewire(True)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        named_sparse_layers = [
            (f"actor/layer_{index}", layer)
            for index, layer in enumerate(raw_agent().actor_trunk.sparse_layers())
        ] + [
            (f"critic/layer_{index}", layer)
            for index, layer in enumerate(raw_agent().critic_trunk.sparse_layers())
        ]
        preclip_grad_norm_sum = torch.zeros((), device=device)
        clip_coefficient_sum = torch.zeros((), device=device)
        clipped_steps = torch.zeros((), device=device)
        sampled_layer_grad_norms = {}
        sampled_layer_effective_grad_norms = {}
        sampled_layer_updates = {}
        optimizer_steps = 0
        iteration_approx_kl_sum = torch.zeros((), device=device)
        iteration_approx_kl_count = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue = agent.get_beta_action_and_value(
                    b_obs[mb_inds], b_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    iteration_approx_kl_sum += approx_kl
                    iteration_approx_kl_count += 1
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                sample_diagnostics = optimizer_steps == 0
                if sample_diagnostics:
                    effective_before = {
                        name: layer.effective_weight().detach().clone()
                        for name, layer in named_sparse_layers
                    }
                    for name, layer in named_sparse_layers:
                        stored_gradient = layer.weight.grad.detach()
                        sampled_layer_grad_norms[name] = stored_gradient.norm()
                        if layer.weight_coordinate == "raw":
                            effective_gradient = stored_gradient * math.sqrt(layer.k)
                        else:
                            effective_gradient = stored_gradient
                        sampled_layer_effective_grad_norms[name] = effective_gradient.norm()
                preclip_grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                # Meta usefulness uses post-backward grads (before Adam step)
                if args.rewire == "meta":
                    raw_agent().update_meta_from_grad()
                optimizer.step()
                with torch.no_grad():
                    if sample_diagnostics:
                        sampled_layer_updates = {
                            name: (
                                layer.effective_weight() - effective_before[name]
                            ).abs().mean()
                            for name, layer in named_sparse_layers
                        }
                    clip_coefficient = torch.clamp(
                        args.max_grad_norm / (preclip_grad_norm + 1e-6), max=1.0
                    )
                    clip_coefficient_sum += clip_coefficient
                    clipped_steps += (preclip_grad_norm > args.max_grad_norm).to(
                        clipped_steps.dtype
                    )
                    preclip_grad_norm_sum += preclip_grad_norm
                    optimizer_steps += 1

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        raw_agent().freeze_rewire(False)
        iteration_approx_kl = iteration_approx_kl_sum / iteration_approx_kl_count

        if rewirer is not None:
            rewirer.step(
                obs,
                zs,
                logprobs,
                advantages,
                returns,
                values,
                float(iteration_approx_kl),
                writer,
                global_step,
            )

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # sparse diagnostics
        with torch.no_grad():
            util_means = []
            abs_w = []
            meta_u_means = []
            mature_fracs = []
            for trunk in (raw_agent().actor_trunk, raw_agent().critic_trunk):
                for layer in trunk.sparse_layers():
                    abs_w.append(layer.effective_weight().detach().abs().mean().item())
                    if layer._track_utility:
                        util_means.append(layer.utility.mean().item())
                    if layer.rewire_mode == "meta" and not layer.fully_dense:
                        meta_u_means.append(layer.meta_u.mean().item())
                        mature_fracs.append(
                            (layer.age >= float(layer.meta_age_min)).float().mean().item()
                        )
            writer.add_scalar("sparse/mean_abs_w", float(np.mean(abs_w)) if abs_w else 0.0, global_step)
            if util_means:
                writer.add_scalar("sparse/mean_utility", float(np.mean(util_means)), global_step)
            if meta_u_means:
                writer.add_scalar("sparse/mean_meta_u", float(np.mean(meta_u_means)), global_step)
                writer.add_scalar("sparse/mature_frac", float(np.mean(mature_fracs)), global_step)
            writer.add_scalar("sparse/rewire_events_iter", episode_rewire_events, global_step)
            writer.add_scalar("sparse/rewire_edges_iter", episode_rewire_edges, global_step)
            thr = raw_agent().current_threshold()
            if thr is not None:
                writer.add_scalar("sparse/utility_threshold", float(thr.detach()), global_step)
        episode_rewire_events = 0
        episode_rewire_edges = 0

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", iteration_approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "diagnostics/preclip_grad_norm",
            float(preclip_grad_norm_sum / optimizer_steps),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/clip_coefficient",
            float(clip_coefficient_sum / optimizer_steps),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/clipped_minibatch_fraction",
            float(clipped_steps / optimizer_steps),
            global_step,
        )
        for name, _ in named_sparse_layers:
            writer.add_scalar(
                f"diagnostics/stored_grad_norm_sample/{name}",
                float(sampled_layer_grad_norms[name]),
                global_step,
            )
            writer.add_scalar(
                f"diagnostics/effective_grad_norm_sample/{name}",
                float(sampled_layer_effective_grad_norms[name]),
                global_step,
            )
            writer.add_scalar(
                f"diagnostics/effective_update_mean_abs_sample/{name}",
                float(sampled_layer_updates[name]),
                global_step,
            )
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(raw_agent().state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()

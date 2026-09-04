# NEXTLAT DISTRIBUTIONAL BELLMAN CUDA v17: CUDA-resident value-grounded predictive coding.
# =====================================================================================
# Residual action-conditioned dynamics imagine h_{t+i}, i in {1,2,4,8}. Each imagined
# state is decoded by a rollout-frozen 511-bin Dreamer3 critic and fit to n-step
# categorical lambda-return targets starting at t+i (n in {1,2,4}); a scale-invariant latent
# residual anchor keeps predictions on the frozen encoder manifold. Auxiliary gradients
# reach only the live source trunk and dynamics model: target encoder, bootstrap critic,
# value decoder, and policy heads are frozen. A private Adam trains dynamics; trunk steps
# are block-locally conflict-projected and capped at 5% of the actual PPO task step.
# Hypothesis: Bellman agreement supplies the missing value semantics in NextLat without
# the inconsistent "future latent predicts source return" target that broke earlier TD arms.
# v17 preserves v11's learning rule while moving both large label constructors to CUDA,
# compiling the dense HL-Gauss projector under --compile, and retaining immutable labels
# on-device across PPO epochs. Tensor-side telemetry and asynchronous finite assertions
# remove avoidable per-minibatch host synchronization. Watch bellman/* and nextlat/*.
# =====================================================================================
import copy
import os
import random
import time
from dataclasses import dataclass
from math import log
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def value_support_bounds(args):
    """Return critic support endpoints in the coordinate system used by bins."""
    return args.v_min, args.v_max


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def parse_horizons(spec):
    """Parse a strictly increasing, positive comma-separated horizon schedule."""
    horizons = tuple(int(item.strip()) for item in spec.split(",") if item.strip())
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be unique and strictly increasing")
    return horizons


def shifted_flat_indices(indices, shift, num_envs, num_steps):
    """Shift T-major flattened indices while preserving their environment id."""
    if torch.is_tensor(indices):
        time_index = torch.clamp(
            torch.div(indices, num_envs, rounding_mode="floor") + shift,
            max=num_steps - 1,
        )
        return time_index * num_envs + torch.remainder(indices, num_envs)
    indices = np.asarray(indices)
    time_index = np.minimum(indices // num_envs + shift, num_steps - 1)
    return time_index * num_envs + indices % num_envs


def _categorical_affine_projection_unchecked(probabilities, shift, scale, support):
    """Tensor-only C51 projection used by the validated and compiled paths."""
    original_shape = probabilities.shape
    probs = probabilities.reshape(-1, support.numel())
    shifts = shift.reshape(-1, 1)
    scales = scale.reshape(-1, 1)
    atoms = (shifts + scales * support.reshape(1, -1)).clamp(
        support[0], support[-1]
    )
    upper = torch.searchsorted(support, atoms).clamp(max=support.numel() - 1)
    lower = (upper - 1).clamp(min=0)
    lower_value = support[lower]
    upper_value = support[upper]
    interval = upper_value - lower_value
    upper_weight = torch.where(
        interval > 0,
        (atoms - lower_value) / interval.clamp_min(torch.finfo(atoms.dtype).eps),
        torch.zeros_like(atoms),
    )
    lower_weight = 1.0 - upper_weight

    projected = torch.zeros_like(probs)
    projected.scatter_add_(1, lower, probs * lower_weight)
    projected.scatter_add_(1, upper, probs * upper_weight)
    return projected.reshape(original_shape)


@torch.no_grad()
def categorical_affine_projection(probabilities, shift, scale, support):
    """Project ``shift + scale * Z`` onto an arbitrary increasing support.

    This is the C51 categorical projection generalized to Dreamer3's nonuniform raw
    symexp buckets. It preserves all probability mass and the expectation whenever the
    transformed atoms remain inside the support. ``shift`` and ``scale`` are per-row.
    """
    if probabilities.shape[-1] != support.numel():
        raise ValueError("probability and support sizes differ")
    if probabilities.shape[:-1] != shift.shape or shift.shape != scale.shape:
        raise ValueError("shift and scale must match the probability batch shape")
    if torch.any(support[1:] <= support[:-1]):
        raise ValueError("support must be strictly increasing")
    return _categorical_affine_projection_unchecked(
        probabilities, shift, scale, support
    )


def _build_nstep_distributional_targets_impl(
    rewards,
    terminations,
    boundaries,
    transition_valids,
    next_value_probabilities,
    next_values,
    support,
    gamma,
    gae_lambda,
    horizons,
):
    """Static tensor program for episode-correct categorical lambda returns.

    There are deliberately no data-dependent Python branches or compacted valid-row
    indices here. Projecting the few invalid rows as well makes this safe to compile and
    avoids CUDA-to-host synchronization; the final mask zeros those rows exactly.
    """
    time_steps, num_envs = rewards.shape
    num_bins = support.numel()
    horizons = tuple(horizons)
    targets = rewards.new_zeros(time_steps, num_envs, len(horizons), num_bins)
    masks = torch.zeros(
        time_steps, num_envs, len(horizons), dtype=torch.bool, device=rewards.device
    )
    center = num_bins // 2

    for horizon_index, horizon in enumerate(horizons):
        # At lambda=0 every nominal horizon is exactly the same one-step target;
        # requiring unavailable later transitions would censor already-complete labels.
        unroll_steps = 1 if gae_lambda == 0.0 else horizon
        prefix = torch.zeros_like(rewards)
        bootstrap_scale = torch.zeros_like(rewards)
        bootstrap_probs = torch.zeros_like(next_value_probabilities)
        bootstrap_probs[..., center] = 1.0  # terminal rows: scale=0, so any unit mass works
        active = torch.ones_like(rewards, dtype=torch.bool)
        valid = torch.zeros_like(active)

        for step in range(unroll_steps):
            valid_length = time_steps - step
            if valid_length <= 0:
                break
            active_now = active[:valid_length]
            reward_now = rewards[step:]
            trace_discount = (gamma * gae_lambda) ** step
            prefix[:valid_length].add_(
                active_now.to(rewards.dtype) * trace_discount * reward_now
            )

            boundary_now = boundaries[step:].bool()
            termination_now = terminations[step:].bool()
            next_valid_now = transition_valids[step:].bool()
            stopped = active_now & boundary_now
            terminal = stopped & termination_now
            truncated = stopped & ~termination_now & next_valid_now
            valid[:valid_length] |= terminal | truncated
            # GAE stops its trace at every reset boundary. A truncation still receives
            # the complete one-step gamma*V(final_obs) bootstrap; it is not attenuated
            # by (1-lambda), because no lambda continuation follows the boundary.
            prefix[:valid_length].add_(
                truncated.to(rewards.dtype)
                * trace_discount
                * gamma
                * next_values[step:]
            )

            continuing = active_now & ~boundary_now & next_valid_now
            if gae_lambda < 1.0:
                prefix[:valid_length].add_(
                    continuing.to(rewards.dtype)
                    * trace_discount
                    * gamma
                    * (1.0 - gae_lambda)
                    * next_values[step:]
                )
            active[:valid_length] = continuing
            if step == unroll_steps - 1:
                bootstrapped = active[:valid_length]
                valid[:valid_length] |= bootstrapped
                bootstrap_probs[:valid_length] = torch.where(
                    bootstrapped.unsqueeze(-1),
                    next_value_probabilities[step:],
                    bootstrap_probs[:valid_length],
                )
                bootstrap_scale[:valid_length] = torch.where(
                    bootstrapped,
                    bootstrap_scale.new_full(
                        (), (gamma * gae_lambda) ** (step + 1)
                    ),
                    bootstrap_scale[:valid_length],
                )

        projected = _categorical_affine_projection_unchecked(
            bootstrap_probs, prefix, bootstrap_scale, support
        )
        targets[:, :, horizon_index] = torch.where(
            valid.unsqueeze(-1), projected, torch.zeros_like(projected)
        )
        masks[:, :, horizon_index] = valid

    return targets, masks


@torch.no_grad()
def build_nstep_distributional_targets(
    rewards,
    terminations,
    boundaries,
    transition_valids,
    next_value_probabilities,
    next_values,
    support,
    gamma,
    gae_lambda,
    horizons,
    projection_chunk=2048,
):
    """Build episode-correct categorical n-step lambda-return targets.

    A true termination ends the return with no bootstrap. A truncation bootstraps from
    its supplied final observation, but is censored when that observation is missing.
    Targets that run off the rollout tail are invalid unless an earlier boundary already
    completed them. For an uninterrupted suffix, the affine target is

        sum_k (gamma * lambda)^k [r_k + gamma * (1-lambda) * V(s_{k+1})]
        + (gamma * lambda)^n * Z(s_n).

    ``projection_chunk`` remains API-compatible with v11 but is intentionally unused:
    static full-row projection is faster on CUDA and avoids dynamic-shape synchronization.
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (time, env)")
    time_steps, num_envs = rewards.shape
    num_bins = support.numel()
    expected = (time_steps, num_envs)
    for name, tensor in (
        ("terminations", terminations),
        ("boundaries", boundaries),
        ("transition_valids", transition_valids),
    ):
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if next_value_probabilities.shape != (time_steps, num_envs, num_bins):
        raise ValueError("next_value_probabilities has the wrong shape")
    if next_values.shape != expected:
        raise ValueError(f"next_values must have shape {expected}")
    if any(
        tensor.device != rewards.device
        for tensor in (
            terminations,
            boundaries,
            transition_valids,
            next_value_probabilities,
            next_values,
            support,
        )
    ):
        raise ValueError("all target tensors must reside on the same device")
    if not rewards.is_floating_point() or any(
        tensor.dtype != rewards.dtype
        for tensor in (next_value_probabilities, next_values, support)
    ):
        raise ValueError(
            "rewards, probabilities, values, and support must share a floating dtype"
        )
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must lie in [0, 1]")
    if projection_chunk <= 0:
        raise ValueError("projection_chunk must be positive")
    horizons = tuple(horizons)
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    if torch.any(support[1:] <= support[:-1]):
        raise ValueError("support must be strictly increasing")
    return _build_nstep_distributional_targets_impl(
        rewards,
        terminations,
        boundaries,
        transition_valids,
        next_value_probabilities,
        next_values,
        support,
        gamma,
        gae_lambda,
        horizons,
    )


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
    norm_adv: bool = True            # ppoadvnorm_batch base: standard PPO advantage standardization ON
    # --- Percentile advantage normalization (OFF in the ppoadvnorm_batch base) ---
    ret_percnorm: bool = False       # scale policy advantage by S = max(floor, P95-P5) of returns
    ret_perc_scope: str = "minibatch"  # "minibatch" = fresh per-mb P95-P5 of mb returns (NO EMA, like the
    #                                  old per-mb advnorm); "batch" = fresh WHOLE-ROLLOUT P95-P5, one S per
    #                                  update, NO EMA (the batch-vs-mb ablation); "ema" = v1's global EMA.
    ret_perc_rate: float = 0.01      # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    # SOFT-ADVANTAGE max-ent (gaussian only; --auto-entropy). Entropy enters the POLICY
    # ADVANTAGE only, NEVER the critic's target. The critic regresses to the RAW reward
    # return (fits the fixed support [v_min,v_max]); a SEPARATE soft-advantage GAE adds the
    # bootstrap bonus b_t = α·H_sq(s_{t+1}), estimated as the single-sample SQUASHED entropy
    # -logπ_sq(a|s), matching SAC's next_state_log_pi units. Rationale: forcing the bounded
    # categorical critic to learn the soft value both wastes capacity and overflows the
    # support (the softboot failure: edge_mass→0.9, expl_var→0). alpha is
    # auto-tuned by SAC's exact dual (target_entropy = -|A|; alpha_loss = -α(logπ + tgt)).
    # The explicit α·H actor bonus supplies the CURRENT-step entropy gradient (the soft
    # advantage's current-state entropy is action-independent => cancels in the PG term);
    # the soft advantage supplies FUTURE-entropy credit. Works WITH rankgauss: the soft
    # value reorders advantages and rankgauss preserves order/sign (magnitude is incidental).
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Dreamer3-style bucket critic: centers are symexp(linspace(v_min, v_max, num_bins)).
    # Defaults match v162's ±20k raw support, expressed in symlog coordinates.
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    # Keep observation normalization, but disable return/reward normalization for
    # this ablation. NormalizeReward tracks discounted returns internally and
    # divides rewards by their running std, so it must remain off here.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "v10"       # d3percnorm: identity -- NO rankgauss. DreamerV3 percentile
    #                                  norm (below) is the sole advantage scaler ("no advantage norm").

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "batch"    # ppoadvnorm_batch base: standardize ONCE over the whole rollout

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    sigma_floor_bins: float = 2.0    # floor sigma(s) at this many bins (avoid /~0)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # v17 CUDA-resident value-grounded NextLat. Comma-separated powers of two keep the CLI compact.
    nextlat: bool = True
    nextlat_horizons: str = "1,2,4,8"
    bellman_horizons: str = "1,2,4"
    nextlat_coef: float = 1.0
    nextlat_bellman_coef: float = 1.0
    nextlat_latent_coef: float = 0.25
    nextlat_huber_beta: float = 0.1
    nextlat_trunk_grad_clip: float = 0.1
    nextlat_predictor_grad_clip: float = 1.0
    bellman_projection_chunk: int = 2048

    # Task and prediction have independent Adam moments;
    # this ratio caps the admitted predictive PARAMETER update relative to the actual
    # actor+critic update on the shared trunk (not relative to raw gradients).
    predadam_trust_ratio: float = 0.05
    predadam_meta_gate: bool = False
    predadam_meta_decay: float = 0.95
    predadam_meta_warmup: int = 32

    # Compile only the hot forward paths. Optimizer surgery intentionally remains eager.
    compile: bool = False
    compile_mode: str = "reduce-overhead"

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, normalize_reward, clip_reward):
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
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    # Plain pre-act MLP. Standard sqrt(2) init on both Linears.
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        ReLUSquared(),
        layer_init(nn.Linear(H, H)),
    )


class ThinkBlock(nn.Module):
    """Bounded convex residual mix of x and x0, then parallel dense + soft MoE."""

    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.n_experts = n_experts

        self.in_proj = layer_init(nn.Linear(in_dim, H))
        # Bounded convex residual gate: g = sigmoid(resid_gate) per channel.
        # init +4 → g ≈ 0.982 → x_in ≈ x at start.
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        # Dense branch. RMSNorm without learnable affine (no free per-channel γ).
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)

        # Soft MoE branch (softmax over all experts, no top-K).
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in))                   # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1) # (B, E, H)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)          # (B, H)

        return x_in + d_dense + d_moe


class ResidualDynamics(nn.Module):
    """Two-stage action-conditioned residual dynamics, initialized as identity."""

    def __init__(self, hidden, action_dim):
        super().__init__()
        input_dim = hidden + action_dim
        self.linear_norm = nn.RMSNorm(input_dim, elementwise_affine=False)
        self.linear_delta = layer_init(nn.Linear(input_dim, hidden), std=0.0)
        self.residual_norm = nn.RMSNorm(input_dim + hidden, elementwise_affine=False)
        self.nonlinear_delta = nn.Sequential(
            layer_init(nn.Linear(input_dim + hidden, hidden)),
            ReLUSquared(),
            layer_init(nn.Linear(hidden, hidden), std=0.0),
        )

    def forward(self, latent, action):
        features = torch.cat([latent, action], dim=-1)
        linear_delta = self.linear_delta(self.linear_norm(features))
        nonlinear_input = self.residual_norm(
            torch.cat([features, linear_delta], dim=-1)
        )
        full_delta = linear_delta + self.nonlinear_delta(nonlinear_input)
        return latent + full_delta, linear_delta


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            block_in_dim = H * (k + 1)
            self.blocks.append(ThinkBlock(block_in_dim, H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # v162 critic: bias-free neutral MTP head. With symmetric symlog support,
        # zero logits decode to a zero raw value without a hidden prior.
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
        # v24: action distribution. Both parameterizations are dreamer4-faithful;
        # the Gaussian path is tanh-squashed like SAC but uses log-variance, not log_std.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            # dreamer4 Gaussian: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # dreamer4 unimodal Beta: two concentration heads, alpha,beta = 1 + softplus.
            self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")
        # Identity at initialization: capacity and gradients focus on state change.
        self.nextlat_predictor = ResidualDynamics(H, act_dim)

    def _actor_dist(self, actor_feat):
        # Build the action distribution and the native-space transforms.
        # Returns (dist, to_action, log_det_fn) where:
        #   to_action(z): map a NATIVE sample z to the env action.
        #   log_det_fn(z): per-sample log|d action / d z| correction to SUBTRACT
        #                  from dist.log_prob(z) (0 where the map is volume-constant).
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
            # dreamer4 soft tanh-rescale bound on log-variance (smooth, no dead grad).
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            std = (0.5 * lv).exp()
            dist = Normal(mean, std)
            to_action = torch.tanh
            log_det_fn = lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
            return dist, to_action, log_det_fn
        # beta
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns value LOGITS (B, mtp, num_bins); horizon 0 is V(s_t).
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_actor_feat(self, x):
        # Actor-side trunk feature used as the live source of imagined rollouts.
        return self._trunks(x)[0]

    def get_action_and_value(self, x, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_logits = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        if self.actor_dist == "gaussian":
            # Reparameterized SQUASHED-entropy estimate H_sq = E_ε[-logπ_sq(tanh(μ+σε))].
            # Base-Normal H = dist.entropy() is monotone↑ in σ, so an entropy bonus rails σ
            # to the ceiling -> tanh saturates -> squashed H collapses, while the α-dual
            # (which targets squashed H) cranks α up: a runaway. The squashed H is BOUNDED
            # with an interior max in σ, so maximizing it settles σ at a finite optimum and
            # is consistent with the α target. Fresh rsample => gradient flows to μ,σ
            # (independent of the replayed z used for the PPO ratio).
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, value_logits, actor_feat

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk). The two
        # distribution heads are clipped together as one actor group (2-way
        # decoupled clip; no separate std budget — gaussian's variance head and
        # both beta concentration heads sit in the same group).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())

    def nextlat_parameters(self):
        # Only the representation and predictor receive the auxiliary gradient.
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        return list(trunk.parameters()) + list(self.nextlat_predictor.parameters())

    def nextlat_trunk_parameters(self):
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        return list(trunk.parameters())

    def nextlat_trunk_parameter_blocks(self):
        """Architectural blocks used by predictive update admission.

        A block-level trust region is substantially less noisy than a tensor-local one,
        while keeping the five default reductions small enough to avoid hundreds of
        scalar CUDA kernels per minibatch.
        """
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        blocks = [list(trunk.entry.parameters())]
        blocks.extend(list(block.parameters()) for block in trunk.blocks)
        blocks.append(list(trunk.out_proj.parameters()))
        blocks = [block for block in blocks if block]
        assert [id(parameter) for block in blocks for parameter in block] == [
            id(parameter) for parameter in trunk.parameters()
        ]
        return blocks

    def nextlat_predictor_parameters(self):
        return list(self.nextlat_predictor.parameters())

    def task_parameters(self):
        predictor_ids = {id(p) for p in self.nextlat_predictor.parameters()}
        return [p for p in self.parameters() if id(p) not in predictor_ids]


class FrozenValueTarget(nn.Module):
    """Iteration-frozen encoder and value decoder used only by the auxiliary."""

    def __init__(self, agent):
        super().__init__()
        if not agent.share_backbone:
            raise ValueError("Bellman NextLat requires a shared actor/critic latent")
        self.encoder = copy.deepcopy(agent.trunk)
        self.decoder = copy.deepcopy(agent.critic_head)
        self.num_bins = agent.num_bins
        self.critic_mtp_horizon = agent.critic_mtp_horizon
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def sync_from(self, agent):
        self.encoder.load_state_dict(agent.trunk.state_dict())
        self.decoder.load_state_dict(agent.critic_head.state_dict())
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, observations):
        return self.encoder(observations)

    @torch.no_grad()
    def value_probabilities(self, observations):
        latent = self.encoder(observations)
        logits = self.decoder(latent).view(
            -1, self.critic_mtp_horizon, self.num_bins
        )[:, 0]
        return torch.softmax(logits, dim=-1)

    def decode_imagined(self, latent):
        """Decode with constant weights while retaining gradients to ``latent``."""
        return self.decoder(latent).view(
            -1, self.critic_mtp_horizon, self.num_bins
        )[:, 0]


def _sum_squares(tensors):
    total = tensors[0].new_zeros(())
    for tensor in tensors:
        total = total + tensor.square().sum()
    return total


def clip_grad_norm_async_(parameters, max_norm, group_name):
    """Clip gradients and fail loudly without a CUDA-to-host scalar read.

    ``error_if_nonfinite=True`` performs a Python-side truth test of the CUDA norm.
    The asynchronous assertion instead stays ordered on the device stream: finite runs
    incur no host synchronization, while NaN/Inf gradients still abort before useful
    work can continue. Materialize the iterable once because parameter generators are
    otherwise exhausted by the norm implementation.
    """
    parameters = list(parameters)
    total_norm = nn.utils.clip_grad_norm_(
        parameters, max_norm, error_if_nonfinite=False
    )
    torch._assert_async(
        torch.isfinite(total_norm), f"non-finite {group_name} gradient norm"
    )
    return total_norm


def _sum_inner_products(left, right):
    total = left[0].new_zeros(())
    for left_tensor, right_tensor in zip(left, right):
        total = total + (left_tensor * right_tensor).sum()
    return total


def _flatten_tensors(tensors):
    """Copy a small architectural parameter block into one contiguous vector."""
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors])


def _flat_views(vector, references):
    views = []
    offset = 0
    for reference in references:
        next_offset = offset + reference.numel()
        views.append(vector[offset:next_offset].view_as(reference))
        offset = next_offset
    if offset != vector.numel():
        raise ValueError("flat vector size does not match its parameter block")
    return views


def _projection_dot_tolerance(gradient_sq, update_sq):
    """Scale-aware roundoff tolerance for one first-order half-space check."""
    eps = torch.finfo(gradient_sq.dtype).eps
    return 64.0 * eps * (gradient_sq * update_sq).clamp_min(0.0).sqrt()


@torch.no_grad()
def _repair_optimizer_moments(optimizer, reference):
    """Report whether optimizer moments were finite and repair invalid entries."""
    moments_finite = reference.new_ones((), dtype=torch.bool)
    for state in optimizer.state.values():
        for name, value in state.items():
            # Adam's scalar clock intentionally advances on a rejected transaction.
            if name == "step" or not torch.is_tensor(value):
                continue
            if not (torch.is_floating_point(value) or torch.is_complex(value)):
                continue
            finite = torch.isfinite(value)
            value_finite = finite.all()
            if value_finite.device != moments_finite.device:
                value_finite = value_finite.to(moments_finite.device)
            moments_finite = moments_finite & value_finite
            value.copy_(torch.where(finite, value, torch.zeros_like(value)))
    return moments_finite


@torch.no_grad()
def _project_block_to_joint_descent(update, actor_gradient, critic_gradient):
    """Euclidean projection onto both first-order task-descent half-spaces."""
    zero = torch.zeros_like(update)
    tiny = torch.finfo(update.dtype).tiny
    eps = torch.finfo(update.dtype).eps
    aa = actor_gradient.square().sum()
    cc = critic_gradient.square().sum()
    ac = (actor_gradient * critic_gradient).sum()
    ap = (actor_gradient * update).sum()
    cp = (critic_gradient * update).sum()

    def halfspaces_feasible(candidate):
        candidate_sq = candidate.square().sum()
        actor_dot = (actor_gradient * candidate).sum()
        critic_dot = (critic_gradient * candidate).sum()
        actor_tolerance = _projection_dot_tolerance(aa, candidate_sq)
        critic_tolerance = _projection_dot_tolerance(cc, candidate_sq)
        finite = (
            torch.isfinite(candidate).all()
            & torch.isfinite(actor_gradient).all()
            & torch.isfinite(critic_gradient).all()
            & torch.isfinite(aa)
            & torch.isfinite(cc)
            & torch.isfinite(candidate_sq)
            & torch.isfinite(actor_dot)
            & torch.isfinite(critic_dot)
            & torch.isfinite(actor_tolerance)
            & torch.isfinite(critic_tolerance)
        )
        return (
            finite
            & (actor_dot <= actor_tolerance)
            & (critic_dot <= critic_tolerance)
        )

    actor_candidate = update - ap.clamp_min(0.0) / aa.clamp_min(tiny) * actor_gradient
    critic_candidate = update - cp.clamp_min(0.0) / cc.clamp_min(tiny) * critic_gradient
    # A direct 2x2 Gram solve loses most of its significant bits for correlated task
    # gradients. Gram-Schmidt expresses the same joint active set without subtracting
    # ``aa * cc - ac**2`` or two huge, nearly cancelling constraint updates.
    safe_aa = aa.clamp_min(tiny)
    actor_norm = aa.sqrt()
    critic_norm = cc.sqrt()
    actor_unit = actor_gradient / actor_norm.clamp_min(tiny)
    critic_unit = critic_gradient / critic_norm.clamp_min(tiny)
    correlation = (actor_unit * critic_unit).sum()
    alignment_sign = torch.where(
        correlation >= 0.0,
        torch.ones_like(correlation),
        -torch.ones_like(correlation),
    )
    angular_difference = critic_unit - alignment_sign * actor_unit
    critic_orthogonal = angular_difference - (
        actor_unit * angular_difference
    ).sum() * actor_unit
    critic_orthogonal = critic_orthogonal - (
        actor_unit * critic_orthogonal
    ).sum() * actor_unit
    critic_orthogonal_sq = critic_orthogonal.square().sum()
    determinant_valid = (
        torch.isfinite(aa)
        & torch.isfinite(cc)
        & torch.isfinite(critic_orthogonal_sq)
        & (aa > tiny)
        & (cc > tiny)
        & (critic_orthogonal_sq > eps * eps)
    )
    safe_orthogonal_sq = torch.where(
        determinant_valid,
        critic_orthogonal_sq,
        torch.ones_like(critic_orthogonal_sq),
    )
    actor_span = (actor_unit * update).sum()
    actor_boundary_candidate = update - actor_span * actor_unit
    orthogonal_multiplier = (
        critic_orthogonal * update
    ).sum() / safe_orthogonal_sq
    critic_multiplier = orthogonal_multiplier / critic_norm.clamp_min(tiny)
    critic_on_actor = alignment_sign + (actor_unit * angular_difference).sum()
    actor_multiplier = (
        actor_span - orthogonal_multiplier * critic_on_actor
    ) / actor_norm.clamp_min(tiny)
    joint_candidate = actor_boundary_candidate - (
        orthogonal_multiplier * critic_orthogonal
    )

    best = zero
    best_distance = update.square().sum()
    candidates = (
        (update, halfspaces_feasible(update)),
        (actor_candidate, halfspaces_feasible(actor_candidate)),
        (critic_candidate, halfspaces_feasible(critic_candidate)),
        (
            joint_candidate,
            determinant_valid
            & (actor_multiplier >= 0.0)
            & (critic_multiplier >= 0.0)
            & halfspaces_feasible(joint_candidate),
        ),
    )
    for candidate, feasible in candidates:
        distance = (candidate - update).square().sum()
        choose = feasible & (distance < best_distance)
        best = torch.where(choose, candidate, best)
        best_distance = torch.where(choose, distance, best_distance)
    feasible = halfspaces_feasible(best)
    return torch.where(feasible, best, zero)


@torch.no_grad()
def admit_predictive_updates(
    task_updates,
    predictive_updates,
    max_ratio,
    *,
    actor_gradients=None,
    critic_gradients=None,
    gates=None,
):
    """Protect and block-locally cap predictive Adam parameter deltas.

    Inputs are one contiguous FP32 vector per architectural trunk block. Each proposal is
    projected so it cannot increase either the actor or critic loss to first order, then
    capped to ``max_ratio`` of that block's actual task-Adam delta. The local caps imply
    the same global norm cap. All decisions stay tensor-side; no FP64 conversions or
    per-minibatch host synchronizations are needed.
    """
    if len(task_updates) != len(predictive_updates):
        raise ValueError("task and predictive update lists must have equal length")
    if not task_updates:
        raise ValueError("at least one update block is required")
    if gates is not None and len(gates) != len(predictive_updates):
        raise ValueError("one meta-gate is required per predictive update block")
    if max_ratio < 0.0:
        raise ValueError("max_ratio must be non-negative")
    for task, predictive in zip(task_updates, predictive_updates):
        if task.shape != predictive.shape or task.device != predictive.device:
            raise ValueError("paired task and predictive blocks must share shape and device")
        if not task.is_floating_point() or task.dtype != predictive.dtype:
            raise ValueError("paired update blocks must share a floating-point dtype")

    # The negative task delta is a reasonable fallback gradient proxy for tests and
    # standalone callers. Training supplies the separately clipped loss gradients.
    if actor_gradients is None:
        actor_gradients = [-update for update in task_updates]
    if critic_gradients is None:
        critic_gradients = [torch.zeros_like(update) for update in task_updates]
    if len(actor_gradients) != len(task_updates) or len(critic_gradients) != len(task_updates):
        raise ValueError("actor/critic gradients must align with update blocks")

    task_sq = _sum_squares(task_updates)
    raw_pred_sq = _sum_squares(predictive_updates)
    raw_dot = _sum_inner_products(task_updates, predictive_updates)
    projected_updates = []
    admitted = []
    actor_conflicts = task_sq.new_zeros(())
    critic_conflicts = task_sq.new_zeros(())
    max_block_ratio = task_sq.new_zeros(())
    for index, (task, predictive, actor_gradient, critic_gradient) in enumerate(
        zip(task_updates, predictive_updates, actor_gradients, critic_gradients)
    ):
        gate = 1.0 if gates is None else gates[index]
        gated = predictive * torch.as_tensor(
            gate, device=predictive.device, dtype=predictive.dtype
        )
        actor_conflicts = actor_conflicts + (
            (actor_gradient * gated).sum() > 0.0
        ).to(task_sq.dtype)
        critic_conflicts = critic_conflicts + (
            (critic_gradient * gated).sum() > 0.0
        ).to(task_sq.dtype)
        projected = _project_block_to_joint_descent(
            gated, actor_gradient, critic_gradient
        )
        task_norm_local = task.square().sum().sqrt()
        projected_norm = projected.square().sum().sqrt()
        local_scale = torch.clamp(
            max_ratio * task_norm_local / projected_norm.clamp_min(1e-20), max=1.0
        )
        local_scale = local_scale * (task_norm_local > 0.0).to(local_scale.dtype)
        admitted_update = projected * local_scale
        admitted_norm_local = admitted_update.square().sum().sqrt()
        max_block_ratio = torch.maximum(
            max_block_ratio,
            admitted_norm_local / task_norm_local.clamp_min(1e-20),
        )
        projected_updates.append(projected)
        admitted.append(admitted_update)

    projected_sq = _sum_squares(projected_updates)
    admitted_sq = _sum_squares(admitted)
    task_norm = task_sq.sqrt()
    raw_cosine = raw_dot / (task_sq * raw_pred_sq).sqrt().clamp_min(1e-20)
    accepted_fraction = admitted_sq.sqrt() / raw_pred_sq.sqrt().clamp_min(1e-20)
    block_count = task_sq.new_tensor(float(len(task_updates)))
    return admitted, {
        "task_norm": task_norm,
        "predictive_norm": raw_pred_sq.sqrt(),
        "raw_cosine": raw_cosine,
        "cap_scale": admitted_sq.sqrt() / projected_sq.sqrt().clamp_min(1e-20),
        "accepted_fraction": accepted_fraction,
        "admitted_norm": admitted_sq.sqrt(),
        "actor_first_order": _sum_inner_products(actor_gradients, admitted),
        "critic_first_order": _sum_inner_products(critic_gradients, admitted),
        "actor_conflict_fraction": actor_conflicts / block_count,
        "critic_conflict_fraction": critic_conflicts / block_count,
        "max_block_ratio": max_block_ratio,
    }


@torch.no_grad()
def apply_predictive_optimizer_transaction(
    trunk_blocks,
    predictor_parameters,
    optimizer,
    gradients,
    task_updates,
    max_ratio,
    *,
    actor_gradients,
    critic_gradients,
    gates=None,
):
    """Advance predictive Adam, then atomically admit its finite safe proposal.

    Predictive Adam owns both the shared trunk and the private dynamics model. Its
    overlapping trunk proposal is always rolled back before admission. If any input,
    optimizer proposal, or admitted delta is nonfinite, the entire parameter transaction
    is restored exactly (including the private predictor); invalid gradients are replaced
    by zero only for the Adam call so its moments cannot be poisoned.
    """
    if not trunk_blocks or not predictor_parameters:
        raise ValueError("trunk blocks and private predictor parameters are required")
    trunk_parameters = [parameter for block in trunk_blocks for parameter in block]
    parameters = trunk_parameters + list(predictor_parameters)
    if len(gradients) != len(parameters):
        raise ValueError("one predictive gradient is required per parameter")
    if not (
        len(task_updates)
        == len(actor_gradients)
        == len(critic_gradients)
        == len(trunk_blocks)
    ):
        raise ValueError("task updates and task gradients must align with trunk blocks")
    if gates is not None and len(gates) != len(trunk_blocks):
        raise ValueError("one meta-gate is required per trunk block")

    reference = parameters[0]
    numeric_valid = torch.ones((), dtype=torch.bool, device=reference.device)
    for parameter, gradient in zip(parameters, gradients):
        if (
            parameter.shape != gradient.shape
            or parameter.device != gradient.device
            or parameter.dtype != gradient.dtype
        ):
            raise ValueError("predictive gradients must match their parameters")
    gradient_lengths = tuple(parameter.numel() for parameter in parameters)
    flat_gradient = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
    # Adam squares gradients. Values can therefore be finite yet overflow its second
    # moment (for FP32, e.g. 1e30). Veto the complete vector before the optimizer call,
    # not after it has poisoned persistent state.
    gradient_safe = torch.isfinite(flat_gradient).all() & torch.isfinite(
        flat_gradient.square()
    ).all()
    numeric_valid = numeric_valid & gradient_safe
    for collections in (task_updates, actor_gradients, critic_gradients):
        for tensor in collections:
            numeric_valid = numeric_valid & torch.isfinite(tensor).all()
    if gates is not None:
        for gate in gates:
            gate_tensor = torch.as_tensor(
                gate, device=reference.device, dtype=reference.dtype
            )
            numeric_valid = numeric_valid & torch.isfinite(gate_tensor).all()

    trunk_before = [_flatten_tensors(block) for block in trunk_blocks]
    predictor_before = _flatten_tensors(predictor_parameters)
    optimizer.zero_grad(set_to_none=True)
    flat_gradient = torch.where(
        gradient_safe,
        flat_gradient,
        torch.zeros_like(flat_gradient),
    )
    for parameter, gradient in zip(parameters, flat_gradient.split(gradient_lengths)):
        parameter.grad = gradient.view_as(parameter)
    optimizer.step()

    raw_updates = [
        _flatten_tensors(block) - before
        for block, before in zip(trunk_blocks, trunk_before)
    ]
    predictor_update = _flatten_tensors(predictor_parameters) - predictor_before
    for tensor in raw_updates + [predictor_update]:
        numeric_valid = numeric_valid & torch.isfinite(tensor).all()
    numeric_valid = numeric_valid & _repair_optimizer_moments(optimizer, reference)

    # The task-updated trunk is the transaction's exact restore point. Admission below
    # sees finite surrogates only; ``numeric_valid`` still records any replacement.
    for block, before in zip(trunk_blocks, trunk_before):
        torch._foreach_copy_(block, _flat_views(before, block))
    safe_task_updates = [
        torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        for tensor in task_updates
    ]
    safe_raw_updates = [
        torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        for tensor in raw_updates
    ]
    safe_actor_gradients = [
        torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        for tensor in actor_gradients
    ]
    safe_critic_gradients = [
        torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        for tensor in critic_gradients
    ]
    safe_gates = None
    if gates is not None:
        safe_gates = [
            torch.nan_to_num(
                torch.as_tensor(gate, device=reference.device, dtype=reference.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for gate in gates
        ]
    admitted_updates, stats = admit_predictive_updates(
        safe_task_updates,
        safe_raw_updates,
        max_ratio,
        actor_gradients=safe_actor_gradients,
        critic_gradients=safe_critic_gradients,
        gates=safe_gates,
    )
    for tensor in admitted_updates:
        numeric_valid = numeric_valid & torch.isfinite(tensor).all()
    for value in stats.values():
        numeric_valid = numeric_valid & torch.isfinite(value).all()

    admitted_updates = [
        torch.where(numeric_valid, tensor, torch.zeros_like(tensor))
        for tensor in admitted_updates
    ]
    for block, update in zip(trunk_blocks, admitted_updates):
        torch._foreach_add_(block, _flat_views(update, block))
    predictor_after = torch.where(
        numeric_valid,
        _flatten_tensors(predictor_parameters),
        predictor_before,
    )
    torch._foreach_copy_(
        predictor_parameters,
        _flat_views(predictor_after, predictor_parameters),
    )
    predictor_step_sq = torch.where(
        numeric_valid,
        predictor_update.square().sum(),
        reference.new_zeros(()),
    )
    returned_raw_updates = [
        torch.where(numeric_valid, tensor, torch.zeros_like(tensor))
        for tensor in safe_raw_updates
    ]
    zero = reference.new_zeros(())
    stats = {
        key: torch.where(numeric_valid, torch.nan_to_num(value), zero)
        for key, value in stats.items()
    }
    stats["numeric_valid"] = numeric_valid.to(dtype=reference.dtype)
    optimizer.zero_grad(set_to_none=True)
    return returned_raw_updates, admitted_updates, predictor_step_sq.sqrt(), stats


def shape_advantage(gae, sigma, u, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform. Works on a full
    batch or a single minibatch (sigma/u must be sliced to match gae)."""
    if args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_std":
        # Per-state-normalized, bounded, magnitude-preserving near 0 (THE FIX).
        return torch.tanh(gae / (args.tanh_kappa * sigma))
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "cdf_probit":
        centered = (2.0 * u - 1.0)
        c = args.cdf_probit_clamp
        return ((2.0 ** 0.5) * torch.erfinv(centered.clamp(-c, c).cpu())).to(device)
    elif args.adv_transform == "clip_z":
        # Winsorized z-score: ONE standardize + hard tail clip. tanh's hard cousin;
        # preserves bulk magnitude exactly (linear in [-c,c]) and is ~self-normalizing
        # (clipping a unit-var signal barely changes its std), so norm_adv ~ no-op.
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return gz.clamp(-args.clip_z_c, args.clip_z_c)
    elif args.adv_transform == "rankgauss":
        # Rank-Gaussian: the principled single op. Replaces each advantage by the
        # Gaussian quantile of its empirical rank -> self-normalizing (bounded,
        # zero-mean, ~unit-scale, fully outlier-immune; no kappa, no double z-score).
        # This is cdf_probit with the EMPIRICAL batch rank instead of the critic's
        # per-state CDF (distribution-free; the distribution was shown incidental).
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)  # 0..n-1
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        return ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
    elif args.adv_transform == "rankgauss_signed":
        # Sign-anchored rank-Gaussian: rank positives and negatives SEPARATELY so the
        # advantage's zero-crossing is preserved exactly. Plain rankgauss anchors the
        # sign flip at the batch MEDIAN, not 0, so on skewed GAE it can assign the
        # wrong sign to near-median samples — and PPO needs the sign right. Each sign
        # group is mapped to its half of the Gaussian by its within-group rank.
        c = args.cdf_probit_clamp
        out = torch.zeros_like(gae)
        for side in (gae > 0, gae < 0):
            if side.any():
                g = gae[side]
                r = g.argsort().argsort().to(torch.float32)
                half = (r + 0.5) / float(g.numel())                  # (0,1) in group
                uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)  # correct side
                ctr = (2.0 * uq - 1.0).clamp(-c, c)
                out[side] = ((2.0 ** 0.5) * torch.erfinv(ctr.cpu())).to(device)
        return out
    elif args.adv_transform == "rankgauss_temp":
        # Rank-Gaussian + tanh temperature: compress the Gaussian quantiles' extremes
        # harder (rank_tanh_kappa < inf). Probes the "more robust still" axis the kappa
        # sweep motivated (tanh_gae kappa=1 > kappa=2). Smaller kappa => harder.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        z = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
        return torch.tanh(z / args.rank_tanh_kappa)
    elif args.adv_transform == "rankgauss_signmag":
        # Sign-correct WITHOUT count distortion: take plain rankgauss's GLOBAL-rank
        # magnitude, then force the sign to match the raw advantage. Fixes the flaw in
        # rankgauss_signed (per-group half-Gaussian over-amplifies the minority sign by
        # COUNT); here magnitude still reflects global rank extremity and only the ~9%
        # near-zero "flips" get re-signed. Nonlinear (not a shift) => survives norm_adv.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c).cpu())).to(device).abs()
        return torch.sign(gae) * mag
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    nextlat_horizons = parse_horizons(args.nextlat_horizons)
    bellman_horizons = parse_horizons(args.bellman_horizons)
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch", "batch_retstd", "minibatch_retstd")
    assert 0.0 <= args.predadam_trust_ratio <= 1.0
    assert 0.0 <= args.predadam_meta_decay < 1.0
    assert args.predadam_meta_warmup >= 0
    assert 0.0 <= args.gae_lambda <= 1.0
    assert args.bellman_projection_chunk > 0
    assert args.separate_grad_clip, "PredAdam requires decoupled task/auxiliary gradients"
    assert args.share_backbone, "Bellman decoding requires the shared critic latent"
    assert max(nextlat_horizons) < args.num_steps
    assert max(bellman_horizons) <= args.num_steps
    assert args.nextlat_huber_beta > 0.0
    assert args.nextlat_trunk_grad_clip > 0.0
    assert args.nextlat_predictor_grad_clip > 0.0
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope in ("batch", "batch_retstd")), \
        "norm_adv_scope=batch/batch_retstd requires adv_transform_scope=batch"
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Ampere+ TensorFloat-32 accelerates the many FP32 trunk/predictor GEMMs while
    # retaining FP32 accumulation. Distributional projections remain explicit FP32.
    torch.set_float32_matmul_precision("high")

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                args.normalize_reward,
                args.clip_reward,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    frozen_value_target = FrozenValueTarget(agent).to(device)
    assert not any(parameter.requires_grad for parameter in frozen_value_target.parameters())
    # Two optimizers intentionally overlap on the shared actor trunk. Their moment
    # estimates never mix: task Adam sees actor+critic gradients; predictive Adam sees
    # NextLat only. Predictor-only parameters are absent from task Adam.
    task_optimizer = optim.Adam(agent.task_parameters(), lr=args.learning_rate, eps=1e-5)
    predictive_optimizer = optim.Adam(
        agent.nextlat_parameters(), lr=args.learning_rate, eps=1e-5
    )
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    nextlat_params = agent.nextlat_parameters()
    nextlat_trunk_blocks = agent.nextlat_trunk_parameter_blocks()
    nextlat_trunk_params = [
        parameter for block in nextlat_trunk_blocks for parameter in block
    ]
    nextlat_predictor_params = agent.nextlat_predictor_parameters()
    assert {id(p) for p in nextlat_params} == {
        id(p) for p in nextlat_trunk_params + nextlat_predictor_params
    }

    # Optional one-step causal meta-gate. Keep it entirely absent on the default-off
    # path; when enabled, its state follows the same few architectural blocks as update
    # admission instead of allocating and reducing hundreds of parameter tensors.
    if args.predadam_meta_gate:
        meta_previous = [
            block[0].new_zeros(sum(parameter.numel() for parameter in block))
            for block in nextlat_trunk_blocks
        ]
        meta_cross = [block[0].new_zeros(()) for block in nextlat_trunk_blocks]
        meta_pred_sq = [block[0].new_zeros(()) for block in nextlat_trunk_blocks]
    else:
        meta_previous = meta_cross = meta_pred_sq = None
    predadam_steps = 0

    def policy_rollout_fn(obs_):
        return agent.get_action_and_value(obs_)

    def policy_update_fn(obs_, z_):
        return agent.get_action_and_value(obs_, z_)

    if args.compile:
        policy_rollout_fn = torch.compile(
            policy_rollout_fn, mode=args.compile_mode, dynamic=False
        )
        policy_update_fn = torch.compile(
            policy_update_fn, mode=args.compile_mode, dynamic=False
        )
        print(f"compiled agent forward paths ({args.compile_mode})")

    # Soft max-entropy temperature (gaussian only). log_alpha is learned so the
    # policy-only future-entropy bonus AND the actor entropy bonus self-tune to
    # hold the SQUASHED entropy at target_entropy via SAC's temperature dual.
    auto_alpha = args.auto_entropy and args.actor_dist == "gaussian"
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    support_min, support_max = value_support_bounds(args)
    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
    )
    support = hl_support.support                       # Dreamer3 raw bucket centers
    bin_width = hl_support.bin_width
    raw_support = support

    def value_logits_to_scalar(logits):
        return hl_support.to_scalar(logits)

    # Static CUDA label programs. They are invoked once per rollout and remain resident
    # across all PPO epochs. The dense HL-Gauss projector compiles cleanly as a full graph;
    # the Bellman mask recurrence stays eager because current Inductor versions can
    # miscompile/fail codegen for its mixed bool/scatter graph. Its operations are still
    # entirely CUDA-resident and contain no data-dependent Python branch.
    def critic_label_fn(return_mtp_):
        return hl_support.project(return_mtp_)

    def bellman_label_fn(
        rewards_,
        terminations_,
        boundaries_,
        transition_valids_,
        next_value_probabilities_,
        next_values_,
    ):
        return _build_nstep_distributional_targets_impl(
            rewards_,
            terminations_,
            boundaries_,
            transition_valids_,
            next_value_probabilities_,
            next_values_,
            support,
            args.gamma,
            args.gae_lambda,
            bellman_horizons,
        )

    if args.compile:
        critic_label_fn = torch.compile(
            critic_label_fn, mode=args.compile_mode, dynamic=False, fullgraph=True
        )
        print(f"compiled CUDA critic-label path ({args.compile_mode})")

    sigma_floor = args.sigma_floor_bins * bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    latents = torch.zeros((args.num_steps, args.num_envs, args.hidden)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            task_optimizer.param_groups[0]["lr"] = lrnow
            predictive_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                action, z, logprob, ent, value_logits, actor_feat = policy_rollout_fn(next_obs)
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
                latents[step] = actor_feat
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obses[step] = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # One immutable semantic frame for all targets and decoder calls in this PPO
        # update. It is synchronized after collection and before any optimizer step.
        frozen_value_target.sync_from(agent)
        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            frozen_next_value_probs = frozen_value_target.value_probabilities(
                next_obses.reshape((-1,) + envs.single_observation_space.shape)
            ).reshape(args.num_steps, args.num_envs, args.num_bins)
            # The frozen target was synchronized immediately above, so these probabilities
            # are exactly the pre-update critic snapshot. Decode them once for GAE and the
            # lambda operator instead of running the expensive live MoE trunk a second time.
            next_transition_values = hl_support.probs_to_scalar(
                frozen_next_value_probs
            )
            # The target encoder is immutable for all PPO epochs. Encode the rollout
            # once instead of repeating five expensive MoE trunk forwards per minibatch
            # (root plus horizons 1/2/4/8). The table is only ~8 MiB at the defaults.
            frozen_obs_latents = frozen_value_target.encode(
                obs.reshape((-1,) + envs.single_observation_space.shape)
            ).reshape(args.num_steps, args.num_envs, args.hidden)
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample a' ~ π(·|s_T) for the bootstrap entropy (SAC's single-sample).
                _, _, boot_logprob, _, _, _ = agent.get_action_and_value(next_obses[-1])
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                next_value_bonus = None
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). Unbiased (state-dependent baseline); its large
            # magnitude is harmless — rankgauss at the optim step rank-normalizes it while
            # PRESERVING the entropy-induced reordering (the actual exploration signal).
            if auto_alpha:
                policy_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                    lambda_nonterminal = 1.0 - transition_boundaries[t]
                    delta = (
                        rewards[t]
                        + args.gamma
                        * (next_transition_values[t] + next_value_bonus[t])
                        * bootstrap_nonterminal
                        - values[t]
                    )
                    policy_adv[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                    )
            else:
                policy_adv = advantages
            # Batch-level percentile advantage normalization (scopes "ema" and "batch"). Both compute the
            # whole-rollout P5/P95 once and scale policy_adv by one S; "ema" smooths the percentiles with a
            # global EMA across iterations (v1), "batch" uses the FRESH per-rollout spread (no EMA -- the
            # batch-vs-mb ablation). scope=="minibatch" SKIPS this and scales fresh per-mb in the update loop,
            # leaving policy_adv RAW here. Divide-only; critic target `returns` stays RAW (valnorm=none).
            if args.ret_percnorm and args.ret_perc_scope in ("ema", "batch"):
                flat_ret = returns.reshape(-1)
                qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=device)
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                if args.ret_perc_scope == "ema":
                    if not ema_perc_inited:
                        ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
                    else:
                        r = args.ret_perc_rate
                        ema_ret_lo += r * (lo - ema_ret_lo)
                        ema_ret_hi += r * (hi - ema_ret_hi)
                    spread = ema_ret_hi - ema_ret_lo
                else:  # "batch": fresh whole-rollout percentile spread, no EMA
                    spread = hi - lo
                ret_perc_scale = max(args.ret_perc_floor, spread)
                policy_adv = policy_adv / ret_perc_scale
            # v162 critic target: scalar-return HL-Gauss MTP. Horizon 0 regresses
            # returns[t]; horizon h regresses returns[t+h] from the same features.
            # A future target is valid only when no reset boundary lies between
            # the source state and target state, and when it stays inside rollout.
            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros(
                (*returns.shape, mtp), dtype=torch.bool, device=returns.device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=returns.device
                )
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                return_mtp[:valid_len, :, h] = returns[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            # Project the ~383 MiB table directly on CUDA. A stable clone insulates labels
            # retained across all PPO epochs from CUDA-graph output-buffer reuse.
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            target_probs_graph = critic_label_fn(return_mtp.detach())
            target_probs = (
                target_probs_graph.clone() if args.compile else target_probs_graph
            )
            del target_probs_graph
            # Critic-consistent distributional lambda-return targets Z(s_j) for every
            # state j. Later,
            # imagined h_hat_{t+i} selects row j=t+i, so reward prefixes always begin at
            # the state represented by the latent (never at the source state t).
            if args.nextlat:
                bellman_targets_fp32, bellman_masks = bellman_label_fn(
                    rewards.detach(),
                    transition_terminations.detach(),
                    transition_boundaries.detach(),
                    transition_valids.detach(),
                    frozen_next_value_probs.detach(),
                    next_transition_values.detach(),
                )
                # This immutable table is ~96 MiB at defaults in fp16. Keep it on CUDA
                # beside the critic labels. Four imagined horizons consume its rows per
                # minibatch. The dtype conversion halves its resident footprint.
                bellman_targets = bellman_targets_fp32.to(dtype=torch.float16)
                del bellman_targets_fp32
                label_storage_mib = (
                    target_probs.numel() * target_probs.element_size()
                    + bellman_targets.numel() * bellman_targets.element_size()
                    + return_mtp_mask.numel() * return_mtp_mask.element_size()
                    + bellman_masks.numel() * bellman_masks.element_size()
                ) / (1024**2)

                # A predicted latent may not cross an episode reset and must have a real
                # s_{t+i} inside the rollout. Bellman masks separately govern the suffix.
                nextlat_path_mask = torch.zeros(
                    (args.num_steps, args.num_envs, len(nextlat_horizons)),
                    dtype=torch.bool,
                    device=device,
                )
                for horizon_index, horizon in enumerate(nextlat_horizons):
                    valid_len = args.num_steps - horizon
                    if valid_len <= 0:
                        break
                    valid_i = torch.ones(
                        (valid_len, args.num_envs), dtype=torch.bool, device=device
                    )
                    for k in range(horizon):
                        valid_i &= transition_boundaries[k : k + valid_len] == 0
                    nextlat_path_mask[:valid_len, :, horizon_index] = valid_i
                latent_batch_std = latents.reshape(-1, args.hidden).std(dim=0).mean().item()
            # These full-rollout bootstrap distributions (~64 MiB) and scalar staging
            # buffers have served GAE/target construction; do not retain redundant device
            # storage throughout the PPO epochs.
            del frozen_next_value_probs, next_transition_values, return_mtp
            # Per-state return std probe from the OLD rollout Z(s_t), decoded to
            # raw return units. The default rankgauss path does not consume this.
            sigma = (value_probs * (raw_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(sigma_floor)
            # CDF-rank u in Dreamer3 bucket order; intervals are uniform in symlog
            # coordinate even though raw bucket centers are exponentially spaced.
            cdf_frac = hl_support.cdf_fraction(returns)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)
        b_u = u.reshape(-1)
        if args.nextlat:
            b_nextlat_path_mask = nextlat_path_mask.reshape(
                -1, len(nextlat_horizons)
            )
            b_frozen_obs_latents = frozen_obs_latents.reshape(-1, args.hidden)
            b_bellman_targets = bellman_targets.reshape(
                -1, len(bellman_horizons), args.num_bins
            )
            b_bellman_masks = bellman_masks.reshape(-1, len(bellman_horizons))
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            b_policy_adv_normed = b_policy_adv / b_returns.std().clamp(min=args.ret_perc_floor)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        # Minibatch telemetry stays on CUDA and synchronizes only once at iteration-end.
        clipfrac_sum = b_returns.new_zeros(())
        clipfrac_count = 0
        ret_perc_scale_log = b_returns.new_tensor(ret_perc_scale)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # Preserve the incumbent NumPy shuffle exactly, but upload it once per epoch.
            # Reusing CUDA indices avoids dozens of implicit NumPy-index transfers in each
            # minibatch, including all imagined-action and Bellman-label gathers.
            b_inds_device = torch.as_tensor(b_inds, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds_device[start:end]

                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                _, _, newlogprob, entropy, value_logits, mb_actor_feat = policy_update_fn(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac_sum = clipfrac_sum + (
                        (ratio - 1.0).abs() > args.clip_coef
                    ).float().mean()
                    clipfrac_count += 1

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_advantages = mb_advantages / b_returns[mb_inds].std().clamp(min=args.ret_perc_floor)
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds]
                    qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device)
                    lo, hi = torch.quantile(mb_ret, qs)
                    mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                    mb_advantages = mb_advantages / mb_perc_scale
                    ret_perc_scale_log = mb_perc_scale.detach()

                # PMPO-style asymmetric pos/neg weighting: scale positive-advantage
                # samples by 2*alpha and negatives by 2*(1-alpha) (alpha=0.5 => identity).
                # alpha>0.5 emphasizes reinforcing good actions over suppressing bad ones.
                # Split on the SHAPED advantage's sign (pre-norm = the true advantage sign).
                if args.pos_neg_alpha != 0.5:
                    mb_advantages = mb_advantages * torch.where(
                        mb_raw_adv >= 0,
                        2.0 * args.pos_neg_alpha,
                        2.0 * (1.0 - args.pos_neg_alpha),
                    )

                # Asymmetric "clip-higher" when clip_coef_high is set: looser upper bound
                # gives positive-advantage actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # v162 HL-Gauss MTP value loss: per-horizon CE to scalar-return
                # targets, summed across valid horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds]
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(dtype=value_ce.dtype)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

                # Bellman NextLat: h_hat_i represents s_{t+i}, so every target prefix
                # starts at t+i. Decoder/target weights are iteration-frozen; gradients
                # pass through them only to h_hat_i, then to source trunk and dynamics.
                if args.nextlat:
                    h_hat = mb_actor_feat
                    target_root = b_frozen_obs_latents[mb_inds]
                    horizon_to_column = {
                        horizon: column
                        for column, horizon in enumerate(nextlat_horizons)
                    }
                    bellman_numerator = value_logits.new_zeros(())
                    bellman_entropy_numerator = value_logits.new_zeros(())
                    bellman_count = value_logits.new_zeros(())
                    bellman_by_n_numerator = value_logits.new_zeros(
                        len(bellman_horizons)
                    )
                    bellman_by_n_count = value_logits.new_zeros(
                        len(bellman_horizons)
                    )
                    latent_numerator = value_logits.new_zeros(())
                    latent_count = value_logits.new_zeros(())
                    direct_kl_numerator = value_logits.new_zeros(())
                    direct_kl_count = value_logits.new_zeros(())
                    movement_scales, latent_nmses = [], []

                    for rollout_step in range(1, max(nextlat_horizons) + 1):
                        action_indices = shifted_flat_indices(
                            mb_inds,
                            rollout_step - 1,
                            args.num_envs,
                            args.num_steps,
                        )
                        h_hat, _ = agent.nextlat_predictor(
                            h_hat, b_actions[action_indices].detach()
                        )
                        if rollout_step not in horizon_to_column:
                            continue

                        column = horizon_to_column[rollout_step]
                        target_indices = shifted_flat_indices(
                            mb_inds,
                            rollout_step,
                            args.num_envs,
                            args.num_steps,
                        )
                        path_mask = b_nextlat_path_mask[mb_inds, column].to(
                            device=device, dtype=value_logits.dtype
                        )
                        path_count = path_mask.sum()
                        with torch.no_grad():
                            target_latent = b_frozen_obs_latents[target_indices]
                            true_logits = frozen_value_target.decode_imagined(
                                target_latent
                            )
                            true_log_probs = torch.log_softmax(true_logits, dim=-1)
                            true_probs = true_log_probs.exp()

                        # Residual-unit normalization prevents both latent shrinkage and
                        # temporally sluggish representations from making the anchor easy.
                        movement = target_latent - target_root
                        movement_mse = (
                            movement.square().mean(-1) * path_mask
                        ).sum() / path_count.clamp_min(1.0)
                        root_rms = target_root.square().mean().sqrt()
                        movement_scale = movement_mse.clamp_min(0.0).sqrt().clamp_min(
                            1e-3 * root_rms + 1e-12
                        )
                        latent_rows = F.smooth_l1_loss(
                            h_hat / movement_scale,
                            target_latent / movement_scale,
                            reduction="none",
                            beta=args.nextlat_huber_beta,
                        ).mean(-1)
                        latent_numerator = latent_numerator + (
                            latent_rows * path_mask
                        ).sum()
                        latent_count = latent_count + path_count
                        movement_scales.append(movement_scale.detach())
                        with torch.no_grad():
                            prediction_mse = (
                                (h_hat - target_latent).square().mean(-1) * path_mask
                            ).sum() / path_count.clamp_min(1.0)
                            latent_nmses.append(
                                prediction_mse
                                / movement_mse.clamp_min(1e-8 * root_rms.square() + 1e-12)
                            )

                        predicted_logits = frozen_value_target.decode_imagined(h_hat)
                        predicted_log_probs = torch.log_softmax(
                            predicted_logits, dim=-1
                        )
                        direct_kl_rows = (
                            true_probs
                            * (true_log_probs - predicted_log_probs)
                        ).sum(-1)
                        direct_kl_numerator = direct_kl_numerator + (
                            direct_kl_rows * path_mask
                        ).sum()
                        direct_kl_count = direct_kl_count + path_count

                        target_distributions = b_bellman_targets[
                            target_indices
                        ].to(dtype=predicted_log_probs.dtype)
                        suffix_mask = b_bellman_masks[target_indices].to(
                            dtype=predicted_log_probs.dtype
                        )
                        pair_mask = path_mask.unsqueeze(-1) * suffix_mask
                        cross_entropy = -(
                            target_distributions
                            * predicted_log_probs.unsqueeze(1)
                        ).sum(-1)
                        bellman_target_entropy = -(
                            target_distributions
                            * target_distributions.clamp_min(1e-8).log()
                        ).sum(-1)
                        bellman_numerator = bellman_numerator + (
                            cross_entropy * pair_mask
                        ).sum()
                        bellman_entropy_numerator = bellman_entropy_numerator + (
                            bellman_target_entropy * pair_mask
                        ).sum()
                        bellman_count = bellman_count + pair_mask.sum()
                        bellman_by_n_numerator = bellman_by_n_numerator + (
                            cross_entropy * pair_mask
                        ).sum(0)
                        bellman_by_n_count = bellman_by_n_count + pair_mask.sum(0)

                    nextlat_bellman_loss = (
                        bellman_numerator / bellman_count.clamp_min(1.0)
                    )
                    nextlat_bellman_kl = (
                        bellman_numerator - bellman_entropy_numerator
                    ) / bellman_count.clamp_min(1.0)
                    nextlat_bellman_by_n = bellman_by_n_numerator / (
                        bellman_by_n_count.clamp_min(1.0)
                    )
                    nextlat_latent_loss = (
                        latent_numerator / latent_count.clamp_min(1.0)
                    )
                    nextlat_direct_kl = direct_kl_numerator / direct_kl_count.clamp_min(
                        1.0
                    )
                    nextlat_movement_scale = torch.stack(movement_scales).mean()
                    nextlat_latent_nmse = torch.stack(latent_nmses).mean()
                    nextlat_loss = (
                        args.nextlat_bellman_coef * nextlat_bellman_loss
                        + args.nextlat_latent_coef * nextlat_latent_loss
                    )
                else:
                    nextlat_loss = None

                entropy_loss = entropy.mean()

                if auto_alpha:
                    # SAC's temperature dual (sac_continuous_action.py), on the
                    # SQUASHED log-prob: alpha_loss = (-α·(logπ + target_entropy)).mean().
                    # With target_entropy=-|A|, drives E[logπ_squashed] -> |A|,
                    # equivalently E[-logπ_squashed] -> -|A|.
                    # The SAME α weights the explicit CURRENT-step actor entropy bonus below
                    # (the soft return's current-state entropy is action-independent => zero
                    # in the PG term, so the bonus supplies the actual entropy gradient).
                    ent_coef_eff = log_alpha.exp().detach()
                    alpha_loss = (-log_alpha.exp() * (newlogprob.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    ent_coef_eff = args.ent_coef

                # Three backwards, two Adam streams. Actor and critic are clipped and
                # summed before TASK Adam exactly as in the incumbent. NextLat is clipped
                # separately and saved for PREDICTIVE Adam, whose trunk proposal is
                # admitted only after seeing the task optimizer's actual parameter delta.
                agent.zero_grad(set_to_none=True)
                (args.vf_coef * v_loss).backward(retain_graph=True)
                critic_gn = clip_grad_norm_async_(
                    critic_params, args.critic_grad_clip, "critic"
                )
                value_grads = {
                    p: p.grad.detach().clone()
                    for p in critic_params
                    if p.grad is not None
                }

                if args.nextlat:
                    agent.zero_grad(set_to_none=True)
                    (args.nextlat_coef * nextlat_loss).backward(retain_graph=True)
                    nextlat_trunk_gn = clip_grad_norm_async_(
                        nextlat_trunk_params,
                        args.nextlat_trunk_grad_clip,
                        "NextLat trunk",
                    )
                    nextlat_predictor_gn = clip_grad_norm_async_(
                        nextlat_predictor_params,
                        args.nextlat_predictor_grad_clip,
                        "NextLat predictor",
                    )
                    nextlat_gn = torch.sqrt(
                        nextlat_trunk_gn.square() + nextlat_predictor_gn.square()
                    )
                    nextlat_grads = {
                        p: p.grad.detach().clone()
                        for p in nextlat_params
                        if p.grad is not None
                    }
                else:
                    nextlat_grads = {}

                agent.zero_grad(set_to_none=True)
                (pg_loss - ent_coef_eff * entropy_loss).backward()
                actor_gn = clip_grad_norm_async_(
                    actor_params, args.actor_grad_clip, "actor"
                )
                # Preserve separately clipped task gradients in a handful of contiguous
                # architectural blocks. Admission uses these actual loss gradients, not
                # Adam's preconditioned aggregate delta, for first-order protection.
                if args.nextlat:
                    actor_gradient_blocks = [
                        _flatten_tensors(
                            [
                                parameter.grad
                                if parameter.grad is not None
                                else torch.zeros_like(parameter)
                                for parameter in block
                            ]
                        )
                        for block in nextlat_trunk_blocks
                    ]
                    critic_gradient_blocks = [
                        _flatten_tensors(
                            [
                                value_grads[parameter]
                                if parameter in value_grads
                                else torch.zeros_like(parameter)
                                for parameter in block
                            ]
                        )
                        for block in nextlat_trunk_blocks
                    ]
                for parameter, gradient in value_grads.items():
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad + gradient
                    )

                if args.nextlat:
                    with torch.no_grad():
                        task_before_blocks = [
                            _flatten_tensors(block) for block in nextlat_trunk_blocks
                        ]
                task_optimizer.step()
                if args.nextlat:
                    with torch.no_grad():
                        task_updates = [
                            _flatten_tensors(block) - before
                            for block, before in zip(
                                nextlat_trunk_blocks, task_before_blocks
                            )
                        ]

                if args.nextlat:
                    # Update optional causal forecast statistics using only PAST
                    # proposals. The default path performs none of this work.
                    if args.predadam_meta_gate:
                        gates = []
                        decay = args.predadam_meta_decay
                        for block_index, task_update in enumerate(task_updates):
                            previous = meta_previous[block_index]
                            observed_cross = (previous * task_update).sum()
                            observed_sq = previous.square().sum()
                            meta_cross[block_index].mul_(decay).add_(
                                observed_cross, alpha=1.0 - decay
                            )
                            meta_pred_sq[block_index].mul_(decay).add_(
                                observed_sq, alpha=1.0 - decay
                            )
                            if predadam_steps >= args.predadam_meta_warmup:
                                gate = torch.clamp(
                                    meta_cross[block_index]
                                    / meta_pred_sq[block_index].clamp_min(1e-20),
                                    min=0.0,
                                    max=1.0,
                                )
                            else:
                                gate = task_update.new_ones(())
                            gates.append(gate)
                    else:
                        gates = None

                    predictive_gradients = [
                        (
                            nextlat_grads[parameter]
                            if parameter in nextlat_grads
                            else torch.zeros_like(parameter)
                        )
                        for parameter in nextlat_params
                    ]
                    (
                        raw_predictive_updates,
                        admitted_updates,
                        predadam_predictor_step_norm,
                        predadam_stats,
                    ) = apply_predictive_optimizer_transaction(
                        nextlat_trunk_blocks,
                        nextlat_predictor_params,
                        predictive_optimizer,
                        predictive_gradients,
                        task_updates,
                        args.predadam_trust_ratio,
                        actor_gradients=actor_gradient_blocks,
                        critic_gradients=critic_gradient_blocks,
                        gates=gates,
                    )
                    with torch.no_grad():
                        if args.predadam_meta_gate:
                            for previous, update in zip(
                                meta_previous, raw_predictive_updates
                            ):
                                previous.copy_(update)
                        predadam_gate_mean = (
                            torch.stack(gates).mean()
                            if gates is not None
                            else task_updates[0].new_ones(())
                        )
                    predadam_steps += 1
                else:
                    zero_stat = torch.zeros((), device=device)
                    predadam_stats = {
                        "task_norm": zero_stat,
                        "predictive_norm": zero_stat,
                        "raw_cosine": zero_stat,
                        "cap_scale": zero_stat,
                        "accepted_fraction": zero_stat,
                        "admitted_norm": zero_stat,
                        "actor_first_order": zero_stat,
                        "critic_first_order": zero_stat,
                        "actor_conflict_fraction": zero_stat,
                        "critic_conflict_fraction": zero_stat,
                        "max_block_ratio": zero_stat,
                        "numeric_valid": zero_stat,
                    }
                    predadam_predictor_step_norm = zero_stat
                    predadam_gate_mean = zero_stat

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_per_h = b_target_probs[..., 0] + b_target_probs[..., -1]
        edge_mask_f = b_target_mask.to(dtype=edge_per_h.dtype)
        edge_mass = ((edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)).item()
        if args.nextlat:
            # Reduce directly from resident labels. Avoid materializing float32 copies of
            # either full table solely for diagnostics.
            bellman_valid_count = bellman_masks.sum()
            bellman_edge_sum = (
                (bellman_targets[..., 0] + bellman_targets[..., -1])
                * bellman_masks
            ).sum(dtype=torch.float32)
            bellman_edge_mass = (
                bellman_edge_sum / bellman_valid_count.clamp_min(1)
            ).item()
            bellman_valid_fraction = bellman_masks.float().mean().item()
            path_valid_fraction = nextlat_path_mask.float().mean().item()

        writer.add_scalar(
            "charts/learning_rate", task_optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar(
            "charts/ret_perc_scale", float(ret_perc_scale_log), global_step
        )
        if args.nextlat:
            writer.add_scalar("nextlat/total_loss", nextlat_loss.item(), global_step)
            writer.add_scalar(
                "nextlat/bellman_cross_entropy",
                nextlat_bellman_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "nextlat/bellman_kl",
                nextlat_bellman_kl.item(),
                global_step,
            )
            writer.add_scalar(
                "nextlat/latent_anchor_loss",
                nextlat_latent_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "nextlat/direct_value_kl_diagnostic",
                nextlat_direct_kl.item(),
                global_step,
            )
            writer.add_scalar(
                "nextlat/latent_nmse_vs_identity",
                nextlat_latent_nmse.item(),
                global_step,
            )
            writer.add_scalar(
                "nextlat/movement_scale",
                nextlat_movement_scale.item(),
                global_step,
            )
            writer.add_scalar(
                "nextlat/trunk_raw_grad_norm", float(nextlat_trunk_gn), global_step
            )
            writer.add_scalar(
                "nextlat/predictor_raw_grad_norm",
                float(nextlat_predictor_gn),
                global_step,
            )
            writer.add_scalar("nextlat/raw_grad_norm", float(nextlat_gn), global_step)
            writer.add_scalar("bellman/target_edge_mass", bellman_edge_mass, global_step)
            writer.add_scalar(
                "bellman/target_valid_fraction", bellman_valid_fraction, global_step
            )
            writer.add_scalar(
                "bellman/path_valid_fraction", path_valid_fraction, global_step
            )
            writer.add_scalar(
                "bellman/device_label_storage_mib", label_storage_mib, global_step
            )
            writer.add_scalar("debug/latent_batch_std", latent_batch_std, global_step)
            for horizon_index, horizon in enumerate(bellman_horizons):
                writer.add_scalar(
                    f"bellman/cross_entropy_n{horizon}",
                    nextlat_bellman_by_n[horizon_index].item(),
                    global_step,
                )
            writer.add_scalar(
                "predadam/task_step_norm",
                float(predadam_stats["task_norm"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/raw_predictive_step_norm",
                float(predadam_stats["predictive_norm"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/admitted_step_norm",
                float(predadam_stats["admitted_norm"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/predictor_step_norm",
                float(predadam_predictor_step_norm),
                global_step,
            )
            writer.add_scalar(
                "predadam/raw_task_cosine",
                float(predadam_stats["raw_cosine"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/accepted_fraction",
                float(predadam_stats["accepted_fraction"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/cap_scale",
                float(predadam_stats["cap_scale"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/actor_first_order",
                float(predadam_stats["actor_first_order"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/critic_first_order",
                float(predadam_stats["critic_first_order"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/actor_conflict_fraction",
                float(predadam_stats["actor_conflict_fraction"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/critic_conflict_fraction",
                float(predadam_stats["critic_conflict_fraction"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/max_block_ratio",
                float(predadam_stats["max_block_ratio"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/numeric_valid",
                float(predadam_stats["numeric_valid"]),
                global_step,
            )
            writer.add_scalar(
                "predadam/meta_gate_mean", float(predadam_gate_mean), global_step
            )
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
            writer.add_scalar("debug/soft_adv_std_ratio", (policy_adv.std() / (advantages.std() + 1e-8)).item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar(
            "losses/clipfrac",
            float(clipfrac_sum / max(clipfrac_count, 1)),
            global_step,
        )
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)
        writer.add_scalar("debug/u_edge_frac", u_edge_frac, global_step)
        writer.add_scalar("debug/sigma_mean", b_sigma.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

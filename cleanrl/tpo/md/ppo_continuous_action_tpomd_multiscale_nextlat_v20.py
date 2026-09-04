# TPO-MD MULTISCALE NEXTLAT v20: isolate prediction topology and temporal scale.
#
# TPO-MD and v17's same-task-Adam gradient composition remain exact. ``recursive`` is
# v17's shared one-step absolute-latent model; ``direct`` predicts each endpoint from the
# source plus its ordered behavior-action prefix; ``hybrid`` combines local recursion with
# direct distant endpoints; ``off`` is exact TPO-MD. All use one frozen pre-update teacher,
# a frozen policy decoder, boundary masks, equal valid-normalized horizon weighting, and a
# single globally clipped auxiliary-trunk budget. A compiled static predictor wrapper
# batches direct endpoints and graph-unrolls recursion. The direct default (H=1,2,3,4,
# raw loss) is the causal recursive-vs-direct intervention; log-spaced horizons are opt-in.
# Exact recursive parity is intentionally limited to H<=4: repeatedly applying v17's
# ReLU-squared map at observed latent scales can overflow by H=8--16, so stable long
# recurrence is a separate residual-dynamics experiment rather than a valid depth control.
# Falsifier: direct is supported only if distant normalized error improves while source and
# action-prefix replacements remain harmful, latent rank/scale and tail CVaR stay healthy, and return
# improves at fixed trunk budget. A lower raw prediction loss alone is not evidence.
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from math import ceil, log
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport, symlog, symexp

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))
MAX_EXACT_RECURSIVE_HORIZON = 4


@torch.no_grad()
def clip_grad_norm_async_fail_loud_(parameters, max_norm, norm_type=2.0):
    """Clip exactly as PyTorch does and enqueue the finite check on-device."""
    total_norm = nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=False,
    )
    torch._assert_async(
        torch.isfinite(total_norm),
        "The total gradient norm is non-finite; refusing the optimizer step",
    )
    return total_norm


@torch.no_grad()
def synchronize_scalar_telemetry(statistics):
    """Materialize CUDA scalar telemetry with one packed device-to-host copy."""
    if not statistics:
        return {}
    names = tuple(statistics)
    scalars = tuple(statistics.values())
    if not all(torch.is_tensor(value) for value in scalars):
        raise TypeError("telemetry values must be tensors")
    if not all(value.numel() == 1 for value in scalars):
        raise ValueError("telemetry values must be scalar tensors")
    if len({value.device for value in scalars}) != 1:
        raise ValueError("telemetry values must share one device")
    host_values = torch.stack(
        [value.detach().reshape(()) for value in scalars]
    ).cpu().tolist()
    return dict(zip(names, host_values))


@torch.no_grad()
def update_scalar_max_(maximum, value):
    """Accumulate a scalar maximum on-device without materializing either value."""
    torch.maximum(maximum, value.detach(), out=maximum)
    return maximum


@torch.no_grad()
def retain_graph_output(output, *, compiled):
    """Detach and clone output whose CUDA-graph storage will be replayed."""
    if not torch.is_tensor(output):
        raise TypeError("retained graph output must be a tensor")
    return output.detach().clone() if compiled else output.detach()


def value_support_bounds(args):
    """Support bounds in the coordinate used for categorical bins."""
    return args.v_min, args.v_max


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


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
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    # NOTE: target_kl epoch-stop would starve the (always-on) critic in this pure
    # variant; default None — the actor is leashed by tpo_kl_breaker instead.
    target_kl: Optional[float] = None

    # TPO mirror descent: probe-scored TPO target with MPO-style adaptive
    # temperature REPLACES the PPO surrogate. Probes run at EVERY rollout state.
    tpo_coef: float = 1.0        # weight of the TPO CE (the entire actor loss besides entropy); must be > 0
    tpo_eta: float = 6.0         # FIXED temperature, used only when tpo_adaptive_eta=False
    tpo_k: int = 8               # candidates per state (ALL probed, incl. the executed action as candidate 0)
    tpo_sigma_scale_coef: float = 1.0  # global score sigma = coef * EMA(one-step TD-residual RMS)
    tpo_eps: float = 0.03        # trust-region CAP / max KL per update (dyn-trust) OR fixed KL target (v1 mode)
    tpo_adaptive_eta: bool = True      # solve eta s.t. mean KL(p_old||q)=tpo_eps; False => fixed tpo_eta
    tpo_dyn_trust: bool = True   # one-sided KL cap on a fixed base temperature (v5 default). False => exact tpomd_v1 fixed-target dual
    tpo_eta_base: float = 1.0    # base temperature for the dynamic-cap path; natural KL at this eta is the signal-determined step (unused when tpo_dyn_trust=False)
    tpo_kl_breaker: float = 0.09 # actor circuit breaker: stop actor epochs when epoch-mean approx_kl exceeds (3x eps)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # v149-aligned distributional critic support. Bounds are already symlog
    # coordinates for raw-return support [-20000, 20000].
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 0.5  # requested sharper HL-Gauss projection sigma

    # Raw-return ablation: keep observations as in the source, but do not divide
    # rewards by NormalizeReward's running discounted-return std and do not clip
    # raw rewards before GAE.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "rankgauss"

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "minibatch"

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

    # Multiscale NextLat. ``nextlat_horizons`` is a comma-separated ordered set.
    # recursive with 1,2,3,4 is exact v17 semantics; ``recursive_depth`` is a
    # convenience override (0 leaves the explicit list untouched).
    pc_mode: str = "direct"  # off | recursive | direct | hybrid
    nextlat_horizons: str = "1,2,3,4"
    nextlat_recursive_depth: int = 0
    nextlat_hybrid_recursive_max: int = 4
    nextlat_horizon_embed_dim: int = 8
    nextlat_direct_hidden: int = 0  # 0 = auto-match the v17 recursive parameter budget
    nextlat_loss_scale: str = "raw"  # raw | target_std
    nextlat_shuffle_diagnostics: bool = False
    nextlat_coef: float = 1.0
    nextlat_kl_coef: float = 1.0
    nextlat_trunk_grad_clip: float = 0.025
    nextlat_predictor_grad_clip: float = 0.25
    nextlat_target_batch_size: int = 8192

    # Non-learning collapse telemetry. Thresholds are comma-separated returns;
    # defaults target HalfCheetah and can be changed for Hopper/Walker.
    tail_risk_window: int = 512
    tail_risk_thresholds: tuple[float, ...] = (1500.0, 5000.0)

    # reduce-overhead enables CUDA graphs for every static neural forward. Raw MuJoCo
    # probes and all TPO control flow remain eager and behavior-equivalent to v13.
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
        clipped_obs_space = gym.spaces.Box(
            low=np.full(env.observation_space.shape, -10.0, dtype=env.observation_space.dtype),
            high=np.full(env.observation_space.shape, 10.0, dtype=env.observation_space.dtype),
            dtype=env.observation_space.dtype,
        )
        try:
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.clip(obs, -10, 10),
                observation_space=clipped_obs_space,
            )
        except TypeError:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def find_wrapper(env, wrapper_type):
    # Walk the .env wrapper chain looking for wrapper_type.
    cur = env
    while cur is not None:
        if isinstance(cur, wrapper_type):
            return cur
        cur = getattr(cur, "env", None)
    return None


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


@dataclass(frozen=True)
class NextLatSpec:
    mode: str
    horizons: tuple[int, ...]
    recursive_horizons: tuple[int, ...]
    direct_horizons: tuple[int, ...]

    @property
    def enabled(self):
        return self.mode != "off"

    @property
    def max_horizon(self):
        return max(self.horizons, default=0)


def parse_nextlat_horizons(value, recursive_depth=0):
    """Parse a deterministic, strictly increasing positive horizon tuple."""
    if recursive_depth < 0:
        raise ValueError("nextlat_recursive_depth must be non-negative")
    if recursive_depth:
        return tuple(range(1, recursive_depth + 1))
    if isinstance(value, str):
        fields = [field.strip() for field in value.split(",")]
        if not fields or any(not field for field in fields):
            raise ValueError("nextlat_horizons must be a comma-separated integer list")
        try:
            horizons = tuple(int(field) for field in fields)
        except ValueError as error:
            raise ValueError("nextlat_horizons must contain only integers") from error
    else:
        horizons = tuple(int(horizon) for horizon in value)
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("nextlat horizons must be non-empty and positive")
    if any(left >= right for left, right in zip(horizons, horizons[1:])):
        raise ValueError("nextlat horizons must be unique and strictly increasing")
    return horizons


def resolve_nextlat_spec(mode, horizons, recursive_depth=0, hybrid_recursive_max=4):
    if mode not in ("off", "recursive", "direct", "hybrid"):
        raise ValueError(f"unknown pc_mode {mode}")
    if mode == "off":
        return NextLatSpec(mode, (), (), ())
    if recursive_depth and mode != "recursive":
        raise ValueError("nextlat_recursive_depth is only valid in recursive mode")
    parsed = parse_nextlat_horizons(horizons, recursive_depth)
    if mode == "recursive":
        if max(parsed) > MAX_EXACT_RECURSIVE_HORIZON:
            raise ValueError(
                "exact v17 recursive prediction supports horizons at most 4; "
                "use direct/hybrid for longer targets"
            )
        return NextLatSpec(mode, parsed, parsed, ())
    if mode == "direct":
        return NextLatSpec(mode, parsed, (), parsed)
    if hybrid_recursive_max < 1:
        raise ValueError("nextlat_hybrid_recursive_max must be positive")
    recursive = tuple(horizon for horizon in parsed if horizon <= hybrid_recursive_max)
    direct = tuple(horizon for horizon in parsed if horizon > hybrid_recursive_max)
    if not recursive or not direct:
        raise ValueError("hybrid mode needs horizons on both sides of its recursive maximum")
    if max(recursive) > MAX_EXACT_RECURSIVE_HORIZON:
        raise ValueError(
            "the hybrid recursive branch supports horizons at most 4; "
            "assign longer targets to its direct branch"
        )
    return NextLatSpec(mode, parsed, recursive, direct)


def recursive_predictor_parameter_count(hidden, action_dim):
    return 2 * hidden * hidden + action_dim * hidden + 2 * hidden


def choose_direct_hidden(hidden, action_dim, max_horizon, embed_dim, override=0):
    """Nearest integer width to v17's shared recursive predictor parameter count."""
    if override < 0:
        raise ValueError("nextlat_direct_hidden must be non-negative")
    if override:
        return override
    input_dim = hidden + max_horizon * action_dim + embed_dim
    fixed = hidden + (max_horizon + 1) * embed_dim
    slope = input_dim + hidden + 1
    ideal = (recursive_predictor_parameter_count(hidden, action_dim) - fixed) / slope
    candidates = {max(1, int(ideal)), max(1, int(ideal) + 1)}
    target = recursive_predictor_parameter_count(hidden, action_dim)
    return min(
        candidates,
        key=lambda width: abs(width * slope + fixed - target),
    )


class DirectHorizonPredictor(nn.Module):
    """Predict endpoints independently from a source and padded behavior-action prefix."""

    def __init__(self, hidden, action_dim, horizons, embed_dim, direct_hidden=0):
        super().__init__()
        self.hidden = hidden
        self.action_dim = action_dim
        self.max_horizon = max(horizons)
        self.register_buffer(
            "horizons", torch.tensor(horizons, dtype=torch.int64), persistent=False
        )
        positions = torch.arange(self.max_horizon, dtype=torch.int64)
        self.register_buffer(
            "prefix_mask",
            positions[None, :] < self.horizons[:, None],
            persistent=False,
        )
        self.horizon_embedding = nn.Embedding(self.max_horizon + 1, embed_dim)
        width = choose_direct_hidden(
            hidden, action_dim, self.max_horizon, embed_dim, direct_hidden
        )
        input_dim = hidden + self.max_horizon * action_dim + embed_dim
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_dim, width)),
            ReLUSquared(),
            layer_init(nn.Linear(width, hidden), std=0.1),
        )

    def forward(self, source, future_actions):
        if future_actions.shape[0] != self.max_horizon:
            raise ValueError("future_actions must be padded to the configured max horizon")
        # (K, B, max_horizon, A); action slots beyond each endpoint are zeroed
        # inside the module, so they cannot leak into a shorter-horizon prediction.
        prefixes = future_actions.transpose(0, 1).unsqueeze(0).expand(
            self.horizons.numel(), -1, -1, -1
        )
        prefixes = torch.where(
            self.prefix_mask[:, None, :, None], prefixes, torch.zeros_like(prefixes)
        )
        prefixes = prefixes.flatten(start_dim=2)
        sources = source.unsqueeze(0).expand(self.horizons.numel(), -1, -1)
        embeddings = self.horizon_embedding(self.horizons)[:, None, :].expand(
            -1, source.shape[0], -1
        )
        return self.network(torch.cat((sources, prefixes, embeddings), dim=-1))


def multiscale_prediction_forward(agent, source, future_actions):
    """Static graphable prediction path: one batched direct call, unrolled recursion."""
    predictions = {}
    if agent.nextlat_spec.recursive_horizons:
        predicted = source
        requested = set(agent.nextlat_spec.recursive_horizons)
        for step in range(1, max(requested) + 1):
            predicted = agent.nextlat_predictor(
                torch.cat((predicted, future_actions[step - 1]), dim=-1)
            )
            if step in requested:
                predictions[step] = predicted
    if agent.nextlat_spec.direct_horizons:
        direct = agent.nextlat_direct_predictor(source, future_actions)
        for index, horizon in enumerate(agent.nextlat_spec.direct_horizons):
            predictions[horizon] = direct[index]
    if not predictions:
        return source.new_empty((0, source.shape[0], source.shape[1]))
    return torch.stack([predictions[horizon] for horizon in agent.nextlat_spec.horizons])


def valid_normalized_mean(per_example, mask):
    mask = mask.to(dtype=per_example.dtype)
    return (per_example * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def compute_rollout_target_spreads(target_features, masks):
    """Detached scalar std per horizon via fixed-shape weighted moments."""
    weights = masks.to(dtype=target_features.dtype).unsqueeze(-1)
    counts = weights.sum(dim=(1, 2)).clamp_min(1.0) * target_features.shape[-1]
    means = (target_features * weights).sum(dim=(1, 2)) / counts
    centered = target_features - means[:, None, None]
    variances = (centered.square() * weights).sum(dim=(1, 2)) / counts
    return variances.clamp_min(0.0).sqrt().clamp_min(1e-6).detach()


def compute_nextlat_loss(agent, predictions, targets, masks, target_spreads, scale_mode):
    """Equal-average, per-horizon valid-normalized latent and frozen-decoder loss."""
    if scale_mode not in ("raw", "target_std"):
        raise ValueError(f"unknown nextlat_loss_scale {scale_mode}")
    prediction_losses, policy_kls = [], []
    for index in range(predictions.shape[0]):
        prediction = predictions[index]
        target = targets[index]
        mask = masks[index]
        latent_error = F.smooth_l1_loss(
            prediction, target, reduction="none"
        ).mean(-1)
        if scale_mode == "target_std":
            # Keep prediction/target coordinates raw (and therefore preserve the
            # Smooth-L1 transition); the stopped rollout statistic is only a
            # scalar loss denominator.
            latent_error = latent_error / target_spreads[index]
        prediction_losses.append(valid_normalized_mean(latent_error, mask))
        with torch.no_grad():
            target_dist, _, _ = agent._actor_dist_frozen_head(target)
        predicted_dist, _, _ = agent._actor_dist_frozen_head(prediction)
        policy_kl = torch.distributions.kl_divergence(target_dist, predicted_dist).sum(-1)
        policy_kls.append(valid_normalized_mean(policy_kl, mask))
    prediction_losses = torch.stack(prediction_losses)
    policy_kls = torch.stack(policy_kls)
    return {
        "prediction": prediction_losses.mean(),
        "policy_kl": policy_kls.mean(),
        "per_horizon_prediction": prediction_losses,
        "per_horizon_policy_kl": policy_kls,
    }


@torch.no_grad()
def normalized_endpoint_errors_vs_persistence(predictions, source, targets, masks):
    """Endpoint MSE divided by source-latent persistence MSE, once per horizon."""
    normalized_errors = []
    for index in range(predictions.shape[0]):
        mask = masks[index]
        predicted_mse = valid_normalized_mean(
            (predictions[index] - targets[index]).square().mean(-1), mask
        )
        persistence_mse = valid_normalized_mean(
            (source - targets[index]).square().mean(-1), mask
        )
        normalized_errors.append(predicted_mse / persistence_mse.clamp_min(1e-8))
    return torch.stack(normalized_errors)


@torch.no_grad()
def latent_scale_and_participation_rank(features):
    """On-device latent RMS and participation rank (tr C)^2 / tr(C^2)."""
    centered = features - features.mean(dim=0, keepdim=True)
    scale = centered.square().mean().sqrt()
    covariance = centered.T @ centered / max(features.shape[0] - 1, 1)
    trace = covariance.diagonal().sum()
    trace_of_square = covariance.square().sum()
    participation_rank = trace.square() / trace_of_square.clamp_min(1e-12)
    return scale, participation_rank


def parse_tail_thresholds(value):
    try:
        fields = value.split(",") if isinstance(value, str) else value
        thresholds = tuple(float(field.strip() if isinstance(field, str) else field) for field in fields)
    except ValueError as error:
        raise ValueError("tail_risk_thresholds must be comma-separated numbers") from error
    if not thresholds or any(not np.isfinite(threshold) for threshold in thresholds):
        raise ValueError("tail_risk_thresholds must be finite and non-empty")
    return thresholds


def metric_number(value):
    return (
        format(float(value), ".17g")
        .replace("-", "neg")
        .replace(".", "p")
        .replace("+", "")
    )


def tail_risk_statistics(returns, thresholds):
    values = np.asarray(tuple(returns), dtype=np.float64)
    if values.size == 0:
        return {}
    ordered = np.sort(values)
    cvar = ordered[: max(1, ceil(0.05 * values.size))].mean()
    median = float(np.median(values))
    below_half_median_count = int((values < 0.5 * median).sum())
    statistics = {
        "tail_risk/window_size": float(values.size),
        "tail_risk/median": median,
        "tail_risk/bottom_5pct_mean": float(cvar),
        "tail_risk/cvar_05": float(cvar),
        "tail_risk/below_half_window_median_count": float(below_half_median_count),
        "tail_risk/below_half_window_median_fraction": below_half_median_count / values.size,
    }
    for threshold in thresholds:
        count = int((values < threshold).sum())
        suffix = metric_number(threshold)
        statistics[f"tail_risk/below_{suffix}_count"] = float(count)
        statistics[f"tail_risk/below_{suffix}_fraction"] = count / values.size
    return statistics


class IndexedTransferBranch(nn.Module):
    def __init__(self, H, history_dim):
        super().__init__()
        if history_dim % H != 0:
            raise ValueError(f"history_dim={history_dim} must be divisible by H={H}")
        self.H = H
        self.history_slots = history_dim // H
        self.current_linear = layer_init(nn.Linear(H, H))
        self.act = ReLUSquared()
        self.out_linear = layer_init(nn.Linear(H, H))
        self.history_weight = nn.Parameter(torch.empty(self.history_slots, H))
        nn.init.normal_(self.history_weight, mean=0.0, std=np.sqrt(2.0 / (H + self.history_slots)))

    def forward(self, x, history):
        preact = self.current_linear(x)
        history = history.reshape(history.shape[0], self.history_slots, self.H)
        same_index_transfer = (history * self.history_weight.to(dtype=history.dtype).unsqueeze(0)).sum(dim=1)
        return self.out_linear(self.act(preact + same_index_transfer))


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
        self.dense = IndexedTransferBranch(H, in_dim)

        # Soft MoE branch (softmax over all experts, no top-K).
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([IndexedTransferBranch(H, in_dim) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in), cat_feats)        # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = torch.stack([e(m_in, cat_feats) for e in self.experts], dim=1)  # (B, E, H)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)          # (B, H)

        return x_in + d_dense + d_moe


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
        self.nextlat_spec = resolve_nextlat_spec(
            args.pc_mode,
            args.nextlat_horizons,
            args.nextlat_recursive_depth,
            args.nextlat_hybrid_recursive_max,
        )
        self.nextlat_action_dim = act_dim
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # v149 critic readout style, without MTP: biasless HL-Gauss value head.
        self.num_bins = args.num_bins
        self.critic_head = layer_init(nn.Linear(H, args.num_bins, bias=False), std=0.1)
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
        # Isolate all auxiliary initialization from the task RNG stream. In ``off``
        # there are no auxiliary modules at all, so state_dict, forwards, task Adam,
        # and the post-Agent RNG state are exact TPO-MD.
        if self.nextlat_spec.enabled:
            with torch.random.fork_rng(devices=[]):
                if self.nextlat_spec.recursive_horizons:
                    self.nextlat_predictor = nn.Sequential(
                        layer_init(nn.Linear(H + act_dim, H)),
                        ReLUSquared(),
                        layer_init(nn.Linear(H, H), std=0.1),
                    )
                if self.nextlat_spec.direct_horizons:
                    self.nextlat_direct_predictor = DirectHorizonPredictor(
                        H,
                        act_dim,
                        self.nextlat_spec.direct_horizons,
                        args.nextlat_horizon_embed_dim,
                        args.nextlat_direct_hidden,
                    )

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

    def _actor_dist_frozen_head(self, actor_feat):
        """Decode a latent in policy geometry without updating policy-head weights."""
        if self.actor_dist == "gaussian":
            mean = F.linear(
                actor_feat,
                self.actor_head.weight.detach(),
                None if self.actor_head.bias is None else self.actor_head.bias.detach(),
            )
            raw_lv = F.linear(
                actor_feat,
                self.actor_logvar_head.weight.detach(),
                None
                if self.actor_logvar_head.bias is None
                else self.actor_logvar_head.bias.detach(),
            )
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            return (
                Normal(mean, (0.5 * lv).exp()),
                torch.tanh,
                lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z)),
            )
        alpha_raw = F.linear(
            actor_feat,
            self.actor_alpha_head.weight.detach(),
            None
            if self.actor_alpha_head.bias is None
            else self.actor_alpha_head.bias.detach(),
        )
        beta_raw = F.linear(
            actor_feat,
            self.actor_beta_head.weight.detach(),
            None
            if self.actor_beta_head.bias is None
            else self.actor_beta_head.bias.detach(),
        )
        return (
            Beta(1.0 + F.softplus(alpha_raw), 1.0 + F.softplus(beta_raw)),
            lambda z: self.action_low + (self.action_high - self.action_low) * z,
            lambda z: 0.0,
        )

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns value LOGITS (B, num_bins); caller converts via support.
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

    def get_actor_feat(self, x):
        return self._trunks(x)[0]

    def get_action_and_value(
        self, x, z=None, candidate_zs=None, return_dist=False, return_actor_feat=False
    ):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        # TPO extensions (both default-off => base behavior/graph/RNG untouched):
        #   candidate_zs (B, K, A): also return per-candidate logprobs (B, K) from
        #     the SAME dist (one trunk forward, consumes no RNG — log_prob only,
        #     evaluated AFTER the gaussian entropy rsample so the RNG order of the
        #     base computation is preserved).
        #   return_dist: also return (dist, to_action, log_det_fn) so the rollout
        #     can sample probe candidates from the already-constructed dist.
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_logits = self.critic_head(critic_feat)
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
        out = (action, z, log_prob, entropy, value_logits)
        if candidate_zs is not None:
            # Evaluate as (K, B, A) so the dist's (B, A) batch shape broadcasts
            # over the K axis, then transpose back to (B, K).
            cz = candidate_zs.transpose(0, 1)
            candidate_log_probs = (dist.log_prob(cz) - log_det_fn(cz)).sum(-1).transpose(0, 1)
            out = out + (candidate_log_probs,)
        if return_dist:
            out = out + (dist, to_action, log_det_fn)
        if return_actor_feat:
            out = out + (actor_feat,)
        return out

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

    def nextlat_trunk_blocks(self):
        """Logical actor-trunk blocks used for local predictive trust regions."""
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        blocks = [[*trunk.entry.parameters()]]
        blocks.extend([list(block.parameters()) for block in trunk.blocks])
        blocks.append([*trunk.out_proj.parameters()])
        return blocks

    def nextlat_trunk_parameters(self):
        if not self.nextlat_spec.enabled:
            return []
        return [parameter for block in self.nextlat_trunk_blocks() for parameter in block]

    def nextlat_predictor_parameters(self):
        parameters = []
        if hasattr(self, "nextlat_predictor"):
            parameters.extend(self.nextlat_predictor.parameters())
        if hasattr(self, "nextlat_direct_predictor"):
            parameters.extend(self.nextlat_direct_predictor.parameters())
        return list(parameters)

    def nextlat_parameters(self):
        return self.nextlat_trunk_parameters() + self.nextlat_predictor_parameters()

    def task_parameters(self):
        predictor_ids = {id(parameter) for parameter in self.nextlat_predictor_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in predictor_ids
        ]

    def nextlat_parameter_budget(self):
        actual = sum(parameter.numel() for parameter in self.nextlat_predictor_parameters())
        reference = recursive_predictor_parameter_count(
            self.critic_head.in_features, self.nextlat_action_dim
        )
        return actual, reference

    def nextlat_forward_flops(self):
        hidden = self.critic_head.in_features
        action_dim = self.nextlat_action_dim
        flops = 0
        if self.nextlat_spec.recursive_horizons:
            per_step_macs = hidden * (hidden + action_dim) + hidden * hidden
            flops += 2 * max(self.nextlat_spec.recursive_horizons) * per_step_macs
        if self.nextlat_spec.direct_horizons:
            predictor = self.nextlat_direct_predictor
            first = predictor.network[0]
            last = predictor.network[2]
            per_endpoint_macs = first.in_features * first.out_features + last.in_features * last.out_features
            flops += 2 * len(self.nextlat_spec.direct_horizons) * per_endpoint_macs
        return flops


def policy_model_forward(agent, observations):
    """Static neural policy/value path with no sampling or RNG side effects."""
    actor_feat, critic_feat = agent._trunks(observations)
    value_logits = agent.critic_head(critic_feat)
    if agent.actor_dist == "gaussian":
        first = agent.actor_head(actor_feat)
        raw_lv = agent.actor_logvar_head(actor_feat)
        second = rescale(
            (raw_lv / (agent.logvar_max - agent.logvar_min)).tanh(),
            (-1.0, 1.0),
            (agent.logvar_min, agent.logvar_max),
        )
    else:
        first = 1.0 + F.softplus(agent.actor_alpha_head(actor_feat))
        second = 1.0 + F.softplus(agent.actor_beta_head(actor_feat))
    return actor_feat, value_logits, first, second


def action_value_from_policy_outputs(
    agent,
    model_outputs,
    z=None,
    candidate_zs=None,
):
    """Apply v13's exact eager sampling/log-prob order to compiled model outputs."""
    actor_feat, value_logits, first, second = model_outputs
    if agent.actor_dist == "gaussian":
        dist = Normal(first, (0.5 * second).exp())
        to_action = torch.tanh
        log_det_fn = lambda sample: 2.0 * (
            log(2.0) - sample - F.softplus(-2.0 * sample)
        )
    else:
        dist = Beta(first, second)
        to_action = lambda sample: agent.action_low + (
            agent.action_high - agent.action_low
        ) * sample
        log_det_fn = lambda sample: 0.0
    if z is None:
        z = dist.sample()
        if agent.actor_dist == "beta":
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    action = to_action(z)
    log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
    if agent.actor_dist == "gaussian":
        zr = dist.rsample()
        entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
    else:
        entropy = dist.entropy().sum(1)
    out = (action, z, log_prob, entropy, value_logits)
    if candidate_zs is not None:
        candidate_transposed = candidate_zs.transpose(0, 1)
        candidate_log_probs = (
            dist.log_prob(candidate_transposed) - log_det_fn(candidate_transposed)
        ).sum(-1).transpose(0, 1)
        out = out + (candidate_log_probs,)
    return out + (actor_feat,), dist, to_action, log_det_fn


def value_forward(agent, observations):
    """Static value-logit wrapper used by transition and probe bootstraps."""
    return agent.get_value(observations)


def target_actor_feat_forward(agent, observations):
    """Static stopped-online target encoder wrapper."""
    return agent.get_actor_feat(observations)


def tpo_restricted_target(anchor_logp, score_signal, eta):
    """TPO-MD v5's anchored K-action mirror-descent target."""
    return torch.softmax(anchor_logp + score_signal / eta, dim=-1)


def tpo_reverse_kl(anchor_logp, score_signal, eta):
    """Batch mean KL(p_old || q_eta), the v5 one-sided-cap statistic."""
    p_old = anchor_logp.exp()
    log_q = F.log_softmax(anchor_logp + score_signal / eta, dim=-1)
    return (p_old * (anchor_logp - log_q)).sum(-1).mean()


def build_nextlat_mask(transition_boundaries, depth):
    """Validity of (state, outgoing actions, future-state) latent sequences."""
    num_steps, num_envs = transition_boundaries.shape
    mask = transition_boundaries.new_zeros((num_steps, num_envs, depth))
    for horizon in range(1, depth + 1):
        valid_len = num_steps - horizon
        if valid_len <= 0:
            break
        valid = torch.ones(
            (valid_len, num_envs), dtype=torch.bool, device=transition_boundaries.device
        )
        for offset in range(horizon):
            valid &= transition_boundaries[offset : offset + valid_len] == 0
        mask[:valid_len, :, horizon - 1] = valid.to(mask.dtype)
    return mask


def build_multiscale_nextlat_mask(transition_boundaries, horizons):
    """Boundary-safe validity for arbitrary future endpoints."""
    horizons = tuple(horizons)
    if not horizons:
        shape = (*transition_boundaries.shape, 0)
        return transition_boundaries.new_zeros(shape)
    dense = build_nextlat_mask(transition_boundaries, max(horizons))
    indices = torch.tensor(
        [horizon - 1 for horizon in horizons],
        dtype=torch.int64,
        device=transition_boundaries.device,
    )
    return dense.index_select(-1, indices)


def make_nextlat_indices(source_indices, num_envs, batch_size, depth):
    """T-major outgoing-action and future-state indices for recursive prediction."""
    action_offsets = np.arange(depth, dtype=np.int64)[:, None] * num_envs
    target_offsets = np.arange(1, depth + 1, dtype=np.int64)[:, None] * num_envs
    action_indices = np.clip(source_indices[None, :] + action_offsets, 0, batch_size - 1)
    target_indices = np.clip(source_indices[None, :] + target_offsets, 0, batch_size - 1)
    return action_indices, target_indices


def make_multiscale_nextlat_indices(source_indices, num_envs, batch_size, horizons):
    """All prefix-action indices plus the selected future endpoints."""
    max_horizon = max(horizons)
    action_offsets = np.arange(max_horizon, dtype=np.int64)[:, None] * num_envs
    target_offsets = np.asarray(horizons, dtype=np.int64)[:, None] * num_envs
    action_indices = np.clip(source_indices[None, :] + action_offsets, 0, batch_size - 1)
    target_indices = np.clip(source_indices[None, :] + target_offsets, 0, batch_size - 1)
    return action_indices, target_indices


@torch.no_grad()
def capture_gradients(parameters):
    """Clone the current (already clipped) gradients as a sparse parameter map."""
    return {
        parameter: parameter.grad.detach().clone()
        for parameter in parameters
        if parameter.grad is not None
    }


@torch.no_grad()
def gradient_group_norm(parameters, gradient_group):
    terms = [
        gradient_group[parameter].square().sum()
        for parameter in parameters
        if parameter in gradient_group
    ]
    if terms:
        return torch.stack(terms).sum().sqrt()
    return parameters[0].new_zeros(())


@torch.no_grad()
def gradient_group_cosine(parameters, left_group, *right_groups):
    """Cosine on one fixed parameter partition; right groups are summed as task."""
    reference = parameters[0]
    dot_terms, left_terms, right_terms = [], [], []
    for parameter in parameters:
        left = left_group.get(parameter)
        rights = [group[parameter] for group in right_groups if parameter in group]
        right = sum(rights[1:], rights[0]) if rights else None
        if left is not None:
            left_terms.append(left.square().sum())
        if right is not None:
            right_terms.append(right.square().sum())
        if left is not None and right is not None:
            dot_terms.append((left * right).sum())
    if not dot_terms or not left_terms or not right_terms:
        return reference.new_zeros(())
    dot = torch.stack(dot_terms).sum()
    denominator = (
        torch.stack(left_terms).sum().sqrt()
        * torch.stack(right_terms).sum().sqrt()
    ).clamp_min(1e-12)
    return dot / denominator


def nextlat_source_feature(actor_feature, *, trunk_active):
    """Root NextLat at the trunk only while policy updates are permitted."""
    return actor_feature if trunk_active else actor_feature.detach()


@torch.no_grad()
def capture_nextlat_gradient_groups(
    trunk_parameters,
    predictor_parameters,
    *,
    trunk_active,
    trunk_max_norm,
    predictor_max_norm,
):
    """Clip/capture only gradient groups that will actually be delivered.

    After the TPO breaker, the loss is rooted at a detached actor feature. The trunk
    therefore has no auxiliary gradient, and deliberately does not pay a norm reduction
    or a full parameter-clone scan. Predictor learning remains independently clipped.
    """
    if trunk_active:
        trunk_norm = clip_grad_norm_async_fail_loud_(
            trunk_parameters,
            trunk_max_norm,
        )
        trunk_gradients = capture_gradients(trunk_parameters)
    else:
        trunk_norm = predictor_parameters[0].new_zeros(())
        trunk_gradients = {}
    predictor_norm = clip_grad_norm_async_fail_loud_(
        predictor_parameters,
        predictor_max_norm,
    )
    predictor_gradients = capture_gradients(predictor_parameters)
    return trunk_norm, predictor_norm, trunk_gradients, predictor_gradients


@torch.no_grad()
def merge_gradient_groups(parameters, *gradient_groups, validate_finite=True):
    """Sum independently clipped gradient groups for one shared optimizer step.

    Missing entries remain missing (not zero gradients), preserving Adam's exact lazy
    state/step behavior. Every provided tensor is checked on-device before installation.
    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError("at least one optimizer parameter is required")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("optimizer parameters must be unique")
    parameter_ids = {id(parameter) for parameter in parameters}
    merged = {}
    for group in gradient_groups:
        for parameter, gradient in group.items():
            if id(parameter) not in parameter_ids:
                raise ValueError("gradient group contains a foreign parameter")
            if gradient.shape != parameter.shape or gradient.device != parameter.device:
                raise ValueError("gradient must match its parameter shape and device")
            if validate_finite:
                torch._assert_async(
                    torch.isfinite(gradient).all(),
                    "A clipped gradient group is non-finite; refusing the shared Adam step",
                )
            if parameter in merged:
                merged[parameter].add_(gradient)
            else:
                merged[parameter] = gradient.detach().clone()
    for parameter in parameters:
        parameter.grad = merged.get(parameter)
    return merged


@torch.no_grad()
def apply_union_optimizer_step(
    parameters,
    optimizer,
    *,
    actor_gradients,
    critic_gradients,
    auxiliary_gradients,
    validate_finite=True,
):
    """Install three clipped groups and advance their coherent shared Adam moments."""
    parameters = list(parameters)
    merged = merge_gradient_groups(
        parameters,
        actor_gradients,
        critic_gradients,
        auxiliary_gradients,
        validate_finite=validate_finite,
    )
    if not merged:
        optimizer.zero_grad(set_to_none=True)
        return merged
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return merged


@torch.no_grad()
def apply_private_predictor_step(parameters, optimizer, gradients):
    """Advance the predictor's private Adam from its independently clipped gradient."""
    parameters = list(parameters)
    merged = merge_gradient_groups(parameters, gradients, validate_finite=False)
    if not merged:
        optimizer.zero_grad(set_to_none=True)
        return
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


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
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope == "batch"), \
        "norm_adv_scope=batch requires adv_transform_scope=batch"
    assert args.tpo_coef > 0.0, "TPO-MD is the entire policy update; tpo_coef must be > 0"
    assert args.tpo_k >= 2, "TPO needs at least two candidates per group"
    assert args.tpo_eps > 0.0, "tpo_eps must be positive"
    assert args.tpo_eta_base > 0.0, "tpo_eta_base must be positive"
    assert args.tpo_kl_breaker > 0.0, "tpo_kl_breaker must be positive"
    nextlat_spec = resolve_nextlat_spec(
        args.pc_mode,
        args.nextlat_horizons,
        args.nextlat_recursive_depth,
        args.nextlat_hybrid_recursive_max,
    )
    assert args.nextlat_loss_scale in ("raw", "target_std")
    assert args.nextlat_horizon_embed_dim >= 1
    assert args.nextlat_direct_hidden >= 0
    assert nextlat_spec.max_horizon < args.num_steps or not nextlat_spec.enabled, \
        "every nextlat horizon must be smaller than num_steps"
    assert args.nextlat_target_batch_size >= 1, "nextlat_target_batch_size must be positive"
    assert args.nextlat_trunk_grad_clip >= 0.0
    assert args.nextlat_predictor_grad_clip >= 0.0
    assert args.tail_risk_window >= 1
    tail_thresholds = parse_tail_thresholds(args.tail_risk_thresholds)
    assert args.separate_grad_clip, "NextLat union requires separate gradient groups"
    # Probe rewards are RAW physics rewards; the critic must live in the same units.
    assert not args.normalize_reward, "TPO probe scores require raw rewards"
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
    # Ampere+ TF32 tensor-core matmuls are substantially faster. ``high`` retains
    # full-size float32 outputs/accumulators; only last-bit eager-v13 numerics may differ.
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

    # TPO-MD probe machinery (always on: TPO-MD IS the policy update; every
    # (env, step) cell is probed — no state-frac mask).
    # Cache once: the raw physics env (walk the .env chain to the unwrapped
    # MujocoEnv) and the NormalizeObservation wrapper reference per env.
    probe_base_envs = [e.unwrapped for e in envs.envs]
    probe_obs_wrappers = [find_wrapper(e, gym.wrappers.NormalizeObservation) for e in envs.envs]
    assert all(w is not None for w in probe_obs_wrappers), "NormalizeObservation wrapper not found"
    probe_action_low = envs.single_action_space.low
    probe_action_high = envs.single_action_space.high
    # Persistent probe RNG stream: saved CPU+CUDA states restored inside
    # torch.random.fork_rng at every sampling site, so candidate sampling
    # never advances the MAIN RNG stream (the PPO trajectory of a tpo run
    # matches an unprobed run exactly).
    probe_cpu_rng_state = None
    probe_cuda_rng_state = None
    td_rms_ema = None  # EMA (decay 0.99) of the one-step TD-residual RMS

    agent = Agent(envs, args).to(device)
    # Task Adam owns the exact TPO model. Only the auxiliary trunk gradient joins its
    # moments; predictor parameters retain a private optimizer as in v13.
    task_params = agent.task_parameters()
    task_optimizer = optim.Adam(task_params, lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    nextlat_params = agent.nextlat_parameters()
    nextlat_trunk_params = agent.nextlat_trunk_parameters()
    nextlat_predictor_params = agent.nextlat_predictor_parameters()
    assert {id(parameter) for parameter in nextlat_params} == {
        id(parameter)
        for parameter in (nextlat_trunk_params + nextlat_predictor_params)
    }
    predictor_optimizer = None
    if nextlat_spec.enabled:
        predictor_optimizer = optim.Adam(
            nextlat_predictor_params,
            lr=args.learning_rate,
            eps=1e-5,
        )
    predictor_parameter_count, recursive_reference_parameter_count = agent.nextlat_parameter_budget()
    predictor_parameter_ratio = predictor_parameter_count / max(
        recursive_reference_parameter_count, 1
    )
    predictor_forward_flops = agent.nextlat_forward_flops()
    recursive_reference_forward_flops = 0
    if nextlat_spec.enabled:
        hidden = args.hidden
        action_dim = int(np.prod(envs.single_action_space.shape))
        recursive_reference_forward_flops = (
            2
            * nextlat_spec.max_horizon
            * (hidden * (hidden + action_dim) + hidden * hidden)
        )
    predictor_flop_ratio = predictor_forward_flops / max(
        recursive_reference_forward_flops, 1
    )
    writer.add_text(
        "nextlat/design",
        f"mode={nextlat_spec.mode}; horizons={nextlat_spec.horizons}; "
        f"recursive={nextlat_spec.recursive_horizons}; direct={nextlat_spec.direct_horizons}; "
        f"predictor_params={predictor_parameter_count}; "
        f"recursive_reference_params={recursive_reference_parameter_count}; "
        f"forward_flops={predictor_forward_flops}; "
        f"recursive_reference_flops={recursive_reference_forward_flops}",
    )

    def policy_rollout_fn(obs_):
        return policy_model_forward(agent, obs_)

    def policy_update_fn(obs_):
        return policy_model_forward(agent, obs_)

    def transition_value_fn(obs_):
        return value_forward(agent, obs_)

    def probe_value_fn(obs_):
        return value_forward(agent, obs_)

    def target_actor_feat_fn(obs_):
        return target_actor_feat_forward(agent, obs_)

    def nextlat_prediction_fn(source_, future_actions_):
        return multiscale_prediction_forward(agent, source_, future_actions_)

    def project_value_targets_fn(targets_):
        return hl_support.project(targets_)

    if args.compile:
        policy_rollout_fn = torch.compile(
            policy_rollout_fn, mode=args.compile_mode, dynamic=False
        )
        policy_update_fn = torch.compile(
            policy_update_fn, mode=args.compile_mode, dynamic=False
        )
        transition_value_fn = torch.compile(
            transition_value_fn, mode=args.compile_mode, dynamic=False
        )
        probe_value_fn = torch.compile(
            probe_value_fn, mode=args.compile_mode, dynamic=False
        )
        if nextlat_spec.enabled:
            target_actor_feat_fn = torch.compile(
                target_actor_feat_fn, mode=args.compile_mode, dynamic=False
            )
            nextlat_prediction_fn = torch.compile(
                nextlat_prediction_fn,
                mode=args.compile_mode,
                dynamic=False,
                fullgraph=True,
            )
        project_value_targets_fn = torch.compile(
            project_value_targets_fn,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        print(f"compiled static agent and target paths ({args.compile_mode})")

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
    hl_support = HLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )
    support = hl_support.support                       # (num_bins,) bin centers
    bin_width = hl_support.bin_width
    scalar_support = symexp(support) if args.value_symlog else support

    def value_logits_to_scalar(logits):
        return hl_support.to_expected_scalar(logits)

    scalar_bin_width = (
        (scalar_support[1:] - scalar_support[:-1]).abs().min()
        if args.value_symlog
        else bin_width
    )

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.ones((args.num_steps, args.num_envs)).to(device)
    next_transition_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_transition_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    # Preallocated probe storage: CPU numpy for the physics outputs, GPU
    # tensors for candidate z's/logprobs (written once per step, no per-env syncs).
    tpo_next_obs_np = np.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + envs.single_observation_space.shape, dtype=np.float32
    )
    tpo_rewards_np = np.zeros((args.num_steps, args.num_envs, args.tpo_k), dtype=np.float32)
    tpo_terms_np = np.zeros((args.num_steps, args.num_envs, args.tpo_k), dtype=np.float32)
    tpo_zs = torch.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + envs.single_action_space.shape
    ).to(device)
    tpo_logprobs = torch.zeros((args.num_steps, args.num_envs, args.tpo_k)).to(device)

    global_step = 0
    start_time = time.time()
    completed_episode_returns = deque(maxlen=args.tail_risk_window)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            task_optimizer.param_groups[0]["lr"] = lrnow
            if predictor_optimizer is not None:
                predictor_optimizer.param_groups[0]["lr"] = lrnow
        probe_seconds = 0.0

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                rollout_outputs = policy_rollout_fn(next_obs)
                (
                    action,
                    z,
                    logprob,
                    ent,
                    value_logits,
                    roll_actor_feat,
                ), roll_dist, roll_to_action, roll_log_det_fn = action_value_from_policy_outputs(
                    agent,
                    rollout_outputs,
                )
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits)
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            # --- TPO probe: K candidates per env, one raw physics step each (every state) ---
            probe_start = time.time()
            with torch.no_grad():
                # Candidate sampling rides the PERSISTENT probe RNG stream inside
                # fork_rng: restore probe state, sample, save state back. The main
                # stream is untouched.
                with torch.random.fork_rng(devices=[device]):
                    if probe_cpu_rng_state is None:
                        torch.manual_seed(args.seed + 1_000_003)
                    else:
                        torch.set_rng_state(probe_cpu_rng_state)
                        torch.cuda.set_rng_state(probe_cuda_rng_state, device)
                    cand_z = roll_dist.sample(torch.Size([args.tpo_k]))   # (K, N, A)
                    probe_cpu_rng_state = torch.get_rng_state()
                    probe_cuda_rng_state = torch.cuda.get_rng_state(device)
                cand_z = cand_z.permute(1, 0, 2).contiguous()             # (N, K, A)
                if args.actor_dist == "beta":
                    cand_z = cand_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                cand_z[:, 0] = z                                          # executed action = candidate 0
                cz = cand_z.transpose(0, 1)                               # (K, N, A)
                cand_logprob = (roll_dist.log_prob(cz) - roll_log_det_fn(cz)).sum(-1).transpose(0, 1)
                tpo_zs[step] = cand_z
                tpo_logprobs[step] = cand_logprob                         # (N, K)
                # One transfer for the whole candidate block (no per-env GPU syncs).
                cand_actions_np = roll_to_action(cand_z).cpu().numpy()
            cand_actions_np = np.clip(cand_actions_np, probe_action_low, probe_action_high)
            for env_i in range(args.num_envs):
                base_env = probe_base_envs[env_i]
                obs_rms = probe_obs_wrappers[env_i].obs_rms
                saved_qpos = base_env.data.qpos.copy()
                saved_qvel = base_env.data.qvel.copy()
                saved_warm = base_env.data.qacc_warmstart.copy()
                saved_time = base_env.data.time
                for cand_i in range(args.tpo_k):
                    # Direct-assign restore (NO mj_forward, NEVER MujocoEnv.set_state):
                    # mj_step recomputes forward dynamics itself; restoring
                    # qacc_warmstart keeps the solver warmstart bit-identical so the
                    # REAL env.step below matches an unprobed run exactly.
                    base_env.data.qpos[:] = saved_qpos
                    base_env.data.qvel[:] = saved_qvel
                    base_env.data.qacc_warmstart[:] = saved_warm
                    base_env.data.time = saved_time
                    probe_obs, probe_rew, probe_term, _, _ = base_env.step(cand_actions_np[env_i, cand_i])
                    # FROZEN wrapper stats (stepping the raw env never updates
                    # obs_rms): float64 math, cast float32, then clip [-10, 10].
                    norm_obs = ((probe_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)).astype(np.float32)
                    tpo_next_obs_np[step, env_i, cand_i] = np.clip(norm_obs, -10.0, 10.0)
                    tpo_rewards_np[step, env_i, cand_i] = probe_rew       # RAW reward (base is raw-return)
                    tpo_terms_np[step, env_i, cand_i] = float(probe_term)
                base_env.data.qpos[:] = saved_qpos
                base_env.data.qvel[:] = saved_qvel
                base_env.data.qacc_warmstart[:] = saved_warm
                base_env.data.time = saved_time
            probe_seconds += time.time() - probe_start

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                transition_next_obs = np.array(next_obs_np, copy=True)
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0
            else:
                transition_next_obs = next_obs_np
            transition_next_obs_t = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                next_transition_logits = transition_value_fn(transition_next_obs_t)
                next_transition_values[step] = value_logits_to_scalar(next_transition_logits)
            next_transition_obses[step] = transition_next_obs_t
            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(terminations, device=device, dtype=torch.float32)
            transition_boundaries[step] = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        completed_episode_returns.append(float(info["episode"]["r"]))

        with torch.no_grad():
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample a' ~ π(·|s_{t+1}) for each transition bootstrap entropy.
                # Use transition_next_obs, not rollout next_obs, so time-limit
                # truncations pair V(final_obs) with H(final_obs) rather than
                # accidentally reading entropy from the reset observation.
                _, _, next_transition_logprob, _, _ = agent.get_action_and_value(
                    next_transition_obses.reshape((-1,) + envs.single_observation_space.shape)
                )
                next_transition_logprob = next_transition_logprob.reshape(args.num_steps, args.num_envs)
                alpha_r = log_alpha.exp().detach()
                next_value_bonus = alpha_r * (-next_transition_logprob)
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
            # Critic target: Dreamer4-style scalar-return HL-Gauss. GAE computes
            # the scalar λ-return; the value encoder projects that scalar target
            # into a Gaussian-smoothed categorical distribution over fixed bins.
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            projected_targets = project_value_targets_fn(returns)
            # This full CUDA-resident table is indexed for all ten PPO epochs. A
            # compiled graph owns a reusable output buffer, so retain an independent
            # clone before any later compiled forward can replay its storage.
            target_probs = retain_graph_output(
                projected_targets,
                compiled=args.compile,
            )
            # Per-state return std sigma(s_t) in raw return units, matching the
            # GAE scale consumed by tanh_std.
            sigma = (value_probs * (scalar_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * scalar_bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            returns_coord = symlog(returns) if args.value_symlog else returns
            cdf_frac = ((returns_coord.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean()  # calib probe (uniform≈0.1)

            # --- TPO-MD target construction (frozen pre-update critic; q fixed across epochs) ---
            # Running one-step TD-residual RMS over EXECUTED transitions -> GLOBAL score sigma.
            td_resid = (
                rewards
                + args.gamma * next_transition_values * (1.0 - transition_terminations) * transition_valids
                - values
            )
            td_rms = td_resid.pow(2).mean().sqrt().item()
            td_rms_ema = td_rms if td_rms_ema is None else 0.99 * td_rms_ema + 0.01 * td_rms
            tpo_sigma_global = max(args.tpo_sigma_scale_coef * td_rms_ema, 1e-6)

            b_tpo_zs = tpo_zs.reshape((-1, args.tpo_k) + envs.single_action_space.shape)
            obs_dim = int(np.array(envs.single_observation_space.shape).prod())
            flat_probe_obs = torch.as_tensor(
                tpo_next_obs_np.reshape(-1, obs_dim), device=device
            )
            # Four static 65,536-row critic forwards at the defaults. Clone each graph
            # output immediately: later replays reuse its otherwise-ephemeral storage.
            probe_value_chunks = []
            for chunk in flat_probe_obs.split(65536):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                chunk_logits = probe_value_fn(chunk)
                chunk_logits = retain_graph_output(
                    chunk_logits,
                    compiled=args.compile,
                )
                probe_value_chunks.append(value_logits_to_scalar(chunk_logits))
            v_next = torch.cat(probe_value_chunks).reshape(
                args.batch_size, args.tpo_k
            )
            r_probe = torch.as_tensor(tpo_rewards_np.reshape(-1, args.tpo_k), device=device)
            term_probe = torch.as_tensor(tpo_terms_np.reshape(-1, args.tpo_k), device=device)
            # Oracle score: raw probe reward + bootstrapped frozen value.
            scores = r_probe + args.gamma * (1.0 - term_probe) * v_next      # (B, K)
            # Center per group, scale by the ONE GLOBAL sigma: cross-state advantage
            # MAGNITUDE survives (per-group z-scoring would erase it); no floor gating —
            # weak groups just contribute u ~= 0 naturally.
            u_scores = (
                (scores - scores.mean(dim=-1, keepdim=True)) / tpo_sigma_global
            ).clamp(-5.0, 5.0)
            group_std = scores.std(dim=-1, unbiased=False)                   # (B,) diagnostics only
            anchor_logp = F.log_softmax(tpo_logprobs.reshape(-1, args.tpo_k), dim=-1)

            def tpo_mean_kl(eta):
                # batch-mean KL(p_old || q(eta)); monotone DECREASING in eta
                # (eta -> inf => q -> p_old => KL -> 0).
                return tpo_reverse_kl(anchor_logp, u_scores, eta).item()

            # kl_base = the NATURAL (uncapped) step the SNR signal produces at the
            # fixed base temperature. In SNR units this is large under real signal
            # and -> 0 when candidates are within the critic's noise floor. Used by
            # the dynamic-cap path; also logged for diagnostics.
            tpo_kl_base = tpo_mean_kl(args.tpo_eta_base)
            tpo_cap_engaged = 0.0
            if u_scores.abs().max().item() < 1e-8:
                # Degenerate scores: target collapses to the anchor regardless of eta.
                # In dyn-trust eta_base is the natural choice (q ~= p_old anyway); in
                # v1 mode the original code returned 1.0 here.
                tpo_eta_solved = args.tpo_eta_base if args.tpo_dyn_trust else 1.0
            elif args.tpo_dyn_trust:
                # One-sided KL cap on the fixed base temperature. KL(eta) is monotone
                # DECREASING in eta, so we only ever RAISE eta above eta_base to pull
                # an over-large natural step DOWN to the cap; we never lower it. Thus
                # eta_solved >= eta_base ALWAYS, the step is bounded above by eps_cap,
                # and is free to shrink to ~0 when kl_base falls below the cap. No
                # lower floor — intentional (this is the late-training fix).
                if tpo_kl_base <= args.tpo_eps:
                    tpo_eta_solved = args.tpo_eta_base       # weak signal: natural step already within cap
                else:
                    tpo_cap_engaged = 1.0                    # strong signal: cap binds
                    log_lo, log_hi = float(np.log(args.tpo_eta_base)), float(np.log(1e4))
                    if tpo_mean_kl(float(np.exp(log_hi))) > args.tpo_eps:
                        tpo_eta_solved = float(np.exp(log_hi))  # even max temperature can't reach cap -> clamp
                    else:
                        # KL(eta_base) > eps and KL(1e4) <= eps -> root bracketed.
                        for _ in range(40):
                            log_mid = 0.5 * (log_lo + log_hi)
                            if tpo_mean_kl(float(np.exp(log_mid))) > args.tpo_eps:
                                log_lo = log_mid             # KL too big -> need larger eta
                            else:
                                log_hi = log_mid
                        tpo_eta_solved = float(np.exp(0.5 * (log_lo + log_hi)))
            elif args.tpo_adaptive_eta:
                # MPO-style dual: bisect log-eta so mean KL(p_old||q) = tpo_eps.
                log_lo, log_hi = float(np.log(1e-2)), float(np.log(1e4))
                if tpo_mean_kl(float(np.exp(log_lo))) < args.tpo_eps:
                    tpo_eta_solved = float(np.exp(log_lo))   # weak scores: even max-strength < eps
                elif tpo_mean_kl(float(np.exp(log_hi))) > args.tpo_eps:
                    tpo_eta_solved = float(np.exp(log_hi))   # huge scores: clamp at max temperature
                else:
                    for _ in range(40):
                        log_mid = 0.5 * (log_lo + log_hi)
                        if tpo_mean_kl(float(np.exp(log_mid))) > args.tpo_eps:
                            log_lo = log_mid                 # KL too big -> need larger eta
                        else:
                            log_hi = log_mid
                    tpo_eta_solved = float(np.exp(0.5 * (log_lo + log_hi)))
            else:
                tpo_eta_solved = args.tpo_eta
            b_tpo_q = tpo_restricted_target(anchor_logp, u_scores, tpo_eta_solved).detach()
            tpo_kl_achieved = tpo_mean_kl(tpo_eta_solved)
            log_q = b_tpo_q.clamp_min(1e-12).log()
            tpo_group_kl = (b_tpo_q * (log_q - anchor_logp)).sum(-1).mean().item()
            tpo_q_entropy = (-(b_tpo_q * log_q).sum(-1)).mean().item()
            tpo_score_std_mean = group_std.mean().item()
            tpo_score_std_p90 = group_std.quantile(0.9).item()
            if nextlat_spec.enabled:
                nextlat_mask = build_multiscale_nextlat_mask(
                    transition_boundaries, nextlat_spec.horizons
                )

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.num_bins)
        b_u = u.reshape(-1)
        if nextlat_spec.enabled:
            b_nextlat_mask = nextlat_mask.reshape(-1, len(nextlat_spec.horizons))
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean()

        b_inds = np.arange(args.batch_size)
        if nextlat_spec.enabled:
            nextlat_action_offsets = (
                torch.arange(
                    nextlat_spec.max_horizon,
                    device=device,
                    dtype=torch.int64,
                )[:, None]
                * args.num_envs
            )
            nextlat_target_offsets = (
                torch.tensor(
                    nextlat_spec.horizons,
                    device=device,
                    dtype=torch.int64,
                )[:, None]
                * args.num_envs
            )
        # v13 converted every float32 minibatch mean to Python before NumPy's
        # float64 average. Accumulate the same sequence asynchronously on CUDA.
        clipfrac_sum = torch.zeros((), dtype=torch.float64, device=device)
        clipfrac_count = 0
        epochs_completed = 0
        actor_epochs_completed = 0
        actor_active = True  # flipped off by the tpo_kl_breaker; critic runs all epochs regardless
        rollout_max_minibatch_approx_kl = torch.full(
            (), float("-inf"), dtype=torch.float64, device=device
        )
        rollout_max_minibatch_old_approx_kl = torch.full(
            (), float("-inf"), dtype=torch.float64, device=device
        )
        rollout_max_epoch_mean_approx_kl = torch.full(
            (), float("-inf"), dtype=torch.float64, device=device
        )
        if nextlat_spec.enabled:
            # One stopped ONLINE (not EMA) target table from the pre-update trunk.
            # Chunked compiled encoding makes all depths simple indexed reads and
            # prevents a moving teacher from chasing ten epochs of student updates.
            target_chunks = []
            with torch.no_grad():
                for target_obs in b_obs.split(args.nextlat_target_batch_size):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    # Every retained chunk needs independent storage because the next
                    # replay reuses the graph-backed output buffer.
                    target_chunks.append(
                        retain_graph_output(
                            target_actor_feat_fn(target_obs),
                            compiled=args.compile,
                        )
                    )
                frozen_actor_feats = torch.cat(target_chunks)
                latent_batch_std = frozen_actor_feats.std(dim=0).mean()
                latent_scale, latent_participation_rank = latent_scale_and_participation_rank(
                    frozen_actor_feats
                )
                all_source_indices = torch.arange(args.batch_size, device=device)
                all_target_indices = (
                    all_source_indices[None, :] + nextlat_target_offsets
                ).clamp_max(args.batch_size - 1)
                all_target_features = frozen_actor_feats[all_target_indices]
                rollout_target_spreads = compute_rollout_target_spreads(
                    all_target_features,
                    b_nextlat_mask.T,
                )
                rollout_valid_fractions = b_nextlat_mask.mean(dim=0)
                del all_target_features, all_target_indices
        last_nextlat_mb_inds = None
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # Keep NumPy's exact seeded permutation, paying one compact transfer per
            # epoch instead of implicit host-index conversion for every buffer read.
            epoch_inds = torch.as_tensor(b_inds, device=device)
            epoch_kl_sum = torch.zeros((), dtype=torch.float64, device=device)
            epoch_kl_count = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = epoch_inds[start:end]

                # Same trunk forward as the base; the candidate logprobs ride the
                # SAME dist (no second trunk pass, consumes no RNG).
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                update_outputs = policy_update_fn(b_obs[mb_inds])
                (
                    _,
                    _,
                    newlogprob,
                    entropy,
                    value_logits,
                    new_cand_logprobs,
                    mb_actor_feat,
                ), _, _, _ = action_value_from_policy_outputs(
                    agent,
                    update_outputs,
                    b_latent_zs[mb_inds],
                    b_tpo_zs[mb_inds],
                )

                with torch.no_grad():
                    # TELEMETRY ONLY: ratio / KL / clipfrac (and pg_loss below) never
                    # reach a backward — the actor update is the TPO CE alone.
                    logratio = newlogprob.detach() - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    update_scalar_max_(rollout_max_minibatch_approx_kl, approx_kl)
                    update_scalar_max_(
                        rollout_max_minibatch_old_approx_kl, old_approx_kl
                    )
                    clipfrac_sum.add_(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    )
                    clipfrac_count += 1
                    epoch_kl_sum.add_(approx_kl)
                    epoch_kl_count += 1

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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

                # TELEMETRY-ONLY clipped surrogate (kept for cross-run comparability;
                # ratio is already detached, so no PG gradient can exist anywhere).
                with torch.no_grad():
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # TPO CE on the K-restricted softmax over ALL states (every state is
                # probed). Targets q are frozen (solved once post-rollout, detached).
                mb_logp_new = F.log_softmax(new_cand_logprobs, dim=-1)
                tpo_ce = (-(b_tpo_q[mb_inds] * mb_logp_new).sum(-1)).mean()
                # PURE mirror descent: the CE is the entire actor objective.
                actor_loss = args.tpo_coef * tpo_ce

                # HL-Gauss value loss: cross-entropy to the fixed scalar-return
                # projection target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

                if nextlat_spec.enabled:
                    action_indices = (
                        mb_inds[None, :] + nextlat_action_offsets
                    ).clamp_max(args.batch_size - 1)
                    target_indices = (
                        mb_inds[None, :] + nextlat_target_offsets
                    ).clamp_max(args.batch_size - 1)
                    future_actions = b_actions[action_indices]
                    future_target_feats = frozen_actor_feats[target_indices]
                    # Once the breaker fires, NextLat becomes a predictor-only
                    # objective. Detaching here prevents construction of an auxiliary
                    # backward path through the trunk while preserving all predictor
                    # recurrence and frozen-target/decoder semantics.
                    h_hat = nextlat_source_feature(
                        mb_actor_feat,
                        trunk_active=actor_active,
                    )
                    predicted_features = nextlat_prediction_fn(h_hat, future_actions)
                    nextlat_losses = compute_nextlat_loss(
                        agent,
                        predicted_features,
                        future_target_feats,
                        b_nextlat_mask[mb_inds].T,
                        rollout_target_spreads,
                        args.nextlat_loss_scale,
                    )
                    nextlat_pred_loss = nextlat_losses["prediction"]
                    nextlat_kl_loss = nextlat_losses["policy_kl"]
                    nextlat_loss = (
                        nextlat_pred_loss + args.nextlat_kl_coef * nextlat_kl_loss
                    )
                    last_nextlat_mb_inds = mb_inds.detach()

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

                # Actor, critic, and auxiliary trunk are independently clipped; their
                # shared-trunk contributions meet only in task Adam. The predictor keeps
                # private moments and never enters the task optimizer.
                agent.zero_grad(set_to_none=True)
                (args.vf_coef * v_loss).backward(
                    retain_graph=actor_active
                )
                critic_gn = clip_grad_norm_async_fail_loud_(
                    critic_params, args.critic_grad_clip
                )
                value_grads = capture_gradients(critic_params)

                nextlat_trunk_grads = {}
                nextlat_predictor_grads = {}
                if nextlat_spec.enabled:
                    agent.zero_grad(set_to_none=True)
                    (args.nextlat_coef * nextlat_loss).backward(
                        retain_graph=actor_active
                    )
                    (
                        nextlat_trunk_gn,
                        nextlat_predictor_gn,
                        nextlat_trunk_grads,
                        nextlat_predictor_grads,
                    ) = capture_nextlat_gradient_groups(
                        nextlat_trunk_params,
                        nextlat_predictor_params,
                        trunk_active=actor_active,
                        trunk_max_norm=args.nextlat_trunk_grad_clip,
                        predictor_max_norm=args.nextlat_predictor_grad_clip,
                    )

                agent.zero_grad(set_to_none=True)
                actor_grads = {}
                if actor_active:
                    # Pure TPO CE remains the entire actor backward.
                    (actor_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = clip_grad_norm_async_fail_loud_(
                        actor_params, args.actor_grad_clip
                    )
                    actor_grads = capture_gradients(actor_params)
                else:
                    actor_gn = torch.zeros((), device=device)

                apply_union_optimizer_step(
                    task_params,
                    task_optimizer,
                    actor_gradients=actor_grads,
                    critic_gradients=value_grads,
                    auxiliary_gradients=nextlat_trunk_grads,
                    # Each nonempty group was already checked once by its global clip.
                    validate_finite=False,
                )
                if nextlat_spec.enabled:
                    apply_private_predictor_step(
                        nextlat_predictor_params,
                        predictor_optimizer,
                        nextlat_predictor_grads,
                    )

            epochs_completed = epoch + 1
            epoch_mean_kl = epoch_kl_sum / epoch_kl_count
            update_scalar_max_(rollout_max_epoch_mean_approx_kl, epoch_mean_kl)
            if actor_active:
                actor_epochs_completed = epoch + 1
                # Circuit breaker (NOT an epoch break): past 3x the per-update KL
                # budget the actor stops, but the critic keeps training all epochs.
                # One epoch-level control-flow synchronization replaces one sync per
                # minibatch while retaining v13's float64 mean and breaker boundary.
                if epoch_mean_kl.item() > args.tpo_kl_breaker:
                    actor_active = False
            # target_kl (default None here) would also stop the critic; kept only as
            # an explicit opt-in escape hatch.
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        mechanism_telemetry = {}
        if nextlat_spec.enabled:
            # These reductions run once on the final logged minibatch, over the
            # already-cloned gradient maps. They never add per-minibatch scans.
            mechanism_telemetry = {
                "nextlat_mechanism/delivered_aux_trunk_norm": gradient_group_norm(
                    nextlat_trunk_params, nextlat_trunk_grads
                ),
                "nextlat_mechanism/aux_actor_cosine": gradient_group_cosine(
                    nextlat_trunk_params, nextlat_trunk_grads, actor_grads
                ),
                "nextlat_mechanism/aux_critic_cosine": gradient_group_cosine(
                    nextlat_trunk_params, nextlat_trunk_grads, value_grads
                ),
                "nextlat_mechanism/aux_task_cosine": gradient_group_cosine(
                    nextlat_trunk_params,
                    nextlat_trunk_grads,
                    actor_grads,
                    value_grads,
                ),
            }

        endpoint_telemetry = {}
        counterfactual_telemetry = {}
        if nextlat_spec.enabled and last_nextlat_mb_inds is not None:
            # Compute endpoint quality once, after all optimizer steps, on the final
            # logged minibatch. This diagnostic-only MSE path never enters autograd or
            # the per-minibatch training hot path.
            with torch.no_grad():
                diagnostic_indices = last_nextlat_mb_inds
                diagnostic_action_indices = (
                    diagnostic_indices[None, :] + nextlat_action_offsets
                ).clamp_max(args.batch_size - 1)
                diagnostic_target_indices = (
                    diagnostic_indices[None, :] + nextlat_target_offsets
                ).clamp_max(args.batch_size - 1)
                diagnostic_actions = b_actions[diagnostic_action_indices]
                diagnostic_targets = frozen_actor_feats[diagnostic_target_indices]
                diagnostic_masks = b_nextlat_mask[diagnostic_indices].T
                diagnostic_source = agent.get_actor_feat(b_obs[diagnostic_indices])
                normal_predictions = multiscale_prediction_forward(
                    agent, diagnostic_source, diagnostic_actions
                )
                normalized_endpoint_errors = normalized_endpoint_errors_vs_persistence(
                    normal_predictions,
                    diagnostic_source,
                    diagnostic_targets,
                    diagnostic_masks,
                )
                for horizon_index, horizon in enumerate(nextlat_spec.horizons):
                    endpoint_telemetry[
                        f"nextlat_h{horizon}/normalized_error_vs_persistence"
                    ] = normalized_endpoint_errors[horizon_index]

                if args.nextlat_shuffle_diagnostics:
                    # ``roll`` replaces each sample's whole action prefix or source.
                    # It does not permute temporal order within a prefix, consumes no
                    # RNG, and never touches optimizer state.
                    action_replacement_predictions = multiscale_prediction_forward(
                        agent, diagnostic_source, diagnostic_actions.roll(1, dims=1)
                    )
                    source_replacement_predictions = multiscale_prediction_forward(
                        agent, diagnostic_source.roll(1, dims=0), diagnostic_actions
                    )
                    diagnostic_losses = compute_nextlat_loss(
                        agent, normal_predictions, diagnostic_targets, diagnostic_masks,
                        rollout_target_spreads, args.nextlat_loss_scale,
                    )
                    action_replacement_losses = compute_nextlat_loss(
                        agent, action_replacement_predictions, diagnostic_targets,
                        diagnostic_masks, rollout_target_spreads, args.nextlat_loss_scale,
                    )
                    source_replacement_losses = compute_nextlat_loss(
                        agent, source_replacement_predictions, diagnostic_targets,
                        diagnostic_masks, rollout_target_spreads, args.nextlat_loss_scale,
                    )
                    baseline = diagnostic_losses["prediction"].clamp_min(1e-8)
                    counterfactual_telemetry.update(
                        {
                            "nextlat_counterfactual/baseline_prediction": diagnostic_losses["prediction"],
                            "nextlat_counterfactual/action_prefix_replacement_prediction": action_replacement_losses["prediction"],
                            "nextlat_counterfactual/source_replacement_prediction": source_replacement_losses["prediction"],
                            "nextlat_counterfactual/action_prefix_replacement_loss_ratio": action_replacement_losses["prediction"] / baseline,
                            "nextlat_counterfactual/source_replacement_loss_ratio": source_replacement_losses["prediction"] / baseline,
                        }
                    )
                    for horizon_index, horizon in enumerate(nextlat_spec.horizons):
                        counterfactual_telemetry.update(
                            {
                                f"nextlat_h{horizon}/action_prefix_replacement_prediction": action_replacement_losses[
                                    "per_horizon_prediction"
                                ][horizon_index],
                                f"nextlat_h{horizon}/source_replacement_prediction": source_replacement_losses[
                                    "per_horizon_prediction"
                                ][horizon_index],
                            }
                        )

        var_y = b_returns.var(correction=0)
        explained_var = torch.where(
            var_y == 0,
            torch.full_like(var_y, float("nan")),
            1.0 - (b_returns - b_values).var(correction=0) / var_y,
        )
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean()
        device_telemetry = {
            "losses/value_loss": v_loss,
            "losses/policy_loss": pg_loss,
            "losses/entropy": entropy_loss,
            "losses/old_approx_kl": old_approx_kl,
            "losses/approx_kl": approx_kl,
            "losses/clipfrac": clipfrac_sum / clipfrac_count,
            "kl_diagnostics/rollout_max_minibatch_approx_kl": rollout_max_minibatch_approx_kl,
            "kl_diagnostics/rollout_max_epoch_mean_approx_kl": rollout_max_epoch_mean_approx_kl,
            "kl_diagnostics/rollout_max_minibatch_old_approx_kl": rollout_max_minibatch_old_approx_kl,
            "losses/explained_variance": explained_var,
            "losses/actor_grad_norm": actor_gn,
            "losses/critic_grad_norm": critic_gn,
            "losses/tpo_ce": tpo_ce,
            "debug/returns_mean": b_returns.mean(),
            "debug/returns_std": b_returns.std(),
            "debug/returns_absmax": b_returns.abs().max(),
            "debug/target_edge_mass": edge_mass,
            "debug/distpg_corr_with_gae": adv_corr,
            "debug/distpg_sign_agree": adv_sign_agree,
            "debug/u_edge_frac": u_edge_frac,
            "debug/sigma_mean": b_sigma.mean(),
        }
        if auto_alpha:
            device_telemetry.update(
                {
                    "losses/alpha": log_alpha.exp(),
                    "debug/squashed_entropy": (-logprobs).mean(),
                    "debug/soft_bootstrap_bonus": next_value_bonus.mean(),
                    "debug/soft_adv_std_ratio": policy_adv.std()
                    / (advantages.std() + 1e-8),
                }
            )
        if nextlat_spec.enabled:
            device_telemetry.update(
                {
                    "losses/nextlat_prediction": nextlat_pred_loss,
                    "losses/nextlat_policy_kl": nextlat_kl_loss,
                    "losses/nextlat_grad_norm": nextlat_trunk_gn,
                    "losses/nextlat_trunk_grad_norm": nextlat_trunk_gn,
                    "losses/nextlat_predictor_grad_norm": nextlat_predictor_gn,
                    "debug/nextlat_latent_std": latent_batch_std,
                    "debug/nextlat_latent_scale": latent_scale,
                    "debug/nextlat_latent_participation_rank": latent_participation_rank,
                }
            )
            for horizon_index, horizon in enumerate(nextlat_spec.horizons):
                device_telemetry.update(
                    {
                        f"nextlat_h{horizon}/target_spread": rollout_target_spreads[horizon_index],
                        f"nextlat_h{horizon}/valid_fraction": rollout_valid_fractions[horizon_index],
                        f"nextlat_h{horizon}/prediction": nextlat_losses[
                            "per_horizon_prediction"
                        ][horizon_index],
                        f"nextlat_h{horizon}/policy_kl": nextlat_losses[
                            "per_horizon_policy_kl"
                        ][horizon_index],
                    }
                )
            device_telemetry.update(endpoint_telemetry)
            device_telemetry.update(counterfactual_telemetry)
            device_telemetry.update(mechanism_telemetry)
        host_telemetry = synchronize_scalar_telemetry(device_telemetry)
        writer.add_scalar(
            "charts/learning_rate", task_optimizer.param_groups[0]["lr"], global_step
        )
        for tag, value in host_telemetry.items():
            writer.add_scalar(tag, value, global_step)
        if auto_alpha:
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
        for tag, value in {
            "debug/epochs_completed": epochs_completed,
            "debug/actor_epochs_completed": actor_epochs_completed,
            "debug/tpo_eta_solved": tpo_eta_solved,
            "debug/tpo_kl_achieved": tpo_kl_achieved,
            "debug/tpo_kl_base": tpo_kl_base,
            "debug/tpo_cap_engaged": tpo_cap_engaged,
            "debug/tpo_group_kl": tpo_group_kl,
            "debug/tpo_score_std_mean": tpo_score_std_mean,
            "debug/tpo_score_std_p90": tpo_score_std_p90,
            "debug/tpo_sigma_global": tpo_sigma_global,
            "debug/tpo_q_entropy": tpo_q_entropy,
        }.items():
            writer.add_scalar(tag, value, global_step)
        if nextlat_spec.enabled:
            writer.add_scalar(
                "nextlat_multiscale/predictor_only_delivery",
                float(not actor_active),
                global_step,
            )
            writer.add_scalar("nextlat_design/predictor_parameters", predictor_parameter_count, global_step)
            writer.add_scalar(
                "nextlat_design/recursive_reference_parameters",
                recursive_reference_parameter_count,
                global_step,
            )
            writer.add_scalar(
                "nextlat_design/predictor_parameter_ratio", predictor_parameter_ratio, global_step
            )
            writer.add_scalar(
                "nextlat_design/predictor_forward_flops", predictor_forward_flops, global_step
            )
            writer.add_scalar(
                "nextlat_design/recursive_reference_forward_flops",
                recursive_reference_forward_flops,
                global_step,
            )
            writer.add_scalar(
                "nextlat_design/predictor_flop_ratio", predictor_flop_ratio, global_step
            )
        for tag, value in tail_risk_statistics(
            completed_episode_returns, tail_thresholds
        ).items():
            writer.add_scalar(tag, value, global_step)
        writer.add_scalar("charts/probe_sps_overhead", probe_seconds, global_step)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)
        del target_probs, b_target_probs

    envs.close()
    writer.close()

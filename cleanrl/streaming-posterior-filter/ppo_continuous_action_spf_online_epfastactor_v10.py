# Streaming episodic smooth fast-actor plasticity v10.
# Family contract: cleanrl/streaming-posterior-filter/FAMILY.md
#
# V9 learned useful local Beta-logit corrections by 1M but repeatedly collapsed
# because its fast maps crossed episode boundaries and used a hard output clamp.
# V10 resets each stream after its terminal TD credit and uses a smooth bounded
# correction with its exact Jacobian. No Q critic, replay, rollout, repeated
# epoch, batch advantage normalization, or BPTT.
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
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
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
    num_envs: int = 64
    num_steps: int = 1
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 1
    update_epochs: int = 1
    norm_adv: bool = False
    # --- Percentile advantage normalization (the sole advantage scaler) ---
    ret_percnorm: bool = False       # ppoadvnorm_batch base: OFF (norm_adv_scope="batch" is
    #                                  the sole advantage scaler). When on: S = max(floor, P95-P5) of returns.
    ret_perc_scope: str = "minibatch"  # "minibatch" = fresh per-mb P95-P5 of mb returns (NO EMA, like the
    #                                  old per-mb advnorm); "batch" = fresh WHOLE-ROLLOUT P95-P5, one S per
    #                                  update, NO EMA (the batch-vs-mb ablation); "ema" = v1's global EMA.
    ret_perc_rate: float = 0.01      # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0            # Ent-PPO: keep 0. The α·KL term IS the entropy bonus (see header).
    # --- ENT-PPO: the single knob. α augments the reward with the pathwise entropy
    # −log π_old(a_t|s_t) AND sets the coefficient of the exact KL(π_new‖π_old) proximal
    # term; the derivation forces them to be the same number. α=0 => the base recipe.
    ent_alpha: float = 0.0           # STATEDYNLR base: 0 (the retstd_batch reference; Ent-PPO OFF)
    # Ablation-only multiplier that BREAKS the derivation's tie between the reward bonus and
    # the proximal coefficient (1.0 = faithful Ent-PPO). Use it to test whether the exact
    # coefficient is actually the right one, not as a tuning knob.
    kl_coef_scale: float = 1.0
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
    target_kl: Optional[float] = None

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Per-environment robust Delta memory over normalized observations.
    fast_memory_eta: float = 0.05
    fast_memory_student_df: float = 5.0
    fast_memory_scale_rate: float = 1e-3
    fast_memory_context_clip: float = 10.0

    # Local score-function fast weights on the two Beta concentration heads.
    fast_actor_eta: float = 0.002
    fast_actor_leak: float = 0.9999
    fast_actor_offset_clip: float = 2.0
    fast_actor_signal_clip: float = 5.0
    fast_actor_score_clip: float = 5.0

    # Dreamer3-style bucket critic: centers are symexp(linspace(v_min, v_max, num_bins)).
    # Defaults match v162's ±20k raw support, expressed in symlog coordinates.
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 1

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

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def validate_online_contract(args):
    """Reject configurations that invalidate local streaming fast-actor credit."""
    required = {
        "num_steps": 1,
        "num_minibatches": 1,
        "update_epochs": 1,
        "norm_adv": False,
        "ret_percnorm": False,
        "adv_transform": "v10",
        "critic_mtp_horizon": 1,
        "target_kl": None,
        "ent_alpha": 0.0,
        "ent_coef": 0.0,
        "auto_entropy": False,
        "actor_dist": "beta",
    }
    violations = [
        f"{name}={getattr(args, name)!r} (required {expected!r})"
        for name, expected in required.items()
        if getattr(args, name) != expected
    ]
    if args.fast_actor_eta <= 0:
        violations.append("fast_actor_eta must be positive")
    if not 0 < args.fast_actor_leak <= 1:
        violations.append("fast_actor_leak must be in (0, 1]")
    if min(
        args.fast_actor_offset_clip,
        args.fast_actor_signal_clip,
        args.fast_actor_score_clip,
    ) <= 0:
        violations.append("fast actor clips must be positive")
    if violations:
        raise ValueError(
            "online episodic fastactor v10 requires one fresh transition per stream and "
            f"one optimizer use; invalid settings: {', '.join(violations)}"
        )


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



class RobustFastDynamics:
    def __init__(self, num_envs, obs_dim, args, device):
        self.obs_dim = obs_dim
        self.eta = args.fast_memory_eta
        self.student_df = args.fast_memory_student_df
        self.scale_rate = args.fast_memory_scale_rate
        self.context_clip = args.fast_memory_context_clip
        self.memory = torch.zeros((num_envs, obs_dim, obs_dim), device=device)
        self.scale = torch.ones(num_envs, device=device)
        self.diagnostic_sum = torch.zeros(5, device=device)
        self.diagnostic_steps = 0

    @torch.no_grad()
    def read(self, obs):
        key = F.normalize(obs, dim=-1)
        prediction = torch.einsum("bij,bj->bi", self.memory, key)
        return prediction.clamp(-self.context_clip, self.context_clip)

    @torch.no_grad()
    def update(self, obs, next_obs, valid):
        key = F.normalize(obs, dim=-1)
        prediction = torch.einsum("bij,bj->bi", self.memory, key)
        innovation = next_obs - prediction
        normalized_error = innovation.square().mean(1) / self.scale
        student_weight = (
            (self.student_df + 1.0)
            / (self.student_df + normalized_error)
        ).clamp_max(1.0)
        update = (
            (self.eta * student_weight).view(-1, 1, 1)
            * innovation.unsqueeze(2)
            * key.unsqueeze(1)
        )
        valid_bool = valid.to(dtype=torch.bool)
        self.memory.add_(update * valid_bool.view(-1, 1, 1))
        scale_target = student_weight * innovation.square().mean(1)
        next_scale = torch.lerp(self.scale, scale_target, self.scale_rate)
        self.scale.copy_(
            torch.where(valid_bool, next_scale, self.scale)
            .clamp_min(torch.finfo(self.scale.dtype).tiny)
        )

        valid_float = valid_bool.to(dtype=prediction.dtype)
        count = valid_float.sum().clamp_min(1.0)
        self.diagnostic_sum.add_(
            torch.stack(
                (
                    (student_weight * valid_float).sum() / count,
                    ((student_weight < 0.5).to(prediction.dtype) * valid_float).sum() / count,
                    (self.scale * valid_float).sum() / count,
                    (update.square().mean((1, 2)).sqrt() * valid_float).sum() / count,
                    (prediction.square().mean(1).sqrt() * valid_float).sum() / count,
                )
            )
        )
        self.diagnostic_steps += 1

    @torch.no_grad()
    def diagnostics(self):
        if self.diagnostic_steps == 0:
            return {}
        values = self.diagnostic_sum / self.diagnostic_steps
        self.diagnostic_sum.zero_()
        self.diagnostic_steps = 0
        return {
            "fast_memory/student_weight": values[0].item(),
            "fast_memory/outlier_fraction": values[1].item(),
            "fast_memory/scale": values[2].item(),
            "fast_memory/update_rms": values[3].item(),
            "fast_memory/prediction_rms": values[4].item(),
            "fast_memory/memory_rms": self.memory.square().mean().sqrt().item(),
        }


class LocalFastActor:
    """Episode-local Beta-logit maps updated by one-step V-TD scores."""

    def __init__(self, num_envs, act_dim, obs_dim, args, device):
        shape = (num_envs, act_dim, obs_dim)
        self.alpha_weight = torch.zeros(shape, device=device)
        self.beta_weight = torch.zeros(shape, device=device)
        self.eta = args.fast_actor_eta
        self.leak = args.fast_actor_leak
        self.offset_clip = args.fast_actor_offset_clip
        self.signal_clip = args.fast_actor_signal_clip
        self.score_clip = args.fast_actor_score_clip
        self.diagnostic_sum = torch.zeros(5, device=device)
        self.diagnostic_steps = 0

    @torch.no_grad()
    def _raw_offsets(self, obs):
        key = F.normalize(obs, dim=-1)
        return (
            key,
            torch.einsum("bao,bo->ba", self.alpha_weight, key),
            torch.einsum("bao,bo->ba", self.beta_weight, key),
        )

    @torch.no_grad()
    def read(self, obs):
        _, raw_alpha_offset, raw_beta_offset = self._raw_offsets(obs)
        return (
            self.offset_clip * torch.tanh(raw_alpha_offset / self.offset_clip),
            self.offset_clip * torch.tanh(raw_beta_offset / self.offset_clip),
        )

    @torch.no_grad()
    def update(
        self,
        obs,
        z,
        alpha,
        beta,
        alpha_jacobian,
        beta_jacobian,
        td_error,
        boundary,
    ):
        key, raw_alpha_offset, raw_beta_offset = self._raw_offsets(obs)
        z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        common = torch.digamma(alpha + beta)
        alpha_score = alpha_jacobian * (
            z.log() - torch.digamma(alpha) + common
        )
        beta_score = beta_jacobian * (
            torch.log1p(-z) - torch.digamma(beta) + common
        )
        alpha_offset_jacobian = 1.0 - torch.tanh(
            raw_alpha_offset / self.offset_clip
        ).square()
        beta_offset_jacobian = 1.0 - torch.tanh(
            raw_beta_offset / self.offset_clip
        ).square()
        alpha_score.mul_(alpha_offset_jacobian)
        beta_score.mul_(beta_offset_jacobian)
        alpha_score.clamp_(-self.score_clip, self.score_clip)
        beta_score.clamp_(-self.score_clip, self.score_clip)
        signal = td_error.clamp(-self.signal_clip, self.signal_clip)
        alpha_update = (
            self.eta
            * signal.view(-1, 1, 1)
            * alpha_score.unsqueeze(2)
            * key.unsqueeze(1)
        )
        beta_update = (
            self.eta
            * signal.view(-1, 1, 1)
            * beta_score.unsqueeze(2)
            * key.unsqueeze(1)
        )
        self.alpha_weight.mul_(self.leak).add_(alpha_update)
        self.beta_weight.mul_(self.leak).add_(beta_update)
        keep = (1.0 - boundary).view(-1, 1, 1)
        self.alpha_weight.mul_(keep)
        self.beta_weight.mul_(keep)

        alpha_offset, beta_offset = self.read(obs)
        score_rms = (
            0.5 * (alpha_score.square().mean() + beta_score.square().mean())
        ).sqrt()
        update_rms = (
            0.5 * (alpha_update.square().mean() + beta_update.square().mean())
        ).sqrt()
        offset_rms = (
            0.5 * (alpha_offset.square().mean() + beta_offset.square().mean())
        ).sqrt()
        weight_rms = (
            0.5
            * (
                self.alpha_weight.square().mean()
                + self.beta_weight.square().mean()
            )
        ).sqrt()
        self.diagnostic_sum.add_(
            torch.stack(
                (
                    signal.abs().mean(),
                    score_rms,
                    update_rms,
                    offset_rms,
                    weight_rms,
                )
            )
        )
        self.diagnostic_steps += 1

    @torch.no_grad()
    def diagnostics(self):
        if self.diagnostic_steps == 0:
            return {}
        values = self.diagnostic_sum / self.diagnostic_steps
        self.diagnostic_sum.zero_()
        self.diagnostic_steps = 0
        return {
            "fast_actor/signal_abs_mean": values[0].item(),
            "fast_actor/score_rms": values[1].item(),
            "fast_actor/update_rms": values[2].item(),
            "fast_actor/offset_rms": values[3].item(),
            "fast_actor/weight_rms": values[4].item(),
        }


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        trunk_input_dim = 2 * obs_dim
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(trunk_input_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(trunk_input_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(trunk_input_dim, H, args.k_blocks, args.n_experts)
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

    def _actor_dist(self, actor_feat, fast_offsets=None):
        # Build the action distribution and the native-space transforms.
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            std = (0.5 * lv).exp()
            dist = Normal(mean, std)
            to_action = torch.tanh
            log_det_fn = lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
            return dist, to_action, log_det_fn, (None, None)
        raw_alpha = self.actor_alpha_head(actor_feat)
        raw_beta = self.actor_beta_head(actor_feat)
        if fast_offsets is not None:
            raw_alpha = raw_alpha + fast_offsets[0]
            raw_beta = raw_beta + fast_offsets[1]
        alpha = 1.0 + F.softplus(raw_alpha)
        beta = 1.0 + F.softplus(raw_beta)
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0
        return (
            dist,
            to_action,
            log_det_fn,
            (torch.sigmoid(raw_alpha), torch.sigmoid(raw_beta)),
        )

    def _trunks(self, x, fast_context):
        # The stored context is the exact causal memory read used by the
        # behavior policy; PPO replay never consults the later live memory.
        features = torch.cat((x, fast_context), dim=-1)
        if self.share_backbone:
            feat = self.trunk(features)
            return feat, feat
        return self.actor_trunk(features), self.critic_trunk(features)

    def get_value(self, x, fast_context):
        # Returns value LOGITS (B, mtp, num_bins); horizon 0 is V(s_t).
        _, critic_feat = self._trunks(x, fast_context)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_action_and_value(self, x, fast_context, z=None, fast_offsets=None):
        # Stored z, context, and fast offsets exactly reproduce the behavior sample.
        actor_feat, critic_feat = self._trunks(x, fast_context)
        dist, to_action, log_det_fn, raw_jacobians = self._actor_dist(
            actor_feat,
            fast_offsets,
        )
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
        return action, z, log_prob, entropy, value_logits, dist, raw_jacobians[0], raw_jacobians[1]

    # ENT-PPO: the analytic KL(π_new‖π_old) needs the OLD distribution, so the rollout
    # stores its two head parameters and the update rebuilds it (parameters, not samples:
    # the KL is all-action/exact, not an importance-weighted single-sample estimate).
    def dist_params(self, dist):
        if self.actor_dist == "gaussian":
            return dist.loc, dist.scale
        return dist.concentration1, dist.concentration0

    def rebuild_dist(self, p1, p2):
        if self.actor_dist == "gaussian":
            # KL is invariant under the tanh bijection, so base-Normal KL == squashed KL.
            return Normal(p1, p2)
        return Beta(p1, p2)

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
    validate_online_contract(args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch", "batch_retstd", "minibatch_retstd")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope in ("batch", "batch_retstd")), \
        "norm_adv_scope=batch/batch_retstd requires adv_transform_scope=batch"
    # Both inject entropy into the advantage; together they double-count it (and auto_entropy
    # keeps the critic entropy-free, which contradicts Ent-PPO's soft value function).
    assert not (args.ent_alpha != 0.0 and args.auto_entropy), \
        "ent_alpha (soft-MDP reward, Ent-PPO) and --auto-entropy (soft-advantage bootstrap) are mutually exclusive"
    assert not (args.ent_alpha != 0.0 and args.ent_coef != 0.0), \
        "ent_coef must stay 0 under Ent-PPO: the alpha*KL proximal term already is the entropy bonus"
    # The proximal term is only well-posed if `adv_div` really is the full scale applied to the
    # advantage. Two supported flags put a factor OUTSIDE that product, so refuse them here
    # rather than silently mis-scaling alpha: (a) any non-identity adv_transform is a
    # non-homogeneous map (tanh saturates; rankgauss* discard the GAE magnitude outright), so
    # after it the advantage is not in return units at all; (b) pos_neg_alpha reweights each
    # sample by a SIGN-dependent factor, which no scalar divisor can represent.
    assert not (args.ent_alpha != 0.0 and args.adv_transform != "v10"), \
        "Ent-PPO needs adv_transform=v10: a non-identity shaping leaves the advantage in unknown units, so alpha/adv_div is no longer the derivation's coefficient"
    assert not (args.ent_alpha != 0.0 and args.pos_neg_alpha != 0.5), \
        "Ent-PPO needs pos_neg_alpha=0.5: its per-sample sign-dependent reweight cannot be folded into adv_div"
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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    fast_memory = RobustFastDynamics(args.num_envs, obs_dim, args, device)
    act_dim = int(np.prod(envs.single_action_space.shape))
    fast_actor = LocalFastActor(
        args.num_envs,
        act_dim,
        obs_dim,
        args,
        device,
    )
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

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
    hl_support_cpu = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )
    support = hl_support.support                       # Dreamer3 raw bucket centers
    bin_width = hl_support.bin_width
    raw_support = support

    def value_logits_to_scalar(logits):
        return hl_support.to_scalar(logits)

    sigma_floor = args.sigma_floor_bins * bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    contexts = torch.zeros_like(obs)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    fast_alpha_offsets = torch.zeros_like(actions)
    fast_beta_offsets = torch.zeros_like(actions)
    # Ent-PPO: behavior-policy head parameters (Beta: concentration1/concentration0;
    # Gaussian: loc/scale) for the exact proximal KL.
    dist_p1 = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    dist_p2 = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    fast_alpha_jacobians = torch.zeros_like(actions)
    fast_beta_jacobians = torch.zeros_like(actions)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    next_contexts = torch.zeros_like(next_obses)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_context = fast_memory.read(next_obs)
    next_done = torch.zeros(args.num_envs).to(device)
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            contexts[step] = next_context
            dones[step] = next_done

            alpha_offset, beta_offset = fast_actor.read(next_obs)
            with torch.no_grad():
                (
                    action,
                    z,
                    logprob,
                    ent,
                    value_logits,
                    dist,
                    alpha_jacobian,
                    beta_jacobian,
                ) = agent.get_action_and_value(
                    next_obs,
                    next_context,
                    fast_offsets=(alpha_offset, beta_offset),
                )
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            p1, p2 = agent.dist_params(dist)
            dist_p1[step] = p1
            dist_p2[step] = p2
            fast_alpha_offsets[step] = alpha_offset
            fast_beta_offsets[step] = beta_offset
            fast_alpha_jacobians[step] = alpha_jacobian
            fast_beta_jacobians[step] = beta_jacobian

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
            transition_valid_tensor = torch.as_tensor(
                transition_valid,
                device=device,
                dtype=torch.float32,
            )
            transition_next_obs_tensor = torch.as_tensor(
                transition_next_obs,
                device=device,
                dtype=torch.float32,
            )
            transition_valids[step] = transition_valid_tensor
            next_obses[step] = transition_next_obs_tensor
            fast_memory.update(obs[step], transition_next_obs_tensor, transition_valid_tensor)
            next_contexts[step] = fast_memory.read(transition_next_obs_tensor)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_context = fast_memory.read(next_obs)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_transition_value_logits = agent.get_value(
                next_obses.reshape((-1,) + envs.single_observation_space.shape),
                next_contexts.reshape((-1,) + envs.single_observation_space.shape),
            )[:, 0]
            next_transition_values = value_logits_to_scalar(next_transition_value_logits).reshape(
                args.num_steps, args.num_envs
            )
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample from every aligned transition successor. Shifting the
                # rollout log-probs would pair a truncation's final observation
                # with the following auto-reset state.
                _, _, next_logprob, _, _, _, _, _ = agent.get_action_and_value(
                    next_obses.reshape((-1,) + envs.single_observation_space.shape),
                    next_contexts.reshape((-1,) + envs.single_observation_space.shape),
                )
                alpha_r = log_alpha.exp().detach()
                next_value_bonus = alpha_r * (
                    -next_logprob.reshape(args.num_steps, args.num_envs)
                )
            else:
                next_value_bonus = None
            # ENT-PPO SOFT MDP: r̃_t = r_t + α·(−log π_old(a_t|s_t)). The pathwise entropy
            # bonus lives INSIDE the reward, so (a) GAE propagates it => an action is credited
            # for the entropy of the states it leads to, and (b) the critic regresses the SOFT
            # return, i.e. it learns the soft value Ṽ the derivation assumes.
            # UNITS: logprobs is the Beta density in the NATIVE z-space (the unit cube), i.e.
            # log_det_fn=0 for the affine z→action map, so this is action-space surprisal plus
            # MINUS the constant |A|·log 2 (a = low + (high−low)·z with high−low = 2, so
            # p_a(a) = p_z(z)/2^|A|). A constant per-step reward shifts every return by
            # c/(1−γ) and cancels exactly in delta_t once the critic fits, so the choice of
            # convention has NO gradient effect; only debug/entropy_bonus_mean reads |A|·log 2
            # LOWER than an action-space entropy would (on HalfCheetah 6·ln2 = 4.16 nats:
            # a logged −7.24 is an action-space differential entropy of −3.08). The KL term is reparameterization-
            # invariant, so the α·H(π_new) identity holds in whichever convention is used.
            # SUPPORT: the bonus is α·(a few nats) against ~6 reward/step, and the ±20k symlog
            # support has ~22x headroom over HalfCheetah's discounted return (MEASURED:
            # debug/returns_absmax peaks at 918 with the bonus vs 943 without, i.e. the bonus
            # barely moves return MAGNITUDE -- its effect is on ordering/variance), so the target
            # still fits (unlike the auto_entropy softboot path, whose auto-tuned alpha
            # overflowed it -- watch debug/target_edge_mass).
            soft_rewards = rewards + args.ent_alpha * (-logprobs)
            # SOFT GAE: critic-consistent soft advantage + soft return.
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = soft_rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
            # Effect-size probe, exact (GAE is linear in the reward sequence at fixed V):
            # A_soft = A_reward + E with E = GAE_λ(α·(−log π_old)) and no value terms.
            # std(E)/std(A_soft) is the entropy share of the policy signal; ≫1 => alpha too big.
            ent_adv = torch.zeros_like(rewards)
            lastentlam = 0
            for t in reversed(range(args.num_steps)):
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                ent_adv[t] = lastentlam = (
                    args.ent_alpha * (-logprobs[t])
                    + args.gamma * args.gae_lambda * lambda_nonterminal * lastentlam
                )
            ent_adv_share = (ent_adv.std() / (advantages.std() + 1e-8)).item()
            ent_ret_shift = ent_adv.mean().item()
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
            fast_actor.update(
                obs[0],
                latent_zs[0],
                dist_p1[0],
                dist_p2[0],
                fast_alpha_jacobians[0],
                fast_beta_jacobians[0],
                advantages[0],
                transition_boundaries[0],
            )
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
            # The full (T,B,MTP,bins) target is large and fixed. Keep it on CPU
            # and move only minibatch labels to CUDA during the value loss.
            target_probs = hl_support_cpu.project(return_mtp.detach().cpu())
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
        b_contexts = contexts.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_fast_alpha_offsets = fast_alpha_offsets.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_fast_beta_offsets = fast_beta_offsets.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()
        b_u = u.reshape(-1)
        b_dist_p1 = dist_p1.reshape((-1,) + envs.single_action_space.shape)
        b_dist_p2 = dist_p2.reshape((-1,) + envs.single_action_space.shape)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        # Ent-PPO scale bookkeeping: the proximal objective is (1/adv_div)·[Â − α·KL], so every
        # divisor applied to the advantage MUST also divide the KL penalty. batch_pre_div covers
        # the whole-rollout percentile scaler already applied to policy_adv above.
        batch_pre_div = (
            ret_perc_scale if (args.ret_percnorm and args.ret_perc_scope in ("ema", "batch")) else 1.0
        )
        batch_norm_div = 1.0
        if args.norm_adv and args.norm_adv_scope == "batch":
            batch_norm_div = b_policy_adv.std() + 1e-8
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / batch_norm_div
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            batch_norm_div = b_returns.std().clamp(min=args.ret_perc_floor)
            b_policy_adv_normed = b_policy_adv / batch_norm_div
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        actor_clip_hits, n_mbs = 0.0, 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                (
                    _,
                    _,
                    newlogprob,
                    entropy,
                    value_logits,
                    new_dist,
                    _,
                    _,
                ) = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_contexts[mb_inds],
                    b_latent_zs[mb_inds],
                    fast_offsets=(
                        b_fast_alpha_offsets[mb_inds],
                        b_fast_beta_offsets[mb_inds],
                    ),
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                adv_div = batch_pre_div          # running product of every divisor hitting the advantage
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                        adv_div = adv_div * batch_norm_div
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_div = b_returns[mb_inds].std().clamp(min=args.ret_perc_floor)
                        mb_advantages = mb_advantages / mb_div
                        adv_div = adv_div * mb_div
                    else:
                        mb_div = mb_advantages.std() + 1e-8
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_div
                        adv_div = adv_div * mb_div
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds]
                    qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device)
                    lo, hi = torch.quantile(mb_ret, qs)
                    mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                    mb_advantages = mb_advantages / mb_perc_scale
                    adv_div = adv_div * mb_perc_scale
                    ret_perc_scale = mb_perc_scale.item()

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

                # ENT-PPO EXACT PROXIMAL TERM (their policy_loss_fn: ppo_clip − KL, coefficient 1).
                #   E_{a~π}[Q̃] + α·H(π) = E_old[ρ·Ã_soft] − α·KL(π_new‖π_old) + const,
                # so this is not a bolted-on trust region: with the reward's −α·log π_old it is an
                # EXACT estimator of α·H(π_new), and α is the same number in both places. All-action
                # and analytic (no importance weight, no clipping), unlike PPO's ratio heuristic;
                # for Beta it is closed form in the concentrations, and it is invariant to the
                # affine z→action rescale (and to tanh in the Gaussian path). Divided by adv_div so
                # the surrogate and the penalty live in the same units.
                exact_kl = torch.distributions.kl_divergence(
                    new_dist, agent.rebuild_dist(b_dist_p1[mb_inds], b_dist_p2[mb_inds])
                ).sum(1)
                kl_coef = args.ent_alpha * args.kl_coef_scale / adv_div
                kl_penalty = kl_coef * exact_kl.mean()

                # v162 HL-Gauss MTP value loss: per-horizon CE to scalar-return
                # targets, summed across valid horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds].to(device=value_logits.device, non_blocking=True)
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(device=value_logits.device, dtype=value_ce.dtype, non_blocking=True)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

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

                if args.separate_grad_clip:
                    # DUAL-BACKWARD decoupled clipping. Backprop value and policy
                    # gradients separately, clip each to its own max-norm, then sum
                    # on the (possibly shared) trunk so the critic's CE gradient
                    # cannot swamp the policy's contribution to shared features.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    # Stash clipped value grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss + kl_penalty - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # Probe whether the actor's independent norm budget binds.
                    actor_clip_hits += float(actor_gn > args.actor_grad_clip)
                    n_mbs += 1
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss + kl_penalty - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_per_h = b_target_probs[..., 0] + b_target_probs[..., -1]
        edge_mask_f = b_target_mask.to(dtype=edge_per_h.dtype)
        edge_mass = ((edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)).item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        for k, v in fast_memory.diagnostics().items():
            writer.add_scalar(k, v, global_step)
        for k, v in fast_actor.diagnostics().items():
            writer.add_scalar(k, v, global_step)
        writer.add_scalar(
            "debug/actor_clip_frac",
            actor_clip_hits / max(1, n_mbs),
            global_step,
        )
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
            writer.add_scalar("debug/soft_adv_std_ratio", (policy_adv.std() / (advantages.std() + 1e-8)).item(), global_step)
        # Ent-PPO: exact all-action KL(π_new‖π_old) at the last minibatch (the penalized
        # quantity) next to the sampled-ratio approx_kl that drives the early-stop leash.
        writer.add_scalar("losses/exact_kl", exact_kl.mean().item(), global_step)
        writer.add_scalar("debug/kl_coef", float(kl_coef), global_step)
        writer.add_scalar("debug/kl_penalty", kl_penalty.detach().item(), global_step)
        writer.add_scalar("debug/entropy_bonus_mean", (-logprobs).mean().item(), global_step)
        writer.add_scalar("debug/ent_adv_share", ent_adv_share, global_step)
        writer.add_scalar("debug/soft_return_shift", ent_ret_shift, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
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

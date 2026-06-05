# PPO + IterThink v74 (v24 Beta + LeWM full-belief + rollout warmup). From v24.
#
# This keeps v24's PPO/Beta/distributional-critic machinery and keeps the
# MLP/MoE ThinkTrunk as the actor/critic trunk. A separate le-wm style auxiliary
# module learns a detached world-model representation:
#   per-feature embeddings -> transformer encoder/predictor -> MTP future tokens.
# The world model is trained on rollout transitions with multi-token prediction:
# teacher-forced latent/action history predicts future obs+outcome target
# embeddings. Summaries include previous reward and continuation outcome tokens;
# targets are label-grounded and regularized on obs tokens. The v24 agent trunk
# reads detached predictor-trunk belief latents, matching the latest le-wm control
# path while preserving the v24 PPO/ThinkTrunk backend.
# v74 aligns WM-only warmup to whole rollouts so PPO never trains on midpoint
# bootstrap actions with fake old logprobs.
#
# --- inherited v24 notes ---
#
# PPO + IterThink v24 (ACTION-DISTRIBUTION TOGGLE: dreamer4-faithful). From v21.
#
# SOFT MAX-ENTROPY add-on (--auto-entropy, gaussian only). This borrows SAC's
# tanh-squashed log-prob, target-entropy heuristic, and temperature dual, but keeps
# the PPO critic on the RAW reward return. Entropy enters the actor two ways:
#   (1) a current-state squashed-entropy actor bonus, -alpha * log pi_sq(a|s);
#   (2) a policy-only soft GAE whose one-step bootstrap adds alpha * H_sq(s_{t+1})
#       using the rollout/bootstrapped squashed log-prob sample.
# The distributional critic target is deliberately entropy-free so the fixed support
# remains calibrated. Off (default) => byte-identical to the v24 base.
#
# WHY v24. The v22/v23 state-dependent Gaussian std hit a 1/sigma^2 pathology
# (confident low-sigma states spike the mean gradient). dreamer4 avoids this two
# ways, and v24 ports BOTH faithfully behind one `--actor-dist` toggle, on the
# UNCHANGED v21 winner machinery (shared backbone, 2-way decoupled clip,
# rankgauss, clip-higher, tkl03) so the ONLY thing that varies is the action
# distribution — a clean A/B.
#
#   actor_dist="beta"  (DEFAULT, the "performs much better" path):
#       unimodal Beta, exactly dreamer4's continuous_dist_type='beta' (which
#       forces unimodal=True) and our beta_relusq:
#           alpha = 1 + softplus(head_a);  beta = 1 + softplus(head_b)   (>=1 => unimodal)
#       native support (0,1) is linearly rescaled to the env action range
#       [low, high]. Sampling clamps z to [eps, 1-eps]; log_prob/entropy are the
#       closed-form Beta values in native z-space (the constant rescale Jacobian
#       is dropped — it cancels in the PPO ratio and the entropy is a constant
#       offset). Bounded support => no squash saturation, no 1/sigma^2 blow-up,
#       no boundary mass leak, no bang-bang (unimodal).
#
#   actor_dist="gaussian"  (the matched control = state-dependent Gaussian scale):
#       dreamer4's Gaussian readout. This is NOT SAC's exact log-std head. It is a
#       state-dependent log-VARIANCE head (not a flat Parameter, not log-std),
#       SOFT-bounded by dreamer4's tanh-rescale (not a hard clamp, so the gradient
#       never dies at the bound):
#           lv  = rescale(tanh(raw_lv/(hi-lo)), (-1,1), (lo,hi))   # symmetric (lo,hi) => lv=0 at init
#           std = exp(0.5 * lv)
#       then the iterthink/SAC tanh-squash + stable Jacobian on the sample (mean
#       stays raw). SAC continuous-action instead uses a state-dependent log_std
#       head bounded to [-5, 2] and std = exp(log_std). Here logvar [-8, 8] implies
#       log_std [-4, 4], so the family matches but the scale parameterization and
#       bounds do not.
#
# PARITY NOTES (both dists): the rollout buffers the distribution-NATIVE sample
# (latent_zs) — pre-tanh z for gaussian, z in (0,1) for beta — and replays it on
# the update pass, so log_prob is recomputed at the same sample (identical to
# v21's z-replay). `actions` holds the env action (tanh(z) / rescaled z). The
# gaussian path is bit-identical to v21 except the flat logstd -> dreamer4 head.
# Bar to beat: v21 flat-Gaussian = 8774.
#
# --- inherited v21 notes ---
# PPO + IterThink v21 (SHARED BACKBONE + DECOUPLED GRAD CLIP). From v19.
#
# WHY v21. v19 used two independent ThinkTrunks (one actor, one critic). The
# classic MuJoCo-PPO result is that shared backbones LOSE, because the value
# loss gradient dominates the shared trunk and corrupts the policy's features.
# v21 tests whether we can have the representation-sharing benefit WITHOUT that
# cost, by decoupling the gradient magnitudes:
#   - share_backbone: one ThinkTrunk feeds both the actor head and the
#     (distributional) critic head; trunk is computed once per forward.
#   - separate_grad_clip: DUAL-BACKWARD clipping. The value gradient
#     (vf_coef * v_loss) and the policy gradient (pg_loss - ent) are each
#     backpropped and clipped to their OWN max-norm (critic_grad_clip /
#     actor_grad_clip), then summed on the shared trunk:
#         trunk.grad = clip_actor(d pg / d trunk) + clip_critic(d vl / d trunk)
#     so the distributional critic's large CE gradient can no longer swamp the
#     shared features. NOTE: the trunk's effective budget is the SUM of the two
#     clips, so each defaults to 0.25 (sum ~= v19's single 0.5 global clip).
# This is targeted: rankgauss already bounds the POLICY gradient (rank-only adv),
# so the dominant imbalance on a shared trunk is the critic -> clip it apart.
# Built on the v19 winner: adv_transform="rankgauss" + clip-higher (0.2/0.28).
# Both knobs are toggles, so this file also runs the {shared,separate} x
# {global,decoupled-clip} 2x2. The bar to beat: rankgauss_cliphigh ~= 8292 (towers).
#
# --- inherited v19 notes ---
# PPO + IterThink v19 (ADVANTAGE SHAPING — magnitude-preserving + attribution). From v17.
#
# WHY v19. A subagent review of v17 (CDF-rank distributional PG) found that in its
# STABLE regime the categorical critic is overconfident, so u=F_Z(G) is bimodal at
# 0/1, the probit saturates, and the advantage DEGENERATES to ≈sign(GAE) (corr 0.92);
# norm_adv then re-standardizes the ±3.3 spikes to ≈±1 binary. So v17 discards the
# advantage MAGNITUDE (the thing PPO needs) and is really a sign-of-TD-error update
# made trainable by KL control. v17's 5867@4M conflates THREE possible causes — the
# distribution, a bounded/outlier-robust advantage, and KL control — introduced at
# once. v19 disentangles them and adds the principled fix, via one `adv_transform`:
#
#   "v10"      : raw GAE (== v10 / dist_pg off). Baseline.
#   "cdf_probit": v17's CDF-rank u -> Phi^-1(u). Reference.
#   "tanh_std" : A~ = tanh( GAE_t / (kappa * sigma(s_t)) ).  THE FIX. Per-state
#                normalized by the critic's return std sigma(s) (v16's good idea),
#                but BOUNDED by tanh (fixes v16's blowup: tiny sigma -> saturate, not
#                explode) AND magnitude-preserving near 0 (fixes v17's sign-collapse:
#                linear in GAE for |GAE|<kappa*sigma). Note G_t-E[Z_t]=GAE_t exactly.
#   "tanh_gae" : A~ = tanh( zscore(GAE)_t / kappa ).  Robust-GAE CONTROL with NO
#                distribution — isolates "bounded/outlier-robust advantage" from the
#                distributional claim. If this matches v17, the distribution is
#                incidental and this is the cleaner lever.
#
# All paths keep the mean-value GAE and the distributional λ-return value target
# (v10) UNCHANGED; only the policy advantage is reshaped. sigma(s) is the std of the
# OLD rollout Z(s_t), floored at `sigma_floor_bins` bins. Pair with target_kl for the
# 2x2 attribution (v10/tanh_gae/cdf_probit x KL-cap). Control: v17 / v10.
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

from cleanrl.shared.hl_gauss import HLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


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
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ≈ N(0, tau^2), sharp at 0
    value_symlog: bool = False

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
    lewm_dim: int = 64
    lewm_encoder_layers: int = 2
    lewm_predictor_layers: int = 4
    lewm_heads: int = 4
    lewm_kv_heads: int = 2
    lewm_ffn_mult: int = 2
    lewm_context: int = 5
    lewm_dyn_horizon: int = 5
    lewm_mtp_len: int = 4
    lewm_update_epochs: int = 1
    lewm_minibatch_size: int = 256
    wm_warmup_steps: int = 100000
    lewm_loss_coef: float = 1.0
    lewm_reward_loss_coef: float = 0.25
    lewm_termination_loss_coef: float = 0.25
    lewm_reward_num_bins: int = 51
    lewm_reward_v_min: float = -3.0
    lewm_reward_v_max: float = 3.0
    lewm_sigreg_coef: float = 0.09
    lewm_sigreg_num_proj: int = 1024
    lewm_sigreg_knots: int = 17
    lewm_sigreg_min_valid: int = 32
    detach_world_model_from_agent: bool = True

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


def relu_sq(x):
    return torch.relu(x).pow(2)


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer used on obs tokens only."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def sample_projection(self, dim, device, dtype):
        A = torch.randn(dim, self.num_proj, device=device, dtype=dtype)
        return A.div_(A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8))

    def forward(self, proj, A=None):
        # proj: (tokens, valid_samples, dim)
        if A is None:
            A = self.sample_projection(proj.size(-1), proj.device, proj.dtype)
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


def xavier_linear(layer, gain=1.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class LeWMTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_mult):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=False)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.ffn_scale = nn.Parameter(torch.ones(dim))
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult, bias=False))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim, bias=False))
        for name, param in self.attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, causal=False):
        attn_mask = None
        if causal:
            seq = x.shape[1]
            attn_mask = torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu(1)
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.attn_scale.view(1, 1, -1).to(x.dtype) * attn_out
        h = self.ffn_norm(x)
        x = x + self.ffn_scale.view(1, 1, -1).to(x.dtype) * self.w2(relu_sq(self.w1(h)))
        return x


class LeWMAxialPredictorBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_mult, axis):
        super().__init__()
        if axis not in {"space", "time"}:
            raise ValueError(f"unknown predictor axis {axis}")
        self.axis = axis
        self.cond_proj = xavier_linear(nn.Linear(dim, dim, bias=False))
        self.block = LeWMTransformerBlock(dim, num_heads, ffn_mult)

    def forward(self, x, action_features):
        # x: (B, T, S, D), action_features: (B, T, D)
        x = x + self.cond_proj(action_features).unsqueeze(2)
        batch, time_len, space_len, width = x.shape
        if self.axis == "space":
            y = x.reshape(batch * time_len, space_len, width)
            y = self.block(y, causal=False)
            return y.reshape(batch, time_len, space_len, width)
        y = x.permute(0, 2, 1, 3).contiguous().reshape(batch * space_len, time_len, width)
        y = self.block(y, causal=True)
        return y.reshape(batch, space_len, time_len, width).permute(0, 2, 1, 3).contiguous()


class LeWMBackbone(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dim = args.lewm_dim
        self.num_obs_tokens = obs_dim
        self.reward_num_bins = args.lewm_reward_num_bins
        self.num_outcome_tokens = 2
        self.num_latent_tokens = self.num_obs_tokens + self.num_outcome_tokens
        self.context = args.lewm_context
        self.mtp_len = args.lewm_mtp_len

        # Latest le-wm tokenizer stance: one learned affine token per observation scalar.
        self.obs_feature_weight = nn.Parameter(torch.empty(obs_dim, self.dim))
        self.obs_feature_bias = nn.Parameter(torch.empty(obs_dim, self.dim))
        nn.init.xavier_uniform_(self.obs_feature_weight)
        nn.init.zeros_(self.obs_feature_bias)
        self.encoder_layers = nn.ModuleList(
            [LeWMTransformerBlock(self.dim, args.lewm_heads, args.lewm_ffn_mult) for _ in range(args.lewm_encoder_layers)]
        )
        self.encoder_norm = nn.RMSNorm(self.dim)
        self.obs_target_proj = xavier_linear(nn.Linear(self.dim, self.dim))
        self.reward_outcome_proj = xavier_linear(nn.Linear(self.reward_num_bins, self.dim))
        self.continuation_outcome_proj = xavier_linear(nn.Linear(1, self.dim))
        self.reward_outcome_unproj = xavier_linear(nn.Linear(self.dim, self.reward_num_bins))
        self.continuation_outcome_unproj = xavier_linear(nn.Linear(self.dim, 1))

        self.action_in = xavier_linear(nn.Linear(1, self.dim))
        self.action_dim_embed = nn.Parameter(torch.empty(act_dim, self.dim))
        nn.init.xavier_uniform_(self.action_dim_embed)
        self.action_cond = xavier_linear(nn.Linear(act_dim, self.dim))
        axes = ["space", "time"] * ((args.lewm_predictor_layers + 1) // 2)
        self.predictor_layers = nn.ModuleList(
            [
                LeWMAxialPredictorBlock(self.dim, args.lewm_heads, args.lewm_ffn_mult, axis)
                for axis in axes[: args.lewm_predictor_layers]
            ]
        )
        self.predictor_norm = nn.RMSNorm(self.dim)
        self.pred_next_proj = xavier_linear(nn.Linear(self.dim, self.dim))
        self.pred_mtp_projs = nn.ModuleList(
            [xavier_linear(nn.Linear(self.dim, self.dim)) for _ in range(max(0, self.mtp_len - 1))]
        )
    def neutral_reward_probs(self, batch, device, dtype):
        probs = torch.zeros(batch, self.reward_num_bins, device=device, dtype=dtype)
        probs[:, self.reward_num_bins // 2] = 1.0
        return probs

    def encode_summary(self, obs):
        batch = obs.shape[0]
        reward_probs = self.neutral_reward_probs(batch, obs.device, obs.dtype)
        continues = torch.ones(batch, device=obs.device, dtype=obs.dtype)
        return self.encode_summary_with_outcomes(obs, reward_probs, continues)

    def encode_summary_with_outcomes(self, obs, reward_probs, continuations):
        batch = obs.shape[0]
        obs_flat = obs.reshape(batch, -1)
        tokens = obs_flat.unsqueeze(-1) * self.obs_feature_weight + self.obs_feature_bias
        for layer in self.encoder_layers:
            tokens = layer(tokens, causal=False)
        obs_tokens = self.obs_target_proj(self.encoder_norm(tokens))
        reward_token = self.reward_outcome_proj(reward_probs.to(obs_tokens.dtype)).unsqueeze(1)
        continuation_token = self.continuation_outcome_proj(
            continuations.reshape(batch, 1).to(obs_tokens.dtype)
        ).unsqueeze(1)
        return torch.cat([obs_tokens, reward_token, continuation_token], dim=1)

    def decode_outcomes(self, summary, detach_summary=True):
        if detach_summary:
            summary = summary.detach()
        reward_token = summary[:, self.num_obs_tokens]
        continuation_token = summary[:, self.num_obs_tokens + 1]
        reward_logits = self.reward_outcome_unproj(reward_token)
        continuation_logits = self.continuation_outcome_unproj(continuation_token).squeeze(-1)
        return reward_logits, continuation_logits

    def _predictor_trunk(self, latent_history, action_history):
        batch, context_len, num_tokens, width = latent_history.shape
        if context_len > self.context:
            latent_history = latent_history[:, -self.context :]
            action_history = action_history[:, -self.context :]
            context_len = self.context
        action_tokens = self.action_in(action_history.unsqueeze(-1))
        action_tokens = action_tokens + self.action_dim_embed.view(1, 1, self.act_dim, width)
        tokens = torch.cat([action_tokens, latent_history], dim=2)
        action_features = self.action_cond(action_history)
        for layer in self.predictor_layers:
            tokens = layer(tokens, action_features)
        tokens = self.predictor_norm(tokens)
        return tokens[:, :, self.act_dim : self.act_dim + num_tokens]

    def belief_from_history(self, latent_history, action_history):
        return self._predictor_trunk(latent_history, action_history)[:, -1]

    def predict_mtp_from_history(self, latent_history, action_history):
        features = self._predictor_trunk(latent_history, action_history)
        preds = [self.pred_next_proj(features)]
        preds.extend(proj(features) for proj in self.pred_mtp_projs)
        return torch.stack(preds, dim=2)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.world_model = LeWMBackbone(obs_dim, act_dim, args)
        self.detach_world_model_from_agent = args.detach_world_model_from_agent
        self.agent_input_dim = self.world_model.num_latent_tokens * self.world_model.dim
        self.agent_input_norm = nn.RMSNorm(self.agent_input_dim)
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(self.agent_input_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(self.agent_input_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(self.agent_input_dim, H, args.k_blocks, args.n_experts)
        # Categorical critic with a PEAKED init: small weight + Gaussian-logit
        # bias so the initial value distribution is sharp at 0 (not uniform),
        # preventing the distributional-bootstrap blowup that sank v9.
        self.critic_head = layer_init(nn.Linear(H, args.num_bins), std=0.1)
        with torch.no_grad():
            z = torch.linspace(args.v_min, args.v_max, args.num_bins)
            self.critic_head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)
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

    def agent_input_from_latent(self, latent):
        if self.detach_world_model_from_agent:
            latent = latent.detach()
        return latent.reshape(latent.shape[0], -1)

    def agent_input_from_history(self, latent_history, action_history):
        latent = self.world_model.belief_from_history(latent_history, action_history)
        return self.agent_input_from_latent(latent)

    def agent_input_from_obs(self, x):
        return self.agent_input_from_latent(self.world_model.encode_summary(x))

    def _trunks_from_agent_input(self, agent_input):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        agent_input = self.agent_input_norm(agent_input)
        if self.share_backbone:
            feat = self.trunk(agent_input)
            return feat, feat
        return self.actor_trunk(agent_input), self.critic_trunk(agent_input)

    def _trunks(self, x):
        return self._trunks_from_agent_input(self.agent_input_from_obs(x))

    def get_value(self, x):
        # Returns value LOGITS (B, num_bins); caller converts via support.
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

    def get_value_from_agent_input(self, agent_input):
        _, critic_feat = self._trunks_from_agent_input(agent_input)
        return self.critic_head(critic_feat)

    def get_action_and_value(self, x, z=None):
        return self.get_action_and_value_from_agent_input(self.agent_input_from_obs(x), z)

    def get_action_and_value_from_agent_input(self, agent_input, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks_from_agent_input(agent_input)
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
        return action, z, log_prob, entropy, value_logits

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
        return list(self.agent_input_norm.parameters()) + list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(self.agent_input_norm.parameters()) + list(trunk.parameters()) + list(self.critic_head.parameters())


def categorical_project(probs, atoms, support, v_min, v_max, bin_width):
    """C51 projection: distribute `probs` (B, n) sitting at positions `atoms`
    (B, n) onto the fixed `support` (n,) via linear interpolation between
    neighbouring bins. Mass-preserving (atoms are pre-clamped to [v_min, v_max]).
    """
    n = support.shape[0]
    tz = atoms.clamp(v_min, v_max)
    b = (tz - v_min) / bin_width                       # (B, n) fractional bin pos
    lo = b.floor()
    hi = b.ceil()
    lo_idx = lo.clamp(0, n - 1).long()
    hi_idx = hi.clamp(0, n - 1).long()
    m = torch.zeros_like(probs)
    m.scatter_add_(1, lo_idx, probs * (hi - b))        # mass to lower bin
    m.scatter_add_(1, hi_idx, probs * (b - lo))        # mass to upper bin
    # When b is integer, lo == hi and both weights are 0; (1 - (hi - lo)) == 1
    # routes the full mass to that bin. Otherwise hi - lo == 1 → adds 0.
    m.scatter_add_(1, lo_idx, probs * (1.0 - (hi - lo)))
    return m


def distributional_lambda_returns(
    rewards, dones, next_done, value_probs, bootstrap_probs, support, v_min, v_max, bin_width, gamma, gae_lambda
):
    """Backward recursion for the distributional λ-return G^λ (probs per step).

        G^λ_t =_D r_t + γ·nonterm·[ (1-λ)·Z(s_{t+1}) + λ·G^λ_{t+1} ]

    Mean-matches the scalar GAE λ-return. Shapes: rewards/dones (T, B);
    value_probs (T, B, n); bootstrap_probs (B, n) = Z(s_T). Returns (T, B, n).
    Entropy/soft-value terms are NOT injected here — the critic regresses to the raw
    reward return; max-ent enters the policy advantage separately (see --auto-entropy).
    """
    T = rewards.shape[0]
    target = torch.zeros_like(value_probs)
    g_next = bootstrap_probs                            # G^λ_{T} ≡ bootstrap
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done               # (B,)
            z_next = bootstrap_probs                    # Z(s_T)
        else:
            nonterminal = 1.0 - dones[t + 1]
            z_next = value_probs[t + 1]                 # Z(s_{t+1})
        mix = (1.0 - gae_lambda) * z_next + gae_lambda * g_next          # (B, n)
        gn = (gamma * nonterminal).unsqueeze(-1)        # (B, 1)
        atoms = rewards[t].unsqueeze(-1) + gn * support  # (B, n) transformed atoms
        g_next = categorical_project(mix, atoms, support, v_min, v_max, bin_width)
        target[t] = g_next
    return target


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
    assert args.lewm_dyn_horizon <= args.lewm_context, "v73 MTP expects dyn_horizon <= lewm_context"
    assert args.lewm_mtp_len <= args.lewm_dyn_horizon, "v73 MTP expects mtp_len <= dyn_horizon"
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
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

    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        0.5,  # sigma_ratio unused (categorical Bellman target, no Gaussian projection)
        device,
        use_symlog=args.value_symlog,
    )
    support = hl_support.support                       # (num_bins,) linear support
    bin_width = hl_support.bin_width
    reward_support = HLGaussSupport(
        args.lewm_reward_num_bins,
        args.lewm_reward_v_min,
        args.lewm_reward_v_max,
        0.5,
        device,
        use_symlog=False,
    )
    sigreg = SIGReg(knots=args.lewm_sigreg_knots, num_proj=args.lewm_sigreg_num_proj).to(device)

    def neutral_reward_probs(batch):
        return reward_support.project(torch.zeros(batch, device=device))

    def masked_token_sigreg(token_latents, token_valids):
        # token_latents: (B, H, S, D), token_valids: (B, H)
        valid_mask = token_valids.bool().reshape(-1)
        valid_count = int(valid_mask.sum().item())
        if valid_count < args.lewm_sigreg_min_valid:
            return token_latents.sum() * 0.0
        flat = token_latents.reshape(-1, token_latents.shape[-2], token_latents.shape[-1])[valid_mask]
        valid_tokens = flat.transpose(0, 1).contiguous()
        return sigreg(valid_tokens)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    agent_inputs = torch.zeros((args.num_steps, args.num_envs, agent.agent_input_dim)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    prev_reward_probs = torch.zeros((args.num_steps, args.num_envs, args.lewm_reward_num_bins), device=device)
    prev_outcome_continues = torch.ones((args.num_steps, args.num_envs), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.ones((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    rollout_latent_history = []
    rollout_action_history = []
    neutral_action = torch.zeros(args.num_envs, int(np.prod(envs.single_action_space.shape)), device=device)
    current_prev_reward_probs = neutral_reward_probs(args.num_envs)
    neutral_env_reward_probs = current_prev_reward_probs
    current_prev_continues = torch.ones(args.num_envs, device=device)

    def build_belief_agent_input(obs_tensor, reward_probs, continuations):
        current_latent = agent.world_model.encode_summary_with_outcomes(obs_tensor, reward_probs, continuations)
        if agent.detach_world_model_from_agent:
            current_latent = current_latent.detach()
        context_len = min(args.lewm_context, len(rollout_latent_history) + 1)
        n_past = context_len - 1
        past_latents = rollout_latent_history[-n_past:] if n_past > 0 else []
        past_actions = rollout_action_history[-n_past:] if n_past > 0 else []
        belief_latents = torch.stack(past_latents + [current_latent], dim=1)
        belief_actions = torch.stack(past_actions + [neutral_action], dim=1)
        return agent.agent_input_from_history(belief_latents, belief_actions), current_latent

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_actor_active = global_step >= args.wm_warmup_steps
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            prev_reward_probs[step] = current_prev_reward_probs
            prev_outcome_continues[step] = current_prev_continues

            with torch.no_grad():
                agent_input, current_latent = build_belief_agent_input(
                    next_obs,
                    current_prev_reward_probs,
                    current_prev_continues,
                )
                if not rollout_actor_active and args.actor_dist == "beta":
                    action = ((agent.action_low + agent.action_high) * 0.5).expand(args.num_envs, -1)
                    z = torch.full_like(action, 0.5)
                    logprob = torch.zeros(args.num_envs, device=device)
                    value_logits = agent.get_value_from_agent_input(agent_input)
                else:
                    action, z, logprob, ent, value_logits = agent.get_action_and_value_from_agent_input(agent_input)
                p = torch.softmax(value_logits, dim=-1)
                agent_inputs[step] = agent_input
                value_probs[step] = p
                values[step] = (p * support).sum(dim=-1)
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            rollout_latent_history.append(current_latent.detach())
            rollout_action_history.append(action.detach())
            if len(rollout_latent_history) > args.lewm_context - 1:
                rollout_latent_history = rollout_latent_history[-(args.lewm_context - 1) :]
                rollout_action_history = rollout_action_history[-(args.lewm_context - 1) :]

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_termination = terminations
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
                transition_next_obs_t = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            else:
                transition_next_obs_t = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            reward_tensor = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            termination_tensor = torch.as_tensor(transition_termination, device=device, dtype=torch.float32)
            boundary_tensor_f = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)
            rewards[step] = reward_tensor
            transition_terminations[step] = termination_tensor
            transition_boundaries[step] = boundary_tensor_f
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)
            next_obses[step] = transition_next_obs_t
            current_prev_reward_probs = reward_support.project(reward_tensor)
            current_prev_continues = 1.0 - termination_tensor
            boundary_tensor = boundary_tensor_f.bool()
            current_prev_reward_probs = torch.where(
                boundary_tensor[:, None],
                neutral_env_reward_probs,
                current_prev_reward_probs,
            )
            current_prev_continues = torch.where(
                boundary_tensor,
                torch.ones_like(current_prev_continues),
                current_prev_continues,
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_agent_input, _ = build_belief_agent_input(
                next_obs,
                current_prev_reward_probs,
                current_prev_continues,
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
                # Sample a' ~ π(·|s_T) for the bootstrap entropy (SAC's single-sample).
                _, _, boot_logprob, _, boot_logits = agent.get_action_and_value_from_agent_input(next_agent_input)
                bootstrap_probs = torch.softmax(boot_logits, dim=-1)            # (B, n) = Z(s_T)
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                bootstrap_probs = torch.softmax(agent.get_value_from_agent_input(next_agent_input), dim=-1)   # (B, n) = Z(s_T)
                next_value_bonus = None
            next_value = (bootstrap_probs * support).sum(dim=-1).reshape(1, -1)
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
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
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). Unbiased (state-dependent baseline); its large
            # magnitude is harmless — rankgauss at the optim step rank-normalizes it while
            # PRESERVING the entropy-induced reordering (the actual exploration signal).
            if auto_alpha:
                policy_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * (nextvalues + next_value_bonus[t]) * nextnonterminal - values[t]
                    policy_adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            else:
                policy_adv = advantages
            # Critic target: RAW reward λ-return (entropy-free => no support overflow).
            target_probs = distributional_lambda_returns(
                rewards, dones, next_done, value_probs, bootstrap_probs,
                support, args.v_min, args.v_max, bin_width, args.gamma, args.gae_lambda,
            )
            # Per-state return std sigma(s_t) from the OLD rollout Z(s_t). Note
            # E[Z(s_t)] == values[t], so G_t - E[Z_t] == GAE advantage exactly.
            sigma = (value_probs * (support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            cdf_frac = ((returns.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

        wm_losses = []
        wm_latent_losses = []
        wm_reward_losses = []
        wm_termination_losses = []
        wm_sigreg_losses = []
        wm_grad_norms = []
        if args.lewm_loss_coef > 0.0:
            wm_b_inds = np.arange(args.batch_size)
            horizon = args.lewm_dyn_horizon
            mtp_len = args.lewm_mtp_len
            max_start = args.num_steps - 1
            obs_shape = envs.single_observation_space.shape
            wm_minibatch_size = min(args.minibatch_size, args.lewm_minibatch_size)
            for _ in range(args.lewm_update_epochs):
                np.random.shuffle(wm_b_inds)
                for start in range(0, args.batch_size, wm_minibatch_size):
                    end = start + wm_minibatch_size
                    mb_inds_np = wm_b_inds[start:end]
                    mb_step_inds = torch.as_tensor(mb_inds_np // args.num_envs, device=device, dtype=torch.long)
                    mb_env_inds = torch.as_tensor(mb_inds_np % args.num_envs, device=device, dtype=torch.long)
                    mb_size = mb_step_inds.numel()
                    hist_offsets = torch.arange(horizon, device=device)
                    hist_step_inds = mb_step_inds[:, None] + hist_offsets[None, :]
                    hist_in_rollout = hist_step_inds < args.num_steps
                    safe_hist_step_inds = hist_step_inds.clamp(max=max_start)
                    env_inds = mb_env_inds[:, None].expand_as(safe_hist_step_inds)

                    future_actions = actions[safe_hist_step_inds, env_inds]
                    future_rewards = rewards[safe_hist_step_inds, env_inds]
                    future_terminations = transition_terminations[safe_hist_step_inds, env_inds]
                    future_boundaries = transition_boundaries[safe_hist_step_inds, env_inds]
                    future_valids = transition_valids[safe_hist_step_inds, env_inds]
                    future_next_obs = next_obses[safe_hist_step_inds, env_inds]

                    initial_summary = agent.world_model.encode_summary_with_outcomes(
                        obs[mb_step_inds, mb_env_inds],
                        prev_reward_probs[mb_step_inds, mb_env_inds],
                        prev_outcome_continues[mb_step_inds, mb_env_inds],
                    )
                    reward_target_probs_flat = reward_support.project(future_rewards.reshape(-1))
                    future_continues_flat = (1.0 - future_terminations).reshape(-1)
                    target_summaries = agent.world_model.encode_summary_with_outcomes(
                        future_next_obs.reshape((-1,) + obs_shape),
                        reward_target_probs_flat,
                        future_continues_flat,
                    ).reshape(
                        mb_size,
                        horizon,
                        agent.world_model.num_latent_tokens,
                        agent.world_model.dim,
                    )
                    teacher_history = torch.cat([initial_summary.unsqueeze(1), target_summaries[:, :-1]], dim=1)
                    pred_mtp = agent.world_model.predict_mtp_from_history(teacher_history, future_actions)

                    prev_continues = torch.cat(
                        [
                            torch.ones(mb_size, 1, device=device),
                            1.0 - future_boundaries[:, :-1],
                        ],
                        dim=1,
                    )
                    step_weight = torch.cumprod(prev_continues, dim=1) * hist_in_rollout.float()
                    latent_weight = step_weight * future_valids
                    reward_target_probs = reward_target_probs_flat.reshape(mb_size, horizon, -1)
                    pred_reward_logits_all, pred_continuation_logits_all = agent.world_model.decode_outcomes(
                        pred_mtp.reshape(
                            mb_size * horizon * mtp_len,
                            agent.world_model.num_latent_tokens,
                            agent.world_model.dim,
                        )
                    )
                    pred_reward_logits_all = pred_reward_logits_all.reshape(mb_size, horizon, mtp_len, -1)
                    pred_continuation_logits_all = pred_continuation_logits_all.reshape(mb_size, horizon, mtp_len)
                    target_reward_logits, target_continuation_logits = agent.world_model.decode_outcomes(
                        target_summaries.reshape(
                            mb_size * horizon,
                            agent.world_model.num_latent_tokens,
                            agent.world_model.dim,
                        ),
                        detach_summary=False,
                    )
                    target_reward_logits = target_reward_logits.reshape(mb_size, horizon, -1)
                    target_continuation_logits = target_continuation_logits.reshape(mb_size, horizon)

                    latent_losses = []
                    reward_losses = []
                    termination_losses = []
                    for mtp_idx in range(mtp_len):
                        valid_horizon = horizon - mtp_idx
                        if valid_horizon <= 0:
                            continue
                        offset_valid = latent_weight[:, mtp_idx:]
                        denom = offset_valid.sum().clamp_min(1.0)
                        pred = pred_mtp[:, :valid_horizon, mtp_idx]
                        target = target_summaries[:, mtp_idx:]
                        latent_loss = F.mse_loss(pred, target, reduction="none").mean(dim=(-1, -2))
                        latent_losses.append((latent_loss * offset_valid).sum() / denom)
                        reward_loss = -(
                            reward_target_probs[:, mtp_idx:].detach()
                            * torch.log_softmax(pred_reward_logits_all[:, :valid_horizon, mtp_idx], dim=-1)
                        ).sum(dim=-1)
                        reward_losses.append((reward_loss * offset_valid).sum() / denom)
                        termination_loss = F.binary_cross_entropy_with_logits(
                            pred_continuation_logits_all[:, :valid_horizon, mtp_idx],
                            1.0 - future_terminations[:, mtp_idx:],
                            reduction="none",
                        )
                        termination_losses.append((termination_loss * offset_valid).sum() / denom)
                    if not latent_losses:
                        continue
                    wm_latent_loss = torch.stack(latent_losses).mean()
                    wm_pred_reward_loss = torch.stack(reward_losses).mean()
                    wm_pred_termination_loss = torch.stack(termination_losses).mean()
                    target_denom = latent_weight.sum().clamp_min(1.0)
                    target_reward_loss = -(
                        reward_target_probs.detach() * torch.log_softmax(target_reward_logits, dim=-1)
                    ).sum(dim=-1)
                    target_reward_loss = (target_reward_loss * latent_weight).sum() / target_denom
                    target_termination_loss = F.binary_cross_entropy_with_logits(
                        target_continuation_logits,
                        1.0 - future_terminations,
                        reduction="none",
                    )
                    target_termination_loss = (target_termination_loss * latent_weight).sum() / target_denom
                    wm_reward_loss = 0.5 * (wm_pred_reward_loss + target_reward_loss)
                    wm_termination_loss = 0.5 * (wm_pred_termination_loss + target_termination_loss)
                    wm_sigreg_loss = masked_token_sigreg(
                        target_summaries[:, :, : agent.world_model.num_obs_tokens],
                        latent_weight,
                    )
                    wm_loss = (
                        args.lewm_loss_coef * wm_latent_loss
                        + args.lewm_reward_loss_coef * wm_reward_loss
                        + args.lewm_termination_loss_coef * wm_termination_loss
                        + args.lewm_sigreg_coef * wm_sigreg_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    wm_loss.backward()
                    wm_gn = nn.utils.clip_grad_norm_(agent.world_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    wm_losses.append(wm_loss.item())
                    wm_latent_losses.append(wm_latent_loss.item())
                    wm_reward_losses.append(wm_reward_loss.item())
                    wm_termination_losses.append(wm_termination_loss.item())
                    wm_sigreg_losses.append(wm_sigreg_loss.item())
                    wm_grad_norms.append(float(wm_gn))

        world_model_only = not rollout_actor_active
        if world_model_only:
            writer.add_scalar("charts/world_model_only", 1.0, global_step)
            if wm_losses:
                writer.add_scalar("lewm/loss", float(np.mean(wm_losses)), global_step)
                writer.add_scalar("lewm/latent_mse", float(np.mean(wm_latent_losses)), global_step)
                writer.add_scalar("lewm/reward_ce", float(np.mean(wm_reward_losses)), global_step)
                writer.add_scalar("lewm/continuation_bce", float(np.mean(wm_termination_losses)), global_step)
                writer.add_scalar("lewm/sigreg", float(np.mean(wm_sigreg_losses)), global_step)
                writer.add_scalar("lewm/grad_norm", float(np.mean(wm_grad_norms)), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            continue
        writer.add_scalar("charts/world_model_only", 0.0, global_step)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_agent_inputs = agent_inputs.reshape(-1, agent.agent_input_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.num_bins)
        b_u = u.reshape(-1)
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
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits = agent.get_action_and_value_from_agent_input(
                    b_agent_inputs[mb_inds], b_latent_zs[mb_inds]
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

                # Asymmetric "clip-higher" when clip_coef_high is set: looser upper bound
                # gives positive-advantage actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Distributional value loss: cross-entropy to the (fixed)
                # distributional λ-return target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

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
                    (pg_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
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
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
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
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        if wm_losses:
            writer.add_scalar("lewm/loss", float(np.mean(wm_losses)), global_step)
            writer.add_scalar("lewm/latent_mse", float(np.mean(wm_latent_losses)), global_step)
            writer.add_scalar("lewm/reward_ce", float(np.mean(wm_reward_losses)), global_step)
            writer.add_scalar("lewm/continuation_bce", float(np.mean(wm_termination_losses)), global_step)
            writer.add_scalar("lewm/sigreg", float(np.mean(wm_sigreg_losses)), global_step)
            writer.add_scalar("lewm/grad_norm", float(np.mean(wm_grad_norms)), global_step)
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

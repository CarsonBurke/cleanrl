# ============================================================================
# DG v14 -- the FAITHFUL delight-gate config: gate ON, renorm OFF, EMA-norm ON, eta=1, peak_ref.
#
# This is the configuration the analysis (vs arXiv:2603.14608v1 + its omega-vs-U figure) converged on:
#
#   * GATE ON (dg_use_gate=True):  w = sigmoid(chi/eta), chi = U * ell,  effective weight omega = w*U.
#   * RENORM OFF (dg_renorm=False): the paper does NOT renorm by mean(w). With renorm off, our per-sample
#       gradient coefficient is EXACTLY the figure's omega = U*sigmoid(U*ell). Renorm (mean(w)=1) was OUR
#       addition; it rescales omega vertically and fluctuates per-minibatch -> it is what pulled us OFF
#       the figure. Cost of removing it: mean(w)~0.5 halves the effective step -- absorbed by the adaptive
#       kl_beta / step-scale trust region, which re-equilibrates to kl_target regardless of gradient scale.
#   * EMA-NORM ON (ret_ema_norm=True, norm_adv=False): the SOLE advantage-scale control. This is what keeps
#       U ~ unit so chi = U*ell stays O(1) and the gate sits in its GRADED band (not the saturated sign-mask
#       you get from raw advantages -- see the killed rawrew/no-EMA run, chi_std~138). The paper achieves the
#       same "U~unit" via reward EMA-norm; eta is held at 1 in BOTH (eta is not the scale lever -- the
#       advantage normalization is). No per-minibatch whitening: EMA rescales globally without recentering,
#       preserving the per-sample advantage sign/magnitude structure the gate reads.
#   * peak_ref surprisal (ell = logp(mode) - logp(a) >= 0): the faithful continuous analog of the paper's
#       ell = -log pi(a). For a Beta, raw -log pi is a DENSITY (sign tracks concentration, not rarity; on
#       our peaked Beta it goes negative -> would INVERT the gate), so peak_ref re-references it to the mode:
#       ell=0 at the most-likely action (paper's high-pi "common"), grows in the tail (low-pi "rare"), >=0
#       always -- matching the paper's sign convention while killing the sharpness confound.
#
# HYPOTHESIS. With the gate genuinely on (renorm off so it isn't vertically rescaled) and U held ~unit by
# EMA-norm, the gate operates in the figure's regime: ~0.5 on the common bulk (PG-like), amplifying the rare
# tail of successes toward w->1 and suppressing the rare tail of failures toward w->0. If DG's reallocation
# helps on dense on-policy control, v14 >= the nogate PPO baseline (v12 nogate survivor). If it doesn't, the
# gentle-bulk + rare-tail-only action of a faithful gate is too weak to matter on HalfCheetah, and we learn
# the mechanism needs a stronger lever (surprisal scale / eta) than the paper's eta=1 default provides.
# Everything else is v13's stack: raw reward -> Dreamer3 511-bucket symexp critic -> EMA-std advantage norm
# -> analytic-KL soft penalty + step-scale trust region. adv_mode stays "gae" (== v12).
# ----------------------------------------------------------------------------
# DG v13 -- v12 + a "reward - baseline" advantage MODE (for the DG delight gate).
#
# Adds adv_mode="reward_minus_baseline": advantage = G_t - stop_gradient(V(s_t)), the discounted
# MC reward-to-go minus the detached value baseline (no GAE smoothing; == GAE lambda=1). Intended to
# pair with the DG delight gate (dg_use_gate=True): the gate w=sigmoid(chi/eta), chi=advantage*surprisal,
# wants a clean per-sample advantage sign/scale, and the simple reward-minus-baseline signal is the
# DG-paper-style utility. adv_mode="gae" (default) reproduces v12 exactly. Everything else is v12:
# step-scale KL, raw rewards, Dreamer3 critic, optional EMA advantage norm.
# ----------------------------------------------------------------------------
# DG v12 -- v11 (full dreamer4 scale stack) + v9's STEP-SCALE KL TRUST REGION.
#
# v11 controlled the KL with the soft penalty ALONE (v7's adaptive kl_beta), which bounds the MEAN
# KL but not the per-update MAX -> periodic spikes. v12 swaps in v9's KL solution: on top of the
# soft penalty, each minibatch's policy-gradient step is SCALED by clip((kl_cap-KL)/(kl_cap-kl_target),
# 0, 1) -- full step in the normal range (KL<=kl_target), ramping to 0 at the ceiling kl_cap=
# kl_target*kl_cap_ratio. Smooth throttle, every minibatch used, soft penalty stays full-strength and
# pulls back at the boundary (self-correcting). (--no-kl-step-scale falls back to v8's hard stop.)
# Everything else is v11: raw rewards, Dreamer3 511-bucket symexp critic, EMA advantage norm,
# analytic-KL soft penalty. So the full stack is: raw reward -> symexp-bucket critic -> EMA-std
# advantage norm -> soft KL penalty + step-scale trust region.
# ----------------------------------------------------------------------------
# DG v11 -- v10 + the FULL dreamer4 reward/return stack: RAW REWARDS + a DREAMER3-BUCKET critic
#           REPLACE the env reward-normalization wrapper. (Keeps v10's EMA advantage norm.)
#
# WHY. v10 kept gym's NormalizeReward wrapper (divide reward by a running std of the discounted
# return) AND clipped reward to [-10,10]. That is a SECOND, redundant scale controller fighting the
# critic's own scale handling, and the [-10,10] reward clip silently DISCARDS signal whenever a raw
# step reward exceeds 10 (HalfCheetah forward-velocity reward routinely does). dreamer4 does neither:
# it feeds RAW rewards to a symexp-twohot value head whose buckets span the whole return range, so
# the critic -- not a reward wrapper -- absorbs the scale, losslessly.
#
# WHAT. (1) Drop NormalizeReward + the reward clip (reward_norm=False) -> the critic sees raw GAE
# returns. (2) Swap the value support from the uniform-in-symlog HLGaussSupport (101 bins, decode
# symexp(E[z])) to v168's Dreamer3BucketHLGaussSupport (511 symexp-spaced raw-center buckets with an
# exact zero bucket, sigma 0.75, decode the Jensen-correct E[symexp(z)] = E[raw center]). Buckets
# span symlog-coords +-9.9035 (~+-20000 raw), so undiscounted raw returns project without saturating
# and the finer 511-grid keeps resolution across the now-wider used range. The Agent's zero-prior
# critic bias -0.5*(coord/tau)^2 still peaks the exact-zero bucket. NO MTP / world model exists in
# this base, so "critic like v168" = exactly this support, nothing imagined.
#
# This REPLACES reward normalization, NOT v10's EMA advantage norm -- that stays ON (ret_ema_norm).
# So the scale stack is now fully dreamer4: raw reward -> wide symexp-bucket critic for the value
# target -> EMA-std advantage norm for the policy step -> analytic-KL trust region for the move.
# HYPOTHESIS. Removing the redundant/clipping reward wrapper and giving the critic a faithful
# Dreamer3 bucket support yields cleaner value targets (no clip-induced bias, Jensen-correct decode)
# and at least matches v10 return while improving explained_variance. (--reward-norm re-enables the
# wrapper; --no-critic-d3bucket falls back to the v10 symlog HLGauss critic.)
# ----------------------------------------------------------------------------
# DG v10 -- v7's soft-penalty KL-trust + DREAMER-V3 RETURN NORMALIZATION (a 3rd spike tool).
#
# CONTEXT. v7's adaptive-beta SOFT penalty controls the MEAN KL but cannot bound the per-update
# MAX -> "almost periodic" KL spikes. v8 fixed this with a HARD per-minibatch cap (stop the pass),
# v9 with a SMOOTH step-scale. Both attack the SYMPTOM (the over-large step). This variant attacks
# a likely CAUSE: the spikes ride on the high-variance, heavy-tailed RAW GAE advantage -- when a
# rollout happens to contain a few huge-|U| samples, the score gradient -(U * dlogpi) momentarily
# explodes before beta (which adapts only once per iteration) can react.
#
# WHAT (DreamerV3 / dreamer4's return-norm, arXiv:2301.04104). Track an EMA of the mean & variance
# of the GAE RETURNS, computed on the (5%,95%) QUANTILE-CLAMPED returns (outlier-robust), and
# normalize the advantage by that running scale. dreamer4 writes it as
#     advantage = (returns - ema_mean)/ema_std - (old_values - ema_mean)/ema_std .
# Since returns = advantages + values, the ema_mean CANCELS and (returns - values) == advantages
# EXACTLY, so this reduces to:  U_normed = U / EMA_std(clamp_{5,95}(returns)) .
# The critic is UNCHANGED: it still regresses RAW returns through the symlog/HL-Gauss head (its own
# scale handling -- exactly dreamer4, which keeps a raw-return symexp-twohot value head). Only the
# ADVANTAGE scale changes: a slowly-moving, outlier-robust denominator instead of the raw,
# per-rollout-volatile magnitude. A stable |U| -> a stable score-gradient scale -> the soft penalty
# faces a far smaller spike to chase.
#
# HYPOTHESIS. Stabilizing the advantage scale reduces the KL-spike magnitude/frequency the v7 soft
# penalty alone could not bound -- a cause-side complement to v8's cap and v9's step-scale -- and
# (vs the protected v7 nogate run, same base + this only change) should at least match return while
# cutting analytic_kl_max. NOTE: a pure GLOBAL rescale of U is partly absorbed at steady state by
# the adaptive beta / Adam's scale-robustness; the first-order win we expect is in the TRANSIENT
# per-rollout spikes (volatile |U|), not the steady-state mean KL. (--no-ret-ema-norm disables it.)
# ----------------------------------------------------------------------------
# DG v7 -- DG gate + an ANALYTIC-KL TRUST REGION (the principled fix for the clip).
#
# WHY (v6's failure mode). v6 added a PPO ratio clip to tame the high-capacity ThinkTrunk
# Beta. It does not bound the policy KL, for two structural reasons:
#   (1) The clip is a GRADIENT MASK, not a step constraint: it zeroes the gradient on samples
#       already outside [1-eps, 1+eps], but the step driven by the OTHER samples still moves
#       them -- KL accumulates across the 32 minibatch steps with no global brake.
#   (2) BETA-SPECIFIC: KL between two Betas is dominated by the CONCENTRATION change (the
#       log B(a,b) / digamma terms). The ratio only measures density at SAMPLED actions, which
#       cluster near the mode. The policy can SHARPEN (collapse variance) -- blowing up KL via
#       the normalizer -- while near-mode sampled ratios stay inside the clip band, so the clip
#       NEVER FIRES on the direction that is actually diverging. Empirically: clipfrac~0.09 yet
#       approx_kl~0.013 and rising; the unclipped arm thrashes (return ~9 at 1.9M).
# The clip controls the wrong quantity (per-sample density ratios at sampled points) as a proxy
# for the thing we care about (the distributional KL).
#
# WHAT (the fix). Replace the leaky proxy with the Beta's CLOSED-FORM KL as an adaptive penalty
# -- the PPO-PENALTY objective (Schulman 2017), the principled sibling of clipping:
#     actor_loss = -(w * U * log pi).mean()  +  beta * E_s[ KL( pi_old(.|s) || pi_new(.|s) ) ]
# KL is exact via torch.distributions.kl_divergence(Beta_old, Beta_new) (digamma/lgamma). We
# STORE alpha_old, beta_old at rollout so the old distribution is recovered exactly. beta is
# SELF-TUNING: x2 when mean KL overshoots kl_target, /2 when it undershoots (clamped). This is a
# proximal / mirror-descent step -- the cheap principled cousin of the natural gradient, whose
# whole purpose is to make every step a BOUNDED-KL step regardless of position in param space.
# It sees the FULL distributional move (incl. the concentration direction the clip misses), and
# retires the duct-tape stack (target_kl early-stop, the actor_lr crutch, the ratio clip) for one
# control measured in KL units. The DG gate is UNCHANGED -- it still reweights the score term;
# the KL term only bounds the step. (--use-ratio-clip re-enables the v6 clip for ablation.)
#
# HYPOTHESIS. A trust region measured in the RIGHT units (analytic KL, which captures the Beta's
# concentration collapse) stabilizes the high-capacity actor where the ratio clip leaks, letting
# the single-pass DG-gated score climb without the late degradation / thrash KL blowups cause.
#
# ----------------------------------------------------------------------------
# DELIGHTFUL POLICY GRADIENT (DG) -- Beta-policy surprisal (the gate).
# Paper: arXiv:2603.14608v1. ell_t = surprisal (>=0), chi_t = U_t*ell_t, w_t = sigmoid(chi/eta)
# DETACHED. "Amplify rare successes, suppress rare failures." For a Beta the literal -log pi
# inverts as the policy concentrates (log-normalizer -> -inf), so we use the peak-referenced
# surprisal ell = log pi(mode) - log pi(a) >= 0 (cancels the normalizer; the Beta analog of the
# Gaussian Mahalanobis term), or the moment-matched "mahalanobis" form 0.5||(z-mu)/sigma||^2.
# ============================================================================
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport, Dreamer3BucketHLGaussSupport

EPS = 1e-6  # clamp for Beta samples / mode to keep log_prob finite


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

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    actor_lr: float = None           # actor Adam LR; None -> learning_rate. With the ratio-clip
    #                                  trust region now bounding the step DIRECTION, the LR crutch
    #                                  is no longer load-bearing, but it stays available to bracket.
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    adv_mode: str = "gae"            # "gae" (default, == v12) | "reward_minus_baseline": advantage =
    #                                  G_t (discounted MC reward-to-go) - stop_gradient(V(s_t)). No GAE
    #                                  smoothing; the detached value is a pure baseline. (== GAE lambda=1.)
    num_minibatches: int = 32
    actor_epochs: int = 10           # MUST be >1: the ratio clip is a no-op on a single on-policy
    #                                  pass (ratio==1). 10 = standard PPO epoch reuse.
    critic_epochs: int = 10          # critic is pure regression -> refit more for good advantages
    norm_adv: bool = False           # faithful line: gate & surrogate see the RAW advantage U
    clip_coef: float = 0.2           # PPO ratio clip: LOWER bound 1-clip_coef (also upper if no high)
    clip_coef_high: float = None     # DAPO "clip-higher": looser UPPER bound 1+clip_coef_high. None ->
    #                                  symmetric. Loosening only the upper clip lets rare-but-good
    #                                  (low-density) actions raise their density -> preserves Beta spread
    #                                  -> directly counters the over-concentration collapse. Base uses 0.28.
    use_ratio_clip: bool = False     # v7: OFF. The KL-trust penalty replaces the leaky ratio clip.
    #                                  True re-enables v6's PPO clipped surrogate (ablation only).
    # --- Analytic-KL trust region (the principled KL control) ---
    kl_trust: bool = True            # add beta * E[ KL(pi_old||pi_new) ] (exact Beta KL) to actor loss
    kl_target: float = 0.02          # per-update KL budget the adaptive beta regulates toward
    kl_beta_init: float = 3.0        # initial penalty coefficient beta
    kl_beta_min: float = 0.1         # adaptive-beta clamp (lower)
    kl_beta_max: float = 300.0       # adaptive-beta clamp (upper)
    # v12: v9's step-scale KL trust region on top of the soft penalty (bounds the per-update MAX KL).
    kl_cap_ratio: float = 3.0        # KL ceiling = kl_target*kl_cap_ratio. step scale hits 0 here. 0 disables.
    kl_step_scale: bool = True       # scale each minibatch's pg step by (kl_cap-KL)/(kl_cap-kl_target) in [0,1]
    #                                  (smooth throttle, uses all minibatches). False -> v8 hard per-mb stop.
    ent_coef: float = 0.0            # paper DG uses no entropy bonus
    max_grad_norm: float = 0.5       # standard PPO clip
    target_kl: float = None          # optional KL early-stop across actor epochs (off by default)
    # --- DreamerV3 return normalization (advantage scale) ---
    ret_ema_norm: bool = True        # normalize advantage by EMA std of quantile-clamped returns
    ret_ema_decay: float = 0.998     # EMA decay for the running return mean/var (dreamer4 default)
    ret_quantile_lo: float = 0.05    # lower quantile for the outlier-robust return-stats clamp
    ret_quantile_hi: float = 0.95    # upper quantile

    # iterthink_v24_beta EXACT ThinkTrunk architecture (separate actor/critic trunks)
    hidden: int = 64                 # trunk hidden width H
    k_blocks: int = 3                # number of ThinkBlocks (DenseNet-style depth)
    n_experts: int = 16              # soft-MoE experts per ThinkBlock
    critic_init_tau: float = 0.5     # init value dist ~ N(0, tau^2): peaked-at-0 critic-head bias

    # HL-Gauss distributional critic
    critic_d3bucket: bool = True     # v11: use v168's Dreamer3 symexp-bucket support (False -> v10 symlog HLGauss)
    critic_num_bins: int = 511       # categorical support size (odd -> exact zero bucket for Dreamer3)
    critic_v_min: float = -9.90353755128617  # symlog-COORD min; symexp(9.9035) ~ +-20000 raw (v168 range)
    critic_v_max: float = 9.90353755128617   # symlog-COORD max
    critic_sigma_ratio: float = 0.75 # HL-Gauss label sigma as a fraction of bin width (paper sweet spot)
    critic_symlog: bool = True       # (only used by the HLGauss fallback; Dreamer3 bucket is intrinsically symlog)
    reward_norm: bool = False        # v11: drop gym NormalizeReward + the [-10,10] reward clip (raw rewards)

    # DG-specific
    dg_use_gate: bool = True         # v14: gate ON. Faithful DG -- amplify rare success / suppress rare
    #                                  failure. (--no-dg-use-gate => w==1, reproduces the nogate PPO baseline.)
    dg_surprisal: str = "peak_ref"   # "peak_ref" (ell=logp(mode)-logp(a)>=0) | "mahalanobis" | "raw"
    dg_eta: float = 1.0              # temperature eta in w = sigmoid(chi/eta). Paper holds eta=1 throughout.
    dg_clip: float = 10.0            # paper C: clip on the surprisal ell
    dg_renorm: bool = False          # v14: renorm OFF. Paper does NOT renorm -> w=sigmoid(chi) is exactly
    #                                  the figure's effective weight omega=U*sigmoid(U*ell). (--dg-renorm restores
    #                                  mean(w)=1 reallocation.) Mean(w)~0.5 halves the step; kl_beta absorbs it.

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, reward_norm=True):
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
        if reward_norm:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ============================================================================
# iterthink_v24_beta EXACT backbone: the "ThinkTrunk" -- a DenseNet-style stack
# of K ThinkBlocks. Each block: bounded-convex residual gate mixing x_in and x0, a
# dense pre-act MLP branch (RMSNorm + ReLU^2), and a soft (full-softmax) MoE branch.
# Used here as SEPARATE actor/critic trunks (share_backbone=False).
# ============================================================================
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
        # init +4 -> g ~ 0.982 -> x_in ~ x at start.
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        # Dense branch. RMSNorm without learnable affine (no free per-channel gamma).
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


class Agent(nn.Module):
    """Beta-policy actor + HL-Gauss distributional critic on iterthink_v24_beta's EXACT
    ThinkTrunk architecture, as SEPARATE actor/critic trunks (share_backbone=False).

    a = 2z - 1 (z the native Beta sample); the constant log-2 Jacobian cancels in both the
    score grad and the peak-referenced surprisal, so we work in native z-space throughout.
    """

    def __init__(self, envs, num_bins, hidden=64, k_blocks=3, n_experts=16,
                 v_min=-10.0, v_max=10.0, critic_init_tau=0.5):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = hidden
        self.critic_trunk = ThinkTrunk(obs_dim, H, k_blocks, n_experts)
        self.actor_trunk = ThinkTrunk(obs_dim, H, k_blocks, n_experts)
        # dreamer4 unimodal Beta: alpha, beta = 1 + softplus(.) > 1 (interior mode). std=0.01.
        self.alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        # Distributional value head over the HL-Gauss support: small weight + Gaussian-logit
        # bias so the initial value distribution is PEAKED at 0 (iterthink's critic init).
        self.critic_head = layer_init(nn.Linear(H, num_bins), std=0.1)
        with torch.no_grad():
            zc = torch.linspace(v_min, v_max, num_bins)
            self.critic_head.bias.copy_(-0.5 * (zc / critic_init_tau) ** 2)

    def get_value(self, x):
        """Returns raw value LOGITS over the HL-Gauss support (caller decodes/projects)."""
        return self.critic_head(self.critic_trunk(x))

    def _dist(self, x):
        h = self.actor_trunk(x)
        alpha = 1.0 + F.softplus(self.alpha_head(h))
        beta = 1.0 + F.softplus(self.beta_head(h))
        return Beta(alpha, beta)

    def get_action_and_value(self, x, z=None):
        """Returns (z_native, action, logp, ell, entropy, value_logits, dist).

        `dist` is the Beta itself -- the caller stores its (alpha, beta) at rollout and uses the
        recomputed dist for the analytic-KL trust region in the actor update.
        """
        dist = self._dist(x)
        if z is None:
            z = dist.sample().clamp(EPS, 1.0 - EPS)
        logp = dist.log_prob(z).sum(1)
        if getattr(self, "dg_surprisal", "peak_ref") == "mahalanobis":
            mean = dist.mean
            std = dist.stddev
            ell = (0.5 * ((z - mean) / (std + 1e-6)) ** 2).sum(1)
        else:
            a, b = dist.concentration1, dist.concentration0
            mode = ((a - 1.0) / (a + b - 2.0).clamp_min(EPS)).clamp(EPS, 1.0 - EPS)
            logp_mode = dist.log_prob(mode).sum(1)
            ell = logp_mode - logp  # >= 0
        entropy = dist.entropy().sum(1)
        action = 2.0 * z - 1.0
        return z, action, logp, ell, entropy, self.get_value(x), dist


if __name__ == "__main__":
    args = tyro.cli(Args)
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, reward_norm=args.reward_norm) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Distributional value support. v11: v168's Dreamer3 symexp-bucket HL-Gauss (raw-center decode,
    # exact zero bucket). Both supports expose the same .to_scalar/.project/.support/.bin_width API,
    # so all call sites below are unchanged.
    if args.critic_d3bucket:
        hlg = Dreamer3BucketHLGaussSupport(
            num_bins=args.critic_num_bins,
            coord_min=args.critic_v_min,
            coord_max=args.critic_v_max,
            sigma_ratio=args.critic_sigma_ratio,
            device=device,
        )
    else:
        hlg = HLGaussSupport(
            num_bins=args.critic_num_bins,
            v_min=args.critic_v_min,
            v_max=args.critic_v_max,
            sigma_ratio=args.critic_sigma_ratio,
            device=device,
            use_symlog=args.critic_symlog,
        )

    agent = Agent(
        envs, args.critic_num_bins, hidden=args.hidden, k_blocks=args.k_blocks,
        n_experts=args.n_experts, v_min=args.critic_v_min, v_max=args.critic_v_max,
        critic_init_tau=args.critic_init_tau,
    ).to(device)
    agent.dg_surprisal = args.dg_surprisal
    actor_params = list(agent.actor_trunk.parameters()) + list(agent.alpha_head.parameters()) + list(agent.beta_head.parameters())
    critic_params = list(agent.critic_trunk.parameters()) + list(agent.critic_head.parameters())
    actor_base_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    actor_opt = optim.Adam(actor_params, lr=actor_base_lr, eps=1e-5)
    critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)
    kl_beta = args.kl_beta_init  # adaptive analytic-KL penalty coefficient (regulated toward kl_target)
    # DreamerV3 return-norm running stats (EMA of quantile-clamped GAE return mean/var).
    ema_ret_mean, ema_ret_var, ema_ret_std, ema_ret_inited = 0.0, 1.0, 1.0, False

    # Storage: store the NATIVE beta sample z (replayed to recompute logp at the same draw),
    # plus the OLD Beta params (alpha, beta) per dim so the rollout policy is recovered EXACTLY
    # for the analytic-KL trust region in the actor update.
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    alphas = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    betas = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_opt.param_groups[0]["lr"] = frac * actor_base_lr
            critic_opt.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                z, action, logprob, _, _, value_logits, dist = agent.get_action_and_value(next_obs)
                values[step] = hlg.to_scalar(value_logits).flatten()
            zs[step] = z
            alphas[step] = dist.concentration1
            betas[step] = dist.concentration0
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Advantage estimate (computed once from rollout values, all detached).
        with torch.no_grad():
            next_value = hlg.to_scalar(agent.get_value(next_obs)).reshape(1, -1)
            if args.adv_mode == "reward_minus_baseline":
                # advantage = G_t - sg(V(s_t)): discounted MC reward-to-go minus the detached value
                # baseline. No GAE smoothing (== GAE lambda=1). values are rollout (no_grad) -> the
                # baseline carries no gradient. Critic target is the same G_t (returns) below.
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_ret = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_ret = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_ret
                advantages = returns - values
            else:
                # GAE (default) -- U = raw advantage.
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

            # DreamerV3 return normalization (dreamer4): scale the advantage by a slow,
            # outlier-robust EMA of the return std. The ema_mean cancels in dreamer4's
            # (normed_returns - normed_values), so only the std rescales U. The critic target
            # (b_returns below) stays RAW -- symlog/HL-Gauss handles its own scale, as in dreamer4.
            if args.ret_ema_norm:
                flat_ret = returns.reshape(-1)
                qs = torch.tensor([args.ret_quantile_lo, args.ret_quantile_hi], device=device)
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                clamped = flat_ret.clamp(lo, hi)
                batch_mean = clamped.mean().item()
                batch_var = clamped.var(unbiased=False).item()
                if not ema_ret_inited:
                    # Cold-start to the first batch stats for immediate (not slowly-warmed) scaling,
                    # but floor the variance at 1.0 (dreamer4's init) so a near-constant first rollout
                    # can't collapse ema_std -> 0 and inflate the advantage. No-op once batch_var >= 1.
                    ema_ret_mean, ema_ret_var, ema_ret_inited = batch_mean, max(batch_var, 1.0), True
                else:
                    d = 1.0 - args.ret_ema_decay
                    ema_ret_mean += d * (batch_mean - ema_ret_mean)
                    ema_ret_var += d * (batch_var - ema_ret_var)
                ema_ret_std = max(ema_ret_var, 1e-10) ** 0.5
                advantages = advantages / ema_ret_std

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_alphas = alphas.reshape((-1,) + envs.single_action_space.shape)
        b_betas = betas.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)

        # ---- Critic update: many epochs of pure HL-Gauss regression (no off-policy bias) ----
        for epoch in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                value_logits = agent.get_value(b_obs[mb_inds])
                target_probs = hlg.project(b_returns[mb_inds])
                v_loss = -(target_probs * F.log_softmax(value_logits, dim=-1)).sum(-1).mean()
                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                critic_opt.step()

        # ---- Actor update: DG-gated score + ANALYTIC-KL trust region (the principled KL control) ----
        gate_means, surp_means, chi_stds, clipfracs, kl_terms = [], [], [], [], []
        scale_terms = []  # v12 (v9): per-minibatch policy-gradient step scale in [0,1]
        approx_kl = torch.zeros((), device=device)
        kl_cap = args.kl_target * args.kl_cap_ratio
        stop_actor, n_capped, n_steps = False, 0, 0
        for epoch in range(args.actor_epochs):
            if stop_actor:
                break
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                _, _, newlogprob, ell, entropy, _, new_dist = agent.get_action_and_value(b_obs[mb_inds], b_zs[mb_inds])

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Optional DG delight gate (detached). nogate => w==1.
                if args.dg_surprisal == "raw":
                    surprisal = (-newlogprob).clamp(-args.dg_clip, args.dg_clip)
                else:
                    surprisal = ell.clamp(0.0, args.dg_clip)
                chi = mb_adv * surprisal
                w = torch.sigmoid(chi / args.dg_eta).detach()
                if not args.dg_use_gate:
                    w = torch.ones_like(w)
                if args.dg_renorm:
                    w = w / (w.mean() + 1e-8)

                if args.use_ratio_clip:
                    # v6 ablation: PPO clipped surrogate (leaky proxy trust region), gated by w.
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    surr1 = -mb_adv * ratio
                    surr2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = (w * torch.max(surr1, surr2)).mean()
                else:
                    # DG gated score (paper Alg.2): minimize -(w * U * log pi). The KL term below is
                    # what bounds the step (no ratio/clip needed).
                    pg_loss = -(w * mb_adv * newlogprob).mean()

                # Analytic KL trust region: exact KL(pi_old || pi_new) from STORED old (alpha,beta),
                # averaged over the rollout states. Captures the full move incl. concentration --
                # the direction the ratio clip is blind to. kl_beta is adapted toward kl_target.
                if args.kl_trust:
                    old_dist = Beta(b_alphas[mb_inds], b_betas[mb_inds])
                    kl = kl_divergence(old_dist, new_dist).sum(1).mean()
                else:
                    kl = torch.zeros((), device=device)

                # v9's KL solution: KL ceiling enforcement. SMOOTH (kl_step_scale) -- scale this
                # minibatch's policy-gradient step down as the exact KL approaches the ceiling,
                # ramping from 1 at kl_target to 0 at kl_cap. No minibatch is discarded; the soft
                # penalty below stays full-strength and pulls back. v8 fallback: HARD stop.
                if args.kl_trust and args.kl_cap_ratio > 0:
                    if args.kl_step_scale:
                        denom = max(kl_cap - args.kl_target, 1e-8)
                        scale = float(np.clip((kl_cap - kl.item()) / denom, 0.0, 1.0))  # detached
                        pg_loss = pg_loss * scale
                        scale_terms.append(scale)
                        if scale < 1.0:
                            n_capped = 1  # iteration touched the throttle band
                    elif kl.item() > kl_cap:
                        kl_terms.append(kl.item())
                        n_capped = 1
                        stop_actor = True
                        break

                actor_loss = pg_loss + kl_beta * kl - args.ent_coef * entropy.mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                actor_opt.step()
                n_steps += 1

                with torch.no_grad():
                    gate_means.append(w.mean().item())
                    surp_means.append(surprisal.mean().item())
                    chi_stds.append(chi.std().item())
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    kl_terms.append(kl.item())
                    approx_kl = ((ratio - 1) - logratio).mean()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Adaptive penalty (Schulman 2017): regulate the realized analytic KL toward kl_target.
        mean_kl = float(np.mean(kl_terms)) if kl_terms else 0.0
        if args.kl_trust:
            if mean_kl > args.kl_target * 1.5:
                kl_beta = min(kl_beta * 2.0, args.kl_beta_max)
            elif mean_kl < args.kl_target / 1.5:
                kl_beta = max(kl_beta / 2.0, args.kl_beta_min)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", actor_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/analytic_kl", mean_kl, global_step)
        writer.add_scalar("losses/kl_beta", kl_beta, global_step)
        writer.add_scalar("losses/ret_ema_std", ema_ret_std, global_step)
        writer.add_scalar("losses/analytic_kl_max", float(np.max(kl_terms)) if kl_terms else 0.0, global_step)
        writer.add_scalar("losses/kl_cap_hit", float(n_capped), global_step)
        writer.add_scalar("losses/actor_steps", n_steps, global_step)
        writer.add_scalar("losses/kl_step_scale", float(np.mean(scale_terms)) if scale_terms else 1.0, global_step)
        writer.add_scalar("losses/kl_step_scale_min", float(np.min(scale_terms)) if scale_terms else 1.0, global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/dg_gate_mean", float(np.mean(gate_means)), global_step)
        writer.add_scalar("charts/dg_surprisal_mean", float(np.mean(surp_means)), global_step)
        writer.add_scalar("charts/dg_chi_std", float(np.mean(chi_stds)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()

# PPO + IterThink v24 Beta + d3bucket MTP + BetaPlast v10: v8 + WIDER (softer) graded negative.
# =====================================================================================
# v10 change over v8: std-normalize the advantage before the negative tanh and WIDEN it
# (signneg_neg_temp=0.5, signneg_neg_norm=True): neg_eff = tanh(0.5 * A/std). The soft counterpart
# to v9 -- brackets the sweet spot from the GENTLE side. Near-zero and moderate negatives are
# suppressed only proportionally (barely-bad actions are barely punished), yet clearly-bad actions
# (|A| >> std) still saturate toward -1, so this is NOT v7's uniform weakening -- it re-shapes,
# not de-strengths. If v10 > v8 > v9, softer/more-proportional suppression is better; the pair
# (v9 sharp / v10 wide) locates the optimum. Positives keep divide-by-std magnitude; gate [0.25,4].
#
# --- inherited v8 notes ---
# PPO + IterThink v24 Beta + d3bucket MTP + BetaPlast v8: v4 + SOFT (graded) negative.
# =====================================================================================
# v8 change over v4: signneg_neg_mode "sign" -> "softtanh". v4 suppressed every negative-adv
# action with a hard, uniform eff_adv=-1; v8 instead uses eff_adv=-tanh(|A|) in (-1,0), so
# worse actions are pushed down harder (graded), but still bounded at 1. Positives keep the same
# divide-by-std magnitude. This is the negative-side control for v4's central design choice:
# does HARD sign genuinely beat GRADED suppression once positives already carry magnitude? (It
# sits between v2's tanh-both and v4's sign-neg.) Plasticity gate = v4's relative [0.25,4].
#
# --- inherited v4 notes ---
# PPO + IterThink v24 Beta + d3bucket MTP + BetaPlast v4: SIGN-neg / PPO-pos hybrid.
# =====================================================================================
# v4 change over v2 (ablation): asymmetric advantage treatment via a NEW objective
# policy_objective="signneg_ppo". This LEAVES the PMPO log-prob/KL objective entirely and
# runs the standard PPO clipped-ratio surrogate on a MODIFIED advantage:
#   - NEGATIVE advantages -> replaced by their SIGN (-1): a uniform, magnitude-free
#     suppression (all bad actions pushed down equally).
#   - POSITIVE advantages -> "normal PPO treatment" (signneg_pos_mode="ppo"): the real
#     magnitude, std-normalized (divide-only so the raw sign is preserved), so genuinely
#     better actions are reinforced proportionally.
# The PPO clip is the sole trust region (no KL term). Rationale: suppression rarely needs
# to be graded (a bad action is bad), but reinforcement benefits from magnitude to chase
# the best actions -- so keep magnitude only where it helps. Plasticity gate = v2's relative
# [0.25,4]. (v5 is the same branch with signneg_pos_mode="sign": pure sign both sides, no advnorm.)
#
# --- inherited v2 notes ---
# PPO + IterThink v24 Beta + d3bucket MTP + PMPO + BetaPlast (v1's RELATIVE gate) v2.
# =====================================================================================
# v2 change over v1: use v1's PROVEN plasticity solution on the PMPO base. The expansive
# non-competitive gates (absolute / bounded_absolute, plast v2/v3) UNDERPERFORMED on the
# v162critic base (~5.9k-6.1k @ 2.4M vs the relative-gate v1's ~8.3k trajectory), so this
# variant reverts the gate to the COMPETITIVE (relative, zero-sum reallocation) design at
# the moderate clamp [0.25, 4] -- i.e. plast_v1's exact config -- grafted onto the d3bucket
# PMPO actor objective. plast_gate_mode="relative": gate_i = clamp(proxy_i/mean(proxy),
# 0.25, 4); the global effective LR is pinned (mean 1), only reallocated across neurons.
# Everything else (per-neuron signed-Beta gains, frozen antithetic noise, curvature-trained
# concentration, PMPO-consistent old/new KL) is unchanged from v1.
#
# --- inherited plasticity notes (gate MODE is overridden to relative above) ---
# PLASTICITY ADD-ON (grafted from betaplast_v2, gate design = plast_v3/bounded_absolute):
# Per-neuron signed-Beta multiplicative gains on the ACTOR's H-dim hidden feature (the
# ThinkTrunk output feeding the actor heads):
#     s_i = 1 + delta * r_i * (1 - u_i^(1/c_i)),  r_i ~ {-1,+1}, u_i ~ U(0,1)   (mean 1)
# with a learned per-neuron concentration c_i = 1 + (c_max-1)*sigmoid(rho_i). (u,r) are
# drawn ONCE per iteration per env (antithetic sign pairs), frozen through the rollout AND
# all update epochs -> coherent parameter-space exploration with exact ratio/KL consistency
# (the perturbed actor IS the rollout policy; the PMPO old/new Beta dists are BOTH built from
# the same per-env gain, so the reverse-KL stays consistent). Gain is applied ACTOR-ONLY
# (after the actor/critic split), so the value target is untouched.
# STEP GATE (v3 bounded_absolute, EXPANSIVE): each gained neuron's POST-Adam update to the
# producing layer (trunk.out_proj row + bias) is scaled by a plasticity gate. The gate is
# DECOUPLED per-neuron (absolute: proxy_i/proxy(c_init), non-competitive -- a plastic neuron
# keeps a big step without taxing neighbors) but the LAYER-MEAN gate is rescaled into
# [gate_mean_min, gate_mean_max] so the GLOBAL LR can breathe (up to 4x/1-4x) yet never
# collapse (the failure mode of the pure-absolute betaplast_v2_1 gate). Per-neuron clamp is
# expansive [1/16, 16]. Mechanism reasoning: local decoupling is the useful part of
# "non-competitive"; the unbounded global float is the harmful part -- so bound only the
# global drift. Reward normalization stays OFF (fixed symlog support needs raw returns).
# See ppo_continuous_action_iterthink_v24_beta_v162critic_mtp_plast_v3.py for the sibling
# variant of this exact gate on the v162critic base.
#
# --- inherited base notes ---
# PPO + IterThink v24 Beta + v162 critic + Dreamer3-bucket HL-Gauss MTP + DREAMER4 PMPO v1.
# =====================================================================================
# v1 = the d3bucket_mtp_mbpercnorm base with its PPO clipped-surrogate actor objective REPLACED by
# dreamer4's PMPO objective (faithfully ported from ppo_continuous_action_pmpo_d4_beta_relusq_v5):
#   - SIGN-BASED update: each taken action's per-dim Beta log-prob is weighted by adv.tanh().abs() in
#     [0,1) -- "essentially sign based" (magnitude is squashed away, only the sign + a soft saturation
#     near 0 survive). NO advantage normalization, NO advantage shaping, NO rank/percentile scaling: the
#     RAW policy GAE advantage feeds the tanh-sign directly (the base's shape_advantage / norm_adv /
#     ret_percnorm / pos_neg_alpha paths are all bypassed under PMPO).
#   - BALANCED pos/neg: positive- and negative-advantage log-probs are each averaged over their OWN count
#     (so a class imbalance cannot dominate), then combined as
#     -w*pos_loss + (1-w)*neg_loss with w=pmpo_pos_to_neg_weight (0.5 = symmetric).
#   - TRUST REGION via a closed-form reverse Beta-KL to the ROLLOUT policy: + pmpo_kl_coef * KL(old||new),
#     coef 0.3 (replaces PPO's ratio clip / advnorm as the step controller).
# Everything else is IDENTICAL to the base: DreamerV3 511-bucket symexp HL-Gauss distributional MTP critic,
# the iterthink ThinkTrunk, the critic MTP machinery, the unimodal Beta actor, reward GAE, decoupled grad
# clipping (max_grad_norm=0.5), optimizer structure, hyperparameters, logging. policy_objective="ppo"
# reproduces the base EXACTLY (this file is a strict superset / clean A/B sibling of the ppoadvnorm runs).
#
# HYPOTHESIS: a PURE sign-based update (adv.tanh().abs(), balanced pos/neg) governed ONLY by a KL-to-rollout
# trust region -- instead of advantage-magnitude + clip/advnorm -- controls the policy step without the
# magnitude-driven over-optimization that the percentile/rank scalers were trying to tame. If PMPO matches
# or beats the ppoadvnorm/retstd siblings, the advantage MAGNITUDE was net-harmful here and the sign + a KL
# leash suffice; if it underperforms, the magnitude carried real credit a pure sign cannot replace.
# =====================================================================================
#
# --- v1 method (changes vs the base, retained below) ---
# Variant of v162critic_dreamer3bucket_hlgauss_mtp_v1: ports DreamerV3's advantage-scale
# stack (as in dg_beta v15/v16) onto this MTP base. THREE changes vs the base:
#   (1) NO rankgauss: adv_transform="v10" (identity). The base's flagship rank-Gaussian
#       shaping is removed -- it already maps advantages to ~N(0,1), so a percentile
#       norm on top of it would be a constant-divide no-op. ("no advantage norm")
#   (2) NO norm_adv: per-minibatch standardization off. ("no advantage norm")
#   (3) DreamerV3 PERCENTILE NORM is the SOLE advantage scaler: policy_adv <- policy_adv /
#       max(1, EMA(P95)-EMA(P5)) over the raw GAE returns (EMA rate 0.01, divide-only).
# Reward stays RAW (base already defaults normalize_reward=clip_reward=False). The CRITIC
# stays in RAW space -- unchanged Dreamer3 511-bucket symexp HL-Gauss MTP head regressing
# raw returns (DreamerV3 valnorm=none; same arrangement as dg_beta v15/v16). Trust region
# is the base's PPO clip-higher (0.2/0.28), which bounds the step at any advantage scale.
#
# HYPOTHESIS: faithful DreamerV3 normalization (percentile spread + hard floor, raw-return
# symexp critic) on the strong MTP base is at least as stable as rankgauss while being a
# cleaner, more principled scaler. Falsifiable: if it underperforms the rankgauss base, the
# rank-only advantage (magnitude-discarding) was doing real work that a pure scale can't
# replace. Watch charts/ret_perc_scale (the EMA percentile spread S).
# =====================================================================================
#
# --- Base method (unchanged below) ---
# Hypothesis: keep iterthink_v24_beta_s1's PPO/Beta actor and ThinkTrunk, but
# replace the v24 distributional lambda-return critic with the v162 critic plus
# Dreamer3-style exponentially spaced value buckets over v162's raw range:
#   - value bucket centers are symexp(linspace(symlog(-20000), symlog(20000), 511))
#   - HL-Gauss target mass is integrated over the matching symlog-coordinate
#     bucket intervals instead of two-hot interpolation
#   - expected-scalar decode E[symexp(bin)] for values and bootstraps
#   - bias-free neutral critic logits instead of a peaked zero prior
#   - critic MTP head predicting returns[t + h] for h=0..5 with boundary masks
# This isolates whether Dreamer3's high-resolution near-zero / wide-tail bucket
# geometry improves the already-strong v162 critic port, without importing the
# v162 world model, CUDA graph path, or imagined updates.
#
# Base: ppo_continuous_action_iterthink_v24_dist.py / iterthink_v24_beta_s1.
# Critic donor: ppo_continuous_action_iterthink_v162_compiled_wmloss_cudagraph_edgeclamp_contdisc_k6.py.
#
# PPO + IterThink v24 (ACTION-DISTRIBUTION TOGGLE: dreamer4-faithful). From v21.
#
# SOFT MAX-ENTROPY add-on (--auto-entropy, gaussian only). This borrows SAC's
# tanh-squashed log-prob, target-entropy heuristic, and temperature dual, but keeps
# the PPO critic on the RAW reward return. Entropy enters the actor two ways:
#   (1) a current-state squashed-entropy actor bonus, -alpha * log pi_sq(a|s);
#   (2) a policy-only soft GAE whose one-step bootstrap adds alpha * H_sq(s_{t+1})
#       using the rollout/bootstrapped squashed log-prob sample.
# The critic target is deliberately entropy-free so the fixed support remains
# calibrated. In this variant the target is v162 scalar-return HL-Gauss MTP over
# Dreamer3-spaced buckets.
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
# All paths keep the mean-value GAE. This variant changes the value target from
# v24's distributional lambda-return to v162 scalar-return HL-Gauss MTP over
# Dreamer3 buckets; only the policy advantage transforms are selected by
# `adv_transform`. sigma(s) is the std of the OLD rollout Z(s_t), floored at
# `sigma_floor_bins` bins.
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
from torch.distributions.kl import kl_divergence
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
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = False           # d3percnorm: NO per-minibatch standardization ("no advantage norm").
    # --- Percentile advantage normalization (the sole advantage scaler) ---
    ret_percnorm: bool = True        # scale policy advantage by S = max(floor, P95-P5) of returns
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

    # BetaPlast: per-neuron signed-Beta plasticity on the actor's hidden feature.
    plast_delta: float = 0.15        # half-width of the multiplicative gain support (1-delta, 1+delta)
    plast_conc_init: float = 3.0     # initial per-neuron Beta concentration c (higher = more rigid)
    plast_conc_max: float = 30.0     # concentration cap: enforces a plasticity floor (min gain variance)
    plast_gate_min: float = 0.25     # lower clamp on the per-neuron post-Adam step gate (v1's proven moderate range)
    plast_gate_max: float = 4.0      # upper clamp on the per-neuron post-Adam step gate (v1's proven moderate range)
    # Gate normalization. "relative": gate_i = proxy_i / mean(proxy) -> zero-sum reallocation
    # (competitive, mean pinned to 1). "absolute": proxy_i / proxy(conc_init) -> decoupled,
    # global LR floats freely. "bounded_absolute" (default): absolute shape, but the layer-MEAN
    # gate is rescaled into [gate_mean_min, gate_mean_max] so global LR breathes but never collapses.
    plast_gate_mode: str = "relative"
    plast_gate_mean_min: float = 0.25  # lower bound on the LAYER-MEAN gate (expansive global LR floor: 1/4x)
    plast_gate_mean_max: float = 4.0   # upper bound on the LAYER-MEAN gate (expansive global LR ceiling: 4x)

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

    # --- dreamer4 PMPO actor objective (policy_objective="pmpo"). Replaces the PPO clipped
    # surrogate with a sign-based update on the RAW policy GAE advantage (NO advnorm / shaping /
    # percentile / pos_neg_alpha -- all bypassed), balanced over pos/neg, with a closed-form
    # reverse Beta-KL trust region to the rollout policy. "ppo" reproduces the base exactly. ---
    #   "signneg_ppo" (v4/v5): PPO clipped-ratio surrogate with a modified advantage --
    #     negative advantages are replaced by their SIGN (-1); positives per signneg_pos_mode.
    policy_objective: str = "signneg_ppo"  # "pmpo" | "ppo" | "signneg_ppo"
    # signneg_pos_mode: how POSITIVE advantages enter the clip surrogate.
    #   "ppo"  (v4): normalized magnitude (divide-by-std; mean NOT subtracted, so the raw sign
    #               the pos/neg split uses is preserved) -- "normal PPO adv treatment" for positives.
    #   "sign" (v5): pure sign (+1) -- eff_adv = sign(A) everywhere, NO advnorm at all.
    signneg_pos_mode: str = "ppo"
    # How NEGATIVE advantages enter eff_adv:
    #   "sign"     (v4): hard -1 (uniform, magnitude-free suppression).
    #   "softtanh" (v8): -tanh(|A|) in (-1,0) -- GRADED suppression (worse actions pushed down
    #                    harder, but bounded at 1). Controls whether hard sign beats soft on the
    #                    negative side, holding magnitude-positive fixed.
    signneg_neg_mode: str = "softtanh"
    # softtanh shaping (scale-robust): neg_eff = tanh(signneg_neg_temp * z), where z is the raw
    # advantage, std-normalized when signneg_neg_norm=True (so temp is in std-units regardless of
    # the raw GAE scale). Larger temp -> sharper (toward hard sign, full suppression sooner);
    # smaller temp -> wider graded region (gentle for near-zero advantages) but STILL saturates to
    # -1 for clearly-bad actions (unlike v7's uniform weakening). v9=2.0 (sharper), v10=0.5 (wider).
    signneg_neg_norm: bool = True
    signneg_neg_temp: float = 0.5
    pmpo_pos_to_neg_weight: float = 0.5  # positive-advantage weight; negative side uses 1 - this
    pmpo_kl_coef: float = 0.3        # reverse KL(old rollout || new) trust-region coefficient
    pmpo_reverse_kl: bool = True     # True: KL(old||new) (reverse); False: KL(new||old) (ablation)

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

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
        # BetaPlast: per-neuron signed-Beta plasticity on the actor's H-dim hidden
        # feature (the trunk output feeding the actor heads). Learned per-neuron
        # concentration c = 1 + (conc_max-1)*sigmoid(rho); frozen per-iteration gain
        # noise (log_u, r) is set by resample_gain_noise. Gain is applied ACTOR-ONLY.
        self.gain_dim = H
        self.delta = args.plast_delta
        self.conc_max = args.plast_conc_max
        p_init = (args.plast_conc_init - 1.0) / (args.plast_conc_max - 1.0)
        rho_init = float(np.log(p_init / (1.0 - p_init)))
        self.rho = nn.Parameter(torch.full((H,), rho_init))
        self.log_u = self.r = None  # frozen per-iteration gain noise
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

    def concentration(self):
        # Per-neuron Beta concentration, bounded (1, conc_max) with gradient alive
        # at both ends (no absorbing rigid state).
        return 1.0 + (self.conc_max - 1.0) * torch.sigmoid(self.rho)

    def resample_gain_noise(self, num_envs, device):
        # One noise draw per env per iteration; antithetic sign pairs across env pairs.
        # Frozen for the whole rollout AND all update epochs so PPO ratios / PMPO KL stay exact.
        assert num_envs % 2 == 0, "antithetic gain noise requires an even number of envs"
        half = num_envs // 2
        flip = torch.tensor([1.0, -1.0], device=device).repeat(half).unsqueeze(1)
        u = torch.rand(half, self.gain_dim, device=device).clamp(1e-6, 1.0).log()
        r = torch.randint(0, 2, (half, self.gain_dim), device=device, dtype=torch.float32) * 2.0 - 1.0
        self.log_u = u.repeat_interleave(2, dim=0)
        self.r = r.repeat_interleave(2, dim=0) * flip

    def gains(self):
        # Signed Beta(1, c) gain: s = 1 + delta * r * (1 - u^(1/c)), differentiable in c.
        c = self.concentration()
        return 1.0 + self.delta * self.r * (1.0 - torch.exp(self.log_u / c))

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

    def get_action_and_value(self, x, z=None, env_ids=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        # env_ids selects each row's frozen per-env gain: the actor feature is
        # perturbed (ACTOR-ONLY), the critic feature is left clean.
        actor_feat, critic_feat = self._trunks(x)
        if env_ids is not None:
            actor_feat = actor_feat * self.gains()[env_ids]
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
        # Also return the action distribution (`dist`) so the PMPO actor objective can
        # read PER-DIM log-probs and rebuild the closed-form Beta-KL to the rollout policy.
        # For Beta, log_det_fn(z)==0 (the linear rescale Jacobian is dropped/constant), so
        # log_prob == dist.log_prob(z).sum(1) EXACTLY -- per-dim log-probs are dist.log_prob(z).
        return action, z, log_prob, entropy, value_logits, dist

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


def plasticity_gates(agent, args):
    # Per-neuron post-Adam step gate from gain variance E[x^2] = 2 / ((c+1)(c+2)).
    #   "relative": normalize to the layer mean -> zero-sum reallocation (competitive).
    #   "absolute": normalize to the INIT concentration -> decoupled, global LR floats.
    #   "bounded_absolute": absolute shape, but the layer-MEAN gate is rescaled into
    #     [mean_min, mean_max] so global LR breathes but never collapses.
    # Returned as a one-element list (one gained layer).
    with torch.no_grad():
        c = agent.concentration()
        proxy = 2.0 / ((c + 1.0) * (c + 2.0))
        if args.plast_gate_mode == "relative":
            ref = proxy.mean()
        elif args.plast_gate_mode in ("absolute", "bounded_absolute"):
            c0 = args.plast_conc_init
            ref = 2.0 / ((c0 + 1.0) * (c0 + 2.0))
        else:
            raise ValueError(f"unknown plast_gate_mode {args.plast_gate_mode}")
        g = proxy / ref
        if args.plast_gate_mode == "bounded_absolute":
            # Keep the decoupled (absolute) shape, but pull the LAYER-MEAN gate into
            # [mean_min, mean_max] so the global LR can breathe but never collapse.
            # The per-neuron clamp below is the harder safety bound and is applied
            # last, so on a strongly bimodal layer it can relax the realized mean
            # slightly ABOVE mean_max (the benign direction); it never lets the mean
            # collapse below mean_min, which is the failure this guards against.
            m = g.mean()
            target = m.clamp(args.plast_gate_mean_min, args.plast_gate_mean_max)
            g = g * (target / m.clamp_min(1e-8))
        return [g.clamp(args.plast_gate_min, args.plast_gate_max)]


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch", "batch_retstd", "minibatch_retstd")
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
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # BetaPlast: the gained neurons live in the actor trunk's output feature, so the
    # layer that PRODUCES them (and whose post-Adam step is plasticity-gated) is that
    # trunk's out_proj. With share_backbone=True this is the shared trunk (the gate
    # then mildly reallocates LR for the critic too, bounded by the gate clamp/band).
    actor_trunk_ref = agent.trunk if agent.share_backbone else agent.actor_trunk
    gated_layers = [actor_trunk_ref.out_proj]
    rollout_env_ids = torch.arange(args.num_envs, device=device)
    b_env_ids = torch.arange(args.batch_size, device=device) % args.num_envs

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

    # dreamer4 PMPO actor objective is active only for the Beta actor (it needs per-dim Beta
    # log-probs and the closed-form Beta-KL). For policy_objective=="ppo" the base path runs.
    pmpo_active = args.policy_objective == "pmpo"
    if pmpo_active and args.actor_dist != "beta":
        raise ValueError("policy_objective='pmpo' requires actor_dist='beta'")
    # v4: hybrid "sign-for-negative, normal-PPO-for-positive" clipped surrogate. Positive
    # advantages keep their (std-normalized) magnitude and run the standard PPO clip; negative
    # advantages are replaced by their SIGN (-1), so they contribute a uniform, magnitude-free
    # suppression under the same clip trust region. Uses raw GAE for the sign decision.
    signneg_active = args.policy_objective == "signneg_ppo"

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # OLD (rollout) Beta concentrations alpha=concentration1, beta=concentration0; per-dim.
    old_alphas = torch.ones((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    old_betas = torch.ones((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
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
            optimizer.param_groups[0]["lr"] = lrnow

        # BetaPlast: fresh gain noise, frozen for this rollout and its update epochs.
        agent.resample_gain_noise(args.num_envs, device)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits, act_dist = agent.get_action_and_value(
                    next_obs, env_ids=rollout_env_ids
                )
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
                if pmpo_active:
                    # Store the OLD (rollout) Beta concentrations so old_dist can be rebuilt
                    # for the PMPO reverse-KL. detach() not needed (no_grad), but explicit.
                    old_alphas[step] = act_dist.concentration1.detach()
                    old_betas[step] = act_dist.concentration0.detach()
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

        with torch.no_grad():
            next_transition_value_logits = agent.get_value(
                next_obses.reshape((-1,) + envs.single_observation_space.shape)
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
                # Sample a' ~ π(·|s_T) for the bootstrap entropy (SAC's single-sample).
                _, _, boot_logprob, _, _, _ = agent.get_action_and_value(
                    next_obses[-1], env_ids=rollout_env_ids
                )
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
            if not pmpo_active and not signneg_active and args.ret_percnorm and args.ret_perc_scope in ("ema", "batch"):
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
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        # OLD Beta concentrations for the PMPO reverse-KL (detached: no grad through old policy).
        b_old_alphas = old_alphas.reshape((-1,) + envs.single_action_space.shape).detach()
        b_old_betas = old_betas.reshape((-1,) + envs.single_action_space.shape).detach()
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()
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
        clipfracs = []
        # PMPO diagnostics (last-minibatch values; harmless defaults for the ppo path).
        reverse_kl_val = torch.zeros((), device=device)
        pmpo_pos_loss_val = torch.zeros((), device=device)
        pmpo_neg_loss_val = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits, new_dist = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds], env_ids=b_env_ids[mb_inds]
                )
                # PMPO needs PER-DIM current log-probs of the taken (replayed) action under the
                # CURRENT Beta policy. For Beta, log_det_fn(z)==0, so the base's summed newlogprob
                # equals new_dist.log_prob(z).sum(1); the per-dim term is exactly new_dist.log_prob(z)
                # (z == b_latent_zs[mb_inds], the Beta-native (0,1) sample replayed from the buffer).
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if pmpo_active:
                    # ===== dreamer4 PMPO actor objective (sign-based; NO advnorm/shaping/percnorm). =====
                    # mb_advantages is the RAW policy GAE for the minibatch (b_advantages == raw GAE
                    # when policy_objective=="pmpo": auto_alpha defaults off and the batch ret_percnorm
                    # block is bypassed under PMPO). The PPO clip max() path is NOT used here.
                    mb_advantages = b_advantages[mb_inds]
                    log_prob_per_dim = new_dist.log_prob(b_latent_zs[mb_inds])  # (mb, act_dim)
                    # Soft sign in [0,1): "essentially sign based" (magnitude squashed away).
                    adv_weight = mb_advantages.tanh().abs().unsqueeze(-1)
                    signed_log_probs = log_prob_per_dim * adv_weight
                    pos_mask = (mb_advantages >= 0).unsqueeze(-1).expand_as(log_prob_per_dim)
                    neg_mask = ~pos_mask
                    pos_count = pos_mask.float().sum().clamp_min(1.0)
                    neg_count = neg_mask.float().sum().clamp_min(1.0)
                    # Balance pos/neg by their OWN counts (class imbalance cannot dominate).
                    pos_loss = signed_log_probs[pos_mask].sum() / pos_count
                    neg_loss = signed_log_probs[neg_mask].sum() / neg_count
                    pg_loss = (
                        -args.pmpo_pos_to_neg_weight * pos_loss
                        + (1.0 - args.pmpo_pos_to_neg_weight) * neg_loss
                    )
                    # Closed-form reverse Beta-KL trust region to the ROLLOUT policy (replaces the clip).
                    old_dist = Beta(b_old_alphas[mb_inds], b_old_betas[mb_inds])
                    reverse_kl = kl_divergence(old_dist, new_dist).sum(-1).mean()  # KL(old||new)
                    pmpo_pos_loss_val = pos_loss.detach()
                    pmpo_neg_loss_val = neg_loss.detach()
                    reverse_kl_val = reverse_kl.detach()
                    if args.pmpo_kl_coef > 0.0:
                        kl_loss = reverse_kl if args.pmpo_reverse_kl else kl_divergence(new_dist, old_dist).sum(-1).mean()
                        pg_loss = pg_loss + args.pmpo_kl_coef * kl_loss
                elif signneg_active:
                    # Hybrid: negative advantages -> sign (-1); positives per signneg_pos_mode.
                    # The PPO clip is the sole trust region (no KL term). Sign is taken from the
                    # RAW GAE so the pos/neg decision matches the true advantage sign.
                    mb_adv_raw = b_advantages[mb_inds]
                    # Negative-advantage magnitude: hard sign (-1) or graded soft -tanh(|A|).
                    if args.signneg_neg_mode == "sign":
                        neg_eff = torch.full_like(mb_adv_raw, -1.0)
                    elif args.signneg_neg_mode == "softtanh":
                        z = mb_adv_raw
                        if args.signneg_neg_norm:
                            z = z / (z.std() + 1e-8)  # std-units so temp is scale-robust
                        neg_eff = (args.signneg_neg_temp * z).tanh()  # graded in (-1,0)
                    else:
                        raise ValueError(f"unknown signneg_neg_mode {args.signneg_neg_mode}")
                    if args.signneg_pos_mode == "ppo":
                        # divide-by-std normalization (mean NOT subtracted -> raw sign preserved).
                        pos_mag = mb_adv_raw / (mb_adv_raw.std() + 1e-8)
                        eff_adv = torch.where(mb_adv_raw >= 0, pos_mag, neg_eff)
                    elif args.signneg_pos_mode == "sign":
                        # pure sign for positives (+1); negatives per neg_eff. NO advnorm.
                        eff_adv = torch.where(mb_adv_raw >= 0, torch.sign(mb_adv_raw), neg_eff)
                    else:
                        raise ValueError(f"unknown signneg_pos_mode {args.signneg_pos_mode}")
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    pg_loss1 = -eff_adv * ratio
                    pg_loss2 = -eff_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                else:
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

                # BetaPlast: snapshot the gated layer(s) before the step so the
                # realized Adam update can be per-neuron rescaled afterwards. Adam's
                # m/v state is left untouched (pre-Adam scaling is undone by Adam).
                gates = plasticity_gates(agent, args)
                prev = [(l.weight.detach().clone(), l.bias.detach().clone()) for l in gated_layers]

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

                # BetaPlast: scale each gained neuron's realized step by its plasticity
                # gate: w <- w_prev + gate * (w_adam - w_prev).
                with torch.no_grad():
                    for layer, gate, (w_prev, b_prev) in zip(gated_layers, gates, prev):
                        layer.weight.copy_(w_prev + gate.unsqueeze(1) * (layer.weight - w_prev))
                        layer.bias.copy_(b_prev + gate * (layer.bias - b_prev))

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
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        if pmpo_active:
            writer.add_scalar("losses/reverse_kl", reverse_kl_val.item(), global_step)
            writer.add_scalar("losses/pmpo_pos_loss", pmpo_pos_loss_val.item(), global_step)
            writer.add_scalar("losses/pmpo_neg_loss", pmpo_neg_loss_val.item(), global_step)
            # How much advantage magnitude survives tanh, vs. how often it saturates to pure sign.
            # adv_tanh_mag -> 1 and satfrac -> 1 mean the |tanh| weight is ~constant => effectively sign-only.
            with torch.no_grad():
                _adv_abs = b_advantages.abs()
                writer.add_scalar("losses/adv_abs_mean", _adv_abs.mean().item(), global_step)
                writer.add_scalar("losses/adv_tanh_mag", b_advantages.tanh().abs().mean().item(), global_step)
                writer.add_scalar("losses/adv_tanh_satfrac", (_adv_abs > 2.0).float().mean().item(), global_step)
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
        with torch.no_grad():
            c = agent.concentration()
            gate = plasticity_gates(agent, args)[0]
            gain_std = args.plast_delta * torch.sqrt(2.0 / ((c + 1.0) * (c + 2.0))).mean()
            writer.add_scalar("plast/c_mean", c.mean().item(), global_step)
            writer.add_scalar("plast/c_min", c.min().item(), global_step)
            writer.add_scalar("plast/c_max", c.max().item(), global_step)
            writer.add_scalar("plast/gate_min", gate.min().item(), global_step)
            writer.add_scalar("plast/gate_max", gate.max().item(), global_step)
            writer.add_scalar("plast/gate_mean", gate.mean().item(), global_step)
            writer.add_scalar("plast/gain_std", gain_std.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

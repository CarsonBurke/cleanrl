# PROBE-DISTILL + DEEP VALUE v1 -- the two best DISTILLATION levers on this disk, combined
# for the first time. HalfCheetah-v4 only. No PPO anywhere: no ratio, no clipped surrogate,
# no policy gradient, no BC/clone term, and ent_coef defaults to 0, so the ENTIRE actor
# objective is CE(q_frozen, pi) = forward KL(q || pi) to a teacher that saw privileged
# information. This is on-policy self-distillation in the paper's actual sense.
# =====================================================================================
# WHY THIS FILE EXISTS. My own OPSD-AdvCond lineage conditions the teacher on the REALIZED
# scalar advantage of the ONE action that was taken. Measured against its own parent chassis
# (identical critic, trunk, PPO actor) on HalfCheetah-v4 seed 1:
#
#            @1M    @2M    @4M    @6M    @8M
#   parent   3079   5012   6716   7548   8278
#   advcond  3826   6267   7713   8579   8812     delta +24% +25% +15% +14% +6.4%
#
# The edge DECAYS MONOTONICALLY. That is the signature of a teacher whose target carries
# exactly the information content of a policy gradient -- one action, one scalar credit --
# so it can only reach the same asymptote sooner. To raise the asymptote the teacher has to
# know something the student structurally cannot.
#
# LEVER 1 (inherited, unchanged): a teacher with genuinely privileged information. At every
# rollout state K=8 candidates are drawn from the rollout policy (candidate 0 is the executed
# action) and each is stepped ONE tick through the raw MuJoCo physics with a direct-assign
# save/restore of qpos/qvel/qacc_warmstart/time, so the real env.step is bit-unaffected.
# Score is the oracle one-step Q, s_k = r_k + gamma(1-term_k) V_old(s'_k), centered per group
# and divided by ONE GLOBAL sigma (an EMA of the executed-transition TD-residual RMS) so
# scores live in signal-to-noise units. Target q = softmax(anchor_logp + u/eta), solved once
# per rollout and detached; eta comes from a ONE-SIDED KL cap (eta >= eta_base always, KL
# free to float below the cap). This file's ancestor scored 10411 with exactly this actor.
# CAVEAT, stated plainly: the probe is SIMULATOR-PRIVILEGED. It evaluates counterfactual
# actions by resetting physics state, which is not a standard RL assumption. It is the
# mechanism behind the best model in this repo, so it is in scope here, but any comparison
# against methods without simulator access is not apples-to-apples.
#
# LEVER 2 (the change): the value readout. This chassis reads V(s) out of ONE biasless
# zero-init Linear(H, num_bins). On a different chassis, replacing exactly that Linear with a
# depth-3 adaLN residual stack -- and changing NOTHING else, verified from both runs' stored
# hyperparameters (identical lr, epochs, minibatches, bins, MTP horizon, sigma ratio, reward
# normalization, hidden, k_blocks, n_experts) -- moved 8455 -> 9810: +245 @2M, +1389 @3M,
# +1397 @4M, +1590 @6M, +1355 end. The frontier lineage never combined these two: it spent
# its second lever on an all-layer representation tape instead. So this pairing is untested.
#
# PARAMETER HONESTY. Unlike the MTP chassis, where the deep head is CHEAPER (163,648 vs
# 196,224 params, 0.83x, because one shared decoder replaces an independent num_bins-way
# projection per horizon), this critic has NO MTP token axis: the head goes 32,704 ->
# 163,328 params, 4.99x. On this file depth and capacity are therefore NOT separated, and
# "it just has more parameters" IS available as an explanation. The companion file
# ppo_continuous_action_opsd_advcond_v11.py is the clean attribution arm (0.83x params);
# this one is the score arm. Do not read this file as evidence about depth per se.
#
# VERIFIED BEFORE LAUNCH, not assumed:
#   * logits exactly 0 at init => V(s) == 0 on the symmetric support, as for the Linear.
#   * the all-zero fixed point is REAL: with z_start forced to 0 the head's total gradient
#     norm is exactly 0.000e+00 (against 5.68e-02 with the learned nonzero start). A zero
#     start would have silently trained nothing.
#   * the dyntrust solver still binds with the new head: kl_achieved 0.0300 == the cap,
#     eta_solved 1.68 -> 1.37 (all >= eta_base 1.0, the one-sided property), cap_engaged 1.0,
#     kl_base 0.083 > cap; explained_variance rising from 0.
#
# FALSIFIABLE PREDICTIONS:
#   1. > 10411 at matched steps (the ancestor's score with the shallow head). If it merely
#      matches, the readout lever does not compose with the probe teacher, and the honest
#      conclusion is that this critic's readout was never the binding constraint.
#   2. explained_variance exceeds the ancestor's BEFORE the return does. If return improves
#      while EV does not, the attribution to the readout is wrong.
#   3. If it beats 11015 it is the best model in this repo, achieved with a strictly
#      distillation actor and without the all-layer tape.
#
# Inherited ancestor notes below, unchanged.
# =====================================================================================
# PPO + IterThink v24 beta d4-HL-Gauss symlog raw-return idx-transfer + TPO-MD v5
# (DYNAMIC, signal-determined trust region; one-sided KL cap).
#
# METHOD: pure mirror-descent policy update — the probe-scored TPO target REPLACES
# the PPO policy gradient. No PG, no ratio clipping in any backward (pg_loss /
# approx_kl / clipfrac remain telemetry only). At EVERY rollout state, K=8
# candidates from the rollout policy (candidate 0 = the executed action) are each
# stepped one tick through the RAW MuJoCo physics (direct-assign sim restore; the
# real trajectory is bit-identical to an unprobed run) and scored with the FROZEN
# pre-update critic: s_k = r_k + gamma * (1 - term) * V_old(s'_k). Scores are
# centered per group but scaled by ONE GLOBAL sigma (tpo_sigma_scale_coef * EMA of
# the one-step TD-residual RMS): global-sigma scaling preserves cross-state
# advantage-MAGNITUDE weighting — the thing naive per-group z-scoring destroys by
# pushing full-strength updates on noise-level groups. Because u_scores are in
# SIGNAL-TO-NOISE units (score advantage measured against the critic's own
# one-step noise floor), the natural KL the raw signal produces at any FIXED
# temperature is large when candidates carry real signal and -> 0 when they don't.
#
# WHY v5 (the forced-noise-step decline): tpomd_v1 solved eta so batch-mean
# KL(p_old||q) == tpo_eps EVERY update — a FIXED target. Late in training, once
# candidates have converged and score diffs shrink inside the critic's noise floor,
# forcing a constant 0.03 step makes the solver pick a SMALL eta that AMPLIFIES
# that noise back up to 0.03 — injecting noise into an already-good policy and
# causing a late-training performance DECLINE. We want the step to SHRINK ON ITS
# OWN when signal is weak.
#
# THE FIX — one-sided KL cap on a fixed base temperature (emergent, no schedule):
# Fix a base temperature eta_base (tpo_eta_base, default 1.0). Let
# kl_base = mean KL(p_old||q(eta_base)) be the NATURAL (uncapped) step the raw
# SNR signal produces at that fixed eta. Reinterpret tpo_eps (0.03) as a CAP, the
# MAX allowed KL — not a target. Each update:
#   kl_base = tpo_mean_kl(eta_base)
#   if kl_base <= eps_cap:  eta_solved = eta_base   # weak signal: natural step is
#                                                   # already within cap; KL FLOATS
#                                                   # BELOW it, shrinking toward 0.
#   else:                   bisect log-eta over [log(eta_base), log(1e4)] for
#                           KL == eps_cap            # strong signal: RAISE eta to
#                                                    # pull the over-large step down.
# KL(p_old||q(eta)) is MONOTONE DECREASING in eta (eta -> inf => q -> p_old =>
# KL -> 0), so raising eta from eta_base strictly lowers KL: eta_solved >= eta_base
# ALWAYS. The step is bounded above by eps_cap (stability, identical to v1's early
# behavior) and is FREE to shrink to ~0 when signal vanishes (the fix). There is
# NO lower floor on KL — intentional. Early training (strong signal) is the same
# trust region as v1 (capped at eps), so nothing destabilizes before the cap
# engages. q is solved once, detached, frozen across epochs. Actor loss =
# tpo_coef * CE(q, softmax of new K-restricted logprobs) - ent_coef * entropy.
#
# Set tpo_dyn_trust=False to reproduce tpomd_v1 EXACTLY (the original fixed-KL-
# target bisection over [1e-2, 1e4]; eta_base unused).
#
# DECOUPLED EPOCHS: the critic ALWAYS trains all update_epochs (target_kl=None
# here). The actor's CE epochs stop only via a circuit breaker — epoch-mean
# approx_kl (executed actions) > tpo_kl_breaker (3x eps) — after which a
# critic-only backward keeps running. Predecessor pure attempt failed via eta=1 +
# the 0.03 target_kl leash -> 1 epoch/iter + critic starvation; both causes are
# removed here (eps-calibrated target, never-starved critic).
#
# KILL CRITERIA: debug/tpo_kl_achieved persistently pinned AT the cap with no
# late-training shrink (dynamic trust region not engaging); debug/tpo_cap_engaged
# never falling off late; debug/actor_epochs_completed pinned at 1; entropy
# freefall vs the base run.
#
# --- inherited base notes ---
# PPO + IterThink v24 beta d4-HL-Gauss symlog raw-return critic idx-transfer v1.
#
# Variant hypothesis: remove Gym's reward/discounted-return normalization and
# reward clipping, then rely on the symlog HL-Gauss critic to keep raw MuJoCo
# returns trainable. The critic uses v149's symmetric symlog-coordinate support
# for raw [-20k, 20k] returns, expected-raw scalar decoding, and a biasless
# HL-Gauss head, but intentionally keeps the original single-horizon value
# target rather than v149's MTP critic. Observation preprocessing and PPO/action
# machinery stay matched to the source run.
#
# Index-transfer variant: keep the existing DenseNet-style trunk concatenation
# and residual branch sum, but each branch MLP hidden perceptron also receives
# the same-index historical perceptron values in its preactivation. For hidden
# channel j, phi sees the normal current-state affine preactivation plus learned
# scalar transfers from cat_feats[:, slot, j]. There is no full cross-channel
# history projection inside the transfer function.
#
# Base run: HalfCheetah-v4__iterthink_v24_beta_d4hlgauss_symlog_v1_hc8m__1__1780787886.
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
# calibrated. In this variant the target is scalar-return HL-Gauss rather than
# v24's distributional lambda-return recursion.
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
# v24's distributional λ-return to scalar-return HL-Gauss projection; only the
# policy advantage transforms are selected by `adv_transform`. sigma(s) is the
# std of the OLD rollout Z(s_t), floored at `sigma_floor_bins` bins.
import math
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

from cleanrl.shared.hl_gauss import HLGaussSupport, symlog, symexp

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


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

    # --- deep value readout (replaces one zero-init Linear) ---
    value_blocks: int = 3         # stack depth; 3 is the measured setting (+1355 end)
    value_mult: int = 2           # block MLP width = value_mult * hidden
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


class ValueTimeEmbed(nn.Module):
    """DiT timestep embedder, retained at its single e2e operating point.

    The source lineage runs this stack in two regimes: a diffusion solve that visits many
    noise levels, and an end-to-end residual control fixed at sigma == 1. ONLY the control
    ever scored (9810 vs its 8455 base); no diffusion arm exists on disk. So the embedder's
    input here is the constant c_noise = 0.25*log(1) = 0, which makes its output a LEARNED
    CONSTANT VECTOR -- and the adaLN shift/scale it drives are learned per-block constants.

    It is kept rather than folded into biases because that is what was measured, and it is
    evaluated on ONE row instead of T: the MLP is row-wise, so a (1,H) result broadcast over
    the token axis is mathematically identical to the (T,H) version at 1/T the cost. It
    agrees to float32 rounding, not to the bit (measured max|diff| = 4.8e-07 at T = 192):
    matmul reduction order differs with batch shape.
    """

    def __init__(self, H, n_freq=16, f_min=0.5, f_max=8.0):
        super().__init__()
        self.register_buffer(
            "freqs", torch.logspace(math.log10(f_min), math.log10(f_max), n_freq)
        )
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(2 * n_freq, H)), ReLUSquared(), layer_init(nn.Linear(H, H))
        )

    def forward(self, c_noise):
        phase = c_noise.unsqueeze(-1) * self.freqs
        return self.mlp(torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1))


class DeepValueHead(nn.Module):
    """Depth-B adaLN residual stack over MTP-horizon TOKENS, replacing one zero-init Linear.

    The MTP heads become B_tok = mtp tokens sharing one conditioning, each carrying its own
    state. There is no attention: given the conditioning the horizon tokens are
    conditionally independent, so a mixer over independent value distributions buys nothing.

        cond_t = feat + h_embed[t]                       (T = B*mtp tokens)
        z_0    = z_start
        z_b    = z_{b-1} + out(ReLU(in([RMS(z),RMS(cond)]) * (1+scale) + shift)^2)
        logits = decoder(z_B)

    THREE INIT PROPERTIES ARE LOAD-BEARING, not decoration:
      * decoder is zero-init, so logits == 0 => uniform bins => V(s) is EXACTLY 0 at init on
        the symmetric symexp support. This reproduces the zero-init Linear it replaces, so
        the critic starts from the same place and the actor sees no init shock.
      * out_w/out_b are zero-init, so every block is EXACTLY the identity at init: z_B ==
        z_start. Depth is therefore free at step 0 and is earned, not imposed.
      * z_start is a LEARNED NONZERO parameter. With a zero start, a zero-init decoder and
        zero-init out_proj, the stack sits on an exact all-zero fixed point: dL/d_decoder =
        dL/d_logits (x) z == 0 and dL/d_out = decoder^T dL/d_logits (x) . == 0, so NOTHING
        learns (the source measured gradient norm exactly 0.00 for 4k steps). A nonzero
        query breaks the deadlock the same way the trunk feature does for a zero-init linear
        head, and V(s) is still exactly 0 at init.
    """

    def __init__(self, H, num_bins, mtp, num_blocks, mult):
        super().__init__()
        Hm = mult * H
        self.H, self.num_bins, self.mtp = H, num_bins, mtp
        self.num_blocks, self.Hm = num_blocks, Hm
        self.h_embed = nn.Embedding(mtp, H)
        nn.init.normal_(self.h_embed.weight, std=0.02)
        self.t_embed = ValueTimeEmbed(H)
        self.z_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.c_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.in_w = nn.Parameter(torch.empty(num_blocks, Hm, 2 * H))
        self.in_b = nn.Parameter(torch.zeros(num_blocks, Hm))
        self.ada_w = nn.Parameter(torch.zeros(num_blocks, 2 * Hm, H))
        self.ada_b = nn.Parameter(torch.zeros(num_blocks, 2 * Hm))
        self.out_w = nn.Parameter(torch.zeros(num_blocks, H, Hm))
        self.out_b = nn.Parameter(torch.zeros(num_blocks, H))
        with torch.no_grad():
            for b in range(num_blocks):
                nn.init.orthogonal_(self.in_w[b], np.sqrt(2))
        self.decoder = nn.Linear(H, num_bins, bias=False)
        with torch.no_grad():
            self.decoder.weight.zero_()
        self.z_start = nn.Parameter(torch.randn(H))
        self.register_buffer("h_ids", torch.arange(mtp), persistent=False)
        self.register_buffer("c_noise0", torch.zeros(1), persistent=False)

    def forward(self, feat):
        """feat (B,H) -> value logits (B, mtp, num_bins)."""
        n = feat.shape[0]
        # (B,1,H) + (1,mtp,H) -> (B*mtp, H): token t of state i is row i*mtp + t.
        cond = (feat.unsqueeze(1) + self.h_embed(self.h_ids).unsqueeze(0)).reshape(-1, self.H)
        n_c = self.c_norm(cond)
        # adaLN modulation from the constant operating point: one row, broadcast over tokens.
        mod = F.linear(self.t_embed(self.c_noise0), self.ada_w.view(-1, self.H))
        mod = mod.view(self.num_blocks, 2 * self.Hm) + self.ada_b
        z = self.z_start.expand(n * self.mtp, self.H)
        for b in range(self.num_blocks):
            u = F.linear(torch.cat([self.z_norm(z), n_c], dim=-1), self.in_w[b], self.in_b[b])
            shift, scale = mod[b].chunk(2, dim=-1)
            u = torch.relu(u * (1.0 + scale) + shift).pow(2)
            z = z + F.linear(u, self.out_w[b], self.out_b[b])
        return self.decoder(z).view(n, self.mtp, self.num_bins)


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
        # v1: the biasless zero-init Linear(H, num_bins) becomes a depth-B adaLN
        # residual stack. Single value token (this critic has no MTP horizon), so the
        # lever here is READOUT DEPTH ONLY -- same support, same trunk feature, and
        # V(s) still exactly 0 at init because the decoder is zero-init.
        self.num_bins = args.num_bins
        self.critic_head = DeepValueHead(
            H, args.num_bins, 1, args.value_blocks, args.value_mult
        )
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
        return self.critic_head(critic_feat).squeeze(1)   # (B,1,bins) -> (B,bins)

    def get_action_and_value(self, x, z=None, candidate_zs=None, return_dist=False):
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
        value_logits = self.critic_head(critic_feat).squeeze(1)
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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        probe_seconds = 0.0

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits, roll_dist, roll_to_action, roll_log_det_fn = (
                    agent.get_action_and_value(next_obs, return_dist=True)
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
                next_transition_logits = agent.get_value(transition_next_obs_t)
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
            target_probs = hl_support.project(returns)
            # Per-state return std sigma(s_t) in raw return units, matching the
            # GAE scale consumed by tanh_std.
            sigma = (value_probs * (scalar_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * scalar_bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            returns_coord = symlog(returns) if args.value_symlog else returns
            cdf_frac = ((returns_coord.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

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
            # Chunked (<=65536 rows) critic forward with the CURRENT (pre-update) net.
            v_next = torch.cat(
                [value_logits_to_scalar(agent.get_value(chunk)) for chunk in flat_probe_obs.split(65536)]
            ).reshape(args.batch_size, args.tpo_k)
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
            p_old = anchor_logp.exp()

            def tpo_mean_kl(eta):
                # batch-mean KL(p_old || q(eta)); monotone DECREASING in eta
                # (eta -> inf => q -> p_old => KL -> 0).
                log_q_eta = F.log_softmax(anchor_logp + u_scores / eta, dim=-1)
                return (p_old * (anchor_logp - log_q_eta)).sum(-1).mean().item()

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
            b_tpo_q = torch.softmax(anchor_logp + u_scores / tpo_eta_solved, dim=-1).detach()
            tpo_kl_achieved = tpo_mean_kl(tpo_eta_solved)
            log_q = b_tpo_q.clamp_min(1e-12).log()
            tpo_group_kl = (b_tpo_q * (log_q - anchor_logp)).sum(-1).mean().item()
            tpo_q_entropy = (-(b_tpo_q * log_q).sum(-1)).mean().item()
            tpo_score_std_mean = group_std.mean().item()
            tpo_score_std_p90 = group_std.quantile(0.9).item()

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
        epochs_completed = 0
        actor_epochs_completed = 0
        actor_active = True  # flipped off by the tpo_kl_breaker; critic runs all epochs regardless
        actor_epoch_ce_means = []  # per-epoch mean CE while the actor is active (convergence probe)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            epoch_kls = []
            epoch_ces = []
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Same trunk forward as the base; the candidate logprobs ride the
                # SAME dist (no second trunk pass, consumes no RNG).
                _, _, newlogprob, entropy, value_logits, new_cand_logprobs = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds], candidate_zs=b_tpo_zs[mb_inds]
                )

                with torch.no_grad():
                    # TELEMETRY ONLY: ratio / KL / clipfrac (and pg_loss below) never
                    # reach a backward — the actor update is the TPO CE alone.
                    logratio = newlogprob.detach() - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    epoch_kls.append(approx_kl.item())

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
                    optimizer.zero_grad(set_to_none=True)
                    if actor_active:
                        # DUAL-BACKWARD decoupled clipping. Backprop value and policy
                        # gradients separately, clip each to its own max-norm, then sum
                        # on the (possibly shared) trunk so the critic's CE gradient
                        # cannot swamp the policy's contribution to shared features.
                        (args.vf_coef * v_loss).backward(retain_graph=True)
                        critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                        # Stash clipped value grads (must survive the zero before the
                        # policy backward; the shared trunk is in this set).
                        value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                        optimizer.zero_grad(set_to_none=True)
                        # The TPO CE is the entire actor backward, clipped under actor_grad_clip.
                        (actor_loss - ent_coef_eff * entropy_loss).backward()
                        actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                        # trunk.grad currently = clip_actor(d ce / d trunk); add the
                        # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                        for p, g in value_grads:
                            p.grad = g if p.grad is None else p.grad + g
                    else:
                        # Breaker tripped: CRITIC-ONLY backward. No retain_graph needed
                        # (single backward); actor-only params stay grad=None via
                        # set_to_none, so Adam skips them — no stale actor grads. The
                        # shared trunk receives only the critic branch, clipped to its
                        # own budget exactly as in the dual path.
                        (args.vf_coef * v_loss).backward()
                        critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                        actor_gn = 0.0
                    optimizer.step()
                else:
                    loss = v_loss * args.vf_coef
                    if actor_active:
                        loss = loss + actor_loss - ent_coef_eff * entropy_loss
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            epochs_completed = epoch + 1
            if actor_active:
                actor_epochs_completed = epoch + 1
                # Circuit breaker (NOT an epoch break): past 3x the per-update KL
                # budget the actor stops, but the critic keeps training all epochs.
                if float(np.mean(epoch_kls)) > args.tpo_kl_breaker:
                    actor_active = False
            # target_kl (default None here) would also stop the critic; kept only as
            # an explicit opt-in escape hatch.
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
        writer.add_scalar("debug/epochs_completed", epochs_completed, global_step)
        writer.add_scalar("debug/actor_epochs_completed", actor_epochs_completed, global_step)
        writer.add_scalar("losses/tpo_ce", tpo_ce.item(), global_step)
        writer.add_scalar("debug/tpo_eta_solved", tpo_eta_solved, global_step)
        writer.add_scalar("debug/tpo_kl_achieved", tpo_kl_achieved, global_step)
        writer.add_scalar("debug/tpo_kl_base", tpo_kl_base, global_step)
        writer.add_scalar("debug/tpo_cap_engaged", tpo_cap_engaged, global_step)
        writer.add_scalar("debug/tpo_group_kl", tpo_group_kl, global_step)
        writer.add_scalar("debug/tpo_score_std_mean", tpo_score_std_mean, global_step)
        writer.add_scalar("debug/tpo_score_std_p90", tpo_score_std_p90, global_step)
        writer.add_scalar("debug/tpo_sigma_global", tpo_sigma_global, global_step)
        writer.add_scalar("debug/tpo_q_entropy", tpo_q_entropy, global_step)
        writer.add_scalar("charts/probe_sps_overhead", probe_seconds, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

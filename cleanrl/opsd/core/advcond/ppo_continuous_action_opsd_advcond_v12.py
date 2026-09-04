# OPSD-AdvCond v12 -- v11 plus ONE mechanism: the all-layer residual-TD(lambda) tape.
# =====================================================================================
# WHAT THE TAPE IS. Every actor-trunk representation r^l in [entry, block_1..block_K,
# output] -- five taps at k_blocks=3, each `hidden`-wide -- owns its OWN horizon
# predictor P^l_h(r^l_t, z_t), weight-shared across the 16 horizon indices by an
# nn.Embedding. Once per iteration the trunk and the predictors are hard-snapshotted
# together and detached labels are built from the rollout:
#       Y^l_1(t) = r^l_{t+1} - r^l_t
#       Y^l_h(t) = (1-lambda) P^{l,-}_{h-1}(t+1) + lambda Y^l_{h-1}(t+1),   h > 1
# with lambda = 0.95, so deep heads keep most of their mass in the detached TD bootstrap.
# The loss is SmoothL1 in each layer's OWN detached innovation-RMS unit, valid-normalized,
# then averaged with EQUAL weight over layers and heads, so no depth can dominate merely
# by carrying a larger representation. It is supervision ON the trunk, not a model: no
# state is rolled out and no temporal autograd graph is built.
#
# THE EVIDENCE, and why it is "all-layer" rather than "tape". Measured on
# ppo_continuous_action_tpomd_alllayer_residual_td_v25 -- the best model on this disk,
# 11015 on HalfCheetah-v4 -- against the byte-identical file with the tape removed
# (Args differ only in comments):
#       @2M     @4M     @6M     end
#       +835    +536    +538    +604
# An OUTPUT-LAYER-ONLY variant of the SAME tape measured -410 @2M. So the load-bearing
# part is supervising EVERY depth, not the existence of a predictive auxiliary task.
# That is the entire reason this file taps five representations instead of one.
#
# WHY IT SHOULD COMPOSE WITH v11. v11's own attribution says the ceiling is set by the
# critic's readout and the teacher only buys rate -- and BOTH act on the shared trunk
# through its final output alone. The tape is the first lever in this lineage that
# constrains the INTERIOR of that trunk, with a signal neither the actor nor the critic
# supplies. If the early depths are underdetermined -- exactly what an output-only
# auxiliary loss cannot fix, per the -410 arm -- this is the lever that fixes them.
#
# WHAT IT CHANGES. ThinkTrunk gains forward_taps() (identical numerics, plus the
# per-depth stack); the Agent gains one independent predictor per depth; the actor
# minibatch gains one auxiliary loss delivered under its own gradient caps.
#
# WHAT IT DELIBERATELY DOES NOT CHANGE. Actor objective, teacher construction and
# snapshot, cond_scale, adv_boost, clone/distill coefficients, distill_kl_clip, epoch
# budgets, lr, minibatch count, reward normalization, critic support, sigma ratio, MTP
# horizon, deep value head, trunk width/depth/experts. `--no-tape` is v11 EXACTLY: the
# predictors are not even constructed and the task gradient path is byte-for-byte v11's.
# Nothing else from the reference is ported: no TPO, no oracle sim probe, no KL breaker,
# no dyntrust, no idxtransfer, no tail-risk telemetry.
#
# GRADIENT ROUTING, which is part of the mechanism. The tape backward runs FIRST (it
# shares the trunk subgraph), its trunk-directed gradient is clipped by ONE global 0.025
# cap and its predictor gradient by 0.25, both are stashed, the buffers are wiped, and
# THEN v11's untouched task backward + global max_grad_norm clip runs. The stashed tape
# gradient is added on top and one shared Adam step is taken. The task clip therefore
# sees exactly the norm v11 would see, and the tape never passes through max_grad_norm.
#
# FALSIFIABLE PREDICTIONS.
#   1. v12 > v11 at every matched step >= 2M, by roughly the reference's +835 @2M /
#      +604 end. A smaller positive delta means the tape is partly redundant with the
#      deep value readout; a zero delta means this chassis' trunk was not the binding
#      constraint and the reference's gain was really about ITS actor.
#   2. `debug/tape_layer0` (the ENTRY projection) is nonzero and FALLS over training. If
#      early-layer tape loss collapses to zero, or only the output layer's loss moves,
#      the mechanism did not port -- that is the exact configuration that measured -410.
#   3. If the return REGRESSES versus v11, the mechanism did not port to this chassis,
#      and the honest conclusion is that the reference's gain came from interaction with
#      TPO rather than from all-layer predictive supervision as such.
#   4. `debug/tape_trunk_gn` (the PRE-clip norm) stays above the 0.025 cap for the whole
#      run, i.e. the cap always binds, AND there is no return delta -- that would mean
#      the auxiliary is fighting the task rather than shaping the trunk.
#
# KNOWN DEVIATION FROM THE REFERENCE, stated up front: the reference bootstraps one
# frozen next-state prediction at truncations and at the artificial rollout edge; this
# file MASKS those heads out instead. Masking is strictly conservative -- it drops labels
# rather than inventing them -- and it removes the reference's separate frozen-action
# sampling path, which is TPO machinery this chassis does not have. Measured cost of the
# masking at num_steps=512: `debug/tape_valid_frac` = 0.983 on HalfCheetah (all of it the
# unavoidable rollout edge) and 0.64-0.70 on Hopper, where episodes genuinely end.
#
# Inherited v11 notes below, unchanged.
# =====================================================================================
# OPSD-AdvCond v11 -- the v6 champion with a DEEP value readout. One lever, one attribution.
# =====================================================================================
# THE DIAGNOSIS THAT MOTIVATES THIS FILE. v6 is not compared against a generic "PPO
# baseline" here; it is compared against its OWN parent chassis, iterthink_v24_beta_
# d3bucket_mtp_ppoadvnorm_batch_v1 -- identical critic support, identical ThinkTrunk,
# PPO actor. Matched-step, HalfCheetah-v4, seed 1 (scripts/score_runs.py --at):
#
#            @1M    @2M    @4M    @6M    @8M
#   parent   3079   5012   6716   7548   8278
#   v6       3826   6267   7713   8579   8812
#   delta    +24%   +25%   +15%   +14%   +6.4%
#
# The advantage-conditioned teacher is a strong ACCELERATOR whose edge DECAYS MONOTONICALLY
# to nearly nothing. Both curves converge on the same asymptote, so the asymptote is not
# set by the actor. v6 finishes at 8718 while the best model on this disk finishes at 11015
# -- and my 8718 sits essentially AT the 8455 level of the parent it accelerates.
#
# What lifts that exact chassis? A deeper value READOUT, with the actor left alone:
# dblockcritic_v1 run as its --no-dblock end-to-end control ("edmvalue_e2e_8m_v1") keeps the
# parent's PPO actor, trunk, 511-bin Dreamer3 support, sigma-ratio 0.75 and MTP-6, changes
# ONLY the readout, and scores 9810 vs 8455: +245 @2M, +1389 @3M, +1397 @4M, +1590 @6M,
# +1355 end. That is the largest actor-agnostic lever measured anywhere in this repo.
#
# THE ABLATION IS CLEAN, THE SCHEDULE IS NOT MINE. Verified from both runs' stored
# hyperparameters: edmvalue and its base agree EXACTLY on learning_rate, update_epochs,
# num_minibatches, num_bins, critic_mtp_horizon, value_sigma_to_bin_ratio,
# normalize_reward, hidden, k_blocks and n_experts -- the head is the only difference, so
# +1355 is a true single-lever delta. But that pair ran lr 3e-4 / 10 shared epochs / mb 32,
# while this chassis runs lr 1e-3 / 4 actor + 4 critic epochs / mb 128. The DIRECTION should
# transfer (the head is V(s) == 0 and exactly the identity at init, so it cannot destabilize
# what it replaces), but the MAGNITUDE is measured on a different schedule and is not owed
# to me. Prediction 1 is conditional on that, and the schedule difference is deliberately
# NOT chased here: it is tuning, not mechanism.
#
# HYPOTHESIS. Rate and ceiling are separable: the teacher buys rate, the readout buys
# ceiling, and they compose. v6's actor reaches any given return ~14% earlier than PPO on
# the shared critic, so lifting the ceiling should land above 9810, not merely at it.
#
# FALSIFIABLE PREDICTIONS, in decreasing order of how much they would teach me:
#   1. v11 > 9810 at 8M. If v11 only MATCHES edmvalue, then rate and ceiling do not
#      compose: the teacher's early edge is worth nothing once the readout is fixed, and
#      the honest conclusion is that this whole lineage's contribution is a faster route to
#      a ceiling set entirely by the critic.
#   2. v11 > v6 at EVERY matched step >= 2M. The head is V(s) == 0 at init and identity at
#      init, so it cannot hurt early; if it does, the extra capacity is fighting the
#      distillation, not the return.
#   3. v11's `losses/explained_variance` exceeds v6's before its return does. This is the
#      mechanism check: if the return improves while EV does not, the win is not the
#      readout and the attribution is wrong.
#
# WHAT IS NOT CHANGED. Actor objective, teacher construction, cond_scale=ema_rms, adv_boost,
# clone/distill coefficients, distill_kl_clip, epoch budgets, lr, minibatch count, reward
# normalization (off), critic support, sigma ratio, MTP horizon, trunk width/depth/experts.
# One lever. The diffusion regime of the source head is deliberately NOT ported: no arm
# using it exists on disk, so only its end-to-end control has evidence.
#
# IT IS NOT "MORE PARAMETERS". Measured at H=64, 511 bins, MTP 6, depth 3, mult 2: the
# depth-3 stack is 163,648 params against the single Linear's 196,224 -- 0.83x, i.e. this
# lever REMOVES 32,576 parameters. The old head paid H*mtp*num_bins for an independent
# 511-way projection per horizon; the stack shares ONE decoder across horizon tokens and
# spends the savings on depth. So any win here is depth and conditioning, not capacity,
# and "it just has more parameters" is not available as an explanation.
#
# VERIFIED BEFORE LAUNCH (not assumed): logits are exactly 0 at init so V(s) == 0 on the
# symmetric support; every block is exactly the identity at init (max|z_B - z_start| = 0);
# and the all-zero fixed point is real -- with z_start forced to 0 the total gradient norm
# is exactly 0.000e+00, against 3.75e-02 with the learned nonzero start.
#
# Inherited v6 notes below, unchanged.
# =====================================================================================
# ==================== RESULT: THIS FILE BEATS PPO ON THE 8M BENCHMARK ================
# HalfCheetah-v4, seed 1, `--cond-scale ema_rms --adv-boost 1 --actor-epochs 4
# --critic-epochs 4 --num-minibatches 128`. Windowed means (+-50k), not point reads:
#
#   arm                @500k    @1M    @2M    @4M    @6M    @8M
#   PPO baseline        1576   3079   5012   6716      -    8278
#   THIS (kappa=1)       994   3826   6267   7680   8598    8819   <- +6.5% over PPO
#   kappa=2             1524   3993   6084   7848   8486    8249
#
# Ahead of PPO from 1M on, and past PPO's FINAL 8M score by 6M. `score_runs.py` last-20
# reads 8718.4 +-497.7. The whole margin came from one variable: how the advantage is
# scaled before it is used as a conditioning input (`cond_scale`, the v6 change). The
# batch-standardized predecessor scored 4390 @2M against this file's 6267.
#
# --------------------------- ABLATIONS AROUND THE CHAMPION ---------------------------
# All HalfCheetah-v4, seed 1, on this chassis. Three of my predictions were falsified.
#
# 1. cond_scale (the v6 hypothesis) -- CONFIRMED, and it is the whole result.
#      ema_rms (scale only, zero preserved)   664 @500k   2962 @1M   5365 @2M
#      batch   (PPO-style standardization)    354        2334       4390
#      raw     (no scale tracking)              4         525       2368
#    `raw` fails exactly as v4's RMS measurement predicted: the residual's RMS reaches
#    ~14.7, so the +-3 conditioning clamp saturates and the channel stops resolving.
#
# 2. Reward normalization -- a large EARLY win that COLLAPSES late. 1577 @500k (vs 664)
#    but 3539 @1M and 5177 @2M, ending below the plain arm. It is off by default.
#
# 3. Margin kappa (`adv_boost`) -- optimum is HORIZON-DEPENDENT, which I did not expect.
#      kappa    @500k    @1M    @2M         kappa=2 leads through 4M (7848 vs 7680) and
#        1        664   2962   5365         then LOSES at 8M (8249 vs 8819). A bigger
#        2        924   3693   6077         margin buys early speed and spends late
#        3       1181   3691   5624         stability; the LR anneal appears to interact.
#        4       1600   4255   3128  <- fastest @1M in the lineage, then destabilized
#    FALSIFIED: annealing kappa 4->1 and 4->2 over 2M did NOT beat constant kappa=2
#    (4415 and 5864 @2M). Speed and stability here are not separable by a linear schedule.
#
# 4. clone_coef -- a NON-MONOTONE optimum at 1.0. Both directions lose:
#      0.5 -> 4636 @2M     1.0 -> 5365 @2M     2.0 -> 4340 @2M
#    FALSIFIED my "more sharpening is strictly better" reading of finding 5.
#
# 5. Removing the clone term's sharpening channel while keeping its calibration (v9) is
#    WORSE (4136 @2M) despite doing exactly what it was designed to do -- less entropy
#    collapse and larger improvement steps. The champion collapses to H=-13.0 and wins.
#    On HalfCheetah the sharpening is a FEATURE, not the pathology I diagnosed it as.
#
# 6. Replacing the clone term with an InfoNCE identifiability objective (v8) is
#    catastrophic (-34 @2M) even though the channel became MORE identifiable. Ordering
#    the deltas is not the same as calibrating them; the improvement operator needs
#    E[a|s,delta] to be the true posterior mean, not merely a monotone map of it.
#
# Open: only seed 1 is measured, so sub-400 differences are noise, and Hopper/Walker2d
# transfer is queued but unmeasured.
# =====================================================================================
# OPSD-AdvCond v6 -- v5's split actor/critic budgets PLUS an honest conditioning scale.
# No PPO anywhere (no ratio, no clipped surrogate, no advantage-weighted policy gradient).
# =====================================================================================
# v6 = v5 exactly, plus `cond_scale`. Setting cond_scale="batch" reproduces v5 bit for
# bit, so the ema_rms arm is a strict single-variable comparison against v5's a1c4_mb128.
#
# The claim: delta is an INPUT to a learned conditional, not a multiplier on a gradient,
# so PPO's batch mean/sd normalization is the wrong tool. Mean subtraction destroys
# delta's natural zero (delta=0 means "as V expected") and the batch sd makes adv_boost's
# units non-stationary. Dividing by a slow RMS keeps the zero and keeps the Fourier
# features in range. v4 measured the raw residual's RMS growing 0.61 -> 2.17 over four
# iterations, so some scale tracking is required; the question this file answers is
# whether the honest one beats the convenient one on return.
# =====================================================================================
# INHERITED v5 EVIDENCE AND NOTES (unchanged below)
# =====================================================================================
# DURABLE EVIDENCE (HalfCheetah-v4, seed 1, 1M steps, 100k-window means -- NOT point
# reads, which invert the ranking: mb128_e4 reads worse than v2_b10 at 500k and ends 700
# ahead). Rightmost column is the return slope over the final third, per 1M steps.
#   arm                        @600k  @800k   @1M   slope/1M
#   v3 mb128 e4                 1456   1870   2908    +3754   <- champion
#   v2_b10  mb32 e4             1426   1865   2176    +1913
#   v3 mb128 e1                  514   1202   1648    +2996
#   v1_q10  mb32 e4 (const q)     494   660    741     +431
#   v3 mb32  e1                     1    75    123     +352
#   PPO baseline on this chassis: 1576 @500k, 3079 @1M.
#
# WHAT THAT ACTUALLY SAYS
#   1. GRADIENT-STEP COUNT per iteration is the dominant lever, not "epochs". mb32 e1 (32
#      steps/iter) is inert; mb128 e1 (128 steps/iter) reaches a HIGHER slope (+2996) than
#      the mb32 e4 champion (+1913). My earlier reading -- "the paper's single pass does
#      not port" -- was wrong: it was the step count, not the single pass.
#   2. But e4 still beat e1 at matched step count (mb128: 2908 vs 1648). That comparison
#      is CONFOUNDED: e4 also handed the critic 4x its regression passes.
#
# THE v5 HYPOTHESIS
#   The two kinds of reuse are different in kind:
#     actor reuse re-fits the SAME sampled action for the same state. It cannot add
#       information; it only sharpens the conditional, and it is paid for in entropy.
#     critic reuse is ordinary supervised regression onto FIXED bootstrap targets, where
#       extra passes just reduce fitting error.
#   So give them separate budgets: actor_epochs=1, critic_epochs=4.
#
# MEASURED BEFORE COMMITTING GPU (this file, 196608 steps, mb=128, end-of-run values)
#   cfg     EV      entropy   value_loss   distill_kl
#   a1c4    0.501   -0.566      20.47        0.046
#   a4c4    0.600   -0.913      24.91        0.177     (EV noisy: 0.33 -> 0.60)
#   a1c8    0.455   -0.733      20.48        0.048
#   Actor reuse costs 60% more entropy collapse AND leaves the critic WORSE (value_loss
#   24.9 vs 20.5): with a shared trunk, four passes of actor gradient fight the value
#   regression. c8 overshoots, so 4 is the sweet spot at this batch size.
#
# Epochs past actor_epochs take a critic-only path: one zeroed-context forward, no policy
# heads, about half the compute of a full step.
# =====================================================================================
# INHERITED v3/v4 NOTES (unchanged below)
# =====================================================================================
# v1/v2 inherited PPO's update ritual (4 epochs of reuse, teacher recomputed inside every
# minibatch). Re-reading the reference shows both are wrong for this algorithm.
#
# WHAT OPSD ACTUALLY DOES (arXiv 2601.18734)
#   Algorithm 1:  while not converged: sample a minibatch B; sample on-policy responses;
#                 compute the token-wise divergence; take ONE gradient step.
#                 -> minibatched, single-pass, NO data reuse. Their whole run is ~100
#                    gradient update steps (Fig. 3 x-axis).
#   Sec. 5 impl.: "We fix the teacher policy to be the initial policy, rather than the
#                 currently updating learning policy, as we find this helps stabilize
#                 training and implicitly acts as regularization."
#
# WHAT PORTS AND WHAT DOES NOT
#   Single-pass minibatching PORTS EXACTLY -> update_epochs = 1 by default.
#     Minibatches here are NOT a trust-region device (there is no ratio to keep valid).
#     They are how one rollout becomes many gradient steps with zero reuse: 32 minibatches
#     = 32 steps on data no step has seen. Full-batch ("no minibatches") would give 1 step
#     per rollout = 244 steps for an entire 8M run.
#   Init-frozen teacher DOES NOT PORT. Their teacher is a pretrained LLM that can already
#     rationalize; ours is randomly initialized, so anchoring to init distills noise
#     forever. The portable half is the staleness -> snapshot the teacher once per
#     iteration, not once per minibatch.
#
# WHY THE STALENESS MATTERS MECHANICALLY
#   v2's teacher was pi(.|s, delta+kappa) evaluated with the CURRENT weights inside each
#   minibatch, so the target moved because the student moved: the student chases a teacher
#   that is itself being dragged by the student's own updates. Snapshotting makes both
#   terms honest fixed regression targets and removes one of three forwards per step.
#
# COMPUTE CONTEXT (measured, this chassis, 16 envs x 2048 steps)
#   rollout 7.6 s/iter vs 0.28 s per update epoch -> the loop is ~96% env-bound, 26% GPU.
#   So gradient steps are nearly free, but the way to spend that is MORE MINIBATCHES
#   (fresh-data steps), not more epochs (reused-data steps).
# =====================================================================================
# INHERITED v2 EVIDENCE (unchanged below)
# =====================================================================================
# WHAT v1 MEASURED (HalfCheetah-v4, seed 1, matched 393216 steps, identical chassis)
#   adv_query   return@393k   cond_gap   entropy@start
#   1.0            179          0.008        -0.88
#   2.0            100          0.010        -1.47
#   3.0            -49          0.012        -2.66
# v1 queried the teacher at a GLOBAL CONSTANT delta for every state. Tripling that
# constant moved the teacher/student gap only 0.008 -> 0.012 while entropy started 1.8
# nats lower and return fell monotonically. So the dial had low authority AND a large
# cost, which is the signature of querying off the fitted support rather than of a step
# size being too large.
#
# WHY A GLOBAL CONSTANT IS THE WRONG PARAMETERIZATION
# The conditional p(a | s, delta) is fit on the realized residuals, which are
# standardized per batch. A state whose action actually scored delta_t = -2 has its local
# conditional supported around -2; asking it for an absolute +1 is a 3-sigma extrapolation
# in a region where that state contributed no training mass. Worse, the number of samples
# supporting an absolute query falls off like the Gaussian tail, so the query gets less
# reliable exactly as it gets more ambitious.
#
# v2 asks every state to beat ITSELF by a fixed margin:
#     query_t = clamp(delta_t + adv_boost, -adv_cond_clip, +adv_cond_clip)
# Every query now sits adv_boost away from an actually observed point, so it is
# interpolation almost everywhere instead of tail extrapolation. The improvement request
# is uniform in SIGNAL units rather than uniform in absolute position, and it is closer to
# the paper, where the teacher conditions on the real privileged information rather than a
# synthetic constant.
#
# Hypothesis: at matched steps v2 beats v1's best (179@393k) and, unlike v1, return
# improves rather than degrades as the margin grows, because the margin no longer trades
# against support. Kill-tell: debug/query_clip_frac large means adv_boost is pushing the
# distribution into the clamp and v2 has degenerated back into a constant query.
# =====================================================================================
# OPSD-AdvCond v1 method (retained below, unchanged)
# -------------------------------------------------------------------------------------
# =====================================================================================
# THE PAPER (arXiv 2601.18734, On-Policy Self-Distillation). One LLM instantiates both
# policies under different conditioning: p_S(.|x) sees only the prompt, p_T(.|x,y*) also
# sees the ground-truth solution y*. The student rolls out, both policies re-read the
# student's own tokens, and the loss is a per-token distribution divergence D(p_T||p_S)
# with gradients flowing ONLY through the student. The teacher is not a bigger network,
# it is the same weights handed privileged information it can rationalize.
#
# THE RL TRANSLATION USED HERE
#   "token"            -> one action dimension (per-dim divergence, per-dim clipping)
#   "privileged y*"    -> THE REALIZED ADVANTAGE of the action actually taken, which the
#                         acting policy provably could not see at action time
#   "rationalization"  -> supervised conditional density fit p(a | s, A)
#   "student context"  -> the same input with the privileged slot ZEROED
#
# So: roll out, record what it did and how well it scored, then go over every step again.
# In the teacher context the network is told the advantage its own action earned and is
# fit to that action; in the student context that slot is zero. Querying the same weights
# at an OPTIMISTIC advantage asks "what would I do if this were a k-sigma better action?"
# and that answer is distilled back into the unconditioned policy. It bootstraps itself.
#
# WHY THE ACTION IS NOT AN INPUT (the one non-obvious design constraint)
# It is tempting to feed (s, a, A) since the model "sees what it did". That is degenerate:
# the rationalization loss is -log p(a | s, ...), so if a appears on BOTH sides the fit
# collapses to the identity map (a Dirac at a), driving the loss to -inf while teaching
# the network nothing about improvement. The teacher would then be exactly the rollout
# policy and the distillation term would be identically zero. So "what it did" enters as
# the regression TARGET and "how well it scored" enters as the conditioning INPUT.
#
# TWO OBJECTIVES (no ratio, no clipped surrogate, no policy gradient)
#   1. Rationalization:  L_clone = -log pi(a_t | s_t, A_t)      [teacher context]
#      A dense, low-variance supervised regression. It makes the privileged channel mean
#      something, and it anchors the conditional to on-policy data, which doubles as a
#      natural trust region and resists entropy collapse.
#   2. Self-distillation: L_distill = sum_d min( KL(pi_T,d || pi_S,d), tau )
#      pi_T = pi(.|s, delta_t + adv_boost) DETACHED, pi_S = pi(.|s, absent). Gradients
#      student context. This is the paper's per-token clipped divergence, verbatim in form.
#
# CRITIC. Stays V(s): no action input, evaluated ONLY with the privileged slot zeroed.
# The advantage is derived from V, so letting V see it would be circular; zeroing keeps
# the value head an honest state-value function and keeps its bias low. HL-Gauss
# multi-token-prediction critic and GAE are inherited unchanged from the 8278@8M chassis.
#
# HYPOTHESIS. The improvement operator is a conditional-density query rather than a
# gradient ascent step, so it inherits the variance of supervised learning, not of policy
# gradients, while remaining strictly on-policy. Advantage conditioning is a scalar, so
# the fit is well posed even when the per-dim advantage/action correlation is weak.
#
# KILL-TELLS (all logged)
#   debug/cond_gap ~ 0        -> the trunk is ignoring the privileged slot; the teacher is
#                                the student, the method is inert. Raise adv_boost or the
#                                clone weight.
#   losses/distill_kl ~ 0 with flat returns -> same failure, seen from the loss side.
#   debug/query_clip_frac high -> the margin is saturating the clamp; v2 has collapsed
#                                back into v1's constant query.
#   entropy collapsing while cond_gap grows -> adv_boost is outrunning the fit.
# =====================================================================================
import copy
import math
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 1e-3
    num_envs: int = 16
    num_steps: int = 2048
    # --- throughput. Neither flag changes the algorithm: identical math, identical
    # per-env seeding. Measured on this chassis: the acting forward is LAUNCH-bound, not
    # compute-bound (eager act = 1149 us at batch 16, 1277 us at batch 256 -- flat), and it
    # dominated the 5.8 ms vec-step while env stepping was only 0.99 ms of it.
    #   compile mode=reduce-overhead: 1149 -> 347 us at batch 16.
    #   AsyncVectorEnv: raw env stepping 16131 -> 40357 samples/s.
    #   marginal end-to-end: 3025 -> 4300 SPS at 16 envs; 6013 at 128 envs.
    # NOTE the 8819 @8M champion was measured EAGER; arms after this patch are compiled, so
    # numerics are not bit-identical (well inside the +-498 CI95 on that measurement).
    async_envs: bool = True
    compile_act: bool = True
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    actor_epochs: int = 1         # passes over the batch for the rationalization + distill
    #                               losses. Reuse here re-fits the SAME action per state, so
    #                               it sharpens the conditional (entropy drops).
    critic_epochs: int = 4        # passes for the value regression only. Reuse here is plain
    #                               supervised learning on fixed targets and is nearly free.

    # --- OPSD advantage conditioning ---
    adv_boost: float = 1.0        # margin, in cond-scale units, each state must beat ITSELF by
    adv_boost_final: float = -1.0 # if >= 0, linearly anneal adv_boost -> this over training.
                                  # Negative (default) = constant margin, i.e. bit-for-bit
                                  # the previous behaviour.
                                  # Motivation, measured at 2M on this chassis:
                                  #   kappa   @500k    @1M     @2M
                                  #     1       664    2962    5365
                                  #     2       924    3693    6077   <- stable optimum
                                  #     3      1181    3691    5624
                                  #     4      1600    4255    3128   <- fastest, then broke
                                  # A big margin buys the fastest early progress in the
                                  # lineage and then destabilizes; a small one is stable but
                                  # slow. The margin is a step size, so decay it.
    adv_cond_clip: float = 3.0    # clamp on the scaled advantage used as conditioning
    cond_scale: str = "ema_rms"   # "ema_rms" | "batch" | "raw"; see the scaling block below
    cond_ema_beta: float = 0.99   # EMA horizon for the RMS scale (~100 iterations)
    adv_embed_freqs: int = 8      # sinusoidal features per phase; privileged block is 2x this
    cond_lambda: float = 0.0      # GAE lambda for the PRIVILEGED channel only; 0 = 1-step
                                  # TD residual, the most action-attributable credit signal
    clone_coef: float = 1.0       # weight on the rationalization fit p(a | s, A)
    distill_coef: float = 1.0     # weight on the per-dim teacher->student divergence
    distill_kl_clip: float = 2.0  # tau: the paper's per-token pointwise divergence clip

    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    # --- v11: deep value readout (replaces one zero-init Linear) ---
    value_blocks: int = 3         # stack depth. 3 is the measured setting (+1355 end)
    value_mult: int = 2           # block MLP width = value_mult * hidden

    normalize_reward: bool = False
    clip_reward: bool = False

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # --- v12: the all-layer residual-TD(lambda) tape (the one new mechanism) ---
    # Defaults are the reference's, unchanged.
    tape: bool = True                       # --no-tape reproduces v11 exactly
    tape_horizon: int = 16
    tape_lambda: float = 0.95
    tape_horizon_embed_dim: int = 16
    tape_predictor_hidden: int = 64
    tape_coef: float = 1.0
    tape_scale_floor_ratio: float = 1e-3
    tape_trunk_grad_clip: float = 0.025     # ONE global cap on the tape's trunk gradient
    tape_predictor_grad_clip: float = 0.25  # looser cap on the predictors' own parameters
    tape_target_batch_size: int = 8192      # label-build chunk; bounds peak memory

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
        env = gym.wrappers.TransformObservation(  # pyright: ignore[reportCallIssue]
            env, lambda observation: np.asarray(observation).clip(-10, 10)
        )
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(
                env, lambda reward: min(max(float(reward), -10.0), 10.0)
            )
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def beta_kl_per_dim(a1, b1, a2, b2):
    """Forward KL( Beta(a1,b1) || Beta(a2,b2) ) per action dim, closed form."""

    def ln_beta_fn(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    return (
        ln_beta_fn(a2, b2)
        - ln_beta_fn(a1, b1)
        + (a1 - a2) * torch.digamma(a1)
        + (b1 - b2) * torch.digamma(b1)
        + (a2 - a1 + b2 - b1) * torch.digamma(a1 + b1)
    )


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
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
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)
        g = torch.sigmoid(self.resid_gate)
        x_in = g * x + (1.0 - g) * x0
        d_dense = self.dense(self.dense_norm(x_in))
        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)
        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            self.blocks.append(ThinkBlock(H * (k + 1), H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))

    def forward_taps(self, x):
        """forward(x), plus the representation at EVERY depth for the v12 tape.

        Returns (output, taps) with taps stacked as [B, L, H], L = K + 2 in depth order:
            tap 0      -- x0, the entry projection                       (H wide)
            tap 1..K   -- feats[k], the output of ThinkBlock k           (H wide)
            tap K+1    -- out_proj(out_norm(cat(feats))), the final output (H wide)
        Every tap is genuinely `hidden`-wide and nothing is projected to make it fit: the
        DenseNet concatenation cat(feats) is only ever an INPUT to the next block, never a
        representation in its own right, so it is not a tap. The ops below are the same
        ops in the same order as forward(), so forward_taps(x)[0] == forward(x) exactly.
        """
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        out = self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))
        return out, torch.stack((*feats, out), dim=-2)


class HorizonTapePredictor(nn.Module):
    """One weight-shared network evaluated at every explicit horizon index.

    The horizon is an INPUT (an embedding lookup), not a separate head per horizon, so all
    16 heads share one body and a head cannot be learned in isolation from its neighbours.
    `output` is zero-init in BOTH weight and bias, so the whole tape predicts exactly 0 at
    init and adds no init shock to the trunk.
    """

    def __init__(self, feature_dim, action_dim, horizon, embed_dim, hidden_dim):
        super().__init__()
        if min(feature_dim, action_dim, horizon, embed_dim, hidden_dim) < 1:
            raise ValueError("tape predictor dimensions must be positive")
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.horizon_embedding = nn.Embedding(horizon, embed_dim)
        self.source = layer_init(nn.Linear(feature_dim + action_dim, hidden_dim))
        self.conditioned = layer_init(nn.Linear(hidden_dim + embed_dim, hidden_dim))
        self.output = layer_init(nn.Linear(hidden_dim, feature_dim), std=0.01)
        self.activation = ReLUSquared()
        self.horizon_indices: torch.Tensor
        self.register_buffer("horizon_indices", torch.arange(horizon), persistent=False)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()

    def forward(self, actor_feature, action):
        if actor_feature.ndim != 2 or action.ndim != 2:
            raise ValueError("actor feature and action must be rank-two batches")
        if actor_feature.shape[0] != action.shape[0]:
            raise ValueError("actor feature and action batch sizes differ")
        base = self.activation(self.source(torch.cat((actor_feature, action), dim=-1)))
        horizons = self.horizon_embedding(self.horizon_indices)
        base = base.unsqueeze(1).expand(-1, self.horizon, -1)
        horizons = horizons.unsqueeze(0).expand(base.shape[0], -1, -1)
        conditioned = self.activation(self.conditioned(torch.cat((base, horizons), dim=-1)))
        return self.output(conditioned)


class AllLayerHorizonTapePredictor(nn.Module):
    """Independent horizon-conditioned residual predictor at every trunk depth.

    Independence is the measured lever: sharing one predictor across depths, or tapping
    only the output, is the arm that scored -410 @2M in the reference lineage.
    """

    def __init__(self, num_layers, feature_dim, action_dim, horizon, embed_dim, hidden_dim):
        super().__init__()
        if num_layers < 1:
            raise ValueError("the all-layer predictor needs at least one layer")
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.predictors = nn.ModuleList(
            [
                HorizonTapePredictor(feature_dim, action_dim, horizon, embed_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, actor_features, action):
        if actor_features.ndim != 3 or action.ndim != 2:
            raise ValueError("all-layer features and actions must be [B,L,D] and [B,A]")
        if actor_features.shape[0] != action.shape[0]:
            raise ValueError("all-layer feature and action batch sizes differ")
        if actor_features.shape[1:] != (self.num_layers, self.feature_dim):
            raise ValueError("all-layer feature channels do not match the predictor")
        predictions = [
            predictor(actor_features[:, layer], action)
            for layer, predictor in enumerate(self.predictors)
        ]
        return torch.stack(predictions, dim=1)


def build_all_layer_horizon_td_tape_targets(
    innovations, frozen_predictions, step_ok, tape_lambda
):
    """Detached finite-horizon residual TD(lambda) labels, one recursion per depth.

        Y^l_1(t) = r^l_{t+1} - r^l_t
        Y^l_h(t) = (1 - lambda) P^{l,-}_{h-1}(t+1) + lambda Y^l_{h-1}(t+1)     (h > 1)

    `innovations` is [T,N,L,D]; `frozen_predictions` is the FROZEN snapshot tape P^{l,-}
    at the stored (s_t, z_t) as [T,N,L,H,D]; `step_ok` is [T,N], True when transition t
    stayed inside its episode. Head h at time t is valid only when the h transitions
    t .. t+h-1 are ALL inside one episode and row t+h exists in the rollout window, so no
    label ever reads across an episode boundary. Invalid entries are written as exact
    zeros and masked out of the loss, so they can neither contribute to it nor propagate
    into a longer head. This is a horizon-index dynamic program over detached tensors: no
    state model is rolled out and no temporal autograd graph is created.
    """
    if innovations.ndim != 4 or frozen_predictions.ndim != 5:
        raise ValueError("innovations and predictions must be [T,N,L,D] and [T,N,L,H,D]")
    if innovations.shape[:3] != frozen_predictions.shape[:3]:
        raise ValueError("innovation and prediction rows do not match")
    if innovations.shape[-1] != frozen_predictions.shape[-1]:
        raise ValueError("innovation and prediction latent dimensions do not match")
    if step_ok.shape != innovations.shape[:2]:
        raise ValueError("step_ok must carry one flag per rollout transition")
    if not 0.0 <= tape_lambda <= 1.0:
        raise ValueError("tape_lambda must be in [0, 1]")
    time_dim, num_envs = innovations.shape[:2]
    horizon = frozen_predictions.shape[-2]
    if horizon < 1 or horizon >= time_dim:
        raise ValueError("the tape horizon must be positive and shorter than the rollout")
    innovations = innovations.detach()
    frozen_predictions = frozen_predictions.detach()
    step_ok = step_ok.detach().bool()

    # masks[t, h-1] == the head-h validity chain. Layer-independent: every depth is read
    # off the same trajectory, so the boundary structure is shared.
    masks = torch.zeros(
        (time_dim, num_envs, horizon), dtype=torch.bool, device=innovations.device
    )
    masks[:-1, :, 0] = step_ok[:-1]
    for head in range(1, horizon):
        masks[:-1, :, head] = step_ok[:-1] & masks[1:, :, head - 1]

    values = torch.zeros_like(frozen_predictions)
    values[..., 0, :] = torch.where(
        masks[:, :, 0].unsqueeze(-1).unsqueeze(-1),
        innovations,
        torch.zeros_like(innovations),
    )
    lam = float(tape_lambda)
    for head in range(1, horizon):
        # masks[t, head] implies masks[t+1, head-1], so the bootstrapped tail read here is
        # always a valid label rather than a masked-out zero.
        mixed = (1.0 - lam) * frozen_predictions[1:, :, :, head - 1, :] + lam * values[
            1:, :, :, head - 1, :
        ]
        values[:-1, :, :, head, :] = torch.where(
            masks[:-1, :, head].unsqueeze(-1).unsqueeze(-1),
            mixed,
            torch.zeros_like(mixed),
        )
    return values.detach(), masks


def horizon_td_tape_loss(
    prediction, target, mask, *, scale=None, latent_rms=None, scale_floor_ratio=1e-3
):
    """Equal-average valid-normalized layer/head SmoothL1 in per-layer units.

    `prediction`/`target` are [..., L, H, D] and `mask` is [..., L, H]. Each layer is
    divided by its OWN DETACHED innovation RMS -- the RMS of Y^l_1 over valid entries,
    floored at `scale_floor_ratio` times that layer's latent RMS -- so a depth whose
    representation happens to be large in absolute units cannot dominate the average, and
    a depth whose innovations collapse cannot blow it up. Layers and heads are then
    averaged with EQUAL weight. Returns (scalar, per-layer scale, per-layer-head detail).
    """
    if prediction.shape != target.shape or prediction.shape[:-1] != mask.shape:
        raise ValueError("prediction, target, and mask shapes are inconsistent")
    if target.ndim < 4 or target.shape[-3] < 1 or target.shape[-2] < 1:
        raise ValueError("target needs nonempty layer, horizon, and latent dimensions")
    if scale_floor_ratio <= 0.0:
        raise ValueError("scale_floor_ratio must be positive")
    num_layers = target.shape[-3]
    sample_dims = tuple(range(target.ndim - 3))
    scale_shape = (1,) * len(sample_dims) + (num_layers, 1, 1)
    weights = mask.to(target.dtype).unsqueeze(-1)
    if scale is None:
        innovation_mask = mask[..., 0].to(target.dtype).unsqueeze(-1)
        safe_innovation = torch.where(
            mask[..., 0].bool().unsqueeze(-1),
            target.detach()[..., 0, :],
            torch.zeros_like(target[..., 0, :]),
        )
        innovation_denominator = (
            innovation_mask.sum(dim=sample_dims + (innovation_mask.ndim - 1,))
            * target.shape[-1]
        ).clamp_min(1.0)
        innovation_rms = (
            safe_innovation.square()
            .mul(innovation_mask)
            .sum(dim=sample_dims + (safe_innovation.ndim - 1,))
            / innovation_denominator
        ).sqrt()
        if latent_rms is None:
            floor = torch.full_like(innovation_rms, torch.finfo(target.dtype).eps)
        else:
            if latent_rms.shape != (num_layers,):
                raise ValueError("latent_rms must contain one scalar per layer")
            floor = latent_rms.detach().to(target) * scale_floor_ratio
        scale = innovation_rms.clamp_min(floor.clamp_min(torch.finfo(target.dtype).eps))
    else:
        if scale.shape != (num_layers,):
            raise ValueError("scale must contain one detached scalar per layer")
        scale = scale.detach().to(target).clamp_min(torch.finfo(target.dtype).eps)
    valid = mask.bool().unsqueeze(-1)
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    normalized_prediction = safe_prediction / scale.reshape(scale_shape)
    normalized_target = safe_target / scale.reshape(scale_shape)
    channel_error = F.smooth_l1_loss(
        normalized_prediction, normalized_target, reduction="none"
    )
    denominator = (
        mask.to(target.dtype).sum(dim=sample_dims) * target.shape[-1]
    ).clamp_min(1.0)
    per_layer_head = (
        channel_error.mul(weights).sum(dim=sample_dims + (channel_error.ndim - 1,))
        / denominator
    )
    return per_layer_head.mean(), scale.detach(), per_layer_head.detach()


class AdvEmbed(nn.Module):
    """Fixed Fourier features of the scalar advantage.

    A single raw advantage channel among 17 observation dims is trivially IGNORABLE: the
    rationalization loss can be driven down almost entirely by modelling the marginal
    p(a|s) and dropping A, because the extra likelihood A buys is small. Measured on
    HalfCheetah with raw scalar conditioning: cond_gap 0.001 and distill_kl 0.0001, i.e.
    the teacher WAS the student and the method was a no-op. Sinusoidal features of the
    scalar fix this the way diffusion timestep embeddings do -- the advantage now occupies
    many channels and separates nearby values at high frequency, so it is both easy to use
    and expensive to ignore. Frequencies are fixed, not learned, so the channel cannot be
    switched off by driving weights to zero.
    """

    def __init__(self, n_freq):
        super().__init__()
        self.n_freq = n_freq
        self.freqs: torch.Tensor
        self.register_buffer("freqs", torch.logspace(-1.0, 3.0, n_freq, base=2.0))

    @property
    def dim(self):
        return 2 * self.n_freq

    def forward(self, adv):
        x = adv * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


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
    """One network, two contexts. The privileged block is the TRAILING input channels.

    Present  -> Fourier features of the realized advantage (teacher context).
    Absent   -> that whole block is zeroed (student context, and always for the critic).
    Zeroing rather than feeding A=0 keeps "no privileged information" a distinct code,
    since cos(0)=1 means the embedding of zero is not the zero vector.
    """

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.adv_embed = AdvEmbed(args.adv_embed_freqs)
        self.cond_dim = self.adv_embed.dim
        self.trunk = ThinkTrunk(obs_dim + self.cond_dim, H, args.k_blocks, args.n_experts)
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        # v11: the single zero-init Linear(H, mtp*num_bins) becomes a depth-B adaLN
        # residual stack over the mtp horizon TOKENS. Same support, same trunk feature,
        # same V(s) == 0 at init; only the readout capacity changes.
        self.critic_head = DeepValueHead(
            H, args.num_bins, args.critic_mtp_horizon, args.value_blocks, args.value_mult
        )
        self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.action_low: torch.Tensor
        self.action_high: torch.Tensor
        self.register_buffer(
            "action_low",
            torch.as_tensor(envs.single_action_space.low, dtype=torch.float32).reshape(-1),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(envs.single_action_space.high, dtype=torch.float32).reshape(-1),
        )
        # --- v12: one INDEPENDENT horizon predictor per trunk depth -------------------
        # Constructed only when the tape is on, so --no-tape leaves the parameter set, the
        # optimizer's parameter list and the init RNG stream exactly as v11 left them.
        self.tape_enabled = args.tape
        self.tape_num_layers = args.k_blocks + 2
        self.tape_feature_dim = H
        self.tape_action_dim = act_dim
        if args.tape:
            self.tape_predictor = AllLayerHorizonTapePredictor(
                self.tape_num_layers,
                H,
                act_dim,
                args.tape_horizon,
                args.tape_horizon_embed_dim,
                args.tape_predictor_hidden,
            )

    def _feat(self, obs, cond):
        return self.trunk(torch.cat([obs, cond], dim=-1))

    def cond_present(self, adv):
        """Privileged context: Fourier features of the standardized advantage."""
        return self.adv_embed(adv)

    def _zero_cond(self, obs):
        """Privileged context ABSENT: the whole block is zero."""
        return obs.new_zeros((obs.shape[0], self.cond_dim))

    def policy(self, obs, cond):
        feat = self._feat(obs, cond)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        return alpha, beta

    def policy_and_value(self, obs, cond):
        feat = self._feat(obs, cond)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        value_logits = self.critic_head(feat)      # (B, mtp, num_bins), reshape is internal
        return alpha, beta, value_logits

    def policy_and_value_taps(self, obs, cond):
        """policy_and_value(), plus the all-depth trunk taps the v12 tape predicts from.

        The three task tensors are produced by exactly the ops policy_and_value uses, in
        the same order, so this is a numerically identical substitute for it.
        """
        feat, taps = self.trunk.forward_taps(torch.cat([obs, cond], dim=-1))
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        value_logits = self.critic_head(feat)      # (B, mtp, num_bins), reshape is internal
        return alpha, beta, value_logits, taps

    def tape_trunk_parameters(self):
        """The trunk parameters the tape gradient reaches, as ONE global clip group."""
        return list(self.trunk.parameters()) if self.tape_enabled else []

    def tape_predictor_parameters(self):
        return list(self.tape_predictor.parameters()) if self.tape_enabled else []

    def act(self, obs):
        """Acting policy == the STUDENT context (privileged slot zeroed)."""
        alpha, beta, value_logits = self.policy_and_value(obs, self._zero_cond(obs))
        distribution = Beta(alpha, beta, validate_args=False)
        z = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        return action, z, value_logits

    def get_value(self, obs):
        """V(s): no action, privileged slot always zeroed."""
        feat = self._feat(obs, self._zero_cond(obs))
        return self.critic_head(feat)              # (B, mtp, num_bins), reshape is internal


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    assert args.batch_size % args.num_minibatches == 0
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.actor_epochs > 0 and args.critic_epochs > 0
    assert args.adv_boost > 0.0, "a non-positive margin makes the teacher no better"
    if args.tape:
        assert args.tape_horizon >= 1
        assert 0.0 <= args.tape_lambda <= 1.0
        assert args.tape_horizon_embed_dim >= 1 and args.tape_predictor_hidden >= 1
        assert args.tape_target_batch_size >= 1
        assert args.tape_coef >= 0.0 and args.tape_scale_floor_ratio > 0.0
        assert args.tape_trunk_grad_clip >= 0.0 and args.tape_predictor_grad_clip >= 0.0
        assert args.num_steps > args.tape_horizon, "otherwise no tape head can be valid"

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
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    vector_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vector_cls(
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
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs, args).to(device)
    # Rollout forward only. The update stays eager: it is a small share of wall clock
    # (512 minibatch steps per iteration against 2048 acting steps), and graphing it would
    # complicate the dual/telemetry paths for little gain.
    act_fn = (
        torch.compile(agent.act, mode="reduce-overhead") if args.compile_act else agent.act
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, device
    )
    # The tape's two gradient groups. Both are inside agent.parameters(), so both take
    # their step from the SAME Adam; only their clipping is separate.
    tape_trunk_params = agent.tape_trunk_parameters()
    tape_predictor_params = agent.tape_predictor_parameters()
    tape_all_params = tape_trunk_params + tape_predictor_params
    tape_num_layers = agent.tape_num_layers

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    assert obs_shape is not None and action_shape is not None
    act_dim = int(np.prod(action_shape))

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)

    # Slow RMS of the raw conditioning residual. Not a gradient statistic -- it exists only
    # to keep the Fourier features inside the range their fixed frequencies resolve.
    cond_ms = torch.zeros((), device=device)
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                action, z, value_logits = act_fn(next_obs)
                values[step] = hl_support.to_scalar(value_logits[:, 0])
            latent_zs[step] = z

            env_action = action.reshape((args.num_envs,) + action_shape)
            next_obs_np, reward, terminations, truncations, infos = envs.step(
                env_action.cpu().numpy()
            )
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [item is not None for item in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32)
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(
                transition_valid, device=device, dtype=torch.float32
            )
            next_obses[step] = torch.as_tensor(
                transition_next_obs, device=device, dtype=torch.float32
            )
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # ================= GAE on V(s), then the MTP critic targets =====================
        with torch.no_grad():
            next_value_logits = agent.get_value(next_obses.reshape((-1,) + obs_shape))[:, 0]
            next_values = hl_support.to_scalar(next_value_logits).reshape(
                args.num_steps, args.num_envs
            )
            # Two credit signals from the SAME V(s). GAE(gae_lambda) for critic targets,
            # and GAE(cond_lambda) for the privileged channel.
            advantages = torch.zeros_like(rewards)
            cond_adv = torch.zeros_like(rewards)
            last_gae = torch.zeros(args.num_envs, device=device)
            last_cond = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_values[t] * bootstrap_nonterminal - values[t]
                last_gae = delta + args.gamma * args.gae_lambda * lambda_nonterminal * last_gae
                advantages[t] = last_gae
                last_cond = delta + args.gamma * args.cond_lambda * lambda_nonterminal * last_cond
                cond_adv[t] = last_cond
            returns = advantages + values

            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros((*returns.shape, mtp), dtype=torch.bool, device=device)
            for horizon in range(mtp):
                valid_len = args.num_steps - horizon
                valid_horizon = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=device
                )
                for boundary_offset in range(horizon):
                    valid_horizon &= (
                        transition_boundaries[boundary_offset : boundary_offset + valid_len] == 0
                    )
                return_mtp[:valid_len, :, horizon] = returns[horizon:]
                return_mtp_mask[:valid_len, :, horizon] = valid_horizon
            target_probs = hl_support.project(return_mtp)

            # THE PRIVILEGED CHANNEL must actually IDENTIFY the action, or the network is
            # right to ignore it. Measured with GAE(0.95): cond_gap 0.002 and falling,
            # distill_kl 3e-4 -- the teacher was the student. The cause is informational,
            # not architectural: with gamma 0.99 and lambda 0.95 the advantage is dominated
            # by ~100 steps of downstream trajectory and value error, so I(a_t ; A_t | s_t)
            # is almost nil. The 1-step residual delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
            # is instead a near-deterministic function of a_t in a deterministic MuJoCo
            # transition, so conditioning on it is learnable, and it is exactly the classic
            # actor-critic advantage -- V(s) only, no action ever enters the critic.
            b_adv = cond_adv.reshape(-1)
            # SCALING A CONDITIONING INPUT IS NOT SCALING A GRADIENT. "batch" (v1-v5,
            # inherited from PPO adv-norm) is wrong twice over here:
            #   (a) mean subtraction destroys delta's natural zero. delta_t = 0 means
            #       "exactly as V expected"; subtracting the batch mean relabels the
            #       least-bad action of an all-bad batch as POSITIVE -- a false sign.
            #   (b) the batch sd makes the units non-stationary, and adv_boost is quoted
            #       in those units, so the same nominal margin means different things at
            #       different times.
            # Raw delta cannot be fed either: AdvEmbed's frequencies are FIXED (0.5..8).
            # Measured (v4, 131k steps): raw delta's RMS grew 0.61 -> 2.17 in 4 iterations
            # and already saturated the +-3 clip at 11%. Entropy preservation ordered
            # raw > ema_rms > batch, exactly as (a) predicts.
            if args.cond_scale == "batch":
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            elif args.cond_scale == "ema_rms":
                ms = b_adv.square().mean()
                cond_ms.mul_(args.cond_ema_beta).add_((1.0 - args.cond_ema_beta) * ms)
                bias = 1.0 - args.cond_ema_beta ** iteration
                b_adv = b_adv / (cond_ms / bias).sqrt().clamp_min(1e-8)
            elif args.cond_scale == "raw":
                pass
            else:
                raise ValueError(f"unknown cond_scale {args.cond_scale!r}")
            cond_scale_used = b_adv.square().mean().sqrt().item()
            b_adv_cond = b_adv.clamp(-args.adv_cond_clip, args.adv_cond_clip).unsqueeze(-1)
            cond_clipped = (b_adv.abs() >= args.adv_cond_clip).float().mean().item()

        b_obs = obs.reshape((-1,) + obs_shape)
        b_z = latent_zs.reshape(-1, act_dim)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).to(torch.float32)

        # ===== PAPER FIDELITY: the teacher is FIXED for the whole update ================
        # The paper fixes the teacher to the INITIAL policy. That cannot port to RL from
        # scratch -- our init is random, so an init-anchored teacher distills noise for the
        # entire run. What ports is the *staleness*: snapshot the teacher once per
        # iteration, so both losses are fixed supervised targets rather than a target that
        # moves every minibatch because the student moved. Also drops one of three
        # forwards per step.
        if args.adv_boost_final >= 0.0:
            anneal_frac = (iteration - 1.0) / max(args.num_iterations - 1.0, 1.0)
            boost = args.adv_boost + anneal_frac * (args.adv_boost_final - args.adv_boost)
        else:
            boost = args.adv_boost
        query_all = (b_adv_cond + boost).clamp(-args.adv_cond_clip, args.adv_cond_clip)
        with torch.no_grad():
            t_alpha, t_beta = [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                a_t, b_t = agent.policy(b_obs[sl], agent.cond_present(query_all[sl]))
                t_alpha.append(a_t)
                t_beta.append(b_t)
            b_t_alpha = torch.cat(t_alpha)
            b_t_beta = torch.cat(t_beta)

        # ===== v12: ONE FROZEN TAPE SNAPSHOT PER ITERATION ==============================
        # Hard-snapshot the trunk AND the predictors together, from the same weights that
        # produced the frozen OPSD teacher immediately above, then build every label under
        # no_grad. Labels and bootstrapped tails are detached by construction.
        if args.tape:
            with torch.no_grad():
                frozen_trunk = copy.deepcopy(agent.trunk).requires_grad_(False)
                frozen_predictor = copy.deepcopy(agent.tape_predictor).requires_grad_(False)
                # The rollout ACTED in the student context (privileged slot zeroed), so
                # the labels are read off the trunk in that same context.
                tap_chunks = []
                for chunk in b_obs.split(args.tape_target_batch_size):
                    zero_cond = chunk.new_zeros((chunk.shape[0], agent.cond_dim))
                    tap_chunks.append(
                        frozen_trunk.forward_taps(torch.cat([chunk, zero_cond], dim=-1))[1]
                    )
                frozen_taps = torch.cat(tap_chunks).reshape(
                    args.num_steps, args.num_envs, tape_num_layers, args.hidden
                )
                del tap_chunks
                # Y^l_1 = r^l_{t+1} - r^l_t. Rollout row t+1 IS the next state whenever
                # transition t stayed inside the episode, which is exactly the condition
                # every head is masked on below, so no boundary row is ever differenced.
                innovations = torch.zeros_like(frozen_taps)
                innovations[:-1] = frozen_taps[1:] - frozen_taps[:-1]
                # Frozen P^{l,-}_h at the stored (s_t, z_t). The ACTION CHANNEL is the
                # rollout's stored Beta latent z in [0,1]^A -- the policy's native
                # coordinate, already recorded in latent_zs -- not the env-scaled action.
                prediction_chunks = []
                for tap_chunk, z_chunk in zip(
                    frozen_taps.reshape(-1, tape_num_layers, args.hidden).split(
                        args.tape_target_batch_size
                    ),
                    b_z.split(args.tape_target_batch_size),
                    strict=True,
                ):
                    prediction_chunks.append(frozen_predictor(tap_chunk, z_chunk))
                frozen_predictions = torch.cat(prediction_chunks).reshape(
                    args.num_steps,
                    args.num_envs,
                    tape_num_layers,
                    args.tape_horizon,
                    args.hidden,
                )
                del prediction_chunks
                # VALIDITY IS v11's, NOT A SECOND NOTION. transition_boundaries (set from
                # terminations|truncations during the rollout) and transition_valids (the
                # per-step flag v11's MTP critic mask and GAE recursion are built from) are
                # reused verbatim; the head chain below is the same idiom as the MTP mask.
                step_ok = (transition_boundaries == 0) & (transition_valids > 0)
                tape_values, tape_step_masks = build_all_layer_horizon_td_tape_targets(
                    innovations, frozen_predictions, step_ok, args.tape_lambda
                )
                del frozen_predictions
                tape_masks = (
                    tape_step_masks.unsqueeze(2)
                    .expand(
                        args.num_steps, args.num_envs, tape_num_layers, args.tape_horizon
                    )
                    .contiguous()
                )
                latent_rms = frozen_taps.square().mean(dim=(0, 1, 3)).sqrt()
                _, tape_target_scale, _ = horizon_td_tape_loss(
                    torch.zeros_like(tape_values),
                    tape_values,
                    tape_masks,
                    latent_rms=latent_rms,
                    scale_floor_ratio=args.tape_scale_floor_ratio,
                )
                b_tape_targets = tape_values.reshape(
                    args.batch_size, tape_num_layers, args.tape_horizon, args.hidden
                )
                b_tape_mask = tape_masks.reshape(
                    args.batch_size, tape_num_layers, args.tape_horizon
                )
                tape_valid_frac = tape_step_masks.float().mean().item()
                del frozen_trunk, frozen_predictor, frozen_taps, innovations, tape_values

        clone_losses, distill_kls, v_losses, ents, gaps, tea_ents = [], [], [], [], [], []
        clip_fracs = []
        tape_losses, tape_trunk_gns, tape_predictor_gns = [], [], []
        tape_layer_losses = []
        # ===== DECOUPLED ACTOR / CRITIC BUDGETS ========================================
        # e4 beat e1 by ~1260 at 1M on mb128 -- but e4 also handed the CRITIC 4x its
        # regression passes, and those two kinds of reuse are not the same thing:
        #   actor reuse re-fits the SAME sampled action for the same state. It adds no
        #     information, it only sharpens the conditional, and it is paid for in entropy.
        #   critic reuse is ordinary supervised regression onto FIXED bootstrap targets,
        #     where extra passes simply reduce fitting error.
        # So they get separate budgets. Epochs past actor_epochs take a critic-only path:
        # one zeroed-context forward, no policy heads, roughly half the compute.
        for epoch in range(max(args.actor_epochs, args.critic_epochs)):
            do_actor = epoch < args.actor_epochs
            do_critic = epoch < args.critic_epochs
            permutation = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = permutation[start : start + args.minibatch_size]

                if not do_actor:
                    value_logits = agent.get_value(b_obs[mb])
                    log_value_probs = torch.log_softmax(value_logits, dim=-1)
                    value_ce = -(b_target_probs[mb] * log_value_probs).sum(dim=-1)
                    v_loss = (value_ce * b_target_mask[mb]).sum(dim=-1).mean()
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        v_losses.append(v_loss.item())
                    continue

                obs_mb, z_mb, adv_mb = b_obs[mb], b_z[mb], b_adv_cond[mb]
                n = obs_mb.shape[0]
                a_tea, b_tea = b_t_alpha[mb], b_t_beta[mb]
                query = query_all[mb]

                # One forward for both remaining contexts:
                #   [privileged at the REALIZED A_t ; privileged ABSENT (the student)].
                cond_absent = adv_mb.new_zeros((n, agent.cond_dim))
                cat_obs_mb = torch.cat([obs_mb, obs_mb], 0)
                cat_cond_mb = torch.cat([agent.cond_present(adv_mb), cond_absent], 0)
                if args.tape:
                    alpha, beta, value_logits, trunk_taps = agent.policy_and_value_taps(
                        cat_obs_mb, cat_cond_mb
                    )
                else:
                    alpha, beta, value_logits = agent.policy_and_value(
                        cat_obs_mb, cat_cond_mb
                    )
                a_cl, b_cl = alpha[:n], beta[:n]
                a_stu, b_stu = alpha[n:], beta[n:]

                # 1. Rationalization: fit p(a_t | s_t, A_t). "What it did" is the target.
                clone_loss = -(
                    Beta(a_cl, b_cl, validate_args=False).log_prob(z_mb).sum(-1).mean()
                )

                # 2. Per-dim clipped forward KL from the detached teacher into the student.
                kl_dims = beta_kl_per_dim(a_tea, b_tea, a_stu, b_stu).clamp_min(0.0)
                distill_loss = kl_dims.clamp(max=args.distill_kl_clip).sum(-1).mean()

                loss = args.clone_coef * clone_loss + args.distill_coef * distill_loss

                # 3. Critic: V(s) from the zeroed-context half only.
                if do_critic:
                    log_value_probs = torch.log_softmax(value_logits[n:], dim=-1)
                    value_ce = -(b_target_probs[mb] * log_value_probs).sum(dim=-1)
                    v_loss = (value_ce * b_target_mask[mb]).sum(dim=-1).mean()
                    loss = loss + args.vf_coef * v_loss
                    v_losses.append(v_loss.item())

                # 4. The tape, predicted from the STUDENT-context taps: the rollout acted
                # with the privileged slot zeroed, so that is the representation the frozen
                # labels were built on. It is a separate loss, never folded into `loss`.
                if args.tape:
                    tape_prediction = agent.tape_predictor(trunk_taps[n:], z_mb)
                    tape_loss, _, tape_head_losses = horizon_td_tape_loss(
                        tape_prediction,
                        b_tape_targets[mb],
                        b_tape_mask[mb],
                        scale=tape_target_scale,
                    )

                # GRADIENT ORDER, chosen deliberately:
                #   (1) tape backward FIRST, with retain_graph=True -- it shares the trunk
                #       subgraph with the task loss, which the task backward would free.
                #   (2) clip the two tape groups separately (trunk 0.025, predictors 0.25),
                #       STASH the result, then wipe .grad.
                #   (3) v11's task backward + global max_grad_norm clip, UNCHANGED. The
                #       grad buffers hold only task gradient at that instant, so the task
                #       clip sees exactly the norm v11 sees and rescales exactly as v11.
                #   (4) ADD the stashed tape gradient on top; one shared Adam step.
                # So the tape never passes through max_grad_norm and never rescales the
                # task gradient, and the task never inflates the tape's caps.
                optimizer.zero_grad(set_to_none=True)
                tape_grads = {}
                if args.tape:
                    (args.tape_coef * tape_loss).backward(retain_graph=True)
                    tape_trunk_gn = nn.utils.clip_grad_norm_(
                        tape_trunk_params, args.tape_trunk_grad_clip
                    )
                    tape_predictor_gn = nn.utils.clip_grad_norm_(
                        tape_predictor_params, args.tape_predictor_grad_clip
                    )
                    tape_grads = {
                        parameter: parameter.grad.detach().clone()
                        for parameter in tape_all_params
                        if parameter.grad is not None
                    }
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                for parameter, tape_grad in tape_grads.items():
                    if parameter.grad is None:
                        parameter.grad = tape_grad
                    else:
                        parameter.grad.add_(tape_grad)
                optimizer.step()

                with torch.no_grad():
                    clone_losses.append(clone_loss.item())
                    distill_kls.append(kl_dims.sum(-1).mean().item())
                    ents.append(
                        Beta(a_stu, b_stu, validate_args=False).entropy().sum(-1).mean().item()
                    )
                    tea_ents.append(
                        Beta(a_tea, b_tea, validate_args=False).entropy().sum(-1).mean().item()
                    )
                    # Does the trunk actually USE the privileged slot? If this is 0 the
                    # teacher is the student and the whole method is a no-op.
                    gaps.append(
                        (
                            a_tea / (a_tea + b_tea) - a_stu / (a_stu + b_stu)
                        ).abs().mean().item()
                    )
                    # If the margin has pushed most queries into the clamp, the relative
                    # query has degenerated back into v1's constant one.
                    clip_fracs.append(
                        (query.abs() >= args.adv_cond_clip - 1e-6).float().mean().item()
                    )
                    if args.tape:
                        tape_losses.append(tape_loss.item())
                        tape_trunk_gns.append(tape_trunk_gn.item())
                        tape_predictor_gns.append(tape_predictor_gn.item())
                        tape_layer_losses.append(
                            tape_head_losses.mean(dim=-1).cpu().numpy()
                        )

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        variance = np.var(y_true)
        explained_variance = np.nan if variance == 0 else 1 - np.var(y_true - y_pred) / variance
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/clone_nll", float(np.mean(clone_losses)), global_step)
        writer.add_scalar("losses/distill_kl", float(np.mean(distill_kls)), global_step)
        writer.add_scalar("losses/value_loss", float(np.mean(v_losses)), global_step)
        writer.add_scalar("losses/entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("debug/cond_gap", float(np.mean(gaps)), global_step)
        writer.add_scalar("debug/query_clip_frac", float(np.mean(clip_fracs)), global_step)
        writer.add_scalar("debug/adv_boost", float(boost), global_step)
        writer.add_scalar("debug/teacher_entropy", float(np.mean(tea_ents)), global_step)
        writer.add_scalar("debug/student_entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("debug/advantage_std", advantages.std().item(), global_step)
        writer.add_scalar("debug/cond_scale_rms", cond_scale_used, global_step)
        writer.add_scalar("debug/cond_clip_frac", cond_clipped, global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        if args.tape:
            writer.add_scalar("losses/tape_loss", float(np.mean(tape_losses)), global_step)
            # Both *_gn scalars are clip_grad_norm_'s return value, i.e. the PRE-clip
            # norm. That is the informative one: it says whether the cap is binding.
            # tape_trunk_gn > 0.025 means the trunk gradient was scaled down to 0.025.
            writer.add_scalar(
                "debug/tape_trunk_gn", float(np.mean(tape_trunk_gns)), global_step
            )
            writer.add_scalar(
                "debug/tape_predictor_gn", float(np.mean(tape_predictor_gns)), global_step
            )
            # PER-LAYER delivered loss. Missing early-block delivery is a falsifier, so
            # layer 0 -- the entry projection -- has to be visible and nonzero.
            tape_layer_mean = np.mean(np.stack(tape_layer_losses), axis=0)
            for layer_index, layer_value in enumerate(tape_layer_mean):
                writer.add_scalar(
                    f"debug/tape_layer{layer_index}", float(layer_value), global_step
                )
            writer.add_scalar(
                "debug/tape_scale", float(tape_target_scale.mean().item()), global_step
            )
            writer.add_scalar("debug/tape_valid_frac", tape_valid_frac, global_step)
        print("SPS:", sps)

    envs.close()
    writer.close()

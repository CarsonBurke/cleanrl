# OPSD-JacTeach v1 -- the champion, with a teacher that points somewhere the policy
# gradient cannot see. One lever (`jac_cos`), one mechanism.
# =====================================================================================
# WHY THIS FILE EXISTS
#
# Every arm in this family so far builds its teacher from the action-credit CORRELATION:
# condition a generative head on realized credit, distil the conditional back. For an
# exponential-family policy that is the score function `grad log pi(a|s)` re-encoded, so
# it can only ever produce a better-CONDITIONED estimate of a direction the policy
# gradient already knows. That is the measured ceiling: 17 advcond arms, three kappa
# ladders, two conditioning scales and a deep value readout all land in 8200-8842, and
# the champion's own margin over PPO decays +24% @1M -> +6.4% @8M while `cond_gap`
# collapses 0.030 -> 0.010. The channel got 3.7x louder and the network used it 3x less.
#
# The family charter (OPSD_FAMILY.md §6.5) names exactly one in-charter escape: a learned
# one-step transition, giving the PATHWISE derivative
#
#       d = d/da [ gamma * V(s + ds_hat(s, a)) ]
#
# This is not Q-learning and not a search. V(s) stays the value object (no Q(s,a), no twin
# critics, no target networks, no max backup); the model is supervised regression on the
# OBSERVED transitions of the single trajectory (no counterfactual stepping, no candidate
# sets, no save/restore); and the model is never rolled forward, so the classic
# compounding-error failure of model-based RL does not arise. It is one step, at states
# the policy actually visited.
#
# VALIDATED BEFORE A LINE OF THIS FILE WAS WRITTEN. 120k real HalfCheetah-v4 transitions,
# learned J vs the simulator's own central differences on 300 held-out states:
#
#   n_train    held-out ds R2    cos(J^T e_vx)    J rel.err
#     5,000         0.841            0.955          0.260
#    20,000         0.906            0.968          0.236
#   100,000         0.944            0.975          0.234
#
# obs[8] = qvel[0] = forward velocity, which is 94.4% of HalfCheetah's reward variance, so
# cos(J^T e_vx) IS the accuracy of the improvement direction. Note the magnitude error
# (0.23) does NOT improve with data while the direction does: the Jacobian is biased in
# scale and excellent in direction. That asymmetry is why this design normalizes the
# direction and takes its dose from the KL budget instead of the gradient norm. Note also
# that 5k samples already gives 0.955 -- one rollout is 32,768 -- so there is no warmup
# problem, and the gate below is a guardrail rather than a schedule. In-run, on the real
# chassis, the head reaches R2 0.95 and `debug/jac_gate` pins at 1.0.
#
# THE MEASURED FACT THAT JUSTIFIES THE ARM
#
#   debug/jac_align = -0.008 .. +0.046   (champion config, full batch, valid rows only)
#
# That is the cosine, in the policy's own metric, between the analytic direction and the
# parent's OWN teacher displacement. It is ~0: the pathwise direction is ORTHOGONAL to the
# credit direction, i.e. genuinely different information rather than the policy gradient
# rediscovered. Had it come out ~1 this file would be pointless, and that was the
# pre-registered kill condition.
#
# WHAT `jac_cos` DOES -- IT NAMES THE ANGLE, EXACTLY
#
# The teacher's displacement is decomposed into the credit direction plus the analytic
# direction's orthogonal remainder and rotated within that plane by exactly `jac_cos`,
# holding ||displacement|| fixed in the KL metric (u = disp/sd, since Beta KL at fixed
# concentration is quadratic in disp/sd, not disp). Verified numerically over 20k random
# rows in float64: achieved cosine equals the argument to 10 decimals, ||u_new|| == ||u_cred||
# to 7.4e-16 at every setting, and `jac_cos = 1` is an ALGEBRAIC identity (8.9e-16) rather
# than a skipped branch.
#
# THE FIRST VERSION OF THIS LEVER WAS WRONG AND THE FAILURE IS INSTRUCTIVE. It added
# `jac_step * g_hat` and renormalized. The achieved angle then depends on ||u_cred||, which
# grows with the teacher's aggressiveness and drifts through training: the same
# `jac_step = 0.10` that turned the teacher 57 degrees at batch 1024 turned it 21 degrees
# at batch 32768. A lever whose meaning moves with batch size cannot support a matched-step
# ladder, so the angle is now specified directly.
#
# DOSE. Holding ||u|| fixed is necessary and NOT sufficient: the loss charges a per-dim
# CLIPPED KL, the parent's direction saturates that clip hardest, and so ANY rotation
# de-saturates the clip and delivers more divergence. Since this lineage has measured that
# dose alone moves score (kappa 1/2/3/4 -> 5365/6077/5624/3128 @2M), the rotated
# displacement is shrunk by one bisected scalar per iteration until the delivered clipped KL
# equals what the parent's own teacher would have delivered on that very batch
# (`debug/jac_dose_scale`).
#
# MEASURED LADDER, CHAMPION CONFIG (16 envs x 2048, mb128, a4/c4, 8 iterations, seed 1):
#
#   jac_cos   jac_rot   jac_align   dose_scale   distill_kl   vs v6    cond_gap
#   parent         --          --           --      0.11192   1.00x     0.02851
#   0.95       0.9500      +0.001       0.9857      0.11897   1.06x     0.03031
#   0.85       0.8500      +0.046       0.9543      0.13924   1.24x     0.03227
#   0.50       0.5000      +0.006       0.8864     21.52324    192x     0.03544
#
# READ THAT LAST ROW HONESTLY. The snapshot-time dose IS matched by construction, so the
# 192x is not a dosing failure -- it is the student being UNABLE TO FOLLOW a teacher rotated
# 60 degrees: the clip binds on many dims, those dims go locally flat, the student gets no
# gradient on them and the divergence never comes down over the update epochs. cos 0.50 is
# therefore not a usable arm at this budget, and it is not launched. 0.95 is the clean
# comparison (+6% delivered, well inside the dose noise band); 0.85 is the aggressive arm
# and its +24% is disclosed rather than hidden. If 0.95 wins, the dose residual cannot be
# the explanation.
#
# SAFETY: CANNOT LOSE THROUGH MODEL INCOMPETENCE
#
# Dose is gated on the model's MEASURED held-out one-step R2 (`debug/jac_gate`), held-out by
# construction because the head has only ever trained on earlier rollouts. Gate closed =>
# no rotation => this file is the parent. Verified: `--jac-cos 1.0` is BIT-EXACT against
# `ppo_continuous_action_opsd_advcond_v6.py` in the champion configuration -- 18 TensorBoard
# scalars, zero differing -- so the champion's 8718 @8M is this file's control for free. The
# head's init draws are taken inside `fork_rng` for exactly that reason.
#
# FALSIFIABLE PREDICTIONS
#  1. `debug/jac_align` stays ~0 for the whole run. If it climbs toward 1 the analytic
#     direction has collapsed onto the credit direction and the mechanism is inert.
#  2. `debug/jac_gate` pins at 1.0 and `debug/jac_r2` stays >0.9. If the gate oscillates the
#     head is underfitting on-policy data and the arm runs at a fraction of its nominal dose.
#  3. The signature to beat is the DECAY, not the level: the parent is +24% @1M -> +6.4% @8M
#     over PPO. A direction carrying information the policy gradient lacks should flatten
#     that, so judge @4M-@8M. A win visible only @1M is a dose artifact.
#  4. `debug/jac_dose_scale` stays near 1. If it falls far below 1 the requested angle is
#     mostly unpayable at this budget and the effective mechanism is much weaker than named.
#  5. NOT PREDICTED: that 0.85 beats 0.95. Larger rotations are the interesting experiment,
#     not the expected winner -- and 0.50 has already been measured unfollowable.
#
# PROVENANCE NOTE. An earlier launch of this file used the FILE defaults (mb32, actor_epochs
# 1) rather than the champion's `--num-minibatches 128 --actor-epochs 4`. That is 32 gradient
# steps per iteration against the champion's 512, and the charter's §4 already records
# gradient-step count as the dominant lever on this chassis. Those runs were cancelled as
# mis-configured, not as failed mechanisms. Every number above is champion config.
#
# ------------------------- INHERITED v6 NOTES BELOW, UNCHANGED -------------------------

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

    # --- v1: ANALYTIC IMPROVEMENT TEACHER (charter route 3) -------------------------
    # jac_step = 0.0 reproduces the parent BIT FOR BIT. One lever, one mechanism.
    jac_cos: float = 0.85       # THE COSINE OF THE TEACHER ROTATION. 1.0 = the parent
                                # exactly (algebraic identity, not a skipped branch); 0.0 =
                                # the teacher's displacement is turned fully into the
                                # analytic direction's orthogonal component. The displacement
                                # MAGNITUDE is held at the parent's in the KL metric, so this
                                # is the only free variable and the dose is not confounded.
                                # Scale-free by construction: debug/jac_rot must equal this.
    jac_hidden: int = 64        # one-step transition head width (= trunk width)
    jac_coef: float = 1.0       # weight on the model's own regression loss
    jac_dose_iters: int = 16    # bisection steps for the delivered-KL dose match. 16 halvings
                                # resolve the scale to 1.5e-5, far below the noise on
                                # distill_kl, and cost 16 closed-form KL evaluations per
                                # iteration against 512 minibatch backward passes.
    jac_r2_min: float = 0.5     # gate closes fully at or below this held-out one-step R2
    jac_r2_open: float = 0.8    # gate fully open at or above. Measured offline on real
                                # HalfCheetah transitions: R2 0.84 from 5k samples, 0.94
                                # from 100k, and cos(J^T e_vx) vs simulator finite
                                # differences 0.955 -> 0.975. The gate is a guardrail for
                                # the first few iterations, not an objective.

    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    normalize_reward: bool = False
    clip_reward: bool = False

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


class TransitionHead(nn.Module):
    """One-step action-conditioned state-delta model: g(s, a) -> standardized ds.

    DELIBERATELY NOT ON THE SHARED TRUNK. Routing model gradients through the actor's
    representation would confound "the analytic direction helps" with "predicting the
    next state is a good auxiliary task", and the family has already been burned by
    exactly that class of confound. This is a separate 2-layer MLP on raw obs + action;
    at hidden=64 it is a rounding error next to the env loop.

    Targets are standardized per dimension with running statistics, because the qpos and
    qvel blocks of a MuJoCo observation differ by ~26x in action sensitivity (measured
    median |dJ| 0.0052 vs 0.1359) and an unstandardized MSE would fit only the velocities.
    """

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, obs_dim), std=0.01),
        )
        self.ds_mean: torch.Tensor
        self.ds_var: torch.Tensor
        self.register_buffer("ds_mean", torch.zeros(obs_dim))
        self.register_buffer("ds_var", torch.ones(obs_dim))

    def forward(self, obs, action):
        """Standardized delta prediction (the space the regression loss lives in)."""
        return self.net(torch.cat([obs, action], dim=-1))

    def next_obs(self, obs, action):
        """Predicted next observation in RAW units -- what V(s') must be evaluated on."""
        ds = self.forward(obs, action) * self.ds_var.sqrt().clamp_min(1e-6) + self.ds_mean
        return obs + ds

    @torch.no_grad()
    def update_stats(self, ds, beta):
        self.ds_mean.mul_(beta).add_((1.0 - beta) * ds.mean(0))
        self.ds_var.mul_(beta).add_((1.0 - beta) * ds.var(0))


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
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
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
        value_logits = self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        return alpha, beta, value_logits

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
        return self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    assert args.batch_size % args.num_minibatches == 0
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.actor_epochs > 0 and args.critic_epochs > 0
    assert args.adv_boost > 0.0, "a non-positive margin makes the teacher no better"

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

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    assert obs_shape is not None and action_shape is not None
    act_dim = int(np.prod(action_shape))
    # jac_step == 0 must be BIT-EXACT vs the parent, so the head is neither built nor
    # trained in that case, and when it IS built its init draws happen inside fork_rng --
    # orthogonal_ init would otherwise advance the global stream and shift every later
    # minibatch permutation. (This is the same class of bug the conseq_v1 probe had.)
    use_jac = args.jac_cos < 1.0
    jac_model = None
    jac_optimizer = None
    if use_jac:
        with torch.random.fork_rng(devices=[device] if args.cuda else []):
            jac_model = TransitionHead(obs_shape[0], act_dim, args.jac_hidden).to(device)
        jac_optimizer = optim.Adam(jac_model.parameters(), lr=args.learning_rate, eps=1e-5)

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

        # ===== ROUTE 3: THE ANALYTIC IMPROVEMENT DIRECTION =============================
        # Everything else in this family builds its teacher from the action-credit
        # CORRELATION, which for an exponential-family policy is the policy gradient's own
        # information -- so it can sharpen the estimate but not change what is known. The
        # one information source that is different and still in charter is the pathwise
        # derivative through a learned one-step transition:
        #       d = d/dz [ gamma * V( s + ds_hat(s, a(z)) ) ]
        # V(s) stays the value object (no Q, no twin critics, no target nets), the model is
        # supervised regression on OBSERVED transitions of the single trajectory, and the
        # model is never rolled forward, so compounding error does not arise.
        #
        # VALIDATED OFFLINE BEFORE THIS FILE EXISTED (real HalfCheetah-v4 transitions,
        # learned J vs simulator central differences on 300 held-out states):
        #   n_train    held-out ds R2    cos(J^T e_vx)    cos(full J)
        #     5,000        0.841            0.955           0.969
        #   100,000        0.944            0.975           0.975
        # e_vx is obs index 8 = qvel[0] = forward velocity, i.e. exactly what HalfCheetah
        # pays for, so cos(J^T e_vx) IS the accuracy of the improvement direction. The
        # Jacobian's MAGNITUDE is ~25% off and does not improve with data -- irrelevant,
        # because the direction is normalized and the dose is jac_step * gate.
        jac_gate = 0.0
        jac_r2 = float("nan")
        jac_align = float("nan")
        jac_rot = float("nan")
        jac_dir_norm = float("nan")
        jac_clamp_frac = float("nan")
        jac_dose_scale = float("nan")
        if use_jac:
            b_next_obs = next_obses.reshape((-1,) + obs_shape)
            # `transition_valids` marks "a next observation exists", which is what GAE
            # needs. A one-step DYNAMICS model needs more: a hard termination is not an
            # ordinary physics step, and GAE itself already multiplies the two
            # (bootstrap_nonterminal = (1 - term) * valid). Use the same product here so the
            # model, its R2, and the direction all agree on what a transition is. No-op on
            # HalfCheetah-v4, which never terminates; correct on envs that do.
            b_valid = (
                transition_valids * (1.0 - transition_terminations)
            ).reshape(-1)
            b_vsel = b_valid > 0
            with torch.no_grad():
                # HELD-OUT by construction: the head has only ever seen earlier rollouts.
                ds_true = b_next_obs - b_obs
                b_act = agent.action_low + (agent.action_high - agent.action_low) * b_z
                ds_pred = (
                    jac_model(b_obs, b_act)
                    * jac_model.ds_var.sqrt().clamp_min(1e-6)
                    + jac_model.ds_mean
                )
                w = b_valid.unsqueeze(-1)
                n_valid = float(b_valid.sum().item())
                if n_valid >= 2.0:
                    mu = (ds_true * w).sum(0) / n_valid
                    sse = (((ds_true - ds_pred) ** 2) * w).sum(0)
                    sst = (((ds_true - mu) ** 2) * w).sum(0).clamp_min(1e-12)
                    jac_r2 = float((1.0 - sse / sst).mean().item())
                # else: jac_r2 stays NaN. With no usable rows, sse and sst are both 0 and
                # the clamp would make R2 read exactly 1.0 -- a *perfect* model on this
                # file's primary safety signal, from zero data. NaN keeps the gate shut.
            # GUARDRAIL, NOT AN OBJECTIVE: an inaccurate model yields a noise direction, so
            # the dose falls to zero and this arm degenerates to the parent exactly. It
            # therefore cannot lose through model incompetence, only through a measured,
            # licensed step.
            jac_gate = float(
                min(max((jac_r2 - args.jac_r2_min) / (args.jac_r2_open - args.jac_r2_min), 0.0), 1.0)
            )
            if jac_gate > 0.0:
                dirs, s_alphas, s_betas = [], [], []
                for start in range(0, args.batch_size, args.minibatch_size):
                    sl = slice(start, start + args.minibatch_size)
                    z_var = b_z[sl].detach().clone().requires_grad_(True)
                    act = (
                        agent.action_low
                        + (agent.action_high - agent.action_low) * z_var
                    )
                    s_next = jac_model.next_obs(b_obs[sl], act)
                    v_next = hl_support.to_scalar(agent.get_value(s_next)[:, 0])
                    (grad_z,) = torch.autograd.grad(
                        (args.gamma * v_next).sum(), z_var
                    )
                    dirs.append(grad_z.detach())
                    with torch.no_grad():
                        s_a, s_b = agent.policy(b_obs[sl], agent._zero_cond(b_obs[sl]))
                        s_alphas.append(s_a)
                        s_betas.append(s_b)
                jac_grad = torch.cat(dirs) * b_valid.unsqueeze(-1)
                with torch.no_grad():
                    conc = b_t_alpha + b_t_beta
                    mean_t = b_t_alpha / conc
                    sd_t = (mean_t * (1.0 - mean_t) / (conc + 1.0)).sqrt()
                    # The student's own mean, needed because the experiment is about the
                    # DIRECTION of the teacher's displacement, so the displacement has to be
                    # named explicitly rather than folded into the teacher's mean. Chunked
                    # at minibatch_size like the teacher query, not one 32768-row forward.
                    s_alpha_all = torch.cat(s_alphas)
                    s_beta_all = torch.cat(s_betas)
                    mean_s = s_alpha_all / (s_alpha_all + s_beta_all)
                    # EVERYTHING BELOW LIVES IN THE POLICY'S OWN METRIC, u = disp / sd.
                    # Two reasons, and they are the same reason twice:
                    #  1. KL between two Betas at fixed concentration is quadratic in
                    #     disp/sd, not in disp. Renormalizing a raw-space displacement holds
                    #     the wrong quantity fixed: measured, it still let distill_kl climb
                    #     0.0035 -> 0.0087 across the rotation ladder, which is exactly the
                    #     dose confound this is supposed to remove.
                    #  2. dV/dz is a COVECTOR. Steepest ascent under the metric the KL
                    #     budget actually charges for is grad * sd, not grad. Mixing raw
                    #     gradients would silently weight the low-variance action dims
                    #     hardest -- the dims the policy is already most certain about.
                    u_cred = (mean_t - mean_s) / sd_t
                    g_u = jac_grad * sd_t
                    g_norm = g_u.norm(dim=-1, keepdim=True)
                    jac_dir_norm = float(g_norm[b_vsel].mean().item()) if b_vsel.any() else float("nan")
                    g_hat = g_u / g_norm.clamp_min(1e-8)
                    # EXACT ROTATION, not an emergent mix. The first version added
                    # `jac_step * g_hat` and renormalized. That is scale-dependent and it
                    # broke in practice: the same jac_step=0.10 that turned the teacher 57
                    # degrees at batch 1024 turned it only 21 degrees at batch 32768,
                    # because the achieved angle depends on ||u_cred||, which grows with the
                    # teacher's aggressiveness and drifts over training. A lever whose
                    # meaning moves with batch size and training stage cannot support a
                    # matched-step ladder.
                    #
                    # So specify the ANGLE directly. Decompose the analytic direction into
                    # the credit direction plus an orthogonal remainder, and rotate within
                    # that plane by exactly the requested cosine:
                    #     u_new = ||u_cred|| * (cos * u_hat + sin * g_perp)
                    # Now `jac_cos` IS the achieved rotation at every scale, ||u_new|| ==
                    # ||u_cred|| holds by construction rather than by a division, and
                    # jac_cos = 1 is an ALGEBRAIC identity to the parent -- which is a
                    # strictly stronger control than the branch-skip at jac_step = 0, since
                    # it exercises this whole block and still has to reproduce v6.
                    u_norm = u_cred.norm(dim=-1, keepdim=True)
                    u_hat = u_cred / u_norm.clamp_min(1e-12)
                    g_perp = g_hat - (g_hat * u_hat).sum(-1, keepdim=True) * u_hat
                    perp_norm = g_perp.norm(dim=-1, keepdim=True)
                    g_perp = g_perp / perp_norm.clamp_min(1e-12)
                    cos_t = min(max(args.jac_cos, -1.0), 1.0)
                    cos_t = 1.0 - jac_gate * (1.0 - cos_t)  # gate closed => no rotation
                    sin_t = math.sqrt(max(1.0 - cos_t * cos_t, 0.0))
                    u_rot = u_norm * (cos_t * u_hat + sin_t * g_perp)
                    # Degenerate rows keep the parent's displacement exactly: a credit
                    # direction of zero has no plane to rotate in, and an analytic direction
                    # parallel to it (or zeroed by the validity mask) has no orthogonal
                    # component to rotate toward.
                    ok = (u_norm > 1e-9) & (perp_norm > 1e-9)
                    u_new = torch.where(ok, u_rot, u_cred)
                    # ===== MATCH THE DELIVERED BUDGET, NOT THE DISPLACEMENT NORM =========
                    # Holding ||u|| fixed is necessary but NOT sufficient, because the loss
                    # charges a per-dim CLIPPED KL (tau = distill_kl_clip). The parent's
                    # direction is the one that saturates that clip hardest, so it delivers
                    # the LEAST divergence per unit displacement, and any rotation
                    # de-saturates the clip and delivers more. Measured, champion config:
                    #   cos 0.85 -> distill_kl 0.132 vs the parent's 0.112   (+18%)
                    #   cos 0.50 -> distill_kl 13.9                          (125x)
                    # So the cosine cannot double as a dose control: every rotation costs
                    # budget, and at cos 0.50 the arm is a dose experiment wearing a
                    # direction experiment's clothes. This lineage has measured that dose
                    # ALONE moves score (kappa 1/2/3/4 -> 5365/6077/5624/3128 @2M), so the
                    # confound is not cosmetic -- it is the size of the effect being chased.
                    #
                    # Fix: shrink the rotated displacement by one scalar per iteration until
                    # the delivered clipped KL equals what the PARENT's own teacher would
                    # have delivered on this very batch, by bisection on the closed-form Beta
                    # KL. Batch-level, because `losses/distill_kl` -- the quantity being
                    # matched -- is itself a batch mean, and because this family's existing
                    # duals (tau, omega, target_distill_kl) are all batch-level bisections.
                    # The arm now differs from the parent in DIRECTION ONLY.
                    def _delivered(u):
                        m = (mean_s + u * sd_t).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                        return (
                            beta_kl_per_dim(
                                m * conc, (1.0 - m) * conc, s_alpha_all, s_beta_all
                            )
                            .clamp_min(0.0)
                            .clamp(max=args.distill_kl_clip)
                            .sum(-1)
                            .mean()
                        )

                    parent_kl = _delivered(u_cred)
                    lo, hi = 0.0, 1.0
                    if _delivered(u_new) > parent_kl:
                        for _ in range(args.jac_dose_iters):
                            mid = 0.5 * (lo + hi)
                            if _delivered(mid * u_new) > parent_kl:
                                hi = mid
                            else:
                                lo = mid
                    else:
                        # Rotation delivered no more than the parent: nothing to shrink. Only
                        # reachable when the clip binds on essentially nothing.
                        lo = hi = 1.0
                    jac_dose_scale = 0.5 * (lo + hi)
                    u_new = jac_dose_scale * u_new
                    mean_new = (mean_s + u_new * sd_t).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                    # THE DECISIVE READOUT. If this cosine is ~1 the pathwise direction is
                    # just the credit direction rediscovered and route 3 buys nothing; if it
                    # is ~0 it is genuinely orthogonal information. Measured on the FULL
                    # batch, in the metric the mixing happens in.
                    # AVERAGED OVER VALID ROWS ONLY. On an invalid row g_hat is the zero
                    # vector, so cosine_similarity returns exactly 0 -- which would drag
                    # jac_align toward 0 and jac_rot toward 1, i.e. bias this arm's own
                    # pre-registered success criterion in the flattering direction by the
                    # invalid fraction. Small (~1-3% near episode boundaries) and entirely
                    # avoidable.
                    cos = torch.nn.functional.cosine_similarity
                    jac_align = float(cos(u_cred, g_hat, dim=-1)[b_vsel].mean().item())
                    # How far the teacher actually turned. jac_step is a rotation weight, so
                    # this is the quantity the ladder is really sweeping.
                    jac_rot = float(cos(u_cred, u_new, dim=-1)[b_vsel].mean().item())
                    # MINOR-4 made auditable rather than invisible: the clamp below is the
                    # one place the constant-dose invariant can break. It only ever SHRINKS
                    # the displacement, so it is safe, but it binds more often as the policy
                    # sharpens (measured ~0.004% of dims at parent-like doses, ~5% for a
                    # near-deterministic policy) and entropy here reaches -9.6 late in a run.
                    # Logging the binding fraction keeps the dose-matched ladder auditable
                    # instead of quietly drifting under budget at 8M.
                    jac_clamp_frac = float(
                        (
                            ((mean_s + u_new * sd_t) <= SAMPLE_EPS)
                            | ((mean_s + u_new * sd_t) >= 1.0 - SAMPLE_EPS)
                        )[b_vsel].float().mean().item()
                    )
                    b_t_alpha = mean_new * conc
                    b_t_beta = (1.0 - mean_new) * conc

        clone_losses, distill_kls, v_losses, ents, gaps, tea_ents = [], [], [], [], [], []
        clip_fracs = []
        jac_losses = []
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
                alpha, beta, value_logits = agent.policy_and_value(
                    torch.cat([obs_mb, obs_mb], 0),
                    torch.cat([agent.cond_present(adv_mb), cond_absent], 0),
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

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                if use_jac:
                    # Plain supervised regression on the OBSERVED transition. Separate
                    # optimizer and separate grad clip: sharing either would let the
                    # model's gradient norm change what the actor's clip does, which is a
                    # confound, and would break the jac_step == 0 control.
                    ds_target = (
                        b_next_obs[mb] - obs_mb - jac_model.ds_mean
                    ) / jac_model.ds_var.sqrt().clamp_min(1e-6)
                    act_mb = (
                        agent.action_low + (agent.action_high - agent.action_low) * z_mb
                    )
                    vmask = b_valid[mb].unsqueeze(-1)
                    jac_loss = (((jac_model(obs_mb, act_mb) - ds_target) ** 2) * vmask).sum() / (
                        vmask.sum().clamp_min(1.0) * obs_shape[0]
                    )
                    jac_optimizer.zero_grad(set_to_none=True)
                    (args.jac_coef * jac_loss).backward()
                    nn.utils.clip_grad_norm_(jac_model.parameters(), args.max_grad_norm)
                    jac_optimizer.step()
                    jac_losses.append(jac_loss.item())

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

        if use_jac:
            # Refresh AFTER the update so next iteration's R2 and direction are computed
            # from a consistent (weights, stats) pair and the R2 stays genuinely held-out.
            with torch.no_grad():
                keep = b_vsel
                # ds.var(0) is the UNBIASED estimator, so one row yields NaN and would
                # silently poison next iteration's standardization, R2 and direction.
                # Unreachable at real batch sizes (1024-32768 rows); the guard is free.
                if int(keep.sum().item()) >= 2:
                    jac_model.update_stats(
                        (b_next_obs[keep] - b_obs[keep]), 0.0 if iteration == 1 else 0.99
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
        if use_jac:
            writer.add_scalar("debug/jac_r2", jac_r2, global_step)
            writer.add_scalar("debug/jac_gate", jac_gate, global_step)
            writer.add_scalar("debug/jac_dir_norm", jac_dir_norm, global_step)
            # THE decisive number: ~1 means route 3 rediscovered the credit direction and
            # buys nothing; ~0 means it is orthogonal information.
            writer.add_scalar("debug/jac_align", jac_align, global_step)
            writer.add_scalar("debug/jac_rot", jac_rot, global_step)
            writer.add_scalar("debug/jac_clamp_frac", jac_clamp_frac, global_step)
            # How much displacement the clip forced the rotation to give back. 1.0 means the
            # rotation was free; a small value means the requested angle is mostly unpayable
            # at this budget, which is the honest signal that the cosine is too aggressive.
            writer.add_scalar("debug/jac_dose_scale", jac_dose_scale, global_step)
            writer.add_scalar(
                "losses/jac_model", float(np.mean(jac_losses)) if jac_losses else float("nan"),
                global_step,
            )
        print("SPS:", sps)

    envs.close()
    writer.close()

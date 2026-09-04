# OPSD-AdvCond v10 -- the v6 champion plus a TRUE one-sided entropy floor.
# =====================================================================================
# WHY. v6 wins on HalfCheetah-v4 (8718 last-20 vs PPO's 8278) while collapsing to
# H = -13.0 nats, i.e. ~-2.2/dim, ~sd 0.037 of the action range. On HalfCheetah that
# is harmless: the episode never terminates, so a near-deterministic policy still
# collects 1000 steps of gradient signal every rollout. On Hopper/Walker2d it is the
# classic early-termination failure: sharpen too early, fall over, and the return
# signal dies before the critic can price the mistake. Entropy collapse is a FEATURE
# on HalfCheetah (v9 removed the clone's sharpening channel and scored WORSE, 3895
# vs 5365 @2M) and the sharpest a-priori risk everywhere else. v6 has no entropy
# regularization of any kind: no ent_coef, no floor, no dual.
#
# THE INSTRUMENT. A hinge, not a regularizer:
#
#     L += ent_alpha * relu(H_target - H_student)
#
# relu() is EXACTLY zero with EXACTLY zero gradient while H > H_target, so above the
# floor this file is v6 bit-for-bit -- no drift, no retuning, no interaction with the
# clone/distill balance that produced the champion. The usual SAC-style term
# alpha*(H_target - H) is NOT INERT: for any alpha >= 0 it is an unconditional
# entropy BONUS -- descending it raises H at every H, including far above target --
# so it perturbs v6 everywhere and silently retunes the clone/distill balance that
# produced the champion. (An earlier draft of this header claimed the two-sided form
# "pushes entropy down toward the target". That is false and was falsified in review:
# d/dtheta[alpha*(target - H)] = -alpha*dH/dtheta, which always increases H. The real
# argument for the relu is inertness, which is strictly stronger.)
# That distinction is the whole design.
#
# ent_alpha is a dual variable, updated once per iteration (not per minibatch --
# the hopsd v28 lineage showed a per-minibatch multiplicative dual lags and rings):
#
#     ent_alpha <- clip(ent_alpha * exp(eta * (H_target - H_measured)),
#                       alpha_min, alpha_max)
#
# so it decays back to alpha_min on its own whenever the floor is not binding. It is
# floored at alpha_min rather than 0 because a multiplicative dual can never climb off
# exactly 0. That is safe: inertness comes from the relu, not from alpha being small.
#
# TARGET UNITS. Per action dim, so one number transfers across envs: Hopper (3 dims)
# gets -3.0, HalfCheetah/Walker2d (6 dims) get -6.0 at the -1.0 default. -1.0/dim is
# a Beta with sd ~= 0.1 of the action range (ln(0.1*sqrt(12)) = -1.06), i.e. it keeps
# ~10% of the range as exploration noise. The champion's -2.2/dim sits well below it,
# so on HalfCheetah the floor WILL bind eventually -- that is a real intervention
# there and is why HalfCheetah is run as a control, not assumed neutral.
#
# FALSIFIABLE PREDICTIONS.
#   1. `debug/ent_hinge` == 0 and `debug/ent_floor_binding` == 0 for every iteration
#      where `losses/entropy` > `debug/ent_target`. NOTE `debug/ent_alpha` is floored
#      at ent_alpha_min = 1e-4 and is never 0; that is not a leak, because 1e-4 * 0
#      is still exactly 0. Verified numerically, not assumed.
#   2. Hopper/Walker2d: v10 > v6. If v6 already transfers, the floor buys nothing and
#      this file should be abandoned, not tuned.
#   3. HalfCheetah: v10 <= v6, and the gap measures what the collapse was worth.
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

    # --- v10: one-sided entropy floor (relu hinge; exactly inert above the floor) ---
    ent_target_per_dim: float = -1.0  # nats per action dim. -1.0 ~= Beta sd 0.1 of the
                                      # action range; summed over dims -> H_target.
    ent_alpha_init: float = 1e-4      # == ent_alpha_min: a multiplicative dual cannot
                                      # climb off exactly 0. Inertness comes from the
                                      # relu, not from alpha, so this is still v6 until
                                      # the floor binds.
    ent_alpha_eta: float = 0.1        # multiplicative dual step, once per iteration
    ent_alpha_max: float = 0.5        # cap; the floor is a guardrail, not an objective
    ent_alpha_min: float = 1e-4       # dual floor so alpha can climb back off zero

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

    # v10: entropy floor. Target is per-dim so one number transfers across envs
    # (Hopper 3 dims -> -3.0, HalfCheetah/Walker2d 6 dims -> -6.0 at the default).
    ent_target = args.ent_target_per_dim * act_dim
    ent_alpha = args.ent_alpha_init

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

        clone_losses, distill_kls, v_losses, ents, gaps, tea_ents = [], [], [], [], [], []
        clip_fracs, hinges = [], []
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

                # 2b. v10 ENTROPY FLOOR. relu() is exactly 0 with exactly 0 gradient
                #     while H > H_target, so above the floor this is v6 to the bit.
                #     A two-sided alpha*(target - H) would instead push entropy DOWN
                #     toward the target and destroy the collapse that wins here.
                ent_stu = Beta(a_stu, b_stu, validate_args=False).entropy().sum(-1).mean()
                ent_hinge = (ent_target - ent_stu).clamp_min(0.0)
                loss = loss + ent_alpha * ent_hinge

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

                with torch.no_grad():
                    clone_losses.append(clone_loss.item())
                    distill_kls.append(kl_dims.sum(-1).mean().item())
                    ents.append(ent_stu.item())   # v10: reuse, no second Beta build
                    hinges.append(ent_hinge.item())
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

        # v10: dual update, ONCE per iteration on the iteration's mean entropy. Doing
        # it per minibatch makes a multiplicative dual lag and ring (hopsd v28). alpha
        # decays on its own whenever the floor is slack, so it cannot accumulate.
        ent_mean = float(np.mean(ents)) if ents else ent_target
        ent_alpha = float(
            np.clip(
                ent_alpha * math.exp(args.ent_alpha_eta * (ent_target - ent_mean)),
                args.ent_alpha_min,
                args.ent_alpha_max,
            )
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
        writer.add_scalar("debug/ent_alpha", ent_alpha, global_step)
        writer.add_scalar("debug/ent_hinge", float(np.mean(hinges)), global_step)
        writer.add_scalar("debug/ent_target", ent_target, global_step)
        writer.add_scalar("debug/ent_floor_binding",
                          float(np.mean([h > 0.0 for h in hinges])), global_step)
        writer.add_scalar("debug/cond_gap", float(np.mean(gaps)), global_step)
        writer.add_scalar("debug/query_clip_frac", float(np.mean(clip_fracs)), global_step)
        writer.add_scalar("debug/adv_boost", float(boost), global_step)
        writer.add_scalar("debug/teacher_entropy", float(np.mean(tea_ents)), global_step)
        writer.add_scalar("debug/student_entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("debug/advantage_std", advantages.std().item(), global_step)
        writer.add_scalar("debug/cond_scale_rms", cond_scale_used, global_step)
        writer.add_scalar("debug/cond_clip_frac", cond_clipped, global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        print("SPS:", sps)

    envs.close()
    writer.close()

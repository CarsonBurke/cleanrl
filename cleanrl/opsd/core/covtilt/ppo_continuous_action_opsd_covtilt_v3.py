# OPSD-CovTilt v3 — optimistic posterior self-teacher, projected through a feasible
# mixture trust region. No PPO ratio or policy-gradient actor loss.
# =====================================================================================
# DURABLE EVIDENCE (HalfCheetah-v4, seed 1):
#   method                 @500k   @1M   @2M   @4M   final
#   PPO baseline             1576   3079   5012  6716  8278@8M
#   hopsd_v27_sdgate         2165   4208   6034  7030  8171@8M
#   hopsd_v36_pooled         3126   4557   6870    --  2.32M
#   CovTilt v1 omega=4       2460   3623   4324  5056  5743@8M
#   CovTilt v2 entdrop=.12   2114   3542   3947  3615  3617@6.77M
#   CovTilt v2 entdrop=.18   2114   3542   3947  3566  3866@6.77M
#
# v1 proved that the structured advantage/action-covariance teacher is real: +56% over
# PPO at 500k and +18% at 1M. But its divergence and entropy dose were too aggressive,
# then its tilt signal annealed: step_kl 0.73@32k -> 0.25@2M -> 0.0167@8M, |Delta|
# 0.088 -> 0.047 -> 0.015. It finished only 5743.
#
# v2 made the fit statistically correct (Gaussian NLL, no Beta-clamp gradient death) and
# fixed a mean-destroying independent alpha/beta cap, but its entropy governor was
# mathematically infeasible. It held the optimistic mean fixed and inflated only sd.
# On bounded distributions, a shifted mean imposes a mean-dependent maximum entropy, so
# sd_inflation pinned at its 5.0 ceiling from the FIRST update in both arms and still
# missed the requested floor. Entropy collapsed to -12; both arms regressed after 2M.
# Their runners later died at 6.77M, but durable curves already dominate the decision:
# neither job should be retried.
#
# v3 fixes the cause, not the symptom. The Gaussian-fit teacher first proposes a raw,
# aggressive posterior q_raw (omega=4, v1's best dose). We then project the WHOLE
# distribution toward the rollout policy using
#
#       q_eta = (1-eta) pi_rollout + eta q_raw,    eta in [0,1].
#
# The mixture's first two moments are exact:
#   mu_eta = (1-eta)mu_old + eta mu_raw
#   E[z^2]_eta = (1-eta)E_old[z^2] + eta E_raw[z^2],
# including the between-policy disagreement variance eta(1-eta)(mu_raw-mu_old)^2.
# One Beta moment-matches those moments. eta=0 is EXACTLY the rollout distribution, so
# both constraints are always feasible. A bisection takes the largest eta satisfying
#   mean_s KL(q_eta || pi_rollout) <= target_distill_kl
#   H(q_eta) >= H(pi_rollout) - max_ent_drop.
# This is a true trust-region projection over the full teacher, not PPO clipping and not
# a lagging entropy bonus. Gradients still flow only through the student in the final
# per-coordinate forward-KL distillation loss.
#
# Numeric validation over 40 randomized realistic Beta batches: KL was monotone in eta
# with 0/2000 violations; the joint-feasible set was an interval in 40/40 trials; every
# trial had a feasible eta because eta=0 is self-distillation's identity element.
#
# PRE-RUN AUDIT CORRECTIONS (all verified numerically before any GPU time):
#  1. Cap headroom. projection_cap = max(cap_d, 0.5(alpha+beta)) never widens a sharp
#     rollout, but with the cap AT the rollout concentration a sharpening proposal is
#     clipped straight back to it: measured achievable entropy drop 0.005 nats at
#     concentration 200, 0.000 at 2000 and 10000. The cap silently became the controller
#     and killed the scale channel of improvement -- v2's failure wearing a new hat.
#     conc_headroom=2 bounds one step at ~0.35 nats/dim, ~17x the 0.12 budget, so
#     max_ent_drop stays the binding controller at every sharpness and the cap stays a
#     pure safety bound (verified: drop 2.06 at conc 200, 2.17 at conc 10000).
#  2. fp64 trust region. moments_to_beta lets the alpha,beta>=1 floor bypass the
#     concentration cap, so a boundary-clamped mean reaches nu ~ 1/MEAN_EPS = 1e6, where
#     fp32 lgamma errs ~0.4 nats and beta_kl_per_dim is a difference of large cancelling
#     lgamma terms; the mixture also recovers variance as E[z^2]-mean^2. Both fed the
#     omega and eta bisections, i.e. the constraints themselves. Phase 2 is now float64
#     (no_grad, once per iteration, consumed via .item()); the student stays fp32.
#  3. DOSING IS ALREADY INFORMATION-PROPORTIONAL, so omega is FIXED by default. Var(u)=1
#     and Var(z_std)=1 by construction, so the per-dim regression R^2 IS Delta^2 and the
#     teacher's KL scales as omega^2 Delta^2. Fixed omega therefore spends KL in
#     proportion to measured advantage/action information and anneals to zero exactly
#     when that correlation vanishes. This reframes v1: step_kl decayed BECAUSE |Delta|
#     decayed, and |Delta| decayed because entropy collapsed to -9.5 -- a near
#     deterministic policy has no action diversity left to correlate with advantage. The
#     death spiral runs entropy -> signal, so max_ent_drop is the fix and forcing the
#     dose is the suspect. --omega-dual bisects omega to spend the budget regardless of
#     |Delta|, which amplifies noise when Delta is noise; it is an ABLATION, not default.
#
# ARMS: mix_e12 (fixed omega=4, entdrop 0.12) is primary, mix_e04 tightens the entropy
# budget, dual12 tests forced dosing. Entropy-preserving improvement is real and measured:
# at max_ent_drop=0 the projection still spends 0.26 nats of teacher KL at exactly -0.0
# nats of entropy, because the mixture's disagreement variance offsets the sharpening.
#
# Retained v2 corrections: heteroscedastic Gaussian MLE of
# z_std = Delta(s)u + exp(kappa(s))eps; concentration (not independent alpha/beta)
# sharpness cap; absolute-EMA-anchored per-dim signal gates; rank-normalized advantages.
#
# Prediction: mix_eta is below 1 while the raw teacher is too hot, teacher entropy drop
# is <=0.12 every iteration, and the early return curve keeps v1's advantage without the
# v1/v2 entropy collapse. Kill-tells: mix_eta -> 0 while |Delta| remains healthy means
# the raw proposal is incompatible with the constraint geometry; |Delta| -> 0 before 2M
# means covariance credit itself is insufficient and further dose tuning is pointless.
# =====================================================================================
#
# --- v1 conceptual method (retained below) ---
# THE PAPER (arXiv 2601.18734, On-Policy Self-Distillation): one network, two contexts.
# The teacher sees privileged info y*, the student sees only x, the student rolls out,
# and the loss is a per-token full-distribution divergence D(p_T || p_S) along the
# student's own trajectory with gradients flowing only through the student. No RL loss.
#
# THE RL TRANSLATION USED HERE: "token" = action dimension. "privileged info" = the
# advantage the rollout actually earned, which the acting policy provably could not see.
# The student re-reads every step it took, is told the standardized rank of the advantage
# that step earned, and rationalizes: "given that this action scored at rank u, what
# distribution should I have had?" That rationalization, queried at an optimistic rank,
# IS the teacher. The student then distills into it. No PPO ratio, no clipping surrogate,
# no advantage-weighted policy gradient anywhere.
#
# WHY THIS SHAPE OF TEACHER (the whole idea).
# Prior outcome-conditioned teachers in this repo failed hard and for one shared reason:
#   hopsd_v21_upsidedown (teacher sees z-scored return g, queried at g*=+1sigma): 1087@1M.
#   hopsd_v33_ucond      (teacher sees outcome score u, queried at u*=1.5):        1736@500k, dead.
#   hopsd_v1             (outcome context + AWR tilt):                             -86@2M, provably degenerate.
# In all three the outcome entered as one more MLP input next to a far more predictive
# channel (the hindsight future-action window phi). The network explains a_t with phi,
# the outcome channel goes inert, and the optimistic query is then an out-of-distribution
# poke at a coordinate the model learned to ignore.
# CovTilt removes that failure mode by construction. The rank u never enters the trunk.
# It enters as a fixed MULTIPLICATIVE term in a structured tilt:
#
#     mean_T(s, u)  =  m_S(s)  +  u * Delta(s) * sd_S(s)          (in units of the policy's own sd)
#     sd_T(s)       =  sd_S(s) * exp(kappa(s))
#
# fit by plain unweighted maximum likelihood of the action actually taken. The channel
# cannot go inert: Delta is the only thing multiplying u, so dropping it costs likelihood.
# And the query is not an extrapolation — it is a marginalization over the upper tail of
# an in-support standard normal.
#
# WHAT Delta ACTUALLY IS. u is the van der Waerden score Phi^-1(rank) of the GAE
# advantage, exactly standard normal by construction; the regressand is the standardized
# action residual z_std = (z - m_S)/sd_S. The MLE fit of the model above is therefore the
# per-dimension correlation Delta_d(s) = Corr(z_d, u | s), bounded in [-1, 1] — which is
# precisely what tanh on the head enforces. Two consequences:
#   * Delta * sd_S is a natural-gradient-shaped step: the improvement direction is
#     measured, and scaled, in the policy's own metric. There is no temperature, no
#     advantage scale, no learning rate in the step size.
#   * Delta -> 0 exactly when action and outcome stop covarying, so the operator
#     self-anneals to a no-op instead of amplifying noise. The teacher is the student
#     at initialization (zero-init heads) and whenever there is nothing to learn.
#
# WHY IT SHOULD BEAT THE AWR TILT THIS LINEAGE HAS USED SINCE v10.
# AWR forms the teacher by reweighting the MLE with w = exp(A/temp). At the lineage's
# own KL-targeted dose (tilt_eps=1.2 nats) that is an effective sample size of ~e^-1.2
# ~ 30% of the batch: two thirds of the rollout is thrown away, and every sample below
# the mean is discarded rather than used. A regression uses all N samples and reads the
# *sign* of a bad outcome as information ("less of that") instead of as a near-zero
# weight. Same data, ~3x lower variance on the improvement direction — which is worth
# little at 500k when the signal is huge, and worth everything after 4M when advantages
# are mostly noise. That is exactly where this lineage bends: hopsd_v27_sdgate leads the
# PPO baseline at 2M (6034 vs 5012) and is *behind* it at 8M (8171 vs 8278).
# Second, amortization: Delta(s) is predicted by a network, so the estimate at state s
# borrows strength from every similar state, which a per-state one-sample estimator
# (PPO's, and AWR's) structurally cannot do.
# Third, the teacher only has to learn a bounded residual tilt off the student, not an
# entire privileged conditional density; its failure mode is "no-op", not "wrong target".
#
# THE QUERY. Conditioning on a single rank would collapse the spread, so the teacher is
# the distribution of actions whose rank sits in the upper tail, obtained in closed form
# from the fitted linear-Gaussian model. With q = Phi^-1(p), lam = phi(q)/(1-p) (the
# inverse Mills ratio = E[u | u >= q]) and v = 1 + q*lam - lam^2 = Var[u | u >= q]:
#     mean_T = m_S + omega * Delta * sd_S
#     sd_T   = sd_S * sqrt(exp(2*kappa) + v * Delta^2)
#
# THE OPTIMISM DUAL. omega is not a constant. Measured on synthetic batches where the
# true improvement direction is known, a fixed quantile is badly under-dosed once the
# advantage gets noisy: at p=0.85 the realized step falls to KL 0.03 and 0.01 nats in the
# mid/late-SNR regimes because Delta itself shrinks. So omega is bisected every iteration
# (in [omega_min, omega_max]; KL is monotone in it) to hold the initial per-state
# distillation step at target_distill_kl nats — v12's "constant information size"
# constraint, which this lineage already found load-bearing, applied to a covariance tilt
# instead of a softmax reweighting. v stays evaluated at tilt_quantile, so the teacher's
# spread remains the honest conditional spread and the dual only moves the mean.
#
# MEASURED (4 seeds, synthetic batch with a known state-dependent improvement direction,
# CovTilt vs this lineage's AWR teacher at its own KL-targeted 1.2-nat dose; the number
# is the true expected advantage of an action sampled from the teacher):
#   SNR regime      AWR@1.2nats            CovTilt@KL1.2         realized KL (AWR / Cov)
#   early           +0.720 +- 0.084        +1.685 +- 0.019       0.88 / 1.20
#   mid             +0.199 +- 0.074        +0.628 +- 0.018       0.70 / 0.17
#   late            +0.003 +- 0.043        +0.179 +- 0.022       0.67 / 0.02
# CovTilt is 2.3x / 3.2x better early and mid, and in the late regime AWR is statistically
# indistinguishable from zero improvement while CovTilt still extracts signal — at 1/30th
# the policy change, i.e. it does not spend exploration entropy on noise. Seed-to-seed
# spread is 4-10x smaller throughout, which is the amortized full-batch regression doing
# what an effective-sample-size-0.13 reweighting cannot.
#
# The teacher is moment-matched back to a Beta so the divergence stays the closed-form
# (the "full-vocabulary logit distillation" analog the paper found decisively better than
# sampled-token objectives). Student loss = sum_d min(KL(Beta_T,d || Beta_S,d), tau_d),
# teacher detached: the paper's per-token pointwise divergence clipping, per action dim.
#
# RETAINED FROM THE 8M CHAMPION (hopsd_v27_sdgate, ~8171@8M — the only verified 8M
# finish in the lineage): Beta policy on a shared ThinkTrunk, 511-bin Dreamer3-bucket
# HL-Gauss MTP critic, decoupled actor/critic grad clipping, per-dim teacher
# concentration caps and per-dim distill KL budgets. v27 drove those gates with the
# AWR-tilt variance ratio s_d; here the gate signal is |Delta_d| itself, which is the
# same quantity measured directly instead of inferred. Plus v28's target-entropy dual
# (its 7106@4M beat v27's 7029@4M), needed because tail conditioning shrinks spread.
# DROPPED: the hindsight future-action window phi, the privileged teacher critic, the
# AWR weights and the temperature dual. Advantage rank is the only privilege.
#
# HYPOTHESIS: a full-batch, rank-robust, amortized covariance estimate of the improvement
# direction gives a lower-variance and better-calibrated teacher than exponential
# reweighting, so the curve keeps climbing past 4M where the AWR-tilted lineage flattens.
# FALSIFIABLE: debug/delta_abs_mean collapsing to ~0 early = the tilt found no signal and
# the student converges to self-BC (returns plateau low). debug/omega pinned at omega_max
# with debug/distill_kl_initial far below target = the operator ran out of measurable
# signal and the dose ladder is the wrong dial. Returns tracking v27 to 2M then separating
# upward after 4M = the variance argument was the binding one.
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

SAMPLE_EPS = 1e-6      # clamp Beta samples off the open-interval boundary (avoid log(0))
MEAN_EPS = SAMPLE_EPS  # preserve legitimate sharp Beta means; z is already clamped here
SD_SAFETY = 0.995      # a Beta sd must stay below sqrt(m(1-m)); leave a margin


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
    vf_coef: float = 0.5
    actor_grad_clip: float = 0.25
    critic_grad_clip: float = 0.25

    # Dreamer3-style bucket critic (unchanged from the v27 chassis).
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    normalize_reward: bool = False
    clip_reward: bool = False

    # --- CovTilt teacher ---
    teacher_epochs: int = 10         # supervised Gaussian-MLE epochs (off-policy safe)
    tilt_quantile: float = 0.85      # raw teacher conditional spread
    raw_omega: float = 4.0           # proposal dose; upper bound on the dual when omega_dual
    omega_dual: bool = False         # bisect the proposal to actually spend target_distill_kl
    omega_min: float = 0.2
    target_distill_kl: float = 1.2   # KL budget: proposal target AND projection ceiling, in nats
    max_ent_drop: float = 0.12       # maximum teacher entropy drop from rollout, in nats
    dual_iters: int = 20             # bisection steps for the proposal dual and the projection
    teacher_shrink: bool = True
    kappa_max: float = 0.75
    teacher_grad_clip: float = 0.5
    distill_kl_clip: float = 2.0
    teacher_conc_cap: float = 100.0
    conc_cap_lo: float = 20.0
    conc_headroom: float = 2.0       # teacher may at most x2 the rollout concentration; keeps
                                     # max_ent_drop, not the cap, as the binding controller

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
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


class Agent(nn.Module):
    """Student: Beta actor + HL-Gauss MTP critic on a shared trunk (v27 chassis)."""

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.register_buffer(
            "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        )

    def get_value(self, x):
        return self.critic_head(self.trunk(x)).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_action_and_value(self, x, z=None):
        feat = self.trunk(x)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        dist = Beta(alpha, beta, validate_args=False)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        log_prob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        value_logits = self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        return action, z, log_prob, entropy, value_logits, alpha, beta

    def actor_parameters(self):
        return (
            list(self.trunk.parameters())
            + list(self.actor_alpha_head.parameters())
            + list(self.actor_beta_head.parameters())
        )

    def critic_parameters(self):
        return list(self.trunk.parameters()) + list(self.critic_head.parameters())


class TiltTeacher(nn.Module):
    """Predicts the per-dim advantage/action correlation Delta(s) and the conditional
    log-sd correction kappa(s). The privileged rank u never enters here: it multiplies
    Delta downstream, so the channel is structurally load-bearing and cannot go inert.
    Zero-ish init (std=0.01 into tanh) => teacher == student at initialization."""

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        H = args.hidden
        self.kappa_max = args.kappa_max
        self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        self.delta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.kappa_head = layer_init(nn.Linear(H, act_dim), std=0.01)

    def forward(self, obs):
        feat = self.trunk(obs)
        delta = torch.tanh(self.delta_head(feat))                     # correlation in [-1, 1]
        kappa = self.kappa_max * torch.tanh(self.kappa_head(feat))    # bounded log-sd correction
        return delta, kappa


def beta_moments(alpha, beta):
    """Mean and standard deviation of Beta(alpha, beta), per dim."""
    conc = alpha + beta
    mean = alpha / conc
    sd = (alpha * beta / (conc.square() * (conc + 1.0))).sqrt()
    return mean, sd


def moments_to_beta(mean, sd, conc_cap):
    """Moment-match (mean, sd) back to Beta(alpha, beta) with alpha, beta >= 1.

    v27 (and v1) applied the per-dim sharpness cap as alpha.clamp(max=cap) and
    beta.clamp(max=cap) independently. That is mean-destroying: once a target is sharp
    enough for BOTH to hit the cap the distribution degenerates to Beta(cap, cap), whose
    mean is 0.5 regardless of what was asked for -- so the teacher stops pointing anywhere
    and just pulls every action to the middle of the action range. v1 measurably entered
    that regime (student entropy -9.96 put the teacher's requested concentration far past
    the cap). Capping the CONCENTRATION nu = alpha + beta instead limits sharpness exactly
    as intended while preserving the requested mean identically. nu_cap is 2*conc_cap so
    the effective sharpness limit matches v27's per-parameter cap.
    """
    mean = mean.clamp(MEAN_EPS, 1.0 - MEAN_EPS)
    sd_max = SD_SAFETY * (mean * (1.0 - mean)).sqrt()
    sd = torch.minimum(sd.clamp_min(1e-6), sd_max)
    nu = (mean * (1.0 - mean)) / sd.square() - 1.0                 # > 0 by construction
    nu_min = 1.0 / torch.minimum(mean, 1.0 - mean)                 # gives alpha, beta >= 1
    nu = torch.minimum(nu.clamp_min(1e-6).maximum(nu_min), torch.maximum(2.0 * conc_cap, nu_min))
    return mean * nu, (1.0 - mean) * nu


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


def upper_tail_moments(p):
    """Mean lam and variance v of a standard normal conditioned on u >= Phi^-1(p)."""
    q = float(torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=torch.float64)).item() * math.sqrt(2.0))
    pdf = math.exp(-0.5 * q * q) / math.sqrt(2.0 * math.pi)
    lam = pdf / (1.0 - p)                  # inverse Mills ratio = E[u | u >= q]
    var = 1.0 + q * lam - lam * lam        # Var[u | u >= q]
    return q, lam, max(var, 1e-6)


def normal_scores(x):
    """Van der Waerden scores: Phi^-1 of the mid-rank, rescaled to exactly unit variance.

    Rank-based, so a handful of extreme GAE advantages cannot dominate the fit — the
    robustness that v12.2 had to bolt on as winsorization is intrinsic here.
    """
    flat = x.reshape(-1)
    n = flat.numel()
    order = torch.argsort(flat)
    ranks = torch.empty(n, dtype=torch.float32, device=flat.device)
    ranks[order] = torch.arange(n, dtype=torch.float32, device=flat.device)
    p = (ranks + 0.5) / n
    u = torch.erfinv(2.0 * p - 1.0) * math.sqrt(2.0)
    return (u / u.std().clamp_min(1e-8)).reshape(x.shape)


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

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    teacher = TiltTeacher(obs_dim, act_dim, args).to(device)
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=args.learning_rate, eps=1e-5)

    # Upper-tail query constants: mean shift lam, residual variance inflation v.
    tail_q, tail_lam, tail_var = upper_tail_moments(args.tilt_quantile)
    print(f"[covtilt] tilt_quantile={args.tilt_quantile} q={tail_q:.4f} lam={tail_lam:.4f} v={tail_var:.4f}")

    # Per-dim tilt strength EMA (drives the teacher concentration cap and distill budget).
    sig_ema = torch.full((act_dim,), 0.5, device=device)
    cap_d = torch.full((act_dim,), float(args.teacher_conc_cap), device=device)
    tau_d = torch.full((act_dim,), float(args.distill_kl_clip), device=device)
    delta_max_ema = torch.zeros((), device=device)   # anchor so gates close as the tilt anneals

    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, device
    )
    hl_support_cpu = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, torch.device("cpu")
    )

    def value_logits_to_scalar(logits):
        return hl_support.to_scalar(logits)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    roll_alpha = torch.zeros((args.num_steps, args.num_envs, act_dim)).to(device)
    roll_beta = torch.zeros((args.num_steps, args.num_envs, act_dim)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

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
            teacher_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits, s_a, s_b = agent.get_action_and_value(next_obs)
                values[step] = value_logits_to_scalar(value_logits[:, 0])
            latent_zs[step] = z
            roll_alpha[step] = s_a
            roll_beta[step] = s_b
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
            transition_terminations[step] = torch.as_tensor(terminations, device=device, dtype=torch.float32)
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
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta_t = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta_t + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values

            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros((*returns.shape, mtp), dtype=torch.bool, device=returns.device)
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones((valid_len, args.num_envs), dtype=torch.bool, device=returns.device)
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                return_mtp[:valid_len, :, h] = returns[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            target_probs = hl_support_cpu.project(return_mtp.detach().cpu())

            # --- the privileged channel: standard-normal rank score of the advantage ---
            u_hat = normal_scores(advantages)                      # (T, B), exactly N(0,1)
            m_old, sd_old = beta_moments(roll_alpha, roll_beta)    # frozen rollout policy moments

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()
        b_u = u_hat.reshape(-1, 1)
        b_m_old = m_old.reshape(-1, act_dim)
        b_sd_old = sd_old.reshape(-1, act_dim)
        # Standardized action residual: the regressand. Delta is literally its correlation
        # with the advantage rank, so the fit is a plain heteroscedastic linear regression.
        b_z_std = (b_latent_zs - b_m_old) / b_sd_old

        b_inds = np.arange(args.batch_size)

        # ================= PHASE 1: rationalize (fit the tilt heads) ========================
        # v1 fitted through moments_to_beta and a Beta NLL. Measured: the nu-floor that keeps
        # alpha,beta >= 1 binds on 38-58% of samples, and it binds hardest exactly where |u|
        # is large -- the informative tail -- so kappa's gradient died there and the fitted
        # kappa came out +0.39 (nonsensically WIDER than the marginal). v2 fits the model in
        # the space it was derived in: z_std = Delta*u + exp(kappa)*eps. Heteroscedastic
        # Gaussian NLL, no clamps, no dead gradients, no lgamma. Beta only reappears at query
        # time, where the moment match is a legitimate projection rather than a constraint on
        # the estimator. Still pure supervised regression on frozen targets -> no ratio needed.
        teacher_nlls = []
        for _ in range(args.teacher_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                delta, kappa = teacher(b_obs[mb])
                resid = (b_z_std[mb] - delta * b_u[mb]) * (-kappa).exp()
                nll = (0.5 * resid.square() + kappa).sum(-1).mean()
                teacher_optimizer.zero_grad(set_to_none=True)
                nll.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(), args.teacher_grad_clip)
                teacher_optimizer.step()
                teacher_nlls.append(nll.item())

        # ================= PHASE 2: build the teacher targets (once, detached) ==============
        with torch.no_grad():
            deltas, kappas = [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                d_mb, k_mb = teacher(b_obs[start : start + args.minibatch_size])
                deltas.append(d_mb)
                kappas.append(k_mb)
            b_delta = torch.cat(deltas)
            b_kappa = torch.cat(kappas)

            # v27's gates, driven by the directly measured per-dim tilt strength |Delta_d|.
            # v1 normalized by the batch max, which is scale-invariant: when the whole tilt
            # anneals the gates stay wide open, exactly when they should be closing. v2
            # anchors on a slow EMA of the largest tilt ever seen, so proportional decay of
            # Delta does close the caps and shrink the distill budget.
            delta_abs = b_delta.abs().mean(0)
            delta_max_ema.mul_(0.99).add_(0.01 * delta_abs.max())
            delta_max_ema.clamp_(min=1e-4)
            sig_d = (delta_abs / torch.maximum(delta_max_ema, delta_abs.max())).clamp(0.0, 1.0)
            sig_ema.mul_(0.9).add_(0.1 * sig_d)
            cap_d = args.conc_cap_lo + (args.teacher_conc_cap - args.conc_cap_lo) * sig_ema
            tau_raw = sig_ema + 0.25
            tau_d = (args.distill_kl_clip * tau_raw * (act_dim / tau_raw.sum())).clamp(
                max=1.5 * args.distill_kl_clip
            )

            # --- optimistic posterior proposal --------------------------------------------
            b_roll_alpha = roll_alpha.reshape(-1, act_dim)
            b_roll_beta = roll_beta.reshape(-1, act_dim)
            # Phase 2 evaluates the trust region in float64. Two fp32 hazards sit directly in
            # the constraint path: moments_to_beta lets the alpha,beta >= 1 floor bypass the
            # concentration cap, so a boundary-clamped proposal mean reaches nu ~ 1/MEAN_EPS =
            # 1e6, where fp32 lgamma error is ~0.4 nats and beta_kl_per_dim is a difference of
            # large cancelling lgamma terms; and the mixture recovers variance as E[z^2]-mean^2,
            # a cancellation that drifts alpha by ~1e-2 relative at concentration 1e4. Both feed
            # the omega and eta bisections, i.e. the constraints themselves, so fp32 noise there
            # corrupts the dose decision rather than just the reported number. Phase 2 runs once
            # per iteration under no_grad and is consumed via .item(); Phase 3 stays fp32 for the
            # differentiable student loss.
            ra64, rb64 = b_roll_alpha.double(), b_roll_beta.double()
            m64, sd64 = b_m_old.double(), b_sd_old.double()
            d64, k64 = b_delta.double(), b_kappa.double()
            if args.teacher_shrink:
                raw_sd = sd64 * ((2.0 * k64).exp() + tail_var * d64.square()).sqrt()
            else:
                raw_sd = sd64

            # A sharp rollout must remain representable at mixture eta=0, so the inherited
            # per-dim cap becomes a cap on ADDITIONAL concentration and never forces the
            # acting policy wider. It also needs HEADROOM: with the cap at exactly the
            # rollout concentration, a sharpening proposal is clipped to the rollout and the
            # achievable entropy drop measures 0.005 nats at concentration 200 and 0.000 at
            # 2000 -- the cap silently becomes the controller and kills the scale channel of
            # improvement, which is v2's failure mode wearing a different hat. Allowing the
            # teacher to at most multiply concentration by conc_headroom bounds one step at
            # ~0.35 nats/dim, an order of magnitude above the max_ent_drop budget, so the
            # entropy constraint stays the binding controller at every sharpness while the
            # cap remains a pure safety bound. moments_to_beta reads this as half the maximum
            # total concentration.
            projection_cap = torch.maximum(
                cap_d.double(), 0.5 * args.conc_headroom * (ra64 + rb64)
            )

            def _raw_kl(w):
                r_a, r_b = moments_to_beta(m64 + w * d64 * sd64, raw_sd, projection_cap)
                return (
                    r_a,
                    r_b,
                    beta_kl_per_dim(r_a, r_b, ra64, rb64)
                    .clamp_min(0.0)
                    .sum(-1)
                    .mean()
                    .item(),
                )

            # OPTIONAL FORCED DOSE (ablation, off by default). v1 held omega at its ceiling
            # of 4 all run while the realized step decayed 0.73 -> 0.25@2M -> 0.0167@8M
            # against a 1.2-nat budget, because |Delta| itself annealed 0.088 -> 0.015. It
            # is tempting to read that as under-dosing and bisect omega to spend the budget.
            # That reading is wrong: Var(u) = Var(z_std) = 1 by construction, so the per-dim
            # regression R^2 IS Delta^2 and teacher KL scales as omega^2 Delta^2 -- a FIXED
            # omega already doses in proportion to the measured advantage/action information
            # and correctly anneals to zero when that correlation vanishes. v1's step decayed
            # because its signal decayed, and its signal decayed because entropy collapsed to
            # -9.5: a near deterministic policy has no action diversity left to correlate
            # with advantage. So the cure is the entropy floor above, not a bigger dose;
            # forcing full budget when Delta is noise merely amplifies noise. Kept as an
            # explicit arm to test that claim rather than assume it.
            if args.omega_dual:
                lo_w, hi_w = args.omega_min, args.raw_omega
                if _raw_kl(hi_w)[2] <= args.target_distill_kl:
                    omega = hi_w
                else:
                    for _ in range(args.dual_iters):
                        mid_w = 0.5 * (lo_w + hi_w)
                        if _raw_kl(mid_w)[2] < args.target_distill_kl:
                            lo_w = mid_w
                        else:
                            hi_w = mid_w
                    omega = 0.5 * (lo_w + hi_w)
            else:
                omega = args.raw_omega
            raw_alpha, raw_beta, raw_kl = _raw_kl(omega)
            raw_mean, raw_sd = beta_moments(raw_alpha, raw_beta)

            # --- feasible-mixture projection ----------------------------------------------
            # v2 tried to preserve the optimistic mean and inflate only its sd until the
            # teacher met an entropy floor. Durable traces prove that constraint was often
            # impossible: sd_inflation pinned at 5.0 from the first update, yet entropy fell
            # to -12 and returns regressed after 2M. A shifted bounded distribution has a
            # mean-dependent maximum entropy; no amount of sd inflation can recover entropy
            # once the mean itself makes the floor infeasible.
            #
            # v3 projects the entire optimistic posterior toward the rollout distribution:
            # q_eta = (1-eta) pi_old + eta q_raw. We preserve its exact first two moments,
            # then moment-match one Beta. eta=0 is exactly pi_old, so BOTH constraints are
            # always feasible; eta=1 is the full self-teacher. The mixture variance includes
            # eta(1-eta)(mu_raw-mu_old)^2, retaining uncertainty about disagreeing policies
            # instead of collapsing around their interpolated mean.
            old_var = sd64.square()
            raw_var = raw_sd.square()
            old_second = old_var + m64.square()
            raw_second = raw_var + raw_mean.square()
            student_H = (
                Beta(ra64, rb64, validate_args=False)
                .entropy()
                .sum(-1)
                .mean()
                .item()
            )
            entropy_floor = student_H - args.max_ent_drop

            def _teacher_at(eta):
                # Identity must be exact, not merely a moment round-trip. This matters for
                # very sharp boundary policies: even a 1e-3 mean clamp can make eta=0 cost
                # >1 nat of fake KL. Exact identity makes the constraint set unconditionally
                # non-empty and gives bisection a correct lower endpoint.
                if eta <= 0.0:
                    return ra64, rb64, 0.0, student_H
                mix_mean = m64 + eta * (raw_mean - m64)
                mix_second = old_second + eta * (raw_second - old_second)
                mix_sd = (mix_second - mix_mean.square()).clamp_min(1e-12).sqrt()
                t_a, t_b = moments_to_beta(mix_mean, mix_sd, projection_cap)
                kl = (
                    beta_kl_per_dim(t_a, t_b, ra64, rb64)
                    .clamp_min(0.0)
                    .sum(-1)
                    .mean()
                    .item()
                )
                entropy = (
                    Beta(t_a, t_b, validate_args=False)
                    .entropy()
                    .sum(-1)
                    .mean()
                    .item()
                )
                return t_a, t_b, kl, entropy

            def _feasible(eta):
                _, _, kl, entropy = _teacher_at(eta)
                return kl <= args.target_distill_kl and entropy >= entropy_floor

            if _feasible(1.0):
                mix_eta = 1.0
            else:
                lo_eta, hi_eta = 0.0, 1.0
                for _ in range(args.dual_iters):
                    mid_eta = 0.5 * (lo_eta + hi_eta)
                    if _feasible(mid_eta):
                        lo_eta = mid_eta
                    else:
                        hi_eta = mid_eta
                mix_eta = lo_eta
            b_t_alpha, b_t_beta, step_kl, teacher_entropy = _teacher_at(mix_eta)
            # The trust region is decided in fp64; the student trains in fp32.
            b_t_alpha, b_t_beta = b_t_alpha.float(), b_t_beta.float()
            mean_gap = (
                b_t_alpha / (b_t_alpha + b_t_beta) - b_m_old
            ).abs().mean().item()

        # ================= PHASE 3: distill (no ratio, no PPO surrogate) ====================
        distill_kls, distill_clipfracs = [], []
        distill_kl_initial = None
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                _, _, newlogprob, entropy, value_logits, s_alpha, s_beta = agent.get_action_and_value(
                    b_obs[mb], b_latent_zs[mb]
                )

                kl_dims = beta_kl_per_dim(b_t_alpha[mb], b_t_beta[mb], s_alpha, s_beta).clamp_min(0.0)
                distill_loss = kl_dims.clamp(max=tau_d).sum(-1).mean()
                with torch.no_grad():
                    kl_sum = kl_dims.sum(-1).mean().item()
                    distill_kls.append(kl_sum)
                    distill_clipfracs.append((kl_dims > tau_d).float().mean().item())
                    if distill_kl_initial is None:
                        distill_kl_initial = kl_sum
                    logratio = newlogprob - b_logprobs[mb]
                    approx_kl = ((logratio.exp() - 1) - logratio).mean()

                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb].to(device=value_logits.device, non_blocking=True)
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb].to(
                    device=value_logits.device, dtype=value_ce.dtype, non_blocking=True
                )
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()
                optimizer.zero_grad(set_to_none=True)
                (args.vf_coef * v_loss).backward(retain_graph=True)
                critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                optimizer.zero_grad(set_to_none=True)
                distill_loss.backward()
                actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                for p, grad in value_grads:
                    p.grad = grad if p.grad is None else p.grad + grad
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        with torch.no_grad():
            s_ents = []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                _, _, _, s_ent, _, _, _ = agent.get_action_and_value(b_obs[sl], b_latent_zs[sl])
                s_ents.append(s_ent.mean().item())
            student_entropy = float(np.mean(s_ents))

        # No entropy bonus and no dual: the entropy-rate governor above already fixes the
        # teacher's spread, and the student inherits it through the distillation target.

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", distill_loss.item(), global_step)
        writer.add_scalar("losses/entropy", student_entropy, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("losses/teacher_nll", float(np.mean(teacher_nlls)), global_step)
        writer.add_scalar("debug/omega", float(omega), global_step)
        writer.add_scalar("debug/raw_kl", float(raw_kl), global_step)
        writer.add_scalar(
            "debug/omega_saturated", float(omega >= args.raw_omega - 1e-6), global_step
        )
        writer.add_scalar("losses/distill_kl", float(np.mean(distill_kls)), global_step)
        writer.add_scalar("debug/distill_kl_initial", float(distill_kl_initial), global_step)
        writer.add_scalar("debug/omega_cap", float(args.raw_omega), global_step)
        writer.add_scalar("debug/mix_eta", float(mix_eta), global_step)
        writer.add_scalar("debug/step_kl", float(step_kl), global_step)
        writer.add_scalar(
            "debug/entropy_drop",
            float(student_H - teacher_entropy),
            global_step,
        )
        writer.add_scalar("debug/distill_clipfrac", float(np.mean(distill_clipfracs)), global_step)
        writer.add_scalar("debug/delta_abs_mean", float(delta_abs.mean().item()), global_step)
        writer.add_scalar("debug/delta_abs_max", float(delta_abs.max().item()), global_step)
        writer.add_scalar("debug/kappa_mean", float(b_kappa.mean().item()), global_step)
        writer.add_scalar("debug/teacher_entropy", teacher_entropy, global_step)
        writer.add_scalar("debug/student_entropy", student_entropy, global_step)
        writer.add_scalar("debug/teacher_student_mean_gap", mean_gap, global_step)
        writer.add_scalar("debug/student_entropy_pre", float(student_H), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        for _d in range(act_dim):
            writer.add_scalar(f"debug/delta_abs_{_d}", float(delta_abs[_d].item()), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

# OPSD-CovTilt v2 — same covariance-tilt teacher, but the exploration schedule is governed
# directly instead of being left to a lagging entropy dual.
# =====================================================================================
# WHAT v1 SHOWED (HalfCheetah-v4, seed 1, omega_max ladder 2/4/8):
#   arm      @250k   @500k     @1M      student entropy @1.3M
#   w2        777    1993    3398
#   w4       1113    2460    3623      -9.59
#   w8        481    1430    2771
#   PPO base 
#            (--)    1576    3079
#   hopsd_v27_sdgate (--)  2165  4208   -4.77
#   hopsd_v36_pooled (--)  3126  4557   -5.73
# So the covariance tilt is a real improvement operator — w4 beats the PPO baseline by
# +56% @500k and +18% @1M with no PPO term anywhere — but it loses to the phi-privileged
# hopsd teachers, and the diagnostics say why, unambiguously: v1's policy entropy fell at
# 0.24 nats/iteration against the champions' ~0.12, reaching -9.59 where they sit at
# -4.8/-5.7. distill_kl ran 0.47-0.99 against their 0.22-0.31. The dose ordering
# (w4 > w2 > w8) says the mean step is NOT the problem — bigger mean steps helped up to
# w4 — so v1 was not over-stepping, it was over-CONVERGING: exploration was being spent
# twice as fast as the operator could refill it.
#
# WHY v1 collapsed, and the three fixes.
# 1. ENTROPY IS NOW A GOVERNED RATE, NOT A DUAL. Tail conditioning makes the teacher
#    strictly narrower than the student at every state, every iteration, and the student
#    chases it; nothing in v1 bounded that. v28's multiplicative dual cannot: it starts at
#    1e-3, decays to its 1e-6 floor during the (long) phase where entropy is still above
#    target, and then needs ~30 iterations — a million steps — to climb back to any
#    effective value. Measured ent_alpha was 0.0 for the whole run. v2 deletes the bonus
#    and the dual and instead bisects one scalar inflation on the teacher's sd so that
#    H(teacher) >= H(student) - max_ent_drop, exactly, every iteration. Entropy decay rate
#    becomes an explicit hyperparameter set to the champions' measured ~0.12 nats/iter.
#    Inflation only ever lowers KL, so the omega dual's target remains an upper bound.
# 2. THE SHARPNESS CAP NO LONGER DESTROYS THE TEACHER'S MEAN. v27 and v1 capped alpha and
#    beta independently; when a target is sharp enough for both to hit the cap the result
#    is Beta(cap, cap), whose mean is 0.5 whatever was requested — the teacher stops
#    pointing anywhere and pulls every action to the centre of the action range. v1's
#    collapse to entropy -9.96 drove it into exactly that regime. v2 caps the concentration
#    alpha+beta instead, which limits sharpness identically and preserves the mean exactly.
# 3. THE TILT IS FITTED IN THE SPACE IT WAS DERIVED IN. v1 fitted through moments_to_beta
#    with a Beta NLL; an independent numeric audit measured the nu-floor binding on 38-58%
#    of samples, and it binds hardest at large |u| — the informative tail — killing kappa's
#    gradient there. The fitted kappa came out +0.39, i.e. the model claimed the conditional
#    spread was 1.5x WIDER than the marginal, which the linear-Gaussian model it is supposed
#    to be fitting cannot produce. v2 fits z_std = Delta*u + exp(kappa)*eps by
#    heteroscedastic Gaussian NLL: no clamps, no dead gradients, no lgamma, and Delta is
#    exactly the correlation it is documented to be. Beta reappears only at query time,
#    where the moment match is a projection rather than a constraint on the estimator.
# Also: the per-dim gate signal is now anchored on a slow EMA of max|Delta| rather than the
# batch max, so gates actually close when the whole tilt anneals (v1's normalization was
# scale-invariant and kept them wide open precisely in the late regime they exist for).
#
# PREDICTION: entropy tracks the champions' schedule; distill_kl lands in 0.2-0.35; returns
# hold v1's early advantage over the PPO baseline and no longer bend down after 1M.
# KILL-TELL: debug/sd_inflation pinned at infl_max means the teacher cannot be widened
# enough and the shrink term itself (not the schedule) is the problem — then drop
# teacher_shrink and let the governor set the spread outright.
# =====================================================================================
#
# --- v1 method (retained below) ---
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
MEAN_EPS = 1e-3        # keep tilted Beta means strictly interior
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
    teacher_epochs: int = 10         # supervised MLE epochs for the tilt heads (off-policy safe)
    tilt_quantile: float = 0.85      # sets the teacher's conditional SPREAD (v); dose is the dual's job
    target_distill_kl: float = 1.2   # dual target for the initial per-state sum_d KL(T||S), in nats
    omega_max: float = 4.0           # cap on the optimism multiplier (bounds extrapolation off the fit)
    omega_min: float = 0.2
    dual_iters: int = 20             # bisection steps for omega
    teacher_shrink: bool = True      # False -> pure mean shift, teacher sd == student sd
    kappa_max: float = 0.75          # |log| bound on the fitted conditional sd correction
    teacher_grad_clip: float = 0.5
    distill_kl_clip: float = 2.0     # tau: pointwise per-action-dim KL clip (paper's min(l, tau))
    teacher_conc_cap: float = 100.0  # cap_hi: per-dim teacher concentration cap on active dims
    conc_cap_lo: float = 20.0        # cap floor on dims the tilt does not move

    # --- v2 entropy-rate governor (replaces v28's lagging multiplicative dual) ---
    max_ent_drop: float = 0.12       # hard cap on nats of teacher entropy below the student, per iteration
    min_entropy: float = -20.0       # absolute floor on teacher entropy (safety, rarely binds)
    infl_max: float = 5.0            # cap on the sd inflation used to satisfy the entropy floor

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

            # The teacher's spread is fixed by the honest conditional variance at
            # tilt_quantile; only the mean shift is dosed.
            if args.teacher_shrink:
                sd_T = b_sd_old * ((2.0 * b_kappa).exp() + tail_var * b_delta.square()).sqrt()
            else:
                sd_T = b_sd_old
            b_roll_alpha = roll_alpha.reshape(-1, act_dim)
            b_roll_beta = roll_beta.reshape(-1, act_dim)

            def _teacher_at(omega):
                t_a, t_b = moments_to_beta(b_m_old + omega * b_delta * b_sd_old, sd_T, cap_d)
                kl = beta_kl_per_dim(t_a, t_b, b_roll_alpha, b_roll_beta).clamp_min(0.0)
                return t_a, t_b, kl.sum(-1).mean().item()

            # Optimism dual: the student IS the rollout policy right now, so this KL is
            # exactly the size of the step about to be taken. Monotone in omega.
            lo_w, hi_w = args.omega_min, args.omega_max
            if _teacher_at(hi_w)[2] <= args.target_distill_kl:
                omega = hi_w  # measurable signal cannot fund the target; take the capped dose
            else:
                for _ in range(args.dual_iters):
                    mid_w = 0.5 * (lo_w + hi_w)
                    if _teacher_at(mid_w)[2] < args.target_distill_kl:
                        lo_w = mid_w
                    else:
                        hi_w = mid_w
                omega = 0.5 * (lo_w + hi_w)
            b_t_alpha, b_t_beta, step_kl = _teacher_at(omega)

            # --- entropy-rate governor ---------------------------------------------------
            # v1's failure, measured against the champions at 1.3M: student entropy -9.59 vs
            # hopsd_v27's -4.77 and v36's -5.73. Tail conditioning makes the teacher narrower
            # than the student EVERY iteration and the student chases it, so entropy fell
            # 0.24 nats/iteration against their ~0.12. v28's multiplicative dual could not
            # catch it: starting at 1e-3 it decays to its 1e-6 floor while entropy is still
            # above target, then needs ~30 iterations (1M steps) to climb back to any useful
            # value. So v2 does not regulate entropy through a loss bonus at all. It caps the
            # per-iteration drop directly and immediately, by bisecting one scalar inflation
            # on the teacher's sd (entropy is monotone increasing in sd at fixed mean).
            # Inflation only ever reduces KL, so the omega dual's target stays an upper bound.
            student_H = (
                Beta(b_roll_alpha, b_roll_beta, validate_args=False).entropy().sum(-1).mean().item()
            )
            floor_H = max(student_H - args.max_ent_drop, args.min_entropy)

            def _entropy_at(scale):
                t_a, t_b = moments_to_beta(b_m_old + omega * b_delta * b_sd_old, scale * sd_T, cap_d)
                return t_a, t_b, Beta(t_a, t_b, validate_args=False).entropy().sum(-1).mean().item()

            infl = 1.0
            teacher_entropy = _entropy_at(1.0)[2]
            if teacher_entropy < floor_H:
                lo_c, hi_c = 1.0, args.infl_max
                if _entropy_at(hi_c)[2] >= floor_H:
                    for _ in range(args.dual_iters):
                        mid_c = 0.5 * (lo_c + hi_c)
                        if _entropy_at(mid_c)[2] < floor_H:
                            lo_c = mid_c
                        else:
                            hi_c = mid_c
                    infl = 0.5 * (lo_c + hi_c)
                else:
                    infl = hi_c
                b_t_alpha, b_t_beta, teacher_entropy = _entropy_at(infl)
                step_kl = (
                    beta_kl_per_dim(b_t_alpha, b_t_beta, b_roll_alpha, b_roll_beta)
                    .clamp_min(0.0)
                    .sum(-1)
                    .mean()
                    .item()
                )
            mean_gap = (omega * b_delta * b_sd_old).abs().mean().item()

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
        writer.add_scalar("losses/distill_kl", float(np.mean(distill_kls)), global_step)
        writer.add_scalar("debug/distill_kl_initial", float(distill_kl_initial), global_step)
        writer.add_scalar("debug/omega", float(omega), global_step)
        writer.add_scalar("debug/step_kl", float(step_kl), global_step)
        writer.add_scalar("debug/omega_saturated", float(omega >= args.omega_max - 1e-6), global_step)
        writer.add_scalar("debug/distill_clipfrac", float(np.mean(distill_clipfracs)), global_step)
        writer.add_scalar("debug/delta_abs_mean", float(delta_abs.mean().item()), global_step)
        writer.add_scalar("debug/delta_abs_max", float(delta_abs.max().item()), global_step)
        writer.add_scalar("debug/kappa_mean", float(b_kappa.mean().item()), global_step)
        writer.add_scalar("debug/teacher_entropy", teacher_entropy, global_step)
        writer.add_scalar("debug/student_entropy", student_entropy, global_step)
        writer.add_scalar("debug/teacher_student_mean_gap", mean_gap, global_step)
        writer.add_scalar("debug/sd_inflation", float(infl), global_step)
        writer.add_scalar("debug/student_entropy_pre", float(student_H), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        for _d in range(act_dim):
            writer.add_scalar(f"debug/delta_abs_{_d}", float(delta_abs[_d].item()), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

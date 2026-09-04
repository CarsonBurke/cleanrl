# OPSD-SFOcc ActTeach FullBeta v2 -- exact full-family action-space projection.
# No PPO: the actor still receives only hindsight clone and detached teacher distillation.
# =====================================================================================
# V1 RESULT AND FAILURE
#
# The fixed-concentration action teacher produced the first substantial positive OPSD-SFOcc
# result, peaking around 900 return near 1M steps, but it then collapsed to 9 by 8M. The SF
# regression did not collapse: post-update psi_r2 stayed around 0.8 and w_r2 around 0.99.
# The teacher geometry did:
#
#   step                 0.79M       4.00M       8.00M
#   delivered KL         0.0638      0.0131      0.0031
#   KL shortfall frac    0.549       0.945       0.994
#   parameter boundary   0.803       0.960       0.983
#   follow alignment     0.412       0.135       0.013
#
# V1 preserved concentration while requiring alpha,beta >= 1. That confined the teacher
# mean to [1/k, 1-1/k]. Its scale search also stopped at 4.0. Scale reached 3.99 while the
# action-gradient remained finite, so the nominal always-active 0.10-KL teacher silently
# became an almost-zero teacher. Shortfall exceeded 50% before the return peak.
#
# ONE CORRECTION
#
# Optimize in the full two-parameter Beta family. For each action dimension:
#
#   theta = [alpha, beta]
#   g     = d(w . psi(s, action(mean(theta)))) / d theta
#   d     = F(theta)^-1 g
#
# F is the exact 2x2 Beta Fisher block. The per-state direction is normalized to unit
# Fisher norm across all action dimensions. A dynamically bracketed exact forward-KL
# projection then moves both alpha and beta along that direction, clamping only the policy
# family's alpha,beta >= 1 invariant. Moving concentration lets the mean continue toward
# an improving boundary after one parameter reaches 1. The bracket expands until it
# contains 0.10 KL; failure is explicit, never converted into weakened teacher pressure.
#
# The actor still sees two fixed supervised targets only:
#   1. hindsight rationalization NLL at the observed occupancy residual;
#   2. forward KL from this detached SF-optimized teacher into the zero-context student.
# No value, advantage, ratio, policy-gradient term, fidelity gate, warm-up, or dose schedule
# enters the actor.
#
# UPDATE ORDER
#
# The occupancy channel is built before updating SF statistics, preserving v3's
# no-statistics-mismatch invariant. The SF optimizer then fits fixed detached targets, the
# full-Beta teacher is frozen from that updated critic, and the actor runs last.
#
# EXACT-DOSE CONTRACT
#
# Nonzero critic gradients must deliver total KL 0.10 per state to numerical bisection
# precision. Exact-zero gradients naturally deliver zero. `teacher_kl_abs_error` measures
# projection accuracy; `teacher_bracket_expansions` exposes search cost;
# `teacher_concentration_ratio` and `teacher_log_concentration_change` show how much
# concentration movement was required; `teacher_boundary_frac` now means the fraction of
# alpha/beta parameters at their floor, not fixed-concentration mean infeasibility.
#
# FINAL FIX DECISION
#
# HalfCheetah-v4, seed 1, 8M steps. This is the one permitted corrective run; no sibling,
# coefficient ladder, or extra seed.
#
#   T1 EXACT PRESSURE: snapshot KL stays at 0.10, KL error remains numerical, and shortfall
#      stays zero except exact-zero critic gradients.
#   T2 OPTIMIZER POINTS UP: predicted gain is positive and nonpositive-gain fraction near
#      zero. This is internal consistency, not held-out validation of the action derivative.
#   T3 POLICY FOLLOWS: post-update teacher/student KL and mean gap fall, with positive
#      follow progress and alignment. Persistent failure here isolates actor interference.
#   T4 CRITIC SUPPORT: post-update teacher psi_r2 and w_r2 remain above 0.5.
#   T5 RETURN: improvement must remain positive and avoid v1's post-1M decay. A transient
#      early peak is not success.
#
# Remaining risk: on-policy value fit does not validate the critic's action derivative.
# This run fixes the directly observed geometric failure without hiding that separate risk.
# =====================================================================================
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
    # per-env seeding. Inherited unchanged from the parent, where the acting forward was
    # measured LAUNCH-bound rather than compute-bound. This file's acting path is strictly
    # cheaper than the parent's -- the MTP critic head is gone, so `act` is policy heads
    # only -- and the update path no longer pays for a per-sample autograd Jacobian or a
    # 6x511-bin softmax. Measured, batch held at 32768 so only env count varies, 3 repeats,
    # cumulative @65536 steps / marginal between iterations (parent convention):
    #   16 envs 9451 / 11678   32 envs 10222 / 13664   64 envs 12327 / 17035
    # against the parent's 4300 at 16 envs and 6013 at 128. GPU was shared throughout.
    async_envs: bool = True
    compile_act: bool = True
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95      # NO GAE IN THIS FILE. Retained under its historical name
    #                               solely as the default horizon that sf_lambda < 0 reuses,
    #                               so the feature-space lambda-return introduces no new knob
    #                               and stays numerically comparable to the lineage.
    num_minibatches: int = 32
    actor_epochs: int = 1         # passes over the batch for the rationalization + distill
    #                               losses. Reuse here re-fits the SAME action per state, so
    #                               it sharpens the conditional (entropy drops).
    critic_epochs: int = 4        # passes for the SF regression only. Same rationale as the
    #                               parent's split budget, and it survives the critic swap
    #                               unchanged: the SF losses are plain supervised regression
    #                               onto FIXED detached targets, so extra passes only reduce
    #                               fitting error, and they no longer touch the actor trunk
    #                               at all -- so unlike the parent's shared-trunk critic they
    #                               cannot fight the policy update.

    # --- OPSD hindsight channel -------------------------------------------------------
    cond_mode: str = "occ"        # "occ" | "occ_w"; only controls the rationalization
    #                               context. The improvement teacher no longer queries this
    #                               channel off-support.
    cond_clip: float = 3.0        # clamp on the observed standardized hindsight channel
    cond_ema_beta: float = 0.99   # EMA horizon for its per-dim RMS
    cond_embed_freqs: int = 8     # sinusoidal features per phase, occ_w only
    clone_coef: float = 1.0       # hindsight rationalization p(a_taken | s, c_observed)
    distill_coef: float = 1.0     # detached SF action-teacher -> student KL
    sf_teacher_kl: float = 0.10   # fixed total KL radius per state. One value, no sweep.

    # --- v1: SUCCESSOR-FEATURE OCCUPANCY CRITIC ------------------------------------
    phi_dim: int = 32             # Dp. 32 is 2x HalfCheetah's action dim and ~2x its obs
    #                               dim, i.e. enough basis to span the reward AND retain
    #                               geometry, small enough that the covariance telemetry
    #                               (eigvalsh on 32x32) is free.
    sf_hidden: int = 0            # 0 = use args.hidden. Same width as the trunk, so the SF
    #                               heads are a rounding error next to the env loop.
    sf_lambda: float = -1.0       # < 0 = reuse gae_lambda. The feature-space lambda-return's
    #                               horizon. Kept as a knob because it is the one quantity
    #                               that trades channel action-attributability (short) against
    #                               occupancy horizon (long) -- but it is NOT swept here.
    sf_coef: float = 0.5          # psi TD regression. Half the reward term because psi's
    #                               target has the larger magnitude by ~1/(1-gamma*lam) and
    #                               would otherwise dominate the shared phi gradient.
    sf_rew_coef: float = 1.0      # w . phi -> r. This is what makes w mean anything, and w
    #                               IS the improvement direction, so it gets full weight.
    sf_cov_coef: float = 1.0      # phi decorrelation. Full weight: rank collapse is one of
    #                               the two named hazards and the penalty is bounded in
    #                               [0, 1] by construction (it is squared correlations), so
    #                               it cannot outrun the regression terms.
    sf_grad_clip: float = 0.5     # matches max_grad_norm; separate knob because the SF
    #                               module has its own optimizer and must not be able to
    #                               change what the actor's clip does.

    max_grad_norm: float = 0.5

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
    """Fixed Fourier features of a SCALAR privileged channel. Used only by
    --cond-mode occ_w, where the channel is the occupancy surprise's projection onto w.

    A single raw scalar channel among 17 observation dims is trivially IGNORABLE: the
    rationalization loss can be driven down almost entirely by modelling the marginal
    p(a|s) and dropping it, because the extra likelihood it buys is small. Measured on
    HalfCheetah with raw scalar conditioning: cond_gap 0.001 and distill_kl 0.0001, i.e.
    the teacher WAS the student and the method was a no-op. Sinusoidal features fix this
    the way diffusion timestep embeddings do -- the scalar now occupies many channels and
    separates nearby values at high frequency, so it is both easy to use and expensive to
    ignore. Frequencies are fixed, not learned, so the channel cannot be switched off by
    driving weights to zero.

    RETAINED DELIBERATELY, because it makes occ_w the STRONGEST possible scalar arm. If
    the vector channel wins anyway, the win cannot be attributed to the scalar arm having
    been given a lazy encoding.
    """

    def __init__(self, n_freq):
        super().__init__()
        self.n_freq = n_freq
        self.freqs: torch.Tensor
        self.register_buffer("freqs", torch.logspace(-1.0, 3.0, n_freq, base=2.0))

    @property
    def dim(self):
        return 2 * self.n_freq

    def forward(self, chan):
        x = chan * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


# =====================================================================================
# BLOCK A -- THE phi / psi / w MODULE. Sibling arms replace pieces of this and nothing
# else, so it is kept whole and self-contained.
# =====================================================================================
class SFCritic(nn.Module):
    """Successor features on RAW obs+action: phi(s,a), psi(s,a), and the reward readout w.

    DELIBERATELY NOT ON THE SHARED TRUNK, for exactly the reason the parent gave for its
    TransitionHead: routing these gradients through the actor's representation would
    confound "the occupancy channel helps" with "predicting the future is a good auxiliary
    task", and this family has already been burned by that class of confound. Separate
    module, separate optimizer, separate grad clip, and the isolation is MEASURED (see the
    header's GRADIENT ISOLATION line) rather than assumed.

    phi ends in a NON-AFFINE LayerNorm. Two facts about that, both load-bearing:
      - It is a PER-SAMPLE operation, so it introduces no batch-statistic nondeterminism
        and no train/eval divergence -- unlike BatchNorm, which would make phi depend on
        who else is in the minibatch and quietly couple the channel to the shuffle.
      - It kills SCALE collapse (phi cannot shrink toward zero to cheat the decorrelation
        penalty, nor blow up to cheat the reward fit) but NOT RANK collapse: a LayerNormed
        output can still live on a 1-dim curve. Rank is the decorrelation penalty's job,
        and even that is necessary-not-sufficient (uncorrelated noise has full rank).

    psi has no LayerNorm -- it is an unbounded discounted SUM of phi's, so normalizing it
    would destroy exactly the magnitude information the value readout needs -- and its
    final layer starts at std=0.01 so the bootstrap begins near zero rather than injecting
    a random occupancy field into iteration 1's channel.

    The channel's standardization buffers mirror TransitionHead.update_stats' EMA pattern
    for its stated reason: MuJoCo observation blocks differ ~26x in action sensitivity
    (measured median |ds/da| 0.0052 on qpos rows vs 0.1359 on qvel rows), so an
    unstandardized channel would be dominated by one block and the trunk would see 32
    channels of which a few carry all the amplitude.
    """

    def __init__(self, obs_dim, act_dim, phi_dim, hidden):
        super().__init__()
        self.phi_dim = phi_dim
        self.phi_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, phi_dim)),
            nn.LayerNorm(phi_dim, elementwise_affine=False),
        )
        self.psi_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, phi_dim), std=0.01),
        )
        # NOT zero-initialized, unlike the parent's critic head: w_unit = w/||w|| is the
        # improvement direction and must be well defined at iteration 1. PyTorch's default
        # uniform init is exactly the right thing here -- a random unit direction that the
        # reward regression then rotates into place (measured: w_r2 -0.68 -> 0.956 in 16
        # iterations, with sf_rew_mse 0.1135 -> 0.0056).
        self.w_head = nn.Linear(phi_dim, 1, bias=True)
        # Per-dim mean SQUARE of the channel, not mean and variance. The family measured
        # that a conditioning input must keep its natural zero: c = 0 means "the future came
        # out exactly as predicted", and subtracting a batch/EMA mean relabels the
        # least-bad row of an all-bad batch as positive -- a false sign. Measured on the
        # champion: ema_rms 5365 @2M vs batch-standardized 4390 vs raw 2368.
        self.c_ms: torch.Tensor
        self.register_buffer("c_ms", torch.ones(phi_dim))
        # v2: PSI TARGET STANDARDIZATION. v1 measured psi_r2 -1.7557 against a >0.5 bar with
        # psi_bias_frac 0.5974, i.e. 60% of the residual was pure per-dim LEVEL error. The
        # cause is arithmetic, not a modelling failure: LayerNorm centers phi ACROSS dims,
        # so each phi dim keeps a nonzero per-dim mean, and the lambda-return's fixed point
        # multiplies it by 1/(1 - gamma*lam) ~ 17.5. A head initialized at std=0.01 outputs
        # ~0 and therefore has to travel ~17 per dim on 512 Adam steps before R2 can even
        # reach 0. So psi_net now predicts in STANDARDIZED target space and the level is
        # supplied by a buffer -- the head starts at R2 = 0 (predicting the target's mean)
        # instead of R2 = -3.3, and only has to learn VARIATION, which is what R2 scores.
        # This follows the family's own TransitionHead precedent, and for the same stated
        # reason: an unstandardized MSE over dims whose scales differ fits only the loud ones.
        # Mean-and-variance here, unlike c_ms's mean-square-only: a REGRESSION TARGET has no
        # natural zero to protect, whereas the CHANNEL does (c = 0 means "the future came out
        # as predicted") -- that is why the two use different standardizations.
        self.psi_mu: torch.Tensor
        self.psi_var: torch.Tensor
        self.register_buffer("psi_mu", torch.zeros(phi_dim))
        self.register_buffer("psi_var", torch.ones(phi_dim))

    def phi(self, obs, action):
        return self.phi_net(torch.cat([obs, action], dim=-1))

    def psi_std(self, obs, action):
        """psi in STANDARDIZED target space. This is what the regression loss compares, so
        every feature dim contributes equally regardless of its level or scale."""
        return self.psi_net(torch.cat([obs, action], dim=-1))

    def psi(self, obs, action):
        """psi in RAW feature-occupancy units -- the public readout. Every consumer (the
        bootstrap in BLOCK B, the channel in BLOCK C, the value readout) uses this, so v2
        changes the PARAMETERIZATION of psi and its loss scaling, never its semantics."""
        return self.psi_std(obs, action) * self.psi_std_dev() + self.psi_mu

    def psi_std_dev(self):
        return self.psi_var.sqrt().clamp_min(1e-6)

    def standardize_psi_target(self, tgt):
        return (tgt - self.psi_mu) / self.psi_std_dev()

    @torch.no_grad()
    def update_psi_stats(self, tgt, beta):
        """Per-dim mean and variance of psi's target. Called with beta=0 (pure batch
        statistics) from a single site placed AFTER the channel is built; see the call site
        for why both of those are load-bearing and were wrong in v2. The beta argument is
        kept so the EMA form stays available without a signature change.

        NOT PopArt: the output layer is not compensated when the stats move. That is
        deliberate and it is safe here for a measured reason -- phi is LayerNorm'd, so the
        target's fixed-point scale is bounded by phi's scale times 1/(1 - gamma*lam) and the
        statistics are near-stationary; the one large update is the iteration-1 warmup, which
        precedes any learning. If sf/psi_r2 ever shows a sawtooth synchronized with the EMA,
        PopArt-style weight compensation is the escalation, not a larger beta."""
        self.psi_mu.mul_(beta).add_((1.0 - beta) * tgt.mean(0))
        self.psi_var.mul_(beta).add_((1.0 - beta) * tgt.var(0))

    def value(self, obs, action):
        """V(s,a) = w . psi(s,a). Linear in the occupancy measure BY CONSTRUCTION, which is
        the entire point: grad_psi V = w, constant, never vanishing.

        This is the module's public value readout and part of the contract the sibling arms
        copy. The update loop deliberately does NOT call it: losses/explained_variance needs
        the SAME linear functional applied to psi AND to psi's target, and psi is already in
        hand there, so the telemetry applies self.w_head directly rather than paying for a
        second psi forward over the whole batch."""
        return self.w_head(self.psi(obs, action)).squeeze(-1)

    def w_vec(self):
        """The reward direction in feature space, detached. Improvement points along w."""
        return self.w_head.weight.detach().reshape(-1)

    def chan_std(self):
        return self.c_ms.sqrt().clamp_min(1e-6)

    @torch.no_grad()
    def update_chan_stats(self, c_raw, beta):
        self.c_ms.mul_(beta).add_((1.0 - beta) * c_raw.square().mean(0))


class Agent(nn.Module):
    """One network, two contexts. The privileged block is the TRAILING input channels.

    Present  -> the standardized occupancy surprise (occ), or Fourier features of its
                projection onto w (occ_w). Hindsight rationalization context.
    Absent   -> that whole block is zeroed. Student and acting context.
    Zeroing rather than feeding a zero-valued channel through an embedding keeps "no
    privileged information" a distinct code, since cos(0)=1 means the embedding of zero is
    not the zero vector. In occ mode zeroing and feeding zero coincide because c = 0 already
    means "the future came out as predicted." The SF action teacher is external to this
    context split: it is a detached distribution built in BLOCK D.

    NO CRITIC HEAD. Value is w . psi from SFCritic, so the trunk carries no value
    regression at all. That removes the parent's measured shared-trunk conflict (four
    passes of actor gradient left its critic WORSE: value_loss 24.9 vs 20.5) as a matter of
    structure rather than of tuning.
    """

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        if args.cond_mode == "occ":
            self.adv_embed = None
            self.cond_dim = args.phi_dim
        elif args.cond_mode == "occ_w":
            self.adv_embed = AdvEmbed(args.cond_embed_freqs)
            self.cond_dim = self.adv_embed.dim
        else:
            raise ValueError(f"unknown cond_mode {args.cond_mode!r}")
        self.trunk = ThinkTrunk(obs_dim + self.cond_dim, H, args.k_blocks, args.n_experts)
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

    def cond_present(self, chan):
        """Privileged context. occ: the standardized occupancy-surprise vector itself.
        occ_w: Fourier features of its scalar projection onto w. Clamping happens where the
        channel is built, so this is pure encoding."""
        return chan if self.adv_embed is None else self.adv_embed(chan)

    def _zero_cond(self, obs):
        """Privileged context ABSENT: the whole block is zero."""
        return obs.new_zeros((obs.shape[0], self.cond_dim))

    def policy(self, obs, cond):
        feat = self._feat(obs, cond)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        return alpha, beta

    def z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def act(self, obs):
        """Acting policy == the STUDENT context (privileged slot zeroed)."""
        alpha, beta = self.policy(obs, self._zero_cond(obs))
        distribution = Beta(alpha, beta, validate_args=False)
        z = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        return self.z_to_action(z), z

    def mean_action(self, obs):
        """a_bar(s): the STUDENT's mean action. This is what psi bootstraps on, and it is
        what future_pred is evaluated at -- so the channel's predicted term carries no
        information about the action actually taken. The Beta mean is used rather than a
        sample so the channel is a deterministic function of the state, adding no sampling
        noise on top of the environment's own."""
        alpha, beta = self.policy(obs, self._zero_cond(obs))
        return self.z_to_action(alpha / (alpha + beta))


def _chunked(fn, *tensors, chunk):
    """Apply `fn` over row chunks, keeping peak activation memory flat in batch size."""
    n = tensors[0].shape[0]
    return torch.cat([fn(*[t[i : i + chunk] for t in tensors]) for i in range(0, n, chunk)])


def phi_corr(phi):
    """Batch CORRELATION matrix of phi. Correlation, not covariance: LayerNorm already
    fixes scale, so penalizing covariance would double-charge for amplitude."""
    x = phi - phi.mean(0, keepdim=True)
    sd = x.square().mean(0).sqrt().clamp_min(1e-6)
    xn = x / sd
    return (xn.T @ xn) / x.shape[0]


def offdiag_decorr(phi):
    """Mean squared OFF-DIAGONAL entry of phi's batch correlation matrix.

    The diagonal is not penalized because it is identically 1 as a FUNCTION of phi, so its
    contribution carries exactly zero gradient and would only add a constant. Bounded in
    [0, 1], which is why sf_cov_coef can be 1.0 without any risk of outrunning the
    regression terms.

    THERE IS A HARD STRUCTURAL FLOOR AND IT MUST BE QUOTED WHENEVER THIS TAG IS READ.
    phi's non-affine LayerNorm forces sum_i phi_i(x) = 0 for EVERY sample, so
    Var(sum_i phi_i) = 0, so the off-diagonal correlations must sum to -d: at equal per-dim
    variance the mean off-diagonal correlation is exactly -1/(d-1). For phi_dim = 32 that
    puts mean|offdiag| >= 1/31 = 0.0323 and this penalty >= 1/31^2 = 0.00104 no matter what
    the network does. So a small value here means "decorrelated up to the constraint", NOT
    "the penalty did all the work", and sf/phi_offdiag_absmean must be compared to 0.0323
    rather than to 0."""
    corr = phi_corr(phi)
    d = corr.shape[0]
    off = corr - torch.diag_embed(torch.diagonal(corr))
    return off.square().sum() / (d * (d - 1))


def _ls_r2(feats, targets, ridge=1e-6):
    """Pooled R2 of the best LINEAR predictor of `targets` from `feats` (+ intercept).

    Float64 normal equations with a TRACE-RELATIVE ridge, so the number is scale-free and
    stays finite when a channel dim is degenerate -- which matters because these are the
    file's leakage and information gates and a NaN there would read as "no finding".
    In-sample by construction; with <= 50 features against 4096-32768 rows the optimistic
    bias is O(p/n) <= 1.2%, well below the thresholds it is compared against."""
    x = torch.cat([feats, torch.ones_like(feats[:, :1])], dim=-1).double()
    y = targets.double()
    xtx = x.T @ x
    xtx = xtx + (ridge * torch.diagonal(xtx).mean()) * torch.eye(
        xtx.shape[0], device=xtx.device, dtype=xtx.dtype
    )
    coef = torch.linalg.solve(xtx, x.T @ y)
    sse = (y - x @ coef).square().sum()
    sst = (y - y.mean(0, keepdim=True)).square().sum().clamp_min(1e-12)
    return float((1.0 - sse / sst).item())


def _r2_per_dim(pred, target):
    """R2 of `pred` against `target`, averaged over the last dimension."""
    sse = (target - pred).square().sum(0)
    sst = (target - target.mean(0, keepdim=True)).square().sum(0).clamp_min(1e-12)
    return float((1.0 - sse / sst).mean().item())


# =====================================================================================
# BLOCK B -- THE FEATURE-SPACE LAMBDA-RETURN (psi's regression target).
# =====================================================================================
@torch.no_grad()
def sf_feature_targets(sf, agent, b_obs, b_next_obs, b_act, nonterm, lam_nonterm, lam, gamma, chunk):
    """phi at the taken action, psi's regression target, and the realized future footprint.

    This is a lambda-return ON FEATURES. THERE IS NO ADVANTAGE ANYWHERE IN IT: nothing is
    subtracted from anything, there is no baseline, and the result is a target for a
    vector regression rather than a weight on a gradient. That is what replacing a scalar
    critic with a vector one buys -- with V linear in psi there is no scalar residual left
    for an advantage to be.

    WHAT OPERATOR IS ACTUALLY BEING REGRESSED, STATED EXACTLY. At lam = 0 the target is the
    one-step occupancy flow constraint
        psi(s,a) = phi(s,a) + gamma * psi(s', a_bar(s')),
    i.e. "the footprint of this step plus the footprint of everything downstream". At
    lam > 0 the bootstrap is a lam-BLEND of that MEAN-action successor feature and the
    realized SAMPLED-action chain (`carry` carries phi(s_{t+k}, a_{t+k}) at the actions the
    policy actually took), so the fixed point is the blended operator's, and it coincides
    with the mean-action flow constraint only at lam = 0 or for a deterministic policy. The
    blend is deliberate -- the sampled chain is where the channel's action-attributable
    information comes from -- but it is not the textbook SF fixed point and should not be
    described as one.

    Either way it is a CONSERVATION LAW rather than a fitted optimum: contraction in
    gamma*lam-weighted expectation, unique fixed point at a fixed policy, and so NO target
    network is required. The usual reason for one is a max/argmax backup, and there is none
    here -- the bootstrap is the student's own mean action, never an optimized one.

    Boundary handling is the parent's, exactly:
      nonterm[t]     = (1 - terminations[t]) * valids[t]   -- may we bootstrap at all
      lam_nonterm[t] = 1 - boundaries[t]                   -- may the lambda chain cross t
    so a truncation bootstraps off the recorded final observation and a true termination
    contributes phi alone. NOTE the parent scales its carry by lam_nonterm ONLY while this
    multiplies the whole blended bootstrap by nonterm[t]; the two agree because the rollout
    guarantees nonterm[t] == 0 => lam_nonterm[t] == 0 (both a termination and a missing
    final observation imply boundaries[t] == 1). That invariant is what makes the shorter
    form here equivalent, so it is written down rather than assumed.
    """
    num_steps, num_envs = nonterm.shape
    phi_taken = _chunked(sf.phi, b_obs, b_act, chunk=chunk)
    a_bar_next = _chunked(agent.mean_action, b_next_obs, chunk=chunk)
    psi_next = _chunked(sf.psi, b_next_obs, a_bar_next, chunk=chunk)

    shape = (num_steps, num_envs, sf.phi_dim)
    phi_s = phi_taken.view(shape)
    psi_n = psi_next.view(shape)
    nt = nonterm.unsqueeze(-1)
    ln = lam_nonterm.unsqueeze(-1)

    tgt = torch.zeros_like(phi_s)
    future = torch.zeros_like(phi_s)
    carry = torch.zeros(num_envs, sf.phi_dim, device=phi_s.device)
    for t in reversed(range(num_steps)):
        # t == num_steps - 1 has no tgt[t+1] to mix with, so the chain weight is 0 there.
        lam_eff = 0.0 if t == num_steps - 1 else lam * ln[t]
        boot = (1.0 - lam_eff) * psi_n[t] + lam_eff * carry
        future[t] = gamma * nt[t] * boot
        tgt[t] = phi_s[t] + future[t]
        carry = tgt[t]
    return phi_taken, tgt.reshape(-1, sf.phi_dim), future.reshape(-1, sf.phi_dim)


# =====================================================================================
# BLOCK C -- THE PRIVILEGED CHANNEL: the occupancy surprise. The load-bearing part.
# =====================================================================================
@torch.no_grad()
def sf_channel(sf, agent, b_obs, future_realized, chunk, ema_beta, warmup):
    """c = (realized future footprint) - (predicted future footprint), per-dim standardized.

    THE CHANNEL IS NOT `tgt - psi(s,a)`. That leaks: tgt contains phi(s_t, a_t) EXPLICITLY,
    so the network could invert phi to recover a_taken and collapse clone_loss to the
    identity map (the parent's own warning, its lines ~327-333). Instead both terms are
    FUTURE-ONLY:

        future_realized[t] = gamma * nonterm[t] * boot[t]          == tgt[t] - phi(s_t,a_t)
        future_pred[t]     = psi(s_t, a_bar(s_t)) - phi(s_t, a_bar(s_t))
        c_raw[t]           = future_realized[t] - future_pred[t]

    phi(s_t, a_t) cancels ALGEBRAICALLY out of the first line, and the second is evaluated
    at the student's MEAN action, so a_t reaches c ONLY through the environment's response
    to it: s_{t+1}, and (on envs that terminate) the termination flag, which zeroes the
    whole future term. There is no closed-form inversion path from c back to a_t -- the
    network would have to invert the simulator. That is exactly the legitimate hindsight
    privilege §6.4 measured at R2(action | s_{t+1}) = +0.4661, and it is the difference
    between privileged information and leakage. Both are audited every iteration:
    sf/chan_action_r2 says the privilege is there, sf/cond_action_r2 says it is not a
    giveaway -- and both are LINEAR probes, so they bound recoverability from below while
    the trunk is nonlinear. A LINEAR gate is the right instrument anyway, because §6.4's
    0.0000 was measured the same way and the comparison is the point.

    Note `future_pred` subtracts phi at the SAME mean action psi was evaluated at, not at
    the taken action. Using the taken action there would reintroduce phi(s_t, a_t) with a
    sign, i.e. reintroduce the leak in the term meant to remove it.

    WHAT DOES NOT FULLY CANCEL, MEASURED. psi carries a large per-dim LEVEL (see
    sf/psi_bias_frac). A constant level L enters future_pred with weight 1 but enters
    future_realized only with weight gamma*(1-lam)/(1-gamma*lam) = 0.832 at gamma 0.99,
    lam 0.95, so 16.8% of L survives in c_raw as a CONSTANT OFFSET (1.0% at lam = 0). It is
    invisible to the R2 gates -- their intercept absorbs it -- but it shifts the channel's
    natural zero and inflates the RMS normalizer c_ms. So "the level cancels" is an 83%
    cancellation, not an identity, and the residual shrinks as psi's level converges.
    """
    a_bar = _chunked(agent.mean_action, b_obs, chunk=chunk)
    psi_bar = _chunked(sf.psi, b_obs, a_bar, chunk=chunk)
    phi_bar = _chunked(sf.phi, b_obs, a_bar, chunk=chunk)
    c_raw = future_realized - (psi_bar - phi_bar)
    # Updated from THIS batch before standardizing. Unlike the parent's held-out jac_r2,
    # this is a normalizer and not a diagnostic, so there is nothing to keep held out; and
    # iteration 1 must not divide by the init value of 1.0 when the true scale differs by
    # orders of magnitude. beta = 0 on the first iteration is the parent's own warm-start
    # convention (jac_model.update_stats(..., 0.0 if iteration == 1 else 0.99)).
    sf.update_chan_stats(c_raw, 0.0 if warmup else ema_beta)
    return c_raw, c_raw / sf.chan_std()


# =====================================================================================
# BLOCK D -- ALWAYS-ACTIVE SUCCESSOR-FEATURE ACTION TEACHER.
# =====================================================================================
def sf_action_teacher(sf, agent, obs, target_kl, chunk):
    """Build a frozen full-Beta teacher by locally maximizing w.psi at exact KL.

    The graph ends at detached snapshot alpha and beta leaves. The exact 2x2 Fisher block
    supplies the natural direction for each action dimension, normalized to unit Fisher
    norm per state. Both Beta parameters may move; alpha,beta >= 1 is the only projection.
    The KL bracket expands before bisection, so a finite search ceiling cannot weaken the
    teacher silently.
    """
    teacher_alphas, teacher_betas = [], []
    student_alphas, student_betas = [], []
    delivered_kls, predicted_gains = [], []
    mean_gaps, metric_grad_norms, step_scales, boundary_fracs = [], [], [], []
    kl_shortfall_fracs, nonpositive_gain_fracs, zero_grad_fracs = [], [], []
    kl_abs_errors, concentration_ratios, log_concentration_changes = [], [], []
    bracket_expansions = []

    for start in range(0, obs.shape[0], chunk):
        obs_mb = obs[start : start + chunk]
        with torch.no_grad():
            s_alpha, s_beta = agent.policy(obs_mb, agent._zero_cond(obs_mb))
            z0 = s_alpha / (s_alpha + s_beta)

        alpha = s_alpha.detach().requires_grad_(True)
        beta = s_beta.detach().requires_grad_(True)
        z = alpha / (alpha + beta)
        predicted_value = sf.value(obs_mb, agent.z_to_action(z))
        grad_alpha, grad_beta = torch.autograd.grad(
            predicted_value.sum(), (alpha, beta), create_graph=False
        )

        with torch.no_grad():
            gradient_ok = (
                torch.isfinite(predicted_value)
                & torch.isfinite(grad_alpha).all(-1)
                & torch.isfinite(grad_beta).all(-1)
            )
            if not bool(gradient_ok.all()):
                raise FloatingPointError("non-finite full-Beta teacher gradient")

            grad_scale = torch.maximum(
                grad_alpha.abs().amax(-1, keepdim=True),
                grad_beta.abs().amax(-1, keepdim=True),
            )
            zero_grad = grad_scale == 0.0
            safe_grad_scale = torch.where(
                zero_grad, torch.ones_like(grad_scale), grad_scale
            )
            scaled_grad_alpha = grad_alpha / safe_grad_scale
            scaled_grad_beta = grad_beta / safe_grad_scale

            concentration = s_alpha + s_beta
            tri_sum = torch.polygamma(1, concentration)
            fisher_aa = torch.polygamma(1, s_alpha) - tri_sum
            fisher_bb = torch.polygamma(1, s_beta) - tri_sum
            fisher_ab = -tri_sum
            fisher_det = fisher_aa * fisher_bb - fisher_ab.square()
            fisher_ok = (
                torch.isfinite(fisher_aa)
                & torch.isfinite(fisher_bb)
                & torch.isfinite(fisher_ab)
                & torch.isfinite(fisher_det)
                & (fisher_aa > 0.0)
                & (fisher_bb > 0.0)
                & (fisher_det > 0.0)
            )
            if not bool(fisher_ok.all()):
                raise FloatingPointError("non-positive or non-finite Beta Fisher block")

            natural_alpha = (
                fisher_bb * scaled_grad_alpha - fisher_ab * scaled_grad_beta
            ) / fisher_det
            natural_beta = (
                fisher_aa * scaled_grad_beta - fisher_ab * scaled_grad_alpha
            ) / fisher_det
            scaled_metric_grad_sq = (
                scaled_grad_alpha * natural_alpha + scaled_grad_beta * natural_beta
            ).sum(-1, keepdim=True)
            metric_ok = torch.isfinite(scaled_metric_grad_sq) & (
                zero_grad | (scaled_metric_grad_sq > 0.0)
            )
            if not bool(metric_ok.all()):
                raise FloatingPointError("invalid full-Beta natural-gradient norm")

            safe_norm = torch.where(
                zero_grad, torch.ones_like(scaled_metric_grad_sq), scaled_metric_grad_sq
            ).sqrt()
            direction_alpha = natural_alpha / safe_norm
            direction_beta = natural_beta / safe_norm
            metric_grad_norm = torch.where(
                zero_grad, torch.zeros_like(grad_scale), grad_scale * safe_norm
            )
            direction_ok = torch.isfinite(direction_alpha) & torch.isfinite(direction_beta)
            if not bool(direction_ok.all()):
                raise FloatingPointError("non-finite full-Beta natural direction")

            def at_scale(scale):
                t_alpha = (s_alpha + scale * direction_alpha).clamp_min(1.0)
                t_beta = (s_beta + scale * direction_beta).clamp_min(1.0)
                z_teacher = t_alpha / (t_alpha + t_beta)
                return z_teacher, t_alpha, t_beta

            def checked_total_kl(candidate_alpha, candidate_beta):
                kl_per_dim = beta_kl_per_dim(
                    candidate_alpha, candidate_beta, s_alpha, s_beta
                )
                if not bool(torch.isfinite(kl_per_dim).all()):
                    raise FloatingPointError("non-finite full-Beta projection KL")
                return kl_per_dim.sum(-1, keepdim=True)

            active = ~zero_grad
            lo = z0.new_zeros((z0.shape[0], 1))
            hi = z0.new_ones((z0.shape[0], 1))
            expansions = z0.new_zeros((z0.shape[0], 1))
            # A unit-Fisher ray reaches 0.10 KL near scale sqrt(0.2). Twenty-four
            # doublings are defensive only; exhaustion raises instead of weakening dose.
            for _ in range(24):
                _, hi_alpha, hi_beta = at_scale(hi)
                hi_kl = checked_total_kl(hi_alpha, hi_beta)
                needs_expansion = active & (hi_kl < target_kl)
                if not bool(needs_expansion.any()):
                    break
                expansions.add_(needs_expansion.to(expansions.dtype))
                hi = torch.where(needs_expansion, 2.0 * hi, hi)

            _, hi_alpha, hi_beta = at_scale(hi)
            hi_kl = checked_total_kl(hi_alpha, hi_beta)
            unbracketed = active & (hi_kl < target_kl)
            if bool(unbracketed.any()):
                raise RuntimeError("full-Beta teacher failed to bracket target KL")

            for _ in range(24):
                mid = 0.5 * (lo + hi)
                _, mid_alpha, mid_beta = at_scale(mid)
                mid_kl = checked_total_kl(mid_alpha, mid_beta)
                below = active & (mid_kl < target_kl)
                lo = torch.where(below, mid, lo)
                hi = torch.where(below, hi, mid)

            scale = 0.5 * (lo + hi)
            scale = torch.where(zero_grad, torch.zeros_like(scale), scale)
            z_teacher, t_alpha, t_beta = at_scale(scale)
            delivered_kl = checked_total_kl(t_alpha, t_beta).squeeze(-1)
            teacher_ok = (
                torch.isfinite(z_teacher).all(-1)
                & torch.isfinite(t_alpha).all(-1)
                & torch.isfinite(t_beta).all(-1)
                & torch.isfinite(delivered_kl)
            )
            if not bool(teacher_ok.all()):
                raise FloatingPointError("non-finite full-Beta teacher")
            dose_error = active.squeeze(-1) & (
                (delivered_kl - target_kl).abs() > 0.01 * target_kl
            )
            if bool(dose_error.any()):
                raise RuntimeError("full-Beta teacher missed target KL")
            teacher_value = sf.value(obs_mb, agent.z_to_action(z_teacher))
            action_gap = (
                agent.z_to_action(z_teacher) - agent.z_to_action(z0)
            ).abs().mean(-1)
            boundary = torch.cat(
                [t_alpha <= 1.0 + 1e-6, t_beta <= 1.0 + 1e-6], dim=-1
            ).to(z0.dtype).mean(-1)
            concentration_ratio = (t_alpha + t_beta) / concentration

            teacher_alphas.append(t_alpha)
            teacher_betas.append(t_beta)
            student_alphas.append(s_alpha)
            student_betas.append(s_beta)
            delivered_kls.append(delivered_kl)
            predicted_gain = teacher_value - predicted_value.detach()
            if not bool(torch.isfinite(predicted_gain).all()):
                raise FloatingPointError("non-finite full-Beta predicted gain")
            predicted_gains.append(predicted_gain)
            mean_gaps.append(action_gap)
            metric_grad_norms.append(metric_grad_norm.squeeze(-1))
            step_scales.append(scale.squeeze(-1))
            boundary_fracs.append(boundary)
            kl_shortfall_fracs.append(
                (active.squeeze(-1) & (delivered_kl < 0.99 * target_kl)).to(z0.dtype)
            )
            nonpositive_gain_fracs.append(
                (active.squeeze(-1) & (predicted_gain <= 0.0)).to(z0.dtype)
            )
            zero_grad_fracs.append(zero_grad.squeeze(-1).to(z0.dtype))
            kl_abs_errors.append(
                torch.where(
                    active.squeeze(-1),
                    (delivered_kl - target_kl).abs(),
                    delivered_kl.abs(),
                )
            )
            concentration_ratios.append(concentration_ratio.mean(-1))
            log_concentration_changes.append(concentration_ratio.log().abs().mean(-1))
            bracket_expansions.append(expansions.squeeze(-1))

    def cat_mean(parts):
        return float(torch.cat(parts).mean().item())

    def cat_max(parts):
        return float(torch.cat(parts).max().item())

    metrics = {
        "snapshot_kl": cat_mean(delivered_kls),
        "pred_gain": cat_mean(predicted_gains),
        "mean_gap": cat_mean(mean_gaps),
        "metric_grad_norm": cat_mean(metric_grad_norms),
        "step_scale": cat_mean(step_scales),
        "boundary_frac": cat_mean(boundary_fracs),
        "kl_shortfall_frac": cat_mean(kl_shortfall_fracs),
        "nonpositive_gain_frac": cat_mean(nonpositive_gain_fracs),
        "zero_grad_frac": cat_mean(zero_grad_fracs),
        "kl_abs_error": cat_mean(kl_abs_errors),
        "kl_abs_error_max": cat_max(kl_abs_errors),
        "concentration_ratio": cat_mean(concentration_ratios),
        "log_concentration_change": cat_mean(log_concentration_changes),
        "bracket_expansions": cat_mean(bracket_expansions),
        "bracket_expansions_max": cat_max(bracket_expansions),
    }
    return (
        torch.cat(teacher_alphas).detach(),
        torch.cat(teacher_betas).detach(),
        torch.cat(student_alphas).detach(),
        torch.cat(student_betas).detach(),
        metrics,
    )


@torch.no_grad()
def teacher_follow_metrics(agent, obs, t_alpha, t_beta, s_alpha_before, s_beta_before, chunk):
    """Measure whether one actor update moved the student toward the frozen teacher."""
    post_alphas, post_betas = [], []
    for start in range(0, obs.shape[0], chunk):
        obs_mb = obs[start : start + chunk]
        alpha, beta = agent.policy(obs_mb, agent._zero_cond(obs_mb))
        post_alphas.append(alpha)
        post_betas.append(beta)
    s_alpha_after = torch.cat(post_alphas)
    s_beta_after = torch.cat(post_betas)

    z_teacher = t_alpha / (t_alpha + t_beta)
    z_before = s_alpha_before / (s_alpha_before + s_beta_before)
    z_after = s_alpha_after / (s_alpha_after + s_beta_after)
    target_step = agent.z_to_action(z_teacher) - agent.z_to_action(z_before)
    actual_step = agent.z_to_action(z_after) - agent.z_to_action(z_before)
    target_sq = target_step.square().sum(-1)
    actual_sq = actual_step.square().sum(-1)
    dot = (actual_step * target_step).sum(-1)
    safe_target_sq = torch.where(target_sq == 0.0, torch.ones_like(target_sq), target_sq)
    progress = torch.where(target_sq == 0.0, torch.zeros_like(dot), dot / safe_target_sq)
    cosine_denom = (target_sq * actual_sq).sqrt()
    safe_cosine_denom = torch.where(
        cosine_denom == 0.0, torch.ones_like(cosine_denom), cosine_denom
    )
    alignment = torch.where(
        cosine_denom == 0.0, torch.zeros_like(dot), dot / safe_cosine_denom
    )

    pre_kl = beta_kl_per_dim(
        t_alpha, t_beta, s_alpha_before, s_beta_before
    ).sum(-1)
    post_kl = beta_kl_per_dim(
        t_alpha, t_beta, s_alpha_after, s_beta_after
    ).sum(-1)
    pre_gap = target_step.abs().mean(-1)
    post_gap = (
        agent.z_to_action(z_teacher) - agent.z_to_action(z_after)
    ).abs().mean(-1)
    reduction = (pre_kl - post_kl) / pre_kl.clamp_min(1e-12)
    conc_ratio = (t_alpha + t_beta).mean() / (s_alpha_after + s_beta_after).mean()
    return {
        "student_kl_pre": float(pre_kl.mean().item()),
        "student_kl_post": float(post_kl.mean().item()),
        "student_kl_reduction": float(reduction.mean().item()),
        "student_gap_pre": float(pre_gap.mean().item()),
        "student_gap_post": float(post_gap.mean().item()),
        "follow_progress": float(progress.mean().item()),
        "follow_alignment": float(alignment.mean().item()),
        "conc_ratio_post": float(conc_ratio.item()),
    }


# =====================================================================================
# BLOCK E -- SF LOSS ASSEMBLY. Its own optimizer, backward and clip; see SFCritic.
# =====================================================================================
def sf_losses(sf, obs, action, psi_target, reward, args):
    """psi TD regression + reward readout + phi decorrelation.

    `psi_target` is detached (built under no_grad in BLOCK B), so this is plain supervised
    regression -- the shape the charter's §3 requires. The reward term regresses w on the
    raw environment reward: with the file's defaults there is no entropy shaping and no
    reward normalization anywhere, so w means "the direction in feature space that the
    environment pays for" with no reinterpretation needed. Passing --normalize-reward or
    --clip-reward (both off by default, both inherited from the chassis) would silently
    redefine w to point at the WRAPPED reward, and the SF teacher action gradient,
    sf/w_r2, and explained_variance would all follow it.
    """
    # v2: both sides in STANDARDIZED target space. Equivalent to a per-dim reweighting of
    # v1's raw MSE by 1/var, which is what stops the loud dims from owning the gradient.
    psi_pred = sf.psi_std(obs, action)
    phi_pred = sf.phi(obs, action)
    psi_mse = (psi_pred - sf.standardize_psi_target(psi_target)).square().mean()
    rew_mse = (sf.w_head(phi_pred).squeeze(-1) - reward).square().mean()
    cov = offdiag_decorr(phi_pred)
    total = args.sf_coef * psi_mse + args.sf_rew_coef * rew_mse + args.sf_cov_coef * cov
    return total, psi_mse, rew_mse, cov


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    assert args.batch_size % args.num_minibatches == 0
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.actor_epochs > 0 and args.critic_epochs > 0
    assert args.sf_teacher_kl > 0.0, "teacher KL must be positive to supply optimization pressure"
    assert args.cond_mode in ("occ", "occ_w"), f"unknown cond_mode {args.cond_mode!r}"
    # offdiag_decorr divides by phi_dim * (phi_dim - 1), so phi_dim = 1 would silently make
    # the SF loss NaN. A one-dimensional occupancy measure is also exactly --cond-mode
    # occ_w's channel with none of its encoding, i.e. the thing this file exists to beat.
    assert args.phi_dim >= 2, "phi_dim must be >= 2; a 1-dim occupancy measure is a scalar"
    assert args.batch_size >= 2, "SF diagnostics require at least two rows"
    sf_lambda = args.sf_lambda if args.sf_lambda >= 0.0 else args.gae_lambda
    sf_hidden = args.sf_hidden if args.sf_hidden > 0 else args.hidden

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

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    assert obs_shape is not None and action_shape is not None
    act_dim = int(np.prod(action_shape))

    agent = Agent(envs, args).to(device)
    # Rollout forward only. The update stays eager: it is a small share of wall clock, and
    # graphing it would complicate the telemetry paths for little gain.
    act_fn = (
        torch.compile(agent.act, mode="reduce-overhead") if args.compile_act else agent.act
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # SFCritic's init draws happen inside fork_rng AND from an explicitly reseeded stream.
    # fork_rng alone stops the SF init from advancing the outer stream (so phi_dim cannot
    # silently shift every later minibatch permutation -- the parent's discipline for its
    # TransitionHead, and the bug class conseq_v1's probe had). But it does not fix the
    # other direction, and that direction matters for THIS file's primary claim: cond_mode
    # changes cond_dim, hence the trunk's input width, hence how many normal draws
    # orthogonal_ consumes building the agent -- so without the reseed the two arms would
    # start from DIFFERENT phi/psi/w as well as different trunks, and the iteration-1
    # channel comparison would not be a single-variable one. Reseeding pins phi, psi and w
    # identically across arms. The TRUNK still differs between them: its input width is the
    # ablation, so that difference is irreducible and is stated rather than papered over.
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(args.seed)
        sf = SFCritic(obs_shape[0], act_dim, args.phi_dim, sf_hidden).to(device)
    # NOT annealed, deliberately, and unlike `optimizer` below. The SF losses are stationary
    # supervised regressions onto detached targets, so there is no trust-region reason to
    # decay their step size; the parent likewise annealed only the actor optimizer and left
    # its jac_optimizer at a constant lr.
    sf_optimizer = optim.Adam(sf.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                action, z = act_fn(next_obs)
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

        b_obs = obs.reshape((-1,) + obs_shape)
        b_next_obs = next_obses.reshape((-1,) + obs_shape)
        b_z = latent_zs.reshape(-1, act_dim)
        b_rewards = rewards.reshape(-1)
        with torch.no_grad():
            b_act = agent.z_to_action(b_z)
        nonterm = (1.0 - transition_terminations) * transition_valids
        lam_nonterm = 1.0 - transition_boundaries

        # ===== BLOCK B: psi's target, a lambda-return on FEATURES =======================
        phi_taken, b_psi_target, b_future = sf_feature_targets(
            sf,
            agent,
            b_obs,
            b_next_obs,
            b_act,
            nonterm,
            lam_nonterm,
            sf_lambda,
            args.gamma,
            args.minibatch_size,
        )

        # ===== BLOCK C: the privileged channel ==========================================
        c_raw, b_c = sf_channel(
            sf,
            agent,
            b_obs,
            b_future,
            args.minibatch_size,
            args.cond_ema_beta,
            iteration == 1,
        )

        # v3: refresh psi's standardization HERE -- strictly AFTER the channel, never
        # between BLOCK B and BLOCK C. v2 updated it in between and measurably corrupted the
        # channel: c differences a realized footprint built in BLOCK B (old stats) against a
        # predicted footprint evaluated in BLOCK C (new stats), so the two footprints sat in
        # different psi parameterizations and their difference carried a spurious offset.
        # Measured cost of that ordering, v2 vs v1 at the identical 16-iteration config:
        # sf/delta 0.6726 -> 1.3970 (c inflated ~2x) and chan_action_r2 0.2402 -> 0.2260.
        # Placing the update after BLOCK C makes every psi consumer inside one iteration
        # share one parameterization, and the loss below picks up the refreshed stats.
        #
        # PURE BATCH STATISTICS, no EMA. v2 used the 0.99 channel horizon (~100 iterations)
        # and measured psi_bias_frac climbing straight back 0.0000 -> 0.4751: the target is
        # a BOOTSTRAP, so it drifts as psi learns, and any smoothing lag becomes exactly the
        # per-dim level error this standardization exists to remove. The target is rebuilt
        # from scratch every iteration over the full batch (4096 x 32 here), so its mean and
        # variance are a property of that batch with a standard error of std/64 -- there is
        # nothing to smooth and a lag is pure harm. This REMOVES a knob rather than adding
        # one; no beta is exposed.
        sf.update_psi_stats(b_psi_target, 0.0)

        # ===== BLOCK D INPUT: observed hindsight channel for rationalization =============
        # The teacher no longer queries this channel. It remains the privileged context for
        # clone_loss, trained only on realized future occupancy. In occ_w mode the scalar
        # projection still uses the pre-update reward direction, matching the critic that
        # built this rollout's channel.
        with torch.no_grad():
            w_full = sf.w_vec()
            w_unit = w_full / w_full.norm().clamp_min(1e-8)
            if args.cond_mode == "occ_w":
                b_chan = (b_c @ w_unit).unsqueeze(-1)
            else:
                b_chan = b_c
            cond_clipped = (b_chan.abs() >= args.cond_clip).float().mean().item()
            b_chan = b_chan.clamp(-args.cond_clip, args.cond_clip)
            chan_rms = b_chan.square().mean().sqrt().item()

        # ===== SF TELEMETRY, measured PRE-UPDATE so it describes the critic that produced
        # ===== this rollout's channel -- the same convention the parent used for EV. =====
        with torch.no_grad():
            psi_taken = _chunked(sf.psi, b_obs, b_act, chunk=args.minibatch_size)
            # Necessary-not-sufficient collapse detector: participation ratio of phi's
            # batch covariance spectrum. Its range is [1, phi_dim - 1], NOT [1, phi_dim]:
            # phi's non-affine LayerNorm forces every row to sum to zero, so the covariance
            # always has one exactly-zero eigenvalue and 31 of 32 is the ceiling. Cast to
            # double BEFORE the 32768-row accumulation, so that null direction is resolved
            # in float64 rather than being fp32 rounding noise inside the eigensolver.
            xc = (phi_taken - phi_taken.mean(0, keepdim=True)).double()
            eig = torch.linalg.eigvalsh((xc.T @ xc) / xc.shape[0]).clamp_min(0.0)
            phi_eff_rank = float(
                (eig.sum().square() / eig.square().sum().clamp_min(1e-30)).item()
            )
            corr = phi_corr(phi_taken)
            phi_offdiag = float(
                (corr - torch.diag_embed(torch.diagonal(corr))).abs().sum().item()
                / (args.phi_dim * (args.phi_dim - 1))
            )
            psi_r2 = _r2_per_dim(psi_taken, b_psi_target)
            # WHY psi_r2 CAN BE VERY NEGATIVE EARLY, MADE DECIDABLE. LayerNorm centers phi
            # ACROSS dims, not across samples, so each feature dim keeps a nonzero mean and
            # psi's lambda-return target is "a large per-dim LEVEL plus a small variation".
            # psi starts at ~0 (final layer std=0.01) and must travel to that level first,
            # which R2 charges brutally. This tag reports the share of psi's MSE that is
            # pure per-dim LEVEL error, so "not converged yet" is distinguishable from
            # "cannot fit" instead of being an argument. Near 1 = still climbing the level.
            # It also explains why the level does not poison the CHANNEL: c is a difference
            # of two future footprints, so 83% of the level cancels (see BLOCK C for the
            # exact residual, and why the remainder is an offset rather than a signal).
            psi_err = b_psi_target - psi_taken
            psi_bias_frac = float(
                (
                    psi_err.mean(0).square().mean() / psi_err.square().mean().clamp_min(1e-12)
                ).item()
            )
            # The MODEL's own reward fit, not the best affine rescaling of w . phi: no free
            # scale or intercept is granted, so this is the number the loss actually pays for.
            w_r2 = _r2_per_dim(sf.w_head(phi_taken), b_rewards.unsqueeze(-1))
            # THE LEAKAGE GATE. If a_taken is linearly recoverable from [c, s] the identity
            # collapse is live and clone_nll will run away to -inf. Threshold: <= 0.9.
            cond_action_r2 = _ls_r2(torch.cat([b_chan, b_obs], dim=-1), b_z)
            # THE INFORMATION GATE. §6.4 measured the scalar version of this at 0.0000.
            # Threshold for the hypothesis to be alive at all: >= 0.15.
            chan_action_r2 = _ls_r2(b_chan, b_z)
            # The same gate on the block the trunk ACTUALLY receives -- occ_w's best case,
            # since a linear probe on 16 Fourier features is strictly stronger than one on
            # the scalar they encode. Identical to chan_action_r2 in occ, where the encoding
            # is the identity, which is itself a useful consistency check on this pair.
            enc_action_r2 = _ls_r2(agent.cond_present(b_chan), b_z)
            # THE CONTROL THAT MAKES THE GATES ABOVE MEAN ANYTHING. As the policy sharpens,
            # a_taken becomes a deterministic function of s, so R2(action | ANY quantity
            # correlated with s) drifts up for a reason that has nothing to do with the
            # channel. §6.4's own method quotes the GAIN in R2 over state alone for exactly
            # this reason. Read every gate above against this one.
            state_action_r2 = _ls_r2(b_obs, b_z)
            # (Targets are the latent z rather than the environment action. R2 is invariant
            # to the fixed affine map between them, so this is R2 on the action.)
            # Value is a LINEAR readout of the vector critic, so EV is computed on the
            # scalar pair (V(s,a), w . tgt) and stays comparable to the parent's tag. The
            # IDENTICAL functional -- SFCritic.value's own w_head, bias included -- is
            # applied to both sides; the bias cancels in EV but sharing the call keeps the
            # tag definitionally "EV of V" rather than "EV of a related projection".
            v_pred = sf.w_head(psi_taken).squeeze(-1).cpu().numpy()
            v_true = sf.w_head(b_psi_target).squeeze(-1).cpu().numpy()
            variance = np.var(v_true)
            explained_variance = (
                np.nan if variance == 0 else 1.0 - np.var(v_true - v_pred) / variance
            )

        # ===== CRITIC FIRST ==============================================================
        # Fit the current rollout's fixed detached SF targets before asking the critic for
        # an action derivative. This is one algorithmic iteration, not a gate or warm-up:
        # every rollout gets the same critic budget followed by the same teacher dose.
        sf_totals, sf_psis, sf_rews, sf_covs = [], [], [], []
        for _ in range(args.critic_epochs):
            permutation = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = permutation[start : start + args.minibatch_size]
                sf_total, sf_psi, sf_rew, sf_cov = sf_losses(
                    sf, b_obs[mb], b_act[mb], b_psi_target[mb], b_rewards[mb], args
                )
                sf_optimizer.zero_grad(set_to_none=True)
                sf_total.backward()
                nn.utils.clip_grad_norm_(sf.parameters(), args.sf_grad_clip)
                sf_optimizer.step()
                with torch.no_grad():
                    sf_totals.append(sf_total.item())
                    sf_psis.append(sf_psi.item())
                    sf_rews.append(sf_rew.item())
                    sf_covs.append(sf_cov.item())
        # The teacher builder uses autograd.grad only with respect to detached snapshot
        # alpha and beta leaves. Clearing the last supervised gradients makes that isolation
        # inspectable and prevents stale `.grad` tensors from being mistaken for a path.
        sf_optimizer.zero_grad(set_to_none=True)
        # T3 describes the exact UPDATED critic snapshot used below, not the pre-update
        # critic that built the rollout channel.
        with torch.no_grad():
            teacher_psi = _chunked(sf.psi, b_obs, b_act, chunk=args.batch_size)
            teacher_phi = _chunked(sf.phi, b_obs, b_act, chunk=args.batch_size)
            teacher_psi_r2 = _r2_per_dim(teacher_psi, b_psi_target)
            teacher_w_r2 = _r2_per_dim(
                sf.w_head(teacher_phi), b_rewards.unsqueeze(-1)
            )
            teacher_w_norm = float(sf.w_vec().norm().item())

        # ===== ALWAYS-ACTIVE SF ACTION TEACHER ===========================================
        # Frozen once per iteration. Its full Beta parameters follow a local natural
        # improvement direction under the UPDATED successor critic, projected to exact KL.
        # The whole 32768-row batch fits comfortably because the differentiated path is only
        # psi's two 64-wide layers; one large launch avoids 32 tiny action-gradient launches.
        (
            b_t_alpha,
            b_t_beta,
            b_s_alpha_before,
            b_s_beta_before,
            teacher_metrics,
        ) = sf_action_teacher(sf, agent, b_obs, args.sf_teacher_kl, args.batch_size)

        # ===== ACTOR LAST ================================================================
        # Both targets are detached: taken actions for hindsight rationalization and the
        # fixed SF-optimized teacher for distillation. The actor loss contains no value,
        # advantage, importance ratio, or critic tensor.
        clone_losses, distill_kls, ents, tea_ents = [], [], [], []
        stu_nlls = []
        for _ in range(args.actor_epochs):
            permutation = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = permutation[start : start + args.minibatch_size]
                obs_mb, z_mb = b_obs[mb], b_z[mb]
                n = obs_mb.shape[0]
                a_tea, b_tea = b_t_alpha[mb], b_t_beta[mb]
                chan_mb = b_chan[mb]
                # One forward for both contexts:
                #   [privileged at observed hindsight c_t ; privileged absent student].
                cond_absent = chan_mb.new_zeros((n, agent.cond_dim))
                alpha, beta = agent.policy(
                    torch.cat([obs_mb, obs_mb], 0),
                    torch.cat([agent.cond_present(chan_mb), cond_absent], 0),
                )
                a_cl, b_cl = alpha[:n], beta[:n]
                a_stu, b_stu = alpha[n:], beta[n:]

                clone_loss = -(
                    Beta(a_cl, b_cl, validate_args=False).log_prob(z_mb).sum(-1).mean()
                )
                kl_dims = beta_kl_per_dim(a_tea, b_tea, a_stu, b_stu).clamp_min(0.0)
                distill_loss = kl_dims.sum(-1).mean()
                loss = args.clone_coef * clone_loss + args.distill_coef * distill_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clone_losses.append(clone_loss.item())
                    stu_nlls.append(
                        -(
                            Beta(a_stu, b_stu, validate_args=False)
                            .log_prob(z_mb)
                            .sum(-1)
                            .mean()
                            .item()
                        )
                    )
                    distill_kls.append(kl_dims.sum(-1).mean().item())
                    ents.append(
                        Beta(a_stu, b_stu, validate_args=False).entropy().sum(-1).mean().item()
                    )
                    tea_ents.append(
                        Beta(a_tea, b_tea, validate_args=False).entropy().sum(-1).mean().item()
                    )
        optimizer.zero_grad(set_to_none=True)
        follow_metrics = teacher_follow_metrics(
            agent,
            b_obs,
            b_t_alpha,
            b_t_beta,
            b_s_alpha_before,
            b_s_beta_before,
            args.batch_size,
        )

        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/clone_nll", float(np.mean(clone_losses)), global_step)
        writer.add_scalar("losses/student_nll", float(np.mean(stu_nlls)), global_step)
        # THE EXACT CHANNEL GAIN IN NATS. Two values of the SAME functional on the SAME
        # actions: how much log-likelihood the privileged block buys. ~0 is §6.4's "a
        # channel that predicts nothing about the action is a channel MLE drops for free",
        # seen from inside the loss instead of from a probe.
        writer.add_scalar(
            "sf/channel_nats",
            float(np.mean(stu_nlls)) - float(np.mean(clone_losses)),
            global_step,
        )
        writer.add_scalar("losses/distill_kl", float(np.mean(distill_kls)), global_step)
        writer.add_scalar("losses/entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("losses/sf_loss", float(np.mean(sf_totals)), global_step)
        writer.add_scalar("losses/sf_psi_mse", float(np.mean(sf_psis)), global_step)
        writer.add_scalar("losses/sf_rew_mse", float(np.mean(sf_rews)), global_step)
        writer.add_scalar("losses/sf_cov", float(np.mean(sf_covs)), global_step)
        writer.add_scalar("debug/teacher_entropy", float(np.mean(tea_ents)), global_step)
        writer.add_scalar("debug/student_entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("debug/cond_scale_rms", chan_rms, global_step)
        writer.add_scalar("debug/cond_clip_frac", cond_clipped, global_step)
        writer.add_scalar("debug/reward_mean", b_rewards.mean().item(), global_step)
        # --- the SF block's own instruments. This file's job is to be DECIDABLE. ---
        writer.add_scalar("sf/phi_eff_rank", phi_eff_rank, global_step)
        writer.add_scalar("sf/phi_offdiag_absmean", phi_offdiag, global_step)
        writer.add_scalar("sf/psi_r2", psi_r2, global_step)
        writer.add_scalar("sf/psi_bias_frac", psi_bias_frac, global_step)
        writer.add_scalar("sf/w_r2", w_r2, global_step)
        writer.add_scalar("sf/w_norm", float(w_full.norm().item()), global_step)
        writer.add_scalar("sf/teacher_psi_r2", teacher_psi_r2, global_step)
        writer.add_scalar("sf/teacher_w_r2", teacher_w_r2, global_step)
        writer.add_scalar("sf/teacher_w_norm", teacher_w_norm, global_step)
        # THE LEAKAGE GATE: near 1 means the identity collapse is live.
        writer.add_scalar("sf/cond_action_r2", cond_action_r2, global_step)
        # THE INFORMATION GATE: the vector analogue of §6.4's 0.0000 scalar measurement.
        writer.add_scalar("sf/chan_action_r2", chan_action_r2, global_step)
        # The same gate on the ENCODED block the trunk receives: occ_w's best case.
        writer.add_scalar("sf/enc_action_r2", enc_action_r2, global_step)
        # THE CONTROL for both R2 gates: how much of them is policy sharpening alone.
        writer.add_scalar("sf/state_action_r2", state_action_r2, global_step)
        writer.add_scalar("sf/teacher_snapshot_kl", teacher_metrics["snapshot_kl"], global_step)
        writer.add_scalar("sf/teacher_pred_gain", teacher_metrics["pred_gain"], global_step)
        writer.add_scalar("sf/teacher_mean_gap", teacher_metrics["mean_gap"], global_step)
        writer.add_scalar(
            "sf/teacher_metric_grad_norm", teacher_metrics["metric_grad_norm"], global_step
        )
        writer.add_scalar("sf/teacher_step_scale", teacher_metrics["step_scale"], global_step)
        writer.add_scalar(
            "sf/teacher_boundary_frac", teacher_metrics["boundary_frac"], global_step
        )
        writer.add_scalar(
            "sf/teacher_kl_shortfall_frac", teacher_metrics["kl_shortfall_frac"], global_step
        )
        writer.add_scalar(
            "sf/teacher_nonpositive_gain_frac",
            teacher_metrics["nonpositive_gain_frac"],
            global_step,
        )
        writer.add_scalar(
            "sf/teacher_zero_grad_frac", teacher_metrics["zero_grad_frac"], global_step
        )
        writer.add_scalar(
            "sf/teacher_kl_abs_error", teacher_metrics["kl_abs_error"], global_step
        )
        writer.add_scalar(
            "sf/teacher_kl_abs_error_max", teacher_metrics["kl_abs_error_max"], global_step
        )
        writer.add_scalar(
            "sf/teacher_concentration_ratio",
            teacher_metrics["concentration_ratio"],
            global_step,
        )
        writer.add_scalar(
            "sf/teacher_log_concentration_change",
            teacher_metrics["log_concentration_change"],
            global_step,
        )
        writer.add_scalar(
            "sf/teacher_bracket_expansions",
            teacher_metrics["bracket_expansions"],
            global_step,
        )
        writer.add_scalar(
            "sf/teacher_bracket_expansions_max",
            teacher_metrics["bracket_expansions_max"],
            global_step,
        )
        writer.add_scalar(
            "sf/teacher_student_kl_pre", follow_metrics["student_kl_pre"], global_step
        )
        writer.add_scalar(
            "sf/teacher_student_kl_post", follow_metrics["student_kl_post"], global_step
        )
        writer.add_scalar(
            "sf/teacher_student_kl_reduction",
            follow_metrics["student_kl_reduction"],
            global_step,
        )
        writer.add_scalar(
            "sf/teacher_student_gap_pre", follow_metrics["student_gap_pre"], global_step
        )
        writer.add_scalar(
            "sf/teacher_student_gap_post", follow_metrics["student_gap_post"], global_step
        )
        writer.add_scalar(
            "sf/teacher_follow_progress", follow_metrics["follow_progress"], global_step
        )
        writer.add_scalar(
            "sf/teacher_follow_alignment", follow_metrics["follow_alignment"], global_step
        )
        writer.add_scalar(
            "sf/teacher_conc_ratio_post", follow_metrics["conc_ratio_post"], global_step
        )
        print("SPS:", sps)

    envs.close()
    writer.close()

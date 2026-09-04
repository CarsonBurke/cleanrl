# WPPG / WPPG-I — Wasserstein Proximal Policy Gradient (arXiv 2603.02576v1),
# implemented on the iterthink_v24_beta_v162critic_mtp backbone.
#
# WHAT THE PAPER DOES (Zhu, Zhang, Gao, Li). Policy improvement is a proximal
# step in the 2-Wasserstein geometry of action distributions rather than a
# KL/ratio-clipped step in parameter space:
#
#   pi_{k+1}(.|s) = argmax_pi  <Q^{pi_k}_tau(s,.), pi(.|s)>
#                              - (1/2eta) W2^2(pi(.|s), pi_k(.|s)) - tau H_pi(s)
#
# A Lie-Trotter split handles the entropy term separately, and for an implicit
# policy pi = g(s,.)#nu the proximal step (Proposition 1) reduces to a
# DRIFT-MAP regression that needs no density and no score function:
#
#   g_{k+1/2} = argmax_g  Q(s, g(s,Z)) - (1/(2eta)) || g(s,Z) - g_k(s,Z) ||^2
#
# whose one-step solution is the transport a -> a + eta * grad_a Q(s,a). The
# entropy half-step convolves with N(0, 2*tau*eta I). Practically (Alg. 2/3)
# this is an OFF-POLICY actor-critic whose actor loss is DIRECTION MATCHING:
#
#   a0_{i,k} = g_theta(s_i, eps_{i,k})                    (K samples, shared eps)
#   G_{i,k}  = grad_a min(Q_w1, Q_w2)(s_i, a) |_{a=a0}    (stop-grad to critics)
#   Delta*   = eta * G + xi,   xi ~ N(0, 2*tau*eta I)
#   Delta    = g_theta(s_i, eps_{i,k}) - a0               (REFORWARD, same eps)
#   L_actor  = mean_{i,k} || Delta - Delta* ||^2
#
# Critics: twin Q with target nets, 1-step TD, multi-sample bootstrap
# (average over K target-actor actions of the per-sample min of the two
# target critics), Polyak sigma=0.005. Entropy is injected as a REWARD bonus
# r + tau*Hhat(s), where Hhat is the plug-in Gaussian-mixture entropy estimator
# of Alg. 1 (M mixture centers, L baseline actions, convolution std sigma_ent) —
# never an explicit log pi, so it works for implicit policies.
#
# NOVELTY / HYPOTHESIS OF THIS FILE. WPPG's actor only ever consumes
# grad_a Q. That makes the *smoothness and calibration of the critic's action
# field* the single most important object in the algorithm — far more than in
# PPO, where the critic only supplies a scalar baseline. So we replace the
# paper's plain scalar-MSE Q with the v162 critic geometry from the base file:
#   - HL-Gauss categorical head over a symlog support (+/-20k raw return),
#     decoded as symexp(E[bin]) -- NOT the base's E[symexp(bin)], see
#     `value_decode` in Args for why bootstrapping forces that change.
#     Cross-entropy to a projected Gaussian label
#     is a far better-conditioned regression than MSE on raw returns, and the
#     decoded scalar is a smooth function of the logits, so grad_a Q inherits
#     that conditioning instead of the raw-MSE curvature.
#   - bias-free zero-init head: symmetric symlog support => zero logits decode
#     to exactly 0, a neutral prior with no hidden offset.
#   - critic MTP: head h additionally regresses the (h+1)-step TD target
#     (h = 0..H-1). Head 0 IS the Q used for bootstrapping and for grad_a Q;
#     heads 1..H-1 are auxiliary multi-horizon predictions that shape the
#     shared [s,a] representation. In the on-policy base MTP predicted
#     returns[t+h]; the off-policy analogue is the n-step TD target, which is
#     why the replay buffer here is TIME-CONTIGUOUS rather than a flat set of
#     transitions.
#   - ThinkTrunk (dense + soft-MoE residual blocks) as the shared body of both
#     the actor and each critic.
# Hypothesis: a distributional, multi-horizon critic yields a smoother and
# better-scaled action-gradient field, which is exactly the quantity WPPG
# transports along, so it should improve on the paper's scalar critic.
# This is TESTABLE, not assumed: --critic-head scalar restores the paper's
# plain-MSE scalar Q on the identical trunk, buffer, actor and targets, so the
# A/B isolates the critic head and nothing else.
#
# WHY THE CRITIC IS THE RIGHT PLACE TO INTERVENE. Writing out the actor step:
# Delta is identically zero at the evaluation point, so the actor gradient is
# -2*(da/dtheta)^T*(eta*G + xi) whereas a deterministic policy gradient is
# -(da/dtheta)^T*G. WPPG's actor step IS the DPG step, scaled by 2*eta plus a
# xi term of std sqrt(2*tau*eta)=4.5e-3 -- and clip_grad_norm_ and Adam are
# both invariant to a uniform rescale, so even eta largely cancels. What is
# genuinely new is (a) the policy is stochastic and K samples transport the
# whole pushforward rather than one deterministic point, (b) the density-free
# entropy half-step, (c) the W2 derivation and its convergence theory. The
# per-step optimizer, though, is DPG -- so the critic's action field, not the
# actor rule, is where an implementation can actually move the needle.
#
# ACTOR VARIANTS (--actor-dist)
#   "implicit" (DEFAULT, = WPPG-I, the paper's stronger method):
#       a = Affine(tanh(f_theta([s, z]))), z ~ N(0, I_M), M = |S| / latent_divisor.
#       Table 3 reads "(1/3) x State Dimension" (the fraction survives pdftotext
#       as a stray "1" above the row, which is easy to misread as "3 x"); the
#       appendix prose settles it -- "setting the latent dimension about
#       one-third of the state dimension provides a good balance between
#       exploration and stability" -- as does their latent-dim ablation, which
#       sweeps {1,10,50,100,150} on Humanoid (|S| ~ 348) and finds 150 already
#       degrading. HalfCheetah |S|=17 => M=6. No closed-form density;
#       stochasticity is state-conditional through the latent.
#   "gaussian" (= WPPG): a = Affine(tanh(mu(s) + sigma(s) * eps)). We keep the
#       BASE's dreamer4 log-VARIANCE head with the soft tanh-rescale bound
#       rather than SAC's clamped log-std; same family, no dead gradient at the
#       bound. sigma(s) is trained purely pathwise by the direction-matching
#       loss (Delta depends on sigma through eps) — no log-prob anywhere.
#   The base's Beta actor is deliberately NOT ported: direction matching needs
#   an explicit noise-conditioned map g(s, eps) that can be re-forwarded at the
#   SAME eps, and torch's Beta only offers implicit reparameterization.
#
# DELIBERATE DEVIATIONS (all flagged, all revertible)
#   - Observation normalization moved OUT of the env wrappers into a shared
#     running normalizer applied at SAMPLE time. Wrapper-side normalization is
#     unsound with a replay buffer: stored observations would be frozen under
#     stale statistics. Buffer stores RAW obs.
#   - num_envs=16 and 8M env steps (repo convention) instead of the paper's 1
#     env / 1M steps. Batch stays at the paper's 256, but we take 0.25 gradient
#     steps per collected transition instead of 1 (64 replayed samples per
#     transition vs the paper's 256). That is still ~2M gradient steps, i.e.
#     ~2x the paper's total optimization work spread over 8x the data; running
#     a true UTD=1 over 8M steps would be ~8M gradient steps of a 32-sample
#     twin-critic update, which does not fit a sane wall clock.
#   - Auxiliary MTP horizons h>=1 bootstrap with mtp_bootstrap_samples target
#     actions instead of the full K (head 0 always uses K, per the paper).
#   - Soft-MoE experts are evaluated as one batched einsum instead of a Python
#     loop over nn.Sequential. Mathematically identical, ~an order of magnitude
#     fewer kernel launches, which this workload is bound by.
#
# FURTHER DEVIATIONS the paper does not specify (kept, but named here so they
# are never mistaken for the paper's own choices)
#   - Gradient clipping to max_grad_norm=0.5, applied to the actor and to each
#     critic separately (the paper trains the twin critics as two independent
#     regressions and specifies no clipping at all). This is in tension with
#     the method: WPPG's actor is supposed to move PROPORTIONALLY to eta*grad_a Q,
#     and a hard norm cap silently converts that into a normalized step whenever
#     the action-gradient field is large. debug/actor_clip_frac logs how often
#     the cap binds; if it sits near 1.0 the proximal step is gone and the knob
#     to reach for is --normalize-action-grad (explicit) rather than the clip.
#   - 10k-step uniform-random action warmup before the policy acts. Alg. 2/3
#     sample from the policy from t=0; Table 1's "learning starts 10000" only
#     gates gradient updates. The entropy bonus IS applied during warmup, so
#     every stored transition carries the same reward definition.
#   - Architecture: the paper prescribes a (256,256) ReLU MLP for HalfCheetah,
#     shared shape between actor and critic. We use ThinkTrunk (hidden 64,
#     3 blocks, 16 soft-MoE experts) instead -- a capacity INCREASE, not a
#     reduction, despite the smaller hidden width.
#   - Adam eps: paper says only "Adam", i.e. the 1e-8 default. This file uses
#     1e-8 (--adam-eps) rather than the CleanRL-PPO convention of 1e-5.
#   - The executed action is clamped to the action box after Gaussian
#     convolution, so the critic learns Q of the action actually executed. The
#     entropy estimator does not clamp, so pi_hat models the unclamped map.
#   - MTP heads are equally weighted in the critic loss (--mtp-aux-coef 1.0,
#     matching the base file), so head 0 -- the ONLY head used for bootstrapping
#     and for grad_a Q -- carries 1/H of the critic gradient. losses/critic_ce_h0
#     is logged separately to detect head-0 underfitting.
#   - Reported returns are TRAINING returns including the sigma_ent exploration
#     convolution, on -v4 envs; the paper evaluates separately on -v5. Numbers
#     are therefore not directly comparable to its Figure 1.
#
# NOTE ON --compile: this file does NOT accept --compile / --compile-mode (the
# repo's stock submit template passes them). tyro would reject them and the job
# would die before writing any tensorboard output. Submit without those flags.
#
# NOTE ON eta: clip_grad_norm_ and Adam are both invariant to a UNIFORM rescale
# of the gradient, and eta multiplies Delta* uniformly. So whenever the actor
# clip binds (see debug/actor_clip_frac), eta largely cancels out of the update
# and is not the exploration/step-size knob it looks like.
#
# Paper hyperparameters kept verbatim: K=32 action samples, eta=0.1,
# tau=1e-4, latent = |S|/3, gamma=0.99, polyak=0.005, lr=3e-4, batch=256,
# buffer=1e6, learning_starts=1e4. sigma_ent / M / L are NOT specified in the
# paper; see the Args defaults for the values chosen here. Their tau ablation
# finds tau in [0, 0.01] accelerates convergence and 0.1 slows it, so tau=1e-4
# is the bottom of a usable range rather than a fixed constant.
import os
import random
import time
import copy
import math
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport, symlog, symexp


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

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    num_envs: int = 16

    # --- WPPG core (paper Tables 2/3/4) ---
    gamma: float = 0.99
    learning_rate: float = 3e-4      # actor and critic alike
    polyak: float = 0.005            # target update coefficient sigma
    batch_size: int = 256
    buffer_size: int = 1000000
    learning_starts: int = 10000     # transitions collected before any update
    random_action_steps: Optional[int] = None  # uniform-action warmup; None => learning_starts
    action_samples: int = 32         # K: samples per state for actor step AND head-0 bootstrap
    eta: float = 0.1                 # Wasserstein proximal step size
    tau: float = 1e-4                # entropy scale (reward bonus AND xi noise std^2 = 2*tau*eta)
    utd_ratio: float = 0.25          # gradient steps per collected environment transition
    train_frequency: int = 1         # vector-env steps between update blocks

    # Plug-in mixture entropy estimator (paper Alg. 1). The paper specifies the
    # estimator but not sigma_ent / M / L; these are our choices.
    entropy_reg: bool = True
    sigma_ent: float = 0.05          # Gaussian convolution std (also the acting noise)
    entropy_mixture: int = 16        # M mixture centers
    entropy_baselines: int = 16      # L baseline actions

    # --- actor ---
    actor_dist: str = "implicit"     # "implicit" (WPPG-I) | "gaussian" (WPPG)
    latent_divisor: int = 3          # implicit: latent dim = obs_dim / latent_divisor (paper: 1/3)
    logvar_min: float = -8.0         # gaussian: dreamer4 soft log-var bound (std in [e^-4, e^4])
    logvar_max: float = 8.0
    normalize_action_grad: bool = False  # off = paper-faithful; on = unit-norm grad_a Q per sample

    # --- critic ---
    # "hlgauss" = the v162 categorical geometry (our addition).
    # "scalar"  = the PAPER's critic: one real number per head, plain MSE on the
    #             raw TD target. This is the honest control for the whole
    #             distributional-critic hypothesis, and it sidesteps the symlog
    #             decode question entirely (no decode, so no Jensen gain and no
    #             contraction concern).
    critic_head: str = "hlgauss"
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 2.0
    # How a categorical head over a SYMLOG support is decoded to a scalar.
    #   "symexp_mean" (DEFAULT) = symexp(E[Z]).  Self-consistent: projecting a
    #       scalar y to a label and decoding it returns y exactly, so the
    #       Bellman backup stays a contraction.
    #   "mean_symexp" = E[symexp(Z)], the v162 base file's decode. symexp is
    #       convex, so by Jensen this OVERSHOOTS by g ~ exp(s^2/2) for a critic
    #       with predictive spread s. Harmless in the base (it regresses MC
    #       returns, so the gain never re-enters its own target) but NOT here:
    #       under bootstrapping the gain compounds. With gamma=0.99 and this
    #       support, a perfectly-fit critic (spread = the 2-bin label width)
    #       still has g=1.00301 -> gamma*g=0.99298 -> a 1.42x inflated fixed
    #       point, and the operator turns EXPANSIVE at only 3.66 bins of
    #       spread. Kept solely as an ablation.
    value_decode: str = "symexp_mean"
    critic_mtp_horizon: int = 6      # 1 disables MTP (plain 1-step TD critic)
    mtp_bootstrap_samples: int = 4   # target actions for auxiliary horizons h>=1
    mtp_aux_coef: float = 1.0        # weight on horizons h>=1 (1.0 = base file's equal weighting)

    # --- trunk ---
    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    max_grad_norm: float = 0.5       # applied separately to the actor and to EACH critic
    adam_eps: float = 1e-8           # paper implies the Adam default, not CleanRL-PPO's 1e-5
    obs_norm: bool = True
    obs_clip: float = 10.0

    log_frequency: int = 1000        # vector-env steps between scalar dumps


def make_env(env_id, idx, capture_video, run_name):
    # NOTE: no NormalizeObservation here — observation normalization is applied
    # at sample time from a shared running estimator (see ObsNormalizer), which
    # is the only sound choice when observations sit in a replay buffer.
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
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


class BatchedExperts(nn.Module):
    """E parallel copies of `_branch_body` evaluated as two einsums.

    Mathematically identical to `nn.ModuleList([_branch_body(H) for _ in range(E)])`
    followed by `torch.stack(..., dim=1)`; the batched form replaces 4*E kernel
    launches per block with 2, which dominates runtime at this hidden size.
    """

    def __init__(self, H, n_experts):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_experts, H, H))
        self.b1 = nn.Parameter(torch.zeros(n_experts, H))
        self.w2 = nn.Parameter(torch.empty(n_experts, H, H))
        self.b2 = nn.Parameter(torch.zeros(n_experts, H))
        for e in range(n_experts):
            # nn.Linear stores (out, in) and computes x @ W^T; we store (in, out).
            for w in (self.w1, self.w2):
                buf = torch.empty(H, H)
                torch.nn.init.orthogonal_(buf, np.sqrt(2))
                with torch.no_grad():
                    w[e].copy_(buf.t())

    def forward(self, x):
        # x: (B, H) -> (B, E, H)
        h = torch.einsum("bh,ehk->bek", x, self.w1) + self.b1
        h = torch.relu(h).pow(2)
        return torch.einsum("bek,ekj->bej", h, self.w2) + self.b2


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
        self.experts = BatchedExperts(H, n_experts)

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in))                   # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = self.experts(m_in)                                  # (B, E, H)
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


class Actor(nn.Module):
    """Noise-conditioned deterministic map g_theta(s, noise) -> action.

    Both variants are exactly that map, which is all WPPG needs: the actor is
    never asked for a density, only to be re-forwardable at a fixed noise draw.
    """

    def __init__(self, obs_dim, act_dim, action_low, action_high, args):
        super().__init__()
        self.actor_dist = args.actor_dist
        self.act_dim = act_dim
        H = args.hidden
        if self.actor_dist == "implicit":
            self.noise_dim = max(1, round(obs_dim / args.latent_divisor))
            self.trunk = ThinkTrunk(obs_dim + self.noise_dim, H, args.k_blocks, args.n_experts)
            self.mean_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        elif self.actor_dist == "gaussian":
            self.noise_dim = act_dim
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.mean_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        else:
            raise ValueError(f"unknown actor_dist {args.actor_dist}")
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def sample_noise(self, n, device, generator=None):
        return torch.randn(n, self.noise_dim, device=device, generator=generator)

    def forward(self, obs, noise):
        # obs: (N, obs_dim); noise: (N, noise_dim) -> action (N, act_dim)
        if self.actor_dist == "implicit":
            feat = self.trunk(torch.cat([obs, noise], dim=-1))
            pre = self.mean_head(feat)
        else:
            feat = self.trunk(obs)
            mean = self.mean_head(feat)
            raw_lv = self.logvar_head(feat)
            # dreamer4 soft tanh-rescale bound on log-variance (smooth, no dead grad).
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            pre = mean + (0.5 * lv).exp() * noise
        return self.action_scale * torch.tanh(pre) + self.action_bias

    def forward_expanded(self, obs, noise):
        # obs: (B, obs_dim); noise: (B, K, noise_dim) -> (B, K, act_dim)
        b, k, _ = noise.shape
        obs_rep = obs.unsqueeze(1).expand(b, k, obs.shape[-1]).reshape(b * k, -1)
        return self(obs_rep, noise.reshape(b * k, -1)).view(b, k, self.act_dim)


class QCritic(nn.Module):
    """Q(s,a) as a v162-style HL-Gauss categorical head with MTP horizons.

    Returns logits of shape (N, mtp, num_bins). Horizon 0 is Q(s,a) proper;
    horizon h regresses the (h+1)-step TD target as an auxiliary task.
    """

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        H = args.hidden
        self.critic_head = args.critic_head
        # Categorical head emits num_bins logits per horizon; scalar head emits 1.
        self.out_dim = args.num_bins if args.critic_head == "hlgauss" else 1
        self.mtp = args.critic_mtp_horizon
        self.trunk = ThinkTrunk(obs_dim + act_dim, H, args.k_blocks, args.n_experts)
        # Bias-free zero head so BOTH variants start at Q == 0 exactly: for the
        # categorical head, zero logits are uniform over a symmetric symlog
        # support and symexp is odd, so the decode is 0 either way.
        self.head = layer_init(nn.Linear(H, self.mtp * self.out_dim, bias=False), std=0.1)
        with torch.no_grad():
            self.head.weight.zero_()

    def forward(self, obs, action):
        feat = self.trunk(torch.cat([obs, action], dim=-1))
        return self.head(feat).view(-1, self.mtp, self.out_dim)


class ObsNormalizer:
    """Running mean/var over raw observations, applied at act AND sample time.

    Equivalent in spirit to gym's NormalizeObservation, but kept outside the
    env so that replayed observations are always normalized with the CURRENT
    statistics rather than whatever was current when they were stored.
    """

    def __init__(self, shape, device, clip):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones(shape, device=device, dtype=torch.float64)
        self.count = 1e-4
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        x = x.to(torch.float64)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def __call__(self, x):
        mean = self.mean.to(x.dtype)
        std = self.var.to(x.dtype).clamp_min(1e-8).sqrt()
        return ((x - mean) / std).clamp(-self.clip, self.clip)


class SequentialReplayBuffer:
    """Time-contiguous circular replay over a vector env.

    Layout is (capacity_per_env, num_envs, ...) so that a sampled index t can be
    walked forward to t+1, ..., t+H-1 within the same env stream — which the
    n-step MTP targets require. Sampling only draws start offsets whose whole
    H-step window lies inside the currently-valid contiguous region, so no
    window ever straddles the write head.
    """

    def __init__(self, capacity_per_env, num_envs, obs_dim, act_dim, device):
        self.capacity = capacity_per_env
        self.num_envs = num_envs
        self.device = device
        self.pos = 0
        self.full = False
        z = lambda *shape: torch.zeros(*shape, device=device, dtype=torch.float32)
        self.obs = z(capacity_per_env, num_envs, obs_dim)
        self.next_obs = z(capacity_per_env, num_envs, obs_dim)
        self.actions = z(capacity_per_env, num_envs, act_dim)
        self.rewards = z(capacity_per_env, num_envs)
        self.terminations = z(capacity_per_env, num_envs)   # true MDP terminal
        self.boundaries = z(capacity_per_env, num_envs)     # terminal OR truncation
        self.boot_valid = z(capacity_per_env, num_envs)     # next_obs usable for bootstrap

    def add(self, obs, action, reward, next_obs, termination, boundary, boot_valid):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminations[i] = termination
        self.boundaries[i] = boundary
        self.boot_valid[i] = boot_valid
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def size(self):
        return self.capacity * self.num_envs if self.full else self.pos * self.num_envs

    def sample_starts(self, batch_size, horizon):
        """Return (time_index, env_index) for `batch_size` valid H-step windows."""
        n = self.capacity if self.full else self.pos
        avail = n - (horizon - 1)
        if avail <= 0:
            return None
        base = self.pos if self.full else 0
        off = torch.randint(0, avail, (batch_size,), device=self.device)
        t = (base + off) % self.capacity
        e = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        return t, e

    def gather(self, t, e, h):
        idx = (t + h) % self.capacity
        return (
            self.obs[idx, e],
            self.actions[idx, e],
            self.rewards[idx, e],
            self.next_obs[idx, e],
            self.terminations[idx, e],
            self.boundaries[idx, e],
            self.boot_valid[idx, e],
        )


@torch.no_grad()
def plugin_mixture_entropy(actor, obs, sigma_ent, n_mixture, n_baseline):
    """Paper Algorithm 1: plug-in Gaussian-mixture differential entropy.

    Centers mu_j = g(s, z_j) for M latents; baselines a_l = g(s, z~_l) + sigma*xi
    for L latents; Hhat = -(1/L) sum_l log[(1/M) sum_j phi_sigma(a_l - mu_j)].
    Needs only samples from the generator — no density, no score.
    """
    b, act_dim = obs.shape[0], actor.act_dim
    device = obs.device
    z_mix = torch.randn(b, n_mixture, actor.noise_dim, device=device)
    mu = actor.forward_expanded(obs, z_mix)                              # (B, M, A)
    z_base = torch.randn(b, n_baseline, actor.noise_dim, device=device)
    a = actor.forward_expanded(obs, z_base)
    a = a + sigma_ent * torch.randn_like(a)                              # (B, L, A)
    d2 = (a.unsqueeze(2) - mu.unsqueeze(1)).pow(2).sum(-1)               # (B, L, M)
    log_phi = -0.5 * d2 / (sigma_ent ** 2) - 0.5 * act_dim * math.log(2.0 * math.pi * sigma_ent ** 2)
    log_pi = torch.logsumexp(log_phi, dim=2) - math.log(n_mixture)       # (B, L)
    return -log_pi.mean(dim=1)                                           # (B,)


def polyak_update(source, target, coef):
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.mul_(1.0 - coef).add_(p, alpha=coef)
        for b, tb in zip(source.buffers(), target.buffers()):
            tb.copy_(b)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.actor_dist in ("implicit", "gaussian")
    assert args.critic_mtp_horizon >= 1
    if args.random_action_steps is None:
        args.random_action_steps = args.learning_starts
    grad_steps = max(1, int(round(args.utd_ratio * args.num_envs * args.train_frequency)))

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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))
    act_low = envs.single_action_space.low
    act_high = envs.single_action_space.high

    actor = Actor(obs_dim, act_dim, act_low, act_high, args).to(device)
    target_actor = copy.deepcopy(actor).to(device).requires_grad_(False)
    qf1 = QCritic(obs_dim, act_dim, args).to(device)
    qf2 = QCritic(obs_dim, act_dim, args).to(device)
    qf1_target = copy.deepcopy(qf1).to(device).requires_grad_(False)
    qf2_target = copy.deepcopy(qf2).to(device).requires_grad_(False)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    critic_params = list(qf1.parameters()) + list(qf2.parameters())
    critic_optimizer = optim.Adam(critic_params, lr=args.learning_rate, eps=args.adam_eps)

    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,
        device,
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )

    assert args.value_decode in ("symexp_mean", "mean_symexp")
    assert args.critic_head in ("hlgauss", "scalar")
    is_hlgauss = args.critic_head == "hlgauss"
    _decode = hl_support.to_scalar if args.value_decode == "symexp_mean" else hl_support.to_expected_scalar

    def q_scalar(pred):
        # Horizon-0 head -> scalar Q, (N,). The SAME reduction is used for the
        # bootstrap target and for grad_a Q, so the actor ascends exactly the
        # quantity the critic is regressed to.
        return _decode(pred[:, 0]) if is_hlgauss else pred[:, 0, 0]

    def critic_head_loss(pred, tgt_probs, tgt_scalars):
        """Per-horizon critic loss -> (B, H). CE for the categorical head, MSE
        for the paper's scalar head."""
        if is_hlgauss:
            return -(tgt_probs * torch.log_softmax(pred, dim=-1)).sum(-1)
        return (pred[..., 0] - tgt_scalars).pow(2)

    obs_norm = ObsNormalizer(obs_dim, device, args.obs_clip)
    normalize = (lambda x: obs_norm(x)) if args.obs_norm else (lambda x: x)

    buf = SequentialReplayBuffer(
        max(args.buffer_size // args.num_envs, args.critic_mtp_horizon + 1),
        args.num_envs,
        obs_dim,
        act_dim,
        device,
    )

    action_scale = torch.as_tensor((act_high - act_low) / 2.0, dtype=torch.float32, device=device)
    action_bias = torch.as_tensor((act_high + act_low) / 2.0, dtype=torch.float32, device=device)
    act_low_t = torch.as_tensor(act_low, dtype=torch.float32, device=device)
    act_high_t = torch.as_tensor(act_high, dtype=torch.float32, device=device)
    xi_std = math.sqrt(2.0 * args.tau * args.eta)
    gamma_pows = torch.tensor(
        [args.gamma ** i for i in range(args.critic_mtp_horizon + 1)], device=device
    )

    global_step = 0
    start_time = time.time()
    next_obs_raw, _ = envs.reset(seed=args.seed)
    next_obs_raw = torch.as_tensor(next_obs_raw, dtype=torch.float32, device=device)
    num_iterations = args.total_timesteps // args.num_envs
    last_log = 0
    ent_mean = 0.0
    has_trained = False
    clip_hits = clip_total = 0

    for iteration in range(num_iterations):
        global_step += args.num_envs

        # ---------------- act ----------------
        if args.obs_norm:
            obs_norm.update(next_obs_raw)
        obs_t = normalize(next_obs_raw)

        with torch.no_grad():
            if global_step <= args.random_action_steps:
                action = (
                    torch.rand(args.num_envs, act_dim, device=device) * (act_high_t - act_low_t)
                    + act_low_t
                )
            else:
                noise = actor.sample_noise(args.num_envs, device)
                action = actor(obs_t, noise)
                # Paper Alg. 2/3: Gaussian-convolve the executed action.
                action = action + args.sigma_ent * torch.randn_like(action)
            action = action.clamp(act_low_t, act_high_t)

            # Estimated for EVERY stored transition, including the random-action
            # warmup: Hhat is a property of the actor at s_t, not of the action
            # taken, so gating it on the warmup would give the oldest 1% of the
            # buffer a different reward definition than the rest.
            if args.entropy_reg:
                ent = plugin_mixture_entropy(
                    actor, obs_t, args.sigma_ent, args.entropy_mixture, args.entropy_baselines
                )
            else:
                ent = torch.zeros(args.num_envs, device=device)
        ent_mean = ent.mean().item()

        step_obs_raw = next_obs_raw
        next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

        transition_boundary = np.logical_or(terminations, truncations)
        transition_next_obs = np.array(next_obs_np, copy=True)
        boot_valid = np.ones(args.num_envs, dtype=np.float32)
        final_obs = infos.get("final_observation")
        final_obs_mask = infos.get("_final_observation")
        if final_obs is not None:
            if final_obs_mask is None:
                final_obs_mask = [fo is not None for fo in final_obs]
            for env_idx, has_final in enumerate(final_obs_mask):
                if has_final and final_obs[env_idx] is not None:
                    # `next_obs_np` already holds the RESET observation; the true
                    # successor state of this transition is the final observation.
                    transition_next_obs[env_idx] = final_obs[env_idx]
                elif transition_boundary[env_idx]:
                    boot_valid[env_idx] = 0.0
        else:
            boot_valid[transition_boundary] = 0.0

        # Terminal states never appear in `next_obs_raw` (that is the post-reset
        # observation), so without this they would be normalized by statistics
        # that never saw them -- and terminal states are exactly the
        # out-of-distribution ones that then clip at +/-obs_clip.
        if args.obs_norm and transition_boundary.any():
            obs_norm.update(
                torch.as_tensor(
                    transition_next_obs[transition_boundary], dtype=torch.float32, device=device
                )
            )

        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
        reward_ent = reward_t + args.tau * ent if args.entropy_reg else reward_t
        buf.add(
            step_obs_raw,
            action,
            reward_ent,
            torch.as_tensor(transition_next_obs, dtype=torch.float32, device=device),
            torch.as_tensor(terminations, dtype=torch.float32, device=device),
            torch.as_tensor(transition_boundary, dtype=torch.float32, device=device),
            torch.as_tensor(boot_valid, dtype=torch.float32, device=device),
        )
        next_obs_raw = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ---------------- learn ----------------
        do_update = global_step >= args.learning_starts and iteration % args.train_frequency == 0
        for _ in range(grad_steps if do_update else 0):
            sampled = buf.sample_starts(args.batch_size, args.critic_mtp_horizon)
            if sampled is None:
                break
            has_trained = True
            t_idx, e_idx = sampled
            B, K, Hh = args.batch_size, args.action_samples, args.critic_mtp_horizon

            # ---- multi-horizon TD targets ----
            # Along the window t..t+H-1 we need, at every offset i, the bootstrap
            # value of next_obs_i. Horizon h then reuses those: it accumulates
            # discounted rewards until the first episode boundary (or i == h) and
            # bootstraps there. So the H distinct bootstrap states are computed
            # ONCE and shared across all H targets.
            with torch.no_grad():
                obs0 = act0 = None
                rewards_h, cont_h, nonterm_h, boundary_h, qnext_h = [], [], [], [], []
                for i in range(Hh):
                    o_i, a_i, r_i, no_i, term_i, bnd_i, bv_i = buf.gather(t_idx, e_idx, i)
                    if i == 0:
                        obs0, act0 = normalize(o_i), a_i
                    no_i = normalize(no_i)
                    n_boot = K if i == 0 else args.mtp_bootstrap_samples
                    z = torch.randn(B, n_boot, target_actor.noise_dim, device=device)
                    a_next = target_actor.forward_expanded(no_i, z).view(B * n_boot, act_dim)
                    o_rep = no_i.unsqueeze(1).expand(B, n_boot, obs_dim).reshape(B * n_boot, obs_dim)
                    q_next = torch.min(
                        q_scalar(qf1_target(o_rep, a_next)), q_scalar(qf2_target(o_rep, a_next))
                    ).view(B, n_boot).mean(dim=1)
                    rewards_h.append(r_i)
                    boundary_h.append(bnd_i)
                    # A boundary with no usable successor observation cannot be
                    # bootstrapped; treat it as terminal rather than fabricate a value.
                    nonterm_h.append((1.0 - term_i) * bv_i)
                    qnext_h.append(q_next)
                # C_i = 1 iff every earlier step in the window stayed inside the episode.
                cont = torch.ones(B, device=device)
                for i in range(Hh):
                    cont_h.append(cont)
                    cont = cont * (1.0 - boundary_h[i])

                targets = []
                for h in range(Hh):
                    y = torch.zeros(B, device=device)
                    for i in range(h + 1):
                        y = y + gamma_pows[i] * cont_h[i] * rewards_h[i]
                        # Bootstrap at the first boundary inside the window, or at i == h.
                        stop = boundary_h[i] if i < h else torch.ones(B, device=device)
                        y = y + (
                            gamma_pows[i] * args.gamma * cont_h[i] * stop * nonterm_h[i] * qnext_h[i]
                        )
                    targets.append(y)
                target_scalars = torch.stack(targets, dim=1)                 # (B, H)
                target_probs = hl_support.project(target_scalars) if is_hlgauss else None

            # ---- critic update ----
            logits1 = qf1(obs0, act0)
            logits2 = qf2(obs0, act0)
            # Per-horizon loss; head 0 (the Q used for bootstrapping and grad_a Q)
            # always carries weight 1, horizons h>=1 carry mtp_aux_coef.
            ce_h1 = critic_head_loss(logits1, target_probs, target_scalars)   # (B, H)
            ce_h2 = critic_head_loss(logits2, target_probs, target_scalars)
            ce_w = torch.full((Hh,), args.mtp_aux_coef, device=device)
            ce_w[0] = 1.0
            critic_loss = ((ce_h1 + ce_h2) * ce_w).sum(-1).mean()
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            # Clipped per critic: the paper trains them as two independent regressions.
            critic_gn = max(
                float(nn.utils.clip_grad_norm_(qf1.parameters(), args.max_grad_norm)),
                float(nn.utils.clip_grad_norm_(qf2.parameters(), args.max_grad_norm)),
            )
            critic_optimizer.step()

            # ---- actor update: Wasserstein direction matching ----
            eps = torch.randn(B, K, actor.noise_dim, device=device)
            a1 = actor.forward_expanded(obs0, eps)                            # (B, K, A), has grad
            a0 = a1.detach().requires_grad_(True)
            obs_rep = obs0.unsqueeze(1).expand(B, K, obs_dim).reshape(B * K, obs_dim)
            a0_flat = a0.view(B * K, act_dim)
            q_min = torch.min(q_scalar(qf1(obs_rep, a0_flat)), q_scalar(qf2(obs_rep, a0_flat)))
            # grad only w.r.t. the action leaf: critic weights receive nothing.
            g = torch.autograd.grad(q_min.sum(), a0)[0]                       # (B, K, A)
            if args.normalize_action_grad:
                g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            delta_star = args.eta * g + xi_std * torch.randn_like(g)
            # `delta` is IDENTICALLY ZERO by value (a0 is a1 detached), so this
            # scalar equals mean||Delta*||^2 and can never fall as the actor
            # improves -- do not read it as a loss curve. Its GRADIENT is the
            # real update: -2*Delta* . d a1/d theta, i.e. transport along Delta*.
            # debug/transport_achieved below is the metric that actually tracks
            # whether the step is landing.
            delta = a1 - a0.detach()
            actor_loss = (delta - delta_star).pow(2).sum(-1).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gn = float(nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm))
            # A clip that binds on every step means the proximal step has become a
            # normalized step, silently. Track it rather than infer it from actor_gn.
            clip_hits += actor_gn > args.max_grad_norm
            clip_total += 1
            actor_optimizer.step()

            polyak_update(qf1, qf1_target, args.polyak)
            polyak_update(qf2, qf2_target, args.polyak)
            polyak_update(actor, target_actor, args.polyak)

        # ---------------- log ----------------
        if global_step - last_log >= args.log_frequency * args.num_envs:
            last_log = global_step
            sps = int(global_step / (time.time() - start_time))
            print("SPS:", sps)
            writer.add_scalar("charts/SPS", sps, global_step)
            if not has_trained:
                continue
            with torch.no_grad():
                q_pred = q_scalar(logits1)
            writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
            # Head 0 alone: the only head feeding the bootstrap and grad_a Q.
            writer.add_scalar("losses/critic_loss_h0", ce_h1[:, 0].mean().item(), global_step)
            if is_hlgauss:
                # Support adequacy: mass piling on the end bins means the return
                # has outgrown [v_min, v_max] and the critic is saturating.
                edge_mass = (target_probs[..., 0] + target_probs[..., -1]).mean().item()
                writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
            # = mean||Delta*||^2, NOT a descending loss curve (see the comment
            # at the actor update). Kept for scale, not for convergence.
            writer.add_scalar("losses/actor_transport_sq", actor_loss.item(), global_step)
            # Fraction of the requested transport the actor actually achieved,
            # projected onto Delta*. ~0 => the step is being eaten (clip,
            # tanh saturation, or Delta* far outside the reachable set).
            with torch.no_grad():
                achieved = actor.forward_expanded(obs0, eps) - a0.detach()
                frac = (achieved * delta_star).sum(-1) / delta_star.pow(2).sum(-1).clamp_min(1e-12)
            writer.add_scalar("debug/transport_achieved", frac.mean().item(), global_step)
            writer.add_scalar("losses/actor_grad_norm", actor_gn, global_step)
            writer.add_scalar("losses/critic_grad_norm", critic_gn, global_step)
            writer.add_scalar("debug/actor_clip_frac", clip_hits / max(clip_total, 1), global_step)
            clip_hits = clip_total = 0
            writer.add_scalar("debug/q_mean", q_pred.mean().item(), global_step)
            writer.add_scalar("debug/td_target_mean", target_scalars[:, 0].mean().item(), global_step)
            # ||Delta*|| >> the reachable action displacement means the proximal
            # step has degenerated into pure grad_a Q ascent; worth watching.
            writer.add_scalar("debug/delta_star_norm", delta_star.norm(dim=-1).mean().item(), global_step)
            writer.add_scalar("debug/action_grad_norm", g.norm(dim=-1).mean().item(), global_step)
            writer.add_scalar("debug/plugin_entropy", ent_mean, global_step)
            writer.add_scalar("debug/buffer_size", buf.size(), global_step)

    envs.close()
    writer.close()

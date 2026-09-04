# PPO + IterThink v24 beta d4-HL-Gauss symlog critic + DG-MODULATED TRUST REGION v12.
#
# WHY v12. The matched-carrier control (v10 c00) showed the aux loss is
# ~neutral at 1M (3496-3546 vs control 3650) with a small dose-monotone edge
# at 500k (1672 -> 1834) — because approx_kl is PINNED at the 0.03 leash, any
# auxiliary force merely SUBSTITUTES for PPO epochs instead of adding. The
# leash is the binding constraint; to beat the carrier's ceiling, DG must
# control what the leash doesn't cap: THE BUDGET ITSELF.
# v12 = budget reallocation across TIME (the paper's Prop 2 mechanism, third
# carrier translation): at each iteration start, probe the replay buffer for
# the gate's pos/neg separation sep = E[w|U>0] - E[w|U<0] under the current
# policy. sep high vs its own EMA = the last updates moved the policy in
# value-relevant directions (breakthrough momentum) -> loosen this iteration's
# target_kl (up to kl_mod_hi x); sep low = movement is noise -> tighten
# (kl_mod_lo x). The [0.5, 2] cap + EMA-relative reference break the
# loosen->stale->loosen feedback that collapsed v5. Run against a STATIC
# loose-leash control (--kl-mod off --target-kl 0.06) to attribute timing
# information beyond "just loosen".
#
# WHY v11. v10 (both coef arms) BEAT baseline at 500k (1707/1819 vs 1500) but
# lagged at 1M (3546/3532 vs 4063) with approx_kl pinned at the 0.03 leash and
# the two arms indistinguishable despite 3.3x coef — the signature of KL-BUDGET
# COMPETITION: under target_kl the aux does not add force, it substitutes for
# PPO epochs, so it only pays when its direction beats PPO's marginal epoch.
# v10 spent leash budget on EVERY replay sample, including the gate~0.5 mass
# where chi = U*ell carries zero delight evidence — plain off-policy PG noise.
# v11 masks the aux to confident delight only: weight = w * 1[|chi| > chi_min]
# (default 1.0). Clear breakthroughs/blunders still get corrected; the noise
# floor no longer taxes the KL budget PPO needs for sharpening.
#
# WHY v10 (carrier inversion). The replay-DG line (v5-v9) proved Algorithm 2's
# gate works as a ratio-free off-policy corrector when fed replay data — every
# health metric clean, gate decisively separating good/bad stale actions
# (pos/neg 0.65/0.31), and the pure-DG actor learned to 4080@4M ON ITS OWN —
# but the whole family saturates at ~50% of the PPO baseline because the gated
# vanilla-PG update lacks PPO's sharpening force (replay-DG entropy stalls at
# -2.5 while baseline reaches -6.2@2M; HalfCheetah >6k needs a sharp gait).
# Conversely, this PPO carrier reaches 9752@8M but is purely on-policy: each
# rollout's lessons are consumed once and discarded.
# v10 composes the proven halves instead of choosing: the UNCHANGED v24 PPO
# surrogate (clip 0.2/0.28, 10 epochs, target_kl 0.03) does the sharpening,
# and the replay-DG term rides as an AUXILIARY actor loss
#   L = L_ppo + dg_aux_coef * (-(sg(w) * U~ * log pi(a_old|s_old)).mean())
#   w = sigmoid(U~ * clip(-log pi, +-10)),  U~ = minibatch-standardized TD(0)
#       under a frozen target critic (re-evaluated FRESH every sample — the
#       v8 lesson: stored credit stalls), actions sampled uniformly from a
#       262k-transition replay ring.
# Mechanism: cross-context budget reallocation over the recent past (paper
# Prop 2) — replayed actions the current policy now disfavors but that the
# fresh critic scores well are amplified (breakthrough consolidation), stale
# bad ones pushed down (blunder suppression) — signal PPO's single-rollout
# window simply cannot see. dg_aux_coef brackets the dosage.
# cc_budget now defaults OFF (attributed harmful in v3's runs: -10/-20% @2M).
#
# WHY v3. Every WITHIN-state delight gate tried (dg_v1 raw, entref_v1 centered,
# entref_tail_v2 tail-only at tau=0.5/1.0) underperformed the PPO baseline, and
# the damage was monotone in gate strength: the within-state asymmetric gate is,
# on a dense-reward env, just an implicit ADAPTIVE ENTROPY BONUS (its even-in-U
# component ~ E[U^2*ell]/2 * grad H), and HalfCheetah punishes any entropy bonus
# (baseline entropy -6.2 @2M vs -2.3/-3.3 gated; returns 6583 vs 4009/4438).
# As gate -> identity, score -> baseline FROM BELOW: no crossover. Abandoned.
#
# v3 keeps the OTHER DG mechanism — the one the paper proves persists at
# infinite data (Prop 2): cross-context gradient-budget reallocation. PG
# allocates update budget toward contexts the policy already masters; DG
# compresses budget toward equality, accelerating hard contexts. Continuous
# translation: a PER-STATE (not per-action) multiplicative weight
#   m(s) = 2*sigmoid( z(R(s)) / cc_eta ),   z = minibatch z-score, detached
# where the "unmastered-ness" reference R(s) is:
#   cc_budget="entropy": R = H(s), the policy's own per-state entropy (closed
#       form for Beta) — uncertain policy = unmastered state (paper-faithful:
#       discrete -log p_n(correct) ~ how unsettled the policy is at context n);
#   cc_budget="sigma":   R = sigma(s), the categorical critic's per-state
#       return std — outcome still unpredictable = unmastered.
# m(s) scales BOTH advantage signs at a state equally, so it injects ZERO
# within-state entropy force by construction — it only moves gradient budget
# from solved states to frontier states. Neutral point: z=0 -> m=1 exact;
# E[m] ~= 1 per minibatch (self-normalizing). PPO carrier unchanged
# (rankgauss U, ratio clip 0.2/0.28, 10 epochs, target_kl 0.03).
# The v2 tail gate is kept behind delightful_pg (now default FALSE) for
# later composition once cc-budget is attributed.
#
# --- inherited v2 notes ---
# WHY v2. entref_v1 centered DG surprisal at the per-state entropy
# (ell~ = -log pi - H(s), E[ell~|s]=0) and gated EVERY sample with
# weight = 2*sigmoid(U*ell~). That bulk coupling has a non-vanishing
# even-in-U component: E[U^2 * ell~ * grad log pi]/2 ~= 0.5 * grad H — an
# accidental entropy-maximization force ~50x a sane ent_coef. Measured: the
# baseline's policy entropy fell -1.8 -> -6.2 over 2M steps while sym v1 was
# pinned at -0.3 (never concentrated, 757 @2M) and amplify_only got half the
# force (-1.3, 2443 @2M). The bulk of the gate is poison; only the tail story
# (amplify rare successes, mute rare blunders) is the paper's actual benefit.
#
# FIX v2 (tail-only gate): act only on genuinely surprising actions:
#   ell^ = clip(ell~ - tau, 0, C)        (tau ~= 1 nat above typical)
#   weight = 2*sigmoid(U * ell^ / eta)   (== 1 EXACTLY for the non-tail bulk)
#   typical or mode action  -> ell^ = 0 -> weight = 1 (exact PPO)
#   tail action, U>0        -> breakthrough, amplified (up to 2x)
#   tail action, U<0        -> blunder, muted (down to 0x)
# Residual implicit entropy push ~= E[U^2 * ell^]/2 ~= 0.01 — a small,
# ADAPTIVE exploration bonus paid only where the policy itself flagged the
# action as rare. Beta-native -log pi and H are closed forms; the env-rescale
# Jacobian cancels in the subtraction.
#   U = existing rankgauss/minibatch-normalized policy advantage (std ~= 1).
#   PPO's clipped policy terms are multiplied by stopgrad(weight); carrier
#   (ratio clip 0.2/0.28, 10 epochs, target_kl 0.03) unchanged.
# dg_gate_mode="amplify_only" gates only U>0 samples (pure rare-success
# amplification; weight >= 1 there, so exploitation signal is never reduced).
#
# Base symlog behavior:
#   scalar GAE return -> symlog -> Gaussian-smoothed categorical target.
#   logits -> E[symlog bin center] -> symexp -> scalar value for GAE/bootstrap.
# Reward normalization/clipping, the raw support range [-10, 10], sigma ratio,
# critic init, and PPO settings stay matched to d4-HL-Gauss. The support edges
# are linear in symlog(v_min)..symlog(v_max), matching hl_gauss_pytorch's
# transform contract while preserving d4's raw normalized-return range.
#
# Base: ppo_continuous_action_iterthink_v24_dist.py / iterthink_v24_beta_s1.
# Only intended algorithmic change: critic target projection style.
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
    if not args.value_symlog:
        return args.v_min, args.v_max
    bounds = torch.tensor([args.v_min, args.v_max], dtype=torch.float32)
    return symlog(bounds).tolist()


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
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ≈ N(0, tau^2), sharp at 0
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 2.0  # Dreamer4 / hl_gauss_pytorch default

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

    # v2 tail gate (within-state; attributed harmful alone — off by default).
    delightful_pg: bool = False
    dg_eta: float = 1.0
    dg_surprisal_clip: float = 10.0
    dg_tail_tau: float = 1.0   # gate only samples with ell~ > tau nats above typical
    dg_gate_mode: str = "sym"  # "sym" (gate all samples) | "amplify_only" (gate U>0 only)
    # v3 cross-context Delightful budget gate (paper Prop 2, per-state weight).
    cc_budget: str = "off"  # "entropy" | "sigma" | "off" (v3 attributed it harmful)
    cc_eta: float = 1.0         # gate temperature on the z-scored reference
    # v10 replay-DG auxiliary (see header). Gate/eta/clip args above are shared.
    dg_aux_coef: float = 0.5          # aux loss weight; 0 disables everything below
    # v11: only samples with |chi| = |U*ell| above this contribute to the aux
    # loss (gate ~0.5 elsewhere = zero delight evidence; pushing them is plain
    # off-policy PG noise that competes with PPO for the target_kl budget).
    dg_aux_chi_min: float = 1.0       # 0 disables the mask (= v10)
    # v12: DG-modulated trust region (see iteration-start probe). "dg" scales
    # this iteration's target_kl by clip(sep/EMA(sep), lo, hi) where sep is the
    # gate's pos/neg separation on a replay probe; "off" = static target_kl.
    kl_mod: str = "dg"
    kl_mod_lo: float = 0.5
    kl_mod_hi: float = 2.0
    kl_mod_ema: float = 0.95          # EMA decay for the separation reference
    dg_u_source: str = "td0"          # "td0" | "nstep" (td0 won the v7/v9 ablation)
    nstep_horizon: int = 5
    norm_td_adv: bool = True          # standardize U per aux minibatch (v7 lesson)
    dg_chi_whiten: bool = False
    replay_size: int = 262_144        # transitions (~8 rollouts)
    target_update_interval: int = 100 # optimizer steps between hard target-critic syncs

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
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
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
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
        # Categorical critic with a PEAKED init: small weight + Gaussian-logit
        # bias so the initial value distribution starts as a sharp zero-return
        # prior instead of a high-variance uniform distribution. The bias lives
        # in the categorical coordinate, which is symlog-space when enabled.
        self.critic_head = layer_init(nn.Linear(H, args.num_bins), std=0.1)
        with torch.no_grad():
            support_min, support_max = value_support_bounds(args)
            edge_width = (support_max - support_min) / args.num_bins
            z = torch.linspace(
                support_min + 0.5 * edge_width,
                support_max - 0.5 * edge_width,
                args.num_bins,
            )
            self.critic_head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)
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
        return self.critic_head(critic_feat)

    def get_action_and_value(self, x, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_logits = self.critic_head(critic_feat)
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
        return action, z, log_prob, entropy, value_logits

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


def delightful_entref_tail_gate(advantages, logprob, entropy, tau=1.0, eta=1.0, clip_bound=10.0):
    """Tail-only entropy-referenced DG gate for continuous actions.

    ell~ = -log pi(a|s) - H(s) is the centered surprisal (E[ell~|s] = 0);
    ell^ = clip(ell~ - tau, 0, C) keeps only the rare tail, so w = sigmoid(
    U * ell^ / eta) is EXACTLY 0.5 (identity after the 2x rescale) for every
    typical or mode action and only tail actions are gated. logprob and
    entropy must share the same coordinate space (both Beta-native here: the
    env-rescale Jacobian constant cancels in the subtraction). Caller detaches
    both inputs; the training loss uses stopgrad(w) so the gradient stays
    w * U * grad log pi.
    """
    if eta <= 0:
        raise ValueError("dg_eta must be positive")
    if clip_bound <= 0:
        raise ValueError("dg_surprisal_clip must be positive")
    if tau < 0:
        raise ValueError("dg_tail_tau must be non-negative")
    raw_surprisal = -logprob - entropy
    tail_surprisal = (raw_surprisal - tau).clamp(0.0, clip_bound)
    delight = advantages * tail_surprisal
    gate = torch.sigmoid(delight / eta)
    return gate, tail_surprisal, raw_surprisal, delight


def delightful_aux_gate(advantages, logprob, eta=1.0, clip_bound=10.0, chi_whiten=False):
    """Algorithm 2 DG gate on raw density surprisal (v10 aux; replay actions).

    ell = clip(-log pi(a|s), -C, C), chi = U * ell, w = sigmoid(chi / eta).
    Unlike the entref tail gate above, NO entropy centering: replayed actions
    carry genuine surprisal spread under the current policy, which is the
    gate's signal (on-policy this degenerates — hence aux-on-replay only).
    """
    if eta <= 0:
        raise ValueError("dg_eta must be positive")
    if clip_bound <= 0:
        raise ValueError("dg_surprisal_clip must be positive")
    raw_surprisal = -logprob
    clipped_surprisal = raw_surprisal.clamp(-clip_bound, clip_bound)
    delight = advantages * clipped_surprisal
    if chi_whiten:
        delight = (delight - delight.mean()) / (delight.std(unbiased=False) + 1e-8)
    gate = torch.sigmoid(delight / eta)
    return gate, clipped_surprisal, raw_surprisal, delight


def dg_aux_density_logprob(agent, ppo_logprob):
    """Beta native logprob -> env action-density logprob (rescale Jacobian).

    Surprisal is an absolute density, so the z -> env linear rescale
    contributes -sum(log(high-low)) to log pi(a|s); PPO ratios drop this
    constant but the DG gate cannot.
    """
    if agent.actor_dist == "beta":
        scale_logdet = torch.log(agent.action_high - agent.action_low).sum()
        return ppo_logprob - scale_logdet
    return ppo_logprob


def list_mean(xs, fallback=0.0):
    return float(np.mean(xs)) if xs else float(fallback)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch")
    assert args.dg_gate_mode in ("sym", "amplify_only")
    assert args.cc_budget in ("entropy", "sigma", "off")
    assert args.cc_eta > 0
    assert args.dg_u_source in ("td0", "nstep")
    assert args.kl_mod in ("dg", "off")
    assert 0 < args.kl_mod_lo <= 1.0 <= args.kl_mod_hi
    assert args.nstep_horizon >= 1
    assert args.dg_aux_coef >= 0
    assert args.replay_size >= args.batch_size, "replay must hold at least one rollout"
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope == "batch"), \
        "norm_adv_scope=batch requires adv_transform_scope=batch"
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
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # v10: frozen target critic for the aux TD advantage (re-evaluated fresh
    # every sample; hard-synced every target_update_interval optimizer steps).
    target_agent = Agent(envs, args).to(device)
    target_agent.load_state_dict(agent.state_dict())
    for p in target_agent.parameters():
        p.requires_grad_(False)
    opt_steps = 0
    kl_mod_sep_ema = None  # v12: EMA of the gate's pos/neg separation probe

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
    if args.value_symlog:
        raw_support = symexp(support)
        raw_edges = symexp(hl_support.edges)
        raw_bin_widths = raw_edges[1:] - raw_edges[:-1]
    else:
        raw_support = support
        raw_bin_widths = torch.full_like(support, bin_width)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    next_obs_buf = torch.zeros_like(obs)
    next_nonterm_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # v10 replay ring buffer for the DG aux term (GPU-resident, ~70MB at the
    # 2^18 default). Native Beta z stored so the current policy re-evaluates
    # log pi(z|s) directly; behavior logprob kept for staleness telemetry.
    r_obs = torch.zeros((args.replay_size,) + envs.single_observation_space.shape, device=device)
    r_next_obs = torch.zeros_like(r_obs)
    r_z = torch.zeros((args.replay_size,) + envs.single_action_space.shape, device=device)
    r_rew = torch.zeros(args.replay_size, device=device)
    r_nonterm = torch.zeros(args.replay_size, device=device)
    r_lp = torch.zeros(args.replay_size, device=device)
    # n-step window columns (dg_u_source="nstep"): discounted reward sum,
    # window-end obs, bootstrap discount gamma^h * alive.
    r_nR = torch.zeros(args.replay_size, device=device)
    r_nobs = torch.zeros_like(r_obs)
    r_ndisc = torch.zeros(args.replay_size, device=device)
    r_ptr = 0
    r_filled = 0

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

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                values[step] = hl_support.to_scalar(value_logits)
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            # v10 replay: s' and the bootstrap mask. On autoreset steps next_obs
            # is the NEW episode's first obs, but nonterm=0 masks the bootstrap.
            next_obs_buf[step] = next_obs
            next_nonterm_buf[step] = 1.0 - next_done

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
                # Sample a' ~ π(·|s_T) for the bootstrap entropy (SAC's single-sample).
                _, _, boot_logprob, _, boot_logits = agent.get_action_and_value(next_obs)
                bootstrap_logits = boot_logits
                bootstrap_probs = torch.softmax(bootstrap_logits, dim=-1)       # (B, n) = Z(s_T)
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                bootstrap_logits = agent.get_value(next_obs)
                bootstrap_probs = torch.softmax(bootstrap_logits, dim=-1)       # (B, n) = Z(s_T)
                next_value_bonus = None
            next_value = hl_support.to_scalar(bootstrap_logits).reshape(1, -1)
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
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
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). Unbiased (state-dependent baseline); its large
            # magnitude is harmless — rankgauss at the optim step rank-normalizes it while
            # PRESERVING the entropy-induced reordering (the actual exploration signal).
            if auto_alpha:
                policy_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * (nextvalues + next_value_bonus[t]) * nextnonterminal - values[t]
                    policy_adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            else:
                policy_adv = advantages
            # Critic target: Dreamer4-style scalar-return HL-Gauss. GAE computes
            # the scalar λ-return; the value encoder projects that scalar target
            # into a Gaussian-smoothed categorical distribution over fixed bins.
            target_probs = hl_support.project(returns)
            # Per-state return std sigma(s_t) in raw return units, matching the
            # GAE scale consumed by tanh_std. For symlog values, logits live in
            # symlog coordinates but PPO bootstraps from symexp(E[z]).
            sigma = (value_probs * (raw_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            if args.value_symlog:
                value_coord = symlog(values).clamp(hl_support.v_min, hl_support.v_max)
                floor_idx = ((value_coord - hl_support.v_min) / bin_width).floor().long().clamp(0, args.num_bins - 1)
                sigma_floor = args.sigma_floor_bins * raw_bin_widths[floor_idx]
            else:
                sigma_floor = args.sigma_floor_bins * bin_width
            sigma = torch.maximum(sigma, sigma_floor)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            returns_coord = symlog(returns) if args.value_symlog else returns
            cdf_frac = ((returns_coord.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

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

        # v10: replay insertion (before the update loop, so iteration 1 samples
        # a fully on-policy buffer). n-step windows precomputed within-rollout:
        # entry (t,e) accumulates gamma^k r_{t+k} while no done intervenes and
        # t+k stays in the rollout; ndisc = gamma^h (0 if the window hit a done);
        # nobs_w = obs after the last included transition. Tail entries get
        # valid shorter windows, not bias.
        with torch.no_grad():
            nR = rewards.clone()
            ndisc = args.gamma * next_nonterm_buf
            nobs_w = next_obs_buf.clone()
            alive = next_nonterm_buf.clone()
            for k in range(1, args.nstep_horizon):
                head = args.num_steps - k
                if head <= 0:
                    break
                ext = alive[:head]
                nR[:head] = nR[:head] + (args.gamma ** k) * ext * rewards[k:]
                extm = ext.unsqueeze(-1) > 0
                nobs_w[:head] = torch.where(extm, next_obs_buf[k:], nobs_w[:head])
                ndisc[:head] = torch.where(
                    ext > 0,
                    (args.gamma ** (k + 1)) * next_nonterm_buf[k:],
                    ndisc[:head],
                )
                alive[:head] = ext * next_nonterm_buf[k:]

            n_new = b_obs.shape[0]
            ins = (r_ptr + torch.arange(n_new, device=device)) % args.replay_size
            r_obs[ins] = b_obs
            r_next_obs[ins] = next_obs_buf.reshape((-1,) + envs.single_observation_space.shape)
            r_z[ins] = b_latent_zs
            r_rew[ins] = rewards.reshape(-1)
            r_nonterm[ins] = next_nonterm_buf.reshape(-1)
            r_lp[ins] = b_logprobs
            r_nR[ins] = nR.reshape(-1)
            r_nobs[ins] = nobs_w.reshape((-1,) + envs.single_observation_space.shape)
            r_ndisc[ins] = ndisc.reshape(-1)
            r_ptr = (r_ptr + n_new) % args.replay_size
            r_filled = min(r_filled + n_new, args.replay_size)

        # v12: DG-modulated trust region. Probe the replay buffer (no_grad)
        # for delight CONCENTRATION — the gate's pos/neg separation
        #   sep = E[w | U>0] - E[w | U<0]
        # under the current policy. High vs its own EMA = recent updates moved
        # the policy in value-relevant directions (breakthrough momentum):
        # loosen this iteration's KL leash, up to kl_mod_hi x. Low = movement
        # is noise: tighten toward kl_mod_lo x. The multiplier cap + EMA
        # reference break the loosen->stale->loosen feedback loop that killed
        # v5. With kl_mod="off" the leash is the static args.target_kl.
        target_kl_eff = args.target_kl
        kl_mod_mult = 1.0
        if args.kl_mod == "dg" and args.target_kl is not None:
            with torch.no_grad():
                pidx = torch.randint(0, r_filled, (4 * args.minibatch_size,), device=device)
                pobs = r_obs[pidx]
                _, _, p_lp, _, _ = agent.get_action_and_value(pobs, r_z[pidx])
                p_vs = hl_support.to_scalar(target_agent.get_value(pobs))
                p_vsp = hl_support.to_scalar(target_agent.get_value(r_next_obs[pidx]))
                p_u = r_rew[pidx] + args.gamma * p_vsp * r_nonterm[pidx] - p_vs
                p_u = (p_u - p_u.mean()) / (p_u.std(unbiased=False) + 1e-8)
                p_gate, _, _, _ = delightful_aux_gate(
                    p_u,
                    dg_aux_density_logprob(agent, p_lp),
                    eta=args.dg_eta,
                    clip_bound=args.dg_surprisal_clip,
                )
                p_pos = p_u > 0
                dg_sep = (p_gate[p_pos].mean() - p_gate[~p_pos].mean()).item()
            if kl_mod_sep_ema is None:
                kl_mod_sep_ema = dg_sep
            kl_mod_mult = float(np.clip(dg_sep / (kl_mod_sep_ema + 1e-8), args.kl_mod_lo, args.kl_mod_hi))
            kl_mod_sep_ema = args.kl_mod_ema * kl_mod_sep_ema + (1.0 - args.kl_mod_ema) * dg_sep
            target_kl_eff = args.target_kl * kl_mod_mult

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        aux_gate_means = []
        aux_keep_fracs = []
        aux_gate_pos_means = []
        aux_gate_neg_means = []
        aux_staleness = []
        aux_surprisal_stds = []
        aux_u_stds = []
        dg_delight_means = []
        dg_delight_stds = []
        dg_surprisal_means = []
        dg_surprisal_stds = []
        dg_weight_means = []
        dg_tail_fracs = []
        cc_weight_means = []
        cc_weight_mins = []
        cc_weight_maxes = []
        cc_pos_corrs = []
        dg_surprisal_maxes = []
        dg_raw_surprisal_maxes = []
        dg_gate_means = []
        dg_gate_mins = []
        dg_gate_maxes = []
        dg_pos_gate_means = []
        dg_neg_gate_means = []
        dg_clipped_surprisal_fracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

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

                if args.delightful_pg:
                    # Beta path: newlogprob and entropy are both native-space closed
                    # forms, so ell~ = -logprob - H(s) is exact and Jacobian-free.
                    dg_gate, dg_surprisal, dg_raw_surprisal, dg_delight = delightful_entref_tail_gate(
                        mb_advantages,
                        newlogprob.detach(),
                        entropy.detach(),
                        tau=args.dg_tail_tau,
                        eta=args.dg_eta,
                        clip_bound=args.dg_surprisal_clip,
                    )
                    # 2x so the neutral gate (chi=0 -> 0.5) is an exact identity.
                    dg_weight = 2.0 * dg_gate.detach()
                    if args.dg_gate_mode == "amplify_only":
                        dg_weight = torch.where(
                            mb_advantages > 0, dg_weight, torch.ones_like(dg_weight)
                        )
                    with torch.no_grad():
                        pos_mask = mb_advantages > 0
                        neg_mask = mb_advantages < 0
                        dg_delight_means.append(dg_delight.mean().item())
                        dg_delight_stds.append(dg_delight.std(unbiased=False).item())
                        dg_surprisal_means.append(dg_surprisal.mean().item())
                        dg_surprisal_stds.append(dg_surprisal.std(unbiased=False).item())
                        dg_weight_means.append(dg_weight.mean().item())
                        dg_surprisal_maxes.append(dg_surprisal.max().item())
                        dg_raw_surprisal_maxes.append(dg_raw_surprisal.max().item())
                        dg_gate_means.append(dg_gate.mean().item())
                        dg_gate_mins.append(dg_gate.min().item())
                        dg_gate_maxes.append(dg_gate.max().item())
                        if pos_mask.any():
                            dg_pos_gate_means.append(dg_gate[pos_mask].mean().item())
                        if neg_mask.any():
                            dg_neg_gate_means.append(dg_gate[neg_mask].mean().item())
                        # tail_frac: share of samples actually gated (ell^ > 0);
                        # clipped_frac: share hitting the upper clip bound C.
                        dg_tail_fracs.append((dg_surprisal > 0).float().mean().item())
                        dg_clipped_surprisal_fracs.append(
                            (dg_surprisal >= args.dg_surprisal_clip).float().mean().item()
                        )
                else:
                    dg_weight = torch.ones_like(mb_advantages)

                if args.cc_budget != "off":
                    # Cross-context budget gate m(s) = 2*sigmoid(z(R)/eta): shift
                    # gradient budget toward unmastered states. Per-state, sign-
                    # neutral within a state; minibatch z-score keeps E[m] ~= 1.
                    with torch.no_grad():
                        if args.cc_budget == "entropy":
                            cc_ref = entropy.detach()
                        else:  # "sigma": critic per-state return std (raw units)
                            cc_ref = b_sigma[mb_inds]
                        cc_z = (cc_ref - cc_ref.mean()) / (cc_ref.std(unbiased=False) + 1e-8)
                        cc_weight = 2.0 * torch.sigmoid(cc_z / args.cc_eta)
                        cc_weight_means.append(cc_weight.mean().item())
                        cc_weight_mins.append(cc_weight.min().item())
                        cc_weight_maxes.append(cc_weight.max().item())
                        # corr(m, U>0): should hover ~0 — the gate must not act
                        # as a disguised advantage reweighter.
                        pos = (mb_advantages > 0).float()
                        cc_pos_corrs.append(
                            ((cc_weight - cc_weight.mean()) * (pos - pos.mean())).mean().item()
                        )
                    dg_weight = dg_weight * cc_weight

                # Asymmetric "clip-higher" when clip_coef_high is set: looser upper bound
                # gives positive-advantage actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -dg_weight * mb_advantages * ratio
                pg_loss2 = -dg_weight * mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # v10 AUX: gated replay-DG corrector added to the PPO surrogate.
                # No importance ratio — the detached gate IS the off-policy
                # correction (Algorithm 2). U re-evaluated fresh under the
                # frozen target critic every sample (the v8 stale-credit lesson).
                if args.dg_aux_coef > 0:
                    ridx = torch.randint(0, r_filled, (args.minibatch_size,), device=device)
                    rsx = r_obs[ridx]
                    _, _, aux_logprob, _, _ = agent.get_action_and_value(rsx, r_z[ridx])
                    with torch.no_grad():
                        v_s = hl_support.to_scalar(target_agent.get_value(rsx))
                        if args.dg_u_source == "nstep":
                            v_b = hl_support.to_scalar(target_agent.get_value(r_nobs[ridx]))
                            aux_u = r_nR[ridx] + r_ndisc[ridx] * v_b - v_s
                        else:  # "td0"
                            v_sp = hl_support.to_scalar(target_agent.get_value(r_next_obs[ridx]))
                            aux_u = r_rew[ridx] + args.gamma * v_sp * r_nonterm[ridx] - v_s
                        aux_u_stds.append(aux_u.std(unbiased=False).item())
                        if args.norm_td_adv:
                            aux_u = (aux_u - aux_u.mean()) / (aux_u.std(unbiased=False) + 1e-8)
                    aux_density_lp = dg_aux_density_logprob(agent, aux_logprob)
                    aux_gate, aux_surprisal, _, aux_chi = delightful_aux_gate(
                        aux_u,
                        aux_density_lp.detach(),
                        eta=args.dg_eta,
                        clip_bound=args.dg_surprisal_clip,
                        chi_whiten=args.dg_chi_whiten,
                    )
                    # v11: spend KL-leash budget ONLY on confident delight.
                    # |chi| < chi_min means the gate is ~0.5 = no evidence; v10
                    # pushed those samples anyway (plain off-policy PG noise that
                    # eats target_kl budget PPO would have used). Mask them out;
                    # only clear breakthroughs/blunders contribute.
                    aux_weight = aux_gate
                    if args.dg_aux_chi_min > 0:
                        chi_mask = (aux_chi.abs() > args.dg_aux_chi_min).float()
                        aux_weight = aux_gate * chi_mask
                        aux_keep_fracs.append(chi_mask.mean().item())
                    aux_pg_loss = -(aux_weight.detach() * aux_u * aux_logprob).mean()
                    pg_loss = pg_loss + args.dg_aux_coef * aux_pg_loss
                    with torch.no_grad():
                        aux_gate_means.append(aux_gate.mean().item())
                        posm = aux_u > 0
                        if posm.any():
                            aux_gate_pos_means.append(aux_gate[posm].mean().item())
                        if (~posm).any():
                            aux_gate_neg_means.append(aux_gate[~posm].mean().item())
                        aux_staleness.append((r_lp[ridx] - aux_logprob).mean().item())
                        aux_surprisal_stds.append(aux_surprisal.std(unbiased=False).item())

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

                # v10: hard target-critic sync for the aux advantage baseline.
                opt_steps += 1
                if opt_steps % args.target_update_interval == 0:
                    target_agent.load_state_dict(agent.state_dict())

            if args.target_kl is not None and approx_kl > target_kl_eff:
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
        if args.dg_aux_coef > 0:
            aux_gate_mean = list_mean(aux_gate_means)
            writer.add_scalar("debug/aux_gate_mean", aux_gate_mean, global_step)
            writer.add_scalar("debug/aux_gate_pos_mean", list_mean(aux_gate_pos_means, aux_gate_mean), global_step)
            writer.add_scalar("debug/aux_gate_neg_mean", list_mean(aux_gate_neg_means, aux_gate_mean), global_step)
            writer.add_scalar("debug/aux_staleness", list_mean(aux_staleness), global_step)
            writer.add_scalar("debug/aux_surprisal_std", list_mean(aux_surprisal_stds), global_step)
            writer.add_scalar("debug/aux_u_std_raw", list_mean(aux_u_stds), global_step)
            writer.add_scalar("debug/aux_pg_loss", aux_pg_loss.item(), global_step)
            if args.dg_aux_chi_min > 0:
                writer.add_scalar("debug/aux_keep_frac", list_mean(aux_keep_fracs), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        if args.delightful_pg:
            gate_mean = list_mean(dg_gate_means)
            writer.add_scalar("debug/dg_delight_mean", list_mean(dg_delight_means), global_step)
            writer.add_scalar("debug/dg_delight_std", list_mean(dg_delight_stds), global_step)
            writer.add_scalar("debug/dg_surprisal_mean", list_mean(dg_surprisal_means), global_step)
            writer.add_scalar("debug/dg_surprisal_std", list_mean(dg_surprisal_stds), global_step)
            writer.add_scalar("debug/dg_weight_mean", list_mean(dg_weight_means), global_step)
            writer.add_scalar("debug/dg_tail_frac", list_mean(dg_tail_fracs), global_step)
            writer.add_scalar("debug/dg_surprisal_max", float(np.max(dg_surprisal_maxes)), global_step)
            writer.add_scalar("debug/dg_raw_surprisal_max", float(np.max(dg_raw_surprisal_maxes)), global_step)
            writer.add_scalar("debug/dg_gate_mean", gate_mean, global_step)
            writer.add_scalar("debug/dg_gate_min", float(np.min(dg_gate_mins)), global_step)
            writer.add_scalar("debug/dg_gate_max", float(np.max(dg_gate_maxes)), global_step)
            writer.add_scalar("debug/dg_gate_pos_mean", list_mean(dg_pos_gate_means, gate_mean), global_step)
            writer.add_scalar("debug/dg_gate_neg_mean", list_mean(dg_neg_gate_means, gate_mean), global_step)
            writer.add_scalar("debug/dg_clipped_surprisal_frac", list_mean(dg_clipped_surprisal_fracs), global_step)
        if args.cc_budget != "off":
            writer.add_scalar("debug/cc_weight_mean", list_mean(cc_weight_means), global_step)
            writer.add_scalar("debug/cc_weight_min", float(np.min(cc_weight_mins)), global_step)
            writer.add_scalar("debug/cc_weight_max", float(np.max(cc_weight_maxes)), global_step)
            writer.add_scalar("debug/cc_pos_corr", list_mean(cc_pos_corrs), global_step)
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

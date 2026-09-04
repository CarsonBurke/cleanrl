# PPO + IterThink v36 (SAC max-entropy IN GAE -- BASELINED future-entropy advantage). From v35.
#
# WHY v36. v35's binding floor RESCUED the terminating envs (Hopper ~990, Walker ~1790) but the
# soft-adv-in-GAE channel COLLAPSED HalfCheetah (~8800 -> ~350). Three independent red-team reviews
# (math, SAC-translation, empirical) converged on ONE implementation bug -- NOT "max-ent hurts HC"
# (SAC gets ~10-12k on HC WITH max-ent, so the mechanism is right; the translation was wrong):
#   v35 added the future-entropy bonus b_t = alpha*H(s_{t+1}) to the GAE *forward return* (the
#   nextvalues + b_t bootstrap) but the BASELINE values[t] and the critic target are ENTROPY-FREE.
#   So  A_soft[t] = A_rew[t] + gamma*alpha * SUM_k (gamma*lambda)^k H(s_{t+1+k}) -- the entropy term
#   is a discounted sum of FUTURE entropies that is NEVER baselined (SAC baselines Q_soft with V_soft;
#   we added the return half but never built the baseline half). This residual is mean-POSITIVE and
#   HORIZON-ACCUMULATING: on HalfCheetah (non-terminating, full 1000-step horizon, nonterminal==1)
#   it saturates to a near-constant pedestal ~0.8 normalized units at EVERY state, swamping HC's
#   small zero-mean reward advantage and -- because rankgauss_signmag takes sign(gae) BEFORE norm --
#   FLIPPING the advantage sign to +1 almost everywhere => the policy gradient direction is destroyed
#   (empirically: soft_adv_std_ratio 1.1->2.5, distpg_corr_with_gae -> ~0/negative, sign_agree ->
#   coin-flip, while the critic EV stayed healthy => the bug is purely in HOW THE ADVANTAGE IS FORMED).
#   Terminating Hopper/Walker truncate the sum at done (nonterminal=0) => smaller, signal-bearing bias
#   => they survived. The v35 alpha/2 split only SCALED the pedestal; it never CENTERED it.
# THE FIX (SAC-faithful): the SAME entropy added to the return must be SUBTRACTED by a baseline.
# v36 learns the future-entropy value with a separate SCALAR head V_ent (MSE; no categorical support
# to overflow -- which is exactly what sank the "softboot" attempt to put entropy in the C51 critic)
# and forms a properly BASELINED entropy advantage via its own GAE:
#     entropy reward  e_t = H(s_{t+1}) = -logpi(a_{t+1}|s_{t+1})        (raw, alpha-free)
#     ent_delta_t     = e_t + gamma*V_ent(s')*nonterm - V_ent(s_t)
#     ent_adv         = GAE(ent_delta)            # CENTERED: realized future entropy - predicted
#     policy_adv      = reward_adv + alpha * ent_adv      # zero-mean perturbation, NO pedestal
# V_ent regresses its own lambda-return (ent_adv + V_ent). The reward critic + reward GAE are
# UNTOUCHED (proven; EV stays healthy). This is SAC's decomposition Q_soft = Q_reward + V_entropy.
# NO DOUBLE-COUNT, so NO alpha/2 split: ent_adv credits action a_t for the entropy of FUTURE states
# (s_{t+1} onward); the direct -alpha*logpi(a_t) term supplies the CURRENT action's entropy gradient
# (which a detached advantage cannot). Disjoint timesteps -- each channel gets the FULL alpha, exactly
# as SAC (future entropy in the value, current entropy in the actor's -alpha*logpi).
# Hypothesis: with the pedestal removed, max-ent-in-GAE helps ALL THREE envs -- HC recovers toward
# its ~8800 (and beyond, if future-entropy credit genuinely aids exploration) while Hopper/Walker
# keep v35's gains. The binding floor + all other v35 machinery are unchanged (isolated change).
#
# ---- inherited v35 notes ----
# PPO + IterThink v35 (SAC max-entropy IN GAE + BINDING entropy floor). From v34.
#
# WHY v35. An empirical audit of v32-v34 on the TERMINATING envs (Hopper/Walker2d, stuck ~480-660)
# found the max-entropy mechanism was IMPLEMENTED but INERT, for one root cause: the alpha-dual
# NEVER BINDS. With the SAC parity target (target_entropy = -1.0*|A| = -3 Hopper / -6 Walker), the
# squashed entropy H lives FAR ABOVE the target the entire run (H/dim ~ 0.67 early, collapsing to
# ~ -0.18 Hopper / +0.26 Walker), so the dual gradient alpha*(H - target) is CONSTANT-SIGN POSITIVE
# => alpha only ever DECREASES => it bleeds to the floor. Two failures, ONE cause:
#   * alpha -> floor => the soft credit alpha*H -> ~0 => the soft-advantage-in-GAE channel is INERT
#     (exactly v28-31's measured soft_adv_std_ratio ~ 1.00). Entropy never reaches the policy.
#   * No entropy floor => sigma COLLAPSES prematurely (data: 0.5 -> 0.14 on Hopper) => the policy
#     converges to a deterministic never-survives gait while EV stays 0.95-0.99.
# THE FIX (pure SAC max-ent, no vanilla-PPO entropy): make the dual BIND as an entropy FLOOR. In
# SAC's fast off-policy regime -|A| is reachable from above; in our SLOW on-policy regime it is not,
# so the dual must instead target a REACHABLE level ABOVE the free-collapse floor. Then when H falls
# below target, alpha(H-target)<0 => alpha RISES => the actor is pushed to RE-EXPAND entropy: a real
# fixed point. With alpha held meaningful, alpha*H in the soft bootstrap V(s')+alpha*H(s') is on-scale
# and the soft-advantage GAE actually REORDERS the policy advantage (no longer inert).
# v35'S CHANGES from v34 (all serving the SAC-max-ent-in-GAE thesis):
#   (1) FULL SAC SOFT OBJECTIVE (both channels, decoupled). SAC's soft objective has TWO entropy
#       terms: future entropy in the soft-Q backup, and the direct -alpha*log_pi term on the current
#       action. Our channels are exactly that decomposition and do NOT double-count (different
#       timesteps): soft_adv=True puts FUTURE entropy IN THE GAE via the soft bootstrap V(s')+alpha*H(s')
#       ("max entropy in GAE"); direct_entropy=True adds the CURRENT-step -alpha*H actor gradient that
#       directly holds sigma up (the soft channel alone cannot -- current-state entropy is action-
#       independent and cancels in the PG, so it credits only high-entropy FUTURES, never inflating
#       sigma now). v28-34 forced these mutually exclusive; v35 decouples them so both run, as in SAC.
#       ALPHA-SPLIT (correctness): on an on-policy GAE, running both channels DOUBLE-COUNTS each
#       interior state's entropy H(s_k) -- once directly at step k, once as the successor entropy of
#       step k-1 in the soft GAE (SAC avoids this because future entropy is a LABEL inside Q, not
#       stacked additively on the same rollout). v35 splits alpha (soft_alpha_frac=0.5 each) so each
#       state's total entropy weight stays ~alpha, matching SAC's single budget.
#   (2) BINDING FLOOR TARGET: target_entropy_coef = +0.5 (per dim). Set ABOVE the measured free-collapse
#       level so the dual binds (Hopper holds H~1.5 vs collapse to -0.5; Walker holds H~3.0 vs 1.5).
#       This REPLACES the -1.0 SAC-parity target, which is unreachable here and un-binds the dual.
#       A live (non-bled) alpha is also what makes the soft-adv channel non-inert (v28-31's bled alpha
#       drove alpha*H -> 0 => soft_adv_std_ratio ~ 1.00).
#   (3) (kept from v34) rankgauss_signmag advantage, share_backbone=True, dual_freq=epoch (gentle
#       cadence), alpha_min=0.02 -- the v32 equilibrium machinery, now with a target it can equilibrate to.
# Hypothesis: the dual equilibrates (alpha dips then RISES to hold H at target), the soft-adv channel
# becomes live, the entropy floor sustains exploration, and Hopper/Walker2d climb past the ~660 ceiling.
# If HC regresses (its v32 win equilibrated near H=0), make the target env-aware.
#
# ---- inherited v34 notes ----
# PPO + IterThink v34 (SIGN-CORRECT advantage ONLY -- minimal change from v32). From v32.
#
# WHY v34. v33 stacked THREE changes (signmag transform + unshared trunk + batch-scope norm_adv)
# and HalfCheetah COLLAPSED 8800 -> 5042 (-43%), though the terminating envs improved (Hopper 905,
# Walker 1344). The regression couldn't be attributed because three levers moved at once. The HC
# win in v32 was built ON the shared ThinkTrunk + distributional-critic synergy, so unsharing the
# trunk (v33's biggest *Hopper* lever) is the prime suspect for breaking HC -- a genuine env-dependent
# conflict that should NOT be a global default.
# v34 isolates the ONE change that should help the terminating envs WITHOUT touching HC's winning
# ingredients: switch ONLY the advantage transform to "rankgauss_signmag" (sign-correct, zero-crossing
# at gae=0 not the batch median) and keep EVERYTHING ELSE at v32 (share_backbone=True, norm_adv_scope
# =minibatch). Rationale: on HalfCheetah the advantages are ~symmetric (median~=0) so signmag is
# numerically ~identical to rankgauss => HC should be PRESERVED; on the left-skewed terminating envs
# the sign correction fixes the ~10-20% wrong-signed gradients (the clip_z diagnostic, also shared
# backbone, already lifted Hopper->785 / Walker->1334; signmag is the sign-EXACT version).
# Hypothesis: HC recovers to ~8800 AND Hopper/Walker improve over v32 -- one global config, no
# per-env trunk sharing. If terminating envs still lag, unshared-trunk becomes an env-specific knob.
#
# ---- inherited v32 notes ----
# PPO + IterThink v32 (EQUILIBRIUM dual -- make alpha look/behave like SAC's). From v31.
#
# WHY v28-v31's alpha looked NOTHING like SAC's. SAC's alpha dips then EQUILIBRATES (jitters
# around a steady positive value) because its dual gradient (H - target) FLIPS SIGN: off-policy
# Q-maximization drives policy entropy to target_entropy within ~tens of thousands of steps, so
# H crosses target early and the dual oscillates around a fixed point. Our ports produced a
# MONOTONE BLEED to the floor instead, for a confirmed, measured reason:
#   * TIMESCALE MISMATCH. PPO contracts entropy ~100x SLOWER than SAC (squashed H descends
#     +4 -> -4 over ~4M steps). With SAC's target -6 (or even -2), H stays ABOVE target for the
#     ENTIRE alive-alpha window, so (H - target) > 0 always => the dual gradient is CONSTANT-SIGN.
#   * ADAM SIGN-STEPPING + 320x CADENCE. Adam on a scalar moves log_alpha by ~+-lr per step
#     regardless of |grad|. Stepping the dual per-minibatch = update_epochs*num_minibatches =
#     320 same-sign steps/rollout on REUSED data => Delta log_alpha ~ -0.32/rollout => alpha rails
#     to its floor by ~230k, MILLIONS of steps before H could ever cross target. (Measured: v29
#     H crosses -6 at 5.57M but alpha floored at 229k; v30/v31 never reach -2 before alpha dies.)
#   NOTE: the entropy ESTIMATE was faithful (summed over dims, tanh Jacobian, fresh rsample,
#   batch-meaned) -- scale was never the bug. And we deliberately keep alpha OUT of the C51 critic
#   target (SAC couples it there) because that resurrects the v26/v27 soft-critic blowup (support
#   overflow / symlog-Jensen, scored 187-288). The DIRECT actor bonus already gives the negative
#   feedback (alpha^ => H^ => grad v) needed for a fixed point, without touching the critic.
#
# v32's three fixes so the dual binds while alpha is alive, then equilibrates:
#   (1) REACHABLE TARGET: target_entropy = 0.0 -- where the policy's free descent sits at ~1-1.5M,
#       i.e. INSIDE the alive-alpha window. H crosses 0 there, the sign flips, alpha turns around.
#   (2) GENTLE CADENCE: step the dual ONCE PER EPOCH (10x/rollout) not per-minibatch (320x), on the
#       epoch-mean fresh entropy. Far fewer same-sign steps => alpha decays gently instead of railing.
#   (3) REAL FLOOR: alpha_min = 0.02 (not 1e-6) keeps temperature alive through the pre-bind transient
#       so it can RISE when the sign flips. alpha_init = 0.1 (SAC-ish), normalized frame, direct channel.
# Hypothesis: alpha now traces SAC's dip-then-equilibrate shape AND functionally holds entropy at 0,
# preventing the late over-collapse seen in v29 (return peaked ~5850 then fell to ~3170 as H->-6).
#
# ---- inherited v28 notes ----
# PPO + IterThink v28 (EQUIVALENT-ALPHA soft max-ent). From v24 (proven 4868 base).
#
# WHY v28. Porting SAC's entropy temperature to this PPO pipeline failed repeatedly for ONE
# root cause: a UNITS mismatch. SAC's alpha lives in RAW-reward units (HalfCheetah per-step
# reward ~O(1)). Here gym.NormalizeReward divides rewards by the running return-std (measured
# ~16 early, growing to ~100+ as returns grow), so the reward the loop sees is ~0.02/step.
# An alpha calibrated in normalized units is therefore ~return_std (16-100x) off from SAC's.
# Consequences we observed:
#   * v26's "alpha collapse" 0.097->0.003 was NOT a healthy equilibrium -- it was a ONE-SIDED
#     bleed toward 0 (squashed H stayed ABOVE target -6, so the dual could only decrease alpha).
#     It happened to pass through a tolerable EFFECTIVE scale (0.003*return_std ~ SAC's 0.1-0.3)
#     where it learned (3795), but it was switching max-ent OFF, not equilibrating.
#   * Holding alpha at 0.08 (v27) = 0.08*return_std ~ 1.3-4 raw = 6-20x SAC -> the entropy term
#     swamped the (tiny, normalized) reward and PINNED the policy at the SQUASHED-ENTROPY PEAK
#     (H_sq is NON-monotone in sigma: peaks +4 at mu~0,std~1 = max-variance CENTERED actions =
#     near random; H_sq only goes very negative once mu SATURATES, e.g. mu=2.5 -> H~-17).
# v28'S FIX (SUPERSEDED by v30 -- see top): inject entropy in RAW units by scaling the bonus by 1/return_std, so the
# learned alpha sits in SAC's native 0.1-0.3 range AND means the same thing. This keeps the
# EFFECTIVE entropy weight small/SAC-like, so the reward gradient can SATURATE the policy
# (instead of pinning it at the H_sq peak); a saturated policy's H drops BELOW -6, at which
# point the dual finally BINDS (two-sided) and alpha equilibrates instead of bleeding to 0.
# Also: REMOVE the explicit -alpha*entropy actor bonus (it is the peak-pinning term AND cannot
# be scale-matched against rankgauss, which strips advantage magnitude). Entropy now enters
# ONLY through the soft policy-advantage (future-entropy credit, rankgauss-reordered). The
# critic stays ENTROPY-FREE (v24's design) so the linear support fits -- no symlog, no Jensen.
#
# ---- inherited v24 notes ----
# SOFT MAX-ENTROPY add-on (--auto-entropy, gaussian only). Tests whether SAC's
# entropy temperature needs a SOFT VALUE to be coherent on-policy. We bake the
# closed-form policy entropy into the RETURN: r~_t = r_t + alpha*H(pi(.|s_t)),
# feeding r~ into BOTH the scalar GAE and the distributional lambda-return, so the
# categorical critic learns the soft value and the bootstrap stays consistent
# (SAC's soft-Q, done on-policy). alpha is auto-tuned to hold H at target. Rationale:
# an earlier attempt added the entropy temperature as an actor bonus on top of an
# entropy-BLIND advantage (incoherent: the bonus pinned H while the advantage pulled
# greedy) and regressed hard. Baking entropy into the return reconciles the two into
# one max-ent objective. Use --no-norm-adv --adv-transform v10 to isolate the soft
# advantage from rankgauss/adv-norm. Off (default) => byte-identical to the v24 base.
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
#   actor_dist="gaussian"  (the matched control = "direct log std" done right):
#       dreamer4's Gaussian readout. A state-dependent log-VARIANCE head (not a
#       flat Parameter, not log-std), SOFT-bounded by dreamer4's tanh-rescale
#       (not a hard clamp, so the gradient never dies at the bound):
#           lv  = rescale(tanh(raw_lv/(hi-lo)), (-1,1), (lo,hi))   # symmetric (lo,hi) => lv=0 at init
#           std = exp(0.5 * lv)
#       then the iterthink tanh-squash + SAC Jacobian on the sample (mean stays
#       raw), base-Normal entropy. (#1 soft-clamp + #2 log-var from the dreamer4
#       parity review; the standing entropy bonus #3 was judged not relevant.)
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
# All paths keep the mean-value GAE and the distributional λ-return value target
# (v10) UNCHANGED; only the policy advantage is reshaped. sigma(s) is the std of the
# OLD rollout Z(s_t), floored at `sigma_floor_bins` bins. Pair with target_kl for the
# 2x2 attribution (v10/tanh_gae/cdf_probit x KL-cap). Control: v17 / v10.
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

from cleanrl.shared.hl_gauss import HLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


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
    # bootstrap bonus b_t = α·H(s_{t+1}) (single-sample SQUASHED entropy -logπ == SAC's
    # next_state_log_pi, bounded since a∈[-1,1]). Rationale: α·H is KNOWN analytically, so
    # forcing the bounded categorical critic to learn the soft value both wastes capacity
    # and overflows the support (the softboot failure: edge_mass→0.9, expl_var→0). alpha is
    # auto-tuned by SAC's exact dual (target_entropy = -|A|; alpha_loss = -α(logπ + tgt)).
    # The explicit α·H actor bonus supplies the CURRENT-step entropy gradient (the soft
    # advantage's current-state entropy is action-independent => cancels in the PG term);
    # the soft advantage supplies FUTURE-entropy credit. Works WITH rankgauss: the soft
    # value reorders advantages and rankgauss preserves order/sign (magnitude is incidental).
    auto_entropy: bool = True       # v28: equivalent-alpha soft max-ent ON by default
    soft_alpha_frac: float = 0.5    # v35 legacy (UNUSED in v36). v36 removes the alpha split: the future
                                    #      channel is now properly BASELINED by V_ent (not double-counting the
                                    #      current step), so both channels get the FULL alpha, as in SAC.
    soft_adv: bool = True           # v36: TRUE => FUTURE entropy enters the policy advantage through a
                                    #      BASELINED entropy GAE: ent_adv = GAE(H(s') + gamma*V_ent(s') -
                                    #      V_ent(s)), policy_adv = reward_adv + alpha*ent_adv. V_ent (scalar
                                    #      MSE head) learns expected discounted future entropy so the advantage
                                    #      is CENTERED -- fixes v35's un-baselined horizon-accumulating pedestal
                                    #      that collapsed HalfCheetah. "max entropy IN GAE", done SAC-faithfully.
    direct_entropy: bool = True     # v35: TRUE => ALSO add the CURRENT-step -alpha*H actor-loss gradient
                                    #      (SAC's -alpha*log_pi term). This is the term that directly holds
                                    #      sigma up; the soft-adv channel alone cannot (current entropy is
                                    #      action-independent => cancels in the PG). NOT vanilla-PPO entropy:
                                    #      alpha is the auto-tuned dual temperature, not a fixed ent_coef.
    target_entropy: Optional[float] = None   # absolute override; if None, resolved PER-DIM from
                                    #      target_entropy_coef * action_dim.
    target_entropy_coef: float = 0.5  # v35: PER-DIM entropy FLOOR target. SAC's -1.0/dim is reachable from
                                    #      above only in SAC's fast off-policy regime; in slow on-policy PPO H
                                    #      stays far above it => the dual never binds and alpha bleeds. Set
                                    #      ABOVE the measured free-collapse level (Hopper H/dim -> -0.18,
                                    #      Walker -> +0.26) so H descends INTO the target and the dual binds as
                                    #      a FLOOR: H<target => alpha rises => actor re-expands entropy. Resolved
                                    #      target = +0.5*dim => +1.5 (Hopper), +3.0 (Walker/HC, 6-dim).
    alpha_lr: float = 1e-3
    alpha_init: float = 0.1         # v32: SAC-ish init; with once/epoch cadence + floor it dips then holds
    alpha_min: float = 0.02         # v32: REAL floor (not 1e-6). Keeps alpha alive through the pre-bind
                                    #      transient so it can RISE when the sign flips, instead of dying.
    dual_freq: str = "epoch"        # v32: step the dual once per EPOCH (10x/rollout) not per-minibatch
                                    #      (320x). 320 same-sign Adam steps/rollout on reused data is the
                                    #      accelerant that rails alpha to the floor before the target binds.
    return_std_floor: float = 1.0   # clamp the NormalizeReward divisor (raw |r|~1) early when var~0
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
    value_symlog: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "rankgauss_signmag"  # v34: ONLY change from v32. Sign-CORRECT (zero-crossing
                                    #      at gae=0, not the batch median) with rankgauss's outlier-immune
                                    #      global-rank magnitude => ~identical to rankgauss on HC's symmetric
                                    #      advantages (HC preserved), fixes wrong-signed grads on skewed envs.

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
    # (dreamer4 direct-log-std: state-dependent log-VARIANCE head, soft tanh-rescale bound).
    actor_dist: str = "gaussian"   # v28: squashed-Gaussian for SAC-style max-ent
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


def get_return_std(envs, floor=1.0):
    """Mean running return-std across the per-env gym.NormalizeReward wrappers.

    This is the divisor NormalizeReward applies to rewards (reward_seen = reward_raw /
    sqrt(return_rms.var)). To inject the entropy bonus in SAC's RAW-reward units, we
    divide alpha*H by this so it lands in the SAME normalized scale as the reward the
    loop sees. Floored (raw |r|~1) to avoid blow-up while return_rms.var is ~0 at the start.
    """
    stds = []
    for e in getattr(envs, "envs", []):
        w = e
        while w is not None and not isinstance(w, gym.wrappers.NormalizeReward):
            w = getattr(w, "env", None)
        if w is not None:
            stds.append(float(np.sqrt(w.return_rms.var + 1e-8)))
    return max(float(np.mean(stds)) if stds else floor, floor)


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
        # bias so the initial value distribution is sharp at 0 (not uniform),
        # preventing the distributional-bootstrap blowup that sank v9.
        self.critic_head = layer_init(nn.Linear(H, args.num_bins), std=0.1)
        with torch.no_grad():
            z = torch.linspace(args.v_min, args.v_max, args.num_bins)
            self.critic_head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)
        # v36: SCALAR future-entropy value head V_ent(s) = E[sum_k gamma^k H(s_{t+1+k})].
        # Trained by MSE on its own lambda-return; used ONLY to BASELINE the future-entropy
        # advantage (policy_adv = reward_adv + alpha*ent_adv). Scalar MSE => no categorical
        # support to overflow (the failure mode that sank putting entropy in the C51 critic).
        # Init at 0 (small weight, zero bias): future-entropy value starts neutral.
        self.ent_value_head = layer_init(nn.Linear(H, 1), std=0.1)
        with torch.no_grad():
            self.ent_value_head.bias.zero_()
        # v24: action distribution. Both parameterizations are dreamer4-faithful.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            # dreamer4 direct-log-std: mean head + state-dependent log-VARIANCE head.
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
        # Returns (value LOGITS (B, num_bins), ent_value (B,)). Caller converts logits via support.
        _, critic_feat = self._trunks(x)
        # ent head reads DETACHED features (auxiliary baseline; never shapes the trunk).
        return self.critic_head(critic_feat), self.ent_value_head(critic_feat.detach()).squeeze(-1)

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
        # ent head reads DETACHED features: an auxiliary baseline that never shapes the shared trunk.
        ent_value = self.ent_value_head(critic_feat.detach()).squeeze(-1)   # v36: scalar future-entropy value
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
        return action, z, log_prob, entropy, value_logits, ent_value

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
        # Params receiving the VALUE gradient (incl. the shared trunk). The v36 entropy-value head is
        # EXCLUDED -- it trains on DETACHED features via its own optimizer (ent_optimizer), so its
        # large-magnitude future-entropy target cannot corrupt the shared trunk or the reward critic.
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())


def categorical_project(probs, atoms, support, v_min, v_max, bin_width):
    """C51 projection: distribute `probs` (B, n) sitting at positions `atoms`
    (B, n) onto the fixed `support` (n,) via linear interpolation between
    neighbouring bins. Mass-preserving (atoms are pre-clamped to [v_min, v_max]).
    """
    n = support.shape[0]
    tz = atoms.clamp(v_min, v_max)
    b = (tz - v_min) / bin_width                       # (B, n) fractional bin pos
    lo = b.floor()
    hi = b.ceil()
    lo_idx = lo.clamp(0, n - 1).long()
    hi_idx = hi.clamp(0, n - 1).long()
    m = torch.zeros_like(probs)
    m.scatter_add_(1, lo_idx, probs * (hi - b))        # mass to lower bin
    m.scatter_add_(1, hi_idx, probs * (b - lo))        # mass to upper bin
    # When b is integer, lo == hi and both weights are 0; (1 - (hi - lo)) == 1
    # routes the full mass to that bin. Otherwise hi - lo == 1 → adds 0.
    m.scatter_add_(1, lo_idx, probs * (1.0 - (hi - lo)))
    return m


def distributional_lambda_returns(
    rewards, dones, next_done, value_probs, bootstrap_probs, support, v_min, v_max, bin_width, gamma, gae_lambda
):
    """Backward recursion for the distributional λ-return G^λ (probs per step).

        G^λ_t =_D r_t + γ·nonterm·[ (1-λ)·Z(s_{t+1}) + λ·G^λ_{t+1} ]

    Mean-matches the scalar GAE λ-return. Shapes: rewards/dones (T, B);
    value_probs (T, B, n); bootstrap_probs (B, n) = Z(s_T). Returns (T, B, n).
    Entropy/soft-value terms are NOT injected here — the critic regresses to the raw
    reward return; max-ent enters the policy advantage separately (see --auto-entropy).
    """
    T = rewards.shape[0]
    target = torch.zeros_like(value_probs)
    g_next = bootstrap_probs                            # G^λ_{T} ≡ bootstrap
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done               # (B,)
            z_next = bootstrap_probs                    # Z(s_T)
        else:
            nonterminal = 1.0 - dones[t + 1]
            z_next = value_probs[t + 1]                 # Z(s_{t+1})
        mix = (1.0 - gae_lambda) * z_next + gae_lambda * g_next          # (B, n)
        gn = (gamma * nonterminal).unsqueeze(-1)        # (B, 1)
        atoms = rewards[t].unsqueeze(-1) + gn * support  # (B, n) transformed atoms
        g_next = categorical_project(mix, atoms, support, v_min, v_max, bin_width)
        target[t] = g_next
    return target


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
    # Main optimizer trains everything EXCEPT the auxiliary entropy-value head (it gets its own
    # optimizer below so its large future-entropy targets can't dominate the shared gradient budget).
    _ent_ids = {id(p) for p in agent.ent_value_head.parameters()}
    optimizer = optim.Adam(
        [p for p in agent.parameters() if id(p) not in _ent_ids], lr=args.learning_rate, eps=1e-5
    )
    # v36: dedicated optimizer for the scalar future-entropy value head V_ent (detached features).
    ent_optimizer = optim.Adam(agent.ent_value_head.parameters(), lr=args.learning_rate, eps=1e-5)
    # v36: running normalization for the future-entropy return (~hundreds on non-terminating envs).
    # V_ent regresses a NORMALIZED target (unit scale => fits in a few hundred steps at the standard lr,
    # instead of ~1e6 steps to crawl Adam's bias to ~300); values are denormalized for the GAE. Count-based
    # (Welford) so the mean locks onto the ~300 scale after the FIRST rollout (an EMA would crawl for ~300).
    ent_ret_mean, ent_ret_var, ent_ret_count = 0.0, 1.0, 1e-4
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # Soft max-entropy temperature (gaussian only). log_alpha is learned so the soft-Q
    # bootstrap weight AND the actor entropy bonus self-tune to hold the SQUASHED entropy
    # at target_entropy via SAC's exact dual.
    auto_alpha = args.auto_entropy and args.actor_dist == "gaussian"
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        # PARITY: per-dim target * action_dim (SAC's -|A| == coef -1.0). An absolute override still
        # wins if explicitly set, but the default scales with action_dim so the setpoint is the SAME
        # PER DIM across HalfCheetah/Walker (6) and Hopper (3) — no more env-inconsistent absolute.
        target_entropy = args.target_entropy if args.target_entropy is not None else args.target_entropy_coef * float(act_dim)
        log_alpha = torch.full((1,), float(np.log(args.alpha_init)), requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        0.5,  # sigma_ratio unused (categorical Bellman target, no Gaussian projection)
        device,
        use_symlog=args.value_symlog,
    )
    support = hl_support.support                       # (num_bins,) linear support
    bin_width = hl_support.bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    ent_values = torch.zeros((args.num_steps, args.num_envs)).to(device)   # v36: V_ent(s_t) per step

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
                action, z, logprob, ent, value_logits, ent_value = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                values[step] = (p * support).sum(dim=-1)
                ent_values[step] = ent_value
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            # v36 BASELINED max-ent. FUTURE entropy enters the policy advantage through a SEPARATE,
            # PROPERLY BASELINED entropy GAE (not stacked onto the entropy-free reward bootstrap as in
            # v35, which left an un-centered horizon-accumulating pedestal that collapsed HC). The
            # reward critic stays entropy-free (fits its fixed support); a scalar V_ent head learns the
            # expected discounted future entropy so the entropy advantage is CENTERED. SAC's decomposition.
            if auto_alpha:
                # Sample a' ~ π(·|s_T) for the bootstrap entropy AND bootstrap V_ent (SAC's single-sample).
                _, _, boot_logprob, _, boot_logits, boot_ent_value = agent.get_action_and_value(next_obs)
                bootstrap_probs = torch.softmax(boot_logits, dim=-1)            # (B, n) = Z(s_T)
                alpha_r = log_alpha.exp().detach()
                # v30 NORMALIZED-FRAME ALPHA: do NOT divide by return_std. alpha lives in the SAME
                # normalized frame as the (renormalized) reward advantage, so alpha*ent_adv is on-scale.
                return_std = get_return_std(envs, args.return_std_floor)
                alpha_eff = alpha_r        # v36: NO split -- ent_adv is baselined (no double-count); full alpha.
            else:
                boot_logits, boot_ent_value = agent.get_value(next_obs)
                bootstrap_probs = torch.softmax(boot_logits, dim=-1)            # (B, n) = Z(s_T)
            next_value = (bootstrap_probs * support).sum(dim=-1).reshape(1, -1)
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support). UNTOUCHED.
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
            # v36 BASELINED ENTROPY GAE (POLICY ONLY). Entropy "reward" e_t = H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1})
            # (alpha-free; the next state's policy entropy, SAC's r + γ·entropy(s') convention). V_ent baselines
            # it on BOTH sides => ent_adv = realized future entropy - predicted => CENTERED (zero-mean, no
            # pedestal). ent_returns = ent_adv + V_ent is the MSE target for the V_ent head (built in the loop).
            # policy_adv = reward_adv + alpha*ent_adv: a zero-mean perturbation that REORDERS by future-entropy
            # WITHOUT the constant positive shift that flipped rankgauss's sign on HC.
            if auto_alpha and args.soft_adv:
                ent_rewards = torch.zeros_like(rewards)
                ent_rewards[:-1] = -logprobs[1:]            # H(s_{t+1}) for t < T-1
                ent_rewards[-1] = -boot_logprob             # H(s_T) bootstrap entropy
                # The head outputs a NORMALIZED value; denormalize to TRUE entropy-return scale
                # (~hundreds on non-terminating envs) before the GAE so the bootstrap is on-scale.
                ent_std = float(np.sqrt(ent_ret_var))
                ev_denorm = ent_values * ent_std + ent_ret_mean
                boot_ev_denorm = boot_ent_value * ent_std + ent_ret_mean
                ent_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_ent_v = boot_ev_denorm
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_ent_v = ev_denorm[t + 1]
                    ent_delta = ent_rewards[t] + args.gamma * next_ent_v * nextnonterminal - ev_denorm[t]
                    ent_adv[t] = lastgaelam = ent_delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                ent_returns = ent_adv + ev_denorm           # TRUE-scale future-entropy lambda-return
                policy_adv = advantages + alpha_eff * ent_adv
                # Update running normalization stats (count-based parallel/Welford) from the true-scale
                # return, then build the NORMALIZED MSE target the head regresses (unit scale => fast fit).
                bm = ent_returns.mean().item()
                bv = ent_returns.var(unbiased=False).item()
                bn = ent_returns.numel()
                delta = bm - ent_ret_mean
                tot = ent_ret_count + bn
                ent_ret_mean = ent_ret_mean + delta * bn / tot
                m_a = ent_ret_var * ent_ret_count
                m_b = bv * bn
                ent_ret_var = (m_a + m_b + delta * delta * ent_ret_count * bn / tot) / tot
                ent_ret_count = tot
                ent_target = (ent_returns - ent_ret_mean) / (float(np.sqrt(ent_ret_var)) + 1e-8)
            else:
                # soft_adv OFF: policy advantage is the pure reward advantage. Entropy enters only via the
                # direct current-step bonus below. Keep the head's target at its own output (no-op train).
                ent_adv = torch.zeros_like(rewards).to(device)
                ent_target = ent_values
                policy_adv = advantages
            # Critic target: RAW reward λ-return (entropy-free => no support overflow).
            target_probs = distributional_lambda_returns(
                rewards, dones, next_done, value_probs, bootstrap_probs,
                support, args.v_min, args.v_max, bin_width, args.gamma, args.gae_lambda,
            )
            # Per-state return std sigma(s_t) from the OLD rollout Z(s_t). Note
            # E[Z(s_t)] == values[t], so G_t - E[Z_t] == GAE advantage exactly.
            sigma = (value_probs * (support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            cdf_frac = ((returns.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
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
        b_ent_returns = ent_target.reshape(-1)           # v36: NORMALIZED MSE target for the V_ent head
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

        def _step_dual(entropy_mean):
            # SAC dual: alpha_loss = -alpha*(logπ + target) = -alpha*(-H + target). Gradient on
            # log_alpha is +alpha*(H - target): when H > target (entropy too high) alpha falls;
            # when H < target it rises. The fixed point (sign flip) is what makes alpha equilibrate.
            alpha_loss = (-log_alpha.exp() * (-entropy_mean + target_entropy))
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            with torch.no_grad():
                log_alpha.clamp_(min=float(np.log(args.alpha_min)))

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            epoch_entropy_sum = torch.zeros((), device=device)
            epoch_entropy_n = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits, ent_value_pred = agent.get_action_and_value(
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

                # Asymmetric "clip-higher" when clip_coef_high is set: looser upper bound
                # gives positive-advantage actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # v31 DIRECT auto-entropy bonus: the standard SAC/PPO actor entropy term, but with
                # alpha auto-tuned by the dual (below) toward target_entropy. entropy CARRIES GRAD
                # (the policy receives the entropy-maximizing gradient); alpha is DETACHED. This is
                # the potent channel: it perturbs the actor gradient directly, NOT laundered through
                # GAE -> rankgauss -> norm_adv (which crushed the indirect soft-adv channel to inert).
                # alpha is in the normalized frame (no /return_std) so alpha*H is on-scale with pg_loss.
                if auto_alpha and args.direct_entropy:
                    # v36: NO split. The direct (current-action) channel and the baselined future-entropy
                    # channel (ent_adv) act on DISJOINT timesteps, so each gets the FULL alpha (as in SAC).
                    direct_frac = 1.0
                    pg_loss = pg_loss - direct_frac * log_alpha.exp().detach() * entropy.mean()

                # Distributional value loss: cross-entropy to the (fixed)
                # distributional λ-return target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

                # v36: train the auxiliary future-entropy value head V_ent by MSE on its own
                # lambda-return (detached features + own optimizer => fully isolated from the reward
                # critic / policy / shared trunk). This baseline is what CENTERS the entropy advantage.
                ent_v_loss = 0.5 * ((ent_value_pred - b_ent_returns[mb_inds]) ** 2).mean()
                ent_optimizer.zero_grad()
                ent_v_loss.backward()
                nn.utils.clip_grad_norm_(agent.ent_value_head.parameters(), args.critic_grad_clip)
                ent_optimizer.step()

                entropy_loss = entropy.mean()  # logging only

                if auto_alpha:
                    # Accumulate the fresh current-policy squashed entropy (== SAC's -log_pi) so the
                    # dual can be stepped once per epoch on the epoch-mean (see end of minibatch loop).
                    epoch_entropy_sum += entropy.detach().mean()
                    epoch_entropy_n += 1
                    if args.dual_freq == "minibatch":
                        _step_dual(entropy.detach().mean())

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
                    pg_loss.backward()   # v28: entropy enters via the soft policy-adv ONLY
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss + v_loss * args.vf_coef   # v28: no actor entropy bonus
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if auto_alpha and args.dual_freq == "epoch" and epoch_entropy_n > 0:
                # ONE dual step per epoch on the epoch-mean fresh entropy: 10x/rollout, not 320x.
                # Far fewer same-sign Adam steps during the one-sided pre-bind transient => alpha
                # survives (decays gently from init toward the 0.02 floor) instead of railing dead,
                # and turns around to equilibrate once H crosses target.
                _step_dual(epoch_entropy_sum / epoch_entropy_n)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # v36: how well does V_ent fit the (normalized) future-entropy return? (baseline quality)
        ye_pred, ye_true = ent_values.reshape(-1).cpu().numpy(), b_ent_returns.cpu().numpy()
        var_ye = np.var(ye_true)
        ent_explained_var = np.nan if var_ye == 0 else 1 - np.var(ye_true - ye_pred) / var_ye

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)         # RAW units (SAC-comparable)
            writer.add_scalar("losses/alpha_eff", alpha_eff.item(), global_step)            # full alpha (no v35 split)
            writer.add_scalar("debug/return_std", return_std, global_step)                  # NormalizeReward divisor
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; the v36 BASELINED entropy advantage diagnostics:
            # ent_adv should be ~zero-mean (centered, no pedestal); ent EV shows V_ent's fit;
            # soft_adv_std_ratio = policy_adv.std / reward_adv.std (the entropy perturbation size,
            # should be a modest >1, NOT the runaway 2.5 that swamped HC in v35).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("losses/ent_value_loss", ent_v_loss.item(), global_step)
            writer.add_scalar("losses/ent_explained_variance", ent_explained_var, global_step)
            writer.add_scalar("debug/ent_adv_mean", ent_adv.mean().item(), global_step)
            writer.add_scalar("debug/ent_adv_std", ent_adv.std().item(), global_step)
            writer.add_scalar("debug/ent_returns_mean", ent_ret_mean, global_step)   # true-scale running mean
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

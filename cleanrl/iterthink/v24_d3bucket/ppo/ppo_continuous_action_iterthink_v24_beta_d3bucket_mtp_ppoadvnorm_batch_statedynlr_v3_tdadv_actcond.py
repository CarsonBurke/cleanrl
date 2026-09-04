# STATEDYNLR V3 TDADV+ACTCOND — v2_tdadv plus ACTION-CONDITIONED prediction.
# =====================================================================================
# V2 TDADV POSTMORTEM (run ..._1787362672): mechanism cleanly inert -- pred_corr ~
# 0.001 (noise) all run, alpha_mean = 1.0000, alpha_std <= 0.003. The zero-mean TD
# targets worked (no rail; benign no-op failure as designed), but the predictor found
# NOTHING PREDICTABLE in y_ij = Â_i * g_ij from its inputs. Diagnosis: g_ij is driven
# overwhelmingly by the SAMPLED ACTION, which f_phi never saw -- given only unit
# activations, advantage-gradient alignment is unlearnable noise.
#
# CHANGE: f_phi now also receives per-sample policy signals -- the squashed action
# vector, log-prob, and entropy of the CURRENT minibatch's forward. With (s, a)
# visible, predicting which units carry advantage-aligned gradient becomes the same
# kind of problem the critic already solves. Everything else identical to v2_tdadv:
# within-step SmoothL1 regression onto clamp(Â*g / RMS_j, +-3), alpha =
# exp(c*(2*sigmoid(logit)-1)) on the POLICY gradient only, regression from update 0,
# modulation after warmup.
#
# KILL SWITCH unchanged: debug/sdlr_pred_corr ~ 0 by 2M steps => even with actions the
# targets are noise => this supervision line is dead. Success looks like pred_corr
# meaningfully > 0 AND alpha_std > 0.05 sustained (only nonuniform alpha survives the
# binding actor clip).
#
# NOISE-FLOOR CAVEAT: v1 control / v2_tdadv / reference are ALL mechanically inert
# variants of the same recipe, yet finals span 8242-9704 (+-18%). Matched-step single-
# seed comparisons cannot resolve effects smaller than that; judge THIS variant first
# on whether the controller ENGAGES (pred_corr, alpha_std), only then on return.
# =====================================================================================
# STATEDYNLR V2 TDADV on the PPOADVNORM(batch) base — per-(sample, unit) plasticity
# PREDICTOR trained by REGRESSION onto TD-based advantage-gradient evidence.
# =====================================================================================
# WHY V2: postmortem of ppoadvnorm_batch_statedynlr_v1 (run ..._1787359157). v1's
# supervision e_j = gmean_j(t+1) * mm_j(t) is PER-UNIT ONLY -- identical for every
# sample in the batch, so f_phi(s) cannot reduce the meta-loss by varying with state;
# a per-unit CONSTANT is the optimal solution. The run converged exactly there:
# alpha_std = 0 with every site railed at exp(c)=e. And because actor_clip_frac = 1.0
# (the separate actor grad-norm clip bound every minibatch), a UNIFORM alpha is
# EXACTLY cancelled by the clip -- v1 was mechanically a no-op after ~0.5M steps.
# Its apparent +20% lead over reference at matched steps is an early-transient/drift
# artifact, NOT state-dependent plasticity. Two structural fixes follow:
#
# FIX 1 -- GRANULARITY: supervision must vary per sample. Target
#     y_ij = clamp( Â_i * g_ij / RMS_j(Â*g), +-3 )
# where i indexes minibatch samples, j trunk units, Â_i the batch-standardized GAE
# advantage (TD(lambda) signal already computed by the base) and g_ij the unit's RAW
# incoming policy gradient captured in the hook (pre-modulation, pre-clip).
# y_ij > 0: unit j's gradient pushes sample i along its advantage => amplifying j's
# gradient in this state reinforces useful signal; y_ij < 0: damped. Per-unit RMS
# normalization puts all units on one scale so the loss can't be gamed by magnitude.
#
# FIX 2 -- TD SUPERVISION + PREDICTION: replace the cross-step hypergradient unroll
# entirely. f_phi(s) PREDICTS y_ij from state features alone (activation squash,
# running sign-consistency m/sqrt(v), z-scored grad-RMS engagement, site embedding --
# advantage is deliberately NOT an input: the predictor must learn WHICH states give a
# unit advantage-aligned gradients). SmoothL1 regression of logits onto y; alpha_ij =
# exp(c*(2*sigmoid(logit)-1)) in [e^-c, e^c] multiplies the POLICY gradient elementwise
# during the actor backward only (critic backward and heads untouched). Sites:
# trunk.entry, ThinkBlock outputs, trunk.out_proj. Within-step pairing means NO
# cross-step graph and no epoch-reuse approximation.
#
# WHY THE TARGET CANNOT RAIL: Â is zero-mean by batch standardization, so E_i[y_ij] is
# bounded by the |corr(Â, g_j)| <= 1 and typically << 1; there is no persistent
# positive-mean pressure like v1's gradient-autocorrelation evidence. Benign failure
# mode is alpha -> 1 everywhere (no-op), not a rail.
#
# HYPOTHESIS: allocating policy-gradient magnitude per state toward units whose
# incoming gradients are advantage-aligned (a learned, per-state PG preconditioner)
# improves the PG estimator's SNR enough to beat the reference.
# PASS: >= ppoadvnorm_batch reference trajectory (8454 end; 5011/6672/7601 @2/4/6M).
# FAIL: >5% below reference at two consecutive checkpoints, or alpha_std < 0.05
#       sustained after warmup (predictor found no state-dependence => no-op).
# WATCH: debug/sdlr_pred_corr (can the predictor fit y AT ALL? ~0 means targets are
#        unpredictable noise => idea dead), debug/sdlr_target_mean/_std,
#        debug/sdlr_alpha_mean/_std (+ per-site), losses/sdlr_ctrl_loss,
#        debug/actor_clip_frac.
# =====================================================================================
# ENT-PPO v1 — entropy-regularized (soft-MDP) PPO with an EXACT analytic proximal term,
# ported from ent-ppo (Zykova-Myzina, Gritsaev, Tiapkin, Morozov; ICML 2026 SPIGM workshop,
# arXiv 2606.15793) onto the retstd_batch recipe (6364 @7.2M reference run).
# =====================================================================================
# WHAT ENT-PPO ACTUALLY IS (their gfnx/baselines/PPO_*.py): a GFlowNet is a soft-optimal
# policy of an entropy-regularized MDP, so their per-step reward is
#   r̃_t = log R + log P_B(s_t|s_{t+1}) − log P_F(a_t|s_t)
# (the `− log π_old` term IS the pathwise entropy bonus), GAE runs on r̃, the baseline
# regresses the SOFT return, and the surrogate is
#   min(ρÂ, clip(ρ)Â)  −  KL(π_new‖π_old)          [exact, all-action, coefficient 1]
# with NO entropy bonus and NO free KL coefficient. Two ingredients, one knob:
#
# (1) SOFT MDP: r̃_t = r_t + α·(−log π_old(a_t|s_t)). Because the bonus rides inside the
#     reward, GAE propagates it: an action is credited for the entropy of everything it
#     leads to (future-entropy credit), and the critic regresses the soft return. PPO's
#     `ent_coef·H(π(·|s_t))` is the myopic special case — current step only, α arbitrary.
# (2) EXACT PROXIMAL TERM: at state s the soft improvement objective is
#       E_{a~π}[Q̃] + α·H(π)
#     and since α·H(π) = −α·KL(π‖π_old) − α·E_{a~π}[log π_old(a)], while the second piece
#     is ALREADY carried by Ã_soft (the −α·log π_old(a_t) sitting in the augmented reward),
#       E_{a~π}[Q̃] + α·H(π) = E_old[ρ·Ã_soft] − α·KL(π_new‖π_old) + const.
#     So the analytic KL is not a heuristic trust region bolted onto PPO: paired with the
#     reward's pathwise surprisal it is an EXACT estimator of α·H(π_new), and its
#     coefficient is FORCED to be the same α that augments the reward. The two knobs PPO
#     treats as independent (ent_coef, KL/clip strength) are one quantity here.
#
# WHY IT SHOULD HELP HERE: the sampled-action surprisal −log π_old(a_t) is an exploration
# bonus that is (a) state-conditioned, (b) temporally propagated, and (c) exactly
# compensated by the KL term, so it perturbs the objective without biasing it toward
# uniform noise. Beta concentration collapse (log-density → +∞ ⇒ bonus → −∞) is penalized
# in RETURN space, i.e. the critic sees it coming k steps ahead instead of the actor eating
# a myopic bonus. Reference PPO here runs ent_coef=0 exactly because the myopic bonus was
# useless; this is the non-myopic version of the same idea.
#
# UNITS (the one thing a naive port gets wrong): the recipe divides the advantage by
# S = std(batch returns) before the PG. A proximal objective is only well-posed if the
# SAME divisor hits the KL term, so kl_coef = α/S (α·KL and Â/S in one geometry). All
# advantage divisors (retstd, percnorm, standardization) are tracked into `adv_div`. Two
# inherited flags put a factor OUTSIDE that product and are therefore asserted off while α>0:
# a non-identity `adv_transform` (non-homogeneous: rankgauss* throw the magnitude away) and
# `pos_neg_alpha != 0.5` (per-sample sign-dependent reweight). Both are already at the
# reference recipe's values, so the default run is unaffected.
#
# α is the sole new hyperparameter. α=0 recovers the retstd_batch reference exactly
# (same RNG stream: no extra sampling), so this file is its own ablation. Rewards here
# are ~6/step and Beta log-densities ~1 nat/dim over 6 dims, so α=0.1 puts the entropy
# term at ~10% of the return; α is in RAW reward units, like the paper's log-space α=1.
#
# DELIBERATELY NOT PORTED (with reasons, so the failure analysis is honest):
#   - their fresh-bootstrap TD(λ) value target (V_next recomputed with the LIVE critic each
#     value epoch): incompatible with this critic, which is a 6-horizon bucket-CE head whose
#     HL-Gauss labels are projected once per rollout on CPU.
#   - their full-batch policy epochs + separate value optimizer/minibatch splits: that is
#     GFlowNet-scale plumbing, orthogonal to the objective.
#   - their no-advantage-normalization: the reference recipe's retstd divisor is kept, since
#     the point is a one-change comparison against the 6364 run.
#   - target_kl=0.03 leash retained. REFUTED as "redundant": v1 exceeded 0.03 in 69.7% of
#     its 244 iterations at α=0.3 and 91.0% at α=0.1, so the leash was the load-bearing
#     trust region for the whole v1 experiment. It is also a CONFOUND in the α ladder: the
#     α=0.3 arm forfeited ~30% more of its epoch budget than α=0.1 did, so 4808-vs-3942
#     is not a clean isolation of the objective. (Both baselines trip more often still --
#     retstd 76.8%, ppoadvnorm_batch 84.4% -- so v1 was not unusually leashed.)
#     Original (wrong) note: it is now redundant with the α·KL penalty; if the
#     penalty is doing its job, `losses/approx_kl` stops tripping it and the policy gets
#     more of its 10-epoch budget. That is a mediated effect of the method, not a
#     confound — watch losses/approx_kl and losses/exact_kl together.
#
# PASS: ≥ retstd_batch reference at matched steps (which run peaked ~6364 @7.2M).
# FAIL: >5% below the reference at two consecutive checkpoints, or
#       debug/target_edge_mass rising off ~0 (soft return overflowing the ±20k support).
# WATCH: debug/ent_adv_share (entropy fraction of the policy signal; ≫1 ⇒ α too large),
#        debug/entropy_bonus_mean (= mean −log π_old, the realized per-step entropy),
#        losses/exact_kl vs losses/approx_kl, debug/kl_coef.
#
# --- Base (unchanged below): PPO + IterThink v24 Beta + v162 critic + Dreamer3-bucket
#     HL-Gauss MTP + MB PERCNORM v2 ---
# =====================================================================================
# v2 = v1 (d3percnorm) but the percentile advantage scaler is computed PER-MINIBATCH with NO EMA
# (ret_perc_scope="minibatch"), i.e. the per-mb analog of the old advnorm. Each minibatch divides its
# advantage by S = max(1, P95-P5) of THAT minibatch's RAW RETURNS (`mb_ret` -- the same return-spread
# statistic v1's EMA tracked, per the design intent), recomputed fresh every minibatch. Same divide-only,
# same return-spread source, same floor; ONLY the timing/memory differs: local & reactive instead of a
# slow global EMA. This isolates the EMA's contribution -- does dreamer3's slow global percentile scale
# beat a fresh per-minibatch one? Everything else == v1: no rankgauss (adv_transform="v10"), norm_adv off,
# raw reward, RAW-space Dreamer3 511-bucket symexp critic, PPO clip-higher (0.2/0.28) trust region.
# (--ret-perc-scope ema reproduces v1's global-EMA scaler exactly.) Watch charts/ret_perc_scale (now the
# last minibatch's S) -- expect it ~v1's magnitude (return spread is value-dominated, stable across mbs of
# 1024 samples) but noisier/reactive vs v1's EMA-smoothed value.
# =====================================================================================
#
# --- v1 method (changes vs the base, retained below) ---
# Variant of v162critic_dreamer3bucket_hlgauss_mtp_v1: ports DreamerV3's advantage-scale
# stack (as in dg_beta v15/v16) onto this MTP base. THREE changes vs the base:
#   (1) NO rankgauss: adv_transform="v10" (identity). The base's flagship rank-Gaussian
#       shaping is removed -- it already maps advantages to ~N(0,1), so a percentile
#       norm on top of it would be a constant-divide no-op. ("no advantage norm")
#   (2) NO norm_adv: per-minibatch standardization off. ("no advantage norm")
#   (3) DreamerV3 PERCENTILE NORM is the SOLE advantage scaler: policy_adv <- policy_adv /
#       max(1, EMA(P95)-EMA(P5)) over the raw GAE returns (EMA rate 0.01, divide-only).
# Reward stays RAW (base already defaults normalize_reward=clip_reward=False). The CRITIC
# stays in RAW space -- unchanged Dreamer3 511-bucket symexp HL-Gauss MTP head regressing
# raw returns (DreamerV3 valnorm=none; same arrangement as dg_beta v15/v16). Trust region
# is the base's PPO clip-higher (0.2/0.28), which bounds the step at any advantage scale.
#
# HYPOTHESIS: faithful DreamerV3 normalization (percentile spread + hard floor, raw-return
# symexp critic) on the strong MTP base is at least as stable as rankgauss while being a
# cleaner, more principled scaler. Falsifiable: if it underperforms the rankgauss base, the
# rank-only advantage (magnitude-discarding) was doing real work that a pure scale can't
# replace. Watch charts/ret_perc_scale (the EMA percentile spread S).
# =====================================================================================
#
# --- Base method (unchanged below) ---
# Hypothesis: keep iterthink_v24_beta_s1's PPO/Beta actor and ThinkTrunk, but
# replace the v24 distributional lambda-return critic with the v162 critic plus
# Dreamer3-style exponentially spaced value buckets over v162's raw range:
#   - value bucket centers are symexp(linspace(symlog(-20000), symlog(20000), 511))
#   - HL-Gauss target mass is integrated over the matching symlog-coordinate
#     bucket intervals instead of two-hot interpolation
#   - expected-scalar decode E[symexp(bin)] for values and bootstraps
#   - bias-free neutral critic logits instead of a peaked zero prior
#   - critic MTP head predicting returns[t + h] for h=0..5 with boundary masks
# This isolates whether Dreamer3's high-resolution near-zero / wide-tail bucket
# geometry improves the already-strong v162 critic port, without importing the
# v162 world model, CUDA graph path, or imagined updates.
#
# Base: ppo_continuous_action_iterthink_v24_dist.py / iterthink_v24_beta_s1.
# Critic donor: ppo_continuous_action_iterthink_v162_compiled_wmloss_cudagraph_edgeclamp_contdisc_k6.py.
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
# calibrated. In this variant the target is v162 scalar-return HL-Gauss MTP over
# Dreamer3-spaced buckets.
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
# v24's distributional lambda-return to v162 scalar-return HL-Gauss MTP over
# Dreamer3 buckets; only the policy advantage transforms are selected by
# `adv_transform`. sigma(s) is the std of the OLD rollout Z(s_t), floored at
# `sigma_floor_bins` bins.
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

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def value_support_bounds(args):
    """Return critic support endpoints in the coordinate system used by bins."""
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
    norm_adv: bool = True            # ppoadvnorm_batch base: standard PPO batch standardization.
    # --- Percentile advantage normalization (the sole advantage scaler) ---
    ret_percnorm: bool = False       # ppoadvnorm_batch base: OFF (norm_adv_scope="batch" is
    #                                  the sole advantage scaler). When on: S = max(floor, P95-P5) of returns.
    ret_perc_scope: str = "minibatch"  # "minibatch" = fresh per-mb P95-P5 of mb returns (NO EMA, like the
    #                                  old per-mb advnorm); "batch" = fresh WHOLE-ROLLOUT P95-P5, one S per
    #                                  update, NO EMA (the batch-vs-mb ablation); "ema" = v1's global EMA.
    ret_perc_rate: float = 0.01      # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0            # Ent-PPO: keep 0. The α·KL term IS the entropy bonus (see header).
    # --- ENT-PPO: the single knob. α augments the reward with the pathwise entropy
    # −log π_old(a_t|s_t) AND sets the coefficient of the exact KL(π_new‖π_old) proximal
    # term; the derivation forces them to be the same number. α=0 => the base recipe.
    ent_alpha: float = 0.0           # STATEDYNLR base: 0 (the retstd_batch reference; Ent-PPO OFF)
    # Ablation-only multiplier that BREAKS the derivation's tie between the reward bonus and
    # the proximal coefficient (1.0 = faithful Ent-PPO). Use it to test whether the exact
    # coefficient is actually the right one, not as a tuning knob.
    kl_coef_scale: float = 1.0
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

    # --- STATEDYNLR v2 TDADV: per-(sample, unit) plasticity predictor regressed onto
    #     TD(λ)-based advantage-gradient evidence (see header) ---
    statedynlr: bool = True          # modulate the POLICY gradient per (sample, trunk unit)
    sdlr_c: float = 1.0              # modulation bound: alpha in [exp(-c), exp(c)]
    sdlr_ema_beta: float = 0.98      # EMA decay for grad-sign consistency m and magnitude v
    sdlr_lr: float = 1e-3            # predictor learning rate (own Adam, regression only)
    sdlr_warmup: int = 100           # pg-updates before modulation engages (regression runs from update 0)

    # Dreamer3-style bucket critic: centers are symexp(linspace(v_min, v_max, num_bins)).
    # Defaults match v162's ±20k raw support, expressed in symlog coordinates.
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    # Keep observation normalization, but disable return/reward normalization for
    # this ablation. NormalizeReward tracks discounted returns internally and
    # divides rewards by their running std, so it must remain off here.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "v10"       # d3percnorm: identity -- NO rankgauss. DreamerV3 percentile
    #                                  norm (below) is the sole advantage scaler ("no advantage norm").

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "batch"    # ppoadvnorm_batch base: standardize ONCE over the whole rollout

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
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
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


class SDLRController(nn.Module):
    """Per-(sample, unit) plasticity predictor trained by REGRESSION onto TD-based
    evidence (v2 TDADV). NO cross-step unroll, NO meta-gradient.

    Each hooked trunk site exposes H units. f_phi reads per-(sample, unit) state
    features -- activation, running sign-consistency m/sqrt(v), z-scored grad-RMS
    engagement, learned site embedding -- and emits logits that BOTH
        alpha_ij(s) = exp(c*(2*sigmoid(logit_ij)-1))   in [e^-c, e^c],
    (applied elementwise to the POLICY gradient in the actor backward only) and serve
    as the PREDICTION regressed onto the realized within-step evidence
        y_ij = clamp( adv_i * g_ij / RMS_j(adv*g), +-3 )
    where adv_i is the minibatch's batch-standardized GAE advantage (the base's
    TD(lambda) signal) and g_ij the unit's RAW incoming policy gradient captured in
    the same backward (pre-modulation, pre-clip). y_ij > 0: unit j pushes sample i
    along its advantage => amplify; y_ij < 0: damp. Per-unit RMS normalization
    equalizes units; the zero-mean advantage bounds E[y], so unlike v1's
    gradient-autocorrelation evidence the targets cannot rail the predictor -- benign
    failure is alpha -> 1 everywhere (no-op).

    Stats are captured in the same backward hook BEFORE the site's own scaling;
    upstream modulation leaks into deeper sites' grads within one backward,
    bounded by alpha in [e^-c, e^c].
    """

    def __init__(self, site_modules, H, act_dim, c=1.0, ema_beta=0.98, lr=1e-3, warmup=100):
        super().__init__()
        self.site_names = list(site_modules.keys())
        self.H = H
        self.act_dim = act_dim
        self.c = c
        self.beta = ema_beta
        self.warmup = warmup
        self.updates = 0
        emb_dim = 8
        self.emb = nn.Embedding(len(self.site_names), emb_dim)
        # inputs: squashed activation | sign consistency | log magnitude | site emb |
        #         squashed action | squashed logp | squashed entropy
        self.net = nn.Sequential(nn.Linear(3 + emb_dim + act_dim + 2, 32), nn.ReLU(), nn.Linear(32, 1))
        for name in self.site_names:
            self.register_buffer(f"m_{name}", torch.zeros(H))   # signed batch-mean grad EMA (feature only)
            self.register_buffer(f"v_{name}", torch.zeros(H))   # batch mean-square grad EMA
            self.register_buffer(f"mu_{name}", torch.zeros(H))  # EMA of log rms
            self.register_buffer(f"m2_{name}", torch.zeros(H))  # EMA of log rms^2 (for sigma)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        # runtime state (not parameters)
        self.pg_pass = False       # hooks act ONLY during the policy backward
        self._acts = {}            # site -> detached activations from the update forward
        self._grads = {}           # site -> detached raw grad (B, H) from THIS policy backward
        self._alphas = {}          # site -> detached (B,H) modulation
        self._logits = []          # [(site, logits with graph)] for THIS minibatch
        self._last = {}            # site -> alphas of the LAST ingest (for diagnostics)
        self._last_y = None        # last normalized target, flattened across sites
        self._last_corr = None     # last mean corr(pred, y) across sites
        self._handles = []
        # NOTE: register_full_backward_hook cannot MODIFY grads on torch 2.12
        # ("hook changed the size of value" even for identity returns), so the
        # modulation attaches a TENSOR hook to each site's output tensor during
        # the update forward instead -- same semantics, supported API.
        for name, mod in site_modules.items():
            self._handles.append(mod.register_forward_hook(self._make_fwd(name)))

    def _make_fwd(self, name):
        def hook(_mod, _inp, out):
            if not torch.is_grad_enabled():
                return
            self._acts[name] = out.detach()
            if out.requires_grad:
                out.register_hook(self._make_bwd(name))
        return hook

    def _make_bwd(self, name):
        def hook(g):
            if not self.pg_pass:
                return None
            self._grads[name] = g.detach()
            a = self._alphas.get(name)
            return None if a is None else g * a
        return hook

    def prepare(self, action, logp, ent):
        """Run f_phi on this minibatch's stored activations plus per-sample policy
        signals (action, logp, entropy); stash detached alphas for the backward hooks
        and keep the logits' graph for the regression."""
        self._alphas.clear()
        self._logits.clear()
        modulate = self.updates >= self.warmup
        embs = self.emb(torch.arange(len(self.site_names), device=self.emb.weight.device))
        feat_a = action / (1.0 + action.abs())               # bounded squash (B, act_dim)
        feat_lp = (logp.clamp(-20.0, 0.0) / 10.0).view(-1, 1, 1)
        feat_ent = (ent / (1.0 + ent)).view(-1, 1, 1)
        for idx, name in enumerate(self.site_names):
            act = self._acts.get(name)
            if act is None:
                continue
            B = act.shape[0]
            x = act / (act.pow(2).mean().sqrt() + 1e-6)
            feat_act = x / (1.0 + x.abs())                       # bounded squash of the state
            sd = getattr(self, f"v_{name}").sqrt()
            cons = (getattr(self, f"m_{name}") / (sd + 1e-8)).clamp(-1, 1)   # (H,) feature only
            lr = (sd + 1e-12).log()                              # current log grad-RMS
            mu = getattr(self, f"mu_{name}")
            sig = (getattr(self, f"m2_{name}") - mu.pow(2)).clamp_min(1e-8).sqrt().clamp_min(0.25)
            mag = ((lr - mu) / sig).clamp(-4.0, 4.0)             # z-scored engagement (H,)
            rows = torch.cat(
                [
                    feat_act.unsqueeze(-1),
                    cons.expand(B, -1).unsqueeze(-1),
                    mag.expand(B, -1).unsqueeze(-1),
                    embs[idx].expand(B, self.H, -1),
                    feat_a.unsqueeze(1).expand(B, self.H, -1),
                    feat_lp.expand(B, self.H, -1),
                    feat_ent.expand(B, self.H, -1),
                ],
                dim=-1,
            )                                                    # (B, H, 3+emb+act_dim+2)
            logits = self.net(rows).squeeze(-1)                  # (B, H)
            self._logits.append((name, logits))
            if modulate:
                alpha = torch.exp(self.c * (2.0 * torch.sigmoid(logits) - 1.0))
            else:
                alpha = logits.new_ones(())
            self._alphas[name] = alpha.detach().expand_as(act)

    def ingest(self, adv):
        """After optimizer.step(): (1) fold this backward's grad stats into the feature
        EMAs; (2) build per-(sample, unit) TD-based targets y from `adv` (detached,
        batch-standardized) and this backward's RAW grads; (3) regress this
        minibatch's logits onto y with SmoothL1. Regression runs from update 0 (the
        predictor gets a head start before modulation engages at warmup).
        Returns the regression loss (or None)."""
        reg_loss = None
        terms, ys, corrs = [], [], []
        for name, logits in self._logits:
            gd = self._grads.get(name)
            if gd is None:
                continue
            gmean = gd.mean(dim=0)
            m = getattr(self, f"m_{name}")
            setattr(self, f"m_{name}", self.beta * m + (1.0 - self.beta) * gmean)
            v_old = getattr(self, f"v_{name}")
            v_new = self.beta * v_old + (1.0 - self.beta) * gd.pow(2).mean(dim=0)
            setattr(self, f"v_{name}", v_new)
            lr_new = (v_new + 1e-12).sqrt().log()
            mu = getattr(self, f"mu_{name}")
            m2 = getattr(self, f"m2_{name}")
            setattr(self, f"mu_{name}", self.beta * mu + (1.0 - self.beta) * lr_new)
            setattr(self, f"m2_{name}", self.beta * m2 + (1.0 - self.beta) * lr_new.pow(2))
            e = adv.unsqueeze(1) * gd                             # (B, H) TD-based evidence
            e = e / (e.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-8)  # per-unit RMS over batch
            y = e.clamp(-3.0, 3.0)
            ys.append(y.reshape(-1))
            terms.append(F.smooth_l1_loss(logits, y))
            with torch.no_grad():
                pv = logits - logits.mean()
                yv = y - y.mean()
                corrs.append((pv * yv).sum() / (pv.norm() * yv.norm() + 1e-8))
        if terms:
            reg_loss = torch.stack(terms).mean()
            self.opt.zero_grad(set_to_none=True)
            reg_loss.backward()
            self.opt.step()
            self._last_y = torch.cat(ys)
            self._last_corr = torch.stack(corrs).mean().item()
        self.updates += 1
        self._last = dict(self._alphas)
        self._acts.clear()
        self._grads.clear()
        self._alphas.clear()
        self._logits = []
        return reg_loss

    def diagnostics(self):
        out = {}
        if self._last_y is not None:
            out["debug/sdlr_target_mean"] = self._last_y.mean().item()
            out["debug/sdlr_target_std"] = self._last_y.std().item()
        if self._last_corr is not None:
            out["debug/sdlr_pred_corr"] = self._last_corr
        alphas = [a for a in self._last.values() if a.numel() > 1]
        if not alphas:
            return out
        all_a = torch.cat([a.reshape(-1) for a in alphas])
        out["debug/sdlr_alpha_mean"] = all_a.mean().item()
        out["debug/sdlr_alpha_std"] = all_a.std().item()
        for name in self.site_names:
            a = self._last.get(name)
            if a is not None and a.numel() > 1:
                out[f"debug/sdlr_alpha_site_{name}"] = a.mean().item()
        return out

    def close(self):
        for h in self._handles:
            h.remove()


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
        # v162 critic: bias-free neutral MTP head. With symmetric symlog support,
        # zero logits decode to a zero raw value without a hidden prior.
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
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
        # Returns value LOGITS (B, mtp, num_bins); horizon 0 is V(s_t).
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

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
        value_logits = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)
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
        return action, z, log_prob, entropy, value_logits, dist

    # ENT-PPO: the analytic KL(π_new‖π_old) needs the OLD distribution, so the rollout
    # stores its two head parameters and the update rebuilds it (parameters, not samples:
    # the KL is all-action/exact, not an importance-weighted single-sample estimate).
    def dist_params(self, dist):
        if self.actor_dist == "gaussian":
            return dist.loc, dist.scale
        return dist.concentration1, dist.concentration0

    def rebuild_dist(self, p1, p2):
        if self.actor_dist == "gaussian":
            # KL is invariant under the tanh bijection, so base-Normal KL == squashed KL.
            return Normal(p1, p2)
        return Beta(p1, p2)

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
    assert args.norm_adv_scope in ("batch", "minibatch", "batch_retstd", "minibatch_retstd")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope in ("batch", "batch_retstd")), \
        "norm_adv_scope=batch/batch_retstd requires adv_transform_scope=batch"
    # Both inject entropy into the advantage; together they double-count it (and auto_entropy
    # keeps the critic entropy-free, which contradicts Ent-PPO's soft value function).
    assert not (args.ent_alpha != 0.0 and args.auto_entropy), \
        "ent_alpha (soft-MDP reward, Ent-PPO) and --auto-entropy (soft-advantage bootstrap) are mutually exclusive"
    assert not (args.ent_alpha != 0.0 and args.ent_coef != 0.0), \
        "ent_coef must stay 0 under Ent-PPO: the alpha*KL proximal term already is the entropy bonus"
    # The proximal term is only well-posed if `adv_div` really is the full scale applied to the
    # advantage. Two supported flags put a factor OUTSIDE that product, so refuse them here
    # rather than silently mis-scaling alpha: (a) any non-identity adv_transform is a
    # non-homogeneous map (tanh saturates; rankgauss* discard the GAE magnitude outright), so
    # after it the advantage is not in return units at all; (b) pos_neg_alpha reweights each
    # sample by a SIGN-dependent factor, which no scalar divisor can represent.
    assert not (args.ent_alpha != 0.0 and args.adv_transform != "v10"), \
        "Ent-PPO needs adv_transform=v10: a non-identity shaping leaves the advantage in unknown units, so alpha/adv_div is no longer the derivation's coefficient"
    assert not (args.ent_alpha != 0.0 and args.pos_neg_alpha != 0.5), \
        "Ent-PPO needs pos_neg_alpha=0.5: its per-sample sign-dependent reweight cannot be folded into adv_div"
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

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # STATEDYNLR: per-perceptron state-dependent LR on the shared trunk's units.
    # Requires the dual-backward path so the hooks gate the POLICY backward only
    # (the combined-loss backward would modulate critic gradients too).
    sdlr = None
    if args.statedynlr:
        assert args.share_backbone and args.separate_grad_clip, \
            "statedynlr modulates the shared trunk under separate_grad_clip only"
        site_modules = {"entry": agent.trunk.entry}
        for k, blk in enumerate(agent.trunk.blocks):
            site_modules[f"block{k}"] = blk
        site_modules["out_proj"] = agent.trunk.out_proj
        # Construct on CPU with the torch RNG state saved/restored: the
        # controller's random init must NOT shift the seeded rollout stream,
        # so a --no-statedynlr arm matches the reference bit-for-bit at the
        # same seed.
        _rng_state = torch.get_rng_state()
        sdlr = SDLRController(
            site_modules,
            args.hidden,
            act_dim=int(np.prod(envs.single_action_space.shape)),
            c=args.sdlr_c,
            ema_beta=args.sdlr_ema_beta,
            lr=args.sdlr_lr,
            warmup=args.sdlr_warmup,
        )
        torch.set_rng_state(_rng_state)
        sdlr = sdlr.to(device)

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
    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
    )
    hl_support_cpu = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )
    support = hl_support.support                       # Dreamer3 raw bucket centers
    bin_width = hl_support.bin_width
    raw_support = support

    def value_logits_to_scalar(logits):
        return hl_support.to_scalar(logits)

    sigma_floor = args.sigma_floor_bins * bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # Ent-PPO: behavior-policy head parameters (Beta: concentration1/concentration0;
    # Gaussian: loc/scale) for the exact proximal KL.
    dist_p1 = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    dist_p2 = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

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
                action, z, logprob, ent, value_logits, dist = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            p1, p2 = agent.dist_params(dist)
            dist_p1[step] = p1
            dist_p2[step] = p2

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
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
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
                _, _, boot_logprob, _, _, _ = agent.get_action_and_value(next_obses[-1])
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                next_value_bonus = None
            # ENT-PPO SOFT MDP: r̃_t = r_t + α·(−log π_old(a_t|s_t)). The pathwise entropy
            # bonus lives INSIDE the reward, so (a) GAE propagates it => an action is credited
            # for the entropy of the states it leads to, and (b) the critic regresses the SOFT
            # return, i.e. it learns the soft value Ṽ the derivation assumes.
            # UNITS: logprobs is the Beta density in the NATIVE z-space (the unit cube), i.e.
            # log_det_fn=0 for the affine z→action map, so this is action-space surprisal plus
            # MINUS the constant |A|·log 2 (a = low + (high−low)·z with high−low = 2, so
            # p_a(a) = p_z(z)/2^|A|). A constant per-step reward shifts every return by
            # c/(1−γ) and cancels exactly in delta_t once the critic fits, so the choice of
            # convention has NO gradient effect; only debug/entropy_bonus_mean reads |A|·log 2
            # LOWER than an action-space entropy would (on HalfCheetah 6·ln2 = 4.16 nats:
            # a logged −7.24 is an action-space differential entropy of −3.08). The KL term is reparameterization-
            # invariant, so the α·H(π_new) identity holds in whichever convention is used.
            # SUPPORT: the bonus is α·(a few nats) against ~6 reward/step, and the ±20k symlog
            # support has ~22x headroom over HalfCheetah's discounted return (MEASURED:
            # debug/returns_absmax peaks at 918 with the bonus vs 943 without, i.e. the bonus
            # barely moves return MAGNITUDE -- its effect is on ordering/variance), so the target
            # still fits (unlike the auto_entropy softboot path, whose auto-tuned alpha
            # overflowed it -- watch debug/target_edge_mass).
            soft_rewards = rewards + args.ent_alpha * (-logprobs)
            # SOFT GAE: critic-consistent soft advantage + soft return.
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = soft_rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
            # Effect-size probe, exact (GAE is linear in the reward sequence at fixed V):
            # A_soft = A_reward + E with E = GAE_λ(α·(−log π_old)) and no value terms.
            # std(E)/std(A_soft) is the entropy share of the policy signal; ≫1 => alpha too big.
            ent_adv = torch.zeros_like(rewards)
            lastentlam = 0
            for t in reversed(range(args.num_steps)):
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                ent_adv[t] = lastentlam = (
                    args.ent_alpha * (-logprobs[t])
                    + args.gamma * args.gae_lambda * lambda_nonterminal * lastentlam
                )
            ent_adv_share = (ent_adv.std() / (advantages.std() + 1e-8)).item()
            ent_ret_shift = ent_adv.mean().item()
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
            # Batch-level percentile advantage normalization (scopes "ema" and "batch"). Both compute the
            # whole-rollout P5/P95 once and scale policy_adv by one S; "ema" smooths the percentiles with a
            # global EMA across iterations (v1), "batch" uses the FRESH per-rollout spread (no EMA -- the
            # batch-vs-mb ablation). scope=="minibatch" SKIPS this and scales fresh per-mb in the update loop,
            # leaving policy_adv RAW here. Divide-only; critic target `returns` stays RAW (valnorm=none).
            if args.ret_percnorm and args.ret_perc_scope in ("ema", "batch"):
                flat_ret = returns.reshape(-1)
                qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=device)
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                if args.ret_perc_scope == "ema":
                    if not ema_perc_inited:
                        ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
                    else:
                        r = args.ret_perc_rate
                        ema_ret_lo += r * (lo - ema_ret_lo)
                        ema_ret_hi += r * (hi - ema_ret_hi)
                    spread = ema_ret_hi - ema_ret_lo
                else:  # "batch": fresh whole-rollout percentile spread, no EMA
                    spread = hi - lo
                ret_perc_scale = max(args.ret_perc_floor, spread)
                policy_adv = policy_adv / ret_perc_scale
            # v162 critic target: scalar-return HL-Gauss MTP. Horizon 0 regresses
            # returns[t]; horizon h regresses returns[t+h] from the same features.
            # A future target is valid only when no reset boundary lies between
            # the source state and target state, and when it stays inside rollout.
            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros(
                (*returns.shape, mtp), dtype=torch.bool, device=returns.device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=returns.device
                )
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                return_mtp[:valid_len, :, h] = returns[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            # The full (T,B,MTP,bins) target is large and fixed. Keep it on CPU
            # and move only minibatch labels to CUDA during the value loss.
            target_probs = hl_support_cpu.project(return_mtp.detach().cpu())
            # Per-state return std probe from the OLD rollout Z(s_t), decoded to
            # raw return units. The default rankgauss path does not consume this.
            sigma = (value_probs * (raw_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(sigma_floor)
            # CDF-rank u in Dreamer3 bucket order; intervals are uniform in symlog
            # coordinate even though raw bucket centers are exponentially spaced.
            cdf_frac = hl_support.cdf_fraction(returns)
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
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()
        b_u = u.reshape(-1)
        b_dist_p1 = dist_p1.reshape((-1,) + envs.single_action_space.shape)
        b_dist_p2 = dist_p2.reshape((-1,) + envs.single_action_space.shape)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        # Ent-PPO scale bookkeeping: the proximal objective is (1/adv_div)·[Â − α·KL], so every
        # divisor applied to the advantage MUST also divide the KL penalty. batch_pre_div covers
        # the whole-rollout percentile scaler already applied to policy_adv above.
        batch_pre_div = (
            ret_perc_scale if (args.ret_percnorm and args.ret_perc_scope in ("ema", "batch")) else 1.0
        )
        batch_norm_div = 1.0
        if args.norm_adv and args.norm_adv_scope == "batch":
            batch_norm_div = b_policy_adv.std() + 1e-8
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / batch_norm_div
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            batch_norm_div = b_returns.std().clamp(min=args.ret_perc_floor)
            b_policy_adv_normed = b_policy_adv / batch_norm_div
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        sdlr_ctrl_loss = None
        actor_clip_hits, n_mbs = 0.0, 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits, new_dist = agent.get_action_and_value(
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
                adv_div = batch_pre_div          # running product of every divisor hitting the advantage
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                        adv_div = adv_div * batch_norm_div
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_div = b_returns[mb_inds].std().clamp(min=args.ret_perc_floor)
                        mb_advantages = mb_advantages / mb_div
                        adv_div = adv_div * mb_div
                    else:
                        mb_div = mb_advantages.std() + 1e-8
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_div
                        adv_div = adv_div * mb_div
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds]
                    qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device)
                    lo, hi = torch.quantile(mb_ret, qs)
                    mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                    mb_advantages = mb_advantages / mb_perc_scale
                    adv_div = adv_div * mb_perc_scale
                    ret_perc_scale = mb_perc_scale.item()

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

                # ENT-PPO EXACT PROXIMAL TERM (their policy_loss_fn: ppo_clip − KL, coefficient 1).
                #   E_{a~π}[Q̃] + α·H(π) = E_old[ρ·Ã_soft] − α·KL(π_new‖π_old) + const,
                # so this is not a bolted-on trust region: with the reward's −α·log π_old it is an
                # EXACT estimator of α·H(π_new), and α is the same number in both places. All-action
                # and analytic (no importance weight, no clipping), unlike PPO's ratio heuristic;
                # for Beta it is closed form in the concentrations, and it is invariant to the
                # affine z→action rescale (and to tanh in the Gaussian path). Divided by adv_div so
                # the surrogate and the penalty live in the same units.
                exact_kl = torch.distributions.kl_divergence(
                    new_dist, agent.rebuild_dist(b_dist_p1[mb_inds], b_dist_p2[mb_inds])
                ).sum(1)
                kl_coef = args.ent_alpha * args.kl_coef_scale / adv_div
                kl_penalty = kl_coef * exact_kl.mean()

                # v162 HL-Gauss MTP value loss: per-horizon CE to scalar-return
                # targets, summed across valid horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds].to(device=value_logits.device, non_blocking=True)
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(device=value_logits.device, dtype=value_ce.dtype, non_blocking=True)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

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
                    if sdlr is not None:
                        # Predict per-(sample, unit) plasticity from this minibatch's
                        # stored activations, then let the hooks scale the POLICY
                        # gradient elementwise during the backward below.
                        sdlr.prepare(b_actions[mb_inds], newlogprob.detach(), entropy.detach())
                        sdlr.pg_pass = True
                    (pg_loss + kl_penalty - ent_coef_eff * entropy_loss).backward()
                    if sdlr is not None:
                        sdlr.pg_pass = False
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # probe: is the actor budget actually binding? (a UNIFORM
                    # sdlr alpha would be cancelled exactly by this clip)
                    actor_clip_hits += float(actor_gn > args.actor_grad_clip)
                    n_mbs += 1
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                    if sdlr is not None:
                        # Fold grad stats into the EMAs and regress the predictions
                        # onto this minibatch's TD-based advantage-gradient targets.
                        sdlr_ctrl_loss = sdlr.ingest(mb_advantages.detach())
                else:
                    loss = pg_loss + kl_penalty - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_per_h = b_target_probs[..., 0] + b_target_probs[..., -1]
        edge_mask_f = b_target_mask.to(dtype=edge_per_h.dtype)
        edge_mass = ((edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)).item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        if sdlr is not None:
            for k, v in sdlr.diagnostics().items():
                writer.add_scalar(k, v, global_step)
            # NaN (not 0.0) while warming up: indistinguishable-from-converged
            # is exactly the misread this must prevent.
            writer.add_scalar(
                "losses/sdlr_ctrl_loss",
                sdlr_ctrl_loss.item() if sdlr_ctrl_loss is not None else float("nan"),
                global_step,
            )
            writer.add_scalar("debug/sdlr_updates", sdlr.updates, global_step)
        writer.add_scalar(
            "debug/actor_clip_frac",
            actor_clip_hits / max(1, n_mbs),
            global_step,
        )
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
            writer.add_scalar("debug/soft_adv_std_ratio", (policy_adv.std() / (advantages.std() + 1e-8)).item(), global_step)
        # Ent-PPO: exact all-action KL(π_new‖π_old) at the last minibatch (the penalized
        # quantity) next to the sampled-ratio approx_kl that drives the early-stop leash.
        writer.add_scalar("losses/exact_kl", exact_kl.mean().item(), global_step)
        writer.add_scalar("debug/kl_coef", float(kl_coef), global_step)
        writer.add_scalar("debug/kl_penalty", kl_penalty.detach().item(), global_step)
        writer.add_scalar("debug/entropy_bonus_mean", (-logprobs).mean().item(), global_step)
        writer.add_scalar("debug/ent_adv_share", ent_adv_share, global_step)
        writer.add_scalar("debug/soft_return_shift", ent_ret_shift, global_step)
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

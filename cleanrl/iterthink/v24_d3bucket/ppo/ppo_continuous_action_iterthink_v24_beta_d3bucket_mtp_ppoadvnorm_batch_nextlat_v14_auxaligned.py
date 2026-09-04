# PPO + IterThink v24 Beta + d3bucket MTP + PPO-ADVNORM(batch) + NEXTLAT v14_auxaligned.
# =====================================================================================
# AUX REALIGNED: v13's 3e-4 aux group smoothed returns (+-34 CI) but starved
# the critic (EV 0 -> -1.0 progressive collapse; level 2265 -> 1757). The slow
# dyn broke teacher-student tracking and fed incoherent shaping grads into the
# critic trunk. So aux modules return to their core LRs -- predictor at actor
# 3e-3, dyn/readout at critic 1e-3 -- i.e. stepping-identical to v10. Deltas
# from v13: optimizer groups reverted to v10's 2-group form; nextlat_lr arg
# removed. Fresh restart at 64x39, 1 epoch, unshared backbone.
#   - ACTOR LR 3e-3 -> 1e-2. KL has ~1e7x headroom to the 0.03 leash, so the
#     policy can afford much bigger steps; 3x (not 10x) because the 2496 batch
#     is noisy. Watch charts/approx_kl: healthy is 1e-4..1e-2, still far below
#     the leash (which cannot bind at 1 epoch -- the KL number itself is the
#     guardrail here).
#   - AUX predictors split into their own group at nextlat_lr=3e-4 (10x below
#     actor, 3x below critic). Trunks/heads keep PG+value LRs, so behavior is
#     driven by the policy/value signal, not aux noise. Smoothing the trunk
#     updates should de-spike the wavy episodic returns.
#   - VMPO-matched rollout: 64 envs x 39 steps = 2496 batch (many envs, short
#     fresh rollouts). Stagger spacing 1000/64 ~= 15.6 steps. 1M budget =>
#     ~374 iterations; warmup costs 64k (6.4%).
#   - SINGLE JOINT UPDATE per rollout (update_epochs=1, num_minibatches=1):
#     one forward/backward/step over the whole batch -- VMPO's cadence. The
#     4-way dual backward still ends in one optimizer.step(). epochs_run reads
#     1 every iteration; the KL leash is a retained no-op at 1 epoch.
#   - UNSHARED backbone (share_backbone=False default): separate actor/critic
#     trunks. Each trunk now receives exactly TWO clipped streams (actor trunk:
#     PG + aux-actor; critic trunk: value + aux-critic) -- the v8 trunk-sum
#     confound is gone by construction.
#   - SPLIT learning rates: one Adam, two param groups (actor 3e-3 / critic 1e-3).
#     Actor keeps the hilr winner value (behavior change needs big steps under the
#     KL leash); critic tracks supervised targets stably at lower LR with fewer
#     spike-driven excursions. Groups are asserted disjoint at build.
#   - Inherited epoch-level target_kl leash (non-binding at 1 epoch; v9
#     debug/epochs_run counter included for the KL-binding report).
# Watch: debug/epochs_run, charts/learning_rate_actor(/_critic),
# losses/nextlat_{actor,critic}_{pred,kl}_loss, charts/phase_*_s.
# =====================================================================================
# PPO + IterThink v24 Beta + d3bucket MTP + PPO-ADVNORM(batch) + NEXTLAT v8_headsplit.
# =====================================================================================
# HEADSPLIT: v7_std's single trunk-latent nextlat becomes TWO head-separated auxes.
# The backbone is SHARED (share_backbone=True), so actor_feat == critic_feat and a
# naive per-side duplication of the latent predictor would be vacuous. Instead the
# split is at the HEAD TARGETS:
#   - actor side (unchanged v7 aux): dynamics MLP([h,a] -> h) rolled to depth d,
#     SmoothL1 vs sg trunk latents + KL(pi(.|sg h) || pi(.|h_hat)) decoded through
#     the ACTOR heads (student path gives the actor heads KL-distill grads).
#   - critic side (NEW): own dynamics MLP([h,a] -> h) + own readout (h -> 511
#     horizon-0 value logits), SmoothL1 vs sg online critic logits + KL between
#     the categorical value distributions. The critic HEAD sees only value-CE
#     grads (student bypasses it); each head is thus shaped by its own objective
#     plus its own aux, and the trunk receives FOUR separately-clipped
#     contributions (actor 0.25 + critic 0.25 + aux-actor 0.25 + aux-critic 0.25).
# BUDGET CONFOUND (documented, not hidden): trunk total is 1.0 vs v7's 0.75. Each
# aux keeps the proven-effective 0.25 per-loss ceiling; if v8 wins, the follow-up
# is a matched-sum ablation (0.125 + 0.125). Control for this A/B is the v7_std
# 1M run (same seed/config/standards), NOT the v1 8M run.
# Watch: losses/nextlat_actor_pred_loss, losses/nextlat_actor_kl_loss,
# losses/nextlat_critic_pred_loss, losses/nextlat_critic_kl_loss,
# losses/nextlat_actor_grad_norm, losses/nextlat_critic_grad_norm.
# =====================================================================================
# PPO + IterThink v24 Beta + d3bucket MTP + PPO-ADVNORM(batch) + NEXTLAT v7_std.
# =====================================================================================
# STANDARDS PORT of batch_nextlat_v1 (the nextlat_v1_clip025 run: 10205.6 final-20
# @6.88M, launched with --nextlat-grad-clip 0.25; all other args default).
# Algorithm is UNCHANGED; only shared-standards machinery is swapped in:
#   - cleanrl/shared/vector_norm.py (runner-side obs norm; reward norm wired
#     behind the existing normalize_reward/clip_reward toggles, both off here)
#   - cleanrl/shared/staggered_envs.py (one-horizon stochastic phase warmup;
#     warmup transitions charged against the budget => num_iterations -1)
#   - cleanrl/shared/ppo_loop.py (device_minibatches, explained_variance,
#     gather_metrics; the file's custom valids-GAE is KEPT -- shared compute_gae
#     has different boundary semantics and this loop is already a single pass)
#   - cleanrl/shared/runtime.py + timing.py (TF32/high precision, phase timers)
# PARITY NOTE: stagger + RNG-stream changes (torch Generator minibatches) make
# bit-parity impossible; parity bar is curve-level (@2M/4M/6M + final-20).
# Original v1 header follows.
# =====================================================================================
# PPO + IterThink v24 Beta + d3bucket MTP + PPO-ADVNORM(batch) + NEXTLAT (paper-aligned) v1.
# =====================================================================================
# NEXTLAT: the PAPER-ALIGNED one-step/multi-step next-latent prediction auxiliary
# (arXiv:2511.05963 "NextLat"), ported faithfully to this PPO base as the CONTROL arm
# against latenttd_v1 (TD/lambda-return latent self-prediction). Mapping to control:
#   - latent h_t = final trunk output (paper: final-layer pre-logit hidden), the same
#     H-dim feature the Beta action heads consume.
#   - "next token" X_{t+1} -> the ACTION a_t: in an MDP rollout the action is the new
#     input that drives h_t -> h_{t+1}. (Conditioning on o_{t+1} would be trivial here:
#     the encoder is stateless, so h_{t+1} is a function of o_{t+1} alone.)
#   - latent dynamics model p_psi = simple MLP([h, a] -> h_next), rolled out to depth d
#     with ALL intermediate steps supervised (paper Sec. 3): h_hat_{t+i} vs
#     sg[h_{t+i}], Smooth L1, RAW latents (paper uses no latent normalization, no EMA,
#     no snapshot -- targets are ONLINE stop-grad recomputes of the current trunk).
#   - KL distillation term (paper Eq. 4, "agreement in token-prediction space"): our
#     token space is the ACTION distribution, so KL( pi(.|sg h_{t+i}) || pi(.|h_hat_{t+i}) )
#     distills the true future policy into predictions decoded from imagined latents.
# Depth masks reuse the critic-MTP boundary logic (no reset between t and t+i). The aux
# gets ITS OWN dual-backward clip budget (nextlat_grad_clip=0.1 vs actor 0.25 / critic
# 0.25) -- identical budget to latenttd_v1, so the A/B isolates the TARGET STRUCTURE
# (one-step/rollout prediction vs TD-horizon) at matched gradient allowance.
# HYPOTHESIS (control): if the user-observed next-latent uplift is horizon-independent,
# this arm matches latenttd_v1; if the TD horizon is the active ingredient, latenttd wins.
# Watch: losses/nextlat_pred_loss, losses/nextlat_kl_loss, debug/latent_batch_std
# (collapse probe: per-dim cross-batch std of raw latents -> 0 means collapse).
#
# CONFIG NOTE: bakes in the "ppoadvnorm_batch" base config as defaults (norm_adv=True,
# norm_adv_scope="batch", ret_percnorm=False) — the config of run
# HalfCheetah-v4__iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_v1__1__1782008060,
# the bar to beat. --no-nextlat recovers that base's algorithm/config (shared params get
# bit-identical init; the extra predictor draws shift the RNG stream afterward).
# =====================================================================================
#
# --- inherited mbpercnorm_v2 notes (percnorm OFF here; kept for lineage) ---
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
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport
from cleanrl.shared.ppo_loop import device_minibatches, explained_variance, gather_metrics
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets,
    episode_horizon,
    run_phase_warmup,
)
from cleanrl.shared.vector_norm import (
    VectorObsNorm,
    VectorRewardNorm,
    make_raw_continuous_env,
)

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
    actor_lr: float = 3e-3          # v10/v14: actor side (incl. predictor)
    critic_lr: float = 1e-3         # v10/v14: critic side (incl. dyn/readout)
    num_envs: int = 64              # v10: VMPO-matched env count
    num_steps: int = 39             # v10: VMPO-matched short rollouts (2496 batch)
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 1        # v10: whole batch at once (VMPO single joint update)
    update_epochs: int = 1          # v10: ONE update per rollout (VMPO cadence, not PPO reuse)
    # NOTE: the KL leash is the inherited epoch-level target_kl break (binds
    # full-batch); v10 adds only the debug/epochs_run counter for it.
    norm_adv: bool = True            # ppoadvnorm_batch base: standard PPO advantage standardization ON
    # --- Percentile advantage normalization (OFF in the ppoadvnorm_batch base) ---
    ret_percnorm: bool = False       # scale policy advantage by S = max(floor, P95-P5) of returns
    ret_perc_scope: str = "minibatch"  # "minibatch" = fresh per-mb P95-P5 of mb returns (NO EMA, like the
    #                                  old per-mb advnorm); "batch" = fresh WHOLE-ROLLOUT P95-P5, one S per
    #                                  update, NO EMA (the batch-vs-mb ablation); "ema" = v1's global EMA.
    ret_perc_rate: float = 0.01      # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
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
    share_backbone: bool = False     # v10: separate actor/critic trunks (LR split needs disjoint groups)
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

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

    # NEXTLAT auxiliary (paper-aligned, arXiv:2511.05963): action-conditioned MLP latent
    # dynamics rolled out to depth d; Smooth L1 on all intermediate predicted latents vs
    # ONLINE stop-grad trunk targets, plus KL distillation of the true future action
    # distribution into the one decoded from the predicted latent (see header).
    nextlat: bool = True
    nextlat_depth: int = 4           # rollout depth d (paper uses 1..8 across tasks)
    nextlat_actor_coef: float = 1.0    # actor-side aux weight (grad clip is the real ceiling)
    nextlat_actor_kl_coef: float = 1.0  # actor KL relative to the actor prediction term
    nextlat_actor_grad_clip: float = 0.25  # actor-aux dual-backward budget (= control's tuned 0.25)
    nextlat_critic_coef: float = 1.0    # critic-side aux weight
    nextlat_critic_kl_coef: float = 1.0  # critic KL relative to the critic prediction term
    nextlat_critic_grad_clip: float = 0.25  # critic-aux dual-backward budget (see trunk-sum note)

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
        # v7_std: normalization moved runner-side (shared/vector_norm.py);
        # make_raw_continuous_env documents the identical raw stack.
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
        # NEXTLAT latent dynamics p_psi: simple MLP([h_t, a_t] -> h_hat_{t+1}) (paper:
        # "simple MLPs"). Small final init => early predictions near zero. Initialized
        # LAST so all base-shared parameters draw identical initial weights from the
        # seeded RNG as the ppoadvnorm_batch base run.
        self.nextlat_predictor = nn.Sequential(
            layer_init(nn.Linear(H + act_dim, H)),
            ReLUSquared(),
            layer_init(nn.Linear(H, H), std=0.1),
        )
        # v8 HEADSPLIT critic side: own latent dynamics + own readout to
        # horizon-0 value logits (num_bins-way). Same structure as the actor
        # side, different head target (value distribution, not trunk latent).
        # Initialized AFTER the actor predictor so all shared/base parameters
        # keep their v7 init draws; only these new params extend the stream.
        self.nextlat_critic_dyn = nn.Sequential(
            layer_init(nn.Linear(H + act_dim, H)),
            ReLUSquared(),
            layer_init(nn.Linear(H, H), std=0.1),
        )
        self.nextlat_critic_readout = layer_init(nn.Linear(H, args.num_bins), std=0.1)

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

    def get_actor_feat(self, x):
        # Actor-side trunk feature only; used for the ONLINE stop-grad nextlat targets.
        return self._trunks(x)[0]

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
        return action, z, log_prob, entropy, value_logits, actor_feat

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

    def nextlat_actor_parameters(self):
        # Params receiving the ACTOR-side aux gradient: the trunk, the actor
        # dynamics MLP, and the action heads (the KL student decodes through
        # them, so they receive PG + KL-distill grads). One dual-backward group.
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + list(self.nextlat_predictor.parameters()) + heads

    def nextlat_critic_parameters(self):
        # Params receiving the CRITIC-side aux gradient: the trunk plus the
        # critic dynamics MLP and value-logit readout. The critic HEAD is
        # deliberately excluded: the student predicts logits directly (teacher
        # is a no-grad head forward), so the head sees only value-CE grads.
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return (
            list(trunk.parameters())
            + list(self.nextlat_critic_dyn.parameters())
            + list(self.nextlat_critic_readout.parameters())
        )


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
    # v7_std: one-horizon phase warmup is charged against the budget.
    warmup_horizon = episode_horizon(args.env_id)
    warmup_transitions = args.num_envs * warmup_horizon
    if warmup_transitions >= args.total_timesteps:
        raise ValueError("total_timesteps must exceed the initial phase warmup")
    args.num_iterations = (args.total_timesteps - warmup_transitions) // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch", "batch_retstd", "minibatch_retstd")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope in ("batch", "batch_retstd")), \
        "norm_adv_scope=batch/batch_retstd requires adv_transform_scope=batch"
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
    configure_runtime()
    phase_timer = PhaseTimer()

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
    # v7_std: runner-side normalization (shared/vector_norm.py). Reward path
    # mirrors the legacy toggles; both are off in the parity config.
    obs_norm = VectorObsNorm(args.num_envs, envs.single_observation_space.shape)
    rew_norm = None
    if args.normalize_reward:
        rew_norm = VectorRewardNorm(
            args.num_envs, args.gamma, clip=10.0 if args.clip_reward else None
        )
    mb_generator = torch.Generator(device="cpu")
    mb_generator.manual_seed(args.seed)

    agent = Agent(envs, args).to(device)
    # v14: one Adam, two LR groups (actor side / critic side), aux aligned to
    # core LRs (v10 stepping). Requires disjoint param sets -- i.e. an
    # unshared backbone -- so no parameter is stepped twice.
    actor_opt_params = list(agent.nextlat_actor_parameters())
    critic_opt_params = list(agent.nextlat_critic_parameters()) + list(agent.critic_head.parameters())
    _actor_ids = {id(p) for p in actor_opt_params}
    _critic_ids = {id(p) for p in critic_opt_params}
    if _actor_ids & _critic_ids:
        raise ValueError("v14 LR split needs disjoint actor/critic groups (unshare the backbone)")
    if {id(p) for p in agent.parameters()} != (_actor_ids | _critic_ids):
        raise ValueError("v14 optimizer groups must cover all parameters exactly once")
    optimizer = optim.Adam(
        [
            {"params": actor_opt_params, "lr": args.actor_lr},
            {"params": critic_opt_params, "lr": args.critic_lr},
        ],
        eps=1e-5,
    )
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    nextlat_actor_params = agent.nextlat_actor_parameters()
    nextlat_critic_params = agent.nextlat_critic_parameters()

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
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    latents = torch.zeros((args.num_steps, args.num_envs, args.hidden)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    # v7_std: phase-staggered warmup (shared/staggered_envs.py). Stochastic
    # draws from the INITIAL policy; transitions count toward the budget.
    phase_offsets = compute_phase_offsets(args.num_envs, warmup_horizon, args.seed)
    writer.add_text("initial_phase_offsets", ",".join(str(o) for o in phase_offsets))

    def _warmup_action(obs_np):
        with torch.no_grad():
            w_action, _, _, _, _, _ = agent.get_action_and_value(
                torch.as_tensor(obs_np, device=device, dtype=torch.float32)
            )
        return w_action.cpu().numpy()

    warm = run_phase_warmup(
        envs,
        obs_norm=obs_norm,
        act_fn=_warmup_action,
        horizon=warmup_horizon,
        phase_offsets=phase_offsets,
        seed=args.seed,
        rew_norm=rew_norm,
    )
    assert warm.transitions == warmup_transitions
    global_step = warm.transitions
    suppress_next_episode_log = warm.suppress_mask
    next_obs = torch.as_tensor(warm.next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs).to(device)
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.actor_lr
            optimizer.param_groups[1]["lr"] = frac * args.critic_lr

        phase_timer.start("rollout")
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits, actor_feat = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
                latents[step] = actor_feat
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            raw_next_obs, raw_reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # v7_std: runner-side normalization (shared/vector_norm.py). Reward
            # path mirrors the legacy toggles (both off in the parity config).
            if rew_norm is not None:
                reward = rew_norm.normalize(raw_reward, terminations)
            elif args.clip_reward:
                reward = np.clip(np.asarray(raw_reward, dtype=np.float32), -10, 10)
            else:
                reward = raw_reward
            normed_next_obs, normed_transition_obs = obs_norm.normalize_step(
                raw_next_obs, terminations, truncations, infos
            )
            next_obs_np = np.array(normed_next_obs, copy=True)
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(normed_transition_obs, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
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
                for env_idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        if suppress_next_episode_log[env_idx]:
                            suppress_next_episode_log[env_idx] = False
                            continue
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        phase_timer.stop()
        phase_timer.start("update")
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
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
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
            # NEXTLAT depth-validity masks: predicted h_hat_{t+i} is supervised only when
            # no reset boundary lies in [t, t+i-1] (so o_{t+i} belongs to the same episode)
            # and t+i stays inside the rollout — the exact critic-MTP masking pattern.
            # Targets themselves are ONLINE (recomputed with stop-grad per minibatch), so
            # only the masks are precomputed here.
            if args.nextlat:
                nextlat_mask = torch.zeros(
                    (args.num_steps, args.num_envs, args.nextlat_depth), device=device
                )
                for i in range(1, args.nextlat_depth + 1):
                    valid_len = args.num_steps - i
                    if valid_len <= 0:
                        break
                    valid_i = torch.ones(
                        (valid_len, args.num_envs), dtype=torch.bool, device=device
                    )
                    for k in range(i):
                        valid_i &= transition_boundaries[k : k + valid_len] == 0
                    nextlat_mask[:valid_len, :, i - 1] = valid_i.float()
                # Collapse probe: per-dim cross-batch std of the RAW rollout latents
                # (paper uses unnormalized latents) -> 0 means representational collapse.
                latent_batch_std = latents.reshape(-1, args.hidden).std(dim=0).mean().item()
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
        if args.nextlat:
            b_nextlat_mask = nextlat_mask.reshape(-1, args.nextlat_depth)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            b_policy_adv_normed = b_policy_adv / b_returns.std().clamp(min=args.ret_perc_floor)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        # v7_std: device-side shuffled indices (shared/ppo_loop.py); the seeded
        # CPU generator replaces np.arange + np.random.shuffle (same coverage
        # distribution, not bit-identical -- parity bar is curve-level).
        # b_target_probs/mask live on CPU (400MB), so index them with the CPU
        # perm and move one async copy per minibatch for the GPU buffers.
        clipfrac_tensors = []
        epochs_run = 0
        for epoch in range(args.update_epochs):
            epochs_run += 1
            for mb_inds_cpu in device_minibatches(
                args.batch_size,
                args.minibatch_size,
                torch.device("cpu"),
                generator=mb_generator,
            ):
                mb_inds = mb_inds_cpu.to(device, non_blocking=True)

                _, _, newlogprob, entropy, value_logits, mb_actor_feat = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # v7_std: accumulate on-device; single sync at logging time.
                    clipfrac_tensors.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    )

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_advantages = mb_advantages / b_returns[mb_inds].std().clamp(min=args.ret_perc_floor)
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds]
                    qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device)
                    lo, hi = torch.quantile(mb_ret, qs)
                    mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                    mb_advantages = mb_advantages / mb_perc_scale
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

                # v162 HL-Gauss MTP value loss: per-horizon CE to scalar-return
                # targets, summed across valid horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds_cpu].to(device=value_logits.device, non_blocking=True)
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds_cpu].to(device=value_logits.device, dtype=value_ce.dtype, non_blocking=True)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

                # NEXTLAT: roll the latent dynamics MLP to depth d from the CURRENT trunk
                # latent, conditioned on the buffered actions; Smooth L1 each predicted
                # latent against the ONLINE stop-grad trunk latent of the true future obs,
                # and KL-distill the true future action distribution into the one decoded
                # from the predicted latent. T-major flattening => index + i*num_envs is
                # (t+i, same env); masks exclude boundary-crossing and rollout-tail rows.
                if args.nextlat:
                    # ACTOR side: unchanged v7 aux (dynamics on trunk latent +
                    # KL distilled through the actor heads).
                    h_hat = mb_actor_feat
                    actor_pred_losses, actor_kl_losses = [], []
                    # CRITIC side: own dynamics + readout to horizon-0 value
                    # logits, supervised by the sg online critic distribution.
                    hc_hat = mb_actor_feat
                    critic_pred_losses, critic_kl_losses = [], []
                    for i in range(1, args.nextlat_depth + 1):
                        act_idx = torch.clamp(mb_inds + (i - 1) * args.num_envs, 0, args.batch_size - 1)
                        tgt_idx = torch.clamp(mb_inds + i * args.num_envs, 0, args.batch_size - 1)
                        h_hat = agent.nextlat_predictor(
                            torch.cat([h_hat, b_actions[act_idx]], dim=-1)
                        )
                        hc_hat = agent.nextlat_critic_dyn(
                            torch.cat([hc_hat, b_actions[act_idx]], dim=-1)
                        )
                        z_hat = agent.nextlat_critic_readout(hc_hat)
                        with torch.no_grad():
                            tgt_feat = agent.get_actor_feat(b_obs[tgt_idx])
                            t_dist, _, _ = agent._actor_dist(tgt_feat)  # teacher (sg)
                            tgt_logits = agent.get_value(b_obs[tgt_idx])[:, 0]  # (B,bins)
                        mask_i = b_nextlat_mask[mb_inds, i - 1]
                        denom = mask_i.sum().clamp_min(1.0)
                        pred_l = F.smooth_l1_loss(h_hat, tgt_feat, reduction="none").mean(-1)
                        actor_pred_losses.append((pred_l * mask_i).sum() / denom)
                        s_dist, _, _ = agent._actor_dist(h_hat)         # student
                        kl_l = torch.distributions.kl_divergence(t_dist, s_dist).sum(-1)
                        actor_kl_losses.append((kl_l * mask_i).sum() / denom)
                        cpred_l = F.smooth_l1_loss(z_hat, tgt_logits, reduction="none").mean(-1)
                        critic_pred_losses.append((cpred_l * mask_i).sum() / denom)
                        ckl_l = torch.distributions.kl_divergence(
                            Categorical(logits=tgt_logits), Categorical(logits=z_hat)
                        )
                        critic_kl_losses.append((ckl_l * mask_i).sum() / denom)
                    nextlat_actor_pred_loss = torch.stack(actor_pred_losses).mean()
                    nextlat_actor_kl_loss = torch.stack(actor_kl_losses).mean()
                    nextlat_actor_loss = (
                        nextlat_actor_pred_loss
                        + args.nextlat_actor_kl_coef * nextlat_actor_kl_loss
                    )
                    nextlat_critic_pred_loss = torch.stack(critic_pred_losses).mean()
                    nextlat_critic_kl_loss = torch.stack(critic_kl_losses).mean()
                    nextlat_critic_loss = (
                        nextlat_critic_pred_loss
                        + args.nextlat_critic_kl_coef * nextlat_critic_kl_loss
                    )
                else:
                    nextlat_actor_loss = nextlat_critic_loss = None

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
                    # DUAL-BACKWARD decoupled clipping (now 4-way). Backprop value,
                    # actor-aux, critic-aux, and policy gradients separately, clip
                    # each to its own max-norm, then sum on the (possibly shared)
                    # trunk so no single loss can swamp the others' contribution
                    # to shared features.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    # Stash clipped value grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    if args.nextlat:
                        optimizer.zero_grad(set_to_none=True)
                        (args.nextlat_actor_coef * nextlat_actor_loss).backward(retain_graph=True)
                        nextlat_actor_gn = nn.utils.clip_grad_norm_(
                            nextlat_actor_params, args.nextlat_actor_grad_clip
                        )
                        nextlat_actor_grads = [
                            (p, p.grad.detach().clone())
                            for p in nextlat_actor_params if p.grad is not None
                        ]
                        optimizer.zero_grad(set_to_none=True)
                        (args.nextlat_critic_coef * nextlat_critic_loss).backward(retain_graph=True)
                        nextlat_critic_gn = nn.utils.clip_grad_norm_(
                            nextlat_critic_params, args.nextlat_critic_grad_clip
                        )
                        nextlat_critic_grads = [
                            (p, p.grad.detach().clone())
                            for p in nextlat_critic_params if p.grad is not None
                        ]
                    else:
                        nextlat_actor_grads = []
                        nextlat_critic_grads = []
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk), clip_aux_actor and
                    # clip_aux_critic. critic_head / predictors / readout get only
                    # their own loss's gradient; the action heads get policy +
                    # clipped actor-KL-distill gradients.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    for p, g in nextlat_actor_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    for p, g in nextlat_critic_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    if args.nextlat:
                        loss = (
                            loss
                            + args.nextlat_actor_coef * nextlat_actor_loss
                            + args.nextlat_critic_coef * nextlat_critic_loss
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # v7_std: end the update phase before any logging sync; on-device
        # explained variance; one packed D2H for all log-time scalars.
        phase_timer.stop()
        update_stats = phase_timer.summary()
        phase_timer.reset()
        log_tensors = {
            "value_loss": v_loss,
            "policy_loss": pg_loss,
            "entropy": entropy_loss,
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "clipfrac": torch.stack(clipfrac_tensors).mean(),
            "explained_variance": explained_variance(b_values, b_returns),
            "returns_mean": b_returns.mean(),
            "returns_std": b_returns.std(),
            "returns_absmax": b_returns.abs().max(),
        }
        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_per_h = b_target_probs[..., 0] + b_target_probs[..., -1]
        edge_mask_f = b_target_mask.to(dtype=edge_per_h.dtype)
        log_tensors["target_edge_mass"] = (edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)
        if args.nextlat:
            log_tensors["nextlat_actor_pred_loss"] = nextlat_actor_pred_loss
            log_tensors["nextlat_actor_kl_loss"] = nextlat_actor_kl_loss
            log_tensors["nextlat_critic_pred_loss"] = nextlat_critic_pred_loss
            log_tensors["nextlat_critic_kl_loss"] = nextlat_critic_kl_loss
        log_vals = gather_metrics(log_tensors)

        writer.add_scalar("charts/learning_rate_actor", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/learning_rate_critic", optimizer.param_groups[1]["lr"], global_step)
        writer.add_scalar("losses/value_loss", log_vals["value_loss"], global_step)
        writer.add_scalar("losses/policy_loss", log_vals["policy_loss"], global_step)
        writer.add_scalar("losses/entropy", log_vals["entropy"], global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        if args.nextlat:
            writer.add_scalar("losses/nextlat_actor_pred_loss", log_vals["nextlat_actor_pred_loss"], global_step)
            writer.add_scalar("losses/nextlat_actor_kl_loss", log_vals["nextlat_actor_kl_loss"], global_step)
            writer.add_scalar("losses/nextlat_critic_pred_loss", log_vals["nextlat_critic_pred_loss"], global_step)
            writer.add_scalar("losses/nextlat_critic_kl_loss", log_vals["nextlat_critic_kl_loss"], global_step)
            if args.separate_grad_clip:
                writer.add_scalar("losses/nextlat_actor_grad_norm", float(nextlat_actor_gn), global_step)
                writer.add_scalar("losses/nextlat_critic_grad_norm", float(nextlat_critic_gn), global_step)
            writer.add_scalar("debug/latent_batch_std", latent_batch_std, global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
            writer.add_scalar("debug/soft_adv_std_ratio", (policy_adv.std() / (advantages.std() + 1e-8)).item(), global_step)
        writer.add_scalar("losses/old_approx_kl", log_vals["old_approx_kl"], global_step)
        writer.add_scalar("losses/approx_kl", log_vals["approx_kl"], global_step)
        writer.add_scalar("losses/clipfrac", log_vals["clipfrac"], global_step)
        writer.add_scalar("losses/explained_variance", log_vals["explained_variance"], global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", log_vals["returns_mean"], global_step)
        writer.add_scalar("debug/returns_std", log_vals["returns_std"], global_step)
        writer.add_scalar("debug/returns_absmax", log_vals["returns_absmax"], global_step)
        writer.add_scalar("debug/target_edge_mass", log_vals["target_edge_mass"], global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)
        writer.add_scalar("debug/u_edge_frac", u_edge_frac, global_step)
        writer.add_scalar("debug/sigma_mean", b_sigma.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # v7_std: per-phase breakdown (seconds in this logging window).
        for _phase, _stat in update_stats.items():
            writer.add_scalar(f"charts/phase_{_phase}_s", _stat["total_s"], global_step)
        writer.add_scalar("debug/epochs_run", epochs_run, global_step)

    envs.close()
    writer.close()

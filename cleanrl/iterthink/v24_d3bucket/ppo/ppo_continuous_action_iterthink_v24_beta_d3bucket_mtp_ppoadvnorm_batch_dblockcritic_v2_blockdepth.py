# PPO + IterThink v24 Beta + d3bucket MTP + PPO-ADVNORM(batch) + DIFFUSIONBLOCKS VALUE v2.
# =====================================================================================
# DiffusionBlocks (ICLR 2026, arXiv:2506.14202) applied to the value function, with a
# built-in CONTROL: --dblock / --no-dblock select block-local vs end-to-end training of the
# SAME value network, so "is DiffusionBlocks helping" is one flag, not two codebases.
#
# THE PAPER. A residual stack z_l = z_{l-1} + f_l(z_{l-1}) is a discretization of an ODE.
# Reinterpret that ODE as the EDM probability-flow ODE on a LATENT of the TARGET, and each
# layer becomes a denoiser for one interval of the noise level. Denoisers at different noise
# levels have INDEPENDENT training problems (score matching factorizes over sigma), so each
# block can be trained alone on the FULL task loss applied to its own one-step denoise --
# never on another block's output. Consequences: no backward through depth, no cross-block
# activations retained, and every block's update computable in parallel. Depth becomes
# diffusion time; a forward pass becomes an ODE solve. B=1 recovers ordinary training.
#
# WHY THE VALUE FUNCTION. The paper needs a network with a TARGET, and the critic is exactly
# the paper's setting: a 511-way classification (Dreamer3 symexp buckets, HL-Gauss labels)
# over an ordered support. The actor has no target -- its loss is a policy-gradient surrogate
# on a sampled action -- so applying DiffusionBlocks to it would require replacing PPO's
# ratio with advantage-weighted diffusion BC (a different algorithm, not a different training
# schedule). The actor is therefore left EXACTLY as the base's tuned ThinkTrunk + Beta heads
# in both regimes, which is also what makes the A/B clean: the only thing that changes
# between the two runs is how the value stack's gradient is computed.
#
# v2 vs v1: PER-BLOCK DEPTH (dblock_layers_per_block, default 2; v1 was hard-wired to 1).
# A "block" in the paper is the contiguous group of L/B transformer LAYERS that one Euler step
# replaces: 4 layers (ViT/DiT/MD4, L=12 B=3), 8 (DiT-L/2, L=24), 3 (AR LM, B=4). The block
# count B and the per-block capacity L/B are separate axes and the paper's Table 8 (CIFAR-10,
# L=12 fixed, gamma=0) shows quality tracks the SECOND: FID 35.47 (L/B=6, beats end-to-end
# 39.83) < 38.03 (4) << 45.43 (3) << 53.32 (2), attributed to "reduced capacity per block".
# v1 ran L/B=1 -- one sub-layer per Euler step, off the end of that axis -- and its value net
# trailed the base's single linear head by ~15% return at matched steps (2637 vs 3079 @1M,
# 4284 vs 5012 @2M) with tied explained_variance (0.870 vs 0.866) and correctly-ordered
# per-block CE (1.290 > 1.275 > 1.248): the ODE was refining, the blocks were just too thin.
# HYPOTHESIS: L/B=2 (B=3, 6 layers total) restores per-step capacity and closes that gap,
# while keeping the memory property intact -- peak scales with L/B, not with B*L/B.
#
# THE ARCHITECTURE (identical parameters, count and readout in both regimes)
#   cond_t   = trunk(obs) + h_embed[t]          M = critic_mtp_horizon TOKENS, one per MTP
#                                               horizon; they are conditionally independent
#                                               given cond, so there is nothing to attend
#                                               over and the DiT block keeps only adaLN.
#   B blocks f_b, each L/B adaLN sub-layers deep, residual WITHIN the block:
#            h = 0; for l: u = in_proj_l([RMSNorm(z if l==0 else h), RMSNorm(cond)]),
#            u = u*(1+scale_l(t)) + shift_l(t),  h = h + out_proj_l(ReLU(u)^2);  F = h
#   decoder  = Linear(H, 511, bias=False), ZERO-init, shared by every block
#   --no-dblock  z_0 = z_start; z_b = z_{b-1} + f_b(z_{b-1}, cond, t(0)); logits = dec(z_B)
#                ONE backward through B composed blocks. This is the network the paper
#                converts, at the same depth and parameter count.
#   --dblock     z0 = lat(p*) of the HL-Gauss label; block b owns the b-th EQUAL-MASS
#                interval of the EDM training lognormal and is trained ALONE on
#                   sigma ~ lognormal | bin_b ,  z_t = z0 + sigma*eps
#                   D_b   = c_skip z_t + c_out f_b(c_in z_t, cond, c_noise)     (EDM precond)
#                   L_b   = lambda(sigma) * CE(decoder(D_b), p*)
#                Inference is the Euler solve of the same ODE from pure noise down the sigma
#                ladder, visiting the block that owns each sigma (num_blocks calls).
#
# Every parameter of the block stack carries a leading (num_blocks,) axis and every op is an
# einsum over it, so TRAINING ALL BLOCKS IS ONE BATCHED CALL -- the sibling structure the
# method creates, expressed in the forward pass. A residual stack cannot do this.
#
# MEASURED (one value training step, batch 1024 x 6 horizons, RTX 5090, /tmp/dblock_scale.py)
#   B x L/B:            3x1   3x2   3x4   6x2  12x2  24x2      (params 164k .. 2.03M)
#   e2e        peak MiB 221   274   379   379   591  1015    step ms 18.9 10.7 11.8 12.7 18.2 30.4
#   dblock-all peak MiB 304   356   459   583  1039  1945    step ms  8.2  9.1  9.2  8.8 10.3 16.5
#   dblock-one peak MiB 188   206   243   209   216   230    step ms  8.0  9.1  9.3  8.0  9.2  9.0
# dblock-one (--no-dblock-train-all-blocks, the paper's mode) is the claim, exactly: peak
# memory tracks L/B (188/206/243 for L/B=1/2/4) and is FLAT in B (206/209/216/230 at B=3/6/
# 12/24, i.e. +12% for 8x the blocks and 12x the parameters, against 3.7x for end-to-end),
# and it is 3.4x faster per step at B=24. dblock-all keeps B graphs alive so it trades that
# memory back for a full update of every block per minibatch, and is still faster than e2e
# (no depth-serial backward). Default is dblock-all at B=3, L/B=2: at this scale samples,
# not bytes, are the binding constraint -- 356 MiB is free on this GPU.
#
# THE VALUE LATENT (the one design decision the paper leaves open). The paper diffuses the
# L2-normalized EMBEDDING of the ground-truth label from a free learned table. A free table
# throws away the fact that our 511 "labels" are ORDERED values. The bin latents here are
# FIXED FOURIER FEATURES of the bin's symlog coordinate u_b in [-1,1]:
#     e_b = [sin(pi f_k u_b), cos(pi f_k u_b)]_k ,  f_k geometric in [1, num_bins/2]
#     lat_b = dblock_latent_norm * normalize(e_b)
# (a) Every row has identical norm, so the discriminability ||lat_b - lat_c||/(2 sigma) that
# decides how much of the answer a block can READ off its input instead of predicting it is
# the same for every value bucket and is set by ONE number. dblock_latent_norm=1.0 with
# sigma_data=0.5 is the paper's geometry (it L2-normalizes its 768-dim label embeddings and
# leaves sigma_data=0.5, a deliberate 14x "mismatch"; at H=64 ours is 4.6x). Measured on the
# 4k-step supervised probe, final explained_variance is flat in this knob -- 0.9989 (norm 0.5)
# / 0.9969 (1.0) / 0.9944 (2.0) / 0.9938 (4.0) / 0.9980 (norm 1.0 with sigma_data lowered to
# the exactly-calibrated 1/sqrt(H)=0.125) -- but the per-block CE series is NOT: at norm 0.5
# it is flat-to-rising (1.53, 1.52, 1.66), i.e. z carries nothing and the blocks degenerate
# into three independent single-shot predictors, while at norm 1-4 it falls monotonically
# (1.47 > 1.43 > 1.42 at norm 1.0), i.e. the solve genuinely refines. Keeping the paper's
# value buys the refinement structure at no measured accuracy cost.
# (b) lat(p) = p @ lat is p's characteristic function sampled at the f_k, so E[z0|p] is
# LINEAR in p: a soft HL-Gauss label maps to its EXACT conditional-mean latent and the
# denoiser is literally doing progressive refinement of the value DISTRIBUTION, coarse
# frequencies first.
# (c) The readout stays a single zero-init bias-free Linear(H, 511), so logits == 0 =>
# uniform probs => V(s) is EXACTLY 0 at init on the symmetric symexp support, in BOTH
# regimes -- the same neutral start as the head this replaces. The control needs a NONZERO
# z_start to avoid an all-zero fixed point (see ValueNet.__init__); the ODE does not, because
# its stream carries the noised label latent.
#
# DELIBERATE DEVIATIONS FROM THE REFERENCE IMPL
#  1. ALL BLOCKS PER STEP (dblock_train_all_blocks=True, off switch provided). The paper
#     samples ONE block per optimizer step because memory is its whole point. Here memory is
#     free and sample efficiency is everything, so every block denoises its own sigma draw on
#     every minibatch. Gradients stay strictly block-local either way: the blocks are
#     siblings, never composed, so this is the same estimator with more samples per step.
#  2. COMMON RANDOM NUMBER for the ODE (one frozen seed, drawn at construction). V(s) MUST be
#     a deterministic function of s -- GAE, the bootstrap and the regression target are all
#     computed from it -- so the solve's start point is frozen forever rather than resampled,
#     which would inject uncorrelated noise straight into every TD residual.
#  3. TIME-EMBEDDING FREQUENCY RANGE. DiT's ladder is calibrated for timesteps in [0, 1000];
#     c_noise = 0.25*log(sigma) spans [-1.55, 1.10] here, so the ladder is re-cut to [0.5, 8]
#     (see TimeEmbed) -- the reference numbers would be a near-constant conditioning signal.
# Everything else follows the reference, including the equal-mass bin partition, the
# gamma-widened per-block sigma ranges, the EDM lambda weighting of the CE, and the
# decode-and-reproject Euler state (which is load-bearing, not cosmetic -- see _ode_logits).
#
# WATCH: charts/explained_variance (the value function is the thing being changed -- if
# block-local training costs accuracy, this drops below the control's immediately);
# losses/dblock_ce_b{0,1,2} (per-block UNWEIGHTED CE; block 0 owns the LARGEST sigmas, so b0
# == "predict the value distribution from the observation alone" == exactly the base critic's
# job, and each later block refines a cleaner latent, so the series should be monotonically
# DECREASING in b -- measured 1.47 > 1.42 > 1.40 on a supervised probe);
# debug/dblock_pred_entropy (uniform ~6.24 => the ODE is not committing).
#
# CONFIG NOTE: this file bakes in the "ppoadvnorm_batch" base config as defaults
# (norm_adv=True, norm_adv_scope="batch", ret_percnorm=False) -- the exact config of run
# HalfCheetah-v4__iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_v1__1__1782008060, whose
# 8454 is the absolute bar. That base's value head is a single Linear on the trunk; BOTH
# regimes here replace it with a depth-3 stack, so --no-dblock is the capacity-matched
# control for the method and 8454 is the reference for the architecture as a whole.
# =====================================================================================
#
# --- inherited mbpercnorm_v2 notes (percnorm OFF here; kept for lineage) ---
# PPO + IterThink v24 Beta + v162 critic + Dreamer3-bucket HL-Gauss MTP + MB PERCNORM v2.
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
from math import exp, log, pi, sqrt
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
VALUE_CHUNK = 4096  # states per chunk of the bootstrap value pass (bounds the decode tensor)


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
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
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

    # --- DiffusionBlocks value network (see header) ---------------------------------
    # THE A/B SWITCH. Both settings use the SAME value architecture, the same parameter
    # count, the same HL-Gauss labels and the same MTP mask:
    #   --dblock     block-local DiffusionBlocks training (no backward through depth)
    #   --no-dblock  the residual stack it converts, trained end-to-end (the control)
    dblock: bool = True
    dblock_num_blocks: int = 3        # B: ODE steps == blocks; block 0 owns the LARGEST sigmas
    # LAYERS PER BLOCK (L/B). The capacity of one Euler step, and the axis v1 pinned at 1.
    # The paper's configs are 4 (ViT/DiT/MD4 at L=12,B=3), 8 (DiT-L/2, L=24) and 3 (AR LM,
    # B=4); its Table 8 sweep at L=12 fixed reads FID 35.47 (L/B=6) < 38.03 (4) << 45.43 (3)
    # << 53.32 (2), i.e. quality tracks per-block capacity, NOT the block count. Peak training
    # memory scales with THIS number in the dblock regime (activations for one block only) and
    # with num_blocks*this end-to-end -- the whole point of the method.
    dblock_layers_per_block: int = 2
    dblock_mult: int = 2              # sub-layer MLP width = mult * hidden
    dblock_sigma_min: float = 0.002   # EDM support endpoints (paper values)
    dblock_sigma_max: float = 80.0
    dblock_sigma_data: float = 0.5    # EDM preconditioning scale (paper value)
    # L2 NORM of a bin latent: it sets the discriminability ||lat_b - lat_c|| / (2*sigma) of
    # two value buckets given the noisy latent, i.e. how much of the answer a block can READ
    # off its input instead of predicting it from the observation. 1.0 is the paper's value
    # (it L2-normalizes its label embeddings and leaves sigma_data=0.5, so the training sigma
    # mass deliberately sits ABOVE the data scale). Measured on a 4k-step supervised probe:
    # decoded explained_variance is flat over norm 0.5-4.0 (0.9989 / 0.9969 / 0.9944 / 0.9938)
    # but the per-block CE series only refines monotonically for norm >= 1.0 -- below that, z
    # carries nothing and the blocks degenerate into independent single-shot predictors.
    dblock_latent_norm: float = 1.0
    dblock_p_mean: float = -1.2       # EDM training sigma ~ lognormal(p_mean, p_std)
    dblock_p_std: float = 1.2
    dblock_gamma: float = 0.05        # widen each block's sigma bin by this fraction of its
                                      # log-range on both sides (paper default), so the
                                      # inference ladder's endpoints are not on a bin edge
    dblock_inference_steps: int = 0   # Euler ladder length; 0 => dblock_num_blocks (1 call/block)
    dblock_train_all_blocks: bool = True   # False => paper-faithful ONE block per minibatch
    dblock_weight_normalize: bool = True   # EDM weights to mean 1 within each block

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


# =====================================================================================
# DiffusionBlocks (arXiv:2506.14202) value head.
# =====================================================================================
# scipy-free standard-normal CDF/quantile (the reference impl uses scipy.stats.norm).
def _std_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / sqrt(2.0)))


def _std_ppf(p):
    return sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


def _cdf_scalar(x):
    return _std_cdf(torch.tensor(x, dtype=torch.float64)).item()


def _ppf_scalar(p):
    return _std_ppf(torch.tensor(p, dtype=torch.float64)).item()


def block_sigma_edges(num_blocks, sigma_min, sigma_max, p_mean, p_std):
    """num_blocks+1 sigma endpoints carving the EDM lognormal into EQUAL PROBABILITY MASS
    intervals (dblock_modules.get_block_sigmas). Ascending: edges[i]..edges[i+1] is the
    mass bin owned by ONE block."""
    cdf_min = _cdf_scalar((log(sigma_min) - p_mean) / p_std)
    cdf_max = _cdf_scalar((log(sigma_max) - p_mean) / p_std)
    return [
        exp(p_mean + p_std * _ppf_scalar(cdf_min + (cdf_max - cdf_min) * (i / num_blocks)))
        for i in range(num_blocks + 1)
    ]


def inference_sigmas(num_steps, sigma_min, sigma_max, p_mean, p_std):
    """DESCENDING equal-mass sigma ladder for the Euler solve
    (dblock_modules.get_discrete_sigmas with dblock=True)."""
    cdf_min = _cdf_scalar((log(sigma_min) - p_mean) / p_std)
    cdf_max = _cdf_scalar((log(sigma_max) - p_mean) / p_std)
    if num_steps == 1:
        # Degenerate ladder = ONE denoise from pure noise: the single-shot obs->value
        # readout, i.e. the base critic's structure. Kept as the ablation baseline.
        return [sigma_max]
    return [
        exp(p_mean + p_std * _ppf_scalar(cdf_max + (cdf_min - cdf_max) * (i / (num_steps - 1))))
        for i in range(num_steps)
    ]


class TimeEmbed(nn.Module):
    """DiT TimestepEmbedder over c_noise = 0.25*log(sigma).

    FREQUENCY RANGE matters and cannot be copied from the reference. DiT embeds integer
    timesteps in [0, 1000] with geometric frequencies in [1e-4, 1]; c_noise here spans only
    [-1.55, 1.10] over the EDM support, so that ladder would be nearly constant, and a
    2**arange(16) ladder is the opposite failure: its top frequency completes 8103 cycles
    across the range, so a 1% change in sigma fully decorrelates half the features
    (measured) and they HASH sigma instead of encoding it -- fatal for adaLN, which must
    interpolate to the three fixed inference sigmas that training never draws exactly.
    Geometric in [0.5, 8]: the top frequency completes ~3.4 cycles over the range, which
    resolves adjacent sigma bins while staying smooth in sigma."""

    def __init__(self, H, n_freq=16, f_min=0.5, f_max=8.0):
        super().__init__()
        self.register_buffer(
            "freqs", torch.logspace(log(f_min) / log(10.0), log(f_max) / log(10.0), n_freq)
        )
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(2 * n_freq, H)),
            ReLUSquared(),
            layer_init(nn.Linear(H, H)),
        )

    def forward(self, c_noise):                                       # (B,) -> (B, H)
        a = c_noise.unsqueeze(-1) * self.freqs
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=-1))


class BlockStack(nn.Module):
    """The B blocks of the value network, with a leading BLOCK axis on every parameter.

    One block is `n_layer` DiT sub-layers, minus attention -- there is no sequence to attend
    over here: the MTP horizon tokens are conditionally independent given the conditioning,
    so an attention mixer over 6 independent value distributions would buy nothing.

        h = 0
        for l in range(n_layer):                          # z: diffusion state, cond: obs code
            s = z if l == 0 else h
            u = in_proj_l([RMSNorm(s), RMSNorm(cond)])
            u = u * (1 + scale_l(t)) + shift_l(t)         # adaLN noise conditioning, ZERO-init
            h = h + out_proj_l(ReLU(u)^2)                 # out_proj ZERO-init
        F = h                                            #  =>  F == 0 at init, every depth

    WHY n_layer > 1. A "block" in the paper is the contiguous group of L/B transformer layers
    that one Euler step of the probability-flow ODE replaces -- 4 layers in the ViT/DiT/MD4
    experiments (L=12, B=3), 8 for DiT-L/2 on ImageNet, 3 for the B=4 autoregressive LM. The
    ODE step count B and the per-block capacity L/B are separate axes, and Table 8 (CIFAR-10,
    L=12 fixed) shows quality is governed by the SECOND one: B=2 (6 layers/block) FID 35.47 <
    B=1 end-to-end 39.83 < B=3 (4) 38.03 << B=4 (3) 45.43 << B=6 (2 layers/block) 53.32, which
    the paper attributes to "reduced capacity per block". v1 of this file made every block a
    SINGLE sub-layer -- L/B = 1, off the end of that axis, past the worst point the paper
    ablates -- and its value stack duly trailed the base's linear head by ~15% return at
    matched steps with tied explained_variance. n_layer is that missing axis.

    Every parameter carries a leading (num_blocks, n_layer) axis and every op is an einsum
    over the block axis, so TRAINING ALL BLOCKS IS ONE BATCHED CALL. That batching is only
    legal because the blocks never compose (block b's loss never flows through block b'): it
    is the same property that removes the depth-serial backward, expressed in the forward.
    Sub-layers WITHIN a block do compose -- that is what makes the block deep -- so the
    inner axis is a python loop, exactly the L/B-layer backward the paper pays for.
    Inference indexes one block per Euler step (params[ids]).
    """

    def __init__(self, num_blocks, n_layer, H, mult):
        super().__init__()
        Hm = mult * H
        self.num_blocks, self.n_layer, self.H, self.Hm = num_blocks, n_layer, H, Hm
        # Affine-free RMSNorm on both inputs (file convention): the trunk feature is
        # unnormalized and z is O(sigma_max) early in the solve, so the ReLU^2 needs both
        # scales bounded before the projection. Stateless, so one instance serves every layer.
        self.z_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.c_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.in_w = nn.Parameter(torch.empty(num_blocks, n_layer, Hm, 2 * H))
        self.in_b = nn.Parameter(torch.zeros(num_blocks, n_layer, Hm))
        self.ada_w = nn.Parameter(torch.zeros(num_blocks, n_layer, 2 * Hm, H))
        self.ada_b = nn.Parameter(torch.zeros(num_blocks, n_layer, 2 * Hm))
        self.out_w = nn.Parameter(torch.zeros(num_blocks, n_layer, H, Hm))
        self.out_b = nn.Parameter(torch.zeros(num_blocks, n_layer, H))
        with torch.no_grad():
            for b in range(num_blocks):
                for l in range(n_layer):
                    nn.init.orthogonal_(self.in_w[b, l], np.sqrt(2))

    def forward(self, z, cond, t_emb, ids=None):
        """z (G,T,H) -- one diffusion state per (block, token); cond (T,H) -- shared by every
        block, the paper's re-read input embedding; t_emb (G,T,H); ids (G,) or None for all."""
        p = (
            (self.in_w, self.in_b, self.ada_w, self.ada_b, self.out_w, self.out_b)
            if ids is None
            else (
                self.in_w[ids], self.in_b[ids], self.ada_w[ids],
                self.ada_b[ids], self.out_w[ids], self.out_b[ids],
            )
        )
        n_c = self.c_norm(cond).unsqueeze(0).expand(p[0].shape[0], *cond.shape)
        h = None
        for l in range(self.n_layer):
            w_in, b_in, w_ada, b_ada, w_out, b_out = (t[:, l] for t in p)
            cat = torch.cat([self.z_norm(z if h is None else h), n_c], dim=-1)   # (G,T,2H)
            u = torch.einsum("gtk,gmk->gtm", cat, w_in) + b_in.unsqueeze(1)
            mod = torch.einsum("gth,gmh->gtm", t_emb, w_ada) + b_ada.unsqueeze(1)
            shift, scale = mod.chunk(2, dim=-1)
            u = u * (1.0 + scale) + shift
            f = torch.einsum("gtm,ghm->gth", torch.relu(u).pow(2), w_out) + b_out.unsqueeze(1)
            h = f if h is None else h + f
        return h


class ValueNet(nn.Module):
    """The value network: ONE architecture, TWO training regimes selected by args.dblock.

    Shared structure (identical parameters, identical count, in both regimes):
        cond_t   = critic_feat + h_embed[t]        (T = B*mtp horizon tokens)
        B blocks (BlockStack), each mapping (z, cond, t_emb) -> F
        decoder  = Linear(H, num_bins, bias=False), ZERO-init, shared by all blocks

    args.dblock == False  (END-TO-END CONTROL)
        z_0 = z_start; z_b = z_{b-1} + F_b(z_{b-1}, cond, t_emb(0)); logits = decoder(z_B).
        A plain residual stack read out at the top, trained by ONE backward through all B
        blocks -- the network DiffusionBlocks converts, with the same depth and parameters.

    args.dblock == True   (DIFFUSIONBLOCKS)
        The residual stream is reinterpreted as the EDM probability-flow ODE state over the
        VALUE LATENT z0 = lat(p*) of the HL-Gauss target distribution p*. Block b owns the
        equal-probability-mass sigma interval [edges[b], edges[b+1]] of the EDM training
        lognormal and is trained ALONE on its own one-step denoising problem:
            sigma ~ lognormal restricted to block b's mass bin
            z_t   = z0 + sigma * eps
            D_b   = c_skip(sigma) z_t + c_out(sigma) F_b(c_in(sigma) z_t, cond, c_noise)
            L_b   = lambda(sigma) * CE(decoder(D_b), p*)
        No block ever consumes another block's output during training, so there is no
        depth-serial backward and no cross-block activation to retain. Inference is the
        Euler solve of the same ODE down the sigma ladder, visiting the blocks in order.
    """

    def __init__(self, H, num_bins, mtp, coord_support, coord_max, args):
        super().__init__()
        self.dblock = args.dblock
        self.H, self.num_bins, self.mtp = H, num_bins, mtp
        self.num_blocks = args.dblock_num_blocks
        self.sigma_data = args.dblock_sigma_data
        self.p_mean, self.p_std = args.dblock_p_mean, args.dblock_p_std
        self.weight_normalize = args.dblock_weight_normalize
        self.train_all_blocks = args.dblock_train_all_blocks

        # Per-horizon token embedding: the M MTP heads become M TOKENS sharing one
        # conditioning, each carrying its own diffusion state (the paper's set-of-tokens
        # structure). Horizon identity lives in the conditioning, never in the decoder.
        self.h_embed = nn.Embedding(mtp, H)
        nn.init.normal_(self.h_embed.weight, std=0.02)
        self.t_embed = TimeEmbed(H)
        self.blocks = BlockStack(
            self.num_blocks, args.dblock_layers_per_block, H, args.dblock_mult
        )
        # Shared readout. ZERO-init, exactly like the single bias-free zero-init Linear it
        # replaces: logits == 0 => uniform probs => V(s) is EXACTLY 0 at init on the
        # symmetric symexp support, in BOTH regimes.
        self.decoder = nn.Linear(H, num_bins, bias=False)
        with torch.no_grad():
            self.decoder.weight.zero_()
        # Start-of-stack state for the e2e control (unused by the ODE, which starts from
        # noise). It MUST be nonzero: with a zero start, a zero-init decoder and zero-init
        # out_proj, the control sits on an all-zero fixed point -- dL/d decoder = dL/dlogits
        # (x) z == 0 and dL/d out_proj = decoder^T dL/dlogits (x) . == 0 -- and NOTHING
        # learns (measured: gradient norm exactly 0.00 for 4k steps). The dblock regime is
        # immune because its stream carries the noised label latent, never zero. A learned
        # nonzero query breaks the deadlock exactly the way the base's nonzero trunk feature
        # does for its zero-init linear head, and V(s) is still EXACTLY 0 at init.
        self.z_start = nn.Parameter(torch.randn(H))

        # ---- EDM schedule (dblock regime) -------------------------------------------
        # ASCENDING sigma endpoints carving the EDM lognormal into num_blocks equal
        # probability-mass bins. ORIENTATION (reference estimate_target_layer, model.py:183):
        # block index counts along the ODE, so BLOCK 0 OWNS THE LARGEST sigmas -- it is the
        # first block the solve visits and the one that must answer "which order of magnitude
        # is this return" from the observation alone -- and the last block polishes at
        # sigma_min. The e2e control composes them in the same order (0 first).
        edges = block_sigma_edges(
            self.num_blocks, args.dblock_sigma_min, args.dblock_sigma_max,
            self.p_mean, self.p_std,
        )
        steps = args.dblock_inference_steps or self.num_blocks
        self.ladder = inference_sigmas(
            steps, args.dblock_sigma_min, args.dblock_sigma_max, self.p_mean, self.p_std
        )
        self.sigma_max = args.dblock_sigma_max
        # Per-block sigma bin, WIDENED by dblock_gamma of its log-range on both sides
        # (reference get_sigmas, model.py:161-169) and clamped to the support. Without the
        # widening the ladder's first and last steps sit exactly ON a bin endpoint, i.e. on
        # the measure-zero boundary of what that block ever trained at.
        lo_hi = []
        for b in range(self.num_blocks):
            lo, hi = edges[self.num_blocks - 1 - b], edges[self.num_blocks - b]
            span = args.dblock_gamma * (log(hi) - log(lo))
            lo = max(exp(log(lo) - span), edges[0])
            hi = min(exp(log(hi) + span), edges[-1])
            lo_hi.append((lo, hi))
        self.block_bins = lo_hi                              # host-side, for block lookup
        # Sampling sigma inside block b is u ~ U(cdf_lo[b], cdf_hi[b]) then
        # sigma = exp(p_mean + p_std * Phi^-1(u)) -- uniform in MASS, not in log sigma.
        self.register_buffer(
            "cdf_lo",
            torch.tensor([_cdf_scalar((log(lo) - self.p_mean) / self.p_std) for lo, _ in lo_hi]),
        )
        self.register_buffer(
            "cdf_hi",
            torch.tensor([_cdf_scalar((log(hi) - self.p_mean) / self.p_std) for _, hi in lo_hi]),
        )
        # FIXED Fourier-feature bin latents: geometric frequencies up to the bin Nyquist
        # (num_bins/2 cycles over the symlog coordinate range) so adjacent buckets stay
        # resolvable. All rows share one norm, so the noise geometry -- and hence every
        # block's discriminability -- is identical for every value bucket. lat(p) = p @ lat
        # is then the target distribution's characteristic function sampled at the f_k:
        # E[z0 | p] is LINEAR in p, so a soft HL-Gauss label maps to its EXACT
        # conditional-mean latent and the denoiser refines the value DISTRIBUTION,
        # coarse frequencies first.
        n_freq = H // 2
        u = coord_support.detach().to(device="cpu", dtype=torch.float32) / coord_max  # (N,) in [-1,1]
        freqs = torch.logspace(0.0, log(num_bins / 2.0) / log(10.0), n_freq)
        ang = pi * u.unsqueeze(-1) * freqs.unsqueeze(0)
        e = torch.cat([ang.sin(), ang.cos()], dim=-1)                  # (N, H)
        self.register_buffer("bin_latents", F.normalize(e, dim=-1) * args.dblock_latent_norm)
        # COMMON RANDOM NUMBER for the ODE: one fixed seed, drawn once, shared by every
        # state and every horizon forever. V(s) must be a DETERMINISTIC function of s --
        # GAE, the bootstrap and the regression target are all computed from it -- so the
        # solve's start point is frozen at construction rather than resampled per call.
        self.register_buffer("z_seed", torch.randn(1, H))

    # ---- schedule helpers ------------------------------------------------------------
    def block_for_sigma(self, sigma):
        """Index of the block owning this sigma. Block 0 owns the LARGEST bin, so the
        descending inference ladder visits 0, 1, ... num_blocks-1 in order."""
        for b in range(self.num_blocks):
            if sigma >= self.block_bins[b][0]:
                return b
        return self.num_blocks - 1

    def sample_sigma(self, block_ids, shape, device):
        """sigma ~ EDM lognormal RESTRICTED to each block's (gamma-widened) mass bin."""
        tail = [1] * (len(shape) - 1)
        lo = self.cdf_lo[block_ids].view(-1, *tail)
        hi = self.cdf_hi[block_ids].view(-1, *tail)
        u = torch.rand(shape, device=device) * (hi - lo) + lo
        return torch.exp(self.p_mean + self.p_std * _std_ppf(u.clamp(1e-7, 1 - 1e-7)))

    def edm_weight(self, sigma):
        """EDM lambda(sigma) = (sigma^2 + sd^2) / (sigma*sd)^2 -- the weighting that makes
        the loss on D equivalent to a unit-weight loss on the raw network output F."""
        sd = self.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    def latent_of(self, probs):
        """E[z0 | distribution over buckets] -- linear in probs, so a soft HL-Gauss label
        maps to its exact conditional-mean latent."""
        return probs @ self.bin_latents

    def cond_tokens(self, critic_feat, horizons=None):
        """(B,H) trunk feature -> (T,H) token conditioning, T = B * len(horizons)."""
        h = self.h_embed.weight if horizons is None else self.h_embed.weight[list(horizons)]
        return (critic_feat.unsqueeze(1) + h.unsqueeze(0)).reshape(-1, self.H)

    # ---- one denoise ------------------------------------------------------------------
    def denoise(self, cond, zt, sigma, ids):
        """EDM-preconditioned denoise: D_theta(zt; cond, sigma) -> the block's estimate of
        the clean value latent. zt (G,T,H); sigma (G,1,1) or (G,T,1); ids (G,) or None."""
        sd = self.sigma_data
        s2 = sigma**2
        c_skip = sd**2 / (s2 + sd**2)
        c_out = sigma * sd / (s2 + sd**2).sqrt()
        c_in = 1.0 / (s2 + sd**2).sqrt()
        c_noise = 0.25 * sigma.log()
        t_emb = self.t_embed(c_noise.expand(zt.shape[0], zt.shape[1], 1).reshape(-1))
        t_emb = t_emb.view(zt.shape[0], zt.shape[1], self.H)
        F_out = self.blocks(c_in * zt, cond, t_emb, ids)
        return c_skip * zt + c_out * F_out

    # ---- inference --------------------------------------------------------------------
    def logits(self, critic_feat, horizons=None):
        """Value-bucket logits, (B, T, num_bins). The ONLY value readout: both regimes go
        through here, so the rollout, the bootstrap and the regression target always see
        the network as it is actually trained."""
        cond = self.cond_tokens(critic_feat, horizons)
        n_tok = cond.shape[0] // critic_feat.shape[0]
        out = self._ode_logits(cond) if self.dblock else self._e2e_logits(cond)
        return out.view(critic_feat.shape[0], n_tok, self.num_bins)

    def _e2e_logits(self, cond):
        T = cond.shape[0]
        t_emb = self.t_embed(torch.zeros(T, device=cond.device)).unsqueeze(0)   # sigma == 1
        z = self.z_start.expand(T, self.H).unsqueeze(0)
        ids = torch.zeros(1, dtype=torch.long, device=cond.device)
        for b in range(self.num_blocks):
            z = z + self.blocks(z, cond, t_emb, ids + b)
        return self.decoder(z.squeeze(0))

    def _ode_logits(self, cond):
        """Euler solve of the probability-flow ODE from pure noise down the sigma ladder,
        visiting the block that owns each sigma (reference diffusion_step, model.py:274-291):
        Euler over ladder[:-1], then one final denoise at the smallest sigma, decoded.

        The clean-latent estimate fed to the Euler step is the DECODED-AND-REPROJECTED
        latent lat(softmax(decoder(D))), not D itself. This is the reference's
        `denoised = probs @ E` and it is load-bearing here: the training loss is a CE through
        a free decoder, and softmax is invariant to inflating ||D|| (it only sharpens the
        logits), so NOTHING anchors ||D|| to ||z0||. Measured after a 4k-step supervised
        probe at latent_norm 1.0: ||z0|| = 0.88 but ||D|| = 69 / 109 / 127 down the ladder
        with cos(D, z0) = 0.44 / 0.48 / 0.50, so blocks 1 and 2 were being handed states 25
        to 100x out of the distribution they train on. The projection restores norm 0.88 /
        0.91 / 0.92 at cos 1.000 / 0.996 / 0.991 -- lat(p) is a convex combination of unit
        rows, so the state cannot leave the latent manifold's scale by construction. The
        FINAL readout is still decoder(D): the projection is only how the ODE state is
        carried, never how the value is decoded."""
        z = (self.z_seed * self.sigma_max).expand(cond.shape[0], self.H).unsqueeze(0)
        one = torch.ones(1, 1, 1, device=cond.device)
        for i, sigma in enumerate(self.ladder):
            ids = torch.full((1,), self.block_for_sigma(sigma), dtype=torch.long, device=cond.device)
            D = self.denoise(cond, z, one * sigma, ids)
            logits = self.decoder(D)
            if i + 1 == len(self.ladder):
                return logits.squeeze(0)
            denoised = self.latent_of(torch.softmax(logits, dim=-1))
            dt = self.ladder[i + 1] - sigma
            z = z + dt * (z - denoised) / sigma          # Euler on d = (z - denoised)/sigma
        raise AssertionError("empty sigma ladder")

    # ---- training ---------------------------------------------------------------------
    def block_losses(self, critic_feat, target_probs, mask, block_ids):
        """Per-block DiffusionBlocks losses, ALL BLOCKS IN ONE BATCHED CALL.

        target_probs (B,M,N) HL-Gauss labels, mask (B,M) valid horizons -- the SAME labels
        and mask the end-to-end control regresses. Returns (G,) losses, one per block in
        block_ids; block b's loss touches ONLY block b's parameters (plus the shared
        conditioning and decoder, which the paper also trains from every block)."""
        cond = self.cond_tokens(critic_feat)                            # (T,H), T = B*M
        z0 = self.latent_of(target_probs).reshape(1, -1, self.H)        # (1,T,H)
        G, T = block_ids.shape[0], cond.shape[0]
        sigma = self.sample_sigma(block_ids, (G, T, 1), cond.device)
        zt = z0 + sigma * torch.randn(G, T, self.H, device=cond.device)
        D = self.denoise(cond, zt, sigma, block_ids)
        logp = torch.log_softmax(self.decoder(D), dim=-1)               # (G,T,N)
        ce = -(target_probs.reshape(1, T, -1) * logp).sum(-1)           # (G,T)
        w = self.edm_weight(sigma.squeeze(-1)) * mask.reshape(1, T)     # (G,T)
        if self.weight_normalize:
            # mean 1 over the VALID tokens WITHIN each block, so no block's loss is
            # up-weighted merely for owning a harder sigma bin.
            w = w / (w.sum(1, keepdim=True) / mask.sum().clamp_min(1.0))
        m = mask.reshape(1, T)
        ce_probe = (ce * m).sum(1) / mask.sum().clamp_min(1.0)          # unweighted, per block
        return (ce * w).view(G, -1, self.mtp).sum(-1).mean(-1), ce_probe.detach()

    def e2e_loss(self, critic_feat, target_probs, mask):
        """END-TO-END control loss: the base's masked HL-Gauss CE on the top-of-stack
        readout, backpropped through all B composed blocks. Same labels, same mask and
        same reduction as block_losses, so the two regimes are directly comparable."""
        logp = torch.log_softmax(self.logits(critic_feat), dim=-1)
        return (-(target_probs * logp).sum(-1) * mask).sum(-1).mean()


class Agent(nn.Module):
    def __init__(self, envs, args, hl_support):
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
        # The value head. The base's single bias-free linear readout Linear(H, mtp*num_bins)
        # is replaced by a DEPTH-`dblock_num_blocks` block stack over an MTP-token latent,
        # trained either block-locally as DiffusionBlocks denoisers (args.dblock) or
        # end-to-end as the residual stack they convert (the control). Both start at
        # V(s) == 0 exactly, like the zero-init head they replace.
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic = ValueNet(
            H,
            args.num_bins,
            args.critic_mtp_horizon,
            hl_support.coord_support,
            args.v_max,
            args,
        )
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

    def value_logits(self, x, horizons=(0,)):
        # Value-bucket logits (B, len(horizons), num_bins). horizons=(0,) => V(s_t) only,
        # which is all the rollout and the bootstrap need; horizons=None => all MTP heads.
        _, critic_feat = self._trunks(x)
        return self.critic.logits(critic_feat, horizons)

    def get_action_and_feat(self, x, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        # Returns the critic CONDITIONING feature instead of value logits: the value head
        # is now a denoiser stack whose training objective needs the target latent, and
        # whose inference needs an ODE solve, so both are driven explicitly by the caller.
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
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
        return action, z, log_prob, entropy, critic_feat

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
        return list(trunk.parameters()) + list(self.critic.parameters())


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

    sigma_floor = args.sigma_floor_bins * bin_width

    agent = Agent(envs, args, hl_support).to(device)
    all_block_ids = torch.arange(agent.critic.num_blocks, device=device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

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

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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
                action, z, logprob, ent, critic_feat = agent.get_action_and_feat(next_obs)
                p = torch.softmax(agent.critic.logits(critic_feat, (0,))[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = hl_support.probs_to_scalar(p)
            actions[step] = action
            latent_zs[step] = z
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
            # Bootstrap V(s') for all T*B transitions through the SAME value readout the
            # rollout used, chunked so the (chunk, num_bins) decode stays small.
            flat_next = next_obses.reshape((-1,) + envs.single_observation_space.shape)
            boot_probs = []
            for c0 in range(0, flat_next.shape[0], VALUE_CHUNK):
                boot_probs.append(
                    torch.softmax(agent.value_logits(flat_next[c0 : c0 + VALUE_CHUNK])[:, 0], dim=-1)
                )
            next_transition_values = hl_support.probs_to_scalar(torch.cat(boot_probs)).reshape(
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
                _, _, boot_logprob, _, _ = agent.get_action_and_feat(next_obses[-1])
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

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        dblock_ce = [float("nan")] * agent.critic.num_blocks
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, mb_critic_feat = agent.get_action_and_feat(
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

                # VALUE LOSS -- the A/B. Same labels, same mask, same support, same
                # reduction in both regimes.
                #   dblock: each block optimizes the HL-Gauss CE on ITS OWN one-step EDM
                #     denoise of the noised label latent, at a sigma from its own equal-mass
                #     interval. The blocks are siblings, never composed, so their losses
                #     share only the conditioning trunk and the decoder -- no gradient
                #     crosses a block boundary and no cross-block activation is retained.
                #     v_loss is the MEAN over blocks: each block's own gradient is then the
                #     one it would get training alone up to a constant Adam absorbs, while
                #     the shared parts see a loss of the same magnitude as the control's.
                #   no-dblock: one CE on the top-of-stack readout, backpropped through all
                #     B composed blocks.
                target_probs_mb = b_target_probs[mb_inds].to(device=device, non_blocking=True)
                value_mask = b_target_mask[mb_inds].to(
                    device=device, dtype=target_probs_mb.dtype, non_blocking=True
                )
                if args.dblock:
                    block_ids = (
                        all_block_ids
                        if args.dblock_train_all_blocks
                        else all_block_ids[random.randrange(agent.critic.num_blocks)].view(1)
                    )
                    losses, block_ce = agent.critic.block_losses(
                        mb_critic_feat, target_probs_mb, value_mask, block_ids
                    )
                    v_loss = losses.mean()
                    for b, ce in zip(block_ids.tolist(), block_ce.tolist()):
                        dblock_ce[b] = ce
                else:
                    v_loss = agent.critic.e2e_loss(mb_critic_feat, target_probs_mb, value_mask)

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
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the stashed
                    # clip_critic(d vl / d trunk). The denoiser blocks and the decoder get
                    # the value gradient only (and each block only its own loss's).
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
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
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        # DiffusionBlocks probes. dblock_ce_b{b} is the UNWEIGHTED masked CE of block b's
        # one-step denoise. Block 0 owns the LARGEST sigmas, i.e. "predict the value
        # distribution from the observation alone" == exactly the base critic's job; each
        # later block refines a cleaner latent, so the series should be monotonically
        # DECREASING in b (measured 1.47 > 1.42 > 1.40 on a supervised probe). pred_entropy
        # is the entropy of the ODE-solved rollout distribution: ~log(511)=6.24 means the
        # solve is not committing to a value; pred_edge_mass flags drift off support.
        if args.dblock:
            for b in range(agent.critic.num_blocks):
                writer.add_scalar(f"losses/dblock_ce_b{b}", dblock_ce[b], global_step)
        writer.add_scalar(
            "debug/dblock_pred_entropy",
            (-(value_probs * value_probs.clamp_min(1e-20).log()).sum(-1)).mean().item(),
            global_step,
        )
        writer.add_scalar(
            "debug/dblock_pred_edge_mass",
            (value_probs[..., 0] + value_probs[..., -1]).mean().item(),
            global_step,
        )
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

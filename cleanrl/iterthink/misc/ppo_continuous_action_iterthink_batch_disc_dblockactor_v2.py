# PPO + IterThink + DISCRETIZED-ACTION DIFFUSIONBLOCKS POLICY v2 (TRUST REGION + REF GEOMETRY).
# =====================================================================================
# DiffusionBlocks (ICLR 2026, arXiv:2506.14202) applied to the ACTOR, with a built-in
# control: --dblock / --no-dblock select block-local vs end-to-end training of the SAME
# policy head, so "is DiffusionBlocks helping" is one flag on one codebase.
#
# v2 vs v1: v1's e2e arm LEARNED but topped out ~5x below the base (1011 @1.7M vs the base's
# ~5000 @2M). The cause was NOT the head. It was two leashes inherited from a chassis tuned
# for a BETA policy, both measured binding on the live run:
#   (1) target_kl = 0.03 pinned approx_kl at 0.030-0.034 EVERY iteration while the
#       mirror-descent objective asked for KL(q||pi_old) = 0.24 nats. Because the early stop
#       fires between epochs, it was also truncating the update after ~1-2 of 10 epochs.
#       Entropy fell 0.036 nats/iter against the 0.143 the operator predicts (/tmp/md_rate.py),
#       so sharpening a 6x15-bin categorical enough to be useful (~13 nats) needed 433
#       iterations = 14M steps against an 8M budget. It was never going to finish.
#       v2: the leash is tpo_kl_slack * KL(q||pi_old), computed per minibatch. q is ALREADY
#       the exact maximizer of Ahat - (1/eta)KL(pi||pi_old), so it already IS a trust region;
#       a second fixed leash below it just double-counts, and the tighter one won.
#   (2) actor_grad_clip = 0.25 clipped 59.6% of actor steps (median norm 0.3955, p90 1.1308).
#       v2 uses 1.0, the REFERENCE's own global clip (main.py:66). critic_grad_clip stays at
#       0.25: that head is unchanged from the base, which scores 8455@8M with it saturated.
# A 15-bin categorical over 6 dims carries 90 numbers per state where a Beta carries 12, and
# it must traverse far more nats to concentrate. Both leashes were sized in absolute units to
# the wrong policy family. This is the same class of bug as v1's eta/percentile-scale
# coupling: a hyperparameter whose correct value silently depended on the parameterization.
#
# v2 also takes two REFERENCE-GROUNDED geometry fixes, which touch ONLY the --dblock arm
# (verified: with --no-dblock the acting path leaves codebook grad None), so the A/B stays
# attributable -- the e2e arm changes through the two leashes and nothing else:
#   (3) UNIT-NORM TARGET. The reference's diffused variable is always the L2-NORMALIZED
#       embedding of a point mass, ||y|| == 1 exactly, renormalized at every use
#       (model.py:143-155). v1 diffused the raw convex combination q @ E, whose norm shrinks
#       with q's entropy. MEASURED: at the near-uniform q that PPO starts from, ||q@E|| was
#       0.105 -- against sigma_data = 0.5 and a ladder reaching sigma = 80. The target was
#       buried under its own noise precisely when training begins. Now 1.000 at every
#       entropy (0.105/0.165/0.284/0.508/0.734/0.914 -> 1.0), still injective in q
#       (max |q - lstsq(z0)| = 7.7e-07).
#   (4) LEARNED TABLE. The reference's table is a trained nn.Embedding (vit.py:151) that can
#       reshape its geometry; v1 froze a hand-built Fourier codebook. v2 keeps that
#       construction as the INIT and lets every block's loss move it, as the reference does.
#
# FINAL VERDICT ON DIFFUSIONBLOCKS-ON-THE-ACTOR: ABANDONED AFTER THREE PORTS.
# 8M HalfCheetah, seed 1, against the base's 8455:
#     v2 --no-dblock  (leash fix only)   4138.5     v2 --no-dblock (leash + clip 1.0)  3586.6
#     v2 --dblock                        -315.5  (random floor, cancelled at 6.5M)
# The e2e control is 2x below the base and the block-local arm never leaves the floor. Beyond
# the two mis-sized leashes above, this head has a THIRD defect that is structural for RL and
# is the reason the first 1M steps score 98 against the base's 3078:
#   ZERO-INIT DECODER = ZERO GRADIENT INTO EVERYTHING. Measured at init, --no-dblock, with a
#   real CE target: gradient into the trunk feature 0.000e+00, into the block stack's in_w
#   0.000e+00, into the decoder 6.3e-03. dL/dz = W^T(p - q) and W == 0, so ONLY the decoder's
#   240 weights can move; the head and the shared trunk receive EXACTLY nothing until W grows.
#   The reference zero-inits its classifier (vit.py:684-685) and is right to: it does
#   supervised CE on a FIXED dataset, where the label embedding still feeds the loss through
#   c_skip. In RL the policy GENERATES its own data, so a policy pinned at uniform emits
#   uninformative rollouts while the trunk gets no actor gradient to shape features from --
#   a self-reinforcing cold start. The base's Beta head uses std=0.01, small but NONZERO, and
#   moves from step 1. The same zero-init is harmless in the VALUE port (which wins, below)
#   because a critic's target is the observed return: informative from step 1, and independent
#   of the critic's own output.
# Three independent ports, three refutations, each with a measured mechanism rather than a
# guess: (1) Beta actor -- no per-example target exists, KL(q||pi_old) 0.014 nats at the
# tuned eta; (2) this file's block-local arm -- capture 46.9% vs the e2e control's 99.7% even
# after restoring w(sigma) and unit-norm targets; (3) this file's e2e arm -- 2x below base.
#
# WHAT ACTUALLY WON, AND IT IS NOT DIFFUSION. The same port on the VALUE head, trained
# END-TO-END, at 8M: 9809.7 against the base's 8454.7 (+16%), ahead at every matched step
# after 1M (2983/5257/8112/9138 vs 3079/5012/6716/7548 @1M/2M/4M/6M). But read what that arm
# computes: with --no-dblock the readout is z_b = z_{b-1} + f_b(z_{b-1}, cond, t(0)) -- ONE
# fixed time embedding, no sigma draw, no noise, no ODE. At constant t the adaLN modulation
# (1+scale(t), shift(t)) is a CONSTANT affine, so that head is provably just a residual MLP
# with per-block fixed affines. The +16% is a DEEPER VALUE HEAD beating a single Linear, and
# the diffusion apparatus around it is inert. The honest experiment is therefore the depth
# curve (base=0 hidden layers -> 3 -> 6 -> 12), not more of this method.
#
# THE PAPER. A residual stack z_l = z_{l-1} + f_l(z_{l-1}) is a discretization of an ODE.
# Reinterpret that ODE as the EDM probability-flow ODE on a LATENT OF THE TARGET, and each
# block becomes a denoiser on one interval of the noise level. Denoisers at different noise
# levels have INDEPENDENT training problems (score matching factorizes over sigma), so each
# block trains ALONE on the FULL task loss applied to its own one-step denoise -- never on
# another block's output. No backward through depth, no cross-block activations.
#
# WHY THIS FILE EXISTS: THE FIRST TWO PORTS FAILED, AND THE PAPER SAYS WHY.
# Eq. (6) L_b = E[w(sigma)*Loss(f_{theta_b|sigma}(x, y+sigma*eps), y)] needs a per-example
# target y that is both the DIFFUSED variable and the label. Measured on this machine:
#   * PPO's actor has no y at all: only (s, a_sampled, Ahat). Port impossible.
#   * TPO-MD's actor HAS one -- q over K=8 probed candidates -- and the port ran, but its
#     block-local arm scored 222% of its own e2e control at 500k (2202 vs 3707) with a FLAT
#     per-block CE series (1.885/1.858/1.863 instead of decreasing). Diagnosis: the target
#     was a distribution over 8 CONTINUOUS candidate actions, so the latent was an RFF
#     characteristic function and the readout had to invert it into 2 Beta concentrations.
#     No linear map does that, so the noised latent carried nothing, every block collapsed
#     to the same single-shot regressor, and B blocks became B copies of the base head.
# The paper never does this. Its rule, across all six of its settings: DIFFUSE THE THING THE
# LOSS DECODES, via identity or ONE linear head over a FINITE support. (Verified against the
# reference impl: `denoised = probs @ E`, model.py:172-179.)
#
# WHAT THIS FILE CHANGES: THE ACTION SPACE, SO THE RULE HOLDS.
# actor_dist="disc": a per-dimension CATEGORICAL over M ORDERED bins of [-1,1], replacing
# the Beta. That single change buys every property the paper needs:
#   (a) FINITE SUPPORT: the target is an (A,M) probability table, exactly the paper's label
#       branch. Latent z0 = q @ E over a fixed unit-norm codebook E (M,d), d=16 > M=15, so
#       rank(E)=M and the encoding is LOSSLESS (measured). Decode = ONE Linear(d,M).
#   (b) EXACT log pi: the native sample IS the bin index, so PPO's ratio needs no Jacobian
#       and there is no intractable likelihood (the reason diffusion POLICIES can't do PPO).
#   (c) CLOSED-FORM per-example target, no probing: the mirror-descent step
#         q_j(m) ~ pi_old,j(m) * exp(eta * Ahat * 1[m == bin(a_j)])
#       is the exact maximizer of Ahat - (1/eta)KL(pi||pi_old) under PPO's OWN single-sample
#       advantage. It is computable from (pi_old, a, Ahat) -- data PPO already has.
#   (d) CAPACITY: Table 8's best configs are >=4 layers/block; my failed ports ran 1-2.
#       dblock_layers_per_block=4 here.
#
# MEASURED (probes: /tmp/disc_ceiling.py, /tmp/disc_verify.py, /tmp/eta_check.py):
#   * Latent SNR DEPENDS ENTIRELY ON THE TARGET'S SHARPNESS, and the first version of this
#     header quoted the wrong distribution. Per-coord SNR for b0/b1/b2, measured:
#         synthetic near-one-hot q           0.309 / 0.825 / 2.247   (d' 3.0 / 8.1 / 22.0)
#         q at eta=1 with the RAW percentile-scaled advantage
#                                            0.006 / 0.015 / 0.042  (d' 0.05 / 0.14 / 0.38)
#         q at eta=1 with the STANDARDIZED advantage (what this file now does)
#                                            0.041 / 0.111 / 0.301
#     The middle row is a dead arm: a diffused target 0.014 nats from uniform carries less
#     signal than the noise the method deliberately adds. See THE ETA COUPLING below.
#   * Decode ceiling: softmax(linear(z0)) cannot exactly hit a SOFT q (softmax^-1 is not
#     linear). Best-case CE - H(q) = 0.005/0.084/0.198/0.225 nats as q sharpens. A fully
#     FREE linear map scores the same (0.1984 vs 0.1976), so this is intrinsic to
#     softmax(linear), not to the codebook -- and BOTH arms share it, so the A/B is clean.
#   * Neutral init: |logits| == 0 exactly in both regimes => uniform policy, no head prior.
#   * Gradient isolation: block b's loss touches only block b's slice, every inner layer of
#     the owning block gets gradient, the shared decoder trains from every block; the e2e
#     control's single backward reaches all (block, layer) pairs. Independently re-verified:
#     with only block 0's loss, per-block in_w grad norms are [8.18, 0.0, 0.0].
#   * EDM schedule verified bit-for-bit against the reference: block_sigma_edges matches
#     get_block_sigmas to 2.8e-17, inference_sigmas matches get_discrete_sigmas to 1.7e-16,
#     block_for_sigma agrees with estimate_target_layer on 0/2000 mismatches, and all four
#     preconditioners match to 8 decimals.
#
# DELIBERATE DEVIATIONS FROM THE REFERENCE IMPL.
#   1. ALL BLOCKS PER STEP (dblock_train_all_blocks=True, off switch provided). The paper
#      samples ONE block per optimizer step because memory is its whole point; here memory
#      is free and sample efficiency is everything. Gradients stay strictly block-local
#      either way -- blocks are siblings, never composed -- so this is the same estimator,
#      more samples per step. Every parameter carries a leading (block, layer) axis and
#      every op is an einsum over the block axis: all blocks train in ONE batched call.
#   2. COMMON RANDOM NUMBER for the ODE (one frozen z_seed), so pi(.|s) is a DETERMINISTIC
#      function of s. The rollout stores log pi and the update re-evaluates it; a resampled
#      start would make the two disagree and inject noise into the PPO ratio.
#   3. Block 0 owns the LARGEST sigmas, so losses/dblock_actor_ce_b0 == "predict the target
#      policy from the observation alone" == exactly the base head's job, and the series
#      should be monotonically DECREASING in b. FLAT means the latent is again carrying
#      nothing and the blocks have degenerated into B independent single-shot predictors --
#      the precise failure signature of the Beta port. This is the load-bearing probe.
#   4. The final ODE rung returns its RAW estimate; reprojection through the codebook exists
#      only to put INTERMEDIATE states back on the target manifold before the next Euler step.
#
# HYPOTHESIS. Discretizing the action makes the paper's structure apply exactly rather than
# approximately, so the per-block CE series separates and block-local training at last
# matches or beats its own end-to-end control -- at flat memory in B.
#
# THE ETA COUPLING, and a verdict I had to retract. The first live A/B produced a DEAD
# treatment arm: losses/dblock_actor_ce_b* pinned at 16.24829 == 6*ln(15) to 7 significant
# figures, policy entropy 16.2483, acting policy exactly state-independent, episodic return at
# the random floor after 848k steps (job 3072, caught by scripts/autocull.py's `dead` verdict
# at progress -0.04). I first read that as the method failing and wrote a REFUTED verdict here,
# arguing that a deterministic target makes p(z0|s) a point mass, so the optimal denoiser
# ignores z_t and every block collapses to one regressor. That argument is real but it was NOT
# what killed the arm. Two of my own bugs were:
#   1. THE STEP SIZE INHERITED THE ENVIRONMENT'S RETURN SCALE. q was built from the
#      percentile-scaled advantage (divided by S = max(1, P95-P5) of raw returns, measured
#      16.3-26.0 on HalfCheetah and drifting ~60% over training), so eta=1 bumped a logit by
#      ~0.3 and left q 0.014 nats from uniform out of 16.25. The e2e arm still captured that
#      sliver -- it optimizes the acting policy directly -- but at per-coord SNR 0.006 the
#      DIFFUSED signal sat below the noise the method adds, so block-local training got
#      nothing. Fixed: the advantage entering q is standardized, making eta dimensionless.
#   2. THE EDM WEIGHT WAS DEFINED AND NEVER APPLIED. w(sigma) == 1/c_out^2 is exactly the
#      attenuation of a block's output before the loss, so unweighted training starved the
#      small-sigma blocks -- 0.61x and 0.25x of block 0's per-unit-output gradient -- and those
#      own the FINAL ladder rungs that decide the answer. Fixed: w normalized within each
#      block, which keeps the compensating shape without letting the raw [24, 2.5e5] span
#      reach a cross-entropy.
#
# WHAT THE CORRECTED MEASUREMENT SAYS (eta_check.py; fixed (cond, q), identical capacity and
# init in both arms, 800 Adam steps). "capture" = fraction of the target's information the
# ACTING policy recovers, (uniform_CE - acting_CE) / (uniform_CE - H(q)):
#     eta   info in q     dblock capture    e2e capture
#     1.0    0.24 nats        46.9%            99.7%
#     3.0    2.26 nats        69.8%            99.6%
#    10.0    6.01 nats        85.3%            99.5%
# Before the two fixes the same measurement gave dblock 0% at the default and 16% at eta=20.
# So block-local training WORKS here and my point-mass verdict was wrong; what survives of it
# is a quantitative version: block-local training needs the TARGET ITSELF to be informative
# relative to the noise it adds, while end-to-end training only needs the target's gradient to
# point the right way. A trust-region target is deliberately close to the current policy, so
# the two requirements pull against each other -- which is why capture rises monotonically
# with eta while e2e is flat at ~99.6% throughout. That tension, not a collapse, is the real
# finding, and it is what the 8M A/B is now testing at a matched eta in both arms.
#
# The other artifact from this line is the EDM value head trained END-TO-END (+21% @4M over
# the linear-head base): ppo_continuous_action_iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_dblockcritic_v1.py.
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport


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
    norm_adv: bool = False           # d3percnorm: NO per-minibatch standardization ("no advantage norm").
    # --- Percentile advantage normalization (the sole advantage scaler) ---
    ret_percnorm: bool = True        # scale policy advantage by S = max(floor, P95-P5) of returns
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
    # KL early-stop leash. MEASURED BINDING AND MISSIZED in v1: the iterthink line tuned
    # 0.03 for a BETA head (2 parameters per action dim, 12 numbers), and the same absolute
    # nat budget throttles this file's 15-bin categorical (90 numbers). v1's live run sat at
    # approx_kl 0.030-0.034 every iteration -- the leash, not the objective, set the step --
    # while its own mirror-descent target asked for KL(q||pi_old) = 0.24 nats. Entropy fell
    # 0.036 nats/iteration against the 0.143 the operator predicts, so reaching a usable
    # policy (~13 nats of sharpening) needed 433 iterations = 14M steps against an 8M budget.
    # v2: the leash is a SLACK MULTIPLE of the target's OWN KL, computed per minibatch. The
    # mirror-descent target q = softmax(log pi_old + eta*Ahat*onehot) is already the exact
    # maximizer of Ahat - (1/eta)KL(pi||pi_old), i.e. ALREADY a trust region; a second, fixed
    # leash below it just double-counts and the tighter one wins. Slack > 1 lets the update
    # actually REACH q while still catching a runaway epoch. Self-calibrating in eta and in
    # the policy's parameterization, so there is no per-head number left to mistune.
    target_kl: Optional[float] = None  # absolute leash; None => use tpo_kl_slack (mdce only)
    tpo_kl_slack: float = 2.0        # early-stop at this multiple of KL(q||pi_old)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    # MEASURED BINDING in v1 (same Beta-vs-categorical mismatch as target_kl): actor grad
    # norm median 0.3955, p90 1.1308, CLIPPED ON 59.6% of steps at 0.25. A 15-bin categorical
    # CE has a larger natural gradient than two Beta concentrations, so the Beta-tuned 0.25
    # attenuated most policy steps on top of the KL leash. 1.0 is the REFERENCE's own value
    # (DiffusionBlocks main.py:66, gradient_clip_val=1.0) and passes ~90% of measured steps.
    # critic_grad_clip is left at 0.25: the critic head is UNCHANGED from the base, which
    # scores 8455@8M with it clipping 100% of the time, so that is its normal operating point.
    # MEASURED, AND THE LOOSENING WAS WRONG: an isolation arm (leash fix only, clip left at
    # 0.25) scored 4138.5 @8M against 3586.6 for leash+clip=1.0, so raising the clip to the
    # reference's 1.0 COST 13%. The reason is visible in the trust region: approx_kl hit 2.25
    # at 1M against a target radius of 0.34, i.e. the update overshoots its own mirror-descent
    # target ~6x, and the tight clip was doing real work damping that overshoot. The
    # reference's 1.0 is calibrated for supervised CE on a fixed dataset, not for a moving
    # RL target whose per-iteration step is already bounded by a trust region.
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
    norm_adv_scope: str = "minibatch"

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "disc"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    # ---- DISCRETIZED DIFFUSION ACTOR (this file) ------------------------------------------
    # actor_dist is forced to "disc": a per-dimension categorical over num_action_bins ORDERED
    # bins. This is the parameterization that makes DiffusionBlocks well-posed on a policy at
    # all (finite support => the paper's shared-linear-codebook readout) while keeping
    # log pi(a|s) exact so the clipped PPO ratio is untouched. See DiscActorHead.
    num_action_bins: int = 15         # bins per action dimension; odd => a center bin at 0
    dblock_latent_dim: int = 16       # per-dimension codebook width; state = act_dim * this
    # ACTOR LOSS. "mdce" is the closed-form mirror-descent cross-entropy CE(q, pi) that
    # supplies the per-example target y DiffusionBlocks requires; "ppoclip" is the base's
    # clipped surrogate, kept so the loss change can be isolated from the head change.
    # ppoclip is only valid with --no-dblock (block-local training needs a target to noise).
    actor_loss: str = "mdce"
    tpo_eta: float = 1.0              # mirror-descent step size in q = pi_old * exp(eta*Ahat)
    dblock: bool = True               # True: block-local (the method). False: end-to-end control.
    dblock_num_blocks: int = 3        # B. The ODE ladder length is dblock_inference_steps, NOT B.
    # LAYERS PER BLOCK is the capacity axis and the paper's Table 8 (CIFAR-10, L=12 fixed) is
    # unambiguous about it: B=2 (6 layers/block) FID 35.47 < B=1 end-to-end 39.83 < B=3 (4)
    # 38.03 << B=4 (3) 45.43 << B=6 (2) 53.32. Its own configs are 4 layers/block (ViT, DiT,
    # MD4), 8 (DiT-L/2) and 3 (the B=4 AR LM). Earlier ports here ran 1 and 2 -- the two worst
    # rows -- so their block-local deficit was partly a capacity artifact. Do not go below 4.
    dblock_layers_per_block: int = 4
    dblock_mult: int = 2              # sub-layer MLP width = mult * state width
    dblock_sigma_min: float = 0.002   # EDM support endpoints (paper values)
    dblock_sigma_max: float = 80.0
    dblock_sigma_data: float = 0.5    # EDM preconditioning scale (paper value)
    dblock_latent_norm: float = 1.0   # codebook row norm. The paper L2-normalizes its label
                                      # embeddings and leaves sigma_data=0.5, i.e. it runs a
                                      # deliberate norm/scale mismatch; kept faithful here.
    dblock_p_mean: float = -1.2       # EDM training sigma ~ lognormal(p_mean, p_std)
    dblock_p_std: float = 1.2
    dblock_gamma: float = 0.05        # widen each block's training bin by this fraction of
                                      # its log-range on both sides (paper default)
    dblock_inference_steps: int = 6   # Euler ladder length. With steps == B the last rung sits
                                      # at sigma_min where c_out == 0.002, which made the final
                                      # block contribute 0.32% of the decoded latent (measured);
                                      # 6 gives 2 visits per block.
    dblock_train_all_blocks: bool = True   # False => paper-faithful ONE block per minibatch

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


# carries a leading (block, layer) axis so all blocks train in ONE batched call.
# ---------------------------------------------------------------------------------------
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

    def __init__(self, num_blocks, n_layer, dim, mult, cond_dim=None):
        """dim = width of the DIFFUSION STATE; cond_dim = width of the conditioning feature
        and of the time embedding. They are separate here because the diffused variable is
        the action-bin latent (act_dim * latent_dim) while the conditioning is the trunk
        feature (hidden), and the paper's dimension-matching requirement is on the BLOCK's
        input and output, not on the conditioning."""
        super().__init__()
        cond_dim = dim if cond_dim is None else cond_dim
        Hm = mult * dim
        self.num_blocks, self.n_layer, self.H, self.Hm = num_blocks, n_layer, dim, Hm
        self.cond_dim = cond_dim
        # Affine-free RMSNorm on both inputs (file convention): the trunk feature is
        # unnormalized and z is O(sigma_max) early in the solve, so the ReLU^2 needs both
        # scales bounded before the projection. Stateless, so one instance serves every layer.
        self.z_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.c_norm = nn.RMSNorm(cond_dim, elementwise_affine=False)
        self.in_w = nn.Parameter(torch.empty(num_blocks, n_layer, Hm, dim + cond_dim))
        self.in_b = nn.Parameter(torch.zeros(num_blocks, n_layer, Hm))
        self.ada_w = nn.Parameter(torch.zeros(num_blocks, n_layer, 2 * Hm, cond_dim))
        self.ada_b = nn.Parameter(torch.zeros(num_blocks, n_layer, 2 * Hm))
        self.out_w = nn.Parameter(torch.zeros(num_blocks, n_layer, dim, Hm))
        self.out_b = nn.Parameter(torch.zeros(num_blocks, n_layer, dim))
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


class DiscActorHead(nn.Module):
    """Per-dimension CATEGORICAL policy whose logits are read out of an EDM probability-flow
    ODE over the TARGET POLICY's own latent. ONE architecture, TWO training regimes (args.dblock).

    WHY DISCRETE -- this is the whole point. DiffusionBlocks Eq. (6) noises a per-example
    target y and trains every block to predict THAT SAME y. Across all six settings the paper
    evaluates, y is decoded either by the IDENTITY (images, VAE latents) or by ONE shared
    linear head over a FINITE support (class labels, vocab tokens), with inference literally
    reconstructing `denoised = softmax(logits) @ E`, a convex combination of codebook rows
    (reference model.py:268-274). A continuous Beta/Gaussian PARAMETER head has no such
    support. An earlier port of this method to a Beta actor diffused a random-Fourier
    characteristic function of the target action distribution, and since no linear map inverts
    that into (alpha, beta), the latent channel carried nothing: every block converged to the
    SAME cross-entropy (measured 1.857 / 1.858 / 1.863) even though block 2's input had 7x the
    per-coordinate SNR of block 0's, and return collapsed to 267 against the end-to-end
    control's 3708. Discretizing each action dimension into `num_action_bins` ORDERED bins
    restores exactly the paper's structure. It also makes log pi(a|s) EXACT, so the base's
    clipped PPO ratio survives unchanged -- no diffusion-policy likelihood approximation.

    THE LATENT is the value head's construction, which is also the paper's label-embedding
    table: a FROZEN Fourier codebook of the bin centers, shared by every action dimension,
        e_m  = [sin(pi f_k u_m), cos(pi f_k u_m)]_k ,  u_m = bin center in [-1,1], f geometric
        E    = dblock_latent_norm * normalize(e)              (num_action_bins, latent_dim)
        z0   = concat_j ( q_j @ E )                           (act_dim * latent_dim,)
    LINEAR in q, so a soft target maps to its exact conditional-mean latent, and because the
    support is ORDERED, neighbouring bins get neighbouring latents -- the denoiser refines the
    target coarse-frequency-first instead of treating bins as unrelated classes. Rows have
    identical norm, so ONE number (dblock_latent_norm, left at the paper's 1.0 against
    sigma_data 0.5) sets how much of the answer a block can READ off its noised input.

    THE TARGET q, IN CLOSED FORM -- what made a PPO actor look ill-posed, and why it is not.
    PPO stores only (s, a, Ahat): no per-example action target, hence no y to noise. But a
    DISCRETE policy has one exactly. The mirror-descent step
        q_j(m)  proportional to  pi_old,j(m) * exp(eta * Ahat * 1[m == bin(a_j)])
    is the closed-form maximizer of  Ahat - (1/eta) KL(pi || pi_old)  under PPO's own
    single-sample estimator (Ahat is known only at the action taken, zero elsewhere), so it
    needs no candidate probing, no extra environment interaction and no second critic. Its
    cross-entropy CE(q, pi) is a trust-region policy-improvement step (the MDPO/TPO family).
    Crucially q carries the advantage SIGN: with Ahat < 0 it moves mass AWAY from the action
    taken, so a block always denoises TOWARD an improved policy. Diffusing y = the action
    taken instead would ask a block to denoise toward an action the loss simultaneously
    pushes probability away from -- a sign conflict that gets worse the cleaner the input.

    args.dblock == False  (END-TO-END CONTROL: same parameters, same readout, one backward)
        z_0 = z_start; z_b = z_{b-1} + F_b(z_{b-1}, cond, t(0)); logits = decode(z_B)
    args.dblock == True   (DIFFUSIONBLOCKS)
        block b owns the b-th EQUAL-MASS interval of the EDM training lognormal and is trained
        ALONE on the full task loss of its own one-step denoise; blocks are siblings, never
        composed. ACTING is the Euler solve down the sigma ladder, and the state is
        REPROJECTED through the codebook every step (denoised = softmax(decode(D)) @ E,
        the reference's own inference rule), which keeps the ODE on the latent manifold
        without any auxiliary anchor term -- the decode is onto a finite support, so there is
        no gauge freedom for ||D|| to exploit.
    """

    def __init__(self, H, act_dim, args):
        super().__init__()
        self.H, self.act_dim = H, act_dim
        self.M, self.d = args.num_action_bins, args.dblock_latent_dim
        self.state_dim = act_dim * self.d
        self.dblock = args.dblock
        self.num_blocks = args.dblock_num_blocks
        self.sigma_data = args.dblock_sigma_data
        self.p_mean, self.p_std = args.dblock_p_mean, args.dblock_p_std
        self.latent_norm = args.dblock_latent_norm
        # ---- EDM schedule. Block 0 owns the LARGEST sigmas, so the descending inference
        # ladder visits 0, 1, ... in order and block 0's job is "predict the target policy
        # from the observation alone" == exactly the base linear head's job.
        self.edges = block_sigma_edges(
            self.num_blocks, args.dblock_sigma_min, args.dblock_sigma_max,
            self.p_mean, self.p_std,
        )
        self.ladder = inference_sigmas(
            args.dblock_inference_steps or self.num_blocks,
            args.dblock_sigma_min, args.dblock_sigma_max, self.p_mean, self.p_std,
        )
        lo_hi = []
        for b in range(self.num_blocks):
            lo, hi = self.edges[b], self.edges[b + 1]
            if args.dblock_gamma > 0.0:
                span = log(hi) - log(lo)
                lo = max(exp(log(lo) - args.dblock_gamma * span), args.dblock_sigma_min)
                hi = min(exp(log(hi) + args.dblock_gamma * span), args.dblock_sigma_max)
            lo_hi.append((lo, hi))
        self.block_bins = lo_hi[::-1]                      # index 0 == largest sigmas
        self.register_buffer("cdf_lo", torch.tensor(
            [_cdf_scalar((log(lo) - self.p_mean) / self.p_std) for lo, _ in self.block_bins]))
        self.register_buffer("cdf_hi", torch.tensor(
            [_cdf_scalar((log(hi) - self.p_mean) / self.p_std) for _, hi in self.block_bins]))
        # ---- ordered Fourier codebook over the bin centers, LEARNED (reference D3)
        # The reference's diffused variable is a LEARNED nn.Embedding table (vit.py:151, init
        # N(0,0.02^2)) trained by every block's loss, free to reshape its geometry to make
        # denoising easy. v1 froze this table at a hand-built Fourier construction, which
        # fixes the geometry to whatever I guessed. v2 keeps the Fourier construction as the
        # INIT -- it is a good one: smooth and injective in the bin coordinate by
        # construction, so adjacent bins start close and the ordering is respected -- and then
        # lets the loss move it, matching the reference.
        centers = torch.linspace(-1.0, 1.0, self.M)
        n_freq = self.d // 2
        freqs = torch.logspace(0.0, log(max(self.M / 2.0, 2.0)) / log(10.0), n_freq)
        ang = pi * freqs.unsqueeze(0) * centers.unsqueeze(1)                  # (M, n_freq)
        e = torch.cat([ang.sin(), ang.cos()], dim=-1)                         # (M, d)
        self.codebook_raw = nn.Parameter(F.normalize(e, dim=-1) * args.dblock_latent_norm)
        # d >= M only makes the encoding lossless POSSIBLE; check the table that was actually
        # built. Measured float32 rank: 15/15 at M=15,d=16 (smin 8.9e-02, cond 1.6e+01), but
        # 28/31 at M=31,d=32 (smin 2.7e-08, cond 6.7e+07) and 47/63 at M=63,d=64 -- there q is
        # NOT recoverable from z0 (max |q - pinv(E)(q@E)| = 1.8e-01 / 3.1e-01) and the denoise
        # target is unidentifiable, which is the failure this whole parameterization exists to
        # avoid. Cost is one SVD of an (M,d) matrix, once, at construction.
        rank = int(torch.linalg.matrix_rank(self.codebook))
        assert rank == self.M, (
            f"codebook float32 rank {rank} < num_action_bins {self.M}: q -> z0 is not "
            f"invertible, so the diffusion target is unidentifiable"
        )
        self.register_buffer("bin_centers", centers)
        self.t_embed = TimeEmbed(H)
        self.blocks = BlockStack(
            self.num_blocks, args.dblock_layers_per_block, self.state_dim,
            args.dblock_mult, cond_dim=H,
        )
        # Shared readout, applied per action dimension: latent_dim -> num_action_bins.
        # ZERO-init, so every logit is 0 at init and the policy is EXACTLY uniform over bins
        # for every state -- the discrete analog of the base's std=0.01 near-neutral heads.
        self.decoder = nn.Linear(self.d, self.M, bias=False)
        with torch.no_grad():
            self.decoder.weight.zero_()
        # Nonzero start for the e2e control: zero start + zero-init decoder + zero-init
        # out_proj is an all-zero fixed point with no gradient anywhere.
        self.z_start = nn.Parameter(torch.randn(1, self.state_dim) * 0.02)
        self.register_buffer("z_seed", torch.randn(1, self.state_dim))   # frozen CRN
        # Constant per-step ladder quantities (see _ode_state).
        self.register_buffer("ladder_cnoise", torch.tensor([0.25 * log(s) for s in self.ladder]))
        self.register_buffer("ladder_ids", torch.tensor(
            [self.block_for_sigma(s) for s in self.ladder], dtype=torch.long))
        self.ladder_precond, self.ladder_dt = [], []
        for i, s in enumerate(self.ladder):
            s2 = s * s + self.sigma_data**2
            self.ladder_precond.append(
                (self.sigma_data**2 / s2, s * self.sigma_data / sqrt(s2), 1.0 / sqrt(s2)))
            if i + 1 < len(self.ladder):
                self.ladder_dt.append((self.ladder[i + 1] - s) / s)

    # ---- latent / readout ---------------------------------------------------------------
    @property
    def codebook(self):
        """Row-normalized table. The reference normalizes its embedding at EVERY use
        (model.py:143-155 normalize_embeddings, called from get_embeds), so the geometry the
        loss sees is scale-free and the learned table cannot cheat by growing its rows."""
        return F.normalize(self.codebook_raw, dim=-1) * self.latent_norm

    def latent_of(self, q):
        """(N,A,M) distribution over bins -> (N,A*d) latent, UNIT-NORM PER ACTION DIM.

        The reference's diffused variable is always the unit-norm embedding of a POINT MASS
        (||y|| == 1 exactly, every example, every sigma). v1 diffused the raw convex
        combination q @ E instead, whose norm SHRINKS with q's entropy: with near-orthogonal
        rows a uniform q gives ||q@E|| ~ 1/sqrt(M) = 0.26, so early in training -- exactly
        when PPO's trust-region target IS near-uniform -- the diffusion target was a nearly
        zero vector while every EDM preconditioner and the whole sigma ladder are calibrated
        for an O(sigma_data)=O(0.5) signal. The target was buried under its own noise.
        Renormalizing puts the DIRECTION in charge of carrying q at a constant radius, which
        is the reference's geometry. Non-linear in q (v1's linearity note no longer holds),
        but still injective: q has M-1 free parameters and the unit sphere in the rank-M
        row space has M-1, and the rank assertion above keeps E full-rank.
        """
        z = q @ self.codebook                                      # (N,A,d)
        return (F.normalize(z, dim=-1) * self.latent_norm).flatten(-2)

    def logits_of(self, state):
        """(...,A*d) latent -> (...,A,M) per-dimension bin logits."""
        return self.decoder(state.unflatten(-1, (self.act_dim, self.d)))

    def reproject(self, state):
        """The reference's inference rule: decode to probabilities, then re-embed as a convex
        combination of codebook rows. Keeps the Euler state on the latent manifold."""
        return self.latent_of(torch.softmax(self.logits_of(state), dim=-1))

    # ---- schedule -----------------------------------------------------------------------
    def block_for_sigma(self, sigma):
        """Buckets on the UNWIDENED equal-mass edges (reference estimate_target_layer). The
        gamma-widened training bins overlap, so bucketing on those would hand a sigma in the
        overlap to its larger-sigma neighbour."""
        for b in range(self.num_blocks):
            if sigma >= self.edges[self.num_blocks - 1 - b]:
                return b
        return self.num_blocks - 1

    def sample_sigma(self, block_ids, shape, device):
        lo = self.cdf_lo[block_ids].view(-1, *([1] * (len(shape) - 1)))
        hi = self.cdf_hi[block_ids].view(-1, *([1] * (len(shape) - 1)))
        u = torch.rand(shape, device=device) * (hi - lo) + lo
        return torch.exp(self.p_mean + self.p_std * _std_ppf(u.clamp(1e-7, 1 - 1e-7)))

    def edm_weight(self, sigma):
        sd = self.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    def denoise(self, cond, zt, sigma, ids):
        """EDM-preconditioned D_theta(zt; cond, sigma): the block's estimate of z0."""
        sd = self.sigma_data
        s2 = sigma**2 + sd**2
        c_skip, c_out, c_in = sd**2 / s2, sigma * sd / s2.sqrt(), 1.0 / s2.sqrt()
        t_emb = self.t_embed((0.25 * sigma.log()).reshape(-1)).view(*sigma.shape[:2], self.H)
        return c_skip * zt + c_out * self.blocks(c_in * zt, cond, t_emb, ids)

    # ---- the policy readout -------------------------------------------------------------
    def forward(self, cond):
        """(...,H) trunk feature -> (...,A,M) bin logits. The ONLY policy readout, so the
        network is always evaluated exactly as it is trained."""
        lead = cond.shape[:-1]
        flat = cond.reshape(-1, self.H)
        state = self._ode_state(flat) if self.dblock else self._e2e_state(flat)
        return self.logits_of(state).view(*lead, self.act_dim, self.M)

    def _e2e_state(self, cond):
        N = cond.shape[0]
        t_emb = self.t_embed(torch.zeros(1, device=cond.device)).view(1, 1, self.H)
        z = self.z_start.expand(N, self.state_dim).unsqueeze(0)
        ids = torch.zeros(1, dtype=torch.long, device=cond.device)
        for b in range(self.num_blocks):
            z = z + self.blocks(z, cond, t_emb, ids + b)
        return z.squeeze(0)

    def _ode_state(self, cond):
        """Euler solve down the sigma ladder, visiting the block owning each sigma, with the
        state REPROJECTED through the codebook each step (reference diffusion_step)."""
        N = cond.shape[0]
        z = (self.z_seed * self.ladder[0]).expand(N, self.state_dim).unsqueeze(0)
        t_all = self.t_embed(self.ladder_cnoise).view(-1, 1, 1, self.H)
        last = len(self.ladder) - 1
        for i in range(last + 1):
            c_skip, c_out, c_in = self.ladder_precond[i]
            raw = c_skip * z + c_out * self.blocks(
                c_in * z, cond, t_all[i], self.ladder_ids[i : i + 1])
            # The LAST rung's estimate is the answer and is decoded once by logits_of.
            # Reprojecting it first would decode-and-re-encode, smoothing the policy through
            # the codebook for no reason; reprojection exists only to put an INTERMEDIATE
            # state back on the target manifold before the next Euler step.
            if i == last:
                return raw.squeeze(0)
            D = self.reproject(raw)
            d = self.ladder_dt[i]
            z = (1.0 + d) * z - d * D
        raise AssertionError("empty sigma ladder")

    # ---- training -----------------------------------------------------------------------
    def block_losses(self, cond, q, block_ids, loss_fn):
        """Per-block DiffusionBlocks losses, ALL BLOCKS IN ONE BATCHED CALL. Block b's loss
        touches ONLY block b's parameters (plus the shared conditioning trunk and codebook
        decoder, which the paper also trains from every block). loss_fn(logits) -> (N,) is
        the task loss, unmodified."""
        N, G = cond.shape[0], block_ids.shape[0]
        z0 = self.latent_of(q).unsqueeze(0)                                   # (1,N,A*d)
        sigma = self.sample_sigma(block_ids, (G, N, 1), cond.device)          # (G,N,1)
        zt = z0 + sigma * torch.randn(G, N, self.state_dim, device=cond.device)
        D = self.denoise(cond, zt, sigma, block_ids)                          # (G,N,A*d)
        per = torch.stack([loss_fn(self.logits_of(D[g])) for g in range(G)])   # (G,N)
        # EDM weight, NORMALIZED WITHIN EACH BLOCK. w(sigma) == 1/c_out(sigma)^2, and c_out is
        # exactly the factor by which a block's output is attenuated before it reaches the
        # loss, so dropping w entirely (as this file first did) under-trains the small-sigma
        # blocks: at the per-block median sigmas 0.804/0.301/0.111 the raw w is 5.55/15.0/85.7,
        # i.e. blocks 1 and 2 saw 0.61x and 0.25x of block 0's per-unit-output gradient -- and
        # those are the blocks owning the FINAL ladder rungs that decide the answer.
        # Normalizing per block (w / mean_N w) keeps the within-bin shape that compensates
        # c_out while giving every block unit mean weight, so the raw [24, 2.5e5] span never
        # reaches the CE and no block's effective sample size collapses.
        w = self.edm_weight(sigma.squeeze(-1))                                 # (G,N)
        w = w / w.mean(-1, keepdim=True).clamp_min(1e-12)
        # Probe stays UNWEIGHTED so per-block CEs remain comparable across blocks.
        return (w * per).mean(-1), per.mean(-1).detach()


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
        elif self.actor_dist == "disc":
            # Per-dimension categorical over ORDERED bins, read out of the EDM block stack.
            self.actor_head_net = DiscActorHead(H, act_dim, args)
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
        # disc: the NATIVE sample is the per-dimension BIN INDEX, so log pi is exact and the
        # PPO ratio needs no correction. to_action maps bin centers in [-1,1] to the env box.
        logits = self.actor_head_net(actor_feat)                       # (N,A,M)
        dist = torch.distributions.Categorical(logits=logits)
        head = self.actor_head_net
        span = 0.5 * (self.action_high - self.action_low)
        mid = 0.5 * (self.action_high + self.action_low)
        to_action = lambda k: mid + span * head.bin_centers[k.long()]
        log_det_fn = lambda k: 0.0     # discrete: no Jacobian
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

    def get_action_and_value(self, x, z=None, actor_no_grad=False):
        # z is the distribution-NATIVE sample (the per-dimension BIN INDEX for disc, pre-tanh
        # for gaussian). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        # actor_no_grad builds the ACTOR readout with NO graph while the CRITIC keeps one, and
        # actor_feat is returned so the caller can reuse this (the ONLY) trunk forward. Under
        # --dblock every actor output here is a diagnostic -- pg_loss comes from
        # block_losses(actor_feat, ...) instead -- yet building the 6-rung Euler ODE graph
        # anyway cost a measured 6393 us per 1024-row minibatch. The trunk runs BEFORE the
        # switch, so actor_feat keeps its graph and the block-local loss still trains the trunk.
        actor_feat, critic_feat = self._trunks(x)
        value_logits = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not actor_no_grad):
            dist, to_action, log_det_fn = self._actor_dist(actor_feat)
            if z is None:
                z = dist.sample()
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
            # dist_logits is the rollout policy's FULL per-dimension distribution. The
            # mirror-descent target q = softmax(log pi_old + eta*Ahat*onehot(bin)) needs all of
            # it, not just log pi(a), and a discrete policy is the only reason it is cheap to keep.
            dist_logits = dist.logits if self.actor_dist == "disc" else None
        return action, z, log_prob, entropy, value_logits, dist_logits, actor_feat

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk). All distribution
        # heads are clipped together as one actor group (2-way decoupled clip).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_head_net.parameters())
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
    # This file's actor is the discretized diffusion head; the Beta path is gone.
    assert args.actor_dist == "disc", "this variant requires --actor-dist disc"
    assert args.actor_loss in ("mdce", "ppoclip")
    # Block-local training needs a per-example target to noise. The clipped surrogate has
    # none (it has a direction, not a target), so ppoclip is end-to-end only.
    assert not (args.dblock and args.actor_loss == "ppoclip"), \
        "--dblock requires --actor-loss mdce (block-local training needs a target to noise)"
    # ENTROPY UNDER --dblock IS A METHOD VIOLATION, not just a cost. The entropy comes from
    # the ACTING readout (_ode_state), where block 0's output feeds block 1, so an entropy
    # bonus backprops through COMPOSED blocks -- exactly what block-local training forbids.
    # Measured per-block in_w grad norms with ONLY block 0's loss in the backward:
    #   --ent-coef 0     -> ['8.1800e+00', '0.0000e+00', '0.0000e+00']   depth-local (correct)
    #   --ent-coef 0.01  -> ['8.1659e+00', '1.0324e-04', '4.1764e-04']   leaks into blocks 1,2
    assert not (args.dblock and args.ent_coef != 0.0), \
        "--ent-coef != 0 under --dblock sends gradient through COMPOSED blocks (see comment)"
    # z0 = q @ codebook with codebook (num_action_bins, dblock_latent_dim). The claim that this
    # encoding is lossless -- distinct q give distinct z0, so the denoise target is identifiable
    # -- needs the M codebook rows to be linearly independent, hence d >= M. Defaults: 16 >= 15.
    assert args.dblock_latent_dim >= args.num_action_bins, \
        "dblock_latent_dim must be >= num_action_bins for the codebook encoding to be lossless"
    # d must also be EVEN: the codebook is [sin, cos] over d//2 frequencies, so odd d builds a
    # (M, d-1) table while state_dim stays act_dim*d -- measured, d=17 gives a (15,16) codebook
    # and the Euler step dies mid-rollout with "size of tensor a (102) must match b (96)".
    assert args.dblock_latent_dim % 2 == 0, \
        "dblock_latent_dim must be even (the codebook is [sin, cos] over dblock_latent_dim//2 freqs)"
    assert args.dblock_layers_per_block >= 1 and args.dblock_num_blocks >= 1
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
    # The rollout policy's FULL per-dimension distribution, needed by the mirror-descent
    # target. num_steps*num_envs*act_dim*num_action_bins floats == 12MB at the defaults.
    old_logits = torch.zeros(
        (args.num_steps, args.num_envs, int(np.prod(envs.single_action_space.shape)),
         args.num_action_bins)
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Every block's id, for the one batched all-blocks training call, plus per-block CE
    # probes accumulated on DEVICE and reduced once per iteration at the writer.
    all_block_ids = torch.arange(agent.actor_head_net.num_blocks, device=device)
    dblock_ce_sum = torch.zeros(agent.actor_head_net.num_blocks, device=device)
    dblock_ce_n = torch.zeros(agent.actor_head_net.num_blocks, device=device)
    # EMA of the mirror-descent target's own KL radius, in nats. Sizes the early-stop leash
    # and is logged so the trust region is observable rather than assumed. Carried ACROSS
    # iterations so the first minibatch of an iteration already has a leash.
    mdce_kl_run = None
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
        dblock_ce_sum.zero_()
        dblock_ce_n.zero_()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits, dist_logits, _ = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            old_logits[step] = dist_logits

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
                _, _, boot_logprob, _, _, _, _ = agent.get_action_and_value(next_obses[-1])
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
        b_old_logits = old_logits.reshape(-1, old_logits.shape[-2], old_logits.shape[-1])
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
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # actor_no_grad under --dblock: pg_loss is built by block_losses on
                # mb_actor_feat below, so this readout's actor outputs are diagnostics only
                # (newlogprob -> ratio/approx_kl/clipfracs/target_kl; entropy -> logging).
                # value_logits carries its graph in BOTH arms, so the value loss is identical.
                (_, _, newlogprob, entropy, value_logits, new_logits,
                 mb_actor_feat) = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds], actor_no_grad=args.dblock
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

                if args.actor_loss == "mdce":
                    # ---- the per-example target y, in CLOSED FORM. The mirror-descent step
                    #   q_j(m) ∝ pi_old,j(m) * exp(eta * Ahat * 1[m == bin(a_j)])
                    # is the exact maximizer of Ahat - (1/eta)KL(pi||pi_old) under PPO's own
                    # single-sample advantage estimator. FROZEN (detached): it is the label,
                    # and CE(q, pi) is then a trust-region policy-improvement step.
                    with torch.no_grad():
                        taken = b_latent_zs[mb_inds].long().unsqueeze(-1)      # (N,A,1)
                        bump = torch.zeros_like(b_old_logits[mb_inds])
                        bump.scatter_(-1, taken, 1.0)
                        # STANDARDIZE the advantage that sets the step size. mb_advantages has
                        # already been divided by this minibatch's return-percentile scale
                        # S = max(1, P95-P5) of raw returns, which on HalfCheetah measures
                        # 16.3-26.0 and DRIFTS ~60% over training. Feeding that to eta makes
                        # the trust-region step size a function of the environment's return
                        # spread: measured std 0.23-0.32, so eta=1 bumps a logit by ~0.3 and
                        # leaves q just 0.014 nats from uniform out of 6*ln(15)=16.25. The
                        # end-to-end arm still captures that sliver (it optimizes the acting
                        # policy directly), but a DIFFUSED target at 0.014 nats has per-coord
                        # SNR 0.006 against sigma in [0.11, 0.80] -- the signal is below the
                        # noise the method deliberately adds, so block-local training gets
                        # nothing. Standardizing makes eta dimensionless: one unit of eta is
                        # one advantage std, independent of env and of training time.
                        adv_q = mb_advantages - mb_advantages.mean()
                        adv_q = adv_q / adv_q.std().clamp_min(1e-8)
                        mb_q = torch.softmax(
                            torch.log_softmax(b_old_logits[mb_inds], dim=-1)
                            + args.tpo_eta * adv_q.view(-1, 1, 1) * bump,
                            dim=-1,
                        )
                        # The target's OWN trust-region radius, in the SAME units as the
                        # approx_kl the early stop compares against: KL(q || pi_old), summed
                        # over action dims because both distributions factorize. This is the
                        # step the objective actually asked for, so the leash is sized to it
                        # rather than to a constant tuned for a different policy family.
                        logp_old = torch.log_softmax(b_old_logits[mb_inds], dim=-1)
                        mdce_kl = (
                            (mb_q * (mb_q.clamp_min(1e-12).log() - logp_old)).sum(-1).sum(-1)
                        ).mean()
                        mdce_kl_run = (
                            mdce_kl if mdce_kl_run is None else 0.9 * mdce_kl_run + 0.1 * mdce_kl
                        )

                    def actor_ce(logits):
                        """The task loss: cross-entropy of the frozen target against a
                        candidate set of per-dimension logits. Summed over action dims, so it
                        is the joint CE of the factorized policy."""
                        return -(mb_q * torch.log_softmax(logits, dim=-1)).sum(-1).sum(-1)

                    if args.dblock:
                        # DiffusionBlocks: each block trained ALONE on the FULL task loss of
                        # its own one-step denoise. Blocks never compose, so one batched call
                        # covers all of them and the backward is depth-local by construction.
                        block_ids = (
                            all_block_ids
                            if args.dblock_train_all_blocks
                            else all_block_ids[torch.randint(
                                0, agent.actor_head_net.num_blocks, (1,), device=device)]
                        )
                        # ONE trunk forward per minibatch: mb_actor_feat is the grad-carrying
                        # actor feature from the call above. This branch used to run
                        # agent._trunks on the same rows a SECOND time; the two features were
                        # verified identical (max diff 0.000e+00), so the pass was pure cost.
                        per_block, per_block_ce = agent.actor_head_net.block_losses(
                            mb_actor_feat, mb_q, block_ids, actor_ce
                        )
                        pg_loss = per_block.mean()
                        dblock_ce_sum.index_add_(0, block_ids, per_block_ce)
                        dblock_ce_n.index_add_(
                            0, block_ids, torch.ones_like(per_block_ce))
                    else:
                        # END-TO-END control: the same stack, the same target, ONE backward
                        # through all B composed blocks.
                        pg_loss = actor_ce(new_logits).mean()

                # v162 HL-Gauss MTP value loss: per-horizon CE to scalar-return
                # targets, summed across valid horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds].to(device=value_logits.device, non_blocking=True)
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(device=value_logits.device, dtype=value_ce.dtype, non_blocking=True)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

                # LOGGED value only, hence detached: under --dblock the actor readout has no
                # graph at all, and in the e2e arm nothing should keep one alive to write a
                # scalar. The bonus below uses entropy.mean() directly when it is live.
                entropy_loss = entropy.detach().mean()

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
                # Is the entropy bonus live at all? It is DROPPED from the backward rather than
                # scaled by ent_coef_eff == 0.0 (the default), because multiplying by zero still
                # walks the entire actor graph for EXACTLY zero gradient: measured max
                # |grad_with - grad_without| = 0.000e+00 over all 235 actor params, costing
                # 14106.5 us of the pre-fix dblock actor step's 29217.0 us (48.3%) against
                # 231.2 us in the e2e arm -- a 61x wall-clock bias against the treatment arm.
                # The dblock arm no longer reaches either case: the --ent-coef assert rejects it
                # at startup and actor_no_grad leaves entropy graphless. So what this guard still
                # buys is the e2e arm's 231.2 us. If that assert is ever relaxed, revisit HERE
                # first: under --dblock a live bonus would be a silent no-op, not an error.
                # (auto_alpha's alpha = exp(log_alpha) is never exactly 0, so it is always live
                # there; it is also gaussian-only, while this file forces --actor-dist disc.)
                ent_bonus_on = auto_alpha or args.ent_coef != 0.0

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
                    actor_obj = pg_loss
                    if ent_bonus_on:
                        actor_obj = actor_obj - ent_coef_eff * entropy.mean()
                    actor_obj.backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss + v_loss * args.vf_coef
                    if ent_bonus_on:
                        loss = loss - ent_coef_eff * entropy.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            # Early stop against the SELF-CALIBRATING leash when no absolute one is set. The
            # slack multiple is on the mirror-descent target's own KL, so the update is free
            # to reach the step the objective asked for and is cut only if an epoch overshoots
            # it. Falls back to the absolute leash whenever --target-kl is given, and to the
            # base's behaviour entirely outside the mdce path (where there is no q).
            kl_leash = args.target_kl
            if kl_leash is None and mdce_kl_run is not None:
                kl_leash = args.tpo_kl_slack * mdce_kl_run.item()
            if kl_leash is not None and approx_kl > kl_leash:
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
        # The trust region, made observable. mdce_target_kl is the radius the objective asked
        # for; kl_leash is what the early stop enforced; approx_kl is what the update achieved.
        # v1's failure was visible in exactly this triple: achieved pinned at the leash, both
        # far below the target. Healthy means achieved TRACKS target and the leash is slack.
        if mdce_kl_run is not None:
            writer.add_scalar("debug/mdce_target_kl", mdce_kl_run.item(), global_step)
            writer.add_scalar("debug/kl_leash", kl_leash, global_step)
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
        # DiffusionBlocks probes: block b's own task CE on its one-step denoise, unweighted
        # and therefore comparable across blocks. Block 0 owns the LARGEST sigmas ("predict
        # the target policy from the observation alone"), so the series should be monotonically
        # DECREASING in b. FLAT means the noised latent is carrying no information and the
        # blocks have degenerated into B independent single-shot predictors -- the exact
        # failure mode that killed the Beta-parameter port.
        if args.dblock:
            means = torch.where(
                dblock_ce_n > 0,
                dblock_ce_sum / dblock_ce_n.clamp_min(1.0),
                torch.full_like(dblock_ce_sum, float("nan")),
            ).tolist()
            for b, ce in enumerate(means):
                writer.add_scalar(f"losses/dblock_actor_ce_b{b}", ce, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

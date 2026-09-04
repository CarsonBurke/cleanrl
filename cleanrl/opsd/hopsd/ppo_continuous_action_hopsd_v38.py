# HOPSD v38 — reward-only privilege tokens (leak-free by construction).
# =====================================================================================
# TWO AUTOPSIES, ONE ROOT CAUSE.
#   V35 (INTERPOLATION LEAK): phi = transformer over future ACTIONS. Actions autocorrelate with
#     a_t (policy smoothness), so the encoder near-RECONSTRUCTS a_t by interpolating neighbors.
#   V37 (INVERSE-DYNAMICS LEAK): phi = transformer over future OUTCOME tokens [norm state-delta,
#     reward]. BOTH arms catastrophically failed the 500k gate — hybrid -13, attn -223 (pooled
#     ref 3139) — with teacher_nll SATURATED to -7.6/-8.9 (base ~-3.5) and teacher_ev 0.99. The
#     leak MOVED, it didn't close: in MuJoCo, obs_t plus the future Δstate sequence determines
#     the action sequence by INVERSE DYNAMICS, so the teacher became a physics-based behavior
#     cloner. The priv budget engaged CORRECTLY (priv_kl 3.3->0.9, lam->2.0) and still could not
#     help: a soft KL penalty loses when the MLE payoff for cheating is that large.
# DESIGN LAW (from the v35+v37 pair). The AWR improvement signal is advantage-weighted SELECTION
#   over an UNCERTAIN action prediction. Any phi with high mutual information I(phi; a_t | s_t)
#   saturates the MLE, makes the tilt weights irrelevant, and degrades the teacher to a behavior
#   clone -- regardless of any penalty. Leak prevention must live in the CONTENT of phi, not in a
#   regularizer. Actor-side privilege needs LOW mutual information with the realized action while
#   still carrying OUTCOME information. Rewards are exactly that channel: r_{t+1..t+H} say which
#   trajectories went well WITHOUT determining the controls (only a weak ||a||^2 trace via the
#   ctrl-cost term).
# V38 — minimal delta on v37 (which keeps the modern encoder + priv budget + critic asymmetry):
#   (1) --token-content {reward (default), outcome}. reward: token_{t+k} = a deterministic
#       FOURIER LIFT of x = r_{t+k}/q_reward_scale -> [x, sin(f_i*x), cos(f_i*x)] for 4 log-spaced
#       f_i in {1,2,4,8} = dim 9 -> tok_proj(9->d_model); RoPE gives temporal position. (The lift
#       is a function of the reward ALONE, so it stays leak-neutral, but it spans >1 dim so the H
#       token embeddings are NOT colinear -- a bare scalar token was rank-1 and fp-ill-conditioned
#       through the encoder.) outcome = v37 behavior, retained for the ledger. --phi-mode kept,
#       hybrid default = [obs, pooled action-features, reward summary]; pooled = v34.1 ACTION control.
#   (2) The v37 ACTOR PRIVILEGE BUDGET is untouched (priv_eps 0.5 default) — cheap insurance that
#       should be NEARLY INACTIVE in reward mode (reward tokens can't predict a_t, so full and
#       null actors barely differ). The teacher CRITIC stays UNBUDGETED (full phi always): reward
#       privilege is a legitimate, rich value signal (trajectory quality) that the actor cannot
#       misuse to clone behavior — the failure modes (uninformative vs unfollowable vs LEAKY) stay
#       separated.
#   (3) LEAK ALARM telemetry (no behavior change): debug/nll_gap = teacher_nll(null phi) -
#       teacher_nll(full phi), the plain (unweighted) NLL of the realized action -- a direct
#       I(phi; a_t | s_t) proxy. POSITIVE = privilege predicts the action; SMALL+STABLE = healthy;
#       the two dead versions would have shown a huge, growing gap. A console WARNING fires when
#       teacher_nll < -5.0 (the saturation signature). NOTE: the sign is null-minus-full so that a
#       leak reads POSITIVE (matches "positive gap = predictivity"); flip if the opposite is wanted.
# HYPOTHESIS: reward-token privilege gives the teacher trajectory-QUALITY insight it CANNOT use to
#   clone behavior -> teacher_nll stays in the healthy -2..-4 band, the tilt stays live, distill_kl
#   holds 0.27-0.30, and the reward summary sharpens the tilt's target selection over pooled alone.
#   Expected signature: nll_gap small-positive and stable (not exploding), lam_priv near its floor,
#   return >= pooled 3139@500k.
# JUDGMENT CALLS: (a) nll_gap uses PLAIN (unweighted) NLL, not the AWR-weighted teacher_nll, so it
#   estimates the mutual-information quantity of the design law directly, not a tilt-weighted one.
#   (b) reward tokens keep the frozen q_reward_scale (units match the critic/return); the state-
#   delta branch (outcome mode) is v37-exact. (c) g appended last (irrelevant at teacher_sees_g=False).
# =====================================================================================
# --- v35 method notes (encoder + compile + audit cuts), carried over unchanged ---
# METHOD. The teacher's privileged future summary phi is no longer a lossy near/far
# mean/std pooling of the H=20 future actions + a valid_frac scalar. Instead a small
# bidirectional TRANSFORMER encodes the raw future-action sequence:
#   tokens = the 20 future actions, each 6-dim action linear-projected to d_model=64;
#   RoPE positional encoding over the 20 action positions (relative temporal order,
#   losslessly, not two coarse buckets); a learnable SUMMARY query token is prepended
#   (position 0, NOT RoPE-rotated) whose final-layer output IS phi; FULL (bidirectional,
#   non-causal) attention; validity handled by a key-padding mask over invalid future
#   steps (replaces valid_frac) — the summary key is always valid, so all-invalid rows
#   attend only to the learned summary token => a well-defined learned-null phi, no NaNs.
#   3 pre-norm RMSNorm layers, 4 heads, SwiGLU FFN + QK-norm, attention via
#   F.scaled_dot_product_attention (see v37 header for the modernized block).
#   The encoder is SHARED: its summary feeds the teacher actor ([obs, summary]) AND the
#   teacher HL-Gauss critic ([obs, summary]) through SEPARATE trunks+heads, so gradient
#   from both teacher losses flows into the encoder. Optionally torch.compiled
#   (mode="reduce-overhead", CUDA graphs) behind --compile-teacher with eager fallback;
#   H is padded to a fixed 20 so shapes stay static for the graph.
# NOVELTY vs the ladder: privilege here is the ORDERED future, read by attention, not a
#   permutation-invariant pool. This is the credible fix for the failure the v19 ablation
#   ladder recorded: RAW per-step future actions via a concat-MLP LOST to pooling (5288 ->
#   4107) because per-step actions correlate with a_t through policy smoothness and inject
#   target JITTER; pooling denoised it but destroyed temporal order. RoPE+attention keeps
#   the order while letting the encoder learn what to denoise.
# HYPOTHESIS: attention over the ordered future is a strictly richer phi than two pooled
#   windows, so the teacher rationalizes better -> teacher_nll and teacher EV improve and
#   the student's distillation target sharpens.
# WATCH (jitter failure mode): teacher_nll and teacher EV must MATCH-OR-BEAT the v34.1
#   base by 200-400k. If they lag, the v15/v19 per-step-jitter failure is back and the
#   encoder is re-importing noise instead of gait structure. debug/teacher_attn_entropy
#   is the mechanistic tell: if it sits near log(N) (uniform) the encoder degenerated to
#   pooling and buys nothing; if it falls, attention is using temporal structure.
# OTHER CHANGES (measured, not exploratory): vdis_coef default 0.0 (the info-mismatch
#   bonus measured inert, ~1% of raw reward scale — flag+telemetry kept); q_updates_per_iter
#   32768 (UTD 1.0); new --q-onpol-only path that drops the replay TD loop entirely and
#   instead hardens both Qs to on-policy lambda-returns for --q-onpol-epochs epochs (no 1M
#   replay tensors allocated). DPG/distill/qadv-normalizer/qgrad-guardrail machinery is
#   carried over byte-identical (twin-agreement gate + on-policy hardening untouched).
# VESTIGIAL-AUDIT SIMPLIFICATIONS (behavior-preserving): (1) when vdis_scale==0 the raw GAE
#   sweep is skipped and returns_raw/advantages_raw alias the shaped sweep (they are
#   byte-identical there); the dual sweep only runs when the vdis bonus is live. (2) the
#   v34.1 q_scale EMA + its full-batch min-Q sweep are removed — q_scale cancelled out of
#   the only actuator (dpg_atten). The DPG sensor now divides by qadv_scale directly, so
#   debug/qgrad_ratio IS the loss-path ratio (NOT comparable to v34.1's q_scale-based
#   qgrad_ratio; the old debug/qgrad_ratio_lp and debug/q_scale tags are gone). q_magnitude
#   still logged as debug/q_mean.
# COMPILE (v35): tiered torch.compile with clean per-tier eager fallback + first-iteration
#   numerics parity check. --compile-teacher graphs the phi-encoder; --compile-update graphs
#   the hot update loops — the twin-Q TD step (reduce-overhead / CUDA graphs; THE hot loop at
#   UTD 1.0, 32768 tiny launches/iter) and the student minibatch forward (distill+DPG+value).
#   Static shapes throughout; replay sampling/indexing and all host reads (.item, temp
#   bisection) stay OUTSIDE compiled regions; telemetry accumulates on-GPU and reads once per
#   iteration. debug/compile_tiers_active logs how many tiers actually engaged.
# =====================================================================================
#
# --- v34.1 method (retained below) ---
# HOPSD v34.1 — v34 DPG path, re-normalized by the proposal's action-ADVANTAGE.
# =====================================================================================
# ROOT-CAUSE FIX over v34. Live at 500k the DPG step-share (debug/qgrad_ratio) collapsed
# 0.2-0.35 -> 0.01 and stuck: the v34 normalizer q_scale = EMA(mean|min(Q1,Q2)|) grows
# with absolute state value (q_mean 32 -> 161), so dividing by it shrank the DPG term's
# contribution toward zero even though the action-space signal was still there. The
# absolute Q level is a NON-STATIONARY normalizer. v34.1 replaces it with a stationary one:
#   qadv = min(Q1,Q2)(s, a_rsample).detach() - min(Q1,Q2)(s, a_realized)  [per state]
#     — the advantage of the PROPOSED action over the actually-EXECUTED action on the same
#     minibatch rows (b_obs[mb_inds] / b_actions[mb_inds] are the same transitions). This is
#     ~return-scale-STATIONARY: it does not grow with the absolute value of the state.
#   qadv_scale = max(EMA_0.99(mean|qadv|), qadv_floor); DPG loss = q_coef * mean(w * (-minQ
#     (s,a_rs))) / qadv_scale. Numerator and gradient DIRECTION are byte-identical to v34;
#     only the divisor changed. The FLOOR (qadv_floor=0.05) is load-bearing: if Q genuinely
#     flattens in action space (exhaustion) qadv -> 0, the normalizer rides the floor and
#     the DPG term FADES GRACEFULLY instead of amplifying a noise direction.
#   No servo, q_coef stays 0.2. q_scale's EMA is still maintained but ONLY as telemetry
#     (it feeds the qgrad_ratio sensor, which stays a pure diagnostic — never an actuator).
#   NEW TELEMETRY (the whole point): debug/q_dpg_gap = signed mean(qadv) per iteration
#     (decaying toward 0 as the policy improves = DPG exhaustion signal; staying material
#     while |minQ| grows = confirmation the old normalizer was the bug) and debug/qadv_scale.
#   COVERAGE-INDEPENDENT FRONTIER: debug/q_frontier = mean|min(Q1,Q2)(s, clamp(a_realized +
#     0.1*u)) - min(Q1,Q2)(s, a_realized)| over ~1024 fresh states, u = random unit action
#     direction, fixed radius 0.1 env-action units (scaled-Q units, comparable to
#     q_dpg_gap). This reads local Q sensitivity around the executed action WITHOUT
#     depending on policy coverage. INTERPRETATION CONTRACT: q_dpg_gap -> 0 means action-
#     space EXHAUSTION only if q_frontier -> 0 too. q_frontier staying POSITIVE while
#     q_dpg_gap -> 0 = COVERAGE COLLAPSE (a_rsample has collapsed onto a_realized as entropy
#     fell) -> flip ent_coef to restore coverage; do NOT bury the arm as exhausted.
# Everything else byte-identical to v34. v34 header retained below.
# =====================================================================================
#
# --- v34 method (retained below) ---
# HOPSD v34 — off-policy twin-Q DPG path that PROPOSES better-than-realized actions.
# =====================================================================================
# Diagnosis: v30's improvement operator is bounded by the best actions REALIZED in fresh
# rollouts — the hindsight teacher can only rationalize what the student already did, and
# teacher-student mean-gap sits at ~0.02 and shrinks. To cross that frontier we need a
# gradient that proposes actions NOT yet taken. v34 adds a SAC-style off-policy twin-Q
# critic whose deterministic-policy-gradient (DPG) through a reparameterized student
# action supplies exactly that, while the hindsight teacher/tilt/distill/vdis stack is
# left BYTE-IDENTICAL to v30 as the trust anchor and exploration prior.
#   TRANSITION REPLAY (ring, cap 1M): raw (obs, next_obs, executed action in [-1,1], raw
#     reward, terminated). Obs are stored raw (recovered by un-normalizing with a pooled
#     per-rollout obs_rms snapshot) and RE-normalized at train time with the CURRENT
#     obs_rms so the Q input distribution does not drift with the wrapper stats; same-
#     iteration round-trip is exact, older entries get the current normalization.
#   TWIN Q + targets: Q1,Q2 = [obs+act, 256, 256, 1] with LayerNorm after each hidden
#     Linear; polyak (q_tau) targets; Adam q_lr. TD target bootstraps THROUGH truncation
#     (mask = 1 - terminated; HalfCheetah never truly terminates) at the REAL final obs.
#     Q reward = RAW env reward scaled by a FROZEN constant q_reward_scale (running raw-
#     reward std over the first vdis_warmup_iters, then frozen) — never the vdis-shaped
#     reward, never the drifting return-std.
#   ON-POLICY HARDENING: after the TD block, one pass regressing BOTH Qs to the raw-reward
#     lambda-returns (frozen scale units) on the fresh rollout, INDEPENDENT shuffles per
#     twin so the multi-step target does not couple them and erode the disagreement gate.
#   DPG on the STUDENT (fresh on-policy states only — same state distribution as the trust
#     region): qterm = -min(Q1,Q2)(s, a_rsample)/q_scale, scale-normalized by an EMA of
#     mean|minQ| so q_coef is reward-scale-invariant; gated per-state by twin agreement
#     w = gm/(|Q1-Q2|+gm) so DPG pushes off-frontier ONLY where the twins agree. Optional
#     ent_coef entropy BONUS (mean over dims, default 0) for Q coverage near the operating
#     point. Q nets are frozen during the student epochs (not in the student optimizer).
# KILL-GATES:
#   Gate0 (~200k): debug/q_ev must beat the student critic's explained_variance, else the
#     Q path is noise — abort.
#   Gate1 (~500k): debug/qgrad_ratio in [0.2, 2] and debug/qgrad_cos >= 0 (DPG is neither
#     dominating nor fighting distill), and distill_kl holds its 0.15-0.25 band.
#   Ongoing: debug/q_mean stationary at return scale (no Q blow-up); student entropy not
#     collapsing below ~-7 with debug/arsample_std shrinking (DPG over-sharpening).
# =====================================================================================
#
# --- v30 method (retained below) ---
# HOPSD v30 — privileged-information mismatch as an intrinsic exploration bonus.
# =====================================================================================
# The teacher critic sees (s, phi) — the hindsight future — while the student critic
# sees only s. Their absolute value gap d(s) = |V_T(s,phi) - V_S(s)| measures how much
# the future CHANGES the value estimate at s: high-d states are where outcomes are not
# yet predictable from the present, i.e. exactly the information-rich states worth
# visiting. v30 adds a normalized, non-negative bonus vdis_coef * d/std(d) to rewards
# BEFORE GAE, so exploration credit flows through the advantage into the tilt (the
# teacher preferentially rationalizes info-seeking actions) and the critic targets —
# directed exploration with no replay, no new networks, and an exogenous gate signal.
# Warmup-gated (linear over vdis_warmup_iters) while the teacher critic is untrained.
# Honest caveat: d also contains both critics' errors, not only information content —
# this is the two-critic cheap version of ensemble disagreement. Kill-tell: if
# debug/vdis_mean does not fall over training (teacher EV rising should shrink d on
# mastered states), the bonus is tracking error, not information.
# =====================================================================================
#
# --- v19 (retained below) ---
# HOPSD v19 — two-window hindsight: coarse temporal structure without per-step jitter.
# =====================================================================================
# The phi ablation ladder: no-phi 4726 < single-window pooled 5288 (privilege is worth
# ~10%); but RAW ordered lags (v15) fell to 4107 — per-step future actions correlate
# with a_t through policy smoothness, so they inject target jitter, not gait signal.
# v19 threads the needle: split the 20-step future window into NEAR (a_{t+1..t+5}) and
# FAR (a_{t+6..t+20}) and pool mean/std WITHIN each. Pooling keeps each window
# denoised and permutation-invariant (the property that works); the near/far SPLIT
# restores exactly one bit of temporal order — the trajectory's direction of drift,
# near-vs-far contrast = where the gait is heading — which the single window
# provably destroys and which raw lags delivered too noisily. Context grows 2A+1 ->
# 4A+1 for both teacher inputs (conditioning stays equalized). Everything else is
# byte-identical to v12.2 (winsorized KL-targeted tilt @ 1.2).
# Watch vs control (2206@500k / 4159@1M / 5288@1.5M): if near/far contrast carries
# improvement signal, returns beat control with teacher_student_mean_gap modestly up;
# if it re-imports the v15 jitter, distill_kl inflates toward 0.3 and returns sag —
# the diagnostic pair separates the two failure stories cleanly.
# =====================================================================================
#
# --- v12.2 method (retained below) ---
# v12's autopsy (0.9M): real GAE advantage batches have extreme outliers; the unclamped
# softmax bisection let a handful of samples eat the whole 1.2-nat KL budget, so the
# dual INFLATED temp (0.60->1.03) to protect the target — flattening the tilt for the
# useful bulk — and the safety clamp then truncated those outliers anyway, collapsing
# realized KL to 0.35. Effective tilt ended WEAKER than the champion's 0.90 nats.
# Meanwhile the clamp-shaped w50 arm leads the fleet: the clamp is not safety, it IS
# the robustness mechanism (clamp at w_max == winsorizing adv_z at temp*ln(w_max);
# w50@temp0.4 == clip at +1.56 sigma, then pure exp tilt on the bulk).
# v12.2 keeps the KL-targeted dual but makes it robust the same way: winsorize adv_z at
# +/- adv_clip (2.0 sigma) FIRST, then bisect temp on the clipped distribution and take
# weights as N*softmax (numerically stable, no post-hoc weight clamp — realized KL now
# equals the target identically). Outliers can't eat the budget; the bulk gets the full
# tilt; tilt_eps is a trustworthy dose dial again. Arms: eps 1.2 (shape control vs w50
# at matched strength) and eps 1.8 (dose escalation, now safe to read).
# Prediction: eps 1.2 arm ~= w50 (>= v10_noleash matched-step); eps 1.8 >= both if
# operator strength is still the binding lever. Watch debug/auto_temp (should now FALL
# as the policy sharpens), debug/awr_ess, debug/clip_frac_adv.
# =====================================================================================
#
# --- v12 method (retained below) ---
# Base = pure no-leash champion config (v2, target_kl inert). Single change: the AWR
# temperature is no longer a fixed constant. Each iteration, geometric bisection finds
# the temp whose softmax tilt over the batch advantages sits at a fixed KL from the
# data distribution: KL(softmax(A/temp) || uniform) = tilt_eps. This is the sample-based
# MPO E-step constraint — the improvement step gets a CONSTANT information size in
# distribution space, instead of whatever exp(A/0.5).clamp(20) happens to produce as
# the advantage distribution's shape drifts over training.
# Calibration (N(0,1) advantages, N=32k): the champion's effective tilt (temp 0.5,
# clamp 20) is KL ~= 0.90, and the clamp — not the temp — is the binding controller
# below temp ~0.5 (temps 0.5/0.4/0.3 all land at KL ~= 0.9). Default tilt_eps = 1.2:
# modestly stronger than the champion (evidence says operator strength is binding),
# equal in strength to the temp0.4+w50 clamp arm launched alongside — so the pair
# tests tilt SHAPE (softmax vs clamp-truncated) at matched strength, and v12 adds
# adaptivity. Weights are N*softmax (mean 1, same normalization as before); the old
# clamp is retained only as a loose safety at 200. Logs: debug/auto_temp,
# debug/tilt_kl_realized. Prediction: auto_temp starts ~0.6 and FALLS as the policy
# sharpens (advantage spread shrinks -> constraint loosens temp); matched-step
# returns >= v10_noleash. Risk: softmax tail domination (few samples own the tilt) —
# watch debug/awr_ess.
# =====================================================================================
#
# --- v2 method (retained below) ---
# v2 = v1 with the outcome grade g (z-scored lambda-return) REMOVED from the teacher-
# actor context (teacher_sees_g=False toggle; the teacher critic never saw it). v1 hit
# the predicted degenerate fixed point at 2M (return -86 vs baseline 5012): distill_kl
# 0.019, teacher-student mean gap 0.009, approx_kl 0.0025 — the teacher collapsed to
# posterior reconstruction of the student. Cause: with g in the context the advantage is
# nearly a FUNCTION of the context, so the AWR weight exp(adv_z) is ~constant within
# each context; it reweights which contexts get fit but never tilts the conditional
# toward better-than-taken actions. With g removed, advantage varies within a context
# and the weighting tilts the teacher's conditional — the teacher becomes a genuine
# hindsight-conditioned improvement operator (AWR expressed through a privileged
# rationalizer). Everything else == v1.
# =====================================================================================
#
# --- v1 method (retained below) ---
# Port of On-Policy Self-Distillation onto the iterthink_v24_beta_d3bucket_mtp base
# (config of the ppoadvnorm_batch_v1 run: raw GAE, batch-scope z-scoring). OPSD's LLM
# recipe: a TEACHER conditioned on privileged info (the verified solution) and a STUDENT
# conditioned on the problem only; the student rolls out; at every position the loss is
# per-position forward KL(teacher || student) over the FULL distribution, with pointwise
# per-entry clipping min(l, tau); gradients flow only through the student. Dense
# distillation replaces sparse-reward RL.
#
# RL translation (no verified answer exists, so hindsight is the privilege):
#   TEACHER ACTOR  pi_T(a_t | s_t, phi_t): a separate ThinkTrunk that sees the realized
#     future phi_t over the next H=20 steps (~ GAE effective horizon 1/(1-gamma*lambda)):
#     future-action mean/std per dim (a_{t+1..t+H}; a_t excluded so the teacher cannot be
#     an identity map), the z-scored lambda-return g_t ("returns for the horizon"), and
#     the valid-horizon fraction. Trained by ADVANTAGE-WEIGHTED Beta NLL on the taken
#     native z: w = exp(adv_z/awr_temp).clamp(w_max), mean-normalized. The weighting is
#     the improvement operator (AWR): given hindsight, the teacher fits "the action that
#     should have been taken", not merely the action that was taken. Rationalization
#     (fit a_t given what followed) is far easier than generation — the paper's core bet.
#   TEACHER CRITIC V_T(s_t, gait_t): a separate ThinkTrunk over [obs, future-action
#     mean/std, valid frac] (NO return features — with them it degenerates into a return
#     copier). HL-Gauss CE to lambda-returns; a horizon-Q "V(s, plan)". v1 role:
#     diagnostic (its EV measures how informative the privileged plan summary is).
#   STUDENT: the unchanged base agent (shared ThinkTrunk -> Beta actor + Dreamer3-bucket
#     511-bin HL-Gauss MTP critic). Its actor objective is ONLY the dense distillation
#     loss sum_d min(KL(Beta_T,d || Beta_S,d), tau) at every rollout state, teacher
#     detached — PPO (ratio/clip/advantages in the actor loss) is fully removed. The
#     critic keeps its raw-return CE loss (it anchors GAE -> adv_z, g). target_kl stays
#     as a drift leash (epoch early-stop off replayed-z logprobs), not as an objective.
#
# Faithful-to-paper choices: forward KL (their decisive winner over reverse KL/JSD);
# full-distribution matching (closed-form Beta KL = the "full-vocab logit" analog);
# pointwise per-dim clipping (their per-vocab-entry min(l, tau)); teacher evaluated at
# the ACHIEVED hindsight context on every student state (no elites, no relabeling);
# gradients only through the student. Deviations forced by RL-from-scratch: the teacher
# must learn online (no frozen-at-init teacher) and needs the AWR weighting because a
# rollout future, unlike a verified solution, is not necessarily good.
#
# HYPOTHESIS: hindsight rationalization + advantage weighting make the teacher a
# per-state full-distribution target that is denser and lower-variance than PPO's
# clipped surrogate, so the student improves at every state including bad ones.
# Falsifiable: if the teacher is just posterior reconstruction (no improvement), the
# student converges to self-BC and returns plateau far below the PPO baseline
# (ppoadvnorm_batch_v1: 5012@2M / 6716@4M / 8278@8M).
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

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

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


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
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 1000000.0     # inert (no-leash recipe); kept for the epoch early-stop plumbing
    vdis_coef: float = 0.0           # v30 bonus scale on |V_teacher - V_student| / std; v35: measured
                                     # inert (~1% of raw reward scale) -> default OFF; flag+telemetry kept
    vdis_warmup_iters: int = 20      # v30: linear bonus ramp while the teacher critic trains

    # v34: off-policy twin-Q DPG path.
    q_coef: float = 0.2              # DPG weight in the student actor loss (scale-normalized term)
    q_updates_per_iter: int = 32768  # off-policy TD updates per iteration (v35: UTD 1.0). Ignored
                                     # when q_onpol_only=True.
    q_batch: int = 256               # transition minibatch for the TD updates
    q_lr: float = 3e-4               # Q optimizer lr
    q_tau: float = 0.005             # polyak coefficient for the target Q nets
    q_onpol_coef: float = 1.0        # weight of the on-policy lambda-return hardening pass
    replay_capacity: int = 1_000_000 # transition ring buffer capacity (NOT allocated if q_onpol_only)
    # v35: pure on-policy Q path. Skip the replay TD loop entirely; instead harden BOTH Qs
    # to the on-policy raw-reward lambda-returns for q_onpol_epochs epochs, independent
    # per-twin minibatch shuffles each epoch. No 1M replay tensors are allocated.
    q_onpol_only: bool = False
    q_onpol_epochs: int = 4          # epochs of on-policy hardening when q_onpol_only=True

    # v35: tiered torch.compile of the training step (clean per-tier eager fallback).
    compile_teacher: bool = True     # torch.compile(encoder, mode="reduce-overhead"); eager fallback
    compile_update: bool = True      # compile the hot update loops: twin-Q TD step (CUDA graphs)
                                     # + the student minibatch forward; per-tier eager fallback +
                                     # first-iteration numerics parity check
    teacher_d_model: int = 64        # encoder width == phi summary width
    teacher_layers: int = 3          # encoder depth (pre-LN blocks)
    teacher_heads: int = 4           # attention heads (head_dim = d_model // heads)
    teacher_ffn_mult: int = 4        # FFN inner expansion
    # v36: how the teacher consumes the future window (phi).
    #   pooled = v34.1 near/far mean-std + valid_frac ONLY (encoder never built; fast control)
    #   attn   = transformer summary ONLY (v35 behavior)
    #   hybrid = [obs, pooled_features, summary]: attention can only ADD over pooled, not replace
    phi_mode: Literal["pooled", "hybrid", "attn"] = "hybrid"
    # v38: what the encoder tokens CONTAIN (see the header design law on I(phi; a_t | s_t)).
    #   reward  = Fourier lift of r_{t+k}/q_reward_scale [x, sin(f x), cos(f x)] (dim 1+2F=9) --
    #             trajectory-quality only, near-zero mutual information with the realized action ->
    #             cannot become a behavior clone; lift breaks token colinearity (see REWARD_FOURIER_FREQS)
    #   outcome = [next_obs-obs (norm state-delta), r_{t+k}/q_reward_scale]  (dim obs_dim+1) = v37;
    #             kept for the ledger, but obs_t + the future delta sequence INVERTS to the actions
    #             (inverse dynamics) -> the actor-side leak v37 died on.
    token_content: Literal["reward", "outcome"] = "reward"
    # v36 privilege regularizer (hybrid + attn only): per-row Bernoulli(phi_dropout) mask during
    # TEACHER TRAINING replaces the summary with null_summary (pooled kept in hybrid). v37
    # SUPERSEDES this with the KL-budget below and defaults it OFF; kept available as an ablation.
    phi_dropout: float = 0.0
    # v37 ACTOR PRIVILEGE BUDGET (hybrid + attn): dual-ascent KL constraint on how far the
    # teacher's full-phi actor may sit from its own null-phi (summary-dropped) actor. penalty
    # lam_priv * priv_kl added to the teacher ACTOR loss (critic is UNBUDGETED — full phi always).
    # lam_priv follows priv_kl toward priv_eps by multiplicative dual ascent after each epoch loop.
    priv_eps: float = 0.5            # target priv_kl in nats (summed over action dims, per state)
    priv_eta: float = 0.05           # dual-ascent step size (log-multiplicative)
    priv_lam_init: float = 0.1       # initial multiplier
    priv_lam_min: float = 1e-3       # clamp floor
    priv_lam_max: float = 100.0      # clamp ceiling

    # v34.1: root-cause DPG normalizer. Divide the DPG term by an EMA of the mean
    # |action-advantage| of the proposal over the REALIZED action (return-scale
    # stationary), floored so the term fades gracefully instead of amplifying noise
    # when Q flattens in action space.
    qadv_floor: float = 0.05         # floor on qadv_scale (scaled-Q units); load-bearing
    qgrad_ceiling: float = 4.0       # ceiling-only clamp on the LOSS-PATH DPG/distill grad-norm
                                     # ratio. Defuses qadv_scale integral windup: the divisor decays
                                     # toward the floor as the gap closes, so the gain would blow up
                                     # on ~zero real signal. Fires only above the ceiling; inert when
                                     # the DPG term is genuinely small. NOT a two-sided servo.
                                     # Set just ABOVE the arm's intended operating point (guardrail,
                                     # not dose knob): demonstrated-healthy ratio at q_coef 0.2 is
                                     # ~3.9, so 4.0 here; the sensor scales with q_coef, so scale the
                                     # ceiling with any q_coef override (e.g. 8.0 at q_coef 0.4) or
                                     # the clamp erases the dose contrast.

    # v21 machinery kept from the base: shared student backbone + decoupled clipping.
    share_backbone: bool = True
    separate_grad_clip: bool = True
    actor_grad_clip: float = 0.25    # max-norm for the distill gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Dreamer3-style bucket critic (student MTP + teacher single-horizon share the support).
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    normalize_reward: bool = False
    clip_reward: bool = False

    # --- HOPSD ---
    hindsight_horizon: int = 20      # H: future window for the privileged features (~1/(1-gamma*lambda))
    awr_temp: float = 0.5            # fallback fixed temp (used only if auto_temp=False)
    auto_temp: bool = True           # v12: bisect temp to hit tilt_eps each iteration
    tilt_eps: float = 1.2            # target KL(softmax(A/temp) || uniform) of the tilt
    adv_clip: float = 2.0            # v12.2: winsorize adv_z here BEFORE the tilt (robust dual)
    distill_coef: float = 1.0        # student actor loss = distill_coef * clipped forward KL
    distill_kl_clip: float = 2.0     # tau: pointwise per-action-dim KL clip (paper's min(l, tau))
    teacher_conc_cap: float = 100.0  # hard cap on teacher Beta concentrations (sane sharp targets)
    teacher_vf_coef: float = 0.5     # teacher critic CE weight inside the teacher update
    teacher_grad_clip: float = 0.5   # teacher's own global clip (separate optimizer)
    teacher_sees_g: bool = False     # v2: g in the teacher-actor context kills the AWR tilt (v1 fixed point)

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
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)

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
    """Student: unchanged base agent (Beta actor + HL-Gauss MTP critic on a shared trunk)."""

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
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

    def _trunks(self, x):
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_action_and_value(self, x, z=None):
        # z is the native Beta sample in (0,1); replaying it recomputes log_prob at the
        # same sample (the base's z-replay). Also returns the Beta params for distillation.
        actor_feat, critic_feat = self._trunks(x)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        dist = Beta(alpha, beta)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        log_prob = dist.log_prob(z).sum(1)  # constant rescale Jacobian dropped (cancels)
        entropy = dist.entropy().sum(1)
        value_logits = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        return action, z, log_prob, entropy, value_logits, alpha, beta

    def actor_parameters(self):
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())


def _rotate_half(x):
    """LLaMA/NeoX RoPE pairing: split the last dim in half, rotate the halves."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """x: (..., N, head_dim); cos/sin: (N, head_dim) broadcast over the leading dims."""
    return x * cos + _rotate_half(x) * sin


def swiglu_hidden_dim(d_model):
    """SwiGLU inner width param-matched to a 4x GELU MLP: 8/3 * d_model (SwiGLU has 3 weight
    matrices vs the MLP's 2), rounded to the NEAREST multiple of 8 (64 -> 168, no param growth)."""
    return int(round((8.0 * d_model / 3.0) / 8.0)) * 8


class EncoderLayer(nn.Module):
    """Modern pre-norm bidirectional self-attention block: RMSNorm pre-norm on both sublayers,
    RoPE + per-head QK-norm on Q/K, a SwiGLU FFN, and a clean (un-normed) residual stream.

    Attention over the (summary + H outcome) tokens. `key_mask` is (B, 1, 1, N) boolean with
    True = attend (the summary key is always True, so no query is ever fully masked -> no NaN).
    QK-norm (per-head RMSNorm on Q and K before the dot product) stabilizes the attention
    logits; it is applied to Q/K OUTSIDE the SDPA kernel, so flash/mem-efficient attention is
    preserved (this is the standard "norm q/k before the SDPA call" placement). The default
    path uses F.scaled_dot_product_attention; return_attn=True computes the softmax explicitly
    over the SAME normalized Q/K to expose the weights for telemetry.
    """

    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Scaled residual-branch init: the two out-projections start small (1/sqrt(2*layers))
        # so the residual stream stays near-identity at init (GPT-2/NanoGPT depth-stability trick).
        resid_std = 1.0 / np.sqrt(2 * n_layers)
        self.norm1 = nn.RMSNorm(d_model)
        self.qkv = layer_init(nn.Linear(d_model, 3 * d_model))
        self.q_norm = nn.RMSNorm(self.head_dim)   # per-head QK-norm (learnable scale)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = layer_init(nn.Linear(d_model, d_model), std=resid_std)
        self.norm2 = nn.RMSNorm(d_model)
        h = swiglu_hidden_dim(d_model)
        self.gate_proj = layer_init(nn.Linear(d_model, h))
        self.up_proj = layer_init(nn.Linear(d_model, h))
        self.down_proj = layer_init(nn.Linear(h, d_model), std=resid_std)

    def _ffn(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x, cos, sin, key_mask, return_attn=False):
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                              # each (B, N, nh, hd)
        q = self.q_norm(q)                                   # per-head QK-norm before RoPE
        k = self.k_norm(k)
        q = apply_rope(q.transpose(1, 2), cos, sin)          # (B, nh, N, hd)
        k = apply_rope(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)
        attn = None
        if return_attn:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            scores = scores.masked_fill(~key_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)             # (B, nh, N, N)
            attn_out = torch.matmul(attn, v)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(attn_out)
        x = x + self._ffn(self.norm2(x))
        return x, attn


class PhiEncoder(nn.Module):
    """Non-lossy hindsight encoder: bidirectional transformer over the H future OUTCOME tokens.

    v37: each token = the outcome of a future step (normalized state-delta + scaled reward),
    NOT the future action. A learnable summary query (position 0, NOT RoPE-rotated) is
    prepended to the H tokens (each linear-projected to d_model). RoPE gives the outcome
    tokens their temporal position; full attention; a key-padding mask hides invalid future
    steps. The final-layer output at the summary position IS phi. Fixed H -> static shapes.
    """

    def __init__(self, token_dim, d_model, n_layers, n_heads, ffn_mult, horizon):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        # ffn_mult (teacher_ffn_mult) is superseded by the SwiGLU param-matched sizing
        # (swiglu_hidden_dim); kept in the signature only for CLI/Args stability.
        self.tok_proj = layer_init(nn.Linear(token_dim, d_model))
        self.summary_token = nn.Parameter(torch.zeros(d_model))  # learned summary query
        # Learned-null summary for rows with NO valid future step (episode tails). Such a row's
        # only valid key is the summary token, so the summary position attends to a single
        # degenerate value; routing those rows to this leaf via torch.where gives a well-defined
        # learned-null phi AND severs the gradient through the degenerate attention path.
        self.null_summary = nn.Parameter(torch.zeros(d_model))
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, n_layers) for _ in range(n_layers)]
        )
        self.out_norm = nn.RMSNorm(d_model)   # final pre-readout norm
        # RoPE tables for N = H+1 positions. Row 0 (summary) is identity (cos=1, sin=0);
        # rows 1..H rotate action positions 0..H-1. Non-persistent buffers -> follow .to().
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(horizon).float()
        freqs = torch.outer(pos, inv_freq)                   # (H, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)              # (H, head_dim)
        cos = torch.cat([torch.ones(1, head_dim), emb.cos()], dim=0)   # (H+1, head_dim)
        sin = torch.cat([torch.zeros(1, head_dim), emb.sin()], dim=0)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, future_tokens, future_valid, return_attn=False):
        # future_tokens: (B, H, token_dim); future_valid: (B, H) bool (True = real future step).
        B = future_tokens.shape[0]
        tok = self.tok_proj(future_tokens)                   # (B, H, d)
        summ = self.summary_token.expand(B, 1, self.d_model) # (B, 1, d)
        x = torch.cat([summ, tok], dim=1)                    # (B, H+1, d)
        # Key mask: summary key always valid; token keys per validity. (B, 1, 1, N).
        summ_valid = torch.ones(B, 1, dtype=torch.bool, device=future_tokens.device)
        key_valid = torch.cat([summ_valid, future_valid.bool()], dim=1)  # (B, N)
        key_mask = key_valid[:, None, None, :]
        attn_last = None
        for i, layer in enumerate(self.layers):
            want = return_attn and (i == len(self.layers) - 1)
            x, a = layer(x, self.rope_cos, self.rope_sin, key_mask, return_attn=want)
            if want:
                attn_last = a
        summary = self.out_norm(x)[:, 0]                     # (B, d) at the summary position
        # All-invalid rows -> learned null (severs the constant-activation LayerNorm grad).
        any_valid = future_valid.bool().any(dim=1, keepdim=True)   # (B, 1)
        summary = torch.where(any_valid, summary, self.null_summary)
        if return_attn:
            return summary, attn_last
        return summary


class HindsightTeacher(nn.Module):
    """Separate teacher actor + teacher critic, privileged by a phi built per phi_mode.

    phi_mode selects what the heads see after obs:
      pooled: [obs, pooled_feats]                       (encoder NOT built)
      attn:   [obs, summary]                            (v35)
      hybrid: [obs, pooled_feats, summary]              (attention only ADDS over pooled)
    pooled_feats = [near_mean, near_std, far_mean, far_std, valid_frac] (4A+1), from
    build_future_features. summary = transformer phi (d_model), from the shared encoder.
    The actor additionally appends g (z-scored lambda-return) iff teacher_sees_g; the critic
    never sees return features. phi_dropout (hybrid/attn) replaces the summary with the
    encoder's learned null on a per-row mask DURING TRAINING only (privilege reliance cap).
    """

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        H = args.hidden
        d = args.teacher_d_model
        self.phi_mode = args.phi_mode
        self.needs_pooled = args.phi_mode in ("pooled", "hybrid")
        self.needs_attn = args.phi_mode in ("attn", "hybrid")
        pooled_dim = 4 * act_dim + 1 if self.needs_pooled else 0
        summ_dim = d if self.needs_attn else 0
        self.encoder = None
        # v38: token width follows --token-content. reward = Fourier lift of the scaled reward
        # (dim 1+2F, the low-I(phi;a_t) channel, colinearity broken); outcome = v37 [norm
        # state-delta, scaled reward] (dim obs_dim+1).
        self.token_dim = reward_token_dim() if args.token_content == "reward" else obs_dim + 1
        if self.needs_attn:
            self.encoder = PhiEncoder(
                self.token_dim, d, args.teacher_layers, args.teacher_heads, args.teacher_ffn_mult,
                args.hindsight_horizon,
            )
        actor_in = obs_dim + pooled_dim + summ_dim + (1 if args.teacher_sees_g else 0)
        critic_in = obs_dim + pooled_dim + summ_dim
        self.actor_trunk = ThinkTrunk(actor_in, H, args.k_blocks, args.n_experts)
        self.alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.critic_trunk = ThinkTrunk(critic_in, H, args.k_blocks, args.n_experts)
        self.critic_head = layer_init(nn.Linear(H, args.num_bins, bias=False), std=0.1)
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.conc_cap = args.teacher_conc_cap
        # torch.compile plumbing (reduce-overhead / CUDA graphs) with eager fallback.
        # Only meaningful when an encoder exists (attn/hybrid); pooled stays eager (no encoder).
        self._eager_encoder = self.encoder
        self._compiled_encoder = None
        self._use_compiled = bool(args.compile_teacher) and self.needs_attn
        if self._use_compiled:
            try:
                self._compiled_encoder = torch.compile(self.encoder, mode="reduce-overhead")
            except Exception as e:  # compilation setup failed -> stay eager
                print(f"[compile-teacher] torch.compile setup failed ({e}); using eager encoder")
                self._use_compiled = False

    def encode(self, future_tokens, future_valid):
        """Fast path (compiled if enabled), no attention weights. attn/hybrid only."""
        if self._use_compiled:
            try:
                return self._compiled_encoder(future_tokens, future_valid)
            except Exception as e:  # first (or any) compiled call failed -> permanent fallback
                print(f"[compile-teacher] compiled encoder call failed ({e}); falling back to eager")
                self._use_compiled = False
        return self._eager_encoder(future_tokens, future_valid)

    def encode_with_attn(self, future_tokens, future_valid):
        """Eager telemetry path: returns (summary, last-layer attention weights)."""
        return self._eager_encoder(future_tokens, future_valid, return_attn=True)

    def _phi_parts(self, pooled, summary, drop_mask):
        """Assemble the post-obs phi pieces for the current mode. drop_mask (B,1 bool, True =
        drop) routes those rows' summary to the learned null (training-time privilege cap)."""
        if summary is not None and drop_mask is not None:
            summary = torch.where(drop_mask, self.encoder.null_summary, summary)
        if self.phi_mode == "pooled":
            return [pooled]
        if self.phi_mode == "attn":
            return [summary]
        return [pooled, summary]  # hybrid

    def actor_params_from(self, obs, pooled, summary, g=None, drop_mask=None):
        ctx = [obs] + self._phi_parts(pooled, summary, drop_mask)
        if g is not None:
            ctx.append(g)
        feat = self.actor_trunk(torch.cat(ctx, dim=-1))
        alpha = (1.0 + F.softplus(self.alpha_head(feat))).clamp(max=self.conc_cap)
        beta = (1.0 + F.softplus(self.beta_head(feat))).clamp(max=self.conc_cap)
        return alpha, beta

    def critic_logits_from(self, obs, pooled, summary, drop_mask=None):
        ctx = [obs] + self._phi_parts(pooled, summary, drop_mask)
        return self.critic_head(self.critic_trunk(torch.cat(ctx, dim=-1)))


class QNet(nn.Module):
    """v34: SAC-style state-action value MLP with LayerNorm (high-UTD stabilizer).

    Input actions live in the ENV action space ([-1, 1] per dim) — the same space the
    executed actions are stored in and the reparameterized student action is mapped to.
    """

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden)),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


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


NEAR_HORIZON = 5  # v19/v34.1: near window = a_{t+1..t+5}; far window = a_{t+6..t+H}


def build_future_features(actions, boundaries, horizon):
    """v34.1 (reconstructed for phi_mode='pooled'/'hybrid'): per-(t, env) mean/std pooled
    WITHIN a near and a far future window.

    actions: (T, B, A); boundaries: (T, B) 1.0 where the transition at t ends an episode.
    Future step t+k (k>=1) is valid iff t+k <= T-1 and no boundary in transitions t..t+k-1
    (same validity rule as build_future_window). Returns (near_mean, near_std, far_mean,
    far_std, valid_frac) with zeros where the respective window has no valid step; valid_frac
    covers the full horizon. Kept byte-identical to v34.1 so 'pooled' is a faithful control.
    """
    T, B, A = actions.shape
    valid = torch.ones(T, B, device=actions.device)
    sums = {
        "near": [torch.zeros(T, B, A, device=actions.device) for _ in range(2)],
        "far": [torch.zeros(T, B, A, device=actions.device) for _ in range(2)],
    }
    cnts = {
        "near": torch.zeros(T, B, device=actions.device),
        "far": torch.zeros(T, B, device=actions.device),
    }
    for k in range(1, horizon + 1):
        if k > T - 1:
            break
        valid = valid.clone()
        valid[: T - k] = valid[: T - k] * (1.0 - boundaries[k - 1 : T - 1])
        valid[T - k :] = 0.0  # window would run past the rollout
        m = valid.unsqueeze(-1)
        a_k = torch.zeros_like(actions)
        a_k[: T - k] = actions[k:]
        w = "near" if k <= NEAR_HORIZON else "far"
        sums[w][0] = sums[w][0] + m * a_k
        sums[w][1] = sums[w][1] + m * a_k.pow(2)
        cnts[w] = cnts[w] + valid
    outs = []
    for w in ("near", "far"):
        denom = cnts[w].clamp_min(1.0).unsqueeze(-1)
        mean = sums[w][0] / denom
        var = (sums[w][1] / denom - mean.pow(2)).clamp_min(0.0)
        std = var.sqrt()
        has = (cnts[w] > 0).float().unsqueeze(-1)
        outs.extend([mean * has, std * has])
    valid_frac = (cnts["near"] + cnts["far"]) / float(horizon)
    return outs[0], outs[1], outs[2], outs[3], valid_frac


# v38: log-spaced frequencies for the reward Fourier lift. Fixed/deterministic (no random
# features) so seeds and the compiled encoder stay clean; the transform is a function of the
# reward ALONE, so it is leak-neutral, but it spans 1 + 2F dims -> the H future-token embeddings
# are no longer colinear (fixes the rank-1 fp-conditioning of a bare scalar token).
REWARD_FOURIER_FREQS = (1.0, 2.0, 4.0, 8.0)


def reward_token_dim():
    return 1 + 2 * len(REWARD_FOURIER_FREQS)


def reward_fourier_features(scaled_r):
    """scaled_r: (..., 1) already divided by the frozen q_reward_scale. Returns (..., 1+2F) =
    [x, sin(f_i * x), cos(f_i * x)] for f_i in REWARD_FOURIER_FREQS. Runs EAGER in the rollout."""
    freqs = torch.as_tensor(REWARD_FOURIER_FREQS, dtype=scaled_r.dtype, device=scaled_r.device)
    xf = scaled_r * freqs                                         # (..., F)
    return torch.cat([scaled_r, torch.sin(xf), torch.cos(xf)], dim=-1)   # (..., 1+2F)


def build_future_window(values, boundaries, horizon):
    """v37: per-(t, env) the raw sequence of the next `horizon` per-step feature vectors
    (v37 feeds OUTCOME tokens = [normalized state-delta, scaled reward]; the fn is generic
    over the last dim), plus a per-step validity mask (the transformer encoder consumes both).

    values: (T, B, D); boundaries: (T, B) 1.0 where the transition at t ends an episode.
    Future step t+k (k=1..H) is valid iff t+k <= T-1 and no boundary in transitions
    t..t+k-1 (same rule the v19 pooled windows used). Invalid steps are zero-padded in the
    returned window and False in the mask; H is fixed so the encoder shape stays static.
    Returns (fut (T, B, H, D), fut_valid (T, B, H) bool, valid_frac (T, B)).
    """
    T, B, D = values.shape
    fut = torch.zeros(T, B, horizon, D, device=values.device)
    fut_valid = torch.zeros(T, B, horizon, dtype=torch.bool, device=values.device)
    valid = torch.ones(T, B, device=values.device)  # carried monotone gate (matches v19)
    for k in range(1, horizon + 1):
        if k > T - 1:
            break
        # extending the window by one step requires transition t+k-1 to be non-boundary
        valid = valid.clone()
        valid[: T - k] = valid[: T - k] * (1.0 - boundaries[k - 1 : T - 1])
        valid[T - k :] = 0.0  # window would run past the rollout
        v_k = torch.zeros_like(values)
        v_k[: T - k] = values[k:]
        fut[:, :, k - 1, :] = v_k * valid.unsqueeze(-1)
        fut_valid[:, :, k - 1] = valid > 0
    valid_frac = fut_valid.float().sum(dim=-1) / float(horizon)
    return fut, fut_valid, valid_frac


# ============================ v35: tiered torch.compile ============================
def _rng_snapshot():
    return (torch.get_rng_state(), torch.cuda.get_rng_state())


def _rng_restore(snap):
    torch.set_rng_state(snap[0])
    torch.cuda.set_rng_state(snap[1])


def _parity(eager_out, comp_out, rtol, atol):
    """Max abs diff across matching tensor outputs; ok iff all within tolerance."""
    mx = 0.0
    for x, y in zip(eager_out, comp_out):
        if isinstance(x, torch.Tensor):
            mx = max(mx, (x - y).abs().max().item())
            if not torch.allclose(x, y, rtol=rtol, atol=atol):
                return False, mx
    return True, mx


class CompiledTier:
    """Wraps an eager fn with a torch.compile'd twin behind a clean fallback ladder.

    On the FIRST call it runs eager then compiled from the SAME RNG state (so the internal
    Beta sampling matches — torch.compile reproduces eager RNG under a matched seed), checks
    allclose, and registers the tier as active. Any compile-setup, first-call, or later
    runtime exception permanently falls the tier back to eager (the run stays correct, only
    slower). A parity mismatch is treated as a failure (falls back) since a compiled path we
    cannot trust numerically is not worth the speed. The RNG state is restored before the
    compiled call so the run's RNG stream is exactly that of a single eager invocation.
    """

    def __init__(self, eager_fn, name, mode, enabled, active_registry, rtol=1e-3, atol=1e-4):
        self.eager = eager_fn
        self.name = name
        self.active = active_registry
        self.rtol, self.atol = rtol, atol
        self.compiled = None
        self.use_compiled = False
        self.checked = False
        if enabled:
            try:
                self.compiled = torch.compile(eager_fn, mode=mode)
                self.use_compiled = True
            except Exception as e:
                print(f"[compile] tier '{name}': torch.compile setup failed ({e}); eager fallback")

    def _fallback(self, args):
        self.use_compiled = False
        if self.name in self.active:
            self.active.remove(self.name)
        return self.eager(*args)

    def __call__(self, *args):
        if not self.use_compiled:
            return self.eager(*args)
        if not self.checked:
            self.checked = True
            try:
                snap = _rng_snapshot()
                eager_out = self.eager(*args)
                _rng_restore(snap)                      # compiled reruns from the same RNG
                comp_out = self.compiled(*args)
                ok, mx = _parity(eager_out, comp_out, self.rtol, self.atol)
                if not ok:
                    print(f"[compile] tier '{self.name}': PARITY MISMATCH (maxdiff={mx:.2e} > "
                          f"rtol {self.rtol}); eager fallback")
                    self.use_compiled = False
                    return eager_out
                print(f"[compile] tier '{self.name}': active (parity OK, maxdiff={mx:.2e})")
                self.active.append(self.name)
                return comp_out
            except Exception as e:
                print(f"[compile] tier '{self.name}': first-call failure ({e}); eager fallback")
                return self._fallback(args)
        try:
            return self.compiled(*args)
        except Exception as e:
            print(f"[compile] tier '{self.name}': runtime failure ({e}); eager fallback")
            return self._fallback(args)


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

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    teacher = HindsightTeacher(obs_dim, act_dim, args).to(device)
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- v34: twin Q nets + polyak targets (independent inits) + separate optimizer ---
    q1 = QNet(obs_dim, act_dim).to(device)
    q2 = QNet(obs_dim, act_dim).to(device)
    q1_target = QNet(obs_dim, act_dim).to(device)
    q2_target = QNet(obs_dim, act_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.q_lr, eps=1e-5)
    q_params = list(q1.parameters()) + list(q2.parameters())
    # Param lists for a launch-cheap _foreach polyak update (fewer kernel launches than a
    # Python zip in the 32768x TD loop; numerically identical to tp = tp*(1-tau) + tau*p).
    q1_p_list, q1_tp_list = list(q1.parameters()), list(q1_target.parameters())
    q2_p_list, q2_tp_list = list(q2.parameters()), list(q2_target.parameters())

    # v35: registry of compile tiers that actually engaged (parity-passed). "teacher" is
    # driven by the encoder's own compile state (HindsightTeacher); "td"/"student" register
    # on their first successful CompiledTier call.
    compile_tiers_active = []

    # Env action bounds in the ENV action space (Beta z in (0,1) maps to [low, high]).
    q_act_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=device)
    q_act_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=device)

    OBS_NORM_EPS = 1e-8   # NormalizeObservation epsilon (default)
    OBS_CLIP = 10.0       # TransformObservation clip constant

    def pooled_obs_rms():
        """Pooled (over the 16 identical envs) current obs_rms mean/var as device tensors."""
        means = np.stack([envs.envs[i].get_wrapper_attr("obs_rms").mean for i in range(args.num_envs)])
        varis = np.stack([envs.envs[i].get_wrapper_attr("obs_rms").var for i in range(args.num_envs)])
        m = torch.as_tensor(means.mean(0), dtype=torch.float32, device=device)
        v = torch.as_tensor(varis.mean(0), dtype=torch.float32, device=device)
        return m, v

    def obs_unnormalize(norm_obs, mean, var):
        return norm_obs * torch.sqrt(var + OBS_NORM_EPS) + mean

    def obs_renormalize(raw_obs, mean, var):
        return ((raw_obs - mean) / torch.sqrt(var + OBS_NORM_EPS)).clamp(-OBS_CLIP, OBS_CLIP)

    # Transition replay ring (GPU), storing RAW obs/next_obs + executed action + raw reward
    # + terminated flag. Wrap-aware writes (capacity need not divide batch_size).
    # v35: allocated LAZILY — q_onpol_only skips the replay TD loop entirely, so the 1M-row
    # tensors are never created in that mode.
    rep_cap = args.replay_capacity
    if not args.q_onpol_only:
        rep_obs = torch.zeros((rep_cap, obs_dim), device=device)
        rep_next_obs = torch.zeros((rep_cap, obs_dim), device=device)
        rep_act = torch.zeros((rep_cap, act_dim), device=device)
        rep_rew = torch.zeros((rep_cap,), device=device)
        rep_term = torch.zeros((rep_cap,), device=device)
    rep_ptr = 0
    rep_filled = 0

    # Frozen raw-reward scale for Q targets (running std over the warmup, then frozen).
    q_rew_sum = 0.0
    q_rew_sumsq = 0.0
    q_rew_count = 0
    q_reward_scale = 1.0
    q_reward_frozen = False

    # v35 (vestigial-audit): the v34.1 q_scale EMA and its full-batch min-Q sweep are gone.
    # q_scale cancelled out of the only actuator (the DPG sensor divided by q_scale and the
    # loss-path rescale multiplied it back). The sensor now measures the ACTUAL loss path
    # directly (divides by qadv_scale), so debug/qgrad_ratio IS the loss-path ratio.

    # v34.1: EMA of mean|action-advantage| (proposal vs realized), the DPG loss normalizer.
    # Carried across iterations and used FIXED within each iteration's student epochs
    # (updated afterward), so the divisor never depends on the current minibatch. Floored
    # at args.qadv_floor. Init at 1.0 to match q_scale's conservative warmup.
    qadv_ema = 1.0
    qadv_scale = max(qadv_ema, args.qadv_floor)
    q_dpg_gap = 0.0
    lam_priv = args.priv_lam_init   # v37: dual variable for the actor privilege budget
    priv_kl_mean = 0.0
    nll_gap = 0.0                   # v38: teacher_nll(null) - teacher_nll(full) leak alarm

    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, device
    )
    hl_support_cpu = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, torch.device("cpu")
    )

    def value_logits_to_scalar(logits):
        return hl_support.to_scalar(logits)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # ---------------- v35: hot-loop bodies as standalone (compilable) functions ----------------
    def student_env_action(norm_obs, reparam):
        """Student Beta over norm_obs -> env-space action ([-1,1]), same affine map as the
        rollout. reparam=True -> differentiable rsample; else detached sample."""
        a_feat, _ = agent._trunks(norm_obs)
        alpha = 1.0 + F.softplus(agent.actor_alpha_head(a_feat))
        beta = 1.0 + F.softplus(agent.actor_beta_head(a_feat))
        dist = Beta(alpha, beta)
        z = (dist.rsample() if reparam else dist.sample()).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        return q_act_low + (q_act_high - q_act_low) * z

    def td_forward(s_obs, s_next, s_act, s_rew, s_term):
        """One twin-Q TD forward+loss (static q_batch shape). Replay sampling/indexing and the
        optimizer/polyak/clip stay OUTSIDE; this is the graphed region. gamma is a constant;
        the reward is pre-scaled outside so no per-iteration scalar is baked in."""
        with torch.no_grad():
            a_next = student_env_action(s_next, reparam=False)
            q_next = torch.min(q1_target(s_next, a_next), q2_target(s_next, a_next))
            y = s_rew + args.gamma * (1.0 - s_term) * q_next
        q1_pred = q1(s_obs, s_act)
        q2_pred = q2(s_obs, s_act)
        q_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
        return q_loss, q1_pred.detach().mean()

    def student_forward(s_obs_mb, s_z_mb, s_act_mb, s_lp_mb, s_tp_mb, s_tm_mb,
                        t_alpha_d, t_beta_d, qadv_scale_t):
        """Student minibatch forward: distill KL + DPG qterm + value CE + telemetry (static
        minibatch shape). qadv_scale is passed as a device scalar tensor (changes per
        iteration -> must not be baked into the graph). The backward/grad-clip surgery and
        optimizer.step stay OUTSIDE so Adam state and the two-phase clip are untouched."""
        _, _, newlogprob, entropy, value_logits, s_alpha, s_beta = agent.get_action_and_value(
            s_obs_mb, s_z_mb
        )
        logratio = newlogprob - s_lp_mb
        ratio = logratio.exp()
        approx_kl = ((ratio - 1) - logratio).mean().detach()
        kl_dims = beta_kl_per_dim(t_alpha_d, t_beta_d, s_alpha, s_beta).clamp_min(0.0)
        distill_loss = kl_dims.clamp(max=args.distill_kl_clip).sum(-1).mean()
        distill_kl_g = kl_dims.sum(-1).mean().detach()
        clipfrac_g = (kl_dims > args.distill_kl_clip).float().mean().detach()
        s_dist_rs = Beta(s_alpha, s_beta)
        z_rs = s_dist_rs.rsample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        a_rs = q_act_low + (q_act_high - q_act_low) * z_rs
        q1_rs, q2_rs = q1(s_obs_mb, a_rs), q2(s_obs_mb, a_rs)
        min_rs = torch.min(q1_rs, q2_rs)   # keeps grad -> DPG path through the actor
        with torch.no_grad():
            gap_rs = (q1_rs - q2_rs).abs()
            gm_rs = gap_rs.mean()
            w_rs = gm_rs / (gap_rs + gm_rs + 1e-8)
            q_real = torch.min(q1(s_obs_mb, s_act_mb), q2(s_obs_mb, s_act_mb))
            qadv = min_rs.detach() - q_real
            qadv_abs_g = qadv.abs().mean()
            qadv_sig_g = qadv.mean()
            gate_g = w_rs.mean()
        qterm = (w_rs * (-min_rs / qadv_scale_t)).mean()
        ent_bonus = (entropy / act_dim).mean()
        value_log_probs = torch.log_softmax(value_logits, dim=-1)
        value_ce = -(s_tp_mb * value_log_probs).sum(dim=-1)
        v_loss = (value_ce * s_tm_mb).sum(dim=-1).mean()
        entropy_mean = entropy.mean()
        return (v_loss, distill_loss, qterm, ent_bonus, entropy_mean, approx_kl,
                distill_kl_g, clipfrac_g, qadv_abs_g, qadv_sig_g, gate_g)

    # reduce-overhead (CUDA graphs) for the 32768x/iter TD step — the dominant launch cost.
    td_step = CompiledTier(td_forward, "td", "reduce-overhead", args.compile_update, compile_tiers_active)
    # default (fusion, no CUDA graphs) for the student forward: its two-phase clip does a
    # retain_graph double-backward + in-place grad surgery, which is safer without graph
    # capture; the win is kernel fusion of the ThinkTrunk/MoE forward.
    student_step = CompiledTier(student_forward, "student", "default", args.compile_update, compile_tiers_active)

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
                action, z, logprob, ent, value_logits, _, _ = agent.get_action_and_value(next_obs)
                values[step] = value_logits_to_scalar(value_logits[:, 0])
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
            next_transition_value_logits = agent.get_value(
                next_obses.reshape((-1,) + envs.single_observation_space.shape)
            )[:, 0]
            next_transition_values = value_logits_to_scalar(next_transition_value_logits).reshape(
                args.num_steps, args.num_envs
            )
            # --- v30: privileged-information mismatch bonus (pre-GAE) ---
            # v37: build only the phi channels this mode needs (they do not depend on
            # advantages). pooled = v34.1 near/far ACTION mean-std + valid_frac (intent);
            # attn = transformer over per-step OUTCOME tokens [normalized state-delta, scaled
            # reward] (where the trajectory went + what it earned). vdis uses the FULL phi.
            if teacher.needs_pooled:
                _nm, _ns, _fm, _fs, pooled_valid_frac = build_future_features(
                    actions, transition_boundaries, args.hindsight_horizon
                )
                pooled_roll = torch.cat([_nm, _ns, _fm, _fs, pooled_valid_frac.unsqueeze(-1)], dim=-1)
                pooled_flat = pooled_roll.reshape(-1, pooled_roll.shape[-1])
            else:
                pooled_flat = None
            if teacher.needs_attn:
                # v38 per-step tokens (--token-content). reward: [scaled reward] only — the
                # low-mutual-information-with-a_t channel (see design law). outcome (v37):
                # [normalized state-delta, scaled reward]. reward is RAW, scaled by the frozen
                # q_reward_scale (matches the Q/return units); the state-delta lives in NORMALIZED
                # obs space (obs and next_obses both normalized, shared mean cancels in the diff).
                # next_obses already carries final_observation at truncation tails, and the
                # future-window validity mask drops steps past an episode boundary.
                scaled_r = (rewards / q_reward_scale).unsqueeze(-1)
                if args.token_content == "reward":
                    step_tokens = reward_fourier_features(scaled_r)   # (T, B, 1+2F)
                else:
                    step_tokens = torch.cat([next_obses - obs, scaled_r], dim=-1)
                fut_tokens, fut_valid, attn_valid_frac = build_future_window(
                    step_tokens, transition_boundaries, args.hindsight_horizon
                )
                token_dim = step_tokens.shape[-1]
                ft_flat = fut_tokens.reshape(-1, args.hindsight_horizon, token_dim)
                fv_flat = fut_valid.reshape(-1, args.hindsight_horizon)
            else:
                ft_flat = fv_flat = None
            fut_valid_frac = attn_valid_frac if teacher.needs_attn else pooled_valid_frac
            obs_flat_roll = obs.reshape(-1, obs_dim)
            vt_chunks = []
            for start in range(0, obs_flat_roll.shape[0], args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                summ = teacher.encode(ft_flat[sl], fv_flat[sl]) if teacher.needs_attn else None
                pooled_sl = pooled_flat[sl] if teacher.needs_pooled else None
                vt_chunks.append(
                    hl_support.to_scalar(teacher.critic_logits_from(obs_flat_roll[sl], pooled_sl, summ))
                )
            v_teacher_roll = torch.cat(vt_chunks).reshape(args.num_steps, args.num_envs)
            vdis = (v_teacher_roll - values).abs()
            vdis_scale = args.vdis_coef * min(1.0, (iteration - 1) / max(1, args.vdis_warmup_iters))
            # v35 (vestigial-audit): when vdis_scale == 0 (vdis off, or still in warmup) the
            # bonus term is exactly 0, so shaped_rewards == rewards and the two GAE sweeps are
            # byte-identical. Skip the second sweep and alias in that case.
            vdis_active = vdis_scale != 0.0
            shaped_rewards = rewards + vdis_scale * vdis / (vdis.std() + 1e-8) if vdis_active else rewards

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = shaped_rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values

            if vdis_active:
                # Second GAE pass on RAW env rewards (NOT the vdis-shaped rewards) ->
                # raw-reward lambda-returns. Used for Q's q_ev and the on-policy hardening
                # target; converted to Q units by the frozen q_reward_scale downstream.
                # CAVEAT: the immediate rewards are raw, but the GAE bootstrap uses the
                # student `values` (a critic trained on the vdis-SHAPED return), so the
                # lambda tail carries a small, decaying trace of the vdis bonus. The TD-block
                # target (rep_rew) is genuinely raw; only this multi-step target/diag is not.
                advantages_raw = torch.zeros_like(rewards).to(device)
                lastgaelam_raw = 0
                for t in reversed(range(args.num_steps)):
                    bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                    lambda_nonterminal = 1.0 - transition_boundaries[t]
                    delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                    advantages_raw[t] = lastgaelam_raw = (
                        delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam_raw
                    )
                returns_raw = advantages_raw + values
            else:
                advantages_raw = advantages
                returns_raw = returns

            # Student-critic MTP targets (unchanged from the base).
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
            target_probs = hl_support_cpu.project(return_mtp.detach().cpu())

            # --- HOPSD privileged context --- (windows already built above for v30)
            adv_z = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # batch scope
            g = (returns - returns.mean()) / (returns.std() + 1e-8)                # batch scope
            # v12.2: winsorize FIRST so outliers cannot eat the KL budget (v12's
            # failure mode: the dual inflated temp to protect the target from a few
            # extreme advantages, flattening the tilt for the bulk). Then the MPO
            # E-step dual — geometric bisection for the temp whose softmax tilt over
            # the CLIPPED advantages sits at tilt_eps nats from uniform (KL is
            # monotone decreasing in temp; 25 softmaxes over the batch).
            adv_c = adv_z.clamp(-args.adv_clip, args.adv_clip)
            a_flat = adv_c.reshape(-1)
            n_samp = float(a_flat.numel())
            if args.auto_temp:
                lo, hi = 0.02, 50.0
                for _ in range(25):
                    mid = (lo * hi) ** 0.5
                    p = torch.softmax(a_flat / mid, dim=0)
                    tilt_kl = (p * (p * n_samp).clamp_min(1e-12).log()).sum().item()
                    if tilt_kl > args.tilt_eps:
                        lo = mid  # too sharp -> need higher temp
                    else:
                        hi = mid
                temp_now = (lo * hi) ** 0.5
            else:
                temp_now = args.awr_temp
            # N*softmax == exp-then-mean-normalize, but numerically stable; no weight
            # clamp — winsorization already bounds the max weight, so realized KL
            # equals the target identically.
            awr_w = (torch.softmax(a_flat / temp_now, dim=0) * n_samp).reshape(adv_z.shape)
            # v35: teacher inputs are (obs, future-action window, validity mask, optional g);
            # the encoder runs per-minibatch inside the update so gradient flows into it.

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()
        # v36: already-flattened phi channels (None for the channel this mode doesn't use).
        b_future_tokens = ft_flat                           # (T*B, H, token_dim) or None (pooled mode)
        b_future_valid = fv_flat                            # (T*B, H) or None
        b_pooled = pooled_flat                              # (T*B, 4A+1) or None (attn mode)
        b_g = g.reshape(-1, 1)                              # teacher actor g feature (if enabled)
        b_awr_w = awr_w.reshape(-1)
        b_actions = actions.reshape(-1, act_dim)            # v34: executed env actions in [-1,1]
        b_returns_raw = returns_raw.reshape(-1)             # v34: raw-reward lambda-returns

        # ================= v34: off-policy twin-Q training =================
        # Snapshot the CURRENT pooled obs_rms once per training block: use it BOTH to
        # recover raw obs for the transitions stored THIS iter (exact round-trip) and to
        # re-normalize sampled (older) transitions to the current stats at train time.
        with torch.no_grad():
            b_rew_raw = rewards.reshape(-1)                 # raw env reward (normalize_reward=False)
            b_term = transition_terminations.reshape(-1)    # terminated only (bootstrap thru truncation)

            # v35: replay bookkeeping only when the TD loop is active. q_onpol_only never
            # allocates or touches the replay tensors.
            if not args.q_onpol_only:
                # Snapshot the CURRENT pooled obs_rms: recover raw obs for this iter's
                # transitions (exact round-trip) and re-normalize older samples at train time.
                rms_mean, rms_var = pooled_obs_rms()
                raw_obs_now = obs_unnormalize(b_obs, rms_mean, rms_var)
                raw_next_obs_now = obs_unnormalize(
                    next_obses.reshape(-1, obs_dim), rms_mean, rms_var
                )
                # Ring write (wrap-aware; capacity need not divide batch_size).
                idx = (rep_ptr + torch.arange(args.batch_size, device=device)) % rep_cap
                rep_obs[idx] = raw_obs_now
                rep_next_obs[idx] = raw_next_obs_now
                rep_act[idx] = b_actions
                rep_rew[idx] = b_rew_raw
                rep_term[idx] = b_term
                rep_ptr = int((rep_ptr + args.batch_size) % rep_cap)
                rep_filled = min(rep_filled + args.batch_size, rep_cap)

            # Frozen raw-reward scale: running std over the first vdis_warmup_iters, then
            # frozen forever (a growing return-std would drift the TD/on-policy target).
            if not q_reward_frozen:
                q_rew_sum += float(b_rew_raw.sum().item())
                q_rew_sumsq += float((b_rew_raw * b_rew_raw).sum().item())
                q_rew_count += int(b_rew_raw.numel())
                _mean = q_rew_sum / max(1, q_rew_count)
                _var = max(q_rew_sumsq / max(1, q_rew_count) - _mean * _mean, 0.0)
                q_reward_scale = max(_var ** 0.5, 1e-3)
                if iteration >= args.vdis_warmup_iters:
                    q_reward_frozen = True

        # --- TD block: q_updates_per_iter off-policy updates (skipped when q_onpol_only) ---
        # Replay sampling/indexing/renormalize stay OUTSIDE the compiled td_step; only the
        # forward+loss is graphed. Telemetry accumulates on-GPU (no per-step .item() sync).
        q_loss_accum = torch.zeros((), device=device)
        q_val_accum = torch.zeros((), device=device)
        if not args.q_onpol_only:
            for _qu in range(args.q_updates_per_iter):
                s_idx = torch.randint(0, rep_filled, (args.q_batch,), device=device)
                s_obs = obs_renormalize(rep_obs[s_idx], rms_mean, rms_var)
                s_next = obs_renormalize(rep_next_obs[s_idx], rms_mean, rms_var)
                s_act = rep_act[s_idx]
                s_rew = rep_rew[s_idx] / q_reward_scale
                s_term = rep_term[s_idx]
                q_loss, q1_mean = td_step(s_obs, s_next, s_act, s_rew, s_term)
                q_optimizer.zero_grad(set_to_none=True)
                q_loss.backward()
                nn.utils.clip_grad_norm_(q_params, 0.5)
                q_optimizer.step()
                with torch.no_grad():
                    # polyak: tp = tp*(1-tau) + tau*p, batched over params (fewer launches).
                    torch._foreach_mul_(q1_tp_list, 1.0 - args.q_tau)
                    torch._foreach_add_(q1_tp_list, q1_p_list, alpha=args.q_tau)
                    torch._foreach_mul_(q2_tp_list, 1.0 - args.q_tau)
                    torch._foreach_add_(q2_tp_list, q2_p_list, alpha=args.q_tau)
                    q_loss_accum += q_loss.detach()
                    q_val_accum += q1_mean

        # --- On-policy lambda-return hardening: regress BOTH Qs to raw-return targets on
        # the fresh rollout, INDEPENDENT shuffles per twin (do not couple the twins). One
        # epoch alongside the TD block normally; q_onpol_epochs epochs (the sole Q signal)
        # when q_onpol_only, each with fresh independent per-twin shuffles. ---
        q_targets_onpol = b_returns_raw / q_reward_scale
        n_hard_epochs = args.q_onpol_epochs if args.q_onpol_only else 1
        for _he in range(n_hard_epochs):
            onpol_inds1 = np.random.permutation(args.batch_size)
            onpol_inds2 = np.random.permutation(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb1 = onpol_inds1[start : start + args.minibatch_size]
                mb2 = onpol_inds2[start : start + args.minibatch_size]
                q1_op = q1(b_obs[mb1], b_actions[mb1])
                q2_op = q2(b_obs[mb2], b_actions[mb2])
                onpol_loss = args.q_onpol_coef * (
                    F.mse_loss(q1_op, q_targets_onpol[mb1]) + F.mse_loss(q2_op, q_targets_onpol[mb2])
                )
                q_optimizer.zero_grad(set_to_none=True)
                onpol_loss.backward()
                nn.utils.clip_grad_norm_(q_params, 0.5)
                q_optimizer.step()

        # v35: the q_scale EMA + its full-batch min-Q sweep are gone (cancelled out of
        # dpg_atten). Q magnitude is still logged as debug/q_mean from the telemetry chunk
        # sweep further below.
        # Freeze Q params during the student epochs: DPG grads must flow only to the
        # actor (through a_rsample), never step or accumulate into the Q nets.
        for p in q_params:
            p.requires_grad_(False)

        # --- qgrad telemetry: DPG vs distill gradient alignment on the actor params
        # (once per iteration, one minibatch; autograd.grad so nothing pollutes .grad). ---
        _tel = slice(0, args.minibatch_size)
        _, _, _, _, _, s_a_t, s_b_t = agent.get_action_and_value(b_obs[_tel], b_latent_zs[_tel])
        with torch.no_grad():
            _summ_tel = teacher.encode(b_future_tokens[_tel], b_future_valid[_tel]) if teacher.needs_attn else None
            _pooled_tel = b_pooled[_tel] if teacher.needs_pooled else None
            _ta, _tb = teacher.actor_params_from(
                b_obs[_tel], _pooled_tel, _summ_tel, b_g[_tel] if args.teacher_sees_g else None
            )
        _kl = beta_kl_per_dim(_ta, _tb, s_a_t, s_b_t).clamp_min(0.0)
        _distill_t = args.distill_coef * _kl.clamp(max=args.distill_kl_clip).sum(-1).mean()
        _z = Beta(s_a_t, s_b_t).rsample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        _a = q_act_low + (q_act_high - q_act_low) * _z
        _q1t, _q2t = q1(b_obs[_tel], _a), q2(b_obs[_tel], _a)
        _minq = torch.min(_q1t, _q2t)
        with torch.no_grad():
            _gap = (_q1t - _q2t).abs()
            _gm = _gap.mean()
            _w = _gm / (_gap + _gm + 1e-8)
        # v35: divide by qadv_scale (the ACTUAL DPG loss normalizer) so qgrad_ratio directly
        # measures the loss-path DPG/distill grad-norm ratio — no q_scale rescale needed.
        _qterm_t = args.q_coef * (_w * (-_minq / qadv_scale)).mean()
        _gd = torch.autograd.grad(_distill_t, actor_params, retain_graph=True, allow_unused=True)
        _gq = torch.autograd.grad(_qterm_t, actor_params, retain_graph=False, allow_unused=True)

        def _flat(gs):
            return torch.cat(
                [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, actor_params)]
            )

        _fd, _fq = _flat(_gd), _flat(_gq)
        _nd, _nq = _fd.norm(), _fq.norm()
        # Loss-path ratio (pre-attenuation): DPG grad norm / distill grad norm. Ceiling-only
        # attenuation for THIS iteration's epochs — sensor stays pure, actuator only reduces.
        qgrad_ratio = (_nq / (_nd + 1e-8)).item()
        dpg_atten = min(1.0, args.qgrad_ceiling / max(qgrad_ratio, 1e-8))
        qgrad_cos = (torch.dot(_fd, _fq) / ((_nd * _nq) + 1e-8)).item()
        arsample_std = _a.std().item()
        q_twin_gap_rsample = _gap.mean().item()
        q_gate_mean = _w.mean().item()

        # --- Q value diagnostics (no grad) ---
        with torch.no_grad():
            q_on_chunks, gap_chunks, absq_chunks = [], [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                _c1, _c2 = q1(b_obs[sl], b_actions[sl]), q2(b_obs[sl], b_actions[sl])
                q_on_chunks.append(torch.min(_c1, _c2))
                gap_chunks.append((_c1 - _c2).abs())
                absq_chunks.append(torch.min(_c1, _c2).abs())
            q_on = torch.cat(q_on_chunks)
            q_tgt = b_returns_raw / q_reward_scale
            _num = (q_tgt - q_on).var(unbiased=False)
            _den = q_tgt.var(unbiased=False)
            q_ev = float("nan") if _den.item() == 0 else (1.0 - (_num / _den)).item()
            q_mean = q_on.mean().item() * q_reward_scale
            q_twin_gap = (torch.cat(gap_chunks).mean() / (torch.cat(absq_chunks).mean() + 1e-8)).item()

            # replay-side EV vs the one-step TD target it actually optimizes (drift telemetry).
            # Undefined without a replay buffer (q_onpol_only) -> report NaN.
            if args.q_onpol_only:
                q_ev_replay = float("nan")
            else:
                _r = torch.randint(0, rep_filled, (args.q_batch,), device=device)
                _so = obs_renormalize(rep_obs[_r], rms_mean, rms_var)
                _sn = obs_renormalize(rep_next_obs[_r], rms_mean, rms_var)
                _an = student_env_action(_sn, reparam=False)
                _y = rep_rew[_r] / q_reward_scale + args.gamma * (1.0 - rep_term[_r]) * torch.min(
                    q1_target(_sn, _an), q2_target(_sn, _an)
                )
                _pred = torch.min(q1(_so, rep_act[_r]), q2(_so, rep_act[_r]))
                _rden = _y.var(unbiased=False)
                q_ev_replay = float("nan") if _rden.item() == 0 else (1.0 - ((_y - _pred).var(unbiased=False) / _rden)).item()
        if args.q_onpol_only:
            q_loss_mean = float("nan")
            q_value_mean = float("nan")
        else:
            q_loss_mean = (q_loss_accum / args.q_updates_per_iter).item()
            q_value_mean = (q_val_accum / args.q_updates_per_iter).item()
        # ================= end v34 twin-Q training =================

        b_inds = np.arange(args.batch_size)
        # On-GPU telemetry accumulators (summed per minibatch, read once per iteration to
        # avoid per-minibatch host syncs). Derived means are computed after the epoch loop.
        acc_teacher_nll = torch.zeros((), device=device)
        acc_distill_kl = torch.zeros((), device=device)
        acc_clipfrac = torch.zeros((), device=device)
        acc_qterm = torch.zeros((), device=device)
        acc_gate = torch.zeros((), device=device)
        acc_qadv_abs = torch.zeros((), device=device)
        acc_qadv_sig = torch.zeros((), device=device)
        acc_phi_drop = torch.zeros((), device=device)   # v36: realized summary-dropout fraction
        acc_priv_kl = torch.zeros((), device=device)    # v37: pre-penalty priv KL(full||null)
        acc_nll_full = torch.zeros((), device=device)   # v38: plain NLL of a_t at full phi
        acc_nll_null = torch.zeros((), device=device)   # v38: plain NLL of a_t at null phi
        teacher_mb_count = 0
        student_mb_count = 0
        last_v_loss = last_pg_loss = last_entropy = last_approx_kl = last_t_v_loss = None
        # qadv_scale as a device scalar so the compiled student_forward does not bake in the
        # per-iteration float (which would force a recompile every iteration).
        qadv_scale_t = torch.as_tensor(qadv_scale, dtype=torch.float32, device=device)
        # The target_kl leash freezes only the STUDENT; the teacher always trains its
        # full epochs (weighted MLE is off-policy safe, and a lagging teacher is worst
        # exactly when distillation drifts the student fast).
        student_stopped = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # ---- teacher update (own optimizer; nothing here touches the student) ----
                # v37: build the phi channels this mode needs. Encode the future OUTCOME tokens
                # ONCE per minibatch (attn/hybrid); the summary feeds both heads so both losses'
                # gradient flows into the shared encoder. (Teacher stays eager; encoder compiled
                # via --compile-teacher.) pooled carries ACTION intent (hybrid).
                mb_obs = b_obs[mb_inds]
                t_summary = teacher.encode(b_future_tokens[mb_inds], b_future_valid[mb_inds]) if teacher.needs_attn else None
                pooled_mb = b_pooled[mb_inds] if teacher.needs_pooled else None
                t_g = b_g[mb_inds] if args.teacher_sees_g else None
                # v36 dropout regularizer (default OFF in v37): per-row summary->null mask on the
                # ACTOR training forward only. Superseded by the KL budget below; kept as ablation.
                if teacher.needs_attn and args.phi_dropout > 0.0:
                    drop_mask = torch.rand(mb_obs.shape[0], 1, device=device) < args.phi_dropout
                    acc_phi_drop += drop_mask.float().mean()
                else:
                    drop_mask = None

                # FULL-phi actor policy: the distillation target and the budget's constrained
                # side. Always computed (with grad). The training policy equals it unless the
                # (ablation) dropout mask is active.
                t_alpha_full, t_beta_full = teacher.actor_params_from(mb_obs, pooled_mb, t_summary, t_g, None)
                if drop_mask is not None:
                    t_alpha, t_beta = teacher.actor_params_from(mb_obs, pooled_mb, t_summary, t_g, drop_mask)
                else:
                    t_alpha, t_beta = t_alpha_full, t_beta_full
                t_dist = Beta(t_alpha, t_beta)
                t_nll = -t_dist.log_prob(b_latent_zs[mb_inds]).sum(-1)
                awr_nll = (b_awr_w[mb_inds] * t_nll).mean()   # pure AWR loss (telemetry: teacher_nll)
                teacher_actor_loss = awr_nll

                # v37 ACTOR PRIVILEGE BUDGET (attn/hybrid): penalize how far the FULL-phi actor
                # sits from the teacher's own NULL-phi actor (summary dropped to the learned null;
                # pooled + g kept, so only the summary channel is governed). priv_kl is per-state
                # Beta KL(full || null) summed over dims; the null reference is detached (a
                # trust-region anchor, as in the tilt E-step) so the penalty moves ONLY the
                # full-phi policy toward followability, not the anchor toward the full policy.
                # lam_priv is a dual variable driven toward priv_eps after the epoch loop.
                if teacher.needs_attn:
                    with torch.no_grad():
                        _null_mask = torch.ones(mb_obs.shape[0], 1, dtype=torch.bool, device=device)
                        t_alpha_null, t_beta_null = teacher.actor_params_from(mb_obs, pooled_mb, t_summary, t_g, _null_mask)
                    priv_kl = beta_kl_per_dim(t_alpha_full, t_beta_full, t_alpha_null, t_beta_null).clamp_min(0.0).sum(-1).mean()
                    teacher_actor_loss = teacher_actor_loss + lam_priv * priv_kl
                    acc_priv_kl += priv_kl.detach()
                    # v38 LEAK ALARM: plain (unweighted) NLL of the realized action at full vs
                    # null phi. nll_gap = null - full estimates I(phi; a_t | s_t) -- how much the
                    # privilege channel predicts the action. Small+stable = healthy (reward tokens);
                    # a large/growing gap is the behavior-clone leak that killed v35 (actions) and
                    # v37 (inverse-dynamics state-deltas).
                    with torch.no_grad():
                        _z = b_latent_zs[mb_inds]
                        acc_nll_full += (-Beta(t_alpha_full, t_beta_full).log_prob(_z).sum(-1)).mean()
                        acc_nll_null += (-Beta(t_alpha_null, t_beta_null).log_prob(_z).sum(-1)).mean()

                # Distill target: FULL phi, detached, from the pre-step teacher (v35 timing).
                t_alpha_q, t_beta_q = t_alpha_full, t_beta_full
                # CRITIC is UNBUDGETED: full phi always (drop_mask=None) — privilege is free for
                # value estimation; followability constrains only the actor channel.
                t_value_logits = teacher.critic_logits_from(mb_obs, pooled_mb, t_summary, None)
                t_target = b_target_probs[mb_inds, 0].to(device=device, non_blocking=True)
                t_v_loss = -(t_target * torch.log_softmax(t_value_logits, dim=-1)).sum(-1).mean()
                teacher_loss = teacher_actor_loss + args.teacher_vf_coef * t_v_loss
                teacher_optimizer.zero_grad(set_to_none=True)
                teacher_loss.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(), args.teacher_grad_clip)
                teacher_optimizer.step()
                acc_teacher_nll += awr_nll.detach()   # pre-budget AWR loss (comparable to v36)
                teacher_mb_count += 1
                last_t_v_loss = t_v_loss.detach()

                if student_stopped:
                    continue

                # ---- student update: dense clipped forward-KL distillation + DPG + critic CE.
                # The forward+loss is the compiled student_step (fallback-guarded); the
                # backward/two-phase-clip surgery and optimizer.step stay eager. The value CE
                # target/mask are gathered + moved to device HERE (out of the compiled region).
                tp_mb = b_target_probs[mb_inds].to(device=device, non_blocking=True)
                tm_mb = b_target_mask[mb_inds].to(device=device, dtype=torch.float32, non_blocking=True)
                (v_loss, distill_loss, qterm, ent_bonus, entropy_mean, approx_kl,
                 distill_kl_g, clipfrac_g, qadv_abs_g, qadv_sig_g, gate_g) = student_step(
                    b_obs[mb_inds], b_latent_zs[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds],
                    tp_mb, tm_mb, t_alpha_q.detach(), t_beta_q.detach(), qadv_scale_t,
                )
                pg_loss = args.distill_coef * distill_loss

                if args.separate_grad_clip:
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss + args.q_coef * dpg_atten * qterm - args.ent_coef * ent_bonus).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    for p, grad in value_grads:
                        p.grad = grad if p.grad is None else p.grad + grad
                    optimizer.step()
                else:
                    loss = (
                        pg_loss
                        + args.q_coef * dpg_atten * qterm
                        - args.ent_coef * ent_bonus
                        + v_loss * args.vf_coef
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                acc_distill_kl += distill_kl_g
                acc_clipfrac += clipfrac_g
                acc_qterm += qterm.detach()
                acc_gate += gate_g
                acc_qadv_abs += qadv_abs_g
                acc_qadv_sig += qadv_sig_g
                student_mb_count += 1
                last_v_loss = v_loss.detach()
                last_pg_loss = pg_loss.detach()
                last_entropy = entropy_mean.detach()
                last_approx_kl = approx_kl   # already detached in student_forward

            if (
                args.target_kl is not None
                and not student_stopped
                and last_approx_kl is not None
                and last_approx_kl.item() > args.target_kl
            ):
                student_stopped = True

        # Re-enable Q grads for the next iteration's Q training block.
        for p in q_params:
            p.requires_grad_(True)

        # Read the GPU telemetry accumulators once (host sync here, not per minibatch).
        _sden = max(student_mb_count, 1)
        teacher_nll_mean = (acc_teacher_nll / max(teacher_mb_count, 1)).item()
        distill_kl_mean = (acc_distill_kl / _sden).item()
        distill_clipfrac_mean = (acc_clipfrac / _sden).item()
        actor_q_term_mean = args.q_coef * dpg_atten * (acc_qterm / _sden).item()
        q_gate_mean_epoch = (acc_gate / _sden).item()
        phi_dropout_frac = (acc_phi_drop / max(teacher_mb_count, 1)).item()

        # v37: dual ascent on the actor privilege budget. priv_kl is measured pre-penalty at
        # full-vs-null phi; lam_priv rises when priv_kl > priv_eps (privilege moved the policy
        # too far to follow) and decays when under budget, driving priv_kl toward priv_eps.
        # Only meaningful with a summary channel (attn/hybrid); pooled leaves lam_priv inert.
        if teacher.needs_attn and teacher_mb_count > 0:
            priv_kl_mean = (acc_priv_kl / teacher_mb_count).item()
            lam_priv = float(np.clip(
                lam_priv * np.exp(args.priv_eta * (priv_kl_mean / args.priv_eps - 1.0)),
                args.priv_lam_min, args.priv_lam_max,
            ))
            # v38 leak alarm: privilege predictivity of the realized action.
            nll_gap = ((acc_nll_null - acc_nll_full) / teacher_mb_count).item()

        # v38 saturation warning: teacher_nll << base (~-3.5) means the teacher has become a
        # behavior clone off the privilege channel (the v35/v37 death mode). Telemetry only.
        if teacher_nll_mean < -5.0:
            print(f"[v38 WARN] teacher_nll={teacher_nll_mean:.2f} < -5.0 (saturation/leak); "
                  f"nll_gap={nll_gap:.2f} @ step {global_step}")

        # v34.1: update the DPG normalizer AFTER the epochs (this iteration's loss used the
        # value carried from the prior iteration, so the divisor never depends on the
        # current update). Floor is load-bearing: as the policy improves and qadv -> 0, the
        # normalizer rides the floor and the DPG term fades gracefully instead of amplifying
        # a noise direction. q_dpg_gap = signed mean(qadv): decaying to 0 = DPG exhaustion.
        if student_mb_count > 0:
            qadv_ema = 0.99 * qadv_ema + 0.01 * (acc_qadv_abs / _sden).item()
            qadv_scale = max(qadv_ema, args.qadv_floor)
            q_dpg_gap = (acc_qadv_sig / _sden).item()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Full-batch teacher diagnostics (chunked, no grad): privileged-critic EV,
        # teacher/student entropies, mean-action gap in native z space.
        with torch.no_grad():
            t_vals, t_ents, s_ents, gaps = [], [], [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                summ = teacher.encode(b_future_tokens[sl], b_future_valid[sl]) if teacher.needs_attn else None
                pooled_sl = b_pooled[sl] if teacher.needs_pooled else None
                ta, tb = teacher.actor_params_from(
                    b_obs[sl], pooled_sl, summ, b_g[sl] if args.teacher_sees_g else None
                )
                t_ents.append(Beta(ta, tb).entropy().sum(-1).mean().item())
                _, _, _, s_ent, _, sa, sb = agent.get_action_and_value(b_obs[sl], b_latent_zs[sl])
                s_ents.append(s_ent.mean().item())
                gaps.append((ta / (ta + tb) - sa / (sa + sb)).abs().mean().item())
                t_vals.append(hl_support.to_scalar(teacher.critic_logits_from(b_obs[sl], pooled_sl, summ)))
            t_vals = torch.cat(t_vals).cpu().numpy()
            teacher_ev = np.nan if var_y == 0 else 1 - np.var(y_true - t_vals) / var_y

            # v35: attention entropy of the summary token's last-layer distribution, averaged
            # over heads, on the first 1024 states. High (near log2 of the valid-key count)
            # = the encoder ignores temporal structure and degenerates toward uniform
            # pooling; falling = attention is using the ordered future. Eager path (SDPA does
            # not expose weights); base-2 nats so the scale is interpretable vs log2(H+1).
            # pooled mode has no encoder -> NaN.
            if teacher.needs_attn:
                _na = min(1024, b_future_tokens.shape[0])
                _, _attn = teacher.encode_with_attn(b_future_tokens[:_na], b_future_valid[:_na])
                _sq = _attn[:, :, 0, :].clamp_min(0.0)          # (n, heads, keys): summary query row
                _ent = -(_sq * _sq.clamp_min(1e-12).log2()).sum(-1)  # per (state, head)
                teacher_attn_entropy = _ent.mean().item()
            else:
                teacher_attn_entropy = float("nan")

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # v35: end-of-iter scalars read the LAST minibatch's detached tensors (last_*) and the
        # once-read GPU accumulator means; no per-minibatch host syncs remain in the epoch loop.
        _nan = float("nan")
        writer.add_scalar("losses/value_loss", last_v_loss.item() if last_v_loss is not None else _nan, global_step)
        writer.add_scalar("losses/policy_loss", last_pg_loss.item() if last_pg_loss is not None else _nan, global_step)
        writer.add_scalar("losses/entropy", last_entropy.item() if last_entropy is not None else _nan, global_step)
        writer.add_scalar("losses/approx_kl", last_approx_kl.item() if last_approx_kl is not None else _nan, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("losses/teacher_nll", teacher_nll_mean, global_step)
        writer.add_scalar("losses/teacher_value_loss", last_t_v_loss.item() if last_t_v_loss is not None else _nan, global_step)
        writer.add_scalar("losses/distill_kl", distill_kl_mean, global_step)
        # --- v34 twin-Q telemetry ---
        writer.add_scalar("losses/q_loss", q_loss_mean, global_step)
        writer.add_scalar("losses/q_value_mean", q_value_mean, global_step)
        writer.add_scalar("losses/actor_q_term", actor_q_term_mean, global_step)
        writer.add_scalar("debug/replay_size", rep_filled, global_step)
        writer.add_scalar("debug/q_reward_scale", q_reward_scale, global_step)
        writer.add_scalar("debug/qadv_scale", qadv_scale, global_step)
        writer.add_scalar("debug/q_dpg_gap", q_dpg_gap, global_step)
        writer.add_scalar("debug/dpg_atten", dpg_atten, global_step)
        # Fixed-radius frontier probe: Q's action-sensitivity at constant delta=0.1,
        # independent of policy concentration. Contract: q_dpg_gap -> 0 means DPG
        # exhaustion ONLY if q_frontier -> 0 too; q_frontier positive while
        # q_dpg_gap -> 0 = coverage collapse (flip ent_coef, don't bury the arm).
        with torch.no_grad():
            _n = min(1024, b_obs.shape[0])
            _fs, _fa = b_obs[:_n], b_actions[:_n]
            _fu = torch.randn_like(_fa)
            _fu = _fu / _fu.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            _fp = (_fa + 0.1 * _fu).clamp(q_act_low, q_act_high)
            _fqr = torch.min(q1(_fs, _fa), q2(_fs, _fa))
            _fqp = torch.min(q1(_fs, _fp), q2(_fs, _fp))
            q_frontier = (_fqp - _fqr).abs().mean().item()
        writer.add_scalar("debug/q_frontier", q_frontier, global_step)
        writer.add_scalar("debug/q_ev", q_ev, global_step)
        writer.add_scalar("debug/q_ev_replay", q_ev_replay, global_step)
        writer.add_scalar("debug/q_mean", q_mean, global_step)
        writer.add_scalar("debug/q_twin_gap", q_twin_gap, global_step)
        writer.add_scalar("debug/q_twin_gap_rsample", q_twin_gap_rsample, global_step)
        writer.add_scalar("debug/q_gate_mean", q_gate_mean_epoch if student_mb_count > 0 else q_gate_mean, global_step)
        writer.add_scalar("debug/qgrad_ratio", qgrad_ratio, global_step)
        writer.add_scalar("debug/qgrad_cos", qgrad_cos, global_step)
        writer.add_scalar("debug/arsample_std", arsample_std, global_step)
        writer.add_scalar("debug/distill_clipfrac", distill_clipfrac_mean, global_step)
        writer.add_scalar("debug/teacher_ev", teacher_ev, global_step)
        writer.add_scalar("debug/teacher_attn_entropy", teacher_attn_entropy, global_step)
        writer.add_scalar("debug/phi_dropout_frac", phi_dropout_frac, global_step)
        writer.add_scalar("debug/priv_kl", priv_kl_mean, global_step)      # v37: pre-penalty KL(full||null)
        writer.add_scalar("debug/lam_priv", lam_priv, global_step)         # v37: budget dual variable
        writer.add_scalar("debug/nll_gap", nll_gap, global_step)           # v38: nll(null)-nll(full) leak alarm
        # v35: how many compile tiers actually engaged. td/student self-register on their first
        # (parity-checked) call; the teacher encoder tier is tracked by its own _use_compiled.
        _teacher_active = bool(getattr(teacher, "_use_compiled", False))
        _n_tiers = len(compile_tiers_active) + int(_teacher_active)
        writer.add_scalar("debug/compile_tiers_active", _n_tiers, global_step)
        if iteration == 1:
            _tier_names = list(compile_tiers_active) + (["teacher"] if _teacher_active else [])
            print(f"[compile] active tiers ({_n_tiers}): {_tier_names}")
        writer.add_scalar("debug/vdis_mean", vdis.mean().item(), global_step)
        writer.add_scalar("debug/vdis_scale", vdis_scale, global_step)
        writer.add_scalar(
            "debug/vdis_bonus_mean", (vdis_scale * vdis / (vdis.std() + 1e-8)).mean().item(), global_step
        )
        writer.add_scalar("debug/teacher_entropy", np.mean(t_ents), global_step)
        writer.add_scalar("debug/student_entropy", np.mean(s_ents), global_step)
        writer.add_scalar("debug/teacher_student_mean_gap", np.mean(gaps), global_step)
        writer.add_scalar("debug/awr_weight_max", awr_w.max().item(), global_step)
        writer.add_scalar("debug/auto_temp", temp_now, global_step)
        writer.add_scalar("debug/clip_frac_adv", (adv_z.abs() > args.adv_clip).float().mean().item(), global_step)
        with torch.no_grad():
            _p = awr_w / awr_w.sum()
            writer.add_scalar(
                "debug/tilt_kl_realized",
                (_p * (_p * float(_p.numel())).clamp_min(1e-12).log()).sum().item(),
                global_step,
            )
            writer.add_scalar("debug/awr_ess", (1.0 / (_p.pow(2).sum() * _p.numel())).item(), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/fut_valid_frac", fut_valid_frac.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

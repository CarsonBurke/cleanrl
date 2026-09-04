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
    vdis_coef: float = 0.1           # v30: intrinsic bonus scale on |V_teacher - V_student| / std
    vdis_warmup_iters: int = 20      # v30: linear bonus ramp while the teacher critic trains

    # v34: off-policy twin-Q DPG path.
    q_coef: float = 0.2              # DPG weight in the student actor loss (scale-normalized term)
    q_updates_per_iter: int = 4096   # off-policy TD updates per iteration (high UTD)
    q_batch: int = 256               # transition minibatch for the TD updates
    q_lr: float = 3e-4               # Q optimizer lr
    q_tau: float = 0.005             # polyak coefficient for the target Q nets
    q_onpol_coef: float = 1.0        # weight of the on-policy lambda-return hardening pass
    replay_capacity: int = 1_000_000 # transition ring buffer capacity

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


class HindsightTeacher(nn.Module):
    """Separate teacher actor + teacher critic, both privileged.

    Actor input:  [obs, future-action mean/std per dim, g (z-scored lambda-return), valid frac]
    Critic input: [obs, future-action mean/std per dim, valid frac]  (no return features)
    """

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        H = args.hidden
        actor_in = obs_dim + 4 * act_dim + 1 + (1 if args.teacher_sees_g else 0)  # v19: near+far windows
        critic_in = obs_dim + 4 * act_dim + 1
        self.actor_trunk = ThinkTrunk(actor_in, H, args.k_blocks, args.n_experts)
        self.alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.critic_trunk = ThinkTrunk(critic_in, H, args.k_blocks, args.n_experts)
        self.critic_head = layer_init(nn.Linear(H, args.num_bins, bias=False), std=0.1)
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.conc_cap = args.teacher_conc_cap

    def actor_params_for(self, x_priv):
        feat = self.actor_trunk(x_priv)
        alpha = (1.0 + F.softplus(self.alpha_head(feat))).clamp(max=self.conc_cap)
        beta = (1.0 + F.softplus(self.beta_head(feat))).clamp(max=self.conc_cap)
        return alpha, beta

    def critic_logits(self, x_gait):
        return self.critic_head(self.critic_trunk(x_gait))


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


NEAR_HORIZON = 5  # v19: near window = a_{t+1..t+5}; far window = a_{t+6..t+H}


def build_future_features(actions, boundaries, horizon):
    """v19: per-(t, env) mean/std pooled WITHIN a near and a far future window.

    actions: (T, B, A); boundaries: (T, B) 1.0 where the transition at t ends an episode.
    Future step t+k (k>=1) is valid iff t+k <= T-1 and no boundary in transitions t..t+k-1.
    Returns (near_mean, near_std, far_mean, far_std, valid_frac) with zeros where the
    respective window has no valid step; valid_frac covers the full horizon.
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
        # extending the window by one step requires transition t+k-1 to be non-boundary
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
    rep_cap = args.replay_capacity
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

    # EMA of mean|min(Q1,Q2)| over the fresh batch. v34.1: this is now a TELEMETRY-ONLY
    # sensor (feeds the qgrad_ratio diagnostic); it is no longer in the DPG loss path.
    q_scale = 1.0

    # v34.1: EMA of mean|action-advantage| (proposal vs realized), the DPG loss normalizer.
    # Carried across iterations and used FIXED within each iteration's student epochs
    # (updated afterward), so the divisor never depends on the current minibatch. Floored
    # at args.qadv_floor. Init at 1.0 to match q_scale's conservative warmup.
    qadv_ema = 1.0
    qadv_scale = max(qadv_ema, args.qadv_floor)
    q_dpg_gap = 0.0

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
            # Windows and the teacher-critic context are built here (moved up from
            # below; neither depends on advantages). teacher_actor_in stays below —
            # it needs g when teacher_sees_g.
            near_mean, near_std, far_mean, far_std, fut_valid_frac = build_future_features(
                actions, transition_boundaries, args.hindsight_horizon
            )
            teacher_critic_in = torch.cat(
                [obs, near_mean, near_std, far_mean, far_std, fut_valid_frac.unsqueeze(-1)],
                dim=-1,
            )
            tc_flat = teacher_critic_in.reshape(-1, teacher_critic_in.shape[-1])
            vt_chunks = []
            for start in range(0, tc_flat.shape[0], args.minibatch_size):
                vt_chunks.append(
                    hl_support.to_scalar(teacher.critic_logits(tc_flat[start : start + args.minibatch_size]))
                )
            v_teacher_roll = torch.cat(vt_chunks).reshape(args.num_steps, args.num_envs)
            vdis = (v_teacher_roll - values).abs()
            vdis_scale = args.vdis_coef * min(1.0, (iteration - 1) / max(1, args.vdis_warmup_iters))
            shaped_rewards = rewards + vdis_scale * vdis / (vdis.std() + 1e-8)

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

            # v34: second GAE pass on RAW env rewards (NOT the vdis-shaped rewards) ->
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
            actor_ctx = [obs, near_mean, near_std, far_mean, far_std]
            if args.teacher_sees_g:
                actor_ctx.append(g.unsqueeze(-1))
            actor_ctx.append(fut_valid_frac.unsqueeze(-1))
            teacher_actor_in = torch.cat(actor_ctx, dim=-1)
            # teacher_critic_in already built above (v30)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()
        b_teacher_actor_in = teacher_actor_in.reshape(-1, teacher_actor_in.shape[-1])
        b_teacher_critic_in = teacher_critic_in.reshape(-1, teacher_critic_in.shape[-1])
        b_awr_w = awr_w.reshape(-1)
        b_actions = actions.reshape(-1, act_dim)            # v34: executed env actions in [-1,1]
        b_returns_raw = returns_raw.reshape(-1)             # v34: raw-reward lambda-returns

        # ================= v34: off-policy twin-Q training =================
        # Snapshot the CURRENT pooled obs_rms once per training block: use it BOTH to
        # recover raw obs for the transitions stored THIS iter (exact round-trip) and to
        # re-normalize sampled (older) transitions to the current stats at train time.
        with torch.no_grad():
            rms_mean, rms_var = pooled_obs_rms()
            raw_obs_now = obs_unnormalize(b_obs, rms_mean, rms_var)
            raw_next_obs_now = obs_unnormalize(
                next_obses.reshape(-1, obs_dim), rms_mean, rms_var
            )
            b_rew_raw = rewards.reshape(-1)                 # raw env reward (normalize_reward=False)
            b_term = transition_terminations.reshape(-1)    # terminated only (bootstrap thru truncation)

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
            # frozen forever (a growing return-std would drift the TD target).
            if not q_reward_frozen:
                q_rew_sum += float(b_rew_raw.sum().item())
                q_rew_sumsq += float((b_rew_raw * b_rew_raw).sum().item())
                q_rew_count += int(b_rew_raw.numel())
                _mean = q_rew_sum / max(1, q_rew_count)
                _var = max(q_rew_sumsq / max(1, q_rew_count) - _mean * _mean, 0.0)
                q_reward_scale = max(_var ** 0.5, 1e-3)
                if iteration >= args.vdis_warmup_iters:
                    q_reward_frozen = True

        def student_env_action(norm_obs, reparam):
            """Student Beta over norm_obs -> env-space action ([-1,1]), same affine map as
            the rollout. reparam=True -> differentiable rsample; else detached sample."""
            a_feat, _ = agent._trunks(norm_obs)
            alpha = 1.0 + F.softplus(agent.actor_alpha_head(a_feat))
            beta = 1.0 + F.softplus(agent.actor_beta_head(a_feat))
            dist = Beta(alpha, beta)
            z = (dist.rsample() if reparam else dist.sample()).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            return q_act_low + (q_act_high - q_act_low) * z

        # --- TD block: q_updates_per_iter off-policy updates ---
        q_losses, q_val_means = [], []
        for _qu in range(args.q_updates_per_iter):
            s_idx = torch.randint(0, rep_filled, (args.q_batch,), device=device)
            s_obs = obs_renormalize(rep_obs[s_idx], rms_mean, rms_var)
            s_next = obs_renormalize(rep_next_obs[s_idx], rms_mean, rms_var)
            s_act = rep_act[s_idx]
            s_rew = rep_rew[s_idx] / q_reward_scale
            s_term = rep_term[s_idx]
            with torch.no_grad():
                a_next = student_env_action(s_next, reparam=False)
                q_next = torch.min(q1_target(s_next, a_next), q2_target(s_next, a_next))
                y = s_rew + args.gamma * (1.0 - s_term) * q_next
            q1_pred = q1(s_obs, s_act)
            q2_pred = q2(s_obs, s_act)
            q_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
            q_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_params, 0.5)
            q_optimizer.step()
            with torch.no_grad():
                for p, tp in zip(q1.parameters(), q1_target.parameters()):
                    tp.mul_(1.0 - args.q_tau).add_(args.q_tau * p)
                for p, tp in zip(q2.parameters(), q2_target.parameters()):
                    tp.mul_(1.0 - args.q_tau).add_(args.q_tau * p)
            q_losses.append(q_loss.item())
            q_val_means.append(q1_pred.mean().item())

        # --- On-policy lambda-return hardening: regress BOTH Qs to raw-return targets on
        # the fresh rollout, INDEPENDENT shuffles per twin (do not couple the twins). ---
        q_targets_onpol = b_returns_raw / q_reward_scale
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

        # --- EMA of mean|min(Q1,Q2)| over the fresh batch -> scale-invariant DPG term ---
        with torch.no_grad():
            minq_abs = []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                minq_abs.append(torch.min(q1(b_obs[sl], b_actions[sl]), q2(b_obs[sl], b_actions[sl])).abs())
            q_scale = 0.99 * q_scale + 0.01 * float(torch.cat(minq_abs).mean().item())
            q_scale = max(q_scale, 1e-3)
        # Freeze Q params during the student epochs: DPG grads must flow only to the
        # actor (through a_rsample), never step or accumulate into the Q nets.
        for p in q_params:
            p.requires_grad_(False)

        # --- qgrad telemetry: DPG vs distill gradient alignment on the actor params
        # (once per iteration, one minibatch; autograd.grad so nothing pollutes .grad). ---
        _tel = slice(0, args.minibatch_size)
        _, _, _, _, _, s_a_t, s_b_t = agent.get_action_and_value(b_obs[_tel], b_latent_zs[_tel])
        with torch.no_grad():
            _ta, _tb = teacher.actor_params_for(b_teacher_actor_in[_tel])
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
        _qterm_t = args.q_coef * (_w * (-_minq / q_scale)).mean()
        _gd = torch.autograd.grad(_distill_t, actor_params, retain_graph=True, allow_unused=True)
        _gq = torch.autograd.grad(_qterm_t, actor_params, retain_graph=False, allow_unused=True)

        def _flat(gs):
            return torch.cat(
                [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, actor_params)]
            )

        _fd, _fq = _flat(_gd), _flat(_gq)
        _nd, _nq = _fd.norm(), _fq.norm()
        qgrad_ratio = (_nq / (_nd + 1e-8)).item()
        # Loss-path ratio: the loss divides by qadv_scale, the sensor above by q_scale;
        # the DPG grad norm scales linearly in the divisor, so rescale instead of a
        # second backward. Ceiling-only attenuation for THIS iteration's epochs —
        # sensor stays pure (measured pre-attenuation), actuator only ever reduces.
        qgrad_ratio_lp = qgrad_ratio * (q_scale / qadv_scale)
        dpg_atten = min(1.0, args.qgrad_ceiling / max(qgrad_ratio_lp, 1e-8))
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
        q_loss_mean = float(np.mean(q_losses))
        q_value_mean = float(np.mean(q_val_means))
        # ================= end v34 twin-Q training =================

        actor_q_terms, q_gate_means = [], []
        qadv_abs_means, qadv_signed_means = [], []   # v34.1: per-mb qadv stats for the EMA + log
        b_inds = np.arange(args.batch_size)
        distill_kls, distill_clipfracs, teacher_nlls = [], [], []
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
                t_alpha, t_beta = teacher.actor_params_for(b_teacher_actor_in[mb_inds])
                t_dist = Beta(t_alpha, t_beta)
                t_nll = -t_dist.log_prob(b_latent_zs[mb_inds]).sum(-1)
                teacher_actor_loss = (b_awr_w[mb_inds] * t_nll).mean()
                t_value_logits = teacher.critic_logits(b_teacher_critic_in[mb_inds])
                t_target = b_target_probs[mb_inds, 0].to(device=device, non_blocking=True)
                t_v_loss = -(t_target * torch.log_softmax(t_value_logits, dim=-1)).sum(-1).mean()
                teacher_loss = teacher_actor_loss + args.teacher_vf_coef * t_v_loss
                teacher_optimizer.zero_grad(set_to_none=True)
                teacher_loss.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(), args.teacher_grad_clip)
                teacher_optimizer.step()
                teacher_nlls.append(teacher_actor_loss.item())

                if student_stopped:
                    continue

                # ---- student update: dense clipped forward-KL distillation + critic CE ----
                _, _, newlogprob, entropy, value_logits, s_alpha, s_beta = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                kl_dims = beta_kl_per_dim(t_alpha.detach(), t_beta.detach(), s_alpha, s_beta)
                kl_dims = kl_dims.clamp_min(0.0)
                distill_loss = kl_dims.clamp(max=args.distill_kl_clip).sum(-1).mean()
                with torch.no_grad():
                    distill_kls.append(kl_dims.sum(-1).mean().item())
                    distill_clipfracs.append((kl_dims > args.distill_kl_clip).float().mean().item())
                pg_loss = args.distill_coef * distill_loss

                # v34.1: DPG on the fresh on-policy states. Reparameterized student action
                # mapped to env space (same affine as rollout), twin-min Q, gated per-state
                # by twin agreement so DPG pushes off the realized frontier only where the
                # twins agree. Normalized by qadv_scale (an EMA of the mean action-advantage
                # of the proposal over the REALIZED executed action) instead of |minQ|, so
                # the term stays at a stationary scale as absolute Q values grow.
                s_dist_rs = Beta(s_alpha, s_beta)
                z_rs = s_dist_rs.rsample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                a_rs = q_act_low + (q_act_high - q_act_low) * z_rs
                q1_rs, q2_rs = q1(b_obs[mb_inds], a_rs), q2(b_obs[mb_inds], a_rs)
                min_rs = torch.min(q1_rs, q2_rs)
                with torch.no_grad():
                    gap_rs = (q1_rs - q2_rs).abs()
                    gm_rs = gap_rs.mean()
                    w_rs = gm_rs / (gap_rs + gm_rs + 1e-8)  # eps: avoid 0/0 if twins coincide
                    # Realized-action baseline on the SAME minibatch rows (b_obs[mb_inds]
                    # and b_actions[mb_inds] are the state/executed-action pair for the same
                    # transitions). qadv = advantage of the proposal over the executed action.
                    q_real = torch.min(
                        q1(b_obs[mb_inds], b_actions[mb_inds]),
                        q2(b_obs[mb_inds], b_actions[mb_inds]),
                    )
                    qadv = min_rs.detach() - q_real
                    qadv_abs_means.append(qadv.abs().mean().item())
                    qadv_signed_means.append(qadv.mean().item())
                # Numerator is still -minQ(s, a_rs) (unchanged gradient direction); only the
                # normalizer changed from q_scale to the stationary, floored qadv_scale.
                qterm = (w_rs * (-min_rs / qadv_scale)).mean()
                ent_bonus = (entropy / act_dim).mean()   # mean over dims (addendum): Q coverage
                actor_q_terms.append((args.q_coef * dpg_atten * qterm).item())
                q_gate_means.append(w_rs.mean().item())

                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds].to(device=value_logits.device, non_blocking=True)
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(
                    device=value_logits.device, dtype=value_ce.dtype, non_blocking=True
                )
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

                entropy_loss = entropy.mean()

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

            if (
                args.target_kl is not None
                and not student_stopped
                and approx_kl > args.target_kl
            ):
                student_stopped = True

        # Re-enable Q grads for the next iteration's Q training block.
        for p in q_params:
            p.requires_grad_(True)

        # v34.1: update the DPG normalizer AFTER the epochs (this iteration's loss used the
        # value carried from the prior iteration, so the divisor never depends on the
        # current update). Floor is load-bearing: as the policy improves and qadv -> 0, the
        # normalizer rides the floor and the DPG term fades gracefully instead of amplifying
        # a noise direction. q_dpg_gap = signed mean(qadv): decaying to 0 = DPG exhaustion.
        if qadv_abs_means:
            qadv_ema = 0.99 * qadv_ema + 0.01 * float(np.mean(qadv_abs_means))
            qadv_scale = max(qadv_ema, args.qadv_floor)
            q_dpg_gap = float(np.mean(qadv_signed_means))

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Full-batch teacher diagnostics (chunked, no grad): privileged-critic EV,
        # teacher/student entropies, mean-action gap in native z space.
        with torch.no_grad():
            t_vals, t_ents, s_ents, gaps = [], [], [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                ta, tb = teacher.actor_params_for(b_teacher_actor_in[sl])
                t_ents.append(Beta(ta, tb).entropy().sum(-1).mean().item())
                _, _, _, s_ent, _, sa, sb = agent.get_action_and_value(b_obs[sl], b_latent_zs[sl])
                s_ents.append(s_ent.mean().item())
                gaps.append((ta / (ta + tb) - sa / (sa + sb)).abs().mean().item())
                t_vals.append(hl_support.to_scalar(teacher.critic_logits(b_teacher_critic_in[sl])))
            t_vals = torch.cat(t_vals).cpu().numpy()
            teacher_ev = np.nan if var_y == 0 else 1 - np.var(y_true - t_vals) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("losses/teacher_nll", np.mean(teacher_nlls), global_step)
        writer.add_scalar("losses/teacher_value_loss", t_v_loss.item(), global_step)
        writer.add_scalar("losses/distill_kl", np.mean(distill_kls), global_step)
        # --- v34 twin-Q telemetry ---
        writer.add_scalar("losses/q_loss", q_loss_mean, global_step)
        writer.add_scalar("losses/q_value_mean", q_value_mean, global_step)
        writer.add_scalar("losses/actor_q_term", float(np.mean(actor_q_terms)) if actor_q_terms else 0.0, global_step)
        writer.add_scalar("debug/replay_size", rep_filled, global_step)
        writer.add_scalar("debug/q_reward_scale", q_reward_scale, global_step)
        writer.add_scalar("debug/q_scale", q_scale, global_step)
        writer.add_scalar("debug/qadv_scale", qadv_scale, global_step)
        writer.add_scalar("debug/q_dpg_gap", q_dpg_gap, global_step)
        writer.add_scalar("debug/qgrad_ratio_lp", qgrad_ratio_lp, global_step)
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
        writer.add_scalar("debug/q_gate_mean", float(np.mean(q_gate_means)) if q_gate_means else q_gate_mean, global_step)
        writer.add_scalar("debug/qgrad_ratio", qgrad_ratio, global_step)
        writer.add_scalar("debug/qgrad_cos", qgrad_cos, global_step)
        writer.add_scalar("debug/arsample_std", arsample_std, global_step)
        writer.add_scalar("debug/distill_clipfrac", np.mean(distill_clipfracs), global_step)
        writer.add_scalar("debug/teacher_ev", teacher_ev, global_step)
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

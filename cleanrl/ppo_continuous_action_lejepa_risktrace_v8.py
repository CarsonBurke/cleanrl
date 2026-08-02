# LeJEPA Risk-Trace Vector Policy Optimization v8.
#
# A state-only critic predicts normalized geometric successor shells through horizon 100.
# A learned state-dependent stopping hazard replaces fixed GAE: constant lambda recovers
# ordinary GAE, while lambda(e_t) learns a distribution over every stopping time. Its
# risk model learns both the direct long-suffix mean and the conditional mean/variance of
# the previous recursive trace. At each state lambda is the closed-form minimizer of the
# next hard-lagged recursive estimator's conditional squared error. No Q-function,
# endpoint mixture, return reconstruction, or EMA is used.
#
# The actor never receives a scalar sample advantage. Vector successor advantages are
# freshly whitened. Exact Beta score features fit actuator-conditioned vector credit;
# held-out spectral mode shrinkage is rotation-equivariant and occurs before the reward
# covector, so it cannot commute into a scalar router. This route is an intentionally
# biased successor-gradient preconditioner around an exact common PPO gradient, not a
# second claim of unbiased credit. The covector performs the final gradient contraction
# (and supplies only a detached sign for common-term PPO clipping). LeJEPA remains
# attached-target predictive MSE + SIGReg: no contrastive pairs, negatives, or EMA teacher.
#
# The long inherited comment is retained as an audit trail for the 10,071-return parent;
# its per-coordinate-lambda and GAE mechanisms are not active in this file.
#
# ARCHIVAL PARENT NOTES
# =====================
# PPO + successor features, B: per-dimension lambda + vector-routed policy advantage.
# v2 = v1's mechanism with five correctness fixes found by review before v1 ever finished
# (v1 was cancelled at 1.6M; its runs/ dir is kept but its numbers are void):
#   1. w_r is zero-init, so adv_vector was IDENTICALLY ZERO on iteration 1 -> pg_loss == 0
#      and the actor got no gradient from the first 32,768 transitions. Now falls back to
#      scalar GAE until w_r has been solved once.
#   2. The tau estimator was ungated. At tau_stat_count == 1 the warmup rate is exactly
#      1.0, so a covariance off a handful of autocorrelated samples could overwrite every
#      mixing time and persist ~20 iterations through the EMA -- into the critic target.
#      Now gated at n_mc >= 256, with the warmup counter inside the guard.
#   3. Degenerate (deterministic) coordinates fell back to gae_lambda. Their discounted
#      sum is deterministic, so MC is exact: they want lambda = 1, and this is the exact
#      coordinate the header names as the motivating example.
#   4. lam_min was 0.50, pinning tau >= 2 and censoring the near-white action blocks --
#      one of the two blocks whose heterogeneity this variant exists to exploit.
#   5. `returns` became w_r.Lambda under vector_adv (no reward tensor in it at all), so
#      losses/explained_variance silently meant different things in the two arms. It is
#      now built from the scalar GAE unconditionally.
# =====================================================================================
# Base: ppo_continuous_action_lejepa_sf_v2.py (9,612 @6.4M on HalfCheetah-v4, vs the
# scalar-critic base's 8,278 @8M). Two changes, both in the advantage pathway; the
# critic head, the SSL path, the encoder, w_r, and PPO itself are untouched. (The critic
# TARGET does change -- per-dimension lambda -- but the head, loss and masks do not.)
#
# THE OBSERVATION THIS IS BUILT ON. v2 computes a 62-dimensional TD(lambda) residual
# E_t = Lambda_t - psi_0(s_t) for the critic, and SEPARATELY a scalar GAE for the policy.
# Those are not two different objects. The identity
#
#       w_r . E_t  ==  A_t      exactly whenever  w_r . phi_t == r_t
#
# says the scalar GAE was always a projection of the vector residual. v2 just computed
# the projection along its own recursion, with one lambda for everything, and threw the
# other 61 dimensions away.
#
# WHY THAT MATTERS. lambda is not a free knob -- it is an assumption about ONE number,
# the effective credit horizon tau = 1/(1 - gamma*lambda). The default lambda=0.95 at
# gamma=0.99 asserts tau = 16.8 steps (0.84 s) for EVERY coordinate. Measured, that is
# false and not by a little:
#
#   obs block      tau ~ 12     joint angles / velocities under a gait
#   e block        tau ~ 7      SIGReg-whitened, decorrelates faster
#   action blocks  tau ~ 1      nearly white given the state
#   constant       tau -> 100   its discounted sum is DETERMINISTIC given the episode
#                               end, so Monte Carlo is exact and zero-variance for it
#
# Accumulating raw samples past a coordinate's mixing time adds variance with NO bias
# reduction: past that point E[phi_{t+k}|s_t] is just the stationary mean, which the
# critic learns trivially. So the bias/variance trade should be made PER COORDINATE.
# A scalar critic structurally cannot do this -- one target, one lambda. A vector target
# has K mixing times and can carry K lambdas.
#
# THE TWO CHANGES.
#   1. per_dim_lambda: lambda_d = (1 - 1/tau_d)/gamma, with tau_d MEASURED each iteration
#      as std(sum_k gamma^k phi_d) / std(phi_d) -- no AR(1) or exponential-decay
#      assumption, so oscillatory coordinates (joint angles under a periodic gait) cancel
#      correctly in the sum where a lag-1 autocorrelation estimate would be badly wrong.
#      tau is computed from the lambda=1 MC sum, NOT from sf_std, precisely so the
#      estimate is lambda-independent: deriving lambda from the spread of Lambda(lambda)
#      closes a feedback loop whose only stable point is lambda_max.
#      Sanity: tau = 16.8 recovers lambda = 0.95 exactly, so this REDUCES to v2 when the
#      heterogeneity is absent.
#   2. vector_adv: the policy advantage becomes w_r . E_t. Without this, per-dimension
#      lambda would change only the critic target and never reach the policy.
#
# HONEST COSTS, both real:
#   - The projection identity is exact only if the probe is exact. It is not
#     (reward_probe_r2 ~ 0.98), so routing through w_r injects the accumulated probe
#     residual sum_k (gamma*lam)^k eps_{t+k} into the advantage. sf/value_err_frac
#     already tracks that quantity; --no-vector-adv is the control that prices it alone.
#   - lambda_d is fit on the policy-induced distribution and moves with the policy. It is
#     EMA'd (tau_ema) and bounded to [lam_min, lam_max] rather than free.
#
# KILL SIGNAL. advB/adv_vec_corr ~ 1.0 means the vector path reproduced scalar GAE and
# the run is a null experiment. advB/lam_spread ~ 0 means the per-coordinate heterogeneity
# this variant exists to exploit is not there.
#
# EXPERIMENTS. Treatment = defaults. Control = --no-vector-adv (per-dimension lambda on
# the critic target only, policy on scalar GAE) isolates the w_r projection cost from the
# lambda effect. A third arm, --no-per-dim-lambda, is v2's advantage with the vector
# routing and nothing else.
# =====================================================================================
# Base: ppo_continuous_action_iterthink_v24_beta_d3bucket_mtp_mbpercnorm_v2.py run with
# --norm-adv --norm-adv-scope batch --no-ret-percnorm (8,278 @8M on HalfCheetah-v4).
# Those three flags are BAKED IN here; everything else is untouched except the critic.
#
# THESIS. Scalar V^pi(s) conflates two objects with different timescales: STRUCTURE
# (where the agent is headed in the MDP -- slow, largely policy-independent) and
# EVALUATION (what that is worth under the current pi and return scale -- moves every
# policy step). GAE bootstraps the evaluation through time, so every actor update
# redefines the multi-step target. The base attacks this only on the CODOMAIN
# (Dreamer3-bucket HL-Gauss pins the range; critic MTP adds horizons) -- neither makes
# the bootstrapped object stationary.
#
#   e_t   = f(s_t)                     encoder, SIGReg-constrained toward N(0, I)
#   phi_t = [ e_t , s_t , a_t , a_t*a_t , 1 ]   reward features; dim = emb_dim + obs_dim
#                                               + 2*act_dim + 1
#   psi_h(s_t) ~ Lambda_{t+h}          CRITIC HEAD on the ThinkTrunk, h = 0..5
#   Lambda_t  = E_pi[ sum_k gamma^k phi_{t+k} ]   successor features -- the ONLY
#                                                 bootstrapped object, vector-valued
#   V(s)  = w_r . psi_0(s)             evaluation, a dot product
#   w_r   solved from w_r.phi_t ~ r_t  IMMEDIATE reward only, closed-form ridge,
#                                      no bootstrapping, recomputed every iteration
#
# The only scalar regression left anywhere is w_r against immediate reward. Everything
# temporal happens in a space whose marginal is externally clamped by SIGReg.
#
# WHY REPLACE THE HEAD, NOT ADD A PATHWAY. share_backbone=True: actor and critic fully
# share the ThinkTrunk. Deleting the critic would strip ~half the trunk's gradient, and
# bolting SF on in parallel would have psi read a near-raw-obs latent while a scalar
# critic reads the policy-shaped trunk -- a regression then would not be attributable to
# "SF vs scalar" at all. So the swap is surgical: same head shape, same MTP semantics,
# same boundary masks; only K goes from 511 bucket logits to emb_dim + obs_dim +
# 2*act_dim + 1 = 62 SF dimensions (HalfCheetah: 32 + 17 + 12 + 1), and the loss from
# cross-entropy to masked MSE on standardized Lambda.
# It also deletes the 402MB CPU target_probs tensor and its ~4GB/iteration H2D traffic.
#
# HONEST ACCOUNTING (the strong stationarity claim does NOT survive):
#   - psi is still policy-dependent; its fixed point moves with pi exactly as V^pi does.
#     What changes is that the bootstrap target's MARGINAL is pinned, not that
#     pi-dependence is gone.
#   - w_r is only approximately policy-independent -- it is fit over the policy-induced
#     distribution, which is pi-invariant only if the regression were well-specified.
#   - a*a is the MOST policy-dependent feature here: with a Beta actor
#     E_pi[a*a|s] = Var_pi + mean^2, so that block tracks policy entropy directly.
#     Correct (ctrl cost genuinely depends on pi) but the vector is not uniformly stable.
#   - SIGReg pins the DISTRIBUTION, not the coordinate FRAME. N(0,I) is rotation-
#     invariant and the prediction loss cannot pin the frame either (the predictor
#     co-rotates), while psi_old and w_r are functions of coordinates. This is a NEW
#     non-stationarity the design introduces. Measured (ssl/frame_drift_*), not
#     pre-emptively patched with an EMA.
#
# SSL PATH. Encoder + causal action-conditioned predictor + SIGReg, in a SEPARATE
# top-level module with its own AdamW, fully detached from the trunk and from w_r. Its
# only job is to define what e means. Exactly two loss terms, LeJEPA-faithful:
# pred MSE with BOTH branches attached + lambda * SIGReg. No EMA teacher, no stop-grad,
# no asymmetry -- SIGReg is what prevents collapse, and a teacher would re-add the second
# unanchored timescale this design exists to remove. Runs ONCE PER ITERATION, outside the
# 320-minibatch PPO loop (+10-20% wall clock; inside would be +100-200%).
#
# INSTRUMENTATION. The obvious EV metric is rigged: b_returns = advantages + values
# contains the critic's own bootstrapped values, and at dt=0.05 errors are strongly
# state-autocorrelated, so a critic scores well against a target built from itself.
# gate/* instead scores three predictors against a common TRUNCATED-MC return:
#   (1) ev_trunk_probe  detached scalar probe on sg(trunk_feat)  -- what a scalar critic
#                       on this trunk would achieve (replaces the lost HL-Gauss comparator)
#   (2) ev_sf_online    w_r.psi_0, the value the actor actually consumed -- the treatment
#   (3) ev_latent_cap   closed-form ridge of MC return onto sg(e) -- how much reward
#                       signal is LINEARLY present in the instantaneous embedding
# (3) is NOT a ceiling on (2): psi reads the trunk, not e_t, and accumulates phi over
# time, so (2) > (3) means the successor-feature construction is buying something a
# linear readout of e_t cannot. The kill signal is (2) << (1) sustained -- the SF critic
# losing to what a plain scalar critic on the same trunk would achieve.
#
# HYPOTHESES. H1 latent-path loss stays smooth across policy shifts where scalar-critic
# EV spikes. H2 latent occupancy captures gait/contact structure a scalar critic lacks.
# H3 (the falsifier) EV(w_r.psi_0) vs a scalar critic's EV on a COMMON target.
#
# SCOPING. At k=1 the predictor only keeps e dynamics-grounded; one-step prediction in
# MuJoCo is near-trivial, so a whitened near-linear map of s can satisfy it. If v1 works,
# do NOT credit the JEPA predictor for it -- k=1 makes this a clean test of the
# SUCCESSOR-FEATURE claim. Multi-horizon k and history-dependent embeddings are the v2
# levers where H2 actually gets tested.
#
# Naming: `z`/`latent_zs` is the Beta actor's native action sample (inherited). The
# encoder latent is `e`/`emb` throughout.
import copy
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
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.lejepa import (
    ActionEncoder,
    ARPredictor,
    CompiledModule,
    MLP,
    SIGReg,
    StateEncoder,
)

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))
TD_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 100)
TD_BETAS = tuple(1.0 - 1.0 / horizon for horizon in TD_HORIZONS)
TD_SHELL_WIDTHS = tuple(
    horizon - (TD_HORIZONS[index - 1] if index else 0)
    for index, horizon in enumerate(TD_HORIZONS)
)
LEJEPA_HORIZONS = (1, 2, 4, 8, 16, 32, 64)


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
    gae_lambda: float = 0.95  # fixed-trace control; learned arm replaces it statewise
    learn_trace: bool = True
    trace_risk_lr: float = 1e-3
    trace_risk_epochs: int = 8
    trace_risk_batch: int = 1024
    trace_risk_min_samples: int = 4096
    trace_variance_floor: float = 1e-4
    vector_whiten_ridge: float = 1e-3
    vector_kl_coef: float = 1.0
    router_ridge: float = 1e-2
    router_max_fraction: float = 0.5
    num_minibatches: int = 32
    update_epochs: int = 10
    # NOTE: the reference run this variant is built against (exp-name
    # "iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_v1", 8278 @8M) was NOT a separate
    # file -- it was mbpercnorm_v2 launched with --norm-adv --norm-adv-scope batch
    # --no-ret-percnorm. Those three are baked in as defaults here so the baseline is
    # reproduced by default rather than by remembering flags.
    norm_adv: bool = False           # vector whitening replaces scalar advantage normalization
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

    # The HL-Gauss bucket critic is GONE. The critic head now emits successor features
    # (see header). MTP semantics and the horizon masks are retained verbatim -- only the
    # per-horizon target changed from a 511-bin scalar-return distribution to a
    # K-dimensional latent occupancy vector.
    critic_mtp_horizon: int = len(TD_HORIZONS)  # one normalized shell per horizon

    # --- LeJEPA successor-feature value pathway -------------------------------------
    emb_dim: int = 32                # d. Obs manifold is 17-dim, so e is rank <= 17 regardless;
    #                                  expect effective rank ~17, which is NOT a failure signal.
    ssl_hidden: int = 256            # encoder / projector MLP width
    pred_depth: int = 2              # causal transformer depth
    pred_heads: int = 4
    pred_dim_head: int = 32
    pred_mlp_dim: int = 256
    seq_len: int = max(LEJEPA_HORIZONS) + 1
    sigreg_weight: float = 0.09      # lambda. NOTE: the Epps-Pulley statistic scales with batch
    #                                  size, so the reference value does not transfer verbatim.
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256     # exact (statistic is a mean over directions); bounds memory
    ssl_lr: float = 5e-5             # LeWM reference optimizer
    ssl_weight_decay: float = 1e-3
    ssl_batch: int = 128             # long sequences; keeps SIGReg and attention intermediates bounded
    ssl_epochs: int = 8
    ssl_grad_clip: float = 1.0       # own clip (reference lewm.yaml gradient_clip_val), fully
    #                                  separate from PPO's -- see ssl/grad_norm for the pre-clip value
    control_encoder_tau: float = 1.0  # hard accept/reject snapshot; never an EMA
    control_encoder_kl: float = 0.001
    control_encoder_emb_drift: float = 0.02
    control_encoder_sf_drift: float = 0.01
    control_encoder_value_drift: float = 0.01
    sf_alpha: float = 1.0            # 1.0 = pure latent prediction, ZERO scalar regression in the
    #                                  network. <1 mixes in MSE(w_r.psi, w_r.Lambda) -- the fallback
    #                                  if the pure form underfits, not the starting point.
    # --- Vector-routed advantage ------------------------------------------------------
    vector_adv: bool = True          # policy advantage = w_r . E_t (the vector TD(lambda)
    #                                  residual) instead of the separate scalar GAE
    #                                  recursion. Required for per_dim_lambda to reach the
    #                                  policy at all; --no-vector-adv is the control that
    #                                  prices the w_r projection on its own.
    sf_ridge: float = 1e-3           # ridge coefficient for the closed-form w_r solve
    mc_window: float = 500           # truncated-MC horizon for the UNRIGGED EV gate (gamma^500=0.0066)

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
    norm_adv_scope: str = "batch"    # ...standardized once over the whole rollout ("batch")

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # --- torch.compile ---------------------------------------------------------------
    # reduce-overhead (CUDA graph trees) is NOT usable on the actor/critic path: v_loss and
    # pg_loss share the trunk forward, and the update runs backward twice with
    # retain_graph=True, cloning and re-adding grads in between. Cudagraph outputs live in
    # the graph pool, clip_grad_norm_ mutates them in place, and the second backward replays
    # over live refs -> either "accessing tensor output of CUDAGraphs that has been
    # overwritten" or a re-record per minibatch (x320/iter), which is SLOWER than eager.
    # So the trunk gets inductor fusion without graphs; the SSL nets (separate optimizer,
    # single plain backward) do get reduce-overhead.
    compile: bool = False            # validate the port eager first -- compile changes numerics
    compile_mode: str = "reduce-overhead"  # applies to the SSL nets only; see above for why the
    #                                        actor/critic trunk cannot take cudagraphs at all
    compile_ssl_cudagraphs: bool = False   # DEFAULT OFF, and not out of caution: LeJepaSSL.forward
    #   chains encoder -> action_encoder -> predictor, and each cudagraph-wrapped call issues its
    #   own cudagraph_mark_step_begin(), which invalidates the still-pending backward of the
    #   modules called earlier in the SAME forward. It raises "accessing tensor output of
    #   CUDAGraphs that has been overwritten" on the FIRST ssl_loss.backward() -- reproduced --
    #   and suppress_errors does NOT catch it (that only covers dynamo compile-time errors).
    #   Cloning outputs does not help either: the invalidated tensors are the saved intermediates.
    #   The SSL path is ~3% of wall clock, so inductor fusion without graphs costs ~nothing.

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


class LeJepaSSL(nn.Module):
    """Encoder + geometric action-conditioned predictors + SIGReg.

    Trained by EXACTLY two terms: a geometric embedding-prediction MSE with BOTH branches
    attached (no stop-gradient on the target -- SIGReg is what prevents collapse, and a
    stop-grad or EMA teacher would re-introduce the second unanchored timescale this
    design exists to remove), and SIGReg itself.

    Its only job is to define what the embedding e means. It never sees a reward, and it
    never receives gradient from the value path -- psi and w_r both read sg(e).
    """

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.encoder = StateEncoder(obs_dim, args.emb_dim, args.ssl_hidden)
        self.action_encoder = ActionEncoder(act_dim, args.emb_dim)
        self.predictor = ARPredictor(
            num_frames=args.seq_len,
            depth=args.pred_depth,
            heads=args.pred_heads,
            mlp_dim=args.pred_mlp_dim,
            input_dim=args.emb_dim,
            hidden_dim=args.emb_dim,
            dim_head=args.pred_dim_head,
            dropout=0.0,
            emb_dropout=0.0,
        )
        self.pred_projs = nn.ModuleDict(
            {
                str(horizon): MLP(2 * args.emb_dim, args.ssl_hidden, args.emb_dim)
                for horizon in LEJEPA_HORIZONS
            }
        )
        self.action_trace_encoders = nn.ModuleDict(
            {
                str(horizon): nn.Conv1d(
                    args.emb_dim, args.emb_dim, kernel_size=horizon
                )
                for horizon in LEJEPA_HORIZONS
            }
        )
        self.sigreg = SIGReg(
            num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk
        )

    def forward(self, obs_seq, act_seq, continuation_seq, sigreg_weight):
        """Predict attached future embeddings over geometric horizons.

        ``continuation_seq[:, t]`` says whether transition t -> t+1 stays in the same
        episode. A source-target pair is valid only when its full context prefix and
        intervening action trace do not cross a reset.
        """
        emb = self.encoder(obs_seq)                              # (N,L,d)
        act_emb = self.action_encoder(act_seq)                   # (N,L,d)
        context = self.predictor(emb, act_emb)                   # (N,L,d)
        prefix_valid = continuation_seq.cumprod(dim=1)
        horizon_losses = []
        for horizon in LEJEPA_HORIZONS:
            # A learned temporal kernel consumes exactly actions [t,t+h). Unlike a mean,
            # it distinguishes the order of stance/thrust phases. The final convolution
            # window begins at t=L-h and has no in-chunk target at t+h, so it is dropped.
            action_trace = self.action_trace_encoders[str(horizon)](
                act_emb.transpose(1, 2)
            ).transpose(1, 2)[:, :-1]
            pred = self.pred_projs[str(horizon)](
                torch.cat([context[:, :-horizon], action_trace], dim=-1)
            )
            err = (pred - emb[:, horizon:]).pow(2).mean(-1)
            valid = prefix_valid[:, horizon - 1 : -1]
            horizon_losses.append((err * valid).sum() / valid.sum().clamp_min(1.0))
        horizon_losses = torch.stack(horizon_losses)
        horizon_weights = horizon_losses.new_tensor(
            [horizon**-0.5 for horizon in LEJEPA_HORIZONS]
        )
        pred_loss = (horizon_weights * horizon_losses).sum() / horizon_weights.sum()
        # THE TRANSPOSE IS LOAD-BEARING. SIGReg reduces the empirical characteristic
        # function over dim -3 and scales by size(-2); both resolve to the BATCH only in
        # (T, B, D) layout. Passing (N, L, d) would average the CF over L=4 samples,
        # silently disabling collapse protection while still logging a plausible number.
        sigreg_loss = self.sigreg(emb.transpose(0, 1))           # (L, N, d)
        return (
            pred_loss + sigreg_weight * sigreg_loss,
            pred_loss,
            sigreg_loss,
            horizon_losses.detach(),
        )


def phi_features(emb, obs, action):
    """phi = [e, s, a, a*a, 1].

    The raw (already NormalizeObservation-scaled) state block is v2's whole change. See
    the header: without it a linear w_r cannot recover the velocity term from a whitened
    nonlinear latent, and v1 measured the resulting residual at lag-1 autocorrelation
    0.474 -- structured error, the expensive kind, injected into every value estimate.

    The action blocks are not decoration. HalfCheetah's reward is x_vel - 0.1*||a||^2, and
    the control cost is not a function of state AT ALL -- so a probe on e alone carries an
    irreducible residual that scales with policy action magnitude, i.e. the very
    non-stationarity this design removes leaks straight back in. `a*a` lets a LINEAR probe
    capture MuJoCo ctrl cost exactly.

    The trailing constant is the regression intercept, carried as a REAL feature rather
    than a separate bias so that V = w_r . psi stays an exact identity (a bias term would
    have no discounted-sum counterpart in psi). It earns its keep twice over: the
    observation is normalized, so e has a running-mean offset that shifts as the policy's
    state distribution moves, and the constant's own discounted sum is 1/(1-gamma)
    truncated at episode end -- that coordinate quietly encodes expected remaining
    episode length.
    """
    ones = emb[..., :1].new_ones(emb.shape[:-1] + (1,))
    return torch.cat([emb, obs, action, action * action, ones], dim=-1)


def normalized_shells_to_cumulative(next_shells, widths):
    """Recover raw cumulative successors M_beta from unit-mass temporal shells."""
    if next_shells.shape[-2] != widths.numel():
        raise ValueError("successor shell count does not match width grid")
    shape = (1,) * (next_shells.ndim - 2) + (-1, 1)
    return (next_shells * widths.view(shape)).cumsum(dim=-2)


def cumulative_to_normalized_shells(cumulative, widths):
    """Convert cumulative M_beta targets into unit-mass temporal shells."""
    raw_shells = cumulative.clone()
    raw_shells[..., 1:, :] = cumulative[..., 1:, :] - cumulative[..., :-1, :]
    shape = (1,) * (cumulative.ndim - 2) + (-1, 1)
    return raw_shells / widths.view(shape)


def build_normalized_shell_targets(phi, next_shells, bootstrap, betas, widths):
    """One-step Bellman targets for normalized temporal successor shells."""
    cumulative_next = normalized_shells_to_cumulative(next_shells, widths)
    shape = (1,) * (phi.ndim - 1) + (-1, 1)
    cumulative_targets = (
        phi.unsqueeze(-2)
        + betas.view(shape)
        * bootstrap.unsqueeze(-1).unsqueeze(-1)
        * cumulative_next
    )
    return cumulative_to_normalized_shells(cumulative_targets, widths)


def build_full_suffix_shell_targets(
    phi, next_shells, bootstrap, continuation, betas, widths
):
    """Use every same-episode suffix sample, bootstrapping only at data boundaries."""
    if next_shells.shape[-2] != betas.numel():
        raise ValueError("geometric successor grid mismatch")
    cumulative_next = normalized_shells_to_cumulative(next_shells, widths)
    cumulative_targets = torch.empty_like(next_shells)
    running = torch.zeros_like(next_shells[0])
    beta = betas.view(1, -1, 1)
    for t in reversed(range(phi.shape[0])):
        if t == phi.shape[0] - 1:
            next_estimate = (
                bootstrap[t].unsqueeze(-1).unsqueeze(-1) * cumulative_next[t]
            )
        else:
            cont = continuation[t].unsqueeze(-1).unsqueeze(-1)
            boundary_bootstrap = (
                (1.0 - cont)
                * bootstrap[t].unsqueeze(-1).unsqueeze(-1)
                * cumulative_next[t]
            )
            next_estimate = cont * running + boundary_bootstrap
        running = phi[t].unsqueeze(-2) + beta * next_estimate
        cumulative_targets[t] = running
    return cumulative_to_normalized_shells(cumulative_targets, widths)


def fixed_horizon_sum(signal, continuation, gamma, horizon):
    """Discounted fixed-n sum that stops at reset boundaries.

    Unlike GAE this has no lambda and no mixture over stopping times. Signal has shape
    (T,B,...); continuation[t]==0 prevents signal[t+1] from entering the sum at t.
    Rollout-tail rows naturally use shorter prefixes.
    """
    out = torch.zeros_like(signal)
    alive = torch.ones_like(continuation)
    for k in range(horizon):
        valid = signal.shape[0] - k
        if valid <= 0:
            break
        weight_shape = (valid, signal.shape[1]) + (1,) * (signal.ndim - 2)
        out[:valid] += (gamma**k) * alive.reshape(weight_shape) * signal[k:]
        if k + 1 < horizon and valid > 1:
            alive = alive[:-1] * continuation[k : k + valid - 1]
    return out


def learned_vector_gae(delta, continuation, gamma, state_lambda):
    """Vector GAE with a state-dependent, rotation-equivariant scalar hazard.

    lambda(s_{t+1}) decides whether the trace continues beyond delta_t. The continuation
    mask cuts resets independently of the one-step bootstrap already present in delta.
    """
    if delta.shape[:2] != continuation.shape or state_lambda.shape != continuation.shape:
        raise ValueError("delta, continuation, and state_lambda time/env axes must match")
    next_lambda = torch.zeros_like(state_lambda)
    next_lambda[:-1] = state_lambda[1:]
    out = torch.zeros_like(delta)
    running = torch.zeros_like(delta[0])
    for t in reversed(range(delta.shape[0])):
        running = delta[t] + (
            gamma
            * continuation[t].unsqueeze(-1)
            * next_lambda[t].unsqueeze(-1)
            * running
        )
        out[t] = running
    return out


def optimal_recursive_lambda(
    bootstrap_mean,
    direct_mean,
    recursive_mean,
    recursive_variance,
):
    """Closed-form local risk minimizer for a hard-lagged recursive trace."""
    bootstrap_error = bootstrap_mean - direct_mean
    continuation_direction = recursive_mean - bootstrap_mean
    numerator = -(bootstrap_error * continuation_direction).mean(-1)
    denominator = (
        continuation_direction.square() + recursive_variance
    ).mean(-1).clamp_min(1e-12)
    return (numerator / denominator).clamp(0.0, 1.0)


def whiten_vector_advantage(advantage, reward_covector, ridge):
    """Whiten a vector advantage and transform its covector without changing reward.

    For centered A, U=A C^-1/2 and q=C^1/2 w, hence Uq=Aw. A single rollout covariance
    is used directly; no running statistics or EMA exist.
    """
    flat = advantage.reshape(-1, advantage.shape[-1])
    centered = flat - flat.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    scale = covariance.diagonal().mean().clamp_min(1e-8)
    covariance = covariance + ridge * scale * torch.eye(
        covariance.shape[0], device=covariance.device, dtype=covariance.dtype
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    eigenvalues = eigenvalues.to(flat.dtype)
    eigenvectors = eigenvectors.to(flat.dtype)
    sqrt_cov = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
    inv_sqrt_cov = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.T
    whitened = (centered @ inv_sqrt_cov).reshape_as(advantage)
    transformed_covector = sqrt_cov @ reward_covector
    return (
        whitened,
        transformed_covector,
        eigenvalues,
        inv_sqrt_cov,
        sqrt_cov,
    )


def action_router_features(actor_features, native_actions, alpha, beta):
    """State features and exact Beta sufficient-statistic policy scores."""
    feature_mean = actor_features.mean(0)
    feature_scale = actor_features.std(0).clamp_min(1e-4)
    normalized_features = (actor_features - feature_mean) / feature_scale
    normalized_features = torch.cat(
        [normalized_features, normalized_features.new_ones(normalized_features.shape[0], 1)],
        dim=-1,
    )
    score_alpha = (
        native_actions.clamp_min(SAMPLE_EPS).log()
        - torch.digamma(alpha)
        + torch.digamma(alpha + beta)
    )
    score_beta = (
        (1.0 - native_actions).clamp_min(SAMPLE_EPS).log()
        - torch.digamma(beta)
        + torch.digamma(alpha + beta)
    )
    action_score = torch.stack([score_alpha, score_beta], dim=2)
    return normalized_features, action_score, feature_mean, feature_scale


def fit_action_router(
    features,
    action_score,
    vector_target,
    ridge,
    train_mask,
    validation_mask,
    inv_sqrt_cov,
    sqrt_cov,
):
    """Fit then spectrally shrink vector action modes on held-out trajectories."""
    design = torch.einsum("nac,nf->nacf", action_score, features).flatten(1)
    design64 = design[train_mask].double()
    target64 = vector_target[train_mask].double()
    gram = design64.T @ design64
    scale = gram.diagonal().mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    coefficients = torch.linalg.solve(gram, design64.T @ target64)
    validation_design = design[validation_mask].double()
    validation_target = vector_target[validation_mask].double()
    inv_sqrt64 = inv_sqrt_cov.double()
    sqrt64 = sqrt_cov.double()
    coefficients_white = coefficients @ inv_sqrt64
    u, singular_values, vh = torch.linalg.svd(
        coefficients_white, full_matrices=False
    )
    mode_prediction = (validation_design @ u) * singular_values
    mode_target = validation_target @ inv_sqrt64 @ vh.T
    mode_gain = (
        (mode_prediction * mode_target).sum(0)
        / mode_prediction.square().sum(0).clamp_min(1e-12)
    ).clamp(0.0, 1.0)
    coefficients_white = (
        u * (singular_values * mode_gain).unsqueeze(0)
    ) @ vh
    coefficients = coefficients_white @ sqrt64
    prediction = validation_design @ coefficients
    target_centered = validation_target - validation_target.mean(0, keepdim=True)
    residual = validation_target - prediction
    r2 = 1.0 - residual.square().sum() / target_centered.square().sum().clamp_min(1e-12)
    effective_rank = (mode_gain > 0.1).sum()
    return (
        coefficients.to(vector_target.dtype),
        r2.to(vector_target.dtype),
        effective_rank,
        mode_gain.mean().to(vector_target.dtype),
    )


def routed_vector_advantage(
    features,
    action_score,
    router_coefficients,
    inv_sqrt_cov,
    common_whitened,
    max_fraction,
):
    """Allocate zero-sum embedding credit across actuator score directions."""
    action_count = action_score.shape[1]
    score_count = action_score.shape[2]
    feature_count = features.shape[-1]
    vector_dim = common_whitened.shape[-1]
    coefficients = router_coefficients.view(
        action_count, score_count, feature_count, vector_dim
    )
    routed = (
        action_score.unsqueeze(-1)
        * torch.einsum("nf,acfd->nacd", features, coefficients)
    ).sum(2)
    routed = routed - routed.mean(1, keepdim=True)
    routed = routed @ inv_sqrt_cov
    flat_routed = routed.flatten(0, 1)
    flat_routed = flat_routed - flat_routed.mean(0, keepdim=True)
    routed_covariance = (
        flat_routed.T @ flat_routed / max(flat_routed.shape[0] - 1, 1)
    )
    spectral_std = torch.linalg.eigvalsh(
        routed_covariance.double()
    ).max().clamp_min(1e-12).sqrt().to(routed.dtype)
    scale = torch.clamp(max_fraction / spectral_std, max=1.0)
    routed = scale * routed
    return common_whitened.unsqueeze(1) + routed, routed, scale


def vector_policy_surrogate(
    joint_ratio,
    marginal_ratio,
    common_advantage,
    routed_advantage,
    reward_covector,
    clip_coef,
    clip_coef_high,
):
    """Clipped joint PPO credit plus un-clipped actuator-specific vector routing."""
    clipped_ratio = joint_ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef_high)
    scalar_common = (common_advantage @ reward_covector).detach()
    effective_ratio = torch.where(
        joint_ratio * scalar_common <= clipped_ratio * scalar_common,
        joint_ratio,
        clipped_ratio,
    )
    differential = routed_advantage - common_advantage.unsqueeze(1)
    return -(
        effective_ratio.unsqueeze(-1) * common_advantage
        + (marginal_ratio.unsqueeze(-1) * differential).sum(1)
    ).mean(0)


class DirectSuccessorRisk(nn.Module):
    """Hard-lagged state-only risk model for direct and recursive trace targets."""

    def __init__(self, input_dim, sf_dim, hidden, variance_floor):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
        )
        self.direct_mean = layer_init(nn.Linear(hidden, sf_dim), std=0.01)
        self.direct_raw_variance = layer_init(nn.Linear(hidden, sf_dim), std=0.01)
        self.trace_mean = layer_init(nn.Linear(hidden, sf_dim), std=0.01)
        self.trace_raw_variance = layer_init(nn.Linear(hidden, sf_dim), std=0.01)
        self.variance_floor = variance_floor

    def forward(self, embedding):
        hidden = self.trunk(embedding)
        direct_mean = self.direct_mean(hidden)
        direct_variance = (
            F.softplus(self.direct_raw_variance(hidden)) + self.variance_floor
        )
        trace_mean = self.trace_mean(hidden)
        trace_variance = (
            F.softplus(self.trace_raw_variance(hidden)) + self.variance_floor
        )
        return direct_mean, direct_variance, trace_mean, trace_variance


def align_latent_frame(source, reference):
    """Orthogonal Procrustes alignment with a translation for finite-sample means."""
    source_mean = source.mean(0)
    reference_mean = reference.mean(0)
    source_centered = source - source_mean
    reference_centered = reference - reference_mean
    u, _, vh = torch.linalg.svd(
        source_centered.T.double() @ reference_centered.double(),
        full_matrices=False,
    )
    rotation = (u @ vh).to(source.dtype)
    bias = reference_mean - source_mean @ rotation
    return rotation, bias, source @ rotation + bias


def solve_reward_probe(phi, reward, ridge):
    """Closed-form ridge solve of w_r from phi -> immediate reward.

    Solved, not gradient-fit: it removes a learning rate, removes the cold-start phase
    where a random V would drive the actor, and makes "thin, fast, stationary readout"
    literal -- w_r is recomputed optimally every iteration. Done in float64; the normal
    equations square the condition number and phi's blocks have very different scales.
    """
    phi64 = phi.double()
    gram = phi64.T @ phi64
    rhs = phi64.T @ reward.double()
    scale = torch.diagonal(gram).mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram, rhs).to(phi.dtype)


def ev_score(pred, target):
    var = target.var()
    return float(1.0 - (target - pred).var() / var.clamp_min(1e-12))


def chunk_sequences(x, seq_len):
    """(T, B, D) -> (n*B, L, D) with time contiguous within a chunk and env fixed.

    The buffer is T-MAJOR, so the usual flatten gives index t*B + e. Chunking the
    FLATTENED tensor would produce sequences that walk across envs at a fixed timestep
    rather than across time -- and because adjacent envs look similar under
    NormalizeObservation, that yields a perfectly plausible prediction loss that no curve
    would ever catch. Chunk before flattening.

    Round-trip: x[n*L + l, b] == out[n*B + b, l].
    """
    t, b = x.shape[0], x.shape[1]
    tail = x.shape[2:]
    n = t // seq_len
    return (
        x[: n * seq_len]
        .view(n, seq_len, b, *tail)
        .permute(0, 2, 1, *range(3, 3 + len(tail)))
        .reshape(n * b, seq_len, *tail)
    )


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        agent_in_dim = obs_dim + args.emb_dim
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(agent_in_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(agent_in_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(agent_in_dim, H, args.k_blocks, args.n_experts)
        # SUCCESSOR-FEATURE critic head. Same shape and MTP semantics as the HL-Gauss head
        # it replaces -- only the per-horizon target space changed, from 511 scalar-return
        # bucket logits to K = emb_dim + obs_dim + 2*act_dim + 1 occupancy dimensions.
        # Horizon h predicts Lambda_{t+h}, the vector-valued lambda-return of phi.
        # Zero-init is deliberate and not merely neutral: with psi_old == 0 the first
        # rollout's Lambda degenerates to the plain discounted sum of phi, i.e. a clean
        # Monte-Carlo target with no bootstrap from a random head.
        self.sf_dim = args.emb_dim + obs_dim + 2 * act_dim + 1  # [e, s, a, a*a, 1]
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * self.sf_dim), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
            self.critic_head.bias.zero_()
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
        # Returns successor features (B, mtp, sf_dim); horizon 0 is psi(s_t), whose
        # scalar readout w_r . psi_0 is V(s_t).
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)

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
        log_prob_dim = dist.log_prob(z) - log_det_fn(z)
        log_prob = log_prob_dim.sum(1)
        value_sf = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)
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
        if self.actor_dist == "beta":
            alpha, beta = dist.concentration1, dist.concentration0
        else:
            alpha = beta = torch.empty(0, device=x.device)
        return action, z, log_prob, log_prob_dim, entropy, value_sf, alpha, beta

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


def shape_advantage(gae, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform.

    The base's "tanh_std" and "cdf_probit" transforms are GONE, not stubbed. Both read
    the critic's per-state value DISTRIBUTION (sigma(s) in bin units, and u = Z(s)'s CDF
    at the return), and this variant has no distributional critic to read -- the head
    predicts successor features. Feeding them a constant sigma=1 / u=0 placeholder would
    make cdf_probit return the same constant for every sample (a zero policy gradient
    after norm_adv) while logging a perfectly healthy-looking curve, so the branches are
    deleted and the arg is rejected at startup instead.
    """
    if args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
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
    if args.critic_mtp_horizon != len(TD_BETAS):
        raise ValueError(
            f"critic_mtp_horizon must be {len(TD_BETAS)} for the fixed shell grid, "
            f"got {args.critic_mtp_horizon}"
        )
    if abs(args.gamma - TD_BETAS[-1]) > 1e-12:
        raise ValueError(
            f"gamma must equal the longest shell beta ({TD_BETAS[-1]:.6g}); "
            f"got {args.gamma}. Change the geometric grid together with gamma."
        )
    if args.seq_len != max(LEJEPA_HORIZONS) + 1:
        raise ValueError(
            f"seq_len must be {max(LEJEPA_HORIZONS) + 1} for the geometric LeJEPA grid"
        )
    if args.control_encoder_tau != 1.0:
        raise ValueError("control_encoder_tau must be 1: v8 forbids encoder EMA")
    if args.control_encoder_kl <= 0.0:
        raise ValueError("control_encoder_kl must be positive")
    if (
        args.control_encoder_emb_drift <= 0.0
        or args.control_encoder_sf_drift <= 0.0
        or args.control_encoder_value_drift <= 0.0
    ):
        raise ValueError("control encoder drift limits must be positive")
    if not args.vector_adv:
        raise ValueError("this variant always optimizes its learned vector trace")
    if args.norm_adv or args.adv_transform != "v10":
        raise ValueError("v8 requires vector whitening and forbids scalar advantage scaling")
    if args.auto_entropy:
        raise ValueError("v8 does not mix scalar entropy returns into its vector trace")
    if args.ent_coef != 0.0:
        raise ValueError("v8 forbids a scalar entropy term in its vector policy objective")
    if args.actor_dist != "beta":
        raise ValueError("v8's vector policy path currently requires the Beta actor")
    if not args.separate_grad_clip:
        raise ValueError("v8 requires separate_grad_clip for vector/critic gradient isolation")
    if args.num_envs < 2:
        raise ValueError("trace risk requires at least two independent environments")
    if (
        args.vector_whiten_ridge <= 0.0
        or args.vector_kl_coef < 0.0
        or args.router_ridge <= 0.0
        or not 0.0 < args.router_max_fraction <= 1.0
    ):
        raise ValueError("invalid vector whitening/KL configuration")
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

    # Fail loud at startup rather than degrading silently mid-run.
    if args.adv_transform in ("tanh_std", "cdf_probit"):
        raise ValueError(
            f"adv_transform={args.adv_transform!r} reads the critic's per-state value "
            "DISTRIBUTION, which this variant does not have (the head predicts successor "
            "features, not bucket logits). Use v10 / tanh_gae / clip_z / rankgauss*."
        )
    if args.auto_entropy:
        raise ValueError("auto_entropy is not implemented for fixed-horizon shell credit")
    n_seq_per_iter = (args.num_steps // args.seq_len) * args.num_envs
    if n_seq_per_iter < args.ssl_batch:
        raise ValueError(
            f"ssl_batch={args.ssl_batch} exceeds the {n_seq_per_iter} sequences a rollout "
            "yields ((num_steps // seq_len) * num_envs). The SSL loop drops the last ragged "
            "minibatch (SIGReg's statistic scales with batch size), so it would take ZERO "
            "steps: the encoder would stay at random init and every sf/ and gate/ metric "
            "would be computed on a frozen random projection."
        )

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
    if auto_alpha and args.vector_adv:
        raise ValueError(
            "vector_adv routes the policy advantage through w_r . E_t, which is a "
            "projection of the REWARD occupancy residual. The soft-advantage max-ent path "
            "adds alpha*H(s') to the bootstrap, and entropy has no phi coordinate, so "
            "there is no vector analogue -- the bonus would be silently dropped. Use "
            "--no-vector-adv, or actor_dist=beta (which disables auto_alpha anyway)."
        )
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    # --- LeJEPA successor-feature value pathway -------------------------------------
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim_sf = int(np.prod(envs.single_action_space.shape))
    sf_dim = agent.sf_dim

    # SSL nets live in their OWN top-level module, deliberately not a submodule of Agent:
    # otherwise their parameters would enter agent.parameters() (the PPO optimizer),
    # actor_parameters()/critic_parameters(), and the 0.25 clip budget -- silently changing
    # the very thing being held fixed.
    ssl = LeJepaSSL(obs_dim, act_dim_sf, args).to(device)
    # The JEPA objective remains symmetric and attached-target. This slow copy is only
    # the control interface: it prevents one SSL optimizer step from rotating the policy
    # and TD feature frame outside PPO's measured trust region.
    control_encoder = copy.deepcopy(ssl.encoder).requires_grad_(False).eval()
    control_rotation = torch.eye(args.emb_dim, device=device)
    control_bias = torch.zeros(args.emb_dim, device=device)

    def encode_control(states):
        return control_encoder(states) @ control_rotation + control_bias

    ssl_optimizer = optim.AdamW(
        ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay
    )
    trace_risk = DirectSuccessorRisk(
        obs_dim + args.emb_dim,
        sf_dim + 1,
        args.ssl_hidden,
        args.trace_variance_floor,
    ).to(device)
    trace_risk_optimizer = optim.Adam(
        trace_risk.parameters(), lr=args.trace_risk_lr, eps=1e-5
    )
    trace_risk_ready = False
    router_ready = False
    router_coefficients = torch.zeros(
        2 * act_dim_sf * (args.hidden + 1), sf_dim + 1, device=device
    )
    router_feature_mean = torch.zeros(args.hidden, device=device)
    router_feature_scale = torch.ones(args.hidden, device=device)

    if args.compile:
        # retain_graph in the dual-backward is incompatible with donated buffers.
        import torch._functorch.config
        import torch._dynamo

        torch._functorch.config.donated_buffer = False
        torch._dynamo.config.suppress_errors = True   # never let compile break the run
        torch._dynamo.config.recompile_limit = 64     # trunk alone sees 4 (shape, grad) combos
        if agent.share_backbone:
            agent.trunk = CompiledModule(agent.trunk, cudagraphs=False)
        else:
            agent.actor_trunk = CompiledModule(agent.actor_trunk, cudagraphs=False)
            agent.critic_trunk = CompiledModule(agent.critic_trunk, cudagraphs=False)
        for name in ("encoder", "action_encoder", "predictor"):
            setattr(ssl, name, CompiledModule(
                getattr(ssl, name), mode=args.compile_mode, cudagraphs=args.compile_ssl_cudagraphs
            ))

    # Normalized successor heads are all O(1), so no horizon-dependent loss scale or
    # moving target normalization is needed.
    sf_loss_scale = torch.ones(
        (len(TD_HORIZONS), 1), device=device, dtype=torch.float32
    )
    # w_r starts at zero => V == 0 for the first rollout, which is the correct "no
    # information" baseline rather than random noise driving the actor.
    w_r = torch.zeros(sf_dim, device=device)
    # ... which makes adv_vector = sf_residual @ w_r IDENTICALLY ZERO on iteration 1, and
    # a zero advantage means pg_loss == 0 and the actor gets no gradient at all from the
    # first 32,768 transitions. v2 was not degenerate here (its scalar GAE reduces to the
    # raw discounted reward sum when values == 0). Fall back to scalar GAE until w_r has
    # been solved once; from iteration 2 the projection is live.
    w_r_solved = False

    # Fixed INDEX set (not a fixed obs batch) for the frame-drift probe: the states are
    # re-drawn from each rollout, so the probe never goes stale w.r.t. the observation
    # normalizer. Stride 31 is COPRIME with num_envs; the buffer is T-major (i = t*B + e),
    # so a stride sharing a factor with num_envs (e.g. 32 at num_envs=16) would alias onto
    # a single environment and measure drift on one trajectory instead of the state marginal.
    drift_probe_idx = torch.arange(0, args.num_steps * args.num_envs, 31, device=device)[:1024]

    shell_widths = torch.as_tensor(
        TD_SHELL_WIDTHS, device=device, dtype=torch.float32
    )

    def psi_raw(normalized_shells):
        """Recover the unnormalized gamma-successor from positive shell weights."""
        shape = (1,) * (normalized_shells.ndim - 2) + (-1, 1)
        return (normalized_shells * shell_widths.view(shape)).sum(dim=-2)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_dim = torch.zeros(
        (args.num_steps, args.num_envs, act_dim_sf), device=device
    )
    old_alphas = torch.zeros_like(logprobs_dim)
    old_betas = torch.zeros_like(logprobs_dim)
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
            # The SSL optimizer is separate, so the base's anneal (which writes
            # param_groups[0] of `optimizer` only) does not reach it. Anneal it too: it
            # freezes the encoder frame late in training, which is exactly when the policy
            # is refining and a drifting frame would do the most damage to the bootstrap.
            ssl_optimizer.param_groups[0]["lr"] = frac * args.ssl_lr

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                next_emb = encode_control(next_obs)
                agent_obs = torch.cat([next_obs, next_emb], dim=-1)
                action, z, logprob, logprob_dim, ent, value_sf, alpha, beta = (
                    agent.get_action_and_value(agent_obs)
                )
                values[step] = psi_raw(value_sf) @ w_r
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            logprobs_dim[step] = logprob_dim
            old_alphas[step] = alpha
            old_betas[step] = beta

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
            flat_obs_shape = (-1,) + envs.single_observation_space.shape
            flat_obs = obs.reshape(flat_obs_shape)
            flat_next_obs = next_obses.reshape(flat_obs_shape)
            emb_buf = encode_control(flat_obs).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            next_emb_buf = encode_control(flat_next_obs).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            flat_agent_obs = torch.cat([flat_obs, emb_buf.reshape(-1, args.emb_dim)], dim=-1)
            flat_next_agent_obs = torch.cat(
                [flat_next_obs, next_emb_buf.reshape(-1, args.emb_dim)], dim=-1
            )
            successor_next = agent.get_value(flat_next_agent_obs).reshape(
                args.num_steps, args.num_envs, args.critic_mtp_horizon, sf_dim
            )
            psi_next = psi_raw(successor_next).reshape(
                args.num_steps, args.num_envs, sf_dim
            )
            # Kept as trunk features, not just psi: the EV gate below needs the SAME
            # features to fit its detached scalar probe, and this variant already does two
            # full-batch trunk forwards per iteration where the base did one.
            critic_feat_buf = agent._trunks(flat_agent_obs)[1]
            successor_cur = agent.critic_head(critic_feat_buf).view(
                args.num_steps,
                args.num_envs,
                args.critic_mtp_horizon,
                sf_dim,
            )
            psi_cur = psi_raw(successor_cur)
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
                _, _, boot_logprob, _, _, _, _, _ = agent.get_action_and_value(
                    flat_next_agent_obs.reshape(
                        args.num_steps, args.num_envs, -1
                    )[-1]
                )
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                next_value_bonus = None
            # ---- SUCCESSOR-FEATURE basis --------------------------------------------
            phi = phi_features(emb_buf, obs, actions)         # (T, B, sf_dim)

            # ---- truncated-MC scaffolding (moved AHEAD of Lambda) --------------------
            # `avail[t]` counts reward terms actually present in mc_ret[t]: it resets at a
            # reset boundary and at the rollout tail, so masking on avail >= mc_window
            # removes BOTH the boundary bias and the tail bias in one condition. The EV
            # gate below still consumes mc_ret/mc_mask; the tau estimator needs them here.
            #
            # mc_phi is the SAME recursion run element-wise on phi, i.e. the lambda=1
            # discounted feature sum. It is what makes the tau estimate below
            # LAMBDA-INDEPENDENT, which is the whole reason it is computed separately
            # instead of being read off sf_std: sf_std is the spread of Lambda(lambda),
            # so deriving lambda from it would close a feedback loop whose only stable
            # point is lambda_max.
            mc_ret = torch.zeros_like(rewards)
            mc_phi = torch.zeros_like(phi)
            mc_avail = torch.zeros_like(rewards)
            mc_run = torch.zeros_like(rewards[0])
            mc_phi_run = torch.zeros_like(phi[0])
            avail_run = torch.zeros_like(rewards[0])
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                mc_run = rewards[t] + args.gamma * cont * mc_run
                mc_phi_run = (
                    phi[t] + args.gamma * cont.unsqueeze(-1) * mc_phi_run
                )
                avail_run = 1.0 + cont * avail_run
                mc_ret[t] = mc_run
                mc_phi[t] = mc_phi_run
                mc_avail[t] = avail_run
            mc_mask = (mc_avail >= args.mc_window).reshape(-1)
            n_mc = int(mc_mask.sum().item())

            # ---- vector TD(lambda) ---------------------------------------------------
            # Element-wise on phi, mirroring the reward GAE. THE TWO MASKS ARE NOT THE SAME
            # MASK. They differ exactly at a TRUNCATION (term=0, valid=1, boundary=1):
            # bootstrap through it, but CUT the lambda trace. Collapsing them would let the
            # next episode's discounted phi-sum bleed backward into this episode's target
            # from a fresh, near-zero-velocity reset state -- ~1.6% of samples corrupted by
            # roughly the reset-vs-running value gap. Written in residual space so the
            # rollout tail (E_T = 0) is handled for free.
            td_betas = torch.as_tensor(TD_BETAS, device=device, dtype=phi.dtype)
            bootstrap = (1.0 - transition_terminations) * transition_valids
            continuation = 1.0 - transition_boundaries
            # Each head predicts one unit-mass temporal shell. Use every available
            # same-episode suffix sample for critic supervision; the actor separately
            # accumulates one-step innovations, so suffix targets cannot double-count
            # policy credit.
            sf_target = build_full_suffix_shell_targets(
                phi,
                successor_next,
                bootstrap,
                continuation,
                td_betas,
                shell_widths,
            )
            one_step_target = build_normalized_shell_targets(
                phi, successor_next, bootstrap, td_betas, shell_widths
            )
            # Positive shell weights reconstruct the ordinary vector Bellman innovation
            # for M_gamma. Shells are one decomposition, not separate policy objectives.
            delta_sf = psi_raw(one_step_target - successor_cur)

            # Fit the immediate-reward covector fresh on this rollout. It is not used to
            # construct a scalar sample advantage; it is transformed only after the vector
            # trace has been learned and whitened.
            flat_phi = phi.reshape(-1, sf_dim)
            flat_rew = rewards.reshape(-1)
            w_r = solve_reward_probe(flat_phi, flat_rew, args.sf_ridge)
            w_r_solved = True
            reward_resid = flat_rew - flat_phi @ w_r
            reward_residual_prev = reward_resid.reshape(
                args.num_steps, args.num_envs
            )
            current_values = psi_raw(successor_cur) @ w_r
            mc_resid = torch.zeros_like(rewards)
            residual_run = torch.zeros_like(rewards[0])
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                residual_run = (
                    reward_residual_prev[t]
                    + args.gamma * cont * residual_run
                )
                mc_resid[t] = residual_run

            # Hard-lagged learned hazard. The direct model was fitted only after the prior
            # rollout's actor update. It predicts the direct conditional mean d and the
            # previous recursive estimator's mean g and variance M. For
            # b=bootstrap, minimizing ||b + lambda(g-b) - d||^2 + lambda^2 M gives the
            # closed-form state hazard below.
            trace_was_ready = trace_risk_ready
            if trace_was_ready:
                (
                    direct_mean,
                    direct_variance,
                    recursive_mean,
                    recursive_variance,
                ) = trace_risk(
                    flat_agent_obs.reshape(
                        args.num_steps, args.num_envs, -1
                    )
                )
                bootstrap_mean = torch.cat(
                    [
                        (1.0 - args.gamma) * psi_raw(successor_cur),
                        ((1.0 - args.gamma) * current_values).unsqueeze(-1),
                    ],
                    dim=-1,
                )
                bootstrap_error = bootstrap_mean - direct_mean
                learned_state_lambda = optimal_recursive_lambda(
                    bootstrap_mean,
                    direct_mean,
                    recursive_mean,
                    recursive_variance,
                ).detach()
                trace_bias = bootstrap_error.square().mean(-1)
                trace_noise = recursive_variance.mean(-1)
            else:
                direct_mean = delta_sf.new_zeros(
                    args.num_steps, args.num_envs, sf_dim + 1
                )
                direct_variance = torch.ones_like(direct_mean)
                recursive_mean = torch.zeros_like(direct_mean)
                recursive_variance = torch.ones_like(direct_mean)
                trace_bias = torch.zeros_like(rewards)
                trace_noise = torch.ones_like(rewards)
                learned_state_lambda = torch.full_like(rewards, args.gae_lambda)
            state_lambda = (
                learned_state_lambda
                if args.learn_trace
                else torch.full_like(rewards, args.gae_lambda)
            )
            if n_mc:
                current_direct_target = (
                    (1.0 - args.gamma)
                    * torch.cat(
                        [mc_phi, mc_ret.unsqueeze(-1)], dim=-1
                    ).reshape(-1, sf_dim + 1)[mc_mask]
                )
                current_direct_prediction = direct_mean.reshape(
                    -1, sf_dim + 1
                )[mc_mask]
                trace_direct_mse = (
                    current_direct_prediction - current_direct_target
                ).square().mean().item()
                trace_direct_calibration = (
                    (current_direct_prediction - current_direct_target).square()
                    / direct_variance.reshape(-1, sf_dim + 1)[mc_mask]
                ).mean().item()
            else:
                trace_direct_mse = float("nan")
                trace_direct_calibration = float("nan")

            delta_vector = torch.cat(
                [delta_sf, reward_residual_prev.unsqueeze(-1)], dim=-1
            )
            vector_advantage = learned_vector_gae(
                delta_vector,
                continuation,
                args.gamma,
                state_lambda,
            )
            reward_covector = torch.cat([w_r, w_r.new_ones(1)])
            (
                policy_adv,
                policy_covector,
                whiten_eigenvalues,
                inverse_sqrt_covariance,
                sqrt_covariance,
            ) = whiten_vector_advantage(
                vector_advantage, reward_covector, args.vector_whiten_ridge
            )
            advantages = vector_advantage @ reward_covector
            centered_advantages = advantages - advantages.mean()
            scalar_advantage_scale = centered_advantages.std().clamp_min(1e-8)
            policy_covector = policy_covector / scalar_advantage_scale

            actor_feature_buffer = agent._trunks(flat_agent_obs)[0].detach()
            fit_router_features, action_score, fit_feature_mean, fit_feature_scale = (
                action_router_features(
                    actor_feature_buffer,
                    latent_zs.reshape(-1, act_dim_sf),
                    old_alphas.reshape(-1, act_dim_sf),
                    old_betas.reshape(-1, act_dim_sf),
                )
            )
            lagged_router_features = torch.cat(
                [
                    (actor_feature_buffer - router_feature_mean)
                    / router_feature_scale,
                    actor_feature_buffer.new_ones(actor_feature_buffer.shape[0], 1),
                ],
                dim=-1,
            )
            if router_ready:
                routed_policy_adv, routed_component, router_scale = (
                    routed_vector_advantage(
                        lagged_router_features,
                        action_score,
                        router_coefficients,
                        inverse_sqrt_covariance,
                        policy_adv.reshape(-1, sf_dim + 1),
                        args.router_max_fraction,
                    )
                )
            else:
                routed_policy_adv = policy_adv.reshape(
                    -1, sf_dim + 1
                ).unsqueeze(1).expand(-1, act_dim_sf, -1)
                routed_component = torch.zeros_like(routed_policy_adv)
                router_scale = policy_adv.new_zeros(())

            # Scalar quantities below are diagnostics/return reporting only. The PPO
            # backward receives policy_adv as a vector and policy_covector as its final VJP.
            vector_projection_error = (
                policy_adv @ policy_covector
                - centered_advantages / scalar_advantage_scale
            ).abs().max().item()
            returns = advantages + current_values
            current_recursive_target = (1.0 - args.gamma) * torch.cat(
                [
                    psi_raw(successor_cur) + vector_advantage[..., :-1],
                    returns.unsqueeze(-1),
                ],
                dim=-1,
            )
            if n_mc:
                current_recursive_prediction = recursive_mean.reshape(
                    -1, sf_dim + 1
                )[mc_mask]
                masked_recursive_target = current_recursive_target.reshape(
                    -1, sf_dim + 1
                )[mc_mask]
                trace_recursive_mse = (
                    current_recursive_prediction - masked_recursive_target
                ).square().mean().item()
                trace_recursive_calibration = (
                    (
                        current_recursive_prediction
                        - masked_recursive_target
                    ).square()
                    / recursive_variance.reshape(-1, sf_dim + 1)[mc_mask]
                ).mean().item()
            else:
                trace_recursive_mse = trace_recursive_calibration = float("nan")
            adv_vector = vector_advantage[..., :-1] @ w_r
            residual_trace = vector_advantage[..., -1]

            fixed_lambda = torch.full_like(state_lambda, args.gae_lambda)
            fixed_vector_advantage = learned_vector_gae(
                delta_vector,
                continuation,
                args.gamma,
                fixed_lambda,
            )
            adv_scalar_gae = fixed_vector_advantage @ reward_covector
            _a, _b = advantages.reshape(-1), adv_scalar_gae.reshape(-1)
            adv_vec_corr = float(
                ((_a - _a.mean()) * (_b - _b.mean())).mean()
                / (_a.std().clamp_min(1e-12) * _b.std().clamp_min(1e-12))
            )
            reward_r2 = 1.0 - (reward_resid.var() / flat_rew.var().clamp_min(1e-12)).item()
            # R^2 alone is the WRONG gate: r_t uses the frame-skip AVERAGED velocity while the
            # observation carries instantaneous qvel, so a structural residual of ~1-3% is
            # expected and harmless IF it is white. What costs EV is a gait-phase-locked,
            # speed-correlated residual, which shows up as autocorrelation, not as R^2.
            resid_tb = reward_resid.reshape(args.num_steps, args.num_envs)
            resid_c = resid_tb - resid_tb.mean()
            resid_var = resid_c.var().clamp_min(1e-12)
            reward_resid_ac = [
                ((resid_c[:-k] * resid_c[k:]).mean() / resid_var).item() for k in (1, 5, 10, 20)
            ]

            # Every state has one target for every temporal band. There is no shifted-MTP
            # mask: termination is represented by a zero bootstrap in the Bellman target.
            sf_mtp = sf_target
            return_mtp_mask = torch.ones(
                (args.num_steps, args.num_envs, args.critic_mtp_horizon),
                dtype=torch.bool,
                device=device,
            )

            # ---- three-way EV gate: the falsifier, computed on an UNRIGGED target ----
            # The usual EV target `b_returns = advantages + values` CONTAINS the critic's
            # own bootstrapped values, and at dt=0.05 the errors are strongly
            # state-autocorrelated -- a critic scores well against a target built from
            # itself. Every predictor below is scored against the same truncated
            # Monte-Carlo discounted return instead.
            #
            if n_mc >= 256:
                flat_mc = mc_ret.reshape(-1)[mc_mask]
                feat_mc = critic_feat_buf[mc_mask]
                ones_mc = feat_mc.new_ones(feat_mc.shape[0], 1)
                # (1) reference: what a scalar critic reading THIS trunk could achieve.
                #     Detached closed-form probe, so no scalar-return gradient ever enters
                #     the network -- it replaces the in-run HL-Gauss comparator that the
                #     head swap removed.
                trunk_feat_mc = torch.cat([feat_mc, ones_mc], dim=-1)
                ev_trunk_probe = ev_score(
                    trunk_feat_mc @ solve_reward_probe(trunk_feat_mc, flat_mc, args.sf_ridge), flat_mc
                )
                # (3) how much reward-relevant signal is LINEARLY present in the
                #     INSTANTANEOUS embedding. Not a ceiling on psi: psi reads the trunk,
                #     not e_t, and accumulates phi over time, so (2) > (3) is expected and
                #     is in fact the successor-feature construction earning its keep.
                #     Treat (3) as a floor-diagnostic on the encoder, not an upper bound.
                e_mc = torch.cat([emb_buf.reshape(-1, args.emb_dim)[mc_mask], ones_mc], dim=-1)
                ev_latent_cap = ev_score(
                    e_mc @ solve_reward_probe(e_mc, flat_mc, args.sf_ridge), flat_mc
                )
                # THE decisive reward-probe metric, and the reason R^2 and the AC lags are
                # both kept only as secondary reads. Since V = w_r.Lambda and
                # Lambda = sum gamma^k phi, we have w_r.Lambda = G_t - sum gamma^k eps_{t+k}:
                # the value error IS the discounted sum of probe residuals. R^2 alone is
                # actively misleading here -- measured offline, adding s' to phi raises R^2
                # from 0.61 to 0.94 while making THIS number WORSE (0.77 -> 0.84), because
                # the discounted sum amplifies a small correlated residual far more than a
                # large white one. Ratio to the value's own spread; lower is better.
                value_err_frac = float(
                    mc_resid.reshape(-1)[mc_mask].std() / flat_mc.std().clamp_min(1e-12)
                )
                # (2) the treatment: the ONLINE value the actor actually consumed.
                ev_sf = ev_score(values.reshape(-1)[mc_mask], flat_mc)
                # (1) and (3) are IN-SAMPLE ridge fits, i.e. optimistic upper bounds; (2) is
                # an honest out-of-sample prediction. So (3) vs (1) is the apples-to-apples
                # comparison -- (3) << (1) means the LATENT is the bottleneck and no amount
                # of psi quality can rescue this. (2) << (3) with (3) ~ (1) instead points
                # at the SF machinery (recursion, standardization), not the encoder.
            else:
                ev_trunk_probe = ev_latent_cap = ev_sf = value_err_frac = float("nan")

        b_obs = flat_agent_obs
        b_logprobs = logprobs.reshape(-1)
        b_logprobs_dim = logprobs_dim.reshape(-1, act_dim_sf)
        b_old_alphas = old_alphas.reshape(-1, act_dim_sf)
        b_old_betas = old_betas.reshape(-1, act_dim_sf)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1, sf_dim + 1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_sf_target = sf_mtp.reshape(-1, args.critic_mtp_horizon, sf_dim)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)
        # The policy buffer remains vector-valued. policy_covector is applied only to the
        # vector loss's VJP, never to individual samples.
        b_policy_adv = b_advantages
        b_routed_policy_adv = routed_policy_adv
        gae = advantages.reshape(-1)  # scalar diagnostic only
        adv_corr = 1.0
        adv_sign_agree = 1.0

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                (
                    _,
                    _,
                    newlogprob,
                    newlogprob_dim,
                    entropy,
                    value_sf,
                    new_alpha,
                    new_beta,
                ) = agent.get_action_and_value(b_obs[mb_inds], b_latent_zs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                marginal_logratio = (
                    newlogprob_dim - b_logprobs_dim[mb_inds]
                )
                marginal_ratio = marginal_logratio.exp()
                kl_loss = kl_divergence(
                    Beta(b_old_alphas[mb_inds], b_old_betas[mb_inds]),
                    Beta(new_alpha, new_beta),
                ).sum(-1).mean()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipped = (ratio < 1.0 - args.clip_coef) | (
                        ratio > 1.0 + args.clip_coef_high
                    )
                    clipfracs.append(clipped.float().mean().item())

                mb_vector_advantage = b_policy_adv[mb_inds]
                policy_loss_vector = vector_policy_surrogate(
                    ratio,
                    marginal_ratio,
                    mb_vector_advantage,
                    b_routed_policy_adv[mb_inds],
                    policy_covector,
                    args.clip_coef,
                    args.clip_coef_high,
                )
                pg_loss = (policy_loss_vector.detach() * policy_covector).sum()

                # Each head predicts a complete normalized successor embedding.
                sf_tgt = b_sf_target[mb_inds]
                sf_raw_err = value_sf - sf_tgt
                sf_err = sf_raw_err / sf_loss_scale
                value_mask = b_target_mask[mb_inds].to(sf_err.dtype)          # (B, mtp)
                v_loss = (sf_err.pow(2).mean(-1) * value_mask).sum(dim=-1).mean()
                if args.sf_alpha < 1.0:
                    # Reward-direction term. Its target w_r.Lambda still comes from the
                    # LATENT lambda-return, so the bootstrap stays in latent space and the
                    # thesis is intact; sf_alpha only decides how much of psi's capacity is
                    # spent on the one direction that is actually read out.
                    horizon_scales = shell_widths.view(1, -1)
                    scalar_err = (
                        (sf_raw_err * w_r).sum(-1) * horizon_scales
                    ).pow(2)
                    v_loss = args.sf_alpha * v_loss + (1.0 - args.sf_alpha) * (
                        scalar_err * value_mask
                    ).sum(dim=-1).mean()
                # Per-horizon psi MSE (last minibatch of the last epoch is what gets logged).
                sf_per_h_mse = (sf_err.detach().pow(2).mean(-1) * value_mask).sum(0) / value_mask.sum(
                    0
                ).clamp_min(1)
                sf_per_h_raw_mse = (
                    sf_raw_err.detach().pow(2).mean(-1) * value_mask
                ).sum(0) / value_mask.sum(0).clamp_min(1)

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
                    torch.autograd.backward(
                        (policy_loss_vector, kl_loss),
                        (
                            policy_covector,
                            kl_loss.new_tensor(args.vector_kl_coef),
                        ),
                    )
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    raise RuntimeError("v8 requires separate_grad_clip")

            if args.target_kl is not None and kl_loss > args.target_kl:
                break

        # Fit only after the actor has consumed the rollout, then expose the coefficients
        # on the next rollout. This prevents same-batch action noise from becoming an
        # adaptive policy weight. The model predicts vector trace innovations, not reward.
        with torch.no_grad():
            flat_environment = (
                torch.arange(args.batch_size, device=device) % args.num_envs
            )
            router_train_mask = flat_environment.remainder(2).eq(iteration % 2)
            router_validation_mask = ~router_train_mask
            (
                next_router_coefficients,
                router_fit_r2,
                router_effective_rank,
                router_mode_gain,
            ) = fit_action_router(
                fit_router_features,
                action_score,
                vector_advantage.reshape(-1, sf_dim + 1)
                - vector_advantage.reshape(-1, sf_dim + 1).mean(0, keepdim=True),
                args.router_ridge,
                router_train_mask,
                router_validation_mask,
                inverse_sqrt_covariance,
                sqrt_covariance,
            )
            router_coefficients = next_router_coefficients
            router_feature_mean = fit_feature_mean
            router_feature_scale = fit_feature_scale
            router_was_ready = router_ready
            router_ready = True

        # ---- hard-lagged direct successor risk model ---------------------------------
        # These labels are direct 500-step suffixes, never the critic's bootstrap. The
        # updated model cannot affect the rollout that trained it; it is first consumed on
        # the next rollout. Ordinary optimizer state is learned state, not an EMA target.
        trace_risk_loss = trace_risk_gn = float("nan")
        trace_risk_validation_nll = trace_risk_validation_calibration = float("nan")
        trace_risk_direct_skill = trace_risk_direct_vs_bootstrap_skill = float("nan")
        trace_risk_recursive_skill = float("nan")
        trace_risk_steps = 0
        risk_indices = mc_mask.nonzero(as_tuple=True)[0]
        risk_environment = risk_indices.remainder(args.num_envs)
        risk_train_selector = risk_environment.remainder(2).eq(iteration % 2)
        risk_train_indices = risk_indices[risk_train_selector]
        risk_validation_indices = risk_indices[~risk_train_selector]
        if (
            risk_train_indices.numel() >= args.trace_risk_min_samples
            and risk_validation_indices.numel() >= 256
        ):
            flat_direct_target = (
                (1.0 - args.gamma)
                * torch.cat(
                    [mc_phi, mc_ret.unsqueeze(-1)], dim=-1
                ).reshape(-1, sf_dim + 1)
            ).detach()
            flat_recursive_target = current_recursive_target.reshape(
                -1, sf_dim + 1
            ).detach()
            risk_x = flat_agent_obs[risk_train_indices].detach()
            risk_direct_y = flat_direct_target[risk_train_indices]
            risk_recursive_y = flat_recursive_target[risk_train_indices]
            risk_loss_sum = risk_gn_sum = 0.0
            for _ in range(args.trace_risk_epochs):
                permutation = torch.randperm(
                    risk_train_indices.numel(), device=device
                )
                for start in range(
                    0, risk_train_indices.numel(), args.trace_risk_batch
                ):
                    batch = permutation[start : start + args.trace_risk_batch]
                    (
                        direct_prediction,
                        direct_prediction_variance,
                        recursive_prediction,
                        recursive_prediction_variance,
                    ) = trace_risk(risk_x[batch])
                    direct_error = (
                        risk_direct_y[batch] - direct_prediction
                    )
                    recursive_error = (
                        risk_recursive_y[batch] - recursive_prediction
                    )
                    direct_nll = (
                        direct_error.square() / direct_prediction_variance
                        + direct_prediction_variance.log()
                    ).mean()
                    recursive_nll = (
                        recursive_error.square() / recursive_prediction_variance
                        + recursive_prediction_variance.log()
                    ).mean()
                    risk_loss = 0.5 * (direct_nll + recursive_nll)
                    trace_risk_optimizer.zero_grad(set_to_none=True)
                    risk_loss.backward()
                    risk_gn = nn.utils.clip_grad_norm_(trace_risk.parameters(), 1.0)
                    trace_risk_optimizer.step()
                    risk_loss_sum += risk_loss.item()
                    risk_gn_sum += float(risk_gn)
                    trace_risk_steps += 1
            trace_risk_loss = risk_loss_sum / max(trace_risk_steps, 1)
            trace_risk_gn = risk_gn_sum / max(trace_risk_steps, 1)
            with torch.no_grad():
                (
                    validation_direct_mean,
                    validation_direct_variance,
                    validation_recursive_mean,
                    validation_recursive_variance,
                ) = trace_risk(flat_agent_obs[risk_validation_indices])
                validation_direct_error = (
                    flat_direct_target[risk_validation_indices]
                    - validation_direct_mean
                )
                validation_recursive_error = (
                    flat_recursive_target[risk_validation_indices]
                    - validation_recursive_mean
                )
                validation_nll = 0.5 * (
                    validation_direct_error.square() / validation_direct_variance
                    + validation_direct_variance.log()
                    + validation_recursive_error.square()
                    / validation_recursive_variance
                    + validation_recursive_variance.log()
                )
                trace_risk_validation_nll = validation_nll.mean().item()
                trace_risk_validation_calibration = (
                    0.5
                    * (
                        validation_direct_error.square()
                        / validation_direct_variance
                        + validation_recursive_error.square()
                        / validation_recursive_variance
                    )
                ).mean().item()
                validation_direct_target = flat_direct_target[
                    risk_validation_indices
                ]
                validation_recursive_target = flat_recursive_target[
                    risk_validation_indices
                ]
                direct_model_mse = validation_direct_error.square().mean()
                recursive_model_mse = validation_recursive_error.square().mean()
                direct_constant_mse = (
                    validation_direct_target - risk_direct_y.mean(0)
                ).square().mean()
                recursive_constant_mse = (
                    validation_recursive_target - risk_recursive_y.mean(0)
                ).square().mean()
                flat_bootstrap_target = (1.0 - args.gamma) * torch.cat(
                    [
                        psi_raw(successor_cur),
                        current_values.unsqueeze(-1),
                    ],
                    dim=-1,
                ).reshape(-1, sf_dim + 1)
                bootstrap_mse = (
                    validation_direct_target
                    - flat_bootstrap_target[risk_validation_indices]
                ).square().mean()
                trace_risk_direct_skill = (
                    1.0 - direct_model_mse / direct_constant_mse.clamp_min(1e-12)
                ).item()
                trace_risk_direct_vs_bootstrap_skill = (
                    1.0 - direct_model_mse / bootstrap_mse.clamp_min(1e-12)
                ).item()
                trace_risk_recursive_skill = (
                    1.0
                    - recursive_model_mse
                    / recursive_constant_mse.clamp_min(1e-12)
                ).item()
                trace_risk_ready = bool(
                    np.isfinite(trace_risk_validation_nll)
                    and trace_risk_direct_skill > 0.0
                    and 0.05
                    <= trace_risk_validation_calibration
                    <= 20.0
                )
        else:
            trace_risk_ready = False

        # ---- LeJEPA SSL step -----------------------------------------------------------
        # ONCE PER ITERATION, OUTSIDE the 320-minibatch PPO loop: inside it costs +100-200%
        # wall clock, outside it costs +10-20%. Placed AFTER the PPO update so the encoder
        # frame is frozen across the entire target-construction + critic-fitting phase. The
        # residual drift is one ITERATION's worth (ssl_epochs * (n_seq // ssl_batch) = 64
        # steps at defaults, not one), which is exactly what ssl/frame_drift_* measures.
        with torch.no_grad():
            seq_obs = chunk_sequences(obs, args.seq_len)
            seq_act = chunk_sequences(actions, args.seq_len)
            seq_cont = chunk_sequences(1.0 - transition_boundaries, args.seq_len)
        n_seq = seq_obs.shape[0]
        ssl_pred_l = ssl_sig_l = ssl_gn_sum = 0.0
        ssl_horizon_l = torch.zeros(len(LEJEPA_HORIZONS), device=device)
        ssl_steps = 0
        # e BEFORE this iteration's SSL step, on states from THIS rollout. Paired with the
        # post-step embedding of the SAME inputs below, this isolates frame movement from
        # distribution shift without needing to keep a copy of the old encoder.
        probe_obs = obs.reshape(flat_obs_shape)[drift_probe_idx]
        probe_control_before = emb_buf.reshape(-1, args.emb_dim)[drift_probe_idx].clone()
        with torch.no_grad():
            probe_online_before = ssl.encoder(probe_obs).clone()
        for _ in range(args.ssl_epochs):
            perm = torch.randperm(n_seq, device=device)
            # DROP-LAST, not for tidiness: the SIGReg statistic scales with the batch size
            # (it multiplies by proj.size(-2)), so a ragged final minibatch would silently
            # reweight the regularizer -- and under dynamic=False it also forces a recompile.
            for s in range(0, n_seq - args.ssl_batch + 1, args.ssl_batch):
                idx = perm[s : s + args.ssl_batch]
                ssl_loss, pred_l, sig_l, horizon_l = ssl(
                    seq_obs[idx], seq_act[idx], seq_cont[idx], args.sigreg_weight
                )
                ssl_optimizer.zero_grad(set_to_none=True)
                ssl_loss.backward()
                ssl_gn = nn.utils.clip_grad_norm_(ssl.parameters(), args.ssl_grad_clip)
                ssl_optimizer.step()
                ssl_pred_l += pred_l.item()
                ssl_sig_l += sig_l.item()
                ssl_horizon_l += horizon_l
                ssl_gn_sum += float(ssl_gn)
                ssl_steps += 1
        ssl_pred_l /= max(ssl_steps, 1)
        ssl_sig_l /= max(ssl_steps, 1)
        ssl_horizon_l /= max(ssl_steps, 1)
        ssl_gn_sum /= max(ssl_steps, 1)

        # ---- encoder health + frame drift ----------------------------------------------
        # SIGReg pins the DISTRIBUTION of e, not its coordinate frame: N(0,I) is
        # rotation-invariant and the prediction loss cannot pin the frame either (the
        # predictor co-rotates). But psi_old and w_r are functions of COORDINATES, so a
        # drifting frame makes the bootstrap target stale. This is a non-stationarity the
        # design INTRODUCES, so it is measured rather than pre-emptively patched.
        #   frame_drift_raw ~ 1 and frame_drift_rot ~ 1  -> stable, bootstrap is sound
        #   frame_drift_raw << 1 but frame_drift_rot ~ 1 -> pure rotation, the failure mode
        # frame_drift_rot is in [0,1] by von Neumann's trace inequality; frame_drift_raw is
        # in [-1,1] (it goes negative under anti-correlation). Both are 1 iff no drift.
        #
        # Both embeddings are of the SAME inputs drawn from THIS rollout, taken before and
        # after this iteration's SSL step. A probe frozen at iteration 1 would be wrong in
        # two ways: those are post-NormalizeObservation vectors carrying iteration-1
        # running stats, so by mid-training they decode to raw states the agent never
        # visits -- and OOD inputs drift MORE, so the metric would cry wolf about a frame
        # instability the actual bootstrap never sees.
        with torch.no_grad():
            probe_emb_after = ssl.encoder(probe_obs).float()
            a = probe_online_before - probe_online_before.mean(0)
            b_ = probe_emb_after - probe_emb_after.mean(0)
            na, nb = a.pow(2).sum(), b_.pow(2).sum()
            denom = (0.5 * (na + nb)).clamp_min(1e-12)
            drift_raw = float(1.0 - (b_ - a).pow(2).sum() / (2.0 * denom))
            drift_rot = float(torch.linalg.svdvals(b_.T.double() @ a.double()).sum() / denom.double())
            # e's marginal: SIGReg should drive these to (0, I). Read from the POST-step
            # embedding, so this panel and ssl/sigreg_epps_pulley describe the same encoder
            # -- emb_buf is the PRE-step encoder and would lag by a full 64 SSL steps,
            # making the very first ssl/emb_std point describe the random init.
            # Effective rank is expected near 17 (the observation dimension), NOT 32: a
            # single-state encoder cannot exceed the manifold's rank, and that is not a
            # failure -- random 1-D projections of a rank-17 pushforward still look
            # near-Gaussian by mixing, so the statistic converges anyway.
            emb_mean_abs = float(probe_emb_after.mean(0).abs().mean())
            emb_std = float(probe_emb_after.std(0).mean())
            cov_e = (b_.T @ b_) / b_.shape[0]
            eig = torch.linalg.eigvalsh(cov_e.double()).clamp_min(0)
            p_eig = eig / eig.sum().clamp_min(1e-12)
            eff_rank = float(torch.exp(-(p_eig * (p_eig + 1e-12).log()).sum()))

            # Align each candidate control frame back to the frame used for this rollout,
            # then line-search its nonlinear residual against the actual policy and raw
            # successor predictions. Pure SIGReg rotations are removed analytically; only
            # genuinely new representation content is admitted through the drift gates.
            old_control_params = [
                parameter.detach().clone() for parameter in control_encoder.parameters()
            ]
            old_agent_probe = torch.cat(
                [probe_obs, probe_control_before], dim=-1
            )
            old_actor_feat, old_critic_feat = agent._trunks(old_agent_probe)
            old_policy = agent._actor_dist(old_actor_feat)[0]
            old_successors = agent.critic_head(old_critic_feat).view(
                -1, args.critic_mtp_horizon, sf_dim
            )
            accepted_rate = args.control_encoder_tau
            encoder_policy_kl = probe_obs.new_tensor(float("inf"))
            encoder_emb_drift = probe_obs.new_tensor(float("inf"))
            encoder_sf_drift = probe_obs.new_tensor(float("inf"))
            encoder_value_drift = probe_obs.new_tensor(float("inf"))
            old_probe_values = psi_raw(old_successors) @ w_r
            # Hard snapshot only: either the fully updated encoder is admitted after
            # analytic frame alignment, or the prior snapshot remains. There is no
            # interpolation, Polyak update, or other EMA.
            for parameter, online_parameter in zip(
                control_encoder.parameters(), ssl.encoder.parameters()
            ):
                parameter.copy_(online_parameter.detach())
            candidate_base = control_encoder(probe_obs)
            (
                candidate_rotation,
                candidate_bias,
                probe_control_after,
            ) = align_latent_frame(candidate_base, probe_control_before)
            new_actor_feat, new_critic_feat = agent._trunks(
                torch.cat([probe_obs, probe_control_after], dim=-1)
            )
            new_policy = agent._actor_dist(new_actor_feat)[0]
            encoder_policy_kl = kl_divergence(old_policy, new_policy).sum(-1).mean()
            encoder_emb_drift = (
                (probe_control_after - probe_control_before)
                .pow(2)
                .mean()
                .sqrt()
                / probe_control_before.pow(2).mean().sqrt().clamp_min(1e-6)
            )
            new_successors = agent.critic_head(new_critic_feat).view(
                -1, args.critic_mtp_horizon, sf_dim
            )
            encoder_sf_drift = (
                ((new_successors - old_successors) / sf_loss_scale)
                .pow(2)
                .mean()
                .sqrt()
            )
            new_probe_values = psi_raw(new_successors) @ w_r
            encoder_value_drift = (
                (new_probe_values - old_probe_values).pow(2).mean().sqrt()
                / old_probe_values.std().clamp_min(1.0)
            )
            accepted = bool(
                encoder_policy_kl <= args.control_encoder_kl
                and encoder_emb_drift <= args.control_encoder_emb_drift
                and encoder_sf_drift <= args.control_encoder_sf_drift
                and encoder_value_drift <= args.control_encoder_value_drift
            )
            if accepted:
                control_rotation = candidate_rotation
                control_bias = candidate_bias
            else:
                accepted_rate = 0.0
                for parameter, old_parameter in zip(
                    control_encoder.parameters(), old_control_params
                ):
                    parameter.copy_(old_parameter)
                probe_control_after = probe_control_before

            ca = probe_control_before - probe_control_before.mean(0)
            cb = probe_control_after - probe_control_after.mean(0)
            control_denom = (0.5 * (ca.pow(2).sum() + cb.pow(2).sum())).clamp_min(
                1e-12
            )
            control_drift_raw = float(
                1.0 - (cb - ca).pow(2).sum() / (2.0 * control_denom)
            )

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


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
        writer.add_scalar("losses/exact_beta_kl", kl_loss.item(), global_step)
        writer.add_scalar(
            "vector_policy/marginal_logratio_abs_p99",
            marginal_logratio.abs().quantile(0.99).item(),
            global_step,
        )
        writer.add_scalar(
            "vector_policy/marginal_logratio_abs_max",
            marginal_logratio.abs().max().item(),
            global_step,
        )
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)

        # ---- successor-feature diagnostics -------------------------------------
        # R^2 is the weak gate; a structural ~1-3% residual is expected (frame-skip
        # averaged reward velocity vs instantaneous qvel). The residual's
        # AUTOCORRELATION is what actually costs EV: white residual at R^2=0.98
        # costs ~0.2% EV, a gait-phase-locked one at the same R^2 costs ~2%.
        writer.add_scalar("sf/value_err_frac", value_err_frac, global_step)
        writer.add_scalar("sf/reward_probe_r2", reward_r2, global_step)
        for lag, ac in zip((1, 5, 10, 20), reward_resid_ac):
            writer.add_scalar(f"sf/reward_resid_ac_lag{lag}", ac, global_step)
        for h, horizon in enumerate(TD_HORIZONS):
            writer.add_scalar(
                f"normshell/mse_horizon_{horizon}",
                sf_per_h_mse[h].item(),
                global_step,
            )
            writer.add_scalar(
                f"normshell/raw_mse_horizon_{horizon}",
                sf_per_h_raw_mse[h].item(),
                global_step,
            )
            writer.add_scalar(
                f"normshell/target_rms_horizon_{horizon}",
                sf_target[..., h, :].pow(2).mean().sqrt().item(),
                global_step,
            )
        writer.add_scalar("sf/w_r_norm", w_r.norm().item(), global_step)

        # Learned vector-trace diagnostics.
        writer.add_scalar("advB/adv_vec_corr", adv_vec_corr, global_step)
        writer.add_scalar("trace/corr_with_fixed_gae", adv_vec_corr, global_step)
        writer.add_scalar("tddelta/reward_residual_std", reward_residual_prev.std().item(), global_step)
        writer.add_scalar(
            "tddelta/embedded_credit_fraction",
            (adv_vector.std() / advantages.std().clamp_min(1e-8)).item(),
            global_step,
        )
        writer.add_scalar("advB/adv_vector_std", adv_vector.std().item(), global_step)
        writer.add_scalar("advB/adv_scalar_std", adv_scalar_gae.std().item(), global_step)
        effective_horizon = 1.0 / (
            1.0 - args.gamma * state_lambda
        ).clamp_min(1e-6)
        realized_trace_mass = learned_vector_gae(
            torch.ones_like(rewards).unsqueeze(-1),
            continuation,
            args.gamma,
            state_lambda,
        ).squeeze(-1)
        writer.add_scalar("trace/lambda_mean", state_lambda.mean().item(), global_step)
        writer.add_scalar("trace/lambda_p10", state_lambda.quantile(0.1).item(), global_step)
        writer.add_scalar("trace/lambda_p90", state_lambda.quantile(0.9).item(), global_step)
        writer.add_scalar(
            "trace/local_horizon_proxy_mean", effective_horizon.mean().item(), global_step
        )
        writer.add_scalar(
            "trace/realized_horizon_mean", realized_trace_mass.mean().item(), global_step
        )
        writer.add_scalar("trace/bootstrap_bias_mean", trace_bias.mean().item(), global_step)
        writer.add_scalar("trace/continuation_noise_mean", trace_noise.mean().item(), global_step)
        writer.add_scalar("trace/direct_next_rollout_mse", trace_direct_mse, global_step)
        writer.add_scalar(
            "trace/direct_next_rollout_calibration",
            trace_direct_calibration,
            global_step,
        )
        writer.add_scalar(
            "trace/recursive_next_rollout_mse", trace_recursive_mse, global_step
        )
        writer.add_scalar(
            "trace/recursive_next_rollout_calibration",
            trace_recursive_calibration,
            global_step,
        )
        writer.add_scalar("trace/risk_train_loss", trace_risk_loss, global_step)
        writer.add_scalar("trace/risk_grad_norm", trace_risk_gn, global_step)
        writer.add_scalar("trace/risk_steps", trace_risk_steps, global_step)
        writer.add_scalar(
            "trace/risk_validation_nll", trace_risk_validation_nll, global_step
        )
        writer.add_scalar(
            "trace/risk_validation_calibration",
            trace_risk_validation_calibration,
            global_step,
        )
        writer.add_scalar(
            "trace/risk_direct_skill", trace_risk_direct_skill, global_step
        )
        writer.add_scalar(
            "trace/risk_direct_vs_bootstrap_skill",
            trace_risk_direct_vs_bootstrap_skill,
            global_step,
        )
        writer.add_scalar(
            "trace/risk_recursive_skill", trace_risk_recursive_skill, global_step
        )
        writer.add_scalar("trace/risk_ready", float(trace_was_ready), global_step)
        writer.add_scalar(
            "vector_policy/projection_max_error", vector_projection_error, global_step
        )
        writer.add_scalar(
            "vector_policy/whiten_min_eigenvalue",
            whiten_eigenvalues.min().item(),
            global_step,
        )
        writer.add_scalar(
            "vector_policy/whiten_condition",
            (
                whiten_eigenvalues.max()
                / whiten_eigenvalues.min().clamp_min(1e-12)
            ).item(),
            global_step,
        )
        writer.add_scalar(
            "vector_policy/covector_norm", policy_covector.norm().item(), global_step
        )
        writer.add_scalar(
            "vector_policy/router_ready", float(router_was_ready), global_step
        )
        writer.add_scalar(
            "vector_policy/router_fit_r2", router_fit_r2.item(), global_step
        )
        writer.add_scalar(
            "vector_policy/router_scale", router_scale.item(), global_step
        )
        writer.add_scalar(
            "vector_policy/router_effective_rank",
            router_effective_rank.item(),
            global_step,
        )
        writer.add_scalar(
            "vector_policy/router_mode_gain", router_mode_gain.item(), global_step
        )
        writer.add_scalar(
            "vector_policy/routed_fraction",
            (
                routed_component.std()
                / policy_adv.std().clamp_min(1e-8)
            ).item(),
            global_step,
        )

        # ---- the falsifier: EV against truncated-MC returns, three predictors ----------
        writer.add_scalar("gate/ev_sf_online", ev_sf, global_step)          # (2) treatment
        writer.add_scalar("gate/ev_trunk_probe", ev_trunk_probe, global_step)  # (1) reference
        writer.add_scalar("gate/ev_latent_cap", ev_latent_cap, global_step)    # (3) ceiling
        writer.add_scalar("gate/mc_frac", n_mc / (args.num_steps * args.num_envs), global_step)

        # ---- SSL path -------------------------------------------------------------------
        writer.add_scalar("ssl/pred_loss", ssl_pred_l, global_step)
        for horizon, loss_value in zip(LEJEPA_HORIZONS, ssl_horizon_l):
            writer.add_scalar(
                f"ssl/pred_loss_horizon_{horizon}", loss_value.item(), global_step
            )
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig_l, global_step)
        writer.add_scalar("ssl/steps_per_iter", ssl_steps, global_step)
        writer.add_scalar("ssl/grad_norm", ssl_gn_sum, global_step)
        writer.add_scalar("ssl/emb_mean_abs", emb_mean_abs, global_step)
        writer.add_scalar("ssl/emb_std", emb_std, global_step)
        writer.add_scalar("ssl/emb_effective_rank", eff_rank, global_step)
        writer.add_scalar("ssl/frame_drift_raw", drift_raw, global_step)
        writer.add_scalar("ssl/frame_drift_rot", drift_rot, global_step)
        writer.add_scalar("ssl/control_encoder_policy_kl", encoder_policy_kl.item(), global_step)
        writer.add_scalar("ssl/control_encoder_emb_drift", encoder_emb_drift.item(), global_step)
        writer.add_scalar("ssl/control_encoder_sf_drift", encoder_sf_drift.item(), global_step)
        writer.add_scalar("ssl/control_encoder_value_drift", encoder_value_drift.item(), global_step)
        writer.add_scalar("ssl/control_encoder_update_accepted", float(accepted), global_step)
        writer.add_scalar("ssl/control_encoder_update_rate", accepted_rate, global_step)
        writer.add_scalar("ssl/control_encoder_drift_raw", control_drift_raw, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

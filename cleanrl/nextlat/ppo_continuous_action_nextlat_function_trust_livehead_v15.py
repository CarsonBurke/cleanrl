# NEXTLAT FUNCTION-TRUST LIVE-HEAD v15: behavior-safe trunk + policy-decoder coding.
# =====================================================================================
# Historical A/B evidence isolates the intervention: increasing the NextLat action-head
# auxiliary clip from 0.0025 to 0.015 improved return by +476 at 1.3M steps, while raising
# the shared-trunk allowance regressed. This version therefore keeps v14's predictive-trunk
# transaction byte-for-byte in spirit and admits only a clipped live-decoder gradient into
# the same task Adam moments as PPO. The future-policy teacher remains stopped; the student
# uses the live Beta or Gaussian action heads. Nonfinite head auxiliary gradients fail closed
# for the whole head group, leaving its task update exactly unchanged.
# Because the mixed head gradient enters task Adam before v14's post-step probe, the actor
# trust ruler intentionally measures the coherent PPO+predictive head movement; that movement
# can finance some later predictive-trunk KL. We log trunk-only and mixed-head KL to make this
# coupling explicit. Thus this is not a clean head-only A/B; a win must be followed by the same
# code with --nextlat-head-grad-clip 0. The critic ruler remains task-only, and exact nonlinear
# verification still vetoes any unsafe incremental trunk proposal. Gaussian support is complete
# but unvalidated experimentally; Beta remains the evidence-backed default.
# =====================================================================================
# Task, predictive-trunk, and predictor gradients retain independent Adam moments. Predictive
# trunk proposals are first projected into actor/critic descent half-spaces, then admitted by
# exact function-space trust regions on a deterministic 256-state probe: policy KL and
# masked categorical-critic KL may each move by at most 5% of the task step in local distance.
# A measured full proposal gets one square-root rescale and one exact nonlinear verification;
# violation or nonfinite behavior rolls it back, while predictive Adam state still advances.
# Hypothesis: behavioral trust spends representation updates where Euclidean parameter caps
# are needlessly conservative, without permitting destructive policy/value function drift.
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
# gets ITS OWN dual-backward clip budgets (trunk/predictor 0.25 vs actor 0.25 / critic
# 0.25), so the A/B isolates the TARGET STRUCTURE
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
from typing import Callable, Optional, Sequence

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

    # NEXTLAT auxiliary (paper-aligned, arXiv:2511.05963): action-conditioned MLP latent
    # dynamics rolled out to depth d; Smooth L1 on all intermediate predicted latents vs
    # ONLINE stop-grad trunk targets, plus KL distillation of the true future action
    # distribution into the one decoded from the predicted latent (see header).
    nextlat: bool = True
    nextlat_depth: int = 4           # rollout depth d (paper uses 1..8 across tasks)
    nextlat_coef: float = 1.0        # lambda_next-h (the dedicated grad clip is the real ceiling)
    nextlat_kl_coef: float = 1.0     # lambda_KL relative to the prediction term
    nextlat_trunk_grad_clip: float = 0.25  # clip025 incumbent; update trust is the real trunk limiter
    nextlat_predictor_grad_clip: float = 0.25  # private predictor budget, independent of trunk size
    nextlat_head_grad_clip: float = 0.015  # live decoder aux budget; one global action-head group

    # v14 optimizer-level admission. The ratio is a local-distance ratio, hence its
    # square multiplies the task-induced KL. The absolute ceiling prevents a single
    # unusually large PPO step from lending an unsafe auxiliary trust region.
    function_trust_ratio: float = 0.05
    function_trust_max_kl: float = 1e-4
    function_trust_probe_size: int = 256
    function_trust_verify_rtol: float = 1e-3
    function_trust_atol: float = 1e-8

    # Compile only the hot forward paths. Optimizer surgery intentionally remains eager.
    compile: bool = False
    compile_mode: str = "reduce-overhead"

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
        alpha, beta = self._beta_concentrations(actor_feat)
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def _beta_concentrations(self, actor_feat):
        if self.actor_dist != "beta":
            raise RuntimeError("function-space policy trust requires the Beta actor")
        return (
            1.0 + F.softplus(self.actor_alpha_head(actor_feat)),
            1.0 + F.softplus(self.actor_beta_head(actor_feat)),
        )

    def _actor_dist_frozen_head(self, actor_feat):
        """Decode a latent through constant policy-head weights.

        Gradients still flow to ``actor_feat`` through the frozen linear maps. This lets
        policy-space KL give the dynamics/trunk a behavioral metric without letting an
        auxiliary loss change the behavior decoder itself.
        """
        if self.actor_dist == "gaussian":
            mean = F.linear(
                actor_feat,
                self.actor_head.weight.detach(),
                None if self.actor_head.bias is None else self.actor_head.bias.detach(),
            )
            raw_lv = F.linear(
                actor_feat,
                self.actor_logvar_head.weight.detach(),
                None
                if self.actor_logvar_head.bias is None
                else self.actor_logvar_head.bias.detach(),
            )
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            dist = Normal(mean, (0.5 * lv).exp())
            return (
                dist,
                torch.tanh,
                lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z)),
            )
        alpha_raw = F.linear(
            actor_feat,
            self.actor_alpha_head.weight.detach(),
            None
            if self.actor_alpha_head.bias is None
            else self.actor_alpha_head.bias.detach(),
        )
        beta_raw = F.linear(
            actor_feat,
            self.actor_beta_head.weight.detach(),
            None
            if self.actor_beta_head.bias is None
            else self.actor_beta_head.bias.detach(),
        )
        dist = Beta(1.0 + F.softplus(alpha_raw), 1.0 + F.softplus(beta_raw))
        return (
            dist,
            lambda z: self.action_low + (self.action_high - self.action_low) * z,
            lambda z: 0.0,
        )

    def _actor_dist_live_head(self, actor_feat):
        """Decode imagined latents through the live action head.

        This is deliberately a named wrapper rather than a second implementation: its
        forward and latent Jacobian are exactly those of ``_actor_dist_frozen_head`` at
        the same parameters, while autograd additionally records action-head gradients.
        """
        return self._actor_dist(actor_feat)

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

    def get_value_h0(self, x):
        """Return only horizon-0 logits without materializing every MTP horizon."""
        _, critic_feat = self._trunks(x)
        bias = self.critic_head.bias
        return F.linear(
            critic_feat,
            self.critic_head.weight[: self.num_bins],
            None if bias is None else bias[: self.num_bins],
        )

    def get_actor_feat(self, x):
        # Actor-side trunk feature only; used for the ONLINE stop-grad nextlat targets.
        return self._trunks(x)[0]

    def get_behavior_probe(self, x):
        """Return deterministic policy parameters and categorical-critic logits."""
        actor_feat, critic_feat = self._trunks(x)
        dist, _, _ = self._actor_dist(actor_feat)
        if self.actor_dist == "beta":
            first, second = dist.concentration1, dist.concentration0
        else:
            first, second = dist.loc, dist.scale
        critic_logits = self.critic_head(critic_feat).view(
            -1, self.critic_mtp_horizon, self.num_bins
        )
        return first, second, critic_logits

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

    def action_head_parameters(self):
        """Policy decoder parameters, excluding every representation parameter."""
        if self.actor_dist == "gaussian":
            return list(self.actor_head.parameters()) + list(
                self.actor_logvar_head.parameters()
            )
        return list(self.actor_alpha_head.parameters()) + list(
            self.actor_beta_head.parameters()
        )

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())

    def nextlat_parameters(self):
        # Only these parameters receive a private optimizer. The live action-head
        # auxiliary is merged into task Adam separately and is intentionally excluded.
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        return list(trunk.parameters()) + list(self.nextlat_predictor.parameters())

    def nextlat_trunk_parameters(self):
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        return list(trunk.parameters())

    def nextlat_predictor_parameters(self):
        return list(self.nextlat_predictor.parameters())

    def task_parameters(self):
        predictor_ids = {id(p) for p in self.nextlat_predictor.parameters()}
        return [p for p in self.parameters() if id(p) not in predictor_ids]


@torch.no_grad()
def make_update_tensor_layout(tensors):
    """Precompute device-side segment metadata for tensor-local update protection."""
    if not tensors:
        raise ValueError("at least one tensor is required")
    device = tensors[0].device
    lengths_tuple = tuple(tensor.numel() for tensor in tensors)
    lengths = torch.tensor(lengths_tuple, device=device, dtype=torch.int64)
    groups = torch.repeat_interleave(
        torch.arange(len(tensors), device=device),
        lengths,
        output_size=sum(lengths_tuple),
    )
    return lengths_tuple, lengths, groups


def _segment_sum(values, lengths):
    return torch.segment_reduce(values, "sum", lengths=lengths)


def _projection_dot_tolerance(gradient_sq, update_sq):
    """Scale-aware roundoff tolerance for one first-order half-space constraint."""
    eps = torch.finfo(gradient_sq.dtype).eps
    return 64.0 * eps * (gradient_sq * update_sq).clamp_min(0.0).sqrt()


@dataclass(frozen=True)
class BehaviorProbe:
    """Deterministic policy and critic outputs on a fixed observation probe."""

    # For Beta these are alpha/beta; for Gaussian they are mean/std. The historical
    # names preserve the exact v14 Beta API and make its tests directly reusable.
    actor_alpha: torch.Tensor
    actor_beta: torch.Tensor
    critic_logits: torch.Tensor
    actor_dist: str = "beta"


@torch.no_grad()
def clone_behavior_probe(probe, *, actor_dist="beta"):
    """Detach and clone compiled outputs before another compiled invocation can reuse them."""
    if isinstance(probe, BehaviorProbe):
        tensors = (probe.actor_alpha, probe.actor_beta, probe.critic_logits)
        actor_dist = probe.actor_dist
    else:
        if len(probe) != 3:
            raise ValueError("a behavior probe must contain two policy tensors and critic logits")
        tensors = probe
    if actor_dist not in ("beta", "gaussian"):
        raise ValueError(f"unknown probe policy distribution {actor_dist}")
    return BehaviorProbe(
        *(tensor.detach().clone() for tensor in tensors), actor_dist=actor_dist
    )


def beta_policy_kl(reference: BehaviorProbe, candidate: BehaviorProbe):
    """Mean statewise Beta KL, summed (not averaged) over action dimensions."""
    if reference.actor_dist != "beta" or candidate.actor_dist != "beta":
        raise ValueError("beta_policy_kl requires Beta behavior probes")
    reference_dist = Beta(
        reference.actor_alpha,
        reference.actor_beta,
        validate_args=False,
    )
    candidate_dist = Beta(
        candidate.actor_alpha,
        candidate.actor_beta,
        validate_args=False,
    )
    return torch.distributions.kl_divergence(reference_dist, candidate_dist).sum(-1).mean()


def gaussian_policy_kl(reference: BehaviorProbe, candidate: BehaviorProbe):
    """Mean statewise Normal KL, summed (not averaged) over action dimensions."""
    if reference.actor_dist != "gaussian" or candidate.actor_dist != "gaussian":
        raise ValueError("gaussian_policy_kl requires Gaussian behavior probes")
    reference_dist = Normal(
        reference.actor_alpha,
        reference.actor_beta,
        validate_args=False,
    )
    candidate_dist = Normal(
        candidate.actor_alpha,
        candidate.actor_beta,
        validate_args=False,
    )
    return torch.distributions.kl_divergence(reference_dist, candidate_dist).sum(-1).mean()


def policy_kl(reference: BehaviorProbe, candidate: BehaviorProbe):
    if reference.actor_dist != candidate.actor_dist:
        raise ValueError("reference and candidate policy distributions must match")
    if reference.actor_dist == "beta":
        return beta_policy_kl(reference, candidate)
    if reference.actor_dist == "gaussian":
        return gaussian_policy_kl(reference, candidate)
    raise ValueError(f"unknown probe policy distribution {reference.actor_dist}")


def masked_categorical_critic_kl(
    reference: BehaviorProbe,
    candidate: BehaviorProbe,
    valid_mask: torch.Tensor,
):
    """Categorical KL averaged over valid state/horizon entries only."""
    if reference.critic_logits.shape != candidate.critic_logits.shape:
        raise ValueError("reference and candidate critic logits must have equal shape")
    if valid_mask.shape != reference.critic_logits.shape[:-1]:
        raise ValueError("critic mask must match the state/horizon dimensions")
    reference_log_probs = F.log_softmax(reference.critic_logits, dim=-1)
    candidate_log_probs = F.log_softmax(candidate.critic_logits, dim=-1)
    per_entry = (
        reference_log_probs.exp() * (reference_log_probs - candidate_log_probs)
    ).sum(-1)
    mask = valid_mask.to(device=per_entry.device, dtype=per_entry.dtype)
    return (per_entry * mask).sum() / mask.sum().clamp_min(1.0)


def behavioral_kls(
    reference: BehaviorProbe,
    candidate: BehaviorProbe,
    critic_mask: torch.Tensor,
):
    return (
        policy_kl(reference, candidate),
        masked_categorical_critic_kl(reference, candidate, critic_mask),
    )


@torch.no_grad()
def project_predictive_updates(
    predictive_updates,
    *,
    actor_gradients,
    critic_gradients,
    layout=None,
):
    """Project each predictive tensor into actor and critic descent half-spaces.

    This is the exact active-set projection used by v9, but with no parameter-space
    magnitude cap. Function-space KL measurement below supplies the trust region.
    """
    if not predictive_updates:
        raise ValueError("at least one update tensor is required")
    if len(actor_gradients) != len(predictive_updates) or len(critic_gradients) != len(
        predictive_updates
    ):
        raise ValueError("actor/critic gradients must align one-to-one with trunk updates")
    for predictive, actor, critic in zip(
        predictive_updates, actor_gradients, critic_gradients
    ):
        if (
            predictive.shape != actor.shape
            or predictive.shape != critic.shape
            or predictive.device != actor.device
            or predictive.device != critic.device
        ):
            raise ValueError("updates and gradients must share shape and device")

    if layout is None:
        layout = make_update_tensor_layout(predictive_updates)
    lengths_tuple, lengths, groups = layout
    if lengths_tuple != tuple(update.numel() for update in predictive_updates):
        raise ValueError("update tensors do not match the supplied layout")

    flat_predictive = torch.cat([update.reshape(-1) for update in predictive_updates])
    flat_actor = torch.cat([gradient.reshape(-1) for gradient in actor_gradients])
    flat_critic = torch.cat([gradient.reshape(-1) for gradient in critic_gradients])

    # Tensor-local sufficient statistics. One segmented reduction replaces hundreds of
    # scalar CUDA kernels while retaining an independent active-set solve per parameter.
    pred_sq_local = _segment_sum(flat_predictive.square(), lengths)
    actor_sq = _segment_sum(flat_actor.square(), lengths)
    critic_sq = _segment_sum(flat_critic.square(), lengths)
    actor_critic = _segment_sum(flat_actor * flat_critic, lengths)
    actor_pred = _segment_sum(flat_actor * flat_predictive, lengths)
    critic_pred = _segment_sum(flat_critic * flat_predictive, lengths)

    tiny = torch.finfo(flat_predictive.dtype).tiny
    eps = torch.finfo(flat_predictive.dtype).eps
    best_distance = pred_sq_local
    best_actor_multiplier = torch.zeros_like(best_distance)
    best_critic_multiplier = torch.zeros_like(best_distance)
    best_active = torch.zeros_like(best_distance, dtype=torch.bool)

    def select_candidate(distance, feasible, actor_multiplier, critic_multiplier, active=True):
        nonlocal best_distance, best_actor_multiplier, best_critic_multiplier, best_active
        choose = feasible & (distance < best_distance)
        best_distance = torch.where(choose, distance, best_distance)
        best_actor_multiplier = torch.where(choose, actor_multiplier, best_actor_multiplier)
        best_critic_multiplier = torch.where(choose, critic_multiplier, best_critic_multiplier)
        if active:
            best_active = best_active | choose

    zeros = torch.zeros_like(best_distance)
    original_actor_tol = _projection_dot_tolerance(actor_sq, pred_sq_local)
    original_critic_tol = _projection_dot_tolerance(critic_sq, pred_sq_local)
    original_feasible = (actor_pred <= original_actor_tol) & (
        critic_pred <= original_critic_tol
    )
    select_candidate(zeros, original_feasible, zeros, zeros)

    actor_multiplier = actor_pred.clamp_min(0.0) / actor_sq.clamp_min(tiny)
    actor_candidate_actor_dot = actor_pred - actor_multiplier * actor_sq
    actor_candidate_critic_dot = critic_pred - actor_multiplier * actor_critic
    actor_candidate_sq = (
        pred_sq_local
        - 2.0 * actor_multiplier * actor_pred
        + actor_multiplier.square() * actor_sq
    ).clamp_min(0.0)
    actor_feasible = (
        actor_candidate_actor_dot
        <= _projection_dot_tolerance(actor_sq, actor_candidate_sq)
    ) & (
        actor_candidate_critic_dot
        <= _projection_dot_tolerance(critic_sq, actor_candidate_sq)
    )
    select_candidate(
        actor_multiplier.square() * actor_sq,
        actor_feasible,
        actor_multiplier,
        zeros,
    )

    critic_multiplier = critic_pred.clamp_min(0.0) / critic_sq.clamp_min(tiny)
    critic_candidate_actor_dot = actor_pred - critic_multiplier * actor_critic
    critic_candidate_critic_dot = critic_pred - critic_multiplier * critic_sq
    critic_candidate_sq = (
        pred_sq_local
        - 2.0 * critic_multiplier * critic_pred
        + critic_multiplier.square() * critic_sq
    ).clamp_min(0.0)
    critic_feasible = (
        critic_candidate_actor_dot
        <= _projection_dot_tolerance(actor_sq, critic_candidate_sq)
    ) & (
        critic_candidate_critic_dot
        <= _projection_dot_tolerance(critic_sq, critic_candidate_sq)
    )
    select_candidate(
        critic_multiplier.square() * critic_sq,
        critic_feasible,
        zeros,
        critic_multiplier,
    )

    determinant = actor_sq * critic_sq - actor_critic.square()
    determinant_valid = determinant > eps * actor_sq * critic_sq
    safe_determinant = torch.where(determinant_valid, determinant, torch.ones_like(determinant))
    joint_actor_multiplier = (
        actor_pred * critic_sq - critic_pred * actor_critic
    ) / safe_determinant
    joint_critic_multiplier = (
        critic_pred * actor_sq - actor_pred * actor_critic
    ) / safe_determinant
    joint_actor_dot = (
        actor_pred
        - joint_actor_multiplier * actor_sq
        - joint_critic_multiplier * actor_critic
    )
    joint_critic_dot = (
        critic_pred
        - joint_actor_multiplier * actor_critic
        - joint_critic_multiplier * critic_sq
    )
    joint_candidate_sq = (
        pred_sq_local
        - 2.0 * joint_actor_multiplier * actor_pred
        - 2.0 * joint_critic_multiplier * critic_pred
        + joint_actor_multiplier.square() * actor_sq
        + joint_critic_multiplier.square() * critic_sq
        + 2.0
        * joint_actor_multiplier
        * joint_critic_multiplier
        * actor_critic
    ).clamp_min(0.0)
    joint_feasible = (
        determinant_valid
        & (joint_actor_multiplier >= 0.0)
        & (joint_critic_multiplier >= 0.0)
        & (
            joint_actor_dot
            <= _projection_dot_tolerance(actor_sq, joint_candidate_sq)
        )
        & (
            joint_critic_dot
            <= _projection_dot_tolerance(critic_sq, joint_candidate_sq)
        )
    )
    joint_distance = (
        joint_actor_multiplier.square() * actor_sq
        + joint_critic_multiplier.square() * critic_sq
        + 2.0 * joint_actor_multiplier * joint_critic_multiplier * actor_critic
    ).clamp_min(0.0)
    select_candidate(
        joint_distance,
        joint_feasible,
        joint_actor_multiplier,
        joint_critic_multiplier,
    )

    flat_projected = (
        flat_predictive
        - best_actor_multiplier[groups] * flat_actor
        - best_critic_multiplier[groups] * flat_critic
    ) * best_active[groups]
    post_actor_dot = _segment_sum(flat_actor * flat_projected, lengths)
    post_critic_dot = _segment_sum(flat_critic * flat_projected, lengths)
    projected_sq_local = _segment_sum(flat_projected.square(), lengths)
    # Active projections land exactly on a boundary in real arithmetic. A strict zero
    # comparison spuriously vetoes them when the final dot rounds a few ulps positive.
    numerically_feasible = (
        post_actor_dot <= _projection_dot_tolerance(actor_sq, projected_sq_local)
    ) & (
        post_critic_dot <= _projection_dot_tolerance(critic_sq, projected_sq_local)
    )
    flat_projected = flat_projected * numerically_feasible[groups]
    projected = [
        flat.view_as(reference)
        for flat, reference in zip(
            flat_projected.split(lengths_tuple), predictive_updates
        )
    ]

    raw_norm = flat_predictive.norm()
    projected_norm = flat_projected.norm()
    return projected, {
        "predictive_norm": raw_norm,
        "projected_norm": projected_norm,
        "projection_fraction": projected_norm / raw_norm.clamp_min(1e-20),
        "actor_first_order": (flat_actor * flat_projected).sum(),
        "critic_first_order": (flat_critic * flat_projected).sum(),
        "actor_conflict_fraction": (actor_pred > 0.0)
        .to(flat_predictive.dtype)
        .mean(),
        "critic_conflict_fraction": (critic_pred > 0.0)
        .to(flat_predictive.dtype)
        .mean(),
        "projected_tensor_fraction": _segment_sum(flat_projected.square(), lengths)
        .gt(0.0)
        .to(flat_predictive.dtype)
        .mean(),
    }


def function_trust_budget(task_kl, trust_ratio, max_kl):
    """Convert task-induced KL to a squared local-distance auxiliary budget."""
    if trust_ratio < 0.0 or max_kl < 0.0:
        raise ValueError("trust ratio and maximum KL must be non-negative")
    task_kl = torch.as_tensor(task_kl)
    ceiling = task_kl.new_tensor(max_kl)
    finite_budget = torch.minimum(
        ceiling,
        task_kl.clamp_min(0.0) * (trust_ratio * trust_ratio),
    )
    return torch.where(torch.isfinite(task_kl), finite_budget, torch.zeros_like(task_kl))


def function_trust_scale(budget, candidate_kl, *, atol=1e-8):
    """One-shot square-root KL scale with a 0.9 nonlinear safety margin."""
    budget = torch.as_tensor(budget)
    candidate_kl = torch.as_tensor(
        candidate_kl, device=budget.device, dtype=budget.dtype
    )
    candidate_nonnegative = candidate_kl.clamp_min(0.0)
    ratio_scale = 0.9 * torch.sqrt(
        budget / candidate_nonnegative.clamp_min(torch.finfo(budget.dtype).tiny)
    )
    scale = ratio_scale.clamp(max=1.0)
    # A truly function-null direction is admissible even when the task step and budget
    # are exactly zero. ``atol`` absorbs only floating-point KL cancellation, not a floor.
    scale = torch.where(candidate_nonnegative <= atol, torch.ones_like(scale), scale)
    return torch.where(
        torch.isfinite(candidate_kl) & torch.isfinite(scale),
        scale,
        torch.zeros_like(scale),
    )


def _probe_is_finite(probe):
    return torch.stack(
        [
            torch.isfinite(tensor).all()
            for tensor in (probe.actor_alpha, probe.actor_beta, probe.critic_logits)
        ]
    ).all()


@torch.no_grad()
def apply_function_trust_transaction(
    parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    gradients: Sequence[torch.Tensor],
    task_updates: Sequence[torch.Tensor],
    *,
    actor_gradients: Sequence[torch.Tensor],
    critic_gradients: Sequence[torch.Tensor],
    pre_task_probe: BehaviorProbe,
    post_task_probe: BehaviorProbe,
    critic_mask: torch.Tensor,
    evaluate_probe: Callable[[], BehaviorProbe],
    trust_ratio: float,
    max_kl: float,
    verify_rtol: float = 1e-3,
    atol: float = 1e-8,
    layout=None,
):
    """Advance predictive Adam, then admit its projected delta by exact behavioral KL.

    The task optimizer has already stepped and is never touched here. Predictive Adam's
    moments advance even when the parameter proposal is rejected. The full proposal is
    measured once, square-root scaled once, and exactly remeasured once; there is no line
    search, parameter-norm fallback, or positive budget floor.
    """
    count = len(parameters)
    if not (
        len(gradients)
        == len(task_updates)
        == len(actor_gradients)
        == len(critic_gradients)
        == count
    ):
        raise ValueError("all trunk parameter, update, and gradient lists must align")
    if count == 0:
        raise ValueError("at least one trunk parameter is required")
    if verify_rtol < 0.0 or atol < 0.0:
        raise ValueError("verification tolerances must be non-negative")
    if layout is None:
        layout = make_update_tensor_layout(parameters)
    lengths_tuple, _, _ = layout
    if lengths_tuple != tuple(parameter.numel() for parameter in parameters):
        raise ValueError("trunk parameters do not match the supplied layout")

    task_actor_kl, task_critic_kl = behavioral_kls(
        pre_task_probe, post_task_probe, critic_mask
    )
    actor_budget = function_trust_budget(task_actor_kl, trust_ratio, max_kl)
    critic_budget = function_trust_budget(task_critic_kl, trust_ratio, max_kl)

    # Advance the predictive optimizer at the post-task parameters, save its proposed
    # delta, and roll parameters (but deliberately not optimizer state) back. Invalid
    # gradient entries become a zero-gradient Adam step: its clock still advances, its
    # moments stay finite, and the transaction is marked invalid and rejected below.
    optimizer.zero_grad(set_to_none=True)
    post_task_parameters = [parameter.detach().clone() for parameter in parameters]
    flat_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients])
    gradient_finite = torch.isfinite(flat_gradient).all()
    flat_gradient = torch.nan_to_num(
        flat_gradient, nan=0.0, posinf=0.0, neginf=0.0
    )
    for parameter, gradient in zip(
        parameters, flat_gradient.split(lengths_tuple)
    ):
        parameter.grad = gradient.view_as(parameter)
    optimizer.step()
    flat_previous = torch.cat(
        [previous.reshape(-1) for previous in post_task_parameters]
    )
    flat_raw = torch.cat(
        [parameter.detach().reshape(-1) for parameter in parameters]
    ) - flat_previous
    raw_finite = torch.isfinite(flat_raw).all()
    numeric_valid = gradient_finite & raw_finite
    flat_raw = torch.nan_to_num(flat_raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw_updates = [
        update.view_as(reference)
        for update, reference in zip(flat_raw.split(lengths_tuple), parameters)
    ]
    for parameter, previous in zip(parameters, post_task_parameters):
        parameter.copy_(previous)

    projected_updates, projection_stats = project_predictive_updates(
        raw_updates,
        actor_gradients=actor_gradients,
        critic_gradients=critic_gradients,
        layout=layout,
    )

    # Measure the complete projected proposal relative to the post-task behavior.
    for parameter, update in zip(parameters, projected_updates):
        parameter.add_(update)
    full_probe = None
    try:
        full_probe = clone_behavior_probe(evaluate_probe())
        full_actor_kl, full_critic_kl = behavioral_kls(
            post_task_probe, full_probe, critic_mask
        )
        full_finite = (
            _probe_is_finite(full_probe)
            & torch.isfinite(full_actor_kl)
            & torch.isfinite(full_critic_kl)
        )
    except (RuntimeError, ValueError, FloatingPointError):
        full_actor_kl = actor_budget.new_tensor(float("inf"))
        full_critic_kl = critic_budget.new_tensor(float("inf"))
        full_finite = torch.zeros_like(actor_budget, dtype=torch.bool)
    finally:
        for parameter, previous in zip(parameters, post_task_parameters):
            parameter.copy_(previous)

    actor_scale = function_trust_scale(actor_budget, full_actor_kl, atol=atol)
    critic_scale = function_trust_scale(critic_budget, full_critic_kl, atol=atol)
    scale = torch.minimum(actor_scale, critic_scale)
    scale = torch.where(full_finite, scale, torch.zeros_like(scale))

    # One exact nonlinear verification at the scaled point. A failed check is a hard
    # rollback; the task parameters and predictive Adam moments remain intact.
    for parameter, update in zip(parameters, projected_updates):
        parameter.add_(update * scale)
    verified_probe = None
    try:
        verified_probe = clone_behavior_probe(evaluate_probe())
        verified_actor_kl, verified_critic_kl = behavioral_kls(
            post_task_probe, verified_probe, critic_mask
        )
        verified_finite = (
            _probe_is_finite(verified_probe)
            & torch.isfinite(verified_actor_kl)
            & torch.isfinite(verified_critic_kl)
            & torch.isfinite(task_actor_kl)
            & torch.isfinite(task_critic_kl)
        )
        actor_ok = verified_actor_kl <= actor_budget * (1.0 + verify_rtol) + atol
        critic_ok = verified_critic_kl <= critic_budget * (1.0 + verify_rtol) + atol
        accepted = (
            numeric_valid
            & full_finite
            & verified_finite
            & actor_ok
            & critic_ok
        )
    except (RuntimeError, ValueError, FloatingPointError):
        verified_actor_kl = actor_budget.new_tensor(float("inf"))
        verified_critic_kl = critic_budget.new_tensor(float("inf"))
        accepted = torch.zeros_like(actor_budget, dtype=torch.bool)

    accepted_scale = scale * accepted.to(dtype=scale.dtype)
    # Materialize the exact accepted state without a device-to-host branch.
    for parameter, previous, update in zip(
        parameters, post_task_parameters, projected_updates
    ):
        admitted = torch.where(
            accepted,
            update * accepted_scale,
            torch.zeros_like(update),
        )
        parameter.copy_(previous).add_(admitted)
    admitted_updates = [
        torch.where(
            accepted,
            update * accepted_scale,
            torch.zeros_like(update),
        )
        for update in projected_updates
    ]

    optimizer.zero_grad(set_to_none=True)
    flat_task = torch.cat([update.reshape(-1) for update in task_updates])
    flat_projected = torch.cat([update.reshape(-1) for update in projected_updates])
    task_norm = flat_task.norm()
    raw_norm = flat_raw.norm()
    projected_norm = flat_projected.norm()
    admitted_norm = projected_norm * accepted_scale
    raw_task_cosine = (flat_task * flat_raw).sum() / (
        task_norm * raw_norm
    ).clamp_min(1e-20)
    stats = {
        **projection_stats,
        "task_norm": task_norm,
        "raw_task_cosine": raw_task_cosine,
        "admitted_norm": admitted_norm,
        "accepted_fraction": admitted_norm / raw_norm.clamp_min(1e-20),
        "scale": accepted_scale,
        "proposal_scale": scale,
        "accepted": accepted.to(dtype=raw_norm.dtype),
        "numeric_valid": numeric_valid.to(dtype=raw_norm.dtype),
        "task_actor_kl": task_actor_kl,
        "task_critic_kl": task_critic_kl,
        "actor_budget": actor_budget,
        "critic_budget": critic_budget,
        "full_actor_kl": full_actor_kl,
        "full_critic_kl": full_critic_kl,
        "verified_actor_kl": verified_actor_kl,
        "verified_critic_kl": verified_critic_kl,
    }
    return raw_updates, projected_updates, admitted_updates, stats


@torch.no_grad()
def apply_private_optimizer_step(parameters, optimizer, gradients):
    """Apply a finite private optimizer step and report its actual parameter norm.

    A nonfinite auxiliary gradient must not permanently poison predictor parameters or
    Adam moments. Invalid entries therefore become zero-gradient entries while the Adam
    clock advances consistently with the overlapping predictive-trunk optimizer.
    """
    if len(parameters) != len(gradients):
        raise ValueError("one gradient is required per private parameter")
    if not parameters:
        raise ValueError("at least one private parameter is required")
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in parameters]
    lengths = tuple(parameter.numel() for parameter in parameters)
    for parameter, gradient in zip(parameters, gradients):
        if parameter.shape != gradient.shape or parameter.device != gradient.device:
            raise ValueError("private gradients must match their parameters")
    flat_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients])
    flat_gradient = torch.nan_to_num(
        flat_gradient, nan=0.0, posinf=0.0, neginf=0.0
    )
    for parameter, gradient in zip(parameters, flat_gradient.split(lengths)):
        parameter.grad = gradient.view_as(parameter)
    optimizer.step()
    update_sq = parameters[0].new_zeros(())
    for parameter, previous in zip(parameters, before):
        update_sq = update_sq + (parameter.detach() - previous).square().sum()
    optimizer.zero_grad(set_to_none=True)
    return update_sq.sqrt()


@torch.no_grad()
def prepare_live_head_auxiliary_gradients(parameters, max_norm):
    """Clone and globally clip the live-decoder auxiliary as one fail-closed group.

    The function never mutates parameter gradients. If one tensor or the aggregate norm
    is nonfinite, every delivered tensor is zero and ``valid`` is false. This all-or-none
    rule prevents a partially valid decoder update from changing coupled Adam moments.
    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError("at least one action-head parameter is required")
    if max_norm < 0.0:
        raise ValueError("the live-head gradient clip must be non-negative")
    gradients = [
        torch.zeros_like(parameter)
        if parameter.grad is None
        else parameter.grad.detach().clone()
        for parameter in parameters
    ]
    raw_sq = torch.stack(
        [gradient.float().square().sum() for gradient in gradients]
    ).sum()
    raw_norm_unchecked = raw_sq.sqrt()
    finite = torch.stack(
        [torch.isfinite(gradient).all() for gradient in gradients]
    ).all() & torch.isfinite(raw_norm_unchecked)
    raw_norm = torch.where(
        finite, raw_norm_unchecked, torch.zeros_like(raw_norm_unchecked)
    )
    clip = raw_norm.new_tensor(max_norm)
    scale = torch.where(
        raw_norm > clip,
        clip / raw_norm.clamp_min(torch.finfo(raw_norm.dtype).tiny),
        torch.ones_like(raw_norm),
    )
    scale = torch.where(finite, scale, torch.zeros_like(scale))
    delivered = [
        torch.where(
            finite,
            gradient * scale.to(gradient.dtype),
            torch.zeros_like(gradient),
        )
        for gradient in gradients
    ]
    delivered_norm = torch.stack(
        [gradient.float().square().sum() for gradient in delivered]
    ).sum().sqrt()
    return delivered, {
        "raw_norm": raw_norm,
        "delivered_norm": delivered_norm,
        "scale": scale,
        "valid": finite.to(dtype=raw_norm.dtype),
    }


@torch.no_grad()
def merge_live_head_auxiliary_gradients(parameters, auxiliary_gradients, *, valid):
    """Add finite auxiliary head gradients to PPO head gradients in-place.

    PPO gradients are authoritative and were checked by ``clip_grad_norm_`` with
    ``error_if_nonfinite=True`` before this call. Invalid auxiliary input adds exact zeros,
    producing task-only Adam parameters and moments without any CUDA host synchronization.
    """
    parameters = list(parameters)
    auxiliary_gradients = list(auxiliary_gradients)
    if len(parameters) != len(auxiliary_gradients) or not parameters:
        raise ValueError("action-head parameters and auxiliary gradients must align")
    task_gradients = [parameter.grad for parameter in parameters]
    for parameter, task_gradient, auxiliary_gradient in zip(
        parameters, task_gradients, auxiliary_gradients
    ):
        if (
            auxiliary_gradient.shape != parameter.shape
            or auxiliary_gradient.device != parameter.device
        ):
            raise ValueError("an auxiliary gradient does not match its action-head parameter")
        if task_gradient is None:
            raise RuntimeError("every action head must receive a PPO task gradient")

    task_sq = torch.stack(
        [gradient.float().square().sum() for gradient in task_gradients]
    ).sum()
    aux_finite = torch.stack(
        [torch.isfinite(gradient).all() for gradient in auxiliary_gradients]
    ).all()
    valid_tensor = torch.as_tensor(valid, device=parameters[0].device, dtype=torch.bool)
    admit = valid_tensor & aux_finite
    delivered = [
        torch.where(admit, gradient, torch.zeros_like(gradient))
        for gradient in auxiliary_gradients
    ]
    aux_sq = torch.stack(
        [gradient.float().square().sum() for gradient in delivered]
    ).sum()
    task_aux_dot = torch.stack(
        [
            (task_gradient.float() * auxiliary_gradient.float()).sum()
            for task_gradient, auxiliary_gradient in zip(
                task_gradients, delivered
            )
        ]
    ).sum()
    task_norm = task_sq.sqrt()
    aux_norm = aux_sq.sqrt()
    cosine = torch.where(
        (task_norm > 0.0) & (aux_norm > 0.0),
        task_aux_dot
        / (task_norm * aux_norm).clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(task_aux_dot),
    )

    for parameter, auxiliary_gradient in zip(parameters, delivered):
        parameter.grad.add_(auxiliary_gradient)
    return {
        "task_norm": task_norm,
        "task_aux_cosine": cosine,
        "valid": (valid_tensor & aux_finite).to(dtype=task_norm.dtype),
    }


@torch.no_grad()
def measure_livehead_probe_decomposition(
    parameters,
    pre_step_parameters,
    *,
    pre_task_probe,
    post_mixed_probe,
    evaluate_probe,
):
    """Measure trunk-only and mixed-head policy KL while exactly restoring heads.

    ``mixed_head_actor_kl`` removes both the PPO and auxiliary action-head movement; it
    must never be interpreted as auxiliary-only. Directional KL is not additive, so the
    joint KL is reported independently rather than reconstructed from the components.
    """
    parameters = list(parameters)
    pre_step_parameters = list(pre_step_parameters)
    if len(parameters) != len(pre_step_parameters) or not parameters:
        raise ValueError("pre-step action-head parameters must align")
    post_step_parameters = [parameter.detach().clone() for parameter in parameters]
    for parameter, before in zip(parameters, pre_step_parameters):
        if parameter.shape != before.shape or parameter.device != before.device:
            raise ValueError("saved action-head parameters must match live parameters")
    try:
        for parameter, before in zip(parameters, pre_step_parameters):
            parameter.copy_(before)
        trunk_only_probe = clone_behavior_probe(
            evaluate_probe(), actor_dist=post_mixed_probe.actor_dist
        )
    finally:
        for parameter, after in zip(parameters, post_step_parameters):
            parameter.copy_(after)
    return trunk_only_probe, {
        "trunk_only_actor_kl": policy_kl(pre_task_probe, trunk_only_probe),
        "mixed_head_actor_kl": policy_kl(trunk_only_probe, post_mixed_probe),
        "joint_actor_kl": policy_kl(pre_task_probe, post_mixed_probe),
    }


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
    assert 0.0 <= args.function_trust_ratio <= 1.0
    assert args.function_trust_max_kl >= 0.0
    assert 0 < args.function_trust_probe_size <= args.minibatch_size
    assert args.function_trust_verify_rtol >= 0.0
    assert args.function_trust_atol >= 0.0
    assert args.nextlat_head_grad_clip >= 0.0
    assert args.actor_dist in ("beta", "gaussian")
    assert args.separate_grad_clip, (
        "function-trust NextLat requires decoupled task/auxiliary gradients"
    )
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

    agent = Agent(envs, args).to(device)
    # Three independent Adam streams: task sees actor+critic, predictive-trunk sees
    # representation gradients, and predictor sees only its private dynamics gradients.
    # Keeping predictor state separate prevents its much smaller tensors from sharing a
    # clip budget or transaction semantics with the protected overlapping trunk.
    task_optimizer = optim.Adam(agent.task_parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    action_head_params = agent.action_head_parameters()
    critic_params = agent.critic_parameters()
    nextlat_params = agent.nextlat_parameters()
    nextlat_trunk_params = agent.nextlat_trunk_parameters()
    nextlat_predictor_params = agent.nextlat_predictor_parameters()
    assert {id(p) for p in nextlat_params} == {
        id(p) for p in nextlat_trunk_params + nextlat_predictor_params
    }
    assert {id(p) for p in action_head_params}.isdisjoint(
        {id(p) for p in nextlat_params}
    )
    predictive_trunk_optimizer = optim.Adam(
        nextlat_trunk_params, lr=args.learning_rate, eps=1e-5
    )
    predictor_optimizer = optim.Adam(
        nextlat_predictor_params, lr=args.learning_rate, eps=1e-5
    )
    function_trust_layout = make_update_tensor_layout(nextlat_trunk_params)

    def policy_rollout_fn(obs_):
        return agent.get_action_and_value(obs_)

    def policy_update_fn(obs_, z_):
        return agent.get_action_and_value(obs_, z_)

    def value_h0_fn(obs_):
        return agent.get_value_h0(obs_)

    def target_actor_feat_fn(obs_):
        return agent.get_actor_feat(obs_)

    def behavior_probe_fn(obs_):
        return agent.get_behavior_probe(obs_)

    if args.compile:
        policy_rollout_fn = torch.compile(
            policy_rollout_fn, mode=args.compile_mode, dynamic=False
        )
        policy_update_fn = torch.compile(
            policy_update_fn, mode=args.compile_mode, dynamic=False
        )
        value_h0_fn = torch.compile(value_h0_fn, mode=args.compile_mode, dynamic=False)
        target_actor_feat_fn = torch.compile(
            target_actor_feat_fn, mode=args.compile_mode, dynamic=False
        )
        behavior_probe_fn = torch.compile(
            behavior_probe_fn, mode=args.compile_mode, dynamic=False
        )
        print(f"compiled agent forward paths ({args.compile_mode})")

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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            task_optimizer.param_groups[0]["lr"] = lrnow
            predictive_trunk_optimizer.param_groups[0]["lr"] = lrnow
            predictor_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                action, z, logprob, ent, value_logits, actor_feat = policy_rollout_fn(next_obs)
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
                latents[step] = actor_feat
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
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            next_transition_value_logits = value_h0_fn(
                next_obses.reshape((-1,) + envs.single_observation_space.shape)
            )
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

        b_inds = np.arange(args.batch_size)
        if args.nextlat:
            nextlat_action_offsets = (
                np.arange(args.nextlat_depth, dtype=np.int64)[:, None] * args.num_envs
            )
            nextlat_target_offsets = (
                np.arange(1, args.nextlat_depth + 1, dtype=np.int64)[:, None]
                * args.num_envs
            )
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                _, _, newlogprob, entropy, value_logits, mb_actor_feat = policy_update_fn(
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

                # v162 HL-Gauss MTP value loss: per-horizon CE to scalar-return
                # targets, summed across valid horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds].to(device=value_logits.device, non_blocking=True)
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(device=value_logits.device, dtype=value_ce.dtype, non_blocking=True)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

                if args.nextlat:
                    # Deterministic first rows of this minibatch: no sampling and a fixed
                    # compiled shape. Reuse the already-computed trunk features and critic
                    # logits, so the pre-task reference needs no extra trunk forward.
                    probe_size = args.function_trust_probe_size
                    probe_obs = b_obs[mb_inds[:probe_size]]
                    probe_critic_mask = value_mask[:probe_size].detach().clone()
                    with torch.no_grad():
                        probe_dist, _, _ = agent._actor_dist(
                            mb_actor_feat[:probe_size]
                        )
                        if args.actor_dist == "beta":
                            probe_first = probe_dist.concentration1
                            probe_second = probe_dist.concentration0
                        else:
                            probe_first = probe_dist.loc
                            probe_second = probe_dist.scale
                        pre_task_probe = BehaviorProbe(
                            probe_first.detach().clone(),
                            probe_second.detach().clone(),
                            value_logits[:probe_size].detach().clone(),
                            actor_dist=args.actor_dist,
                        )

                # NEXTLAT: roll the latent dynamics MLP to depth d from the CURRENT trunk
                # latent, conditioned on the buffered actions; Smooth L1 each predicted
                # latent against the ONLINE stop-grad trunk latent of the true future obs,
                # and KL-distill the true future action distribution into the one decoded
                # from the predicted latent. T-major flattening => index + i*num_envs is
                # (t+i, same env); masks exclude boundary-crossing and rollout-tail rows.
                if args.nextlat:
                    action_indices = np.clip(
                        mb_inds[None, :] + nextlat_action_offsets,
                        0,
                        args.batch_size - 1,
                    )
                    target_indices = np.clip(
                        mb_inds[None, :] + nextlat_target_offsets,
                        0,
                        args.batch_size - 1,
                    )
                    future_actions = b_actions[action_indices]
                    with torch.no_grad():
                        # One target-trunk call for every horizon. The old loop issued d
                        # identical-shape trunk calls and rebuilt d eager/compiled graphs.
                        future_target_feats = target_actor_feat_fn(
                            b_obs[target_indices.reshape(-1)]
                        ).reshape(args.nextlat_depth, args.minibatch_size, args.hidden)
                    h_hat = mb_actor_feat
                    pred_losses, kl_losses = [], []
                    for i in range(1, args.nextlat_depth + 1):
                        h_hat = agent.nextlat_predictor(
                            torch.cat([h_hat, future_actions[i - 1]], dim=-1)
                        )
                        with torch.no_grad():
                            tgt_feat = future_target_feats[i - 1]
                            t_dist, _, _ = agent._actor_dist(tgt_feat)  # teacher (sg)
                        mask_i = b_nextlat_mask[mb_inds, i - 1]
                        denom = mask_i.sum().clamp_min(1.0)
                        pred_l = F.smooth_l1_loss(h_hat, tgt_feat, reduction="none").mean(-1)
                        pred_losses.append((pred_l * mask_i).sum() / denom)
                        # Live student decoder: its latent Jacobian is mathematically
                        # identical to v14's frozen-weight decode at current parameters,
                        # while its own weights receive a separately budgeted gradient.
                        s_dist, _, _ = agent._actor_dist_live_head(h_hat)
                        kl_l = torch.distributions.kl_divergence(t_dist, s_dist).sum(-1)
                        kl_losses.append((kl_l * mask_i).sum() / denom)
                    nextlat_pred_loss = torch.stack(pred_losses).mean()
                    nextlat_kl_loss = torch.stack(kl_losses).mean()
                    nextlat_loss = nextlat_pred_loss + args.nextlat_kl_coef * nextlat_kl_loss
                else:
                    nextlat_loss = None

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

                # Three backwards, three Adam streams. Actor and critic are clipped and
                # summed before task Adam exactly as in the incumbent. NextLat trunk and
                # predictor gradients have separate clip budgets and optimizer states.
                agent.zero_grad(set_to_none=True)
                (args.vf_coef * v_loss).backward(retain_graph=True)
                critic_gn = nn.utils.clip_grad_norm_(
                    critic_params,
                    args.critic_grad_clip,
                    error_if_nonfinite=True,
                )
                value_grads = {
                    p: p.grad.detach().clone()
                    for p in critic_params
                    if p.grad is not None
                }

                if args.nextlat:
                    agent.zero_grad(set_to_none=True)
                    (args.nextlat_coef * nextlat_loss).backward(retain_graph=True)
                    nextlat_trunk_gn = nn.utils.clip_grad_norm_(
                        nextlat_trunk_params, args.nextlat_trunk_grad_clip
                    )
                    nextlat_predictor_gn = nn.utils.clip_grad_norm_(
                        nextlat_predictor_params, args.nextlat_predictor_grad_clip
                    )
                    (
                        nextlat_head_grads,
                        livehead_aux_stats,
                    ) = prepare_live_head_auxiliary_gradients(
                        action_head_params,
                        args.nextlat_head_grad_clip,
                    )
                    nextlat_trunk_grads = [
                        parameter.grad.detach().clone()
                        for parameter in nextlat_trunk_params
                    ]
                    nextlat_predictor_grads = [
                        parameter.grad.detach().clone()
                        for parameter in nextlat_predictor_params
                    ]
                else:
                    nextlat_trunk_grads = nextlat_predictor_grads = []
                    nextlat_head_grads = []

                agent.zero_grad(set_to_none=True)
                (pg_loss - ent_coef_eff * entropy_loss).backward()
                actor_gn = nn.utils.clip_grad_norm_(
                    actor_params,
                    args.actor_grad_clip,
                    error_if_nonfinite=True,
                )
                if args.nextlat:
                    actor_trunk_grads = [
                        parameter.grad.detach().clone()
                        if parameter.grad is not None
                        else torch.zeros_like(parameter)
                        for parameter in nextlat_trunk_params
                    ]
                    critic_trunk_grads = [
                        value_grads[parameter]
                        if parameter in value_grads
                        else torch.zeros_like(parameter)
                        for parameter in nextlat_trunk_params
                    ]
                for parameter, gradient in value_grads.items():
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad + gradient
                    )

                if args.nextlat:
                    livehead_merge_stats = merge_live_head_auxiliary_gradients(
                        action_head_params,
                        nextlat_head_grads,
                        valid=livehead_aux_stats["valid"],
                    )
                    with torch.no_grad():
                        task_before = [
                            parameter.detach().clone()
                            for parameter in nextlat_trunk_params
                        ]
                        livehead_before = [
                            parameter.detach().clone()
                            for parameter in action_head_params
                        ]
                task_optimizer.step()
                if args.nextlat:
                    task_updates = [
                        parameter.detach() - previous
                        for parameter, previous in zip(
                            nextlat_trunk_params, task_before
                        )
                    ]
                    # The joint PPO+live-head Adam step is the actor ruler; the critic
                    # ruler remains task-only. Clone compiled outputs because later probe
                    # calls may reuse graph buffers.
                    with torch.no_grad():
                        post_task_probe = clone_behavior_probe(
                            behavior_probe_fn(probe_obs),
                            actor_dist=args.actor_dist,
                        )
                        livehead_after = [
                            parameter.detach().clone()
                            for parameter in action_head_params
                        ]
                        livehead_step_norm = torch.stack(
                            [
                                (after - before).float().square().sum()
                                for after, before in zip(
                                    livehead_after, livehead_before
                                )
                            ]
                        ).sum().sqrt()

                        # Diagnostics are emitted once per PPO iteration, so avoid an
                        # otherwise wasted extra trunk forward on non-final minibatches.
                        if end >= args.batch_size:
                            (
                                _,
                                livehead_probe_stats,
                            ) = measure_livehead_probe_decomposition(
                                action_head_params,
                                livehead_before,
                                pre_task_probe=pre_task_probe,
                                post_mixed_probe=post_task_probe,
                                evaluate_probe=lambda: behavior_probe_fn(
                                    probe_obs
                                ),
                            )

                    agent.zero_grad(set_to_none=True)
                    function_trust_predictor_step_norm = apply_private_optimizer_step(
                        nextlat_predictor_params,
                        predictor_optimizer,
                        nextlat_predictor_grads,
                    )
                    (
                        raw_predictive_updates,
                        projected_predictive_updates,
                        admitted_updates,
                        function_trust_stats,
                    ) = apply_function_trust_transaction(
                        nextlat_trunk_params,
                        predictive_trunk_optimizer,
                        nextlat_trunk_grads,
                        task_updates,
                        actor_gradients=actor_trunk_grads,
                        critic_gradients=critic_trunk_grads,
                        pre_task_probe=pre_task_probe,
                        post_task_probe=post_task_probe,
                        critic_mask=probe_critic_mask,
                        evaluate_probe=lambda: clone_behavior_probe(
                            behavior_probe_fn(probe_obs),
                            actor_dist=args.actor_dist,
                        ),
                        trust_ratio=args.function_trust_ratio,
                        max_kl=args.function_trust_max_kl,
                        verify_rtol=args.function_trust_verify_rtol,
                        atol=args.function_trust_atol,
                        layout=function_trust_layout,
                    )
                else:
                    zero_stat = torch.zeros((), device=device)
                    function_trust_stats = {
                        "task_norm": zero_stat,
                        "predictive_norm": zero_stat,
                        "projected_norm": zero_stat,
                        "raw_task_cosine": zero_stat,
                        "projection_fraction": zero_stat,
                        "accepted_fraction": zero_stat,
                        "admitted_norm": zero_stat,
                        "scale": zero_stat,
                        "proposal_scale": zero_stat,
                        "accepted": zero_stat,
                        "numeric_valid": zero_stat,
                        "actor_first_order": zero_stat,
                        "critic_first_order": zero_stat,
                        "actor_conflict_fraction": zero_stat,
                        "critic_conflict_fraction": zero_stat,
                        "projected_tensor_fraction": zero_stat,
                        "task_actor_kl": zero_stat,
                        "task_critic_kl": zero_stat,
                        "actor_budget": zero_stat,
                        "critic_budget": zero_stat,
                        "full_actor_kl": zero_stat,
                        "full_critic_kl": zero_stat,
                        "verified_actor_kl": zero_stat,
                        "verified_critic_kl": zero_stat,
                    }
                    function_trust_predictor_step_norm = zero_stat

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_per_h = b_target_probs[..., 0] + b_target_probs[..., -1]
        edge_mask_f = b_target_mask.to(dtype=edge_per_h.dtype)
        edge_mass = ((edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)).item()

        writer.add_scalar(
            "charts/learning_rate", task_optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        if args.nextlat:
            writer.add_scalar("losses/nextlat_pred_loss", nextlat_pred_loss.item(), global_step)
            writer.add_scalar("losses/nextlat_kl_loss", nextlat_kl_loss.item(), global_step)
            writer.add_scalar(
                "losses/nextlat_trunk_grad_norm", float(nextlat_trunk_gn), global_step
            )
            # Preserve the incumbent tag for matched-run dashboards; it denotes the
            # representation/trunk auxiliary before function-space admission.
            writer.add_scalar(
                "losses/nextlat_grad_norm", float(nextlat_trunk_gn), global_step
            )
            writer.add_scalar(
                "losses/nextlat_predictor_grad_norm",
                float(nextlat_predictor_gn),
                global_step,
            )
            writer.add_scalar("debug/latent_batch_std", latent_batch_std, global_step)
            writer.add_scalar(
                "function_trust/predictor_step_norm",
                float(function_trust_predictor_step_norm),
                global_step,
            )
            livehead_tags = {
                "raw_aux_grad_norm": livehead_aux_stats["raw_norm"],
                "delivered_aux_grad_norm": livehead_aux_stats["delivered_norm"],
                "aux_grad_scale": livehead_aux_stats["scale"],
                "numeric_valid": livehead_merge_stats["valid"],
                "task_grad_norm": livehead_merge_stats["task_norm"],
                "task_aux_cosine": livehead_merge_stats["task_aux_cosine"],
                "mixed_head_step_norm": livehead_step_norm,
                "trunk_only_actor_kl": livehead_probe_stats["trunk_only_actor_kl"],
                "mixed_head_actor_kl": livehead_probe_stats["mixed_head_actor_kl"],
                "joint_actor_kl": livehead_probe_stats["joint_actor_kl"],
            }
            for tag, statistic in livehead_tags.items():
                writer.add_scalar(
                    f"livehead/{tag}", float(statistic), global_step
                )
            function_trust_tags = {
                "task_step_norm": "task_norm",
                "raw_predictive_step_norm": "predictive_norm",
                "projected_step_norm": "projected_norm",
                "admitted_step_norm": "admitted_norm",
                "raw_task_cosine": "raw_task_cosine",
                "projection_fraction": "projection_fraction",
                "accepted_fraction": "accepted_fraction",
                "admitted_scale": "scale",
                "proposal_scale": "proposal_scale",
                "accepted": "accepted",
                "numeric_valid": "numeric_valid",
                "projected_tensor_fraction": "projected_tensor_fraction",
                "actor_conflict_fraction": "actor_conflict_fraction",
                "critic_conflict_fraction": "critic_conflict_fraction",
                "actor_first_order": "actor_first_order",
                "critic_first_order": "critic_first_order",
                "task_actor_kl": "task_actor_kl",
                "task_critic_kl": "task_critic_kl",
                "actor_budget": "actor_budget",
                "critic_budget": "critic_budget",
                "full_actor_kl": "full_actor_kl",
                "full_critic_kl": "full_critic_kl",
                "verified_actor_kl": "verified_actor_kl",
                "verified_critic_kl": "verified_critic_kl",
            }
            for tag, stat in function_trust_tags.items():
                writer.add_scalar(
                    f"function_trust/{tag}",
                    float(function_trust_stats[stat]),
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

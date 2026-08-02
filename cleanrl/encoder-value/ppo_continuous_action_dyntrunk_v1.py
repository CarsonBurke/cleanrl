# DYNTRUNK v1 -- THE TRUNK IS THE REPRESENTATION.
#
# THE OBSERVATION THIS FILE IS BUILT ON. Every representation learner in this family has
# been a DETACHED SIDE-CAR with its own optimizer and no gradient path into the trunk that
# the actor and critic actually read: the LeJEPA encoder, the JEPA predictor, SIGReg, and
# most recently CLA's dynamics model. Not one of them could change the features the policy
# sees. The measured consequence is a family where deleting the encoder outright
# (sf_noe_v1, 9953 +/-109 @8M -- this file's PARENT) beats every variant that keeps it.
#
# It also explains the kill-switch number that ended the latent-critic line:
#   gate/ev_latent_cap  (ridge of MC return onto sg(e))     0.32 @1M / 0.66 late
#   gate/ev_trunk_probe (the same ridge on trunk features)  0.61 @1M / 0.88 late
# The latent was a worse basis for value than the trunk because it was trained by
# objectives unrelated to value. That is not an argument against learned representations.
# It is an argument against UNSHARED ones.
#
# THE CHANGE. One auxiliary loss, with a real gradient path:
#
#   trunk_feat = Trunk(s)                       # actor, critic and phi ALL read this
#   L_dyn      = MSE( DynHead(trunk_feat, a),  s' - s )      masked on valids
#   loss       = pg_loss + vf_coef*v_loss + dyn_coef*L_dyn
#   phi        = [ sg(trunk_feat), s, a, a*a, 1 ]
#
# Two halves, and both are needed for the hypothesis to be tested rather than half-tested:
# L_dyn SHAPES the trunk, phi_trunk makes the value path CONSUME it (the SF critic must now
# predict the discounted occupancy of its own representation). --no-phi-trunk ablates the
# second half; --dyn-coef 0 ablates the first. Only BOTH together recover the parent
# exactly -- --dyn-coef 0 alone still widens phi from 30 to 94 dimensions and so is a
# different value objective, not a control.
#
# WHY THE TARGET IS THE OBSERVATION DELTA. Predicting next TRUNK FEATURES under a stop-grad
# is a BYOL objective and collapses without an EMA teacher or SIGReg -- this file has
# neither, by design. An observation-space target is grounded and collapse-free. The delta
# rather than s' for the SHORTCUT reason, not a scale one: s' = s + Delta, and the head is
# handed s's own trunk encoding, so regressing s' admits a near-identity solution that is
# accurate without predicting anything -- exactly the degenerate optimum this loss exists
# to forbid. It is NOT that the delta is small: measured on the parent's rollouts the
# per-dimension Var(s' - s) averages 1.36, LARGER than Var(s) = 1, because at frame_skip=5
# the normalized velocity dimensions are nearly uncorrelated across steps. So the delta is
# the harder target as well as the honest one, and dyn/target_var is logged rather than
# assumed.
#
# UNDER --no-share-backbone THE HEADLINE CLAIM ABOVE DOES NOT HOLD, and a reader of a run
# using it should know: dyn_head and phi both read `critic_feat`, so with separate trunks the
# auxiliary shapes the CRITIC trunk ONLY and the actor gets an untouched trunk of its own.
# That makes it a test of whether forward-prediction pressure helps the VALUE path -- a
# cleaner question in one way (the auxiliary can no longer perturb the policy's features
# behind the policy gradient's back) and a weaker one in another (the policy never reads the
# shaped representation at all).
#
# WHY THIS IS NOT CLA. CLA's dynamics model had its own Adam and ran under no_grad; its
# only channel was the value baseline, so it could not have been doing representation
# learning at all. Here the dynamics gradient lands ON the shared trunk. That is the
# difference the +3.8%-over-control-at-4M result could not distinguish.
#
# HYPOTHESIS, and what would falsify it. A trunk pressured to be predictive of its own
# one-step future is a better basis for both value and policy, so this should beat the
# parent (9953) and the family best (10071 +/-67). Falsifiers, logged from iteration 1:
# dyn/r2 near 0 means the auxiliary never learned and the trunk got noise; dyn/r2 near 1
# from the start means the task was too easy to shape anything and this is a null.
# =====================================================================================
# THE QUESTION. lejepa_sf_v2 beats its base by a wide margin (9,371 @5.7M vs 8,278 @8M),
# but it changes TWO things at once relative to the base: (a) the critic head stops
# predicting scalar returns and predicts vector-valued successor features instead, and
# (b) a separately-trained LeJEPA encoder supplies 32 of phi's 62 dimensions. This file
# removes (b) and keeps (a), so the two are separable:
#
#     phi = [ s , a , a*a , 1 ]        instead of   [ e , s , a , a*a , 1 ]
#     sf_dim = 30                      instead of   62
#
# and the entire SSL path -- encoder, action encoder, causal predictor, SIGReg, its
# optimizer, its sequence chunking, its drift probe -- is DELETED, not disabled. Nothing
# else differs from v2 by a single line.
#
# WHY IT IS WORTH A RUN AND NOT AN ARGUMENT. v2's phi already contains the raw
# (NormalizeObservation-scaled) state, so on the REWARD-probe axis e is provably almost
# redundant: measured offline, [s,s',a,a2,1] scored R^2 = 0.9370 and [e,s,s',a,a2,1]
# scored 0.9382. e bought 0.001. But the reward probe is not the only thing phi does --
# Lambda is the DISCOUNTED OCCUPANCY of phi, so e's block contributes 32 extra coordinates
# of "where this policy goes", which is exactly the structure a linear function of
# instantaneous s cannot express. Whether that occupancy structure is worth anything is a
# question about the critic's regression target, and no probe answers it.
#
# WHAT EACH OUTCOME MEANS.
#   noe ~= v2   -> the encoder is dead weight and the real discovery is much simpler and
#                  stronger: SUCCESSOR FEATURES OVER RAW STATE beat a scalar HL-Gauss
#                  critic. 247k params, SIGReg, and the transformer all go away. It would
#                  also mean the JEPA model is being UNDER-used, not un-needed: its output
#                  currently enters only as 32 reward-basis coordinates that a linear
#                  readout collapses back to one scalar. The response is to give the
#                  embedding a job the scalar path cannot do (a vector-valued policy
#                  signal replacing GAE), not to delete it.
#   noe <  v2   -> e's occupancy block is carrying real credit-assignment structure, and
#                  the gap is the first honest measurement of how much.
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
#   phi_t = [ s_t , a_t , a_t*a_t , 1 ]   reward features; dim = obs_dim + 2*act_dim + 1
#   psi_h(s_t) ~ Lambda_{t+h}          CRITIC HEAD on the ThinkTrunk, h = 0..5
#   Lambda_t  = E_pi[ sum_k gamma^k phi_{t+k} ]   successor features -- the ONLY
#                                                 bootstrapped object, vector-valued
#   V(s)  = w_r . psi_0(s)             evaluation, a dot product
#   w_r   solved from w_r.phi_t ~ r_t  IMMEDIATE reward only, closed-form ridge,
#                                      no bootstrapping, recomputed every iteration
#
# The only scalar regression left anywhere is w_r against immediate reward.
#
# WHY REPLACE THE HEAD, NOT ADD A PATHWAY. share_backbone=True: actor and critic fully
# share the ThinkTrunk. Deleting the critic would strip ~half the trunk's gradient, and
# bolting SF on in parallel would have psi read a near-raw-obs latent while a scalar
# critic reads the policy-shaped trunk -- a regression then would not be attributable to
# "SF vs scalar" at all. So the swap is surgical: same head shape, same MTP semantics,
# same boundary masks; only K goes from 511 bucket logits to obs_dim + 2*act_dim + 1 = 30
# SF dimensions (HalfCheetah: 17 + 12 + 1), and the loss from cross-entropy to masked MSE
# on standardized Lambda.
# It also deletes the 402MB CPU target_probs tensor and its ~4GB/iteration H2D traffic.
#
# HONEST ACCOUNTING (the strong stationarity claim does NOT survive):
#   - psi is still policy-dependent; its fixed point moves with pi exactly as V^pi does.
#     What changes is that the bootstrap target is an OCCUPANCY object with a marginal
#     pinned by the observation normalizer, not that pi-dependence is gone.
#   - w_r is only approximately policy-independent -- it is fit over the policy-induced
#     distribution, which is pi-invariant only if the regression were well-specified.
#   - a*a is the MOST policy-dependent feature here: with a Beta actor
#     E_pi[a*a|s] = Var_pi + mean^2, so that block tracks policy entropy directly.
#     Correct (ctrl cost genuinely depends on pi) but the vector is not uniformly stable.
#   - Dropping the encoder also drops v2's one self-inflicted non-stationarity (SIGReg
#     pins the distribution of e, not its coordinate FRAME, while psi_old and w_r are
#     functions of coordinates). v2 measured frame_drift_raw = 0.9993, i.e. that risk
#     never materialized, so this is a simplification and not an improvement.
#
# INSTRUMENTATION. The obvious EV metric is rigged: b_returns = advantages + values
# contains the critic's own bootstrapped values, and at dt=0.05 errors are strongly
# state-autocorrelated, so a critic scores well against a target built from itself.
# gate/* instead scores three predictors against a common TRUNCATED-MC return:
#   (1) ev_trunk_probe  detached scalar probe on sg(trunk_feat)  -- what a scalar critic
#                       on this trunk would achieve (replaces the lost HL-Gauss comparator)
#   (2) ev_sf_online    w_r.psi_0, the value the actor actually consumed -- the treatment
#   (3) ev_obs_probe    closed-form ridge of MC return onto raw s -- how much reward
#                       signal is LINEARLY present in the INSTANTANEOUS state. This slot
#                       held ev_latent_cap (the same probe on e) in v1/v2, so the two
#                       runs' curves are directly comparable: the difference between them
#                       is precisely what the encoder adds to a one-step linear readout.
# (3) is NOT a ceiling on (2): psi reads the trunk and accumulates phi over time, so
# (2) > (3) is expected and is the successor-feature construction earning its keep. The
# kill signal is (2) << (1) sustained -- the SF critic losing to what a plain scalar
# critic on the same trunk would achieve.
#
# HYPOTHESES. H1 latent-path loss stays smooth across policy shifts where scalar-critic
# EV spikes. H2 (THE ONE THIS FILE TESTS, by removing it) latent occupancy captures
# gait/contact structure a linear function of s lacks. H3 (the falsifier) EV(w_r.psi_0)
# vs a scalar critic's EV on a COMMON target.
#
# Naming: `z`/`latent_zs` is the Beta actor's native action sample (inherited).
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

from cleanrl.shared.lejepa import CompiledModule

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
    # NOTE: the reference run this variant is built against (exp-name
    # "iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_v1", 8278 @8M) was NOT a separate
    # file -- it was mbpercnorm_v2 launched with --norm-adv --norm-adv-scope batch
    # --no-ret-percnorm. Those three are baked in as defaults here so the baseline is
    # reproduced by default rather than by remembering flags.
    norm_adv: bool = True            # ppoadvnorm: plain PPO advantage standardization...
    # --- Percentile advantage normalization (disabled: superseded by norm_adv) ---
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
    # -- SHARED DYNAMICS TRUNK ---------------------------------------------------------
    dyn_coef: float = 1.0            # weight of the forward-prediction auxiliary. This is the
    #   whole point of the file: it is the ONLY term in this family that has ever put a
    #   representation-learning gradient into the trunk the actor and critic actually read.
    #   0.0 recovers the parent exactly.
    dyn_head_hidden: int = 256
    dyn_grad_clip: float = 0.25      # the auxiliary gets its OWN backward and its OWN clip.
    #   Measured on the parent's tfevents, losses/critic_grad_norm runs 0.30/0.74/1.01/0.65
    #   against critic_grad_clip=0.25 -- the critic clip is SATURATED for essentially the
    #   whole run. So the clip is not a safety net, it is an always-on normalizer, and
    #   folding dyn_loss into the value backward would not ADD to the critic update, it
    #   would ROTATE a fixed-norm budget: the value share becomes 0.25*g_v/||g_v+g_dyn||.
    #   With the two losses the same order of magnitude that silently HALVES the critic's
    #   effective learning rate, and a regression would be misread as "the representation
    #   hypothesis is false" when the real cause was a critic trained at half speed.
    phi_trunk_loss_coef: float = 0.5  # weight of the trunk block in the value loss. v_loss is
    #   a mean over sf_dim, so a bare 30->94 widening would drop the raw-obs block -- the one
    #   w_r needs to recover the velocity term -- from 17/30 = 57% of the loss to 17/94 = 18%.
    #   The trunk block is therefore ADDED as a separate weighted term rather than diluting
    #   the parent's basis, so the reward-relevant gradient keeps its parent magnitude.
    phi_trunk: bool = True           # put the (detached) trunk features into phi, so the
    #   SF critic must predict the discounted occupancy of its OWN representation. Without
    #   this the auxiliary shapes the trunk but the value path never consumes it, and the
    #   experiment only half-tests the hypothesis. --no-phi-trunk is the ablation.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # The HL-Gauss bucket critic is GONE. The critic head now emits successor features
    # (see header). MTP semantics and the horizon masks are retained verbatim -- only the
    # per-horizon target changed from a 511-bin scalar-return distribution to a
    # K-dimensional occupancy vector.
    critic_mtp_horizon: int = 6

    # --- successor-feature value pathway ---------------------------------------------
    # No encoder args: phi = [s, a, a*a, 1] and the whole SSL path is gone (see header).
    sf_alpha: float = 1.0            # 1.0 = pure occupancy prediction, ZERO scalar regression in the
    #                                  network. <1 mixes in MSE(w_r.psi, w_r.Lambda) -- the fallback
    #                                  if the pure form underfits, not the starting point.
    sf_ridge: float = 1e-3           # ridge coefficient for the closed-form w_r solve
    sf_target_ema: float = 0.01      # EMA rate for per-dimension Lambda standardization
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
    # So the trunk gets inductor fusion without graphs. (v2 also compiled the SSL nets with
    # reduce-overhead; there are none left here, so compile_mode is accepted and unused --
    # kept so the canonical mlq submit line is identical across the two runs.)
    compile: bool = False            # validate the port eager first -- compile changes numerics
    compile_mode: str = "reduce-overhead"  # accepted but UNUSED (no SSL nets left to apply it to).
    #   Kept so the canonical submit line parses. NOTE parity is partial: --compile-ssl-cudagraphs
    #   was deleted with the SSL path, so that one v2 flag is a tyro error here.

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


def phi_features(obs, action, trunk_feat=None):
    """phi = [s, a, a*a, 1] -- v2's basis with the learned-embedding block removed.

    The raw block is the already-NormalizeObservation-scaled state, which is what makes
    HalfCheetah's velocity term exactly linearly recoverable by w_r (v1, which had only
    the embedding, measured a lag-1 residual autocorrelation of 0.474 -- structured error
    injected into every value estimate; v2 with s present measured 0.254 at R^2 0.988).

    The action blocks are not decoration. HalfCheetah's reward is x_vel - 0.1*||a||^2, and
    the control cost is not a function of state AT ALL -- so a probe on s alone carries an
    irreducible residual that scales with policy action magnitude, i.e. the very
    non-stationarity this design removes leaks straight back in. `a*a` lets a LINEAR probe
    capture MuJoCo ctrl cost exactly.

    The trailing constant is the regression intercept, carried as a REAL feature rather
    than a separate bias so that V = w_r . psi stays an exact identity (a bias term would
    have no discounted-sum counterpart in psi). It earns its keep twice over: the
    observation is normalized, so s has a running-mean offset that shifts as the policy's
    state distribution moves, and the constant's own discounted sum is 1/(1-gamma)
    truncated at episode end -- that coordinate quietly encodes expected remaining
    episode length.
    """
    ones = obs[..., :1].new_ones(obs.shape[:-1] + (1,))
    parts = [obs, action, action * action, ones]
    if trunk_feat is not None:
        # PREPENDED, and DETACHED by construction (built inside the rollout's no_grad
        # block). Detachment matters: phi is a TARGET. If gradient could reach the trunk
        # through it, the trunk could shrink its own features to make the successor-feature
        # regression trivially easy -- a collapse whose only symptom would be a beautifully
        # falling value loss. The forward-prediction auxiliary is what actually holds the
        # representation up; phi_trunk is what makes the value path consume it.
        # RMS-NORMALIZED to unit scale per row. ThinkTrunk applies out_norm BEFORE
        # out_proj, so nothing otherwise pins critic_feat's scale -- and the Lambda
        # standardizer's EMA rate is floored at 0.01, i.e. ~100 iterations of lag on a
        # 244-iteration run. An auxiliary loss actively reshaping the trunk could grow its
        # output scale faster than sf_std tracks it, inflating the trunk block's share of
        # the value loss with no logged cause. Normalizing here pins it at the source and
        # simultaneously removes any across-iteration incentive to shrink the features.
        tf = trunk_feat * (
            trunk_feat.shape[-1] ** 0.5 / trunk_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        parts = [tf] + parts
    return torch.cat(parts, dim=-1)


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
        # SUCCESSOR-FEATURE critic head. Same shape and MTP semantics as the HL-Gauss head
        # it replaces -- only the per-horizon target space changed, from 511 scalar-return
        # bucket logits to K = obs_dim + 2*act_dim + 1 occupancy dimensions.
        # Horizon h predicts Lambda_{t+h}, the vector-valued lambda-return of phi.
        # Zero-init is deliberate and not merely neutral: with psi_old == 0 the first
        # rollout's Lambda degenerates to the plain discounted sum of phi, i.e. a clean
        # Monte-Carlo target with no bootstrap from a random head.
        self.phi_trunk = args.phi_trunk
        self.sf_dim = (H if args.phi_trunk else 0) + obs_dim + 2 * act_dim + 1
        # FORWARD-PREDICTION AUXILIARY. Reads the SHARED trunk features, so its gradient
        # lands on the representation the actor and critic both consume -- unlike every
        # previous representation learner in this family (encoder, JEPA predictor, SIGReg,
        # the CLA dynamics model), each of which lived in its own optimizer with no gradient
        # path into the trunk at all. Target is the OBSERVATION delta, not next trunk
        # features: a stop-grad feature target is a BYOL objective and collapses without an
        # EMA teacher or SIGReg, and this file deliberately has neither.
        # STANDARD init on the output layer, NOT zero-init. Zero-init is right whenever a
        # head's OUTPUT enters the value or action path (CLA's ObsDynamics, AdaLN-zero
        # modulation) because it makes an untrained module an exact identity. It is exactly
        # wrong here: with W = 0 the gradient w.r.t. the head's INPUT is grad_out @ W = 0,
        # so the auxiliary would be unable to reach the trunk at all on the first step --
        # silently disabling the one mechanism this file exists to test. The output is only
        # ever consumed by a loss, so a non-trivial init costs nothing but a larger initial
        # dyn/loss. std=1.0 rather than the near-zero 0.01 a first draft used: the argument
        # above scales linearly, and at std=0.01 the initial trunk-bound gradient is ~1% of
        # its natural magnitude, which is the same failure in miniature. dyn_grad_clip bounds
        # the cost of a large init; nothing bounds the cost of a severed path.
        self.dyn_head = nn.Sequential(
            layer_init(nn.Linear(H + act_dim, args.dyn_head_hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(args.dyn_head_hidden, obs_dim), std=1.0),
        )
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * self.sf_dim, bias=False), std=0.1
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
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
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
        return action, z, log_prob, entropy, value_sf, critic_feat

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

    def dyn_parameters(self):
        # Params receiving the AUXILIARY gradient: the shared trunk and the dyn head. A
        # THIRD decoupled clip group, extending the file's existing actor/critic split for
        # the reason in dyn_grad_clip's note -- the critic clip is saturated, so sharing it
        # would make dyn_coef a divisor on the value gradient instead of a weight on the
        # auxiliary. The trunk deliberately appears in all three groups; each branch clips
        # its own contribution and they are summed on the trunk, which is the parent's design.
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.dyn_head.parameters())


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
    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    dyn_params = agent.dyn_parameters()

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

    # --- successor-feature value pathway ---------------------------------------------
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    sf_dim = agent.sf_dim

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

    # Per-dimension EMA standardization of the SF target. The blocks of phi are wildly
    # heteroscedastic: for a zero-mean unit-variance s block std(sum gamma^k s) ~ tau ~ 20
    # (the effective autocorrelation horizon, NOT 1/(1-gamma)=100), while the a*a block is
    # strictly positive with a large mean and small variance, and the constant block is
    # ~1/(1-gamma) exactly. A single global rescale leaves the MSE dominated by the wrong
    # block.
    sf_mean = torch.zeros(sf_dim, device=device)
    sf_std = torch.ones(sf_dim, device=device)
    sf_stat_count = 0
    # w_r starts at zero => V == 0 for the first rollout, which is the correct "no
    # information" baseline rather than random noise driving the actor.
    w_r = torch.zeros(sf_dim, device=device)

    def psi_raw(sf_standardized):
        """Un-standardize the head output back into phi-accumulation units."""
        return sf_standardized * sf_std + sf_mean

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
                action, z, logprob, ent, value_sf, _ = agent.get_action_and_value(next_obs)
                values[step] = psi_raw(value_sf[:, 0]) @ w_r
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
            flat_obs_shape = (-1,) + envs.single_observation_space.shape
            psi_next = psi_raw(agent.get_value(next_obses.reshape(flat_obs_shape))[:, 0]).reshape(
                args.num_steps, args.num_envs, sf_dim
            )
            # Kept as trunk features, not just psi: the EV gate below needs the SAME
            # features to fit its detached scalar probe, and this variant already does two
            # full-batch trunk forwards per iteration where the base did one.
            critic_feat_buf = agent._trunks(obs.reshape(flat_obs_shape))[1]
            psi_cur = psi_raw(
                agent.critic_head(critic_feat_buf).view(-1, args.critic_mtp_horizon, sf_dim)[:, 0]
            ).reshape(args.num_steps, args.num_envs, sf_dim)
            next_transition_values = psi_next @ w_r
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
            # ---- SUCCESSOR-FEATURE target ------------------------------------------
            # critic_feat_buf is the SAME full-batch trunk forward that produced psi_cur,
            # reused rather than recomputed so phi and psi are guaranteed to describe one
            # representation. A second, independently built forward is exactly how a
            # one-iteration skew between the features and their own successor target hides.
            trunk_phi = (
                critic_feat_buf.reshape(args.num_steps, args.num_envs, -1)
                if args.phi_trunk
                else None
            )
            phi = phi_features(obs, actions, trunk_phi)       # (T, B, sf_dim)

            # Vector-valued TD(lambda), element-wise on phi, mirroring the reward GAE above.
            # THE TWO MASKS ARE NOT THE SAME MASK. They differ exactly at a TRUNCATION
            # (term=0, valid=1, boundary=1): bootstrap through it, but CUT the lambda trace.
            # Collapsing them would let the next episode's discounted phi-sum bleed backward
            # into this episode's target from a fresh, near-zero-velocity reset state --
            # ~1.6% of samples corrupted by roughly the reset-vs-running value gap.
            # Written in residual space so the rollout tail (E_T = 0) is handled for free.
            sf_residual = torch.zeros_like(phi)
            last_sf = torch.zeros_like(phi[0])
            for t in reversed(range(args.num_steps)):
                boot = ((1.0 - transition_terminations[t]) * transition_valids[t]).unsqueeze(-1)
                cont = (1.0 - transition_boundaries[t]).unsqueeze(-1)
                delta_sf = phi[t] + args.gamma * boot * psi_next[t] - psi_cur[t]
                last_sf = delta_sf + args.gamma * args.gae_lambda * cont * last_sf
                sf_residual[t] = last_sf
            sf_target = sf_residual + psi_cur                 # Lambda_t

            # Closed-form w_r on IMMEDIATE reward -- no bootstrapping, no multi-step. This
            # is the only scalar regression anywhere in the design. Solved after the GAE so
            # values[] and next_transition_values[] both use the same (previous) w_r and the
            # advantage stays self-consistent.
            flat_phi = phi.reshape(-1, sf_dim)
            flat_rew = rewards.reshape(-1)
            w_r = solve_reward_probe(flat_phi, flat_rew, args.sf_ridge)
            reward_resid = flat_rew - flat_phi @ w_r
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

            # Per-dimension EMA standardization of the target (see setup for why).
            # The 1/count warmup is NOT cosmetic. e's own scale moves fast for the first
            # few dozen iterations while the obs normalizer settles, but 1/sf_target_ema = 100
            # iterations is 3.3M steps of lag -- the standardization would spend a third of
            # the run tracking a distribution phi left behind. rate = 1/count is an
            # exact running mean early (and an exact init at count == 1, replacing a
            # separate first-iteration branch) and decays into the plain EMA once the
            # normalizer settles.
            sf_stat_count += 1
            sf_rate = max(args.sf_target_ema, 1.0 / sf_stat_count)
            flat_tgt = sf_target.reshape(-1, sf_dim)
            tgt_mean, tgt_std = flat_tgt.mean(0), flat_tgt.std(0).clamp_min(1e-6)
            sf_mean = sf_mean + sf_rate * (tgt_mean - sf_mean)
            sf_std = sf_std + sf_rate * (tgt_std - sf_std)

            # MTP: horizon h regresses Lambda_{t+h}. Masks are the base's, verbatim -- a
            # future target is valid only when no reset boundary lies between source and
            # target state and it stays inside the rollout.
            mtp = args.critic_mtp_horizon
            sf_mtp = sf_target.new_zeros((args.num_steps, args.num_envs, mtp, sf_dim))
            return_mtp_mask = torch.zeros(
                (args.num_steps, args.num_envs, mtp), dtype=torch.bool, device=device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones((valid_len, args.num_envs), dtype=torch.bool, device=device)
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                sf_mtp[:valid_len, :, h] = sf_target[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            sf_mtp = (sf_mtp - sf_mean) / sf_std              # head regresses standardized units

            # ---- three-way EV gate: the falsifier, computed on an UNRIGGED target ----
            # The usual EV target `b_returns = advantages + values` CONTAINS the critic's
            # own bootstrapped values, and at dt=0.05 the errors are strongly
            # state-autocorrelated -- a critic scores well against a target built from
            # itself. Every predictor below is scored against the same truncated
            # Monte-Carlo discounted return instead.
            #
            # `avail[t]` counts reward terms actually present in mc_ret[t]: it resets at a
            # reset boundary and at the rollout tail, so masking on avail >= mc_window
            # removes BOTH the boundary bias and the tail bias in one condition.
            mc_ret = torch.zeros_like(rewards)
            mc_avail = torch.zeros_like(rewards)
            mc_resid = torch.zeros_like(rewards)          # discounted PROBE residual
            mc_run = torch.zeros_like(rewards[0])
            avail_run = torch.zeros_like(rewards[0])
            resid_run = torch.zeros_like(rewards[0])
            resid_tb2 = reward_resid.reshape(args.num_steps, args.num_envs)
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                mc_run = rewards[t] + args.gamma * cont * mc_run
                avail_run = 1.0 + cont * avail_run
                resid_run = resid_tb2[t] + args.gamma * cont * resid_run
                mc_ret[t] = mc_run
                mc_avail[t] = avail_run
                mc_resid[t] = resid_run
            mc_mask = (mc_avail >= args.mc_window).reshape(-1)
            n_mc = int(mc_mask.sum().item())
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
                #     INSTANTANEOUS state. Not a ceiling on psi: psi reads the trunk, not
                #     s_t, and accumulates phi over time, so (2) > (3) is expected and is
                #     in fact the successor-feature construction earning its keep.
                #     THIS IS THE SLOT v1/v2 USED FOR gate/ev_latent_cap (the same probe on
                #     e). Comparing that curve against this one is the cleanest read on
                #     what the encoder adds to a one-step linear readout -- but note it is
                #     only a ONE-STEP read and says nothing about the occupancy structure
                #     e contributes to Lambda, which is what this whole run is testing.
                s_mc = torch.cat([obs.reshape(flat_obs_shape)[mc_mask], ones_mc], dim=-1)
                ev_obs_probe = ev_score(
                    s_mc @ solve_reward_probe(s_mc, flat_mc, args.sf_ridge), flat_mc
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
                # comparison -- (3) << (1) means the reward BASIS is the bottleneck and no
                # amount of psi quality can rescue this. (2) << (3) with (3) ~ (1) instead
                # points at the SF machinery (recursion, standardization).
            else:
                ev_trunk_probe = ev_obs_probe = ev_sf = value_err_frac = float("nan")

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # Forward-prediction targets. DELTA, not s': under NormalizeObservation s is O(1)
        # while the per-step change is far smaller, so regressing s' directly would spend
        # most of the head's capacity re-emitting its own input. Masked on `valids`, which
        # is 0 exactly where no real successor exists (a boundary with no final_observation);
        # at every other boundary next_obses holds the TRUE successor, which is real physics
        # and must be kept.
        b_dyn_target = (next_obses - obs).reshape(-1, obs_dim)
        b_dyn_mask = transition_valids.reshape(-1)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_sf_target = sf_mtp.reshape(-1, args.critic_mtp_horizon, sf_dim)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, args, device)
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

                _, _, newlogprob, entropy, value_sf, critic_feat = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], args, device)
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

                # SUCCESSOR-FEATURE value loss: per-horizon masked MSE against the
                # standardized OCCUPANCY lambda-return, summed over valid horizons per row.
                # No scalar-return regression anywhere at the default sf_alpha=1.
                # ---- FORWARD-PREDICTION AUXILIARY --------------------------------------
                # Computed on the SAME minibatch and the SAME trunk forward as the policy
                # and value losses, so the representation is shaped by the identical data
                # distribution the heads are fit on.
                mb_dyn_mask = b_dyn_mask[mb_inds]
                dyn_pred = agent.dyn_head(torch.cat([critic_feat, b_actions[mb_inds]], dim=-1))
                dyn_err = (dyn_pred - b_dyn_target[mb_inds]).pow(2).mean(-1)
                dyn_loss = (dyn_err * mb_dyn_mask).sum() / mb_dyn_mask.sum().clamp_min(1.0)

                sf_tgt = b_sf_target[mb_inds]
                sf_err = value_sf - sf_tgt                                    # (B, mtp, sf_dim)
                value_mask = b_target_mask[mb_inds].to(sf_err.dtype)          # (B, mtp)
                # BLOCK-WEIGHTED, not a flat mean over sf_dim. A flat mean over the widened
                # basis would silently rescale the ONLY block w_r reads for the velocity term:
                # the raw-obs block falls from 17/30 = 57% of the value loss to 17/94 = 18%,
                # a 3.1x cut in the reward-relevant gradient that has nothing to do with the
                # hypothesis under test and would confound the whole run. Keeping the parent's
                # blocks on their own mean preserves their absolute weight exactly, and the
                # trunk block is ADDED at phi_trunk_loss_coef rather than diluting them.
                if args.phi_trunk:
                    H_phi = args.hidden
                    e_trunk = sf_err[..., :H_phi].pow(2).mean(-1)             # (B, mtp)
                    e_base = sf_err[..., H_phi:].pow(2).mean(-1)              # (B, mtp)
                    per_row = e_base + args.phi_trunk_loss_coef * e_trunk
                    # kept as tensors, NOT float() -- a .item() here would force a
                    # device sync on every one of the ~320 minibatches per iteration.
                    sf_loss_trunk = (e_trunk.detach() * value_mask).sum(-1).mean()
                    sf_loss_base = (e_base.detach() * value_mask).sum(-1).mean()
                else:
                    per_row = sf_err.pow(2).mean(-1)
                    sf_loss_trunk = torch.zeros((), device=sf_err.device)
                    sf_loss_base = (per_row.detach() * value_mask).sum(-1).mean()
                v_loss = (per_row * value_mask).sum(dim=-1).mean()
                if args.sf_alpha < 1.0:
                    # Reward-direction term. Its target w_r.Lambda still comes from the
                    # VECTOR lambda-return, so the bootstrap stays in occupancy space and the
                    # thesis is intact; sf_alpha only decides how much of psi's capacity is
                    # spent on the one direction that is actually read out.
                    scalar_err = (sf_err @ (w_r * sf_std)).pow(2)             # (B, mtp)
                    v_loss = args.sf_alpha * v_loss + (1.0 - args.sf_alpha) * (
                        scalar_err * value_mask
                    ).sum(dim=-1).mean()
                # Per-horizon psi MSE (last minibatch of the last epoch is what gets logged).
                sf_per_h_mse = (per_row.detach() * value_mask).sum(0) / value_mask.sum(
                    0
                ).clamp_min(1)

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
                    # THIRD branch, extending the file's actor/critic split to the auxiliary.
                    # Measured on the parent, critic_grad_norm (pre-clip) runs 0.30 -> 1.01
                    # against critic_grad_clip=0.25: the clip is saturated essentially always,
                    # so it is a normalizer, not a safety net. Folding dyn_loss in here would
                    # therefore not add to the critic step, it would ROTATE a fixed-norm budget
                    # -- the value share becoming 0.25*g_v/||g_v + g_dyn|| -- and with the two
                    # losses the same order of magnitude that halves the critic's effective LR.
                    # A regression would then read as "the representation hypothesis is false"
                    # when the real cause was a critic trained at half speed. Own clip, own log.
                    (args.dyn_coef * dyn_loss).backward(retain_graph=True)
                    dyn_gn = nn.utils.clip_grad_norm_(dyn_params, args.dyn_grad_clip)
                    dyn_grads = [(p, p.grad.detach().clone()) for p in dyn_params if p.grad is not None]
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
                    for p, g in value_grads + dyn_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = (
                        pg_loss
                        - ent_coef_eff * entropy_loss
                        + v_loss * args.vf_coef
                        + args.dyn_coef * dyn_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    # Single global clip: the auxiliary shares it by construction here. This
                    # branch is off by default precisely because that coupling is what the
                    # dual/triple-clip design exists to avoid.
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    dyn_gn = critic_gn
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # Aggregate R^2 of the forward model, in-sample and MONITORING ONLY -- nothing
        # gates on it, so there is no incentive to flatter it. If this sits near 0 the
        # auxiliary is not learning and the trunk is getting noise; if it sits near 1 the
        # task is too easy to be shaping the representation and dyn_coef is doing nothing.
        with torch.no_grad():
            dyn_tgt_var = float(b_dyn_target[b_dyn_mask > 0.5].var(0).mean())
        writer.add_scalar("dyn/loss", dyn_loss.item(), global_step)
        writer.add_scalar("dyn/r2", 1.0 - dyn_loss.item() / max(dyn_tgt_var, 1e-12), global_step)
        writer.add_scalar("dyn/target_var", dyn_tgt_var, global_step)
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
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        # Pre-clip norm of the AUXILIARY branch. Read against dyn_grad_clip: if this sits
        # far above it the auxiliary is clip-bound and dyn_coef is inert (the clip, not the
        # coefficient, is setting its magnitude); far below and dyn_coef is the live knob.
        writer.add_scalar("losses/dyn_grad_norm", float(dyn_gn), global_step)
        writer.add_scalar("sf/loss_base_block", sf_loss_base.item(), global_step)
        writer.add_scalar("sf/loss_trunk_block", sf_loss_trunk.item(), global_step)
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
        for h in range(args.critic_mtp_horizon):
            writer.add_scalar(f"sf/psi_mse_h{h}", sf_per_h_mse[h].item(), global_step)
        # ||Lambda|| per block against the tau ~= 20 prediction for a zero-mean,
        # unit-variance s block under gamma=0.99, lambda=0.95. The raw-state block's scale
        # is pinned by NormalizeObservation's running stats; watch it for normalizer drift.
        # (v1/v2 also logged sf/lambda_std_emb and sf/lambda_absmean_emb -- there is no
        # embedding block here, so those two tags are absent by construction, not missing.)
        # OFFSET by the prepended trunk block, or every one of these three would silently
        # report trunk statistics under an "obs"/"act" label.
        o0 = args.hidden if args.phi_trunk else 0
        if args.phi_trunk:
            writer.add_scalar("sf/lambda_std_trunk", sf_std[:o0].mean().item(), global_step)
        writer.add_scalar("sf/lambda_std_obs", sf_std[o0:o0 + obs_dim].mean().item(), global_step)
        writer.add_scalar("sf/lambda_std_act", sf_std[o0 + obs_dim:-1].mean().item(), global_step)
        writer.add_scalar(
            "sf/lambda_absmean_obs", sf_mean[o0:o0 + obs_dim].abs().mean().item(), global_step
        )
        writer.add_scalar("sf/w_r_norm", w_r.norm().item(), global_step)

        # ---- the falsifier: EV against truncated-MC returns, three predictors ----------
        writer.add_scalar("gate/ev_sf_online", ev_sf, global_step)          # (2) treatment
        writer.add_scalar("gate/ev_trunk_probe", ev_trunk_probe, global_step)  # (1) reference
        # (3) occupies v1/v2's gate/ev_latent_cap slot, on raw s instead of e -- the two
        # curves side by side are the one-step read on what the encoder was contributing.
        writer.add_scalar("gate/ev_obs_probe", ev_obs_probe, global_step)
        writer.add_scalar("gate/mc_frac", n_mc / (args.num_steps * args.num_envs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

# PPO + IterThink v24 beta + HYPERSPHERICAL DenseNet+MoE trunk (hypermoe_v8).
#
# KEY IDEA. The pure SimBaV2 hyperspherical trunk (simbav2_v1..v7) caps ~6-7k on
# HalfCheetah (v7 shared = 6752), ~3k under v24_beta's DenseNet+MoE trunk (~9785). The
# realization: the hypersphere is a CONDITIONING/normalization scheme (fixes gradient
# conditioning / plasticity, which on-policy PPO DOES suffer) but supplies NO inductive
# bias. v24 wins on FUNCTION CLASS: soft-MoE conditional computation + dense cross-depth
# feature reuse + shared coupling. Conditioning != expressiveness, so the two should
# STACK, not compete. v8 applies SimBaV2's hyperspherical normalization TO v24's
# DenseNet+MoE trunk (--trunk hypermoe): v24's expressive mechanism + SimBaV2 conditioning.
#
# HYPERSPHERICAL DenseNet+MoE trunk (v24's ThinkTrunk ported onto the unit sphere):
#   HyperEmbedder entry -> K HyperThinkBlocks (dense reach-back) -> embedder-style out_proj.
#   HyperThinkBlock = v24's ThinkBlock on the sphere:
#     in_proj:   l2norm(concat of prior unit-norm block outputs) -> HyperDense -> Scaler -> l2norm
#     residual:  spherical convex gate  x_in = l2norm( g*x + (1-g)*x0 ),  g=sigmoid(gate) per-chan
#     dense:     HyperMLP(x_in)                              (inverted bottleneck, unit-norm out)
#     soft-MoE:  softmax(Linear(x_in)) over n_experts (no top-k); each expert a HyperMLP;
#                d_moe = l2norm( sum_e w_e * expert_e(x_in) )   (mix unit-norm experts -> sphere)
#     combine:   out = l2norm( x_in + alpha_dense*(d_dense - x_in) + alpha_moe*(d_moe - x_in) )
#                two per-channel learnable Scaler alphas (SimBaV2 LERP, budget 1/(2K+1)).
#   The MoE GATE stays a plain Linear (off-sphere logits): routing logits must be unbounded
#   to specialize; a HyperDense gate (unit weights . unit input in [-1,1]) collapses to ~uniform.
#   Everything else on the sphere; all HyperDense weights unit-projected each optimizer step.
#   --trunk simbav2 recovers the pure SimBaV2 backbone (the A/B control).
#
# SimBaV2 backbone (ported 1:1 from scale_rl/agents/simbaV2, JAX->PyTorch):
#   HyperEmbedder:  x -> append const c_shift -> l2norm -> HyperDense -> Scaler -> l2norm
#   HyperLERPBlock: x <- l2norm( res + alpha*(HyperMLP(res) - res) )   [spherical LERP residual]
#   HyperMLP:       w1(h->4h) -> Scaler -> ReLU+eps -> w2(4h->h) -> l2norm   [inverted bottleneck]
#   HyperDense:     bias-free Linear, orthogonal init, weights PROJECTED to the unit
#                   sphere (per output unit) at init AND after every optimizer step.
#   Scaler:         learnable per-channel gain (reparam: init via scaler/forward split).
#   Asymmetric, SEPARATE actor/critic backbones (SimBaV2 default): actor H=128/1 block,
#   critic H=512/2 blocks (big-critic/small-actor). c_shift=3, scaler_init=scaler_scale
#   =sqrt(2/H), alpha_init=1/(blocks+1), alpha_scale=1/sqrt(H).
#
# OPTIMIZER (matches the SimBaV2 paper/code, NOT v24's): plain Adam, linear LR
# 1e-4 -> 5e-5, and NO weight decay -- the per-step unit-sphere weight projection
# replaces decay (decay-to-zero would fight projection-to-unit-norm). `--weight-decay`
# is exposed (default 0) to switch to AdamW if ever wanted.
#
# KEPT (our validated action solution + PPO machinery, byte-identical to v24_beta):
#   unimodal Beta actor, HL-Gauss symlog distributional critic (511 bins, +-10),
#   rankgauss advantage, clip-higher (0.2/0.28), target_kl 0.03, dual-backward
#   decoupled grad clip (0.25/0.25), 16 envs x 2048 steps, 8M timesteps.
# This isolates the SimBaV2 BACKBONE as the only lever vs the v24 bar (~9785).
# v2 candidates (deferred, kept fixed here to avoid confounds): SimBaV2's own
# hyperspherical Normal-tanh / categorical heads, RSNorm input, reward scaling.
#
# --- inherited v24_beta notes (HL-Gauss symlog critic) ---
# PPO + IterThink v24 (ACTION-DISTRIBUTION TOGGLE: dreamer4-faithful). From v21.
#
# SOFT MAX-ENTROPY add-on (--auto-entropy, gaussian only). This borrows SAC's
# tanh-squashed log-prob, target-entropy heuristic, and temperature dual, but keeps
# the PPO critic on the RAW reward return. Entropy enters the actor two ways:
#   (1) a current-state squashed-entropy actor bonus, -alpha * log pi_sq(a|s);
#   (2) a policy-only soft GAE whose one-step bootstrap adds alpha * H_sq(s_{t+1})
#       using the rollout/bootstrapped squashed log-prob sample.
# The critic target is deliberately entropy-free so the fixed support remains
# calibrated. In this variant the target is scalar-return HL-Gauss rather than
# v24's distributional lambda-return recursion.
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
# v24's distributional λ-return to scalar-return HL-Gauss projection; only the
# policy advantage transforms are selected by `adv_transform`. sigma(s) is the
# std of the OLD rollout Z(s_t), floored at `sigma_floor_bins` bins.
import os
import random
import time
import math
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

from cleanrl.shared.hl_gauss import HLGaussSupport, symlog, symexp

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def value_support_bounds(args):
    """Support bounds in the coordinate used for categorical bins."""
    if not args.value_symlog:
        return args.v_min, args.v_max
    bounds = torch.tensor([args.v_min, args.v_max], dtype=torch.float32)
    return symlog(bounds).tolist()


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
    learning_rate: float = 1e-4           # legacy single-lr (unused by the decoupled optimizer below)
    learning_rate_end: float = 5e-5
    weight_decay: float = 0.0             # SimBaV2 uses NONE (unit-sphere projection replaces it); >0 -> AdamW
    # DECOUPLED actor/critic learning rates. The SimBaV2 actor needs a high PPO lr to
    # move (under-driven at 1e-4 -> 1203@1.6M), but the distributional hyperspherical
    # critic destabilizes there (explained_variance 0.88 -> ~0 at 3e-4). So: actor hot,
    # critic cool, each on its own linear anneal. Set equal to recover a single lr.
    actor_lr: float = 3e-4
    actor_lr_end: float = 0.0
    critic_lr: float = 1e-4
    critic_lr_end: float = 5e-5
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

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ≈ N(0, tau^2), sharp at 0
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 2.0  # Dreamer4 / hl_gauss_pytorch default

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "rankgauss"

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

    # SimBaV2 backbone geometry. share_backbone=True (inherited Args) -> ONE shared trunk
    # (shared_hidden/shared_blocks) feeds both heads with decoupled dual-clip (v24's v21
    # winner); False -> SEPARATE asymmetric actor/critic trunks (SimBaV2 default).
    shared_hidden: int = 512
    shared_blocks: int = 2
    actor_hidden: int = 128
    actor_blocks: int = 1
    critic_hidden: int = 512
    critic_blocks: int = 2
    c_shift: float = 3.0       # constant feature appended in the HyperEmbedder
    critic_head: str = "linear"  # "linear" (plain) | "hyperspherical" (SimBaV2 on-sphere value head)

    # Trunk selector. "hypermoe" (v8 default) = hyperspherical DenseNet+soft-MoE (v24's
    # ThinkTrunk on the sphere); "simbav2" = pure SimBaV2 residual-MLP backbone (the A/B
    # control). hypermoe uses shared_hidden/shared_blocks as H/K (default 64/3 = v24's
    # winning geometry) plus n_experts/expansion below.
    trunk: str = "hypermoe"
    n_experts: int = 16          # soft-MoE expert count (hypermoe only)
    expansion: int = 2           # inverted-bottleneck expansion in each HyperMLP branch
    gate_temp: float = 0.0       # MoE gate logit temperature; <=0 => sqrt(H) (restores v24
                                 #   routing on unit-norm input; raw logits would be ~uniform)

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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ============================ SimBaV2 hyperspherical backbone ============================
# Ported 1:1 from scale_rl/agents/simbaV2/simbaV2_layer.py (flax -> torch). Every
# activation and every weight row lives on the unit hypersphere.

def l2normalize(x, dim=-1, eps=1e-8):
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


class HyperDense(nn.Linear):
    """Bias-free linear, orthogonal init. Its weight rows (one per output unit) are
    projected onto the unit sphere at init and after every optimizer step
    (Agent.project_hyperdense_weights), matching the reference l2normalize_network."""

    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.weight, gain=1.0)
        with torch.no_grad():
            self.weight.copy_(l2normalize(self.weight, dim=1))


class Scaler(nn.Module):
    """Learnable per-channel gain. Param is initialized to `scale`; forward multiplies
    by `init/scale`, so the effective initial gain is `init` (SimBaV2's reparam)."""

    def __init__(self, dim, init=1.0, scale=1.0):
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), float(scale)))
        self.forward_scaler = init / scale

    def forward(self, x):
        return self.scaler * self.forward_scaler * x


class HyperMLP(nn.Module):
    """w1 (in->hidden) -> Scaler -> ReLU+eps -> w2 (hidden->out) -> l2normalize."""

    def __init__(self, in_dim, hidden_dim, out_dim, scaler_init, scaler_scale, eps=1e-8):
        super().__init__()
        self.w1 = HyperDense(in_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(hidden_dim, out_dim)
        self.eps = eps

    def forward(self, x):
        x = self.w1(x)
        x = self.scaler(x)
        x = F.relu(x) + self.eps   # eps avoids the zero vector before l2normalize
        x = self.w2(x)
        return l2normalize(x)


class HyperEmbedder(nn.Module):
    """Append a constant c_shift feature, l2normalize, HyperDense, Scaler, l2normalize."""

    def __init__(self, in_dim, hidden_dim, scaler_init, scaler_scale, c_shift):
        super().__init__()
        self.c_shift = c_shift
        self.w = HyperDense(in_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x):
        const = x.new_full(x.shape[:-1] + (1,), self.c_shift)
        x = torch.cat([x, const], dim=-1)
        x = l2normalize(x)
        x = self.w(x)
        x = self.scaler(x)
        return l2normalize(x)


class HyperLERPBlock(nn.Module):
    """Spherical-LERP residual: x <- l2normalize( res + alpha*(MLP(res) - res) )."""

    def __init__(self, hidden_dim, scaler_init, scaler_scale, alpha_init, alpha_scale, expansion=4):
        super().__init__()
        self.mlp = HyperMLP(
            hidden_dim,
            hidden_dim * expansion,
            hidden_dim,
            scaler_init / math.sqrt(expansion),
            scaler_scale / math.sqrt(expansion),
        )
        self.alpha_scaler = Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x):
        res = x
        x = self.mlp(x)
        x = res + self.alpha_scaler(x - res)
        return l2normalize(x)


class SimbaV2Backbone(nn.Module):
    """SimBaV2 trunk: HyperEmbedder -> num_blocks x HyperLERPBlock. Output is unit-norm.
    Drop-in for ThinkTrunk: forward maps (B, in_dim) -> (B, hidden_dim)."""

    def __init__(self, in_dim, hidden_dim, num_blocks, c_shift):
        super().__init__()
        scaler = math.sqrt(2.0 / hidden_dim)          # scaler_init = scaler_scale = sqrt(2/H)
        alpha_init = 1.0 / (num_blocks + 1)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)
        self.embedder = HyperEmbedder(in_dim, hidden_dim, scaler, scaler, c_shift)
        self.blocks = nn.ModuleList(
            [
                HyperLERPBlock(hidden_dim, scaler, scaler, alpha_init, alpha_scale)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        y = self.embedder(x)
        for block in self.blocks:
            y = block(y)
        return y


# ============== Hyperspherical DenseNet + soft-MoE trunk (v8, hypermoe) ==============
# v24_beta's ThinkTrunk/ThinkBlock ported onto the unit sphere using the SimBaV2
# primitives above. Keeps v24's expressive mechanism (dense cross-depth reach-back +
# soft-MoE conditional computation + convex residual gate) and adds the hypersphere's
# conditioning (l2norm everywhere, unit-projected HyperDense weights, learnable Scalers).

class HyperThinkBlock(nn.Module):
    """v24 ThinkBlock on the sphere: spherical convex residual mix of the projected
    dense input and the entry embedding x0, then parallel hyperspherical dense + soft-MoE
    branches combined by per-channel learnable alpha LERPs, renormalized to the sphere."""

    def __init__(self, in_dim, H, n_experts, scaler_init, scaler_scale,
                 alpha_init, alpha_scale, expansion=2, gate_temp=None):
        super().__init__()
        self.n_experts = n_experts
        # Gate logit temperature. v24's gate reads RMSNorm(x_in) (RMS~1, L2~sqrt(H)); our
        # x_in is L2-UNIT (RMS~1/sqrt(H)), so without rescaling the logits are sqrt(H)x too
        # small and the softmax collapses to ~uniform (entropy~ln E) => the soft-MoE
        # degenerates into a uniform expert AVERAGE with no routing (the v8_v1 bug:
        # measured gate entropy 2.758 vs ln16=2.773). Scaling logits by sqrt(H) restores
        # v24's routing sharpness exactly (entropy 2.04 == v24's RMS-input 2.05).
        self.gate_temp = math.sqrt(H) if gate_temp is None else gate_temp
        # in_proj: embedder-style (input concat is already unit-norm per-feature; the
        # forward l2norms the concat, then HyperDense -> Scaler -> l2norm).
        self.in_proj = HyperDense(in_dim, H)
        self.in_scaler = Scaler(H, scaler_init, scaler_scale)
        # Convex residual gate (per channel): spherical LERP between x and x0. init +4
        # => g ~ 0.982 => x_in ~ x at start (matches v24's resid_gate init).
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))
        # Dense branch: inverted-bottleneck HyperMLP (unit-norm output).
        self.dense = HyperMLP(
            H, H * expansion, H,
            scaler_init / math.sqrt(expansion), scaler_scale / math.sqrt(expansion),
        )
        self.dense_alpha = Scaler(H, alpha_init, alpha_scale)
        # Soft-MoE branch (softmax over ALL experts, no top-k). GATE is a plain Linear:
        # routing logits must be unbounded to specialize -- a HyperDense gate (unit
        # weight . unit input in [-1,1]) softmaxes to ~uniform and kills routing.
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([
            HyperMLP(
                H, H * expansion, H,
                scaler_init / math.sqrt(expansion), scaler_scale / math.sqrt(expansion),
            )
            for _ in range(n_experts)
        ])
        self.moe_alpha = Scaler(H, alpha_init, alpha_scale)

    def forward(self, cat_feats, x0):
        x = l2normalize(cat_feats)
        x = self.in_scaler(self.in_proj(x))
        x = l2normalize(x)                                       # (B,H) unit-norm
        g = torch.sigmoid(self.resid_gate)
        x_in = l2normalize(g * x + (1.0 - g) * x0)               # spherical convex residual
        d_dense = self.dense(x_in)                               # unit-norm
        weights = torch.softmax(self.gate(x_in) * self.gate_temp, dim=-1)  # (B,E) sqrt(H) temp restores routing
        all_out = torch.stack([e(x_in) for e in self.experts], dim=1)  # (B,E,H) unit-norm
        d_moe = l2normalize((weights.unsqueeze(-1) * all_out).sum(dim=1))  # mix -> sphere
        # SimBaV2 LERP combine: two per-channel alpha steps from x_in, then back to sphere.
        out = x_in + self.dense_alpha(d_dense - x_in) + self.moe_alpha(d_moe - x_in)
        return l2normalize(out)


class HyperThinkTrunk(nn.Module):
    """Hyperspherical DenseNet trunk: HyperEmbedder entry -> K HyperThinkBlocks with
    dense cross-depth reach-back (concat of unit-norm block outputs, l2normed before each
    block), embedder-style out_proj. Drop-in for SimbaV2Backbone: (B,in_dim)->(B,H)."""

    def __init__(self, in_dim, hidden_dim, num_blocks, c_shift, n_experts, expansion=2, gate_temp=None):
        super().__init__()
        H = hidden_dim
        scaler = math.sqrt(2.0 / H)                              # scaler_init = scaler_scale
        # Two branch-deltas per block => spread the residual budget over 2K deltas.
        alpha_init = 1.0 / (2 * num_blocks + 1)
        alpha_scale = 1.0 / math.sqrt(H)
        self.entry = HyperEmbedder(in_dim, H, scaler, scaler, c_shift)
        self.blocks = nn.ModuleList()
        for k in range(num_blocks):
            block_in_dim = H * (k + 1)                           # DenseNet reach-back
            self.blocks.append(
                HyperThinkBlock(block_in_dim, H, n_experts, scaler, scaler,
                                alpha_init, alpha_scale, expansion, gate_temp)
            )
        cat_dim = H * (num_blocks + 1)
        self.out_proj = HyperDense(cat_dim, H)
        self.out_scaler = Scaler(H, scaler, scaler)

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        out = l2normalize(torch.cat(feats, dim=-1))
        out = self.out_scaler(self.out_proj(out))
        return l2normalize(out)


class HyperCategoricalHead(nn.Module):
    """SimBaV2 hyperspherical categorical value head (simbaV2_layer.HyperCategoricalValue):
    HyperDense -> Scaler(1,1) -> HyperDense + bias -> logits. Keeps the critic on the unit
    sphere THROUGH the readout (the two HyperDense weights are projected each step), so the
    critic can run at a hot lr without the EV instability of a plain Linear head on a
    unit-norm trunk. Outputs raw logits; our HL-Gauss symlog target/support are unchanged."""

    def __init__(self, hidden_dim, num_bins):
        super().__init__()
        self.w1 = HyperDense(hidden_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, init=1.0, scale=1.0)
        self.w2 = HyperDense(hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_bins))

    def forward(self, x):
        x = self.w1(x)
        x = self.scaler(x)
        return self.w2(x) + self.bias


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.share_backbone = args.share_backbone

        def make_trunk(H, blocks):
            if args.trunk == "hypermoe":
                gate_temp = args.gate_temp if args.gate_temp > 0 else None
                return HyperThinkTrunk(obs_dim, H, blocks, args.c_shift, args.n_experts,
                                       args.expansion, gate_temp)
            return SimbaV2Backbone(obs_dim, H, blocks, args.c_shift)

        if self.share_backbone:
            # ONE hyperspherical trunk feeds both heads (v24's v21 winner). The decoupled
            # dual-backward clip keeps the value gradient from swamping the policy features.
            actor_H = critic_H = args.shared_hidden
            self.trunk = make_trunk(args.shared_hidden, args.shared_blocks)
        else:
            # SEPARATE asymmetric actor/critic backbones (SimBaV2 default geometry).
            actor_H = args.actor_hidden
            critic_H = args.critic_hidden
            self.actor_trunk = make_trunk(actor_H, args.actor_blocks)
            self.critic_trunk = make_trunk(critic_H, args.critic_blocks)
        # Categorical critic with a PEAKED init: small weight + Gaussian-logit
        # bias so the initial value distribution starts as a sharp zero-return
        # prior instead of a high-variance uniform distribution. The bias lives
        # in the categorical coordinate, which is symlog-space when enabled.
        if args.critic_head == "hyperspherical":
            self.critic_head = HyperCategoricalHead(critic_H, args.num_bins)
        else:
            self.critic_head = layer_init(nn.Linear(critic_H, args.num_bins), std=0.1)
        with torch.no_grad():
            support_min, support_max = value_support_bounds(args)
            edge_width = (support_max - support_min) / args.num_bins
            z = torch.linspace(
                support_min + 0.5 * edge_width,
                support_max - 0.5 * edge_width,
                args.num_bins,
            )
            self.critic_head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)
        # v24: action distribution. Both parameterizations are dreamer4-faithful;
        # the Gaussian path is tanh-squashed like SAC but uses log-variance, not log_std.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            # dreamer4 Gaussian: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(actor_H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(actor_H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # dreamer4 unimodal Beta: two concentration heads, alpha,beta = 1 + softplus.
            self.actor_alpha_head = layer_init(nn.Linear(actor_H, act_dim), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(actor_H, act_dim), std=0.01)
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
        # Returns value LOGITS (B, num_bins); caller converts via support.
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

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
        return action, z, log_prob, entropy, value_logits

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

    @torch.no_grad()
    def project_hyperdense_weights(self):
        # SimBaV2: after every optimizer step, project each HyperDense weight back onto
        # the unit hypersphere (per output unit = per row of the (out,in) weight),
        # matching the reference l2normalize_network on `hyper_dense` kernels. The plain
        # nn.Linear heads (critic/actor) are NOT projected -- they train normally.
        for m in self.modules():
            if isinstance(m, HyperDense):
                m.weight.copy_(l2normalize(m.weight, dim=1))


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
    agent.project_hyperdense_weights()  # SimBaV2: unit-sphere weights at init
    # Disjoint actor/critic param partition (fully separate backbones, no sharing).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    # SimBaV2 optimizer: plain Adam + per-step unit-sphere projection (NO weight decay).
    # weight_decay>0 -> AdamW (off the paper recipe).
    opt_cls = optim.AdamW if args.weight_decay > 0 else optim.Adam
    opt_kwargs = {"eps": 1e-5}
    if args.weight_decay > 0:
        opt_kwargs["weight_decay"] = args.weight_decay
    if args.share_backbone:
        # The shared trunk is in BOTH actor_params and critic_params, so it cannot go in
        # two optimizer groups. ONE group / one lr (actor_lr); the decoupled dual-backward
        # CLIP still separates the policy/value gradients on the shared trunk.
        optimizer = opt_cls(agent.parameters(), lr=args.actor_lr, **opt_kwargs)
    else:
        # Separate backbones -> disjoint partition -> DECOUPLED per-group lr (actor/critic).
        optimizer = opt_cls(
            [
                {"params": actor_params, "lr": args.actor_lr},
                {"params": critic_params, "lr": args.critic_lr},
            ],
            **opt_kwargs,
        )

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
    hl_support = HLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )
    support = hl_support.support                       # (num_bins,) bin centers
    bin_width = hl_support.bin_width
    if args.value_symlog:
        raw_support = symexp(support)
        raw_edges = symexp(hl_support.edges)
        raw_bin_widths = raw_edges[1:] - raw_edges[:-1]
    else:
        raw_support = support
        raw_bin_widths = torch.full_like(support, bin_width)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            # Linear schedule per group: group 0 (actor / or all params when shared),
            # group 1 (critic, separate-backbone only).
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = args.actor_lr_end + frac * (args.actor_lr - args.actor_lr_end)
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = args.critic_lr_end + frac * (args.critic_lr - args.critic_lr_end)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                values[step] = hl_support.to_scalar(value_logits)
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
                _, _, boot_logprob, _, boot_logits = agent.get_action_and_value(next_obs)
                bootstrap_logits = boot_logits
                bootstrap_probs = torch.softmax(bootstrap_logits, dim=-1)       # (B, n) = Z(s_T)
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                bootstrap_logits = agent.get_value(next_obs)
                bootstrap_probs = torch.softmax(bootstrap_logits, dim=-1)       # (B, n) = Z(s_T)
                next_value_bonus = None
            next_value = hl_support.to_scalar(bootstrap_logits).reshape(1, -1)
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
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
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). Unbiased (state-dependent baseline); its large
            # magnitude is harmless — rankgauss at the optim step rank-normalizes it while
            # PRESERVING the entropy-induced reordering (the actual exploration signal).
            if auto_alpha:
                policy_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * (nextvalues + next_value_bonus[t]) * nextnonterminal - values[t]
                    policy_adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            else:
                policy_adv = advantages
            # Critic target: Dreamer4-style scalar-return HL-Gauss. GAE computes
            # the scalar λ-return; the value encoder projects that scalar target
            # into a Gaussian-smoothed categorical distribution over fixed bins.
            target_probs = hl_support.project(returns)
            # Per-state return std sigma(s_t) in raw return units, matching the
            # GAE scale consumed by tanh_std. For symlog values, logits live in
            # symlog coordinates but PPO bootstraps from symexp(E[z]).
            sigma = (value_probs * (raw_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            if args.value_symlog:
                value_coord = symlog(values).clamp(hl_support.v_min, hl_support.v_max)
                floor_idx = ((value_coord - hl_support.v_min) / bin_width).floor().long().clamp(0, args.num_bins - 1)
                sigma_floor = args.sigma_floor_bins * raw_bin_widths[floor_idx]
            else:
                sigma_floor = args.sigma_floor_bins * bin_width
            sigma = torch.maximum(sigma, sigma_floor)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            returns_coord = symlog(returns) if args.value_symlog else returns
            cdf_frac = ((returns_coord.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
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
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits = agent.get_action_and_value(
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

                # HL-Gauss value loss: cross-entropy to the fixed scalar-return
                # projection target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

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
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                    agent.project_hyperdense_weights()  # SimBaV2: re-project to unit sphere
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    agent.project_hyperdense_weights()  # SimBaV2: re-project to unit sphere

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()

        writer.add_scalar("charts/actor_lr", optimizer.param_groups[0]["lr"], global_step)
        if len(optimizer.param_groups) > 1:
            writer.add_scalar("charts/critic_lr", optimizer.param_groups[1]["lr"], global_step)
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

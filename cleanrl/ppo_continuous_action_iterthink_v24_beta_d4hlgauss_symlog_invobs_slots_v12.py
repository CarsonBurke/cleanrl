# PPO + IterThink v24 beta d4-HL-Gauss symlog + inverted-attention slot transformer v12.
#
# WHY v12 (COMPILE-OPTIMAL, BEHAVIOR-IDENTICAL TO v11). v11 (nGPT hypersphere) is the
# clear winner of the inverted-slot line (~6200 @2.6M vs v10 ~5500 @5.9M) but runs ~37%
# slower per step than v10 -- entirely model-side overhead, not env. Profiling showed the
# trunk forward is ~2.5ms and FLAT in batch size (b=16 == b=1024): a textbook
# KERNEL-LAUNCH-BOUND signature (dozens of tiny justnorm/lerp/elementwise ops per
# forward; the arithmetic itself is free). The fix is fusion via torch.compile, which a
# microbenchmark confirms gives 2.2-2.5x on the trunk (fwd AND fwd+bwd) in DEFAULT mode.
# (reduce-overhead/CUDA-graphs BACKFIRED here -- capture overhead dominates tiny ops;
# bf16 was SLOWER -- cast overhead, no FLOP win; flash/flex are inapplicable -- inverted
# softmax axis + cross-token renorm can't map to those kernels, and 4x17 is too small to
# matter anyway.) v12 is a PURE-PERFORMANCE twin of v11 -- the math is identical; only
# two things change so the graph compiles cleanly:
#   1. The assign-entropy diagnostic is REMOVED from the hot forward (the in-forward
#      `no_grad` block + Python-int `_assign_count += 1` were forcing graph breaks /
#      recompiles). It is recomputed ONCE PER ITERATION in a separate eager pass
#      (InvertedSlotStack.assign_entropy) over a sample of obs -- same logged signal,
#      negligible cost, zero effect on training.
#   2. The trunk(s) are wrapped in torch.compile (default mode) and TF32 matmul is
#      enabled (torch.set_float32_matmul_precision("high")). --compile toggles it.
# Everything else -- the nGPT hypersphere recipe, inverted attention, eigen-lr residuals,
# SwiGLU, per-step weight normalization, the entire PPO/critic/advantage stack -- is
# byte-identical to v11. Gradient parity vs eager v11 is validated offline (the
# dual-backward + retain_graph through a compiled shared trunk is the one real risk).
# Base: ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_invobs_slots_v11.py.
#
# --- inherited v11 notes ---
#
# WHY v11 (nGPT: REPRESENTATION ON THE UNIT HYPERSPHERE). v11 = v10's exact
# inverted-slot architecture, but every hidden representation in the TRUNK lives on
# the unit hypersphere, per nGPT (Loshchilov et al. 2024, "nGPT: Normalized
# Transformer with Representation Learning on the Hypersphere", arXiv:2410.01131).
# nGPT reports 4-20x fewer steps to a target loss and removes LayerNorm/weight-decay
# entirely -- the hypersphere constraint IS the normalization. The interpretation that
# motivates trying it here: each transformer block becomes one step of variance-metric
# gradient descent on the sphere toward an input-conditioned target, with a LEARNED,
# PER-CHANNEL step size ("eigen learning rate"). For our setting the appeal is that
# the slow, mis-conditioned early learning that sank v7/v8/v9 (all CONDITIONING, not
# capacity, failures) is exactly what nGPT's bounded, well-conditioned sphere geometry
# is built to fix -- and v10's own keep/kill showed it merely TRACKS v6, so the lever
# left is conditioning/optimization, not more params.
#
# What changes (TRUNK ONLY -- tokenizer + the two inverted-slot stacks):
#   - All LayerNorm REMOVED. Slots and obs tokens are L2-normalized along H (justnorm)
#     so they sit on the unit sphere; the constraint replaces every norm layer.
#   - RESIDUALS -> normalized lerp with a learned per-channel eigen-lr (init 0.05):
#       slots = justnorm( slots + alpha_*(justnorm(sublayer(slots)) - slots) )
#     replacing v10's plain `slots += sublayer(slots)`. alpha_attn, alpha_mlp are
#     learned H-vectors (abs'd; nGPT's "controllable per-dim step toward the target").
#   - ATTENTION on the sphere: q,k are per-vector normalized along d_qk and scaled by a
#     learned sqk gain; the score scale flips from 1/sqrt(d_qk) (temper unbounded dots)
#     to *sqrt(d_qk) (restore range to bounded cosines). The INVERTED axis is UNCHANGED
#     -- softmax over slots, renorm over tokens. v stays full-rank/unnormalized (carries
#     info); the value & output projections are sphere-normalized by the weight step below.
#   - FFN -> SwiGLU with nGPT's learned suv input gain (replaces v10's GELU), at the same
#     mlp_mult=3 width so the v10 param-reallocation rationale is preserved.
#   - WEIGHT NORMALIZATION after every optimizer.step(): each trunk matrix is renormed
#     to unit rows/cols along its embedding dim (q,k,v,c_fc by input row; the output
#     projections by column; tokenizer value/pos by token vector). This keeps weights on
#     the sphere so the eigen-lr step sizes stay calibrated -- nGPT's replacement for
#     weight decay. Slot inits are justnormed at use (not stepped on the sphere).
#
# DELIBERATELY HELD FIXED (the project's architecture-only contract): the PPO/critic/
# advantage stack is byte-identical to v10/v6 -- separate actor/critic stacks, decoupled
# dual-backward grad clip, rankgauss advantage, clip-higher, HL-Gauss symlog
# distributional critic, beta action dist, target_kl. The actor/critic HEADS are NOT
# put on the sphere (their peaked-Gaussian critic-bias init and beta concentration heads
# are calibration we must not perturb); instead the on-sphere slot features are rescaled
# by sqrt(H) at readout so the head-input scale matches v10's old final-LayerNorm output
# (per-channel RMS ~= 1). nGPT's logit scale sz is the LM-head analog and is out of scope.
#
# This is a clean A/B isolating ONE thing: does putting v10's representation on the
# hypersphere fix the conditioning/underfitting? If v11 learns markedly faster early
# (the nGPT speedup signature) and/or lifts the asymptote, the lever was geometry.
# Base: ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_invobs_slots_v10.py.
#
# --- inherited v10 notes ---
#
# WHY v10 (PARAM REALLOCATION: Q/K -> FFN, at parity). Prior probes off v6 (the clean
# 8561 @8M HalfCheetah baseline) all REGRESSED: v7 (LayerScale throttle), v8 (nonlinear
# tokenizer residual, -23% by 2.6M), v9 (depth-mixing readout, -12% @1.3M). The naive
# read was "adding params hurts" -- but adversarial analysis showed those were
# CONDITIONING failures (a multiplicative residual throttle; a new in-series pathway;
# changed gradient fan-in), NOT capacity failures. The real signal is that v6 is
# UNDERFITTING and mis-allocated, not saturated.
#
#   THE BOTTLENECK IS WHERE THE PARAMS GO, NOT HOW MANY. Assigning 17 dense
#   proprioceptive tokens to 4 slots is a tiny relational problem, yet v6 spends ~75%
#   of each block on full H-dim multi-head Q/K/proj relational plumbing -- capacity a
#   17-D state with NO set/object structure cannot exploit. Meanwhile the winning
#   baseline (v24) is a DenseNet-MoE *MLP*: it wins on POINTWISE mixing capacity, the
#   exact axis attention starves. v10 reallocates, at ~parity (+3% params):
#     - Q/K collapse to a SINGLE LOW-RANK head (d_qk=8): just enough to compute the
#       assignment. Value and output proj stay full-rank (they carry the information).
#     - The freed budget goes into a WIDER pointwise FFN (mlp_mult 2 -> 3).
#   Everything else is byte-identical to v6 (inverted attention unchanged, plain
#   residuals, separate actor/critic stacks, flatten readout, all PPO machinery).
#
# This is also a clean FALSIFICATION: if shifting the budget toward the capacity axis
# that the MLP wins on STILL cannot approach v24, the deficit is the attention
# inductive-bias mismatch itself, and the transformer line should be retired -- with
# evidence, not vibes. (Caveat: single-seed effect sizes are within HalfCheetah seed
# spread; a v10 win/tie gets multi-seed confirmation before any claim vs v24.)
#
# WHY v6. v4 grafted the inverted-attention idea (Wu et al. 2024, "Inverted-Attention
# Transformers can Learn Object Representations: Insights from Slot Attention",
# openreview WgQZNoQ5AB) onto a stack of Parameter-Golf machinery borrowed from a
# DIFFERENT architecture (causal self-attention LM): initial-slot reinjection via a
# learned resid_mix, per-channel attn/mlp residual scales, zero-init output
# projections, a LIFO U-Net encoder/decoder skip structure, QK-norm with learned
# head gains, and ReLU^2 FFNs. The paper's entire thesis is MINIMALITY: the ONLY
# change needed to turn a standard pre-LN, cross-attention-only Transformer decoder
# into the object-discovery-capable "TF-Inv" is to INVERT the attention
# normalization axis -- softmax over the SLOT (query) axis so slots COMPETE to
# explain each token, followed by a per-slot renormalization over tokens. Every
# graft v4 piled on top (a) is unvalidated by the paper, (b) several were designed
# for a different dataflow, and (c) collectively they make it impossible to
# attribute anything to inverted attention. The initial-slot reinjection also
# actively fights the iterative-refinement fixed-point dynamic the paper relies on.
#
# v6 strips ALL of it back to the paper's faithful minimal recipe and keeps ONLY the
# inverted axis flip, on the proven v24-beta PPO machinery (shared tokenizer +
# separate actor/critic slot stacks, decoupled grad clip, rankgauss advantage,
# clip-higher, HL-Gauss symlog distributional critic, beta action dist, target_kl):
#   - Plain pre-LN cross-attention decoder block: slots += InvAttn(LN(slots), LN(tokens));
#     slots += FFN(LN(slots)). Standard scaled dot-product (no QK-norm/head-gain),
#     GELU FFN (no ReLU^2), standard init (no zero-init proj).
#   - N independent layers (no weight sharing, no GRU, no self-attention), no U-Net
#     skips, no initial-slot reinjection, no learned residual scales.
#   - INVERTED attention is the only non-standard op: softmax over slots, renorm over
#     tokens (paper Fig. 2 / Algorithm 1 normalization axis).
# Learned (non-sampled) slot inits are kept deliberately: the PPO head consumes the
# FLATTENED slot state (n_slots*H), so slots must be order-identified, not exchangeable
# (paper App. A.3: learned inits cost a little FG-ARI but are valid). assign-entropy is
# logged per stack to watch for slot collapse (the small-K + no-reconstruction risk).
#
# This is the clean A/B for "does inverted attention help proprioceptive control".
# Open follow-up (v7, deliberately NOT bundled here to avoid changing many things at
# once): add a self-supervised obs/next-obs reconstruction aux loss so the slot
# competition has the binding pressure the paper's results actually depend on.
#
# Base: ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_invobs_slots_v4.py.
#
# --- inherited symlog_v1 notes ---
#
# PPO + IterThink v24 beta d4-HL-Gauss symlog critic v1.
#
# Variant hypothesis: keep v24 beta PPO machinery and d4-HL-Gauss behavior, but
# add only Dreamer-style symlog/symexp value semantics:
#   scalar GAE return -> symlog -> Gaussian-smoothed categorical target.
#   logits -> E[symlog bin center] -> symexp -> scalar value for GAE/bootstrap.
# Reward normalization/clipping, the raw support range [-10, 10], sigma ratio,
# critic init, and PPO settings stay matched to d4-HL-Gauss. The support edges
# are linear in symlog(v_min)..symlog(v_max), matching hl_gauss_pytorch's
# transform contract while preserving d4's raw normalized-return range.
#
# Base: ppo_continuous_action_iterthink_v24_dist.py / iterthink_v24_beta_s1.
# Only intended algorithmic change: critic target projection style.
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
from dataclasses import dataclass
from math import log, sqrt
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
    learning_rate: float = 3e-4
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

    hidden: int = 64
    slot_dim: int = 64
    actor_slots: int = 4
    critic_slots: int = 4
    slot_blocks: int = 4
    slot_d_qk: int = 8       # v10: low-rank single-head Q/K dim (assignment is cheap)
    slot_mlp_mult: int = 3   # v10: wider pointwise FFN (params reallocated from Q/K)
    compile: bool = True     # v12: torch.compile the (launch-bound) trunk; ~2.5x model-side

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


def justnorm(x, dim=-1):
    """Project onto the unit hypersphere along `dim` (nGPT's representation constraint)."""
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(1e-12)


class ObsTokenizer(nn.Module):
    """Env-normalized scalar observation dimensions -> ordered, dimension-specific tokens.
    v11: the token vectors are projected onto the unit hypersphere (nGPT) so the slot
    stack operates entirely on the sphere. value_weight/pos are kept sphere-normalized
    per token by the post-optimizer-step weight normalization."""

    def __init__(self, obs_dim, H):
        super().__init__()
        self.value_weight = nn.Parameter(torch.randn(obs_dim, H) / sqrt(H))
        self.pos = nn.Parameter(torch.randn(obs_dim, H) / sqrt(H))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.value_weight.unsqueeze(0) + self.pos.unsqueeze(0)
        return justnorm(tokens)


class SlotMLP(nn.Module):
    """nGPT SwiGLU FFN. The single up-projection emits 2*mult*H, gated by a learned
    per-channel `suv` scale (init 1, parameterized with a base_scale prefactor so the
    raw param has unit gradient dynamics), then SwiGLU: u * silu(v). No bias; the
    weights are sphere-normalized after each optimizer step. Width matches v10
    (mlp_mult), preserving v10's Q/K -> FFN param reallocation rationale."""

    def __init__(self, H, mult, base_scale):
        super().__init__()
        self.H = H
        self.fc = nn.Linear(H, 2 * mult * H, bias=False)
        self.proj = nn.Linear(mult * H, H, bias=False)
        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = nn.Parameter(self.suv_init_scaling * torch.ones(2 * mult * H))

    def forward(self, x):
        uv = self.fc(x)
        suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.H ** 0.5))
        uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        return self.proj(u * F.silu(v))


class InvertedCrossAttention(nn.Module):
    """INVERTED cross-attention (Wu et al. 2024) with a LOW-RANK, SINGLE-HEAD
    assignment (v10). The inversion is unchanged: softmax over the SLOT (query) axis
    so slots COMPETE to explain each token, then each slot's weights are renormalized
    over tokens into a convex combination.

    v10 reallocation rationale: assigning 17 dense proprioceptive tokens to 4 slots is
    a TINY relational problem -- full H-dim multi-head Q/K (75% of v6's block params)
    is wasted relational plumbing on a 17-D state with no set/object structure. The
    Q/K projections are collapsed to a single low-rank head (d_qk << H) that computes
    only the assignment; the VALUE and output projection stay full-rank (they carry
    the actual per-slot information). The params freed from Q/K are moved into a wider
    pointwise FFN (mlp_mult 2->3) -- the capacity axis that the winning MLP (v24)
    proves matters for dense control. Net block params ~= v6 (parity), but the budget
    shifts from attention pattern to pointwise mixing."""

    def __init__(self, H, d_qk, base_scale):
        super().__init__()
        self.H = H
        self.d_qk = d_qk
        self.q = nn.Linear(H, d_qk, bias=False)    # low-rank, single-head assignment
        self.k = nn.Linear(H, d_qk, bias=False)
        self.v = nn.Linear(H, H, bias=False)       # full-rank value (carries info)
        self.proj = nn.Linear(H, H, bias=False)    # full-rank output mix
        # nGPT: q,k normalized along d_qk and scaled by a learned per-channel gain.
        # Parameterized with base_scale so the raw param has unit gradient dynamics;
        # effective sqk = 1 at init.
        self.sqk_init_value = 1.0
        self.sqk_init_scaling = base_scale
        self.sqk = nn.Parameter(self.sqk_init_scaling * torch.ones(d_qk))

    def _assign(self, slots, tokens):
        # Shared assignment math (the INVERTED softmax over slots). v12: factored out so
        # both forward and the diagnostic assign_entropy use one source of truth.
        q = self.q(slots)                                          # (B, S, d_qk)
        k = self.k(tokens)                                         # (B, T, d_qk)
        sqk = self.sqk * (self.sqk_init_value / self.sqk_init_scaling)
        q = sqk * justnorm(q)                                      # on the sphere, gained
        k = sqk * justnorm(k)
        # nGPT score scale: *sqrt(d_qk) (restore range to bounded cosines), not /sqrt.
        logits = torch.matmul(q, k.transpose(-1, -2)) * sqrt(self.d_qk)   # (B, S, T)
        return torch.softmax(logits, dim=-2)                    # INVERTED: token -> competing slots

    def forward(self, slots, tokens):
        # v12: PURE (no side effects) so torch.compile captures it without graph breaks.
        assign = self._assign(slots, tokens)
        attn = assign / (assign.sum(dim=-1, keepdim=True) + 1e-8)  # slot -> token renorm (convex)
        y = torch.matmul(attn, self.v(tokens))                                     # (B, S, H)
        return self.proj(y)

    @torch.no_grad()
    def assign_entropy(self, slots, tokens):
        # Diagnostic only (slot-collapse watch): entropy of the inverted assignment over
        # slots, per token, averaged. Recomputed once per iteration in eager mode.
        p = self._assign(slots, tokens).clamp_min(1e-8)
        return -(p * p.log()).sum(dim=-2).mean()


class InvertedSlotBlock(nn.Module):
    """Standard pre-LN cross-attention decoder block (paper Algorithm 2) with the
    attention axis inverted. Plain residual updates -- no initial-slot reinjection,
    no learned residual scales. v10: low-rank single-head assignment + wider FFN."""

    def __init__(self, H, d_qk, mlp_mult, base_scale):
        super().__init__()
        # nGPT: no LayerNorm. Each sublayer is a normalized lerp toward its (sphere-
        # projected) output with a learned per-channel eigen learning rate (init 0.05).
        self.attn = InvertedCrossAttention(H, d_qk, base_scale)
        self.mlp = SlotMLP(H, mlp_mult, base_scale)
        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = base_scale
        self.attn_alpha = nn.Parameter(self.attn_alpha_init_scaling * torch.ones(H))
        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = base_scale
        self.mlp_alpha = nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(H))

    def _lerp(self, slots, update, alpha, init_value, init_scaling):
        # h <- justnorm( h + lr * (justnorm(update) - h) ), lr learned per channel.
        lr = (alpha * (init_value / init_scaling)).abs()
        a = justnorm(slots)
        b = justnorm(update)
        return justnorm(a + lr * (b - a))

    def forward(self, slots, tokens):
        # slots and tokens already on the unit sphere (justnorm at stack entry / per lerp).
        h_att = self.attn(slots, tokens)
        slots = self._lerp(slots, h_att, self.attn_alpha,
                           self.attn_alpha_init_value, self.attn_alpha_init_scaling)
        h_mlp = self.mlp(slots)
        slots = self._lerp(slots, h_mlp, self.mlp_alpha,
                           self.mlp_alpha_init_value, self.mlp_alpha_init_scaling)
        return slots


class InvertedSlotStack(nn.Module):
    """N independent inverted cross-attention layers refining a set of learned slots
    over the observation tokens (paper TF-Inv: cross-attention only, no weight
    sharing, no GRU, no self-attention). The flattened slot state feeds the PPO head."""

    def __init__(self, n_slots, H, n_blocks, d_qk, mlp_mult, base_scale):
        super().__init__()
        assert n_blocks >= 1
        self.n_slots = n_slots
        self.H = H
        self.slots = nn.Parameter(torch.randn(n_slots, H) / sqrt(H))
        self.blocks = nn.ModuleList(
            [InvertedSlotBlock(H, d_qk, mlp_mult, base_scale) for _ in range(n_blocks)]
        )
        # nGPT: no final norm; slots leave each block on the unit sphere. They are
        # rescaled by sqrt(H) at readout so the head-input scale matches v10's old
        # final-LayerNorm output (per-channel RMS ~= 1), leaving the fixed heads
        # uncalibrated-against.

    @property
    def out_dim(self):
        return self.n_slots * self.H

    def forward(self, tokens):
        # v12: PURE (no stat side effects) so torch.compile captures the whole stack.
        B = tokens.shape[0]
        slots = justnorm(self.slots).unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            slots = block(slots, tokens)
        return (slots * sqrt(self.H)).flatten(1)

    @torch.no_grad()
    def assign_entropy(self, tokens):
        # Diagnostic, eager, once per iteration. Mirrors forward's slot trajectory but
        # reads each block's assignment entropy instead of mutating hot-path state.
        slots = justnorm(self.slots).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        ents = []
        for block in self.blocks:
            ents.append(block.attn.assign_entropy(slots, tokens))
            slots = block(slots, tokens)
        return torch.stack(ents).mean()


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, actor_slots, critic_slots, slot_blocks, slot_d_qk, slot_mlp_mult):
        super().__init__()
        base_scale = 1.0 / sqrt(H)   # nGPT base scale (single scale for alpha/sqk/suv params)
        self.tokenizer = ObsTokenizer(in_dim, H)
        self.actor_stack = (
            InvertedSlotStack(actor_slots, H, slot_blocks, slot_d_qk, slot_mlp_mult, base_scale)
            if actor_slots is not None
            else None
        )
        self.critic_stack = (
            InvertedSlotStack(critic_slots, H, slot_blocks, slot_d_qk, slot_mlp_mult, base_scale)
            if critic_slots is not None
            else None
        )

    @torch.no_grad()
    def normalize_weights(self):
        """nGPT weight normalization: after each optimizer step, project every trunk
        matrix back onto the unit hypersphere along its embedding dim (replaces weight
        decay; keeps the eigen-lr step sizes calibrated). Input-side matrices (q,k,v,
        c_fc, tokenizer value/pos) are normalized per output/token row (dim=1); the
        output projections (attn proj, mlp proj) per input column (dim=0)."""
        self.tokenizer.value_weight.copy_(justnorm(self.tokenizer.value_weight, dim=1))
        self.tokenizer.pos.copy_(justnorm(self.tokenizer.pos, dim=1))
        for stack in (self.actor_stack, self.critic_stack):
            if stack is None:
                continue
            for block in stack.blocks:
                block.attn.q.weight.copy_(justnorm(block.attn.q.weight, dim=1))
                block.attn.k.weight.copy_(justnorm(block.attn.k.weight, dim=1))
                block.attn.v.weight.copy_(justnorm(block.attn.v.weight, dim=1))
                block.attn.proj.weight.copy_(justnorm(block.attn.proj.weight, dim=0))
                block.mlp.fc.weight.copy_(justnorm(block.mlp.fc.weight, dim=1))
                block.mlp.proj.weight.copy_(justnorm(block.mlp.proj.weight, dim=0))

    def encode_tokens(self, x):
        return self.tokenizer(x)

    @property
    def actor_out_dim(self):
        assert self.actor_stack is not None
        return self.actor_stack.out_dim

    @property
    def critic_out_dim(self):
        assert self.critic_stack is not None
        return self.critic_stack.out_dim

    def forward_actor(self, x):
        assert self.actor_stack is not None
        return self.actor_stack(self.encode_tokens(x))

    def forward_critic(self, x):
        assert self.critic_stack is not None
        return self.critic_stack(self.encode_tokens(x))

    def forward(self, x):
        tokens = self.encode_tokens(x)
        assert self.actor_stack is not None and self.critic_stack is not None
        return self.actor_stack(tokens), self.critic_stack(tokens)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        assert args.slot_dim == H, "slot_dim must equal hidden in v4"
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(
                obs_dim,
                H,
                args.actor_slots,
                args.critic_slots,
                args.slot_blocks,
                args.slot_d_qk,
                args.slot_mlp_mult,
            )
            actor_feat_dim = self.trunk.actor_out_dim
            critic_feat_dim = self.trunk.critic_out_dim
        else:
            self.critic_trunk = ThinkTrunk(
                obs_dim,
                H,
                None,
                args.critic_slots,
                args.slot_blocks,
                args.slot_d_qk,
                args.slot_mlp_mult,
            )
            self.actor_trunk = ThinkTrunk(
                obs_dim,
                H,
                args.actor_slots,
                None,
                args.slot_blocks,
                args.slot_d_qk,
                args.slot_mlp_mult,
            )
            actor_feat_dim = self.actor_trunk.actor_out_dim
            critic_feat_dim = self.critic_trunk.critic_out_dim
        self.actor_feat_dim = actor_feat_dim
        self.critic_feat_dim = critic_feat_dim
        # Categorical critic with a PEAKED init: small weight + Gaussian-logit
        # bias so the initial value distribution starts as a sharp zero-return
        # prior instead of a high-variance uniform distribution. The bias lives
        # in the categorical coordinate, which is symlog-space when enabled.
        self.critic_head = layer_init(nn.Linear(critic_feat_dim, args.num_bins), std=0.1)
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
            self.actor_head = layer_init(nn.Linear(actor_feat_dim, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(actor_feat_dim, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # dreamer4 unimodal Beta: two concentration heads, alpha,beta = 1 + softplus.
            self.actor_alpha_head = layer_init(nn.Linear(actor_feat_dim, act_dim), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(actor_feat_dim, act_dim), std=0.01)
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")

    @torch.no_grad()
    def normalize_trunk_weights(self):
        """nGPT weight re-normalization, applied to every trunk after each optimizer
        step (and once at init). Heads are deliberately excluded."""
        if self.share_backbone:
            self.trunk.normalize_weights()
        else:
            self.actor_trunk.normalize_weights()
            self.critic_trunk.normalize_weights()

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
        # Return (actor_feat, critic_feat). In shared mode, actor and critic use
        # separate inverted-attention slot stacks over the same obs tokens.
        if self.share_backbone:
            return self.trunk(x)
        return self.actor_trunk.forward_actor(x), self.critic_trunk.forward_critic(x)

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
    agent.normalize_trunk_weights()   # nGPT: matrices start on the unit hypersphere
    if args.compile:
        # v12: fuse the launch-bound slot STACKS (where the ~48 tiny justnorm/lerp ops
        # live). DEFAULT mode (not reduce-overhead/cudagraphs, which backfire on tiny ops).
        #
        # CRITICAL: compile each STACK separately, NOT the whole trunk. A single compiled
        # forward emitting BOTH (actor_feat, critic_feat) is backwarded TWICE by the
        # decoupled dual-clip (critic loss w/ retain_graph, then actor loss) -- each call
        # supplies a cotangent for only one output, which AOTAutograd's compiled backward
        # rejects. With separate per-stack compiled functions, the critic backward hits
        # only critic_stack's graph (once, full cotangent) and the actor backward only
        # actor_stack's; the shared tokenizer stays eager so its double-backward is eager
        # and fine. Params are graph inputs => in-place weight-norm and the optimizer
        # stepping the same tensors stay correct; eager methods (assign_entropy,
        # normalize_weights, encode_tokens) remain reachable via OptimizedModule delegation.
        # NOTE: TF32 (set_float32_matmul_precision("high")) was tried and is STRICTLY
        # WORSE here -- ~10% SLOWER than fp32 on these tiny matmuls (we are launch-bound,
        # not FLOP-bound, so TF32 buys no FLOPs and only adds overhead) AND lower precision
        # for the 511-bin distributional critic / PPO ratios. Left at fp32 (default).
        # The decoupled dual-clip backprops the value loss with retain_graph=True (to
        # reuse the shared-trunk graph for the policy backward). AOTAutograd's
        # donated-buffer optimization frees saved buffers assuming single-use and is
        # incompatible with retain_graph -> disable it (PyTorch's prescribed fix).
        import torch._functorch.config as _functorch_config
        _functorch_config.donated_buffer = False

        def _compile_stacks(trunk):
            if trunk.actor_stack is not None:
                trunk.actor_stack = torch.compile(trunk.actor_stack)
            if trunk.critic_stack is not None:
                trunk.critic_stack = torch.compile(trunk.critic_stack)

        if args.share_backbone:
            _compile_stacks(agent.trunk)
        else:
            _compile_stacks(agent.actor_trunk)
            _compile_stacks(agent.critic_trunk)
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
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

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
                    agent.normalize_trunk_weights()   # nGPT: re-project weights onto the sphere
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    agent.normalize_trunk_weights()   # nGPT: re-project weights onto the sphere

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()

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
        # v12: assign-entropy diagnostic recomputed once per iteration in a separate
        # eager pass (removed from the compiled hot forward). encode_tokens / stacks
        # remain reachable through OptimizedModule delegation when compiled.
        with torch.no_grad():
            stat_obs = b_obs[: min(1024, b_obs.shape[0])]
            if args.share_backbone:
                stat_toks = agent.trunk.encode_tokens(stat_obs)
                actor_ent = agent.trunk.actor_stack.assign_entropy(stat_toks)
                critic_ent = agent.trunk.critic_stack.assign_entropy(stat_toks)
            else:
                actor_ent = agent.actor_trunk.actor_stack.assign_entropy(
                    agent.actor_trunk.encode_tokens(stat_obs))
                critic_ent = agent.critic_trunk.critic_stack.assign_entropy(
                    agent.critic_trunk.encode_tokens(stat_obs))
        writer.add_scalar("debug/invslot_actor_assign_entropy", actor_ent.item(), global_step)
        writer.add_scalar("debug/invslot_critic_assign_entropy", critic_ent.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

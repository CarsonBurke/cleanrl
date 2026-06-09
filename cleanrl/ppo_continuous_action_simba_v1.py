# PPO + IterThink v24 beta d4-HL-Gauss symlog + SimBa dense-MLP trunk (simba_v1).
#
# THE PIVOT. ~17 architecturally-diverse DSRG (sparse recurrent graph) variants all plateaued
# in a tight 6900-7400 band on HalfCheetah, while off-policy SAC reaches ~10-12k. A plateau
# INVARIANT to wildly different trunks (routing, dendrites, norm placement, tick count) means
# the trunk representation was never the binding constraint -- the binding constraint is what
# was held FIXED across all of them: the optimization config + the trunk's small width. Three
# independent analyses (config diagnostic, SOTA literature, adversarial critique) converged on
# this. simba_v1 acts on it with TWO orthogonal, well-supported levers and drops the graph:
#
#   (1) CAPACITY + NORMALIZATION (SimBa, ICLR'25). Replace the routed graph with a wide
#       residual MLP: obs -> Linear(W=512) -> [pre-LN inverted-bottleneck residual blocks] ->
#       post-LN -> actor/critic projections (H=256). Naive PPO widening REGRESSES; the
#       LayerNorm + residual "simplicity bias" is what lets capacity convert to score (also
#       the top mitigation for on-policy plasticity loss). The old trunk was width-64.
#   (2) UN-THROTTLE THE POLICY. The lineage's own diagnostics showed critic EV ~0.95 but
#       clipfrac 0.30 vs 0.11 baseline + KL hitting its cap after ~1-2 of 10 epochs = the
#       policy was leashed. Fix: target_kl 0.03 -> None, grad-clip via the standard global
#       0.5 (separate_grad_clip off), adv_transform rankgauss -> none (rankgauss flattened
#       advantage magnitude). See Args.
#
# KEPT (the genuinely-good, validated pieces): HL-Gauss symlog categorical critic, Beta actor,
# GAE, obs/reward normalization wrappers, 16 envs. The ENTIRE training loop is byte-identical
# to the DSRG files -- MLPTrunk is a drop-in with the same public surface (forward->(B,2,H),
# zero route_loss, empty cells). HYPOTHESIS: this breaks the 7000 plateau decisively.
#
# ATTRIBUTION (companion runs): dsrg_v18 = the SAME graph + only the config fix (isolates the
# config lever); simba_v1 --mlp-width 64 --mlp-depth 2 --hidden 64 = matched-param tiny MLP +
# fixed config (isolates the capacity lever, and falsifies the graph if it ties the big MLP).
#
# --- v8 lineage note ---
# v8 made the per-tick CARRY a PURE ADDITIVE RESIDUAL `norm(h + hhat)` (vs v7's LSTM-style
# gated carry): an unconditional identity highway, no gate that can saturate off the update/
# carry path. Best DSRG to date: 6931 @ 6.7M (~5x the dense PPO baseline ~1400).
#
# --- v7 lineage note ---
# WHAT CHANGES vs DSRG v6: the READOUT becomes COMPETITIVE and obs-CONDITIONAL. v6 fixed
# the graph's EDGE routing (competition not saturation, +26% over v5) but diagnostics
# then said the remaining gap to baseline is the ACTOR/OUTPUT path, not the graph: the
# critic is excellent (EV ~0.95) yet returns sit ~60% of a plain-MLP PPO at 1M. The
# PathwayReadout -- the actor's ONLY obs->action path -- used a STATIC, input-independent
# sigmoid gate (a bare Parameter): the exact saturating, frozen selection v6 fixed
# elsewhere, and it laundered the policy through a fixed random feature map. v7 makes the
# readout select WHICH perceptrons each output reads via a competitive softmax(top-m)
# gate QUERIED FROM THE OBS, so actor/critic read task-relevant perceptrons per state.
# Variance controls keep the clipfrac win (v4->v5 cut clipfrac 0.30->0.13): the obs query
# starts ~0 (a built-in beta-anneal from a static competitive base), the actor reads
# FEWER sources than the critic (asymmetric m; critic is not in the PPO ratio), and a
# learnable per-output scale compensates the softmax mass. Queued for v8 (kept out of v7
# for clean attribution): shrink route_bias to unlock graph-routing conditionality,
# node-dim/H capacity, and a direct low-var obs-residual into the actor feature.
#
# --- v6 lineage note ---
# WHAT CHANGED vs DSRG v5: the ROUTING PRIMITIVE. v5's gates were independent per-edge
# SIGMOIDS made crisp by saturation (a budget held the open MEAN at ~0.15, an entropy
# PENALTY pushed each gate to {0,1}). That is self-defeating: crispness comes from
# ABSOLUTE logit magnitude, and a saturated sigmoid has p(1-p)~0, so ALL gradient on
# the routing logits vanishes. Measured on HC: by ~200k the gate entropy collapsed
# 0.23->0.009 and the route grad-norm fell 0.33->0.008 (12x below its clip) -- the
# topology FROZE, committed to immature features, input-INDEPENDENT (conditionality
# ~0.01, a static graph). Annealing the penalty only delays the freeze.
#
# v6 makes crispness come from COMPETITION, not saturation. Each receiver node already
# has K fixed candidate sources; selecting among them is sparse attention / MoE top-k
# routing: softmax over the K candidates (temp tau), keep top-m and renormalize -> EXACT
# m/K sparsity, structurally, with NO budget and NO entropy penalty. Competition-based
# selection has live gradient at ANY score scale (nothing drives scores to +-inf), so
# the active SET keeps reordering (plastic) and depends on q_i(state) (input-conditional
# -- restores conditionality). Load over source perceptrons is balanced by the standard
# Switch/GShard aux loss (P_j*f_j), replacing both the budget and the entropy penalty
# with one principled term. Net: fewer knobs (drop lambda_route, lambda_gate_ent,
# straight_through, the anneal), no freeze, genuinely dynamic circuits. Everything else
# -- per-tick untied cells, low-rank per-edge weights, carry highway, pathway readout,
# decoupled route-grad clip -- is unchanged from v5.
#
# --- v5 lineage note ---
# WHAT CHANGED vs DSRG v4: the READOUT. v4 used PMA (attention pool) -- an output
# node that adaptively attends over ALL N perceptrons. v5 instead makes the readout
# obey the same fan-in-K sparse-pathway structure as the rest of the graph: each
# output node (actor, critic) reads a BOUNDED, FIXED set of K_read source perceptrons
# via learnable gated pathways, then projects (K_read*d -> H). Biological motivation:
# output/control neurons receive a bounded set of specific gated projections, not a
# dense readout of the whole population. Properties: (a) low-variance like PMA (fixes
# the clipfrac 0.30 bottleneck) but (b) it is a HARDER bottleneck -- it forces the
# network to ROUTE task-relevant info into those specific source perceptrons over the
# ticks, so the routing finally matters for the output; (c) separate source sets +
# projection per output -> decoupled actor/critic readout off one shared graph trunk
# (the "1 seed actor / 1 seed critic" idea, as bounded pathways instead of attention).
# Same (B, 2, H) -> (actor_feat, critic_feat) interface as v4; only the readout module
# differs. Functional routing (v3 strong-lambda decoupled route grad) retained.
#
# --- v4 lineage note ---
# WHAT CHANGED vs DSRG v3 (diagnosed across the v1->v3 sweep on HC-8M: the per-tick
# untied cells gave an EXCELLENT critic, EV 0.1->0.95; the decoupled-gradient fix
# made the routing genuinely FUNCTIONAL, active->0.15 & gate-entropy->0, sparse and
# committed; BUT functional routing was score-NEUTRAL vs inert, and the whole line
# tops out at ~7071 = 72% of the 9785 baseline. The remaining gap is POLICY-side:
# clipfrac ~0.30 vs baseline 0.11. Root cause: the readout. v1-v3 flattened the
# whole 1024x8 neuron state into one 8192-vector -> a high-variance policy feature
# (small head/trunk moves swing the action dist a lot -> high clipfrac, KL hits the
# 0.03 leash after 1-2 epochs), AND it forced the actor and critic to share one
# readout, so the strong critic's representation pressure competed with the policy's.
#   * PMA READOUT (Set-Transformer "Pooling by Multihead Attention"). Replace the
#     flatten with k=2 learned SEED queries that multihead-attend over the N
#     perceptron states: seed 0 -> actor feature, seed 1 -> critic feature. The
#     softmax attention is a low-variance, permutation-invariant soft-SELECTION over
#     all N perceptrons (no active-subset special case -- DSRG updates all neurons
#     densely; sparsity is in the edges, and the attention subsumes any neuron
#     relevance). Separate seeds DECOUPLE the actor/critic readout off one shared
#     graph trunk: the critic seed can pool value-predictive perceptrons while the
#     actor seed pools policy-relevant ones, killing the readout contention without
#     a 2nd backbone. Hypothesis: lower-variance + decoupled policy feature -> lower
#     clipfrac -> the policy trains for more epochs under the KL leash -> closes the
#     gap. Functional routing (v3 strong-lambda decoupled route grad) is retained
#     so sparse specialized perceptrons can feed the PMA selection.
#
# WHAT CHANGES vs DSRG v2 (diagnosed: v2's CRITIC became excellent -- EV 0.1->0.95
# from the per-tick untied cells -- and returns climb with the gap to baseline
# NARROWING 58%->73% by 4M; but the ROUTING was STILL inert, route_active flat at
# 0.50. Root cause: the route-balance + commitment loss rode INSIDE the policy
# backward and was scaled ~12x down by the aggressive actor grad-clip (raw actor
# norm ~3.3 -> 0.25), starving the routing params of gradient.):
#   * DECOUPLED ROUTE GRADIENT. route_loss now gets its OWN backward + its OWN
#     grad-clip on ONLY the routing params (q_proj, key, route_bias), mirroring the
#     existing critic/actor dual-backward decoupling. Full-strength, dedicated
#     pressure -> the gates can finally leave 0.5 and commit to a sparse routing.
#     route_loss is REMOVED from the policy backward (no longer double-counted).
#     Tests the core DSRG hypothesis: does functional conditional routing improve
#     the policy (the remaining gap), or was dense-soft routing already best?
#
# WHAT CHANGES vs DSRG v1 (diagnosed: v1's routing was INERT and the cell was
# under-capacity vs the MoE baseline -> ~65% of baseline score, gap widening):
#   1. ROUTING NOW ROUTES. v1's gates were frozen at sigmoid(0)=0.5 (route_active
#      flat 0.50, gate_entropy ~ln2 = max): a symmetric init the weak regularizers
#      + a counterproductive entropy BONUS never escaped, so "conditional routing"
#      degenerated to uniform soft averaging of one frozen graph (the spec sec16
#      failure). Fixes: (a) learnable per-edge routing BIAS (breaks the 0.5
#      symmetry at step 0, gives the budget/commitment losses a direct knob);
#      (b) the gate-entropy term flips from a BONUS to a PENALTY -> pushes each
#      gate to {0,1} (commit) while the route-balance budget holds the OPEN MEAN
#      at target -> genuine sparse routing, not mush; (c) straight_through ON
#      (hard 0/1 forward, soft backward) so the net actually experiences on/off
#      edges; (d) sharper tau 0.5->0.35.
#   2. CAPACITY. v1 reused ONE weight-tied cell for all T ticks vs the baseline's
#      T independent blocks x 16-expert MoE. v2 UNTIES the cell per tick (tie_ticks
#      flag, default off) -> a depth-T routed sparse graph, closing the gap.
#   3. Gradient checkpointing OFF by default (result-neutral; mem is ~1GB at this N).
#
# WHAT CHANGES vs symlog_v1: ONLY the backbone. The non-recurrent ThinkTrunk
# (entry proj + K dense-cat residual blocks, each dense + soft-MoE) is replaced by
# a WEIGHT-TIED RECURRENT SPARSE GRAPH with receiver-side conditional routing
# (DSRGTrunk). Everything else -- HL-Gauss symlog categorical critic, unimodal
# Beta actor, rankgauss advantage, decoupled dual-backward grad clip, KL early
# stop -- is byte-for-byte the symlog_v1 winner. This is a clean A/B on the
# backbone: is "iterative thinking" better as a routed recurrent graph than as
# stacked residual-cat blocks?
#
# THE GRAPH (spec: "Dynamic Sparse Recurrent Graph w/ receiver-side routing"):
#   N perceptrons, each with state h_i in R^d (lean broadcast). A FIXED sparse
#   directed topology (fan-in K, E = N*K edges, k-regular random, seeded once).
#   State evolves over T fixed internal "ticks" by a single weight-tied cell:
#
#     contribution_ji = g_ji * (W_ji . h_j)        # gate x weight x input (factored)
#     m_i = sum_{j -> i} contribution_ji           # scatter-add (B=1 perceptron)
#     h_i(t) = nu( (1-f_i) . phi(m_i) + f_i . h_i(t-1) )   # LSTM-style carry highway
#
#   - g_ji in (0,1): RECEIVER-SIDE conditional routing. logit = <q_i, k_j>/sqrt(dk)
#     where q_i is a query projected from the receiver's CURRENT state (+ tick
#     embedding) and k_j a learned per-source address. This query/key addressing
#     -- not sender-only gating -- is the load-bearing novelty (state-dependent
#     *structure*, weights stay state-independent). g is the binary-ish "whether";
#     W is the graded "how much"; never merged.
#   - W_ji: shared-base + low-rank per-edge modulation (spec sec 10) so params
#     decouple from N. Orthogonal-init shared base (vanishing control).
#   - f_i: per-channel carry gate -> near-identity gradient highway across ticks.
#   - tick_emb[t]: lets the weight-tied cell know which tick it is on.
#
# DELIBERATE DEVIATIONS FROM THE v4 SPEC (corrections / env-alignment):
#   * N=10k,K=64 -> N=1024,K=16,d=8,T=4. The spec's scale is for a 10k-neuron
#     scaling demo; here obs is 17-dim, act 6-dim. Compute is trivial at this
#     scale; per-edge message memory (held for BPTT) is the only cost -> bounded
#     by per-tick gradient checkpointing. N is the cheap sweep-up axis.
#   * Independent-per-edge d x d weights -> shared-base + low-rank (rank r). Lets
#     N grow without param blowup (spec sec 10).
#   * NO JIT frontier / liveness window / dynamic active set. Dense update of all
#     neurons every tick (spec build-step 1). Sparsity lives in the ROUTING gates,
#     not the neuron set -- a dynamic discrete neuron set breaks dense GPU
#     batching AND BPTT gradient flow, and is a sparse-execution optimization, not
#     a correctness requirement at this scale.
#   * NO PonderNet halting. Fixed T. In PPO the policy must act every env step;
#     "ticks" are internal deliberation, not env steps. Adaptive compute is a
#     separate (v2) feature.
#   * DETERMINISTIC gates g=sigmoid(logit/tau) (constant tau, straight-through
#     optional) -- NOT Gumbel-sigmoid sampling. Stochastic routing would make the
#     policy net random per forward, so the update-pass newlogprob would use
#     DIFFERENT routing than the rollout logprob, corrupting the PPO importance
#     ratio. Determinism keeps the ratio valid; gradients still flow through the
#     sigmoid (reparameterized backprop, not policy gradient), preserving intent.
#   * B=1 perceptron (spec default). branches>1 / divisive inhibition / sphere /
#     SMT / modularity probes are out of scope for v1.
#
# HYPOTHESIS: weight-tied recurrent routing reaches the same or better feature
# quality as the deeper feed-forward ThinkTrunk with a structurally state-
# dependent computation, and the receiver-side routing gives a useful inductive
# bias for continuous control. Bar to beat: the symlog_v1 / iterthink line.
#
# --- inherited symlog_v1 notes (unchanged machinery) ---
# Dreamer-style symlog/symexp value semantics: scalar GAE return -> symlog ->
# Gaussian-smoothed categorical target; logits -> E[symlog bin center] -> symexp
# -> scalar for GAE/bootstrap. d4-HL-Gauss support range [-10,10], sigma ratio,
# critic peaked init, PPO settings all matched. actor_dist="beta" (unimodal,
# dreamer4 default) is the winner path; "gaussian" (state-dep log-variance head,
# soft tanh bound) is the matched control. rankgauss advantage, clip-higher
# (0.2/0.28), shared backbone + decoupled dual-backward clip, target_kl=0.03.
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
from torch.utils.checkpoint import checkpoint
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
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # global clip (simba uses this path: separate_grad_clip=False)
    target_kl: Optional[float] = None  # simba: KL early-stop DISABLED (the 0.03 leash throttled policy improvement)

    # SimBa MLP uses the simple global-clip path (no routing -> no decoupled route backward).
    share_backbone: bool = True      # one trunk for both actor and critic heads
    separate_grad_clip: bool = False # simba: global max_grad_norm clip (the MLP has no routing aux/params)
    actor_grad_clip: float = 0.5     # (unused when separate_grad_clip=False)
    critic_grad_clip: float = 0.5    # (unused when separate_grad_clip=False)
    route_grad_clip: float = 0.1     # (unused -- no routing params)

    # Distributional critic support. Tight + well-resolved.
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ~ N(0, tau^2), sharp at 0
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 2.0  # Dreamer4 / hl_gauss_pytorch default

    # Advantage shaping. simba: DISABLED (rankgauss flattened advantage magnitude). "none" = raw GAE.
    adv_transform: str = "none"
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "minibatch"

    # action distribution. "beta" (unimodal, dreamer4 default) | "gaussian".
    actor_dist: str = "beta"
    logvar_min: float = -8.0
    logvar_max: float = 8.0

    tanh_kappa: float = 2.0
    sigma_floor_bins: float = 2.0
    clip_z_c: float = 2.0
    rank_tanh_kappa: float = 1.5
    pos_neg_alpha: float = 0.5
    cdf_probit_clamp: float = 0.999

    hidden: int = 256                # head feature width H (MLPTrunk proj -> H -> actor/critic heads). Widened from 64.

    # --- SimBa MLP trunk ---
    mlp_width: int = 512             # residual-block width (the trunk's capacity). The matched-param ablation uses --mlp-width 64 --mlp-depth 2 --hidden 64.
    mlp_depth: int = 2               # number of pre-LN residual blocks

    # --- DSRG backbone (UNUSED by the MLP trunk; kept so the shared Args/loop are unchanged) ---
    n_neurons: int = 1024            # N perceptrons (B=1). Cheap to scale via low-rank weights.
    fan_in: int = 16                 # K incoming edges per neuron => E = N*K edges
    n_branches: int = 4              # v15: dendritic branches per neuron (must divide fan_in); each a nonlinear subunit
    node_dim: int = 8                # d: per-neuron broadcast-state dim (lean)
    n_ticks: int = 4                 # T fixed internal deliberation ticks (BPTT through these)
    n_input_neurons: int = 128       # obs encoded into the first N_in neurons at t=0
    weight_rank: int = 4             # r: low-rank per-edge modulation of the shared d x d weight
    route_dk: int = 16              # routing query/key dim
    gate_tau: float = 1.0            # v6: SOFTMAX temperature over the K candidates (competition sharpness)
    route_top_m: int = 4             # v6: # of K candidate edges kept per receiver (exact m/K sparsity)
    lambda_balance: float = 1e-2     # v6: Switch/GShard load-balance aux weight (uniform source usage)
    straight_through: bool = True    # (v5; UNUSED in v6 -- top-m provides the hard selection)
    route_bias_init: float = 0.5     # std of the learnable per-edge routing bias (symmetry breaker)
    route_target_active: float = 0.15  # (v5; UNUSED in v6 -- sparsity is structural at top_m/K)
    lambda_route: float = 100.0      # (v5; UNUSED in v6 -- no open-mean budget)
    lambda_gate_ent: float = 10.0    # (v5; UNUSED in v6 -- no entropy penalty; competition is crisp)
    tie_ticks: bool = False          # False: T independent cells (depth-T graph); True: v1 weight-tied recurrence
    carry_bias_init: float = 1.0     # (v7; UNUSED in v8 -- carry gate replaced by pure additive residual)
    grad_checkpoint: bool = False    # checkpoint each tick to bound BPTT activation memory (off: ~1GB at this N)
    compile: bool = True             # torch.compile the routed cell (fuses the per-tick kernels)

    # --- pathway readout (fan-in-K sparse gated readout, v5) ---
    # Each output node (actor=seed0, critic=seed1) reads K_read FIXED source
    # perceptrons via learnable gated pathways, then projects (K_read*d -> H).
    n_read: int = 64                 # K_read: CANDIDATE source perceptrons per output node (sparse, << N)
    read_m_actor: int = 24           # v7: top-m candidates the ACTOR actually reads (low var -> protects clipfrac)
    read_m_critic: int = 48          # v7: top-m for the CRITIC (not in PPO ratio -> free capacity, reads more)
    read_dk: int = 16                # v7: readout gate query/key dim (obs-conditional selection)
    read_tau: float = 1.0            # v7: readout softmax temperature (competition over candidates)
    # n output nodes FIXED at 2: 0 -> actor feature, 1 -> critic feature.

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


def build_topology(n, k, seed):
    """Fixed k-regular random directed topology. Each neuron i gets K distinct
    non-self incoming sources. Returns src of shape (N, K): src[i] holds the K
    source-neuron indices feeding receiver i. The fan-in-regular (N, K) layout
    lets the cell aggregate by summing over the K axis (no scatter) and apply
    the edge weights as batched GEMMs over N neurons."""
    g = torch.Generator().manual_seed(seed)
    src = torch.empty(n, k, dtype=torch.long)
    for i in range(n):
        perm = torch.randperm(n - 1, generator=g)[:k]
        perm[perm >= i] += 1                              # skip self
        src[i] = perm
    return src


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


class RoutedGraphCell(nn.Module):
    """One weight-tied tick of the sparse recurrent graph, in fan-in-regular
    (B, N, K, .) layout so the whole tick is batched GEMMs over N neurons + a
    reduction over the K incoming edges (no per-edge weight gather, no scatter).

    For receiver i with incoming sources src[i] = (j_1..j_K):
        a_i        = sum_k g_ik h_{j_k}                       # gated aggregation
        s_i        = sum_k g_ik (e_ik (.) (V_{j_k} h_{j_k}))  # low-rank, diag e
        m_i        = W_shared a_i  +  U_i s_i                 # == sum_k g_ik W_ik h_{j_k}
    which is algebraically identical to per-edge sum_k g_ik (W_shared + U_i diag(e_ik) V_{j_k}) h_{j_k}
    because W and the per-edge weight are linear. Routing is receiver-side and
    COMPETITIVE (v6): the per-edge weight is rw_ik = renorm(top_m_k softmax_k(
    <q_i, key_{j_k}>/sqrt(dk)/tau + bias_ik)) -- a softmax over the K candidates,
    top-m kept and renormalized, so each receiver reads an m-sparse, input-conditional
    convex mix of its candidate sources (no sigmoid saturation -> stays plastic)."""

    def __init__(self, n, d, src, args):
        super().__init__()
        self.n, self.d, self.K = n, d, src.shape[1]
        self.dk = args.route_dk
        self.r = args.weight_rank
        self.tau = args.gate_tau
        self.top_m = min(args.route_top_m, src.shape[1])  # v6: kept candidates per receiver
        self.register_buffer("src", src)                  # (N, K) source idx per receiver
        # v15: DENDRITIC SUBUNITS. Partition the K fan-in into B branches of Kb=K/B.
        # Each branch is a nonlinear subunit (inner layer); the soma sums them (outer
        # layer) -> each neuron is a 2-layer net (Poirazi-Mel), zero new params.
        self.nb = args.n_branches
        assert self.K % self.nb == 0, f"fan_in {self.K} must be divisible by n_branches {self.nb}"
        self.kb = self.K // self.nb                       # candidates per branch

        # Weight: shared d x d base (orthogonal -> vanishing control) + low-rank
        # per-edge modulation. e is init SMALL-NONZERO (not 0): with e=0 the U,V
        # gradients vanish (dU,dV ~ e) and the per-edge path dies. Nonzero e keeps
        # all three alive while staying << W_shared (init ~= shared base).
        ws = torch.empty(d, d)
        nn.init.orthogonal_(ws)
        self.W_shared = nn.Parameter(ws)
        self.U = nn.Parameter(torch.randn(n, d, self.r) / sqrt(d))         # per-dst lift
        self.V = nn.Parameter(torch.randn(n, self.r, d) / sqrt(d))         # per-src project
        self.e = nn.Parameter(torch.randn(n, self.K, self.r) * 0.1)        # per-edge scale

        # Routing: receiver query from state(+tick), learned per-source key, PLUS a
        # learnable per-EDGE bias. The bias is the symmetry breaker: v1's gates began
        # at a symmetric ~0.5 (logit~0) that the weak regularizers never escaped, so
        # routing stayed inert. A per-edge bias gives every edge its own learnable
        # default conductance at step 0 -> gates start differentiated AND the
        # route-balance/commitment losses have a direct, dense knob to shape them.
        self.q_proj = layer_init(nn.Linear(d, self.dk), std=1.0)
        self.key = nn.Parameter(torch.randn(n, self.dk) * (1.0 / sqrt(self.dk)))
        self.route_bias = nn.Parameter(torch.randn(n, self.K) * args.route_bias_init)

        # Carry gate (per channel). Bias>0 => start near-identity (gradient highway).
        self.norm = nn.RMSNorm(d, elementwise_affine=False)   # v8: no carry gate -> f_proj removed

    def route_params(self):
        # Params that the routing logit (hence the route-structural loss) depends on
        # directly: the receiver query, the per-source key, the per-edge bias. The
        # decoupled route backward (v3) clips/applies the structural gradient to ONLY
        # these -- NOT the transformation weights -- so routing commits without the
        # actor grad-clip starving it. (Compiled cell delegates attr access to the
        # real param tensors, so these are the same objects the optimizer holds.)
        return [self.q_proj.weight, self.q_proj.bias, self.key, self.route_bias]

    @torch.no_grad()
    def gate_probs(self, h, tick_t, Ksrc):
        # Eager recompute of the per-edge routing WEIGHTS rw (B, N, K) for circuit
        # instrumentation. Identical formula/inputs to forward(), so it reflects the
        # actual routing; kept off the compiled forward (which emits only scalars).
        # Non-selected edges are exactly 0; the top-m kept edges carry the (renormalized)
        # softmax mass -- so a >0 test recovers the active set, and std-over-batch of rw
        # measures input-conditional routing.
        q = self.q_proj(h + tick_t)
        logit = torch.bmm(q.transpose(0, 1), Ksrc.transpose(1, 2)).transpose(0, 1) / sqrt(self.dk)
        logit = (logit + self.route_bias.unsqueeze(0)) / self.tau
        w_soft = torch.softmax(logit, dim=-1)
        topi = w_soft.topk(self.top_m, dim=-1).indices
        mask = torch.zeros_like(w_soft).scatter_(-1, topi, 1.0)
        rw = w_soft * mask
        return rw / (rw.sum(-1, keepdim=True) + 1e-9)

    def forward(self, h, tick_t, Ksrc):
        # h: (B, N, d); tick_t: (N, d) per-neuron tick embedding; Ksrc: (N, K, dk)
        # gathered source keys (tick-invariant, hoisted out of the unroll).
        N, K = self.n, self.K
        src = self.src

        # --- receiver-side COMPETITIVE routing (softmax over the K candidates, top-m
        #     kept). Crispness comes from competition, not sigmoid saturation, so the
        #     selection stays differentiable at ANY score scale -> the active SET keeps
        #     reordering (plastic) and is input-conditional (depends on q_i(state)).
        # logit[b,i,k] = <q_{b,i}, key_{src[i,k]}>. bmm over neurons avoids any
        # (B, N, K, dk) materialization.
        q = self.q_proj(h + tick_t)                                       # (B, N, dk)
        logit = torch.bmm(q.transpose(0, 1), Ksrc.transpose(1, 2))        # (N, B, K)
        logit = logit.transpose(0, 1) / sqrt(self.dk)                     # (B, N, K) state-dep match
        logit = (logit + self.route_bias.unsqueeze(0)) / self.tau         # + per-edge bias, then temperature
        w_soft = torch.softmax(logit, dim=-1)                            # (B, N, K) competition over candidates
        topi = w_soft.topk(self.top_m, dim=-1).indices                    # (B, N, m) kept edges (by rank)
        mask = torch.zeros_like(w_soft).scatter_(-1, topi, 1.0)           # hard top-m keep mask (no grad)
        rw = w_soft * mask                                                # drop non-top-m
        rw = rw / (rw.sum(-1, keepdim=True) + 1e-9)                       # renormalize -> m-sparse, sums to 1
        gg = rw.unsqueeze(-1)                                             # (B, N, K, 1) routing weights

        # --- DENDRITIC SUBUNITS (v15): each neuron is a 2-LAYER net. The K fan-in is
        #     partitioned into nb branches of kb=K/nb; each branch pools ITS inputs and
        #     passes them through a subunit nonlinearity (inner layer), then the soma sums
        #     the subunit outputs through a second nonlinearity (outer layer). Routing,
        #     params, and active-edge count are identical to v8 -- the message just
        #     reorganizes. Branches let CLUSTERED selected edges interact supralinearly
        #     (the NMDA-spike coincidence-detection story) instead of being averaged flat.
        B = h.shape[0]
        nb, kb = self.nb, self.kb
        ggb = gg.view(B, N, nb, kb, 1)                                    # (B, N, nb, kb, 1) per-branch gates
        hsb = h[:, src].view(B, N, nb, kb, self.d)                        # (B, N, nb, kb, d) per-branch sources
        a_b = (ggb * hsb).sum(3)                                          # (B, N, nb, d) per-branch gated pool
        ms_b = a_b @ self.W_shared.t()                                    # (B, N, nb, d) shared transform (broadcast over nb)

        # low-rank per-edge, computed per branch: w_j = V_j h_j (per neuron, branch-invariant)
        w = torch.bmm(h.transpose(0, 1), self.V.transpose(1, 2)).transpose(0, 1)   # (B, N, r)
        wsb = w[:, src].view(B, N, nb, kb, self.r)                        # (B, N, nb, kb, r)
        eb = self.e.view(self.n, nb, kb, self.r)                          # (N, nb, kb, r) per-edge scale, branched
        s_b = (ggb * eb.unsqueeze(0) * wsb).sum(3)                        # (B, N, nb, r) per-branch low-rank coeff
        ml_b = torch.einsum("bnar,ndr->bnad", s_b, self.U)               # (B, N, nb, d) + U_i s_i, per branch

        z_b = ms_b + ml_b                                                # (B, N, nb, d) per-branch pre-activation
        u_b = F.silu(z_b)                                                # (B, N, nb, d) BRANCH subunit nonlinearity (inner)
        m = u_b.sum(2)                                                   # (B, N, d) soma pools the subunit outputs

        hhat = F.silu(m + tick_t)                                        # SOMA nonlinearity (outer); NO inner norm

        # --- PURE ADDITIVE residual (v8), then a SINGLE normalization nu ---
        # v8 replaces v7's LSTM-style carry gate `norm((1-f)*hhat + f*h)` with the
        # iterthink-style additive residual `norm(h + hhat)`. The sigmoid carry could
        # SATURATE -- f->1 zeroes the update gradient (dead compute), f->0 zeroes the
        # carry gradient -- the same magnitude-saturation pathology v6 fixed in routing.
        # A plain h + hhat is an UNCONDITIONAL identity highway: gradient flows straight
        # back across all T ticks to the obs-injected perceptrons. nu is applied once,
        # after the add, NOT on phi(m): the always-nonzero
        # +h floor keeps the post-add vector well-conditioned (no 1/sqrt(eps) RMSNorm
        # blow-up) and holds h at unit scale for the next tick's routing dot-products.
        h_new = self.norm(h + hhat)

        # Route-health stats reduced to scalars INSIDE the (compiled) cell, so the
        # full (B, N, K) gate tensor never leaves the fused kernel and the .mean()/
        # entropy reductions are fused rather than run eagerly per tick.
        # --- routing health (scalars; the (B, N, K) tensor stays in the fused kernel) ---
        # Switch/GShard load balance over SOURCE perceptrons: P_j = mean soft routing
        # mass landing on source j (differentiable); f_j = fraction of HARD top-m
        # selections landing on j (detached). lb = N * sum_j P_j f_j is minimized (->1)
        # iff usage is uniform -> all perceptrons recruited, no hub collapse, diverse
        # circuits. This single term replaces v5's open-mean budget AND entropy penalty.
        src_flat = self.src.reshape(-1)                                   # (N*K,)
        Pj = rw.new_zeros(self.n).index_add(0, src_flat, w_soft.mean(0).reshape(-1))   # soft mass (grad)
        Fj = rw.new_zeros(self.n).index_add(0, src_flat, mask.mean(0).reshape(-1))     # hard load (no grad)
        Pj = Pj / (Pj.sum() + 1e-9)
        Fj = Fj / (Fj.sum() + 1e-9)
        lb = self.n * (Pj * Fj).sum()                                     # >=1; uniform usage -> 1
        # competition entropy (monitoring only): how peaked the softmax is per edge.
        rent = (-(w_soft.clamp_min(1e-9).log() * w_soft).sum(-1)).mean()
        return h_new, lb, rent


class PathwayReadout(nn.Module):
    """Competitive, obs-CONDITIONAL fan-in-K sparse readout (v7). n_out output nodes
    (0->actor, 1->critic) each read from a FIXED set of K_read candidate source
    perceptrons, but WHICH candidates are read is chosen by a COMPETITIVE, INPUT-
    CONDITIONAL gate -- softmax over the candidates -> top-m -> renormalize -- replacing
    v5/v6's STATIC, input-independent sigmoid gate (the same saturation/freeze pattern
    the v6 graph routing fixed, and the actor's only obs->action path was being laundered
    through it). The gate query comes from the OBS (low-dim, low-variance, obs-faithful),
    so the policy selects task-relevant perceptrons PER STATE. Variance controls for PPO
    ratio/clipfrac stability:
      * the obs-query starts ~0 (q_proj std 1e-3) -> readout begins as a static
        competitive base (gate_bias) and grows conditionality only as it earns return
        (a built-in beta-anneal -> bounded clipfrac risk);
      * ASYMMETRIC fan-in: actor reads m_actor sources (low var), critic reads m_critic
        > m_actor (critic is not in the ratio -> free capacity);
      * softmax mass sums to 1 over m (vs v6's ~0.5*K_read); a learnable per-output
        log-scale (init to the old aggregate) compensates so features don't collapse.

        hs = h[:, src]                              # (B, n_out, K_read, d)
        g  = renorm(top_m( softmax(q(obs)·key/sqrt(dk) + gate_bias) ))   # (B, n_out, K_read)
        out[o] = (scale_o * g . hs).reshape(K_read*d) @ proj[o] + bias[o]
    """

    def __init__(self, n, d, k_read, H, obs_dim, n_out=2, m=(24, 48), dk=16, tau=1.0, seed=0):
        super().__init__()
        self.n_out, self.K, self.d, self.dk, self.tau = n_out, k_read, d, dk, tau
        m_list = [min(int(mi), k_read) for mi in list(m)[:n_out]]       # clamp m <= candidate pool
        self.register_buffer("m_per_out", torch.tensor(m_list, dtype=torch.long))
        src = build_topology(n, k_read, seed + 9973)[:n_out].clone()   # (n_out, K_read)
        self.register_buffer("src", src)
        self.key = nn.Parameter(torch.randn(n_out, k_read, dk) / sqrt(dk))
        self.q_proj = nn.Linear(obs_dim, n_out * dk)
        nn.init.normal_(self.q_proj.weight, std=1e-3)                   # near-zero query at init
        nn.init.zeros_(self.q_proj.bias)
        self.gate_bias = nn.Parameter(torch.zeros(n_out, k_read))       # static competitive base
        self.proj = nn.Parameter(torch.randn(n_out, k_read * d, H) / sqrt(k_read * d))
        # mass-faithful per-output init: scale*g_k ~= 0.5 per kept entry (matches v6's
        # sigmoid(0)=0.5 aggregate), so the actor feature norm starts ~equal to v6 (no
        # init clipfrac nudge). scale = 0.5*m -> per-output since actor/critic m differ.
        self.out_scale = nn.Parameter(torch.log(0.5 * torch.tensor(m_list, dtype=torch.float32)))
        self.out_bias = nn.Parameter(torch.zeros(n_out, H))

    def forward(self, h, x):                        # h: (B, N, d); x: (B, obs_dim) obs
        B = h.shape[0]
        hs = h[:, self.src]                         # (B, n_out, K_read, d) gather candidates
        q = self.q_proj(x).view(B, self.n_out, self.dk)                 # (B, n_out, dk) obs query
        logit = torch.einsum("bod,okd->bok", q, self.key) / sqrt(self.dk)  # (B, n_out, K_read)
        logit = logit + self.gate_bias.unsqueeze(0)                    # + static competitive base
        w_soft = torch.softmax(logit / self.tau, dim=-1)               # competition over candidates
        gs = []                                                        # per-output top-m (asymmetric)
        for o in range(self.n_out):
            wo = w_soft[:, o]                                          # (B, K_read)
            topi = wo.topk(int(self.m_per_out[o]), dim=-1).indices     # (B, m_o)
            keep = torch.zeros_like(wo).scatter_(-1, topi, 1.0) * wo
            gs.append(keep / (keep.sum(-1, keepdim=True) + 1e-9))      # renorm -> sums to 1 over m_o
        g = torch.stack(gs, dim=1)                                     # (B, n_out, K_read)
        scale = self.out_scale.exp().view(1, self.n_out, 1, 1)
        hs = (g.unsqueeze(-1) * hs) * scale                           # (B, n_out, K_read, d) gated, rescaled
        feat = hs.reshape(B, self.n_out, self.K * self.d)             # (B, n_out, K_read*d)
        out = torch.einsum("bok,okh->boh", feat, self.proj) + self.out_bias   # (B, n_out, H)
        return out                                  # split by output node outside


class ResidualBlock(nn.Module):
    """SimBa pre-LN inverted-bottleneck residual block: x + Linear(SiLU(Linear(LN(x))))."""

    def __init__(self, width, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = layer_init(nn.Linear(width, expansion * width))
        self.fc2 = layer_init(nn.Linear(expansion * width, width))

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class MLPTrunk(nn.Module):
    """SimBa-style dense trunk -- the DROP-IN ablation of the routed graph. obs -> Linear
    stem -> [pre-LN residual blocks] -> post-LN -> (actor_feat, critic_feat). A plain,
    high-capacity MLP with LayerNorm (SimBa, ICLR'25); no routing, no recurrence, no
    sparsity. Tests directly whether the trunk representation was ever the binding
    constraint -- if this matches/beats the 900-line graph, the graph was inert.

    Keeps the exact public surface of DSRGTrunk so the training loop is byte-identical:
      * forward(x) -> (B, 2, H): [:,0]=actor feat, [:,1]=critic feat
      * self.route_loss: a device-matched constant 0 (no routing aux; with
        separate_grad_clip=False it is simply added to the loss and contributes nothing)
      * self.cells: empty ModuleList -> Agent.route_parameters() is empty and the
        per-iteration circuit-metrics probe is skipped (guarded on len(cells)>0)
      * self._active/_gate_ent/_lb: 0.0 gauges (logged as route_* debug scalars)"""

    def __init__(self, obs_dim, H, args):
        super().__init__()
        W = args.mlp_width
        self.stem = layer_init(nn.Linear(obs_dim, W))
        self.blocks = nn.ModuleList([ResidualBlock(W) for _ in range(args.mlp_depth)])
        self.post_norm = nn.LayerNorm(W)
        self.actor_proj = layer_init(nn.Linear(W, H))
        self.critic_proj = layer_init(nn.Linear(W, H))
        # --- DSRGTrunk-compatible no-op surface for a dense MLP ---
        self.cells = nn.ModuleList()          # empty -> no route params, circuit probe skipped
        self.route_loss = torch.zeros(())     # device-matched in forward()
        self._active = 0.0
        self._gate_ent = 0.0
        self._lb = 0.0

    def forward(self, x):
        self.route_loss = x.new_zeros(())     # no routing aux; on the input's device
        z = self.stem(x)
        for b in self.blocks:
            z = b(z)
        z = self.post_norm(z)
        return torch.stack([self.actor_proj(z), self.critic_proj(z)], dim=1)  # (B, 2, H)


@torch.no_grad()
def circuit_metrics(gates, src, n):
    """Quantify how well the routing forms CONDITIONAL, MODULAR, DIVERSE circuits.
    gates: (B, T, N, K) gate probs; src: (N, K) source-perceptron idx; n: #perceptrons.
      conditionality : mean_edge std_batch(p)   -- gates that VARY with the input (true
                       conditional routing) vs static (a fixed learned sparse graph).
      tick_dynamism  : mean std_T(p)            -- routing that changes across the T
                       deliberation ticks ("thinking" reshapes the circuit).
      diversity      : 1 - mean pairwise cosine of per-sample active masks -- distinct
                       circuits per input (vs one circuit reused for everything).
      reuse_cv       : CV of per-source active out-degree -- hub perceptrons reused by
                       many circuits (modular sub-routines) vs uniform usage.
      live_frac      : fraction of perceptrons that are an active source for >=1 edge."""
    B, T, N, K = gates.shape
    m = (gates > 1e-6).float()        # v6: routing weights are exact-0 off the top-m set
    conditionality = gates.std(dim=0).mean().item()
    tick_dynamism = gates.std(dim=1).mean().item()
    # diversity over a random pairing of samples (active-mask cosine)
    mf = m.reshape(B, -1)
    perm = torch.randperm(B, device=gates.device)
    h = B // 2
    aa, bb = mf[perm[:h]], mf[perm[h:2 * h]]
    cos = (aa * bb).sum(-1) / (aa.norm(dim=-1) * bb.norm(dim=-1) + 1e-8)
    diversity = (1.0 - cos.mean()).item()
    # per-source active out-degree (scatter edge activity onto source perceptrons)
    deg = torch.zeros(n, device=gates.device)
    deg.index_add_(0, src.reshape(-1), m.sum(dim=(0, 1)).reshape(-1))
    reuse_cv = (deg.std() / (deg.mean() + 1e-8)).item()
    live_frac = (deg > 0).float().mean().item()
    return dict(conditionality=conditionality, tick_dynamism=tick_dynamism,
                diversity=diversity, reuse_cv=reuse_cv, live_frac=live_frac)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            self.trunk = MLPTrunk(obs_dim, H, args)
        else:
            self.critic_trunk = MLPTrunk(obs_dim, H, args)
            self.actor_trunk = MLPTrunk(obs_dim, H, args)
        # Categorical critic with a PEAKED init (sharp zero-return prior). Bias in
        # the categorical coordinate (symlog-space when enabled).
        self.critic_head = layer_init(nn.Linear(H, args.num_bins), std=0.1)
        with torch.no_grad():
            support_min, support_max = value_support_bounds(args)
            edge_width = (support_max - support_min) / args.num_bins
            z = torch.linspace(
                support_min + 0.5 * edge_width,
                support_max - 0.5 * edge_width,
                args.num_bins,
            )
            self.critic_head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
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
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
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
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0
        return dist, to_action, log_det_fn

    def _trunks(self, x):
        # The DSRGTrunk PMA readout returns (B, 2, H): seed 0 -> actor, seed 1 -> critic.
        # With a shared backbone the two seeds decouple the actor/critic readout off
        # the one graph trunk. With separate backbones each trunk owns its seed.
        if self.share_backbone:
            pooled = self.trunk(x)
            return pooled[:, 0], pooled[:, 1]
        return self.actor_trunk(x)[:, 0], self.critic_trunk(x)[:, 1]

    def route_loss(self):
        # Structural-prior loss from the trunk that carries the policy gradient.
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        return trunk.route_loss

    def get_value(self, x):
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

    def get_action_and_value(self, x, z=None):
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
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, value_logits

    def actor_parameters(self):
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())

    def route_parameters(self):
        # Routing params of the trunk that produces route_loss (the actor/shared one),
        # deduped by identity so tie_ticks (shared cell across ticks) isn't counted
        # T times in the route grad-clip norm.
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        seen, ps = set(), []
        for cell in trunk.cells:
            for p in cell.route_params():
                if id(p) not in seen:
                    seen.add(id(p))
                    ps.append(p)
        return ps


def shape_advantage(gae, sigma, u, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform."""
    if args.adv_transform == "none" or args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_std":
        return torch.tanh(gae / (args.tanh_kappa * sigma))
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "cdf_probit":
        centered = (2.0 * u - 1.0)
        c = args.cdf_probit_clamp
        return (2.0 ** 0.5) * torch.erfinv(centered.clamp(-c, c))
    elif args.adv_transform == "clip_z":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return gz.clamp(-args.clip_z_c, args.clip_z_c)
    elif args.adv_transform == "rankgauss":
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        return (2.0 ** 0.5) * torch.erfinv(centered)
    elif args.adv_transform == "rankgauss_signed":
        c = args.cdf_probit_clamp
        out = torch.zeros_like(gae)
        for side in (gae > 0, gae < 0):
            if side.any():
                g = gae[side]
                r = g.argsort().argsort().to(torch.float32)
                half = (r + 0.5) / float(g.numel())
                uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)
                ctr = (2.0 * uq - 1.0).clamp(-c, c)
                out[side] = (2.0 ** 0.5) * torch.erfinv(ctr)
        return out
    elif args.adv_transform == "rankgauss_temp":
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        zq = (2.0 ** 0.5) * torch.erfinv(centered)
        return torch.tanh(zq / args.rank_tanh_kappa)
    elif args.adv_transform == "rankgauss_signmag":
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c))).abs()
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
    # TF32 tensor-core matmul/conv path (the cell GEMMs, bmms, and the 8192->H
    # readout). ~10-25% on the matmul-bound ops; precision drops to ~10-bit
    # mantissa (PPO/HL-Gauss tolerate it). Independent of cudnn.deterministic.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    if args.compile:
        # Fuse the per-tick routed-cell kernels (the bandwidth-bound bottleneck).
        # Compile the cell (inside the BPTT/checkpoint unroll), not the whole trunk.
        # donated_buffer must be off: the decoupled dual-backward backprops the
        # value loss with retain_graph=True (shared trunk), which is incompatible
        # with compiled donated buffers.
        import torch._functorch.config
        torch._functorch.config.donated_buffer = False
        for tr in ([agent.trunk] if args.share_backbone else [agent.actor_trunk, agent.critic_trunk]):
            # Compile each UNIQUE cell once (tie_ticks reuses one module across ticks),
            # then rebuild the ModuleList preserving the per-tick ordering/sharing.
            compiled = {id(c): torch.compile(c) for c in dict((id(c), c) for c in tr.cells).values()}
            tr.cells = nn.ModuleList([compiled[id(c)] for c in tr.cells])
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    route_params = agent.route_parameters()   # routing-only params for the decoupled route backward

    auto_alpha = args.auto_entropy and args.actor_dist == "gaussian"
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
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
            if auto_alpha:
                _, _, boot_logprob, _, boot_logits = agent.get_action_and_value(next_obs)
                bootstrap_logits = boot_logits
                bootstrap_probs = torch.softmax(bootstrap_logits, dim=-1)
                alpha_r = log_alpha.exp().detach()
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                bootstrap_logits = agent.get_value(next_obs)
                bootstrap_probs = torch.softmax(bootstrap_logits, dim=-1)
                next_value_bonus = None
            next_value = hl_support.to_scalar(bootstrap_logits).reshape(1, -1)
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
            target_probs = hl_support.project(returns)
            sigma = (value_probs * (raw_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            if args.value_symlog:
                value_coord = symlog(values).clamp(hl_support.v_min, hl_support.v_max)
                floor_idx = ((value_coord - hl_support.v_min) / bin_width).floor().long().clamp(0, args.num_bins - 1)
                sigma_floor = args.sigma_floor_bins * raw_bin_widths[floor_idx]
            else:
                sigma_floor = args.sigma_floor_bins * bin_width
            sigma = torch.maximum(sigma, sigma_floor)
            returns_coord = symlog(returns) if args.value_symlog else returns
            cdf_frac = ((returns_coord.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.num_bins)
        b_u = u.reshape(-1)
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        route_trunk = agent.trunk if agent.share_backbone else agent.actor_trunk
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                route_loss = agent.route_loss()
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

                if args.pos_neg_alpha != 0.5:
                    mb_advantages = mb_advantages * torch.where(
                        mb_raw_adv >= 0,
                        2.0 * args.pos_neg_alpha,
                        2.0 * (1.0 - args.pos_neg_alpha),
                    )

                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

                entropy_loss = entropy.mean()

                if auto_alpha:
                    ent_coef_eff = log_alpha.exp().detach()
                    alpha_loss = (-log_alpha.exp() * (newlogprob.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    ent_coef_eff = args.ent_coef

                if args.separate_grad_clip:
                    # TRIPLE-BACKWARD decoupled clipping (v3). Value, ROUTE-aux, and
                    # policy gradients are each backpropped + clipped to their own
                    # max-norm, then summed. In v6 route_loss is ONLY the load-balance
                    # aux, applied (decoupled-clipped) to the routing params so it can't
                    # be drowned by the policy grad-norm; the TASK gradient still reaches
                    # the routing params via the actor backward (they are in actor_params),
                    # which is what keeps competitive routing plastic and task-adaptive.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    route_loss.backward(retain_graph=True)
                    route_gn = nn.utils.clip_grad_norm_(route_params, args.route_grad_clip)
                    route_grads = [(p, p.grad.detach().clone()) for p in route_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    for p, g in route_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef + route_loss
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    route_gn = actor_gn
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Route-health gauges read ONCE here (last minibatch's values), not per
        # minibatch -- the per-mb float() was a host sync stalling the GPU 2x/mb.
        route_active = float(route_trunk._active)
        route_ent = float(route_trunk._gate_ent)
        route_lb = float(route_trunk._lb)

        # Circuit-quality probe (once per iteration, off the hot path): does the
        # routing build CONDITIONAL, MODULAR, DIVERSE circuits, or a static graph?
        cm = circuit_metrics(*route_trunk.probe_gates(b_obs[:256]), args.n_neurons) if len(route_trunk.cells) > 0 else {}
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
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
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)
        writer.add_scalar("debug/u_edge_frac", u_edge_frac, global_step)
        writer.add_scalar("debug/sigma_mean", b_sigma.mean().item(), global_step)
        writer.add_scalar("debug/route_active_frac", route_active, global_step)
        writer.add_scalar("debug/route_gate_entropy", route_ent, global_step)
        writer.add_scalar("debug/route_load_balance", route_lb, global_step)
        writer.add_scalar("losses/route_grad_norm", float(route_gn), global_step)
        for _k, _v in cm.items():
            writer.add_scalar(f"circuit/{_k}", _v, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

# PPO + IterThink v24 beta d4-HL-Gauss symlog + Dynamic Sparse Recurrent Graph (DSRG) v2.
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
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash

    # shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one DSRGTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Distributional critic support. Tight + well-resolved.
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ~ N(0, tau^2), sharp at 0
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 2.0  # Dreamer4 / hl_gauss_pytorch default

    # Advantage shaping (v19). See header / shape_advantage.
    adv_transform: str = "rankgauss"
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

    hidden: int = 64                 # head feature width H (DSRG readout -> H -> actor/critic heads)

    # --- DSRG backbone ---
    n_neurons: int = 1024            # N perceptrons (B=1). Cheap to scale via low-rank weights.
    fan_in: int = 16                 # K incoming edges per neuron => E = N*K edges
    node_dim: int = 8                # d: per-neuron broadcast-state dim (lean)
    n_ticks: int = 4                 # T fixed internal deliberation ticks (BPTT through these)
    n_input_neurons: int = 128       # obs encoded into the first N_in neurons at t=0
    weight_rank: int = 4             # r: low-rank per-edge modulation of the shared d x d weight
    route_dk: int = 16              # routing query/key dim
    gate_tau: float = 0.35           # CONSTANT gate temperature g=sigmoid(logit/tau) (no anneal); sharper than v1
    straight_through: bool = True    # hard {0,1} forward, soft backward (true binary routing)
    route_bias_init: float = 0.5     # std of the learnable per-edge routing bias (symmetry breaker)
    route_target_active: float = 0.15  # target fraction of open gates (route-balance budget)
    lambda_route: float = 3e-2       # weight on (mean_active - target)^2 budget (binds the open mean)
    lambda_gate_ent: float = 3e-3    # gate-entropy PENALTY (pushes each gate to {0,1} -> commit)
    tie_ticks: bool = False          # False: T independent cells (depth-T graph); True: v1 weight-tied recurrence
    carry_bias_init: float = 1.0     # f-gate bias init (>0 => start near-identity highway)
    grad_checkpoint: bool = False    # checkpoint each tick to bound BPTT activation memory (off: ~1GB at this N)
    compile: bool = True             # torch.compile the routed cell (fuses the per-tick kernels)

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
    because W and the scalar gate are linear. Routing is receiver-side and
    deterministic: g_ik = sigmoid(<q_i, key_{j_k}> / sqrt(dk) / tau)."""

    def __init__(self, n, d, src, args):
        super().__init__()
        self.n, self.d, self.K = n, d, src.shape[1]
        self.dk = args.route_dk
        self.r = args.weight_rank
        self.tau = args.gate_tau
        self.straight_through = args.straight_through
        self.register_buffer("src", src)                  # (N, K) source idx per receiver

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
        self.f_proj = layer_init(nn.Linear(2 * d, d), std=1.0, bias_const=args.carry_bias_init)
        self.norm = nn.RMSNorm(d, elementwise_affine=False)

    def forward(self, h, tick_t, Ksrc):
        # h: (B, N, d); tick_t: (N, d) per-neuron tick embedding; Ksrc: (N, K, dk)
        # gathered source keys (tick-invariant, hoisted out of the unroll).
        N, K = self.n, self.K
        src = self.src

        # --- receiver-side routing gates (deterministic), as batched GEMM ---
        # logit[b,i,k] = <q_{b,i}, key_{src[i,k]}>. bmm over neurons avoids any
        # (B, N, K, dk) materialization.
        q = self.q_proj(h + tick_t)                                       # (B, N, dk)
        logit = torch.bmm(q.transpose(0, 1), Ksrc.transpose(1, 2))        # (N, B, K)
        logit = logit.transpose(0, 1) / sqrt(self.dk)                     # (B, N, K) state-dep match
        logit = (logit + self.route_bias.unsqueeze(0)) / self.tau         # + per-edge bias, then temperature
        p = torch.sigmoid(logit)                                          # gate prob (for stats)
        g = p + ((p > 0.5).float() - p).detach() if self.straight_through else p
        gg = g.unsqueeze(-1)                                              # (B, N, K, 1)

        # --- shared part: a_i = sum_k g_ik h_{src} ; m_shared = a @ W_shared^T ---
        hs = h[:, src]                                                    # (B, N, K, d)
        a = (gg * hs).sum(2)                                              # (B, N, d)
        m = a @ self.W_shared.t()                                         # (B, N, d)

        # --- low-rank part: w_j = V_j h_j (per neuron); s_i = sum_k g_ik e_ik (.) w_{src} ---
        w = torch.bmm(h.transpose(0, 1), self.V.transpose(1, 2)).transpose(0, 1)   # (B, N, r)
        ws = w[:, src]                                                    # (B, N, K, r)
        s = (gg * self.e.unsqueeze(0) * ws).sum(2)                        # (B, N, r)
        m = m + torch.bmm(s.transpose(0, 1), self.U.transpose(1, 2)).transpose(0, 1)  # + U_i s_i

        hhat = F.silu(m + tick_t)                                         # phi (tick-aware); NO inner norm

        # --- carry highway, then a SINGLE normalization nu ---
        # nu is applied once, after the carry mix, NOT on phi(m): RMSNorm on a
        # near-zero vector (a fully gated-off neuron) has a ~1/sqrt(eps) Jacobian
        # that explodes the gradient. The carry mix always carries the nonzero
        # f*h floor, so the post-mix vector is well-conditioned.
        f = torch.sigmoid(self.f_proj(torch.cat([h, hhat], dim=-1)))     # (B, N, d)
        h_new = self.norm((1.0 - f) * hhat + f * h)

        # Route-health stats reduced to scalars INSIDE the (compiled) cell, so the
        # full (B, N, K) gate tensor never leaves the fused kernel and the .mean()/
        # entropy reductions are fused rather than run eagerly per tick.
        active = p.mean()                                                 # open-fraction
        pc = p.clamp(1e-6, 1.0 - 1e-6)
        gate_ent = (-(pc * pc.log() + (1.0 - pc) * (1.0 - pc).log())).mean()
        return h_new, active, gate_ent


class DSRGTrunk(nn.Module):
    """obs -> input-neuron states -> T weight-tied routed ticks -> global readout -> H."""

    def __init__(self, obs_dim, H, args):
        super().__init__()
        self.n = args.n_neurons
        self.d = args.node_dim
        self.T = args.n_ticks
        self.n_input = min(args.n_input_neurons, args.n_neurons)
        self.grad_checkpoint = args.grad_checkpoint
        self.lambda_route = args.lambda_route
        self.lambda_gate_ent = args.lambda_gate_ent
        self.target_active = args.route_target_active

        # Shared FIXED topology (the wiring is constant); per-tick weights/routing
        # live in T distinct cells (tie_ticks=True collapses to one shared cell =
        # v1's weight-tied recurrence). A depth-T routed graph matches the baseline's
        # T-independent-block capacity instead of reusing one low-capacity cell.
        src = build_topology(args.n_neurons, args.fan_in, args.seed)
        if args.tie_ticks:
            shared = RoutedGraphCell(args.n_neurons, args.node_dim, src, args)
            self.cells = nn.ModuleList([shared] * self.T)        # same params reused each tick
        else:
            self.cells = nn.ModuleList(
                [RoutedGraphCell(args.n_neurons, args.node_dim, src, args) for _ in range(self.T)]
            )

        # obs enters at the first layer (input neurons only); no distributed encode.
        self.encoder = layer_init(nn.Linear(obs_dim, self.n_input * self.d), std=1.0)
        # Learned nonzero initial state for ALL neurons (input neurons get
        # overwritten by the obs encoder at t=0). Nonzero so tick-0 RMSNorm is
        # well-conditioned (vs zero-init -> 1/sqrt(eps) gradient blowup).
        self.h0 = nn.Parameter(torch.randn(self.n, self.d))
        # Per-(tick, neuron) embedding: every perceptron gets its own code for the
        # current tick. Per-neuron (not a broadcast (T,d) vector) so its gradient
        # scale matches other per-neuron params -- a globally-broadcast tick bias
        # accumulates grad N-fold and dominates the shared grad-norm clip.
        self.tick_emb = nn.Parameter(torch.zeros(self.T, self.n, self.d))

        # global readout over all neuron states.
        self.out_norm = nn.RMSNorm(self.n * self.d, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(self.n * self.d, H))

        self.route_loss = torch.zeros(())   # stashed each forward; read by the update loop

    def forward(self, x):
        B = x.shape[0]
        h = self.h0.unsqueeze(0).expand(B, -1, -1).contiguous()
        h[:, :self.n_input] = self.encoder(x).view(B, self.n_input, self.d)

        p_sum = x.new_zeros(())          # accumulate gate open-fraction / entropy over ticks
        ent_sum = x.new_zeros(())
        for t in range(self.T):
            cell = self.cells[t]
            tick_t = self.tick_emb[t]
            Ksrc = cell.key[cell.src]         # (N, K, dk) per-cell keys; gathered once per tick
            if self.grad_checkpoint and self.training:
                h, active_t, ent_t = checkpoint(cell, h, tick_t, Ksrc, use_reentrant=False)
            else:
                h, active_t, ent_t = cell(h, tick_t, Ksrc)
            p_sum = p_sum + active_t
            ent_sum = ent_sum + ent_t

        active = p_sum / self.T
        gate_ent = ent_sum / self.T
        # route-balance BUDGET (hold the open MEAN at target) + entropy PENALTY
        # (push each gate to {0,1}). Together: ~target-fraction of edges commit OPEN,
        # the rest commit CLOSED -> genuine sparse routing. v1 had the entropy as a
        # BONUS, which pinned every gate at the 0.5 max-entropy point (inert).
        self.route_loss = self.lambda_route * (active - self.target_active) ** 2 + self.lambda_gate_ent * gate_ent
        self._active = active.detach()
        self._gate_ent = gate_ent.detach()

        feat = self.out_proj(self.out_norm(h.reshape(B, self.n * self.d)))
        return feat


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            self.trunk = DSRGTrunk(obs_dim, H, args)
        else:
            self.critic_trunk = DSRGTrunk(obs_dim, H, args)
            self.actor_trunk = DSRGTrunk(obs_dim, H, args)
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
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

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


def shape_advantage(gae, sigma, u, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform."""
    if args.adv_transform == "v10":
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
                    # DUAL-BACKWARD decoupled clipping. Value grad and policy grad
                    # (incl. the route structural-prior, which rides with the policy)
                    # are backpropped + clipped to their own max-norms, then summed
                    # on the shared trunk.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss - ent_coef_eff * entropy_loss + route_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef + route_loss
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Route-health gauges read ONCE here (last minibatch's values), not per
        # minibatch -- the per-mb float() was a host sync stalling the GPU 2x/mb.
        route_active = float(route_trunk._active)
        route_ent = float(route_trunk._gate_ent)
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

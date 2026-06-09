# CircuitNet v2 — K-slot blackboard (Global Workspace) over the shared expert pool.
#
# Spec: specs/circuit_net_spec.md (§4.2 K-slot blackboard backlog item; §2 instrumentation;
# Appendix A defaults). This is the §3 PRE-REGISTERED TIE-BREAKER: if v1 (single D=256
# workspace) ties/loses the matched-FLOP dense control while routing_mi + delta_ratio are
# healthy (machinery works, score capped), the gate is blind to the single-workspace
# bottleneck — v2 relieves it with K parallel workspace slots before declaring the
# representation axis dead. Successor to CircuitNet v1.
#
# IDEA: v1 carried ONE workspace w in R^{B,D} through T ticks. v2 carries K SLOTS
# W in R^{B,K,D} (K = n_slots, default 4) over the SAME shared expert pool + router
# (one pool reused across both ticks AND parallel slots — the reuse substrate widened).
# Each tick: per-slot routed expert update, then a single cross-slot multi-head
# self-attention "blackboard" lets slots read/write each other (the only cross-slot
# mixing — O(K^2*D), tiny at K=4; NO expert<->expert edges, so the no-O(N^2) property
# holds). A learned-query cross-attention reads the K final slots down to one (B,D)
# vector for the (unchanged) Beta / HL-Gauss head.
#
#   obs --encoder--> W[:,k,:] = encoder(obs) + slot_emb[k]   (W in R^{B,K,D})
#   for t in 1..T:                         (experts+router SHARED across ticks AND slots)
#       w_hat = LayerNorm(W) + phase_emb[t]    (flattened to B*K; per-step PHASE code)
#       logits = router(w_hat)  (+ warmup noise sigma*randn, sigma=max(0,1-step/75k))
#       top-2 by logit; softmax over the 2 kept logits only
#       Delta = sum_{e in top2} gate_e * Expert_e(w_hat)      (reshaped back to B,K,D)
#       W = W + gamma * Delta                  (per-channel LayerScale gamma, init 0.1)
#       W = W + slot_attn(LayerNorm(W))        (BLACKBOARD: cross-slot MHSA broadcast)
#   head_in = LayerNorm( readout_attn(query=readout_query, kv=W_T) )   (learned-query readout)
#
# MECHANISM: (a) each expert now sees K* more selection signal per tick (route_loss /
# metrics accumulate over B*K rows) -> denser per-expert gradient; (b) K parallel
# pathways + cross-slot attention = parallel-circuit expressivity past the single-D cap.
#
# ANTI-COLLAPSE PACKAGE (from step 0): batch-level load-balance (coef 0.01),
# router z-loss (coef 1e-3), warmup logit noise. SINGLE shallow-start attenuation:
# expert W_out init 1e-3*randn (so Delta~0 at t=0, net starts as encoder->readout);
# gamma stays a healthy 0.1 (NOT double-attenuated). Phase emb is the only mechanism
# that genuinely breaks the weight-tied iterated-map fixed point. slot_emb breaks
# slot symmetry so slots specialize.
#
# SEPARATE actor & critic trunks (non-negotiable): the HL-Gauss critic CE gradient on a
# shared trunk specializes experts toward value features -> weak actor. Each trunk owns
# its encoder/experts/router/dense head. Actor head -> Beta(alpha,beta); critic head ->
# HL-Gauss symlog categorical bins (both byte-for-byte v15 math).
#
# FALSIFIABILITY (§2, logged under circuit/): routing_mi (conditionality), eff_rank
# (illusory-width detector), dead_expert_frac, delta_ratio[t], mean_gamma + diagnostics.
# ReDo (§1.6): every 250k steps reset dormant experts + zero their Adam moments.
# CONTROL BATTERY (§3): dense_control (no routing, matched FLOPs), frozen_router
# (fixed random top-2, no router grad), n_ticks=1 (no recurrence).
#
# Kept from v1 (byte-for-byte): PPO loop, GAE, Beta actor, HL-Gauss symlog critic,
# obs/reward norm, env wrappers, triple-backward decoupled grad-clip, control-battery
# flags (dense_control/frozen_router), ReDo, Args/tyro pattern.
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
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: Optional[float] = None  # v1: KL early-stop disabled (un-throttle the actor)

    # separate actor/critic trunks + decoupled (triple-backward) gradient clipping.
    share_backbone: bool = False     # v1: separate CircuitNetTrunk for actor and critic (NON-NEGOTIABLE)
    separate_grad_clip: bool = True  # clip policy-, value-, and route-gradients to their own norms
    actor_grad_clip: float = 0.5     # max-norm for the policy gradient (incl. its actor-trunk part)
    critic_grad_clip: float = 0.5    # max-norm for the value gradient (incl. its critic-trunk part)
    route_grad_clip: float = 0.5     # max-norm for the DECOUPLED route-structural gradient (both routers)

    # Distributional critic support. Tight + well-resolved.
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ~ N(0, tau^2), sharp at 0
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 2.0  # Dreamer4 / hl_gauss_pytorch default

    # Advantage shaping (v19). See header / shape_advantage. v1: "none" = raw GAE (no rankgauss).
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

    # --- CircuitNet trunk (recurrent sparse MoE over a residual workspace; Appendix A) ---
    d_model: int = 256               # D: workspace / residual-stream width
    n_experts: int = 64              # M: expert pool size (top-k routed; FLOPs decoupled from M)
    expert_hidden: int = 512         # h: SwiGLU expert hidden width (h = 2D)
    top_k: int = 2                   # k: experts kept per step by top-k routing
    n_ticks: int = 2                 # T: recurrent steps (experts weight-tied across them)
    n_slots: int = 4                 # K: blackboard workspace slots (v2: real K-slot Global Workspace)
    gamma_init: float = 0.1          # LayerScale gamma init (per-channel residual-branch scale)
    lb_coef: float = 0.01            # batch-level load-balance aux coef
    z_coef: float = 1e-3             # router z-loss coef (anti logit blow-up)
    route_noise_steps: int = 75000   # warmup logit noise sigma=max(0,1-step/route_noise_steps)
    redo_interval: int = 250000      # ReDo dormant-expert reset cadence (env-steps)
    redo_tau: float = 0.025          # ReDo / dead-expert dormancy threshold
    compile: bool = True             # torch.compile the (fixed-shape, dense-all-M) trunk forward

    # --- control battery (§3; run concurrently for the go/no-go gate) ---
    dense_control: bool = False      # True: no routing -- all M experts as a plain residual MLP (matched active FLOPs)
    frozen_router: bool = False      # True: fixed random top-2, router gets NO gradient (and is dropped from route_params)

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


class SwiGLUExpert(nn.Module):
    """A single SwiGLU expert: D -> h -> D, out = W_out( SiLU(W_gate(x)) * W_in(x) ).
    W_out is the SINGLE shallow-start attenuation (init 1e-3*randn) so Delta~=0 at t=0
    and the trunk starts as encoder->readout. W_in/W_gate use standard init."""

    def __init__(self, d, h):
        super().__init__()
        self.W_in = nn.Linear(d, h)
        self.W_gate = nn.Linear(d, h)
        self.W_out = nn.Linear(h, d)
        # standard init for in/gate; near-zero output projection (the one attenuation).
        nn.init.zeros_(self.W_in.bias)
        nn.init.zeros_(self.W_gate.bias)
        with torch.no_grad():
            self.W_out.weight.copy_(1e-3 * torch.randn_like(self.W_out.weight))
            nn.init.zeros_(self.W_out.bias)

    def forward(self, x):
        return self.W_out(F.silu(self.W_gate(x)) * self.W_in(x))


class CircuitNetTrunk(nn.Module):
    """obs -> encoder -> K-slot blackboard W in R^{B,K,D}; T recurrent steps of top-k
    sparse-MoE over each slot (experts+router SHARED across ticks AND slots), with a
    cross-slot multi-head self-attention "blackboard" broadcast each tick. A learned-query
    cross-attention reads the K final slots down to one (B,D) DENSE head input. §4.2 + §1.2.

    MoE forward = dense-compute-all-M then mask to top-2 (no gather/scatter at M=64),
    run on the flattened (B*K, D) rows so the shared pool sees every slot. Anti-collapse
    package (LB + z-loss + warmup logit noise) is first-class from step 0, accumulated over
    the B*K routing decisions. Per-step phase_emb makes the weight-tied T-step selection an
    ORDERED circuit; slot_emb breaks slot symmetry; slot_attn is the ONLY cross-slot mixing.

    Stashes (read by the update loop / metrics each forward):
      self.route_loss            : lb_coef*(sum_t LB_t)/T + z_coef*(sum_t z_t)/T  (over B*K rows)
      self._gate_full            : (T, B*K, M) full gate-mass per step (0 off top-k)
      self._expert_out_stack     : (M, B*K, D) per-expert output on the LAST step (for eff_rank)
      self._expert_absh          : (T, M) per-expert mean |hidden| over B*K rows (for ReDo)
      self._delta_ratio          : (T,) mean over (B,K) of ||gamma*Delta_t|| / ||W||
      self._router_logits_last   : (B*K, M) last-step logits (router_entropy diagnostic)
      self._slot_attn_entropy     : () mean entropy of the cross-slot attention weights
    """

    def __init__(self, obs_dim, args):
        super().__init__()
        self.D = args.d_model
        self.M = args.n_experts
        self.h = args.expert_hidden
        self.T = args.n_ticks
        self.K = args.n_slots                         # number of blackboard slots
        self.top_k = min(args.top_k, args.n_experts)
        self.lb_coef = args.lb_coef
        self.z_coef = args.z_coef
        self.route_noise_steps = args.route_noise_steps
        self.dense_control = args.dense_control
        self.frozen_router = args.frozen_router

        self.encoder = layer_init(nn.Linear(obs_dim, self.D), std=1.0)
        # M SwiGLU experts, weight-tied across T AND across the K slots (one shared pool).
        self.experts = nn.ModuleList([SwiGLUExpert(self.D, self.h) for _ in range(self.M)])
        self.router = nn.Linear(self.D, self.M)
        nn.init.zeros_(self.router.bias)
        self.pre_ln = nn.LayerNorm(self.D)            # pre-router LN, applied each step
        self.attn_ln = nn.LayerNorm(self.D)           # pre-blackboard LN (cross-slot attn input)
        self.final_ln = nn.LayerNorm(self.D)          # LayerNorm(readout) -> head input
        self.phase_emb = nn.Parameter(torch.zeros(self.T, self.D))   # per-step ordered code
        self.gamma = nn.Parameter(torch.full((self.D,), args.gamma_init))  # LayerScale, per-channel

        # K learned slot identities (break symmetry so slots specialize). Trunk params.
        self.slot_emb = nn.Parameter(0.02 * torch.randn(self.K, self.D))
        # BLACKBOARD: single-layer cross-slot multi-head self-attention (slots as K tokens).
        # O(K^2*D) — the only cross-slot mixing; NOT a router (still trains under frozen_router).
        self.slot_attn = nn.MultiheadAttention(self.D, num_heads=4, batch_first=True)
        # READOUT: one learned query attends over the K final slots -> one (B,D) vector.
        self.readout_attn = nn.MultiheadAttention(self.D, num_heads=4, batch_first=True)
        self.readout_query = nn.Parameter(torch.randn(self.D) * 0.02)

        # global_step is injected by the training loop each forward (for warmup noise).
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.route_loss = torch.zeros(())   # stashed each forward; read by the update loop

    def router_params(self):
        """Router weight+bias (the only params the route_loss / decoupled route clip touch).
        Empty when frozen_router (no router grad) or dense_control (router unused)."""
        if self.frozen_router or self.dense_control:
            return []
        return [self.router.weight, self.router.bias]

    def _all_experts(self, w_hat):
        # Dense compute of ALL M experts on the flattened (B*K, D) rows (shared pool over
        # ticks AND slots). Returns (M, B*K, D) and (M, B*K) |hidden|.
        outs, absh = [], []
        for e in self.experts:
            hid = F.silu(e.W_gate(w_hat)) * e.W_in(w_hat)   # (B*K, h)
            outs.append(e.W_out(hid))                       # (B*K, D)
            absh.append(hid.abs().mean(dim=-1))             # (B*K,) per-row mean |hidden|
        return torch.stack(outs, dim=0), torch.stack(absh, dim=0)   # (M,B*K,D), (M,B*K)

    def forward(self, x):
        B = x.shape[0]
        K = self.K
        BK = B * K
        gstep = self.global_step
        sigma = (1.0 - gstep.float() / self.route_noise_steps).clamp_min(0.0)

        # Init the K-slot blackboard: W[:,k,:] = encoder(obs) + slot_emb[k].
        enc = self.encoder(x)                               # (B, D)
        W = enc.unsqueeze(1) + self.slot_emb.unsqueeze(0)   # (B, K, D)

        lb_sum = W.new_zeros(())
        z_sum = W.new_zeros(())
        gate_full_steps, absh_steps, delta_ratio_steps = [], [], []
        attn_entropy_steps = []
        last_expert_out = None
        last_logits = None

        for t in range(self.T):
            # ---- per-slot routed update: flatten (B,K,D) -> (B*K, D), shared pool/router ----
            w_hat = self.pre_ln(W) + self.phase_emb[t]      # (B, K, D) pre-LN + ordered phase code
            w_hat = w_hat.reshape(BK, self.D)               # (B*K, D)
            expert_out, absh = self._all_experts(w_hat)     # (M, B*K, D), (M, B*K)
            absh_steps.append(absh.mean(dim=1))             # (M,) per-expert mean |hidden| over B*K

            if self.dense_control:
                # CONTROL: no expert routing. All M experts summed as a plain residual MLP
                # per slot (top_k = M effectively, no router grad, no LB/z). Matched active
                # FLOPs only when n_experts is set to the control's matched count by the
                # caller. The K-slot blackboard + cross-slot attention is KEPT.
                delta = expert_out.sum(dim=0)               # (B*K, D)
                gate_full = expert_out.new_zeros(BK, self.M)  # routing metrics undefined -> zeros
            else:
                logits = self.router(w_hat)                 # (B*K, M)
                if self.frozen_router:
                    logits = logits.detach()                # fixed random top-k, no router grad
                if self.training and sigma > 0:
                    logits = logits + sigma * torch.randn_like(logits)   # warmup logit noise
                last_logits = logits
                # top-k by logit; softmax over the kept logits ONLY.
                topv, topi = logits.topk(self.top_k, dim=-1)             # (B*K, k)
                gate_k = torch.softmax(topv, dim=-1)                     # (B*K, k) over kept logits
                # scatter the kept gate mass back to full M (0 elsewhere).
                gate_full = logits.new_zeros(BK, self.M).scatter(1, topi, gate_k)   # (B*K, M)
                # Delta = sum_{e in topk} gate_e * Expert_e(w_hat); dense-all-M masked to top-k.
                delta = (gate_full.t().unsqueeze(-1) * expert_out).sum(dim=0)        # (B*K, D)

                # --- anti-collapse aux (batch-level over B*K rows, summed over T then /T) ---
                P = torch.softmax(logits, dim=-1).mean(0)                # (M,) full-M soft mass
                onehot = logits.new_zeros(BK, self.M).scatter(1, topi, 1.0)
                f = onehot.mean(0).detach()                             # (M,) hard load (detached)
                lb_sum = lb_sum + self.M * (f * P).sum()
                z_sum = z_sum + torch.logsumexp(logits, dim=-1).pow(2).mean()

            gate_full_steps.append(gate_full.detach())
            scaled_delta = (self.gamma * delta).reshape(B, K, self.D)   # (B, K, D)
            W = W + scaled_delta                            # LayerScale residual write
            last_expert_out = expert_out

            # ---- BLACKBOARD: cross-slot multi-head self-attention broadcast ----
            # slots are the K-length token sequence; the ONLY cross-slot mixing (O(K^2*D)).
            attn_in = self.attn_ln(W)                       # (B, K, D)
            attn_out, attn_w = self.slot_attn(
                attn_in, attn_in, attn_in, need_weights=True, average_attn_weights=True
            )                                               # (B, K, D), (B, K, K)
            W = W + attn_out                                # residual blackboard read/write
            # mean entropy of the cross-slot attention weights (over B and query slots).
            attn_entropy_steps.append(
                (-(attn_w.clamp_min(1e-9) * attn_w.clamp_min(1e-9).log()).sum(-1)).mean()
            )
            # delta_ratio numerator per step: ||gamma*Delta|| per (B,K) row (the expert write).
            delta_ratio_steps.append(scaled_delta.norm(dim=-1))         # (B, K)

        # ---- READOUT: one learned query attends over the K final slots -> (B, D) ----
        query = self.readout_query.view(1, 1, self.D).expand(B, 1, self.D)  # (B, 1, D)
        readout, _ = self.readout_attn(query, W, W, need_weights=False)     # (B, 1, D)
        head_in = self.final_ln(readout.squeeze(1))         # (B, D) head input

        W_norm = W.norm(dim=-1).clamp_min(1e-8)             # (B, K) per-slot residual-stream norm

        # stash route_loss (both terms /T) -- zero in dense_control.
        self.route_loss = self.lb_coef * (lb_sum / self.T) + self.z_coef * (z_sum / self.T)
        # stash metrics tensors (detached) for the probe/logging path.
        self._gate_full = torch.stack(gate_full_steps, dim=0)           # (T, B*K, M)
        self._expert_out_stack = last_expert_out.detach()              # (M, B*K, D) last step
        self._expert_absh = torch.stack(absh_steps, dim=0).detach()     # (T, M)
        self._delta_ratio = torch.stack(
            [dr.mean() / W_norm.mean() for dr in delta_ratio_steps]     # (T,) mean over (B,K)
        ).detach()
        self._router_logits_last = (last_logits.detach() if last_logits is not None
                                    else W.new_zeros(BK, self.M))
        self._slot_attn_entropy = torch.stack(attn_entropy_steps).mean().detach()
        return head_in


@torch.no_grad()
def circuit_metrics(trunk):
    """§2 falsifiability instrumentation, computed from a single probe forward's stashes.

    routing_mi   : H(mean_b q_b) - mean_b H(q_b), q_b in R^M = per-(sample,slot) top-k
                   gate-mass — each (sample,slot) row of the B*K decisions is one "b".
                   We average over the T steps to get one q_b per row, then take the entropy
                   gap. Natural log (nats).
    eff_rank     : exp(entropy of normalized singular values) of the stacked expert-output
                   matrix (M x D) -- the per-expert mean output over the probe (B*K) rows
                   (the §2 "stacked expert outputs over a probe batch" reading). Illusory-
                   width / collapse detector.
    dead_expert_frac : fraction of experts with dormancy s_e = E|h_e| / mean_e E|h_e| < tau.
    delta_ratio[t] : mean over (B,K) of ||gamma*Delta_t|| / ||W|| per step.
    plus mean_gamma, router_entropy, expert_util_entropy, slot_attn_entropy diagnostics.
    """
    M = trunk.M
    gate_full = trunk._gate_full                       # (T, B, M) gate masses (0 off top-k)
    # per-sample q_b = mean over steps of the per-step top-k gate-mass, renormalized to a dist.
    q = gate_full.mean(dim=0)                          # (B, M)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    eps = 1e-9
    H_mean = -(q.mean(0).clamp_min(eps) * q.mean(0).clamp_min(eps).log()).sum()       # H(mean_b q_b)
    H_each = -(q.clamp_min(eps) * q.clamp_min(eps).log()).sum(-1).mean()              # mean_b H(q_b)
    routing_mi = (H_mean - H_each).item()
    routing_mi_frac = routing_mi / log(M)

    # eff_rank of the stacked per-expert mean output (M x D).
    em = trunk._expert_out_stack.mean(dim=1)           # (M, D) per-expert mean output over batch
    sv = torch.linalg.svdvals(em.float())              # (min(M,D),)
    p = sv / sv.sum().clamp_min(eps)
    eff_rank = torch.exp(-(p.clamp_min(eps) * p.clamp_min(eps).log()).sum()).item()

    # dormancy s_e over the probe batch (mean over T steps of per-expert mean |hidden|).
    absh = trunk._expert_absh.mean(dim=0)              # (M,)
    s = absh / absh.mean().clamp_min(eps)              # (M,) dormancy (dead_expert_frac via tau in caller)

    # diagnostics
    mean_gamma = trunk.gamma.mean().item()
    logits = trunk._router_logits_last                 # (B, M)
    rp = torch.softmax(logits, dim=-1)                 # (B, M)
    router_entropy = (-(rp.clamp_min(eps) * rp.clamp_min(eps).log()).sum(-1)).mean().item()
    util = q.mean(0)                                    # (M,) population usage distribution
    expert_util_entropy = (-(util.clamp_min(eps) * util.clamp_min(eps).log()).sum()).item()

    # cross-slot blackboard attention entropy (do slots share or ignore each other?).
    slot_attn_entropy = trunk._slot_attn_entropy.item()

    out = dict(
        routing_mi=routing_mi,
        routing_mi_frac=routing_mi_frac,
        eff_rank=eff_rank,
        mean_gamma=mean_gamma,
        router_entropy=router_entropy,
        expert_util_entropy=expert_util_entropy,
        slot_attn_entropy=slot_attn_entropy,
    )
    return out, s          # return dormancy scores too (for dead_expert_frac + ReDo)


@torch.no_grad()
def redo_reset(trunk, dormancy, tau, optimizer):
    """§1.6 ReDo: for each dormant expert (s_e < tau) re-init W_in/W_gate (standard init),
    set W_out -> 1e-3*randn, and ZERO that expert's Adam exp_avg/exp_avg_sq state (keep step).
    `dormancy` is the (M,) per-expert score from circuit_metrics(trunk). Returns #reset."""
    dead = (dormancy < tau).nonzero(as_tuple=False).flatten().tolist()
    n_reset = 0
    for ei in dead:
        e = trunk.experts[ei]
        # re-init the three projections (standard init for in/gate; 1e-3*randn for out).
        e.W_in.reset_parameters()
        e.W_gate.reset_parameters()
        nn.init.zeros_(e.W_in.bias)
        nn.init.zeros_(e.W_gate.bias)
        e.W_out.weight.copy_(1e-3 * torch.randn_like(e.W_out.weight))
        nn.init.zeros_(e.W_out.bias)
        # zero the per-expert Adam moment rows (leave `step`). Match the param tensors by id.
        for p in (e.W_in.weight, e.W_in.bias, e.W_gate.weight, e.W_gate.bias,
                  e.W_out.weight, e.W_out.bias):
            st = optimizer.state.get(p, None)
            if st:
                if "exp_avg" in st:
                    st["exp_avg"].zero_()
                if "exp_avg_sq" in st:
                    st["exp_avg_sq"].zero_()
        n_reset += 1
    return n_reset



class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        D = args.d_model
        # §1.1: SEPARATE actor & critic trunks (non-negotiable). Each owns its own
        # slot_emb / slot_attn / readout_query (separate K-slot blackboards).
        assert not args.share_backbone, "CircuitNet v2 requires share_backbone=False (separate trunks)"
        self.share_backbone = False
        self.critic_trunk = CircuitNetTrunk(obs_dim, args)
        self.actor_trunk = CircuitNetTrunk(obs_dim, args)
        # Categorical critic with a PEAKED init (sharp zero-return prior). Bias in
        # the categorical coordinate (symlog-space when enabled). DENSE head off D.
        self.critic_head = layer_init(nn.Linear(D, args.num_bins), std=0.1)
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
            self.actor_head = layer_init(nn.Linear(D, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(D, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            self.actor_alpha_head = layer_init(nn.Linear(D, act_dim), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(D, act_dim), std=0.01)
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
        # Separate trunks: each CircuitNetTrunk returns its own LayerNorm(w_T) head input (B, D).
        return self.actor_trunk(x), self.critic_trunk(x)

    def route_loss(self):
        # BOTH routers get their aux gradient: LB+z from the actor trunk AND the critic trunk.
        return self.actor_trunk.route_loss + self.critic_trunk.route_loss

    def get_value(self, x):
        # Critic trunk only: avoid a wasted actor-trunk forward whose sole effect
        # would be clobbering actor_trunk's route_loss/metric stashes mid-rollout.
        return self.critic_head(self.critic_trunk(x))

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
        # actor_trunk (encoder + experts + phase + gamma + router + LNs + slot_emb +
        # slot_attn + readout_attn + readout_query) + actor head. The blackboard params are
        # standard trunk params (NOT route_params) -> clipped+updated via the actor backward.
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(self.actor_trunk.parameters()) + heads

    def critic_parameters(self):
        # critic_trunk + critic head.
        return list(self.critic_trunk.parameters()) + list(self.critic_head.parameters())

    def route_parameters(self):
        # Router weight/bias of BOTH trunks (deduped by identity). The decoupled route
        # backward clips the LB+z gradient on ONLY these. Empty for a trunk under
        # frozen_router (that router takes no gradient). The router params are ALSO in
        # actor_parameters()/critic_parameters() (so the task/CE gradient reaches them);
        # the triple-backward sums those gradients -- the route group only adds the aux,
        # matching v15's grouping discipline.
        seen, ps = set(), []
        for trunk in (self.actor_trunk, self.critic_trunk):
            for p in trunk.router_params():
                if id(p) not in seen:
                    seen.add(id(p))
                    ps.append(p)
        return ps


def shape_advantage(gae, sigma, u, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform."""
    if args.adv_transform in ("none", "v10"):
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
        # Compile the CircuitNetTrunk.forward (fixed shapes, dense-compute-all-M, fixed K=4
        # cross-slot attention -> no recompiles, T=2 -> no grad-checkpoint). donated_buffer
        # off: the triple-backward backprops value/route with retain_graph=True, incompatible
        # with donated buffers. Stashing self._* attrs (and the MultiheadAttention calls) may
        # graph-break but still runs correctly; if compile fails at all, fall back to eager
        # via suppress_errors / the except (never break the run).
        import torch._functorch.config
        import torch._dynamo
        torch._functorch.config.donated_buffer = False
        # suppress_errors: if dynamo can't trace some op, fall back to eager for that
        # region instead of raising -- guarantees compile never breaks the run.
        torch._dynamo.config.suppress_errors = True
        try:
            for tr in (agent.actor_trunk, agent.critic_trunk):
                tr.forward = torch.compile(tr.forward)
        except Exception as ex:  # pragma: no cover - compile is best-effort
            print(f"[circuitnet] torch.compile failed ({ex}); running eager.")
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
    last_redo = 0          # ReDo guard: last 250k-boundary at which a reset ran
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # propagate the env-step counter into both trunks (drives the warmup logit noise).
        agent.actor_trunk.global_step.fill_(global_step)
        agent.critic_trunk.global_step.fill_(global_step)
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
                    # max-norm, then summed. route_loss is the (LB+z) aux from BOTH trunks'
                    # routers, applied (decoupled-clipped) to the router params only so it
                    # can't be drowned by the policy/value grad-norm; the TASK gradient still
                    # reaches the routers via the actor/critic backward (they are in
                    # actor_params/critic_params), keeping routing plastic and task-adaptive.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    # route_loss has no grad path under dense_control/frozen_router (or empty
                    # route_params) -> skip its backward to avoid a "does not require grad" error.
                    if route_params and route_loss.requires_grad:
                        route_loss.backward(retain_graph=True)
                        route_gn = nn.utils.clip_grad_norm_(route_params, args.route_grad_clip)
                        route_grads = [(p, p.grad.detach().clone()) for p in route_params if p.grad is not None]
                    else:
                        route_gn = torch.zeros((), device=device)
                        route_grads = []
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

        # §2 circuit instrumentation + ReDo, computed off the hot path on a probe batch.
        # A single eval forward per trunk repopulates the metric stashes (warmup noise is
        # train-only so eval gives a clean read). routing_mi is the kill-switch metric.
        probe = b_obs[:256]
        agent.eval()
        with torch.no_grad():
            agent.actor_trunk(probe)
            agent.critic_trunk(probe)
        a_cm, a_dorm = circuit_metrics(agent.actor_trunk)
        c_cm, c_dorm = circuit_metrics(agent.critic_trunk)
        agent.train()

        # ReDo: at most once per redo_interval boundary, reset dormant experts in BOTH trunks.
        if global_step - last_redo >= args.redo_interval:
            last_redo += args.redo_interval
            n_redo = (redo_reset(agent.actor_trunk, a_dorm, args.redo_tau, optimizer)
                      + redo_reset(agent.critic_trunk, c_dorm, args.redo_tau, optimizer))
            writer.add_scalar("circuit/redo_reset_count", n_redo, global_step)

        # circuit/ tags: report actor trunk as primary (the throttled path), critic as _critic.
        cm = dict(a_cm)
        cm["dead_expert_frac"] = (a_dorm < args.redo_tau).float().mean().item()
        for ti, dr in enumerate(agent.actor_trunk._delta_ratio.tolist()):
            cm[f"delta_ratio_t{ti}"] = dr
        cm_critic = dict(c_cm)
        cm_critic["dead_expert_frac"] = (c_dorm < args.redo_tau).float().mean().item()
        for ti, dr in enumerate(agent.critic_trunk._delta_ratio.tolist()):
            cm_critic[f"delta_ratio_t{ti}"] = dr

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
        writer.add_scalar("losses/route_grad_norm", float(route_gn), global_step)
        for _k, _v in cm.items():
            writer.add_scalar(f"circuit/{_k}", _v, global_step)
        for _k, _v in cm_critic.items():
            writer.add_scalar(f"circuit/{_k}_critic", _v, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

# PPO + DepthMix attn v3: MULTI-SLOT depth readout + attention-entropy-over-depth logging.
#
# WHAT'S NEW vs v2 (ONLY the "attn" path; seq/dwa unchanged).
#  (1) MULTI-SLOT depth readout. v2's depth-attention used a SINGLE query per block
#      (q = Wq(LN(s))) -> ONE per-head convex average of the depth bank: a rank-1
#      bottleneck (each head can only read one weighted blend of past layers). v3
#      generalizes to S learned query SLOTS per block: a per-block slot_embed
#      (S, n_heads, d_qk) is ADDED to the shared query base, giving S distinct queries
#      Q[b,s] = Wq(LN(s)) + slot_embed[s]. Each slot attends over depth independently
#      (its own softmax-over-L), producing S contexts; these are concatenated and mixed
#      back into the SINGLE residual stream by Wo: (S*H)->H, so s = s + Wo(concat) keeps
#      the residual-ADD identity path exact. Slots start at small DISTINCT random values
#      so they specialize rather than collapse. This breaks v2's single-query convex-
#      average bottleneck -> richer, higher-rank retrieval from the depth bank.
#      slots=1 recovers v2's single-query behavior exactly (Wo is then H->H).
#      COST: Wo grows H->H to (S*H)->H, so the trunk grows ~+0.4M params at S=4
#      (H=128, 8 blocks); that capacity confound is reported, not hidden.
#  (2) ATTENTION-ENTROPY-OVER-DEPTH logging. To KNOW whether the attention actually
#      reaches back over depth or just collapses onto the most-recent bank entry
#      (bank[-1]), the per-block depth-softmax entropy H_block = -sum_L w*log w (mean
#      over batch/heads/slots, raw nats) is computed under no_grad each forward and
#      logged: charts/attn_depth_entropy (mean over blocks) + block0/last representative
#      values. High entropy (~log L) => broad reach-back; ~0 => collapse to one layer.
#
# Everything else (PPO/GAE, Beta actor, HL-Gauss symlog critic, advantage transforms,
# separate-trunk default, decoupled clip, optimizer, hyperparameters) is inherited
# byte-for-byte from v2. seq/dwa modes do NOT use slots and skip the entropy logging.
#
# ----- v2 rationale (retained) -----------------------------------------------------
# PPO + DepthMix attn v2: STABILIZED depth-attention residual trunk.
#
# WHY. depthmix_v1's "attn" mode collapsed (exploding grads / KL). Two structural
# faults: (a) the bank entries feeding Wq/Wk/Wv were NOT layer-normalized, so their
# magnitude grew with depth and inflated the attention logits/values; (b) the
# attention output REPLACED the residual stream (`h = mix + body(LN(mix))`, where
# `mix = Wo(ctx)`), destroying the identity skip — the stream norm compounded
# multiplicatively across depth.
#
# FIX (this file, ONLY the "attn" path). Restructure to the standard pre-LN
# transformer-decoder-over-depth form: a single persistent residual stream `s` with
# TWO pre-LN residual-ADD sublayers per block. NO residual throttling (no
# LayerScale / ReZero / small-init gates — those bound the stream but slow learning).
# Per block i:
#   sublayer 1 (depth-attention): q = Wq(LN_q(s)); K,V = Wk/Wv(LN_kv(stack(bank)));
#       softmax over depth; s = s + Wo(ctx)                    (residual ADD)
#   sublayer 2 (FFN):            s = s + body(LN_ffn(s))       (residual ADD)
#   bank.append(s)
# Crucially LayerNorm is applied ONLY to BRANCH INPUTS (q/kv/ffn), NEVER to the main
# stream `s` itself, so the identity path is exact and every branch output is bounded
# (its input is unit-norm). Output = final_ln(s). Optional QK-norm (--qk-norm) L2-
# normalizes q/K per head with a per-block learnable logit scale (init so the effective
# scale matches 1/sqrt(d_qk)). The "seq" and "dwa" modes are UNCHANGED from v1
# (they don't explode; kept as available controls).
#
# HYPOTHESIS. With bounded branch outputs + an exact identity skip, attention over
# previous layers trains stably (no exploding grads/KL), so the depth-attention
# mechanism can be fairly evaluated against the param-matched plain MLP rather than
# diverging before it can learn.
#
# The PPO/GAE loop, Beta actor, HL-Gauss symlog distributional critic, advantage
# transforms, separate-trunk default, max_grad_norm/decoupled clip, optimizer, and
# all hyperparameters are inherited byte-for-byte from depthmix_v1 / the v24 beta
# symlog base; ONLY the trunk's attention path is restructured.
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
    exp_name: str = "depthmix_attn_v3"
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
    share_backbone: bool = True      # one DepthMixTrunk for both actor and critic heads
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

    # DepthMix trunk (plain MLP + depth-mixing residuals; NO MoE).
    #   reach_mode: "seq" (sequential residual) | "dwa" (DenseFormer static add)
    #               | "attn" (input-dependent per-head depth attention).
    reach_mode: str = "attn"
    trunk_hidden: int = 128          # H (must be divisible by trunk_heads for attn)
    trunk_blocks: int = 8            # number of depth-mixing MLP blocks
    trunk_heads: int = 4             # attention heads (attn mode)
    trunk_slots: int = 4             # v3: S learned query slots per block (attn mode); 1 == v2
    trunk_d_qk: int = 32             # per-head query/key dim (attn mode)
    trunk_mlp_mult: int = 2          # body hidden width multiplier (m*H)
    qk_norm: bool = False            # attn: L2-normalize q/K per head + learnable logit scale

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


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _mlp_body(H, mlp_mult):
    # Plain pre-LN MLP body: Linear(H, m*H) -> ReLU^2 -> Linear(m*H, H).
    # Same ReLUSquared activation v24's expert body used. sqrt(2) init.
    return nn.Sequential(
        layer_init(nn.Linear(H, mlp_mult * H)),
        ReLUSquared(),
        layer_init(nn.Linear(mlp_mult * H, H)),
    )


class DepthMixTrunk(nn.Module):
    """Plain-MLP trunk with depth-mixing residuals (NO mixture-of-experts).

    Each block mixes over ALL previous layer outputs (the "bank") via one of:
      "seq" : plain sequential residual (reach back = 1).            [v1, unchanged]
      "dwa" : DenseFormer static depth-weighted add (input-independent softmax
              convex combination of all previous outputs).           [v1, unchanged]
      "attn": per-head depth attention over previous layers.         [v2, RESTRUCTURED]

    seq/dwa keep v1's form: mixed = _mix(i, bank); h = mixed + body(LN(mixed)).

    attn (v3) is the standard pre-LN transformer-decoder-over-depth form: ONE
    persistent residual stream `s` with TWO pre-LN residual-ADD sublayers per block —
    depth-attention then FFN. LayerNorm touches only BRANCH INPUTS (q/kv/ffn), never
    the stream `s`, so the identity skip is exact and unbounded growth is impossible
    (each branch sees a unit-norm input). NO residual throttling. Output = final_ln(s).
    The depth-attention sublayer uses S learned query SLOTS (`n_slots`): a per-block
    slot_embed (S, n_heads, d_qk) is added to the shared query base, each slot attends
    over depth independently, and the S contexts are concatenated and projected back to
    H by Wo: (S*H)->H (so the residual ADD is exact). n_slots=1 == v2's single query.
    Per forward it also records the depth-softmax entropy per block (diagnostic) in
    `_last_depth_entropy` (scalar mean over blocks) and `_last_depth_entropy_blocks`.
    Output dim = H.
    """

    def __init__(self, in_dim, H, n_blocks, reach_mode, n_heads, d_qk, mlp_mult, qk_norm=False, n_slots=1):
        super().__init__()
        assert reach_mode in ("seq", "dwa", "attn"), f"unknown reach_mode {reach_mode}"
        if reach_mode == "attn":
            assert H % n_heads == 0, f"H={H} must be divisible by n_heads={n_heads}"
            assert n_slots >= 1, f"n_slots={n_slots} must be >= 1"
        self.reach_mode = reach_mode
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_qk = d_qk
        self.qk_norm = qk_norm
        self.n_slots = n_slots
        # Diagnostics: most-recent-forward depth-softmax entropy (attn mode only).
        self._last_depth_entropy = None          # scalar (python float / 0-dim tensor) mean over blocks
        self._last_depth_entropy_blocks = None   # list of per-block scalars

        # No input LayerNorm: env wrappers already normalize observations.
        self.entry = layer_init(nn.Linear(in_dim, H))

        self.bodies = nn.ModuleList()    # plain MLP (FFN) body per block
        for _ in range(n_blocks):
            self.bodies.append(_mlp_body(H, mlp_mult))

        if reach_mode == "attn":
            # v2 add-form block: pre-LN the FFN branch input (not the stream).
            self.ln_ffn = nn.ModuleList([nn.LayerNorm(H) for _ in range(n_blocks)])
            # Pre-LN the depth-attention branch inputs: query (current stream) and
            # the key/value bank entries. This bounds the bank magnitude that feeds
            # Wq/Wk/Wv (v1's exploding root cause).
            self.ln_q = nn.ModuleList([nn.LayerNorm(H) for _ in range(n_blocks)])
            self.ln_kv = nn.ModuleList([nn.LayerNorm(H) for _ in range(n_blocks)])
            # Per-block depth-attention projections (no bias on q/k/v).
            self.Wq = nn.ModuleList(
                [layer_init(nn.Linear(H, n_heads * d_qk, bias=False)) for _ in range(n_blocks)]
            )
            self.Wk = nn.ModuleList(
                [layer_init(nn.Linear(H, n_heads * d_qk, bias=False)) for _ in range(n_blocks)]
            )
            self.Wv = nn.ModuleList(
                [layer_init(nn.Linear(H, H, bias=False)) for _ in range(n_blocks)]
            )
            # v3: S learned query slots per block. slot_embed is ADDED to the shared
            # per-head query base; init to small DISTINCT random values so slots start
            # specialized rather than collapsed. (n_slots=1 => one near-zero offset,
            # recovering v2's single query.)
            self.slot_embed = nn.ParameterList(
                [nn.Parameter(torch.randn(n_slots, n_heads, d_qk) * 0.02) for _ in range(n_blocks)]
            )
            # v3: Wo maps the S concatenated per-slot contexts (S*H) back to H. The
            # residual ADD s = s + Wo(concat) keeps the identity skip exact.
            self.Wo = nn.ModuleList(
                [layer_init(nn.Linear(n_slots * H, H)) for _ in range(n_blocks)]
            )
            if qk_norm:
                # Per-block, per-head learnable logit scale applied to the cosine
                # (L2-normalized q.K) scores. Init exp(logit_scale) = sqrt(d_qk) so
                # the effective initial scaling matches the standard 1/sqrt(d_qk):
                # cosine logits are O(1) after dividing by sqrt(d_qk) would shrink
                # them; instead we MULTIPLY normalized logits by exp(logit_scale),
                # initialized to sqrt(d_qk) to restore a comparable logit range.
                self.logit_scale = nn.ParameterList(
                    [nn.Parameter(torch.full((n_heads,), float(np.log(np.sqrt(d_qk)))))
                     for _ in range(n_blocks)]
                )
        else:
            # seq / dwa keep v1's per-block pre-LN before the body.
            self.lns = nn.ModuleList([nn.LayerNorm(H) for _ in range(n_blocks)])
            if reach_mode == "dwa":
                # One learned scalar per previous-output index. Block i (1-indexed)
                # mixes a bank of length i (indices 0..i-1), so alpha_i has i entries.
                self.alphas = nn.ParameterList(
                    [nn.Parameter(torch.zeros(i + 1)) for i in range(n_blocks)]
                )

        self.final_ln = nn.LayerNorm(H)

    def _mix(self, i, bank):
        # bank: list of (B, H) tensors, length i+1 (the running state is bank[-1]).
        # Only used by seq / dwa (attn has its own add-form forward).
        if self.reach_mode == "seq":
            return bank[-1]
        # dwa
        w = torch.softmax(self.alphas[i], dim=0)                  # (L,)
        stack = torch.stack(bank, dim=1)                          # (B, L, H)
        return (w.view(1, -1, 1) * stack).sum(dim=1)              # (B, H)

    def _attn_forward(self, x):
        # Stable pre-LN ADD-form depth-attention trunk with S learned query SLOTS.
        # `s` is the persistent residual stream; `bank` snapshots its state after each
        # block. LayerNorm is applied ONLY to branch inputs (q_in / kv / ffn), so `s`
        # flows through an exact identity skip (no throttle) and every branch output is
        # bounded. Each of the S slots attends over depth independently; the S contexts
        # are concatenated and Wo: (S*H)->H projects them back to the stream.
        nh, dqk, ns = self.n_heads, self.d_qk, self.n_slots
        s = self.entry(x)                                          # (B, H)
        B, H = s.shape
        dv = H // nh
        bank = [s]                                                 # snapshots of the stream
        depth_entropy_blocks = []                                  # per-block depth-softmax entropy (nats)
        for i in range(self.n_blocks):
            L = len(bank)
            # --- sublayer 1: multi-slot depth-attention (pre-LN, residual ADD) ---
            q_in = self.ln_q[i](s)                                 # (B, H)
            kv = self.ln_kv[i](torch.stack(bank, dim=1))           # (B, L, H)
            qbase = self.Wq[i](q_in).view(B, nh, dqk)              # (B, nh, dqk)
            # Broadcast S slot offsets onto the shared query base: (B, S, nh, dqk).
            Q = qbase.unsqueeze(1) + self.slot_embed[i].unsqueeze(0)  # (B, S, nh, dqk)
            K = self.Wk[i](kv).view(B, L, nh, dqk)                 # (B, L, nh, dqk)
            V = self.Wv[i](kv).view(B, L, nh, dv)                  # (B, L, nh, dv)
            if self.qk_norm:
                Q = F.normalize(Q, dim=-1)
                K = F.normalize(K, dim=-1)
                scale = self.logit_scale[i].exp().view(1, 1, nh, 1)   # (1, 1, nh, 1)
                scores = torch.einsum("bshd,blhd->bshl", Q, K) * scale  # (B, S, nh, L)
            else:
                scores = torch.einsum("bshd,blhd->bshl", Q, K) / (dqk ** 0.5)
            w = torch.softmax(scores, dim=-1)                      # over depth L; (B, S, nh, L)
            ctx = torch.einsum("bshl,blhd->bshd", w, V)            # (B, S, nh, dv)
            ctx = ctx.reshape(B, ns * H)                           # concat over slots -> (B, S*H)
            s = s + self.Wo[i](ctx)                                # residual ADD (identity preserved)
            # Diagnostic: depth-softmax entropy (mean over batch, slots, heads). No grad.
            with torch.no_grad():
                ent = -(w * (w + 1e-9).log()).sum(dim=-1)         # (B, S, nh)
                depth_entropy_blocks.append(ent.mean())
            # --- sublayer 2: FFN (pre-LN, residual ADD) ---
            s = s + self.bodies[i](self.ln_ffn[i](s))              # residual ADD
            bank.append(s)
        # Stash most-recent-forward depth-attention entropy diagnostics.
        with torch.no_grad():
            self._last_depth_entropy_blocks = [float(e) for e in depth_entropy_blocks]
            self._last_depth_entropy = float(torch.stack(depth_entropy_blocks).mean())
        return self.final_ln(s)

    def forward(self, x):
        if self.reach_mode == "attn":
            return self._attn_forward(x)
        x0 = self.entry(x)
        bank = [x0]
        for i in range(self.n_blocks):
            mixed = self._mix(i, bank)
            h = mixed + self.bodies[i](self.lns[i](mixed))
            bank.append(h)
        return self.final_ln(bank[-1])


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.trunk_hidden
        self.share_backbone = args.share_backbone

        def _make_trunk():
            return DepthMixTrunk(
                obs_dim,
                H,
                args.trunk_blocks,
                args.reach_mode,
                args.trunk_heads,
                args.trunk_d_qk,
                args.trunk_mlp_mult,
                args.qk_norm,
                args.trunk_slots,
            )

        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = _make_trunk()
        else:
            self.critic_trunk = _make_trunk()
            self.actor_trunk = _make_trunk()
        # Categorical critic with a PEAKED init: small weight + Gaussian-logit
        # bias so the initial value distribution starts as a sharp zero-return
        # prior instead of a high-variance uniform distribution. The bias lives
        # in the categorical coordinate, which is symlog-space when enabled.
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
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

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
        # v3: attention-entropy-over-depth (attn mode only). Tells whether the depth
        # attention reaches broadly back over the bank (~log L) or collapses onto one
        # layer (~0). Read from the actor trunk (or the single shared trunk). seq/dwa
        # have no attention => the diagnostic is None and these scalars are skipped.
        diag_trunk = agent.trunk if agent.share_backbone else agent.actor_trunk
        depth_ent = getattr(diag_trunk, "_last_depth_entropy", None)
        if depth_ent is not None:
            writer.add_scalar("charts/attn_depth_entropy", depth_ent, global_step)
            ent_blocks = diag_trunk._last_depth_entropy_blocks
            if ent_blocks:
                writer.add_scalar("charts/attn_entropy_block0", ent_blocks[0], global_step)
                writer.add_scalar(f"charts/attn_entropy_block{len(ent_blocks) - 1}", ent_blocks[-1], global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

# TD7-CTX v3_3 — USER-SHAPED body (truncated end-to-end recompute). Third arm of a three-arm ablation.
# --------------------------------------------------------------------------------------------------
# THREE-ARM ABLATION — "accurate vs useful vs user-shaped representation". All three share the SAME
# decoder-only causal Transformer "body" over the ViT-style patch-token stream (n_ctx_layers=2,
# d_model=64, 4 heads x 16, FFN 256, RoPE on the absolute patch index, ctx_window W=2560, per-layer KV
# cache, 250-step full_reprefill). They differ ONLY in WHAT SHAPES THE BODY'S REPRESENTATION:
#   v3_1  PREDICTIVE aux : body trained by a self-supervised LM (next patch-token + next reward-sum).
#                          -> an ACCURATE forward model of the token stream.
#   v3_2  VALUE-AWARE aux: body trained by a reward-to-go + critic-distillation LM.
#                          -> a USEFUL (value-predictive) representation.
#   v3_3  USER-SHAPED    : the body makes NO predictions of its own. Its representation is shaped ONLY by
#         (THIS FILE)     its user's (the RL critic's) gradients, via TRUNCATED end-to-end recompute.
#                          -> a representation shaped purely by downstream utility to the critic.
#
# IDEA (shared base): this is td7_hist_v1 (obs augmented with the last hist_k executed actions) EXACTLY,
# plus a "context" pathway. Every patch_len=50 CONTIGUOUS steps within an episode is compressed into one
# 64-d PATCH token. The actor/critic form a query token from their normalized first feature and run it as
# the FINAL token of the causal stream (last-token readout): it attends the last W=2560 cached patch
# positions via the KV-cache, composing token->token (induction / in-context learning).
#
# WHAT v3_3 CHANGES vs v3_1 (everything else is byte-identical):
#  (1) THE LM IS GONE. No lm_tok_head / lm_rsum_head / lm_optimizer / lm_update. The PatchEncoder aux
#      (per-patch reward-sum regression + live-token refresh) is UNCHANGED — it still trains the encoder
#      and refreshes store.tokens; only the body's self-prediction objective is removed.
#  (2) TRUNCATED END-TO-END RECOMPUTE (decode_truncated). The ONLINE critic's context readout gives the
#      RL TD gradient a REAL path into the body's attention+FFN composition (not just the query
#      projection). For a readout query at abs patch position p attending the band [p-W, p):
#        * FROZEN PREFIX [p-W, p-k_live): read from the frozen KV cache, DETACHED (no grad; truncation).
#        * LIVE SEGMENT [p-k_live, p): K/V RECOMPUTED IN-GRAPH from the (detached) patch tokens through
#          the CURRENT body. Each live position i attends (a) the earlier live positions [p-k_live, i]
#          in-graph AND (b) the frozen prefix [i-W, p-k_live) detached, under a SINGLE JOINT softmax over
#          the union (NOT two separate softmaxes) — reconstructing i's TRUE full-window context so the
#          next layer's live K/V are exact. The readout at p then attends [frozen prefix detached] +
#          [live segment in-graph], one joint softmax. Gradient flows from the critic TD loss through the
#          readout + the in-graph live K/V into the body params (q/k/v/out, FFN, LN), truncated at the
#          frozen prefix. Implementation = per-b gather of the W-wide frozen band below the live segment
#          (detached cache) concatenated with the <=k_live in-graph live K/V, then ONE dense masked softmax
#          over the (W + k_live) union. (NOT the block-sparse FlexAttention decode() uses: flex's EAGER
#          backward materializes a dense score matrix over the FULL padded cache (~24k keys) -> multi-GB for
#          k_live+1 queries; the W-band gather bounds keys to the true window (~2624). decode() keeps flex —
#          it has a single query.) COST: the (B, k_live+1) x (W+k_live) recompute (fwd+bwd) dominates the
#          step. k_live is a pure grad-depth/speed knob -- the FORWARD stays exact for any k_live (the
#          frozen prefix covers the remainder), so lowering it cuts cost ~linearly in the query count
#          without breaking equivalence. DEFAULT k_live=32 (~1600 steps of gradient-carrying context;
#          ~half the recompute cost + backward transient of 64). See the solo cuda-synced bench for the
#          measured ms/step + peak memory at 32.
#  (3) OPTIMIZER: the body params are in the CRITIC optimizer ONLY. So critic_optimizer.step() moves the
#      body; actor_optimizer.step() does not (body excluded from it, and any actor-deposited body grad is
#      wiped by the next critic zero_grad). This is the whole ablation: the body is USER-SHAPED.
#  (4) GRAD CLIP (v3_3-specific): between critic_loss.backward() and the critic step we clip the BODY
#      grad-norm to 0.5 for truncated-BPTT stability. Critic params are NOT clipped -- TD7/v3_1 critic
#      dynamics are preserved, and the no-context floor holds exactly on the same substrate every prior
#      number was measured on. debug/body_grad_norm reports the POST-clip body grad norm.
#
# ASYMMETRY (deliberate, flagged): the TARGET path (body_target over the frozen cache) stays FULLY FROZEN
# / detached — cheap and stable. The ACTOR path (actor loss + its critic Q) uses the CHEAP fully-frozen
# decode() (the body is NOT moved by the actor). So the critic's ONLINE context is a truncated-recompute
# readout while the actor's context is a frozen-cache readout: actor-ctx(frozen) vs critic-online-ctx
# (recompute) is an approximation we accept (the body is trained to be useful to the critic Q, and the
# actor maximizes that same Q, so their contexts are consistent in expectation across a 250-step block).
#
# KV-CACHE + 250-STEP RE-PREFILL (unchanged): per-layer POST-RoPE K and raw V are cached for every patch
# position (live + frozen copies). add_patch does a cheap incremental prefill_append; the 250-step hard
# update calls full_reprefill(), rebuilding both caches exactly and snapshotting body_target. Online RL
# reads the FROZEN cache with the live body (critic: truncated-recompute; actor: frozen); TARGET RL reads
# the FROZEN cache with body_target. Acting (select_action) reads the LIVE cache with the live body.
#
# WHY THE td7_hist FLOOR IS GUARANTEED: the actor/critic out projections are ZERO-INITIALIZED (weight AND
# bias). At init `ctx == 0` regardless of body internals (decode_truncated returns a finite hidden even
# for empty context -> 0*hidden = 0, no NaN), so the network is bitwise td7_hist_v1. `debug/ctx_gate`
# (actor out-proj weight norm) tracks whether the pathway ever moved.
#
# ENCODER (unchanged): PatchEncoder flattens the zero-padded 50x step block, appends frac_valid, maps
# Linear->ELU->Linear->AvgL1Norm, trained OFF the RL path by an aux head predicting the z-scored per-patch
# reward-sum. The body consumes these patch tokens (detached) as its sequence.
# --------------------------------------------------------------------------------------------------
# Base method header (td7_hist / TD7) retained for reference:
# TD7 = TD3 + SALE (dynamics-grounded state / state-action embeddings the actor & critic condition on
# via frozen snapshots) + LAP (prioritized replay with a bias-cancelling Huber loss) + policy
# checkpointing (burst training at episode boundaries; evaluate the best-worst-case policy) + target
# value clipping (TD targets clamped to the running [min,max] snapshot). Requires num_envs=1.
import copy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.tensorboard import SummaryWriter

# Patch-token cache capacity. HalfCheetah-v4 at 8M steps produces ~160000 full patches (8M/50);
# 220000 leaves headroom for early-terminating envs whose partial patches inflate the count. Writes
# and reads beyond it are clamped (with a one-time warning); overflow positions degrade to the ctx=0 floor.
MAX_PATCHES = 220000

# --- v3_1 DECODE: block-sparse FlexAttention (the only change vs v3) -------------------------------
# v3's decode ran a plain two-GEMM over ONE shared cache span covering the whole batch's position
# spread (~22.5k wide for a 1M-step buffer), of which each query only needs its own ctx_window=2560
# band -> ~89% of the B*H*S softmax was masked waste (~3.4 ms/decode forward, ~24.7 ms/step).
# v3_1 replaces ONLY the per-layer decode attend with torch.nn.attention.flex_attention: a block-sparse
# sliding-window + causal + validity mask over the cache broadcast (stride-0 expand, no copy) across the
# batch. Each query is ONE token at its own absolute position; keys/values are the per-layer cache.
# NUMERICS are identical to v3's masked scaled-dot-product to float tolerance (RoPE already applied to q
# and cached K; V raw; scale = head_dim**-0.5; fully-masked rows -> exactly 0.0, matching _attend).
#
# RECOMPILE CONTROL: create_block_mask is torch.compile'd (eager mask-build is ~10 ms and would dominate;
# compiled is ~0.1 ms), and flex_attention is torch.compile'd for the block-sparse kernel. The per-step
# drift of n_written and qpos is fed in as TENSORS (never python ints), so dynamo does NOT specialize on
# their values -> zero per-step recompiles. The only shape guards are the batch size B (256 replay / 1
# act) and the BUCKETED KV_LEN (n_written rounded up to a multiple of DECODE_KV_BUCKET, capped at
# max_patches), so KV_LEN takes only a handful of distinct values over a run -> a handful of compiles.
# The mask is positional-only (independent of the K/V values and layer) so it is built ONCE per decode
# and reused across both layers. Backward flows to the query (in_proj) through flex_attention; the cache
# is a constant (no grad). DEVIATION vs v3: FlexAttention does not cheaply expose the attention
# distribution, so the last-layer attention-entropy telemetry (debug/ctx_attn_entropy) is reported as 0.
_flex_attention_c = torch.compile(flex_attention)
_create_block_mask_c = torch.compile(create_block_mask)
DECODE_KV_BUCKET = 4096  # KV_LEN bucket granularity; keeps compiled shapes on a small fixed grid


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # --- PERF (opt-in) ---
    compile: bool = False
    """torch.compile the hot compute cores (critic/actor/aux); default off keeps the eager code path"""
    compile_mode: str = "default"
    """torch.compile mode when --compile is set. NOTE: "reduce-overhead" (CUDA graphs) is unsafe here —
    the three interleaved forward/backward cores per step overwrite each other's cudagraph static pool
    ("accessing tensor output of CUDAGraphs that has been overwritten"). "default" (inductor fusion, no
    cudagraphs) is the validated mode; "max-autotune-no-cudagraphs" is an alternative."""

    # --- HIST ---
    hist_k: int = 4
    """number of most-recent executed actions appended to the observation"""

    # --- CTX (ViT-style patch tokens + decoder-only causal Transformer body) ---
    ctx_window: int = 2560
    """sliding-window size: a query at abs position i attends cached positions [i-ctx_window, i]"""
    patch_len: int = 50
    """steps per patch token (patches never cross episode boundaries)"""
    tok_dim: int = 64
    """dimensionality of a patch token = transformer d_model (= n_heads * head_dim)"""
    n_heads: int = 4
    """transformer attention heads (head_dim = tok_dim // n_heads = 16)"""
    n_ctx_layers: int = 2
    """number of decoder-only transformer layers in the shared history body"""
    ffn_dim: int = 256
    """FFN inner width of each transformer layer"""
    raw_patches: int = 20480
    """ring capacity of raw patch step blocks kept for the aux encoder + LM losses"""
    aux_batch: int = 32
    """patches sampled per auxiliary encoder update"""
    lm_windows: int = 4
    """(v3_3: UNUSED — the LM is removed; kept only so v3_1 configs/tests load unchanged)"""
    lm_window_len: int = 256
    """(v3_3: UNUSED — the LM is removed; kept only so v3_1 configs/tests load unchanged)"""
    k_live: int = 32
    """v3_3 truncated recompute: # most-recent cache positions recomputed IN-GRAPH per critic readout
    (frozen prefix below this is detached). Truncation length of the end-to-end body BPTT."""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments (TD7 requires 1, see header)"""
    learning_starts: int = 25000
    """timesteps of uniform-random actions before training starts"""
    use_checkpoints: bool = True
    """train in episode-boundary bursts and evaluate the checkpointed best-worst-case policy"""
    eval_freq: int = 5000
    """evaluate every N env steps"""
    eval_eps: int = 10
    """number of evaluation episodes"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_update_rate: int = 250
    """hard-update the target networks and fixed encoders every N training steps"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    target_policy_noise: float = 0.2
    """the scale of target policy smoothing noise"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    policy_freq: int = 2
    """the frequency of training the policy (delayed)"""
    lap_alpha: float = 0.4
    """LAP prioritization exponent"""
    min_priority: float = 1.0
    """LAP minimum priority (and Huber loss threshold)"""
    max_eps_when_checkpointing: int = 20
    """episodes to assess a policy before checkpointing (after steps_before_checkpointing)"""
    steps_before_checkpointing: int = 750000
    """training steps of early exploration before full-length checkpoint assessment begins"""
    reset_weight: float = 0.9
    """discount applied to best_min_return when switching to full checkpoint assessment"""
    zs_dim: int = 256
    """dimensionality of the SALE embeddings"""
    hidden_dim: int = 256
    """hidden layer width of all networks"""
    encoder_lr: float = 3e-4
    """the learning rate of the encoder optimizer"""
    critic_lr: float = 3e-4
    """the learning rate of the critic optimizer"""
    actor_lr: float = 3e-4
    """the learning rate of the actor optimizer"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def lap_huber(td_loss, min_priority=1.0):
    return torch.where(td_loss < min_priority, 0.5 * td_loss.pow(2), min_priority * td_loss).sum(1).mean()


class ReturnNorm:
    """Welford running mean/std of raw episode returns; the aux target is the clipped z-score."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)

    @property
    def std(self):
        if self.count < 2:
            return 1.0
        return float(np.sqrt(self.M2 / self.count))

    def normalize(self, x):
        return float(np.clip((x - self.mean) / (self.std + 1e-6), -5.0, 5.0))


class LAPBuffer:
    """LAP replay buffer: priorities live on-device, sampling by inverse-CDF over the priority cumsum.

    CTX addition: an int64 `patch_idx` column = number of completed patches at the time the transition
    was collected. Its context window is the token span [patch_idx-ctx_window, patch_idx).
    """

    def __init__(self, state_dim, action_dim, device, max_size, batch_size, max_action):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.patch_idx = np.zeros((self.max_size, 1), dtype=np.int64)

        self.priority = torch.zeros(self.max_size, device=device)
        self.max_priority = 1.0

        self.normalize_actions = float(max_action)

    def add(self, state, action, next_state, reward, done, patch_idx):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.patch_idx[self.ptr] = patch_idx

        self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        csum = torch.cumsum(self.priority[: self.size], 0)
        val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
        self.ind = torch.searchsorted(csum, val).cpu().data.numpy()

        return (
            torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.patch_idx[self.ind].reshape(-1), dtype=torch.long, device=self.device),
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())


class PatchEncoder(nn.Module):
    """Compress a zero-padded patch (B, patch_len, step_dim) -> a tok_dim token, ViT-style: flatten the
    block, append frac_valid (len/patch_len), then Linear->ELU->Linear->AvgL1Norm. No pooling. Trained
    OFF the RL gradient path by an aux head predicting the z-scored per-patch reward-sum."""

    def __init__(self, step_dim, patch_len=50, tok_dim=64, hdim=128):
        super().__init__()
        self.patch_len = patch_len
        self.step_dim = step_dim
        self.l1 = nn.Linear(patch_len * step_dim + 1, hdim)
        self.l2 = nn.Linear(hdim, tok_dim)
        self.aux = nn.Linear(tok_dim, 1)

    def forward(self, patches, frac_valid):
        # patches: (B, patch_len, step_dim) zero-padded; frac_valid: (B,) or (B,1) in [0,1]
        B = patches.shape[0]
        x = torch.cat([patches.reshape(B, -1), frac_valid.reshape(B, 1)], dim=1)
        h = F.elu(self.l1(x))
        return avg_l1_norm(self.l2(h))  # (B, tok_dim)

    def predict_return(self, tok):
        return self.aux(tok)


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


class HistoryTransformerLayer(nn.Module):
    """One pre-LN decoder-only block: full QKV attention (RoPE applied by the parent body) + FFN."""

    def __init__(self, d_model=64, n_heads=4, ffn=256):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(), nn.Linear(ffn, d_model))


class HistoryTransformer(nn.Module):
    """Shared decoder-only causal Transformer over the patch-token stream, with a per-layer KV cache.

    Three read paths, all using SPAN-MASKED attention against ONE contiguous slice of the shared cache:
      - prefill_append(token, pos): cheap incremental writer (causal-inclusive: pos attends [pos-W, pos]).
      - full_reprefill(...): exact chunked rebuild of both caches over the last raw_patches positions.
      - decode(query, positions, K, V): current-state query token at abs position `pos` attends STRICTLY
        earlier cached positions [pos-W, pos-1]; its own K/V are never written. Grads flow through the
        activations to the caller's in_proj; the cache tensors are constants (no grad).
    forward_seq(x, positions): self-contained full-causal forward (no cache) used by the LM objective.

    RoPE is applied to q and k at the ABSOLUTE patch position (fp64 angles -> fp32); V is NOT rotated.
    """

    def __init__(self, d_model=64, n_heads=4, n_layers=2, ffn=256, ctx_window=2560, max_patches=MAX_PATCHES):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_layers = n_layers
        self.ctx_window = ctx_window
        self.max_patches = max_patches
        self.scale = self.head_dim ** -0.5
        self.layers = nn.ModuleList([HistoryTransformerLayer(d_model, n_heads, ffn) for _ in range(n_layers)])
        # RoPE inverse frequencies (head_dim//2 pairs), kept in float64 for large-position precision.
        i = torch.arange(self.head_dim // 2, dtype=torch.float64)
        inv_freq = 10000.0 ** (-(2.0 * i / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x, pos):
        """x: (..., H, head_dim); pos: (...) integer absolute positions (leading dims of x sans H, Dh)."""
        ang = pos.to(torch.float64)[..., None, None] * self.inv_freq  # (..., 1, Dh/2) fp64
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        cos = torch.cat([cos, cos], dim=-1).to(x.dtype)  # (..., 1, Dh)
        sin = torch.cat([sin, sin], dim=-1).to(x.dtype)
        return x * cos + rotate_half(x) * sin

    def _attend(self, q_rope, qpos, K_layer, V_layer, n_written, lo_off, need_ent=False):
        """Span-masked attention. q_rope: (B, H, Dh); qpos: (B,); K_layer/V_layer: (P, d_model).
        lo_off=0 -> causal-inclusive (prefill, key delta in [0, W]); lo_off=1 -> strictly earlier (decode)."""
        B, H, Dh = q_rope.shape
        W = self.ctx_window
        bounds = torch.stack([qpos.min(), qpos.max()])
        qmin, qmax = bounds.tolist()  # single host sync; needed for the contiguous cache slice
        lo = max(int(qmin) - W, 0)
        hi = min(int(qmax) + 1 - lo_off, n_written)  # decode drops the query's own position (lo_off=1)
        if hi <= lo:  # no valid keys for the whole batch (very early training)
            out = q_rope.new_zeros(B, H * Dh)
            return (out, q_rope.new_zeros(())) if need_ent else out
        span_K = K_layer[lo:hi].view(-1, H, Dh)  # (S, H, Dh)
        span_V = V_layer[lo:hi].view(-1, H, Dh)
        S = span_K.shape[0]
        key_pos = torch.arange(lo, lo + S, device=q_rope.device)  # (S,)
        logits = torch.einsum("bhd,shd->bhs", q_rope, span_K) * self.scale  # (B, H, S)
        delta = qpos[:, None] - key_pos[None, :]  # (B, S)
        valid = (delta >= lo_off) & (delta <= W)  # key_pos < n_written already guaranteed by the hi clamp
        no_valid = ~valid.any(dim=1)  # (B,) rows with zero valid keys
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~valid[:, None, :], neg)  # broadcast over heads
        logits = logits.masked_fill(no_valid[:, None, None], 0.0)  # NaN guard: all-(-inf) rows -> uniform
        attn = torch.softmax(logits, dim=-1)  # (B, H, S)
        out = torch.einsum("bhs,shd->bhd", attn, span_V).reshape(B, H * Dh)
        out = out.masked_fill(no_valid[:, None], 0.0)  # zero-output empty rows
        if need_ent:
            ent_bh = -(attn.clamp_min(1e-12).log() * attn).sum(-1)  # (B, H)
            w = (~no_valid).float()
            ent = (ent_bh * w[:, None]).sum() / (w.sum() * H).clamp_min(1.0)
            return out, ent.detach()
        return out

    def _decode_block_mask(self, positions, n_written, B):
        """Build the block-sparse decode mask ONCE (identical across layers: positional only).
        Strictly-earlier sliding window + validity: for query b at abs position positions[b] and key at
        abs position kv_idx, attend iff 1 <= positions[b]-kv_idx <= ctx_window AND kv_idx < n_written.
        This is exactly _attend(lo_off=1)'s (delta in [1, W]) & (key_pos < n_written) condition. qpos and
        nw are passed as TENSORS so the compiled mask-build never specializes on their per-step values;
        N_bucket (a shape) is the only value guard. Returns (block_mask, N_bucket)."""
        W = self.ctx_window
        N_bucket = min(((int(n_written) + DECODE_KV_BUCKET - 1) // DECODE_KV_BUCKET) * DECODE_KV_BUCKET,
                       self.max_patches)
        qpos = positions.to(torch.long)  # (B,) captured as a lifted tensor input (not specialized)
        nw = torch.as_tensor(int(n_written), device=positions.device, dtype=torch.long)

        def mask_mod(b, h, q_idx, kv_idx):  # q_idx is always 0 (single query token)
            delta = qpos[b] - kv_idx
            return (delta >= 1) & (delta <= W) & (kv_idx < nw)

        return _create_block_mask_c(mask_mod, B, None, 1, N_bucket, device=positions.device), N_bucket

    def _flex_decode(self, q_rope, K_layer, V_layer, block_mask, N_bucket, B):
        """Block-sparse decode attend for ONE layer. q_rope: (B, H, Dh) (RoPE already applied); K_layer/
        V_layer: (P, d_model) cache (post-RoPE K, raw V), indexed by ABSOLUTE position from 0. The cache is
        viewed (N_bucket, H, Dh) and broadcast over the batch via stride-0 expand (no copy). Fully-masked
        rows return exactly 0.0 (flex safe-softmax), matching _attend's zeroed empty rows. Returns
        (B, H*Dh)."""
        H, Dh = self.n_heads, self.head_dim
        k = K_layer[:N_bucket].view(N_bucket, H, Dh).permute(1, 0, 2).unsqueeze(0).expand(B, H, N_bucket, Dh)
        v = V_layer[:N_bucket].view(N_bucket, H, Dh).permute(1, 0, 2).unsqueeze(0).expand(B, H, N_bucket, Dh)
        q = q_rope.unsqueeze(2)  # (B, H, 1, Dh)
        out = _flex_attention_c(q, k, v, block_mask=block_mask, scale=self.scale)  # (B, H, 1, Dh)
        return out.reshape(B, H * Dh)

    def decode(self, query, positions, K_cache, V_cache, n_written):
        """query: (B, d_model); positions: (B,) abs; K_cache/V_cache: (n_layers, P, d_model).
        Returns (hidden (B, d_model), last-layer attention entropy scalar).

        v3_1: block-sparse FlexAttention decode (numerically == v3's _attend(lo_off=1) to float tolerance).
        The attention-entropy telemetry is reported as 0 (flex does not cheaply expose the distribution)."""
        B = query.shape[0]
        H, Dh = self.n_heads, self.head_dim
        x = query
        ent = x.new_zeros(())  # entropy telemetry not available under flex -> logged as 0 (see header)
        empty = int(n_written) == 0  # no keys yet (very early training): every row's attn_out is 0
        block_mask = N_bucket = None
        if not empty:
            block_mask, N_bucket = self._decode_block_mask(positions, n_written, B)  # built once, reused
        for li, layer in enumerate(self.layers):
            h = layer.ln1(x)
            q = self._rope(layer.q_proj(h).view(B, H, Dh), positions)
            if empty:  # match _attend's zero-output empty rows (out_proj(0) still adds its bias)
                attn_out = x.new_zeros(B, H * Dh)
            else:
                attn_out = self._flex_decode(q, K_cache[li], V_cache[li], block_mask, N_bucket, B)
            x = x + layer.out_proj(attn_out)
            x = x + layer.ffn(layer.ln2(x))
        return x, ent

    def decode_truncated(self, query, positions, tokens, K_frozen, V_frozen, frozen_nw, k_live=32):
        """v3_3 TRUNCATED END-TO-END RECOMPUTE decode (used ONLY by the ONLINE critic; see header).

        Same OUTPUT SEMANTICS as decode(): the readout query at abs position p attends the strictly-earlier
        band [p-W, p) intersect [0, frozen_nw) (delta in [1, W]). BUT the most-recent k_live cache positions
        are RECOMPUTED IN-GRAPH from the current body so the critic TD gradient flows into the body's
        attention+FFN. The frozen prefix [p-W, p-k_live) is read from the frozen KV cache DETACHED (grads
        are truncated there). Returns (ctx (B, d_model), 0.0).

        APPROACH B (per-b gather + ONE dense JOINT softmax): for each b we gather the W-wide frozen band
        below the live segment ([p-k_live-W, p-k_live), detached) and concatenate the <=k_live in-graph live
        K/V; every readout/live query then does a single masked softmax over that (W + k_live) key set — one
        joint softmax over {frozen prefix} ∪ {live segment}, no online-merge. Live position i attends earlier
        live positions [p-k_live, i] (causal-inclusive) AND frozen [i-W, p-k_live); the readout at p attends
        all live positions (strictly earlier) AND its frozen prefix [p-W, p-k_live). k_live<<W so the within-
        live window is never truncated by W; the per-key delta mask slides the window exactly.
        (We do NOT use the block-sparse FlexAttention of decode() here: its EAGER backward materializes a
        dense (B,H,QL,bucketed_KV) score matrix over the FULL padded cache (~24k) -> multi-GB for QL=k_live+1;
        the per-b W-band gather bounds keys to the true window (~2624) instead. decode() keeps flex: QL=1.)

        Grad path: critic TD loss -> readout -> in-graph live K/V -> body params; truncated at the frozen
        prefix (detached cache K/V). Empty context (p<=0, k_live=0, or frozen_nw==0) -> finite zeros hidden
        -> the zero-init out projection floor holds bitwise (no NaN)."""
        B = query.shape[0]
        H, Dh = self.n_heads, self.head_dim
        W = self.ctx_window
        KL = int(k_live)
        nw = int(frozen_nw)
        dev = query.device
        if nw == 0 or KL <= 0:  # no cache / no live segment -> pure floor path (matches decode empty branch)
            x = query
            for layer in self.layers:
                x = x + layer.out_proj(x.new_zeros(B, H * Dh))
                x = x + layer.ffn(layer.ln2(x))
            return x, query.new_zeros(())
        P = tokens.shape[0]
        p = positions.to(torch.long)                                     # (B,) readout abs positions
        # --- live segment: abs positions [p-KL, p), detached patch tokens (in-graph from here) ---
        live_pos = p[:, None] - torch.arange(KL, 0, -1, device=dev)[None, :]   # (B, KL) = [p-KL, ..., p-1]
        live_valid = (live_pos >= 0) & (live_pos < nw)                   # (B, KL) keep decode's kv<nw semantics
        x_live = tokens[live_pos.clamp(0, P - 1)].detach()              # (B, KL, d_model) in-graph from here
        rope_lp = live_pos.clamp_min(0)                                  # safe RoPE positions (invalid rows masked)
        # --- frozen band: the W positions [p-KL-W, p-KL) below the live segment (covers every query's
        #     [i-W, p-KL) prefix); gathered per-b from the DETACHED cache and reused across layers. ---
        band_pos = (p[:, None] - KL) - torch.arange(W, 0, -1, device=dev)[None, :]  # (B, W) = [p-KL-W, ..., p-KL-1]
        band_valid = (band_pos >= 0) & (band_pos < nw)                   # (B, W) (band_pos < p-KL is guaranteed)
        band_idx = band_pos.clamp(0, P - 1)                             # (B, W) safe gather indices
        # --- combined key geometry: [frozen band (W)] ++ [live segment (KL)] ; QL queries = [live (KL)] ++ [readout] ---
        key_pos = torch.cat([band_pos, live_pos], dim=1)                # (B, W+KL) abs key positions
        key_valid = torch.cat([band_valid, live_valid], dim=1)          # (B, W+KL)
        QL = KL + 1
        allq_pos = torch.cat([live_pos, p[:, None]], dim=1)            # (B, QL), QL = KL+1
        # per-query lower delta bound: live queries causal-INCLUSIVE (attend self, lo=0), readout STRICT (lo=1)
        lo_off = torch.cat([torch.zeros(KL, device=dev), torch.ones(1, device=dev)])  # (QL,)
        delta = allq_pos[:, :, None] - key_pos[:, None, :]             # (B, QL, W+KL)
        mask = key_valid[:, None, :] & (delta >= lo_off[None, :, None]) & (delta <= W)  # (B, QL, W+KL)
        no_valid = ~mask.any(dim=-1)                                    # (B, QL) rows with no key (-> zero output)
        neg = torch.finfo(query.dtype).min

        xq = query  # readout token, evolved through the layers like decode()
        for li, layer in enumerate(self.layers):
            # live K/V/Q recomputed IN-GRAPH from the current live-segment representation
            h_live = layer.ln1(x_live)
            ql = self._rope(layer.q_proj(h_live).view(B, KL, H, Dh), rope_lp)   # (B, KL, H, Dh)
            kl = self._rope(layer.k_proj(h_live).view(B, KL, H, Dh), rope_lp)
            vl = layer.v_proj(h_live).view(B, KL, H, Dh)
            # readout Q recomputed IN-GRAPH at abs pos p
            qr = self._rope(layer.q_proj(layer.ln1(xq)).view(B, 1, H, Dh), p[:, None])  # (B, 1, H, Dh)
            allq = torch.cat([ql, qr], dim=1)                                   # (B, QL, H, Dh)
            # frozen band K/V from the DETACHED cache (grad truncated); post-RoPE K, raw V (as cached)
            kb = K_frozen[li].detach()[band_idx].view(B, W, H, Dh)             # (B, W, H, Dh)
            vb = V_frozen[li].detach()[band_idx].view(B, W, H, Dh)
            allk = torch.cat([kb, kl], dim=1)                                   # (B, W+KL, H, Dh)
            allv = torch.cat([vb, vl], dim=1)
            # ONE joint masked softmax over the union frozen ∪ live
            logits = torch.einsum("bqhd,bkhd->bhqk", allq, allk) * self.scale   # (B, H, QL, W+KL)
            logits = logits.masked_fill(~mask[:, None], neg)
            attn = torch.softmax(logits, dim=-1)
            out = torch.einsum("bhqk,bkhd->bhqd", attn, allv)                  # (B, H, QL, Dh)
            out = out.permute(0, 2, 1, 3).reshape(B, QL, H * Dh)               # (B, QL, d_model)
            out = out.masked_fill(no_valid[:, :, None], 0.0)                   # empty rows -> 0 (floor, no NaN)
            # residual + FFN (identical block structure to decode/forward_seq)
            x_live = x_live + layer.out_proj(out[:, :KL, :])
            x_live = x_live + layer.ffn(layer.ln2(x_live))
            xq = xq + layer.out_proj(out[:, KL, :])
            xq = xq + layer.ffn(layer.ln2(xq))
        return xq, query.new_zeros(())

    @torch.no_grad()
    def prefill_append(self, token, pos, K_live, V_live):
        """Incremental writer: run the new patch token at abs position `pos` through both layers, writing
        each layer's post-RoPE K and raw V into the LIVE cache. Uses CURRENT body weights."""
        H, Dh = self.n_heads, self.head_dim
        x = token.view(1, self.d_model)
        qpos = torch.full((1,), pos, device=x.device, dtype=torch.long)
        for li, layer in enumerate(self.layers):
            h = layer.ln1(x)
            q = self._rope(layer.q_proj(h).view(1, H, Dh), qpos)
            k = self._rope(layer.k_proj(h).view(1, H, Dh), qpos)
            v = layer.v_proj(h)  # (1, d_model) raw, un-RoPE'd
            K_live[li, pos] = k.reshape(self.d_model)  # write BEFORE attend -> causal-inclusive
            V_live[li, pos] = v.reshape(self.d_model)
            attn_out = self._attend(q, qpos, K_live[li], V_live[li], pos + 1, lo_off=0)
            x = x + layer.out_proj(attn_out)
            x = x + layer.ffn(layer.ln2(x))

    @torch.no_grad()
    def full_reprefill(self, tokens, n_patch, K_live, V_live, K_frozen, V_frozen, raw_patches, chunk=2048):
        """Rebuild both caches over the last min(n_patch, raw_patches) positions. Each layer is fully
        written before its (chunked, windowed causal) attention reads it, so the rebuild is EXACT for a
        fresh sequence (start==0; test f). When n_patch > raw_patches, queries near `start` attend the
        <=ctx_window band below `start` which is not rewritten this call -> those carry bounded-staleness
        (last-rewritten) values; this is the accepted staleness for the oldest resident positions."""
        n_written = min(int(n_patch), self.max_patches)
        if n_written == 0:
            return
        start = max(0, n_written - raw_patches)
        M = n_written - start
        H, Dh = self.n_heads, self.head_dim
        pos_abs = torch.arange(start, n_written, device=tokens.device)  # (M,)
        x = tokens[start:n_written].clone()  # (M, d_model) layer-0 input = detached patch tokens
        for li, layer in enumerate(self.layers):
            h = layer.ln1(x)
            q = self._rope(layer.q_proj(h).view(M, H, Dh), pos_abs)
            k = self._rope(layer.k_proj(h).view(M, H, Dh), pos_abs)
            v = layer.v_proj(h)  # (M, d_model)
            K_live[li, start:n_written] = k.reshape(M, self.d_model)
            V_live[li, start:n_written] = v
            attn_out = torch.empty(M, self.d_model, device=tokens.device)
            for c0 in range(0, M, chunk):
                c1 = min(c0 + chunk, M)
                attn_out[c0:c1] = self._attend(
                    q[c0:c1], pos_abs[c0:c1], K_live[li], V_live[li], n_written, lo_off=0
                )
            x = x + layer.out_proj(attn_out)
            x = x + layer.ffn(layer.ln2(x))
        # frozen := full snapshot of the live cache (identical contents over the written range)
        K_frozen[:, :n_written].copy_(K_live[:, :n_written])
        V_frozen[:, :n_written].copy_(V_live[:, :n_written])

    def forward_seq(self, x, positions):
        """Self-contained full-causal forward (no cache) over batched windows. x: (Bw, T, d_model);
        positions: (Bw, T) abs. Used by the LM objective (grads flow to body + LM heads)."""
        Bw, T, _ = x.shape
        H, Dh = self.n_heads, self.head_dim
        ti = torch.arange(T, device=x.device)
        delta = ti[:, None] - ti[None, :]  # (T, T)
        causal = (delta >= 0) & (delta <= self.ctx_window)  # True = attend (T<=ctx_window here)
        neg = torch.finfo(x.dtype).min
        for layer in self.layers:
            h = layer.ln1(x)
            q = self._rope(layer.q_proj(h).view(Bw, T, H, Dh), positions)
            k = self._rope(layer.k_proj(h).view(Bw, T, H, Dh), positions)
            v = layer.v_proj(h).view(Bw, T, H, Dh)
            logits = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale  # (Bw, H, T, T)
            logits = logits.masked_fill(~causal[None, None], neg)
            attn = torch.softmax(logits, dim=-1)
            out = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(Bw, T, self.d_model)
            x = x + layer.out_proj(out)
            x = x + layer.ffn(layer.ln2(x))
        return x


class Actor(nn.Module):
    """Adapter over the shared history body: in_proj forms the query token from a0; the decode output is
    read through a ZERO-INIT out projection (ctx == 0 at init -> td7_hist floor). The body + KV cache are
    passed in as arguments so actor.parameters() excludes the body (body is trained only by the LM)."""

    def __init__(self, state_dim, action_dim, tok_dim=64, n_heads=4, zs_dim=256, hdim=256, ctx_out=64):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.in_proj = nn.Linear(hdim, tok_dim)  # a0 -> query token
        self.out = nn.Linear(tok_dim, ctx_out)  # zero-init read-out of the decode hidden
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.l1 = nn.Linear(zs_dim + hdim + ctx_out, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs, body, K_cache, V_cache, positions, n_written):
        a0 = avg_l1_norm(self.l0(state))
        hidden, _ = body.decode(self.in_proj(a0), positions, K_cache, V_cache, n_written)
        ctx = self.out(hidden)  # (B, ctx_out); == 0 at init
        a = torch.cat([a0, zs, ctx], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = F.elu(self.zs1(state))
        zs = F.elu(self.zs2(zs))
        zs = avg_l1_norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        zsa = F.elu(self.zsa1(torch.cat([zs, action], 1)))
        zsa = F.elu(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class Critic(nn.Module):
    """Adapter over the shared history body (see Actor). A SINGLE shared ctx (queried by the q1-branch
    normalized feature, read through a zero-init out projection) is appended to BOTH twins."""

    def __init__(self, state_dim, action_dim, tok_dim=64, n_heads=4, zs_dim=256, hdim=256, ctx_out=64):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.in_proj = nn.Linear(hdim, tok_dim)  # q1f -> query token
        self.out = nn.Linear(tok_dim, ctx_out)  # zero-init read-out of the decode hidden
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.q1 = nn.Linear(2 * zs_dim + hdim + ctx_out, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim + ctx_out, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs, body, K_cache, V_cache, positions, n_written, decode_fn=None):
        # decode_fn: optional (query, positions, n_written) -> (hidden, ent). Default None uses the CHEAP
        # fully-frozen body.decode() (byte-identical to v3_1). The ONLINE critic core passes the v3_3
        # TRUNCATED-recompute decode so body grad flows (target/actor paths keep the frozen decode).
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1f = avg_l1_norm(self.q01(sa))
        # single shared context; ctx_ent is RETURNED (not stashed on self) so compiled cores stay pure.
        qtok = self.in_proj(q1f)
        if decode_fn is None:
            hidden, ctx_ent = body.decode(qtok, positions, K_cache, V_cache, n_written)
        else:
            hidden, ctx_ent = decode_fn(qtok, positions, n_written)
        ctx = self.out(hidden)  # (B, ctx_out); == 0 at init

        q1 = torch.cat([q1f, embeddings, ctx], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        q2f = avg_l1_norm(self.q02(sa))
        q2 = torch.cat([q2f, embeddings, ctx], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1), ctx_ent


class PatchStore:
    """Ring of raw patch blocks (for the aux loss) + a GPU token cache indexed by global patch id.

    tokens: LIVE cache, written at add_patch and refreshed by the aux encoder step. Replay forwards do
    NOT read patch tokens directly — they read the frozen KV cache (K_frozen/V_frozen), which the
    250-step full_reprefill rebuilds from `tokens`. So there is no separate frozen-token snapshot.
    """

    def __init__(self, step_dim, tok_dim, patch_len, encoder, device, raw_patches=20480, max_patches=MAX_PATCHES):
        self.step_dim = step_dim
        self.tok_dim = tok_dim
        self.patch_len = patch_len
        self.encoder = encoder
        self.device = device
        self.raw_patches = raw_patches
        self.max_patches = max_patches

        # raw ring: fixed (raw_patches, patch_len, step_dim) zero-padded blocks + valid length + reward-sum
        self.raw_blocks = np.zeros((raw_patches, patch_len, step_dim), dtype=np.float32)
        self.raw_lens = np.zeros(raw_patches, dtype=np.int64)
        self.raw_rsum = np.zeros(raw_patches, dtype=np.float32)
        self.raw_global_idx = np.full(raw_patches, -1, dtype=np.int64)

        self.tokens = torch.zeros(max_patches, tok_dim, device=device)
        self.n_patch = 0
        self._warned = False

    def add_patch(self, block, length, rsum):
        # block: (patch_len, step_dim) zero-padded; length: valid steps (>=1); rsum: patch reward-sum
        idx = self.n_patch
        slot = idx % self.raw_patches
        self.raw_blocks[slot] = block
        self.raw_lens[slot] = length
        self.raw_rsum[slot] = rsum
        self.raw_global_idx[slot] = idx

        with torch.no_grad():
            b = torch.as_tensor(block[None], dtype=torch.float32, device=self.device)  # (1, patch_len, step_dim)
            frac = torch.tensor([length / self.patch_len], dtype=torch.float32, device=self.device)
            tok = self.encoder(b, frac)[0]
        if idx < self.max_patches:
            self.tokens[idx] = tok
            written_idx = idx
        elif not self._warned:
            print(f"[PatchStore] n_patch exceeded MAX_PATCHES={self.max_patches}; clamping token writes/reads")
            self._warned = True
            self.tokens[self.max_patches - 1] = tok
            written_idx = self.max_patches - 1
        else:
            self.tokens[self.max_patches - 1] = tok
            written_idx = self.max_patches - 1
        self.n_patch += 1
        # return the global slot the token landed in (< max_patches) and the token, for KV prefill
        return written_idx, tok


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_action, args: Args, device, writer: SummaryWriter, step_dim):
        self.args = args
        self.device = device
        self.writer = writer
        self.ctx_window = args.ctx_window
        self.patch_len = args.patch_len

        self.actor = Actor(state_dim, action_dim, args.tok_dim, args.n_heads, args.zs_dim, args.hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, args.tok_dim, args.n_heads, args.zs_dim, args.hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # NB: critic_optimizer is built AFTER the body (v3_3: the body params are trained by the CRITIC
        # optimizer, so the body is USER-SHAPED by the critic TD gradient — see the body block below).

        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = LAPBuffer(state_dim, action_dim, device, args.buffer_size, args.batch_size, max_action)

        # CTX: patch encoder (trained off the RL path) + immutable token store + reward-sum normalizer
        self.tok_encoder = PatchEncoder(step_dim, args.patch_len, args.tok_dim).to(device)
        self.tok_encoder_optimizer = torch.optim.Adam(self.tok_encoder.parameters(), lr=3e-4)
        self.store = PatchStore(step_dim, args.tok_dim, args.patch_len, self.tok_encoder, device, args.raw_patches)
        self.patch_norm = ReturnNorm()
        self.last_aux_loss = 0.0

        # CTX v3_3: the SHARED decoder-only history body + a target copy (hard-updated with the RL targets).
        # The body is NOT a submodule of actor/critic (actor.parameters()/critic.parameters() exclude it),
        # but its params ARE registered in the CRITIC optimizer below: the body is USER-SHAPED, moved ONLY
        # by the critic TD gradient (via decode_truncated's truncated end-to-end recompute). No LM.
        self.body = HistoryTransformer(
            args.tok_dim, args.n_heads, args.n_ctx_layers, args.ffn_dim, args.ctx_window, MAX_PATCHES
        ).to(device)
        self.body_target = copy.deepcopy(self.body)
        # CRITIC optimizer OWNS the body params (v3_3). actor_optimizer excludes them, so actor steps never
        # move the body; any actor-deposited body grad is wiped by the next critic zero_grad (grad hygiene).
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.body.parameters()), lr=args.critic_lr
        )
        # v3_3-specific grad clip: clip the BODY params ONLY (clip_grad_norm_ at 0.5). Critic params stay
        # UNCLIPPED exactly like TD7/v3_1, so plain-TD7 critic dynamics (and the no-context floor: when the
        # ctx path contributes 0, v3_3 == td7_hist) are preserved on the substrate every prior number used.
        # The clip constrains truncated-BPTT gradients only where they arise (the new in-graph body path).
        self.clip_params = list(self.body.parameters())
        self.last_body_grad_norm = 0.0
        self.last_reprefill_ms = 0.0
        # positions actually populated in the FROZEN cache by the last full_reprefill. Replay-path decodes
        # must clamp to this (NOT the live n_patch) so they never attend never-snapshotted (zero) positions.
        self.frozen_n_written = 0

        # Per-layer KV cache (post-RoPE K, raw V), owned by the agent and shared by body/body_target.
        # LIVE: appended by add_patch (drifting weights) and rebuilt at each hard update; read by acting.
        # FROZEN: exact snapshot rebuilt at each hard update; read by ALL replay-path forwards.
        n_layers = args.n_ctx_layers
        self.K_live = torch.zeros(n_layers, MAX_PATCHES, args.tok_dim, device=device)
        self.V_live = torch.zeros(n_layers, MAX_PATCHES, args.tok_dim, device=device)
        self.K_frozen = torch.zeros(n_layers, MAX_PATCHES, args.tok_dim, device=device)
        self.V_frozen = torch.zeros(n_layers, MAX_PATCHES, args.tok_dim, device=device)

        # PERF: opt-in torch.compile of the pure-tensor hot cores. Default (off) binds the raw eager
        # methods so the compile=False code path is unchanged. select_action/eval stay eager.
        if args.compile:
            self.critic_core = torch.compile(self._critic_core, mode=args.compile_mode)
            self.actor_core = torch.compile(self._actor_core, mode=args.compile_mode)
            self.aux_core = torch.compile(self._aux_core, mode=args.compile_mode)
        else:
            self.critic_core = self._critic_core
            self.actor_core = self._actor_core
            self.aux_core = self._aux_core

        self.max_action = max_action
        self.training_steps = 0

        # checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # target value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0.0
        self.min_target = 0.0

    def n_written(self):
        """Number of written cache positions (<= MAX_PATCHES); cache index bound for span attention."""
        return min(self.store.n_patch, MAX_PATCHES)

    def add_patch(self, step_list):
        # step_list: list of raw step vecs (len 1..patch_len). Zero-pad to patch_len, compute reward-sum.
        n = len(step_list)
        a = np.asarray(step_list, dtype=np.float32)
        block = np.zeros((self.patch_len, self.store.step_dim), dtype=np.float32)
        block[:n] = a
        rsum = float(a[:, -1].sum())  # reward is the last column of a step record
        self.patch_norm.update(rsum)
        idx, tok = self.store.add_patch(block, n, rsum)
        # incremental KV write for this new position (current body weights); exactness restored by the
        # 250-step full_reprefill. Only positions < MAX_PATCHES are cached (overflow degrades to floor).
        if idx < MAX_PATCHES:
            self.body.prefill_append(tok, idx, self.K_live, self.V_live)

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
            # act path reads the LIVE cache with the live body; query position = current completed-patch count
            patch_idx = torch.tensor([self.store.n_patch], dtype=torch.long, device=self.device)
            nw = self.n_written()

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs, self.body, self.K_live, self.V_live, patch_idx, nw)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs, self.body, self.K_live, self.V_live, patch_idx, nw)

            if use_exploration:
                action = action + torch.randn_like(action) * self.args.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    # ---- hot compute cores (pure tensor in / tensor out; optionally torch.compile'd, see __init__) ----
    # CTX v3: TARGET forwards read body_target over the FROZEN cache; ONLINE forwards read the live body
    # over the FROZEN cache. Both are a consistent (weights, cache) pair snapshotted at the last hard
    # update. positions = the batch patch_idx; nw = number of written cache positions.
    def _critic_core(self, state, action, next_state, reward, not_done, noise,
                     positions, nw, min_t, max_t):
        """TD target + critic loss + LAP priority + telemetry scalars. No in-place cache/opt mutation.
        min_t/max_t are 0-d tensors (not python floats) so the target clip does NOT trigger recompiles."""
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)
            next_action = (
                self.actor_target(next_state, fixed_target_zs, self.body_target, self.K_frozen, self.V_frozen,
                                  positions, nw)
                + noise
            ).clamp(-1, 1)
            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)
            Qt, _ = self.critic_target(
                next_state, next_action, fixed_target_zsa, fixed_target_zs, self.body_target,
                self.K_frozen, self.V_frozen, positions, nw
            )
            Q_target = Qt.min(1, keepdim=True)[0]
            Q_target = reward + not_done * self.args.gamma * Q_target.clamp(min_t, max_t)
            qt_max = Q_target.max()
            qt_min = Q_target.min()
            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        # ONLINE critic: route the context readout through the v3_3 TRUNCATED end-to-end recompute so the
        # critic TD gradient flows into the body params (not just the query projection). The frozen prefix
        # is the frozen cache (detached); the last k_live positions are recomputed in-graph. positions/nw
        # match decode()'s masking (nw = frozen_n_written).
        def trunc_decode(query, positions_, nw_):
            return self.body.decode_truncated(
                query, positions_, self.store.tokens, self.K_frozen, self.V_frozen, nw_, self.args.k_live
            )

        Q, ctx_ent = self.critic(
            state, action, fixed_zsa, fixed_zs, self.body, self.K_frozen, self.V_frozen, positions, nw,
            decode_fn=trunc_decode,
        )
        td_loss = (Q - Q_target).abs()
        critic_loss = lap_huber(td_loss, self.args.min_priority)
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.lap_alpha).detach()

        return critic_loss, priority, Q.mean().detach(), qt_max, qt_min, ctx_ent

    def _actor_core(self, state, positions, nw):
        """Deterministic-policy-gradient actor loss. fixed_zs is recomputed here (identical value to the
        critic core's) rather than threaded across cores, to keep this a self-contained compiled graph."""
        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(state)
        actor_action = self.actor(state, fixed_zs, self.body, self.K_frozen, self.V_frozen, positions, nw)
        actor_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_action)
        actor_Q, _ = self.critic(
            state, actor_action, actor_fixed_zsa, fixed_zs, self.body, self.K_frozen, self.V_frozen, positions, nw
        )
        return -actor_Q.mean()

    def _aux_core(self, blocks_t, frac_t, targets_t):
        """Patch-encoder aux loss: predict z-scored per-patch reward-sums."""
        tok = self.tok_encoder(blocks_t, frac_t)
        return F.mse_loss(self.tok_encoder.predict_return(tok), targets_t)

    def aux_encoder_update(self):
        """Sample resident patches, predict their z-scored reward-sums, and refresh their live tokens."""
        n_resident = min(self.store.n_patch, self.store.raw_patches)
        if n_resident < 1:
            return
        b = min(self.args.aux_batch, n_resident)
        sel = np.random.choice(n_resident, size=b, replace=False)

        # patches are tiny (<=50 steps): encode the full block, no subsampling
        blocks = self.store.raw_blocks[sel]  # (b, patch_len, step_dim)
        lens = self.store.raw_lens[sel]
        frac = (lens.astype(np.float32) / self.patch_len)
        targets = np.asarray([self.patch_norm.normalize(r) for r in self.store.raw_rsum[sel]], dtype=np.float32)

        blocks_t = torch.as_tensor(blocks, device=self.device)
        frac_t = torch.as_tensor(frac, device=self.device)
        targets_t = torch.as_tensor(targets, device=self.device).reshape(-1, 1)

        aux_loss = self.aux_core(blocks_t, frac_t, targets_t)  # compiled loss compute

        self.tok_encoder_optimizer.zero_grad()
        aux_loss.backward()
        self.tok_encoder_optimizer.step()
        # keep the loss on-device; the host sync is deferred to the 500-step telemetry guard
        self.last_aux_loss = aux_loss.detach()

        # refresh the LIVE token cache with the freshly (post-step) encoded tokens.
        # single vectorized scatter (no Python loop, no per-element host round-trip). resident
        # global indices are always in range; the mask branch only guards the degenerate overflow case.
        with torch.no_grad():
            fresh = self.tok_encoder(blocks_t, frac_t)
            gidx = self.store.raw_global_idx[sel]
            valid = (gidx >= 0) & (gidx < self.store.max_patches)
            if valid.all():
                gidx_t = torch.as_tensor(gidx, device=self.device, dtype=torch.long)
                self.store.tokens[gidx_t] = fresh
            else:
                gidx_t = torch.as_tensor(gidx[valid], device=self.device, dtype=torch.long)
                self.store.tokens[gidx_t] = fresh[torch.as_tensor(valid, device=self.device)]

    def train(self):
        self.training_steps += 1

        state, action, next_state, reward, not_done, patch_idx = self.replay_buffer.sample()
        # s and s' share the collection-time patch context (they differ by at most one mid-episode patch
        # boundary; ignored). All replay forwards decode against the FROZEN KV cache at query pos patch_idx,
        # clamped to frozen_n_written (the last snapshot) so no never-written frozen position is attended.
        nw = self.frozen_n_written

        # SALE encoder update (kept eager: small, and coupled to fixed-encoder snapshot bookkeeping)
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # critic update: compiled loss/target/priority compute, eager backward + step + in-place LAP update.
        # noise is drawn eagerly and passed in (keeps RNG out of the graph for cudagraph friendliness);
        # min/max target clip bounds are passed as 0-d tensors so their 250-step changes don't recompile.
        noise = (torch.randn_like(action) * self.args.target_policy_noise).clamp(
            -self.args.noise_clip, self.args.noise_clip
        )
        min_t = torch.tensor(self.min_target, dtype=torch.float32, device=self.device)
        max_t = torch.tensor(self.max_target, dtype=torch.float32, device=self.device)
        critic_loss, priority, q_mean, qt_max, qt_min, ctx_ent = self.critic_core(
            state, action, next_state, reward, not_done, noise, patch_idx, nw, min_t, max_t
        )
        # critic_optimizer.zero_grad FIRST wipes any stray body grad from the previous actor step (grad
        # hygiene): the body is moved ONLY by this critic TD gradient. v3_3-SPECIFIC ADDITION: clip the
        # BODY grad-norm to 0.5 for truncated-BPTT stability. Critic params are NOT clipped (TD7/v3_1
        # dynamics + floor preserved). debug/body_grad_norm samples the POST-clip body grad norm (500-cadence).
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.clip_params, 0.5)
        if self.training_steps % 500 == 0:
            with torch.no_grad():
                bgn = torch.stack([p.grad.detach().norm() for p in self.body.parameters() if p.grad is not None])
                self.last_body_grad_norm = float(bgn.norm()) if bgn.numel() else 0.0
        self.critic_optimizer.step()
        self.replay_buffer.update_priority(priority)
        self.max = max(self.max, float(qt_max))
        self.min = min(self.min, float(qt_min))

        # update actor (delayed) + auxiliary token-encoder update on the same cadence. The actor uses the
        # CHEAP frozen decode() (body NOT moved by the actor): its backward deposits body grads (query path
        # only), but actor_optimizer excludes the body so they never move it, and the next critic
        # zero_grad wipes them. aux_encoder_update trains only the PatchEncoder (+ refreshes store.tokens).
        if self.training_steps % self.args.policy_freq == 0:
            actor_loss = self.actor_core(state, patch_idx, nw)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.aux_encoder_update()

            if self.training_steps % 500 == 0:
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.training_steps)

        # hard target/fixed-encoder updates + target clip range snapshot + frozen token & KV snapshot
        if self.training_steps % self.args.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.body_target.load_state_dict(self.body.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

            # EXACTLY rebuild both KV caches from the live token cache so body_target + frozen cache are a
            # consistent pair for the next 250 steps. Time the rebuild for telemetry.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            self.body.full_reprefill(
                self.store.tokens, self.store.n_patch, self.K_live, self.V_live, self.K_frozen, self.V_frozen,
                self.store.raw_patches,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.last_reprefill_ms = (time.time() - t0) * 1000.0
            # replay decodes may now read the freshly-snapshotted frozen positions [0, frozen_n_written)
            self.frozen_n_written = min(self.store.n_patch, MAX_PATCHES)

        # losses are logged against training_steps: with checkpointing, training happens in
        # episode-boundary bursts, so global_step would clump thousands of points at one x value
        if self.training_steps % 500 == 0:
            self.writer.add_scalar("losses/encoder_loss", encoder_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/critic_loss", critic_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/q_values", q_mean.item(), self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target, self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target, self.training_steps)
            self.writer.add_scalar("debug/aux_loss", float(self.last_aux_loss), self.training_steps)
            self.writer.add_scalar("debug/body_grad_norm", self.last_body_grad_norm, self.training_steps)
            self.writer.add_scalar("debug/reprefill_ms", self.last_reprefill_ms, self.training_steps)
            self.writer.add_scalar("debug/ctx_gate", self.actor.out.weight.norm().item(), self.training_steps)
            self.writer.add_scalar("debug/ctx_attn_entropy", float(ctx_ent), self.training_steps)
            self.writer.add_scalar("debug/n_patch", self.store.n_patch, self.training_steps)

    # if using checkpoints: run when each episode terminates
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # end assessment of the current policy early: it already lost to the checkpoint
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset()

    # batch training at episode boundaries
    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.args.steps_before_checkpointing:
                self.best_min_return *= self.args.reset_weight
                self.max_eps_before_update = self.args.max_eps_when_checkpointing

            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8


def evaluate(agent: TD7Agent, eval_env, eval_eps, use_checkpoint, hist_k, action_dim, max_action):
    returns = np.zeros(eval_eps)
    for ep in range(eval_eps):
        state, _ = eval_env.reset()
        act_hist = np.zeros((hist_k, action_dim), dtype=np.float32)
        done = False
        while not done:
            state_aug = np.concatenate([np.array(state, dtype=np.float32), act_hist.flatten()])
            action = agent.select_action(state_aug, use_checkpoint=use_checkpoint, use_exploration=False)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            act_hist = np.concatenate([(action / max_action)[None], act_hist[:-1]], axis=0)
            returns[ep] += reward
            done = terminated or truncated
    return returns.mean()


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "TD7 requires num_envs=1 (1:1 train/env-step ratio and episodic checkpointing)"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.compile:
        torch.set_float32_matmul_precision("high")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    eval_env = gym.make(args.env_id)
    eval_env.action_space.seed(args.seed + 100)

    state_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = int(np.prod(envs.single_action_space.shape))
    max_action = float(envs.single_action_space.high[0])

    # HIST: every network and the replay buffer see the augmented observation
    state_dim_aug = int(state_dim + args.hist_k * action_dim)
    # CTX: patch step records use RAW obs (not the hist-augmented obs), normalized action, reward
    step_dim = int(state_dim + action_dim + 1)
    agent = TD7Agent(state_dim_aug, action_dim, max_action, args, device, writer, step_dim)

    start_time = time.time()
    allow_train = False

    obs, _ = envs.reset(seed=args.seed)
    act_hist = np.zeros((args.hist_k, action_dim), dtype=np.float32)  # most-recent first
    cur_patch_steps = []  # raw (obs, normalized action, reward) rows for the in-progress patch
    eval_seeded = False
    # total_timesteps + 1 so the final evaluation at exactly total_timesteps fires (as in the original)
    for global_step in range(args.total_timesteps + 1):
        if global_step % args.eval_freq == 0:
            if not eval_seeded:
                eval_env.reset(seed=args.seed + 100)
                eval_seeded = True
            eval_return = evaluate(agent, eval_env, args.eval_eps, args.use_checkpoints,
                                   args.hist_k, action_dim, max_action)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(f"global_step={global_step}, eval_return={eval_return:.3f}")

        # ALGO LOGIC: put action logic here
        obs_aug = np.concatenate([np.array(obs[0], dtype=np.float32), act_hist.flatten()])
        if allow_train:
            actions = agent.select_action(obs_aug)[None]
        else:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # CTX: record the raw step (raw obs BEFORE hist augmentation, normalized action, reward)
        cur_patch_steps.append(
            np.concatenate(
                [
                    np.array(obs[0], dtype=np.float32),
                    (actions[0] / max_action).astype(np.float32),
                    np.array([rewards[0]], dtype=np.float32),
                ]
            )
        )

        # save data to replay buffer; handle `final_observation`. The original bootstraps at the
        # timeout step even when the env truly terminated there (`done = ep_finished if
        # ep_timesteps < max_episode_steps else 0`), hence `and not truncations[0]`
        real_next_obs = next_obs[0]
        if truncations[0] or terminations[0]:
            real_next_obs = infos["final_observation"][0]
        done = float(terminations[0] and not truncations[0])
        # HIST: the successor observation includes the just-executed action at the head
        next_hist = np.concatenate([(actions[0] / max_action)[None], act_hist[:-1]], axis=0)
        real_next_obs_aug = np.concatenate([np.array(real_next_obs, dtype=np.float32), next_hist.flatten()])
        # CTX: patch_idx is the count of COMPLETED patches when this transition was collected (BEFORE
        # this step possibly completes a patch below)
        agent.replay_buffer.add(obs_aug, actions[0], real_next_obs_aug, rewards[0], done, agent.store.n_patch)
        act_hist = (np.zeros((args.hist_k, action_dim), dtype=np.float32)
                    if (terminations[0] or truncations[0]) else next_hist)

        # CTX: emit a full patch mid-episode as soon as patch_len steps accumulate — the completed
        # patch enters the LIVE cache immediately, keeping the acting context causal and fresh
        if len(cur_patch_steps) == args.patch_len:
            agent.add_patch(cur_patch_steps)
            cur_patch_steps = []

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training (per-step when not checkpointing)
        if allow_train and not args.use_checkpoints:
            agent.train()

        # episode boundary: log return, flush the partial patch, run burst training, enable training
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_return = float(info["episode"]["r"])
                    ep_length = int(info["episode"]["l"])
                    print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # CTX: flush the episode's partial remainder as a patch (patches never cross episode
                    # boundaries) BEFORE the burst so it trains with the fresh token; reset the accumulator
                    if len(cur_patch_steps) > 0:
                        agent.add_patch(cur_patch_steps)
                    cur_patch_steps = []

                    if allow_train and args.use_checkpoints:
                        agent.maybe_train_and_checkpoint(ep_length, ep_return)

                    if global_step >= args.learning_starts:
                        allow_train = True
                    break

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "encoder": agent.encoder.state_dict(),
                "critic": agent.critic.state_dict(),
                "tok_encoder": agent.tok_encoder.state_dict(),
                "checkpoint_actor": agent.checkpoint_actor.state_dict(),
                "checkpoint_encoder": agent.checkpoint_encoder.state_dict(),
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    envs.close()
    eval_env.close()
    writer.close()

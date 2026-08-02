# TD7-CTX v2 — TD7-HIST + ViT-style PATCH tokens + dense identity-K/V attention over a 2560-token window.
# --------------------------------------------------------------------------------------------------
# IDEA: this is td7_hist_v1 (obs augmented with the last hist_k executed actions) EXACTLY, plus a
# read-only "context" pathway. There are NO episode tokens. Instead every patch_len=50 CONTIGUOUS
# steps within an episode is compressed into one 64-d PATCH token (a ViT-style flatten+MLP over the
# 50x(state,action,reward) block). The actor and critic densely cross-attend over the last
# ctx_window=2560 patch tokens = 128 full HalfCheetah episodes at 50-step granularity — a fine-grained,
# fixed-length in-context view of the recent trajectory distribution — and fuse the result into trunks.
#
# WHY THE td7_hist FLOOR IS GUARANTEED: the attention output projection is ZERO-INITIALIZED (weight
# AND bias). At init the context contribution `ctx` is identically 0, so the wider first hidden layer
# sees a zero block and the network is functionally td7_hist_v1 with an unused parameter slab. The
# context pathway therefore *earns* its influence via gradient descent; it cannot regress the base.
#
# IDENTITY K/V (the load-bearing design choice): learned key/value projections over a 2560-token
# window cannot be trained cheaply. Recomputing them per replay sample would be ~ctx_window x the
# attention cost; serving them from the immutable frozen token cache would mean they NEVER receive a
# gradient (dead weights). So there are NO K/V projections: the keys and values ARE the cached tokens,
# sliced into n_heads=4 x 16-d heads. Only the query projection, a per-head ALiBi-style learned age
# bias, and the zero-init output projection carry RL gradients; the token content is shaped entirely
# off-path by the aux encoder. Recency is encoded by the age bias (log2 age buckets), not a token add.
#
# WHAT TOKENS CAN CARRY IN A FIXED MDP: the dynamics are Markov in the raw obs, so context is not
# needed for optimality. What it *can* provide is a fine-grained view of the recent behavior policy
# and local outcomes (per-patch reward), enabling in-context self-correction faster than slow weight
# updates. KNOWN RISK: in a stationary MDP the optimal use of context can collapse to a constant; the
# zero-init floor makes that harmless and `debug/ctx_gate` tracks whether the pathway moved at all.
#
# ENCODER (no pooling): PatchEncoder flattens the zero-padded 50x step block, appends frac_valid
# (len/50), and maps it Linear->ELU->Linear->AvgL1Norm. Trained OFF the RL gradient path by an aux
# head predicting the z-scored per-patch reward-sum (20x more labels than an episode-return signal).
# All replay-path forwards read a FROZEN token snapshot (refreshed with the target nets every
# target_update_rate steps); only select_action/eval read the live cache. COST: tokens are encoded
# once per patch and cached immutably; the read-side attention is q.token dot-products over <=2560
# cached vectors (no K/V projections) — roughly 1.2-1.5x td7_hist per step. Floor = td7_hist via zero-init.
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
from torch.utils.tensorboard import SummaryWriter

# Patch-token cache capacity. HalfCheetah-v4 at 8M steps produces ~160000 full patches (8M/50);
# 220000 leaves headroom for early-terminating envs whose partial patches inflate the count. Writes
# and reads beyond it are clamped (with a one-time warning) and masked (see PatchStore / build_window).
MAX_PATCHES = 220000


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

    # --- CTX (ViT-style patch tokens + dense identity-K/V attention) ---
    ctx_window: int = 2560
    """number of most-recent patch tokens the actor/critic densely cross-attend over (=128 HC episodes)"""
    patch_len: int = 50
    """steps per patch token (patches never cross episode boundaries)"""
    tok_dim: int = 64
    """dimensionality of a patch token (= n_heads * head_dim)"""
    n_heads: int = 4
    """context-attention heads (identity keys/values sliced from the token)"""
    raw_patches: int = 20480
    """ring capacity of raw patch step blocks kept for the aux encoder loss"""
    aux_batch: int = 32
    """patches sampled per auxiliary encoder update"""

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


class ContextAttention(nn.Module):
    """Multi-head cross-attention with IDENTITY keys/values: keys = values = the token itself, sliced
    into n_heads x head_dim. No K/V projections exist (they could not be trained cheaply over a frozen
    2560-token cache). Only the query projection, a per-head ALiBi-style learned age bias, and the
    ZERO-INITIALIZED output projection carry gradients -> `ctx == 0` at init (td7_hist floor)."""

    def __init__(self, hdim, tok_dim=64, n_heads=4, ctx_out=64, n_age=16):
        super().__init__()
        assert tok_dim % n_heads == 0 and ctx_out == tok_dim
        self.n_heads = n_heads
        self.head_dim = tok_dim // n_heads
        self.q = nn.Linear(hdim, tok_dim)  # -> (B, n_heads, head_dim)
        self.age_bias = nn.Embedding(n_age, n_heads)  # ALiBi-style per-head logit bias by age bucket
        nn.init.zeros_(self.age_bias.weight)  # start recency-neutral
        self.out = nn.Linear(ctx_out, ctx_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q_in, tok_win, tok_age, tok_mask):
        # q_in: (B, hdim); tok_win: (B, W, tok_dim); tok_age: (B, W) long>=0; tok_mask: (B, W) bool
        B, W, _ = tok_win.shape
        H, D = self.n_heads, self.head_dim
        kv = tok_win.reshape(B, W, H, D)  # identity keys AND values, sliced per head
        q = self.q(q_in).reshape(B, H, D)
        logits = torch.einsum("bhd,bwhd->bhw", q, kv) / (D ** 0.5)  # (B, H, W)

        bucket = torch.log2(tok_age.float() + 1.0).floor().long().clamp(0, 15)  # (B, W)
        logits = logits + self.age_bias(bucket).permute(0, 2, 1)  # add (B, H, W)

        mask = tok_mask.bool()  # (B, W)
        no_valid = ~mask.any(dim=1)  # (B,) rows with no valid token
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask.unsqueeze(1), neg)  # broadcast over heads
        # guard the all-masked softmax (NaN) by feeding zero logits for empty rows, then zero the output
        logits = torch.where(no_valid.view(B, 1, 1), torch.zeros_like(logits), logits)
        attn = torch.softmax(logits, dim=-1)  # (B, H, W)
        out = torch.einsum("bhw,bwhd->bhd", attn, kv).reshape(B, H * D)  # (B, tok_dim)
        out = torch.where(no_valid.view(B, 1), torch.zeros_like(out), out)
        ctx = self.out(out)  # (B, ctx_out); == 0 at init
        return ctx, attn, mask


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, tok_dim=64, n_heads=4, zs_dim=256, hdim=256, ctx_out=64):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.attn = ContextAttention(hdim, tok_dim, n_heads=n_heads, ctx_out=ctx_out)
        self.l1 = nn.Linear(zs_dim + hdim + ctx_out, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs, tok_win, tok_age, tok_mask):
        a0 = avg_l1_norm(self.l0(state))
        ctx, _, _ = self.attn(a0, tok_win, tok_age, tok_mask)
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
    def __init__(self, state_dim, action_dim, tok_dim=64, n_heads=4, zs_dim=256, hdim=256, ctx_out=64):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.attn = ContextAttention(hdim, tok_dim, n_heads=n_heads, ctx_out=ctx_out)
        self.q1 = nn.Linear(2 * zs_dim + hdim + ctx_out, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim + ctx_out, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs, tok_win, tok_age, tok_mask):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1f = avg_l1_norm(self.q01(sa))
        # single shared context, queried by the q1-branch normalized feature, appended to both twins.
        # attn/mask are RETURNED (not stashed on self) so the compiled cores stay side-effect-free.
        ctx, attn, mask = self.attn(q1f, tok_win, tok_age, tok_mask)

        q1 = torch.cat([q1f, embeddings, ctx], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        q2f = avg_l1_norm(self.q02(sa))
        q2 = torch.cat([q2f, embeddings, ctx], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1), attn, mask


class PatchStore:
    """Ring of raw patch blocks (for the aux loss) + a GPU token cache indexed by global patch id.

    tokens: LIVE cache, written at add_patch and refreshed by the aux encoder step.
    tokens_frozen: snapshot copied from tokens at each hard target update; used by ALL replay forwards.
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
        self.tokens_frozen = torch.zeros(max_patches, tok_dim, device=device)
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
        elif not self._warned:
            print(f"[PatchStore] n_patch exceeded MAX_PATCHES={self.max_patches}; clamping token writes/reads")
            self._warned = True
            self.tokens[self.max_patches - 1] = tok
        else:
            self.tokens[self.max_patches - 1] = tok
        self.n_patch += 1


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
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

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

    def build_window(self, patch_idx, tokens):
        """Given patch_idx (B,) and a token tensor, return (tok_win (B,W,tok_dim), age (B,W), mask (B,W)).

        Token index for window position p in [0,W) is patch_idx-W+p; valid where that index >= 0. The
        most recent token (p=W-1) has age 0; age = W-1-p (independent of patch_idx). Invalid slots are
        clamped for the gather and zeroed / masked out.
        """
        W = self.ctx_window
        p = torch.arange(W, device=self.device)  # (W,)
        idx = patch_idx.unsqueeze(1) - W + p.unsqueeze(0)  # (B, W)
        # valid where the token index is in-range; the upper guard matters only past MAX_PATCHES
        # (short-episode envs) — overflow rows then mask out and degrade to the zero-context floor
        mask = (idx >= 0) & (idx < tokens.shape[0])
        idx_clamped = idx.clamp(min=0, max=tokens.shape[0] - 1)
        tok_win = tokens[idx_clamped]  # (B, W, tok_dim)
        tok_win = tok_win * mask.unsqueeze(-1).float()  # zero invalid slots
        age = (W - 1 - p).unsqueeze(0).expand(patch_idx.shape[0], W).long()
        return tok_win, age, mask

    def add_patch(self, step_list):
        # step_list: list of raw step vecs (len 1..patch_len). Zero-pad to patch_len, compute reward-sum.
        n = len(step_list)
        a = np.asarray(step_list, dtype=np.float32)
        block = np.zeros((self.patch_len, self.store.step_dim), dtype=np.float32)
        block[:n] = a
        rsum = float(a[:, -1].sum())  # reward is the last column of a step record
        self.patch_norm.update(rsum)
        self.store.add_patch(block, n, rsum)

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
            # LIVE tokens, current completed-patch count
            patch_idx = torch.tensor([self.store.n_patch], dtype=torch.long, device=self.device)
            tok_win, tok_age, tok_mask = self.build_window(patch_idx, self.store.tokens)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs, tok_win, tok_age, tok_mask)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs, tok_win, tok_age, tok_mask)

            if use_exploration:
                action = action + torch.randn_like(action) * self.args.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    # ---- hot compute cores (pure tensor in / tensor out; optionally torch.compile'd, see __init__) ----
    def _critic_core(self, state, action, next_state, reward, not_done, noise,
                     tok_win, tok_age, tok_mask, min_t, max_t):
        """TD target + critic loss + LAP priority + telemetry scalars. No in-place cache/opt mutation.
        min_t/max_t are 0-d tensors (not python floats) so the target clip does NOT trigger recompiles."""
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)
            next_action = (self.actor_target(next_state, fixed_target_zs, tok_win, tok_age, tok_mask) + noise).clamp(
                -1, 1
            )
            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)
            Qt, _, _ = self.critic_target(
                next_state, next_action, fixed_target_zsa, fixed_target_zs, tok_win, tok_age, tok_mask
            )
            Q_target = Qt.min(1, keepdim=True)[0]
            Q_target = reward + not_done * self.args.gamma * Q_target.clamp(min_t, max_t)
            qt_max = Q_target.max()
            qt_min = Q_target.min()
            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q, attn, mask = self.critic(state, action, fixed_zsa, fixed_zs, tok_win, tok_age, tok_mask)
        td_loss = (Q - Q_target).abs()
        critic_loss = lap_huber(td_loss, self.args.min_priority)
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.lap_alpha).detach()

        m = mask.float().unsqueeze(1)  # (B, 1, W)
        ctx_ent = -(attn.clamp(min=1e-12).log() * attn * m).sum(-1).mean()  # scalar entropy over valid slots
        return critic_loss, priority, Q.mean().detach(), qt_max, qt_min, ctx_ent.detach()

    def _actor_core(self, state, tok_win, tok_age, tok_mask):
        """Deterministic-policy-gradient actor loss. fixed_zs is recomputed here (identical value to the
        critic core's) rather than threaded across cores, to keep this a self-contained compiled graph."""
        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(state)
        actor_action = self.actor(state, fixed_zs, tok_win, tok_age, tok_mask)
        actor_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_action)
        actor_Q, _, _ = self.critic(state, actor_action, actor_fixed_zsa, fixed_zs, tok_win, tok_age, tok_mask)
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
        # boundary; ignored). All replay forwards read the frozen token snapshot.
        tok_win, tok_age, tok_mask = self.build_window(patch_idx, self.store.tokens_frozen)

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
            state, action, next_state, reward, not_done, noise, tok_win, tok_age, tok_mask, min_t, max_t
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.replay_buffer.update_priority(priority)
        self.max = max(self.max, float(qt_max))
        self.min = min(self.min, float(qt_min))

        # update actor (delayed) + auxiliary token-encoder update on the same cadence
        if self.training_steps % self.args.policy_freq == 0:
            actor_loss = self.actor_core(state, tok_win, tok_age, tok_mask)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.aux_encoder_update()

            if self.training_steps % 500 == 0:
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.training_steps)

        # hard target/fixed-encoder updates + target clip range snapshot + frozen token snapshot
        if self.training_steps % self.args.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

            self.store.tokens_frozen.copy_(self.store.tokens)

        # losses are logged against training_steps: with checkpointing, training happens in
        # episode-boundary bursts, so global_step would clump thousands of points at one x value
        if self.training_steps % 500 == 0:
            self.writer.add_scalar("losses/encoder_loss", encoder_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/critic_loss", critic_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/q_values", q_mean.item(), self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target, self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target, self.training_steps)
            self.writer.add_scalar("debug/aux_loss", float(self.last_aux_loss), self.training_steps)
            self.writer.add_scalar("debug/ctx_gate", self.actor.attn.out.weight.norm().item(), self.training_steps)
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

# TD7-STEPMEM v1 — td7_hist_v1 + per-network PRIVATE cross-attention over the last 64 raw steps.
# ------------------------------------------------------------------------------------------------
# WHY (post-mortem of the patch-token lineage): td7_ctx_v3_x compressed history into 50-step PATCH
# tokens fed through a shared, KV-cached transformer body. A design audit traced its step-matched
# underperformance vs plain TD7 to four coupled defects: (a) tokens were supervised only by a rank-1
# reward-sum aux (a value bottleneck), (b) >=50-step structural staleness (a patch hides intra-patch
# dynamics and the KV cache is rebuilt only every 250 steps), (c) the shared body's gradient explained
# RETURN VARIANCE for the critic rather than improving actions, and (d) a fast-churning shared body
# violates SALE's input-stability principle (actor/critic condition on a representation that moves under
# them every step). td7_stepmem_v1 dissolves all four by construction and scales the ONE history
# mechanism that already works in this repo — td7_hist's last-k executed actions — from 4 scalars to 64
# FULL transitions at STEP resolution.
#
# WHAT IT IS (everything below the memory pathway is byte-identical TD7/hist; hist_k=4 aug kept exactly):
#   * Memory content: the last mem_len=64 raw steps (raw_obs, normalized_action, reward) of the CURRENT
#     episode, one token per step. Token = Linear(step_dim -> d_mem=64), step_dim = raw_obs + act + 1.
#   * Readout: a SINGLE cross-attention layer per consumer (4 heads x 16), pre-LN on the token
#     embeddings, RoPE on the token's AGE (1=previous step .. mem_len), query at age 0. Validity mask =
#     min(mem_len, steps-so-far-in-episode); zero valid tokens (episode start) -> memory output := 0.
#     The out-projection is ZERO-INIT (weight+bias), so at init the memory adds exactly 0 and the network
#     is bitwise td7_hist_v1 (exact floor). Output (d_mem->hdim) is ADDED to the consumer's post-l0 trunk
#     feature (additive residual, NOT a concat) so no base Linear changes width -> base init RNG is
#     preserved bit-for-bit vs hist (memory params are initialised off a forked RNG stream; see _RNGFork).
#   * PRIVATE, no sharing: the actor owns its token-proj+attention (in the actor optimizer, moved by the
#     actor loss only); the critic owns its own (in the critic optimizer, moved by the critic loss only;
#     both Q-twins share the critic's single memory output). Actor query = its post-l0 state feature;
#     critic query = its (s,a) feature q1f, so grad(memory)/grad(action) exists. target/checkpoint copies
#     are plain deepcopies following the SAME hard-update/checkpoint flow as the rest of the network — no
#     special casing, no caches.
#   * Replay-time reconstruction (EXACT BPTT, no cache, zero staleness): the LAP ring stores a per-
#     transition int32 t_in_ep and an int64 global write index (gidx). For a sampled transition, the
#     s-memory is the mem_len transitions strictly before it and the s'-memory is the window ending at the
#     transition itself; both are masked by (age <= steps-in-episode) AND (predecessor still resident in
#     the ring) so episode boundaries and ring wraparound never leak. Windows are gathered (B x mem_len x
#     step_dim) and attention is recomputed in-graph every update.
#   * Why the four defects are gone: STEP resolution (no patch), zero staleness (recomputed in-graph from
#     the raw ring every step), RL-gradients-ONLY (no aux, no self-prediction — pure actor/critic loss),
#     and PRIVATE per-network modules (nothing shared churns under SALE's frozen embeddings).
# HYPOTHESIS: td7_hist proved fine-grained recent memory helps on these near-Markov tasks; giving each
# network its own attention over 64 full transitions (vs 4 bare actions) should extend that signal
# without importing any of the patch lineage's failure modes.
# ------------------------------------------------------------------------------------------------
# TD7 = TD3 + SALE + LAP + policy checkpointing + target value clipping (Fujimoto et al. 2023,
# https://arxiv.org/abs/2306.02451). See td7_hist_v1.py for the full base description. Requires num_envs=1.
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

    # --- HIST ---
    hist_k: int = 4
    """number of most-recent executed actions appended to the observation (unchanged from td7_hist_v1)"""

    # --- STEPMEM (per-network private cross-attention over the last mem_len raw steps) ---
    mem_len: int = 64
    """number of most-recent raw steps of the current episode kept as memory tokens"""
    d_mem: int = 64
    """memory token / attention model dimension (= mem_heads * head_dim)"""
    mem_heads: int = 4
    """memory cross-attention heads (head_dim = d_mem // mem_heads = 16)"""
    mem_enabled: bool = True
    """if False, the memory pathway is inert (exact td7_hist_v1) — used for the transparency floor test"""

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


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


class _RNGFork:
    """Context manager: run the enclosed init off a FORKED (deterministic) torch RNG stream, restoring the
    main CPU+CUDA RNG state on exit. Used to initialise the memory modules WITHOUT advancing the main
    stream, so the base network (actor/critic/encoder) draws exactly td7_hist_v1's init RNG -> exact floor.
    """

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self._cpu = torch.get_rng_state()
        self._cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self._cpu)
        if self._cuda is not None:
            torch.cuda.set_rng_state_all(self._cuda)
        return False


class LAPBuffer:
    """LAP replay buffer: priorities live on-device, sampling by inverse-CDF over the priority cumsum.

    STEPMEM addition: per transition we store an int32 t_in_ep (episode-step index, 0 at episode start)
    and an int64 gidx (monotonic global write index). At sample time we reconstruct, for each sampled
    transition, the memory windows for BOTH its state s and its successor s' as (B, mem_len, step_dim)
    token blocks + validity masks, gathered from the ring. A memory token at age a (1 = most-recent) is
    valid iff (a <= steps-in-episode-at-that-state) AND (the predecessor transition is still resident in
    the ring, checked via gidx vs the oldest resident global index). This makes episode boundaries and
    ring wraparound exact — no cross-episode or overwritten-slot leakage.
    """

    def __init__(self, state_dim, action_dim, device, max_size, batch_size, max_action,
                 raw_state_dim, mem_len):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.total_added = 0  # monotonic count of transitions ever added (global write index source)

        self.device = device
        self.batch_size = batch_size
        self.raw_state_dim = int(raw_state_dim)
        self.action_dim = int(action_dim)
        self.mem_len = int(mem_len)
        self.mem_step_dim = self.raw_state_dim + self.action_dim + 1

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.t_in_ep = np.zeros(self.max_size, dtype=np.int64)  # episode-step index of `state`
        self.gidx = np.zeros(self.max_size, dtype=np.int64)     # global write index of this slot's datum

        self.priority = torch.zeros(self.max_size, device=device)
        self.max_priority = 1.0

        self.normalize_actions = float(max_action)

    def add(self, state, action, next_state, reward, done, t_in_ep):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.t_in_ep[self.ptr] = t_in_ep
        self.gidx[self.ptr] = self.total_added

        self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.total_added += 1

    def _gather_windows(self):
        """Build the s- and s'- memory token blocks + masks for the current sampled indices self.ind.

        For a sampled slot i (global gidx g_i, episode-step t): the s-window token at age a (1..mem_len)
        is transition i-a (global g_i-a); the s'-window token at age a is transition i-(a-1) (global
        g_i-(a-1), i.e. age 1 = transition i itself). A token is valid iff age <= in-episode step count
        (t for s, t+1 for s') AND its global index is still resident (>= oldest resident global). Slots are
        computed mod max_size; invalid tokens are zeroed (and additionally masked in the attention)."""
        i = self.ind.astype(np.int64)                                   # (B,) sampled slots
        B = i.shape[0]
        M = self.mem_len
        g_i = self.gidx[i]                                              # (B,) global write index
        t = self.t_in_ep[i]                                            # (B,) episode-step of state s
        oldest_g = self.total_added - self.size                        # smallest resident global index
        ages = np.arange(1, M + 1, dtype=np.int64)                     # (M,) age 1..mem_len

        # s-window: age a -> transition i-a
        s_slots = (i[:, None] - ages[None, :]) % self.max_size          # (B, M)
        s_gpred = g_i[:, None] - ages[None, :]                          # (B, M)
        s_valid = (ages[None, :] <= t[:, None]) & (s_gpred >= oldest_g) & (s_gpred >= 0)
        # s'-window: age a -> transition i-(a-1)  (age 1 == transition i)
        sp_slots = (i[:, None] - (ages[None, :] - 1)) % self.max_size   # (B, M)
        sp_gpred = g_i[:, None] - (ages[None, :] - 1)                   # (B, M)
        sp_valid = (ages[None, :] <= (t[:, None] + 1)) & (sp_gpred >= oldest_g) & (sp_gpred >= 0)

        def build(slots, valid):
            raw_obs = self.state[slots][..., : self.raw_state_dim]      # (B, M, raw_state_dim)
            act = self.action[slots]                                    # (B, M, action_dim)  (normalized)
            rew = self.reward[slots]                                    # (B, M, 1)
            tok = np.concatenate([raw_obs, act, rew], axis=-1)          # (B, M, step_dim)
            tok = tok * valid[..., None]                               # zero invalid tokens (belt+braces)
            return (
                torch.as_tensor(tok, dtype=torch.float, device=self.device),
                torch.as_tensor(valid, device=self.device),
            )

        s_tok, s_mask = build(s_slots, s_valid)
        sp_tok, sp_mask = build(sp_slots, sp_valid)
        return s_tok, s_mask, sp_tok, sp_mask

    def sample(self):
        csum = torch.cumsum(self.priority[: self.size], 0)
        val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
        self.ind = torch.searchsorted(csum, val).cpu().data.numpy()

        s_tok, s_mask, sp_tok, sp_mask = self._gather_windows()
        return (
            torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device),
            s_tok, s_mask, sp_tok, sp_mask,
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())


class StepMemory(nn.Module):
    """Private single-layer cross-attention over the last mem_len raw-step tokens (see header).

    query_feat: (B, query_dim) the consumer's own feature. tokens: (B, mem_len, step_dim) most-recent-
    first. mask: (B, mem_len) bool validity. Returns (B, out_dim) ADDED to the consumer's trunk feature;
    ZERO at init (out_proj zero-init) and ZERO for rows with no valid tokens (episode start)."""

    def __init__(self, step_dim, query_dim, out_dim, d_mem=64, n_heads=4, mem_len=64):
        super().__init__()
        assert d_mem % n_heads == 0
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.head_dim = d_mem // n_heads
        self.mem_len = mem_len
        self.scale = self.head_dim ** -0.5

        self.token_proj = nn.Linear(step_dim, d_mem)
        self.ln = nn.LayerNorm(d_mem)               # pre-LN on token embeddings
        self.q_proj = nn.Linear(query_dim, d_mem)
        self.k_proj = nn.Linear(d_mem, d_mem)
        self.v_proj = nn.Linear(d_mem, d_mem)
        self.out_proj = nn.Linear(d_mem, out_dim)   # ZERO-INIT -> exact td7_hist floor at init
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # RoPE inverse frequencies over head_dim (fp64 angles -> input dtype); applied to KEY age only.
        idx = torch.arange(self.head_dim // 2, dtype=torch.float64)
        inv_freq = 10000.0 ** (-(2.0 * idx / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x, ages):
        # x: (B, M, H, Dh); ages: (M,) integer positions -> rotate each key by its age.
        ang = ages.to(torch.float64)[None, :, None, None] * self.inv_freq  # (1, M, 1, Dh/2)
        cos = torch.cat([torch.cos(ang), torch.cos(ang)], dim=-1).to(x.dtype)
        sin = torch.cat([torch.sin(ang), torch.sin(ang)], dim=-1).to(x.dtype)
        return x * cos + rotate_half(x) * sin

    def forward(self, query_feat, tokens, mask):
        B, M, _ = tokens.shape
        H, Dh = self.n_heads, self.head_dim
        te = self.ln(self.token_proj(tokens))                          # (B, M, d_mem) pre-LN
        k = self._rope(self.k_proj(te).view(B, M, H, Dh),
                       torch.arange(1, M + 1, device=tokens.device))    # RoPE keys at age 1..M
        v = self.v_proj(te).view(B, M, H, Dh)
        q = self.q_proj(query_feat).view(B, 1, H, Dh)                  # query at age 0 (RoPE identity)

        logits = torch.einsum("bqhd,bkhd->bhqk", q, k) * self.scale     # (B, H, 1, M)
        neg = torch.finfo(logits.dtype).min
        no_valid = ~mask.any(dim=1)                                    # (B,) rows with zero valid tokens
        logits = logits.masked_fill(~mask[:, None, None, :], neg)
        logits = logits.masked_fill(no_valid[:, None, None, None], 0.0)  # NaN guard: all-(-inf) -> uniform
        attn = torch.softmax(logits, dim=-1)                           # (B, H, 1, M)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v).reshape(B, H * Dh)  # (B, d_mem)
        out = self.out_proj(out)                                       # (B, out_dim); 0 at init
        return out.masked_fill(no_valid[:, None], 0.0)                 # empty memory -> exactly 0


class Actor(nn.Module):
    """td7_hist Actor + a private StepMemory added (residually) to the post-l0 state feature. Query = that
    feature. At init the memory adds 0 (zero-init out_proj) -> bitwise td7_hist. mem_step_dim etc. thread
    in the memory geometry; base l0..l3 keep td7_hist's exact dims and (via _RNGFork) init RNG."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256,
                 mem_step_dim=0, d_mem=64, mem_heads=4, mem_len=64, mem_enabled=True, mem_seed=0):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        self.mem_enabled = mem_enabled
        if mem_enabled:
            with _RNGFork(mem_seed):  # init memory off a forked stream -> base l0..l3 RNG == td7_hist
                self.mem = StepMemory(mem_step_dim, hdim, hdim, d_mem, mem_heads, mem_len)

    def forward(self, state, zs, mem_tokens, mem_mask):
        a = avg_l1_norm(self.l0(state))
        if self.mem_enabled:
            a = a + self.mem(a, mem_tokens, mem_mask)  # 0 at init / empty memory
        a = torch.cat([a, zs], 1)
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
    """td7_hist Critic + a single private StepMemory shared by both Q-twins. Query = the q1 branch (s,a)
    feature q1f (so grad(memory)/grad(action) exists); the memory output is added (residually) to BOTH
    twins' post-q0 features. Zero-init out_proj -> bitwise td7_hist at init."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256,
                 mem_step_dim=0, d_mem=64, mem_heads=4, mem_len=64, mem_enabled=True, mem_seed=0):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)
        self.mem_enabled = mem_enabled
        if mem_enabled:
            with _RNGFork(mem_seed):  # init memory off a forked stream -> base q0..q6 RNG == td7_hist
                self.mem = StepMemory(mem_step_dim, hdim, hdim, d_mem, mem_heads, mem_len)

    def forward(self, state, action, zsa, zs, mem_tokens, mem_mask):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1f = avg_l1_norm(self.q01(sa))
        m = self.mem(q1f, mem_tokens, mem_mask) if self.mem_enabled else 0.0  # shared, 0 at init/empty

        q1 = torch.cat([q1f + m, embeddings], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        q2f = avg_l1_norm(self.q02(sa))
        q2 = torch.cat([q2f + m, embeddings], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_action, args: Args, device, writer: SummaryWriter,
                 raw_state_dim):
        self.args = args
        self.device = device
        self.writer = writer
        self.mem_len = args.mem_len
        self.mem_step_dim = raw_state_dim + action_dim + 1

        mem_kw = dict(mem_step_dim=self.mem_step_dim, d_mem=args.d_mem, mem_heads=args.mem_heads,
                      mem_len=args.mem_len, mem_enabled=args.mem_enabled)
        self.actor = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim,
                           mem_seed=args.seed + 9001, **mem_kw).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, args.zs_dim, args.hidden_dim,
                             mem_seed=args.seed + 9002, **mem_kw).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = LAPBuffer(state_dim, action_dim, device, args.buffer_size, args.batch_size,
                                       max_action, raw_state_dim, args.mem_len)

        self.max_action = max_action
        self.training_steps = 0
        self.last_mem_mag = 0.0  # telemetry: mean |memory output| magnitude (grows from 0)

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

    def select_action(self, state, mem_tokens, mem_mask, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
            mt = torch.as_tensor(mem_tokens[None], dtype=torch.float, device=self.device)
            mk = torch.as_tensor(mem_mask[None], device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs, mt, mk)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs, mt, mk)

            if use_exploration:
                action = action + torch.randn_like(action) * self.args.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def train(self):
        self.training_steps += 1

        state, action, next_state, reward, not_done, s_tok, s_mask, sp_tok, sp_mask = \
            self.replay_buffer.sample()

        # update encoder: predict the next state's embedding from (zs, action) — no memory pathway (SALE)
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # update critic (target path uses the s'-window memory; online path uses the s-window)
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.args.target_policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            next_action = (self.actor_target(next_state, fixed_target_zs, sp_tok, sp_mask) + noise).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            Q_target = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs,
                                          sp_tok, sp_mask).min(1, keepdim=True)[0]
            Q_target = reward + not_done * self.args.gamma * Q_target.clamp(self.min_target, self.max_target)

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs, s_tok, s_mask)
        td_loss = (Q - Q_target).abs()
        critic_loss = lap_huber(td_loss, self.args.min_priority)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update LAP priorities
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.lap_alpha).detach()
        self.replay_buffer.update_priority(priority)

        # update actor (delayed) — actor + its scoring critic both use the s-window memory
        if self.training_steps % self.args.policy_freq == 0:
            actor_action = self.actor(state, fixed_zs, s_tok, s_mask)
            actor_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_action)
            actor_Q = self.critic(state, actor_action, actor_fixed_zsa, fixed_zs, s_tok, s_mask)
            actor_loss = -actor_Q.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.training_steps % 500 == 0:
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.training_steps)

        # hard target/fixed-encoder updates + target clip range snapshot
        if self.training_steps % self.args.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        # losses are logged against training_steps: with checkpointing, training happens in
        # episode-boundary bursts, so global_step would clump thousands of points at one x value
        if self.training_steps % 500 == 0:
            if self.args.mem_enabled:
                with torch.no_grad():
                    self.last_mem_mag = float(self.critic.mem(
                        avg_l1_norm(self.critic.q01(torch.cat([state, action], 1))), s_tok, s_mask
                    ).abs().mean())
            self.writer.add_scalar("losses/encoder_loss", encoder_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/critic_loss", critic_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/q_values", Q.mean().item(), self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target, self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target, self.training_steps)
            self.writer.add_scalar("debug/mem_out_mag", self.last_mem_mag, self.training_steps)

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


class EpisodeMemory:
    """Rolling most-recent-first window of the current episode's last mem_len raw steps (act/eval time).
    Token = [raw_obs, normalized_action, reward]; matches the replay-time reconstruction exactly."""

    def __init__(self, mem_len, step_dim):
        self.mem_len = mem_len
        self.step_dim = step_dim
        self.reset()

    def reset(self):
        self.buf = np.zeros((self.mem_len, self.step_dim), dtype=np.float32)
        self.count = 0

    def tokens_and_mask(self):
        mask = np.arange(self.mem_len) < self.count  # (mem_len,) age a=idx+1 valid iff idx < count
        return self.buf, mask

    def push(self, raw_obs, norm_action, reward):
        rec = np.concatenate([np.asarray(raw_obs, dtype=np.float32),
                              np.asarray(norm_action, dtype=np.float32),
                              np.asarray([reward], dtype=np.float32)])
        self.buf = np.concatenate([rec[None], self.buf[:-1]], axis=0)  # most-recent-first
        self.count = min(self.count + 1, self.mem_len)


def evaluate(agent: TD7Agent, eval_env, eval_eps, use_checkpoint, hist_k, action_dim, max_action,
             mem_len, mem_step_dim, raw_state_dim):
    returns = np.zeros(eval_eps)
    mem = EpisodeMemory(mem_len, mem_step_dim)
    for ep in range(eval_eps):
        state, _ = eval_env.reset()
        act_hist = np.zeros((hist_k, action_dim), dtype=np.float32)
        mem.reset()
        done = False
        while not done:
            state_aug = np.concatenate([np.array(state, dtype=np.float32), act_hist.flatten()])
            mt, mk = mem.tokens_and_mask()
            action = agent.select_action(state_aug, mt, mk, use_checkpoint=use_checkpoint, use_exploration=False)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            # push the completed step (raw obs seen, normalized action, reward) into memory
            mem.push(np.array(state, dtype=np.float32)[:raw_state_dim], action / max_action, reward)
            act_hist = np.concatenate([(action / max_action)[None], act_hist[:-1]], axis=0)
            returns[ep] += reward
            state = next_state
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
    raw_state_dim = int(state_dim)
    state_dim_aug = int(state_dim + args.hist_k * action_dim)
    mem_step_dim = raw_state_dim + action_dim + 1
    agent = TD7Agent(state_dim_aug, action_dim, max_action, args, device, writer, raw_state_dim)

    start_time = time.time()
    allow_train = False

    obs, _ = envs.reset(seed=args.seed)
    act_hist = np.zeros((args.hist_k, action_dim), dtype=np.float32)  # most-recent first
    mem = EpisodeMemory(args.mem_len, mem_step_dim)  # STEPMEM: current-episode rolling step window
    ep_t = 0  # STEPMEM: episode-step index of the current transition (0 at episode start)
    eval_seeded = False
    # total_timesteps + 1 so the final evaluation at exactly total_timesteps fires (as in the original)
    for global_step in range(args.total_timesteps + 1):
        if global_step % args.eval_freq == 0:
            if not eval_seeded:
                eval_env.reset(seed=args.seed + 100)
                eval_seeded = True
            eval_return = evaluate(agent, eval_env, args.eval_eps, args.use_checkpoints,
                                   args.hist_k, action_dim, max_action, args.mem_len, mem_step_dim, raw_state_dim)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(f"global_step={global_step}, eval_return={eval_return:.3f}")

        # ALGO LOGIC: put action logic here
        obs_aug = np.concatenate([np.array(obs[0], dtype=np.float32), act_hist.flatten()])
        mt, mk = mem.tokens_and_mask()  # STEPMEM: memory as of the current obs (prior steps this episode)
        if allow_train:
            actions = agent.select_action(obs_aug, mt, mk)[None]
        else:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

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
        # STEPMEM: store this transition's episode-step index (ep_t) for replay-time window masking
        agent.replay_buffer.add(obs_aug, actions[0], real_next_obs_aug, rewards[0], done, ep_t)
        # STEPMEM: push the completed step into the current-episode memory (raw obs, norm action, reward)
        mem.push(np.array(obs[0], dtype=np.float32)[:raw_state_dim], actions[0] / max_action, rewards[0])
        if terminations[0] or truncations[0]:
            act_hist = np.zeros((args.hist_k, action_dim), dtype=np.float32)
            mem.reset()
            ep_t = 0
        else:
            act_hist = next_hist
            ep_t += 1

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training (per-step when not checkpointing)
        if allow_train and not args.use_checkpoints:
            agent.train()

        # episode boundary: log return, run burst training/checkpointing, enable training
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_return = float(info["episode"]["r"])
                    ep_length = int(info["episode"]["l"])
                    print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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
                "checkpoint_actor": agent.checkpoint_actor.state_dict(),
                "checkpoint_encoder": agent.checkpoint_encoder.state_dict(),
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    envs.close()
    eval_env.close()
    writer.close()

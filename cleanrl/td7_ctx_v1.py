# TD7-CTX v1 — TD7-HIST + cross-episode context tokens (in-context conditioning on prior rollouts).
# --------------------------------------------------------------------------------------------------
# IDEA: this is td7_hist_v1 (obs augmented with the last hist_k executed actions) EXACTLY, plus a
# read-only "context" pathway. Each completed episode is compressed ONCE into a small immutable
# token (a 64-d summary of its (state, action, reward) trace). The actor and critic additionally
# cross-attend over the K most recent episode tokens — an in-context view of what the recent policy
# was doing and how it turned out — and fuse the attention output into their trunks.
#
# WHY THE td7_hist FLOOR IS GUARANTEED: the attention output projection is ZERO-INITIALIZED (weight
# AND bias). At init the context contribution `ctx` is identically 0, so the wider first hidden layer
# sees a zero block and the network is functionally td7_hist_v1 with an unused parameter slab. The
# context pathway therefore *earns* its influence via gradient descent; it cannot regress the base.
#
# WHAT TOKENS CAN CARRY IN A FIXED MDP: the dynamics are Markov in the raw obs, so context is not
# needed for optimality. What it *can* provide is recent-policy identity + outcome (return, visited
# regions), enabling in-context self-correction: the critic/actor can condition on "the last K
# rollouts looked like X and scored Y" and shift behavior faster than slow weight updates alone.
# KNOWN RISK: in a stationary MDP the optimal use of context can collapse to a constant (ignore it),
# so the pathway may converge to ~0 influence — the zero-init floor makes that outcome harmless, and
# `debug/ctx_gate` tracks whether it moved at all.
#
# ENCODER (no pooling, per user requirement): each episode trace is compressed by a 2-layer pre-LN
# CAUSAL transformer (d_model=64, single head) over its ordered (state, action, reward) steps with
# learned positional embeddings; the token is the causal readout — the hidden state at the last valid
# step, i.e. attention over the whole ordered rollout, never a mean/max. HIERARCHY RATIONALE: feeding
# raw per-step tokens across many episodes into the actor/critic would be 100-1000x the train cost
# (K episodes x ~1000 steps of attention every gradient step); compressing each episode ONCE into a
# single cached token instead lets us keep ~128 full rollouts in context at ~1.2-1.5x.
# DESIGN CHOICES: tokens are immutable-by-default (episodes never change) and cached on GPU; the
# token ENCODER is trained OFF the RL gradient path by an auxiliary episode-return prediction loss
# (so context representation drift is decoupled from value learning, mirroring TD7's frozen-encoder
# philosophy). All replay-path forwards read a FROZEN token snapshot (refreshed with the target nets
# every target_update_rate steps); only select_action/eval read the live cache. COST: tokens are
# cached and reused (encoded once per episode on the full trace, cheaply refreshed in the aux step on
# a <=512-step contiguous slice), the read-side attention is over K<=64 cached vectors with a single
# head — roughly 1.2-1.5x td7_hist per step, dominated by the aux transformer forwards.
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

# Token cache capacity. HalfCheetah-v4 at 8M steps produces ~8000 episodes (fixed 1000-step episodes),
# comfortably below this; writes/reads beyond it are clamped and masked (see EpisodeStore / build_window).
MAX_EPISODES = 10000


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
    """number of most-recent executed actions appended to the observation"""

    # --- CTX (cross-episode context tokens) ---
    ctx_k: int = 64
    """number of most-recent episode tokens the actor/critic cross-attend over"""
    tok_dim: int = 64
    """dimensionality of a per-episode context token"""
    raw_episodes: int = 1024
    """ring capacity of raw episode step traces kept for the aux encoder loss"""
    aux_batch: int = 8
    """episodes sampled per auxiliary encoder update"""

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

    CTX addition: an int64 `ep_idx` column = number of completed episodes at the time the transition
    was collected. Its context window is the token span [ep_idx-K, ep_idx).
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
        self.ep_idx = np.zeros((self.max_size, 1), dtype=np.int64)

        self.priority = torch.zeros(self.max_size, device=device)
        self.max_priority = 1.0

        self.normalize_actions = float(max_action)

    def add(self, state, action, next_state, reward, done, ep_idx):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.ep_idx[self.ptr] = ep_idx

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
            torch.tensor(self.ep_idx[self.ind].reshape(-1), dtype=torch.long, device=self.device),
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())


MAX_EP_LEN = 1000  # positional-embedding table size; MuJoCo tasks truncate at 1000 steps


class TokenEncoder(nn.Module):
    """Compress a padded episode trace (B, L, step_dim) -> a tok_dim token with a CAUSAL transformer:
    per-step projection + learned positional embedding, a 2-layer pre-LN causal encoder (step t attends
    only to steps <= t), and a CAUSAL READOUT — the token is the hidden state at the last valid step.
    No pooling anywhere: attention over the ordered steps does the compression. Trained OFF the RL
    gradient path by an aux episode-return head."""

    def __init__(self, step_dim, tok_dim=64, d_model=64, nhead=1, ffn=128, nlayers=2):
        super().__init__()
        self.step_proj = nn.Linear(step_dim, d_model)
        self.pos_emb = nn.Embedding(MAX_EP_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False)
        self.out_proj = nn.Linear(d_model, tok_dim)
        self.aux = nn.Linear(tok_dim, 1)

    def _encode_seq(self, steps, mask, positions):
        # steps: (B, L, step_dim); mask: (B, L) 1.0 valid; positions: (B, L) long TRUE step indices.
        # Returns the full hidden sequence (B, L, d_model).
        L = steps.shape[1]
        x = self.step_proj(steps) + self.pos_emb(positions.clamp(max=MAX_EP_LEN - 1))
        # bool masks (True == disallowed) for both; same type avoids the mismatched-mask deprecation
        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=steps.device), diagonal=1
        )  # (L, L): step t cannot attend to steps > t
        pad = mask == 0  # (B, L) True where padded (key masked out)
        # Valid query rows are never fully masked (padding is contiguous at the end, so keys 0..t<L are
        # allowed by both masks), and padded query rows keep keys 0..L-1 — no all-masked softmax / NaN.
        return self.transformer(x, mask=causal, src_key_padding_mask=pad)

    def forward(self, steps, mask, positions):
        h = self._encode_seq(steps, mask, positions)  # (B, L, d_model)
        last_idx = (mask.sum(1).long() - 1).clamp(min=0)  # (B,) last valid step
        readout = h[torch.arange(h.shape[0], device=h.device), last_idx]  # (B, d_model)
        return avg_l1_norm(self.out_proj(readout))  # (B, tok_dim)

    def predict_return(self, tok):
        return self.aux(tok)


class ContextAttention(nn.Module):
    """Single-head cross-attention: a host-trunk query attends over the K most recent episode tokens.
    Output projection is ZERO-INITIALIZED, so `ctx == 0` at init and the pathway must earn influence."""

    def __init__(self, hdim, tok_dim=64, dim=64, ctx_out=64, n_age=8):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(hdim, dim)
        self.k = nn.Linear(tok_dim, dim)
        self.v = nn.Linear(tok_dim, dim)
        self.age_emb = nn.Embedding(n_age, tok_dim)
        self.out = nn.Linear(dim, ctx_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q_in, tok_win, tok_age, tok_mask):
        # q_in: (B, hdim); tok_win: (B, K, tok_dim); tok_age: (B, K) long>=0; tok_mask: (B, K) bool
        bucket = torch.log2(tok_age.float() + 1.0).floor().long().clamp(0, 7)
        toks = tok_win + self.age_emb(bucket)  # (B, K, tok_dim)
        k = self.k(toks)  # (B, K, dim)
        v = self.v(toks)
        q = self.q(q_in).unsqueeze(1)  # (B, 1, dim)
        logits = (q * k).sum(-1) / (self.dim ** 0.5)  # (B, K)

        mask = tok_mask.bool()
        no_valid = ~mask.any(dim=1, keepdim=True)  # (B, 1) rows with no valid token
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, neg)
        # guard the all-masked softmax (NaN) by feeding zero logits for empty rows, then zero the output
        logits = torch.where(no_valid, torch.zeros_like(logits), logits)
        attn = torch.softmax(logits, dim=1)  # (B, K)
        out = (attn.unsqueeze(-1) * v).sum(1)  # (B, dim)
        out = torch.where(no_valid, torch.zeros_like(out), out)
        ctx = self.out(out)  # (B, ctx_out); == 0 at init
        return ctx, attn, mask


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, tok_dim=64, zs_dim=256, hdim=256, ctx_out=64):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.attn = ContextAttention(hdim, tok_dim, ctx_out=ctx_out)
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
    def __init__(self, state_dim, action_dim, tok_dim=64, zs_dim=256, hdim=256, ctx_out=64):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.attn = ContextAttention(hdim, tok_dim, ctx_out=ctx_out)
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
        # single shared context, queried by the q1-branch normalized feature, appended to both twins
        ctx, self._last_attn, self._last_mask = self.attn(q1f, tok_win, tok_age, tok_mask)

        q1 = torch.cat([q1f, embeddings, ctx], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        q2f = avg_l1_norm(self.q02(sa))
        q2 = torch.cat([q2f, embeddings, ctx], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)


class EpisodeStore:
    """Ring of raw episode traces (for the aux loss) + a GPU token cache indexed by global episode id.

    tokens: LIVE cache, written at add_episode and refreshed by the aux encoder step.
    tokens_frozen: snapshot copied from tokens at each hard target update; used by ALL replay forwards.
    """

    def __init__(self, step_dim, tok_dim, tok_encoder, device, raw_episodes=1024, max_episodes=MAX_EPISODES):
        self.step_dim = step_dim
        self.tok_dim = tok_dim
        self.tok_encoder = tok_encoder
        self.device = device
        self.raw_episodes = raw_episodes
        self.max_episodes = max_episodes

        self.raw_steps = [None] * raw_episodes  # each: np.float32 (ep_len, step_dim)
        self.raw_returns = np.zeros(raw_episodes, dtype=np.float32)
        self.raw_ep_global_idx = np.full(raw_episodes, -1, dtype=np.int64)

        self.tokens = torch.zeros(max_episodes, tok_dim, device=device)
        self.tokens_frozen = torch.zeros(max_episodes, tok_dim, device=device)
        self.n_ep = 0

    def add_episode(self, steps_array, ep_return):
        idx = self.n_ep
        slot = idx % self.raw_episodes
        self.raw_steps[slot] = steps_array
        self.raw_returns[slot] = float(ep_return)
        self.raw_ep_global_idx[slot] = idx

        with torch.no_grad():
            L = steps_array.shape[0]
            s = torch.as_tensor(steps_array[None], dtype=torch.float32, device=self.device)  # (1, L, step_dim)
            m = torch.ones(1, L, device=self.device)
            pos = torch.arange(L, device=self.device).unsqueeze(0)  # full-episode true step indices
            tok = self.tok_encoder(s, m, pos)[0]
        if idx < self.max_episodes:
            self.tokens[idx] = tok
        self.n_ep += 1


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_action, args: Args, device, writer: SummaryWriter, step_dim):
        self.args = args
        self.device = device
        self.writer = writer
        self.ctx_k = args.ctx_k

        self.actor = Actor(state_dim, action_dim, args.tok_dim, args.zs_dim, args.hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, args.tok_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = LAPBuffer(state_dim, action_dim, device, args.buffer_size, args.batch_size, max_action)

        # CTX: token encoder (trained off the RL path) + immutable token store + return normalizer
        self.tok_encoder = TokenEncoder(step_dim, args.tok_dim).to(device)
        self.tok_encoder_optimizer = torch.optim.Adam(self.tok_encoder.parameters(), lr=3e-4)
        self.store = EpisodeStore(step_dim, args.tok_dim, self.tok_encoder, device, args.raw_episodes)
        self.ret_norm = ReturnNorm()
        self.last_aux_loss = 0.0

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

    def build_window(self, ep_idx, tokens):
        """Given ep_idx (B,) and a token tensor, return (tok_win (B,K,tok_dim), age (B,K), mask (B,K)).

        Token index for window position p in [0,K) is ep_idx-K+p; valid where that index >= 0. The
        most recent token (p=K-1) has age 0; age = K-1-p (independent of ep_idx). Invalid slots are
        clamped for the gather and zeroed / masked out.
        """
        K = self.ctx_k
        p = torch.arange(K, device=self.device)  # (K,)
        idx = ep_idx.unsqueeze(1) - K + p.unsqueeze(0)  # (B, K)
        mask = idx >= 0
        idx_clamped = idx.clamp(min=0, max=tokens.shape[0] - 1)
        tok_win = tokens[idx_clamped]  # (B, K, tok_dim)
        tok_win = tok_win * mask.unsqueeze(-1).float()  # zero invalid slots
        age = (K - 1 - p).unsqueeze(0).expand(ep_idx.shape[0], K).long()
        return tok_win, age, mask

    def add_episode(self, steps_array, ep_return):
        self.ret_norm.update(float(ep_return))
        self.store.add_episode(steps_array, ep_return)

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
            # LIVE tokens, current episode count
            ep_idx = torch.tensor([self.store.n_ep], dtype=torch.long, device=self.device)
            tok_win, tok_age, tok_mask = self.build_window(ep_idx, self.store.tokens)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs, tok_win, tok_age, tok_mask)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs, tok_win, tok_age, tok_mask)

            if use_exploration:
                action = action + torch.randn_like(action) * self.args.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def aux_encoder_update(self):
        """Sample resident episodes, predict their normalized returns, and refresh their live tokens."""
        n_resident = min(self.store.n_ep, self.store.raw_episodes)
        if n_resident < 1:
            return
        b = min(self.args.aux_batch, n_resident)
        sel = np.random.choice(n_resident, size=b, replace=False)

        # subsample each episode to <=512 CONTIGUOUS steps from a random offset (contiguity preserves
        # the causal structure; positional embeddings use the TRUE step indices of the slice)
        sub, poss, lens, targets = [], [], [], []
        for slot in sel:
            s = self.store.raw_steps[slot]
            L = s.shape[0]
            n = min(512, L)
            off = 0 if L <= n else int(np.random.randint(0, L - n + 1))
            sub.append(s[off : off + n])
            poss.append(np.arange(off, off + n, dtype=np.int64))
            lens.append(n)
            targets.append(self.ret_norm.normalize(self.store.raw_returns[slot]))

        Lmax = max(lens)
        batch = np.zeros((b, Lmax, self.store.step_dim), dtype=np.float32)
        maskn = np.zeros((b, Lmax), dtype=np.float32)
        posn = np.zeros((b, Lmax), dtype=np.int64)
        for i, s in enumerate(sub):
            batch[i, : lens[i]] = s
            maskn[i, : lens[i]] = 1.0
            posn[i, : lens[i]] = poss[i]
        batch_t = torch.as_tensor(batch, device=self.device)
        mask_t = torch.as_tensor(maskn, device=self.device)
        pos_t = torch.as_tensor(posn, device=self.device)
        targets_t = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=self.device).reshape(-1, 1)

        tok = self.tok_encoder(batch_t, mask_t, pos_t)
        aux_loss = F.mse_loss(self.tok_encoder.predict_return(tok), targets_t)

        self.tok_encoder_optimizer.zero_grad()
        aux_loss.backward()
        self.tok_encoder_optimizer.step()
        self.last_aux_loss = aux_loss.item()

        # refresh the LIVE token cache with the freshly (post-step) encoded tokens
        with torch.no_grad():
            fresh = self.tok_encoder(batch_t, mask_t, pos_t)
            for i, slot in enumerate(sel):
                gidx = int(self.store.raw_ep_global_idx[slot])
                if 0 <= gidx < self.store.max_episodes:
                    self.store.tokens[gidx] = fresh[i]

    def train(self):
        self.training_steps += 1

        state, action, next_state, reward, not_done, ep_idx = self.replay_buffer.sample()
        # s and s' share the collection-time context (episodes complete only at boundaries), frozen cache
        tok_win, tok_age, tok_mask = self.build_window(ep_idx, self.store.tokens_frozen)

        # update encoder: predict the next state's embedding from (zs, action)
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # update critic
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.args.target_policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            next_action = (self.actor_target(next_state, fixed_target_zs, tok_win, tok_age, tok_mask) + noise).clamp(
                -1, 1
            )

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            Q_target = self.critic_target(
                next_state, next_action, fixed_target_zsa, fixed_target_zs, tok_win, tok_age, tok_mask
            ).min(1, keepdim=True)[0]
            Q_target = reward + not_done * self.args.gamma * Q_target.clamp(self.min_target, self.max_target)

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs, tok_win, tok_age, tok_mask)
        td_loss = (Q - Q_target).abs()
        critic_loss = lap_huber(td_loss, self.args.min_priority)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update LAP priorities
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.lap_alpha).detach()
        self.replay_buffer.update_priority(priority)

        # update actor (delayed) + auxiliary token-encoder update on the same cadence
        if self.training_steps % self.args.policy_freq == 0:
            actor_action = self.actor(state, fixed_zs, tok_win, tok_age, tok_mask)
            actor_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_action)
            actor_Q = self.critic(state, actor_action, actor_fixed_zsa, fixed_zs, tok_win, tok_age, tok_mask)
            actor_loss = -actor_Q.mean()

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
            self.writer.add_scalar("losses/q_values", Q.mean().item(), self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target, self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target, self.training_steps)

            with torch.no_grad():
                attn = self.critic._last_attn  # (B, K) from the current-critic forward above
                m = self.critic._last_mask.float()
                ent = -(attn.clamp(min=1e-12).log() * attn * m).sum(1)  # (B,) entropy over valid slots
                ctx_gate = self.actor.attn.out.weight.norm().item()
            self.writer.add_scalar("debug/aux_loss", self.last_aux_loss, self.training_steps)
            self.writer.add_scalar("debug/ctx_gate", ctx_gate, self.training_steps)
            self.writer.add_scalar("debug/ctx_attn_entropy", ent.mean().item(), self.training_steps)
            self.writer.add_scalar("debug/n_ep", self.store.n_ep, self.training_steps)

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
    # CTX: episode traces are built from RAW obs (not the hist-augmented obs), normalized action, reward
    step_dim = int(state_dim + action_dim + 1)
    agent = TD7Agent(state_dim_aug, action_dim, max_action, args, device, writer, step_dim)

    start_time = time.time()
    allow_train = False

    obs, _ = envs.reset(seed=args.seed)
    act_hist = np.zeros((args.hist_k, action_dim), dtype=np.float32)  # most-recent first
    cur_ep_steps = []  # raw (obs, normalized action, reward) rows for the ongoing episode
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
        cur_ep_steps.append(
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
        # CTX: ep_idx is the count of COMPLETED episodes when this transition was collected
        agent.replay_buffer.add(obs_aug, actions[0], real_next_obs_aug, rewards[0], done, agent.store.n_ep)
        act_hist = (np.zeros((args.hist_k, action_dim), dtype=np.float32)
                    if (terminations[0] or truncations[0]) else next_hist)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training (per-step when not checkpointing)
        if allow_train and not args.use_checkpoints:
            agent.train()

        # episode boundary: log return, compress the episode into a token, run burst training, enable training
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_return = float(info["episode"]["r"])
                    ep_length = int(info["episode"]["l"])
                    print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # CTX: encode this episode into a fresh token BEFORE the burst so it trains with it
                    agent.add_episode(np.asarray(cur_ep_steps, dtype=np.float32), ep_return)
                    cur_ep_steps = []

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

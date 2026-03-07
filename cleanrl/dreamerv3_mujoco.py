# DreamerV3 for MuJoCo continuous control (proprioceptive).
# Full world-model-based agent: RSSM dynamics, imagination rollouts, actor-critic.
# Ported from the original JAX implementation to single-file PyTorch.
#
# Key differences from PPO variants in this repo:
# - Off-policy: learns from a replay buffer of past episodes
# - Model-based: learns a world model (RSSM) and trains actor-critic in imagination
# - Actor uses REINFORCE (not PPO clipping)
# - Critic uses symexp-twohot distributional head
# - Return normalization via percentiles (not advantage normalization)
# - Learned discount via continue predictor

import copy
import math
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
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False

    # Environment
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    num_envs: int = 1

    # RSSM
    rssm_deter: int = 256
    rssm_hidden: int = 128
    rssm_stoch: int = 16
    rssm_classes: int = 16
    rssm_blocks: int = 8
    free_nats: float = 1.0
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    unimix: float = 0.01

    # Networks
    mlp_units: int = 256
    mlp_layers: int = 2
    act_fn: str = "silu"
    num_bins: int = 255

    # Training
    learning_rate: float = 1e-4
    eps: float = 1e-8
    batch_size: int = 16
    batch_length: int = 32
    train_ratio: float = 64.0
    imag_last: int = 8  # only start imagination from last K timesteps (0=all)
    max_grad_norm: float = 1000.0
    weight_decay: float = 0.0
    compile: bool = True  # torch.compile for speed
    amp: bool = False  # automatic mixed precision (bfloat16) - compile alone is faster

    # Imagination
    imag_horizon: int = 15

    # Actor-Critic
    horizon: int = 333
    lam: float = 0.95
    actor_entropy: float = 3e-4
    actor_minstd: float = 0.1
    actor_maxstd: float = 1.0
    slow_target_rate: float = 0.02
    slow_reg: float = 1.0
    return_norm_rate: float = 0.01
    return_norm_perclo: float = 5.0
    return_norm_perchi: float = 95.0
    return_norm_limit: float = 1.0

    # Replay
    replay_size: int = 1_000_000
    prefill_steps: int = 5000


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def build_symexp_bins(num_bins):
    if num_bins % 2 == 1:
        half = torch.linspace(-20.0, 0.0, (num_bins - 1) // 2 + 1)
        half = symexp(half)
        bins = torch.cat([half, -half[:-1].flip(0)])
    else:
        half = torch.linspace(-20.0, 0.0, num_bins // 2)
        half = symexp(half)
        bins = torch.cat([half, -half.flip(0)])
    return bins


def twohot_encode(target, bins):
    """Two-hot encode scalar targets into bin space."""
    below = (bins <= target.unsqueeze(-1)).sum(-1) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1
    below_val = bins[below]
    above_val = bins[above]
    weight = (target - below_val) / (above_val - below_val + 1e-8)
    weight = weight.clamp(0, 1)
    result = torch.zeros(*target.shape, len(bins), device=target.device)
    result.scatter_(-1, below.unsqueeze(-1), (1 - weight).unsqueeze(-1))
    result.scatter_(-1, above.unsqueeze(-1), weight.unsqueeze(-1))
    return result


def twohot_loss(logits, target, bins):
    """Cross-entropy loss between predicted logits and two-hot encoded target."""
    two_hot = twohot_encode(target, bins)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(two_hot * log_probs).sum(-1)


def twohot_mean(logits, bins):
    """Expected value under the predicted distribution."""
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(-1)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, layers, act="silu", norm=True,
                 out_scale=1.0):
        super().__init__()
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[act]
        dims = [in_dim] + [hidden] * layers + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no act/norm on last layer
                if norm:
                    mods.append(RMSNorm(dims[i + 1]))
                mods.append(act_fn())
        self.net = nn.Sequential(*mods)
        # Init last layer
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        if out_scale == 0.0:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            fan_in = last.weight.shape[1]
            nn.init.trunc_normal_(last.weight, std=out_scale / math.sqrt(fan_in))
            nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)


class BlockLinear(nn.Module):
    """Block-diagonal linear layer. Each block processes its own features."""
    def __init__(self, in_features, out_features, blocks):
        super().__init__()
        assert in_features % blocks == 0 and out_features % blocks == 0
        self.blocks = blocks
        self.in_per = in_features // blocks
        self.out_per = out_features // blocks
        self.weight = nn.Parameter(torch.empty(blocks, self.in_per, self.out_per))
        self.bias = nn.Parameter(torch.zeros(blocks, self.out_per))
        for b in range(blocks):
            nn.init.xavier_uniform_(self.weight[b])

    def forward(self, x):
        *batch, _ = x.shape
        x = x.reshape(*batch, self.blocks, self.in_per)
        out = torch.einsum('...bi,bio->...bo', x, self.weight) + self.bias
        return out.reshape(*batch, self.blocks * self.out_per)


# ---------------------------------------------------------------------------
# RSSM World Model
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    def __init__(self, act_dim, deter=512, hidden=256, stoch=32, classes=32,
                 blocks=8, unimix=0.01, act="silu"):
        super().__init__()
        self.deter_dim = deter
        self.hidden_dim = hidden
        self.stoch = stoch
        self.classes = classes
        self.blocks = blocks
        self.unimix = unimix
        self.stoch_dim = stoch * classes
        self.feat_dim = deter + self.stoch_dim
        act_fn_cls = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[act]

        # GRU core input projections
        self.dyn_in_deter = nn.Linear(deter, hidden)
        self.dyn_in_stoch = nn.Linear(self.stoch_dim, hidden)
        self.dyn_in_act = nn.Linear(act_dim, hidden)
        self.dyn_norm0 = RMSNorm(hidden)
        self.dyn_norm1 = RMSNorm(hidden)
        self.dyn_norm2 = RMSNorm(hidden)
        self.dyn_act0 = act_fn_cls()
        self.dyn_act1 = act_fn_cls()
        self.dyn_act2 = act_fn_cls()

        # GRU core: block-diagonal layers
        core_in = deter + blocks * 3 * hidden
        assert core_in % blocks == 0 or True  # we handle non-divisible below
        # For the core, each block gets: deter//blocks + 3*hidden features
        self.core_in_per_block = deter // blocks + 3 * hidden
        core_in_total = blocks * self.core_in_per_block
        self.dyn_hidden = BlockLinear(core_in_total, deter, blocks)
        self.dyn_hidden_norm = RMSNorm(deter)
        self.dyn_hidden_act = act_fn_cls()
        self.dyn_gru = BlockLinear(deter, 3 * deter, blocks)

        # Prior (predict stoch from deter only)
        self.prior = nn.Sequential(
            nn.Linear(deter, hidden), RMSNorm(hidden), act_fn_cls(),
            nn.Linear(hidden, hidden), RMSNorm(hidden), act_fn_cls(),
            nn.Linear(hidden, stoch * classes),
        )

        # Posterior (predict stoch from deter + tokens)
        self.post = nn.Sequential(
            nn.Linear(deter + hidden, hidden),  # hidden here is token_dim (encoder output)
            RMSNorm(hidden), act_fn_cls(),
            nn.Linear(hidden, stoch * classes),
        )
        # Note: post input dim will be set dynamically via a wrapper

    def set_post_input_dim(self, token_dim):
        """Re-create posterior network with correct input dim."""
        hidden = self.hidden_dim
        act_fn_cls = type(self.dyn_act0)  # reuse same activation
        self.post = nn.Sequential(
            nn.Linear(self.deter_dim + token_dim, hidden),
            RMSNorm(hidden), act_fn_cls(),
            nn.Linear(hidden, self.stoch * self.classes),
        )

    def initial(self, batch_size, device):
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch, self.classes, device=device),
        }

    def get_feat(self, state):
        """Flatten state into feature vector."""
        deter = state['deter']
        stoch = state['stoch'].reshape(*state['stoch'].shape[:-2], -1)
        return torch.cat([deter, stoch], -1)

    def _sample_categorical(self, logits):
        """Sample from categorical with straight-through and uniform mixing."""
        # logits: (..., stoch, classes)
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / self.classes
        probs = (1 - self.unimix) * probs + self.unimix * uniform
        # Straight-through: sample one-hot, but use probs for gradient
        # Use torch.multinomial instead of Categorical for speed
        flat = probs.reshape(-1, self.classes)
        indices = torch.multinomial(flat, 1).squeeze(-1)
        indices = indices.reshape(logits.shape[:-1])
        one_hot = F.one_hot(indices, self.classes).float()
        return one_hot + probs - probs.detach()

    def _gru_core(self, deter, stoch, action):
        """GRU-style dynamics core with block-diagonal structure."""
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        # Soft-clip action
        action = action / action.abs().clamp(min=1).detach()

        # Project each input
        x0 = self.dyn_act0(self.dyn_norm0(self.dyn_in_deter(deter)))
        x1 = self.dyn_act1(self.dyn_norm1(self.dyn_in_stoch(stoch_flat)))
        x2 = self.dyn_act2(self.dyn_norm2(self.dyn_in_act(action)))

        # Combine: replicate combined features for each block
        combined = torch.cat([x0, x1, x2], -1)  # (B, 3*hidden)
        combined = combined.unsqueeze(1).expand(-1, self.blocks, -1)  # (B, blocks, 3*hidden)

        # Block-diagonal decomposition of deter
        B = deter.shape[0]
        deter_blocked = deter.reshape(B, self.blocks, self.deter_dim // self.blocks)

        # Concatenate within each block then flatten
        x = torch.cat([deter_blocked, combined], -1)  # (B, blocks, deter//blocks + 3*hidden)
        x = x.reshape(B, -1)  # (B, blocks * core_in_per_block)

        # Hidden layer
        x = self.dyn_hidden_act(self.dyn_hidden_norm(self.dyn_hidden(x)))

        # GRU gates
        gates = self.dyn_gru(x)  # (B, 3*deter)
        gates = gates.reshape(B, 3, self.deter_dim)
        reset_raw, cand_raw, update_raw = gates[:, 0], gates[:, 1], gates[:, 2]

        reset = torch.sigmoid(reset_raw)
        cand = torch.tanh(reset * cand_raw)
        update = torch.sigmoid(update_raw - 1)  # biased toward not updating
        new_deter = update * cand + (1 - update) * deter
        return new_deter

    def observe_step(self, state, token, action, is_first):
        """Single-step RSSM update with observation (posterior)."""
        B = is_first.shape[0]
        mask = (~is_first).float()

        # Reset state at episode boundaries
        deter = state['deter'] * mask.unsqueeze(-1)
        stoch = state['stoch'] * mask.unsqueeze(-1).unsqueeze(-1)
        action = action * mask.unsqueeze(-1)

        # Dynamics
        new_deter = self._gru_core(deter, stoch, action)

        # Posterior (uses observation tokens)
        post_input = torch.cat([new_deter, token], -1)
        post_logit = self.post(post_input).reshape(B, self.stoch, self.classes)
        post_stoch = self._sample_categorical(post_logit)

        # Prior (for KL loss)
        prior_logit = self.prior(new_deter).reshape(B, self.stoch, self.classes)

        new_state = {'deter': new_deter, 'stoch': post_stoch}
        return new_state, prior_logit, post_logit

    def observe(self, tokens, actions, is_first):
        """Process a sequence through the RSSM with observations (posterior).

        Args:
            tokens: (B, T, token_dim)
            actions: (B, T, act_dim) - previous actions
            is_first: (B, T) bool
        Returns:
            states: dict with deter (B,T,D) and stoch (B,T,S,C)
            prior_logits: (B, T, S, C)
            post_logits: (B, T, S, C)
        """
        B, T, _ = tokens.shape
        device = tokens.device
        state = self.initial(B, device)

        all_deter, all_stoch = [], []
        all_prior, all_post = [], []

        for t in range(T):
            state, prior_logit, post_logit = self.observe_step(
                state, tokens[:, t], actions[:, t], is_first[:, t])
            all_deter.append(state['deter'])
            all_stoch.append(state['stoch'])
            all_prior.append(prior_logit)
            all_post.append(post_logit)

        states = {
            'deter': torch.stack(all_deter, 1),
            'stoch': torch.stack(all_stoch, 1),
        }
        return states, torch.stack(all_prior, 1), torch.stack(all_post, 1)

    def imagine_step(self, state, action):
        """Single-step RSSM forward without observation (prior only)."""
        new_deter = self._gru_core(state['deter'], state['stoch'], action)
        prior_logit = self.prior(new_deter).reshape(
            new_deter.shape[0], self.stoch, self.classes)
        prior_stoch = self._sample_categorical(prior_logit)
        return {'deter': new_deter, 'stoch': prior_stoch}


# ---------------------------------------------------------------------------
# Encoder / Decoder / Heads
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """MLP encoder with symlog input preprocessing."""
    def __init__(self, obs_dim, units, layers, act="silu"):
        super().__init__()
        self.mlp = MLP(obs_dim, units, units, layers, act=act, out_scale=1.0)

    def forward(self, obs):
        return self.mlp(symlog(obs))


class Decoder(nn.Module):
    """MLP decoder predicting observations in symlog space."""
    def __init__(self, feat_dim, obs_dim, units, layers, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, obs_dim, units, layers, act=act, out_scale=1.0)

    def forward(self, feat):
        return self.mlp(feat)

    def loss(self, feat, obs_target):
        """Symlog MSE loss."""
        pred = self.forward(feat)
        return 0.5 * (pred - symlog(obs_target)).pow(2).sum(-1)


class RewardHead(nn.Module):
    """MLP head predicting rewards via symexp-twohot distribution."""
    def __init__(self, feat_dim, units, num_bins=255, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, num_bins, units, 1, act=act, out_scale=0.0)
        self.register_buffer('bins', build_symexp_bins(num_bins))

    def forward(self, feat):
        return self.mlp(feat)

    def loss(self, feat, target):
        logits = self.forward(feat)
        return twohot_loss(logits, target, self.bins)

    def predict(self, feat):
        logits = self.forward(feat)
        return twohot_mean(logits, self.bins)


class ContinueHead(nn.Module):
    """MLP head predicting episode continuation (Bernoulli)."""
    def __init__(self, feat_dim, units, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, 1, units, 1, act=act, out_scale=1.0)

    def forward(self, feat):
        return self.mlp(feat).squeeze(-1)

    def loss(self, feat, target):
        logits = self.forward(feat)
        return F.binary_cross_entropy_with_logits(logits, target)

    def predict(self, feat):
        return torch.sigmoid(self.forward(feat))


class Actor(nn.Module):
    """MLP actor outputting Normal distribution."""
    def __init__(self, feat_dim, act_dim, units, layers, act="silu",
                 minstd=0.1, maxstd=1.0):
        super().__init__()
        self.minstd = minstd
        self.maxstd = maxstd
        self.mlp = MLP(feat_dim, 2 * act_dim, units, layers, act=act, out_scale=0.01)
        self.act_dim = act_dim

    def forward(self, feat):
        out = self.mlp(feat)
        mean, raw_std = out.split(self.act_dim, -1)
        std = self.minstd + (self.maxstd - self.minstd) * torch.sigmoid(raw_std)
        return Normal(mean, std)


class Critic(nn.Module):
    """MLP critic with symexp-twohot distributional head."""
    def __init__(self, feat_dim, units, layers, num_bins=255, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, num_bins, units, layers, act=act, out_scale=0.0)
        self.register_buffer('bins', build_symexp_bins(num_bins))

    def forward(self, feat):
        return self.mlp(feat)

    def predict(self, feat):
        return twohot_mean(self.forward(feat), self.bins)

    def loss(self, logits, target):
        return twohot_loss(logits, target, self.bins)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class EpisodeReplayBuffer:
    """Stores completed episodes. Samples contiguous subsequences."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.episodes = []
        self.total_steps = 0

    @property
    def num_steps(self):
        return self.total_steps

    def add_episode(self, episode):
        length = len(episode['reward'])
        if length < 2:
            return
        self.episodes.append(episode)
        self.total_steps += length
        while self.total_steps > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed['reward'])

    def can_sample(self, batch_length):
        return any(len(ep['reward']) >= batch_length for ep in self.episodes)

    def sample(self, batch_size, batch_length):
        """Sample batch of contiguous subsequences."""
        valid = [ep for ep in self.episodes if len(ep['reward']) >= batch_length]
        if not valid:
            valid = self.episodes

        batch = {k: [] for k in valid[0].keys()}
        for _ in range(batch_size):
            ep = valid[np.random.randint(len(valid))]
            ep_len = len(ep['reward'])
            if ep_len >= batch_length:
                start = np.random.randint(0, ep_len - batch_length + 1)
            else:
                start = 0
            end = start + batch_length
            for k in batch:
                chunk = ep[k][start:end]
                if len(chunk) < batch_length:
                    pad_shape = (batch_length - len(chunk),) + chunk.shape[1:]
                    chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=chunk.dtype)])
                batch[k].append(chunk)

        return {k: np.stack(v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Return Normalization (percentile-based)
# ---------------------------------------------------------------------------

class PercentileReturnNorm:
    def __init__(self, rate=0.01, perclo=5.0, perchi=95.0, limit=1.0):
        self.rate = rate
        self.perclo = perclo
        self.perchi = perchi
        self.limit = limit
        self.lo = 0.0
        self.hi = 0.0
        self.initialized = False

    def update(self, returns):
        """Update running percentile estimates."""
        flat = returns.detach().cpu().numpy().ravel()
        lo = np.percentile(flat, self.perclo)
        hi = np.percentile(flat, self.perchi)
        if not self.initialized:
            self.lo = lo
            self.hi = hi
            self.initialized = True
        else:
            self.lo += self.rate * (lo - self.lo)
            self.hi += self.rate * (hi - self.hi)

    def scale(self):
        return max(self.limit, self.hi - self.lo)


# ---------------------------------------------------------------------------
# Lambda Returns
# ---------------------------------------------------------------------------

def compute_lambda_returns(rew, con, val, lam):
    """Compute TD(lambda) returns for imagined trajectories.

    Args:
        rew: (B, H+1) rewards at each imagined state
        con: (B, H+1) continue probabilities (learned discount)
        val: (B, H+1) value estimates
        lam: float, lambda parameter
    Returns:
        (B, H) lambda returns for states 0..H-1
    """
    H = rew.shape[1] - 1
    rets = [val[:, -1]]
    for t in reversed(range(H)):
        ret_t = rew[:, t + 1] + con[:, t + 1] * ((1 - lam) * val[:, t + 1] + lam * rets[-1])
        rets.append(ret_t)
    rets = list(reversed(rets[:-1]))
    return torch.stack(rets, 1)


# ---------------------------------------------------------------------------
# make_env
# ---------------------------------------------------------------------------

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(
            f"|{k}|{v}|" for k, v in vars(args).items()))

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision('high')  # enable TF32 for faster matmul

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name)
         for i in range(args.num_envs)])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    # Build models
    encoder = Encoder(obs_dim, args.mlp_units, args.mlp_layers, act=args.act_fn).to(device)
    token_dim = args.mlp_units  # encoder output dimension

    rssm = RSSM(
        act_dim, deter=args.rssm_deter, hidden=args.rssm_hidden,
        stoch=args.rssm_stoch, classes=args.rssm_classes,
        blocks=args.rssm_blocks, unimix=args.unimix, act=args.act_fn,
    ).to(device)
    rssm.set_post_input_dim(token_dim)
    rssm = rssm.to(device)  # re-to after posterior recreation

    feat_dim = rssm.feat_dim

    decoder = Decoder(feat_dim, obs_dim, args.mlp_units, args.mlp_layers, act=args.act_fn).to(device)
    reward_head = RewardHead(feat_dim, args.mlp_units, args.num_bins, act=args.act_fn).to(device)
    continue_head = ContinueHead(feat_dim, args.mlp_units, act=args.act_fn).to(device)
    actor = Actor(feat_dim, act_dim, args.mlp_units, args.mlp_layers,
                  act=args.act_fn, minstd=args.actor_minstd, maxstd=args.actor_maxstd).to(device)
    critic = Critic(feat_dim, args.mlp_units, args.mlp_layers, args.num_bins, act=args.act_fn).to(device)
    slow_critic = copy.deepcopy(critic).to(device)
    slow_critic.requires_grad_(False)

    # Optimizers
    wm_params = (list(encoder.parameters()) + list(rssm.parameters()) +
                 list(decoder.parameters()) + list(reward_head.parameters()) +
                 list(continue_head.parameters()))
    wm_opt = torch.optim.Adam(wm_params, lr=args.learning_rate, eps=args.eps,
                              weight_decay=args.weight_decay)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.learning_rate, eps=args.eps)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.learning_rate, eps=args.eps)

    # torch.compile for speed (compiles step functions called in tight loops)
    # Note: don't compile actor - it returns Normal distribution objects
    # which torch.compile doesn't handle reliably
    if args.compile:
        rssm.observe_step = torch.compile(rssm.observe_step)
        rssm.imagine_step = torch.compile(rssm.imagine_step)
        encoder = torch.compile(encoder)
        decoder = torch.compile(decoder)
        reward_head = torch.compile(reward_head)
        continue_head = torch.compile(continue_head)
        critic = torch.compile(critic)
        slow_critic = torch.compile(slow_critic)

    # AMP setup
    amp_dtype = torch.bfloat16 if args.amp and torch.cuda.is_bf16_supported() else None
    amp_enabled = amp_dtype is not None

    bins = build_symexp_bins(args.num_bins).to(device)
    return_norm = PercentileReturnNorm(
        rate=args.return_norm_rate,
        perclo=args.return_norm_perclo,
        perchi=args.return_norm_perchi,
        limit=args.return_norm_limit,
    )

    # Replay buffer
    replay = EpisodeReplayBuffer(args.replay_size)
    cont_target_val = 1.0 - 1.0 / args.horizon  # ~0.997

    # Episode tracking per env
    ongoing = [None] * args.num_envs
    def init_ongoing(env_id, obs):
        ongoing[env_id] = {
            'obs': [obs.copy()],
            'prev_act': [np.zeros(act_dim, dtype=np.float32)],
            'reward': [0.0],
            'is_first': [True],
            'is_terminal': [False],
        }

    def finish_ongoing(env_id, final_obs, last_action, last_reward, terminated):
        ep = ongoing[env_id]
        ep['obs'].append(final_obs.copy())
        ep['prev_act'].append(last_action.copy())
        ep['reward'].append(float(last_reward))
        ep['is_first'].append(False)
        ep['is_terminal'].append(bool(terminated))
        episode = {k: np.array(v, dtype=np.float32 if k not in ('is_first', 'is_terminal')
                                else bool) for k, v in ep.items()}
        replay.add_episode(episode)

    def add_step(env_id, next_obs, action, reward, is_first, is_terminal):
        ep = ongoing[env_id]
        ep['obs'].append(next_obs.copy())
        ep['prev_act'].append(action.copy())
        ep['reward'].append(float(reward))
        ep['is_first'].append(bool(is_first))
        ep['is_terminal'].append(bool(is_terminal))

    # --- Training step ---
    free_nats_t = torch.tensor(args.free_nats, device=device)

    def train_step():
        batch_np = replay.sample(args.batch_size, args.batch_length)
        batch = {k: torch.tensor(v, device=device, dtype=torch.float32 if v.dtype != bool
                                 else torch.bool) for k, v in batch_np.items()}
        obs = batch['obs']           # (B, T, obs_dim)
        prev_act = batch['prev_act'] # (B, T, act_dim)
        rew = batch['reward']        # (B, T)
        is_first = batch['is_first'] # (B, T)
        is_term = batch['is_terminal']  # (B, T)
        B, T = obs.shape[:2]

        # Force is_first at sequence start (RSSM always resets at start)
        is_first = is_first.clone()
        is_first[:, 0] = True

        # ===== World Model =====
        wm_opt.zero_grad()

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
            tokens = encoder(obs.reshape(B * T, -1)).reshape(B, T, -1)
            states, prior_logits, post_logits = rssm.observe(tokens, prev_act, is_first)
            feat = rssm.get_feat(states)  # (B, T, feat_dim)

            # Reconstruction loss
            rec_loss = decoder.loss(feat.reshape(B * T, -1), obs.reshape(B * T, -1)).reshape(B, T).mean()

            # Reward prediction loss
            rew_loss = reward_head.loss(feat.reshape(B * T, -1), rew.reshape(B * T)).reshape(B, T).mean()

            # Continue prediction loss
            con_target = (~is_term).float() * cont_target_val
            con_logits = continue_head(feat.reshape(B * T, -1))
            con_loss = F.binary_cross_entropy_with_logits(
                con_logits, con_target.reshape(B * T)).mean()

            # KL losses (free nats) - compute in float32 for numerical stability
            prior_logits_f = prior_logits.float()
            post_logits_f = post_logits.float()
            prior_probs = F.softmax(prior_logits_f, -1)
            post_probs = F.softmax(post_logits_f, -1)
            prior_probs_um = (1 - args.unimix) * prior_probs + args.unimix / args.rssm_classes
            post_probs_um = (1 - args.unimix) * post_probs + args.unimix / args.rssm_classes

            # dyn loss: KL(sg(post) || prior) - trains prior to match posterior
            dyn_kl = (post_probs_um.detach() * (
                post_probs_um.detach().log() - prior_probs_um.log()
            )).sum(-1).sum(-1)
            dyn_loss = torch.maximum(dyn_kl, free_nats_t).mean()

            # rep loss: KL(post || sg(prior)) - trains posterior to be predictable
            rep_kl = (post_probs_um * (
                post_probs_um.log() - prior_probs_um.detach().log()
            )).sum(-1).sum(-1)
            rep_loss = torch.maximum(rep_kl, free_nats_t).mean()

            wm_loss = rec_loss + rew_loss + con_loss + args.dyn_scale * dyn_loss + args.rep_scale * rep_loss

        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm_params, args.max_grad_norm)
        wm_opt.step()

        # ===== Imagination =====
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                # Start imagination from last K posterior states
                K = min(args.imag_last, T) if args.imag_last > 0 else T
                BK = B * K
                start_feat = {k: v[:, -K:].reshape(BK, *v.shape[2:]).detach()
                              for k, v in states.items()}

                # Imagine forward
                img_feats = [rssm.get_feat(start_feat)]
                img_acts = []
                state = start_feat
                for h in range(args.imag_horizon):
                    action = actor(rssm.get_feat(state)).sample()
                    img_acts.append(action)
                    state = rssm.imagine_step(state, action)
                    img_feats.append(rssm.get_feat(state))

                img_feats = torch.stack(img_feats, 1)  # (BK, H+1, feat_dim)
                img_acts = torch.stack(img_acts, 1)     # (BK, H, act_dim)

                # Predict rewards, continues, values
                H1 = args.imag_horizon + 1
                img_feats_flat = img_feats.reshape(BK * H1, -1)
                img_rew = reward_head.predict(img_feats_flat).reshape(BK, H1)
                img_con = continue_head.predict(img_feats_flat).reshape(BK, H1)
                img_val = twohot_mean(critic(img_feats_flat), bins).reshape(BK, H1)
                img_slow_val = twohot_mean(slow_critic(img_feats_flat), bins).reshape(BK, H1)

            # Lambda returns in float32
            img_rew = img_rew.float()
            img_con = img_con.float()
            img_slow_val = img_slow_val.float()
            returns = compute_lambda_returns(img_rew, img_con, img_slow_val, args.lam)

            # Return normalization
            return_norm.update(returns)
            ret_scale = return_norm.scale()
            advantages = (returns - img_slow_val[:, :-1]) / ret_scale

            # Importance weights (cumulative continue)
            weight = torch.cumprod(img_con[:, :-1], dim=1)  # (BK, H)

        # ===== Actor =====
        actor_opt.zero_grad()

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
            policy_dist = actor(img_feats[:, :-1].reshape(BK * args.imag_horizon, -1).detach())
            logpi = policy_dist.log_prob(
                img_acts.reshape(BK * args.imag_horizon, -1).detach()
            ).sum(-1).reshape(BK, args.imag_horizon)
            entropy = policy_dist.entropy().sum(-1).reshape(BK, args.imag_horizon)

            actor_loss = -(weight.detach() * (
                logpi.float() * advantages.detach() + args.actor_entropy * entropy.float()
            )).mean()

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        actor_opt.step()

        # ===== Critic =====
        critic_opt.zero_grad()

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
            critic_feats = img_feats[:, :-1].reshape(BK * args.imag_horizon, -1).detach()
            val_logits = critic(critic_feats)
            slow_val_pred = twohot_mean(
                slow_critic(critic_feats), bins
            ).detach()

            loss_returns = twohot_loss(
                val_logits, returns.reshape(-1).detach(), bins
            ).reshape(BK, args.imag_horizon)
            loss_slow = twohot_loss(
                val_logits, slow_val_pred.reshape(-1), bins
            ).reshape(BK, args.imag_horizon)
            critic_loss = (weight.detach() * (loss_returns + args.slow_reg * loss_slow)).mean()

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        critic_opt.step()

        # Update slow critic
        with torch.no_grad():
            for p, sp in zip(critic.parameters(), slow_critic.parameters()):
                sp.data.lerp_(p.data, args.slow_target_rate)

        return {
            'wm_loss': wm_loss.item(),
            'rec_loss': rec_loss.item(),
            'rew_loss': rew_loss.item(),
            'con_loss': con_loss.item(),
            'dyn_kl': dyn_kl.mean().item(),
            'rep_kl': rep_kl.mean().item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item(),
            'returns': returns.mean().item(),
            'ret_scale': ret_scale,
        }

    # === Data collection and training loop ===
    import sys
    # Flush stdout for background task visibility
    _print = print
    def print(*a, **kw):
        kw.setdefault('flush', True)
        _print(*a, **kw)

    print(f"Starting DreamerV3 on {args.env_id} with {args.num_envs} envs")
    print(f"Model: deter={args.rssm_deter}, stoch={args.rssm_stoch}x{args.rssm_classes}, "
          f"hidden={args.rssm_hidden}, mlp={args.mlp_units}x{args.mlp_layers}")
    total_params = sum(p.numel() for p in wm_params) + sum(
        p.numel() for p in actor.parameters()) + sum(
        p.numel() for p in critic.parameters())
    print(f"Total parameters: {total_params:,}")

    obs, _ = envs.reset(seed=args.seed)
    for i in range(args.num_envs):
        init_ongoing(i, obs[i])

    # RSSM state for policy
    rssm_state = rssm.initial(args.num_envs, device)
    prev_action = torch.zeros(args.num_envs, act_dim, device=device)
    is_first_flag = np.ones(args.num_envs, dtype=bool)

    global_step = 0
    train_steps = 0
    start_time = time.time()
    steps_per_train = args.batch_size * args.batch_length

    while global_step < args.total_timesteps:
        # --- Collect one step ---
        prefilling = global_step < args.prefill_steps

        with torch.no_grad():
            if prefilling:
                actions_np = np.array([envs.single_action_space.sample()
                                       for _ in range(args.num_envs)])
                action_tensor = torch.tensor(actions_np, device=device, dtype=torch.float32)
            else:
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
                    is_first_tensor = torch.tensor(is_first_flag, device=device, dtype=torch.bool)
                    tokens = encoder(obs_tensor)
                    rssm_state, _, _ = rssm.observe_step(
                        rssm_state, tokens, prev_action, is_first_tensor)
                    feat = rssm.get_feat(rssm_state)
                    dist = actor(feat)
                    action_tensor = dist.sample().float().clamp(-1, 1)
                actions_np = action_tensor.cpu().numpy()

        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions_np)
        dones = terminateds | truncateds
        global_step += args.num_envs

        # Store transitions and handle episode boundaries
        for i in range(args.num_envs):
            if dones[i]:
                # Get true final observation
                if "final_observation" in infos and infos["final_observation"][i] is not None:
                    final_obs = infos["final_observation"][i]
                else:
                    final_obs = next_obs[i]
                finish_ongoing(i, final_obs, actions_np[i], rewards[i], terminateds[i])
                init_ongoing(i, next_obs[i])

                # Log episodic return
                if "final_info" in infos and infos["final_info"][i] is not None:
                    ep_info = infos["final_info"][i].get("episode", None)
                    if ep_info is not None:
                        ep_return = float(ep_info["r"])
                        ep_length = int(ep_info["l"])
                        print(f"global_step={global_step}, episodic_return={ep_return:.1f}, "
                              f"episodic_length={ep_length}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
            else:
                add_step(i, next_obs[i], actions_np[i], rewards[i], False, False)

        # Update state for next step
        is_first_flag = dones.copy()
        prev_action = action_tensor.clone()
        obs = next_obs

        # --- Train ---
        if not prefilling and replay.can_sample(args.batch_length):
            # Only count steps since prefill ended to avoid training burst
            effective_step = global_step - args.prefill_steps
            target_trains = int(effective_step * args.train_ratio / steps_per_train)
            num_trains = max(0, target_trains - train_steps)
            # Cap per-step training to keep collection flowing
            max_per_step = max(1, int(args.num_envs * args.train_ratio / steps_per_train) + 1)
            num_trains = min(num_trains, max_per_step * 2)

            for _ in range(num_trains):
                metrics = train_step()
                train_steps += 1

                if train_steps % 100 == 0:
                    for k, v in metrics.items():
                        writer.add_scalar(f"train/{k}", v, global_step)
                    sps = global_step / (time.time() - start_time)
                    writer.add_scalar("charts/SPS", sps, global_step)
                if train_steps % 500 == 0:
                    print(f"  train#{train_steps}: wm={metrics['wm_loss']:.2f} "
                          f"rec={metrics['rec_loss']:.2f} rew={metrics['rew_loss']:.2f} "
                          f"kl={metrics['dyn_kl']:.2f} ent={metrics['entropy']:.2f} "
                          f"ret={metrics['returns']:.1f}")

        # Periodic status
        if global_step % 10000 < args.num_envs:
            sps = global_step / max(1, time.time() - start_time)
            print(f"step={global_step}, train_steps={train_steps}, "
                  f"replay={replay.num_steps}, SPS={sps:.0f}")

    envs.close()
    writer.close()
    print("Done.")

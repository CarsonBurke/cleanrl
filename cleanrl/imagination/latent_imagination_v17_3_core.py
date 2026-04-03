# v17.3 — Bug fixes for dreamerv3/dreamer4 alignment
#
# Fixes from v17 (v16):
# 1. Discount weighting in PMPO actor loss (was binary mask, now proper weights)
# 2. Discount weight computation includes starting state continuation
# 3. Critic loss properly discount-weighted
# 4. KL direction fixed to KL(old||new) matching dreamer4
# 5. Value clipping replaced with dreamerv3-style slow-critic regularization
#
# Base architecture unchanged:
# - RSSM (block-diagonal GRU + 16x16 categorical stochastic state)
# - Decoder with symlog-MSE reconstruction loss
# - Split KL with free nats (dyn_scale=1.0, rep_scale=0.1)
# - Slow critic (EMA tau=0.02) + percentile return normalization
# - State-dependent std (sigmoid-squashed [0.1, 1.0])
# - PMPO actor loss with entropy + KL constraint
# - Separate actor, critic, world-model optimizers
# - Continue head (learned discount)
# - Longer imagination horizon (15)
# - No obs/reward normalization wrappers; symlog handles scale
# - RMSNorm + SiLU throughout
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
import torch.optim as optim
import tyro
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def build_symexp_bins(num_bins: int, bin_range: float = 20.0) -> torch.Tensor:
    if num_bins % 2 == 1:
        half = torch.linspace(-bin_range, 0.0, (num_bins - 1) // 2 + 1)
        half = symexp(half)
        return torch.cat([half, -half[:-1].flip(0)])
    half = torch.linspace(-bin_range, 0.0, num_bins // 2)
    half = symexp(half)
    return torch.cat([half, -half.flip(0)])


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    below = (bins <= x.unsqueeze(-1)).sum(-1) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1
    below_val = bins[below]
    above_val = bins[above]
    weight = (x - below_val) / (above_val - below_val + 1e-8)
    weight = weight.clamp(0, 1)
    result = torch.zeros(*x.shape, len(bins), device=x.device)
    result.scatter_(-1, below.unsqueeze(-1), (1 - weight).unsqueeze(-1))
    result.scatter_(-1, above.unsqueeze(-1), weight.unsqueeze(-1))
    return result


def twohot_predict(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins).sum(-1)


def twohot_loss(logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    two_hot = twohot_encode(target, bins)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(two_hot * log_probs).sum(-1)


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
    def __init__(self, in_dim, out_dim, hidden, layers, act="silu", norm=True, out_scale=1.0):
        super().__init__()
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[act]
        dims = [in_dim] + [hidden] * layers + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm:
                    mods.append(RMSNorm(dims[i + 1]))
                mods.append(act_fn())
        self.net = nn.Sequential(*mods)
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
    def __init__(self, act_dim, deter=256, hidden=128, stoch=16, classes=16,
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

        self.dyn_in_deter = nn.Linear(deter, hidden)
        self.dyn_in_stoch = nn.Linear(self.stoch_dim, hidden)
        self.dyn_in_act = nn.Linear(act_dim, hidden)
        self.dyn_norm0 = RMSNorm(hidden)
        self.dyn_norm1 = RMSNorm(hidden)
        self.dyn_norm2 = RMSNorm(hidden)
        self.dyn_act0 = act_fn_cls()
        self.dyn_act1 = act_fn_cls()
        self.dyn_act2 = act_fn_cls()

        self.core_in_per_block = deter // blocks + 3 * hidden
        core_in_total = blocks * self.core_in_per_block
        self.dyn_hidden = BlockLinear(core_in_total, deter, blocks)
        self.dyn_hidden_norm = RMSNorm(deter)
        self.dyn_hidden_act = act_fn_cls()
        self.dyn_gru = BlockLinear(deter, 3 * deter, blocks)

        self.prior = nn.Sequential(
            nn.Linear(deter, hidden), RMSNorm(hidden), act_fn_cls(),
            nn.Linear(hidden, hidden), RMSNorm(hidden), act_fn_cls(),
            nn.Linear(hidden, stoch * classes),
        )
        # Posterior: input dim set via set_post_input_dim
        self.post = nn.Sequential(
            nn.Linear(deter + hidden, hidden),
            RMSNorm(hidden), act_fn_cls(),
            nn.Linear(hidden, stoch * classes),
        )

    def set_post_input_dim(self, token_dim):
        hidden = self.hidden_dim
        act_fn_cls = type(self.dyn_act0)
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
        deter = state['deter']
        stoch = state['stoch'].reshape(*state['stoch'].shape[:-2], -1)
        return torch.cat([deter, stoch], -1)

    def _sample_categorical(self, logits):
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / self.classes
        probs = (1 - self.unimix) * probs + self.unimix * uniform
        flat = probs.reshape(-1, self.classes)
        indices = torch.multinomial(flat, 1).squeeze(-1)
        indices = indices.reshape(logits.shape[:-1])
        one_hot = F.one_hot(indices, self.classes).float()
        return one_hot + probs - probs.detach()

    def _gru_core(self, deter, stoch, action):
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        action = action / action.abs().clamp(min=1).detach()

        x0 = self.dyn_act0(self.dyn_norm0(self.dyn_in_deter(deter)))
        x1 = self.dyn_act1(self.dyn_norm1(self.dyn_in_stoch(stoch_flat)))
        x2 = self.dyn_act2(self.dyn_norm2(self.dyn_in_act(action)))

        combined = torch.cat([x0, x1, x2], -1)
        combined = combined.unsqueeze(1).expand(-1, self.blocks, -1)

        B = deter.shape[0]
        deter_blocked = deter.reshape(B, self.blocks, self.deter_dim // self.blocks)
        x = torch.cat([deter_blocked, combined], -1)
        x = x.reshape(B, -1)

        x = self.dyn_hidden_act(self.dyn_hidden_norm(self.dyn_hidden(x)))

        gates = self.dyn_gru(x).reshape(B, 3, self.deter_dim)
        reset_raw, cand_raw, update_raw = gates[:, 0], gates[:, 1], gates[:, 2]

        reset = torch.sigmoid(reset_raw)
        cand = torch.tanh(reset * cand_raw)
        update = torch.sigmoid(update_raw - 1)
        new_deter = update * cand + (1 - update) * deter
        return new_deter

    def observe_step(self, state, token, action, is_first):
        B = is_first.shape[0]
        mask = (~is_first).float()

        deter = state['deter'] * mask.unsqueeze(-1)
        stoch = state['stoch'] * mask.unsqueeze(-1).unsqueeze(-1)
        action = action * mask.unsqueeze(-1)

        new_deter = self._gru_core(deter, stoch, action)

        post_input = torch.cat([new_deter, token], -1)
        post_logit = self.post(post_input).reshape(B, self.stoch, self.classes)
        post_stoch = self._sample_categorical(post_logit)

        prior_logit = self.prior(new_deter).reshape(B, self.stoch, self.classes)

        new_state = {'deter': new_deter, 'stoch': post_stoch}
        return new_state, prior_logit, post_logit

    def observe(self, tokens, actions, is_first):
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
        new_deter = self._gru_core(state['deter'], state['stoch'], action)
        prior_logit = self.prior(new_deter).reshape(
            new_deter.shape[0], self.stoch, self.classes)
        prior_stoch = self._sample_categorical(prior_logit)
        return {'deter': new_deter, 'stoch': prior_stoch}


# ---------------------------------------------------------------------------
# Encoder / Decoder / Heads
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, obs_dim, units, layers, act="silu"):
        super().__init__()
        self.mlp = MLP(obs_dim, units, units, layers, act=act, out_scale=1.0)

    def forward(self, obs):
        return self.mlp(symlog(obs))


class Decoder(nn.Module):
    def __init__(self, feat_dim, obs_dim, units, layers, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, obs_dim, units, layers, act=act, out_scale=1.0)

    def forward(self, feat):
        return self.mlp(feat)

    def loss(self, feat, obs_target):
        pred = self.forward(feat)
        return 0.5 * (pred - symlog(obs_target)).pow(2).sum(-1)


class RewardHead(nn.Module):
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
        return twohot_predict(logits, self.bins)


class ContinueHead(nn.Module):
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
    def __init__(self, feat_dim, units, layers, num_bins=255, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, num_bins, units, layers, act=act, out_scale=0.0)
        self.register_buffer('bins', build_symexp_bins(num_bins))

    def forward(self, feat):
        return self.mlp(feat)

    def predict(self, feat):
        return twohot_predict(self.forward(feat), self.bins)

    def loss(self, logits, target):
        return twohot_loss(logits, target, self.bins)


# ---------------------------------------------------------------------------
# Percentile Return Normalization
# ---------------------------------------------------------------------------

class PercentileReturnNorm:
    """Percentile-based return normalization (dreamerv3 style).
    Tracks 5th/95th percentile of returns, advantages = (ret - val) / scale."""
    def __init__(self, rate=0.01, perclo=5.0, perchi=95.0, limit=1.0):
        self.rate = rate
        self.perclo = perclo
        self.perchi = perchi
        self.limit = limit
        self.lo = 0.0
        self.hi = 0.0
        self.initialized = False

    def update(self, returns):
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
    H = rew.shape[1] - 1
    rets = [val[:, -1]]
    for t in reversed(range(H)):
        ret_t = rew[:, t + 1] + con[:, t + 1] * ((1 - lam) * val[:, t + 1] + lam * rets[-1])
        rets.append(ret_t)
    rets = list(reversed(rets[:-1]))
    return torch.stack(rets, 1)


# ---------------------------------------------------------------------------
# Replay Buffer (episode-based, like dreamer4)
# ---------------------------------------------------------------------------

class EpisodeReplayBuffer:
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
        valid = [ep for ep in self.episodes if len(ep['reward']) >= batch_length]
        if not valid:
            valid = self.episodes

        keys = tuple(valid[0].keys())
        sample_ep = valid[0]
        batch = {}
        for k in keys:
            shape = sample_ep[k].shape[1:]
            batch[k] = np.zeros((batch_size, batch_length) + shape, dtype=sample_ep[k].dtype)

        for batch_index in range(batch_size):
            ep = valid[np.random.randint(len(valid))]
            ep_len = len(ep['reward'])
            if ep_len >= batch_length:
                start = np.random.randint(0, ep_len - batch_length + 1)
            else:
                start = 0
            end = start + batch_length
            for k in keys:
                chunk = ep[k][start:end]
                batch[k][batch_index, : len(chunk)] = chunk

        return batch

    def sample_tensors(self, batch_size, batch_length, device: torch.device):
        batch_np = self.sample(batch_size, batch_length)
        batch = {}
        for key, value in batch_np.items():
            tensor = torch.from_numpy(value)
            if tensor.dtype == torch.bool:
                batch[key] = tensor.to(device=device)
            else:
                batch[key] = tensor.to(device=device, dtype=torch.float32)
        return batch


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

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

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    num_envs: int = 8

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
    num_bins: int = 255

    # Training
    learning_rate: float = 1e-4  # dreamerv3: 1e-4 for all
    wm_learning_rate: float = 1e-4
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    eps: float = 1e-8
    batch_size_per_env: int = 16
    batch_length: int = 32
    train_ratio: float = 64.0
    imag_last: int = 8
    max_grad_norm: float = 1000.0  # dreamerv3: 1000
    compile: bool = True

    # Imagination
    imag_horizon: int = 15

    # Actor-Critic
    horizon: int = 333
    lam: float = 0.95
    actor_entropy: float = 3e-4  # dreamerv3: 3e-4
    pmpo_pos_neg_weight: float = 0.5  # α in PMPO: weight for positive advantages
    pmpo_kl_weight: float = 0.3  # reverse KL constraint weight
    actor_minstd: float = 0.1
    actor_maxstd: float = 1.0
    slow_target_rate: float = 0.02
    slow_reg: float = 1.0
    return_norm_rate: float = 0.01  # dreamerv3: rate for percentile EMA
    return_norm_perclo: float = 5.0
    return_norm_perchi: float = 95.0
    return_norm_limit: float = 1.0  # dreamerv3: minimum scale

    # Replay
    replay_size: int = 1_000_000
    prefill_steps: int = 5000

    num_iterations: int = 0


# ---------------------------------------------------------------------------
# make_env (no normalization wrappers - symlog handles scale)
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

def main(args_class=Args):
    args = tyro.cli(args_class)
    args.batch_size = args.batch_size_per_env * args.num_envs
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
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    # Build models
    encoder = Encoder(obs_dim, args.mlp_units, args.mlp_layers).to(device)
    token_dim = args.mlp_units

    rssm = RSSM(
        act_dim, deter=args.rssm_deter, hidden=args.rssm_hidden,
        stoch=args.rssm_stoch, classes=args.rssm_classes,
        blocks=args.rssm_blocks, unimix=args.unimix,
    ).to(device)
    rssm.set_post_input_dim(token_dim)
    rssm = rssm.to(device)

    feat_dim = rssm.feat_dim

    decoder = Decoder(feat_dim, obs_dim, args.mlp_units, args.mlp_layers).to(device)
    reward_head = RewardHead(feat_dim, args.mlp_units, args.num_bins).to(device)
    continue_head = ContinueHead(feat_dim, args.mlp_units).to(device)
    actor = Actor(feat_dim, act_dim, args.mlp_units, args.mlp_layers,
                  minstd=args.actor_minstd, maxstd=args.actor_maxstd).to(device)
    critic = Critic(feat_dim, args.mlp_units, args.mlp_layers, args.num_bins).to(device)
    slow_critic = copy.deepcopy(critic).to(device)
    slow_critic.requires_grad_(False)

    # Optimizers (separate for WM, actor, critic)
    wm_params = (list(encoder.parameters()) + list(rssm.parameters()) +
                 list(decoder.parameters()) + list(reward_head.parameters()) +
                 list(continue_head.parameters()))
    wm_opt = optim.Adam(wm_params, lr=args.wm_learning_rate, eps=args.eps)
    actor_opt = optim.Adam(actor.parameters(), lr=args.actor_learning_rate, eps=args.eps)
    critic_opt = optim.Adam(critic.parameters(), lr=args.critic_learning_rate, eps=args.eps)

    # torch.compile
    # Compile actor MLP separately (Normal distribution breaks torch.compile)
    actor_mlp_compiled = actor.mlp
    if args.compile:
        rssm.observe_step = torch.compile(rssm.observe_step)
        rssm.imagine_step = torch.compile(rssm.imagine_step)
        encoder = torch.compile(encoder)
        decoder = torch.compile(decoder)
        reward_head = torch.compile(reward_head)
        continue_head = torch.compile(continue_head)
        critic = torch.compile(critic)
        slow_critic = torch.compile(slow_critic)
        actor_mlp_compiled = torch.compile(actor.mlp)

    bins = build_symexp_bins(args.num_bins).to(device)
    return_norm = PercentileReturnNorm(
        rate=args.return_norm_rate,
        perclo=args.return_norm_perclo,
        perchi=args.return_norm_perchi,
        limit=args.return_norm_limit,
    )

    replay = EpisodeReplayBuffer(args.replay_size)
    cont_target_val = 1.0 - 1.0 / args.horizon

    # Episode tracking per env
    ongoing = [None] * args.num_envs

    def init_ongoing(env_id, obs_np):
        ongoing[env_id] = {
            'obs': [obs_np.copy()],
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
        episode = {k: np.array(v, dtype=np.float32 if k not in ('is_first', 'is_terminal') else bool)
                   for k, v in ep.items()}
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
        batch = replay.sample_tensors(args.batch_size, args.batch_length, device)
        obs = batch['obs']
        prev_act = batch['prev_act']
        rew = batch['reward']
        is_first = batch['is_first']
        is_term = batch['is_terminal']
        B, T = obs.shape[:2]

        is_first = is_first.clone()
        is_first[:, 0] = True

        # ===== World Model =====
        wm_opt.zero_grad()

        tokens = encoder(obs.reshape(B * T, -1)).reshape(B, T, -1)
        states, prior_logits, post_logits = rssm.observe(tokens, prev_act, is_first)
        feat = rssm.get_feat(states)
        feat_flat = feat.reshape(B * T, -1)

        rec_loss = decoder.loss(feat_flat, obs.reshape(B * T, -1)).reshape(B, T).mean()
        rew_loss = reward_head.loss(feat_flat, rew.reshape(B * T)).reshape(B, T).mean()

        con_target = (~is_term).float() * cont_target_val
        con_logits = continue_head(feat_flat)
        con_loss = F.binary_cross_entropy_with_logits(con_logits, con_target.reshape(B * T)).mean()

        # KL losses with free nats
        prior_logits_f = prior_logits.float()
        post_logits_f = post_logits.float()
        prior_probs = F.softmax(prior_logits_f, -1)
        post_probs = F.softmax(post_logits_f, -1)
        prior_probs_um = (1 - args.unimix) * prior_probs + args.unimix / args.rssm_classes
        post_probs_um = (1 - args.unimix) * post_probs + args.unimix / args.rssm_classes

        dyn_kl = (post_probs_um.detach() * (
            post_probs_um.detach().log() - prior_probs_um.log()
        )).sum(-1).sum(-1)
        dyn_loss = torch.maximum(dyn_kl, free_nats_t).mean()

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
            K = min(args.imag_last, T) if args.imag_last > 0 else T
            BK = B * K
            start_feat = {k: v[:, -K:].reshape(BK, *v.shape[2:]).detach()
                          for k, v in states.items()}

            cur_feat = rssm.get_feat(start_feat)
            img_feats = [cur_feat]
            img_acts = []
            img_old_means = []
            img_old_stds = []
            state = start_feat
            for h in range(args.imag_horizon):
                # Manual actor forward (compiled MLP + manual Normal sample)
                actor_out = actor_mlp_compiled(cur_feat)
                act_mean, act_raw_std = actor_out.split(act_dim, -1)
                act_std = actor.minstd + (actor.maxstd - actor.minstd) * torch.sigmoid(act_raw_std)
                action = act_mean + act_std * torch.randn_like(act_std)
                # Store old policy params for KL constraint
                img_old_means.append(act_mean)
                img_old_stds.append(act_std)
                img_acts.append(action)
                state = rssm.imagine_step(state, action)
                cur_feat = rssm.get_feat(state)
                img_feats.append(cur_feat)

            img_feats = torch.stack(img_feats, 1)
            img_acts = torch.stack(img_acts, 1)
            img_old_means = torch.stack(img_old_means, 1)  # (BK, H, act_dim)
            img_old_stds = torch.stack(img_old_stds, 1)

            H1 = args.imag_horizon + 1
            img_feats_flat = img_feats.reshape(BK * H1, -1)
            img_rew = reward_head.predict(img_feats_flat).reshape(BK, H1).float()
            img_con = continue_head.predict(img_feats_flat).reshape(BK, H1).float()
            img_slow_val = twohot_predict(slow_critic(img_feats_flat), bins).reshape(BK, H1).float()

            returns = compute_lambda_returns(img_rew, img_con, img_slow_val, args.lam)

            # Return normalization (dreamerv3: percentile scale)
            return_norm.update(returns)
            ret_scale = return_norm.scale()
            advantages = (returns - img_slow_val[:, :-1]) / ret_scale

            # Importance weights (cumulative continue — dreamerv3 style)
            weight = torch.cumprod(img_con[:, :-1], dim=1)

            # Pre-compute actor/critic input feats and slow critic targets
            imag_actor_feats = img_feats[:, :-1].reshape(BK * args.imag_horizon, -1)
            imag_acts_flat = img_acts.reshape(BK * args.imag_horizon, -1)
            slow_val_pred = twohot_predict(
                slow_critic(imag_actor_feats), bins
            ).detach()

        # ===== Actor (PMPO — dreamer4 style) =====
        actor_opt.zero_grad()

        policy_dist = actor(imag_actor_feats.detach())
        logpi = policy_dist.log_prob(imag_acts_flat.detach()).sum(-1).reshape(BK, args.imag_horizon)
        entropy = policy_dist.entropy().sum(-1).reshape(BK, args.imag_horizon)

        # PMPO: boolean mask + masked_mean (dreamer4 convention)
        adv_detached = advantages.detach()
        w = weight.detach()
        pos_mask = (adv_detached >= 0)
        neg_mask = ~pos_mask

        alpha = args.pmpo_pos_neg_weight
        pos_count = pos_mask.sum().clamp(min=1)
        neg_count = neg_mask.sum().clamp(min=1)
        pos_term = (logpi * pos_mask).sum() / pos_count
        neg_term = -(logpi * neg_mask).sum() / neg_count
        pmpo_loss = -(alpha * pos_term + (1 - alpha) * neg_term)

        # Entropy bonus (discount-weighted — dreamerv3 style)
        entropy_loss = -(w * entropy.float()).mean()

        # KL constraint: KL(old || new) — mode-covering (dreamer4)
        if args.pmpo_kl_weight > 0:
            old_mu = img_old_means.detach().reshape(BK * args.imag_horizon, -1)
            old_sig = img_old_stds.detach().reshape(BK * args.imag_horizon, -1)
            new_mu = policy_dist.loc
            new_sig = policy_dist.scale
            kl_div = (torch.log(new_sig / old_sig) +
                      (old_sig**2 + (old_mu - new_mu)**2) / (2 * new_sig**2) - 0.5).sum(-1)
            kl_loss = kl_div.reshape(BK, args.imag_horizon).mean()
        else:
            kl_loss = torch.zeros(1, device=device)

        actor_loss = pmpo_loss + args.actor_entropy * entropy_loss + args.pmpo_kl_weight * kl_loss

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        actor_opt.step()

        # ===== Critic (dreamerv3 style: discount-weighted + slow-critic reg) =====
        critic_opt.zero_grad()

        val_logits = critic(imag_actor_feats.detach())
        returns_flat = returns.reshape(-1).detach()
        w_critic = weight.detach()

        # Return prediction loss, discount-weighted
        loss_returns = twohot_loss(
            val_logits, returns_flat, bins
        ).reshape(BK, args.imag_horizon)

        # Slow-critic regularization (dreamerv3)
        loss_slow = twohot_loss(
            val_logits, slow_val_pred.reshape(-1), bins
        ).reshape(BK, args.imag_horizon)

        critic_loss = (w_critic * (loss_returns + args.slow_reg * loss_slow)).mean()

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
            'ret_scale': return_norm.scale(),
        }

    # === Flush stdout for background task visibility ===
    import builtins
    _print = builtins.print
    def print(*a, **kw):
        kw.setdefault('flush', True)
        _print(*a, **kw)

    print(f"Starting latent_imagination_v17.3 on {args.env_id} with {args.num_envs} envs")
    total_params = sum(p.numel() for p in wm_params) + sum(
        p.numel() for p in actor.parameters()) + sum(
        p.numel() for p in critic.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"RSSM: deter={args.rssm_deter}, stoch={args.rssm_stoch}x{args.rssm_classes}, "
          f"feat_dim={feat_dim}")
    print(f"Batch: {args.batch_size} ({args.batch_size_per_env}/env × {args.num_envs} envs), "
          f"length={args.batch_length}, train_ratio={args.train_ratio}")

    obs, _ = envs.reset(seed=args.seed)
    for i in range(args.num_envs):
        init_ongoing(i, obs[i])

    rssm_state = rssm.initial(args.num_envs, device)
    prev_action = torch.zeros(args.num_envs, act_dim, device=device)
    is_first_flag = np.ones(args.num_envs, dtype=bool)

    global_step = 0
    train_steps = 0
    imagined_steps = 0
    imag_steps_per_train = args.batch_size * (min(args.imag_last, args.batch_length) if args.imag_last > 0 else args.batch_length) * args.imag_horizon
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

        for i in range(args.num_envs):
            if dones[i]:
                if "final_observation" in infos and infos["final_observation"][i] is not None:
                    final_obs = infos["final_observation"][i]
                else:
                    final_obs = next_obs[i]
                finish_ongoing(i, final_obs, actions_np[i], rewards[i], terminateds[i])
                init_ongoing(i, next_obs[i])

                if "final_info" in infos and infos["final_info"][i] is not None:
                    ep_info = infos["final_info"][i].get("episode", None)
                    if ep_info is not None:
                        ep_return = float(ep_info["r"])
                        ep_length = int(ep_info["l"])
                        print(f"global_step={global_step}, episodic_return={ep_return:.1f}, "
                              f"episodic_length={ep_length}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        total_steps = global_step + imagined_steps
                        writer.add_scalar("charts_total/episodic_return", ep_return, total_steps)
                        writer.add_scalar("charts_total/total_steps", total_steps, global_step)
            else:
                add_step(i, next_obs[i], actions_np[i], rewards[i], False, False)

        prev_action = action_tensor
        is_first_flag = dones.copy()
        obs = next_obs

        # --- Training ---
        if not prefilling and replay.can_sample(args.batch_length):
            effective_step = global_step - args.prefill_steps
            target_train_steps = int(effective_step * args.train_ratio / steps_per_train)
            # Cap training burst (match dreamer4 ref)
            max_steps_per_cycle = max(1, int(args.num_envs * args.train_ratio / steps_per_train)) * 2
            steps_this_cycle = 0
            while train_steps < target_train_steps and steps_this_cycle < max_steps_per_cycle:
                stats = train_step()
                train_steps += 1
                imagined_steps += imag_steps_per_train
                steps_this_cycle += 1

                if train_steps % 100 == 0:
                    writer.add_scalar("losses/wm_loss", stats['wm_loss'], global_step)
                    writer.add_scalar("losses/rec_loss", stats['rec_loss'], global_step)
                    writer.add_scalar("losses/rew_loss", stats['rew_loss'], global_step)
                    writer.add_scalar("losses/con_loss", stats['con_loss'], global_step)
                    writer.add_scalar("losses/dyn_kl", stats['dyn_kl'], global_step)
                    writer.add_scalar("losses/rep_kl", stats['rep_kl'], global_step)
                    writer.add_scalar("losses/actor_loss", stats['actor_loss'], global_step)
                    writer.add_scalar("losses/critic_loss", stats['critic_loss'], global_step)
                    writer.add_scalar("imagination/entropy", stats['entropy'], global_step)
                    writer.add_scalar("imagination/returns", stats['returns'], global_step)
                    writer.add_scalar("imagination/ret_scale", stats['ret_scale'], global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("charts/train_steps", train_steps, global_step)
                    writer.add_scalar("charts/imagined_steps", imagined_steps, global_step)
                    writer.add_scalar("charts_total/SPS", int((global_step + imagined_steps) / (time.time() - start_time)), global_step)

        if global_step % 10000 < args.num_envs:
            sps = int(global_step / (time.time() - start_time))
            print(f"SPS: {sps}, global_step={global_step}, replay={replay.num_steps}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        state_dict = {
            'encoder': encoder.state_dict() if not isinstance(encoder, torch._dynamo.eval_frame.OptimizedModule) else encoder._orig_mod.state_dict(),
            'rssm': rssm.state_dict(),
            'decoder': decoder.state_dict() if not isinstance(decoder, torch._dynamo.eval_frame.OptimizedModule) else decoder._orig_mod.state_dict(),
            'reward_head': reward_head.state_dict() if not isinstance(reward_head, torch._dynamo.eval_frame.OptimizedModule) else reward_head._orig_mod.state_dict(),
            'continue_head': continue_head.state_dict() if not isinstance(continue_head, torch._dynamo.eval_frame.OptimizedModule) else continue_head._orig_mod.state_dict(),
            'actor': actor.state_dict(),
            'critic': critic.state_dict() if not isinstance(critic, torch._dynamo.eval_frame.OptimizedModule) else critic._orig_mod.state_dict(),
        }
        torch.save(state_dict, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()

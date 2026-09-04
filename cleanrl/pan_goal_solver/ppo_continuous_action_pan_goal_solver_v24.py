# Pan Goal Solver v24: the direct construction. Hindsight goal-conditioned
# policy, one loss, nothing else.
#
# Online Pan-1 without pretraining data or a specific goal is three
# sentences: relabel your own trajectories with frames you actually reached
# later; train one policy pi(a | context, goal-frame) on the actions that
# got you there; command goal frames from your own experience. v23 wrapped
# that in hypothesis heads, occupancy gates, and imagined-frame machinery —
# and in doing so routed the goal signal through the one-step conditional
# measured goal-blind three times (v19/v20/v21), while deleting the only
# channel ever measured alive: first-action regression on the far hindsight
# goal (v16/v17, goal_removal 0.05-0.09 even with a fixed command).
#
# v24:
# - World model: pure LeJEPA (attached online target, token-space SIGReg,
#   no EMA), trained by nothing but its own losses. Unchanged.
# - Policy: pi(a | belief, f_t, G) — a single MLP with a single loss,
#   MSE against the action actually taken, where G is the frame summary of
#   an observation actually reached later (geometric offsets, mean ~50).
# - Collection: commanded goal frames switch within episodes (64 steps):
#   mostly random achieved frames, some top-reward frames, some
#   velocity-scaled top-reward frames (fixed scale set, no gate). Random
#   warmup, then policy + noise. No explorer, no anchor, no reward head,
#   no occupancy, no successor/inverse heads, no proposer.
# - Goals are frames (a specific future observation). Reward's only role is
#   sorting which achieved frames get commanded/scaled.
# No PPO, no critic, no policy gradient, no planning; one direct pass.
#
# Pre-registered 1.5M falsifiers:
# 1. evaluation/reach_servo_gap sustained > 0 (matched vs shuffled command).
# 2. diagnostics/goal_removal_action_mse alive (not pinned at ~0.001).
# 3. top-reward and scaled eval arms separating from random-goal behavior
#    while episodic returns climb.

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
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 16
    num_steps: int = 512
    history: int = 16
    replay_size: int = 1_000_000
    batch_size: int = 512
    updates_per_iteration: int = 64
    follower_updates_per_iteration: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    max_grad_norm: float = 1.0
    compile: bool = False
    compile_mode: str = "reduce-overhead"

    latent_dim: int = 64
    encoder_layers: int = 2
    temporal_layers: int = 2
    heads: int = 4
    ffn_mult: int = 2
    sigreg_coef: float = 0.09
    sigreg_projections: int = 256

    follower_hidden: int = 512
    goal_discount: float = 0.98
    max_goal_offset: int = 256
    follower_warmup_steps: int = 20_000
    random_warmup_steps: int = 100_000

    # Goal-directed collection: commanded goal frames switch within
    # episodes. Mixture: uniform achieved frames (diversity), top-reward
    # frames (focus), velocity-scaled top-reward frames (the remainder;
    # reach past demonstrated speed).
    goal_switch_steps: int = 64
    command_uniform_fraction: float = 0.6
    command_top_fraction: float = 0.2
    command_scales: tuple[float, ...] = (1.25, 1.5, 2.0)
    good_frame_candidates: int = 256

    policy_noise_rho: float = 0.9
    policy_noise_std: float = 0.15
    policy_innovation_std: float = 0.3
    eval_interval: int = 100_000
    eval_envs: int = 4


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


def xavier_linear(layer, gain=1.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def relu_sq(x):
    return F.relu(x).square()


class SIGReg(nn.Module):
    """Epps-Pulley random-projection regularizer toward an isotropic Gaussian."""

    def __init__(self, knots=17, num_proj=256):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        quadrature = torch.full((knots,), 2 * dt)
        quadrature[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", quadrature * phi)

    def forward(self, samples):
        if samples.ndim == 2:
            samples = samples.unsqueeze(0)
        dim = samples.shape[-1]
        projections = torch.randn(dim, self.num_proj, device=samples.device, dtype=samples.dtype)
        projections = projections / projections.norm(dim=0, keepdim=True).clamp_min(1e-8)
        projected = (samples @ projections).unsqueeze(-1) * self.t.to(samples.dtype)
        error = (
            (projected.cos().mean(-3) - self.phi.to(samples.dtype)).square()
            + projected.sin().mean(-3).square()
        )
        return ((error @ self.weights.to(samples.dtype)) * samples.shape[-2]).mean()


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_mult):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult, bias=False))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim, bias=False))
        for name, parameter in self.attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x, causal=False):
        mask = None
        if causal:
            length = x.shape[1]
            mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.w2(relu_sq(self.w1(self.ffn_norm(x))))


class FrameEncoder(nn.Module):
    """Raw per-feature affine tokens with spatial attention."""

    def __init__(self, obs_dim, dim, layers, heads, ffn_mult):
        super().__init__()
        self.obs_scale = nn.Parameter(torch.empty(obs_dim, dim))
        self.obs_bias = nn.Parameter(torch.empty(obs_dim, dim))
        self.feature_embed = nn.Parameter(torch.empty(obs_dim, dim))
        nn.init.normal_(self.obs_scale, std=0.02)
        nn.init.normal_(self.obs_bias, std=0.02)
        nn.init.normal_(self.feature_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, ffn_mult) for _ in range(layers)]
        )
        self.out_norm = nn.RMSNorm(dim)

    def forward(self, obs):
        shape = obs.shape[:-1]
        x = (
            obs.reshape(-1, obs.shape[-1], 1) * self.obs_scale
            + self.obs_bias
            + self.feature_embed
        )
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x).reshape(*shape, obs.shape[-1], -1)


class TemporalBelief(nn.Module):
    """Action-free token streams plus one shifted incoming-action stream."""

    def __init__(self, obs_tokens, act_dim, dim, layers, heads, ffn_mult):
        super().__init__()
        self.obs_tokens = obs_tokens
        self.action_embed = xavier_linear(nn.Linear(act_dim, dim, bias=False))
        self.action_slot = nn.Parameter(torch.empty(1, 1, 1, dim))
        nn.init.normal_(self.action_slot, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, ffn_mult) for _ in range(layers)]
        )

    def forward(self, observation_tokens, incoming_actions):
        batch, history, obs_tokens, dim = observation_tokens.shape
        action_tokens = self.action_slot.expand(batch, history, -1, -1)
        action_tokens = action_tokens + self.action_embed(incoming_actions).unsqueeze(-2)
        x = torch.cat([observation_tokens, action_tokens], dim=-2)
        x = x.permute(0, 2, 1, 3).reshape(batch * (obs_tokens + 1), history, dim)
        for block in self.blocks:
            x = block(x, causal=True)
        latest = x[:, -1].reshape(batch, obs_tokens + 1, dim)
        return latest[:, :obs_tokens].mean(dim=1), latest


class TransitionPredictor(nn.Module):
    def __init__(self, obs_tokens, act_dim, dim, heads, ffn_mult):
        super().__init__()
        self.obs_tokens = obs_tokens
        self.queries = nn.Parameter(torch.empty(obs_tokens, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.action_embed = xavier_linear(nn.Linear(act_dim, dim, bias=False))
        self.query_norm = nn.RMSNorm(dim)
        self.memory_norm = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.ffn_norm = nn.RMSNorm(dim)
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult, bias=False))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim, bias=False))

    def forward(self, latest_memory, action):
        queries = self.queries.unsqueeze(0).expand(latest_memory.shape[0], -1, -1)
        queries = queries + self.action_embed(action).unsqueeze(1)
        memory = self.memory_norm(latest_memory)
        queries = queries + self.attn(self.query_norm(queries), memory, memory, need_weights=False)[0]
        return queries + self.w2(relu_sq(self.w1(self.ffn_norm(queries))))


class WorldModel(nn.Module):
    """Pure LeJEPA state predictor. Nothing else trains it."""

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.frame_encoder = FrameEncoder(
            obs_dim, args.latent_dim, args.encoder_layers, args.heads, args.ffn_mult
        )
        self.temporal = TemporalBelief(
            obs_dim,
            act_dim,
            args.latent_dim,
            args.temporal_layers,
            args.heads,
            args.ffn_mult,
        )
        self.transition = TransitionPredictor(
            obs_dim, act_dim, args.latent_dim, args.heads, args.ffn_mult
        )
        self.sigreg = SIGReg(num_proj=args.sigreg_projections)

    def encode_history(self, obs_history, incoming_action_history):
        tokens = self.frame_encoder(obs_history)
        frame_summaries = tokens.mean(dim=-2)
        belief, latest_memory = self.temporal(tokens, incoming_action_history)
        return tokens, frame_summaries, belief, latest_memory

    def forward(self, obs_history, incoming_action_history):
        return self.encode_history(obs_history, incoming_action_history)


class GoalPolicy(nn.Module):
    """The direct object: pi(a | belief, f_t, G). Hindsight first-action
    regression on the far goal is the one conditioning channel this family
    ever measured alive; this head is that channel and nothing else."""

    def __init__(self, observation_tokens, belief_dim, act_dim, hidden):
        super().__init__()
        self.belief_readout = nn.Sequential(
            nn.RMSNorm(observation_tokens * belief_dim),
            xavier_linear(nn.Linear(observation_tokens * belief_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, belief_dim)),
        )
        self.action_net = nn.Sequential(
            nn.RMSNorm(2 * belief_dim),
            xavier_linear(nn.Linear(2 * belief_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, act_dim), gain=0.05),
        )

    @staticmethod
    def bounded_delta(source, target):
        delta = target - source
        return delta / (1.0 + delta.norm(dim=-1, keepdim=True))

    def read_belief(self, observation_memory):
        return self.belief_readout(observation_memory.flatten(1))

    def action(self, belief, current_frame, goal_frame):
        features = self.bounded_delta(current_frame, goal_frame)
        return torch.tanh(self.action_net(torch.cat([belief, features], dim=-1)))

    def forward(self, belief, current_frame, goal_frame):
        return self.action(belief, current_frame, goal_frame)


class PanGoalSolver(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.world_model = WorldModel(obs_dim, act_dim, args)
        self.follower = GoalPolicy(
            obs_dim, args.latent_dim, act_dim, args.follower_hidden
        )

    def encode_frames(self, obs):
        """Frame summaries of single observations: the goal space."""
        return self.world_model.frame_encoder(obs).mean(dim=-2)

    def follower_state(self, obs_history, incoming_action_history):
        _, frame_summaries, _, memory = self.world_model(
            obs_history, incoming_action_history
        )
        belief = self.follower.read_belief(memory[:, :-1])
        return belief, frame_summaries[:, -1]

    def act(self, obs_history, incoming_action_history, goal_frames):
        belief, current_frame = self.follower_state(obs_history, incoming_action_history)
        if goal_frames.ndim == 1:
            goal_frames = goal_frames.unsqueeze(0).expand(obs_history.shape[0], -1)
        action = self.follower(belief, current_frame, goal_frames)
        return action, current_frame

    def parameter_groups(self):
        world = list(self.world_model.parameters())
        follower = list(self.follower.parameters())
        if {id(p) for p in world} & {id(p) for p in follower}:
            raise RuntimeError("world and follower parameters must be disjoint")
        return world, follower


class VectorReplayBuffer:
    """Raw per-environment ring buffer; latents are always freshly encoded."""

    def __init__(self, total_capacity, num_envs, obs_dim, act_dim):
        self.num_envs = num_envs
        self.capacity = max(2, total_capacity // num_envs)
        self.obs = np.zeros((num_envs, self.capacity, obs_dim), np.float32)
        self.incoming_action = np.zeros((num_envs, self.capacity, act_dim), np.float32)
        self.action = np.zeros((num_envs, self.capacity, act_dim), np.float32)
        self.reward = np.zeros((num_envs, self.capacity), np.float32)
        self.done = np.zeros((num_envs, self.capacity), np.bool_)
        self.episode = np.zeros((num_envs, self.capacity), np.int64)
        self.absolute = np.full((num_envs, self.capacity), -1, np.int64)
        self.total_steps = 0
        self.episode_ids = np.arange(num_envs, dtype=np.int64)

    @property
    def size(self):
        return min(self.total_steps, self.capacity) * self.num_envs

    def add(self, obs, incoming_action, action, reward, done):
        slot = self.total_steps % self.capacity
        self.obs[:, slot] = obs
        self.incoming_action[:, slot] = incoming_action
        self.action[:, slot] = action
        self.reward[:, slot] = reward
        self.done[:, slot] = done
        self.episode[:, slot] = self.episode_ids
        self.absolute[:, slot] = self.total_steps
        self.episode_ids = self.episode_ids + done.astype(np.int64) * self.num_envs
        self.total_steps += 1

    def _gather(self, array, env, absolute_indices):
        return array[env[:, None], absolute_indices % self.capacity]

    def _valid_end(self, env, end):
        return self._gather(self.absolute, env, end[:, None]).reshape(-1) == end

    def _padded_history(self, array, env, end, history, zero_pad=False):
        indices = end[:, None] - np.arange(history - 1, -1, -1)[None]
        values = self._gather(array, env, indices).copy()
        absolute = self._gather(self.absolute, env, indices)
        episodes = self._gather(self.episode, env, indices)
        valid = (absolute == indices) & (episodes == episodes[:, -1:])
        for row in range(values.shape[0]):
            first = int(np.argmax(valid[row]))
            if zero_pad:
                values[row, :first] = 0
            else:
                values[row, :first] = values[row, first]
        return values

    def _sample_base(self, batch_size, history, future_room, rng):
        earliest = max(history - 1, self.total_steps - self.capacity + history - 1)
        latest = self.total_steps - 1 - future_room
        if latest < earliest:
            raise RuntimeError("replay does not yet contain a valid history")
        env = rng.integers(0, self.num_envs, size=batch_size)
        end = rng.integers(earliest, latest + 1, size=batch_size)
        return env, end

    def sample_transitions(self, batch_size, history, rng):
        accepted = []
        while sum(len(x[0]) for x in accepted) < batch_size:
            env, end = self._sample_base(batch_size * 2, history, 1, rng)
            valid = self._valid_end(env, end) & self._valid_end(env, end + 1)
            valid &= ~self._gather(self.done, env, end[:, None]).reshape(-1)
            accepted.append((env[valid], end[valid]))
        env = np.concatenate([x[0] for x in accepted])[:batch_size]
        end = np.concatenate([x[1] for x in accepted])[:batch_size]
        slot = end % self.capacity
        return {
            "obs": self._padded_history(self.obs, env, end, history),
            "incoming": self._padded_history(self.incoming_action, env, end, history, zero_pad=True),
            "next_obs": self._padded_history(self.obs, env, end + 1, history),
            "action": self.action[env, slot],
        }

    def sample_frames(self, count, rng):
        """Random single achieved frames with their instantaneous rewards."""
        steps = min(self.total_steps, self.capacity)
        if steps == 0:
            raise RuntimeError("replay buffer is empty")
        env = rng.integers(0, self.num_envs, size=count)
        slot = rng.integers(0, steps, size=count)
        return self.obs[env, slot], self.reward[env, slot]

    @staticmethod
    def _geometric_offset(size, maximum, discount, rng):
        uniform = rng.random(size)
        normalizer = 1 - discount**maximum
        offset = np.floor(np.log1p(-uniform * normalizer) / np.log(discount)).astype(np.int64) + 1
        return np.clip(offset, 1, maximum)

    def _hindsight_rows(self, batch_size, history, max_offset, discount, rng):
        accepted = []
        while sum(len(x[0]) for x in accepted) < batch_size:
            env, end = self._sample_base(batch_size * 3, history, 1, rng)
            offset = self._geometric_offset(end.shape[0], max_offset, discount, rng)
            future = end + offset
            within = future < self.total_steps
            future = np.minimum(future, self.total_steps - 1)
            valid = within & self._valid_end(env, end) & self._valid_end(env, end + 1)
            valid &= self._valid_end(env, future)
            source_episode = self._gather(self.episode, env, end[:, None]).reshape(-1)
            future_episode = self._gather(self.episode, env, future[:, None]).reshape(-1)
            valid &= source_episode == future_episode
            accepted.append((env[valid], end[valid], future[valid]))
        env = np.concatenate([x[0] for x in accepted])[:batch_size]
        end = np.concatenate([x[1] for x in accepted])[:batch_size]
        future = np.concatenate([x[2] for x in accepted])[:batch_size]
        return env, end, future

    def sample_hindsight(self, batch_size, history, max_offset, discount, rng):
        """Hindsight tuples with FRAME goals: the goal is the single future
        observation actually reached from here; the target is the action
        actually taken."""
        env, end, future = self._hindsight_rows(
            batch_size, history, max_offset, discount, rng
        )
        return {
            "obs": self._padded_history(self.obs, env, end, history),
            "incoming": self._padded_history(self.incoming_action, env, end, history, zero_pad=True),
            "action": self.action[env, end % self.capacity],
            "goal_frame": self.obs[env, future % self.capacity],
        }


def as_tensor(batch, device):
    return torch.as_tensor(batch, device=device, dtype=torch.float32)


def train_world_step(agent, optimizer, batch, args, device):
    obs = as_tensor(batch["obs"], device)
    incoming = as_tensor(batch["incoming"], device)
    next_obs = as_tensor(batch["next_obs"], device)
    action = as_tensor(batch["action"], device)
    tokens, frame, _, memory = agent.world_model(obs, incoming)
    prediction = agent.world_model.transition(memory, action)
    # Pure LeJEPA: attached online target; SIGReg alone prevents collapse,
    # covering both the frame-summary space and the per-token-position
    # distributions the attached MSE lives in.
    target_tokens = agent.world_model.frame_encoder(next_obs)
    prediction_loss = F.mse_loss(prediction, target_tokens[:, -1])
    sigreg_loss = agent.world_model.sigreg(frame[:, -1]) + agent.world_model.sigreg(
        target_tokens[:, -1].transpose(0, 1)
    )
    loss = prediction_loss + args.sigreg_coef * sigreg_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.world_model.parameters(), args.max_grad_norm)
    optimizer.step()
    return {
        "wm/loss": loss.item(),
        "wm/prediction_mse": prediction_loss.item(),
        "wm/sigreg": sigreg_loss.item(),
        "wm/grad_norm": float(grad_norm),
    }


def train_follower_step(agent, optimizer, batch, args, device):
    obs = as_tensor(batch["obs"], device)
    incoming = as_tensor(batch["incoming"], device)
    action = as_tensor(batch["action"], device)
    with torch.no_grad():
        goal_frame = agent.encode_frames(as_tensor(batch["goal_frame"], device))
        _, frame_summaries, _, memory = agent.world_model(obs, incoming)
        current_frame = frame_summaries[:, -1]
    belief = agent.follower.read_belief(memory[:, :-1])
    # The one loss: the first action toward a frame actually reached later.
    prediction = agent.follower.action(belief, current_frame, goal_frame)
    loss = F.mse_loss(prediction, action)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.follower.parameters(), args.max_grad_norm)
    optimizer.step()
    return {
        "follower/action_mse": loss.item(),
        "follower/grad_norm": float(grad_norm),
    }


def mean_metrics(metrics):
    result = {}
    for row in metrics:
        for key, value in row.items():
            result.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in result.items()}


@torch.no_grad()
def evaluate_direct_policy(agent, args, device, run_name, goal_frame, seed_offset=10_000):
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, False, run_name + "-eval") for index in range(args.eval_envs)]
    )
    obs, _ = envs.reset(seed=args.seed + seed_offset)
    obs = np.asarray(obs, np.float32)
    obs_history = np.repeat(obs[:, None], args.history, axis=1)
    incoming_history = np.zeros(
        (args.eval_envs, args.history, envs.single_action_space.shape[0]), np.float32
    )
    returns = []
    while len(returns) < args.eval_envs:
        action, _ = agent.act(
            as_tensor(obs_history, device),
            as_tensor(incoming_history, device),
            goal_frame,
        )
        next_obs, _, terminations, truncations, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    returns.append(float(info["episode"]["r"]))
        next_obs = np.asarray(next_obs, np.float32)
        obs_history = np.roll(obs_history, -1, axis=1)
        obs_history[:, -1] = next_obs
        incoming_history = np.roll(incoming_history, -1, axis=1)
        incoming_history[:, -1] = action.cpu().numpy()
        if done.any():
            obs_history[done] = np.repeat(next_obs[done, None], args.history, axis=1)
            incoming_history[done] = 0
    envs.close()
    return float(np.mean(returns[: args.eval_envs]))


@torch.no_grad()
def evaluate_goal_reaching(agent, args, device, run_name, goal_frames):
    """Command achieved frames; measure frame-space distance to the matched
    command vs a shuffled one. Servoing requires matched < shuffled."""
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, False, run_name + "-reach") for index in range(args.eval_envs)]
    )
    obs, _ = envs.reset(seed=args.seed + 20_000)
    obs = np.asarray(obs, np.float32)
    obs_history = np.repeat(obs[:, None], args.history, axis=1)
    incoming_history = np.zeros(
        (args.eval_envs, args.history, envs.single_action_space.shape[0]), np.float32
    )
    shuffled_goals = goal_frames.roll(1, dims=0)
    matched_sum = np.zeros(args.eval_envs)
    shuffled_sum = np.zeros(args.eval_envs)
    steps = np.zeros(args.eval_envs)
    active = np.ones(args.eval_envs, dtype=np.bool_)
    while active.any():
        action, current_frame = agent.act(
            as_tensor(obs_history, device), as_tensor(incoming_history, device), goal_frames
        )
        matched = (current_frame - goal_frames).square().mean(-1).cpu().numpy()
        shuffled = (current_frame - shuffled_goals).square().mean(-1).cpu().numpy()
        matched_sum += matched * active
        shuffled_sum += shuffled * active
        steps += active
        next_obs, _, terminations, truncations, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        active &= ~done
        next_obs = np.asarray(next_obs, np.float32)
        obs_history = np.roll(obs_history, -1, axis=1)
        obs_history[:, -1] = next_obs
        incoming_history = np.roll(incoming_history, -1, axis=1)
        incoming_history[:, -1] = action.cpu().numpy()
        if done.any():
            obs_history[done] = np.repeat(next_obs[done, None], args.history, axis=1)
            incoming_history[done] = 0
    envs.close()
    return float((matched_sum / steps).mean()), float((shuffled_sum / steps).mean())


def main():
    args = tyro.cli(Args)
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("Pan Goal Solver training requires CUDA")
    if args.history < 2:
        raise ValueError("history must be at least two observations")
    if args.random_warmup_steps < args.follower_warmup_steps:
        raise ValueError("follower warmup must not exceed random exploration warmup")
    if not (
        0 <= args.command_uniform_fraction
        and 0 <= args.command_top_fraction
        and args.command_uniform_fraction + args.command_top_fraction <= 1
    ):
        raise ValueError("command mixture must be a sub-distribution")
    if args.eval_envs < 2:
        raise ValueError("reaching eval's shuffle control needs at least 2 eval envs")
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
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, args.capture_video, run_name) for index in range(args.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("only continuous Box actions are supported")
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    base_model = getattr(envs.envs[0].unwrapped, "model", None)
    if base_model is None or not hasattr(base_model, "nv"):
        raise TypeError("velocity scaling requires a MuJoCo env exposing model.nv")
    velocity_dims = int(base_model.nv)
    if not 0 < velocity_dims < obs_dim:
        raise ValueError(f"model.nv={velocity_dims} incompatible with obs_dim={obs_dim}")
    agent = PanGoalSolver(obs_dim, act_dim, args).to(device)
    world_parameters, follower_parameters = agent.parameter_groups()
    world_optimizer = torch.optim.AdamW(
        world_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    follower_optimizer = torch.optim.AdamW(
        follower_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.compile:
        agent.world_model = torch.compile(agent.world_model, mode=args.compile_mode)

    replay = VectorReplayBuffer(args.replay_size, args.num_envs, obs_dim, act_dim)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = np.asarray(next_obs, np.float32)
    obs_history = np.repeat(next_obs[:, None], args.history, axis=1)
    incoming_history = np.zeros((args.num_envs, args.history, act_dim), np.float32)
    incoming_action = np.zeros((args.num_envs, act_dim), np.float32)
    ou_noise = np.zeros((args.num_envs, act_dim), np.float32)

    # Commanded goal state: raw obs per env (re-encoded online each
    # iteration so coordinates stay fresh), kind 0 = uniform achieved,
    # 1 = top-reward, 2 = scaled top-reward.
    env_goal_obs = np.zeros((args.num_envs, obs_dim), np.float32)
    env_goal_kind = np.zeros(args.num_envs, np.int64)
    env_goals = torch.zeros(args.num_envs, args.latent_dim, device=device)
    steps_since_switch = np.zeros(args.num_envs, np.int64)
    goals_live = False

    def top_reward_frame():
        """Best instantaneous reward among random candidates. Fresh every
        call — nothing is ever frozen. Reward's only role in the system."""
        frames, rewards = replay.sample_frames(args.good_frame_candidates, rng)
        return frames[int(np.argmax(rewards))]

    def scaled_top_frame(scale):
        frame = top_reward_frame().copy()
        frame[-velocity_dims:] *= scale
        return frame

    @torch.no_grad()
    def refresh_env_goals():
        env_goals.copy_(agent.encode_frames(as_tensor(env_goal_obs, device)))

    @torch.no_grad()
    def resample_env_goals(indices):
        """Command mixture: uniform achieved / top-reward / scaled
        top-reward. Kinds 0 / 1 / 2 for the metric split."""
        if not len(indices):
            return
        for env_index in indices:
            draw = rng.random()
            if draw < args.command_uniform_fraction:
                frames, _ = replay.sample_frames(1, rng)
                env_goal_obs[env_index] = frames[0]
                env_goal_kind[env_index] = 0
            elif draw < args.command_uniform_fraction + args.command_top_fraction:
                env_goal_obs[env_index] = top_reward_frame()
                env_goal_kind[env_index] = 1
            else:
                scale = args.command_scales[rng.integers(len(args.command_scales))]
                env_goal_obs[env_index] = scaled_top_frame(scale)
                env_goal_kind[env_index] = 2
        steps_since_switch[indices] = 0
        refresh_env_goals()

    global_step = 0
    start_time = time.time()
    next_eval_step = args.eval_interval
    num_iterations = math.ceil(args.total_timesteps / (args.num_envs * args.num_steps))
    for iteration in range(1, num_iterations + 1):
        rollout_steps = min(
            args.num_steps, (args.total_timesteps - global_step) // args.num_envs
        )
        for _ in range(rollout_steps):
            with torch.no_grad():
                obs_tensor = as_tensor(obs_history, device)
                incoming_tensor = as_tensor(incoming_history, device)
                if goals_live:
                    policy_action, _ = agent.act(obs_tensor, incoming_tensor, env_goals)
                    policy_action = policy_action.cpu().numpy()
            if global_step < args.random_warmup_steps or not goals_live:
                action = rng.uniform(-1, 1, size=(args.num_envs, act_dim)).astype(np.float32)
            else:
                innovation = rng.standard_normal(ou_noise.shape).astype(np.float32)
                ou_noise = (
                    args.policy_noise_rho * ou_noise
                    + args.policy_noise_std
                    * np.sqrt(1 - args.policy_noise_rho**2)
                    * innovation
                )
                iid_noise = (
                    args.policy_innovation_std
                    * rng.standard_normal(ou_noise.shape).astype(np.float32)
                )
                action = np.clip(policy_action + ou_noise + iid_noise, -1, 1)

            step_obs, reward, terminations, truncations, infos = envs.step(action)
            done = np.logical_or(terminations, truncations)
            replay.add(next_obs, incoming_action, action, reward, done)
            global_step += args.num_envs
            steps_since_switch += 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episodic_return:.3f}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar(
                            "charts/episodic_length", float(info["episode"]["l"]), global_step
                        )

            next_obs = np.asarray(step_obs, np.float32)
            obs_history = np.roll(obs_history, -1, axis=1)
            obs_history[:, -1] = next_obs
            incoming_history = np.roll(incoming_history, -1, axis=1)
            incoming_history[:, -1] = action
            incoming_action = action.astype(np.float32)
            if done.any():
                obs_history[done] = np.repeat(next_obs[done, None], args.history, axis=1)
                incoming_history[done] = 0
                incoming_action[done] = 0
                ou_noise[done] = 0

            if goals_live:
                refresh = done | (steps_since_switch >= args.goal_switch_steps)
                if refresh.any():
                    resample_env_goals(np.flatnonzero(refresh))

        metrics = []
        if replay.size >= args.batch_size * 2:
            for _ in range(args.updates_per_iteration):
                batch = replay.sample_transitions(args.batch_size, args.history, rng)
                metrics.append(train_world_step(agent, world_optimizer, batch, args, device))
        if global_step >= args.follower_warmup_steps:
            for _ in range(args.follower_updates_per_iteration):
                batch = replay.sample_hindsight(
                    args.batch_size,
                    args.history,
                    args.max_goal_offset,
                    args.goal_discount,
                    rng,
                )
                metrics.append(
                    train_follower_step(agent, follower_optimizer, batch, args, device)
                )

        aggregated = mean_metrics(metrics)
        for key, value in aggregated.items():
            writer.add_scalar(key, value, global_step)

        # Goals go live once the policy phase begins; refresh coordinates
        # every iteration (weights just changed).
        if global_step >= args.random_warmup_steps and replay.total_steps >= args.history:
            if not goals_live:
                goals_live = True
                resample_env_goals(np.arange(args.num_envs))
            else:
                refresh_env_goals()

        for kind_value, kind_name in ((0, "uniform"), (1, "top"), (2, "scaled")):
            writer.add_scalar(
                f"goal/command_fraction_{kind_name}",
                float((env_goal_kind == kind_value).mean()),
                global_step,
            )

        if goals_live:
            with torch.no_grad():
                belief, current_frame = agent.follower_state(
                    as_tensor(obs_history, device), as_tensor(incoming_history, device)
                )
                commanded_action = agent.follower(belief, current_frame, env_goals)
                null_action = agent.follower(belief, current_frame, current_frame)
                writer.add_scalar(
                    "diagnostics/goal_removal_action_mse",
                    (commanded_action - null_action).square().mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "diagnostics/pursuit_frame_mse",
                    (current_frame - env_goals).square().mean().item(),
                    global_step,
                )

        if args.eval_interval > 0 and global_step >= next_eval_step and goals_live:
            # A/B/C arms: top-reward command, its scaled version, and a
            # uniform achieved command as the control.
            base = top_reward_frame()
            scaled = base.copy()
            scaled[-velocity_dims:] *= args.command_scales[-1]
            uniform_frames, _ = replay.sample_frames(1, rng)
            with torch.no_grad():
                arm_goals = agent.encode_frames(
                    as_tensor(np.stack([base, scaled, uniform_frames[0]]), device)
                )
            top_return = evaluate_direct_policy(agent, args, device, run_name, arm_goals[0])
            scaled_return = evaluate_direct_policy(agent, args, device, run_name, arm_goals[1])
            uniform_return = evaluate_direct_policy(agent, args, device, run_name, arm_goals[2])
            writer.add_scalar("evaluation/top_goal_return", top_return, global_step)
            writer.add_scalar("evaluation/scaled_goal_return", scaled_return, global_step)
            writer.add_scalar("evaluation/uniform_goal_return", uniform_return, global_step)
            writer.add_scalar(
                "evaluation/top_minus_uniform", top_return - uniform_return, global_step
            )
            # Reaching: the follower's unit test.
            reach_obs, _ = replay.sample_frames(args.eval_envs, rng)
            with torch.no_grad():
                reach_goals = agent.encode_frames(as_tensor(reach_obs, device))
            matched, shuffled = evaluate_goal_reaching(
                agent, args, device, run_name, reach_goals
            )
            writer.add_scalar("evaluation/reach_matched_mse", matched, global_step)
            writer.add_scalar("evaluation/reach_shuffled_mse", shuffled, global_step)
            writer.add_scalar("evaluation/reach_servo_gap", shuffled - matched, global_step)
            next_eval_step += args.eval_interval

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/replay_size", replay.size, global_step)
        print(
            f"iteration={iteration}/{num_iterations}, step={global_step}, "
            f"SPS={sps}, replay={replay.size}"
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save({"model": agent.state_dict(), "args": vars(args)}, model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

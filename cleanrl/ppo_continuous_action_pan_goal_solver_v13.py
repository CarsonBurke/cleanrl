# Pan Goal Solver v13: Pan successor-to-action control toward a decoded open goal.
#
# A reward-free LeJEPA learns observation tokens and temporal control belief.
# Pan hindsight and occupancy ground the follower in real trajectories. A goal
# head maps detached world belief to one goal latent on every step. A separate
# a frozen reconstruction decoder grounds reward evaluation. A Pan-style
# goal-conditioned desired successor and inverse head emit one action directly.
# There is no PPO, action critic, policy gradient, search, MPC, or rollout.
# Hypothesis: predicting the real next latent under hindsight goals bridges the
# off-manifold far goal to locally grounded inverse action without online search.

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
    goal_updates_per_iteration: int = 64
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
    goal_reconstruction_coef: float = 0.1
    goal_overlap_coef: float = 0.005
    ema_decay: float = 0.995

    follower_hidden: int = 512
    action_coef: float = 1.0
    future_hindsight_coef: float = 1.0
    successor_coef: float = 1.0
    occupancy_coef: float = 0.1
    goal_discount: float = 0.98
    max_action_goal_offset: int = 256
    max_occupancy_offset: int = 256
    follower_warmup_steps: int = 20_000
    random_warmup_steps: int = 100_000

    reward_prediction_coef: float = 1.0
    goal_reward_coef: float = 1.0

    explorer_fraction: float = 0.5
    exploration_innovation_std: float = 0.3
    primitive_min_steps: int = 64
    primitive_max_steps: int = 256
    primitive_min_period: float = 20.0
    primitive_max_period: float = 80.0
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
    """v215-style raw per-feature affine tokens with spatial attention."""

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


def history_statistics(frame_summaries):
    """Phase-robust ordered-history statistics; returns one vector per history."""
    mean = frame_summaries.mean(dim=-2)
    rms = frame_summaries.square().mean(dim=-2).add(1e-6).sqrt()
    differences = frame_summaries[..., 1:, :] - frame_summaries[..., :-1, :]
    diff_rms = differences.square().mean(dim=-2).add(1e-6).sqrt()
    lag_product = (frame_summaries[..., 1:, :] * frame_summaries[..., :-1, :]).mean(dim=-2)
    return torch.cat([mean, rms, diff_rms, lag_product], dim=-1)


class GoalProjector(nn.Module):
    """Compress observation-only motion-regime statistics into one goal latent."""

    def __init__(self, obs_dim, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.RMSNorm(4 * dim),
            xavier_linear(nn.Linear(4 * dim, 2 * dim)),
            nn.SiLU(),
            xavier_linear(nn.Linear(2 * dim, dim)),
        )
        self.decoder = nn.Sequential(
            nn.RMSNorm(dim),
            xavier_linear(nn.Linear(dim, 2 * dim)),
            nn.SiLU(),
            xavier_linear(nn.Linear(2 * dim, 4 * obs_dim)),
        )

    def encode(self, frame_summaries):
        stats = history_statistics(frame_summaries)
        goal = self.encoder(stats)
        return goal, stats

    def forward(self, frame_summaries, observation_history):
        goal, stats = self.encode(frame_summaries)
        raw_stats = history_statistics(torch.asinh(observation_history))
        return goal, raw_stats, self.decoder(goal)


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
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.frame_encoder = FrameEncoder(
            obs_dim, args.latent_dim, args.encoder_layers, args.heads, args.ffn_mult
        )
        self.goal_projector = GoalProjector(obs_dim, args.latent_dim)
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
        goal, stats, reconstruction = self.goal_projector(frame_summaries, obs_history)
        belief, latest_memory = self.temporal(tokens, incoming_action_history)
        return tokens, frame_summaries, goal, stats, reconstruction, belief, latest_memory

    def forward(self, obs_history, incoming_action_history):
        return self.encode_history(obs_history, incoming_action_history)

class GoalPredictor(nn.Module):
    """Predict one desired latent from detached world belief."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(dim),
            xavier_linear(nn.Linear(dim, 256)),
            nn.SiLU(),
            xavier_linear(nn.Linear(256, 256)),
            nn.SiLU(),
            xavier_linear(nn.Linear(256, dim), gain=0.05),
        )

    def forward(self, belief):
        return self.net(belief)


class RewardPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            xavier_linear(nn.Linear(dim, 256)),
            nn.SiLU(),
            xavier_linear(nn.Linear(256, 256)),
            nn.SiLU(),
            xavier_linear(nn.Linear(256, 1)),
        )

    def forward(self, goal):
        return self.net(goal).squeeze(-1)


class DirectGoalFollower(nn.Module):
    """One-pass outcome-conditioned policy; action MSE is fixed-Gaussian NLL."""

    def __init__(self, observation_tokens, belief_dim, goal_dim, act_dim, hidden):
        super().__init__()
        self.belief_readout = nn.Sequential(
            nn.RMSNorm(observation_tokens * belief_dim),
            xavier_linear(nn.Linear(observation_tokens * belief_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, belief_dim)),
        )
        self.action_trunk = nn.Sequential(
            nn.RMSNorm(belief_dim),
            xavier_linear(nn.Linear(belief_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, hidden)),
            nn.SiLU(),
        )
        self.action_head = xavier_linear(
            nn.Linear(hidden, act_dim * goal_dim, bias=False), gain=0.05
        )
        self.successor_trunk = nn.Sequential(
            nn.RMSNorm(belief_dim),
            xavier_linear(nn.Linear(belief_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, hidden)),
            nn.SiLU(),
        )
        self.successor_head = xavier_linear(
            nn.Linear(hidden, goal_dim * goal_dim, bias=False), gain=0.05
        )
        self.act_dim = act_dim
        self.goal_dim = goal_dim
        self.occupancy_belief = xavier_linear(nn.Linear(belief_dim, belief_dim, bias=False))
        self.occupancy_goal = xavier_linear(nn.Linear(goal_dim, belief_dim, bias=False))
        self.occupancy_log_temperature = nn.Parameter(torch.tensor(np.log(0.1)))

    def goal_features(self, current_goal, desired_goal):
        delta = desired_goal - current_goal
        # Preserve magnitude for achieved local transitions so inverse action
        # MSE can distinguish weak from strong actions. For a far goal this
        # smoothly converges to direction-only control, making distance moot.
        return delta / (1.0 + delta.norm(dim=-1, keepdim=True))

    def read_belief(self, observation_memory):
        return self.belief_readout(observation_memory.flatten(1))

    def action(self, belief, current_goal, desired_goal):
        direction = self.goal_features(current_goal, desired_goal)
        jacobian = self.action_head(self.action_trunk(belief)).view(
            -1, self.act_dim, self.goal_dim
        )
        return torch.tanh(torch.einsum("bad,bd->ba", jacobian, direction))

    def desired_successor(self, belief, current_goal, desired_goal):
        direction = self.goal_features(current_goal, desired_goal)
        successor_map = self.successor_head(self.successor_trunk(belief)).view(
            -1, self.goal_dim, self.goal_dim
        )
        delta = torch.einsum("bde,be->bd", successor_map, direction)
        return current_goal + delta

    def occupancy_logits(self, belief, goals):
        belief_query = F.normalize(self.occupancy_belief(belief), dim=-1)
        goal_key = F.normalize(self.occupancy_goal(goals), dim=-1)
        temperature = self.occupancy_log_temperature.exp().clamp(0.01, 1.0)
        return belief_query @ goal_key.T / temperature

    def forward(self, belief, current_goal, desired_goal):
        desired_next_goal = self.desired_successor(
            belief, current_goal, desired_goal
        )
        return self.action(belief, current_goal, desired_next_goal)


class PanGoalSolver(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.world_model = WorldModel(obs_dim, act_dim, args)
        self.target_representation = copy.deepcopy(
            nn.ModuleDict(
                {
                    "frame_encoder": self.world_model.frame_encoder,
                    "goal_projector": self.world_model.goal_projector,
                }
            )
        )
        self.target_representation.requires_grad_(False)
        self.follower = DirectGoalFollower(
            obs_dim, args.latent_dim, args.latent_dim, act_dim, args.follower_hidden
        )
        self.goal_predictor = GoalPredictor(args.latent_dim)
        self.reward_predictor = RewardPredictor(4 * obs_dim)

    def target_encode(self, obs_history):
        tokens = self.target_representation["frame_encoder"](obs_history)
        goal, _ = self.target_representation["goal_projector"].encode(tokens.mean(dim=-2))
        return tokens, goal

    def control_state(self, obs_history, incoming_action_history):
        outputs = self.world_model(obs_history, incoming_action_history)
        return outputs[5], outputs[2], outputs

    def follower_state(self, obs_history, incoming_action_history):
        outputs = self.world_model(obs_history, incoming_action_history)
        online_tokens, world_belief, memory = outputs[0], outputs[5], outputs[6]
        _, goal = self.target_encode(obs_history)
        # TemporalBelief keeps feature streams separate, so the follower readout
        # excludes its action slot. The online pass supplies the trained belief;
        # the EMA encoder supplies the shared P/R/follower goal coordinates.
        belief = self.follower.read_belief(memory[:, :-1])
        return belief, goal, online_tokens, world_belief

    def direct_action(self, obs_history, incoming_action_history, goal):
        belief, current_goal, _, _ = self.follower_state(
            obs_history, incoming_action_history
        )
        desired_goal = goal.expand(obs_history.shape[0], -1) if goal.ndim == 1 else goal
        action = self.follower(belief, current_goal, desired_goal)
        return action, current_goal

    def predict_reward(self, goal):
        decoded_statistics = self.target_representation["goal_projector"].decoder(goal)
        return self.reward_predictor(decoded_statistics)

    def act(self, obs_history, incoming_action_history):
        belief, current_goal, _, world_belief = self.follower_state(
            obs_history, incoming_action_history
        )
        goal = self.goal_predictor(world_belief.detach())
        action = self.follower(belief, current_goal, goal)
        return action, current_goal, goal

    @torch.no_grad()
    def update_target(self, decay):
        online = nn.ModuleDict(
            {
                "frame_encoder": self.world_model.frame_encoder,
                "goal_projector": self.world_model.goal_projector,
            }
        )
        for target_parameter, online_parameter in zip(
            self.target_representation.parameters(), online.parameters(), strict=True
        ):
            target_parameter.lerp_(online_parameter, 1 - decay)

    def parameter_groups(self):
        world = list(self.world_model.parameters())
        follower = list(self.follower.parameters())
        goal = list(self.goal_predictor.parameters()) + list(self.reward_predictor.parameters())
        ids = [{id(p) for p in group} for group in (world, follower, goal)]
        if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("world, follower, and reward-goal parameters must be disjoint")
        return world, follower, goal


class VectorReplayBuffer:
    """Raw per-environment ring buffer; every sampled latent is freshly encoded."""

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

    def _valid_history(self, env, end, history):
        indices = end[:, None] - np.arange(history - 1, -1, -1)[None]
        absolute = self._gather(self.absolute, env, indices)
        episodes = self._gather(self.episode, env, indices)
        return (absolute == indices).all(1) & (episodes == episodes[:, -1:]).all(1)

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
        history_offsets = np.arange(history - 1, -1, -1)
        current_indices = end[:, None] - history_offsets
        next_indices = current_indices + 1
        slot = end % self.capacity
        return {
            "obs": self._padded_history(self.obs, env, end, history),
            "incoming": self._padded_history(self.incoming_action, env, end, history, zero_pad=True),
            "next_obs": self._padded_history(self.obs, env, end + 1, history),
            "next_incoming": self._padded_history(
                self.incoming_action, env, end + 1, history, zero_pad=True
            ),
            "action": self.action[env, slot],
            "reward": self.reward[env, slot],
        }

    @staticmethod
    def _geometric_offset(size, maximum, discount, rng):
        uniform = rng.random(size)
        normalizer = 1 - discount**maximum
        offset = np.floor(np.log1p(-uniform * normalizer) / np.log(discount)).astype(np.int64) + 1
        return np.clip(offset, 1, maximum)

    def sample_hindsight(
        self, batch_size, history, max_action_offset, max_occupancy_offset, discount, rng
    ):
        # Pan-style first-action supervision samples one achieved goal across
        # the full context. Episode safety is the only trajectory restriction.
        accepted = []
        while sum(len(x[0]) for x in accepted) < batch_size:
            env, end = self._sample_base(batch_size * 3, history, 1, rng)
            offset = self._geometric_offset(
                end.shape[0], max_action_offset, discount, rng
            )
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
        occupancy = []
        while sum(len(x[0]) for x in occupancy) < batch_size:
            occ_env, occ_end = self._sample_base(batch_size * 3, history, 1, rng)
            occ_offset = self._geometric_offset(
                occ_end.shape[0], max_occupancy_offset, discount, rng
            )
            occ_future = occ_end + occ_offset
            within = occ_future < self.total_steps
            occ_future = np.minimum(occ_future, self.total_steps - 1)
            valid = within & self._valid_end(occ_env, occ_end)
            valid &= self._valid_end(occ_env, occ_future)
            source_episode = self._gather(
                self.episode, occ_env, occ_end[:, None]
            ).reshape(-1)
            future_episode = self._gather(
                self.episode, occ_env, occ_future[:, None]
            ).reshape(-1)
            valid &= source_episode == future_episode
            occupancy.append((occ_env[valid], occ_end[valid], occ_future[valid]))
        occ_env = np.concatenate([x[0] for x in occupancy])[:batch_size]
        occ_end = np.concatenate([x[1] for x in occupancy])[:batch_size]
        occ_future = np.concatenate([x[2] for x in occupancy])[:batch_size]
        return {
            "obs": self._padded_history(self.obs, env, end, history),
            "incoming": self._padded_history(self.incoming_action, env, end, history, zero_pad=True),
            "next_obs": self._padded_history(self.obs, env, end + 1, history),
            "future_obs": self._padded_history(self.obs, env, future, history),
            "action": self.action[env, end % self.capacity],
            "offset": (future - end).astype(np.float32),
            "occupancy_obs": self._padded_history(
                self.obs, occ_env, occ_end, history
            ),
            "occupancy_incoming": self._padded_history(
                self.incoming_action, occ_env, occ_end, history, zero_pad=True
            ),
            "occupancy_future_obs": self._padded_history(
                self.obs, occ_env, occ_future, history
            ),
        }

class FourierExplorer:
    def __init__(self, num_envs, act_dim, args, rng):
        self.num_envs = num_envs
        self.act_dim = act_dim
        self.args = args
        self.rng = rng
        self.time = np.zeros(num_envs, np.int64)
        self.remaining = np.zeros(num_envs, np.int64)
        self.period = np.ones(num_envs, np.float32)
        self.amplitude = np.zeros((num_envs, 3, act_dim), np.float32)
        self.phase = np.zeros((num_envs, 3, act_dim), np.float32)
        self.offset = np.zeros((num_envs, act_dim), np.float32)
        self.resample(np.ones(num_envs, dtype=np.bool_))

    def resample(self, mask):
        indices = np.flatnonzero(mask)
        if not len(indices):
            return
        self.time[indices] = 0
        self.remaining[indices] = self.rng.integers(
            self.args.primitive_min_steps, self.args.primitive_max_steps + 1, size=len(indices)
        )
        self.period[indices] = self.rng.uniform(
            self.args.primitive_min_period, self.args.primitive_max_period, size=len(indices)
        )
        self.amplitude[indices] = self.rng.uniform(
            -0.7, 0.7, size=(len(indices), 3, self.act_dim)
        )
        self.amplitude[indices] /= np.arange(1, 4, dtype=np.float32)[None, :, None]
        self.phase[indices] = self.rng.uniform(
            0, 2 * np.pi, size=(len(indices), 3, self.act_dim)
        )
        self.offset[indices] = self.rng.uniform(-0.15, 0.15, size=(len(indices), self.act_dim))

    def action(self):
        expired = self.remaining <= 0
        self.resample(expired)
        harmonics = np.arange(1, 4, dtype=np.float32)[None, :, None]
        angle = 2 * np.pi * harmonics * self.time[:, None, None] / self.period[:, None, None]
        action = self.offset + (self.amplitude * np.sin(angle + self.phase)).sum(1)
        self.time += 1
        self.remaining -= 1
        return np.tanh(action).astype(np.float32)


def as_tensor(batch, device):
    return torch.as_tensor(batch, device=device, dtype=torch.float32)


def train_world_step(agent, optimizer, batch, args, device):
    obs = as_tensor(batch["obs"], device)
    incoming = as_tensor(batch["incoming"], device)
    next_obs = as_tensor(batch["next_obs"], device)
    action = as_tensor(batch["action"], device)
    outputs = agent.world_model(obs, incoming)
    tokens, frame, goal, stats, reconstruction, _, memory = outputs
    prediction = agent.world_model.transition(memory, action)
    with torch.no_grad():
        target_tokens, target_goal = agent.target_encode(next_obs)
    prediction_loss = F.mse_loss(prediction, target_tokens[:, -1])
    reconstruction_loss = F.mse_loss(reconstruction, stats.detach())
    overlap_loss = F.mse_loss(goal, target_goal)
    frame_samples = frame[:, -1]
    sigreg_loss = agent.world_model.sigreg(frame_samples) + agent.world_model.sigreg(goal)
    loss = (
        prediction_loss
        + args.goal_reconstruction_coef * reconstruction_loss
        + args.goal_overlap_coef * overlap_loss
        + args.sigreg_coef * sigreg_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.world_model.parameters(), args.max_grad_norm)
    optimizer.step()
    agent.update_target(args.ema_decay)
    return {
        "wm/loss": loss.item(),
        "wm/prediction_mse": prediction_loss.item(),
        "wm/goal_reconstruction_mse": reconstruction_loss.item(),
        "wm/goal_overlap_mse": overlap_loss.item(),
        "wm/sigreg": sigreg_loss.item(),
        "wm/grad_norm": float(grad_norm),
    }


def train_follower_step(agent, optimizer, batch, args, device):
    obs = as_tensor(batch["obs"], device)
    incoming = as_tensor(batch["incoming"], device)
    next_obs = as_tensor(batch["next_obs"], device)
    future_obs = as_tensor(batch["future_obs"], device)
    occupancy_obs = as_tensor(batch["occupancy_obs"], device)
    occupancy_incoming = as_tensor(batch["occupancy_incoming"], device)
    occupancy_future_obs = as_tensor(batch["occupancy_future_obs"], device)
    action = as_tensor(batch["action"], device)
    with torch.no_grad():
        _, all_goals = agent.target_encode(
            torch.cat(
                [obs, next_obs, future_obs, occupancy_future_obs],
                dim=0,
            )
        )
        current_goal, next_goal, future_goal, occupancy_future_goal = all_goals.chunk(4)
        combined_outputs = agent.world_model(
            torch.cat([obs, occupancy_obs], dim=0),
            torch.cat(
                [
                    incoming,
                    occupancy_incoming,
                ],
                dim=0,
            ),
        )
        combined_memory = combined_outputs[6]
        current_memory, occupancy_memory = combined_memory.chunk(2)
    belief = agent.follower.read_belief(current_memory[:, :-1].detach())
    occupancy_belief = agent.follower.read_belief(occupancy_memory[:, :-1].detach())
    one_step_action = agent.follower.action(belief, current_goal, next_goal)
    future_action = agent.follower(belief, current_goal, future_goal)
    one_step_successor = agent.follower.desired_successor(
        belief, current_goal, next_goal
    )
    future_successor = agent.follower.desired_successor(
        belief, current_goal, future_goal
    )
    one_step_action_loss = F.mse_loss(one_step_action, action)
    future_action_loss = F.mse_loss(future_action, action)
    action_loss = one_step_action_loss + args.future_hindsight_coef * future_action_loss
    successor_mse = (
        F.mse_loss(one_step_successor, next_goal)
        + F.mse_loss(future_successor, next_goal)
    )
    successor_scale = (next_goal - current_goal).square().mean().detach().clamp_min(1e-4)
    successor_loss = successor_mse / successor_scale
    occupancy_logits = agent.follower.occupancy_logits(
        occupancy_belief, occupancy_future_goal
    )
    occupancy_loss = F.cross_entropy(
        occupancy_logits, torch.arange(action.shape[0], device=device)
    )
    loss = (
        args.action_coef * action_loss
        + args.successor_coef * successor_loss
        + args.occupancy_coef * occupancy_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.follower.parameters(), args.max_grad_norm)
    optimizer.step()
    with torch.no_grad():
        achieved_progress = (
            (current_goal - future_goal).square().mean(-1)
            - (next_goal - future_goal).square().mean(-1)
        )
    return {
        "follower/loss": loss.item(),
        "follower/action_mse": action_loss.item(),
        "follower/one_step_action_mse": one_step_action_loss.item(),
        "follower/future_action_mse": future_action_loss.item(),
        "follower/successor_mse": successor_mse.item(),
        "follower/successor_relative_mse": successor_loss.item(),
        "follower/occupancy_loss": occupancy_loss.item(),
        "follower/achieved_progress": achieved_progress.mean().item(),
        "follower/achieved_progress_positive_fraction": (achieved_progress > 0).float().mean().item(),
        "follower/grad_norm": float(grad_norm),
    }


def train_goal_step(agent, goal_optimizer, reward_optimizer, batch, args, device):
    obs = as_tensor(batch["obs"], device)
    incoming = as_tensor(batch["incoming"], device)
    action = as_tensor(batch["action"], device)
    reward = as_tensor(batch["reward"], device)
    with torch.no_grad():
        outputs = agent.world_model(obs, incoming)
        world_belief, latest_memory = outputs[5], outputs[6]
        current_tokens, _ = agent.target_encode(obs)
        predicted_next_tokens = agent.world_model.transition(latest_memory, action)
        frame_history = current_tokens.mean(dim=-2)
        predicted_frame = predicted_next_tokens.mean(dim=-2)
        predicted_history = torch.cat(
            [frame_history[:, 1:], predicted_frame[:, None]], dim=1
        )
        predicted_next_goal, _ = agent.target_representation["goal_projector"].encode(
            predicted_history
        )

    reward_prediction = agent.predict_reward(predicted_next_goal.detach())
    reward_loss = F.mse_loss(reward_prediction, reward)
    reward_optimizer.zero_grad(set_to_none=True)
    (args.reward_prediction_coef * reward_loss).backward()
    reward_parameters = list(agent.reward_predictor.parameters())
    reward_grad_norm = nn.utils.clip_grad_norm_(reward_parameters, args.max_grad_norm)
    reward_optimizer.step()

    for parameter in reward_parameters:
        parameter.requires_grad_(False)
    goal = agent.goal_predictor(world_belief.detach())
    predicted_goal_reward = agent.predict_reward(goal)
    goal_reward_loss = -predicted_goal_reward.mean()
    goal_optimizer.zero_grad(set_to_none=True)
    (args.goal_reward_coef * goal_reward_loss).backward()
    for parameter in reward_parameters:
        parameter.requires_grad_(True)
    goal_grad_norm = nn.utils.clip_grad_norm_(
        agent.goal_predictor.parameters(), args.max_grad_norm
    )
    goal_optimizer.step()
    loss = (
        args.reward_prediction_coef * reward_loss.detach()
        + args.goal_reward_coef * goal_reward_loss.detach()
    )
    return {
        "goal/loss": loss.item(),
        "goal/reward_prediction_mse": reward_loss.item(),
        "goal/reward_prediction_mae": (reward_prediction - reward).abs().mean().item(),
        "goal/predicted_goal_reward": predicted_goal_reward.mean().item(),
        "goal/predicted_goal_norm": goal.norm(dim=-1).mean().item(),
        "goal/reward_grad_norm": float(reward_grad_norm),
        "goal/predictor_grad_norm": float(goal_grad_norm),
        "goal/reward_mean": reward.mean().item(),
        "goal/reward_max": reward.max().item(),
    }


def mean_metrics(metrics):
    result = {}
    for row in metrics:
        for key, value in row.items():
            result.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in result.items()}


@torch.no_grad()
def evaluate_direct_policy(agent, args, device, run_name):
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, False, run_name + "-eval") for index in range(args.eval_envs)]
    )
    obs, _ = envs.reset(seed=args.seed + 10_000)
    obs = np.asarray(obs, np.float32)
    obs_history = np.repeat(obs[:, None], args.history, axis=1)
    incoming_history = np.zeros(
        (args.eval_envs, args.history, envs.single_action_space.shape[0]), np.float32
    )
    returns = []
    while len(returns) < args.eval_envs:
        action, _, _ = agent.act(
            as_tensor(obs_history, device),
            as_tensor(incoming_history, device),
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
    return returns[: args.eval_envs]


def main():
    args = tyro.cli(Args)
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("Pan Goal Solver training requires CUDA")
    if args.history < 2:
        raise ValueError("history must be at least two observations")
    if args.random_warmup_steps < args.follower_warmup_steps:
        raise ValueError("follower warmup must not exceed random exploration warmup")
    if args.goal_updates_per_iteration != args.follower_updates_per_iteration:
        raise ValueError("the goal predictor must update on every follower optimization cycle")
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
    agent = PanGoalSolver(obs_dim, act_dim, args).to(device)
    world_parameters, follower_parameters, _ = agent.parameter_groups()
    world_optimizer = torch.optim.AdamW(
        world_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    jacobian_parameters = list(agent.follower.action_head.parameters())
    jacobian_ids = {id(parameter) for parameter in jacobian_parameters}
    follower_optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter in follower_parameters
                    if id(parameter) not in jacobian_ids
                ],
                "weight_decay": args.weight_decay,
            },
            {"params": jacobian_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    goal_optimizer = torch.optim.AdamW(
        agent.goal_predictor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    reward_optimizer = torch.optim.AdamW(
        agent.reward_predictor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.compile:
        agent.world_model = torch.compile(agent.world_model, mode=args.compile_mode)

    replay = VectorReplayBuffer(args.replay_size, args.num_envs, obs_dim, act_dim)
    explorer = FourierExplorer(args.num_envs, act_dim, args, rng)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = np.asarray(next_obs, np.float32)
    obs_history = np.repeat(next_obs[:, None], args.history, axis=1)
    incoming_history = np.zeros((args.num_envs, args.history, act_dim), np.float32)
    incoming_action = np.zeros((args.num_envs, act_dim), np.float32)
    ou_noise = np.zeros((args.num_envs, act_dim), np.float32)
    episode_explorer = np.ones(args.num_envs, dtype=np.bool_)
    revision_anchor_batch = None
    previous_anchor_goals = None
    next_eval_step = args.eval_interval

    global_step = 0
    start_time = time.time()
    num_iterations = math.ceil(args.total_timesteps / (args.num_envs * args.num_steps))
    for iteration in range(1, num_iterations + 1):
        rollout_steps = min(
            args.num_steps, (args.total_timesteps - global_step) // args.num_envs
        )
        goal_mse_before = np.zeros((rollout_steps, args.num_envs), np.float32)
        goal_mse_after = np.full((rollout_steps, args.num_envs), np.nan, np.float32)
        sampled_goals = np.zeros(
            (rollout_steps, args.num_envs, args.latent_dim), np.float32
        )
        transition_done = np.zeros((rollout_steps, args.num_envs), np.bool_)
        policy_transition = np.zeros((rollout_steps, args.num_envs), np.bool_)
        policy_action_abs = []
        for step in range(rollout_steps):
            primitive_action = np.clip(
                explorer.action()
                + args.exploration_innovation_std
                * rng.standard_normal((args.num_envs, act_dim)).astype(np.float32),
                -1,
                1,
            )
            with torch.no_grad():
                obs_tensor = as_tensor(obs_history, device)
                incoming_tensor = as_tensor(incoming_history, device)
                policy_action, current_goal, step_goal = agent.act(obs_tensor, incoming_tensor)
                if step > 0:
                    previous_step_goal = as_tensor(sampled_goals[step - 1], device)
                    goal_mse_after[step - 1] = (
                        current_goal - previous_step_goal
                    ).square().mean(-1).cpu().numpy()
                goal_mse_before[step] = (
                    current_goal - step_goal
                ).square().mean(-1).cpu().numpy()
                sampled_goals[step] = step_goal.cpu().numpy()
                policy_action = policy_action.cpu().numpy()
            if global_step < args.random_warmup_steps:
                action = primitive_action
            else:
                innovation = rng.standard_normal(ou_noise.shape).astype(np.float32)
                ou_noise = (
                    args.policy_noise_rho * ou_noise
                    + args.policy_noise_std
                    * np.sqrt(1 - args.policy_noise_rho**2)
                    * innovation
                )
                iid_policy_noise = (
                    args.policy_innovation_std
                    * rng.standard_normal(ou_noise.shape).astype(np.float32)
                )
                noisy_policy = np.clip(
                    policy_action + ou_noise + iid_policy_noise, -1, 1
                )
                policy_transition[step] = ~episode_explorer
                action = np.where(episode_explorer[:, None], primitive_action, noisy_policy)
                policy_action_abs.append(np.abs(policy_action[~episode_explorer]).mean() if (~episode_explorer).any() else 0.0)

            next_step_obs, reward, terminations, truncations, infos = envs.step(action)
            done = np.logical_or(terminations, truncations)
            replay.add(next_obs, incoming_action, action, reward, done)
            global_step += args.num_envs
            transition_done[step] = done

            if "final_info" in infos:
                for env_index, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        episodic_return = float(info["episode"]["r"])
                        episodic_length = float(info["episode"]["l"])
                        print(f"global_step={global_step}, episodic_return={episodic_return:.3f}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                        tag = "explorer" if episode_explorer[env_index] else "policy"
                        writer.add_scalar(f"evaluation/{tag}_episodic_return", episodic_return, global_step)

            next_obs = np.asarray(next_step_obs, np.float32)
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
                explorer.resample(done)
                if global_step < args.random_warmup_steps:
                    episode_explorer[done] = True
                else:
                    episode_explorer[done] = rng.random(done.sum()) < args.explorer_fraction

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
                    args.max_action_goal_offset,
                    args.max_occupancy_offset,
                    args.goal_discount,
                    rng,
                )
                metrics.append(
                    train_follower_step(
                        agent,
                        follower_optimizer,
                        batch,
                        args,
                        device,
                    )
                )
                # The reward and goal heads receive one fresh optimization on
                # every follower cycle, never an occasional outer-loop update.
                goal_batch = replay.sample_transitions(args.batch_size, args.history, rng)
                metrics.append(
                    train_goal_step(
                        agent,
                        goal_optimizer,
                        reward_optimizer,
                        goal_batch,
                        args,
                        device,
                    )
                )
        if global_step >= args.random_warmup_steps:
            if revision_anchor_batch is None:
                revision_anchor_batch = replay.sample_transitions(
                    args.batch_size, args.history, rng
                )

        aggregated = mean_metrics(metrics)
        for key, value in aggregated.items():
            writer.add_scalar(key, value, global_step)
        if revision_anchor_batch is not None:
            with torch.no_grad():
                anchor_obs = as_tensor(revision_anchor_batch["obs"], device)
                anchor_incoming = as_tensor(revision_anchor_batch["incoming"], device)
                _, _, _, anchor_world_belief = agent.follower_state(
                    anchor_obs, anchor_incoming
                )
                anchor_goals = agent.goal_predictor(anchor_world_belief)
                anchor_goal_rewards = agent.predict_reward(anchor_goals)
                writer.add_scalar(
                    "goal/aspirational_predicted_reward",
                    anchor_goal_rewards.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "goal/aspirational_norm",
                    anchor_goals.norm(dim=-1).mean().item(),
                    global_step,
                )
                if previous_anchor_goals is not None:
                    writer.add_scalar(
                        "goal/revision_mse",
                        (anchor_goals - previous_anchor_goals).square().mean().item(),
                        global_step,
                    )
                previous_anchor_goals = anchor_goals.clone()
        if global_step >= args.random_warmup_steps and rollout_steps > 1:
            valid = (~transition_done[:-1]) & policy_transition[:-1]
            progress = goal_mse_before[:-1][valid] - goal_mse_after[:-1][valid]
            if policy_transition.any():
                writer.add_scalar(
                    "goal/frozen_policy_mse",
                    float(goal_mse_before[policy_transition].mean()),
                    global_step,
                )
            if progress.size:
                writer.add_scalar("goal/frozen_progress_mean", float(progress.mean()), global_step)
                writer.add_scalar("goal/frozen_progress_positive_fraction", float((progress > 0).mean()), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/replay_size", replay.size, global_step)
        writer.add_scalar("charts/explorer_fraction", float(episode_explorer.mean()), global_step)
        if policy_action_abs:
            writer.add_scalar("follower/policy_action_abs_mean", float(np.mean(policy_action_abs)), global_step)
        if global_step >= args.random_warmup_steps:
            with torch.no_grad():
                diagnostic_obs = as_tensor(obs_history, device)
                diagnostic_incoming = as_tensor(incoming_history, device)
                diagnostic_belief, diagnostic_current_goal, _, diagnostic_world_belief = agent.follower_state(
                    diagnostic_obs, diagnostic_incoming
                )
                predicted_goal = agent.goal_predictor(diagnostic_world_belief)
                goal_action = agent.follower(
                    diagnostic_belief, diagnostic_current_goal, predicted_goal
                )
                zero_action = agent.follower(
                    diagnostic_belief, diagnostic_current_goal, diagnostic_current_goal
                )
                shuffled_action = agent.follower(
                    diagnostic_belief,
                    diagnostic_current_goal,
                    predicted_goal.roll(1, dims=0),
                )
                writer.add_scalar(
                    "diagnostics/goal_removal_action_mse",
                    (goal_action - zero_action).square().mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "diagnostics/goal_shuffle_action_mse",
                    (goal_action - shuffled_action).square().mean().item(),
                    global_step,
                )
        if args.eval_interval > 0 and global_step >= next_eval_step and global_step >= args.random_warmup_steps:
            evaluation_returns = evaluate_direct_policy(agent, args, device, run_name)
            writer.add_scalar(
                "evaluation/direct_episodic_return",
                float(np.mean(evaluation_returns)),
                global_step,
            )
            next_eval_step += args.eval_interval
        print(
            f"iteration={iteration}/{num_iterations}, step={global_step}, "
            f"SPS={int(global_step / (time.time() - start_time))}, replay={replay.size}"
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save({"model": agent.state_dict(), "args": vars(args)}, model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

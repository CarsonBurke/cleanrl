# Pan Goal Solver v3: outcome-conditioned direct control toward an aspirational latent.
#
# A reward-free LeJEPA learns observation tokens and temporal control belief.
# Each action is paired with a future reached inside the same behavior primitive;
# local branch pairs force the direct policy to use that outcome. Reward trains
# only a global affine goal ray P(y)=mu+s(y)d;
# the cached above-support point is never conditioned on the current state.
# There is no PPO, action critic, policy gradient, search, MPC, or rollout.
# Hypothesis: coherent reward-independent exploration supplies controllable gait
# fragments that the direct follower can recombine toward an unreachable ideal.

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
    updates_per_iteration: int = 16
    follower_updates_per_iteration: int = 64
    goal_updates_per_iteration: int = 8
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
    occupancy_coef: float = 0.1
    counterfactual_coef: float = 1.0
    branch_ranking_coef: float = 0.25
    branch_margin: float = 0.05
    counterfactual_match_radius: float = 1.0
    goal_discount: float = 0.98
    max_action_goal_offset: int = 16
    max_occupancy_offset: int = 256
    follower_warmup_steps: int = 20_000
    random_warmup_steps: int = 100_000

    reward_past: int = 32
    reward_future: int = 32
    reward_scale: float = 5.0
    aspiration_reward: float = 25.0
    ray_anchor_reward: float = 5.0
    utility_coef: float = 1.0
    direction_tangent_coef: float = 0.25

    explorer_fraction: float = 0.5
    exploration_innovation_std: float = 0.3
    primitive_min_steps: int = 64
    primitive_max_steps: int = 256
    primitive_min_period: float = 20.0
    primitive_max_period: float = 80.0
    policy_noise_rho: float = 0.9
    policy_noise_std: float = 0.25
    eval_interval: int = 250_000
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

class AffineGoalProposer(nn.Module):
    """One global deterministic reward ray; no current-state input exists."""

    def __init__(self, dim, reward_scale, anchor_reward):
        super().__init__()
        self.reward_scale = reward_scale
        self.anchor_reward = anchor_reward
        self.mu = nn.Parameter(torch.zeros(dim))
        self.direction = nn.Parameter(torch.randn(dim) * 0.01)

    def reward_coordinate(self, reward_rate):
        return torch.asinh(reward_rate / self.reward_scale)

    def forward(self, reward_rate):
        anchor = self.reward_coordinate(
            torch.as_tensor(self.anchor_reward, device=reward_rate.device, dtype=reward_rate.dtype)
        )
        coordinate = (self.reward_coordinate(reward_rate) - anchor).unsqueeze(-1)
        return self.mu + coordinate * self.direction


class UtilityModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(dim),
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
            nn.Linear(hidden, act_dim * goal_dim, bias=False), gain=0.01
        )
        self.act_dim = act_dim
        self.goal_dim = goal_dim
        self.occupancy_belief = xavier_linear(nn.Linear(belief_dim, belief_dim, bias=False))
        self.occupancy_goal = xavier_linear(nn.Linear(goal_dim, belief_dim, bias=False))
        self.occupancy_log_temperature = nn.Parameter(torch.tensor(np.log(0.1)))

    def goal_features(self, current_goal, desired_goal):
        # This is exactly the negative normalized gradient of Euclidean goal
        # MSE. Distance along the same far-away ray is intentionally irrelevant.
        return F.normalize(desired_goal - current_goal, dim=-1, eps=1e-6)

    def read_belief(self, observation_memory):
        return self.belief_readout(observation_memory.flatten(1))

    def action(self, belief, current_goal, desired_goal):
        direction = self.goal_features(current_goal, desired_goal)
        jacobian = self.action_head(self.action_trunk(belief)).view(
            -1, self.act_dim, self.goal_dim
        )
        return torch.tanh(
            torch.einsum("bad,bd->ba", jacobian, direction)
            / np.sqrt(self.goal_dim)
        )

    def occupancy_logits(self, belief, goals):
        belief_query = F.normalize(self.occupancy_belief(belief), dim=-1)
        goal_key = F.normalize(self.occupancy_goal(goals), dim=-1)
        temperature = self.occupancy_log_temperature.exp().clamp(0.01, 1.0)
        return belief_query @ goal_key.T / temperature

    def forward(self, belief, current_goal, desired_goal):
        return self.action(belief, current_goal, desired_goal)


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
        self.proposer = AffineGoalProposer(
            args.latent_dim, args.reward_scale, args.ray_anchor_reward
        )
        self.utility = UtilityModel(args.latent_dim)

    def target_encode(self, obs_history):
        tokens = self.target_representation["frame_encoder"](obs_history)
        goal, _ = self.target_representation["goal_projector"].encode(tokens.mean(dim=-2))
        return tokens, goal

    def control_state(self, obs_history, incoming_action_history):
        outputs = self.world_model(obs_history, incoming_action_history)
        return outputs[5], outputs[2], outputs

    def follower_state(self, obs_history, incoming_action_history):
        tokens, goal = self.target_encode(obs_history)
        # The follower is deliberately observation-only. Persistent exploration
        # makes a_t too predictable from recent actions and otherwise lets H/pi
        # ignore the goal; outbound actions remain available to WM training.
        _, memory = self.world_model.temporal(tokens, torch.zeros_like(incoming_action_history))
        belief = self.follower.read_belief(memory[:, :-1])
        return belief, goal, tokens

    def direct_action(self, obs_history, incoming_action_history, cached_goal):
        belief, current_goal, _ = self.follower_state(obs_history, incoming_action_history)
        goal = cached_goal.expand(obs_history.shape[0], -1)
        action = self.follower(belief, current_goal, goal)
        return action, current_goal

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
        goal = list(self.proposer.parameters()) + list(self.utility.parameters())
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
        self.segment = np.zeros((num_envs, self.capacity), np.int64)
        self.absolute = np.full((num_envs, self.capacity), -1, np.int64)
        self.total_steps = 0
        self.episode_ids = np.arange(num_envs, dtype=np.int64)

    @property
    def size(self):
        return min(self.total_steps, self.capacity) * self.num_envs

    def add(self, obs, incoming_action, action, reward, done, segment=None):
        slot = self.total_steps % self.capacity
        self.obs[:, slot] = obs
        self.incoming_action[:, slot] = incoming_action
        self.action[:, slot] = action
        self.reward[:, slot] = reward
        self.done[:, slot] = done
        self.episode[:, slot] = self.episode_ids
        self.segment[:, slot] = 0 if segment is None else segment
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
        # First-action supervision uses an outcome reached under the same
        # coherent behavior command. Occupancy remains Pan-style and samples
        # the full episode independently below.
        accepted = []
        while sum(len(x[0]) for x in accepted) < batch_size:
            env, end = self._sample_base(batch_size * 3, history, 1, rng)
            offset = self._geometric_offset(
                end.shape[0], max_action_offset, discount, rng
            )
            offset[rng.random(end.shape[0]) < 0.5] = 1
            future = end + offset
            within = future < self.total_steps
            future = np.minimum(future, self.total_steps - 1)
            valid = within & self._valid_end(env, end) & self._valid_end(env, end + 1)
            valid &= self._valid_end(env, future)
            source_episode = self._gather(self.episode, env, end[:, None]).reshape(-1)
            future_episode = self._gather(self.episode, env, future[:, None]).reshape(-1)
            source_segment = self._gather(self.segment, env, end[:, None]).reshape(-1)
            future_segment = self._gather(self.segment, env, future[:, None]).reshape(-1)
            # A future reached after a different motor command is not evidence
            # for which first action pursues the requested outcome.
            valid &= (source_episode == future_episode) & (source_segment == future_segment)
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

    def sample_reward_goals(self, batch_size, history, past, future, rng):
        accepted = []
        room = max(1, future)
        while sum(len(x[0]) for x in accepted) < batch_size:
            env, end = self._sample_base(batch_size * 3, max(history, past + 1), room, rng)
            reward_start = end - past
            reward_end = end + future - 1
            valid = self._valid_end(env, end)
            valid &= self._valid_history(env, reward_start, 1) & self._valid_history(env, reward_end, 1)
            source_episode = self._gather(self.episode, env, end[:, None]).reshape(-1)
            start_episode = self._gather(self.episode, env, reward_start[:, None]).reshape(-1)
            end_episode = self._gather(self.episode, env, reward_end[:, None]).reshape(-1)
            valid &= (source_episode == start_episode) & (source_episode == end_episode)
            accepted.append((env[valid], end[valid]))
        env = np.concatenate([x[0] for x in accepted])[:batch_size]
        end = np.concatenate([x[1] for x in accepted])[:batch_size]
        history_indices = end[:, None] - np.arange(history - 1, -1, -1)
        reward_indices = end[:, None] + np.arange(-past, future)[None]
        reward_window = self._gather(self.reward, env, reward_indices)
        center = past
        scales = [width for width in (8, 32, past + future) if width <= past + future]
        rates = []
        for width in dict.fromkeys(scales):
            if width == past + future:
                rates.append(reward_window.mean(1))
            else:
                left = center - width // 2
                rates.append(reward_window[:, left : left + width].mean(1))
        reward_rate = np.stack(rates, axis=1).mean(1)
        return self._padded_history(self.obs, env, end, history), reward_rate


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
        self.segment_ids = np.arange(num_envs, dtype=np.int64)
        self.resample(np.ones(num_envs, dtype=np.bool_))

    def resample(self, mask):
        indices = np.flatnonzero(mask)
        if not len(indices):
            return
        self.time[indices] = 0
        self.segment_ids[indices] += self.num_envs
        self.remaining[indices] = self.rng.integers(
            self.args.primitive_min_steps, self.args.primitive_max_steps + 1, size=len(indices)
        )
        self.period[indices] = self.rng.uniform(
            self.args.primitive_min_period, self.args.primitive_max_period, size=len(indices)
        )
        self.amplitude[indices] = self.rng.uniform(-0.7, 0.7, size=(len(indices), 3, self.act_dim))
        self.amplitude[indices] /= np.arange(1, 4, dtype=np.float32)[None, :, None]
        self.phase[indices] = self.rng.uniform(0, 2 * np.pi, size=(len(indices), 3, self.act_dim))
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


def physical_band_ids(reward_rate):
    boundaries = torch.tensor(
        [-1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0],
        device=reward_rate.device,
        dtype=reward_rate.dtype,
    )
    return torch.bucketize(reward_rate, boundaries)


def physical_band_balanced_mse(prediction, target, reward_rate):
    bin_id = physical_band_ids(reward_rate)
    losses = (prediction - target).square().mean(-1)
    band_losses = [losses[bin_id == index].mean() for index in bin_id.unique()]
    return torch.stack(band_losses).mean()


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


def train_follower_step(agent, optimizer, batch, args, device, rng):
    obs = as_tensor(batch["obs"], device)
    incoming = as_tensor(batch["incoming"], device)
    next_obs = as_tensor(batch["next_obs"], device)
    future_obs = as_tensor(batch["future_obs"], device)
    occupancy_obs = as_tensor(batch["occupancy_obs"], device)
    occupancy_incoming = as_tensor(batch["occupancy_incoming"], device)
    occupancy_future_obs = as_tensor(batch["occupancy_future_obs"], device)
    action = as_tensor(batch["action"], device)
    with torch.no_grad():
        all_tokens, all_goals = agent.target_encode(
            torch.cat(
                [obs, next_obs, future_obs, occupancy_obs, occupancy_future_obs],
                dim=0,
            )
        )
        current_tokens, _, _, occupancy_tokens, _ = all_tokens.chunk(5)
        current_goal, next_goal, future_goal, _, occupancy_future_goal = all_goals.chunk(5)
        _, combined_memory = agent.world_model.temporal(
            torch.cat([current_tokens, occupancy_tokens], dim=0),
            torch.zeros_like(torch.cat([incoming, occupancy_incoming], dim=0)),
        )
        current_memory, occupancy_memory = combined_memory.chunk(2)
    belief = agent.follower.read_belief(current_memory[:, :-1].detach())
    occupancy_belief = agent.follower.read_belief(occupancy_memory[:, :-1].detach())
    one_step_action = agent.follower(belief, current_goal, next_goal)
    future_action = agent.follower(belief, current_goal, future_goal)
    one_step_action_loss = F.mse_loss(one_step_action, action)
    future_action_loss = F.mse_loss(future_action, action)
    action_loss = 0.5 * (one_step_action_loss + future_action_loss)
    occupancy_logits = agent.follower.occupancy_logits(
        occupancy_belief, occupancy_future_goal
    )
    occupancy_loss = F.cross_entropy(
        occupancy_logits, torch.arange(action.shape[0], device=device)
    )
    with torch.no_grad():
        match_state = torch.cat([belief.detach(), current_goal], dim=-1)
        match_state = (match_state - match_state.mean(0)) / match_state.std(
            0, unbiased=False
        ).clamp_min(1e-3)
        goal_distance = torch.cdist(match_state, match_state) / np.sqrt(match_state.shape[-1])
        action_distance = torch.cdist(action, action)
        invalid_match = action_distance < 0.5
        invalid_match.fill_diagonal_(True)
        goal_distance.masked_fill_(invalid_match, torch.inf)
        matched = goal_distance.argmin(dim=1)
        nearest_distance = goal_distance.gather(1, matched[:, None]).squeeze(1)
        valid_match = (
            torch.isfinite(nearest_distance)
            & (nearest_distance <= args.counterfactual_match_radius)
        )
    if valid_match.any():
        selected = torch.nonzero(valid_match, as_tuple=False).squeeze(1)
        partner = matched[selected]
        action_a = agent.follower(
            belief[selected], current_goal[selected], future_goal[selected]
        )
        action_b = agent.follower(
            belief[selected], current_goal[selected], future_goal[partner]
        )
        counterfactual_loss = 0.5 * (
            F.mse_loss(action_a, action[selected])
            + F.mse_loss(action_b, action[partner])
        )
        correct_error = 0.5 * (
            (action_a - action[selected]).square().mean(-1)
            + (action_b - action[partner]).square().mean(-1)
        )
        swapped_error = 0.5 * (
            (action_a - action[partner]).square().mean(-1)
            + (action_b - action[selected]).square().mean(-1)
        )
        branch_ranking_loss = F.relu(
            args.branch_margin + correct_error - swapped_error
        ).mean()
        counterfactual_action_sensitivity = (action_a - action_b).square().mean()
    else:
        counterfactual_loss = action_loss.new_zeros(())
        branch_ranking_loss = action_loss.new_zeros(())
        counterfactual_action_sensitivity = action_loss.new_zeros(())
    loss = (
        args.action_coef * action_loss
        + args.occupancy_coef * occupancy_loss
        + args.counterfactual_coef * counterfactual_loss
        + args.branch_ranking_coef * branch_ranking_loss
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
        aspiration_goal = agent.proposer(
            torch.tensor([args.aspiration_reward], device=device)
        ).detach()
        aspirational_progress = (
            (current_goal - aspiration_goal).square().mean(-1)
            - (next_goal - aspiration_goal).square().mean(-1)
        )
    return {
        "follower/loss": loss.item(),
        "follower/action_mse": action_loss.item(),
        "follower/one_step_action_mse": one_step_action_loss.item(),
        "follower/future_action_mse": future_action_loss.item(),
        "follower/occupancy_loss": occupancy_loss.item(),
        "follower/counterfactual_loss": counterfactual_loss.item(),
        "follower/branch_ranking_loss": branch_ranking_loss.item(),
        "follower/counterfactual_action_sensitivity": counterfactual_action_sensitivity.item(),
        "follower/counterfactual_match_fraction": valid_match.float().mean().item(),
        "follower/achieved_progress": achieved_progress.mean().item(),
        "follower/achieved_progress_positive_fraction": (achieved_progress > 0).float().mean().item(),
        "follower/aspirational_progress": aspirational_progress.mean().item(),
        "follower/grad_norm": float(grad_norm),
    }


def train_goal_step(agent, optimizer, obs, reward_rate, args, device):
    obs = as_tensor(obs, device)
    reward_rate = as_tensor(reward_rate, device)
    with torch.no_grad():
        _, goal_target = agent.target_encode(obs)
    prediction = agent.proposer(reward_rate)
    utility_prediction = agent.utility(goal_target)
    proposer_loss = physical_band_balanced_mse(prediction, goal_target, reward_rate)
    reward_coordinate = agent.proposer.reward_coordinate(reward_rate)
    centered_coordinate = reward_coordinate - reward_coordinate.mean()
    centered_goal = goal_target - goal_target.mean(0)
    empirical_tangent = (
        (centered_coordinate[:, None] * centered_goal).sum(0)
        / centered_coordinate.square().sum().clamp_min(1e-4)
    )
    # Regression fixes scale; this auxiliary only stabilizes the orientation of
    # the high-reward ray instead of exploding on sparsely populated band gaps.
    tangent_loss = 1 - F.cosine_similarity(
        agent.proposer.direction.unsqueeze(0), empirical_tangent.unsqueeze(0)
    ).mean()
    utility_loss = F.huber_loss(utility_prediction, reward_rate)
    loss = (
        proposer_loss
        + args.direction_tangent_coef * tangent_loss
        + args.utility_coef * utility_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(
        list(agent.proposer.parameters()) + list(agent.utility.parameters()), args.max_grad_norm
    )
    optimizer.step()
    return {
        "goal/loss": loss.item(),
        "goal/proposer_mse": proposer_loss.item(),
        "goal/direction_tangent_cosine_loss": tangent_loss.item(),
        "goal/utility_huber": utility_loss.item(),
        "goal/utility_mae": (utility_prediction - reward_rate).abs().mean().item(),
        "goal/grad_norm": float(grad_norm),
        "goal/reward_rate_mean": reward_rate.mean().item(),
        "goal/reward_rate_max": reward_rate.max().item(),
    }


def mean_metrics(metrics):
    result = {}
    for row in metrics:
        for key, value in row.items():
            result.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in result.items()}


def nearest_anchor_mse(anchor_latents, goal):
    return (anchor_latents - goal).square().mean(-1).min()


@torch.no_grad()
def evaluate_direct_policy(agent, cached_goal, args, device, run_name):
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
        action, _ = agent.direct_action(
            as_tensor(obs_history, device), as_tensor(incoming_history, device), cached_goal
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
    world_parameters, follower_parameters, goal_parameters = agent.parameter_groups()
    world_optimizer = torch.optim.AdamW(
        world_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    follower_optimizer = torch.optim.AdamW(
        follower_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    goal_optimizer = torch.optim.AdamW(
        goal_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
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
    aspiration = torch.tensor([args.aspiration_reward], device=device)
    with torch.no_grad():
        cached_goal = agent.proposer(aspiration)[0].detach()
    previous_goal = cached_goal.clone()
    previous_direction = agent.proposer.direction.detach().clone()
    revision_anchor_obs = None
    previous_anchor_latents = None
    next_eval_step = args.eval_interval

    global_step = 0
    start_time = time.time()
    num_iterations = math.ceil(args.total_timesteps / (args.num_envs * args.num_steps))
    for iteration in range(1, num_iterations + 1):
        rollout_steps = min(
            args.num_steps, (args.total_timesteps - global_step) // args.num_envs
        )
        goal_mse = np.zeros((rollout_steps, args.num_envs), np.float32)
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
            if global_step < args.random_warmup_steps:
                action = primitive_action
            else:
                with torch.no_grad():
                    obs_tensor = as_tensor(obs_history, device)
                    incoming_tensor = as_tensor(incoming_history, device)
                    policy_action, current_goal = agent.direct_action(
                        obs_tensor, incoming_tensor, cached_goal
                    )
                    goal_mse[step] = (
                        current_goal - cached_goal.unsqueeze(0)
                    ).square().mean(-1).cpu().numpy()
                    policy_action = policy_action.cpu().numpy()
                innovation = rng.standard_normal(ou_noise.shape).astype(np.float32)
                ou_noise = (
                    args.policy_noise_rho * ou_noise
                    + args.policy_noise_std
                    * np.sqrt(1 - args.policy_noise_rho**2)
                    * innovation
                )
                noisy_policy = np.clip(policy_action + ou_noise, -1, 1)
                policy_transition[step] = ~episode_explorer
                action = np.where(episode_explorer[:, None], primitive_action, noisy_policy)
                policy_action_abs.append(np.abs(policy_action[~episode_explorer]).mean() if (~episode_explorer).any() else 0.0)

            next_step_obs, reward, terminations, truncations, infos = envs.step(action)
            done = np.logical_or(terminations, truncations)
            behavior_segment = np.where(
                episode_explorer, explorer.segment_ids, -iteration
            )
            replay.add(next_obs, incoming_action, action, reward, done, behavior_segment)
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
                    train_follower_step(agent, follower_optimizer, batch, args, device, rng)
                )
        if global_step >= args.random_warmup_steps:
            for _ in range(args.goal_updates_per_iteration):
                goal_obs, reward_rate = replay.sample_reward_goals(
                    args.batch_size,
                    args.history,
                    args.reward_past,
                    args.reward_future,
                    rng,
                )
                metrics.append(
                    train_goal_step(agent, goal_optimizer, goal_obs, reward_rate, args, device)
                )
            with torch.no_grad():
                cached_goal = agent.proposer(aspiration)[0].detach()
            if revision_anchor_obs is None:
                revision_anchor_obs, revision_anchor_rewards = replay.sample_reward_goals(
                    args.batch_size,
                    args.history,
                    args.reward_past,
                    args.reward_future,
                    rng,
                )

        aggregated = mean_metrics(metrics)
        for key, value in aggregated.items():
            writer.add_scalar(key, value, global_step)
        with torch.no_grad():
            predicted_utility = agent.utility(cached_goal.unsqueeze(0)).item()
            raw_revision = (cached_goal - previous_goal).square().mean().item()
            direction_norm = agent.proposer.direction.norm().item()
            goal_norm = cached_goal.norm().item()
            direction_cosine = F.cosine_similarity(
                agent.proposer.direction, previous_direction, dim=0
            ).item()
        writer.add_scalar("goal/aspirational_predicted_utility", predicted_utility, global_step)
        writer.add_scalar("goal/raw_revision_mse", raw_revision, global_step)
        writer.add_scalar("goal/direction_norm", direction_norm, global_step)
        writer.add_scalar("goal/direction_update_cosine", direction_cosine, global_step)
        writer.add_scalar("goal/aspirational_norm", goal_norm, global_step)
        if "goal/reward_rate_max" in aggregated:
            writer.add_scalar(
                "goal/aspiration_support_margin",
                args.aspiration_reward - aggregated["goal/reward_rate_max"],
                global_step,
            )
        if revision_anchor_obs is not None:
            with torch.no_grad():
                anchor_obs_tensor = as_tensor(revision_anchor_obs, device)
                _, anchor_latents = agent.target_encode(anchor_obs_tensor)
                heldout_utility = agent.utility(anchor_latents)
                heldout_rewards = as_tensor(revision_anchor_rewards, device)
                writer.add_scalar(
                    "goal/heldout_utility_mae",
                    (heldout_utility - heldout_rewards).abs().mean().item(),
                    global_step,
                )
                nearest = nearest_anchor_mse(anchor_latents, cached_goal)
                writer.add_scalar("goal/aspirational_nearest_anchor_mse", nearest.item(), global_step)
                if previous_anchor_latents is not None:
                    current_mean = anchor_latents.mean(0)
                    previous_mean = previous_anchor_latents.mean(0)
                    covariance = (anchor_latents - current_mean).T @ (
                        previous_anchor_latents - previous_mean
                    )
                    u, _, vh = torch.linalg.svd(covariance)
                    rotation = u @ vh
                    aligned_goal = (cached_goal - current_mean) @ rotation + previous_mean
                    aligned_revision = (aligned_goal - previous_goal).square().mean()
                    aligned_anchor = (anchor_latents - current_mean) @ rotation + previous_mean
                    alignment_error = (aligned_anchor - previous_anchor_latents).square().mean()
                    writer.add_scalar("goal/aligned_revision_mse", aligned_revision.item(), global_step)
                    writer.add_scalar("goal/anchor_alignment_mse", alignment_error.item(), global_step)
                previous_anchor_latents = anchor_latents.clone()
        previous_goal = cached_goal.clone()
        previous_direction = agent.proposer.direction.detach().clone()
        if global_step >= args.random_warmup_steps and rollout_steps > 1:
            valid = (~transition_done[:-1]) & policy_transition[:-1]
            progress = goal_mse[:-1][valid] - goal_mse[1:][valid]
            if policy_transition.any():
                writer.add_scalar(
                    "goal/frozen_policy_mse",
                    float(goal_mse[policy_transition].mean()),
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
                diagnostic_belief, diagnostic_current_goal, _ = agent.follower_state(
                    diagnostic_obs, diagnostic_incoming
                )
                zero_action = agent.follower(
                    diagnostic_belief, diagnostic_current_goal, diagnostic_current_goal
                )
                aspiration_actions = []
                for level in (10.0, 15.0, 20.0, 25.0):
                    level_goal = agent.proposer(torch.tensor([level], device=device))[0]
                    level_action = agent.follower(
                        diagnostic_belief,
                        diagnostic_current_goal,
                        level_goal.expand(args.num_envs, -1),
                    )
                    aspiration_actions.append(level_action)
                    writer.add_scalar(
                        f"diagnostics/aspiration_{int(level)}_action_abs",
                        level_action.abs().mean().item(),
                        global_step,
                    )
                writer.add_scalar(
                    "diagnostics/goal_removal_action_mse",
                    (aspiration_actions[-1] - zero_action).square().mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "diagnostics/aspiration_10_to_25_action_mse",
                    (aspiration_actions[-1] - aspiration_actions[0]).square().mean().item(),
                    global_step,
                )
        if args.eval_interval > 0 and global_step >= next_eval_step and global_step >= args.random_warmup_steps:
            evaluation_returns = evaluate_direct_policy(agent, cached_goal, args, device, run_name)
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

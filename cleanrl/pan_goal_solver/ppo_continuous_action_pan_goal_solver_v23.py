# Pan Goal Solver v23: faithful online Pan-1 — two quantities, frame goals,
# goal-directed collection, imagined-observation aspiration. Minimal stack.
#
# The user's directive: "Pan-1 but without needing pretraining data or a
# specific goal", with the learning rule being "whatever Pan-1 does".
# Per PAN1.md that is exactly two central state-only quantities plus a thin
# action attachment:
#   #1 goal-occupancy value  — here a contrastive successor readout over
#      (belief, goal-frame) pairs at geometrically discounted offsets, made
#      LOAD-BEARING: it grounds imagined commands by picking the largest
#      velocity scale whose predicted occupancy still clears the achieved-
#      frame pool (Pan-1's "nearest achievable interpretation" behavior).
#   #2 next-frame DISTRIBUTION of the goal-conditioned policy — a K-way
#      winner-take-all hypothesis head D(belief, f_t, G) (v19-v22's
#      deterministic MSE successor contradicted the reference and collapsed
#      multimodal futures to off-manifold means).
#   actions — single-step inverse dynamics I(belief, f_t, f_target), dense
#      and goal-free (Pan-1: "closer to a single-step prediction problem").
#      Acting composes: pick D's most likely hypothesis, invert it.
#
# What replaces Pan-1's pretraining corpus: goal-directed collection from
# our own replay. Every policy env is commanded a goal FRAME that switches
# within episodes (~64 steps, Pan-like horizons): mostly random achieved
# frames (diversity => I(action; goal | belief) > 0 in the data, the
# v14-v22 identifiability hole), partly imagined frames — a top-reward
# replay frame with its velocity coordinates scaled, occupancy-gated.
# Rebuilt fresh at every switch: there is no anchor, no ratchet, no frozen
# aspiration, no reward head, no scripted explorer, no proposer network.
# Reward's only role: sorting which achieved frame gets scaled (and later,
# eval commands). Goals are frame-level (a specific future situation, as in
# Pan-1) — the phase-robust statistics latents of v14-v22 made the
# successor target not-a-function-of-the-goal and are gone.
#
# The world model is pure v17-lineage LeJEPA (attached online target,
# token-space SIGReg, no EMA) and nothing but its own losses trains it.
# No PPO, no reward critic, no policy gradient, no planning; inference is
# one direct policy pass.
#
# Pre-registered 1.5M falsifiers:
# 1. evaluation/reach_servo_gap sustained > 0 (frame-space matched vs
#    shuffled command distance) — the follower's unit test.
# 2. diagnostics/goal_removal_action_mse alive (>= 0.05-ish, not 0.001).
# 3. commanded-aspiration eval arm separating from the base arm while
#    episodic returns climb past the family's ~1500-1800 retrieval band.

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
    # Quantity #2: K-hypothesis winner-take-all next-frame head. WTA keeps
    # hypotheses on-manifold where MSE would average modes; the epsilon term
    # keeps losing hypotheses trainable.
    next_frame_hypotheses: int = 8
    wta_epsilon: float = 0.05
    distribution_coef: float = 1.0
    inverse_coef: float = 1.0
    composed_coef: float = 0.5
    occupancy_coef: float = 1.0
    goal_discount: float = 0.98
    max_goal_offset: int = 256
    follower_warmup_steps: int = 20_000
    random_warmup_steps: int = 100_000

    # Goal-directed collection: commanded goal frames switch within
    # episodes; a fraction are imagined (velocity-scaled top-reward frames),
    # the rest random achieved frames.
    goal_switch_steps: int = 64
    imagined_fraction: float = 0.4
    good_frame_candidates: int = 256
    aspiration_scales: tuple[float, ...] = (1.25, 1.5, 2.0, 3.0)
    # Occupancy gate: an imagined frame is feasible if its predicted
    # occupancy clears this quantile of a random achieved-frame pool.
    occupancy_pool: int = 64
    occupancy_feasibility_quantile: float = 0.25

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


class GoalFollower(nn.Module):
    """Pan-1's two quantities plus a thin action attachment.

    D(belief, f_t, G): K-hypothesis next-frame distribution of the
    goal-conditioned policy (quantity #2). I(belief, f_t, f_target):
    single-step inverse dynamics (the action attachment). Occupancy
    (quantity #1): contrastive successor readout over (belief, goal-frame)
    pairs, consumed by the imagined-goal feasibility gate."""

    def __init__(self, observation_tokens, belief_dim, act_dim, hidden, hypotheses):
        super().__init__()
        self.hypotheses = hypotheses
        self.belief_dim = belief_dim
        self.belief_readout = nn.Sequential(
            nn.RMSNorm(observation_tokens * belief_dim),
            xavier_linear(nn.Linear(observation_tokens * belief_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, belief_dim)),
        )
        self.distribution_net = nn.Sequential(
            nn.RMSNorm(2 * belief_dim),
            xavier_linear(nn.Linear(2 * belief_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, hypotheses * (belief_dim + 1)), gain=0.05),
        )
        self.inverse_net = nn.Sequential(
            nn.RMSNorm(2 * belief_dim),
            xavier_linear(nn.Linear(2 * belief_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, act_dim), gain=0.05),
        )
        self.occupancy_belief = xavier_linear(nn.Linear(belief_dim, belief_dim, bias=False))
        self.occupancy_goal = xavier_linear(nn.Linear(belief_dim, belief_dim, bias=False))
        self.occupancy_log_temperature = nn.Parameter(torch.tensor(np.log(0.1)))

    @staticmethod
    def bounded_delta(source, target):
        delta = target - source
        return delta / (1.0 + delta.norm(dim=-1, keepdim=True))

    def read_belief(self, observation_memory):
        return self.belief_readout(observation_memory.flatten(1))

    def next_frame_distribution(self, belief, current_frame, goal_frame):
        features = self.bounded_delta(current_frame, goal_frame)
        raw = self.distribution_net(torch.cat([belief, features], dim=-1))
        raw = raw.view(-1, self.hypotheses, self.belief_dim + 1)
        hypotheses = current_frame.unsqueeze(1) + raw[..., : self.belief_dim]
        logits = raw[..., -1]
        return hypotheses, logits

    def inverse_action(self, belief, current_frame, target_frame):
        features = self.bounded_delta(current_frame, target_frame)
        return torch.tanh(self.inverse_net(torch.cat([belief, features], dim=-1)))

    def action(self, belief, current_frame, goal_frame):
        hypotheses, logits = self.next_frame_distribution(belief, current_frame, goal_frame)
        chosen = hypotheses.gather(
            1,
            logits.argmax(dim=-1).view(-1, 1, 1).expand(-1, 1, hypotheses.shape[-1]),
        ).squeeze(1)
        return self.inverse_action(belief, current_frame, chosen)

    def occupancy_logits(self, belief, goal_frames):
        belief_query = F.normalize(self.occupancy_belief(belief), dim=-1)
        goal_key = F.normalize(self.occupancy_goal(goal_frames), dim=-1)
        temperature = self.occupancy_log_temperature.exp().clamp(0.01, 1.0)
        return belief_query @ goal_key.T / temperature

    def forward(self, belief, current_frame, goal_frame):
        return self.action(belief, current_frame, goal_frame)


class PanGoalSolver(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.world_model = WorldModel(obs_dim, act_dim, args)
        self.follower = GoalFollower(
            obs_dim,
            args.latent_dim,
            act_dim,
            args.follower_hidden,
            args.next_frame_hypotheses,
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
            "next_incoming": self._padded_history(
                self.incoming_action, env, end + 1, history, zero_pad=True
            ),
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
        observation actually reached; the distribution target is the single
        next observation; occupancy rows are sampled independently."""
        env, end, future = self._hindsight_rows(
            batch_size, history, max_offset, discount, rng
        )
        occ_env, occ_end, occ_future = self._hindsight_rows(
            batch_size, history, max_offset, discount, rng
        )
        return {
            "obs": self._padded_history(self.obs, env, end, history),
            "incoming": self._padded_history(self.incoming_action, env, end, history, zero_pad=True),
            "action": self.action[env, end % self.capacity],
            "next_frame": self.obs[env, (end + 1) % self.capacity],
            "goal_frame": self.obs[env, future % self.capacity],
            "occ_obs": self._padded_history(self.obs, occ_env, occ_end, history),
            "occ_incoming": self._padded_history(
                self.incoming_action, occ_env, occ_end, history, zero_pad=True
            ),
            "occ_goal_frame": self.obs[occ_env, occ_future % self.capacity],
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
        frames = agent.encode_frames(
            torch.cat(
                [
                    as_tensor(batch["next_frame"], device),
                    as_tensor(batch["goal_frame"], device),
                    as_tensor(batch["occ_goal_frame"], device),
                ],
                dim=0,
            )
        )
        next_frame, goal_frame, occ_goal_frame = frames.chunk(3)
        combined = agent.world_model(
            torch.cat([obs, as_tensor(batch["occ_obs"], device)], dim=0),
            torch.cat([incoming, as_tensor(batch["occ_incoming"], device)], dim=0),
        )
        memory = combined[3]
        frame_summaries = combined[1]
        current_memory, occ_memory = memory.chunk(2)
        current_frame = frame_summaries.chunk(2)[0][:, -1]
    belief = agent.follower.read_belief(current_memory[:, :-1].detach())
    occ_belief = agent.follower.read_belief(occ_memory[:, :-1].detach())

    # Quantity #2: WTA over K hypotheses — the closest hypothesis owns the
    # target (mode-seeking, stays on-manifold); logits learn which mode the
    # goal-conditioned policy actually realizes.
    hypotheses, logits = agent.follower.next_frame_distribution(
        belief, current_frame, goal_frame
    )
    distances = (hypotheses - next_frame.unsqueeze(1)).square().mean(-1)
    winner = distances.argmin(dim=-1)
    wta_loss = distances.gather(1, winner.unsqueeze(-1)).mean()
    spread_loss = distances.mean()
    logit_loss = F.cross_entropy(logits, winner)
    distribution_loss = (
        wta_loss + args.wta_epsilon * spread_loss + logit_loss
    )

    # Action attachment: dense single-step inverse dynamics, plus a composed
    # term training I on D's chosen (detached) hypothesis so the act-time
    # path sees its own inputs.
    inverse_prediction = agent.follower.inverse_action(belief, current_frame, next_frame)
    inverse_loss = F.mse_loss(inverse_prediction, action)
    chosen = hypotheses.detach().gather(
        1,
        logits.detach().argmax(dim=-1).view(-1, 1, 1).expand(-1, 1, hypotheses.shape[-1]),
    ).squeeze(1)
    composed_prediction = agent.follower.inverse_action(belief, current_frame, chosen)
    composed_loss = F.mse_loss(composed_prediction, action)

    # Quantity #1: contrastive successor over independent (belief, achieved
    # future frame) pairs.
    occupancy_logits = agent.follower.occupancy_logits(occ_belief, occ_goal_frame)
    occupancy_loss = F.cross_entropy(
        occupancy_logits, torch.arange(action.shape[0], device=device)
    )

    loss = (
        args.distribution_coef * distribution_loss
        + args.inverse_coef * inverse_loss
        + args.composed_coef * composed_loss
        + args.occupancy_coef * occupancy_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.follower.parameters(), args.max_grad_norm)
    optimizer.step()
    return {
        "follower/loss": loss.item(),
        "follower/wta_mse": wta_loss.item(),
        "follower/wta_identity_mse": F.mse_loss(current_frame, next_frame).item(),
        "follower/hypothesis_spread_mse": spread_loss.item(),
        "follower/logit_ce": logit_loss.item(),
        "follower/inverse_action_mse": inverse_loss.item(),
        "follower/composed_action_mse": composed_loss.item(),
        "follower/occupancy_loss": occupancy_loss.item(),
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
    if not 0 <= args.imagined_fraction <= 1:
        raise ValueError("imagined fraction must be a probability")
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
    # iteration so coordinates stay fresh), kind 0 = achieved, 1 = imagined.
    env_goal_obs = np.zeros((args.num_envs, obs_dim), np.float32)
    env_goal_kind = np.zeros(args.num_envs, np.int64)
    env_goals = torch.zeros(args.num_envs, args.latent_dim, device=device)
    steps_since_switch = np.zeros(args.num_envs, np.int64)
    goals_live = False

    def top_reward_frame():
        """A frame worth aspiring from: best instantaneous reward among
        random candidates. Fresh every call — nothing is ever frozen."""
        frames, rewards = replay.sample_frames(args.good_frame_candidates, rng)
        return frames[int(np.argmax(rewards))]

    @torch.no_grad()
    def imagined_goal_obs(belief_row):
        """Imagined observation: velocity-scale a top-reward frame, command
        the largest scale whose predicted occupancy clears the achieved-pool
        quantile (Pan-1's feasibility grounding, quantity #1 at work)."""
        base = top_reward_frame()
        scales = (1.0,) + tuple(args.aspiration_scales)
        candidates = np.repeat(base[None], len(scales), axis=0)
        for row, scale in enumerate(scales):
            candidates[row, -velocity_dims:] *= scale
        pool, _ = replay.sample_frames(args.occupancy_pool, rng)
        frames = agent.encode_frames(
            as_tensor(np.concatenate([candidates, pool], axis=0), device)
        )
        logits = agent.follower.occupancy_logits(belief_row.unsqueeze(0), frames)[0]
        candidate_logits = logits[: len(scales)]
        floor = torch.quantile(logits[len(scales):], args.occupancy_feasibility_quantile)
        passing = [row for row in range(len(scales)) if candidate_logits[row] >= floor]
        chosen = max(passing) if passing else 0
        return candidates[chosen], scales[chosen]

    @torch.no_grad()
    def refresh_env_goals():
        env_goals.copy_(agent.encode_frames(as_tensor(env_goal_obs, device)))

    @torch.no_grad()
    def resample_env_goals(indices, beliefs):
        if not len(indices):
            return
        chosen_scales = []
        for position, env_index in enumerate(indices):
            if rng.random() < args.imagined_fraction:
                goal_obs, scale = imagined_goal_obs(beliefs[position])
                env_goal_obs[env_index] = goal_obs
                # Gate fallback to the unscaled base is an achieved command.
                env_goal_kind[env_index] = 1 if scale > 1.0 else 0
                chosen_scales.append(scale)
            else:
                frames, _ = replay.sample_frames(1, rng)
                env_goal_obs[env_index] = frames[0]
                env_goal_kind[env_index] = 0
        steps_since_switch[indices] = 0
        refresh_env_goals()
        return chosen_scales

    global_step = 0
    start_time = time.time()
    next_eval_step = args.eval_interval
    num_iterations = math.ceil(args.total_timesteps / (args.num_envs * args.num_steps))
    for iteration in range(1, num_iterations + 1):
        rollout_steps = min(
            args.num_steps, (args.total_timesteps - global_step) // args.num_envs
        )
        iteration_scales = []
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
                    indices = np.flatnonzero(refresh)
                    with torch.no_grad():
                        beliefs, _ = agent.follower_state(
                            as_tensor(obs_history, device),
                            as_tensor(incoming_history, device),
                        )
                    scales = resample_env_goals(indices, beliefs[indices])
                    if scales:
                        iteration_scales.extend(scales)

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
                with torch.no_grad():
                    beliefs, _ = agent.follower_state(
                        as_tensor(obs_history, device), as_tensor(incoming_history, device)
                    )
                resample_env_goals(np.arange(args.num_envs), beliefs)
            else:
                refresh_env_goals()

        if iteration_scales:
            writer.add_scalar(
                "goal/imagined_scale_mean", float(np.mean(iteration_scales)), global_step
            )
            writer.add_scalar(
                "goal/imagined_scale_max", float(np.max(iteration_scales)), global_step
            )
        writer.add_scalar(
            "goal/imagined_command_fraction",
            float((env_goal_kind == 1).mean()),
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
            # Aspiration A/B: imagined command vs its own unscaled base.
            base = top_reward_frame()
            with torch.no_grad():
                eval_belief, _ = agent.follower_state(
                    as_tensor(obs_history[: args.eval_envs], device),
                    as_tensor(incoming_history[: args.eval_envs], device),
                )
                imagined_obs, imagined_scale = imagined_goal_obs(eval_belief[0])
                goal_pair = agent.encode_frames(
                    as_tensor(np.stack([base, imagined_obs]), device)
                )
            base_return = evaluate_direct_policy(agent, args, device, run_name, goal_pair[0])
            imagined_return = evaluate_direct_policy(agent, args, device, run_name, goal_pair[1])
            writer.add_scalar("evaluation/base_goal_return", base_return, global_step)
            writer.add_scalar("evaluation/imagined_goal_return", imagined_return, global_step)
            writer.add_scalar(
                "evaluation/imagined_minus_base", imagined_return - base_return, global_step
            )
            writer.add_scalar("evaluation/imagined_eval_scale", imagined_scale, global_step)
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

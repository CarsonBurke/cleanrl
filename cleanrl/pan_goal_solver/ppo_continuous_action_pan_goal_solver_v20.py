# Pan Goal Solver v20: v19 with the composed action loss attached through S.
#
# v19 evidence at 1.5M: successor beats the identity baseline 2x and inverse
# dynamics is sharp (0.039 MSE), but goal_removal_action_mse flatlined at
# 0.001 (vs 0.05+ for v16/v17's monolithic head) — the goal channel died.
# Cause: S's frame-MSE is minimized by the near-marginal next frame (the
# G-conditional component is second-order in a near-deterministic env), and
# detaching S in the composed loss removed the only first-order goal->action
# gradient. v20 attaches it: hindsight first-action error now trains S's
# goal-dependence exactly where control needs it, while the frame-MSE
# (coef 1.0 vs composed 0.5) and I's true-frame inverse loss keep both
# heads anchored to real dynamics. Everything else is v19 (below).
#
# ---- v19 header ----
# Pan Goal Solver v19: v17 with the Pan-1 follower factorization restored.
#
# v16's monolithic MLP head (belief, goal delta) -> action silently dropped
# the family's Pan-style factorization when it replaced the failed linear
# servo. v19 restores it with nonlinear heads and a deliberate choice of
# intermediate space: the desired-successor S(b, f_t, g, G) = f_t + net(b,
# bounded(G - g)) is conditioned in goal-latent space but PREDICTS the next
# frame summary (supervised toward the frame that actually followed), and
# one-step inverse dynamics I(b, f_t, f_target) is trained densely on every
# replay transition, goal-free. The goal latent cannot be the intermediate:
# its 16-frame phase-robust statistics are engineered to be near
# one-step-invariant, so one-step goal-latent deltas carry ~no action
# information — frame summaries carry it at full strength (and this matches
# Pan-1's literal "next-frame distribution"). Acting composes both heads in
# one fused pass: a = I(b, f_t, S(b, f_t, g, G)) — a feedback law that
# re-plots the one-step frame target from wherever the agent actually is,
# then closes that one-step error. A composed action loss trains the
# act-time path on hindsight tuples with S detached. The world model is
# pure v17 LeJEPA (attached online target, token-space SIGReg, no EMA) and
# nothing outside its own losses backpropagates into it — occupancy trains
# follower-side readout heads on a detached belief only.
# There is no PPO, action critic, policy gradient, search, MPC, or rollout.
# Hypothesis: dense inverse dynamics carries the control burden, the
# successor isolates goal-directedness where hindsight supervision is
# strongest, and the composition turns commanded-goal pursuit into repeated
# one-step error reduction — consolidating the intermittent v16/v17
# ignitions into sustained running.

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
    goal_reconstruction_coef: float = 0.1
    goal_overlap_coef: float = 0.005

    follower_hidden: int = 512
    # Factored follower losses: desired-successor prediction (hindsight),
    # dense inverse dynamics, and the composed act-time path.
    successor_coef: float = 1.0
    inverse_coef: float = 1.0
    composed_coef: float = 0.5
    occupancy_coef: float = 0.1
    goal_discount: float = 0.98
    max_action_goal_offset: int = 256
    max_occupancy_offset: int = 256
    follower_warmup_steps: int = 20_000
    random_warmup_steps: int = 100_000

    reward_prediction_coef: float = 1.0
    # Sustained reward-rate window for anchor selection and the rate head. Long
    # enough to span several gait cycles so transient bursts score poorly.
    anchor_reward_window: int = 100
    # Additive forward-velocity aspiration: for a forward-moving anchor the
    # commanded goal is the anchor uniformly velocity-scaled by
    # min((v_fwd + delta)/v_fwd, cap); stationary/backwards anchors are
    # commanded unscaled (uniform scaling preserves gait direction, so
    # amplifying them would be wrong-direction pressure). Slow anchors get
    # cap-bounded pressure; fast anchors a shrinking multiplicative margin.
    velocity_aspiration_delta: float = 1.0
    velocity_scale_cap: float = 3.0

    explorer_fraction: float = 0.5
    exploration_innovation_std: float = 0.3
    primitive_min_steps: int = 64
    primitive_max_steps: int = 256
    # Gait-frequency band: HalfCheetah strides span ~5-15 steps; the previous
    # 20-80 band could not excite locomotion (measured, see header).
    primitive_min_period: float = 6.0
    primitive_max_period: float = 30.0
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

class RewardPredictor(nn.Module):
    """Detached windowed reward-rate probe over goal latents (diagnostics only)."""
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
    """Pan-style factored follower: desired successor, then inverse dynamics.

    S(belief, f_t, g, G) is conditioned on the goal in goal-latent space but
    predicts the next FRAME SUMMARY (the goal latent is phase-robust and near
    one-step-invariant, so it cannot carry the intermediate); I(belief, f_t,
    f_target) is one-step inverse dynamics over frame deltas. Acting composes
    them in one fused pass: a = I(b, f, S(b, f, g, G)). Hindsight supervises
    S toward the frame that actually followed; every replay transition
    supervises I densely and goal-independently, so the hard goal-directed
    part is isolated in a state-space prediction problem (Pan-1's framing)."""

    def __init__(self, observation_tokens, belief_dim, goal_dim, act_dim, hidden):
        super().__init__()
        self.belief_readout = nn.Sequential(
            nn.RMSNorm(observation_tokens * belief_dim),
            xavier_linear(nn.Linear(observation_tokens * belief_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, belief_dim)),
        )
        self.successor_net = nn.Sequential(
            nn.RMSNorm(belief_dim + goal_dim),
            xavier_linear(nn.Linear(belief_dim + goal_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, goal_dim), gain=0.05),
        )
        self.inverse_net = nn.Sequential(
            nn.RMSNorm(belief_dim + goal_dim),
            xavier_linear(nn.Linear(belief_dim + goal_dim, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            xavier_linear(nn.Linear(hidden, act_dim), gain=0.05),
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

    def desired_successor(self, belief, current_frame, current_goal, desired_goal):
        """Predict the next frame summary en route to the goal. The goal
        conditions in goal-latent space; the prediction lives in frame space,
        where one env step carries full-strength action information (the
        goal latent is deliberately phase-robust and near one-step-invariant,
        so it cannot serve as the intermediate)."""
        features = self.goal_features(current_goal, desired_goal)
        return current_frame + self.successor_net(torch.cat([belief, features], dim=-1))

    def inverse_action(self, belief, current_frame, target_frame):
        features = self.goal_features(current_frame, target_frame)
        return torch.tanh(self.inverse_net(torch.cat([belief, features], dim=-1)))

    def action(self, belief, current_frame, current_goal, desired_goal):
        successor = self.desired_successor(
            belief, current_frame, current_goal, desired_goal
        )
        return self.inverse_action(belief, current_frame, successor)

    def occupancy_logits(self, belief, goals):
        belief_query = F.normalize(self.occupancy_belief(belief), dim=-1)
        goal_key = F.normalize(self.occupancy_goal(goals), dim=-1)
        temperature = self.occupancy_log_temperature.exp().clamp(0.01, 1.0)
        return belief_query @ goal_key.T / temperature

    def forward(self, belief, current_frame, current_goal, desired_goal):
        return self.action(belief, current_frame, current_goal, desired_goal)


class PanGoalSolver(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.world_model = WorldModel(obs_dim, act_dim, args)
        self.follower = DirectGoalFollower(
            obs_dim, args.latent_dim, args.latent_dim, act_dim, args.follower_hidden
        )
        self.reward_predictor = RewardPredictor(args.latent_dim)

    def encode_goal(self, obs_history):
        """Online frame/goal encoding (pure LeJEPA: no EMA teacher). Callers
        that must not update the world model wrap this in no_grad."""
        tokens = self.world_model.frame_encoder(obs_history)
        goal, _ = self.world_model.goal_projector.encode(tokens.mean(dim=-2))
        return tokens, goal

    def follower_state(self, obs_history, incoming_action_history):
        outputs = self.world_model(obs_history, incoming_action_history)
        online_tokens, frame_summaries, goal, world_belief, memory = (
            outputs[0], outputs[1], outputs[2], outputs[5], outputs[6],
        )
        # TemporalBelief keeps feature streams separate, so the follower readout
        # excludes its action slot. One online pass supplies the trained belief,
        # the shared P/R/follower goal coordinates, and the current frame
        # summary (the successor/inverse intermediate space).
        belief = self.follower.read_belief(memory[:, :-1])
        return belief, frame_summaries[:, -1], goal, online_tokens, world_belief

    def act(self, obs_history, incoming_action_history, desired_goal):
        belief, current_frame, current_goal, _, _ = self.follower_state(
            obs_history, incoming_action_history
        )
        goal = desired_goal.unsqueeze(0).expand(obs_history.shape[0], -1)
        action = self.follower(belief, current_frame, current_goal, goal)
        return action, current_goal, goal

    @torch.no_grad()
    def encode_scaled_anchor(self, anchor_history, velocity_dims, scales, device):
        """Encode one raw anchor obs-history at several velocity scales.

        Uniformly scaling every velocity coordinate of a real gait history is,
        to first order, the same gait traversed faster; positions are left
        untouched so the anchor stays near the data manifold.
        """
        anchor = torch.as_tensor(anchor_history, device=device, dtype=torch.float32)
        scaled = anchor.unsqueeze(0).repeat(len(scales), 1, 1)
        factors = torch.tensor(scales, device=device, dtype=torch.float32)
        scaled[..., -velocity_dims:] *= factors.view(-1, 1, 1)
        _, goals = self.encode_goal(scaled)
        return goals

    def parameter_groups(self):
        world = list(self.world_model.parameters())
        follower = list(self.follower.parameters())
        reward = list(self.reward_predictor.parameters())
        ids = [{id(p) for p in group} for group in (world, follower, reward)]
        if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("world, follower, and reward parameters must be disjoint")
        return world, follower, reward


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

    def sample_transitions(self, batch_size, history, rng, reward_window=0):
        accepted = []
        while sum(len(x[0]) for x in accepted) < batch_size:
            env, end = self._sample_base(batch_size * 2, history, 1, rng)
            valid = self._valid_end(env, end) & self._valid_end(env, end + 1)
            valid &= ~self._gather(self.done, env, end[:, None]).reshape(-1)
            accepted.append((env[valid], end[valid]))
        env = np.concatenate([x[0] for x in accepted])[:batch_size]
        end = np.concatenate([x[1] for x in accepted])[:batch_size]
        slot = end % self.capacity
        batch = {
            "obs": self._padded_history(self.obs, env, end, history),
            "incoming": self._padded_history(self.incoming_action, env, end, history, zero_pad=True),
            "next_obs": self._padded_history(self.obs, env, end + 1, history),
            "next_incoming": self._padded_history(
                self.incoming_action, env, end + 1, history, zero_pad=True
            ),
            "action": self.action[env, slot],
            "reward": self.reward[env, slot],
        }
        if reward_window > 0:
            # Sustained rate over the realized future window; samples whose
            # window leaves the episode or the buffer are masked out of the
            # rate loss instead of receiving a biased label. Note: on envs
            # with failure terminations (Hopper/Walker) masking hides the
            # pre-fall regime from the rate head, which then overestimates
            # unstable fast gaits; revisit before trusting the alpha sweep
            # off HalfCheetah.
            window_indices = end[:, None] + np.arange(reward_window)[None]
            clipped = np.minimum(window_indices, self.total_steps - 1)
            rewards = self._gather(self.reward, env, clipped)
            valid = (window_indices < self.total_steps) & (
                self._gather(self.absolute, env, clipped) == window_indices
            )
            valid &= (
                self._gather(self.episode, env, clipped)
                == self._gather(self.episode, env, end[:, None])
            )
            full = valid.all(axis=1)
            batch["reward_rate"] = np.where(
                full, (rewards * valid).sum(axis=1) / reward_window, 0.0
            ).astype(np.float32)
            batch["reward_rate_valid"] = full.astype(np.float32)
        return batch

    def best_windowed_anchor(self, window, history):
        """Best sustained-rate state whose obs history and reward window are
        fully inside one episode. Returns (rate, raw obs history) or None."""
        steps = min(self.total_steps, self.capacity)
        if steps < window + history - 1:
            return None
        best_rate = -np.inf
        best_history = None
        for env in range(self.num_envs):
            if self.total_steps > self.capacity:
                start = self.total_steps % self.capacity
                order = np.concatenate(
                    [np.arange(start, self.capacity), np.arange(start)]
                )
            else:
                order = np.arange(steps)
            rewards = self.reward[env, order]
            episodes = self.episode[env, order]
            cumulative = np.concatenate([[0.0], np.cumsum(rewards, dtype=np.float64)])
            t = np.arange(history - 1, steps - window + 1)
            rates = (cumulative[t + window] - cumulative[t]) / window
            # Episode ids are monotone per env, so endpoint equality implies
            # the whole span shares one episode.
            valid = (episodes[t + window - 1] == episodes[t]) & (
                episodes[t] == episodes[t - history + 1]
            )
            if not valid.any():
                continue
            rates = np.where(valid, rates, -np.inf)
            index = int(np.argmax(rates))
            if rates[index] > best_rate:
                best_rate = float(rates[index])
                end = t[index]
                best_history = self.obs[env, order[end - history + 1 : end + 1]].copy()
        if best_history is None:
            return None
        return best_rate, best_history

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


# Fixed aspiration sweep for goal-semantics diagnostics: predicted sustained
# rate should rise monotonically with the velocity scale while the encoder
# extrapolation holds; where it breaks marks the usable aspiration range.
ALPHA_SWEEP = (1.0, 1.25, 1.5, 2.0, 3.0)


def train_world_step(agent, optimizer, batch, args, device):
    obs = as_tensor(batch["obs"], device)
    incoming = as_tensor(batch["incoming"], device)
    next_obs = as_tensor(batch["next_obs"], device)
    action = as_tensor(batch["action"], device)
    outputs = agent.world_model(obs, incoming)
    tokens, frame, goal, stats, reconstruction, _, memory = outputs
    prediction = agent.world_model.transition(memory, action)
    # Pure LeJEPA: the prediction target is the online encoder, attached, so
    # gradient flows through both branches; SIGReg alone prevents collapse.
    target_tokens, target_goal = agent.encode_goal(next_obs)
    prediction_loss = F.mse_loss(prediction, target_tokens[:, -1])
    reconstruction_loss = F.mse_loss(reconstruction, stats.detach())
    overlap_loss = F.mse_loss(goal, target_goal)
    frame_samples = frame[:, -1]
    # SIGReg must cover the space the attached prediction target lives in
    # (per-token-position batch distributions), not just the token means —
    # otherwise the unregularized deviation-from-mean subspace can collapse
    # to constants under the attached MSE. transpose puts the batch on the
    # sample axis for each of the token positions independently.
    sigreg_loss = (
        agent.world_model.sigreg(frame_samples)
        + agent.world_model.sigreg(goal)
        + agent.world_model.sigreg(target_tokens[:, -1].transpose(0, 1))
    )
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
        all_tokens, all_goals = agent.encode_goal(
            torch.cat(
                [obs, next_obs, future_obs, occupancy_future_obs],
                dim=0,
            )
        )
        current_goal, next_goal, future_goal, occupancy_future_goal = all_goals.chunk(4)
        # Frame summaries: the successor/inverse intermediate space. The last
        # frame of the obs history is f_t; of the next-obs history, f_{t+1}.
        all_frames = all_tokens.mean(dim=-2)[:, -1]
        current_frame, next_frame, _, _ = all_frames.chunk(4)
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
    # Desired successor: the hindsight goal is a latent actually reached
    # later; the supervised target is the frame summary that actually
    # followed while en route. Conditioning on the current belief makes this
    # a feedback law: wherever the agent is, S re-plots the one-step target.
    successor_prediction = agent.follower.desired_successor(
        belief, current_frame, current_goal, future_goal
    )
    successor_loss = F.mse_loss(successor_prediction, next_frame)
    # Inverse dynamics: dense and goal-independent — every transition is a
    # label, no hindsight sparsity, and the one-step frame delta carries
    # full-strength action information. The action problem stays single-step.
    inverse_prediction = agent.follower.inverse_action(belief, current_frame, next_frame)
    inverse_loss = F.mse_loss(inverse_prediction, action)
    # Composed act-time path on the same hindsight tuples, ATTACHED through
    # the successor. v19 detached S here and its goal channel died
    # (goal_removal_action_mse 0.001 vs 0.05+ in v16/v17): S's frame-MSE
    # alone is minimized by the near-marginal next frame, so the composed
    # gradient is the only first-order path teaching S the goal-dependent
    # part of the prediction that control actually needs. The frame-MSE
    # (coef 1.0) and I's true-frame training keep both heads grounded.
    composed_prediction = agent.follower.inverse_action(
        belief, current_frame, successor_prediction
    )
    composed_loss = F.mse_loss(composed_prediction, action)
    occupancy_logits = agent.follower.occupancy_logits(
        occupancy_belief, occupancy_future_goal
    )
    occupancy_loss = F.cross_entropy(
        occupancy_logits, torch.arange(action.shape[0], device=device)
    )
    loss = (
        args.successor_coef * successor_loss
        + args.inverse_coef * inverse_loss
        + args.composed_coef * composed_loss
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
        "follower/successor_mse": successor_loss.item(),
        # Identity baseline: S must beat "predict no frame change" or it has
        # learned nothing beyond persistence.
        "follower/successor_identity_mse": F.mse_loss(current_frame, next_frame).item(),
        "follower/inverse_action_mse": inverse_loss.item(),
        "follower/composed_action_mse": composed_loss.item(),
        "follower/occupancy_loss": occupancy_loss.item(),
        "follower/achieved_progress": achieved_progress.mean().item(),
        "follower/achieved_progress_positive_fraction": (achieved_progress > 0).float().mean().item(),
        "follower/grad_norm": float(grad_norm),
    }


def train_reward_step(agent, reward_optimizer, batch, args, device):
    """Supervised windowed reward-rate regression on real target-encoded goal
    latents. The head is a diagnostic probe of goal semantics; nothing in the
    world model, follower, or proposer differentiates through it."""
    obs = as_tensor(batch["obs"], device)
    rate = as_tensor(batch["reward_rate"], device)
    rate_valid = as_tensor(batch["reward_rate_valid"], device)
    with torch.no_grad():
        _, goal_latent = agent.encode_goal(obs)
    reward_prediction = agent.reward_predictor(goal_latent)
    per_sample = (reward_prediction - rate).square() * rate_valid
    normalizer = rate_valid.sum().clamp_min(1.0)
    reward_loss = per_sample.sum() / normalizer
    reward_optimizer.zero_grad(set_to_none=True)
    (args.reward_prediction_coef * reward_loss).backward()
    reward_grad_norm = nn.utils.clip_grad_norm_(
        agent.reward_predictor.parameters(), args.max_grad_norm
    )
    reward_optimizer.step()
    absolute_error = ((reward_prediction - rate).abs() * rate_valid).sum() / normalizer
    return {
        "goal/reward_rate_mse": reward_loss.item(),
        "goal/reward_rate_mae": absolute_error.item(),
        "goal/reward_grad_norm": float(reward_grad_norm),
        "goal/reward_rate_mean": ((rate * rate_valid).sum() / normalizer).item(),
        "goal/reward_rate_valid_fraction": rate_valid.mean().item(),
    }


def mean_metrics(metrics):
    result = {}
    for row in metrics:
        for key, value in row.items():
            result.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in result.items()}


@torch.no_grad()
def evaluate_direct_policy(agent, args, device, run_name, commanded_goal):
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
            commanded_goal,
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
    if args.velocity_aspiration_delta <= 0:
        raise ValueError("velocity aspiration delta must be positive")
    if args.velocity_scale_cap < 1.0:
        raise ValueError("velocity scale cap must be at least 1")
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
    world_parameters, follower_parameters, _ = agent.parameter_groups()
    world_optimizer = torch.optim.AdamW(
        world_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    follower_optimizer = torch.optim.AdamW(
        follower_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
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
    anchor_rate = -np.inf
    anchor_history = None
    anchor_age = 0
    base_goal = None
    commanded_goal = torch.zeros(args.latent_dim, device=device)
    previous_commanded_goal = None
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
                policy_action, current_goal, step_goal = agent.act(
                    obs_tensor, incoming_tensor, commanded_goal
                )
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
                if (~episode_explorer).any():
                    policy_action_abs.append(np.abs(policy_action[~episode_explorer]).mean())

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
                # The rate head receives one fresh optimization on every
                # follower cycle, never an occasional outer-loop update.
                rate_batch = replay.sample_transitions(
                    args.batch_size, args.history, rng, args.anchor_reward_window
                )
                metrics.append(
                    train_reward_step(agent, reward_optimizer, rate_batch, args, device)
                )

        aggregated = mean_metrics(metrics)
        for key, value in aggregated.items():
            writer.add_scalar(key, value, global_step)
        # Refresh the anchor and re-encode both goals AFTER the training block:
        # no parameters update until the next block, so these latents are
        # exact for the upcoming rollout, the diagnostics below, and eval
        # alike (no encoder-snapshot mixing).
        anchor_result = replay.best_windowed_anchor(
            args.anchor_reward_window, args.history
        )
        anchor_updated = False
        if anchor_result is not None:
            # Persistent gap between the buffer's current best and the
            # ratcheted anchor rate means the anchor is not reproducible,
            # i.e. the curriculum is locked on a fluke window.
            writer.add_scalar(
                "goal/anchor_buffer_best_rate", anchor_result[0], global_step
            )
            if anchor_result[0] > anchor_rate:
                anchor_rate, anchor_history = anchor_result
                anchor_updated = True
                anchor_age = 0
            else:
                anchor_age += 1
        if anchor_history is not None:
            # Additive aspiration: demand a fixed forward-velocity increment
            # over the anchor, expressed as a uniform velocity scale so the
            # whole gait is time-compressed coherently. Near-stationary
            # anchors hit the cap (bounded but real pressure); fast anchors
            # get a shrinking multiplicative margin (delta/v_fwd).
            anchor_forward = float(anchor_history[:, -velocity_dims].mean())
            if anchor_forward > 0:
                commanded_scale = min(
                    (anchor_forward + args.velocity_aspiration_delta) / anchor_forward,
                    args.velocity_scale_cap,
                )
            else:
                # Uniform scaling preserves gait direction, so amplifying a
                # stationary/backwards anchor is wrong-direction pressure;
                # command it unscaled until a forward anchor appears.
                commanded_scale = 1.0
            anchor_goals = agent.encode_scaled_anchor(
                anchor_history, velocity_dims, (1.0, commanded_scale), device
            )
            base_goal, commanded_goal = anchor_goals[0], anchor_goals[1]
            writer.add_scalar("goal/anchor_updated", float(anchor_updated), global_step)
            writer.add_scalar("goal/anchor_age_iterations", float(anchor_age), global_step)
            writer.add_scalar("goal/anchor_forward_velocity", anchor_forward, global_step)
            writer.add_scalar("goal/commanded_velocity_scale", commanded_scale, global_step)
        if anchor_history is not None:
            with torch.no_grad():
                sweep_goals = agent.encode_scaled_anchor(
                    anchor_history, velocity_dims, ALPHA_SWEEP, device
                )
                sweep_rates = agent.reward_predictor(sweep_goals)
                commanded_rate = agent.reward_predictor(commanded_goal.unsqueeze(0))
            for scale, value in zip(ALPHA_SWEEP, sweep_rates.tolist()):
                writer.add_scalar(
                    f"diagnostics/anchor_predicted_rate_alpha_{str(scale).replace('.', '_')}",
                    value,
                    global_step,
                )
            writer.add_scalar("goal/anchor_rate", anchor_rate, global_step)
            writer.add_scalar(
                "goal/commanded_predicted_rate", commanded_rate.item(), global_step
            )
            writer.add_scalar(
                "goal/aspiration_latent_distance",
                (commanded_goal - base_goal).norm().item(),
                global_step,
            )
            writer.add_scalar(
                "goal/commanded_goal_norm", commanded_goal.norm().item(), global_step
            )
            if previous_commanded_goal is not None:
                writer.add_scalar(
                    "goal/revision_mse",
                    (commanded_goal - previous_commanded_goal).square().mean().item(),
                    global_step,
                )
            previous_commanded_goal = commanded_goal.clone()
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
        if global_step >= args.random_warmup_steps and anchor_history is not None:
            with torch.no_grad():
                diagnostic_obs = as_tensor(obs_history, device)
                diagnostic_incoming = as_tensor(incoming_history, device)
                diagnostic_belief, diagnostic_frame, diagnostic_current_goal, _, _ = (
                    agent.follower_state(diagnostic_obs, diagnostic_incoming)
                )
                commanded = commanded_goal.unsqueeze(0).expand(args.num_envs, -1)
                base = base_goal.unsqueeze(0).expand(args.num_envs, -1)
                goal_action = agent.follower(
                    diagnostic_belief, diagnostic_frame, diagnostic_current_goal, commanded
                )
                base_action = agent.follower(
                    diagnostic_belief, diagnostic_frame, diagnostic_current_goal, base
                )
                null_action = agent.follower(
                    diagnostic_belief,
                    diagnostic_frame,
                    diagnostic_current_goal,
                    diagnostic_current_goal,
                )
                writer.add_scalar(
                    "diagnostics/goal_action_power",
                    goal_action.square().mean().item(),
                    global_step,
                )
                # Goal-removal test: the MLP head does not vanish at
                # goal_features(g, g) = 0, so compare against the explicit
                # null-goal action instead of raw action power.
                writer.add_scalar(
                    "diagnostics/goal_removal_action_mse",
                    (goal_action - null_action).square().mean().item(),
                    global_step,
                )
                # Sensitivity of actions to aspiration alone (alpha vs 1.0):
                # zero here means the follower cannot express the extrapolated
                # demand and the aspiration channel is dead.
                writer.add_scalar(
                    "diagnostics/aspiration_action_mse",
                    (goal_action - base_action).square().mean().item(),
                    global_step,
                )
                # Cosine of the two servo directions: ~1 while distances stay
                # apart means the encoder contracted the scaled anchor back
                # onto the base direction (aspiration adds nothing).
                commanded_features = agent.follower.goal_features(
                    diagnostic_current_goal, commanded
                )
                base_features = agent.follower.goal_features(
                    diagnostic_current_goal, base
                )
                writer.add_scalar(
                    "diagnostics/aspiration_direction_cosine",
                    F.cosine_similarity(commanded_features, base_features, dim=-1)
                    .mean()
                    .item(),
                    global_step,
                )
        if args.eval_interval > 0 and global_step >= next_eval_step and global_step >= args.random_warmup_steps:
            evaluation_returns = evaluate_direct_policy(
                agent, args, device, run_name, commanded_goal
            )
            writer.add_scalar(
                "evaluation/direct_episodic_return",
                float(np.mean(evaluation_returns)),
                global_step,
            )
            # A/B the aspiration itself: the same policy commanded the
            # unscaled anchor. The commanded-minus-base gap is the direct
            # evidence for or against the velocity-scaling hypothesis.
            if base_goal is not None:
                base_returns = evaluate_direct_policy(
                    agent, args, device, run_name, base_goal
                )
                writer.add_scalar(
                    "evaluation/base_goal_episodic_return",
                    float(np.mean(base_returns)),
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

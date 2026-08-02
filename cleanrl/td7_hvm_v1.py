# TD7 + Hindsight Value Modelling (HVM_v1).
#
# Base: td7_continuous_action.py (Fujimoto et al. 2023, TD7). This file keeps TD7's actor,
# encoder (SALE), LAP buffer, target-clipping ratchet, hard updates, episodic checkpointing,
# num_envs=1 loop and eval protocol byte-identical. The ONLY additions are a value-driven
# hindsight pathway adapted from Guez et al. 2020 ("Value-driven hindsight modelling").
#
# Idea. TD7's twin critics regress a noisy 1-step target; future stochasticity they cannot see
# from (s,a) shows up as irreducible target variance. We give the *value function* privileged
# access to a compressed description of the realized future, without biasing it:
#   - phi_t: a fixed, hand-pooled summary of the ACTUAL future (pooled future actions a_{t+1..t+H}
#     and pooled future states s_{t+1..t+H}). This is privileged (uses information unavailable at
#     decision time), so it may NEVER touch the twins directly.
#   - GBottleneck compresses phi -> g (8-dim). A 5-step hindsight value head H(s,a,zs,g) fits a
#     5-step bootstrapped return; because g is squeezed through 8 dims, the bottleneck is forced to
#     keep only the future features that actually move the value (value-driven, not reconstruction).
#   - GPredictor estimates g from (s,a) alone: g_hat = p(s,a,zs). This is (s,a)-measurable.
#   - The twins consume g_hat (never g/phi) as an extra input. So every input the twins ever see is
#     a deterministic function of (s,a): zero leakage, zero target bias. The privilege only shapes
#     WHICH latent features the predictor learns to surface, giving the critic a denoised, forward-
#     looking feature to condition on.
#
# Why privilege belongs in the critic (not the actor): the critic is trained toward a target, so a
# feature that "explains away" future randomness reduces target variance and speeds value fitting.
# The actor reads the same critic through g_hat treated as a constant feature (DPG gradient does not
# flow through the predictor input), so the deterministic policy, exploration and checkpointing stay
# exactly TD7.
#
# Hypothesis: denoising the twin-critic target via an (s,a)-measurable predicted future-feature
# accelerates and stabilizes value learning on MuJoCo control, improving sample efficiency and final
# return over TD7 without changing the policy class.
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

HORIZON = 20      # phi pools the next H executed transitions (windows never cross an episode)
NEAR_HORIZON = 5  # near window = t+1..t+5, far window = t+6..t+H
NSTEP = 5         # hindsight value head fits an NSTEP bootstrapped return


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
    g_dim: int = 8
    """dimensionality of the hindsight bottleneck feature g (and the predicted g_hat)"""
    encoder_lr: float = 3e-4
    """the learning rate of the encoder optimizer"""
    critic_lr: float = 3e-4
    """the learning rate of the critic optimizer (reused for the hindsight/predictor optimizers)"""
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


def _pool_meanstd(window, dim):
    """near/far mean+std pooling of `window` (shape (m, dim), m may be 0): [near_mean, near_std,
    far_mean, far_std] (4*dim). Population std (ddof=0); zeros for an empty sub-window."""
    parts = []
    for w in (window[:NEAR_HORIZON], window[NEAR_HORIZON:]):
        if len(w) > 0:
            parts.append(w.mean(0))
            parts.append(w.std(0))
        else:
            parts.append(np.zeros(dim, dtype=np.float32))
            parts.append(np.zeros(dim, dtype=np.float32))
    return parts


def pooled_phi(future_actions, future_states, act_dim, state_dim):
    """Privileged summary of the realized future for one transition:
      [ pooled future ACTIONS (4A) , valid_frac (1) , pooled future STATES (4S) ]  -> 4A+1+4S.
    future_actions has shape (ma, A) with ma <= H (possibly 0); future_states (ms, S), ms >= 1
    (a transition always has its own next_state). valid_frac is over the action window only."""
    ma = future_actions.shape[0]
    parts = _pool_meanstd(future_actions, act_dim)
    parts.append(np.array([ma / float(HORIZON)], dtype=np.float32))
    parts += _pool_meanstd(future_states, state_dim)
    return np.concatenate(parts).astype(np.float32)


class PhiQueue:
    """Delay queue building the privileged phi and the NSTEP hindsight quantities at insert time
    from a single env's stream. A transition's phi pools the NEXT H executed actions/states; the
    NSTEP quantities look up to NSTEP transitions ahead. Windows never cross an episode boundary:
    a record releases once its full H-window exists, and at a boundary ALL pending records flush
    with partial windows (this happens before the main loop's training burst, so nothing is lost).

    Alignment (verified against td7_hop_v1's future-actions pattern):
      fut_a[i] = a_i  (record i's OWN action) -> record i's future-action window starts at i+1.
      fut_s[i] = s_{i+1} (record i's OWN next_state) -> record i's future-STATE window starts at i
                 (fut_s[i] is already the first future state s_{i+1}).
      fut_r[i] = r_i, fut_d[i] = TD7 done-quirk of record i (1 iff it terminated and not truncated).
    """

    def __init__(self, horizon, act_dim, state_dim, gamma):
        self.h = horizon
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.pending = []  # [state, action, next_state, reward, done, n_future_seen]
        self.fut_a = []    # fut_a[i] = record i's own action
        self.fut_s = []    # fut_s[i] = record i's own next_state (= s_{i+1})
        self.fut_r = []    # fut_r[i] = record i's own reward
        self.fut_d = []    # fut_d[i] = record i's own done-quirk

    def _emit(self, i):
        s, a, ns, r, d, _ = self.pending[i]
        fut_actions = np.array(self.fut_a[i + 1 : i + 1 + self.h], dtype=np.float32).reshape(-1, self.act_dim)
        fut_states = np.array(self.fut_s[i : i + self.h], dtype=np.float32).reshape(-1, self.state_dim)
        phi = pooled_phi(fut_actions, fut_states, self.act_dim, self.state_dim)

        # NSTEP bootstrapped-return ingredients. m = steps available before the boundary.
        m = min(NSTEP, len(self.pending) - i)
        r5 = 0.0
        disc = 1.0
        for j in range(m):
            r5 += disc * self.fut_r[i + j]
            disc *= self.gamma
        s5 = np.array(self.fut_s[i + m - 1], dtype=np.float32)  # state reached after m steps (s_{t+m})
        disc5 = (self.gamma ** m) * (1.0 - self.fut_d[i + m - 1])  # 0 iff the last included step terminated
        return (s, a, ns, float(r), float(d), phi, s5, np.float32(r5), np.float32(disc5))

    def push(self, state, action, next_state, reward, done, boundary):
        # `action` is the executed env action already normalized to [-1, 1]; it is a future action
        # for every already-pending record and record i's own action for the new record.
        for rec in self.pending:
            rec[5] += 1
        self.pending.append(
            [np.array(state, copy=True), np.array(action, copy=True), np.array(next_state, copy=True),
             float(reward), float(done), 0]
        )
        self.fut_a.append(np.array(action, copy=True))
        self.fut_s.append(np.array(next_state, copy=True))
        self.fut_r.append(float(reward))
        self.fut_d.append(float(done))

        out = []
        if boundary:
            for i in range(len(self.pending)):
                out.append(self._emit(i))
            self.pending.clear()
            self.fut_a.clear()
            self.fut_s.clear()
            self.fut_r.clear()
            self.fut_d.clear()
        elif self.pending[0][5] >= self.h:
            out.append(self._emit(0))
            self.pending.pop(0)
            self.fut_a.pop(0)
            self.fut_s.pop(0)
            self.fut_r.pop(0)
            self.fut_d.pop(0)
        return out


class LAPBuffer:
    """TD7's LAP replay buffer + the hindsight columns (phi, s5, r5, disc5). The LAP priority
    machinery is byte-identical to base TD7. Actions arrive already normalized (the caller divides
    by max_action once, since the same normalized action also feeds phi pooling), so add() does not
    re-normalize."""

    def __init__(self, state_dim, action_dim, phi_dim, device, max_size, batch_size):
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
        self.phi = np.zeros((self.max_size, phi_dim), dtype=np.float32)
        self.s5 = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.r5 = np.zeros((self.max_size, 1), dtype=np.float32)
        self.disc5 = np.zeros((self.max_size, 1), dtype=np.float32)

        self.priority = torch.zeros(self.max_size, device=device)
        self.max_priority = 1.0

    def add(self, state, action, next_state, reward, done, phi, s5, r5, disc5):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.phi[self.ptr] = phi
        self.s5[self.ptr] = s5
        self.r5[self.ptr] = r5
        self.disc5[self.ptr] = disc5

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
            torch.tensor(self.phi[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.s5[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.r5[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.disc5[self.ind], dtype=torch.float, device=self.device),
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = avg_l1_norm(self.l0(state))
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
    """TD7's twin critic, with the (s,a)-measurable predicted hindsight feature g_hat (g_dim)
    appended to the embeddings concat of BOTH twins. g_hat is the only structural change vs base."""

    def __init__(self, state_dim, action_dim, zs_dim=256, g_dim=8, hdim=256):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim + g_dim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim + g_dim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs, g_hat):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs, g_hat], 1)

        q1 = avg_l1_norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        q2 = avg_l1_norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)


class GBottleneck(nn.Module):
    """Compresses the privileged summary phi to an 8-dim value-driven feature g. AvgL1Norm on the
    output keeps g's scale stable as a downstream input feature."""

    def __init__(self, phi_dim, g_dim=8, hdim=128):
        super().__init__()
        self.l1 = nn.Linear(phi_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, g_dim)

    def forward(self, phi):
        g = F.elu(self.l1(phi))
        g = F.elu(self.l2(g))
        return avg_l1_norm(self.l3(g))


class HindsightValue(nn.Module):
    """TD7-critic-style single head fitting the NSTEP bootstrapped return, conditioned on the
    hindsight bottleneck feature g. Its only role is to give GBottleneck a value-driven training
    signal; it is never used at decision time."""

    def __init__(self, state_dim, action_dim, zs_dim=256, g_dim=8, hdim=256):
        super().__init__()
        self.h0 = nn.Linear(state_dim + action_dim, hdim)
        self.h1 = nn.Linear(hdim + zs_dim + g_dim, hdim)
        self.h2 = nn.Linear(hdim, hdim)
        self.h3 = nn.Linear(hdim, 1)

    def forward(self, state, action, zs, g):
        h = avg_l1_norm(self.h0(torch.cat([state, action], 1)))
        h = torch.cat([h, zs, g], 1)
        h = F.elu(self.h1(h))
        h = F.elu(self.h2(h))
        return self.h3(h)


class GPredictor(nn.Module):
    """Estimates the bottleneck feature g from (s,a) alone: g_hat = p(s,a,zs). Lives in the same
    AvgL1Norm space as g so the critic sees a consistent feature at train and prediction time."""

    def __init__(self, state_dim, action_dim, zs_dim=256, g_dim=8, hdim=256):
        super().__init__()
        self.p0 = nn.Linear(state_dim + action_dim, hdim)
        self.p1 = nn.Linear(hdim + zs_dim, hdim)
        self.p2 = nn.Linear(hdim, hdim)
        self.p3 = nn.Linear(hdim, g_dim)

    def forward(self, state, action, zs):
        p = avg_l1_norm(self.p0(torch.cat([state, action], 1)))
        p = torch.cat([p, zs], 1)
        p = F.elu(self.p1(p))
        p = F.elu(self.p2(p))
        return avg_l1_norm(self.p3(p))


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_action, phi_dim, args: Args, device, writer: SummaryWriter):
        self.args = args
        self.device = device
        self.writer = writer

        self.actor = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, args.zs_dim, args.g_dim, args.hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        # hindsight machinery
        self.g_bottleneck = GBottleneck(phi_dim, args.g_dim).to(device)
        self.hindsight = HindsightValue(state_dim, action_dim, args.zs_dim, args.g_dim, args.hidden_dim).to(device)
        # GBottleneck + H share one optimizer (H's loss is the only training signal for the bottleneck)
        self.hindsight_optimizer = torch.optim.Adam(
            list(self.g_bottleneck.parameters()) + list(self.hindsight.parameters()), lr=args.critic_lr
        )
        self.g_predictor = GPredictor(state_dim, action_dim, args.zs_dim, args.g_dim, args.hidden_dim).to(device)
        self.g_predictor_optimizer = torch.optim.Adam(self.g_predictor.parameters(), lr=args.critic_lr)
        self.g_predictor_target = copy.deepcopy(self.g_predictor)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = LAPBuffer(state_dim, action_dim, phi_dim, device, args.buffer_size, args.batch_size)

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

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs)

            if use_exploration:
                action = action + torch.randn_like(action) * self.args.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def train(self):
        self.training_steps += 1

        state, action, next_state, reward, not_done, phi, s5, r5, disc5 = self.replay_buffer.sample()

        # update encoder: predict the next state's embedding from (zs, action)
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # update critic (twins now condition on the predicted hindsight feature g_hat)
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.args.target_policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            g_hat_next = self.g_predictor_target(next_state, next_action, fixed_target_zs)
            Q_target = self.critic_target(
                next_state, next_action, fixed_target_zsa, fixed_target_zs, g_hat_next
            ).min(1, keepdim=True)[0]
            Q_target = reward + not_done * self.args.gamma * Q_target.clamp(self.min_target, self.max_target)

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)
            g_hat = self.g_predictor(state, action, fixed_zs)

        Q = self.critic(state, action, fixed_zsa, fixed_zs, g_hat)
        td_loss = (Q - Q_target).abs()
        critic_loss = lap_huber(td_loss, self.args.min_priority)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update LAP priorities
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.lap_alpha).detach()
        self.replay_buffer.update_priority(priority)

        # hindsight update: fit the NSTEP bootstrapped return with a bottleneck-compressed view of
        # the REALIZED future (phi). Trains GBottleneck + H jointly.
        with torch.no_grad():
            zs5 = self.fixed_encoder_target.zs(s5)
            a5 = self.actor_target(s5, zs5)  # no target-smoothing noise on the NSTEP anchor
            zsa5 = self.fixed_encoder_target.zsa(zs5, a5)
            g_hat5 = self.g_predictor_target(s5, a5, zs5)
            Q5 = self.critic_target(s5, a5, zsa5, zs5, g_hat5).min(1, keepdim=True)[0].clamp(
                self.min_target, self.max_target
            )
            target5 = r5 + disc5 * Q5

        g = self.g_bottleneck(phi)
        h = self.hindsight(state, action, fixed_zs, g)
        h_loss = F.mse_loss(h, target5)

        self.hindsight_optimizer.zero_grad()
        h_loss.backward()
        self.hindsight_optimizer.step()

        # predictor update: regress the (s,a)-measurable g_hat onto the (detached) bottleneck g
        g_pred = self.g_predictor(state, action, fixed_zs)
        p_loss = F.mse_loss(g_pred, g.detach())

        self.g_predictor_optimizer.zero_grad()
        p_loss.backward()
        self.g_predictor_optimizer.step()

        # update actor (delayed); g_hat is a constant feature (no DPG gradient through the predictor)
        if self.training_steps % self.args.policy_freq == 0:
            actor_action = self.actor(state, fixed_zs)
            actor_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_action)
            with torch.no_grad():
                g_hat_a = self.g_predictor(state, actor_action.detach(), fixed_zs)
            actor_Q = self.critic(state, actor_action, actor_fixed_zsa, fixed_zs, g_hat_a)
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
            self.g_predictor_target.load_state_dict(self.g_predictor.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        # losses are logged against training_steps: with checkpointing, training happens in
        # episode-boundary bursts, so global_step would clump thousands of points at one x value
        if self.training_steps % 500 == 0:
            self.writer.add_scalar("losses/encoder_loss", encoder_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/critic_loss", critic_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/q_values", Q.mean().item(), self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target, self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target, self.training_steps)
            self.writer.add_scalar("debug/h_loss", h_loss.item(), self.training_steps)
            self.writer.add_scalar("debug/p_loss", p_loss.item(), self.training_steps)
            self.writer.add_scalar(
                "debug/h_minus_q", (h.mean() - Q.min(1)[0].mean()).item(), self.training_steps
            )
            self.writer.add_scalar("debug/g_norm", g.abs().mean().item(), self.training_steps)

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


def evaluate(agent: TD7Agent, eval_env, eval_eps, use_checkpoint):
    returns = np.zeros(eval_eps)
    for ep in range(eval_eps):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(np.array(state), use_checkpoint=use_checkpoint, use_exploration=False)
            state, reward, terminated, truncated, _ = eval_env.step(action)
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
    phi_dim = 4 * action_dim + 1 + 4 * state_dim

    agent = TD7Agent(state_dim, action_dim, max_action, phi_dim, args, device, writer)
    phi_queue = PhiQueue(HORIZON, action_dim, state_dim, args.gamma)

    start_time = time.time()
    allow_train = False

    obs, _ = envs.reset(seed=args.seed)
    eval_seeded = False
    # total_timesteps + 1 so the final evaluation at exactly total_timesteps fires (as in the original)
    for global_step in range(args.total_timesteps + 1):
        if global_step % args.eval_freq == 0:
            if not eval_seeded:
                eval_env.reset(seed=args.seed + 100)
                eval_seeded = True
            eval_return = evaluate(agent, eval_env, args.eval_eps, use_checkpoint=args.use_checkpoints)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(f"global_step={global_step}, eval_return={eval_return:.3f}")

        # ALGO LOGIC: put action logic here
        if allow_train:
            actions = agent.select_action(np.array(obs[0]))[None]
        else:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # phi delay queue -> replay buffer; handle `final_observation`. TD7's done quirk kept
        # (bootstrap at the timeout step even on true termination there). The queue holds a
        # transition until its H-step future window exists; ALL pending records flush at a
        # boundary, which precedes the training burst below.
        real_next_obs = next_obs[0]
        if truncations[0] or terminations[0]:
            real_next_obs = infos["final_observation"][0]
        boundary = bool(terminations[0] or truncations[0])
        done = float(terminations[0] and not truncations[0])
        exec_action_norm = actions[0] / max_action
        for rec in phi_queue.push(obs[0], exec_action_norm, real_next_obs, float(rewards[0]), done, boundary):
            agent.replay_buffer.add(*rec)

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

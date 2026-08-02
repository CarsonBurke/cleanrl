# TD7-TAP v1 — teacher-as-policy: the deployed policy IS the hindsight-privileged actor.
# =====================================================================================
# Lesson from the td7_hop line (v1/v2, HalfCheetah eval at matched steps): privilege routed
# through a teacher->KL->student->data chain gave a real but FRONT-LOADED gain (+12% at
# 150k) that decayed to a loss by 300k, at 2.5-4x compute. The chain, not the privilege,
# was the bottleneck; and the machinery wasn't the operator.
# TAP removes every intermediate stage. There is ONE policy: a TD7 deterministic actor
# A(s, phi) conditioned on hindsight phi (pooled stats of the next H=20 executed actions),
# trained by TD7's own DPG operator. Because phi is unknown at act time, a predictor
# phi_hat(s) is trained supervised on replay (phi is stored per transition) and the actor
# is deployed as A(s, phi_hat(s)) + TD7 noise. The DPG update evaluates the actor at BOTH
# the real replayed phi (hindsight conditioning: the gradient can specialize "what to do
# given how the trajectory continues", a credit-assignment shortcut that reduces gradient
# interference across replay eras) and at phi_hat(s) (the act-time input distribution, so
# the deployed composition is trained directly, DAgger-style).
#   - No student, no distillation loss, no KL dual, no Beta policies, no eps-mix: the
#     privileged net is relied on FULLY; privilege flows through shared weights.
#   - As the policy converges, phi|s becomes nearly deterministic, so phi_hat becomes exact
#     and A(s, phi_hat(s)) smoothly degenerates to a TD7 actor with an extra learned input:
#     the design's floor is the baseline, not below it (what killed the hop students).
#   - Compute: one extra small net + 2x actor/critic forward in the delayed actor update
#     (~1.15x TD7 per step).
# Substrate (encoder/critic/LAP/target-clip ratchet/checkpointing/burst loop) byte-preserved.
# HYPOTHESIS: hindsight conditioning helps exactly where TD7's replay hurts — stale eras in
# the buffer prescribe conflicting actions at similar states, and phi disambiguates the era;
# debug/priv_gap = minQ(A(s,phi_real)) - minQ(A(s,phi_hat)) measures the privilege's live
# worth (if it pins near 0 the mechanism is inert and the run should track the baseline).
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

NEAR_HORIZON = 5  # near window = a_{t+1..t+5}; far window = a_{t+6..t+H}


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

    # TD7 substrate (unchanged defaults)
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments (TD7 requires 1)"""
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
    """the frequency of the delayed actor update"""
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
    """the learning rate of the actor (and phi-predictor) optimizers"""

    # --- TAP ---
    hindsight_horizon: int = 20
    """H: future-action window pooled into phi"""
    tap_real_frac: float = 0.5
    """fraction of the DPG loss evaluated at the REAL replayed phi (hindsight conditioning);
    the rest is evaluated at phi_hat(s) (the act-time input distribution)"""


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


def pooled_phi(future_actions, horizon):
    """Pooled hindsight from a window of future executed actions, shape (m, A), m <= H:
    [near_mean, near_std, far_mean, far_std, valid_frac] (4A+1). Population std; zeros for
    an empty window; valid_frac over the full horizon."""
    m, A = future_actions.shape
    parts = []
    for w in (future_actions[:NEAR_HORIZON], future_actions[NEAR_HORIZON:]):
        if len(w) > 0:
            parts.append(w.mean(0))
            parts.append(w.std(0))
        else:
            parts.append(np.zeros(A, dtype=np.float32))
            parts.append(np.zeros(A, dtype=np.float32))
    parts.append(np.array([m / float(horizon)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


class PhiQueue:
    """Delay queue building phi at insert time from a single env's stream. A transition's
    phi pools the NEXT `horizon` executed actions; windows never cross an episode boundary.
    A record is released once its full window exists; at a boundary all pending records
    flush with partial windows (before the training burst fires in the main loop)."""

    def __init__(self, horizon, act_dim):
        self.h = horizon
        self.act_dim = act_dim
        self.pending = []  # [state, action, next_state, reward, done, n_future_seen]
        self.future = []   # future[i] is pending[i]'s own executed action (kept aligned)

    def push(self, state, action, next_state, reward, done, boundary):
        for rec in self.pending:
            rec[5] += 1
        self.future.append(np.array(action, copy=True))
        self.pending.append(
            [np.array(state, copy=True), np.array(action, copy=True),
             np.array(next_state, copy=True), float(reward), float(done), 0]
        )
        out = []
        if boundary:
            for i, (s, a, ns, r, d, m) in enumerate(self.pending):
                fut = np.array(self.future[i + 1 :], dtype=np.float32).reshape(-1, self.act_dim)
                out.append((s, a, ns, r, d, pooled_phi(fut, self.h)))
            self.pending.clear()
            self.future.clear()
        elif self.pending[0][5] >= self.h:
            s, a, ns, r, d, m = self.pending.pop(0)
            fut = np.array(self.future[1 : self.h + 1], dtype=np.float32).reshape(-1, self.act_dim)
            out.append((s, a, ns, r, d, pooled_phi(fut, self.h)))
            self.future.pop(0)
        return out


class LAPBuffer:
    """TD7's LAP buffer + a phi column. sample() additionally returns phi."""

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

        self.priority = torch.zeros(self.max_size, device=device)
        self.max_priority = 1.0

    def add(self, state, action, next_state, reward, done, phi):
        # `action` is already normalized to [-1, 1] (the caller divides by max_action once,
        # since the same normalized action also feeds the phi pooling)
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.phi[self.ptr] = phi

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
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())


class Actor(nn.Module):
    """TD7's actor with phi appended to the state input of l0."""

    def __init__(self, state_dim, phi_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.l0 = nn.Linear(state_dim + phi_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, phi, zs):
        a = avg_l1_norm(self.l0(torch.cat([state, phi], 1)))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class PhiPredictor(nn.Module):
    """Predicts phi (the pooled next-H-actions summary) from the present. Same trunk shape
    as the TD7 actor; linear output (phi components are bounded pooled statistics)."""

    def __init__(self, state_dim, phi_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, phi_dim)

    def forward(self, state, zs):
        p = avg_l1_norm(self.l0(state))
        p = torch.cat([p, zs], 1)
        p = F.relu(self.l1(p))
        p = F.relu(self.l2(p))
        return self.l3(p)


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
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

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


class TD7TapAgent:
    def __init__(self, state_dim, action_dim, max_action, args: Args, device, writer: SummaryWriter):
        self.args = args
        self.device = device
        self.writer = writer
        phi_dim = 4 * int(action_dim) + 1

        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.critic = Critic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor = Actor(state_dim, phi_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.predictor = PhiPredictor(state_dim, phi_dim, args.zs_dim, args.hidden_dim).to(device)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=args.actor_lr)
        self.predictor_target = copy.deepcopy(self.predictor)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_predictor = copy.deepcopy(self.predictor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = LAPBuffer(state_dim, action_dim, phi_dim, device,
                                       args.buffer_size, args.batch_size)

        self.max_action = max_action
        self.training_steps = 0

        # checkpointing tracked values (TD7 machinery, unchanged)
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

        self.tap_stats = {}

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                phi_hat = self.checkpoint_predictor(state, zs)
                action = self.checkpoint_actor(state, phi_hat, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                phi_hat = self.predictor(state, zs)
                action = self.actor(state, phi_hat, zs)
                if use_exploration:
                    action = action + torch.randn_like(action) * self.args.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def train(self):
        self.training_steps += 1

        state, action, next_state, reward, not_done, phi = self.replay_buffer.sample()

        # update encoder (TD7, unchanged)
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # update critic (TD7, unchanged except the target action comes from the composed
        # policy A(s', phi_hat_target(s')))
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.args.target_policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            next_phi_hat = self.predictor_target(next_state, fixed_target_zs)
            next_action = (self.actor_target(next_state, next_phi_hat, fixed_target_zs) + noise).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            Q_target = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs).min(
                1, keepdim=True
            )[0]
            Q_target = reward + not_done * self.args.gamma * Q_target.clamp(self.min_target, self.max_target)

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)
        td_loss = (Q - Q_target).abs()
        critic_loss = lap_huber(td_loss, self.args.min_priority)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update LAP priorities (unchanged)
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.lap_alpha).detach()
        self.replay_buffer.update_priority(priority)

        # update phi predictor (every step; supervised, low-variance target)
        phi_pred = self.predictor(state, fixed_zs)
        predictor_loss = F.mse_loss(phi_pred, phi)
        self.predictor_optimizer.zero_grad()
        predictor_loss.backward()
        self.predictor_optimizer.step()

        # update actor (delayed): DPG at the real replayed phi (hindsight conditioning) and
        # at phi_hat(s) (the act-time composition), mixed by tap_real_frac
        if self.training_steps % self.args.policy_freq == 0:
            phi_hat = self.predictor(state, fixed_zs).detach()

            a_hat = self.actor(state, phi_hat, fixed_zs)
            zsa_hat = self.fixed_encoder.zsa(fixed_zs, a_hat)
            Q_hat = self.critic(state, a_hat, zsa_hat, fixed_zs)

            a_real = self.actor(state, phi, fixed_zs)
            zsa_real = self.fixed_encoder.zsa(fixed_zs, a_real)
            Q_real = self.critic(state, a_real, zsa_real, fixed_zs)

            f = self.args.tap_real_frac
            actor_loss = -(1.0 - f) * Q_hat.mean() - f * Q_real.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            with torch.no_grad():
                self.tap_stats = {
                    "priv_gap": float(Q_real.min(1)[0].mean() - Q_hat.min(1)[0].mean()),
                    "phi_pred_mse": float(predictor_loss),
                    "actor_loss": float(actor_loss),
                }

        # hard target/fixed-encoder updates + target clip range snapshot (TD7 + predictor)
        if self.training_steps % self.args.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.predictor_target.load_state_dict(self.predictor.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        if self.training_steps % 500 == 0:
            self.writer.add_scalar("losses/encoder_loss", encoder_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/critic_loss", critic_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/q_values", Q.mean().item(), self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target, self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target, self.training_steps)
            for tag, val in self.tap_stats.items():
                self.writer.add_scalar(f"debug/{tag}", val, self.training_steps)

    # ---------- TD7 checkpointing machinery (unchanged; checkpoints actor+predictor) ----------
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        if self.min_return < self.best_min_return:
            self.train_and_reset()

        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_predictor.load_state_dict(self.predictor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset()

    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.args.steps_before_checkpointing:
                self.best_min_return *= self.args.reset_weight
                self.max_eps_before_update = self.args.max_eps_when_checkpointing

            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8


def evaluate(agent: TD7TapAgent, eval_env, eval_eps, use_checkpoint):
    returns = np.zeros(eval_eps)
    for ep in range(eval_eps):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(
                np.array(state), use_checkpoint=use_checkpoint, use_exploration=False
            )
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

    agent = TD7TapAgent(state_dim, action_dim, max_action, args, device, writer)
    phi_queue = PhiQueue(args.hindsight_horizon, action_dim)

    start_time = time.time()
    allow_train = False

    obs, _ = envs.reset(seed=args.seed)
    eval_seeded = False
    # total_timesteps + 1 so the final evaluation at exactly total_timesteps fires (as in TD7)
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

        # phi delay queue -> replay buffer; TD7's done quirk kept. All pending records flush
        # at a boundary, which precedes the training burst below.
        real_next_obs = next_obs[0]
        if truncations[0] or terminations[0]:
            real_next_obs = infos["final_observation"][0]
        boundary = bool(terminations[0] or truncations[0])
        done = float(terminations[0] and not truncations[0])
        for s, a, ns, r, d, phi in phi_queue.push(
            obs[0], actions[0] / max_action, real_next_obs, float(rewards[0]), done, boundary
        ):
            agent.replay_buffer.add(s, a, ns, r, d, phi)

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
                "encoder": agent.encoder.state_dict(),
                "critic": agent.critic.state_dict(),
                "actor": agent.actor.state_dict(),
                "predictor": agent.predictor.state_dict(),
                "checkpoint_actor": agent.checkpoint_actor.state_dict(),
                "checkpoint_predictor": agent.checkpoint_predictor.state_dict(),
                "checkpoint_encoder": agent.checkpoint_encoder.state_dict(),
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    envs.close()
    eval_env.close()
    writer.close()

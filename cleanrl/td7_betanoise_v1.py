# TD7: single-file CleanRL-style port of the author implementation (../TD7, Fujimoto et al. 2023,
# "For SALE: State-Action Representation Learning for Deep Reinforcement Learning", https://arxiv.org/abs/2306.02451).
#
# TD7 = TD3 + 4 additions:
#   1. SALE: an encoder learns state embeddings zs and state-action embeddings zsa by predicting
#      the next state's embedding (dynamics-grounded representation). Actor and critic condition on
#      *frozen* snapshots of these embeddings (fixed_encoder / fixed_encoder_target), decoupling
#      representation drift from value learning. AvgL1Norm keeps embedding scale stable.
#   2. LAP: prioritized replay with priority = max TD error^alpha and a matching Huber loss,
#      which cancels the prioritization bias without importance weights.
#   3. Policy checkpointing: training pauses during data collection; at episode boundaries the
#      agent trains in a burst, and the best-worst-case policy (by min episodic return over a
#      window) is checkpointed and used for evaluation.
#   4. Target value clipping: TD targets are clamped to the running [min, max] of past targets,
#      snapshotted at each hard target-network update (every 250 steps).
#
# Faithfulness notes vs ../TD7: online mode only (offline D4RL/TD3+BC branch omitted); ported from
# gym to gymnasium (terminations/truncations replace the _max_episode_steps timeout check);
# hyperparameters are the author defaults. Requires num_envs=1: TD7 assumes a strict 1 gradient
# step per env step and episode-boundary checkpointing, both of which break with parallel envs.
# Evaluation (eval/episodic_return) uses the checkpoint policy without exploration noise, as in the
# paper; charts/episodic_return logs the noisy behavior-policy training episodes.
#
# BETANOISE v1 — v215-style unimodal Beta actor head as state-dependent exploration for TD7.
#
# Motivation: fixed Gaussian exploration noise (and even the learned-sigma sdnoise variant) adds
# UNBOUNDED noise that is then clamped to [-1, 1]. On saturating MuJoCo policies whose mean sits at
# the action boundary, that clamp piles probability mass exactly on +-1 (a clamp-distortion "spike")
# and the noise no longer explores symmetrically. A Beta(conc_a, conc_b) policy over (0,1) mapped to
# (-1,1) has bounded support by construction: it never needs clamping, its noise shape adapts to the
# state (skewing away from the boundary near saturation), and 1+softplus keeps it unimodal (Dreamer4
# / v215). We transplant only this exploration head onto TD7: the actor outputs two concentrations
# instead of a mean, samples Beta for behavior/actor-update, and everything on the critic/TD side
# stays byte-identical TD3/TD7 (min-twin targets, fixed 0.2 target smoothing on the target actor's
# deterministic mean, LAP, target clipping, checkpointing). Entropy shapes ONLY the behavior
# sampling and the actor loss, never the value targets; alpha_ent is autotuned so the Beta's spread
# matches a sigma=0.1 Gaussian exploration budget. With --no-beta-noise the file reduces to the exact
# TD7 baseline (no Beta heads constructed, so parameter-init RNG is unchanged).
#
# Hypothesis: bounded Beta exploration removes the boundary clamp-mass distortion that Gaussian noise
# suffers on saturating policies, giving cleaner state-dependent exploration without biasing the
# value function, improving over fixed-noise TD7.
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
    """the scale of exploration noise (used only when beta_noise=False, i.e. baseline TD7 behavior)"""
    beta_noise: bool = True
    """use a v215-style unimodal Beta actor head (bounded state-dependent exploration) for behavior
    and the actor update; False falls back to exact baseline TD7 with fixed exploration_noise"""
    target_sigma: float = 0.1
    """action-space std whose Gaussian differential entropy sets the Beta entropy target (budget)"""
    alpha_ent: float = 0.2
    """fixed entropy-regularization coefficient used only when alpha_autotune=False"""
    alpha_autotune: bool = True
    """automatically tune the entropy coefficient toward the sigma=target_sigma-equivalent entropy"""
    alpha_lr: float = 1e-3
    """learning rate of the alpha_ent (log_alpha) optimizer, matching SAC's q_lr for its temperature"""
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


class LAPBuffer:
    """LAP replay buffer: priorities live on-device, sampling by inverse-CDF over the priority cumsum."""

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

        self.priority = torch.zeros(self.max_size, device=device)
        self.max_priority = 1.0

        self.normalize_actions = float(max_action)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

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
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, use_beta=False):
        super().__init__()
        self.use_beta = use_beta
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        # The two concentration heads are constructed only in Beta mode, so that with beta_noise=False
        # the module is parameter- and RNG-identical to the baseline TD7 actor (l3 is the mean head).
        if use_beta:
            self.alpha_head = nn.Linear(hdim, action_dim)
            self.beta_head = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        # Baseline mode returns (tanh(l3(.)), None); Beta mode returns the two concentrations
        # (conc_a, conc_b) of a unimodal Beta(>=1) over (0,1), later mapped to actions in (-1,1).
        a = avg_l1_norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        if not self.use_beta:
            return torch.tanh(self.l3(a)), None
        # 1 + softplus keeps each concentration >= 1 (unimodal Beta, Dreamer4 / v215). Clamp the
        # softplus output so extreme concentrations do not degrade Beta's implicit reparam gradients.
        conc_a = 1.0 + F.softplus(self.alpha_head(a)).clamp(max=1e4)
        conc_b = 1.0 + F.softplus(self.beta_head(a)).clamp(max=1e4)
        return conc_a, conc_b


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


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_action, args: Args, device, writer: SummaryWriter):
        self.args = args
        self.device = device
        self.writer = writer

        self.actor = Actor(
            state_dim, action_dim, args.zs_dim, args.hidden_dim, use_beta=args.beta_noise
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        # State-dependent Beta exploration temperature. Entropy is confined to the actor: alpha_ent
        # never enters the critic targets. The target is the z-space entropy of a Gaussian with
        # action-space std target_sigma: H_gauss = ln(sigma) + 0.5*ln(2*pi*e), minus ln(2) for the
        # (0,1)->(-1,1) action scaling (dist.entropy() is measured in the Beta's own (0,1) support).
        if args.beta_noise:
            if args.alpha_autotune:
                per_dim = float(np.log(args.target_sigma) + 0.5 * np.log(2 * np.pi * np.e) - np.log(2))
                self.target_entropy = action_dim * per_dim
                self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
                self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
                self.alpha_ent = self.log_alpha.exp().item()
            else:
                self.alpha_ent = args.alpha_ent

        self.critic = Critic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = LAPBuffer(state_dim, action_dim, device, args.buffer_size, args.batch_size, max_action)

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
                out_a, out_b = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                out_a, out_b = self.actor(state, zs)

            if self.args.beta_noise:
                # out_a, out_b are the Beta concentrations. Actions are natively in (-1,1): no clamp.
                if use_exploration:
                    z = torch.distributions.Beta(out_a, out_b).sample().clamp(1e-6, 1 - 1e-6)
                else:
                    # Deterministic acting (eval / checkpoint): Beta mean mapped to (-1,1).
                    z = out_a / (out_a + out_b)
                action = 2.0 * z - 1.0
                return action.cpu().data.numpy().flatten() * self.max_action

            # Baseline fallback: out_a is the tanh mean, out_b is None.
            action = out_a
            if use_exploration:
                action = action + torch.randn_like(action) * self.args.exploration_noise
            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def train(self):
        self.training_steps += 1

        state, action, next_state, reward, not_done = self.replay_buffer.sample()

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
            # TD3/TD7 target smoothing is applied to the target actor's deterministic mean only; the
            # Beta concentrations never touch the value targets (no entropy in targets).
            t_a, t_b = self.actor_target(next_state, fixed_target_zs)
            if self.args.beta_noise:
                next_mean = 2.0 * (t_a / (t_a + t_b)) - 1.0
            else:
                next_mean = t_a
            next_action = (next_mean + noise).clamp(-1, 1)

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

        # update LAP priorities
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.lap_alpha).detach()
        self.replay_buffer.update_priority(priority)

        # update actor (delayed)
        if self.training_steps % self.args.policy_freq == 0:
            out_a, out_b = self.actor(state, fixed_zs)
            if self.args.beta_noise:
                # Beta has implicit-reparameterization gradients through rsample; z in (0,1) mapped
                # to a in (-1,1). Actions are natively bounded, so no unclamped extrapolation is
                # needed to keep the gradient path clean (unlike the Gaussian sdnoise variant).
                dist = torch.distributions.Beta(out_a, out_b)
                z_r = dist.rsample().clamp(1e-6, 1 - 1e-6)
                actor_action = 2.0 * z_r - 1.0
            else:
                actor_action = out_a
            actor_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_action)
            actor_Q = self.critic(state, actor_action, actor_fixed_zsa, fixed_zs)
            actor_loss = -actor_Q.mean()
            if self.args.beta_noise:
                # Beta differential entropy (analytic, differentiable in the concentrations);
                # maximizing it via -alpha_ent*entropy encourages exploration. Only place entropy enters.
                ent = dist.entropy().sum(-1)
                actor_loss = actor_loss - self.alpha_ent * ent.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # autotune alpha_ent toward the sigma=target_sigma-equivalent entropy (holds the spread)
            if self.args.beta_noise and self.args.alpha_autotune:
                with torch.no_grad():
                    d_a, d_b = self.actor(state, fixed_zs)
                    ent_detached = torch.distributions.Beta(d_a, d_b).entropy().sum(-1)
                alpha_loss = (
                    -self.log_alpha.exp() * (self.target_entropy - ent_detached)
                ).mean()
                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha_ent = self.log_alpha.exp().item()

            if self.training_steps % 500 == 0:
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.training_steps)
            if self.args.beta_noise and self.training_steps % 1000 == 0:
                with torch.no_grad():
                    mean_conc = (0.5 * (out_a + out_b)).mean()
                    ent_mean = dist.entropy().sum(-1).mean()
                    det_action = 2.0 * (out_a / (out_a + out_b)) - 1.0
                    frac_saturated = (det_action.abs() > 0.9).float().mean()
                self.writer.add_scalar("charts/beta_mean_conc", mean_conc.item(), self.training_steps)
                self.writer.add_scalar("charts/beta_entropy", ent_mean.item(), self.training_steps)
                self.writer.add_scalar("losses/alpha_ent", self.alpha_ent, self.training_steps)
                self.writer.add_scalar("charts/beta_frac_saturated", frac_saturated.item(), self.training_steps)

        # hard target/fixed-encoder updates + target clip range snapshot
        if self.training_steps % self.args.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

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
    action_dim = np.prod(envs.single_action_space.shape)
    max_action = float(envs.single_action_space.high[0])

    agent = TD7Agent(state_dim, action_dim, max_action, args, device, writer)

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

        # save data to replay buffer; handle `final_observation`. The original bootstraps at the
        # timeout step even when the env truly terminated there (`done = ep_finished if
        # ep_timesteps < max_episode_steps else 0`), hence `and not truncations[0]`
        real_next_obs = next_obs[0]
        if truncations[0] or terminations[0]:
            real_next_obs = infos["final_observation"][0]
        done = float(terminations[0] and not truncations[0])
        agent.replay_buffer.add(obs[0], actions[0], real_next_obs, rewards[0], done)

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

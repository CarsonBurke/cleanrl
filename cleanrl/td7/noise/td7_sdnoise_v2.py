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
# SDNOISE v2 — temporally correlated (AR(1)/OU) exploration on top of sdnoise's state-dependent std.
#
# Base: td7_sdnoise_v1 (SAC's state-dependent log_std exploration head on TD7, without SAC's maxent
# objective; see that file). v1's diagnostics show sigma is genuinely state-dependent (sigma_min ~
# 0.01, sigma_max ~ 1.0, sigma_mean ~ 0.13 stable), i.e. the per-step noise budget is already
# allocated near-optimally across states — curvature-matched sigma_i^2 ~ alpha/|H_ii| is the optimal
# per-step allocation. But that noise is WHITE: independent each step. Over a locomotion stride the
# per-step perturbations average out, so at sigma ~ 0.13 exploration barely displaces the trajectory
# and the policy stops discovering new state regions — the diagnosed 8M bottleneck.
#
# v2 adds two ORTHOGONAL, independently-flagged mechanisms (both default-off-able, so this one file
# serves the OU-only, OU+decouple, and decouple-only arms):
#
# (A) noise_corr (rho, default 0.8): the exploration noise in select_action becomes temporally
# correlated via an AR(1) / discrete Ornstein-Uhlenbeck process on a unit-variance driver eps_t:
#   ou_eps_t = rho * ou_eps_{t-1} + sqrt(1 - rho^2) * z_t,   z_t ~ N(0, I)
# then the action is mu(s) + sigma(s) * ou_eps_t (same clamp). sqrt(1 - rho^2) is the exact factor
# that keeps the STATIONARY marginal variance of ou_eps at 1, so the entropy-budget accounting
# (alpha autotune toward act_dim*log(target_sigma)) is completely untouched — not extra noise, the
# SAME per-step magnitude given temporal STRUCTURE. Correlated noise drifts coherently along flat-Q
# directions across a stride and reaches new state regions instead of cancelling. ou_eps is
# STATIONARY-reset (a fresh N(0,I), not zeros) at each episode boundary and at init, so the
# variance-1 claim is exact from step 1 of every episode (a zero reset gives Var = 1 - rho^(2t),
# under-noised early). Those stationary draws use a DEDICATED generator, leaving the global RNG
# stream untouched: the per-step exploration draw is byte-identical to v1's, so rho=0 (--noise-corr
# 0) recovers v1 EXACTLY. Default rho=0.8 sits in the 0.7-0.85 safe band — checkpoint-selection
# corruption grows like (1+rho)/(1-rho) (TD7 selects by MIN return over noisy training episodes),
# and rho~0.95 is where it bites.
#
# (B) decouple_greedy (default False): v1/v2's mu is trained on the noise-smoothed Q(s, mu+sigma*eps)
# and so sits O(sigma^2 * Q''') off the true greedy argmax. With this flag mu instead ascends the
# DETERMINISTIC Q(s, mu) (the exact baseline TD7 DPG fixed point), while the log_std head keeps its
# unchanged gradient path through Q(s, mu.detach()+sigma*eps). Two critic evals, but per-parameter
# each still gets exactly one -Q gradient (mu only from the greedy term, sigma only from the noisy
# term), so the sigma(s) mechanism is byte-identical and only mu's fixed point moves.
#
# Everything else — the white actor rsample when coupled, entropy/alpha autotune, all value targets,
# LAP, checkpointing — is byte-identical to v1. With noise_corr=0 AND decouple_greedy=False the file
# reduces to v1 exactly.
#
# Registered predictions: (A) OU tracks v1 through ~200k (v1: 5404 @50k, 9726 @100k, 12020 @200k),
# then pulls ahead from ~300k as coherent drift reaches states white noise could not; bar >= 16000
# @400k (v1 ~15490), higher asymptote; falsified early if below v1-minus-noise at 100k. (B) decouple
# gives a small late-game EVAL gain (unbiased greedy target), within noise early. Watch-item: TD7
# checkpoint assessment scores TRAINING episodes (with exploration noise), so correlated noise raises
# per-episode return variance and may make checkpoint selection noisier — diagnosed by the new
# charts/ep_return_var and charts/ckpt_resets.
import copy
import os
import random
import time
from collections import deque
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
    """the scale of exploration noise (used only when sd_noise=False, i.e. baseline TD7 behavior)"""
    sd_noise: bool = True
    """use SAC-style state-dependent Gaussian noise (learned log_std head) for exploration and the
    actor update; False falls back to exact baseline TD7 with fixed exploration_noise"""
    target_sigma: float = 0.1
    """target per-dim exploration std the alpha autotuner holds the average sigma near (TD3 budget)"""
    alpha: float = 0.2
    """fixed entropy-regularization coefficient used only when alpha_autotune=False"""
    alpha_autotune: bool = True
    """automatically tune the entropy coefficient alpha toward act_dim*log(target_sigma)"""
    alpha_lr: float = 1e-3
    """learning rate of the alpha (log_alpha) optimizer, matching SAC's q_lr for its temperature"""
    noise_corr: float = 0.8
    """AR(1)/OU coefficient rho for temporally correlated exploration noise in select_action:
    ou_eps = rho*ou_eps + sqrt(1-rho^2)*randn (stationary marginal variance 1, entropy budget
    unchanged); stationary-init (N(0,I)) at each episode boundary. 0.0 recovers v1 exploration
    exactly. Default 0.8 sits in the 0.7-0.85 safe band: checkpoint-selection corruption grows like
    (1+rho)/(1-rho) because TD7 picks checkpoints by MIN return over noisy training episodes"""
    decouple_greedy: bool = False
    """P3 (orthogonal to noise_corr): train mu on the DETERMINISTIC greedy Q(s, mu) (exact baseline
    TD7 DPG fixed point) instead of the noise-smoothed Q(s, mu+sigma*eps), while the log_std head
    keeps its Q(s, mu.detach()+sigma*eps) gradient path unchanged. Restores the unbiased argmax for
    mu (v2's mu sits O(sigma^2) off it) while leaving the sigma(s) mechanism byte-identical"""
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


# Log-std clamp bounds for the state-dependent Gaussian policy. Actions live in [-1, 1], so the
# upper bound is 0 (sigma <= 1) rather than SAC's 2; the lower bound matches SAC (sigma >= ~6.7e-3).
LOG_STD_MIN = -5
LOG_STD_MAX = 0


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
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, state_dependent=False):
        super().__init__()
        self.state_dependent = state_dependent
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        # The log_std head is constructed only in state-dependent mode, so that with sd_noise=False
        # the module is parameter- and RNG-identical to the baseline TD7 actor.
        if state_dependent:
            self.l3_logstd = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        # Returns (mu, log_std). mu is the bounded deterministic action tanh(l3(.)), matching the
        # baseline actor's output exactly; log_std is None unless the state-dependent head exists.
        a = avg_l1_norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        mu = torch.tanh(self.l3(a))
        if not self.state_dependent:
            return mu, None
        log_std = torch.tanh(self.l3_logstd(a))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mu, log_std


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
            state_dim, action_dim, args.zs_dim, args.hidden_dim, state_dependent=args.sd_noise
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        # State-dependent exploration temperature. Entropy is confined to the actor: alpha never
        # enters the critic targets. Autotuning holds the average per-dim sigma near target_sigma.
        if args.sd_noise:
            if args.alpha_autotune:
                self.target_entropy = action_dim * float(np.log(args.target_sigma))
                self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
                self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
                self.alpha = self.log_alpha.exp().item()
            else:
                self.alpha = args.alpha

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
        self.ckpt_resets = 0  # cumulative early-abort count (current policy already lost checkpoint)

        # target value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0.0
        self.min_target = 0.0

        # AR(1)/OU correlated-exploration state (v2): unit-variance driver carried across steps within
        # an episode, stationary-reset at episode boundaries. Shape broadcasts against mu
        # (1, action_dim). The stationary init/reset draws (N(0,I)) come from a DEDICATED generator so
        # the global torch RNG stream is untouched: the per-step exploration draw stays byte-identical
        # to v1's, hence noise_corr=0 reproduces v1 exactly, while the marginal variance is exactly 1
        # at every step (zero-reset would give Var = 1 - rho^(2t), under-noised early in the episode).
        self._action_dim = int(action_dim)
        self.ou_gen = torch.Generator(device=device)
        self.ou_gen.manual_seed(int(args.seed) + 1_000_003)
        self.ou_eps = torch.randn(self._action_dim, generator=self.ou_gen, device=device)

    def reset_exploration_noise(self):
        # Stationary reset at episode boundaries: draw a fresh N(0,I) (marginal variance 1) rather
        # than zeros, so the variance-1 budget claim holds from the first step of every episode.
        # Uses the dedicated ou_gen, leaving the global RNG stream (and thus v1 parity) intact.
        self.ou_eps = torch.randn(self._action_dim, generator=self.ou_gen, device=self.device)

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                mu, log_std = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                mu, log_std = self.actor(state, zs)

            if use_exploration:
                # AR(1)/OU correlated driver: exactly one N(0,I) draw per step (same RNG draw count
                # as v1's white noise), carried across steps within the episode. sqrt(1-rho^2) holds
                # the stationary marginal variance at 1, so sigma(s)*ou_eps has the same per-step
                # magnitude as v1 and the entropy budget is untouched. rho=0 => ou_eps == the fresh
                # draw => byte-identical to v1's white exploration.
                rho = self.args.noise_corr
                eps = torch.randn_like(mu)
                self.ou_eps = rho * self.ou_eps + (1.0 - rho * rho) ** 0.5 * eps
                if self.args.sd_noise:
                    # State-dependent Gaussian: a = mu + sigma(s) * ou_eps, clamped to action bounds.
                    action = mu + log_std.exp() * self.ou_eps
                else:
                    action = mu + self.ou_eps * self.args.exploration_noise
            else:
                # Deterministic acting (eval / checkpoint policy), exactly as the baseline.
                action = mu

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
            # TD3/TD7 target smoothing is applied to the target actor's deterministic mean only;
            # the learned exploration std never touches the value targets (no entropy in targets).
            next_mu, _ = self.actor_target(next_state, fixed_target_zs)
            next_action = (next_mu + noise).clamp(-1, 1)

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
            mu, log_std = self.actor(state, fixed_zs)
            if self.args.sd_noise and self.args.decouple_greedy:
                # P3 decoupled greedy: mu ascends the DETERMINISTIC Q(s, mu) (exact baseline DPG
                # fixed point, unbiased by exploration noise), while the log_std head keeps its v2
                # gradient through Q(s, mu.detach()+sigma*eps). Two critic evals, but each parameter
                # gets exactly ONE -Q gradient: mu only from the greedy term (sigma absent), log_std
                # only from the noisy term (mu detached). One randn draw, same as the coupled path.
                greedy_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, mu)
                actor_Q = self.critic(state, mu, greedy_fixed_zsa, fixed_zs)
                noisy_action = mu.detach() + log_std.exp() * torch.randn_like(mu)
                noisy_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, noisy_action)
                noisy_Q = self.critic(state, noisy_action, noisy_fixed_zsa, fixed_zs)
                actor_loss = -actor_Q.mean() - noisy_Q.mean()
            else:
                if self.args.sd_noise:
                    # Reparameterized sample a_r = mu + sigma*eps (grad flows to both the mean and the
                    # log_std head). Left unclamped so the gradient path stays clean; the resulting
                    # action is mu(in (-1,1)) plus small Gaussian noise, well inside the critic's domain.
                    actor_action = mu + log_std.exp() * torch.randn_like(mu)
                else:
                    actor_action = mu
                actor_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_action)
                actor_Q = self.critic(state, actor_action, actor_fixed_zsa, fixed_zs)
                actor_loss = -actor_Q.mean()
            if self.args.sd_noise:
                # Gaussian entropy (up to an additive constant) is sum(log_std); maximizing it via
                # -alpha*entropy encourages exploration. This is the ONLY place entropy enters.
                gauss_ent = log_std.sum(-1)
                actor_loss = actor_loss - self.alpha * gauss_ent.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # autotune alpha toward the target entropy (holds average sigma near target_sigma)
            if self.args.sd_noise and self.args.alpha_autotune:
                with torch.no_grad():
                    _, log_std_detached = self.actor(state, fixed_zs)
                gauss_ent_detached = log_std_detached.sum(-1)
                alpha_loss = (
                    -self.log_alpha.exp() * (self.target_entropy - gauss_ent_detached)
                ).mean()
                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

            if self.training_steps % 500 == 0:
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.training_steps)
            if self.args.sd_noise and self.training_steps % 1000 == 0:
                with torch.no_grad():
                    sigma = log_std.exp()
                self.writer.add_scalar("charts/sigma_mean", sigma.mean().item(), self.training_steps)
                self.writer.add_scalar("charts/sigma_min", sigma.min().item(), self.training_steps)
                self.writer.add_scalar("charts/sigma_max", sigma.max().item(), self.training_steps)
                self.writer.add_scalar("losses/alpha", self.alpha, self.training_steps)

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
            self.ckpt_resets += 1  # v2 diagnostic: rises faster when noisy returns cut assessment short
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

    # v2 diagnostic: trailing window of training episodic returns whose variance is the estimator
    # TD7's min-return checkpoint selector consumes; correlated noise is expected to inflate it.
    ep_return_hist = deque(maxlen=agent.args.max_eps_when_checkpointing)

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

        # v2 diagnostic: norm of the correlated-noise state (~sqrt(action_dim) at stationarity,
        # held across episode resets by the stationary N(0,I) reset); logged on the env-step axis.
        if allow_train and global_step % 1000 == 0:
            writer.add_scalar("charts/ou_eps_norm", float(agent.ou_eps.norm()), global_step)

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

                    # v2 diagnostics: variance of recent training returns (the checkpoint-selector's
                    # estimator) and the cumulative early-abort count.
                    ep_return_hist.append(ep_return)
                    if len(ep_return_hist) >= 2:
                        writer.add_scalar("charts/ep_return_var", float(np.var(ep_return_hist)), global_step)
                    writer.add_scalar("charts/ckpt_resets", agent.ckpt_resets, global_step)

                    # v2: stationary-reset the AR(1)/OU correlated-noise state so temporal correlation
                    # does not carry across the episode boundary into the freshly reset env.
                    agent.reset_exploration_noise()

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

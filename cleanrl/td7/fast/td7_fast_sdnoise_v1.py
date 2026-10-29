# TD7 FAST v1 — throughput port of ../noise/td7_sdnoise_v1.py (TD7 + SAC-style state-dependent
# exploration noise, entropy confined to the actor; --no-sd-noise gives plain TD7). Same algorithm,
# hyperparameters and training schedule; only the execution changes:
#   - replay storage and LAP priorities live on the GPU; sampling (cumsum + searchsorted), gathers,
#     priority scatter, max-priority tracking and target-value clipping never touch the host;
#   - a whole train step (encoder, twin critic, priorities, delayed actor, alpha) is one CUDA-graph
#     replay built from compiled losses; the actor-step and critic-only steps are separate graphs;
#   - TD7 trains only in bursts at episode boundaries (policy checkpointing), so the behavior and
#     checkpoint policies are host (NumPy) mirrors refreshed once per burst: acting never syncs the GPU;
#   - an episode's transitions are uploaded in one copy at its end, before its training burst;
#   - the 10 evaluation episodes run as one batched lockstep rollout.
# Deviations that do not change the algorithm: different RNG streams, TF32 matmuls, fused twin critics
# (one batched matmul per layer), and only checkpointed training (the TD7 default) is supported.
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

from cleanrl.shared.graph_step import capture_steps
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.timing import PhaseTimer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    compile: bool = True
    """compile the losses with Inductor before capturing the train-step graphs"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_starts: int = 25000
    """timesteps of uniform-random actions before training starts"""
    eval_freq: int = 5000
    """evaluate every N env steps"""
    eval_eps: int = 10
    """number of evaluation episodes (run as one batched rollout)"""
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
    alpha_lr: float = 1e-3
    """learning rate of the alpha (log_alpha) optimizer"""
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


# Log-std bounds of the state-dependent Gaussian: sigma in [~6.7e-3, 1] for actions in [-1, 1].
LOG_STD_MIN = -5
LOG_STD_MAX = 0


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def lap_huber(td_loss, min_priority=1.0):
    return torch.where(td_loss < min_priority, 0.5 * td_loss.pow(2), min_priority * td_loss).sum(1).mean()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, state_dependent=False):
        super().__init__()
        self.state_dependent = state_dependent
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        if state_dependent:
            self.l3_logstd = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = avg_l1_norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        mu = torch.tanh(self.l3(a))
        if not self.state_dependent:
            return mu, None
        log_std = torch.tanh(self.l3_logstd(a))
        return mu, LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = F.elu(self.zs1(state))
        zs = F.elu(self.zs2(zs))
        return avg_l1_norm(self.zs3(zs))

    def zsa(self, zs, action):
        zsa = F.elu(self.zsa1(torch.cat([zs, action], 1)))
        zsa = F.elu(self.zsa2(zsa))
        return self.zsa3(zsa)


class EnsembleLinear(nn.Module):
    """``members`` independent nn.Linear layers applied to (members, batch, in) as one baddbmm."""

    def __init__(self, members, in_features, out_features):
        super().__init__()
        bound = in_features**-0.5  # nn.Linear's default init bound, for weight and bias
        self.weight = nn.Parameter(torch.empty(members, in_features, out_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(members, 1, out_features).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class TwinCritic(nn.Module):
    """TD7's two critics, fused: identical per-member math, one kernel per layer for both."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.hdim = hdim
        self.q0 = nn.Linear(state_dim + action_dim, 2 * hdim)  # q01 and q02 side by side
        self.q1 = EnsembleLinear(2, 2 * zs_dim + hdim, hdim)
        self.q2 = EnsembleLinear(2, hdim, hdim)
        self.q3 = EnsembleLinear(2, hdim, 1)
        # Match nn.Linear(state_dim + action_dim, hdim) init per member (bound from fan_in, not 2*hdim).
        bound = (state_dim + action_dim) ** -0.5
        nn.init.uniform_(self.q0.weight, -bound, bound)
        nn.init.uniform_(self.q0.bias, -bound, bound)

    def forward(self, state, action, zsa, zs):
        batch = state.shape[0]
        h = avg_l1_norm(self.q0(torch.cat([state, action], 1)).view(batch, 2, self.hdim)).transpose(0, 1)
        embeddings = torch.cat([zsa, zs], 1).expand(2, batch, -1)
        h = F.elu(self.q1(torch.cat([h, embeddings], 2)))
        h = F.elu(self.q2(h))
        return self.q3(h).squeeze(-1).transpose(0, 1)  # (batch, 2)


class HostPolicy:
    """NumPy mirror of fixed_encoder.zs + actor for acting without touching the GPU.

    ``refresh`` pulls the device weights in one flat D2H copy; call it whenever the device policy
    changes (after every training burst). ``copy_from`` snapshots another mirror (checkpointing).
    """

    def __init__(self, encoder: Encoder, actor: Actor):
        self.state_dependent = actor.state_dependent
        head_w = [actor.l3.weight] + ([actor.l3_logstd.weight] if self.state_dependent else [])
        head_b = [actor.l3.bias] + ([actor.l3_logstd.bias] if self.state_dependent else [])
        self._sources = [
            (encoder.zs1.weight,), (encoder.zs1.bias,), (encoder.zs2.weight,), (encoder.zs2.bias,),
            (encoder.zs3.weight,), (encoder.zs3.bias,), (actor.l0.weight,), (actor.l0.bias,),
            (actor.l1.weight,), (actor.l1.bias,), (actor.l2.weight,), (actor.l2.bias,),
            tuple(head_w), tuple(head_b),
        ]
        shapes = [(sum(t.shape[0] for t in group),) + tuple(group[0].shape[1:]) for group in self._sources]
        self._flat = np.zeros(sum(int(np.prod(s)) for s in shapes), dtype=np.float32)
        self._views, offset = [], 0
        for shape in shapes:
            size = int(np.prod(shape))
            view = self._flat[offset : offset + size].reshape(shape)
            self._views.append(view.T if view.ndim == 2 else view)  # weights as (in, out)
            offset += size
        self.action_dim = actor.l3.out_features

    def refresh(self):
        flat = torch.cat([t.detach().reshape(-1) for group in self._sources for t in group])
        self._flat[:] = flat.cpu().numpy()

    def copy_from(self, other: "HostPolicy"):
        self._flat[:] = other._flat

    def __call__(self, state):
        (w1, b1, w2, b2, w3, b3, w0, b0, wl1, bl1, wl2, bl2, wh, bh) = self._views
        zs = state @ w1 + b1
        zs = np.where(zs > 0, zs, np.expm1(np.minimum(zs, 0)))
        zs = zs @ w2 + b2
        zs = np.where(zs > 0, zs, np.expm1(np.minimum(zs, 0)))
        zs = zs @ w3 + b3
        zs /= np.maximum(np.abs(zs).mean(-1, keepdims=True), 1e-8)
        a = state @ w0 + b0
        a /= np.maximum(np.abs(a).mean(-1, keepdims=True), 1e-8)
        a = np.maximum(np.concatenate([a, zs], 1) @ wl1 + bl1, 0)
        a = np.maximum(a @ wl2 + bl2, 0)
        out = a @ wh + bh
        mu = np.tanh(out[:, : self.action_dim])
        if not self.state_dependent:
            return mu, None
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (np.tanh(out[:, self.action_dim :]) + 1)
        return mu, log_std


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_episode_steps, args: Args, device, writer: SummaryWriter):
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        adam = dict(fused=True)

        self.actor = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim, args.sd_noise).to(device)
        self.actor_target = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim, args.sd_noise).to(device)
        self.critic = TwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder_target = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        for target, source in (
            (self.actor_target, self.actor),
            (self.critic_target, self.critic),
            (self.fixed_encoder, self.encoder),
            (self.fixed_encoder_target, self.encoder),
        ):
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, **adam)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr, **adam)
        # (destination, source) pairs of every hard update, in TD7's order (target before fixed).
        self._hard_updates = [
            (list(self.actor_target.parameters()), list(self.actor.parameters())),
            (list(self.critic_target.parameters()), list(self.critic.parameters())),
            (list(self.fixed_encoder_target.parameters()), list(self.fixed_encoder.parameters())),
            (list(self.fixed_encoder.parameters()), list(self.encoder.parameters())),
        ]

        # Entropy is confined to the actor loss; autotuning holds the mean sigma near target_sigma.
        optimizers = [self.encoder_optimizer, self.critic_optimizer, self.actor_optimizer]
        if args.sd_noise:
            self.target_entropy = action_dim * float(np.log(args.target_sigma))
            self.log_alpha = torch.zeros((), device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
            optimizers.append(self.alpha_optimizer)

        # Replay storage: rows are [state | action | next_state | reward | not_done].
        self.row_dim = 2 * state_dim + action_dim + 2
        self.storage = torch.zeros(args.buffer_size, self.row_dim, device=device)
        self.priority = torch.zeros(args.buffer_size, device=device)  # zero == empty slot, never sampled
        # Index of the last filled slot. A parallel scan can drift a few ulps through the all-zero
        # tail in either direction, so draws are scaled by csum[last] and indices clamped to last.
        self.last = torch.zeros(1, dtype=torch.long, device=device)
        self.ptr = 0
        self.size = 0
        self.staging = torch.zeros(max_episode_steps, self.row_dim, pin_memory=True)
        self.staging_np = self.staging.numpy()
        self.staged = 0

        f32 = dict(device=device, dtype=torch.float32)
        self.max_priority = torch.ones((), **f32)
        self.running_max = torch.full((), -1e8, **f32)
        self.running_min = torch.full((), 1e8, **f32)
        self.max_target = torch.zeros((), **f32)
        self.min_target = torch.zeros((), **f32)
        self.metric_names = ["encoder_loss", "critic_loss", "q_values", "actor_loss", "sigma_mean", "sigma_min", "sigma_max"]
        self.metrics = torch.zeros(len(self.metric_names), **f32)

        maybe_compile = (
            (lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}))
            if args.compile
            else (lambda fn: fn)
        )
        self._encoder_loss = maybe_compile(self._encoder_loss_fn)
        self._critic_loss = maybe_compile(self._critic_loss_fn)
        self._actor_loss = maybe_compile(self._actor_loss_fn)
        self._entropy = maybe_compile(self._entropy_fn)
        # training_steps % policy_freq == 0 selects the actor-step graph (TD7's delayed update).
        self.graphs = capture_steps(
            [self._make_step(update_actor=False), self._make_step(update_actor=True)],
            optimizers=optimizers,
            state_tensors=[self.priority, self.max_priority, self.running_max, self.running_min, self.metrics],
        )

        self.policy = HostPolicy(self.fixed_encoder, self.actor)
        self.checkpoint_policy = HostPolicy(self.fixed_encoder, self.actor)
        self.policy.refresh()
        self.checkpoint_policy.copy_from(self.policy)
        self.training_steps = 0

        # checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

    # ---- replay -------------------------------------------------------------------------------
    def stage(self, state, action, next_state, reward, done):
        row = self.staging_np[self.staged]
        s, a = self.state_dim, self.action_dim
        row[:s] = state
        row[s : s + a] = action
        row[s + a : 2 * s + a] = next_state
        row[-2] = reward
        row[-1] = 1.0 - done
        self.staged += 1

    def flush(self):
        """Upload the staged episode; new rows get the current max priority, as in LAP's add()."""
        n = self.staged
        index = (self.ptr + torch.arange(n, device=self.device)) % self.args.buffer_size
        self.storage.index_copy_(0, index, self.staging[:n].to(self.device))
        self.priority.index_put_((index,), self.max_priority.expand(n))
        self.ptr = (self.ptr + n) % self.args.buffer_size
        self.size = min(self.size + n, self.args.buffer_size)
        self.last.fill_(self.size - 1)
        self.staged = 0

    # ---- losses -------------------------------------------------------------------------------
    def _split(self, batch):
        s, a = self.state_dim, self.action_dim
        return batch[:, :s], batch[:, s : s + a], batch[:, s + a : 2 * s + a], batch[:, -2:-1], batch[:, -1:]

    def _encoder_loss_fn(self, batch):
        state, action, next_state, _, _ = self._split(batch)
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
        pred_zs = self.encoder.zsa(self.encoder.zs(state), action)
        return F.mse_loss(pred_zs, next_zs)

    def _critic_loss_fn(self, batch, noise, min_target, max_target):
        state, action, next_state, reward, not_done = self._split(batch)
        args = self.args
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)
            noise = (noise * args.target_policy_noise).clamp(-args.noise_clip, args.noise_clip)
            # Target smoothing acts on the target actor's mean; the learned std never enters targets.
            next_mu, _ = self.actor_target(next_state, fixed_target_zs)
            next_action = (next_mu + noise).clamp(-1, 1)
            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)
            q_target = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs)
            q_target = q_target.min(1, keepdim=True)[0].clamp(min_target, max_target)
            q_target = reward + not_done * args.gamma * q_target
            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)
        q = self.critic(state, action, fixed_zsa, fixed_zs)
        td = (q - q_target).abs()
        return lap_huber(td, args.min_priority), td.detach(), q_target.max(), q_target.min(), q.detach().mean(), fixed_zs

    def _actor_loss_fn(self, state, fixed_zs, eps, alpha):
        mu, log_std = self.actor(state, fixed_zs)
        # Reparameterized a = mu + sigma*eps (grad to mean and log_std head), unclamped as in v1.
        action = mu + log_std.exp() * eps if self.args.sd_noise else mu
        q = self.critic(state, action, self.fixed_encoder.zsa(fixed_zs, action), fixed_zs)
        loss = -q.mean()  # TD7 averages both critics
        if self.args.sd_noise:
            loss = loss - alpha * log_std.sum(-1).mean()
        return loss, log_std

    def _entropy_fn(self, state, fixed_zs):
        with torch.no_grad():
            return self.actor(state, fixed_zs)[1].sum(-1)

    # ---- train step ---------------------------------------------------------------------------
    def _make_step(self, update_actor):
        args, device = self.args, self.device

        def step():
            csum = torch.cumsum(self.priority, 0)
            ind = torch.searchsorted(csum, torch.rand(args.batch_size, device=device) * csum.gather(0, self.last))
            ind = torch.minimum(ind, self.last)  # the scan is only monotone up to rounding
            batch = self.storage.index_select(0, ind)

            self.encoder_optimizer.zero_grad(set_to_none=False)
            encoder_loss = self._encoder_loss(batch)
            encoder_loss.backward()
            self.encoder_optimizer.step()

            noise = torch.randn(args.batch_size, self.action_dim, device=device)
            self.critic_optimizer.zero_grad(set_to_none=False)
            critic_loss, td, qt_max, qt_min, q_mean, fixed_zs = self._critic_loss(
                batch, noise, self.min_target, self.max_target
            )
            critic_loss.backward()
            self.critic_optimizer.step()

            with torch.no_grad():
                torch.maximum(self.running_max, qt_max, out=self.running_max)
                torch.minimum(self.running_min, qt_min, out=self.running_min)
                priority = td.max(1)[0].clamp(min=args.min_priority).pow(args.lap_alpha)
                self.priority[ind] = priority
                torch.maximum(self.max_priority, priority.max(), out=self.max_priority)
                self.metrics[0].copy_(encoder_loss.detach())
                self.metrics[1].copy_(critic_loss.detach())
                self.metrics[2].copy_(q_mean)

            if update_actor:
                state = batch[:, : self.state_dim]
                eps = torch.randn(args.batch_size, self.action_dim, device=device)
                alpha = self.log_alpha.detach().exp() if args.sd_noise else None
                self.actor_optimizer.zero_grad(set_to_none=False)
                actor_loss, log_std = self._actor_loss(state, fixed_zs, eps, alpha)
                actor_loss.backward(inputs=list(self.actor.parameters()))
                self.actor_optimizer.step()
                with torch.no_grad():
                    self.metrics[3].copy_(actor_loss.detach())
                if args.sd_noise:
                    entropy = self._entropy(state, fixed_zs)
                    self.alpha_optimizer.zero_grad(set_to_none=False)
                    alpha_loss = -(self.log_alpha.exp() * (self.target_entropy - entropy)).mean()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    with torch.no_grad():
                        sigma = log_std.detach().exp()
                        self.metrics[4:7].copy_(torch.stack([sigma.mean(), sigma.min(), sigma.max()]))

        return step

    def hard_update(self):
        with torch.no_grad():
            for destination, source in self._hard_updates:
                torch._foreach_copy_(destination, source)
            self.max_priority.copy_(self.priority.max())
            self.max_target.copy_(self.running_max)
            self.min_target.copy_(self.running_min)

    def train(self):
        self.training_steps += 1
        self.graphs[int(self.training_steps % self.args.policy_freq == 0)].replay()
        if self.training_steps % self.args.target_update_rate == 0:
            self.hard_update()
        if self.training_steps % 500 == 0:
            values = dict(zip(self.metric_names, self.metrics.tolist()))
            for name in ("encoder_loss", "critic_loss", "q_values", "actor_loss"):
                self.writer.add_scalar(f"losses/{name}", values[name], self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target.item(), self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target.item(), self.training_steps)
            if self.args.sd_noise and self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.training_steps)

    # ---- acting -------------------------------------------------------------------------------
    def select_action(self, state, rng):
        mu, log_std = self.policy(state[None])
        sigma = np.exp(log_std) if self.args.sd_noise else self.args.exploration_noise
        return np.clip(mu + sigma * rng.standard_normal(mu.shape, dtype=np.float32), -1, 1)[0]

    # ---- checkpointing (run when each episode terminates) -------------------------------------
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps
        self.min_return = min(self.min_return, ep_return)

        # end assessment of the current policy early: it already lost to the checkpoint
        if self.min_return < self.best_min_return:
            self.train_and_reset()
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_policy.copy_from(self.policy)  # policy mirror == device actor + fixed encoder
            self.train_and_reset()

    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.args.steps_before_checkpointing:
                self.best_min_return *= self.args.reset_weight
                self.max_eps_before_update = self.args.max_eps_when_checkpointing
            self.train()
        self.policy.refresh()
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8


def evaluate(policy: HostPolicy, eval_envs, max_action):
    """Deterministic checkpoint-policy returns, one episode per env, stepped in lockstep."""
    states = np.stack([env.reset()[0] for env in eval_envs]).astype(np.float32)
    returns = np.zeros(len(eval_envs))
    active = np.ones(len(eval_envs), dtype=bool)
    while active.any():
        actions, _ = policy(states)
        for i in np.flatnonzero(active):
            state, reward, terminated, truncated, _ = eval_envs[i].step(actions[i] * max_action)
            states[i] = state
            returns[i] += reward
            active[i] = not (terminated or truncated)
    return returns.mean()


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    configure_runtime()
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")

    env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    eval_envs = [gym.make(args.env_id) for _ in range(args.eval_eps)]
    for i, eval_env in enumerate(eval_envs):
        eval_env.reset(seed=args.seed + 100 + i)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    agent = TD7Agent(state_dim, action_dim, env.spec.max_episode_steps, args, device, writer)
    timer = PhaseTimer()

    start_time = time.time()
    allow_train = False
    obs, _ = env.reset(seed=args.seed)
    obs = obs.astype(np.float32)
    ep_return, ep_length = 0.0, 0
    next_timing_log = 10000
    # total_timesteps + 1 so the final evaluation at exactly total_timesteps fires (as in the original)
    for global_step in range(args.total_timesteps + 1):
        if global_step % args.eval_freq == 0:
            with timer.span("eval", use_cuda=False):
                eval_return = evaluate(agent.checkpoint_policy, eval_envs, max_action)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(f"global_step={global_step}, eval_return={eval_return:.3f}")

        with timer.span("rollout", use_cuda=False):
            if allow_train:
                action = agent.select_action(obs, rng)
            else:
                action = env.action_space.sample() / max_action
        with timer.span("env", use_cuda=False):
            next_obs, reward, terminated, truncated, _ = env.step(action * max_action)
        next_obs = next_obs.astype(np.float32)
        # The original bootstraps at the timeout step even when the env truly terminated there.
        agent.stage(obs, action, next_obs, reward, float(terminated and not truncated))
        obs = next_obs
        ep_return += float(reward)
        ep_length += 1

        if terminated or truncated:
            print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            with timer.span("update"):
                agent.flush()
                if allow_train:
                    agent.maybe_train_and_checkpoint(ep_length, ep_return)
            if global_step >= args.learning_starts:
                allow_train = True
            if global_step >= next_timing_log:
                next_timing_log += 10000
                for name, stats in timer.summary().items():
                    writer.add_scalar(f"time/{name}_s", stats["total_s"], global_step)
                timer.reset()
            obs, _ = env.reset()
            obs = obs.astype(np.float32)
            ep_return, ep_length = 0.0, 0

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "encoder": agent.encoder.state_dict(),
                "critic": agent.critic.state_dict(),
                "checkpoint_policy": agent.checkpoint_policy._flat,
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    env.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()

# TD7 LATENT ACTION TWO-CRITIC v2 — one critic judges the actor's intention, another its motor command.
#
# The actor's trunk ends in a stochastic latent z = clip(m(s) + sigma(s) eps) (latent_dim); a detached head
# decodes it, a = tanh(z W + MLP(z)), and that action goes to the environment.
# - Latent critic Qz(s, z) and its own SALE encoder (zsz(zs_z, z) predicts the next zs_z) see only the state
#   and the latent: never the action, mu or sigma. The trunk is trained by Qz alone, through reparameterized
#   latent samples, so it learns which intentions are good. Its entropy budget is spent in action space:
#   0.5 logdet(J Sigma J^T) with J the head's (detached) Jacobian at m. A per-latent-dim budget is slack
#   (latentaction_v1: dims the head ignores sat at sigma=1, the rest at the floor, alpha ~0.004).
# - Action critic Qa(s, a), a TD7 critic with its own SALE encoder, trains only the head (the gradient stops
#   at the latent), on the mean intent m: the head turns intentions into the best actions. On samples it
#   would learn to absorb latent noise and erase exploration.
# W starts as v1's orthonormal gain matrix and the MLP's last layer at zero, so warmup and early exploration
# match td7_latentaction_v1 and the head learns a correction, slowly (head_lr): Qz values latents through
# the current head and replay holds latents decoded by older heads.
# Hypothesis: credit to intentions, judged without the motor details, generalizes better than action credit,
# while the action critic keeps execution sharp. fresh/*_z and fresh/*_a show each critic's error on unseen
# episodes; a growing Qz error with a drifting head is the failure signature.
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
    """timesteps of exploration (latent m=0, sigma=warmup_sigma through the initial head) before training"""
    warmup_sigma: float = 0.6
    """std of the Gaussian warmup latent"""
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
    target_sigma: float = 0.1
    """target per-dim action noise std: the autotuner holds the pushed-forward action entropy at action_dim*log(this)"""
    target_noise: float = 0.2
    """TD7 target policy smoothing std, in latent units for Qz and action units for Qa"""
    noise_clip: float = 0.5
    """clip of the target policy smoothing noise"""
    alpha_lr: float = 1e-3
    """learning rate of the alpha (log_alpha) optimizer"""
    policy_freq: int = 2
    """the frequency of training the trunk and head (delayed)"""
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
    """the learning rate of the trunk optimizer"""
    head_lr: float = 1e-4
    """the learning rate of the head optimizer (slow: Qz values latents through the current head)"""
    latent_dim: int = 64
    """dimensionality of the actor's stochastic final latent"""
    decoder_gain: float = 1.5
    """initial head: a = tanh(gain * W z), W with orthonormal rows (latent sigma 0.1 -> ~0.15 action noise)"""
    fresh_tail_weight: float = 0.01
    """fresh-data optimism uses steps whose truncated-return tail weight gamma^k is below this"""


# Log-std bounds of the state-dependent latent Gaussian: sigma in [~6.7e-3, 1] for latents in [-1, 1].
LOG_STD_MIN = -5
LOG_STD_MAX = 0


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def lap_huber(td_loss, min_priority=1.0):
    return torch.where(td_loss < min_priority, 0.5 * td_loss.pow(2), min_priority * td_loss).sum(1).mean()


class Trunk(nn.Module):
    """The actor up to its final latent: (state, zs_z) -> latent mean m in (-1, 1) and log std."""

    def __init__(self, state_dim, latent_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, latent_dim)
        self.l3_logstd = nn.Linear(hdim, latent_dim)

    def forward(self, state, zs):
        a = avg_l1_norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        log_std = torch.tanh(self.l3_logstd(a))
        return torch.tanh(self.l3(a)), LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)


class Head(nn.Module):
    """The detached motor head: a = tanh(z W + MLP(z)), W initialized to gain * orthonormal rows, MLP to zero."""

    def __init__(self, latent_dim, action_dim, gain, seed, hdim=256):
        super().__init__()
        basis = np.linalg.qr(np.random.default_rng(seed).standard_normal((latent_dim, action_dim)))[0]
        self.base = nn.Parameter(torch.as_tensor(gain * basis, dtype=torch.float32))  # (latent_dim, action_dim)
        self.h1 = nn.Linear(latent_dim, hdim)
        self.h2 = nn.Linear(hdim, hdim)
        self.h3 = nn.Linear(hdim, action_dim)
        nn.init.zeros_(self.h3.weight)
        nn.init.zeros_(self.h3.bias)

    def forward(self, z):
        h = F.elu(self.h1(z))
        h = F.elu(self.h2(h))
        return torch.tanh(z @ self.base + self.h3(h))

    def jacobian(self, z):
        """Analytic da/dz, (batch, action_dim, latent_dim), through tanh, the base map and the ELU MLP."""
        p1 = self.h1(z)
        p2 = self.h2(F.elu(p1))
        a = torch.tanh(z @ self.base + self.h3(F.elu(p2)))
        d1 = torch.where(p1 > 0, 1.0, p1.exp())  # ELU'
        d2 = torch.where(p2 > 0, 1.0, p2.exp())
        mlp = ((self.h3.weight * d2[:, None, :]) @ self.h2.weight * d1[:, None, :]) @ self.h1.weight
        return (1 - a.pow(2))[:, :, None] * (self.base.T + mlp)


class Encoder(nn.Module):
    """SALE: zs(s), and zsx(zs, x) predicting the next state's zs from x (an action or a latent)."""

    def __init__(self, state_dim, input_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        self.zsa1 = nn.Linear(zs_dim + input_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = F.elu(self.zs1(state))
        zs = F.elu(self.zs2(zs))
        return avg_l1_norm(self.zs3(zs))

    def zsa(self, zs, x):
        zsa = F.elu(self.zsa1(torch.cat([zs, x], 1)))
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
    """Fused TD7 twin critic Q(s, x, zsx, zs) for an input x (an action or a latent); returns (batch, 2)."""

    def __init__(self, state_dim, input_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.hdim = hdim
        self.q0 = nn.Linear(state_dim + input_dim, 2 * hdim)  # both members side by side
        self.q1 = EnsembleLinear(2, 2 * zs_dim + hdim, hdim)
        self.q2 = EnsembleLinear(2, hdim, hdim)
        self.q3 = EnsembleLinear(2, hdim, 1)
        bound = (state_dim + input_dim) ** -0.5  # per-member nn.Linear init
        nn.init.uniform_(self.q0.weight, -bound, bound)
        nn.init.uniform_(self.q0.bias, -bound, bound)

    def forward(self, state, x, zsx, zs):
        batch = state.shape[0]
        h = avg_l1_norm(self.q0(torch.cat([state, x], 1)).view(batch, 2, self.hdim)).transpose(0, 1)
        h = F.elu(self.q1(torch.cat([h, torch.cat([zsx, zs], 1).expand(2, batch, -1)], 2)))
        h = F.elu(self.q2(h))
        return self.q3(h).squeeze(-1).transpose(0, 1)


class HostPolicy:
    """NumPy mirror of the latent encoder's zs, the trunk and the head, for acting without touching the GPU.

    ``refresh`` pulls the device weights in one flat D2H copy (after every training burst); ``copy_from``
    snapshots another mirror (checkpointing).
    """

    def __init__(self, encoder: Encoder, trunk: Trunk, head: Head):
        linears = [
            encoder.zs1, encoder.zs2, encoder.zs3, trunk.l0, trunk.l1, trunk.l2, trunk.l3, trunk.l3_logstd,
            head.h1, head.h2, head.h3,
        ]
        self._sources = [t for layer in linears for t in (layer.weight, layer.bias)] + [head.base]
        self._flat = np.zeros(sum(t.numel() for t in self._sources), dtype=np.float32)
        self._views, offset = [], 0
        for t in self._sources:
            view = self._flat[offset : offset + t.numel()].reshape(t.shape)
            self._views.append(view.T if t is not head.base and view.ndim == 2 else view)  # Linear weights as (in, out)
            offset += t.numel()

    def refresh(self):
        self._flat[:] = torch.cat([t.detach().reshape(-1) for t in self._sources]).cpu().numpy()

    def copy_from(self, other: "HostPolicy"):
        self._flat[:] = other._flat

    @staticmethod
    def _elu(x):
        return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))

    def latent(self, state):
        """state (batch, state_dim) -> latent mean m and log std."""
        (w1, b1, w2, b2, w3, b3, w0, b0, wl1, bl1, wl2, bl2, wm, bm, ws, bs) = self._views[:16]
        zs = self._elu(self._elu(state @ w1 + b1) @ w2 + b2) @ w3 + b3
        zs /= np.maximum(np.abs(zs).mean(-1, keepdims=True), 1e-8)
        a = state @ w0 + b0
        a /= np.maximum(np.abs(a).mean(-1, keepdims=True), 1e-8)
        a = np.maximum(np.concatenate([a, zs], 1) @ wl1 + bl1, 0)
        a = np.maximum(a @ wl2 + bl2, 0)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (np.tanh(a @ ws + bs) + 1)
        return np.tanh(a @ wm + bm), log_std

    def head(self, z):
        """latent (batch, latent_dim) -> action in [-1, 1]."""
        (h1, c1, h2, c2, h3, c3, base) = self._views[16:]
        h = self._elu(self._elu(z @ h1 + c1) @ h2 + c2)
        return np.tanh(z @ base + h @ h3 + c3)


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_episode_steps, args: Args, device, writer: SummaryWriter):
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim, self.latent_dim = state_dim, action_dim, args.latent_dim
        adam = dict(fused=True)
        z_dim, hdim, zs_dim = args.latent_dim, args.hidden_dim, args.zs_dim

        self.trunk = Trunk(state_dim, z_dim, zs_dim, hdim).to(device)
        self.trunk_target = Trunk(state_dim, z_dim, zs_dim, hdim).to(device)
        self.head = Head(z_dim, action_dim, args.decoder_gain, args.seed, hdim).to(device)
        self.head_target = Head(z_dim, action_dim, args.decoder_gain, args.seed, hdim).to(device)
        # Latent side: everything the intention critic sees. Action side: everything the motor critic sees.
        self.critic_z = TwinCritic(state_dim, z_dim, zs_dim, hdim).to(device)
        self.critic_z_target = TwinCritic(state_dim, z_dim, zs_dim, hdim).to(device)
        self.critic_a = TwinCritic(state_dim, action_dim, zs_dim, hdim).to(device)
        self.critic_a_target = TwinCritic(state_dim, action_dim, zs_dim, hdim).to(device)
        self.encoder_z = Encoder(state_dim, z_dim, zs_dim, hdim).to(device)
        self.fixed_encoder_z = Encoder(state_dim, z_dim, zs_dim, hdim).to(device)
        self.fixed_encoder_z_target = Encoder(state_dim, z_dim, zs_dim, hdim).to(device)
        self.encoder_a = Encoder(state_dim, action_dim, zs_dim, hdim).to(device)
        self.fixed_encoder_a = Encoder(state_dim, action_dim, zs_dim, hdim).to(device)
        self.fixed_encoder_a_target = Encoder(state_dim, action_dim, zs_dim, hdim).to(device)
        for target, source in (
            (self.trunk_target, self.trunk),
            (self.head_target, self.head),
            (self.critic_z_target, self.critic_z),
            (self.critic_a_target, self.critic_a),
            (self.fixed_encoder_z, self.encoder_z),
            (self.fixed_encoder_z_target, self.encoder_z),
            (self.fixed_encoder_a, self.encoder_a),
            (self.fixed_encoder_a_target, self.encoder_a),
        ):
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        self.trunk_optimizer = torch.optim.Adam(self.trunk.parameters(), lr=args.actor_lr, **adam)
        self.head_optimizer = torch.optim.Adam(self.head.parameters(), lr=args.head_lr, **adam)
        self.critic_optimizer = torch.optim.Adam(
            [*self.critic_z.parameters(), *self.critic_a.parameters()], lr=args.critic_lr, **adam
        )
        self.encoder_optimizer = torch.optim.Adam(
            [*self.encoder_z.parameters(), *self.encoder_a.parameters()], lr=args.encoder_lr, **adam
        )
        # (destination, source) pairs of every hard update, in TD7's order (target before fixed).
        pairs = [
            (self.trunk_target, self.trunk),
            (self.head_target, self.head),
            (self.critic_z_target, self.critic_z),
            (self.critic_a_target, self.critic_a),
            (self.fixed_encoder_z_target, self.fixed_encoder_z),
            (self.fixed_encoder_z, self.encoder_z),
            (self.fixed_encoder_a_target, self.fixed_encoder_a),
            (self.fixed_encoder_a, self.encoder_a),
        ]
        self._hard_updates = [(list(d.parameters()), list(s.parameters())) for d, s in pairs]

        # Entropy is confined to the trunk loss and measured in action space (see _policy_loss_fn).
        self.target_entropy = action_dim * float(np.log(args.target_sigma))
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
        optimizers = [
            self.encoder_optimizer, self.critic_optimizer, self.trunk_optimizer, self.head_optimizer,
            self.alpha_optimizer,
        ]

        # Replay rows keep the executed latent and the action the head decoded it to at the time.
        self._fields = {}
        offset = 0
        for name, width in (
            ("state", state_dim), ("latent", z_dim), ("action", action_dim), ("next_state", state_dim),
            ("reward", 1), ("not_done", 1),
        ):
            self._fields[name] = slice(offset, offset + width)
            offset += width
        self.row_dim = offset
        self.storage = torch.zeros(args.buffer_size, self.row_dim, device=device)
        self.priority = torch.zeros(args.buffer_size, device=device)  # zero == empty slot, never sampled
        # Index of the last filled slot; draws are scaled by csum[last] and indices clamped to last.
        self.last = torch.zeros(1, dtype=torch.long, device=device)
        self.ptr = 0
        self.size = 0
        self.staging = torch.zeros(max_episode_steps, self.row_dim, pin_memory=True)
        self.staging_np = self.staging.numpy()
        self.staged = 0

        f32 = dict(device=device, dtype=torch.float32)
        self.max_priority = torch.ones((), **f32)
        # Per-critic target bounds (index 0: Qz, 1: Qa): running extrema and the frozen copies the targets use.
        self.running_max = torch.full((2,), -1e8, **f32)
        self.running_min = torch.full((2,), 1e8, **f32)
        self.max_target = torch.zeros(2, **f32)
        self.min_target = torch.zeros(2, **f32)
        self.metric_names = [
            "encoder_z_loss", "encoder_a_loss", "critic_z_loss", "critic_a_loss", "qz_values", "qa_values",
            "trunk_loss", "head_loss", "sigma_mean", "sigma_min", "sigma_max", "head_correction", "action_entropy",
        ]
        self.metrics = torch.zeros(len(self.metric_names), **f32)
        self.fresh_cut = math.ceil(math.log(args.fresh_tail_weight) / math.log(args.gamma))

        maybe_compile = (
            (lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}))
            if args.compile
            else (lambda fn: fn)
        )
        self._encoder_loss = maybe_compile(self._encoder_loss_fn)
        self._critic_loss = maybe_compile(self._critic_loss_fn)
        self._policy_loss = maybe_compile(self._policy_loss_fn)
        # training_steps % policy_freq == 0 selects the trunk/head-step graph (TD7's delayed update).
        self.graphs = capture_steps(
            [self._make_step(update_policy=False), self._make_step(update_policy=True)],
            optimizers=optimizers,
            state_tensors=[self.priority, self.max_priority, self.running_max, self.running_min, self.metrics],
        )

        self.policy = HostPolicy(self.fixed_encoder_z, self.trunk, self.head)
        self.checkpoint_policy = HostPolicy(self.fixed_encoder_z, self.trunk, self.head)
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
    def stage(self, **values):
        row = self.staging_np[self.staged]
        for name, value in values.items():
            row[self._fields[name]] = value
        self.staged += 1

    def flush(self, global_step, terminated):
        """Upload the staged episode; new rows get the current max priority, as in LAP's add().

        The episode is first scored for fresh-data optimism by both critics, before any update sees it.
        """
        n = self.staged
        rows = self.staging[:n].to(self.device)
        if self.training_steps > 0:
            self._log_fresh(rows, self.staging_np[:n, self._fields["reward"]].ravel(), terminated, global_step)
        index = (self.ptr + torch.arange(n, device=self.device)) % self.args.buffer_size
        self.storage.index_copy_(0, index, rows)
        self.priority.index_put_((index,), self.max_priority.expand(n))
        self.ptr = (self.ptr + n) % self.args.buffer_size
        self.size = min(self.size + n, self.args.buffer_size)
        self.last.fill_(self.size - 1)
        self.staged = 0

    # ---- losses -------------------------------------------------------------------------------
    def _encoder_loss_fn(self, batch):
        f = self._fields
        state, next_state = batch[:, f["state"]], batch[:, f["next_state"]]
        losses = []
        for encoder, x in ((self.encoder_z, batch[:, f["latent"]]), (self.encoder_a, batch[:, f["action"]])):
            with torch.no_grad():
                next_zs = encoder.zs(next_state)
            losses.append(F.mse_loss(encoder.zsa(encoder.zs(state), x), next_zs))
        return losses[0] + losses[1], torch.stack(losses).detach()

    def _critic_loss_fn(self, batch, min_target, max_target):
        f, args = self._fields, self.args
        state, next_state = batch[:, f["state"]], batch[:, f["next_state"]]
        latent, action = batch[:, f["latent"]], batch[:, f["action"]]
        reward, not_done = batch[:, f["reward"]], batch[:, f["not_done"]]
        with torch.no_grad():
            # Target intention m', smoothed in latent units for Qz; its decoded action, smoothed in action
            # units, for Qa (TD7's target policy smoothing on each side).
            next_zs_z = self.fixed_encoder_z_target.zs(next_state)
            next_m, _ = self.trunk_target(next_state, next_zs_z)
            noise_z = (torch.randn_like(next_m) * args.target_noise).clamp(-args.noise_clip, args.noise_clip)
            next_z = (next_m + noise_z).clamp(-1, 1)
            next_a = self.head_target(next_m)
            noise_a = (torch.randn_like(next_a) * args.target_noise).clamp(-args.noise_clip, args.noise_clip)
            next_a = (next_a + noise_a).clamp(-1, 1)
            next_zs_a = self.fixed_encoder_a_target.zs(next_state)
            q_next_z = self.critic_z_target(next_state, next_z, self.fixed_encoder_z_target.zsa(next_zs_z, next_z), next_zs_z)
            q_next_a = self.critic_a_target(next_state, next_a, self.fixed_encoder_a_target.zsa(next_zs_a, next_a), next_zs_a)
            target_z = reward + not_done * args.gamma * q_next_z.min(1, keepdim=True)[0].clamp(min_target[0], max_target[0])
            target_a = reward + not_done * args.gamma * q_next_a.min(1, keepdim=True)[0].clamp(min_target[1], max_target[1])
            fixed_zs_z = self.fixed_encoder_z.zs(state)
            fixed_zs_a = self.fixed_encoder_a.zs(state)
            zsz = self.fixed_encoder_z.zsa(fixed_zs_z, latent)
            zsa = self.fixed_encoder_a.zsa(fixed_zs_a, action)
        q_z = self.critic_z(state, latent, zsz, fixed_zs_z)
        q_a = self.critic_a(state, action, zsa, fixed_zs_a)
        td_z, td_a = (q_z - target_z).abs(), (q_a - target_a).abs()
        loss_z, loss_a = lap_huber(td_z, args.min_priority), lap_huber(td_a, args.min_priority)
        with torch.no_grad():
            diagnostics = torch.stack([
                loss_z, loss_a, q_z.mean(), q_a.mean(),
                target_z.max(), target_a.max(), target_z.min(), target_a.min(),
            ])
        td = torch.maximum(td_z.max(1)[0], td_a.max(1)[0])
        return loss_z + loss_a, td.detach(), diagnostics, fixed_zs_z, fixed_zs_a

    def _policy_loss_fn(self, state, fixed_zs_z, fixed_zs_a, alpha):
        """Trunk: ascend Qz at a reparameterized latent sample (+ entropy). Head: ascend Qa at its decoding
        of the (detached) mean intention. Neither critic sees the other side's input."""
        m, log_std = self.trunk(state, fixed_zs_z)
        z = (m + log_std.exp() * torch.randn_like(m)).clamp(-1, 1)
        q_z = self.critic_z(state, z, self.fixed_encoder_z.zsa(fixed_zs_z, z), fixed_zs_z).mean(1)
        # Gaussian entropy of the action noise J diag(sigma^2) J^T, J detached: sigma alone pays for exploration,
        # and latent directions the head ignores earn none.
        with torch.no_grad():
            jac = self.head.jacobian(m.detach())
        cov = (jac * log_std.exp().pow(2)[:, None, :]) @ jac.transpose(1, 2)
        cov = cov + 1e-6 * torch.eye(self.action_dim, device=cov.device)
        chol, _ = torch.linalg.cholesky_ex(cov)  # no info check: no host sync inside the graph
        entropy = torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(-1)  # 0.5 logdet, up to a constant
        trunk_loss = -q_z.mean() - alpha * entropy.mean()
        intention = m.detach()
        a = self.head(intention)
        q_a = self.critic_a(state, a, self.fixed_encoder_a.zsa(fixed_zs_a, a), fixed_zs_a).mean(1)
        head_loss = -q_a.mean()
        with torch.no_grad():
            correction = (a - torch.tanh(intention @ self.head.base)).abs().mean()
        return trunk_loss, head_loss, log_std, entropy.detach(), correction

    # ---- train step ---------------------------------------------------------------------------
    def _make_step(self, update_policy):
        args, device = self.args, self.device

        def step():
            csum = torch.cumsum(self.priority, 0)
            ind = torch.searchsorted(csum, torch.rand(args.batch_size, device=device) * csum.gather(0, self.last))
            ind = torch.minimum(ind, self.last)  # the scan is only monotone up to rounding
            batch = self.storage.index_select(0, ind)

            self.encoder_optimizer.zero_grad(set_to_none=False)
            encoder_loss, encoder_parts = self._encoder_loss(batch)
            encoder_loss.backward()
            self.encoder_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=False)
            critic_loss, td, diagnostics, fixed_zs_z, fixed_zs_a = self._critic_loss(batch, self.min_target, self.max_target)
            critic_loss.backward()
            self.critic_optimizer.step()

            with torch.no_grad():
                torch.maximum(self.running_max, diagnostics[4:6], out=self.running_max)
                torch.minimum(self.running_min, diagnostics[6:8], out=self.running_min)
                priority = td.clamp(min=args.min_priority).pow(args.lap_alpha)
                self.priority[ind] = priority
                torch.maximum(self.max_priority, priority.max(), out=self.max_priority)
                self.metrics[0:2].copy_(encoder_parts)
                self.metrics[2:6].copy_(diagnostics[:4])

            if update_policy:
                state = batch[:, self._fields["state"]]
                self.trunk_optimizer.zero_grad(set_to_none=False)
                self.head_optimizer.zero_grad(set_to_none=False)
                trunk_loss, head_loss, log_std, entropy, correction = self._policy_loss(
                    state, fixed_zs_z, fixed_zs_a, self.log_alpha.detach().exp()
                )
                # One backward: the compiled loss is a single autograd node. head_loss reaches only the head
                # (through m.detach()) and trunk_loss never touches it, so the sum keeps the two separate.
                (trunk_loss + head_loss).backward(inputs=[*self.trunk.parameters(), *self.head.parameters()])
                self.trunk_optimizer.step()
                self.head_optimizer.step()
                self.alpha_optimizer.zero_grad(set_to_none=False)
                alpha_loss = -(self.log_alpha.exp() * (self.target_entropy - entropy)).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    sigma = log_std.detach().exp()
                    self.metrics[6:13].copy_(torch.stack([
                        trunk_loss.detach(), head_loss.detach(), sigma.mean(), sigma.min(), sigma.max(), correction,
                        entropy.mean(),
                    ]))

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
            for name in (
                "encoder_z_loss", "encoder_a_loss", "critic_z_loss", "critic_a_loss", "qz_values", "qa_values",
                "trunk_loss", "head_loss", "head_correction",
            ):
                self.writer.add_scalar(f"losses/{name}", values[name], self.training_steps)
            if self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max", "action_entropy"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.training_steps)

    # ---- fresh-data optimism ------------------------------------------------------------------
    def _log_fresh(self, rows, rewards, terminated, global_step):
        """Each critic's prediction at the executed input minus the realized discounted return, on an
        episode no update has seen. Truncated episodes keep only steps whose unobserved tail weighs less
        than fresh_tail_weight."""
        n = len(rewards)
        valid = n if terminated else n - self.fresh_cut
        if valid <= 0:
            return
        returns, g = np.empty(n), 0.0
        for t in range(n - 1, -1, -1):
            g = rewards[t] + self.args.gamma * g
            returns[t] = g
        f = self._fields
        rows = rows[:valid]
        state, latent, action = rows[:, f["state"]], rows[:, f["latent"]], rows[:, f["action"]]
        realized = torch.as_tensor(returns[:valid], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            zs_z, zs_a = self.fixed_encoder_z.zs(state), self.fixed_encoder_a.zs(state)
            q_z = self.critic_z(state, latent, self.fixed_encoder_z.zsa(zs_z, latent), zs_z).mean(1)
            q_a = self.critic_a(state, action, self.fixed_encoder_a.zsa(zs_a, action), zs_a).mean(1)
            scale = realized.abs().mean().clamp(min=1e-8)
            error_z, error_a = q_z - realized, q_a - realized
            stats = torch.stack([
                error_z.mean(), error_z.mean() / scale, error_z.abs().mean(),
                error_a.mean(), error_a.mean() / scale, error_a.abs().mean(),
            ]).tolist()
        names = ("optimism_z", "rel_optimism_z", "abs_err_z", "optimism_a", "rel_optimism_a", "abs_err_a")
        for name, value in zip(names, stats):
            self.writer.add_scalar(f"fresh/{name}", value, global_step)

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
            self.checkpoint_policy.copy_from(self.policy)  # mirror == device trunk + head + fixed latent encoder
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
    """Deterministic checkpoint-policy returns (the head's decoding of the latent mean), in lockstep."""
    states = np.stack([env.reset()[0] for env in eval_envs]).astype(np.float32)
    returns = np.zeros(len(eval_envs))
    active = np.ones(len(eval_envs), dtype=bool)
    while active.any():
        actions = policy.head(policy.latent(states)[0])
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
    eval_envs = [gym.make(args.env_id) for _ in range(args.eval_eps)]
    for i, eval_env in enumerate(eval_envs):
        eval_env.reset(seed=args.seed + 100 + i)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    agent = TD7Agent(state_dim, action_dim, env.spec.max_episode_steps, args, device, writer)
    timer = PhaseTimer()
    warmup_m = np.zeros(args.latent_dim, dtype=np.float32)
    warmup_sigma = np.full(args.latent_dim, args.warmup_sigma, dtype=np.float32)

    start_time = time.time()
    allow_train = False
    obs, _ = env.reset(seed=args.seed)
    obs = obs.astype(np.float32)
    ep_return, ep_length = 0.0, 0
    ep_noise_sq, ep_saturated = 0.0, 0.0
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
                m, log_std = agent.policy.latent(obs[None])
                m, sigma = m[0], np.exp(log_std[0])
            else:
                m, sigma = warmup_m, warmup_sigma
            z = np.clip(m + sigma * rng.standard_normal(args.latent_dim, dtype=np.float32), -1, 1)
            action, action_mean = agent.policy.head(np.stack([z, m]))
            ep_noise_sq += float(np.square(action - action_mean).mean())
            ep_saturated += float((np.abs(action) > 0.99).mean())
        with timer.span("env", use_cuda=False):
            next_obs, reward, terminated, truncated, _ = env.step(action * max_action)
        next_obs = next_obs.astype(np.float32)
        # The original bootstraps at the timeout step even when the env truly terminated there.
        agent.stage(
            state=obs, latent=z, action=action, next_state=next_obs, reward=reward,
            not_done=1.0 - float(terminated and not truncated),
        )
        obs = next_obs
        ep_return += float(reward)
        ep_length += 1

        if terminated or truncated:
            print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            writer.add_scalar("charts/action_noise_std", (ep_noise_sq / ep_length) ** 0.5, global_step)
            writer.add_scalar("charts/action_saturation", ep_saturated / ep_length, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            with timer.span("update"):
                agent.flush(global_step, terminated and not truncated)
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
            ep_noise_sq, ep_saturated = 0.0, 0.0

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            {
                "trunk": agent.trunk.state_dict(),
                "head": agent.head.state_dict(),
                "encoder_z": agent.encoder_z.state_dict(),
                "encoder_a": agent.encoder_a.state_dict(),
                "critic_z": agent.critic_z.state_dict(),
                "critic_a": agent.critic_a.state_dict(),
                "checkpoint_policy": agent.checkpoint_policy._flat,
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    env.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()

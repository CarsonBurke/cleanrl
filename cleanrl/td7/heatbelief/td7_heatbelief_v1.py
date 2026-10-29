# TD7 HEAT-BELIEF v1 — the critic values the actor's belief N(mu, sigma^2) exactly, in closed form.
#
# The value of a belief is a Gaussian transform of the point value, V(s, mu, S) = E_{u~N(mu,S)} Q(s, u),
# so it obeys the heat equation dV/dS = 1/2 Hess_mu V in every environment. latentq v5 learns V as a
# free function of (mu, sigma, u) with Hermite tangents: its belief and point queries conflict at u=mu,
# its actor gradient also flows through u=mu and zsa(mu), sigma credit is a noisy He2 regression, and it
# is trained at stale stored beliefs. Here Q(s, a) = V0(s) + sum_k c_k(s) prod_i exp(-(a_i - m_ki)^2 /
# 2 l_ki^2) with c_k >= 0. For a diagonal belief u ~ N(mu, sigma^2) and the env's a = clip(u), the
# expectation factorizes per dimension into a widened bump on the box plus the boundary masses (erf),
# so every belief's value is exact in one elementwise pass, clipping included. The critic is trained
# only on what happened (point TD at the realized action); the actor is credited only for what it
# believes (exact dV/dmu, dV/dsigma at its current belief; no action path). The target is the exact
# expectation under TD7's 0.2 smoothing (no sampled target noise). Q is bounded in a, so the actor cannot
# climb a linear extrapolation, and bumps narrower than sigma are discounted by the transform itself.
# --chart adds zsa(zs, mu_current) (detached) as trunk context, the only way SALE's zsa enters.
# Hypothesis: exact, current belief credit from a bounded critic beats v5's learned belief critic.
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
    """timesteps of latent-free exploration (mu=0, sigma=warmup_sigma) before training starts"""
    warmup_sigma: float = 0.6
    """std of the Gaussian warmup policy (the TD7 uniform warmup has no Gaussian latent; 0.6 ~ its std)"""
    bumps: int = 32
    """Gaussian bumps per twin member in the point critic"""
    chart: bool = False
    """feed zsa(zs, current policy mean) to the critic trunk as context (never differentiated by the actor)"""
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
    """target per-dim exploration std the alpha autotuner holds the average sigma near (TD3 budget)"""
    target_smoothing_sigma: float = 0.2
    """per-dim std of the (exact) Gaussian smoothing of the target policy's mean, TD7's 0.2"""
    alpha_lr: float = 1e-3
    """learning rate of the alpha (log_alpha) optimizer"""
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
    fresh_tail_weight: float = 0.01
    """fresh-data optimism uses steps whose truncated-return tail weight gamma^k is below this"""


# Log-std bounds of the state-dependent Gaussian: sigma in [~6.7e-3, 1] for actions in [-1, 1].
LOG_STD_MIN = -5
LOG_STD_MAX = 0


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def lap_huber(td_loss, min_priority=1.0):
    return torch.where(td_loss < min_priority, 0.5 * td_loss.pow(2), min_priority * td_loss).sum(1).mean()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        self.l3_logstd = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = avg_l1_norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        log_std = torch.tanh(self.l3_logstd(a))
        return torch.tanh(self.l3(a)), LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.zs_dim = zs_dim
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


def clipped_bump_belief(center, width, mu, sigma=None):
    """E[prod_i exp(-(a_i - m_i)^2 / 2 l_i^2)] with a = clip(u, -1, 1), u ~ N(mu, diag sigma^2).

    The environment sees clip(u), so Q is flat beyond the action box. For axis-aligned bumps and a
    diagonal belief the expectation factorizes per dimension, and each factor is exact: the Gaussian
    product on (-1, 1) (a widened bump times the mass of N(c, tau^2) inside the box) plus the bump's
    boundary values weighted by the belief's mass beyond each bound. sigma=None is the point value at
    a = mu. All shapes broadcast; the product runs over the last (action) dimension.
    """
    if sigma is None:
        return torch.exp(-0.5 * ((mu - center) / width).square().sum(-1))
    var, width2 = sigma.square(), width.square()
    total = width2 + var
    log_scale = 0.5 * (width2.log() - total.log()) - 0.5 * (mu - center).square() / total
    inner_mean = (mu * width2 + center * var) / total
    inner_std = width * sigma / total.sqrt()
    interior = log_scale.exp() * (
        torch.special.ndtr((1 - inner_mean) / inner_std) - torch.special.ndtr((-1 - inner_mean) / inner_std)
    )
    upper = torch.exp(-0.5 * ((1 - center) / width).square()) * torch.special.ndtr((mu - 1) / sigma)
    lower = torch.exp(-0.5 * ((1 + center) / width).square()) * torch.special.ndtr((-1 - mu) / sigma)
    return (interior + upper + lower).prod(-1)


class HeatKernelTwinCritic(nn.Module):
    """Fused twin point critic Q(s, a) = V0(s) + sum_k c_k(s) bump_k(a), valued at any belief in closed form."""

    LOG_WIDTH_MID, LOG_WIDTH_HALF = math.log(0.3), math.log(10.0)  # bump widths in [0.03, 3] action units

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, bumps=32, chart=False):
        super().__init__()
        self.hdim, self.bumps, self.action_dim = hdim, bumps, action_dim
        self.q0 = nn.Linear(state_dim, 2 * hdim)  # both members side by side
        self.q1 = EnsembleLinear(2, hdim + zs_dim * (2 if chart else 1), hdim)
        self.q2 = EnsembleLinear(2, hdim, hdim)
        self.splits = [1, bumps, bumps * action_dim, bumps * action_dim]
        self.q3 = EnsembleLinear(2, hdim, sum(self.splits))
        with torch.no_grad():
            # Bumps start spread over the action box at unit width, so every bump sees data, with weak
            # state dependence that TD then grows.
            shape = slice(1 + bumps, None)
            self.q3.weight[..., shape].mul_(0.1)
            self.q3.bias[..., 1 + bumps : 1 + bumps + bumps * action_dim].uniform_(-1, 1)
            self.q3.bias[..., 1 + bumps + bumps * action_dim :].fill_(
                math.atanh(-self.LOG_WIDTH_MID / self.LOG_WIDTH_HALF)  # width 1
            )

    def forward(self, state, zs, mu, sigma=None, chart=None):
        """(batch, 2) values of the belief N(mu, diag sigma^2) under action clipping, or the point value at
        a = mu if sigma is None. Also returns each bump's contribution (2, batch, K) and mean log width.
        """
        batch = state.shape[0]
        h = avg_l1_norm(self.q0(state).view(batch, 2, self.hdim)).transpose(0, 1)
        context = zs if chart is None else torch.cat([zs, chart], 1)
        h = F.elu(self.q1(torch.cat([h, context.expand(2, batch, -1)], 2)))
        h = F.elu(self.q2(h))
        v0, amp, center, log_width = self.q3(h).split(self.splits, -1)
        center = center.view(2, batch, self.bumps, self.action_dim)
        log_width = self.LOG_WIDTH_MID + self.LOG_WIDTH_HALF * torch.tanh(log_width.view_as(center))
        contrib = F.softplus(amp) * clipped_bump_belief(
            center, log_width.exp(), mu[None, :, None], None if sigma is None else sigma[None, :, None]
        )
        q = v0.squeeze(-1) + contrib.sum(-1)
        return q.transpose(0, 1), contrib, log_width.mean()


class HostPolicy:
    """NumPy mirror of fixed_encoder.zs + actor for acting without touching the GPU.

    ``refresh`` pulls the device weights in one flat D2H copy; call it whenever the device policy
    changes (after every training burst). ``copy_from`` snapshots another mirror (checkpointing).
    """

    def __init__(self, encoder: Encoder, actor: Actor):
        self._sources = [
            (encoder.zs1.weight,), (encoder.zs1.bias,), (encoder.zs2.weight,), (encoder.zs2.bias,),
            (encoder.zs3.weight,), (encoder.zs3.bias,), (actor.l0.weight,), (actor.l0.bias,),
            (actor.l1.weight,), (actor.l1.bias,), (actor.l2.weight,), (actor.l2.bias,),
            (actor.l3.weight, actor.l3_logstd.weight), (actor.l3.bias, actor.l3_logstd.bias),
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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (np.tanh(out[:, self.action_dim :]) + 1)
        return np.tanh(out[:, : self.action_dim]), log_std


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_episode_steps, args: Args, device, writer: SummaryWriter):
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        adam = dict(fused=True)

        self.actor = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        critic = lambda: HeatKernelTwinCritic(  # noqa: E731
            state_dim, action_dim, args.zs_dim, args.hidden_dim, args.bumps, args.chart
        ).to(device)
        self.critic, self.critic_target = critic(), critic()
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
        self.target_entropy = action_dim * float(np.log(args.target_sigma))
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
        optimizers = [self.encoder_optimizer, self.critic_optimizer, self.actor_optimizer, self.alpha_optimizer]

        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (("state", s), ("action", a), ("next_state", s), ("reward", 1), ("not_done", 1)):
            self._fields[name] = slice(offset, offset + width)
            offset += width
        self.row_dim = offset
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
        self.smoothing_sigma = torch.full((action_dim,), args.target_smoothing_sigma, **f32)
        self.metric_names = [
            "encoder_loss", "critic_loss", "q_values", "noise_cost", "bump_eff", "bump_active", "log_width",
            "bump_mean", "actor_loss", "sigma_mean", "sigma_min", "sigma_max",
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
    def stage(self, **values):
        row = self.staging_np[self.staged]
        for name, value in values.items():
            row[self._fields[name]] = value
        self.staged += 1

    def flush(self, global_step, terminated):
        """Upload the staged episode; new rows get the current max priority, as in LAP's add().

        The episode was collected by one policy against the current critic, so it is scored for
        fresh-data optimism before any training step sees it.
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
        with torch.no_grad():
            next_zs = self.encoder.zs(batch[:, f["next_state"]])
        pred_zs = self.encoder.zsa(self.encoder.zs(batch[:, f["state"]]), batch[:, f["action"]])
        return F.mse_loss(pred_zs, next_zs)

    def _critic_loss_fn(self, batch, min_target, max_target):
        f, args = self._fields, self.args
        state, next_state, action = batch[:, f["state"]], batch[:, f["next_state"]], batch[:, f["action"]]
        with torch.no_grad():
            # Exact value of the target policy's belief N(mu', 0.2^2) at s'; no smoothing noise is sampled.
            next_zs = self.fixed_encoder_target.zs(next_state)
            next_mu, _ = self.actor_target(next_state, next_zs)
            next_chart = self.fixed_encoder_target.zsa(next_zs, next_mu) if args.chart else None
            q_target = self.critic_target(
                next_state, next_zs, next_mu, self.smoothing_sigma.expand_as(next_mu), next_chart
            )[0]
            q_target = q_target.min(1, keepdim=True)[0].clamp(min_target, max_target)
            q_target = batch[:, f["reward"]] + batch[:, f["not_done"]] * args.gamma * q_target
            fixed_zs = self.fixed_encoder.zs(state)
            chart = self.fixed_encoder.zsa(fixed_zs, self.actor(state, fixed_zs)[0]) if args.chart else None
        # The critic learns only what happened: the point value of the executed (clipped) action.
        q, contrib, log_width = self.critic(state, fixed_zs, action, None, chart)
        td = (q - q_target).abs()
        loss = lap_huber(td, args.min_priority)
        with torch.no_grad():
            share = contrib / contrib.sum(-1, keepdim=True).clamp(min=1e-12)
            diagnostics = torch.stack([
                q.mean(),
                q_target.max(),
                q_target.min(),
                # Effective and >1%-share bump counts at the executed action, and the action-dependent
                # part of Q above its floor V0.
                (-(share * share.clamp(min=1e-12).log()).sum(-1)).exp().mean(),
                (share > 0.01).float().sum(-1).mean(),
                log_width,
                contrib.sum(-1).mean(),
            ])
        return loss, td.detach(), diagnostics, fixed_zs

    def _actor_loss_fn(self, state, fixed_zs, alpha):
        """Ascend the exact value of the current belief; the chart context is never differentiated."""
        mu, log_std = self.actor(state, fixed_zs)
        chart = self.fixed_encoder.zsa(fixed_zs, mu.detach()) if self.args.chart else None
        v = self.critic(state, fixed_zs, mu, log_std.exp(), chart)[0].mean(1)
        with torch.no_grad():
            # Value the belief gives up to its own noise: point value at mu minus belief value.
            noise_cost = (self.critic(state, fixed_zs, mu, None, chart)[0].mean(1) - v).mean()
        return -v.mean() - alpha * log_std.sum(-1).mean(), log_std, noise_cost

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

            self.critic_optimizer.zero_grad(set_to_none=False)
            critic_loss, td, diagnostics, fixed_zs = self._critic_loss(batch, self.min_target, self.max_target)
            critic_loss.backward()
            self.critic_optimizer.step()

            with torch.no_grad():
                torch.maximum(self.running_max, diagnostics[1], out=self.running_max)
                torch.minimum(self.running_min, diagnostics[2], out=self.running_min)
                priority = td.max(1)[0].clamp(min=args.min_priority).pow(args.lap_alpha)
                self.priority[ind] = priority
                torch.maximum(self.max_priority, priority.max(), out=self.max_priority)
                self.metrics[0].copy_(encoder_loss.detach())
                self.metrics[1].copy_(critic_loss.detach())
                self.metrics[2].copy_(diagnostics[0])
                self.metrics[4:8].copy_(diagnostics[3:7])

            if update_actor:
                state = batch[:, self._fields["state"]]
                self.actor_optimizer.zero_grad(set_to_none=False)
                actor_loss, log_std, noise_cost = self._actor_loss(state, fixed_zs, self.log_alpha.detach().exp())
                actor_loss.backward(inputs=list(self.actor.parameters()))
                self.actor_optimizer.step()
                entropy = self._entropy(state, fixed_zs)
                self.alpha_optimizer.zero_grad(set_to_none=False)
                alpha_loss = -(self.log_alpha.exp() * (self.target_entropy - entropy)).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    sigma = log_std.detach().exp()
                    self.metrics[3].copy_(noise_cost)
                    self.metrics[8:12].copy_(torch.stack([actor_loss.detach(), sigma.mean(), sigma.min(), sigma.max()]))

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
                "encoder_loss", "critic_loss", "q_values", "noise_cost", "bump_eff", "bump_active", "log_width",
                "bump_mean", "actor_loss",
            ):
                self.writer.add_scalar(f"losses/{name}", values[name], self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target.item(), self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target.item(), self.training_steps)
            if self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.training_steps)

    # ---- fresh-data optimism ------------------------------------------------------------------
    def _log_fresh(self, rows, rewards, terminated, global_step):
        """Critic prediction minus realized discounted return on an episode no update has seen.

        Truncated episodes keep only steps whose unobserved tail weighs less than fresh_tail_weight.
        Scored at the executed action, the point query the critic is trained on (twin mean).
        """
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
        state, action = rows[:, f["state"]], rows[:, f["action"]]
        realized = torch.as_tensor(returns[:valid], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            zs = self.fixed_encoder.zs(state)
            chart = self.fixed_encoder.zsa(zs, self.actor(state, zs)[0]) if self.args.chart else None
            q = self.critic(state, zs, action, None, chart)[0].mean(1)
            error = q - realized
            stats = torch.stack([
                error.mean(), error.mean() / realized.abs().mean().clamp(min=1e-8), error.abs().mean()
            ]).tolist()
        for name, value in zip(("optimism", "rel_optimism", "abs_err"), stats):
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
    """Deterministic checkpoint-policy returns (the latent mean), one episode per env, in lockstep."""
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
    eval_envs = [gym.make(args.env_id) for _ in range(args.eval_eps)]
    for i, eval_env in enumerate(eval_envs):
        eval_env.reset(seed=args.seed + 100 + i)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    agent = TD7Agent(state_dim, action_dim, env.spec.max_episode_steps, args, device, writer)
    timer = PhaseTimer()
    warmup_mu = np.zeros(action_dim, dtype=np.float32)
    warmup_sigma = np.full(action_dim, args.warmup_sigma, dtype=np.float32)

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
                mu, log_std = agent.policy(obs[None])
                mu, sigma = mu[0], np.exp(log_std[0])
            else:
                mu, sigma = warmup_mu, warmup_sigma
            action = np.clip(mu + sigma * rng.standard_normal(action_dim, dtype=np.float32), -1, 1)
        with timer.span("env", use_cuda=False):
            next_obs, reward, terminated, truncated, _ = env.step(action * max_action)
        next_obs = next_obs.astype(np.float32)
        # The original bootstraps at the timeout step even when the env truly terminated there.
        agent.stage(
            state=obs, action=action, next_state=next_obs, reward=reward, not_done=1.0 - float(terminated and not truncated),
        )
        obs = next_obs
        ep_return += float(reward)
        ep_length += 1

        if terminated or truncated:
            print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
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

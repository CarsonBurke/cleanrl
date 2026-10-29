# TD7 LATENT-Q ANALYTIC CLIPPED-GAUSSIAN MOMENTS v21.
# State/SALE-conditioned fixed physical quadratic + five hinges per joint.
# Point returns identify one function; distribution values and actor derivatives
# integrate that SAME function exactly, including clipped-Gaussian boundary atoms.
# Package change: observed-return MSE with exact LAP compensation; remove ineffective
# future CVs. Preserve lambda=.95/horizon32 and v20 behavior-cached target refresh.
# Lane 0 is excluded from critic fitting, but retained in actor/SALE minibatches.
# Its return/score residuals are noisy bootstrap-dependent checks, not Q oracles.
# Hypothesis: consistent physical action credit improves learning. Small sigma can
# still leave curvature/hinges unidentified; exact convolution is no accuracy claim.
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
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    compile: bool = True
    """compile the losses with Inductor before capturing the train-step graphs"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_steps: int = 1024
    """steps per environment per rollout"""
    env_threads: int = 2
    """maximum physics threads"""
    updates_per_step: float = 1.0
    """fresh-rollout minibatch updates per environment step"""
    warmup_rollouts: int = 1
    """recorded rollouts of latent-free exploration (mu=0, sigma=warmup_sigma) before training starts"""
    warmup_sigma: float = 0.6
    """std of the Gaussian warmup policy (the TD7 uniform warmup has no Gaussian latent; 0.6 ~ its std)"""
    trace_lambda: float = 0.95
    """mixing coefficient for behavior-policy traces with refreshed bootstraps"""
    trace_horizon: int = 32
    """independent temporal trace segment length; interaction rollout remains 1024 steps"""
    eval_interval: int = 2
    """evaluate the policy mean every N rollouts"""
    eval_eps: int = 10
    """number of evaluation episodes (run as one batched rollout)"""
    batch_size: int = 256
    """minibatch size sampled from the latest rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_update_rate: int = 250
    """hard-update the target networks and fixed encoders every N training steps"""
    target_sigma: float = 0.1
    """target per-dim exploration std the alpha autotuner holds the average sigma near (TD3 budget)"""
    target_smoothing_sigma: float = 0.2
    """legacy target query std; unused by cached behavior-policy labels"""
    alpha_lr: float = 1e-3
    """learning rate of the alpha (log_alpha) optimizer"""
    policy_freq: int = 2
    """the frequency of training the policy"""
    lap_alpha: float = 0.4
    """within-rollout LAP prioritization exponent"""
    min_priority: float = 1.0
    """LAP minimum priority"""
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


def lambda_returns(rewards, not_done, boundaries, next_values, gamma, trace_lambda, trace_horizon=32):
    """Time-major lane-local traces with independent segments and autoreset cuts.

    Terminations mask all bootstrap. Truncations bootstrap their final observation
    but stop recursion. Each segment's final real row uses its own next value;
    padding only batches a short final segment and cannot affect real targets.
    """
    assert rewards.shape == not_done.shape == boundaries.shape == next_values.shape
    assert rewards.shape[0] > 0 and trace_horizon > 0
    length = rewards.shape[0]
    segments = (length + trace_horizon - 1) // trace_horizon
    padding = segments * trace_horizon - length
    cuts = boundaries.clone()
    cuts[-1] = 1  # preserve the last real bootstrap when final segment is padded

    def segmented(x, fill=0):
        if padding:
            x = torch.cat([x, x.new_full((padding,) + x.shape[1:], fill)])
        return x.reshape(segments, trace_horizon, *x.shape[1:]).transpose(0, 1)

    r, mask, cut, values = segmented(rewards), segmented(not_done), segmented(cuts, 1), segmented(next_values)
    continuation = values[-1]
    targets = []
    for t in range(trace_horizon - 1, -1, -1):
        trace = torch.where(cut[t].bool(), values[t], continuation)
        continuation = r[t] + gamma * mask[t] * ((1 - trace_lambda) * values[t] + trace_lambda * trace)
        targets.append(continuation)
    targets = torch.stack(targets[::-1]).transpose(0, 1)
    return targets.reshape(segments * trace_horizon, *rewards.shape[1:])[:length]


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def clipped_gaussian_moments(mu, sigma, low=-1., high=1.):
    """First two raw moments of clip(mu + sigma*eps, low, high).

    The deterministic branch uses safe finite denominators even when unused,
    so zero-std point queries have finite autograd derivatives.
    """
    positive = sigma > 0
    safe_sigma = torch.where(positive, sigma, torch.ones_like(sigma))
    alpha, beta = (low - mu) / safe_sigma, (high - mu) / safe_sigma
    cdf_low = 0.5 * torch.erfc(-alpha / 2**0.5)
    tail_high = 0.5 * torch.erfc(beta / 2**0.5)
    interior = 1 - cdf_low - tail_high
    phi_low = torch.exp(-0.5 * alpha.square()) / (2 * torch.pi)**0.5
    phi_high = torch.exp(-0.5 * beta.square()) / (2 * torch.pi)**0.5
    first = mu * interior + safe_sigma * (phi_low - phi_high) + low * cdf_low + high * tail_high
    second = (mu.square() + safe_sigma.square()) * interior + safe_sigma * (
        (mu + low) * phi_low - (mu + high) * phi_high
    ) + low**2 * cdf_low + high**2 * tail_high
    point = mu.clamp(low, high)
    return torch.where(positive, first, point), torch.where(positive, second, point.square())


def clipped_gaussian_hinge(mu, sigma, knots, high=1.):
    """E[(clip(U)-k)+] = GaussianCall(k)-GaussianCall(high), k interior.

    Output (..., action_dim, num_knots); fixed physical units, no moving anchors.
    """
    positive = sigma > 0
    safe_sigma = torch.where(positive, sigma, torch.ones_like(sigma))
    def call(strike):
        difference = mu.unsqueeze(-1) - strike
        std = safe_sigma.unsqueeze(-1)
        z = difference / std
        return difference * (0.5 * torch.erfc(-z / 2**0.5)) + std * torch.exp(-0.5 * z.square()) / (2 * torch.pi)**0.5
    expectation = call(knots) - call(high)
    point = (mu.clamp(max=high).unsqueeze(-1) - knots).relu()
    return torch.where(positive.unsqueeze(-1), expectation, point)


def analytic_basis_value(coefficients, mu, sigma, knots, quadratic_indices):
    """Coefficients (B,twins,C): V, g, upper-triangle H, flattened hinges.

    Diagonal H entries multiply .5*E[a_i²]; offdiagonal entries multiply
    E[a_i]*E[a_j], because symmetric H appears twice in .5*a^T H*a.
    Independent clipped coordinates retain full cross-joint quadratic terms.
    """
    dimension = mu.shape[-1]
    first, second = clipped_gaussian_moments(mu, sigma)
    i, j = quadratic_indices[0], quadratic_indices[1]
    quadratic = torch.where(i == j, 0.5 * second[:, i], first[:, i] * first[:, j])
    hinges = clipped_gaussian_hinge(mu, sigma, knots).flatten(1)
    basis = torch.cat([torch.ones_like(first[:, :1]), first, quadratic, hinges], 1)
    return (coefficients * basis.unsqueeze(1)).sum(-1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.sale = zs_dim > 0  # zs_dim=0: state-only actor, ``zs`` is ignored
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        self.l3_logstd = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = avg_l1_norm(self.l0(state))
        if self.sale:
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


class AnalyticLatentTwinCritic(nn.Module):
    """Only state and SALE zs condition coefficients; queries enter fixed basis."""
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.hdim = hdim
        self.q0 = nn.Linear(state_dim, 2 * hdim)
        self.q1 = EnsembleLinear(2, zs_dim + hdim, hdim)
        self.q2 = EnsembleLinear(2, hdim, hdim)
        count = 1 + action_dim + action_dim * (action_dim + 1) // 2 + 5 * action_dim
        self.q3 = EnsembleLinear(2, hdim, count)
        self.register_buffer("knots", torch.linspace(-2 / 3, 2 / 3, 5))
        self.register_buffer("quadratic_indices", torch.triu_indices(action_dim, action_dim))

    def coefficients(self, state, zs):
        batch = state.shape[0]
        h = avg_l1_norm(self.q0(state).view(batch, 2, self.hdim)).transpose(0, 1)
        h = F.elu(self.q1(torch.cat([h, zs.expand(2, batch, -1)], 2)))
        h = F.elu(self.q2(h))
        return self.q3(h).transpose(0, 1)

    def forward(self, state, mu, sigma, zs):
        return analytic_basis_value(self.coefficients(state, zs), mu, sigma, self.knots, self.quadratic_indices)


class HostPolicy:
    """NumPy mirror of the actor (and fixed_encoder.zs when it feeds the actor) for acting on the host.

    ``refresh`` pulls the device weights in one flat D2H copy; call it whenever the device policy
    changes (after every training burst).
    """

    def __init__(self, encoder: Encoder, actor: Actor):
        self.sale = actor.sale
        encoder_sources = [
            (encoder.zs1.weight,), (encoder.zs1.bias,), (encoder.zs2.weight,), (encoder.zs2.bias,),
            (encoder.zs3.weight,), (encoder.zs3.bias,),
        ]
        self._sources = (encoder_sources if self.sale else []) + [
            (actor.l0.weight,), (actor.l0.bias,),
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

    def __call__(self, state):
        w0, b0, wl1, bl1, wl2, bl2, wh, bh = self._views[-8:]
        a = state @ w0 + b0
        a /= np.maximum(np.abs(a).mean(-1, keepdims=True), 1e-8)
        if self.sale:
            w1, b1, w2, b2, w3, b3 = self._views[:6]
            zs = state @ w1 + b1
            zs = np.where(zs > 0, zs, np.expm1(np.minimum(zs, 0)))
            zs = zs @ w2 + b2
            zs = np.where(zs > 0, zs, np.expm1(np.minimum(zs, 0)))
            zs = zs @ w3 + b3
            zs /= np.maximum(np.abs(zs).mean(-1, keepdims=True), 1e-8)
            a = np.concatenate([a, zs], 1)
        a = np.maximum(a @ wl1 + bl1, 0)
        a = np.maximum(a @ wl2 + bl2, 0)
        out = a @ wh + bh
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (np.tanh(out[:, self.action_dim :]) + 1)
        return np.tanh(out[:, : self.action_dim]), log_std


def gaussian_kl(mu_old, sigma_old, mu, log_std):
    """Per-state KL(old || new) of diagonal Gaussians, split into its mean and scale terms."""
    mean_term = (0.5 * ((mu_old - mu) / log_std.exp()).square()).sum(-1)
    ratio = sigma_old.log() - log_std
    scale_term = (0.5 * (2 * ratio).exp() - ratio - 0.5).sum(-1)
    return mean_term, scale_term


class TD7Agent:
    def __init__(self, state_dim, action_dim, args: Args, device, writer: SummaryWriter):
        assert 0 <= args.trace_lambda <= 1 and args.trace_horizon > 0
        assert args.num_envs > 1, "lane 0 heldout requires at least two environments"
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        adam = dict(fused=True)

        actor_zs_dim = args.zs_dim
        self.actor = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.critic = AnalyticLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_target = AnalyticLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder_target = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        targets = [
            (self.critic_target, self.critic),
            (self.fixed_encoder, self.encoder),
            (self.fixed_encoder_target, self.encoder),
        ]
        targets.append((self.actor_target, self.actor))
        for target, source in targets:
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, **adam)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr, **adam)
        # (destination, source) pairs of every hard update, in TD7's order (target before fixed).
        self._hard_updates = [
            (list(self.critic_target.parameters()), list(self.critic.parameters())),
            (list(self.fixed_encoder_target.parameters()), list(self.fixed_encoder.parameters())),
            (list(self.fixed_encoder.parameters()), list(self.encoder.parameters())),
        ]
        optimizers = [self.encoder_optimizer, self.critic_optimizer]

        self.target_entropy = action_dim * float(np.log(args.target_sigma))
        self._hard_updates.insert(0, (list(self.actor_target.parameters()), list(self.actor.parameters())))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
        optimizers += [self.actor_optimizer, self.alpha_optimizer]

        # Fresh-rollout rows retain both the clipped environment action and its pre-action latent.
        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (("state", s), ("action", a), ("mu", a), ("sigma", a), ("eps", a), ("pre_action", a), ("next_state", s), ("reward", 1), ("not_done", 1), ("boundary", 1), ("target", 1)):
            self._fields[name] = slice(offset, offset + width)
            offset += width
        self.row_dim = offset
        self.rollout_size = args.num_envs * args.num_steps
        self.priority = torch.ones(self.rollout_size, device=device)
        self.fitting_mask = (torch.arange(self.rollout_size, device=device) % args.num_envs != 0).float()
        self.fitting_count = args.num_steps * (args.num_envs - 1)
        self.last = torch.full((1,), self.rollout_size - 1, dtype=torch.long, device=device)
        self.updates_per_rollout = round(self.rollout_size * args.updates_per_step)
        self.staging = torch.zeros(self.rollout_size, self.row_dim, pin_memory=True)
        self.staging_np = self.staging.numpy()
        self.fresh = torch.zeros(self.rollout_size, self.row_dim, device=device)  # the latest rollout

        f32 = dict(device=device, dtype=torch.float32)
        self.max_priority = torch.ones((), **f32)
        self.running_max = torch.full((), -1e8, **f32)
        self.running_min = torch.full((), 1e8, **f32)
        self.max_target = torch.zeros((), **f32)
        self.min_target = torch.zeros((), **f32)
        self.metric_names = [
            "encoder_loss", "critic_loss", "q_values", "point_r2", "point_residual_std",
            "actor_loss", "sigma_mean", "sigma_min", "sigma_max", "distribution_gap", "importance_mean",
        ]
        self.metrics = torch.zeros(len(self.metric_names), **f32)

        maybe_compile = (
            (lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}))
            if args.compile
            else (lambda fn: fn)
        )
        self.continuation_mu = torch.zeros(self.rollout_size, action_dim, device=device)
        self.continuation_sigma = torch.zeros_like(self.continuation_mu)
        self.initial_raw_targets = torch.zeros(self.rollout_size, 1, device=device)
        self.latest_raw_targets = torch.zeros_like(self.initial_raw_targets)
        self.refresh_metrics = torch.zeros(3, device=device)
        self._behavior_continuation = maybe_compile(self._behavior_continuation_fn)
        self._lambda_targets = maybe_compile(self._lambda_targets_fn)
        self._encoder_loss = maybe_compile(self._encoder_loss_fn)
        self._critic_loss = maybe_compile(self._critic_loss_fn)
        self._actor_loss = maybe_compile(self._actor_loss_fn)
        self._entropy = maybe_compile(self._entropy_fn)
        # TD7's delayed actor cadence, using only current-rollout minibatches.
        steps = [self._make_step(update_actor=False), self._make_step(update_actor=True)]
        self.graphs = capture_steps(
            steps,
            optimizers=optimizers,
            state_tensors=[self.priority, self.max_priority, self.running_max, self.running_min, self.metrics],
        )

        self.policy = HostPolicy(self.fixed_encoder, self.actor)
        self.policy.refresh()
        self.training_steps = 0

    # ---- fresh rollout -------------------------------------------------------------------------------
    def stage(self, step, **values):
        """Write one vector-env step: every value is (num_envs, width)."""
        rows = self.staging_np[step * self.args.num_envs : (step + 1) * self.args.num_envs]
        for name, value in values.items():
            rows[:, self._fields[name]] = value

    def flush(self, warmup=False, global_step=0):
        """Cache behavior continuation before fit; targets refresh at hard updates."""
        self.fresh.copy_(self.staging)
        self.priority.fill_(1.0)
        self.max_priority.fill_(1.0)
        with torch.no_grad():
            mu, sigma = self._behavior_continuation(self.fresh, warmup)
            self.continuation_mu.copy_(mu)
            self.continuation_sigma.copy_(sigma)
            raw = self._lambda_targets(self.fresh)
            self.initial_raw_targets.copy_(raw)
            self.latest_raw_targets.copy_(raw)
            self.fresh[:, self._fields["target"]].copy_(raw)
            self.refresh_metrics.zero_()
        self.log_heldout(global_step, "prefit")

    def _behavior_continuation_fn(self, rollout, warmup):
        f, args = self._fields, self.args
        if warmup:
            mu = torch.zeros_like(rollout[:, f["mu"]])
            sigma = torch.full_like(mu, args.warmup_sigma)
        else:
            next_state = rollout[:, f["next_state"]]
            mu, log_std = self.actor(next_state, self.fixed_encoder.zs(next_state))
            sigma = log_std.exp()
        return mu, sigma

    def _lambda_targets_fn(self, rollout):
        f, args = self._fields, self.args
        next_state = rollout[:, f["next_state"]]
        mu, sigma = self.continuation_mu, self.continuation_sigma
        # Behavior coordinates are cached before fitting, even after actor/features
        # move. Only target critic/paired encoder and TD7 bounds are refreshed.
        zs = self.fixed_encoder_target.zs(next_state)
        q = self.critic_target(next_state, mu, sigma, zs)
        # Exact integration holds separately per twin. TD7 retains its original
        # min-after-expectation backup; this is not E[min(q1(s,a), q2(s,a))].
        value = q.min(1, keepdim=True)[0].clamp(self.min_target, self.max_target)
        shape = (args.num_steps, args.num_envs, 1)
        targets = lambda_returns(
            rollout[:, f["reward"]].view(shape), rollout[:, f["not_done"]].view(shape),
            rollout[:, f["boundary"]].view(shape), value.view(shape), args.gamma, args.trace_lambda,
            args.trace_horizon,
        )
        return targets.reshape(-1, 1)

    # ---- losses -------------------------------------------------------------------------------
    def _encoder_loss_fn(self, batch):
        f = self._fields
        with torch.no_grad():
            next_zs = self.encoder.zs(batch[:, f["next_state"]])
        pred_zs = self.encoder.zsa(self.encoder.zs(batch[:, f["state"]]), batch[:, f["action"]])
        return F.mse_loss(pred_zs, next_zs)

    def _critic_loss_fn(self, batch, importance):
        f = self._fields
        state, action = batch[:, f["state"]], batch[:, f["action"]]
        with torch.no_grad():
            target = batch[:, f["target"]]
            fixed_zs = self.fixed_encoder.zs(state)
        coefficients = self.critic.coefficients(state, fixed_zs)
        point = analytic_basis_value(coefficients, action, torch.zeros_like(action),
                                     self.critic.knots, self.critic.quadratic_indices)
        residual = point - target
        # Shared full-rollout LAP sampling, exact inverse probability on fitting
        # rows only: E[mask/(N_fit*p_i)*residual²] is the uniform fitting MSE.
        loss = (importance.unsqueeze(1) * residual.square()).sum(1).mean()
        with torch.no_grad():
            center = analytic_basis_value(coefficients, batch[:, f["mu"]], batch[:, f["sigma"]],
                                          self.critic.knots, self.critic.quadratic_indices)
            r2 = 1 - residual.square().mean() / (target - target.mean()).square().mean().clamp(min=1e-8)
            diagnostics = torch.stack([point.mean(), r2, residual.std(unbiased=False),
                                       target.max(), target.min(), (point - center).mean(), importance.mean()])
        return loss, residual.detach().abs(), diagnostics, fixed_zs

    def _actor_loss_fn(self, state, fixed_zs, alpha):
        mu, log_std = self.actor(state, fixed_zs)
        q = self.critic(state, mu, log_std.exp(), fixed_zs)
        return -q.mean() - alpha * log_std.sum(-1).mean(), log_std

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
            batch = self.fresh.index_select(0, ind)
            probability = (self.priority.index_select(0, ind) / csum.gather(0, self.last)).detach()
            fitting = self.fitting_mask.index_select(0, ind)
            importance = (fitting / (self.fitting_count * probability)).detach()

            self.encoder_optimizer.zero_grad(set_to_none=False)
            encoder_loss = self._encoder_loss(batch)
            encoder_loss.backward()
            self.encoder_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=False)
            critic_loss, td, diagnostics, fixed_zs = self._critic_loss(batch, importance)
            critic_loss.backward()
            self.critic_optimizer.step()

            with torch.no_grad():
                torch.maximum(self.running_max, diagnostics[3], out=self.running_max)
                torch.minimum(self.running_min, diagnostics[4], out=self.running_min)
                priority = td.max(1)[0].clamp(min=args.min_priority).pow(args.lap_alpha)
                self.priority[ind] = torch.where(fitting.bool(), priority, torch.ones_like(priority))
                torch.maximum(self.max_priority, priority.max(), out=self.max_priority)
                self.metrics[0].copy_(encoder_loss.detach())
                self.metrics[1].copy_(critic_loss.detach())
                self.metrics[2:5].copy_(diagnostics[:3])
                self.metrics[9:11].copy_(diagnostics[5:7])

            if update_actor:
                state = batch[:, self._fields["state"]]
                self.actor_optimizer.zero_grad(set_to_none=False)
                actor_loss, log_std = self._actor_loss(state, fixed_zs, self.log_alpha.detach().exp())
                actor_loss.backward(inputs=list(self.actor.parameters()))
                self.actor_optimizer.step()
                entropy = self._entropy(state, fixed_zs)
                self.alpha_optimizer.zero_grad(set_to_none=False)
                alpha_loss = -(self.log_alpha.exp() * (self.target_entropy - entropy)).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    sigma = log_std.detach().exp()
                    self.metrics[5:9].copy_(torch.stack([actor_loss.detach(), sigma.mean(), sigma.min(), sigma.max()]))

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
        update_actor = self.training_steps % self.args.policy_freq == 0
        self.graphs[int(update_actor)].replay()
        if self.training_steps % self.args.target_update_rate == 0:
            self.hard_update()
            with torch.no_grad():
                raw = self._lambda_targets(self.fresh)
                self.latest_raw_targets.copy_(raw)
                clean = raw
                previous = self.fresh[:, self._fields["target"]]
                self.refresh_metrics[0].add_(1)
                self.refresh_metrics[1].add_((clean - previous).square().mean())
                self.refresh_metrics[2].copy_((raw - self.initial_raw_targets).square().mean())
                previous.copy_(clean)
        if self.training_steps % 500 == 0:
            values = dict(zip(self.metric_names, self.metrics.tolist()))
            for name in ("encoder_loss", "critic_loss", "q_values", "point_r2", "point_residual_std", "distribution_gap", "importance_mean"):
                self.writer.add_scalar(f"losses/{name}", values[name], self.current_global_step)
            self.writer.add_scalar("charts/max_target", self.max_target.item(), self.current_global_step)
            self.writer.add_scalar("charts/min_target", self.min_target.item(), self.current_global_step)
            if self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.current_global_step)
                self.writer.add_scalar("losses/actor_loss", values["actor_loss"], self.current_global_step)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.current_global_step)

    # ---- one rollout of learning -------------------------------------------------------------
    def learn(self, global_step):
        self.current_global_step = global_step
        # Compare against the actual Gaussian that generated this staged rollout.
        state = self.fresh[:, self._fields["state"]]
        mu_old = self.fresh[:, self._fields["mu"]]
        sigma_old = self.fresh[:, self._fields["sigma"]]
        for _ in range(self.updates_per_rollout):
            self.train()
        with torch.no_grad():
            mu, log_std = self.actor(state, self.fixed_encoder.zs(state))
            mean_term, scale_term = gaussian_kl(mu_old, sigma_old, mu, log_std)
            stats = torch.stack([(mean_term + scale_term).mean(), mean_term.mean(), scale_term.mean()]).tolist()
        for tag, stat in zip(("actor/kl", "actor/kl_mean_term", "actor/kl_scale_term"), stats):
            self.writer.add_scalar(tag, stat, global_step)
        self.writer.add_scalar("actor/kl_state_count", self.rollout_size, global_step)
        self.writer.add_scalar("charts/optimizer_steps", self.training_steps, global_step)
        self.writer.add_scalar("critic_targets/mean", self.fresh[:, self._fields["target"]].mean().item(), global_step)
        self.writer.add_scalar("critic_targets/std", self.fresh[:, self._fields["target"]].std().item(), global_step)
        self.log_heldout(global_step, "postfit")
        self.writer.add_scalar("critic_refresh/final/initial_raw_mean", self.initial_raw_targets.mean().item(), global_step)
        self.writer.add_scalar("critic_refresh/final/final_raw_mean", self.latest_raw_targets.mean().item(), global_step)
        refresh_stats = self.refresh_metrics.tolist()
        self.writer.add_scalar("critic_refresh/count", refresh_stats[0], global_step)
        self.writer.add_scalar("critic_refresh/mean_increment_mse", refresh_stats[1] / max(refresh_stats[0], 1), global_step)
        self.writer.add_scalar("critic_refresh/final_raw_displacement_mse", refresh_stats[2], global_step)
        self.policy.refresh()


    def log_heldout(self, global_step, phase):
        """Lane-local action-credit residuals; no extra interactions or fitted labels.

        Lane 0 never fits critic coefficients, but does fit SALE and actor and its
        targets share fitted bootstraps and global target-bound updates. These
        are noisy moment checks, not an independent counterfactual oracle.
        SE uses horizon-sized temporal blocks;
        blocks can remain correlated through episodes and bootstrapping.
        """
        f, args = self._fields, self.args
        with torch.no_grad():
            rows = self.fresh.view(args.num_steps, args.num_envs, self.row_dim)[:, 0]
            state, action = rows[:, f["state"]], rows[:, f["action"]]
            q = self.critic(state, action, torch.zeros_like(action), self.fixed_encoder.zs(state)).mean(1, keepdim=True)
            residual = rows[:, f["target"]] - q
            eps, sigma = rows[:, f["eps"]], rows[:, f["sigma"]]
            # Raw Gaussian scores are valid for expectations of clipped actions,
            # including atoms; never divide by clipped-action density or reweight
            # behavior .1 to target .2. This measures residual mean/scale moments.
            moments = torch.cat([residual, residual * eps / sigma,
                                 residual * (eps.square() - 1) / sigma], 1)
            blocks = args.num_steps // args.trace_horizon
            block_means = moments[:blocks * args.trace_horizon].view(blocks, args.trace_horizon, -1).mean(1)
            mean = moments.mean(0)
            se = block_means.std(0, unbiased=True) / blocks**0.5 if blocks > 1 else torch.full_like(mean, float("nan"))
            values = torch.cat([mean, se, residual.square().mean().view(1), residual.std(unbiased=False).view(1)]).tolist()
        names = ["return_residual"] + [f"mean_score_{i}" for i in range(self.action_dim)] + [f"scale_score_{i}" for i in range(self.action_dim)]
        tags = [f"{name}/mean" for name in names] + [f"{name}/block_se" for name in names] + ["point_mse", "return_residual_std"]
        for tag, value in zip(tags, values):
            self.writer.add_scalar(f"critic_heldout/{phase}/{tag}", value, global_step)
        self.writer.add_scalar(f"critic_heldout/{phase}/rows", args.num_steps, global_step)
        self.writer.add_scalar(f"critic_heldout/{phase}/temporal_blocks", blocks, global_step)


def transition_observations(next_obs, boundaries, infos):
    """next_obs with autoreset rows replaced by the finished episode's final observation."""
    if not boundaries.any():
        return next_obs
    transition_obs = next_obs.copy()
    for index in np.flatnonzero(boundaries):
        transition_obs[index] = infos["final_observation"][index]
    return transition_obs


def evaluate(policy: HostPolicy, eval_envs, max_action):
    """Deterministic policy returns (the latent mean), one episode per env, in lockstep."""
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

    envs = make_mujoco_vector_env(args.env_id, args.num_envs, num_threads=args.env_threads)
    eval_envs = [gym.make(args.env_id) for _ in range(args.eval_eps)]
    for i, eval_env in enumerate(eval_envs):
        eval_env.reset(seed=args.seed + 100 + i)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    state_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    max_action = float(envs.single_action_space.high[0])
    shape = (args.num_envs, action_dim)

    agent = TD7Agent(state_dim, action_dim, args, device, writer)
    timer = PhaseTimer()
    warmup_mu = np.zeros(shape, dtype=np.float32)
    warmup_sigma = np.full(shape, args.warmup_sigma, dtype=np.float32)

    start_time = time.time()
    # One unrecorded horizon of warmup-policy draws spreads the episode ages across the envs.
    horizon = episode_horizon(args.env_id)
    phase_offsets = compute_phase_offsets(args.num_envs, horizon, args.seed)
    writer.add_text("initial_phase_offsets", ",".join(map(str, phase_offsets)))
    warm = run_phase_warmup(
        envs, obs_norm=None, horizon=horizon, phase_offsets=phase_offsets, seed=args.seed,
        act_fn=lambda _: np.clip(args.warmup_sigma * rng.standard_normal(shape, dtype=np.float32), -1, 1) * max_action,
    )
    obs = warm.next_obs
    global_step = warm.transitions
    suppress = warm.suppress_mask.copy()
    num_iterations = (args.total_timesteps - warm.transitions) // agent.rollout_size
    for iteration in range(1, num_iterations + 1):
        training = iteration > args.warmup_rollouts
        for step in range(args.num_steps):
            with timer.span("rollout", use_cuda=False):
                if training:
                    mu, log_std = agent.policy(obs)
                    sigma = np.exp(log_std)
                else:
                    mu, sigma = warmup_mu, warmup_sigma
                eps = rng.standard_normal(shape, dtype=np.float32)
                pre_action = mu + sigma * eps
                action = np.clip(pre_action, -1, 1)
            with timer.span("env", use_cuda=False):
                next_obs, reward, terminated, truncated, infos = envs.step(action * max_action)
            with timer.span("rollout", use_cuda=False):
                next_obs = next_obs.astype(np.float32)
                agent.stage(
                    step, state=obs, action=action, mu=mu, sigma=sigma, eps=eps, pre_action=pre_action,
                    next_state=transition_observations(next_obs, terminated | truncated, infos),
                    reward=reward[:, None], not_done=1.0 - terminated[:, None],
                    boundary=(terminated | truncated)[:, None],
                )
                obs = next_obs
            global_step += args.num_envs
            for index, info in enumerate(infos.get("final_info", ())):
                if info and "episode" in info:
                    if suppress[index]:
                        suppress[index] = False
                        continue
                    episode_return = float(info["episode"]["r"])
                    print(f"global_step={global_step}, episodic_return={episode_return:.3f}")
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

        with timer.span("update"):
            agent.flush(warmup=not training, global_step=global_step)
            if training:
                agent.learn(global_step)
        if iteration % args.eval_interval == 0 or iteration == num_iterations:
            with timer.span("eval", use_cuda=False):
                eval_return = evaluate(agent.policy, eval_envs, max_action)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(f"global_step={global_step}, eval_return={eval_return:.3f}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        for name, stats in timer.summary().items():
            writer.add_scalar(f"time/{name}_s", stats["total_s"], global_step)
        timer.reset()

    torch.save(dict(args=vars(args), actor=agent.actor.state_dict(), critic=agent.critic.state_dict(),
                    fixed_encoder=agent.fixed_encoder.state_dict(), global_step=global_step,
                    optimizer_steps=agent.training_steps), f"runs/{run_name}/checkpoint.pt")
    envs.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()

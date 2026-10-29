# TD7 LATENT-Q SNAPSHOT-CALIBRATED LAMBDA CV v19.
# Preserve latent-Q, lambda=.95/32-step horizon, heavy cadence and Huber/LAP.
# Unit-strength v18 controls added noise with negligible return covariance.
# Fit two control coefficients on a PREVIOUS collection; deploy them with that
# EXACT retained prefit critic/encoder snapshot on the NEXT collection.
# No same-collection coefficient fitting, coefficient clipping or label clipping.
# Initial coefficients are zero. Snapshot and coefficients are fixed before draws;
# only future innovations are subtracted, preserving conditional label means.
# Out-of-sample variance reduction is measured, never assumed.
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
    """mixing coefficient for frozen behavior-policy traces; 0 gives frozen one-step labels"""
    trace_horizon: int = 32
    """independent temporal trace segment length; interaction rollout remains 1024 steps"""
    hermite_order: int = 2
    """0: plain Q~ regression; 1: + sigma*eps*dQ~/dmu; 2: + (eps^2-1)*sigma/2*dQ~/dsigma"""
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
    """legacy target query std; unused by frozen behavior-policy labels"""
    alpha_lr: float = 1e-3
    """learning rate of the alpha (log_alpha) optimizer"""
    policy_freq: int = 2
    """the frequency of training the policy"""
    lap_alpha: float = 0.4
    """within-rollout LAP prioritization exponent"""
    min_priority: float = 1.0
    """LAP minimum priority (and Huber loss threshold)"""
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


def future_controls(innovations, not_done, boundaries, gamma, trace_lambda, trace_horizon=32):
    """Discount future innovations without including the current action's noise.

    Innovations are (time, env, order). Boundary masks broadcast over orders.
    Each lane/segment ends with zero control, including a short final segment.
    """
    assert innovations.shape[:2] == not_done.shape[:2] == boundaries.shape[:2]
    assert innovations.shape[0] > 0 and trace_horizon > 0
    length = innovations.shape[0]
    segments = (length + trace_horizon - 1) // trace_horizon
    padding = segments * trace_horizon - length
    cuts = boundaries.clone()
    cuts[-1] = 1

    def segmented(x, fill=0):
        if padding:
            x = torch.cat([x, x.new_full((padding,) + x.shape[1:], fill)])
        return x.reshape(segments, trace_horizon, *x.shape[1:]).transpose(0, 1)

    noise, mask, cut = segmented(innovations), segmented(not_done), segmented(cuts, 1)
    continuation = torch.zeros_like(noise[-1])
    controls = [continuation]
    for t in range(trace_horizon - 2, -1, -1):
        continuation = gamma * trace_lambda * mask[t] * (~cut[t].bool()) * (noise[t + 1] + continuation)
        controls.append(continuation)
    controls = torch.stack(controls[::-1]).transpose(0, 1)
    return controls.reshape(segments * trace_horizon, *innovations.shape[1:])[:length]


def clean_labels(raw, controls, strength):
    """Apply fixed controls without clipping or normalization."""
    return raw if strength == 0 else raw - strength * controls.sum(-1, keepdim=True)



def calibrate_controls(raw, center, controls):
    """Joint least-squares coefficients; deploy only on a later collection.

    Double precision and the numerical Moore-Penrose inverse handle collinear
    or absent orders without a tuned ridge, bounds, or a fitted intercept.
    Centering here estimates covariance; it never centers deployed innovations.
    """
    x = controls.double()
    y = (raw - center).double()
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    covariance = x.T @ x / len(x)
    cross_covariance = (x.T @ y / len(x)).squeeze(-1)
    coefficient = torch.linalg.pinv(covariance, hermitian=True) @ cross_covariance
    return coefficient.to(controls.dtype)

def paired_residual_metrics(raw, clean, center, controls):
    """Empirical paired residual dispersion; not an irreducible-noise estimate."""
    raw_residual, clean_residual = raw - center, clean - center
    total_control = controls.sum(-1, keepdim=True)
    raw_variance = raw_residual.var(unbiased=False)
    clean_variance = clean_residual.var(unbiased=False)
    covariance = ((raw_residual - raw_residual.mean()) * (total_control - total_control.mean())).mean()
    nan = raw_variance.new_full((), float("nan"))
    return dict(
        raw_residual_variance=raw_variance,
        clean_residual_variance=clean_variance,
        variance_ratio=torch.where(raw_variance > 0, clean_variance / raw_variance, nan),
        variance_reduction_fraction=torch.where(raw_variance > 0, 1 - clean_variance / raw_variance, nan),
        residual_control_covariance=covariance,
    )


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def avg_l1_norm_jvp(x, t, eps=1e-8):
    """avg_l1_norm and its forward-mode derivative along tangent t."""
    scale = x.abs().mean(-1, keepdim=True).clamp(min=eps)
    y = x / scale
    return y, (t - y * (x.sign() * t).mean(-1, keepdim=True)) / scale


def elu_jvp(x, t):
    return F.elu(x), torch.where(x > 0, t, t * x.clamp(max=0).exp())


def lap_huber(td_loss, min_priority=1.0):
    return torch.where(td_loss < min_priority, 0.5 * td_loss.pow(2), min_priority * td_loss).sum(1).mean()


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

    def zsa_jvp(self, zs, action, t_action):
        """zsa(zs, action) and its derivative along an action tangent (zs held fixed)."""
        h, t = elu_jvp(self.zsa1(torch.cat([zs, action], 1)), t_action @ self.zsa1.weight[:, self.zs_dim :].T)
        h, t = elu_jvp(self.zsa2(h), t @ self.zsa2.weight.T)
        return self.zsa3(h), t @ self.zsa3.weight.T


class EnsembleLinear(nn.Module):
    """``members`` independent nn.Linear layers applied to (members, batch, in) as one baddbmm."""

    def __init__(self, members, in_features, out_features):
        super().__init__()
        bound = in_features**-0.5  # nn.Linear's default init bound, for weight and bias
        self.weight = nn.Parameter(torch.empty(members, in_features, out_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(members, 1, out_features).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)

    def jvp(self, x, t):
        """Value and tangent in one bmm: the tangent rows get the weight but not the bias."""
        out = torch.bmm(torch.cat([x, t], 1), self.weight)
        return out[:, : x.shape[1]] + self.bias, out[:, x.shape[1] :]


class PreActionLatentTwinCritic(nn.Module):
    """Fused twin critic conditioned on (policy mean, policy std, pre-action latent)."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.hdim, self.state_dim = hdim, state_dim
        latent_dim = 3 * action_dim  # mean | std | pre-action latent
        self.q0 = nn.Linear(state_dim + latent_dim, 2 * hdim)  # both members side by side
        self.q1 = EnsembleLinear(2, 2 * zs_dim + hdim, hdim)
        self.q2 = EnsembleLinear(2, hdim, hdim)
        self.q3 = EnsembleLinear(2, hdim, 1)
        bound = (state_dim + latent_dim) ** -0.5  # per-member nn.Linear init
        nn.init.uniform_(self.q0.weight, -bound, bound)
        nn.init.uniform_(self.q0.bias, -bound, bound)

    def forward(self, state, latent, pre_action, zsa, zs, t_latent=None, t_pre_action=None, t_zsa=None):
        batch = state.shape[0]
        latent_input = torch.cat([latent, pre_action], 1)
        pre = self.q0(torch.cat([state, latent_input], 1)).view(batch, 2, self.hdim)
        embeddings = torch.cat([zsa, zs], 1).expand(2, batch, -1)
        if t_latent is None:
            h = avg_l1_norm(pre).transpose(0, 1)
            h = F.elu(self.q1(torch.cat([h, embeddings], 2)))
            h = F.elu(self.q2(h))
            return self.q3(h).squeeze(-1).transpose(0, 1)
        t_latent_input = torch.cat([t_latent, t_pre_action], 1)
        t_pre = (t_latent_input @ self.q0.weight[:, self.state_dim :].T).view(batch, 2, self.hdim)
        h, t = avg_l1_norm_jvp(pre, t_pre)
        t_embeddings = torch.cat([t_zsa, torch.zeros_like(zs)], 1).expand(2, batch, -1)
        h = torch.cat([h.transpose(0, 1), embeddings], 2)
        t = torch.cat([t.transpose(0, 1), t_embeddings], 2)
        h, t = elu_jvp(*self.q1.jvp(h, t))
        h, t = elu_jvp(*self.q2.jvp(h, t))
        q, t = self.q3.jvp(h, t)
        return q.squeeze(-1).transpose(0, 1), t.squeeze(-1).transpose(0, 1)


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
        assert args.hermite_order in (0, 1, 2), "hermite_order must be 0, 1 or 2"
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        adam = dict(fused=True)

        actor_zs_dim = args.zs_dim
        self.actor = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.critic = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_target = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
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
        self.cv_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.cv_encoder = copy.deepcopy(self.fixed_encoder).requires_grad_(False)
        self.cv_coefficients = torch.zeros(2, device=device)
        self.cv_calibration_step = 0
        self._cv_snapshot_updates = [
            (list(self.cv_critic.parameters()), list(self.critic.parameters())),
            (list(self.cv_encoder.parameters()), list(self.fixed_encoder.parameters())),
        ]
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
            "encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std",
            "actor_loss", "sigma_mean", "sigma_min", "sigma_max", "smoothing_gap", "preaction_gap",
        ]
        self.metrics = torch.zeros(len(self.metric_names), **f32)

        maybe_compile = (
            (lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}))
            if args.compile
            else (lambda fn: fn)
        )
        self._lambda_targets = maybe_compile(self._lambda_targets_fn)
        self._hermite_controls = maybe_compile(self._hermite_controls_fn)
        self._snapshot_controls = maybe_compile(self._snapshot_controls_fn)
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
        """Apply previously calibrated snapshot; prepare the next one before fitting."""
        self.fresh.copy_(self.staging)
        self.priority.fill_(1.0)
        self.max_priority.fill_(1.0)
        with torch.no_grad():
            raw = self._lambda_targets(self.fresh, warmup)
            # Deployed snapshot/coefficients were fixed before this collection.
            center, innovations, unscaled_controls = self._snapshot_controls(self.fresh)
            controls = unscaled_controls * self.cv_coefficients
            clean = clean_labels(raw, controls, 1.0)
            # Candidate online model was also unchanged throughout this collection.
            # Its fitted coefficients are NEVER applied to these labels.
            candidate_center, _, candidate_controls = self._hermite_controls(self.fresh)
            next_coefficients = calibrate_controls(raw, candidate_center, candidate_controls)
            self.fresh[:, self._fields["target"]].copy_(clean)
            metrics = paired_residual_metrics(raw, clean, center, controls)
            metrics.update(coefficient_order1=self.cv_coefficients[0].clone(),
                           coefficient_order2=self.cv_coefficients[1].clone(),
                           next_coefficient_order1=next_coefficients[0],
                           next_coefficient_order2=next_coefficients[1],
                           calibration_step=raw.new_tensor(self.cv_calibration_step),
                           candidate_control_std=candidate_controls.sum(-1).std(unbiased=False))
            metrics.update(raw_target_std=raw.std(unbiased=False), clean_target_std=clean.std(unbiased=False),
                           control_shift_mean=(raw - clean).mean(),
                           control_total_mean=controls.sum(-1).mean(),
                           control_total_std=controls.sum(-1).std(unbiased=False))
            for order in range(2):
                metrics[f"innovation_order{order + 1}_mean"] = innovations[:, order].mean()
                metrics[f"innovation_order{order + 1}_std"] = innovations[:, order].std(unbiased=False)
                metrics[f"future_control_order{order + 1}_mean"] = controls[:, order].mean()
                metrics[f"future_control_order{order + 1}_std"] = controls[:, order].std(unbiased=False)
            shape = (self.args.num_steps, self.args.num_envs, -1)
            time_raw, time_clean, time_center, time_controls = [x.view(shape) for x in (raw, clean, center, controls)]
            for position in range(min(self.args.trace_horizon, self.args.num_steps)):
                selected = slice(position, None, self.args.trace_horizon)
                position_metrics = paired_residual_metrics(time_raw[selected], time_clean[selected],
                                                           time_center[selected], time_controls[selected])
                metrics.update({f"position_{position:02d}/{name}": value for name, value in position_metrics.items()})
            # Carry the exact candidate predictor with its coefficients. Updating
            # either model without recalibration would break scale matching.
            for destinations, sources in self._cv_snapshot_updates:
                torch._foreach_copy_(destinations, sources)
            self.cv_coefficients.copy_(next_coefficients)
            self.cv_calibration_step = global_step
            # One batched device-to-host read per upload, never inside the fitting path.
            values = torch.stack(list(metrics.values())).tolist()
            for name, value in zip(metrics, values):
                self.writer.add_scalar(f"critic_cv/{name}", value, global_step)

    def _hermite_controls_fn(self, rollout):
        return self._controls_for(rollout, self.critic, self.fixed_encoder)

    def _snapshot_controls_fn(self, rollout):
        return self._controls_for(rollout, self.cv_critic, self.cv_encoder)

    def _controls_for(self, rollout, critic, encoder):
        f, args = self._fields, self.args
        state, mu, sigma, eps = [rollout[:, f[name]] for name in ("state", "mu", "sigma", "eps")]
        zs = encoder.zs(state)
        t_mu = sigma * eps
        t_sigma = 0.5 * sigma * (eps.square() - 1)
        zeros = torch.zeros_like(mu)
        zs2 = torch.cat([zs, zs])
        mu2 = torch.cat([mu, mu])
        zsa2, t_zsa2 = encoder.zsa_jvp(zs2, mu2, torch.cat([t_mu, zeros]))
        latent = torch.cat([mu, sigma], 1)
        q, tangents = critic(
            torch.cat([state, state]), torch.cat([latent, latent]), mu2, zsa2, zs2,
            torch.cat([torch.cat([t_mu, zeros], 1), torch.cat([zeros, t_sigma], 1)]),
            torch.cat([t_mu, zeros]), t_zsa2,
        )
        count = state.shape[0]
        # Twin-mean matches actor-facing center. Neither center query uses eps/action.
        center = q[:count].mean(1, keepdim=True)
        innovations = torch.stack([tangents[:count].mean(1), tangents[count:].mean(1)], 1)
        shape = (args.num_steps, args.num_envs, -1)
        controls = future_controls(innovations.view(shape), rollout[:, f["not_done"]].view(args.num_steps, args.num_envs, 1),
                                   rollout[:, f["boundary"]].view(args.num_steps, args.num_envs, 1),
                                   args.gamma, args.trace_lambda, args.trace_horizon)
        return center, innovations, controls.reshape(-1, 2)

    def _lambda_targets_fn(self, rollout, warmup):
        f, args = self._fields, self.args
        next_state = rollout[:, f["next_state"]]
        if warmup:
            mu = torch.zeros_like(rollout[:, f["mu"]])
            sigma = torch.full_like(mu, args.warmup_sigma)
        else:
            mu, log_std = self.actor(next_state, self.fixed_encoder.zs(next_state))
            sigma = log_std.exp()
        # Behavior distribution uses current actor features; critic features stay
        # paired with its frozen target encoder. No Gaussian integration is sampled.
        zs = self.fixed_encoder_target.zs(next_state)
        zsa = self.fixed_encoder_target.zsa(zs, mu)
        q = self.critic_target(next_state, torch.cat([mu, sigma], 1), mu, zsa, zs)
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

    def _critic_loss_fn(self, batch, min_target, max_target):
        f, args = self._fields, self.args
        state = batch[:, f["state"]]
        action, mu, sigma, eps = batch[:, f["action"]], batch[:, f["mu"]], batch[:, f["sigma"]], batch[:, f["eps"]]
        pre_action = batch[:, f["pre_action"]]
        with torch.no_grad():
            q_target = batch[:, f["target"]]
            # Three queries share one critic and one 3B pass: distribution center (mu, sigma, mu)
            # with Hermite tangent, observed pre-action latent (mu, sigma, u), and point action
            # (a, 0, a). The latter two get direct TD labels; center and point coincide at sigma=0.
            fixed_zs = self.fixed_encoder.zs(state)
            t_mu = sigma * eps if args.hermite_order >= 1 else torch.zeros_like(mu)
            t_sigma = 0.5 * sigma * (eps.square() - 1) if args.hermite_order >= 2 else torch.zeros_like(sigma)
            zs3 = torch.cat([fixed_zs, fixed_zs, fixed_zs])
            zsa3, t_zsa3 = self.fixed_encoder.zsa_jvp(
                zs3, torch.cat([mu, action, action]), torch.cat([t_mu, torch.zeros_like(action), torch.zeros_like(action)])
            )
            latent3 = torch.cat([
                torch.cat([mu, sigma], 1),
                torch.cat([mu, sigma], 1),
                torch.cat([action, torch.zeros_like(sigma)], 1),
            ])
            pre_actions3 = torch.cat([mu, pre_action, action])
            t_pre_actions3 = torch.cat([t_mu, torch.zeros_like(action), torch.zeros_like(action)])
            t_latent3 = torch.cat([
                torch.cat([t_mu, t_sigma], 1),
                torch.zeros_like(latent3[: len(mu)]),
                torch.zeros_like(latent3[: len(mu)]),
            ])
        states3 = torch.cat([state, state, state])
        q3, tangent3 = self.critic(
            states3, latent3, pre_actions3, zsa3, zs3, t_latent3, t_pre_actions3, t_zsa3
        )
        batch_size = state.shape[0]
        q_dist, tangent = q3[:batch_size], tangent3[:batch_size]
        q_preaction, q_action = q3[batch_size : 2 * batch_size], q3[2 * batch_size :]
        td_latent = (q_dist + tangent - q_target).abs()
        td_preaction = (q_preaction - q_target).abs()
        td_action = (q_action - q_target).abs()
        loss = (
            lap_huber(td_latent, args.min_priority)
            + lap_huber(td_preaction, args.min_priority)
            + lap_huber(td_action, args.min_priority)
        ) / 3
        with torch.no_grad():
            # Fraction of the realized-noise variation in y that the Hermite terms explain.
            r2 = 1 - td_latent.square().mean() / (q_target - q_dist).square().mean().clamp(min=1e-8)
            smoothing_gap = (q_action - q_dist).mean()
            preaction_gap = (q_preaction - q_action).mean()
            diagnostics = torch.stack([
                q_dist.mean(), r2, tangent.std(), q_target.max(), q_target.min(), smoothing_gap, preaction_gap
            ])
        td = torch.stack([td_latent, td_preaction, td_action], dim=0).amax(0)
        return loss, td.detach(), diagnostics, fixed_zs

    def _actor_loss_fn(self, state, fixed_zs, alpha):
        mu, log_std = self.actor(state, fixed_zs)
        latent = torch.cat([mu, log_std.exp()], 1)
        q = self.critic(state, latent, mu, self.fixed_encoder.zsa(fixed_zs, mu), fixed_zs)
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

            self.encoder_optimizer.zero_grad(set_to_none=False)
            encoder_loss = self._encoder_loss(batch)
            encoder_loss.backward()
            self.encoder_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=False)
            critic_loss, td, diagnostics, fixed_zs = self._critic_loss(batch, self.min_target, self.max_target)
            critic_loss.backward()
            self.critic_optimizer.step()

            with torch.no_grad():
                torch.maximum(self.running_max, diagnostics[3], out=self.running_max)
                torch.minimum(self.running_min, diagnostics[4], out=self.running_min)
                priority = td.max(1)[0].clamp(min=args.min_priority).pow(args.lap_alpha)
                self.priority[ind] = priority
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
        if self.training_steps % 500 == 0:
            values = dict(zip(self.metric_names, self.metrics.tolist()))
            for name in ("encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std", "smoothing_gap", "preaction_gap"):
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
        self.policy.refresh()


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
                    cv_critic=agent.cv_critic.state_dict(), cv_encoder=agent.cv_encoder.state_dict(),
                    cv_coefficients=agent.cv_coefficients, cv_calibration_step=agent.cv_calibration_step,
                    optimizer_steps=agent.training_steps), f"runs/{run_name}/checkpoint.pt")
    envs.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()

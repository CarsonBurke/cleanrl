# TD7 LATENT-Q PRE-ACTION ON-POLICY v8 — v5's critic, PPO's data cadence, one actor step per rollout.
#
# The critic Q~(s, mu, sigma, u) and its replay/LAP/SALE training are v5's, unchanged. Data comes from
# 16 staggered envs x 1024 steps. actor_mode=trust_region: after the critic's replay updates, the
# actor takes ONE full-batch step on the fresh rollout's states: the natural gradient of the mean
# Q~(s, mu(s), sigma(s), u=mu), with the Gaussian Fisher at the behavior policy as the metric, scaled
# so the measured KL(behavior || new) equals target_kl. The Hermite tangents fit Q~ at the behavior
# (mu, sigma), so that KL ball is the region where the critic is a fitted model. The alpha autotuner
# (a Lagrangian solved over ~10^5 SGD steps) becomes the exact two-constraint step: KL = target_kl and
# linearized entropy >= target. The actor is state-only: a SALE input would let the policy drift on
# every encoder swap between actor steps. actor_mode=replay keeps v5's actor (cadence-only control).
# Hypothesis: with pathwise critic gradients, a few hundred KL-bounded steps beat PPO's clipped
# surrogate at equal rollouts; the gap to v5 measures what the actor's step count buys.
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
    actor_mode: str = "trust_region"
    """trust_region: one KL-sized natural-gradient step per rollout; replay: v5's delayed replay actor"""
    actor_sale: bool = False
    """feed the fixed SALE state embedding to the actor (v5's actor; replay mode only)"""
    target_kl: float = 0.03
    """mean KL(behavior || new) of the trust-region step (sepclip PPO measures 0.02-0.04 per rollout)"""
    cg_iters: int = 10
    """conjugate-gradient iterations per Fisher solve"""
    cg_damping: float = 0.1
    """Tikhonov damping of the Fisher solves"""
    entropy_budget: float = 0.5
    """largest share of the KL radius an active entropy constraint may spend"""
    updates_per_step: float = 1.0
    """replay updates per environment step"""
    warmup_rollouts: int = 1
    """recorded rollouts of latent-free exploration (mu=0, sigma=warmup_sigma) before training starts"""
    warmup_sigma: float = 0.6
    """std of the Gaussian warmup policy (the TD7 uniform warmup has no Gaussian latent; 0.6 ~ its std)"""
    hermite_order: int = 2
    """0: plain Q~ regression; 1: + sigma*eps*dQ~/dmu; 2: + (eps^2-1)*sigma/2*dQ~/dsigma"""
    eval_interval: int = 2
    """evaluate the policy mean every N rollouts"""
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
    """learning rate of the alpha (log_alpha) optimizer (replay mode)"""
    policy_freq: int = 2
    """the frequency of training the policy (replay mode)"""
    lap_alpha: float = 0.4
    """LAP prioritization exponent"""
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


def conjugate_gradient(matvec, b, iters):
    x, r, p = torch.zeros_like(b), b.clone(), b.clone()
    rs = r @ r
    for _ in range(iters):
        ap = matvec(p)
        step = rs / (p @ ap)
        x += step * p
        r -= step * ap
        rs_next = r @ r
        p = r + (rs_next / rs) * p
        rs = rs_next
    return x


def gaussian_kl(mu_old, sigma_old, mu, log_std):
    """Per-state KL(old || new) of diagonal Gaussians, split into its mean and scale terms."""
    mean_term = (0.5 * ((mu_old - mu) / log_std.exp()).square()).sum(-1)
    ratio = sigma_old.log() - log_std
    scale_term = (0.5 * (2 * ratio).exp() - ratio - 0.5).sum(-1)
    return mean_term, scale_term


class TD7Agent:
    def __init__(self, state_dim, action_dim, args: Args, device, writer: SummaryWriter):
        assert args.hermite_order in (0, 1, 2), "hermite_order must be 0, 1 or 2"
        assert args.actor_mode in ("trust_region", "replay"), "actor_mode must be trust_region or replay"
        # A SALE actor's outputs move with every fixed-encoder swap; one step per rollout cannot track that.
        assert not (args.actor_sale and args.actor_mode == "trust_region"), "actor_sale requires actor_mode=replay"
        self.trust_region = args.actor_mode == "trust_region"
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        adam = dict(fused=True)

        actor_zs_dim = args.zs_dim if args.actor_sale else 0
        self.actor = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        # The trust-region actor is constant across a rollout's critic updates: it is its own target.
        self.actor_target = (
            self.actor if self.trust_region else Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        )
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
        if not self.trust_region:
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

        # Entropy is confined to the actor update, which holds the mean log-sigma at log(target_sigma):
        # the alpha autotuner in replay mode, an explicit constraint of the trust-region step otherwise.
        self.target_entropy = action_dim * float(np.log(args.target_sigma))
        if not self.trust_region:
            self._hard_updates.insert(0, (list(self.actor_target.parameters()), list(self.actor.parameters())))
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
            self.log_alpha = torch.zeros((), device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
            optimizers += [self.actor_optimizer, self.alpha_optimizer]

        # Replay rows retain both the clipped environment action and its pre-action latent.
        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (("state", s), ("action", a), ("mu", a), ("sigma", a), ("eps", a), ("pre_action", a), ("next_state", s), ("reward", 1), ("not_done", 1)):
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
        self.rollout_size = args.num_envs * args.num_steps
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
        self._encoder_loss = maybe_compile(self._encoder_loss_fn)
        self._critic_loss = maybe_compile(self._critic_loss_fn)
        self._actor_loss = maybe_compile(self._actor_loss_fn)
        self._entropy = maybe_compile(self._entropy_fn)
        # Replay mode: training_steps % policy_freq == 0 selects the actor-step graph (TD7's delayed update).
        steps = [self._make_step(update_actor=False)]
        if not self.trust_region:
            steps.append(self._make_step(update_actor=True))
        self.graphs = capture_steps(
            steps,
            optimizers=optimizers,
            state_tensors=[self.priority, self.max_priority, self.running_max, self.running_min, self.metrics],
        )

        self.policy = HostPolicy(self.fixed_encoder, self.actor)
        self.policy.refresh()
        self.training_steps = 0

    # ---- replay -------------------------------------------------------------------------------
    def stage(self, step, **values):
        """Write one vector-env step: every value is (num_envs, width)."""
        rows = self.staging_np[step * self.args.num_envs : (step + 1) * self.args.num_envs]
        for name, value in values.items():
            rows[:, self._fields[name]] = value

    def flush(self):
        """Upload the staged rollout; new rows get the current max priority, as in LAP's add()."""
        n = self.rollout_size
        self.fresh.copy_(self.staging)
        index = (self.ptr + torch.arange(n, device=self.device)) % self.args.buffer_size
        self.storage.index_copy_(0, index, self.fresh)
        self.priority.index_put_((index,), self.max_priority.expand(n))
        self.ptr = (self.ptr + n) % self.args.buffer_size
        self.size = min(self.size + n, self.args.buffer_size)
        self.last.fill_(self.size - 1)

    # ---- losses -------------------------------------------------------------------------------
    def _encoder_loss_fn(self, batch):
        f = self._fields
        with torch.no_grad():
            next_zs = self.encoder.zs(batch[:, f["next_state"]])
        pred_zs = self.encoder.zsa(self.encoder.zs(batch[:, f["state"]]), batch[:, f["action"]])
        return F.mse_loss(pred_zs, next_zs)

    def _critic_loss_fn(self, batch, min_target, max_target):
        f, args = self._fields, self.args
        state, next_state = batch[:, f["state"]], batch[:, f["next_state"]]
        action, mu, sigma, eps = batch[:, f["action"]], batch[:, f["mu"]], batch[:, f["sigma"]], batch[:, f["eps"]]
        pre_action = batch[:, f["pre_action"]]
        with torch.no_grad():
            # Deterministic center query at TD7's target smoothing std; no smoothing action is sampled.
            next_zs = self.fixed_encoder_target.zs(next_state)
            next_mu, _ = self.actor_target(next_state, next_zs)
            next_zsa = self.fixed_encoder_target.zsa(next_zs, next_mu)
            next_latent = torch.cat([next_mu, torch.full_like(next_mu, args.target_smoothing_sigma)], 1)
            q_target = self.critic_target(next_state, next_latent, next_mu, next_zsa, next_zs)
            q_target = q_target.min(1, keepdim=True)[0].clamp(min_target, max_target)
            q_target = batch[:, f["reward"]] + batch[:, f["not_done"]] * args.gamma * q_target

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
        update_actor = not self.trust_region and self.training_steps % self.args.policy_freq == 0
        self.graphs[int(update_actor)].replay()
        if self.training_steps % self.args.target_update_rate == 0:
            self.hard_update()
        if self.training_steps % 500 == 0:
            values = dict(zip(self.metric_names, self.metrics.tolist()))
            for name in ("encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std", "smoothing_gap", "preaction_gap"):
                self.writer.add_scalar(f"losses/{name}", values[name], self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target.item(), self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target.item(), self.training_steps)
            if not self.trust_region and self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
                self.writer.add_scalar("losses/actor_loss", values["actor_loss"], self.training_steps)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.training_steps)

    # ---- one rollout of learning -------------------------------------------------------------
    def learn(self, global_step):
        if self.trust_region:
            for _ in range(self.updates_per_rollout):
                self.train()
            self.trust_region_step(global_step)
        else:
            # How far a rollout of replay updates moves the policy on the states it just visited, in
            # the trust-region step's units (the SALE actor also moves with the fixed encoder).
            state = self.fresh[:, self._fields["state"]]
            with torch.no_grad():
                mu_old, log_std_old = self.actor(state, self.fixed_encoder.zs(state))
            for _ in range(self.updates_per_rollout):
                self.train()
            with torch.no_grad():
                mu, log_std = self.actor(state, self.fixed_encoder.zs(state))
                mean_term, scale_term = gaussian_kl(mu_old, log_std_old.exp(), mu, log_std)
                stats = torch.stack([(mean_term + scale_term).mean(), mean_term.mean(), scale_term.mean()]).tolist()
            for tag, stat in zip(("actor/kl", "actor/kl_mean_term", "actor/kl_scale_term"), stats):
                self.writer.add_scalar(tag, stat, global_step)
        self.policy.refresh()

    def trust_region_step(self, global_step):
        """max g_Q.d  s.t.  KL(behavior || theta + d) = target_kl,  entropy(theta + d) >= target_entropy.

        Solved in span{F^-1 g_Q, F^-1 g_H} with the linearized entropy, then rescaled to the measured KL.
        """
        args = self.args
        state = self.fresh[:, self._fields["state"]]
        names = [name for name, _ in self.actor.named_parameters()]
        theta = list(self.actor.parameters())
        sizes = [p.numel() for p in theta]
        count = state.shape[0]

        def flatten(tensors):
            return torch.cat([t.reshape(-1) for t in tensors])

        def unflatten(vector):
            return tuple(v.view_as(p) for v, p in zip(vector.split(sizes), theta))

        def policy(params):
            return torch.func.functional_call(self.actor, dict(zip(names, params)), (state, None))

        def value(mu, log_std, fixed_zs):
            latent = torch.cat([mu, log_std.exp()], 1)
            return self.critic(state, latent, mu, self.fixed_encoder.zsa(fixed_zs, mu), fixed_zs).mean()

        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(state)
        mu, log_std = self.actor(state, None)
        q = value(mu, log_std, fixed_zs)
        entropy = log_std.sum(-1).mean()
        g_q = flatten(torch.autograd.grad(q, theta, retain_graph=True))
        # The mean head does not reach the entropy: its gradient materializes as zeros.
        g_h = flatten(torch.autograd.grad(entropy, theta, retain_graph=True, allow_unused=True, materialize_grads=True))
        # The behavior policy is this actor's host mirror; the device outputs are the exact reference.
        mu_old, sigma_old = mu.detach(), log_std.detach().exp()

        # Fisher of the behavior Gaussian in (mu, log sigma): v'Fv/2 is the mean KL to second order.
        detached = tuple(p.detach() for p in theta)

        def fisher(vector):
            _, (t_mu, t_log_std) = torch.func.jvp(policy, (detached,), (unflatten(vector),))
            weighted = ((mu * (t_mu / sigma_old.square())).sum() + (log_std * (2 * t_log_std)).sum()) / count
            return flatten(torch.autograd.grad(weighted, theta, retain_graph=True))

        def damped(vector):
            return fisher(vector) + args.cg_damping * vector

        x_q = conjugate_gradient(damped, g_q, args.cg_iters)
        x_h = conjugate_gradient(damped, g_h, args.cg_iters)
        f_xq, f_xh = fisher(x_q), fisher(x_h)
        with torch.no_grad():
            a_qq, a_qh, a_hh = x_q @ f_xq, x_q @ f_xh, x_h @ f_xh
            # Predicted Q and entropy changes along each solve. g_Q.F^-1 g_H is one number; truncated CG
            # returns two estimates of it, and the root choice below relies on their agreement.
            gain_q, ent_h = g_q @ x_q, g_h @ x_h
            gain_h = ent_q = 0.5 * (g_q @ x_h + g_h @ x_q)
            radius = 2 * args.target_kl

            # Unconstrained step, and the entropy it would leave behind.
            a_free = (radius / a_qq).sqrt()
            need = self.target_entropy - entropy.detach()
            active = a_free * ent_q < need
            # Constrained: d = a x_q + b x_h on the line g_H.d = c, at the KL radius.
            reach = (radius / a_hh).sqrt() * ent_h  # largest entropy gain inside the ball
            c = torch.maximum(torch.minimum(need, args.entropy_budget * reach), a_free * ent_q)
            k2 = a_qq - 2 * a_qh * ent_q / ent_h + a_hh * (ent_q / ent_h).square()
            k1 = 2 * c * (a_qh / ent_h - a_hh * ent_q / ent_h.square())
            k0 = a_hh * (c / ent_h).square() - radius
            a_tied = (-k1 + (k1.square() - 4 * k2 * k0).clamp(min=0).sqrt()) / (2 * k2)
            a = torch.where(active, a_tied, a_free)
            b = torch.where(active, (c - a_tied * ent_q) / ent_h, torch.zeros_like(a))
            direction = unflatten(a * x_q + b * x_h)

            def trial(scale):
                new_mu, new_log_std = policy(tuple(p + scale * d for p, d in zip(detached, direction)))
                mean_term, scale_term = gaussian_kl(mu_old, sigma_old, new_mu, new_log_std)
                return new_mu, new_log_std, mean_term.mean(), scale_term.mean()

            # The quadratic model puts scale=1 at the radius; KL ~ scale^2 corrects damping and curvature.
            scale = torch.ones((), device=self.device)
            for _ in range(4):
                _, _, mean_term, scale_term = trial(scale)
                scale = scale * (args.target_kl / (mean_term + scale_term).clamp(min=1e-10)).sqrt()
            new_mu, new_log_std, mean_term, scale_term = trial(scale)
            assert torch.isfinite(mean_term + scale_term), "trust-region step produced a non-finite policy"
            for p, d in zip(theta, direction):
                p.add_(scale * d)

            sigma = new_log_std.exp()
            stats = torch.stack([
                mean_term + scale_term, mean_term, scale_term, scale, active.float(),
                scale * (a * gain_q + b * gain_h), value(new_mu, new_log_std, fixed_zs) - q,
                entropy, new_log_std.sum(-1).mean(), sigma.mean(), sigma.min(), sigma.max(),
                args.cg_damping * (x_q @ x_q) / (a_qq + args.cg_damping * (x_q @ x_q)),
                gain_h / (gain_q * ent_h).sqrt(), (new_mu.abs() > 0.99).float().mean(),
            ]).tolist()
        tags = (
            "actor/kl", "actor/kl_mean_term", "actor/kl_scale_term", "actor/step_scale", "actor/entropy_active",
            "actor/q_gain_predicted", "actor/q_gain_critic", "actor/entropy_before", "actor/entropy_after",
            "charts/sigma_mean", "charts/sigma_min", "charts/sigma_max",
            "actor/damping_share", "actor/q_entropy_cosine", "actor/mu_saturated",
        )
        for tag, stat in zip(tags, stats):
            self.writer.add_scalar(tag, stat, global_step)


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
            agent.flush()
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

    envs.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()

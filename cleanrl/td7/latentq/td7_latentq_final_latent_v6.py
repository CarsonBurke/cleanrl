# TD7 LATENT-Q FINAL-LAYER v6 — condition the distribution critic on both actor and noise latents.
#
# Add the actor's final hidden activation, raw action-head output (before tanh), and sampled
# pre-action value u = mu + sigma*eps to v3's Q~(s, mu, sigma). The distribution query uses current
# actor latents and u=mu with a Hermite tangent; the behavior query uses replayed latents and u,
# with a direct TD label. This tests whether the actor representation and realized noise improve
# value conditioning while retaining (mu, sigma).
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
    hermite_order: int = 2
    """0: plain Q~ regression; 1: + sigma*eps*dQ~/dmu; 2: + (eps^2-1)*sigma/2*dQ~/dsigma"""
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
        actor_latent = self.l3(a)
        log_std = torch.tanh(self.l3_logstd(a))
        mu = torch.tanh(actor_latent)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mu, log_std, a, actor_latent


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
    """Fused twin critic conditioned on policy mean/std, raw actor output and pre-action value."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.hdim, self.state_dim = hdim, state_dim
        latent_dim = 4 * action_dim + hdim  # mean | std | final hidden | raw head | pre-action
        self.q0 = nn.Linear(state_dim + latent_dim, 2 * hdim)  # both members side by side
        self.q1 = EnsembleLinear(2, 2 * zs_dim + hdim, hdim)
        self.q2 = EnsembleLinear(2, hdim, hdim)
        self.q3 = EnsembleLinear(2, hdim, 1)
        bound = (state_dim + latent_dim) ** -0.5  # per-member nn.Linear init
        nn.init.uniform_(self.q0.weight, -bound, bound)
        nn.init.uniform_(self.q0.bias, -bound, bound)

    def forward(
        self, state, latent, actor_hidden, actor_head, pre_action, zsa, zs,
        t_latent=None, t_actor_hidden=None, t_actor_head=None, t_pre_action=None, t_zsa=None,
    ):
        batch = state.shape[0]
        latent_input = torch.cat([latent, actor_hidden, actor_head, pre_action], 1)
        pre = self.q0(torch.cat([state, latent_input], 1)).view(batch, 2, self.hdim)
        embeddings = torch.cat([zsa, zs], 1).expand(2, batch, -1)
        if t_latent is None:
            h = avg_l1_norm(pre).transpose(0, 1)
            h = F.elu(self.q1(torch.cat([h, embeddings], 2)))
            h = F.elu(self.q2(h))
            return self.q3(h).squeeze(-1).transpose(0, 1)
        t_latent_input = torch.cat([t_latent, t_actor_hidden, t_actor_head, t_pre_action], 1)
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
        actor_latent = out[:, : self.action_dim]
        return np.tanh(actor_latent), log_std, a, actor_latent


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_episode_steps, args: Args, device, writer: SummaryWriter):
        assert args.hermite_order in (0, 1, 2), "hermite_order must be 0, 1 or 2"
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        adam = dict(fused=True)

        self.actor = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_target = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
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

        # Replay rows retain both the clipped environment action and its pre-action latent.
        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (("state", s), ("action", a), ("mu", a), ("sigma", a), ("actor_hidden", args.hidden_dim), ("actor_head", a), ("eps", a), ("pre_action", a), ("next_state", s), ("reward", 1), ("not_done", 1)):
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
        actor_hidden, actor_head = batch[:, f["actor_hidden"]], batch[:, f["actor_head"]]
        pre_action = batch[:, f["pre_action"]]
        with torch.no_grad():
            # Deterministic center query at TD7's target smoothing std; no smoothing action is sampled.
            next_zs = self.fixed_encoder_target.zs(next_state)
            next_mu, _, next_hidden, next_head = self.actor_target(next_state, next_zs)
            next_zsa = self.fixed_encoder_target.zsa(next_zs, next_mu)
            next_latent = torch.cat([next_mu, torch.full_like(next_mu, args.target_smoothing_sigma)], 1)
            q_target = self.critic_target(next_state, next_latent, next_hidden, next_head, next_mu, next_zsa, next_zs)
            q_target = q_target.min(1, keepdim=True)[0].clamp(min_target, max_target)
            q_target = batch[:, f["reward"]] + batch[:, f["not_done"]] * args.gamma * q_target

        # Compare a current-policy distribution query with the actual behavior context. The
        # Hermite tangent varies mu/sigma and SALE's action embedding, holding actor features fixed.
        fixed_zs = self.fixed_encoder.zs(state)
        policy_mu, policy_log_std, policy_hidden, policy_head = self.actor(state, fixed_zs)
        policy_sigma = policy_log_std.exp()
        t_mu = policy_sigma * eps if args.hermite_order >= 1 else torch.zeros_like(policy_mu)
        t_sigma = 0.5 * policy_sigma * (eps.square() - 1) if args.hermite_order >= 2 else torch.zeros_like(policy_sigma)
        zs2 = torch.cat([fixed_zs, fixed_zs])
        zsa2, t_zsa2 = self.fixed_encoder.zsa_jvp(
            zs2, torch.cat([policy_mu, action]), torch.cat([t_mu, torch.zeros_like(action)])
        )
        latent2 = torch.cat([
            torch.cat([policy_mu, policy_sigma], 1), torch.cat([mu, sigma], 1)
        ])
        hidden2, head2 = torch.cat([policy_hidden, actor_hidden]), torch.cat([policy_head, actor_head])
        pre_actions2 = torch.cat([policy_mu, pre_action])
        t_latent2 = torch.cat([
            torch.cat([t_mu, t_sigma], 1), torch.zeros_like(latent2[: len(mu)])
        ])
        t_hidden2, t_head2 = torch.zeros_like(hidden2), torch.zeros_like(head2)
        t_pre_actions2 = torch.cat([t_mu, torch.zeros_like(pre_action)])
        states2 = torch.cat([state, state])
        q2, tangent2 = self.critic(
            states2, latent2, hidden2, head2, pre_actions2, zsa2, zs2,
            t_latent2, t_hidden2, t_head2, t_pre_actions2, t_zsa2
        )
        batch_size = state.shape[0]
        q_dist, tangent = q2[:batch_size], tangent2[:batch_size]
        q_behavior = q2[batch_size:]
        td_dist = (q_dist + tangent - q_target).abs()
        td_behavior = (q_behavior - q_target).abs()
        loss = (lap_huber(td_dist, args.min_priority) + lap_huber(td_behavior, args.min_priority)) / 2
        with torch.no_grad():
            r2 = 1 - td_dist.square().mean() / (q_target - q_dist).square().mean().clamp(min=1e-8)
            smoothing_gap = (q_behavior - q_dist).mean()
            preaction_gap = (q_behavior - q_dist - tangent).mean()
            diagnostics = torch.stack([
                q_dist.mean(), r2, tangent.std(), q_target.max(), q_target.min(), smoothing_gap, preaction_gap
            ])
        td = torch.stack([td_dist, td_behavior], dim=0).amax(0)
        return loss, td.detach(), diagnostics, fixed_zs

    def _actor_loss_fn(self, state, fixed_zs, alpha):
        mu, log_std, actor_hidden, actor_head = self.actor(state, fixed_zs)
        latent = torch.cat([mu, log_std.exp()], 1)
        q = self.critic(state, latent, actor_hidden, actor_head, mu, self.fixed_encoder.zsa(fixed_zs, mu), fixed_zs)
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
        self.graphs[int(self.training_steps % self.args.policy_freq == 0)].replay()
        if self.training_steps % self.args.target_update_rate == 0:
            self.hard_update()
        if self.training_steps % 500 == 0:
            values = dict(zip(self.metric_names, self.metrics.tolist()))
            for name in ("encoder_loss", "critic_loss", "q_values", "actor_loss", "hermite_r2", "tangent_std", "smoothing_gap", "preaction_gap"):
                self.writer.add_scalar(f"losses/{name}", values[name], self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target.item(), self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target.item(), self.training_steps)
            if self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.training_steps)

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
        actions = policy(states)[0]
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
                mu, log_std, actor_hidden, actor_head = agent.policy(obs[None])
                mu, sigma = mu[0], np.exp(log_std[0])
            else:
                mu, sigma = warmup_mu, warmup_sigma
                actor_hidden = np.zeros(agent.args.hidden_dim, dtype=np.float32)
                actor_head = np.zeros(action_dim, dtype=np.float32)
            eps = rng.standard_normal(action_dim, dtype=np.float32)
            pre_action = mu + sigma * eps
            action = np.clip(pre_action, -1, 1)
        with timer.span("env", use_cuda=False):
            next_obs, reward, terminated, truncated, _ = env.step(action * max_action)
        next_obs = next_obs.astype(np.float32)
        # The original bootstraps at the timeout step even when the env truly terminated there.
        agent.stage(
            state=obs, action=action, mu=mu, sigma=sigma, actor_hidden=actor_hidden, actor_head=actor_head,
            eps=eps, pre_action=pre_action, next_state=next_obs,
            reward=reward, not_done=1.0 - float(terminated and not truncated),
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

# TD7 CONSEQUENCE-Q COSTATE-CLOSED v3 — close the costate recursion and decouple it from exploration noise.
#
# Lineage: v1 Q = r^ + gamma (1 - d^) V(s^'(s, a)) with a supervised one-step model and state-value V;
# v2 added the costate (value-gradient) Bellman loss D_u V(x) -> D_u [r^(x, pi_t) + gamma c V_t(f(x, pi_t))]
# along u_k = sigma_k df/da_k. Two flaws, fixed here:
#  - Scaling u_k by the policy's sigma_k removed supervision exactly on the near-deterministic dimensions
#    the actor keeps climbing (its mu-gradient does not scale with sigma). The target is physics-propagated,
#    not data-limited, so v3 uses a fixed perturbation scale (target_sigma) for every action direction.
#  - The target reads D V_t along transported directions (J_s + J_a dpi/dx) u at the next state, which v2
#    never supervised, so its recursion mostly smoothed v1's own gradient field. v3 also supervises random
#    full-rank state directions, isotropic in one-step-change units and as large as the action-induced
#    perturbation, so every direction the recursion consumes is itself a regression target.
# Loss = expected squared value error under each perturbation family (exact sum over the action
# directions, Monte Carlo mean over the random ones), in the TD loss's units.
# Hypothesis: a closed costate recursion propagates physics over the horizon, correcting the actor's
# improvement direction where v1/v2 extrapolated it by MLP smoothness.
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
    """hidden layer width of the actor, encoder and value networks"""
    model_hidden_dim: int = 512
    """hidden layer width of the consequence model"""
    encoder_lr: float = 3e-4
    """the learning rate of the encoder optimizer"""
    model_lr: float = 3e-4
    """the learning rate of the consequence-model optimizer"""
    value_lr: float = 3e-4
    """the learning rate of the value optimizer"""
    actor_lr: float = 3e-4
    """the learning rate of the actor optimizer"""
    costate_state_tangents: int = 6
    """random full-rank state directions per sample that the costate recursion supervises (closure)"""
    fresh_tail_weight: float = 0.01
    """fresh-data optimism uses steps whose truncated-return tail weight gamma^k is below this"""


# Log-std bounds of the state-dependent Gaussian: sigma in [~6.7e-3, 1] for actions in [-1, 1].
LOG_STD_MIN = -5
LOG_STD_MAX = 0


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def avg_l1_norm_jvp(x, t, eps=1e-8):
    """avg_l1_norm and its forward-mode derivative; t is (batch, K, dim) for K tangents at once."""
    scale = x.abs().mean(-1, keepdim=True).clamp(min=eps)
    y = x / scale
    dscale = (x.sign().unsqueeze(1) * t).mean(-1, keepdim=True)
    return y, (t - y.unsqueeze(1) * dscale) / scale.unsqueeze(1)


def linear_jvp(layer, x, t):
    return layer(x), t @ layer.weight.T


def elu_jvp(x, t):
    y = F.elu(x)
    return y, torch.where(x > 0, 1.0, y + 1).unsqueeze(1) * t


def relu_jvp(x, t):
    return F.relu(x), (x > 0).unsqueeze(1) * t


def silu_jvp(x, t):
    sig = torch.sigmoid(x)
    return x * sig, (sig * (1 + x * (1 - sig))).unsqueeze(1) * t


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

    def jvp(self, state, zs, t_state, t_zs):
        a, t = avg_l1_norm_jvp(*linear_jvp(self.l0, state, t_state))
        a, t = torch.cat([a, zs], 1), torch.cat([t, t_zs], 2)
        a, t = relu_jvp(*linear_jvp(self.l1, a, t))
        a, t = relu_jvp(*linear_jvp(self.l2, a, t))
        mu, t_mu = linear_jvp(self.l3, a, t)
        mu = torch.tanh(mu)
        raw, t_raw = linear_jvp(self.l3_logstd, a, t)
        raw = torch.tanh(raw)
        half_range = 0.5 * (LOG_STD_MAX - LOG_STD_MIN)
        return (
            mu, LOG_STD_MIN + half_range * (raw + 1),
            (1 - mu.square()).unsqueeze(1) * t_mu, half_range * (1 - raw.square()).unsqueeze(1) * t_raw,
        )


class Encoder(nn.Module):
    """SALE. Its zsa head only trains zs; no value network sees zsa."""

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

    def zs_jvp(self, state, t):
        zs, t = elu_jvp(*linear_jvp(self.zs1, state, t))
        zs, t = elu_jvp(*linear_jvp(self.zs2, zs, t))
        return avg_l1_norm_jvp(*linear_jvp(self.zs3, zs, t))

    def zsa(self, zs, action):
        zsa = F.elu(self.zsa1(torch.cat([zs, action], 1)))
        zsa = F.elu(self.zsa2(zsa))
        return self.zsa3(zsa)


class ConsequenceModel(nn.Module):
    """Deterministic one-step model: (normalized s, a) -> (normalized delta s | normalized r | done logit).

    SiLU keeps the action Jacobian smooth: it is the path the actor's gradient takes into V.
    """

    def __init__(self, state_dim, action_dim, hdim=512):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, hdim)
        self.out = nn.Linear(hdim, state_dim + 2)

    def forward(self, state_n, action):
        h = F.silu(self.l1(torch.cat([state_n, action], 1)))
        h = F.silu(self.l2(h))
        h = F.silu(self.l3(h))
        return self.out(h)

    def jvp(self, state_n, action, t_state_n, t_action):
        h, t = silu_jvp(*linear_jvp(self.l1, torch.cat([state_n, action], 1), torch.cat([t_state_n, t_action], 2)))
        h, t = silu_jvp(*linear_jvp(self.l2, h, t))
        h, t = silu_jvp(*linear_jvp(self.l3, h, t))
        return linear_jvp(self.out, h, t)


class Value(nn.Module):
    """V(s) over the raw state and its SALE embedding: TD7's critic trunk with no action inputs."""

    def __init__(self, state_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.v0 = nn.Linear(state_dim, hdim)
        self.v1 = nn.Linear(hdim + zs_dim, hdim)
        self.v2 = nn.Linear(hdim, hdim)
        self.v3 = nn.Linear(hdim, 1)

    def forward(self, state, zs):
        h = avg_l1_norm(self.v0(state))
        h = F.elu(self.v1(torch.cat([h, zs], 1)))
        h = F.elu(self.v2(h))
        return self.v3(h)

    def jvp(self, state, zs, t_state, t_zs):
        h, t = avg_l1_norm_jvp(*linear_jvp(self.v0, state, t_state))
        h, t = elu_jvp(*linear_jvp(self.v1, torch.cat([h, zs], 1), torch.cat([t, t_zs], 2)))
        h, t = elu_jvp(*linear_jvp(self.v2, h, t))
        return linear_jvp(self.v3, h, t)


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
        self.value = Value(state_dim, args.zs_dim, args.hidden_dim).to(device)
        self.value_target = Value(state_dim, args.zs_dim, args.hidden_dim).to(device)
        self.model = ConsequenceModel(state_dim, action_dim, args.model_hidden_dim).to(device)
        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder_target = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        for target, source in (
            (self.actor_target, self.actor),
            (self.value_target, self.value),
            (self.fixed_encoder, self.encoder),
            (self.fixed_encoder_target, self.encoder),
        ):
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=args.value_lr, **adam)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.model_lr, **adam)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr, **adam)
        # (destination, source) pairs of every hard update, in TD7's order (target before fixed). The
        # model has no target copy: it is supervised, never bootstrapped.
        self._hard_updates = [
            (list(self.actor_target.parameters()), list(self.actor.parameters())),
            (list(self.value_target.parameters()), list(self.value.parameters())),
            (list(self.fixed_encoder_target.parameters()), list(self.fixed_encoder.parameters())),
            (list(self.fixed_encoder.parameters()), list(self.encoder.parameters())),
        ]

        # Entropy is confined to the actor loss; autotuning holds the mean sigma near target_sigma.
        self.target_entropy = action_dim * float(np.log(args.target_sigma))
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
        optimizers = [
            self.encoder_optimizer, self.model_optimizer, self.value_optimizer, self.actor_optimizer, self.alpha_optimizer
        ]

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
        # Model normalizers: replay-wide moments of s, delta s and r, recomputed in place at every flush
        # (the captured graphs read these tensors on each replay).
        self.state_mean, self.state_std = torch.zeros(s, **f32), torch.ones(s, **f32)
        self.delta_mean, self.delta_std = torch.zeros(s, **f32), torch.ones(s, **f32)
        self.reward_mean, self.reward_std = torch.zeros(1, **f32), torch.ones(1, **f32)
        self.max_priority = torch.ones((), **f32)
        self.running_max = torch.full((), -1e8, **f32)
        self.running_min = torch.full((), 1e8, **f32)
        self.max_target = torch.zeros((), **f32)
        self.min_target = torch.zeros((), **f32)
        self.metric_names = [
            "encoder_loss", "model_dyn_mse", "model_reward_mse", "model_done_bce", "value_loss", "q_values",
            "value_gap", "reward_err", "actor_loss", "sigma_mean", "sigma_min", "sigma_max",
            "grad_loss", "grad_target_rms", "grad_pred_rms", "grad_cos",
        ]
        self.metrics = torch.zeros(len(self.metric_names), **f32)
        self.fresh_cut = math.ceil(math.log(args.fresh_tail_weight) / math.log(args.gamma))

        maybe_compile = (
            (lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}))
            if args.compile
            else (lambda fn: fn)
        )
        self._encoder_loss = maybe_compile(self._encoder_loss_fn)
        self._model_loss = maybe_compile(self._model_loss_fn)
        self._value_loss = maybe_compile(self._value_loss_fn)
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
        fresh-data optimism before the normalizers or any training step see it.
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
        self._refresh_normalizers()

    def _refresh_normalizers(self):
        f = self._fields
        replay = self.storage[: self.size]
        state = replay[:, f["state"]]
        with torch.no_grad():
            for (mean, std), x in (
                ((self.state_mean, self.state_std), state),
                ((self.delta_mean, self.delta_std), replay[:, f["next_state"]] - state),
                ((self.reward_mean, self.reward_std), replay[:, f["reward"]]),
            ):
                x_std, x_mean = torch.std_mean(x, 0, correction=0)
                mean.copy_(x_mean)
                std.copy_(x_std.clamp(min=1e-3))  # constant coordinates (e.g. a zero reward stretch)

    # ---- the consequence critic ---------------------------------------------------------------
    def _model_out(self, state, action):
        out = self.model((state - self.state_mean) / self.state_std, action)
        return out[:, : self.state_dim], out[:, self.state_dim : self.state_dim + 1], out[:, self.state_dim + 1 :]

    def _consequence(self, state, action):
        """Predicted (next state, reward, continuation probability) in raw units."""
        delta, reward, done_logit = self._model_out(state, action)
        next_state = state + self.delta_mean + self.delta_std * delta
        return next_state, self.reward_mean + self.reward_std * reward, torch.sigmoid(-done_logit)

    def _consequence_jvp(self, state, action, t_state, t_action):
        """_consequence and its derivatives along K (state, action) tangents of shape (batch, K, dim)."""
        out, t_out = self.model.jvp(
            (state - self.state_mean) / self.state_std, action, t_state / self.state_std, t_action
        )
        sd = self.state_dim
        cont = torch.sigmoid(-out[:, sd + 1 :])
        return (
            state + self.delta_mean + self.delta_std * out[:, :sd],
            self.reward_mean + self.reward_std * out[:, sd : sd + 1],
            cont,
            t_state + self.delta_std * t_out[..., :sd],
            self.reward_std * t_out[..., sd : sd + 1],
            -(cont * (1 - cont)).unsqueeze(1) * t_out[..., sd + 1 :],
        )

    def _backup_jvp(self, x, t_x, eps):
        """Directional derivatives of the target backup r^(x, a') + gamma c V_t(f(x, a')), a' = pi_t(x, eps),
        along t_x (batch, K, state_dim): the costate recursion, including the target policy's feedback."""
        zs, t_zs = self.fixed_encoder_target.zs_jvp(x, t_x)
        mu, log_std, t_mu, t_log_std = self.actor_target.jvp(x, zs, t_x, t_zs)
        sigma = log_std.exp()
        pre = mu + sigma * eps
        t_action = (t_mu + (sigma * eps).unsqueeze(1) * t_log_std) * (pre.abs() < 1).unsqueeze(1)
        x2, _, cont, t_x2, t_reward, t_cont = self._consequence_jvp(x, pre.clamp(-1, 1), t_x, t_action)
        zs2, t_zs2 = self.fixed_encoder_target.zs_jvp(x2, t_x2)
        v2, t_v2 = self.value_target.jvp(x2, zs2, t_x2, t_zs2)
        return (t_reward + self.args.gamma * (t_cont * v2.unsqueeze(1) + cont.unsqueeze(1) * t_v2)).squeeze(-1)

    def _q(self, state, action, encoder, value):
        next_state, reward, cont = self._consequence(state, action)
        return reward + self.args.gamma * cont * value(next_state, encoder.zs(next_state))

    # ---- losses -------------------------------------------------------------------------------
    def _encoder_loss_fn(self, batch):
        f = self._fields
        with torch.no_grad():
            next_zs = self.encoder.zs(batch[:, f["next_state"]])
        pred_zs = self.encoder.zsa(self.encoder.zs(batch[:, f["state"]]), batch[:, f["action"]])
        return F.mse_loss(pred_zs, next_zs)

    def _model_loss_fn(self, batch):
        f = self._fields
        state, next_state = batch[:, f["state"]], batch[:, f["next_state"]]
        delta, reward, done_logit = self._model_out(state, batch[:, f["action"]])
        dyn = (delta - (next_state - state - self.delta_mean) / self.delta_std).square().mean()
        rew = (reward - (batch[:, f["reward"]] - self.reward_mean) / self.reward_std).square().mean()
        done = F.binary_cross_entropy_with_logits(done_logit, 1 - batch[:, f["not_done"]])
        return dyn + rew + done, torch.stack([dyn, rew, done]).detach()

    def _value_loss_fn(self, batch, target_eps, actor_eps, costate_eps, state_dirs, min_target, max_target):
        f, args = self._fields, self.args
        state, next_state = batch[:, f["state"]], batch[:, f["next_state"]]
        reward, not_done = batch[:, f["reward"]], batch[:, f["not_done"]]
        with torch.no_grad():
            # Backup of the stochastic target policy through the model at the real next state.
            next_zs = self.fixed_encoder_target.zs(next_state)
            next_mu, next_log_std = self.actor_target(next_state, next_zs)
            next_action = (next_mu + next_log_std.exp() * target_eps).clamp(-1, 1)
            q_next = self._q(next_state, next_action, self.fixed_encoder_target, self.value_target)
            q_next = q_next.clamp(min_target, max_target)
            q_target = reward + not_done * args.gamma * q_next
            pred_next, pred_reward, pred_cont = self._consequence(state, batch[:, f["action"]])
            value_states = torch.cat([pred_next, next_state])
            value_zs = self.fixed_encoder.zs(value_states)

            # Costate supervision at x = f(s, a) for the actor's own sample a = clip(mu + sigma*eps):
            # the directions the actor reads, u_k = target_sigma * df/da_k (zero where a_k is clipped),
            # plus random state directions w with E||w||^2 = sum_k ||u_k||^2 in one-step-change units.
            mu, log_std = self.actor(state, self.fixed_encoder.zs(state))
            pre = mu + log_std.exp() * actor_eps
            t_action = torch.diag_embed(args.target_sigma * (pre.abs() < 1).float())
            x, _, x_cont, u, _, _ = self._consequence_jvp(
                state, pre.clamp(-1, 1), torch.zeros(*t_action.shape[:2], self.state_dim, device=state.device), t_action
            )
            size = (u / self.delta_std).square().sum((1, 2)).sqrt()[:, None, None]
            w = self.delta_std * state_dirs / state_dirs.norm(dim=-1, keepdim=True) * size
            tangents = torch.cat([u, w], 1)
            grad_target = self._backup_jvp(x, tangents, costate_eps)
            zs_x, t_zs_x = self.fixed_encoder.zs_jvp(x, tangents)
        v_pred, v_real = self.value(value_states, value_zs).chunk(2)
        q_behavior = pred_reward + args.gamma * pred_cont * v_pred
        td = (q_behavior - q_target).abs()
        td_anchor = not_done * (v_real - q_next).abs()
        grad_pred = self.value.jvp(x, zs_x, tangents, t_zs_x)[1].squeeze(-1)
        grad_err = x_cont * (grad_pred - grad_target).abs()
        n_action = self.action_dim
        grad_loss = (
            lap_huber(grad_err[:, :n_action], args.min_priority)
            + lap_huber(grad_err[:, n_action:], args.min_priority) / args.costate_state_tangents
        )
        loss = (lap_huber(td, args.min_priority) + lap_huber(td_anchor, args.min_priority) + grad_loss) / 3
        with torch.no_grad():
            diagnostics = torch.stack([
                q_behavior.mean(),
                q_target.max(),
                q_target.min(),
                # V's disagreement between predicted and real next state: the dynamics error the
                # value function can see, i.e. the part of the model's surprise that matters.
                args.gamma * (v_pred - v_real).abs().mean(),
                (pred_reward - reward).abs().mean(),
                grad_loss.detach(),
                # Along the actor's directions only, the part its gradient reads.
                grad_target[:, :n_action].square().mean().sqrt(),
                grad_pred.detach()[:, :n_action].square().mean().sqrt(),
                F.cosine_similarity(grad_pred.detach()[:, :n_action], grad_target[:, :n_action], dim=1).mean(),
            ])
        return loss, torch.maximum(td, td_anchor).detach(), diagnostics

    def _actor_loss_fn(self, state, eps, alpha):
        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(state)
        mu, log_std = self.actor(state, fixed_zs)
        action = (mu + log_std.exp() * eps).clamp(-1, 1)
        q = self._q(state, action, self.fixed_encoder, self.value)
        return -q.mean() - alpha * log_std.sum(-1).mean(), log_std, fixed_zs

    def _entropy_fn(self, state, fixed_zs):
        with torch.no_grad():
            return self.actor(state, fixed_zs)[1].sum(-1)

    # ---- train step ---------------------------------------------------------------------------
    def _make_step(self, update_actor):
        args, device = self.args, self.device
        noise_shape = (args.batch_size, self.action_dim)

        def step():
            csum = torch.cumsum(self.priority, 0)
            ind = torch.searchsorted(csum, torch.rand(args.batch_size, device=device) * csum.gather(0, self.last))
            ind = torch.minimum(ind, self.last)  # the scan is only monotone up to rounding
            batch = self.storage.index_select(0, ind)

            self.encoder_optimizer.zero_grad(set_to_none=False)
            encoder_loss = self._encoder_loss(batch)
            encoder_loss.backward()
            self.encoder_optimizer.step()

            self.model_optimizer.zero_grad(set_to_none=False)
            model_loss, model_terms = self._model_loss(batch)
            model_loss.backward()
            self.model_optimizer.step()

            self.value_optimizer.zero_grad(set_to_none=False)
            target_eps, value_actor_eps, costate_eps = torch.randn((3, *noise_shape), device=device).unbind(0)
            state_dirs = torch.randn(args.batch_size, args.costate_state_tangents, self.state_dim, device=device)
            value_loss, td, diagnostics = self._value_loss(
                batch, target_eps, value_actor_eps, costate_eps, state_dirs, self.min_target, self.max_target
            )
            value_loss.backward()
            self.value_optimizer.step()

            with torch.no_grad():
                torch.maximum(self.running_max, diagnostics[1], out=self.running_max)
                torch.minimum(self.running_min, diagnostics[2], out=self.running_min)
                priority = td.max(1)[0].clamp(min=args.min_priority).pow(args.lap_alpha)
                self.priority[ind] = priority
                torch.maximum(self.max_priority, priority.max(), out=self.max_priority)
                self.metrics[0].copy_(encoder_loss.detach())
                self.metrics[1:4].copy_(model_terms)
                self.metrics[4].copy_(value_loss.detach())
                self.metrics[5].copy_(diagnostics[0])
                self.metrics[6:8].copy_(diagnostics[3:5])
                self.metrics[12:16].copy_(diagnostics[5:9])

            if update_actor:
                state = batch[:, self._fields["state"]]
                self.actor_optimizer.zero_grad(set_to_none=False)
                actor_eps = torch.randn(noise_shape, device=device)
                actor_loss, log_std, fixed_zs = self._actor_loss(state, actor_eps, self.log_alpha.detach().exp())
                actor_loss.backward(inputs=list(self.actor.parameters()))
                self.actor_optimizer.step()
                entropy = self._entropy(state, fixed_zs)
                self.alpha_optimizer.zero_grad(set_to_none=False)
                alpha_loss = -(self.log_alpha.exp() * (self.target_entropy - entropy)).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    sigma = log_std.detach().exp()
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
                "encoder_loss", "model_dyn_mse", "model_reward_mse", "model_done_bce", "value_loss", "q_values",
                "value_gap", "reward_err", "actor_loss", "grad_loss", "grad_target_rms", "grad_pred_rms", "grad_cos",
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
        Positive optimism is critic error the actor has steered into; the model terms split the
        critic's surprise into its vector (dynamics) and value-visible parts.
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
        state, action, next_state = rows[:, f["state"]], rows[:, f["action"]], rows[:, f["next_state"]]
        realized = torch.as_tensor(returns[:valid], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred_next, pred_reward, pred_cont = self._consequence(state, action)
            v_pred, v_real = self.value(
                torch.cat([pred_next, next_state]), self.fixed_encoder.zs(torch.cat([pred_next, next_state]))
            ).chunk(2)
            q = (pred_reward + self.args.gamma * pred_cont * v_pred).squeeze(1)
            dyn = (((next_state - pred_next) / self.delta_std).square().mean())
            error = q - realized
            stats = torch.stack([
                error.mean(),
                error.mean() / realized.abs().mean().clamp(min=1e-8),
                error.abs().mean(),
                dyn,
                self.args.gamma * (v_pred - v_real).abs().mean(),
            ]).tolist()
        for name, value in zip(("optimism", "rel_optimism", "abs_err", "model_dyn_mse", "value_gap"), stats):
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
            state=obs, action=action, next_state=next_obs, reward=reward,
            not_done=1.0 - float(terminated and not truncated),
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
                "value": agent.value.state_dict(),
                "model": agent.model.state_dict(),
                "checkpoint_policy": agent.checkpoint_policy._flat,
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    env.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()

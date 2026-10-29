# DISCO103 x TD7 TARGET ABLATION v4 — v3 with the policy target swappable; everything else is v3's.
#
# v3 trailed its TD7 base by 1.5-2k at matched steps while moving the policy ~23x less per rollout
# (KL 0.56 vs 13.2): Disco103's pi_hat sits ~0.02 nats per dim from pi. This version asks whether the
# bottleneck is the meta-learned target or the categorical/marginal-Q pipeline around it.
# --policy-target disco: pi_hat, extrapolated by disco_scale in log space (1 = v3).
# --policy-target mpo: pi * exp(normalized qv_adv / eta) per dim, eta bisected per row to KL = mpo_kl,
#   a known improvement operator on the same marginal Q (y, z and aux still train on Disco103).
# Both log Disco103's alignment with the critic: the pi-weighted correlation of log pi_hat - log pi with qv_adv.
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared import disco103
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
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_steps: int = 64
    """steps per environment per rollout"""
    env_threads: int = 2
    """maximum physics threads"""
    actor_sale: bool = True
    """feed the fixed SALE state embedding to the actor (the base's replay actor)"""
    num_bins: int = 11
    """evenly spaced actions in [-1, 1] per action dim"""
    segment_length: int = 16
    """steps per replayed actor segment; batch_size // segment_length segments per actor step"""
    policy_target: str = "disco"
    """disco: Disco103's pi_hat (scaled by disco_scale); mpo: KL-bounded exponentiated marginal advantage"""
    disco_scale: float = 1.0
    """log-space extrapolation of pi_hat away from pi (1 = Disco103's own target)"""
    mpo_kl: float = 0.1
    """per-dim KL(target || pi) of the mpo target"""
    actor_window: int = 65536
    """actor segments come from the newest N transitions (Disco103 was meta-trained near on-policy)"""
    model_lstm_size: int = 128
    """width of the action-conditional model's LSTM transition"""
    model_head_hidden: int = 128
    """hidden width of the model's z and next-policy heads"""
    updates_per_step: float = 1.0
    """replay updates per environment step"""
    warmup_rollouts: int = 25
    """recorded rollouts of the warmup policy before training starts"""
    warmup_sigma: float = 0.6
    """std of the warmup policy: bin logits of a discretized N(0, warmup_sigma^2)"""
    hermite_order: int = 2
    """0: plain Q~ regression; 1: + sigma*eps*dQ~/dmu; 2: + (eps^2-1)*sigma/2*dQ~/dsigma"""
    eval_interval: int = 5
    """evaluate the policy mean every N rollouts"""
    eval_eps: int = 10
    """number of evaluation episodes (run as one batched rollout)"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    batch_size: int = 256
    """the batch size of sample from the replay memory (critic transitions and actor segment steps)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_update_rate: int = 250
    """hard-update the target networks and fixed encoders every N training steps"""
    target_smoothing_sigma: float = 0.2
    """per-dim std of the (exact) Gaussian smoothing of the target policy's mean, TD7's 0.2"""
    policy_freq: int = 2
    """the frequency of training the policy"""
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


# The base Gaussian's smallest std; floors the categorical's std where the critic's latent reads it.
SIGMA_MIN = float(np.exp(-5))
# A categorical's standardized draw is heavy-tailed (|eps| ~ 1/sqrt(p)); the second-order Hermite term
# reads it clipped at 3, which leaves the base's Gaussian eps essentially untouched.
HERMITE_EPS_CLIP = 3.0


@contextmanager
def matmul_precision(precision):
    """Float32 matmul precision for the kernels launched (and graph-captured) inside the block."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


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


def haiku_linear(in_features, out_features, std=None):
    """nn.Linear with hk.Linear's TruncatedNormal(1/sqrt(fan_in)), or (std), weight init."""
    layer = nn.Linear(in_features, out_features)
    std = std if std is not None else in_features**-0.5
    nn.init.trunc_normal_(layer.weight, std=std, a=-2.0 * std, b=2.0 * std)
    nn.init.zeros_(layer.bias)
    return layer


def haiku_mlp(*sizes):
    layers = []
    for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        if index:
            layers.append(nn.ReLU())
        layers.append(haiku_linear(fan_in, fan_out))
    return nn.Sequential(*layers)


def moments(probs, bins):
    """Mean and floored std of per-dim categoricals over the bin actions: [..., A] -> [...] twice."""
    mu = probs @ bins
    return mu, (probs @ bins.square() - mu.square()).clamp_min(0.0).sqrt().clamp_min(SIGMA_MIN)


def take(x, index):
    """x[..., index, :] per row: [..., A, F] with index [...] -> [..., F]."""
    return x.gather(-2, index[..., None, None].expand(*index.shape, 1, x.shape[-1])).squeeze(-2)


def taken_first(x, taken):
    """Swap each row's taken action into index 0 of the action axis (dim 2); its own inverse."""
    order = torch.arange(x.shape[2], device=x.device).expand(*taken.shape, -1)
    order = torch.where(order == 0, taken[..., None], torch.where(order == taken[..., None], 0, order))
    return x.gather(2, order.view(*order.shape, *[1] * (x.dim() - 3)).expand_as(x))


class Actor(nn.Module):
    """The base's (SALE) actor torso with disco_rl's agent heads, factorized over action dims.

    Per-dim bin logits [..., d, A] and y [..., d, P] read the torso; the action-conditional model
    takes one LSTM transition from a root embedding on each (dim, bin) one-hot, predicting z and that
    dim's next-state logits.
    """

    def __init__(self, state_dim, action_dim, num_bins, zs_dim, hdim, lstm, head_hidden):
        super().__init__()
        self.sale = zs_dim > 0  # zs_dim=0: state-only actor, ``zs`` is ignored
        self.action_dim, self.num_bins = action_dim, num_bins
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.policy_head = haiku_linear(hdim, action_dim * num_bins, std=1e-2)
        self.y_head = haiku_linear(hdim, action_dim * disco103.PREDICTION_SIZE, std=1e-2)
        self.root = haiku_linear(hdim, lstm)
        self.transition = haiku_linear(action_dim * num_bins + lstm, 4 * lstm)
        self.z_head = haiku_mlp(lstm, head_hidden, disco103.PREDICTION_SIZE)
        self.aux_head = haiku_mlp(lstm, head_hidden, num_bins)

    def torso(self, state, zs):
        a = avg_l1_norm(self.l0(state))
        if self.sale:
            a = torch.cat([a, zs], -1)
        a = F.relu(self.l1(a))
        return F.relu(self.l2(a))

    def heads(self, h):
        logits = self.policy_head(h).unflatten(-1, (self.action_dim, self.num_bins))
        return logits, self.y_head(h).unflatten(-1, (self.action_dim, disco103.PREDICTION_SIZE))

    def forward(self, state, zs):
        return self.heads(self.torso(state, zs))[0]

    def model(self, h):
        """One LSTM transition per (dim, bin): [..., H] -> hidden [..., d, A, lstm]."""
        cell = self.root(h)
        input_weight, hidden_weight = self.transition.weight.split(
            (self.action_dim * self.num_bins, cell.shape[-1]), 1)
        root_gates = F.linear(torch.tanh(cell), hidden_weight, self.transition.bias)
        # A one-hot input selects one input-weight column per (dim, bin).
        gates = input_weight.T.unflatten(0, (self.action_dim, self.num_bins)) + root_gates[..., None, None, :]
        return disco103.lstm_gates(gates, cell[..., None, None, :])[0]


class HostPolicy:
    """NumPy mirror of the actor's logits (and fixed_encoder.zs when it feeds the actor) for acting.

    ``refresh`` pulls the device weights in one flat D2H copy; call it whenever the device policy
    changes (after every training burst).
    """

    def __init__(self, encoder: Encoder, actor: Actor):
        self.sale = actor.sale
        encoder_sources = [
            encoder.zs1.weight, encoder.zs1.bias, encoder.zs2.weight, encoder.zs2.bias,
            encoder.zs3.weight, encoder.zs3.bias,
        ]
        self._sources = (encoder_sources if self.sale else []) + [
            actor.l0.weight, actor.l0.bias, actor.l1.weight, actor.l1.bias, actor.l2.weight, actor.l2.bias,
            actor.policy_head.weight, actor.policy_head.bias,
        ]
        self._flat = np.zeros(sum(t.numel() for t in self._sources), dtype=np.float32)
        self._views, offset = [], 0
        for tensor in self._sources:
            view = self._flat[offset : offset + tensor.numel()].reshape(tensor.shape)
            self._views.append(view.T if view.ndim == 2 else view)  # weights as (in, out)
            offset += tensor.numel()
        self.shape = (actor.action_dim, actor.num_bins)

    def refresh(self):
        self._flat[:] = torch.cat([t.detach().reshape(-1) for t in self._sources]).cpu().numpy()

    def __call__(self, state):
        """Per-dim bin logits [N, d, A]."""
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
        return (a @ wh + bh).reshape(len(state), *self.shape)


def host_policy_stats(logits, bins):
    """Probabilities, mean and floored std of per-dim categoricals on the host."""
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    mu = probs @ bins
    return probs, mu, np.maximum(np.sqrt(np.maximum(probs @ bins**2 - mu**2, 0)), SIGMA_MIN)


def sample_bins(probs, rng):
    """One inverse-CDF draw per action dim: [N, d, A] -> bin indices [N, d]."""
    cdf = np.cumsum(probs, -1)
    u = rng.random(cdf.shape[:-1], dtype=np.float32)[..., None] * cdf[..., -1:]
    return np.minimum((cdf <= u).sum(-1), probs.shape[-1] - 1)


class DiscoTD7Agent:
    def __init__(self, state_dim, action_dim, bins, args: Args, device, writer: SummaryWriter):
        assert args.buffer_size % args.num_envs == 0, "replay slots hold one step of every env"
        assert args.batch_size % args.segment_length == 0, "actor batches are whole segments"
        assert args.policy_target in ("disco", "mpo"), "policy_target must be disco or mpo"
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        self.bins = bins
        adam = dict(fused=True)

        actor_zs_dim = args.zs_dim if args.actor_sale else 0

        def make_actor():
            return Actor(state_dim, action_dim, args.num_bins, actor_zs_dim, args.hidden_dim,
                         args.model_lstm_size, args.model_head_hidden).to(device)

        self.actor = make_actor()
        self.actor_target = make_actor()  # the base's hard-updated target, for the critic's bootstrap
        self.disco_target = make_actor()  # disco_rl's EMA target, for the meta-network's target inputs
        self.critic = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_target = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder_target = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        for target, source in (
            (self.actor_target, self.actor),
            (self.disco_target, self.actor),
            (self.critic_target, self.critic),
            (self.fixed_encoder, self.encoder),
            (self.fixed_encoder_target, self.encoder),
        ):
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, **adam)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr, **adam)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
        # (destination, source) pairs of every hard update, in TD7's order (target before fixed).
        self._hard_updates = [
            (list(self.actor_target.parameters()), list(self.actor.parameters())),
            (list(self.critic_target.parameters()), list(self.critic.parameters())),
            (list(self.fixed_encoder_target.parameters()), list(self.fixed_encoder.parameters())),
            (list(self.fixed_encoder.parameters()), list(self.encoder.parameters())),
        ]
        self.meta = disco103.Disco103().to(device)
        self.lifetime_state = self.meta.initial_state(device)
        self.adv_ema = disco103.ema_init(device)

        # Replay rows: the base's fields plus the taken bins, behaviour logits and episode resets.
        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (
            ("state", s), ("action", a), ("mu", a), ("sigma", a), ("eps", a), ("pre_action", a),
            ("bin", a), ("logits", a * args.num_bins), ("next_state", s), ("reward", 1), ("not_done", 1),
            ("reset", 1),
        ):
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
        # Row = slot * num_envs + env, a slot being one vector-env step; (oldest actor slot, actor slots).
        self.slots = args.buffer_size // args.num_envs
        self.actor_slots = min(args.actor_window, args.buffer_size) // args.num_envs
        self.window = torch.zeros(2, dtype=torch.long, device=device)
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
            "encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std", "smoothing_gap",
            "preaction_gap", "disco_loss", "pi_loss", "y_loss", "z_loss", "aux_loss", "entropy",
            "pi_hat_entropy", "sigma_mean", "rho", "joint_trace", "adv_abs", "qv_spread", "pi_hat_kl",
            "pi_hat_alignment", "target_kl",
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
        # training_steps % policy_freq == 0 selects the actor-step graph (TD7's delayed update).
        self.graphs = capture_steps(
            [self._make_step(update_actor=False), self._make_step(update_actor=True)],
            optimizers=[self.encoder_optimizer, self.critic_optimizer, self.actor_optimizer],
            state_tensors=[
                self.priority, self.max_priority, self.running_max, self.running_min, self.metrics,
                self.lifetime_state, self.adv_ema, *self.disco_target.parameters(),
            ],
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
        count = min(self.size // self.args.num_envs, self.actor_slots)
        self.window[0].fill_((self.ptr // self.args.num_envs - count) % self.slots)
        self.window[1].fill_(count)

    def sample_segments(self):
        """Uniform length-(T+1) single-env segments of the actor window, behind the write head."""
        args = self.args
        length = args.segment_length + 1
        count = args.batch_size // args.segment_length
        starts = self.window[1] - length + 1
        start = (torch.rand(count, device=self.device) * starts).long().clamp(max=starts - 1).clamp(min=0)
        slot = (self.window[0] + start + torch.arange(length, device=self.device)[:, None]) % self.slots
        env = torch.randint(0, args.num_envs, (count,), device=self.device)
        return self.storage[slot * args.num_envs + env]

    # ---- critic -------------------------------------------------------------------------------
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
            next_mu, _ = moments(torch.softmax(self.actor_target(next_state, next_zs), -1), self.bins)
            next_zsa = self.fixed_encoder_target.zsa(next_zs, next_mu)
            next_latent = torch.cat([next_mu, torch.full_like(next_mu, args.target_smoothing_sigma)], 1)
            q_target = self.critic_target(next_state, next_latent, next_mu, next_zsa, next_zs)
            q_target = q_target.min(1, keepdim=True)[0].clamp(min_target, max_target)
            q_target = batch[:, f["reward"]] + batch[:, f["not_done"]] * args.gamma * q_target

            # Three queries share one critic and one 3B pass: distribution center (mu, sigma, mu)
            # with Hermite tangent, observed pre-action latent (mu, sigma, u), and point action
            # (a, 0, a). The Hermite terms only need eps standardized, which the categorical's is.
            # pre_action == action for a categorical, so the latter two share their TD label.
            fixed_zs = self.fixed_encoder.zs(state)
            t_mu = sigma * eps if args.hermite_order >= 1 else torch.zeros_like(mu)
            t_sigma = (
                0.5 * sigma * (eps.clamp(-HERMITE_EPS_CLIP, HERMITE_EPS_CLIP).square() - 1)
                if args.hermite_order >= 2 else torch.zeros_like(sigma)
            )
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
        return loss, td.detach(), diagnostics

    # ---- actor --------------------------------------------------------------------------------
    def critic_values(self, state, zs, mu, sigma, pre_action=None):
        """Twin-mean Q~(s, mu, sigma, u): the center query (u = mu) by default.

        [..., S] states, [..., Z] embeddings and [..., d] latents/pre-actions -> [...].
        """
        u = mu if pre_action is None else pre_action
        lead = u.shape[:-1]
        state, zs, mu, sigma, u = (x.expand(*lead, x.shape[-1]).reshape(-1, x.shape[-1]) for x in (state, zs, mu, sigma, u))
        q = self.critic(state, torch.cat([mu, sigma], 1), u, self.fixed_encoder.zsa(zs, u), zs)
        return q.mean(1).view(lead)

    def bin_values(self, state, zs, mu, sigma):
        """Q_d(s, b): the pre-action query at the policy's latent with u_d = b and u_-d = mu_-d.

        [..., S] states and [..., d] policy moments -> [..., d, A]. The latent stays the trained
        policy's own; only the pre-action moves, along one dim, onto actions the policy draws.
        """
        pinned = torch.eye(self.action_dim, dtype=torch.bool, device=self.device)[:, None, :]  # [d, 1, d]
        u = torch.where(pinned, self.bins[:, None], mu[..., None, None, :])  # [..., d, A, d]
        grid = lambda x: x[..., None, None, :]
        return self.critic_values(grid(state), grid(zs), grid(mu), grid(sigma), u)

    def mpo_target(self, log_pi, advantages, iterations=30):
        """log pi + advantages / eta, with eta per row bisected (in log space) to KL(target || pi) = mpo_kl."""
        low = torch.full(log_pi.shape[:-1], -8.0, device=log_pi.device)  # log(1 / eta)
        high = torch.full_like(low, 8.0)

        def target(log_inverse_eta):
            return torch.log_softmax(log_pi + log_inverse_eta.exp()[..., None] * advantages, -1)

        for _ in range(iterations):
            middle = 0.5 * (low + high)
            candidate = target(middle)
            over = (candidate.exp() * (candidate - log_pi)).sum(-1) > self.args.mpo_kl
            high = torch.where(over, middle, high)
            low = torch.where(over, low, middle)
        return target(low)

    def _actor_loss_fn(self, batch, adv_ema, lifetime_state):
        """Disco103's agent loss on replayed segments, one meta-batch row per (segment, action dim)."""
        f, args = self._fields, self.args
        steps, segments = args.segment_length, batch.shape[1]
        d, bins = self.action_dim, args.num_bins
        actor, target = self.actor, self.disco_target
        # T+1 rows: row t+1 is step t's successor unless step t reset; next_state holds the final obs.
        state, final = batch[..., f["state"]], batch[:-1, :, f["next_state"]]
        rewards, not_done, resets = (batch[:-1, :, f[name]].squeeze(-1) for name in ("reward", "not_done", "reset"))
        taken = batch[..., f["bin"]].long()  # [T+1, B, d]
        behaviour = batch[:-1, :, f["logits"]].unflatten(-1, (d, bins))

        with torch.no_grad():
            zs, final_zs = self.fixed_encoder.zs(state), self.fixed_encoder.zs(final)
        h = actor.torso(state, zs)
        logits, y = actor.heads(h)
        hidden = actor.model(h)
        z = actor.z_head(hidden)  # [T+1, B, d, A, P]
        aux = actor.aux_head(take(hidden[:-1], taken[:-1]))  # next-state logits of the taken bin's dim

        with torch.no_grad():
            final_h = actor.torso(final, final_zs)
            final_logits, final_y = actor.heads(final_h)
            final_z = actor.z_head(actor.model(final_h))
            target_h, final_target_h = target.torso(state, zs), target.torso(final, final_zs)
            target_logits, target_y = target.heads(target_h)
            target_z = target.z_head(target.model(target_h))
            final_target_y = target.heads(final_target_h)[1]
            final_target_z = target.z_head(target.model(final_target_h))

            def successor(current, final_value):
                ended = resets.bool().view(steps, segments, *[1] * (final_value.dim() - 2))
                return torch.where(ended, final_value, current[1:])

            # Values are the center query; per-bin Q the pre-action query, centred on pi per dim, so
            # target_q averages to the value under pi as in disco_rl.
            pi = torch.softmax(logits.detach(), -1)
            mu, sigma = moments(pi, self.bins)
            values = self.critic_values(state, zs, mu, sigma)  # [T+1, B]
            q = self.bin_values(state, zs, mu, sigma)  # [T+1, B, d, A]
            q_taken = self.critic_values(state, zs, mu, sigma, self.bins[taken])  # Q(s_t, a_t), all dims
            final_values = self.critic_values(final, final_zs, *moments(torch.softmax(final_logits, -1), self.bins))
            next_values = successor(values, final_values)
            next_logits = successor(logits.detach(), final_logits)
            qv_advantages = q[:-1] - (pi * q).sum(-1, keepdim=True)[:-1]

            # Joint Retrace on the executed actions: G_t = r_t + g_t [V(s_t+1) + c_t+1 (G_t+1 - Q(s_t+1, a_t+1))]
            # with c = lambda min(1, prod_j rho_j), cut at resets.
            log_pi = torch.log_softmax(logits.detach(), -1).gather(-1, taken[..., None]).squeeze(-1)
            log_mu = torch.log_softmax(behaviour, -1).gather(-1, taken[:-1, ..., None]).squeeze(-1)
            log_rho = log_pi[:-1] - log_mu  # [T, B, d]
            joint_log_rho = log_rho.sum(-1)
            trace = disco103.TD_LAMBDA * joint_log_rho.exp().clamp_max(1.0)
            discounts = args.gamma * not_done
            continues = 1.0 - resets
            target_return = rewards[-1] + discounts[-1] * next_values[-1]
            q_targets = [target_return]
            for t in range(steps - 2, -1, -1):
                correction = continues[t] * trace[t + 1] * (target_return - q_taken[t + 1])
                target_return = rewards[t] + discounts[t] * (next_values[t] + correction)
                q_targets.append(target_return)
            residual = torch.stack(q_targets[::-1]) - q_taken[:-1]
            # Dim d's advantage: its critic marginal plus the return residual, the latter weighted by the
            # other dims' clipped ratio (their executed actions came from mu, not pi).
            others = (joint_log_rho[..., None] - log_rho).exp().clamp_max(1.0)
            advantages = qv_advantages.gather(-1, taken[:-1, ..., None]).squeeze(-1) + others * residual[..., None]
            adv_ema = disco103.ema_update(adv_ema, advantages)
            terminations = (1.0 - not_done)[..., None].expand(steps, segments, d)
            per_dim = lambda x: x[..., None].expand(steps, segments, d)

            first = taken[:-1].flatten(1, 2)  # [T, B*d]
            rows = lambda x: x.flatten(1, 2)
            current = lambda x: taken_first(rows(x), first)
            pi_hat, y_hat, z_hat, lifetime_state = self.meta(disco103.MetaInputs(
                logits=current(logits[:-1].detach()), next_logits=rows(next_logits),
                behaviour_logits=current(behaviour), target_logits=current(target_logits[:-1]),
                rewards=rows(per_dim(rewards)), terminations=rows(terminations), resets=rows(per_dim(resets)),
                values=rows(per_dim(values[:-1])), next_values=rows(per_dim(next_values)),
                advantages=rows(advantages),
                normalized_advantages=rows(disco103.ema_normalize(adv_ema, advantages)),
                y=rows(y[:-1].detach()), next_y=rows(successor(y.detach(), final_y)),
                target_y=rows(target_y[:-1]), next_target_y=rows(successor(target_y, final_target_y)),
                z=current(z[:-1].detach()), next_z=rows(successor(z.detach(), final_z)),
                target_z=current(target_z[:-1]), next_target_z=rows(successor(target_z, final_target_z)),
                target_q=current(values[:-1, :, None, None] + qv_advantages), qv_advantages=current(qv_advantages),
                normalized_qv_advantages=current(disco103.ema_normalize(adv_ema, qv_advantages)),
            ), lifetime_state)
            pi_hat = taken_first(pi_hat, first).unflatten(1, (segments, d))
            y_hat, z_hat = y_hat.unflatten(1, (segments, d)), z_hat.unflatten(1, (segments, d))

            log_pi_now = torch.log_softmax(logits[:-1].detach(), -1)
            tilt = torch.log_softmax(pi_hat, -1) - log_pi_now
            if args.policy_target == "disco":
                policy_target = log_pi_now + args.disco_scale * tilt
            else:
                policy_target = self.mpo_target(log_pi_now, disco103.ema_normalize(adv_ema, qv_advantages))
            # pi-weighted correlation of Disco103's tilt with the marginal advantage, per (state, dim).
            pi_now = log_pi_now.exp()
            centred = lambda x: x - (pi_now * x).sum(-1, keepdim=True)
            tilt_c, adv_c = centred(tilt), centred(qv_advantages)
            alignment = (pi_now * tilt_c * adv_c).sum(-1) / (
                (pi_now * tilt_c.square()).sum(-1) * (pi_now * adv_c.square()).sum(-1)).clamp_min(1e-12).sqrt()

        policy_loss = disco103.categorical_kl(policy_target, logits[:-1])
        y_loss = disco103.categorical_kl(y_hat, y[:-1])
        z_loss = disco103.categorical_kl(z_hat, take(z[:-1], taken[:-1]))
        aux_loss = disco103.categorical_kl(next_logits, aux) * (1.0 - terminations)
        losses = torch.stack((policy_loss.mean(), y_loss.mean(), z_loss.mean(), aux_loss.mean()))
        costs = losses.new_tensor((disco103.PI_COST, disco103.Y_COST, disco103.Z_COST, disco103.AUX_POLICY_COST))
        total = (costs * losses).sum()

        with torch.no_grad():
            def entropy(categorical_logits):
                log_p = torch.log_softmax(categorical_logits, -1)
                return -(log_p.exp() * log_p).sum(-1).mean()

            metrics = torch.stack((
                total, *losses, entropy(logits[:-1]), entropy(pi_hat), sigma.mean(),
                log_rho.exp().mean(), trace.mean(), advantages.abs().mean(), qv_advantages.std(-1).mean(),
                disco103.categorical_kl(pi_hat, logits[:-1].detach()).mean(), alignment.mean(),
                disco103.categorical_kl(policy_target, logits[:-1].detach()).mean(),
            ))
        return total, metrics, adv_ema, lifetime_state

    # ---- train step ---------------------------------------------------------------------------
    def _make_step(self, update_actor):
        args, device = self.args, self.device
        target_params, online_params = list(self.disco_target.parameters()), list(self.actor.parameters())

        def step():
            csum = torch.cumsum(self.priority, 0)
            ind = torch.searchsorted(csum, torch.rand(args.batch_size, device=device) * csum.gather(0, self.last))
            ind = torch.minimum(ind, self.last)  # the scan is only monotone up to rounding
            batch = self.storage.index_select(0, ind)

            # The base's TF32 for the critic and encoder; full FP32 for the actor's per-bin Q differences.
            with matmul_precision("high"):
                self.encoder_optimizer.zero_grad(set_to_none=False)
                encoder_loss = self._encoder_loss(batch)
                encoder_loss.backward()
                self.encoder_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=False)
                critic_loss, td, diagnostics = self._critic_loss(batch, self.min_target, self.max_target)
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
                self.metrics[5:7].copy_(diagnostics[5:7])

            if update_actor:
                self.actor_optimizer.zero_grad(set_to_none=False)
                loss, metrics, adv_ema, lifetime_state = self._actor_loss(
                    self.sample_segments(), self.adv_ema, self.lifetime_state)
                loss.backward(inputs=online_params)
                with torch.no_grad():
                    # disco_rl's target tracks the pre-update parameters.
                    torch._foreach_lerp_(target_params, online_params, 1.0 - disco103.TARGET_PARAMS_COEFF)
                    self.adv_ema.copy_(adv_ema)
                    self.lifetime_state.copy_(lifetime_state)
                    self.metrics[7:].copy_(metrics)
                self.actor_optimizer.step()

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
        if self.training_steps % 1000 == 0:
            for name, value in zip(self.metric_names, self.metrics.tolist()):
                self.writer.add_scalar(f"losses/{name}", value, self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target.item(), self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target.item(), self.training_steps)

    # ---- one rollout of learning -------------------------------------------------------------
    def learn(self, global_step):
        # How far a rollout of replay updates moves the policy on the states it just visited.
        state = self.fresh[:, self._fields["state"]]
        with torch.no_grad():
            old = torch.log_softmax(self.actor(state, self.fixed_encoder.zs(state)), -1)
        for _ in range(self.updates_per_rollout):
            self.train()
        with torch.no_grad():
            new = torch.log_softmax(self.actor(state, self.fixed_encoder.zs(state)), -1)
            kl = (old.exp() * (old - new)).sum((-2, -1)).mean()
        self.writer.add_scalar("actor/kl", kl.item(), global_step)
        self.policy.refresh()


def transition_observations(next_obs, boundaries, infos):
    """next_obs with autoreset rows replaced by the finished episode's final observation."""
    if not boundaries.any():
        return next_obs
    transition_obs = next_obs.copy()
    for index in np.flatnonzero(boundaries):
        transition_obs[index] = infos["final_observation"][index]
    return transition_obs




def evaluate(policy: HostPolicy, bins, eval_envs, max_action):
    """Deterministic policy returns (the categoricals' means), one episode per env, in lockstep."""
    states = np.stack([env.reset()[0] for env in eval_envs]).astype(np.float32)
    returns = np.zeros(len(eval_envs))
    active = np.ones(len(eval_envs), dtype=bool)
    while active.any():
        _, actions, _ = host_policy_stats(policy(states), bins)
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
    # Full FP32 by default (per-bin Q differences sit below TF32 rounding of Q-sized readouts); the
    # critic and encoder steps opt back into the base's TF32.
    configure_runtime(matmul_precision="highest", allow_tf32=False)
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
    bins = np.linspace(-1, 1, args.num_bins, dtype=np.float32)

    agent = DiscoTD7Agent(state_dim, action_dim, torch.as_tensor(bins, device=device), args, device, writer)
    timer = PhaseTimer()
    warmup_logits = np.broadcast_to(-0.5 * (bins / args.warmup_sigma) ** 2, (*shape, args.num_bins))

    start_time = time.time()
    # One unrecorded horizon of warmup-policy draws spreads the episode ages across the envs.
    horizon = episode_horizon(args.env_id)
    phase_offsets = compute_phase_offsets(args.num_envs, horizon, args.seed)
    writer.add_text("initial_phase_offsets", ",".join(map(str, phase_offsets)))
    warm = run_phase_warmup(
        envs, obs_norm=None, horizon=horizon, phase_offsets=phase_offsets, seed=args.seed,
        act_fn=lambda _: bins[sample_bins(host_policy_stats(warmup_logits, bins)[0], rng)] * max_action,
    )
    obs = warm.next_obs
    global_step = warm.transitions
    suppress = warm.suppress_mask.copy()
    num_iterations = (args.total_timesteps - warm.transitions) // agent.rollout_size
    for iteration in range(1, num_iterations + 1):
        training = iteration > args.warmup_rollouts
        for step in range(args.num_steps):
            with timer.span("rollout", use_cuda=False):
                logits = agent.policy(obs) if training else warmup_logits
                probs, mu, sigma = host_policy_stats(logits, bins)
                bin_index = sample_bins(probs, rng)
                action = bins[bin_index]
            with timer.span("env", use_cuda=False):
                next_obs, reward, terminated, truncated, infos = envs.step(action * max_action)
            with timer.span("rollout", use_cuda=False):
                next_obs = next_obs.astype(np.float32)
                boundaries = terminated | truncated
                agent.stage(
                    step, state=obs, action=action, mu=mu, sigma=sigma, eps=(action - mu) / sigma,
                    pre_action=action, bin=bin_index, logits=logits.reshape(args.num_envs, -1),
                    next_state=transition_observations(next_obs, boundaries, infos),
                    reward=reward[:, None], not_done=1.0 - terminated[:, None], reset=boundaries[:, None],
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
                eval_return = evaluate(agent.policy, bins, eval_envs, max_action)
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

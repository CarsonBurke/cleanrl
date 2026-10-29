# TD7 LATENT ACTION TWO-CRITIC v4 — td7_latentaction_v1's latent side, plus an optional learned head judged
# by its own action critic.
#
# Latent side (unchanged from v1): the actor ends in a latent_dim belief N(m(s), sigma(s)^2); z = clip(m +
# sigma eps) is the "action" for replay, SALE's zsa(zs, z), the belief critic Q~(s, m, sigma, z) with its
# Hermite labels, and the actor objective. A head a = tanh(z W + MLP(z)) maps z to the motor command; W starts
# as v1's 1.5 * orthonormal decoder and the MLP's last layer at zero, so every arm starts from v1's exact map.
# Action side (learn_head): a plain TD7 critic Qa(s, a) with its own SALE encoder learns from the executed
# motor actions, and the head (slow, head_lr) ascends Qa at its decoding of the detached mean intention m;
# the trunk never sees Qa and the head never sees Q~. v3 had the latent side; v4 puts v1's belief critic back.
# Ablations (flags):
#   learn_head     False: v1's frozen decoder (no Qa, no head training).
#   entropy_space  "latent": v1's sum log sigma over latent dims; "motor": 0.5 logdet(J diag(sigma^2) J^T)
#                  with J the detached pre-tanh head Jacobian at m (target action_dim log target_sigma).
#   action_noise   std of post-tanh Gaussian motor noise (as v27 adds after its tanh); replay keeps z and
#                  the noisy executed action.
#   arm a: defaults | b: --no-learn-head | c: --action-noise 0.1 | d: --entropy-space latent
# Hypothesis: the belief critic's intention credit plus a head that turns intentions into Qa-best actions beats
# both v1's fixed random decoder and v3's plain latent critic.
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

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
    fresh_tail_weight: float = 0.01
    """fresh-data optimism uses steps whose truncated-return tail weight gamma^k is below this"""
    latent_dim: int = 64
    """dimensionality of the actor's stochastic final latent, the critic's action"""
    decoder_gain: float = 1.5
    """initial head a = tanh(gain * W z), W's rows orthonormal; latent sigma 0.1 -> ~0.15 per-joint action noise"""
    learn_head: bool = True
    """train the head against an action critic Qa(s, a); False keeps v1's frozen decoder (no Qa)"""
    head_lr: float = 1e-4
    """the learning rate of the head optimizer (slow: Q~ values latents through the current head)"""
    qa_target_noise: float = 0.2
    """TD7 target policy smoothing std of Qa's target action"""
    qa_noise_clip: float = 0.5
    """clip of Qa's target policy smoothing noise"""
    entropy_space: Literal["latent", "motor"] = "motor"
    """space of the actor's entropy bonus and alpha target: latent sum log sigma, or pushed-forward motor noise"""
    action_noise: float = 0.0
    """std of Gaussian noise added to the executed action after the head's tanh (0: latent noise only)"""


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


class TwinCritic(nn.Module):
    """Fused plain TD7 twin critic Q(s, a, zsa, zs) for the motor action; returns (batch, 2)."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.hdim = hdim
        self.q0 = nn.Linear(state_dim + action_dim, 2 * hdim)  # both members side by side
        self.q1 = EnsembleLinear(2, 2 * zs_dim + hdim, hdim)
        self.q2 = EnsembleLinear(2, hdim, hdim)
        self.q3 = EnsembleLinear(2, hdim, 1)
        bound = (state_dim + action_dim) ** -0.5  # per-member nn.Linear init
        nn.init.uniform_(self.q0.weight, -bound, bound)
        nn.init.uniform_(self.q0.bias, -bound, bound)

    def forward(self, state, action, zsa, zs):
        batch = state.shape[0]
        h = avg_l1_norm(self.q0(torch.cat([state, action], 1)).view(batch, 2, self.hdim)).transpose(0, 1)
        h = F.elu(self.q1(torch.cat([h, torch.cat([zsa, zs], 1).expand(2, batch, -1)], 2)))
        h = F.elu(self.q2(h))
        return self.q3(h).squeeze(-1).transpose(0, 1)


class Head(nn.Module):
    """The motor head: a = tanh(z W + MLP(z)), W initialized to v1's gain * orthonormal decoder, MLP to zero."""

    def __init__(self, latent_dim, action_dim, gain, seed, hdim=256):
        super().__init__()
        # Same draw and QR as v1's FrozenDecoder: the initial map is v1's, bit for bit in its weights.
        basis = np.linalg.qr(np.random.default_rng(seed).standard_normal((latent_dim, action_dim)))[0]
        self.base = nn.Parameter(torch.as_tensor((gain * basis).astype(np.float32)))  # (latent_dim, action_dim)
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
        """Analytic du/dz of the pre-tanh motor command u, (batch, action_dim, latent_dim)."""
        p1 = self.h1(z)
        p2 = self.h2(F.elu(p1))
        d1 = torch.where(p1 > 0, 1.0, p1.exp())  # ELU'
        d2 = torch.where(p2 > 0, 1.0, p2.exp())
        mlp = ((self.h3.weight * d2[:, None, :]) @ self.h2.weight * d1[:, None, :]) @ self.h1.weight
        return self.base.T + mlp


class HostPolicy:
    """NumPy mirror of fixed_encoder.zs + actor + head for acting without touching the GPU.

    ``refresh`` pulls the device weights in one flat D2H copy; call it whenever the device policy
    changes (after every training burst). ``copy_from`` snapshots another mirror (checkpointing).
    """

    def __init__(self, encoder: Encoder, actor: Actor, head: Head):
        self._sources = [
            (encoder.zs1.weight,), (encoder.zs1.bias,), (encoder.zs2.weight,), (encoder.zs2.bias,),
            (encoder.zs3.weight,), (encoder.zs3.bias,), (actor.l0.weight,), (actor.l0.bias,),
            (actor.l1.weight,), (actor.l1.bias,), (actor.l2.weight,), (actor.l2.bias,),
            (actor.l3.weight, actor.l3_logstd.weight), (actor.l3.bias, actor.l3_logstd.bias),
            (head.h1.weight,), (head.h1.bias,), (head.h2.weight,), (head.h2.bias,),
            (head.h3.weight,), (head.h3.bias,), (head.base,),
        ]
        shapes = [(sum(t.shape[0] for t in group),) + tuple(group[0].shape[1:]) for group in self._sources]
        self._flat = np.zeros(sum(int(np.prod(s)) for s in shapes), dtype=np.float32)
        self._views, offset = [], 0
        for group, shape in zip(self._sources, shapes):
            size = int(np.prod(shape))
            view = self._flat[offset : offset + size].reshape(shape)
            # nn.Linear weights as (in, out); head.base already is (latent_dim, action_dim)
            self._views.append(view.T if view.ndim == 2 and group[0] is not head.base else view)
            offset += size
        self.action_dim = actor.l3.out_features

    def refresh(self):
        flat = torch.cat([t.detach().reshape(-1) for group in self._sources for t in group])
        self._flat[:] = flat.cpu().numpy()

    def copy_from(self, other: "HostPolicy"):
        self._flat[:] = other._flat

    def head(self, z):
        """latent (batch, latent_dim) -> motor action in [-1, 1]."""
        (h1, c1, h2, c2, h3, c3, base) = self._views[14:]
        h = z @ h1 + c1
        h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))
        h = h @ h2 + c2
        h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))
        return np.tanh(z @ base + h @ h3 + c3)

    def __call__(self, state):
        (w1, b1, w2, b2, w3, b3, w0, b0, wl1, bl1, wl2, bl2, wh, bh) = self._views[:14]
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
    def __init__(self, state_dim, action_dim, motor_dim, max_episode_steps, args: Args, device, writer: SummaryWriter):
        """action_dim is the latent's width (v1's "action"); motor_dim is the environment's action width."""
        assert args.hermite_order in (0, 1, 2), "hermite_order must be 0, 1 or 2"
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim, self.motor_dim = state_dim, action_dim, motor_dim
        self.learn_head = args.learn_head
        adam = dict(fused=True)

        self.actor = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_target = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder_target = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        # Built after v1's modules, in every arm, so the latent side's initialization matches v1 and the arms.
        self.head = Head(action_dim, motor_dim, args.decoder_gain, args.seed, args.hidden_dim).to(device)
        self.head_target = Head(action_dim, motor_dim, args.decoder_gain, args.seed, args.hidden_dim).to(device)
        self.head_base_init = self.head.base.detach().clone()  # v1's decoder, the reference for head_correction
        pairs = [
            (self.actor_target, self.actor),
            (self.critic_target, self.critic),
            (self.fixed_encoder, self.encoder),
            (self.fixed_encoder_target, self.encoder),
            (self.head_target, self.head),
        ]
        if self.learn_head:
            # Action side: a plain TD7 critic of the executed motor action with its own SALE encoder.
            self.critic_a = TwinCritic(state_dim, motor_dim, args.zs_dim, args.hidden_dim).to(device)
            self.critic_a_target = TwinCritic(state_dim, motor_dim, args.zs_dim, args.hidden_dim).to(device)
            self.encoder_a = Encoder(state_dim, motor_dim, args.zs_dim, args.hidden_dim).to(device)
            self.fixed_encoder_a = Encoder(state_dim, motor_dim, args.zs_dim, args.hidden_dim).to(device)
            self.fixed_encoder_a_target = Encoder(state_dim, motor_dim, args.zs_dim, args.hidden_dim).to(device)
            pairs += [
                (self.critic_a_target, self.critic_a),
                (self.fixed_encoder_a, self.encoder_a),
                (self.fixed_encoder_a_target, self.encoder_a),
            ]
        else:
            self.head.requires_grad_(False)  # v1's frozen decoder
        for target, source in pairs:
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        critic_params, encoder_params = list(self.critic.parameters()), list(self.encoder.parameters())
        if self.learn_head:
            critic_params += list(self.critic_a.parameters())
            encoder_params += list(self.encoder_a.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=args.critic_lr, **adam)
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=args.encoder_lr, **adam)
        # (destination, source) pairs of every hard update, in TD7's order (target before fixed).
        pairs = [
            (self.actor_target, self.actor),
            (self.critic_target, self.critic),
            (self.fixed_encoder_target, self.fixed_encoder),
            (self.fixed_encoder, self.encoder),
        ]
        if self.learn_head:
            pairs += [
                (self.head_target, self.head),
                (self.critic_a_target, self.critic_a),
                (self.fixed_encoder_a_target, self.fixed_encoder_a),
                (self.fixed_encoder_a, self.encoder_a),
            ]
        self._hard_updates = [(list(d.parameters()), list(s.parameters())) for d, s in pairs]

        # Entropy is confined to the actor loss; autotuning holds it at target_sigma per dim of its space.
        entropy_dim = motor_dim if args.entropy_space == "motor" else action_dim
        self.target_entropy = entropy_dim * float(np.log(args.target_sigma))
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
        optimizers = [self.encoder_optimizer, self.critic_optimizer, self.actor_optimizer, self.alpha_optimizer]
        if self.learn_head:
            self.head_optimizer = torch.optim.Adam(self.head.parameters(), lr=args.head_lr, **adam)
            optimizers.append(self.head_optimizer)

        # Replay rows retain the clipped latent ("action"), its pre-action latent, and the executed motor action.
        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (("state", s), ("action", a), ("mu", a), ("sigma", a), ("eps", a), ("pre_action", a), ("motor", motor_dim), ("next_state", s), ("reward", 1), ("not_done", 1)):
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
        # Per-critic target bounds (index 0: belief critic Q~, 1: action critic Qa).
        self.running_max = torch.full((2,), -1e8, **f32)
        self.running_min = torch.full((2,), 1e8, **f32)
        self.max_target = torch.zeros(2, **f32)
        self.min_target = torch.zeros(2, **f32)
        self.metric_names = [
            "encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std",
            "actor_loss", "sigma_mean", "sigma_min", "sigma_max", "smoothing_gap", "preaction_gap",
            "encoder_a_loss", "critic_a_loss", "qa_values", "head_loss", "head_correction", "action_entropy",
        ]
        self.metrics = torch.zeros(len(self.metric_names), **f32)
        self.fresh_cut = math.ceil(math.log(args.fresh_tail_weight) / math.log(args.gamma))

        maybe_compile = (
            (lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}))
            if args.compile
            else (lambda fn: fn)
        )
        # One backward per compiled function: the action side gets its own encoder and critic functions.
        self._encoder_loss = maybe_compile(self._encoder_loss_fn)
        self._critic_loss = maybe_compile(self._critic_loss_fn)
        self._encoder_a_loss = maybe_compile(self._encoder_a_loss_fn)
        self._critic_a_loss = maybe_compile(self._critic_a_loss_fn)
        self._actor_loss = maybe_compile(self._actor_loss_fn)
        self._entropy = maybe_compile(self._entropy_fn)
        # training_steps % policy_freq == 0 selects the actor-step graph (TD7's delayed update).
        self.graphs = capture_steps(
            [self._make_step(update_actor=False), self._make_step(update_actor=True)],
            optimizers=optimizers,
            state_tensors=[self.priority, self.max_priority, self.running_max, self.running_min, self.metrics],
        )

        self.policy = HostPolicy(self.fixed_encoder, self.actor, self.head)
        self.checkpoint_policy = HostPolicy(self.fixed_encoder, self.actor, self.head)
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

    def _encoder_a_loss_fn(self, batch):
        """Qa's SALE: zsa(zs(s), a) predicts zs(s') from the executed motor action."""
        f = self._fields
        with torch.no_grad():
            next_zs = self.encoder_a.zs(batch[:, f["next_state"]])
        pred_zs = self.encoder_a.zsa(self.encoder_a.zs(batch[:, f["state"]]), batch[:, f["motor"]])
        return F.mse_loss(pred_zs, next_zs)

    def _critic_a_loss_fn(self, batch, min_target, max_target):
        """Plain TD7 TD loss of Qa at the executed motor action; the target action is the target head's
        decoding of the target actor's mean latent, smoothed in action units."""
        f, args = self._fields, self.args
        state, next_state, motor = batch[:, f["state"]], batch[:, f["next_state"]], batch[:, f["motor"]]
        with torch.no_grad():
            next_mu, _ = self.actor_target(next_state, self.fixed_encoder_target.zs(next_state))
            next_a = self.head_target(next_mu)
            noise = (torch.randn_like(next_a) * args.qa_target_noise).clamp(-args.qa_noise_clip, args.qa_noise_clip)
            next_a = (next_a + noise).clamp(-1, 1)
            next_zs = self.fixed_encoder_a_target.zs(next_state)
            q_target = self.critic_a_target(next_state, next_a, self.fixed_encoder_a_target.zsa(next_zs, next_a), next_zs)
            q_target = q_target.min(1, keepdim=True)[0].clamp(min_target, max_target)
            q_target = batch[:, f["reward"]] + batch[:, f["not_done"]] * args.gamma * q_target
            fixed_zs = self.fixed_encoder_a.zs(state)
            zsa = self.fixed_encoder_a.zsa(fixed_zs, motor)
        q = self.critic_a(state, motor, zsa, fixed_zs)
        td = (q - q_target).abs()
        loss = lap_huber(td, args.min_priority)
        with torch.no_grad():
            diagnostics = torch.stack([q.mean(), q_target.max(), q_target.min()])
        return loss, td.detach(), diagnostics, fixed_zs

    def _motor_entropy(self, mu, log_std):
        """Gaussian entropy (0.5 logdet, up to a constant) of the motor-command noise J diag(sigma^2) J^T, J the
        head's detached pre-tanh Jacobian at the mean. float64: TF32 products lose definiteness."""
        with torch.no_grad():
            jac = self.head.jacobian(mu.detach()).double()
        cov = (jac * (2 * log_std.double()).exp()[:, None, :]) @ jac.transpose(1, 2)
        cov = cov + 1e-8 * torch.eye(self.motor_dim, device=cov.device, dtype=cov.dtype)
        chol, _ = torch.linalg.cholesky_ex(cov)  # no info check: no host sync inside the graph
        return torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(-1).float()

    def _policy_entropy(self, mu, log_std):
        """The entropy the actor is paid and alpha is tuned on, in args.entropy_space."""
        if self.args.entropy_space == "motor":
            return self._motor_entropy(mu, log_std)
        return log_std.sum(-1)

    def _actor_loss_fn(self, state, fixed_zs, fixed_zs_a, alpha):
        """Actor: v1's belief-critic objective (+ entropy). Head: ascend Qa at its decoding of the detached mean
        latent. The head loss cannot reach the actor (m.detach()) and the actor loss cannot reach the head
        (Q~ never sees it; the entropy's Jacobian is detached)."""
        mu, log_std = self.actor(state, fixed_zs)
        latent = torch.cat([mu, log_std.exp()], 1)
        q = self.critic(state, latent, mu, self.fixed_encoder.zsa(fixed_zs, mu), fixed_zs)
        actor_loss = -q.mean() - alpha * self._policy_entropy(mu, log_std).mean()
        if not self.learn_head:
            return actor_loss, torch.zeros_like(actor_loss), log_std, torch.zeros_like(actor_loss)
        intention = mu.detach()
        a = self.head(intention)
        head_loss = -self.critic_a(state, a, self.fixed_encoder_a.zsa(fixed_zs_a, a), fixed_zs_a).mean()
        with torch.no_grad():
            # Total drift of the head from v1's decoder: mean |head(m) - tanh(m W_init)|.
            correction = (a - torch.tanh(intention @ self.head_base_init)).abs().mean()
        return actor_loss, head_loss, log_std, correction

    def _entropy_fn(self, state, fixed_zs):
        """Post-update entropy for the alpha step, and the motor entropy (logged in every arm)."""
        with torch.no_grad():
            mu, log_std = self.actor(state, fixed_zs)
            motor = self._motor_entropy(mu, log_std)
            return (motor if self.args.entropy_space == "motor" else log_std.sum(-1)), motor.mean()

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
            if self.learn_head:
                encoder_a_loss = self._encoder_a_loss(batch)
                (encoder_loss + encoder_a_loss).backward()
            else:
                encoder_loss.backward()
            self.encoder_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=False)
            critic_loss, td, diagnostics, fixed_zs = self._critic_loss(batch, self.min_target[0], self.max_target[0])
            fixed_zs_a = None
            if self.learn_head:
                critic_a_loss, td_a, diagnostics_a, fixed_zs_a = self._critic_a_loss(
                    batch, self.min_target[1], self.max_target[1]
                )
                (critic_loss + critic_a_loss).backward()
            else:
                critic_loss.backward()
            self.critic_optimizer.step()

            with torch.no_grad():
                td_max = td.max(1)[0]
                if self.learn_head:
                    td_max = torch.maximum(td_max, td_a.max(1)[0])
                    target_max = torch.stack([diagnostics[3], diagnostics_a[1]])
                    target_min = torch.stack([diagnostics[4], diagnostics_a[2]])
                else:
                    target_max, target_min = diagnostics[3].expand(2), diagnostics[4].expand(2)
                torch.maximum(self.running_max, target_max, out=self.running_max)
                torch.minimum(self.running_min, target_min, out=self.running_min)
                priority = td_max.clamp(min=args.min_priority).pow(args.lap_alpha)
                self.priority[ind] = priority
                torch.maximum(self.max_priority, priority.max(), out=self.max_priority)
                self.metrics[0].copy_(encoder_loss.detach())
                self.metrics[1].copy_(critic_loss.detach())
                self.metrics[2:5].copy_(diagnostics[:3])
                self.metrics[9:11].copy_(diagnostics[5:7])
                if self.learn_head:
                    self.metrics[11:14].copy_(torch.stack([
                        encoder_a_loss.detach(), critic_a_loss.detach(), diagnostics_a[0]
                    ]))

            if update_actor:
                state = batch[:, self._fields["state"]]
                self.actor_optimizer.zero_grad(set_to_none=False)
                parameters = list(self.actor.parameters())
                if self.learn_head:
                    self.head_optimizer.zero_grad(set_to_none=False)
                    parameters += list(self.head.parameters())
                actor_loss, head_loss, log_std, correction = self._actor_loss(
                    state, fixed_zs, fixed_zs_a, self.log_alpha.detach().exp()
                )
                # One backward through the single compiled loss; the two losses share no parameters.
                (actor_loss + head_loss).backward(inputs=parameters)
                self.actor_optimizer.step()
                if self.learn_head:
                    self.head_optimizer.step()
                entropy, action_entropy = self._entropy(state, fixed_zs)
                self.alpha_optimizer.zero_grad(set_to_none=False)
                alpha_loss = -(self.log_alpha.exp() * (self.target_entropy - entropy)).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    sigma = log_std.detach().exp()
                    self.metrics[5:9].copy_(torch.stack([actor_loss.detach(), sigma.mean(), sigma.min(), sigma.max()]))
                    self.metrics[14:17].copy_(torch.stack([head_loss.detach(), correction, action_entropy]))

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
            names = ["encoder_loss", "critic_loss", "q_values", "actor_loss", "hermite_r2", "tangent_std", "smoothing_gap", "preaction_gap"]
            if self.learn_head:
                names += ["encoder_a_loss", "critic_a_loss", "qa_values", "head_loss", "head_correction"]
            for name in names:
                self.writer.add_scalar(f"losses/{name}", values[name], self.training_steps)
            (max_q, max_qa), (min_q, min_qa) = self.max_target.tolist(), self.min_target.tolist()
            self.writer.add_scalar("charts/max_target", max_q, self.training_steps)
            self.writer.add_scalar("charts/min_target", min_q, self.training_steps)
            if self.learn_head:
                self.writer.add_scalar("charts/max_target_a", max_qa, self.training_steps)
                self.writer.add_scalar("charts/min_target_a", min_qa, self.training_steps)
            if self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max", "action_entropy"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.training_steps)

    # ---- fresh-data optimism ------------------------------------------------------------------
    def _log_fresh(self, rows, rewards, terminated, global_step):
        """Critic prediction minus realized discounted return on an episode no update has seen.

        Truncated episodes keep only steps whose unobserved tail weighs less than fresh_tail_weight.
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
        latent = torch.cat([rows[:, f["mu"]], rows[:, f["sigma"]]], 1)
        realized = torch.as_tensor(returns[:valid], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            zs = self.fixed_encoder.zs(state)
            q = self.critic(state, latent, rows[:, f["pre_action"]], self.fixed_encoder.zsa(zs, action), zs).mean(1)
            scale = realized.abs().mean().clamp(min=1e-8)
            error = q - realized
            stats = [error.mean(), error.mean() / scale, error.abs().mean()]
            names = ["optimism", "rel_optimism", "abs_err"]
            if self.learn_head:
                # Qa at the executed (possibly noisy) motor action.
                motor = rows[:, f["motor"]]
                zs_a = self.fixed_encoder_a.zs(state)
                error_a = self.critic_a(state, motor, self.fixed_encoder_a.zsa(zs_a, motor), zs_a).mean(1) - realized
                stats += [error_a.mean(), error_a.mean() / scale, error_a.abs().mean()]
                names += ["optimism_a", "rel_optimism_a", "abs_err_a"]
            stats = torch.stack(stats).tolist()
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
    """Deterministic checkpoint-policy returns (the head's decoding of the latent mean, no action noise), one
    episode per env, in lockstep."""
    states = np.stack([env.reset()[0] for env in eval_envs]).astype(np.float32)
    returns = np.zeros(len(eval_envs))
    active = np.ones(len(eval_envs), dtype=bool)
    while active.any():
        actions = policy.head(policy(states)[0])
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

    # From here on the agent's "action" is the latent; the environment receives head(latent) (+ action noise).
    agent = TD7Agent(state_dim, args.latent_dim, action_dim, env.spec.max_episode_steps, args, device, writer)
    timer = PhaseTimer()
    warmup_mu = np.zeros(args.latent_dim, dtype=np.float32)
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
                mu, log_std = agent.policy(obs[None])
                mu, sigma = mu[0], np.exp(log_std[0])
            else:
                mu, sigma = warmup_mu, warmup_sigma
            eps = rng.standard_normal(args.latent_dim, dtype=np.float32)
            pre_action = mu + sigma * eps
            action = np.clip(pre_action, -1, 1)  # the clipped latent: what the critic and SALE see
            motor, motor_mean = agent.policy.head(np.stack([action, mu]))
            if args.action_noise > 0:  # post-tanh motor noise; replay stores the action actually executed
                motor = np.clip(motor + args.action_noise * rng.standard_normal(action_dim, dtype=np.float32), -1, 1)
            ep_noise_sq += float(np.square(motor - motor_mean).mean())
            ep_saturated += float((np.abs(motor) > 0.99).mean())
        with timer.span("env", use_cuda=False):
            next_obs, reward, terminated, truncated, _ = env.step(motor * max_action)
        next_obs = next_obs.astype(np.float32)
        # The original bootstraps at the timeout step even when the env truly terminated there.
        agent.stage(
            state=obs, action=action, mu=mu, sigma=sigma, eps=eps, pre_action=pre_action, motor=motor,
            next_state=next_obs,
            reward=reward, not_done=1.0 - float(terminated and not truncated),
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
        checkpoint = {
            "actor": agent.actor.state_dict(),
            "encoder": agent.encoder.state_dict(),
            "critic": agent.critic.state_dict(),
            "head": agent.head.state_dict(),
            "checkpoint_policy": agent.checkpoint_policy._flat,
        }
        if args.learn_head:
            checkpoint |= {"critic_a": agent.critic_a.state_dict(), "encoder_a": agent.encoder_a.state_dict()}
        torch.save(checkpoint, model_path)
        print(f"model saved to {model_path}")

    env.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()

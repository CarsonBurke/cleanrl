# TD7 INTENTION CONTRASTIVE v1 — the judge classifies futures instead of regressing a scalar.
#
# Derived from intention/td7_intention_judge_v1; premise kept: the judge sees state, zs and the
# intention g = zsa(zs, a) only, so a reaches it through zsa alone. The judge becomes a TD-InfoNCE
# critic (contrastive RL, Eysenbach et al. 2022; TD-InfoNCE, Zheng et al. 2023): f(s, g, s_f) =
# exp(log_scale) cos(phi(s, zs, g), psi(s_f)), and a softmax over a contrast set estimates the
# discounted occupancy of g's futures. Contrast set = the batch's own next states s'_j (no extra env or
# action samples) plus one learned absorbing column ⊥ with reward 0.
#
# Loss, row i, member m:  CE(f_m(s_i, g_i, .), y_i),  g_i = zsa(fixed_zs_i, a_executed_i),
#   y_i = (1-gamma) e_i + gamma [not_done_i w_i + done_i e_⊥],   w_i = mean_m softmax f_target_m(s'_i, g'_i, .),
#   g'_i = zsa_target(zs'_i, clip(mu_target(s'_i) + clip(0.2 N, +-0.5), -1, 1)), all target nets (stop-grad).
# Terminals: reaching s'_i is step 1 (mass 1-gamma, reward r_i); after it the chain sits in a zero-reward
# absorbing state, so the gamma mass goes to ⊥, not to s'_i. One-hot on s'_i would read a terminal step
# out as r_i / (1-gamma) instead of r_i. ⊥ mass inside w_i carries termination risk backwards.
# Readout: Q(s, g) = 1/(1-gamma) sum_j p_j r_j, r_j the reward of the transition that produced s'_j
# (r_⊥ = 0). Treats r as a function of the reached state: HalfCheetah's velocity reward is, ctrl cost is not.
# Cosine logits are bounded by the scale (learned, clamped to [1, 100]), so CRL's logsumexp penalty, which
# pins the free row offset of raw dot products, defaults off. Twin phi members, shared psi: labels are the
# members' mean target distribution and the actor ascends the mean readout (a min of two distributions
# is not a distribution; the judge's actor also used the twin mean).
#
# Why: every row is classified against N futures, an N-way signal per sample instead of one bootstrapped
# scalar, and the pure arm has no scalar TD regression or target clamp at all. Known risk: the pure
# readout is a convex combination of batch rewards, so it cannot value beyond the best batch future and is
# coarse at N = 256 (`readout_bank` M > 0 reads out over M sampled replay rows instead; ⊥'s logit gets
# + log(M/N) so its mass does not depend on the set size).
#
# Arms (--judge):
# - contrastive: mu ascends the readout Q(s, zs, zsa(zs, mu)); sigma ascends the antithetic readout belief
#   value; entropy/alpha as the judge. No TD critic, and no center channel: the judge trains on g(a) only
#   (hermite_order is hybrid-only). charts/absorbing_mass watches ⊥ (reward 0, unanchored without
#   terminations) at the actor's intentions, an off-data column the actor could climb toward. LAP is off (uniform replay): the classifier learns
#   ratios against the sampling marginal, so loss-driven priorities would move its reference measure, and
#   the CE's irreducible floor H(y_i) makes per-row loss no error signal. fresh/* scores the readout
#   against a freshly sampled replay batch.
# - hybrid: the judge's TD critic, LAP and actor unchanged; the critic's q2 features (the layer its linear
#   value head reads) are also phi of a TD-InfoNCE loss with a separate psi, weight contrastive_weight.
#
# Hypothesis: the N-way classification signal shapes the intention representation better than scalar
# TD; hybrid tests that as an auxiliary on the judge, contrastive as the judge itself.
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
    hermite_order: int = 1
    """hybrid center channel label: 0 plain Q(g_mu) regression; 1 + dQ/dg . J_zsa (sigma eps). Unused by --judge contrastive"""
    belief_sigma: bool = True
    """sigma ascends the antithetic belief value of its intentions (order-2 Hermite in g); off: entropy only"""
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
    """per-dim std of the sampled Gaussian smoothing of the target policy's mean, TD7's 0.2"""
    target_noise_clip: float = 0.5
    """clip of the target smoothing noise, TD7's 0.5"""
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

    # Contrastive judge (TD-InfoNCE)
    judge: Literal["contrastive", "hybrid"] = "contrastive"
    """contrastive: the readout of a TD-InfoNCE critic is the judge; hybrid: judge_v1's TD critic + TD-InfoNCE on its q2 features"""
    contrastive_dim: int = 256
    """width of phi and psi in the contrastive arm (hybrid: phi is the critic's q2 features, width hidden_dim)"""
    contrastive_weight: float = 0.1
    """hybrid weight of the TD-InfoNCE loss; init CE ~ log(N+1) = 5.55 vs ~0.5-1 TD Huber on warmup rewards"""
    init_logit_scale: float = 10.0
    """initial cosine-logit scale exp(log_scale); learned, clamped to [1, 100]"""
    logsumexp_coef: float = 0.0
    """CRL's logsumexp^2 penalty on the logits; off since cosine logits are bounded"""
    readout_bank: int = 0
    """futures of the value readout: 0 = the batch's own next states; M > 0 = M sampled replay rows (readout only)"""


# Log-std bounds of the state-dependent Gaussian: sigma in [~6.7e-3, 1] for actions in [-1, 1].
LOG_STD_MIN = -5
LOG_STD_MAX = 0


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


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


class IntentionTwinCritic(nn.Module):
    """Fused twin judge Q(s, zs, g) of an intention g = zsa(zs, a); no action-side input exists.

    q0 embeds the state alone (TD7's q0 also took the action); g enters with zs at q1, so the only
    tangent is along g and it enters through q1's g-columns. ``features=True`` also returns the
    (2, batch, hdim) q2 activations, the representation the linear value head q3 reads.
    """

    def __init__(self, state_dim, zs_dim=256, hdim=256, out_dim=1):
        super().__init__()
        self.hdim, self.zs_dim = hdim, zs_dim
        self.q0 = nn.Linear(state_dim, 2 * hdim)  # both members side by side (nn.Linear init == per member)
        self.q1 = EnsembleLinear(2, hdim + 2 * zs_dim, hdim)  # [state embedding | g | zs]
        self.q2 = EnsembleLinear(2, hdim, hdim)
        self.q3 = EnsembleLinear(2, hdim, out_dim)

    def trunk_input(self, state, g, zs):
        batch = state.shape[0]
        h = avg_l1_norm(self.q0(state).view(batch, 2, self.hdim)).transpose(0, 1)
        return torch.cat([h, torch.cat([g, zs], 1).expand(2, batch, -1)], 2)

    def forward(self, state, g, zs, t_g=None, features=False):
        h = self.trunk_input(state, g, zs)
        if t_g is None:
            h = F.elu(self.q2(F.elu(self.q1(h))))
            q = self.q3(h).squeeze(-1).transpose(0, 1)
            return (q, h) if features else q
        t = torch.bmm(t_g.expand(2, state.shape[0], -1), self.q1.weight[:, self.hdim : self.hdim + self.zs_dim])
        h, t = elu_jvp(self.q1(h), t)
        h, t = elu_jvp(*self.q2.jvp(h, t))
        q, t = self.q3.jvp(h, t)
        q, t = q.squeeze(-1).transpose(0, 1), t.squeeze(-1).transpose(0, 1)
        return (q, t, h) if features else (q, t)


class IntentionPhi(IntentionTwinCritic):
    """Twin contrastive embedding phi(s, zs, g) -> (2, batch, d): the judge's trunk with a d-wide head."""

    def __init__(self, state_dim, zs_dim=256, hdim=256, out_dim=256):
        super().__init__(state_dim, zs_dim, hdim, out_dim)

    def forward(self, state, g, zs):
        return self.q3(F.elu(self.q2(F.elu(self.q1(self.trunk_input(state, g, zs))))))


class FuturePsi(nn.Module):
    """psi(s_f) from [fixed-encoder zs(s_f) | raw s_f], plus the absorbing column ⊥ and the logit scale.

    The raw state gets TD7's avg-L1-normalized linear embedding (as q0 / the actor's l0) before the MLP.
    """

    def __init__(self, state_dim, zs_dim=256, hdim=256, out_dim=256, init_scale=10.0):
        super().__init__()
        self.p0 = nn.Linear(state_dim, hdim)
        self.p1 = nn.Linear(hdim + zs_dim, hdim)
        self.p2 = nn.Linear(hdim, hdim)
        self.p3 = nn.Linear(hdim, out_dim)
        self.absorbing = nn.Parameter(torch.randn(out_dim) * out_dim**-0.5)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self, zs_f, s_f):
        h = torch.cat([avg_l1_norm(self.p0(s_f)), zs_f], 1)
        return self.p3(F.elu(self.p2(F.elu(self.p1(h)))))

    def logits(self, phi, psi_f, absorbing_offset=0.0):
        """(2, B, d) x (N, d) -> (2, B, N + 1) cosine logits; column N is ⊥ (+ absorbing_offset)."""
        futures = F.normalize(torch.cat([psi_f, self.absorbing[None]]), dim=-1)
        logits = self.log_scale.exp() * F.normalize(phi, dim=-1) @ futures.T
        if absorbing_offset:
            logits = torch.cat([logits[..., :-1], logits[..., -1:] + absorbing_offset], -1)
        return logits


def readout(logits, rewards, gamma):
    """Q(s, g) per member, (B, 2): 1/(1-gamma) E_{p = softmax(logits)}[r] over futures; r_⊥ = 0."""
    return (logits.softmax(-1)[..., :-1] @ rewards).transpose(0, 1) / (1 - gamma)


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
        assert args.hermite_order in (0, 1), "hermite_order must be 0 or 1 (order 2 needs a sigma input)"
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        self.contrastive = args.judge == "contrastive"
        adam = dict(fused=True)

        def pair(make):
            return make().to(device), make().to(device)

        hdim, zs_dim = args.hidden_dim, args.zs_dim
        self.actor, self.actor_target = pair(lambda: Actor(state_dim, action_dim, zs_dim, hdim))
        # The contrastive arm's judge is phi itself; hybrid's phi is the TD critic's q2 features.
        if self.contrastive:
            psi_dim = args.contrastive_dim
            self.critic, self.critic_target = pair(lambda: IntentionPhi(state_dim, zs_dim, hdim, psi_dim))
        else:
            psi_dim = hdim
            self.critic, self.critic_target = pair(lambda: IntentionTwinCritic(state_dim, zs_dim, hdim))
        self.psi, self.psi_target = pair(lambda: FuturePsi(state_dim, zs_dim, hdim, psi_dim, args.init_logit_scale))
        self.encoder = Encoder(state_dim, action_dim, zs_dim, hdim).to(device)
        self.fixed_encoder = Encoder(state_dim, action_dim, zs_dim, hdim).to(device)
        self.fixed_encoder_target = Encoder(state_dim, action_dim, zs_dim, hdim).to(device)
        for target, source in (
            (self.actor_target, self.actor),
            (self.critic_target, self.critic),
            (self.psi_target, self.psi),
            (self.fixed_encoder, self.encoder),
            (self.fixed_encoder_target, self.encoder),
        ):
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
        judge_params = [*self.critic.parameters(), *self.psi.parameters()]
        self.critic_optimizer = torch.optim.Adam(judge_params, lr=args.critic_lr, **adam)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr, **adam)
        # (destination, source) pairs of every hard update, in TD7's order (target before fixed).
        self._hard_updates = [
            (list(self.actor_target.parameters()), list(self.actor.parameters())),
            (list(self.critic_target.parameters()), list(self.critic.parameters())),
            (list(self.psi_target.parameters()), list(self.psi.parameters())),
            (list(self.fixed_encoder_target.parameters()), list(self.fixed_encoder.parameters())),
            (list(self.fixed_encoder.parameters()), list(self.encoder.parameters())),
        ]

        # Entropy is confined to the actor loss; autotuning holds the mean sigma near target_sigma.
        self.target_entropy = action_dim * float(np.log(args.target_sigma))
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
        optimizers = [self.encoder_optimizer, self.critic_optimizer, self.actor_optimizer, self.alpha_optimizer]

        # Replay rows keep the clipped action plus the belief (mu, sigma, eps) for the center tangent.
        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (("state", s), ("action", a), ("mu", a), ("sigma", a), ("eps", a), ("next_state", s), ("reward", 1), ("not_done", 1)):
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

        # TD-InfoNCE label pieces over the (B, B + 1) contrast set: e_i (the row's own next state) and e_⊥.
        n = args.batch_size
        self.own_future = torch.eye(n, n + 1, device=device)
        self.absorbed = torch.zeros(n, n + 1, device=device)
        self.absorbed[:, -1] = 1
        # Readout futures: the batch (N) or a sampled bank (M); ⊥ gets + log(M/N), a size-invariant mass.
        self.bank_size = args.readout_bank or n
        self.absorbing_offset = math.log(self.bank_size / n)

        f32 = dict(device=device, dtype=torch.float32)
        self.max_priority = torch.ones((), **f32)
        self.running_max = torch.full((), -1e8, **f32)
        self.running_min = torch.full((), 1e8, **f32)
        self.max_target = torch.zeros((), **f32)
        self.min_target = torch.zeros((), **f32)
        self.metric_names = [
            "encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std",
            "actor_loss", "sigma_mean", "sigma_min", "sigma_max", "smoothing_gap", "belief_gap",
            "contrastive_loss", "contrastive_acc", "logit_scale", "readout_q", "absorbing_mass",
        ]
        self.metrics = torch.zeros(len(self.metric_names), **f32)
        self.fresh_cut = math.ceil(math.log(args.fresh_tail_weight) / math.log(args.gamma))

        maybe_compile = (
            (lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}))
            if args.compile
            else (lambda fn: fn)
        )
        self._encoder_loss = maybe_compile(self._encoder_loss_fn)
        if self.contrastive:
            self._critic_loss = maybe_compile(self._contrastive_loss_fn)
            self._actor_loss = maybe_compile(self._readout_actor_loss_fn)
        else:
            self._critic_loss = maybe_compile(self._hybrid_critic_loss_fn)
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
    def _draw(self, csum, n):
        """n replay indices proportional to priority (uniform in the contrastive arm, where all are 1)."""
        ind = torch.searchsorted(csum, torch.rand(n, device=self.device) * csum.gather(0, self.last))
        return torch.minimum(ind, self.last)  # the scan is only monotone up to rounding

    def _target_intention(self, next_state):
        """zs'(s') and the target actor's intention under TD7's sampled policy smoothing."""
        args = self.args
        next_zs = self.fixed_encoder_target.zs(next_state)
        next_mu, _ = self.actor_target(next_state, next_zs)
        noise = (torch.randn_like(next_mu) * args.target_smoothing_sigma).clamp(
            -args.target_noise_clip, args.target_noise_clip
        )
        return next_zs, self.fixed_encoder_target.zsa(next_zs, (next_mu + noise).clamp(-1, 1))

    def _td_infonce(self, logits, target_logits, not_done):
        """TD-InfoNCE CE of each member's (2, B, B + 1) logits; returns loss, mean CE, top-1 next-state acc.

        Labels (1-gamma) e_i + gamma w_i, w_i the members' mean target softmax at (s'_i, g'_i); done rows
        put the gamma mass on ⊥ (zero-reward absorbing state), not on s'_i.
        """
        gamma = self.args.gamma
        with torch.no_grad():
            w = target_logits.softmax(-1).mean(0)
            labels = (1 - gamma) * self.own_future + gamma * torch.lerp(self.absorbed, w, not_done)
        ce = -(labels * logits.log_softmax(-1)).sum(-1).mean()
        loss = ce
        if self.args.logsumexp_coef > 0:
            loss = loss + self.args.logsumexp_coef * logits.logsumexp(-1).square().mean()
        with torch.no_grad():
            own = torch.arange(logits.shape[1], device=logits.device)
            acc = (logits.argmax(-1) == own).float().mean()
        return loss, ce.detach(), acc

    def _encoder_loss_fn(self, batch):
        f = self._fields
        with torch.no_grad():
            next_zs = self.encoder.zs(batch[:, f["next_state"]])
        pred_zs = self.encoder.zsa(self.encoder.zs(batch[:, f["state"]]), batch[:, f["action"]])
        return F.mse_loss(pred_zs, next_zs)

    def _contrastive_loss_fn(self, batch):
        """Contrastive arm: phi = self.critic (IntentionPhi) on the executed intention, futures s'_j."""
        f, args = self._fields, self.args
        state, next_state, action = batch[:, f["state"]], batch[:, f["next_state"]], batch[:, f["action"]]
        with torch.no_grad():
            next_zs, next_g = self._target_intention(next_state)
            # psi_target reads zs'(s'_j) from the target encoder, the same zs' the target phi sees.
            target_logits = self.psi_target.logits(
                self.critic_target(next_state, next_g, next_zs), self.psi_target(next_zs, next_state)
            )
            fixed_zs = self.fixed_encoder.zs(state)
            g = self.fixed_encoder.zsa(fixed_zs, action)
            future_zs = self.fixed_encoder.zs(next_state)
        logits = self.psi.logits(self.critic(state, g, fixed_zs), self.psi(future_zs, next_state))
        loss, ce, acc = self._td_infonce(logits, target_logits, batch[:, f["not_done"]])
        with torch.no_grad():
            q = readout(logits, batch[:, f["reward"]].squeeze(1), args.gamma).mean()
            diagnostics = torch.stack([ce, acc, self.psi.log_scale.exp(), q])
        return loss, diagnostics, fixed_zs

    def _hybrid_critic_loss_fn(self, batch, min_target, max_target):
        """judge_v1's critic loss, plus TD-InfoNCE on the q2 features of the point channel g(a)."""
        f, args = self._fields, self.args
        state, next_state = batch[:, f["state"]], batch[:, f["next_state"]]
        action, mu, sigma, eps = batch[:, f["action"]], batch[:, f["mu"]], batch[:, f["sigma"]], batch[:, f["eps"]]
        with torch.no_grad():
            next_zs, next_g = self._target_intention(next_state)
            q_target, target_features = self.critic_target(next_state, next_g, next_zs, features=True)
            target_logits = self.psi_target.logits(target_features, self.psi_target(next_zs, next_state))
            q_target = q_target.min(1, keepdim=True)[0].clamp(min_target, max_target)
            q_target = batch[:, f["reward"]] + batch[:, f["not_done"]] * args.gamma * q_target

            # Two intentions share one 2B pass: the belief center g(mu) with the order-1 Hermite
            # tangent J_zsa (sigma eps), and the executed g(a) (v27's point and pre-action channels,
            # identical once the judge cannot see u or sigma).
            fixed_zs = self.fixed_encoder.zs(state)
            t_mu = sigma * eps if args.hermite_order >= 1 else torch.zeros_like(mu)
            zs2 = torch.cat([fixed_zs, fixed_zs])
            g2, t_g2 = self.fixed_encoder.zsa_jvp(
                zs2, torch.cat([mu, action]), torch.cat([t_mu, torch.zeros_like(action)])
            )
            future_zs = self.fixed_encoder.zs(next_state)
        q2, tangent2, features2 = self.critic(torch.cat([state, state]), g2, zs2, t_g2, features=True)
        batch_size = state.shape[0]
        q_dist, tangent, q_action = q2[:batch_size], tangent2[:batch_size], q2[batch_size:]
        td_latent = (q_dist + tangent - q_target).abs()
        td_action = (q_action - q_target).abs()
        # v27's weights: center 1/3, point 1/3 + the merged pre-action channel's 1/3.
        loss = (lap_huber(td_latent, args.min_priority) + 2 * lap_huber(td_action, args.min_priority)) / 3
        logits = self.psi.logits(features2[:, batch_size:], self.psi(future_zs, next_state))
        contrastive_loss, ce, acc = self._td_infonce(logits, target_logits, batch[:, f["not_done"]])
        td_loss = loss.detach()
        loss = loss + args.contrastive_weight * contrastive_loss
        with torch.no_grad():
            # Fraction of the realized-noise variation in y that the Hermite term explains.
            r2 = 1 - td_latent.square().mean() / (q_target - q_dist).square().mean().clamp(min=1e-8)
            smoothing_gap = (q_action - q_dist).mean()
            q_readout = readout(logits, batch[:, f["reward"]].squeeze(1), args.gamma).mean()
            diagnostics = torch.stack([
                q_dist.mean(), r2, tangent.std(), q_target.max(), q_target.min(), smoothing_gap,
                ce, acc, self.psi.log_scale.exp(), q_readout, td_loss,
            ])
        td = torch.maximum(td_latent, td_action)
        return loss, td.detach(), diagnostics, fixed_zs

    def _actor_loss_fn(self, state, fixed_zs, alpha):
        mu, log_std = self.actor(state, fixed_zs)
        if not self.args.belief_sigma:
            q = self.critic(state, self.fixed_encoder.zsa(fixed_zs, mu), fixed_zs)
            return -q.mean() - alpha * log_std.sum(-1).mean(), log_std, torch.zeros((), device=mu.device)
        # One 3B pass: mu's intention, then the antithetic pair around the detached mean. The pair's
        # first-order terms cancel, so sigma's gradient is the second-order Hermite / Stein term.
        delta = log_std.exp() * torch.randn_like(mu)
        mu_d = mu.detach()
        actions = torch.cat([mu, (mu_d + delta).clamp(-1, 1), (mu_d - delta).clamp(-1, 1)])
        zs3 = torch.cat([fixed_zs, fixed_zs, fixed_zs])
        q3 = self.critic(torch.cat([state, state, state]), self.fixed_encoder.zsa(zs3, actions), zs3).mean(1)
        batch_size = state.shape[0]
        q_mu, q_belief = q3[:batch_size], 0.5 * (q3[batch_size : 2 * batch_size] + q3[2 * batch_size :])
        loss = -q_mu.mean() - q_belief.mean() - alpha * log_std.sum(-1).mean()
        return loss, log_std, (q_belief - q_mu).mean().detach()

    def _readout(self, state, zs, g, psi_f, rewards):
        """Twin-mean readout Q(s, zs, g) over the futures psi_f (bank-size-corrected ⊥) and ⊥'s mass, (B,) each."""
        logits = self.psi.logits(self.critic(state, g, zs), psi_f, self.absorbing_offset)
        return readout(logits, rewards, self.args.gamma).mean(1), logits.softmax(-1)[..., -1].mean(0)

    def _readout_actor_loss_fn(self, state, fixed_zs, future_state, future_reward, alpha):
        """The judge actor's loss with Q replaced by the readout over the given futures."""
        with torch.no_grad():
            psi_f = self.psi(self.fixed_encoder.zs(future_state), future_state)
        mu, log_std = self.actor(state, fixed_zs)
        if not self.args.belief_sigma:
            q, absorbing = self._readout(state, fixed_zs, self.fixed_encoder.zsa(fixed_zs, mu), psi_f, future_reward)
            diagnostics = torch.stack([torch.zeros((), device=mu.device), absorbing.mean().detach()])
            return -q.mean() - alpha * log_std.sum(-1).mean(), log_std, diagnostics
        delta = log_std.exp() * torch.randn_like(mu)
        mu_d = mu.detach()
        actions = torch.cat([mu, (mu_d + delta).clamp(-1, 1), (mu_d - delta).clamp(-1, 1)])
        zs3 = torch.cat([fixed_zs, fixed_zs, fixed_zs])
        q3, absorbing = self._readout(
            torch.cat([state, state, state]), zs3, self.fixed_encoder.zsa(zs3, actions), psi_f, future_reward
        )
        batch_size = state.shape[0]
        q_mu, q_belief = q3[:batch_size], 0.5 * (q3[batch_size : 2 * batch_size] + q3[2 * batch_size :])
        loss = -q_mu.mean() - q_belief.mean() - alpha * log_std.sum(-1).mean()
        diagnostics = torch.stack([(q_belief - q_mu).mean(), absorbing[:batch_size].mean()]).detach()
        return loss, log_std, diagnostics

    def _entropy_fn(self, state, fixed_zs):
        with torch.no_grad():
            return self.actor(state, fixed_zs)[1].sum(-1)

    # ---- train step ---------------------------------------------------------------------------
    def _make_step(self, update_actor):
        args, f = self.args, self._fields
        max_log_scale = math.log(100.0)

        def step():
            csum = torch.cumsum(self.priority, 0)
            ind = self._draw(csum, args.batch_size)
            batch = self.storage.index_select(0, ind)

            self.encoder_optimizer.zero_grad(set_to_none=False)
            encoder_loss = self._encoder_loss(batch)
            encoder_loss.backward()
            self.encoder_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=False)
            if self.contrastive:
                critic_loss, diagnostics, fixed_zs = self._critic_loss(batch)
            else:
                critic_loss, td, diagnostics, fixed_zs = self._critic_loss(batch, self.min_target, self.max_target)
            critic_loss.backward()
            self.critic_optimizer.step()

            with torch.no_grad():
                self.psi.log_scale.clamp_(0.0, max_log_scale)  # clamp the parameter, not its use (no dead grad)
                self.metrics[0].copy_(encoder_loss.detach())
                self.metrics[1].copy_(critic_loss.detach())
                if self.contrastive:
                    # Uniform replay: priorities stay at their insertion value 1 (see header).
                    self.metrics[2].copy_(diagnostics[3])
                    self.metrics[11:15].copy_(diagnostics)
                else:
                    torch.maximum(self.running_max, diagnostics[3], out=self.running_max)
                    torch.minimum(self.running_min, diagnostics[4], out=self.running_min)
                    priority = td.max(1)[0].clamp(min=args.min_priority).pow(args.lap_alpha)
                    self.priority[ind] = priority
                    torch.maximum(self.max_priority, priority.max(), out=self.max_priority)
                    self.metrics[2:5].copy_(diagnostics[:3])
                    self.metrics[9].copy_(diagnostics[5])
                    self.metrics[1].copy_(diagnostics[10])  # the TD loss alone, comparable to judge_v1
                    self.metrics[11:15].copy_(diagnostics[6:10])

            if update_actor:
                state = batch[:, f["state"]]
                alpha = self.log_alpha.detach().exp()
                self.actor_optimizer.zero_grad(set_to_none=False)
                if self.contrastive:
                    futures = self.storage.index_select(0, self._draw(csum, args.readout_bank)) if args.readout_bank else batch
                    actor_loss, log_std, actor_diagnostics = self._actor_loss(
                        state, fixed_zs, futures[:, f["next_state"]], futures[:, f["reward"]].squeeze(1), alpha
                    )
                    belief_gap = actor_diagnostics[0]
                    self.metrics[15].copy_(actor_diagnostics[1])
                else:
                    actor_loss, log_std, belief_gap = self._actor_loss(state, fixed_zs, alpha)
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
                    self.metrics[10].copy_(belief_gap)

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
            names = ["encoder_loss", "critic_loss", "q_values", "actor_loss", "belief_gap", "contrastive_loss", "readout_q"]
            if not self.contrastive:
                names += ["hermite_r2", "tangent_std", "smoothing_gap"]
                self.writer.add_scalar("charts/max_target", self.max_target.item(), self.training_steps)
                self.writer.add_scalar("charts/min_target", self.min_target.item(), self.training_steps)
            for name in names:
                self.writer.add_scalar(f"losses/{name}", values[name], self.training_steps)
            for name in ("contrastive_acc", "logit_scale") + (("absorbing_mass",) if self.contrastive else ()):
                self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
            if self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.training_steps)

    # ---- fresh-data optimism ------------------------------------------------------------------
    def _log_fresh(self, rows, rewards, terminated, global_step):
        """Judged value of the executed intention minus realized discounted return, on an episode no
        update has seen, plus SALE's own error there (intent_gap: predicted vs reached next embedding).

        The contrastive arm judges with the readout over futures sampled from replay now (the batch a
        training step would draw, or a readout_bank-sized one). Truncated episodes keep only steps
        whose unobserved tail weighs less than fresh_tail_weight.
        """
        n = len(rewards)
        valid = n if terminated else n - self.fresh_cut
        returns, ret = np.empty(n), 0.0
        for t in range(n - 1, -1, -1):
            ret = rewards[t] + self.args.gamma * ret
            returns[t] = ret
        f = self._fields
        with torch.no_grad():
            zs = self.fixed_encoder.zs(rows[:, f["state"]])
            intent = self.fixed_encoder.zsa(zs, rows[:, f["action"]])
            reached = self.fixed_encoder.zs(rows[:, f["next_state"]])
            gap = (intent - reached).norm(dim=1)
            stats = [gap.mean(), (gap / reached.norm(dim=1).clamp(min=1e-8)).mean()]
            if valid > 0:
                realized = torch.as_tensor(returns[:valid], dtype=torch.float32, device=self.device)
                state = rows[:valid, f["state"]]
                if self.contrastive:
                    bank = self.storage.index_select(0, self._draw(torch.cumsum(self.priority, 0), self.bank_size))
                    future_state = bank[:, f["next_state"]]
                    psi_f = self.psi(self.fixed_encoder.zs(future_state), future_state)
                    judged = self._readout(state, zs[:valid], intent[:valid], psi_f, bank[:, f["reward"]].squeeze(1))[0]
                else:
                    judged = self.critic(state, intent[:valid], zs[:valid]).mean(1)
                error = judged - realized
                stats += [error.mean(), error.mean() / realized.abs().mean().clamp(min=1e-8), error.abs().mean()]
            stats = torch.stack(stats).tolist()
        for name, value in zip(("intent_gap", "intent_rel_gap", "optimism", "rel_optimism", "abs_err"), stats):
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
            eps = rng.standard_normal(action_dim, dtype=np.float32)
            pre_action = mu + sigma * eps
            action = np.clip(pre_action, -1, 1)
        with timer.span("env", use_cuda=False):
            next_obs, reward, terminated, truncated, _ = env.step(action * max_action)
        next_obs = next_obs.astype(np.float32)
        # The original bootstraps at the timeout step even when the env truly terminated there.
        agent.stage(
            state=obs, action=action, mu=mu, sigma=sigma, eps=eps, next_state=next_obs,
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
                "psi": agent.psi.state_dict(),
                "checkpoint_policy": agent.checkpoint_policy._flat,
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    env.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()

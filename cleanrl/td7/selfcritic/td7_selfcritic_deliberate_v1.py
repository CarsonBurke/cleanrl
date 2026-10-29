# TD7 SELF-CRITIC DELIBERATE v1 — latentq v5 that knows where its own judgment fails and thinks before acting.
#
# Two capabilities on top of v5's belief critic Q~(s, mu, sigma, u):
# 1. Self-model. b(s, mu, sigma, u) predicts the critic's own error, Q~ minus the realized discounted
#    return, regressed on the last sm_window transitions (near the current policy, steps with <1% tail
#    weight). It is supervised by reality, not by bootstrapping. The actor maximizes Q~ - relu(b): it is
#    repelled exactly where its critic has been optimistic, the error sign that sank every optimistic
#    variant (consequenceq) while v5's consistently pessimistic critic won.
# 2. Deliberation. A supervised one-step model (s', r, d) imagines plan_horizon steps for plan_candidates
#    draws from the actor's own belief (its mean always included), continuing under the actor and judged
#    by the actor's own self-critical objective at the horizon, twin-mean Q~ - relu(b). The best candidate
#    becomes the belief mean and the executed action is still a Gaussian draw around it, so v5's Hermite
#    labels stay valid. Evaluation deliberates with the checkpointed actor, critic, model and self-model.
# Hypothesis: calibrated self-distrust plus self-checked lookahead raises return over v5. fresh/sm_* test
# whether the self-model actually predicts the critic's error on episodes it has never seen.
import copy
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
    """returns (fresh probe and self-model targets) use steps whose truncated tail weight gamma^k is below this"""
    sm_window: int = 20000
    """the self-model regresses on the most recent this-many transitions"""
    sm_lr: float = 3e-4
    """the learning rate of the self-model optimizer"""
    model_hidden_dim: int = 512
    """hidden layer width of the one-step model"""
    model_lr: float = 3e-4
    """the learning rate of the one-step model optimizer"""
    plan_candidates: int = 64
    """belief draws deliberated per decision (the belief mean is always one of them)"""
    plan_horizon: int = 3
    """imagined steps per candidate before the critic judges"""


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


class SelfModel(nn.Module):
    """b(s, mu, sigma, u, age): the critic's own error (Q~ minus realized return) at a belief query.

    age is how many training steps ago (in sm_window units) the return was earned. The current critic
    values the current policy, so on older rows the error also contains the policy's improvement since;
    with age as an input that drift has its own coordinate, and the actor queries b at age 0.
    """

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.l0 = nn.Linear(state_dim + 3 * action_dim + 1, hdim)
        self.l1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, 1)

    def forward(self, state, latent, pre_action, age, zsa, zs):
        h = avg_l1_norm(self.l0(torch.cat([state, latent, pre_action, age], 1)))
        h = F.elu(self.l1(torch.cat([h, zsa, zs], 1)))
        h = F.elu(self.l2(h))
        return self.l3(h)


class ConsequenceModel(nn.Module):
    """Deterministic one-step model: (normalized s, a) -> (normalized delta s | normalized r | done logit)."""

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


class Mind(nn.Module):
    """Everything deliberation reads, as one module so TD7's checkpoint snapshots it whole.

    Holds (shares, for the live mind) the actor, the fixed encoder that both actor and critic read, the
    critic, the one-step model with its normalizer buffers and the self-model. The checkpoint mind is a deep copy
    refreshed in one _foreach_copy_, so evaluation deliberates with exactly the system that was assessed.
    """

    def __init__(self, actor, encoder, critic, model, self_model, state_dim, args):
        super().__init__()
        self.actor, self.encoder, self.critic, self.model = actor, encoder, critic, model
        self.self_model = self_model
        self.state_dim = state_dim
        self.gamma, self.horizon, self.smoothing = args.gamma, args.plan_horizon, args.target_smoothing_sigma
        # Model normalizers: replay-wide moments of s, delta s and r, recomputed in place at every flush.
        for name, width in (("state", state_dim), ("delta", state_dim), ("reward", 1)):
            self.register_buffer(f"{name}_mean", torch.zeros(width))
            self.register_buffer(f"{name}_std", torch.ones(width))

    def model_out(self, state, action):
        out = self.model((state - self.state_mean) / self.state_std, action)
        return out[:, : self.state_dim], out[:, self.state_dim : self.state_dim + 1], out[:, self.state_dim + 1 :]

    def consequence(self, state, action):
        """Predicted (next state, reward, continuation probability) in raw units."""
        delta, reward, done_logit = self.model_out(state, action)
        next_state = state + self.delta_mean + self.delta_std * delta
        return next_state, self.reward_mean + self.reward_std * reward, torch.sigmoid(-done_logit)

    def belief_value(self, state):
        """Twin values (batch, 2) of the actor's belief at state, queried as the TD target is (smoothing sigma)."""
        zs = self.encoder.zs(state)
        mu, _ = self.actor(state, zs)
        latent = torch.cat([mu, torch.full_like(mu, self.smoothing)], 1)
        return self.critic(state, latent, mu, self.encoder.zsa(zs, mu), zs)

    def judged_value(self, state):
        """The actor's own objective at state: twin-mean Q~ of its belief minus the self-model's optimism.

        b was fitted to twin-mean error against what this deliberating agent actually earns, so this is
        the terminal value corrected toward reality (never upward), queried exactly as the actor ascends.
        """
        zs = self.encoder.zs(state)
        mu, log_std = self.actor(state, zs)
        latent = torch.cat([mu, log_std.exp()], 1)
        zsa = self.encoder.zsa(zs, mu)
        q = self.critic(state, latent, mu, zsa, zs).mean(1, keepdim=True)
        return q - F.relu(self.self_model(state, latent, mu, torch.zeros_like(q), zsa, zs))

    def plan(self, state, eps):
        """Deliberate over belief draws: imagine `horizon` steps, judge, keep the best as the new mean."""
        batch, candidates, _ = eps.shape
        mu, log_std = self.actor(state, self.encoder.zs(state))
        sigma = log_std.exp()
        first = (mu[:, None] + sigma[:, None] * eps).clamp(-1, 1)
        x = state.repeat_interleave(candidates, 0)
        action = first.reshape(batch * candidates, -1)
        value, discount = torch.zeros_like(x[:, :1]), torch.ones_like(x[:, :1])
        for k in range(self.horizon):
            if k > 0:
                action = self.actor(x, self.encoder.zs(x))[0]
            x, reward, cont = self.consequence(x, action)
            value = value + discount * reward
            discount = discount * self.gamma * cont
        value = (value + discount * self.judged_value(x)).view(batch, candidates)
        best = value.argmax(1)
        mu_plan = first[torch.arange(batch, device=state.device), best]
        gain = value.max(1)[0] - value[:, 0]  # deliberated value over the belief mean's (max of draws: biased up)
        return torch.cat([mu_plan, sigma, gain[:, None], (best == 0).float()[:, None]], 1)


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
        self.self_model = SelfModel(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.model = ConsequenceModel(state_dim, action_dim, args.model_hidden_dim).to(device)
        for target, source in (
            (self.actor_target, self.actor),
            (self.critic_target, self.critic),
            (self.fixed_encoder, self.encoder),
            (self.fixed_encoder_target, self.encoder),
        ):
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        self.mind = Mind(self.actor, self.fixed_encoder, self.critic, self.model, self.self_model, state_dim, args).to(device)
        # Device snapshot of the assessed system (TD7 checkpointing); evaluation deliberates with it.
        self.checkpoint_mind = copy.deepcopy(self.mind).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, **adam)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr, **adam)
        self.self_model_optimizer = torch.optim.Adam(self.self_model.parameters(), lr=args.sm_lr, **adam)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.model_lr, **adam)
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
        optimizers = [
            self.encoder_optimizer, self.critic_optimizer, self.model_optimizer, self.self_model_optimizer,
            self.actor_optimizer, self.alpha_optimizer,
        ]

        # Replay rows retain both the clipped environment action and its pre-action latent.
        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (("state", s), ("action", a), ("mu", a), ("sigma", a), ("eps", a), ("pre_action", a), ("next_state", s), ("reward", 1), ("not_done", 1), ("ret", 1), ("ret_valid", 1), ("collected", 1)):
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
        # The self-model's window, the last recent_count rows before ptr, as device scalars the captured
        # graphs read; refreshed at every flush.
        self.ptr_t = torch.zeros((), dtype=torch.long, device=device)
        self.recent_count = torch.ones((), **f32)
        self.train_step_t = torch.zeros((), **f32)  # training steps taken, advanced inside the graphs
        self.max_priority = torch.ones((), **f32)
        self.running_max = torch.full((), -1e8, **f32)
        self.running_min = torch.full((), 1e8, **f32)
        self.max_target = torch.zeros((), **f32)
        self.min_target = torch.zeros((), **f32)
        self.metric_names = [
            "encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std",
            "actor_loss", "sigma_mean", "sigma_min", "sigma_max", "smoothing_gap", "preaction_gap",
            "model_dyn_mse", "model_reward_mse", "model_done_bce", "sm_loss", "sm_bias", "sm_error", "sm_penalty",
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
        self._model_loss = maybe_compile(self._model_loss_fn)
        self._self_model_loss = maybe_compile(self._self_model_loss_fn)
        # training_steps % policy_freq == 0 selects the actor-step graph (TD7's delayed update).
        self.graphs = capture_steps(
            [self._make_step(update_actor=False), self._make_step(update_actor=True)],
            optimizers=optimizers,
            state_tensors=[
                self.priority, self.max_priority, self.running_max, self.running_min, self.metrics, self.train_step_t,
            ],
        )

        # Deliberation graphs: one acting decision (current actor) and one batched evaluation step
        # (checkpoint mind). Each reads plan_state and writes [mu_plan | sigma | gain | kept_mean].
        self.act_plan = self._make_planner(1, self.mind, maybe_compile)
        self.eval_plan = self._make_planner(args.eval_eps, self.checkpoint_mind, maybe_compile)
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
        fresh-data optimism (and self-model calibration) before any training step sees it. Each row
        stores its discounted return, bootstrapped past a truncation by the critic's belief value at
        the final state; rows whose bootstrapped tail weighs more than fresh_tail_weight are masked out
        of the self-model's targets.
        """
        n = self.staged
        f, gamma = self._fields, self.args.gamma
        rewards = self.staging_np[:n, f["reward"]].ravel()
        tail = 0.0
        if not terminated:
            with torch.no_grad():
                final = self.staging[n - 1 : n, f["next_state"]].to(self.device)
                tail = self.mind.belief_value(final).mean().item()
        returns, g = np.empty(n, dtype=np.float32), tail
        for t in range(n - 1, -1, -1):
            g = rewards[t] + gamma * g
            returns[t] = g
        valid = n if terminated else max(n - self.fresh_cut, 0)
        self.staging_np[:n, f["ret"]] = returns[:, None]
        self.staging_np[:n, f["ret_valid"]] = (np.arange(n) < valid)[:, None]
        self.staging_np[:n, f["collected"]] = self.training_steps
        rows = self.staging[:n].to(self.device)
        if self.training_steps > 0 and valid > 0:
            self._log_fresh(rows[:valid], global_step)
        index = (self.ptr + torch.arange(n, device=self.device)) % self.args.buffer_size
        self.storage.index_copy_(0, index, rows)
        self.priority.index_put_((index,), self.max_priority.expand(n))
        self.ptr = (self.ptr + n) % self.args.buffer_size
        self.size = min(self.size + n, self.args.buffer_size)
        self.last.fill_(self.size - 1)
        self.ptr_t.fill_(self.ptr)
        self.recent_count.fill_(float(min(self.args.sm_window, self.size)))
        self.staged = 0
        self._refresh_normalizers()

    def _refresh_normalizers(self):
        f = self._fields
        replay = self.storage[: self.size]
        state = replay[:, f["state"]]
        with torch.no_grad():
            for (mean, std), x in (
                ((self.mind.state_mean, self.mind.state_std), state),
                ((self.mind.delta_mean, self.mind.delta_std), replay[:, f["next_state"]] - state),
                ((self.mind.reward_mean, self.mind.reward_std), replay[:, f["reward"]]),
            ):
                x_std, x_mean = torch.std_mean(x, 0, correction=0)
                mean.copy_(x_mean)
                std.copy_(x_std.clamp(min=1e-3))  # constant coordinates (e.g. a zero reward stretch)

    # ---- deliberation ------------------------------------------------------------------------
    def _make_planner(self, batch, mind, maybe_compile):
        args = self.args
        state = torch.zeros(batch, self.state_dim, device=self.device)
        out = torch.zeros(batch, 2 * self.action_dim + 2, device=self.device)
        mean_first = torch.ones(1, args.plan_candidates, 1, device=self.device)
        mean_first[:, 0] = 0  # candidate 0 is the belief mean itself
        plan = maybe_compile(mind.plan)

        def step():
            with torch.no_grad():
                eps = torch.randn(batch, args.plan_candidates, self.action_dim, device=self.device) * mean_first
                out.copy_(plan(state, eps))

        graph = capture_steps([step], optimizers=[], state_tensors=[out])[0]
        return state, out, graph

    def deliberate(self, planner, states):
        """Host states (batch, state_dim) -> host [mu_plan | sigma | gain | kept_mean] rows."""
        state, out, graph = planner
        state.copy_(torch.from_numpy(states))
        graph.replay()
        return out.cpu().numpy()

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

    def _model_loss_fn(self, batch):
        f = self._fields
        state, next_state = batch[:, f["state"]], batch[:, f["next_state"]]
        mind = self.mind
        delta, reward, done_logit = mind.model_out(state, batch[:, f["action"]])
        dyn = (delta - (next_state - state - mind.delta_mean) / mind.delta_std).square().mean()
        rew = (reward - (batch[:, f["reward"]] - mind.reward_mean) / mind.reward_std).square().mean()
        done = F.binary_cross_entropy_with_logits(done_logit, 1 - batch[:, f["not_done"]])
        return dyn + rew + done, torch.stack([dyn, rew, done]).detach()

    def _self_model_loss_fn(self, recent):
        """Regress b onto the current critic's error against realized returns at recent behavior queries."""
        f = self._fields
        state, action, pre_action = recent[:, f["state"]], recent[:, f["action"]], recent[:, f["pre_action"]]
        latent = torch.cat([recent[:, f["mu"]], recent[:, f["sigma"]]], 1)
        valid = recent[:, f["ret_valid"]]
        age = (self.train_step_t - recent[:, f["collected"]]) / self.args.sm_window
        with torch.no_grad():
            zs = self.fixed_encoder.zs(state)
            zsa = self.fixed_encoder.zsa(zs, action)
            error = self.critic(state, latent, pre_action, zsa, zs).mean(1, keepdim=True) - recent[:, f["ret"]]
        bias = self.self_model(state, latent, pre_action, age, zsa, zs)
        count = valid.sum().clamp(min=1)
        loss = (valid * (bias - error).square()).sum() / count
        with torch.no_grad():
            diagnostics = torch.stack([(valid * bias).sum() / count, (valid * error).sum() / count])
        return loss, diagnostics

    def _actor_loss_fn(self, state, fixed_zs, alpha):
        """Ascend the belief value minus the self-model's predicted optimism at the same query."""
        mu, log_std = self.actor(state, fixed_zs)
        latent = torch.cat([mu, log_std.exp()], 1)
        zsa = self.fixed_encoder.zsa(fixed_zs, mu)
        q = self.critic(state, latent, mu, zsa, fixed_zs).mean(1)
        penalty = F.relu(self.self_model(state, latent, mu, torch.zeros_like(mu[:, :1]), zsa, fixed_zs)).squeeze(1)
        return -(q - penalty).mean() - alpha * log_std.sum(-1).mean(), log_std, penalty.mean().detach()

    def _entropy_fn(self, state, fixed_zs):
        with torch.no_grad():
            return self.actor(state, fixed_zs)[1].sum(-1)

    # ---- train step ---------------------------------------------------------------------------
    def _make_step(self, update_actor):
        args, device = self.args, self.device

        def step():
            self.train_step_t.add_(1)
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

            self.model_optimizer.zero_grad(set_to_none=False)
            model_loss, model_terms = self._model_loss(batch)
            model_loss.backward()
            self.model_optimizer.step()

            # The self-model's window: uniform over the last recent_count rows before ptr.
            offset = (torch.rand(args.batch_size, device=device) * self.recent_count).long()
            recent = self.storage.index_select(0, (self.ptr_t - 1 - offset) % args.buffer_size)
            self.self_model_optimizer.zero_grad(set_to_none=False)
            sm_loss, sm_diagnostics = self._self_model_loss(recent)
            sm_loss.backward()
            self.self_model_optimizer.step()

            with torch.no_grad():
                self.metrics[11:14].copy_(model_terms)
                self.metrics[14].copy_(sm_loss.detach())
                self.metrics[15:17].copy_(sm_diagnostics)
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
                actor_loss, log_std, sm_penalty = self._actor_loss(state, fixed_zs, self.log_alpha.detach().exp())
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
                    self.metrics[17].copy_(sm_penalty)

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
                "encoder_loss", "critic_loss", "q_values", "actor_loss", "hermite_r2", "tangent_std", "smoothing_gap",
                "preaction_gap", "model_dyn_mse", "model_reward_mse", "model_done_bce", "sm_loss", "sm_bias",
                "sm_error", "sm_penalty",
            ):
                self.writer.add_scalar(f"losses/{name}", values[name], self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target.item(), self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target.item(), self.training_steps)
            if self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.training_steps)

    # ---- fresh-data optimism ------------------------------------------------------------------
    def _log_fresh(self, rows, global_step):
        """Critic prediction minus realized discounted return on an episode no update has seen.

        ``rows`` are the episode's steps with <fresh_tail_weight bootstrapped tail. Scored at the
        TD-labeled behavior query (twin mean). The sm_* terms test the self-model where it has never
        trained: its predicted error against the actual error, and the corrected critic's error.
        """
        f = self._fields
        state, action = rows[:, f["state"]], rows[:, f["action"]]
        latent = torch.cat([rows[:, f["mu"]], rows[:, f["sigma"]]], 1)
        realized = rows[:, f["ret"]].squeeze(1)
        with torch.no_grad():
            zs = self.fixed_encoder.zs(state)
            zsa = self.fixed_encoder.zsa(zs, action)
            q = self.critic(state, latent, rows[:, f["pre_action"]], zsa, zs).mean(1)
            # Fresh rows were earned by the current policy: age 0, the query the actor uses.
            bias = self.self_model(state, latent, rows[:, f["pre_action"]], torch.zeros_like(state[:, :1]), zsa, zs).squeeze(1)
            error = q - realized
            corrected = error - bias
            centered_b, centered_e = bias - bias.mean(), error - error.mean()
            corr = (centered_b * centered_e).mean() / (centered_b.std(correction=0) * centered_e.std(correction=0)).clamp(min=1e-8)
            stats = torch.stack([
                error.mean(), error.mean() / realized.abs().mean().clamp(min=1e-8), error.abs().mean(),
                bias.mean(), corr, corrected.mean(), corrected.abs().mean(),
            ]).tolist()
        names = ("optimism", "rel_optimism", "abs_err", "sm_bias", "sm_corr", "sm_corrected_optimism", "sm_corrected_abs_err")
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
            with torch.no_grad():  # nothing deliberation reads has changed since the last burst
                torch._foreach_copy_(
                    [*self.checkpoint_mind.parameters(), *self.checkpoint_mind.buffers()],
                    [*self.mind.parameters(), *self.mind.buffers()],
                )
            self.train_and_reset()

    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.args.steps_before_checkpointing:
                self.best_min_return *= self.args.reset_weight
                self.max_eps_before_update = self.args.max_eps_when_checkpointing
            self.train()
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8


def evaluate(agent: TD7Agent, eval_envs, max_action):
    """Checkpoint-mind returns with deliberation (the deliberated mean, no execution noise), one episode per env."""
    states = np.stack([env.reset()[0] for env in eval_envs]).astype(np.float32)
    returns = np.zeros(len(eval_envs))
    active = np.ones(len(eval_envs), dtype=bool)
    while active.any():
        actions = agent.deliberate(agent.eval_plan, states)[:, : agent.action_dim]
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
    plan_gain, plan_kept = 0.0, 0.0
    next_timing_log = 10000
    # total_timesteps + 1 so the final evaluation at exactly total_timesteps fires (as in the original)
    for global_step in range(args.total_timesteps + 1):
        if global_step % args.eval_freq == 0:
            with timer.span("eval", use_cuda=False):
                eval_return = evaluate(agent, eval_envs, max_action)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(f"global_step={global_step}, eval_return={eval_return:.3f}")

        with timer.span("rollout", use_cuda=False):
            if allow_train:
                plan = agent.deliberate(agent.act_plan, obs[None])[0]
                mu, sigma = plan[:action_dim], plan[action_dim : 2 * action_dim]
                plan_gain += plan[-2]
                plan_kept += plan[-1]
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
            state=obs, action=action, mu=mu, sigma=sigma, eps=eps, pre_action=pre_action, next_state=next_obs,
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
            if allow_train:
                writer.add_scalar("charts/plan_gain", plan_gain / ep_length, global_step)
                writer.add_scalar("charts/plan_kept_mean", plan_kept / ep_length, global_step)
            plan_gain, plan_kept = 0.0, 0.0
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
                "self_model": agent.self_model.state_dict(),
                "model": agent.model.state_dict(),
                "mind": agent.mind.state_dict(),
                "checkpoint_mind": agent.checkpoint_mind.state_dict(),
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    env.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()

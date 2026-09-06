# SDPLAST v1 -- state-dependent plasticity: every perceptron sets its own
# per-sample learning rate, supervised by an EXACT first-order meta-gradient.
# =====================================================================================
# PREMISE
#   Each hidden unit i owns a plasticity policy g_i(state) and, for every single
#   sample t in the minibatch, decides how strongly that sample's gradient is
#   allowed to move its own incoming weights w_i (and bias b_i):
#       grad w_i = sum_t  g_{t,i} * delta_{t,i} * x_t          (delta = dL/dz)
#   Nothing else changes: the value the layer computes is BIT-IDENTICAL to
#   nn.Linear, and the gradient flowing back to earlier layers is UNGATED. A
#   unit throttles its own learning, never the network's credit assignment.
#
# HOW g IS LEARNED (the part every prior attempt got wrong)
#   Write H_i = m_i / sqrt(v_i) for unit i's row of Adam's normalized momentum:
#   the direction the unit has actually been travelling, built from PAST steps
#   only, and scale-free: m and sqrt(v) carry the same units, so H is invariant
#   to a uniform rescale of every gradient. A unit cannot earn plasticity by
#   being loud -- only by being consistent. This is why H is Adam's NORMALIZED
#   momentum and not the raw gradient trace, and it is what removes the
#   magnitude-reward confound v8 had to dodge by hand.
#   One Adam step is dw_i ~ -lr*H_i, so the first-order change in future loss is
#       dL/dg_{t,i}  =  -c * delta_{t,i} * ( <x_t, H_i> + H^b_i ).
#   Agreement between THIS sample's gradient contribution and the direction that
#   has been working => raise plasticity; conflict => lower it. That is Sutton's
#   IDBD / hypergradient rule, resolved at (sample, unit) granularity.
#   It is injected exactly, not regressed: the layer adds a term whose VALUE is
#   identically zero and whose derivative w.r.t. g is that expression, so the
#   gate head is trained by plain backprop on the true meta-objective.
#
# WHY THIS IS THE FIX, NOT A VARIATION
#   `iterthink/.../statedynlr_v8_postadam` (9519.8 vs its 8454.7 base) supervises
#   a controller with e_j = -<grad_w_j, raw_dw_j>, i.e. exactly lr*<G_j, H_j>,
#   z-scored across units and regressed onto a mean-one target. Summing our
#   per-sample signal over t gives precisely <G_i, H_i>: v8's target is the
#   SAMPLE-MARGINAL of this one. Collapsing to units makes the state-conditional
#   part unidentifiable, which is why v8's realized rate std was 0.005-0.013
#   against a target std of 0.28-0.31, and why v9 had to reach for a unit-id
#   embedding (i.e. give up on state). Here the per-sample resolution is retained
#   and the regression, z-scoring, target construction and hooks all disappear.
#   `beta_policy/betaplast_*` gates on a FREE parameter (state-independent, and
#   its gates never left 1+-20%); `idbd/` adapts per-weight step sizes from
#   gradient correlation only and was measurably inert at 8M.
#
# TWO COMPONENTS, BECAUSE ADAM EATS ONE OF THEM
#   A per-row gradient rescale cancels inside Adam (m and sqrt(v) scale
#   together) -- statedynlr_v7 collapsed to 1869 proving it. So g is split, as
#   in v8, into the two parts that survive:
#     data_{t,i} = g_{t,i} / mean_t(g_{t,i})   pre-Adam, per-sample; changes the
#                                              gradient DIRECTION, so Adam cannot
#                                              normalize it away
#     rate_i     = mean_t(g_{t,i}) / mean(.)   applied to the realized weight-row
#                                              and bias step AFTER optimizer.step
#   Both are normalized to mean exactly one, and the gate logit has its layer
#   mean subtracted INSIDE the graph, so the uniform direction carries exactly
#   zero gradient. Consequence: the mechanism is learning-rate-neutral by
#   construction and CANNOT degenerate into a global LR change -- the failure
#   mode that made statedynlr_v1 its lineage's top scorer (alpha_std = 0.000, a
#   uniform 2.72x rail). Any gain here is attributable to redistribution alone.
#
# PERFORMANCE (no compromise on the premise, no compromise on speed)
#   The gate is expressed as detach-algebra over ordinary ops, so the whole
#   thing lives inside the existing fullgraph/cudagraph-compiled PPO loss: no
#   autograd hooks, no per-minibatch clones of the network, no custom Function.
#   Forward is provably exact (`data * (linear(x.detach(), W, b) - z)` has value
#   identically 0), so the host actor mirror and the rollout path are untouched
#   and the GPU never sees an extra rollout op. Cost is +1 small matmul per
#   gated layer for <x_t, H_i> plus a rank-8 context, and ~40 foreach/in-place
#   launches per optimizer step (~10 s over 8M steps).
#   At initialization every gate parameter is zero => g == 0.5 uniformly =>
#   data == rate == 1 exactly => this file is bit-identical to the baseline.
#
# HYPOTHESIS
#   MuJoCo locomotion gradients are phase-structured: a unit's contribution is
#   reliable during stance and conflicting during flight, and which unit is
#   reliable changes with the state. PPO spends 10 epochs averaging those
#   conflicting contributions into one direction per unit. Letting each unit
#   decline credit in the states where its gradients historically fight itself
#   should cut gradient interference without touching the policy gradient's
#   sign, its expectation over states, or the layer's total learning rate.
#
# PASS: beats the shared-chassis baseline (8455 final / 8278 @8M) at matched
#   steps with gate dispersion that actually grows (sdp/data_std, sdp/rate_std
#   materially above statedynlr's 0.013) and rate mean pinned at 1.
# FAIL: >5% below baseline at two consecutive checkpoints, logits pinned at the
#   +-gate_cap rails, or dispersion stuck near zero (meta-signal too weak).
# WATCH: sdp/data_std, sdp/rate_std, sdp/logit_abs per gated site,
#   losses/explained_variance, losses/clipfrac, timing/update_s.
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.host_actor import HostMLP
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions, sample_beta_actions_host
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # State-dependent plasticity
    plastic_actor: bool = True
    """gate the plasticity of the actor's hidden units"""
    plastic_critic: bool = True
    """gate the plasticity of the critic's hidden units"""
    ctx_dim: int = 8
    """rank of the shared layer-context each unit reads alongside its own state"""
    gate_cap: float = 2.0
    """logit bound; gate in (sigmoid(-cap), sigmoid(cap)), so <=7.4x plasticity spread"""
    gate_lr: float = 3e-4
    """learning rate of the per-unit plasticity policies (annealed with the network)"""
    gate_clip: float = 0.5
    """gradient-norm clip for plasticity parameters, kept off the network's clip"""
    meta_coef: float = 1.0
    """scale of the injected plasticity meta-gradient"""
    post_adam_rate: bool = True
    """apply each unit's mean gate to its realized Adam step (the Adam-surviving half)"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 4
    """maximum physics threads, capped at num_envs"""
    compile: bool = True
    """compile deterministic policy statistics, PPO loss and GAE"""
    compile_mode: str = "reduce-overhead"
    """PyTorch compilation mode for fixed-shape paths"""
    non_blocking_transfers: bool = False
    """opt into event-protected asynchronous pinned transfers"""
    staggered_starts: bool = True
    """stagger parallel environments; warmup counts toward total_timesteps"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PlasticLinear(nn.Linear):
    """Linear layer whose per-(sample, unit) weight-gradient weight is chosen by
    a per-unit plasticity policy reading that unit's own state.

    Subclasses ``nn.Linear`` on purpose: the forward value is bit-identical, so
    ``HostMLP`` mirrors it and the rollout is unaffected. Under ``no_grad`` the
    gate machinery is skipped entirely, which is every inference path.

    Gradient contract, with delta = dL/dz the incoming pre-activation gradient:
        dL/dW_i = sum_t data_{t,i} * delta_{t,i} * x_t      (gated)
        dL/db_i = sum_t data_{t,i} * delta_{t,i}            (gated)
        dL/dx_t = W^T delta_t                               (UNGATED)
        dL/dg   = -meta_coef * delta * (<x, H> + H_b)        (meta-gradient)
    ``rate`` (the unit-mean gate) is exported through a buffer because Adam
    cancels a per-row gradient rescale; the trainer applies it to the realized
    step instead.
    """

    def __init__(self, in_features, out_features, ctx_dim=8, gate_cap=2.0, meta_coef=1.0):
        super().__init__(in_features, out_features)
        self.gate_cap = float(gate_cap)
        self.meta_coef = float(meta_coef)
        # Each unit's own plasticity policy: an affine readout over its own
        # (bounded, scale-free) state features plus a rank-ctx_dim view of the
        # layer input. All zero at init => gate is uniform => exactly baseline.
        self.ctx_proj = nn.Parameter(torch.empty(int(ctx_dim), in_features))
        nn.init.orthogonal_(self.ctx_proj, 1.0)
        self.ctx_read = nn.Parameter(torch.zeros(out_features, int(ctx_dim)))
        self.w_state = nn.Parameter(torch.zeros(out_features))
        self.w_mag = nn.Parameter(torch.zeros(out_features))
        self.g_bias = nn.Parameter(torch.zeros(out_features))
        # Adam's normalized momentum for this perceptron: the direction it has
        # been travelling, from past steps only. Refreshed by the trainer.
        self.register_buffer("hdir", torch.zeros(out_features, in_features))
        self.register_buffer("hbias", torch.zeros(out_features))
        # Exports read once per optimizer step / once per iteration.
        self.register_buffer("rate", torch.ones(out_features))
        self.register_buffer("stats", torch.zeros(3))

    def gate_parameters(self):
        return [self.ctx_proj, self.ctx_read, self.w_state, self.w_mag, self.g_bias]

    def forward(self, x):
        if not torch.is_grad_enabled():
            return F.linear(x, self.weight, self.bias)
        xd = x.detach()
        wd, bd = self.weight.detach(), self.bias.detach()
        z = F.linear(xd, wd, bd)

        # --- the perceptron's state: its own pre-activation, batch-RMS
        # normalized per unit (scale-free) and squashed (bounded features).
        zn = z * torch.rsqrt(z.square().mean(0, keepdim=True) + 1e-8)
        zs = zn.square()
        f_state = zn / (1.0 + zn.abs())
        f_mag = zs / (1.0 + zs) - 0.5
        ctx = torch.tanh(F.linear(xd, self.ctx_proj))
        raw = self.g_bias + self.w_state * f_state + self.w_mag * f_mag + F.linear(ctx, self.ctx_read)
        # Subtract the layer mean INSIDE the graph: the uniform direction then
        # has exactly zero gradient, so no parameter setting can turn this into
        # a global learning-rate change.
        raw = raw - raw.mean()
        logit = self.gate_cap * torch.tanh(raw / self.gate_cap)
        gate = torch.sigmoid(logit)

        # --- exact mean-one split into the two Adam-surviving factors.
        gd = gate.detach()
        row = gd.mean(0)
        data = gd / row
        rate = row / row.mean()
        self.rate.copy_(rate)
        self.stats.copy_(torch.stack((data.std(), rate.std(), logit.detach().abs().mean())))

        # --- first-order meta-gradient: alignment of this sample's gradient
        # contribution with the direction this unit has been travelling.
        align = F.linear(xd, self.hdir, self.hbias)
        align = align * torch.rsqrt(align.square().mean() + 1e-12)
        align = align.clamp(-8.0, 8.0)
        meta = gate * (-self.meta_coef * align)

        # Value is exactly F.linear(x, W, b): the second term is data * 0 and the
        # third is (meta - meta), both identically zero in floating point.
        y = F.linear(x, wd, bd)
        y = y + data * (F.linear(xd, self.weight, self.bias) - z)
        return y + (meta - meta.detach())


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta PPO requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())

        def hidden(in_features, out_features, plastic):
            if not plastic:
                return nn.Linear(in_features, out_features)
            return PlasticLinear(in_features, out_features, args.ctx_dim, args.gate_cap, args.meta_coef)

        self.critic = nn.Sequential(
            layer_init(hidden(observation_dim, 64, args.plastic_critic)), nn.Tanh(),
            layer_init(hidden(64, 64, args.plastic_critic)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(hidden(observation_dim, 64, args.plastic_actor)), nn.Tanh(),
            layer_init(hidden(64, 64, args.plastic_actor)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        self.plastic_sites = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, PlasticLinear)
        ]

    def gate_parameters(self):
        return [p for _, module in self.plastic_sites for p in module.gate_parameters()]

    def network_parameters(self):
        gate_ids = {id(p) for p in self.gate_parameters()}
        return [p for p in self.parameters() if id(p) not in gate_ids]

    def get_value(self, x):
        return self.critic(x)

    def get_policy_and_value(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(x)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, x, action=None):
        """Public API uses physical actions; training retains native samples."""
        alpha, beta, value = self.get_policy_and_value(x)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((x.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(x.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, value


class PlasticityStepper:
    """The two update-path duties the compiled loss cannot own.

    1. ``apply_rates``: Adam is invariant to a per-row gradient rescale, so each
       unit's mean gate is applied to the REALIZED step instead, as the
       correction w_i += (rate_i - 1) * (w_i^after - w_i^before), leaving Adam's
       moments untouched. Written as a correction rather than as
       w_i^before + rate_i * step it is bit-exact for a neutral unit
       (rate_i == 1) instead of rounding the whole step through |w|.
       ``statedynlr_v7`` proved the pre-Adam alternative collapses (1869
       @1.58M, dw-correlation -0.36).
    2. ``refresh``: republish H = m / sqrt(v) from Adam's own state, so the
       meta-gradient reads the direction the unit actually travelled, built from
       past steps only (no self-correlation with the current sample) and
       scale-free (so a unit cannot earn plasticity by having large gradients).
    """

    def __init__(self, sites, optimizer, apply_rates):
        self.layers = [module for _, module in sites]
        self.optimizer = optimizer
        self.enabled = apply_rates and bool(self.layers)
        self.params = [p for layer in self.layers for p in (layer.weight, layer.bias)]
        self.buffers = [b for layer in self.layers for b in (layer.hdir, layer.hbias)]
        self.snapshots = [torch.empty_like(p) for p in self.params]
        self._moments = None

    def stash(self):
        if self.enabled:
            torch._foreach_copy_(self.snapshots, self.params)

    @torch.no_grad()
    def apply_rates(self):
        if not self.enabled:
            return
        for index, layer in enumerate(self.layers):
            offset = layer.rate - 1.0
            for param, snapshot, gain in (
                (layer.weight, self.snapshots[2 * index], offset.unsqueeze(1)),
                (layer.bias, self.snapshots[2 * index + 1], offset),
            ):
                # snapshot: w_before -> -(step) -> -(rate-1)*step, then subtract.
                snapshot.sub_(param).mul_(gain)
                param.sub_(snapshot)

    @torch.no_grad()
    def refresh(self):
        if not self.layers:
            return
        if self._moments is None:
            state = self.optimizer.state
            if any(p not in state for p in self.params):
                return  # Adam allocates its moments on the first step.
            self._moments = (
                [state[p]["exp_avg"] for p in self.params],
                [state[p]["exp_avg_sq"] for p in self.params],
            )
        first, second = self._moments
        scale = torch._foreach_sqrt(second)
        torch._foreach_add_(scale, 1e-8)
        torch._foreach_copy_(self.buffers, first)
        torch._foreach_div_(self.buffers, scale)


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args):
    """Pure clipped PPO loss on native Beta samples; no inverse action scaling."""
    alpha, beta, newvalue = agent.get_policy_and_value(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac))
    return loss, metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    if args.ctx_dim <= 0 or args.gate_cap <= 0.0:
        raise ValueError("ctx_dim and gate_cap must be positive")
    if args.minibatch_size < 2:
        raise ValueError("per-unit batch statistics require at least two samples per minibatch")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id, args.num_envs, backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video, run_name=run_name,
    )


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=args.torch_deterministic,
                      matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); FP32; native-action storage; host actor mirror")
        writer.add_text("plasticity", "per-(sample,unit) gate; pre-Adam data weight x post-Adam row rate; "
                                      "IDBD meta-gradient from Adam m/sqrt(v)")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        gate_params = agent.gate_parameters()
        net_params = agent.network_parameters()
        optimizer = optim.Adam(
            [{"params": net_params, "lr": args.learning_rate},
             {"params": gate_params, "lr": args.gate_lr}],
            lr=args.learning_rate, eps=1e-5, fused=True,
        )
        plasticity = PlasticityStepper(agent.plastic_sites, optimizer, args.post_adam_rate)
        site_names = [name for name, _ in agent.plastic_sites]
        print(f"plastic sites: {site_names} | gate params: {sum(p.numel() for p in gate_params)}")
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        host_actor = HostMLP(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)

        def act(observations):
            native, action = sample_beta_actions_host(host_actor(observations), action_low, action_high, sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=warmup_action, horizon=horizon,
                                    phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                fraction = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
                optimizer.param_groups[1]["lr"] = fraction * args.gate_lr
            bootstraps.reset()
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native)
                global_step += args.num_envs
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs = rollout_statistics(b_obs, b_native)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(net_params, args.max_grad_norm)
                        if gate_params:
                            nn.utils.clip_grad_norm_(gate_params, args.gate_clip)
                        plasticity.stash()
                        optimizer.step()
                        plasticity.apply_rates()
                        plasticity.refresh()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            logged = {
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
            }
            for name, module in agent.plastic_sites:
                site = name.replace(".", "_")
                logged[f"sdp/{site}_data_std"] = module.stats[0]
                logged[f"sdp/{site}_rate_std"] = module.stats[1]
                logged[f"sdp/{site}_logit_abs"] = module.stats[2]
            logged = gather_metrics(logged)
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        transfer.close()
        envs.close()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

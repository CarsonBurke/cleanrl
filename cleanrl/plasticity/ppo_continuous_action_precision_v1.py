# Per-unit ANISOTROPIC PRECISION plasticity on the realized optimizer step.
#
# METHOD. Every hidden perceptron keeps the inverse geometry of the input
# directions along which IT has already been modified:
#
#     P_h = (I + lambda * C_h)^-1,   C_h = sum_t c_h,t * u_h,t u_h,t^T
#
# `u_h,t` is the unit-norm input direction unit h was moved along at update t and
# `c_h,t` is how far it actually moved along it, measured in units of the current
# learning rate (`mean|dw_h| / lr`, so it is dimensionless and LR-anneal
# invariant). `C_h` is maintained implicitly: `P_h` is updated by an exact
# Sherman-Morrison rank-1 downdate, never inverted. The unit's incoming-row step
# is then projected, `dw_h <- P_h dw_h`, so it is FREE along input directions
# novel to that unit and DAMPED along the ones it is already committed to. The
# realized plasticity of unit h on this update is the Rayleigh quotient
# `dw_h^T P_h dw_h / dw_h^T dw_h` in (0, 1].
#
# NOVELTY. Every step-size rule in this repo's falsified lineage (t-tests, FDP
# levels, IDBD, Kalman/RLS, spike-and-slab, meta-gradient readouts) produces a
# per-weight SCALAR, i.e. a function of TIME only. A scalar provably cannot say
# "move for these inputs and not for those" -- that is a statement about
# DIRECTIONS in the unit's input space, so the state must be a per-unit matrix.
# The diagonal of exactly this object is an ordinary per-weight RMS accumulator,
# which is why `--precision-diagonal` is the control and not a variant: it holds
# the accumulator and drops only the directional content.
#
# SYNTHETIC EVIDENCE IS NULL. The 4-seed synthetic result that motivated this
# form did not survive 8 seeds: on `cleanrl/plasticity/novelty_stream.py`
# (shift 12, LR swept per arm, 8 seeds) adam scores 0.6201 and precision 0.6530,
# i.e. WORSE, with the diagonal at 0.6645 and a single layer-wide shared
# geometry at 0.6210. A geometry probe on that task shows the learned geometry
# is genuinely anisotropic (median max/min eigenvalue 543x) but that different
# units' most-damped directions have median |cos| = 0.77 -- the shift-12 region
# centre dominates every unit's input, so the task has directional content and
# almost no PER-UNIT content. Nothing here is justified by a synthetic win.
# This file exists to test the mechanism where the non-stationarity is real, and
# it changes the one thing the probe indicted: the committed direction is each
# unit's OWN normalised gradient row, not a batch-mean input shared by every
# unit, so per-unit divergence is possible in principle rather than aliased onto
# a common input mean. `--precision-direction mean` restores the shared form.
#
# MEASURED HERE (`/tmp/verify_precision_v1.py`, 400 real PPO updates on real
# HalfCheetah-v4 rollouts, lambda=0.5, decay=1e-3). The mechanism is bit-exactly
# the baseline while the geometry is still the identity (max parameter gap
# 0.0e+00 on update 1, and 0.0e+00 for 40 updates at lambda=0). It then does
# change the direction: mean angle between the raw Adam step and the projected
# step is 18.8 deg, against 6.1 deg for the diagonal ablation matched to the
# same geometric-mean plasticity, i.e. the directional content is 3.1x. Realized
# plasticity is dispersed across units (per-unit quantiles 0.23 / 0.36 / 0.45 /
# 0.59 / 0.87, cross-unit sd 0.081) and its geometric mean is 0.505, so the mean
# effective learning rate is 4.1e-4 -- BELOW the 8.1e-4 baseline that scores
# 8369 and above the 3e-4 baseline that scores 7468. A win therefore cannot be
# explained by a better mean step than either baseline.
#   HONEST CAVEAT, same measurement: the median |cos| between different units'
# most-damped directions is 0.72 to 0.95 by layer. So on real HalfCheetah too
# the units largely agree on WHICH direction to damp -- the per-unit content is
# weak and this is closer to a per-layer anisotropic preconditioner than to
# per-perceptron autonomy. That is a property of the task, not of the
# implementation, and it is the honest reading of the prior.
#
# HYPOTHESIS. PPO's non-stationarity is state-conditional: as the policy
# improves, the visited region of observation space moves, and the trunk is asked
# to fit a new mapping there while keeping the old one. A unit that has already
# committed along a direction should spend its step budget elsewhere. If this is
# what limits PPO, the full form beats the diagonal ablation at the same mean
# step, and beats the LR-tuned baseline (HalfCheetah-v4, 8369 -- NOT the 7468
# default-LR baseline, since every earlier win in this family was an LR
# artifact) despite a LOWER mean effective learning rate. Prior expectation is
# low given the synthetic null; the full/diagonal pair is controlled and
# informative in either direction.
#
# ORDERING. The projection is applied POST-optimizer, to the step Adam actually
# took. Adam divides out any persistent per-row rescale of its input, so a
# pre-optimizer projection would contribute only direction distortion and no
# magnitude control (measured in `cleanrl/shared/state_plasticity.py`).
#
# DEPTH. Both hidden layers carry a FULL d x d inverse per unit: d = 17 (obs) and
# d = 64 (trunk width), stored zero-padded to a single 64x64 block per unit, so
# 4.19 MB of state for actor and critic together (measured) -- a full inverse is
# simply affordable here, and no sketch can be justified while it is. The readout
# layer is deliberately NOT projected: its incoming row is the unit's OUTGOING
# weight, and damping both ends of a unit whose outgoing weight starts near zero
# pins its incoming gradient at zero forever (measured deadlock, reported on the
# synthetic harness). A trunk wide enough to make d^2 unaffordable would need a
# low-rank sketch of C_h; that is not this configuration.
#
# Baseline: bounded-action Beta PPO (`cleanrl/ppo_continuous_action.py`). Default
# LR is the tuned 8.1e-4 control, not the 3e-4 default, so the mechanism's mean
# effective step lands BELOW the strongest baseline rather than above it.
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

from cleanrl.shared.host_actor import HostMLP
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
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
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 8.1e-4
    """the learning rate of the optimizer; the LR-tuned baseline control, so the
    mechanism's mean effective step is bracketed by measured baselines"""
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

    # Per-unit anisotropic precision plasticity.
    precision: bool = True
    """enable the mechanism; off reproduces the Beta PPO baseline exactly"""
    precision_lambda: float = 0.5
    """weight on a perceptron's accumulated input geometry. Larger commits
    harder, i.e. damps more along already-used directions. 0.5 is calibrated to
    a geometric-mean realized plasticity of 0.505, which puts the mean effective
    learning rate at 4.1e-4, between the two measured baselines. The DIAGONAL
    ablation needs 1.5 to reach the same 0.502 -- it must be matched on realized
    plasticity, not on lambda, or it is not a control"""
    precision_decay: float = 1e-3
    """per-update leak of that geometry back toward the identity. Without a leak
    the accumulated commitment saturates over 78k updates and the rule degrades
    into a plain global learning-rate cut"""
    precision_diagonal: bool = False
    """ABLATION. Keep only the diagonal of the inverse geometry, which is an
    ordinary per-weight accumulator and provably cannot represent a direction.
    This is the control that isolates the state-conditional content"""
    precision_direction: str = "grad"
    """`grad`: the direction the unit was actually moved along this update, its
    own normalised gradient row. `mean`: the batch-mean layer input shared by
    every unit (exact parity with the synthetic harness; at PPO's minibatch size
    of 1024 the mean of a normalised observation is near zero, so this form
    carries almost no state information for the first layer)"""

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


# Public evaluation and historical GAE compatibility helpers; not the training path.
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


class Agent(nn.Module):
    def __init__(self, envs):
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
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )

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


class UnitPrecision:
    """Per-unit anisotropic precision applied to the realized optimizer step.

    Usage, around an existing optimizer::

        loss.backward()
        precision.before_step(observations)     # snapshot weights (+ mean inputs)
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        optimizer.step()
        precision.after_step(lr)                # project the realized step, commit

    One instance owns every hidden ``nn.Linear`` of the supplied trunks; the last
    ``nn.Linear`` of each trunk is the readout and is left untouched.

    LAYOUT. Every tracked unit of every tracked layer lives in ONE tensor, rows
    zero-padded out to the widest fan-in: geometry ``(U, D, D)`` in full mode and
    ``(U, D)`` in diagonal mode, where ``U`` is the total unit count. Padding is
    exact, not approximate: a pad column of the step is identically zero, so it
    contributes nothing to the projection, the Rayleigh quotient or the rank-1
    commitment, and the pad block of the geometry provably never leaves the
    identity. The point is throughput -- this path is launch-bound, not
    FLOP-bound, so one batched op over 256 units costs what one op over 64 does
    and the whole mechanism is ~20 kernels per update instead of ~90. It wastes
    memory only when fan-ins differ wildly; here D = 64 against fan-ins 17 and
    64, i.e. 4.2 MB of state for actor and critic together.
    """

    def __init__(self, trunks, lam, decay, diagonal=False, direction="grad"):
        if direction not in ("grad", "mean"):
            raise ValueError("precision_direction must be `grad` or `mean`")
        if not 0.0 <= decay < 1.0:
            raise ValueError("precision_decay must lie in [0, 1)")
        if lam < 0.0:
            raise ValueError("precision_lambda must be non-negative")
        self.lam, self.decay = float(lam), float(decay)
        self.diagonal, self.direction = bool(diagonal), direction
        self.trunks, self.weights = [], []
        for trunk in trunks:
            modules = list(trunk)
            linears = [i for i, module in enumerate(modules) if isinstance(module, nn.Linear)]
            if len(linears) < 2:
                raise ValueError("a trunk needs at least one hidden layer plus a readout")
            tracked = linears[:-1]
            self.trunks.append((modules, tracked))
            self.weights.extend(modules[index].weight for index in tracked)
        shapes = [tuple(weight.shape) for weight in self.weights]
        units = sum(shape[0] for shape in shapes)
        width = max(shape[1] for shape in shapes)
        weight = self.weights[0]
        device, dtype = weight.device, weight.dtype
        if self.diagonal:
            self.geometry = torch.ones((units, width), device=device, dtype=dtype)
        else:
            self.geometry = torch.eye(width, device=device, dtype=dtype
                                      ).expand(units, width, width).clone()
        self.snapshot = torch.zeros((units, width), device=device, dtype=dtype)
        self.step = torch.zeros((units, width), device=device, dtype=dtype)
        self.gradient = torch.zeros((units, width), device=device, dtype=dtype)
        # Reciprocal true fan-in per row: the padded columns must not dilute the
        # mean absolute movement that weights each commitment.
        self.inverse_fan_in = torch.zeros((units, 1), device=device, dtype=dtype)
        self.blocks, offset = [], 0
        for rows, fan_in in shapes:
            self.blocks.append((slice(offset, offset + rows), fan_in))
            self.inverse_fan_in[offset:offset + rows] = 1.0 / fan_in
            offset += rows
        self.snapshot_views = [self.snapshot[rows, :fan_in] for rows, fan_in in self.blocks]
        self.step_views = [self.step[rows, :fan_in] for rows, fan_in in self.blocks]
        self.gradient_views = [self.gradient[rows, :fan_in] for rows, fan_in in self.blocks]
        self.directions = [None] * len(self.weights)
        self.updates = 0
        # Plasticity telemetry stays on the device; one D2H sync per log interval.
        self._zero = torch.zeros((), device=device)
        self.stat_sum = self._zero.clone()
        self.stat_log_sum = self._zero.clone()
        self.stat_sq_sum = self._zero.clone()
        self.stat_spread = self._zero.clone()
        self.stat_min = torch.ones((), device=device)
        self.stat_count = 0
        self.stat_every = 4

    def parameters_state_bytes(self):
        return self.geometry.numel() * self.geometry.element_size()

    def unit_geometry(self, index):
        """The geometry of tracked layer `index`, padding removed. Diagnostics only."""
        rows, fan_in = self.blocks[index]
        if self.diagonal:
            return self.geometry[rows, :fan_in]
        return self.geometry[rows, :fan_in, :fan_in]

    @torch.no_grad()
    def before_step(self, observations=None):
        """Snapshot the weights, and for `mean` mode the batch-mean layer inputs."""
        torch._foreach_copy_(self.snapshot_views, self.weights)
        if self.direction != "mean":
            return
        if observations is None:
            raise ValueError("`mean` direction needs the minibatch observations")
        slot = 0
        for modules, tracked in self.trunks:
            activations, last = observations, tracked[-1]
            for index, module in enumerate(modules):
                if index in tracked:
                    self.directions[slot] = activations.mean(0)
                    slot += 1
                    if index == last:
                        break
                activations = module(activations)

    @torch.no_grad()
    def after_step(self, lr):
        """Project the realized step per unit, then commit this update's direction."""
        self.updates += 1
        inverse_lr = 1.0 / max(float(lr), 1e-12)
        state, step = self.geometry, self.step
        torch._foreach_copy_(self.step_views, self.weights)
        step.sub_(self.snapshot)                                        # (U, D)
        projected = state * step if self.diagonal else torch.bmm(
            state, step.unsqueeze(-1)).squeeze(-1)
        energy = step.square().sum(-1)
        # Rayleigh quotient in (0, 1]: this unit's realized plasticity on this
        # update. Reported, never fed back as a gate.
        plasticity = torch.where(energy > 0,
                                 (projected * step).sum(-1) / energy.clamp_min(1e-30),
                                 torch.ones_like(energy))
        # How far this unit ACTUALLY moved, in units of the current learning
        # rate: dimensionless, and invariant to LR annealing.
        commitment = projected.abs().sum(-1, keepdim=True).mul_(
            self.inverse_fan_in * inverse_lr)                           # (U, 1)
        # Write back the correction rather than reconstructing the weight from
        # the snapshot: at P = I the correction is identically zero, so the whole
        # mechanism is a BIT-EXACT no-op until it has committed to something.
        projected.sub_(step)
        torch._foreach_add_(self.weights,
                            [projected[rows, :fan_in] for rows, fan_in in self.blocks])
        if self.direction == "grad":
            torch._foreach_copy_(self.gradient_views,
                                 [weight.grad for weight in self.weights])
            unit = self.gradient / self.gradient.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        else:
            for view, shared in zip(self.gradient_views, self.directions):
                view.copy_(shared / shared.norm().clamp_min(1e-20))
            unit = self.gradient
        # Sherman-Morrison downdate of P = (I + lambda C)^-1 for the rank-1
        # commitment lambda * c_h * u u^T. Exact; no matrix is ever inverted.
        factor = self.lam * commitment
        if self.diagonal:
            mapped = state * unit
            quadratic = (mapped * unit).sum(-1, keepdim=True)
            state.sub_(mapped.square_().mul_(factor / (1.0 + factor * quadratic)))
        else:
            mapped = torch.bmm(state, unit.unsqueeze(-1)).squeeze(-1)
            quadratic = (mapped * unit).sum(-1, keepdim=True)
            weighted = mapped * (factor / (1.0 + factor * quadratic))
            state.baddbmm_(weighted.unsqueeze(-1), mapped.unsqueeze(-2), alpha=-1.0)
        if self.decay:
            # Leak commitments back toward the identity. A finite memory is
            # required: the alternative is saturation, at which point every
            # direction is damped equally and the rule is a global LR cut.
            if self.diagonal:
                state.mul_(1.0 - self.decay).add_(self.decay)
            else:
                state.mul_(1.0 - self.decay)
                state.diagonal(dim1=-2, dim2=-1).add_(self.decay)
        if self.updates % self.stat_every == 0:
            mean = plasticity.mean()
            mean_square = plasticity.square().mean()
            self.stat_sum += mean
            self.stat_sq_sum += mean_square
            self.stat_spread += (mean_square - mean.square()).clamp_min(0).sqrt()
            self.stat_log_sum += plasticity.clamp_min(1e-12).log().mean()
            self.stat_min = torch.minimum(self.stat_min, plasticity.min())
            self.stat_count += 1
        return plasticity

    def plasticity_stats(self, reset=True):
        """Mean, geometric mean and cross-unit dispersion of realized plasticity."""
        if self.stat_count == 0:
            return {}
        count = float(self.stat_count)
        stats = {
            "plasticity/mean": self.stat_sum / count,
            "plasticity/geomean": (self.stat_log_sum / count).exp(),
            "plasticity/rms": (self.stat_sq_sum / count).sqrt(),
            "plasticity/unit_spread": self.stat_spread / count,
            "plasticity/min": self.stat_min.clone(),
        }
        if reset:
            self.stat_sum = self._zero.clone()
            self.stat_log_sum = self._zero.clone()
            self.stat_sq_sum = self._zero.clone()
            self.stat_spread = self._zero.clone()
            self.stat_min = torch.ones_like(self._zero)
            self.stat_count = 0
        return stats


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
    if args.precision_diagonal and not args.precision:
        raise ValueError("the diagonal ablation needs the mechanism enabled")
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
        writer.add_text("policy", "Beta PPO; per-unit anisotropic precision on the realized step")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value
        precision = None
        if args.precision:
            precision = UnitPrecision([agent.actor, agent.critic],
                                      lam=args.precision_lambda, decay=args.precision_decay,
                                      diagonal=args.precision_diagonal,
                                      direction=args.precision_direction)
            writer.add_text("precision", f"lambda={args.precision_lambda} decay={args.precision_decay} "
                                         f"diagonal={args.precision_diagonal} "
                                         f"direction={args.precision_direction} "
                                         f"state_bytes={precision.parameters_state_bytes()}")

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
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
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
            current_lr = optimizer.param_groups[0]["lr"]
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        minibatch_obs = b_obs[indices]
                        loss, metrics = loss_model(
                            minibatch_obs, b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if precision is not None:
                            precision.before_step(minibatch_obs)
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        if precision is not None:
                            precision.after_step(current_lr)
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            logged = {
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
            }
            if precision is not None:
                logged.update(precision.plasticity_stats())
            logged = gather_metrics(logged)
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", current_lr, global_step)
            if precision is not None:
                writer.add_scalar("charts/effective_learning_rate",
                                  current_lr * logged["plasticity/geomean"], global_step)
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

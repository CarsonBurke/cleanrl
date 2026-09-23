# Performance-only v7: capture the complete tangent-value minibatch update.
# Hypothesis: one manual CUDA graph removes launch overhead without changing
# v6 optimizer mathematics, actor, architecture, rollout cadence or training budget.
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Literal

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
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
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
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    actor_update: Literal["tangent", "adam"] = "tangent"
    """v5 rollout-level tangent actor or baseline minibatch Adam PPO actor"""
    learning_rate: float = 3e-4
    """Adam actor initial LR, linearly annealed; tangent actor/critic have no LR"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """critic fitting epochs; Adam actor shares minibatches, tangent actor steps once"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    vf_coef: float = 0.5
    """critic half-MSE weight; six fresh GN-ray trials each minibatch, no value clip"""
    actor_kl: float = 0.01
    """exact behavior-policy KL budget per rollout, not an LR cap"""
    fisher_damping: float = 0.01
    """Tikhonov regularization for the singular empirical Fisher solve"""
    cg_iters: int = 10
    """fixed device-side conjugate-gradient iterations"""
    backtracks: int = 10
    """checked policy trials; origin retained if none is feasible"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile deterministic policy/value functions, functional gradients/JVPs and GAE"""
    value_graph: bool = True
    """capture each fixed-shape value update; --no-value-graph keeps the v6 execution path"""
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    log_action_scale: torch.Tensor

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


class TangentValueAdam(optim.Optimizer):
    """Fresh Gauss-Newton ray search on each minibatch, without LR history.

    Adam moments are observed once, including when every trial is rejected.
    step returns device scalars in this order: actual gain, model gain, absolute
    scale, accepted, fallback, slope, curvature, origin half-MSE, selected
    half-MSE, nonfinite trial fraction. Gains/slope/curvature include vf_coef;
    the two half-MSE measurements do not. The critic must be deterministic.
    """

    def __init__(self, critic: nn.Module, vf_coef=0.5, betas=(0.9, 0.999),
                 eps=1e-5, backtracks=6, compile=True):
        parameters = tuple(critic.parameters())
        if not parameters or len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("critic parameters must be nonempty and distinct")
        if not np.isfinite(vf_coef) or vf_coef < 0 or not np.isfinite(eps) or eps <= 0:
            raise ValueError("vf_coef must be nonnegative and epsilon positive, both finite")
        if len(betas) != 2 or not all(np.isfinite(beta) and 0 <= beta < 1 for beta in betas):
            raise ValueError("Adam decays must lie in [0, 1)")
        if not isinstance(backtracks, int) or backtracks < 1:
            raise ValueError("backtracks must be a positive integer")
        device = parameters[0].device
        if device.type != "cuda" or any(p.device != device for p in parameters):
            raise ValueError("TangentValueAdam requires parameters on one CUDA device")
        super().__init__(parameters, dict(betas=betas, eps=eps))
        self.parameters = parameters
        self.backtracks = backtracks
        self._compiled = compile
        names = tuple(name for name, _ in critic.named_parameters())
        buffers = dict(critic.named_buffers())
        for parameter in parameters:
            self.state[parameter] = dict(
                step=torch.zeros((), device=device), exp_avg=torch.zeros_like(parameter),
                exp_avg_sq=torch.zeros_like(parameter))
        self._cache_state()
        self._selected = tuple(torch.empty_like(parameter) for parameter in parameters)
        # actual gain, model gain, scale, half-MSE, accepted, invalid trial count.
        self._selection = parameters[0].new_zeros(6)
        self._fractions = tuple(parameters[0].new_tensor(0.5 ** index) for index in range(backtracks))

        def values(params, observations):
            return torch.func.functional_call(
                critic, (dict(zip(names, params)), buffers), (observations,), strict=True).flatten()

        def objective(params, observations, targets):
            return 0.5 * vf_coef * (values(params, observations) - targets).square().mean()

        def tangent(params, direction, observations):
            return torch.func.jvp(lambda p: values(p, observations), (params,), (direction,))[1]

        def prepare(gradients, moments, seconds, steps, beta1, beta2, epsilon):
            adam, current = [], []
            for gradient, mean, variance, step in zip(gradients, moments, seconds, steps):
                step.add_(1)
                mean.lerp_(gradient, 1.0 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                denominator = (variance / (1.0 - beta2 ** step)).sqrt() + epsilon
                adam.append((mean / (1.0 - beta1 ** step)) / denominator)
                current.append(gradient / denominator)
            fallback = tree_dot(gradients, adam) <= 0
            direction = tuple(torch.where(fallback, fresh, historical)
                              for fresh, historical in zip(current, adam))
            return direction, fallback

        def initialize(params, selected, selection, old_values, targets):
            for saved, parameter in zip(selected, params):
                saved.copy_(parameter)
            residual = old_values - targets
            origin_loss = 0.5 * residual.square().mean()
            selection.zero_()
            selection[3].copy_(origin_loss)
            return residual, origin_loss

        def model(residual, value_tangent):
            slope = vf_coef * (residual * value_tangent).mean()
            curvature = vf_coef * value_tangent.square().mean()
            usable = (torch.isfinite(slope) & (slope > 0)
                      & torch.isfinite(curvature) & (curvature > 0))
            scale = slope / torch.where(usable, curvature, 1.0)
            scale = torch.where(usable & torch.isfinite(scale) & (scale > 0), scale, 0.0)
            return slope, curvature, scale

        def trial(params, direction, scale, fraction):
            trial_scale = scale * fraction
            candidate = tuple(p - trial_scale * d for p, d in zip(params, direction))
            finite = torch.stack([torch.isfinite(p).all() for p in candidate]).all()
            moved = torch.stack([(new != old).any() for new, old in zip(candidate, params)]).any()
            return candidate, trial_scale, finite, moved

        def select(selected, selection, candidate, new_values, old_values, targets,
                   residual, trial_scale, slope, curvature, parameters_finite, moved):
            change = old_values - new_values
            # Paired per-example differences avoid subtracting large aggregate
            # losses. Identical values give an exact zero, even at tiny scales.
            actual_gain = vf_coef * (change * (residual - 0.5 * change)).mean()
            # eta*s - eta^2*q/2, factored to avoid an unnecessary eta^2 overflow.
            model_gain = trial_scale * (slope - 0.5 * trial_scale * curvature)
            trial_loss = 0.5 * (new_values - targets).square().mean()
            finite = (parameters_finite & torch.isfinite(new_values).all()
                      & torch.isfinite(actual_gain) & torch.isfinite(model_gain)
                      & torch.isfinite(trial_loss))
            take = finite & moved & (trial_scale > 0) & (actual_gain > selection[0])
            for saved, proposed in zip(selected, candidate):
                saved.copy_(torch.where(take, proposed, saved))
            selection.copy_(torch.stack((
                torch.where(take, actual_gain, selection[0]),
                torch.where(take, model_gain, selection[1]),
                torch.where(take, trial_scale, selection[2]),
                torch.where(take, trial_loss, selection[3]),
                torch.where(take, 1.0, selection[4]),
                selection[5] + (~finite).float(),
            )))

        def commit(params, selected, selection, fallback, slope, curvature, origin_loss):
            for parameter, saved in zip(params, selected):
                parameter.copy_(saved)
            return torch.stack((selection[0], selection[1], selection[2], selection[4],
                                fallback.float(), slope, curvature, origin_loss,
                                selection[3], selection[5] / backtracks))

        wrap = graph_compile if compile else lambda fn: fn
        # Never inline this into the gradient, JVP, or selection graphs. Every
        # origin/trial prediction uses this SAME no-grad compiled forward.
        self._values = wrap(values)
        self._gradient = wrap(torch.func.grad(objective))
        self._tangent = wrap(tangent)
        self._prepare = wrap(prepare)
        self._initialize = wrap(initialize)
        self._model = wrap(model)
        self._trial = wrap(trial)
        self._select = wrap(select)
        self._commit = wrap(commit)

    def _cache_state(self):
        self._moments = tuple(self.state[p]["exp_avg"] for p in self.parameters)
        self._seconds = tuple(self.state[p]["exp_avg_sq"] for p in self.parameters)
        self._steps = tuple(self.state[p]["step"] for p in self.parameters)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._cache_state()

    @torch.no_grad()
    def step(self, observations, targets):
        # Detached aliases are safe: model parameters remain untouched until
        # commit, and all selected candidates live in separate cached buffers.
        origin = tuple(parameter.detach() for parameter in self.parameters)
        targets = targets.flatten()
        old_values = self._values(origin, observations)
        residual, origin_loss = self._initialize(origin, self._selected, self._selection, old_values, targets)
        gradients = self._gradient(origin, observations, targets)
        settings = self.param_groups[0]
        direction, fallback = self._prepare(
            gradients, self._moments, self._seconds, self._steps, *settings["betas"], settings["eps"])
        value_tangent = self._tangent(origin, direction, observations)
        slope, curvature, scale = self._model(residual, value_tangent)
        for fraction in self._fractions:
            candidate, trial_scale, finite, moved = self._trial(origin, direction, scale, fraction)
            new_values = self._values(candidate, observations)
            self._select(self._selected, self._selection, candidate, new_values, old_values, targets,
                         residual, trial_scale, slope, curvature, finite, moved)
        return self._commit(self.parameters, self._selected, self._selection,
                            fallback, slope, curvature, origin_loss)


class GraphValueUpdate:
    """Capture one compiled value step without observing an extra minibatch.

    Construct after loading optimizer state; parameter/state storage and optimizer
    settings must stay fixed for this graph's lifetime. Inputs may change contents
    and strides, but not device, dtype or observation shape/target count. Replay
    and consumers run on the caller's stream. Returned metrics occupy permanent
    storage overwritten by the next step: consume or copy them immediately.
    """

    def __init__(self, optimizer: TangentValueAdam, observations, targets, *, warmup=3):
        if not optimizer._compiled:
            raise ValueError("value graphs require a compiled TangentValueAdam")
        if warmup < 1:
            raise ValueError("value graph warmup must be positive")
        self.optimizer = optimizer
        self.device = optimizer.parameters[0].device
        if observations.device != self.device or targets.device != self.device:
            raise ValueError("value graph inputs must share the optimizer's CUDA device")
        if observations.ndim < 1 or observations.shape[0] == 0 or targets.numel() != observations.shape[0]:
            raise ValueError("value graph requires one target per observation in a nonempty batch")
        self._observations = observations.detach().clone(memory_format=torch.contiguous_format)
        self._targets = targets.detach().flatten().clone()
        self._graph = torch.cuda.CUDAGraph()
        self._capture(warmup)

    @torch.no_grad()
    def _capture(self, warmup):
        # These are the actual tensors mutated by step, not detached replacements
        # that would leave the graph pointing at temporary optimizer state.
        mutable = self.optimizer.parameters + tuple(
            value for parameter in self.optimizer.parameters
            for value in self.optimizer.state[parameter].values())
        with torch.cuda.device(self.device), torch.random.fork_rng(devices=[self.device.index]):
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                saved = tuple(value.clone() for value in mutable)
            try:
                with torch.cuda.stream(stream):
                    for _ in range(warmup):
                        self.optimizer.step(self._observations, self._targets)
                    for value, original in zip(mutable, saved):
                        value.copy_(original)
                # All graph_compile paths (including every ray trial) must have
                # finished compiling/autotuning before manual capture begins.
                stream.synchronize()
                with torch.cuda.graph(self._graph, stream=stream):
                    self.metrics = self.optimizer.step(self._observations, self._targets)
            finally:
                with torch.cuda.stream(stream):
                    for value, original in zip(mutable, saved):
                        value.copy_(original)
                stream.synchronize()
        # Fractions are read-only. initialize overwrites _selected/_selection
        # before use on every step; capture scratch and metrics need no restore.

    @torch.no_grad()
    def step(self, observations, targets):
        if (observations.shape != self._observations.shape
                or observations.dtype != self._observations.dtype
                or observations.device != self.device
                or targets.numel() != self._targets.numel()
                or targets.dtype != self._targets.dtype
                or targets.device != self.device):
            raise ValueError("value graph inputs must keep their captured shape, dtype and CUDA device")
        self._observations.copy_(observations)
        self._targets.copy_(targets.flatten())
        self._graph.replay()
        return self.metrics


def tree_dot(left, right):
    return torch.stack([(a * b).sum() for a, b in zip(left, right)]).sum()


def beta_kl(old_alpha, old_beta, alpha, beta):
    """Exact KL(old || new), summed over independent action coordinates."""
    # Pure tensor formula: torch.distributions KL dispatch consults a mutable
    # Python registry and fails fullgraph tracing on a cold process.
    old_sum, new_sum = old_alpha + old_beta, alpha + beta
    log_normalizer_change = (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(new_sum)
                             - torch.lgamma(old_alpha) - torch.lgamma(old_beta) + torch.lgamma(old_sum))
    mean_log_x = torch.digamma(old_alpha) - torch.digamma(old_sum)
    mean_log_one_minus_x = torch.digamma(old_beta) - torch.digamma(old_sum)
    return (log_normalizer_change + (old_alpha - alpha) * mean_log_x
            + (old_beta - beta) * mean_log_one_minus_x).sum(-1)


def beta_fisher_metric(logits):
    """Cache F=L L^T in logit coordinates; factor trigamma differences in FP64."""
    alpha, beta = (F.softplus(logits) + 1.0).double().chunk(2, dim=-1)
    sa, sb = logits.double().sigmoid().chunk(2, dim=-1)
    shared = torch.polygamma(1, alpha + beta)
    l11 = (torch.polygamma(1, alpha) - shared).sqrt()
    l21 = -shared / l11
    l22 = (torch.polygamma(1, beta) - shared - l21.square()).sqrt()
    return tuple(value.to(logits.dtype) for value in (sa * l11, sb * l21, sb * l22))


def apply_beta_fisher(metric, tangent):
    l11, l21, l22 = metric
    da, db = tangent.chunk(2, dim=-1)
    first, second = l11 * da + l21 * db, l22 * db
    return torch.cat((l11 * first, l21 * first + l22 * second), dim=-1)


def cg_update(solution, residual, direction, product, residual_sq, initial_sq):
    denominator = tree_dot(direction, product)
    active = (residual_sq > initial_sq * 1e-10) & (denominator > 0) & torch.isfinite(denominator)
    step = torch.where(active, residual_sq / torch.where(active, denominator, 1.0), 0.0)
    solution = tuple(x + step * p for x, p in zip(solution, direction))
    residual = tuple(r - step * ap for r, ap in zip(residual, product))
    next_sq = tree_dot(residual, residual)
    ratio = torch.where(active, next_sq / torch.where(active, residual_sq, 1.0), 0.0)
    direction = tuple(r + ratio * p for r, p in zip(residual, direction))
    return solution, residual, direction, next_sq


class TangentPolicyOptimizer:
    """One cross-trajectory-checked policy-space step per fresh rollout.

    A natural direction solves the local Beta Fisher system. A tangent network
    z(theta + delta) ~= z(theta) + J delta predicts the *nonlinear* Beta policy,
    not just a scalar training loss. Scalar model steps target a declared KL
    budget; the actual network must satisfy that budget on fit AND held-out
    trajectories and improve both surrogates. Search uses fit streams only;
    the sealed holdout gets ONE final accept/reject decision, never retries.
    """

    def __init__(self, actor, num_envs, num_steps, kl_budget=0.01, damping=0.01,
                 cg_iters=10, backtracks=10, norm_adv=True, compile=True):
        self.actor = actor
        self.names = tuple(name for name, _ in actor.named_parameters())
        self.parameters = tuple(actor.parameters())
        self.num_envs, self.num_steps = num_envs, num_steps
        self.holdout_envs = max(1, num_envs // 4)
        self.fit_envs = num_envs - self.holdout_envs
        if num_envs < 4 or num_steps < 1 or cg_iters < 1 or backtracks < 1:
            raise ValueError("policy model requires >=4 environments and positive iteration counts")
        if not all(np.isfinite(v) and v > 0 for v in (kl_budget, damping)):
            raise ValueError("policy KL budget and Fisher damping must be positive and finite")
        if any(p.device.type != "cuda" for p in self.parameters):
            raise ValueError("policy model requires CUDA")
        self.kl_budget, self.damping = kl_budget, damping
        self.cg_iters, self.backtracks, self.norm_adv = cg_iters, backtracks, norm_adv

        def logits(params, observations):
            return torch.func.functional_call(actor, dict(zip(self.names, params)), (observations,), strict=True)

        def gain(params, observations, native, old_logprob, advantage):
            alpha, beta = (F.softplus(logits(params, observations)) + 1.0).chunk(2, -1)
            logprob = Beta(alpha, beta, validate_args=False).log_prob(native).sum(-1)
            return (torch.expm1(logprob - old_logprob) * advantage).mean()

        def fisher(params, vector, observations, metric):
            forward = lambda p: logits(p, observations)
            tangent = torch.func.jvp(forward, (params,), (vector,))[1]
            pullback = torch.func.vjp(forward, params)[1]
            return pullback(apply_beta_fisher(metric, tangent) / observations.shape[0])[0]

        def tangent(params, vector, observations):
            return torch.func.jvp(lambda p: logits(p, observations), (params,), (vector,))

        def check(output, old_alpha, old_beta, native, old_logprob, advantage):
            alpha, beta = (F.softplus(output) + 1.0).chunk(2, -1)
            distribution = Beta(alpha, beta, validate_args=False)
            logratio = distribution.log_prob(native).sum(-1) - old_logprob
            # PPO's pessimistic ratio objective stays intact; no value/grad clip.
            change = torch.expm1(logratio)
            gains = torch.minimum(change * advantage, change.clamp(-0.2, 0.2) * advantage)
            kls = beta_kl(old_alpha, old_beta, alpha, beta)
            streams = output.shape[0] // num_steps
            gains = gains.view(num_steps, streams).mean(0)
            kl_means = kls.view(num_steps, streams).mean(0)
            return torch.stack((gains, kl_means))

        wrap = graph_compile if compile else lambda fn: fn
        self._logits = wrap(logits)
        self._gradient = wrap(torch.func.grad(gain))
        self._fisher = wrap(fisher)
        self._tangent = wrap(tangent)
        self._check = wrap(check)
        self._cg_update = wrap(cg_update)
        self._dot = wrap(tree_dot)
        self._metric = wrap(beta_fisher_metric)

    @torch.no_grad()
    def step(self, observations, native, advantages, iteration):
        # Rotate whole environment streams, independently of current rewards.
        order = torch.arange(self.num_envs, device=observations.device).roll(iteration % self.num_envs)
        obs = observations.view(self.num_steps, self.num_envs, -1)[:, order].reshape_as(observations)
        actions = native.view(self.num_steps, self.num_envs, -1)[:, order].reshape_as(native)
        adv = advantages.view(self.num_steps, self.num_envs)[:, order]
        fit_adv = adv[:, :self.fit_envs].reshape(-1)
        if self.norm_adv:
            adv = (adv - fit_adv.mean()) / (fit_adv.std() + 1e-8)
        adv = adv.reshape(-1)

        def fit(tensor):
            return tensor.view(self.num_steps, self.num_envs, *tensor.shape[1:])[:, :self.fit_envs].flatten(0, 1)

        origin = tuple(p.detach().clone() for p in self.parameters)
        old_logits = self._logits(origin, obs)
        old_alpha, old_beta = (F.softplus(old_logits) + 1.0).chunk(2, -1)
        # Fixed action-range Jacobians cancel; use native densities throughout.
        old_logprob = Beta(old_alpha, old_beta, validate_args=False).log_prob(actions).sum(-1)
        fit_obs, fit_logits = fit(obs), fit(old_logits)
        fit_check = (fit(old_alpha), fit(old_beta), fit(actions), fit(old_logprob), fit(adv))
        metric = self._metric(fit_logits)
        gradient = self._gradient(origin, fit_obs, fit_check[2], fit_check[3], fit_check[4])
        solution = tuple(torch.zeros_like(g) for g in gradient)
        residual, direction = gradient, gradient
        initial_sq = self._dot(gradient, gradient)
        residual_sq = initial_sq
        for _ in range(self.cg_iters):
            product = self._fisher(origin, direction, fit_obs, metric)
            product = tuple(fp + self.damping * p for fp, p in zip(product, direction))
            solution, residual, direction, residual_sq = self._cg_update(
                solution, residual, direction, product, residual_sq, initial_sq)
        fisher_direction = self._fisher(origin, solution, fit_obs, metric)
        curvature = self._dot(solution, fisher_direction)
        slope = self._dot(gradient, solution)
        valid = torch.isfinite(curvature) & (curvature > 0) & torch.isfinite(slope) & (slope > 0)
        scale = torch.where(valid, (2.0 * self.kl_budget / torch.where(valid, curvature, 1.0)).sqrt(), 0.0)
        fit_tangent = self._tangent(origin, solution, fit_obs)[1]
        # Fresh output-space model, not inherited scalar LR. Only fit data.
        for _ in range(3):
            predicted = self._check(fit_logits + scale * fit_tangent, *fit_check)
            model_kl = predicted[1].mean()
            usable = torch.isfinite(model_kl) & (model_kl > 0)
            scale = torch.where(usable, scale * (self.kl_budget / torch.where(usable, model_kl, 1.0)).sqrt(), scale * 0.5)
        selected = torch.zeros((), device=obs.device, dtype=torch.bool)
        selected_scale = torch.zeros_like(scale)
        selected_fraction = torch.zeros_like(scale)
        for index in range(self.backtracks):
            fraction = 0.5 ** index
            trial_scale = scale * fraction
            trial = tuple(p + trial_scale * d for p, d in zip(origin, solution))
            actual = self._check(self._logits(trial, fit_obs), *fit_check)
            predicted = self._check(fit_logits + trial_scale * fit_tangent, *fit_check)
            # Sufficient gain is measured against the nonlinear tangent model,
            # not an EMA of ratios from unrelated minibatch objectives.
            feasible = (valid & torch.isfinite(actual).all() & torch.isfinite(predicted).all()
                        & (actual[1].mean() <= self.kl_budget)
                        & (predicted[0].mean() > 0)
                        & (actual[0].mean() >= 0.1 * predicted[0].mean()))
            take = feasible & ~selected
            selected_scale = torch.where(take, trial_scale, selected_scale)
            selected_fraction = torch.where(take, fraction, selected_fraction)
            selected = selected | take
        # Seal the candidate before examining held-out labels. A rejection
        # returns to the origin and waits for fresh data, never tries a new step.
        candidate = tuple(torch.where(selected, p + selected_scale * d, p) for p, d in zip(origin, solution))
        displacement = tuple(new - old for new, old in zip(candidate, origin))
        output_displacement = self._tangent(origin, displacement, obs)[1]
        actual_logits = self._logits(candidate, obs)
        check_args = (old_alpha, old_beta, actions, old_logprob, adv)
        actual = self._check(actual_logits, *check_args)
        predicted = self._check(old_logits + output_displacement, *check_args)
        held_gain = actual[0, self.fit_envs:]
        held_prediction = predicted[0, self.fit_envs:]
        held_kl = actual[1, self.fit_envs:].mean()
        agreement = held_gain - 0.1 * held_prediction
        accepted = (selected & torch.isfinite(actual).all() & torch.isfinite(predicted).all()
                    & (held_kl <= self.kl_budget) & (held_gain.mean() > 0)
                    & (held_prediction.mean() > 0) & (agreement.mean() > 0))
        for parameter, saved, proposed in zip(self.parameters, origin, candidate):
            parameter.copy_(torch.where(accepted, proposed, saved))
        # Cluster SE is diagnostic, not an IID-transition confidence interval.
        held_error = held_gain - held_prediction
        error_se = (held_error.var(unbiased=False) / max(1, self.holdout_envs - 1)).sqrt()
        gain_se = (held_gain.var(unbiased=False) / max(1, self.holdout_envs - 1)).sqrt()
        final_logits = torch.where(accepted, actual_logits, old_logits)
        alpha, beta = (F.softplus(final_logits) + 1.0).chunk(2, -1)
        predicted_alpha, predicted_beta = (F.softplus(old_logits + output_displacement) + 1.0).chunk(2, -1)
        actual_alpha, actual_beta = (F.softplus(actual_logits) + 1.0).chunk(2, -1)
        damping_energy = self.damping * self._dot(solution, solution)
        metrics = {
            "accepted": accepted.float(), "fit_selected": selected.float(),
            "step_scale": torch.where(accepted, selected_scale, 0.0),
            "accepted_fraction": torch.where(accepted, selected_fraction, 0.0),
            "cg_relative_residual": (residual_sq / initial_sq.clamp_min(1e-30)).sqrt(),
            "gradient_norm": initial_sq.sqrt(),
            "linear_predicted_gain": self._dot(gradient, displacement),
            "quadratic_predicted_kl": 0.5 * selected_scale.square() * curvature,
            "damping_energy_fraction": damping_energy / (curvature + damping_energy).clamp_min(1e-30),
            "parameter_displacement_norm": self._dot(displacement, displacement).sqrt(),
            "logit_linearization_rmse": (actual_logits - old_logits - output_displacement).square().mean().sqrt(),
            "prediction_policy_kl": beta_kl(predicted_alpha, predicted_beta, actual_alpha, actual_beta).mean(),
            "heldout_gain_se": gain_se, "heldout_model_error": held_error.mean(),
            "heldout_model_error_se": error_se,
            "entropy_native": Beta(alpha, beta, validate_args=False).entropy().sum(-1).mean(),
            "concentration_floor_fraction": ((alpha == 1.0) | (beta == 1.0)).float().mean(),
            "output_derivative_mean": final_logits.sigmoid().mean(),
            "committed_full_kl": torch.where(accepted, actual[1].mean(), 0.0),
        }
        for row, name in enumerate(("gain", "kl")):
            metrics["candidate_fit_" + name] = actual[row, :self.fit_envs].mean()
            metrics["candidate_heldout_" + name] = actual[row, self.fit_envs:].mean()
            metrics["predicted_fit_" + name] = predicted[row, :self.fit_envs].mean()
            metrics["predicted_heldout_" + name] = predicted[row, self.fit_envs:].mean()
        return {"policy_model/" + name: value for name, value in metrics.items()}


def value_metrics(checks):
    labels = ("selected_actual_gain", "selected_model_gain", "selected_scale",
              "accepted_fraction", "fallback_fraction", "slope", "curvature",
              "origin_half_mse", "selected_half_mse", "nonfinite_trial_fraction")
    metrics = {}
    for column, label in enumerate(labels):
        values = checks[:, column]
        if label in {"slope", "curvature"}:
            finite = torch.isfinite(values)
            metrics["value_model/" + label + "_finite_mean"] = (
                torch.where(finite, values, 0.0).sum() / finite.sum().clamp_min(1))
            metrics["value_model/" + label + "_nonfinite_fraction"] = (~finite).float().mean()
        else:
            metrics["value_model/" + label] = values.mean()
    metrics["value_model/origin_fraction"] = 1.0 - checks[:, 3].mean()
    return metrics


def adam_actor_loss(agent, observations, native_actions, old_logprobs, advantages, norm_adv):
    """Baseline actor-only PPO; critic targets/GAE remain frozen for the rollout."""
    alpha, beta = (F.softplus(agent.actor(observations)) + 1.0).chunk(2, dim=-1)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2)).mean()
    with torch.no_grad():
        metrics = torch.stack((loss.detach(), entropy.mean(), (-logratio).mean(),
                               ((ratio - 1.0) - logratio).mean(),
                               ((ratio - 1.0).abs() > 0.2).float().mean()))
    return loss, metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs, args.cg_iters, args.backtracks) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.value_graph and not args.compile:
        raise ValueError("value graphs require --compile; use --no-value-graph for eager execution")
    if args.value_graph and args.batch_size % args.minibatch_size:
        raise ValueError("value graphs require equal-sized minibatches; use --no-value-graph for a remainder")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if args.actor_update not in {"tangent", "adam"}:
        raise ValueError("actor_update must be tangent or adam")
    if args.actor_update == "tangent" and args.num_envs < 4:
        raise ValueError("trajectory holdout requires at least four environments")
    if args.actor_update == "adam" and (not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("Adam actor learning rate must be positive and finite")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
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
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        if args.actor_update == "tangent":
            actor_optimizer = TangentPolicyOptimizer(
                agent.actor, args.num_envs, args.num_steps, args.actor_kl, args.fisher_damping,
                args.cg_iters, args.backtracks, args.norm_adv, args.compile)
        else:
            actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        optimizer = TangentValueAdam(agent.critic, vf_coef=args.vf_coef, backtracks=6, compile=args.compile)
        value_update = None if args.value_graph else optimizer
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def actor_loss_model(observations, native, old_logprobs, advantages):
            return adam_actor_loss(agent, observations, native, old_logprobs, advantages, args.norm_adv)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            if args.actor_update == "adam":
                actor_loss_model = torch.compile(
                    actor_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
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
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        prediction_metrics = torch.empty((max_updates, 10), device=device)
        actor_metrics = torch.empty((max_updates, 5), device=device) if args.actor_update == "adam" else None
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
            if args.actor_update == "adam":
                actor_lr = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
                actor_optimizer.param_groups[0]["lr"] = actor_lr
                writer.add_scalar("charts/learning_rate", actor_lr, global_step)
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
            if args.actor_update == "tangent":
                with timer.span("actor_update"):
                    policy_metrics = actor_optimizer.step(b_obs, b_native, b_advantages, iteration)
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        mb_obs, mb_returns = b_obs[indices], b_returns[indices]
                        if args.actor_update == "adam":
                            actor_loss, metrics = actor_loss_model(
                                mb_obs, b_native[indices], b_logprobs[indices], b_advantages[indices])
                            actor_optimizer.zero_grad(set_to_none=True)
                            actor_loss.backward()
                            actor_optimizer.step()
                            actor_metrics[updates].copy_(metrics)
                        if value_update is None:
                            value_update = GraphValueUpdate(optimizer, mb_obs, mb_returns)
                        prediction_metrics[updates].copy_(value_update.step(mb_obs, mb_returns))
                        updates += 1
            if args.actor_update == "adam":
                last_actor = actor_metrics[updates - 1]
                policy_metrics = {
                    "losses/policy_loss": last_actor[0], "losses/entropy": last_actor[1],
                    "losses/old_approx_kl": last_actor[2], "losses/approx_kl": last_actor[3],
                    "losses/clipfrac": actor_metrics[:updates, 4].mean(),
                }
            with torch.no_grad():
                final_values = value_model(b_obs).flatten()
                logged = gather_metrics({
                    "losses/value_loss": 0.5 * (final_values - b_returns).square().mean(),
                    "losses/explained_variance": explained_variance(b_values, b_returns),
                    "losses/fitted_explained_variance": explained_variance(final_values, b_returns),
                    **value_metrics(prediction_metrics[:updates]), **policy_metrics,
                })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if "explained_variance" not in name):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
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
            torch.save(
                {
                    "model": agent.state_dict(),
                    "args": vars(args),
                    "obs_norm": {
                        "means": torch.from_numpy(obs_norm.means.copy()),
                        "variances": torch.from_numpy(obs_norm.variances.copy()),
                        "counts": torch.from_numpy(obs_norm.counts.copy()),
                        "epsilon": obs_norm.epsilon,
                        "clip": obs_norm.clip,
                    },
                },
                model_path,
            )
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

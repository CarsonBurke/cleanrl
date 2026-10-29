# Sepclip covariance confidence polar v6: trajectory-fold noise sets a metric.
# Propagate noise through momentum with correlated reuse inside each rollout.
# Preserve directional confidence; lane equalization is off by default.
# FP32 small-Gram eigensolves resolve covariance; no model SVD.
# Hypothesis: whitening reliable directions improves PPO over uniform polar.
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
    TruncationBootstrapCache, explained_variance,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch
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

def noise_covariance(noise):
    """Measured shorter-axis covariance, retaining trajectory-noise alignment."""
    x = noise.mT if noise.shape[-2] > noise.shape[-1] else noise
    return x @ x.mT


def soft_polar(matrix, covariance, noise_multiplier=1.0):
    """(M M.T + multiplier * C_noise)^(-1/2) M on the shorter axis.

    FP32 covariance/eigh is necessary: BF16 Gram rounding can overwhelm the
    covariance's small eigenvalues. The floor is rank times FP32 epsilon times
    the largest metric eigenvalue, bounding roundoff, not a statistical cutoff.
    """
    rank = min(matrix.shape[-2:])
    tall = matrix.shape[-2] > matrix.shape[-1]
    x = matrix.mT if tall else matrix
    metric = x @ x.mT + noise_multiplier * covariance
    eigenvalues, vectors = torch.linalg.eigh(metric)
    floor = eigenvalues.abs().amax(-1, keepdim=True) * (rank * torch.finfo(torch.float32).eps)
    floor = floor.clamp_min(torch.finfo(torch.float32).tiny)
    inverse = eigenvalues.clamp_min(floor).rsqrt()
    direction = (vectors * inverse.unsqueeze(-2)) @ (vectors.mT @ x)
    return direction.mT if tall else direction


def matrix_update(p, velocity, second, covariance_moment, covariance_prior, gradient,
                  noise, coefficients, lr, momentum, beta2, scale, noise_multiplier, lane_equalize):
    velocity.lerp_(gradient, 1 - momentum)
    lookahead = gradient.lerp(velocity, momentum)
    current = noise_covariance(noise)
    prior_moment, current_moment, prior_lookahead, current_lookahead = coefficients.unbind()
    covariance_moment.copy_(prior_moment * covariance_prior + current_moment * current)
    covariance = prior_lookahead * covariance_prior + current_lookahead * current
    direction = soft_polar(lookahead, covariance, noise_multiplier)
    norm = direction.norm(dim=(-2, -1))
    if lane_equalize:
        # Optional ablation: equalization can flatten confidence across lanes,
        # even when the matrix's total soft norm is preserved.
        dim = -1 if gradient.shape[-2] >= gradient.shape[-1] else -2
        second.lerp_(direction.square().mean(dim, keepdim=True), 1 - beta2)
        adapted = direction * second.clamp_min(1e-10).rsqrt()
        direction = adapted * norm[..., None, None] / adapted.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
    p.sub_(lr * scale * direction)
    noise_ratio = covariance.diagonal(dim1=-2, dim2=-1).sum(-1) / lookahead.square().sum((-2, -1)).clamp_min(1e-20)
    return torch.stack((noise_ratio, norm, direction.norm(dim=(-2, -1))), dim=-1)


class MatrixOptimizer(torch.optim.Optimizer):
    """Compiled, checkpoint-safe matrix banks with externally measured noise."""
    def __init__(self, params, lr=.005, momentum=.85, beta2=.9, noise_multiplier=1.0, lane_equalize=False):
        if not (0 <= momentum < 1 and 0 <= beta2 < 1 and np.isfinite(noise_multiplier) and noise_multiplier >= 0):
            raise ValueError("invalid momentum, beta2 or noise multiplier")
        super().__init__(params, dict(lr=lr, momentum=momentum, beta2=beta2, noise_multiplier=noise_multiplier,
                                     lane_equalize=lane_equalize, epoch=0))
        self.update = torch.compile(matrix_update, fullgraph=True, dynamic=False,
                                    options={"triton.cudagraphs": False})
        self._rebuild_stacks()

    def _rebuild_stacks(self):
        self.stacks, self.last_metrics, self.host_coefficients = [], [], []
        for group in self.param_groups:
            if not group["params"]:
                raise ValueError("matrix optimizer groups must be nonempty")
            first = group["params"][0]
            group["lr"] = torch.as_tensor(group["lr"], device=first.device, dtype=torch.float32).clone()
            group["coefficients"] = torch.empty(4, device=first.device)
            self.host_coefficients.append(torch.empty(4, pin_memory=True))
            by_shape = {}
            for parameter in group["params"]:
                if parameter.ndim != 2 or parameter.dtype != torch.float32 or parameter.device.type != "cuda":
                    raise ValueError("noise polar requires FP32 CUDA matrices")
                by_shape.setdefault(parameter.shape, []).append(parameter)
            for shape, parameters in by_shape.items():
                state = self.state[parameters[0]]
                lanes = (len(parameters), shape[0], 1) if shape[0] >= shape[1] else (len(parameters), 1, shape[1])
                if not state:
                    state["velocity"] = torch.zeros(len(parameters), *shape, device=first.device)
                    state["second"] = torch.zeros(lanes, device=first.device)
                    rank = min(shape)
                    state["covariance_moment"] = torch.zeros(len(parameters), rank, rank, device=first.device)
                    state["covariance_prior"] = torch.zeros_like(state["covariance_moment"])
                covariance_shape = (len(parameters), min(shape), min(shape))
                if (state["velocity"].shape != (len(parameters), *shape) or state["second"].shape != lanes
                        or state["covariance_moment"].shape != covariance_shape
                        or state["covariance_prior"].shape != covariance_shape):
                    raise ValueError("incompatible covariance-polar checkpoint")
                self.stacks.append((group, parameters, state, max(1., shape[0] / shape[1]) ** .5))
                self.last_metrics.append(torch.zeros(len(parameters), 3, device=first.device))

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._rebuild_stacks()
        return result

    @torch.no_grad()
    def set_lr(self, lr):
        for group in self.param_groups:
            group["lr"].fill_(lr)

    @torch.no_grad()
    def begin_rollout(self):
        for group in self.param_groups:
            group["epoch"] = 0
        for _, _, state, _ in self.stacks:
            state["covariance_prior"].copy_(state["covariance_moment"])

    @torch.no_grad()
    def step(self, noises):
        for group, staging in zip(self.param_groups, self.host_coefficients):
            group["epoch"] += 1
            k, beta = group["epoch"], group["momentum"]
            # Treat each rollout's fixed noise as fully correlated across its
            # optimization epochs, and different rollouts as independent. This
            # working model ignores parameter drift and changes in clip scale.
            staging[0], staging[1] = beta ** (2 * k), (1 - beta ** k) ** 2
            staging[2], staging[3] = beta ** (2 * (k + 1)), (1 - beta ** (k + 1)) ** 2
            group["coefficients"].copy_(staging)
        for index, (group, parameters, state, scale) in enumerate(self.stacks):
            if any(parameter.grad is None for parameter in parameters):
                raise RuntimeError("all matrix parameters require gradients")
            stacked = torch.stack(parameters)
            metrics = self.update(stacked, state["velocity"], state["second"],
                                  state["covariance_moment"], state["covariance_prior"],
                                  torch.stack([parameter.grad for parameter in parameters]),
                                  torch.stack([noises[parameter] for parameter in parameters]),
                                  group["coefficients"], group["lr"], group["momentum"], group["beta2"], scale,
                                  group["noise_multiplier"], group["lane_equalize"])
            self.last_metrics[index].copy_(metrics)
            torch._foreach_copy_(parameters, list(stacked.unbind(0)))


def make_optimizers(agent, args):
    matrices = [layer.weight for stage in (agent.actor.first, agent.actor.second)
                for layer in (stage[0].gate, stage[0].up, stage[0].down)]
    matrix_ids = {id(parameter) for parameter in matrices}
    auxiliary = [parameter for parameter in agent.parameters() if id(parameter) not in matrix_ids]
    return (MatrixOptimizer(matrices, lr=args.normuon_lr, momentum=args.normuon_momentum,
                            beta2=args.normuon_beta2, noise_multiplier=args.noise_multiplier,
                            lane_equalize=args.lane_equalize),
            optim.Adam(auxiliary, lr=args.learning_rate, eps=1e-5, fused=True))


def environment_folds(num_steps, num_envs, generator, device):
    """Each entire environment trajectory stays in one randomized equal fold."""
    if num_envs < 2 or num_envs % 2:
        raise ValueError("equal trajectory folds require an even number of environments")
    order = torch.randperm(num_envs, generator=generator, device=device)
    times = torch.arange(num_steps, device=device)[:, None] * num_envs
    return tuple((times + environments[None, :]).flatten()
                 for environments in order.chunk(2))


def fold_gradients(loss_fn, batch, folds, parameters, matrices):
    """Two equally weighted means reproduce the complete raw-advantage loss.

    The half-gradient difference estimates noise of their mean, assuming
    independent folds. Temporal dependence is preserved inside each fold.
    """
    parameters = tuple(parameters)
    first = None
    metric_mean = None
    noises = {}
    for fold in folds:
        for parameter in parameters:
            parameter.grad = None
        loss, metrics = loss_fn(*(field[fold] for field in batch))
        metrics = metrics.detach().clone()
        loss.backward()
        if first is None:
            first = tuple(parameter.grad.detach().clone() for parameter in parameters)
            metric_mean = .5 * metrics
        else:
            metric_mean.add_(metrics, alpha=.5)
            for parameter, first_gradient in zip(parameters, first):
                if id(parameter) in matrices:
                    noises[parameter] = (first_gradient - parameter.grad) * .5
                parameter.grad.add_(first_gradient).mul_(.5)
    return metric_mean, noises


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
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0024
    """the learning rate of the optimizer"""
    normuon_network: str = "actor"
    """which hidden matrices use NorMuon: actor, critic, or both"""
    normuon_lr: float = 0.005
    """hidden-matrix LR; soft-polar steps use sqrt(max(1, out/in)) scaling"""
    normuon_momentum: float = 0.85
    normuon_beta2: float = 0.9
    noise_multiplier: float = 1.0
    """multiplier of the measured Nesterov noise covariance"""
    lane_equalize: bool = False
    """optional long-axis RMS lanes; can flatten confidence, disabled by default"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """lower policy ratio clipping coefficient, also the value clipping radius"""
    clip_coef_upper: float = 0.2
    """upper policy ratio clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
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
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def bootstrap_observations(next_obs, truncations, infos):
    """Replace autoreset observations with final observations at time limits."""
    bootstrap_obs = np.array(next_obs, copy=True)
    truncations = np.asarray(truncations, dtype=bool)
    if not np.any(truncations):
        return bootstrap_obs

    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None:
        raise RuntimeError("truncated transition missing infos['final_observation']")

    for env_idx in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_idx]:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        final_observation = final_observations[env_idx]
        if final_observation is None:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        bootstrap_obs[env_idx] = final_observation
    return bootstrap_obs


def compute_gae(
    rewards,
    values,
    terminations,
    truncations,
    truncation_bootstrap_values,
    rollout_tail_value,
    gamma,
    gae_lambda,
):
    """Compute GAE with distinct bootstrap and reset-boundary semantics."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rollout_tail_value)
    for t in reversed(range(rewards.shape[0])):
        ordinary_next_value = rollout_tail_value if t == rewards.shape[0] - 1 else values[t + 1]
        next_value = torch.where(
            truncations[t].bool(),
            truncation_bootstrap_values[t],
            ordinary_next_value,
        )
        bootstrap_nonterminal = 1.0 - terminations[t]
        trace_nonterminal = 1.0 - torch.maximum(terminations[t], truncations[t])
        delta = rewards[t] + gamma * bootstrap_nonterminal * next_value - values[t]
        last_advantage = delta + gamma * gae_lambda * trace_nonterminal * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualMLP(nn.Module):
    """Unit residual directions with variance-restored branches and readouts."""

    def __init__(self, observation_dim, output_dim, output_std):
        super().__init__()

        def stage(in_dim):
            return nn.Sequential(init_situglu_branch(SiTUGLUBranch(in_dim, 64)))

        self.first = stage(observation_dim)
        self.second = stage(64)
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))
        self.readout_gain = nn.Parameter(torch.ones(output_dim))

    def forward(self, x):
        h = F.normalize(self.first(x), p=2, dim=-1)
        branch = F.normalize(self.second(8.0 * h), p=2, dim=-1)
        return self.readout_gain * self.head(8.0 * F.normalize(h + branch, p=2, dim=-1))


class ResidualHostMirror:
    """Compose fused FP32 stage mirrors without changing shared implementations.

    Stage outputs are permanent buffers; copy the first before calling the
    second, and add into our own permanent buffer before evaluating the head.
    """

    def __init__(self, actor, num_envs):
        self.first = make_host_mirror(actor.first, num_envs)
        self.second = make_host_mirror(actor.second, num_envs)
        self.head = make_host_mirror(actor.head, num_envs)
        self.hidden = np.empty((num_envs, 64), dtype=np.float32)
        self.branch = np.empty_like(self.hidden)
        self.squared = np.empty_like(self.hidden)
        self.norm = np.empty((num_envs, 1), dtype=np.float32)
        self.actor = actor
        self.readout_gain = np.empty(actor.readout_gain.numel(), dtype=np.float32)
        self.refresh()

    def refresh(self):
        self.first.refresh()
        self.second.refresh()
        self.head.refresh()
        np.copyto(self.readout_gain, self.actor.readout_gain.detach().cpu().numpy())

    def _normalize(self, values):
        np.multiply(values, values, out=self.squared)
        np.sum(self.squared, axis=1, keepdims=True, out=self.norm)
        np.sqrt(self.norm, out=self.norm)
        # Match F.normalize's zero-vector definition, including its epsilon.
        np.maximum(self.norm, np.float32(1e-12), out=self.norm)
        np.divide(values, self.norm, out=values)

    def __call__(self, observations):
        np.copyto(self.hidden, self.first(observations))
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.branch)
        np.copyto(self.branch, self.second(self.branch))
        self._normalize(self.branch)
        np.add(self.hidden, self.branch, out=self.hidden)
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.hidden)
        output = self.head(self.hidden)
        np.multiply(output, self.readout_gain, out=output)
        return output


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
        self.critic = ResidualMLP(observation_dim, 1, output_std=1.0)
        self.actor = ResidualMLP(observation_dim, 2 * self.action_dim, output_std=0.01)

    @torch.no_grad()
    def normalize_matrices(self):
        """Match nGPT's matrix axes without changing optimizer moments or biases."""
        for trunk in (self.actor, self.critic):
            for stage in (trunk.first, trunk.second):
                branch = stage[0]
                for weight, dim in ((branch.gate.weight, 1),
                                    (branch.up.weight, 1),
                                    (branch.down.weight, 0)):
                    weight.div_(torch.linalg.vector_norm(weight, dim=dim, keepdim=True))
            weight = trunk.head[0].weight
            weight.div_(torch.linalg.vector_norm(weight, dim=1, keepdim=True))

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
        clipfrac = ((ratio < 1 - args.clip_coef) | (ratio > 1 + args.clip_coef_upper)).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef_upper)
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


def exact_policy_kl(agent, observations, old_alpha, old_beta):
    alpha, beta, _ = agent.get_policy_and_value(observations)
    old_sum, new_sum = old_alpha + old_beta, alpha + beta
    # Explicit Beta KL avoids distribution registry dispatch inside fullgraph.
    return (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(new_sum)
            - torch.lgamma(old_alpha) - torch.lgamma(old_beta) + torch.lgamma(old_sum)
            + (old_alpha - alpha) * torch.digamma(old_alpha)
            + (old_beta - beta) * torch.digamma(old_beta)
            + (new_sum - old_sum) * torch.digamma(old_sum)).sum(-1).mean()


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
    if args.normuon_network not in {"actor", "critic", "both"}:
        raise ValueError("normuon_network must be actor, critic, or both")
    if not (np.isfinite(args.learning_rate) and args.learning_rate > 0
            and np.isfinite(args.normuon_lr) and args.normuon_lr > 0):
        raise ValueError("optimizer learning rates must be finite and positive")
    if not (0 <= args.normuon_momentum < 1 and 0 <= args.normuon_beta2 < 1):
        raise ValueError("NorMuon momentum and beta2 must lie in [0, 1)")
    if args.normuon_network != "actor" or args.norm_adv or args.num_minibatches != 1:
        raise ValueError("noise polar requires actor matrices, raw advantages and one full minibatch")
    if args.num_envs < 2 or args.num_envs % 2:
        raise ValueError("noise polar requires an even number of environments")
    if not np.isfinite(args.noise_multiplier) or args.noise_multiplier < 0:
        raise ValueError("noise_multiplier must be finite and nonnegative")
    if not args.compile:
        raise ValueError("noise polar requires compiled CUDA updates")
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
        writer.add_text("architecture", "residual SiTU-GLU; unit-L2 matrices and residual directions; sqrt(width) branch/readout scale; learned output gains; no learned mixing")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        matrix_optimizer, optimizer = make_optimizers(agent, args)
        actor_parameters, critic_parameters = tuple(agent.actor.parameters()), tuple(agent.critic.parameters())
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native), alpha, beta

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        policy_kl = lambda obs, alpha, beta: exact_policy_kl(agent, obs, alpha, beta)
        if args.compile:
            policy_kl = torch.compile(policy_kl, fullgraph=True, options={"triton.cudagraphs": False})
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = ResidualHostMirror(agent.actor, args.num_envs)
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
        # Trajectory folds must not consume the policy sampler's random stream.
        fold_generator = torch.Generator(device=device).manual_seed(args.seed + 17041)
        all_parameters = tuple(agent.parameters())
        matrix_ids = {id(p) for group in matrix_optimizer.param_groups for p in group["params"]}
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
        grad_norms = torch.empty((max_updates, 2), device=device)
        epoch_kls = torch.empty(args.update_epochs, device=device)
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

        captured_steps = set()
        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                fraction = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
                matrix_optimizer.set_lr(fraction * args.normuon_lr)
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
                b_values, b_logprobs, old_alpha, old_beta = rollout_statistics(b_obs, b_native)
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
            for capture_step in (4000000, args.num_iterations * args.batch_size + horizon * args.num_envs):
                if global_step >= capture_step and capture_step not in captured_steps:
                    torch.save({
                        "model": agent.state_dict(), "args": vars(args),
                        "global_step": global_step, "optimizer": optimizer.state_dict(),
                        "matrix_optimizer": matrix_optimizer.state_dict(),
                        "batch": {"obs": b_obs, "native": b_native, "logprob": b_logprobs,
                                  "advantages": b_advantages, "returns": b_returns, "values": b_values},
                        "obs_norm": {name: getattr(obs_norm, name) for name in ("means", "variances", "counts")},
                        "rew_norm": {name: getattr(rew_norm, name) for name in ("returns", "means", "variances", "counts")},
                    }, f"runs/{run_name}/diagnostic_snapshot_{global_step}.pt")
                    captured_steps.add(capture_step)
            folds = environment_folds(args.num_steps, args.num_envs, fold_generator, device)
            matrix_optimizer.begin_rollout()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    if epoch == 0:
                        metrics, rollout_noises = fold_gradients(
                            loss_model, (b_obs, b_native, b_logprobs, b_advantages, b_returns, b_values),
                            folds, all_parameters, matrix_ids)
                    else:
                        # Repeated PPO epochs are correlated optimization of
                        # one rollout, not independent noise observations.
                        agent.zero_grad(set_to_none=True)
                        loss, metrics = loss_model(b_obs, b_native, b_logprobs,
                                                   b_advantages, b_returns, b_values)
                        loss.backward()
                    actor_norm = nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm, foreach=True)
                    grad_norms[updates, 0] = actor_norm
                    grad_norms[updates, 1] = nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
                    factor = (args.max_grad_norm / (actor_norm + 1e-6)).clamp(max=1.)
                    # Freeze raw noise throughout the rollout; convert it to
                    # the same units as this epoch's clipped mean gradient.
                    noises = {parameter: noise * factor for parameter, noise in rollout_noises.items()}
                    matrix_optimizer.step(noises)
                    optimizer.step()
                    normalize_matrices()
                    update_metrics[updates].copy_(metrics)
                    updates += 1
                    with torch.no_grad():
                        epoch_kls[epoch] = policy_kl(b_obs, old_alpha, old_beta)
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            with torch.no_grad():
                current_alpha, current_beta, _ = agent.get_policy_and_value(b_obs)
                old_concentration = old_alpha + old_beta
                concentration = current_alpha + current_beta
                policy_metrics = {
                    "policy/concentration": concentration.mean(),
                    "policy/mean_step_rms": ((current_alpha / concentration
                        - old_alpha / old_concentration).square().mean()).sqrt(),
                    "policy/log_concentration_step": (concentration.log()
                        - old_concentration.log()).mean(),
                }
                for bank, metrics in enumerate(matrix_optimizer.last_metrics):
                    policy_metrics[f"matrix_last/bank{bank}_noise_to_signal"] = metrics[:, 0].mean()
                    policy_metrics[f"matrix_last/bank{bank}_soft_polar_norm"] = metrics[:, 1].mean()
                    policy_metrics[f"matrix_last/bank{bank}_effective_polar_norm"] = metrics[:, 2].mean()
            last = update_metrics[updates - 1]
            logged = gather_metrics({
                **policy_metrics,
                "losses/post_first_epoch_exact_kl": epoch_kls[0],
                "losses/post_final_epoch_exact_kl": epoch_kls[epoch],
                "losses/mean_preupdate_kl": update_metrics[:updates, 4].mean(),
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "grad/actor_preclip_norm": grad_norms[:updates, 0].mean(),
                "grad/critic_preclip_norm": grad_norms[:updates, 1].mean(),
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/normuon_learning_rate", args.normuon_lr * (fraction if args.anneal_lr else 1.0), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        torch.save({"model": agent.state_dict(), "args": vars(args), "global_step": global_step},
                   f"runs/{run_name}/{args.exp_name}.pt")
        transfer.close()
        envs.close()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            from cleanrl_utils.evals.ppo_eval import evaluate
            episodic_returns = evaluate(
                model_path, make_env, args.env_id, eval_episodes=10,
                run_name=f"{run_name}-eval", Model=Agent, device=device, gamma=args.gamma,
            )
            for index, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, index)
            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub
                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

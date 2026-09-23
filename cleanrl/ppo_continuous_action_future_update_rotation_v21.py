# Finite-horizon, model-free learning-to-learn PPO with GAE and categorical V(s).
# A Gaussian controller samples three actor-update rotation angles, held for H
# learning rounds. Only subsequent real rollout rewards train that controller:
# the first K <= H outcomes define the return; every arm waits H outcomes before
# another controller update. K changes credit horizon, not angle persistence.
# Conditional on the window-start learner/environment state, the angle score
# credits all ensuing optimizer, critic, data and environment effects. There is
# no dynamics/Q model, pathwise meta-gradient, or full-training optimality claim.
# Rotations preserve each Adam displacement's norm before the unchanged matrix
# projection; ordinary Adam moments and the base PPO/GAE/critic loss are intact.
# No controller clipping, KL caps, learned learning rates or same-batch objective.
import copy
import json
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
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
from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport
from cleanrl.shared.two_hot import DreamerTwoHotSupport
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, make_raw_continuous_env

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
    """also export the agent state dict; a full final learning checkpoint is always saved"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0024
    """the learning rate of the optimizer"""
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
    """lower policy ratio clipping coefficient"""
    clip_coef_upper: float = 0.28
    """upper policy ratio clipping coefficient"""
    value_target: Literal["hlgauss", "twohot"] = "hlgauss"
    """raw-mean-matched Gaussian labels or Dreamer raw-space twohot"""
    value_atoms: int = 101
    """number of categorical critic atoms"""
    value_min: float = -20000.0
    """lower support bound in raw discounted-return units"""
    value_max: float = 20000.0
    """upper support bound in raw discounted-return units"""
    value_sigma_bins: float = 0.5
    """Gaussian smoothing in bin widths; unused by twohot"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    variance_gain: bool = False
    """multiply CE by detached predicted raw variance; false uses plain CE"""
    critic_gain_coordinates: Literal["direct", "relative"] = "direct"
    """direct gamma=0.01, or dimensionless u=1 with effective gamma=0.01*u"""
    grad_clip: Literal["global", "readout", "none"] = "none"
    """clip all gradients, only the critic readout gain, or none"""

    # The control window and credited horizon differ for matched H1/H4 ablations.
    meta_mode: Literal["learn", "random", "none"] = "learn"
    """learned Gaussian angles, frozen zero-mean random angles, or exact Adam identity"""
    meta_horizon: int = 4
    """number of PPO rollout updates sharing one sampled angle vector"""
    meta_credit_horizon: int = 4
    """number of initial post-action rollouts credited, at most meta_horizon"""
    meta_std: float = 0.15
    """fixed Gaussian angular exploration standard deviation, in radians"""
    meta_lr: float = 0.003
    """Adam learning rate for the Gaussian controller, not the PPO learner"""
    meta_reward_scale: float = 10.0
    """fixed divisor of the future-return-minus-prewindow-reward score weight"""

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


def validate_meta_args(args):
    if args.meta_mode not in {"learn", "random", "none"}:
        raise ValueError("meta_mode must be learn, random, or none")
    if isinstance(args.meta_horizon, bool) or not isinstance(args.meta_horizon, int) or args.meta_horizon <= 0:
        raise ValueError("meta_horizon must be a positive integer")
    if (isinstance(args.meta_credit_horizon, bool) or not isinstance(args.meta_credit_horizon, int)
            or not 1 <= args.meta_credit_horizon <= args.meta_horizon):
        raise ValueError("meta_credit_horizon must be an integer in [1, meta_horizon]")
    for name in ("meta_std", "meta_lr", "meta_reward_scale"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def rotate_displacement(displacement, descent, angle):
    """Rotate an Adam displacement toward raw descent without changing its norm.

    Modified Gram-Schmidt with one reorthogonalization removes FP32 cancellation
    along d. If no numerically meaningful second direction exists, use d itself.
    This is geometric degeneracy handling, not a gradient or angle magnitude cap.
    """
    if displacement.numel() <= 1:
        return displacement
    tiny = torch.finfo(displacement.dtype).tiny
    norm = torch.linalg.vector_norm(displacement)
    unit = displacement / norm.clamp_min(tiny)
    orthogonal = descent - (descent * unit).sum() * unit
    orthogonal = orthogonal - (orthogonal * unit).sum() * unit
    orthogonal_norm = torch.linalg.vector_norm(orthogonal)
    threshold = 32.0 * torch.finfo(displacement.dtype).eps * torch.linalg.vector_norm(descent)
    valid = (norm > tiny) & (orthogonal_norm > threshold)
    rotated = angle.cos() * displacement + angle.sin() * norm * (
        orthogonal / orthogonal_norm.clamp_min(tiny)
    )
    return torch.where(valid & (angle != 0), rotated, displacement)


class UpdateRotation:
    """Capture raw actor gradients, then rotate actual Adam steps before projection.

    Fixed buffers and compiled mutation-only kernels avoid optimizer-path host
    synchronization. Scalar parameters cannot support a rotation and are skipped.
    The three groups are actor.first, actor.second, and all readout parameters.
    """

    def __init__(self, actor, compile: bool):
        named = tuple((name, parameter) for name, parameter in actor.named_parameters()
                      if parameter.requires_grad and parameter.numel() > 1)
        self.parameters = tuple(parameter for _, parameter in named)
        self.groups = tuple(0 if name.startswith("first.") else 1 if name.startswith("second.") else 2
                            for name, _ in named)
        reference = next(actor.parameters())
        self.before = tuple(torch.empty_like(parameter) for parameter in self.parameters)
        self.descent = tuple(torch.empty_like(parameter) for parameter in self.parameters)
        self.stats = reference.new_tensor((0.0, 1.0))
        self.capture = self._capture
        self.apply = self._apply
        if compile:
            # Input gradient storage can change after zero_grad(set_to_none=True).
            # Inductor fuses operations; no CUDA graph owns our persistent buffers.
            options = {"triton.cudagraphs": False}
            self.capture = torch.compile(self._capture, fullgraph=True, options=options)
            self.apply = torch.compile(self._apply, fullgraph=True, options=options)

    @torch.no_grad()
    def _capture(self):
        for parameter, before, descent in zip(self.parameters, self.before, self.descent):
            if parameter.grad is None:
                raise RuntimeError("capture must follow backward on every actor parameter")
            before.copy_(parameter)
            descent.copy_(-parameter.grad)

    @torch.no_grad()
    def _apply(self, angles):
        errors, cosines = [], []
        for parameter, before, descent, group in zip(
            self.parameters, self.before, self.descent, self.groups,
        ):
            displacement = parameter - before
            angle = angles[group]
            rotated = rotate_displacement(displacement, descent, angle)
            # Reconstructing before + (after - before) need not be bitwise exact.
            # Preserve the actual Adam result explicitly for every zero angle.
            parameter.copy_(torch.where(angle == 0, parameter, before + rotated))
            applied = parameter - before
            norm = torch.linalg.vector_norm(displacement)
            applied_norm = torch.linalg.vector_norm(applied)
            tiny = torch.finfo(parameter.dtype).tiny
            errors.append((applied_norm - norm).abs() / norm.clamp_min(tiny))
            cosine = ((displacement / norm.clamp_min(tiny)) *
                      (applied / applied_norm.clamp_min(tiny))).sum()
            cosines.append(torch.where((norm > tiny) & (applied_norm > tiny), cosine,
                                       torch.ones_like(cosine)))
        if errors:
            self.stats[0].copy_(torch.stack(errors).max())
            self.stats[1].copy_(torch.stack(cosines).mean())


class FutureUpdateController:
    """One detached Gaussian action and one causal score update per full window.

    observe is called after a complete rollout and BEFORE its PPO update. The
    first observation is the action-independent baseline, never a credited
    outcome. The next H observations follow H controlled PPO updates. No partial
    final window is trained; its full pending state is retained in checkpoints.
    """

    def __init__(self, args, device):
        validate_meta_args(args)
        self.device = torch.device(device)
        self.mode = args.meta_mode
        self.horizon = args.meta_horizon
        self.credit_horizon = args.meta_credit_horizon
        self.std = args.meta_std
        self.reward_scale = args.meta_reward_scale
        self.weights = nn.Parameter(torch.zeros((3, 4), device=self.device))
        self.optimizer = optim.Adam([self.weights], lr=args.meta_lr,
                                    fused=self.device.type == "cuda")
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed + 1_000_003)
        self._angles = torch.zeros(3, device=self.device)
        self.mean = torch.zeros_like(self._angles)
        self.features = torch.zeros(4, device=self.device)
        self.baseline = torch.zeros((), device=self.device)
        self.previous_reward = torch.zeros_like(self.baseline)
        self.current_reward = torch.zeros_like(self.baseline)
        self.window_rewards = torch.zeros(self.horizon, device=self.device)
        self.last_completed_rewards = torch.zeros_like(self.window_rewards)
        # Last completed objective, advantage, loss, score norm, full-window mean.
        self.last_metrics = torch.zeros(5, device=self.device)
        self.pending = False
        self.has_previous_reward = False
        self.outcomes = 0
        self.windows = 0
        self.updates = 0

    @property
    def angles(self):
        return self._angles

    def _complete_window(self):
        objective = self.window_rewards[:self.credit_horizon].mean()
        advantage = ((objective - self.baseline) / self.reward_scale).detach()
        mean = self.weights @ self.features
        # Detaching the sampled action is essential: reparameterizing this score
        # would cancel the mean gradient instead of crediting the future return.
        standardized = (self._angles.detach() - mean) / self.std
        log_prob = (-0.5 * standardized.square() -
                    math.log(self.std) - 0.5 * math.log(2.0 * math.pi)).sum()
        loss = -advantage * log_prob
        with torch.no_grad():
            score = ((self._angles - mean.detach()) / self.std**2).unsqueeze(1) * self.features
            score_norm = torch.linalg.vector_norm(score)
            self.last_metrics.copy_(torch.stack((
                objective, advantage, loss.detach(), score_norm, self.window_rewards.mean(),
            )))
            self.last_completed_rewards.copy_(self.window_rewards)
        if self.mode == "learn":
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.updates += 1
        self.windows += 1
        self.pending = False

    @torch.no_grad()
    def _start_window(self, progress):
        self.features[0] = 1.0
        self.features[1] = progress
        self.features[2].copy_((self.current_reward / 10.0).tanh())
        previous = self.previous_reward if self.has_previous_reward else self.current_reward
        self.features[3].copy_(((self.current_reward - previous) / 10.0).tanh())
        self.baseline.copy_(self.current_reward)
        if self.mode == "learn":
            self.mean.copy_(self.weights @ self.features)
        else:
            self.mean.zero_()
        if self.mode == "none":
            self._angles.zero_()
        else:
            self._angles.copy_(self.mean + self.std * torch.randn(
                3, device=self.device, generator=self.generator,
            ))
        self.window_rewards.zero_()
        self.outcomes = 0
        self.pending = True

    def observe(self, reward: float, progress: float) -> dict[str, float]:
        if not math.isfinite(reward) or not math.isfinite(progress):
            raise ValueError("controller observations must be finite")
        with torch.no_grad():
            self.current_reward.fill_(reward)
            if self.pending:
                self.window_rewards[self.outcomes].copy_(self.current_reward)
                self.outcomes += 1
        if self.pending and self.outcomes == self.horizon:
            self._complete_window()
        if not self.pending:
            self._start_window(progress)
        with torch.no_grad():
            self.previous_reward.copy_(self.current_reward)
            self.has_previous_reward = True
            # One device-to-host metrics transfer at the rollout boundary only.
            scalars = torch.cat((self.mean, self._angles, self.last_metrics,
                                 self.baseline.reshape(1))).cpu().tolist()
        names = ("mean_first", "mean_second", "mean_readout",
                 "angle_first", "angle_second", "angle_readout",
                 "objective", "advantage", "loss", "score_norm", "full_window_reward", "baseline")
        metrics = {f"meta/{name}": float(value) for name, value in zip(names, scalars)}
        metrics.update({
            "meta/rollout_reward": float(reward),
            "meta/windows": float(self.windows), "meta/updates": float(self.updates),
            "meta/pending_outcomes": float(self.outcomes),
            "meta/window_horizon": float(self.horizon),
            "meta/credit_horizon": float(self.credit_horizon),
        })
        return metrics

    def state_dict(self):
        """Snapshot the controller, including a partially observed causal window."""
        tensors = ("mean", "features", "baseline", "previous_reward", "current_reward",
                   "window_rewards", "last_completed_rewards", "last_metrics")
        return {
            "version": 1, "mode": self.mode, "horizon": self.horizon,
            "credit_horizon": self.credit_horizon, "std": self.std,
            "reward_scale": self.reward_scale,
            "weights": self.weights.detach().clone(),
            "angles": self._angles.clone(),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "generator_state": self.generator.get_state().clone(),
            "pending": self.pending, "has_previous_reward": self.has_previous_reward,
            "outcomes": self.outcomes, "windows": self.windows, "updates": self.updates,
            **{name: getattr(self, name).clone() for name in tensors},
        }

    def load_state_dict(self, state):
        for name, expected in (("version", 1), ("mode", self.mode), ("horizon", self.horizon),
                               ("credit_horizon", self.credit_horizon), ("std", self.std),
                               ("reward_scale", self.reward_scale)):
            if state[name] != expected:
                raise ValueError(f"incompatible controller checkpoint {name}: {state[name]!r}")
        if not 0 <= state["outcomes"] < self.horizon:
            raise ValueError("checkpoint pending outcomes must be smaller than the window")
        with torch.no_grad():
            self.weights.copy_(state["weights"])
            self._angles.copy_(state["angles"])
            for name in ("mean", "features", "baseline", "previous_reward", "current_reward",
                         "window_rewards", "last_completed_rewards", "last_metrics"):
                getattr(self, name).copy_(state[name])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator_state"].cpu())
        for name in ("pending", "has_previous_reward", "outcomes", "windows", "updates"):
            setattr(self, name, state[name])


class ObservationNormalizedEvaluationEnv(gym.ObservationWrapper):
    """Use the vector observation normalizer while preserving raw rewards."""
    def __init__(self, env):
        super().__init__(env)
        self.obs_norm = VectorObsNorm(1, env.observation_space.shape)

    def observation(self, observation):
        return self.obs_norm.normalize(np.asarray(observation)[None])[0]


# Public evaluation and historical GAE compatibility helpers; not the training path.
def make_env(env_id, idx, capture_video, run_name, gamma):
    raw_factory = make_raw_continuous_env(env_id, idx, capture_video, run_name)

    def thunk():
        return ObservationNormalizedEvaluationEnv(raw_factory())

    return thunk




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
        self.register_buffer("readout_scale", torch.ones(()))

    def forward(self, x):
        h = F.normalize(self.first(x), p=2, dim=-1)
        branch = F.normalize(self.second(8.0 * h), p=2, dim=-1)
        gain = self.readout_gain * self.readout_scale
        return gain * self.head(8.0 * F.normalize(h + branch, p=2, dim=-1))


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
        gain = self.actor.readout_gain * self.actor.readout_scale
        np.copyto(self.readout_gain, gain.detach().cpu().numpy())

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
    def __init__(self, envs, args=None):
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
        args = Args() if args is None else args
        # Build the original pair first to preserve v7's actor RNG initialization.
        # Only then replace the critic readout; this is a categorical-head ablation.
        self.critic.head = nn.Sequential(layer_init(nn.Linear(64, args.value_atoms), std=1.0))
        self.critic.readout_gain = nn.Parameter(torch.full((args.value_atoms,), 0.01))
        if args.critic_gain_coordinates == "relative":
            # Preserve the forward function; change only the optimizer coordinates.
            with torch.no_grad():
                self.critic.readout_gain.fill_(1.0)
                self.critic.readout_scale.fill_(0.01)
        # Both target encodings share registered raw nodes and the same decoder.
        self.histogram = DreamerTwoHotSupport(args.value_atoms, args.value_max)

    @torch.no_grad()
    def normalize_matrices(self):
        """Match nGPT's matrix axes without changing Adam moments or biases."""
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
        return self.histogram.to_scalar(self.critic(x)).unsqueeze(-1)

    def get_policy_and_value(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.get_value(x)

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


def make_target_projector(args, histogram):
    """Build training-only target geometry after the model moves to CUDA."""
    if args.value_target == "twohot":
        return histogram.project
    if args.value_target != "hlgauss":
        raise ValueError("value_target must be hlgauss or twohot")
    bound = math.log1p(args.value_max)
    gaussian = Dreamer3BucketHLGaussSupport(
        args.value_atoms, -bound, bound, args.value_sigma_bins, histogram.support.device,
    )
    # The solver and model must use exactly the same FP32 raw nodes.
    gaussian.support = histogram.support
    return gaussian.project_moment_matched


def categorical_value_terms(logits, target_probs, histogram, variance_gain=True):
    """Plain CE or detached raw-variance-scaled CE, with identical targets."""
    log_probs = logits.log_softmax(dim=-1)
    raw_ce = -(target_probs.detach() * log_probs).sum(dim=-1).mean()
    with torch.no_grad():
        probabilities = log_probs.detach().exp()
        mean = (probabilities * histogram.support).sum(dim=-1, keepdim=True)
        raw_gain = (probabilities * (histogram.support - mean).square()).sum(dim=-1).mean()
        gain = raw_gain if variance_gain else torch.ones_like(raw_gain)
    return raw_ce * gain, raw_ce, gain, raw_gain


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, target_probs, args, old_alpha, old_beta):
    """Pure clipped PPO loss on native Beta samples; no inverse action scaling."""
    alpha, beta = (F.softplus(agent.actor(observations)) + 1.0).chunk(2, dim=-1)
    logits = agent.critic(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio < 1 - args.clip_coef) | (ratio > 1 + args.clip_coef_upper)).float().mean()
        old_sum, new_sum = old_alpha + old_beta, alpha + beta
        exact_kl = (
            alpha.lgamma() + beta.lgamma() + old_sum.lgamma()
            - old_alpha.lgamma() - old_beta.lgamma() - new_sum.lgamma()
            + (old_alpha - alpha) * old_alpha.digamma()
            + (old_beta - beta) * old_beta.digamma()
            + (new_sum - old_sum) * old_sum.digamma()
        ).sum(-1).mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef_upper)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    v_loss, raw_ce, value_gain, raw_gain = categorical_value_terms(
        logits, target_probs, agent.histogram, args.variance_gain,
    )
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac, raw_ce.detach(), value_gain, raw_gain, exact_kl))
    return loss, metrics


def apply_gradient_clipping(parameters, readout_parameters, max_norm, mode):
    """Clipping the gain must not modify any actor or other critic gradient."""
    if mode == "global":
        return nn.utils.clip_grad_norm_(parameters, max_norm)
    if mode == "readout":
        return nn.utils.clip_grad_norm_(readout_parameters, max_norm)
    if mode == "none":
        return None
    raise ValueError("grad_clip must be global, readout, or none")


def validate_args(args):
    validate_meta_args(args)
    if args.grad_clip != "none":
        raise ValueError("update rotation requires unclipped raw gradients: grad_clip must be none")
    if args.value_atoms < 3 or args.value_atoms % 2 != 1:
        raise ValueError("mean-preserving support requires an odd atom count >= 3")
    if not math.isfinite(args.value_max) or args.value_max <= 0 or args.value_min != -args.value_max:
        raise ValueError("raw support bounds must be finite, positive-radius and symmetric")
    if not math.isfinite(args.value_sigma_bins) or args.value_sigma_bins <= 0:
        raise ValueError("value_sigma_bins must be finite and positive")
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
        metrics_file = resources.enter_context(open(f"runs/{run_name}/metrics.jsonl", "w", buffering=1))
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); FP32; native-action storage; host actor mirror")
        writer.add_text("architecture", f"original residual SiTU-GLU trunks and matrix projection; raw K{args.value_atoms}/sigma{args.value_sigma_bins} mean-preserving labels; variance multiplier={args.variance_gain}; critic gain coordinates={args.critic_gain_coordinates}; gradient clipping={args.grad_clip}; no value gate, gain EMA or KL cap")
        writer.add_text("meta_objective",
                        f"conditional finite-horizon Gaussian score; mode={args.meta_mode}; "
                        f"angles held {args.meta_horizon} PPO updates; first {args.meta_credit_horizon} "
                        "subsequent raw rollout means credited; frozen prewindow baseline; no pathwise claim")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        parameters = tuple(agent.parameters())
        optimizer = optim.Adam(parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        controller = FutureUpdateController(args, device)
        rotation = None if args.meta_mode == "none" else UpdateRotation(agent.actor, args.compile)
        actor_parameters = tuple(agent.actor.parameters())
        readout_parameters = (agent.critic.readout_gain,)
        critic_other_parameters = tuple(p for name, p in agent.critic.named_parameters() if name != "readout_gain")
        previous_readout = torch.empty_like(agent.critic.readout_gain)
        value_model = agent.get_value
        project_targets = make_target_projector(args, agent.histogram)

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native), alpha, beta

        def loss_model(observations, native, old_logprobs, advantages, target_probs, old_alpha, old_beta):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, target_probs, args, old_alpha, old_beta)

        @torch.no_grad()
        def prepare_targets(returns):
            labels = project_targets(returns)
            decoded = agent.histogram.probs_to_scalar(labels)
            saturation = ((returns < args.value_min) | (returns > args.value_max)).float().mean()
            label_error = (decoded - returns).abs().mean()
            label_bias = (decoded - returns).mean()
            projection_error = (decoded - returns.clamp(args.value_min, args.value_max)).abs().mean()
            entropy = -(labels * labels.clamp_min(1e-30).log()).sum(-1).mean()
            return labels, saturation, label_error, label_bias, projection_error, entropy

        if args.compile:
            prepare_targets = torch.compile(prepare_targets, fullgraph=True,
                                            options={"triton.cudagraphs": False})
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
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 10), device=device)
        rotation_metrics = torch.empty((max_updates, 2), device=device) if rotation is not None else None
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=None,
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
            rollout_reward_sum = 0.0
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                    rollout_reward_sum += float(np.sum(raw_reward, dtype=np.float64))
                with timer.span("normalize_transfer", use_cuda=False):
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(step, raw_reward, terms, truncs, observations=obs_step, native_actions=native)
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

            with timer.span("controller"):
                meta_metrics = controller.observe(
                    rollout_reward_sum / args.batch_size, iteration / args.num_iterations,
                )

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs, b_alpha, b_beta = rollout_statistics(b_obs, b_native)
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
                b_target_probs, saturated, label_error, label_bias, projection_error, label_entropy = prepare_targets(b_returns)
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_target_probs[indices], b_alpha[indices], b_beta[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        selected_preclip_norm = apply_gradient_clipping(
                            parameters, readout_parameters, args.max_grad_norm, args.grad_clip,
                        )
                        with torch.no_grad():
                            previous_readout.copy_(agent.critic.readout_gain)
                        if rotation is not None:
                            rotation.capture()
                        optimizer.step()
                        if rotation is not None:
                            rotation.apply(controller.angles)
                            rotation_metrics[updates].copy_(rotation.stats)
                        normalize_matrices()
                        update_metrics[updates].copy_(metrics)
                        updates += 1

            last = update_metrics[updates - 1]
            with torch.no_grad():
                post_values = value_model(b_obs).flatten()
                post_error = post_values - b_returns
                actor_grad_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in actor_parameters]))
                readout_grad_norm = agent.critic.readout_gain.grad.norm()
                critic_other_grad_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in critic_other_parameters]))
                critic_grad_norm = torch.sqrt(critic_other_grad_norm.square() + readout_grad_norm.square())
                postclip_norm = torch.sqrt(actor_grad_norm.square() + critic_grad_norm.square())
                if args.grad_clip == "global":
                    total_preclip_norm = selected_preclip_norm
                elif args.grad_clip == "readout":
                    total_preclip_norm = torch.sqrt(
                        actor_grad_norm.square() + critic_other_grad_norm.square() + selected_preclip_norm.square()
                    )
                else:
                    total_preclip_norm = postclip_norm
                grad_norm = postclip_norm.clamp_min(1e-30)
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "critic/outside_support": saturated, "critic/label_error": label_error,
                "critic/target_mean": b_returns.mean(), "critic/target_scale": b_returns.std(),
                "critic/label_bias": label_bias, "critic/projection_mean_error": projection_error,
                "critic/label_entropy": label_entropy, "critic/excess_ce": last[6] - label_entropy,
                "critic/value_bias": post_error.mean(), "critic/decoded_mse": post_error.square().mean(),
                "critic/raw_ce": last[6], "critic/value_gain": last[7],
                "critic/value_gain_raw": last[8],
                "losses/exact_beta_kl_last": last[9],
                "losses/exact_beta_kl_max": update_metrics[:updates, 9].max(),
                "losses/sampled_kl_max": update_metrics[:updates, 4].max(),
                "grad/readout_postclip_norm": readout_grad_norm,
                "critic/effective_readout_gain_norm": (agent.critic.readout_gain * agent.critic.readout_scale).norm(),
                "critic/readout_relative_update": (agent.critic.readout_gain - previous_readout).norm() / previous_readout.norm(),
                "critic/preupdate_mse": b_advantages.square().mean(),
                "grad/total_preclip_norm": total_preclip_norm,
                "grad/actor_postclip_norm": actor_grad_norm, "grad/critic_postclip_norm": critic_grad_norm,
                "grad/actor_norm_fraction": actor_grad_norm / grad_norm,
                "grad/critic_norm_fraction": critic_grad_norm / grad_norm,
                "critic/postupdate_ev": explained_variance(post_values, b_returns),
                "critic/advantage_mean": b_advantages.mean(), "critic/advantage_std": b_advantages.std(),
                **({
                    "rotation/preprojection_norm_error": rotation_metrics[:updates, 0].max(),
                    "rotation/preprojection_cosine": rotation_metrics[:updates, 1].mean(),
                } if rotation is not None else {}),
            })
            logged.update(meta_metrics)
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            now = time.perf_counter()
            logged.update({
                "charts/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "charts/SPS": float(global_step / (now - start_time)),
                "charts/interval_SPS": float((global_step - interval_step) / (now - interval_start)),
            })
            for phase, timing in timer.summary().items():
                logged[f"timing/{phase}_s"] = float(timing["total_s"])
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            # Explained variance alone may be undefined for a constant return.
            record = {name: value if math.isfinite(value) else None for name, value in logged.items()}
            record.update(step=global_step, iteration=iteration)
            metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        checkpoint_path = f"runs/{run_name}/final_learning_checkpoint.pt"
        temporary_path = checkpoint_path + ".tmp"
        # Auditing state, not a claim of bitwise environment-level resume: live
        # MuJoCo state is not serialized by the shared vector environment.
        checkpoint = {
            "version": 1, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
            "controller": controller.state_dict(), "steps": global_step,
            "iteration": args.num_iterations, "args": asdict(args),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "policy_sampler_state": sampler.bit_generator.state,
            "shuffle_generator_state": shuffle_generator.get_state(),
            "observation_normalizer": {
                "means": obs_norm.means.copy(), "variances": obs_norm.variances.copy(),
                "counts": obs_norm.counts.copy(), "epsilon": obs_norm.epsilon, "clip": obs_norm.clip,
            },
            "next_observation": next_obs_np.copy(),
        }
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        print(f"learning checkpoint saved to {checkpoint_path}")
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

# MDL weight posteriors on the sepclip PPO control (awr/ppo_sepclip_control_v1).
# --posterior none is that control, line for line. mdl and svd replace every weight matrix of
# both networks by a factorised Gaussian posterior q(w) = N(mu, sigma^2) and add its
# description length to the loss as beta * KL(q || p) / N, N = transitions per rollout (the
# batch the K epochs fit). Extra epochs therefore leave the objective unchanged: a weight is
# worth its bits only if it explains enough of that batch.
#   mdl: Gaussian prior whose per-matrix variance is the empirical-Bayes optimum (Graves 2011),
#        so the KL is the two-part code length of the weights.
#   svd: sparse variational dropout (Molchanov 2017): log-uniform prior, additive
#        reparameterisation; weights with log alpha > 3 are pruned from the decoded model.
# Actor: one weight sample per env per noise segment (64 steps: 256 samples per rollout, so
# sigma sees more than 16 draws), recorded and replayed in the update
# (consistent dropout, Hausknecht & Wagener 2022), so the first ratio is exactly one and the
# data term is the PPO surrogate of the sampled policies, i.e. E_q[J]. Its KL is in units of
# normalised advantage (scaled by the rollout's std A), the weighting norm-adv PPO would imply.
# Critic: local reparameterisation in the loss, decoded mean for GAE; its MSE is a Gaussian
# likelihood whose noise variance is the pre-update (held-out) MSE on the rollout, so its KL
# weight is vf_coef * that MSE, fixed per rollout like N.
# Hypothesis: charging bits for batch-specific fit stops K-epoch overfitting, so returns hold
# or improve, and keep doing so as update_epochs grows.
# Run with --no-norm-adv --no-clip-vloss --clip-coef-upper 0.2 to match the 50M reference.
import functools
import math
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
from torch.distributions import Beta, kl_divergence
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch, situ_glu
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
POSTERIORS = ("none", "mdl", "svd")
# Molchanov et al. (2017) eq. 14: -KL ~= K1 sigmoid(K2 + K3 log alpha) - 0.5 log1p(1/alpha) - K1.
SVD_K1, SVD_K2, SVD_K3 = 0.63576, 1.87320, 1.48695
# Guards log(mu^2) at an exactly zero mean, as in the reference implementation.
LOG_MU2_EPS = 1e-16

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
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """lower policy ratio clipping coefficient, also the value clipping radius"""
    clip_coef_upper: float = 0.28
    """upper policy ratio clipping coefficient"""
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

    # Weight posterior (the method). `none` is the unmodified control.
    posterior: str = "none"
    """none | mdl (Gaussian prior, empirical-Bayes variance) | svd (sparse variational dropout)"""
    kl_scale: float = 1.0
    """beta on KL(q||p)/N; 1 is the description-length (ELBO) weighting"""
    sigma_lr: float = 0.01
    """Adam rate of the log sigma^2 parameters, annealed with learning_rate"""
    init_log_sigma2: float = -12.0
    """initial log sigma^2: about 1% of the unit-row weight scale, i.e. step 0 ~= the control"""
    svd_threshold: float = 3.0
    """svd: weights with log alpha above this are pruned from the decoded model"""
    noise_segment: int = 64
    """steps per actor weight sample; each env draws num_steps / noise_segment samples per rollout"""

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


# -- weight posteriors --------------------------------------------------------------------
# A trunk's seven matrices, in `ResidualMLP.matrices()` order, enter `residual_forward` as
# per-layer specs; `linear(x, spec)` decides what a spec is: a dense matrix (decoded model),
# a per-env stack of sampled matrices (actor), or a (mean, variance) pair (critic, LRT).

def dense_linear(x, weight):
    return F.linear(x, weight)


def env_linear(x, weights):
    """x: (envs, rows, in); weights: (envs, out, in), one sampled matrix per env."""
    return torch.matmul(x, weights.transpose(-1, -2))


def lrt_linear(x, spec):
    """Local reparameterisation: exact per-row Gaussian preactivations of N(mu, var) weights."""
    mean, variance = spec
    loc = F.linear(x, mean)
    return loc + F.linear(x.square(), variance).sqrt() * torch.randn_like(loc)


def residual_forward(x, specs, head_bias, readout_gain, linear):
    """ResidualMLP.forward with replaceable matrices; identical maths to the module path."""
    gate1, up1, down1, gate2, up2, down2, head = specs
    h = F.normalize(linear(situ_glu(linear(x, gate1), linear(x, up1)), down1), p=2, dim=-1)
    scaled = 8.0 * h
    branch = F.normalize(linear(situ_glu(linear(scaled, gate2), linear(scaled, up2)), down2), p=2, dim=-1)
    return readout_gain * (linear(8.0 * F.normalize(h + branch, p=2, dim=-1), head) + head_bias)


def log_alpha(mean, log_sigma2):
    return log_sigma2 - torch.log(mean.square() + LOG_MU2_EPS)


def mdl_kl(mean, log_sigma2):
    """KL(q || N(0, s^2)) summed over a matrix, at the optimal s^2 = mean(mu^2 + sigma^2).

    The prior variance is profiled out: sum_i KL_i = n/2 log s^2 - 1/2 sum_i log sigma_i^2.
    By the envelope theorem the gradient through the optimal s^2 is the partial derivative at
    fixed s^2, so no detach is needed.
    """
    second_moment = (mean.square() + log_sigma2.exp()).mean()
    return 0.5 * mean.numel() * torch.log(second_moment) - 0.5 * log_sigma2.sum()


def svd_kl(mean, log_sigma2):
    """Molchanov et al. (2017) approximation to KL(q || log-uniform), summed over a matrix."""
    la = log_alpha(mean, log_sigma2)
    return (SVD_K1 - SVD_K1 * torch.sigmoid(SVD_K2 + SVD_K3 * la) + 0.5 * F.softplus(-la)).sum()


POSTERIOR_KL = {"mdl": mdl_kl, "svd": svd_kl}


class ResidualMLP(nn.Module):
    """Unit residual directions with variance-restored branches and readouts."""

    def __init__(self, observation_dim, output_dim, output_std, init_log_sigma2=None):
        super().__init__()

        def stage(in_dim):
            return nn.Sequential(init_situglu_branch(SiTUGLUBranch(in_dim, 64)))

        self.first = stage(observation_dim)
        self.second = stage(64)
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))
        self.readout_gain = nn.Parameter(torch.ones(output_dim))
        # Posterior log-variances, one per matrix; biases and gains stay point estimates.
        self.log_sigma2 = None if init_log_sigma2 is None else nn.ParameterList(
            nn.Parameter(torch.full_like(weight, init_log_sigma2)) for weight in self.matrices()
        )

    def matrices(self):
        first, second = self.first[0], self.second[0]
        return (first.gate.weight, first.up.weight, first.down.weight,
                second.gate.weight, second.up.weight, second.down.weight,
                self.head[0].weight)

    def forward(self, x):
        h = F.normalize(self.first(x), p=2, dim=-1)
        branch = F.normalize(self.second(8.0 * h), p=2, dim=-1)
        return self.readout_gain * self.head(8.0 * F.normalize(h + branch, p=2, dim=-1))

    def forward_with(self, x, specs, linear):
        return residual_forward(x, specs, self.head[0].bias, self.readout_gain, linear)


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


class PerEnvHostMirror:
    """FP32 host actor where env e acts with its own sampled weight matrices.

    The fused kernel shares one weight set across rows, so this mirror runs the same
    residual SiTU-GLU maths as `residual_forward(..., env_linear)` with batched NumPy
    matmuls over (envs, 1, in) x (envs, in, out). `refresh(samples)` takes the seven
    (envs, out, in) device matrices of the current noise segment; buffers are permanent and the
    returned logits are overwritten by the next call.
    """

    def __init__(self, actor, num_envs):
        self.actor = actor
        self.num_envs = num_envs
        pin = actor.readout_gain.is_cuda
        shapes = [tuple(weight.shape) for weight in actor.matrices()]
        self._hosts = [torch.empty((num_envs,) + shape, pin_memory=pin) for shape in shapes]
        self._transposed = [np.empty((num_envs, shape[1], shape[0]), dtype=np.float32) for shape in shapes]
        out_dim = shapes[-1][0]
        self._bias = torch.empty(out_dim, pin_memory=pin)
        self._gain = torch.empty(out_dim, pin_memory=pin)
        self._bias_view, self._gain_view = self._bias.numpy(), self._gain.numpy()
        self._event = torch.cuda.Event() if pin else None

        def buffer(width):
            return np.empty((num_envs, 1, width), dtype=np.float32)

        hidden1, hidden2 = shapes[0][0], shapes[3][0]
        self.gate1, self.up1, self.tmp1 = buffer(hidden1), buffer(hidden1), buffer(hidden1)
        self.gate2, self.up2, self.tmp2 = buffer(hidden2), buffer(hidden2), buffer(hidden2)
        self.hidden, self.branch, self.scaled, self.squared = (buffer(64) for _ in range(4))
        self.norm = np.empty((num_envs, 1, 1), dtype=np.float32)
        self.logits = buffer(out_dim)

    @torch.no_grad()
    def refresh(self, samples):
        for host, sample in zip(self._hosts, samples):
            host.copy_(sample, non_blocking=self._event is not None)
        self._bias.copy_(self.actor.head[0].bias, non_blocking=self._event is not None)
        self._gain.copy_(self.actor.readout_gain, non_blocking=self._event is not None)
        if self._event is not None:
            self._event.record()
            self._event.synchronize()
        for host, transposed in zip(self._hosts, self._transposed):
            np.copyto(transposed, host.numpy().transpose(0, 2, 1))

    def _normalize(self, values):
        np.multiply(values, values, out=self.squared)
        np.sum(self.squared, axis=-1, keepdims=True, out=self.norm)
        np.sqrt(self.norm, out=self.norm)
        np.maximum(self.norm, np.float32(1e-12), out=self.norm)
        np.divide(values, self.norm, out=values)

    @staticmethod
    def _situ_glu(gate, up, tmp):
        """In place: gate <- 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)."""
        np.negative(gate, out=tmp)
        # Very negative gates overflow exp toward inf; the reciprocal maps that to the 0 limit.
        with np.errstate(over="ignore"):
            np.exp(tmp, out=tmp)
        np.add(tmp, np.float32(1.0), out=tmp)
        np.reciprocal(tmp, out=tmp)
        np.multiply(gate, np.float32(0.25), out=gate)
        np.tanh(gate, out=gate)
        np.multiply(gate, np.float32(4.0), out=gate)
        np.multiply(gate, tmp, out=gate)
        np.multiply(up, np.float32(1.0 / 25.0), out=up)
        np.tanh(up, out=up)
        np.multiply(up, np.float32(25.0), out=up)
        np.multiply(gate, up, out=gate)

    def __call__(self, observations):
        gate1, up1, down1, gate2, up2, down2, head = self._transposed
        x = observations.reshape(self.num_envs, 1, -1)
        np.matmul(x, gate1, out=self.gate1)
        np.matmul(x, up1, out=self.up1)
        self._situ_glu(self.gate1, self.up1, self.tmp1)
        np.matmul(self.gate1, down1, out=self.hidden)
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.scaled)
        np.matmul(self.scaled, gate2, out=self.gate2)
        np.matmul(self.scaled, up2, out=self.up2)
        self._situ_glu(self.gate2, self.up2, self.tmp2)
        np.matmul(self.gate2, down2, out=self.branch)
        self._normalize(self.branch)
        np.add(self.hidden, self.branch, out=self.hidden)
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.hidden)
        np.matmul(self.hidden, head, out=self.logits)
        np.add(self.logits, self._bias_view, out=self.logits)
        np.multiply(self.logits, self._gain_view, out=self.logits)
        return self.logits.reshape(self.num_envs, -1)


class Agent(nn.Module):
    def __init__(self, envs, posterior="none", init_log_sigma2=-12.0, svd_threshold=3.0):
        super().__init__()
        if posterior not in POSTERIORS:
            raise ValueError(f"posterior must be one of {POSTERIORS}")
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta PPO requires finite, strictly ordered action bounds")
        self.posterior = posterior
        self.svd_threshold = float(svd_threshold)
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        log_sigma2 = None if posterior == "none" else init_log_sigma2
        self.critic = ResidualMLP(observation_dim, 1, output_std=1.0, init_log_sigma2=log_sigma2)
        self.actor = ResidualMLP(observation_dim, 2 * self.action_dim, output_std=0.01,
                                 init_log_sigma2=log_sigma2)

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

    # -- posterior views ----------------------------------------------------------------
    def decoded_matrices(self, trunk):
        """The deterministic model the posterior encodes: means, minus pruned weights for svd."""
        if self.posterior != "svd":
            return trunk.matrices()
        return tuple(mean * (log_alpha(mean, log_sigma2) <= self.svd_threshold)
                     for mean, log_sigma2 in zip(trunk.matrices(), trunk.log_sigma2))

    def description_length(self, trunk):
        """KL(q || p) of a trunk's matrices, in nats."""
        kl = POSTERIOR_KL[self.posterior]
        return sum(kl(mean, log_sigma2) for mean, log_sigma2 in zip(trunk.matrices(), trunk.log_sigma2))

    def sample_actor_noise(self, num_groups):
        return tuple(torch.randn((num_groups,) + tuple(mean.shape), device=mean.device)
                     for mean in self.actor.matrices())

    def actor_samples(self, noise):
        """Per-group weights mu + sigma * eps; differentiable in mu and log sigma^2."""
        return tuple(mean + (0.5 * log_sigma2).exp() * eps
                     for mean, log_sigma2, eps in zip(self.actor.matrices(), self.actor.log_sigma2, noise))

    def sampled_policy(self, env_observations, noise):
        """Beta parameters of each group's sampled actor; observations are (groups, rows, obs)."""
        logits = self.actor.forward_with(env_observations, self.actor_samples(noise), env_linear)
        return (F.softplus(logits) + 1.0).chunk(2, dim=-1)

    def decoded_policy(self, x):
        logits = self.actor.forward_with(x, self.decoded_matrices(self.actor), dense_linear)
        return (F.softplus(logits) + 1.0).chunk(2, dim=-1)

    def lrt_value(self, x):
        specs = tuple((mean, log_sigma2.exp())
                      for mean, log_sigma2 in zip(self.critic.matrices(), self.critic.log_sigma2))
        return self.critic.forward_with(x, specs, lrt_linear)

    # -- control interface ----------------------------------------------------------------
    def get_value(self, x):
        if self.posterior == "svd":
            return self.critic.forward_with(x, self.decoded_matrices(self.critic), dense_linear)
        return self.critic(x)

    def get_policy_and_value(self, x):
        """Control path; for a posterior, the decoded model (evaluation and public API)."""
        if self.posterior != "none":
            alpha, beta = self.decoded_policy(x)
            return alpha, beta, self.get_value(x)
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


def posterior_ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns,
                       old_values, noise, advantage_scale, noise_variance, args):
    """Negative ELBO: the control's PPO loss under q, plus beta * KL / N per network.

    Rows arrive grouped, (groups, rows, ...), one group per recorded (env, noise segment)
    weight sample. The actor KL is charged in normalised-advantage units (`advantage_scale`
    = the rollout's std A, the weighting norm-adv implies); the critic KL at vf_coef *
    `noise_variance`, the rollout's held-out MSE as the Gaussian likelihood's variance.
    """
    alpha, beta = agent.sampled_policy(observations, noise)
    distribution = Beta(alpha.flatten(0, 1), beta.flatten(0, 1), validate_args=False)
    native_actions, old_logprobs = native_actions.flatten(0, 1), old_logprobs.flatten()
    advantages, returns, old_values = advantages.flatten(), returns.flatten(), old_values.flatten()
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
    newvalue = agent.lrt_value(observations.flatten(0, 1)).view(-1)
    v_loss_unclipped = (newvalue - returns) ** 2
    if args.clip_vloss:
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * v_loss_unclipped.mean()
    actor_kl = agent.description_length(agent.actor)
    critic_kl = agent.description_length(agent.critic)
    entropy_loss = entropy.mean()
    loss = (pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            + args.kl_scale / args.batch_size * (advantage_scale * actor_kl
                                                 + args.vf_coef * noise_variance * critic_kl))
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac, actor_kl.detach(), critic_kl.detach()))
    return loss, metrics


def steps_to_groups(step_major, segment):
    """(steps, envs, ...) -> (envs * steps / segment, segment, ...), contiguous; group
    g = env * (steps / segment) + segment index holds one weight sample's transitions."""
    env_major = step_major.transpose(0, 1).contiguous()
    return env_major.view((-1, segment) + tuple(step_major.shape[2:]))


def groups_to_steps(grouped, num_envs):
    """Inverse of `steps_to_groups` for (groups, segment) tensors: (steps, envs), contiguous."""
    return grouped.reshape(num_envs, -1).transpose(0, 1).contiguous()


def group_minibatches(num_groups, group_size, num_minibatches, device, generator):
    """Per-group row permutations split into minibatches: every minibatch holds
    group_size / num_minibatches rows of every weight sample, as (groups, rows) indices."""
    order = torch.rand((num_groups, group_size), device=device, generator=generator).argsort(dim=1)
    return order.split(group_size // num_minibatches, dim=1)


@torch.no_grad()
def posterior_telemetry(agent, env_observations, noise, threshold):
    """Bits, noise scale, sparsity and the sampled actors' divergence from the decoded one."""
    stats = {}
    for name, trunk in (("actor", agent.actor), ("critic", agent.critic)):
        means, log_sigma2 = trunk.matrices(), tuple(trunk.log_sigma2)
        weights = sum(mean.numel() for mean in means)
        alphas = torch.cat([log_alpha(m, s).flatten() for m, s in zip(means, log_sigma2)])
        bits = agent.description_length(trunk) / math.log(2.0)
        stats[f"posterior/{name}_bits"] = bits
        stats[f"posterior/{name}_bits_per_weight"] = bits / weights
        stats[f"posterior/{name}_log_alpha_median"] = alphas.median()
        stats[f"posterior/{name}_pruned_fraction"] = (alphas > threshold).float().mean()
        stats[f"posterior/{name}_log_sigma2_mean"] = torch.cat([s.flatten() for s in log_sigma2]).mean()
    alpha, beta = agent.sampled_policy(env_observations, noise)
    mean_alpha, mean_beta = agent.decoded_policy(env_observations)
    stats["posterior/actor_sample_policy_kl"] = kl_divergence(
        Beta(mean_alpha, mean_beta, validate_args=False), Beta(alpha, beta, validate_args=False),
    ).sum(-1).mean()
    return stats


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
    if args.posterior not in POSTERIORS:
        raise ValueError(f"posterior must be one of {POSTERIORS}")
    if args.posterior != "none":
        if args.noise_segment <= 0 or args.num_steps % args.noise_segment:
            raise ValueError("noise_segment must divide num_steps")
        if args.noise_segment % args.num_minibatches:
            raise ValueError("posterior minibatches split each noise segment's steps evenly")
        if args.kl_scale < 0 or args.sigma_lr <= 0:
            raise ValueError("kl_scale must be non-negative and sigma_lr positive")
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
    bayesian = args.posterior != "none"
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); FP32; native-action storage; host actor mirror")
        writer.add_text("architecture", "residual SiTU-GLU; unit-L2 matrices and residual directions; sqrt(width) branch/readout scale; learned output gains; no learned mixing")
        writer.add_text("posterior", {
            "none": "none: deterministic weights (control)",
            "mdl": "mdl: N(mu, sigma^2) weights, empirical-Bayes Gaussian prior, KL/N per rollout",
            "svd": "svd: sparse variational dropout, log-uniform prior, KL/N per rollout, log alpha pruning",
        }[args.posterior])
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args.posterior, args.init_log_sigma2, args.svd_threshold).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        if bayesian:
            variances = {id(p) for trunk in (agent.actor, agent.critic) for p in trunk.log_sigma2}
            optimizer = optim.Adam([
                {"params": [p for p in agent.parameters() if id(p) not in variances], "lr": args.learning_rate},
                {"params": [p for p in agent.parameters() if id(p) in variances], "lr": args.sigma_lr},
            ], eps=1e-5, fused=True)
        else:
            optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        actor_parameters, critic_parameters = tuple(agent.actor.parameters()), tuple(agent.critic.parameters())
        value_model = agent.get_value

        if bayesian:
            def rollout_statistics(env_observations, env_native, noise):
                """Decoded values and the sampled actors' old log-probabilities, grouped."""
                alpha, beta = agent.sampled_policy(env_observations, noise)
                value = agent.get_value(env_observations.flatten(0, 1)).view(env_observations.shape[:2])
                return value, agent.action_logprob(alpha, beta, env_native)

            def loss_model(observations, native, old_logprobs, advantages, returns, old_values, noise,
                           advantage_scale, noise_variance):
                return posterior_ppo_loss(agent, observations, native, old_logprobs, advantages,
                                          returns, old_values, noise, advantage_scale, noise_variance, args)
        else:
            def rollout_statistics(observations, native):
                """Old log-probabilities and values for a whole uploaded rollout in one forward."""
                alpha, beta, value = agent.get_policy_and_value(observations)
                return value.flatten(), agent.action_logprob(alpha, beta, native)

            def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
                return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        if bayesian:
            # Group g = env * segments + segment holds one weight sample's transitions.
            segments = args.num_steps // args.noise_segment
            num_groups = args.num_envs * segments

            def to_groups(step_major):
                return steps_to_groups(step_major, args.noise_segment)

            def to_steps(grouped):
                return groups_to_steps(grouped, args.num_envs)

            def draw_actor_samples():
                noise = agent.sample_actor_noise(num_groups)
                with torch.no_grad():
                    samples = tuple(sample.view((args.num_envs, segments) + tuple(sample.shape[1:]))
                                    for sample in agent.actor_samples(noise))
                return noise, samples

            host_actor = PerEnvHostMirror(agent.actor, args.num_envs)
            actor_noise, actor_samples = draw_actor_samples()
            host_actor.refresh(tuple(sample[:, 0] for sample in actor_samples))
        else:
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
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 8 if bayesian else 6), device=device)
        grad_norms = torch.empty((max_updates, 2), device=device)
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
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = frac * base_lr
            bootstraps.reset()
            if bayesian:
                # Fresh weight samples per env and segment for this rollout, replayed by the update.
                actor_noise, actor_samples = draw_actor_samples()
            else:
                host_actor.refresh()
            for step in range(args.num_steps):
                if bayesian and step % args.noise_segment == 0:
                    segment = step // args.noise_segment
                    host_actor.refresh(tuple(sample[:, segment] for sample in actor_samples))
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
                if bayesian:
                    # Grouped layout so each (env, segment) sample meets its own transitions.
                    e_obs = to_groups(batch.fields["observations"])
                    e_native = to_groups(batch.fields["native_actions"])
                    e_values, e_logprobs = rollout_statistics(e_obs, e_native, actor_noise)
                    values = to_steps(e_values)
                    b_values = values.flatten()
                else:
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
                if bayesian:
                    e_advantages, e_returns = to_groups(advantages), to_groups(returns)
                    advantage_scale = b_advantages.new_ones(()) if args.norm_adv else b_advantages.std()
                    noise_variance = (b_values - b_returns).square().mean()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    if bayesian:
                        minibatches = group_minibatches(num_groups, args.noise_segment,
                                                        args.num_minibatches, device, shuffle_generator)
                    else:
                        minibatches = device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator)
                    for indices in minibatches:
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        if bayesian:
                            rows = indices.unsqueeze(-1)
                            loss, metrics = loss_model(
                                e_obs.take_along_dim(rows, dim=1), e_native.take_along_dim(rows, dim=1),
                                e_logprobs.take_along_dim(indices, dim=1),
                                e_advantages.take_along_dim(indices, dim=1),
                                e_returns.take_along_dim(indices, dim=1),
                                e_values.take_along_dim(indices, dim=1), actor_noise,
                                advantage_scale, noise_variance,
                            )
                        else:
                            loss, metrics = loss_model(
                                b_obs[indices], b_native[indices], b_logprobs[indices],
                                b_advantages[indices], b_returns[indices], b_values[indices],
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norms[updates, 0] = nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm, foreach=True)
                        grad_norms[updates, 1] = nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
                        optimizer.step()
                        normalize_matrices()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            with torch.no_grad():
                # The pre-update critic never saw this rollout's targets (held out); the
                # post-update decoded critic has fit them. Their gap is batch-specific fit.
                fit_values = (to_steps(agent.get_value(e_obs.flatten(0, 1))).flatten()
                              if bayesian else agent.get_value(b_obs).flatten())
                named = {
                    "losses/policy_loss": last[0], "losses/value_loss": last[1],
                    "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                    "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                    "losses/explained_variance": explained_variance(b_values, b_returns),
                    "grad/actor_preclip_norm": grad_norms[:updates, 0].mean(),
                    "grad/critic_preclip_norm": grad_norms[:updates, 1].mean(),
                    "critic/heldout_mse": (b_values - b_returns).square().mean(),
                    "critic/fit_mse": (fit_values - b_returns).square().mean(),
                }
                if bayesian:
                    named.update(posterior_telemetry(agent, e_obs, actor_noise, args.svd_threshold))
                    named["charts/sigma_lr"] = torch.tensor(optimizer.param_groups[1]["lr"])
            logged = gather_metrics(named)
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
            from cleanrl_utils.evals.ppo_eval import evaluate
            episodic_returns = evaluate(
                model_path, make_env, args.env_id, eval_episodes=10,
                run_name=f"{run_name}-eval", device=device, gamma=args.gamma,
                Model=functools.partial(Agent, posterior=args.posterior,
                                        init_log_sigma2=args.init_log_sigma2,
                                        svd_threshold=args.svd_threshold),
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

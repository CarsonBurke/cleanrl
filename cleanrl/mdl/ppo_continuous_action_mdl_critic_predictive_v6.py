# Critic MDL whose values are the posterior predictive mean (mdl v6).
# v5 (online MDL: VOGN precisions under q, sequential prior) made the posterior self-consistent:
# Monte-Carlo and Gauss-Newton pulls agree within ~5% and sigma sits at its fixed point. Its b1 runs
# still lost to the control at 10 epochs (empirical 4830, sequential 6140 against 8413). The cause
# was the value GAE uses. The means are fit so that the noisy critic matches the returns, so
# E_q[v] ~ R. GAE, however, used the critic at the means, v(mu), which exceeds E_q[v] by the Jensen
# gap. The held-out bias equalled minus that gap: +0.11 for empirical b1, +0.03 for sequential
# gamma 0.5. Without advantage normalisation, a value offset b shifts every advantage by about
# -(1 - gamma) b / (1 - gamma lambda) ~ -0.17 b. That lowers every sampled action's probability,
# so entropy stayed high (-0.3 against the control's -4.0), and returns followed the bias.
# v6 uses the Bayes prediction the ELBO trains:
#   v(x) = 1/K sum_k v(x; mu + sigma eps_k),
# a K-network ensemble drawn once per rollout. eps is fixed, so every state of a rollout sees the
# same K critics, and their Monte-Carlo error is correlated across neighbouring states and largely
# cancels in the TD differences. Everything else is v5.
# Hypothesis: with the offset gone, b1 matches or beats the control at 10 epochs, and the 40-epoch
# gain persists (v5's sequential gamma 0.9 at 40 epochs scored 7804 against the control's 6288).
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
from torch.distributions import Beta
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
POSTERIORS = ("none", "mdl")
PRIORS = ("empirical", "sequential")
# Telemetry only: weights whose noise exceeds e^3 times their mean's square count as prunable.
PRUNABLE_LOG_ALPHA = 3.0
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
    """critic posterior: none | mdl (diagonal Gaussian weights; see --prior)"""
    kl_scale: float = 1.0
    """beta on KL(q||p)/N; 1 is the description-length (ELBO) weighting"""
    prior: str = "sequential"
    """empirical: N(0, s_p^2) every rollout (v4) | sequential: the previous posterior, tempered by forgetting"""
    forgetting: float = 0.9
    """sequential prior: gamma, the fraction of the previous posterior's precision kept per rollout"""
    precision_draws: int = 4
    """weight samples per evaluation of the Gauss-Newton diagonal under q"""
    precision_iterations: int = 4
    """damped fixed-point iterations of the precision per rollout (F_q depends on sigma)"""
    precision_damping: float = 0.5
    """rho of each natural-gradient precision iteration"""
    init_log_sigma2: float = -12.0
    """log sigma^2 before the first rollout's precision step"""
    predictive_draws: int = 16
    """K: weight samples whose mean critic gives the values GAE uses (drawn once per rollout)"""

    # Diagnostics (no effect on training).
    diagnostic_draws: int = 8
    """Monte-Carlo local-reparameterisation draws per diagnostic evaluation"""
    checkpoint_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    """fractions of training after which the agent's state is saved to the run directory"""

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
# per-layer specs; `linear(x, spec)` decides what a spec is: a dense matrix (the means, or a
# weight sample for the Gauss-Newton recording) or a (mean, variance) pair (local reparameterisation).

def ensemble_linear(x, weights):
    """K weight samples (K, out, in) applied to x (n, in) or (K, n, in): (K, n, out)."""
    return torch.matmul(x, weights.transpose(-1, -2))


def lrt_linear(x, spec):
    """Local reparameterisation: exact per-row Gaussian preactivations of N(mu, var) weights."""
    mean, variance = spec
    loc = F.linear(x, mean)
    return loc + F.linear(x.square(), variance).sqrt() * torch.randn_like(loc)


def lrt_linear_from(generator):
    """`lrt_linear` drawing from a private generator, so diagnostics leave training's stream alone."""
    def linear(x, spec):
        mean, variance = spec
        loc = F.linear(x, mean)
        noise = torch.randn(loc.shape, generator=generator, device=loc.device, dtype=loc.dtype)
        return loc + F.linear(x.square(), variance).sqrt() * noise
    return linear


def residual_forward(x, specs, head_bias, readout_gain, linear):
    """ResidualMLP.forward with replaceable matrices; identical maths to the module path."""
    gate1, up1, down1, gate2, up2, down2, head = specs
    h = F.normalize(linear(situ_glu(linear(x, gate1), linear(x, up1)), down1), p=2, dim=-1)
    scaled = 8.0 * h
    branch = F.normalize(linear(situ_glu(linear(scaled, gate2), linear(scaled, up2)), down2), p=2, dim=-1)
    return readout_gain * (linear(8.0 * F.normalize(h + branch, p=2, dim=-1), head) + head_bias)


def log_alpha(mean, log_sigma2):
    return log_sigma2 - torch.log(mean.square() + LOG_MU2_EPS)


def gaussian_kl(mean, log_sigma2, prior_precision, prior_mean):
    """KL(N(mu, sigma^2) || N(m, 1/Lambda)) summed over a matrix."""
    return 0.5 * (prior_precision * (log_sigma2.exp() + (mean - prior_mean).square())
                  - 1.0 - prior_precision.log() - log_sigma2).sum()


def base_precision(mean, log_sigma2):
    """1/s_p^2 of the empirical-Bayes prior N(0, s_p^2), s_p^2 = mean(mu^2 + sigma^2) over the matrix."""
    return 1.0 / (mean.square() + log_sigma2.exp()).mean()


class ResidualMLP(nn.Module):
    """Unit residual directions with variance-restored branches and readouts."""

    def __init__(self, observation_dim, output_dim, output_std, init_log_sigma2=None, predictive_draws=1):
        super().__init__()

        def stage(in_dim):
            return nn.Sequential(init_situglu_branch(SiTUGLUBranch(in_dim, 64)))

        self.first = stage(observation_dim)
        self.second = stage(64)
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))
        self.readout_gain = nn.Parameter(torch.ones(output_dim))
        # Posterior log-variances and this rollout's prior, one per matrix; biases and gains stay
        # point estimates. They are buffers: the precision step sets them, no optimiser touches them.
        self.has_posterior = init_log_sigma2 is not None
        if self.has_posterior:
            for index, weight in enumerate(self.matrices()):
                self.register_buffer(f"log_sigma2_{index}", torch.full_like(weight, init_log_sigma2))
                self.register_buffer(f"prior_precision_{index}", torch.ones_like(weight))
                self.register_buffer(f"prior_mean_{index}", torch.zeros_like(weight))
                # Standard-normal draws eps_k of the predictive ensemble; weights mu + sigma eps_k.
                self.register_buffer(f"predictive_noise_{index}",
                                     torch.zeros((predictive_draws,) + tuple(weight.shape)))

    def _posterior_buffers(self, name):
        if not self.has_posterior:
            return None
        return tuple(getattr(self, f"{name}_{index}") for index in range(len(self.matrices())))

    @property
    def log_sigma2(self):
        return self._posterior_buffers("log_sigma2")

    @property
    def prior_precision(self):
        return self._posterior_buffers("prior_precision")

    @property
    def prior_mean(self):
        return self._posterior_buffers("prior_mean")

    @property
    def predictive_noise(self):
        return self._posterior_buffers("predictive_noise")

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


class Agent(nn.Module):
    def __init__(self, envs, posterior="none", init_log_sigma2=-12.0, predictive_draws=16):
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
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.critic = ResidualMLP(observation_dim, 1, output_std=1.0,
                                  init_log_sigma2=None if posterior == "none" else init_log_sigma2,
                                  predictive_draws=predictive_draws)
        self.actor = ResidualMLP(observation_dim, 2 * self.action_dim, output_std=0.01)

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

    # -- critic posterior -----------------------------------------------------------------
    def description_length(self):
        """KL(q || p) of the critic's matrices against this rollout's prior, in nats."""
        critic = self.critic
        return sum(gaussian_kl(*terms) for terms in zip(critic.matrices(), critic.log_sigma2,
                                                         critic.prior_precision, critic.prior_mean))

    def anchor(self):
        """The KL's mean term, 1/2 sum Lambda (mu - m)^2: what moving the means costs, in nats."""
        critic = self.critic
        return 0.5 * sum((precision * (mean - prior_mean).square()).sum() for mean, precision, prior_mean
                         in zip(critic.matrices(), critic.prior_precision, critic.prior_mean))

    def lrt_value(self, x, linear=lrt_linear):
        specs = tuple((mean, log_sigma2.exp())
                      for mean, log_sigma2 in zip(self.critic.matrices(), self.critic.log_sigma2))
        return self.critic.forward_with(x, specs, linear)

    @torch.no_grad()
    def resample_predictive(self, generator):
        """Draw the ensemble's eps_k in place (the compiled value graphs read these buffers)."""
        for noise in self.critic.predictive_noise:
            noise.normal_(generator=generator)

    def predictive_value(self, x):
        """Posterior predictive mean 1/K sum_k v(x; mu + sigma eps_k), the value the ELBO fits."""
        critic = self.critic
        samples = tuple(mean + (0.5 * log_sigma2).exp() * noise for mean, log_sigma2, noise
                        in zip(critic.matrices(), critic.log_sigma2, critic.predictive_noise))
        return critic.forward_with(x, samples, ensemble_linear).mean(0)

    # -- control interface ----------------------------------------------------------------
    def get_value(self, x):
        """The control critic, or the posterior predictive mean."""
        return self.critic(x) if self.posterior == "none" else self.predictive_value(x)

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


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args,
             kl_weight=None):
    """Pure clipped PPO loss on native Beta samples; no inverse action scaling.

    With a critic posterior, the value prediction is one local-reparameterisation draw (the ELBO's
    expected likelihood for the means). A sequential prior adds the KL's mean term,
    kl_weight * anchor. Against a zero-mean prior (the empirical arm; a sequential run's first
    rollout) that term is radial, so the unit-norm rows make it constant; the empirical arm omits it.
    The precisions are set outside the loss by `natural_precision_step`.
    """
    alpha, beta = (F.softplus(agent.actor(observations)) + 1.0).chunk(2, dim=-1)
    newvalue = agent.critic(observations) if agent.posterior == "none" else agent.lrt_value(observations)
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
    if agent.posterior != "none" and args.prior == "sequential":
        loss = loss + kl_weight * agent.anchor()
    metrics = (pg_loss.detach(), v_loss.detach(), entropy_loss.detach(), old_approx_kl, approx_kl, clipfrac)
    return loss, torch.stack(metrics)


@torch.no_grad()
def posterior_telemetry(agent, kl_weight, sequential):
    """Critic bits against the current prior, noise scale and sparsity; for a sequential prior, the anchor."""
    means, log_sigma2 = agent.critic.matrices(), tuple(agent.critic.log_sigma2)
    alphas = torch.cat([log_alpha(m, s).flatten() for m, s in zip(means, log_sigma2)])
    bits = agent.description_length() / math.log(2.0)
    out = {
        "posterior/critic_bits": bits,
        "posterior/critic_bits_per_weight": bits / sum(mean.numel() for mean in means),
        "posterior/critic_log_alpha_median": alphas.median(),
        "posterior/critic_prunable_fraction": (alphas > PRUNABLE_LOG_ALPHA).float().mean(),
        "posterior/critic_log_sigma2_mean": torch.cat([s.flatten() for s in log_sigma2]).mean(),
        "posterior/prior_log_precision_median":
            torch.cat([p.flatten() for p in agent.critic.prior_precision]).log().median(),
    }
    if sequential:
        critic = agent.critic
        anchor_grads = [kl_weight * precision * (mean - prior_mean) for mean, precision, prior_mean
                        in zip(critic.matrices(), critic.prior_precision, critic.prior_mean)]
        out["posterior/critic_anchor_nats"] = agent.anchor()
        out["grad/critic_anchor_norm"] = torch.linalg.vector_norm(torch.cat([g.flatten() for g in anchor_grads]))
    return out


def critic_error(prefix, values, returns):
    """Mean-squared error split into constant bias and residual variance."""
    residual = values - returns
    return {f"{prefix}_bias": residual.mean(), f"{prefix}_residual_var": residual.var(unbiased=False)}


@torch.no_grad()
def posterior_diagnostics(prefix, agent, observations, returns, decoded, generator, draws):
    """Noisy critic (what the loss trains) against the values GAE uses (the K-sample predictive mean).

    In v6 the "jensen gap" is the predictive ensemble's Monte-Carlo error, no longer an offset.
    """
    linear = lrt_linear_from(generator)
    noisy = torch.stack([agent.lrt_value(observations, linear).flatten() for _ in range(draws)])
    ensemble = noisy.mean(0)
    # A finite ensemble keeps noise_var / draws of the noise; remove it to estimate the infinite one.
    noise_var = noisy.var(0).mean()
    finite = noise_var / draws
    return {
        f"{prefix}_noise_std": noise_var.sqrt(),
        f"{prefix}_jensen_gap": (ensemble - decoded).mean(),
        f"{prefix}_jensen_gap_rms": ((ensemble - decoded).square().mean() - finite).clamp_min(0).sqrt(),
        f"{prefix}_noisy_mse": (noisy - returns).square().mean(),
        f"{prefix}_ensemble_mse": (ensemble - returns).square().mean() - finite,
    }


def gauss_newton_under_posterior(agent, observations, generator, draws):
    """E_q of the per-sample Gauss-Newton diagonal, (dv_n/dw)^2, by whole-batch weight samples.

    At a weight sample w ~ q a linear map's (dv_n/dw_ij)^2 is delta_ni^2 x_nj^2, with x and the
    backpropagated delta both taken through the sampled network. This is VOGN's curvature; by Price's
    theorem 1/2 E_q of it is the Gauss-Newton part of d E_q[1/2 (v - R)^2] / d sigma^2. Unlike the
    GN at the means, it sees gates that noise unsaturates and each GLU factor's noise in the other's
    sensitivity. One sample is shared by the whole batch, so it is unbiased per observation and costs
    a dense forward and backward. Local reparameterisation with the noise scale detached would
    backpropagate through the means and miss the second stage's noise (0.03-0.07x of the exact
    derivative on v4's checkpoints, against 0.6-1.3x for weight samples).
    """
    total = None
    for _ in range(draws):
        inputs, outputs = [], []

        def recording_linear(x, weight):
            y = F.linear(x, weight)
            inputs.append(x.detach())
            outputs.append(y)
            return y

        samples = tuple(
            (mean + (0.5 * log_sigma2).exp() * torch.randn(mean.shape, generator=generator, device=mean.device,
                                                           dtype=mean.dtype)).requires_grad_()
            for mean, log_sigma2 in zip(agent.critic.matrices(), agent.critic.log_sigma2))
        with torch.enable_grad():
            value = agent.critic.forward_with(observations, samples, recording_linear)
            deltas = torch.autograd.grad(value.sum(), outputs)
        fisher = [delta.square().T @ x.square() / observations.shape[0] for delta, x in zip(deltas, inputs)]
        total = fisher if total is None else [a + b for a, b in zip(total, fisher)]
    return [f / draws for f in total]


@torch.no_grad()
def set_empirical_prior(agent):
    """N(0, s_p^2) per matrix, s_p^2 = mean(mu^2 + sigma^2): v4's per-rollout code."""
    critic = agent.critic
    for mean, log_sigma2, precision, prior_mean in zip(critic.matrices(), critic.log_sigma2,
                                                       critic.prior_precision, critic.prior_mean):
        precision.copy_(base_precision(mean, log_sigma2).expand_as(precision))
        prior_mean.zero_()


@torch.no_grad()
def set_sequential_prior(agent, forgetting):
    """The previous posterior, tempered by forgetting and mixed with the empirical-Bayes base prior.

    Lambda = gamma lambda + (1 - gamma)/s_p^2,  m = gamma lambda mu / Lambda (the base prior's mean is 0).
    """
    critic = agent.critic
    for mean, log_sigma2, precision, prior_mean in zip(critic.matrices(), critic.log_sigma2,
                                                       critic.prior_precision, critic.prior_mean):
        kept = forgetting * (-log_sigma2).exp()
        precision.copy_(kept + (1.0 - forgetting) * base_precision(mean, log_sigma2))
        prior_mean.copy_(kept * mean / precision)


@torch.no_grad()
def natural_precision_step(agent, observations, kl_weight, vf_coef, generator, draws, iterations, damping,
                           refresh_prior=None):
    """Natural-gradient iterations of the posterior precisions toward their fixed point, in place.

    For q = N(mu, sigma^2) under vf_coef/2 E_q||v - R||^2 + kl_weight KL(q || N(m, 1/Lambda)), the
    Bayesian learning rule moves the precision by lambda <- (1 - rho) lambda + rho (Lambda + vf_coef F_q / kl_weight).
    F_q depends on sigma, so the step is iterated. The first iteration is a full step (rho = 1): the
    starting sigma (the initialisation, or a posterior fit to other means) carries no information
    the target lacks. Later iterations use rho = damping, averaging F_q's Monte-Carlo noise over
    draws. The empirical prior depends on sigma, so it is refreshed each iteration.
    """
    for iteration in range(iterations):
        if refresh_prior is not None:
            refresh_prior(agent)
        fisher = gauss_newton_under_posterior(agent, observations, generator, draws)
        rho = 1.0 if iteration == 0 else damping
        for log_sigma2, precision, f in zip(agent.critic.log_sigma2, agent.critic.prior_precision, fisher):
            target = precision + vf_coef * f / kl_weight
            log_sigma2.copy_(-torch.lerp((-log_sigma2).exp(), target, rho).log())
    if refresh_prior is not None:
        refresh_prior(agent)


COLD_START_ITERATIONS = 4  # multiplier on precision_iterations for the first rollout


def fit_posterior(agent, observations, kl_weight, args, generator, prior, cold=False):
    """Set this rollout's prior, then the precisions; prior is "empirical", "sequential" or "kept".

    "sequential" turns the stored posterior into the prior; "kept" refits sigma against the current
    prior at the current means (the post-update refit). The first rollout has no previous posterior,
    so a sequential run starts from the empirical prior. Later rollouts start from the previous sigma,
    near the new fixed point; the first starts from the initialisation, far from it, and the
    iteration contracts slowly there (F_q falls as the noise grows), so it gets more iterations.
    """
    if prior == "sequential":
        set_sequential_prior(agent, args.forgetting)
    iterations = args.precision_iterations * (COLD_START_ITERATIONS if cold else 1)
    natural_precision_step(agent, observations, kl_weight, args.vf_coef, generator, args.precision_draws,
                           iterations, args.precision_damping,
                           refresh_prior=set_empirical_prior if prior == "empirical" else None)


@torch.no_grad()
def sigma_equilibrium(agent, observations, returns, kl_weight, vf_coef, generator, draws):
    """How far the update moved the posterior from its optimum: forces and fixed point at the new means.

    Pull on log sigma^2 from the data: vf_coef/2 sigma^2 F_q. The Monte-Carlo pull (LRT gradients are
    unbiased) checks it in aggregate. Push from the KL: kl_weight/2 (1 - Lambda sigma^2), positive
    where the KL wants more noise. A weight's sigma falls where pull > push.
    """
    fisher = gauss_newton_under_posterior(agent, observations, generator, draws)
    critic = agent.critic
    means, log_sigma2, prior_precision = critic.matrices(), critic.log_sigma2, critic.prior_precision
    leaves = [s.detach().clone().requires_grad_() for s in log_sigma2]
    linear = lrt_linear_from(generator)
    with torch.enable_grad():
        specs = tuple((m.detach(), s.exp()) for m, s in zip(means, leaves))
        noisy = torch.stack([critic.forward_with(observations, specs, linear).flatten() for _ in range(draws)])
        mc_grads = torch.autograd.grad(vf_coef * 0.5 * (noisy - returns).square().mean(), leaves)
    flat = lambda tensors: torch.cat([t.flatten() for t in tensors])
    pull = flat([0.5 * vf_coef * s.exp() * f for s, f in zip(log_sigma2, fisher)])
    push = flat([0.5 * kl_weight * (1 - lam * s.exp()) for s, lam in zip(log_sigma2, prior_precision)])
    pushed = push > 0
    target = flat([-(lam + vf_coef * f / kl_weight).log() for lam, f in zip(prior_precision, fisher)])
    current = flat(log_sigma2)
    return {
        "equilibrium/data_dominant_fraction": (pull > push).float().mean(),
        "equilibrium/kl_pushes_up_fraction": pushed.float().mean(),
        "equilibrium/log_pull_over_push_median": (pull[pushed] / push[pushed]).log().nanmedian(),
        "equilibrium/mc_over_gn_pull": flat(mc_grads).sum() / pull.sum(),
        "equilibrium/fisher_median": flat(fisher).median(),
        "equilibrium/log_sigma2_target_median": target.median(),
        "equilibrium/log_sigma2_gap_median": (target - current).median(),
        "equilibrium/log_sigma2_gap_abs_mean": (target - current).abs().mean(),
    }


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
    if args.posterior != "none" and (args.kl_scale <= 0 or not 0 < args.precision_damping <= 1
                                     or min(args.precision_draws, args.precision_iterations) < 1):
        raise ValueError("kl_scale must be positive, precision_damping in (0, 1], draws and iterations >= 1")
    if args.prior not in PRIORS or not 0 < args.forgetting < 1:
        raise ValueError(f"prior must be one of {PRIORS} and forgetting in (0, 1)")
    if args.posterior != "none" and args.clip_vloss:
        raise ValueError("the precision step assumes the unclipped value loss: pass --no-clip-vloss")
    if args.predictive_draws < 1:
        raise ValueError("predictive_draws must be positive")
    if args.diagnostic_draws < 2:
        raise ValueError("diagnostic_draws must be at least 2 to estimate output noise")
    if any(not 0 < f <= 1 for f in args.checkpoint_fractions):
        raise ValueError("checkpoint fractions must lie in (0, 1]")
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
            "none": "none: deterministic critic (control)",
            "mdl": f"mdl: critic N(mu, sigma^2) weights, {args.prior} prior, natural-gradient precisions under q",
        }[args.posterior])
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args.posterior, args.init_log_sigma2, args.predictive_draws).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        actor_parameters, critic_parameters = tuple(agent.actor.parameters()), tuple(agent.critic.parameters())
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and (predictive) values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values, kl_weight):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args,
                            kl_weight)

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
        # Diagnostic draws likewise get their own stream: they must not perturb training's RNG.
        diagnostic_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        posterior_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
        predictive_generator = torch.Generator(device=device).manual_seed(args.seed + 3)
        kl_weight = torch.zeros((), device=device)  # an input of the compiled loss: updated in place
        sequential = bayesian and args.prior == "sequential"
        checkpoint_iterations = {max(1, round(f * args.num_iterations)) for f in args.checkpoint_fractions}
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
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
                if bayesian:  # one ensemble for every value of this rollout, fitted ones included
                    agent.resample_predictive(predictive_generator)
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
                # The pre-update critic never saw these targets: its MSE is held out, and
                # is the Gaussian likelihood's variance for this rollout's critic ELBO.
                heldout_mse = (b_values - b_returns).square().mean()
                diagnostics = {
                    "critic/return_mean": b_returns.mean(),
                    "critic/return_var": b_returns.var(unbiased=False),
                    **critic_error("critic/heldout", b_values, b_returns),
                }
                if bayesian:
                    # v5's value, the critic at its means, kept to measure the offset v6 removes.
                    diagnostics.update(critic_error("critic/heldout_at_means", agent.critic(b_obs).flatten(), b_returns))
                    # In place: the compiled loss reads these buffers at fixed addresses.
                    # A perfectly predicted rollout would make the precision infinite; never reached in practice.
                    kl_weight.copy_(args.kl_scale * args.vf_coef * heldout_mse.clamp_min(1e-12) / args.batch_size)
                    fit_posterior(agent, b_obs, kl_weight, args, posterior_generator,
                                  "sequential" if sequential and iteration > 1 else "empirical", cold=iteration == 1)
                    diagnostics.update(posterior_diagnostics(
                        "critic/heldout", agent, b_obs, b_returns, b_values,
                        diagnostic_generator, args.diagnostic_draws))
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        inputs = (b_obs[indices], b_native[indices], b_logprobs[indices],
                                  b_advantages[indices], b_returns[indices], b_values[indices])
                        loss, metrics = loss_model(*inputs, kl_weight)
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
                # Post-update predictive critic on the targets it has now fit: the gap to the
                # held-out MSE is batch-specific fit.
                named = {
                    "losses/policy_loss": last[0], "losses/value_loss": last[1],
                    "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                    "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                    "losses/explained_variance": explained_variance(b_values, b_returns),
                    "grad/actor_preclip_norm": grad_norms[:updates, 0].mean(),
                    "grad/critic_preclip_norm": grad_norms[:updates, 1].mean(),
                    "critic/heldout_mse": heldout_mse,
                }
                fitted = agent.get_value(b_obs).flatten()
                named.update(diagnostics)
                named["critic/fit_mse"] = (fitted - b_returns).square().mean()
                named.update(critic_error("critic/fit", fitted, b_returns))
                if bayesian:
                    named["posterior/kl_weight"] = kl_weight
                    named.update(posterior_telemetry(agent, kl_weight, sequential))
                    named.update(posterior_diagnostics("critic/fit", agent, b_obs, b_returns, fitted,
                                                       diagnostic_generator, args.diagnostic_draws))
                    if not args.clip_vloss:  # the forces assume the unclipped loss
                        named.update(sigma_equilibrium(agent, b_obs, b_returns, kl_weight, args.vf_coef,
                                                       diagnostic_generator, args.diagnostic_draws))
                if sequential:
                    # The next prior is this rollout's posterior: refit sigma at the updated means
                    # (after the equilibrium metrics, which measure how stale the update left it).
                    fit_posterior(agent, b_obs, kl_weight, args, posterior_generator, "kept")
            logged = gather_metrics(named)
            # Diagnostic ratios may be undefined (no weight pushed up); they never stop training.
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance" and not name.startswith("equilibrium/")):
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
            if iteration in checkpoint_iterations:
                torch.save({
                    "iteration": iteration, "global_step": global_step, "args": vars(args),
                    "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                    "obs_norm": {k: getattr(obs_norm, k).copy() for k in ("means", "variances", "counts")},
                    "rew_norm": {k: getattr(rew_norm, k).copy() for k in ("returns", "means", "variances", "counts")},
                    # This iteration's rollout, so every diagnostic can be recomputed offline.
                    "rollout": {"observations": b_obs.cpu(), "returns": b_returns.cpu(),
                                "values": b_values.cpu(), "heldout_mse": heldout_mse.cpu()},
                }, f"runs/{run_name}/agent_{iteration:05d}.pt")

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
                                        predictive_draws=args.predictive_draws),
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

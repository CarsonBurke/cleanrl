# Surprise tilt v4: return reaches the policy only as what the agent failed to predict about itself.
# With any critic V and any model of its own actions' gain, GAE splits exactly into
#   A_t = own_t + sum_{k>=1} (gamma lambda)^k own_{t+k} + sum_{k>=0} (gamma lambda)^k (delta_{t+k} - own_{t+k})
# where own_t = Q~(s_t, a_t) - E_pi Q~(s_t, .), Q~ = r(s, a) + gamma V(s'(s, a)) from the consequence model.
# - own effect: what the agent already expected of this action against the alternatives it considered.
#   Its gradient is an expectation over considered actions (v3) and needs no sampled return.
# - luck: the own effects of *later* actions. Zero mean against this step's score; with gamma lambda =
#   0.94 it carries ~8x the own effect's variance, and PG / GAE reinforce it anyway.
# - surprise: prediction error, the only part where experience says something the agent did not know.
# Substituting r_t - own_t for the reward in stock GAE returns exactly the surprise sum, so the policy
# slope is u = E_pi[own z] + E[surprise z | s]: GAE's gradient for any model, minus the luck. A residual
# field (v1's slope network, plus a baseline) carries the surprise term; the closed-form Beta tilt and
# analytic actor fit are unchanged. Consequences are policy-independent, so they are remembered: the
# model trains on a replay of past transitions, not only the current rollout.
# Hypothesis: the agent's predictions of its own behaviour remove most of the return's variance, so a
# rollout carries several times the usable direction per unit of KL.
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

SAMPLE_EPS = 1e-6
# The 1 + softplus head cannot represent smaller concentrations.
MIN_CONCENTRATION = 1.0
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
    update_epochs: int = 30
    """the K epochs to fit the actor to the tilted target"""
    model_epochs: int = 10
    """epochs fitting the surprise field and the critic"""
    consequence_steps: int = 128
    """consequence-model steps per rollout; it is plain supervised regression on memory"""
    consequence_batch: int = 4096
    """half drawn from the newest rollout, half from all remembered transitions"""
    memory_rollouts: int = 64
    """rollouts of transitions remembered for the consequence model"""
    consequence_learning_rate: float = 0.001
    """constant: the predictions must not freeze while the policy still moves"""
    consequence_width: int = 512
    """hidden width of the consequence model"""
    predict_own: bool = True
    """ablation: without it every advantage is surprise and this is v1 with a baseline"""
    considered_actions: int = 128
    """actions per state whose predicted consequences are compared"""
    tilt_kl: float = 0.02
    """batch-mean KL(tilted target || old policy) solved for each rollout"""
    tilt_step: float | None = None
    """if set, tilt by this step in correlation units and treat tilt_kl as a ceiling"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping, applied per network"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile deterministic policy statistics, losses and GAE"""
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


class ConsequenceModel(nn.Module):
    """One-step effect of a native action: state change, raw reward and termination."""

    def __init__(self, observation_dim, action_dim, width):
        super().__init__()
        self.observation_dim = observation_dim
        self.net = nn.Sequential(
            layer_init(nn.Linear(observation_dim + action_dim, width)), nn.SiLU(),
            layer_init(nn.Linear(width, width)), nn.SiLU(),
            layer_init(nn.Linear(width, width)), nn.SiLU(),
            layer_init(nn.Linear(width, observation_dim + 2), std=0.01))
        self.register_buffer("scale", torch.ones(observation_dim + 1))
        self.register_buffer("calibrated", torch.zeros(()))

    @torch.no_grad()
    def calibrate(self, changes, rewards, valid, momentum=0.99):
        """Track the RMS scale of state changes and raw rewards over rollouts."""
        count = valid.sum().clamp_min(1.0)
        square = torch.cat(((changes.square() * valid[:, None]).sum(0) / count, rewards.square().mean()[None]))
        mixed = torch.where(self.calibrated.bool(), momentum * self.scale.square() + (1.0 - momentum) * square, square)
        self.scale.copy_(mixed.sqrt().clamp_min(1e-6))
        self.calibrated.fill_(1.0)

    def forward(self, observations, native_actions):
        return self.net(torch.cat((observations, 2.0 * native_actions - 1.0), dim=-1))

    def consequences(self, observations, native_actions):
        """Predicted (next observation, raw reward, termination probability)."""
        prediction = self(observations, native_actions)
        changes, reward, terminal = prediction.split((self.observation_dim, 1, 1), dim=-1)
        return (observations + changes * self.scale[:-1], reward.squeeze(-1) * self.scale[-1],
                terminal.squeeze(-1).sigmoid())


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
        self.critic = ResidualMLP(observation_dim, 1, output_std=1.0)
        self.actor = ResidualMLP(observation_dim, 2 * self.action_dim, output_std=0.01)
        self.consequence = ConsequenceModel(observation_dim, self.action_dim, args.consequence_width)
        # Baseline and whitened slope of the surprise; a zero gain starts from "nothing unexplained".
        self.residual = ResidualMLP(observation_dim, 1 + 2 * self.action_dim, output_std=0.01)
        nn.init.zeros_(self.residual.readout_gain)

    @torch.no_grad()
    def normalize_matrices(self):
        """Match nGPT's matrix axes without changing Adam moments or biases."""
        for trunk in (self.actor, self.critic, self.residual):
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


def beta_mean_statistics(alpha, beta):
    """E[log a] and E[log(1 - a)], concatenated like the policy head."""
    total = torch.digamma(alpha + beta)
    return torch.cat((torch.digamma(alpha) - total, torch.digamma(beta) - total), dim=-1)


def beta_cross_entropy(alpha, beta, mean_statistics):
    """-E_p[log Beta(a; alpha, beta)] for any p with the given mean statistics."""
    mean_log, mean_log1m = mean_statistics.chunk(2, dim=-1)
    log_normalizer = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return (log_normalizer - (alpha - 1.0) * mean_log - (beta - 1.0) * mean_log1m).sum(-1)


def beta_fisher_cholesky(alpha, beta):
    """Lower factor (l11, l21, l22) of each action dimension's Fisher matrix in (alpha, beta)."""
    tri_alpha, tri_beta, tri_total = (torch.polygamma(1, x) for x in (alpha, beta, alpha + beta))
    l11 = (tri_alpha - tri_total).sqrt()
    determinant = tri_alpha * tri_beta - tri_total * (tri_alpha + tri_beta)
    return l11, -tri_total / l11, determinant.sqrt() / l11


def whitened_score(alpha, beta, native_actions):
    """z = L^-1 grad_(alpha,beta) log pi(a): zero mean and identity covariance under pi."""
    mean_log, mean_log1m = beta_mean_statistics(alpha, beta).chunk(2, dim=-1)
    l11, l21, l22 = beta_fisher_cholesky(alpha, beta)
    first = (native_actions.log() - mean_log) / l11
    second = (torch.log1p(-native_actions) - mean_log1m - l21 * first) / l22
    return torch.cat((first, second), dim=-1)


def natural_step(alpha, beta, slope):
    """L^-T u: the (alpha, beta) displacement whose log-density tilt is u . z."""
    l11, l21, l22 = beta_fisher_cholesky(alpha, beta)
    first, second = slope.chunk(2, dim=-1)
    step_beta = second / l22
    return (first - l21 * step_beta) / l11, step_beta


def tilt_target(alpha, beta, slope, kl_budget, bisections=32):
    """Exponentially tilt Beta(alpha, beta) along slope until the batch-mean KL meets its budget.

    Returns the target's mean statistics and entropy per state, and
    (step, kl, clamp_fraction, slope_square) diagnostics.
    """
    step_alpha, step_beta = natural_step(alpha, beta, slope)
    old_statistics = beta_mean_statistics(alpha, beta)
    old_entropy = beta_cross_entropy(alpha, beta, old_statistics)

    def tilted(step):
        target_alpha = (alpha + step * step_alpha).clamp_min(MIN_CONCENTRATION)
        target_beta = (beta + step * step_beta).clamp_min(MIN_CONCENTRATION)
        statistics = beta_mean_statistics(target_alpha, target_beta)
        entropy = beta_cross_entropy(target_alpha, target_beta, statistics)
        kl = beta_cross_entropy(alpha, beta, statistics) - entropy
        return target_alpha, target_beta, statistics, entropy, kl

    slope_square = slope.square().sum(-1).mean()
    # KL = step^2 |u|^2 / 2 to second order; bisect the exact, monotone KL around that root.
    center = 0.5 * (torch.log(2.0 * kl_budget / slope_square.clamp_min(1e-30)))
    low, high = center - 6.0, center + 6.0
    for _ in range(bisections):
        middle = 0.5 * (low + high)
        over = tilted(middle.exp())[4].mean() > kl_budget
        low, high = torch.where(over, low, middle), torch.where(over, middle, high)
    step = low.exp()
    target_alpha, target_beta, statistics, entropy, kl = tilted(step)
    clamped = ((target_alpha <= MIN_CONCENTRATION) | (target_beta <= MIN_CONCENTRATION)).double().mean()
    return statistics, entropy, torch.stack((step, kl.mean(), clamped, slope_square))


def consequence_loss(agent, observations, native_actions, targets, valid):
    """Supervised consequences; a state change across an episode boundary is not one.

    targets are (state change, raw reward, terminal) in stored units.
    """
    model = agent.consequence
    changes, reward, terminal = model(observations, native_actions).split((model.observation_dim, 1, 1), dim=-1)
    target_changes, target_reward, target_terminal = targets.split((model.observation_dim, 1, 1), dim=-1)
    change_error = ((changes - target_changes / model.scale[:-1]) ** 2).mean(-1)
    change_loss = (change_error * valid).sum() / valid.sum().clamp_min(1.0)
    reward_loss = ((reward - target_reward / model.scale[-1]) ** 2).mean()
    terminal_loss = F.binary_cross_entropy_with_logits(terminal, target_terminal)
    return change_loss + reward_loss + terminal_loss, torch.stack(
        (change_loss.detach(), reward_loss.detach(), terminal_loss.detach()))


def residual_loss(agent, observations, scores, surprise, returns, args):
    """Least-squares surprise field with a state baseline, and the critic; no shared parameters."""
    baseline, slope = agent.residual(observations).split((1, scores.shape[-1]), dim=-1)
    surprise_loss = ((surprise - baseline.squeeze(-1) - (slope * scores).sum(-1)) ** 2).mean()
    v_loss = 0.5 * ((agent.critic(observations).view(-1) - returns) ** 2).mean()
    loss = surprise_loss + args.vf_coef * v_loss
    return loss, torch.stack((surprise_loss.detach(), v_loss.detach()))


@torch.no_grad()
def considered_gains(agent, observations, alpha, beta, native_actions, reward_divisor, gamma, count, chunk=8):
    """What the agent expects of the action it took against alternatives it would consider.

    Q~(s, a) is the predicted normalized reward plus the discounted value of the predicted next
    state. Returns own = Q~(s, a_taken) - mean_k Q~(s, a_k), the slope E_pi[Q~ z] estimated from
    the independent draws a_k, and the share of gain variance on half the draws that the slope
    fitted on the other half captures.
    """
    def expected_gain(actions):
        next_observations, reward, terminal = agent.consequence.consequences(observations.expand(len(actions), -1, -1), actions)
        return reward / reward_divisor + gamma * (1.0 - terminal) * agent.critic(next_observations).squeeze(-1)

    actions = Beta(alpha.float(), beta.float(), validate_args=False).sample((count,)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    scores = whitened_score(alpha, beta, actions.double())
    # Chunked: every considered action for every state at once is gigabytes of activations.
    considered = torch.cat([expected_gain(part) for part in actions.split(chunk)]).double()
    baseline = considered.mean(0)
    considered = considered - baseline
    slope = (considered[..., None] * scores).sum(0) / (count - 1)
    half = count // 2
    fitted = (considered[:half, :, None] * scores[:half]).mean(0)
    residual = considered[half:] - (fitted * scores[half:]).sum(-1)
    linear_share = 1.0 - residual.square().mean() / considered[half:].square().mean().clamp_min(1e-30)
    return expected_gain(native_actions[None])[0].double() - baseline, slope, linear_share


def actor_loss(agent, observations, target_statistics, target_entropy, old_statistics, old_entropy):
    """Exact cross-entropy to the tilted target; its gradient is the mean-statistic mismatch."""
    alpha, beta = (F.softplus(agent.actor(observations)) + 1.0).double().chunk(2, dim=-1)
    loss = beta_cross_entropy(alpha, beta, target_statistics).mean()
    with torch.no_grad():
        entropy = beta_cross_entropy(alpha, beta, beta_mean_statistics(alpha, beta)).mean()
        metrics = torch.stack((
            loss.detach() - target_entropy.mean(),
            beta_cross_entropy(alpha, beta, old_statistics).mean() - old_entropy.mean(),
            entropy + agent.log_action_scale.sum(),
        ))
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
    if args.consequence_batch % 2 or min(args.consequence_steps, args.memory_rollouts) <= 0 or args.considered_actions < 4:
        raise ValueError("invalid consequence batch, steps, memory or considered-action count")
    if args.model_epochs <= 0 or args.tilt_kl <= 0.0 or (args.tilt_step is not None and args.tilt_step <= 0.0):
        raise ValueError("model epochs, the tilt KL budget and the tilt step must be positive")
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
        agent = Agent(envs, args).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        actor_parameters, critic_parameters, consequence_parameters, residual_parameters = (
            tuple(network.parameters()) for network in (agent.actor, agent.critic, agent.consequence, agent.residual))
        # The predictions are fitted to the rollout before the actor moves, so they step separately.
        model_optimizer = optim.Adam(critic_parameters + residual_parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        consequence_optimizer = optim.Adam(consequence_parameters, lr=args.consequence_learning_rate, eps=1e-5, fused=True)
        actor_optimizer = optim.Adam(actor_parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value
        kl_budget = torch.tensor(args.tilt_kl, dtype=torch.float64, device=device)

        def rollout_statistics(observations, native):
            """Values and the old policy's FP64 geometry for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            alpha, beta = alpha.double(), beta.double()
            statistics = beta_mean_statistics(alpha, beta)
            entropy = beta_cross_entropy(alpha, beta, statistics)
            scores = whitened_score(alpha, beta, native.double()).float()
            return value.flatten(), alpha, beta, statistics, entropy, scores

        def target_model(alpha, beta, slope):
            budget = kl_budget
            if args.tilt_step is not None:
                # Least squares shrinks an unpredictive slope toward zero; a fixed step keeps
                # that shrinkage as a smaller move instead of renormalizing it to the full budget.
                budget = budget.clamp_max(0.5 * args.tilt_step ** 2 * slope.square().sum(-1).mean())
            return tilt_target(alpha, beta, slope, budget)

        def residual_loss_model(observations, scores, surprise, returns):
            return residual_loss(agent, observations, scores, surprise, returns, args)

        def consequence_loss_model(observations, native, targets, valid):
            return consequence_loss(agent, observations, native, targets, valid)

        def actor_loss_model(observations, target_statistics, target_entropy, old_statistics, old_entropy):
            return actor_loss(agent, observations, target_statistics, target_entropy, old_statistics, old_entropy)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            target_model = graph_compile(target_model)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            residual_loss_model = torch.compile(residual_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            consequence_loss_model = torch.compile(consequence_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            actor_loss_model = torch.compile(actor_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
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
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,),
                                           "raw_rewards": ()})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        epoch_updates = (args.batch_size + args.minibatch_size - 1) // args.minibatch_size
        model_metrics = torch.empty((args.model_epochs, 2), device=device)
        model_grad_norms = torch.empty((args.model_epochs, 2), device=device)
        consequence_metrics = torch.empty((args.consequence_steps, 3), device=device)
        # Remembered transitions: observation, native action, (state change, raw reward, terminal), valid.
        observation_dim = int(np.prod(obs_shape))
        memory = torch.empty((args.memory_rollouts * args.batch_size, 2 * observation_dim + agent.action_dim + 3), device=device)
        memory_fields = (observation_dim, agent.action_dim, observation_dim + 2, 1)
        b_environments = torch.arange(args.num_envs, device=device).repeat(args.num_steps)
        half = args.consequence_batch // 2
        actor_metrics = torch.empty((args.update_epochs * epoch_updates, 3), dtype=torch.float64, device=device)
        actor_grad_norms = torch.empty(args.update_epochs * epoch_updates, device=device)
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
            learning_rate = (1.0 - (iteration - 1.0) / args.num_iterations if args.anneal_lr else 1.0) * args.learning_rate
            model_optimizer.param_groups[0]["lr"] = learning_rate
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
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native,
                                  raw_rewards=raw_reward)
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
                b_values, b_alpha, b_beta, b_statistics, b_entropy, b_scores = rollout_statistics(b_obs, b_native)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_returns = returns.flatten().clone()
                b_advantages = advantages.flatten().clone()
                following = torch.cat((batch.fields["observations"][1:], next_obs[None])).flatten(0, 1)
                b_valid = (1.0 - torch.maximum(batch.terminations, batch.truncations)).flatten()
                raw_rewards = batch.fields["raw_rewards"].flatten()
                agent.consequence.calibrate(following - b_obs, raw_rewards, b_valid)
                # Observations keep the normalization they were collected under; rewards are kept raw
                # because their normalizer's scale moves with the policy.
                newest = ((iteration - 1) % args.memory_rollouts) * args.batch_size
                remembered = min(iteration, args.memory_rollouts) * args.batch_size
                memory[newest:newest + args.batch_size] = torch.cat(
                    (b_obs, b_native, following - b_obs, raw_rewards[:, None],
                     batch.terminations.flatten()[:, None], b_valid[:, None]), dim=-1)
                reward_divisor = torch.as_tensor(np.sqrt(rew_norm.variances + rew_norm.epsilon), dtype=torch.float32,
                                                 device=device)[b_environments]
            with timer.span("update"):
                with torch.no_grad():
                    # Expectations are formed before either predictor trains on this rollout: the
                    # same critic defines the advantages and the gains, and the consequence model
                    # has not seen the taken actions, so own + luck + surprise is the advantage
                    # exactly and luck has zero mean against this step's score.
                    expected = consequence_loss(agent, *memory[newest:newest + args.batch_size].split(memory_fields, dim=-1)[:3],
                                                b_valid)[1]
                    own, model_slope, linearity = considered_gains(
                        agent, b_obs, b_alpha, b_beta, b_native, reward_divisor, args.gamma, args.considered_actions)
                    own = own.float()
                    continuing = (1.0 - torch.maximum(batch.terminations, batch.truncations))[:-1]
                    # The compiled GAE reuses its output buffers, so read the cloned advantages.
                    deltas = b_advantages.view(args.num_steps, args.num_envs).clone()
                    deltas[:-1] -= args.gamma * args.gae_lambda * continuing * deltas[1:].clone()
                    resolution = torch.corrcoef(torch.stack((own, deltas.flatten())))[0, 1]
                    if not args.predict_own:
                        own, model_slope = torch.zeros_like(own), torch.zeros_like(model_slope)
                    surprise = gae_fn(
                        batch.rewards - own.view(args.num_steps, args.num_envs), values, batch.terminations,
                        batch.truncations, truncation_values, tail_value, args.gamma, args.gae_lambda,
                    )[0].flatten().clone()
                    scale = b_advantages.square().mean().sqrt()
                    b_surprise = surprise / scale
                    shares = torch.stack((own.square().mean(), (b_advantages - surprise - own).square().mean(),
                                          surprise.square().mean())) / scale.square()
                for update in range(args.consequence_steps):
                    indices = torch.cat((
                        newest + torch.randint(args.batch_size, (half,), device=device, generator=shuffle_generator),
                        torch.randint(remembered, (half,), device=device, generator=shuffle_generator)))
                    observations, native, targets, valid = memory[indices].split(memory_fields, dim=-1)
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    loss, metrics = consequence_loss_model(observations, native, targets, valid.squeeze(-1))
                    consequence_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(consequence_parameters, args.max_grad_norm, foreach=True)
                    consequence_optimizer.step()
                    consequence_metrics[update].copy_(metrics)
                for update in range(args.model_epochs):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    loss, metrics = residual_loss_model(b_obs, b_scores, b_surprise, b_returns)
                    model_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    model_grad_norms[update, 0] = nn.utils.clip_grad_norm_(residual_parameters, args.max_grad_norm, foreach=True)
                    model_grad_norms[update, 1] = nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
                    model_optimizer.step()
                    normalize_matrices()
                    model_metrics[update].copy_(metrics)
                with torch.no_grad():
                    surprise_slope = agent.residual(b_obs)[:, 1:].double()
                    model_slope = model_slope / scale
                    slope = model_slope + surprise_slope
                    b_target_statistics, b_target_entropy, tilt = target_model(b_alpha, b_beta, slope)
                    slope_squares = torch.stack((model_slope.square().sum(-1).mean(), surprise_slope.square().sum(-1).mean()))
                updates = 0
                # Each rollout's target is a new regression problem: momentum toward the last
                # target is dropped, and the step decays to zero so the fit settles on this one
                # instead of travelling a fixed Adam distance past it.
                for state in actor_optimizer.state.values():
                    state["exp_avg"].zero_()
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        actor_optimizer.param_groups[0]["lr"] = 0.5 * learning_rate * (
                            1.0 + math.cos(math.pi * updates / len(actor_grad_norms)))
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = actor_loss_model(
                            b_obs[indices], b_target_statistics[indices], b_target_entropy[indices],
                            b_statistics[indices], b_entropy[indices])
                        actor_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        actor_grad_norms[updates] = nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm, foreach=True)
                        actor_optimizer.step()
                        normalize_matrices()
                        actor_metrics[updates].copy_(metrics)
                        updates += 1

            # Standardized targets: 1 - loss is an R^2. "expected" was scored on this rollout before
            # any step on it: a test of what the agent expected to happen.
            surprise_square = b_surprise.square().mean()
            last_actor = actor_metrics[-epoch_updates:].mean(0)
            logged = gather_metrics({
                "losses/value_loss": model_metrics[-1, 1],
                "losses/entropy": last_actor[2], "losses/approx_kl": last_actor[1],
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "consequence/change_r2_heldout": 1.0 - expected[0],
                "consequence/reward_r2_heldout": 1.0 - expected[1],
                "consequence/change_r2_fit": 1.0 - consequence_metrics[-1, 0],
                "consequence/terminal_loss": consequence_metrics[-1, 2],
                "credit/own_share": shares[0], "credit/luck_share": shares[1], "credit/surprise_share": shares[2],
                "credit/own_delta_correlation": resolution,
                "surprise/r2_heldout": 1.0 - model_metrics[0, 0] / surprise_square,
                "surprise/r2_fit": 1.0 - model_metrics[-1, 0] / surprise_square,
                "slope/linear_share": linearity, "slope/square": tilt[3],
                "slope/model_square": slope_squares[0], "slope/surprise_square": slope_squares[1],
                "tilt/step": tilt[0], "tilt/target_kl": tilt[1], "tilt/clamp_fraction": tilt[2],
                "tilt/residual_kl": last_actor[0],
                "tilt/initial_kl": actor_metrics[:epoch_updates, 0].mean(),
                "grad/actor_preclip_norm": actor_grad_norms.mean(),
                "grad/residual_preclip_norm": model_grad_norms[:, 0].mean(),
                "grad/critic_preclip_norm": model_grad_norms[:, 1].mean(),
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", learning_rate, global_step)
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
                run_name=f"{run_name}-eval", Model=lambda envs: Agent(envs, args), device=device, gamma=args.gamma,
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

# Critic uncertainty for PPO, v1. Base: awr/ppo_continuous_action_ppo_sepclip_control_v1.py
# with its 50M reference flags as defaults (no adv norm, no value clip, symmetric 0.2 clip).
#
# The critic is a K-member ensemble, each fit on a Poisson(1) bootstrap reweighting of the
# rollout; V is the ensemble mean. A separate variance net regresses, via variance NLL
# (log v + target / v, minimized at v = E[target | s]), two per-state variances:
#   total(s)     ~ E[A_hat^2 | s]           noise of the lambda-return around V
#   epistemic(s) ~ E[Var_k V_k(s) | s]      disagreement, i.e. how wrong V itself may be
# Distilling the raw K-sample variance smooths its chi-square(K-1) noise across states.
#
# --uncertainty-mode lambda: per-state GAE lambda. Mixing a bootstrap V(s') (error e^2) with
# the continued return (noise sigma^2) is inverse-variance weighted at lambda = e^2/(e^2+sigma^2).
# total ~ sigma^2 + e^2, so with rho = epistemic/total that is odds(lambda) = rho/(1-rho).
# The epistemic scale is uncalibrated, so keep the per-state odds shape (beta=1) and fix the
# level: a scalar logit offset, solved per rollout, holds mean lambda at lambda0. This tests
# where lambda is spent, not how much. The total target uses a fixed-lambda0 GAE pass so the
# variance net never regresses on advantages its own lambda shaped.
# Hypothesis: bootstrap where V is trusted (variance cut), stay Monte-Carlo where it isn't (bias cut).
# --uncertainty-mode weight: mean-1 state weights ~ total(s)^-p on the policy loss (ablation;
# state-only weights keep each state's ascent direction, action-dependent ones would not).
# --uncertainty-mode none: ensemble-mean critic at fixed lambda, the control for both.
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
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """lower policy ratio clipping coefficient"""
    clip_coef_upper: float = 0.2
    """upper policy ratio clipping coefficient"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # Critic uncertainty
    uncertainty_mode: str = "lambda"
    """none | lambda (per-state GAE lambda) | weight (per-state policy-loss weights)"""
    ensemble_size: int = 4
    """number of bootstrapped critic members"""
    match_base_init: bool = False
    """build critic 0 then the actor first, so both match the base's seed-1 initialization"""
    bootstrap: bool = True
    """Poisson(1) bootstrap weights per member; off trains every member on the full rollout"""
    lambda_temperature: float = 1.0
    """beta in logit lambda(s) = beta * log odds(epistemic/total) + offset (mean held at gae_lambda)"""
    lambda_min: float = 0.5
    """lower clamp on per-state lambda"""
    lambda_max: float = 0.995
    """upper clamp on per-state lambda"""
    weight_power: float = 1.0
    """p in w(s) ~ total(s)^-p; 1 is inverse-variance weighting"""
    weight_max: float = 4.0
    """state weights are clamped to [1/weight_max, weight_max] before mean-1 renormalization"""

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
    def __init__(self, envs, ensemble_size=4, match_base_init=False):
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
        def critic():
            return ResidualMLP(observation_dim, 1, output_std=1.0)

        def actor():
            return ResidualMLP(observation_dim, 2 * self.action_dim, output_std=0.01)

        # The base draws its critic then its actor; optionally replay that order.
        first, early_actor = (critic(), actor()) if match_base_init else (None, None)
        self.critics = nn.ModuleList(([first] if first is not None else [])
                                     + [critic() for _ in range(ensemble_size - (first is not None))])
        # Outputs (log total variance, log epistemic variance); small readout starts both at 1.
        self.variance = ResidualMLP(observation_dim, 2, output_std=0.01)
        self.actor = early_actor if early_actor is not None else actor()

    @torch.no_grad()
    def normalize_matrices(self):
        """Match nGPT's matrix axes without changing Adam moments or biases."""
        for trunk in (self.actor, self.variance, *self.critics):
            for stage in (trunk.first, trunk.second):
                branch = stage[0]
                for weight, dim in ((branch.gate.weight, 1),
                                    (branch.up.weight, 1),
                                    (branch.down.weight, 0)):
                    weight.div_(torch.linalg.vector_norm(weight, dim=dim, keepdim=True))
            weight = trunk.head[0].weight
            weight.div_(torch.linalg.vector_norm(weight, dim=1, keepdim=True))

    def member_values(self, x):
        return torch.cat([critic(x) for critic in self.critics], dim=-1)

    def get_value(self, x):
        return self.member_values(x).mean(-1, keepdim=True)

    def get_policy(self, x):
        return (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)

    def get_policy_and_value(self, x):
        alpha, beta = self.get_policy(x)
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


def variance_nll(log_variance, target):
    """Minimized at exp(log_variance) = E[target | s] for any nonnegative target."""
    return log_variance + target * torch.exp(-log_variance)


def mean_matched_lambda(log_odds, target, low, high, steps=40):
    """sigmoid(log_odds + c) clamped to [low, high], with scalar c set so the mean is target.

    The clamped mean is monotone in c, so a fixed-length bisection is exact to
    ~1e-10 in c and stays sync-free. The bracket covers any finite log-odds batch.
    """
    centre = log_odds.mean()
    lower = torch.logit(torch.tensor(target, device=log_odds.device)) - centre - 50.0
    upper = lower + 100.0
    for _ in range(steps):
        middle = 0.5 * (lower + upper)
        above = torch.sigmoid(log_odds + middle).clamp(low, high).mean() > target
        lower, upper = torch.where(above, lower, middle), torch.where(above, middle, upper)
    return torch.sigmoid(log_odds + 0.5 * (lower + upper)).clamp(low, high)


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns,
             state_weights, bootstrap_weights, total_targets, epistemic_targets, args):
    """Clipped PPO on native Beta samples, bootstrapped ensemble regression, variance NLL."""
    alpha, beta = agent.get_policy(observations)
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
    pg_loss = (state_weights * torch.max(pg_loss1, pg_loss2)).mean()
    # Per member, the base critic loss under its own bootstrap reweighting.
    squared_errors = (agent.member_values(observations) - returns.unsqueeze(-1)) ** 2
    member_v_losses = 0.5 * (bootstrap_weights * squared_errors).mean(0)
    v_loss = member_v_losses.sum()
    log_total, log_epistemic = agent.variance(observations).unbind(-1)
    variance_loss = (variance_nll(log_total, total_targets)
                     + variance_nll(log_epistemic, epistemic_targets)).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + variance_loss
    metrics = torch.stack((pg_loss.detach(), member_v_losses.detach().mean(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac, variance_loss.detach()))
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
    if args.uncertainty_mode not in {"none", "lambda", "weight"}:
        raise ValueError("uncertainty_mode must be none, lambda or weight")
    if args.ensemble_size < 2:
        raise ValueError("epistemic variance needs at least two critic members")
    if not 0.0 <= args.lambda_min <= args.gae_lambda <= args.lambda_max <= 1.0:
        raise ValueError("require 0 <= lambda_min <= gae_lambda <= lambda_max <= 1")
    if args.uncertainty_mode == "lambda" and not 0.0 < args.gae_lambda < 1.0:
        raise ValueError("per-state lambda needs a base lambda strictly inside (0, 1)")
    if args.weight_max < 1.0:
        raise ValueError("weight_max must be at least 1")
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
        writer.add_text("critic", f"{args.ensemble_size}-member Poisson-bootstrap ensemble; variance net (total, epistemic); mode={args.uncertainty_mode}")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args.ensemble_size, args.match_base_init).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        # Every network keeps its own max_grad_norm budget, each critic member included,
        # so a member trains exactly as the single base critic would on its bootstrap.
        clip_groups = (tuple(agent.actor.parameters()), tuple(agent.variance.parameters()),
                       *(tuple(critic.parameters()) for critic in agent.critics))
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities, member values and variance predictions in one forward."""
            alpha, beta = agent.get_policy(observations)
            return (agent.member_values(observations), agent.action_logprob(alpha, beta, native),
                    agent.variance(observations))

        def loss_model(observations, native, old_logprobs, advantages, returns,
                       state_weights, bootstrap_weights, total_targets, epistemic_targets):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns,
                            state_weights, bootstrap_weights, total_targets, epistemic_targets, args)

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
        bootstrap_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        bootstrap_rates = torch.ones((args.batch_size, args.ensemble_size), device=device)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 7), device=device)
        grad_norms = torch.empty((max_updates, len(clip_groups)), device=device)
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
                b_members, b_logprobs, b_log_variances = rollout_statistics(b_obs, b_native)
                b_values = b_members.mean(-1)
                b_epistemic = b_members.var(-1)
                log_total, log_epistemic = b_log_variances.unbind(-1)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                rho = (log_epistemic - log_total).exp().clamp(1e-6, 1.0 - 1e-6)
                log_odds = rho.log() - torch.log1p(-rho)
                state_lambda = mean_matched_lambda(
                    args.lambda_temperature * log_odds, args.gae_lambda, args.lambda_min, args.lambda_max,
                ).view(args.num_steps, args.num_envs)
                gae_inputs = (batch.rewards, values, batch.terminations, batch.truncations,
                              truncation_values, tail_value, args.gamma)
                advantages, returns = gae_fn(*gae_inputs, args.gae_lambda)
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
                b_total_targets = b_advantages.square()
                if args.uncertainty_mode == "lambda":
                    # Row t continues into state t+1, so it carries that state's lambda.
                    gae_lambda = torch.cat((state_lambda[1:], state_lambda[-1:]))
                    advantages, returns = gae_fn(*gae_inputs, gae_lambda)
                    b_advantages = advantages.flatten().clone()
                    b_returns = returns.flatten().clone()
                state_weights = torch.exp(-args.weight_power * (log_total - log_total.mean()))
                state_weights = state_weights.clamp(1.0 / args.weight_max, args.weight_max)
                state_weights = state_weights / state_weights.mean()
                if args.uncertainty_mode != "weight":
                    state_weights = torch.ones_like(state_weights)
                bootstrap_weights = (torch.poisson(bootstrap_rates, generator=bootstrap_generator)
                                     if args.bootstrap else bootstrap_rates)
                uncertainty_metrics = {
                    "uncertainty/lambda_mean": state_lambda.mean(),
                    "uncertainty/lambda_p10": state_lambda.flatten().quantile(0.1),
                    "uncertainty/lambda_p90": state_lambda.flatten().quantile(0.9),
                    "uncertainty/log_odds_std": log_odds.std(),
                    "uncertainty/epistemic_mean": b_epistemic.mean(),
                    "uncertainty/pred_epistemic_mean": log_epistemic.exp().mean(),
                    "uncertainty/adv_square_mean": b_total_targets.mean(),
                    "uncertainty/pred_total_mean": log_total.exp().mean(),
                    # Rank agreement of predicted and realized per-state noise: is total(s) informative?
                    "uncertainty/total_log_corr": torch.corrcoef(torch.stack(
                        (log_total, b_total_targets.clamp_min(1e-12).log())))[0, 1],
                    "uncertainty/weight_max": state_weights.max(),
                }
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], state_weights[indices],
                            bootstrap_weights[indices], b_total_targets[indices], b_epistemic[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        for group, parameters in enumerate(clip_groups):
                            grad_norms[updates, group] = nn.utils.clip_grad_norm_(
                                parameters, args.max_grad_norm, foreach=True)
                        optimizer.step()
                        normalize_matrices()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/variance_loss": last[6],
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "grad/actor_preclip_norm": grad_norms[:updates, 0].mean(),
                "grad/variance_preclip_norm": grad_norms[:updates, 1].mean(),
                "grad/critic_preclip_norm": grad_norms[:updates, 2:].mean(),
                **uncertainty_metrics,
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name not in ("losses/explained_variance", "uncertainty/total_log_corr")):
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

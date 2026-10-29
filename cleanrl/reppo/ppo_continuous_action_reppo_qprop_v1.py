# REPPO-critic Q-Prop v1. Base: awr/ppo_continuous_action_ppo_sepclip_control_v1.py (run as
# noadv / novalueclip / symmetric 0.2 clip, now the defaults) plus REPPO's Q(s, a) critic.
# Standalone REPPO plateaued near 5k on HalfCheetah at 8M (PPO: 8.4k) with a well-fit critic and
# a held KL bound: its only actor signal, grad_a Q, is biased by extrapolation around one sampled
# action per state. Earlier pathwise mixes (IPG) paid that bias too; Q-Prop does not.
# Q-Prop with REPPO's critic: g = grad_a Q(s, mu_old) at the rollout-time Beta mean,
#   A_bar = g^T (a - mu_old)                        zero mean under pi_old (control variate)
#   surrogate = clip(ratio) (A - eta A_bar) + eta pessimistic_clip(g^T (mu_new - mu_old))
# The gradient is unbiased for any g and eta; eta* = max(0, cov(A, A_bar) / var(A_bar)) is the
# variance-optimal batch weight, so a useless Q switches itself off and the scale of Q (raw
# rewards x 0.1 on an HL-Gauss support) never has to match PPO's normalized advantages.
# Q is REPPO's (HL-Gauss categorical, lambda-returns from sampled next actions Q(s', a' ~ pi),
# self-predictive next-feature loss) with its own optimizer and epochs. g is taken before Q
# trains on the batch, so it cannot fit this batch's noise.
# Diagnostics: corr(A, A_bar) (first-order signal) and corr(A, Q(s, a) - E_pi Q(s, .)) (full Q).
# Earlier Q-Prop (latent_qprop_v1-v5) used first-order g heads whose A_bar explained ~0% of A.
# Hypothesis: REPPO's full Q carries usable action signal (it trains a 5k policy on its own);
# as a control variate it lowers PPO's gradient variance without importing its bias.
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
from cleanrl.shared.hl_gauss import HLGaussConfig
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
ZERO_DIST_SCALE = 40.9  # REPPO's fixed multiplier on the learned zero-value logit offset
PPO_METRICS = ("policy_loss", "value_loss", "entropy", "old_approx_kl", "approx_kl", "clipfrac",
               "analytic", "analytic_clipfrac")
Q_METRICS = ("q_ce", "q_aux", "q_mse")

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

    # Q-Prop
    qprop: bool = True
    """use the control variate and analytic term; off = PPO with the Q critic trained for diagnostics"""
    eta_mode: str = "optimal"
    """optimal: eta = max(0, cov/var); conservative: eta = 1 iff cov > 0"""
    fullq_samples: int = 4
    """policy samples per state estimating E_pi Q(s, .) for the full-Q diagnostic"""

    # REPPO Q critic
    q_learning_rate: float = 3e-4
    """AdamW learning rate of the Q critic (REPPO's)"""
    q_epochs: int = 4
    """Q critic epochs per rollout"""
    q_minibatches: int = 8
    """Q critic minibatches per epoch"""
    q_lambda: float = 0.95
    """lambda of the Q critic's lambda-return target"""
    reward_scale: float = 0.1
    """fixed reward multiplier for Q (keeps the categorical support stationary)"""
    q_hidden_dim: int = 512
    """Q critic width"""
    num_bins: int = 151
    """HL-Gauss categories"""
    v_min: float = -50.0
    """lowest supported scaled return"""
    v_max: float = 250.0
    """highest supported scaled return"""
    aux_loss_coef: float = 1.0
    """weight of the next-feature prediction loss"""

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
    q_minibatch_size: int = 0
    """the Q critic mini-batch size (computed in runtime)"""
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


def normed_layer(in_dim, out_dim):
    return [nn.Linear(in_dim, out_dim), nn.RMSNorm(out_dim, eps=1e-5), nn.SiLU()]


class QCritic(nn.Module):
    """REPPO's Q(s, a) on physical actions: encoder -> {categorical value head, next-feature predictor}."""

    def __init__(self, observation_dim, action_dim, args, device):
        super().__init__()
        width = args.q_hidden_dim
        self.encoder = nn.Sequential(*normed_layer(observation_dim + action_dim, width), nn.Linear(width, width))
        self.value_head = nn.Sequential(nn.SiLU(), *normed_layer(width, width), nn.Linear(width, args.num_bins))
        self.predictor = nn.Sequential(nn.SiLU(), *normed_layer(width, width), nn.Linear(width, width))
        self.hl = HLGaussConfig(v_min=args.v_min, v_max=args.v_max, num_bins=args.num_bins).build(device)
        self.zero_dist = nn.Parameter(self.hl.project(torch.zeros((), device=device)))

    def features(self, observations, actions):
        return self.encoder(torch.cat((observations, actions), dim=-1))

    def logits(self, features):
        return self.value_head(features) + ZERO_DIST_SCALE * self.zero_dist

    def value(self, observations, actions):
        return self.hl.to_scalar(self.logits(self.features(observations, actions)))


def beta_moments(alpha, beta):
    total = alpha + beta
    return alpha / total, (alpha * beta / (total.square() * (total + 1.0))).sqrt()


def lambda_returns(gae_fn, rewards, next_values, terminations, truncations, gamma, lam):
    """G_t = r_t + gamma [(1 - lam) Q'_t + lam G_{t+1}] via the explicit-next-value GAE kernel.

    Q'_t = Q(s_{t+1}, a' ~ pi). Boundaries (and the rollout end) bootstrap from Q'_t alone;
    terminations do not bootstrap. The kernel returns A_t + v_t with
    A_t = r_t + gamma Q'_t - v_t + gamma lam A_{t+1}, which is exactly G_t when v_{t+1} = Q'_t:
    set v_t = Q'_{t-1} (v_0 cancels).
    """
    values = torch.cat((torch.zeros_like(next_values[:1]), next_values[:-1]))
    _, returns = gae_fn(rewards, values, terminations, truncations, next_values, gamma, lam)
    return returns


def centered_corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return (x * y).mean() / (x.square().mean() * y.square().mean()).sqrt().clamp_min(1e-12)


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values,
             old_means, old_stds, gradients, control, eta, args):
    """Q-Prop PPO on native Beta samples; control = A_bar from the rollout-time g (native units)."""
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
    residual = advantages - eta * control
    pg_loss1 = -residual * ratio
    pg_loss2 = -residual * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef_upper)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    # Analytic term E_new[A_bar] = g^T (mu_new - mu_old), pessimistically clipped per dimension
    # at clip_coef old standard deviations: no gain from moving the mean further than that.
    mean_shift = beta_moments(alpha, beta)[0] - old_means
    radius = args.clip_coef * old_stds
    gain = torch.minimum(gradients * mean_shift, gradients * torch.clamp(mean_shift, -radius, radius))
    analytic = gain.sum(-1).mean()
    pg_loss = pg_loss - eta * analytic
    with torch.no_grad():
        analytic_clipfrac = (mean_shift.abs() > radius).float().mean()
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
                           old_approx_kl, approx_kl, clipfrac, analytic.detach(), analytic_clipfrac))
    return loss, metrics


def q_loss(critic, observations, actions, labels, next_features, continuing, targets, aux_coef):
    """REPPO's critic loss: HL-Gauss cross-entropy plus masked next-feature prediction."""
    features = critic.features(observations, actions)
    logits = critic.logits(features)
    ce = -(labels * logits.log_softmax(dim=-1)).sum(-1).mean()
    prediction_error = (critic.predictor(features) - next_features).square().mean(-1)
    aux = (continuing * prediction_error).mean()
    q_mse = (critic.hl.to_scalar(logits.detach()) - targets).square().mean()
    return ce + aux_coef * aux, torch.stack((ce.detach(), aux.detach(), q_mse))


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
    if min(args.q_epochs, args.q_minibatches, args.fullq_samples) <= 0 or args.batch_size % args.q_minibatches:
        raise ValueError("Q epochs/samples must be positive and q_minibatches must divide the batch")
    args.q_minibatch_size = args.batch_size // args.q_minibatches
    if args.eta_mode not in {"optimal", "conservative"}:
        raise ValueError("eta_mode must be optimal or conservative")
    if not args.v_min < 0.0 < args.v_max:
        raise ValueError("the Q support must contain zero (the critic's initial prediction)")
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
        obs_shape = envs.single_observation_space.shape
        observation_dim, action_dim = int(np.prod(obs_shape)), agent.action_dim
        critic_q = QCritic(observation_dim, action_dim, args, device).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        q_optimizer = optim.AdamW(critic_q.parameters(), lr=args.q_learning_rate, fused=True)
        actor_parameters, critic_parameters = tuple(agent.actor.parameters()), tuple(agent.critic.parameters())
        q_parameters = tuple(critic_q.parameters())
        value_model = agent.get_value
        action_low, action_scale = agent.action_low, agent.action_scale

        def policy_parameters(observations, next_observations):
            """Beta parameters at the rollout states and their successors, in one forward."""
            both = torch.cat((observations, next_observations))
            alpha, beta = (F.softplus(agent.actor(both)) + 1.0).chunk(2, dim=-1)
            return alpha, beta

        def rollout_statistics(observations, native, alpha, beta, next_observations, next_native, samples):
            """Everything the update reads from the rollout-time networks, in one pass.

            V and old log-probabilities for PPO; Beta moments, g = grad_a Q at the mean (native
            units) and A_bar for Q-Prop; Q's next-state values and features for its targets; and
            the full-Q advantage Q(s, a) - E_pi Q(s, .) for the diagnostic.
            """
            value = agent.critic(observations).flatten()
            logprob = agent.action_logprob(alpha, beta, native)
            mean, std = beta_moments(alpha, beta)
            physical_mean = action_low + action_scale * mean
            gradient = torch.func.grad(lambda a: critic_q.value(observations, a).sum())(physical_mean)
            gradient = gradient * action_scale
            control = (gradient * (native - mean)).sum(-1)
            data_q = critic_q.value(observations, action_low + action_scale * native)
            sample_q = critic_q.value(observations.expand(samples.shape[0], -1, -1),
                                      action_low + action_scale * samples).mean(0)
            next_features = critic_q.features(next_observations, action_low + action_scale * next_native)
            next_q = critic_q.hl.to_scalar(critic_q.logits(next_features))
            return (value, logprob, mean, std, gradient, control, data_q, data_q - sample_q,
                    next_q, next_features)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values,
                       old_means, old_stds, gradients, control, eta):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values,
                            old_means, old_stds, gradients, control, eta, args)

        def q_loss_model(observations, actions, labels, next_features, continuing, targets):
            return q_loss(critic_q, observations, actions, labels, next_features, continuing, targets,
                          args.aux_loss_coef)

        if args.compile:
            policy_parameters = graph_compile(policy_parameters)
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            q_loss_model = torch.compile(q_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        q_gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode, explicit_next_values=True)
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = ResidualHostMirror(agent.actor, args.num_envs)
        host_low, host_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, host_low, host_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   store_transition_observations=True,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,),
                                           "q_rewards": ()})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, len(PPO_METRICS)), device=device)
        grad_norms = torch.empty((max_updates, 2), device=device)
        q_sums = torch.zeros(len(Q_METRICS) + 1, device=device)
        q_updates = args.q_epochs * args.q_minibatches
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
                    transfer.push(step, reward, terms, truncs, transition_observations=transition_obs,
                                  observations=obs_step, native_actions=native,
                                  q_rewards=np.asarray(raw_reward) * args.reward_scale)
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
                b_next_obs = batch.transition_observations.flatten(0, 1)
                alpha, beta = policy_parameters(b_obs, b_next_obs)
                policy = Beta(alpha, beta, validate_args=False)
                draws = policy.sample((args.fullq_samples,)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                samples, next_native = draws[:, : args.batch_size], draws[0, args.batch_size :]
                (b_values, b_logprobs, b_means, b_stds, b_gradients, b_control, data_q, fullq_advantage,
                 next_q, next_features) = rollout_statistics(
                    b_obs, b_native, alpha[: args.batch_size], beta[: args.batch_size],
                    b_next_obs, next_native, samples)
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
                # Variance-optimal control-variate weight; any eta leaves the gradient unbiased.
                centered_advantages = b_advantages - b_advantages.mean()
                centered_control = b_control - b_control.mean()
                covariance = (centered_advantages * centered_control).mean()
                if args.eta_mode == "optimal":
                    eta = (covariance / centered_control.square().mean().clamp_min(1e-12)).clamp_min(0.0)
                else:
                    eta = (covariance > 0).float()
                if not args.qprop:
                    eta = torch.zeros_like(eta)
                q_targets = lambda_returns(
                    q_gae_fn, batch.fields["q_rewards"], next_q.view(args.num_steps, args.num_envs),
                    batch.terminations, batch.truncations, args.gamma, args.q_lambda,
                ).flatten().clone()
                q_labels = critic_q.hl.project(q_targets)
                b_q_actions = action_low + action_scale * b_native
                b_continuing = 1.0 - batch.terminations.flatten()
                diagnostics = {
                    "qprop/eta": eta,
                    "qprop/control_corr": centered_corr(b_advantages, b_control),
                    "qprop/fullq_corr": centered_corr(b_advantages, fullq_advantage),
                    "qprop/control_std_ratio": eta * b_control.std() / b_advantages.std().clamp_min(1e-12),
                    "qprop/gradient_norm": b_gradients.norm(dim=-1).mean(),
                    "q/explained_variance": explained_variance(data_q, q_targets),
                    "q/target_mean": q_targets.mean(),
                    "q/target_max": q_targets.max(),
                    "q/support_clip_frac": ((q_targets <= args.v_min) | (q_targets >= args.v_max)).float().mean(),
                }
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                            b_means[indices], b_stds[indices], b_gradients[indices], b_control[indices], eta,
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
            with timer.span("q_update"):
                # Q trains after g was taken, so A_bar never saw this batch's returns.
                q_sums.zero_()
                for epoch in range(args.q_epochs):
                    for indices in device_minibatches(args.batch_size, args.q_minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = q_loss_model(
                            b_obs[indices], b_q_actions[indices], q_labels[indices],
                            next_features[indices], b_continuing[indices], q_targets[indices],
                        )
                        q_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        q_sums[-1] += nn.utils.clip_grad_norm_(q_parameters, args.max_grad_norm, foreach=True)
                        q_optimizer.step()
                        q_sums[:-1] += metrics

            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "qprop/analytic": update_metrics[:updates, 6].mean(),
                "qprop/analytic_clipfrac": update_metrics[:updates, 7].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "grad/actor_preclip_norm": grad_norms[:updates, 0].mean(),
                "grad/critic_preclip_norm": grad_norms[:updates, 1].mean(),
                "grad/q_preclip_norm": q_sums[-1] / q_updates,
                **{f"q/{name}": q_sums[i] / q_updates for i, name in enumerate(Q_METRICS)},
                **diagnostics,
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if not name.endswith("explained_variance")):
                raise FloatingPointError("nonfinite Q-Prop learner metrics")
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

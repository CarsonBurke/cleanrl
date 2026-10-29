# Calibrated surprise v6: the PPO control, reinforced only by what its own predictions leave unexplained.
# v1-v5 moved the policy by fitting it to a closed-form tilted target; every one lost to the control
# because ten to thirty Adam steps close only ~20% of the distance to a target, and v4's full-weight
# V(s') made predicted own effects ~6x larger than what returns confirmed. This version keeps the
# control's clipped update exactly and changes only the signal:
# - own(s, a) = c_r * [r(s, a) - E_pi r] + c_v * [gamma V(s'(s, a)) - E_pi gamma V(s')], from the
#   consequence model over considered actions. Any per-state-centred own leaves the gradient unbiased.
# - (c_r, c_v) are refit every rollout by least squares so the discounted sum of own effects explains
#   as much of the GAE advantage as it really does: how far the agent's predictions of its own
#   behaviour are trusted is measured, not assumed. GAE(lambda) itself trusts V(s') by only 1 - lambda.
# - taken actions are reinforced by the surprise A - sum_k (gamma lambda)^k own_{t+k}: their own effect
#   and the luck of later actions are removed.
# - the own effect returns as an expectation: the same clipped surrogate over considered actions,
#   weighted by their predicted own effect, with no sampled return.
# With --no-predict-own this is the control. Hypothesis: calibrated predictions explain a large share
# of advantage variance on HalfCheetah, so the same batch gives a cleaner direction.
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
from cleanrl.shared.linear_scan import reverse_linear_scan
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
    predict_own: bool = True
    """ablation: without it the advantage is all surprise and this is the PPO control"""
    considered_actions: int = 32
    """actions per state whose predicted consequences are compared"""
    consequence_steps: int = 128
    """consequence-model steps per rollout; it is plain supervised regression on memory"""
    consequence_batch: int = 4096
    """half drawn from the newest rollout, half from all remembered transitions"""
    consequence_learning_rate: float = 0.001
    """constant: the predictions must not freeze while the policy still moves"""
    consequence_width: int = 512
    """hidden width of the consequence model"""
    memory_rollouts: int = 64
    """rollouts of transitions remembered for the consequence model"""

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
    def __init__(self, envs, consequence_width=512):
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
        self.consequence = ConsequenceModel(observation_dim, self.action_dim, consequence_width)

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


@torch.no_grad()
def considered_gains(agent, observations, alpha, beta, native_actions, reward_divisor, gamma, count, chunk=8):
    """Predicted gains of the taken action against alternatives the old policy would consider.

    A gain has two parts, (normalized reward, discounted value of the next state), kept separate
    so their trust can be calibrated. Returns the considered actions (N, count, A), their old
    log-probabilities (N, count), the taken action's centred gain (N, 2) and the considered
    actions' centred gains (N, count, 2); centring uses the considered draws only, so the taken
    action's own effect is independent of its baseline.
    """
    distribution = Beta(alpha, beta, validate_args=False)
    actions = distribution.sample((count,)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    logprobs = (distribution.log_prob(actions) - agent.log_action_scale).sum(-1)

    def gains(part):
        next_observations, reward, terminal = agent.consequence.consequences(observations.expand(len(part), -1, -1), part)
        return torch.stack((reward / reward_divisor,
                            gamma * (1.0 - terminal) * agent.critic(next_observations).squeeze(-1)), dim=-1)

    # Chunked: every considered action for every state at once is gigabytes of activations.
    gain = torch.cat([gains(part) for part in torch.cat((native_actions[None], actions)).split(chunk)])
    baseline = gain[1:].mean(0)
    considered = (gain[1:] - baseline) * (count / (count - 1.0))
    return actions.transpose(0, 1), logprobs.transpose(0, 1), gain[0] - baseline, considered.transpose(0, 1)


@torch.no_grad()
def calibrate_own(advantages, own, continuing, discount):
    """Least-squares trust in each part of the own effect, and the surprise it leaves.

    advantages (T, N), own (T, N, P) and continuing (T, N) are time-major. The discounted sum of
    own effects over the rest of the episode is this step's own effect plus later luck; the
    coefficients make that sum explain as much of the advantage as it can.
    Returns (coefficients (P,), surprise (T, N), explained share of advantage square).
    """
    coefficients = (discount * continuing)[:, :, None].expand_as(own)
    sums = reverse_linear_scan(own.flatten(1).contiguous(), coefficients.flatten(1).contiguous()).view(own.shape)
    design, target = sums.flatten(0, 1).double(), advantages.flatten().double()
    gram = design.T @ design
    ridge = 1e-6 * torch.diagonal(gram).mean() * torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    solved = torch.linalg.solve(gram + ridge, design.T @ target)
    surprise = target - design @ solved
    explained = 1.0 - surprise.square().mean() / target.square().mean().clamp_min(1e-30)
    return solved.float(), surprise.float().view(advantages.shape), explained


def surprise_ppo_loss(agent, observations, native_actions, old_logprobs, surprise, returns, old_values,
                      considered_actions, considered_logprobs, considered_own, args):
    """The control's clipped loss on the surprise, plus the same surrogate in expectation over
    considered actions weighted by their predicted own effect."""
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
        surprise = (surprise - surprise.mean()) / (surprise.std() + 1e-8)
    pg_loss = torch.max(-surprise * ratio,
                        -surprise * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef_upper)).mean()
    considered_distribution = Beta(alpha[:, None], beta[:, None], validate_args=False)
    considered_ratio = ((considered_distribution.log_prob(considered_actions) - agent.log_action_scale).sum(-1)
                        - considered_logprobs).exp()
    own_loss = torch.max(
        -considered_own * considered_ratio,
        -considered_own * torch.clamp(considered_ratio, 1 - args.clip_coef, 1 + args.clip_coef_upper)).mean()
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss + own_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac, own_loss.detach()))
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
    if args.consequence_batch % 2 or min(args.consequence_steps, args.memory_rollouts) <= 0 or args.considered_actions < 2:
        raise ValueError("invalid consequence batch, steps, memory or considered-action count")
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
        agent = Agent(envs, args.consequence_width).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        actor_parameters, critic_parameters = tuple(agent.actor.parameters()), tuple(agent.critic.parameters())
        consequence_parameters = tuple(agent.consequence.parameters())
        optimizer = optim.Adam(actor_parameters + critic_parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        consequence_optimizer = optim.Adam(consequence_parameters, lr=args.consequence_learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old policy, log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native), alpha, beta

        def loss_model(observations, native, old_logprobs, surprise, returns, old_values,
                       considered_actions, considered_logprobs, considered_own):
            return surprise_ppo_loss(agent, observations, native, old_logprobs, surprise, returns, old_values,
                                     considered_actions, considered_logprobs, considered_own, args)

        def consequence_loss_model(observations, native, targets, valid):
            return consequence_loss(agent, observations, native, targets, valid)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            consequence_loss_model = torch.compile(consequence_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
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
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 7), device=device)
        consequence_metrics = torch.empty((args.consequence_steps, 3), device=device)
        # Remembered transitions: observation, native action, (state change, raw reward, terminal), valid.
        observation_dim = int(np.prod(obs_shape))
        memory = torch.empty((args.memory_rollouts * args.batch_size, 2 * observation_dim + agent.action_dim + 3), device=device)
        memory_fields = (observation_dim, agent.action_dim, observation_dim + 2, 1)
        b_environments = torch.arange(args.num_envs, device=device).repeat(args.num_steps)
        half = args.consequence_batch // 2
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
                # Expectations are formed before the consequence model or the critic train on this
                # rollout, so the taken action's own effect is a prediction, not a fit.
                following = torch.cat((batch.fields["observations"][1:], next_obs[None])).flatten(0, 1)
                continuing = 1.0 - torch.maximum(batch.terminations, batch.truncations)
                b_valid, raw_rewards = continuing.flatten(), batch.fields["raw_rewards"].flatten()
                agent.consequence.calibrate(following - b_obs, raw_rewards, b_valid)
                # Observations keep the normalization they were collected under; rewards are kept raw
                # because their normalizer's scale moves with the policy.
                newest = ((iteration - 1) % args.memory_rollouts) * args.batch_size
                remembered = min(iteration, args.memory_rollouts) * args.batch_size
                memory[newest:newest + args.batch_size] = torch.cat(
                    (b_obs, b_native, following - b_obs, raw_rewards[:, None],
                     batch.terminations.flatten()[:, None], b_valid[:, None]), dim=-1)
                expected = consequence_loss(agent, *memory[newest:newest + args.batch_size].split(memory_fields, dim=-1)[:3],
                                            b_valid)[1]
                reward_divisor = torch.as_tensor(np.sqrt(rew_norm.variances + rew_norm.epsilon), dtype=torch.float32,
                                                 device=device)[b_environments]
                b_considered, b_considered_logprobs, own, considered = considered_gains(
                    agent, b_obs, b_alpha, b_beta, b_native, reward_divisor, args.gamma, args.considered_actions)
                trust, surprise, explained = calibrate_own(
                    b_advantages.view(args.num_steps, args.num_envs), own.view(args.num_steps, args.num_envs, 2),
                    continuing, args.gamma * args.gae_lambda)
                if not args.predict_own:
                    trust, surprise = torch.zeros_like(trust), b_advantages.view(args.num_steps, args.num_envs)
                b_surprise, b_considered_own = surprise.flatten(), considered @ trust
                credit = torch.stack((
                    trust[0], trust[1], explained.float(),
                    (own @ trust).square().mean() / b_advantages.square().mean(),
                    torch.corrcoef(torch.stack((own[:, 0], b_advantages)))[0, 1],
                    torch.corrcoef(torch.stack((own[:, 1], b_advantages)))[0, 1]))
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_surprise[indices], b_returns[indices], b_values[indices],
                            b_considered[indices], b_considered_logprobs[indices], b_considered_own[indices],
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

            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "losses/own_loss": last[6],
                "credit/reward_trust": credit[0], "credit/value_trust": credit[1],
                "credit/explained_share": credit[2], "credit/own_share": credit[3],
                "credit/reward_own_correlation": credit[4], "credit/value_own_correlation": credit[5],
                "consequence/change_r2_heldout": 1.0 - expected[0],
                "consequence/reward_r2_heldout": 1.0 - expected[1],
                "consequence/change_r2_fit": 1.0 - consequence_metrics[-1, 0],
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
                run_name=f"{run_name}-eval", Model=lambda envs: Agent(envs, args.consequence_width), device=device, gamma=args.gamma,
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

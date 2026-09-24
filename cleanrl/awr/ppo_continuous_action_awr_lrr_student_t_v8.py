# AWR log-ratio regression v8: Student-t weighted fit. Base: v6.
# v6's batch z-scores have kurtosis 50-170. At that tail weight, about half of the least-squares
# target energy sits in 1% of samples, so those samples set the update direction.
# fit_r2 decays from 0.14 to 0.02, and the step becomes noise-driven late in training. PPO's clip caps
# each sample's influence and overtakes v6 after about 6M steps. v7 tested whether the tails
# were per-state heteroscedasticity: the learned sigma(s) spread only e^0.3-0.4 and barely
# changed kurtosis (raw 108 -> 108 at 30M). The tails are per-sample, not predictable from state.
# The principled estimator for a signal observed through heavy-tailed noise is the Student-t
# likelihood. Its EM form is weighted least squares with precision weights
#   w_i = (nu + 1) / (nu + z_i^2 / s^2),
# a redescending influence (w z -> 0 as |z| -> inf) whose shape is set by the data.
# nu and s are fitted by profile maximum likelihood on each rollout: no hand-set threshold,
# no rank transform, and a near-Gaussian batch gives w ~ 1 and recovers v6.
# The weights are computed from the targets alone. With fit R^2 at 0.02-0.14, the residual stays
# close to -z for the whole update, so fixed weights match IRLS and the loss keeps
# a fixed quadratic attractor.
#   L = 0.5 E[w (l - z/eta)^2],  w normalized to mean one.
# AWR log-ratio regression v6: least-squares fit of the per-state-centered log-ratio to A / eta. Base: v5.
# Mirror descent's KL-penalized improvement is pi* = pi_old exp(A / eta) / Z(s), i.e.
# log pi*/pi_old = A/eta - log Z(s). The CE family (v1-v5, V-MPO) fits this through
# exponentiated weights. On noisy, heavy-tailed GAE advantages that amplifies noise:
# ESS collapses (v5 eta1: ESS 0.11, max weight 397 x N) and the M-step fits a few outliers.
# The only fix inside that family is a flatter target, which also makes the step tiny
# (v5 kl0.01: ESS 0.98, approx_kl 0.0017). V-MPO's top-half truncation and tiny covariance
# KL patch the same thing. Truncation selection itself shrinks the covariance,
# which is what the covariance constraint then has to hold back.
# Here we regress in log space instead, so no exponent ever touches A:
#   l(s,a) = log pi(a|s) - log pi_old(a|s) + KL(pi_old || pi)(s),   L = 0.5 E[(l - z/eta)^2].
# E_{pi_old}[l | s] = 0 exactly for every theta, via the closed-form Beta KL. That
# absorbs log Z(s), and any per-state offset in the targets (critic bias) is orthogonal to l.
# Properties:
# - At theta_old the gradient is the policy gradient of z/eta.
# - The Gauss-Newton curvature is the Fisher, so the fitted step is a natural-gradient step of size 1/eta.
# - The loss equals -E[z l]/eta + 0.5 E[l^2]: a surrogate with an exact log-ratio variance trust region instead of PPO's clip.
# - The target is fixed, so 10 epochs converge to it rather than past it.
# - Zero-mean advantage noise enters linearly, so it averages out instead of being exponentiated.
# - The Beta concentration moves only where A's curvature in a supports it.
# No E-step budget, no gradient clipping. Hypothesis: a soft, unbiased, self-limiting
# target that takes PPO-sized steps without PPO's clip.
import os
import random
import time
from contextlib import ExitStack
from math import log10
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
    """the K epochs to update the critic"""
    actor_epochs: int = 10
    """the first actor_epochs of the update_epochs also update the actor"""
    advantage_temperature: float = 1.0
    """eta on z-scored advantages: the centered log-ratio target is z / eta"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = float("inf")
    """per-network gradient clipping norm; inf disables it (norms are still logged)"""

    # Execution controls, independent of the learner's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile deterministic policy statistics, the loss, targets and GAE"""
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


# Public evaluation helper; not the training path.
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


# Profile-likelihood grid for the Student-t degrees of freedom; nu = 1000 is Gaussian
# for practical purposes, so a light-tailed batch falls back to v6's unweighted fit.
STUDENT_NU_MIN, STUDENT_NU_MAX, STUDENT_NU_POINTS = 0.5, 1000.0, 48
STUDENT_EM_ITERATIONS = 30


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
            raise TypeError("Beta AWR requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta AWR requires finite, strictly ordered action bounds")
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

    def policy(self, x):
        return (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)

    def get_policy_and_value(self, x):
        alpha, beta = self.policy(x)
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


def student_t_weights(z):
    """Profile-MLE Student-t fit to z (location 0); returns EM precision weights, nu, scale.

    For each nu on a log grid, EM iterates the scale to its conditional MLE. The grid
    point with the highest likelihood wins. All grid points run in one [G, N] batch.
    """
    nu = torch.logspace(log10(STUDENT_NU_MIN), log10(STUDENT_NU_MAX), STUDENT_NU_POINTS,
                        device=z.device, dtype=z.dtype).unsqueeze(1)
    squared = z.square().unsqueeze(0)
    variance = torch.ones_like(nu)
    for _ in range(STUDENT_EM_ITERATIONS):
        weights = (nu + 1.0) / (nu + squared / variance)
        variance = (weights * squared).mean(dim=1, keepdim=True)
    log_likelihood = (torch.lgamma(0.5 * (nu + 1.0)) - torch.lgamma(0.5 * nu)
                      - 0.5 * (nu * variance).log()
                      - 0.5 * (nu + 1.0) * (squared / (nu * variance)).log1p().mean(dim=1, keepdim=True))
    best = log_likelihood.argmax(dim=0, keepdim=True)
    best_nu, best_variance = nu.gather(0, best).squeeze(), variance.gather(0, best).squeeze()
    weights = (best_nu + 1.0) / (best_nu + z.square() / best_variance)
    return weights / weights.mean(), best_nu, best_variance.sqrt()


def beta_kl(old_alpha, old_beta, alpha, beta):
    """Closed-form per-dimension KL(Beta(old) || Beta(new)); traceable, unlike kl_divergence dispatch."""
    old_sum, new_sum = old_alpha + old_beta, alpha + beta
    return (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(new_sum)
            - torch.lgamma(old_alpha) - torch.lgamma(old_beta) + torch.lgamma(old_sum)
            + (old_alpha - alpha) * torch.digamma(old_alpha)
            + (old_beta - beta) * torch.digamma(old_beta)
            + (new_sum - old_sum) * torch.digamma(old_sum))


def awr_loss(agent, observations, native_actions, old_logprobs, old_alpha, old_beta,
             targets, precision, returns, args):
    """Least-squares fit of the per-state-centered log-ratio to the fixed targets, plus unclipped value MSE."""
    alpha, beta = agent.policy(observations)
    newvalue = agent.critic(observations).view(-1)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    # E_{pi_old}[logratio | s] = -KL(pi_old || pi)(s), so adding it back centers l per state.
    anchor_kl = beta_kl(old_alpha, old_beta, alpha, beta).sum(-1)
    centered = logratio + anchor_kl
    residual = centered - targets
    policy_loss = 0.5 * (precision * residual.square()).mean()
    value_loss = 0.5 * (newvalue - returns).square().mean()
    loss = policy_loss + args.vf_coef * value_loss
    with torch.no_grad():
        approx_kl = (logratio.exp() - 1 - logratio).mean()
        # Weighted, so it measures the fit on the samples the loss actually trusts.
        fit_r2 = 1.0 - (precision * residual.square()).mean() / (precision * targets.square()).mean()
        target_gain = (targets * centered).mean()
    metrics = torch.stack((policy_loss.detach(), value_loss.detach(), entropy.mean().detach(),
                           anchor_kl.mean().detach(), approx_kl, fit_r2, target_gain))
    return loss, metrics


def critic_loss(agent, observations, returns, args):
    """The critic term alone, for epochs past actor_epochs; actor grads stay None."""
    value_loss = 0.5 * (agent.critic(observations).view(-1) - returns).square().mean()
    return args.vf_coef * value_loss, value_loss.detach()


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0 or args.batch_size % args.minibatch_size:
        raise ValueError("num_minibatches must divide batch_size")
    if not args.advantage_temperature > 0.0:
        raise ValueError("advantage_temperature must be positive")
    if not 1 <= args.actor_epochs <= args.update_epochs:
        raise ValueError("actor_epochs must lie in [1, update_epochs]")
    if not args.cuda:
        raise ValueError("the shared trainer requires CUDA")
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
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); Student-t weighted least-squares fit of log-ratio + KL(pi_old||pi) to z(A) / eta")
        writer.add_text("architecture", "residual SiTU-GLU; unit-L2 matrices and residual directions; scalar MSE critic")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        actor_parameters, critic_parameters = tuple(agent.actor.parameters()), tuple(agent.critic.parameters())
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native), alpha, beta

        def loss_model(observations, native, old_logprobs, old_alpha, old_beta, targets, precision, returns):
            return awr_loss(agent, observations, native, old_logprobs, old_alpha, old_beta,
                            targets, precision, returns, args)

        @torch.no_grad()
        def prepare_targets(advantages):
            """The batch-wide log-ratio target z(A) / eta and its Student-t precision weights."""
            advantage_std = advantages.std()
            z = (advantages - advantages.mean()) / advantage_std
            precision, nu, scale = student_t_weights(z)
            energy, tail = z.square(), z.abs() > 4.0
            diagnostics = {
                "awr/advantage_std": advantage_std,
                "awr/max_abs_z": z.abs().max(),
                "awr/z_kurtosis": z.pow(4).mean(),
                "awr/t_nu": nu, "awr/t_scale": scale,
                "awr/precision_ess": precision.sum().square() / precision.square().sum() / z.numel(),
                # Share of the regression's target energy carried by |z| > 4, before and after weighting.
                "awr/tail_energy_raw": (energy * tail).sum() / energy.sum(),
                "awr/tail_energy_weighted": (precision * energy * tail).sum() / (precision * energy).sum(),
            }
            return z / args.advantage_temperature, precision, diagnostics

        def critic_loss_model(observations, returns):
            return critic_loss(agent, observations, returns, args)

        if args.compile:
            critic_loss_model = torch.compile(critic_loss_model, mode=args.compile_mode,
                                              fullgraph=True, dynamic=False)
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
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.actor_epochs * args.num_minibatches
        last_value_loss = torch.empty((), device=device)
        grad_norms = torch.zeros((args.update_epochs * args.num_minibatches, 2), device=device)
        update_metrics = torch.empty((max_updates, 7), device=device)
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
                b_values, b_logprobs, b_alpha, b_beta = rollout_statistics(b_obs, b_native)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_returns = returns.flatten().clone()
                b_targets, b_precision, target_diagnostics = prepare_targets(advantages.flatten())
            updates = steps = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        if epoch < args.actor_epochs:
                            loss, metrics = loss_model(
                                b_obs[indices], b_native[indices], b_logprobs[indices],
                                b_alpha[indices], b_beta[indices],
                                b_targets[indices], b_precision[indices], b_returns[indices],
                            )
                            value_loss = metrics[1]
                        else:
                            # A separate graph leaves actor grads None, so Adam skips
                            # the actor entirely instead of coasting on its momentum.
                            loss, value_loss = critic_loss_model(b_obs[indices], b_returns[indices])
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norms[steps, 1] = nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
                        if epoch < args.actor_epochs:
                            grad_norms[steps, 0] = nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm, foreach=True)
                        steps += 1
                        optimizer.step()
                        normalize_matrices()
                        last_value_loss.copy_(value_loss)
                        if epoch < args.actor_epochs:
                            update_metrics[updates].copy_(metrics)
                            updates += 1

            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last_value_loss,
                "grad/actor_preclip_norm": grad_norms[:updates, 0].mean(),
                "grad/critic_preclip_norm": grad_norms[:steps, 1].mean(),
                "losses/entropy": last[2], "awr/exact_kl": last[3],
                "losses/approx_kl": last[4], "awr/fit_r2": last[5], "awr/target_gain": last[6],
                "losses/explained_variance": explained_variance(b_values, b_returns),
                **target_diagnostics,
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite learner metrics")
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

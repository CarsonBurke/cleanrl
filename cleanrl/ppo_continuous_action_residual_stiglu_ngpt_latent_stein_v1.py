# Latent-conditioned critic U(s, L) with a Stein/Q-Prop latent gradient, v1.
# Base: residual SiTU-GLU nGPT-scaled PPO (asymclip_v7 run as noadv/novalueclip,
# symmetric 0.2 clip). Each action is a two-level decision per env step:
#   L ~ N(mu(s), sigma(s))        per-dim Beta "tilt" (pre-action latent)
#   a ~ Beta(1+sp(k(s)+L), 1+sp(k(s)-L))   execution noise given the latent
# (k, L) is a bijection of the base (alpha, beta) logits, so only L's
# stochasticity is new. Critic: U(s,L) = V(s) + g(s,L) - E_{L~p_old} g, with the
# expectation by sigma points, so V(s) = E_L U(s,L) holds by construction.
# GAE over the augmented chain s -> (s,L) -> s' with lambda_U = 1 reduces to
#   A_exec = A_gae - (U(s,L) - V(s)),  and both nodes share the lambda-return.
# Actor gradient (unbiased for any g): joint (L, a) PPO surrogate with A_exec,
# plus the pathwise term d/dtheta g(s, mu + sigma*eps) through the rollout-time
# g snapshot that defined A_exec. Per state, a Gaussian KL trust region switches
# off the pathwise term and its control variate together. Actor and critic grad
# norms are clipped separately.
# Hypothesis: the critic carries the credit for the latent choice (lower-variance
# LR residual + DPG-like pathwise steering), improving sample efficiency.
import copy
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
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
LATENT_SIGMA_INIT = 1.0
"""initial latent tilt std offset. nGPT renormalizes head rows at init, so every
head output (mu, sigma and k logits) also spreads ~N(0, 1) across states, as in
the base's Beta logits; the latent noise is added on top of that spread"""
CONCENTRATION_OFFSET = 1.0
UPDATE_METRICS = ("policy_loss", "value_loss", "latent_value_loss", "old_approx_kl",
                  "approx_kl", "clipfrac", "latent_kl", "pathwise_gate", "latent_sigma",
                  "execution_entropy")


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
    clip_coef: float = 0.2
    """lower joint (latent, action) ratio clipping coefficient"""
    clip_coef_upper: float = 0.2
    """upper joint (latent, action) ratio clipping coefficient"""
    pathwise_max_kl: float = 0.03
    """per-state KL(p_old(L|s) || p_new(L|s)) beyond which the pathwise term is
    dropped; matches the base run's end-of-update joint approx_kl (~0.03)"""
    vf_coef: float = 0.5
    """coefficient of each of the V and U regression losses"""
    max_grad_norm: float = 0.5
    """the maximum gradient norm, applied separately to actor and critic"""
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
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

    def features(self, x):
        h = F.normalize(self.first(x), p=2, dim=-1)
        branch = F.normalize(self.second(8.0 * h), p=2, dim=-1)
        return 8.0 * F.normalize(h + branch, p=2, dim=-1)

    def readout(self, features):
        return self.readout_gain * self.head(features)

    def forward(self, x):
        return self.readout(self.features(x))


class LatentCritic(nn.Module):
    """g(s, L), the latent-conditional part of U(s, L) = V(s) + g(s, L) - E_L g(s, L).

    Consumes the critic trunk's features and the latent's physical effect
    tanh(L/2) = 2*sigmoid(L)-1 (the Beta mean saturates in L the same way),
    scaled so the latent block carries a fixed share of the input norm.
    Zero readout gain: U = V at initialization, i.e. the run starts as PPO.
    """

    def __init__(self, feature_dim, latent_dim):
        super().__init__()
        self.latent_scale = 8.0 / math.sqrt(latent_dim)
        self.branch = init_situglu_branch(SiTUGLUBranch(feature_dim + latent_dim, 64))
        self.head = layer_init(nn.Linear(64, 1, bias=False), std=1.0)
        self.readout_gain = nn.Parameter(torch.zeros(1))

    def forward(self, features, latent):
        x = torch.cat((features, self.latent_scale * torch.tanh(0.5 * latent)), dim=-1)
        hidden = F.normalize(self.branch(x), p=2, dim=-1)
        return (self.readout_gain * self.head(8.0 * hidden)).squeeze(-1)


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


class HostLatentPolicy:
    """Host rollout policy: sample L ~ N(mu, sigma), then the Beta action given L.

    Returns permanent (native, latent, action) buffers overwritten by the next call.
    """

    def __init__(self, agent, num_envs, rng, sample_beta):
        dim = agent.action_dim
        self.dim = dim
        self.mirror = ResidualHostMirror(agent.actor, num_envs)
        self.offset = agent.head_offset.cpu().numpy()
        self.params = np.empty((num_envs, 3 * dim), dtype=np.float32)
        self.sigma = np.empty((num_envs, dim), dtype=np.float32)
        self.latent = np.empty_like(self.sigma)
        self.logits = np.empty((num_envs, 2 * dim), dtype=np.float32)
        self.rng = rng
        self.sample_beta = sample_beta

    def refresh(self):
        self.mirror.refresh()

    def __call__(self, observations):
        dim = self.dim
        np.add(self.mirror(observations), self.offset, out=self.params)
        np.logaddexp(np.float32(0.0), self.params[:, dim:2 * dim], out=self.sigma)
        self.rng.standard_normal(dtype=np.float32, out=self.latent)
        np.multiply(self.latent, self.sigma, out=self.latent)
        np.add(self.latent, self.params[:, :dim], out=self.latent)
        concentration = self.params[:, 2 * dim:]
        np.add(concentration, self.latent, out=self.logits[:, :dim])
        np.subtract(concentration, self.latent, out=self.logits[:, dim:])
        native, action = self.sample_beta(self.logits, self.rng)
        return native, self.latent, action


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
        self.action_dim = dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.critic = ResidualMLP(observation_dim, 1, output_std=1.0)
        self.actor = ResidualMLP(observation_dim, 3 * dim, output_std=0.01)
        self.latent_critic = LatentCritic(64, dim)
        offset = torch.zeros(3 * dim)
        offset[dim:2 * dim] = math.log(math.expm1(LATENT_SIGMA_INIT))
        offset[2 * dim:] = CONCENTRATION_OFFSET
        self.register_buffer("head_offset", offset)
        # Unscented sigma points with zero centre weight: exact for quadratic g.
        eye = torch.eye(dim)
        self.register_buffer("sigma_directions", math.sqrt(dim) * torch.cat((eye, -eye)))

    @torch.no_grad()
    def normalize_matrices(self):
        """Match nGPT's matrix axes without changing Adam moments or biases."""
        branches = [stage[0] for trunk in (self.actor, self.critic) for stage in (trunk.first, trunk.second)]
        branches.append(self.latent_critic.branch)
        for branch in branches:
            for weight, dim in ((branch.gate.weight, 1), (branch.up.weight, 1), (branch.down.weight, 0)):
                weight.div_(torch.linalg.vector_norm(weight, dim=dim, keepdim=True))
        for weight in (self.actor.head[0].weight, self.critic.head[0].weight, self.latent_critic.head.weight):
            weight.div_(torch.linalg.vector_norm(weight, dim=1, keepdim=True))

    def get_value(self, x):
        return self.critic(x)

    def policy_params(self, x):
        mu, sigma, concentration = (self.actor(x) + self.head_offset).chunk(3, dim=-1)
        return mu, F.softplus(sigma), concentration

    def execution_distribution(self, concentration, latent):
        return Beta(1.0 + F.softplus(concentration + latent), 1.0 + F.softplus(concentration - latent),
                    validate_args=False)

    def execution_logprob(self, concentration, latent, native_action):
        distribution = self.execution_distribution(concentration, latent)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    @staticmethod
    def latent_logprob(mu, sigma, latent):
        z = (latent - mu) / sigma
        return (-0.5 * z * z - sigma.log() - HALF_LOG_2PI).sum(-1)

    def latent_value_mean(self, features, mu, sigma):
        """E_{L ~ N(mu, sigma)} g(s, L) by sigma points."""
        points = mu.unsqueeze(-2) + sigma.unsqueeze(-2) * self.sigma_directions
        expanded = features.unsqueeze(-2).expand(*points.shape[:-1], features.shape[-1])
        return self.latent_critic(expanded, points).mean(-1)

    def get_action_and_value(self, x, action=None):
        """Public API: physical actions, the execution log-prob given a fresh latent, and V."""
        mu, sigma, concentration = self.policy_params(x)
        latent = mu + sigma * torch.randn_like(mu)
        distribution = self.execution_distribution(concentration, latent)
        if action is None:
            native = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = (self.action_low + self.action_scale * native).reshape((x.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(x.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, self.critic(x)


def gaussian_kl(old_mu, old_sigma, mu, sigma):
    """KL(N(old) || N(new)) summed over latent dimensions."""
    ratio = old_sigma / sigma
    return (0.5 * (ratio * ratio + ((mu - old_mu) / sigma) ** 2 - 1.0) - ratio.log()).sum(-1)


class CriticSnapshot(nn.Module):
    """Rollout-time copy of the critic trunk and g, the exact function whose
    control variate is subtracted from A_gae. Parameters are refreshed in
    place so compiled graphs keep their addresses."""

    def __init__(self, agent):
        super().__init__()
        self.critic = copy.deepcopy(agent.critic).requires_grad_(False)
        self.latent_critic = copy.deepcopy(agent.latent_critic).requires_grad_(False)
        self.sources = list(agent.critic.parameters()) + list(agent.latent_critic.parameters())
        self.targets = list(self.critic.parameters()) + list(self.latent_critic.parameters())

    @torch.no_grad()
    def refresh(self):
        torch._foreach_copy_(self.targets, self.sources)

    def forward(self, observations, latent):
        return self.latent_critic(self.critic.features(observations), latent)


def stein_ppo_loss(agent, snapshot, observations, native_actions, latents, noise, old_mu, old_sigma,
                   old_logprobs, gae_advantages, control_variates, returns, args):
    """Joint-ratio PPO on A_exec + gated pathwise g term; V and U regressions."""
    mu, sigma, concentration = agent.policy_params(observations)
    execution = agent.execution_distribution(concentration, latents)
    newlogprob = (agent.latent_logprob(mu, sigma, latents)
                  + (execution.log_prob(native_actions) - agent.log_action_scale).sum(-1))
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        latent_kl = gaussian_kl(old_mu, old_sigma, mu, sigma)
        gate = (latent_kl <= args.pathwise_max_kl).float()
    # Stein pair: subtract the control variate only where its pathwise
    # expectation is added back, so gated-off states fall back to A_gae.
    advantages = gae_advantages - gate * control_variates
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef_upper)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Same noise as the stored sample, so epoch 0 differentiates at the taken latent.
    pathwise = snapshot(observations, mu + sigma * noise)
    pathwise_loss = -(gate * pathwise).mean()

    features = agent.critic.features(observations)

    value = agent.critic.readout(features).squeeze(-1)
    latent_value = (value.detach() + agent.latent_critic(features, latents)
                    - agent.latent_value_mean(features, old_mu, old_sigma))
    v_loss = 0.5 * ((value - returns) ** 2).mean()
    u_loss = 0.5 * ((latent_value - returns) ** 2).mean()
    loss = pg_loss + pathwise_loss + args.vf_coef * (v_loss + u_loss)
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio < 1 - args.clip_coef) | (ratio > 1 + args.clip_coef_upper)).float().mean()
        execution_entropy = (execution.entropy() + agent.log_action_scale).sum(-1).mean()
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), u_loss.detach(), old_approx_kl,
                           approx_kl, clipfrac, latent_kl.mean(), gate.mean(),
                           sigma.detach().mean(), execution_entropy))
    return loss, metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.pathwise_max_kl < 0:
        raise ValueError("pathwise_max_kl must be nonnegative")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
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
        writer.add_text("policy", "L~N(mu,sigma) tilt; a~Beta(1+sp(k+L),1+sp(k-L)); FP32; host actor mirror")
        writer.add_text("critic", "U(s,L)=V(s)+g(s,L)-E_L g (sigma points); A_exec=A_gae-(U-V); Stein pathwise via frozen g, KL-gated")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        actor_params = list(agent.actor.parameters())
        critic_params = list(agent.critic.parameters()) + list(agent.latent_critic.parameters())
        value_model = agent.get_value
        snapshot = CriticSnapshot(agent)

        def rollout_statistics(observations, native, latents):
            """Old joint log-probs, latent stats, V and U-V for a whole rollout in one forward."""
            mu, sigma, concentration = agent.policy_params(observations)
            logprob = (agent.latent_logprob(mu, sigma, latents)
                       + agent.execution_logprob(concentration, latents, native))
            features = agent.critic.features(observations)
            value = agent.critic.readout(features).squeeze(-1)
            latent_advantage = (agent.latent_critic(features, latents)
                                - agent.latent_value_mean(features, mu, sigma))
            return value, logprob, (latents - mu) / sigma, mu, sigma, latent_advantage

        def loss_model(*batch):
            return stein_ppo_loss(agent, snapshot, *batch, args)

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
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        host_policy = HostLatentPolicy(
            agent, args.num_envs, sampler,
            make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high),
        )

        def act(observations):
            native, latent, action = host_policy(observations)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, latent, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,),
                                           "latents": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, len(UPDATE_METRICS)), device=device)
        approx_kl_column = UPDATE_METRICS.index("approx_kl")
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[2]

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
            host_policy.refresh()
            snapshot.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, latent, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(step, reward, terms, truncs, observations=obs_step,
                                  native_actions=native, latents=latent)
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
                b_latents = batch.fields["latents"].flatten(0, 1)
                (b_values, b_logprobs, b_noise, b_mu, b_sigma,
                 b_latent_advantage) = rollout_statistics(b_obs, b_native, b_latents)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_gae_advantages = advantages.flatten().clone()
                # lambda_U = 1: both chain nodes share the lambda-return, and the
                # execution node's advantage swaps V(s) for U(s, L) as its baseline.
                b_advantages = b_gae_advantages - b_latent_advantage
                b_returns = returns.flatten().clone()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_latents[indices], b_noise[indices],
                            b_mu[indices], b_sigma[indices], b_logprobs[indices],
                            b_gae_advantages[indices], b_latent_advantage[indices], b_returns[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                        nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                        optimizer.step()
                        normalize_matrices()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, approx_kl_column] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            named = {f"losses/{name}": last[column] for column, name in enumerate(UPDATE_METRICS)}
            named["losses/clipfrac"] = update_metrics[:updates, UPDATE_METRICS.index("clipfrac")].mean()
            named["losses/pathwise_gate"] = update_metrics[:updates, UPDATE_METRICS.index("pathwise_gate")].mean()
            named["losses/explained_variance"] = explained_variance(b_values, b_returns)
            named["losses/explained_variance_latent"] = explained_variance(b_values + b_latent_advantage, b_returns)
            named["latent/advantage_variance_ratio"] = b_advantages.var() / b_gae_advantages.var()
            named["latent/critic_std"] = b_latent_advantage.std()
            named["latent/critic_mean"] = b_latent_advantage.mean()
            logged = gather_metrics(named)
            if any(not np.isfinite(value) for name, value in logged.items()
                   if not name.startswith("losses/explained_variance")):
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

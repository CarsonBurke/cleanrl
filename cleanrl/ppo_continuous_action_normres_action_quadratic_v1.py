# Action-quadratic critic PPO v1, based only on normres_indclip_v4 norm-MSE.
# A shared value trunk predicts centered linear/quadratic/cross-action effects.
# ordinary: auxiliary action regression, unchanged scalar PPO surrogate.
# corrected: lagged action control variate + exact Beta expectation gradient;
# actor proposals are backtracked against an exact rollout-policy KL budget.
# Hypothesis: action structure improves representations and reduces gradient noise
# without replacing experienced returns by an exploitable learned Q objective.
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
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
    TruncationBootstrapCache,
    device_minibatches,
    explained_variance,
    gather_metrics,
    get_gae_fn,
)
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets,
    episode_horizon,
    run_phase_warmup,
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
    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 9.6e-3
    """32x the 3e-4 PPO default; one Adam covers actor and critic"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 2048
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
    """raw GAE in the surrogate; no minibatch standardization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """toggle PPO value-loss clipping independently of gradient clipping"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum gradient norm, applied independently to actor and critic"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    reward_norm: bool = True
    """normalize/clip rewards before GAE; never normalize GAE return targets"""

    placement: Literal["pre", "post"] = "pre"
    """normalize branch inputs with an identity stream, or residual outputs"""
    norm_kind: Literal["layer", "rms"] = "rms"
    """non-affine centered LayerNorm or uncentered RMSNorm, epsilon 1e-5"""
    activation: Literal["lrelusq", "stiglu"] = "stiglu"
    """squared leaky-ReLU pair or parameter-matched SiTU-GLU branch"""

    actor_update: Literal["ordinary", "corrected"] = "ordinary"
    """shared-critic ablation or analytically corrected residual PPO"""
    action_loss_coef: float = 1.0
    """action-advantage MSE weight inside the independently clipped critic loss"""
    trust_kl: float = 0.02
    """corrected update: maximum mean exact KL(old policy || candidate)"""
    max_backtracks: int = 12
    """halve an actor proposal this many times before restoring it and its Adam state"""
    gradient_diagnostic_interval: int = 32
    """rollout interval for empirical per-transition actor-gradient diagnostics"""
    gradient_diagnostic_samples: int = 128
    """evenly spaced rollout samples; diagnostic only, not a training minibatch"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads, capped at num_envs"""
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def action_features(native_actions, pair_i, pair_j):
    """Fixed bounded action coordinates; no division by a shrinking policy std."""
    y = 2.0 * native_actions - 1.0
    return torch.cat((y, y.square(), y[..., pair_i] * y[..., pair_j]), dim=-1)


def beta_feature_means(alpha, beta, pair_i, pair_j):
    total = alpha + beta
    mean = 2.0 * alpha / total - 1.0
    variance = 4.0 * alpha * beta / (total.square() * (total + 1.0))
    return torch.cat((mean, variance + mean.square(), mean[..., pair_i] * mean[..., pair_j]), dim=-1)


def beta_kl_reference(alpha, beta):
    """Cache old-policy log partition and derivatives once per rollout."""
    total = alpha + beta
    log_partition = alpha.lgamma() + beta.lgamma() - total.lgamma()
    total_digamma = total.digamma()
    return alpha, beta, log_partition, alpha.digamma() - total_digamma, beta.digamma() - total_digamma


def beta_kl(alpha, beta, reference):
    """KL(old || candidate), without torch.distributions' Python dispatcher."""
    old_alpha, old_beta, old_partition, old_da, old_db = reference
    partition = alpha.lgamma() + beta.lgamma() - (alpha + beta).lgamma()
    return partition - old_partition + (old_alpha - alpha) * old_da + (old_beta - beta) * old_db


class StructuredCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, *, placement, norm_kind, activation):
        super().__init__()
        self.trunk = make_norm_residual_trunk(
            observation_dim, 64, placement=placement, norm_kind=norm_kind, activation=activation
        )
        self.value_head = layer_init(nn.Linear(64, 1), std=1.0)
        # Do not perturb the base actor's initialization RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.action_head = layer_init(nn.Linear(64, 2 * action_dim + action_dim * (action_dim - 1) // 2), std=0.0)

    def forward(self, observations):
        hidden = self.trunk(observations)
        return self.value_head(hidden), self.action_head(hidden)

    def value(self, observations):
        return self.value_head(self.trunk(observations))


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    log_action_scale: torch.Tensor

    def __init__(self, envs, *, placement="pre", norm_kind="rms", activation="stiglu"):
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
        pairs = torch.triu_indices(self.action_dim, self.action_dim, offset=1)
        self.register_buffer("pair_i", pairs[0])
        self.register_buffer("pair_j", pairs[1])
        self.critic = StructuredCritic(
            observation_dim, self.action_dim, placement=placement, norm_kind=norm_kind, activation=activation
        )
        self.actor = nn.Sequential(
            make_norm_residual_trunk(
                observation_dim, 64, placement=placement, norm_kind=norm_kind, activation=activation
            ),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic.value(x)

    def get_policy_and_value(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic.value(x)

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


def clip_gradients(actor_parameters, critic_parameters, max_grad_norm):
    """Clip each network independently after the weighted joint loss backward."""
    actor_preclip_norm = nn.utils.clip_grad_norm_(actor_parameters, max_grad_norm, foreach=True)
    critic_preclip_norm = nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm, foreach=True)
    return actor_preclip_norm, critic_preclip_norm


def ppo_loss(
    agent, observations, native_actions, old_logprobs, advantages, targets, old_values,
    old_means, old_coefficients, args,
):
    alpha, beta = (F.softplus(agent.actor(observations)) + 1.0).chunk(2, dim=-1)
    newvalue, coefficients = agent.critic(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    centered_features = action_features(native_actions, agent.pair_i, agent.pair_j) - old_means.detach()
    prediction = (coefficients * centered_features).sum(-1)
    action_loss = 0.5 * (prediction - advantages.detach()).square().mean()
    frozen_coefficients = old_coefficients.detach()
    frozen_prediction = (frozen_coefficients * centered_features).sum(-1)
    if args.actor_update == "corrected":
        residual = advantages - frozen_prediction
        current_means = beta_feature_means(alpha, beta, agent.pair_i, agent.pair_j)
        correction = (frozen_coefficients * (current_means - old_means.detach())).sum(-1).mean()
    else:
        residual = advantages
        correction = ratio.new_zeros(())
    pg_loss1 = -residual * ratio
    pg_loss2 = -residual * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean() - correction
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - targets) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - targets) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - targets) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * (v_loss + args.action_loss_coef * action_loss)
    metrics = torch.stack(
        (pg_loss.detach(), v_loss.detach(), entropy_loss.detach(), old_approx_kl, approx_kl, clipfrac,
         action_loss.detach(), correction.detach())
    )
    return loss, metrics


class ActorTrustRegion:
    """Backtrack only actor parameters; rejected proposals restore actor Adam too.

    The critic has independent parameters and clipping and always keeps its step.
    Accepted fractional proposals keep Adam moments: line search scales the proposed
    displacement, not the gradient estimator. Exact KL checks intentionally sync
    only at proposal boundaries, outside the compiled loss/optimizer computation.
    """
    def __init__(self, actor, optimizer, *, budget, max_backtracks):
        self.parameters = tuple(actor.parameters())
        self.optimizer = optimizer
        self.budget = budget
        self.max_backtracks = max_backtracks
        self.before = tuple(torch.empty_like(p) for p in self.parameters)
        self.delta = tuple(torch.empty_like(p) for p in self.parameters)
        self.moment_copies = []
        self.had_state = []

    @torch.no_grad()
    def snapshot(self):
        self.had_state = []
        if not self.moment_copies:
            self.moment_copies = [dict() for _ in self.parameters]
        for parameter, before, saved in zip(self.parameters, self.before, self.moment_copies):
            before.copy_(parameter)
            state = self.optimizer.state.get(parameter, {})
            self.had_state.append(bool(state))
            for key, value in state.items():
                if key not in saved:
                    saved[key] = torch.empty_like(value)
                saved[key].copy_(value)

    @torch.no_grad()
    def accept(self, measure_kl):
        for delta, parameter, before in zip(self.delta, self.parameters, self.before):
            torch.sub(parameter, before, out=delta)
        fraction = 1.0
        for backtracks in range(self.max_backtracks + 1):
            kl = measure_kl()
            if torch.isfinite(kl) and kl <= self.budget:
                return kl, fraction, backtracks
            if backtracks < self.max_backtracks:
                fraction *= 0.5
                for parameter, before, delta in zip(self.parameters, self.before, self.delta):
                    parameter.copy_(before).add_(delta, alpha=fraction)
        for parameter, before, saved, had_state in zip(
            self.parameters, self.before, self.moment_copies, self.had_state
        ):
            parameter.copy_(before)
            if had_state:
                for key, value in saved.items():
                    self.optimizer.state[parameter][key].copy_(value)
            else:
                self.optimizer.state.pop(parameter, None)
        return measure_kl(), 0.0, self.max_backtracks + 1


def gradient_diagnostics(agent, observations, actions, advantages, old_means, coefficients):
    """Empirical full-actor gradient moments across transitions, not advantage variance.

    These are finite-batch diagnostics, not estimates from independent trajectories.
    Coefficients/advantages are frozen. Compute all three estimators at the old policy.
    """
    parameters = dict(agent.actor.named_parameters())

    def sample_terms(parameters, observation, action, advantage, means, coefficient):
        output = torch.func.functional_call(agent.actor, parameters, (observation.unsqueeze(0),)).squeeze(0)
        alpha, beta = (F.softplus(output) + 1.0).chunk(2, dim=-1)
        logprob = Beta(alpha, beta, validate_args=False).log_prob(action).sum()
        prediction = (coefficient * (action_features(action, agent.pair_i, agent.pair_j) - means)).sum()
        expectation = (coefficient * (beta_feature_means(alpha, beta, agent.pair_i, agent.pair_j) - means)).sum()
        return torch.stack((logprob * advantage, logprob * (advantage - prediction), expectation))

    jacobian = torch.func.vmap(torch.func.jacrev(sample_terms), in_dims=(None, 0, 0, 0, 0, 0))(
        parameters, observations, actions, advantages, old_means, coefficients
    )
    # Accumulate parameter-wise to avoid concatenating a second giant gradient tensor.
    ordinary_variance = observations.new_zeros(())
    residual_variance = observations.new_zeros(())
    corrected_variance = observations.new_zeros(())
    residual_norm2 = observations.new_zeros(())
    analytic_norm2 = observations.new_zeros(())
    inner = observations.new_zeros(())
    for tensor in jacobian.values():
        gradients = tensor.detach().flatten(2)
        ordinary, residual, analytic = gradients.unbind(1)
        ordinary_variance += ordinary.var(0, unbiased=False).sum()
        residual_variance += residual.var(0, unbiased=False).sum()
        corrected_variance += (residual + analytic).var(0, unbiased=False).sum()
        residual_mean, analytic_mean = residual.mean(0), analytic.mean(0)
        residual_norm2 += residual_mean.square().sum()
        analytic_norm2 += analytic_mean.square().sum()
        inner += (residual_mean * analytic_mean).sum()
    return {
        "gradient/ordinary_variance_trace": ordinary_variance,
        "gradient/residual_variance_trace": residual_variance,
        "gradient/corrected_variance_trace": corrected_variance,
        "gradient/residual_mean_norm": residual_norm2.sqrt(),
        "gradient/analytic_mean_norm": analytic_norm2.sqrt(),
        "gradient/residual_analytic_cosine": inner / (residual_norm2 * analytic_norm2).sqrt().clamp_min(1e-12),
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
    if args.norm_adv:
        raise ValueError("this matched raw-GAE ablation requires --no-norm-adv")
    if args.action_loss_coef <= 0 or args.trust_kl <= 0 or args.max_backtracks < 0:
        raise ValueError("action loss and KL budget must be positive; backtracks must be nonnegative")
    if args.gradient_diagnostic_interval < 0 or args.gradient_diagnostic_samples < 2:
        raise ValueError("invalid gradient diagnostic settings")
    if args.num_minibatches != 1:
        raise ValueError("v1 preserves the base's one-full-rollout-minibatch updates")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id,
        args.num_envs,
        backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video,
        run_name=run_name,
    )


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision="highest", allow_tf32=False)
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

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
        )
        writer.add_text(
            "policy",
            f"Beta host actor; additive {args.placement}-{args.norm_kind}norm "
            f"{args.activation} residual trunk; 32x LR; 1 minibatch; raw GAE; "
            f"quadratic shared critic; {args.actor_update} actor",
        )
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, placement=args.placement, norm_kind=args.norm_kind, activation=args.activation).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        actor_parameters = tuple(agent.actor.parameters())
        critic_parameters = tuple(agent.critic.parameters())
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            alpha, beta = (F.softplus(agent.actor(observations)) + 1.0).chunk(2, dim=-1)
            value, coefficients = agent.critic(observations)
            means = beta_feature_means(alpha, beta, agent.pair_i, agent.pair_j)
            return value.flatten(), agent.action_logprob(alpha, beta, native), coefficients, means, alpha, beta

        def loss_model(observations, native, old_logprobs, advantages, targets, old_values, old_means, coefficients):
            return ppo_loss(
                agent, observations, native, old_logprobs, advantages, targets, old_values, old_means, coefficients, args
            )

        def exact_policy_kl(observations, reference):
            alpha, beta = (F.softplus(agent.actor(observations)) + 1.0).chunk(2, dim=-1)
            return beta_kl(alpha, beta, reference).sum(-1).mean()

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            exact_policy_kl = torch.compile(exact_policy_kl, mode=args.compile_mode, fullgraph=True, dynamic=False)
        trust_region = (
            ActorTrustRegion(agent.actor, optimizer, budget=args.trust_kl, max_backtracks=args.max_backtracks)
            if args.actor_update == "corrected" else None
        )
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)
        sampler_rng = np.random.default_rng(args.seed)

        def act(observations):
            native, action = sampler(host_actor(observations), sampler_rng)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(
            args.num_steps,
            args.num_envs,
            obs_shape,
            device,
            non_blocking=args.non_blocking_transfers,
            fields={"observations": obs_shape, "native_actions": (agent.action_dim,)},
        )
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma) if args.reward_norm else None
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 8), device=device)
        grad_norms = torch.empty((max_updates, 2), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(
                envs,
                obs_norm=obs_norm,
                rew_norm=rew_norm,
                act_fn=warmup_action,
                horizon=horizon,
                phase_offsets=phases,
                seed=args.seed,
            )
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
                    reward = rew_norm.normalize(raw_reward, terms) if rew_norm is not None else raw_reward
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
                # Snapshot before fitting this rollout: no same-rollout action-target leakage.
                b_values, b_logprobs, b_coefficients, b_means, b_alpha, b_beta = (
                    tensor.clone() for tensor in rollout_statistics(b_obs, b_native)
                )
                b_kl_reference = beta_kl_reference(b_alpha, b_beta)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards,
                    values,
                    batch.terminations,
                    batch.truncations,
                    truncation_values,
                    tail_value,
                    args.gamma,
                    args.gae_lambda,
                )
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
            with timer.span("diagnostics"), torch.no_grad():
                frozen_prediction = (
                    b_coefficients * (action_features(b_native, agent.pair_i, agent.pair_j) - b_means)
                ).sum(-1)
                diagnostic_metrics = {
                    "action/lagged_mse": (frozen_prediction - b_advantages).square().mean(),
                    "action/zero_prediction_mse": b_advantages.square().mean(),
                    "action/lagged_explained_variance": explained_variance(frozen_prediction, b_advantages),
                    "action/lagged_prediction_rms": frozen_prediction.square().mean().sqrt(),
                }
            if args.gradient_diagnostic_interval and (iteration - 1) % args.gradient_diagnostic_interval == 0:
                with timer.span("gradient_diagnostics"):
                    sample_indices = torch.linspace(
                        0, args.batch_size - 1, min(args.gradient_diagnostic_samples, args.batch_size), device=device
                    ).long()
                    diagnostic_metrics.update(gradient_diagnostics(
                        agent, b_obs[sample_indices], b_native[sample_indices], b_advantages[sample_indices],
                        b_means[sample_indices], b_coefficients[sample_indices],
                    ))
            updates = 0
            accepted_updates = 0
            backtrack_count = 0
            accepted_fraction_sum = 0.0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices],
                            b_native[indices],
                            b_logprobs[indices],
                            b_advantages[indices],
                            b_returns[indices],
                            b_values[indices],
                            b_means[indices],
                            b_coefficients[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        actor_preclip_norm, critic_preclip_norm = clip_gradients(
                            actor_parameters,
                            critic_parameters,
                            args.max_grad_norm,
                        )
                        grad_norms[updates, 0].copy_(actor_preclip_norm)
                        grad_norms[updates, 1].copy_(critic_preclip_norm)
                        if trust_region is not None:
                            trust_region.snapshot()
                        optimizer.step()
                        if trust_region is not None:
                            _, fraction, backtracks = trust_region.accept(
                                lambda: exact_policy_kl(b_obs, b_kl_reference)
                            )
                            del _
                            accepted_updates += int(fraction > 0)
                            accepted_fraction_sum += fraction
                            backtrack_count += backtracks
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            with torch.no_grad():
                final_exact_kl = exact_policy_kl(b_obs, b_kl_reference).clone()
            last = update_metrics[updates - 1]
            performed_grad_norms = grad_norms[:updates]
            mean_grad_norms = performed_grad_norms.mean(dim=0)
            grad_clip_fractions = (performed_grad_norms > args.max_grad_norm).float().mean(dim=0)
            logged = gather_metrics(
                {
                    "losses/policy_loss": last[0],
                    "losses/value_loss": last[1],
                    "losses/entropy": last[2],
                    "losses/old_approx_kl": last[3],
                    "losses/approx_kl": last[4],
                    "losses/clipfrac": update_metrics[:updates, 5].mean(),
                    "losses/explained_variance": explained_variance(b_values, b_returns),
                    "losses/action_loss": last[6],
                    "losses/analytic_correction": last[7],
                    "losses/exact_kl": final_exact_kl,
                    **diagnostic_metrics,
                    "grad/actor_preclip_norm": mean_grad_norms[0],
                    "grad/critic_preclip_norm": mean_grad_norms[1],
                    "grad/actor_clip_fraction": grad_clip_fractions[0],
                    "grad/critic_clip_fraction": grad_clip_fractions[1],
                }
            )
            if trust_region is not None:
                logged.update({
                    "trust/rejected_updates": updates - accepted_updates,
                    "trust/backtracks": backtrack_count,
                    "trust/mean_accepted_fraction": accepted_fraction_sum / updates,
                })
            if any(not np.isfinite(value) for name, value in logged.items() if not name.endswith("explained_variance")):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar(
                "charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step
            )
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(
                f"update={iteration} step={global_step} exact_kl={logged['losses/exact_kl']:.6f} "
                f"action_mse={logged['action/lagged_mse']:.6f} "
                f"zero_mse={logged['action/zero_prediction_mse']:.6f} "
                f"SPS={int(global_step / (time.perf_counter() - start_time))}"
            )
            interval_start, interval_step = time.perf_counter(), global_step

        transfer.close()
        envs.close()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(
                {
                    "model": agent.state_dict(),
                    "args": vars(args),
                    "obs_norm": {
                        "means": torch.from_numpy(obs_norm.means.copy()),
                        "variances": torch.from_numpy(obs_norm.variances.copy()),
                        "counts": torch.from_numpy(obs_norm.counts.copy()),
                        "epsilon": obs_norm.epsilon,
                        "clip": obs_norm.clip,
                    },
                },
                model_path,
            )
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()

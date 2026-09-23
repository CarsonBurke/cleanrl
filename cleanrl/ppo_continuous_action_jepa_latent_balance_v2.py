# Attached JEPA latent-depth v2: first-layer or final pre-head latent prediction.
# Both time branches and the full PPO trunks remain attached; no EMA/stop-grad,
# variance regularizer, or loss normalization. Same 64x64 Beta PPO as the control.
# Compare weighted PPO/JEPA gradients on the shared encoder before clipping;
# scalar loss magnitude alone is not evidence that one learning signal dominates.
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
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
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

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
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

    jepa_mode: Literal["actor", "critic", "both", "none"] = "both"
    """which PPO trunks receive attached next-latent prediction loss"""
    jepa_coef: float = 1.0
    """coefficient per enabled branch's mean squared latent prediction error"""

    jepa_layer: Literal["first", "final"] = "final"
    """latent to predict: first hidden layer or final hidden layer before the output head"""
    jepa_projection: Literal["concat", "adaln"] = "concat"
    """concatenated MLP or le-wm-style AdaLN-zero residual latent predictor"""
    jepa_critic_condition: Literal["action", "value"] = "action"
    """critic predictor condition: chosen action or current attached value prediction"""
    balance_interval: int = 10
    """measure loss gradients on one minibatch every N rollouts, including the first; 0 disables"""

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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LatentPredictor(nn.Module):
    """Single-vector conditional projection; AdaLN adapts le-wm's gated FF block."""

    def __init__(self, condition_dim, projection):
        super().__init__()
        self.projection = projection
        if projection == "concat":
            self.net = nn.Sequential(
                layer_init(nn.Linear(64 + condition_dim, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, 64), std=1.0),
            )
        else:
            self.norm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
            self.condition = layer_init(nn.Linear(condition_dim, 64))
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(64, 3 * 64))
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)
            self.net = nn.Sequential(
                layer_init(nn.Linear(64, 64)), nn.GELU(),
                layer_init(nn.Linear(64, 64), std=1.0),
            )

    def forward(self, latent, condition):
        if self.projection == "concat":
            return self.net(torch.cat((latent, condition), dim=-1))
        shift, scale, gate = self.modulation(self.condition(condition)).chunk(3, dim=-1)
        modulated = self.norm(latent) * (1.0 + scale) + shift
        return latent + gate * self.net(modulated)


class Agent(nn.Module):
    def __init__(self, envs, jepa_mode="both", jepa_layer="final",
                 jepa_projection="concat", jepa_critic_condition="action"):
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
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )

        if jepa_mode not in {"actor", "critic", "both", "none"}:
            raise ValueError("invalid JEPA mode")
        if jepa_layer not in {"first", "final"}:
            raise ValueError("invalid JEPA layer")
        if jepa_projection not in {"concat", "adaln"}:
            raise ValueError("invalid JEPA projection")
        if jepa_critic_condition not in {"action", "value"}:
            raise ValueError("invalid JEPA critic condition")
        self.jepa_critic_condition = jepa_critic_condition
        self.jepa_layer = jepa_layer
        # Initialize after both PPO trunks: mode=none preserves baseline RNG and weights.
        self.actor_predictor = (
            LatentPredictor(self.action_dim, jepa_projection) if jepa_mode in {"actor", "both"} else None
        )
        self.critic_predictor = (
            LatentPredictor(1 if jepa_critic_condition == "value" else self.action_dim, jepa_projection)
            if jepa_mode in {"critic", "both"} else None
        )

    def branch_prediction(self, trunk, predictor, current, next_observations, condition):
        # Both time branches use the same attached encoder, through the selected depth.
        target = trunk[1](trunk[0](next_observations))
        if self.jepa_layer == "final":
            target = trunk[3](trunk[2](target))
        prediction = predictor(current, condition)
        loss = F.mse_loss(prediction, target)
        # Diagnostics only, never regularizers. Monitor collapse without preventing it.
        with torch.no_grad():
            latent_std = target.std(dim=0, correction=0).mean()
        return loss, latent_std

    def jepa_losses(self, actor_latent, critic_latent, next_observations, native_actions, value):
        zero = actor_latent.new_zeros(())
        actor_loss, actor_std = zero, zero
        critic_loss, critic_std = zero, zero
        action_condition = 2.0 * native_actions - 1.0
        if self.actor_predictor is not None:
            actor_loss, actor_std = self.branch_prediction(
                self.actor, self.actor_predictor, actor_latent, next_observations, action_condition,
            )
        if self.critic_predictor is not None:
            # Current prediction, not rollout-old values or returns; keep this path attached.
            condition = value.reshape(-1, 1) if self.jepa_critic_condition == "value" else action_condition
            critic_loss, critic_std = self.branch_prediction(
                self.critic, self.critic_predictor, critic_latent, next_observations, condition,
            )
        return actor_loss, critic_loss, torch.stack((
            actor_loss.detach(), critic_loss.detach(), actor_std, critic_std,
        ))

    def get_value(self, x):
        return self.critic(x)

    def get_policy_and_value(self, x):
        alpha, beta, value, _, _ = self.get_policy_value_latents(x)
        return alpha, beta, value

    def get_policy_value_latents(self, x):
        actor_latent = self.actor[1](self.actor[0](x))
        critic_latent = self.critic[1](self.critic[0](x))
        actor_final = self.actor[3](self.actor[2](actor_latent))
        critic_final = self.critic[3](self.critic[2](critic_latent))
        logits = self.actor[4](actor_final)
        value = self.critic[4](critic_final)
        if self.jepa_layer == "final":
            actor_latent, critic_latent = actor_final, critic_final
        alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
        return alpha, beta, value, actor_latent, critic_latent

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


def loss_components(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args, next_observations=None):
    """Four weighted objectives: actor PPO, critic PPO, actor JEPA, critic JEPA."""
    alpha, beta, newvalue, actor_latent, critic_latent = agent.get_policy_value_latents(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    entropy_loss = entropy.mean()
    actor_objective = pg_loss - args.ent_coef * entropy_loss
    critic_objective = v_loss * args.vf_coef
    actor_auxiliary = critic_auxiliary = observations.new_zeros(())
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac))
    if args.jepa_mode != "none":
        actor_auxiliary, critic_auxiliary, jepa_metrics = agent.jepa_losses(
            actor_latent, critic_latent, next_observations, native_actions, newvalue,
        )
        metrics = torch.cat((metrics, jepa_metrics))
    components = torch.stack((actor_objective, critic_objective,
                              args.jepa_coef * actor_auxiliary, args.jepa_coef * critic_auxiliary))
    return components, metrics


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args, next_observations=None):
    components, metrics = loss_components(
        agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args, next_observations,
    )
    return (components[0] + components[1]) + (components[2] + components[3]), metrics


def gradient_balance(agent, components):
    """Measure weighted gradients without touching .grad or changing the update.

    Predictor parameters are excluded. The critic output head is included only
    when value conditioning makes it shared with JEPA. Both attached temporal
    paths count, before clipping/Adam. The epsilon is diagnostic-only.
    """
    metrics = {}
    for index, name in enumerate(("actor", "critic")):
        if getattr(agent, name + "_predictor") is None:
            continue
        trunk = getattr(agent, name)
        depth = 2 if agent.jepa_layer == "first" else 4
        if name == "critic" and agent.jepa_critic_condition == "value":
            depth = 5
        parameters = tuple(parameter for layer in list(trunk.children())[:depth] for parameter in layer.parameters())
        primary = torch.autograd.grad(components[index], parameters, retain_graph=True)
        auxiliary = torch.autograd.grad(components[index + 2], parameters, retain_graph=True)
        with torch.no_grad():
            primary_sq = torch.stack([grad.square().sum() for grad in primary]).sum()
            auxiliary_sq = torch.stack([grad.square().sum() for grad in auxiliary]).sum()
            dot = torch.stack([(left * right).sum() for left, right in zip(primary, auxiliary)]).sum()
            primary_norm, auxiliary_norm = primary_sq.sqrt(), auxiliary_sq.sqrt()
            prefix = f"balance/{name}"
            metrics[prefix + "_ppo_loss"] = components[index].detach()
            metrics[prefix + "_jepa_loss"] = components[index + 2].detach()
            metrics[prefix + "_ppo_grad_norm"] = primary_norm
            metrics[prefix + "_jepa_grad_norm"] = auxiliary_norm
            metrics[prefix + "_jepa_to_ppo_grad_ratio"] = auxiliary_norm / primary_norm.clamp_min(1e-12)
            metrics[prefix + "_grad_cosine"] = dot / (primary_norm * auxiliary_norm).clamp_min(1e-12)
    return metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.jepa_mode not in {"actor", "critic", "both", "none"}:
        raise ValueError("invalid JEPA mode")
    if not np.isfinite(args.jepa_coef) or args.jepa_coef < 0:
        raise ValueError("jepa_coef must be finite and nonnegative")
    if args.jepa_layer not in {"first", "final"}:
        raise ValueError("invalid JEPA layer")
    if args.jepa_projection not in {"concat", "adaln"}:
        raise ValueError("invalid JEPA projection")
    if args.jepa_critic_condition not in {"action", "value"}:
        raise ValueError("invalid JEPA critic condition")
    if args.balance_interval < 0:
        raise ValueError("balance_interval must be nonnegative")
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
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); FP32; native-action storage; host actor mirror")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args.jepa_mode, args.jepa_layer,
                      args.jepa_projection, args.jepa_critic_condition).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values, next_observations):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values,
                            args, next_observations)

        def component_model(observations, native, old_logprobs, advantages, returns, old_values, next_observations):
            return loss_components(agent, observations, native, old_logprobs, advantages, returns, old_values,
                                   args, next_observations)[0]

        if args.compile:
            # Multiple per-objective backward calls must not alias CUDA graph buffers.
            component_model = torch.compile(component_model, fullgraph=True, dynamic=False,
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
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        use_jepa = args.jepa_mode != "none"
        fields = {"observations": obs_shape, "native_actions": (agent.action_dim,)}
        if use_jepa:
            fields["next_observations"] = obs_shape
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields=fields)
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 10 if use_jepa else 6), device=device)
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
                    if use_jepa:
                        # Includes physical terminal/time-limit observations, never reset states.
                        transfer.push(step, reward, terms, truncs, observations=obs_step,
                                      native_actions=native, next_observations=transition_obs)
                    else:
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
                b_next_obs = batch.fields["next_observations"].flatten(0, 1) if use_jepa else None
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
            updates = 0
            balance_metrics = {}
            measure_balance = (use_jepa and args.balance_interval > 0
                               and (iteration == 1 or iteration % args.balance_interval == 0))
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        minibatch = (
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                            b_next_obs[indices] if use_jepa else None,
                        )
                        if measure_balance and updates == 0:
                            components = component_model(*minibatch)
                            balance_metrics = gradient_balance(agent, components)
                            del components
                        loss, metrics = loss_model(*minibatch)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            metric_values = {
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
            }
            if use_jepa:
                for offset, name in enumerate(("actor_loss", "critic_loss", "actor_latent_std", "critic_latent_std"), 6):
                    metric_values[f"jepa/{name}"] = update_metrics[:updates, offset].mean()
            metric_values.update(balance_metrics)
            logged = gather_metrics(metric_values)
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

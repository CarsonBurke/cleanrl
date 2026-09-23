# Shared-encoder LeWM PPO v4: one 64-D JEPA encoder, separate full PPO FFNs.
# Each selected policy/value FFN has baseline 64x64 Tanh hidden layers and reads
# detached features. One attached temporal prediction + SIGReg objective trains
# the shared encoder; auxiliary modules are never part of rollout inference.
# Hypothesis: nonlinear task readouts recover control capacity lost by v3's
# linear probes. This preserves PPO depth/width, not its total parameter count.
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

from cleanrl.shared.lejepa import ActionEncoder, FeedForward, MLP, SIGReg
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
    """which PPO FFNs read the shared detached JEPA encoder; others read observations"""
    sigreg_weight: float = 0.09
    """LeWM coefficient on the raw, batch-size-scaled Epps-Pulley statistic"""
    sigreg_num_proj: int = 1024
    """fresh random projection directions per SIGReg call, matching le-wm"""
    sigreg_proj_chunk: int = 256
    """directions per memory-bounded chunk; does not change the statistic"""
    ssl_learning_rate: float = 5e-5
    """constant representation AdamW learning rate, matching le-wm"""
    ssl_weight_decay: float = 1e-3
    """representation AdamW weight decay, matching le-wm"""
    balance_interval: int = 10
    """sample prediction/SIGReg encoder-gradient balance every N rollouts; 0 disables"""

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


class ActionConditionedMLP(nn.Module):
    """LeWM's AdaLN-zero feed-forward path, without attention or temporal tokens."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(64, 64)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(64, 3 * 64))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.output_norm = nn.LayerNorm(64)

    def forward(self, latent, condition):
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        hidden = self.norm(latent) * (1.0 + scale) + shift
        return self.output_norm(latent + gate * self.mlp(hidden))


class LeWMBranch(nn.Module):
    """Training-only LeWM projection/prediction stack on the shared encoder.

    Use the repo's LayerNorm MLP port (not running-stat BatchNorm) for CUDA-graph
    safety. Its output is unbounded: SIGReg must not test bounded tanh features.
    The statistic sees (time=2, batch, latent), testing each temporal marginal.
    """

    def __init__(self, action_dim, num_proj, proj_chunk):
        super().__init__()
        self.projector = MLP(64, 64, 64)
        self.action_encoder = ActionEncoder(action_dim, 64)
        self.predictor = ActionConditionedMLP()
        self.pred_proj = MLP(64, 64, 64)
        self.sigreg = SIGReg(knots=17, num_proj=num_proj, proj_chunk=proj_chunk)

    def forward(self, current, following, native_actions):
        # A single shared projector encodes both attached temporal branches.
        embeddings = self.projector(torch.stack((current, following)))
        prediction = self.pred_proj(self.predictor(
            embeddings[0], self.action_encoder(2.0 * native_actions - 1.0),
        ))
        prediction_loss = F.mse_loss(prediction, embeddings[1])
        regularization = self.sigreg(embeddings)
        with torch.no_grad():
            projected_std = embeddings.std(dim=1, correction=0).mean()
            backbone_std = following.std(dim=0, correction=0).mean()
        return prediction_loss, regularization, projected_std, backbone_std


class Agent(nn.Module):
    def __init__(self, envs, jepa_mode="both", sigreg_num_proj=1024, sigreg_proj_chunk=256):
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
        if jepa_mode not in {"actor", "critic", "both", "none"}:
            raise ValueError("invalid JEPA mode")
        if min(sigreg_num_proj, sigreg_proj_chunk) <= 0:
            raise ValueError("SIGReg projection counts must be positive")
        self.actor_uses_encoder = jepa_mode in {"actor", "both"}
        self.critic_uses_encoder = jepa_mode in {"critic", "both"}
        # Preserve baseline initialization order and exact none-mode behavior.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64 if self.critic_uses_encoder else observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(64 if self.actor_uses_encoder else observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
        ) if jepa_mode != "none" else None
        self.ssl = LeWMBranch(self.action_dim, sigreg_num_proj, sigreg_proj_chunk) if self.encoder is not None else None

    def parameter_groups(self):
        """PPO owns both complete FFNs; SSL alone owns the shared representation."""
        policy = tuple(self.actor.parameters()) + tuple(self.critic.parameters())
        representation = (() if self.encoder is None else
                          tuple(self.encoder.parameters()) + tuple(self.ssl.parameters()))
        return policy, representation

    def parameter_counts(self):
        counts = {
            "shared_encoder": 0 if self.encoder is None else sum(p.numel() for p in self.encoder.parameters()),
            "actor_ffn": sum(p.numel() for p in self.actor.parameters()),
            "critic_ffn": sum(p.numel() for p in self.critic.parameters()),
        }
        counts["inference"] = sum(counts.values())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["training_only"] = counts["total"] - counts["inference"]
        return counts

    def rollout_actor(self):
        """Flatten the shared encoder + actor for the native host graph.

        References the live modules, not copies: refresh includes SSL updates.
        Detach is irrelevant to inference, which never builds an autograd graph.
        """
        if self.actor_uses_encoder:
            return nn.Sequential(*self.encoder, *self.actor)
        return self.actor

    def ssl_losses(self, latent, next_observations, native_actions):
        prediction, regularization, projected_std, backbone_std = self.ssl(
            latent, self.encoder(next_observations), native_actions,
        )
        metrics = torch.stack(tuple(term.detach() for term in (
            prediction, regularization, projected_std, backbone_std,
        )))
        return prediction, regularization, metrics

    def get_value(self, x):
        features = self.encoder(x).detach() if self.critic_uses_encoder else x
        return self.critic(features)

    def get_policy_and_value(self, x):
        alpha, beta, value, _ = self.get_policy_value_latents(x)
        return alpha, beta, value

    def get_policy_value_latents(self, x):
        # One encoder evaluation and one shared feature tensor for both heads.
        latent = self.encoder(x) if self.encoder is not None else None
        detached = latent.detach() if latent is not None else None
        logits = self.actor(detached if self.actor_uses_encoder else x)
        value = self.critic(detached if self.critic_uses_encoder else x)
        alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
        return alpha, beta, value, latent

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
    """PPO actor/value and a single shared prediction/SIGReg objective."""
    alpha, beta, newvalue, latent = agent.get_policy_value_latents(observations)
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
    prediction = regularization = observations.new_zeros(())
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac))
    if args.jepa_mode != "none":
        prediction, regularization, ssl_metrics = agent.ssl_losses(
            latent, next_observations, native_actions,
        )
        metrics = torch.cat((metrics, ssl_metrics))
    components = torch.stack((actor_objective, critic_objective,
                              prediction, args.sigreg_weight * regularization))
    return components, metrics


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args, next_observations=None):
    components, metrics = loss_components(
        agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args, next_observations,
    )
    return components[0] + components[1], components[2:].sum(), metrics


def gradient_balance(agent, components):
    """Measure prediction versus weighted SIGReg on the single shared encoder."""
    if agent.encoder is None:
        return {}
    parameters = tuple(agent.encoder.parameters())
    prediction = torch.autograd.grad(components[2], parameters, retain_graph=True)
    regularization = torch.autograd.grad(components[3], parameters, retain_graph=True)
    with torch.no_grad():
        prediction_norm = torch.stack([grad.square().sum() for grad in prediction]).sum().sqrt()
        sigreg_norm = torch.stack([grad.square().sum() for grad in regularization]).sum().sqrt()
        dot = torch.stack([(left * right).sum() for left, right in zip(prediction, regularization)]).sum()
        return {
            "balance/shared_prediction_grad_norm": prediction_norm,
            "balance/shared_sigreg_grad_norm": sigreg_norm,
            "balance/shared_sigreg_to_prediction_grad_ratio": sigreg_norm / prediction_norm.clamp_min(1e-12),
            "balance/shared_grad_cosine": dot / (prediction_norm * sigreg_norm).clamp_min(1e-12),
        }


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.jepa_mode not in {"actor", "critic", "both", "none"}:
        raise ValueError("invalid JEPA mode")
    if not np.isfinite(args.sigreg_weight) or args.sigreg_weight < 0:
        raise ValueError("sigreg_weight must be finite and nonnegative")
    if min(args.sigreg_num_proj, args.sigreg_proj_chunk) <= 0:
        raise ValueError("SIGReg projection counts must be positive")
    if not np.isfinite(args.ssl_learning_rate) or args.ssl_learning_rate <= 0:
        raise ValueError("ssl_learning_rate must be finite and positive")
    if not np.isfinite(args.ssl_weight_decay) or args.ssl_weight_decay < 0:
        raise ValueError("ssl_weight_decay must be finite and nonnegative")
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
        agent = Agent(envs, args.jepa_mode, args.sigreg_num_proj, args.sigreg_proj_chunk).to(device)
        policy_parameters, ssl_parameters = agent.parameter_groups()
        optimizer = optim.Adam(policy_parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        ssl_optimizer = optim.AdamW(ssl_parameters, lr=args.ssl_learning_rate,
                                   weight_decay=args.ssl_weight_decay, fused=True) if ssl_parameters else None
        counts = agent.parameter_counts()
        writer.add_text("parameter_counts", str(counts))
        print(f"parameter_counts={counts}")
        for name, count in counts.items():
            writer.add_scalar(f"parameters/{name}", count, 0)
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
        host_actor = make_host_mirror(agent.rollout_actor(), args.num_envs)
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
        gradient_norms = torch.empty((max_updates, 2), device=device)
        zero_norm = torch.zeros((), device=device)
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
                        policy_loss, ssl_loss, metrics = loss_model(*minibatch)
                        optimizer.zero_grad(set_to_none=True)
                        if ssl_optimizer is not None:
                            ssl_optimizer.zero_grad(set_to_none=True)
                        (policy_loss + ssl_loss).backward()
                        # Separate clipping is necessary: a large SIGReg gradient must not
                        # shrink a detached PPO head's update through a global norm.
                        gradient_norms[updates, 0] = nn.utils.clip_grad_norm_(policy_parameters, args.max_grad_norm)
                        gradient_norms[updates, 1] = (
                            nn.utils.clip_grad_norm_(ssl_parameters, args.max_grad_norm)
                            if ssl_optimizer is not None else zero_norm
                        )
                        optimizer.step()
                        if ssl_optimizer is not None:
                            ssl_optimizer.step()
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
            metric_values["gradients/ppo_norm"] = gradient_norms[:updates, 0].mean()
            if use_jepa:
                metric_values["gradients/ssl_norm"] = gradient_norms[:updates, 1].mean()
                for offset, name in enumerate(("prediction_loss", "sigreg_loss", "projected_std", "backbone_std")):
                    metric_values[f"ssl/shared_{name}"] = update_metrics[:updates, 6 + offset].mean()
            metric_values.update(balance_metrics)
            logged = gather_metrics(metric_values)
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            if ssl_optimizer is not None:
                writer.add_scalar("charts/ssl_learning_rate", ssl_optimizer.param_groups[0]["lr"], global_step)
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

# Pre-RMS SiTU-GLU PPO v4 plus a NextLat-style self-predictive auxiliary on the
# CRITIC trunk only. Mechanism: read the critic trunk at unit RMS
# (h = trunk(s)/output_scale, the value head still consumes h*output_scale, so the
# value path is an exact algebraic re-parameterization), predict the next step's
# feature with a residual MLP conditioned on the native Beta sample
# (hhat = h + MLP(RMSNorm(cat[Linear(A,64)(a), h])), N(0,0.02) init, no biases, so it
# starts at the identity), and add two terms at horizon 1, plain sum, no warmup:
# masked element-mean SmoothL1 to the detached successor feature, and the
# unit-variance-Gaussian-KL analogue of NextLat's decode KL, i.e. half the squared
# value error read out through a weight- AND bias-detached value head.
# Porting choices: predictor parameters join optimizer.param_groups[0] (the only group
# the LR anneal touches) and join the CRITIC gradient-clip tuple, so the auxiliary
# competes for the critic's 0.5 norm budget. The actor owns a separate trunk and
# receives ZERO auxiliary gradient. Successor of flattened row i is row i+num_envs,
# eligible iff step < num_steps-1 and the step neither terminated nor truncated;
# indices are clamped plus masked, never sliced, to keep cudagraph shapes static.
# Honest scope: NextLat's belief-state theorem is VACUOUS here -- a 3-block MLP on a
# Markov MuJoCo observation has no history to compress, h_t = f(s_t) already. And the
# decode term loses ~50000x its constraining power: a 50304-way softmax pins nearly
# every direction of the reference latent, Linear(64,1) pins exactly one. So the
# 63-dimensional null space of the critic head carries no supervision and the latent
# loss has a free degenerate minimum. What is actually being tested is SPR-style
# self-prediction as a smoothness prior on critic features, not NextLat's theorem.
# Falling nextlat/latent_loss together with falling nextlat/feature_dispersion is
# COLLAPSE, not progress. Hypothesis: one-step predictability of critic features
# regularizes the 32x-LR critic enough to improve value accuracy and returns;
# --nextlat off reproduces the baseline exactly and is the in-file control.
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
from cleanrl.shared.sampling import sample_beta_actions, sample_beta_actions_host
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
    total_timesteps: int = 50000000
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

    nextlat: Literal["off", "on"] = "on"
    """self-predictive critic-feature auxiliary; off reproduces the v4 baseline exactly"""
    nextlat_latent_coef: float = 1.0
    """coefficient on masked element-mean SmoothL1 against the detached successor feature"""
    nextlat_value_coef: float = 1.0
    """coefficient on the decoded-value term through the detached value head"""

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
        self.critic = nn.Sequential(
            make_norm_residual_trunk(
                observation_dim, 64, placement=placement, norm_kind=norm_kind, activation=activation
            ),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            make_norm_residual_trunk(
                observation_dim, 64, placement=placement, norm_kind=norm_kind, activation=activation
            ),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        # The trunk ends in `h * output_scale` (= width**-0.5), so its features sit at
        # RMS 0.125. The auxiliary regresses them at unit RMS instead, which is the
        # scale NextLat's coefficients were tuned at; per-element squared error would
        # otherwise be 64x smaller and lambda=1.0 would not transfer.
        self.critic_output_scale = float(self.critic[0].output_scale)
        self.critic_inverse_scale = 1.0 / self.critic_output_scale

    def get_value(self, x):
        return self.critic(x)

    def get_policy_and_value(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(x)

    def get_policy_value_and_features(self, x):
        """One critic trunk pass feeding both readouts: the value head consumes the
        trunk's native output, the auxiliary the same features at unit RMS."""
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        trunk_out = self.critic[0](x)
        return alpha, beta, self.critic[1](trunk_out), trunk_out * self.critic_inverse_scale

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


class NextLatPredictor(nn.Module):
    """Residual next-feature predictor transcribed from NextLat's dynamics model.

    ``hhat = h + W3 GELU(W2 GELU(W1 RMSNorm(cat[E(a), h])))``, hidden width
    ``128 * round(proj_factor * 2 * width / 128)``, every linear at ``N(0, 0.02)`` with
    no bias. The affine pre-MLP norm and the residual are both load-bearing: together
    they make the predictor start at the identity, so the objective is "predict the
    change" rather than "reconstruct the state". Conditioning uses the native Beta
    sample in (0,1)^A -- bounded, and already the policy's own coordinate system.
    """

    def __init__(self, action_dim, width=64, proj_factor=1.0):
        super().__init__()
        input_dim = 2 * width
        hidden_dim = 128 * max(1, round(proj_factor * input_dim / 128))
        self.action_embed = nn.Linear(action_dim, width, bias=False)
        self.norm_x = nn.RMSNorm(input_dim, eps=1e-5, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, width, bias=False),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, features, native_actions):
        conditioned = torch.cat((self.action_embed(native_actions), features), dim=-1)
        return features + self.mlp(self.norm_x(conditioned))


def successor_pairing(indices, eligible, num_envs, batch_size):
    """Successor positions *inside* a full-batch minibatch, with static shapes.

    Row ``i`` of the flattened ``(T, N)`` rollout is followed by row ``i + num_envs``.
    ``indices`` is a permutation of the whole batch (guaranteed by the
    ``num_minibatches == 1`` validation), so its inverse maps global rows back to
    minibatch positions. Last-step rows are clamped to a valid row rather than sliced
    away and carry weight 0, which is what keeps Inductor's cudagraph shapes fixed.
    """
    inverse = torch.empty_like(indices)
    inverse[indices] = torch.arange(indices.shape[0], device=indices.device)
    successors = (indices + num_envs).clamp_max(batch_size - 1)
    return inverse[successors], eligible[indices]


def nextlat_terms(predictor, features, native_actions, successor_index, eligible_weight, value_head, output_scale):
    """Horizon-1 self-prediction losses plus collapse diagnostics.

    The latent term is masked SmoothL1 reduced over masked *elements* (a row-count
    denominator would inflate it, and its gradient, by the 64 feature coordinates).
    The decoded term is the scalar-readout analogue of NextLat's decode KL: a
    unit-variance Gaussian KL is half the squared mean error, and the value head's
    weight *and* bias are detached on both sides so the auxiliary cannot satisfy it by
    shrinking the head. Only the source feature carries gradient; the successor
    teacher is detached even though the same trunk produced it.
    """
    predicted = predictor(features, native_actions)
    target = features.detach()[successor_index]
    weight = eligible_weight.unsqueeze(-1)
    errors = F.smooth_l1_loss(predicted, target, reduction="none")
    elements = weight.expand_as(errors).sum().clamp_min(1.0)
    latent_loss = (errors * weight).sum() / elements
    head_weight = value_head.weight.detach()
    head_bias = value_head.bias.detach()
    student = F.linear(predicted * output_scale, head_weight, None) + head_bias
    teacher = F.linear(target * output_scale, head_weight, None) + head_bias
    rows = eligible_weight.sum().clamp_min(1.0)
    value_loss = (0.5 * (student - teacher).square().sum(-1) * eligible_weight).sum() / rows
    with torch.no_grad():
        # Identity-copy watchdog: a predictor that learns delta ~ 0 reports ~0 here.
        residual_ratio = (predicted - features).square().mean().sqrt() / features.square().mean().sqrt().clamp_min(
            1e-12
        )
        # Dispersion of centred, L2-normalized rows about their mean direction. The
        # closed form 1 - ||mean(u)||^2 is evaluated in fp64 because it loses
        # resolution decades above the collapse it is watching.
        centred = features - features.mean(dim=0, keepdim=True)
        unit = centred / centred.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        mean_direction = unit.to(torch.float64).mean(dim=0)
        dispersion = (1.0 - mean_direction.square().sum()).clamp_min(0.0).to(features.dtype)
    return latent_loss, value_loss, residual_ratio, dispersion


def ppo_loss(
    agent,
    predictor,
    observations,
    native_actions,
    old_logprobs,
    advantages,
    targets,
    old_values,
    successor_index,
    eligible_weight,
    args,
):
    """Clipped surrogate, optionally clipped scalar MSE, and the critic auxiliary."""
    alpha, beta, newvalue, features = agent.get_policy_value_and_features(observations)
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
        v_loss_unclipped = (newvalue - targets) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - targets) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - targets) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    if predictor is None:
        latent_loss = value_loss = residual_ratio = dispersion = torch.zeros_like(pg_loss.detach())
    else:
        latent_loss, value_loss, residual_ratio, dispersion = nextlat_terms(
            predictor,
            features,
            native_actions,
            successor_index,
            eligible_weight,
            agent.critic[1],
            agent.critic_output_scale,
        )
        loss = loss + args.nextlat_latent_coef * latent_loss + args.nextlat_value_coef * value_loss
    metrics = torch.stack(
        (
            pg_loss.detach(),
            v_loss.detach(),
            entropy_loss.detach(),
            old_approx_kl,
            approx_kl,
            clipfrac,
            latent_loss.detach(),
            value_loss.detach(),
            residual_ratio,
            dispersion,
        )
    )
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
    if args.nextlat == "on":
        if args.num_minibatches != 1:
            raise ValueError(
                "the NextLat critic auxiliary pairs row i with row i+num_envs of the whole "
                "batch; num_minibatches must be 1 so every successor stays in the minibatch"
            )
        if min(args.nextlat_latent_coef, args.nextlat_value_coef) < 0.0:
            raise ValueError("NextLat auxiliary coefficients must be non-negative")
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
            f"{args.activation} residual trunk; 32x LR; 1 minibatch; raw GAE",
        )
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, placement=args.placement, norm_kind=args.norm_kind, activation=args.activation).to(device)
        aux_enabled = args.nextlat == "on"
        predictor = NextLatPredictor(agent.action_dim, 64).to(device) if aux_enabled else None
        # One param group, so the anneal line below reaches the predictor too.
        trained_parameters = list(agent.parameters())
        if predictor is not None:
            trained_parameters += list(predictor.parameters())
        optimizer = optim.Adam(trained_parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        actor_parameters = tuple(agent.actor.parameters())
        critic_parameters = tuple(agent.critic.parameters())
        if predictor is not None:
            # The auxiliary shares the critic's max_grad_norm budget.
            critic_parameters += tuple(predictor.parameters())
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(
            observations, native, old_logprobs, advantages, targets, old_values, successor_index, eligible_weight
        ):
            return ppo_loss(
                agent,
                predictor,
                observations,
                native,
                old_logprobs,
                advantages,
                targets,
                old_values,
                successor_index,
                eligible_weight,
                args,
            )

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)

        def act(observations):
            native, action = sample_beta_actions_host(host_actor(observations), action_low, action_high, sampler)
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
        update_metrics = torch.empty((max_updates, 10), device=device)
        grad_norms = torch.empty((max_updates, 2), device=device)
        # Last-step rows have no successor inside the rollout; mask them out once.
        final_step_mask = torch.ones((args.num_steps, args.num_envs), device=device)
        final_step_mask[-1] = 0.0
        idle_pairing = (
            torch.zeros(args.minibatch_size, dtype=torch.long, device=device),
            torch.zeros(args.minibatch_size, device=device),
        )
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
                b_values, b_logprobs = rollout_statistics(b_obs, b_native)
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
                # A transition is self-predictable only if the episode continued past
                # it, on the unshuffled (T, N) layout. Computed once per rollout.
                continued = (1.0 - batch.terminations) * (1.0 - batch.truncations)
                b_eligible = (continued * final_step_mask).flatten().clone()
                eligible_fraction = b_eligible.mean()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if aux_enabled:
                            pairing = successor_pairing(indices, b_eligible, args.num_envs, args.batch_size)
                        else:
                            pairing = idle_pairing
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices],
                            b_native[indices],
                            b_logprobs[indices],
                            b_advantages[indices],
                            b_returns[indices],
                            b_values[indices],
                            pairing[0],
                            pairing[1],
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
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

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
                    "grad/actor_preclip_norm": mean_grad_norms[0],
                    "grad/critic_preclip_norm": mean_grad_norms[1],
                    "grad/actor_clip_fraction": grad_clip_fractions[0],
                    "grad/critic_clip_fraction": grad_clip_fractions[1],
                    "nextlat/latent_loss": last[6],
                    "nextlat/value_loss": last[7],
                    "nextlat/residual_ratio": last[8],
                    "nextlat/feature_dispersion": last[9],
                    "nextlat/eligible_fraction": eligible_fraction,
                }
            )
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
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

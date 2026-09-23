# PPO with one-step action feedback, v1.
#
# The policy is conditioned on the normalized action it produced at the
# previous environment step: pi(a_t | s_t, a_{t-1}). This makes the actor's
# own behavior an explicit input instead of forcing the MLP to infer short
# action history from observations. The critic remains V(s), matching the
# standard PPO baseline and avoiding a bootstrap contract change for
# time-limit terminal observations.
#
# Native Beta actions in [0, 1] are used as feedback, so the input is
# independent of physical action bounds. At episode boundaries feedback is
# reset to the Beta midpoint (0.5), preventing actions from one episode from
# leaking into the next. The rollout stores the exact actor input used for
# every sampled action; old log-probabilities and PPO updates therefore
# evaluate the same conditional policy.
#
# Hypothesis: action feedback gives the policy a cheap one-step memory channel
# for momentum and actuator persistence, improving control without recurrent
# state or a larger rollout-time model.
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

from cleanrl.shared.hl_gauss_std import StandardizedHistogram
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache,
    device_minibatches,
    explained_variance,
    gather_metrics,
    get_gae_fn,
)
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
    """PPO value trust region; the categorical arms use the decoded-value analogue"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum gradient norm, applied independently to actor and critic"""
    clip_heads: bool = False
    """include final actor/critic linear heads in their independent clipping budgets"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    reward_norm: bool = True
    """normalize/clip rewards before GAE; never normalize GAE return targets"""

    value_loss: Literal["mse", "mse_popart", "hlgauss", "mse_softmax"] = "mse"
    """scalar regression, scalar regression through the same per-rollout target
    standardization the categorical arms get, standardized HL-Gauss
    cross-entropy, or MSE decoded from a K-way head"""
    value_bins: int = 101
    """categorical head resolution; bin width is 2*half_span/(bins-1) target std"""
    value_half_span: float = 5.0
    """support half-width in target standard deviations"""
    value_sigma_bins: float = 0.75
    """HL-Gauss sigma as a multiple of the bin width (Farebrother et al. default)"""
    ce_scale: Literal["raw", "matched"] = "raw"
    """`matched` multiplies CE by Var_p(z)*scale^2 per sample, which is the exact
    first-order factor making its value-space gradient equal MSE's, so vf_coef
    keeps the meaning it was tuned with"""
    critic_width: int = 64
    """critic trunk width; the actor stays at 64"""

    placement: Literal["pre", "post"] = "pre"
    """normalize branch inputs with an identity stream, or residual outputs"""
    norm_kind: Literal["layer", "rms"] = "rms"
    """non-affine centered LayerNorm or uncentered RMSNorm, epsilon 1e-5"""
    activation: Literal["lrelusq", "stiglu"] = "stiglu"
    """squared leaky-ReLU pair or parameter-matched SiTU-GLU branch"""

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
    histogram: StandardizedHistogram | None
    critic: nn.Sequential
    actor: nn.Sequential

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
        self.categorical = args.value_loss in ("hlgauss", "mse_softmax")
        # `mse_popart` is the control the proxy demanded: a scalar head carrying
        # the *same* per-rollout location/scale bookkeeping as the categorical
        # arms, so any remaining categorical win is attributable to the loss
        # rather than to free absorption of global return drift.
        self.standardized = self.categorical or args.value_loss == "mse_popart"
        trunk_kwargs = dict(placement=args.placement, norm_kind=args.norm_kind, activation=args.activation)
        # Trunks and the actor head are built before the critic head so every arm
        # shares one initial policy and one initial critic trunk; only the value
        # readout differs. Small-gain categorical logits keep the initial value at
        # the support center while still giving the critic trunk a gradient on the
        # first step, which a zero-initialized head does not.
        critic_trunk = make_norm_residual_trunk(observation_dim, args.critic_width, **trunk_kwargs)
        # The actor receives the current observation and the previous native
        # action. The critic stays observation-only so its time-limit bootstrap
        # remains identical to the base PPO contract.
        self.actor_input_dim = observation_dim + self.action_dim
        self.actor = nn.Sequential(
            make_norm_residual_trunk(self.actor_input_dim, 64, **trunk_kwargs),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        if self.categorical:
            head = layer_init(nn.Linear(args.critic_width, args.value_bins), std=0.01)
        else:
            head = layer_init(nn.Linear(args.critic_width, 1), std=1.0)
        if self.standardized:
            self.histogram = StandardizedHistogram(
                args.value_bins,
                half_span=args.value_half_span,
                sigma_bins=args.value_sigma_bins,
                min_scale=1e-3,
            )
        else:
            self.histogram = None
        self.critic = nn.Sequential(critic_trunk, head)

    def decode(self, readout):
        """Scalar value of the critic readout, shaped (n, 1) for both heads."""
        if self.histogram is None:
            return readout
        if not self.categorical:
            return self.histogram.mean + self.histogram.scale * readout
        return self.histogram.decode(readout).unsqueeze(-1)

    def get_value(self, observations):
        return self.decode(self.critic(observations))

    def get_policy_and_readout(self, observations, actor_inputs):
        alpha, beta = (F.softplus(self.actor(actor_inputs)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(observations)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, observations, action=None, previous_native_action=None):
        """Public API uses physical actions; feedback is native Beta action."""
        if previous_native_action is None:
            previous_native_action = torch.full(
                (observations.shape[0], self.action_dim),
                0.5,
                dtype=observations.dtype,
                device=observations.device,
            )
        actor_inputs = torch.cat((observations.flatten(1), previous_native_action.flatten(1)), dim=-1)
        alpha, beta, readout = self.get_policy_and_readout(observations, actor_inputs)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((observations.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(observations.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, self.decode(readout)


def clipping_parameters(agent, clip_heads):
    """Select each independent budget once; excluded heads still belong to Adam."""
    actor = agent.actor if clip_heads else agent.actor[0]
    critic = agent.critic if clip_heads else agent.critic[0]
    return tuple(actor.parameters()), tuple(critic.parameters())


def clip_gradients(actor_parameters, critic_parameters, max_grad_norm):
    """Clip each network independently after the weighted joint loss backward."""
    actor_preclip_norm = nn.utils.clip_grad_norm_(actor_parameters, max_grad_norm, foreach=True)
    critic_preclip_norm = nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm, foreach=True)
    return actor_preclip_norm, critic_preclip_norm


def scalar_value_loss(newvalue, targets, old_values, args):
    """PPO's clipped scalar regression, unchanged from the baseline."""
    squared = (newvalue - targets) ** 2
    if not args.clip_vloss:
        return 0.5 * squared.mean(), torch.zeros((), device=newvalue.device)
    clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
    clipped_squared = (clipped - targets) ** 2
    frozen = (clipped_squared > squared).float()
    return 0.5 * torch.max(squared, clipped_squared).mean(), frozen.mean()


def categorical_value_loss(agent, readout, newvalue, targets, old_values, target_probs, args):
    """Cross-entropy on frozen HL-Gauss labels with the decoded-value trust region.

    The trust region is the scalar rule's own decision, applied to the decoded
    value: PPO's `max(unclipped, clipped)` contributes zero value gradient
    exactly when the clipped branch is the larger error, because `clamp` is
    saturated there. The same sample set is dropped from the cross-entropy.
    """
    log_probs = readout.log_softmax(dim=-1)
    cross_entropy = -(target_probs * log_probs).sum(dim=-1)
    if args.clip_vloss:
        with torch.no_grad():
            clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
            frozen = ((clipped - targets).square() > (newvalue - targets).square()).float()
        cross_entropy = cross_entropy * (1.0 - frozen)
        clipfrac = frozen.mean()
    else:
        clipfrac = torch.zeros((), device=readout.device)
    if args.ce_scale == "matched":
        with torch.no_grad():
            gain = agent.histogram.value_gradient_gain(log_probs.exp())
        cross_entropy = cross_entropy * gain
    return cross_entropy.mean(), clipfrac


def policy_terms(agent, actor_inputs, observations, native_actions, old_logprobs, advantages, args):
    alpha, beta, readout = agent.get_policy_and_readout(observations, actor_inputs)
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
    return readout, pg_loss, entropy.mean(), (old_approx_kl, approx_kl, clipfrac)


def assemble(pg_loss, entropy_loss, v_loss, v_clipfrac, diagnostics, args):
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    old_approx_kl, approx_kl, clipfrac = diagnostics
    metrics = torch.stack(
        (
            pg_loss.detach(),
            v_loss.detach(),
            entropy_loss.detach(),
            old_approx_kl,
            approx_kl,
            clipfrac,
            v_clipfrac,
        )
    )
    return loss, metrics


def scalar_ppo_loss(
    agent, actor_inputs, observations, native_actions, old_logprobs, advantages, targets, old_values, args
):
    readout, pg_loss, entropy_loss, diagnostics = policy_terms(
        agent, actor_inputs, observations, native_actions, old_logprobs, advantages, args
    )
    newvalue = agent.decode(readout).view(-1)
    v_loss, v_clipfrac = scalar_value_loss(newvalue, targets, old_values, args)
    return assemble(pg_loss, entropy_loss, v_loss, v_clipfrac, diagnostics, args)


def categorical_ppo_loss(
    agent,
    actor_inputs,
    observations,
    native_actions,
    old_logprobs,
    advantages,
    targets,
    old_values,
    target_probs,
    args,
):
    readout, pg_loss, entropy_loss, diagnostics = policy_terms(
        agent, actor_inputs, observations, native_actions, old_logprobs, advantages, args
    )
    newvalue = agent.histogram.decode(readout)
    if args.value_loss == "mse_softmax":
        v_loss, v_clipfrac = scalar_value_loss(newvalue, targets, old_values, args)
    else:
        v_loss, v_clipfrac = categorical_value_loss(
            agent, readout, newvalue, targets, old_values, target_probs, args
        )
    return assemble(pg_loss, entropy_loss, v_loss, v_clipfrac, diagnostics, args)


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.critic_width <= 0:
        raise ValueError("critic_width must be positive")
    if args.value_loss != "hlgauss" and args.ce_scale != "raw":
        raise ValueError("ce_scale only applies to value_loss='hlgauss'")
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
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        histogram = agent.histogram
        writer.add_text(
            "critic",
            f"value_loss={args.value_loss}; bins={args.value_bins}; "
            f"half_span={args.value_half_span} target std; "
            f"bin_width={0 if histogram is None else histogram.bin_width:.4g} target std; "
            f"sigma={0 if histogram is None else histogram.sigma:.4g} target std; "
            f"ce_scale={args.ce_scale}; "
            f"critic_width={args.critic_width}; clip_vloss={args.clip_vloss}; "
            "actor_feedback=native_previous_action",
        )
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        actor_parameters, critic_parameters = clipping_parameters(agent, args.clip_heads)
        clip_scope = "" if args.clip_heads else "_trunk"
        value_model = agent.get_value

        def rollout_statistics(actor_inputs, observations, native):
            """Old log-probabilities and values for the exact conditional rollout."""
            alpha, beta, readout = agent.get_policy_and_readout(observations, actor_inputs)
            return agent.decode(readout).flatten(), agent.action_logprob(alpha, beta, native)

        if not agent.categorical:

            def loss_model(
                actor_inputs, observations, native, old_logprobs, advantages, targets, old_values, target_probs
            ):
                return scalar_ppo_loss(
                    agent,
                    actor_inputs,
                    observations,
                    native,
                    old_logprobs,
                    advantages,
                    targets,
                    old_values,
                    args,
                )

        else:

            def loss_model(
                actor_inputs, observations, native, old_logprobs, advantages, targets, old_values, target_probs
            ):
                return categorical_ppo_loss(
                    agent,
                    actor_inputs,
                    observations,
                    native,
                    old_logprobs,
                    advantages,
                    targets,
                    old_values,
                    target_probs,
                    args,
                )

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        actor_input_shape = (int(np.prod(obs_shape)) + agent.action_dim,)
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # feedback-conditioned actor, stage everything in pinned memory, upload once.
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)

        def act(actor_inputs):
            native, action = sample_beta_actions_host(host_actor(actor_inputs), action_low, action_high, sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(
            args.num_steps,
            args.num_envs,
            obs_shape,
            device,
            non_blocking=args.non_blocking_transfers,
            fields={"observations": obs_shape, "actor_inputs": actor_input_shape, "native_actions": (agent.action_dim,)},
        )
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma) if args.reward_norm else None
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 7), device=device)
        grad_norms = torch.empty((max_updates, 2), device=device)
        placeholder = torch.zeros(
            (args.minibatch_size, args.value_bins if histogram is not None else 1), device=device
        )
        overflow = label_error = torch.zeros((), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        previous_native = np.full((args.num_envs, agent.action_dim), 0.5, dtype=np.float32)
        warmup_step = 0
        warmup_reset_schedule = {}

        def warmup_action(observations):
            nonlocal warmup_step
            actor_inputs = np.concatenate((observations.reshape(args.num_envs, -1), previous_native), axis=1)
            native, action = act(actor_inputs)
            previous_native[...] = native
            warmup_step += 1
            for index in warmup_reset_schedule.get(warmup_step, ()):
                previous_native[index] = 0.5
            return action

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            warmup_reset_schedule = {}
            for index, offset in enumerate(phases):
                warmup_reset_schedule.setdefault(horizon - int(offset), []).append(index)
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
                    actor_inputs = np.concatenate((obs_step.reshape(args.num_envs, -1), previous_native), axis=1)
                    native, host_action = act(actor_inputs)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms) if rew_norm is not None else raw_reward
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(
                        step,
                        reward,
                        terms,
                        truncs,
                        observations=obs_step,
                        actor_inputs=actor_inputs,
                        native_actions=native,
                    )
                previous_native[...] = native
                previous_native[np.logical_or(terms, truncs)] = 0.5
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
                b_actor_inputs = batch.fields["actor_inputs"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs = rollout_statistics(b_actor_inputs, b_obs, b_native)
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
                b_old_values = b_values
                b_target_probs = placeholder
                if histogram is not None:
                    previous_mean, previous_scale = histogram.mean.clone(), histogram.scale.clone()
                    histogram.observe(b_returns)
                    # The head is unchanged, only the readout convention: move the
                    # rollout values into the new frame so the value trust region
                    # measures policy-update movement, not a normalizer step.
                    b_old_values = histogram.mean + histogram.scale * (b_values - previous_mean) / previous_scale
                    if agent.categorical:
                        b_target_probs = histogram.project(b_returns)
                        standardized = histogram.standardize(b_returns)
                        overflow = (standardized.abs() > histogram.half_span).float().mean()
                        label_error = (histogram.decode_probs(b_target_probs) - b_returns).abs().mean()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_actor_inputs[indices],
                            b_obs[indices],
                            b_native[indices],
                            b_logprobs[indices],
                            b_advantages[indices],
                            b_returns[indices],
                            b_old_values[indices],
                            b_target_probs[indices] if agent.categorical else placeholder,
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
            scalars = {
                "losses/policy_loss": last[0],
                "losses/value_loss": last[1],
                "losses/entropy": last[2],
                "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4],
                "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/value_clipfrac": update_metrics[:updates, 6].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                f"grad/actor{clip_scope}_preclip_norm": mean_grad_norms[0],
                f"grad/critic{clip_scope}_preclip_norm": mean_grad_norms[1],
                f"grad/actor{clip_scope}_clip_fraction": grad_clip_fractions[0],
                f"grad/critic{clip_scope}_clip_fraction": grad_clip_fractions[1],
            }
            if histogram is not None:
                scalars["value/target_mean"] = histogram.mean
                scalars["value/target_scale"] = histogram.scale
            if agent.categorical:
                scalars["value/support_overflow_fraction"] = overflow
                scalars["value/label_decode_error"] = label_error
            logged = gather_metrics(scalars)
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

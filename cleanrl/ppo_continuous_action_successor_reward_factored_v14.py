# Direct reward-factored successor critic v14: no JEPA, encoder or world model.
# Hypothesis: predicting the two observed task reward components gives the critic
# useful return structure without a compulsory self-supervised actor bottleneck.
# Both arms: separate raw-normalized-observation -> Tanh64 -> Tanh64 FFNs,
# identical Beta actor initialization/PPO, scalar reward-direction loss and schedule.
# SF predicts Psi=[discounted reward_run, discounted reward_ctrl]; V=Psi.sum(-1).
# These ARE the critic outputs, not detached slots consumed by another value MLP.
# Head parameters are value/orthogonal-contrast coordinates, followed by a fixed
# factor transform: adding a second factor cannot double the Adam value-head step.
# Actor Adam owns actor only (PPO/entropy); critic Adam owns BOTH critic stages and
# its entire head (scalar value loss + SF reward-orthogonal regression). Separate
# matched owner clipping prevents auxiliary loss from rescaling the actor update.
# All observed targets, old values/logprobs, advantages and bootstraps are detached.
# Reward basis is exact Gym/native HalfCheetah info, including final_info: run uses
# displacement/dt, NOT instantaneous qvel; ctrl=-weight*sum(clipped physical a^2).
# Preserve VectorRewardNorm in BOTH arms, including per-env, per-step moments and
# clipping. Scale BOTH observed factors by THAT transition's divisor/clipping;
# Psi predicts sums of these normalized factors, NEVER raw Psi times today's scale.
# No (1-gamma) occupancy factor: Psi, V, TD(lambda) and PPO returns share units.
# Auxiliary RMS is a detached rollout LOSS metric only: no target/prediction/head
# reparameterization, and the task-direction coefficient is independent of D.
# Risks: normalization is history-dependent/nonstationary and hidden from both
# critics; exact observed factorization does not make it Markov. SF adds 65 head
# parameters and one standardized contrast objective; this is not an isolation of
# capacity or gradient conditioning. Own-TD EV is not heldout value accuracy.
# No multi-timescale treatment in this first matched pair. Historical ~10k SF used
# three blocks/16 experts/shared trunk/raw rewards, NOT this separate 64x64 design.
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

from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import device_minibatches, explained_variance, gather_metrics, get_gae_fn
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
FACTOR_NAMES = ("reward_run", "reward_ctrl")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    save_model: bool = False
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 1024
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    critic_mode: Literal["scalar", "successor"] = "successor"
    """Only treatment: scalar return versus direct two-factor successor critic."""
    env_backend: Literal["native"] = "native"
    env_threads: int = 2
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    non_blocking_transfers: bool = False
    staggered_starts: bool = True
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class TaskFFN(nn.Module):
    """Proper two-stage 64x64 FFN; no shared encoder or detached hidden stage."""

    def __init__(self, input_dim, output_dim, output_std):
        super().__init__()
        self.first = nn.Sequential(layer_init(nn.Linear(input_dim, 64)), nn.Tanh())
        self.second = nn.Sequential(layer_init(nn.Linear(64, 64)), nn.Tanh())
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))

    def forward(self, x):
        return self.head(self.second(self.first(x)))


class HostPolicy:
    """One fused native FP32 actor graph, refreshed once after learner updates."""

    def __init__(self, agent, num_rows):
        self.fused = make_host_mirror(nn.Sequential(
            *agent.actor.first, *agent.actor.second, *agent.actor.head,
        ), num_rows)

    def refresh(self):
        self.fused.refresh()

    def __call__(self, observations):
        return self.fused(observations)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        if args.critic_mode not in {"scalar", "successor"}:
            raise ValueError("critic_mode must be scalar or successor")
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta PPO requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        self.successor_dim = 2 if args.critic_mode == "successor" else 1
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        # Actor and both critic stages consume exactly the same RNG in both arms.
        self.actor = TaskFFN(observation_dim, 2 * self.action_dim, 0.01)
        self.critic = TaskFFN(observation_dim, 1, 1.0)
        if self.successor_dim == 2:
            # Preserve the scalar value parameterization and its Adam update, not
            # just initial predictions. The second coordinate is reward contrast;
            # get_successor applies the fixed inverse transform to physical factors.
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                scalar = self.critic.head[0]
                contrast = layer_init(nn.Linear(64, 1), std=1.0)
                head = nn.Linear(64, 2)
                head.weight[0].copy_(scalar.weight[0])
                head.weight[1].copy_(contrast.weight[0])
                head.bias[0].copy_(scalar.bias[0])
                head.bias[1].copy_(contrast.bias[0])
                self.critic.head[0] = head

    def parameter_groups(self):
        """Exclusive actor/critic Adam owners; every trainable parameter is owned."""
        return tuple(self.actor.parameters()), tuple(self.critic.parameters())

    def parameter_counts(self):
        actor, critic = self.parameter_groups()
        actor_count = sum(p.numel() for p in actor)
        critic_count = sum(p.numel() for p in critic)
        return {"actor_ffn": actor_count, "critic_ffn": critic_count,
                "inference": actor_count + critic_count, "total": actor_count + critic_count}

    def get_successor(self, observations):
        coordinates = self.critic(observations.detach())
        if self.successor_dim == 1:
            return coordinates
        value, contrast = coordinates.unbind(-1)
        return torch.stack((0.5 * value + contrast / np.sqrt(2),
                            0.5 * value - contrast / np.sqrt(2)), dim=-1)

    def get_value(self, observations):
        return self.get_successor(observations).sum(dim=-1, keepdim=True)

    def get_policy_and_successor(self, observations):
        logits = self.actor(observations.detach())
        alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.get_successor(observations)

    def get_policy_and_value(self, observations):
        alpha, beta, successors = self.get_policy_and_successor(observations)
        return alpha, beta, successors.sum(dim=-1, keepdim=True)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action.detach()) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, x, action=None):
        """Public API uses physical actions; training stores unit-interval samples."""
        alpha, beta, value = self.get_policy_and_value(x)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((x.shape[0],) + self.action_shape)
        else:
            native = ((action.detach().reshape(x.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS,
            )
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, value


def observed_reward_factors(raw_rewards, terminations, truncations, infos):
    """Read actual task components, including completed transitions before reset.

    Source: shared/mujoco_env.py NativeMujocoVectorEnv.step_wait and Gymnasium
    0.29.1 half_cheetah_v4.py step/control_cost. reward_ctrl already incorporates
    native->physical scaling, ClipAction, input-dtype square/sum and task weight.
    Recomputing it from centered native samples would be wrong for general bounds.
    """
    raw = np.asarray(raw_rewards, dtype=np.float64)
    boundaries = np.logical_or(terminations, truncations)
    factors = np.empty((raw.size, 2), dtype=np.float64)
    active = ~boundaries
    for column, name in enumerate(FACTOR_NAMES):
        if np.any(active):
            if name not in infos:
                raise RuntimeError(f"missing factual reward component {name}")
            present = infos.get("_" + name)
            if present is not None and not np.all(np.asarray(present)[active]):
                raise RuntimeError(f"missing active reward component {name}")
            factors[active, column] = np.asarray(infos[name])[active]
    for index in np.flatnonzero(boundaries):
        finals = infos.get("final_info")
        present = infos.get("_final_info")
        if finals is None or (present is not None and not present[index]) or finals[index] is None:
            raise RuntimeError("completed transition missing factual final_info")
        for column, name in enumerate(FACTOR_NAMES):
            if name not in finals[index]:
                raise RuntimeError(f"final_info missing reward component {name}")
            factors[index, column] = finals[index][name]
    if not np.isfinite(factors).all() or not np.allclose(factors.sum(-1), raw, rtol=1e-10, atol=1e-10):
        raise ValueError("HalfCheetah observed reward factors do not reconstruct raw reward")
    return factors


def normalize_reward_factors(raw_factors, raw_rewards, normalized_rewards, reward_norm):
    """Scale each observed component by its transition's exact normalization map.

    Call AFTER reward_norm.normalize, before its next update. With sigma equal to
    that update's sqrt(var+eps), c=1/max(sigma, abs(raw_reward)/clip). Then c*r is
    exactly the clipped normalized reward, even at r=0; there is no r_norm/r 0/0.
    This is a factorization of observed normalized rewards, not a transform of
    existing/raw bootstrap predictions. Floating-point reconstruction is audited.
    """
    divisor = np.sqrt(reward_norm.variances + reward_norm.epsilon)
    if reward_norm.clip is not None:
        divisor = np.maximum(divisor, np.abs(raw_rewards) / reward_norm.clip)
    normalized = np.asarray(raw_factors, dtype=np.float64) / divisor[:, None]
    if not np.allclose(normalized.sum(-1), normalized_rewards, rtol=2e-6, atol=2e-7):
        raise ValueError("normalized factors do not reconstruct the stored PPO reward")
    return normalized


@torch.no_grad()
def successor_lambda_targets(features, predictions, next_predictions, terminations, truncations,
                             gamma, gae_lambda, gae_fn):
    """Direct discounted-return TD(lambda), with v8's explicit factual masks.

    All inputs are time-major (T,N,D), except masks (T,N). Termination cuts both
    bootstrap and trace, truncation keeps factual final bootstrap but cuts trace,
    and the rollout tail bootstraps exactly once without unavailable future data.
    Unlike v8 occupancy targets, no (1-gamma) multiplier changes PPO value units.
    """
    steps, environments, channels = features.shape
    terms = terminations.detach().unsqueeze(-1).expand(steps, environments, channels).reshape(steps, -1)
    truncs = truncations.detach().unsqueeze(-1).expand(steps, environments, channels).reshape(steps, -1)
    _, targets = gae_fn(
        features.detach().reshape(steps, -1), predictions.detach().reshape(steps, -1), terms, truncs,
        next_predictions.detach().reshape(steps, -1), gamma, gae_lambda,
    )
    return targets.reshape_as(features).detach().clone()


def reward_nullspace(x):
    """Orthogonal projection away from fixed w=ones reward readout."""
    return x - x.mean(dim=-1, keepdim=True)


@torch.no_grad()
def successor_loss_scale(targets):
    """One frozen uncentered RMS in the reward-null subspace, ONLY for the loss."""
    if targets.shape[-1] == 1:
        return targets.new_ones(())
    energy = reward_nullspace(targets.detach()).square().sum(-1).mean() / (targets.shape[-1] - 1)
    return (energy + 1e-8).sqrt().detach()


def critic_losses(predictions, targets, old_values, args, auxiliary_scale):
    """Task-direction weight is not averaged over channels or RMS-normalized.

    SF adds fixed-weight mean squared nullspace error normalized by frozen target
    RMS. This changes the loss metric, NEVER the prediction/target coordinates.
    In two factors the nullspace is the run-versus-control contrast. Both output
    rows and both representation stages are attached to the critic objectives.
    """
    targets, old_values = targets.detach(), old_values.detach()
    values, returns = predictions.sum(-1), targets.sum(-1)
    squared_error = (values - returns).square()
    if args.clip_vloss:
        clipped = old_values + (values - old_values).clamp(-args.clip_coef, args.clip_coef)
        squared_error = torch.maximum(squared_error, (clipped - returns).square())
    value_loss = 0.5 * squared_error.mean()
    if predictions.shape[-1] == 1:
        auxiliary_loss = value_loss.new_zeros(())
    else:
        residual = reward_nullspace(predictions - targets) / auxiliary_scale.detach()
        auxiliary_loss = 0.5 * residual.square().sum(-1).mean() / (predictions.shape[-1] - 1)
    return value_loss, auxiliary_loss


def policy_loss(agent, observations, native_actions, old_logprobs, advantages, targets,
                old_values, args, auxiliary_scale):
    """One forward/backward; PPO cannot differentiate through factual teachers."""
    alpha, beta, predictions = agent.get_policy_and_successor(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions.detach()) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1).mean()
    logratio = newlogprob - old_logprobs.detach()
    ratio = logratio.exp()
    advantages = advantages.detach()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = torch.maximum(-advantages * ratio,
                            -advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    value_loss, auxiliary_loss = critic_losses(predictions, targets, old_values, args, auxiliary_scale)
    objective = pg_loss - args.ent_coef * entropy + args.vf_coef * (value_loss + auxiliary_loss)
    with torch.no_grad():
        metrics = {
            "losses/policy_loss": pg_loss.detach(), "losses/value_loss": value_loss.detach(),
            "successor/nullspace_loss": auxiliary_loss.detach(), "losses/entropy": entropy.detach(),
            "losses/old_approx_kl": (-logratio).mean(),
            "losses/approx_kl": ((ratio - 1) - logratio).mean(),
            "losses/clipfrac": ((ratio - 1).abs() > args.clip_coef).float().mean(),
        }
    return objective, metrics


@torch.no_grad()
def rollout_statistics(agent, observations, native_actions):
    alpha, beta, predictions = agent.get_policy_and_successor(observations)
    return predictions, agent.action_logprob(alpha, beta, native_actions)


def optimizer_step(optimizers, parameters, max_grad_norm, norms):
    """Clip each exclusive owner separately after the one combined backward."""
    for index, group in enumerate(parameters):
        norms[index].copy_(nn.utils.clip_grad_norm_(group, max_grad_norm))
    for optimizer in optimizers:
        optimizer.step()


def validate_args(args):
    if args.env_id != "HalfCheetah-v4" or args.env_backend != "native":
        raise ValueError("exact reward basis requires native HalfCheetah-v4")
    if not args.cuda or not args.compile:
        raise ValueError("v14 requires CUDA and compiled learner execution")
    if args.track:
        raise ValueError("v14 matched runs use local TensorBoard, not WandB")
    if args.critic_mode not in {"scalar", "successor"}:
        raise ValueError("invalid critic_mode")
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs, args.env_threads) <= 0:
        raise ValueError("environment, rollout, minibatch, epoch and thread counts must be positive")
    if not (0 < args.gamma < 1 and 0 <= args.gae_lambda <= 1):
        raise ValueError("gamma must be in (0,1), lambda in [0,1]")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not (0 < args.clip_coef < 1) or not np.isfinite(args.vf_coef) or args.vf_coef <= 0:
        raise ValueError("invalid clipping or critic coefficient")
    if not np.isfinite(args.ent_coef) or args.ent_coef < 0:
        raise ValueError("ent_coef must be finite and nonnegative")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches:
        raise ValueError("num_minibatches must divide batch_size exactly")
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size < 2 or args.total_timesteps < args.batch_size:
        raise ValueError("need at least two examples per minibatch and a complete rollout")
    return args


def make_training_env(args, run_name):
    return make_mujoco_vector_env(args.env_id, args.num_envs, backend=args.env_backend,
                                  num_threads=min(args.env_threads, args.num_envs),
                                  capture_video=False, run_name=run_name)


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
    with ExitStack() as resources:
        writer = SummaryWriter(f"runs/{run_name}")
        resources.callback(writer.close)
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("successor_contract", "Exact [reward_run,reward_ctrl] observed factors; "
                        "same per-transition VectorRewardNorm scale/clipping; direct V=sum(Psi); "
                        "PPO normalized-return units without occupancy rescaling; fixed readout; "
                        "two separate Tanh64x64 FFNs; one PPO/value update schedule; "
                        "actor/critic exclusive Adam owners and separate clipping; "
                        "nullspace RMS is a detached rollout loss metric, never a bootstrap coordinate; "
                        "normalization state is hidden/nonstationary in BOTH arms; no JEPA; "
                        "SF adds65 head parameters; own-TD EV is not heldout accuracy")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        parameters = agent.parameter_groups()
        optimizers = tuple(optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in parameters)
        counts = agent.parameter_counts()
        writer.add_text("parameter_counts", str(counts))
        print(f"parameter_counts={counts}")
        for name, count in counts.items():
            writer.add_scalar(f"parameters/{name}", count, 0)

        def statistics_model(observations, native):
            return rollout_statistics(agent, observations, native)

        def objective_model(observations, native, old_logprobs, advantages, targets, old_values, scale):
            return policy_loss(agent, observations, native, old_logprobs, advantages,
                               targets, old_values, args, scale)

        statistics_model = graph_compile(statistics_model)
        successor_model = graph_compile(agent.get_successor)
        objective_model = torch.compile(objective_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=True, mode=args.compile_mode, explicit_next_values=True)
        obs_shape = envs.single_observation_space.shape
        host_actor = HostPolicy(agent, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        fields = {"observations": obs_shape, "native_actions": (agent.action_dim,),
                  "next_observations": obs_shape, "reward_factors": (2,)}
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers, fields=fields)
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        updates_per_rollout = args.update_epochs * args.num_minibatches
        gradient_norms = torch.empty((updates_per_rollout, 2), device=device)
        metric_names = ("losses/policy_loss", "losses/value_loss", "successor/nullspace_loss",
                        "losses/entropy", "losses/old_approx_kl", "losses/approx_kl", "losses/clipfrac")
        sums = {name: torch.zeros((), device=device) for name in metric_names}
        total_updates = 0
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=warmup_action, horizon=horizon, phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                for optimizer in optimizers:
                    optimizer.param_groups[0]["lr"] = (1 - (iteration - 1) / args.num_iterations) * args.learning_rate
            raw_reconstruction_max = normalized_reconstruction_max = 0.0
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    raw_factors = observed_reward_factors(raw_reward, terms, truncs, infos)
                    factors = normalize_reward_factors(raw_factors, raw_reward, reward, rew_norm)
                    raw_reconstruction_max = max(raw_reconstruction_max, float(np.abs(raw_factors.sum(-1) - raw_reward).max()))
                    normalized_reconstruction_max = max(normalized_reconstruction_max,
                                                        float(np.abs(factors.sum(-1) - reward).max()))
                    next_obs_np, factual_next_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    # Every next row is factual, including terminal/time-limit finals.
                    # Explicit-next GAE needs no separate tail/reset bootstrap cache.
                    transfer.push(step, reward, terms, truncs, observations=obs_step,
                                  native_actions=native, next_observations=factual_next_obs, reward_factors=factors)
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
                predictions, logprobs = statistics_model(b_obs, b_native)
                b_predictions, b_logprobs = predictions.clone(), logprobs.clone()
                b_values = b_predictions.sum(-1)
                next_predictions = successor_model(batch.fields["next_observations"].flatten(0, 1)).clone()
                features = batch.fields["reward_factors"] if agent.successor_dim == 2 else batch.rewards.unsqueeze(-1)
                targets = successor_lambda_targets(
                    features, b_predictions.view(args.num_steps, args.num_envs, agent.successor_dim),
                    next_predictions.view(args.num_steps, args.num_envs, agent.successor_dim),
                    batch.terminations, batch.truncations, args.gamma, args.gae_lambda, gae_fn,
                )
                b_targets = targets.flatten(0, 1)
                b_returns = b_targets.sum(-1)
                b_advantages = (b_returns - b_values).detach().clone()
                loss_scale = successor_loss_scale(b_targets)
                transfer_reconstruction_max = (batch.fields["reward_factors"].sum(-1) - batch.rewards).abs().max()
            for accumulator in sums.values():
                accumulator.zero_()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        torch.compiler.cudagraph_mark_step_begin()
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        objective, metrics = objective_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices], b_advantages[indices],
                            b_targets[indices], b_values[indices], loss_scale,
                        )
                        objective.backward()
                        optimizer_step(optimizers, parameters, args.max_grad_norm, gradient_norms[updates])
                        for name, metric in metrics.items():
                            sums[name].add_(metric.detach())
                        updates += 1
                host_actor.refresh()
            total_updates += updates
            metric_values = {name: value / updates for name, value in sums.items()}
            metric_values.update({
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "gradients/actor_norm": gradient_norms[:, 0].mean(),
                "gradients/critic_norm": gradient_norms[:, 1].mean(),
                "successor/nullspace_loss_scale": loss_scale,
                "reward/transfer_reconstruction_max": transfer_reconstruction_max,
            })
            logged = gather_metrics(metric_values)
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            writer.add_scalar("reward/raw_reconstruction_max", raw_reconstruction_max, global_step)
            writer.add_scalar("reward/normalized_reconstruction_max", normalized_reconstruction_max, global_step)
            for name, value in {
                "ppo_steps": updates, "critic_steps": updates, "ppo_steps_total": total_updates,
                "critic_steps_total": total_updates, "ppo_examples": updates * args.minibatch_size,
                "critic_examples": updates * args.minibatch_size,
                "ppo_examples_total": total_updates * args.minibatch_size,
                "critic_examples_total": total_updates * args.minibatch_size,
                "ppo_minibatch_size": args.minibatch_size, "epochs": args.update_epochs,
            }.items():
                writer.add_scalar(f"updates/{name}", value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizers[0].param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (now - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save({
                "model": agent.state_dict(), "args": vars(args), "global_step": global_step,
                "parameter_counts": counts, "factor_names": FACTOR_NAMES,
                "reward_readout": "sum", "successor_units": "normalized_discounted_return",
                "loss_scale": loss_scale.detach().cpu(), "optimizer_steps": total_updates,
                "optimizers": {"actor": optimizers[0].state_dict(), "critic": optimizers[1].state_dict()},
                "reward_norm": {
                    "returns": torch.from_numpy(rew_norm.returns.copy()),
                    "means": torch.from_numpy(rew_norm.means.copy()),
                    "variances": torch.from_numpy(rew_norm.variances.copy()),
                    "counts": torch.from_numpy(rew_norm.counts.copy()),
                    "gamma": rew_norm.gamma, "epsilon": rew_norm.epsilon, "clip": rew_norm.clip,
                },
                "obs_norm": {
                    "means": torch.from_numpy(obs_norm.means.copy()),
                    "variances": torch.from_numpy(obs_norm.variances.copy()),
                    "counts": torch.from_numpy(obs_norm.counts.copy()),
                    "epsilon": obs_norm.epsilon, "clip": obs_norm.clip,
                },
            }, model_path)
            print(f"model saved to {model_path}")


if __name__ == "__main__":
    main()

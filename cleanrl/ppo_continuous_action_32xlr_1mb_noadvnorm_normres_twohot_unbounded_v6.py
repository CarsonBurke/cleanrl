# Raw symexp two-hot SiTU pre-RMS PPO, v6: no gradient clipping or final RMS.
# Exact v4 d3range reference except actor/critic final norms become identities
# and gradient norms are observed without scaling. Keep all branch-input norms,
# the 1/sqrt(width) readout multiplier, zero categorical head and raw targets.
# Hypothesis: allow stream magnitude to reach both heads without clip coupling.
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
from cleanrl.shared.two_hot import DreamerTwoHotSupport
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
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    value_loss: Literal["twohot", "mse"] = "twohot"
    """two-hot cross-entropy or unclipped scalar half-MSE; neither uses value clipping"""
    reward_norm: bool = False
    """normalize/clip rewards before GAE; never normalize GAE return targets"""
    value_num_bins: int = 255
    """Dreamer categorical atoms; ignored by scalar MSE controls"""
    value_max_abs: float = 485165194.4097903
    """symmetric support endpoint in raw critic-target units"""
    value_spacing: Literal["symexp", "linear"] = "symexp"
    """Dreamer symexp placement or uniform raw centers; labels/decoding stay raw"""

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
    value_support: DreamerTwoHotSupport | None

    def __init__(
        self,
        envs,
        *,
        placement="pre",
        norm_kind="rms",
        activation="stiglu",
        value_loss="twohot",
        value_num_bins=255,
        value_max_abs=485165194.4097903,
        value_spacing="symexp",
    ):
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
        self.actor[0].final_norm = nn.Identity()
        self.critic[0].final_norm = nn.Identity()
        # Construct the base scalar critic and actor before replacing the head:
        # neither bin count nor support may change the behavior policy's RNG.
        self.value_support = None
        if value_loss == "twohot":
            self.value_support = DreamerTwoHotSupport(value_num_bins, value_max_abs, spacing=value_spacing)
            head = nn.Linear(64, value_num_bins)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.critic[-1] = head
        elif value_loss != "mse":
            raise ValueError("unknown critic objective")

    def get_value(self, x):
        output = self.critic(x)
        if self.value_support is not None:
            return self.value_support.to_scalar(output).unsqueeze(-1)
        return output

    def get_policy_and_value(self, x):
        alpha, beta, output = self.get_policy_and_value_output(x)
        if self.value_support is not None:
            output = self.value_support.to_scalar(output).unsqueeze(-1)
        return alpha, beta, output

    def get_policy_and_value_output(self, x):
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


def gradient_norms(actor_parameters, critic_parameters):
    """Observe weighted gradient magnitudes without mutating any gradient."""
    return tuple(
        nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None], foreach=True)
        for parameters in (actor_parameters, critic_parameters)
    )


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, targets, args):
    """Unchanged PPO; targets are fixed two-hot probabilities or scalar returns."""
    alpha, beta, output = agent.get_policy_and_value_output(observations)
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
    if args.value_loss == "twohot":
        v_loss = -(targets.detach() * output.float().log_softmax(dim=-1)).sum(dim=-1).mean()
    else:
        newvalue = output.view(-1)
        v_loss = 0.5 * ((newvalue - targets) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack(
        (pg_loss.detach(), v_loss.detach(), entropy_loss.detach(), old_approx_kl, approx_kl, clipfrac)
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
            f"{args.activation} residual trunk; no final RMS or gradient clipping; "
            f"LR={args.learning_rate}; epochs={args.update_epochs}; 1 minibatch; raw GAE",
        )
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(
            envs,
            placement=args.placement,
            norm_kind=args.norm_kind,
            activation=args.activation,
            value_loss=args.value_loss,
            value_num_bins=args.value_num_bins,
            value_max_abs=args.value_max_abs,
            value_spacing=args.value_spacing,
        ).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        actor_parameters = tuple(agent.actor.parameters())
        critic_parameters = tuple(agent.critic.parameters())
        value_model = agent.get_value
        target_model = agent.value_support.project if agent.value_support is not None else None

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, targets):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, targets, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            if target_model is not None:
                target_model = graph_compile(target_model)
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
        update_metrics = torch.empty((max_updates, 6), device=device)
        update_grad_norms = torch.empty((max_updates, 2), device=device)
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
                b_targets = target_model(b_returns) if target_model is not None else b_returns
                value_diagnostics = {
                    "critic/return_mean": b_returns.mean(),
                    "critic/return_std": b_returns.std(unbiased=False),
                    "critic/return_min": b_returns.min(),
                    "critic/return_max": b_returns.max(),
                }
                if agent.value_support is not None:
                    projection_error = agent.value_support.probs_to_scalar(b_targets) - b_returns
                    support = agent.value_support.support
                    value_diagnostics.update(
                        {
                            "critic/support_overflow_fraction": ((b_returns < support[0]) | (b_returns > support[-1]))
                            .float()
                            .mean(),
                            "critic/projection_target_bias": projection_error.mean(),
                            "critic/projection_target_rmse": projection_error.square().mean().sqrt(),
                        }
                    )
            updates = 0
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
                            b_targets[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        actor_norm, critic_norm = gradient_norms(
                            actor_parameters, critic_parameters
                        )
                        update_grad_norms[updates, 0].copy_(actor_norm)
                        update_grad_norms[updates, 1].copy_(critic_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            with torch.no_grad():
                post_error = value_model(b_obs).flatten() - b_returns
                value_diagnostics["critic/postupdate_value_bias"] = post_error.mean()
                value_diagnostics["critic/postupdate_value_rmse"] = post_error.square().mean().sqrt()
            last = update_metrics[updates - 1]
            performed_grad_norms = update_grad_norms[:updates]
            mean_grad_norms = performed_grad_norms.mean(dim=0)
            logged = gather_metrics(
                {
                    "losses/policy_loss": last[0],
                    "losses/value_loss": last[1],
                    "losses/entropy": last[2],
                    "losses/old_approx_kl": last[3],
                    "losses/approx_kl": last[4],
                    "losses/clipfrac": update_metrics[:updates, 5].mean(),
                    "losses/explained_variance": explained_variance(b_values, b_returns),
                    "grad/actor_norm": mean_grad_norms[0],
                    "grad/critic_norm": mean_grad_norms[1],
                    **value_diagnostics,
                }
            )
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/optimizer_steps", updates, global_step)
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

# Peri PPO v21: isolate sensitive actor-readout updates from trunk learning.
# Inspired by modded-nanogpt's parameter-specific rates, not a KL controller.
# Preserve Peri/full-GAE/Adam; only the actor head gets a separate LR multiplier.
# Log analytic old-to-new Beta KL AFTER the step, not the near-zero pre-step KL.
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
from cleanrl.shared.trl_projection import beta_kl_reverse

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
    actor_head_lr_scale: float = 0.1
    """actor final linear LR multiplier; 1.0 is the matched control"""
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
    update_epochs: int = 1
    """one joint actor/critic update per rollout in this component ablation"""
    critic_trace_steps: int = 0
    """critic-only GAE trace boundaries; zero preserves the full rollout trace"""
    critic_minibatches: int = 1
    """one-pass critic minibatches per rollout; the first shares the actor step"""
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
    grad_clip: bool = False
    """apply independent gradient clipping; disabling preserves every raw gradient"""
    clip_heads: bool = True
    """include final actor/critic linear heads in their independent clipping budgets"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    reward_norm: bool = True
    """normalize/clip rewards before GAE; never normalize GAE return targets"""

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

    def __init__(self, envs):
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
                observation_dim, 64, placement="peri", norm_kind="rms", activation="stiglu"
            ),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            make_norm_residual_trunk(
                observation_dim, 64, placement="peri", norm_kind="rms", activation="stiglu"
            ),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        self.actor[0].final_norm = nn.Identity()
        self.critic[0].final_norm = nn.Identity()

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


def clipping_parameters(agent, clip_heads):
    """Select each independent budget once; excluded heads still belong to Adam."""
    actor = agent.actor if clip_heads else agent.actor[0]
    critic = agent.critic if clip_heads else agent.critic[0]
    return tuple(actor.parameters()), tuple(critic.parameters())


def clip_gradients(actor_parameters, critic_parameters, max_grad_norm, enabled=True):
    """Measure weighted gradients, optionally clipping each network independently."""
    if not enabled:
        return tuple(
            nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None], foreach=True)
            for parameters in (actor_parameters, critic_parameters)
        )
    actor_preclip_norm = nn.utils.clip_grad_norm_(actor_parameters, max_grad_norm, foreach=True)
    critic_preclip_norm = nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm, foreach=True)
    return actor_preclip_norm, critic_preclip_norm


def critic_trace_boundaries(truncations: torch.Tensor, trace_steps: int) -> torch.Tensor:
    """Cut only the critic's continuation trace, retaining every real boundary.

    Artificial cuts retain online next-state bootstrapping: they are not terminal
    transitions and never alter rewards, actor GAE or environment normalization.
    """
    if trace_steps < 0:
        raise ValueError("critic trace steps must be nonnegative")
    if trace_steps == 0:
        return truncations
    boundaries = truncations.bool()
    timesteps = torch.arange(1, truncations.shape[0] + 1, device=truncations.device)
    cuts = (timesteps % trace_steps == 0).reshape((-1,) + (1,) * (truncations.ndim - 1))
    # The shared bounded-code CUDA recurrence requires float32 masks.
    return (boundaries | cuts).to(dtype=truncations.dtype)


def scalar_value_loss(prediction, targets, old_values, clip_coef, clip_vloss=True):
    """The base PPO scalar MSE, with detached targets and pre-update clip anchors."""
    prediction = prediction.reshape(-1)
    targets = targets.detach().reshape(-1)
    old_values = old_values.detach().reshape(-1)
    error = (prediction - targets) ** 2
    if clip_vloss:
        clipped = old_values + torch.clamp(prediction - old_values, -clip_coef, clip_coef)
        error = torch.max(error, (clipped - targets) ** 2)
    return 0.5 * error.mean()


def critic_statistics(agent, observations, next_observations):
    """Fixed-rollout-shape online V(s), V(s') tuple, evaluated under no_grad by callers."""
    return agent.get_value(observations).flatten(), agent.get_value(next_observations).flatten()


def critic_step(agent, optimizer, observations, targets, old_values, args, loss_model=None):
    """Regress fresh targets without advancing actor parameters or their Adam state.

    The caller owns online target regeneration. A compiled loss callable receives
    (observations, targets, old_values) and returns the vf_coef-weighted loss.
    """
    optimizer.zero_grad(set_to_none=True)
    if loss_model is None:
        loss = args.vf_coef * scalar_value_loss(
            agent.get_value(observations), targets, old_values, args.clip_coef, args.clip_vloss
        )
    else:
        loss = loss_model(observations, targets, old_values)
    loss.backward()
    critic = agent.critic if args.clip_heads else agent.critic[0]
    parameters = tuple(critic.parameters())
    if args.grad_clip:
        grad_norm = nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, foreach=True)
    else:
        grad_norm = nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None], foreach=True)
    optimizer.step()
    return grad_norm


def compute_rollout_targets(
    gae_fn, rewards, values, terms, truncs, next_values, gamma, gae_lambda, trace_steps
):
    """Own graph outputs before another GAE call can reuse their backing storage.

    Actor GAE always follows the real episode boundaries; artificial cuts apply
    only to critic targets. Outputs retain the input (time, environment) shape.
    """
    advantages, returns = gae_fn(rewards, values, terms, truncs, next_values, gamma, gae_lambda)
    actor_advantages = advantages.detach().clone()
    long_returns = returns.detach().clone()
    if trace_steps == 0:
        return actor_advantages, long_returns, long_returns
    _, targets = gae_fn(
        rewards, values, terms, critic_trace_boundaries(truncs, trace_steps), next_values, gamma, gae_lambda
    )
    return actor_advantages, long_returns, targets.detach().clone()


def ppo_loss(
    agent, observations, native_actions, old_logprobs, advantages, targets, old_values, args, critic_indices=None
):
    """Clipped policy surrogate and optionally clipped scalar MSE on native Beta samples."""
    alpha, beta, newvalue = agent.get_policy_and_value(observations)
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
    newvalue = newvalue.flatten()
    if critic_indices is not None:
        newvalue = newvalue[critic_indices]
        targets = targets[critic_indices]
        old_values = old_values[critic_indices]
    v_loss = scalar_value_loss(newvalue, targets, old_values, args.clip_coef, args.clip_vloss)
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack(
        (pg_loss.detach(), v_loss.detach(), entropy_loss.detach(), old_approx_kl, approx_kl, clipfrac)
    )
    return loss, metrics


def validate_args(args):
    if not np.isfinite(args.actor_head_lr_scale) or args.actor_head_lr_scale <= 0:
        raise ValueError("actor_head_lr_scale must be finite and positive")
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.update_epochs != 1 or args.num_minibatches != 1:
        raise ValueError("critic-target ablations require update_epochs=1 and num_minibatches=1")
    if args.critic_trace_steps < 0 or args.critic_minibatches < 1:
        raise ValueError("critic_trace_steps must be nonnegative and critic_minibatches must be positive")
    args.batch_size = args.num_envs * args.num_steps
    if args.critic_minibatches > args.batch_size:
        raise ValueError("critic_minibatches cannot exceed batch_size")
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


def make_optimizer(agent, args):
    head = tuple(agent.actor[-1].parameters())
    head_ids = {id(parameter) for parameter in head}
    rest = [parameter for parameter in agent.parameters() if id(parameter) not in head_ids]
    return optim.Adam([
        {"params": rest, "lr": args.learning_rate, "lr_scale": 1.0},
        {"params": head, "lr": args.learning_rate * args.actor_head_lr_scale,
         "lr_scale": args.actor_head_lr_scale},
    ], eps=1e-5, fused=True)


def set_learning_rate(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * group["lr_scale"]


@torch.no_grad()
def post_update_statistics(agent, observations, old_alpha, old_beta, targets):
    alpha, beta, values = agent.get_policy_and_value(observations)
    # This helper returns KL(first || second); frozen behavior goes first.
    divergence = beta_kl_reverse(old_alpha, old_beta, alpha, beta).sum(-1)
    concentration = alpha + beta
    return torch.stack((divergence.mean(), divergence.max(), concentration.mean(),
                        concentration.max(), (values.flatten() - targets).square().mean().sqrt()))


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
            f"Beta fused host actor; fixed Peri-RMS SiTU-GLU, no final norms; LR={args.learning_rate}; "
            f"epochs={args.update_epochs}; grad_clip={args.grad_clip}; "
            f"clip_heads={args.clip_heads}; full policy GAE; actor_head_lr_scale={args.actor_head_lr_scale}; "
            f"critic_trace_steps={args.critic_trace_steps}; critic_minibatches={args.critic_minibatches}",
        )
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        optimizer = make_optimizer(agent, args)
        actor_parameters, critic_parameters = clipping_parameters(agent, args.clip_heads)
        clip_scope = "" if args.clip_heads else "_trunk"
        def rollout_statistics(observations, native, next_observations):
            """Frozen behavior statistics and online transition values for the full rollout."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            next_value = agent.get_value(next_observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native), next_value.flatten(), alpha, beta

        def loss_model(observations, native, old_logprobs, advantages, targets, old_values, critic_indices):
            return ppo_loss(
                agent, observations, native, old_logprobs, advantages, targets, old_values, args, critic_indices
            )

        def critic_statistics_model(observations, next_observations):
            return critic_statistics(agent, observations, next_observations)

        def critic_loss_model(observations, targets, old_values):
            return args.vf_coef * scalar_value_loss(
                agent.get_value(observations), targets, old_values, args.clip_coef, args.clip_vloss
            )

        def post_update_model(observations, old_alpha, old_beta, targets):
            return post_update_statistics(agent, observations, old_alpha, old_beta, targets)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            post_update_model = graph_compile(post_update_model)
            critic_statistics_model = graph_compile(critic_statistics_model)
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            critic_loss_model = torch.compile(
                critic_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False
            )
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode, explicit_next_values=True)
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
            store_transition_observations=True,
            fields={"observations": obs_shape, "native_actions": (agent.action_dim,)},
        )
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma) if args.reward_norm else None
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        update_metrics = torch.empty(6, device=device)
        critic_grad_norms = torch.empty(args.critic_minibatches, device=device)
        critic_update_count = torch.tensor(args.critic_minibatches, device=device)
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
                set_learning_rate(optimizer, (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate)
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
                    transfer.push(
                        step, reward, terms, truncs, transition_obs, observations=obs_step, native_actions=native
                    )
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
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                assert batch.transition_observations is not None
                b_next_obs = batch.transition_observations.flatten(0, 1)
                b_values, b_logprobs, b_next_values, old_alpha, old_beta = rollout_statistics(
                    b_obs, b_native, b_next_obs)
                # These behavior statistics remain fixed through all critic substeps.
                b_values = b_values.clone()
                b_logprobs = b_logprobs.clone()
                old_alpha, old_beta = old_alpha.clone(), old_beta.clone()
                values = b_values.view(args.num_steps, args.num_envs)
                next_values = b_next_values.view(args.num_steps, args.num_envs)
                advantages, returns, targets = compute_rollout_targets(
                    gae_fn, batch.rewards, values, batch.terminations, batch.truncations,
                    next_values, args.gamma, args.gae_lambda, args.critic_trace_steps,
                )
                b_advantages = advantages.flatten()
                b_returns = returns.flatten()
                b_targets = targets.flatten()
                critic_boundaries = critic_trace_boundaries(batch.truncations, args.critic_trace_steps)

                # Evaluate the NEW rollout before learning, in normalized reward/return
                # units. Short traces make raw target RMSE easier: report scale and
                # full-policy-trace comparisons rather than comparing raw RMSE alone.
                target_rmse = (b_values - b_targets).square().mean().sqrt()
                target_std = b_targets.std(unbiased=False)
                long_rmse = (b_values - b_returns).square().mean().sqrt()
                long_std = b_returns.std(unbiased=False)
                long_explained_variance = explained_variance(b_values, b_returns)
                td_error = batch.rewards + args.gamma * next_values * (1.0 - batch.terminations) - values
                critic_diagnostics = {
                    "critic/target_rmse": target_rmse,
                    "critic/target_relative_rmse": target_rmse / target_std.clamp_min(1e-8),
                    "critic/target_std": target_std,
                    "critic/long_gae_rmse": long_rmse,
                    "critic/long_gae_relative_rmse": long_rmse / long_std.clamp_min(1e-8),
                    "critic/long_gae_explained_variance": long_explained_variance,
                    "critic/td_rmse": td_error.square().mean().sqrt(),
                    "critic/target_return_min": b_targets.min(),
                    "critic/target_return_max": b_targets.max(),
                    "critic/updates": critic_update_count,
                    "critic/minibatches": critic_update_count,
                }

            with timer.span("update"):
                # Preserve the base actor's one shuffled full-batch update and RNG
                # consumption. Critic chunks partition that SAME permutation once.
                indices = device_minibatches(
                    args.batch_size, args.minibatch_size, device, shuffle_generator
                )[0]
                critic_chunks = torch.tensor_split(indices, args.critic_minibatches)
                first_critic_indices = None if args.critic_minibatches == 1 else slice(0, critic_chunks[0].numel())
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                loss, metrics = loss_model(
                    b_obs[indices], b_native[indices], b_logprobs[indices],
                    b_advantages[indices], b_targets[indices], b_values[indices], first_critic_indices,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_preclip_norm, critic_preclip_norm = clip_gradients(
                    actor_parameters, critic_parameters, args.max_grad_norm, enabled=args.grad_clip
                )
                critic_grad_norms[0].copy_(critic_preclip_norm)
                optimizer.step()
                update_metrics.copy_(metrics)

                for substep, critic_indices in enumerate(critic_chunks[1:], start=1):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    # Rebootstrap BOTH values from the updated ONLINE critic over
                    # the frozen rollout; train only this not-yet-consumed chunk.
                    with torch.no_grad():
                        current_values, current_next_values = critic_statistics_model(b_obs, b_next_obs)
                        current_values = current_values.clone()
                        _, current_targets = gae_fn(
                            batch.rewards, current_values.view(args.num_steps, args.num_envs),
                            batch.terminations, critic_boundaries,
                            current_next_values.view(args.num_steps, args.num_envs),
                            args.gamma, args.gae_lambda,
                        )
                        current_targets = current_targets.detach().flatten().clone()
                    critic_grad_norms[substep].copy_(critic_step(
                        agent, optimizer, b_obs[critic_indices], current_targets[critic_indices],
                        current_values[critic_indices], args, loss_model=critic_loss_model,
                    ))

            with timer.span("metrics"):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                post_metrics = post_update_model(b_obs, old_alpha, old_beta, b_targets)
                # Actor statistics are from its ONLY step, not diluted by critic-only
                # steps. Critic gradient summaries cover every one-pass minibatch.
                actor_clip_fraction = ((actor_preclip_norm > args.max_grad_norm).float()
                                       if args.grad_clip else torch.zeros_like(actor_preclip_norm))
                critic_clip_fraction = ((critic_grad_norms > args.max_grad_norm).float().mean()
                                        if args.grad_clip else torch.zeros_like(actor_preclip_norm))
                logged = gather_metrics(
                    {
                        "losses/policy_loss": update_metrics[0],
                        "losses/value_loss": update_metrics[1],
                        "losses/entropy": update_metrics[2],
                        "losses/old_approx_kl": update_metrics[3],
                        "losses/approx_kl": update_metrics[4],
                        "losses/clipfrac": update_metrics[5],
                        "losses/explained_variance": long_explained_variance,
                        f"grad/actor{clip_scope}_preclip_norm": actor_preclip_norm,
                        f"grad/critic{clip_scope}_preclip_norm": critic_grad_norms.mean(),
                        f"grad/actor{clip_scope}_clip_fraction": actor_clip_fraction,
                        f"grad/critic{clip_scope}_clip_fraction": critic_clip_fraction,
                        **critic_diagnostics,
                        "policy/post_update_kl_mean": post_metrics[0],
                        "policy/post_update_kl_max": post_metrics[1],
                        "policy/concentration_mean": post_metrics[2],
                        "policy/concentration_max": post_metrics[3],
                        "critic/post_update_target_rmse": post_metrics[4],
                    }
                )
            nan_variance_metrics = {"losses/explained_variance", "critic/long_gae_explained_variance"}
            if any(not np.isfinite(value) for name, value in logged.items() if name not in nan_variance_metrics):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            timings = timer.summary()
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/actor_head_learning_rate", optimizer.param_groups[1]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar(
                "charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step
            )
            for phase, timing in timings.items():
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

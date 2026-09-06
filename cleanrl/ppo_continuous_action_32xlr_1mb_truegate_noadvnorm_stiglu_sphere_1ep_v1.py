# 32x LR + 1 minibatch + truegate with no advantage normalization.
# Same gate as _truegate_v1 (v9/v10 Alg. 2). v9/v10 default is norm_adv=False:
# gate-weighted raw GAE goes into PPO with no minibatch standardization.
# Truegate + hypersphered SiTU-GLU trunk with ONE actor epoch per rollout.
# Same 32x LR (9.6e-3) and nGPT-style sphere trunk as
# _truegate_noadvnorm_stiglu_sphere_v1, but actor and critic train on
# separate optimizers and schedules: the policy takes a single full-batch
# PPO pass (minimal trust-region movement on top of truegate's delight
# weighting) while the critic keeps 10 epochs of value fitting. Hypothesis:
# one delight-weighted policy step already captures the usable update, and
# extra actor epochs only add KL without signal.
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
from cleanrl.shared.host_actor import HostSiTUSphereActor, make_situ_sphere_trunk
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions, sample_beta_actions_host
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
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 9.6e-3
    """32x the 3e-4 PPO default; one Adam covers actor and critic"""
    num_envs: int = 1
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
    actor_epochs: int = 1
    """the K epochs to update the actor policy per rollout"""
    critic_epochs: int = 10
    """the K epochs to update the critic value function per rollout"""
    norm_adv: bool = False
    """truegate v9/v10: gated raw GAE, no minibatch standardization"""
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
    dg_eta: float = 1.0
    """truegate temperature eta in w = sigmoid(U*l/eta)"""
    dg_surprisal_clip: float = 10.0
    """clip bound C on surprisal l = -log-density"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 4
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


# Public evaluation and historical GAE compatibility helpers; not the training path.
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


def bootstrap_observations(next_obs, truncations, infos):
    """Replace autoreset observations with final observations at time limits."""
    bootstrap_obs = np.array(next_obs, copy=True)
    truncations = np.asarray(truncations, dtype=bool)
    if not np.any(truncations):
        return bootstrap_obs

    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None:
        raise RuntimeError("truncated transition missing infos['final_observation']")

    for env_idx in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_idx]:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        final_observation = final_observations[env_idx]
        if final_observation is None:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        bootstrap_obs[env_idx] = final_observation
    return bootstrap_obs


def compute_gae(
    rewards,
    values,
    terminations,
    truncations,
    truncation_bootstrap_values,
    rollout_tail_value,
    gamma,
    gae_lambda,
):
    """Compute GAE with distinct bootstrap and reset-boundary semantics."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rollout_tail_value)
    for t in reversed(range(rewards.shape[0])):
        ordinary_next_value = rollout_tail_value if t == rewards.shape[0] - 1 else values[t + 1]
        next_value = torch.where(
            truncations[t].bool(),
            truncation_bootstrap_values[t],
            ordinary_next_value,
        )
        bootstrap_nonterminal = 1.0 - terminations[t]
        trace_nonterminal = 1.0 - torch.maximum(terminations[t], truncations[t])
        delta = rewards[t] + gamma * bootstrap_nonterminal * next_value - values[t]
        last_advantage = delta + gamma * gae_lambda * trace_nonterminal * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.critic = nn.Sequential(
            make_situ_sphere_trunk(observation_dim, 64),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            make_situ_sphere_trunk(observation_dim, 64),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_policy(self, x):
        return (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)

    def get_policy_and_value(self, x):
        alpha, beta = self.get_policy(x)
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


def truegate_weights(advantages, physical_logprobs, eta, surprisal_clip):
    """Paper Alg. 2 on physical log-density; Jacobian already in `physical_logprobs`."""
    surprisal = (-physical_logprobs).clamp(-surprisal_clip, surprisal_clip)
    return torch.sigmoid(advantages * surprisal / eta)


def actor_loss(agent, observations, native_actions, old_logprobs, advantages, args):
    """Clipped PPO surrogate on native Beta samples; policy parameters only."""
    alpha, beta = agent.get_policy(observations)
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
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss
    metrics = torch.stack((pg_loss.detach(), entropy_loss.detach(), old_approx_kl, approx_kl, clipfrac))
    return loss, metrics


def critic_loss(agent, observations, returns, old_values, args):
    """Clipped value loss (vf_coef applied); critic parameters only."""
    newvalue = agent.get_value(observations).view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    return v_loss * args.vf_coef


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.actor_epochs, args.critic_epochs) <= 0:
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
    if args.dg_eta <= 0 or args.dg_surprisal_clip <= 0:
        raise ValueError("truegate eta and surprisal clip must be positive")
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
        writer.add_text("policy", "Beta host actor; truegate; hypersphered SiTU-GLU trunk; 1 actor epoch + 10 critic epochs; no advantage normalization")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        actor_opt = optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        critic_opt = optim.Adam(agent.critic.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def actor_loss_model(observations, native, old_logprobs, advantages):
            return actor_loss(agent, observations, native, old_logprobs, advantages, args)

        def critic_loss_model(observations, returns, old_values):
            return critic_loss(agent, observations, returns, old_values, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            actor_loss_model = torch.compile(actor_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            critic_loss_model = torch.compile(critic_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = HostSiTUSphereActor(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)

        def act(observations):
            native, action = sample_beta_actions_host(host_actor(observations), action_low, action_high, sampler)
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
        n_batches = (args.batch_size + args.minibatch_size - 1) // args.minibatch_size
        actor_metrics = torch.empty((args.actor_epochs * n_batches, 5), device=device)
        critic_vloss = torch.empty((args.critic_epochs * n_batches,), device=device)
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
                lr_now = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
                actor_opt.param_groups[0]["lr"] = lr_now
                critic_opt.param_groups[0]["lr"] = lr_now
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
                dg_w = truegate_weights(b_advantages, b_logprobs, args.dg_eta, args.dg_surprisal_clip)
                dg_pos = (dg_w * (b_advantages > 0).to(dg_w.dtype)).sum() / dg_w.sum().clamp_min(1e-12)
                b_advantages = b_advantages * dg_w
            au, cu = 0, 0
            with timer.span("update"):
                for _ in range(args.actor_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = actor_loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices],
                        )
                        actor_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                        actor_opt.step()
                        actor_metrics[au].copy_(metrics)
                        au += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and actor_metrics[au - 1, 3] > args.target_kl:
                        break
                for _ in range(args.critic_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        v_loss = critic_loss_model(
                            b_obs[indices], b_returns[indices], b_values[indices],
                        )
                        critic_opt.zero_grad(set_to_none=True)
                        v_loss.backward()
                        nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                        critic_opt.step()
                        critic_vloss[cu].copy_(v_loss.detach())
                        cu += 1

            last = actor_metrics[au - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": critic_vloss[cu - 1],
                "losses/entropy": last[1], "losses/old_approx_kl": last[2],
                "losses/approx_kl": last[3], "losses/clipfrac": actor_metrics[:au, 4].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "debug/dg_w_mean": dg_w.mean(), "debug/dg_w_std": dg_w.std(unbiased=False),
                "debug/dg_pos_mass_frac": dg_pos,
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", actor_opt.param_groups[0]["lr"], global_step)
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

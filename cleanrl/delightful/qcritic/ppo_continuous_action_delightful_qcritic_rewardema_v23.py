# Delightful Policy Gradient with stable paired comparison v23.
#
# This is the successful v18 actor geometry and paired same-state comparison,
# with one isolated stability fix: state-dependent sigma is smoothly bounded
# in [0.1, 1.0] instead of collapsing near zero. Each taken action is compared
# against one independent current-policy action at the same state, giving a
# symmetric within-state preference signal without v16's eight Q samples.
# The tanh-squashed score and batch-256 DG cadence are otherwise unchanged.

import copy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch import nn, optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

MIN_POLICY_STD = 0.1
MAX_POLICY_STD = 1.0
INITIAL_POLICY_STD = float(np.exp(-1.5))
INITIAL_SCALE_FRACTION = (INITIAL_POLICY_STD - MIN_POLICY_STD) / (MAX_POLICY_STD - MIN_POLICY_STD)
RAW_SCALE_INIT = float(np.log(INITIAL_SCALE_FRACTION / (1.0 - INITIAL_SCALE_FRACTION)))


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False
    compile: bool = False
    compile_mode: str = "reduce-overhead"

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 16
    gamma: float = 0.99
    max_grad_norm: float = 0.5

    replay_capacity: int = 2_000_000
    critic_batch_size: int = 256
    critic_updates_per_iteration: int = 4
    learning_starts: int = 2_048
    target_update_interval: int = 100
    dg_eta: float = 1.0
    dg_surprisal_clip: float = 10.0
    reward_ema_decay: float = 0.999
    reward_norm_eps: float = 1e-8

    batch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    del gamma

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Replay requires a stationary observation representation. In contrast
        # to on-policy CleanRL PPO, do not store observations normalized under
        # different historical running statistics.
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def state_dependent_logstd(raw_scale):
    scale = MIN_POLICY_STD + (MAX_POLICY_STD - MIN_POLICY_STD) * torch.sigmoid(raw_scale)
    return torch.log(scale)


class EMARewardNormalizer:
    """Scale rewards by a bias-corrected EMA root second moment."""

    def __init__(self, decay=0.999, epsilon=1e-8):
        if not 0.0 <= decay < 1.0:
            raise ValueError("reward EMA decay must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("reward normalization epsilon must be positive")
        self.decay = decay
        self.epsilon = epsilon
        self.squared_reward_ema = 0.0
        self.num_updates = 0

    def update(self, rewards):
        batch_second_moment = float(np.mean(np.square(rewards, dtype=np.float64)))
        self.squared_reward_ema = self.decay * self.squared_reward_ema + (1.0 - self.decay) * batch_second_moment
        self.num_updates += 1
        correction = 1.0 - self.decay**self.num_updates
        return np.sqrt(self.squared_reward_ema / correction) + self.epsilon


def delightful_gate(advantages, action_logprobs, eta=1.0, surprisal_clip=10.0):
    if eta <= 0:
        raise ValueError("eta must be positive")
    if surprisal_clip <= 0:
        raise ValueError("surprisal_clip must be positive")
    surprisal = (-action_logprobs.detach()).clamp(-surprisal_clip, surprisal_clip)
    delight = advantages.detach() * surprisal
    return torch.sigmoid(delight / eta), surprisal, delight


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


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def one_step_q_target(rewards, discounts, next_q_values):
    return rewards + discounts * next_q_values


def validate_replay_config(replay_capacity, critic_batch_size, learning_starts):
    if replay_capacity < critic_batch_size:
        raise ValueError("replay_capacity must be at least critic_batch_size")
    if learning_starts < critic_batch_size:
        raise ValueError("learning_starts must be at least critic_batch_size")
    if learning_starts > replay_capacity:
        raise ValueError("learning_starts cannot exceed replay_capacity")


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape, seed):
        self.capacity = capacity
        self.observations = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observations, actions, rewards, next_observations, discounts):
        batch_size = len(observations)
        if not all(len(array) == batch_size for array in (actions, rewards, next_observations, discounts)):
            raise ValueError("replay fields must have equal batch length")
        # An oversized insertion has duplicate ring indices under NumPy
        # advanced assignment. Make the intended policy explicit: retain the
        # newest `capacity` transitions in their correct ring positions.
        skip = max(0, batch_size - self.capacity)
        write_position = (self.position + skip) % self.capacity
        indices = (np.arange(batch_size - skip) + write_position) % self.capacity
        self.observations[indices] = observations[skip:]
        self.actions[indices] = actions[skip:]
        self.rewards[indices] = rewards[skip:]
        self.next_observations[indices] = next_observations[skip:]
        self.discounts[indices] = discounts[skip:]
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size, device):
        if self.size < batch_size:
            raise ValueError(f"cannot sample {batch_size} transitions from replay of size {self.size}")
        indices = self.rng.integers(self.size, size=batch_size)
        fields = (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.discounts[indices],
        )
        return tuple(torch.as_tensor(field, dtype=torch.float32, device=device) for field in fields)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_low, action_high):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.raw_scale = layer_init(nn.Linear(256, action_dim), std=0.01, bias_const=RAW_SCALE_INIT)
        self.register_buffer("action_scale", torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def parameters_for(self, observations):
        features = self.trunk(observations)
        return self.mean(features), state_dependent_logstd(self.raw_scale(features))

    def transform(self, raw_actions):
        return torch.tanh(raw_actions) * self.action_scale + self.action_bias

    def logprobs(self, observations, raw_actions):
        mean, logstd = self.parameters_for(observations)
        gaussian_logprob = Normal(mean, logstd.exp()).log_prob(raw_actions).sum(dim=-1)
        log_tanh_jacobian = 2.0 * (np.log(2.0) - raw_actions - torch.nn.functional.softplus(-2.0 * raw_actions))
        action_logprob = gaussian_logprob - (log_tanh_jacobian + torch.log(self.action_scale)).sum(dim=-1)
        return gaussian_logprob, action_logprob, mean, logstd

    def sample(self, observations):
        mean, logstd = self.parameters_for(observations)
        raw_actions = Normal(mean, logstd.exp()).sample()
        actions = self.transform(raw_actions)
        return actions, raw_actions

    def deterministic(self, observations):
        mean, _ = self.parameters_for(observations)
        return self.transform(mean)


class QCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            # Near-zero Q initialization keeps the first frozen Bellman target
            # reward-dominated instead of bootstrapping arbitrary random scale.
            layer_init(nn.Linear(256, 1), std=0.01),
        )

    def forward(self, observations, actions):
        return self.network(torch.cat((observations, actions), dim=-1)).squeeze(-1)


@torch.no_grad()
def evaluate(actor, args, run_name, device, episodes=10):
    env = make_env(args.env_id, 0, False, run_name, args.gamma)()
    returns = []
    observation, _ = env.reset(seed=args.seed + 10_000)
    while len(returns) < episodes:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action = actor.deterministic(obs_tensor).squeeze(0).cpu().numpy()
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            returns.append(float(info["episode"]["r"]))
            observation, _ = env.reset()
    env.close()
    return returns


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.env_id != "HalfCheetah-v4":
        raise ValueError("v23 is tuned for HalfCheetah-v4")
    if args.batch_size != 256:
        raise ValueError(f"fresh actor batch must be 256, got {args.batch_size}")
    validate_replay_config(args.replay_capacity, args.critic_batch_size, args.learning_starts)
    if args.critic_updates_per_iteration <= 0 or args.target_update_interval <= 0:
        raise ValueError("critic update counts must be positive")
    if not 0.0 <= args.reward_ema_decay < 1.0:
        raise ValueError("reward_ema_decay must be in [0, 1)")
    if args.reward_norm_eps <= 0:
        raise ValueError("reward_norm_eps must be positive")

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
    rows = "\n".join(f"|{key}|{value}|" for key, value in vars(args).items())
    writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n{rows}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("this experiment requires CUDA")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, args.capture_video, run_name, args.gamma) for index in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    actor = Actor(
        obs_dim,
        action_dim,
        envs.single_action_space.low,
        envs.single_action_space.high,
    ).to(device)
    critic = QCritic(obs_dim, action_dim).to(device)
    target_actor = copy.deepcopy(actor).requires_grad_(False)
    target_critic = copy.deepcopy(critic).requires_grad_(False)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate, eps=1e-5)
    replay = ReplayBuffer(
        args.replay_capacity,
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        args.seed,
    )
    reward_normalizer = EMARewardNormalizer(args.reward_ema_decay, args.reward_norm_eps)

    sample_actor = actor.sample
    actor_logprobs = actor.logprobs
    critic_value = critic.forward
    target_actor_sample = target_actor.sample
    target_critic_value = target_critic.forward
    if args.compile:
        sample_actor = torch.compile(sample_actor, mode=args.compile_mode, dynamic=False)
        actor_logprobs = torch.compile(actor_logprobs, mode=args.compile_mode, dynamic=False)
        critic_value = torch.compile(critic_value, mode=args.compile_mode, dynamic=False)
        target_actor_sample = torch.compile(target_actor_sample, mode=args.compile_mode, dynamic=False)
        target_critic_value = torch.compile(target_critic_value, mode=args.compile_mode, dynamic=False)
        print(f"compiled actor and Q functions ({args.compile_mode})")

    observations = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    raw_actions = torch.zeros_like(actions)
    raw_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_observations_buffer = torch.zeros_like(observations)
    discounts = torch.zeros_like(raw_rewards)

    global_step = 0
    learner_step = 0
    start_time = time.time()
    reward_rms = 1.0
    previous_iteration_reward_rms = reward_rms
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

    for iteration in range(1, args.num_iterations + 1):
        for step in range(args.num_steps):
            global_step += args.num_envs
            observations[step] = next_obs
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                action, raw_action = sample_actor(next_obs)
            actions[step] = action
            raw_actions[step] = raw_action

            next_obs_np, raw_reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            bootstrap_obs_np = bootstrap_observations(next_obs_np, truncations, infos)
            raw_rewards[step] = torch.as_tensor(raw_reward, dtype=torch.float32, device=device)
            # The paper leaves EMA cadence unspecified. Updating from each
            # vector step gives a ~11k-transition half-life at 16 envs, so the
            # scale tracks HalfCheetah's rapidly changing reward magnitude.
            reward_rms = reward_normalizer.update(raw_reward)
            next_observations_buffer[step] = torch.as_tensor(bootstrap_obs_np, dtype=torch.float32, device=device)
            discounts[step] = args.gamma * (1.0 - torch.as_tensor(terminations, dtype=torch.float32, device=device))
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

            for info in infos.get("final_info", ()):
                if info and "episode" in info:
                    episodic_return = float(info["episode"]["r"])
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # One host transfer per field and rollout, rather than synchronizing
        # CUDA on every environment step.
        raw_rewards_np = raw_rewards.reshape(args.batch_size).cpu().numpy()
        reward_rms_relative_change = reward_rms / previous_iteration_reward_rms - 1.0
        previous_iteration_reward_rms = reward_rms
        replay.add(
            observations.reshape(args.batch_size, obs_dim).cpu().numpy(),
            actions.reshape(args.batch_size, action_dim).cpu().numpy(),
            raw_rewards_np,
            next_observations_buffer.reshape(args.batch_size, obs_dim).cpu().numpy(),
            discounts.reshape(args.batch_size).cpu().numpy(),
        )

        critic_losses = []
        critic_grad_norms = []
        td_target_means = []
        td_target_stds = []
        if replay.size >= args.learning_starts:
            for _ in range(args.critic_updates_per_iteration):
                b_obs, b_actions, b_rewards, b_next_obs, b_discounts = replay.sample(args.critic_batch_size, device)
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    next_actions, _ = target_actor_sample(b_next_obs)
                    # reduce-overhead outputs alias reusable CUDA-graph
                    # storage; the target-critic call is a separate graph.
                    next_actions = next_actions.clone()
                    td_target = one_step_q_target(
                        b_rewards / reward_rms,
                        b_discounts,
                        target_critic_value(b_next_obs, next_actions).clone(),
                    )
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                q_prediction = critic_value(b_obs, b_actions)
                critic_loss = 0.5 * (q_prediction - td_target).square().mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()
                learner_step += 1
                if learner_step % args.target_update_interval == 0:
                    hard_update(target_actor, actor)
                    hard_update(target_critic, critic)
                critic_losses.append(critic_loss.item())
                critic_grad_norms.append(critic_grad_norm.item())
                td_target_means.append(td_target.mean().item())
                td_target_stds.append(td_target.std(unbiased=False).item())

        # A configured replay warmup may span multiple rollouts. Do not train
        # the actor against the critic's random initialization.
        if learner_step == 0:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("critic/replay_size", replay.size, global_step)
            normalized_rewards = raw_rewards / reward_rms
            writer.add_scalar("reward/normalized_mean", normalized_rewards.mean().item(), global_step)
            writer.add_scalar("reward/normalized_min", normalized_rewards.min().item(), global_step)
            writer.add_scalar("reward/normalized_max", normalized_rewards.max().item(), global_step)
            writer.add_scalar("reward/raw_mean", raw_rewards.mean().item(), global_step)
            writer.add_scalar("reward/ema_rms", reward_rms, global_step)
            writer.add_scalar("reward/ema_relative_change", reward_rms_relative_change, global_step)
            continue

        b_obs = observations.reshape(args.batch_size, obs_dim)
        b_actions = actions.reshape(args.batch_size, action_dim)
        b_raw_actions = raw_actions.reshape(args.batch_size, action_dim)
        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            # This value survives the later compiled actor call for diagnostics
            # and the actor advantage, so it must own its storage.
            q_taken = critic_value(b_obs, b_actions).clone()
            # One independent action at the same state gives a symmetric
            # within-state comparison without v16's eight-sample integration.
            # The whole comparison is detached from actor and critic gradients.
            baseline_actions, _ = actor.sample(b_obs)
            q_baseline = critic(b_obs, baseline_actions)
            advantages = q_taken - q_baseline
            fresh_next_obs = next_observations_buffer.reshape(args.batch_size, obs_dim)
            # Diagnostics must not alter the algorithm's subsequent rollout
            # RNG stream, which matters for strict seed-level comparisons.
            with torch.random.fork_rng(devices=[device]):
                fresh_next_actions, _ = target_actor.sample(fresh_next_obs)
            fresh_td_target = one_step_q_target(
                raw_rewards.reshape(args.batch_size) / reward_rms,
                discounts.reshape(args.batch_size),
                target_critic(fresh_next_obs, fresh_next_actions),
            )
            fresh_td_errors = q_taken - fresh_td_target
            fresh_td_rmse = fresh_td_errors.square().mean().sqrt()
            fresh_td_target_std = fresh_td_target.std(unbiased=False)

        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        gaussian_logprob, action_logprob, _, logstd = actor_logprobs(b_obs, b_raw_actions)
        gate, surprisal, delight = delightful_gate(
            advantages,
            action_logprob,
            eta=args.dg_eta,
            surprisal_clip=args.dg_surprisal_clip,
        )
        actor_loss = -(gate.detach() * advantages.detach() * gaussian_logprob).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        actor_optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
        writer.add_scalar("losses/actor", actor_loss.item(), global_step)
        writer.add_scalar("losses/critic_mse", 2.0 * np.mean(critic_losses), global_step)
        writer.add_scalar("losses/fresh_bellman_rmse", fresh_td_rmse.item(), global_step)
        writer.add_scalar(
            "losses/fresh_bellman_normalized_rmse",
            (fresh_td_rmse / fresh_td_target_std.clamp_min(1e-8)).item(),
            global_step,
        )
        writer.add_scalar("losses/critic_grad_norm", np.mean(critic_grad_norms), global_step)
        writer.add_scalar("losses/actor_grad_norm", actor_grad_norm.item(), global_step)
        writer.add_scalar("critic/q_taken_mean", q_taken.mean().item(), global_step)
        writer.add_scalar("critic/q_taken_std", q_taken.std(unbiased=False).item(), global_step)
        writer.add_scalar("critic/q_baseline_mean", q_baseline.mean().item(), global_step)
        writer.add_scalar("critic/td_target_mean", np.mean(td_target_means), global_step)
        writer.add_scalar("critic/td_target_std", np.mean(td_target_stds), global_step)
        writer.add_scalar("critic/fresh_td_target_mean", fresh_td_target.mean().item(), global_step)
        writer.add_scalar("critic/fresh_td_target_std", fresh_td_target_std.item(), global_step)
        writer.add_scalar("critic/replay_size", replay.size, global_step)
        normalized_rewards = raw_rewards / reward_rms
        writer.add_scalar("reward/normalized_mean", normalized_rewards.mean().item(), global_step)
        writer.add_scalar("reward/normalized_min", normalized_rewards.min().item(), global_step)
        writer.add_scalar("reward/normalized_max", normalized_rewards.max().item(), global_step)
        writer.add_scalar("reward/raw_mean", raw_rewards.mean().item(), global_step)
        writer.add_scalar("reward/ema_rms", reward_rms, global_step)
        writer.add_scalar("reward/ema_relative_change", reward_rms_relative_change, global_step)
        writer.add_scalar("advantage/mean", advantages.mean().item(), global_step)
        writer.add_scalar("advantage/std", advantages.std(unbiased=False).item(), global_step)
        writer.add_scalar("dg/surprisal_mean", surprisal.mean().item(), global_step)
        writer.add_scalar(
            "dg/surprisal_clip_fraction",
            ((-action_logprob.detach()).abs() > args.dg_surprisal_clip).float().mean().item(),
            global_step,
        )
        writer.add_scalar("dg/delight_mean", delight.mean().item(), global_step)
        writer.add_scalar("dg/gate_mean", gate.mean().item(), global_step)
        positive_advantage = advantages > 0
        negative_advantage = advantages < 0
        writer.add_scalar("dg/positive_advantage_count", positive_advantage.sum().item(), global_step)
        writer.add_scalar("dg/negative_advantage_count", negative_advantage.sum().item(), global_step)
        if positive_advantage.any():
            writer.add_scalar(
                "dg/gate_positive_advantage",
                gate[positive_advantage].mean().item(),
                global_step,
            )
        if negative_advantage.any():
            writer.add_scalar(
                "dg/gate_negative_advantage",
                gate[negative_advantage].mean().item(),
                global_step,
            )
        writer.add_scalar("policy/logstd_mean", logstd.mean().item(), global_step)
        writer.add_scalar("policy/logstd_min", logstd.min().item(), global_step)
        writer.add_scalar("policy/logstd_max", logstd.max().item(), global_step)
        normalized_actions = (b_actions - actor.action_bias) / actor.action_scale
        writer.add_scalar(
            "policy/action_saturation_fraction",
            (normalized_actions.abs() > 0.95).float().mean().item(),
            global_step,
        )

    episodic_returns = evaluate(actor, args, run_name, device)
    for index, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, global_step + index)
    print(f"eval_return_mean={np.mean(episodic_returns):.3f}, eval_return_std={np.std(episodic_returns):.3f}")
    envs.close()
    writer.close()

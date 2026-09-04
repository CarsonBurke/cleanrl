# Critic-Meta OPSD v2 — every critic-grounded actor gradient is applied.
# =====================================================================================
# Standard OPSD asks an ungrounded conditional policy to interpret a hand-selected
# "better" context, then gives the actor only a detached teacher KL. This file changes the
# learning boundary: a frozen action-value critic scores a reparameterized actor action
# inside the actor graph, and a learned loss generator controls how that score, optional
# TD likelihood, transformed entropy, and free policy-coordinate gradients combine. TD is
# isolated to its explicit branch; it cannot condition the shared loss-generator outputs.
#
# The loss generator is trained through the EXACT proposed actor Adam step. Its outer
# objective is the post-update score from a second, independently initialized Bellman
# critic trained with separate replay draws, optimizer, and target critic. The actor
# therefore receives gradients selected for independently predicted post-update return
# rather than gradients prescribed by advantage weighting, candidate comparisons, outcome
# conditioning, or a fixed teacher geometry. At initialization the generated loss is
# exactly -Q, so improvement pressure exists before the meta-learner knows anything.
#
# Both critics are frozen snapshots during actor/meta construction while dQ/da remains live
# through the inner critic. One reparameterized policy sample scores the whole distribution,
# giving scale a legitimate return gradient rather than scoring only the mean. A direct-Q
# Adam shadow is logged under the identical states and optimizer state. Every finite learned
# actor gradient and every outer meta-gradient is applied. Predicted gain and exact
# transformed-Gaussian KL are diagnostics, never gates: v1's gates rejected most updates and
# froze learning despite a healthy critic. No teacher exists.
#
# PASS: positive held-out predicted gain that remains positive under the next fitted critic,
# finite nonzero dQ/da, a learned gradient that stays useful when it departs from direct Q,
# and return beyond the 10,362 chassis. FAIL: normalized Bellman RMSE stays near or above 1,
# next-critic gain is nonpositive, or two matched checkpoints trail the chassis by >5%.
# =====================================================================================

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
from torch.distributions import Normal, kl_divergence
from torch.func import functional_call
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
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    learned_loss_learning_rate: float = 1e-3
    learned_loss_hidden: int = 128
    num_envs: int = 16
    num_steps: int = 16
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    learned_loss_max_grad_norm: float = 1.0
    meta_fraction: float = 0.5
    gradient_diagnostic_interval: int = 100

    replay_capacity: int = 2_000_000
    critic_batch_size: int = 256
    critic_updates_per_iteration: int = 4
    learning_starts: int = 2_048
    target_update_interval: int = 100
    reward_ema_decay: float = 0.999
    reward_norm_eps: float = 1e-8

    batch_size: int = 0
    num_iterations: int = 0


class ClipObservation(gym.ObservationWrapper):
    """Stationary elementwise clipping without Gymnasium's version-dependent wrapper API."""

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.clip(observation, -10.0, 10.0)


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
        env = ClipObservation(env)
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
        self.action_scale: torch.Tensor
        self.action_bias: torch.Tensor
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, observations):
        features = self.trunk(observations)
        return self.mean(features), state_dependent_logstd(self.raw_scale(features))

    def parameters_for(self, observations):
        return self(observations)

    def transform(self, raw_actions):
        return torch.tanh(raw_actions) * self.action_scale + self.action_bias

    def logprobs(self, observations, raw_actions):
        mean, logstd = self(observations)
        gaussian_logprob = Normal(mean, logstd.exp()).log_prob(raw_actions).sum(dim=-1)
        log_tanh_jacobian = 2.0 * (
            np.log(2.0) - raw_actions - torch.nn.functional.softplus(-2.0 * raw_actions)
        )
        action_logprob = gaussian_logprob - (
            log_tanh_jacobian + torch.log(self.action_scale)
        ).sum(dim=-1)
        return gaussian_logprob, action_logprob, mean, logstd

    def reparameterized(self, observations, noise):
        mean, logstd = self(observations)
        raw_actions = mean + logstd.exp() * noise
        return self.transform(raw_actions), raw_actions, mean, logstd

    def sample(self, observations):
        mean, logstd = self(observations)
        raw_actions = Normal(mean, logstd.exp()).sample()
        return self.transform(raw_actions), raw_actions

    def deterministic(self, observations):
        mean, _ = self(observations)
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


class CriticGroundedActorLoss(nn.Module):
    """Emit a scalar actor loss, initialized to exact direct predicted-return ascent.

    Inputs are detached observations of the current learning problem. The emitted
    coefficients remain differentiable with respect to this module, while the actor sees
    only four explicit gradient paths: critic score, logged-action likelihood, entropy, and
    direct mean/log-scale coordinates. The outer post-update critic score decides their use.
    """

    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        input_dim = obs_dim + 4 * action_dim + 2
        self.body = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden, 3 + 2 * action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.action_dim = action_dim

    def forward(
        self,
        observations,
        mean,
        logstd,
        raw_action,
        normalized_action,
        critic_score,
        critic_action_grad,
        td_advantage,
        logged_action_logprob,
    ):
        score_scale = critic_score.detach().std(unbiased=False).clamp_min(1.0)
        score_feature = (
            (critic_score.detach() - critic_score.detach().mean()) / score_scale
        ).unsqueeze(-1)
        td_scale = td_advantage.detach().std(unbiased=False).clamp_min(1.0)
        td_feature = (td_advantage.detach() / td_scale).clamp(-5.0, 5.0)
        grad_scale = critic_action_grad.detach().square().mean().sqrt().clamp_min(1e-6)
        # TD is deliberately absent from the shared features. It can affect the actor only
        # through the explicit optional likelihood branch below, so pg_weight==0 proves the
        # learned score/entropy/coordinate gradient is outcome-free.
        features = torch.cat(
            [
                (observations.detach() / 10.0).clamp(-1.0, 1.0),
                mean.detach() / (1.0 + mean.detach().abs()),
                logstd.detach(),
                normalized_action.detach(),
                critic_action_grad.detach() / grad_scale,
                score_feature,
                logged_action_logprob.detach().unsqueeze(-1) / 10.0,
            ],
            dim=-1,
        )
        raw = self.output(self.body(features))
        q_logit, pg_logit, entropy_logit = raw[:, :3].unbind(-1)
        mean_coeff = raw[:, 3 : 3 + self.action_dim].tanh()
        logstd_coeff = raw[:, 3 + self.action_dim :].tanh()
        q_weight = 2.0 * torch.sigmoid(q_logit)
        pg_weight = torch.tanh(pg_logit)
        entropy_weight = torch.tanh(entropy_logit)

        score_term = -q_weight * critic_score
        coordinate_term = (
            mean_coeff * mean + logstd_coeff * logstd
        ).mean(dim=-1)
        pg_term = -pg_weight * td_feature * logged_action_logprob
        gaussian_logprob = Normal(mean, logstd.exp()).log_prob(raw_action).sum(dim=-1)
        log_tanh_jacobian = 2.0 * (
            np.log(2.0) - raw_action - torch.nn.functional.softplus(-2.0 * raw_action)
        ).sum(dim=-1)
        transformed_entropy_sample = -(gaussian_logprob - log_tanh_jacobian)
        entropy_term = -entropy_weight * transformed_entropy_sample
        total = (score_term + coordinate_term + pg_term + entropy_term).mean()
        return total, {
            "q_weight": q_weight,
            "pg_weight": pg_weight,
            "entropy_weight": entropy_weight,
            "mean_coeff": mean_coeff,
            "logstd_coeff": logstd_coeff,
            "score_term": score_term,
            "coordinate_term": coordinate_term,
            "pg_term": pg_term,
            "entropy_term": entropy_term,
        }


def differentiable_global_norm_clip(gradients, max_norm):
    total_norm = torch.stack(
        [gradient.float().square().sum() for gradient in gradients]
    ).sum().sqrt()
    clip_coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    return (
        tuple(gradient * clip_coefficient for gradient in gradients),
        total_norm,
        clip_coefficient,
    )


def functional_adam_step(named_parameters, gradients, optimizer):
    """Return the exact next Adam parameters without mutating parameters or optimizer state."""
    if len(optimizer.param_groups) != 1:
        raise ValueError("critic-meta actor requires one Adam parameter group")
    group = optimizer.param_groups[0]
    if group["weight_decay"] != 0 or group["amsgrad"] or group["maximize"]:
        raise ValueError("functional Adam supports the configured plain Adam only")
    beta1, beta2 = group["betas"]
    updated = {}
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        state = optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg", torch.zeros_like(parameter))
        exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter))
        previous_step = state.get("step", 0)
        if isinstance(previous_step, torch.Tensor):
            previous_step = int(previous_step.item())
        step = previous_step + 1
        next_exp_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
        next_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient.square()
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        # Adam's forward value is sqrt(v)+eps, but d sqrt(v)/dv is undefined at v=0.
        # A fresh actor has exactly-zero coordinates, which would turn the outer gradient
        # into NaN despite a finite proposed step. `tiny` is 1e-38 in fp32: its square root
        # is 14 orders below Adam's eps, so it cannot change the represented update while
        # providing the mathematically appropriate zero subgradient at exact-zero v.
        safe_exp_avg_sq = next_exp_avg_sq.clamp_min(
            torch.finfo(next_exp_avg_sq.dtype).tiny
        )
        denominator = safe_exp_avg_sq.sqrt() / np.sqrt(bias_correction2) + group["eps"]
        updated[name] = parameter - group["lr"] * (next_exp_avg / bias_correction1) / denominator
    return updated


def transformed_gaussian_kl(old_mean, old_logstd, new_mean, new_logstd):
    # Both distributions use the same invertible tanh-affine transform, so KL is invariant.
    return kl_divergence(
        Normal(old_mean, old_logstd.exp()),
        Normal(new_mean, new_logstd.exp()),
    ).sum(dim=-1)


def detached_actor_gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    parts = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not parts:
        return loss.new_zeros(())
    return torch.stack(parts).sum().sqrt()


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
        raise ValueError("critic-meta v1 is scoped to HalfCheetah-v4")
    if args.batch_size != 256:
        raise ValueError(f"fresh actor batch must be 256, got {args.batch_size}")
    validate_replay_config(args.replay_capacity, args.critic_batch_size, args.learning_starts)
    if args.critic_updates_per_iteration <= 0 or args.target_update_interval <= 0:
        raise ValueError("critic update counts must be positive")
    if not 0.0 < args.meta_fraction < 1.0:
        raise ValueError("meta_fraction must leave nonempty actor and meta partitions")
    if args.learned_loss_max_grad_norm <= 0.0:
        raise ValueError("learned_loss_max_grad_norm must be positive")
    if args.gradient_diagnostic_interval <= 0:
        raise ValueError("gradient_diagnostic_interval must be positive")
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
    assert envs.single_observation_space.shape is not None
    assert envs.single_action_space.shape is not None
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    actor = Actor(
        obs_dim,
        action_dim,
        envs.single_action_space.low,
        envs.single_action_space.high,
    ).to(device)
    critic = QCritic(obs_dim, action_dim).to(device)
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(args.seed + 1_000_003)
        validation_critic = QCritic(obs_dim, action_dim).to(device)
    target_actor = copy.deepcopy(actor).requires_grad_(False)
    target_critic = copy.deepcopy(critic).requires_grad_(False)
    target_validation_critic = copy.deepcopy(validation_critic).requires_grad_(False)
    score_critic = copy.deepcopy(critic).requires_grad_(False)
    validation_score_critic = copy.deepcopy(validation_critic).requires_grad_(False)
    previous_actor = copy.deepcopy(actor).requires_grad_(False)
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(args.seed + 2_000_003)
        learned_actor_loss = CriticGroundedActorLoss(
            obs_dim, action_dim, args.learned_loss_hidden
        ).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate, eps=1e-5)
    validation_critic_optimizer = optim.Adam(
        validation_critic.parameters(), lr=args.learning_rate, eps=1e-5
    )
    learned_loss_optimizer = optim.Adam(
        learned_actor_loss.parameters(),
        lr=args.learned_loss_learning_rate,
        eps=1e-5,
    )
    meta_generator = torch.Generator(device=device)
    meta_generator.manual_seed(args.seed + 3_000_003)
    diagnostic_generator = torch.Generator(device=device)
    diagnostic_generator.manual_seed(args.seed + 4_000_003)
    previous_update_available = False
    actor_named_parameters = tuple(actor.named_parameters())
    actor_parameters = tuple(parameter for _, parameter in actor_named_parameters)
    replay = ReplayBuffer(
        args.replay_capacity,
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        args.seed,
    )
    reward_normalizer = EMARewardNormalizer(args.reward_ema_decay, args.reward_norm_eps)

    sample_actor = actor.sample
    critic_value = critic.forward
    validation_critic_value = validation_critic.forward
    target_actor_sample = target_actor.sample
    target_critic_value = target_critic.forward
    target_validation_critic_value = target_validation_critic.forward
    if args.compile:
        sample_actor = torch.compile(sample_actor, mode=args.compile_mode, dynamic=False)
        critic_value = torch.compile(critic_value, mode=args.compile_mode, dynamic=False)
        validation_critic_value = torch.compile(
            validation_critic_value, mode=args.compile_mode, dynamic=False
        )
        target_actor_sample = torch.compile(
            target_actor_sample, mode=args.compile_mode, dynamic=False
        )
        target_critic_value = torch.compile(
            target_critic_value, mode=args.compile_mode, dynamic=False
        )
        target_validation_critic_value = torch.compile(
            target_validation_critic_value, mode=args.compile_mode, dynamic=False
        )
        print(f"compiled rollout actor and Bellman Q functions ({args.compile_mode})")

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
        validation_critic_losses = []
        validation_critic_grad_norms = []
        td_target_means = []
        td_target_stds = []
        validation_td_target_means = []
        validation_td_target_stds = []
        if replay.size >= args.learning_starts:
            for _ in range(args.critic_updates_per_iteration):
                b_obs, b_actions, b_rewards, b_next_obs, b_discounts = replay.sample(
                    args.critic_batch_size, device
                )
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    next_actions, _ = target_actor_sample(b_next_obs)
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
                if not bool(
                    torch.isfinite(q_prediction).all()
                    and torch.isfinite(td_target).all()
                    and torch.isfinite(critic_loss)
                ):
                    raise FloatingPointError("non-finite training critic update")
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    critic.parameters(),
                    args.max_grad_norm,
                    error_if_nonfinite=True,
                )
                critic_optimizer.step()

                # Independent initialization, replay draw, bootstrap critic, and optimizer.
                # This critic never supplies the inner dQ/da path; it judges the proposed
                # update and prevents that path from validating its own extrapolation error.
                (
                    v_obs,
                    v_actions,
                    v_rewards,
                    v_next_obs,
                    v_discounts,
                ) = replay.sample(args.critic_batch_size, device)
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    v_next_actions, _ = target_actor_sample(v_next_obs)
                    v_next_actions = v_next_actions.clone()
                    validation_td_target = one_step_q_target(
                        v_rewards / reward_rms,
                        v_discounts,
                        target_validation_critic_value(
                            v_next_obs, v_next_actions
                        ).clone(),
                    )
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                validation_q_prediction = validation_critic_value(
                    v_obs, v_actions
                )
                validation_critic_loss = 0.5 * (
                    validation_q_prediction - validation_td_target
                ).square().mean()
                if not bool(
                    torch.isfinite(validation_q_prediction).all()
                    and torch.isfinite(validation_td_target).all()
                    and torch.isfinite(validation_critic_loss)
                ):
                    raise FloatingPointError("non-finite validation critic update")
                validation_critic_optimizer.zero_grad(set_to_none=True)
                validation_critic_loss.backward()
                validation_critic_grad_norm = nn.utils.clip_grad_norm_(
                    validation_critic.parameters(),
                    args.max_grad_norm,
                    error_if_nonfinite=True,
                )
                validation_critic_optimizer.step()

                learner_step += 1
                if learner_step % args.target_update_interval == 0:
                    hard_update(target_actor, actor)
                    hard_update(target_critic, critic)
                    hard_update(target_validation_critic, validation_critic)
                critic_losses.append(critic_loss.item())
                critic_grad_norms.append(critic_grad_norm.item())
                validation_critic_losses.append(validation_critic_loss.item())
                validation_critic_grad_norms.append(
                    validation_critic_grad_norm.item()
                )
                td_target_means.append(td_target.mean().item())
                td_target_stds.append(td_target.std(unbiased=False).item())
                validation_td_target_means.append(
                    validation_td_target.mean().item()
                )
                validation_td_target_stds.append(
                    validation_td_target.std(unbiased=False).item()
                )
        if critic_losses:
            hard_update(score_critic, critic)
            hard_update(validation_score_critic, validation_critic)

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
        fresh_next_obs = next_observations_buffer.reshape(args.batch_size, obs_dim)
        with torch.no_grad():
            q_taken = score_critic(b_obs, b_actions)
            validation_q_taken = validation_score_critic(b_obs, b_actions)
            # This is optional score-function evidence for the learned loss, not an actor
            # objective. The meta-learner is free to set its emitted PG weight to zero.
            with torch.random.fork_rng(devices=[device]):
                fresh_next_actions, _ = target_actor.sample(fresh_next_obs)
            fresh_td_target = one_step_q_target(
                raw_rewards.reshape(args.batch_size) / reward_rms,
                discounts.reshape(args.batch_size),
                target_critic(fresh_next_obs, fresh_next_actions),
            )
            validation_fresh_td_target = one_step_q_target(
                raw_rewards.reshape(args.batch_size) / reward_rms,
                discounts.reshape(args.batch_size),
                target_validation_critic(
                    fresh_next_obs, fresh_next_actions
                ),
            )
            td_advantages = fresh_td_target - q_taken
            fresh_td_errors = q_taken - fresh_td_target
            fresh_td_rmse = fresh_td_errors.square().mean().sqrt()
            fresh_td_target_std = fresh_td_target.std(unbiased=False)
            validation_fresh_td_errors = (
                validation_q_taken - validation_fresh_td_target
            )
            validation_fresh_td_rmse = (
                validation_fresh_td_errors.square().mean().sqrt()
            )
            validation_fresh_td_target_std = (
                validation_fresh_td_target.std(unbiased=False)
            )
            logged_critic_disagreement = (
                q_taken - validation_q_taken
            ).abs().mean()

        # Confirm the PREVIOUS actor step under a newly fitted critic and new states. The
        # same-noise counterfactual is still model-based, but unlike the outer objective it
        # cannot be positive merely because the previous step optimized a frozen snapshot.
        with torch.no_grad():
            if previous_update_available:
                confirmation_noise = torch.randn(
                    (args.batch_size, action_dim),
                    generator=diagnostic_generator,
                    device=device,
                )
                current_action, _, _, _ = actor.reparameterized(
                    b_obs, confirmation_noise
                )
                previous_action, _, _, _ = previous_actor.reparameterized(
                    b_obs, confirmation_noise
                )
                next_critic_confirmed_gain = (
                    validation_score_critic(b_obs, current_action)
                    - validation_score_critic(b_obs, previous_action)
                ).mean()
            else:
                next_critic_confirmed_gain = torch.full(
                    (), float("nan"), device=device
                )
            # Preserve the exact pre-update actor for confirmation on the next rollout.
            hard_update(previous_actor, actor)

        # Disjoint fresh-state partitions: one creates the actor gradient; the other
        # supervises the learned loss through the exact proposed Adam step.
        permutation = torch.randperm(
            args.batch_size, generator=meta_generator, device=device
        )
        actor_count = int(round(args.batch_size * (1.0 - args.meta_fraction)))
        actor_count = min(max(actor_count, 1), args.batch_size - 1)
        actor_indices = permutation[:actor_count]
        meta_indices = permutation[actor_count:]
        actor_obs = b_obs[actor_indices]
        actor_mean, actor_logstd = actor(actor_obs)
        actor_noise = torch.randn(
            actor_mean.shape, generator=meta_generator, device=device
        )
        actor_raw_sample = actor_mean + actor_logstd.exp() * actor_noise
        actor_action = actor.transform(actor_raw_sample)
        actor_score = score_critic(actor_obs, actor_action)
        (critic_action_grad,) = torch.autograd.grad(
            actor_score.sum(), actor_action, retain_graph=True
        )
        with torch.no_grad():
            validation_actor_score = validation_score_critic(
                actor_obs, actor_action.detach()
            )
            actor_score_disagreement = (
                actor_score.detach() - validation_actor_score
            ).abs().mean()
        logged_action_logprob = Normal(
            actor_mean, actor_logstd.exp()
        ).log_prob(b_raw_actions[actor_indices]).sum(dim=-1)
        normalized_actor_action = (
            actor_action - actor.action_bias
        ) / actor.action_scale
        actor_loss, loss_outputs = learned_actor_loss(
            actor_obs,
            actor_mean,
            actor_logstd,
            actor_raw_sample,
            normalized_actor_action,
            actor_score,
            critic_action_grad,
            td_advantages[actor_indices],
            logged_action_logprob,
        )
        if not bool(
            torch.isfinite(actor_score).all()
            and torch.isfinite(critic_action_grad).all()
            and torch.isfinite(actor_loss)
        ):
            raise FloatingPointError("non-finite critic-grounded actor objective")
        score_term_grad_norm = torch.full((), float("nan"), device=device)
        coordinate_term_grad_norm = torch.full((), float("nan"), device=device)
        pg_term_grad_norm = torch.full((), float("nan"), device=device)
        entropy_term_grad_norm = torch.full((), float("nan"), device=device)
        if iteration % args.gradient_diagnostic_interval == 0:
            score_term_grad_norm = detached_actor_gradient_norm(
                loss_outputs["score_term"].mean(), actor_parameters
            )
            coordinate_term_grad_norm = detached_actor_gradient_norm(
                loss_outputs["coordinate_term"].mean(), actor_parameters
            )
            pg_term_grad_norm = detached_actor_gradient_norm(
                loss_outputs["pg_term"].mean(), actor_parameters
            )
            entropy_term_grad_norm = detached_actor_gradient_norm(
                loss_outputs["entropy_term"].mean(), actor_parameters
            )

        # The direct -Q gradient is both the exact initialization and the reference
        # direction. A learned loss that rotates away must earn that rotation in return.
        direct_score_loss = -actor_score.mean()
        direct_gradients = torch.autograd.grad(
            direct_score_loss,
            actor_parameters,
            retain_graph=True,
            allow_unused=False,
        )
        actor_gradients = torch.autograd.grad(
            actor_loss,
            actor_parameters,
            create_graph=True,
            allow_unused=False,
        )
        if not all(
            bool(torch.isfinite(gradient).all())
            for gradient in (*direct_gradients, *actor_gradients)
        ):
            raise FloatingPointError("non-finite critic-grounded actor gradient")
        clipped_actor_gradients, actor_grad_norm, actor_clip_coefficient = (
            differentiable_global_norm_clip(actor_gradients, args.max_grad_norm)
        )
        (
            clipped_direct_gradients,
            direct_grad_norm,
            direct_clip_coefficient,
        ) = differentiable_global_norm_clip(
            direct_gradients, args.max_grad_norm
        )
        direct_sq = torch.stack(
            [
                gradient.detach().float().square().sum()
                for gradient in direct_gradients
            ]
        ).sum()
        learned_sq = torch.stack(
            [
                gradient.detach().float().square().sum()
                for gradient in actor_gradients
            ]
        ).sum()
        direct_learned_dot = torch.stack(
            [
                (direct.detach().float() * learned.detach().float()).sum()
                for direct, learned in zip(
                    direct_gradients, actor_gradients, strict=True
                )
            ]
        ).sum()
        direct_learned_cosine = direct_learned_dot / (
            direct_sq.sqrt() * learned_sq.sqrt()
        ).clamp_min(1e-12)

        shadow_parameters = functional_adam_step(
            actor_named_parameters,
            clipped_actor_gradients,
            actor_optimizer,
        )
        direct_shadow_parameters = functional_adam_step(
            actor_named_parameters,
            clipped_direct_gradients,
            actor_optimizer,
        )
        learned_delta_sq = torch.stack(
            [
                (shadow_parameters[name] - parameter).detach().float().square().sum()
                for name, parameter in actor_named_parameters
            ]
        ).sum()
        direct_delta_sq = torch.stack(
            [
                (direct_shadow_parameters[name] - parameter)
                .detach()
                .float()
                .square()
                .sum()
                for name, parameter in actor_named_parameters
            ]
        ).sum()
        delta_dot = torch.stack(
            [
                (
                    (shadow_parameters[name] - parameter).detach().float()
                    * (direct_shadow_parameters[name] - parameter).detach().float()
                ).sum()
                for name, parameter in actor_named_parameters
            ]
        ).sum()
        direct_learned_update_cosine = delta_dot / (
            learned_delta_sq.sqrt() * direct_delta_sq.sqrt()
        ).clamp_min(1e-12)
        with torch.no_grad():
            old_mean, old_logstd = actor(b_obs)
            proposed_mean, proposed_logstd = functional_call(
                actor, shadow_parameters, (b_obs,)
            )
            proposed_actor_kl = transformed_gaussian_kl(
                old_mean, old_logstd, proposed_mean, proposed_logstd
            ).mean()
            if not bool(torch.isfinite(proposed_actor_kl)):
                raise FloatingPointError("non-finite proposed actor KL")

        meta_obs = b_obs[meta_indices]
        meta_noise = torch.randn(
            (meta_obs.shape[0], action_dim),
            generator=meta_generator,
            device=device,
        )
        with torch.no_grad():
            meta_action_before, _, _, _ = actor.reparameterized(
                meta_obs, meta_noise
            )
            meta_score_before = validation_score_critic(
                meta_obs, meta_action_before
            ).mean()
        shadow_mean, shadow_logstd = functional_call(
            actor, shadow_parameters, (meta_obs,)
        )
        shadow_raw_action = shadow_mean + shadow_logstd.exp() * meta_noise
        shadow_action = actor.transform(shadow_raw_action)
        meta_score_after = validation_score_critic(
            meta_obs, shadow_action
        ).mean()
        inner_meta_score_after = score_critic(
            meta_obs, shadow_action
        ).mean()
        meta_predicted_gain = meta_score_after - meta_score_before
        meta_loss = -meta_score_after

        with torch.no_grad():
            direct_shadow_mean, direct_shadow_logstd = functional_call(
                actor, direct_shadow_parameters, (meta_obs,)
            )
            direct_shadow_raw_action = (
                direct_shadow_mean
                + direct_shadow_logstd.exp() * meta_noise
            )
            direct_shadow_action = actor.transform(direct_shadow_raw_action)
            direct_meta_score_after = validation_score_critic(
                meta_obs, direct_shadow_action
            ).mean()
            direct_meta_predicted_gain = (
                direct_meta_score_after - meta_score_before
            )
            shadow_critic_disagreement = (
                inner_meta_score_after.detach() - meta_score_after.detach()
            ).abs()
            if not bool(
                torch.isfinite(meta_score_before)
                and torch.isfinite(meta_score_after)
                and torch.isfinite(inner_meta_score_after)
                and torch.isfinite(direct_meta_score_after)
            ):
                raise FloatingPointError("non-finite independent critic score")

        learned_loss_gradients = torch.autograd.grad(
            meta_loss,
            tuple(learned_actor_loss.parameters()),
            allow_unused=False,
        )
        if not all(
            bool(torch.isfinite(gradient).all())
            for gradient in learned_loss_gradients
        ):
            raise FloatingPointError("non-finite learned-loss meta gradient")
        learned_loss_optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(
            learned_actor_loss.parameters(),
            learned_loss_gradients,
            strict=True,
        ):
            parameter.grad = gradient.detach()
        meta_grad_norm = nn.utils.clip_grad_norm_(
            learned_actor_loss.parameters(),
            args.learned_loss_max_grad_norm,
            error_if_nonfinite=True,
        )
        learned_loss_optimizer.step()

        # Commit exactly the update scored above. Neither critic score nor KL acts as a gate:
        # the outer gradient learns from bad proposals, and the actor continually receives
        # the critic-grounded gradient rather than stalling on a noisy binary decision.
        actor_optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(
            actor_parameters, clipped_actor_gradients, strict=True
        ):
            parameter.grad = gradient.detach()
        actor_optimizer.step()
        with torch.no_grad():
            shadow_commit_error = torch.stack(
                [
                    (parameter - shadow_parameters[name]).abs().max()
                    for name, parameter in actor_named_parameters
                ]
            ).max()
            if shadow_commit_error > 1e-6:
                raise RuntimeError(
                    "committed actor does not match scored Adam shadow"
                )
        previous_update_available = True

        with torch.no_grad():
            _, logstd = actor(b_obs)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
        writer.add_scalar(
            "charts/learned_loss_learning_rate",
            args.learned_loss_learning_rate,
            global_step,
        )
        writer.add_scalar("losses/actor_generated", actor_loss.item(), global_step)
        writer.add_scalar("losses/meta_objective", meta_loss.item(), global_step)
        writer.add_scalar("losses/critic_mse", 2.0 * np.mean(critic_losses), global_step)
        writer.add_scalar(
            "losses/validation_critic_mse",
            2.0 * np.mean(validation_critic_losses),
            global_step,
        )
        writer.add_scalar("losses/fresh_bellman_rmse", fresh_td_rmse.item(), global_step)
        writer.add_scalar(
            "losses/fresh_bellman_normalized_rmse",
            (fresh_td_rmse / fresh_td_target_std.clamp_min(1e-8)).item(),
            global_step,
        )
        writer.add_scalar(
            "losses/validation_fresh_bellman_rmse",
            validation_fresh_td_rmse.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/validation_fresh_bellman_normalized_rmse",
            (
                validation_fresh_td_rmse
                / validation_fresh_td_target_std.clamp_min(1e-8)
            ).item(),
            global_step,
        )
        writer.add_scalar("losses/critic_grad_norm", np.mean(critic_grad_norms), global_step)
        writer.add_scalar(
            "losses/validation_critic_grad_norm",
            np.mean(validation_critic_grad_norms),
            global_step,
        )
        writer.add_scalar("losses/actor_grad_norm", actor_grad_norm.item(), global_step)
        writer.add_scalar("losses/meta_grad_norm", meta_grad_norm.item(), global_step)
        writer.add_scalar(
            "critic/q_taken_mean", q_taken.mean().item(), global_step
        )
        writer.add_scalar(
            "critic/q_taken_std", q_taken.std(unbiased=False).item(), global_step
        )
        writer.add_scalar(
            "critic/validation_q_taken_mean",
            validation_q_taken.mean().item(),
            global_step,
        )
        writer.add_scalar(
            "critic/validation_q_taken_std",
            validation_q_taken.std(unbiased=False).item(),
            global_step,
        )
        writer.add_scalar(
            "critic/logged_action_disagreement",
            logged_critic_disagreement.item(),
            global_step,
        )
        writer.add_scalar(
            "critic/actor_action_disagreement",
            actor_score_disagreement.item(),
            global_step,
        )
        writer.add_scalar(
            "critic/shadow_action_disagreement",
            shadow_critic_disagreement.item(),
            global_step,
        )
        writer.add_scalar("critic/td_target_mean", np.mean(td_target_means), global_step)
        writer.add_scalar("critic/td_target_std", np.mean(td_target_stds), global_step)
        writer.add_scalar(
            "critic/validation_td_target_mean",
            np.mean(validation_td_target_means),
            global_step,
        )
        writer.add_scalar(
            "critic/validation_td_target_std",
            np.mean(validation_td_target_stds),
            global_step,
        )
        writer.add_scalar(
            "critic/fresh_td_target_mean", fresh_td_target.mean().item(), global_step
        )
        writer.add_scalar(
            "critic/fresh_td_target_std", fresh_td_target_std.item(), global_step
        )
        writer.add_scalar(
            "critic/validation_fresh_td_target_mean",
            validation_fresh_td_target.mean().item(),
            global_step,
        )
        writer.add_scalar(
            "critic/validation_fresh_td_target_std",
            validation_fresh_td_target_std.item(),
            global_step,
        )
        writer.add_scalar("critic/replay_size", replay.size, global_step)
        writer.add_scalar(
            "critic/action_gradient_rms",
            critic_action_grad.detach().square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/predicted_score_before", meta_score_before.item(), global_step
        )
        writer.add_scalar(
            "meta/predicted_score_after", meta_score_after.item(), global_step
        )
        writer.add_scalar(
            "meta/predicted_gain", meta_predicted_gain.item(), global_step
        )
        writer.add_scalar(
            "meta/direct_predicted_score_after",
            direct_meta_score_after.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/direct_predicted_gain",
            direct_meta_predicted_gain.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/learned_minus_direct_predicted_gain",
            (meta_predicted_gain.detach() - direct_meta_predicted_gain).item(),
            global_step,
        )
        writer.add_scalar(
            "meta/training_critic_shadow_score",
            inner_meta_score_after.detach().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/next_critic_confirmed_gain",
            next_critic_confirmed_gain.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/proposed_actor_kl", proposed_actor_kl.item(), global_step
        )
        writer.add_scalar(
            "meta/actor_update_applied",
            1.0,
            global_step,
        )
        writer.add_scalar(
            "meta/meta_update_applied",
            1.0,
            global_step,
        )
        writer.add_scalar(
            "meta/shadow_commit_max_error", shadow_commit_error.item(), global_step
        )
        writer.add_scalar(
            "meta/direct_learned_gradient_cosine",
            direct_learned_cosine.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/direct_learned_update_cosine",
            direct_learned_update_cosine.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/direct_actor_grad_norm",
            direct_grad_norm.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/actor_clip_coefficient",
            actor_clip_coefficient.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/direct_clip_coefficient",
            direct_clip_coefficient.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/q_weight_mean",
            loss_outputs["q_weight"].detach().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/q_weight_std",
            loss_outputs["q_weight"].detach().std(unbiased=False).item(),
            global_step,
        )
        writer.add_scalar(
            "meta/q_weight_absmean",
            loss_outputs["q_weight"].detach().abs().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/q_weight_rms",
            loss_outputs["q_weight"].detach().square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/pg_weight_mean",
            loss_outputs["pg_weight"].detach().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/pg_weight_absmean",
            loss_outputs["pg_weight"].detach().abs().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/pg_weight_std",
            loss_outputs["pg_weight"].detach().std(unbiased=False).item(),
            global_step,
        )
        writer.add_scalar(
            "meta/pg_weight_rms",
            loss_outputs["pg_weight"].detach().square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/entropy_weight_mean",
            loss_outputs["entropy_weight"].detach().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/entropy_weight_absmean",
            loss_outputs["entropy_weight"].detach().abs().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/entropy_weight_std",
            loss_outputs["entropy_weight"].detach().std(unbiased=False).item(),
            global_step,
        )
        writer.add_scalar(
            "meta/entropy_weight_rms",
            loss_outputs["entropy_weight"].detach().square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/mean_coordinate_coefficient_absmean",
            loss_outputs["mean_coeff"].detach().abs().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/mean_coordinate_coefficient_rms",
            loss_outputs["mean_coeff"].detach().square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/logstd_coordinate_coefficient_absmean",
            loss_outputs["logstd_coeff"].detach().abs().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/logstd_coordinate_coefficient_rms",
            loss_outputs["logstd_coeff"].detach().square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/score_term_mean",
            loss_outputs["score_term"].detach().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/coordinate_term_mean",
            loss_outputs["coordinate_term"].detach().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/pg_term_mean",
            loss_outputs["pg_term"].detach().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "meta/entropy_term_mean",
            loss_outputs["entropy_term"].detach().mean().item(),
            global_step,
        )
        if iteration % args.gradient_diagnostic_interval == 0:
            writer.add_scalar(
                "gradient_paths/score_term_norm",
                score_term_grad_norm.item(),
                global_step,
            )
            writer.add_scalar(
                "gradient_paths/coordinate_term_norm",
                coordinate_term_grad_norm.item(),
                global_step,
            )
            writer.add_scalar(
                "gradient_paths/optional_pg_term_norm",
                pg_term_grad_norm.item(),
                global_step,
            )
            writer.add_scalar(
                "gradient_paths/transformed_entropy_term_norm",
                entropy_term_grad_norm.item(),
                global_step,
            )
        normalized_rewards = raw_rewards / reward_rms
        writer.add_scalar("reward/normalized_mean", normalized_rewards.mean().item(), global_step)
        writer.add_scalar("reward/normalized_min", normalized_rewards.min().item(), global_step)
        writer.add_scalar("reward/normalized_max", normalized_rewards.max().item(), global_step)
        writer.add_scalar("reward/raw_mean", raw_rewards.mean().item(), global_step)
        writer.add_scalar("reward/ema_rms", reward_rms, global_step)
        writer.add_scalar("reward/ema_relative_change", reward_rms_relative_change, global_step)
        writer.add_scalar(
            "optional_td_advantage/mean", td_advantages.mean().item(), global_step
        )
        writer.add_scalar(
            "optional_td_advantage/std",
            td_advantages.std(unbiased=False).item(),
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

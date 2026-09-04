# Critic-Metric OPSD RawReward v5 — stationary raw-reward Bellman targets.
# =====================================================================================
# v4 stabilizes after roughly 4M steps and keeps improving, but its final return remains weak.
# Its EMA reward RMS changes the Bellman operator throughout training: as the policy's reward
# magnitude changes, every historical replay reward is rescaled by the latest statistic. That
# couples policy progress back into critic scale, target-network lag, gradient clipping, and the
# learned action metric even though the environment's reward function is stationary.
#
# v5 is the clean raw-reward ablation. Training, validation, and fresh-rollout TD targets all use
# the environment reward exactly as stored; the EMA state and normalized-reward diagnostics are
# removed. Every v4 stability and throughput change remains unchanged, so the run isolates the
# effect of a stationary Bellman target rather than retuning around it.
#
# Raw rewards enlarge Q values and critic gradients. This is intentional: changing critic rates
# or clipping simultaneously would confound the ablation. Adam and the existing global-norm caps
# still bound parameter updates, while diagnostics reveal whether raw scale causes persistent
# clipping, critic underfit, or KL-cap activation.
#
# PASS: learning begins no later than v4, improvement continues beyond 4M, critic relative RMSE
# remains controlled, and episodic return materially exceeds v4 without dominant KL capping.
# FAIL: critic clipping/relative error saturates, actor learning stalls, non-finite values occur,
# or late return does not improve over v4.
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
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 3e-4
    metric_learning_rate: float = 1e-3
    metric_hidden: int = 128
    num_envs: int = 16
    num_steps: int = 16
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    metric_max_grad_norm: float = 1.0
    target_actor_kl: float = 0.01
    critic_gain_disagreement_penalty: float = 1.0
    meta_fraction: float = 0.5
    metric_identity_fraction: float = 0.25
    bf16_critic: bool = True
    async_vector_env: bool = True
    log_interval: int = 10

    replay_capacity: int = 2_000_000
    critic_batch_size: int = 256
    critic_updates_per_iteration: int = 4
    learning_starts: int = 2_048
    actor_learning_starts: int = 25_600
    target_update_interval: int = 100

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
    """CUDA-resident replay; the full 2M-transition buffer is ~336 MiB."""

    def __init__(self, capacity, obs_shape, action_shape, seed, device):
        self.capacity = capacity
        self.device = device
        self.observations = torch.empty(
            (capacity, *obs_shape), dtype=torch.float32, device=device
        )
        self.actions = torch.empty(
            (capacity, *action_shape), dtype=torch.float32, device=device
        )
        self.rewards = torch.empty(capacity, dtype=torch.float32, device=device)
        self.next_observations = torch.empty(
            (capacity, *obs_shape), dtype=torch.float32, device=device
        )
        self.discounts = torch.empty(
            capacity, dtype=torch.float32, device=device
        )
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def add(self, observations, actions, rewards, next_observations, discounts):
        batch_size = observations.shape[0]
        if not all(
            field.shape[0] == batch_size
            for field in (
                actions,
                rewards,
                next_observations,
                discounts,
            )
        ):
            raise ValueError("replay fields must have equal batch length")
        skip = max(0, batch_size - self.capacity)
        write_position = (self.position + skip) % self.capacity
        indices = (
            torch.arange(
                batch_size - skip,
                device=self.device,
            )
            + write_position
        ) % self.capacity
        self.observations.index_copy_(0, indices, observations[skip:])
        self.actions.index_copy_(0, indices, actions[skip:])
        self.rewards.index_copy_(0, indices, rewards[skip:])
        self.next_observations.index_copy_(
            0, indices, next_observations[skip:]
        )
        self.discounts.index_copy_(0, indices, discounts[skip:])
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay")
        indices = torch.randint(
            self.size,
            (batch_size,),
            generator=self.generator,
            device=self.device,
        )
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.discounts[indices],
        )


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


class CriticGradientMetric(nn.Module):
    """Learn action-space geometry without creating a critic-independent actor gradient.

    The emitted positive-definite metric preconditions dQ/da. All network inputs are
    detached, so the actor has exactly one gradient path: action -> metric @ dQ/da. A
    zero-initialized output gives the identity metric and therefore the exact direct-Q actor
    gradient. The trace normalization fixes scale so the meta-learner can change geometry,
    not manufacture an arbitrarily large update.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden,
        identity_fraction,
    ):
        super().__init__()
        input_dim = obs_dim + 4 * action_dim + 1
        self.body = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden, action_dim * action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.action_dim = action_dim
        self.identity_fraction = identity_fraction
        self.identity: torch.Tensor
        self.register_buffer("identity", torch.eye(action_dim))

    def forward(
        self,
        observations,
        mean,
        logstd,
        action,
        normalized_action,
        critic_score,
        critic_action_grad,
    ):
        score_scale = critic_score.detach().std(unbiased=False).clamp_min(1.0)
        score_feature = (
            (critic_score.detach() - critic_score.detach().mean()) / score_scale
        ).unsqueeze(-1)
        detached_action_grad = critic_action_grad.detach()
        grad_scale = detached_action_grad.square().mean().sqrt().clamp_min(1e-6)
        features = torch.cat(
            [
                (observations.detach() / 10.0).clamp(-1.0, 1.0),
                mean.detach() / (1.0 + mean.detach().abs()),
                logstd.detach(),
                normalized_action.detach(),
                detached_action_grad / grad_scale,
                score_feature,
            ],
            dim=-1,
        )
        raw_factor = self.output(self.body(features)).view(
            -1, self.action_dim, self.action_dim
        )
        identity = self.identity.expand(raw_factor.shape[0], -1, -1)
        factor = identity + raw_factor
        metric = factor @ factor.transpose(-1, -2) + 1e-4 * identity
        metric_trace = metric.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        metric = metric * (
            self.action_dim / metric_trace.clamp_min(1e-8)
        ).view(-1, 1, 1)
        metric = (
            self.identity_fraction * identity
            + (1.0 - self.identity_fraction) * metric
        )
        action_direction = (
            metric @ detached_action_grad.unsqueeze(-1)
        ).squeeze(-1)
        actor_loss = -(action_direction * action).sum(dim=-1).mean()
        ascent_dot = (action_direction * detached_action_grad).sum(dim=-1)
        action_alignment = ascent_dot / (
            action_direction.norm(dim=-1)
            * detached_action_grad.norm(dim=-1)
        ).clamp_min(1e-8)
        return actor_loss, {
            "metric": metric,
            "action_direction": action_direction,
            "ascent_dot": ascent_dot,
            "action_alignment": action_alignment,
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


def policy_normalized_adam_step(
    actor,
    named_parameters,
    gradients,
    optimizer,
    observations,
    target_kl,
    differentiable_scale=True,
):
    """Apply every Adam direction while enforcing a differentiable policy-KL cap."""
    full_step_parameters = functional_adam_step(
        named_parameters, gradients, optimizer
    )
    with torch.no_grad():
        old_mean, old_logstd = actor(observations)

        def detached_kl_at_scale(scale):
            trial_parameters = {
                name: parameter.detach()
                + scale
                * (
                    full_step_parameters[name].detach()
                    - parameter.detach()
                )
                for name, parameter in named_parameters
            }
            trial_mean, trial_logstd = functional_call(
                actor, trial_parameters, (observations,)
            )
            return transformed_gaussian_kl(
                old_mean, old_logstd, trial_mean, trial_logstd
            ).mean()

        one = torch.ones((), device=observations.device)
        full_step_kl = detached_kl_at_scale(one)
        if not bool(torch.isfinite(full_step_kl)):
            raise FloatingPointError("non-finite full Adam-step policy KL")
        cap_active = bool(full_step_kl > target_kl)
        fallback_used = False
        if cap_active:
            root_scale = (
                target_kl / full_step_kl.clamp_min(1e-12)
            ).sqrt()
            for _ in range(2):
                trial_kl = detached_kl_at_scale(root_scale)
                if not bool(torch.isfinite(trial_kl)):
                    raise FloatingPointError("non-finite normalized policy KL")
                root_scale = root_scale * (
                    target_kl / trial_kl.clamp_min(1e-12)
                ).sqrt()
            applied_kl = detached_kl_at_scale(root_scale)
            tolerance = max(1e-7, target_kl * 1e-4)
            if (
                not bool(torch.isfinite(applied_kl))
                or abs(applied_kl.item() - target_kl) > tolerance
            ):
                fallback_used = True
                low = torch.zeros_like(root_scale)
                high = torch.ones_like(root_scale)
                for _ in range(22):
                    midpoint = 0.5 * (low + high)
                    midpoint_kl = detached_kl_at_scale(midpoint)
                    if not bool(torch.isfinite(midpoint_kl)):
                        raise FloatingPointError(
                            "non-finite policy KL during cap solve"
                        )
                    if bool(midpoint_kl <= target_kl):
                        low = midpoint
                    else:
                        high = midpoint
                root_scale = low
                applied_kl = detached_kl_at_scale(root_scale)
            if (
                not bool(torch.isfinite(applied_kl))
                or applied_kl > target_kl + tolerance
                or abs(applied_kl.item() - target_kl) > tolerance
            ):
                raise RuntimeError("policy KL cap solve missed its tolerance")
        else:
            root_scale = one
            applied_kl = full_step_kl

    if cap_active and differentiable_scale:
        scale_leaf = root_scale.detach().requires_grad_(True)
        constraint_parameters = {
            name: parameter
            + scale_leaf * (full_step_parameters[name] - parameter)
            for name, parameter in named_parameters
        }
        constraint_mean, constraint_logstd = functional_call(
            actor, constraint_parameters, (observations,)
        )
        constraint_kl = transformed_gaussian_kl(
            old_mean, old_logstd, constraint_mean, constraint_logstd
        ).mean()
        (constraint_slope,) = torch.autograd.grad(
            constraint_kl, scale_leaf, retain_graph=True
        )
        if (
            not bool(torch.isfinite(constraint_slope))
            or constraint_slope.abs() <= 1e-10
        ):
            raise FloatingPointError("invalid policy KL constraint slope")
        # Forward value is the solved root. The zero-valued correction supplies the
        # implicit derivative d alpha / d metric for KL(alpha, metric) = target.
        step_scale = root_scale - (
            constraint_kl - constraint_kl.detach()
        ) / constraint_slope.detach()
    else:
        step_scale = root_scale

    scaled_parameters = {
        name: parameter
        + step_scale
        * (full_step_parameters[name] - parameter)
        for name, parameter in named_parameters
    }
    return (
        scaled_parameters,
        full_step_kl,
        applied_kl,
        root_scale,
        fallback_used,
    )


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
        raise ValueError("critic-metric RawReward v5 is scoped to HalfCheetah-v4")
    if args.batch_size != 256:
        raise ValueError(f"fresh actor batch must be 256, got {args.batch_size}")
    validate_replay_config(args.replay_capacity, args.critic_batch_size, args.learning_starts)
    if not args.learning_starts <= args.actor_learning_starts <= args.replay_capacity:
        raise ValueError(
            "actor_learning_starts must be between learning_starts and replay_capacity"
        )
    if args.critic_updates_per_iteration <= 0 or args.target_update_interval <= 0:
        raise ValueError("critic update counts must be positive")
    if not 0.0 < args.meta_fraction < 1.0:
        raise ValueError("meta_fraction must leave nonempty actor and meta partitions")
    if args.metric_max_grad_norm <= 0.0:
        raise ValueError("metric_max_grad_norm must be positive")
    if args.target_actor_kl <= 0.0:
        raise ValueError("target_actor_kl must be positive")
    if args.critic_gain_disagreement_penalty < 0.0:
        raise ValueError("critic_gain_disagreement_penalty must be nonnegative")
    if not 0.0 < args.metric_identity_fraction < 1.0:
        raise ValueError("metric_identity_fraction must be in (0, 1)")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")

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

    vector_env_class = (
        gym.vector.AsyncVectorEnv
        if args.async_vector_env
        else gym.vector.SyncVectorEnv
    )
    vector_env_kwargs = (
        {"shared_memory": True, "context": "spawn"}
        if args.async_vector_env
        else {}
    )
    envs = vector_env_class(
        [
            make_env(
                args.env_id,
                index,
                args.capture_video,
                run_name,
                args.gamma,
            )
            for index in range(args.num_envs)
        ],
        **vector_env_kwargs,
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
    previous_actor = copy.deepcopy(actor).requires_grad_(False)
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(args.seed + 2_000_003)
        gradient_metric = CriticGradientMetric(
            obs_dim,
            action_dim,
            args.metric_hidden,
            args.metric_identity_fraction,
        ).to(device)
    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=args.actor_learning_rate,
        eps=1e-5,
    )
    critic_optimizer = optim.Adam(
        critic.parameters(),
        lr=args.critic_learning_rate,
        eps=1e-5,
        fused=True,
    )
    validation_critic_optimizer = optim.Adam(
        validation_critic.parameters(),
        lr=args.critic_learning_rate,
        eps=1e-5,
        fused=True,
    )
    metric_optimizer = optim.Adam(
        gradient_metric.parameters(),
        lr=args.metric_learning_rate,
        eps=1e-5,
        fused=True,
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
        device,
    )

    sample_actor = actor.sample
    target_actor_sample = target_actor.sample

    def critic_fit_value(observations, action_values):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.bf16_critic,
        ):
            return critic(observations, action_values).float()

    def validation_critic_fit_value(observations, action_values):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.bf16_critic,
        ):
            return validation_critic(observations, action_values).float()

    def target_critic_fit_value(observations, action_values):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.bf16_critic,
        ):
            return target_critic(observations, action_values).float()

    def target_validation_critic_fit_value(observations, action_values):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.bf16_critic,
        ):
            return target_validation_critic(
                observations, action_values
            ).float()

    critic_value = critic_fit_value
    validation_critic_value = validation_critic_fit_value
    target_critic_value = target_critic_fit_value
    target_validation_critic_value = target_validation_critic_fit_value
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
        print(
            f"compiled rollout and BF16={args.bf16_critic} Bellman Q functions "
            f"({args.compile_mode})"
        )

    observations = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    raw_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_observations_buffer = torch.zeros_like(observations)
    discounts = torch.zeros_like(raw_rewards)

    global_step = 0
    learner_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

    for iteration in range(1, args.num_iterations + 1):
        log_iteration = iteration % args.log_interval == 0
        for step in range(args.num_steps):
            global_step += args.num_envs
            observations[step] = next_obs
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                action, _ = sample_actor(next_obs)
            actions[step] = action

            next_obs_np, raw_reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            bootstrap_obs_np = bootstrap_observations(next_obs_np, truncations, infos)
            raw_rewards[step] = torch.as_tensor(raw_reward, dtype=torch.float32, device=device)
            next_observations_buffer[step] = torch.as_tensor(bootstrap_obs_np, dtype=torch.float32, device=device)
            discounts[step] = args.gamma * (1.0 - torch.as_tensor(terminations, dtype=torch.float32, device=device))
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

            for info in infos.get("final_info", ()):
                if info and "episode" in info:
                    episodic_return = float(info["episode"]["r"])
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        replay.add(
            observations.reshape(args.batch_size, obs_dim),
            actions.reshape(args.batch_size, action_dim),
            raw_rewards.reshape(args.batch_size),
            next_observations_buffer.reshape(args.batch_size, obs_dim),
            discounts.reshape(args.batch_size),
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
            critic.requires_grad_(True)
            validation_critic.requires_grad_(True)
            for _ in range(args.critic_updates_per_iteration):
                b_obs, b_actions, b_rewards, b_next_obs, b_discounts = replay.sample(
                    args.critic_batch_size
                )
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    next_actions, _ = target_actor_sample(b_next_obs)
                    next_actions = next_actions.clone()
                    td_target = one_step_q_target(
                        b_rewards,
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
                ) = replay.sample(args.critic_batch_size)
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    v_next_actions, _ = target_actor_sample(v_next_obs)
                    v_next_actions = v_next_actions.clone()
                    validation_td_target = one_step_q_target(
                        v_rewards,
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
                critic_losses.append(critic_loss.detach())
                critic_grad_norms.append(critic_grad_norm.detach())
                validation_critic_losses.append(
                    validation_critic_loss.detach()
                )
                validation_critic_grad_norms.append(
                    validation_critic_grad_norm.detach()
                )
                td_target_means.append(td_target.mean())
                td_target_stds.append(td_target.std(unbiased=False))
                validation_td_target_means.append(
                    validation_td_target.mean()
                )
                validation_td_target_stds.append(
                    validation_td_target.std(unbiased=False)
                )
        if critic_losses:
            critic_optimizer.zero_grad(set_to_none=True)
            validation_critic_optimizer.zero_grad(set_to_none=True)
            critic.requires_grad_(False)
            validation_critic.requires_grad_(False)

        # Fit both critics before asking either one to define actor geometry. This is a data
        # precondition, not a score/KL gate: after warmup every finite update is applied.
        if replay.size < args.actor_learning_starts:
            if not log_iteration:
                continue
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("critic/replay_size", replay.size, global_step)
            writer.add_scalar("reward/raw_mean", raw_rewards.mean().item(), global_step)
            writer.add_scalar("reward/raw_min", raw_rewards.min().item(), global_step)
            writer.add_scalar("reward/raw_max", raw_rewards.max().item(), global_step)
            continue

        b_obs = observations.reshape(args.batch_size, obs_dim)
        with torch.no_grad():
            if log_iteration:
                b_actions = actions.reshape(args.batch_size, action_dim)
                fresh_next_obs = next_observations_buffer.reshape(
                    args.batch_size, obs_dim
                )
                q_taken = critic(b_obs, b_actions)
                validation_q_taken = validation_critic(b_obs, b_actions)
                with torch.random.fork_rng(devices=[device]):
                    fresh_next_actions, _ = target_actor.sample(fresh_next_obs)
                fresh_td_target = one_step_q_target(
                    raw_rewards.reshape(args.batch_size),
                    discounts.reshape(args.batch_size),
                    target_critic_value(
                        fresh_next_obs, fresh_next_actions
                    ),
                )
                validation_fresh_td_target = one_step_q_target(
                    raw_rewards.reshape(args.batch_size),
                    discounts.reshape(args.batch_size),
                    target_validation_critic_value(
                        fresh_next_obs, fresh_next_actions
                    ),
                )
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

                # Confirm the previous actor step under a newly fitted critic and new
                # states. Sampling only on log iterations avoids diagnostic forwards
                # without changing the every-iteration actor snapshot below.
                if previous_update_available:
                    confirmation_noise = torch.randn(
                        (args.batch_size, action_dim),
                        generator=diagnostic_generator,
                        device=device,
                    )
                    current_action, _, _, _ = actor.reparameterized(
                        b_obs, confirmation_noise
                    )
                    previous_action, _, _, _ = (
                        previous_actor.reparameterized(
                            b_obs, confirmation_noise
                        )
                    )
                    next_critic_confirmed_gain = (
                        validation_critic(b_obs, current_action)
                        - validation_critic(b_obs, previous_action)
                    ).mean()
                else:
                    next_critic_confirmed_gain = torch.full(
                        (), float("nan"), device=device
                    )
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
        actor_score = critic(actor_obs, actor_action)
        (critic_action_grad,) = torch.autograd.grad(
            actor_score.sum(), actor_action, retain_graph=True
        )
        with torch.no_grad():
            validation_actor_score = validation_critic(
                actor_obs, actor_action.detach()
            )
            actor_score_disagreement = (
                actor_score.detach() - validation_actor_score
            ).abs().mean()
        normalized_actor_action = (
            actor_action - actor.action_bias
        ) / actor.action_scale
        actor_loss, metric_outputs = gradient_metric(
            actor_obs,
            actor_mean,
            actor_logstd,
            actor_action,
            normalized_actor_action,
            actor_score,
            critic_action_grad,
        )
        if not bool(
            torch.isfinite(actor_score).all()
            and torch.isfinite(critic_action_grad).all()
            and torch.isfinite(actor_loss)
            and torch.isfinite(metric_outputs["metric"]).all()
            and torch.isfinite(metric_outputs["action_direction"]).all()
        ):
            raise FloatingPointError("non-finite critic-metric actor objective")
        if bool((metric_outputs["ascent_dot"] < -1e-6).any()):
            raise RuntimeError("critic metric produced a non-ascent action direction")

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

        (
            shadow_parameters,
            full_step_actor_kl,
            applied_actor_kl,
            actor_step_scale,
            actor_cap_fallback,
        ) = policy_normalized_adam_step(
            actor,
            actor_named_parameters,
            clipped_actor_gradients,
            actor_optimizer,
            b_obs,
            args.target_actor_kl,
        )
        (
            direct_shadow_parameters,
            direct_full_step_actor_kl,
            direct_applied_actor_kl,
            direct_actor_step_scale,
            direct_cap_fallback,
        ) = policy_normalized_adam_step(
            actor,
            actor_named_parameters,
            clipped_direct_gradients,
            actor_optimizer,
            b_obs,
            args.target_actor_kl,
            differentiable_scale=False,
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
            validation_meta_score_before = validation_critic(
                meta_obs, meta_action_before
            ).mean()
            training_meta_score_before = critic(
                meta_obs, meta_action_before
            ).mean()
        shadow_mean, shadow_logstd = functional_call(
            actor, shadow_parameters, (meta_obs,)
        )
        shadow_raw_action = shadow_mean + shadow_logstd.exp() * meta_noise
        shadow_action = actor.transform(shadow_raw_action)
        validation_meta_score_after = validation_critic(
            meta_obs, shadow_action
        ).mean()
        training_meta_score_after = critic(
            meta_obs, shadow_action
        ).mean()
        validation_meta_gain = (
            validation_meta_score_after - validation_meta_score_before
        )
        training_meta_gain = (
            training_meta_score_after - training_meta_score_before
        )
        meta_gain_disagreement = (
            validation_meta_gain - training_meta_gain
        ).abs()
        conservative_meta_gain = (
            validation_meta_gain
            - args.critic_gain_disagreement_penalty
            * meta_gain_disagreement
        )
        meta_loss = -conservative_meta_gain

        with torch.no_grad():
            direct_shadow_mean, direct_shadow_logstd = functional_call(
                actor, direct_shadow_parameters, (meta_obs,)
            )
            direct_shadow_raw_action = (
                direct_shadow_mean
                + direct_shadow_logstd.exp() * meta_noise
            )
            direct_shadow_action = actor.transform(direct_shadow_raw_action)
            direct_validation_score_after = validation_critic(
                meta_obs, direct_shadow_action
            ).mean()
            direct_training_score_after = critic(
                meta_obs, direct_shadow_action
            ).mean()
            direct_validation_gain = (
                direct_validation_score_after
                - validation_meta_score_before
            )
            direct_training_gain = (
                direct_training_score_after
                - training_meta_score_before
            )
            direct_gain_disagreement = (
                direct_validation_gain - direct_training_gain
            ).abs()
            direct_conservative_gain = (
                direct_validation_gain
                - args.critic_gain_disagreement_penalty
                * direct_gain_disagreement
            )
            shadow_critic_disagreement = (
                training_meta_score_after.detach()
                - validation_meta_score_after.detach()
            ).abs()
            if not bool(
                torch.isfinite(conservative_meta_gain)
                and torch.isfinite(direct_conservative_gain)
                and torch.isfinite(shadow_critic_disagreement)
            ):
                raise FloatingPointError("non-finite conservative critic gain")

        metric_gradients = torch.autograd.grad(
            meta_loss,
            tuple(gradient_metric.parameters()),
            allow_unused=False,
        )
        if not all(
            bool(torch.isfinite(gradient).all())
            for gradient in metric_gradients
        ):
            raise FloatingPointError("non-finite critic-metric meta gradient")
        metric_optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(
            gradient_metric.parameters(),
            metric_gradients,
            strict=True,
        ):
            parameter.grad = gradient.detach()
        meta_grad_norm = nn.utils.clip_grad_norm_(
            gradient_metric.parameters(),
            args.metric_max_grad_norm,
            error_if_nonfinite=True,
        )
        metric_optimizer.step()

        # Adam's moments advance on the full gradient, but the parameter displacement is
        # continuously normalized in policy space. Every direction is applied; none is
        # admitted or rejected by KL or predicted gain.
        actor_optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(
            actor_parameters, clipped_actor_gradients, strict=True
        ):
            parameter.grad = gradient.detach()
        actor_optimizer.step()
        with torch.no_grad():
            for name, parameter in actor_named_parameters:
                parameter.copy_(shadow_parameters[name].detach())
            shadow_commit_error = torch.stack(
                [
                    (parameter - shadow_parameters[name]).abs().max()
                    for name, parameter in actor_named_parameters
                ]
            ).max()
            if shadow_commit_error > 1e-6:
                raise RuntimeError(
                    "committed actor does not match scored normalized Adam shadow"
                )
        previous_update_available = True
        if not log_iteration:
            continue
        with torch.no_grad():
            _, logstd = actor(b_obs)
            metric_eigenvalues = torch.linalg.eigvalsh(
                metric_outputs["metric"].detach()
            )
            metric_trace = metric_outputs["metric"].detach().diagonal(
                dim1=-2, dim2=-1
            ).sum(dim=-1)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar(
            "charts/actor_learning_rate",
            args.actor_learning_rate,
            global_step,
        )
        writer.add_scalar(
            "charts/critic_learning_rate",
            args.critic_learning_rate,
            global_step,
        )
        writer.add_scalar(
            "charts/metric_learning_rate",
            args.metric_learning_rate,
            global_step,
        )
        writer.add_scalar("losses/actor_metric", actor_loss.item(), global_step)
        writer.add_scalar("losses/meta_objective", meta_loss.item(), global_step)
        writer.add_scalar(
            "losses/critic_mse",
            2.0 * torch.stack(critic_losses).mean().item(),
            global_step,
        )
        writer.add_scalar(
            "losses/validation_critic_mse",
            2.0 * torch.stack(validation_critic_losses).mean().item(),
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
        writer.add_scalar(
            "losses/critic_grad_norm",
            torch.stack(critic_grad_norms).mean().item(),
            global_step,
        )
        writer.add_scalar(
            "losses/validation_critic_grad_norm",
            torch.stack(validation_critic_grad_norms).mean().item(),
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
        writer.add_scalar(
            "critic/td_target_mean",
            torch.stack(td_target_means).mean().item(),
            global_step,
        )
        writer.add_scalar(
            "critic/td_target_std",
            torch.stack(td_target_stds).mean().item(),
            global_step,
        )
        writer.add_scalar(
            "critic/validation_td_target_mean",
            torch.stack(validation_td_target_means).mean().item(),
            global_step,
        )
        writer.add_scalar(
            "critic/validation_td_target_std",
            torch.stack(validation_td_target_stds).mean().item(),
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
            "meta/validation_score_before",
            validation_meta_score_before.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/validation_score_after",
            validation_meta_score_after.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/training_score_before",
            training_meta_score_before.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/training_score_after",
            training_meta_score_after.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/validation_gain", validation_meta_gain.item(), global_step
        )
        writer.add_scalar(
            "meta/training_gain", training_meta_gain.item(), global_step
        )
        writer.add_scalar(
            "meta/gain_disagreement",
            meta_gain_disagreement.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/conservative_gain",
            conservative_meta_gain.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/direct_validation_gain",
            direct_validation_gain.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/direct_training_gain",
            direct_training_gain.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/direct_gain_disagreement",
            direct_gain_disagreement.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/direct_conservative_gain",
            direct_conservative_gain.item(),
            global_step,
        )
        writer.add_scalar(
            "meta/learned_minus_direct_conservative_gain",
            (
                conservative_meta_gain.detach()
                - direct_conservative_gain
            ).item(),
            global_step,
        )
        writer.add_scalar(
            "meta/next_critic_confirmed_gain",
            next_critic_confirmed_gain.item(),
            global_step,
        )
        writer.add_scalar(
            "policy_step/full_adam_kl",
            full_step_actor_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "policy_step/applied_kl",
            applied_actor_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "policy_step/scale", actor_step_scale.item(), global_step
        )
        writer.add_scalar(
            "policy_step/cap_fallback",
            float(actor_cap_fallback),
            global_step,
        )
        writer.add_scalar(
            "policy_step/direct_full_adam_kl",
            direct_full_step_actor_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "policy_step/direct_applied_kl",
            direct_applied_actor_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "policy_step/direct_scale",
            direct_actor_step_scale.item(),
            global_step,
        )
        writer.add_scalar(
            "policy_step/direct_cap_fallback",
            float(direct_cap_fallback),
            global_step,
        )
        writer.add_scalar(
            "meta/actor_update_applied", 1.0, global_step
        )
        writer.add_scalar(
            "meta/meta_update_applied", 1.0, global_step
        )
        writer.add_scalar(
            "meta/shadow_commit_max_error",
            shadow_commit_error.item(),
            global_step,
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
            "metric/action_alignment_mean",
            metric_outputs["action_alignment"].detach().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "metric/action_alignment_min",
            metric_outputs["action_alignment"].detach().min().item(),
            global_step,
        )
        writer.add_scalar(
            "metric/ascent_dot_mean",
            metric_outputs["ascent_dot"].detach().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "metric/ascent_dot_min",
            metric_outputs["ascent_dot"].detach().min().item(),
            global_step,
        )
        writer.add_scalar(
            "metric/action_direction_rms",
            metric_outputs["action_direction"]
            .detach()
            .square()
            .mean()
            .sqrt()
            .item(),
            global_step,
        )
        writer.add_scalar(
            "metric/eigenvalue_min",
            metric_eigenvalues.min().item(),
            global_step,
        )
        writer.add_scalar(
            "metric/eigenvalue_mean",
            metric_eigenvalues.mean().item(),
            global_step,
        )
        writer.add_scalar(
            "metric/eigenvalue_max",
            metric_eigenvalues.max().item(),
            global_step,
        )
        writer.add_scalar(
            "metric/condition_max",
            (
                metric_eigenvalues[:, -1]
                / metric_eigenvalues[:, 0].clamp_min(1e-8)
            ).max().item(),
            global_step,
        )
        writer.add_scalar(
            "metric/trace_mean", metric_trace.mean().item(), global_step
        )
        writer.add_scalar(
            "metric/trace_max_error",
            (metric_trace - action_dim).abs().max().item(),
            global_step,
        )
        writer.add_scalar("reward/raw_mean", raw_rewards.mean().item(), global_step)
        writer.add_scalar("reward/raw_min", raw_rewards.min().item(), global_step)
        writer.add_scalar("reward/raw_max", raw_rewards.max().item(), global_step)
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

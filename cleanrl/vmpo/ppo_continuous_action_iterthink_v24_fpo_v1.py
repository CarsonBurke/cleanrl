# FPO v1 — idiomatic Flow Policy Optimization on the v30 IterThink stack.
#
# McAllister et al. 2025, playground defaults: reverse-time OT flow (t=1 noise
# to t=0 action), velocity network, epsilon-supervised CFM, likelihood-ratio
# surrogate exp(mean L_old - mean L_new), PPO clip 0.05, 10 Euler steps, 8 CFM
# samples per action, 4 full-batch reuse epochs. Sampling is a black-box ODE;
# the CFM (t, eps) pairs are independent of the Euler path.
#
# Kept from v30: IterThink trunk, Dreamer moment-matched HL-Gauss critic,
# per-env observation and reward normalization, GAE, percentile advantage
# scale, phase staggering, BF16 compiled full-graph training. Dropped: Beta
# carrier, V-MPO duals, top-k/eta, and the lagged behavior target. Rollouts
# use the online flow policy.

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport


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

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    num_envs: int = 64
    num_steps: int = 39
    gamma: float = 0.99
    gae_lambda: float = 0.95

    flow_steps: int = 10
    n_cfm_samples: int = 8
    timestep_embed_dim: int = 8
    policy_mlp_output_scale: float = 0.25
    clip_coef: float = 0.05
    update_epochs: int = 4

    return_percentile_low: float = 0.05
    return_percentile_high: float = 0.95
    return_percentile_floor: float = 1.0
    num_value_bins: int = 51
    value_support_limit: float = 20_000.0
    value_sigma_to_bin_ratio: float = 0.75

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    compile: bool = True
    compile_mode: str = "reduce-overhead"
    bf16: bool = True
    log_interval: int = 10

    batch_size: int = 0
    num_iterations: int = 0
    initial_phase_warmup_steps: int = 0


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def normalize_observations(
    observations,
    observation_means,
    observation_variances,
    observation_counts,
    env_indices=None,
):
    """Update independent singleton moments and return clipped normalization."""
    observations = np.asarray(observations)
    if env_indices is None:
        means = observation_means
        variances = observation_variances
        counts = observation_counts
    else:
        means = observation_means[env_indices]
        variances = observation_variances[env_indices]
        counts = observation_counts[env_indices]

    count_axes = (slice(None),) + (None,) * (observations.ndim - 1)
    expanded_counts = counts[count_axes]
    total_counts = counts + 1.0
    expanded_total_counts = total_counts[count_axes]
    delta = observations - means
    new_means = means + delta / expanded_total_counts
    new_variances = (
        variances * expanded_counts
        + np.square(delta) * expanded_counts / expanded_total_counts
    ) / expanded_total_counts

    if env_indices is None:
        observation_means[...] = new_means
        observation_variances[...] = new_variances
        observation_counts[...] = total_counts
    else:
        observation_means[env_indices] = new_means
        observation_variances[env_indices] = new_variances
        observation_counts[env_indices] = total_counts

    return np.clip(
        (observations - new_means) / np.sqrt(new_variances + 1e-8),
        -10,
        10,
    )


def normalize_rewards(
    rewards,
    terminations,
    discounted_returns,
    return_means,
    return_variances,
    return_counts,
    gamma,
):
    """Match independent NormalizeReward wrappers in one vectorized update."""
    raw_rewards = np.asarray(rewards, dtype=np.float64)
    termination_mask = np.asarray(terminations, dtype=np.float64)
    discounted_returns *= gamma * (1.0 - termination_mask)
    discounted_returns += raw_rewards

    total_counts = return_counts + 1.0
    delta = discounted_returns - return_means
    new_means = return_means + delta / total_counts
    new_variances = (
        return_variances * return_counts
        + np.square(delta) * return_counts / total_counts
    ) / total_counts
    return_means[...] = new_means
    return_variances[...] = new_variances
    return_counts[...] = total_counts

    return np.clip(
        raw_rewards / np.sqrt(new_variances + 1e-8),
        -10.0,
        10.0,
    )


def normalize_vector_step(
    raw_next_observations,
    terminations,
    truncations,
    infos,
    observation_means,
    observation_variances,
    observation_counts,
):
    """Normalize a same-step autoreset batch in per-environment wrapper order."""
    boundaries = np.logical_or(terminations, truncations)
    boundary_indices = np.flatnonzero(boundaries)
    raw_transition_observations = np.array(raw_next_observations, copy=True)
    if boundary_indices.size:
        final_observations = infos.get("final_observation")
        final_mask = infos.get("_final_observation")
        if final_observations is None:
            raise RuntimeError("completed transition missing final_observation")
        for env_index in boundary_indices:
            if final_mask is not None and not final_mask[env_index]:
                raise RuntimeError(
                    f"completed environment {env_index} has no final observation"
                )
            final_observation = final_observations[env_index]
            if final_observation is None:
                raise RuntimeError(
                    f"completed environment {env_index} has no final observation"
                )
            raw_transition_observations[env_index] = final_observation

    normalized_transition_observations = normalize_observations(
        raw_transition_observations,
        observation_means,
        observation_variances,
        observation_counts,
    )
    normalized_next_observations = np.array(
        normalized_transition_observations, copy=True
    )
    if boundary_indices.size:
        normalized_next_observations[boundary_indices] = normalize_observations(
            raw_next_observations[boundary_indices],
            observation_means,
            observation_variances,
            observation_counts,
            boundary_indices,
        )
    return normalized_next_observations, normalized_transition_observations


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


def branch_body(hidden):
    return nn.Sequential(
        layer_init(nn.Linear(hidden, hidden)),
        ReLUSquared(),
        layer_init(nn.Linear(hidden, hidden)),
    )


class FusedExperts(nn.Module):
    """Equivalent expert MLPs stored as two batched GEMMs."""

    def __init__(self, n_experts, hidden):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(n_experts, hidden, hidden))
        self.bias1 = nn.Parameter(torch.zeros(n_experts, hidden))
        self.weight2 = nn.Parameter(torch.empty(n_experts, hidden, hidden))
        self.bias2 = nn.Parameter(torch.zeros(n_experts, hidden))
        with torch.no_grad():
            for expert_index in range(n_experts):
                nn.init.orthogonal_(self.weight1[expert_index], np.sqrt(2))
                nn.init.orthogonal_(self.weight2[expert_index], np.sqrt(2))

    def forward(self, x):
        hidden = torch.einsum("bi,eoi->beo", x, self.weight1) + self.bias1
        hidden = torch.relu(hidden).square()
        return torch.einsum("bei,eoi->beo", hidden, self.weight2) + self.bias2


class ThinkBlock(nn.Module):
    def __init__(self, in_dim, hidden, n_experts):
        super().__init__()
        self.in_proj = layer_init(nn.Linear(in_dim, hidden))
        self.resid_gate = nn.Parameter(torch.full((hidden,), 4.0))
        self.dense_norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.dense = branch_body(hidden)
        self.moe_norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(hidden, n_experts))
        self.experts = FusedExperts(n_experts, hidden)

    def forward(self, features, x0):
        x = self.in_proj(features)
        gate = torch.sigmoid(self.resid_gate)
        residual = gate * x + (1.0 - gate) * x0
        dense = self.dense(self.dense_norm(residual))
        moe_input = self.moe_norm(residual)
        weights = torch.softmax(self.gate(moe_input), dim=-1)
        expert_outputs = self.experts(moe_input)
        moe = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return residual + dense + moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, hidden, k_blocks, n_experts):
        super().__init__()
        self.output_dim = hidden * (k_blocks + 1)
        self.entry = layer_init(nn.Linear(in_dim, hidden))
        self.blocks = nn.ModuleList(
            [ThinkBlock(hidden * (index + 1), hidden, n_experts) for index in range(k_blocks)]
        )
        self.out_norm = nn.RMSNorm(self.output_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(self.output_dim, self.output_dim))

    def forward(self, x):
        x0 = self.entry(x)
        features = [x0]
        for block in self.blocks:
            features.append(block(torch.cat(features, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(features, dim=-1)))


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor
    flow_t_current: torch.Tensor
    flow_t_next: torch.Tensor
    time_freqs: torch.Tensor

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.flow_steps = args.flow_steps
        self.n_cfm_samples = args.n_cfm_samples
        self.output_scale = args.policy_mlp_output_scale
        self.trunk = ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
        self.policy_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        velocity_in = 256 + action_dim + args.timestep_embed_dim
        self.velocity_hidden = nn.Sequential(
            layer_init(nn.Linear(velocity_in, 256)),
            ReLUSquared(),
        )
        self.velocity_out = layer_init(nn.Linear(256, action_dim), std=1.0)
        self.value_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        self.value_head = nn.Linear(256, args.num_value_bins, bias=False)
        with torch.no_grad():
            self.value_head.weight.zero_()
        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )
        t_path = torch.linspace(1.0, 0.0, args.flow_steps + 1)
        self.register_buffer("flow_t_current", t_path[:-1])
        self.register_buffer("flow_t_next", t_path[1:])
        self.register_buffer(
            "time_freqs",
            (2 ** torch.arange(args.timestep_embed_dim // 2)).float(),
        )

    def encode(self, observations):
        features = self.trunk(observations)
        policy_features = self.policy_mlp(features)
        value_logits = self.value_head(self.value_mlp(features))
        return policy_features, value_logits

    def value_logits(self, observations):
        return self.value_head(self.value_mlp(self.trunk(observations)))

    def embed_time(self, t):
        scaled = t * self.time_freqs.to(dtype=t.dtype)
        return torch.cat([scaled.cos(), scaled.sin()], dim=-1)

    def velocity(self, policy_features, x_t, t):
        dtype = policy_features.dtype
        hidden = self.velocity_hidden(
            torch.cat(
                [
                    policy_features,
                    x_t.to(dtype=dtype),
                    self.embed_time(t.to(dtype=dtype)),
                ],
                dim=-1,
            )
        )
        return self.velocity_out(hidden) * self.output_scale

    def sample_actions(self, policy_features, noise):
        x_t = noise
        for step in range(self.flow_steps):
            t = self.flow_t_current[step].to(dtype=x_t.dtype).expand(x_t.shape[0], 1)
            dt = (self.flow_t_next[step] - self.flow_t_current[step]).to(
                dtype=x_t.dtype
            )
            x_t = x_t + dt * self.velocity(policy_features, x_t, t)
        return x_t

    def cfm_losses(self, policy_features, actions, eps, t):
        """Epsilon-supervised CFM on the reverse-time OT path, shape (B, N)."""
        dtype = policy_features.dtype
        actions = actions.to(dtype=dtype)
        eps = eps.to(dtype=dtype)
        t = t.to(dtype=dtype)
        sample_count = eps.shape[1]
        x_t = t * eps + (1.0 - t) * actions.unsqueeze(1)
        flat_features = (
            policy_features.unsqueeze(1)
            .expand(-1, sample_count, -1)
            .reshape(-1, policy_features.shape[-1])
        )
        velocity = self.velocity(
            flat_features,
            x_t.reshape(-1, self.action_dim),
            t.reshape(-1, 1),
        ).view_as(eps)
        x0_pred = x_t - t * velocity
        x1_pred = x0_pred + velocity
        return (eps - x1_pred).square().mean(dim=-1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    env_spec = gym.spec(args.env_id)
    if env_spec.max_episode_steps is None:
        raise ValueError("phase staggering requires a finite episode horizon")
    args.initial_phase_warmup_steps = int(env_spec.max_episode_steps)
    warmup_transitions = args.num_envs * args.initial_phase_warmup_steps
    if warmup_transitions >= args.total_timesteps:
        raise ValueError("total_timesteps must exceed the initial phase warmup")
    args.num_iterations = (
        args.total_timesteps - warmup_transitions
    ) // args.batch_size
    if args.num_steps != 39:
        raise ValueError("OpenAI Gym paper alignment requires a 39-step unroll")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if args.flow_steps < 1:
        raise ValueError("flow_steps must be positive")
    if args.n_cfm_samples < 1:
        raise ValueError("n_cfm_samples must be positive")
    if args.timestep_embed_dim < 2 or args.timestep_embed_dim % 2 != 0:
        raise ValueError("timestep_embed_dim must be even and at least 2")
    if args.clip_coef <= 0.0:
        raise ValueError("clip_coef must be positive")
    if args.update_epochs < 1:
        raise ValueError("update_epochs must be positive")
    if args.policy_mlp_output_scale <= 0.0:
        raise ValueError("policy_mlp_output_scale must be positive")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.bf16 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 support is required")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if not 0.0 <= args.return_percentile_low < args.return_percentile_high <= 1.0:
        raise ValueError("return percentiles must satisfy 0 <= low < high <= 1")
    if args.return_percentile_floor <= 0.0:
        raise ValueError("return_percentile_floor must be positive")
    if args.num_value_bins < 3 or args.num_value_bins % 2 == 0:
        raise ValueError("num_value_bins must be odd and at least three")
    if args.value_support_limit <= 0.0:
        raise ValueError("value_support_limit must be positive")
    if args.value_sigma_to_bin_ratio <= 0.0:
        raise ValueError("value_sigma_to_bin_ratio must be positive")

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
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    return_percentile_levels = torch.tensor(
        [args.return_percentile_low, args.return_percentile_high],
        device=device,
    )
    symlog_limit = float(np.log1p(args.value_support_limit))
    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_value_bins,
        -symlog_limit,
        symlog_limit,
        args.value_sigma_to_bin_ratio,
        device,
    )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, index, args.capture_video, run_name)
            for index in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("FPO continuous control requires a Box action space")
    if envs.envs[0].spec is None or (
        envs.envs[0].spec.max_episode_steps != args.initial_phase_warmup_steps
    ):
        raise RuntimeError("constructed environment horizon differs from gym spec")

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(
        agent.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )

    autocast_dtype = torch.bfloat16

    def rollout_model(observations):
        batch = observations.shape[0]
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            policy_features, value_logits = agent.encode(observations)
            sample_noise = torch.randn(
                batch, agent.action_dim, device=observations.device, dtype=policy_features.dtype
            )
            actions = agent.sample_actions(policy_features, sample_noise)
            cfm_eps = torch.randn(
                batch,
                args.n_cfm_samples,
                agent.action_dim,
                device=observations.device,
                dtype=policy_features.dtype,
            )
            t_index = torch.randint(
                0,
                args.flow_steps,
                (batch, args.n_cfm_samples),
                device=observations.device,
            )
            cfm_t = agent.flow_t_current[t_index].unsqueeze(-1)
            initial_cfm = agent.cfm_losses(
                policy_features, actions, cfm_eps, cfm_t
            )
        values = hl_support.to_scalar(value_logits.float())
        return (
            actions.float(),
            values,
            cfm_eps.float(),
            cfm_t.float(),
            initial_cfm.float(),
        )

    def gae_model(
        transition_next_observations,
        reward_batch,
        value_batch,
        terminations,
        boundaries,
    ):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            next_value_logits = agent.value_logits(
                transition_next_observations.reshape(
                    (args.batch_size,) + transition_next_observations.shape[2:]
                )
            )
        next_values = hl_support.to_scalar(next_value_logits.float()).view(
            args.num_steps, args.num_envs
        )
        advantages = torch.empty_like(reward_batch)
        running_advantage = torch.zeros_like(next_values[-1])
        for reverse_step in reversed(range(args.num_steps)):
            delta = (
                reward_batch[reverse_step]
                + args.gamma
                * next_values[reverse_step]
                * (1.0 - terminations[reverse_step])
                - value_batch[reverse_step]
            )
            continuing_advantage = (
                delta
                + args.gamma
                * args.gae_lambda
                * running_advantage
            )
            running_advantage = torch.where(
                boundaries[reverse_step],
                delta,
                continuing_advantage,
            )
            advantages[reverse_step] = running_advantage
        returns = advantages + value_batch
        return advantages, returns

    def update_loss_model(
        observations,
        actions,
        cfm_eps,
        cfm_t,
        initial_cfm,
        advantages,
        value_targets,
    ):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            policy_features, value_logits = agent.encode(observations)
            new_cfm = agent.cfm_losses(policy_features, actions, cfm_eps, cfm_t)
        value_log_probs = F.log_softmax(value_logits.float(), dim=-1)
        value_probs = value_log_probs.exp()
        values = hl_support.probs_to_scalar(value_probs)
        new_cfm = new_cfm.float()
        ratio = (
            initial_cfm.mean(dim=-1) - new_cfm.mean(dim=-1)
        ).exp()
        surrogate_unclipped = ratio * advantages
        surrogate_clipped = ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef) * advantages
        policy_loss = -torch.min(surrogate_unclipped, surrogate_clipped).mean()
        with torch.no_grad():
            value_target_probs = hl_support.project_moment_matched(value_targets)
        value_loss = -(value_target_probs * value_log_probs).sum(dim=-1).mean()
        total_loss = policy_loss + value_loss

        clip_fraction = (ratio - 1.0).abs().gt(args.clip_coef).float().mean()
        cfm_old = initial_cfm.mean()
        cfm_new = new_cfm.mean()
        value_error = values - value_targets
        value_rmse = value_error.square().mean().sqrt()
        explained_variance = 1.0 - value_error.var(unbiased=False) / (
            value_targets.var(unbiased=False) + 1e-8
        )
        target_outside_support = (
            value_targets.abs() > args.value_support_limit
        ).float().mean()
        target_edge_mass = (
            value_target_probs[:, 0] + value_target_probs[:, -1]
        ).mean()
        prediction_edge_mass = (
            value_probs[:, 0] + value_probs[:, -1]
        ).mean()
        metrics = torch.stack(
            (
                policy_loss.detach(),
                value_loss.detach(),
                ratio.mean().detach(),
                ratio.min().detach(),
                ratio.max().detach(),
                clip_fraction.detach(),
                cfm_old.detach(),
                cfm_new.detach(),
                (cfm_old - cfm_new).detach(),
                advantages.mean().detach(),
                advantages.std().detach(),
                actions.mean().detach(),
                actions.std().detach(),
                actions.min().detach(),
                actions.max().detach(),
                value_rmse.detach(),
                explained_variance.detach(),
                target_outside_support.detach(),
                target_edge_mass.detach(),
                prediction_edge_mass.detach(),
            )
        )
        return total_loss, metrics

    if args.compile:
        rollout_model = torch.compile(
            rollout_model,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        gae_model = torch.compile(
            gae_model,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        update_loss_model = torch.compile(
            update_loss_model,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        print(f"compiled BF16 fullgraph training paths ({args.compile_mode})")

    if (
        envs.single_observation_space.shape is None
        or envs.single_action_space.shape is None
    ):
        raise ValueError("continuous-control observation and action shapes are required")
    observation_shape = tuple(envs.single_observation_space.shape)
    action_shape = tuple(envs.single_action_space.shape)
    observation_means = np.zeros(
        (args.num_envs,) + observation_shape, dtype=np.float64
    )
    observation_variances = np.ones_like(observation_means)
    observation_counts = np.full(args.num_envs, 1e-4, dtype=np.float64)
    discounted_returns = np.zeros(args.num_envs, dtype=np.float64)
    reward_return_means = np.zeros(args.num_envs, dtype=np.float64)
    reward_return_variances = np.ones(args.num_envs, dtype=np.float64)
    reward_return_counts = np.full(args.num_envs, 1e-4, dtype=np.float64)
    rollout_shape = (args.num_steps, args.num_envs)
    observations = torch.empty(rollout_shape + observation_shape, device=device)
    actions = torch.empty(rollout_shape + action_shape, device=device)
    rewards = torch.empty(rollout_shape, device=device)
    values = torch.empty(rollout_shape, device=device)
    cfm_eps_buffer = torch.empty(
        rollout_shape + (args.n_cfm_samples,) + action_shape, device=device
    )
    cfm_t_buffer = torch.empty(
        rollout_shape + (args.n_cfm_samples, 1), device=device
    )
    initial_cfm_buffer = torch.empty(
        rollout_shape + (args.n_cfm_samples,), device=device
    )
    next_observations = torch.empty_like(observations)
    terminations_buffer = torch.empty(rollout_shape, device=device)
    boundaries_buffer = torch.empty_like(terminations_buffer, dtype=torch.bool)

    global_step = 0
    start_time = time.time()
    raw_next_observations, _ = envs.reset(seed=args.seed)
    next_observation_np = normalize_observations(
        raw_next_observations,
        observation_means,
        observation_variances,
        observation_counts,
    )

    phase_offsets = (
        np.arange(args.num_envs, dtype=np.int64)
        * args.initial_phase_warmup_steps
        // args.num_envs
    )
    phase_rng = np.random.default_rng(args.seed)
    phase_rng.shuffle(phase_offsets)
    writer.add_text(
        "initial_phase_offsets",
        ",".join(str(offset) for offset in phase_offsets),
    )
    scheduled_resets = args.initial_phase_warmup_steps - phase_offsets

    for warmup_step in range(1, args.initial_phase_warmup_steps + 1):
        warmup_observation = torch.as_tensor(
            next_observation_np,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            warmup_action, _, _, _, _ = rollout_model(warmup_observation)
            warmup_env_action = warmup_action.clamp(agent.action_low, agent.action_high)
        (
            raw_next_observations,
            warmup_reward,
            warmup_terminations,
            warmup_truncations,
            warmup_infos,
        ) = envs.step(warmup_env_action.cpu().numpy())
        normalize_rewards(
            warmup_reward,
            warmup_terminations,
            discounted_returns,
            reward_return_means,
            reward_return_variances,
            reward_return_counts,
            args.gamma,
        )
        next_observation_np, _ = normalize_vector_step(
            raw_next_observations,
            warmup_terminations,
            warmup_truncations,
            warmup_infos,
            observation_means,
            observation_variances,
            observation_counts,
        )

        for env_index in np.flatnonzero(scheduled_resets == warmup_step):
            reset_observation, _ = envs.envs[env_index].reset()
            next_observation_np[env_index] = normalize_observations(
                reset_observation[None, ...],
                observation_means,
                observation_variances,
                observation_counts,
                slice(env_index, env_index + 1),
            )[0]

    suppress_next_episode_log = phase_offsets > 0

    global_step = warmup_transitions
    next_observation = torch.as_tensor(
        next_observation_np,
        device=device,
        dtype=torch.float32,
    )

    for iteration in range(1, args.num_iterations + 1):
        for step in range(args.num_steps):
            observations[step].copy_(next_observation)
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                action, value, cfm_eps, cfm_t, initial_cfm = rollout_model(
                    next_observation
                )
                env_action = action.clamp(agent.action_low, agent.action_high)
            actions[step].copy_(action)
            values[step].copy_(value)
            cfm_eps_buffer[step].copy_(cfm_eps)
            cfm_t_buffer[step].copy_(cfm_t)
            initial_cfm_buffer[step].copy_(initial_cfm)

            (
                raw_next_observations,
                raw_reward,
                terminations,
                truncations,
                infos,
            ) = envs.step(env_action.cpu().numpy())
            reward = normalize_rewards(
                raw_reward,
                terminations,
                discounted_returns,
                reward_return_means,
                reward_return_variances,
                reward_return_counts,
                args.gamma,
            )
            boundary = np.logical_or(terminations, truncations)
            next_observation_np, normalized_transition_observations = (
                normalize_vector_step(
                    raw_next_observations,
                    terminations,
                    truncations,
                    infos,
                    observation_means,
                    observation_variances,
                    observation_counts,
                )
            )
            transition_next_observation = np.array(
                next_observation_np, copy=True
            )
            transition_next_observation[truncations] = (
                normalized_transition_observations[truncations]
            )

            rewards[step].copy_(
                torch.as_tensor(reward, device=device, dtype=torch.float32)
            )
            terminations_buffer[step].copy_(
                torch.as_tensor(terminations, device=device, dtype=torch.float32)
            )
            boundaries_buffer[step].copy_(
                torch.as_tensor(boundary, device=device, dtype=torch.bool)
            )
            next_observations[step].copy_(
                torch.as_tensor(
                    transition_next_observation, device=device, dtype=torch.float32
                )
            )
            next_observation = torch.as_tensor(
                next_observation_np, device=device, dtype=torch.float32
            )
            global_step += args.num_envs

            if "final_info" in infos:
                for env_index, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        if suppress_next_episode_log[env_index]:
                            suppress_next_episode_log[env_index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        episode_length = float(info["episode"]["l"])
                        print(
                            f"global_step={global_step}, episodic_return={episode_return}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", episode_return, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", episode_length, global_step
                        )

        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            gae_advantages, returns = gae_model(
                next_observations,
                rewards,
                values,
                terminations_buffer,
                boundaries_buffer,
            )
            advantages = gae_advantages.reshape(-1).detach().clone()
            value_targets = returns.reshape(-1).detach().clone()
            return_percentiles = torch.quantile(
                value_targets, return_percentile_levels
            )
            return_percentile_scale = (
                return_percentiles[1] - return_percentiles[0]
            ).clamp_min(args.return_percentile_floor)
            advantages.div_(return_percentile_scale)
            del gae_advantages, returns

        flat_observations = observations.reshape(
            (args.batch_size,) + observation_shape
        )
        flat_actions = actions.reshape((args.batch_size,) + action_shape)
        flat_cfm_eps = cfm_eps_buffer.reshape(
            args.batch_size, args.n_cfm_samples, agent.action_dim
        )
        flat_cfm_t = cfm_t_buffer.reshape(args.batch_size, args.n_cfm_samples, 1)
        flat_initial_cfm = initial_cfm_buffer.reshape(
            args.batch_size, args.n_cfm_samples
        )

        should_log = iteration % args.log_interval == 0 or iteration == 1
        metrics = None
        for epoch in range(args.update_epochs):
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            total_loss, metrics = update_loss_model(
                flat_observations,
                flat_actions,
                flat_cfm_eps,
                flat_cfm_t,
                flat_initial_cfm,
                advantages,
                value_targets,
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

        if should_log:
            assert metrics is not None
            packed = torch.cat(
                (
                    metrics,
                    return_percentile_scale.view(1),
                )
            ).cpu().tolist()
            (
                policy_loss_value,
                value_loss_value,
                ratio_mean_value,
                ratio_min_value,
                ratio_max_value,
                clip_fraction_value,
                cfm_old_value,
                cfm_new_value,
                cfm_difference_value,
                advantage_mean,
                advantage_std,
                action_mean_value,
                action_std_value,
                action_min_value,
                action_max_value,
                value_rmse_value,
                explained_variance_value,
                target_outside_support_value,
                target_edge_mass_value,
                prediction_edge_mass_value,
                return_percentile_scale_value,
            ) = packed
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/policy_loss", policy_loss_value, global_step)
            writer.add_scalar("losses/value_loss", value_loss_value, global_step)
            writer.add_scalar("fpo/ratio_mean", ratio_mean_value, global_step)
            writer.add_scalar("fpo/ratio_min", ratio_min_value, global_step)
            writer.add_scalar("fpo/ratio_max", ratio_max_value, global_step)
            writer.add_scalar("fpo/clip_fraction", clip_fraction_value, global_step)
            writer.add_scalar("fpo/cfm_old", cfm_old_value, global_step)
            writer.add_scalar("fpo/cfm_new", cfm_new_value, global_step)
            writer.add_scalar("fpo/cfm_difference", cfm_difference_value, global_step)
            writer.add_scalar("fpo/action_mean", action_mean_value, global_step)
            writer.add_scalar("fpo/action_std", action_std_value, global_step)
            writer.add_scalar("fpo/action_min", action_min_value, global_step)
            writer.add_scalar("fpo/action_max", action_max_value, global_step)
            writer.add_scalar("fpo/learner_updates", iteration, global_step)
            writer.add_scalar("debug/advantage_mean", advantage_mean, global_step)
            writer.add_scalar("debug/advantage_std", advantage_std, global_step)
            writer.add_scalar(
                "debug/return_percentile_scale",
                return_percentile_scale_value,
                global_step,
            )
            writer.add_scalar("debug/value_rmse", value_rmse_value, global_step)
            writer.add_scalar(
                "debug/value_explained_variance",
                explained_variance_value,
                global_step,
            )
            writer.add_scalar(
                "critic/target_outside_support",
                target_outside_support_value,
                global_step,
            )
            writer.add_scalar(
                "critic/target_edge_mass",
                target_edge_mass_value,
                global_step,
            )
            writer.add_scalar(
                "critic/prediction_edge_mass",
                prediction_edge_mass_value,
                global_step,
            )
            print(f"SPS: {sps}")

    envs.close()
    writer.close()

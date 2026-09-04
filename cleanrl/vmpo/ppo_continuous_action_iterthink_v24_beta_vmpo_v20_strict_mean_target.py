# Projected-target v20 — strict mean-coordinate target fitting.
#
# V19's full Beta fitting KL found a shortcut: it reduced concentration toward
# the alpha,beta > 1 boundary instead of matching per-state target means. V20
# keeps the same single-sample Fisher mean target and behavior-to-target KL
# budget, but evaluates target-to-online KL after rebuilding the online Beta at
# the detached behavior concentration. The actor loss therefore differentiates
# only through online mean and cannot improve by broadening the distribution.
#
# The stopped mean target remains self-terminating, requires no candidate
# actions, and omits the noisy one-action concentration policy gradient. GAE,
# PopArt, staggered phases, vectorized per-environment observation
# normalization, the one-update schedule, and the IterThink model are unchanged.

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
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

SAMPLE_EPS = 1e-6
ADVANTAGE_EPS = 1e-8
FISHER_EPS = 1e-8
MIN_BETA_SHAPE = 1.0 + 1e-6


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

    target_mean_kl: float = 0.007071067811865476
    target_bracket_steps: int = 8
    target_bisection_steps: int = 24

    # PopArt, including the single-task setting used by the paper.
    popart_rate: float = 1e-4
    popart_std_min: float = 1e-2
    popart_std_max: float = 1e6

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
    popart_mean: torch.Tensor
    popart_sq_mean: torch.Tensor
    popart_std: torch.Tensor

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.trunk = ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
        self.policy_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        self.actor_alpha = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.value_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        self.value_head = layer_init(nn.Linear(256, 1), std=1.0)
        with torch.no_grad():
            self.value_head.weight.zero_()
            self.value_head.bias.zero_()
        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )
        self.register_buffer("popart_mean", torch.zeros(()))
        self.register_buffer("popart_sq_mean", torch.ones(()))
        self.register_buffer("popart_std", torch.ones(()))

    def policy(self, observations):
        features = self.policy_mlp(self.trunk(observations))
        alpha = 1.0 + F.softplus(self.actor_alpha(features))
        beta = 1.0 + F.softplus(self.actor_beta(features))
        return alpha, beta

    def forward(self, observations):
        features = self.trunk(observations)
        policy_features = self.policy_mlp(features)
        alpha = 1.0 + F.softplus(self.actor_alpha(policy_features))
        beta = 1.0 + F.softplus(self.actor_beta(policy_features))
        normalized_value = self.value_head(self.value_mlp(features)).squeeze(-1)
        return alpha, beta, normalized_value

    def value(self, observations):
        features = self.value_mlp(self.trunk(observations))
        return self.value_head(features).squeeze(-1)

    @torch.no_grad()
    def update_popart(self, returns, rate, std_min, std_max):
        old_mean = self.popart_mean.clone()
        old_std = self.popart_std.clone()
        new_mean = torch.lerp(old_mean, returns.mean(), rate)
        new_sq_mean = torch.lerp(self.popart_sq_mean, returns.square().mean(), rate)
        new_std = (
            (new_sq_mean - new_mean.square())
            .clamp_min(0.0)
            .sqrt()
            .clamp(std_min, std_max)
        )

        # Preserve every unnormalized value prediction while changing normalization.
        self.value_head.weight.mul_(old_std / new_std)
        self.value_head.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)
        self.popart_mean.copy_(new_mean)
        self.popart_sq_mean.copy_(new_sq_mean)
        self.popart_std.copy_(new_std)
        return (returns - new_mean) / new_std




def beta_kl(old_alpha, old_beta, new_alpha, new_beta):
    old_sum = old_alpha + old_beta
    new_sum = new_alpha + new_beta
    kl = (
        torch.lgamma(new_alpha)
        + torch.lgamma(new_beta)
        - torch.lgamma(new_sum)
        - torch.lgamma(old_alpha)
        - torch.lgamma(old_beta)
        + torch.lgamma(old_sum)
        + (old_alpha - new_alpha) * torch.digamma(old_alpha)
        + (old_beta - new_beta) * torch.digamma(old_beta)
        + (new_sum - old_sum) * torch.digamma(old_sum)
    )
    return kl.sum(-1)




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
    if args.target_mean_kl <= 0.0:
        raise ValueError("target_mean_kl must be positive")
    if args.target_bracket_steps <= 0 or args.target_bisection_steps <= 0:
        raise ValueError("target line-search step counts must be positive")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.bf16 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 support is required")

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

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, index, args.capture_video, run_name)
            for index in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("V-MPO continuous control requires a Box action space")
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
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            behavior_alpha, behavior_beta, normalized_value = agent(observations)
        value = normalized_value.float() * agent.popart_std + agent.popart_mean
        return behavior_alpha.float(), behavior_beta.float(), value


    def gae_model(
        transition_next_observations,
        reward_batch,
        value_batch,
        terminations,
        boundaries,
    ):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            normalized_next_values = agent.value(
                transition_next_observations.reshape(
                    (args.batch_size,) + transition_next_observations.shape[2:]
                )
            )
        next_values = (
            normalized_next_values.float() * agent.popart_std + agent.popart_mean
        ).view(args.num_steps, args.num_envs)
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
        native_actions,
        old_alpha,
        old_beta,
        advantages,
        normalized_returns,
    ):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            new_alpha, new_beta, normalized_values = agent(observations)
        new_alpha = new_alpha.float()
        new_beta = new_beta.float()
        normalized_values = normalized_values.float()

        with torch.no_grad():
            advantage_mean = advantages.mean()
            advantage_std = advantages.std()
            advantage_scale = advantages.var(unbiased=False).add(ADVANTAGE_EPS).sqrt()
            standardized_advantages = (advantages - advantage_mean) / advantage_scale

            old_concentration = old_alpha + old_beta
            old_mean = old_alpha / old_concentration
            clamped_actions = native_actions.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

            # Score and Fisher information for the Beta mean while holding
            # concentration fixed. Multiplying F^-1 score by the standardized
            # advantage gives the local natural policy-gradient direction.
            mean_score = old_concentration * (
                clamped_actions.log()
                - torch.log1p(-clamped_actions)
                - torch.digamma(old_alpha)
                + torch.digamma(old_beta)
            )
            mean_fisher = old_concentration.square() * (
                torch.polygamma(1, old_alpha) + torch.polygamma(1, old_beta)
            )
            mean_direction = (
                standardized_advantages.unsqueeze(-1)
                * mean_score
                / mean_fisher.clamp_min(FISHER_EPS)
            )

            # Keep every target inside the alpha,beta > 1 carrier without
            # moving a behavior shape that is already within numerical epsilon
            # of the boundary.
            shape_floor = torch.minimum(
                torch.minimum(old_alpha, old_beta),
                torch.full_like(old_alpha, MIN_BETA_SHAPE),
            )
            minimum_mean = shape_floor / old_concentration
            maximum_mean = 1.0 - minimum_mean

            # The local quadratic model supplies the initial bracket. Fixed
            # expansion and bisection counts keep this path CUDA-graphable.
            quadratic_kl = (
                0.5 * mean_fisher * mean_direction.square()
            ).sum(-1).mean()
            scale_high = (
                args.target_mean_kl / quadratic_kl.clamp_min(FISHER_EPS)
            ).sqrt()
            for _ in range(args.target_bracket_steps):
                bracket_mean = (
                    old_mean + scale_high * mean_direction
                ).clamp(minimum_mean, maximum_mean)
                bracket_alpha = bracket_mean * old_concentration
                bracket_beta = (1.0 - bracket_mean) * old_concentration
                bracket_kl = beta_kl(
                    old_alpha, old_beta, bracket_alpha, bracket_beta
                ).mean()
                scale_high = torch.where(
                    bracket_kl < args.target_mean_kl,
                    scale_high * 2.0,
                    scale_high,
                )

            scale_low = torch.zeros_like(scale_high)
            for _ in range(args.target_bisection_steps):
                scale_mid = 0.5 * (scale_low + scale_high)
                candidate_mean = (
                    old_mean + scale_mid * mean_direction
                ).clamp(minimum_mean, maximum_mean)
                candidate_alpha = candidate_mean * old_concentration
                candidate_beta = (1.0 - candidate_mean) * old_concentration
                candidate_kl = beta_kl(
                    old_alpha, old_beta, candidate_alpha, candidate_beta
                ).mean()
                below_budget = candidate_kl < args.target_mean_kl
                scale_low = torch.where(below_budget, scale_mid, scale_low)
                scale_high = torch.where(below_budget, scale_high, scale_mid)

            target_unclamped_mean = old_mean + scale_low * mean_direction
            target_mean = target_unclamped_mean.clamp(minimum_mean, maximum_mean)
            target_alpha = target_mean * old_concentration
            target_beta = (1.0 - target_mean) * old_concentration

            behavior_target_kl = beta_kl(
                old_alpha, old_beta, target_alpha, target_beta
            ).mean()
            target_behavior_kl = beta_kl(
                target_alpha, target_beta, old_alpha, old_beta
            ).mean()
            target_kl_residual = args.target_mean_kl - behavior_target_kl
            target_saturation_fraction = (
                (target_unclamped_mean <= minimum_mean)
                | (target_unclamped_mean >= maximum_mean)
            ).float().mean()
            target_mean_shift = (target_mean - old_mean).abs().mean()
            natural_score_norm = (
                mean_score.square() / mean_fisher.clamp_min(FISHER_EPS)
            ).sum(-1)

        # Compare target and online means at the same detached behavior
        # concentration. This remains a proper forward Beta KL in the optimized
        # coordinate, but removes v19's concentration-broadening shortcut.
        online_mean = new_alpha / (new_alpha + new_beta)
        mean_fit_alpha = online_mean * old_concentration
        mean_fit_beta = (1.0 - online_mean) * old_concentration
        target_fit_kl = beta_kl(
            target_alpha, target_beta, mean_fit_alpha, mean_fit_beta
        ).mean()
        # Normalize out the target scale so the local gradient at behavior is
        # the ordinary projected policy-gradient magnitude.
        policy_loss = target_fit_kl / scale_low.clamp_min(1e-3)
        value_loss = 0.5 * (normalized_values - normalized_returns).square().mean()
        total_loss = policy_loss + value_loss

        value_error = (
            normalized_values - normalized_returns
        ) * agent.popart_std
        unnormalized_returns = (
            normalized_returns * agent.popart_std + agent.popart_mean
        )
        value_rmse = value_error.square().mean().sqrt()
        explained_variance = 1.0 - value_error.var(unbiased=False) / (
            unnormalized_returns.var(unbiased=False) + ADVANTAGE_EPS
        )
        policy_concentration = (new_alpha + new_beta).mean()
        policy_variance = (
            new_alpha
            * new_beta
            / (
                (new_alpha + new_beta).square()
                * (new_alpha + new_beta + 1.0)
            )
        ).mean()
        metrics = torch.stack(
            (
                policy_loss.detach(),
                target_fit_kl.detach(),
                value_loss.detach(),
                behavior_target_kl.detach(),
                target_behavior_kl.detach(),
                target_kl_residual.detach(),
                scale_low.detach(),
                target_mean_shift.detach(),
                target_saturation_fraction.detach(),
                natural_score_norm.mean().detach(),
                natural_score_norm.max().detach(),
                advantage_mean.detach(),
                advantage_std.detach(),
                value_rmse.detach(),
                explained_variance.detach(),
                policy_concentration.detach(),
                policy_variance.detach(),
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
    # One RunningMeanStd row per environment, matching v13's 64 independent
    # NormalizeObservation wrappers rather than pooling trajectories.
    observation_means = np.zeros(
        (args.num_envs,) + observation_shape, dtype=np.float64
    )
    observation_variances = np.ones_like(observation_means)
    observation_counts = np.full(args.num_envs, 1e-4, dtype=np.float64)
    rollout_shape = (args.num_steps, args.num_envs)
    observations = torch.empty(rollout_shape + observation_shape, device=device)
    native_actions = torch.empty(rollout_shape + action_shape, device=device)
    rewards = torch.empty(rollout_shape, device=device)
    values = torch.empty(rollout_shape, device=device)
    old_alphas = torch.empty(rollout_shape + action_shape, device=device)
    old_betas = torch.empty_like(old_alphas)
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

    # Even spacing guarantees phase coverage; only the env-to-phase assignment
    # is randomized, using an isolated seed so global NumPy sampling is unchanged.
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
            warmup_alpha, warmup_beta, _ = rollout_model(warmup_observation)
            warmup_native_action = Beta(warmup_alpha, warmup_beta).sample().clamp(
                SAMPLE_EPS,
                1.0 - SAMPLE_EPS,
            )
            warmup_action = agent.action_low + (
                agent.action_high - agent.action_low
            ) * warmup_native_action
        (
            raw_next_observations,
            _,
            warmup_terminations,
            warmup_truncations,
            warmup_infos,
        ) = envs.step(warmup_action.cpu().numpy())
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

    # Current partial episodes contain warmup rewards. Suppress exactly their
    # first completion so every reported episode is entirely post-warmup.
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
                old_alpha, old_beta, value = rollout_model(next_observation)
                distribution = Beta(old_alpha, old_beta)
                native_action = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                action = agent.action_low + (
                    agent.action_high - agent.action_low
                ) * native_action
            native_actions[step].copy_(native_action)
            old_alphas[step].copy_(old_alpha)
            old_betas[step].copy_(old_beta)
            values[step].copy_(value)

            (
                raw_next_observations,
                reward,
                terminations,
                truncations,
                infos,
            ) = envs.step(action.cpu().numpy())
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
            flat_returns = returns.reshape(-1)
            normalized_returns = agent.update_popart(
                flat_returns,
                args.popart_rate,
                args.popart_std_min,
                args.popart_std_max,
            ).detach()
            del gae_advantages, returns, flat_returns

        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        total_loss, metrics = update_loss_model(
            observations.reshape((args.batch_size,) + observation_shape),
            native_actions.reshape((args.batch_size,) + action_shape),
            old_alphas.reshape((args.batch_size,) + action_shape),
            old_betas.reshape((args.batch_size,) + action_shape),
            advantages,
            normalized_returns,
        )
        should_log = iteration % args.log_interval == 0 or iteration == 1
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if should_log:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    updated_alpha, updated_beta = agent.policy(
                        observations.reshape(
                            (args.batch_size,) + observation_shape
                        )
                    )
                post_update_behavior_kl = beta_kl(
                    old_alphas.reshape((args.batch_size,) + action_shape),
                    old_betas.reshape((args.batch_size,) + action_shape),
                    updated_alpha.float(),
                    updated_beta.float(),
                ).mean()
            packed = torch.cat(
                (
                    metrics,
                    post_update_behavior_kl.view(1),
                    agent.popart_mean.view(1),
                    agent.popart_std.view(1),
                )
            ).cpu().tolist()
            (
                policy_loss_value,
                target_fit_kl_value,
                value_loss_value,
                behavior_target_kl_value,
                target_behavior_kl_value,
                target_kl_residual_value,
                target_scale_value,
                target_mean_shift_value,
                target_saturation_fraction_value,
                natural_score_norm_mean_value,
                natural_score_norm_max_value,
                advantage_mean,
                advantage_std,
                value_rmse_value,
                explained_variance_value,
                policy_concentration_value,
                policy_variance_value,
                post_update_behavior_kl_value,
                popart_mean_value,
                popart_std_value,
            ) = packed
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/policy_loss", policy_loss_value, global_step)
            writer.add_scalar(
                "projected_target/target_fit_kl",
                target_fit_kl_value,
                global_step,
            )
            writer.add_scalar("losses/value_loss", value_loss_value, global_step)
            writer.add_scalar(
                "projected_target/behavior_target_kl",
                behavior_target_kl_value,
                global_step,
            )
            writer.add_scalar(
                "projected_target/target_behavior_kl",
                target_behavior_kl_value,
                global_step,
            )
            writer.add_scalar(
                "projected_target/kl_budget_residual",
                target_kl_residual_value,
                global_step,
            )
            writer.add_scalar(
                "projected_target/step_scale", target_scale_value, global_step
            )
            writer.add_scalar(
                "projected_target/mean_absolute_shift",
                target_mean_shift_value,
                global_step,
            )
            writer.add_scalar(
                "projected_target/saturation_fraction",
                target_saturation_fraction_value,
                global_step,
            )
            writer.add_scalar(
                "projected_target/natural_score_norm_mean",
                natural_score_norm_mean_value,
                global_step,
            )
            writer.add_scalar(
                "projected_target/natural_score_norm_max",
                natural_score_norm_max_value,
                global_step,
            )
            writer.add_scalar(
                "projected_target/post_update_behavior_kl",
                post_update_behavior_kl_value,
                global_step,
            )
            writer.add_scalar(
                "projected_target/learner_updates", iteration, global_step
            )
            writer.add_scalar("debug/advantage_mean", advantage_mean, global_step)
            writer.add_scalar("debug/advantage_std", advantage_std, global_step)
            writer.add_scalar("debug/value_rmse", value_rmse_value, global_step)
            writer.add_scalar(
                "debug/value_explained_variance",
                explained_variance_value,
                global_step,
            )
            writer.add_scalar(
                "debug/policy_concentration", policy_concentration_value, global_step
            )
            writer.add_scalar(
                "debug/policy_native_variance", policy_variance_value, global_step
            )
            writer.add_scalar("popart/mean", popart_mean_value, global_step)
            writer.add_scalar("popart/std", popart_std_value, global_step)
            print(f"SPS: {sps}")

    envs.close()
    writer.close()

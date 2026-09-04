# V-MPO v41 — conditional MPO with a replay-trained pessimistic twin-Q teacher.
#
# The E-step normalizes K counterfactual actions independently at every state;
# it is not a joint batch distribution or a global elite filter. A fresh
# candidate is fit, exactly Beta-KL projected, and held-out Q tested before the
# complete candidate is atomically promoted to the frozen deployed policy.

import copy
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
ETA_FLOOR = 1e-6


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
    num_envs: int = 64
    num_steps: int = 39
    gamma: float = 0.99

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16
    critic_hidden: int = 256

    replay_capacity: int = 1_000_000
    replay_batch_size: int = 1024
    critic_learning_starts: int = 100_000
    # Fit Q to the initial replay distribution before the first policy cycle.
    critic_initial_updates: int = 1_024
    critic_updates_per_rollout: int = 8
    critic_tau: float = 0.005
    critic_huber_delta: float = 1.0
    critic_max_grad_norm: float = 10.0

    counterfactual_actions: int = 32
    epsilon_eta: float = 1.0
    eta_bisection_steps: int = 48
    m_step_updates: int = 8
    m_step_minibatch_size: int = 256
    actor_max_grad_norm: float = 10.0

    epsilon_policy_kl: float = 0.01
    projection_steps: int = 12
    policy_kl_tolerance: float = 1e-6
    acceptance_actions: int = 32
    acceptance_z: float = 1.645

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
    """Update independent per-environment moments and return clipped values."""
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
        (observations - new_means) / np.sqrt(new_variances + 1e-8), -10, 10
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
    """Normalize autoreset observations in per-environment wrapper order."""
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
    return normalized_next_observations, raw_transition_observations


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


class Actor(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor

    def __init__(self, observation_shape, action_space, args):
        super().__init__()
        observation_dim = int(np.prod(observation_shape))
        self.action_shape = tuple(action_space.shape)
        action_dim = int(np.prod(self.action_shape))
        self.trunk = ThinkTrunk(
            observation_dim, args.hidden, args.k_blocks, args.n_experts
        )
        self.policy_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        self.actor_alpha = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.register_buffer(
            "action_low",
            torch.tensor(action_space.low, dtype=torch.float32).reshape(-1),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(action_space.high, dtype=torch.float32).reshape(-1),
        )

    def forward(self, observations):
        flat_observations = observations.reshape(observations.shape[0], -1)
        features = self.policy_mlp(self.trunk(flat_observations))
        alpha = 1.0 + F.softplus(self.actor_alpha(features))
        beta = 1.0 + F.softplus(self.actor_beta(features))
        return alpha, beta

    def to_environment_action(self, native_action):
        scaled_action = self.action_low + (
            self.action_high - self.action_low
        ) * native_action
        return scaled_action.reshape((native_action.shape[0],) + self.action_shape)


class QNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(observation_dim + action_dim, hidden)),
            nn.LayerNorm(hidden),
            ReLUSquared(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            ReLUSquared(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, normalized_observations, native_actions):
        observations = normalized_observations.reshape(
            normalized_observations.shape[0], -1
        )
        actions = native_actions.reshape(native_actions.shape[0], -1)
        return self.network(torch.cat((observations, actions), dim=-1)).squeeze(-1)


class TwinQ(nn.Module):
    """Two parameter-disjoint LayerNorm Q MLPs."""

    def __init__(self, observation_shape, action_shape, hidden):
        super().__init__()
        observation_dim = int(np.prod(observation_shape))
        action_dim = int(np.prod(action_shape))
        self.q1 = QNetwork(observation_dim, action_dim, hidden)
        self.q2 = QNetwork(observation_dim, action_dim, hidden)

    def forward(self, normalized_observations, native_actions):
        return (
            self.q1(normalized_observations, native_actions),
            self.q2(normalized_observations, native_actions),
        )


class ReplayRing:
    """Fixed-capacity CUDA ring containing raw transitions."""

    def __init__(self, capacity, observation_shape, action_dim, num_envs, device):
        self.capacity = capacity
        self.device = device
        self.raw_observations = torch.empty(
            (capacity,) + observation_shape, device=device, dtype=torch.float32
        )
        self.raw_next_observations = torch.empty_like(self.raw_observations)
        self.native_actions = torch.empty(
            (capacity, action_dim), device=device, dtype=torch.float32
        )
        self.raw_rewards = torch.empty(capacity, device=device, dtype=torch.float32)
        self.terminations = torch.empty(capacity, device=device, dtype=torch.float32)
        self.environment_indices = torch.empty(
            capacity, device=device, dtype=torch.int64
        )
        self.batch_environment_indices = torch.arange(
            num_envs, device=device, dtype=torch.int64
        )
        self.position = 0
        self.size = 0

    def add(
        self,
        raw_observations,
        raw_next_observations,
        native_actions,
        raw_rewards,
        terminations,
    ):
        batch_size = raw_observations.shape[0]
        indices = (
            torch.arange(batch_size, device=self.device, dtype=torch.int64)
            + self.position
        ).remainder(self.capacity)
        self.raw_observations[indices] = torch.as_tensor(
            raw_observations, device=self.device, dtype=torch.float32
        )
        self.raw_next_observations[indices] = torch.as_tensor(
            raw_next_observations, device=self.device, dtype=torch.float32
        )
        self.native_actions[indices] = native_actions
        self.raw_rewards[indices] = torch.as_tensor(
            raw_rewards, device=self.device, dtype=torch.float32
        )
        self.terminations[indices] = torch.as_tensor(
            terminations, device=self.device, dtype=torch.float32
        )
        self.environment_indices[indices] = self.batch_environment_indices
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(self.size, (batch_size,), device=self.device)
        return (
            self.raw_observations[indices],
            self.raw_next_observations[indices],
            self.native_actions[indices],
            self.raw_rewards[indices],
            self.terminations[indices],
            self.environment_indices[indices],
        )


def beta_log_prob(alpha, beta, native_action):
    native_action = native_action.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    log_normalizer = (
        torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    )
    return (
        (alpha - 1.0) * native_action.log()
        + (beta - 1.0) * torch.log1p(-native_action)
        - log_normalizer
    ).sum(-1)


def beta_kl(old_alpha, old_beta, new_alpha, new_beta):
    """Full analytic forward KL for factorized Beta policies."""
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


def normalize_replay_observations(
    raw_observations, environment_indices, observation_means, observation_variances
):
    means = observation_means[environment_indices]
    variances = observation_variances[environment_indices]
    return ((raw_observations - means) * torch.rsqrt(variances + 1e-8)).clamp(
        -10.0, 10.0
    )


@torch.no_grad()
def solve_conditional_weights(scores, epsilon_eta, bisection_steps):
    """Solve one shared eta for mean statewise KL(q_s || Uniform(K))."""
    if not torch.isfinite(scores).all():
        raise RuntimeError("non-finite target-Q score in conditional E-step")
    num_actions = scores.shape[1]
    log_num_actions = scores.new_tensor(float(np.log(num_actions)))
    uniform_weights = torch.full_like(scores, 1.0 / num_actions)
    centered_scores = scores - scores.max(dim=1, keepdim=True).values
    score_span = -centered_scores.min()
    if score_span == 0.0:
        return scores.new_tensor(ETA_FLOOR), uniform_weights, scores.new_zeros(())

    def weights_and_kl(eta):
        log_weights = F.log_softmax(centered_scores / eta, dim=1)
        weights = log_weights.exp()
        mean_kl = (weights * (log_weights + log_num_actions)).sum(dim=1).mean()
        return weights, mean_kl

    eta_floor = scores.new_tensor(ETA_FLOOR)
    floor_weights, floor_kl = weights_and_kl(eta_floor)
    # Ties can make the requested KL infeasible. The floor solution remains a
    # valid conditional distribution and is uniform over exactly tied maxima.
    if floor_kl <= epsilon_eta:
        return eta_floor, floor_weights, floor_kl

    eta_low = eta_floor
    eta_high = torch.maximum(
        score_span / epsilon_eta, eta_floor * scores.new_tensor(2.0)
    )
    _, high_kl = weights_and_kl(eta_high)
    for _ in range(32):
        if high_kl <= epsilon_eta:
            break
        eta_high = eta_high * 2.0
        _, high_kl = weights_and_kl(eta_high)
    if high_kl > epsilon_eta:
        raise RuntimeError("failed to bracket conditional MPO temperature")

    log_eta_low = eta_low.log()
    log_eta_high = eta_high.log()
    for _ in range(bisection_steps):
        log_eta_mid = 0.5 * (log_eta_low + log_eta_high)
        eta_mid = log_eta_mid.exp()
        _, mid_kl = weights_and_kl(eta_mid)
        if mid_kl > epsilon_eta:
            log_eta_low = log_eta_mid
        else:
            log_eta_high = log_eta_mid
    eta = log_eta_high.exp()
    weights, mean_kl = weights_and_kl(eta)
    return eta, weights, mean_kl


@torch.no_grad()
def polyak_update(source, target, tau):
    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        target_parameter.lerp_(source_parameter, tau)


def validate_args(args):
    if args.num_envs <= 0 or args.num_steps <= 0:
        raise ValueError("num_envs and num_steps must be positive")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size < 4:
        raise ValueError("rollout batch must provide at least two fit and held-out states")
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if args.actor_learning_rate <= 0.0 or args.critic_learning_rate <= 0.0:
        raise ValueError("actor and critic learning rates must be positive")
    if args.hidden <= 0 or args.k_blocks <= 0 or args.n_experts <= 0:
        raise ValueError("actor architecture dimensions must be positive")
    if args.critic_hidden <= 0:
        raise ValueError("critic_hidden must be positive")
    if args.replay_capacity < args.num_envs:
        raise ValueError("replay_capacity must hold at least one vector step")
    if not 0 < args.replay_batch_size <= args.replay_capacity:
        raise ValueError("replay_batch_size must be in [1, replay_capacity]")
    if args.critic_learning_starts <= 0:
        raise ValueError("critic_learning_starts must be positive")
    if args.critic_updates_per_rollout <= 0 or args.critic_initial_updates <= 0:
        raise ValueError("critic update counts must be positive")
    if not 0.0 < args.critic_tau <= 1.0:
        raise ValueError("critic_tau must be in (0, 1]")
    if args.critic_huber_delta <= 0.0 or args.critic_max_grad_norm <= 0.0:
        raise ValueError("critic loss scale and gradient norm must be positive")
    if args.counterfactual_actions < 2:
        raise ValueError("counterfactual_actions must be at least two")
    if not 0.0 < args.epsilon_eta < np.log(args.counterfactual_actions):
        raise ValueError("epsilon_eta must be in (0, log(counterfactual_actions))")
    if args.eta_bisection_steps <= 0:
        raise ValueError("eta_bisection_steps must be positive")
    if args.m_step_updates <= 0 or args.m_step_minibatch_size <= 0:
        raise ValueError("M-step update count and minibatch size must be positive")
    fit_state_count = args.batch_size // 2
    if args.m_step_minibatch_size > fit_state_count:
        raise ValueError("m_step_minibatch_size cannot exceed the fit-state half")
    if args.actor_max_grad_norm <= 0.0:
        raise ValueError("actor_max_grad_norm must be positive")
    if args.epsilon_policy_kl <= 0.0:
        raise ValueError("epsilon_policy_kl must be positive")
    if args.projection_steps <= 0 or args.policy_kl_tolerance < 0.0:
        raise ValueError("projection steps must be positive and tolerance nonnegative")
    if args.acceptance_actions < 2 or args.acceptance_z <= 0.0:
        raise ValueError("acceptance action count and z score must be valid")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if args.num_steps != 39:
        raise ValueError("OpenAI Gym paper alignment requires a 39-step unroll")

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
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps leaves no complete post-warmup rollout")
    if args.critic_learning_starts >= args.total_timesteps:
        raise ValueError("critic_learning_starts must precede total_timesteps")
    return warmup_transitions


if __name__ == "__main__":
    args = tyro.cli(Args)
    warmup_transitions = validate_args(args)
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 support is required when bf16 is enabled")

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
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, index, args.capture_video, run_name)
            for index in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("conditional MPO continuous control requires a Box action space")
    if (
        envs.single_observation_space.shape is None
        or envs.single_action_space.shape is None
    ):
        raise ValueError("continuous-control observation and action shapes are required")
    if envs.envs[0].spec is None or (
        envs.envs[0].spec.max_episode_steps != args.initial_phase_warmup_steps
    ):
        raise RuntimeError("constructed environment horizon differs from gym spec")
    observation_shape = tuple(envs.single_observation_space.shape)
    action_shape = tuple(envs.single_action_space.shape)

    deployed_actor = Actor(observation_shape, envs.single_action_space, args).to(device)
    deployed_actor.requires_grad_(False)
    candidate_actor = Actor(observation_shape, envs.single_action_space, args).to(device)
    candidate_actor.load_state_dict(deployed_actor.state_dict())
    critic = TwinQ(observation_shape, action_shape, args.critic_hidden).to(device)
    target_critic = copy.deepcopy(critic).requires_grad_(False)
    critic_optimizer = optim.Adam(
        critic.parameters(),
        lr=args.critic_learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )

    deployed_policy_model = deployed_actor
    candidate_policy_model = candidate_actor
    critic_model = critic
    target_critic_model = target_critic
    if args.compile:
        deployed_policy_model = torch.compile(
            deployed_actor, mode=args.compile_mode, dynamic=True, fullgraph=True
        )
        candidate_policy_model = torch.compile(
            candidate_actor, mode=args.compile_mode, dynamic=True, fullgraph=True
        )
        critic_model = torch.compile(
            critic, mode=args.compile_mode, dynamic=True, fullgraph=True
        )
        target_critic_model = torch.compile(
            target_critic, mode=args.compile_mode, dynamic=True, fullgraph=True
        )
        print(f"compiled BF16 dynamic training paths ({args.compile_mode})")

    def policy_parameters(policy_model, normalized_observations):
        with torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=args.bf16
        ):
            alpha, beta = policy_model(normalized_observations)
        return alpha.float(), beta.float()

    def q_values(q_model, normalized_observations, native_actions):
        with torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=args.bf16
        ):
            q1, q2 = q_model(normalized_observations, native_actions)
        return q1.float(), q2.float()

    replay = ReplayRing(
        args.replay_capacity,
        observation_shape,
        int(np.prod(action_shape)),
        args.num_envs,
        device,
    )
    observation_means = np.zeros(
        (args.num_envs,) + observation_shape, dtype=np.float64
    )
    observation_variances = np.ones_like(observation_means)
    observation_counts = np.full(args.num_envs, 1e-4, dtype=np.float64)
    rollout_raw_observations = torch.empty(
        (args.num_steps, args.num_envs) + observation_shape,
        device=device,
        dtype=torch.float32,
    )

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
        "initial_phase_offsets", ",".join(str(offset) for offset in phase_offsets)
    )
    scheduled_resets = args.initial_phase_warmup_steps - phase_offsets

    for warmup_step in range(1, args.initial_phase_warmup_steps + 1):
        raw_observation_batch = np.array(raw_next_observations, copy=True)
        warmup_observation = torch.as_tensor(
            next_observation_np, device=device, dtype=torch.float32
        )
        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            warmup_alpha, warmup_beta = policy_parameters(
                deployed_policy_model, warmup_observation
            )
            warmup_native_action = Beta(warmup_alpha, warmup_beta).sample().clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
            warmup_action = deployed_actor.to_environment_action(
                warmup_native_action
            )
        (
            raw_next_observations,
            warmup_reward,
            warmup_terminations,
            warmup_truncations,
            warmup_infos,
        ) = envs.step(warmup_action.cpu().numpy())
        next_observation_np, raw_transition_observations = normalize_vector_step(
            raw_next_observations,
            warmup_terminations,
            warmup_truncations,
            warmup_infos,
            observation_means,
            observation_variances,
            observation_counts,
        )
        replay.add(
            raw_observation_batch,
            raw_transition_observations,
            warmup_native_action,
            warmup_reward,
            warmup_terminations,
        )
        global_step += args.num_envs

        for env_index in np.flatnonzero(scheduled_resets == warmup_step):
            reset_observation, _ = envs.envs[env_index].reset()
            raw_next_observations[env_index] = reset_observation
            next_observation_np[env_index] = normalize_observations(
                reset_observation[None, ...],
                observation_means,
                observation_variances,
                observation_counts,
                slice(env_index, env_index + 1),
            )[0]

    suppress_next_episode_log = phase_offsets > 0
    if global_step != warmup_transitions:
        raise RuntimeError("phase warmup transition accounting mismatch")
    next_observation = torch.as_tensor(
        next_observation_np, device=device, dtype=torch.float32
    )

    accept_count = 0
    candidate_cycle_count = 0
    bellman_rmse_value = 0.0
    twin_gap_value = 0.0
    eta_value = 0.0
    e_step_kl_value = 0.0
    e_step_ess_value = 0.0
    projected_fraction_value = 0.0
    final_kl_value = 0.0
    q_gain_mean_value = 0.0
    q_gain_lcb_value = 0.0
    policy_concentration_value = 0.0
    policy_variance_value = 0.0

    for iteration in range(1, args.num_iterations + 1):
        for step in range(args.num_steps):
            raw_observation_batch = np.array(raw_next_observations, copy=True)
            raw_observation_tensor = torch.as_tensor(
                raw_observation_batch, device=device, dtype=torch.float32
            )
            rollout_raw_observations[step].copy_(raw_observation_tensor)
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                alpha, beta = policy_parameters(
                    deployed_policy_model, next_observation
                )
                native_action = Beta(alpha, beta).sample().clamp(
                    SAMPLE_EPS, 1.0 - SAMPLE_EPS
                )
                action = deployed_actor.to_environment_action(native_action)

            (
                raw_next_observations,
                raw_reward,
                terminations,
                truncations,
                infos,
            ) = envs.step(action.cpu().numpy())
            next_observation_np, raw_transition_observations = normalize_vector_step(
                raw_next_observations,
                terminations,
                truncations,
                infos,
                observation_means,
                observation_variances,
                observation_counts,
            )
            replay.add(
                raw_observation_tensor,
                raw_transition_observations,
                native_action,
                raw_reward,
                terminations,
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

        learning_ready = (
            global_step >= args.critic_learning_starts
            and replay.size >= args.replay_batch_size
        )
        if learning_ready:
            observation_means_cuda = torch.as_tensor(
                observation_means, device=device, dtype=torch.float32
            )
            observation_variances_cuda = torch.as_tensor(
                observation_variances, device=device, dtype=torch.float32
            )
            rmse_accumulator = torch.zeros((), device=device)
            gap_accumulator = torch.zeros((), device=device)
            critic_update_count = (
                args.critic_initial_updates
                if candidate_cycle_count == 0
                else args.critic_updates_per_rollout
            )
            for _ in range(critic_update_count):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                (
                    replay_raw_observations,
                    replay_raw_next_observations,
                    replay_native_actions,
                    replay_raw_rewards,
                    replay_terminations,
                    replay_environment_indices,
                ) = replay.sample(args.replay_batch_size)
                replay_observation_batch = normalize_replay_observations(
                    replay_raw_observations,
                    replay_environment_indices,
                    observation_means_cuda,
                    observation_variances_cuda,
                )
                replay_next_observation_batch = normalize_replay_observations(
                    replay_raw_next_observations,
                    replay_environment_indices,
                    observation_means_cuda,
                    observation_variances_cuda,
                )
                with torch.no_grad():
                    next_alpha, next_beta = policy_parameters(
                        deployed_policy_model, replay_next_observation_batch
                    )
                    next_native_actions = Beta(next_alpha, next_beta).sample().clamp(
                        SAMPLE_EPS, 1.0 - SAMPLE_EPS
                    )
                    target_q1, target_q2 = q_values(
                        target_critic_model,
                        replay_next_observation_batch,
                        next_native_actions,
                    )
                    bellman_target = replay_raw_rewards + args.gamma * (
                        1.0 - replay_terminations
                    ) * torch.minimum(target_q1, target_q2)

                q1, q2 = q_values(
                    critic_model, replay_observation_batch, replay_native_actions
                )
                critic_loss = F.smooth_l1_loss(
                    q1, bellman_target, beta=args.critic_huber_delta
                ) + F.smooth_l1_loss(
                    q2, bellman_target, beta=args.critic_huber_delta
                )
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    critic.parameters(), args.critic_max_grad_norm
                )
                critic_optimizer.step()
                polyak_update(critic, target_critic, args.critic_tau)
                with torch.no_grad():
                    rmse_accumulator += (
                        0.5
                        * (
                            (q1 - bellman_target).square()
                            + (q2 - bellman_target).square()
                        ).mean()
                    ).sqrt()
                    gap_accumulator += (q1 - q2).abs().mean()
            bellman_rmse_value, twin_gap_value = (
                torch.stack((rmse_accumulator, gap_accumulator))
                .div(critic_update_count)
                .cpu()
                .tolist()
            )

            fresh_raw_states = rollout_raw_observations.reshape(
                (args.batch_size,) + observation_shape
            )
            fresh_environment_indices = replay.batch_environment_indices.repeat(
                args.num_steps
            )
            fresh_states = normalize_replay_observations(
                fresh_raw_states,
                fresh_environment_indices,
                observation_means_cuda,
                observation_variances_cuda,
            )
            state_order = torch.randperm(args.batch_size, device=device)
            fit_count = args.batch_size // 2
            fit_states = fresh_states[state_order[:fit_count]].detach()
            held_states = fresh_states[state_order[fit_count:]].detach()

            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                old_fit_alpha, old_fit_beta = policy_parameters(
                    deployed_policy_model, fit_states
                )
                fit_actions = Beta(old_fit_alpha, old_fit_beta).sample(
                    (args.counterfactual_actions,)
                ).permute(1, 0, 2).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                fit_states_expanded = fit_states[:, None, :].expand(
                    -1, args.counterfactual_actions, -1
                )
                flat_fit_states = fit_states_expanded.reshape(
                    fit_count * args.counterfactual_actions, -1
                )
                flat_fit_actions = fit_actions.reshape(
                    fit_count * args.counterfactual_actions, -1
                )
                fit_q1, fit_q2 = q_values(
                    target_critic_model, flat_fit_states, flat_fit_actions
                )
                pessimistic_scores = torch.minimum(fit_q1, fit_q2).view(
                    fit_count, args.counterfactual_actions
                )
                eta, conditional_weights, e_step_kl = solve_conditional_weights(
                    pessimistic_scores,
                    args.epsilon_eta,
                    args.eta_bisection_steps,
                )
                conditional_weights = conditional_weights.detach()
                e_step_ess = conditional_weights.square().sum(dim=1).reciprocal().mean()

            # Every cycle starts from the exact deployed parameters and a fresh
            # Adam, so rejected or projected-away moments cannot leak forward.
            candidate_actor.load_state_dict(deployed_actor.state_dict())
            candidate_optimizer = optim.Adam(
                candidate_actor.parameters(),
                lr=args.actor_learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=True,
            )
            shuffled_states = torch.randperm(fit_count, device=device)
            minibatch_cursor = 0
            for _ in range(args.m_step_updates):
                if minibatch_cursor == fit_count:
                    shuffled_states = torch.randperm(fit_count, device=device)
                    minibatch_cursor = 0
                minibatch_end = min(
                    minibatch_cursor + args.m_step_minibatch_size, fit_count
                )
                minibatch_indices = shuffled_states[
                    minibatch_cursor:minibatch_end
                ]
                minibatch_cursor = minibatch_end
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                candidate_alpha, candidate_beta = policy_parameters(
                    candidate_policy_model, fit_states[minibatch_indices]
                )
                log_probabilities = beta_log_prob(
                    candidate_alpha[:, None, :],
                    candidate_beta[:, None, :],
                    fit_actions[minibatch_indices],
                )
                actor_loss = -(
                    conditional_weights[minibatch_indices] * log_probabilities
                ).sum(dim=1).mean()
                candidate_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    candidate_actor.parameters(), args.actor_max_grad_norm
                )
                candidate_optimizer.step()

            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                deployed_snapshot = {
                    name: parameter.detach().clone()
                    for name, parameter in deployed_actor.named_parameters()
                }
                unprojected_snapshot = {
                    name: parameter.detach().clone()
                    for name, parameter in candidate_actor.named_parameters()
                }
                held_old_alpha, held_old_beta = policy_parameters(
                    deployed_policy_model, held_states
                )

                def set_candidate_fraction(fraction):
                    for name, parameter in candidate_actor.named_parameters():
                        parameter.copy_(
                            torch.lerp(
                                deployed_snapshot[name],
                                unprojected_snapshot[name],
                                fraction,
                            )
                        )

                def held_full_beta_kl():
                    held_new_alpha, held_new_beta = policy_parameters(
                        candidate_policy_model, held_states
                    )
                    return beta_kl(
                        held_old_alpha,
                        held_old_beta,
                        held_new_alpha,
                        held_new_beta,
                    ).mean()

                unprojected_kl = held_full_beta_kl()
                if unprojected_kl <= args.epsilon_policy_kl:
                    projected_fraction = 1.0
                else:
                    feasible_fraction = 0.0
                    infeasible_fraction = 1.0
                    for _ in range(args.projection_steps):
                        midpoint_fraction = 0.5 * (
                            feasible_fraction + infeasible_fraction
                        )
                        set_candidate_fraction(midpoint_fraction)
                        midpoint_kl = held_full_beta_kl()
                        if midpoint_kl <= args.epsilon_policy_kl:
                            feasible_fraction = midpoint_fraction
                        else:
                            infeasible_fraction = midpoint_fraction
                    projected_fraction = feasible_fraction
                    set_candidate_fraction(projected_fraction)
                final_kl = held_full_beta_kl()

                held_candidate_alpha, held_candidate_beta = policy_parameters(
                    candidate_policy_model, held_states
                )
                candidate_actions = Beta(
                    held_candidate_alpha, held_candidate_beta
                ).sample((args.acceptance_actions,)).permute(1, 0, 2).clamp(
                    SAMPLE_EPS, 1.0 - SAMPLE_EPS
                )
                deployed_actions = Beta(
                    held_old_alpha, held_old_beta
                ).sample((args.acceptance_actions,)).permute(1, 0, 2).clamp(
                    SAMPLE_EPS, 1.0 - SAMPLE_EPS
                )
                held_count = held_states.shape[0]
                held_states_expanded = held_states[:, None, :].expand(
                    -1, args.acceptance_actions, -1
                ).reshape(held_count * args.acceptance_actions, -1)
                candidate_q1, candidate_q2 = q_values(
                    target_critic_model,
                    held_states_expanded,
                    candidate_actions.reshape(
                        held_count * args.acceptance_actions, -1
                    ),
                )
                # Materialize this result before the next compiled target-Q
                # call, whose CUDA-graph output storage may be reused.
                candidate_state_values = torch.minimum(
                    candidate_q1, candidate_q2
                ).view(held_count, args.acceptance_actions).mean(dim=1).clone()
                deployed_q1, deployed_q2 = q_values(
                    target_critic_model,
                    held_states_expanded,
                    deployed_actions.reshape(
                        held_count * args.acceptance_actions, -1
                    ),
                )
                deployed_state_values = torch.minimum(
                    deployed_q1, deployed_q2
                ).view(held_count, args.acceptance_actions).mean(dim=1)
                paired_gains = candidate_state_values - deployed_state_values
                q_gain_mean = paired_gains.mean()
                q_gain_standard_error = paired_gains.std(unbiased=True) / np.sqrt(
                    held_count
                )
                q_gain_lcb = q_gain_mean - args.acceptance_z * q_gain_standard_error
                within_kl = final_kl <= (
                    args.epsilon_policy_kl + args.policy_kl_tolerance
                )
                accepted = bool((q_gain_lcb > 0.0).item() and within_kl.item())
                if accepted:
                    deployed_actor.load_state_dict(candidate_actor.state_dict())
                    accept_count += 1
                else:
                    candidate_actor.load_state_dict(deployed_actor.state_dict())
                candidate_cycle_count += 1

                deployed_alpha, deployed_beta = policy_parameters(
                    deployed_policy_model, held_states
                )
                deployed_concentration = deployed_alpha + deployed_beta
                deployed_variance = (
                    deployed_alpha
                    * deployed_beta
                    / (
                        deployed_concentration.square()
                        * (deployed_concentration + 1.0)
                    )
                )
                (
                    eta_value,
                    e_step_kl_value,
                    e_step_ess_value,
                    projected_fraction_value,
                    final_kl_value,
                    q_gain_mean_value,
                    q_gain_lcb_value,
                    policy_concentration_value,
                    policy_variance_value,
                ) = torch.stack(
                    (
                        eta,
                        e_step_kl,
                        e_step_ess,
                        eta.new_tensor(projected_fraction),
                        final_kl,
                        q_gain_mean,
                        q_gain_lcb,
                        deployed_concentration.mean(),
                        deployed_variance.mean(),
                    )
                ).cpu().tolist()

        should_log = iteration % args.log_interval == 0 or iteration == 1
        if should_log:
            sps = int(global_step / (time.time() - start_time))
            accept_rate = accept_count / max(candidate_cycle_count, 1)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar(
                "charts/actor_learning_rate", args.actor_learning_rate, global_step
            )
            writer.add_scalar(
                "charts/critic_learning_rate", args.critic_learning_rate, global_step
            )
            writer.add_scalar("replay/size", replay.size, global_step)
            writer.add_scalar("critic/bellman_rmse", bellman_rmse_value, global_step)
            writer.add_scalar("critic/twin_gap", twin_gap_value, global_step)
            writer.add_scalar("conditional_mpo/eta", eta_value, global_step)
            writer.add_scalar(
                "conditional_mpo/e_step_kl", e_step_kl_value, global_step
            )
            writer.add_scalar(
                "conditional_mpo/e_step_ess", e_step_ess_value, global_step
            )
            writer.add_scalar(
                "conditional_mpo/e_step_ess_fraction",
                e_step_ess_value / args.counterfactual_actions,
                global_step,
            )
            writer.add_scalar(
                "conditional_mpo/projected_fraction",
                projected_fraction_value,
                global_step,
            )
            writer.add_scalar(
                "conditional_mpo/final_full_beta_kl", final_kl_value, global_step
            )
            writer.add_scalar(
                "conditional_mpo/q_gain_mean", q_gain_mean_value, global_step
            )
            writer.add_scalar(
                "conditional_mpo/q_gain_lcb", q_gain_lcb_value, global_step
            )
            writer.add_scalar(
                "conditional_mpo/accept_count", accept_count, global_step
            )
            writer.add_scalar(
                "conditional_mpo/accept_rate", accept_rate, global_step
            )
            writer.add_scalar(
                "debug/actor_concentration", policy_concentration_value, global_step
            )
            writer.add_scalar(
                "debug/actor_native_variance", policy_variance_value, global_step
            )
            print(f"SPS: {sps}")

    envs.close()
    writer.close()

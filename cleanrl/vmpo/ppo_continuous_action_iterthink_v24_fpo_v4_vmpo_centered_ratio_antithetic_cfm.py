# FPO v4 — V-MPO M-step as a centered FPO ELBO-ratio surrogate with antithetic
# CFM noise (McAllister et al. 2025, arXiv:2507.21053) on the v30 stack.
#
# v3 post-mortem (job 4545, final -506, never above random): the weighted CFM
# M-step sum_i w_i L_i has a common-mode gradient (1/B) sum_i grad L_i that is
# NOT zero at the current flow. A Gaussian is the exact MLE of its own samples,
# so V-MPO's weighted NLL has zero drift and every step is advantage signal; a
# velocity net is never the CFM optimum of its own Euler-sampled marginal, so
# every step also follows a "self-fit" drift unrelated to advantages (online
# CFM loss sat below the target's on shared pairs at every log, ELBO gap -0.04).
# Per-window spread drift of ~2.6% compounded over ~50 promotions into a x3.7
# pre-tanh variance blow-up, 48% action saturation, and pure control cost. FPO
# is immune only because standardized advantages sum to zero.
#
# v4 M-step: FPO's estimator proper. On shared stratified (t, eps) pairs,
#   r_i = exp(L_target,i - L_online,i)           (ELBO ratio, = 1 at promotion)
#   policy_loss = -sum_i (w_i - 1/B) r_i
# with V-MPO's top-50% softmax weights w. Zero-sum weights kill the common-mode
# term exactly in expectation at every theta (the ratio keeps the control
# variate unbiased off-policy within a target window), leaving only the
# advantage tilt: the top half is pulled up and the bottom half pushed down by
# 1/B each, with the softmax tilt on top. No clip: V-MPO's moment trust region
# and per-rollout single step replace PPO's clipping.
#
# Antithetic eps: each CFM pair (t, eps) is drawn with its mirror (t, -eps).
# The eps-residual gradient t^2 (v_hat - x + eps) grad v_hat has its leading
# eps term cancel across the mirror pair, leaving the sample-dependent term
# that carries the signal; same evaluation count as v3 (16 = 8 mirrored pairs).
#
# Everything else is v3: forward-time OT flow in pre-tanh space with 10 Euler
# steps and tanh box squash; decoupled Gaussian-moment KL trust region from
# K=16 coupled noise draws through online and target flows; eta bisection,
# duals, Dreamer HL-Gauss critic, normalization, compile contract from v30.
#
# Hypothesis: with the self-fit drift removed the flow M-step is the exact
# FPO first-order update of V-MPO's tilted target q, and the mode-seeking
# flow fits q's sharp edge modes better than the Beta head.
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

DUAL_FLOOR = 1e-8
VARIANCE_FLOOR = 1e-6
SATURATION_LEVEL = 0.99


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

    target_update_period: int = 100
    topk_fraction: float = 0.5
    epsilon_eta: float = 0.01
    # Neutral geometric midpoints of the paper's Gym log-uniform search ranges.
    epsilon_alpha_mean: float = 0.007071067811865476
    epsilon_alpha_spread: float = 1.5811388300841898e-5
    initial_alpha_mean: float = 1.0
    initial_alpha_spread: float = 1.0
    return_percentile_low: float = 0.05
    return_percentile_high: float = 0.95
    return_percentile_floor: float = 1.0
    num_value_bins: int = 51
    value_support_limit: float = 20_000.0
    value_sigma_to_bin_ratio: float = 0.75

    # Flow carrier.
    flow_steps: int = 10
    cfm_samples: int = 16
    moment_samples: int = 16
    time_embed_dim: int = 8
    velocity_output_std: float = 0.01
    # "epsilon": 0.5 * ||eps_hat - eps||^2 (paper default); "velocity": 0.5 * ||v_hat - u||^2.
    cfm_target: str = "epsilon"

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    compile: bool = True
    compile_mode: str = "reduce-overhead"
    bf16: bool = True
    log_interval: int = 10

    batch_size: int = 0
    topk_size: int = 0
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
    flow_times: torch.Tensor
    time_freqs: torch.Tensor

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.flow_steps = args.flow_steps
        self.epsilon_target = args.cfm_target == "epsilon"
        self.trunk = ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
        self.policy_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        self.velocity_mlp = nn.Sequential(
            layer_init(nn.Linear(256 + action_dim + args.time_embed_dim, 256)),
            ReLUSquared(),
        )
        # Near-zero velocity at init: the flow starts as the identity map, so
        # pre-tanh samples are N(0, I) and tanh gives broad box exploration.
        self.velocity_out = layer_init(
            nn.Linear(256, action_dim), std=args.velocity_output_std
        )
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
        self.register_buffer(
            "flow_times",
            torch.arange(args.flow_steps, dtype=torch.float32) / args.flow_steps,
        )
        self.register_buffer(
            "time_freqs",
            np.pi * (2 ** torch.arange(args.time_embed_dim // 2)).float(),
        )

    def policy_features(self, observations):
        return self.policy_mlp(self.trunk(observations))

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
        """Velocity field v(x_t, t | s); rows of (features, x_t, t) are aligned.

        Matmuls run under the caller's BF16 autocast; the output is promoted to
        FP32 so the integrator state and moment statistics stay in FP32.
        """
        dtype = policy_features.dtype
        hidden = self.velocity_mlp(
            torch.cat(
                [
                    policy_features,
                    x_t.to(dtype=dtype),
                    self.embed_time(t.to(dtype=dtype)),
                ],
                dim=-1,
            )
        )
        return self.velocity_out(hidden).float()

    def integrate(self, policy_features, noise):
        """Forward-time OT flow, z at t=0 to x at t=1, Euler with FP32 state."""
        x = noise.float()
        rows = x.shape[0]
        step_size = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t = self.flow_times[step].expand(rows, 1)
            x = x + step_size * self.velocity(policy_features, x, t)
        return x

    def squash(self, pretanh):
        return self.action_low + (self.action_high - self.action_low) * (
            0.5 * (torch.tanh(pretanh) + 1.0)
        )

    def cfm_losses(self, policy_features, x_data, noise, t):
        """Per-sample CFM loss on the OT path x_t = (1 - t) z + t x, shape (B,).

        x_data: (B, A) pre-tanh actions; noise: (B, N, A); t: (B, N, 1).
        The conditional velocity is u = x - z. With eps-supervision the
        residual is eps_hat - eps = t (v_hat - u).
        """
        x_data = x_data.float().unsqueeze(1)
        noise = noise.float()
        t = t.float()
        sample_count = noise.shape[1]
        x_t = (1.0 - t) * noise + t * x_data
        conditional_velocity = x_data - noise
        flat_features = (
            policy_features.unsqueeze(1)
            .expand(-1, sample_count, -1)
            .reshape(-1, policy_features.shape[-1])
        )
        velocity = self.velocity(
            flat_features,
            x_t.reshape(-1, self.action_dim),
            t.reshape(-1, 1),
        ).view_as(conditional_velocity)
        residual = velocity - conditional_velocity
        if self.epsilon_target:
            residual = t * residual
        return 0.5 * residual.square().sum(dim=-1).mean(dim=-1)


def flow_moments(paths):
    """Per-state mean and unbiased variance over coupled noise draws, (B, K, A)."""
    return paths.mean(dim=1), paths.var(dim=1, unbiased=True)


def decoupled_moment_kl(target_mean, target_variance, online_mean, online_variance):
    """Gaussian-moment KL(target || online) split into mean and spread moves.

    mean_kl holds the target spread fixed and moves the mean; spread_kl holds
    the target mean fixed and moves the variance. Both sum over action dims.
    """
    target_variance = target_variance.clamp_min(VARIANCE_FLOOR)
    variance_ratio = online_variance.clamp_min(VARIANCE_FLOOR) / target_variance
    mean_kl = 0.5 * ((online_mean - target_mean).square() / target_variance).sum(-1)
    spread_kl = 0.5 * (variance_ratio - 1.0 - variance_ratio.log()).sum(-1)
    return mean_kl, spread_kl


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.topk_size = int(args.batch_size * args.topk_fraction)
    if not 0 < args.topk_size <= args.batch_size:
        raise ValueError("topk_fraction produces an invalid global top-k size")
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
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.bf16 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 support is required")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if args.target_update_period <= 0:
        raise ValueError("target_update_period must be positive")
    if args.target_update_period % args.log_interval != 0:
        raise ValueError(
            "target_update_period must be divisible by log_interval so the "
            "maximum hold uses the existing synchronized promotion check"
        )
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
    if args.flow_steps <= 0:
        raise ValueError("flow_steps must be positive")
    if args.cfm_samples <= 0 or args.cfm_samples % 2:
        raise ValueError("cfm_samples must be a positive even count (antithetic pairs)")
    if args.moment_samples < 2:
        raise ValueError("moment_samples must be at least two for a variance")
    if args.time_embed_dim <= 0 or args.time_embed_dim % 2 != 0:
        raise ValueError("time_embed_dim must be a positive even number")
    if args.velocity_output_std <= 0.0:
        raise ValueError("velocity_output_std must be positive")
    if args.cfm_target not in ("epsilon", "velocity"):
        raise ValueError("cfm_target must be 'epsilon' or 'velocity'")

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
        raise TypeError("V-MPO continuous control requires a Box action space")
    if envs.envs[0].spec is None or (
        envs.envs[0].spec.max_episode_steps != args.initial_phase_warmup_steps
    ):
        raise RuntimeError("constructed environment horizon differs from gym spec")

    agent = Agent(envs, args).to(device)
    target_agent = copy.deepcopy(agent).requires_grad_(False)
    duals = nn.Parameter(
        torch.tensor(
            [
                args.initial_alpha_mean,
                args.initial_alpha_spread,
            ],
            device=device,
        )
    )
    optimizer = optim.Adam(
        [*agent.parameters(), duals],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )

    autocast_dtype = torch.bfloat16

    def rollout_model(observations):
        batch = observations.shape[0]
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            target_features = target_agent.policy_features(observations)
            value_logits = agent.value_logits(observations)
            noise = torch.randn(
                batch, agent.action_dim, device=observations.device, dtype=torch.float32
            )
            pretanh_action = target_agent.integrate(target_features, noise)
        env_action = target_agent.squash(pretanh_action)
        values = hl_support.to_scalar(value_logits.float())
        return pretanh_action, env_action, values

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

    METRIC_NAMES = (
        "losses/policy_loss",
        "losses/value_loss",
        "losses/temperature_loss",
        "vmpo/mean_kl",
        "vmpo/spread_kl",
        "vmpo/full_moment_kl",
        "vmpo/weight_ess_fraction",
        "vmpo/top_advantage_min",
        "debug/advantage_mean",
        "debug/advantage_std",
        "vmpo/e_step_kl",
        "vmpo/eta_stationarity",
        "vmpo/weight_perplexity_fraction",
        "vmpo/max_weight",
        "vmpo/weight_ess",
        "vmpo/mean_kl_residual",
        "vmpo/spread_kl_residual",
        "debug/value_rmse",
        "debug/value_explained_variance",
        "critic/target_outside_support",
        "critic/target_edge_mass",
        "critic/prediction_edge_mass",
        "fpo/cfm_online",
        "fpo/cfm_target",
        "fpo/elbo_kl",
        "fpo/weighted_elbo_kl",
        "fpo/ratio_std",
        "fpo/ratio_max",
        "fpo/log_ratio_abs_mean",
        "fpo/selected_ratio_gap",
        "fpo/pretanh_spread",
        "fpo/pretanh_mean_abs",
        "fpo/pretanh_abs_max",
        "fpo/action_saturation",
        "fpo/flow_displacement",
        "vmpo/eta",
    )

    def update_loss_model(
        observations,
        pretanh_actions,
        advantages,
        value_targets,
    ):
        batch = observations.shape[0]
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            policy_features, value_logits = agent.encode(observations)
            with torch.no_grad():
                target_features = target_agent.policy_features(observations)

            # Antithetic CFM pairs: N/2 stratified times (one draw per stratum
            # of width 2/N per sample, covering the whole path) each evaluated
            # at (t, eps) and its mirror (t, -eps).
            half_samples = args.cfm_samples // 2
            half_noise = torch.randn(
                batch,
                half_samples,
                agent.action_dim,
                device=observations.device,
                dtype=torch.float32,
            )
            cfm_noise = torch.cat((half_noise, -half_noise), dim=1)
            half_t = (
                (
                    torch.arange(
                        half_samples,
                        device=observations.device,
                        dtype=torch.float32,
                    )
                    + torch.rand(
                        batch,
                        half_samples,
                        device=observations.device,
                        dtype=torch.float32,
                    )
                )
                / half_samples
            ).unsqueeze(-1)
            cfm_t = torch.cat((half_t, half_t), dim=1)
            online_cfm = agent.cfm_losses(
                policy_features, pretanh_actions, cfm_noise, cfm_t
            )
            with torch.no_grad():
                target_cfm = target_agent.cfm_losses(
                    target_features, pretanh_actions, cfm_noise, cfm_t
                )

            # Coupled pushforward moments: identical noise through both flows.
            moment_noise = torch.randn(
                batch * args.moment_samples,
                agent.action_dim,
                device=observations.device,
                dtype=torch.float32,
            )
            online_paths = agent.integrate(
                policy_features.unsqueeze(1)
                .expand(-1, args.moment_samples, -1)
                .reshape(-1, policy_features.shape[-1]),
                moment_noise,
            ).view(batch, args.moment_samples, agent.action_dim)
            with torch.no_grad():
                target_paths = target_agent.integrate(
                    target_features.unsqueeze(1)
                    .expand(-1, args.moment_samples, -1)
                    .reshape(-1, target_features.shape[-1]),
                    moment_noise,
                ).view(batch, args.moment_samples, agent.action_dim)
        value_log_probs = F.log_softmax(value_logits.float(), dim=-1)
        value_probs = value_log_probs.exp()
        values = hl_support.probs_to_scalar(value_probs)

        alpha_mean = duals[0].clamp_min(DUAL_FLOOR)
        alpha_spread = duals[1].clamp_min(DUAL_FLOOR)

        with torch.no_grad():
            # Match RLax's inclusive kth threshold, including every cutoff tie.
            topk_threshold = torch.sort(advantages).values[-args.topk_size]
            selected = advantages >= topk_threshold
            selected_count = selected.sum().to(advantages.dtype)
            log_selected_count = selected_count.log()

            # Center at the selected maximum before division. This preserves
            # softmax weights and the eta derivative while avoiding a large
            # common logit that would erase near-maximum differences in FP32.
            maximum_advantage = advantages.max()
            centered_advantages = advantages - maximum_advantage
            selected_span = maximum_advantage - topk_threshold

            # selected_span / epsilon is a valid feasible upper bracket: the
            # selected logit range is then at most epsilon, so KL to the
            # selected uniform distribution cannot exceed epsilon. Bisection
            # is geometric because eta can span many orders of magnitude.
            log_eta_low = torch.full_like(topk_threshold, np.log(DUAL_FLOOR))
            log_eta_high = (
                selected_span.div(args.epsilon_eta)
                .clamp_min(DUAL_FLOOR)
                .log()
            )
            for _ in range(32):
                log_eta_mid = 0.5 * (log_eta_low + log_eta_high)
                eta_mid = log_eta_mid.exp()
                mid_logits = torch.where(
                    selected,
                    centered_advantages / eta_mid,
                    -torch.inf,
                )
                mid_log_weights = mid_logits - torch.logsumexp(mid_logits, dim=0)
                mid_weights = mid_log_weights.exp()
                safe_mid_log_weights = torch.where(selected, mid_log_weights, 0.0)
                mid_kl = (
                    mid_weights
                    * (safe_mid_log_weights + log_selected_count)
                ).sum()
                log_eta_low = torch.where(
                    mid_kl > args.epsilon_eta,
                    log_eta_mid,
                    log_eta_low,
                )
                log_eta_high = torch.where(
                    mid_kl > args.epsilon_eta,
                    log_eta_high,
                    log_eta_mid,
                )
            eta = log_eta_high.exp()

            policy_logits = torch.where(
                selected,
                centered_advantages / eta,
                -torch.inf,
            )
            log_weights = policy_logits - torch.logsumexp(policy_logits, dim=0)
            weights = log_weights.exp()
            selected_log_weights = torch.where(selected, log_weights, 0.0)
            temperature_kl = (
                weights * (selected_log_weights + log_selected_count)
            ).sum()
            temperature_loss = maximum_advantage + eta * (
                args.epsilon_eta
                + torch.logsumexp(policy_logits, dim=0)
                - log_selected_count
            )

        # FPO ELBO ratio on shared pairs, exactly one at promotion. Zero-sum
        # weights remove the flow's self-fit drift (see header); the ratio
        # keeps that control variate unbiased while online drifts from target.
        log_ratio = target_cfm - online_cfm
        ratio = log_ratio.exp()
        centered_weights = weights - 1.0 / batch
        policy_loss = -(centered_weights * ratio).sum()

        online_mean, online_variance = flow_moments(online_paths)
        target_mean, target_variance = flow_moments(target_paths)
        mean_kl, spread_kl = decoupled_moment_kl(
            target_mean, target_variance, online_mean, online_variance
        )
        mean_kl_average = mean_kl.mean()
        spread_kl_average = spread_kl.mean()
        mean_constraint_loss = (
            alpha_mean * (args.epsilon_alpha_mean - mean_kl_average.detach())
            + alpha_mean.detach() * mean_kl_average
        )
        spread_constraint_loss = (
            alpha_spread * (args.epsilon_alpha_spread - spread_kl_average.detach())
            + alpha_spread.detach() * spread_kl_average
        )
        with torch.no_grad():
            value_target_probs = hl_support.project_moment_matched(value_targets)
        value_loss = -(value_target_probs * value_log_probs).sum(dim=-1).mean()
        total_loss = (
            policy_loss
            + mean_constraint_loss
            + spread_constraint_loss
            + value_loss
        )

        with torch.no_grad():
            full_moment_kl = (mean_kl + spread_kl).mean()
            effective_sample_size = weights.square().sum().reciprocal()
            effective_sample_fraction = effective_sample_size / selected_count
            eta_stationarity = args.epsilon_eta - temperature_kl
            mean_kl_residual = mean_kl_average - args.epsilon_alpha_mean
            spread_kl_residual = spread_kl_average - args.epsilon_alpha_spread
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
            elbo_kl = -log_ratio.mean()
            weighted_elbo_kl = -(weights * log_ratio).sum()
            ratio_std = ratio.std()
            ratio_max = ratio.max()
            log_ratio_abs_mean = log_ratio.abs().mean()
            selected_ratio_gap = (ratio * selected).sum() / selected_count - (
                ratio * ~selected
            ).sum() / (batch - selected_count)
            pretanh_spread = online_variance.sqrt().mean()
            pretanh_mean_abs = online_mean.abs().mean()
            pretanh_abs_max = pretanh_actions.abs().max()
            action_saturation = (
                torch.tanh(pretanh_actions).abs() > SATURATION_LEVEL
            ).float().mean()
            flow_displacement = (online_paths - moment_noise.view_as(online_paths)).norm(
                dim=-1
            ).mean()
            metrics = torch.stack(
                (
                    policy_loss,
                    value_loss,
                    temperature_loss,
                    mean_kl_average,
                    spread_kl_average,
                    full_moment_kl,
                    effective_sample_fraction,
                    topk_threshold,
                    advantages.mean(),
                    advantages.std(),
                    temperature_kl,
                    eta_stationarity,
                    (-temperature_kl).exp(),
                    weights.max(),
                    effective_sample_size,
                    mean_kl_residual,
                    spread_kl_residual,
                    value_rmse,
                    explained_variance,
                    target_outside_support,
                    target_edge_mass,
                    prediction_edge_mass,
                    online_cfm.mean(),
                    target_cfm.mean(),
                    elbo_kl,
                    weighted_elbo_kl,
                    ratio_std,
                    ratio_max,
                    log_ratio_abs_mean,
                    selected_ratio_gap,
                    pretanh_spread,
                    pretanh_mean_abs,
                    pretanh_abs_max,
                    action_saturation,
                    flow_displacement,
                    eta,
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
    discounted_returns = np.zeros(args.num_envs, dtype=np.float64)
    reward_return_means = np.zeros(args.num_envs, dtype=np.float64)
    reward_return_variances = np.ones(args.num_envs, dtype=np.float64)
    reward_return_counts = np.full(args.num_envs, 1e-4, dtype=np.float64)
    rollout_shape = (args.num_steps, args.num_envs)
    observations = torch.empty(rollout_shape + observation_shape, device=device)
    pretanh_actions = torch.empty(rollout_shape + action_shape, device=device)
    rewards = torch.empty(rollout_shape, device=device)
    values = torch.empty(rollout_shape, device=device)
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
            _, warmup_action, _ = rollout_model(warmup_observation)
        (
            raw_next_observations,
            warmup_reward,
            warmup_terminations,
            warmup_truncations,
            warmup_infos,
        ) = envs.step(warmup_action.cpu().numpy())
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

    # Current partial episodes contain warmup rewards. Suppress exactly their
    # first completion so every reported episode is entirely post-warmup.
    suppress_next_episode_log = phase_offsets > 0

    global_step = warmup_transitions
    next_observation = torch.as_tensor(
        next_observation_np,
        device=device,
        dtype=torch.float32,
    )
    target_age_batches = 0
    target_promotions = 0


    for iteration in range(1, args.num_iterations + 1):
        rollout_target_age_batches = target_age_batches
        for step in range(args.num_steps):
            observations[step].copy_(next_observation)
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                pretanh_action, action, value = rollout_model(next_observation)
            pretanh_actions[step].copy_(pretanh_action)
            values[step].copy_(value)

            (
                raw_next_observations,
                raw_reward,
                terminations,
                truncations,
                infos,
            ) = envs.step(action.cpu().numpy())
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

        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        total_loss, metrics = update_loss_model(
            observations.reshape((args.batch_size,) + observation_shape),
            pretanh_actions.reshape((args.batch_size,) + action_shape),
            advantages,
            value_targets,
        )
        should_log = iteration % args.log_interval == 0 or iteration == 1
        duals_before = duals.detach().clone() if should_log else None
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            duals.clamp_(min=DUAL_FLOOR)
        target_age_batches += 1

        if should_log:
            assert duals_before is not None
            dual_delta = duals.detach() - duals_before
            packed = torch.cat(
                (
                    metrics,
                    duals.detach(),
                    dual_delta,
                    return_percentile_scale.view(1),
                )
            ).cpu().tolist()
            metric_values = dict(zip(METRIC_NAMES, packed[: len(METRIC_NAMES)]))
            (
                alpha_mean_value,
                alpha_spread_value,
                alpha_mean_delta_value,
                alpha_spread_delta_value,
                return_percentile_scale_value,
            ) = packed[len(METRIC_NAMES) :]
            promote_for_mean_kl = (
                iteration % args.log_interval == 0
                and metric_values["vmpo/mean_kl"] >= args.epsilon_alpha_mean
            )
            promote_for_max_age = (
                target_age_batches >= args.target_update_period
            )
            target_promoted = promote_for_mean_kl or promote_for_max_age
            if target_promoted:
                with torch.no_grad():
                    target_agent.load_state_dict(agent.state_dict())
                target_age_batches = 0
                target_promotions += 1
            sps = int(global_step / (time.time() - start_time))
            for name, value in metric_values.items():
                writer.add_scalar(name, value, global_step)
            writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("vmpo/alpha_mean", alpha_mean_value, global_step)
            writer.add_scalar("vmpo/alpha_spread", alpha_spread_value, global_step)
            writer.add_scalar(
                "vmpo/alpha_mean_delta", alpha_mean_delta_value, global_step
            )
            writer.add_scalar(
                "vmpo/alpha_spread_delta", alpha_spread_delta_value, global_step
            )
            writer.add_scalar(
                "vmpo/target_age_batches",
                rollout_target_age_batches,
                global_step,
            )
            writer.add_scalar(
                "vmpo/target_age_transitions",
                rollout_target_age_batches * args.batch_size,
                global_step,
            )
            writer.add_scalar(
                "vmpo/target_promoted",
                float(target_promoted),
                global_step,
            )
            writer.add_scalar(
                "vmpo/target_promoted_for_mean_kl",
                float(promote_for_mean_kl),
                global_step,
            )
            writer.add_scalar(
                "vmpo/target_promoted_for_max_age",
                float(promote_for_max_age),
                global_step,
            )
            writer.add_scalar(
                "vmpo/target_promotions",
                target_promotions,
                global_step,
            )
            writer.add_scalar("vmpo/learner_updates", iteration, global_step)
            writer.add_scalar(
                "debug/return_percentile_scale",
                return_percentile_scale_value,
                global_step,
            )
            print(f"SPS: {sps}")

    envs.close()
    writer.close()

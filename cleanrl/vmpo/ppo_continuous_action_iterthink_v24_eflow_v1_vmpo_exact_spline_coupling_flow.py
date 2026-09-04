# EFLOW v1 — exact-likelihood spline coupling flow on the V-MPO v30 Beta head.
#
# Why not FPO here (fpo_v1..v5 post-mortems in FAMILY.md): V-MPO takes one
# Adam step per rollout batch, about 3200 steps per 8M run and ten per target
# promotion. The M-step must move the policy a full trust-region budget in
# those ten steps from the E-step's tilted weights. That needs an exact
# score, a bounded weighted-MLE objective, no self-fit drift, and a head whose
# parameters map directly onto the distribution. The CFM ELBO ratio has none
# of these in six dimensions, and FPO exists for policies whose likelihood is
# intractable. Ours is not: a conditional normalizing flow gives exact
# log-density in both directions.
#
# Policy: a_native = T_theta(u; s), u ~ Beta(alpha(s), beta(s)) per dimension,
# T a stack of coupling layers, each a monotone rational-quadratic spline on
# [0, 1] (Durkan et al. 2019) whose knots come from an MLP on the policy
# features and the untransformed half. Domain (0, 1) is preserved, so the
# Beta's edge mass and v30's action mapping are untouched. The last conditioner
# layer is zero-initialised, so T is the identity at start and the run begins
# exactly as v30; the transforms learn only where the weighted NLL asks.
#
# Exact quantities, no Monte Carlo in the M-step:
#   log pi(a|s) = log Beta(T^-1(a)) + log|det dT^-1/da|
# Trust region: v30's decoupled Beta KL on the base is exact for the
# pushforward under the target transform (KL is invariant under a shared
# diffeomorphism), so mean and concentration budgets keep their meaning. The
# transform gets its own decoupled component, KL(T_t # B_t || T_o # B_t),
# estimated on K=8 fresh base draws per state through the frozen target flow
# (exact log-densities, only the sampling is Monte Carlo), with the same dual
# mechanics as the other two. Promotion fires on either the base mean or the
# transform component reaching the mean budget.
#
# Hypothesis: with exact likelihoods the expressive head costs v30 nothing
# in cadence, and per-state cross-dimensional dependence and skew that the
# factorised Beta cannot express is where any gain over v30 must come from.
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

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6
DUAL_FLOOR = 1e-8
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
    epsilon_alpha_concentration: float = 1.5811388300841898e-5
    initial_alpha_mean: float = 1.0
    initial_alpha_concentration: float = 1.0
    epsilon_alpha_transform: float = 0.007071067811865476
    initial_alpha_transform: float = 1.0
    return_percentile_low: float = 0.05
    return_percentile_high: float = 0.95
    return_percentile_floor: float = 1.0
    num_value_bins: int = 51
    value_support_limit: float = 20_000.0
    value_sigma_to_bin_ratio: float = 0.75


    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # Spline coupling flow on the Beta base.
    coupling_layers: int = 3
    spline_bins: int = 8
    coupling_hidden: int = 128
    # Fresh base draws per state for the transform KL component.
    kl_samples: int = 8

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


SPLINE_MIN_BIN = 1e-3
SPLINE_MIN_DERIVATIVE = 1e-3


def rational_quadratic_spline(inputs, widths_raw, heights_raw, derivatives_raw, inverse):
    """Monotone rational-quadratic spline on [0, 1] (Durkan et al. 2019).

    All raw parameter tensors broadcast against ``inputs[..., None]``; zero raw
    parameters give the identity map with zero log-determinant. Returns the
    mapped values and the log absolute derivative of the applied direction.
    """
    bins = widths_raw.shape[-1]
    widths = F.softmax(widths_raw, dim=-1) * (1.0 - bins * SPLINE_MIN_BIN) + SPLINE_MIN_BIN
    heights = F.softmax(heights_raw, dim=-1) * (1.0 - bins * SPLINE_MIN_BIN) + SPLINE_MIN_BIN
    unit_offset = float(np.log(np.expm1(1.0 - SPLINE_MIN_DERIVATIVE)))
    derivatives = SPLINE_MIN_DERIVATIVE + F.softplus(derivatives_raw + unit_offset)

    cum_widths = torch.cumsum(widths, dim=-1)
    cum_widths = F.pad(cum_widths, (1, 0))
    cum_heights = torch.cumsum(heights, dim=-1)
    cum_heights = F.pad(cum_heights, (1, 0))

    x = inputs.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS).unsqueeze(-1)
    knots = cum_heights if inverse else cum_widths
    index = (x >= knots[..., 1:-1]).sum(dim=-1, keepdim=True)
    width = torch.gather(widths, -1, index)
    height = torch.gather(heights, -1, index)
    left = torch.gather(cum_widths, -1, index)
    bottom = torch.gather(cum_heights, -1, index)
    derivative_left = torch.gather(derivatives, -1, index)
    derivative_right = torch.gather(derivatives, -1, index + 1)
    slope = height / width
    curvature = derivative_right + derivative_left - 2.0 * slope

    if inverse:
        offset = x - bottom
        a = height * (slope - derivative_left) + offset * curvature
        b = height * derivative_left - offset * curvature
        c = -slope * offset
        discriminant = (b.square() - 4.0 * a * c).clamp_min(0.0)
        theta = 2.0 * c / (-b - discriminant.sqrt())
        outputs = left + theta * width
    else:
        theta = (x - left) / width
    theta_complement = theta * (1.0 - theta)
    denominator = slope + curvature * theta_complement
    if not inverse:
        outputs = bottom + height * (
            slope * theta.square() + derivative_left * theta_complement
        ) / denominator
    log_derivative = (
        2.0 * slope.log()
        + (
            derivative_right * theta.square()
            + 2.0 * slope * theta_complement
            + derivative_left * (1.0 - theta).square()
        ).log()
        - 2.0 * denominator.log()
    )
    if inverse:
        log_derivative = -log_derivative
    return outputs.squeeze(-1), log_derivative.squeeze(-1)


class SplineCouplingFlow(nn.Module):
    """Conditional coupling flow of rational-quadratic splines on (0, 1)^d.

    Layer l transforms the dimensions with (dim + l) % 2 == 0 given the policy
    context and the untransformed dimensions. Conditioner output layers are
    zero-initialised, so the flow starts as the identity.
    """

    def __init__(self, action_dim, context_dim, layers, bins, hidden):
        super().__init__()
        if layers <= 0 or bins <= 0 or action_dim < 2:
            raise ValueError("coupling flow needs layers, bins, and at least two dims")
        self.action_dim = action_dim
        self.bins = bins
        masks = torch.stack(
            [
                (torch.arange(action_dim) + layer) % 2 == 0
                for layer in range(layers)
            ]
        )
        self.register_buffer("masks", masks)
        params_per_dim = 3 * bins + 1
        self.conditioners = nn.ModuleList()
        for _ in range(layers):
            output = nn.Linear(hidden, action_dim * params_per_dim)
            with torch.no_grad():
                output.weight.zero_()
                output.bias.zero_()
            self.conditioners.append(
                nn.Sequential(
                    layer_init(nn.Linear(context_dim + action_dim, hidden)),
                    ReLUSquared(),
                    output,
                )
            )

    def _layer(self, layer, x, context, inverse):
        mask = self.masks[layer]
        conditioner_input = torch.cat(
            (context, torch.where(mask, 0.0, x).to(context.dtype)), dim=-1
        )
        params = self.conditioners[layer](conditioner_input).to(x.dtype)
        params = params.view(*x.shape, 3 * self.bins + 1)
        widths_raw, heights_raw, derivatives_raw = torch.split(
            params, (self.bins, self.bins, self.bins + 1), dim=-1
        )
        transformed, log_derivative = rational_quadratic_spline(
            x, widths_raw, heights_raw, derivatives_raw, inverse
        )
        outputs = torch.where(mask, transformed, x)
        log_abs_det = torch.where(mask, log_derivative, 0.0).sum(dim=-1)
        return outputs, log_abs_det

    def forward(self, base, context):
        """Base sample u -> native action a with log|det dT/du|."""
        x = base
        log_abs_det = torch.zeros(base.shape[:-1], device=base.device)
        for layer in range(len(self.conditioners)):
            x, layer_log_abs_det = self._layer(layer, x, context, inverse=False)
            log_abs_det = log_abs_det + layer_log_abs_det
        return x, log_abs_det

    def inverse(self, action, context):
        """Native action a -> base sample u with log|det dT^-1/da|."""
        x = action
        log_abs_det = torch.zeros(action.shape[:-1], device=action.device)
        for layer in reversed(range(len(self.conditioners))):
            x, layer_log_abs_det = self._layer(layer, x, context, inverse=True)
            log_abs_det = log_abs_det + layer_log_abs_det
        return x, log_abs_det


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor

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
        self.flow = SplineCouplingFlow(
            action_dim,
            256,
            args.coupling_layers,
            args.spline_bins,
            args.coupling_hidden,
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

    def policy(self, observations):
        features = self.policy_mlp(self.trunk(observations))
        alpha = 1.0 + F.softplus(self.actor_alpha(features))
        beta = 1.0 + F.softplus(self.actor_beta(features))
        return alpha, beta, features

    def forward(self, observations):
        features = self.trunk(observations)
        policy_features = self.policy_mlp(features)
        alpha = 1.0 + F.softplus(self.actor_alpha(policy_features))
        beta = 1.0 + F.softplus(self.actor_beta(policy_features))
        value_logits = self.value_head(self.value_mlp(features))
        return alpha, beta, policy_features, value_logits

    def value_logits(self, observations):
        features = self.value_mlp(self.trunk(observations))
        return self.value_head(features)



def beta_log_prob(alpha, beta, action):
    action = action.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    log_normalizer = (
        torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    )
    return (
        (alpha - 1.0) * action.log()
        + (beta - 1.0) * torch.log1p(-action)
        - log_normalizer
    ).sum(-1)


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


def decoupled_beta_kl(old_alpha, old_beta, new_alpha, new_beta):
    old_concentration = old_alpha + old_beta
    new_concentration = new_alpha + new_beta
    old_mean = old_alpha / old_concentration
    new_mean = new_alpha / new_concentration

    mean_alpha = (new_mean * old_concentration).clamp_min(SAMPLE_EPS)
    mean_beta = ((1.0 - new_mean) * old_concentration).clamp_min(SAMPLE_EPS)
    concentration_alpha = (old_mean * new_concentration).clamp_min(SAMPLE_EPS)
    concentration_beta = ((1.0 - old_mean) * new_concentration).clamp_min(SAMPLE_EPS)
    mean_kl = beta_kl(old_alpha, old_beta, mean_alpha, mean_beta)
    concentration_kl = beta_kl(
        old_alpha, old_beta, concentration_alpha, concentration_beta
    )
    return mean_kl, concentration_kl


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
                args.initial_alpha_concentration,
                args.initial_alpha_transform,
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
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            target_alpha, target_beta, target_context = target_agent.policy(
                observations
            )
            value_logits = agent.value_logits(observations)
        values = hl_support.to_scalar(value_logits.float())
        return target_alpha.float(), target_beta.float(), target_context, values

    def transform_model(context, alpha, beta, base):
        """Push a Beta base draw through the target flow with its exact log-density."""
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            native_action, log_abs_det = target_agent.flow(base, context)
        native_action = native_action.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        log_prob = beta_log_prob(alpha, beta, base) - log_abs_det
        return native_action, log_prob


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
        native_actions,
        old_alpha,
        old_beta,
        old_log_probs,
        kl_base_samples,
        advantages,
        value_targets,
    ):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            new_alpha, new_beta, new_context, value_logits = agent(observations)
            with torch.no_grad():
                _, _, old_context = target_agent.policy(observations)
            base_actions, inverse_log_abs_det = agent.flow.inverse(
                native_actions, new_context
            )
            # Transform component: K fresh target-base draws through the
            # frozen target flow, scored exactly under target and online flows.
            kl_count = kl_base_samples.shape[1]
            with torch.no_grad():
                kl_actions, target_forward_log_abs_det = target_agent.flow(
                    kl_base_samples,
                    old_context.unsqueeze(1).expand(-1, kl_count, -1),
                )
                kl_actions = kl_actions.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            kl_base_online, online_inverse_log_abs_det = agent.flow.inverse(
                kl_actions, new_context.unsqueeze(1).expand(-1, kl_count, -1)
            )
        new_alpha = new_alpha.float()
        new_beta = new_beta.float()
        old_alpha_expanded = old_alpha.unsqueeze(1)
        old_beta_expanded = old_beta.unsqueeze(1)
        with torch.no_grad():
            kl_target_log_prob = (
                beta_log_prob(old_alpha_expanded, old_beta_expanded, kl_base_samples)
                - target_forward_log_abs_det
            )
        kl_transform_log_prob = (
            beta_log_prob(old_alpha_expanded, old_beta_expanded, kl_base_online)
            + online_inverse_log_abs_det
        )
        kl_online_log_prob = (
            beta_log_prob(new_alpha.unsqueeze(1), new_beta.unsqueeze(1), kl_base_online)
            + online_inverse_log_abs_det
        )
        transform_kl_average = (kl_target_log_prob - kl_transform_log_prob).mean()
        value_log_probs = F.log_softmax(value_logits.float(), dim=-1)
        value_probs = value_log_probs.exp()
        values = hl_support.probs_to_scalar(value_probs)

        alpha_mean = duals[0].clamp_min(DUAL_FLOOR)
        alpha_concentration = duals[1].clamp_min(DUAL_FLOOR)
        alpha_transform = duals[2].clamp_min(DUAL_FLOOR)

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

        log_prob = beta_log_prob(new_alpha, new_beta, base_actions) + inverse_log_abs_det
        policy_loss = -(weights * log_prob).sum()

        mean_kl, concentration_kl = decoupled_beta_kl(
            old_alpha, old_beta, new_alpha, new_beta
        )
        mean_kl_average = mean_kl.mean()
        concentration_kl_average = concentration_kl.mean()
        mean_constraint_loss = (
            alpha_mean * (args.epsilon_alpha_mean - mean_kl_average.detach())
            + alpha_mean.detach() * mean_kl_average
        )
        concentration_constraint_loss = (
            alpha_concentration
            * (
                args.epsilon_alpha_concentration
                - concentration_kl_average.detach()
            )
            + alpha_concentration.detach() * concentration_kl_average
        )
        transform_constraint_loss = (
            alpha_transform
            * (args.epsilon_alpha_transform - transform_kl_average.detach())
            + alpha_transform.detach() * transform_kl_average
        )
        with torch.no_grad():
            value_target_probs = hl_support.project_moment_matched(value_targets)
        value_loss = -(value_target_probs * value_log_probs).sum(dim=-1).mean()
        total_loss = (
            policy_loss
            + mean_constraint_loss
            + concentration_constraint_loss
            + transform_constraint_loss
            + value_loss
        )

        full_kl = beta_kl(old_alpha, old_beta, new_alpha, new_beta).mean()
        full_flow_kl = (kl_target_log_prob - kl_online_log_prob).mean()
        transform_kl_residual = transform_kl_average - args.epsilon_alpha_transform
        policy_entropy = -old_log_probs.mean()
        transform_log_abs_det = target_forward_log_abs_det.abs().mean()
        action_saturation = (
            (native_actions < 1.0 - SATURATION_LEVEL)
            | (native_actions > SATURATION_LEVEL)
        ).float().mean()
        effective_sample_size = weights.square().sum().reciprocal()
        effective_sample_fraction = effective_sample_size / selected_count
        eta_stationarity = args.epsilon_eta - temperature_kl
        mean_kl_residual = mean_kl_average - args.epsilon_alpha_mean
        concentration_kl_residual = (
            concentration_kl_average - args.epsilon_alpha_concentration
        )
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
                value_loss.detach(),
                temperature_loss.detach(),
                mean_kl_average.detach(),
                concentration_kl_average.detach(),
                full_kl.detach(),
                effective_sample_fraction.detach(),
                topk_threshold.detach(),
                advantages.mean().detach(),
                advantages.std().detach(),
                temperature_kl.detach(),
                eta_stationarity.detach(),
                (-temperature_kl).exp().detach(),
                weights.max().detach(),
                effective_sample_size.detach(),
                mean_kl_residual.detach(),
                concentration_kl_residual.detach(),
                value_rmse.detach(),
                explained_variance.detach(),
                target_outside_support.detach(),
                target_edge_mass.detach(),
                prediction_edge_mass.detach(),
                policy_concentration.detach(),
                policy_variance.detach(),
                eta.detach(),
                transform_kl_average.detach(),
                transform_kl_residual.detach(),
                full_flow_kl.detach(),
                policy_entropy.detach(),
                transform_log_abs_det.detach(),
                action_saturation.detach(),
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
        transform_model = torch.compile(
            transform_model,
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
    native_actions = torch.empty(rollout_shape + action_shape, device=device)
    rewards = torch.empty(rollout_shape, device=device)
    values = torch.empty(rollout_shape, device=device)
    old_alphas = torch.empty(rollout_shape + action_shape, device=device)
    old_betas = torch.empty_like(old_alphas)
    old_log_probs = torch.empty(rollout_shape, device=device)
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
            warmup_alpha, warmup_beta, warmup_context, _ = rollout_model(
                warmup_observation
            )
            warmup_base = Beta(warmup_alpha, warmup_beta).sample().clamp(
                SAMPLE_EPS,
                1.0 - SAMPLE_EPS,
            )
            warmup_native_action, _ = transform_model(
                warmup_context, warmup_alpha, warmup_beta, warmup_base
            )
            warmup_action = target_agent.action_low + (
                target_agent.action_high - target_agent.action_low
            ) * warmup_native_action
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
                old_alpha, old_beta, old_context, value = rollout_model(
                    next_observation
                )
                base_sample = Beta(old_alpha, old_beta).sample().clamp(
                    SAMPLE_EPS, 1.0 - SAMPLE_EPS
                )
                native_action, old_log_prob = transform_model(
                    old_context, old_alpha, old_beta, base_sample
                )
                action = target_agent.action_low + (
                    target_agent.action_high - target_agent.action_low
                ) * native_action
            native_actions[step].copy_(native_action)
            old_alphas[step].copy_(old_alpha)
            old_betas[step].copy_(old_beta)
            old_log_probs[step].copy_(old_log_prob)
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

        flat_old_alphas = old_alphas.reshape((args.batch_size,) + action_shape)
        flat_old_betas = old_betas.reshape((args.batch_size,) + action_shape)
        with torch.no_grad():
            kl_base_samples = (
                Beta(flat_old_alphas, flat_old_betas)
                .sample((args.kl_samples,))
                .transpose(0, 1)
                .clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                .contiguous()
            )
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        total_loss, metrics = update_loss_model(
            observations.reshape((args.batch_size,) + observation_shape),
            native_actions.reshape((args.batch_size,) + action_shape),
            flat_old_alphas,
            flat_old_betas,
            old_log_probs.reshape(args.batch_size),
            kl_base_samples,
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
            (
                policy_loss_value,
                value_loss_value,
                temperature_loss_value,
                mean_kl_value,
                concentration_kl_value,
                full_kl_value,
                ess_fraction_value,
                top_advantage_min,
                advantage_mean,
                advantage_std,
                temperature_kl_value,
                eta_stationarity_value,
                perplexity_fraction_value,
                max_weight_value,
                ess_value,
                mean_kl_residual_value,
                concentration_kl_residual_value,
                value_rmse_value,
                explained_variance_value,
                target_outside_support_value,
                target_edge_mass_value,
                prediction_edge_mass_value,
                policy_concentration_value,
                policy_variance_value,
                eta_value,
                transform_kl_value,
                transform_kl_residual_value,
                full_flow_kl_value,
                policy_entropy_value,
                transform_log_abs_det_value,
                action_saturation_value,
                alpha_mean_value,
                alpha_concentration_value,
                alpha_transform_value,
                alpha_mean_delta_value,
                alpha_concentration_delta_value,
                alpha_transform_delta_value,
                return_percentile_scale_value,
            ) = packed
            promote_for_mean_kl = iteration % args.log_interval == 0 and (
                mean_kl_value >= args.epsilon_alpha_mean
                or transform_kl_value >= args.epsilon_alpha_transform
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
            writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/policy_loss", policy_loss_value, global_step)
            writer.add_scalar("losses/value_loss", value_loss_value, global_step)
            writer.add_scalar("losses/temperature_loss", temperature_loss_value, global_step)
            writer.add_scalar("vmpo/eta", eta_value, global_step)
            writer.add_scalar("vmpo/alpha_mean", alpha_mean_value, global_step)
            writer.add_scalar(
                "vmpo/alpha_concentration", alpha_concentration_value, global_step
            )
            writer.add_scalar("vmpo/mean_kl", mean_kl_value, global_step)
            writer.add_scalar("vmpo/transform_kl", transform_kl_value, global_step)
            writer.add_scalar(
                "vmpo/transform_kl_residual", transform_kl_residual_value, global_step
            )
            writer.add_scalar("vmpo/full_flow_kl", full_flow_kl_value, global_step)
            writer.add_scalar("vmpo/alpha_transform", alpha_transform_value, global_step)
            writer.add_scalar(
                "vmpo/alpha_transform_delta", alpha_transform_delta_value, global_step
            )
            writer.add_scalar("flow/policy_entropy", policy_entropy_value, global_step)
            writer.add_scalar(
                "flow/transform_log_abs_det", transform_log_abs_det_value, global_step
            )
            writer.add_scalar("flow/action_saturation", action_saturation_value, global_step)
            writer.add_scalar(
                "vmpo/concentration_kl", concentration_kl_value, global_step
            )
            writer.add_scalar("vmpo/full_beta_kl", full_kl_value, global_step)
            writer.add_scalar(
                "vmpo/weight_ess_fraction", ess_fraction_value, global_step
            )
            writer.add_scalar("vmpo/weight_ess", ess_value, global_step)
            writer.add_scalar("vmpo/e_step_kl", temperature_kl_value, global_step)
            writer.add_scalar(
                "vmpo/eta_stationarity", eta_stationarity_value, global_step
            )
            writer.add_scalar(
                "vmpo/weight_perplexity_fraction",
                perplexity_fraction_value,
                global_step,
            )
            writer.add_scalar("vmpo/max_weight", max_weight_value, global_step)
            writer.add_scalar(
                "vmpo/mean_kl_residual", mean_kl_residual_value, global_step
            )
            writer.add_scalar(
                "vmpo/concentration_kl_residual",
                concentration_kl_residual_value,
                global_step,
            )
            writer.add_scalar(
                "vmpo/alpha_mean_delta", alpha_mean_delta_value, global_step
            )
            writer.add_scalar(
                "vmpo/alpha_concentration_delta",
                alpha_concentration_delta_value,
                global_step,
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
            writer.add_scalar("vmpo/top_advantage_min", top_advantage_min, global_step)
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
            writer.add_scalar(
                "debug/policy_concentration", policy_concentration_value, global_step
            )
            writer.add_scalar(
                "debug/policy_native_variance", policy_variance_value, global_step
            )
            print(f"SPS: {sps}")

    envs.close()
    writer.close()

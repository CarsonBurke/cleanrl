# V-MPO v39 — historical-window denoised trust-region V-MPO.
#
# Derived from v30 (reward-normalized Dreamer moment-matched HL-Gauss critic).
# v30 spends each rollout batch on one Adam step and forgets it. Here the
# learner's memory is the data itself: every update sees a rolling window of
# the last `window_batches` rollouts, grounded exactly to the current target
# policy through stored behaviour Beta densities.
#
#   Grounded historical targets. Advantages and value targets are recomputed
#   for the whole window every update with the current critic and V-trace
#   (lambda, clipped rho/c) against the current policy. The E-step's top-k
#   selection and temperature are solved over the window with an
#   importance-weighted proposal p_i ~ pi(a_i|s_i)/pi_behaviour(a_i|s_i), so
#   the M-step regresses onto a target built from ~100x more samples than a
#   batch. The target is promoted every update: each step starts from its own
#   policy, the E-step is rebuilt on it, and the improvement direction is
#   renewed instead of exhausted against a stale target. The KL dual of the
#   M-step is therefore gone; the cross-validated prior precision below is
#   the trust region, and the E-step epsilon is set for signal (0.1), since
#   weight variance is handled by the denoiser rather than by a tiny epsilon.
#
#   Gradient denoising from data first, time second. The window gradient is
#   computed as G equal, time-interleaved group gradients. Two half-window
#   gradients each seed a Lanczos Krylov basis of the exact Gauss-Newton
#   Fisher of the Beta policy and categorical critic on a time-stratified
#   subsample. The prior precision (damping) is not a hyperparameter: on a
#   log grid relative to the top Ritz value, the damped solve of one half is
#   scored against the other half's gradient, and the damping maximizing the
#   cross-fitted signal per unit model KL is chosen. The cross term
#   g_b^T (F+lam)^-1 g_a is an unbiased estimate of the signal quadratic
#   while the self terms carry the noise, so their ratio is the Wiener gain
#   of the step measured in the KL metric. A window carries a bounded amount
#   of signal, so each half can also keep EMAs over several horizons in
#   updates; the halves stay disjoint in data, the EMAs stay independent, and
#   the same criterion picks the horizon. Measured on HalfCheetah the longer
#   horizons score negative: a Newton step consumes its signal, so the lagged
#   gradient opposes the next one. The default keeps the bare window.
#
#   No learning rate. The metric is budget-normalized: the Beta policy
#   Fisher over step_policy_kl, the Beta concentration Fisher (V-MPO's
#   decoupled constraint, which stops the policy sharpening before it has
#   moved) over step_concentration_kl, and the categorical critic Fisher over
#   step_value_kl, so a unit of model KL is a unit of budget. Along the
#   natural direction the loss is modelled as -s*signal + s^2*quadratic; the
#   step is its minimizer (the Newton step shrunk by the Wiener gain),
#   realized by a down-only two-refinement line search that caps each of the
#   three KLs exactly. fp32 forwards make the KL measurements exact at 1e-6.
#
#   Exploration as a separate control. Estimation noise is removed, so
#   exploration is re-injected deliberately in two places. Each environment's
#   behaviour policy is the target with a per-episode parameter perturbation
#   scaled by the inverse estimated gradient power (posterior-metric noise);
#   its global sigma is calibrated each update from the measured per-state KL
#   per sigma^2 to a KL budget. The learner step can carry a Langevin term of
#   the same shape at its own KL budget (learner_explore_kl), calibrated by
#   the same coefficient: the persisted random movement a noisy optimizer
#   supplies for free and the next E-step selects on. Measured on HalfCheetah
#   at 1e-3 per update it slowed learning (1158 vs 1622 at 3M), so it is off
#   by default. Budgets anneal to zero over the final fraction. Stored
#   behaviour densities keep this exactly grounded; a clean fraction of
#   environments acts unperturbed for reference returns.
#
# Hypothesis: with the step's drift denoised the policy improves per rollout
# at the rate its data supports, without a learning rate, with the critic fit
# from the whole window rather than a batch.
#
# Result (HalfCheetah, seed 1, 8M): the cross-fitted signal of a 80k-sample
# window is small and bursty; most updates carry no detectable signal and the
# Newton-Wiener step is ~0, while v30's Adam moves ~1.5e-3 KL per update
# regardless. The denoised learner is honest but slower than v30 (which
# reaches ~7.7k).

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
import tyro
from torch.distributions.beta import Beta
from torch.func import functional_call, grad, jvp, vjp, vmap
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6
ETA_FLOOR = 1e-8
POWER_EPS = 1e-8
KL_EPS = 1e-12


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
    gae_lambda: float = 0.95

    topk_fraction: float = 0.5
    # Denoised M-step: the E-step trust region is set for signal, not for
    # weight variance, which the cross-validated prior precision absorbs.
    epsilon_eta: float = 0.1
    return_percentile_low: float = 0.05
    return_percentile_high: float = 0.95
    return_percentile_floor: float = 1.0
    num_value_bins: int = 51
    value_support_limit: float = 20_000.0
    value_sigma_to_bin_ratio: float = 0.75

    # Historical window and off-policy correction.
    window_batches: int = 32
    gradient_groups: int = 8
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    # Cross-fitted natural-gradient step under function-space KL budgets.
    krylov_dim: int = 16
    # Gradient memory horizons in updates; 1 is the bare window gradient.
    gradient_horizons: tuple[int, ...] = (1,)
    damping_grid_size: int = 21
    damping_log_min: float = -3.0
    damping_log_max: float = 4.0
    step_policy_kl: float = 1e-2
    # v30 realized about 30:1 mean-to-concentration KL per update; same ratio here.
    step_concentration_kl: float = 3e-4
    step_value_kl: float = 5e-2
    noise_ema: float = 0.9

    # Posterior-metric parameter-space exploration on the behaviour policy.
    explore_kl: float = 0.007071067811865476
    # Persisted Langevin noise on the learner step (per-update KL budget).
    learner_explore_kl: float = 0.0
    explore_clean_fraction: float = 0.125
    explore_anneal_fraction: float = 0.1
    explore_sigma_init: float = 0.01
    explore_calibration_ema: float = 0.9

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    compile: bool = True
    compile_mode: str = "reduce-overhead"
    bf16: bool = True
    log_interval: int = 10

    batch_size: int = 0
    window_size: int = 0
    group_size: int = 0
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


def update_reward_moments(
    rewards,
    terminations,
    discounted_returns,
    return_means,
    return_variances,
    return_counts,
    gamma,
):
    """Match independent NormalizeReward wrapper statistics in one update."""
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


def normalize_vector_step(
    raw_next_observations,
    terminations,
    truncations,
    infos,
    observation_means,
    observation_variances,
    observation_counts,
):
    """Normalize a same-step autoreset batch in per-environment wrapper order.

    Returns the normalized next observations, the raw transition observations
    (final observations substituted at episode boundaries), and the same
    transition observations normalized.
    """
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

    def forward(self, observations):
        features = self.trunk(observations)
        policy_features = self.policy_mlp(features)
        alpha = 1.0 + F.softplus(self.actor_alpha(policy_features))
        beta = 1.0 + F.softplus(self.actor_beta(policy_features))
        value_logits = self.value_head(self.value_mlp(features))
        return alpha, beta, value_logits

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


def categorical_kl(old_logits, new_logits):
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    return (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum(-1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.window_size = args.window_batches * args.batch_size
    args.topk_size = int(args.window_size * args.topk_fraction)
    if not 0 < args.topk_size <= args.window_size:
        raise ValueError("topk_fraction produces an invalid window top-k size")
    if args.window_batches <= 0 or args.gradient_groups <= 0:
        raise ValueError("window_batches and gradient_groups must be positive")
    if args.gradient_groups < 2:
        raise ValueError("gradient noise estimation needs at least two groups")
    if args.window_batches % args.gradient_groups != 0:
        raise ValueError("window_batches must be divisible by gradient_groups")
    args.group_size = args.window_size // args.gradient_groups
    env_spec = gym.spec(args.env_id)
    if env_spec.max_episode_steps is None:
        raise ValueError("phase staggering requires a finite episode horizon")
    args.initial_phase_warmup_steps = int(env_spec.max_episode_steps)
    warmup_transitions = args.num_envs * args.initial_phase_warmup_steps
    if warmup_transitions + args.window_size >= args.total_timesteps:
        raise ValueError(
            "total_timesteps must exceed the initial phase warmup and window fill"
        )
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
    if args.vtrace_rho_clip <= 0.0 or args.vtrace_c_clip <= 0.0:
        raise ValueError("V-trace clips must be positive")
    if min(args.step_policy_kl, args.step_concentration_kl, args.step_value_kl) <= 0.0:
        raise ValueError("per-step KL caps must be positive")
    if not 0.0 <= args.noise_ema < 1.0:
        raise ValueError("noise_ema must be in [0, 1)")
    if min(args.explore_kl, args.learner_explore_kl, args.explore_sigma_init) < 0.0:
        raise ValueError("exploration KL and initial sigma must be non-negative")
    if not 0.0 <= args.explore_clean_fraction <= 1.0:
        raise ValueError("explore_clean_fraction must be in [0, 1]")
    if not 0.0 <= args.explore_anneal_fraction <= 1.0:
        raise ValueError("explore_anneal_fraction must be in [0, 1]")
    if not 0.0 <= args.explore_calibration_ema < 1.0:
        raise ValueError("explore_calibration_ema must be in [0, 1)")
    if args.krylov_dim < 2:
        raise ValueError("krylov_dim must be at least two")
    if not args.gradient_horizons or any(h < 1 for h in args.gradient_horizons):
        raise ValueError("gradient_horizons must be non-empty positive update counts")
    if args.damping_grid_size < 2 or args.damping_log_min >= args.damping_log_max:
        raise ValueError("damping grid must span at least two increasing values")

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
    # Exact fp32 matmuls: the per-step KL line search resolves budgets of 1e-6
    # nats, below the rounding floor of TF32 or BF16 outputs. Heavy paths run
    # under explicit BF16 autocast.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
    if (
        envs.single_observation_space.shape is None
        or envs.single_action_space.shape is None
    ):
        raise ValueError("continuous-control observation and action shapes are required")
    observation_shape = tuple(envs.single_observation_space.shape)
    action_shape = tuple(envs.single_action_space.shape)
    if len(observation_shape) != 1 or len(action_shape) != 1:
        raise ValueError("flat observation and action vectors are required")
    obs_dim = observation_shape[0]
    action_dim = action_shape[0]

    agent = Agent(envs, args).to(device)
    target_agent = copy.deepcopy(agent).requires_grad_(False)

    # Flat parameter bookkeeping. The network is updated through one flat
    # vector; policy-affecting coordinates (everything but the critic head)
    # receive the behaviour-side exploration perturbation.
    param_names = [name for name, _ in agent.named_parameters()]
    param_shapes = [parameter.shape for _, parameter in agent.named_parameters()]
    param_numels = [parameter.numel() for _, parameter in agent.named_parameters()]
    num_params = sum(param_numels)
    policy_param_names = [
        name for name in param_names if not name.startswith("value_")
    ]
    policy_param_shapes = [
        shape for name, shape in zip(param_names, param_shapes) if name in policy_param_names
    ]
    policy_param_numels = [
        numel for name, numel in zip(param_names, param_numels) if name in policy_param_names
    ]
    num_policy_params = sum(policy_param_numels)
    policy_coordinate_mask = torch.cat(
        [
            torch.full((numel,), not name.startswith("value_"), device=device)
            for name, numel in zip(param_names, param_numels)
        ]
    )
    policy_coordinate_indices = policy_coordinate_mask.nonzero().squeeze(1)

    def unflatten_all(flat):
        return {
            name: chunk.view(shape)
            for name, chunk, shape in zip(
                param_names, torch.split(flat, param_numels), param_shapes
            )
        }

    def unflatten_policy(flat):
        return {
            name: chunk.view(shape)
            for name, chunk, shape in zip(
                policy_param_names,
                torch.split(flat, policy_param_numels),
                policy_param_shapes,
            )
        }

    def write_parameters(flat):
        """In-place copy so parameter storage stays static for cudagraphs."""
        for parameter, chunk in zip(agent.parameters(), torch.split(flat, param_numels)):
            parameter.copy_(chunk.view_as(parameter))

    def target_policy_vector():
        return parameters_to_vector(
            [target_agent.get_parameter(name) for name in policy_param_names]
        ).detach()

    target_policy_flat = target_policy_vector().clone()

    autocast_dtype = torch.bfloat16
    num_envs = args.num_envs
    num_steps = args.num_steps
    window_batches = args.window_batches
    num_groups = args.gradient_groups
    num_horizons = len(args.gradient_horizons)
    window_size = args.window_size
    group_size = args.group_size
    fisher_samples = group_size
    gamma = args.gamma

    def rollout_model(policy_flat, perturbations, observations):
        """Per-environment perturbed target policies in one vmapped forward.

        Runs in fp32 so the stored behaviour density is the same function the
        update evaluates for the target: within a target period rho == 1 up
        to observation-statistic drift.
        """

        def single(flat, observation):
            alpha, beta, _ = functional_call(
                target_agent, unflatten_policy(flat), (observation.unsqueeze(0),)
            )
            return alpha.squeeze(0), beta.squeeze(0)

        return vmap(single)(policy_flat + perturbations, observations)

    def grouped(tensor):
        """Interleave window slots into equal, time-stratified groups."""
        trailing = tensor.shape[3:]
        return (
            tensor.view(window_batches // num_groups, num_groups, num_steps, num_envs, *trailing)
            .transpose(0, 1)
            .reshape(num_groups, group_size, *trailing)
        )

    def prepare_model(
        raw_observations,
        raw_next_observations,
        observation_mean,
        observation_variance,
        raw_rewards,
        reward_scale,
        actions,
        behaviour_alpha,
        behaviour_beta,
        terminations,
        boundaries,
        newest_slot_mask,
    ):
        """Ground the whole window to the current target policy and critic."""
        observation_scale = torch.rsqrt(observation_variance + 1e-8)
        observations = (
            (raw_observations - observation_mean) * observation_scale
        ).clamp(-10.0, 10.0)
        next_observations = (
            (raw_next_observations - observation_mean) * observation_scale
        ).clamp(-10.0, 10.0)
        rewards = (raw_rewards * reward_scale).clamp(-10.0, 10.0)
        flat_observations = observations.reshape(window_size, obs_dim)
        flat_next_observations = next_observations.reshape(window_size, obs_dim)
        flat_actions = actions.reshape(window_size, action_dim)
        flat_behaviour_alpha = behaviour_alpha.reshape(window_size, action_dim)
        flat_behaviour_beta = behaviour_beta.reshape(window_size, action_dim)

        target_alpha, target_beta, _ = target_agent(flat_observations)
        log_rho = beta_log_prob(target_alpha, target_beta, flat_actions) - beta_log_prob(
            flat_behaviour_alpha, flat_behaviour_beta, flat_actions
        )
        values = hl_support.to_scalar(agent.value_logits(flat_observations))
        next_values = hl_support.to_scalar(agent.value_logits(flat_next_observations))
        values = values.view(window_batches, num_steps, num_envs)
        next_values = next_values.view(window_batches, num_steps, num_envs)
        rho = log_rho.exp().view(window_batches, num_steps, num_envs)
        clipped_rho = rho.clamp_max(args.vtrace_rho_clip)
        trace = args.gae_lambda * rho.clamp_max(args.vtrace_c_clip)

        # V-trace(lambda) on stored segments: with rho == 1 this is exactly the
        # v30 GAE recursion, with the bootstrap masked at terminations and the
        # trace cut at every episode boundary.
        deltas = clipped_rho * (
            rewards + gamma * next_values * (1.0 - terminations) - values
        )
        advantages = torch.empty_like(deltas)
        running_advantage = torch.zeros_like(deltas[:, -1])
        for reverse_step in reversed(range(num_steps)):
            running_advantage = deltas[:, reverse_step] + gamma * trace[
                :, reverse_step
            ] * torch.where(boundaries[:, reverse_step], 0.0, running_advantage)
            advantages[:, reverse_step] = running_advantage
        value_targets = (values + advantages).reshape(window_size)
        advantages = advantages.reshape(window_size)
        flat_values = values.reshape(window_size)

        return_percentiles = torch.quantile(value_targets, return_percentile_levels)
        return_percentile_scale = (
            return_percentiles[1] - return_percentiles[0]
        ).clamp_min(args.return_percentile_floor)
        advantages = advantages / return_percentile_scale

        # E-step over the window with the importance-weighted proposal.
        topk_threshold = torch.sort(advantages).values[-args.topk_size]
        selected = advantages >= topk_threshold
        selected_count = selected.sum().to(advantages.dtype)
        # Truncated importance sampling, the same rho clip V-trace applies to
        # the value targets: the policy moves every update, so unclipped
        # ratios on the oldest slots would concentrate the window's proposal
        # on a handful of samples.
        log_proposal = torch.where(
            selected, log_rho.clamp_max(np.log(args.vtrace_rho_clip)), -torch.inf
        )
        log_proposal = log_proposal - torch.logsumexp(log_proposal, dim=0)
        safe_log_proposal = torch.where(selected, log_proposal, 0.0)
        proposal_ess = (-torch.logsumexp(2.0 * log_proposal, dim=0)).exp()

        maximum_advantage = advantages.max()
        centered_advantages = advantages - maximum_advantage
        selected_span = maximum_advantage - topk_threshold
        log_eta_low = torch.full_like(topk_threshold, np.log(ETA_FLOOR))
        log_eta_high = (
            selected_span.div(args.epsilon_eta).clamp_min(ETA_FLOOR).log()
        )
        for _ in range(32):
            log_eta_mid = 0.5 * (log_eta_low + log_eta_high)
            eta_mid = log_eta_mid.exp()
            mid_logits = torch.where(
                selected,
                log_proposal + centered_advantages / eta_mid,
                -torch.inf,
            )
            mid_log_weights = mid_logits - torch.logsumexp(mid_logits, dim=0)
            mid_weights = mid_log_weights.exp()
            safe_mid_log_weights = torch.where(selected, mid_log_weights, 0.0)
            mid_kl = (
                mid_weights * (safe_mid_log_weights - safe_log_proposal)
            ).sum()
            log_eta_low = torch.where(mid_kl > args.epsilon_eta, log_eta_mid, log_eta_low)
            log_eta_high = torch.where(mid_kl > args.epsilon_eta, log_eta_high, log_eta_mid)
        eta = log_eta_high.exp()
        policy_logits = torch.where(
            selected,
            log_proposal + centered_advantages / eta,
            -torch.inf,
        )
        log_weights = policy_logits - torch.logsumexp(policy_logits, dim=0)
        weights = log_weights.exp()
        safe_log_weights = torch.where(selected, log_weights, 0.0)
        temperature_kl = (weights * (safe_log_weights - safe_log_proposal)).sum()
        temperature_loss = maximum_advantage + eta * (
            args.epsilon_eta + torch.logsumexp(policy_logits, dim=0)
        )
        effective_sample_size = weights.square().sum().reciprocal()

        value_target_probs = hl_support.project_moment_matched(value_targets)

        # Behaviour-perturbation KL per environment on the newest rollout.
        explore_kl_per_sample = beta_kl(
            target_alpha, target_beta, flat_behaviour_alpha, flat_behaviour_beta
        ).view(window_batches, num_steps, num_envs)
        explore_kl_newest = (
            explore_kl_per_sample * newest_slot_mask[:, None, None]
        ).sum(dim=0)

        value_error = flat_values - value_targets
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
        rho_clip_fraction = (rho > args.vtrace_rho_clip).float().mean()
        window_rho_ess_fraction = (
            log_rho.logsumexp(0).mul(2.0) - (2.0 * log_rho).logsumexp(0)
        ).exp() / window_size

        metrics = torch.stack(
            (
                temperature_loss,
                eta,
                temperature_kl,
                effective_sample_size / selected_count,
                effective_sample_size,
                weights.max(),
                proposal_ess / selected_count,
                topk_threshold,
                advantages.mean(),
                advantages.std(),
                return_percentile_scale,
                value_rmse,
                explained_variance,
                target_outside_support,
                target_edge_mass,
                log_rho.mean(),
                log_rho.std(),
                rho_clip_fraction,
                window_rho_ess_fraction,
                explore_kl_newest.mean(),
            )
        )
        window_view = lambda tensor: tensor.view(window_batches, num_steps, num_envs, *tensor.shape[1:])
        return (
            grouped(observations),
            grouped(actions),
            grouped(window_view(weights)),
            grouped(window_view(value_target_probs)),
            flat_observations,
            metrics,
            explore_kl_newest,
        )

    def group_loss(
        flat,
        observations,
        actions,
        weights,
        value_target_probs,
    ):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            new_alpha, new_beta, value_logits = functional_call(
                agent, unflatten_all(flat), (observations,)
            )
        new_alpha = new_alpha.float()
        new_beta = new_beta.float()
        value_log_probs = F.log_softmax(value_logits.float(), dim=-1)
        # Weights are normalized over the whole window; the group mean of
        # num_groups * group sums recovers the window sum. The target policy is
        # the current policy at every step start, so the KL constraint of the
        # M-step has zero gradient there; the prior precision chosen by
        # cross-validation plays its role instead.
        log_prob = beta_log_prob(new_alpha, new_beta, actions)
        policy_loss = -(weights * log_prob).sum() * num_groups
        value_loss = -(value_target_probs * value_log_probs).sum(dim=-1).mean()
        total_loss = policy_loss + value_loss
        return total_loss, torch.stack((policy_loss / num_groups, value_loss))

    group_grad_model = grad(group_loss, has_aux=True)

    # Block weights: the budget metric divides each block by its per-update
    # KL cap so a unit of metric quadratic is a unit of budget; the loss
    # curvature is the plain Gauss-Newton Hessian of the M-step objective
    # (weighted Beta likelihood plus categorical cross-entropy), which has no
    # concentration term.
    budget_weights = (
        1.0 / args.step_policy_kl,
        1.0 / args.step_concentration_kl,
        1.0 / args.step_value_kl,
    )
    hessian_weights = (1.0, 0.0, 1.0)

    def fisher_vector_product(flat, vector, observations, damping, weights, sample_weights):
        """Weighted Gauss-Newton products of the Beta policy and categorical critic.

        F v = J^T F_out J v with J the network Jacobian on the Fisher
        subsample and F_out the closed-form output-distribution Fisher
        (trigamma for Beta, diag(p) - p p^T for the categorical), in three
        blocks weighted by `weights`: the Beta policy Fisher, the Beta
        concentration Fisher (the decoupled V-MPO constraint that keeps the
        policy from sharpening before it has moved; the Beta Fisher along the
        fixed-mean curve alpha = m c, beta = (1 - m) c), and the categorical
        critic Fisher.
        """

        def outputs(theta):
            return functional_call(agent, unflatten_all(theta), (observations,))

        (alpha, beta, logits), (d_alpha, d_beta, d_logits) = jvp(
            outputs, (flat,), (vector,)
        )
        trigamma_alpha = torch.special.polygamma(1, alpha)
        trigamma_beta = torch.special.polygamma(1, beta)
        trigamma_sum = torch.special.polygamma(1, alpha + beta)
        u_alpha = (trigamma_alpha - trigamma_sum) * d_alpha - trigamma_sum * d_beta
        u_beta = -trigamma_sum * d_alpha + (trigamma_beta - trigamma_sum) * d_beta
        mean = alpha / (alpha + beta)
        concentration_fisher = (
            mean.square() * trigamma_alpha
            + (1.0 - mean).square() * trigamma_beta
            - trigamma_sum
        )
        u_concentration = concentration_fisher * (d_alpha + d_beta)
        probs = torch.softmax(logits, dim=-1)
        u_logits = probs * d_logits - probs * (probs * d_logits).sum(-1, keepdim=True)
        policy_weight, concentration_weight, value_weight = weights
        # Per-sample policy weights: uniform for the metric (a KL over states),
        # the E-step weights for the loss curvature (the M-step is a weighted
        # likelihood, so its Hessian is the weight-averaged Fisher).
        policy_scale = (policy_weight * sample_weights)[:, None]
        _, pullback = vjp(outputs, flat)
        (fisher_vector,) = pullback(
            (
                u_alpha * policy_scale + u_concentration * (concentration_weight / fisher_samples),
                u_beta * policy_scale + u_concentration * (concentration_weight / fisher_samples),
                u_logits * (value_weight / fisher_samples),
            )
        )
        return fisher_vector + damping * vector

    def lanczos_model(flat, rhs, observations, sample_weights):
        """Krylov basis of the undamped budget metric from rhs, reorthogonalized.

        Returns the orthonormal basis Q (k, P), the tridiagonal T = Q F Q^T as
        (diagonal, off-diagonal), the next Lanczos vector and its coefficient
        (so F Q^T y is exact for any y), |rhs|, and the true Gauss-Newton
        Hessian products H Q^T (k, P) so the loss curvature of any direction
        in the subspace is exact. Every damped solve (F + lambda I)^-1 rhs
        then costs a k x k system, which lets the damping be chosen by
        cross-validation without further Fisher products.
        """
        uniform_weights = torch.full_like(sample_weights, 1.0 / fisher_samples)
        rhs_norm = rhs.norm()
        basis = [rhs / rhs_norm.clamp_min(KL_EPS)]
        diagonal = []
        off_diagonal = []
        previous = torch.zeros_like(rhs)
        beta = torch.zeros_like(rhs_norm)
        for index in range(args.krylov_dim):
            current = basis[-1]
            candidate = fisher_vector_product(
                flat, current, observations, 0.0, budget_weights, uniform_weights
            )
            alpha = candidate.dot(current)
            candidate = candidate - alpha * current - beta * previous
            for vector in basis:
                candidate = candidate - candidate.dot(vector) * vector
            beta = candidate.norm()
            diagonal.append(alpha)
            previous = current
            next_vector = candidate / beta.clamp_min(KL_EPS)
            if index < args.krylov_dim - 1:
                off_diagonal.append(beta)
                basis.append(next_vector)
        # Loss curvature: the M-step's weights (a group's share of the window
        # sum, rescaled to a convex average over the subsample).
        loss_weights = sample_weights * num_groups
        products = [
            fisher_vector_product(flat, vector, observations, 0.0, hessian_weights, loss_weights)
            for vector in basis
        ]
        return (
            torch.stack(basis),
            torch.stack(diagonal),
            torch.stack(off_diagonal),
            next_vector,
            beta,
            rhs_norm,
            torch.stack(products),
        )

    def select_damping(krylov_a, krylov_b, grad_a, grad_b):
        """Cross-validated prior precision for the damped natural gradient.

        For each damping on a log grid relative to the top Ritz value, the
        half-window solutions x_h = (F + lambda I)^-1 g_h are formed in their
        Krylov coordinates, F being the budget metric. g_b^T x_a and g_a^T x_b
        are unbiased for the signal mu^T x because the half-window noises are
        independent, while the combined step's budget quadratic x^T F x is
        exact from the Lanczos relation. The damping maximizing signal per
        unit root-budget is selected; the signal / total ratio at that damping
        is the step's Wiener gain. The step length along the chosen direction
        is the Newton length under the true loss curvature, not the metric.
        """
        basis_a, diag_a, off_a, last_a, beta_a, norm_a, products_a = krylov_a
        basis_b, diag_b, off_b, last_b, beta_b, norm_b, products_b = krylov_b
        hessian_a = basis_a @ products_a.T
        hessian_b = basis_b @ products_b.T
        tri_a = torch.diag(diag_a) + torch.diag(off_a, 1) + torch.diag(off_a, -1)
        tri_b = torch.diag(diag_b) + torch.diag(off_b, 1) + torch.diag(off_b, -1)
        ritz_max = torch.maximum(
            torch.linalg.eigvalsh(tri_a)[-1], torch.linalg.eigvalsh(tri_b)[-1]
        ).clamp_min(KL_EPS)
        grid = ritz_max * damping_grid_exponents
        identity = torch.eye(args.krylov_dim, device=device)
        shifted_a = tri_a[None] + grid[:, None, None] * identity[None]
        shifted_b = tri_b[None] + grid[:, None, None] * identity[None]
        unit = torch.zeros(args.krylov_dim, device=device)
        unit[0] = 1.0
        coords_a = torch.linalg.solve(shifted_a, (norm_a * unit).expand(grid.shape[0], -1))
        coords_b = torch.linalg.solve(shifted_b, (norm_b * unit).expand(grid.shape[0], -1))
        cross_a = coords_a @ (basis_a @ grad_b)
        cross_b = coords_b @ (basis_b @ grad_a)
        self_a = coords_a[:, 0] * norm_a
        self_b = coords_b[:, 0] * norm_b
        quad_a = ((coords_a @ tri_a) * coords_a).sum(dim=1)
        quad_b = ((coords_b @ tri_b) * coords_b).sum(dim=1)
        # x_a^T F x_b with F Q_b^T y = Q_b^T T_b y + beta_b y[-1] q_b,last.
        overlap = basis_a @ basis_b.T
        last_overlap = basis_a @ last_b
        mixed = ((coords_a @ overlap) * (coords_b @ tri_b)).sum(dim=1) + beta_b * coords_b[
            :, -1
        ] * (coords_a @ last_overlap)
        combined_quadratic = (0.25 * (quad_a + quad_b + 2.0 * mixed)).clamp_min(0.0)
        # Loss curvature of the same directions under the true Gauss-Newton
        # Hessian, exact from the stored products H Q^T: x^T H x for the
        # combined step 0.5 (x_a + x_b).
        curvature_a = ((coords_a @ hessian_a) * coords_a).sum(dim=1)
        curvature_b = ((coords_b @ hessian_b) * coords_b).sum(dim=1)
        curvature_ab = ((coords_a @ (basis_a @ products_b.T)) * coords_b).sum(dim=1)
        combined_curvature = (
            0.25 * (curvature_a + curvature_b + 2.0 * curvature_ab)
        ).clamp_min(0.0)
        signal = 0.5 * (cross_a + cross_b)
        total = 0.25 * (self_a + self_b + cross_a + cross_b)
        criterion = signal / combined_quadratic.sqrt().clamp_min(KL_EPS)
        best = torch.argmax(criterion)
        signal_fraction = (signal[best] / total[best].clamp_min(KL_EPS)).clamp(0.0, 1.0)
        quadratic_kl = 0.5 * combined_quadratic[best]
        # Quadratic model of the loss along x: -s * signal + 0.5 s^2 x^T H x.
        # Its minimizer is the Newton step shrunk by the cross-fitted Wiener
        # gain; its KL in budget units is newton_scale^2 * quadratic_kl.
        newton_scale = (signal[best] / combined_curvature[best].clamp_min(KL_EPS)).clamp_min(
            0.0
        )
        model_kl = newton_scale.square() * quadratic_kl
        metrics = torch.stack(
            (
                grid[best],
                damping_grid_exponents[best],
                ritz_max,
                criterion[best],
                signal[best],
                total[best],
                criterion[-1] / criterion[best].clamp_min(KL_EPS),
                criterion[0] / criterion[best].clamp_min(KL_EPS),
                signal_fraction,
                model_kl,
                newton_scale,
                combined_curvature[best],
            )
        )
        return coords_a[best], coords_b[best], quadratic_kl, newton_scale, metrics

    def step_model(
        flat,
        group_grads,
        basis_a,
        basis_b,
        y_a,
        y_b,
        quadratic_kl,
        newton_scale,
        learner_noise,
        noise_ema,
        power_ema,
        step_count,
        observations,
    ):
        """Cross-fitted natural-gradient step under exact KL budgets."""
        grad_mean = group_grads.mean(dim=0)
        grad_noise = group_grads.var(dim=0, unbiased=True) / num_groups
        new_noise_ema = args.noise_ema * noise_ema + (1.0 - args.noise_ema) * grad_noise
        new_power_ema = args.noise_ema * power_ema + (1.0 - args.noise_ema) * grad_mean.square()
        bias_correction = 1.0 - args.noise_ema ** (step_count + 1.0)
        noise_hat = new_noise_ema / bias_correction
        power_hat = new_power_ema / bias_correction
        coordinate_signal = (power_hat - noise_hat).clamp_min(0.0)
        coordinate_signal_fraction = coordinate_signal.sum() / power_hat.sum().clamp_min(
            POWER_EPS**2
        )
        gradient_noise_scale = (
            window_size * noise_hat.sum() / coordinate_signal.sum().clamp_min(POWER_EPS**2)
        )

        # Natural direction on the full window by linearity of the damped
        # solve; y_a, y_b are the Krylov coordinates chosen by select_damping.
        # The step is the Newton-Wiener step, pre-shrunk so its model KL is
        # within one unit of budget, then capped exactly per component (policy
        # KL, concentration KL, critic KL): the line search only scales down.
        natural_direction = 0.5 * (y_a @ basis_a + y_b @ basis_b)
        model_kl = newton_scale.square() * quadratic_kl
        model_scale = model_kl.clamp_min(KL_EPS).rsqrt().clamp_max(1.0)
        delta = -natural_direction * newton_scale * model_scale

        base_alpha, base_beta, base_logits = functional_call(
            agent, unflatten_all(flat), (observations,)
        )

        def policy_at(scale):
            new_alpha, new_beta, new_logits = functional_call(
                agent, unflatten_all(flat + scale * delta), (observations,)
            )
            return new_alpha, new_beta, new_logits

        def kl_of(new_alpha, new_beta, new_logits):
            policy_kl = beta_kl(base_alpha, base_beta, new_alpha, new_beta).mean()
            _, concentration_kl = decoupled_beta_kl(base_alpha, base_beta, new_alpha, new_beta)
            value_kl = categorical_kl(base_logits, new_logits).mean()
            return policy_kl, concentration_kl.mean(), value_kl

        def cap_scale(policy_kl, concentration_kl, value_kl):
            return torch.minimum(
                torch.minimum(
                    (args.step_policy_kl / policy_kl.clamp_min(KL_EPS)).sqrt(),
                    (args.step_concentration_kl / concentration_kl.clamp_min(KL_EPS)).sqrt(),
                ),
                (args.step_value_kl / value_kl.clamp_min(KL_EPS)).sqrt(),
            ).clamp_max(1.0)

        unit_scale = torch.ones_like(quadratic_kl)
        kl_unit = kl_of(*policy_at(unit_scale))
        first_scale = cap_scale(*kl_unit)
        scale = first_scale * cap_scale(*kl_of(*policy_at(first_scale)))
        # The Langevin term is added after the caps are set on the denoised
        # step alone; realized KLs are measured on the full update.
        new_flat = flat + scale * delta + learner_noise
        new_alpha, new_beta, new_logits = functional_call(
            agent, unflatten_all(new_flat), (observations,)
        )
        policy_kl_realized, concentration_kl, value_kl_realized = kl_of(
            new_alpha, new_beta, new_logits
        )
        policy_kl_unit, concentration_kl_unit, value_kl_unit = kl_unit

        # Decoupled split of the realized policy step: mean movement versus
        # concentration movement, both measured from the step start.
        mean_kl, _ = decoupled_beta_kl(base_alpha, base_beta, new_alpha, new_beta)
        mean_kl = mean_kl.mean()
        base_probs = F.softmax(base_logits, dim=-1)
        prediction_edge_mass = (base_probs[:, 0] + base_probs[:, -1]).mean()
        policy_concentration = (base_alpha + base_beta).mean()
        policy_variance = (
            base_alpha
            * base_beta
            / ((base_alpha + base_beta).square() * (base_alpha + base_beta + 1.0))
        ).mean()

        # Posterior-metric exploration scale: inverse root gradient power with
        # the mean power as prior precision, normalized to unit RMS.
        policy_power = power_hat[policy_coordinate_indices]
        explore_scale = torch.rsqrt(policy_power + policy_power.mean())
        explore_scale = explore_scale / explore_scale.square().mean().sqrt()

        metrics = torch.stack(
            (
                model_kl,
                gradient_noise_scale,
                coordinate_signal_fraction,
                grad_mean.norm(),
                noise_hat.sum().sqrt(),
                natural_direction.norm(),
                delta.norm() * scale,
                model_scale,
                newton_scale,
                policy_kl_unit,
                value_kl_unit,
                concentration_kl_unit,
                first_scale,
                policy_kl_realized,
                value_kl_realized,
                scale,
                mean_kl,
                concentration_kl,
                prediction_edge_mass,
                policy_concentration,
                policy_variance,
                quadratic_kl,
            )
        )
        return new_flat, new_noise_ema, new_power_ema, explore_scale, metrics

    if args.compile:
        rollout_model = torch.compile(
            rollout_model, mode=args.compile_mode, dynamic=False, fullgraph=True
        )
        prepare_model = torch.compile(
            prepare_model, mode=args.compile_mode, dynamic=False, fullgraph=True
        )
        group_grad_model = torch.compile(
            group_grad_model, mode=args.compile_mode, dynamic=False, fullgraph=True
        )
        lanczos_model = torch.compile(
            lanczos_model, mode=args.compile_mode, dynamic=False, fullgraph=True
        )
        step_model = torch.compile(
            step_model, mode=args.compile_mode, dynamic=False, fullgraph=True
        )
        print(f"compiled fullgraph training paths ({args.compile_mode})")

    # One RunningMeanStd row per environment, matching v13's 64 independent
    # NormalizeObservation wrappers rather than pooling trajectories.
    observation_means = np.zeros((num_envs,) + observation_shape, dtype=np.float64)
    observation_variances = np.ones_like(observation_means)
    observation_counts = np.full(num_envs, 1e-4, dtype=np.float64)
    discounted_returns = np.zeros(num_envs, dtype=np.float64)
    reward_return_means = np.zeros(num_envs, dtype=np.float64)
    reward_return_variances = np.ones(num_envs, dtype=np.float64)
    reward_return_counts = np.full(num_envs, 1e-4, dtype=np.float64)

    # Rolling window of raw rollouts; normalization is re-applied at update
    # time with the current statistics so every sample shares one scale.
    window_shape = (window_batches, num_steps, num_envs)
    window_raw_observations = torch.zeros(window_shape + observation_shape, device=device)
    window_raw_next_observations = torch.zeros_like(window_raw_observations)
    window_actions = torch.zeros(window_shape + action_shape, device=device)
    window_behaviour_alpha = torch.ones(window_shape + action_shape, device=device)
    window_behaviour_beta = torch.ones_like(window_behaviour_alpha)
    window_raw_rewards = torch.zeros(window_shape, device=device)
    window_terminations = torch.zeros(window_shape, device=device)
    window_boundaries = torch.zeros(window_shape, device=device, dtype=torch.bool)

    # Persistent copies of compiled outputs (cudagraph outputs are transient).
    group_observations = torch.zeros((num_groups, group_size, obs_dim), device=device)
    group_actions = torch.zeros((num_groups, group_size, action_dim), device=device)
    group_weights = torch.zeros((num_groups, group_size), device=device)
    group_value_target_probs = torch.zeros(
        (num_groups, group_size, args.num_value_bins), device=device
    )
    window_observations = torch.zeros((window_size, obs_dim), device=device)
    group_grads = torch.zeros((num_groups, num_params), device=device)
    group_aux = torch.zeros((num_groups, 2), device=device)
    half_grad_a = torch.zeros(num_params, device=device)
    half_grad_b = torch.zeros(num_params, device=device)
    # Time memory of the two half-window gradients, one EMA per horizon. The
    # halves stay disjoint in data, so the EMAs stay independent-noise
    # estimates and the cross-fit criterion selects the horizon exactly as it
    # selects the damping: longest memory whose signal has not gone stale.
    horizon_decays = torch.tensor(
        [1.0 - 1.0 / horizon for horizon in args.gradient_horizons], device=device
    )
    memory_a = torch.zeros((num_horizons, num_params), device=device)
    memory_b = torch.zeros((num_horizons, num_params), device=device)
    bases_a = torch.zeros((num_horizons, args.krylov_dim, num_params), device=device)
    bases_b = torch.zeros((num_horizons, args.krylov_dim, num_params), device=device)
    damping_grid_exponents = torch.logspace(
        args.damping_log_min, args.damping_log_max, args.damping_grid_size, device=device
    )
    noise_ema = torch.zeros(num_params, device=device)
    power_ema = torch.zeros(num_params, device=device)
    step_count = torch.zeros((), device=device)
    explore_scale = torch.ones(num_policy_params, device=device)
    perturbations = torch.zeros((num_envs, num_policy_params), device=device)
    perturbation_sigma = torch.zeros(num_envs, device=device)
    perturbation_step = torch.full((num_envs,), -1, device=device, dtype=torch.int64)
    explore_sigma = torch.tensor(args.explore_sigma_init, device=device)
    learner_noise = torch.zeros(num_params, device=device)
    explore_kl_newest = torch.zeros((num_steps, num_envs), device=device)
    explore_calibration = None
    num_clean_envs = int(round(num_envs * args.explore_clean_fraction))
    explore_env_mask = torch.arange(num_envs, device=device) >= num_clean_envs
    explore_env_mask_np = explore_env_mask.cpu().numpy()
    newest_slot_mask = torch.zeros(window_batches, device=device, dtype=torch.bool)
    exploration_armed = False

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
        np.arange(num_envs, dtype=np.int64) * args.initial_phase_warmup_steps // num_envs
    )
    phase_rng = np.random.default_rng(args.seed)
    phase_rng.shuffle(phase_offsets)
    writer.add_text(
        "initial_phase_offsets",
        ",".join(str(offset) for offset in phase_offsets),
    )
    scheduled_resets = args.initial_phase_warmup_steps - phase_offsets

    def act(observation_np):
        observation = torch.as_tensor(observation_np, device=device, dtype=torch.float32)
        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            alpha, beta = rollout_model(target_policy_flat, perturbations, observation)
            native_action = Beta(alpha, beta).sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = target_agent.action_low + (
                target_agent.action_high - target_agent.action_low
            ) * native_action
        return alpha, beta, native_action, action.cpu().numpy()

    reset_observations = np.zeros((num_envs,) + observation_shape, dtype=np.float64)
    for warmup_step in range(1, args.initial_phase_warmup_steps + 1):
        _, _, _, warmup_action = act(next_observation_np)
        (
            raw_next_observations,
            warmup_reward,
            warmup_terminations,
            warmup_truncations,
            warmup_infos,
        ) = envs.step(warmup_action)
        update_reward_moments(
            warmup_reward,
            warmup_terminations,
            discounted_returns,
            reward_return_means,
            reward_return_variances,
            reward_return_counts,
            gamma,
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
            reset_observations[env_index] = reset_observation
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
    raw_observation_np = np.array(raw_next_observations, copy=True)
    for env_index in np.flatnonzero(scheduled_resets == args.initial_phase_warmup_steps):
        raw_observation_np[env_index] = reset_observations[env_index]

    for iteration in range(1, args.num_iterations + 1):
        slot = (iteration - 1) % window_batches
        progress = global_step / args.total_timesteps
        anneal = float(
            np.clip((1.0 - progress) / max(args.explore_anneal_fraction, 1e-12), 0.0, 1.0)
        )
        explore_kl_target = args.explore_kl * anneal
        learner_kl_target = args.learner_explore_kl * anneal
        for step in range(num_steps):
            window_raw_observations[slot, step].copy_(
                torch.as_tensor(raw_observation_np, device=device, dtype=torch.float32)
            )
            alpha, beta, native_action, action = act(next_observation_np)
            window_actions[slot, step].copy_(native_action)
            window_behaviour_alpha[slot, step].copy_(alpha)
            window_behaviour_beta[slot, step].copy_(beta)

            (
                raw_next_observations,
                raw_reward,
                terminations,
                truncations,
                infos,
            ) = envs.step(action)
            update_reward_moments(
                raw_reward,
                terminations,
                discounted_returns,
                reward_return_means,
                reward_return_variances,
                reward_return_counts,
                gamma,
            )
            boundary = np.logical_or(terminations, truncations)
            next_observation_np, raw_transition_observations = normalize_vector_step(
                raw_next_observations,
                terminations,
                truncations,
                infos,
                observation_means,
                observation_variances,
                observation_counts,
            )
            raw_observation_np = np.array(raw_next_observations, copy=True)

            window_raw_rewards[slot, step].copy_(
                torch.as_tensor(raw_reward, device=device, dtype=torch.float32)
            )
            window_terminations[slot, step].copy_(
                torch.as_tensor(terminations, device=device, dtype=torch.float32)
            )
            window_boundaries[slot, step].copy_(
                torch.as_tensor(boundary, device=device, dtype=torch.bool)
            )
            window_raw_next_observations[slot, step].copy_(
                torch.as_tensor(
                    raw_transition_observations, device=device, dtype=torch.float32
                )
            )
            global_step += num_envs

            # Fresh per-episode perturbations for exploring environments.
            resample = np.logical_and(boundary, explore_env_mask_np)
            if exploration_armed and resample.any():
                with torch.no_grad():
                    resample_mask = torch.as_tensor(resample, device=device)
                    fresh = explore_sigma * explore_scale * torch.randn_like(perturbations)
                    perturbations.copy_(
                        torch.where(resample_mask[:, None], fresh, perturbations)
                    )
                    perturbation_sigma.copy_(
                        torch.where(resample_mask, explore_sigma, perturbation_sigma)
                    )
                    perturbation_step.copy_(
                        torch.where(
                            resample_mask,
                            torch.full_like(perturbation_step, iteration * num_steps + step),
                            perturbation_step,
                        )
                    )

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
                        group_tag = (
                            "charts/episodic_return_explore"
                            if explore_env_mask_np[env_index]
                            else "charts/episodic_return_clean"
                        )
                        writer.add_scalar(group_tag, episode_return, global_step)

        if iteration < window_batches:
            continue

        newest_slot_mask.zero_()
        newest_slot_mask[slot] = True
        observation_mean = torch.as_tensor(
            observation_means, device=device, dtype=torch.float32
        )
        observation_variance = torch.as_tensor(
            observation_variances, device=device, dtype=torch.float32
        )
        reward_scale = torch.as_tensor(
            1.0 / np.sqrt(reward_return_variances + 1e-8),
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            prepared = prepare_model(
                window_raw_observations,
                window_raw_next_observations,
                observation_mean,
                observation_variance,
                window_raw_rewards,
                reward_scale,
                window_actions,
                window_behaviour_alpha,
                window_behaviour_beta,
                window_terminations,
                window_boundaries,
                newest_slot_mask,
            )
            group_observations.copy_(prepared[0])
            group_actions.copy_(prepared[1])
            group_weights.copy_(prepared[2])
            group_value_target_probs.copy_(prepared[3])
            window_observations.copy_(prepared[4])
            prepare_metrics = prepared[5].clone()
            explore_kl_newest.copy_(prepared[6])
            del prepared

        flat_params = parameters_to_vector(agent.parameters()).detach().clone()
        for group_index in range(num_groups):
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            group_grad, aux = group_grad_model(
                flat_params,
                group_observations[group_index],
                group_actions[group_index],
                group_weights[group_index],
                group_value_target_probs[group_index],
            )
            group_grads[group_index].copy_(group_grad)
            group_aux[group_index].copy_(aux)
            del group_grad, aux

        with torch.no_grad():
            half_grad_a.copy_(group_grads[: num_groups // 2].mean(dim=0))
            half_grad_b.copy_(group_grads[num_groups // 2 :].mean(dim=0))
            memory_a.mul_(horizon_decays[:, None]).addcmul_(
                (1.0 - horizon_decays)[:, None], half_grad_a[None, :]
            )
            memory_b.mul_(horizon_decays[:, None]).addcmul_(
                (1.0 - horizon_decays)[:, None], half_grad_b[None, :]
            )
            memory_correction = 1.0 - horizon_decays ** (step_count + 1.0)
            candidates = []
            for horizon_index in range(num_horizons):
                krylov = []
                for memory, bases in ((memory_a, bases_a), (memory_b, bases_b)):
                    corrected = memory[horizon_index] / memory_correction[horizon_index]
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    outputs = lanczos_model(
                        flat_params, corrected, group_observations[0], group_weights[0]
                    )
                    bases[horizon_index].copy_(outputs[0])
                    krylov.append(
                        (bases[horizon_index],) + tuple(tensor.clone() for tensor in outputs[1:])
                    )
                    del outputs
                candidates.append(
                    select_damping(
                        krylov[0],
                        krylov[1],
                        memory_a[horizon_index] / memory_correction[horizon_index],
                        memory_b[horizon_index] / memory_correction[horizon_index],
                    )
                )
            horizon_criteria = torch.stack([candidate[4][3] for candidate in candidates])
            best_horizon = int(horizon_criteria.argmax())
            coords_a, coords_b, quadratic_kl, newton_scale, damping_metrics = candidates[
                best_horizon
            ]
            # Langevin term in the posterior metric: the same shaped noise the
            # behaviour perturbations use, so their measured KL per sigma^2
            # calibrates it, at its own annealed KL budget. This is the
            # deliberate exploration that replaces the estimation noise a
            # noisy optimizer would have injected and the E-step then selects.
            learner_noise.zero_()
            if learner_kl_target > 0.0 and explore_calibration is not None:
                learner_sigma = float(np.sqrt(learner_kl_target / explore_calibration))
                learner_noise[policy_coordinate_indices] = (
                    learner_sigma * explore_scale * torch.randn_like(explore_scale)
                )
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            new_flat, new_noise_ema, new_power_ema, new_explore_scale, step_metrics = step_model(
                flat_params,
                group_grads,
                bases_a[best_horizon],
                bases_b[best_horizon],
                coords_a,
                coords_b,
                quadratic_kl,
                newton_scale,
                learner_noise,
                noise_ema,
                power_ema,
                step_count,
                window_observations,
            )
            write_parameters(new_flat)
            noise_ema.copy_(new_noise_ema)
            power_ema.copy_(new_power_ema)
            explore_scale.copy_(new_explore_scale)
            step_metrics = step_metrics.clone()
            del new_flat, new_noise_ema, new_power_ema, new_explore_scale
            step_count += 1.0
            exploration_armed = True
            # Every step starts from its own target: the E-step is rebuilt on
            # the new policy next update, so the improvement direction is
            # renewed rather than exhausted against a stale target.
            target_agent.load_state_dict(agent.state_dict())
            target_policy_flat.copy_(target_policy_vector())


        # Exploration sigma calibration: per-state KL is quadratic in sigma, so
        # the newest rollout's exploring environments give KL / sigma^2 directly
        # and sigma is set to hit the (annealed) budget without a control loop.
        with torch.no_grad():
            # Only steps whose action was drawn under the current perturbation
            # count; clean environments measure the KL floor from observation
            # statistic drift, which is not exploration and is subtracted.
            rollout_steps = iteration * num_steps + torch.arange(num_steps, device=device)
            covered = rollout_steps[:, None] > perturbation_step[None, :]
            covered_count = covered.sum(dim=0)
            covered_kl = (explore_kl_newest * covered).sum(dim=0) / covered_count.clamp_min(1)
            if num_clean_envs > 0:
                kl_floor = explore_kl_newest[:, ~explore_env_mask].mean()
            else:
                kl_floor = torch.zeros((), device=device)
            calibrating = explore_env_mask & (perturbation_sigma > 0.0) & (covered_count > 0)
            if bool(calibrating.any()):
                coefficient = float(
                    (
                        (covered_kl[calibrating] - kl_floor).clamp_min(0.0)
                        / perturbation_sigma[calibrating].square()
                    ).mean()
                )
                if coefficient > 0.0:
                    explore_calibration = (
                        coefficient
                        if explore_calibration is None
                        else args.explore_calibration_ema * explore_calibration
                        + (1.0 - args.explore_calibration_ema) * coefficient
                    )
            if explore_kl_target <= 0.0:
                explore_sigma.zero_()
            elif explore_calibration is not None:
                explore_sigma.fill_(float(np.sqrt(explore_kl_target / explore_calibration)))
        explore_kl_measured = float(prepare_metrics[19])

        should_log = iteration % args.log_interval == 0 or iteration == window_batches
        if should_log:
            packed = torch.cat(
                (
                    prepare_metrics,
                    step_metrics,
                    group_aux.mean(dim=0),
                    explore_sigma.view(1),
                )
            ).cpu().tolist()
            (
                temperature_loss_value,
                eta_value,
                temperature_kl_value,
                ess_fraction_value,
                ess_value,
                max_weight_value,
                proposal_ess_fraction_value,
                top_advantage_min,
                advantage_mean,
                advantage_std,
                return_percentile_scale_value,
                value_rmse_value,
                explained_variance_value,
                target_outside_support_value,
                target_edge_mass_value,
                log_rho_mean_value,
                log_rho_std_value,
                rho_clip_fraction_value,
                window_rho_ess_fraction_value,
                explore_kl_measured_value,
                model_kl_value,
                gradient_noise_scale_value,
                coordinate_signal_fraction_value,
                grad_norm_value,
                noise_norm_value,
                direction_norm_value,
                update_norm_value,
                model_scale_value,
                newton_scale_value,
                policy_kl_unit_value,
                value_kl_unit_value,
                concentration_kl_unit_value,
                first_scale_value,
                policy_kl_realized_value,
                value_kl_realized_value,
                scale_value,
                mean_kl_value,
                concentration_kl_value,
                prediction_edge_mass_value,
                policy_concentration_value,
                policy_variance_value,
                quadratic_kl_value,
                policy_loss_value,
                value_loss_value,
                explore_sigma_value,
            ) = packed
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/step_scale", scale_value, global_step)
            writer.add_scalar("charts/update_norm", update_norm_value, global_step)
            writer.add_scalar("losses/policy_loss", policy_loss_value, global_step)
            writer.add_scalar("losses/value_loss", value_loss_value, global_step)
            writer.add_scalar("losses/temperature_loss", temperature_loss_value, global_step)
            writer.add_scalar("vmpo/eta", eta_value, global_step)
            writer.add_scalar("step/mean_kl", mean_kl_value, global_step)
            writer.add_scalar("step/concentration_kl", concentration_kl_value, global_step)
            writer.add_scalar("vmpo/weight_ess_fraction", ess_fraction_value, global_step)
            writer.add_scalar("vmpo/weight_ess", ess_value, global_step)
            writer.add_scalar("vmpo/proposal_ess_fraction", proposal_ess_fraction_value, global_step)
            writer.add_scalar("vmpo/e_step_kl", temperature_kl_value, global_step)
            writer.add_scalar("vmpo/eta_stationarity", args.epsilon_eta - temperature_kl_value, global_step)
            writer.add_scalar("vmpo/max_weight", max_weight_value, global_step)
            writer.add_scalar("vmpo/learner_updates", iteration - window_batches + 1, global_step)
            writer.add_scalar("vmpo/top_advantage_min", top_advantage_min, global_step)
            writer.add_scalar("window/log_rho_mean", log_rho_mean_value, global_step)
            writer.add_scalar("window/log_rho_std", log_rho_std_value, global_step)
            writer.add_scalar("window/rho_clip_fraction", rho_clip_fraction_value, global_step)
            writer.add_scalar("window/rho_ess_fraction", window_rho_ess_fraction_value, global_step)
            writer.add_scalar("denoise/signal_fraction", float(damping_metrics[8]), global_step)
            writer.add_scalar("denoise/gradient_noise_scale", gradient_noise_scale_value, global_step)
            writer.add_scalar(
                "denoise/coordinate_signal_fraction", coordinate_signal_fraction_value, global_step
            )
            writer.add_scalar("denoise/grad_norm", grad_norm_value, global_step)
            writer.add_scalar("denoise/noise_norm", noise_norm_value, global_step)
            writer.add_scalar("denoise/direction_norm", direction_norm_value, global_step)
            writer.add_scalar("step/model_kl", model_kl_value, global_step)
            writer.add_scalar("step/model_scale", model_scale_value, global_step)
            writer.add_scalar("step/newton_scale", newton_scale_value, global_step)
            writer.add_scalar("step/policy_kl_unit", policy_kl_unit_value, global_step)
            writer.add_scalar("step/value_kl_unit", value_kl_unit_value, global_step)
            writer.add_scalar("step/concentration_kl_unit", concentration_kl_unit_value, global_step)
            writer.add_scalar("step/first_scale", first_scale_value, global_step)
            writer.add_scalar("step/policy_kl_realized", policy_kl_realized_value, global_step)
            writer.add_scalar("step/value_kl_realized", value_kl_realized_value, global_step)
            writer.add_scalar("step/scale", scale_value, global_step)
            writer.add_scalar("step/quadratic_kl", quadratic_kl_value, global_step)
            writer.add_scalar("fisher/damping", float(damping_metrics[0]), global_step)
            writer.add_scalar("fisher/damping_ratio", float(damping_metrics[1]), global_step)
            writer.add_scalar("fisher/top_ritz_value", float(damping_metrics[2]), global_step)
            writer.add_scalar("fisher/criterion", float(damping_metrics[3]), global_step)
            writer.add_scalar("fisher/signal_quadratic", float(damping_metrics[4]), global_step)
            writer.add_scalar("fisher/total_quadratic", float(damping_metrics[5]), global_step)
            writer.add_scalar("fisher/sgd_criterion_ratio", float(damping_metrics[6]), global_step)
            writer.add_scalar("fisher/undamped_criterion_ratio", float(damping_metrics[7]), global_step)
            writer.add_scalar("fisher/loss_curvature", float(damping_metrics[11]), global_step)
            writer.add_scalar(
                "fisher/memory_horizon", args.gradient_horizons[best_horizon], global_step
            )
            for horizon, criterion_value in zip(
                args.gradient_horizons, horizon_criteria.tolist()
            ):
                writer.add_scalar(f"fisher/criterion_h{horizon}", criterion_value, global_step)
            writer.add_scalar("explore/sigma", explore_sigma_value, global_step)
            writer.add_scalar("explore/kl_measured", explore_kl_measured_value, global_step)
            writer.add_scalar("explore/kl_target", explore_kl_target, global_step)
            writer.add_scalar("explore/learner_kl_target", learner_kl_target, global_step)
            writer.add_scalar(
                "explore/kl_per_sigma_squared",
                0.0 if explore_calibration is None else explore_calibration,
                global_step,
            )
            writer.add_scalar("debug/advantage_mean", advantage_mean, global_step)
            writer.add_scalar("debug/advantage_std", advantage_std, global_step)
            writer.add_scalar(
                "debug/return_percentile_scale", return_percentile_scale_value, global_step
            )
            writer.add_scalar("debug/value_rmse", value_rmse_value, global_step)
            writer.add_scalar(
                "debug/value_explained_variance", explained_variance_value, global_step
            )
            writer.add_scalar(
                "critic/target_outside_support", target_outside_support_value, global_step
            )
            writer.add_scalar("critic/target_edge_mass", target_edge_mass_value, global_step)
            writer.add_scalar(
                "critic/prediction_edge_mass", prediction_edge_mass_value, global_step
            )
            writer.add_scalar(
                "debug/policy_concentration", policy_concentration_value, global_step
            )
            writer.add_scalar("debug/policy_native_variance", policy_variance_value, global_step)
            print(f"SPS: {sps}")

    envs.close()
    writer.close()

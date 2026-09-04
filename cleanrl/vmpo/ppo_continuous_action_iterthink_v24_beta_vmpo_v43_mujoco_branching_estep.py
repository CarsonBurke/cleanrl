# V-MPO v43 — native conditional E-step from counted MuJoCo clone cohorts.
#
# Each cohort clones one on-policy anchor state into K physical environments, so
# its K independent pi_old actions are genuine within-state return comparisons.
# Choosing the propagated member uniformly before any action or reward is seen
# makes the selected action marginal exactly pi_old; cloning its successor thus
# preserves the d_pi_old anchor-state distribution without extra simulator steps.

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
ETA_FLOOR = 1e-8
POLICY_KL_TOLERANCE = 1e-6
POLICY_PROJECTION_STEPS = 12


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
    clean_envs: int = 16
    branch_actions: int = 4
    num_steps: int = 39
    gamma: float = 0.99
    gae_lambda: float = 0.95

    epsilon_eta: float = 0.5
    eta_bisection_steps: int = 64
    epsilon_policy_kl: float = 0.01
    m_step_updates: int = 16
    m_step_minibatch_size: int = 128

    num_value_bins: int = 51
    value_support_limit: float = 20_000.0
    value_sigma_to_bin_ratio: float = 0.75

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    verify_cohort_states_every_step: bool = True
    cohort_state_tolerance: float = 1e-12
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    bf16: bool = True
    log_interval: int = 10

    search_cohorts: int = 0
    logical_envs: int = 0
    batch_size: int = 0
    logical_batch_size: int = 0
    num_iterations: int = 0
    initial_phase_warmup_steps: int = 0


def make_clean_env(env_id, idx, capture_video, run_name):
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


def make_search_env(env_id):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    return env


def normalize_observations(
    observations,
    observation_means,
    observation_variances,
    observation_counts,
    logical_indices=None,
):
    """Update independent logical-trajectory moments and normalize the samples."""
    observations = np.asarray(observations)
    if logical_indices is None:
        means = observation_means
        variances = observation_variances
        counts = observation_counts
    else:
        means = observation_means[logical_indices]
        variances = observation_variances[logical_indices]
        counts = observation_counts[logical_indices]

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

    if logical_indices is None:
        observation_means[...] = new_means
        observation_variances[...] = new_variances
        observation_counts[...] = total_counts
    else:
        observation_means[logical_indices] = new_means
        observation_variances[logical_indices] = new_variances
        observation_counts[logical_indices] = total_counts

    return np.clip(
        (observations - new_means) / np.sqrt(new_variances + 1e-8),
        -10.0,
        10.0,
    )


def normalize_observations_current(
    observations, observation_means, observation_variances, logical_indices
):
    means = observation_means[logical_indices]
    variances = observation_variances[logical_indices]
    return np.clip(
        (np.asarray(observations) - means) / np.sqrt(variances + 1e-8),
        -10.0,
        10.0,
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
    """Update v30-style discounted-return moments for logical trajectories."""
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
    return np.clip(raw_rewards / np.sqrt(new_variances + 1e-8), -10.0, 10.0)


def normalize_vector_step(
    raw_next_observations,
    terminations,
    truncations,
    infos,
    observation_means,
    observation_variances,
    observation_counts,
):
    """Normalize a clean same-step autoreset batch in wrapper order."""
    boundaries = np.logical_or(terminations, truncations)
    boundary_indices = np.flatnonzero(boundaries)
    raw_transition_observations = np.array(raw_next_observations, copy=True)
    if boundary_indices.size:
        final_observations = infos.get("final_observation")
        final_mask = infos.get("_final_observation")
        if final_observations is None:
            raise RuntimeError("completed clean transition missing final_observation")
        for env_index in boundary_indices:
            if final_mask is not None and not final_mask[env_index]:
                raise RuntimeError(
                    f"completed clean environment {env_index} has no final observation"
                )
            final_observation = final_observations[env_index]
            if final_observation is None:
                raise RuntimeError(
                    f"completed clean environment {env_index} has no final observation"
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


def time_limit_elapsed_steps(env):
    current = env
    while True:
        if hasattr(current, "_elapsed_steps"):
            elapsed = current._elapsed_steps
            if elapsed is None:
                raise RuntimeError("search TimeLimit has no initialized elapsed-step state")
            return int(elapsed)
        if not hasattr(current, "env"):
            break
        current = current.env
    raise RuntimeError("search environment must contain a TimeLimit wrapper")


def set_time_limit_elapsed_steps(env, elapsed_steps):
    current = env
    found = False
    while True:
        if hasattr(current, "_elapsed_steps"):
            current._elapsed_steps = int(elapsed_steps)
            found = True
        if not hasattr(current, "env"):
            break
        current = current.env
    if not found:
        raise RuntimeError("search environment must contain a TimeLimit wrapper")


def validate_mujoco_env(env):
    base = env.unwrapped
    missing = []
    if not hasattr(base, "data"):
        missing.append("data")
    else:
        if not hasattr(base.data, "qpos"):
            missing.append("data.qpos")
        if not hasattr(base.data, "qvel"):
            missing.append("data.qvel")
    if not callable(getattr(base, "set_state", None)):
        missing.append("set_state")
    if missing:
        raise RuntimeError(
            "search environment is incompatible with MuJoCo qpos/qvel branching; "
            f"missing {', '.join(missing)}"
        )
    current = env
    while True:
        if hasattr(current, "_elapsed_steps"):
            break
        if not hasattr(current, "env"):
            raise RuntimeError("search environment must contain a TimeLimit wrapper")
        current = current.env


def snapshot_mujoco_state(env):
    validate_mujoco_env(env)
    base = env.unwrapped
    return np.array(base.data.qpos, copy=True), np.array(base.data.qvel, copy=True)


def clone_cohort_from_member(cohort, member_index):
    qpos, qvel = snapshot_mujoco_state(cohort[member_index])
    elapsed_steps = time_limit_elapsed_steps(cohort[member_index])
    for env in cohort:
        env.unwrapped.set_state(qpos, qvel)
        set_time_limit_elapsed_steps(env, elapsed_steps)


def verify_cohort_state(cohort, tolerance):
    anchor_qpos, anchor_qvel = snapshot_mujoco_state(cohort[0])
    anchor_elapsed = time_limit_elapsed_steps(cohort[0])
    for member_index, env in enumerate(cohort[1:], start=1):
        qpos, qvel = snapshot_mujoco_state(env)
        elapsed = time_limit_elapsed_steps(env)
        if elapsed != anchor_elapsed or not np.allclose(
            qpos, anchor_qpos, rtol=tolerance, atol=tolerance
        ) or not np.allclose(qvel, anchor_qvel, rtol=tolerance, atol=tolerance):
            qpos_error = float(np.max(np.abs(qpos - anchor_qpos)))
            qvel_error = float(np.max(np.abs(qvel - anchor_qvel)))
            raise RuntimeError(
                "MuJoCo cohort diverged before branching: "
                f"member={member_index}, elapsed={elapsed}/{anchor_elapsed}, "
                f"qpos_error={qpos_error:.3e}, qvel_error={qvel_error:.3e}"
            )


def initialize_cohort(cohort, seeds, tolerance):
    anchor_observation = None
    for member_index, (env, seed) in enumerate(zip(cohort, seeds, strict=True)):
        observation, _ = env.reset(seed=seed)
        if member_index == 0:
            anchor_observation = np.array(observation, copy=True)
    clone_cohort_from_member(cohort, 0)
    verify_cohort_state(cohort, tolerance)
    if anchor_observation is None:
        raise RuntimeError("cannot initialize an empty search cohort")
    return anchor_observation


def reset_complete_cohort(cohort, anchor_member, tolerance):
    anchor_observation, _ = cohort[anchor_member].reset()
    clone_cohort_from_member(cohort, anchor_member)
    verify_cohort_state(cohort, tolerance)
    return np.array(anchor_observation, copy=True)


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


class BetaActor(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor

    def __init__(self, observation_space, action_space, args):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        self.trunk = ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
        self.policy_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        self.actor_alpha = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.register_buffer(
            "action_low", torch.tensor(action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(action_space.high, dtype=torch.float32)
        )

    def forward(self, observations):
        features = self.policy_mlp(self.trunk(observations))
        alpha = 1.0 + F.softplus(self.actor_alpha(features))
        beta = 1.0 + F.softplus(self.actor_beta(features))
        return alpha, beta


class ValueCritic(nn.Module):
    def __init__(self, observation_space, args):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        self.trunk = ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
        self.value_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        self.value_head = nn.Linear(256, args.num_value_bins, bias=False)
        with torch.no_grad():
            self.value_head.weight.zero_()

    def forward(self, observations):
        return self.value_head(self.value_mlp(self.trunk(observations)))


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
    """Exact full forward KL for factorized Beta policies."""
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


@torch.no_grad()
def solve_conditional_weights(scores, epsilon_eta, bisection_steps):
    """Geometrically solve mean_s KL(softmax(A_s/eta) || Uniform(K))."""
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("conditional scores must have shape [states, actions>=2]")
    if not torch.isfinite(scores).all():
        raise RuntimeError("non-finite branch score in conditional E-step")
    num_actions = scores.shape[1]
    log_num_actions = scores.new_tensor(float(np.log(num_actions)))
    uniform_weights = torch.full_like(scores, 1.0 / num_actions)
    centered_scores = scores - scores.max(dim=1, keepdim=True).values
    score_span = -centered_scores.min()
    if bool((score_span == 0.0).item()):
        return scores.new_tensor(ETA_FLOOR), uniform_weights, scores.new_zeros(())

    def weights_and_kl(eta):
        log_weights = F.log_softmax(centered_scores / eta, dim=1)
        weights = log_weights.exp()
        mean_kl = (weights * (log_weights + log_num_actions)).sum(dim=1).mean()
        return weights, mean_kl

    eta_low = scores.new_tensor(ETA_FLOOR)
    floor_weights, floor_kl = weights_and_kl(eta_low)
    if bool((floor_kl <= epsilon_eta).item()):
        return eta_low, floor_weights, floor_kl

    eta_high = torch.maximum(
        score_span / epsilon_eta, eta_low * scores.new_tensor(2.0)
    )
    _, high_kl = weights_and_kl(eta_high)
    for _ in range(32):
        if bool((high_kl <= epsilon_eta).item()):
            break
        eta_high *= 2.0
        _, high_kl = weights_and_kl(eta_high)
    if bool((high_kl > epsilon_eta).item()):
        raise RuntimeError("failed to bracket conditional E-step temperature")

    log_eta_low = eta_low.log()
    log_eta_high = eta_high.log()
    for _ in range(bisection_steps):
        log_eta_mid = 0.5 * (log_eta_low + log_eta_high)
        eta_mid = log_eta_mid.exp()
        _, mid_kl = weights_and_kl(eta_mid)
        if bool((mid_kl > epsilon_eta).item()):
            log_eta_low = log_eta_mid
        else:
            log_eta_high = log_eta_mid
    eta = log_eta_high.exp()
    weights, mean_kl = weights_and_kl(eta)
    return eta, weights, mean_kl


def clone_parameters(module):
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def load_parameters(module, state):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(state[name])


def load_interpolated_parameters(module, low_state, high_state, fraction):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(torch.lerp(low_state[name], high_state[name], fraction))


def validate_args(args):
    if args.num_steps != 39:
        raise ValueError("OpenAI Gym paper alignment requires a 39-step unroll")
    if args.num_envs <= 0 or not 0 < args.clean_envs < args.num_envs:
        raise ValueError("num_envs must exceed a positive clean_envs count")
    if args.branch_actions < 2:
        raise ValueError("branch_actions must be at least two")
    search_physical_envs = args.num_envs - args.clean_envs
    if search_physical_envs % args.branch_actions:
        raise ValueError(
            "num_envs-clean_envs must be exactly divisible by branch_actions"
        )
    inferred_cohorts = search_physical_envs // args.branch_actions
    if args.search_cohorts not in (0, inferred_cohorts):
        raise ValueError("search_cohorts is derived and cannot override the partition")
    args.search_cohorts = inferred_cohorts
    args.logical_envs = args.clean_envs + args.search_cohorts
    if args.clean_envs + args.search_cohorts * args.branch_actions != args.num_envs:
        raise RuntimeError("physical environment partition does not sum to num_envs")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
    if not 0.0 < args.epsilon_eta < np.log(args.branch_actions):
        raise ValueError("epsilon_eta must be in (0, log(branch_actions))")
    if args.eta_bisection_steps <= 0:
        raise ValueError("eta_bisection_steps must be positive")
    if args.epsilon_policy_kl <= 0.0:
        raise ValueError("epsilon_policy_kl must be positive")
    if args.m_step_updates <= 0 or args.m_step_minibatch_size <= 0:
        raise ValueError("M-step update and minibatch counts must be positive")
    search_state_count = args.num_steps * args.search_cohorts
    if args.m_step_minibatch_size > search_state_count:
        raise ValueError("m_step_minibatch_size cannot exceed search anchor states")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.hidden <= 0 or args.k_blocks <= 0 or args.n_experts <= 0:
        raise ValueError("IterThink architecture dimensions must be positive")
    if args.num_value_bins < 3 or args.num_value_bins % 2 == 0:
        raise ValueError("num_value_bins must be odd and at least three")
    if args.value_support_limit <= 0.0 or args.value_sigma_to_bin_ratio <= 0.0:
        raise ValueError("value support limit and sigma ratio must be positive")
    if args.cohort_state_tolerance < 0.0:
        raise ValueError("cohort_state_tolerance cannot be negative")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.bf16 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 support is required")

    env_spec = gym.spec(args.env_id)
    if env_spec.max_episode_steps is None:
        raise ValueError("phase staggering requires a finite episode horizon")
    args.initial_phase_warmup_steps = int(env_spec.max_episode_steps)
    args.batch_size = args.num_envs * args.num_steps
    args.logical_batch_size = args.logical_envs * args.num_steps
    warmup_transitions = args.num_envs * args.initial_phase_warmup_steps
    if warmup_transitions >= args.total_timesteps:
        raise ValueError("total_timesteps must exceed the one-horizon warmup")
    args.num_iterations = (
        args.total_timesteps - warmup_transitions
    ) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps leaves no complete policy cycle")
    return warmup_transitions


def main():
    args = tyro.cli(Args)
    warmup_transitions = validate_args(args)
    planned_physical_transitions = warmup_transitions + args.num_iterations * args.batch_size

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

    clean_vector = None
    search_envs = []
    try:
        clean_vector = gym.vector.SyncVectorEnv(
            [
                make_clean_env(args.env_id, index, args.capture_video, run_name)
                for index in range(args.clean_envs)
            ]
        )
        if not isinstance(clean_vector.single_action_space, gym.spaces.Box):
            raise TypeError("V-MPO continuous control requires a Box action space")
        if (
            clean_vector.single_observation_space.shape is None
            or clean_vector.single_action_space.shape is None
        ):
            raise ValueError("continuous observation and action shapes are required")
        if clean_vector.envs[0].spec is None or (
            clean_vector.envs[0].spec.max_episode_steps
            != args.initial_phase_warmup_steps
        ):
            raise RuntimeError("constructed clean environment horizon differs from gym spec")

        for _ in range(args.search_cohorts * args.branch_actions):
            env = make_search_env(args.env_id)
            search_envs.append(env)
        for env in search_envs:
            validate_mujoco_env(env)
            if env.observation_space != clean_vector.single_observation_space:
                raise RuntimeError("search and clean observation spaces differ")
            if env.action_space != clean_vector.single_action_space:
                raise RuntimeError("search and clean action spaces differ")
        cohorts = [
            search_envs[
                cohort_index * args.branch_actions : (cohort_index + 1)
                * args.branch_actions
            ]
            for cohort_index in range(args.search_cohorts)
        ]

        device = torch.device("cuda")
        autocast_dtype = torch.bfloat16
        observation_shape = tuple(clean_vector.single_observation_space.shape)
        action_shape = tuple(clean_vector.single_action_space.shape)

        symlog_limit = float(np.log1p(args.value_support_limit))
        hl_support = Dreamer3BucketHLGaussSupport(
            args.num_value_bins,
            -symlog_limit,
            symlog_limit,
            args.value_sigma_to_bin_ratio,
            device,
        )

        initial_actor = BetaActor(
            clean_vector.single_observation_space,
            clean_vector.single_action_space,
            args,
        ).to(device)
        deployed_actor = copy.deepcopy(initial_actor).requires_grad_(False)
        candidate_actor = copy.deepcopy(initial_actor)
        del initial_actor
        value_critic = ValueCritic(clean_vector.single_observation_space, args).to(device)
        actor_parameter_ids = {
            id(parameter)
            for module in (deployed_actor, candidate_actor)
            for parameter in module.parameters()
        }
        critic_parameter_ids = {id(parameter) for parameter in value_critic.parameters()}
        if actor_parameter_ids & critic_parameter_ids:
            raise RuntimeError("actor and critic parameters must be disjoint")
        if not all(
            tensor.is_cuda
            for module in (deployed_actor, candidate_actor, value_critic)
            for tensor in (*module.parameters(), *module.buffers())
        ):
            raise RuntimeError("all actor and critic tensors must reside on CUDA")

        value_optimizer = optim.Adam(
            value_critic.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )

        def actor_forward(actor, observations):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                alpha, beta = actor(observations)
            return alpha.float(), beta.float()

        def critic_logits_forward(observations):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                return value_critic(observations)

        if args.compile:
            deployed_policy_model = torch.compile(
                lambda observations: actor_forward(deployed_actor, observations),
                mode=args.compile_mode,
                dynamic=False,
                fullgraph=True,
            )
            candidate_policy_model = torch.compile(
                lambda observations: actor_forward(candidate_actor, observations),
                mode=args.compile_mode,
                dynamic=False,
                fullgraph=True,
            )
            critic_logits_model = torch.compile(
                critic_logits_forward,
                mode=args.compile_mode,
                dynamic=False,
                fullgraph=True,
            )
            print(f"compiled BF16 fullgraph model paths ({args.compile_mode})")
        else:
            deployed_policy_model = lambda observations: actor_forward(
                deployed_actor, observations
            )
            candidate_policy_model = lambda observations: actor_forward(
                candidate_actor, observations
            )
            critic_logits_model = critic_logits_forward

        def mark_compiled_step():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()

        def scalar_values(observations):
            mark_compiled_step()
            logits = critic_logits_model(observations)
            return hl_support.to_scalar(logits.float())

        def gae_from_values(reward_batch, values, next_values, terminations, boundaries):
            advantages = torch.empty_like(reward_batch)
            running_advantage = torch.zeros_like(next_values[-1])
            for reverse_step in reversed(range(reward_batch.shape[0])):
                delta = (
                    reward_batch[reverse_step]
                    + args.gamma
                    * next_values[reverse_step]
                    * (1.0 - terminations[reverse_step])
                    - values[reverse_step]
                )
                continuing = (
                    delta
                    + args.gamma * args.gae_lambda * running_advantage
                )
                running_advantage = torch.where(
                    boundaries[reverse_step], delta, continuing
                )
                advantages[reverse_step] = running_advantage
            return advantages, advantages + values

        def value_metrics(predictions, targets):
            error = predictions - targets
            rmse = error.square().mean().sqrt()
            explained_variance = 1.0 - error.var(unbiased=False) / (
                targets.var(unbiased=False) + 1e-8
            )
            return rmse, explained_variance

        def evaluate_candidate(
            policy_observations,
            search_observations,
            branch_native_actions,
            conditional_weights,
            old_policy_alpha,
            old_policy_beta,
            old_branch_log_prob,
        ):
            with torch.no_grad():
                mark_compiled_step()
                new_policy_alpha, new_policy_beta = candidate_policy_model(
                    policy_observations
                )
                new_policy_alpha = new_policy_alpha.clone()
                new_policy_beta = new_policy_beta.clone()
                full_kl_rows = beta_kl(
                    old_policy_alpha,
                    old_policy_beta,
                    new_policy_alpha,
                    new_policy_beta,
                )
                mark_compiled_step()
                new_search_alpha, new_search_beta = candidate_policy_model(
                    search_observations
                )
                new_branch_log_prob = beta_log_prob(
                    new_search_alpha[:, None, :],
                    new_search_beta[:, None, :],
                    branch_native_actions,
                )
                gain = (
                    conditional_weights
                    * (new_branch_log_prob - old_branch_log_prob)
                ).sum(dim=1).mean()
                concentration = new_policy_alpha + new_policy_beta
                native_variance = (
                    new_policy_alpha
                    * new_policy_beta
                    / (concentration.square() * (concentration + 1.0))
                )
                finite = torch.isfinite(
                    torch.cat(
                        (
                            new_policy_alpha.reshape(-1),
                            new_policy_beta.reshape(-1),
                            full_kl_rows.reshape(-1),
                            new_branch_log_prob.reshape(-1),
                            gain.reshape(-1),
                        )
                    )
                ).all()
                return (
                    full_kl_rows.mean(),
                    gain,
                    concentration.mean(),
                    native_variance.mean(),
                    bool(finite.item()),
                )

        observation_means = np.zeros(
            (args.logical_envs,) + observation_shape, dtype=np.float64
        )
        observation_variances = np.ones_like(observation_means)
        observation_counts = np.full(args.logical_envs, 1e-4, dtype=np.float64)
        discounted_returns = np.zeros(args.logical_envs, dtype=np.float64)
        reward_return_means = np.zeros(args.logical_envs, dtype=np.float64)
        reward_return_variances = np.ones(args.logical_envs, dtype=np.float64)
        reward_return_counts = np.full(args.logical_envs, 1e-4, dtype=np.float64)
        search_logical_indices = np.arange(
            args.clean_envs, args.logical_envs, dtype=np.int64
        )

        rollout_shape = (args.num_steps,)
        clean_shape = rollout_shape + (args.clean_envs,)
        search_shape = rollout_shape + (args.search_cohorts,)
        branch_shape = search_shape + (args.branch_actions,)
        clean_observations = torch.empty(
            clean_shape + observation_shape, device=device
        )
        clean_next_observations = torch.empty_like(clean_observations)
        clean_native_actions = torch.empty(
            clean_shape + action_shape, device=device
        )
        clean_rewards = torch.empty(clean_shape, device=device)
        clean_terminations = torch.empty(clean_shape, device=device)
        clean_boundaries = torch.empty(clean_shape, device=device, dtype=torch.bool)
        search_observations = torch.empty(
            search_shape + observation_shape, device=device
        )
        search_anchor_next_observations = torch.empty_like(search_observations)
        search_anchor_rewards = torch.empty(search_shape, device=device)
        search_anchor_terminations = torch.empty(search_shape, device=device)
        search_anchor_boundaries = torch.empty(
            search_shape, device=device, dtype=torch.bool
        )
        search_anchor_native_actions = torch.empty(
            search_shape + action_shape, device=device
        )
        search_propagation_indices = torch.empty(
            search_shape, device=device, dtype=torch.long
        )
        branch_native_actions_buffer = torch.empty(
            branch_shape + action_shape, device=device
        )
        branch_rewards_buffer = torch.empty(branch_shape, device=device)
        branch_next_observations = torch.empty(
            branch_shape + observation_shape, device=device
        )
        branch_terminations_buffer = torch.empty(branch_shape, device=device)
        branch_truncations_buffer = torch.empty(branch_shape, device=device)

        raw_clean_observations, _ = clean_vector.reset(seed=args.seed)
        clean_next_np = normalize_observations(
            raw_clean_observations,
            observation_means[: args.clean_envs],
            observation_variances[: args.clean_envs],
            observation_counts[: args.clean_envs],
        )
        raw_search_observations = np.empty(
            (args.search_cohorts,) + observation_shape, dtype=np.float64
        )
        cohort_reset_count = args.search_cohorts
        cohort_clone_count = args.search_cohorts
        for cohort_index, cohort in enumerate(cohorts):
            seed_base = args.seed + args.clean_envs + cohort_index * args.branch_actions
            raw_search_observations[cohort_index] = initialize_cohort(
                cohort,
                [seed_base + member for member in range(args.branch_actions)],
                args.cohort_state_tolerance,
            )
        search_next_np = normalize_observations(
            raw_search_observations,
            observation_means,
            observation_variances,
            observation_counts,
            search_logical_indices,
        )

        phase_offsets = (
            np.arange(args.logical_envs, dtype=np.int64)
            * args.initial_phase_warmup_steps
            // args.logical_envs
        )
        phase_rng = np.random.default_rng(args.seed)
        phase_rng.shuffle(phase_offsets)
        writer.add_text(
            "initial_phase_offsets",
            ",".join(str(offset) for offset in phase_offsets),
        )
        scheduled_resets = args.initial_phase_warmup_steps - phase_offsets
        propagation_rng = np.random.default_rng(args.seed + 1_000_003)

        global_step = 0
        logical_vector_steps = 0
        clean_physical_transitions = 0
        branch_physical_transitions = 0
        boundary_cohort_resets = 0
        start_time = time.time()
        suppress_next_clean_episode_log = np.zeros(args.clean_envs, dtype=bool)

        def account_physical_step():
            nonlocal global_step
            nonlocal logical_vector_steps
            nonlocal clean_physical_transitions
            nonlocal branch_physical_transitions
            logical_vector_steps += 1
            clean_physical_transitions += args.clean_envs
            branch_physical_transitions += args.search_cohorts * args.branch_actions
            global_step += args.num_envs
            if clean_physical_transitions + branch_physical_transitions != global_step:
                raise RuntimeError("physical transition accounting invariant failed")

        def force_phase_reset(logical_index):
            nonlocal cohort_reset_count
            nonlocal cohort_clone_count
            if logical_index < args.clean_envs:
                reset_observation, _ = clean_vector.envs[logical_index].reset()
                clean_next_np[logical_index] = normalize_observations(
                    np.asarray(reset_observation)[None, ...],
                    observation_means[: args.clean_envs],
                    observation_variances[: args.clean_envs],
                    observation_counts[: args.clean_envs],
                    slice(logical_index, logical_index + 1),
                )[0]
                return
            cohort_index = logical_index - args.clean_envs
            reset_observation = reset_complete_cohort(
                cohorts[cohort_index], 0, args.cohort_state_tolerance
            )
            cohort_reset_count += 1
            cohort_clone_count += 1
            search_next_np[cohort_index] = normalize_observations(
                reset_observation[None, ...],
                observation_means,
                observation_variances,
                observation_counts,
                slice(logical_index, logical_index + 1),
            )[0]

        def advance_physical_environments(store_step=None, log_clean_episodes=False):
            nonlocal clean_next_np
            nonlocal search_next_np
            nonlocal cohort_reset_count
            nonlocal cohort_clone_count
            nonlocal boundary_cohort_resets

            propagation_indices = propagation_rng.integers(
                args.branch_actions, size=args.search_cohorts
            )
            if args.verify_cohort_states_every_step:
                for cohort in cohorts:
                    verify_cohort_state(cohort, args.cohort_state_tolerance)

            repeated_search_observations = np.repeat(
                search_next_np[:, None, ...], args.branch_actions, axis=1
            ).reshape(
                (args.search_cohorts * args.branch_actions,) + observation_shape
            )
            physical_observations_np = np.concatenate(
                (clean_next_np, repeated_search_observations), axis=0
            )
            physical_observations = torch.as_tensor(
                physical_observations_np, device=device, dtype=torch.float32
            )
            with torch.no_grad():
                mark_compiled_step()
                alpha, beta = deployed_policy_model(physical_observations)
                physical_native_actions = Beta(alpha, beta).sample().clamp(
                    SAMPLE_EPS, 1.0 - SAMPLE_EPS
                )
                physical_actions = deployed_actor.action_low + (
                    deployed_actor.action_high - deployed_actor.action_low
                ) * physical_native_actions
            clean_native_action = physical_native_actions[: args.clean_envs]
            search_native_action = physical_native_actions[
                args.clean_envs :
            ].reshape(
                (args.search_cohorts, args.branch_actions) + action_shape
            )
            clean_action_np = physical_actions[: args.clean_envs].cpu().numpy()
            search_action_np = physical_actions[args.clean_envs :].cpu().numpy().reshape(
                (args.search_cohorts, args.branch_actions) + action_shape
            )

            if store_step is not None:
                clean_observations[store_step].copy_(
                    torch.as_tensor(clean_next_np, device=device, dtype=torch.float32)
                )
                clean_native_actions[store_step].copy_(clean_native_action)
                search_observations[store_step].copy_(
                    torch.as_tensor(search_next_np, device=device, dtype=torch.float32)
                )
                branch_native_actions_buffer[store_step].copy_(search_native_action)
                propagation_index_tensor = torch.as_tensor(
                    propagation_indices, device=device, dtype=torch.long
                )
                search_propagation_indices[store_step].copy_(
                    propagation_index_tensor
                )
                search_anchor_native_actions[store_step].copy_(
                    search_native_action[
                        torch.arange(args.search_cohorts, device=device),
                        propagation_index_tensor,
                    ]
                )

            (
                raw_clean_next,
                raw_clean_reward,
                clean_terminated,
                clean_truncated,
                clean_infos,
            ) = clean_vector.step(clean_action_np)
            raw_branch_next = np.empty(
                (args.search_cohorts, args.branch_actions) + observation_shape,
                dtype=np.float64,
            )
            raw_branch_rewards = np.empty(
                (args.search_cohorts, args.branch_actions), dtype=np.float64
            )
            branch_terminated = np.empty(
                (args.search_cohorts, args.branch_actions), dtype=bool
            )
            branch_truncated = np.empty_like(branch_terminated)
            for cohort_index, cohort in enumerate(cohorts):
                for member_index, env in enumerate(cohort):
                    (
                        branch_observation,
                        branch_reward,
                        terminated,
                        truncated,
                        _,
                    ) = env.step(search_action_np[cohort_index, member_index])
                    raw_branch_next[cohort_index, member_index] = branch_observation
                    raw_branch_rewards[cohort_index, member_index] = branch_reward
                    branch_terminated[cohort_index, member_index] = terminated
                    branch_truncated[cohort_index, member_index] = truncated
            account_physical_step()

            clean_reward = normalize_rewards(
                raw_clean_reward,
                clean_terminated,
                discounted_returns[: args.clean_envs],
                reward_return_means[: args.clean_envs],
                reward_return_variances[: args.clean_envs],
                reward_return_counts[: args.clean_envs],
                args.gamma,
            )
            clean_boundary = np.logical_or(clean_terminated, clean_truncated)
            clean_next_np, clean_transition_next = normalize_vector_step(
                raw_clean_next,
                clean_terminated,
                clean_truncated,
                clean_infos,
                observation_means[: args.clean_envs],
                observation_variances[: args.clean_envs],
                observation_counts[: args.clean_envs],
            )

            cohort_rows = np.arange(args.search_cohorts)
            selected_raw_next = raw_branch_next[cohort_rows, propagation_indices]
            selected_raw_rewards = raw_branch_rewards[cohort_rows, propagation_indices]
            selected_terminated = branch_terminated[cohort_rows, propagation_indices]
            selected_truncated = branch_truncated[cohort_rows, propagation_indices]
            selected_boundaries = np.logical_or(
                selected_terminated, selected_truncated
            )
            normalize_observations(
                selected_raw_next,
                observation_means,
                observation_variances,
                observation_counts,
                search_logical_indices,
            )
            normalized_branch_next = normalize_observations_current(
                raw_branch_next,
                observation_means,
                observation_variances,
                search_logical_indices[:, None],
            )
            normalize_rewards(
                selected_raw_rewards,
                selected_terminated,
                discounted_returns[args.clean_envs :],
                reward_return_means[args.clean_envs :],
                reward_return_variances[args.clean_envs :],
                reward_return_counts[args.clean_envs :],
                args.gamma,
            )
            normalized_branch_rewards = np.clip(
                raw_branch_rewards
                / np.sqrt(
                    reward_return_variances[args.clean_envs :, None] + 1e-8
                ),
                -10.0,
                10.0,
            )
            selected_transition_next = normalized_branch_next[
                cohort_rows, propagation_indices
            ]
            new_search_next = np.array(selected_transition_next, copy=True)
            for cohort_index, member_index in enumerate(propagation_indices):
                if selected_boundaries[cohort_index]:
                    reset_observation = reset_complete_cohort(
                        cohorts[cohort_index],
                        int(member_index),
                        args.cohort_state_tolerance,
                    )
                    cohort_reset_count += 1
                    boundary_cohort_resets += 1
                    cohort_clone_count += 1
                    new_search_next[cohort_index] = normalize_observations(
                        reset_observation[None, ...],
                        observation_means,
                        observation_variances,
                        observation_counts,
                        slice(
                            args.clean_envs + cohort_index,
                            args.clean_envs + cohort_index + 1,
                        ),
                    )[0]
                else:
                    clone_cohort_from_member(cohorts[cohort_index], int(member_index))
                    cohort_clone_count += 1
            search_next_np = new_search_next

            if store_step is not None:
                clean_rewards[store_step].copy_(
                    torch.as_tensor(clean_reward, device=device, dtype=torch.float32)
                )
                clean_terminations[store_step].copy_(
                    torch.as_tensor(clean_terminated, device=device, dtype=torch.float32)
                )
                clean_boundaries[store_step].copy_(
                    torch.as_tensor(clean_boundary, device=device, dtype=torch.bool)
                )
                clean_next_observations[store_step].copy_(
                    torch.as_tensor(
                        clean_transition_next, device=device, dtype=torch.float32
                    )
                )
                branch_rewards_buffer[store_step].copy_(
                    torch.as_tensor(
                        normalized_branch_rewards, device=device, dtype=torch.float32
                    )
                )
                branch_terminations_buffer[store_step].copy_(
                    torch.as_tensor(
                        branch_terminated, device=device, dtype=torch.float32
                    )
                )
                branch_truncations_buffer[store_step].copy_(
                    torch.as_tensor(
                        branch_truncated, device=device, dtype=torch.float32
                    )
                )
                branch_next_observations[store_step].copy_(
                    torch.as_tensor(
                        normalized_branch_next, device=device, dtype=torch.float32
                    )
                )
                search_anchor_rewards[store_step].copy_(
                    torch.as_tensor(
                        normalized_branch_rewards[cohort_rows, propagation_indices],
                        device=device,
                        dtype=torch.float32,
                    )
                )
                search_anchor_terminations[store_step].copy_(
                    torch.as_tensor(
                        selected_terminated, device=device, dtype=torch.float32
                    )
                )
                search_anchor_boundaries[store_step].copy_(
                    torch.as_tensor(
                        selected_boundaries, device=device, dtype=torch.bool
                    )
                )
                search_anchor_next_observations[store_step].copy_(
                    torch.as_tensor(
                        selected_transition_next, device=device, dtype=torch.float32
                    )
                )

            if log_clean_episodes and "final_info" in clean_infos:
                for env_index, info in enumerate(clean_infos["final_info"]):
                    if info and "episode" in info:
                        if suppress_next_clean_episode_log[env_index]:
                            suppress_next_clean_episode_log[env_index] = False
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

        for warmup_step in range(1, args.initial_phase_warmup_steps + 1):
            advance_physical_environments()
            for logical_index in np.flatnonzero(scheduled_resets == warmup_step):
                force_phase_reset(int(logical_index))
        if global_step != warmup_transitions:
            raise RuntimeError("warmup physical transition accounting is incorrect")
        suppress_next_clean_episode_log[:] = phase_offsets[: args.clean_envs] > 0

        policy_cycles = 0
        policy_promotions = 0
        last_value_loss = torch.zeros((), device=device)

        for iteration in range(1, args.num_iterations + 1):
            for step in range(args.num_steps):
                advance_physical_environments(
                    store_step=step, log_clean_episodes=True
                )

            with torch.no_grad():
                flat_clean_observations = clean_observations.reshape(
                    (-1,) + observation_shape
                )
                flat_clean_next_observations = clean_next_observations.reshape(
                    (-1,) + observation_shape
                )
                flat_search_observations = search_observations.reshape(
                    (-1,) + observation_shape
                )
                flat_branch_next_observations = branch_next_observations.reshape(
                    (-1,) + observation_shape
                )

                clean_values = scalar_values(flat_clean_observations).view(
                    args.num_steps, args.clean_envs
                )
                clean_next_values = scalar_values(
                    flat_clean_next_observations
                ).view(args.num_steps, args.clean_envs)
                _, clean_value_targets = gae_from_values(
                    clean_rewards,
                    clean_values,
                    clean_next_values,
                    clean_terminations,
                    clean_boundaries,
                )
                search_anchor_values = scalar_values(flat_search_observations).view(
                    args.num_steps, args.search_cohorts
                )
                branch_next_values = scalar_values(
                    flat_branch_next_observations
                ).view(
                    args.num_steps, args.search_cohorts, args.branch_actions
                )
                branch_targets = (
                    branch_rewards_buffer
                    + args.gamma
                    * (1.0 - branch_terminations_buffer)
                    * branch_next_values
                )
                search_value_targets = branch_targets.mean(dim=-1)
                stopped_branch_advantages = (
                    branch_targets - search_anchor_values.unsqueeze(-1)
                ).detach()
                clean_rmse, clean_explained_variance = value_metrics(
                    clean_values, clean_value_targets
                )
                search_rmse, search_explained_variance = value_metrics(
                    search_anchor_values, search_value_targets
                )

                combined_value_observations = torch.cat(
                    (flat_clean_observations, flat_search_observations), dim=0
                )
                combined_value_targets = torch.cat(
                    (
                        clean_value_targets.reshape(-1),
                        search_value_targets.reshape(-1),
                    ),
                    dim=0,
                ).detach()
                value_target_probs = hl_support.project_moment_matched(
                    combined_value_targets
                )

            mark_compiled_step()
            value_logits = critic_logits_model(combined_value_observations)
            value_log_probs = F.log_softmax(value_logits.float(), dim=-1)
            value_loss = -(value_target_probs * value_log_probs).sum(dim=-1).mean()
            value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            value_optimizer.step()
            last_value_loss = value_loss.detach()

            flat_scores = stopped_branch_advantages.reshape(
                -1, args.branch_actions
            )
            eta, conditional_weights, achieved_e_step_kl = (
                solve_conditional_weights(
                    flat_scores, args.epsilon_eta, args.eta_bisection_steps
                )
            )
            state_ess = conditional_weights.square().sum(dim=1).reciprocal()
            ess = state_ess.mean()
            ess_fraction = ess / args.branch_actions
            max_weight = conditional_weights.max()
            score_spread = (
                flat_scores.max(dim=1).values - flat_scores.min(dim=1).values
            ).mean()

            flat_branch_native_actions = branch_native_actions_buffer.reshape(
                (-1, args.branch_actions) + action_shape
            )
            policy_observations = torch.cat(
                (flat_clean_observations, flat_search_observations), dim=0
            )
            with torch.no_grad():
                mark_compiled_step()
                old_policy_alpha, old_policy_beta = deployed_policy_model(
                    policy_observations
                )
                old_policy_alpha = old_policy_alpha.clone()
                old_policy_beta = old_policy_beta.clone()
                old_search_alpha = old_policy_alpha[
                    flat_clean_observations.shape[0] :
                ]
                old_search_beta = old_policy_beta[
                    flat_clean_observations.shape[0] :
                ]
                old_branch_log_prob = beta_log_prob(
                    old_search_alpha[:, None, :],
                    old_search_beta[:, None, :],
                    flat_branch_native_actions,
                )

            candidate_actor.load_state_dict(deployed_actor.state_dict())
            deployed_parameter_state = clone_parameters(deployed_actor)
            candidate_optimizer = optim.Adam(
                candidate_actor.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=True,
            )
            search_state_count = flat_search_observations.shape[0]
            permutation = torch.randperm(search_state_count, device=device)
            permutation_cursor = 0
            for _ in range(args.m_step_updates):
                if permutation_cursor + args.m_step_minibatch_size > search_state_count:
                    permutation = torch.randperm(search_state_count, device=device)
                    permutation_cursor = 0
                minibatch_indices = permutation[
                    permutation_cursor : permutation_cursor
                    + args.m_step_minibatch_size
                ]
                permutation_cursor += args.m_step_minibatch_size
                mark_compiled_step()
                candidate_alpha, candidate_beta = candidate_policy_model(
                    flat_search_observations[minibatch_indices]
                )
                candidate_log_prob = beta_log_prob(
                    candidate_alpha[:, None, :],
                    candidate_beta[:, None, :],
                    flat_branch_native_actions[minibatch_indices],
                )
                policy_loss = -(
                    conditional_weights[minibatch_indices] * candidate_log_prob
                ).sum(dim=1).mean()
                candidate_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                candidate_optimizer.step()
            del candidate_optimizer

            unprojected_parameter_state = clone_parameters(candidate_actor)
            (
                full_kl,
                candidate_gain,
                policy_concentration,
                policy_native_variance,
                candidate_outputs_finite,
            ) = evaluate_candidate(
                policy_observations,
                flat_search_observations,
                flat_branch_native_actions,
                conditional_weights,
                old_policy_alpha,
                old_policy_beta,
                old_branch_log_prob,
            )
            projected_fraction = 1.0
            if candidate_outputs_finite and bool(
                (full_kl > args.epsilon_policy_kl).item()
            ):
                low_fraction = 0.0
                high_fraction = 1.0
                best_state = deployed_parameter_state
                for _ in range(POLICY_PROJECTION_STEPS):
                    middle_fraction = 0.5 * (low_fraction + high_fraction)
                    load_interpolated_parameters(
                        candidate_actor,
                        deployed_parameter_state,
                        unprojected_parameter_state,
                        middle_fraction,
                    )
                    (
                        middle_kl,
                        _,
                        _,
                        _,
                        middle_finite,
                    ) = evaluate_candidate(
                        policy_observations,
                        flat_search_observations,
                        flat_branch_native_actions,
                        conditional_weights,
                        old_policy_alpha,
                        old_policy_beta,
                        old_branch_log_prob,
                    )
                    if middle_finite and bool(
                        (middle_kl <= args.epsilon_policy_kl).item()
                    ):
                        low_fraction = middle_fraction
                        best_state = clone_parameters(candidate_actor)
                    else:
                        high_fraction = middle_fraction
                projected_fraction = low_fraction
                load_parameters(candidate_actor, best_state)
                (
                    full_kl,
                    candidate_gain,
                    policy_concentration,
                    policy_native_variance,
                    candidate_outputs_finite,
                ) = evaluate_candidate(
                    policy_observations,
                    flat_search_observations,
                    flat_branch_native_actions,
                    conditional_weights,
                    old_policy_alpha,
                    old_policy_beta,
                    old_branch_log_prob,
                )

            actor_metrics = torch.stack(
                (
                    full_kl,
                    candidate_gain,
                    policy_concentration,
                    policy_native_variance,
                    eta,
                    achieved_e_step_kl,
                    ess,
                    ess_fraction,
                    max_weight,
                    score_spread,
                )
            )
            promote = (
                candidate_outputs_finite
                and bool(torch.isfinite(actor_metrics).all().item())
                and bool((candidate_gain > 0.0).item())
                and bool(
                    (
                        full_kl
                        <= args.epsilon_policy_kl + POLICY_KL_TOLERANCE
                    ).item()
                )
            )
            if promote:
                deployed_actor.load_state_dict(candidate_actor.state_dict())
                policy_promotions += 1
            else:
                candidate_actor.load_state_dict(deployed_actor.state_dict())

            policy_cycles += 1
            promotion_rate = policy_promotions / policy_cycles
            sps = int(global_step / (time.time() - start_time))
            logical_transitions = logical_vector_steps * args.logical_envs
            writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar(
                "losses/value_loss", float(last_value_loss.item()), global_step
            )
            writer.add_scalar("vmpo/eta", float(eta.item()), global_step)
            writer.add_scalar(
                "vmpo/e_step_kl", float(achieved_e_step_kl.item()), global_step
            )
            writer.add_scalar("vmpo/weight_ess", float(ess.item()), global_step)
            writer.add_scalar(
                "vmpo/weight_ess_fraction", float(ess_fraction.item()), global_step
            )
            writer.add_scalar(
                "vmpo/max_weight", float(max_weight.item()), global_step
            )
            writer.add_scalar(
                "vmpo/score_spread", float(score_spread.item()), global_step
            )
            writer.add_scalar(
                "vmpo/candidate_gain", float(candidate_gain.item()), global_step
            )
            writer.add_scalar(
                "vmpo/projected_fraction", projected_fraction, global_step
            )
            writer.add_scalar(
                "vmpo/full_beta_kl", float(full_kl.item()), global_step
            )
            writer.add_scalar("vmpo/target_promoted", float(promote), global_step)
            writer.add_scalar("vmpo/promotion_rate", promotion_rate, global_step)
            writer.add_scalar("vmpo/policy_cycles", policy_cycles, global_step)
            writer.add_scalar(
                "debug/policy_concentration",
                float(policy_concentration.item()),
                global_step,
            )
            writer.add_scalar(
                "debug/policy_native_variance",
                float(policy_native_variance.item()),
                global_step,
            )
            writer.add_scalar(
                "debug/clean_value_rmse", float(clean_rmse.item()), global_step
            )
            writer.add_scalar(
                "debug/clean_value_explained_variance",
                float(clean_explained_variance.item()),
                global_step,
            )
            writer.add_scalar(
                "debug/search_value_rmse", float(search_rmse.item()), global_step
            )
            writer.add_scalar(
                "debug/search_value_explained_variance",
                float(search_explained_variance.item()),
                global_step,
            )
            writer.add_scalar(
                "accounting/physical_transitions", global_step, global_step
            )
            writer.add_scalar(
                "accounting/logical_transitions", logical_transitions, global_step
            )
            writer.add_scalar(
                "accounting/clean_physical_transitions",
                clean_physical_transitions,
                global_step,
            )
            writer.add_scalar(
                "accounting/branch_physical_transitions",
                branch_physical_transitions,
                global_step,
            )
            writer.add_scalar(
                "accounting/physical_per_logical_vector_step",
                args.num_envs,
                global_step,
            )
            writer.add_scalar(
                "debug/cohort_resets", cohort_reset_count, global_step
            )
            writer.add_scalar(
                "debug/cohort_boundary_resets", boundary_cohort_resets, global_step
            )
            writer.add_scalar(
                "debug/cohort_clones", cohort_clone_count, global_step
            )
            writer.add_scalar(
                "debug/cohort_member_clone_assignments",
                cohort_clone_count * (args.branch_actions - 1),
                global_step,
            )
            if iteration % args.log_interval == 0 or iteration == 1:
                print(f"SPS: {sps}")

        if global_step != planned_physical_transitions:
            raise RuntimeError("final physical transition accounting is incorrect")
    finally:
        if clean_vector is not None:
            clean_vector.close()
        for env in search_envs:
            env.close()
        writer.close()


if __name__ == "__main__":
    main()

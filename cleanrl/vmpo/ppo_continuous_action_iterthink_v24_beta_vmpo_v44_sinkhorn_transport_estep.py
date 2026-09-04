# V-MPO v44 — bistochastic transport E-step for native Beta policies.
#
# Joint V-MPO's unconstrained sample weights can spend mass on favorable
# states instead of isolating the action-correlated advantage tilt.  Over each
# frozen eight-rollout target window, this variant computes the exact
# maximum-entropy bistochastic transport projection identifiable from one
# executed action per state.  It preserves coarse target occupancy across both
# environment trajectories and rollout time, eliminating learned partition
# error without Q estimates, replay, simulator branches, counterfactual
# actions, or contrastive/signed updates.  A fresh candidate is fit by positive
# weighted likelihood and atomically promoted only after full-window analytic
# Beta-KL projection.

import copy
import math
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
ETA_FLOOR = 1e-6
ETA_KL_TOLERANCE = 1e-8
ETA_BRACKET_STEPS = 128
MARGINAL_TOLERANCE = 1e-6
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
    num_steps: int = 39
    gamma: float = 0.99
    gae_lambda: float = 0.95

    window_batches: int = 8
    # log(2) from v30's top-half selection plus 0.01 nat of extra tilt.
    epsilon_eta: float = 0.7031471805599453
    sinkhorn_iterations: int = 64
    eta_bisection_steps: int = 40
    epsilon_policy_kl: float = 0.01
    m_step_minibatches: int = 2048
    m_step_epochs: int = 4
    m_step_convergence_tolerance: float = 1e-6

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

    batch_size: int = 0
    window_size: int = 0
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
    """Update independent per-environment singleton moments."""
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
    """Normalize same-step autoreset observations in wrapper order."""
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


class BetaActor(nn.Module):
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
        self.register_buffer(
            "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        )

    def forward(self, observations):
        features = self.policy_mlp(self.trunk(observations))
        alpha = 1.0 + F.softplus(self.actor_alpha(features))
        beta = 1.0 + F.softplus(self.actor_beta(features))
        return alpha, beta


class ValueCritic(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.num_steps != 39:
        raise ValueError("OpenAI Gym paper alignment requires a 39-step unroll")
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if args.window_batches <= 0:
        raise ValueError("window_batches must be positive")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    args.batch_size = args.num_envs * args.num_steps
    args.window_size = args.window_batches * args.batch_size
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
        raise ValueError("total_timesteps must contain at least one complete rollout")

    if not math.isfinite(args.epsilon_eta) or args.epsilon_eta <= 0.0:
        raise ValueError("epsilon_eta must be finite and positive")
    if type(args.sinkhorn_iterations) is not int or args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be a positive integer")
    if type(args.eta_bisection_steps) is not int or args.eta_bisection_steps <= 0:
        raise ValueError("eta_bisection_steps must be a positive integer")
    if args.epsilon_policy_kl <= 0.0:
        raise ValueError("epsilon_policy_kl must be positive")
    if args.m_step_epochs <= 0 or args.m_step_epochs > 4:
        raise ValueError("m_step_epochs must be between one and four")
    if args.m_step_minibatches <= 0 or args.m_step_minibatches > args.window_size:
        raise ValueError("m_step_minibatches must be within the cycle window")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.m_step_convergence_tolerance < 0.0:
        raise ValueError("m_step_convergence_tolerance cannot be negative")
    if not 0.0 <= args.return_percentile_low < args.return_percentile_high <= 1.0:
        raise ValueError("return percentiles must satisfy 0 <= low < high <= 1")
    if args.return_percentile_floor <= 0.0:
        raise ValueError("return_percentile_floor must be positive")
    if args.num_value_bins < 3 or args.num_value_bins % 2 == 0:
        raise ValueError("num_value_bins must be odd and at least three")
    if args.value_support_limit <= 0.0 or args.value_sigma_to_bin_ratio <= 0.0:
        raise ValueError("value support limit and sigma ratio must be positive")
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
    autocast_dtype = torch.bfloat16
    return_percentile_levels = torch.tensor(
        [args.return_percentile_low, args.return_percentile_high], device=device
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
    if envs.single_observation_space.shape is None or envs.single_action_space.shape is None:
        raise ValueError("continuous-control observation and action shapes are required")

    observation_shape = tuple(envs.single_observation_space.shape)
    action_shape = tuple(envs.single_action_space.shape)

    initial_actor = BetaActor(envs, args).to(device)
    deployed_actor = copy.deepcopy(initial_actor).requires_grad_(False)
    candidate_actor = copy.deepcopy(initial_actor)
    del initial_actor
    value_critic = ValueCritic(envs, args).to(device)
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

    def gae_from_values(
        reward_batch,
        value_batch,
        next_value_batch,
        terminations,
        boundaries,
    ):
        advantages = torch.empty_like(reward_batch)
        running_advantage = torch.zeros_like(next_value_batch[-1])
        for reverse_step in reversed(range(reward_batch.shape[0])):
            delta = (
                reward_batch[reverse_step]
                + args.gamma
                * next_value_batch[reverse_step]
                * (1.0 - terminations[reverse_step])
                - value_batch[reverse_step]
            )
            continuing_advantage = (
                delta
                + args.gamma * args.gae_lambda * running_advantage
            )
            running_advantage = torch.where(
                boundaries[reverse_step], delta, continuing_advantage
            )
            advantages[reverse_step] = running_advantage
        return advantages, advantages + value_batch

    def recompute_rollout_targets(
        rollout_observations,
        rollout_next_observations,
        rollout_rewards,
        rollout_terminations,
        rollout_boundaries,
    ):
        with torch.no_grad():
            values = scalar_values(rollout_observations.reshape((-1,) + observation_shape))
            next_values = scalar_values(
                rollout_next_observations.reshape((-1,) + observation_shape)
            )
            values = values.view(rollout_rewards.shape)
            next_values = next_values.view(rollout_rewards.shape)
            return gae_from_values(
                rollout_rewards,
                values,
                next_values,
                rollout_terminations,
                rollout_boundaries,
            )

    def update_value_critic(
        rollout_observations,
        rollout_next_observations,
        rollout_rewards,
        rollout_terminations,
        rollout_boundaries,
    ):
        _, value_targets = recompute_rollout_targets(
            rollout_observations,
            rollout_next_observations,
            rollout_rewards,
            rollout_terminations,
            rollout_boundaries,
        )
        flat_observations = rollout_observations.reshape((-1,) + observation_shape)
        flat_targets = value_targets.reshape(-1).detach()
        with torch.no_grad():
            target_probs = hl_support.project_moment_matched(flat_targets)
        mark_compiled_step()
        value_logits = critic_logits_model(flat_observations)
        value_log_probs = F.log_softmax(value_logits.float(), dim=-1)
        value_loss = -(target_probs * value_log_probs).sum(dim=-1).mean()
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        value_optimizer.step()
        return value_loss.detach()

    def evaluate_actor(
        actor_model,
        observations,
        native_actions,
        weights,
        old_alpha,
        old_beta,
        old_log_prob,
    ):
        total_kl = torch.zeros((), device=device, dtype=torch.float64)
        total_gain = torch.zeros((), device=device, dtype=torch.float64)
        total_concentration = torch.zeros((), device=device, dtype=torch.float64)
        total_variance = torch.zeros((), device=device, dtype=torch.float64)
        outputs_finite = True
        for start in range(0, args.window_size, args.m_step_minibatches):
            stop = min(start + args.m_step_minibatches, args.window_size)
            mark_compiled_step()
            new_alpha, new_beta = actor_model(observations[start:stop])
            new_log_prob = beta_log_prob(
                new_alpha, new_beta, native_actions[start:stop]
            )
            kl = beta_kl(
                old_alpha[start:stop], old_beta[start:stop], new_alpha, new_beta
            )
            concentration = new_alpha + new_beta
            variance = new_alpha * new_beta / (
                concentration.square() * (concentration + 1.0)
            )
            outputs_finite = outputs_finite and bool(
                torch.isfinite(
                    torch.cat(
                        (
                            new_alpha.reshape(-1),
                            new_beta.reshape(-1),
                            new_log_prob.reshape(-1),
                            kl.reshape(-1),
                        )
                    )
                ).all().item()
            )
            total_kl += kl.double().sum()
            total_gain += (
                weights[start:stop] * (
                    new_log_prob - old_log_prob[start:stop]
                )
            ).double().sum()
            total_concentration += concentration.double().sum()
            total_variance += variance.double().sum()
        action_elements = args.window_size * int(np.prod(action_shape))
        return (
            (total_kl / args.window_size).float(),
            (total_gain / args.window_size).float(),
            (total_concentration / action_elements).float(),
            (total_variance / action_elements).float(),
            outputs_finite,
        )

    def transport_for_eta(centered_advantages, eta_value):
        """Project exp(A / eta) onto uniform row and environment marginals."""
        time_rows, environment_columns = centered_advantages.shape
        eta = torch.tensor(eta_value, device=device, dtype=torch.float64)
        log_kernel = centered_advantages / eta
        log_row_mass = math.log(environment_columns)
        log_column_mass = math.log(time_rows)
        log_row_scale = torch.zeros(time_rows, device=device, dtype=torch.float64)
        log_column_scale = torch.zeros(
            environment_columns, device=device, dtype=torch.float64
        )

        for _ in range(args.sinkhorn_iterations):
            log_row_scale = log_row_mass - torch.logsumexp(
                log_kernel + log_column_scale.unsqueeze(0), dim=1
            )
            log_column_scale = log_column_mass - torch.logsumexp(
                log_kernel + log_row_scale.unsqueeze(1), dim=0
            )
            # Gauge recentering keeps the two dual potentials similarly sized
            # without changing their sum or the represented transport.
            gauge = log_row_scale.mean()
            log_row_scale = log_row_scale - gauge
            log_column_scale = log_column_scale + gauge

        log_weights = (
            log_kernel
            + log_row_scale.unsqueeze(1)
            + log_column_scale.unsqueeze(0)
        )
        # Finite precision may underflow mathematically positive tail mass once
        # the transport becomes nearly deterministic.  Preserve that support at
        # the smallest representable float64 value, then rebalance to restore
        # both marginals after the floor.
        log_tiny = math.log(torch.finfo(torch.float64).tiny)
        weights = torch.exp(log_weights.clamp_min(log_tiny))
        for _ in range(args.sinkhorn_iterations):
            weights = weights * (
                environment_columns / weights.sum(dim=1, keepdim=True)
            )
            weights = weights * (time_rows / weights.sum(dim=0, keepdim=True))
        weights.clamp_min_(torch.finfo(torch.float64).tiny)

        row_residual = (weights.mean(dim=1) - 1.0).abs().max()
        column_residual = (weights.mean(dim=0) - 1.0).abs().max()
        e_step_kl = torch.special.xlogy(weights, weights).mean()
        ess_fraction = weights.sum().square() / (
            weights.numel() * weights.square().sum()
        )
        weighted_advantage_gain = (weights * centered_advantages).mean()
        return (
            weights,
            e_step_kl,
            ess_fraction,
            weights.max(),
            weighted_advantage_gain,
            row_residual,
            column_residual,
        )

    def solve_transport_estep(advantage_matrix):
        expected_shape = (
            args.window_batches * args.num_steps,
            args.num_envs,
        )
        if advantage_matrix.shape != expected_shape:
            raise RuntimeError(
                f"transport advantages must have shape {expected_shape}, "
                f"got {tuple(advantage_matrix.shape)}"
            )
        if not bool(torch.isfinite(advantage_matrix).all().item()):
            raise FloatingPointError("transport advantages must be finite")

        centered_advantages = advantage_matrix.double()
        centered_advantages = centered_advantages - centered_advantages.mean()

        with torch.no_grad():
            low_eta = ETA_FLOOR
            low_result = transport_for_eta(centered_advantages, low_eta)
            if bool((low_result[1] <= args.epsilon_eta).item()):
                eta = low_eta
                result = low_result
            else:
                high_eta = low_eta * 2.0
                for _ in range(ETA_BRACKET_STEPS):
                    high_result = transport_for_eta(centered_advantages, high_eta)
                    if bool((high_result[1] <= args.epsilon_eta).item()):
                        break
                    low_eta = high_eta
                    high_eta *= 2.0
                else:
                    raise RuntimeError("failed to bracket the transport eta")

                # KL decreases monotonically with eta.  Geometric bisection
                # resolves the shared positive scale without an optimizer.
                for _ in range(args.eta_bisection_steps):
                    middle_eta = math.sqrt(low_eta * high_eta)
                    middle_result = transport_for_eta(
                        centered_advantages, middle_eta
                    )
                    if bool((middle_result[1] > args.epsilon_eta).item()):
                        low_eta = middle_eta
                    else:
                        high_eta = middle_eta
                        high_result = middle_result
                eta = high_eta
                result = high_result

        (
            weights,
            e_step_kl,
            ess_fraction,
            max_weight,
            weighted_advantage_gain,
            row_residual,
            column_residual,
        ) = result
        global_mean_residual = (weights.mean() - 1.0).abs()
        transport_metrics = torch.stack(
            (
                torch.tensor(eta, device=device, dtype=torch.float64),
                e_step_kl,
                ess_fraction,
                max_weight,
                weighted_advantage_gain,
                row_residual,
                column_residual,
                global_mean_residual,
            )
        )
        transport_valid = (
            bool(torch.isfinite(transport_metrics).all().item())
            and bool(torch.isfinite(weights).all().item())
            and bool((weights > 0.0).all().item())
            and bool((row_residual <= MARGINAL_TOLERANCE).item())
            and bool((column_residual <= MARGINAL_TOLERANCE).item())
            and bool((global_mean_residual <= MARGINAL_TOLERANCE).item())
            and bool((e_step_kl <= args.epsilon_eta + ETA_KL_TOLERANCE).item())
            and (
                eta == ETA_FLOOR
                or bool(
                    (
                        (e_step_kl - args.epsilon_eta).abs()
                        <= ETA_KL_TOLERANCE
                    ).item()
                )
            )
        )
        if not transport_valid:
            raise FloatingPointError(
                "transport projection invalid: "
                f"eta={eta:.9g}, kl={e_step_kl.item():.12g}, "
                f"row_residual={row_residual.item():.12g}, "
                f"column_residual={column_residual.item():.12g}, "
                f"global_residual={global_mean_residual.item():.12g}, "
                f"min_weight={weights.min().item():.12g}, "
                f"max_weight={weights.max().item():.12g}"
            )
        return (
            transport_metrics[0].float(),
            weights.reshape(-1).detach(),
            e_step_kl.float(),
            ess_fraction.float(),
            max_weight.float(),
            weighted_advantage_gain.float(),
            row_residual.float(),
            column_residual.float(),
        )

    def fit_candidate(
        flat_observations,
        flat_actions,
        weights,
        old_alpha,
        old_beta,
        old_log_prob,
    ):
        candidate_actor.load_state_dict(deployed_actor.state_dict())
        candidate_optimizer = optim.Adam(
            candidate_actor.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )
        last_feasible_state = clone_parameters(candidate_actor)
        previous_feasible_gain = torch.zeros((), device=device)
        projected_fraction = 1.0
        epochs_consumed = 0
        crossing_detected = False
        converged = False

        for epoch in range(args.m_step_epochs):
            permutation = torch.randperm(args.window_size, device=device)
            for start in range(0, args.window_size, args.m_step_minibatches):
                indices = permutation[start : start + args.m_step_minibatches]
                mark_compiled_step()
                new_alpha, new_beta = candidate_policy_model(flat_observations[indices])
                log_prob = beta_log_prob(new_alpha, new_beta, flat_actions[indices])
                policy_loss = -(weights[indices] * log_prob).mean()
                candidate_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                candidate_optimizer.step()
            epochs_consumed = epoch + 1

            with torch.no_grad():
                full_kl, gain, _, _, outputs_finite = evaluate_actor(
                    candidate_policy_model,
                    flat_observations,
                    flat_actions,
                    weights,
                    old_alpha,
                    old_beta,
                    old_log_prob,
                )
            if outputs_finite and bool((full_kl <= args.epsilon_policy_kl).item()):
                last_feasible_state = clone_parameters(candidate_actor)
                gain_change = (gain - previous_feasible_gain).abs()
                convergence_scale = previous_feasible_gain.abs().clamp_min(1.0)
                if bool(
                    (gain_change <= args.m_step_convergence_tolerance * convergence_scale).item()
                ):
                    converged = True
                    break
                previous_feasible_gain = gain
                continue

            crossing_detected = outputs_finite and bool(
                torch.isfinite(full_kl).item()
            )
            if not crossing_detected:
                load_parameters(candidate_actor, last_feasible_state)
                break
            crossed_state = clone_parameters(candidate_actor)
            low_fraction = 0.0
            high_fraction = 1.0
            best_state = last_feasible_state
            for _ in range(POLICY_PROJECTION_STEPS):
                middle_fraction = 0.5 * (low_fraction + high_fraction)
                load_interpolated_parameters(
                    candidate_actor,
                    last_feasible_state,
                    crossed_state,
                    middle_fraction,
                )
                with torch.no_grad():
                    middle_kl, _, _, _, middle_finite = evaluate_actor(
                        candidate_policy_model,
                        flat_observations,
                        flat_actions,
                        weights,
                        old_alpha,
                        old_beta,
                        old_log_prob,
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
            break

        with torch.no_grad():
            final_kl, final_gain, concentration, native_variance, outputs_finite = (
                evaluate_actor(
                    candidate_policy_model,
                    flat_observations,
                    flat_actions,
                    weights,
                    old_alpha,
                    old_beta,
                    old_log_prob,
                )
            )
        budget_consumed = crossing_detected
        return (
            final_kl,
            final_gain,
            concentration,
            native_variance,
            outputs_finite,
            projected_fraction,
            epochs_consumed,
            budget_consumed,
            converged,
        )

    observation_means = np.zeros(
        (args.num_envs,) + observation_shape, dtype=np.float64
    )
    observation_variances = np.ones_like(observation_means)
    observation_counts = np.full(args.num_envs, 1e-4, dtype=np.float64)
    discounted_returns = np.zeros(args.num_envs, dtype=np.float64)
    reward_return_means = np.zeros(args.num_envs, dtype=np.float64)
    reward_return_variances = np.ones(args.num_envs, dtype=np.float64)
    reward_return_counts = np.full(args.num_envs, 1e-4, dtype=np.float64)

    cycle_shape = (args.window_batches, args.num_steps, args.num_envs)
    cycle_observations = torch.empty(
        cycle_shape + observation_shape, device=device
    )
    cycle_next_observations = torch.empty_like(cycle_observations)
    cycle_native_actions = torch.empty(cycle_shape + action_shape, device=device)
    cycle_rewards = torch.empty(cycle_shape, device=device)
    cycle_terminations = torch.empty(cycle_shape, device=device)
    cycle_boundaries = torch.empty(cycle_shape, device=device, dtype=torch.bool)

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
        warmup_observation = torch.as_tensor(
            next_observation_np, device=device, dtype=torch.float32
        )
        with torch.no_grad():
            mark_compiled_step()
            warmup_alpha, warmup_beta = deployed_policy_model(warmup_observation)
            warmup_native_action = Beta(warmup_alpha, warmup_beta).sample().clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
            warmup_action = deployed_actor.action_low + (
                deployed_actor.action_high - deployed_actor.action_low
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

    suppress_next_episode_log = phase_offsets > 0
    global_step = warmup_transitions
    next_observation = torch.as_tensor(
        next_observation_np, device=device, dtype=torch.float32
    )
    target_cycles = 0
    target_promotions = 0
    last_value_loss = torch.zeros((), device=device)

    for iteration in range(1, args.num_iterations + 1):
        cycle_index = (iteration - 1) % args.window_batches
        rollout_observations = cycle_observations[cycle_index]
        rollout_next_observations = cycle_next_observations[cycle_index]
        rollout_native_actions = cycle_native_actions[cycle_index]
        rollout_rewards = cycle_rewards[cycle_index]
        rollout_terminations = cycle_terminations[cycle_index]
        rollout_boundaries = cycle_boundaries[cycle_index]

        for step in range(args.num_steps):
            rollout_observations[step].copy_(next_observation)
            with torch.no_grad():
                mark_compiled_step()
                alpha, beta = deployed_policy_model(next_observation)
                distribution = Beta(alpha, beta)
                native_action = distribution.sample().clamp(
                    SAMPLE_EPS, 1.0 - SAMPLE_EPS
                )
                action = deployed_actor.action_low + (
                    deployed_actor.action_high - deployed_actor.action_low
                ) * native_action
            rollout_native_actions[step].copy_(native_action)

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
            rollout_rewards[step].copy_(
                torch.as_tensor(reward, device=device, dtype=torch.float32)
            )
            rollout_terminations[step].copy_(
                torch.as_tensor(terminations, device=device, dtype=torch.float32)
            )
            rollout_boundaries[step].copy_(
                torch.as_tensor(boundary, device=device, dtype=torch.bool)
            )
            rollout_next_observations[step].copy_(
                torch.as_tensor(
                    normalized_transition_observations,
                    device=device,
                    dtype=torch.float32,
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

        last_value_loss = update_value_critic(
            rollout_observations,
            rollout_next_observations,
            rollout_rewards,
            rollout_terminations,
            rollout_boundaries,
        )

        if cycle_index + 1 != args.window_batches:
            continue

        with torch.no_grad():
            sequence_observations = cycle_observations.reshape(
                (args.window_batches * args.num_steps, args.num_envs) + observation_shape
            )
            sequence_next_observations = cycle_next_observations.reshape(
                (args.window_batches * args.num_steps, args.num_envs) + observation_shape
            )
            sequence_rewards = cycle_rewards.reshape(
                args.window_batches * args.num_steps, args.num_envs
            )
            sequence_terminations = cycle_terminations.reshape_as(sequence_rewards)
            sequence_boundaries = cycle_boundaries.reshape_as(sequence_rewards)
            advantages, value_targets = recompute_rollout_targets(
                sequence_observations,
                sequence_next_observations,
                sequence_rewards,
                sequence_terminations,
                sequence_boundaries,
            )
            current_values = value_targets - advantages
            value_error = current_values - value_targets
            critic_rmse = value_error.square().mean().sqrt()
            explained_variance = 1.0 - value_error.var(unbiased=False) / (
                value_targets.var(unbiased=False) + 1e-8
            )
            return_percentiles = torch.quantile(
                value_targets.reshape(-1), return_percentile_levels
            )
            return_scale = (
                return_percentiles[1] - return_percentiles[0]
            ).clamp_min(args.return_percentile_floor)
            # Layout is explicitly [window_batch, step, environment].  Merging
            # only the first two axes gives the [312, 64] transport matrix;
            # flattening that matrix therefore matches observation/action
            # flattening exactly, with environment as the fastest sample axis.
            advantage_matrix = (
                advantages.detach() / return_scale
            ).reshape(args.window_batches * args.num_steps, args.num_envs)
            flat_observations = cycle_observations.reshape(
                (args.window_size,) + observation_shape
            )
            flat_actions = cycle_native_actions.reshape(
                (args.window_size,) + action_shape
            )
            if advantage_matrix.numel() != flat_observations.shape[0]:
                raise RuntimeError("transport and policy window order disagree")

        (
            eta,
            weights,
            achieved_e_step_kl,
            ess_fraction,
            max_weight,
            weighted_advantage_gain,
            row_mean_residual,
            column_mean_residual,
        ) = solve_transport_estep(advantage_matrix)


        with torch.no_grad():
            # Compiled CUDAGraph outputs may reuse their carrier storage on
            # the next marked step, so preserve each target-policy chunk now.
            old_alpha = torch.empty_like(flat_actions)
            old_beta = torch.empty_like(flat_actions)
            for start in range(0, args.window_size, args.m_step_minibatches):
                stop = min(start + args.m_step_minibatches, args.window_size)
                mark_compiled_step()
                part_alpha, part_beta = deployed_policy_model(
                    flat_observations[start:stop]
                )
                old_alpha[start:stop].copy_(part_alpha)
                old_beta[start:stop].copy_(part_beta)
            old_log_prob = beta_log_prob(old_alpha, old_beta, flat_actions)

        (
            final_kl,
            candidate_gain,
            policy_concentration,
            policy_native_variance,
            actor_outputs_finite,
            projected_fraction,
            epochs_consumed,
            budget_consumed,
            converged,
        ) = fit_candidate(
            flat_observations,
            flat_actions,
            weights,
            old_alpha,
            old_beta,
            old_log_prob,
        )

        actor_metrics = torch.stack(
            (
                final_kl,
                candidate_gain,
                policy_concentration,
                policy_native_variance,
                critic_rmse,
                explained_variance,
                return_scale,
            )
        )
        all_actor_metrics_finite = actor_outputs_finite and bool(
            torch.isfinite(actor_metrics).all().item()
        )
        promote = (
            all_actor_metrics_finite
            and bool((candidate_gain > 0.0).item())
            and bool(
                (final_kl <= args.epsilon_policy_kl + POLICY_KL_TOLERANCE).item()
            )
        )
        if promote:
            with torch.no_grad():
                deployed_actor.load_state_dict(candidate_actor.state_dict())
            target_promotions += 1
        else:
            candidate_actor.load_state_dict(deployed_actor.state_dict())

        target_cycles += 1
        promotion_rate = target_promotions / target_cycles
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/value_loss", float(last_value_loss.item()), global_step)
        writer.add_scalar("vmpo/eta", float(eta.item()), global_step)
        writer.add_scalar(
            "vmpo/achieved_mean_w_log_w",
            float(achieved_e_step_kl.item()),
            global_step,
        )
        writer.add_scalar(
            "vmpo/transport_row_mean_residual_max",
            float(row_mean_residual.item()),
            global_step,
        )
        writer.add_scalar(
            "vmpo/transport_column_mean_residual_max",
            float(column_mean_residual.item()),
            global_step,
        )
        writer.add_scalar(
            "vmpo/weighted_advantage_gain",
            float(weighted_advantage_gain.item()),
            global_step,
        )
        writer.add_scalar("vmpo/weight_ess_fraction", float(ess_fraction.item()), global_step)
        writer.add_scalar("vmpo/max_weight", float(max_weight.item()), global_step)
        writer.add_scalar("vmpo/candidate_gain", float(candidate_gain.item()), global_step)
        writer.add_scalar("vmpo/projected_fraction", projected_fraction, global_step)
        writer.add_scalar("vmpo/epochs_consumed", epochs_consumed, global_step)
        writer.add_scalar("vmpo/full_beta_kl", float(final_kl.item()), global_step)
        writer.add_scalar("vmpo/budget_consumed", float(budget_consumed), global_step)
        writer.add_scalar("vmpo/candidate_converged", float(converged), global_step)
        writer.add_scalar("vmpo/target_promoted", float(promote), global_step)
        writer.add_scalar("vmpo/target_promotions", target_promotions, global_step)
        writer.add_scalar("vmpo/promotion_rate", promotion_rate, global_step)
        writer.add_scalar("vmpo/target_cycles", target_cycles, global_step)
        writer.add_scalar(
            "debug/policy_concentration", float(policy_concentration.item()), global_step
        )
        writer.add_scalar(
            "debug/policy_native_variance",
            float(policy_native_variance.item()),
            global_step,
        )
        writer.add_scalar("debug/value_rmse", float(critic_rmse.item()), global_step)
        writer.add_scalar(
            "debug/value_explained_variance",
            float(explained_variance.item()),
            global_step,
        )
        writer.add_scalar("debug/return_percentile_scale", float(return_scale.item()), global_step)
        print(f"SPS: {sps}")

    envs.close()
    writer.close()

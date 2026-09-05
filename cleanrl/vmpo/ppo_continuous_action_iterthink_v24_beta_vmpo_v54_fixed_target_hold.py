# V-MPO v54 — v53 with a strictly fixed target-policy hold.
#
# Hold the behavior/KL reference for exactly target_update_period learner
# updates (default 100); KL violations never trigger early promotion.
# Hypothesis: stationary KL origins let the existing Adam duals settle.
# Model, raw GAE advantages, loss, KL budgets and optimizer are unchanged.
# Shared normalization/warmup/GAE and native-launcher support change execution
# only. Full-batch next-value evaluation preserves v53's BF16 batch geometry.

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
from cleanrl.shared.ppo_loop import gather_metrics, get_gae_fn
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets,
    episode_horizon,
    run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm, make_raw_continuous_env

SAMPLE_EPS = 1e-6
DUAL_FLOOR = 1e-8


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
    topk_size: int = 0
    num_iterations: int = 0
    initial_phase_warmup_steps: int = 0


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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.topk_size = int(args.batch_size * args.topk_fraction)
    if not 0 < args.topk_size <= args.batch_size:
        raise ValueError("topk_fraction produces an invalid global top-k size")
    args.initial_phase_warmup_steps = episode_horizon(args.env_id)
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
    configure_runtime(cudnn_deterministic=args.torch_deterministic)
    device = torch.device("cuda")
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
            make_raw_continuous_env(args.env_id, index, args.capture_video, run_name)
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
            target_alpha, target_beta = target_agent.policy(observations)
            value_logits = agent.value_logits(observations)
        values = hl_support.to_scalar(value_logits.float())
        return target_alpha.float(), target_beta.float(), values


    gae_fn = get_gae_fn(
        compiled=args.compile, mode=args.compile_mode, explicit_next_values=True
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
        # Passing the union mask as truncations is equivalent: shared GAE
        # ORs it with terminations and retains terminated-only bootstrapping.
        return gae_fn(
            reward_batch, value_batch, terminations, boundaries,
            next_values, args.gamma, args.gae_lambda,
        )

    def update_loss_model(
        observations,
        native_actions,
        old_alpha,
        old_beta,
        advantages,
        value_targets,
    ):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            new_alpha, new_beta, value_logits = agent(observations)
        new_alpha = new_alpha.float()
        new_beta = new_beta.float()
        value_log_probs = F.log_softmax(value_logits.float(), dim=-1)
        value_probs = value_log_probs.exp()
        values = hl_support.probs_to_scalar(value_probs)

        alpha_mean = duals[0].clamp_min(DUAL_FLOOR)
        alpha_concentration = duals[1].clamp_min(DUAL_FLOOR)

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

        log_prob = beta_log_prob(new_alpha, new_beta, native_actions)
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
        with torch.no_grad():
            value_target_probs = hl_support.project_moment_matched(value_targets)
        value_loss = -(value_target_probs * value_log_probs).sum(dim=-1).mean()
        total_loss = (
            policy_loss
            + mean_constraint_loss
            + concentration_constraint_loss
            + value_loss
        )

        full_kl = beta_kl(old_alpha, old_beta, new_alpha, new_beta).mean()
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
    obs_norm = VectorObsNorm(args.num_envs, observation_shape)
    rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
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
    phase_offsets = compute_phase_offsets(
        args.num_envs, args.initial_phase_warmup_steps, args.seed
    )
    writer.add_text("initial_phase_offsets", ",".join(map(str, phase_offsets)))

    @torch.no_grad()
    def warmup_action(next_observation_np):
        warmup_observation = torch.as_tensor(
            next_observation_np, device=device, dtype=torch.float32
        )
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        warmup_alpha, warmup_beta, _ = rollout_model(warmup_observation)
        warmup_native_action = Beta(warmup_alpha, warmup_beta).sample().clamp(
            SAMPLE_EPS, 1.0 - SAMPLE_EPS
        )
        action = target_agent.action_low + (
            target_agent.action_high - target_agent.action_low
        ) * warmup_native_action
        return action.cpu().numpy()

    warm = run_phase_warmup(
        envs, obs_norm=obs_norm, rew_norm=rew_norm,
        act_fn=warmup_action, horizon=args.initial_phase_warmup_steps,
        phase_offsets=phase_offsets, seed=args.seed,
    )
    next_observation_np = warm.next_obs
    suppress_next_episode_log = warm.suppress_mask
    global_step = warm.transitions
    writer.add_scalar("timing/warmup_s", time.time() - start_time, global_step)
    next_observation = torch.as_tensor(
        next_observation_np,
        device=device,
        dtype=torch.float32,
    )
    target_age_batches = 0
    target_promotions = 0
    timer = PhaseTimer()
    interval_start, interval_step = time.perf_counter(), global_step


    for iteration in range(1, args.num_iterations + 1):
        rollout_target_age_batches = target_age_batches
        for step in range(args.num_steps):
            timer.start("rollout", use_cuda=False)
            observations[step].copy_(next_observation)
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                old_alpha, old_beta, value = rollout_model(next_observation)
                distribution = Beta(old_alpha, old_beta)
                native_action = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                action = target_agent.action_low + (
                    target_agent.action_high - target_agent.action_low
                ) * native_action
            native_actions[step].copy_(native_action)
            old_alphas[step].copy_(old_alpha)
            old_betas[step].copy_(old_beta)
            values[step].copy_(value)

            timer.stop()
            timer.start("env", use_cuda=False)
            (
                raw_next_observations,
                raw_reward,
                terminations,
                truncations,
                infos,
            ) = envs.step(action.cpu().numpy())
            timer.stop()
            timer.start("normalize_transfer", use_cuda=False)
            reward = rew_norm.normalize(raw_reward, terminations)
            boundary = np.logical_or(terminations, truncations)
            next_observation_np, normalized_transition_observations = obs_norm.normalize_step(
                raw_next_observations, terminations, truncations, infos
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
            timer.stop()

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

        timer.start("gae")
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
            del gae_advantages, returns
        timer.stop()
        timer.start("update")

        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        total_loss, metrics = update_loss_model(
            observations.reshape((args.batch_size,) + observation_shape),
            native_actions.reshape((args.batch_size,) + action_shape),
            old_alphas.reshape((args.batch_size,) + action_shape),
            old_betas.reshape((args.batch_size,) + action_shape),
            advantages,
            value_targets,
        )
        should_log = (
            iteration % args.log_interval == 0 or iteration == 1
            or target_age_batches + 1 >= args.target_update_period
        )
        duals_before = duals.detach().clone() if should_log else None
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            duals.clamp_(min=DUAL_FLOOR)
        timer.stop()
        target_age_batches += 1
        # Exactly one learner update per iteration. Promotion is independent
        # of log cadence and never resets the duals or their Adam momentum.
        promote_for_mean_kl = False
        promote_for_max_age = target_age_batches >= args.target_update_period
        target_promoted = promote_for_max_age
        if target_promoted:
            with torch.no_grad():
                target_agent.load_state_dict(agent.state_dict())
            target_age_batches = 0
            target_promotions += 1

        if should_log:
            assert duals_before is not None
            dual_delta = duals.detach() - duals_before
            log_tensors = torch.cat((metrics, duals.detach(), dual_delta))
            packed = gather_metrics(dict(enumerate(log_tensors.unbind()))).values()
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
                alpha_mean_value,
                alpha_concentration_value,
                alpha_mean_delta_value,
                alpha_concentration_delta_value,
            ) = packed
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
            now = time.perf_counter()
            writer.add_scalar(
                "charts/interval_SPS",
                (global_step - interval_step) / (now - interval_start), global_step,
            )
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            interval_start, interval_step = time.perf_counter(), global_step
            print(
                f"SPS: {sps}, learner_update={iteration}, "
                f"target_age={rollout_target_age_batches}, promoted={target_promoted}"
            )

    envs.close()
    writer.close()

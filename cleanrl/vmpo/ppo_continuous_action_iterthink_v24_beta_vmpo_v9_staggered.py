# V-MPO v9 STAGGERED — exact eta with phase-diverse synchronous rollouts.
#
# SyncVectorEnv advances every environment once per vector step. Equal
# 1,000-step time limits therefore keep all 64 HalfCheetah environments at the
# same episode age, so each 39-step learner batch sees a narrow temporal phase.
#
# Before recording training data, this version runs one episode-horizon of
# stochastic actions from the seeded initial target policy. During that warmup,
# environments are reset on a deterministic evenly spaced phase schedule,
# permuted by a seed-local RNG. At warmup completion their episode ages cover
# [0, horizon) at intervals of horizon / num_envs. Normal episode horizons are
# unchanged thereafter. Warmup interactions count against total timesteps.
#
# The exact-temperature E-step, one joint Adam epoch per fresh rollout,
# full-width ReLU-squared IterThink Beta model, and all other v8 behavior remain.

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

    target_update_period: int = 100
    topk_fraction: float = 0.5
    epsilon_eta: float = 0.01
    # Neutral geometric midpoints of the paper's Gym log-uniform search ranges.
    epsilon_alpha_mean: float = 0.007071067811865476
    epsilon_alpha_concentration: float = 1.5811388300841898e-5
    initial_alpha_mean: float = 1.0
    initial_alpha_concentration: float = 1.0

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


def bootstrap_observations(next_observations, truncations, infos):
    bootstrap = np.array(next_observations, copy=True)
    if not np.any(truncations):
        return bootstrap
    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None:
        raise RuntimeError("truncated transition missing final_observation")
    for env_index in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_index]:
            raise RuntimeError(f"truncated environment {env_index} has no final observation")
        final_observation = final_observations[env_index]
        if final_observation is None:
            raise RuntimeError(f"truncated environment {env_index} has no final observation")
        bootstrap[env_index] = final_observation
    return bootstrap


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
            normalized_value = agent.value(observations)
        value = normalized_value.float() * agent.popart_std + agent.popart_mean
        return target_alpha.float(), target_beta.float(), value


    def bootstrap_returns_model(
        transition_next_observations,
        reward_batch,
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
        output = torch.empty_like(reward_batch)
        running_return = next_values[-1]
        for reverse_step in reversed(range(args.num_steps)):
            boundary_return = reward_batch[reverse_step] + (
                args.gamma
                * next_values[reverse_step]
                * (1.0 - terminations[reverse_step])
            )
            continuing_return = reward_batch[reverse_step] + args.gamma * running_return
            running_return = torch.where(
                boundaries[reverse_step],
                boundary_return,
                continuing_return,
            )
            output[reverse_step] = running_return
        return output

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
        value_loss = 0.5 * (normalized_values - normalized_returns).square().mean()
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
        value_error = (
            normalized_values - normalized_returns
        ) * agent.popart_std
        unnormalized_returns = (
            normalized_returns * agent.popart_std + agent.popart_mean
        )
        value_rmse = value_error.square().mean().sqrt()
        explained_variance = 1.0 - value_error.var(unbiased=False) / (
            unnormalized_returns.var(unbiased=False) + 1e-8
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
        bootstrap_returns_model = torch.compile(
            bootstrap_returns_model,
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
    next_observation_np, _ = envs.reset(seed=args.seed)

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
            warmup_action = target_agent.action_low + (
                target_agent.action_high - target_agent.action_low
            ) * warmup_native_action
        next_observation_np, _, _, _, _ = envs.step(warmup_action.cpu().numpy())

        for env_index in np.flatnonzero(scheduled_resets == warmup_step):
            reset_observation, _ = envs.envs[env_index].reset()
            next_observation_np[env_index] = reset_observation

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
                action = target_agent.action_low + (
                    target_agent.action_high - target_agent.action_low
                ) * native_action
            native_actions[step].copy_(native_action)
            old_alphas[step].copy_(old_alpha)
            old_betas[step].copy_(old_beta)
            values[step].copy_(value)

            (
                next_observation_np,
                reward,
                terminations,
                truncations,
                infos,
            ) = envs.step(action.cpu().numpy())
            boundary = np.logical_or(terminations, truncations)
            transition_next_observation = bootstrap_observations(
                next_observation_np, truncations, infos
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
            returns = bootstrap_returns_model(
                next_observations,
                rewards,
                terminations_buffer,
                boundaries_buffer,
            )
            advantages = (returns - values).reshape(-1).detach()
            flat_returns = returns.reshape(-1)
            normalized_returns = agent.update_popart(
                flat_returns,
                args.popart_rate,
                args.popart_std_min,
                args.popart_std_max,
            ).detach()

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
        duals_before = duals.detach().clone() if should_log else None
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            duals.clamp_(min=DUAL_FLOOR)
            if iteration % args.target_update_period == 0:
                target_agent.load_state_dict(agent.state_dict())

        if should_log:
            assert duals_before is not None
            dual_delta = duals.detach() - duals_before
            packed = torch.cat(
                (
                    metrics,
                    duals.detach(),
                    dual_delta,
                    agent.popart_mean.view(1),
                    agent.popart_std.view(1),
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
                policy_concentration_value,
                policy_variance_value,
                eta_value,
                alpha_mean_value,
                alpha_concentration_value,
                alpha_mean_delta_value,
                alpha_concentration_delta_value,
                popart_mean_value,
                popart_std_value,
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
            target_age_batches = (iteration - 1) % args.target_update_period
            writer.add_scalar(
                "vmpo/target_age_batches", target_age_batches, global_step
            )
            writer.add_scalar(
                "vmpo/target_age_transitions",
                target_age_batches * args.batch_size,
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

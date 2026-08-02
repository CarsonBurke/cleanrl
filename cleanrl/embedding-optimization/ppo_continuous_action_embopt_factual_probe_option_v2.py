# Embedding Optimization: Factual-Probe Latent Options v2
#
# A LeWM-style causal vector world model learns an unrestricted Euclidean chart
# from factual observation/action sequences with attached-target MSE and SIGReg.
# Persistent free latent goals stop through pessimistic C(state, goal)-H(state).
# A deterministic balanced action probe gives the policy factual credit from the
# encoded actual successor; no policy gradient crosses the world model.  Task
# reward trains the goal/value system, while the default low-level utility is
# goal progress minus exact control cost.  There are no policy likelihoods,
# action-values, baselines, target models, moving averages, relabeled goals, or
# online action search.
import contextlib
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


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
    num_envs: int = 16
    num_steps: int = 128
    history_length: int = 3
    replay_capacity: int = 131_072
    recent_replay_size: int = 65_536
    warmup_steps: int = 32_768
    minibatch_size: int = 512
    model_updates_per_iteration: int = 16

    latent_dim: int = 64
    hidden_dim: int = 256
    transformer_layers: int = 3
    transformer_heads: int = 4
    option_heads: int = 5
    bootstrap_probability: float = 0.8

    world_lr: float = 1e-4
    h_lr: float = 3e-4
    c_lr: float = 3e-4
    goal_lr: float = 1e-4
    policy_lr: float = 1e-4
    reward_rate_lr: float = 3e-4
    weight_decay: float = 1e-4
    sigreg_coef: float = 0.09
    reward_scale: float = 0.1
    average_reward: bool = True
    option_discount: float = 1.0
    discounted_option_discount: float = 0.99
    manager_lambda: float = 0.95
    pessimism_coef: float = 1.0
    action_probe_amplitude: float = 0.08
    goal_probe_amplitude: float = 0.5
    utility_reward_coef: float = 0.0
    max_grad_norm: float = 0.5
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    batch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        kwargs = {"render_mode": "rgb_array"} if capture_video and idx == 0 else {}
        env = gym.make(env_id, **kwargs)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def factual_next_observations(next_observations, infos):
    factual = np.array(next_observations, copy=True)
    if "final_observation" not in infos:
        return factual
    mask = infos.get(
        "_final_observation",
        np.asarray([x is not None for x in infos["final_observation"]], dtype=bool),
    )
    for index in np.flatnonzero(mask):
        factual[index] = infos["final_observation"][index]
    return factual


def initialize_context(observations, history_length, action_dim):
    observations = np.asarray(observations, dtype=np.float32)
    obs_context = np.repeat(observations[:, None], history_length, axis=1)
    past_actions = np.zeros(
        (len(observations), history_length - 1, action_dim), dtype=np.float32
    )
    valid = np.zeros((len(observations), history_length), dtype=bool)
    valid[:, -1] = True
    return obs_context, past_actions, valid


def transition_sequence(obs_context, past_actions, valid, action, factual_next):
    actions = np.concatenate([past_actions, action[:, None]], axis=1)
    observations = np.concatenate([obs_context, factual_next[:, None]], axis=1)
    observation_valid = np.concatenate(
        [valid, np.ones((len(valid), 1), dtype=bool)], axis=1
    )
    return observations, actions, observation_valid


def next_live_context(
    observations,
    actions,
    observation_valid,
    autoreset_observations,
    done,
):
    obs_context = np.array(observations[:, 1:], copy=True)
    past_actions = np.array(actions[:, 1:], copy=True)
    valid = np.array(observation_valid[:, 1:], copy=True)
    if np.any(done):
        reset = initialize_context(
            np.asarray(autoreset_observations)[done],
            obs_context.shape[1],
            actions.shape[-1],
        )
        obs_context[done], past_actions[done], valid[done] = reset
    return obs_context, past_actions, valid


def counter_hash32(value):
    value &= 0xFFFF_FFFF
    value = ((value >> 16) ^ value) * 0x045D_9F3B & 0xFFFF_FFFF
    value = ((value >> 16) ^ value) * 0x045D_9F3B & 0xFFFF_FFFF
    return (value >> 16) ^ value


def deterministic_balanced_probe(step, count, dimension, amplitude, device=None):
    """Counter-rotated Walsh probes; exactly balanced when count is a power of two."""
    if count < 2:
        raise ValueError("balanced probes require at least two parallel environments")
    order = 1 << (count - 1).bit_length()
    row_seed = counter_hash32(step)
    row = torch.arange(count, device=device, dtype=torch.int64)
    odd_multiplier = 2 * (row_seed % max(1, order // 2)) + 1
    shift = counter_hash32(step + 0x9E37_79B9) % order
    row = (row * odd_multiplier + shift) % order
    coordinate = torch.arange(dimension, device=device, dtype=torch.int64)
    strides = [
        candidate
        for candidate in range(1, order)
        if math.gcd(candidate, order - 1) == 1
    ]
    stride = strides[counter_hash32(step + 1) % len(strides)]
    offset = counter_hash32(step + 2) % (order - 1)
    columns = 1 + (coordinate * stride + offset) % (order - 1)
    intersection = row[:, None].bitwise_and(columns[None])
    parity = torch.zeros_like(intersection)
    for bit in range(order.bit_length() - 1):
        parity.bitwise_xor_((intersection >> bit).bitwise_and(1))
    signs = 1 - 2 * parity
    sign_seed = counter_hash32(step + 3)
    column_sign = 1 - 2 * (
        coordinate * (2 * (sign_seed % 8) + 1) + (sign_seed >> 3)
    ).remainder(2)
    return amplitude * (signs * column_sign[None]).to(torch.float32)


def deterministic_action_probe(
    step, num_envs, action_dim, amplitude, device=None
):
    return deterministic_balanced_probe(
        step, num_envs, action_dim, amplitude, device
    )


def deterministic_goal_probe(
    step, num_envs, latent_dim, amplitude, device=None
):
    return deterministic_balanced_probe(
        step ^ 0xA511_E9B3, num_envs, latent_dim, amplitude, device
    )


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden_dim, out_dim, out_std=1.0, zero_output=False):
    output = layer_init(nn.Linear(hidden_dim, out_dim), std=out_std)
    if zero_output:
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden_dim)),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        layer_init(nn.Linear(hidden_dim, hidden_dim)),
        nn.SiLU(),
        output,
    )


class SIGReg(nn.Module):
    def __init__(self, projections=256, knots=17, reference_samples=128):
        super().__init__()
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2)
        self.projections = projections
        self.reference_samples = reference_samples
        self.register_buffer("t", t)
        self.register_buffer("gaussian_cf", gaussian_cf)
        self.register_buffer("weights", weights * gaussian_cf)

    def forward(self, x):
        axes = torch.randn(
            x.shape[-1], self.projections, device=x.device, dtype=x.dtype
        )
        axes = axes / axes.norm(dim=0, keepdim=True).clamp_min(1e-8)
        phase = (x @ axes).unsqueeze(-1) * self.t
        error = (
            (phase.cos().mean(0) - self.gaussian_cf).square()
            + phase.sin().mean(0).square()
        )
        return ((error @ self.weights) * self.reference_samples).mean()


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ActionAdaLNBlock(nn.Module):
    """Faithful vector AdaLN-zero block: every outgoing action conditions every block."""

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self, x, action_condition, causal_mask, key_padding_mask=None, token_valid=None
    ):
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(action_condition).chunk(6, dim=-1)
        query = modulate(self.norm1(x), shift_attn, scale_attn)
        attended = self.attention(
            query,
            query,
            query,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + gate_attn * attended
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        if token_valid is not None:
            x = torch.where(token_valid.unsqueeze(-1), x, torch.zeros_like(x))
        return x


class CausalActionWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_dim,
        hidden_dim,
        history_length,
        transformer_layers,
        transformer_heads,
    ):
        super().__init__()
        self.history_length = history_length
        self.observation_encoder = mlp(obs_dim, hidden_dim, latent_dim)
        self.encoder_projector = mlp(latent_dim, 2 * latent_dim, latent_dim)
        self.action_encoder = mlp(action_dim, hidden_dim, latent_dim, out_std=0.1)
        self.position = nn.Parameter(torch.zeros(1, history_length, latent_dim))
        nn.init.normal_(self.position, std=0.01)
        self.predictor = nn.ModuleList(
            [ActionAdaLNBlock(latent_dim, transformer_heads) for _ in range(transformer_layers)]
        )
        self.predictor_norm = nn.LayerNorm(latent_dim)
        self.predictor_projector = mlp(
            latent_dim, 2 * latent_dim, latent_dim, out_std=0.01
        )
        self.register_buffer("alignment", torch.eye(latent_dim))

    def encode_raw(self, observations):
        return self.encoder_projector(self.observation_encoder(observations))

    def encode(self, observations):
        return self.encode_raw(observations) @ self.alignment

    @staticmethod
    def causal_mask(length, device):
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def predict_from_latents(
        self, source_latents, outgoing_actions, source_valid=None
    ):
        if source_latents.shape[:2] != outgoing_actions.shape[:2]:
            raise ValueError("each source observation must have one aligned outgoing action")
        length = source_latents.shape[1]
        x = source_latents + self.position[:, :length]
        if source_valid is not None:
            x = torch.where(source_valid.unsqueeze(-1), x, torch.zeros_like(x))
        condition = self.action_encoder(outgoing_actions)
        mask = self.causal_mask(length, x.device)
        key_padding_mask = None if source_valid is None else ~source_valid
        for block in self.predictor:
            x = block(x, condition, mask, key_padding_mask, source_valid)
        return self.predictor_projector(self.predictor_norm(x))

    def predict_sequence(
        self, source_observations, outgoing_actions, source_valid=None
    ):
        return self.predict_from_latents(
            self.encode(source_observations), outgoing_actions, source_valid
        )

    def predict_final(
        self, observation_context, outgoing_actions, source_valid=None
    ):
        return self.predict_sequence(
            observation_context, outgoing_actions, source_valid
        )[:, -1]


class GoalProposal(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.network = mlp(latent_dim, hidden_dim, latent_dim, out_std=0.01)

    def forward(self, z):
        return self.network(z)


class GoalPolicy(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.network = mlp(3 * latent_dim, hidden_dim, action_dim, out_std=0.01)

    def raw_action(self, z, goal_z):
        return self.network(torch.cat([z, goal_z, goal_z - z], dim=-1))

    def forward(self, z, goal_z):
        return torch.tanh(self.raw_action(z, goal_z))


class DifferentialRewardRate(nn.Module):
    """Learned average reward; optimized from detached Bellman residual targets."""

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self):
        return self.value


class StateBiasEnsemble(nn.Module):
    def __init__(self, latent_dim, hidden_dim, heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [mlp(latent_dim, hidden_dim, 1, zero_output=True) for _ in range(heads)]
        )

    def forward(self, z):
        return torch.stack([head(z).squeeze(-1) for head in self.heads], dim=0)


class ContinuationEnsemble(nn.Module):
    def __init__(self, latent_dim, hidden_dim, heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [mlp(3 * latent_dim, hidden_dim, 1, zero_output=True) for _ in range(heads)]
        )

    def forward(self, z, goal_z):
        features = torch.cat([z, goal_z, goal_z - z], dim=-1)
        return torch.stack([head(features).squeeze(-1) for head in self.heads], dim=0)


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, args):
        super().__init__()
        self.world = CausalActionWorldModel(
            obs_dim,
            action_dim,
            args.latent_dim,
            args.hidden_dim,
            args.history_length,
            args.transformer_layers,
            args.transformer_heads,
        )
        self.h = StateBiasEnsemble(
            args.latent_dim, args.hidden_dim, args.option_heads
        )
        self.c = ContinuationEnsemble(
            args.latent_dim, args.hidden_dim, args.option_heads
        )
        self.goal = GoalProposal(args.latent_dim, args.hidden_dim)
        self.policy = GoalPolicy(args.latent_dim, args.hidden_dim, action_dim)
        self.reward_rate = DifferentialRewardRate()


@contextlib.contextmanager
def frozen_parameters(*modules):
    parameters = [p for module in modules for p in module.parameters()]
    previous = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, required in zip(parameters, previous):
            parameter.requires_grad_(required)


def masked_world_objective(
    world, sigreg, observations, actions, observation_valid, sigreg_coef
):
    """Token j is exactly (o_j, outgoing a_j) -> attached E(o_{j+1})."""
    encoded = world.encode(observations)
    source_valid = observation_valid[:, :-1]
    # Invalid repeated-reset padding is a constant null token: it contributes
    # neither an encoder gradient nor a fabricated transition target.
    source_encoded = torch.where(
        source_valid.unsqueeze(-1), encoded[:, :-1], torch.zeros_like(encoded[:, :-1])
    )
    predicted = world.predict_from_latents(
        source_encoded, actions, source_valid=source_valid
    )
    valid_targets = observation_valid[:, :-1] & observation_valid[:, 1:]
    per_token = (predicted - encoded[:, 1:]).square().mean(dim=-1)
    prediction_loss = (
        per_token * valid_targets
    ).sum() / valid_targets.sum().clamp_min(1)
    sigreg_loss = sigreg(encoded[observation_valid])
    return (
        prediction_loss + sigreg_coef * sigreg_loss,
        prediction_loss,
        sigreg_loss,
    )


def surplus_heads(h_values, c_values):
    return c_values - h_values


def pessimistic_surplus(h_values, c_values, coefficient):
    differences = surplus_heads(h_values, c_values)
    return differences.mean(dim=0) - coefficient * differences.std(
        dim=0, unbiased=False
    )


def continue_decision(h_values, c_values, coefficient):
    return pessimistic_surplus(h_values, c_values, coefficient) > 0


def select_next_branch(h_values, c_values, coefficient):
    """Choose one aggregate branch, then retain corresponding per-head values."""
    continuing = continue_decision(h_values, c_values, coefficient)
    return torch.where(continuing.unsqueeze(0), c_values, h_values), continuing


def continuation_targets(
    adjusted_rewards, next_h, next_c, discount, bootstrap, coefficient
):
    selected, continuing = select_next_branch(next_h, next_c, coefficient)
    targets = adjusted_rewards.unsqueeze(0) + discount * bootstrap.unsqueeze(0) * selected
    return targets.detach(), continuing


def manager_lambda_returns(
    adjusted_rewards,
    next_manager_values,
    bootstrap,
    trace,
    discount,
    manager_lambda,
):
    """Factual manager returns with a per-head shared-decision bootstrap."""
    returns = torch.empty_like(next_manager_values)
    accumulator = next_manager_values[-1]
    for step in reversed(range(adjusted_rewards.shape[0])):
        baseline = next_manager_values[step]
        continuation = baseline + manager_lambda * trace[step].unsqueeze(-1) * (
            accumulator - baseline
        )
        accumulator = adjusted_rewards[step].unsqueeze(-1) + (
            discount * bootstrap[step].unsqueeze(-1) * continuation
        )
        returns[step] = accumulator
    return returns.detach()


class CumulativeRewardRate:
    def __init__(self):
        self.reward_sum = 0.0
        self.step_count = 0

    @property
    def value(self):
        return self.reward_sum / self.step_count if self.step_count else 0.0

    def update(self, scaled_rewards):
        rewards = np.asarray(scaled_rewards, dtype=np.float64)
        self.reward_sum += float(rewards.sum())
        self.step_count += int(rewards.size)
        return self.value


def differential_rate_loss(
    reward_rate, scaled_rewards, bootstrap, selected_next, current_c
):
    target = (
        scaled_rewards
        + bootstrap * selected_next.mean(dim=0)
        - current_c.mean(dim=0)
    ).detach()
    return (reward_rate() - target).square().mean(), target


def centered_option_rewards(scaled_rewards, reward_rate, average_reward):
    if not average_reward:
        return scaled_rewards
    return scaled_rewards - reward_rate().detach()


class PersistentGoals:
    def __init__(self, num_envs, latent_dim):
        self.latents = np.zeros((num_envs, latent_dim), dtype=np.float32)
        self.has_goal = np.zeros(num_envs, dtype=bool)
        self.age = np.zeros(num_envs, dtype=np.int64)

    def apply(self, proposed, propose_mask):
        proposed = np.asarray(proposed, dtype=np.float32)
        self.latents[propose_mask] = proposed[propose_mask]
        self.has_goal[propose_mask] = True
        self.age[propose_mask] = 0
        self.age[~propose_mask & self.has_goal] += 1
        return np.array(self.latents, copy=True)

    def reset(self, done):
        self.has_goal[done] = False
        self.age[done] = 0


def option_decision(agent, current_observations, persistent, goal_probe, args):
    """Stop on a pessimistic tie; proposals are unrestricted latent displacements."""
    with torch.no_grad():
        z = agent.world.encode(current_observations)
        goal_z = torch.as_tensor(
            persistent.latents,
            device=current_observations.device,
            dtype=current_observations.dtype,
        )
        continuing = continue_decision(
            agent.h(z), agent.c(z, goal_z), args.pessimism_coef
        ).cpu().numpy()
        propose = ~persistent.has_goal | ~continuing
        proposed = z + agent.goal(z) + goal_probe
    goal_used = persistent.apply(proposed.cpu().numpy(), propose)
    return goal_used, propose


def exact_scaled_control_cost(actions, control_cost_weight, reward_scale):
    return reward_scale * control_cost_weight * actions.square().sum(dim=-1)


def raw_probe_action(raw_action, requested_probe):
    nominal = torch.tanh(raw_action)
    executed = torch.tanh(raw_action + requested_probe)
    return executed, nominal, executed - nominal


def probe_covariance(probes):
    centered = probes - probes.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(1, centered.shape[0])


def directional_policy_surrogate(raw_actions, residuals, probes):
    """One-policy zeroth-order update, whitened by the factual probe covariance."""
    centered = probes - probes.mean(dim=0, keepdim=True)
    covariance = probe_covariance(probes)
    inverse = torch.linalg.pinv(covariance.detach(), hermitian=True)
    directions = centered @ inverse
    objective = (
        raw_actions * (residuals.unsqueeze(-1) * directions).detach()
    ).sum(dim=-1)
    return -objective.mean(), covariance, directions


def covariance_diagnostics(samples):
    covariance = probe_covariance(samples)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    finite = eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).eps)
    return covariance, eigenvalues, finite.max() / finite.min()


def safe_correlation(x, y):
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denominator = x.square().mean().sqrt() * y.square().mean().sqrt()
    if float(denominator) == 0.0:
        return x.new_zeros(())
    return (x * y).mean() / denominator


def factual_probe_policy_loss(
    agent,
    current_observations,
    factual_next_observations_tensor,
    goal_latents,
    stored_raw_actions,
    executed_actions,
    requested_raw_probes,
    actual_bounded_deltas,
    centered_scaled_rewards,
    control_cost_weight,
    reward_scale,
    reward_utility_coef,
):
    """Policy credit comes only from the encoded factual successor and probes."""
    with torch.no_grad():
        z = agent.world.encode(current_observations)
        factual_next_z = agent.world.encode(factual_next_observations_tensor)
        before_mse = (z - goal_latents).square().mean(dim=-1)
        after_mse = (factual_next_z - goal_latents).square().mean(dim=-1)
        progress = before_mse - after_mse
        control_cost = exact_scaled_control_cost(
            executed_actions, control_cost_weight, reward_scale
        )
        reward_component = reward_utility_coef * centered_scaled_rewards
        utility = progress - control_cost + reward_component

    residual = utility.detach()
    recomputed_raw = agent.policy.raw_action(z.detach(), goal_latents.detach())
    policy_loss, covariance, directions = directional_policy_surrogate(
        recomputed_raw, residual, requested_raw_probes
    )
    eigenvalues = torch.linalg.eigvalsh(covariance.detach())
    finite_eigenvalues = eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).eps)
    condition = finite_eigenvalues.max() / finite_eigenvalues.min()
    bounded_covariance, bounded_eigenvalues, bounded_condition = (
        covariance_diagnostics(actual_bounded_deltas.detach())
    )
    diagonal = covariance.diagonal()
    off_diagonal = covariance - torch.diag(diagonal)
    utility_centered = utility - utility.mean()
    probe_utility_correlation = (
        (utility_centered.unsqueeze(-1) * requested_raw_probes).mean(dim=0)
        / (
            utility_centered.square().mean().sqrt()
            * requested_raw_probes.square().mean(dim=0).sqrt().clamp_min(1e-8)
        ).clamp_min(1e-8)
    )
    return policy_loss, recomputed_raw, {
        "utility": utility.mean().detach(),
        "residual_mean": residual.mean(),
        "residual_std": residual.std(unbiased=False),
        "factual_goal_progress": progress.mean(),
        "actual_goal_mse": after_mse.mean(),
        "control_cost": control_cost.mean(),
        "reward_component": reward_component.mean(),
        "reward_utility_correlation": safe_correlation(
            centered_scaled_rewards, utility
        ).detach(),
        "probe_utility_correlation_norm": probe_utility_correlation.norm().detach(),
        "probe_utility_correlation_max": probe_utility_correlation.abs().max().detach(),
        "probe_covariance_min_eigenvalue": eigenvalues.min(),
        "probe_covariance_max_eigenvalue": eigenvalues.max(),
        "probe_covariance_condition": condition,
        "probe_covariance_offdiag_rms": off_diagonal.square().mean().sqrt(),
        "probe_norm": requested_raw_probes.norm(dim=-1).mean(),
        "bounded_delta_norm": actual_bounded_deltas.norm(dim=-1).mean(),
        "bounded_delta_covariance_rank": torch.linalg.matrix_rank(
            bounded_covariance
        ).to(torch.float32),
        "bounded_delta_covariance_min_eigenvalue": bounded_eigenvalues.min(),
        "bounded_delta_covariance_condition": bounded_condition,
        "direction_norm": directions.norm(dim=-1).mean().detach(),
        "action_saturation": (executed_actions.abs() > 0.95).float().mean(),
        "raw_action_recompute_mse": (
            recomputed_raw.detach() - stored_raw_actions
        ).square().mean(),
    }


def goal_proposal_loss(
    agent,
    current_observations,
    stored_goal_probe,
    proposal_mask,
    pessimism_coef,
):
    if not proposal_mask.any():
        return None, {}
    current = current_observations[proposal_mask]
    probe = stored_goal_probe[proposal_mask]
    with torch.no_grad():
        z = agent.world.encode(current)
    proposed_z = z.detach() + agent.goal(z.detach()) + probe
    with frozen_parameters(agent.h, agent.c):
        score = pessimistic_surplus(
            agent.h(z.detach()), agent.c(z.detach(), proposed_z), pessimism_coef
        )
    return -score.mean(), {
        "proposal_surplus": score.mean().detach(),
        "latent_displacement_norm": (proposed_z - z).norm(dim=-1).mean().detach(),
        "latent_goal_norm": proposed_z.norm(dim=-1).mean().detach(),
    }


def ensemble_mse(predictions, targets, item_mask, bootstrap_probability):
    mask = item_mask.unsqueeze(0).expand_as(predictions)
    bootstrap_mask = torch.rand_like(predictions) < bootstrap_probability
    selected = mask & bootstrap_mask
    error = (predictions - targets).square()
    return (error * selected).sum() / selected.sum().clamp_min(1)


def update_proposal_h(
    h,
    z,
    targets,
    proposal_mask,
    optimizer,
    bootstrap_probability,
    max_grad_norm,
):
    """No proposal means no H optimizer step, including no AdamW decay."""
    if not proposal_mask.any():
        zero = z.new_zeros(())
        return zero, zero
    predictions = h(z)
    loss = ensemble_mse(
        predictions, targets, proposal_mask, bootstrap_probability
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(h.parameters(), max_grad_norm)
    optimizer.step()
    return loss.detach(), gradient_norm.detach()


def goal_recompute_mse(
    agent, current_observations, excitation, stored_goals, proposal_mask
):
    if not proposal_mask.any():
        return torch.zeros((), device=current_observations.device)
    with torch.no_grad():
        z = agent.world.encode(current_observations[proposal_mask])
        recomputed = z + agent.goal(z) + excitation[proposal_mask]
    return (recomputed - stored_goals[proposal_mask]).square().mean()


class SequenceReplay:
    def __init__(self, capacity, history_length, obs_dim, action_dim):
        self.capacity = capacity
        self.observations = np.empty(
            (capacity, history_length + 1, obs_dim), dtype=np.float32
        )
        self.actions = np.empty(
            (capacity, history_length, action_dim), dtype=np.float32
        )
        self.valid = np.empty((capacity, history_length + 1), dtype=bool)
        self.pointer = 0
        self.size = 0

    def add(self, observations, actions, valid):
        slots = (self.pointer + np.arange(len(observations))) % self.capacity
        self.observations[slots] = observations
        self.actions[slots] = actions
        self.valid[slots] = valid
        self.pointer = (self.pointer + len(observations)) % self.capacity
        self.size = min(self.capacity, self.size + len(observations))

    def sample(self, count, recent_size, rng, device):
        available = min(self.size, recent_size)
        offsets = rng.integers(0, available, size=count)
        indices = (self.pointer - 1 - offsets) % self.capacity
        return (
            torch.as_tensor(self.observations[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.valid[indices], device=device),
        )


def world_parameter_ids(agent):
    return {id(parameter) for parameter in agent.world.parameters()}


def optimizer_parameter_groups(agent):
    return {
        "world": set(agent.world.parameters()),
        "h": set(agent.h.parameters()),
        "c": set(agent.c.parameters()),
        "goal": set(agent.goal.parameters()),
        "policy": set(agent.policy.parameters()),
        "reward_rate": set(agent.reward_rate.parameters()),
    }


def world_action_sensitivity(world, observations, actions, source_valid=None):
    actions = actions.detach().clone().requires_grad_(True)
    prediction = world.predict_final(observations, actions, source_valid)
    gradient = torch.autograd.grad(prediction.square().mean(), actions)[0]
    return gradient[:, -1].norm(dim=-1).mean()


def adaln_gate_magnitude(world):
    gates = []
    for block in world.predictor:
        bias = block.adaLN_modulation[-1].bias.view(6, -1)
        gates.extend([bias[2].abs().mean(), bias[5].abs().mean()])
    return torch.stack(gates).mean()


def procrustes_alignment(raw_new, global_old):
    u, _, vh = torch.linalg.svd(raw_new.T @ global_old, full_matrices=False)
    return u @ vh


def update_world(agent, sigreg, replay, optimizer, args, rng, device):
    observations, actions, valid = replay.sample(
        args.minibatch_size, args.recent_replay_size, rng, device
    )
    with torch.no_grad():
        old_global = agent.world.encode(observations[valid])
    total, prediction, regularization = masked_world_objective(
        agent.world, sigreg, observations, actions, valid, args.sigreg_coef
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        agent.world.parameters(), args.max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        raw_new = agent.world.encode_raw(observations[valid])
        alignment = procrustes_alignment(raw_new, old_global)
        residual = (raw_new @ alignment - old_global).square().mean().sqrt()
        agent.world.alignment.copy_(alignment)
    return {
        "loss": total.detach(),
        "prediction_mse": prediction.detach(),
        "sigreg": regularization.detach(),
        "gradient_norm": gradient_norm.detach(),
        "chart_residual": residual.detach(),
    }


def boundary_next_option_values(
    next_h,
    next_c_same_goal,
    live_next_h,
    factual_next_h,
    terminated,
    truncated,
    average_reward,
    pessimism_coef,
):
    selected, continuing = select_next_branch(
        next_h, next_c_same_goal, pessimism_coef
    )
    done = terminated | truncated
    if average_reward:
        # Exact regenerative stream: the discarded option is replaced by a
        # fresh proposal at the autoreset observation, including true terminals.
        selected = torch.where(done.unsqueeze(0), live_next_h, selected)
        bootstrap = torch.ones_like(terminated, dtype=next_h.dtype)
    else:
        # Episodic discounted ablation: true terminals have no bootstrap;
        # TimeLimit uses H at the factual final observation as an untruncated
        # surrogate, with the old option considered stopped.
        selected = torch.where(truncated.unsqueeze(0), factual_next_h, selected)
        bootstrap = (~terminated).to(next_h.dtype)
    return selected, bootstrap, continuing


def control_cost_weight(envs):
    weights = [
        float(getattr(env.unwrapped, "_ctrl_cost_weight", 0.0))
        for env in envs.envs
    ]
    if not np.allclose(weights, weights[0]):
        raise ValueError("vector environments must share one control-cost weight")
    return weights[0]


def add_episode_metrics(writer, infos, global_step):
    if "final_info" not in infos:
        return
    mask = infos.get(
        "_final_info",
        np.asarray([x is not None for x in infos["final_info"]], dtype=bool),
    )
    for index in np.flatnonzero(mask):
        info = infos["final_info"][index]
        if info is not None and "episode" in info:
            writer.add_scalar(
                "charts/episodic_return", float(info["episode"]["r"]), global_step
            )
            writer.add_scalar(
                "charts/episodic_length", float(info["episode"]["l"]), global_step
            )


def validate_runtime_args(args):
    if args.history_length < 1:
        raise ValueError("history_length must be at least one")
    if args.replay_capacity <= 0 or args.recent_replay_size <= 0:
        raise ValueError("replay sizes must be positive")
    if args.minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if args.recent_replay_size > args.replay_capacity:
        raise ValueError("recent_replay_size cannot exceed replay_capacity")
    if args.average_reward and args.option_discount != 1.0:
        raise ValueError("average-reward mode requires option_discount=1")
    if args.action_probe_amplitude <= 0:
        raise ValueError("action_probe_amplitude must be positive")
    if args.goal_probe_amplitude <= 0:
        raise ValueError("goal_probe_amplitude must be positive")


def main():
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    validate_runtime_args(args)
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("factual-probe option training requires CUDA")
    discount = (
        args.option_discount
        if args.average_reward
        else args.discounted_option_discount
    )

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, index, args.capture_video, run_name)
            for index in range(args.num_envs)
        ]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    ctrl_weight = control_cost_weight(envs)
    agent = Agent(obs_dim, action_dim, args).to(device)
    sigreg = SIGReg().to(device)
    groups = optimizer_parameter_groups(agent)
    world_optimizer = optim.AdamW(
        list(agent.world.parameters()), lr=args.world_lr, weight_decay=args.weight_decay
    )
    h_optimizer = optim.AdamW(
        list(agent.h.parameters()), lr=args.h_lr, weight_decay=args.weight_decay
    )
    c_optimizer = optim.AdamW(
        list(agent.c.parameters()), lr=args.c_lr, weight_decay=args.weight_decay
    )
    goal_optimizer = optim.AdamW(
        list(agent.goal.parameters()), lr=args.goal_lr, weight_decay=args.weight_decay
    )
    policy_optimizer = optim.AdamW(
        list(agent.policy.parameters()), lr=args.policy_lr, weight_decay=args.weight_decay
    )
    reward_rate_optimizer = optim.Adam(
        list(agent.reward_rate.parameters()), lr=args.reward_rate_lr
    )

    def collection_policy(current_observations, goal_latents):
        z = agent.world.encode(current_observations)
        return agent.policy.raw_action(z, goal_latents)

    if args.compile:
        collection_policy = torch.compile(
            collection_policy, mode=args.compile_mode, dynamic=False
        )

    replay = SequenceReplay(
        args.replay_capacity,
        args.history_length,
        obs_dim,
        action_dim,
    )
    observations, _ = envs.reset(seed=args.seed)
    obs_context, past_actions, context_valid = initialize_context(
        observations, args.history_length, action_dim
    )
    persistent = PersistentGoals(args.num_envs, args.latent_dim)
    cumulative_rate = CumulativeRewardRate()
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        controlled = global_step >= args.warmup_steps
        shape = (args.num_steps, args.num_envs)
        rollout_context = np.empty(
            (*shape, args.history_length, obs_dim), dtype=np.float32
        )
        rollout_factual_next = np.empty((*shape, obs_dim), dtype=np.float32)
        rollout_live_next = np.empty((*shape, obs_dim), dtype=np.float32)
        rollout_actions = np.empty((*shape, action_dim), dtype=np.float32)
        rollout_raw_actions = np.zeros_like(rollout_actions)
        rollout_raw_probes = np.zeros_like(rollout_actions)
        rollout_bounded_deltas = np.zeros_like(rollout_actions)
        rollout_goal_probes = np.zeros(
            (*shape, args.latent_dim), dtype=np.float32
        )
        rollout_goals = np.zeros((*shape, args.latent_dim), dtype=np.float32)
        rollout_proposal = np.zeros(shape, dtype=bool)
        rollout_rewards = np.empty(shape, dtype=np.float32)
        rollout_terminated = np.empty(shape, dtype=bool)
        rollout_truncated = np.empty(shape, dtype=bool)
        raw_forward_total = 0.0
        raw_control_total = 0.0

        for step in range(args.num_steps):
            rollout_context[step] = obs_context
            if controlled:
                current = torch.as_tensor(
                    observations, device=device, dtype=torch.float32
                )
                goal_probe = deterministic_goal_probe(
                    global_step // args.num_envs,
                    args.num_envs,
                    args.latent_dim,
                    args.goal_probe_amplitude,
                    device,
                )
                goal_used, proposal = option_decision(
                    agent, current, persistent, goal_probe, args
                )
                requested_probe = deterministic_action_probe(
                    global_step // args.num_envs,
                    args.num_envs,
                    action_dim,
                    args.action_probe_amplitude,
                    device,
                )
                with torch.no_grad():
                    raw_action = collection_policy(
                        current,
                        torch.as_tensor(goal_used, device=device),
                    )
                    chosen, _, bounded_delta = raw_probe_action(
                        raw_action, requested_probe
                    )
                actions = chosen.cpu().numpy()
                rollout_goals[step] = goal_used
                rollout_proposal[step] = proposal
                rollout_goal_probes[step, proposal] = (
                    goal_probe[proposal].cpu().numpy()
                )
                rollout_raw_actions[step] = raw_action.cpu().numpy()
                rollout_raw_probes[step] = requested_probe.cpu().numpy()
                rollout_bounded_deltas[step] = bounded_delta.cpu().numpy()
            else:
                actions = np.stack(
                    [
                        envs.single_action_space.sample()
                        for _ in range(args.num_envs)
                    ]
                ).astype(np.float32)

            next_observations, rewards, terminations, truncations, infos = envs.step(
                actions
            )
            factual_next = factual_next_observations(next_observations, infos)
            sequence_obs, sequence_actions, sequence_valid = transition_sequence(
                obs_context,
                past_actions,
                context_valid,
                actions,
                factual_next,
            )
            done = np.logical_or(terminations, truncations)
            next_context, next_past, next_valid = next_live_context(
                sequence_obs,
                sequence_actions,
                sequence_valid,
                next_observations,
                done,
            )
            replay.add(sequence_obs, sequence_actions, sequence_valid)
            rollout_factual_next[step] = factual_next
            rollout_live_next[step] = next_observations
            rollout_actions[step] = actions
            rollout_rewards[step] = rewards
            rollout_terminated[step] = terminations
            rollout_truncated[step] = truncations

            if "reward_run" in infos:
                raw_forward_total += float(np.asarray(infos["reward_run"]).mean())
            elif "reward_forward" in infos:
                raw_forward_total += float(np.asarray(infos["reward_forward"]).mean())
            if "reward_ctrl" in infos:
                raw_control_total += float(np.asarray(infos["reward_ctrl"]).mean())
            global_step += args.num_envs
            observations = next_observations
            obs_context, past_actions, context_valid = (
                next_context,
                next_past,
                next_valid,
            )
            persistent.reset(done)
            add_episode_metrics(writer, infos, global_step)

        flat_count = args.batch_size
        context_tensor = torch.as_tensor(
            rollout_context.reshape(flat_count, args.history_length, obs_dim),
            device=device,
        )
        factual_next_tensor = torch.as_tensor(
            rollout_factual_next.reshape(flat_count, obs_dim), device=device
        )
        live_next_tensor = torch.as_tensor(
            rollout_live_next.reshape(flat_count, obs_dim), device=device
        )
        goal_tensor = torch.as_tensor(
            rollout_goals.reshape(flat_count, args.latent_dim), device=device
        )
        raw_action_tensor = torch.as_tensor(
            rollout_raw_actions.reshape(flat_count, action_dim), device=device
        )
        raw_probe_tensor = torch.as_tensor(
            rollout_raw_probes.reshape(flat_count, action_dim), device=device
        )
        bounded_delta_tensor = torch.as_tensor(
            rollout_bounded_deltas.reshape(flat_count, action_dim), device=device
        )
        goal_probe_tensor = torch.as_tensor(
            rollout_goal_probes.reshape(flat_count, args.latent_dim), device=device
        )
        proposal_tensor = torch.as_tensor(
            rollout_proposal.reshape(flat_count), device=device
        )

        if controlled:
            scaled_rewards = args.reward_scale * torch.as_tensor(
                rollout_rewards, device=device
            )
            cumulative_rho = cumulative_rate.update(scaled_rewards.cpu().numpy())
            training_rho = agent.reward_rate().detach()
            adjusted = centered_option_rewards(
                scaled_rewards, agent.reward_rate, args.average_reward
            )

            # 1. Factual low-level credit before any surface or chart moves.
            executed_action_tensor = torch.as_tensor(
                rollout_actions.reshape(flat_count, action_dim), device=device
            )
            policy_loss, _, policy_metrics = (
                factual_probe_policy_loss(
                    agent,
                    context_tensor[:, -1],
                    factual_next_tensor,
                    goal_tensor,
                    raw_action_tensor,
                    executed_action_tensor,
                    raw_probe_tensor,
                    bounded_delta_tensor,
                    adjusted.reshape(flat_count),
                    ctrl_weight,
                    args.reward_scale,
                    args.utility_reward_coef,
                )
            )
            policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_grad = nn.utils.clip_grad_norm_(
                agent.policy.parameters(), args.max_grad_norm
            )
            policy_optimizer.step()
            for key, value in policy_metrics.items():
                writer.add_scalar(f"policy/{key}", float(value), global_step)
            writer.add_scalar("policy/loss", float(policy_loss.detach()), global_step)
            writer.add_scalar("policy/gradient_norm", float(policy_grad), global_step)

            # 2. G sees only factual proposal states and the exact stored probes.
            proposal_loss, proposal_metrics = goal_proposal_loss(
                agent,
                context_tensor[:, -1],
                goal_probe_tensor,
                proposal_tensor,
                args.pessimism_coef,
            )
            goal_exact_mse = goal_recompute_mse(
                agent,
                context_tensor[:, -1],
                goal_probe_tensor,
                goal_tensor,
                proposal_tensor,
            )
            if proposal_loss is not None:
                goal_optimizer.zero_grad(set_to_none=True)
                proposal_loss.backward()
                goal_grad = nn.utils.clip_grad_norm_(
                    agent.goal.parameters(), args.max_grad_norm
                )
                goal_optimizer.step()
                writer.add_scalar("goal/gradient_norm", float(goal_grad), global_step)
                for key, value in proposal_metrics.items():
                    writer.add_scalar(f"goal/{key}", float(value), global_step)
            writer.add_scalar("goal/exact_recompute_mse", float(goal_exact_mse), global_step)

            # 3. Factual H/C targets. No replay goals and no relabeling.
            terminated = torch.as_tensor(
                rollout_terminated.reshape(flat_count), device=device
            )
            truncated = torch.as_tensor(
                rollout_truncated.reshape(flat_count), device=device
            )
            with torch.no_grad():
                z = agent.world.encode(context_tensor[:, -1])
                goal_z = goal_tensor
                factual_next_z = agent.world.encode(factual_next_tensor)
                live_next_z = agent.world.encode(live_next_tensor)
                current_h = agent.h(z)
                current_c = agent.c(z, goal_z)
                next_h = agent.h(factual_next_z)
                next_c = agent.c(factual_next_z, goal_z)
                live_h = agent.h(live_next_z)
                selected_next, bootstrap, next_continue = boundary_next_option_values(
                    next_h,
                    next_c,
                    live_h,
                    next_h,
                    terminated,
                    truncated,
                    args.average_reward,
                    args.pessimism_coef,
                )
                one_step_c = (
                    adjusted.reshape(flat_count).unsqueeze(0)
                    + discount * bootstrap.unsqueeze(0) * selected_next
                )
                manager_returns = manager_lambda_returns(
                    adjusted,
                    selected_next.T.reshape(
                        args.num_steps, args.num_envs, args.option_heads
                    ),
                    bootstrap.reshape(args.num_steps, args.num_envs),
                    (~(terminated | truncated)).reshape(
                        args.num_steps, args.num_envs
                    ).to(adjusted.dtype),
                    discount,
                    args.manager_lambda,
                ).reshape(flat_count, args.option_heads).T
                c_targets = torch.where(
                    proposal_tensor.unsqueeze(0), manager_returns, one_step_c
                )

            if args.average_reward:
                rate_loss, _ = differential_rate_loss(
                    agent.reward_rate,
                    scaled_rewards.reshape(flat_count),
                    bootstrap,
                    selected_next,
                    current_c,
                )
                reward_rate_optimizer.zero_grad(set_to_none=True)
                rate_loss.backward()
                rate_grad = nn.utils.clip_grad_norm_(
                    agent.reward_rate.parameters(), args.max_grad_norm
                )
                reward_rate_optimizer.step()
            else:
                rate_loss = scaled_rewards.new_zeros(())
                rate_grad = scaled_rewards.new_zeros(())

            c_predictions = agent.c(z.detach(), goal_z.detach())
            c_loss = ensemble_mse(
                c_predictions,
                c_targets,
                torch.ones_like(proposal_tensor),
                args.bootstrap_probability,
            )
            c_optimizer.zero_grad(set_to_none=True)
            c_loss.backward()
            c_grad = nn.utils.clip_grad_norm_(agent.c.parameters(), args.max_grad_norm)
            c_optimizer.step()

            h_loss, h_grad = update_proposal_h(
                agent.h,
                z.detach(),
                manager_returns,
                proposal_tensor,
                h_optimizer,
                args.bootstrap_probability,
                args.max_grad_norm,
            )
            surplus = pessimistic_surplus(
                current_h, current_c, args.pessimism_coef
            )
            writer.add_scalar(
                "option/rho_learned", float(training_rho), global_step
            )
            writer.add_scalar(
                "option/rho_cumulative_metric", cumulative_rho, global_step
            )
            writer.add_scalar(
                "option/rho_loss", float(rate_loss.detach()), global_step
            )
            writer.add_scalar(
                "option/rho_gradient_norm", float(rate_grad), global_step
            )
            writer.add_scalar("option/h_loss", float(h_loss), global_step)
            writer.add_scalar("option/c_loss", float(c_loss), global_step)
            writer.add_scalar(
                "option/h_ensemble_spread",
                float(current_h.std(dim=0, unbiased=False).mean()),
                global_step,
            )
            writer.add_scalar(
                "option/c_ensemble_spread",
                float(current_c.std(dim=0, unbiased=False).mean()),
                global_step,
            )
            writer.add_scalar("option/h_gradient_norm", float(h_grad), global_step)
            writer.add_scalar("option/c_gradient_norm", float(c_grad), global_step)
            writer.add_scalar(
                "option/switch_rate", float(proposal_tensor.float().mean()), global_step
            )
            writer.add_scalar(
                "option/surplus_active",
                float(surplus[~proposal_tensor].mean()) if (~proposal_tensor).any() else 0.0,
                global_step,
            )
            writer.add_scalar(
                "option/surplus_proposal",
                float(surplus[proposal_tensor].mean()) if proposal_tensor.any() else 0.0,
                global_step,
            )
            writer.add_scalar(
                "option/next_continue_rate", float(next_continue.float().mean()), global_step
            )
            writer.add_scalar(
                "option/realized_adjusted_reward", float(adjusted.mean()), global_step
            )
            writer.add_scalar(
                "goal/latent_norm", float(goal_tensor.norm(dim=-1).mean()), global_step
            )
            writer.add_scalar(
                "goal/age_mean",
                float(persistent.age[persistent.has_goal].mean())
                if persistent.has_goal.any()
                else 0.0,
                global_step,
            )

        # 4. The representation moves last; Procrustes keeps persistent latent
        # goals and all online consumers in the preceding global chart.
        if replay.size >= args.minibatch_size:
            totals = {}
            for _ in range(args.model_updates_per_iteration):
                metrics = update_world(
                    agent,
                    sigreg,
                    replay,
                    world_optimizer,
                    args,
                    rng,
                    device,
                )
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
            for key, value in totals.items():
                writer.add_scalar(
                    f"world/{key}",
                    value / args.model_updates_per_iteration,
                    global_step,
                )
            sampled_obs, sampled_actions, sampled_valid = replay.sample(
                min(args.minibatch_size, replay.size),
                args.recent_replay_size,
                rng,
                device,
            )
            sensitivity = world_action_sensitivity(
                agent.world,
                sampled_obs[:, :-1],
                sampled_actions,
                sampled_valid[:, :-1],
            )
            writer.add_scalar(
                "world/action_sensitivity", float(sensitivity), global_step
            )
            writer.add_scalar(
                "world/adaln_gate_magnitude",
                float(adaln_gate_magnitude(agent.world).detach()),
                global_step,
            )

        writer.add_scalar(
            "reward/raw_total", float(rollout_rewards.mean()), global_step
        )
        writer.add_scalar("diagnostics/probes_active", float(controlled), global_step)
        writer.add_scalar(
            "reward/raw_forward", raw_forward_total / args.num_steps, global_step
        )
        writer.add_scalar(
            "reward/raw_control", raw_control_total / args.num_steps, global_step
        )
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        if iteration == 1 or iteration % 10 == 0:
            print(
                f"iteration={iteration} step={global_step} controlled={controlled} "
                f"SPS={int(global_step / (time.time() - start_time))}"
            )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

# Embedding Optimization: Locally Identified Jacobian Options v2
#
# A LeJEPA causal world model factorizes factual dynamics into a nominal
# successor and an explicit local action Jacobian. Persistent deterministic
# action probes identify that Jacobian. A deterministic actor is trained from
# factual encoded successors plus only the identified local action correction.
# Free latent goals persist until pessimistic C(state, goal)-H(state) says to
# refresh them. There are no policy likelihoods, action-values, target models,
# moving averages, relabeled goals, planners, or search at action time.
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
    reward_rate_lr: float = 1e-3
    weight_decay: float = 1e-4
    sigreg_coef: float = 0.09
    reward_scale: float = 0.1
    average_reward: bool = True
    option_discount: float = 1.0
    discounted_option_discount: float = 0.99
    manager_lambda: float = 0.95
    pessimism_coef: float = 1.0
    action_probe_amplitude: float = 0.12
    action_probe_period: int = 257
    goal_probe_amplitude: float = 0.5
    goal_probe_period: int = 379
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
    nominal_actions = np.zeros(
        (len(observations), history_length - 1, action_dim), dtype=np.float32
    )
    action_deltas = np.zeros_like(nominal_actions)
    valid = np.zeros((len(observations), history_length), dtype=bool)
    valid[:, -1] = True
    return obs_context, nominal_actions, action_deltas, valid


def transition_sequence(
    obs_context,
    past_nominal_actions,
    past_action_deltas,
    valid,
    nominal_action,
    action_delta,
    factual_next,
):
    nominal_actions = np.concatenate(
        [past_nominal_actions, nominal_action[:, None]], axis=1
    )
    action_deltas = np.concatenate(
        [past_action_deltas, action_delta[:, None]], axis=1
    )
    observations = np.concatenate([obs_context, factual_next[:, None]], axis=1)
    observation_valid = np.concatenate(
        [valid, np.ones((len(valid), 1), dtype=bool)], axis=1
    )
    return observations, nominal_actions, action_deltas, observation_valid


def next_live_context(
    observations,
    nominal_actions,
    action_deltas,
    observation_valid,
    autoreset_observations,
    done,
):
    obs_context = np.array(observations[:, 1:], copy=True)
    past_nominal = np.array(nominal_actions[:, 1:], copy=True)
    past_delta = np.array(action_deltas[:, 1:], copy=True)
    valid = np.array(observation_valid[:, 1:], copy=True)
    if np.any(done):
        reset = initialize_context(
            np.asarray(autoreset_observations)[done],
            obs_context.shape[1],
            nominal_actions.shape[-1],
        )
        obs_context[done], past_nominal[done], past_delta[done], valid[done] = reset
    return obs_context, past_nominal, past_delta, valid


def sylvester_hadamard(order, device=None):
    if order < 2 or order & (order - 1):
        raise ValueError("probe environment count must be a power of two")
    matrix = torch.ones((1, 1), device=device)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix


def deterministic_probe(step, count, dimension, amplitude, period, device=None):
    """Balanced counter-based probes with full action rank at every step."""
    del period  # Retained in the CLI for comparable run metadata.
    hadamard = sylvester_hadamard(count, device=device)
    rows = (
        torch.arange(count, device=device) * (2 * (step % (count // 2)) + 1)
        + step
    ) % count
    columns = (
        torch.arange(dimension, device=device) + step * max(1, dimension)
    ) % (count - 1) + 1
    signs = torch.where(
        ((torch.arange(dimension, device=device) + 3 * step) % 2) == 0,
        1.0,
        -1.0,
    )
    return amplitude * hadamard[rows][:, columns] * signs


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
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x,
        nominal_action_condition,
        causal_mask,
        key_padding_mask=None,
    ):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(nominal_action_condition).chunk(6, dim=-1)
        )
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
        return x


class LocalJacobianWorldModel(nn.Module):
    """Base sees nominal actions only; factual prediction adds J times probe delta."""

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
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.observation_encoder = mlp(obs_dim, hidden_dim, latent_dim)
        self.encoder_projector = mlp(latent_dim, 2 * latent_dim, latent_dim)
        self.action_encoder = mlp(action_dim, hidden_dim, latent_dim, out_std=0.1)
        self.position = nn.Parameter(torch.zeros(1, history_length, latent_dim))
        nn.init.normal_(self.position, std=0.01)
        self.predictor = nn.ModuleList(
            [
                ActionAdaLNBlock(latent_dim, transformer_heads)
                for _ in range(transformer_layers)
            ]
        )
        self.predictor_norm = nn.LayerNorm(latent_dim)
        self.base_head = mlp(latent_dim, 2 * latent_dim, latent_dim, out_std=0.01)
        self.jacobian_head = mlp(
            latent_dim,
            2 * latent_dim,
            latent_dim * action_dim,
            out_std=0.01,
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

    def hidden_from_latents(
        self, source_latents, nominal_actions, source_valid=None
    ):
        if source_latents.shape[:2] != nominal_actions.shape[:2]:
            raise ValueError(
                "each source observation needs one aligned nominal outgoing action"
            )
        length = source_latents.shape[1]
        x = source_latents + self.position[:, :length]
        condition = self.action_encoder(nominal_actions)
        mask = self.causal_mask(length, x.device)
        padding_mask = None if source_valid is None else ~source_valid
        for block in self.predictor:
            x = block(x, condition, mask, padding_mask)
        return self.predictor_norm(x)

    def predict_components_from_latents(
        self, source_latents, nominal_actions, source_valid=None
    ):
        hidden = self.hidden_from_latents(
            source_latents, nominal_actions, source_valid
        )
        base = self.base_head(hidden)
        jacobian = self.jacobian_head(hidden).reshape(
            *hidden.shape[:-1], self.latent_dim, self.action_dim
        )
        return base, jacobian

    def predict_components(
        self, source_observations, nominal_actions, source_valid=None
    ):
        return self.predict_components_from_latents(
            self.encode(source_observations), nominal_actions, source_valid
        )

    def factual_prediction_from_latents(
        self,
        source_latents,
        nominal_actions,
        action_deltas,
        source_valid=None,
    ):
        base, jacobian = self.predict_components_from_latents(
            source_latents, nominal_actions, source_valid
        )
        return base + torch.einsum("...za,...a->...z", jacobian, action_deltas)

    def factual_prediction(
        self,
        source_observations,
        nominal_actions,
        action_deltas,
        source_valid=None,
    ):
        return self.factual_prediction_from_latents(
            self.encode(source_observations),
            nominal_actions,
            action_deltas,
            source_valid,
        )

    def final_jacobian(
        self, observation_context, nominal_actions, source_valid=None
    ):
        return self.predict_components(
            observation_context, nominal_actions, source_valid
        )[1][:, -1]


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
            [
                mlp(3 * latent_dim, hidden_dim, 1, zero_output=True)
                for _ in range(heads)
            ]
        )

    def forward(self, z, goal_z):
        features = torch.cat([z, goal_z, goal_z - z], dim=-1)
        return torch.stack([head(features).squeeze(-1) for head in self.heads], dim=0)


class LearnedRewardRate(nn.Module):
    """Bellman-identified average reward; this is an optimized scalar, not an EMA."""

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, args):
        super().__init__()
        self.world = LocalJacobianWorldModel(
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
        self.reward_rate = LearnedRewardRate()


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
    world,
    sigreg,
    observations,
    nominal_actions,
    action_deltas,
    observation_valid,
    sigreg_coef,
):
    """Token j is (o_j, nominal_j, delta_j) -> attached E(o_{j+1})."""
    encoded = world.encode(observations)
    source_valid = observation_valid[:, :-1]
    source_encoded = torch.where(
        source_valid.unsqueeze(-1),
        encoded[:, :-1],
        torch.zeros_like(encoded[:, :-1]),
    )
    predicted = world.factual_prediction_from_latents(
        source_encoded,
        nominal_actions,
        action_deltas,
        source_valid,
    )
    valid_targets = source_valid & observation_valid[:, 1:]
    per_token = (predicted - encoded[:, 1:]).square().mean(dim=-1)
    prediction_loss = (
        per_token * valid_targets
    ).sum() / valid_targets.sum().clamp_min(1)
    sigreg_loss = sigreg(encoded[observation_valid])
    return prediction_loss + sigreg_coef * sigreg_loss, prediction_loss, sigreg_loss


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
    continuing = continue_decision(h_values, c_values, coefficient)
    return torch.where(continuing.unsqueeze(0), c_values, h_values), continuing


def manager_lambda_returns(
    adjusted_rewards,
    next_manager_values,
    bootstrap,
    trace,
    discount,
    manager_lambda,
):
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
    """Exact all-controlled-transitions rate retained only as telemetry."""

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


def reward_rate_loss(
    scaled_rewards,
    current_c,
    selected_next,
    bootstrap,
    reward_rate,
):
    """Fit rho to detached one-step differential Bellman residuals."""
    residual = (
        scaled_rewards.detach()
        + bootstrap.detach() * selected_next.detach().mean(dim=0)
        - current_c.detach().mean(dim=0)
        - reward_rate.value
    )
    return residual.square().mean()


class PersistentLatentGoals:
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
    """Stop on a pessimistic tie and store the exact free latent proposal."""
    with torch.no_grad():
        z = agent.world.encode(current_observations)
        stored_goal = torch.as_tensor(
            persistent.latents,
            device=current_observations.device,
            dtype=current_observations.dtype,
        )
        continuing = continue_decision(
            agent.h(z), agent.c(z, stored_goal), args.pessimism_coef
        ).cpu().numpy()
        propose = ~persistent.has_goal | ~continuing
        proposed = z + agent.goal(z) + goal_probe
    used = persistent.apply(proposed.cpu().numpy(), propose)
    return used, propose


def exact_scaled_control_cost(actions, control_cost_weight, reward_scale):
    return reward_scale * control_cost_weight * actions.square().sum(dim=-1)


def factual_local_actor_loss(
    agent,
    observation_context,
    context_valid,
    past_nominal_actions,
    factual_next_observations_tensor,
    stored_goal_latents,
    stored_nominal_actions,
    stored_action_deltas,
    control_cost_weight,
    reward_scale,
):
    """Factual anchor plus an explicitly identified local action correction."""
    with torch.no_grad():
        z = agent.world.encode(observation_context[:, -1])
        actual_next = agent.world.encode(factual_next_observations_tensor)
        aligned_stored = torch.cat(
            [past_nominal_actions, stored_nominal_actions[:, None]], dim=1
        )
        jacobian = agent.world.final_jacobian(
            observation_context, aligned_stored, context_valid
        )
        nominal_anchor = actual_next - torch.einsum(
            "bza,ba->bz", jacobian, stored_action_deltas
        )
    fresh_nominal = agent.policy(z.detach(), stored_goal_latents.detach())
    local_next = nominal_anchor + torch.einsum(
        "bza,ba->bz",
        jacobian.detach(),
        fresh_nominal - stored_nominal_actions,
    )
    goal_mse = (local_next - stored_goal_latents.detach()).square().mean(dim=-1)
    cost = exact_scaled_control_cost(
        fresh_nominal, control_cost_weight, reward_scale
    )
    loss = (goal_mse + cost).mean()
    factual_mse = (
        actual_next - stored_goal_latents.detach()
    ).square().mean(dim=-1)
    return loss, fresh_nominal, {
        "objective": -(goal_mse + cost).mean().detach(),
        "local_goal_mse": goal_mse.mean().detach(),
        "factual_goal_mse": factual_mse.mean().detach(),
        "control_cost": cost.mean().detach(),
        "jacobian_norm": jacobian.square().sum(dim=(-2, -1)).sqrt().mean().detach(),
        "action_saturation": (fresh_nominal.abs() > 0.95).float().mean().detach(),
    }


def goal_proposal_loss(
    agent, current_observations, stored_goal_probe, proposal_mask, pessimism_coef
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


def goal_recompute_mse(
    agent, current_observations, probe, stored_goals, proposal_mask
):
    if not proposal_mask.any():
        return torch.zeros((), device=current_observations.device)
    with torch.no_grad():
        z = agent.world.encode(current_observations[proposal_mask])
        recomputed = z + agent.goal(z) + probe[proposal_mask]
    return (recomputed - stored_goals[proposal_mask]).square().mean()


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


class SequenceReplay:
    def __init__(self, capacity, history_length, obs_dim, action_dim):
        self.capacity = capacity
        self.observations = np.empty(
            (capacity, history_length + 1, obs_dim), dtype=np.float32
        )
        self.nominal_actions = np.empty(
            (capacity, history_length, action_dim), dtype=np.float32
        )
        self.action_deltas = np.empty_like(self.nominal_actions)
        self.valid = np.empty((capacity, history_length + 1), dtype=bool)
        self.pointer = 0
        self.size = 0

    def add(self, observations, nominal_actions, action_deltas, valid):
        slots = (self.pointer + np.arange(len(observations))) % self.capacity
        self.observations[slots] = observations
        self.nominal_actions[slots] = nominal_actions
        self.action_deltas[slots] = action_deltas
        self.valid[slots] = valid
        self.pointer = (self.pointer + len(observations)) % self.capacity
        self.size = min(self.capacity, self.size + len(observations))

    def sample(self, count, recent_size, rng, device):
        available = min(self.size, recent_size)
        offsets = rng.integers(0, available, size=count)
        indices = (self.pointer - 1 - offsets) % self.capacity
        return (
            torch.as_tensor(self.observations[indices], device=device),
            torch.as_tensor(self.nominal_actions[indices], device=device),
            torch.as_tensor(self.action_deltas[indices], device=device),
            torch.as_tensor(self.valid[indices], device=device),
        )


def world_parameter_ids(agent):
    return {id(parameter) for parameter in agent.world.parameters()}


def optimizer_parameter_groups(agent):
    return {
        "world": world_parameter_ids(agent),
        "h": {id(parameter) for parameter in agent.h.parameters()},
        "c": {id(parameter) for parameter in agent.c.parameters()},
        "goal": {id(parameter) for parameter in agent.goal.parameters()},
        "policy": {id(parameter) for parameter in agent.policy.parameters()},
        "reward_rate": {
            id(parameter) for parameter in agent.reward_rate.parameters()
        },
    }


def jacobian_probe_sensitivity(
    world, observations, nominal_actions, action_deltas, source_valid=None
):
    with torch.no_grad():
        _, jacobian = world.predict_components(
            observations, nominal_actions, source_valid
        )
        correction = torch.einsum(
            "...za,...a->...z", jacobian, action_deltas
        )
        return correction.square().mean().sqrt()


def probe_covariance_metrics(action_deltas):
    flattened = action_deltas.reshape(-1, action_deltas.shape[-1])
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(centered))
    eigenvalues = torch.linalg.eigvalsh(covariance)
    positive = eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).eps)
    return (
        eigenvalues.min(),
        eigenvalues.max() / positive.min(),
        torch.linalg.matrix_rank(covariance),
    )


def adaln_gate_magnitude(world):
    values = [
        block.adaLN_modulation[-1].weight.abs().mean()
        for block in world.predictor
    ]
    return torch.stack(values).mean()


def procrustes_alignment(raw_new, global_old):
    u, _, vh = torch.linalg.svd(raw_new.T @ global_old, full_matrices=False)
    return u @ vh


def update_world(agent, sigreg, replay, optimizer, args, rng, device):
    observations, nominal_actions, action_deltas, valid = replay.sample(
        args.minibatch_size, args.recent_replay_size, rng, device
    )
    with torch.no_grad():
        anchor_observations = observations[valid]
        old_global = agent.world.encode(anchor_observations)
    optimizer.zero_grad(set_to_none=True)
    loss, prediction_loss, sigreg_loss = masked_world_objective(
        agent.world,
        sigreg,
        observations,
        nominal_actions,
        action_deltas,
        valid,
        args.sigreg_coef,
    )
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        agent.world.parameters(), args.max_grad_norm
    )
    optimizer.step()
    with torch.no_grad():
        raw_new = agent.world.encode_raw(anchor_observations)
        agent.world.alignment.copy_(procrustes_alignment(raw_new, old_global))
    return {
        "loss": loss.detach(),
        "prediction_mse": prediction_loss.detach(),
        "sigreg": sigreg_loss.detach(),
        "gradient_norm": gradient_norm.detach(),
    }


def boundary_next_option_values(
    next_h,
    next_c,
    live_h,
    factual_h,
    terminated,
    truncated,
    average_reward,
    pessimism_coef,
):
    selected, continuing = select_next_branch(
        next_h, next_c, pessimism_coef
    )
    done = terminated | truncated
    if average_reward:
        selected = torch.where(done.unsqueeze(0), live_h, selected)
        bootstrap = torch.ones_like(terminated, dtype=next_h.dtype)
    else:
        selected = torch.where(truncated.unsqueeze(0), factual_h, selected)
        bootstrap = (~terminated).to(next_h.dtype)
    return selected, bootstrap, continuing


def control_cost_weight(envs):
    env = envs.envs[0]
    while hasattr(env, "env"):
        env = env.env
    if not hasattr(env, "_ctrl_cost_weight"):
        raise AttributeError("environment does not expose _ctrl_cost_weight")
    return float(env._ctrl_cost_weight)


def add_episode_metrics(writer, infos, global_step):
    if "final_info" not in infos:
        return
    mask = infos.get(
        "_final_info",
        np.asarray([x is not None for x in infos["final_info"]], dtype=bool),
    )
    for index in np.flatnonzero(mask):
        final_info = infos["final_info"][index]
        if final_info is None or "episode" not in final_info:
            continue
        episode = final_info["episode"]
        writer.add_scalar(
            "charts/episodic_return", float(np.asarray(episode["r"])), global_step
        )
        writer.add_scalar(
            "charts/episodic_length", float(np.asarray(episode["l"])), global_step
        )


def validate_runtime_args(args):
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("this experiment requires CUDA")
    if args.history_length < 1:
        raise ValueError("history_length must be positive")
    if args.num_envs < 2:
        raise ValueError("deterministic zero-mean probes require num_envs >= 2")
    if args.action_probe_amplitude <= 0:
        raise ValueError("action probes must persist with positive amplitude")
    if args.goal_probe_amplitude <= 0:
        raise ValueError("goal probes must persist with positive amplitude")
    if args.average_reward and args.option_discount != 1.0:
        raise ValueError("average-reward options require option_discount=1")
    if args.minibatch_size > args.replay_capacity:
        raise ValueError("minibatch_size cannot exceed replay_capacity")
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.warmup_steps % args.batch_size:
        raise ValueError("warmup_steps must align to a rollout boundary")


def main():
    args = tyro.cli(Args)
    validate_runtime_args(args)
    discount = (
        args.option_discount
        if args.average_reward
        else args.discounted_option_discount
    )
    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    )
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
    world_optimizer = optim.AdamW(
        agent.world.parameters(),
        lr=args.world_lr,
        weight_decay=args.weight_decay,
    )
    h_optimizer = optim.AdamW(
        agent.h.parameters(), lr=args.h_lr, weight_decay=args.weight_decay
    )
    c_optimizer = optim.AdamW(
        agent.c.parameters(), lr=args.c_lr, weight_decay=args.weight_decay
    )
    goal_optimizer = optim.AdamW(
        agent.goal.parameters(),
        lr=args.goal_lr,
        weight_decay=args.weight_decay,
    )
    policy_optimizer = optim.AdamW(
        agent.policy.parameters(),
        lr=args.policy_lr,
        weight_decay=args.weight_decay,
    )
    reward_rate_optimizer = optim.Adam(
        agent.reward_rate.parameters(), lr=args.reward_rate_lr
    )

    def collection_policy(current_observations, goal_latents):
        z = agent.world.encode(current_observations)
        raw = agent.policy.raw_action(z, goal_latents)
        return raw, torch.tanh(raw)

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
    (
        obs_context,
        past_nominal_actions,
        past_action_deltas,
        context_valid,
    ) = initialize_context(observations, args.history_length, action_dim)
    persistent = PersistentLatentGoals(args.num_envs, args.latent_dim)
    cumulative_rate = CumulativeRewardRate()
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        controlled = global_step >= args.warmup_steps
        shape = (args.num_steps, args.num_envs)
        rollout_context = np.empty(
            (*shape, args.history_length, obs_dim), dtype=np.float32
        )
        rollout_past_nominal = np.empty(
            (*shape, args.history_length - 1, action_dim), dtype=np.float32
        )
        rollout_context_valid = np.empty(
            (*shape, args.history_length), dtype=bool
        )
        rollout_factual_next = np.empty((*shape, obs_dim), dtype=np.float32)
        rollout_live_next = np.empty((*shape, obs_dim), dtype=np.float32)
        rollout_nominal = np.empty((*shape, action_dim), dtype=np.float32)
        rollout_action_delta = np.empty_like(rollout_nominal)
        rollout_goal_probe = np.zeros(
            (*shape, args.latent_dim), dtype=np.float32
        )
        rollout_goals = np.zeros(
            (*shape, args.latent_dim), dtype=np.float32
        )
        rollout_proposal = np.zeros(shape, dtype=bool)
        rollout_rewards = np.empty(shape, dtype=np.float32)
        rollout_terminated = np.empty(shape, dtype=bool)
        rollout_truncated = np.empty(shape, dtype=bool)
        raw_forward_total = 0.0
        raw_control_total = 0.0

        for step in range(args.num_steps):
            rollout_context[step] = obs_context
            rollout_past_nominal[step] = past_nominal_actions
            rollout_context_valid[step] = context_valid
            probe_step = global_step // args.num_envs
            raw_action_probe = deterministic_probe(
                probe_step,
                args.num_envs,
                action_dim,
                args.action_probe_amplitude,
                args.action_probe_period,
                device,
            )
            if controlled:
                current = torch.as_tensor(
                    observations, device=device, dtype=torch.float32
                )
                goal_probe = deterministic_probe(
                    probe_step,
                    args.num_envs,
                    args.latent_dim,
                    args.goal_probe_amplitude,
                    args.goal_probe_period,
                    device,
                )
                goal_used, proposal = option_decision(
                    agent, current, persistent, goal_probe, args
                )
                goal_tensor = torch.as_tensor(goal_used, device=device)
                with torch.no_grad():
                    raw_nominal, nominal_tensor = collection_policy(
                        current, goal_tensor
                    )
                    actual_tensor = torch.tanh(raw_nominal + raw_action_probe)
                nominal_action = nominal_tensor.cpu().numpy()
                actual_action = actual_tensor.cpu().numpy()
                action_delta = actual_action - nominal_action
                rollout_goals[step] = goal_used
                rollout_proposal[step] = proposal
                rollout_goal_probe[step, proposal] = (
                    goal_probe[proposal].cpu().numpy()
                )
            else:
                # Exploratory nominal actions cover the space; the small,
                # orthogonal delta remains a genuinely local secant.
                nominal_action = 0.75 * np.stack(
                    [
                        envs.single_action_space.sample()
                        for _ in range(args.num_envs)
                    ]
                ).astype(np.float32)
                requested_delta = raw_action_probe.cpu().numpy()
                actual_action = np.clip(
                    nominal_action + requested_delta, -1.0, 1.0
                ).astype(np.float32)
                action_delta = actual_action - nominal_action

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = envs.step(actual_action)
            factual_next = factual_next_observations(next_observations, infos)
            (
                sequence_obs,
                sequence_nominal,
                sequence_delta,
                sequence_valid,
            ) = transition_sequence(
                obs_context,
                past_nominal_actions,
                past_action_deltas,
                context_valid,
                nominal_action,
                action_delta,
                factual_next,
            )
            done = np.logical_or(terminations, truncations)
            (
                next_context,
                next_past_nominal,
                next_past_delta,
                next_valid,
            ) = next_live_context(
                sequence_obs,
                sequence_nominal,
                sequence_delta,
                sequence_valid,
                next_observations,
                done,
            )
            replay.add(
                sequence_obs,
                sequence_nominal,
                sequence_delta,
                sequence_valid,
            )
            rollout_factual_next[step] = factual_next
            rollout_live_next[step] = next_observations
            rollout_nominal[step] = nominal_action
            rollout_action_delta[step] = action_delta
            rollout_rewards[step] = rewards
            rollout_terminated[step] = terminations
            rollout_truncated[step] = truncations

            if "reward_run" in infos:
                raw_forward_total += float(
                    np.asarray(infos["reward_run"]).mean()
                )
            elif "reward_forward" in infos:
                raw_forward_total += float(
                    np.asarray(infos["reward_forward"]).mean()
                )
            if "reward_ctrl" in infos:
                raw_control_total += float(
                    np.asarray(infos["reward_ctrl"]).mean()
                )
            global_step += args.num_envs
            observations = next_observations
            (
                obs_context,
                past_nominal_actions,
                past_action_deltas,
                context_valid,
            ) = (
                next_context,
                next_past_nominal,
                next_past_delta,
                next_valid,
            )
            persistent.reset(done)
            add_episode_metrics(writer, infos, global_step)

        flat_count = args.batch_size
        context_tensor = torch.as_tensor(
            rollout_context.reshape(
                flat_count, args.history_length, obs_dim
            ),
            device=device,
        )
        past_nominal_tensor = torch.as_tensor(
            rollout_past_nominal.reshape(
                flat_count, args.history_length - 1, action_dim
            ),
            device=device,
        )
        context_valid_tensor = torch.as_tensor(
            rollout_context_valid.reshape(flat_count, args.history_length),
            device=device,
        )
        factual_next_tensor = torch.as_tensor(
            rollout_factual_next.reshape(flat_count, obs_dim), device=device
        )
        live_next_tensor = torch.as_tensor(
            rollout_live_next.reshape(flat_count, obs_dim), device=device
        )
        nominal_tensor = torch.as_tensor(
            rollout_nominal.reshape(flat_count, action_dim), device=device
        )
        action_delta_tensor = torch.as_tensor(
            rollout_action_delta.reshape(flat_count, action_dim), device=device
        )
        goal_tensor = torch.as_tensor(
            rollout_goals.reshape(flat_count, args.latent_dim), device=device
        )
        goal_probe_tensor = torch.as_tensor(
            rollout_goal_probe.reshape(flat_count, args.latent_dim), device=device
        )
        proposal_tensor = torch.as_tensor(
            rollout_proposal.reshape(flat_count), device=device
        )

        if controlled:
            # 1. The actor sees factual successors and only J supplies d(next)/d(action).
            (
                policy_loss,
                recomputed_nominal,
                policy_metrics,
            ) = factual_local_actor_loss(
                agent,
                context_tensor,
                context_valid_tensor,
                past_nominal_tensor,
                factual_next_tensor,
                goal_tensor,
                nominal_tensor,
                action_delta_tensor,
                ctrl_weight,
                args.reward_scale,
            )
            policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_grad = nn.utils.clip_grad_norm_(
                agent.policy.parameters(), args.max_grad_norm
            )
            action_recompute = (
                recomputed_nominal - nominal_tensor
            ).square().mean()
            policy_optimizer.step()
            for key, value in policy_metrics.items():
                writer.add_scalar(f"policy/{key}", float(value), global_step)
            writer.add_scalar(
                "policy/action_recompute_mse",
                float(action_recompute),
                global_step,
            )
            writer.add_scalar(
                "policy/gradient_norm", float(policy_grad), global_step
            )

            # 2. G receives gradients only through the frozen H/C goal surface.
            proposal_loss, proposal_metrics = goal_proposal_loss(
                agent,
                context_tensor[:, -1],
                goal_probe_tensor,
                proposal_tensor,
                args.pessimism_coef,
            )
            exact_goal_mse = goal_recompute_mse(
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
                writer.add_scalar(
                    "goal/gradient_norm", float(goal_grad), global_step
                )
                for key, value in proposal_metrics.items():
                    writer.add_scalar(f"goal/{key}", float(value), global_step)
            writer.add_scalar(
                "goal/exact_recompute_mse", float(exact_goal_mse), global_step
            )

            # 3. H/C learn factual option returns. Rho is an optimized Bellman
            # scalar; the exact cumulative rate below is telemetry only.
            scaled_rewards = args.reward_scale * torch.as_tensor(
                rollout_rewards, device=device
            )
            cumulative_rho = cumulative_rate.update(
                scaled_rewards.cpu().numpy()
            )
            training_rho = (
                agent.reward_rate.value.detach()
                if args.average_reward
                else scaled_rewards.new_zeros(())
            )
            adjusted = (
                scaled_rewards - training_rho
                if args.average_reward
                else scaled_rewards
            )
            terminated = torch.as_tensor(
                rollout_terminated.reshape(flat_count), device=device
            )
            truncated = torch.as_tensor(
                rollout_truncated.reshape(flat_count), device=device
            )
            with torch.no_grad():
                z = agent.world.encode(context_tensor[:, -1])
                factual_next_z = agent.world.encode(factual_next_tensor)
                live_next_z = agent.world.encode(live_next_tensor)
                current_h = agent.h(z)
                current_c = agent.c(z, goal_tensor)
                next_h = agent.h(factual_next_z)
                next_c = agent.c(factual_next_z, goal_tensor)
                live_h = agent.h(live_next_z)
                (
                    selected_next,
                    bootstrap,
                    next_continue,
                ) = boundary_next_option_values(
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
                        args.num_steps,
                        args.num_envs,
                        args.option_heads,
                    ),
                    bootstrap.reshape(args.num_steps, args.num_envs),
                    (~(terminated | truncated))
                    .reshape(args.num_steps, args.num_envs)
                    .to(adjusted.dtype),
                    discount,
                    args.manager_lambda,
                ).reshape(flat_count, args.option_heads).T
                c_targets = torch.where(
                    proposal_tensor.unsqueeze(0),
                    manager_returns,
                    one_step_c,
                )

            if args.average_reward:
                rate_loss = reward_rate_loss(
                    scaled_rewards.reshape(flat_count),
                    current_c,
                    selected_next,
                    bootstrap,
                    agent.reward_rate,
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

            c_predictions = agent.c(z.detach(), goal_tensor.detach())
            c_loss = ensemble_mse(
                c_predictions,
                c_targets,
                torch.ones_like(proposal_tensor),
                args.bootstrap_probability,
            )
            c_optimizer.zero_grad(set_to_none=True)
            c_loss.backward()
            c_grad = nn.utils.clip_grad_norm_(
                agent.c.parameters(), args.max_grad_norm
            )
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
                "option/rho_learned",
                float(agent.reward_rate.value.detach()),
                global_step,
            )
            writer.add_scalar(
                "option/rho_cumulative_metric", cumulative_rho, global_step
            )
            writer.add_scalar("option/rho_loss", float(rate_loss.detach()), global_step)
            writer.add_scalar("option/rho_gradient_norm", float(rate_grad), global_step)
            writer.add_scalar("option/h_loss", float(h_loss), global_step)
            writer.add_scalar("option/c_loss", float(c_loss), global_step)
            writer.add_scalar(
                "option/h_gradient_norm", float(h_grad), global_step
            )
            writer.add_scalar(
                "option/c_gradient_norm", float(c_grad), global_step
            )
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
            writer.add_scalar(
                "option/switch_rate",
                float(proposal_tensor.float().mean()),
                global_step,
            )
            writer.add_scalar(
                "option/surplus_active",
                float(surplus[~proposal_tensor].mean())
                if (~proposal_tensor).any()
                else 0.0,
                global_step,
            )
            writer.add_scalar(
                "option/surplus_proposal",
                float(surplus[proposal_tensor].mean())
                if proposal_tensor.any()
                else 0.0,
                global_step,
            )
            writer.add_scalar(
                "option/next_continue_rate",
                float(next_continue.float().mean()),
                global_step,
            )
            writer.add_scalar(
                "option/realized_adjusted_reward",
                float(adjusted.mean()),
                global_step,
            )
            writer.add_scalar(
                "goal/latent_norm",
                float(goal_tensor.norm(dim=-1).mean()),
                global_step,
            )
            writer.add_scalar(
                "goal/age_mean",
                float(persistent.age[persistent.has_goal].mean())
                if persistent.has_goal.any()
                else 0.0,
                global_step,
            )

        # 4. Only attached-target MSE plus SIGReg update any world parameter.
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
            (
                sampled_obs,
                sampled_nominal,
                sampled_delta,
                sampled_valid,
            ) = replay.sample(
                min(args.minibatch_size, replay.size),
                args.recent_replay_size,
                rng,
                device,
            )
            sensitivity = jacobian_probe_sensitivity(
                agent.world,
                sampled_obs[:, :-1],
                sampled_nominal,
                sampled_delta,
                sampled_valid[:, :-1],
            )
            writer.add_scalar(
                "world/probe_correction_rms",
                float(sensitivity),
                global_step,
            )
            writer.add_scalar(
                "world/action_delta_rms",
                float(sampled_delta.square().mean().sqrt()),
                global_step,
            )
            (
                delta_min_eigenvalue,
                delta_condition,
                delta_rank,
            ) = probe_covariance_metrics(sampled_delta)
            writer.add_scalar(
                "world/action_delta_cov_min_eigenvalue",
                float(delta_min_eigenvalue),
                global_step,
            )
            writer.add_scalar(
                "world/action_delta_cov_condition",
                float(delta_condition),
                global_step,
            )
            writer.add_scalar(
                "world/action_delta_cov_rank",
                float(delta_rank),
                global_step,
            )
            writer.add_scalar(
                "world/adaln_gate_magnitude",
                float(adaln_gate_magnitude(agent.world).detach()),
                global_step,
            )

        writer.add_scalar(
            "reward/raw_total", float(rollout_rewards.mean()), global_step
        )
        writer.add_scalar(
            "reward/raw_forward",
            raw_forward_total / args.num_steps,
            global_step,
        )
        writer.add_scalar(
            "reward/raw_control",
            raw_control_total / args.num_steps,
            global_step,
        )
        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )
        if iteration == 1 or iteration % 10 == 0:
            print(
                f"iteration={iteration} step={global_step} "
                f"controlled={controlled} "
                f"SPS={int(global_step / (time.time() - start_time))}"
            )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

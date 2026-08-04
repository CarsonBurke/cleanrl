# Elite Detached INTACT World Direct v12
#
# A detached physical INTACT auxiliary anchors the shared H12 law only to the
# highest-return demonstrated replay chunks in each minibatch. The world model
# remains predictability-only, while the same law and a forward-objective token
# learn amortized closed-loop control through frozen dynamics. The hypothesis is
# that elite support prevents collapsed on-policy chunks from self-reinforcing.
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
from torch.distributions import Normal
from torch.func import functional_call
from torch.utils.tensorboard import SummaryWriter


VELOCITY_OBSERVATION_INDEX = 8


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
    latent_dim: int = 64
    hidden_dim: int = 256

    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    minibatch_size: int = 512
    gradient_updates: int = 8
    replay_capacity_steps: int = 100_000
    warmup_steps: int = 25_000
    model_horizon: int = 12
    forward_coef: float = 1.0
    sigreg_coef: float = 0.02
    velocity_coef: float = 1.0
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    chunk_horizon: int = 12
    controller_horizon: int = 12
    intact_fixed_std: float = 0.2
    chunk_tail_weight: float = 0.1
    detached_intact_coef: float = 0.03
    intact_elite_fraction: float = 0.25
    direct_objective_coef: float = 1.0
    controller_lr: float = 5e-4
    action_cost_coef: float = 0.1
    factual_intent_scale_ema: float = 0.99
    recent_replay_fraction: float = 0.5
    recent_replay_steps: int = 10_000

    sigreg_projections: int = 256
    sigreg_knots: int = 17
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    batch_size: int = 0
    num_iterations: int = 0


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


def factual_transition_observations(next_observations, infos):
    """Replace vector-env autoreset observations with exact terminal states."""

    factual = np.array(next_observations, copy=True)
    final_observations = infos.get("final_observation")
    if final_observations is None:
        return factual
    final_mask = infos.get(
        "_final_observation",
        np.asarray([obs is not None for obs in final_observations], dtype=bool),
    )
    for env_index in np.flatnonzero(final_mask):
        factual[env_index] = final_observations[env_index]
    return factual


def exact_interval_velocities(infos, num_envs):
    """Recover step-time x_velocity across vector-env autoreset boundaries."""

    velocities = np.empty(num_envs, dtype=np.float32)
    found = np.zeros(num_envs, dtype=bool)
    current = infos.get("x_velocity")
    if current is not None:
        current = np.asarray(current, dtype=np.float32)
        if current.shape != (num_envs,):
            raise RuntimeError(
                "HalfCheetah x_velocity must contain one scalar per environment"
            )
        current_mask = np.asarray(
            infos.get("_x_velocity", np.ones(num_envs, dtype=bool)),
            dtype=bool,
        )
        velocities[current_mask] = current[current_mask]
        found[current_mask] = True

    final_infos = infos.get("final_info")
    if final_infos is not None:
        final_mask = np.asarray(
            infos.get(
                "_final_info",
                [info is not None for info in final_infos],
            ),
            dtype=bool,
        )
        for env_index in np.flatnonzero(final_mask):
            final_info = final_infos[env_index]
            if final_info is None or "x_velocity" not in final_info:
                raise RuntimeError("final_info is missing exact x_velocity")
            velocities[env_index] = np.float32(final_info["x_velocity"])
            found[env_index] = True

    if not found.all():
        missing = np.flatnonzero(~found).tolist()
        raise RuntimeError(
            f"HalfCheetah info is missing exact x_velocity for envs {missing}"
        )
    return velocities


def forward_velocity(observations):
    """HalfCheetah's root x velocity is qvel[0], raw observation index 8."""

    if observations.shape[-1] <= VELOCITY_OBSERVATION_INDEX:
        raise ValueError("observation does not contain HalfCheetah forward velocity")
    return observations[..., VELOCITY_OBSERVATION_INDEX]


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden_dim, out_dim, *, final_std=1.0, layers=2):
    modules = []
    current_dim = in_dim
    for _ in range(layers):
        modules.extend(
            [layer_init(nn.Linear(current_dim, hidden_dim)), nn.SiLU()]
        )
        current_dim = hidden_dim
    modules.append(layer_init(nn.Linear(current_dim, out_dim), std=final_std))
    return nn.Sequential(*modules)


class SIGReg(nn.Module):
    """Sketched isotropic-Gaussian regularizer over explicit [T,B,D] input."""

    def __init__(self, projections=256, knots=17):
        super().__init__()
        if knots < 2:
            raise ValueError("SIGReg requires at least two quadrature knots")
        self.projections = projections
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        quadrature = torch.full((knots,), 2 * dt, dtype=torch.float32)
        quadrature[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("gaussian_cf", gaussian_cf)
        self.register_buffer("weights", quadrature * gaussian_cf)

    def forward(self, x, mask=None):
        if x.ndim != 3:
            raise ValueError("SIGReg expects [time,batch,latent_dim]")
        if mask is None:
            sample_weights = torch.ones(
                x.shape[:2], device=x.device, dtype=x.dtype
            )
        else:
            if mask.shape != x.shape[:2]:
                raise ValueError("SIGReg mask must have shape [time,batch]")
            sample_weights = mask.to(device=x.device, dtype=x.dtype)
        sample_count = sample_weights.sum(dim=1).clamp_min(1.0)
        axes = torch.randn(
            x.shape[-1], self.projections, device=x.device, dtype=x.dtype
        )
        axes = axes / axes.norm(dim=0, keepdim=True).clamp_min(1e-8)
        phase = (x @ axes).unsqueeze(-1) * self.t
        weights = sample_weights[:, :, None, None]
        empirical_cos = phase.cos().mul(weights).sum(dim=1) / sample_count[:, None, None]
        empirical_sin = phase.sin().mul(weights).sum(dim=1) / sample_count[:, None, None]
        error = (empirical_cos - self.gaussian_cf).square() + empirical_sin.square()
        statistic = (error @ self.weights) * sample_count[:, None]
        valid_time = sample_weights.any(dim=1)
        return statistic[valid_time].mean()


class IntentActionLaw(nn.Module):
    """Shared INTACT Gaussian over one flattened fixed-horizon action chunk."""

    def __init__(
        self,
        latent_dim,
        action_dim,
        chunk_horizon,
        hidden_dim,
        fixed_std,
        tail_weight,
    ):
        super().__init__()
        if chunk_horizon < 1:
            raise ValueError("chunk horizon must be positive")
        if fixed_std <= 0.0:
            raise ValueError("fixed standard deviation must be positive")
        if not 0.0 <= tail_weight <= 1.0:
            raise ValueError("chunk tail weight must be in [0,1]")
        self.action_embedding = mlp(action_dim, hidden_dim, latent_dim, layers=1)
        chunk_dim = chunk_horizon * action_dim
        self.predictor = mlp(
            4 * latent_dim,
            hidden_dim,
            chunk_dim,
            final_std=0.01,
            layers=3,
        )
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.chunk_dim = chunk_dim
        self.fixed_log_std = math.log(fixed_std)
        self.tail_weight = tail_weight

    def parameters_for(self, z, intent, previous_action):
        if z.shape != intent.shape:
            raise ValueError("state and intent must have identical shapes")
        embedded_previous_action = self.action_embedding(previous_action)
        features = torch.cat(
            [z, intent, z * intent, embedded_previous_action], dim=-1
        )
        mean = self.predictor(features)
        log_std = torch.full_like(mean, self.fixed_log_std)
        return mean, log_std

    def forward(self, z, intent, previous_action):
        mean, log_std = self.parameters_for(z, intent, previous_action)
        return Normal(mean, log_std.exp())

    def nll(self, z, intent, previous_action, action_chunk):
        if action_chunk.shape[-2:] != (self.chunk_horizon, self.action_dim):
            raise ValueError("action chunk has incompatible horizon or action size")
        flat_chunk = action_chunk.reshape(*action_chunk.shape[:-2], self.chunk_dim)
        log_prob = self(z, intent, previous_action).log_prob(flat_chunk)
        per_coordinate_nll = -log_prob.reshape(
            *log_prob.shape[:-1], self.chunk_horizon, self.action_dim
        )
        first_action_nll = per_coordinate_nll[..., 0, :].mean(dim=-1)
        if self.chunk_horizon == 1:
            return first_action_nll
        tail_nll = per_coordinate_nll[..., 1:, :].mean(dim=(-1, -2))
        return first_action_nll + self.tail_weight * tail_nll


class Agent(nn.Module):
    """World model plus the isolated shared chunk controller."""

    def __init__(self, obs_dim, action_dim, args):
        super().__init__()
        latent_dim, hidden_dim = args.latent_dim, args.hidden_dim
        self.encoder = mlp(obs_dim, hidden_dim, latent_dim, layers=2)
        self.dynamics = mlp(
            latent_dim + action_dim,
            hidden_dim,
            latent_dim,
            final_std=0.01,
            layers=2,
        )
        self.intact_action_law = IntentActionLaw(
            latent_dim,
            action_dim,
            args.chunk_horizon,
            hidden_dim,
            args.intact_fixed_std,
            args.chunk_tail_weight,
        )
        self.forward_objective_intent = nn.Parameter(torch.randn(latent_dim))
        self.register_buffer("factual_intent_mean_square", torch.ones(()))
        self.register_buffer("factual_intent_scale_initialized", torch.tensor(False))
        self.velocity_head = mlp(
            latent_dim, hidden_dim, 1, final_std=0.01, layers=1
        )
        self.interval_velocity_head = mlp(
            latent_dim, hidden_dim, 1, final_std=0.01, layers=1
        )
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def encode(self, observation):
        return self.encoder(observation)

    def predict_next(self, z, action):
        return z + self.dynamics(torch.cat([z, action], dim=-1))

    def predict_velocity(self, z):
        return self.velocity_head(z).squeeze(-1)

    def predict_interval_velocity(self, successor_z):
        return self.interval_velocity_head(successor_z).squeeze(-1)

    def objective_intent(self, batch_size):
        norm = self.forward_objective_intent.norm().clamp_min(1e-8)
        direction = self.forward_objective_intent / norm
        scale = self.factual_intent_mean_square.clamp_min(1e-12).sqrt()
        return (direction * scale).unsqueeze(0).expand(batch_size, -1)

    @torch.no_grad()
    def update_factual_intent_scale(self, factual_intents, ema):
        if factual_intents.ndim != 2 or factual_intents.shape[-1] != self.latent_dim:
            raise ValueError("factual intents must have shape [batch,latent_dim]")
        if not 0.0 <= ema < 1.0:
            raise ValueError("factual intent scale EMA must be in [0,1)")
        mean_square = factual_intents.detach().square().sum(dim=-1).mean()
        if self.factual_intent_scale_initialized:
            self.factual_intent_mean_square.lerp_(mean_square, 1.0 - ema)
        else:
            self.factual_intent_mean_square.copy_(mean_square)
            self.factual_intent_scale_initialized.fill_(True)


def masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype, device=values.device)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)




@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    previous_actions: torch.Tensor
    interval_velocities: torch.Tensor
    valid: torch.Tensor


class SequenceReplayBuffer:
    """CUDA ring replay whose first axis is vector-environment time."""

    def __init__(self, capacity_steps, num_envs, obs_dim, action_dim, device):
        if capacity_steps < 2:
            raise ValueError("replay capacity must be at least two vector steps")
        self.capacity_steps = capacity_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.observations = torch.empty(
            (capacity_steps, num_envs, obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.next_observations = torch.empty_like(self.observations)
        self.actions = torch.empty(
            (capacity_steps, num_envs, action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_actions = torch.empty_like(self.actions)
        self.interval_velocities = torch.empty(
            (capacity_steps, num_envs), dtype=torch.float32, device=self.device
        )
        self.dones = torch.empty(
            (capacity_steps, num_envs), dtype=torch.bool, device=self.device
        )
        self.pointer = 0
        self.size = 0

    def add(
        self,
        observations,
        actions,
        previous_actions,
        next_observations,
        interval_velocities,
        dones,
    ):
        expected_envs = self.num_envs
        arrays = (
            observations,
            actions,
            previous_actions,
            next_observations,
            interval_velocities,
            dones,
        )
        if any(array.shape[0] != expected_envs for array in arrays):
            raise ValueError("every replay field must contain one item per environment")
        index = self.pointer
        self.observations[index].copy_(
            torch.as_tensor(observations, device=self.device)
        )
        self.actions[index].copy_(torch.as_tensor(actions, device=self.device))
        self.previous_actions[index].copy_(
            torch.as_tensor(previous_actions, device=self.device)
        )
        self.next_observations[index].copy_(
            torch.as_tensor(next_observations, device=self.device)
        )
        self.interval_velocities[index].copy_(
            torch.as_tensor(interval_velocities, device=self.device)
        )
        self.dones[index].copy_(torch.as_tensor(dones, device=self.device))
        self.pointer = (self.pointer + 1) % self.capacity_steps
        self.size = min(self.size + 1, self.capacity_steps)

    def _physical_indices(self, logical_indices):
        oldest = self.pointer if self.size == self.capacity_steps else 0
        return (oldest + logical_indices) % self.capacity_steps

    def sample(
        self,
        batch_size,
        model_horizon,
        device=None,
        recent_fraction=0.0,
        recent_steps=0,
    ):
        if device is not None and torch.device(device) != self.device:
            raise ValueError("replay sampling must stay on its storage device")
        if self.size < model_horizon:
            raise RuntimeError("not enough chronological replay for requested horizon")
        # Rejection sampling keeps every H-window within one episode. A terminal
        # transition may be the final member, whose exact terminal observation is
        # a valid o_{t+H}; no earlier reset seam is admitted.
        accepted_starts = []
        accepted_envs = []
        attempts = 0
        maximum_start = self.size - model_horizon
        offsets = torch.arange(model_horizon, device=self.device)
        while sum(item.numel() for item in accepted_starts) < batch_size:
            draw_count = max(2 * batch_size, 128)
            uniform_starts = torch.randint(
                maximum_start + 1,
                (draw_count,),
                device=self.device,
            )
            recent_width = max(1, recent_steps)
            recent_minimum = max(0, maximum_start - recent_width + 1)
            recent_starts = torch.randint(
                recent_minimum,
                maximum_start + 1,
                (draw_count,),
                device=self.device,
            )
            choose_recent = torch.rand(draw_count, device=self.device) < recent_fraction
            starts = torch.where(choose_recent, recent_starts, uniform_starts)
            envs = torch.randint(
                self.num_envs, (draw_count,), device=self.device
            )
            logical = starts[:, None] + offsets[None]
            physical = self._physical_indices(logical)
            dones = self.dones[physical, envs[:, None]]
            keep = ~dones[:, :-1].any(dim=1)
            accepted_starts.append(starts[keep])
            accepted_envs.append(envs[keep])
            attempts += 1
            if attempts >= 64 and sum(
                item.numel() for item in accepted_starts
            ) < batch_size:
                raise RuntimeError(
                    "replay contains too few episode-contiguous H-windows"
                )
        starts = torch.cat(accepted_starts)[:batch_size]
        envs = torch.cat(accepted_envs)[:batch_size]
        logical = starts[:, None] + offsets[None]
        physical = self._physical_indices(logical)
        next_observation_span = self.next_observations[physical, envs[:, None]]
        observations = torch.empty(
            batch_size,
            model_horizon + 1,
            self.observations.shape[-1],
            dtype=torch.float32,
            device=self.device,
        )
        observations[:, 0] = self.observations[
            self._physical_indices(starts), envs
        ]
        observations[:, 1:] = next_observation_span
        return ReplayBatch(
            observations=observations,
            actions=self.actions[physical, envs[:, None]],
            previous_actions=self.previous_actions[physical, envs[:, None]],
            interval_velocities=self.interval_velocities[
                physical, envs[:, None]
            ],
            valid=torch.ones(
                batch_size,
                model_horizon,
                dtype=torch.bool,
                device=self.device,
            ),
        )


def world_model_objective(agent, sigreg, batch, args):
    """H12 JEPA, SIGReg, and exact velocity losses with no controller route."""

    factual_z = agent.encode(batch.observations)
    predicted_z = factual_z[:, 0]
    predicted_successors = []
    for step in range(batch.actions.shape[1]):
        predicted_z = agent.predict_next(predicted_z, batch.actions[:, step])
        predicted_successors.append(predicted_z)
    predicted_successors = torch.stack(predicted_successors, dim=1)

    transition_error = (predicted_successors - factual_z[:, 1:]).square().mean(-1)
    forward_loss = masked_mean(transition_error, batch.valid)
    latent_valid = torch.cat(
        [
            torch.ones_like(batch.valid[:, :1]),
            batch.valid,
        ],
        dim=1,
    )
    # SIGReg's contract is time-major so that each temporal occurrence remains
    # an explicit member of the shared latent batch.
    sigreg_loss = sigreg(
        factual_z.transpose(0, 1), latent_valid.transpose(0, 1)
    )

    state_velocity_target = forward_velocity(batch.observations)
    state_velocity_error = nn.functional.smooth_l1_loss(
        agent.predict_velocity(factual_z),
        state_velocity_target,
        reduction="none",
    )
    factual_interval_error = nn.functional.smooth_l1_loss(
        agent.predict_interval_velocity(factual_z[:, 1:]),
        batch.interval_velocities,
        reduction="none",
    )
    predicted_interval_error = nn.functional.smooth_l1_loss(
        agent.predict_interval_velocity(predicted_successors),
        batch.interval_velocities,
        reduction="none",
    )
    state_velocity_loss = masked_mean(state_velocity_error, latent_valid)
    factual_interval_loss = masked_mean(factual_interval_error, batch.valid)
    predicted_interval_loss = masked_mean(predicted_interval_error, batch.valid)
    velocity_loss = (
        state_velocity_loss
        + factual_interval_loss
        + predicted_interval_loss
    ) / 3.0

    total = (
        args.forward_coef * forward_loss
        + args.sigreg_coef * sigreg_loss
        + args.velocity_coef * velocity_loss
    )
    metrics = {
        "wm_loss": total.detach(),
        "wm_forward_loss": forward_loss.detach(),
        "wm_sigreg_loss": sigreg_loss.detach(),
        "wm_state_velocity_huber": state_velocity_loss.detach(),
        "wm_factual_interval_huber": factual_interval_loss.detach(),
        "wm_predicted_interval_huber": predicted_interval_loss.detach(),
    }
    return total, metrics


def demonstrated_window_returns(action_chunk, interval_velocities, action_cost_coef):
    """Exact summed HalfCheetah reward for each demonstrated replay window."""

    if action_chunk.ndim != 3:
        raise ValueError("action chunks must have shape [batch,horizon,action]")
    if interval_velocities.shape != action_chunk.shape[:2]:
        raise ValueError("interval velocities must match action batch and horizon")
    action_cost = action_cost_coef * action_chunk.square().sum(dim=-1)
    return (interval_velocities - action_cost).sum(dim=-1)


def deterministic_elite_indices(demonstrated_returns, elite_fraction):
    """Select stable top-return indices, preferring lower indices on exact ties."""

    if demonstrated_returns.ndim != 1 or demonstrated_returns.numel() < 1:
        raise ValueError("demonstrated returns must be a nonempty vector")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite fraction must be in (0,1]")
    elite_count = max(1, math.ceil(elite_fraction * demonstrated_returns.shape[0]))
    ranking = torch.argsort(demonstrated_returns, descending=True, stable=True)
    return ranking[:elite_count]


def detached_intact_objective(
    agent,
    factual_z,
    previous_action,
    action_chunk,
    interval_velocities,
    starting_velocities,
    args,
):
    """Fit only elite physical H12 chunks with every world input detached."""

    z_start = factual_z[:, 0].detach()
    local_intent = (factual_z[:, -1] - factual_z[:, 0]).detach()
    previous_action = previous_action.detach()
    action_chunk = action_chunk.detach()
    interval_velocities = interval_velocities.detach()
    starting_velocities = starting_velocities.detach()
    if starting_velocities.shape != action_chunk.shape[:1]:
        raise ValueError("starting velocities must have shape [batch]")
    demonstrated_returns = demonstrated_window_returns(
        action_chunk,
        interval_velocities,
        args.action_cost_coef,
    )
    elite_indices = deterministic_elite_indices(
        demonstrated_returns,
        args.intact_elite_fraction,
    )
    flat_mean, log_std = agent.intact_action_law.parameters_for(
        z_start, local_intent, previous_action
    )
    per_sample_nll = agent.intact_action_law.nll(
        z_start, local_intent, previous_action, action_chunk
    )
    elite_nll = per_sample_nll[elite_indices].mean()
    mean_chunk = flat_mean.reshape_as(action_chunk)
    elite_returns = demonstrated_returns[elite_indices]
    metrics = {
        "demonstrated_return_mean": demonstrated_returns.mean().detach(),
        "elite_return_threshold": elite_returns[-1].detach(),
        "elite_return_mean": elite_returns.mean().detach(),
        "elite_starting_velocity_mean": (
            starting_velocities[elite_indices].mean().detach()
        ),
        "elite_fraction": demonstrated_returns.new_tensor(
            elite_indices.numel() / demonstrated_returns.numel()
        ),
        "elite_count": demonstrated_returns.new_tensor(elite_indices.numel()),
        "elite_detached_nll": elite_nll.detach(),
        "elite_first_action_mae": (
            mean_chunk[elite_indices, 0] - action_chunk[elite_indices, 0]
        ).abs().mean().detach(),
        "elite_chunk_mean_mae": (
            mean_chunk[elite_indices] - action_chunk[elite_indices]
        ).abs().mean().detach(),
        "fixed_std": log_std[elite_indices].exp().mean().detach(),
    }
    return elite_nll, metrics


def _detached_module_state(module):
    state = {
        name: parameter.detach()
        for name, parameter in module.named_parameters()
    }
    state.update({name: buffer.detach() for name, buffer in module.named_buffers()})
    return state


def _frozen_world_step(
    agent,
    predicted_z,
    action,
    action_cost_coef,
    dynamics_state,
    interval_head_state,
):
    delta = functional_call(
        agent.dynamics,
        dynamics_state,
        (torch.cat([predicted_z, action], dim=-1),),
    )
    successor_z = predicted_z + delta
    interval_velocity = functional_call(
        agent.interval_velocity_head,
        interval_head_state,
        (successor_z,),
    ).squeeze(-1)
    action_cost = action_cost_coef * action.square().sum(dim=-1)
    return successor_z, interval_velocity, action_cost


def closed_loop_direct_score(
    agent, z_start, previous_action, horizon, action_cost_coef
):
    """Score deployment-matched replanning through a frozen differentiable WM."""

    if horizon < 1:
        raise ValueError("controller horizon must be positive")
    predicted_z = z_start.detach()
    previous_action = previous_action.detach()
    objective_intent = agent.objective_intent(predicted_z.shape[0])
    dynamics_state = _detached_module_state(agent.dynamics)
    interval_head_state = _detached_module_state(agent.interval_velocity_head)
    score = torch.zeros(
        predicted_z.shape[0], dtype=predicted_z.dtype, device=predicted_z.device
    )
    velocities = []
    action_costs = []
    raw_actions = []
    actions = []
    for _ in range(horizon):
        flat_mean, _ = agent.intact_action_law.parameters_for(
            predicted_z, objective_intent, previous_action
        )
        raw_action = flat_mean.reshape(
            predicted_z.shape[0],
            agent.intact_action_law.chunk_horizon,
            agent.action_dim,
        )[:, 0]
        action = raw_action.clamp(-1.0, 1.0)
        predicted_z, interval_velocity, action_cost = _frozen_world_step(
            agent,
            predicted_z,
            action,
            action_cost_coef,
            dynamics_state,
            interval_head_state,
        )
        score = score + interval_velocity - action_cost
        velocities.append(interval_velocity)
        action_costs.append(action_cost)
        raw_actions.append(raw_action)
        actions.append(action)
        previous_action = action
    return (
        score,
        torch.stack(velocities, dim=1),
        torch.stack(action_costs, dim=1),
        torch.stack(raw_actions, dim=1),
        torch.stack(actions, dim=1),
    )


def direct_controller_objective(agent, z_start, previous_action, args):
    """Maximize closed-loop interval reward without differentiating into the WM."""

    score, velocities, action_costs, raw_actions, actions = closed_loop_direct_score(
        agent,
        z_start,
        previous_action,
        args.controller_horizon,
        args.action_cost_coef,
    )
    objective_intent = agent.objective_intent(z_start.shape[0])
    objective_mean, _ = agent.intact_action_law.parameters_for(
        z_start.detach(), objective_intent, previous_action.detach()
    )
    zero_mean, _ = agent.intact_action_law.parameters_for(
        z_start.detach(),
        torch.zeros_like(objective_intent),
        previous_action.detach(),
    )
    loss = -score.mean() / args.controller_horizon
    metrics = {
        "direct_objective_loss": loss.detach(),
        "imagined_score": score.mean().detach(),
        "imagined_interval_velocity": velocities.mean().detach(),
        "imagined_action_cost": action_costs.mean().detach(),
        "forward_objective_token_norm": (
            agent.forward_objective_intent.norm().detach()
        ),
        "scaled_objective_intent_norm": objective_intent[0].norm().detach(),
        "objective_token_use": (
            objective_mean - zero_mean
        ).abs().mean().detach(),
        "direct_mean_magnitude": raw_actions.abs().mean().detach(),
        "direct_action_saturation": (actions.abs() > 0.95).float().mean().detach(),
    }
    return loss, metrics


def world_model_parameters(agent):
    modules = (
        agent.encoder,
        agent.dynamics,
        agent.velocity_head,
        agent.interval_velocity_head,
    )
    return tuple(parameter for module in modules for parameter in module.parameters())


def intact_law_parameters(agent):
    return tuple(agent.intact_action_law.parameters())


def objective_token_parameters(agent):
    return (agent.forward_objective_intent,)


def controller_parameters(agent):
    return (*intact_law_parameters(agent), *objective_token_parameters(agent))


def combined_controller_objective(
    agent,
    factual_z,
    previous_action,
    action_chunk,
    interval_velocities,
    starting_velocities,
    args,
):
    """Jointly fit physical chunks and amortize forward control through the WM."""

    detached_nll, detached_metrics = detached_intact_objective(
        agent,
        factual_z,
        previous_action,
        action_chunk,
        interval_velocities,
        starting_velocities,
        args,
    )
    direct_loss, direct_metrics = direct_controller_objective(
        agent,
        factual_z[:, 0].detach(),
        previous_action,
        args,
    )
    total = (
        args.detached_intact_coef * detached_nll
        + args.direct_objective_coef * direct_loss
    )
    metrics = {
        "controller_loss": total.detach(),
        **detached_metrics,
        **direct_metrics,
    }
    return total, metrics


def assert_parameters_have_no_gradients(parameters, owner):
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError(f"{owner} received a forbidden cross-objective gradient")


@torch.no_grad()
def direct_action_chunk(agent, z, previous_action):
    """Decode one objective-conditioned chunk and execute its first mean action."""

    objective_intent = agent.objective_intent(z.shape[0])
    flat_mean, log_std = agent.intact_action_law.parameters_for(
        z, objective_intent, previous_action
    )
    chunk = flat_mean.reshape(
        z.shape[0],
        agent.intact_action_law.chunk_horizon,
        agent.action_dim,
    ).clamp(-1.0, 1.0)
    diagnostics = {
        "direct_mean_magnitude": flat_mean.abs().mean(dim=-1),
        "direct_std_mean": log_std.exp().mean(dim=-1),
        "forward_objective_token_norm": agent.forward_objective_intent.norm().expand(
            z.shape[0]
        ),
    }
    return chunk[:, 0], chunk, diagnostics


def participation_rank(x):
    centered = x - x.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(x.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.cuda and torch.cuda.is_available(), "this experiment requires CUDA"
    assert args.env_id == "HalfCheetah-v4", "velocity index 8 is HalfCheetah-specific"
    assert args.latent_dim == 64, "detached INTACT World Direct v12 uses 64-D latents"
    assert args.model_horizon == 12
    assert args.chunk_horizon == args.model_horizon
    assert 1 <= args.controller_horizon <= args.model_horizon
    assert args.intact_fixed_std > 0.0
    assert 0.0 <= args.chunk_tail_weight <= 1.0
    assert args.detached_intact_coef >= 0.0
    assert 0.0 < args.intact_elite_fraction <= 1.0
    assert args.direct_objective_coef >= 0.0
    assert args.controller_lr > 0.0
    assert args.action_cost_coef >= 0.0
    assert 0.0 <= args.factual_intent_scale_ema < 1.0
    assert 0.0 <= args.recent_replay_fraction <= 1.0
    assert args.recent_replay_steps >= 1

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
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, index, args.capture_video, run_name)
            for index in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    assert obs_dim == 17, "HalfCheetah-v4 must expose the raw 17-D state"

    agent = Agent(obs_dim, action_dim, args).to(device)
    sigreg = SIGReg(
        args.sigreg_projections,
        args.sigreg_knots,
    ).to(device)
    wm_parameters = world_model_parameters(agent)
    law_parameters = intact_law_parameters(agent)
    token_parameters = objective_token_parameters(agent)
    controller_params = controller_parameters(agent)
    parameter_groups = [
        {id(parameter) for parameter in parameters}
        for parameters in (wm_parameters, controller_params)
    ]
    assert not (parameter_groups[0] & parameter_groups[1])
    assert set.union(*parameter_groups) == {
        id(parameter) for parameter in agent.parameters()
    }
    wm_optimizer = optim.AdamW(
        wm_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )
    controller_optimizer = optim.AdamW(
        [
            {"params": law_parameters, "weight_decay": args.weight_decay},
            # The token is a direction, so radial weight decay is meaningless.
            {"params": token_parameters, "weight_decay": 0.0},
        ],
        lr=args.controller_lr,
        eps=1e-5,
    )
    wm_loss_function = world_model_objective
    controller_loss_function = combined_controller_objective
    if args.compile:
        wm_loss_function = torch.compile(
            wm_loss_function, mode=args.compile_mode, dynamic=False
        )
        controller_loss_function = torch.compile(
            controller_loss_function, mode=args.compile_mode, dynamic=False
        )
        print(
            "[elite_detached_intact_world_direct_v12] "
            f"compiled isolated WM and controller objectives (mode={args.compile_mode!r})"
        )

    replay = SequenceReplayBuffer(
        args.replay_capacity_steps,
        args.num_envs,
        obs_dim,
        action_dim,
        device,
    )
    next_observation_numpy, _ = envs.reset(seed=args.seed)
    next_observation = torch.as_tensor(
        next_observation_numpy, dtype=torch.float32, device=device
    )
    next_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    previous_action = torch.zeros(
        args.num_envs, action_dim, dtype=torch.float32, device=device
    )
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            wm_optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
            for parameter_group in controller_optimizer.param_groups:
                parameter_group["lr"] = fraction * args.controller_lr

        rollout_saturation = torch.zeros((), device=device)
        rollout_direct_mean_magnitude = torch.zeros((), device=device)
        rollout_direct_std_mean = torch.zeros((), device=device)
        rollout_objective_token_norm = torch.zeros((), device=device)
        rollout_realized_interval_velocity = torch.zeros((), device=device)
        for _ in range(args.num_steps):
            reset = next_done
            previous_action = torch.where(
                reset[:, None], torch.zeros_like(previous_action), previous_action
            )
            observation_before_step = next_observation
            previous_action_before_step = previous_action
            with torch.no_grad():
                z = agent.encode(observation_before_step)
            if global_step < args.warmup_steps:
                action = torch.empty(
                    args.num_envs, action_dim, device=device
                ).uniform_(-1.0, 1.0)
                zero = torch.zeros(args.num_envs, device=device)
                diagnostics = {
                    "direct_mean_magnitude": zero,
                    "direct_std_mean": zero,
                    "forward_objective_token_norm": zero,
                }
            else:
                action, _, diagnostics = direct_action_chunk(
                    agent,
                    z,
                    previous_action,
                )

            (
                next_observation_numpy,
                _,
                terminations,
                truncations,
                infos,
            ) = envs.step(action.cpu().numpy())
            factual_next_numpy = factual_transition_observations(
                next_observation_numpy, infos
            )
            done_numpy = np.logical_or(terminations, truncations)
            interval_velocity_numpy = exact_interval_velocities(
                infos, args.num_envs
            )
            replay.add(
                observation_before_step,
                action,
                previous_action_before_step,
                torch.as_tensor(factual_next_numpy, device=device),
                torch.as_tensor(interval_velocity_numpy, device=device),
                torch.as_tensor(done_numpy, device=device),
            )

            realized_interval_velocity = torch.as_tensor(
                interval_velocity_numpy, device=device
            )
            rollout_saturation += (action.abs() > 0.95).float().mean()
            rollout_direct_mean_magnitude += diagnostics[
                "direct_mean_magnitude"
            ].mean()
            rollout_direct_std_mean += diagnostics["direct_std_mean"].mean()
            rollout_objective_token_norm += diagnostics[
                "forward_objective_token_norm"
            ].mean()
            rollout_realized_interval_velocity += realized_interval_velocity.mean()

            global_step += args.num_envs
            next_observation = torch.as_tensor(
                next_observation_numpy, dtype=torch.float32, device=device
            )
            next_done = torch.as_tensor(done_numpy, device=device)
            previous_action = action
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        update_metrics = []
        required_replay = args.model_horizon
        if replay.size >= required_replay:
            training_batches = [
                replay.sample(
                    args.minibatch_size,
                    args.model_horizon,
                    device,
                    args.recent_replay_fraction,
                    args.recent_replay_steps,
                )
                for _ in range(args.gradient_updates)
            ]
            with torch.no_grad():
                pooled_factual_intents = []
                for batch in training_batches:
                    factual_z = agent.encode(batch.observations)
                    pooled_factual_intents.append(
                        factual_z[:, -1] - factual_z[:, 0]
                    )
                agent.update_factual_intent_scale(
                    torch.cat(pooled_factual_intents),
                    args.factual_intent_scale_ema,
                )
            for batch in training_batches:
                wm_optimizer.zero_grad(set_to_none=True)
                controller_optimizer.zero_grad(set_to_none=True)
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                wm_loss, wm_metrics = wm_loss_function(agent, sigreg, batch, args)
                wm_loss.backward()
                assert_parameters_have_no_gradients(
                    controller_params, "controller"
                )
                nn.utils.clip_grad_norm_(wm_parameters, args.max_grad_norm)
                wm_optimizer.step()
                wm_metrics = {
                    key: value.clone() for key, value in wm_metrics.items()
                }
                wm_optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    factual_z = agent.encode(batch.observations)
                controller_optimizer.zero_grad(set_to_none=True)
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                controller_loss, controller_metrics = controller_loss_function(
                    agent,
                    factual_z,
                    batch.previous_actions[:, 0],
                    batch.actions,
                    batch.interval_velocities,
                    forward_velocity(batch.observations[:, 0]),
                    args,
                )
                controller_loss.backward()
                assert_parameters_have_no_gradients(wm_parameters, "world model")
                # Keep the NLL coefficient from implicitly rescaling the
                # token's reward-only update when law gradients are clipped.
                nn.utils.clip_grad_norm_(law_parameters, args.max_grad_norm)
                nn.utils.clip_grad_norm_(token_parameters, args.max_grad_norm)
                controller_optimizer.step()
                controller_metrics = {
                    key: value.clone() for key, value in controller_metrics.items()
                }
                controller_optimizer.zero_grad(set_to_none=True)
                update_metrics.append(
                    {**wm_metrics, **controller_metrics}
                )

        with torch.no_grad():
            replay_count = min(
                args.minibatch_size, replay.size * args.num_envs
            )
            logical_steps = torch.randint(
                replay.size, (replay_count,), device=device
            )
            env_indices = torch.randint(
                args.num_envs, (replay_count,), device=device
            )
            replay_observations = replay.observations[
                replay._physical_indices(logical_steps), env_indices
            ]
            latent_rank = participation_rank(agent.encode(replay_observations))

        if update_metrics:
            for key in update_metrics[0]:
                writer.add_scalar(
                    f"losses/{key}",
                    torch.stack([metrics[key] for metrics in update_metrics]).mean(),
                    global_step,
                )
        writer.add_scalar(
            "diagnostics/action_saturation",
            rollout_saturation / args.num_steps,
            global_step,
        )
        writer.add_scalar(
            "direct/mean_magnitude",
            rollout_direct_mean_magnitude / args.num_steps,
            global_step,
        )
        writer.add_scalar(
            "direct/std_mean",
            rollout_direct_std_mean / args.num_steps,
            global_step,
        )
        writer.add_scalar(
            "direct/objective_token_norm",
            rollout_objective_token_norm / args.num_steps,
            global_step,
        )
        writer.add_scalar("diagnostics/latent_rank", latent_rank, global_step)
        writer.add_scalar(
            "diagnostics/realized_interval_velocity",
            rollout_realized_interval_velocity / args.num_steps,
            global_step,
        )
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"SPS: {sps}")

    envs.close()
    writer.close()

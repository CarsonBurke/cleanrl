# INTACT Velocity MPC v1
#
# Policy-free online control for HalfCheetah: a LeJEPA latent world model is
# shaped by the shared local/goal INTACT action likelihood, while deployment
# disables that action law and runs sequence CEM on a learned velocity head.
# The hypothesis is that INTACT preserves controllable latent coordinates and
# velocity supervision gives model-predictive search the task objective without
# introducing a policy, Q-function, reward model, or terminal latent target.
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
    local_nll_coef: float = 0.1
    goal_nll_coef: float = 0.05
    velocity_coef: float = 1.0
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    planning_horizon: int = 12
    cem_population: int = 256
    cem_iterations: int = 4
    cem_elites: int = 32
    cem_global_fraction: float = 0.125
    cem_initial_std: float = 0.6
    cem_min_std: float = 0.08
    cem_update_rate: float = 0.7
    # Pure velocity is the requested objective. Set to 0.1 to match the
    # HalfCheetah benchmark's analytic control penalty.
    action_cost_coef: float = 0.0
    random_action_probability: float = 0.02

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
    """Shared INTACT diagonal Gaussian with the exact four-slot grammar."""

    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.action_embedding = mlp(action_dim, hidden_dim, latent_dim, layers=1)
        self.predictor = mlp(
            4 * latent_dim,
            hidden_dim,
            2 * action_dim,
            final_std=0.01,
            layers=3,
        )
        self.action_dim = action_dim

    def parameters_for(self, z, intent, previous_action):
        if z.shape != intent.shape:
            raise ValueError("state and intent must have identical shapes")
        embedded_previous_action = self.action_embedding(previous_action)
        features = torch.cat(
            [z, intent, z * intent, embedded_previous_action], dim=-1
        )
        mean, log_std = self.predictor(features).split(self.action_dim, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def forward(self, z, intent, previous_action):
        mean, log_std = self.parameters_for(z, intent, previous_action)
        return Normal(mean, log_std.exp())

    def nll(self, z, intent, previous_action, action):
        return -self(z, intent, previous_action).log_prob(action).sum(dim=-1)


class Agent(nn.Module):
    """Trainable representation; its INTACT law is never called by planning."""

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
        self.intact_action_law = IntentActionLaw(latent_dim, action_dim, hidden_dim)
        self.velocity_head = mlp(
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


def masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype, device=values.device)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def intact_nll_losses(agent, z, z_next, z_future, action, previous_action):
    """Return local and goal NLLs with INTACT's asymmetric endpoint routes.

    The physical successor remains attached. Only the future endpoint in the
    goal intent is stop-gradient; its current-state endpoint remains plastic.
    There is deliberately no pointwise loss between the two intent families.
    """

    local_intent = z_next - z
    goal_intent = z_future.detach() - z
    local_nll = agent.intact_action_law.nll(
        z, local_intent, previous_action, action
    )
    goal_nll = agent.intact_action_law.nll(
        z, goal_intent, previous_action, action
    )
    return local_nll, goal_nll


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    previous_actions: torch.Tensor
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
        self.dones = torch.empty(
            (capacity_steps, num_envs), dtype=torch.bool, device=self.device
        )
        self.pointer = 0
        self.size = 0

    def add(self, observations, actions, previous_actions, next_observations, dones):
        expected_envs = self.num_envs
        arrays = (observations, actions, previous_actions, next_observations, dones)
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
        self.dones[index].copy_(torch.as_tensor(dones, device=self.device))
        self.pointer = (self.pointer + 1) % self.capacity_steps
        self.size = min(self.size + 1, self.capacity_steps)

    def _physical_indices(self, logical_indices):
        oldest = self.pointer if self.size == self.capacity_steps else 0
        return (oldest + logical_indices) % self.capacity_steps

    def sample(self, batch_size, model_horizon, device=None):
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
            starts = torch.randint(
                maximum_start + 1,
                (draw_count,),
                device=self.device,
            )
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
            valid=torch.ones(
                batch_size,
                model_horizon,
                dtype=torch.bool,
                device=self.device,
            ),
        )


def training_objective(agent, sigreg, batch, args):
    """Horizon-matched self-fed LeJEPA plus INTACT and velocity losses."""

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

    factual_velocity_targets = forward_velocity(batch.observations[:, 1:])
    factual_velocity_error = nn.functional.smooth_l1_loss(
        agent.predict_velocity(factual_z[:, 1:]),
        factual_velocity_targets,
        reduction="none",
    )
    predicted_velocity_error = nn.functional.smooth_l1_loss(
        agent.predict_velocity(predicted_successors),
        factual_velocity_targets,
        reduction="none",
    )
    factual_velocity_loss = masked_mean(factual_velocity_error, batch.valid)
    predicted_velocity_loss = masked_mean(predicted_velocity_error, batch.valid)
    velocity_loss = 0.5 * (factual_velocity_loss + predicted_velocity_loss)

    current_z = factual_z[:, :-1]
    next_z = factual_z[:, 1:]
    # The one fixed terminal anchor z_g=E(o_{t+H}) conditions every action in
    # the sampled H-window. Only this occurrence is detached in the goal route.
    terminal_anchor = factual_z[:, -1:].expand_as(current_z)
    local_intent = next_z - current_z
    goal_intent = terminal_anchor.detach() - current_z
    local_mean, local_log_std = agent.intact_action_law.parameters_for(
        current_z, local_intent, batch.previous_actions
    )
    goal_mean, goal_log_std = agent.intact_action_law.parameters_for(
        current_z, goal_intent, batch.previous_actions
    )
    local_nll = -Normal(local_mean, local_log_std.exp()).log_prob(
        batch.actions
    ).sum(dim=-1)
    local_nll_loss = masked_mean(local_nll, batch.valid)
    goal_nll = -Normal(goal_mean, goal_log_std.exp()).log_prob(
        batch.actions
    ).sum(dim=-1)
    goal_nll_loss = masked_mean(goal_nll, batch.valid)
    shuffled_goal_intent = goal_intent.roll(1, dims=0)
    shuffled_goal_mean, _ = agent.intact_action_law.parameters_for(
        current_z, shuffled_goal_intent, batch.previous_actions
    )

    total = (
        args.forward_coef * forward_loss
        + args.sigreg_coef * sigreg_loss
        + args.local_nll_coef * local_nll_loss
        + args.goal_nll_coef * goal_nll_loss
        + args.velocity_coef * velocity_loss
    )
    metrics = {
        "loss": total.detach(),
        "forward_loss": forward_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
        "local_nll": local_nll_loss.detach(),
        "goal_nll": goal_nll_loss.detach(),
        "factual_velocity_huber": factual_velocity_loss.detach(),
        "predicted_velocity_huber": predicted_velocity_loss.detach(),
        "local_mean_mae": masked_mean(
            (local_mean - batch.actions).abs().mean(-1), batch.valid
        ).detach(),
        "goal_mean_mae": masked_mean(
            (goal_mean - batch.actions).abs().mean(-1), batch.valid
        ).detach(),
        "logstd_saturation": masked_mean(
            ((goal_log_std <= -4.99) | (goal_log_std >= 1.99))
            .float()
            .mean(-1),
            batch.valid,
        ).detach(),
        "goal_condition_use": masked_mean(
            (goal_mean - shuffled_goal_mean).abs().mean(-1), batch.valid
        ).detach(),
    }
    return total, metrics


def sample_cem_action_sequences(
    center,
    std,
    population,
    global_fraction,
    incumbent=None,
    generator=None,
):
    """Sample [B,P,H,A] sequences with global coverage and references."""

    if center.ndim != 3 or std.shape != center.shape:
        raise ValueError("center and std must have shape [batch,horizon,action]")
    if population < 2:
        raise ValueError("CEM population must be at least two")
    if not 0.0 <= global_fraction < 1.0:
        raise ValueError("global_fraction must be in [0,1)")
    global_count = min(population - 1, int(round(population * global_fraction)))
    local_count = population - global_count
    local = center[:, None] + std[:, None] * torch.randn(
        center.shape[0],
        local_count,
        center.shape[1],
        center.shape[2],
        device=center.device,
        dtype=center.dtype,
        generator=generator,
    )
    local[:, 0] = center
    if incumbent is not None:
        if incumbent.shape != center.shape:
            raise ValueError("incumbent must have the same shape as center")
        local[:, 0] = incumbent
        if local_count > 1:
            local[:, 1] = center
    if global_count:
        global_samples = 2.0 * torch.rand(
            center.shape[0],
            global_count,
            center.shape[1],
            center.shape[2],
            device=center.device,
            dtype=center.dtype,
            generator=generator,
        ) - 1.0
        candidates = torch.cat([local, global_samples], dim=1)
    else:
        candidates = local
    return candidates.clamp(-1.0, 1.0)


def cem_sequence_update(
    candidates,
    scores,
    elite_count,
    center,
    std,
    update_rate,
    minimum_std,
):
    """Refit all horizon/action coordinates from elite sequence moments."""

    if candidates.ndim != 4 or scores.shape != candidates.shape[:2]:
        raise ValueError("expected candidates [B,P,H,A] and scores [B,P]")
    if not 1 <= elite_count <= candidates.shape[1]:
        raise ValueError("invalid elite count")
    elite_indices = scores.topk(elite_count, dim=1).indices
    elites = candidates.gather(
        1,
        elite_indices[:, :, None, None].expand(
            -1, -1, candidates.shape[2], candidates.shape[3]
        ),
    )
    elite_center = elites.mean(dim=1)
    elite_std = elites.std(dim=1, unbiased=False)
    next_center = center.lerp(elite_center, update_rate).clamp(-1.0, 1.0)
    next_std = std.lerp(elite_std, update_rate).clamp_min(minimum_std)
    best_indices = scores.argmax(dim=1)
    best = candidates[
        torch.arange(candidates.shape[0], device=candidates.device), best_indices
    ]
    return next_center, next_std, best, elites


@torch.no_grad()
def score_action_sequences(
    agent,
    z,
    action_sequences,
    action_cost_coef=0.0,
    return_velocities=False,
):
    """Score recurrent predicted successors by their horizon velocity sum."""

    if action_sequences.ndim != 4:
        raise ValueError("action_sequences must have shape [B,P,H,A]")
    batch, population, horizon, action_dim = action_sequences.shape
    if z.shape[0] != batch:
        raise ValueError("latent batch does not match candidate batch")
    predicted_z = z[:, None].expand(-1, population, -1).reshape(
        batch * population, -1
    )
    velocities = []
    for step in range(horizon):
        action = action_sequences[:, :, step].reshape(batch * population, action_dim)
        predicted_z = agent.predict_next(predicted_z, action)
        velocities.append(agent.predict_velocity(predicted_z).reshape(batch, population))
    velocities = torch.stack(velocities, dim=-1)
    scores = velocities.sum(dim=-1)
    if action_cost_coef:
        scores = scores - action_cost_coef * action_sequences.square().sum(dim=(-1, -2))
    return (scores, velocities) if return_velocities else scores


def shift_action_sequence(sequence, tail_action=None):
    """Shift an optimized sequence one receding-horizon step to the left."""

    if sequence.ndim != 3:
        raise ValueError("sequence must have shape [B,H,A]")
    shifted = torch.empty_like(sequence)
    shifted[:, :-1] = sequence[:, 1:]
    if tail_action is None:
        shifted[:, -1].zero_()
    else:
        shifted[:, -1] = tail_action
    return shifted


@torch.no_grad()
def direct_action_sequence(agent, z, goal, previous_action, horizon):
    """INTACT's search-free Direct plan, exposed independently of MPC.

    The default executable below deliberately uses velocity CEM for the user's
    cost-based controller. This function retains INTACT's native deployment
    interface: recursively apply the shared Gaussian mean and forward model.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    predicted_z = z
    prior_action = previous_action
    actions = []
    for _ in range(horizon):
        intent = goal - predicted_z
        mean, _ = agent.intact_action_law.parameters_for(
            predicted_z, intent, prior_action
        )
        action = mean.clamp(-1.0, 1.0)
        actions.append(action)
        predicted_z = agent.predict_next(predicted_z, action)
        prior_action = action
    return torch.stack(actions, dim=1)


@torch.no_grad()
def plan_velocity_cem(
    agent,
    z,
    warm_start,
    args,
    generator=None,
    score_function=score_action_sequences,
):
    """Actor-disabled sequence CEM; returns action and shifted warm start."""

    center = warm_start.clone()
    std = torch.full_like(center, args.cem_initial_std)
    incumbent = center.clone()
    incumbent_score = score_function(
        agent, z, incumbent[:, None], args.action_cost_coef
    ).squeeze(1)
    first_random_mean = None
    final_scores = final_elites = None
    for _ in range(args.cem_iterations):
        candidates = sample_cem_action_sequences(
            center,
            std,
            args.cem_population,
            args.cem_global_fraction,
            incumbent,
            generator,
        )
        scores = score_function(
            agent, z, candidates, args.action_cost_coef
        )
        if first_random_mean is None:
            global_count = min(
                args.cem_population - 1,
                int(round(args.cem_population * args.cem_global_fraction)),
            )
            first_random_mean = (
                scores[:, -global_count:].mean(dim=1)
                if global_count
                else scores.mean(dim=1)
            )
        center, std, iteration_best, elites = cem_sequence_update(
            candidates,
            scores,
            args.cem_elites,
            center,
            std,
            args.cem_update_rate,
            args.cem_min_std,
        )
        iteration_best_score = scores.max(dim=1).values
        improved = iteration_best_score > incumbent_score
        incumbent = torch.where(improved[:, None, None], iteration_best, incumbent)
        incumbent_score = torch.maximum(incumbent_score, iteration_best_score)
        final_scores, final_elites = scores, elites

    _, selected_velocities = score_function(
        agent,
        z,
        incumbent[:, None],
        args.action_cost_coef,
        return_velocities=True,
    )
    diagnostics = {
        "score_spread": final_scores.std(dim=1, unbiased=False),
        "elite_std": final_elites.std(dim=1, unbiased=False).mean(dim=(-1, -2)),
        "planner_advantage": incumbent_score - first_random_mean,
        "selected_score": incumbent_score,
        "predicted_first_velocity": selected_velocities[:, 0, 0],
    }
    return incumbent[:, 0], shift_action_sequence(incumbent), diagnostics


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
    assert args.latent_dim == 64, "INTACT velocity MPC v1 uses a 64-D state embedding"
    assert args.model_horizon >= 3, "LeJEPA consistency must be at least three-step"
    assert args.model_horizon == args.planning_horizon, (
        "v1 matches the self-fed training and planning horizons"
    )
    assert args.cem_iterations >= 1
    assert 1 <= args.cem_elites < args.cem_population
    assert args.cem_population * args.cem_global_fraction >= 1

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
    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )
    loss_function = training_objective
    score_function = score_action_sequences
    if args.compile:
        loss_function = torch.compile(
            loss_function, mode=args.compile_mode, dynamic=False
        )
        score_function = torch.compile(
            score_function, mode=args.compile_mode, dynamic=False
        )
        print(
            "[intact_velocity_mpc_v1] "
            f"torch.compile(mode={args.compile_mode!r}, dynamic=False)"
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
    warm_start = torch.zeros(
        args.num_envs,
        args.planning_horizon,
        action_dim,
        dtype=torch.float32,
        device=device,
    )
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = fraction * args.learning_rate

        rollout_velocity_error = torch.zeros((), device=device)
        rollout_saturation = torch.zeros((), device=device)
        rollout_spread = torch.zeros((), device=device)
        rollout_elite_std = torch.zeros((), device=device)
        rollout_advantage = torch.zeros((), device=device)
        rollout_realized_velocity = torch.zeros((), device=device)
        for _ in range(args.num_steps):
            reset = next_done
            previous_action = torch.where(
                reset[:, None], torch.zeros_like(previous_action), previous_action
            )
            warm_start = torch.where(
                reset[:, None, None], torch.zeros_like(warm_start), warm_start
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
                        "score_spread": zero,
                        "elite_std": zero,
                        "planner_advantage": zero,
                        "predicted_first_velocity": agent.predict_velocity(
                            agent.predict_next(z, action)
                        ),
                    }
                else:
                    action, warm_start, diagnostics = plan_velocity_cem(
                        agent,
                        z,
                        warm_start,
                        args,
                        score_function=score_function,
                    )
                    explore = (
                        torch.rand(args.num_envs, device=device)
                        < args.random_action_probability
                    )
                    random_action = torch.empty_like(action).uniform_(-1.0, 1.0)
                    action = torch.where(explore[:, None], random_action, action)
                    warm_start = torch.where(
                        explore[:, None, None],
                        torch.zeros_like(warm_start),
                        warm_start,
                    )
                    # This diagnostic must describe the action actually sent to
                    # MuJoCo, including continuing post-warmup exploration.
                    diagnostics["predicted_first_velocity"] = agent.predict_velocity(
                        agent.predict_next(z, action)
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
            replay.add(
                observation_before_step,
                action,
                previous_action_before_step,
                torch.as_tensor(factual_next_numpy, device=device),
                torch.as_tensor(done_numpy, device=device),
            )

            realized_velocity = torch.as_tensor(
                forward_velocity(factual_next_numpy), device=device
            )
            rollout_velocity_error += (
                diagnostics["predicted_first_velocity"] - realized_velocity
            ).abs().mean()
            rollout_saturation += (action.abs() > 0.95).float().mean()
            rollout_spread += diagnostics["score_spread"].mean()
            rollout_elite_std += diagnostics["elite_std"].mean()
            rollout_advantage += diagnostics["planner_advantage"].mean()
            rollout_realized_velocity += realized_velocity.mean()

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
            for _ in range(args.gradient_updates):
                batch = replay.sample(
                    args.minibatch_size,
                    args.model_horizon,
                    device,
                )
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                loss, metrics = loss_function(agent, sigreg, batch, args)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                # reduce-overhead uses static CUDA Graph output buffers. Clone
                # each scalar before the next replay step overwrites that graph
                # storage; retaining the compiled outputs themselves is invalid.
                update_metrics.append(
                    {key: value.clone() for key, value in metrics.items()}
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
            "diagnostics/predicted_realized_velocity_mae",
            rollout_velocity_error / args.num_steps,
            global_step,
        )
        writer.add_scalar(
            "diagnostics/action_saturation",
            rollout_saturation / args.num_steps,
            global_step,
        )
        writer.add_scalar(
            "planner/score_spread", rollout_spread / args.num_steps, global_step
        )
        writer.add_scalar(
            "planner/elite_action_std",
            rollout_elite_std / args.num_steps,
            global_step,
        )
        writer.add_scalar(
            "planner/advantage", rollout_advantage / args.num_steps, global_step
        )
        writer.add_scalar("diagnostics/latent_rank", latent_rank, global_step)
        writer.add_scalar(
            "diagnostics/realized_forward_velocity",
            rollout_realized_velocity / args.num_steps,
            global_step,
        )
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"SPS: {sps}")

    envs.close()
    writer.close()

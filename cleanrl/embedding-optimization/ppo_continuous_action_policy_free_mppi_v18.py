# Policy-Free MPPI v18
#
# A prediction-only latent world model is trained with H12 JEPA, SIGReg, and
# exact HalfCheetah velocity supervision. Control is entirely online: a compiled
# receding-horizon MPPI solver scores antithetic action perturbations by predicted
# cumulative velocity minus the exact action cost. There is no learned action,
# value, intent, or return model. The 65-candidate/two-update default is the
# lightweight planner setting intended for three concurrent benchmark runs.
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
    recent_replay_fraction: float = 0.5
    recent_replay_steps: int = 10_000

    planner_horizon: int = 12
    mppi_population: int = 65
    mppi_updates: int = 2
    mppi_noise_std: float = 0.5
    mppi_temperature: float = 1.0
    action_cost_coef: float = 0.1

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
            infos.get("_x_velocity", np.ones(num_envs, dtype=bool)), dtype=bool
        )
        velocities[current_mask] = current[current_mask]
        found[current_mask] = True

    final_infos = infos.get("final_info")
    if final_infos is not None:
        final_mask = np.asarray(
            infos.get("_final_info", [info is not None for info in final_infos]),
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
    """HalfCheetah root x velocity is qvel[0], raw observation index 8."""

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


class Agent(nn.Module):
    """Prediction-only latent dynamics and exact velocity heads."""

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


def masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype, device=values.device)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
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
        next_observations,
        interval_velocities,
        dones,
    ):
        arrays = (
            observations,
            actions,
            next_observations,
            interval_velocities,
            dones,
        )
        if any(array.shape[0] != self.num_envs for array in arrays):
            raise ValueError("every replay field must contain one item per environment")
        index = self.pointer
        self.observations[index].copy_(
            torch.as_tensor(observations, device=self.device)
        )
        self.actions[index].copy_(torch.as_tensor(actions, device=self.device))
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
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError("recent fraction must be in [0,1]")

        accepted_starts = []
        accepted_envs = []
        attempts = 0
        maximum_start = self.size - model_horizon
        offsets = torch.arange(model_horizon, device=self.device)
        while sum(item.numel() for item in accepted_starts) < batch_size:
            draw_count = max(2 * batch_size, 128)
            uniform_starts = torch.randint(
                maximum_start + 1, (draw_count,), device=self.device
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
    """H12 JEPA, SIGReg, and exact velocity losses; the sole training objective."""

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
        [torch.ones_like(batch.valid[:, :1]), batch.valid], dim=1
    )
    sigreg_loss = sigreg(
        factual_z.transpose(0, 1), latent_valid.transpose(0, 1)
    )

    state_velocity_target = forward_velocity(batch.observations)
    state_velocity_error = nn.functional.smooth_l1_loss(
        agent.predict_velocity(factual_z), state_velocity_target, reduction="none"
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


def antithetic_gaussian_perturbations(
    updates,
    batch_size,
    population,
    horizon,
    action_dim,
    *,
    device,
    dtype,
    generator=None,
):
    """Return [I,B,K,H,A] noise with zero candidate and exact +/- pairs."""

    if updates < 1:
        raise ValueError("MPPI updates must be positive")
    if population < 3 or population % 2 != 1:
        raise ValueError("MPPI population must be odd and at least three")
    pair_count = (population - 1) // 2
    positive = torch.randn(
        updates,
        batch_size,
        pair_count,
        horizon,
        action_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    zero = torch.zeros(
        updates,
        batch_size,
        1,
        horizon,
        action_dim,
        device=device,
        dtype=dtype,
    )
    return torch.cat([zero, positive, -positive], dim=2)


def shift_action_sequence(action_sequence, reset):
    """Shift one consumed action, repeat the tail, and zero reset environments."""

    if action_sequence.ndim != 3:
        raise ValueError("action sequence must have shape [batch,horizon,action]")
    if reset.shape != action_sequence.shape[:1]:
        raise ValueError("reset must have shape [batch]")
    shifted = torch.cat([action_sequence[:, 1:], action_sequence[:, -1:]], dim=1)
    return torch.where(reset[:, None, None], torch.zeros_like(shifted), shifted)


def action_sequence_population_score(
    agent, z_start, action_sequences, action_cost_coef, return_velocities=False
):
    """Score [B,K,H,A] sequences with reward emitted from successor latents."""

    if action_sequences.ndim != 4:
        raise ValueError("action sequences must have shape [batch,population,horizon,action]")
    batch_size, population, horizon, action_dim = action_sequences.shape
    if z_start.shape != (batch_size, agent.latent_dim):
        raise ValueError("starting latent has incompatible shape")
    if action_dim != agent.action_dim:
        raise ValueError("action sequence has incompatible action size")
    predicted_z = z_start[:, None].expand(-1, population, -1)
    score = torch.zeros(
        batch_size,
        population,
        dtype=z_start.dtype,
        device=z_start.device,
    )
    velocities = []
    for step in range(horizon):
        action = action_sequences[:, :, step]
        flat_z = predicted_z.reshape(batch_size * population, agent.latent_dim)
        flat_action = action.reshape(batch_size * population, action_dim)
        successor_z = agent.predict_next(flat_z, flat_action)
        interval_velocity = agent.predict_interval_velocity(successor_z).reshape(
            batch_size, population
        )
        predicted_z = successor_z.reshape(batch_size, population, agent.latent_dim)
        action_cost = action_cost_coef * action.square().sum(dim=-1)
        score = score + interval_velocity - action_cost
        velocities.append(interval_velocity)
    if return_velocities:
        return score, torch.stack(velocities, dim=-1)
    return score


@torch.no_grad()
def mppi_plan(
    agent,
    z_start,
    warm_start,
    perturbations,
    noise_std,
    temperature,
    action_cost_coef,
):
    """Apply soft score-weighted MPPI updates; every candidate contributes."""

    if temperature <= 0.0:
        raise ValueError("MPPI temperature must be positive")
    if noise_std <= 0.0:
        raise ValueError("MPPI noise standard deviation must be positive")
    if perturbations.ndim != 5:
        raise ValueError("perturbations must have shape [updates,batch,pop,H,A]")
    if perturbations.shape[1] != warm_start.shape[0]:
        raise ValueError("perturbation and warm-start batches must match")
    if perturbations.shape[3:] != warm_start.shape[1:]:
        raise ValueError("perturbation horizon/action shape must match warm start")

    mean = warm_start.detach().clone()
    initial_mean = mean.clone()
    final_scores = None
    final_weights = None
    for update in range(perturbations.shape[0]):
        candidates = (
            mean[:, None] + noise_std * perturbations[update]
        ).clamp(-1.0, 1.0)
        scores = action_sequence_population_score(
            agent, z_start, candidates, action_cost_coef
        )
        logits = (scores - scores.max(dim=1, keepdim=True).values) / temperature
        weights = logits.softmax(dim=1)
        mean = (weights[:, :, None, None] * candidates).sum(dim=1).clamp(-1.0, 1.0)
        final_scores = scores
        final_weights = weights

    selected_score, selected_velocities = action_sequence_population_score(
        agent,
        z_start,
        mean[:, None],
        action_cost_coef,
        return_velocities=True,
    )
    entropy = -(final_weights * final_weights.clamp_min(1e-12).log()).sum(dim=1)
    diagnostics = {
        "selected_score": selected_score[:, 0],
        "predicted_interval_velocity": selected_velocities[:, 0, 0],
        "score_spread": final_scores.std(dim=1, unbiased=False),
        "weight_entropy": entropy,
        "effective_sample_size": final_weights.square().sum(dim=1).reciprocal(),
        "action_change": (mean[:, 0] - initial_mean[:, 0]).norm(dim=-1),
        "sequence_change": (mean - initial_mean).flatten(1).norm(dim=-1),
    }
    return mean, diagnostics


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
    assert args.latent_dim == 64, "policy-free MPPI v18 uses 64-D latents"
    assert args.model_horizon == 12
    assert args.planner_horizon == 12
    assert args.mppi_population >= 3 and args.mppi_population % 2 == 1
    assert args.mppi_updates >= 1
    assert args.mppi_noise_std > 0.0
    assert args.mppi_temperature > 0.0
    assert args.action_cost_coef >= 0.0
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
    sigreg = SIGReg(args.sigreg_projections, args.sigreg_knots).to(device)
    wm_parameters = tuple(agent.parameters())
    wm_optimizer = optim.AdamW(
        wm_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )
    wm_loss_function = world_model_objective
    planner_function = mppi_plan
    if args.compile:
        wm_loss_function = torch.compile(
            wm_loss_function, mode=args.compile_mode, dynamic=False
        )
        planner_function = torch.compile(
            planner_function, mode=args.compile_mode, dynamic=False
        )
        print(
            "[policy_free_mppi_v18] compiled prediction loss and MPPI planner "
            f"(mode={args.compile_mode!r})"
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
    planned_action_sequence = torch.zeros(
        args.num_envs,
        args.planner_horizon,
        action_dim,
        dtype=torch.float32,
        device=device,
    )
    planner_generator = torch.Generator(device=device)
    planner_generator.manual_seed(args.seed)
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            wm_optimizer.param_groups[0]["lr"] = fraction * args.learning_rate

        rollout_saturation = torch.zeros((), device=device)
        rollout_realized_interval_velocity = torch.zeros((), device=device)
        rollout_predicted_interval_velocity = torch.zeros((), device=device)
        rollout_score = torch.zeros((), device=device)
        rollout_score_spread = torch.zeros((), device=device)
        rollout_weight_entropy = torch.zeros((), device=device)
        rollout_effective_sample_size = torch.zeros((), device=device)
        rollout_action_change = torch.zeros((), device=device)
        for _ in range(args.num_steps):
            reset = next_done
            warm_start_sequence = shift_action_sequence(
                planned_action_sequence, reset
            )
            observation_before_step = next_observation
            with torch.no_grad():
                z = agent.encode(observation_before_step)
            if global_step < args.warmup_steps:
                action = torch.empty(
                    args.num_envs, action_dim, device=device
                ).uniform_(-1.0, 1.0)
                planned_action_sequence = torch.zeros_like(planned_action_sequence)
                zero = torch.zeros(args.num_envs, device=device)
                diagnostics = {
                    "selected_score": zero,
                    "predicted_interval_velocity": zero,
                    "score_spread": zero,
                    "weight_entropy": zero,
                    "effective_sample_size": zero,
                    "action_change": zero,
                }
            else:
                perturbations = antithetic_gaussian_perturbations(
                    args.mppi_updates,
                    args.num_envs,
                    args.mppi_population,
                    args.planner_horizon,
                    action_dim,
                    device=device,
                    dtype=z.dtype,
                    generator=planner_generator,
                )
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                planned_action_sequence, diagnostics = planner_function(
                    agent,
                    z,
                    warm_start_sequence,
                    perturbations,
                    args.mppi_noise_std,
                    args.mppi_temperature,
                    args.action_cost_coef,
                )
                planned_action_sequence = planned_action_sequence.clone()
                diagnostics = {
                    key: value.clone() for key, value in diagnostics.items()
                }
                action = planned_action_sequence[:, 0]

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
                torch.as_tensor(factual_next_numpy, device=device),
                torch.as_tensor(interval_velocity_numpy, device=device),
                torch.as_tensor(done_numpy, device=device),
            )

            realized_interval_velocity = torch.as_tensor(
                interval_velocity_numpy, device=device
            )
            rollout_saturation += (action.abs() > 0.95).float().mean()
            rollout_realized_interval_velocity += realized_interval_velocity.mean()
            rollout_predicted_interval_velocity += diagnostics[
                "predicted_interval_velocity"
            ].mean()
            rollout_score += diagnostics["selected_score"].mean()
            rollout_score_spread += diagnostics["score_spread"].mean()
            rollout_weight_entropy += diagnostics["weight_entropy"].mean()
            rollout_effective_sample_size += diagnostics[
                "effective_sample_size"
            ].mean()
            rollout_action_change += diagnostics["action_change"].mean()

            global_step += args.num_envs
            next_observation = torch.as_tensor(
                next_observation_numpy, dtype=torch.float32, device=device
            )
            next_done = torch.as_tensor(done_numpy, device=device)
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
        if replay.size >= args.model_horizon:
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
            for batch in training_batches:
                wm_optimizer.zero_grad(set_to_none=True)
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                wm_loss, wm_metrics = wm_loss_function(agent, sigreg, batch, args)
                wm_loss.backward()
                nn.utils.clip_grad_norm_(wm_parameters, args.max_grad_norm)
                wm_optimizer.step()
                wm_metrics = {
                    key: value.clone() for key, value in wm_metrics.items()
                }
                wm_optimizer.zero_grad(set_to_none=True)
                update_metrics.append(wm_metrics)

        with torch.no_grad():
            replay_count = min(args.minibatch_size, replay.size * args.num_envs)
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
        divisor = args.num_steps
        writer.add_scalar(
            "diagnostics/action_saturation", rollout_saturation / divisor, global_step
        )
        writer.add_scalar(
            "diagnostics/realized_interval_velocity",
            rollout_realized_interval_velocity / divisor,
            global_step,
        )
        writer.add_scalar(
            "planner/predicted_interval_velocity",
            rollout_predicted_interval_velocity / divisor,
            global_step,
        )
        writer.add_scalar("planner/selected_score", rollout_score / divisor, global_step)
        writer.add_scalar(
            "planner/score_spread", rollout_score_spread / divisor, global_step
        )
        writer.add_scalar(
            "planner/weight_entropy", rollout_weight_entropy / divisor, global_step
        )
        writer.add_scalar(
            "planner/effective_sample_size",
            rollout_effective_sample_size / divisor,
            global_step,
        )
        writer.add_scalar(
            "planner/action_change", rollout_action_change / divisor, global_step
        )
        writer.add_scalar("diagnostics/latent_rank", latent_rank, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"SPS: {sps}")

    envs.close()
    writer.close()

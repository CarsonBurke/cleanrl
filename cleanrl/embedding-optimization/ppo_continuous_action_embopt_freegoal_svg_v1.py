# Embedding Optimization: Free-Goal State-Value Gradients v1
#
# A stochastic goal actor proposes unrestricted Euclidean one-step latent
# displacements. A factual local affine controllability model maps each proposal
# to a bounded action. The primary goal update is an unclipped on-policy
# score-function gradient from factual generalized advantages computed with a
# state-only value ensemble. A confidence-gated pathwise state-value gradient
# through a nonlinear forward ensemble is auxiliary. Goals are neither replay
# transitions nor projected directions. LeJEPA and SIGReg learn the online latent
# geometry. There is no action-conditioned value, target model, moving-average
# model, likelihood ratio between policies, or goal horizon.
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
    replay_capacity: int = 262_144
    recent_replay_size: int = 131_072
    warmup_steps: int = 32_768
    minibatch_size: int = 1_024
    updates_per_iteration: int = 32

    latent_dim: int = 32
    hidden_dim: int = 256
    ensemble_size: int = 5
    goal_logstd_min: float = -4.0
    goal_logstd_max: float = 1.0
    controller_ridge: float = 0.05

    representation_lr: float = 1e-4
    model_lr: float = 3e-4
    actor_lr: float = 1e-4
    weight_decay: float = 1e-4
    sigreg_coef: float = 0.09
    reward_scale: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    pessimism_coef: float = 1.0
    pathwise_aux_coef: float = 0.05
    model_confidence_scale: float = 0.05
    goal_delivery_coef: float = 0.25
    disagreement_coef: float = 0.10
    goal_entropy_coef: float = 1e-3
    max_grad_norm: float = 0.5

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


def factual_next_observations(next_observations, infos):
    """Restore terminal observations hidden by vector-environment autoreset."""
    factual = np.array(next_observations, copy=True)
    if "final_observation" not in infos:
        return factual
    mask = infos.get(
        "_final_observation",
        np.asarray([item is not None for item in infos["final_observation"]], dtype=bool),
    )
    for index in np.flatnonzero(mask):
        factual[index] = infos["final_observation"][index]
    return factual


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden_dim, out_dim, out_std=1.0, layer_norm=False):
    layers = [layer_init(nn.Linear(in_dim, hidden_dim))]
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.extend(
        [
            nn.SiLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_dim, out_dim), std=out_std),
        ]
    )
    return nn.Sequential(*layers)


class SIGReg(nn.Module):
    """Characteristic-function regularizer used by the LeJEPA lineage."""

    def __init__(self, projections=256, knots=17, reference_samples=128):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        quadrature = torch.full((knots,), 2 * dt, dtype=torch.float32)
        quadrature[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2)
        self.projections = projections
        self.reference_samples = reference_samples
        self.register_buffer("t", t)
        self.register_buffer("gaussian_cf", gaussian_cf)
        self.register_buffer("weights", quadrature * gaussian_cf)

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


class GoalActor(nn.Module):
    """State-conditioned distribution over unrestricted Euclidean displacements."""

    def __init__(self, latent_dim, hidden_dim, logstd_min, logstd_max):
        super().__init__()
        self.trunk = mlp(latent_dim, hidden_dim, 2 * latent_dim, out_std=0.01)
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

    def forward(self, y):
        mean, raw_logstd = self.trunk(y).chunk(2, dim=-1)
        # Only uncertainty is bounded. The displacement mean and sampled goal
        # retain their native Euclidean magnitude.
        logstd = self.logstd_min + 0.5 * (
            self.logstd_max - self.logstd_min
        ) * (torch.tanh(raw_logstd) + 1)
        return mean, logstd

    def sample(self, y):
        mean, logstd = self(y)
        noise = torch.randn_like(mean)
        goal_delta = mean + logstd.exp() * noise
        return goal_delta, mean, logstd

    def log_prob(self, y, goal_delta):
        mean, logstd = self(y)
        standardized = (goal_delta - mean) / logstd.exp()
        return -0.5 * (
            standardized.square() + 2 * logstd + math.log(2 * math.pi)
        ).sum(dim=-1)

    def entropy(self, y):
        _, logstd = self(y)
        return (
            logstd + 0.5 * (1.0 + math.log(2 * math.pi))
        ).sum(dim=-1)


def ridge_solve(control_matrix, target_delta, ridge):
    """Differentiable batched ridge solution for B a = target_delta."""
    transposed = control_matrix.transpose(-2, -1)
    gram = transposed @ control_matrix
    identity = torch.eye(
        gram.shape[-1], device=gram.device, dtype=gram.dtype
    ).expand_as(gram)
    rhs = (transposed @ target_delta.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.solve(gram + ridge * identity, rhs)


class LocalAffineController(nn.Module):
    """Learned local dynamics delta = drift(y) + B(y) action."""

    def __init__(self, latent_dim, action_dim, hidden_dim, ridge):
        super().__init__()
        self.model = mlp(
            latent_dim,
            hidden_dim,
            latent_dim + latent_dim * action_dim,
            out_std=0.01,
            layer_norm=True,
        )
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.ridge = ridge

    def coefficients(self, y):
        output = self.model(y)
        drift = output[..., : self.latent_dim]
        control = output[..., self.latent_dim :].reshape(
            *y.shape[:-1], self.latent_dim, self.action_dim
        )
        return drift, control

    def predict_delta(self, y, action):
        drift, control = self.coefficients(y)
        return drift + (control @ action.unsqueeze(-1)).squeeze(-1)

    def action_for_goal(self, y, desired_delta):
        drift, control = self.coefficients(y)
        unconstrained_action = ridge_solve(
            control, desired_delta - drift, self.ridge
        )
        return torch.tanh(unconstrained_action)


class ForwardEnsemble(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim, ensemble_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                mlp(
                    latent_dim + action_dim,
                    hidden_dim,
                    latent_dim,
                    out_std=0.01,
                    layer_norm=True,
                )
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, y, action):
        features = torch.cat([y, action], dim=-1)
        return torch.stack([head(features) for head in self.heads], dim=0)


class StateValueEnsemble(nn.Module):
    """An ensemble of scalar functions of latent state only."""

    def __init__(self, latent_dim, hidden_dim, ensemble_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [mlp(latent_dim, hidden_dim, 1, out_std=0.01) for _ in range(ensemble_size)]
        )

    def forward(self, y):
        return torch.stack(
            [head(y).squeeze(-1) for head in self.heads], dim=0
        )


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, args):
        super().__init__()
        latent = args.latent_dim
        hidden = args.hidden_dim
        ensemble = args.ensemble_size
        self.encoder = mlp(obs_dim, hidden, latent, layer_norm=True)
        self.lejepa_dynamics = mlp(
            latent + action_dim,
            hidden,
            latent,
            out_std=0.01,
            layer_norm=True,
        )
        self.goal_actor = GoalActor(
            latent,
            hidden,
            args.goal_logstd_min,
            args.goal_logstd_max,
        )
        self.controller = LocalAffineController(
            latent, action_dim, hidden, args.controller_ridge
        )
        self.forward_ensemble = ForwardEnsemble(
            latent, action_dim, hidden, ensemble
        )
        self.value_ensemble = StateValueEnsemble(latent, hidden, ensemble)
        self.register_buffer("alignment", torch.eye(latent))

    def encode_raw(self, obs):
        return self.encoder(obs)

    def encode(self, obs):
        return self.encode_raw(obs) @ self.alignment

    def lejepa_predict_next(self, y, action):
        return y + self.lejepa_dynamics(torch.cat([y, action], dim=-1))

    def act(self, obs):
        y = self.encode(obs)
        desired_delta, mean, logstd = self.goal_actor.sample(y)
        action = self.controller.action_for_goal(y, desired_delta)
        return action, desired_delta, mean, logstd


class Replay:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.pointer = 0
        self.size = 0

    def add(self, obs, actions, rewards, next_obs, terminated):
        count = obs.shape[0]
        slots = (np.arange(count) + self.pointer) % self.capacity
        self.obs[slots] = obs
        self.actions[slots] = actions
        self.rewards[slots] = rewards
        self.next_obs[slots] = next_obs
        self.terminated[slots] = terminated
        self.pointer = (self.pointer + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample_recent_indices(self, count, recent_size, rng):
        if self.size == 0:
            raise ValueError("cannot sample empty replay")
        available = min(self.size, recent_size)
        offsets = rng.integers(0, available, size=count)
        return (self.pointer - 1 - offsets) % self.capacity


@contextlib.contextmanager
def frozen_parameters(*modules):
    """Freeze model weights while preserving derivatives with respect to inputs."""
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous):
            parameter.requires_grad_(requires_grad)


def pessimistic_estimate(predictions, coefficient):
    return predictions.mean(dim=0) - coefficient * predictions.std(
        dim=0, unbiased=False
    )


def procrustes_alignment(raw_new, global_old):
    """Orthogonal chart transport preserving Euclidean distances and magnitude."""
    u, _, vh = torch.linalg.svd(raw_new.T @ global_old, full_matrices=False)
    return u @ vh


def factual_gae(
    rewards,
    values,
    next_values,
    bootstrap_nonterminal,
    trace_nonterminal,
    gamma,
    gae_lambda,
):
    """Advantages from factual rewards with episode-safe reverse recursion."""
    deltas = rewards + gamma * bootstrap_nonterminal * next_values - values
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(rewards[-1])
    for step in reversed(range(rewards.shape[0])):
        accumulator = (
            deltas[step]
            + gamma * gae_lambda * trace_nonterminal[step] * accumulator
        )
        advantages[step] = accumulator
    return advantages


def goal_score_loss(goal_actor, y, factual_goals, advantages):
    """Unclipped causal gradient for the goals that generated the rollout."""
    log_probability = goal_actor.log_prob(y, factual_goals)
    return -(log_probability * advantages.detach()).mean(), log_probability


def pathwise_auxiliary(agent, y, args):
    """Confidence-gated analytic state-value gradient; never the primary signal."""
    desired_delta, mean, logstd = agent.goal_actor.sample(y.detach())
    with frozen_parameters(
        agent.controller,
        agent.forward_ensemble,
        agent.value_ensemble,
    ):
        action = agent.controller.action_for_goal(y.detach(), desired_delta)
        predicted_deltas = agent.forward_ensemble(y.detach(), action)
        next_states = y.detach().unsqueeze(0) + predicted_deltas
        ensemble_size, batch_size, latent_dim = next_states.shape
        flat_next = next_states.reshape(-1, latent_dim)
        value_matrix = agent.value_ensemble(flat_next)
        next_values = value_matrix.reshape(
            ensemble_size, ensemble_size, batch_size
        )
        head_indices = torch.arange(ensemble_size, device=y.device)
        next_values = next_values[head_indices, head_indices]
        pessimistic_next_value = pessimistic_estimate(
            next_values, args.pessimism_coef
        )

        achieved_mean = predicted_deltas.mean(dim=0)
        delivery_error = (desired_delta - achieved_mean).square().mean(dim=-1)
        disagreement = predicted_deltas.var(dim=0, unbiased=False).mean(dim=-1)
        confidence = (
            1.0 / (1.0 + disagreement / args.model_confidence_scale)
        ).detach()
        auxiliary_objective = confidence * (
            args.gamma * pessimistic_next_value
            - args.goal_delivery_coef * delivery_error
            - args.disagreement_coef * disagreement
        )

    metrics = {
        "pathwise_objective": auxiliary_objective.mean().detach(),
        "predicted_next_value": pessimistic_next_value.mean().detach(),
        "model_confidence": confidence.mean().detach(),
        "delivery_mse": delivery_error.mean().detach(),
        "forward_disagreement": disagreement.mean().detach(),
        "goal_mean_norm": mean.norm(dim=-1).mean().detach(),
        "goal_sample_norm": desired_delta.norm(dim=-1).mean().detach(),
        "goal_logstd": logstd.mean().detach(),
        "predicted_action_saturation": (action.abs() > 0.95).float().mean().detach(),
    }
    return -auxiliary_objective.mean(), metrics


def fresh_actor_loss(agent, y, factual_goals, advantages, args):
    score_loss, log_probability = goal_score_loss(
        agent.goal_actor, y.detach(), factual_goals.detach(), advantages
    )
    auxiliary_loss, metrics = pathwise_auxiliary(agent, y.detach(), args)
    entropy = agent.goal_actor.entropy(y.detach()).mean()
    total = (
        score_loss
        + args.pathwise_aux_coef * auxiliary_loss
        - args.goal_entropy_coef * entropy
    )
    metrics.update(
        {
            "actor_loss": total.detach(),
            "goal_score_loss": score_loss.detach(),
            "goal_log_probability": log_probability.mean().detach(),
            "goal_entropy": entropy.detach(),
            "factual_advantage": advantages.mean().detach(),
            "factual_advantage_std": advantages.std(unbiased=False).detach(),
        }
    )
    return total, metrics


def update_models(
    agent,
    sigreg,
    replay,
    optimizers,
    args,
    rng,
    device,
):
    indices = replay.sample_recent_indices(
        args.minibatch_size, args.recent_replay_size, rng
    )
    obs = torch.as_tensor(replay.obs[indices], device=device)
    actions = torch.as_tensor(replay.actions[indices], device=device)
    rewards = torch.as_tensor(replay.rewards[indices], device=device)
    next_obs = torch.as_tensor(replay.next_obs[indices], device=device)
    terminated = torch.as_tensor(replay.terminated[indices], device=device)

    with torch.no_grad():
        old_global = agent.encode(obs)
    y = agent.encode(obs)
    next_y = agent.encode(next_obs)
    lejepa_prediction = agent.lejepa_predict_next(y, actions)
    prediction_loss = nn.functional.mse_loss(
        lejepa_prediction, next_y.detach()
    )
    sigreg_loss = 0.5 * (sigreg(y) + sigreg(next_y))
    representation_loss = prediction_loss + args.sigreg_coef * sigreg_loss
    optimizers["representation"].zero_grad(set_to_none=True)
    representation_loss.backward()
    nn.utils.clip_grad_norm_(
        list(agent.encoder.parameters()) + list(agent.lejepa_dynamics.parameters()),
        args.max_grad_norm,
    )
    optimizers["representation"].step()

    with torch.no_grad():
        new_raw = agent.encode_raw(obs)
        new_alignment = procrustes_alignment(new_raw, old_global)
        chart_residual = (
            (new_raw @ new_alignment - old_global).square().mean().sqrt()
        )
        agent.alignment.copy_(new_alignment)
        y = agent.encode(obs)
        next_y = agent.encode(next_obs)
        factual_delta = next_y - y

    affine_prediction = agent.controller.predict_delta(y, actions)
    controller_loss = nn.functional.mse_loss(
        affine_prediction, factual_delta
    )
    optimizers["controller"].zero_grad(set_to_none=True)
    controller_loss.backward()
    nn.utils.clip_grad_norm_(agent.controller.parameters(), args.max_grad_norm)
    optimizers["controller"].step()

    forward_predictions = agent.forward_ensemble(y, actions)
    forward_loss = nn.functional.mse_loss(
        forward_predictions,
        factual_delta.unsqueeze(0).expand_as(forward_predictions),
    )
    optimizers["forward"].zero_grad(set_to_none=True)
    forward_loss.backward()
    nn.utils.clip_grad_norm_(agent.forward_ensemble.parameters(), args.max_grad_norm)
    optimizers["forward"].step()

    scaled_rewards = args.reward_scale * rewards
    values = agent.value_ensemble(y)
    with torch.no_grad():
        next_values = agent.value_ensemble(next_y)
        value_targets = scaled_rewards.unsqueeze(0) + args.gamma * (
            1.0 - terminated.unsqueeze(0)
        ) * next_values
    value_loss = nn.functional.mse_loss(values, value_targets)
    optimizers["value"].zero_grad(set_to_none=True)
    value_loss.backward()
    nn.utils.clip_grad_norm_(agent.value_ensemble.parameters(), args.max_grad_norm)
    optimizers["value"].step()

    metrics = {
        "representation_loss": representation_loss.detach(),
        "prediction_mse": prediction_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
        "chart_residual": chart_residual.detach(),
        "controller_mse": controller_loss.detach(),
        "forward_mse": forward_loss.detach(),
        "value_td_mse": value_loss.detach(),
        "factual_delta_norm": factual_delta.norm(dim=-1).mean().detach(),
    }
    return metrics


def main():
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("free-goal SVG v1 requires CUDA")
    if args.recent_replay_size > args.replay_capacity:
        raise ValueError("recent_replay_size cannot exceed replay_capacity")

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
        + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
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
    agent = Agent(obs_dim, action_dim, args).to(device)
    sigreg = SIGReg().to(device)
    optimizers = {
        "representation": optim.AdamW(
            list(agent.encoder.parameters())
            + list(agent.lejepa_dynamics.parameters()),
            lr=args.representation_lr,
            weight_decay=args.weight_decay,
        ),
        "controller": optim.AdamW(
            agent.controller.parameters(),
            lr=args.model_lr,
            weight_decay=args.weight_decay,
        ),
        "forward": optim.AdamW(
            agent.forward_ensemble.parameters(),
            lr=args.model_lr,
            weight_decay=args.weight_decay,
        ),
        "value": optim.AdamW(
            agent.value_ensemble.parameters(),
            lr=args.model_lr,
            weight_decay=args.weight_decay,
        ),
        "actor": optim.AdamW(
            agent.goal_actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.weight_decay,
        ),
    }

    act = agent.act
    if args.compile:
        act = torch.compile(act, mode=args.compile_mode)

    replay = Replay(args.replay_capacity, obs_dim, action_dim)
    observations, _ = envs.reset(seed=args.seed)
    global_step = 0
    start_time = time.time()
    latest_metrics = {}

    for iteration in range(1, args.num_iterations + 1):
        rollout_obs = np.empty(
            (args.num_steps, args.num_envs, obs_dim), dtype=np.float32
        )
        rollout_next_obs = np.empty_like(rollout_obs)
        rollout_goals = np.empty(
            (args.num_steps, args.num_envs, args.latent_dim), dtype=np.float32
        )
        rollout_rewards = np.empty(
            (args.num_steps, args.num_envs), dtype=np.float32
        )
        rollout_bootstrap_nonterminal = np.empty_like(rollout_rewards)
        rollout_trace_nonterminal = np.empty_like(rollout_rewards)
        collection_action_saturation = 0.0
        collection_goal_norm = 0.0
        collection_goal_std = 0.0
        controlled_steps = 0
        controlled_iteration = global_step >= args.warmup_steps

        for step in range(args.num_steps):
            rollout_obs[step] = observations
            if not controlled_iteration:
                actions = np.stack(
                    [envs.single_action_space.sample() for _ in range(args.num_envs)]
                )
                rollout_goals[step].fill(0)
            else:
                obs_tensor = torch.as_tensor(
                    observations, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    action_tensor, goal_delta, _, logstd = act(obs_tensor)
                actions = action_tensor.cpu().numpy()
                rollout_goals[step] = goal_delta.cpu().numpy()
                collection_goal_norm += goal_delta.norm(dim=-1).mean().item()
                collection_goal_std += logstd.exp().mean().item()
                controlled_steps += 1

            next_observations, rewards, terminations, truncations, infos = envs.step(
                actions
            )
            factual_next = factual_next_observations(next_observations, infos)
            rollout_next_obs[step] = factual_next
            rollout_rewards[step] = rewards
            rollout_bootstrap_nonterminal[step] = 1.0 - terminations.astype(
                np.float32
            )
            rollout_trace_nonterminal[step] = 1.0 - np.logical_or(
                terminations, truncations
            ).astype(np.float32)
            replay.add(
                observations,
                actions,
                rewards,
                factual_next,
                terminations.astype(np.float32),
            )
            collection_action_saturation += float((np.abs(actions) > 0.95).mean())
            global_step += args.num_envs
            observations = next_observations

            if "final_info" in infos:
                final_mask = infos.get(
                    "_final_info",
                    np.asarray(
                        [item is not None for item in infos["final_info"]], dtype=bool
                    ),
                )
                for index in np.flatnonzero(final_mask):
                    final_info = infos["final_info"][index]
                    if final_info is not None and "episode" in final_info:
                        writer.add_scalar(
                            "charts/episodic_return",
                            float(final_info["episode"]["r"]),
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            float(final_info["episode"]["l"]),
                            global_step,
                        )

        if controlled_iteration:
            flat_obs = torch.as_tensor(
                rollout_obs.reshape(-1, obs_dim), device=device
            )
            flat_next_obs = torch.as_tensor(
                rollout_next_obs.reshape(-1, obs_dim), device=device
            )
            factual_goals = torch.as_tensor(
                rollout_goals.reshape(-1, args.latent_dim), device=device
            )
            with torch.no_grad():
                rollout_y = agent.encode(flat_obs)
                rollout_next_y = agent.encode(flat_next_obs)
                values = agent.value_ensemble(rollout_y).mean(dim=0).reshape(
                    args.num_steps, args.num_envs
                )
                next_values = agent.value_ensemble(rollout_next_y).mean(
                    dim=0
                ).reshape(args.num_steps, args.num_envs)
                rewards_tensor = args.reward_scale * torch.as_tensor(
                    rollout_rewards, device=device
                )
                advantages = factual_gae(
                    rewards_tensor,
                    values,
                    next_values,
                    torch.as_tensor(
                        rollout_bootstrap_nonterminal, device=device
                    ),
                    torch.as_tensor(rollout_trace_nonterminal, device=device),
                    args.gamma,
                    args.gae_lambda,
                )
                flat_advantages = advantages.reshape(-1)
                flat_advantages = (
                    flat_advantages - flat_advantages.mean()
                ) / flat_advantages.std(unbiased=False).clamp_min(1e-6)
                factual_deltas = rollout_next_y - rollout_y
                factual_delivery_mse = (
                    factual_deltas - factual_goals
                ).square().mean()
                factual_delta_norm = factual_deltas.norm(dim=-1).mean()

            # Exactly one actor update from the fresh rollout, before any
            # representation, executor, forward, or value parameters move.
            goal_actor_loss, actor_metrics = fresh_actor_loss(
                agent,
                rollout_y.detach(),
                factual_goals,
                flat_advantages,
                args,
            )
            optimizers["actor"].zero_grad(set_to_none=True)
            goal_actor_loss.backward()
            actor_gradient_norm = nn.utils.clip_grad_norm_(
                agent.goal_actor.parameters(), args.max_grad_norm
            )
            optimizers["actor"].step()
            actor_metrics["actor_gradient_norm"] = actor_gradient_norm.detach()
            actor_metrics["factual_delivery_mse"] = factual_delivery_mse
            actor_metrics["rollout_factual_delta_norm"] = factual_delta_norm
            for key, value in actor_metrics.items():
                writer.add_scalar(f"actor/{key}", float(value), global_step)

        if replay.size >= args.minibatch_size:
            accumulated = {}
            for _ in range(args.updates_per_iteration):
                latest_metrics = update_models(
                    agent,
                    sigreg,
                    replay,
                    optimizers,
                    args,
                    rng,
                    device,
                )
                for key, value in latest_metrics.items():
                    accumulated[key] = accumulated.get(key, 0.0) + float(value)
            for key, value in accumulated.items():
                writer.add_scalar(
                    f"losses/{key}"
                    if key.endswith(("loss", "mse"))
                    else f"diagnostics/{key}",
                    value / args.updates_per_iteration,
                    global_step,
                )

        writer.add_scalar(
            "diagnostics/collection_action_saturation",
            collection_action_saturation / args.num_steps,
            global_step,
        )
        if controlled_steps:
            writer.add_scalar(
                "goals/collection_goal_norm",
                collection_goal_norm / controlled_steps,
                global_step,
            )
            writer.add_scalar(
                "goals/collection_goal_std",
                collection_goal_std / controlled_steps,
                global_step,
            )
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if iteration == 1 or iteration % 10 == 0:
            score_loss = (
                float(actor_metrics["goal_score_loss"])
                if controlled_iteration
                else 0.0
            )
            print(
                f"iteration={iteration} step={global_step} "
                f"goal_score_loss={score_loss:.3f} "
                f"SPS={int(global_step / (time.time() - start_time))}"
            )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

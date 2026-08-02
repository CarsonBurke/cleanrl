# Embedding Optimization: Factual Straight-Through Free Goals v2
#
# A compact causal history model produces a Euclidean belief and predicts its
# action-conditioned successor. Deterministic networks propose an unrestricted
# next-belief goal and execute it with a full goal-conditioned action policy.
# On fresh experience, the actor's forward objective is evaluated at the factual
# next belief while its backward derivative follows the frozen learned dynamics.
# A state-only value ensemble supplies long-horizon reward credit. LeWM-style
# prediction and SIGReg train both sides of each factual transition end to end;
# specifically, the factual prediction target remains attached to the encoder.
# There are no policy likelihoods, action-values, target models, moving averages,
# replay goals, directional projections, planning loops, or runtime optimization.
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
    history_length: int = 8
    replay_capacity: int = 131_072
    recent_replay_size: int = 65_536
    warmup_steps: int = 32_768
    minibatch_size: int = 512
    model_updates_per_iteration: int = 16

    latent_dim: int = 32
    hidden_dim: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 4
    predictor_heads: int = 3
    value_heads: int = 5
    bootstrap_probability: float = 0.8

    representation_lr: float = 1e-4
    value_lr: float = 3e-4
    reward_lr: float = 3e-4
    actor_lr: float = 1e-4
    weight_decay: float = 1e-4
    sigreg_coef: float = 0.09
    reward_scale: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    pessimism_coef: float = 1.0
    goal_delivery_coef: float = 0.25
    action_cost_coef: float = 0.002
    uncertainty_coef: float = 0.10
    excitation_amplitude: float = 0.12
    excitation_period: int = 257
    goal_excitation_amplitude: float = 0.15
    goal_excitation_period: int = 379
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
    """Restore terminal states hidden by vector-environment autoreset."""
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


def initialize_histories(observations, history_length, action_dim):
    observation_history = np.repeat(
        np.asarray(observations, dtype=np.float32)[:, None, :],
        history_length,
        axis=1,
    )
    action_history = np.zeros(
        (observations.shape[0], history_length, action_dim), dtype=np.float32
    )
    return observation_history, action_history


def transition_next_histories(
    observation_history, action_history, actions, next_observations
):
    """Construct the factual successor history for a transition."""
    next_observation_history = np.concatenate(
        [
            observation_history[:, 1:],
            np.asarray(next_observations, dtype=np.float32)[:, None, :],
        ],
        axis=1,
    )
    next_action_history = np.concatenate(
        [
            action_history[:, 1:],
            np.asarray(actions, dtype=np.float32)[:, None, :],
        ],
        axis=1,
    )
    return next_observation_history, next_action_history


def online_successor_histories(
    factual_observation_history,
    factual_action_history,
    autoreset_observations,
    done,
):
    """Keep factual terminal histories in replay but reset live histories."""
    live_observations = np.array(factual_observation_history, copy=True)
    live_actions = np.array(factual_action_history, copy=True)
    if np.any(done):
        reset_observations, reset_actions = initialize_histories(
            np.asarray(autoreset_observations)[done],
            factual_observation_history.shape[1],
            factual_action_history.shape[-1],
        )
        live_observations[done] = reset_observations
        live_actions[done] = reset_actions
    return live_observations, live_actions


def deterministic_action_excitation(
    transition_step,
    num_envs,
    action_dim,
    amplitude,
    period,
    device=None,
):
    """Fixed structured excitation, not policy or goal sampling."""
    dtype = torch.float32
    env_phase = (
        2
        * math.pi
        * torch.arange(num_envs, device=device, dtype=dtype)
        / max(1, num_envs)
    )
    # Distinct harmonics make the time-stacked probes full rank. Phase-shifted
    # copies of one frequency span at most sine and cosine, leaving most of a
    # multi-dimensional action Jacobian unidentified.
    frequencies = torch.arange(
        1, action_dim + 1, device=device, dtype=dtype
    )
    time_phase = 2 * math.pi * transition_step / period
    return amplitude * torch.sin(
        time_phase * frequencies[None, :] + env_phase[:, None]
    )


def deterministic_goal_excitation(
    transition_step,
    num_envs,
    latent_dim,
    amplitude,
    period,
    device=None,
):
    """Fixed goal-space probes identify how the policy uses its goal input."""
    dtype = torch.float32
    env_phase = (
        2
        * math.pi
        * torch.arange(num_envs, device=device, dtype=dtype)
        / max(1, num_envs)
    )
    frequencies = torch.arange(
        1, latent_dim + 1, device=device, dtype=dtype
    )
    time_phase = 2 * math.pi * transition_step / period
    return amplitude * torch.sin(
        time_phase * frequencies[None, :] + env_phase[:, None]
    )


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


class CausalHistoryWorldModel(nn.Module):
    """Lightweight causal belief model with action-conditioned prediction."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_dim,
        hidden_dim,
        history_length,
        transformer_layers,
        transformer_heads,
        predictor_heads,
    ):
        super().__init__()
        self.observation_encoder = mlp(
            obs_dim, hidden_dim, latent_dim, layer_norm=True
        )
        self.action_token = layer_init(
            nn.Linear(action_dim, latent_dim), std=0.1
        )
        self.position = nn.Parameter(
            torch.zeros(1, history_length, latent_dim)
        )
        nn.init.normal_(self.position, std=0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=2 * latent_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        # SIGReg must act on an unconstrained output. The post-normalization
        # projector follows LeWM's design and prevents the final LayerNorm from
        # fixing the per-sample statistics that SIGReg is meant to learn.
        self.belief_projector = mlp(
            latent_dim,
            2 * latent_dim,
            latent_dim,
            out_std=1.0,
        )
        self.predictors = nn.ModuleList(
            [
                mlp(
                    latent_dim + action_dim,
                    hidden_dim,
                    latent_dim,
                    out_std=0.01,
                    layer_norm=True,
                )
                for _ in range(predictor_heads)
            ]
        )
        self.register_buffer("alignment", torch.eye(latent_dim))

    def causal_mask(self, length, device):
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def belief_sequence_raw(self, observation_history, action_history):
        tokens = (
            self.observation_encoder(observation_history)
            + self.action_token(action_history)
            + self.position[:, : observation_history.shape[1]]
        )
        hidden = self.transformer(
            tokens,
            mask=self.causal_mask(tokens.shape[1], tokens.device),
        )
        return self.belief_projector(self.output_norm(hidden))

    def belief_raw(self, observation_history, action_history):
        return self.belief_sequence_raw(
            observation_history, action_history
        )[:, -1]

    def belief(self, observation_history, action_history):
        return self.belief_raw(observation_history, action_history) @ self.alignment

    def predict_next(self, belief, action):
        features = torch.cat([belief, action], dim=-1)
        return torch.stack(
            [belief + predictor(features) for predictor in self.predictors],
            dim=0,
        )


class FreeGoalNetwork(nn.Module):
    """An unrestricted deterministic Euclidean goal."""

    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.network = mlp(latent_dim, hidden_dim, latent_dim, out_std=0.01)

    def forward(self, belief):
        return self.network(belief)


class GoalConditionedPolicy(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.network = mlp(
            3 * latent_dim, hidden_dim, action_dim, out_std=0.01
        )

    def raw_action(self, belief, goal):
        features = torch.cat([belief, goal, goal - belief], dim=-1)
        return self.network(features)

    def forward(self, belief, goal, excitation=None):
        raw_action = self.raw_action(belief, goal)
        if excitation is not None:
            raw_action = raw_action + excitation
        return torch.tanh(raw_action)


class StateValueEnsemble(nn.Module):
    def __init__(self, latent_dim, hidden_dim, value_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                mlp(latent_dim, hidden_dim, 1, out_std=0.01)
                for _ in range(value_heads)
            ]
        )

    def forward(self, belief):
        return torch.stack(
            [head(belief).squeeze(-1) for head in self.heads], dim=0
        )


class TransitionRewardEnsemble(nn.Module):
    """Factual reward as a function only of state and state difference."""

    def __init__(self, latent_dim, hidden_dim, reward_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                mlp(3 * latent_dim, hidden_dim, 1, out_std=0.01)
                for _ in range(reward_heads)
            ]
        )

    def forward(self, belief, next_belief):
        features = torch.cat(
            [belief, next_belief, next_belief - belief], dim=-1
        )
        return torch.stack(
            [head(features).squeeze(-1) for head in self.heads], dim=0
        )


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, args):
        super().__init__()
        self.world = CausalHistoryWorldModel(
            obs_dim,
            action_dim,
            args.latent_dim,
            args.hidden_dim,
            args.history_length,
            args.transformer_layers,
            args.transformer_heads,
            args.predictor_heads,
        )
        self.goal = FreeGoalNetwork(args.latent_dim, args.hidden_dim)
        self.policy = GoalConditionedPolicy(
            args.latent_dim, action_dim, args.hidden_dim
        )
        self.value = StateValueEnsemble(
            args.latent_dim, args.hidden_dim, args.value_heads
        )
        self.reward = TransitionRewardEnsemble(
            args.latent_dim, args.hidden_dim, args.value_heads
        )

    def act(
        self,
        observation_history,
        action_history,
        action_excitation=None,
        goal_excitation=None,
    ):
        belief = self.world.belief(observation_history, action_history)
        base_goal = self.goal(belief)
        commanded_goal = (
            base_goal
            if goal_excitation is None
            else base_goal + goal_excitation
        )
        action = self.policy(
            belief, commanded_goal, action_excitation
        )
        return action, commanded_goal, belief


class HistoryReplay:
    def __init__(self, capacity, history_length, obs_dim, action_dim):
        self.capacity = capacity
        self.observation_history = np.empty(
            (capacity, history_length, obs_dim), dtype=np.float32
        )
        self.action_history = np.empty(
            (capacity, history_length, action_dim), dtype=np.float32
        )
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.pointer = 0
        self.size = 0

    def add(
        self,
        observation_history,
        action_history,
        actions,
        next_observations,
        rewards,
        terminated,
    ):
        count = observation_history.shape[0]
        slots = (np.arange(count) + self.pointer) % self.capacity
        self.observation_history[slots] = observation_history
        self.action_history[slots] = action_history
        self.actions[slots] = actions
        self.next_observations[slots] = next_observations
        self.rewards[slots] = rewards
        self.terminated[slots] = terminated
        self.pointer = (self.pointer + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample_recent_indices(self, count, recent_size, rng):
        available = min(self.size, recent_size)
        if available == 0:
            raise ValueError("cannot sample empty replay")
        offsets = rng.integers(0, available, size=count)
        return (self.pointer - 1 - offsets) % self.capacity


@contextlib.contextmanager
def frozen_parameters(*modules):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous):
            parameter.requires_grad_(requires_grad)


def straight_through_actual(predicted_next, actual_next):
    """Factual forward value with the predictor's local backward derivative."""
    return predicted_next + (actual_next - predicted_next).detach()


def pessimistic_value(values, coefficient):
    return values.mean(dim=0) - coefficient * values.std(
        dim=0, unbiased=False
    )


def procrustes_alignment(raw_new, global_old):
    u, _, vh = torch.linalg.svd(raw_new.T @ global_old, full_matrices=False)
    return u @ vh


def prediction_mse_with_live_target(predicted_next, actual_next):
    """Both the predictor and factual embedding receive representation gradients."""
    target = actual_next.unsqueeze(0).expand_as(predicted_next)
    return (predicted_next - target).square().mean()


def representation_objective(
    predicted_next, actual_next, sigreg_loss, sigreg_coef
):
    prediction_loss = prediction_mse_with_live_target(
        predicted_next, actual_next
    )
    return prediction_loss + sigreg_coef * sigreg_loss, prediction_loss


def factual_gae(
    rewards,
    values,
    next_values,
    bootstrap_nonterminal,
    trace_nonterminal,
    gamma,
    gae_lambda,
):
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


def factual_st_actor_loss(
    agent,
    beliefs,
    actual_next_beliefs,
    action_excitation,
    goal_excitation,
    args,
):
    """One fresh-rollout actor objective with factual forward semantics."""
    beliefs = beliefs.detach()
    actual_next_beliefs = actual_next_beliefs.detach()
    base_goal = agent.goal(beliefs)
    commanded_goal = base_goal + goal_excitation
    raw_action = agent.policy.raw_action(beliefs, commanded_goal)
    base_action = torch.tanh(raw_action)
    action = torch.tanh(raw_action + action_excitation)
    with frozen_parameters(agent.world, agent.reward, agent.value):
        predicted_next_heads = agent.world.predict_next(beliefs, action)
        predicted_next = predicted_next_heads.mean(dim=0)
        training_next = straight_through_actual(
            predicted_next, actual_next_beliefs
        )
        reward_heads = agent.reward(beliefs, training_next)
        next_value_heads = agent.value(training_next)
        factual_return = pessimistic_value(
            reward_heads + args.gamma * next_value_heads,
            args.pessimism_coef,
        )

        # A separate route detaches the goal before the policy. Delivery,
        # uncertainty, and action cost therefore shape pi but cannot update G.
        delivery_action = agent.policy(
            beliefs, commanded_goal.detach(), action_excitation
        )
        delivery_predictions = agent.world.predict_next(
            beliefs, delivery_action
        )
        delivery_prediction = delivery_predictions.mean(dim=0)
        delivery_training_next = straight_through_actual(
            delivery_prediction, actual_next_beliefs
        )
        delivery_mse = (
            delivery_training_next - commanded_goal.detach()
        ).square().mean(dim=-1)
        uncertainty = delivery_predictions.var(
            dim=0, unbiased=False
        ).mean(dim=-1)
        action_cost = delivery_action.square().mean(dim=-1)
        objective = (
            factual_return
            - args.goal_delivery_coef * delivery_mse
            - args.action_cost_coef * action_cost
            - args.uncertainty_coef * uncertainty
        )

    with torch.no_grad():
        zero_goal_action = agent.policy(
            beliefs,
            torch.zeros_like(commanded_goal),
            action_excitation,
        )
        action_probe = (action + 0.01).clamp(-1, 1)
        probe_prediction = agent.world.predict_next(
            beliefs, action_probe
        ).mean(dim=0)
        predictor_action_sensitivity = (
            (probe_prediction - predicted_next).norm(dim=-1) / 0.01
        ).mean()

    metrics = {
        "actor_objective": objective.mean().detach(),
        "actual_next_value": next_value_heads.mean().detach(),
        "factual_forward_reward": reward_heads.mean().detach(),
        "factual_forward_return": factual_return.mean().detach(),
        "actual_goal_mse": (
            actual_next_beliefs - commanded_goal
        ).square().mean().detach(),
        "predicted_goal_mse": (
            predicted_next - commanded_goal
        ).square().mean().detach(),
        "model_uncertainty": uncertainty.mean().detach(),
        "action_cost": action_cost.mean().detach(),
        "action_excitation_norm": (
            action_excitation.norm(dim=-1).mean().detach()
        ),
        "goal_excitation_norm": (
            goal_excitation.norm(dim=-1).mean().detach()
        ),
        "base_executed_action_mse": (
            base_action - action
        ).square().mean().detach(),
        "predictor_action_sensitivity": predictor_action_sensitivity.detach(),
        "goal_action_influence": (
            action - zero_goal_action
        ).norm(dim=-1).mean().detach(),
        "base_goal_norm": base_goal.norm(dim=-1).mean().detach(),
        "commanded_goal_norm": commanded_goal.norm(dim=-1).mean().detach(),
        "actual_next_norm": actual_next_beliefs.norm(dim=-1).mean().detach(),
        "predicted_next_norm": predicted_next.norm(dim=-1).mean().detach(),
        "action_saturation": (action.abs() > 0.95).float().mean().detach(),
    }
    return -objective.mean(), metrics, action


def bootstrap_ensemble_mse(predictions, target, probability):
    """Independent factual bootstrap masks preserve useful ensemble diversity."""
    per_item = (predictions - target.unsqueeze(0)).square()
    while per_item.ndim > 2:
        per_item = per_item.mean(dim=-1)
    mask = (
        torch.rand_like(per_item) < probability
    ).to(per_item.dtype)
    return (per_item * mask).sum() / mask.sum().clamp_min(1.0)


def replay_batch(replay, args, rng, device):
    indices = replay.sample_recent_indices(
        args.minibatch_size, args.recent_replay_size, rng
    )
    observation_history = torch.as_tensor(
        replay.observation_history[indices], device=device
    )
    action_history = torch.as_tensor(
        replay.action_history[indices], device=device
    )
    actions = torch.as_tensor(replay.actions[indices], device=device)
    next_observations = np.array(
        replay.observation_history[indices, 1:], copy=True
    )
    next_observations = np.concatenate(
        [
            next_observations,
            replay.next_observations[indices, None, :],
        ],
        axis=1,
    )
    next_actions = np.concatenate(
        [
            replay.action_history[indices, 1:],
            replay.actions[indices, None, :],
        ],
        axis=1,
    )
    return (
        observation_history,
        action_history,
        actions,
        torch.as_tensor(next_observations, device=device),
        torch.as_tensor(next_actions, device=device),
    )


def update_representation(
    agent,
    sigreg,
    replay,
    optimizer,
    args,
    rng,
    device,
):
    (
        observation_history,
        action_history,
        actions,
        next_observation_history,
        next_action_history,
    ) = replay_batch(replay, args, rng, device)
    with torch.no_grad():
        old_global = agent.world.belief(
            observation_history, action_history
        )

    belief = agent.world.belief(observation_history, action_history)
    actual_next = agent.world.belief(
        next_observation_history, next_action_history
    )
    predicted_next = agent.world.predict_next(belief, actions)
    sigreg_loss = 0.5 * (sigreg(belief) + sigreg(actual_next))
    loss, prediction_loss = representation_objective(
        predicted_next, actual_next, sigreg_loss, args.sigreg_coef
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        agent.world.parameters(), args.max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        new_raw = agent.world.belief_raw(
            observation_history, action_history
        )
        alignment = procrustes_alignment(new_raw, old_global)
        chart_residual = (
            new_raw @ alignment - old_global
        ).square().mean().sqrt()
        agent.world.alignment.copy_(alignment)

    return {
        "representation_loss": loss.detach(),
        "prediction_mse": prediction_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
        "representation_gradient_norm": gradient_norm.detach(),
        "chart_residual": chart_residual.detach(),
    }


def main():
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("factual straight-through free goals require CUDA")
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
    representation_optimizer = optim.AdamW(
        agent.world.parameters(),
        lr=args.representation_lr,
        weight_decay=args.weight_decay,
    )
    value_optimizer = optim.AdamW(
        agent.value.parameters(),
        lr=args.value_lr,
        weight_decay=args.weight_decay,
    )
    reward_optimizer = optim.AdamW(
        agent.reward.parameters(),
        lr=args.reward_lr,
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(agent.goal.parameters()) + list(agent.policy.parameters()),
        lr=args.actor_lr,
        weight_decay=args.weight_decay,
    )

    act = agent.act
    if args.compile:
        act = torch.compile(act, mode=args.compile_mode)

    replay = HistoryReplay(
        args.replay_capacity,
        args.history_length,
        obs_dim,
        action_dim,
    )
    observations, _ = envs.reset(seed=args.seed)
    observation_history, action_history = initialize_histories(
        observations, args.history_length, action_dim
    )
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        rollout_observation_history = np.empty(
            (
                args.num_steps,
                args.num_envs,
                args.history_length,
                obs_dim,
            ),
            dtype=np.float32,
        )
        rollout_action_history = np.empty(
            (
                args.num_steps,
                args.num_envs,
                args.history_length,
                action_dim,
            ),
            dtype=np.float32,
        )
        rollout_next_observation_history = np.empty_like(
            rollout_observation_history
        )
        rollout_next_action_history = np.empty_like(rollout_action_history)
        rollout_actions = np.empty(
            (args.num_steps, args.num_envs, action_dim), dtype=np.float32
        )
        rollout_excitations = np.empty_like(rollout_actions)
        rollout_goal_excitations = np.empty(
            (
                args.num_steps,
                args.num_envs,
                args.latent_dim,
            ),
            dtype=np.float32,
        )
        rollout_rewards = np.empty(
            (args.num_steps, args.num_envs), dtype=np.float32
        )
        rollout_bootstrap = np.empty_like(rollout_rewards)
        rollout_trace = np.empty_like(rollout_rewards)
        controlled_iteration = global_step >= args.warmup_steps
        collection_saturation = 0.0
        collection_goal_norm = 0.0

        for step in range(args.num_steps):
            rollout_observation_history[step] = observation_history
            rollout_action_history[step] = action_history
            if controlled_iteration:
                observation_tensor = torch.as_tensor(
                    observation_history, dtype=torch.float32, device=device
                )
                action_tensor = torch.as_tensor(
                    action_history, dtype=torch.float32, device=device
                )
                excitation = deterministic_action_excitation(
                    global_step // args.num_envs,
                    args.num_envs,
                    action_dim,
                    args.excitation_amplitude,
                    args.excitation_period,
                    device,
                )
                goal_excitation = deterministic_goal_excitation(
                    global_step // args.num_envs,
                    args.num_envs,
                    args.latent_dim,
                    args.goal_excitation_amplitude,
                    args.goal_excitation_period,
                    device,
                )
                with torch.no_grad():
                    chosen_action, goal, _ = act(
                        observation_tensor,
                        action_tensor,
                        excitation,
                        goal_excitation,
                    )
                actions = chosen_action.cpu().numpy()
                rollout_excitations[step] = excitation.cpu().numpy()
                rollout_goal_excitations[step] = (
                    goal_excitation.cpu().numpy()
                )
                collection_goal_norm += goal.norm(dim=-1).mean().item()
            else:
                actions = np.stack(
                    [
                        envs.single_action_space.sample()
                        for _ in range(args.num_envs)
                    ]
                ).astype(np.float32)
                rollout_excitations[step].fill(0)
                rollout_goal_excitations[step].fill(0)

            next_observations, rewards, terminations, truncations, infos = envs.step(
                actions
            )
            factual_next = factual_next_observations(next_observations, infos)
            (
                factual_observation_history,
                factual_action_history,
            ) = transition_next_histories(
                observation_history,
                action_history,
                actions,
                factual_next,
            )
            done = np.logical_or(terminations, truncations)
            (
                live_observation_history,
                live_action_history,
            ) = online_successor_histories(
                factual_observation_history,
                factual_action_history,
                next_observations,
                done,
            )

            rollout_next_observation_history[step] = factual_observation_history
            rollout_next_action_history[step] = factual_action_history
            rollout_actions[step] = actions
            rollout_rewards[step] = rewards
            rollout_bootstrap[step] = 1.0 - terminations.astype(np.float32)
            rollout_trace[step] = 1.0 - done.astype(np.float32)
            replay.add(
                observation_history,
                action_history,
                actions,
                factual_next,
                rewards,
                terminations.astype(np.float32),
            )

            collection_saturation += float((np.abs(actions) > 0.95).mean())
            global_step += args.num_envs
            observations = next_observations
            observation_history = live_observation_history
            action_history = live_action_history

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

        flat_observation_history = torch.as_tensor(
            rollout_observation_history.reshape(
                -1, args.history_length, obs_dim
            ),
            device=device,
        )
        flat_action_history = torch.as_tensor(
            rollout_action_history.reshape(
                -1, args.history_length, action_dim
            ),
            device=device,
        )
        flat_next_observation_history = torch.as_tensor(
            rollout_next_observation_history.reshape(
                -1, args.history_length, obs_dim
            ),
            device=device,
        )
        flat_next_action_history = torch.as_tensor(
            rollout_next_action_history.reshape(
                -1, args.history_length, action_dim
            ),
            device=device,
        )
        with torch.no_grad():
            rollout_beliefs = agent.world.belief(
                flat_observation_history, flat_action_history
            )
            rollout_next_beliefs = agent.world.belief(
                flat_next_observation_history, flat_next_action_history
            )
            values = agent.value(rollout_beliefs).mean(dim=0).reshape(
                args.num_steps, args.num_envs
            )
            next_values = agent.value(rollout_next_beliefs).mean(dim=0).reshape(
                args.num_steps, args.num_envs
            )
            advantages = factual_gae(
                args.reward_scale
                * torch.as_tensor(rollout_rewards, device=device),
                values,
                next_values,
                torch.as_tensor(rollout_bootstrap, device=device),
                torch.as_tensor(rollout_trace, device=device),
                args.gamma,
                args.gae_lambda,
            )
            factual_returns = (advantages + values).reshape(-1)

        if controlled_iteration:
            # Exactly one update from fresh facts, before world/value movement.
            actor_loss, actor_metrics, recomputed_actions = factual_st_actor_loss(
                agent,
                rollout_beliefs,
                rollout_next_beliefs,
                torch.as_tensor(
                    rollout_excitations.reshape(-1, action_dim), device=device
                ),
                torch.as_tensor(
                    rollout_goal_excitations.reshape(
                        -1, args.latent_dim
                    ),
                    device=device,
                ),
                args,
            )
            action_recompute_mse = (
                recomputed_actions
                - torch.as_tensor(
                    rollout_actions.reshape(-1, action_dim), device=device
                )
            ).square().mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = nn.utils.clip_grad_norm_(
                list(agent.goal.parameters()) + list(agent.policy.parameters()),
                args.max_grad_norm,
            )
            actor_optimizer.step()
            actor_metrics["action_recompute_mse"] = action_recompute_mse.detach()
            actor_metrics["actor_gradient_norm"] = actor_gradient_norm.detach()
            for key, value in actor_metrics.items():
                writer.add_scalar(f"actor/{key}", float(value), global_step)

        value_predictions = agent.value(rollout_beliefs.detach())
        value_loss = bootstrap_ensemble_mse(
            value_predictions,
            factual_returns.detach(),
            args.bootstrap_probability,
        )
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        value_gradient_norm = nn.utils.clip_grad_norm_(
            agent.value.parameters(), args.max_grad_norm
        )
        value_optimizer.step()
        value_optimizer.zero_grad(set_to_none=True)
        writer.add_scalar("value/factual_gae_mse", value_loss.item(), global_step)
        writer.add_scalar(
            "value/gradient_norm", float(value_gradient_norm), global_step
        )
        writer.add_scalar(
            "value/factual_advantage_mean", advantages.mean().item(), global_step
        )
        writer.add_scalar(
            "value/factual_advantage_std",
            advantages.std(unbiased=False).item(),
            global_step,
        )

        factual_reward_predictions = agent.reward(
            rollout_beliefs.detach(), rollout_next_beliefs.detach()
        )
        factual_reward_targets = args.reward_scale * torch.as_tensor(
            rollout_rewards.reshape(-1), device=device
        )
        reward_loss = bootstrap_ensemble_mse(
            factual_reward_predictions,
            factual_reward_targets,
            args.bootstrap_probability,
        )
        reward_optimizer.zero_grad(set_to_none=True)
        reward_loss.backward()
        reward_gradient_norm = nn.utils.clip_grad_norm_(
            agent.reward.parameters(), args.max_grad_norm
        )
        reward_optimizer.step()
        reward_optimizer.zero_grad(set_to_none=True)
        writer.add_scalar(
            "reward/factual_transition_mse", reward_loss.item(), global_step
        )
        writer.add_scalar(
            "reward/gradient_norm", float(reward_gradient_norm), global_step
        )

        if replay.size >= args.minibatch_size:
            totals = {}
            for _ in range(args.model_updates_per_iteration):
                model_metrics = update_representation(
                    agent,
                    sigreg,
                    replay,
                    representation_optimizer,
                    args,
                    rng,
                    device,
                )
                for key, value in model_metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
            for key, value in totals.items():
                writer.add_scalar(
                    f"model/{key}",
                    value / args.model_updates_per_iteration,
                    global_step,
                )

        writer.add_scalar(
            "diagnostics/collection_action_saturation",
            collection_saturation / args.num_steps,
            global_step,
        )
        if controlled_iteration:
            writer.add_scalar(
                "goals/collection_goal_norm",
                collection_goal_norm / args.num_steps,
                global_step,
            )
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        if iteration == 1 or iteration % 10 == 0:
            print(
                f"iteration={iteration} step={global_step} "
                f"controlled={controlled_iteration} "
                f"SPS={int(global_step / (time.time() - start_time))}"
            )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

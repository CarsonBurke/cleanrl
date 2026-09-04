# Direct-Outcome Sampled MuZero v1
#
# A single live network learns only action-conditioned rewards, true termination,
# finite-horizon returns, and search-improved action distributions.  Full recurrent
# BPTT and Predictron return composition train the latent state without reconstruction,
# target networks, teachers, or PPO.  Fixed-shape batched MCTS turns additional
# inference compute into the behavior policy; benchmark claims await measurement.

import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter


LOG_2PI = math.log(2.0 * math.pi)


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
    save_model: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 16
    learning_rate: float = 3e-4
    policy_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    unroll_steps: int = 16
    latent_dim: int = 512
    bottleneck_dim: int = 256
    representation_depth: int = 4
    dynamics_depth: int = 4
    prediction_depth: int = 3
    max_grad_norm: float = 0.5

    root_candidates: int = 32
    child_candidates: int = 8
    simulations: int = 256
    simulation_wave: int = 16
    search_depth: int = 16
    pb_c_init: float = 1.25
    pb_c_base: float = 19_652.0
    min_log_std: float = -5.0
    max_log_std: float = 2.0

    compile: bool = False
    compile_mode: str = "reduce-overhead"
    log_frequency: int = 10_000
    tail_window: int = 512


def make_env(env_id: str, index: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and index == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def factual_next_observations(
    autoreset_observations: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    infos: dict[str, Any],
) -> np.ndarray:
    """Recover post-action observations before vector-environment autoreset."""
    result = np.array(autoreset_observations, copy=True)
    done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
    if not np.any(done):
        return result
    final = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final is None:
        raise RuntimeError("finished transition is missing infos['final_observation']")
    for env_index in np.flatnonzero(done):
        if final_mask is not None and not bool(final_mask[env_index]):
            raise RuntimeError(f"environment {env_index} is missing its final observation")
        if final[env_index] is None:
            raise RuntimeError(f"environment {env_index} has a null final observation")
        result[env_index] = np.asarray(final[env_index], dtype=result.dtype)
    return result


def map_normalized_action(
    normalized_action: torch.Tensor | np.ndarray,
    low: torch.Tensor | np.ndarray,
    high: torch.Tensor | np.ndarray,
):
    return low + (normalized_action + 1.0) * 0.5 * (high - low)


def shifted_tanh_log_std(
    raw: torch.Tensor,
    minimum: float = -5.0,
    maximum: float = 2.0,
) -> torch.Tensor:
    """Smooth bounded log standard deviation whose zero input maps exactly to zero."""
    if not minimum < 0.0 < maximum:
        raise ValueError("log-standard-deviation bounds must straddle zero")
    midpoint = 0.5 * (minimum + maximum)
    half_range = 0.5 * (maximum - minimum)
    shift = math.atanh(-midpoint / half_range)
    return half_range * (torch.tanh(raw + shift) - math.tanh(shift))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.orthogonal_(self.down.weight, math.sqrt(2.0))
        nn.init.zeros_(self.down.bias)
        nn.init.orthogonal_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.rms_norm(inputs, (inputs.shape[-1],))
        hidden = F.silu(self.down(hidden))
        return inputs + self.up(hidden) / math.sqrt(2.0)


class ResidualTower(nn.Module):
    def __init__(self, input_dim: int, width: int, bottleneck: int, depth: int):
        super().__init__()
        self.stem = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([ResidualBlock(width, bottleneck) for _ in range(depth)])
        nn.init.orthogonal_(self.stem.weight, math.sqrt(2.0))
        nn.init.zeros_(self.stem.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.stem(inputs))
        for block in self.blocks:
            hidden = block(hidden)
        return F.rms_norm(hidden, (hidden.shape[-1],))


class InitialInference(NamedTuple):
    latent: torch.Tensor
    value_rate: torch.Tensor
    policy_mean: torch.Tensor
    policy_log_std: torch.Tensor


class RecurrentInference(NamedTuple):
    latent: torch.Tensor
    reward: torch.Tensor
    termination_logit: torch.Tensor
    value_rate: torch.Tensor
    policy_mean: torch.Tensor
    policy_log_std: torch.Tensor


class DirectOutcomeMuZero(nn.Module):
    """One recurrent model, grounded exclusively by factual task outcomes."""

    def __init__(self, obs_dim: int, action_dim: int, args: Args):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = args.latent_dim
        self.min_log_std = args.min_log_std
        self.max_log_std = args.max_log_std

        self.representation = ResidualTower(
            obs_dim + 1,
            args.latent_dim,
            args.bottleneck_dim,
            args.representation_depth,
        )
        self.dynamics = ResidualTower(
            args.latent_dim + action_dim + 1,
            args.latent_dim,
            args.bottleneck_dim,
            args.dynamics_depth,
        )
        self.latent_delta = nn.Linear(args.latent_dim, args.latent_dim)
        self.reward_head = nn.Linear(args.latent_dim, 1)
        self.termination_head = nn.Linear(args.latent_dim, 1)

        self.prediction = ResidualTower(
            args.latent_dim + 1,
            args.latent_dim,
            args.bottleneck_dim,
            args.prediction_depth,
        )
        self.value_head = nn.Linear(args.latent_dim, 1)
        self.policy_mean_head = nn.Linear(args.latent_dim, action_dim)
        self.policy_log_std_head = nn.Linear(args.latent_dim, action_dim)

        # Exchangeability before evidence: all outcome and policy predictions are exact zero.
        nn.init.orthogonal_(self.latent_delta.weight)
        nn.init.zeros_(self.latent_delta.bias)
        for head in (
            self.reward_head,
            self.termination_head,
            self.value_head,
            self.policy_mean_head,
            self.policy_log_std_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def encode(self, obs: torch.Tensor, remaining_rate: torch.Tensor) -> torch.Tensor:
        return self.representation(torch.cat([obs, remaining_rate.unsqueeze(-1)], dim=-1))

    def predict(
        self, latent: torch.Tensor, remaining_rate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.prediction(torch.cat([latent, remaining_rate.unsqueeze(-1)], dim=-1))
        value_rate = self.value_head(hidden).squeeze(-1)
        policy_mean = self.policy_mean_head(hidden)
        raw_log_std = self.policy_log_std_head(hidden)
        policy_log_std = shifted_tanh_log_std(
            raw_log_std, self.min_log_std, self.max_log_std
        )
        return value_rate, policy_mean, policy_log_std

    def initial_inference(
        self, obs: torch.Tensor, remaining_rate: torch.Tensor
    ) -> InitialInference:
        latent = self.encode(obs, remaining_rate)
        value_rate, policy_mean, policy_log_std = self.predict(latent, remaining_rate)
        return InitialInference(latent, value_rate, policy_mean, policy_log_std)

    def recurrent_inference(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        next_remaining_rate: torch.Tensor,
    ) -> RecurrentInference:
        hidden = self.dynamics(
            torch.cat([latent, action, next_remaining_rate.unsqueeze(-1)], dim=-1)
        )
        next_latent = F.rms_norm(
            latent + self.latent_delta(hidden) / math.sqrt(2.0),
            (latent.shape[-1],),
        )
        reward = self.reward_head(hidden).squeeze(-1)
        termination_logit = self.termination_head(hidden).squeeze(-1)
        value_rate, policy_mean, policy_log_std = self.predict(
            next_latent, next_remaining_rate
        )
        return RecurrentInference(
            next_latent,
            reward,
            termination_logit,
            value_rate,
            policy_mean,
            policy_log_std,
        )


class Episode(NamedTuple):
    observations: np.ndarray
    remaining: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    returns: np.ndarray
    root_candidates: np.ndarray
    root_weights: np.ndarray
    predicted_scores: np.ndarray
    generation_id: int


class SynchronousGenerationReplay:
    """One immutable on-policy search generation with atomic complete episodes."""

    EMPTY = "empty"
    COLLECT = "collect"
    TRAIN = "train"

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        root_candidates: int,
        max_episode_steps: int,
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.candidates = root_candidates
        self.max_episode_steps = max_episode_steps
        self.phase = self.EMPTY
        self.generation_id = -1
        self.vector_steps = 0
        self.active = np.zeros(num_envs, dtype=bool)
        self.episodes: list[Episode] = []
        self.pending: list[dict[str, list[Any]]] = []
        self.tensors: dict[str, torch.Tensor] = {}

    @property
    def completed_transitions(self) -> int:
        return sum(len(episode.actions) for episode in self.episodes)

    @property
    def ready(self) -> bool:
        return (
            self.phase == self.COLLECT
            and not np.any(self.active)
            and len(self.episodes) == self.num_envs
            and all(not fields["actions"] for fields in self.pending)
        )

    def begin(self, generation_id: int) -> None:
        if self.phase != self.EMPTY or self.episodes or self.tensors:
            raise RuntimeError("generation replay must be empty before collection")
        self.phase = self.COLLECT
        self.generation_id = int(generation_id)
        self.vector_steps = 0
        self.active[:] = True
        self.pending = [
            {
                "observations": [],
                "remaining": [],
                "actions": [],
                "rewards": [],
                "terminated": [],
                "root_candidates": [],
                "root_weights": [],
                "predicted_scores": [],
            }
            for _ in range(self.num_envs)
        ]

    def add_batch(
        self,
        observations: np.ndarray,
        remaining: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        factual_next: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        root_candidates: np.ndarray,
        root_weights: np.ndarray,
        predicted_scores: np.ndarray,
    ) -> None:
        if self.phase != self.COLLECT:
            raise RuntimeError("transitions can only be added during collection")
        expected = {
            "observations": (self.num_envs, self.obs_dim),
            "remaining": (self.num_envs,),
            "actions": (self.num_envs, self.action_dim),
            "root_candidates": (self.num_envs, self.candidates, self.action_dim),
            "root_weights": (self.num_envs, self.candidates),
            "predicted_scores": (self.num_envs,),
        }
        actual = {
            "observations": np.shape(observations),
            "remaining": np.shape(remaining),
            "actions": np.shape(actions),
            "root_candidates": np.shape(root_candidates),
            "root_weights": np.shape(root_weights),
            "predicted_scores": np.shape(predicted_scores),
        }
        if actual != expected:
            raise ValueError(f"generation batch shapes are {actual}, expected {expected}")

        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        active_before = self.active.copy()
        for env_index in range(self.num_envs):
            if not active_before[env_index]:
                continue
            fields = self.pending[env_index]
            if not fields["observations"]:
                fields["observations"].append(
                    np.asarray(observations[env_index], dtype=np.float32).copy()
                )
                fields["remaining"].append(float(remaining[env_index]))
            fields["actions"].append(np.asarray(actions[env_index], dtype=np.float32).copy())
            fields["rewards"].append(float(rewards[env_index]))
            fields["terminated"].append(float(bool(terminated[env_index])))
            fields["root_candidates"].append(
                np.asarray(root_candidates[env_index], dtype=np.float32).copy()
            )
            fields["root_weights"].append(
                np.asarray(root_weights[env_index], dtype=np.float32).copy()
            )
            fields["predicted_scores"].append(float(predicted_scores[env_index]))
            fields["observations"].append(
                np.asarray(factual_next[env_index], dtype=np.float32).copy()
            )
            fields["remaining"].append(max(float(remaining[env_index]) - 1.0, 0.0))
            if done[env_index]:
                self._finalize(env_index)
                self.active[env_index] = False
        self.vector_steps += 1

    def _finalize(self, env_index: int) -> None:
        fields = self.pending[env_index]
        length = len(fields["actions"])
        if length == 0:
            raise RuntimeError("cannot finalize an empty episode")
        rewards = np.asarray(fields["rewards"], dtype=np.float32)
        returns = np.empty(length + 1, dtype=np.float32)
        returns[length] = 0.0
        for index in range(length - 1, -1, -1):
            returns[index] = rewards[index] + returns[index + 1]
        episode = Episode(
            observations=np.asarray(fields["observations"], dtype=np.float32),
            remaining=np.asarray(fields["remaining"], dtype=np.float32),
            actions=np.asarray(fields["actions"], dtype=np.float32),
            rewards=rewards,
            terminated=np.asarray(fields["terminated"], dtype=np.float32),
            returns=returns,
            root_candidates=np.asarray(fields["root_candidates"], dtype=np.float32),
            root_weights=np.asarray(fields["root_weights"], dtype=np.float32),
            predicted_scores=np.asarray(fields["predicted_scores"], dtype=np.float32),
            generation_id=self.generation_id,
        )
        if episode.observations.shape != (length + 1, self.obs_dim):
            raise RuntimeError("episode state topology is inconsistent")
        self.episodes.append(episode)
        self.pending[env_index] = {key: [] for key in fields}

    def start_training(self, device: torch.device | str) -> None:
        if not self.ready:
            raise RuntimeError("all generation episodes must finish before training")
        if any(episode.generation_id != self.generation_id for episode in self.episodes):
            raise RuntimeError("generation ids were mixed")
        self.phase = self.TRAIN
        device = torch.device(device)
        max_length = max(len(episode.actions) for episode in self.episodes)
        e = self.num_envs
        observations = torch.zeros(e, max_length + 1, self.obs_dim, device=device)
        remaining = torch.zeros(e, max_length + 1, device=device)
        actions = torch.zeros(e, max_length, self.action_dim, device=device)
        rewards = torch.zeros(e, max_length, device=device)
        terminated = torch.zeros(e, max_length, device=device)
        returns = torch.zeros(e, max_length + 1, device=device)
        candidates = torch.zeros(
            e, max_length, self.candidates, self.action_dim, device=device
        )
        weights = torch.zeros(e, max_length, self.candidates, device=device)
        lengths = torch.empty(e, dtype=torch.long, device=device)
        flat_episode: list[int] = []
        flat_time: list[int] = []
        for env_index, episode in enumerate(self.episodes):
            length = len(episode.actions)
            observations[env_index, : length + 1].copy_(
                torch.as_tensor(episode.observations, device=device)
            )
            remaining[env_index, : length + 1].copy_(
                torch.as_tensor(episode.remaining, device=device)
            )
            actions[env_index, :length].copy_(torch.as_tensor(episode.actions, device=device))
            rewards[env_index, :length].copy_(torch.as_tensor(episode.rewards, device=device))
            terminated[env_index, :length].copy_(
                torch.as_tensor(episode.terminated, device=device)
            )
            returns[env_index, : length + 1].copy_(
                torch.as_tensor(episode.returns, device=device)
            )
            candidates[env_index, :length].copy_(
                torch.as_tensor(episode.root_candidates, device=device)
            )
            weights[env_index, :length].copy_(
                torch.as_tensor(episode.root_weights, device=device)
            )
            lengths[env_index] = length
            flat_episode.extend([env_index] * length)
            flat_time.extend(range(length))
        self.tensors = {
            "observations": observations,
            "remaining": remaining,
            "actions": actions,
            "rewards": rewards,
            "terminated": terminated,
            "returns": returns,
            "root_candidates": candidates,
            "root_weights": weights,
            "lengths": lengths,
            "flat_episode": torch.tensor(flat_episode, dtype=torch.long, device=device),
            "flat_time": torch.tensor(flat_time, dtype=torch.long, device=device),
        }

    def prequential_diagnostics(self) -> dict[str, float]:
        if not self.ready:
            raise RuntimeError("prequential diagnostics require a complete generation")
        predicted = np.concatenate([episode.predicted_scores for episode in self.episodes])
        factual = np.concatenate([episode.returns[:-1] for episode in self.episodes])
        error = predicted - factual
        predicted_centered = predicted - predicted.mean()
        factual_centered = factual - factual.mean()
        denominator = np.sqrt(
            np.square(predicted_centered).sum() * np.square(factual_centered).sum()
        )
        correlation = (
            float(np.dot(predicted_centered, factual_centered) / denominator)
            if denominator > 0.0
            else 0.0
        )
        return {
            "model_exploitation/selected_q_bias": float(error.mean()),
            "model_exploitation/selected_q_mae": float(np.abs(error).mean()),
            "model_exploitation/selected_q_return_correlation": correlation,
            "model_exploitation/selected_q_min": float(predicted.min()),
            "model_exploitation/selected_q_max": float(predicted.max()),
        }

    def sample(
        self,
        batch_size: int,
        unroll_steps: int,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        if self.phase != self.TRAIN or not self.tensors:
            raise RuntimeError("generation is not materialized for training")
        data = self.tensors
        total = data["flat_episode"].numel()
        sample_index = torch.randint(
            total,
            (batch_size,),
            generator=generator,
            device=data["flat_episode"].device,
        )
        return self._gather_sequences(sample_index, unroll_steps)

    def fixed_sequences(
        self, batch_size: int, unroll_steps: int
    ) -> dict[str, torch.Tensor]:
        """Deterministic factual roots used only for across-phase drift diagnostics."""
        if self.phase != self.TRAIN or not self.tensors:
            raise RuntimeError("generation is not materialized for training")
        total = self.tensors["flat_episode"].numel()
        count = min(batch_size, total)
        sample_index = torch.linspace(
            0,
            total - 1,
            steps=count,
            device=self.tensors["flat_episode"].device,
            dtype=torch.float64,
        ).round().to(torch.long)
        return self._gather_sequences(sample_index, unroll_steps)

    def _gather_sequences(
        self, sample_index: torch.Tensor, unroll_steps: int
    ) -> dict[str, torch.Tensor]:
        data = self.tensors
        episode = data["flat_episode"][sample_index]
        start = data["flat_time"][sample_index]
        lengths = data["lengths"][episode]
        state_offset = torch.arange(
            unroll_steps + 1, device=episode.device, dtype=torch.long
        )
        transition_offset = state_offset[:-1]
        state_time = start[:, None] + state_offset[None]
        transition_time = start[:, None] + transition_offset[None]
        state_mask = state_time <= lengths[:, None]
        transition_mask = transition_time < lengths[:, None]
        safe_state_time = torch.minimum(state_time, lengths[:, None])
        safe_transition_time = torch.minimum(
            transition_time, (lengths - 1)[:, None]
        )
        episode_state = episode[:, None].expand_as(safe_state_time)
        episode_transition = episode[:, None].expand_as(safe_transition_time)
        return {
            "obs0": data["observations"][episode, start],
            "remaining": data["remaining"][episode_state, safe_state_time],
            "actions": data["actions"][episode_transition, safe_transition_time],
            "rewards": data["rewards"][episode_transition, safe_transition_time],
            "terminated": data["terminated"][episode_transition, safe_transition_time],
            "returns": data["returns"][episode_state, safe_state_time],
            "root_candidates": data["root_candidates"][
                episode_transition, safe_transition_time
            ],
            "root_weights": data["root_weights"][episode_transition, safe_transition_time],
            "transition_mask": transition_mask.to(torch.float32),
            "state_mask": state_mask.to(torch.float32),
        }

    def all_roots(self) -> dict[str, torch.Tensor]:
        """Every factual generation root exactly once, in deterministic env/time order."""
        if self.phase != self.TRAIN or not self.tensors:
            raise RuntimeError("generation is not materialized for training")
        data = self.tensors
        episode = data["flat_episode"]
        time_index = data["flat_time"]
        return {
            "observations": data["observations"][episode, time_index],
            "remaining": data["remaining"][episode, time_index],
            "root_candidates": data["root_candidates"][episode, time_index],
            "root_weights": data["root_weights"][episode, time_index],
        }

    def clear(self) -> None:
        if self.phase != self.TRAIN:
            raise RuntimeError("only a trained generation can be cleared")
        if any(fields["actions"] for fields in self.pending):
            raise RuntimeError("pending fragments cannot cross a generation boundary")
        self.episodes.clear()
        self.pending.clear()
        self.tensors.clear()
        self.active[:] = False
        self.phase = self.EMPTY
        self.generation_id = -1
        self.vector_steps = 0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class LossOutputs(NamedTuple):
    total: torch.Tensor
    reward: torch.Tensor
    termination: torch.Tensor
    value: torch.Tensor
    plan: torch.Tensor


def predictron_composed_return_loss(
    predicted_reward: torch.Tensor,
    termination_logit: torch.Tensor,
    value_rate: torch.Tensor,
    remaining: torch.Tensor,
    returns: torch.Tensor,
    transition_mask: torch.Tensor,
    max_episode_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All valid subroot/depth composed returns in stationary rate units."""
    k = predicted_reward.shape[1]
    exact_time_continue = (remaining[:, :-1] > 1.0).to(predicted_reward.dtype)
    predicted_continue = exact_time_continue * torch.sigmoid(-termination_logit)
    error_sum = torch.zeros((), device=predicted_reward.device, dtype=predicted_reward.dtype)
    valid_count = torch.zeros_like(error_sum)
    for subroot in range(k):
        survival = torch.ones_like(predicted_reward[:, 0])
        prefix = torch.zeros_like(survival)
        for depth in range(1, k - subroot + 1):
            edge = subroot + depth - 1
            prefix = prefix + survival * predicted_reward[:, edge]
            survival = survival * predicted_continue[:, edge]
            composed = prefix + survival * float(max_episode_steps) * value_rate[:, edge + 1]
            valid = transition_mask[:, subroot] * transition_mask[:, edge]
            error = 0.5 * (
                composed / float(max_episode_steps)
                - returns[:, subroot] / float(max_episode_steps)
            ).square()
            error_sum = error_sum + (error * valid).sum()
            valid_count = valid_count + valid.sum()
    return error_sum / valid_count.clamp_min(1.0), valid_count


def direct_outcome_loss(
    network: DirectOutcomeMuZero,
    obs0: torch.Tensor,
    remaining: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    returns: torch.Tensor,
    transition_mask: torch.Tensor,
    state_mask: torch.Tensor,
    max_episode_steps: int,
) -> LossOutputs:
    """K-step live recurrent loss; every Predictron path retains full BPTT."""
    k = actions.shape[1]
    remaining_rate = remaining / float(max_episode_steps)
    initial = network.initial_inference(obs0, remaining_rate[:, 0])
    latents = [initial.latent]
    value_rates = [initial.value_rate]
    predicted_rewards: list[torch.Tensor] = []
    termination_logits: list[torch.Tensor] = []
    latent = initial.latent
    for step in range(k):
        recurrent = network.recurrent_inference(
            latent, actions[:, step], remaining_rate[:, step + 1]
        )
        latent = recurrent.latent
        latents.append(latent)
        predicted_rewards.append(recurrent.reward)
        termination_logits.append(recurrent.termination_logit)
        value_rates.append(recurrent.value_rate)

    predicted_reward = torch.stack(predicted_rewards, dim=1)
    termination_logit = torch.stack(termination_logits, dim=1)
    value_rate = torch.stack(value_rates, dim=1)

    reward_loss = _masked_mean(
        0.5 * (predicted_reward - rewards).square(), transition_mask
    )
    termination_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            termination_logit, terminated, reduction="none"
        ),
        transition_mask,
    )
    value_loss = _masked_mean(
        0.5 * (value_rate - returns / float(max_episode_steps)).square(), state_mask
    )

    plan_loss, _plan_count = predictron_composed_return_loss(
        predicted_reward,
        termination_logit,
        value_rate,
        remaining,
        returns,
        transition_mask,
        max_episode_steps,
    )

    total = reward_loss + termination_loss + value_loss + plan_loss
    return LossOutputs(
        total,
        reward_loss,
        termination_loss,
        value_loss,
        plan_loss,
    )


class RootPolicyMetrics(NamedTuple):
    centered_loss: torch.Tensor
    uncentered_cross_entropy: torch.Tensor
    grad_norm: torch.Tensor
    clipped_grad_norm: torch.Tensor
    grad_clip_scale: torch.Tensor
    parameter_delta_norm: torch.Tensor
    exactly_uniform_fraction: torch.Tensor
    predicted_std_geometric: torch.Tensor
    predicted_std_arithmetic: torch.Tensor
    target_std: torch.Tensor
    target_current_variance_ratio: torch.Tensor
    visit_entropy: torch.Tensor
    visit_effective_candidates: torch.Tensor
    projection_excess_kurtosis_abs: torch.Tensor
    logstd_lower_fraction: torch.Tensor
    logstd_upper_fraction: torch.Tensor


def centered_root_policy_loss(
    network: DirectOutcomeMuZero,
    observations: torch.Tensor,
    remaining: torch.Tensor,
    root_candidates: torch.Tensor,
    root_weights: torch.Tensor,
    max_episode_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """On-generation score gradient with the exact sampled-policy control variate."""
    inference = network.initial_inference(
        observations, remaining / float(max_episode_steps)
    )
    mean = inference.policy_mean
    log_std = inference.policy_log_std
    log_prob = (
        -log_std[:, None]
        - 0.5
        * (root_candidates - mean[:, None]).square()
        * torch.exp(-2.0 * log_std[:, None])
        - 0.5 * LOG_2PI
    )
    uniform = 1.0 / root_weights.shape[1]
    improvement = root_weights - uniform
    centered = -(improvement[:, :, None] * log_prob).sum(dim=(1, 2)).mean()
    cross_entropy = -(root_weights[:, :, None] * log_prob).sum(dim=(1, 2)).mean()

    target_mean = (root_weights[:, :, None] * root_candidates).sum(dim=1)
    centered_candidates = root_candidates - target_mean[:, None]
    target_variance = (
        root_weights[:, :, None] * centered_candidates.square()
    ).sum(dim=1)
    target_std = torch.sqrt(
        target_variance.clamp_min(torch.finfo(observations.dtype).tiny)
    )
    current_variance = torch.exp(2.0 * log_std)
    fourth = (
        root_weights[:, :, None] * centered_candidates.pow(4)
    ).sum(dim=1)
    excess_kurtosis = fourth / target_variance.square().clamp_min(
        torch.finfo(observations.dtype).tiny
    ) - 3.0
    entropy = -(
        root_weights
        * torch.where(
            root_weights > 0,
            torch.log(root_weights),
            torch.zeros_like(root_weights),
        )
    ).sum(dim=-1)
    return centered, {
        "uncentered_cross_entropy": cross_entropy,
        "exactly_uniform_fraction": (improvement == 0.0).all(dim=-1).to(torch.float32).mean(),
        "predicted_std_geometric": torch.exp(log_std.mean()),
        "predicted_std_arithmetic": torch.exp(log_std).mean(),
        "target_std": target_std.mean(),
        "target_current_variance_ratio": (target_variance / current_variance).mean(),
        "visit_entropy": entropy.mean(),
        "visit_effective_candidates": torch.exp(entropy).mean(),
        "projection_excess_kurtosis_abs": excess_kurtosis.abs().mean(),
        "logstd_lower_fraction": (log_std - network.min_log_std < 0.05)
        .to(torch.float32)
        .mean(),
        "logstd_upper_fraction": (network.max_log_std - log_std < 0.05)
        .to(torch.float32)
        .mean(),
    }


def proposal_predictions(
    network: DirectOutcomeMuZero,
    roots: dict[str, torch.Tensor],
    max_episode_steps: int,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    means: list[torch.Tensor] = []
    log_stds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, roots["observations"].shape[0], batch_size):
            stop = min(start + batch_size, roots["observations"].shape[0])
            inference = network.initial_inference(
                roots["observations"][start:stop],
                roots["remaining"][start:stop] / float(max_episode_steps),
            )
            means.append(inference.policy_mean)
            log_stds.append(inference.policy_log_std)
    return torch.cat(means), torch.cat(log_stds)


def recurrent_proposal_predictions(
    network: DirectOutcomeMuZero,
    batch: dict[str, torch.Tensor],
    max_episode_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Policy predictions along a fixed factual recurrent unroll, without detaches."""
    remaining_rate = batch["remaining"] / float(max_episode_steps)
    with torch.no_grad():
        initial = network.initial_inference(batch["obs0"], remaining_rate[:, 0])
        means = [initial.policy_mean]
        log_stds = [initial.policy_log_std]
        latent = initial.latent
        for step in range(batch["actions"].shape[1]):
            recurrent = network.recurrent_inference(
                latent, batch["actions"][:, step], remaining_rate[:, step + 1]
            )
            latent = recurrent.latent
            means.append(recurrent.policy_mean)
            log_stds.append(recurrent.policy_log_std)
    return torch.stack(means, dim=1), torch.stack(log_stds, dim=1)


def diagonal_gaussian_kl(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    """KL(old || new), summed over the joint diagonal action distribution."""
    return (
        new_log_std
        - old_log_std
        + 0.5
        * (
            torch.exp(2.0 * old_log_std)
            + (old_mean - new_mean).square()
        )
        * torch.exp(-2.0 * new_log_std)
        - 0.5
    ).sum(dim=-1)


def normalized_mean_shift(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    standardized = (new_mean - old_mean) * torch.exp(-old_log_std)
    return torch.sqrt(standardized.square().mean()), standardized.abs().mean()


def apply_generation_policy_improvement(
    network: DirectOutcomeMuZero,
    roots: dict[str, torch.Tensor],
    max_episode_steps: int,
    learning_rate: float,
    batch_size: int = 256,
    max_grad_norm: float = 0.5,
) -> RootPolicyMetrics:
    """Exactly one stateless SGD score-gradient step over every generation root."""
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    total_roots = roots["observations"].shape[0]
    if total_roots == 0:
        raise RuntimeError("policy improvement requires at least one factual root")
    network.zero_grad(set_to_none=True)
    metric_sums: dict[str, torch.Tensor] = {}
    centered_sum = torch.zeros((), device=roots["observations"].device)
    for start in range(0, total_roots, batch_size):
        stop = min(start + batch_size, total_roots)
        count = stop - start
        loss, metrics = centered_root_policy_loss(
            network,
            roots["observations"][start:stop],
            roots["remaining"][start:stop],
            roots["root_candidates"][start:stop],
            roots["root_weights"][start:stop],
            max_episode_steps,
        )
        fraction = count / total_roots
        (loss * fraction).backward()
        centered_sum = centered_sum + loss.detach() * fraction
        for name, value in metrics.items():
            metric_sums[name] = metric_sums.get(name, torch.zeros_like(value)) + value.detach() * fraction
    grad_square = torch.zeros((), device=roots["observations"].device)
    with torch.no_grad():
        for parameter in network.parameters():
            if parameter.grad is not None:
                grad_square = grad_square + parameter.grad.square().sum()
        grad_norm = torch.sqrt(grad_square)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("nonfinite centered policy-gradient norm")
        clip_scale = torch.minimum(
            torch.ones_like(grad_norm),
            torch.as_tensor(max_grad_norm, device=grad_norm.device) / grad_norm.clamp_min(1e-12),
        )
        for parameter in network.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad * clip_scale, alpha=-learning_rate)
        torch._assert_async(
            torch.stack([parameter.isfinite().all() for parameter in network.parameters()]).all(),
            "nonfinite parameter after centered policy update",
        )
        clipped_grad_norm = grad_norm * clip_scale
        parameter_delta_norm = learning_rate * clipped_grad_norm
    network.zero_grad(set_to_none=True)
    return RootPolicyMetrics(
        centered_sum,
        metric_sums["uncentered_cross_entropy"],
        grad_norm,
        clipped_grad_norm,
        clip_scale,
        parameter_delta_norm,
        metric_sums["exactly_uniform_fraction"],
        metric_sums["predicted_std_geometric"],
        metric_sums["predicted_std_arithmetic"],
        metric_sums["target_std"],
        metric_sums["target_current_variance_ratio"],
        metric_sums["visit_entropy"],
        metric_sums["visit_effective_candidates"],
        metric_sums["projection_excess_kurtosis_abs"],
        metric_sums["logstd_lower_fraction"],
        metric_sums["logstd_upper_fraction"],
    )


def evaluate_centered_policy_surrogate(
    network: DirectOutcomeMuZero,
    roots: dict[str, torch.Tensor],
    max_episode_steps: int,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Diagnostic-only full-generation value of the fixed collection surrogate."""
    total_roots = roots["observations"].shape[0]
    result = torch.zeros((), device=roots["observations"].device)
    with torch.no_grad():
        for start in range(0, total_roots, batch_size):
            stop = min(start + batch_size, total_roots)
            loss, _ = centered_root_policy_loss(
                network,
                roots["observations"][start:stop],
                roots["remaining"][start:stop],
                roots["root_candidates"][start:stop],
                roots["root_weights"][start:stop],
                max_episode_steps,
            )
            result = result + loss * ((stop - start) / total_roots)
    return result


class SearchOutputs(NamedTuple):
    root_pre_tanh: torch.Tensor
    root_normalized_actions: torch.Tensor
    visit_history: torch.Tensor
    q_history: torch.Tensor
    q_min: torch.Tensor
    q_max: torch.Tensor
    expansion_depth_mean: torch.Tensor
    expansion_depth_max: torch.Tensor


def _scatter_add_edges(
    target: torch.Tensor,
    nodes: torch.Tensor,
    edges: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Functional additive update for possibly repeated [environment,node,edge] keys."""
    environments, max_nodes, max_candidates = target.shape
    env = torch.arange(environments, device=target.device, dtype=torch.long)
    env = env.view(environments, *([1] * (nodes.ndim - 1))).expand_as(nodes)
    flat_index = (env * max_nodes + nodes) * max_candidates + edges
    return target.reshape(-1).scatter_add(0, flat_index.reshape(-1), values.reshape(-1)).view_as(
        target
    )


class SampledMuZeroSearch(nn.Module):
    """Fixed tensor-tree, wave-batched Sampled MuZero with exact virtual reservations.

    Q and its tree-wide min/max are frozen within each wave.  Each lane reserves one
    open path using N+R, and unexpanded frontiers are made exclusive.  All unique
    frontiers are recurrently expanded together, all lane backups are then added, and
    R is discarded.  This is a defined parallel-MCTS approximation, not serial MCTS.
    """

    def __init__(
        self,
        network: DirectOutcomeMuZero,
        max_episode_steps: int,
        root_candidates: int = 32,
        child_candidates: int = 8,
        simulations: int = 256,
        wave_size: int = 16,
        search_depth: int = 16,
        pb_c_init: float = 1.25,
        pb_c_base: float = 19_652.0,
    ):
        super().__init__()
        if simulations <= 0 or simulations % wave_size:
            raise ValueError("simulations must be a positive multiple of wave_size")
        if root_candidates < wave_size:
            raise ValueError("root candidate count must cover the first reservation wave")
        if child_candidates <= 0 or child_candidates > root_candidates:
            raise ValueError("child candidate count is invalid")
        self.network = network
        self.max_episode_steps = max_episode_steps
        self.root_candidates = root_candidates
        self.child_candidates = child_candidates
        self.simulations = simulations
        self.wave_size = wave_size
        self.search_depth = search_depth
        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base
        self.max_nodes = simulations + 1
        self.waves = simulations // wave_size

    def _node_open(
        self,
        node_count: torch.Tensor,
        node_depth: torch.Tensor,
        node_remaining: torch.Tensor,
        candidate_count: torch.Tensor,
        edge_child: torch.Tensor,
        frontier_reserved: torch.Tensor,
    ) -> torch.Tensor:
        environments = edge_child.shape[0]
        node_index = torch.arange(self.max_nodes, device=edge_child.device)[None]
        exists = node_index < node_count[:, None]
        node_open = exists & (
            (node_depth >= self.search_depth)
            | (node_remaining <= 0.0)
            | (candidate_count == 0)
        )
        candidate_index = torch.arange(self.root_candidates, device=edge_child.device)
        for depth in range(self.search_depth - 1, -1, -1):
            child_index = edge_child.clamp_min(0)
            child_is_open = torch.gather(node_open, 1, child_index.flatten(1)).view(
                environments, self.max_nodes, self.root_candidates
            )
            valid_candidate = candidate_index[None, None] < candidate_count[:, :, None]
            edge_is_open = valid_candidate & (
                (edge_child == -2)
                | ((edge_child == -1) & ~frontier_reserved)
                | ((edge_child >= 0) & child_is_open)
            )
            at_depth = exists & (node_depth == depth)
            node_open = torch.where(at_depth, edge_is_open.any(dim=-1), node_open)
        return node_open

    def _tree_minmax(
        self, edge_visit: torch.Tensor, edge_value_sum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visited = edge_visit > 0
        q = torch.where(
            visited,
            edge_value_sum / edge_visit.clamp_min(1).to(edge_value_sum.dtype),
            torch.zeros_like(edge_value_sum),
        )
        positive_inf = torch.full_like(q, torch.inf)
        negative_inf = torch.full_like(q, -torch.inf)
        q_min = torch.where(visited, q, positive_inf).flatten(1).amin(dim=1)
        q_max = torch.where(visited, q, negative_inf).flatten(1).amax(dim=1)
        any_visited = visited.flatten(1).any(dim=1)
        q_min = torch.where(any_visited, q_min, torch.zeros_like(q_min))
        q_max = torch.where(any_visited, q_max, torch.zeros_like(q_max))
        span = q_max - q_min
        normalized = torch.where(
            visited & (span[:, None, None] > 0.0),
            (q - q_min[:, None, None]) / span[:, None, None].clamp_min(
                torch.finfo(q.dtype).tiny
            ),
            torch.zeros_like(q),
        )
        return normalized, q_min, q_max

    def _update_open_path(
        self,
        node_open: torch.Tensor,
        candidate_count: torch.Tensor,
        edge_child: torch.Tensor,
        frontier_reserved: torch.Tensor,
        path_node: torch.Tensor,
        path_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate one exclusive frontier reservation through its ancestors."""
        environments = edge_child.shape[0]
        env = torch.arange(environments, device=edge_child.device)
        candidate_index = torch.arange(self.root_candidates, device=edge_child.device)[None]
        for path_depth in range(self.search_depth - 1, -1, -1):
            node = path_node[:, path_depth]
            valid_path_node = path_valid[:, path_depth]
            children = edge_child[env, node]
            child_is_open = torch.gather(node_open, 1, children.clamp_min(0))
            valid_candidate = candidate_index < candidate_count[env, node, None]
            edge_is_open = valid_candidate & (
                (children == -2)
                | ((children == -1) & ~frontier_reserved[env, node])
                | ((children >= 0) & child_is_open)
            )
            old_node_open = node_open[env, node]
            node_open[env, node] = torch.where(
                valid_path_node, edge_is_open.any(dim=-1), old_node_open
            )
        return node_open

    def forward(
        self,
        observations: torch.Tensor,
        remaining: torch.Tensor,
        root_noise: torch.Tensor,
        child_noise: torch.Tensor,
    ) -> SearchOutputs:
        environments = observations.shape[0]
        device = observations.device
        dtype = observations.dtype
        if root_noise.shape[1:] != (self.root_candidates, self.network.action_dim):
            raise ValueError("root noise has the wrong trailing shape")
        if child_noise.shape[1] < self.simulations:
            raise ValueError("child noise does not cover all possible created nodes")

        initial = self.network.initial_inference(
            observations, remaining / float(self.max_episode_steps)
        )
        root_pre_tanh = (
            initial.policy_mean[:, None]
            + torch.exp(initial.policy_log_std)[:, None] * root_noise
        )

        latent = torch.zeros(
            environments, self.max_nodes, self.network.latent_dim, device=device, dtype=dtype
        )
        node_remaining = torch.zeros(environments, self.max_nodes, device=device, dtype=dtype)
        node_depth = torch.zeros(
            environments, self.max_nodes, device=device, dtype=torch.long
        )
        node_value_rate = torch.zeros(
            environments, self.max_nodes, device=device, dtype=dtype
        )
        candidate_count = torch.zeros(
            environments, self.max_nodes, device=device, dtype=torch.long
        )
        candidate_pre_tanh = torch.zeros(
            environments,
            self.max_nodes,
            self.root_candidates,
            self.network.action_dim,
            device=device,
            dtype=dtype,
        )
        edge_child = torch.full(
            (environments, self.max_nodes, self.root_candidates),
            -1,
            device=device,
            dtype=torch.long,
        )
        edge_reward = torch.zeros_like(edge_child, dtype=dtype)
        edge_continue = torch.zeros_like(edge_child, dtype=dtype)
        edge_visit = torch.zeros_like(edge_child)
        edge_value_sum = torch.zeros_like(edge_reward)
        node_count = torch.ones(environments, device=device, dtype=torch.long)
        env = torch.arange(environments, device=device, dtype=torch.long)

        latent[:, 0] = initial.latent
        node_remaining[:, 0] = remaining
        node_value_rate[:, 0] = initial.value_rate
        candidate_count[:, 0] = self.root_candidates
        candidate_pre_tanh[:, 0] = root_pre_tanh

        visit_history: list[torch.Tensor] = []
        q_history: list[torch.Tensor] = []
        expansion_depth_sum = torch.zeros(environments, device=device, dtype=dtype)
        expansion_depth_count = torch.zeros(environments, device=device, dtype=dtype)
        expansion_depth_max = torch.zeros(environments, device=device, dtype=dtype)

        for _wave in range(self.waves):
            normalized_q, _wave_q_min, _wave_q_max = self._tree_minmax(
                edge_visit, edge_value_sum
            )
            reservation = torch.zeros_like(edge_visit)
            frontier_reserved = torch.zeros_like(edge_child, dtype=torch.bool)
            path_node = torch.zeros(
                environments,
                self.wave_size,
                self.search_depth,
                device=device,
                dtype=torch.long,
            )
            path_edge = torch.zeros_like(path_node)
            path_valid = torch.zeros_like(path_node, dtype=torch.bool)
            frontier_node = torch.zeros(
                environments, self.wave_size, device=device, dtype=torch.long
            )
            frontier_edge = torch.zeros_like(frontier_node)
            frontier_valid = torch.zeros_like(frontier_node, dtype=torch.bool)
            leaf_value = torch.zeros(
                environments, self.wave_size, device=device, dtype=dtype
            )
            node_open = self._node_open(
                node_count,
                node_depth,
                node_remaining,
                candidate_count,
                edge_child,
                frontier_reserved,
            )

            for lane in range(self.wave_size):
                current = torch.zeros(environments, device=device, dtype=torch.long)
                active = torch.ones(environments, device=device, dtype=torch.bool)
                for path_depth in range(self.search_depth):
                    current_depth = node_depth[env, current]
                    current_remaining = node_remaining[env, current]
                    current_candidate_count = candidate_count[env, current]
                    terminal_node = active & (
                        (current_depth >= self.search_depth)
                        | (current_remaining <= 0.0)
                        | (current_candidate_count == 0)
                    )
                    current_leaf = (
                        float(self.max_episode_steps) * node_value_rate[env, current]
                    )
                    leaf_value[:, lane] = torch.where(
                        terminal_node, current_leaf, leaf_value[:, lane]
                    )
                    active = active & ~terminal_node

                    children = edge_child[env, current]
                    child_is_open = torch.gather(
                        node_open, 1, children.clamp_min(0)
                    )
                    valid_candidate = (
                        torch.arange(self.root_candidates, device=device)[None]
                        < current_candidate_count[:, None]
                    )
                    eligible = valid_candidate & (
                        (children == -2)
                        | ((children == -1) & ~frontier_reserved[env, current])
                        | ((children >= 0) & child_is_open)
                    )
                    has_eligible = eligible.any(dim=-1)
                    saturated = active & ~has_eligible
                    leaf_value[:, lane] = torch.where(
                        saturated, current_leaf, leaf_value[:, lane]
                    )
                    active = active & has_eligible

                    actual_visit = edge_visit[env, current]
                    virtual_visit = reservation[env, current]
                    effective_visit = actual_visit + virtual_visit
                    unvisited = eligible & (effective_visit == 0)
                    forced = torch.argmax(unvisited.to(torch.int64), dim=-1)
                    parent_visit = (
                        effective_visit
                        * valid_candidate.to(effective_visit.dtype)
                    ).sum(dim=-1)
                    parent_visit_float = parent_visit.to(dtype)
                    pb_c = (
                        self.pb_c_init
                        + torch.log(
                            (parent_visit_float + self.pb_c_base + 1.0)
                            / self.pb_c_base
                        )
                    )
                    exploration = (
                        pb_c[:, None]
                        * (1.0 / current_candidate_count.clamp_min(1).to(dtype))[:, None]
                        * torch.sqrt(parent_visit_float)[:, None]
                        / (1.0 + effective_visit.to(dtype))
                    )
                    score = normalized_q[env, current] + exploration
                    score = torch.where(
                        eligible, score, torch.full_like(score, -torch.inf)
                    )
                    puct = torch.argmax(score, dim=-1)
                    chosen = torch.where(unvisited.any(dim=-1), forced, puct)

                    path_node[:, lane, path_depth] = current
                    path_edge[:, lane, path_depth] = chosen
                    path_valid[:, lane, path_depth] = active
                    chosen_child = children.gather(1, chosen[:, None]).squeeze(1)
                    is_frontier = active & (chosen_child == -1)
                    is_terminal_edge = active & (chosen_child == -2)
                    frontier_node[:, lane] = torch.where(
                        is_frontier, current, frontier_node[:, lane]
                    )
                    frontier_edge[:, lane] = torch.where(
                        is_frontier, chosen, frontier_edge[:, lane]
                    )
                    frontier_valid[:, lane] = frontier_valid[:, lane] | is_frontier
                    active = active & ~is_frontier & ~is_terminal_edge
                    current = torch.where(active, chosen_child.clamp_min(0), current)

                leaf_value[:, lane] = torch.where(
                    active,
                    float(self.max_episode_steps) * node_value_rate[env, current],
                    leaf_value[:, lane],
                )
                lane_values = path_valid[:, lane].to(edge_value_sum.dtype)
                reservation = _scatter_add_edges(
                    reservation,
                    path_node[:, lane],
                    path_edge[:, lane],
                    lane_values.to(reservation.dtype),
                )
                old_reserved = frontier_reserved[
                    env, frontier_node[:, lane], frontier_edge[:, lane]
                ]
                frontier_reserved[
                    env, frontier_node[:, lane], frontier_edge[:, lane]
                ] = torch.where(
                    frontier_valid[:, lane],
                    torch.ones_like(old_reserved),
                    old_reserved,
                )
                # Reserving one frontier can only close nodes on its unique root path.
                # Recomputing those ancestors bottom-up is exactly equivalent to a full
                # tree OPEN scan before the next lane, while avoiding a 16x tree scan.
                node_open = self._update_open_path(
                    node_open,
                    candidate_count,
                    edge_child,
                    frontier_reserved,
                    path_node[:, lane],
                    path_valid[:, lane],
                )

            parent_latent = latent[
                env[:, None], frontier_node
            ]
            selected_pre_tanh = candidate_pre_tanh[
                env[:, None], frontier_node, frontier_edge
            ]
            selected_action = torch.tanh(selected_pre_tanh)
            parent_remaining = node_remaining[env[:, None], frontier_node]
            next_remaining = (parent_remaining - 1.0).clamp_min(0.0)
            recurrent = self.network.recurrent_inference(
                parent_latent.flatten(0, 1),
                selected_action.flatten(0, 1),
                (next_remaining / float(self.max_episode_steps)).flatten(),
            )
            recurrent_latent = recurrent.latent.view(
                environments, self.wave_size, self.network.latent_dim
            )
            recurrent_reward = recurrent.reward.view(environments, self.wave_size)
            recurrent_termination = recurrent.termination_logit.view(
                environments, self.wave_size
            )
            recurrent_value = recurrent.value_rate.view(environments, self.wave_size)
            recurrent_mean = recurrent.policy_mean.view(
                environments, self.wave_size, self.network.action_dim
            )
            recurrent_log_std = recurrent.policy_log_std.view(
                environments, self.wave_size, self.network.action_dim
            )
            exact_time_continue = (parent_remaining > 1.0).to(dtype)
            recurrent_continue = exact_time_continue * torch.sigmoid(-recurrent_termination)
            create_child = frontier_valid & (next_remaining > 0.0)
            child_offset = torch.cumsum(create_child.to(torch.long), dim=1) - 1
            new_node = node_count[:, None] + child_offset.clamp_min(0)
            new_depth = node_depth[env[:, None], frontier_node] + 1

            for lane in range(self.wave_size):
                parent = frontier_node[:, lane]
                edge = frontier_edge[:, lane]
                valid = frontier_valid[:, lane]
                create = create_child[:, lane]
                old_reward = edge_reward[env, parent, edge]
                old_continue = edge_continue[env, parent, edge]
                old_child = edge_child[env, parent, edge]
                edge_reward[env, parent, edge] = torch.where(
                    valid, recurrent_reward[:, lane], old_reward
                )
                edge_continue[env, parent, edge] = torch.where(
                    valid, recurrent_continue[:, lane], old_continue
                )
                edge_child[env, parent, edge] = torch.where(
                    valid,
                    torch.where(create, new_node[:, lane], torch.full_like(old_child, -2)),
                    old_child,
                )

                node = new_node[:, lane]
                old_latent = latent[env, node]
                old_remaining = node_remaining[env, node]
                old_depth = node_depth[env, node]
                old_value = node_value_rate[env, node]
                old_count = candidate_count[env, node]
                noise_index = (node - 1).clamp(min=0, max=self.simulations - 1)
                noise = child_noise[env, noise_index]
                child_pre_tanh = (
                    recurrent_mean[:, lane, None]
                    + torch.exp(recurrent_log_std[:, lane, None]) * noise
                )
                old_candidates = candidate_pre_tanh[env, node]
                padded_candidates = old_candidates.clone()
                padded_candidates[:, : self.child_candidates] = child_pre_tanh
                latent[env, node] = torch.where(
                    create[:, None], recurrent_latent[:, lane], old_latent
                )
                node_remaining[env, node] = torch.where(
                    create, next_remaining[:, lane], old_remaining
                )
                node_depth[env, node] = torch.where(create, new_depth[:, lane], old_depth)
                node_value_rate[env, node] = torch.where(
                    create, recurrent_value[:, lane], old_value
                )
                count = torch.where(
                    (new_depth[:, lane] < self.search_depth)
                    & (next_remaining[:, lane] > 0.0),
                    torch.full_like(old_count, self.child_candidates),
                    torch.zeros_like(old_count),
                )
                candidate_count[env, node] = torch.where(create, count, old_count)
                candidate_pre_tanh[env, node] = torch.where(
                    create[:, None, None], padded_candidates, old_candidates
                )
                leaf_value[:, lane] = torch.where(
                    create,
                    float(self.max_episode_steps) * recurrent_value[:, lane],
                    leaf_value[:, lane],
                )

            created_this_wave = create_child.to(torch.long).sum(dim=1)
            node_count = node_count + created_this_wave
            depth_float = new_depth.to(dtype)
            expansion_depth_sum = expansion_depth_sum + (
                depth_float * create_child.to(dtype)
            ).sum(dim=1)
            expansion_depth_count = expansion_depth_count + create_child.to(dtype).sum(dim=1)
            expansion_depth_max = torch.maximum(
                expansion_depth_max,
                torch.where(
                    create_child,
                    depth_float,
                    torch.zeros_like(depth_float),
                ).amax(dim=1),
            )

            backed_up = leaf_value
            for path_depth in range(self.search_depth - 1, -1, -1):
                nodes = path_node[:, :, path_depth]
                edges = path_edge[:, :, path_depth]
                valid = path_valid[:, :, path_depth]
                reward = edge_reward[env[:, None], nodes, edges]
                continuation = edge_continue[env[:, None], nodes, edges]
                edge_return = reward + continuation * backed_up
                contribution = torch.where(valid, edge_return, torch.zeros_like(edge_return))
                edge_value_sum = _scatter_add_edges(
                    edge_value_sum, nodes, edges, contribution
                )
                edge_visit = _scatter_add_edges(
                    edge_visit,
                    nodes,
                    edges,
                    valid.to(edge_visit.dtype),
                )
                backed_up = torch.where(valid, edge_return, backed_up)

            root_visit = edge_visit[:, 0, : self.root_candidates]
            root_q = torch.where(
                root_visit > 0,
                edge_value_sum[:, 0, : self.root_candidates]
                / root_visit.clamp_min(1).to(dtype),
                torch.zeros_like(edge_value_sum[:, 0, : self.root_candidates]),
            )
            visit_history.append(root_visit.clone())
            q_history.append(root_q)

        _normalized, q_min, q_max = self._tree_minmax(edge_visit, edge_value_sum)
        return SearchOutputs(
            root_pre_tanh,
            torch.tanh(root_pre_tanh),
            torch.stack(visit_history, dim=0),
            torch.stack(q_history, dim=0),
            q_min,
            q_max,
            expansion_depth_sum / expansion_depth_count.clamp_min(1.0),
            expansion_depth_max,
        )


def root_visit_weights(visit_count: torch.Tensor) -> torch.Tensor:
    return visit_count.to(torch.float32) / visit_count.sum(dim=-1, keepdim=True).clamp_min(1)


def select_root_candidate(
    visit_count: torch.Tensor,
    selection_uniform: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    if deterministic:
        return torch.argmax(visit_count, dim=-1)
    weights = root_visit_weights(visit_count)
    cumulative = torch.cumsum(weights, dim=-1)
    return (cumulative < selection_uniform[:, None]).sum(dim=-1).clamp_max(
        visit_count.shape[-1] - 1
    )


class BehaviorTransfer(NamedTuple):
    environment_action: np.ndarray
    normalized_action: np.ndarray
    root_pre_tanh: np.ndarray
    root_weights: np.ndarray
    selected_q: np.ndarray


def pack_behavior_transfer(
    environment_action: torch.Tensor,
    normalized_action: torch.Tensor,
    root_pre_tanh: torch.Tensor,
    root_weights: torch.Tensor,
    selected_q: torch.Tensor,
) -> BehaviorTransfer:
    """Pack all per-decision host data into one device synchronization and transfer."""
    environments, action_dim = normalized_action.shape
    candidates = root_weights.shape[1]
    packed = torch.cat(
        [
            environment_action,
            normalized_action,
            root_pre_tanh.flatten(1),
            root_weights,
            selected_q[:, None],
        ],
        dim=1,
    )
    host = packed.detach().cpu().numpy()
    offset = 0
    environment_host = host[:, offset : offset + action_dim]
    offset += action_dim
    normalized_host = host[:, offset : offset + action_dim]
    offset += action_dim
    root_host = host[:, offset : offset + candidates * action_dim].reshape(
        environments, candidates, action_dim
    )
    offset += candidates * action_dim
    weight_host = host[:, offset : offset + candidates]
    offset += candidates
    q_host = host[:, offset]
    return BehaviorTransfer(
        environment_host, normalized_host, root_host, weight_host, q_host
    )


def _clone_search_outputs(outputs: SearchOutputs) -> SearchOutputs:
    return SearchOutputs(*(value.clone() for value in outputs))


def assert_valid_search(outputs: SearchOutputs, expected_simulations: int) -> None:
    floating = (
        outputs.root_pre_tanh,
        outputs.root_normalized_actions,
        outputs.q_history,
        outputs.q_min,
        outputs.q_max,
        outputs.expansion_depth_mean,
        outputs.expansion_depth_max,
    )
    torch._assert_async(
        torch.stack([value.isfinite().all() for value in floating]).all(),
        "nonfinite Sampled MuZero search output",
    )
    final_visits = outputs.visit_history[-1]
    torch._assert_async(
        ((final_visits >= 0).all() & (final_visits.sum(dim=-1) == expected_simulations).all()),
        "Sampled MuZero root visits do not equal the simulation budget",
    )


def is_evaluation_step(completed_steps: int) -> bool:
    return completed_steps in {
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        4_000_000,
        6_000_000,
        8_000_000,
    }


def evaluate_inference_scaling(
    network: DirectOutcomeMuZero,
    planner: Any,
    envs: gym.vector.SyncVectorEnv,
    args: Args,
    obs_dim: int,
    action_dim: int,
    max_episode_steps: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Paired fixed-seed proposal/search evaluation at meaningful nested budgets."""
    results: dict[str, float] = {}
    modes = (0, 64, 128, 256)
    evaluation_seeds = [args.seed + 100_000 + index for index in range(args.num_envs)]
    network.eval()
    for budget in modes:
        observations, _ = envs.reset(seed=evaluation_seeds)
        active = np.ones(args.num_envs, dtype=bool)
        returns = np.zeros(args.num_envs, dtype=np.float64)
        episode_steps = np.zeros(args.num_envs, dtype=np.int64)
        initial_prediction = np.zeros(args.num_envs, dtype=np.float64)
        root_generator = torch.Generator(device=device)
        root_generator.manual_seed(args.seed + 701_003)
        child_generator = torch.Generator(device=device)
        child_generator.manual_seed(args.seed + 902_137)
        for environment_step in range(max_episode_steps):
            obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
            remaining = torch.as_tensor(
                max_episode_steps - episode_steps, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                if budget == 0:
                    inference = network.initial_inference(
                        obs, remaining / float(max_episode_steps)
                    )
                    normalized_action = torch.tanh(inference.policy_mean)
                    predicted = float(max_episode_steps) * inference.value_rate
                else:
                    root_noise = torch.randn(
                        (args.num_envs, args.root_candidates, action_dim),
                        generator=root_generator,
                        device=device,
                    )
                    # Every budget consumes the identical maximal noise block; smaller
                    # trees use its prefix, making candidate randomness nested exactly.
                    child_noise = torch.randn(
                        (
                            args.num_envs,
                            args.simulations,
                            args.child_candidates,
                            action_dim,
                        ),
                        generator=child_generator,
                        device=device,
                    )
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    search = _clone_search_outputs(
                        planner(obs, remaining, root_noise, child_noise)
                    )
                    assert_valid_search(search, args.simulations)
                    snapshot = budget // args.simulation_wave - 1
                    visits = search.visit_history[snapshot]
                    chosen = select_root_candidate(
                        visits,
                        torch.zeros(args.num_envs, device=device),
                        deterministic=True,
                    )
                    env_index = torch.arange(args.num_envs, device=device)
                    normalized_action = search.root_normalized_actions[env_index, chosen]
                    predicted = search.q_history[snapshot][env_index, chosen]
            if environment_step == 0:
                initial_prediction[:] = predicted.detach().cpu().numpy()
            environment_action = map_normalized_action(
                normalized_action, action_low, action_high
            )
            next_observations, rewards, terminated, truncated, _ = envs.step(
                environment_action.detach().cpu().numpy()
            )
            active_before = active.copy()
            returns += np.asarray(rewards) * active_before
            episode_steps += 1
            done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
            active &= ~done
            episode_steps[done] = 0
            observations = next_observations
            if not np.any(active):
                break
        if np.any(active):
            raise RuntimeError(f"evaluation budget {budget} did not finish all episodes")
        label = "proposal" if budget == 0 else f"search_{budget}"
        results[f"eval/{label}_return_mean"] = float(returns.mean())
        results[f"eval/{label}_return_median"] = float(np.median(returns))
        results[f"eval/{label}_return_min"] = float(returns.min())
        results[f"eval/{label}_initial_predicted_score"] = float(initial_prediction.mean())
        results[f"eval/{label}_initial_optimism"] = float(
            (initial_prediction - returns).mean()
        )
    network.train()
    return results


def main() -> None:
    args = tyro.cli(Args)
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("Direct-Outcome Sampled MuZero requires CUDA")
    if args.num_envs != 16:
        raise ValueError("v1 fixes num_envs=16 for synchronous generations")
    if args.latent_dim != 512 or args.unroll_steps != 16:
        raise ValueError("v1 fixes D=512 and K=16")
    if (
        args.root_candidates != 32
        or args.child_candidates != 8
        or args.simulations != 256
        or args.simulation_wave != 16
        or args.search_depth != 16
    ):
        raise ValueError("v1 fixes Croot=32, Cchild=8, S=256, W=16, H=16")
    if args.batch_size != 256:
        raise ValueError("v1 fixes batch_size=256")
    if args.policy_learning_rate <= 0.0:
        raise ValueError("policy_learning_rate must be positive")
    if args.min_log_std != -5.0 or args.max_log_std != 2.0:
        raise ValueError("v1 fixes the smooth policy log-standard-deviation range [-5,2]")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=asdict(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, args.capture_video, run_name) for index in range(args.num_envs)]
    )
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, False, f"{run_name}_eval") for index in range(args.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("continuous Box actions are required")
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    max_episode_steps = envs.envs[0].spec.max_episode_steps
    if max_episode_steps is None:
        raise RuntimeError("an exact finite episode horizon is required")
    if any(environment.spec.max_episode_steps != max_episode_steps for environment in envs.envs):
        raise RuntimeError("vector environments disagree on episode horizon")
    action_low = torch.as_tensor(envs.single_action_space.low, device=device)
    action_high = torch.as_tensor(envs.single_action_space.high, device=device)

    network = DirectOutcomeMuZero(obs_dim, action_dim, args).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    search_module = SampledMuZeroSearch(
        network,
        max_episode_steps,
        args.root_candidates,
        args.child_candidates,
        args.simulations,
        args.simulation_wave,
        args.search_depth,
        args.pb_c_init,
        args.pb_c_base,
    )

    def loss_call(batch: dict[str, torch.Tensor]) -> LossOutputs:
        return direct_outcome_loss(
            network,
            batch["obs0"],
            batch["remaining"],
            batch["actions"],
            batch["rewards"],
            batch["terminated"],
            batch["returns"],
            batch["transition_mask"],
            batch["state_mask"],
            max_episode_steps,
        )

    if args.compile:
        search_fn = torch.compile(search_module, fullgraph=True, mode=args.compile_mode)
        loss_fn = torch.compile(loss_call, fullgraph=True, mode=args.compile_mode)
    else:
        search_fn = search_module
        loss_fn = loss_call

    replay = SynchronousGenerationReplay(
        args.num_envs,
        obs_dim,
        action_dim,
        args.root_candidates,
        max_episode_steps,
    )
    behavior_generator = torch.Generator(device=device)
    behavior_generator.manual_seed(args.seed + 11_003)
    child_generator = torch.Generator(device=device)
    child_generator.manual_seed(args.seed + 19_019)
    selection_generator = torch.Generator(device=device)
    selection_generator.manual_seed(args.seed + 27_071)
    training_generator = torch.Generator(device=device)
    training_generator.manual_seed(args.seed + 39_013)

    reset_seeds = [args.seed + index for index in range(args.num_envs)]
    observations, _ = envs.reset(seed=reset_seeds)
    generation_id = 0
    replay.begin(generation_id)
    collection_parameter_version = 0
    parameter_version = 0
    optimizer_steps = 0
    policy_steps = 0
    generation_returns = np.zeros(args.num_envs, dtype=np.float64)
    environment_episode_steps = np.zeros(args.num_envs, dtype=np.int64)
    recent_returns: deque[float] = deque(maxlen=args.tail_window)
    last_loss: LossOutputs | None = None
    last_grad_norm = torch.zeros((), device=device)
    search_metric_sum = {
        "q_min": torch.zeros((), device=device),
        "q_max": torch.zeros((), device=device),
        "visit_entropy": torch.zeros((), device=device),
        "visit_effective_candidates": torch.zeros((), device=device),
        "visit_max_weight": torch.zeros((), device=device),
        "expansion_depth_mean": torch.zeros((), device=device),
        "expansion_depth_max": torch.zeros((), device=device),
        "action_saturation_fraction": torch.zeros((), device=device),
    }
    search_metric_count = 0
    start_time = time.time()
    global_step = 0

    while global_step < args.total_timesteps:
        if replay.phase != replay.COLLECT:
            raise RuntimeError("environment interaction escaped the COLLECT phase")
        if parameter_version != collection_parameter_version:
            raise RuntimeError("network parameters changed within a search generation")
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        remaining_np = max_episode_steps - environment_episode_steps
        remaining_tensor = torch.as_tensor(remaining_np, dtype=torch.float32, device=device)
        root_noise = torch.randn(
            (args.num_envs, args.root_candidates, action_dim),
            generator=behavior_generator,
            device=device,
        )
        child_noise = torch.randn(
            (
                args.num_envs,
                args.simulations,
                args.child_candidates,
                action_dim,
            ),
            generator=child_generator,
            device=device,
        )
        selection_uniform = torch.rand(
            args.num_envs, generator=selection_generator, device=device
        )
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            search = _clone_search_outputs(
                search_fn(obs_tensor, remaining_tensor, root_noise, child_noise)
            )
            assert_valid_search(search, args.simulations)
            visits = search.visit_history[-1]
            visit_weights = root_visit_weights(visits)
            chosen = select_root_candidate(visits, selection_uniform, deterministic=False)
            env_index = torch.arange(args.num_envs, device=device)
            normalized_action = search.root_normalized_actions[env_index, chosen]
            selected_predicted_q = search.q_history[-1][env_index, chosen]
        torch._assert_async(
            torch.isfinite(normalized_action).all(),
            "nonfinite search action reached environment execution",
        )

        weight_entropy = -(
            visit_weights
            * torch.where(
                visit_weights > 0,
                torch.log(visit_weights),
                torch.zeros_like(visit_weights),
            )
        ).sum(dim=-1)
        for key, values in (
            ("q_min", search.q_min),
            ("q_max", search.q_max),
            ("visit_entropy", weight_entropy),
            ("visit_effective_candidates", torch.exp(weight_entropy)),
            ("visit_max_weight", visit_weights.max(dim=-1).values),
            ("expansion_depth_mean", search.expansion_depth_mean),
            ("expansion_depth_max", search.expansion_depth_max),
            (
                "action_saturation_fraction",
                (normalized_action.abs() > 0.95).to(torch.float32).mean(dim=-1),
            ),
        ):
            search_metric_sum[key].add_(values.sum())
        search_metric_count += args.num_envs

        environment_action = map_normalized_action(
            normalized_action, action_low, action_high
        )
        transfer = pack_behavior_transfer(
            environment_action,
            normalized_action,
            search.root_pre_tanh,
            visit_weights,
            selected_predicted_q,
        )
        next_observations, rewards, terminated, truncated, infos = envs.step(
            transfer.environment_action
        )
        factual_next = factual_next_observations(
            next_observations, terminated, truncated, infos
        )
        active_before = replay.active.copy()
        replay.add_batch(
            observations,
            remaining_np.astype(np.float32),
            transfer.normalized_action,
            rewards,
            factual_next,
            terminated,
            truncated,
            transfer.root_pre_tanh,
            transfer.root_weights,
            transfer.selected_q,
        )
        generation_returns += np.asarray(rewards) * active_before
        environment_episode_steps += 1
        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        for env_index_np in np.flatnonzero(done & active_before):
            episode_return = float(generation_returns[env_index_np])
            recent_returns.append(episode_return)
            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar(
                "charts/episodic_length",
                int(max_episode_steps - remaining_np[env_index_np] + 1),
                global_step,
            )
        environment_episode_steps[done] = 0
        observations = next_observations
        global_step += args.num_envs

        if global_step % args.log_frequency == 0:
            torch.cuda.synchronize()
            if last_loss is not None:
                for name, value in zip(LossOutputs._fields, last_loss):
                    writer.add_scalar(f"loss_and_policy/{name}", value, global_step)
                writer.add_scalar("grad/global_norm", last_grad_norm, global_step)
            if search_metric_count:
                for name, value in search_metric_sum.items():
                    writer.add_scalar(
                        f"search/{name}", value / search_metric_count, global_step
                    )
                    value.zero_()
                search_metric_count = 0
            writer.add_scalar("updates/optimizer", optimizer_steps, global_step)
            writer.add_scalar("updates/policy", policy_steps, global_step)
            writer.add_scalar("updates/parameter_version", parameter_version, global_step)
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
            if recent_returns:
                ordered = np.sort(np.asarray(recent_returns))
                tail_count = max(1, math.ceil(0.1 * len(ordered)))
                writer.add_scalar("charts/return_mean", float(ordered.mean()), global_step)
                writer.add_scalar(
                    "charts/return_cvar10", float(ordered[:tail_count].mean()), global_step
                )
            print(
                f"step={global_step} sps={int(global_step / (time.time() - start_time))} "
                f"generation={generation_id} optimizer_steps={optimizer_steps}"
            )

        if is_evaluation_step(global_step):
            # Evaluation precedes a coincident generation TRAIN phase; in particular,
            # the 8M point measures the policy that actually collected the final data.
            evaluation = evaluate_inference_scaling(
                network,
                search_fn,
                eval_envs,
                args,
                obs_dim,
                action_dim,
                max_episode_steps,
                action_low,
                action_high,
                device,
            )
            for name, value in evaluation.items():
                writer.add_scalar(name, value, global_step)
            writer.add_scalar("eval/parameter_version", parameter_version, global_step)

        if replay.ready:
            if parameter_version != collection_parameter_version:
                raise RuntimeError("parameter version changed before generation finalization")
            if args.env_id == "HalfCheetah-v4":
                expected_transitions = args.num_envs * max_episode_steps
                if replay.completed_transitions != expected_transitions:
                    raise RuntimeError(
                        "HalfCheetah generation must contain exactly 16 complete 1000-step episodes"
                    )
                if replay.vector_steps != max_episode_steps:
                    raise RuntimeError("HalfCheetah generation must be exactly 1000 vector steps")
            writer.add_scalar(
                "generation/completed_transitions", replay.completed_transitions, global_step
            )
            writer.add_scalar("generation/vector_steps", replay.vector_steps, global_step)
            for name, value in replay.prequential_diagnostics().items():
                writer.add_scalar(name, value, global_step)
            updates_this_generation = replay.vector_steps
            replay.start_training(device)
            roots = replay.all_roots()
            fixed_unroll = replay.fixed_sequences(args.batch_size, args.unroll_steps)
            proposal_mean_before, proposal_log_std_before = proposal_predictions(
                network, roots, max_episode_steps
            )
            recurrent_mean_before, recurrent_log_std_before = recurrent_proposal_predictions(
                network, fixed_unroll, max_episode_steps
            )
            policy_metrics = apply_generation_policy_improvement(
                network,
                roots,
                max_episode_steps,
                args.policy_learning_rate,
                args.batch_size,
                args.max_grad_norm,
            )
            parameter_version += 1
            policy_steps += 1
            proposal_mean_after_policy, proposal_log_std_after_policy = proposal_predictions(
                network, roots, max_episode_steps
            )
            recurrent_mean_after_policy, recurrent_log_std_after_policy = (
                recurrent_proposal_predictions(network, fixed_unroll, max_episode_steps)
            )
            for name, value in zip(RootPolicyMetrics._fields, policy_metrics):
                writer.add_scalar(f"policy/{name}", value, global_step)
            writer.add_scalar(
                "policy/surrogate_collection",
                policy_metrics.centered_loss,
                global_step,
            )
            writer.add_scalar(
                "policy/surrogate_post_policy",
                evaluate_centered_policy_surrogate(
                    network, roots, max_episode_steps
                ),
                global_step,
            )
            writer.add_scalar(
                "policy/step_mean_absolute_drift",
                (proposal_mean_after_policy - proposal_mean_before).abs().mean(),
                global_step,
            )
            writer.add_scalar(
                "policy/step_logstd_absolute_drift",
                (proposal_log_std_after_policy - proposal_log_std_before).abs().mean(),
                global_step,
            )
            writer.add_scalar(
                "policy/kl_collection_to_post_policy",
                diagonal_gaussian_kl(
                    proposal_mean_before,
                    proposal_log_std_before,
                    proposal_mean_after_policy,
                    proposal_log_std_after_policy,
                ).mean(),
                global_step,
            )
            policy_shift_rms, policy_shift_abs = normalized_mean_shift(
                proposal_mean_before,
                proposal_log_std_before,
                proposal_mean_after_policy,
            )
            writer.add_scalar("policy/normalized_mean_shift_rms", policy_shift_rms, global_step)
            writer.add_scalar("policy/normalized_mean_shift_abs", policy_shift_abs, global_step)
            for _ in range(updates_this_generation):
                batch = replay.sample(args.batch_size, args.unroll_steps, training_generator)
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                loss_outputs = loss_fn(batch)
                torch._assert_async(
                    torch.stack([value.isfinite() for value in loss_outputs]).all(),
                    "nonfinite direct-outcome loss",
                )
                optimizer.zero_grad(set_to_none=True)
                loss_outputs.total.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    network.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer_steps += 1
                parameter_version += 1
                last_loss = LossOutputs(*(value.detach() for value in loss_outputs))
                last_grad_norm = grad_norm.detach()
            proposal_mean_after_outcome, proposal_log_std_after_outcome = proposal_predictions(
                network, roots, max_episode_steps
            )
            recurrent_mean_after_outcome, recurrent_log_std_after_outcome = (
                recurrent_proposal_predictions(network, fixed_unroll, max_episode_steps)
            )
            writer.add_scalar(
                "policy/surrogate_post_outcome",
                evaluate_centered_policy_surrogate(
                    network, roots, max_episode_steps
                ),
                global_step,
            )
            writer.add_scalar(
                "policy/outcome_phase_mean_absolute_drift",
                (proposal_mean_after_outcome - proposal_mean_after_policy).abs().mean(),
                global_step,
            )
            writer.add_scalar(
                "policy/outcome_phase_logstd_absolute_drift",
                (proposal_log_std_after_outcome - proposal_log_std_after_policy).abs().mean(),
                global_step,
            )
            for phase_name, old_mean, old_log_std, new_mean, new_log_std in (
                (
                    "post_policy_to_post_outcome",
                    proposal_mean_after_policy,
                    proposal_log_std_after_policy,
                    proposal_mean_after_outcome,
                    proposal_log_std_after_outcome,
                ),
                (
                    "collection_to_post_outcome",
                    proposal_mean_before,
                    proposal_log_std_before,
                    proposal_mean_after_outcome,
                    proposal_log_std_after_outcome,
                ),
            ):
                writer.add_scalar(
                    f"policy/kl_{phase_name}",
                    diagonal_gaussian_kl(
                        old_mean, old_log_std, new_mean, new_log_std
                    ).mean(),
                    global_step,
                )
                shift_rms, shift_abs = normalized_mean_shift(
                    old_mean, old_log_std, new_mean
                )
                writer.add_scalar(
                    f"policy/{phase_name}_normalized_mean_shift_rms",
                    shift_rms,
                    global_step,
                )
                writer.add_scalar(
                    f"policy/{phase_name}_normalized_mean_shift_abs",
                    shift_abs,
                    global_step,
                )
            recurrent_phases = (
                (
                    "collection_to_post_policy",
                    recurrent_mean_before,
                    recurrent_log_std_before,
                    recurrent_mean_after_policy,
                    recurrent_log_std_after_policy,
                ),
                (
                    "post_policy_to_post_outcome",
                    recurrent_mean_after_policy,
                    recurrent_log_std_after_policy,
                    recurrent_mean_after_outcome,
                    recurrent_log_std_after_outcome,
                ),
                (
                    "collection_to_post_outcome",
                    recurrent_mean_before,
                    recurrent_log_std_before,
                    recurrent_mean_after_outcome,
                    recurrent_log_std_after_outcome,
                ),
            )
            state_mask = fixed_unroll["state_mask"]
            for phase_name, old_mean, old_log_std, new_mean, new_log_std in recurrent_phases:
                recurrent_kl = diagonal_gaussian_kl(
                    old_mean, old_log_std, new_mean, new_log_std
                )
                for depth in range(args.unroll_steps + 1):
                    valid = state_mask[:, depth]
                    writer.add_scalar(
                        f"policy_recurrent/kl_{phase_name}_depth_{depth}",
                        (recurrent_kl[:, depth] * valid).sum() / valid.sum().clamp_min(1.0),
                        global_step,
                    )
                recurrent_valid = state_mask[:, 1:]
                writer.add_scalar(
                    f"policy_recurrent/kl_{phase_name}_mean_depth_1_{args.unroll_steps}",
                    (recurrent_kl[:, 1:] * recurrent_valid).sum()
                    / recurrent_valid.sum().clamp_min(1.0),
                    global_step,
                )
            if parameter_version - collection_parameter_version != updates_this_generation + 1:
                raise RuntimeError("generation optimizer-update accounting failed")
            replay.clear()
            # Every env is reset: ignored post-completion episodes must never leak across
            # the fixed-parameter generation boundary on early-terminating tasks.
            observations, _ = envs.reset()
            environment_episode_steps.fill(0)
            generation_returns.fill(0.0)
            generation_id += 1
            replay.begin(generation_id)
            collection_parameter_version = parameter_version

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save({"network": network.state_dict(), "args": asdict(args)}, model_path)
    envs.close()
    eval_envs.close()
    writer.close()


if __name__ == "__main__":
    main()

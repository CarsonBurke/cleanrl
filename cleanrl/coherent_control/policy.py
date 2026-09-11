"""Fixed-shape FP32 CUDA affine policies and raw native MuJoCo evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from cleanrl.collective_control.control import ObservationAdapter
from cleanrl.shared.rollout_graph import RolloutStepGraph, graph_compile


def _fixed_adapter(adapter: ObservationAdapter, observation_dim: int) -> ObservationAdapter:
    mean = np.array(adapter.mean, dtype=np.float32, copy=True)
    scale = np.array(adapter.scale, dtype=np.float32, copy=True)
    if mean.shape != (observation_dim,) or scale.shape != (observation_dim,):
        raise ValueError("adapter dimensions must match the observation space")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("adapter mean must be finite and scale finite and positive")
    mean.flags.writeable = scale.flags.writeable = False
    return ObservationAdapter(mean, scale)


@graph_compile
def _step(observation, parameters, mean, scale, low, high, blind):
    observation_dim = mean.shape[0]
    action_dim = low.shape[0]
    weights = parameters[:, :action_dim * observation_dim].reshape(-1, action_dim, observation_dim)
    bias = parameters[:, action_dim * observation_dim:]
    encoded = (torch.where(blind, mean, observation) - mean) / scale
    encoded = encoded.reshape(parameters.shape[0], -1, observation_dim)
    raw = (weights[:, None] * encoded[:, :, None]).sum(-1) + bias[:, None]
    action = (low + high) * 0.5 + (high - low) * 0.5 * raw
    return {"action": action.clamp(low, high).reshape(-1, action_dim)}


class _AffinePolicy:
    """A fixed population/episode shape with in-place parameter and blind reloads."""

    def __init__(self, parameters: np.ndarray, episodes: int, adapter: ObservationAdapter,
                 action_low: np.ndarray, action_high: np.ndarray, device: torch.device, *, blind: bool = False):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("coherent control requires CUDA; CPU fallback is disabled")
        self.observation_dim = len(adapter.mean)
        fixed = _fixed_adapter(adapter, self.observation_dim)
        low = np.asarray(action_low, dtype=np.float32)
        high = np.asarray(action_high, dtype=np.float32)
        if (low.ndim != 1 or not low.size or high.shape != low.shape
                or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))
                or np.any(high <= low)):
            raise ValueError("action bounds must be finite vectors with high greater than low")
        if episodes < 1:
            raise ValueError("episode count must be positive")
        self.action_dim = low.size
        self.parameter_dim = self.action_dim * (self.observation_dim + 1)
        values = self._validated_parameters(parameters)
        self.population, self.episodes = len(values), episodes
        self._parameters = torch.tensor(values, dtype=torch.float32, device=self.device)
        self._mean = torch.tensor(fixed.mean, dtype=torch.float32, device=self.device)
        self._scale = torch.tensor(fixed.scale, dtype=torch.float32, device=self.device)
        self._low = torch.tensor(low, dtype=torch.float32, device=self.device)
        self._high = torch.tensor(high, dtype=torch.float32, device=self.device)
        self._blind = torch.full((), blind, dtype=torch.bool, device=self.device)
        self._graph = None

    def _validated_parameters(self, parameters: np.ndarray) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != self.parameter_dim:
            raise ValueError(f"parameters must have shape (positive population, {self.parameter_dim})")
        if not np.all(np.isfinite(values)):
            raise ValueError("policy parameters must be finite FP32 values")
        return values

    @torch.inference_mode()
    def reload(self, parameters: np.ndarray, *, blind: bool = False) -> None:
        values = self._validated_parameters(parameters)
        if values.shape[0] != self.population:
            raise ValueError("reload must preserve population size")
        self._parameters.copy_(torch.from_numpy(values))
        self._blind.fill_(blind)

    def _policy(self, observation):
        return _step(observation, self._parameters, self._mean, self._scale,
                     self._low, self._high, self._blind)

    @torch.inference_mode()
    def _action_buffer(self, observations: np.ndarray) -> np.ndarray:
        """Borrowed pinned actions; consume before the next call, without retaining."""
        values = np.asarray(observations)
        environments = self.population * self.episodes
        if values.shape != (environments, self.observation_dim):
            raise ValueError("observations must match the population-major episode batch")
        if self._graph is None:
            self._graph = RolloutStepGraph(self._policy, 1, environments,
                                           (self.observation_dim,), self.device)
        return self._graph.step(values)

    def action(self, observations: np.ndarray) -> np.ndarray:
        """Return owned actions that remain unchanged across subsequent calls."""
        return self._action_buffer(observations).copy()


class AffineEvaluator:
    """Cache a native environment and compiled graph for each (population, seeds).

    Parameters are action-major weights followed by action biases. Seed columns
    are paired across policies; returned scores sum raw rewards through the first
    terminal transition, including that transition's reward. The fixed adapter is
    copied on construction and is never updated by evaluation.
    """

    def __init__(self, env_id: str, adapter: ObservationAdapter, device: torch.device, num_threads: int):
        from cleanrl.shared.mujoco_env import make_mujoco_vector_env

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("coherent control requires CUDA; CPU fallback is disabled")
        self.env_id, self.num_threads = env_id, num_threads
        self._slots = {}
        self._closed = False
        self.evaluation_transitions = 0
        # Retain the dimension environment for singleton evaluation, not a second
        # throwaway construction. It is owned and closed even if never stepped.
        self._dimension_env = make_mujoco_vector_env(
            env_id, 1, backend="native", num_threads=num_threads, copy=False)
        try:
            self.observation_dim = int(np.prod(self._dimension_env.single_observation_space.shape))
            self.action_dim = int(np.prod(self._dimension_env.single_action_space.shape))
            self.parameter_dim = self.action_dim * (self.observation_dim + 1)
            self.adapter = _fixed_adapter(adapter, self.observation_dim)
            self.action_low = np.array(self._dimension_env.single_action_space.low, dtype=np.float32, copy=True)
            self.action_high = np.array(self._dimension_env.single_action_space.high, dtype=np.float32, copy=True)
            self.action_low.flags.writeable = self.action_high.flags.writeable = False
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        if self._closed:
            raise RuntimeError("evaluator is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        """Release every native environment, even if one close raises."""
        environments = [env for env, _ in self._slots.values()]
        self._slots.clear()
        if self._dimension_env is not None:
            environments.append(self._dimension_env)
            self._dimension_env = None
        self._closed = True
        error = None
        for env in environments:
            try:
                env.close()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def evaluate(self, parameters: np.ndarray, seeds: Sequence[int], horizon: int, *, blind: bool = False) -> np.ndarray:
        """Evaluate P policies on E seeds, returning raw FP64 returns shaped (P, E)."""
        from cleanrl.shared.mujoco_env import make_mujoco_vector_env

        if self._closed:
            raise RuntimeError("evaluator is closed")
        try:
            values = np.asarray(parameters, dtype=np.float32)
            if (values.ndim != 2 or not values.shape[0] or values.shape[1] != self.parameter_dim
                    or not np.all(np.isfinite(values))):
                raise ValueError(f"parameters must be finite with shape (positive population, {self.parameter_dim})")
            if len(seeds) < 1 or horizon < 1:
                raise ValueError("evaluation requires seeds and a positive horizon")
            population, episodes = len(values), len(seeds)
            key = (population, episodes)
            if key not in self._slots:
                if key == (1, 1) and self._dimension_env is not None:
                    env, self._dimension_env = self._dimension_env, None
                else:
                    env = make_mujoco_vector_env(self.env_id, population * episodes, backend="native",
                                                 num_threads=self.num_threads, copy=False)
                try:
                    policy = _AffinePolicy(values, episodes, self.adapter,
                                           self.action_low, self.action_high, self.device, blind=blind)
                except BaseException:
                    env.close()
                    raise
                self._slots[key] = env, policy
            else:
                env, policy = self._slots[key]
                policy.reload(values, blind=blind)
            rollout_seeds = [int(seed) for _ in range(population) for seed in seeds]
            observation, _ = env.reset(seed=rollout_seeds)
            returns = np.zeros(population * episodes, dtype=np.float64)
            active = np.ones(population * episodes, dtype=bool)
            for _ in range(horizon):
                # Native stepping clips into its own buffer before the next graph
                # replay: no action alias escapes or needs a per-step copy here.
                action = policy._action_buffer(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                self.evaluation_transitions += population * episodes
                np.add(returns, reward, out=returns, where=active)
                active &= ~np.logical_or(terminated, truncated)
                if not np.any(active):
                    break
            return returns.reshape(population, episodes)
        except BaseException:
            self.close()
            raise

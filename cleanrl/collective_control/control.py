"""Compiled recurrent collectives with reusable native MuJoCo rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from cleanrl.shared.rollout_graph import RolloutStepGraph

from .genome import Genome


@dataclass
class ObservationAdapter:
    mean: np.ndarray
    scale: np.ndarray

    def to_json(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_json(cls, value: dict[str, list[float]]) -> "ObservationAdapter":
        return cls(np.asarray(value["mean"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32))


@torch.compile(fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
def _step(observation, mean, scale, low, high, sources, q, rate, outputs, authorities, state, previous, evolved):
    encoded = 0.5 + 0.5 * torch.tanh((observation - mean) / scale)
    values = torch.cat((encoded[:, None, :].expand(-1, state.shape[1], -1), state, previous,
                        torch.zeros_like(state[:, :, :1]), torch.full_like(state[:, :, :1], 0.5)), dim=2)
    left = values.gather(2, sources[:, :, :, 0])
    right = values.gather(2, sources[:, :, :, 1])
    q00, q01, q10, q11 = q.unbind(-1)
    mixed = ((1.0 - left) * (1.0 - right) * q00
             + (1.0 - left) * right * q01
             + left * (1.0 - right) * q10
             + left * right * q11)
    updated = (1.0 - rate) * state + rate * mixed
    gathered = updated.gather(2, outputs.clamp_min(0))
    normalized = torch.where(outputs.ge(0), gathered, 0.5)
    actions = low[None, None] + normalized * (high - low)[None, None]
    if evolved:
        weights = updated.gather(2, authorities.clamp_min(0)[:, :, None]).squeeze(2)
        weights = torch.where(authorities.ge(0), weights, 0.5)
        total_weight = torch.sum(weights, 1, keepdim=True)
        has_authority = total_weight > 0.0
        weighted_action = torch.sum(actions * weights[:, :, None], 1) / torch.where(
            has_authority, total_weight, 1.0
        )
        action = torch.where(has_authority, weighted_action, low + 0.5 * (high - low))
    else:
        action = actions.mean(1)
    state.copy_(updated)
    previous.copy_(normalized)
    return {"action": action.clamp(low, high)}


class CollectiveController:
    """Fixed-capacity recurrent controller; genome reloads retain CUDA graphs."""

    def __init__(self, genomes: Sequence[Genome] | Sequence[Sequence[Genome]], adapter: ObservationAdapter,
                 action_low: np.ndarray, action_high: np.ndarray, max_nodes: int,
                 authority: str, device: torch.device):
        if device.type != "cuda":
            raise ValueError("collective control requires CUDA; CPU fallback is disabled")
        if authority not in {"uniform", "evolved"}:
            raise ValueError(f"unknown authority mode: {authority}")
        self.device = device
        self.obs_mean = torch.as_tensor(adapter.mean, dtype=torch.float32, device=device)
        self.obs_scale = torch.as_tensor(adapter.scale, dtype=torch.float32, device=device)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        self.authority = authority
        self.action_dim = len(action_low)
        self.max_nodes = max_nodes
        self._parameters = {}
        self._bound = {}
        self._graph = None
        self.reload(genomes)
        self.reset()

    @torch.inference_mode()
    def reload(self, genomes: Sequence[Genome] | Sequence[Sequence[Genome]]) -> None:
        """Replace immutable team inputs; call reset before the next rollout."""
        if not genomes:
            raise ValueError("collective must have residents")
        teams = [list(genomes)] if isinstance(genomes[0], Genome) else [list(team) for team in genomes]
        residents = len(teams[0])
        if residents == 0 or any(len(team) != residents for team in teams):
            raise ValueError("batched teams must have equal, positive resident counts")
        if self._parameters and (len(teams), residents) != (len(self.teams), self.residents):
            raise ValueError("reload must preserve team and resident counts")
        nodes = self.max_nodes
        if any(not genome.nodes or len(genome.nodes) > nodes for team in teams for genome in team):
            raise ValueError("team genome exceeds configured node cap or is empty")
        self.teams, self.residents = teams, residents
        # Teams differ at one slot in the common case. Encode each shared genome
        # once, not once per candidate team. The cache lasts only this reload.
        unique = {}
        mapping = []
        for team in teams:
            row = []
            for genome in team:
                key = id(genome)
                if key not in unique:
                    unique[key] = (len(unique), genome)
                row.append(unique[key][0])
            mapping.append(row)
        count = len(unique)
        obs = self.obs_mean.numel()
        half = obs + nodes + self.action_dim + 1
        arrays = {
            "sources": np.full((count, nodes, 2), half, dtype=np.int64),
            "q": np.full((count, nodes, 4), 0.5, dtype=np.float32),
            "rate": np.zeros((count, nodes), dtype=np.float32),
            "initial": np.full((count, nodes), 0.5, dtype=np.float32),
            "outputs": np.full((count, self.action_dim), -1, dtype=np.int64),
            "authorities": np.full(count, -1, dtype=np.int64),
        }
        for index, genome in unique.values():
            positions = {node.node_id: i for i, node in enumerate(genome.nodes)}
            for i, node in enumerate(genome.nodes):
                arrays["q"][index, i] = node.q
                arrays["rate"][index, i] = node.update_rate
                arrays["initial"][index, i] = node.initial
                for side, source in enumerate(node.sources):
                    if source.kind == 0:
                        address = min(max(0, source.index), obs - 1)
                    elif source.kind == 1 and source.index in positions:
                        address = obs + positions[source.index]
                    elif source.kind == 2:
                        # Each resident reads its own previous normalized proposal,
                        # never the collective's executed action.
                        address = obs + nodes + min(max(0, source.index), self.action_dim - 1)
                    elif source.kind == 3:
                        address = half - 1
                    else:
                        address = half
                    arrays["sources"][index, i, side] = address
            arrays["outputs"][index] = [positions.get(node_id, -1) for node_id in genome.outputs]
            arrays["authorities"][index] = positions.get(genome.authority, -1)
        indices = np.asarray(mapping)
        for name, array in arrays.items():
            tensor = torch.as_tensor(array[indices], device=self.device)
            if name in self._parameters:
                self._parameters[name].copy_(tensor)
            else:
                self._parameters[name] = tensor

    @torch.inference_mode()
    def reset(self, environments: int = 1, team_indices: Sequence[int] | None = None) -> None:
        if team_indices is None:
            team_indices = [0] * environments
        if environments < 1 or len(team_indices) != environments:
            raise ValueError("team index count must equal positive environment count")
        if any(index < 0 or index >= len(self.teams) for index in team_indices):
            raise ValueError("team index is out of range")
        indices = torch.as_tensor(team_indices, dtype=torch.long, device=self.device)
        changed_shape = not self._bound or self.state.shape[0] != environments
        if changed_shape:
            self._graph = None
            self._bound = {name: value[indices] for name, value in self._parameters.items()}
            self.state = self._bound["initial"].clone()
            self.previous_action = torch.full((environments, self.residents, self.action_dim), 0.5, device=self.device)
        else:
            for name, value in self._parameters.items():
                self._bound[name].copy_(value[indices])
            self.state.copy_(self._bound["initial"])
            self.previous_action.fill_(0.5)

    def _policy(self, observation):
        p = self._bound
        return _step(observation, self.obs_mean, self.obs_scale, self.action_low, self.action_high,
                     p["sources"], p["q"], p["rate"], p["outputs"], p["authorities"],
                     self.state, self.previous_action, self.authority == "evolved")

    @torch.inference_mode()
    def _action_buffer(self, observation: np.ndarray) -> np.ndarray:
        """Borrowed action buffer, overwritten on the next step."""
        value = np.asarray(observation, dtype=np.float32)
        batched = value.ndim > 1
        if not batched:
            value = value[None, :]
        if value.shape != (self.state.shape[0], self.obs_mean.numel()):
            raise ValueError("observation shape must match reset environment count and adapter")
        if self._graph is None:
            self._graph = RolloutStepGraph(self._policy, 1, self.state.shape[0],
                                           (self.obs_mean.numel(),), self.device)
            # Capture executes the recurrence during warmup. The first real
            # observation must still see the pristine episode state.
            self.state.copy_(self._bound["initial"])
            self.previous_action.fill_(0.5)
        result = self._graph.step(value)
        return result if batched else result[0]

    def action(self, observation: np.ndarray) -> np.ndarray:
        return self._action_buffer(observation).copy()


def calibrate_observations(env_id: str, seeds: Iterable[int], steps: int, num_threads: int) -> ObservationAdapter:
    from cleanrl.shared.mujoco_env import make_mujoco_vector_env

    seed_values = [int(seed) for seed in seeds]
    env = make_mujoco_vector_env(env_id, len(seed_values), backend="native", num_threads=num_threads, copy=False)
    try:
        observation, _ = env.reset(seed=seed_values)
        samples = [np.asarray(observation, dtype=np.float32).copy()]
        rng = np.random.default_rng(0xC011EC71)
        low = np.asarray(env.single_action_space.low, dtype=np.float32)
        high = np.asarray(env.single_action_space.high, dtype=np.float32)
        for _ in range(steps):
            action = rng.uniform(low, high, size=(len(seed_values),) + env.single_action_space.shape).astype(np.float32)
            observation, _, terminated, truncated, _ = env.step(action)
            samples.append(np.asarray(observation, dtype=np.float32).copy())
            if np.all(np.logical_or(terminated, truncated)):
                break
    finally:
        env.close()
    values = np.concatenate(samples, axis=0)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.0)
    return ObservationAdapter(mean, scale)


class TeamEvaluator:
    """Own native environments and captured controllers across evaluations."""

    def __init__(self, env_id: str, adapter: ObservationAdapter, max_nodes: int,
                 authority: str, device: torch.device, num_threads: int):
        self.env_id, self.adapter, self.max_nodes = env_id, adapter, max_nodes
        self.authority, self.device, self.num_threads = authority, device, num_threads
        self._slots = {}
        # Count every simulated slot, including already-finished episodes.
        self.evaluation_transitions = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for env, _ in self._slots.values():
            env.close()
        self._slots.clear()

    def evaluate(self, teams: Sequence[Sequence[Genome]], seeds: Sequence[int], horizon: int) -> np.ndarray:
        from cleanrl.shared.mujoco_env import make_mujoco_vector_env

        if not teams or not seeds:
            raise ValueError("batched evaluation needs teams and seeds")
        key = (len(teams), len(seeds))
        if key not in self._slots:
            env = make_mujoco_vector_env(self.env_id, len(teams) * len(seeds), backend="native",
                                         num_threads=self.num_threads, copy=False)
            try:
                controller = CollectiveController(teams, self.adapter, env.single_action_space.low,
                                                  env.single_action_space.high, self.max_nodes, self.authority, self.device)
            except BaseException:
                env.close()
                raise
            self._slots[key] = env, controller
        else:
            env, controller = self._slots[key]
            controller.reload(teams)
        team_indices = np.repeat(np.arange(len(teams)), len(seeds)).tolist()
        rollout_seeds = np.tile(np.asarray(seeds, dtype=np.uint32), len(teams)).tolist()
        observation, _ = env.reset(seed=rollout_seeds)
        controller.reset(len(rollout_seeds), team_indices)
        returns = np.zeros(len(rollout_seeds), dtype=np.float64)
        active = np.ones(len(rollout_seeds), dtype=bool)
        for _ in range(horizon):
            action = controller._action_buffer(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            self.evaluation_transitions += len(rollout_seeds)
            np.add(returns, reward, out=returns, where=active)
            active &= ~np.logical_or(terminated, truncated)
            if not np.any(active):
                break
        return returns.reshape(len(teams), len(seeds))


def evaluate_teams(env_id: str, teams: Sequence[Sequence[Genome]], adapter: ObservationAdapter,
                   seeds: Sequence[int], horizon: int, max_nodes: int, authority: str,
                   device: torch.device, num_threads: int) -> np.ndarray:
    with TeamEvaluator(env_id, adapter, max_nodes, authority, device, num_threads) as evaluator:
        return evaluator.evaluate(teams, seeds, horizon)


def evaluate(env_id: str, genomes: list[Genome], adapter: ObservationAdapter, seeds: Sequence[int],
             horizon: int, max_nodes: int, authority: str, device: torch.device, num_threads: int) -> np.ndarray:
    return evaluate_teams(env_id, [genomes], adapter, seeds, horizon, max_nodes, authority, device, num_threads)[0]

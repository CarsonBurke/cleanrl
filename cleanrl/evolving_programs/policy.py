"""Signed synchronous arithmetic programs, compiled into one physical-step CUDA graph."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from cleanrl.collective_control.control import ObservationAdapter
from cleanrl.shared.rollout_graph import RolloutStepGraph

from .genome import Genome, Op, SourceKind, needed_nodes


@torch.compile(fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
def _step(observation, mean, scale, sources, ops, literal, active, outputs,
          state, previous_action, valid, ticks):
    encoded = torch.asinh((observation - mean) / scale)
    zero = torch.zeros_like(previous_action[:, :1])
    one = torch.ones_like(zero)
    current = state
    alive = valid
    # The physical feedback is deliberately fixed throughout the microticks.
    for _ in range(ticks):
        values = torch.cat((encoded, current, previous_action, literal, zero, one), dim=1)
        left = values.gather(1, sources[:, :, 0])
        right = values.gather(1, sources[:, :, 1])
        result = torch.where(ops == 1, left + right, left)
        result = torch.where(ops == 2, left - right, result)
        result = torch.where(ops == 3, left * right, result)
        # An unselected DIV is not an executed operation; in particular COPY(x,0)
        # must not be invalidated by an irrelevant zero denominator.
        denominator = torch.where(ops == 4, right, one)
        result = torch.where(ops == 4, left / denominator, result)
        result = torch.where(ops == 5, torch.tanh(left), result)
        result = torch.where(ops == 6, literal, result)
        alive = alive & (torch.isfinite(result) | ~active).all(dim=1)
        current = result
    raw_action = torch.cat((current, zero), dim=1).gather(1, outputs)
    alive = alive & torch.isfinite(raw_action).all(dim=1)
    action = torch.where(alive[:, None], raw_action.clamp(-1.0, 1.0), zero)
    state.copy_(current)
    previous_action.copy_(action)
    valid.copy_(alive)
    return {"action": action}


class ProgramController:
    """Fixed-capacity population parameters and independently bound lifetime lanes.

    The adapter sees 19 raw channels (including raw previous reward), followed by
    exactly one asinh here. Missing stable node IDs resolve to zero, not another
    node's current position. Only the output dependency closure can kill a lane.
    """

    def __init__(self, genomes: list[Genome], adapter: ObservationAdapter,
                 max_nodes: int, ticks: int, device: torch.device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("arithmetic programs require CUDA; CPU fallback is disabled")
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
            raise ValueError("ticks must be a positive integer")
        mean = np.array(adapter.mean, dtype=np.float32, copy=True)
        scale = np.array(adapter.scale, dtype=np.float32, copy=True)
        if mean.shape != (19,) or scale.shape != (19,):
            raise ValueError("arithmetic programs require a 19-channel adapter")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("adapter mean must be finite and scale finite and positive")
        self.max_nodes = max_nodes
        self.ticks = ticks
        self.action_dim = 6
        self.obs_mean = torch.as_tensor(mean, device=self.device)
        self.obs_scale = torch.as_tensor(scale, device=self.device)
        self._parameters = {}
        self._bound = {}
        self._graph = None
        self._genome_indices = (0,)
        self.reload(genomes)

    @torch.inference_mode()
    def reload(self, genomes: list[Genome]) -> None:
        """Reload equal-size populations in place and discard every old lifetime."""
        if not genomes:
            raise ValueError("population must not be empty")
        if self._parameters and len(genomes) != len(self.genomes):
            raise ValueError("reload must preserve population size")
        for genome in genomes:
            genome.validate(19, self.action_dim, self.max_nodes)
        population, capacity = len(genomes), self.max_nodes
        literal_start = 19 + capacity + self.action_dim
        zero = literal_start + capacity
        arrays = {
            "sources": np.full((population, capacity, 2), zero, dtype=np.int64),
            "ops": np.full((population, capacity), int(Op.CONST), dtype=np.int64),
            "literal": np.zeros((population, capacity), dtype=np.float32),
            "initial": np.zeros((population, capacity), dtype=np.float32),
            "active": np.zeros((population, capacity), dtype=np.bool_),
            "outputs": np.full((population, self.action_dim), capacity, dtype=np.int64),
        }
        for row, genome in enumerate(genomes):
            positions = {node.node_id: index for index, node in enumerate(genome.nodes)}
            active = needed_nodes(genome)
            for index, node in enumerate(genome.nodes):
                arrays["ops"][row, index] = node.op
                arrays["literal"][row, index] = node.literal
                arrays["initial"][row, index] = node.initial
                arrays["active"][row, index] = node.node_id in active
                consumed = 0 if node.op == Op.CONST else 1 if node.op in (Op.COPY, Op.TANH) else 2
                for side in range(consumed):
                    source = node.sources[side]
                    if source.kind == SourceKind.OBS:
                        address = source.index
                    elif source.kind == SourceKind.NODE:
                        address = 19 + positions[source.index] if source.index in positions else zero
                    elif source.kind == SourceKind.PREVIOUS_ACTION:
                        address = 19 + capacity + source.index
                    elif source.kind == SourceKind.LITERAL:
                        address = literal_start + index
                    elif source.kind == SourceKind.ONE:
                        address = zero + 1
                    else:
                        address = zero
                    arrays["sources"][row, index, side] = address
            arrays["outputs"][row] = [positions.get(node_id, capacity) for node_id in genome.outputs]
        for name, array in arrays.items():
            tensor = torch.as_tensor(array, device=self.device)
            if name in self._parameters:
                self._parameters[name].copy_(tensor)
            else:
                self._parameters[name] = tensor
        self.genomes = list(genomes)
        self.reset(len(self._genome_indices), self._genome_indices)

    @torch.inference_mode()
    def reset(self, environments: int = 1, genome_indices: Sequence[int] | None = None) -> None:
        """Bind each lane to its genome's germline, with no acquired state carry."""
        if isinstance(environments, bool) or not isinstance(environments, int) or environments < 1:
            raise ValueError("environment count must be a positive integer")
        mapping = (0,) * environments if genome_indices is None else tuple(genome_indices)
        if len(mapping) != environments:
            raise ValueError("genome index count must equal environment count")
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
               or index < 0 or index >= len(self.genomes) for index in mapping):
            raise ValueError("genome index is out of range")
        indices = torch.as_tensor(mapping, dtype=torch.long, device=self.device)
        if not self._bound or self.state.shape[0] != environments:
            self._graph = None
            self._bound = {name: value[indices] for name, value in self._parameters.items()}
            self.state = self._bound["initial"].clone()
            self.previous_action = torch.zeros((environments, self.action_dim), dtype=torch.float32, device=self.device)
            self.valid = torch.ones(environments, dtype=torch.bool, device=self.device)
        else:
            for name, value in self._parameters.items():
                self._bound[name].copy_(value[indices])
            self.state.copy_(self._bound["initial"])
            self.previous_action.zero_()
            self.valid.fill_(True)
        self._genome_indices = mapping
        if self._graph is not None:
            self._graph.reset()

    def _policy(self, observation):
        p = self._bound
        return _step(observation, self.obs_mean, self.obs_scale, p["sources"], p["ops"],
                     p["literal"], p["active"], p["outputs"], self.state,
                     self.previous_action, self.valid, self.ticks)

    @torch.inference_mode()
    def _ensure_graph(self) -> None:
        if self._graph is not None:
            return
        state = self.state.clone()
        previous_action = self.previous_action.clone()
        valid = self.valid.clone()
        try:
            self._graph = RolloutStepGraph(self._policy, 1, self.state.shape[0], (19,), self.device)
        finally:
            # Warmup executes real recurrences. Restore the exact acquired state,
            # not just germline: interventions may capture after support/replay.
            self.state.copy_(state)
            self.previous_action.copy_(previous_action)
            self.valid.copy_(valid)

    @torch.inference_mode()
    def _action_buffer(self, observation: np.ndarray) -> np.ndarray:
        """Borrowed pinned actions; consume before the next physical step."""
        values = np.asarray(observation, dtype=np.float32)
        unbatched = values.ndim == 1
        if unbatched:
            values = values[None, :]
        if values.shape != (self.state.shape[0], 19):
            raise ValueError("observation shape must match reset environment count and adapter")
        self._ensure_graph()
        actions = self._graph.step(values)
        return actions[0] if unbatched else actions

    def action(self, observation: np.ndarray) -> np.ndarray:
        """Owned actions remain unchanged across subsequent steps."""
        return self._action_buffer(observation).copy()


def _capture_preserving_state(controller: ProgramController) -> None:
    controller._ensure_graph()

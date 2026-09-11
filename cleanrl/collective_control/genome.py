"""Recurrent policy genomes mutated without PPO, SGD, or other gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Source:
    kind: int  # 0 observation, 1 node, 2 resident's previous normalized proposal, 3 zero, 4 half
    index: int = 0

    def to_json(self) -> dict[str, int]:
        return {"kind": self.kind, "index": self.index}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Source":
        return cls(int(value["kind"]), int(value.get("index", 0)))


@dataclass
class Node:
    node_id: int
    sources: tuple[Source, Source]
    q: np.ndarray
    update_rate: float
    initial: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "sources": [source.to_json() for source in self.sources],
            "q": self.q.tolist(),
            "update_rate": self.update_rate,
            "initial": self.initial,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Node":
        return cls(
            int(value["id"]),
            (Source.from_json(value["sources"][0]), Source.from_json(value["sources"][1])),
            np.asarray(value["q"], dtype=np.float32),
            float(value["update_rate"]),
            float(value["initial"]),
        )


@dataclass
class Genome:
    nodes: list[Node]
    outputs: list[int]
    next_id: int
    authority: int | None = None

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        observation_dim: int,
        action_dim: int,
        node_count: int,
        evolved_authority: bool,
    ) -> "Genome":
        count = max(1, int(node_count))
        nodes: list[Node] = []
        for node_id in range(count):
            nodes.append(
                Node(
                    node_id=node_id,
                    sources=(
                        random_source(rng, observation_dim, action_dim, range(count)),
                        random_source(rng, observation_dim, action_dim, range(count)),
                    ),
                    q=np.clip(rng.normal(0.5, 0.1, 4).astype(np.float32), 0.0, 1.0),
                    update_rate=float(rng.random()),
                    initial=float(np.clip(rng.normal(0.5, 0.05), 0.0, 1.0)),
                )
            )
        return cls(
            nodes=nodes,
            outputs=[int(rng.integers(count)) for _ in range(action_dim)],
            next_id=count,
            authority=int(rng.integers(count)) if evolved_authority else None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_json() for node in self.nodes],
            "outputs": self.outputs,
            "next_id": self.next_id,
            "authority": self.authority,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Genome":
        return cls(
            nodes=[Node.from_json(node) for node in value["nodes"]],
            outputs=[int(output) for output in value["outputs"]],
            next_id=int(value["next_id"]),
            authority=None if value.get("authority") is None else int(value["authority"]),
        )

    def clone(self) -> "Genome":
        return Genome(
            nodes=[
                Node(
                    node.node_id,
                    (Source(node.sources[0].kind, node.sources[0].index), Source(node.sources[1].kind, node.sources[1].index)),
                    node.q.copy(),
                    node.update_rate,
                    node.initial,
                )
                for node in self.nodes
            ],
            outputs=self.outputs.copy(),
            next_id=self.next_id,
            authority=self.authority,
        )

    def mutate(
        self,
        rng: np.random.Generator,
        observation_dim: int,
        action_dim: int,
        max_nodes: int,
        events: float,
        length_probability: float,
        evolved_authority: bool,
    ) -> None:
        count = int(events) + int(rng.random() < (events - int(events)))
        for _ in range(count):
            field_count = len(self.nodes) * 8 + action_dim + int(evolved_authority)
            field = int(rng.integers(max(1, field_count)))
            node_fields = len(self.nodes) * 8
            if field < node_fields:
                node_index, field_index = divmod(field, 8)
                node = self.nodes[node_index]
                if field_index < 2:
                    sources = list(node.sources)
                    sources[field_index] = random_source(rng, observation_dim, action_dim, (item.node_id for item in self.nodes))
                    node.sources = (sources[0], sources[1])
                elif field_index < 6:
                    node.q[field_index - 2] = mutate_probability(rng, float(node.q[field_index - 2]))
                elif field_index == 6:
                    node.update_rate = mutate_probability(rng, node.update_rate)
                else:
                    node.initial = mutate_probability(rng, node.initial)
            elif field < node_fields + action_dim:
                self.outputs[field - node_fields] = int(rng.choice(self.nodes).node_id)
            elif evolved_authority:
                self.authority = int(rng.choice(self.nodes).node_id)

        if rng.random() < length_probability:
            if rng.random() < 0.5 and len(self.nodes) < max_nodes:
                source = self.nodes[int(rng.integers(len(self.nodes)))]
                self.nodes.append(
                    Node(
                        self.next_id,
                        (Source(source.sources[0].kind, source.sources[0].index), Source(source.sources[1].kind, source.sources[1].index)),
                        source.q.copy(),
                        source.update_rate,
                        source.initial,
                    )
                )
                self.next_id += 1
            elif len(self.nodes) > 1:
                del self.nodes[int(rng.integers(len(self.nodes)))]

    def validate(self, action_dim: int, max_nodes: int, evolved_authority: bool) -> None:
        if not self.nodes:
            raise ValueError("genome must contain a node")
        if len(self.nodes) > max_nodes:
            raise ValueError("genome exceeds node cap")
        if len(self.outputs) != action_dim:
            raise ValueError("action output dimension mismatch")
        ids = {node.node_id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate node identity")
        if evolved_authority and self.authority is None:
            raise ValueError("evolved authority is missing")
        for node in self.nodes:
            if node.q.shape != (4,) or not np.all(np.isfinite(node.q)):
                raise ValueError("invalid truth table")
            if not np.all((node.q >= 0.0) & (node.q <= 1.0)):
                raise ValueError("truth table outside [0, 1]")
            if not 0.0 <= node.update_rate <= 1.0 or not 0.0 <= node.initial <= 1.0:
                raise ValueError("invalid state parameter")


def random_source(rng: np.random.Generator, observation_dim: int, action_dim: int, node_ids: Any) -> Source:
    node_ids = tuple(node_ids)
    choice = int(rng.integers(observation_dim + len(node_ids) + action_dim + 2))
    if choice < observation_dim:
        return Source(0, choice)
    choice -= observation_dim
    if choice < len(node_ids):
        return Source(1, node_ids[choice])
    choice -= len(node_ids)
    if choice < action_dim:
        return Source(2, choice)
    return Source(3 + choice - action_dim)


def mutate_probability(rng: np.random.Generator, value: float) -> float:
    if rng.random() < 0.25:
        return float(rng.random())
    value += (1.0 if rng.random() < 0.5 else -1.0) * 10.0 ** (-4.0 * rng.random())
    if value < 0.0:
        value = -value
    elif value > 1.0:
        value = 2.0 - value
    return float(np.clip(value, 0.0, 1.0))

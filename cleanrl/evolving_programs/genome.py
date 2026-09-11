"""Unseeded recurrent arithmetic graphs with heritable, stable node identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral, Real
from typing import Any, Iterable

import numpy as np


class SourceKind(IntEnum):
    OBS = 0
    NODE = 1
    PREVIOUS_ACTION = 2
    LITERAL = 3
    ZERO = 4
    ONE = 5


class Op(IntEnum):
    COPY = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    TANH = 5
    CONST = 6


MAX_NODE_ID = int(np.iinfo(np.int64).max)
_MAX_FLOAT32 = float(np.finfo(np.float32).max)


@dataclass
class Source:
    kind: int
    index: int = 0

    def to_json(self) -> dict[str, int]:
        return {"kind": int(self.kind), "index": int(self.index)}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Source":
        return cls(value["kind"], value.get("index", 0))


@dataclass
class Node:
    node_id: int
    op: int
    sources: tuple[Source, Source]
    literal: float
    initial: float

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": int(self.node_id),
            "op": int(self.op),
            "sources": [source.to_json() for source in self.sources],
            "literal": float(self.literal),
            "initial": float(self.initial),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Node":
        sources = value["sources"]
        if len(sources) != 2:
            raise ValueError("arithmetic nodes require exactly two source slots")
        return cls(
            value["node_id"], value["op"],
            (Source.from_json(sources[0]), Source.from_json(sources[1])),
            value["literal"], value["initial"],
        )

    def clone(self) -> "Node":
        return Node(
            self.node_id, self.op,
            (Source(self.sources[0].kind, self.sources[0].index), Source(self.sources[1].kind, self.sources[1].index)),
            self.literal, self.initial,
        )


@dataclass
class Genome:
    nodes: list[Node]
    outputs: list[int]
    next_id: int

    @classmethod
    def random(
        cls, rng: np.random.Generator, observation_dim: int, action_dim: int, node_count: int,
    ) -> "Genome":
        _integer(observation_dim, 1, MAX_NODE_ID, "observation dimension")
        _integer(action_dim, 1, MAX_NODE_ID, "action dimension")
        _integer(node_count, 1, MAX_NODE_ID, "node count")
        node_ids = tuple(range(node_count))
        nodes = [
            Node(
                node_id, int(rng.integers(len(Op))),
                (random_source(rng, observation_dim, action_dim, node_ids),
                 random_source(rng, observation_dim, action_dim, node_ids)),
                _float32(rng.normal(0.0, 1.0)), _float32(rng.normal(0.0, 0.1)),
            )
            for node_id in node_ids
        ]
        return cls(nodes, [int(rng.integers(node_count)) for _ in range(action_dim)], node_count)

    def clone(self) -> "Genome":
        return Genome([node.clone() for node in self.nodes], self.outputs.copy(), self.next_id)

    def to_json(self) -> dict[str, Any]:
        return {"nodes": [node.to_json() for node in self.nodes], "outputs": [int(output) for output in self.outputs],
                "next_id": int(self.next_id)}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Genome":
        return cls([Node.from_json(node) for node in value["nodes"]], list(value["outputs"]), value["next_id"])

    def validate(self, observation_dim: int, action_dim: int, max_nodes: int) -> None:
        _integer(observation_dim, 1, MAX_NODE_ID, "observation dimension")
        _integer(action_dim, 1, MAX_NODE_ID, "action dimension")
        _integer(max_nodes, 1, MAX_NODE_ID, "node capacity")
        if not 1 <= len(self.nodes) <= max_nodes:
            raise ValueError("genome must contain between one and max_nodes nodes")
        if len(self.outputs) != action_dim:
            raise ValueError("action output dimension mismatch")
        _integer(self.next_id, 1, MAX_NODE_ID, "next node identity")
        ids: set[int] = set()
        for node in self.nodes:
            _integer(node.node_id, 0, self.next_id - 1, "node identity")
            if node.node_id in ids:
                raise ValueError("duplicate node identity")
            ids.add(node.node_id)
            _integer(node.op, 0, len(Op) - 1, "arithmetic opcode")
            _finite_parameter(node.literal)
            _finite_parameter(node.initial)
            if len(node.sources) != 2:
                raise ValueError("arithmetic nodes require exactly two source slots")
            for source in node.sources:
                _integer(source.kind, 0, len(SourceKind) - 1, "source kind")
                if source.kind == SourceKind.OBS:
                    upper = observation_dim - 1
                elif source.kind == SourceKind.PREVIOUS_ACTION:
                    upper = action_dim - 1
                elif source.kind == SourceKind.NODE:
                    # Historical (deleted) IDs remain valid zero-valued references.
                    # Future IDs are forbidden: births must remain disconnected.
                    upper = self.next_id - 1
                else:
                    upper = 0
                _integer(source.index, 0, upper, "source index")
        for output in self.outputs:
            _integer(output, 0, self.next_id - 1, "output identity")

    def mutate(
        self, rng: np.random.Generator, observation_dim: int, action_dim: int,
        max_nodes: int, events: float, length_probability: float,
    ) -> None:
        """Apply independent point events, then at most one requested length event.

        Each of the five slots per node and each motor root has equal probability.
        Pass events=0 for an isolated structural proposal; duplication never rewires
        existing nodes, including self references, or recruits its new identity.
        """
        self.validate(observation_dim, action_dim, max_nodes)
        if not isinstance(events, Real) or not np.isfinite(events) or events < 0:
            raise ValueError("point mutation events must be finite and nonnegative")
        if not isinstance(length_probability, Real) or not np.isfinite(length_probability) or not 0 <= length_probability <= 1:
            raise ValueError("length probability must be in [0, 1]")
        count = int(events)
        remainder = events - count
        if remainder > 0:
            count += int(rng.random() < remainder)
        node_ids = tuple(node.node_id for node in self.nodes)
        for _ in range(count):
            field = int(rng.integers(len(self.nodes) * 5 + action_dim))
            if field >= len(self.nodes) * 5:
                self.outputs[field - len(self.nodes) * 5] = node_ids[int(rng.integers(len(node_ids)))]
                continue
            node_index, slot = divmod(field, 5)
            node = self.nodes[node_index]
            if slot == 0:
                node.op = int(rng.integers(len(Op)))
            elif slot in (1, 2):
                sources = list(node.sources)
                sources[slot - 1] = random_source(rng, observation_dim, action_dim, node_ids)
                node.sources = (sources[0], sources[1])
            elif slot == 3:
                node.literal = mutate_real(rng, node.literal)
            else:
                node.initial = mutate_real(rng, node.initial)
        if length_probability == 0 or rng.random() >= length_probability:
            return
        if rng.random() < 0.5:
            if len(self.nodes) >= max_nodes:
                return
            if self.next_id >= MAX_NODE_ID:
                raise ValueError("node identity space exhausted")
            duplicate = self.nodes[int(rng.integers(len(self.nodes)))].clone()
            duplicate.node_id = self.next_id
            self.nodes.append(duplicate)
            self.next_id += 1
        elif len(self.nodes) > 1:
            del self.nodes[int(rng.integers(len(self.nodes)))]


def _integer(value: Any, minimum: int, maximum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")


def _finite_parameter(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or abs(value) > _MAX_FLOAT32:
        raise ValueError("arithmetic parameters must be finite float32 values")


def _float32(value: float) -> float:
    _finite_parameter(value)
    return float(np.float32(value))


def random_source(
    rng: np.random.Generator, observation_dim: int, action_dim: int, node_ids: Iterable[int],
) -> Source:
    """Sample uniformly over individual channels/IDs and the three constants."""
    node_ids = tuple(node_ids)
    choice = int(rng.integers(observation_dim + len(node_ids) + action_dim + 3))
    if choice < observation_dim:
        return Source(SourceKind.OBS, choice)
    choice -= observation_dim
    if choice < len(node_ids):
        return Source(SourceKind.NODE, node_ids[choice])
    choice -= len(node_ids)
    if choice < action_dim:
        return Source(SourceKind.PREVIOUS_ACTION, choice)
    return Source(int(SourceKind.LITERAL) + choice - action_dim)


def mutate_real(rng: np.random.Generator, value: float) -> float:
    """Unbounded signed proposals; reject overflow instead of clipping/repairing."""
    _finite_parameter(value)
    if rng.random() < 0.25:
        return _float32(rng.normal(0.0, 1.0))
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return _float32(value + sign * 10.0 ** (-4.0 * rng.random()))


def consumed_sources(node: Node) -> tuple[Source, ...]:
    if node.op == Op.CONST:
        return ()
    if node.op in (Op.COPY, Op.TANH):
        return node.sources[:1]
    return node.sources


def needed_nodes(genome: Genome) -> set[int]:
    """Transitive live dependencies from every motor root, including cycles."""
    nodes = {node.node_id: node for node in genome.nodes}
    pending = list(genome.outputs)
    needed: set[int] = set()
    while pending:
        node_id = pending.pop()
        if node_id in needed or node_id not in nodes:
            continue
        needed.add(node_id)
        pending.extend(source.index for source in consumed_sources(nodes[node_id]) if source.kind == SourceKind.NODE)
    return needed


def same_genotype(left: Genome, right: Genome) -> bool:
    return left == right


def same_computation(left: Genome, right: Genome) -> bool:
    """Conservative structural proof, not a behavioral or fitness admission gate."""
    if left.outputs != right.outputs:
        return False
    active = needed_nodes(left)
    if active != needed_nodes(right):
        return False
    left_nodes = {node.node_id: node for node in left.nodes}
    right_nodes = {node.node_id: node for node in right.nodes}
    for node_id in active:
        a, b = left_nodes[node_id], right_nodes[node_id]
        if a.op != b.op or a.initial != b.initial or consumed_sources(a) != consumed_sources(b):
            return False
        if a.op == Op.CONST or any(source.kind == SourceKind.LITERAL for source in consumed_sources(a)):
            if a.literal != b.literal:
                return False
    return True

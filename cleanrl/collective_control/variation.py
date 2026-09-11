"""Exact neutrality proofs and unmixed structural/point mutation proposals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genome import Genome, Node


def _same_node(left: Node, right: Node) -> bool:
    return (
        left.node_id == right.node_id
        and left.sources == right.sources
        and np.array_equal(left.q, right.q)
        and left.update_rate == right.update_rate
        and left.initial == right.initial
    )


def same_genotype(left: Genome, right: Genome) -> bool:
    """Compare all inherited fields, including latent nodes and allocator state."""
    return (
        left.outputs == right.outputs
        and left.next_id == right.next_id
        and left.authority == right.authority
        and len(left.nodes) == len(right.nodes)
        and all(_same_node(a, b) for a, b in zip(left.nodes, right.nodes))
    )


def same_computation(left: Genome, right: Genome) -> bool:
    """Prove identical recurrence for every input under a uniform observer.

    Stable IDs, not graph isomorphism, define correspondence. Every output is
    a root: previous-proposal sources can couple any action back into the graph.
    Missing IDs resolve to half in the controller, but a newly present ID must
    not be mistaken for an unchanged dangling reference. Unreachable nodes,
    allocator state, and unused authority do not enter this proof.
    """
    if left.outputs != right.outputs:
        return False
    left_nodes = {node.node_id: node for node in left.nodes}
    right_nodes = {node.node_id: node for node in right.nodes}
    if len(left_nodes) != len(left.nodes) or len(right_nodes) != len(right.nodes):
        return False
    pending = list(left.outputs)
    visited: set[int] = set()
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        a, b = left_nodes.get(node_id), right_nodes.get(node_id)
        if a is None or b is None:
            if a is not b:
                return False
            continue
        if not _same_node(a, b):
            return False
        pending.extend(source.index for source in a.sources if source.kind == 1)
    return True


@dataclass
class Proposal:
    genome: Genome
    operator: str
    neutral: bool


def propose(
    parent: Genome,
    rng: np.random.Generator,
    observation_dim: int,
    action_dim: int,
    max_nodes: int,
    events: float,
    candidates: int,
) -> list[Proposal]:
    """Return genotype-distinct offspring; every eighth attempt is structural.

    Structural attempts never bundle point mutations. No-ops and repeated
    offspring are omitted rather than resampled, so impossible moves do not
    consume an unbounded number of attempts or bias the neutral reservoir.
    """
    proposals: list[Proposal] = []
    for index in range(candidates):
        structural = index % 8 == 0
        child = parent.clone()
        child.mutate(
            rng,
            observation_dim,
            action_dim,
            max_nodes=max_nodes,
            events=0.0 if structural else events,
            length_probability=1.0 if structural else 0.0,
            evolved_authority=False,
        )
        if same_genotype(parent, child) or any(same_genotype(item.genome, child) for item in proposals):
            continue
        proposals.append(Proposal(child, "structure" if structural else "point", same_computation(parent, child)))
    return proposals

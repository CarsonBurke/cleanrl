import numpy as np
import pytest

from cleanrl.collective_control.genome import Genome, Node, Source
from cleanrl.collective_control.variation import propose, same_computation, same_genotype


def _genome():
    return Genome(
        [
            Node(10, (Source(1, 20), Source(0, 0)), np.array([0.1, 0.8, 0.3, 0.6], dtype=np.float32), 0.7, 0.2),
            Node(20, (Source(1, 10), Source(2, 1)), np.array([0.9, 0.2, 0.7, 0.4], dtype=np.float32), 0.6, 0.8),
            Node(30, (Source(3), Source(4)), np.array([0.2, 0.3, 0.8, 0.9], dtype=np.float32), 0.9, 0.4),
        ],
        [10, 20],
        31,
    )


def _trace(genome, observations):
    """Independent scalar reference for a single uniform-observer organism."""
    state = {node.node_id: node.initial for node in genome.nodes}
    previous = np.full(len(genome.outputs), 0.5)
    result = []
    for observation in observations:
        encoded = 0.5 + 0.5 * np.tanh(observation)

        def value(source):
            if source.kind == 0:
                return encoded[np.clip(source.index, 0, len(encoded) - 1)]
            if source.kind == 1:
                return state.get(source.index, 0.5)
            if source.kind == 2:
                return previous[np.clip(source.index, 0, len(previous) - 1)]
            return 0.0 if source.kind == 3 else 0.5

        updated = {}
        for node in genome.nodes:
            a, b = map(value, node.sources)
            q00, q01, q10, q11 = node.q
            mixed = (1 - a) * (1 - b) * q00 + (1 - a) * b * q01 + a * (1 - b) * q10 + a * b * q11
            updated[node.node_id] = (1 - node.update_rate) * state[node.node_id] + node.update_rate * mixed
        state = updated
        previous = np.array([state.get(node_id, 0.5) for node_id in genome.outputs])
        result.append(previous)
    return np.asarray(result)


def test_neutral_duplicate_can_accumulate_latent_edits_before_recruitment():
    parent = _genome()
    child = parent.clone()
    duplicate = child.clone().nodes[0]
    duplicate.node_id = child.next_id
    child.next_id += 1
    child.nodes.append(duplicate)
    assert same_computation(parent, child)
    assert not same_genotype(parent, child)

    duplicate.q[:] = 0.95
    duplicate.initial = 0.9
    observations = np.array([[-2.0], [0.0], [1.0], [-0.5], [3.0]])
    assert same_computation(parent, child)
    np.testing.assert_array_equal(_trace(parent, observations), _trace(child, observations))

    recruited = child.clone()
    recruited.nodes[1].sources = (Source(1, duplicate.node_id), Source(2, 1))
    assert not same_computation(child, recruited)
    assert not np.array_equal(_trace(child, observations), _trace(recruited, observations))


@pytest.mark.parametrize("field", ["sources", "q", "update_rate", "initial"])
def test_active_recurrent_node_changes_are_not_proven_equivalent(field):
    parent = _genome()
    child = parent.clone()
    node = child.nodes[1]
    if field == "sources":
        node.sources = (Source(1, 10), Source(2, 0))
    elif field == "q":
        node.q[0] = np.nextafter(node.q[0], np.float32(1.0))
    else:
        setattr(node, field, np.nextafter(getattr(node, field), 1.0))
    assert not same_computation(parent, child)
    assert not same_computation(child, parent)
    assert not same_genotype(parent, child)


def test_all_output_roots_include_previous_proposal_feedback_dependencies():
    parent = _genome()
    parent.nodes[0].sources = (Source(2, 1), Source(0, 0))
    parent.nodes[1].sources = (Source(3), Source(4))
    child = parent.clone()
    child.nodes[1].q[:] = 0.0
    observations = np.zeros((4, 1))
    before, after = _trace(parent, observations), _trace(child, observations)
    assert before[0, 0] == after[0, 0]
    assert before[1, 0] != after[1, 0]
    assert not same_computation(parent, child)


def test_missing_references_are_neutral_only_while_missing_on_both_sides():
    parent = _genome()
    parent.nodes[0].sources = (Source(1, 31), Source(0, 0))
    child = parent.clone()
    child.nodes.pop()
    assert same_computation(parent, child)

    activated = parent.clone()
    activated.nodes[-1].node_id = 31
    activated.next_id = 32
    assert not same_computation(parent, activated)
    assert not same_computation(activated, parent)
    observations = np.zeros((3, 1))
    assert not np.array_equal(_trace(parent, observations), _trace(activated, observations))


def test_missing_and_changed_output_roots_are_not_silently_ignored():
    parent = _genome()
    parent.outputs = [10, 99]
    child = parent.clone()
    child.nodes[1].q[:] = 0.0
    # Node 20 remains reachable through node 10 even with a missing second root.
    assert not same_computation(parent, child)
    child = parent.clone()
    child.outputs[1] = 98
    assert not same_computation(parent, child)
    child = parent.clone()
    child.nodes[-1].node_id = 99
    assert not same_computation(parent, child)
    child = parent.clone()
    child.nodes = [node for node in child.nodes if node.node_id != 10]
    assert not same_computation(parent, child)
    assert same_computation(parent, parent.clone())


def test_mutable_initial_state_changes_the_first_action():
    parent = _genome()
    child = parent.clone()
    child.nodes[0].initial = 0.9
    assert not same_computation(parent, child)
    assert _trace(parent, np.zeros((1, 1)))[0, 0] != _trace(child, np.zeros((1, 1)))[0, 0]


@pytest.mark.parametrize("field", ["next_id", "authority", "latent_q", "node_order"])
def test_full_inherited_genotype_is_distinct_from_uniform_computation(field):
    parent = _genome()
    child = parent.clone()
    assert same_genotype(parent, child)
    if field == "next_id":
        child.next_id += 1
    elif field == "authority":
        child.authority = 30
    elif field == "latent_q":
        child.nodes[-1].q[0] = 0.7
    else:
        child.nodes.reverse()
    assert same_computation(parent, child)
    assert not same_genotype(parent, child)


class _FirstChoiceRng:
    """Deterministic duplicate births and repeated point edits, without mocks."""

    def __init__(self):
        self.structural_source = 0

    def random(self):
        return 0.0

    def integers(self, high):
        if high == 3:
            result = self.structural_source % 3
            self.structural_source += 1
            return result
        return 0


def test_structural_births_do_not_bundle_points_and_repeat_every_eighth_attempt():
    parent = _genome()
    original = parent.clone()
    proposals = propose(parent, _FirstChoiceRng(), 1, 2, 8, events=2.0, candidates=17)
    assert [item.operator for item in proposals] == ["structure", "point", "structure", "structure"]
    assert same_genotype(parent, original)
    for item in proposals:
        if item.operator == "point":
            assert len(item.genome.nodes) == len(parent.nodes)
            assert not item.neutral
            continue
        assert item.neutral
        assert len(item.genome.nodes) == len(parent.nodes) + 1
        inherited = item.genome.clone()
        inherited.nodes.pop()
        inherited.next_id = parent.next_id
        assert same_genotype(parent, inherited)
        # Duplicates are independent inherited state, not aliases into the parent.
        item.genome.nodes[-1].q[:] = 0.0
        assert same_genotype(parent, original)
    for index, item in enumerate(proposals):
        assert not same_genotype(parent, item.genome)
        assert all(not same_genotype(item.genome, other.genome) for other in proposals[:index])


def test_impossible_structure_and_zero_event_points_return_no_clones():
    parent = Genome([_genome().nodes[0]], [10, 10], 11)
    assert propose(parent, _FirstChoiceRng(), 1, 2, 1, events=0.0, candidates=17) == []

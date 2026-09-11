import json

import numpy as np
import pytest

from cleanrl.evolving_programs.genome import (
    MAX_NODE_ID,
    Genome,
    Node,
    Op,
    Source,
    SourceKind,
    mutate_real,
    needed_nodes,
    random_source,
    same_computation,
    same_genotype,
)


class _Rng:
    """Choose exact proposal paths without depending on a seeded PRNG's sequence."""

    def __init__(self, integers=(), random=(), normal=()):
        self._integers = iter(integers)
        self._random = iter(random)
        self._normal = iter(normal)

    def integers(self, high):
        value = next(self._integers)
        assert 0 <= value < high
        return value

    def random(self):
        return next(self._random)

    def normal(self, mean, scale):
        return next(self._normal)


def _genome():
    return Genome(
        [
            Node(10, Op.ADD, (Source(SourceKind.NODE, 10), Source(SourceKind.LITERAL)), -0.125, 0.25),
            Node(20, Op.COPY, (Source(SourceKind.PREVIOUS_ACTION, 0), Source(SourceKind.NODE, 30)), 2.0, -0.5),
            Node(30, Op.CONST, (Source(SourceKind.NODE, 30), Source(SourceKind.ONE)), 0.75, 0.125),
        ],
        [10, 20],
        31,
    )


def _trace(genome, observations):
    """Test-only synchronous scalar reference; four ticks per physical action."""
    state = {node.node_id: node.initial for node in genome.nodes}
    previous = np.zeros(len(genome.outputs))
    result = []
    for observation in observations:
        encoded = np.arcsinh(observation)
        for _ in range(4):
            updated = {}
            for node in genome.nodes:
                def value(source):
                    if source.kind == SourceKind.OBS:
                        return encoded[source.index]
                    if source.kind == SourceKind.NODE:
                        return state.get(source.index, 0.0)
                    if source.kind == SourceKind.PREVIOUS_ACTION:
                        return previous[source.index]
                    if source.kind == SourceKind.LITERAL:
                        return node.literal
                    return float(source.kind == SourceKind.ONE)

                if node.op == Op.CONST:
                    output = node.literal
                else:
                    a = value(node.sources[0])
                    if node.op == Op.COPY:
                        output = a
                    elif node.op == Op.TANH:
                        output = np.tanh(a)
                    else:
                        b = value(node.sources[1])
                        if node.op == Op.ADD:
                            output = a + b
                        elif node.op == Op.SUB:
                            output = a - b
                        elif node.op == Op.MUL:
                            output = a * b
                        else:
                            output = a / b
                updated[node.node_id] = output
            state = updated
        previous = np.clip([state.get(node_id, 0.0) for node_id in genome.outputs], -1.0, 1.0)
        result.append(previous.copy())
    return np.asarray(result)


def test_arithmetic_json_roundtrip_preserves_ops_signed_parameters_and_identity():
    genome = _genome()
    for op in Op:
        genome.nodes.append(Node(genome.next_id, op, (Source(SourceKind.OBS), Source(SourceKind.ONE)), -2.75, 4.125))
        genome.next_id += 1
    restored = Genome.from_json(json.loads(json.dumps(genome.to_json(), allow_nan=False)))
    restored.validate(1, 2, 10)
    assert same_genotype(genome, restored)
    assert restored.to_json() == genome.to_json()
    restored.nodes[0].sources[0].index = 30
    restored.outputs[1] = 30
    assert genome.nodes[0].sources[0].index == 10
    assert genome.outputs[1] == 20


def test_active_dependency_walk_ignores_unused_operands_and_terminates_cycles():
    genome = _genome()
    assert needed_nodes(genome) == {10, 20}
    genome.nodes[1].op = Op.TANH
    assert needed_nodes(genome) == {10, 20}
    genome.nodes[1].op = Op.MUL
    assert needed_nodes(genome) == {10, 20, 30}
    genome.nodes[0].op = Op.CONST
    genome.nodes[0].sources = (Source(SourceKind.NODE, 30), Source(SourceKind.NODE, 30))
    genome.outputs = [10, 29]
    assert needed_nodes(genome) == {10}


def test_all_motor_roots_matter_through_previous_action_feedback():
    genome = _genome()
    changed = genome.clone()
    changed.nodes[0].literal = 0.0625
    before = _trace(genome, np.zeros((3, 1)))
    after = _trace(changed, np.zeros((3, 1)))
    assert before[0, 1] == after[0, 1] == 0
    assert before[1, 1] != after[1, 1]
    assert not same_computation(genome, changed)


def test_isolated_duplicate_preserves_original_program_until_recruitment():
    genome = _genome()
    duplicate = genome.clone()
    duplicate.mutate(_Rng(integers=[0], random=[0.0, 0.0]), 1, 2, 4, events=0, length_probability=1)
    assert duplicate.nodes[:-1] == genome.nodes
    assert duplicate.outputs == genome.outputs
    assert duplicate.nodes[-1].sources[0].index == 10  # A self edge is not retargeted.
    assert duplicate.nodes[-1].node_id == 31
    assert same_computation(genome, duplicate)
    assert not same_genotype(genome, duplicate)
    duplicate.nodes[-1].literal = 0.25
    duplicate.nodes[-1].initial = -0.125
    observations = np.zeros((3, 1))
    np.testing.assert_array_equal(_trace(genome, observations), _trace(duplicate, observations))
    duplicate.outputs[0] = duplicate.nodes[-1].node_id
    assert not same_computation(genome, duplicate)
    assert not np.array_equal(_trace(genome, observations), _trace(duplicate, observations))


def test_deletion_leaves_zero_dangling_sources_and_outputs_without_id_reuse():
    genome = _genome()
    genome.nodes[1].sources = (Source(SourceKind.NODE, 10), Source(SourceKind.ONE))
    genome.mutate(_Rng(integers=[0], random=[0.0, 0.9]), 1, 2, 4, events=0, length_probability=1)
    genome.validate(1, 2, 4)
    assert genome.outputs == [10, 20]
    assert genome.nodes[0].sources[0].index == 10
    np.testing.assert_array_equal(_trace(genome, np.zeros((3, 1))), np.zeros((3, 2)))
    genome.mutate(_Rng(integers=[1], random=[0.0, 0.0]), 1, 2, 4, events=0, length_probability=1)
    assert genome.nodes[-1].node_id == 31
    assert genome.next_id == 32
    np.testing.assert_array_equal(_trace(genome, np.zeros((3, 1))), np.zeros((3, 2)))


def test_ignored_fields_are_neutral_but_active_literal_and_initial_are_not():
    genome = _genome()
    changed = genome.clone()
    changed.nodes[1].sources = (changed.nodes[1].sources[0], Source(SourceKind.NODE, 10))
    changed.nodes[1].literal = -9.0
    changed.nodes[2].literal = 20.0
    assert same_computation(genome, changed)
    changed.nodes[0].literal = 0.5
    assert not same_computation(genome, changed)
    changed = genome.clone()
    changed.nodes[0].initial = 0.5
    assert not same_computation(genome, changed)
    assert not np.array_equal(_trace(genome, np.zeros((1, 1))), _trace(changed, np.zeros((1, 1))))


def test_each_channel_node_and_constant_has_one_uniform_source_choice():
    choices = [random_source(_Rng(integers=[index]), 2, 2, [10, 30]) for index in range(9)]
    assert choices == [
        Source(SourceKind.OBS, 0), Source(SourceKind.OBS, 1),
        Source(SourceKind.NODE, 10), Source(SourceKind.NODE, 30),
        Source(SourceKind.PREVIOUS_ACTION, 0), Source(SourceKind.PREVIOUS_ACTION, 1),
        Source(SourceKind.LITERAL), Source(SourceKind.ZERO), Source(SourceKind.ONE),
    ]


def test_real_mutation_is_signed_unbounded_and_float32_finite():
    assert mutate_real(_Rng(random=[0.5, 0.0, 0.0]), 2.0) == 3.0
    assert mutate_real(_Rng(random=[0.5, 0.9, 0.0]), -2.0) == -3.0
    assert mutate_real(_Rng(random=[0.0], normal=[-7.0]), 0.5) == -7.0
    assert mutate_real(_Rng(random=[0.5, 0.0, 0.5]), 0.0) == float(np.float32(0.01))
    with pytest.raises(ValueError):
        mutate_real(_Rng(random=[0.0], normal=[1e40]), 0.0)


@pytest.mark.parametrize("field,value", [("literal", float("nan")), ("initial", float("inf")), ("literal", 1e40), ("op", 7), ("node_id", -1)])
def test_invalid_node_parameters_are_rejected(field, value):
    genome = _genome()
    setattr(genome.nodes[0], field, value)
    with pytest.raises(ValueError):
        genome.validate(1, 2, 4)


@pytest.mark.parametrize("source", [Source(6), Source(SourceKind.OBS, 1), Source(SourceKind.PREVIOUS_ACTION, 2), Source(SourceKind.NODE, 31), Source(SourceKind.ONE, 1)])
def test_invalid_sources_are_rejected_even_in_ignored_slots(source):
    genome = _genome()
    genome.nodes[2].sources = (source, Source(SourceKind.ZERO))
    with pytest.raises(ValueError):
        genome.validate(1, 2, 4)


def test_identity_validation_prevents_collisions_and_future_reference_recruitment():
    genome = _genome()
    genome.nodes[1].node_id = 10
    with pytest.raises(ValueError):
        genome.validate(1, 2, 4)
    genome = _genome()
    genome.next_id = 30
    with pytest.raises(ValueError):
        genome.validate(1, 2, 4)
    genome = _genome()
    genome.outputs[0] = 31
    with pytest.raises(ValueError):
        genome.validate(1, 2, 4)
    genome = _genome()
    genome.next_id = MAX_NODE_ID
    with pytest.raises(ValueError):
        genome.mutate(_Rng(integers=[0], random=[0.0, 0.0]), 1, 2, 4, events=0, length_probability=1)
    assert genome.next_id == MAX_NODE_ID
    assert len(genome.nodes) == 3

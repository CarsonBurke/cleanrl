"""Constructed circuits are test fixtures only; production genomes remain unseeded."""

import numpy as np
import pytest
import torch

from cleanrl.collective_control.control import ObservationAdapter
from cleanrl.evolving_programs.genome import Genome, Node, Op, Source, SourceKind
from cleanrl.evolving_programs.policy import ProgramController, _capture_preserving_state


Z = Source(SourceKind.ZERO)
ONE = Source(SourceKind.ONE)
LITERAL = Source(SourceKind.LITERAL)


def _node(node_id, op, left=Z, right=Z, literal=0.0, initial=0.0):
    return Node(node_id, op, (left, right), literal, initial)


def _genome(nodes, outputs=None):
    outputs = [nodes[-1].node_id] * 6 if outputs is None else outputs
    references = [node.node_id for node in nodes] + list(outputs)
    references += [source.index for node in nodes for source in node.sources if source.kind == SourceKind.NODE]
    return Genome(nodes, outputs, max(references) + 1)


def _adapter():
    return ObservationAdapter(np.zeros(19, dtype=np.float32), np.ones(19, dtype=np.float32))


@pytest.fixture
def controller_factory():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Dynamo caches specializations process-wide, outside each controller's life.
    torch._dynamo.reset()
    controllers = []

    def create(genomes, capacity=None, ticks=4, adapter=None):
        controller = ProgramController(genomes, _adapter() if adapter is None else adapter,
                                       capacity or max(len(genome.nodes) for genome in genomes),
                                       ticks, torch.device("cuda"))
        controllers.append(controller)
        return controller

    try:
        yield create
    finally:
        controllers.clear()
        torch._dynamo.reset()


class _ScalarReference:
    """Independent float32 scalar interpreter, including synchronous node clocks."""

    def __init__(self, genome, adapter, ticks):
        self.genome, self.adapter, self.ticks = genome, adapter, ticks
        self.state = {node.node_id: np.float32(node.initial) for node in genome.nodes}
        self.previous = np.zeros(6, dtype=np.float32)
        self.valid = True
        nodes = {node.node_id: node for node in genome.nodes}
        self.active = set()
        pending = list(genome.outputs)
        while pending:
            node_id = pending.pop()
            if node_id in self.active or node_id not in nodes:
                continue
            self.active.add(node_id)
            node = nodes[node_id]
            sources = () if node.op == Op.CONST else node.sources[:1] if node.op in (Op.COPY, Op.TANH) else node.sources
            pending.extend(source.index for source in sources if source.kind == SourceKind.NODE)

    def action(self, observation):
        encoded = np.arcsinh((np.asarray(observation, dtype=np.float32) - self.adapter.mean) / self.adapter.scale)

        def source_value(source, node):
            if source.kind == SourceKind.OBS:
                return encoded[source.index]
            if source.kind == SourceKind.NODE:
                return self.state.get(source.index, np.float32(0))
            if source.kind == SourceKind.PREVIOUS_ACTION:
                return self.previous[source.index]
            if source.kind == SourceKind.LITERAL:
                return np.float32(node.literal)
            return np.float32(source.kind == SourceKind.ONE)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for _ in range(self.ticks):
                updated = {}
                for node in self.genome.nodes:
                    if node.op == Op.CONST:
                        value = np.float32(node.literal)
                    else:
                        left = source_value(node.sources[0], node)
                        if node.op == Op.COPY:
                            value = left
                        elif node.op == Op.TANH:
                            value = np.tanh(left)
                        else:
                            right = source_value(node.sources[1], node)
                            if node.op == Op.ADD:
                                value = left + right
                            elif node.op == Op.SUB:
                                value = left - right
                            elif node.op == Op.MUL:
                                value = left * right
                            else:
                                value = left / right
                    updated[node.node_id] = np.float32(value)
                    if node.node_id in self.active and not np.isfinite(value):
                        self.valid = False
                self.state = updated
        raw = np.asarray([self.state.get(node_id, 0) for node_id in self.genome.outputs], dtype=np.float32)
        self.valid = self.valid and bool(np.isfinite(raw).all())
        self.previous = np.clip(raw, -1, 1) if self.valid else np.zeros(6, dtype=np.float32)
        return self.previous.copy()


def test_cpu_policy_fallback_is_rejected():
    genome = _genome([_node(0, Op.CONST)])
    with pytest.raises(ValueError, match="CUDA"):
        ProgramController([genome], _adapter(), 1, 4, torch.device("cpu"))


@pytest.mark.cuda
def test_scalar_reference_signed_arithmetic_and_population_isolation(controller_factory):
    nodes = [
        _node(10, Op.COPY, Source(SourceKind.OBS, 0)),
        _node(20, Op.ADD, Source(SourceKind.NODE, 10), LITERAL, literal=-0.125),
        _node(30, Op.SUB, Source(SourceKind.NODE, 20), Source(SourceKind.PREVIOUS_ACTION, 0)),
        _node(40, Op.MUL, Source(SourceKind.NODE, 30), LITERAL, literal=0.5),
        _node(50, Op.DIV, Source(SourceKind.NODE, 40), LITERAL, literal=1.25),
        _node(60, Op.TANH, Source(SourceKind.NODE, 50)),
        _node(70, Op.CONST, literal=-1.75),
    ]
    first = _genome(nodes, [10, 20, 30, 40, 60, 70])
    second = _genome([_node(90, Op.MUL, Source(SourceKind.NODE, 90), Source(SourceKind.OBS, 1), initial=0.75)])
    adapter = ObservationAdapter(np.linspace(-0.2, 0.2, 19, dtype=np.float32),
                                 np.linspace(0.75, 1.5, 19, dtype=np.float32))
    controller = controller_factory([first, second], capacity=10, adapter=adapter)
    mapping = [1, 0, 1, 0]
    controller.reset(4, mapping)
    references = [_ScalarReference([first, second][index], adapter, 4) for index in mapping]
    rng = np.random.default_rng(71)
    for _ in range(12):
        observations = rng.uniform(-0.8, 0.8, (4, 19)).astype(np.float32)
        expected = np.stack([reference.action(obs) for reference, obs in zip(references, observations)])
        np.testing.assert_allclose(controller.action(observations), expected, rtol=2e-5, atol=2e-6)
        states = controller.state.cpu().numpy()
        for lane, reference in enumerate(references):
            np.testing.assert_allclose(states[lane, :len(reference.genome.nodes)],
                                       [reference.state[node.node_id] for node in reference.genome.nodes],
                                       rtol=2e-5, atol=2e-6)
    assert controller.valid.cpu().tolist() == [True] * 4


@pytest.mark.cuda
@pytest.mark.parametrize("ticks", [1, 4])
def test_synchronous_reaction_latency_and_cross_observation_carry(controller_factory, ticks):
    nodes = [_node(10, Op.COPY, Source(SourceKind.OBS, 0))]
    nodes += [_node(10 * (index + 1), Op.COPY, Source(SourceKind.NODE, 10 * index))
              for index in range(1, 2 * ticks + 1)]
    genome = _genome(list(reversed(nodes)), [nodes[-1].node_id] * 6)
    controller = controller_factory([genome], ticks=ticks)
    outputs = []
    for encoded in (0.125, -0.25, 0.5, 0.75):
        observation = np.zeros(19, dtype=np.float32)
        observation[0] = np.sinh(np.float32(encoded))
        outputs.append(controller.action(observation)[0])
    np.testing.assert_allclose(outputs, [0, 0, 0.125, -0.25], atol=2e-7)


@pytest.mark.cuda
def test_exact_add_carry_and_motor_only_clamping(controller_factory):
    genome = _genome([_node(7, Op.ADD, Source(SourceKind.NODE, 7), LITERAL, literal=0.125, initial=-0.25)])
    controller = controller_factory([genome])
    outputs = [controller.action(np.zeros(19, dtype=np.float32))[0] for _ in range(4)]
    np.testing.assert_array_equal(outputs, [0.25, 0.75, 1.0, 1.0])
    assert controller.state.cpu().item() == 1.75


@pytest.mark.cuda
def test_previous_action_is_signed_and_fixed_through_microticks(controller_factory):
    genome = _genome([_node(7, Op.ADD, Source(SourceKind.PREVIOUS_ACTION, 0), LITERAL, literal=0.125)])
    controller = controller_factory([genome])
    with torch.inference_mode():
        controller.previous_action.fill_(-0.5)
    outputs = [controller.action(np.zeros(19, dtype=np.float32))[0] for _ in range(4)]
    np.testing.assert_array_equal(outputs, [-0.375, -0.25, -0.125, 0.0])


@pytest.mark.cuda
def test_division_and_overflow_death_are_permanent_and_lane_local(controller_factory):
    division = _genome([
        _node(10, Op.COPY, Source(SourceKind.OBS, 0), initial=0.5),
        _node(20, Op.DIV, ONE, Source(SourceKind.NODE, 10)),
    ])
    overflow = _genome([_node(40, Op.MUL, Source(SourceKind.NODE, 40), Source(SourceKind.NODE, 40), initial=2)])
    healthy = _genome([_node(90, Op.CONST, literal=-0.25)])
    controller = controller_factory([division, overflow, healthy], capacity=3)
    controller.reset(3, [0, 1, 2])
    observations = np.zeros((3, 19), dtype=np.float32)
    observations[0, 0] = 0.5
    np.testing.assert_array_equal(controller.action(observations)[:, 0], [1, 1, -0.25])
    assert controller.valid.cpu().tolist() == [True, True, True]
    observations[0, 0] = 0
    np.testing.assert_array_equal(controller.action(observations)[:, 0], [0, 0, -0.25])
    assert controller.valid.cpu().tolist() == [False, False, True]
    observations[0, 0] = 0.5
    np.testing.assert_array_equal(controller.action(observations)[:, 0], [0, 0, -0.25])
    assert controller.valid.cpu().tolist() == [False, False, True]
    controller.reset(3, [0, 1, 2])
    np.testing.assert_array_equal(controller.action(observations)[:, 0], [1, 1, -0.25])


@pytest.mark.cuda
def test_ignored_sources_unreachable_overflow_and_missing_ids_are_harmless(controller_factory):
    bad = Source(SourceKind.NODE, 50)
    nodes = [
        _node(50, Op.MUL, bad, bad, initial=1e20),
        _node(20, Op.COPY, LITERAL, bad, literal=-0.375),
        _node(30, Op.TANH, Z, bad),
        _node(40, Op.CONST, bad, bad, literal=0.625),
        _node(70, Op.ADD, Source(SourceKind.NODE, 99), ONE),
    ]
    genome = _genome(nodes, [20, 30, 40, 70, 99, 20])
    controller = controller_factory([genome], capacity=8)
    for _ in range(3):
        np.testing.assert_array_equal(controller.action(np.zeros(19, dtype=np.float32)),
                                      [-0.375, 0, 0.625, 1, 0, -0.375])
        assert controller.valid.cpu().item()


@pytest.mark.cuda
def test_capture_preserves_acquired_state_feedback_and_invalid_flags(controller_factory):
    genome = _genome([_node(2, Op.ADD, Source(SourceKind.NODE, 2), Source(SourceKind.PREVIOUS_ACTION, 0))])
    controller = controller_factory([genome], capacity=3)
    controller.reset(2, [0, 0])
    with torch.inference_mode():
        controller.state.fill_(-0.25)
        controller.previous_action.fill_(0.125)
        controller.valid[1] = False
    _capture_preserving_state(controller)
    np.testing.assert_array_equal(controller.state.cpu().numpy(), np.full((2, 3), -0.25))
    np.testing.assert_array_equal(controller.previous_action.cpu().numpy(), np.full((2, 6), 0.125))
    assert controller.valid.cpu().tolist() == [True, False]
    np.testing.assert_array_equal(controller.action(np.zeros((2, 19), dtype=np.float32))[:, 0], [0.25, 0])
    assert controller.valid.cpu().tolist() == [True, False]


@pytest.mark.cuda
def test_reset_remapping_reload_and_owned_actions(controller_factory):
    first = _genome([_node(1, Op.ADD, Source(SourceKind.NODE, 1), LITERAL, literal=0.125)])
    second = _genome([_node(8, Op.ADD, Source(SourceKind.NODE, 8), LITERAL, literal=-0.0625, initial=-0.125)])
    controller = controller_factory([first, second], capacity=4)
    controller.reset(2, [0, 1])
    observations = np.zeros((2, 19), dtype=np.float32)
    owned = controller.action(observations)
    np.testing.assert_array_equal(owned[:, 0], [0.5, -0.375])
    controller.action(observations)
    np.testing.assert_array_equal(owned[:, 0], [0.5, -0.375])
    controller.reset(2, [1, 0])
    np.testing.assert_array_equal(controller.action(observations)[:, 0], [-0.375, 0.5])
    controller.reload([second, first])
    np.testing.assert_array_equal(controller.action(observations)[:, 0], [0.5, -0.375])
    controller.reset()
    np.testing.assert_array_equal(controller.action(np.zeros(19, dtype=np.float32)), [-0.375] * 6)
    with pytest.raises(ValueError, match="population size"):
        controller.reload([first])
    with pytest.raises(ValueError, match="index"):
        controller.reset(1, [2])
    with pytest.raises(ValueError, match="count"):
        controller.reset(2, [0])
    with pytest.raises(ValueError, match="shape"):
        controller.action(np.zeros((2, 19), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        controller.action(np.zeros(18, dtype=np.float32))


@pytest.mark.cuda
def test_graph_capture_failure_is_propagated_without_eager_fallback(controller_factory, monkeypatch):
    import cleanrl.evolving_programs.policy as policy

    genome = _genome([_node(1, Op.CONST, literal=0.5)])
    controller = controller_factory([genome])

    def failed_capture(*args, **kwargs):
        with torch.inference_mode():
            controller.state.fill_(9)
            controller.previous_action.fill_(9)
            controller.valid.fill_(False)
        raise RuntimeError("capture unavailable")

    monkeypatch.setattr(policy, "RolloutStepGraph", failed_capture)
    with pytest.raises(RuntimeError, match="capture unavailable"):
        controller.action(np.zeros(19, dtype=np.float32))
    np.testing.assert_array_equal(controller.state.cpu().numpy(), [[0]])
    np.testing.assert_array_equal(controller.previous_action.cpu().numpy(), np.zeros((1, 6)))
    assert controller.valid.cpu().item()

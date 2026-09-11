import numpy as np
import pytest
import torch

from cleanrl.collective_control.control import CollectiveController, ObservationAdapter
from cleanrl.collective_control.genome import Genome, Node, Source, random_source


def _constant(node_id, value):
    return Node(node_id, (Source(4), Source(4)), np.full(4, value, dtype=np.float32), 1.0, value)


def _copy_source(node_id, source):
    return Node(node_id, (source, Source(4)), np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32), 1.0, 0.5)


def _controller(genomes, action_dim, authority="uniform", low=None, high=None):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by the controller")
    return CollectiveController(
        genomes,
        ObservationAdapter(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)),
        np.full(action_dim, -1.0, dtype=np.float32) if low is None else low,
        np.full(action_dim, 1.0, dtype=np.float32) if high is None else high,
        max_nodes=8,
        authority=authority,
        device=torch.device("cuda"),
    )


@pytest.mark.cuda
def test_observation_address_survives_disconnected_lower_id_deletion():
    genome = Genome([_constant(0, 0.1), _copy_source(1, Source(0, 1))], [1], 2)
    controller = _controller([genome], 1)
    observation = np.array([-2.0, 2.0], dtype=np.float32)
    before = controller.action(observation)
    np.testing.assert_allclose(before, [np.tanh(2.0)], atol=1e-6)

    smaller = genome.clone()
    del smaller.nodes[0]
    controller.reload([smaller])
    controller.reset()
    # The old decoder changed +0.964 to -0.964 by mapping sensor 1 to node slot 0.
    np.testing.assert_allclose(controller.action(observation), before, atol=1e-6)


@pytest.mark.cuda
def test_previous_proposal_address_survives_disconnected_lower_id_deletion():
    genome = Genome(
        [_constant(0, 0.1), _constant(1, 0.2), _constant(2, 0.8), _copy_source(3, Source(2, 1))],
        [1, 2, 3],
        4,
    )
    controller = _controller([genome], 3)
    observation = np.zeros(2, dtype=np.float32)
    np.testing.assert_allclose(controller.action(observation), [-0.6, 0.6, 0.0], atol=1e-6)
    before = controller.action(observation)
    np.testing.assert_allclose(before, [-0.6, 0.6, 0.6], atol=1e-6)

    smaller = genome.clone()
    del smaller.nodes[0]
    controller.reload([smaller])
    controller.reset()
    controller.action(observation)
    np.testing.assert_allclose(controller.action(observation), before, atol=1e-6)


@pytest.mark.cuda
def test_node_sources_follow_stable_ids_and_deleted_references_are_neutral():
    genome = Genome([_constant(0, 0.1), _constant(7, 0.8), _copy_source(9, Source(1, 7))], [9], 10)
    controller = _controller([genome], 1)
    observation = np.zeros(2, dtype=np.float32)
    np.testing.assert_allclose(controller.action(observation), [0.6], atol=1e-6)

    smaller = genome.clone()
    del smaller.nodes[0]
    smaller.nodes.reverse()
    controller.reload([smaller])
    controller.reset()
    np.testing.assert_allclose(controller.action(observation), [0.6], atol=1e-6)

    smaller.nodes = [node for node in smaller.nodes if node.node_id != 7]
    controller.reload([smaller])
    controller.reset()
    np.testing.assert_allclose(controller.action(observation), [0.0], atol=1e-6)


@pytest.mark.cuda
def test_previous_proposal_is_resident_local_not_collective_action():
    residents = []
    for proposal in (0.2, 0.8):
        squared_previous = Node(
            20, (Source(2, 0), Source(2, 0)), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), 1.0, 0.5
        )
        residents.append(Genome([_constant(10, proposal), squared_previous], [10, 20], 21))
    controller = _controller(residents, 2)
    observation = np.zeros(2, dtype=np.float32)
    controller.action(observation)
    # Mean of squared resident proposals differs from the squared collective mean.
    expected = 2.0 * ((0.2 ** 2 + 0.8 ** 2) / 2.0) - 1.0
    np.testing.assert_allclose(controller.action(observation), [0.0, expected], atol=1e-6)


class _EnumeratedChoice:
    def __init__(self, choice):
        self.choice = choice

    def integers(self, high):
        if not 0 <= self.choice < high:
            raise ValueError("enumerated choice is outside the source domain")
        return self.choice


@pytest.mark.cuda
def test_observation_and_proposal_indices_keep_endpoint_clamping():
    genome = Genome(
        [
            _constant(20, 0.2),
            _copy_source(40, Source(0, -3)),
            _copy_source(50, Source(0, 99)),
            _copy_source(60, Source(2, -3)),
            _copy_source(70, Source(2, 99)),
            _constant(30, 0.8),
        ],
        [20, 40, 50, 60, 70, 30],
        71,
    )
    controller = _controller([genome], 6)
    observation = np.array([-2.0, 2.0], dtype=np.float32)
    controller.action(observation)
    np.testing.assert_allclose(
        controller.action(observation),
        [-0.6, np.tanh(-2.0), np.tanh(2.0), -0.6, 0.6, 0.6],
        atol=1e-6,
    )


def test_source_generation_reaches_every_previous_proposal_channel():
    observation_dim, action_dim, node_ids = 2, 6, (1, 7)
    sources = [
        random_source(_EnumeratedChoice(choice), observation_dim, action_dim, node_ids)
        for choice in range(observation_dim + len(node_ids) + action_dim + 2)
    ]
    assert {source.index for source in sources if source.kind == 2} == set(range(action_dim))


@pytest.mark.cuda
def test_zero_total_authority_is_neutral_without_changing_weighted_actions():
    teams = []
    for weights in ((0.0, 0.0), (0.0, 1.0), (0.25, 0.75)):
        teams.append([
            Genome([_constant(10, proposal), _constant(20, weight)], [10, 10], 21, authority=20)
            for proposal, weight in zip((0.9, 0.7), weights)
        ])
    low = np.array([-2.0, 1.0], dtype=np.float32)
    high = np.array([4.0, 5.0], dtype=np.float32)
    controller = _controller(teams, 2, authority="evolved", low=low, high=high)
    controller.reset(3, [0, 1, 2])
    observation = np.zeros((3, 2), dtype=np.float32)
    normalized = np.array([[0.5], [0.7], [0.75]])
    expected = low + normalized * (high - low)
    for _ in range(2):
        action = controller.action(observation)
        assert np.all(np.isfinite(action))
        np.testing.assert_allclose(action, expected, atol=1e-6)

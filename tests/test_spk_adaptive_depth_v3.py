import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "sparse-nn"
    / "ppo_continuous_action_spk_adaptive_depth_v3.py"
)
SPEC = importlib.util.spec_from_file_location("spk_adaptive_depth_v3", SCRIPT)
spk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spk)


def make_layer(
    in_features=10,
    out_features=4,
    initial_connections=3,
    rewire_fraction=0.0,
    edge_capacity=None,
):
    return spk.AdaptiveSparseLinear(
        in_features=in_features,
        out_features=out_features,
        initial_connections=initial_connections,
        min_connections=1,
        edge_capacity=edge_capacity or in_features * out_features,
        capacity_tau=0.5,
        utility_ema=0.9,
        utility_rewire_fraction=rewire_fraction,
        utility_age_min=2,
    )


def make_depth_args(**overrides):
    values = dict(
        width=4,
        initial_connections=3,
        min_connections=1,
        edge_capacity=100,
        initial_hidden_layers=2,
        min_hidden_layers=1,
        max_hidden_layers=4,
        depth_tau=0.5,
        depth_learning_rate=3e-5,
        layer_overhead_connections=0.0,
        pool="prior",
        utility_rewire_fraction=0.0,
    )
    values.update(overrides)
    return spk.Args(**values)


def make_fake_env(obs_dim=5, action_dim=2):
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(obs_dim,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(action_dim,)),
    )


def set_capacities(layer, capacities):
    with torch.no_grad():
        layer.capacity_raw.copy_(torch.tensor(capacities, dtype=torch.float32))


def assert_unique_live_sources(layer):
    for destination in range(layer.out_features):
        ids = (layer.is_live & (layer.destination == destination)).nonzero().flatten()
        sources = layer.source[ids]
        assert sources.unique().numel() == sources.numel()


def test_complete_arena_needs_no_probe_when_every_perceptron_is_dense():
    layer = make_layer(in_features=7, out_features=3, initial_connections=7)

    assert layer.edge_capacity == 21
    assert layer.hard_connections() == 21
    assert not layer.is_probe.any()
    assert layer(torch.randn(5, 7)).shape == (5, 3)


def test_independent_capacities_shrink_and_grow_with_packed_edges():
    torch.manual_seed(0)
    layer = make_layer()
    set_capacities(layer, [1, 2, 5, 9])
    layer.age[layer.is_live] = 10

    grown, pruned, _ = layer.materialize()

    assert layer.live_count.tolist() == [1, 2, 5, 9]
    assert layer.hard_connections() == 17
    assert (grown, pruned) == (8, 3)
    assert layer.executed_ids.numel() == 17 + 4
    assert_unique_live_sources(layer)
    assert layer(torch.randn(8, 10)).shape == (8, 4)

    # Each non-dense unit's boundary probe can become a real edge next round.
    set_capacities(layer, [2, 3, 6, 10])
    grown, pruned, _ = layer.materialize()
    assert layer.live_count.tolist() == [2, 3, 6, 10]
    assert (grown, pruned) == (4, 0)
    assert_unique_live_sources(layer)


def test_non_full_arena_reserves_probe_slots_at_global_capacity():
    layer = make_layer(in_features=100, edge_capacity=32)
    set_capacities(layer, [7, 7, 7, 7])

    grown, pruned, _ = layer.materialize()

    assert (grown, pruned) == (16, 0)
    assert layer.hard_connections() == 28
    assert layer.is_probe.sum() == 4
    assert layer.executed_ids.numel() == layer.edge_capacity
    assert_unique_live_sources(layer)


def test_checkpoint_rebuilds_variable_length_execution_plan():
    source = make_layer()
    source.age[source.is_live] = 10
    set_capacities(source, [1, 2, 5, 9])
    source.materialize()
    restored = make_layer()

    restored.load_state_dict(source.state_dict())

    assert torch.equal(restored.executed_ids, source.executed_ids)
    inputs = torch.randn(6, 10)
    assert torch.allclose(restored(inputs), source(inputs))


def test_saturated_arena_retains_a_compute_shrink_gradient():
    layer = make_layer(in_features=7, out_features=3, initial_connections=7)

    layer.expected_connections().backward()

    assert torch.isfinite(layer.capacity_raw.grad).all()
    assert torch.all(layer.capacity_raw.grad > 0)


def test_capacity_above_source_ceiling_can_still_learn_to_shrink():
    layer = make_layer()
    with torch.no_grad():
        layer.capacity_raw.fill_(50.0)

    layer.expected_connections().backward()

    assert torch.all(layer.capacity_raw.grad > 0)


def test_capacity_at_minimum_retains_a_regrowth_gradient():
    layer = make_layer()
    with torch.no_grad():
        layer.capacity_raw.fill_(-20.0)

    gradient = torch.autograd.grad(layer.expected_connections(), layer.capacity_raw)[0]

    assert torch.all(gradient == 1)


def test_compute_multiplier_doubles_and_proxy_is_loss_offset_invariant():
    connections = torch.tensor(20_000.0)
    assert torch.isclose(torch.exp2(connections / 20_000.0), torch.tensor(2.0))
    layer = make_layer()
    inputs = torch.randn(6, 10)
    task_loss = layer(inputs).square().mean()
    multiplier = torch.exp2(layer.expected_connections() / 20.0)
    gradients = []
    for offset in (0.0, 100.0, -100.0):
        shifted = task_loss + offset
        proxy = multiplier * (1.0 + shifted - shifted.detach())
        gradients.append(
            torch.autograd.grad(proxy, layer.capacity_raw, retain_graph=True)[0]
        )

    assert torch.allclose(gradients[0], gradients[1])
    assert torch.allclose(gradients[0], gradients[2])


def test_capacity_distribution_reports_global_statistics_and_histogram():
    class RecordingWriter:
        def __init__(self):
            self.scalars = {}
            self.histograms = {}

        def add_scalar(self, name, value, step):
            self.scalars[name] = (value, step)

        def add_histogram(self, name, values, step):
            self.histograms[name] = (values.clone(), step)

    writer = RecordingWriter()
    values = torch.tensor([1.0, 1.0, 3.0, 7.0])

    stats = spk.log_capacity_distribution(writer, "sparse/test", values, 123)

    assert stats["mean"] == 3.0
    assert stats["total"] == 12.0
    assert stats["median"] == 2.0
    assert stats["min"] == 1.0
    assert stats["max"] == 7.0
    assert stats["unique"] == 3.0
    assert writer.scalars["sparse/test_median"] == (2.0, 123)
    assert torch.equal(writer.histograms["sparse/test_histogram"][0], values)


def test_arena_supports_thousands_of_connections_for_one_perceptron():
    layer = make_layer(
        in_features=5000,
        out_features=2,
        initial_connections=2000,
        edge_capacity=5000,
    )

    assert layer.live_count.tolist() == [2000, 2000]
    assert layer(torch.randn(2, 5000)).shape == (2, 2)


def test_capacity_receives_task_and_positive_compute_gradients():
    torch.manual_seed(1)
    layer = make_layer()
    inputs = torch.randn(6, 10)
    task_loss = layer(inputs).square().mean()
    expected_connections = layer.expected_connections()
    multiplier = torch.exp2(expected_connections / 20.0)
    objective = multiplier * (1.0 + task_loss - task_loss.detach())

    objective.backward()

    assert torch.isfinite(layer.capacity_raw.grad).all()
    assert layer.capacity_raw.grad.abs().sum() > 0
    assert torch.isclose(objective.detach(), multiplier.detach())


def test_gradient_utility_updates_persistent_age_and_ema_buffers():
    layer = make_layer()
    layer(torch.randn(8, 10)).square().mean().backward()
    live_ids = layer.is_live.nonzero().flatten()

    layer.update_utility_from_grad()

    assert torch.all(layer.age[live_ids] == 1)
    assert torch.all(layer.utility[live_ids] >= 0)
    assert torch.count_nonzero(layer.utility[live_ids]) > 0


def test_taylor_utility_controls_one_for_one_rewiring():
    torch.manual_seed(2)
    layer = make_layer(initial_connections=5, rewire_fraction=0.4)
    live_ids = layer.is_live.nonzero().flatten()
    with torch.no_grad():
        layer.age[live_ids] = 10
        layer.utility[live_ids] = torch.arange(live_ids.numel(), dtype=torch.float32)
    old_counts = layer.live_count.clone()
    old_sources = layer.source.clone()
    old_live = layer.is_live.clone()

    grown, pruned, rewired = layer.materialize()

    assert (grown, pruned) == (0, 0)
    assert rewired == 8
    assert torch.equal(layer.live_count, old_counts)
    assert ((layer.source != old_sources) & old_live).sum() == rewired
    assert_unique_live_sources(layer)


def test_rewiring_clears_adam_moments_for_changed_weight_slots():
    torch.manual_seed(3)
    layer = make_layer(initial_connections=5, rewire_fraction=0.4)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    layer(torch.randn(16, 10)).square().mean().backward()
    optimizer.step()
    live_ids = layer.is_live.nonzero().flatten()
    with torch.no_grad():
        layer.age[live_ids] = 10
        layer.utility[live_ids] = torch.arange(live_ids.numel(), dtype=torch.float32)
    old_sources = layer.source.clone()

    layer.materialize(optimizer)

    changed = live_ids[layer.source[live_ids] != old_sources[live_ids]]
    state = optimizer.state[layer.weight]
    assert changed.numel() == 8
    assert torch.count_nonzero(state["exp_avg"][changed]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"][changed]) == 0


def test_segment_gather_matches_logical_concatenation():
    torch.manual_seed(4)
    layer = make_layer(in_features=10, initial_connections=5)
    left = torch.randn(7, 3)
    right = torch.randn(7, 7)

    concatenated = layer(torch.cat([left, right], dim=-1))
    segmented = layer.forward_segments([left, right])

    assert torch.allclose(segmented, concatenated)


def test_frontier_is_zero_forward_but_supplies_depth_gradient():
    torch.manual_seed(5)
    trunk = spk.SparseTrunk(obs_dim=5, args=make_depth_args())
    inputs = torch.randn(9, 5)
    reference = trunk(inputs)
    frontier = trunk.blocks[int(trunk.hard_depth) - 1]
    with torch.no_grad():
        frontier.weight.add_(100.0)
        frontier.bias.add_(100.0)

    perturbed = trunk(inputs)
    depth_gradient = torch.autograd.grad(perturbed.square().mean(), trunk.depth_log)[0]

    assert torch.allclose(perturbed, reference)
    assert torch.isfinite(depth_gradient)
    assert depth_gradient.abs() > 0

    trunk.zero_grad(set_to_none=True)
    trunk(inputs).square().mean().backward()
    assert all(parameter.grad is None for parameter in frontier.parameters())


def test_frontier_does_not_allocate_adam_state():
    torch.manual_seed(50)
    trunk = spk.SparseTrunk(obs_dim=5, args=make_depth_args())
    frontier = trunk.blocks[trunk._hard_depth - 1]
    optimizer = torch.optim.Adam(trunk.parameters())

    trunk(torch.randn(9, 5)).square().mean().backward()
    optimizer.step()

    assert all(parameter not in optimizer.state for parameter in frontier.parameters())


def test_active_depth_matches_the_sequential_spk_path_exactly():
    torch.manual_seed(51)
    trunk = spk.SparseTrunk(obs_dim=5, args=make_depth_args())
    inputs = torch.randn(9, 5)

    stem = torch.nn.functional.silu(trunk.stem(inputs))
    expected = torch.nn.functional.silu(
        trunk.blocks[0].forward_segments([inputs, stem])
    )

    assert torch.allclose(trunk(inputs), expected)


def test_soft_depth_changes_do_not_change_forward_until_materialized():
    torch.manual_seed(6)
    trunk = spk.SparseTrunk(obs_dim=5, args=make_depth_args())
    inputs = torch.randn(8, 5)
    before = trunk(inputs)
    with torch.no_grad():
        trunk.depth_log.fill_(math.log(3.0))

    still_frozen = trunk(inputs)
    trunk.materialize(None)
    after = trunk(inputs)

    assert torch.allclose(still_frozen, before)
    assert int(trunk.hard_depth) == 3
    assert not torch.allclose(after, before)


def test_depth_materialization_moves_only_to_the_probed_frontier():
    trunk = spk.SparseTrunk(obs_dim=5, args=make_depth_args())
    with torch.no_grad():
        trunk.depth_log.fill_(math.log(4.0))

    trunk.materialize(None)

    assert trunk._hard_depth == 3
    assert int(trunk.hard_depth) == 3


def test_checkpoint_restores_python_depth_cache():
    source = spk.SparseTrunk(obs_dim=5, args=make_depth_args())
    with torch.no_grad():
        source.depth_log.fill_(math.log(3.0))
    source.materialize(None)
    restored = spk.SparseTrunk(obs_dim=5, args=make_depth_args())

    restored.load_state_dict(source.state_dict())

    assert restored._hard_depth == 3
    assert restored._hard_depth == int(restored.hard_depth)


def test_depth_cost_counts_deployed_blocks_and_detaches_dormant_capacities():
    trunk = spk.SparseTrunk(obs_dim=5, args=make_depth_args())
    expected = trunk.expected_connections()
    active_capacity = trunk.blocks[0].capacity_raw
    frontier_capacity = trunk.blocks[1].capacity_raw
    dormant_capacity = trunk.blocks[2].capacity_raw
    active_grad, frontier_grad, dormant_grad = torch.autograd.grad(
        expected,
        [active_capacity, frontier_capacity, dormant_capacity],
        allow_unused=True,
    )

    assert float(expected.detach()) == trunk.hard_connections()
    assert active_grad is not None and active_grad.abs().sum() > 0
    assert frontier_grad is None
    assert dormant_grad is None
    assert trunk.physical_connections() > trunk.hard_connections()


def test_compute_cost_adds_overhead_without_polluting_edge_metrics():
    trunk = spk.SparseTrunk(
        obs_dim=5,
        args=make_depth_args(layer_overhead_connections=37.0),
    )

    edge_cost = trunk.expected_edge_connections()
    compute_cost = trunk.expected_connections()

    assert torch.allclose(compute_cost - edge_cost, torch.tensor(37.0))


def test_far_dormant_blocks_do_not_contribute_unprobed_depth_cost():
    trunk = spk.SparseTrunk(obs_dim=5, args=make_depth_args())
    before = torch.autograd.grad(trunk.expected_connections(), trunk.depth_log)[0]
    with torch.no_grad():
        trunk.blocks[2].capacity_raw.fill_(100.0)
    after = torch.autograd.grad(trunk.expected_connections(), trunk.depth_log)[0]

    assert torch.allclose(after, before)


def test_actor_and_critic_depths_materialize_independently():
    agent = spk.Agent(make_fake_env(), make_depth_args())
    with torch.no_grad():
        agent.actor_trunk.depth_log.fill_(math.log(3.0))
        agent.critic_trunk.depth_log.fill_(math.log(1.0))

    agent.materialize(None)

    assert agent.hard_depths() == (3, 1)

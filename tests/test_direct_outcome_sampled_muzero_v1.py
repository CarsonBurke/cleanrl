import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from cleanrl.recurrent_search.direct_outcome_sampled_muzero_v1 import (
    Args,
    DirectOutcomeMuZero,
    InitialInference,
    RecurrentInference,
    SampledMuZeroSearch,
    SynchronousGenerationReplay,
    apply_generation_policy_improvement,
    centered_root_policy_loss,
    diagonal_gaussian_kl,
    pack_behavior_transfer,
    direct_outcome_loss,
    predictron_composed_return_loss,
    root_visit_weights,
    select_root_candidate,
    shifted_tanh_log_std,
)


def tiny_args(**overrides):
    values = dict(
        latent_dim=16,
        bottleneck_dim=8,
        representation_depth=1,
        dynamics_depth=1,
        prediction_depth=1,
        unroll_steps=4,
        root_candidates=4,
        child_candidates=2,
        simulations=8,
        simulation_wave=2,
        search_depth=3,
    )
    values.update(overrides)
    return Args(**values)


def test_shifted_tanh_log_std_has_exact_neutral_origin_and_open_bounds():
    raw = torch.tensor([-100.0, 0.0, 100.0])
    value = shifted_tanh_log_std(raw)
    assert value[0] >= -5.0
    assert value[2] <= 2.0
    assert value[1].item() == 0.0


def test_zero_prediction_prior_but_action_sensitive_recurrent_latent():
    torch.manual_seed(0)
    args = tiny_args()
    network = DirectOutcomeMuZero(obs_dim=3, action_dim=2, args=args)
    obs = torch.randn(5, 3)
    remaining = torch.ones(5)
    initial = network.initial_inference(obs, remaining)
    assert torch.equal(initial.value_rate, torch.zeros_like(initial.value_rate))
    assert torch.equal(initial.policy_mean, torch.zeros_like(initial.policy_mean))
    assert torch.equal(initial.policy_log_std, torch.zeros_like(initial.policy_log_std))
    action_a = torch.zeros(5, 2)
    action_b = torch.ones(5, 2)
    recurrent_a = network.recurrent_inference(initial.latent, action_a, remaining)
    recurrent_b = network.recurrent_inference(initial.latent, action_b, remaining)
    for prediction in (recurrent_a, recurrent_b):
        assert torch.equal(prediction.reward, torch.zeros_like(prediction.reward))
        assert torch.equal(
            prediction.termination_logit, torch.zeros_like(prediction.termination_logit)
        )
        assert torch.equal(prediction.value_rate, torch.zeros_like(prediction.value_rate))
        assert torch.equal(prediction.policy_mean, torch.zeros_like(prediction.policy_mean))
        assert torch.equal(
            prediction.policy_log_std, torch.zeros_like(prediction.policy_log_std)
        )
    assert not torch.allclose(recurrent_a.latent, recurrent_b.latent)


def _generation_batch(step, env0_done, env1_done):
    observations = np.asarray([[10.0 + step], [20.0 + step]], dtype=np.float32)
    factual = np.asarray([[100.0 + step], [200.0 + step]], dtype=np.float32)
    return dict(
        observations=observations,
        remaining=np.asarray([3 - step, 3 - step], dtype=np.float32),
        actions=np.asarray([[0.1], [0.2]], dtype=np.float32),
        rewards=np.asarray([1.0, -2.0 if step == 0 else 3.0], dtype=np.float32),
        factual_next=factual,
        terminated=np.asarray([env0_done, False]),
        truncated=np.asarray([False, env1_done]),
        root_candidates=np.zeros((2, 4, 1), dtype=np.float32),
        root_weights=np.full((2, 4), 0.25, dtype=np.float32),
        predicted_scores=np.asarray([0.5, -0.5], dtype=np.float32),
    )


def test_synchronous_generation_is_atomic_uses_final_state_and_clears():
    replay = SynchronousGenerationReplay(2, 1, 1, 4, max_episode_steps=3)
    replay.begin(generation_id=7)
    with pytest.raises(RuntimeError, match="not materialized"):
        replay.sample(1, 2, torch.Generator())
    replay.add_batch(**_generation_batch(0, env0_done=True, env1_done=False))
    assert replay.completed_transitions == 1
    assert not replay.ready
    # Env 0 is already inactive; this apparent new-episode transition must be ignored.
    replay.add_batch(**_generation_batch(1, env0_done=False, env1_done=True))
    assert replay.ready
    assert replay.completed_transitions == 3
    assert [episode.generation_id for episode in replay.episodes] == [7, 7]
    assert replay.episodes[0].observations[-1, 0] == 100.0
    np.testing.assert_allclose(replay.episodes[0].returns, [1.0, 0.0])
    np.testing.assert_allclose(replay.episodes[1].returns, [1.0, 3.0, 0.0])
    diagnostics = replay.prequential_diagnostics()
    assert set(diagnostics) >= {
        "model_exploitation/selected_q_bias",
        "model_exploitation/selected_q_mae",
        "model_exploitation/selected_q_return_correlation",
    }

    replay.start_training("cpu")
    fixed = replay.fixed_sequences(3, 1)
    assert fixed["obs0"][:, 0].tolist() == [10.0, 20.0, 200.0]
    with pytest.raises(RuntimeError, match="only be added"):
        replay.add_batch(**_generation_batch(2, False, False))
    assert replay.tensors["flat_episode"].numel() == 3
    replay.clear()
    assert replay.phase == replay.EMPTY
    assert replay.completed_transitions == 0
    assert not replay.pending


def test_generation_sequence_masks_and_state_topology_are_exact():
    replay = SynchronousGenerationReplay(1, 1, 1, 2, max_episode_steps=4)
    replay.begin(0)
    for step, reward in enumerate((1.0, -2.0, 3.0)):
        replay.add_batch(
            observations=np.asarray([[step]], dtype=np.float32),
            remaining=np.asarray([4 - step], dtype=np.float32),
            actions=np.asarray([[0.0]], dtype=np.float32),
            rewards=np.asarray([reward], dtype=np.float32),
            factual_next=np.asarray([[step + 1]], dtype=np.float32),
            terminated=np.asarray([step == 2]),
            truncated=np.asarray([False]),
            root_candidates=np.zeros((1, 2, 1), dtype=np.float32),
            root_weights=np.full((1, 2), 0.5, dtype=np.float32),
            predicted_scores=np.asarray([0.0], dtype=np.float32),
        )
    replay.start_training("cpu")
    np.testing.assert_allclose(replay.episodes[0].returns, [2.0, 1.0, 3.0, 0.0])
    generator = torch.Generator().manual_seed(5)
    batch = replay.sample(128, 4, generator)
    for row in range(128):
        start = int(batch["obs0"][row, 0].item())
        local_length = 3 - start
        expected_transition = [1.0] * local_length + [0.0] * (4 - local_length)
        expected_state = [1.0] * (local_length + 1) + [0.0] * (4 - local_length)
        assert batch["transition_mask"][row].tolist() == expected_transition
        assert batch["state_mask"][row].tolist() == expected_state


def test_predictron_all_subroot_depth_oracle_and_live_gradients():
    max_steps = 4
    reward = torch.tensor([[1.0, -2.0, 3.0]], requires_grad=True)
    termination_logit = torch.full((1, 3), -30.0)
    returns = torch.tensor([[2.0, 1.0, 3.0, 0.0]])
    value_rate = returns / max_steps
    remaining = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    mask = torch.ones(1, 3)
    loss, count = predictron_composed_return_loss(
        reward,
        termination_logit,
        value_rate,
        remaining,
        returns,
        mask,
        max_steps,
    )
    assert count.item() == 6
    assert loss.item() < 1e-12
    perturbed = reward + torch.tensor([[0.2, 0.0, 0.0]])
    perturbed_loss, _ = predictron_composed_return_loss(
        perturbed,
        termination_logit,
        value_rate,
        remaining,
        returns,
        mask,
        max_steps,
    )
    perturbed_loss.backward()
    assert reward.grad is not None
    assert reward.grad.abs().sum() > 0


class ZeroSearchNetwork(nn.Module):
    def __init__(self, action_dim=1, reward_from_action=False):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = 2
        self.reward_from_action = reward_from_action

    def initial_inference(self, obs, remaining_rate):
        batch = obs.shape[0]
        zero = torch.zeros(batch, device=obs.device, dtype=obs.dtype)
        return InitialInference(
            torch.zeros(batch, 2, device=obs.device, dtype=obs.dtype),
            zero,
            torch.zeros(batch, self.action_dim, device=obs.device, dtype=obs.dtype),
            torch.zeros(batch, self.action_dim, device=obs.device, dtype=obs.dtype),
        )

    def recurrent_inference(self, latent, action, next_remaining_rate):
        batch = latent.shape[0]
        zero = torch.zeros(batch, device=latent.device, dtype=latent.dtype)
        reward = action[:, 0] if self.reward_from_action else zero
        return RecurrentInference(
            latent,
            reward,
            torch.full_like(zero, -30.0),
            zero,
            torch.zeros(batch, self.action_dim, device=latent.device, dtype=latent.dtype),
            torch.zeros(batch, self.action_dim, device=latent.device, dtype=latent.dtype),
        )


def test_identity_search_visits_every_root_candidate_exactly_eight_times():
    network = ZeroSearchNetwork()
    search = SampledMuZeroSearch(
        network,
        max_episode_steps=1000,
        root_candidates=32,
        child_candidates=8,
        simulations=256,
        wave_size=16,
        search_depth=16,
    )
    output = search(
        torch.zeros(1, 1),
        torch.tensor([1000.0]),
        torch.linspace(-2.0, 2.0, 32).view(1, 32, 1),
        torch.zeros(1, 256, 8, 1),
    )
    assert output.visit_history[-1, 0].tolist() == [8] * 32
    assert output.q_min.item() == 0.0
    assert output.q_max.item() == 0.0


def test_search_allocates_more_visits_to_higher_predicted_return():
    network = ZeroSearchNetwork(reward_from_action=True)
    search = SampledMuZeroSearch(
        network,
        max_episode_steps=10,
        root_candidates=4,
        child_candidates=2,
        simulations=32,
        wave_size=2,
        search_depth=3,
    )
    output = search(
        torch.zeros(1, 1),
        torch.tensor([10.0]),
        torch.tensor([[[-2.0], [-0.5], [0.5], [2.0]]]),
        torch.zeros(1, 32, 2, 1),
    )
    visits = output.visit_history[-1, 0]
    assert visits.sum().item() == 32
    assert visits[-1] > visits[0]
    assert torch.argmax(visits).item() == 3


def test_wave_one_matches_slow_serial_depth_one_oracle():
    network = ZeroSearchNetwork(reward_from_action=True)
    candidates = torch.tensor([-1.5, -0.2, 0.4, 1.2])
    simulations = 13
    search = SampledMuZeroSearch(
        network,
        max_episode_steps=10,
        root_candidates=4,
        child_candidates=2,
        simulations=simulations,
        wave_size=1,
        search_depth=1,
    )
    output = search(
        torch.zeros(1, 1),
        torch.tensor([10.0]),
        candidates.view(1, 4, 1),
        torch.zeros(1, simulations, 2, 1),
    )

    visits = torch.zeros(4, dtype=torch.long)
    value_sum = torch.zeros(4)
    edge_return = torch.tanh(candidates)
    for _ in range(simulations):
        unvisited = visits == 0
        if unvisited.any():
            chosen = int(torch.argmax(unvisited.to(torch.long)))
        else:
            q = value_sum / visits
            span = q.max() - q.min()
            q_normalized = (q - q.min()) / span if span > 0 else torch.zeros_like(q)
            parent = visits.sum().to(torch.float32)
            pb_c = 1.25 + torch.log((parent + 19_652.0 + 1.0) / 19_652.0)
            score = q_normalized + pb_c * 0.25 * torch.sqrt(parent) / (1.0 + visits)
            chosen = int(torch.argmax(score))
        visits[chosen] += 1
        value_sum[chosen] += edge_return[chosen]
    assert torch.equal(output.visit_history[-1, 0], visits)
    torch.testing.assert_close(output.q_history[-1, 0], value_sum / visits.clamp_min(1))


def test_incremental_open_matches_full_rescan_when_child_frontier_closes():
    network = ZeroSearchNetwork()
    search = SampledMuZeroSearch(
        network,
        max_episode_steps=10,
        root_candidates=4,
        child_candidates=2,
        simulations=4,
        wave_size=2,
        search_depth=2,
    )
    node_count = torch.tensor([3])
    node_depth = torch.tensor([[0, 1, 1, 0, 0]])
    node_remaining = torch.tensor([[10.0, 9.0, 9.0, 0.0, 0.0]])
    candidate_count = torch.tensor([[4, 2, 2, 0, 0]])
    edge_child = torch.full((1, 5, 4), -1, dtype=torch.long)
    edge_child[0, 0, 0] = 1
    edge_child[0, 0, 1] = 2
    reserved = torch.zeros_like(edge_child, dtype=torch.bool)
    incremental = search._node_open(
        node_count, node_depth, node_remaining, candidate_count, edge_child, reserved
    )
    path_node = torch.tensor([[0, 1]])
    path_valid = torch.tensor([[True, True]])

    reserved[0, 1, 0] = True
    incremental = search._update_open_path(
        incremental,
        candidate_count,
        edge_child,
        reserved,
        path_node,
        path_valid,
    )
    brute = search._node_open(
        node_count, node_depth, node_remaining, candidate_count, edge_child, reserved
    )
    assert torch.equal(incremental, brute)
    assert incremental[0, 1]

    reserved[0, 1, 1] = True
    incremental = search._update_open_path(
        incremental,
        candidate_count,
        edge_child,
        reserved,
        path_node,
        path_valid,
    )
    brute = search._node_open(
        node_count, node_depth, node_remaining, candidate_count, edge_child, reserved
    )
    assert torch.equal(incremental, brute)
    assert not incremental[0, 1]
    assert incremental[0, 0]  # lane 3 must reroute through child 2 or a root frontier.


def test_visit_behavior_sampling_and_deterministic_ties():
    visits = torch.tensor([[1, 3, 0, 0], [2, 2, 0, 0]])
    weights = root_visit_weights(visits)
    torch.testing.assert_close(weights[0], torch.tensor([0.25, 0.75, 0.0, 0.0]))
    chosen = select_root_candidate(
        visits, torch.tensor([0.3, 0.99]), deterministic=False
    )
    assert chosen.tolist() == [1, 1]
    deterministic = select_root_candidate(
        visits, torch.zeros(2), deterministic=True
    )
    assert deterministic.tolist() == [1, 0]


def test_behavior_transfer_packing_roundtrips_all_fields():
    environment_action = torch.arange(6, dtype=torch.float32).view(2, 3)
    normalized_action = environment_action + 10
    root_pre_tanh = torch.arange(24, dtype=torch.float32).view(2, 4, 3) + 20
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    selected_q = torch.tensor([7.0, -3.0])
    transfer = pack_behavior_transfer(
        environment_action,
        normalized_action,
        root_pre_tanh,
        weights,
        selected_q,
    )
    np.testing.assert_array_equal(transfer.environment_action, environment_action.numpy())
    np.testing.assert_array_equal(transfer.normalized_action, normalized_action.numpy())
    np.testing.assert_array_equal(transfer.root_pre_tanh, root_pre_tanh.numpy())
    np.testing.assert_array_equal(transfer.root_weights, weights.numpy())
    np.testing.assert_array_equal(transfer.selected_q, selected_q.numpy())


def test_direct_loss_is_finite_and_full_unroll_reaches_all_live_heads():
    torch.manual_seed(9)
    args = tiny_args()
    network = DirectOutcomeMuZero(3, 2, args)
    batch = 4
    k = 4
    transition_mask = torch.ones(batch, k)
    state_mask = torch.ones(batch, k + 1)
    root_weights = torch.full((batch, k, 4), 0.25)
    inputs = dict(
        obs0=torch.randn(batch, 3),
        remaining=torch.tensor([[5, 4, 3, 2, 1]] * batch, dtype=torch.float32),
        actions=torch.tanh(torch.randn(batch, k, 2)),
        rewards=torch.randn(batch, k),
        terminated=torch.zeros(batch, k),
        returns=torch.randn(batch, k + 1),
        transition_mask=transition_mask,
        state_mask=state_mask,
    )
    first = direct_outcome_loss(network, max_episode_steps=5, **inputs)
    assert all(torch.isfinite(value) for value in first)
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    first.total.backward()
    assert network.reward_head.weight.grad.abs().sum() > 0
    assert network.value_head.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = direct_outcome_loss(network, max_episode_steps=5, **inputs)
    second.total.backward()
    assert network.representation.stem.weight.grad.abs().sum() > 0
    assert network.dynamics.stem.weight.grad.abs().sum() > 0


def test_uniform_search_policy_is_exact_noop_even_with_nonzero_adam_state():
    torch.manual_seed(21)
    args = tiny_args()
    network = DirectOutcomeMuZero(3, 2, args)
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4)
    # Populate nonzero Adam state through an unrelated factual outcome update.
    optimizer.zero_grad(set_to_none=True)
    ((network.reward_head.weight - 1.0).square().sum() + (network.reward_head.bias - 1.0).square().sum()).backward()
    optimizer.step()
    roots = {
        "observations": torch.randn(7, 3),
        "remaining": torch.full((7,), 5.0),
        "root_candidates": torch.randn(7, 4, 2),
        "root_weights": torch.full((7, 4), 0.25),
    }
    before = [parameter.detach().clone() for parameter in network.parameters()]
    metrics = apply_generation_policy_improvement(
        network, roots, max_episode_steps=5, learning_rate=3e-4, batch_size=3
    )
    after = list(network.parameters())
    assert metrics.centered_loss.item() == 0.0
    assert metrics.grad_norm.item() == 0.0
    assert metrics.exactly_uniform_fraction.item() == 1.0
    for expected, actual in zip(before, after):
        assert torch.equal(expected, actual)


def test_diagonal_gaussian_kl_has_known_direction_and_joint_sum():
    old_mean = torch.zeros(3, 2)
    old_log_std = torch.zeros_like(old_mean)
    new_mean = torch.ones_like(old_mean)
    new_log_std = torch.zeros_like(old_mean)
    identical = diagonal_gaussian_kl(
        old_mean, old_log_std, old_mean, old_log_std
    )
    shifted = diagonal_gaussian_kl(
        old_mean, old_log_std, new_mean, new_log_std
    )
    assert torch.equal(identical, torch.zeros_like(identical))
    torch.testing.assert_close(shifted, torch.ones_like(shifted))


def test_centered_root_policy_gradient_favors_search_improved_candidate():
    torch.manual_seed(22)
    args = tiny_args()
    network = DirectOutcomeMuZero(2, 1, args)
    observations = torch.randn(8, 2)
    remaining = torch.full((8,), 5.0)
    candidates = torch.tensor([[-1.0, -0.3, 0.3, 1.0]]).view(1, 4, 1).expand(8, -1, -1)
    weights = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(8, -1)
    loss, metrics = centered_root_policy_loss(
        network, observations, remaining, candidates, weights, max_episode_steps=5
    )
    network.zero_grad(set_to_none=True)
    loss.backward()
    assert network.policy_mean_head.bias.grad.item() < 0.0
    assert metrics["exactly_uniform_fraction"].item() == 0.0


def test_loss_and_search_compile_as_full_graph():
    torch.manual_seed(3)
    args = tiny_args(root_candidates=4)
    network = DirectOutcomeMuZero(2, 1, args)
    search = SampledMuZeroSearch(
        network,
        max_episode_steps=5,
        root_candidates=2,
        child_candidates=1,
        simulations=2,
        wave_size=1,
        search_depth=1,
    )
    compiled_search = torch.compile(search, backend="inductor", fullgraph=True)
    observations = torch.zeros(1, 2)
    remaining = torch.tensor([5.0])
    root_noise = torch.randn(1, 2, 1)
    child_noise = torch.randn(1, 2, 1, 1)
    eager = search(observations, remaining, root_noise, child_noise)
    compiled = compiled_search(observations, remaining, root_noise, child_noise)
    for eager_value, compiled_value in zip(eager, compiled):
        if eager_value.dtype.is_floating_point:
            torch.testing.assert_close(compiled_value, eager_value)
        else:
            assert torch.equal(compiled_value, eager_value)

    batch = 3
    k = 2
    loss_inputs = (
        torch.randn(batch, 2),
        torch.tensor([[3.0, 2.0, 1.0]] * batch),
        torch.tanh(torch.randn(batch, k, 1)),
        torch.randn(batch, k),
        torch.zeros(batch, k),
        torch.randn(batch, k + 1),
        torch.ones(batch, k),
        torch.ones(batch, k + 1),
    )

    def loss_call(*values):
        return direct_outcome_loss(network, *values, max_episode_steps=5)

    network.zero_grad(set_to_none=True)
    eager_loss = loss_call(*loss_inputs)
    eager_loss.total.backward()
    eager_grad = network.reward_head.weight.grad.detach().clone()
    network.zero_grad(set_to_none=True)
    compiled_loss_call = torch.compile(loss_call, backend="eager", fullgraph=True)
    compiled_loss = compiled_loss_call(*loss_inputs)
    compiled_loss.total.backward()
    for eager_value, compiled_value in zip(eager_loss, compiled_loss):
        torch.testing.assert_close(compiled_value, eager_value)
    torch.testing.assert_close(network.reward_head.weight.grad, eager_grad)

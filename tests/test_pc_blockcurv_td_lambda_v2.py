import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions.beta import Beta

from cleanrl.pc.ppo_continuous_action_pc_blockcurv_td_lambda_v2 import (
    Args,
    Agent,
    LocalPredictor,
    PCHierarchy,
    RecentObservationReservoir,
    RunningTDRMS,
    TraceBank,
    apply_head_directions,
    apply_local_directions,
    beta_head_scores,
    bootstrap_observations,
    clip_direction_blocks,
    compensate_critic_nudge_tensors,
    finite_example_rows,
    linear_scores,
    zero_invalid_rows,
)


def directional_finite_difference(fn, tensors, directions, eps=1e-4):
    originals = [tensor.clone() for tensor in tensors]
    with torch.no_grad():
        for tensor, original, direction in zip(tensors, originals, directions):
            tensor.copy_(original + eps * direction)
    plus = fn().item()
    with torch.no_grad():
        for tensor, original, direction in zip(tensors, originals, directions):
            tensor.copy_(original - eps * direction)
    minus = fn().item()
    with torch.no_grad():
        for tensor, original in zip(tensors, originals):
            tensor.copy_(original)
    return (plus - minus) / (2.0 * eps)


def test_exact_beta_actor_score_matches_finite_difference():
    torch.manual_seed(0)
    batch, hidden, actions = 3, 4, 2
    h = torch.randn(batch, hidden)
    alpha_head = torch.nn.Linear(hidden, actions)
    beta_head = torch.nn.Linear(hidden, actions)
    z = torch.rand(batch, actions).clamp(0.1, 0.9)
    alpha_raw = alpha_head(h)
    beta_raw = beta_head(h)
    alpha = 1.0 + F.softplus(alpha_raw)
    beta = 1.0 + F.softplus(beta_raw)
    alpha_score, beta_score = beta_head_scores(alpha_raw, beta_raw, alpha, beta, z)
    aw, ab = linear_scores(alpha_score, h)
    bw, bb = linear_scores(beta_score, h)
    directions = [aw.sum(0), ab.sum(0), bw.sum(0), bb.sum(0)]
    tensors = [alpha_head.weight, alpha_head.bias, beta_head.weight, beta_head.bias]

    def objective():
        a = 1.0 + F.softplus(alpha_head(h))
        b = 1.0 + F.softplus(beta_head(h))
        return Beta(a, b).log_prob(z).sum()

    finite_diff = directional_finite_difference(objective, tensors, directions)
    analytic = sum(direction.square().sum() for direction in directions).item()
    assert finite_diff > 0
    assert np.isclose(finite_diff, analytic, rtol=2e-2, atol=2e-2)


def test_value_feature_score_matches_finite_difference():
    torch.manual_seed(1)
    h = torch.randn(4, 5)
    head = torch.nn.Linear(5, 1)
    weight_score, bias_score = linear_scores(torch.ones(4, 1), h)
    directions = [weight_score.sum(0), bias_score.sum(0)]
    finite_diff = directional_finite_difference(
        lambda: head(h).sum(), [head.weight, head.bias], directions
    )
    analytic = sum(direction.square().sum() for direction in directions).item()
    assert finite_diff > 0
    assert np.isclose(finite_diff, analytic, rtol=2e-3, atol=2e-3)


def test_local_precision_residual_score_is_negative_energy_direction():
    torch.manual_seed(2)
    args = Args(pc_initial_precision=1.7, pc_hidden_activation="tanh")
    edge = LocalPredictor(3, 2, "tanh", args)
    source = torch.randn(4, 3)
    target = torch.randn(4, 2)
    phi = edge.augmented_features(source)
    score = edge.precision_residual(source, target).unsqueeze(2) * phi.unsqueeze(1)
    direction = score.sum(0)

    def log_likelihood_without_constant():
        residual = edge.residual(source, target)
        return -0.5 * (residual.square() * edge.precision).sum()

    finite_diff = directional_finite_difference(
        log_likelihood_without_constant,
        [edge.weight, edge.bias],
        [direction[:, :-1], direction[:, -1]],
    )
    analytic = direction.square().sum().item()
    assert finite_diff > 0
    assert np.isclose(finite_diff, analytic, rtol=2e-3, atol=2e-2)


def test_trace_rows_are_independent_and_terminal_delta_precedes_reset():
    trace = TraceBank([torch.empty(3, 1)])
    assert torch.count_nonzero(trace.traces[0]) == 0
    trace.accumulate([torch.tensor([[1.0], [2.0], [3.0]])], decay=0.5)
    trace.accumulate([torch.tensor([[2.0], [0.0], [-1.0]])], decay=0.5)
    # Current traces are [2.5, 1, 0.5]. The terminal row (1) must contribute now.
    update = trace.modulated_mean(torch.tensor([0.0, 6.0, 0.0]))[0]
    assert torch.allclose(update, torch.tensor([2.0]))
    trace.reset(torch.tensor([False, True, False]))
    assert torch.allclose(trace.traces[0].squeeze(), torch.tensor([2.5, 0.0, 0.5]))


def test_truncation_bootstraps_from_final_observation():
    next_obs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    truncations = np.array([True, False])
    infos = {
        "final_observation": np.array([np.array([8.0, 9.0]), None], dtype=object),
        "_final_observation": np.array([True, False]),
    }
    result = bootstrap_observations(next_obs, truncations, infos)
    np.testing.assert_array_equal(result[0], np.array([8.0, 9.0]))
    np.testing.assert_array_equal(result[1], next_obs[1])


def test_settling_is_batch_invariant():
    torch.manual_seed(3)
    args = Args(
        pc_num_hidden_layers=3,
        hidden_size=5,
        pc_inference_steps=5,
        pc_state_clip=20.0,
    )
    hierarchy = PCHierarchy(4, args.hidden_size, args)
    x = torch.randn(1, 4)
    terminal = lambda h: torch.full_like(h, 0.2)
    single, _, _ = hierarchy.settle(x, terminal, args)
    repeated, _, _ = hierarchy.settle(x.repeat(7, 1), terminal, args)
    for one, many in zip(single, repeated):
        assert torch.allclose(many, one.repeat(7, 1), atol=1e-6, rtol=1e-6)


def test_reverse_gauss_seidel_reaches_lowest_layer():
    torch.manual_seed(4)
    args = Args(
        pc_num_hidden_layers=6,
        hidden_size=8,
        pc_state_clip=20.0,
    )
    hierarchy = PCHierarchy(5, args.hidden_size, args)
    x = torch.randn(2, 5)
    states, _, diagnostics = hierarchy.settle(x, lambda h: torch.ones_like(h), args)
    scores = hierarchy.local_scores(x, states)
    assert diagnostics["displacements"][0].mean() > 0.1
    assert scores[0].norm() > 0.1


def test_one_layer_pc_local_actor_direction_aligns_with_chain_rule_score():
    torch.manual_seed(5)
    args = Args(pc_num_hidden_layers=1, hidden_size=4, pc_inference_steps=3, pc_state_clip=20.0)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    alpha_head = torch.nn.Linear(4, 2)
    beta_head = torch.nn.Linear(4, 2)
    x = torch.randn(2, 3)
    deployed = hierarchy.initial_states(x, args)[0]
    alpha_raw, beta_raw = alpha_head(deployed), beta_head(deployed)
    alpha, beta = 1 + F.softplus(alpha_raw), 1 + F.softplus(beta_raw)
    z = torch.tensor([[0.25, 0.7], [0.8, 0.4]])
    alpha_score, beta_score = beta_head_scores(alpha_raw, beta_raw, alpha, beta, z)
    logprob_h_grad = F.linear(alpha_score, alpha_head.weight.T) + F.linear(beta_score, beta_head.weight.T)
    states, _, _ = hierarchy.settle(x, lambda h: -logprob_h_grad, args)
    pc_direction = hierarchy.local_scores(x, states)[0]
    chain_rule = logprob_h_grad.unsqueeze(2) * hierarchy.edges[0].augmented_features(x).unsqueeze(1)
    assert (pc_direction * chain_rule).sum() > 0


def test_one_layer_pc_local_value_direction_aligns_and_lowers_energy():
    torch.manual_seed(6)
    args = Args(pc_num_hidden_layers=1, hidden_size=4, pc_inference_steps=5, pc_state_clip=20.0)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    value_weight = torch.randn(1, 4)
    x = torch.randn(2, 3)
    deployed = hierarchy.initial_states(x, args)[0]

    def energy(h):
        residual = hierarchy.edges[0].residual(x, h)
        return 0.5 * (residual.square() * hierarchy.edges[0].precision).sum() - (h @ value_weight.T).sum()

    states, _, _ = hierarchy.settle(x, lambda h: -value_weight.expand_as(h), args)
    pc_direction = hierarchy.local_scores(x, states)[0]
    chain_rule = value_weight.expand_as(states[0]).unsqueeze(2) * hierarchy.edges[0].augmented_features(x).unsqueeze(1)
    assert (pc_direction * chain_rule).sum() > 0
    assert energy(states[0]) < energy(deployed)


def test_precision_update_uses_cached_pre_mutation_residual():
    torch.manual_seed(7)
    args = Args(
        pc_num_hidden_layers=1,
        hidden_size=3,
        pc_precision_ema=1.0,
        pc_precision_ridge=0.2,
        pc_precision_min=0.01,
        pc_precision_max=100.0,
    )
    hierarchy = PCHierarchy(2, args.hidden_size, args)
    cached_residual = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    with torch.no_grad():
        hierarchy.edges[0].weight.fill_(1_000.0)
    hierarchy.update_precisions([cached_residual], args)
    expected = (cached_residual.square().mean(0) + args.pc_precision_ridge).reciprocal()
    assert torch.allclose(hierarchy.edges[0].precision, expected)


def test_curvature_cache_refreshes_in_place():
    torch.manual_seed(8)
    args = Args(pc_num_hidden_layers=3, hidden_size=4, pc_curvature_refresh_interval=1)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    x = torch.randn(2, 3)
    states = hierarchy.initial_states(x, args)
    first, _ = hierarchy.curvature_factors(x, states, args)
    pointer = first.data_ptr()
    first_value = first.clone()
    with torch.no_grad():
        hierarchy.edges[1].precision.mul_(2.0)
    hierarchy._curvature_cache_age = args.pc_curvature_refresh_interval
    second, _ = hierarchy.curvature_factors(x, states, args)
    assert second.data_ptr() == pointer
    assert not torch.equal(first_value, second)


def test_full_block_curvature_matches_formula_and_is_spd():
    torch.manual_seed(9)
    args = Args(
        pc_num_hidden_layers=2,
        hidden_size=4,
        pc_curvature_damping=0.07,
        pc_curvature_refresh_interval=1,
    )
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    x = torch.randn(5, 3)
    states = hierarchy.initial_states(x, args)
    factors, conditions = hierarchy.curvature_factors(x, states, args)
    downstream = hierarchy.edges[1]
    derivatives = 1.0 - torch.tanh(states[0]).square()
    derivative_gram = derivatives.T @ derivatives / derivatives.shape[0]
    weighted = downstream.weight.T @ (downstream.precision[:, None] * downstream.weight)
    expected = torch.diag(hierarchy.edges[0].precision)
    expected = expected + weighted * derivative_gram
    expected = expected + args.pc_curvature_damping * torch.eye(args.hidden_size)
    assert torch.allclose(hierarchy.cached_curvature_blocks[0], expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(factors[0] @ factors[0].T, expected, atol=1e-6, rtol=1e-6)
    assert torch.all(torch.linalg.eigvalsh(hierarchy.cached_curvature_blocks) > 0)
    assert torch.all(torch.isfinite(conditions))


def test_five_block_gauss_seidel_sweeps_lower_multilayer_pc_energy():
    torch.manual_seed(10)
    args = Args(pc_num_hidden_layers=6, hidden_size=8, pc_state_clip=20.0)
    hierarchy = PCHierarchy(5, args.hidden_size, args)
    x = torch.randn(4, 5)
    terminal_gradient = 0.1 * torch.randn(4, args.hidden_size)
    deployed = hierarchy.initial_states(x, args)

    def energy(states):
        residual_energy = sum(
            0.5 * (residual.square() * edge.precision).sum()
            for residual, edge in zip(hierarchy.residuals(x, states), hierarchy.edges)
        )
        return residual_energy + (terminal_gradient * states[-1]).sum()

    settled, _, diagnostics = hierarchy.settle(x, lambda h: terminal_gradient, args)
    assert diagnostics["steps"] == 5
    assert energy(settled) < energy(deployed)


def feedforward_with_grad(hierarchy, x, args):
    state = x
    for edge in hierarchy.edges:
        state = F.linear(edge.features(state), edge.weight, edge.bias)
        state = state.clamp(-args.pc_state_clip, args.pc_state_clip)
    return state


def flatten_local_direction(direction):
    return torch.cat([direction[:, :-1].reshape(-1), direction[:, -1].reshape(-1)])


def test_six_layer_pc_bottom_direction_aligns_materially_with_backprop():
    torch.manual_seed(11)
    args = Args(pc_num_hidden_layers=6, hidden_size=8, pc_state_clip=20.0)
    hierarchy = PCHierarchy(5, args.hidden_size, args)
    x = torch.randn(3, 5)
    for edge in hierarchy.edges:
        edge.weight.requires_grad_(True)
        edge.bias.requires_grad_(True)
    terminal_direction = 0.01 * torch.randn(3, args.hidden_size)
    output = feedforward_with_grad(hierarchy, x, args)
    bp_weight, bp_bias = torch.autograd.grad(
        (output * terminal_direction).sum(),
        [hierarchy.edges[0].weight, hierarchy.edges[0].bias],
    )
    bp_direction = torch.cat([bp_weight.reshape(-1), bp_bias.reshape(-1)])
    for edge in hierarchy.edges:
        edge.weight.requires_grad_(False)
        edge.bias.requires_grad_(False)

    settled, _, _ = hierarchy.settle(x, lambda h: -terminal_direction, args)
    pc_direction = flatten_local_direction(hierarchy.local_scores(x, settled)[0].sum(0))
    cosine = F.cosine_similarity(pc_direction, bp_direction, dim=0)
    norm_ratio = pc_direction.norm() / bp_direction.norm()

    # Five full-block sweeps preserve the chain-rule sign and a material bottom
    # score. They are intentionally a finite-inference approximation, not the
    # fifty-sweep near-equilibrium solve used by v1.
    assert cosine > 0.9
    assert norm_ratio > 0.04
    assert norm_ratio < 0.2


def test_six_layer_pc_preserves_terminal_signal_scale():
    torch.manual_seed(12)
    args = Args(pc_num_hidden_layers=6, hidden_size=8, pc_state_clip=20.0)
    hierarchy = PCHierarchy(5, args.hidden_size, args)
    x = torch.randn(2, 5)
    terminal_direction = 0.01 * torch.randn(2, args.hidden_size)
    once, _, _ = hierarchy.settle(x, lambda h: -terminal_direction, args)
    twice, _, _ = hierarchy.settle(x, lambda h: -2.0 * terminal_direction, args)
    negative, _, _ = hierarchy.settle(x, lambda h: terminal_direction, args)
    once_direction = flatten_local_direction(hierarchy.local_scores(x, once)[0].sum(0))
    twice_direction = flatten_local_direction(hierarchy.local_scores(x, twice)[0].sum(0))
    negative_direction = flatten_local_direction(hierarchy.local_scores(x, negative)[0].sum(0))

    assert F.cosine_similarity(once_direction, twice_direction, dim=0) > 0.995
    relative_error = (twice_direction - 2.0 * once_direction).norm() / (2.0 * once_direction.norm())
    assert relative_error < 0.03
    sign_error = (negative_direction + once_direction).norm() / once_direction.norm()
    assert sign_error < 0.03


class DummyVecEnv:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def test_seeded_deployment_initialization_matches_backprop_control():
    from cleanrl.pc.ppo_continuous_action_streaming_bp_tdlam_control_v1 import (
        Agent as BPAgent,
        Args as BPArgs,
    )

    pc_args = Args(
        num_envs=2,
        hidden_size=4,
        pc_num_hidden_layers=2,
        pc_state_clip=5.0,
        compile=False,
        cuda=False,
    )
    bp_args = BPArgs(
        num_envs=2,
        hidden_size=4,
        num_hidden_layers=2,
        hidden_state_clip=5.0,
        compile=False,
        cuda=False,
    )
    torch.manual_seed(13)
    pc_agent = Agent(DummyVecEnv(), pc_args)
    torch.manual_seed(13)
    bp_agent = BPAgent(DummyVecEnv(), bp_args)

    for pc_edge, bp_layer in zip(pc_agent.actor_pc.edges, bp_agent.actor.features.layers):
        assert torch.equal(pc_edge.weight, bp_layer.weight)
        assert torch.equal(pc_edge.bias, bp_layer.bias)
    assert torch.equal(pc_agent.actor_alpha_head.weight, bp_agent.actor.alpha_head.weight)
    assert torch.equal(pc_agent.actor_alpha_head.bias, bp_agent.actor.alpha_head.bias)
    assert torch.equal(pc_agent.actor_beta_head.weight, bp_agent.actor.beta_head.weight)
    assert torch.equal(pc_agent.actor_beta_head.bias, bp_agent.actor.beta_head.bias)
    for pc_edge, bp_layer in zip(pc_agent.critic_pc.edges, bp_agent.critic.features.layers):
        assert torch.equal(pc_edge.weight, bp_layer.weight)
        assert torch.equal(pc_edge.bias, bp_layer.bias)
    assert torch.equal(pc_agent.critic_head.weight, bp_agent.critic.value_head.weight)
    assert torch.equal(pc_agent.critic_head.bias, bp_agent.critic.value_head.bias)


def test_recent_observation_reservoir_retains_only_latest_capacity():
    reservoir = RecentObservationReservoir(capacity=4, observation_dim=2, device="cpu")
    reservoir.add(torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    reservoir.add(torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]))
    assert reservoir.count == 4
    retained = sorted(row[0].item() for row in reservoir.observations())
    assert retained == [2.0, 3.0, 4.0, 5.0]


def test_actor_and_critic_snapshot_rollback_is_bit_exact():
    torch.manual_seed(14)
    args = Args(hidden_size=4, pc_num_hidden_layers=2, cuda=False)
    agent = Agent(DummyVecEnv(), args)
    actor_snapshot = agent.snapshot_actor()
    critic_snapshot = agent.snapshot_critic()
    actor_head_directions = [
        torch.randn_like(agent.actor_alpha_head.weight),
        torch.randn_like(agent.actor_alpha_head.bias),
        torch.randn_like(agent.actor_beta_head.weight),
        torch.randn_like(agent.actor_beta_head.bias),
    ]
    actor_local_directions = [
        torch.randn(edge.weight.shape[0], edge.weight.shape[1] + 1)
        for edge in agent.actor_pc.edges
    ]
    critic_head_directions = [
        torch.randn_like(agent.critic_head.weight), torch.randn_like(agent.critic_head.bias)
    ]
    critic_local_directions = [
        torch.randn(edge.weight.shape[0], edge.weight.shape[1] + 1)
        for edge in agent.critic_pc.edges
    ]
    apply_head_directions(agent, actor_head_directions, 0.3, actor=True)
    apply_local_directions(agent.actor_pc, actor_local_directions, 0.3)
    apply_head_directions(agent, critic_head_directions, 0.3, actor=False)
    apply_local_directions(agent.critic_pc, critic_local_directions, 0.3)
    agent.restore_actor(actor_snapshot)
    agent.restore_critic(critic_snapshot)
    assert all(torch.equal(tensor, original) for tensor, original in zip(agent.snapshot_actor(), actor_snapshot))
    assert all(torch.equal(tensor, original) for tensor, original in zip(agent.snapshot_critic(), critic_snapshot))


def test_default_block_inference_is_five_sweeps_and_controller_scale_is_bounded():
    args = Args()
    assert args.pc_inference_steps == 5
    assert args.kl_scale_max <= 4.0


def test_per_block_caps_prevent_exploding_edge_from_suppressing_other_blocks():
    args = Args(
        max_head_direction_norm=0.25,
        max_local_block_direction_norm=0.15,
        max_direction_norm=10.0,
    )
    head = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    exploding = torch.full((2, 3), 100.0)
    healthy = torch.full((2, 3), 0.01)
    clipped_head, clipped_local, _, max_raw_fraction, max_accepted_fraction, cap_fraction = clip_direction_blocks(
        head, [exploding, healthy], args
    )
    assert torch.isclose(torch.cat([tensor.flatten() for tensor in clipped_head]).norm(), torch.tensor(0.25))
    assert torch.isclose(clipped_local[0].norm(), torch.tensor(0.15))
    assert torch.equal(clipped_local[1], healthy)
    assert torch.allclose(clipped_local[0] / exploding, torch.full_like(exploding, clipped_local[0][0, 0] / 100.0))
    assert max_raw_fraction > 0.99
    assert 0.0 < max_accepted_fraction < max_raw_fraction
    assert torch.isclose(cap_fraction, torch.tensor(2.0 / 3.0))


def test_accepted_weight_drift_invalidates_and_refreshes_curvature_cache():
    torch.manual_seed(15)
    args = Args(
        pc_num_hidden_layers=3,
        hidden_size=4,
        pc_curvature_refresh_interval=1_000,
        pc_curvature_relative_invalidation=1e-4,
    )
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    x = torch.randn(2, 3)
    states = hierarchy.initial_states(x, args)
    initial_factors, _ = hierarchy.curvature_factors(x, states, args)
    initial_factors = initial_factors.clone()
    directions = [
        torch.zeros(edge.weight.shape[0], edge.weight.shape[1] + 1)
        for edge in hierarchy.edges
    ]
    directions[1][:, :-1].fill_(1.0)
    apply_local_directions(hierarchy, directions, lr=0.01)
    hierarchy.note_accepted_weight_update(directions, learning_rate=0.01, args=args)
    assert hierarchy._curvature_cache_age == args.pc_curvature_refresh_interval
    refreshed, _ = hierarchy.curvature_factors(x, hierarchy.initial_states(x, args), args)
    assert not torch.equal(initial_factors, refreshed)
    assert hierarchy._curvature_accumulated_relative_drift == 0.0


def test_agent_hidden_and_head_parameters_are_all_autograd_free():
    agent = Agent(DummyVecEnv(), Args(hidden_size=4, pc_num_hidden_layers=2, cuda=False))
    assert all(not parameter.requires_grad for parameter in agent.parameters())


def test_actual_beta_energy_guard_backtracks_adversarial_late_weight_states():
    backtracked = 0
    for seed in (5, 12, 15, 16):
        torch.manual_seed(seed)
        args = Args(
            cuda=False,
            num_envs=4,
            hidden_size=8,
            pc_num_hidden_layers=6,
            pc_state_clip=5.0,
        )
        agent = Agent(DummyVecEnv(), args)
        hierarchy = agent.actor_pc
        with torch.no_grad():
            for edge in hierarchy.edges:
                edge.weight.mul_(2.0)
                edge.precision.copy_(10 ** torch.empty_like(edge.precision).uniform_(-1.0, 1.0))
        x = torch.randn(args.num_envs, 3)
        deployed = hierarchy.initial_states(x, args)
        _, _, deployed_dist = agent.actor_outputs(deployed[-1])
        action_z = deployed_dist.sample().clamp(1e-6, 1.0 - 1e-6)
        terminal_energy = lambda h: -agent.actor_outputs(h)[2].log_prob(action_z).sum(1)
        initial_energy = hierarchy.per_example_energy(x, deployed, terminal_energy)
        settled, _, diagnostics = agent.settle_actor(x, action_z, args, False)
        final_energy = hierarchy.per_example_energy(x, settled, terminal_energy)
        tolerance = 2e-6 * (1.0 + initial_energy.abs())
        assert torch.all(torch.isfinite(final_energy))
        assert torch.all(final_energy <= initial_energy + tolerance)
        backtracked += float(diagnostics["backtrack_scale_mean"]) < 0.999
    assert backtracked >= 3


def test_nonfinite_examples_are_zeroed_without_contaminating_td_rms_or_precision():
    rms = RunningTDRMS(torch.device("cpu"), decay=0.9, minimum=0.1)
    normalized = rms.normalize(torch.tensor([2.0, float("nan")]), clip=10.0)
    assert torch.equal(normalized, torch.tensor([1.0, 0.0]))
    assert torch.isfinite(rms.mean_square)

    scores = [torch.tensor([[1.0, 2.0], [float("nan"), 3.0]])]
    valid = finite_example_rows(scores)
    sanitized = zero_invalid_rows(scores, valid)[0]
    assert torch.equal(valid, torch.tensor([True, False]))
    assert torch.equal(sanitized, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))

    args = Args(
        pc_initial_precision=1.0,
        pc_precision_ema=1.0,
        pc_precision_ridge=0.2,
        pc_precision_min=0.01,
        pc_precision_max=100.0,
    )
    edge = LocalPredictor(2, 2, "identity", args)
    residual = torch.tensor([[1.0, 2.0], [float("nan"), float("nan")]])
    edge.update_precision(residual, args)
    assert torch.allclose(edge.precision, torch.tensor([1 / 1.2, 1 / 4.2]))
    preserved = edge.precision.clone()
    edge.update_precision(torch.full_like(residual, float("nan")), args)
    assert torch.equal(edge.precision, preserved)


def test_energy_guard_fallback_stays_finite_for_nonfinite_proposal():
    torch.manual_seed(17)
    args = Args(cuda=False, pc_num_hidden_layers=2, hidden_size=4)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    x = torch.randn(2, 3)
    deployed = hierarchy.initial_states(x, args)
    proposed = [state.clone() for state in deployed]
    proposed[0][0, 0] = float("nan")
    diagnostics = {"displacements": [], "residuals": []}
    chosen, residuals, diagnostics = hierarchy.guard_settlement(
        x,
        proposed,
        lambda h: h.new_zeros(h.shape[:-1]),
        args,
        diagnostics,
    )
    assert all(torch.all(torch.isfinite(state)) for state in chosen)
    assert all(torch.all(torch.isfinite(residual)) for residual in residuals)
    assert torch.isfinite(diagnostics["energy_delta"])
    assert torch.equal(chosen[0][0], deployed[0][0])


def test_small_critic_nudge_compensates_only_local_scores():
    scores = [torch.randn(3, 4, 5), torch.randn(3, 4, 5)]
    compensated = compensate_critic_nudge_tensors(
        scores,
        Args(pc_critic_terminal_coef=0.1, pc_compensate_critic_local_score=True),
    )
    for original, actual in zip(scores, compensated):
        torch.testing.assert_close(actual, 10.0 * original)
    unchanged = compensate_critic_nudge_tensors(
        scores,
        Args(pc_critic_terminal_coef=0.1, pc_compensate_critic_local_score=False),
    )
    assert all(actual is original for actual, original in zip(unchanged, scores))

    unit_args = Args(pc_precision_ema=1.0, pc_precision_ridge=0.1)
    nudged_args = Args(
        pc_precision_ema=1.0,
        pc_precision_ridge=0.1,
        pc_critic_terminal_coef=0.1,
        pc_compensate_critic_local_score=True,
    )
    unit_edge = LocalPredictor(2, 2, "identity", unit_args)
    nudged_edge = LocalPredictor(2, 2, "identity", nudged_args)
    residual = torch.randn(8, 2)
    unit_edge.update_precision(residual, unit_args)
    restored_residual = compensate_critic_nudge_tensors([0.1 * residual], nudged_args)[0]
    nudged_edge.update_precision(restored_residual, nudged_args)
    torch.testing.assert_close(nudged_edge.precision, unit_edge.precision)

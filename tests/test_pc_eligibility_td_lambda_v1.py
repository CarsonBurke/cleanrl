import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions.beta import Beta

from cleanrl.ppo_continuous_action_pc_eligibility_td_lambda_v1 import (
    Args,
    Agent,
    LocalPredictor,
    PCHierarchy,
    TraceBank,
    beta_head_scores,
    bootstrap_observations,
    linear_scores,
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
        pc_momentum=0.3,
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
    first = hierarchy.curvature_bounds(args)
    pointer = first.data_ptr()
    first_value = first.clone()
    with torch.no_grad():
        hierarchy.edges[1].precision.mul_(2.0)
    hierarchy._curvature_cache_age = args.pc_curvature_refresh_interval
    second = hierarchy.curvature_bounds(args)
    assert second.data_ptr() == pointer
    assert not torch.equal(first_value, second)


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

    # These thresholds reject both ten-step SGD extinction and coordinatewise
    # normalization while leaving room for finite-inference approximation.
    assert cosine > 0.95
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
    once_direction = flatten_local_direction(hierarchy.local_scores(x, once)[0].sum(0))
    twice_direction = flatten_local_direction(hierarchy.local_scores(x, twice)[0].sum(0))

    assert F.cosine_similarity(once_direction, twice_direction, dim=0) > 0.995
    relative_error = (twice_direction - 2.0 * once_direction).norm() / (2.0 * once_direction.norm())
    assert relative_error < 0.03


class DummyVecEnv:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def test_seeded_deployment_initialization_matches_backprop_control():
    from cleanrl.ppo_continuous_action_streaming_bp_tdlam_control_v1 import (
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

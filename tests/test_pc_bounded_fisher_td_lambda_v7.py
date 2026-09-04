from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

from cleanrl.pc.ppo_continuous_action_pc_bounded_fisher_td_lambda_v7 import (
    Args,
    Agent,
    AugmentedAdamW,
    LocalPredictor,
    OutputPredictor,
    PCHierarchy,
    TraceBank,
    bounded_beta_fisher_metric,
    bounded_beta_jacobian,
    bounded_beta_parameters,
    bounded_beta_score,
    bootstrap_observations,
    metric_times,
    fixed_endpoint_force,
)


def directional_finite_difference(fn, value, direction, eps=1e-5):
    plus = fn(value + eps * direction)
    minus = fn(value - eps * direction)
    return (plus - minus) / (2.0 * eps)


def _bounded_geometry(raw):
    alpha, beta, allocation, concentration = bounded_beta_parameters(raw, 1.0, 1.0, 32.0)
    return alpha, beta, allocation, concentration


def _bounded_score_and_metric(raw, action, damping=1e-9):
    alpha, beta, allocation, concentration = _bounded_geometry(raw)
    score = bounded_beta_score(raw, alpha, beta, allocation, concentration, action, 1.0, 32.0)
    metric = bounded_beta_fisher_metric(
        raw, alpha, beta, allocation, concentration, 1.0, 32.0, damping
    )
    return score, metric


def test_bounded_beta_score_matches_directional_finite_difference():
    torch.manual_seed(0)
    raw = torch.randn(4, 6, dtype=torch.float64)
    action = torch.rand(4, 3, dtype=torch.float64).clamp(0.05, 0.95)
    score, _ = _bounded_score_and_metric(raw, action)
    direction = torch.randn_like(raw)

    def objective(candidate):
        candidate_alpha, candidate_beta, _, _ = _bounded_geometry(candidate)
        dist = Beta(candidate_alpha, candidate_beta)
        return dist.log_prob(action).sum()

    finite_difference = directional_finite_difference(objective, raw, direction)
    torch.testing.assert_close(finite_difference, (score * direction).sum(), rtol=2e-6, atol=2e-6)


def test_concentration_map_is_smooth_monotone_and_bounded():
    concentration_raw = torch.linspace(-20.0, 20.0, 1000, dtype=torch.float64)
    raw = torch.stack([torch.zeros_like(concentration_raw), concentration_raw], dim=1)
    alpha, beta, allocation, concentration = _bounded_geometry(raw)
    assert torch.all(concentration[1:] > concentration[:-1])
    assert torch.all(concentration >= 1.0)
    assert torch.all(concentration < 32.0)
    torch.testing.assert_close(allocation, torch.full_like(allocation, 0.5))
    torch.testing.assert_close(alpha + beta, 2.0 + concentration)


@pytest.mark.parametrize("scale", [0.1, 1.0, 8.0])
def test_pulled_back_fisher_is_spd_and_direct_force_is_exact(scale):
    torch.manual_seed(1)
    raw = scale * torch.randn(7, 6)
    action = torch.rand(7, 3).clamp(1e-5, 1.0 - 1e-5)
    score, metric = _bounded_score_and_metric(raw, action, damping=1e-3)
    torch.linalg.cholesky(metric)
    nudge = 0.037
    recovered = fixed_endpoint_force(raw, raw, metric, score, nudge) / nudge
    torch.testing.assert_close(recovered, score, rtol=3e-4, atol=3e-4)


def test_direct_force_equals_solved_target_form_without_runtime_inverse():
    torch.manual_seed(19)
    free_output = torch.randn(5, 4, dtype=torch.float64)
    output = torch.randn_like(free_output)
    raw_metric = torch.randn(5, 4, 4, dtype=torch.float64)
    metric = raw_metric @ raw_metric.transpose(1, 2) + 0.1 * torch.eye(4, dtype=torch.float64)
    score = torch.randn_like(free_output)
    nudge = 0.05
    target = free_output + nudge * torch.linalg.solve(metric, score.unsqueeze(-1)).squeeze(-1)
    old_force = metric_times(metric, target - output)
    direct_force = fixed_endpoint_force(output, free_output, metric, score, nudge)
    torch.testing.assert_close(direct_force, old_force, rtol=2e-12, atol=2e-12)


def test_beta_fisher_matches_score_covariance_monte_carlo():
    torch.manual_seed(2)
    raw = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
    alpha, beta, allocation, concentration = _bounded_geometry(raw)
    samples = Beta(alpha, beta).sample((300_000,)).view(-1, 1)
    repeated_raw = raw.expand(samples.shape[0], -1)
    scores = bounded_beta_score(
        repeated_raw,
        alpha.expand_as(samples),
        beta.expand_as(samples),
        allocation.expand_as(samples),
        concentration.expand_as(samples),
        samples,
        1.0,
        32.0,
    )
    empirical = scores.T @ scores / scores.shape[0]
    analytic = bounded_beta_fisher_metric(
        raw, alpha, beta, allocation, concentration, 1.0, 32.0, damping=1e-12
    )[0]
    torch.testing.assert_close(empirical, analytic, rtol=1.5e-2, atol=1.5e-3)


def test_fisher_is_exact_analytic_pullback():
    torch.manual_seed(23)
    raw = torch.randn(4, 6, dtype=torch.float64)
    alpha, beta, allocation, concentration = _bounded_geometry(raw)
    jacobian = bounded_beta_jacobian(raw, allocation, concentration, 1.0, 32.0)
    common = torch.polygamma(1, alpha + beta)
    fisher_ab = raw.new_empty(4, 3, 2, 2)
    fisher_ab[..., 0, 0] = torch.polygamma(1, alpha) - common
    fisher_ab[..., 1, 1] = torch.polygamma(1, beta) - common
    fisher_ab[..., 0, 1] = fisher_ab[..., 1, 0] = -common
    expected_blocks = jacobian.transpose(-1, -2) @ fisher_ab @ jacobian
    metric = bounded_beta_fisher_metric(
        raw, alpha, beta, allocation, concentration, 1.0, 32.0, damping=1e-6
    )
    for action_idx in range(3):
        indices = torch.tensor([action_idx, 3 + action_idx])
        actual = metric[:, indices[:, None], indices]
        torch.testing.assert_close(
            actual - 1e-6 * torch.eye(2, dtype=torch.float64),
            expected_blocks[:, action_idx],
            rtol=2e-12,
            atol=2e-12,
        )


def test_free_states_have_zero_residual_without_hidden_clamps():
    torch.manual_seed(3)
    args = Args(hidden_size=5, pc_num_hidden_layers=3)
    hierarchy = PCHierarchy(4, args.hidden_size, args)
    with torch.no_grad():
        hierarchy.edges[0].weight.mul_(20.0)
        hierarchy.edges[0].bias.fill_(9.0)
    observation = torch.randn(6, 4)
    free = hierarchy.initial_states(observation)
    assert free[0].abs().max() > 5.0
    for residual in hierarchy.residuals(observation, free):
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)


def test_top_curvature_contains_full_output_metric():
    torch.manual_seed(4)
    args = Args(hidden_size=4, pc_num_hidden_layers=2, pc_curvature_damping=0.07)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 3, std=0.2)
    observation = torch.randn(8, 3)
    states = hierarchy.initial_states(observation)
    raw = torch.randn(8, 3, 3)
    metric = raw @ raw.transpose(1, 2) + 0.2 * torch.eye(3)
    factors, _, _ = hierarchy.curvature_factors(states, output, metric, args)
    recovered = factors[-1] @ factors[-1].T
    expected = (
        (1.0 + args.pc_curvature_damping) * torch.eye(args.hidden_size)
        + output.weight.T @ metric.mean(0) @ output.weight
    )
    torch.testing.assert_close(recovered, expected, rtol=2e-6, atol=2e-6)


def _settled_directions(nudge):
    torch.manual_seed(5)
    args = Args(
        hidden_size=5,
        pc_num_hidden_layers=3,
        pc_inference_steps=20,
        pc_inference_scale=1.0,
        pc_curvature_damping=0.05,
    )
    hierarchy = PCHierarchy(4, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 2, std=0.15)
    observation = 0.3 * torch.randn(9, 4)
    free = hierarchy.initial_states(observation)
    prediction = output(free[-1])
    metric = torch.tensor([[1.7, 0.25], [0.25, 0.8]]).expand(9, -1, -1).clone()
    force = torch.randn(9, 2)
    states, _, _ = hierarchy.settle(
        observation, output, prediction, force, metric, nudge, args
    )
    directions = hierarchy.local_scores(observation, states, nudge)
    output_force = fixed_endpoint_force(output(states[-1]), prediction, metric, force, nudge)
    directions.append(
        output_force.unsqueeze(2) * output.augmented_features(states[-1]).unsqueeze(1) / nudge
    )
    return directions


def test_inverse_nudge_compensation_has_linear_small_nudge_limit():
    medium = _settled_directions(2e-3)
    small = _settled_directions(1e-3)
    for medium_direction, small_direction in zip(medium, small):
        cosine = F.cosine_similarity(medium_direction.flatten(), small_direction.flatten(), dim=0)
        norm_ratio = medium_direction.norm() / small_direction.norm().clamp_min(1e-12)
        assert cosine > 0.9999
        assert 0.995 < norm_ratio < 1.005


def _actor_pc_and_exact_score_cosine(head_scale):
    torch.manual_seed(42)
    args = Args(
        hidden_size=16,
        pc_num_hidden_layers=6,
        pc_inference_steps=10,
        pc_actor_nudge=0.05,
    )
    hierarchy = PCHierarchy(7, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 4, std=0.01)
    with torch.no_grad():
        output.weight.mul_(head_scale)
    observation = 0.2 * torch.randn(16, 7)
    free = hierarchy.initial_states(observation)
    raw = output(free[-1])
    alpha, beta, allocation, concentration = _bounded_geometry(raw)
    action = torch.rand(16, 2).clamp(0.05, 0.95)
    score = bounded_beta_score(raw, alpha, beta, allocation, concentration, action, 1.0, 32.0)
    metric = bounded_beta_fisher_metric(
        raw,
        alpha,
        beta,
        allocation,
        concentration,
        1.0,
        32.0,
        args.pc_fisher_damping,
    )
    states, _, _ = hierarchy.settle(
        observation,
        output,
        raw,
        score,
        metric,
        args.pc_actor_nudge,
        args,
    )
    pc_directions = hierarchy.local_scores(observation, states, args.pc_actor_nudge)
    output_force = fixed_endpoint_force(
        output(states[-1]), raw, metric, score, args.pc_actor_nudge
    )
    pc_directions.append(
        output_force.unsqueeze(2)
        * output.augmented_features(states[-1]).unsqueeze(1)
        / args.pc_actor_nudge
    )
    pc_flat = torch.cat([direction.sum(0).flatten() for direction in pc_directions])

    parameters = []
    for edge in [*hierarchy.edges, output]:
        edge.weight.requires_grad_(True)
        edge.bias.requires_grad_(True)
        parameters.extend([edge.weight, edge.bias])
    source = observation
    for edge in hierarchy.edges:
        source = edge(source)
    exact_raw = output(source)
    exact_alpha, exact_beta, _, _ = _bounded_geometry(exact_raw)
    objective = Beta(exact_alpha, exact_beta).log_prob(action).sum()
    gradients = torch.autograd.grad(objective, parameters)
    exact_augmented = [
        torch.cat([weight_gradient, bias_gradient[:, None]], dim=1).flatten()
        for weight_gradient, bias_gradient in zip(gradients[::2], gradients[1::2])
    ]
    exact_flat = torch.cat(exact_augmented)
    return F.cosine_similarity(pc_flat, exact_flat, dim=0)


def test_six_layer_compensated_actor_direction_tracks_exact_score_at_runtime_nudge():
    ordinary_cosine = _actor_pc_and_exact_score_cosine(head_scale=1.0)
    high_sensitivity_cosine = _actor_pc_and_exact_score_cosine(head_scale=100.0)
    assert ordinary_cosine > 0.99
    # Strong output sensitivity changes the coherent PC equilibrium geometry;
    # the direction must remain positively aligned even in this stress regime.
    assert high_sensitivity_cosine > 0.70
    assert high_sensitivity_cosine < ordinary_cosine


def test_ten_sweeps_reduce_total_coherent_pc_energy():
    torch.manual_seed(61)
    args = Args(hidden_size=8, pc_num_hidden_layers=6, pc_inference_steps=10)
    hierarchy = PCHierarchy(5, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 4, std=0.2)
    observation = torch.randn(16, 5)
    free = hierarchy.initial_states(observation)
    prediction = output(free[-1])
    raw_metric = torch.randn(16, 4, 4)
    metric = raw_metric @ raw_metric.transpose(1, 2) + 0.1 * torch.eye(4)
    score = torch.randn(16, 4)
    nudge = 0.05

    def energy(states):
        hidden = sum(0.5 * residual.square().sum(1) for residual in hierarchy.residuals(observation, states))
        displacement = output(states[-1]) - prediction
        endpoint = 0.5 * torch.einsum(
            "bi,bij,bj->b", displacement, metric, displacement
        ) - nudge * (score * displacement).sum(1)
        return hidden + endpoint

    before = energy(free)
    settled, _, _ = hierarchy.settle(
        observation, output, prediction, score, metric, nudge, args
    )
    after = energy(settled)
    assert after.sum() < before.sum()
    assert torch.all(after <= before + 1e-6)


def test_quadratic_output_local_score_is_negative_energy_direction():
    torch.manual_seed(6)
    output = OutputPredictor(3, 2, std=0.2).double()
    source = torch.randn(5, 3, dtype=torch.float64)
    free_output = torch.randn(5, 2, dtype=torch.float64)
    score = torch.randn(5, 2, dtype=torch.float64)
    nudge = 0.13
    raw = torch.randn(5, 2, 2, dtype=torch.float64)
    metric = raw @ raw.transpose(1, 2) + 0.3 * torch.eye(2, dtype=torch.float64)
    endpoint_force = fixed_endpoint_force(output(source), free_output, metric, score, nudge)
    phi = output.augmented_features(source)
    direction = (endpoint_force.unsqueeze(2) * phi.unsqueeze(1)).sum(0)
    augmented = torch.cat([output.weight, output.bias[:, None]], dim=1).detach()

    def negative_endpoint_energy(candidate):
        prediction = F.linear(source, candidate[:, :-1], candidate[:, -1])
        displacement = prediction - free_output
        return -0.5 * torch.einsum(
            "bi,bij,bj->", displacement, metric, displacement
        ) + nudge * (score * displacement).sum()

    finite_difference = directional_finite_difference(
        negative_endpoint_energy, augmented, direction
    )
    torch.testing.assert_close(finite_difference, direction.square().sum(), rtol=2e-6, atol=2e-6)


def test_agent_endpoint_eligibility_uses_settled_feature_and_correct_force_sign():
    torch.manual_seed(31)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(5,), dtype=np.float32
        ),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )
    args = Args(hidden_size=8, pc_num_hidden_layers=3, pc_inference_steps=5)
    assert args.weight_decay == 0.0
    agent = Agent(envs, args)
    observation = torch.randn(4, 5)
    action = torch.rand(4, 2).clamp(0.05, 0.95)
    free_states = agent.actor_pc.initial_states(observation)
    free_output, metric, score = agent.actor_endpoint(free_states[-1], action, args)
    torch.testing.assert_close(
        fixed_endpoint_force(
            free_output, free_output, metric, score, args.pc_actor_nudge
        ),
        args.pc_actor_nudge * score,
    )

    states, _, output_eligibility, _ = agent.settle_actor(
        observation, action, args, collect_diagnostics=False
    )
    settled_force = fixed_endpoint_force(
        agent.actor_output(states[-1]),
        free_output,
        metric,
        score,
        args.pc_actor_nudge,
    )
    expected = (
        settled_force.unsqueeze(2)
        * agent.actor_output.augmented_features(states[-1]).unsqueeze(1)
        / args.pc_actor_nudge
    )
    torch.testing.assert_close(output_eligibility, expected)


def test_trace_is_per_environment_and_terminal_delta_precedes_reset():
    trace = TraceBank([torch.empty(3, 1)])
    trace.accumulate([torch.tensor([[1.0], [2.0], [3.0]])], decay=0.5)
    trace.accumulate([torch.tensor([[2.0], [0.0], [-1.0]])], decay=0.5)
    update = trace.modulated_mean(torch.tensor([0.0, 6.0, 0.0]))[0]
    torch.testing.assert_close(update, torch.tensor([2.0]))
    trace.reset(torch.tensor([False, True, False]))
    torch.testing.assert_close(trace.traces[0].squeeze(), torch.tensor([2.5, 0.0, 0.5]))


def test_augmented_adamw_has_parameter_moments_and_does_not_decay_bias():
    edge = LocalPredictor(2, 1, "identity", std=0.1)
    with torch.no_grad():
        edge.weight.fill_(2.0)
        edge.bias.fill_(3.0)
    optimizer = AugmentedAdamW([edge], beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.1)
    direction = torch.tensor([[4.0, -2.0, 0.0]])
    optimizer.step([direction], learning_rate=0.01)
    torch.testing.assert_close(edge.weight, torch.tensor([[2.008, 1.988]]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(edge.bias, torch.tensor([3.0]), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.first[0], 0.1 * direction)
    torch.testing.assert_close(optimizer.second[0], 0.001 * direction.square())


def test_all_local_parameters_are_frozen_to_autograd():
    args = Args(hidden_size=4, pc_num_hidden_layers=2)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 2, std=0.1)
    assert all(not parameter.requires_grad for parameter in hierarchy.parameters())
    assert all(not parameter.requires_grad for parameter in output.parameters())


def test_truncation_bootstraps_from_final_observation():
    next_obs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    infos = {
        "final_observation": np.array([np.array([8.0, 9.0]), None], dtype=object),
        "_final_observation": np.array([True, False]),
    }
    result = bootstrap_observations(next_obs, np.array([True, False]), infos)
    np.testing.assert_array_equal(result[0], np.array([8.0, 9.0]))
    np.testing.assert_array_equal(result[1], next_obs[1])

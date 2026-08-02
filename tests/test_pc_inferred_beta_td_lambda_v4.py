from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

from cleanrl.ppo_continuous_action_pc_inferred_beta_td_lambda_v4 import (
    Agent,
    Args,
    AugmentedAdam,
    LocalPredictor,
    OutputPredictor,
    PCHierarchy,
    TraceBank,
    bounded_beta_parameters,
    bounded_beta_score,
)


def directional_finite_difference(fn, value, direction, eps=1e-5):
    return (fn(value + eps * direction) - fn(value - eps * direction)) / (2.0 * eps)


def test_bounded_beta_score_matches_directional_finite_difference():
    torch.manual_seed(0)
    raw = torch.randn(5, 6, dtype=torch.float64)
    action = torch.rand(5, 3, dtype=torch.float64).clamp(0.05, 0.95)
    alpha, beta, allocation, concentration = bounded_beta_parameters(raw, 1.0, 1.0, 32.0)
    score = bounded_beta_score(
        raw, alpha, beta, allocation, concentration, action, 1.0, 32.0
    )
    direction = torch.randn_like(raw)

    def objective(candidate):
        candidate_alpha, candidate_beta, _, _ = bounded_beta_parameters(
            candidate, 1.0, 1.0, 32.0
        )
        return Beta(candidate_alpha, candidate_beta).log_prob(action).sum()

    finite_difference = directional_finite_difference(objective, raw, direction)
    torch.testing.assert_close(finite_difference, (score * direction).sum(), rtol=3e-6, atol=3e-6)


def test_concentration_map_is_smooth_monotone_and_asymptotically_bounded():
    concentration_raw = torch.linspace(-20.0, 20.0, 1000, dtype=torch.float64)
    raw = torch.stack([torch.zeros_like(concentration_raw), concentration_raw], dim=1)
    alpha, beta, allocation, concentration = bounded_beta_parameters(raw, 1.0, 1.0, 32.0)
    assert torch.all(concentration[1:] > concentration[:-1])
    assert torch.all(concentration >= 1.0)
    assert torch.all(concentration < 32.0)
    assert concentration[0] < 1.000001
    assert concentration[-1] > 12.0
    torch.testing.assert_close(allocation, torch.full_like(allocation, 0.5))
    torch.testing.assert_close(alpha + beta, 2.0 + concentration)


@pytest.mark.parametrize(
    "edge_offset, concentration_min, concentration_max",
    [(0.0, 1.0, 32.0), (1.0, -1.0, 32.0), (1.0, 2.0, 2.0)],
)
def test_invalid_bounded_beta_geometry_is_rejected(edge_offset, concentration_min, concentration_max):
    with pytest.raises(ValueError):
        bounded_beta_parameters(
            torch.zeros(2, 4), edge_offset, concentration_min, concentration_max
        )


def test_free_states_have_exactly_zero_residual_without_clamps():
    torch.manual_seed(1)
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


def test_eliminated_output_contributes_force_but_no_hidden_curvature():
    torch.manual_seed(2)
    args = Args(hidden_size=4, pc_num_hidden_layers=2, pc_curvature_damping=0.07)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 3, std=0.2)
    observation = torch.randn(8, 3)
    states = hierarchy.initial_states(observation)
    terminal_score = torch.randn(8, 3)
    factors, _, _ = hierarchy.curvature_factors(states, args)
    recovered_top = factors[-1] @ factors[-1].T
    expected_top = (1.0 + args.pc_curvature_damping) * torch.eye(args.hidden_size)
    torch.testing.assert_close(recovered_top, expected_top, rtol=2e-6, atol=2e-6)

    nudged, _, _ = hierarchy.settle(
        observation, output, terminal_score, args.pc_actor_nudge, args
    )
    assert (nudged[-1] - states[-1]).norm() > 0


def test_sweeps_reduce_schur_eliminated_pc_energy():
    torch.manual_seed(12)
    args = Args(hidden_size=8, pc_num_hidden_layers=6, pc_inference_steps=10)
    hierarchy = PCHierarchy(5, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 4, std=0.2)
    observation = 0.2 * torch.randn(16, 5)
    terminal_score = torch.randn(16, 4)
    free = hierarchy.initial_states(observation)

    def energy(states):
        hidden = sum(
            0.5 * residual.square().sum(1)
            for residual in hierarchy.residuals(observation, states)
        )
        terminal = -args.pc_actor_nudge * (
            terminal_score * output(states[-1])
        ).sum(1)
        return hidden + terminal

    before = energy(free)
    settled, _, _ = hierarchy.settle(
        observation, output, terminal_score, args.pc_actor_nudge, args
    )
    after = energy(settled)
    assert after.sum() < before.sum()
    assert torch.all(after <= before + 1e-6)


def _pc_and_exact_direction(head_scale, nudge):
    torch.manual_seed(42)
    args = Args(
        hidden_size=16,
        pc_num_hidden_layers=6,
        pc_inference_steps=10,
        pc_actor_nudge=nudge,
    )
    hierarchy = PCHierarchy(7, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 4, std=0.01)
    with torch.no_grad():
        output.weight.mul_(head_scale)
    observation = 0.2 * torch.randn(16, 7)
    free = hierarchy.initial_states(observation)
    raw = output(free[-1])
    alpha, beta, allocation, concentration = bounded_beta_parameters(raw, 1.0, 1.0, 32.0)
    action = torch.rand(16, 2).clamp(0.05, 0.95)
    score = bounded_beta_score(
        raw, alpha, beta, allocation, concentration, action, 1.0, 32.0
    )
    states, _, _ = hierarchy.settle(observation, output, score, nudge, args)
    pc_directions = hierarchy.local_scores(observation, states, nudge)
    pc_directions.append(
        score.unsqueeze(2) * output.augmented_features(free[-1]).unsqueeze(1)
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
    exact_alpha, exact_beta, _, _ = bounded_beta_parameters(exact_raw, 1.0, 1.0, 32.0)
    objective = Beta(exact_alpha, exact_beta).log_prob(action).sum()
    gradients = torch.autograd.grad(objective, parameters)
    exact_flat = torch.cat(
        [
            torch.cat([weight_gradient, bias_gradient[:, None]], dim=1).flatten()
            for weight_gradient, bias_gradient in zip(gradients[::2], gradients[1::2])
        ]
    )
    return pc_flat, exact_flat


def test_eliminated_endpoint_tracks_exact_policy_score_and_has_a_nudge_limit():
    ordinary, exact = _pc_and_exact_direction(head_scale=1.0, nudge=0.05)
    small, small_exact = _pc_and_exact_direction(head_scale=1.0, nudge=0.005)
    assert F.cosine_similarity(ordinary, exact, dim=0) > 0.999
    assert F.cosine_similarity(small, small_exact, dim=0) > 0.999
    assert F.cosine_similarity(ordinary, small, dim=0) > 0.99999
    torch.testing.assert_close(exact, small_exact, rtol=0, atol=0)


def test_stress_head_remains_positively_aligned_without_fisher_spring():
    pc_direction, exact = _pc_and_exact_direction(head_scale=100.0, nudge=0.005)
    cosine = F.cosine_similarity(pc_direction, exact, dim=0)
    # This guards the ten-sweep approximation, not convergence: large heads expose
    # remaining finite-inference attenuation and motivate accelerated inference.
    assert cosine > 0.75


def test_agent_actor_and_critic_output_eligibilities_are_exact_behavior_scores():
    torch.manual_seed(7)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )
    args = Args(hidden_size=8, pc_num_hidden_layers=3, pc_inference_steps=5)
    agent = Agent(envs, args)
    observation = torch.randn(4, 5)
    actor_free = agent.actor_pc.initial_states(observation)
    action = torch.rand(4, 2).clamp(0.05, 0.95)
    score = agent.actor_terminal_score(actor_free[-1], action, args)
    _, _, actor_output_score, _ = agent.settle_actor(
        observation, action, args, collect_diagnostics=False
    )
    expected_actor = score.unsqueeze(2) * agent.actor_output.augmented_features(
        actor_free[-1]
    ).unsqueeze(1)
    torch.testing.assert_close(actor_output_score, expected_actor)

    critic_free = agent.critic_pc.initial_states(observation)
    _, _, critic_output_score, _ = agent.settle_critic(
        observation, args, collect_diagnostics=False
    )
    expected_critic = agent.critic_output.augmented_features(critic_free[-1]).unsqueeze(1)
    torch.testing.assert_close(critic_output_score, expected_critic)


def test_trace_uses_current_terminal_transition_before_reset():
    trace = TraceBank([torch.empty(3, 1)])
    trace.accumulate([torch.tensor([[1.0], [2.0], [3.0]])], decay=0.5)
    trace.accumulate([torch.tensor([[2.0], [0.0], [-1.0]])], decay=0.5)
    update = trace.modulated_mean(torch.tensor([0.0, 6.0, 0.0]))[0]
    torch.testing.assert_close(update, torch.tensor([2.0]))
    trace.reset(torch.tensor([False, True, False]))
    torch.testing.assert_close(trace.traces[0].squeeze(), torch.tensor([2.5, 0.0, 0.5]))


def test_augmented_adam_updates_weights_and_bias_without_decay():
    edge = LocalPredictor(2, 1, "identity", std=0.1)
    with torch.no_grad():
        edge.weight.fill_(2.0)
        edge.bias.fill_(3.0)
    optimizer = AugmentedAdam([edge], beta1=0.9, beta2=0.999, epsilon=1e-8)
    direction = torch.tensor([[4.0, -2.0, 0.0]])
    optimizer.step([direction], learning_rate=0.01)
    torch.testing.assert_close(edge.weight, torch.tensor([[2.01, 1.99]]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(edge.bias, torch.tensor([3.0]), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.first[0], 0.1 * direction)
    torch.testing.assert_close(optimizer.second[0], 0.001 * direction.square())

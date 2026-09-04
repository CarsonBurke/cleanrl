from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

from cleanrl.pc.ppo_continuous_action_pc_momentum_dynamics_td_lambda_v5 import (
    Agent,
    Args,
    AugmentedAdam,
    LocalPredictor,
    OutputPredictor,
    PCHierarchy,
    TraceBank,
    bounded_beta_parameters,
    bounded_beta_score,
    linearized_gradients,
    linearized_residuals,
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


def test_linearized_local_gradients_are_exact_energy_gradients():
    torch.manual_seed(1)
    args = Args(hidden_size=5, pc_num_hidden_layers=3)
    hierarchy = PCHierarchy(4, args.hidden_size, args).double()
    output = OutputPredictor(args.hidden_size, 2, std=0.2).double()
    observation = torch.randn(6, 4, dtype=torch.float64)
    free = hierarchy.initial_states(observation)
    derivatives, _ = hierarchy.response_geometry(free, args)
    weights = tuple(edge.weight for edge in hierarchy.edges)
    responses = [torch.randn_like(state, requires_grad=True) for state in free]
    terminal_score = torch.randn(6, 2, dtype=torch.float64)
    residuals = linearized_residuals(responses, weights, derivatives)
    energy = sum(0.5 * residual.square().sum() for residual in residuals) - (
        terminal_score * F.linear(responses[-1], output.weight)
    ).sum()
    autograd_gradients = torch.autograd.grad(energy, responses)
    local_gradients = linearized_gradients(
        residuals, weights, derivatives, output.weight, terminal_score
    )
    for actual, expected in zip(local_gradients, autograd_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_response_blocks_are_exact_per_example_local_hessians():
    torch.manual_seed(2)
    args = Args(hidden_size=4, pc_num_hidden_layers=3, pc_dynamics_block_damping=0.07)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    observation = torch.randn(7, 3)
    free = hierarchy.initial_states(observation)
    derivatives, factors = hierarchy.response_geometry(free, args)
    for layer_idx in range(len(hierarchy.edges) - 1):
        jacobian = hierarchy.edges[layer_idx + 1].weight.unsqueeze(0) * derivatives[
            layer_idx
        ].unsqueeze(1)
        expected = jacobian.transpose(1, 2) @ jacobian + 1.07 * torch.eye(args.hidden_size)
        torch.testing.assert_close(
            factors[layer_idx] @ factors[layer_idx].transpose(1, 2),
            expected,
            rtol=2e-6,
            atol=2e-6,
        )
    expected_top = 1.07 * torch.eye(args.hidden_size).expand(7, -1, -1)
    torch.testing.assert_close(
        factors[-1] @ factors[-1].transpose(1, 2), expected_top, rtol=2e-6, atol=2e-6
    )


def test_response_is_fresh_linear_and_zero_without_a_terminal_score():
    torch.manual_seed(3)
    args = Args(hidden_size=8, pc_num_hidden_layers=4)
    hierarchy = PCHierarchy(5, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 3, std=0.2)
    observation = torch.randn(9, 5)
    score = torch.randn(9, 3)
    free, residuals, _ = hierarchy.response(observation, output, score, args)
    free_twice, residuals_twice, _ = hierarchy.response(
        observation, output, 2.0 * score, args
    )
    _, residuals_repeat, _ = hierarchy.response(observation, output, score, args)
    _, residuals_zero, diagnostics_zero = hierarchy.response(
        observation, output, torch.zeros_like(score), args
    )
    for state, repeated_state in zip(free, free_twice):
        torch.testing.assert_close(state, repeated_state, rtol=0, atol=0)
    for residual, twice, repeated, zero in zip(
        residuals, residuals_twice, residuals_repeat, residuals_zero
    ):
        torch.testing.assert_close(twice, 2.0 * residual, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(repeated, residual, rtol=0, atol=0)
        torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
    torch.testing.assert_close(diagnostics_zero["response_rms"], torch.zeros(()))


def test_momentum_dynamics_lowers_response_energy_and_approaches_stationarity():
    torch.manual_seed(4)
    args = Args(hidden_size=16, pc_num_hidden_layers=6)
    hierarchy = PCHierarchy(7, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 4, std=0.2)
    observation = 0.2 * torch.randn(16, 7)
    score = torch.randn(16, 4)
    _, _, diagnostics = hierarchy.response(observation, output, score, args)
    assert diagnostics["energy_mean"] < 0
    assert diagnostics["convergence_ratio"] < 0.6
    assert torch.isfinite(diagnostics["velocity_rms"])


def _pc_and_exact_direction(head_scale):
    torch.manual_seed(42)
    args = Args(hidden_size=16, pc_num_hidden_layers=6)
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
    free, residuals, _ = hierarchy.response(
        observation, output, score, args, free_states=free
    )
    pc_directions = hierarchy.local_scores(observation, free, residuals)
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


@pytest.mark.parametrize("head_scale,minimum_cosine", [(1.0, 0.999), (25.0, 0.97), (100.0, 0.93)])
def test_momentum_response_tracks_exact_policy_gradient(head_scale, minimum_cosine):
    pc_direction, exact = _pc_and_exact_direction(head_scale)
    assert F.cosine_similarity(pc_direction, exact, dim=0) > minimum_cosine


def test_agent_output_eligibilities_use_exact_behavior_score_and_free_features():
    torch.manual_seed(7)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )
    args = Args(hidden_size=8, pc_num_hidden_layers=3, pc_dynamics_steps=8)
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


def test_augmented_adam_updates_without_decay():
    edge = LocalPredictor(2, 1, "identity", std=0.1)
    with torch.no_grad():
        edge.weight.fill_(2.0)
        edge.bias.fill_(3.0)
    optimizer = AugmentedAdam([edge], beta1=0.9, beta2=0.999, epsilon=1e-8)
    direction = torch.tensor([[4.0, -2.0, 0.0]])
    optimizer.step([direction], learning_rate=0.01)
    torch.testing.assert_close(edge.weight, torch.tensor([[2.01, 1.99]]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(edge.bias, torch.tensor([3.0]), rtol=0, atol=0)

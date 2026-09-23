"""CUDA contracts for one unified vector return critic and observed-segment TD."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_vector_unified_td_v16 import (
    Agent, ActorNaturalGradient, UnifiedCritic, basis_geometry, beta_geometry,
    beta_kl_reference, integrated_gain_kl, kernel_statistics, kernels_observed,
    response_credit, td_segments, vector_td_target,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), 'Run through mlq on CUDA'
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)


def test_unified_vector_prediction_and_expectation_match_explicit_fixed_basis():
    critic = UnifiedCritic(5, 3, width=32).cuda().double()
    torch.nn.init.normal_(critic.coefficient_head.weight, std=.02)
    obs = torch.randn(12, 5, device='cuda', dtype=torch.float64)
    age = torch.arange(12, device='cuda') * 100
    action = torch.rand(12, 3, device='cuda', dtype=torch.float64).clamp(.01, .99)
    scores, factor, basis = basis_geometry(action)
    powers = critic.powers(obs, age)
    coefficients = critic.coefficients(obs, age)
    assert coefficients.shape == (12, 4, 1 + 6 + powers.shape[1])
    basis_mean, basis_std = kernel_statistics(basis, powers)
    observed_kernel = kernels_observed(action, powers)
    features = torch.cat((torch.ones(12, 1, device='cuda', dtype=torch.float64), scores.flatten(1),
                          (observed_kernel - basis_mean) / basis_std), -1)
    predicted = critic.predict(obs, age, action)
    assert predicted.shape == (12, 4)
    manual = torch.einsum('bcf,bf->bc', coefficients, features) * (age < 1000)[:, None]
    torch.testing.assert_close(predicted, manual, rtol=1e-9, atol=1e-11)
    current = 1.1 + 5 * torch.rand_like(basis)
    alpha, beta = current.unbind(-1)
    basis_alpha, basis_beta = basis.unbind(-1)
    da = alpha.digamma() - (alpha + beta).digamma() - basis_alpha.digamma() + (basis_alpha + basis_beta).digamma()
    db = beta.digamma() - (alpha + beta).digamma() - basis_beta.digamma() + (basis_alpha + basis_beta).digamma()
    first = da / factor[..., 0]
    second = (db - factor[..., 1] * first) / factor[..., 2]
    current_mean, _ = kernel_statistics(current, powers)
    expected_features = torch.cat((torch.ones(12, 1, device='cuda', dtype=torch.float64),
                                   torch.stack((first, second), -1).flatten(1),
                                   (current_mean - basis_mean) / basis_std), -1)
    expected = torch.einsum('bcf,bf->bc', coefficients, expected_features) * (age < 1000)[:, None]
    integrated = critic.expected(obs, age, current)
    torch.testing.assert_close(integrated, expected, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(critic.expected(obs, age, basis), coefficients[..., 0] * (age < 1000)[:, None],
                               rtol=1e-9, atol=1e-11)
    assert (predicted[age >= 1000] == 0).all()
    assert (integrated[age >= 1000] == 0).all()


def test_every_component_trains_the_shared_trunk_and_state_conditioned_kernels():
    critic = UnifiedCritic(5, 4, width=32).cuda()
    torch.nn.init.normal_(critic.coefficient_head.weight, std=.02)
    obs = torch.randn(16, 5, device='cuda', requires_grad=True)
    age = torch.arange(16, device='cuda') * 30
    action = torch.rand(16, 4, device='cuda').clamp(.01, .99)
    powers = critic.powers(obs, age)
    assert powers.ndim == 4 and powers.shape[-2:] == (4, 2)
    assert ((powers > 0).any(-1).sum(-1) >= 3).any()
    assert not torch.allclose(powers[0], powers[1])
    predicted = critic.predict(obs, age, action)
    assert predicted.shape == (16, 5)
    shared_parameters = tuple(critic.trunk.parameters())
    for component in range(5):
        gradients = torch.autograd.grad(predicted[:, component].square().mean(), shared_parameters, retain_graph=True)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.square().sum() for gradient in gradients) > 0
    predicted.square().mean().backward()
    assert critic.power_head.weight.grad is not None
    assert torch.isfinite(critic.power_head.weight.grad).all()
    assert critic.power_head.weight.grad.square().sum() > 0


def test_constant_feature_drops_only_from_actor_gradient_not_vector_prediction():
    critic = UnifiedCritic(5, 3, width=32).cuda().double()
    torch.nn.init.normal_(critic.coefficient_head.weight, std=.02)
    obs = torch.randn(16, 5, device='cuda', dtype=torch.float64)
    age = torch.arange(16, device='cuda')
    action = torch.rand(16, 3, device='cuda', dtype=torch.float64).clamp(.01, .99)
    logits = torch.randn(16, 6, device='cuda', dtype=torch.float64, requires_grad=True)
    _, _, parameters = beta_geometry(logits.detach(), action)
    _, basis_factor, anchor = basis_geometry(action)
    coefficients = critic.coefficients(obs, age).detach()
    powers = critic.powers(obs, age).detach()
    _, basis_std = kernel_statistics(anchor, powers)
    old_mean, _ = kernel_statistics(parameters, powers)
    objective_coefficients = coefficients.sum(1)[:, 1:]
    credit = response_credit(parameters, basis_factor, objective_coefficients, powers, None, basis_std)
    gain, _ = integrated_gain_kl(logits, beta_kl_reference(parameters), basis_factor, objective_coefficients,
                                 powers, None, old_mean, basis_std, torch.ones(16, device='cuda', dtype=torch.float64))
    expected, = torch.autograd.grad(gain, logits)
    actual = torch.cat((credit[..., 0], credit[..., 1]), -1) * logits.detach().sigmoid() / 16
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    changed = coefficients.clone()
    changed[..., 0] += torch.arange(4, device='cuda', dtype=torch.float64)[None] + 1
    torch.testing.assert_close(changed.sum(1)[:, 1:], objective_coefficients, rtol=0, atol=0)
    assert (changed[..., 0] != coefficients[..., 0]).all()


@pytest.mark.parametrize('steps,segment_length', [(45, 32), (137, 32), (19, 64)])
def test_unified_segments_match_explicit_sums_with_real_episode_endpoints(steps, segment_length):
    environments, channels = 2, 4
    components = torch.randn(steps, environments, channels, device='cuda', dtype=torch.float64)
    boundaries = (tuple(value for value in (5, 20, 44, 98, 136) if value < steps),
                  tuple(value for value in (0, 34, 100) if value < steps))
    ends = torch.zeros(steps, environments, dtype=torch.bool, device='cuda')
    for environment, values in enumerate(boundaries):
        ends[list(values), environment] = True
    observed, endpoint, consumed, terminal = td_segments(components, ends, td_steps=segment_length)
    assert observed.shape == (steps * environments, channels)
    expected = torch.zeros_like(observed)
    for time in range(steps):
        for environment in range(environments):
            row = time * environments + environment
            next_end = next((value for value in boundaries[environment] if value >= time), steps - 1)
            count = min(segment_length, steps - time, next_end - time + 1)
            expected[row] = components[time:time + count, environment].sum(0)
            assert int(consumed[row]) == count
            assert int(endpoint[row]) == (time + count - 1) * environments + environment
            assert bool(terminal[row]) == ((time + count - 1) in boundaries[environment])
    torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-10)


def test_vector_td_target_stops_real_terminal_but_bootstraps_rollout_boundary():
    components = torch.tensor([[[1., -.1], [2., -.2]], [[3., -.3], [4., -.4]],
                               [[5., -.5], [6., -.6]]], device='cuda')
    ends = torch.zeros(3, 2, dtype=torch.bool, device='cuda')
    ends[1, 0] = True
    observed, endpoint, consumed, terminal = td_segments(components, ends, 32)
    following = torch.full_like(observed, 10., requires_grad=True)
    target = vector_td_target(observed, following, terminal)
    expected = observed + 10 * (~terminal)[:, None]
    torch.testing.assert_close(target, expected)
    assert not target.requires_grad and target.shape == (6, 2)
    assert int(consumed[0]) == 2 and bool(terminal[0])
    assert int(consumed[1]) == 3 and not bool(terminal[1])
    torch.testing.assert_close(target[0], components[:2, 0].sum(0))
    torch.testing.assert_close(target[1], components[:, 1].sum(0) + 10)
    snapshot = target.clone()
    with torch.no_grad():
        following.add_(100)
    torch.testing.assert_close(target, snapshot, rtol=0, atol=0)


def test_one_step_unified_segments_degenerate_to_vector_td():
    components = torch.randn(17, 3, 4, device='cuda')
    ends = torch.rand(17, 3, device='cuda') < .2
    observed, endpoint, consumed, terminal = td_segments(components, ends, 1)
    torch.testing.assert_close(observed, components.flatten(0, 1))
    torch.testing.assert_close(endpoint, torch.arange(51, device='cuda'))
    torch.testing.assert_close(consumed, torch.ones_like(consumed))
    torch.testing.assert_close(terminal, ends.flatten())
    following = torch.randn_like(observed, requires_grad=True)
    actual = vector_td_target(observed, following, terminal)
    torch.testing.assert_close(actual, observed + following.detach() * (~terminal)[:, None])
    assert not actual.requires_grad


def test_compiled_unified_fitting_and_two_actor_updates_use_one_vector_prediction():
    environment = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (5,)),
                                  single_action_space=gym.spaces.Box(-1., 1., (3,)))
    actor = Agent(environment).cuda().actor
    critic = UnifiedCritic(5, 3, width=32).cuda()
    optimizer = torch.optim.Adam(critic.parameters(), lr=.001, fused=True)
    steps, environments = 17, 2
    batch = steps * environments
    observations = torch.randn(batch, 5, device='cuda')
    next_observations = observations + .1 * torch.randn_like(observations)
    ages = (torch.arange(steps, device='cuda')[:, None] + torch.tensor([975, 983], device='cuda')[None]).flatten()
    ends = ages.reshape(steps, environments) == 999
    actions = torch.rand(batch, 3, device='cuda').clamp(.01, .99)
    components = torch.cat((.1 * torch.randn(batch, 1, device='cuda'), -.1 * (2 * actions - 1).square()), -1).reshape(steps, environments, 4)
    _, basis_factor, anchor = basis_geometry(actions)
    weights = torch.ones(batch, device='cuda')

    def fitting_loss(obs, age, observed_action, target):
        return (critic.predict(obs, age, observed_action) - target).square().mean()

    def actor_measure(obs, reference, factor, coefficients, powers, mean, std):
        return integrated_gain_kl(actor(obs), reference, factor, coefficients, powers, None, mean, std, weights)

    policy = torch.compile(actor.forward, fullgraph=True, mode='reduce-overhead')
    segment_fn = torch.compile(td_segments, fullgraph=True, mode='reduce-overhead')
    expected_fn = torch.compile(critic.expected, fullgraph=True, mode='reduce-overhead')
    target_fn = torch.compile(vector_td_target, fullgraph=True, mode='reduce-overhead')
    fit = torch.compile(fitting_loss, fullgraph=True, mode='reduce-overhead')
    measure_fn = torch.compile(actor_measure, fullgraph=True, mode='reduce-overhead')
    solver = ActorNaturalGradient(actor, budget=.01, cg_iterations=20, line_search_steps=16, compile=True)
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            logits = policy(observations).clone()
            _, collection_factor, parameters = beta_geometry(logits, actions)
            observed, endpoint, consumed, terminal = (value.clone() for value in segment_fn(components, ends, 8))
            endpoint_observations = next_observations[endpoint]
            next_logits = policy(endpoint_observations).clone()
            _, _, next_parameters = beta_geometry(next_logits, actions)
            continuation = expected_fn(endpoint_observations, ages + consumed, next_parameters).clone()
            target = target_fn(observed, continuation, terminal).clone()
            frozen = target.clone()
        for fitting_step in range(2):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            loss = fit(observations, ages, actions, target)
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in critic.parameters())
            if fitting_step == 1:
                assert critic.power_head.weight.grad.square().sum() > 0
            assert all(parameter.grad is None for parameter in actor.parameters())
            optimizer.step()
            del loss
        torch.testing.assert_close(target, frozen, rtol=0, atol=0)
        with torch.no_grad():
            coefficients = critic.coefficients(observations, ages).sum(1)[:, 1:].clone()
            powers = critic.powers(observations, ages).clone()
            old_mean, _ = kernel_statistics(parameters, powers)
            _, basis_std = kernel_statistics(anchor, powers)
            old_mean, basis_std = old_mean.clone(), basis_std.clone()
            credit = response_credit(parameters, basis_factor, coefficients, powers, None, basis_std).clone()
            reference = beta_kl_reference(parameters)

        def measure():
            gain, kl = measure_fn(observations, reference, basis_factor, coefficients, powers, old_mean, basis_std)
            return gain.clone(), kl.clone()

        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            old_gain, old_kl = measure()
        assert abs(float(old_gain)) < 1e-6 and abs(float(old_kl)) < 1e-10
        result = solver.step(observations, logits, collection_factor, credit, weights, measure)
        assert result['policy/accepted_gain'] > 0
        assert 0 < result['policy/exact_joint_kl'] <= solver.budget


def test_production_shape_static_compiled_unified_segments_and_retained_targets():
    with torch.no_grad():
        steps, environments, channels = 2048, 16, 7
        positions = torch.arange(steps, device='cuda')[:, None]
        phases = (torch.arange(environments, device='cuda') * 1000 // environments)[None]
        ages = (positions + phases) % 1000
        ends = ages == 999
        components = torch.randn(steps, environments, channels, device='cuda')
        expected = td_segments(components, ends, 32)
        torch._dynamo.reset()
        segment_fn = torch.compile(td_segments, fullgraph=True, dynamic=False, mode='reduce-overhead')
        target_fn = torch.compile(vector_td_target, fullgraph=True, dynamic=False, mode='reduce-overhead')
        torch.compiler.cudagraph_mark_step_begin()
        actual = tuple(value.clone() for value in segment_fn(components, ends, 32))
        assert actual[0].shape == (32768, 7)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-5)
        observed, endpoint, consumed, terminal = actual
        expected_count = torch.minimum(torch.full_like(ages, 32), torch.minimum(steps - positions, 1000 - ages))
        torch.testing.assert_close(consumed, expected_count.flatten())
        expected_endpoint = ((positions + expected_count - 1) * environments + torch.arange(environments, device='cuda')[None]).flatten()
        torch.testing.assert_close(endpoint, expected_endpoint)
        torch.testing.assert_close(terminal, ends.flatten()[endpoint])
        following = torch.randn_like(observed)
        target = target_fn(observed, following, terminal).clone()
        torch.testing.assert_close(target, vector_td_target(observed, following, terminal))
        frozen_target = target.clone()
        torch.compiler.cudagraph_mark_step_begin()
        changed = tuple(value.clone() for value in segment_fn(components + 1, ends, 32))
        torch.testing.assert_close(changed[0] - observed, consumed[:, None].expand_as(observed).float(), rtol=1e-6, atol=1e-5)
        target_fn(changed[0], following, changed[3])
        torch.testing.assert_close(target, frozen_target, rtol=0, atol=0)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-5)

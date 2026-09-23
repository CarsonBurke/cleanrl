"""CUDA contracts for state-conditioned vector TD return-response learning.

Run through mlq. No reduced training or fixed-policy promotion gates.
"""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_vector_td_response_v14 import (
    Agent, ActorNaturalGradient, ResponseCritic, basis_geometry, beta_geometry, beta_kl_reference,
    integrated_gain_kl, kernel_moments, kernel_statistics, kernels_observed,
    query_horizons, resolved_ages, response_credit, td_targets,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), 'Run through mlq on CUDA'
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)


def test_state_conditioned_kernel_moments_and_parameter_derivatives_match_fp64_autograd():
    parameters = (1.1 + 5 * torch.rand(8, 3, 2, device='cuda', dtype=torch.float64)).requires_grad_()
    powers = (.2 + torch.rand(8, 4, 3, 2, device='cuda', dtype=torch.float64)).requires_grad_()
    mean, derivative = kernel_moments(parameters, powers)
    alpha, beta = parameters.unbind(-1)
    p, q = powers.unbind(-1)
    expected = (torch.lgamma(alpha[:, None] + p) + torch.lgamma(beta[:, None] + q)
                - torch.lgamma(alpha[:, None] + beta[:, None] + p + q)
                - torch.lgamma(alpha[:, None]) - torch.lgamma(beta[:, None])
                + torch.lgamma(alpha[:, None] + beta[:, None])).sum(-1).exp()
    torch.testing.assert_close(mean, expected, rtol=1e-11, atol=1e-12)
    assert derivative.shape == (8, 4, 3, 2)
    for kernel in range(powers.shape[1]):
        expected_derivative, = torch.autograd.grad(expected[:, kernel].sum(), parameters, retain_graph=True)
        torch.testing.assert_close(derivative[:, kernel], expected_derivative, rtol=1e-10, atol=1e-12)
    actual_parameter_grad, actual_power_grad = torch.autograd.grad(mean.sum(), (parameters, powers))
    expected_parameter_grad, expected_power_grad = torch.autograd.grad(expected.sum(), (parameters, powers))
    torch.testing.assert_close(actual_parameter_grad, expected_parameter_grad, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual_power_grad, expected_power_grad, rtol=1e-10, atol=1e-12)
    assert actual_power_grad.square().sum() > 0


def test_dynamic_kernel_powers_retain_static_compatibility_and_three_action_interaction():
    parameters = 1.1 + 4 * torch.rand(16, 3, 2, device='cuda', dtype=torch.float64)
    static = torch.tensor([[[1., 0.], [1., 0.], [1., 0.]],
                           [[2., 0.], [0., 0.], [0., 0.]]], device='cuda', dtype=torch.float64)
    dynamic = static[None].expand(16, -1, -1, -1).clone()
    for actual, expected in zip(kernel_moments(parameters, dynamic), kernel_moments(parameters, static)):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    mean, _ = kernel_moments(parameters, dynamic)
    alpha, beta = parameters.unbind(-1)
    torch.testing.assert_close(mean[:, 0], (alpha / (alpha + beta)).prod(-1), rtol=1e-11, atol=1e-12)
    actions = (.1 + .8 * torch.rand(16, 3, device='cuda', dtype=torch.float64)).requires_grad_()
    observed = kernels_observed(actions, dynamic)
    torch.testing.assert_close(observed[:, 0], actions.prod(-1))
    first, = torch.autograd.grad(observed[:, 0].sum(), actions, create_graph=True)
    second, = torch.autograd.grad(first[:, 0].sum(), actions, create_graph=True)
    third, = torch.autograd.grad(second[:, 1].sum(), actions)
    # An additive/pairwise-only feature family cannot produce this derivative.
    torch.testing.assert_close(third[:, 2], torch.ones(16, device='cuda', dtype=torch.float64), rtol=1e-11, atol=1e-12)


def test_td_targets_preserve_channels_and_stop_terminal_or_one_step_bootstrap():
    components = torch.tensor([[2., -.3, -.7], [4., -.1, -.5], [1., -.8, -.9]], device='cuda')
    horizons = torch.tensor([[0, 1, 2, 7, 0, 5], [1, 2, 3, 0, 4, 1], [0, 1, 2, 3, 4, 5]], device='cuda')
    ends = torch.tensor([False, False, True], device='cuda')
    following = torch.randn(3, 6, 3, device='cuda', requires_grad=True)
    actual = td_targets(components, following, horizons, ends)
    continuation = (horizons > 1) & ~ends[:, None]
    expected = components[:, None] * (horizons > 0)[..., None] + following.detach() * continuation[..., None]
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3, 6, 3) and not actual.requires_grad
    assert (actual[horizons == 0] == 0).all()
    torch.testing.assert_close(actual[2, 1:], components[2].expand(5, -1))
    saved = actual.clone()
    with torch.no_grad():
        following[..., 1].add_(100)
    torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    changed = td_targets(components, following, horizons, ends)
    torch.testing.assert_close(changed[..., 0], saved[..., 0], rtol=0, atol=0)
    torch.testing.assert_close(changed[..., 2], saved[..., 2], rtol=0, atol=0)
    torch.testing.assert_close(changed[..., 1] - saved[..., 1], 100 * continuation.float(), rtol=1e-6, atol=1e-5)


def test_query_horizons_keep_full_remaining_horizon_and_zero_terminal_queries():
    ages = torch.tensor([0, 1, 500, 990, 999, 1000], device='cuda')
    actual = query_horizons(ages)
    remaining = 1000 - ages
    assert actual.shape == (6, 6)
    assert not actual.is_floating_point()
    assert (actual >= 0).all() and (actual <= remaining[:, None]).all()
    torch.testing.assert_close(actual[:, 0], remaining.clamp_max(1))
    torch.testing.assert_close(actual[:, -1], remaining)
    assert (actual[-1] == 0).all()
    low = query_horizons(ages, fractions=torch.full((6, 4), .01, device='cuda'))
    high = query_horizons(ages, fractions=torch.full((6, 4), .99, device='cuda'))
    assert (low <= high).all()
    assert (low[:3, 1:-1] < high[:3, 1:-1]).any()
    torch.testing.assert_close(low[:, [0, -1]], high[:, [0, -1]])


def test_resolved_ages_infer_only_from_real_observed_terminal():
    ages = torch.tensor([[-1, -1], [-1, -1], [-1, 0], [-1, 1], [0, 2],
                         [1, 3], [2, 4], [3, 5], [4, 0], [5, 1]], device='cuda')
    ends = torch.zeros(10, 2, dtype=torch.bool, device='cuda')
    ends[[3, 9], 0] = True
    ends[[1, 7], 1] = True
    actual = resolved_ages(ages, ends, horizon=6).reshape_as(ages)
    expected = torch.tensor([[2, 4], [3, 5], [4, 0], [5, 1], [0, 2],
                            [1, 3], [2, 4], [3, 5], [4, 0], [5, 1]], device='cuda')
    torch.testing.assert_close(actual, expected)
    unknown = torch.full((4, 2), -1, device='cuda')
    unresolved = resolved_ages(unknown, torch.zeros_like(unknown, dtype=torch.bool), horizon=6)
    assert (unresolved == -1).all()


def test_production_shape_compiled_td_targets_preserve_frozen_vector_snapshot():
    with torch.no_grad():
        batch, channels = 32768, 7
        ages = torch.arange(batch, device='cuda') % 1001
        horizons = query_horizons(ages)
        components = torch.randn(batch, channels, device='cuda')
        following = torch.randn(batch, 6, channels, device='cuda')
        ends = (torch.arange(batch, device='cuda') % 17) == 0
        expected = td_targets(components, following, horizons, ends)
        # The first trace must have the actual training dimensions, without
        # automatic dynamic-shape decisions inherited from earlier toy calls.
        torch._dynamo.reset()
        compiled = torch.compile(td_targets, fullgraph=True, dynamic=False, mode='reduce-overhead')
        torch.compiler.cudagraph_mark_step_begin()
        actual = compiled(components, following, horizons, ends).clone()
        assert actual.shape == (32768, 6, 7)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert (actual[horizons == 0] == 0).all()
        torch.compiler.cudagraph_mark_step_begin()
        changed = compiled(components, following + 1, horizons, ends).clone()
        continuation = ((horizons > 1) & ~ends[:, None])[..., None].expand_as(actual)
        torch.testing.assert_close(changed - actual, continuation.float(), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fixed_reference_expected_profile_integrates_current_policy_without_redefining_critic():
    critic = ResponseCritic(5, 3, width=32).cuda().double()
    torch.nn.init.normal_(critic.response_head.weight, std=.02)
    torch.nn.init.normal_(critic.baseline_head.weight, std=.02)
    observation = torch.randn(12, 5, device='cuda', dtype=torch.float64)
    age = torch.arange(12, device='cuda') * 50
    horizons = query_horizons(age)
    horizons[:, 2] = 0
    actions = torch.rand(12, 3, device='cuda', dtype=torch.float64).clamp(.01, .99)
    scores, factor, anchor = basis_geometry(actions)
    powers = critic.powers(observation, age)
    baseline = critic.baseline(observation, age, horizons)
    coefficients = critic.coefficients(observation, age, horizons)
    fixed_prediction = critic.response_profile(observation, age, horizons, scores, actions, anchor).detach().clone()
    basis_mean, basis_std = kernel_statistics(anchor, powers)
    anchor_a, anchor_b = anchor.unbind(-1)
    anchor_total = anchor_a + anchor_b
    anchor_log_a = anchor_a.digamma() - anchor_total.digamma()
    anchor_log_b = anchor_b.digamma() - anchor_total.digamma()
    first_policy = 1.2 + 4 * torch.rand_like(anchor)
    second_policy = first_policy + torch.tensor([2., .1], device='cuda', dtype=torch.float64)
    predictions = []
    for parameters in (first_policy, second_policy):
        alpha, beta = parameters.unbind(-1)
        total = alpha + beta
        da = alpha.digamma() - total.digamma() - anchor_log_a
        db = beta.digamma() - total.digamma() - anchor_log_b
        first = da / factor[..., 0]
        second = (db - factor[..., 1] * first) / factor[..., 2]
        current_mean, _ = kernel_statistics(parameters, powers)
        features = torch.cat((torch.stack((first, second), -1).flatten(1),
                              (current_mean - basis_mean) / basis_std), -1)
        expected = baseline + torch.einsum('bhcf,bf->bhc', coefficients, features)
        actual = critic.expected_profile(observation, age, horizons, parameters)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
        assert (actual[horizons == 0] == 0).all()
        predictions.append(actual)
    assert (predictions[0] - predictions[1]).square().sum() > 1e-6
    torch.testing.assert_close(critic.expected_profile(observation, age, horizons, anchor), baseline,
                               rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(critic.response_profile(observation, age, horizons, scores, actions, anchor),
                               fixed_prediction, rtol=0, atol=0)
    assert not torch.allclose(predictions[0], baseline)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_fixed_basis_response_credit_matches_current_policy_integrated_gradient(dtype):
    critic = ResponseCritic(5, 3, width=32).cuda().to(dtype)
    observation = torch.randn(16, 5, device='cuda', dtype=dtype)
    age = torch.arange(16, device='cuda')
    logits = torch.randn(16, 6, device='cuda', dtype=dtype, requires_grad=True)
    actions = torch.rand(16, 3, device='cuda', dtype=dtype).clamp(.01, .99)
    _, _, parameters = beta_geometry(logits.detach(), actions)
    _, basis_factor, anchor = basis_geometry(actions)
    coefficients = torch.randn(16, critic.features, device='cuda', dtype=dtype)
    powers = critic.powers(observation, age).detach()
    old_mean, _ = kernel_statistics(parameters, powers)
    _, basis_std = kernel_statistics(anchor, powers)
    weights = torch.rand(16, device='cuda', dtype=dtype) + .1
    reference = beta_kl_reference(parameters)
    gain, kl = integrated_gain_kl(logits, reference, basis_factor, coefficients, powers,
                                  None, old_mean, basis_std, weights)
    assert abs(float(gain.detach())) < 1e-10 and abs(float(kl.detach())) < 1e-10
    actual, = torch.autograd.grad(gain, logits)
    eta = response_credit(parameters, basis_factor, coefficients, powers, None, basis_std)
    expected = torch.cat((eta[..., 0], eta[..., 1]), -1) * logits.detach().sigmoid()
    expected = expected * (weights / weights.sum())[:, None]
    torch.testing.assert_close(actual, expected, rtol=2e-4 if dtype == torch.float32 else 1e-9,
                               atol=2e-7 if dtype == torch.float32 else 1e-11)


def test_state_conditioned_powers_have_higher_order_support_and_trainable_state_dependence():
    critic = ResponseCritic(5, 4, width=32).cuda()
    observation = torch.randn(16, 5, device='cuda', requires_grad=True)
    age = torch.arange(16, device='cuda') * 40
    powers = critic.powers(observation, age)
    assert powers.ndim == 4 and powers.shape[0] == 16 and powers.shape[-2:] == (4, 2)
    active_actions = (powers > 0).any(-1).sum(-1)
    assert (active_actions >= 3).any() and (active_actions == 4).any()
    gradient, = torch.autograd.grad(powers.sum(), observation, retain_graph=True)
    assert torch.isfinite(gradient).all() and gradient.square().sum() > 0
    assert not torch.allclose(powers[0], powers[1])
    powers.square().mean().backward()
    assert critic.power_head.weight.grad is not None
    assert torch.isfinite(critic.power_head.weight.grad).all()
    assert critic.power_head.weight.grad.square().sum() > 0


def test_zero_horizon_masks_baseline_response_and_policy_expectation():
    critic = ResponseCritic(5, 3, width=32).cuda()
    torch.nn.init.normal_(critic.baseline_head.weight, std=.02)
    torch.nn.init.normal_(critic.response_head.weight, std=.02)
    observation = torch.randn(16, 5, device='cuda')
    age = torch.arange(16, device='cuda') + 985
    horizons = query_horizons(age)
    horizons[:, 2] = 0
    actions = torch.rand(16, 3, device='cuda').clamp(.01, .99)
    scores, _, anchor = basis_geometry(actions)
    parameters = 1.2 + torch.rand(16, 3, 2, device='cuda')
    baseline = critic.baseline(observation, age, horizons)
    response = critic.response_profile(observation, age, horizons, scores, actions, anchor)
    expectation = critic.expected_profile(observation, age, horizons, parameters)
    assert baseline.shape == response.shape == expectation.shape == (16, 6, 4)
    for value in (baseline, response, expectation):
        assert (value[horizons == 0] == 0).all()
    assert (expectation[horizons > 0] != 0).any()


def test_compiled_joint_vector_fitting_and_two_frozen_basis_actor_updates():
    environment = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (5,)),
                                  single_action_space=gym.spaces.Box(-1., 1., (3,)))
    actor = Agent(environment).cuda().actor
    critic = ResponseCritic(5, 3, width=32).cuda()
    optimizer = torch.optim.Adam(critic.parameters(), lr=.001, fused=True)
    observations = torch.randn(128, 5, device='cuda')
    next_observations = observations + .1 * torch.randn_like(observations)
    ages = torch.arange(128, device='cuda')
    horizons = query_horizons(ages)
    actions = torch.rand(128, 3, device='cuda').clamp(.01, .99)
    components = torch.cat((.1 * torch.randn(128, 1, device='cuda'), -.1 * (2 * actions - 1).square()), -1)
    ends = torch.zeros(128, dtype=torch.bool, device='cuda')
    ends[::17] = True
    weights = torch.ones(128, device='cuda')
    scores, basis_factor, anchor = basis_geometry(actions)

    def fitting_loss(obs, age, queries, observed, score, basis, target):
        prediction = critic.baseline(obs, age, queries) + critic.response_profile(obs, age, queries, score, observed, basis)
        return (prediction - target).square().mean()

    def actor_measure(obs, reference, factor, coefficients, powers, mean, std):
        return integrated_gain_kl(actor(obs), reference, factor, coefficients, powers, None, mean, std, weights)

    policy = torch.compile(actor.forward, fullgraph=True, mode='reduce-overhead')
    predict_expected = torch.compile(critic.expected_profile, fullgraph=True, mode='reduce-overhead')
    target_fn = torch.compile(td_targets, fullgraph=True, mode='reduce-overhead')
    fit = torch.compile(fitting_loss, fullgraph=True, mode='reduce-overhead')
    measure_fn = torch.compile(actor_measure, fullgraph=True, mode='reduce-overhead')
    solver = ActorNaturalGradient(actor, budget=.01, cg_iterations=20, line_search_steps=16, compile=True)
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            logits = policy(observations).clone()
            _, collection_factor, parameters = beta_geometry(logits, actions)
            next_logits = policy(next_observations).clone()
            _, _, next_parameters = beta_geometry(next_logits, actions)
            next_expected = predict_expected(next_observations, ages + 1, (horizons - 1).clamp_min(0), next_parameters).clone()
            target = target_fn(components, next_expected, horizons, ends).clone()
            frozen = target.clone()
        for fitting_step in range(2):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            loss = fit(observations, ages, horizons, actions, scores, anchor, target)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
            if fitting_step == 1:
                assert critic.power_head.weight.grad.square().sum() > 0
            assert all(p.grad is None for p in actor.parameters())
            optimizer.step()
            del loss
        torch.testing.assert_close(target, frozen, rtol=0, atol=0)
        with torch.no_grad():
            # Overlapping horizons are separate supervision queries. Only the
            # complete remaining-horizon response defines the return objective.
            coefficients = critic.coefficients(observations, ages, horizons[:, -1:])[:, 0].sum(1).clone()
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

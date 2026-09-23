"""CUDA contracts for vector response targets and exact Beta expectations.

Run through mlq. These are numerical/semantic checks, not training gates.
"""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_vector_return_response_v13 import (
    Agent, ActorNaturalGradient, ResponseCritic, band_lengths, beta_geometry, beta_kl_reference,
    integrated_gain_kl, kernel_moments, kernel_statistics, kernels_observed, reward_profiles, response_credit,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), 'Run through mlq on CUDA'
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)


def test_reward_profiles_preserve_actual_signed_rewards_and_episode_boundaries():
    steps, environments, action_dim, horizon = 10, 2, 2, 6
    edges = (0, 1, 3, 6)
    rewards = torch.arange(steps * environments, device='cuda', dtype=torch.float64).reshape(steps, environments) / 7 - 1
    actions = torch.linspace(-.8, .9, steps * environments * action_dim,
                             device='cuda', dtype=torch.float64).reshape(steps, environments, action_dim)
    ends = torch.zeros(steps, environments, dtype=torch.bool, device='cuda')
    ends[[3, 9], 0] = True
    ends[[1, 7], 1] = True
    ages = torch.tensor([[-1, -1], [-1, -1], [-1, 0], [-1, 1], [0, 2],
                         [1, 3], [2, 4], [3, 5], [4, 0], [5, 1]], device='cuda')
    actual, valid, resolved, lengths = reward_profiles(rewards, actions, ends, ages, .2, edges=edges, horizon=horizon)
    assert actual.shape == (steps * environments, 3, action_dim + 1)
    assert valid.shape == resolved.shape == (steps * environments,)
    assert lengths.shape == (steps * environments, 3)
    expected_ages = torch.tensor([[2, 4], [3, 5], [4, 0], [5, 1], [0, 2],
                                  [1, 3], [2, 4], [3, 5], [4, 0], [5, 1]], device='cuda')
    torch.testing.assert_close(resolved.reshape(steps, environments), expected_ages)
    components = torch.cat(((rewards + .2 * actions.square().sum(-1))[..., None], -.2 * actions.square()), -1)
    expected = torch.zeros_like(actual).reshape(steps, environments, 3, action_dim + 1)
    expected_lengths = torch.zeros_like(lengths).reshape(steps, environments, 3)
    expected_valid = torch.ones_like(valid).reshape(steps, environments)
    expected_valid[8:, 1] = False
    for environment, boundaries in enumerate(((3, 9), (1, 7))):
        for time in range(steps):
            ending = next((boundary for boundary in boundaries if boundary >= time), None)
            if ending is None:
                continue
            for band, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                begin, stop = time + lo, min(time + hi, ending + 1)
                if begin < stop:
                    expected[time, environment, band] = components[begin:stop, environment].sum(0)
                    expected_lengths[time, environment, band] = stop - begin
            torch.testing.assert_close(actual.reshape_as(expected)[time, environment].sum(),
                                       rewards[time:ending + 1, environment].sum(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(valid.reshape_as(expected_valid), expected_valid)
    torch.testing.assert_close(actual.reshape_as(expected), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(lengths.reshape_as(expected_lengths), expected_lengths)
    assert (actual[..., 1:] <= 0).all()


def test_reward_profiles_do_not_bootstrap_right_censored_episodes():
    rewards = torch.ones(4, 2, device='cuda')
    actions = torch.zeros(4, 2, 1, device='cuda')
    ends = torch.zeros(4, 2, dtype=torch.bool, device='cuda')
    ages = torch.arange(4, device='cuda')[:, None].expand(-1, 2)
    target, valid, resolved, lengths = reward_profiles(rewards, actions, ends, ages, .1,
                                                       edges=(0, 1, 3, 6), horizon=6)
    assert not valid.any()
    assert (target == 0).all() and (lengths == 0).all()
    torch.testing.assert_close(resolved.reshape_as(ages), ages)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_kernel_expectations_match_exact_beta_polynomial_moments(dtype):
    parameters = 1.1 + 4 * torch.rand(32, 2, 2, device='cuda', dtype=dtype)
    powers = torch.tensor([[[0., 0.], [0., 0.]],
                           [[1., 0.], [0., 0.]],
                           [[2., 0.], [0., 0.]],
                           [[1., 1.], [0., 0.]],
                           [[1., 0.], [0., 1.]]], device='cuda', dtype=dtype)
    actual, derivatives = kernel_moments(parameters, powers)
    alpha, beta = parameters.double().unbind(-1)
    total = alpha + beta
    expected = torch.stack((torch.ones_like(alpha[:, 0]), alpha[:, 0] / total[:, 0],
                            alpha[:, 0] * (alpha[:, 0] + 1) / (total[:, 0] * (total[:, 0] + 1)),
                            alpha[:, 0] * beta[:, 0] / (total[:, 0] * (total[:, 0] + 1)),
                            alpha[:, 0] / total[:, 0] * beta[:, 1] / total[:, 1]), -1)
    assert actual.dtype == derivatives.dtype == dtype
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6 if dtype == torch.float32 else 1e-11,
                               atol=1e-8 if dtype == torch.float32 else 1e-12)
    assert derivatives.shape == (32, 5, 2, 2)
    torch.testing.assert_close(derivatives[:, 0], torch.zeros_like(derivatives[:, 0]), rtol=0, atol=1e-12)


def test_kernel_expectation_derivatives_match_fp64_autograd():
    parameters = (1.1 + 6 * torch.rand(8, 3, 2, device='cuda', dtype=torch.float64)).requires_grad_()
    powers = 2 * torch.rand(5, 3, 2, device='cuda', dtype=torch.float64)
    actual, derivatives = kernel_moments(parameters, powers)
    alpha, beta = parameters.unbind(-1)
    p, q = powers.unbind(-1)
    log_moment = (torch.lgamma(alpha[:, None] + p) + torch.lgamma(beta[:, None] + q)
                  - torch.lgamma(alpha[:, None] + beta[:, None] + p + q)
                  - torch.lgamma(alpha[:, None]) - torch.lgamma(beta[:, None])
                  + torch.lgamma(alpha[:, None] + beta[:, None])).sum(-1)
    expected = log_moment.exp()
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    for kernel in range(powers.shape[0]):
        derivative, = torch.autograd.grad(expected[:, kernel].sum(), parameters, retain_graph=True)
        torch.testing.assert_close(derivatives[:, kernel], derivative, rtol=1e-10, atol=1e-12)


def test_observed_kernel_values_match_direct_products_without_action_reduction():
    actions = torch.tensor([[.2, .7], [.8, .3]], device='cuda', dtype=torch.float64)
    powers = torch.tensor([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]],
                           [[.5, 1.5], [1.25, .75]]], device='cuda', dtype=torch.float64)
    actual = kernels_observed(actions, powers)
    expected = (actions[:, None].pow(powers[None, ..., 0])
                * (1 - actions[:, None]).pow(powers[None, ..., 1])).prod(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual.shape == (2, 3)
    assert not torch.equal(actual[:, 0], actual[:, 1])


def test_analytic_kernel_means_keep_gradients_to_learned_shapes():
    parameters = (1.1 + 6 * torch.rand(8, 3, 2, device='cuda', dtype=torch.float64)).requires_grad_()
    powers = (.2 + torch.rand(5, 3, 2, device='cuda', dtype=torch.float64)).requires_grad_()
    actual, _ = kernel_moments(parameters, powers)
    actual_derivatives = torch.autograd.grad(actual.sum(), (parameters, powers))
    alpha, beta = parameters.unbind(-1)
    p, q = powers.unbind(-1)
    expected = (torch.lgamma(alpha[:, None] + p) + torch.lgamma(beta[:, None] + q)
                - torch.lgamma(alpha[:, None] + beta[:, None] + p + q)
                - torch.lgamma(alpha[:, None]) - torch.lgamma(beta[:, None])
                + torch.lgamma(alpha[:, None] + beta[:, None])).sum(-1).exp()
    expected_derivatives = torch.autograd.grad(expected.sum(), (parameters, powers))
    for result, reference in zip(actual_derivatives, expected_derivatives):
        torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-12)


def test_sparse_kernel_moments_and_derivative_scatter_match_dense_features():
    parameters = 1.1 + 4 * torch.rand(16, 3, 2, device='cuda', dtype=torch.float64)
    indices = torch.tensor([[0, 0], [1, 2], [2, 0]], device='cuda')
    sparse = torch.tensor([[[2., 0.], [0., 0.]], [[.7, .2], [1.1, .3]],
                           [[.5, .8], [.2, .4]]], device='cuda', dtype=torch.float64)
    dense = torch.zeros(3, 3, 2, device='cuda', dtype=torch.float64)
    dense.scatter_add_(1, indices[..., None].expand(-1, -1, 2), sparse)
    actual_mean, actual_derivative = kernel_moments(parameters, sparse, indices)
    expected_mean, expected_derivative = kernel_moments(parameters, dense)
    torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual_derivative, expected_derivative, rtol=1e-10, atol=1e-12)
    actions = torch.rand(16, 3, device='cuda', dtype=torch.float64).clamp(.01, .99)
    torch.testing.assert_close(kernels_observed(actions, sparse, indices), kernels_observed(actions, dense))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_response_credit_matches_gradient_of_integrated_objective(dtype):
    critic = ResponseCritic(5, 3, width=32).cuda().to(dtype)
    logits = torch.randn(32, 6, device='cuda', dtype=dtype, requires_grad=True)
    actions = torch.rand(32, 3, device='cuda', dtype=dtype).clamp(.01, .99)
    _, factor, parameters = beta_geometry(logits.detach(), actions)
    coefficients = torch.randn(32, critic.features, device='cuda', dtype=dtype)
    powers = critic.powers().detach()
    mean, std = kernel_statistics(parameters, powers, critic.kernel_indices)
    assert mean.dtype == std.dtype == torch.float64
    weights = torch.rand(32, device='cuda', dtype=dtype)
    reference = beta_kl_reference(parameters)
    gain, kl = integrated_gain_kl(logits, reference, factor, coefficients, powers,
                                  critic.kernel_indices, mean, std, weights)
    assert abs(float(gain.detach())) < 1e-10 and abs(float(kl.detach())) < 1e-10
    actual, = torch.autograd.grad(gain, logits)
    eta = response_credit(parameters, factor, coefficients, powers, critic.kernel_indices, std)
    expected = torch.cat((eta[..., 0], eta[..., 1]), -1) * logits.detach().sigmoid()
    expected = expected * (weights / weights.sum())[:, None]
    torch.testing.assert_close(actual, expected, rtol=2e-4 if dtype == torch.float32 else 1e-9,
                               atol=2e-7 if dtype == torch.float32 else 1e-11)


def test_response_profile_has_exact_terminal_support_and_learns_kernel_shapes():
    critic = ResponseCritic(5, 3, width=32).cuda()
    torch.nn.init.normal_(critic.response_head.weight, std=.02)
    observation = torch.randn(64, 5, device='cuda')
    age = torch.arange(64, device='cuda') + 970
    actions = torch.rand(64, 3, device='cuda').clamp(.01, .99)
    scores, _, parameters = beta_geometry(torch.randn(64, 6, device='cuda'), actions)
    lengths = band_lengths(age)
    prediction = critic.response_profile(observation, age, scores, actions, parameters, lengths)
    assert prediction.shape == (64, 6, 4)
    assert (prediction[lengths == 0] == 0).all()
    prediction.square().mean().backward()
    assert critic.power_logits.grad is not None
    assert torch.isfinite(critic.power_logits.grad).all()
    assert critic.power_logits.grad.square().sum() > 0
    assert all(p.grad is None for p in critic.baseline_trunk.parameters())
    assert all(p.grad is None for p in critic.baseline_head.parameters())


def test_nonlinear_line_search_keeps_interior_return_maximum():
    actor = torch.nn.Linear(1, 1, bias=False).cuda()
    with torch.no_grad():
        actor.weight.zero_()
    solver = ActorNaturalGradient(actor, budget=.5, line_search_steps=20)
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    direction = torch.ones(1, device='cuda')

    def measure():
        displacement = actor.weight.squeeze()
        return 2 * displacement - 4 * displacement.square(), .5 * displacement.square()

    gain, kl, scale, evaluations, _ = solver.search(before, direction, 1., measure)
    assert abs(scale - .25) < .003
    assert gain > .2499 and 0 < kl < solver.budget
    assert evaluations <= solver.line_search_steps
    torch.testing.assert_close(actor.weight, torch.full_like(actor.weight, scale))


def test_compiled_profile_fitting_and_two_exact_expectation_actor_updates():
    environment = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (5,)),
                                  single_action_space=gym.spaces.Box(-1., 1., (3,)))
    actor = Agent(environment).cuda().actor
    critic = ResponseCritic(5, 3, width=32).cuda()
    baseline_parameters = tuple(critic.baseline_trunk.parameters()) + tuple(critic.baseline_head.parameters())
    response_parameters = tuple(critic.response_trunk.parameters()) + tuple(critic.response_head.parameters()) + (critic.power_logits,)
    baseline_optimizer = torch.optim.Adam(baseline_parameters, lr=.001, fused=True)
    response_optimizer = torch.optim.Adam(response_parameters, lr=.001, fused=True)
    observations = torch.randn(128, 5, device='cuda')
    ages = torch.arange(128, device='cuda')
    lengths = band_lengths(ages).float()
    actions = torch.rand(128, 3, device='cuda').clamp(.01, .99)
    target_rates = .1 * torch.randn(128, 6, 4, device='cuda')
    weights = torch.ones(128, device='cuda')

    def baseline_loss(observation, age, target):
        return (critic.baseline_rates(observation, age) - target).square().mean()

    def response_loss(observation, age, scores, action, parameters, length, residual):
        prediction = critic.response_profile(observation, age, scores, action, parameters, length)
        return ((prediction - residual) / length.clamp_min(1)[..., None]).square().mean()

    def actor_measure(observation, reference, factor, coefficients, powers, mean, std):
        return integrated_gain_kl(actor(observation), reference, factor, coefficients, powers,
                                  critic.kernel_indices, mean, std, weights)

    policy = torch.compile(actor.forward, fullgraph=True, mode='reduce-overhead')
    fit_baseline = torch.compile(baseline_loss, fullgraph=True, mode='reduce-overhead')
    fit_response = torch.compile(response_loss, fullgraph=True, mode='reduce-overhead')
    measure_fn = torch.compile(actor_measure, fullgraph=True, mode='reduce-overhead')
    solver = ActorNaturalGradient(actor, budget=.01, cg_iterations=20, line_search_steps=16, compile=True)
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            logits = policy(observations).clone()
            scores, factor, parameters = beta_geometry(logits, actions)
        for fitting_step in range(2):
            torch.compiler.cudagraph_mark_step_begin()
            baseline_optimizer.zero_grad(set_to_none=True)
            loss = fit_baseline(observations, ages, target_rates)
            loss.backward()
            baseline_optimizer.step()
            baseline_optimizer.zero_grad(set_to_none=True)
            del loss
            with torch.no_grad():
                residual = (target_rates - critic.baseline_rates(observations, ages)) * lengths[..., None]
            torch.compiler.cudagraph_mark_step_begin()
            response_optimizer.zero_grad(set_to_none=True)
            loss = fit_response(observations, ages, scores, actions, parameters, lengths, residual)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in response_parameters)
            if fitting_step == 1:
                assert critic.power_logits.grad.square().sum() > 0
            assert all(p.grad is None for p in baseline_parameters)
            assert all(p.grad is None for p in actor.parameters())
            response_optimizer.step()
            del loss
        with torch.no_grad():
            coefficients = (critic.coefficients(observations, ages) * lengths[..., None, None]).sum((1, 2)).clone()
            powers = critic.powers().clone()
            mean, std = (value.clone() for value in kernel_statistics(parameters, powers, critic.kernel_indices))
            credit = response_credit(parameters, factor, coefficients, powers, critic.kernel_indices, std).clone()
            reference = beta_kl_reference(parameters)

        def measure():
            gain, kl = measure_fn(observations, reference, factor, coefficients, powers, mean, std)
            return gain.clone(), kl.clone()

        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            old_gain, old_kl = measure()
        # Collection geometry and the fused measurement follow distinct FP32
        # arithmetic paths. Gain is first-order sensitive to their roundoff;
        # unchanged-policy KL is second order and retains its tighter check.
        assert abs(float(old_gain.detach())) < 1e-6
        assert abs(float(old_kl.detach())) < 1e-10
        result = solver.step(observations, logits, factor, credit, weights, measure)
        assert result['policy/accepted_gain'] > 0
        assert 0 < result['policy/exact_joint_kl'] <= solver.budget


def test_compiled_reward_profiles_match_eager_boundaries_and_keep_staged_outputs():
    steps, environments, horizon = 18, 3, 8
    edges = (0, 1, 3, 8)
    positions = torch.arange(steps, device='cuda')[:, None]
    phases = torch.tensor([4, 0, 6], device='cuda')[None]
    true_ages = (positions + phases) % horizon
    ends = true_ages == horizon - 1
    recorded_ages = torch.where((phases > 0) & (positions < horizon - phases), -1, true_ages)
    rewards = torch.randn(steps, environments, device='cuda')
    actions = torch.rand(steps, environments, 2, device='cuda') * 2 - 1
    eager = reward_profiles(rewards, actions, ends, recorded_ages, .1, edges=edges, horizon=horizon)
    compiled = torch.compile(reward_profiles, fullgraph=True, mode='reduce-overhead')
    torch.compiler.cudagraph_mark_step_begin()
    actual = tuple(value.clone() for value in compiled(rewards, actions, ends, recorded_ages, .1,
                                                       edges=edges, horizon=horizon))
    for result, expected in zip(actual, eager):
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
    expected_valid = positions <= torch.tensor([11, 15, 17], device='cuda')[None]
    torch.testing.assert_close(actual[1].reshape_as(expected_valid), expected_valid)
    torch.testing.assert_close(actual[2].reshape_as(true_ages), true_ages)
    assert (actual[0][~actual[1]] == 0).all()
    assert (actual[3][~actual[1]] == 0).all()
    torch.testing.assert_close(actual[3].sum(-1), ((horizon - true_ages) * expected_valid).flatten().float())
    torch.compiler.cudagraph_mark_step_begin()
    changed = tuple(value.clone() for value in compiled(rewards + 1, actions, ends, recorded_ages, .1,
                                                        edges=edges, horizon=horizon))
    # A unit progress-reward increment adds exactly the observed band length;
    # cloned earlier outputs must survive the second compiled invocation.
    torch.testing.assert_close(changed[0][..., 0] - actual[0][..., 0], actual[3], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(changed[0][..., 1:], actual[0][..., 1:], rtol=0, atol=0)


def test_production_shape_compiled_reward_profiles_and_retained_outputs():
    with torch.no_grad():
        # Inductor's scan/index lowering depends on dimensions. This exact training
        # shape exercises both terminal scans and the 7-component prefix sum.
        steps, environments, action_dim, horizon = 2048, 16, 6, 1000
        positions = torch.arange(steps, device='cuda')[:, None]
        phases = (torch.arange(environments, device='cuda') * horizon // environments)[None]
        true_ages = (positions + phases) % horizon
        ends = true_ages == horizon - 1
        ages = torch.where((phases > 0) & (positions < horizon - phases), -1, true_ages)
        rewards = torch.randn(steps, environments, device='cuda')
        actions = 2 * torch.rand(steps, environments, action_dim, device='cuda') - 1
        expected = reward_profiles(rewards, actions, ends, ages, .1)
        # Start from a fresh frame cache: earlier small-shape tests must not
        # silently switch this production trace to automatic dynamic shapes.
        torch._dynamo.reset()
        compiled = torch.compile(reward_profiles, fullgraph=True, dynamic=False, mode='reduce-overhead')
        torch.compiler.cudagraph_mark_step_begin()
        actual = tuple(value.clone() for value in compiled(rewards, actions, ends, ages, .1))
        assert actual[0].shape == (32768, 6, 7)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-5)
        last_ends = torch.where(ends, positions, -1).amax(0)
        valid = positions <= last_ends[None]
        torch.testing.assert_close(actual[1].reshape_as(valid), valid)
        torch.testing.assert_close(actual[2].reshape_as(true_ages), true_ages)
        assert (actual[0][~actual[1]] == 0).all()
        assert (actual[3][~actual[1]] == 0).all()
        torch.testing.assert_close(actual[3].sum(-1), ((horizon - true_ages) * valid).flatten().float())
        torch.compiler.cudagraph_mark_step_begin()
        changed = tuple(value.clone() for value in compiled(rewards + 1, actions, ends, ages, .1))
        torch.testing.assert_close(changed[0][..., 0] - actual[0][..., 0], actual[3], rtol=1e-6, atol=1e-4)
        torch.testing.assert_close(changed[0][..., 1:], actual[0][..., 1:], rtol=0, atol=0)
        # Verify the first invocation remains intact after reusing compiled storage.
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-5)

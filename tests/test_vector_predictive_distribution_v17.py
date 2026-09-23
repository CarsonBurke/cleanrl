"""CUDA contracts for a joint vector predictive distribution and actor correction."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from cleanrl.ppo_continuous_action_vector_predictive_distribution_v17 import (
    Agent, ActorNaturalGradient, PredictiveCritic, beta_geometry, beta_kl_reference,
    beta_log_prob, conjugate_gradient, corrected_credit, corrected_gain_kl, corrected_weights,
    distributional_td_target, lift_returns, td_segments,
)
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.host_graph import make_host_mirror


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), 'Run through mlq on CUDA'
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)


@pytest.mark.parametrize('horizon', [13, 1000])
def test_complex_td_composition_matches_lifted_vector_return_and_terminal_atom(horizon):
    channels, count = 4, 7
    frequencies = torch.randn(count, channels, device='cuda', dtype=torch.float64)
    observed = 20 * torch.randn(16, channels, device='cuda', dtype=torch.float64)
    following_returns = 40 * torch.randn_like(observed)
    ends = torch.zeros(16, dtype=torch.bool, device='cuda')
    ends[::3] = True
    following = lift_returns(following_returns, frequencies, horizon=horizon).requires_grad_()
    target = distributional_td_target(observed, following, ends, frequencies, horizon=horizon)
    expected = lift_returns(observed + following_returns * (~ends)[:, None], frequencies, horizon=horizon)
    assert target.shape == (16, channels + 2 * count)
    assert not target.requires_grad
    torch.testing.assert_close(target, expected, rtol=1e-10, atol=1e-11)
    manual_means = (observed + following_returns * (~ends)[:, None]) / (horizon * channels ** .5)
    torch.testing.assert_close(target[:, :channels], manual_means, rtol=1e-12, atol=1e-12)
    atom = lift_returns(torch.zeros_like(observed), frequencies, horizon=horizon)
    assert (atom[:, :channels] == 0).all() and (atom[:, channels + count:] == 0).all()
    torch.testing.assert_close(atom[:, channels:channels + count], torch.full_like(atom[:, channels:channels + count], count ** -.5))
    frozen = target.clone()
    with torch.no_grad():
        following.add_(100)
    torch.testing.assert_close(target, frozen, rtol=0, atol=0)


def test_successive_observed_segments_compose_in_same_joint_distribution():
    frequencies = torch.randn(9, 3, device='cuda', dtype=torch.float64)
    first = 10 * torch.randn(12, 3, device='cuda', dtype=torch.float64)
    second = 20 * torch.randn_like(first)
    rest = 30 * torch.randn_like(first)
    continuing = torch.zeros(12, dtype=torch.bool, device='cuda')
    future = lift_returns(rest, frequencies, horizon=1000)
    after_second = distributional_td_target(second, future, continuing, frequencies, horizon=1000)
    after_both = distributional_td_target(first, after_second, continuing, frequencies, horizon=1000)
    torch.testing.assert_close(after_both, lift_returns(first + second + rest, frequencies, horizon=1000), rtol=1e-10, atol=1e-11)
    # Fourier coordinates mix physical components before sine/cosine; this is
    # a joint return transform rather than independent marginal distributions.
    phases = (first + second + rest) @ frequencies.T / 1000
    torch.testing.assert_close(after_both[:, 3:12], phases.cos() / 3, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(after_both[:, 12:], phases.sin() / 3, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize('samples', [2, 5])
def test_corrected_vector_weights_use_leave_one_out_model_centering(samples):
    model = torch.randn(16, samples, 4, device='cuda', dtype=torch.float64)
    observed_model = torch.randn(16, 4, device='cuda', dtype=torch.float64)
    observed_target = torch.randn_like(observed_model)
    actual = corrected_weights(model, observed_model, observed_target)
    assert actual.shape == (16, samples + 1, 4)
    expected_model = []
    for sample in range(samples):
        others = torch.cat((model[:, :sample], model[:, sample + 1:]), 1).mean(1)
        expected_model.append((model[:, sample] - others) / samples)
    torch.testing.assert_close(actual[:, :samples], torch.stack(expected_model, 1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual[:, -1], observed_target - observed_model)
    torch.testing.assert_close(actual[:, :samples].sum(1), torch.zeros_like(observed_model), rtol=0, atol=1e-12)
    constant = observed_model[:, None].expand(-1, samples, -1)
    centered = corrected_weights(constant, observed_model, observed_target)
    torch.testing.assert_close(centered[:, :samples], torch.zeros_like(centered[:, :samples]), rtol=0, atol=1e-12)


def test_perfect_conditional_mean_keeps_stochastic_observed_return_residual():
    conditional_mean = torch.randn(16, 4, device='cuda')
    realized_noise = torch.randn_like(conditional_mean)
    observed_return = conditional_mean + realized_noise
    alternative_means = torch.randn(16, 4, 4, device='cuda')
    result = corrected_weights(alternative_means, conditional_mean, observed_return)
    torch.testing.assert_close(result[:, -1], observed_return.double() - conditional_mean.double())
    assert result[:, -1].square().sum() > 0


@pytest.mark.parametrize('logit_shift', [-1.5, 0., 2.])
def test_corrected_credit_matches_autograd_at_three_distinct_beta_policies(logit_shift):
    batch, samples, action_dim, channels = 24, 5, 3, 4
    logits = (torch.randn(batch, 2 * action_dim, device='cuda', dtype=torch.float64) + logit_shift).requires_grad_()
    actions = .01 + .98 * torch.rand(batch, samples, action_dim, device='cuda', dtype=torch.float64)
    alpha, beta = (F.softplus(logits.detach()) + 1).chunk(2, -1)
    parameters = torch.stack((alpha, beta), -1)
    vector_weights = torch.randn(batch, samples, channels, device='cuda', dtype=torch.float64)
    utility_weights = vector_weights.sum(-1)
    weights = torch.rand(batch, device='cuda', dtype=torch.float64) + .1
    old_log_prob = beta_log_prob(parameters, actions).detach()
    independent_log_prob = torch.distributions.Beta(alpha[:, None], beta[:, None]).log_prob(actions).sum(-1)
    torch.testing.assert_close(old_log_prob, independent_log_prob, rtol=1e-11, atol=1e-12)
    gain, kl = corrected_gain_kl(logits, beta_kl_reference(parameters), actions, old_log_prob, utility_weights, weights)
    assert abs(float(gain.detach())) < 1e-10 and abs(float(kl.detach())) < 1e-10
    actual, = torch.autograd.grad(gain, logits)
    credit = corrected_credit(parameters, actions, vector_weights)
    expected = torch.cat((credit[..., 0], credit[..., 1]), -1) * logits.detach().sigmoid()
    expected = expected * (weights / weights.sum())[:, None]
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    candidate = logits.detach() + .1 * torch.randn_like(logits)
    gain, actual_kl = corrected_gain_kl(candidate, beta_kl_reference(parameters), actions, old_log_prob, utility_weights, weights)
    new_alpha, new_beta = (F.softplus(candidate) + 1).chunk(2, -1)
    new_dist = torch.distributions.Beta(new_alpha[:, None], new_beta[:, None])
    ratio = (new_dist.log_prob(actions).sum(-1) - independent_log_prob).exp()
    expected_gain = (((ratio - 1) * utility_weights).sum(-1) * weights).sum() / weights.sum()
    expected_kl = torch.distributions.kl_divergence(torch.distributions.Beta(alpha, beta),
                                                   torch.distributions.Beta(new_alpha, new_beta)).sum(-1)
    torch.testing.assert_close(gain, expected_gain, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(actual_kl, (expected_kl * weights).sum() / weights.sum(), rtol=1e-10, atol=1e-11)


def test_preconditioned_cg_solves_spd_with_small_actual_residual():
    width = 12
    scaling = torch.logspace(-2, 2, width, device='cuda', dtype=torch.float64).sqrt()
    low_rank = torch.randn(width, 3, device='cuda', dtype=torch.float64)
    central = torch.eye(width, device='cuda', dtype=torch.float64) + .1 * low_rank @ low_rank.T
    matrix = scaling[:, None] * central * scaling[None]
    rhs = torch.randn(width, device='cuda', dtype=torch.float64)
    solution, residual = conjugate_gradient(lambda vector: matrix @ vector, rhs, 30,
                                           relative_tolerance=1e-10, diagonal=matrix.diag())
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs), rtol=1e-8, atol=1e-9)
    recomputed = (rhs - matrix @ solution).norm() / rhs.norm()
    torch.testing.assert_close(residual, recomputed, rtol=1e-7, atol=1e-13)
    assert recomputed < 1e-8
    zero, zero_residual = conjugate_gradient(lambda vector: matrix @ vector, torch.zeros_like(rhs), 30,
                                             diagonal=matrix.diag())
    assert (zero == 0).all() and zero_residual == 0


def test_joint_prediction_raw_mean_readout_terminal_atom_and_complex_disk_bound():
    critic = PredictiveCritic(5, 3, width=32).cuda()
    torch.nn.init.normal_(critic.head.weight, std=.02)
    obs = torch.randn(16, 5, device='cuda')
    ages = torch.tensor([0, 1, 50, 100, 200, 300, 400, 500, 700, 800, 900, 990, 998, 999, 1000, 1001], device='cuda')
    actions = torch.rand(16, 3, device='cuda').clamp(.01, .99)
    channels = 4
    count = critic.frequencies.shape[0]
    prediction = critic.predict(obs, ages, actions)
    assert prediction.shape == (16, channels + 2 * count)
    assert critic.frequencies.shape[1] == channels and not critic.frequencies.requires_grad
    assert ((critic.frequencies != 0).sum(-1) > 1).any()
    components = critic.components(obs, ages, actions)
    torch.testing.assert_close(components, prediction[:, :channels] * (1000 * channels ** .5))
    radius_squared = prediction[:, channels:channels + count].square() + prediction[:, channels + count:].square()
    # Necessary coordinate-wise complex-disk condition, not a claim that a
    # finite moment vector certifies an entire probability distribution.
    assert (radius_squared <= 1 / count + 1e-6).all()
    terminal = ages >= 1000
    atom = lift_returns(torch.zeros_like(components[terminal]), critic.frequencies)
    torch.testing.assert_close(prediction[terminal], atom, rtol=1e-6, atol=1e-7)
    assert (components[terminal] == 0).all()


def test_every_joint_prediction_group_uses_shared_state_action_trunk():
    critic = PredictiveCritic(5, 3, width=32).cuda()
    torch.nn.init.normal_(critic.head.weight, std=.02)
    obs = torch.randn(16, 5, device='cuda', requires_grad=True)
    action = (.1 + .8 * torch.rand(16, 3, device='cuda')).requires_grad_()
    age = torch.arange(16, device='cuda') * 50
    prediction = critic.predict(obs, age, action)
    count = critic.frequencies.shape[0]
    shared = tuple(critic.trunk.parameters())
    for group in (prediction[:, :4], prediction[:, 4:4 + count], prediction[:, 4 + count:]):
        gradients = torch.autograd.grad(group.square().mean(), shared + (obs, action), retain_graph=True)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.square().sum() for gradient in gradients[:-2]) > 0
        assert gradients[-2].square().sum() > 0 and gradients[-1].square().sum() > 0
    original_action_gradient, = torch.autograd.grad(prediction[:, 0].sum(), action)
    changed_prediction = critic.predict(obs.detach() + .5, age, action)
    changed_action_gradient, = torch.autograd.grad(changed_prediction[:, 0].sum(), action)
    assert not torch.allclose(original_action_gradient, changed_action_gradient)


def test_observed_segment_composition_uses_actual_terminal_endpoint():
    steps, environments, channels = 19, 2, 4
    components = torch.randn(steps, environments, channels, device='cuda', dtype=torch.float64)
    ends = torch.zeros(steps, environments, dtype=torch.bool, device='cuda')
    ends[7, 0] = True
    frequencies = torch.randn(7, channels, device='cuda', dtype=torch.float64)
    observed, endpoint, consumed, terminal = td_segments(components, ends, td_steps=32)
    endpoint_returns = torch.randn(steps * environments, channels, device='cuda', dtype=torch.float64)
    future = lift_returns(endpoint_returns[endpoint], frequencies)
    target = distributional_td_target(observed, future, terminal, frequencies)
    exact = observed + endpoint_returns[endpoint] * (~terminal)[:, None]
    torch.testing.assert_close(target, lift_returns(exact, frequencies), rtol=1e-10, atol=1e-11)
    assert int(consumed[0]) == 8 and bool(terminal[0])
    assert int(consumed[1]) == 19 and not bool(terminal[1])
    torch.testing.assert_close(target[0], lift_returns(components[:8, 0].sum(0, keepdim=True), frequencies)[0], rtol=1e-10, atol=1e-11)


def test_compiled_two_joint_fit_and_corrected_actor_cycles_preserve_prefit_snapshot():
    environment = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (5,)),
                                  single_action_space=gym.spaces.Box(-1., 1., (3,)))
    actor = Agent(environment).cuda().actor
    critic = PredictiveCritic(5, 3, width=32).cuda()
    optimizer = torch.optim.Adam(critic.parameters(), lr=.001, fused=True)
    steps, environments, samples = 17, 2, 4
    batch, channels = steps * environments, 4
    observations = torch.randn(batch, 5, device='cuda')
    next_observations = observations + .1 * torch.randn_like(observations)
    ages = (torch.arange(steps, device='cuda')[:, None] + torch.tensor([975, 983], device='cuda')[None]).flatten()
    ends = ages.reshape(steps, environments) == 999
    observed_actions = torch.rand(batch, 3, device='cuda').clamp(.01, .99)
    components = torch.cat((.1 * torch.randn(batch, 1, device='cuda'), -.1 * (2 * observed_actions - 1).square()), -1).reshape(steps, environments, channels)
    state_weights = torch.ones(batch, device='cuda')

    def fitting_loss(obs, age, action, target):
        return (critic.predict(obs, age, action) - target).square().mean()

    def actor_measure(obs, reference, actions, old_log_prob, utility_weights):
        return corrected_gain_kl(actor(obs), reference, actions, old_log_prob, utility_weights, state_weights)

    policy = torch.compile(actor.forward, fullgraph=True, mode='reduce-overhead')
    predict = torch.compile(critic.predict, fullgraph=True, mode='reduce-overhead')
    means = torch.compile(critic.components, fullgraph=True, mode='reduce-overhead')
    target_fn = torch.compile(distributional_td_target, fullgraph=True, mode='reduce-overhead')
    weights_fn = torch.compile(corrected_weights, fullgraph=True, mode='reduce-overhead')
    credit_fn = torch.compile(corrected_credit, fullgraph=True, mode='reduce-overhead')
    fit = torch.compile(fitting_loss, fullgraph=True, mode='reduce-overhead')
    measure_fn = torch.compile(actor_measure, fullgraph=True, mode='reduce-overhead')
    solver = ActorNaturalGradient(actor, budget=.01, cg_iterations=30, line_search_steps=16, compile=True)
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            logits = policy(observations).clone()
            _, collection_factor, parameters = beta_geometry(logits, observed_actions)
            observed, endpoint, consumed, terminal = td_segments(components, ends, td_steps=8)
            endpoint_obs = next_observations[endpoint]
            next_logits = policy(endpoint_obs).clone()
            alpha, beta = (F.softplus(next_logits) + 1).chunk(2, -1)
            next_action = torch.distributions.Beta(alpha, beta).sample()
            future = predict(endpoint_obs, ages + consumed, next_action).clone()
            target = target_fn(observed, future, terminal, critic.frequencies).clone()
            raw_target = target[:, :channels] * (1000 * channels ** .5)
            alpha, beta = parameters.unbind(-1)
            alternatives = torch.distributions.Beta(alpha, beta).sample((samples,)).transpose(0, 1).contiguous()
            actions = torch.cat((alternatives, observed_actions[:, None]), 1)
            repeated_obs = observations[:, None].expand(-1, samples + 1, -1).reshape(-1, observations.shape[-1])
            repeated_age = ages[:, None].expand(-1, samples + 1).reshape(-1)
            modeled = means(repeated_obs, repeated_age, actions.flatten(0, 1)).clone().reshape(batch, samples + 1, channels)
            vector_weights = weights_fn(modeled[:, :samples], modeled[:, -1], raw_target).clone()
            old_log_prob = beta_log_prob(parameters, actions).clone()
            credit = credit_fn(parameters, actions, vector_weights).clone()
            utility_weights = vector_weights.sum(-1)
            reference = beta_kl_reference(parameters)
            frozen_weights, frozen_target = vector_weights.clone(), target.clone()
        for _ in range(2):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            loss = fit(observations, ages, observed_actions, target)
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in critic.parameters())
            assert all(parameter.grad is None for parameter in actor.parameters())
            optimizer.step()
            del loss
        torch.testing.assert_close(vector_weights, frozen_weights, rtol=0, atol=0)
        torch.testing.assert_close(target, frozen_target, rtol=0, atol=0)

        def measure():
            gain, kl = measure_fn(observations, reference, actions, old_log_prob, utility_weights)
            return gain.clone(), kl.clone()

        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            old_gain, old_kl = measure()
        assert abs(float(old_gain)) < 1e-6 and abs(float(old_kl)) < 1e-10
        result = solver.step(observations, logits, collection_factor, credit, state_weights, measure)
        assert result['policy/accepted_gain'] > 0
        assert 0 < result['policy/exact_joint_kl'] <= solver.budget


def test_production_shape_static_segment_and_distributional_target_compilation():
    with torch.no_grad():
        steps, environments, channels = 2048, 16, 7
        positions = torch.arange(steps, device='cuda')[:, None]
        phases = (torch.arange(environments, device='cuda') * 1000 // environments)[None]
        ages = (positions + phases) % 1000
        ends = ages == 999
        components = torch.randn(steps, environments, channels, device='cuda')
        frequencies = PredictiveCritic(17, 6, width=32).cuda().frequencies
        expected_segments = td_segments(components, ends, td_steps=32)
        # Fresh production trace: small earlier tests must not silently select
        # a dynamic scan layout that the first real training call does not use.
        torch._dynamo.reset()
        segment_fn = torch.compile(td_segments, fullgraph=True, dynamic=False, mode='reduce-overhead')
        target_fn = torch.compile(distributional_td_target, fullgraph=True, dynamic=False, mode='reduce-overhead')
        torch.compiler.cudagraph_mark_step_begin()
        actual = tuple(value.clone() for value in segment_fn(components, ends, 32))
        observed, endpoint, consumed, terminal = actual
        assert observed.shape == (32768, 7)
        for result, reference in zip(actual, expected_segments):
            torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-5)
        expected_count = torch.minimum(torch.full_like(ages, 32), torch.minimum(steps - positions, 1000 - ages))
        torch.testing.assert_close(consumed, expected_count.flatten())
        expected_endpoint = ((positions + expected_count - 1) * environments + torch.arange(environments, device='cuda')[None]).flatten()
        torch.testing.assert_close(endpoint, expected_endpoint)
        torch.testing.assert_close(terminal, ends.flatten()[endpoint])
        remaining_return = 20 * torch.randn_like(observed)
        future = lift_returns(remaining_return, frequencies)
        eager_target = distributional_td_target(observed, future, terminal, frequencies)
        target = target_fn(observed, future, terminal, frequencies).clone()
        assert target.shape == (32768, channels + 2 * frequencies.shape[0])
        torch.testing.assert_close(target, eager_target, rtol=1e-6, atol=1e-6)
        frozen_target = target.clone()
        torch.compiler.cudagraph_mark_step_begin()
        changed = tuple(value.clone() for value in segment_fn(components + 1, ends, 32))
        torch.testing.assert_close(changed[0] - observed, consumed[:, None].expand_as(observed).float(), rtol=1e-6, atol=1e-5)
        changed_target = target_fn(changed[0], future, changed[3], frequencies).clone()
        expected_changed = lift_returns(changed[0] + remaining_return * (~changed[3])[:, None], frequencies)
        torch.testing.assert_close(changed_target, expected_changed, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(target, frozen_target, rtol=0, atol=0)
        for result, reference in zip(actual, expected_segments):
            torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-5)


def test_rollout_host_mirror_matches_cuda_actor_before_and_after_refresh():
    # Exercise the public rollout boundary without inspecting its implementation.
    # Use the actual HalfCheetah actor/input dimensions and vector-environment
    # count, since lowering can depend on both shape and trunk operations.
    environments, state_dim, action_dim = 16, 17, 6
    environment = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (state_dim,)),
                                  single_action_space=gym.spaces.Box(-1., 1., (action_dim,)))
    actor = Agent(environment).cuda().actor
    observations = np.random.default_rng(1).normal(size=(environments, state_dim)).astype(np.float32)
    cuda_observations = torch.as_tensor(observations, device='cuda')
    mirror = make_host_mirror(actor, environments)
    # Host mirrors return persistent buffers; copy each result before reuse.
    original_host = np.array(mirror(observations), copy=True)
    with torch.no_grad():
        original_cuda = actor(cuda_observations).detach().cpu().numpy()
    assert original_host.shape == (environments, 2 * action_dim)
    assert np.isfinite(original_host).all()
    np.testing.assert_allclose(original_host, original_cuda, rtol=2e-5, atol=3e-6)
    with torch.no_grad():
        actor[-1].weight.mul_(1.125)
        actor[-1].bias.add_(torch.linspace(-.08, .08, 2 * action_dim, device='cuda'))
        updated_cuda = actor(cuda_observations).detach().cpu().numpy()
    mirror.refresh()
    updated_host = np.array(mirror(observations), copy=True)
    assert np.isfinite(updated_host).all()
    assert not np.allclose(updated_host, original_host, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(updated_host, updated_cuda, rtol=2e-5, atol=3e-6)

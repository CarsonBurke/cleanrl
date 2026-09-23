"""CUDA contracts for observed vector return segments and frozen continuation.

Execute through mlq. These validate target semantics and compiled integration.
"""
import pytest
import torch

from cleanrl.ppo_continuous_action_vector_segment_td_v15 import (
    ResponseCritic, basis_geometry, query_horizons, segment_td_targets, td_segments,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), 'Run through mlq on CUDA'
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)


@pytest.mark.parametrize('steps,segment_length', [(45, 32), (137, 32), (19, 64)])
def test_observed_segments_match_explicit_sums_and_common_real_endpoints(steps, segment_length):
    environments, channels = 2, 4
    components = torch.randn(steps, environments, channels, device='cuda', dtype=torch.float64)
    boundaries = (tuple(value for value in (5, 20, 44, 98, 136) if value < steps),
                  tuple(value for value in (0, 34, 100) if value < steps))
    ends = torch.zeros(steps, environments, dtype=torch.bool, device='cuda')
    for environment, values in enumerate(boundaries):
        ends[list(values), environment] = True
    queries = torch.tensor([0, 1, 7, 16, 32, 64], device='cuda').expand(steps * environments, -1).clone()
    observed, endpoint, consumed, endpoint_ends = td_segments(components, ends, queries, td_steps=segment_length)
    assert observed.shape == (steps * environments, 6, channels)
    expected = torch.zeros_like(observed)
    for time in range(steps):
        for environment in range(environments):
            row = time * environments + environment
            next_end = next((value for value in boundaries[environment] if value >= time), steps - 1)
            count = min(segment_length, steps - time, next_end - time + 1)
            assert int(consumed[row]) == count
            assert int(endpoint[row]) == (time + count - 1) * environments + environment
            assert bool(endpoint_ends[row]) == ((time + count - 1) in boundaries[environment])
            for query, horizon in enumerate((0, 1, 7, 16, 32, 64)):
                expected[row, query] = components[time:time + min(horizon, count), environment].sum(0)
    torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-10)
    assert (observed[:, 0] == 0).all()


def test_actual_terminal_suppresses_bootstrap_but_rollout_cut_does_not():
    components = torch.tensor([[[1., -.1], [2., -.2]],
                               [[3., -.3], [4., -.4]],
                               [[5., -.5], [6., -.6]]], device='cuda')
    ends = torch.zeros(3, 2, dtype=torch.bool, device='cuda')
    ends[1, 0] = True
    queries = torch.tensor([0, 1, 2, 5], device='cuda').expand(6, -1).clone()
    observed, endpoint, consumed, terminal = td_segments(components, ends, queries, td_steps=32)
    remaining = (queries - consumed[:, None]).clamp_min(0)
    following = torch.full_like(observed, 10., requires_grad=True)
    actual = segment_td_targets(observed, following, remaining, terminal)
    continuation = (remaining > 0) & ~terminal[:, None]
    expected = observed + 10 * continuation[..., None]
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    # First environment actually ends after two transitions; the other merely
    # reaches the end of the available rollout and must bootstrap there.
    assert int(consumed[0]) == 2 and bool(terminal[0])
    assert int(consumed[1]) == 3 and not bool(terminal[1])
    torch.testing.assert_close(actual[0, -1], components[:2, 0].sum(0))
    torch.testing.assert_close(actual[1, -1], components[:, 1].sum(0) + 10)
    assert (actual[:, 0] == 0).all()
    torch.testing.assert_close(actual[:, 1], components.flatten(0, 1))
    frozen = actual.clone()
    with torch.no_grad():
        following.add_(100)
    torch.testing.assert_close(actual, frozen, rtol=0, atol=0)


def test_one_transition_segments_recover_vector_one_step_targets():
    steps, environments, channels = 17, 3, 4
    components = torch.randn(steps, environments, channels, device='cuda')
    ends = torch.rand(steps, environments, device='cuda') < .2
    queries = torch.tensor([0, 1, 2, 9, 32, 100], device='cuda').expand(steps * environments, -1)
    observed, endpoint, consumed, terminal = td_segments(components, ends, queries, td_steps=1)
    torch.testing.assert_close(endpoint, torch.arange(steps * environments, device='cuda'))
    torch.testing.assert_close(consumed, torch.ones_like(consumed))
    torch.testing.assert_close(terminal, ends.flatten())
    torch.testing.assert_close(observed, components.flatten(0, 1)[:, None] * (queries > 0)[..., None])
    following = torch.randn_like(observed, requires_grad=True)
    actual = segment_td_targets(observed, following, (queries - 1).clamp_min(0), terminal)
    expected = components.flatten(0, 1)[:, None] * (queries > 0)[..., None]
    expected = expected + following.detach() * ((queries > 1) & ~ends.flatten()[:, None])[..., None]
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_compiled_joint_critic_fit_uses_real_endpoint_and_frozen_segment_target():
    steps, environments, state_dim, action_dim = 17, 2, 5, 3
    batch = steps * environments
    critic = ResponseCritic(state_dim, action_dim, width=32).cuda()
    optimizer = torch.optim.Adam(critic.parameters(), lr=.001, fused=True)
    observations = torch.randn(batch, state_dim, device='cuda')
    physical_next_observations = observations + .1 * torch.randn_like(observations)
    ages = (torch.arange(steps, device='cuda')[:, None] + torch.tensor([975, 983], device='cuda')[None]).flatten()
    queries = query_horizons(ages)
    actions = torch.rand(batch, action_dim, device='cuda').clamp(.01, .99)
    scores, _, basis = basis_geometry(actions)
    components = torch.cat((torch.randn(batch, 1, device='cuda'), -.1 * (2 * actions - 1).square()), -1).reshape(steps, environments, -1)
    ends = (ages.reshape(steps, environments) == 999)
    current_parameters = 1.2 + torch.rand(batch, action_dim, 2, device='cuda')

    def loss_function(obs, age, horizon, score, action, basis_parameters, target):
        prediction = critic.baseline(obs, age, horizon) + critic.response_profile(obs, age, horizon, score, action, basis_parameters)
        return (prediction - target).square().mean()

    segment_fn = torch.compile(td_segments, fullgraph=True, mode='reduce-overhead')
    expected_fn = torch.compile(critic.expected_profile, fullgraph=True, mode='reduce-overhead')
    target_fn = torch.compile(segment_td_targets, fullgraph=True, mode='reduce-overhead')
    fit = torch.compile(loss_function, fullgraph=True, mode='reduce-overhead')
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            observed, endpoint, consumed, terminal = (value.clone() for value in segment_fn(components, ends, queries, 8))
            remaining = (queries - consumed[:, None]).clamp_min(0)
            continuation = expected_fn(physical_next_observations[endpoint], ages + consumed,
                                       remaining, current_parameters[endpoint]).clone()
            target = target_fn(observed, continuation, remaining, terminal).clone()
            frozen = target.clone()
        for fitting_step in range(2):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            loss = fit(observations, ages, queries, scores, actions, basis, target)
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in critic.parameters())
            if fitting_step == 1:
                assert critic.power_head.weight.grad.square().sum() > 0
            optimizer.step()
            del loss
        torch.testing.assert_close(target, frozen, rtol=0, atol=0)


def test_production_shape_static_compiled_segments_and_target_output_lifetimes():
    with torch.no_grad():
        steps, environments, channels = 2048, 16, 7
        positions = torch.arange(steps, device='cuda')[:, None]
        phases = (torch.arange(environments, device='cuda') * 1000 // environments)[None]
        ages = (positions + phases) % 1000
        ends = ages == 999
        queries = query_horizons(ages.flatten())
        components = torch.randn(steps, environments, channels, device='cuda')
        expected = td_segments(components, ends, queries, td_steps=32)
        torch._dynamo.reset()
        compiled = torch.compile(td_segments, fullgraph=True, dynamic=False, mode='reduce-overhead')
        target_fn = torch.compile(segment_td_targets, fullgraph=True, dynamic=False, mode='reduce-overhead')
        torch.compiler.cudagraph_mark_step_begin()
        actual = tuple(value.clone() for value in compiled(components, ends, queries, 32))
        assert actual[0].shape == (32768, 6, 7)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-5)
        observed, endpoint, consumed, terminal = actual
        expected_count = torch.minimum(torch.full_like(ages, 32), torch.minimum(steps - positions, 1000 - ages))
        torch.testing.assert_close(consumed, expected_count.flatten())
        expected_endpoint = ((positions + expected_count - 1) * environments + torch.arange(environments, device='cuda')[None]).flatten()
        torch.testing.assert_close(endpoint, expected_endpoint)
        torch.testing.assert_close(terminal, ends.flatten()[endpoint])
        remaining = (queries - consumed[:, None]).clamp_min(0)
        following = torch.randn_like(observed)
        target = target_fn(observed, following, remaining, terminal).clone()
        torch.testing.assert_close(target, segment_td_targets(observed, following, remaining, terminal))
        frozen_target = target.clone()
        torch.compiler.cudagraph_mark_step_begin()
        changed = tuple(value.clone() for value in compiled(components + 1, ends, queries, 32))
        observed_count = torch.minimum(queries, consumed[:, None]).float()
        torch.testing.assert_close(changed[0] - observed, observed_count[..., None].expand_as(observed), rtol=1e-6, atol=1e-5)
        target_fn(changed[0], following, remaining, changed[3])
        torch.testing.assert_close(target, frozen_target, rtol=0, atol=0)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-5)

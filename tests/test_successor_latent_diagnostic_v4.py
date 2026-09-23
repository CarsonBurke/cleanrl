"""CUDA numerical contracts for the fresh successor-latent diagnosis.

Run through mlq. These test estimator identities and compiled differentiation;
they are not reduced training runs and provide no benchmark-score evidence.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl.ppo_continuous_action_successor_latent_diagnostic_v4 import (
    FrozenObsNorm, RewardDecoder, SuccessorCritic, TransitionRepresentation, draw_geometric, energy_score,
    make_logit_projector, pathwise_credit, projected_score, sample_parameter_directions,
    decoder_inputs, discounted_windows, counterfactual_advantages, counterfactual_contexts, projected_coordinate_scores,
    select_successor_target, successor_indices, summarize_gate, trajectory_spans,
    transition_features,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), 'These contracts must run on CUDA through mlq'
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)


def test_geometric_occupancy_includes_current_transition_and_discount_mass():
    horizon = draw_geometric((1000000,), .8, 'cuda')
    assert abs((horizon == 0).float().mean().item() - .2) < .002
    # For transition utility r_t=t, sum gamma**t*r_t = gamma/(1-gamma)**2.
    assert abs(horizon.double().mean().item() / .2 - 20) < .15
    assert abs(horizon.double().square().mean().item() - 36) < .5


def test_boundary_targets_distinguish_terminal_truncation_and_rollout_end():
    terms = np.array([[False, False], [True, False], [False, False], [False, False]])
    truncs = np.array([[False, False], [False, True], [False, False], [False, False]])
    spans = trajectory_spans(terms, truncs, 3)
    np.testing.assert_array_equal(spans, [[2, 2], [1, 1], [2, 2], [1, 1]])
    latent = torch.arange(8, device='cuda', dtype=torch.float64)[:, None]
    observed, selected = successor_indices(torch.tensor(spans.flatten(), device='cuda'),
                                           torch.full((8,), 100, device='cuda'), 2)
    # Caller supplies G(final_next_obs[selected], fresh_action, noise) per row.
    bootstrap = (100 + selected).double()[:, None]
    actual = select_successor_target(observed, selected, latent, torch.tensor(terms.flatten(), device='cuda'), bootstrap)
    torch.testing.assert_close(actual[:, 0], torch.tensor([0, 103, 0, 103, 106, 107, 106, 107], device='cuda', dtype=torch.float64))
    observed, selected = successor_indices(torch.tensor(spans.flatten(), device='cuda'), torch.zeros(8, device='cuda', dtype=torch.long), 2)
    actual = select_successor_target(observed, selected, latent, torch.tensor(terms.flatten(), device='cuda'), bootstrap)
    torch.testing.assert_close(actual, latent)


def test_latent_is_injective_and_uses_physical_terminal_observation():
    state = {'means': torch.zeros(2, 2), 'variances': torch.ones(2, 2), 'counts': torch.ones(2),
             'epsilon': 0., 'clip': 100.}
    normalizer = FrozenObsNorm(state, 2, (2,))
    normalized, physical = normalizer.normalize_step(np.array([[20., 20.], [30., 30.]]),
                                                     np.array([True, False]), np.array([False, True]),
                                                     {'final_observation': [np.array([1., 2.]), np.array([3., 4.])]})
    np.testing.assert_array_equal(physical, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(normalized, [[20, 20], [30, 30]])
    observations = torch.randn(32, 7, device='cuda')
    actions = torch.rand(32, 3, device='cuda')
    next_obs = torch.randn_like(observations)
    latent = transition_features(observations, actions, next_obs)
    torch.testing.assert_close(latent[:, :7], observations)
    torch.testing.assert_close((latent[:, 7:10] + 1) / 2, actions)
    torch.testing.assert_close(latent[:, 10:17] * 2**.5 + latent[:, :7], next_obs)
    torch.testing.assert_close(latent[:, 17], torch.ones(32, device='cuda'))


def test_energy_score_prefers_distribution_over_collapsed_mean():
    # Exact expectation over all eight combinations of independent two-point draws.
    draws = torch.tensor([[i, j, k] for i in (-1., 1.) for j in (-1., 1.) for k in (-1., 1.)], device='cuda')
    first, second, target = (draws[:, i:i + 1] for i in range(3))
    correct = energy_score(first, second, target).mean()
    collapsed = energy_score(torch.zeros_like(first), torch.zeros_like(second), target).mean()
    torch.testing.assert_close(correct, torch.tensor(.5, device='cuda'))
    torch.testing.assert_close(collapsed, torch.tensor(1., device='cuda'))


def test_learned_representation_preserves_vector_supervision_and_absorbing_origin():
    representation = TransitionRepresentation(12, latent_dim=16, width=32).cuda()
    decoder = RewardDecoder(16, width=32).cuda()
    transitions = torch.randn(256, 12, device='cuda')
    compiled = torch.compile(representation.loss, fullgraph=True, mode='reduce-overhead')
    compiled(transitions).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in representation.parameters())
    assert all(p.grad is None for p in decoder.parameters())
    representation.zero_grad(set_to_none=True)
    latent = representation(transitions).detach()
    assert latent.shape == (256, 16)
    decoder(latent).square().mean().backward()
    assert all(p.grad is None for p in representation.parameters())
    torch.testing.assert_close(representation(torch.zeros(1, 12, device='cuda')), torch.zeros(1, 16, device='cuda'))
    torch.testing.assert_close(representation.reconstruct(torch.zeros(1, 16, device='cuda')), torch.zeros(1, 12, device='cuda'))


def test_beta_pathwise_vector_credit_matches_analytic_expectation():
    n = 131072
    raw = torch.tensor([-.7, .5, 1.1, -.2], device='cuda', dtype=torch.float64)
    logits = raw.expand(n, -1).clone()
    weights = torch.tensor([1.7, -.6], device='cuda', dtype=torch.float64)
    observations = torch.zeros(n, 1, device='cuda', dtype=torch.float64)
    noise = torch.zeros_like(observations)
    actual, action, vector, credit = pathwise_credit(logits, observations, lambda s, a, e: a,
                                                   lambda z: (z * weights).sum(-1), noise, 0.)
    independent = raw.clone().requires_grad_(True)
    alpha, beta = (F.softplus(independent) + 1).chunk(2, -1)
    expected, = torch.autograd.grad((alpha / (alpha + beta) * weights).sum(), independent)
    torch.testing.assert_close(actual.mean(0), expected, rtol=.007, atol=.0003)
    torch.testing.assert_close(action, weights.expand_as(action))
    torch.testing.assert_close(credit, weights.expand_as(credit))
    assert vector.shape == (n, 2) and vector.var(0).min() > 0


def test_counterfactual_vector_advantages_match_coordinate_effects_and_expected_gradient():
    count, dimensions = 262144, 2
    raw = torch.tensor([-.7, .5, 1.1, -.2], device='cuda', dtype=torch.float64)
    alpha, beta = (F.softplus(raw) + 1).chunk(2, -1)
    distribution = torch.distributions.Beta(alpha, beta)
    actions = distribution.sample((count,))
    donors = distribution.sample((count, dimensions))
    contexts = counterfactual_contexts(actions, donors)
    torch.testing.assert_close(contexts.diagonal(dim1=1, dim2=2), actions)
    torch.testing.assert_close(contexts[:, 0, 1], donors[:, 0, 1])
    torch.testing.assert_close(contexts[:, 1, 0], donors[:, 1, 0])
    observations = torch.zeros(count, 1, device='cuda', dtype=torch.float64)
    noise = torch.zeros(count, dimensions, 1, device='cuda', dtype=torch.float64)
    weights = torch.tensor([1.7, -.6], device='cuda', dtype=torch.float64)
    advantages, latent = counterfactual_advantages(observations, actions, donors, noise,
                                                   lambda s, a, e: a, lambda z: (z * weights).sum(-1), 0.)
    expected = (actions - donors.diagonal(dim1=1, dim2=2)) * weights
    torch.testing.assert_close(advantages, expected)
    assert latent.shape == (count, dimensions, dimensions)
    torch.testing.assert_close(latent[:, 0, 1], torch.zeros_like(latent[:, 0, 1]))
    torch.testing.assert_close(latent[:, 1, 0], torch.zeros_like(latent[:, 1, 0]))
    logits = raw.expand(count, -1)
    tangents = torch.eye(4, device='cuda', dtype=torch.float64)[:, None, :].expand(-1, count, -1)
    actual_gradient = (projected_coordinate_scores(logits, tangents, actions) * advantages[:, None, :]).sum(-1).mean(0)
    independent = raw.clone().requires_grad_(True)
    a, b = (F.softplus(independent) + 1).chunk(2, -1)
    expected_gradient, = torch.autograd.grad((weights * a / (a + b)).sum(), independent)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=.025, atol=.0005)


def test_compiled_model_backward_and_reward_separation():
    critic = SuccessorCritic(5, 2, noise_dim=4, width=16).cuda()
    decoder = RewardDecoder(12, width=16).cuda()
    obs = torch.randn(256, 5, device='cuda')
    action = torch.rand(256, 2, device='cuda')
    noise1, noise2 = torch.randn(2, 256, 4, device='cuda').unbind()
    target = torch.randn(256, 12, device='cuda')

    def loss_model(s, a, e1, e2, y):
        return energy_score(critic(s, a, e1), critic(s, a, e2), y).mean()

    compiled = torch.compile(loss_model, fullgraph=True, mode='reduce-overhead')
    compiled(obs, action, noise1, noise2, target).backward()
    assert all(p.grad is None for p in decoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
    critic.zero_grad(set_to_none=True)
    decoder(target).square().mean().backward()
    assert all(p.grad is None for p in critic.parameters())
    decoder.zero_grad(set_to_none=True)
    critic.requires_grad_(False)
    decoder.requires_grad_(False)
    latent_model = torch.compile(critic, fullgraph=True, mode='reduce-overhead')
    reward_model = torch.compile(decoder, fullgraph=True, mode='reduce-overhead')
    def counterfactual_model(s, a, b, e):
        return counterfactual_advantages(s, a, b, e, critic, decoder, .99)

    compiled_counterfactual = torch.compile(counterfactual_model, fullgraph=True, mode='reduce-overhead')
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        accumulated = torch.zeros(256, 4, device='cuda')
        for _ in range(4):
            with torch.no_grad():
                vector_adv, difference = compiled_counterfactual(obs, action, torch.rand(256, 2, 2, device='cuda'),
                                                                 torch.randn(256, 2, 4, device='cuda'))
                assert vector_adv.shape == (256, 2) and difference.shape == (256, 2, 12)
                assert torch.isfinite(vector_adv).all()
            credit, action_credit, vector, costate = pathwise_credit(torch.randn(256, 4, device='cuda'), obs,
                                                                   latent_model, reward_model, noise1, .99)
            assert torch.isfinite(credit).all() and torch.isfinite(action_credit).all()
            assert credit.norm() > 0
            assert vector.shape == costate.shape == target.shape
            accumulated.add_(credit)
        with torch.no_grad():
            forecast = latent_model(obs, action, noise1).clone()
            assert torch.isfinite(forecast).all() and torch.isfinite(accumulated).all()
    torch.testing.assert_close(decoder(torch.zeros(1, 12, device='cuda')), torch.zeros(1, device='cuda'))


def test_compiled_projection_matches_score_autograd():
    actor = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4)).cuda()
    obs = torch.randn(32, 3, device='cuda')
    native = torch.rand(32, 2, device='cuda').clamp(.01, .99)
    directions = sample_parameter_directions(actor, 4, torch.Generator(device='cuda').manual_seed(123))
    projection = torch.compile(make_logit_projector(actor), fullgraph=True, mode='reduce-overhead')
    with torch.no_grad():
        tangents = projection(obs, directions).clone()
        actual = projected_score(actor(obs), tangents, native).mean(0)
    alpha, beta = (F.softplus(actor(obs)) + 1).chunk(2, -1)
    gradient = torch.autograd.grad(torch.distributions.Beta(alpha, beta).log_prob(native).sum(-1).mean(), tuple(actor.parameters()))
    expected = sum((directions[name] * grad).flatten(1).sum(1) for (name, _), grad in zip(actor.named_parameters(), gradient))
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


def test_gate_rejects_bias_and_uncertain_reference():
    rows = [{'reference_gradient': [1., 2., 3.], 'model_gradient': [1., 2., 3.], 'reward_mse': .001,
             'reward_variance': 1., 'representation_normalized_mse': .01} for _ in range(32)]
    assert summarize_gate(rows, 1)['passed']
    biased = [dict(row, model_gradient=[-1., -2., -3.]) for row in rows]
    assert summarize_gate(biased, 1)['status'] == 'failed'
    uncertain = [dict(row, reference_gradient=[(-1.)**i, 0., 0.]) for i, row in enumerate(rows)]
    assert summarize_gate(uncertain, 1)['status'] == 'inconclusive'


def test_decoder_factorial_inputs_keep_latent_and_context_separate():
    transitions = torch.randn(64, 9, device='cuda')
    latents = torch.randn(64, 12, device='cuda')
    context = torch.eye(4, device='cuda').repeat(16, 1)
    torch.testing.assert_close(decoder_inputs('transition_256', transitions, latents, context), transitions)
    torch.testing.assert_close(decoder_inputs('latent_128_scaled', transitions, latents, context), latents)
    actual = decoder_inputs('latent_context_256', transitions, latents, context)
    torch.testing.assert_close(actual[:, :12], latents)
    torch.testing.assert_close(actual[:, 12:], context)
    actual = decoder_inputs('transition_context_256', transitions, latents, context)
    torch.testing.assert_close(actual[:, :9], transitions)
    torch.testing.assert_close(actual[:, 9:], context)


def test_same_state_horizon_targets_use_discounted_rewards_and_correct_tail():
    rewards = torch.arange(20, device='cuda', dtype=torch.float64).reshape(10, 2)
    next_values = 100 + rewards
    gamma, horizon = .8, 3
    zero, value = discounted_windows(rewards, next_values, horizon, gamma)
    for start in range(10 - horizon + 1):
        expected = rewards[start] + gamma * rewards[start + 1] + gamma**2 * rewards[start + 2]
        torch.testing.assert_close(zero[start], expected)
        torch.testing.assert_close(value[start], expected + gamma**horizon * next_values[start + horizon - 1])
    # Actual use applies the SAME long-span mask to all horizons and the model.
    same_states = torch.arange(10, device='cuda') < 10 - 6 + 1
    zero_long, _ = discounted_windows(rewards, next_values, 6, gamma)
    assert zero_long[same_states].shape == zero[same_states].shape


def test_reference_split_check_rejects_opposing_halves_despite_nonzero_mean():
    rows = [{'reference_gradient': [10., 0.], 'model_gradient': [3., 0.],
             'reward_mse': .001, 'reward_variance': 1., 'representation_normalized_mse': .01} for _ in range(128)]
    rows += [dict(row, reference_gradient=[-4., 0.]) for row in rows]
    result = summarize_gate(rows, 1)
    assert result['reference_snr'] > 3
    assert result['reference_split_mean_dot_lower95'] < 0
    assert result['status'] == 'inconclusive' and not result['passed']

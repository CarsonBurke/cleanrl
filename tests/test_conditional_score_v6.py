"""CUDA estimator contracts; execute through mlq, not as reduced training runs."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl.ppo_continuous_action_conditional_score_v6 import (
    ConditionalScoreCritic, FrozenObsNorm, actor_surrogate, beta_geometry,
    complete_episode_layout, compare_gradients, discounted_episode_returns, episode_moments,
    make_logit_projector, project_latent, sample_band, sample_parameter_directions,
    score_tangents, summarize_gate, unwhiten,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), 'Run CUDA contracts through mlq'
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)


def test_whitened_beta_statistics_have_identity_covariance_and_exact_inverse():
    logits = torch.tensor([[-1., .7, 1.3, -.2]], device='cuda', dtype=torch.float64).expand(250000, -1)
    alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
    action = torch.distributions.Beta(alpha, beta).sample()
    z, factor, _ = beta_geometry(logits, action)
    score = torch.stack((action.log() - alpha.digamma() + (alpha + beta).digamma(),
                         torch.log1p(-action) - beta.digamma() + (alpha + beta).digamma()), -1)
    torch.testing.assert_close(unwhiten(z, factor), score, rtol=1e-12, atol=1e-12)
    for dimension in range(2):
        torch.testing.assert_close(torch.cov(z[:, dimension].T), torch.eye(2, device='cuda', dtype=torch.float64), rtol=.025, atol=.012)
    assert z.mean(0).abs().max() < .009


def test_compiled_projection_matches_autograd_in_both_alpha_beta_directions():
    actor = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)).cuda()
    state = torch.randn(128, 3, device='cuda')
    actions = torch.rand(128, 2, device='cuda').clamp(.01, .99)
    directions = sample_parameter_directions(actor, 8, torch.Generator(device='cuda').manual_seed(12))
    projector = torch.compile(make_logit_projector(actor), fullgraph=True, mode='reduce-overhead')
    geometry = torch.compile(beta_geometry, fullgraph=True, mode='reduce-overhead')
    with torch.no_grad():
        logits = actor(state)
        z, factor, _ = (x.clone() for x in geometry(logits, actions))
        tangents = score_tangents(logits, projector(state, directions).clone(), factor)
        actual = project_latent(z, tangents).mean(0)
    alpha, beta = (F.softplus(actor(state)) + 1).chunk(2, -1)
    gradients = torch.autograd.grad(torch.distributions.Beta(alpha, beta).log_prob(actions).sum(-1).mean(), tuple(actor.parameters()))
    expected = sum((directions[name] * grad).flatten(1).sum(1) for (name, _), grad in zip(actor.named_parameters(), gradients))
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)


def test_vector_actor_surrogate_is_exact_local_score_gradient():
    logits = torch.randn(512, 6, device='cuda', requires_grad=True)
    actions = torch.rand(512, 3, device='cuda').clamp(.01, .99)
    rewards = torch.randn(512, device='cuda')
    z, factor, _ = beta_geometry(logits.detach(), actions)
    credit = unwhiten(z, factor) * rewards[:, None, None]
    actual, = torch.autograd.grad(actor_surrogate(logits, credit), logits)
    alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
    expected, = torch.autograd.grad((torch.distributions.Beta(alpha, beta).log_prob(actions).sum(-1) * rewards).mean(), logits)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-8)
    assert torch.linalg.matrix_rank(credit.flatten(1)) == 6


def test_finite_strata_recover_discount_mass_and_expected_time():
    remaining = torch.full((200000,), 137, device='cuda', dtype=torch.long)
    mass_sum = torch.zeros_like(remaining, dtype=torch.float64)
    time_sum = mass_sum.clone()
    for band in range(9):
        h, mass = sample_band(remaining, torch.full_like(remaining, band), .99)
        assert (h < remaining).all()
        mass_sum += mass
        time_sum += mass * h
    h = torch.arange(1, 137, device='cuda', dtype=torch.float64)
    expected_mass = (.99 ** h).sum()
    expected_time = (.99 ** h * h).sum()
    torch.testing.assert_close(mass_sum.mean(), expected_mass, rtol=2e-7, atol=2e-6)
    torch.testing.assert_close(time_sum.mean(), expected_time, rtol=.002, atol=.1)


def test_empty_strata_and_last_episode_step_have_safe_zero_weight():
    remaining = torch.tensor([1, 2, 3, 5, 17], device='cuda')
    for band in range(9):
        h, mass = sample_band(remaining, torch.full_like(remaining, band), .9)
        assert h[0] == 0 and mass[0] == 0
        assert ((h >= 0) & (h < remaining)).all()
        assert (h[mass == 0] == 0).all()


def test_complete_episode_selection_drops_entire_boundary_fragments():
    ages = np.array([[2, 0], [3, 1], [0, 2], [1, 3], [2, 0], [3, 1], [0, 2], [1, 3], [2, 0]])
    truncs = ages == 3
    ids, count = complete_episode_layout(ages, np.zeros_like(truncs), truncs, 4)
    assert count == 3
    np.testing.assert_array_equal(ids[:, 0], [-1, -1, 0, 0, 0, 0, -1, -1, -1])
    np.testing.assert_array_equal(ids[:, 1], [1, 1, 1, 1, 2, 2, 2, 2, -1])
    terms = np.zeros_like(truncs)
    terms[3, 0] = True
    with pytest.raises(ValueError, match='early terminations'):
        complete_episode_layout(ages, terms, truncs, 4)


def test_parallel_discount_scan_matches_exact_episode_sums_without_bootstrap():
    ids = torch.tensor([[-1, 0], [-1, 0], [1, 0], [1, 0], [1, 2], [1, 2], [-1, 2], [-1, 2], [-1, -1]], device='cuda')
    rewards = torch.arange(18, device='cuda', dtype=torch.float64).reshape(9, 2)
    scan = torch.compile(discounted_episode_returns, fullgraph=True, mode='reduce-overhead')
    result = scan(rewards, ids, .83)
    for t in range(9):
        for env in range(2):
            expected = 0.
            if ids[t, env] >= 0:
                for end in range(t, 9):
                    if ids[end, env] != ids[t, env]:
                        break
                    expected += .83 ** (end - t) * float(rewards[end, env])
            assert abs(float(result[t, env]) - expected) < 1e-10


def test_episode_covariance_keeps_cross_time_terms():
    ids = torch.tensor([0, 0, 1, 1, -1], device='cuda')
    contributions = torch.tensor([[1., 2.], [-1., -2.], [2., 1.], [-2., -1.], [100., 100.]], device='cuda')
    mean, square, sums = episode_moments(contributions, ids, 2)
    torch.testing.assert_close(sums, torch.zeros(2, 2, device='cuda', dtype=torch.float64))
    assert square == 0 and mean.norm() == 0


@pytest.mark.parametrize('physical', [True, False])
def test_compiled_vector_critic_backward_and_frozen_prediction(physical):
    model = ConditionalScoreCritic(5, 3, 4, width=32, physical_outcome=physical).cuda()
    state = torch.randn(256, 5, device='cuda')
    parameters = torch.rand(256, 3, 2, device='cuda') + 1
    age = torch.arange(256, device='cuda')
    environment = age % 4
    outcome = torch.randn(256, 14, device='cuda')
    reward = torch.randn(256, device='cuda')
    horizon = torch.ones_like(age)
    target = torch.randn(256, 3, 2, device='cuda')

    def loss(*inputs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(*inputs)
        return (output.float() - target).square().mean()

    compiled = torch.compile(loss, fullgraph=True, mode='reduce-overhead')
    compiled(state, parameters, age, environment, outcome, reward, horizon).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.requires_grad_(False)
    with torch.no_grad():
        predicted = model(state, parameters, age, environment, outcome, reward, horizon)
        assert predicted.shape == (256, 3, 2)
        changed = model(state, parameters, age, environment, outcome + 10, reward, horizon)
        if not physical:
            torch.testing.assert_close(predicted, changed)
        else:
            assert not torch.allclose(predicted, changed)


def test_conditional_score_identity_and_per_term_covariance_on_exact_joint_law():
    # Four actions, three REAL outcomes, vector policy score. All calculations
    # enumerate the exact joint law: no learned posterior or Monte Carlo excuse.
    joint = torch.tensor([[.12, .05, .03], [.01, .07, .02], [.03, .09, .18], [.14, .09, .17]], device='cuda', dtype=torch.float64)
    prior = joint.sum(1)
    scores = torch.eye(4, device='cuda', dtype=torch.float64) - prior
    outcome_probability = joint.sum(0)
    posterior_mean = joint.T @ scores / outcome_probability[:, None]
    rewards = torch.tensor([-2., 1., 4.], device='cuda', dtype=torch.float64)
    original = torch.einsum('ay,y,ak->k', joint, rewards, scores)
    assert original.norm() > .1
    conditioned = torch.einsum('y,y,yk->k', outcome_probability, rewards, posterior_mean)
    torch.testing.assert_close(original, conditioned)
    original_second = torch.einsum('ay,y,ak,al->kl', joint, rewards.square(), scores, scores)
    conditional_second = torch.einsum('y,y,yk,yl->kl', outcome_probability, rewards.square(), posterior_mean, posterior_mean)
    assert torch.linalg.eigvalsh(original_second - conditional_second).min() > -1e-12


def test_rich_outcome_that_reveals_action_cannot_reduce_score_variance():
    prior = torch.tensor([.2, .3, .5], device='cuda', dtype=torch.float64)
    scores = torch.eye(3, device='cuda', dtype=torch.float64) - prior
    joint = torch.diag(prior)
    conditional = joint.T @ scores / prior[:, None]
    torch.testing.assert_close(conditional, scores)


def test_frozen_normalization_uses_physical_final_observation():
    state = {'means': torch.zeros(2, 2), 'variances': torch.ones(2, 2), 'counts': torch.ones(2), 'epsilon': 0., 'clip': 100.}
    normalizer = FrozenObsNorm(state, 2, (2,))
    normalized, physical = normalizer.normalize_step(np.array([[20., 20.], [30., 30.]]), np.array([False, False]),
                                                     np.array([True, True]), {'final_observation': [np.array([1., 2.]), np.array([3., 4.])]})
    np.testing.assert_array_equal(physical, [[1., 2.], [3., 4.]])
    np.testing.assert_array_equal(normalized, [[20., 20.], [30., 30.]])


def gate_rows():
    rows = []
    for index in range(32):
        mean = np.array([1 + .01 * (-1) ** index, 2.])
        row = {'episodes': 10}
        for name in ('reference', 'sampled_reference', 'physical', 'reduced'):
            row[name + '_gradient'] = mean.tolist()
            variance = 1. if name in ('reference', 'sampled_reference') else .5
            row[name + '_episode_square_sum'] = 10 * float(mean @ mean) + 9 * variance
        rows.append(row)
    return rows


def test_gate_requires_mean_agreement_and_variance_improvement():
    rows = gate_rows()
    assert summarize_gate(rows, 1)['passed']
    for row in rows:
        row['physical_gradient'] = (-np.asarray(row['physical_gradient'])).tolist()
    assert summarize_gate(rows, 1)['status'] == 'failed'
    rows = gate_rows()
    for row in rows:
        row['physical_episode_square_sum'] = row['reference_episode_square_sum']
    assert not summarize_gate(rows, 1)['passed']


def test_exploratory_ablation_cannot_promote_failed_primary():
    rows = gate_rows()
    for row in rows:
        row['physical_episode_square_sum'] = row['reference_episode_square_sum']
    report = summarize_gate(rows, 1)
    assert report['comparisons']['reduced']['passed']
    assert not report['passed']


def test_shrinking_gradient_alone_cannot_pass_variance_gate():
    rows = gate_rows()
    for row in rows:
        row['physical_gradient'] = (.6 * np.asarray(row['reference_gradient'])).tolist()
        row['physical_episode_square_sum'] = .36 * row['reference_episode_square_sum']
    report = summarize_gate(rows, 1)
    assert report['comparisons']['physical']['episode_variance_ratio'] < .4
    assert abs(report['comparisons']['physical']['signal_normalized_variance_ratio'] - 1) < 1e-12
    assert not report['passed']


def test_gradient_comparison_weights_unequal_episode_counts():
    reference = np.tile([[1., 0.], [0., 1.]], (16, 1))
    counts = np.tile([1., 9.], 16)
    model = np.tile([.1, .9], (32, 1))
    rng = np.random.default_rng(1)
    indices = rng.integers(32, size=(4096, 32))
    result = compare_gradients(reference, model, rng, indices, counts)
    assert result['relative_gradient_error'] < 1e-12
    assert abs(result['norm_ratio'] - 1) < 1e-12

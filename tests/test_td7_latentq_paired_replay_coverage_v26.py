"""Pure tensor/probability and metadata contracts; no models or simulations."""
import copy
import importlib.util
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl/td7/latentq/td7_latentq_paired_replay_coverage_v26.py"
SPEC = importlib.util.spec_from_file_location("paired_replay_coverage_v26", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_paired_squared_loss_decomposes_into_mean_and_quarter_contrast():
    center = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    odd = torch.tensor([[0.5, -0.25], [2.0, -1.0]])
    even = torch.tensor([[-0.5, 0.25], [1.0, 0.5]])
    plus, minus = torch.tensor([[4.0], [9.0]]), torch.tensor([[1.0], [3.0]])
    loss = MODULE.paired_hermite_mse(center + odd + even, center - odd + even, plus, minus)
    mean_loss = (center + even - 0.5 * (plus + minus)).square().sum(-1).mean()
    contrast_loss = (2 * odd - (plus - minus)).square().sum(-1).mean()
    torch.testing.assert_close(loss, mean_loss + 0.25 * contrast_loss)


def test_uniform_direction_projection_recovers_clipped_gaussian_mean_derivative():
    # Numerical probability integration, not a neural model. For G=clip(U),
    # d_mu E[G] is the probability that the raw Gaussian is inside its bounds.
    eps = torch.linspace(-9.0, 9.0, 40001, dtype=torch.float64)
    density = torch.exp(-0.5 * eps.square()) / np.sqrt(2 * np.pi)
    mu, sigma = 0.8, 0.4
    plus, minus = (mu + sigma * eps).clamp(-1, 1), (mu - sigma * eps).clamp(-1, 1)
    contrast = plus - minus
    phi = lambda x: 0.5 * (1 + torch.erf(torch.as_tensor(x, dtype=torch.float64) / np.sqrt(2)))
    expected_derivative = phi((1 - mu) / sigma) - phi((-1 - mu) / sigma)
    moment = torch.trapezoid(eps * contrast * density, eps) / (2 * sigma)
    torch.testing.assert_close(moment, expected_derivative, atol=2e-8, rtol=2e-8)
    g = expected_derivative.detach().requires_grad_()
    objective = torch.trapezoid((2 * sigma * eps * g - contrast).square() * density, eps)
    gradient = torch.autograd.grad(objective, g)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-8, rtol=0)
    # Actual clipped displacements are not valid substitutes for raw sigma*eps.
    assert not torch.allclose(plus - minus, 2 * sigma * eps)


def test_lambda_stops_at_truncation_but_bootstraps_final_observation():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[100.0]], [[4.0]]])
    next_values = torch.tensor([[[10.0]], [[20.0]], [[30.0]], [[40.0]]])
    mask = torch.ones_like(rewards)
    boundary = torch.tensor([[[0.0]], [[1.0]], [[0.0]], [[0.0]]])
    target = MODULE.lambda_returns(rewards, mask, boundary, next_values, 0.9, 0.5, 4)
    torch.testing.assert_close(target[:2], torch.tensor([[[14.5]], [[20.0]]]))
    rewards[2:] += 1000
    changed = MODULE.lambda_returns(rewards, mask, boundary, next_values, 0.9, 0.5, 4)
    torch.testing.assert_close(changed[:2], target[:2])
    mask[1] = 0
    terminal = MODULE.lambda_returns(rewards, mask, boundary, next_values, 0.9, 0.5, 4)
    torch.testing.assert_close(terminal[1], torch.tensor([[2.0]]))


def test_trace_segments_do_not_recurse_into_the_next_segment():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[100.0]], [[200.0]]])
    values = torch.full_like(rewards, 10)
    target = MODULE.lambda_returns(rewards, torch.ones_like(rewards), torch.zeros_like(rewards), values, 0.9, 1.0, 2)
    torch.testing.assert_close(target[:2], torch.tensor([[[10.9]], [[11.0]]]))


def test_snapshot_metadata_copies_native_age_and_rng_without_aliasing():
    def fake(count, seed):
        return SimpleNamespace(
            _episode_lengths=np.arange(count, dtype=np.int32) + 9,
            _episode_returns=np.arange(count, dtype=np.float32) + 13,
            _episode_starts=np.arange(count, dtype=np.float32) + 17,
            _limits=[SimpleNamespace(_elapsed_steps=9 + i) for i in range(count)],
            _bases=[SimpleNamespace(np_random=np.random.default_rng(seed + i)) for i in range(count)],
            _statistics=[SimpleNamespace(return_queue=deque([3.0]), length_queue=deque([7]), episode_count=2) for _ in range(count)],
            observations=np.arange(count * 3, dtype=np.float64).reshape(count, 3),
        )
    source, destination = fake(1, 3), fake(2, 90)
    initial_rng = copy.deepcopy(source._bases[0].np_random.bit_generator.state)
    MODULE.copy_episode_metadata(source, destination, 0, 1, source.observations[0])
    assert destination._episode_lengths[1] == source._episode_lengths[0]
    assert destination._limits[1]._elapsed_steps == source._limits[0]._elapsed_steps
    assert destination._episode_returns[1] == source._episode_returns[0]
    np.testing.assert_array_equal(destination.observations[1], source.observations[0])
    assert destination._bases[1].np_random.bit_generator.state == initial_rng
    destination._bases[1].np_random.normal()
    assert source._bases[0].np_random.bit_generator.state == initial_rng
    destination._statistics[1].return_queue.append(100)
    assert list(source._statistics[0].return_queue) == [3.0]
    destination.observations[1] = -999
    assert source.observations[0, 0] == 0


def test_budget_counts_every_shadow_step_including_masked_post_boundary_rows():
    anchors = MODULE.segment_anchor_steps(1024, 32)
    np.testing.assert_array_equal(anchors, np.arange(8) * 128)
    cost = MODULE.rollout_physics_budget(1024, 16, 32)
    assert cost == 16384 + 4096
    warmup_transitions = 10000
    iterations = (8000000 - warmup_transitions) // cost
    assert warmup_transitions + iterations * cost <= 8000000
    assert warmup_transitions + (iterations + 1) * cost > 8000000


def test_odd_even_projection_identifies_clipped_mean_and_scale_moments():
    eps = torch.linspace(-9.0, 9.0, 40001, dtype=torch.float64)
    density = torch.exp(-0.5 * eps.square()) / np.sqrt(2 * np.pi)
    mu, sigma = 0.8, 0.4
    alpha, beta = (-1 - mu) / sigma, (1 - mu) / sigma
    cdf = lambda x: 0.5 * (1 + torch.erf(torch.as_tensor(x, dtype=torch.float64) / np.sqrt(2)))
    pdf = lambda x: torch.exp(-0.5 * torch.as_tensor(x, dtype=torch.float64).square()) / np.sqrt(2 * np.pi)
    probability = cdf(beta) - cdf(alpha)
    true_center = mu * probability + sigma * (pdf(alpha) - pdf(beta)) + 1 - cdf(beta) - cdf(alpha)
    center = true_center.detach().requires_grad_()
    mean_derivative = probability.detach().requires_grad_()
    scale_derivative = (pdf(alpha) - pdf(beta)).detach().requires_grad_()
    odd = sigma * eps * mean_derivative
    even = 0.5 * sigma * (eps.square() - 1) * scale_derivative
    plus, minus = (mu + sigma * eps).clamp(-1, 1), (mu - sigma * eps).clamp(-1, 1)
    objective = torch.trapezoid(0.5 * ((center + odd + even - plus).square() +
                                    (center - odd + even - minus).square()) * density, eps)
    grads = torch.autograd.grad(objective, (center, mean_derivative, scale_derivative))
    for gradient in grads:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-8, rtol=0)


def test_zero_norm_statistics_are_undefined_not_measured_orthogonality():
    ratios = MODULE.defined_stat_ratio(torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 2.0]))
    assert torch.isnan(ratios[:2]).all()
    torch.testing.assert_close(ratios[2], torch.tensor(0.0))


def test_snapshot_observation_uses_rollout_state_when_native_cache_is_stale():
    source = SimpleNamespace(
        _episode_lengths=np.array([0], dtype=np.int32),
        _episode_returns=np.zeros(1, dtype=np.float32),
        _episode_starts=np.zeros(1, dtype=np.float32),
        _limits=[SimpleNamespace(_elapsed_steps=0)],
        _bases=[SimpleNamespace(np_random=np.random.default_rng(1))],
        _statistics=[SimpleNamespace()],
        observations=np.array([[100.0, 200.0, 300.0]]),
    )
    destination = SimpleNamespace(
        _episode_lengths=np.array([999], dtype=np.int32),
        _episode_returns=np.zeros(1, dtype=np.float32),
        _episode_starts=np.zeros(1, dtype=np.float32),
        _limits=[SimpleNamespace(_elapsed_steps=999)],
        _bases=[SimpleNamespace(np_random=np.random.default_rng(2))],
        _statistics=[SimpleNamespace()],
        observations=np.zeros((1, 3)),
    )
    rollout_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    MODULE.copy_episode_metadata(source, destination, 0, 0, rollout_obs)
    np.testing.assert_array_equal(destination.observations[0], rollout_obs)
    np.testing.assert_array_equal(source.observations[0], [100.0, 200.0, 300.0])
    assert destination._episode_lengths[0] == destination._limits[0]._elapsed_steps == 0
    rollout_obs[:] = -50
    np.testing.assert_array_equal(destination.observations[0], [1.0, 2.0, 3.0])


def test_branch_validity_includes_first_boundary_and_excludes_autoreset():
    boundary = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 1., 0.], [1., 0., 1.]])
    expected = torch.tensor([[1., 1., 1.], [1., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    torch.testing.assert_close(MODULE.branch_validity(boundary), expected)


def test_static_uniform_compensation_and_observation_count_mixture():
    eligible = torch.tensor([1., 0., 1., 0., 1., 0.])
    observations = torch.tensor([2., 100., 4., 100., 9., 100.])
    weights, count = MODULE.masked_uniform_weight(eligible)
    # Enumerating every equiprobable fixed slot gives the eligible mean exactly.
    torch.testing.assert_close((weights * observations).mean(), torch.tensor(5.))
    main = torch.tensor([1., 3.])
    result = MODULE.observation_mixture(main.mean(), (weights * observations).mean(), len(main), count)
    torch.testing.assert_close(result, torch.tensor((1 + 3 + 2 + 4 + 9) / 5))
    empty_weights, empty_count = MODULE.masked_uniform_weight(torch.zeros(6))
    torch.testing.assert_close(empty_weights, torch.zeros(6))
    torch.testing.assert_close(MODULE.observation_mixture(main.mean(), torch.tensor(0.), len(main), empty_count), main.mean())


def test_prediction_and_label_clocks_disentangle_refresh_from_fit():
    old, new = torch.tensor([0., 0.]), torch.tensor([1., 1.])
    initial, final = torch.tensor([0., 0.]), torch.tensor([2., 2.])
    errors = MODULE.prediction_label_errors(old, new, initial, final)
    # A refreshed-label error improves 4->1 while frozen-label error worsens0->1.
    torch.testing.assert_close(errors, torch.tensor([0., 4., 1., 1.]))


def test_branch_range_excludes_invalid_extremes_and_empty_batch_is_neutral():
    target = torch.tensor([[3.], [1e9], [-1e9], [7.]])
    maximum, minimum = MODULE.masked_target_extrema(target, torch.tensor([1., 0., 0., 1.]))
    torch.testing.assert_close(maximum, torch.tensor(7.))
    torch.testing.assert_close(minimum, torch.tensor(3.))
    empty_max, empty_min = MODULE.masked_target_extrema(target, torch.zeros(4))
    assert torch.isneginf(empty_max) and torch.isposinf(empty_min)
    torch.testing.assert_close(torch.maximum(torch.tensor(9.), empty_max), torch.tensor(9.))
    torch.testing.assert_close(torch.minimum(torch.tensor(2.), empty_min), torch.tensor(2.))


def test_quadratic_shared_preaction_target_conflicts_with_center_average():
    # Three-node normal quadrature integrates the second/fourth moments exactly.
    eps = torch.tensor([-3.0 ** 0.5, 0., 3.0 ** 0.5], dtype=torch.float64)
    probability = torch.tensor([1/6, 2/3, 1/6], dtype=torch.float64)
    sigma = .4
    b = torch.tensor(.75, dtype=torch.float64, requires_grad=True)
    # F(mu,sigma,u)=u²+b sigma²; at mu0 the Hermite predictions areb sigma² eps².
    pair_loss = (probability * ((b - 1) * sigma**2 * eps.square()).square()).sum()
    pre_loss = b.square() * sigma**4
    original_gradient = torch.autograd.grad(pair_loss + pre_loss, b, retain_graph=True)[0]
    torch.testing.assert_close(original_gradient, torch.tensor(0., dtype=torch.float64), atol=1e-15, rtol=0)
    # Removing incompatible preaction supervision still pushesb toward1.
    assert torch.autograd.grad(pair_loss, b)[0] < 0
    exact_b = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    exact_pair = (probability * ((exact_b - 1) * sigma**2 * eps.square()).square()).sum()
    torch.testing.assert_close(torch.autograd.grad(exact_pair, exact_b)[0], torch.tensor(0., dtype=torch.float64))
    # Sigma0 point queries remain exactlyu² for everyb.
    raw = sigma * eps
    torch.testing.assert_close(raw.square() + exact_b * 0., raw.square())


def test_critic_loss_and_priority_depend_on_point_not_untrained_preaction():
    pre = torch.tensor([[100., -200.], [300., -400.]], requires_grad=True)
    point = torch.tensor([[1., 3.], [4., 6.]], requires_grad=True)
    target = torch.tensor([[2.], [5.]])
    importance = torch.tensor([.5, 1.5])
    fake = SimpleNamespace(_fields={"target": slice(0, 1)},
                           _direct_values=lambda batch: (pre, point, torch.zeros(2, 1)))
    loss, priority, _, _ = MODULE.TD7Agent._critic_loss_fn(fake, target, importance)
    grad_pre, grad_point = torch.autograd.grad(loss, (pre, point), allow_unused=True)
    assert grad_pre is None
    torch.testing.assert_close(grad_point, torch.tensor([[-.5, .5], [-1.5, 1.5]]))
    torch.testing.assert_close(priority, torch.ones(2, 2))
    assert not priority.requires_grad


def test_raw_history_ring_wrap_owns_stable_atomic_pair_copies():
    ring = MODULE.RawHistoryRing(3, (2, 2), 'cpu')
    source = torch.arange(8.).view(2, 2, 2)
    ring.append(source, 10)
    expected = source.clone()
    source.fill_(-999)
    torch.testing.assert_close(ring.rows[:2], expected)
    assert ring.position == 2 and ring.size == 2 and ring.count == 2
    newest = torch.arange(8., 16.).view(2, 2, 2)
    ring.append(newest, 20)
    torch.testing.assert_close(ring.rows, torch.stack([newest[1], expected[1], newest[0]]))
    sampled, stamps = ring.sample(100)
    # Every sampled pair is a whole original antithetic entry, not crossed rows.
    for pair, stamp in zip(sampled, stamps):
        assert any(torch.equal(pair, entry) for entry in ring.rows)
        assert stamp in (10, 20)
    sampled.fill_(-888)
    assert not (ring.rows == -888).any()
    oversized = torch.arange(20.).view(5, 2, 2)
    ring.append(oversized, 30)
    assert ring.size == 3 and ring.count == 3
    assert all(any(torch.equal(entry, row) for row in ring.rows) for entry in oversized[-3:])
    assert (ring.stamps == 30).all()
    stable = ring.rows.clone()
    ring.append(torch.empty(0, 2, 2), 40)
    torch.testing.assert_close(ring.rows, stable)


def test_history_append_excludes_lane_zero_and_post_boundary_but_keeps_terminal_row():
    # Three offsets, two lanes. Lane1 terminates at offset1, so row5 is reset.
    fresh = torch.tensor([[0.], [1.], [2.], [3.]])
    branches = torch.tensor([[10.], [11.], [12.], [13.], [14.], [15.]])
    writer = SimpleNamespace(add_scalar=lambda *args: None)
    fake = SimpleNamespace(
        fresh=fresh, branches=branches, branch_size=6, pair_count=2,
        branch_valid=MODULE.branch_validity(torch.tensor([[0., 0.], [0., 1.], [0., 0.]])).flatten(),
        fitting_mask=torch.tensor([False, True, False, True]),
        anchor_indices=torch.tensor([0, 1]), pair_fit_indices=torch.tensor([1]),
        device='cpu', args=SimpleNamespace(num_envs=2), writer=writer,
        point_history=MODULE.RawHistoryRing(8, (1,), 'cpu'),
        pair_history=MODULE.RawHistoryRing(8, (2, 1), 'cpu'),
    )
    MODULE.TD7Agent.append_history(fake, 100)
    torch.testing.assert_close(fake.point_history.rows[:4, 0], torch.tensor([1., 3., 11., 13.]))
    torch.testing.assert_close(fake.pair_history.rows[0, :, 0], torch.tensor([1., 11.]))
    assert fake.point_history.size == 4 and fake.pair_history.size == 1
    fresh.fill_(999)
    branches.fill_(888)
    torch.testing.assert_close(fake.point_history.rows[:4, 0], torch.tensor([1., 3., 11., 13.]))


def test_history_empty_sampling_and_loss_gating_recover_fresh_weights():
    ring = MODULE.RawHistoryRing(4, (2,), 'cpu')
    batch, stamps = ring.sample(32)
    assert batch.shape == (32, 2) and not batch.any() and not stamps.any()
    for enabled, expected in ((0., 8.), (1., 14.)):
        torch.testing.assert_close(MODULE.replay_mixture(torch.tensor(8.), torch.tensor(20.),
                                                       torch.tensor(enabled)), torch.tensor(expected))
    direct = MODULE.replay_mixture(torch.tensor(4.), torch.tensor(12.), torch.tensor(1.))
    hermite = MODULE.replay_mixture(torch.tensor(8.), torch.tensor(16.), torch.tensor(1.))
    # Four equal quarter-weight critic channels when both rings are populated.
    torch.testing.assert_close((direct + hermite) / 2, torch.tensor((4 + 12 + 8 + 16) / 4))


def test_one_step_terminal_and_truncation_depend_only_on_correct_final_observation():
    reward = torch.tensor([[2.], [3.]])
    mask = torch.tensor([[0.], [1.]])
    # First row true terminal; second truncated with final-observation value10.
    values = torch.tensor([[1000.], [10.]])
    target = MODULE.one_step_targets(reward, mask, values, .9)
    torch.testing.assert_close(target, torch.tensor([[2.], [12.]]))
    values[0] = -1000
    torch.testing.assert_close(MODULE.one_step_targets(reward, mask, values, .9), target)


def test_history_relabel_has_frozen_behavior_and_live_target_clocks_with_independent_members():
    # Fake scalar algebra only: no modules, neural forwards, or simulations.
    fields = {name: slice(i, i + 1) for i, name in enumerate(('next_state', 'reward', 'not_done', 'target'))}
    frozen = {'actor': 2., 'encoder': 3.}
    live = {'actor': 99., 'encoder': 77., 'critic': 1., 'target_encoder': 4.}
    fake = SimpleNamespace(
        _fields=fields, args=SimpleNamespace(gamma=.5, warmup_sigma=.7),
        behavior_warmup=torch.tensor(False), min_target=torch.tensor(-1e6), max_target=torch.tensor(1e6),
        behavior_actor=lambda state, zs: (zs * frozen['actor'], torch.zeros_like(state)),
        behavior_encoder=SimpleNamespace(zs=lambda state: state * frozen['encoder']),
        fixed_encoder_target=SimpleNamespace(zs=lambda state: state * live['target_encoder'], zsa=lambda zs, mu: zs + mu),
        critic_target=lambda state, latent, mu, zsa, zs: torch.cat([mu + zs + live['critic'], mu + zs + live['critic'] + 10], 1),
    )
    raw = torch.tensor([[1., 5., 1., -999.], [2., 7., 1., -999.]])
    relabeled = MODULE.TD7Agent._relabel_history_fn(fake, raw.clone())
    torch.testing.assert_close(relabeled[:, fields['target']], torch.tensor([[10.5], [17.5]]))
    live['actor'], live['encoder'] = -900., -800.
    same = MODULE.TD7Agent._relabel_history_fn(fake, raw.clone())
    torch.testing.assert_close(same, relabeled)
    live['critic'], live['target_encoder'] = 5., 6.
    changed = MODULE.TD7Agent._relabel_history_fn(fake, raw.clone())
    torch.testing.assert_close(changed[:, fields['target']], torch.tensor([[13.5], [21.5]]))
    fake.behavior_warmup.fill_(True)
    warmup = MODULE.TD7Agent._relabel_history_fn(fake, raw.clone())
    torch.testing.assert_close(warmup[:, fields['target']], torch.tensor([[10.5], [15.5]]))
    # Stored rows retain no fitted labels or mutated future states.
    torch.testing.assert_close(raw[:, fields['target']], torch.full((2, 1), -999.))


def test_crossed_prediction_probe_changes_only_requested_network_clock():
    # Scalar algebra verifies network injection without constructing a model.
    fields = {name: slice(i, i + 1) for i, name in enumerate(('state', 'mu', 'sigma', 'eps'))}
    batch = torch.tensor([[2., .5, .4, 1.5], [3., -.1, .2, -1.]])
    fake = SimpleNamespace(_fields=fields)
    def encoder(scale):
        return SimpleNamespace(zs=lambda state: scale * state,
                               zsa_jvp=lambda zs, action, tangent: (zs + action, scale * tangent))
    def critic(scale):
        def algebra(state, latent, action, zsa, zs, t_latent, t_action, t_zsa):
            center = scale * (zs + zsa + action)
            tangent = scale * (t_latent.sum(1, keepdim=True) + t_zsa)
            return center.expand(-1, 2), tangent.expand(-1, 2)
        return algebra
    old = MODULE.TD7Agent._pair_predictions_with_networks(fake, batch, encoder(1.), critic(2.))
    crossed = MODULE.TD7Agent._pair_predictions_with_networks(fake, batch, encoder(3.), critic(2.))
    new = MODULE.TD7Agent._pair_predictions_with_networks(fake, batch, encoder(3.), critic(5.))
    # Critic scaling changes all predictions by5/2 with the encoder clock fixed.
    for x, y in zip(crossed, new):
        torch.testing.assert_close(y, 2.5 * x)
    assert not torch.equal(old[2], crossed[2])
    # Both antithetic members retain equal centers despite network injection.
    torch.testing.assert_close((crossed[0] + crossed[1]) / 2 - crossed[2],
                               torch.tensor([[.5], [0.]]).expand(-1, 2))

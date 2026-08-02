import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "cleanrl" / "ppo_continuous_action_bpc_full_mnw_td_lambda_v1.py"
SPEC = importlib.util.spec_from_file_location("bpc_full_mnw_v1", MODULE_PATH)
BPC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BPC)


def random_spd(size, dtype=torch.float64):
    matrix = torch.randn(size, size, dtype=dtype)
    return matrix @ matrix.T + 0.5 * torch.eye(size, dtype=dtype)


def small_args(**overrides):
    values = dict(
        hidden_size=4,
        num_hidden_layers=2,
        compile=False,
        posterior_jitter=1e-9,
        prior_column_covariance=2.0,
        prior_wishart_scale=5.0,
    )
    values.update(overrides)
    return BPC.Args(**values)


def test_natural_roundtrip_spd_and_degrees_of_freedom():
    torch.manual_seed(0)
    dy, dx = 3, 5
    original = BPC.MNWParameters(
        torch.randn(dy, dx, dtype=torch.float64),
        random_spd(dx),
        random_spd(dy),
        torch.tensor(7.5, dtype=torch.float64),
    )
    natural = BPC.mnw_to_natural(original, jitter=1e-12)
    recovered = BPC.natural_to_mnw(natural, jitter=1e-12)
    for expected, actual in zip(original, recovered):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    torch.linalg.cholesky(recovered.V)
    torch.linalg.cholesky(recovered.Psi)
    assert recovered.nu > dy - 1

    invalid = natural._replace(xi=torch.tensor(-20.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="degrees of freedom"):
        BPC.natural_to_mnw(invalid)


def test_batched_recovery_matches_individual_recovery_and_stays_spd():
    torch.manual_seed(19)
    naturals = []
    for _ in range(5):
        parameters = BPC.MNWParameters(
            torch.randn(4, 5, dtype=torch.float64),
            random_spd(5),
            random_spd(4),
            torch.tensor(7.0, dtype=torch.float64),
        )
        naturals.append(BPC.mnw_to_natural(parameters))
    batched = BPC.recover_natural_group(naturals, 1e-9)
    for natural, actual in zip(naturals, batched):
        expected = BPC.natural_to_mnw(natural, 1e-9)
        for expected_value, actual_value in zip(expected, actual):
            torch.testing.assert_close(actual_value, expected_value)
        torch.linalg.cholesky(actual.V)
        torch.linalg.cholesky(actual.Psi)


def test_sufficient_statistic_orientation_and_sequential_conjugacy():
    x = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]], dtype=torch.float64)
    y = torch.tensor([[4.0, -2.0, 1.0], [0.5, 2.0, -1.0], [-3.0, 1.0, 2.0]], dtype=torch.float64)
    stats = BPC.sufficient_statistics(x, y)
    assert stats.Syx.shape == (3, 2)
    torch.testing.assert_close(stats.Syx, y.T @ x)
    assert stats.N.item() == 3

    prior_params = BPC.MNWParameters(
        torch.randn(3, 2, dtype=torch.float64),
        random_spd(2),
        random_spd(3),
        torch.tensor(5.0, dtype=torch.float64),
    )
    prior = BPC.mnw_to_natural(prior_params)
    batch_candidate = BPC.conjugate_candidate(prior, stats)
    sequential = BPC.add_statistics(prior, BPC.sufficient_statistics(x[:1], y[:1]))
    sequential = BPC.add_statistics(sequential, BPC.sufficient_statistics(x[1:], y[1:]))
    for expected, actual in zip(batch_candidate, sequential):
        torch.testing.assert_close(actual, expected)


def test_paper_style_stochastic_natural_update_targets_conjugate_candidate():
    torch.manual_seed(1)
    params = BPC.MNWParameters(
        torch.randn(2, 3, dtype=torch.float64),
        random_spd(3),
        random_spd(2),
        torch.tensor(4.0, dtype=torch.float64),
    )
    prior = BPC.mnw_to_natural(params)
    stats = BPC.sufficient_statistics(torch.randn(7, 3, dtype=torch.float64), torch.randn(7, 2, dtype=torch.float64))
    candidate = BPC.conjugate_candidate(prior, stats)
    updated = BPC.stochastic_natural_update(prior, candidate, 0.25)
    for old, target, actual in zip(prior, candidate, updated):
        torch.testing.assert_close(actual, 0.75 * old + 0.25 * target)
    # Convex natural interpolation remains a valid MNW posterior.
    recovered = BPC.natural_to_mnw(updated)
    torch.linalg.cholesky(recovered.V)
    torch.linalg.cholesky(recovered.Psi)


def test_discounted_conjugate_update_endpoints_and_spd():
    torch.manual_seed(11)
    prior_params = BPC.MNWParameters(
        torch.randn(2, 3, dtype=torch.float64),
        random_spd(3),
        random_spd(2),
        torch.tensor(4.0, dtype=torch.float64),
    )
    prior = BPC.mnw_to_natural(prior_params)
    old_stats = BPC.sufficient_statistics(torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64))
    new_stats = BPC.sufficient_statistics(torch.randn(7, 3, dtype=torch.float64), torch.randn(7, 2, dtype=torch.float64))
    current = BPC.add_statistics(prior, old_stats)
    rho_zero = BPC.discounted_conjugate_update(prior, current, new_stats, 0.0)
    rho_one = BPC.discounted_conjugate_update(prior, current, new_stats, 1.0)
    for actual, expected in zip(rho_zero, BPC.conjugate_candidate(prior, new_stats)):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(rho_one, BPC.add_statistics(current, new_stats)):
        torch.testing.assert_close(actual, expected)
    middle = BPC.natural_to_mnw(BPC.discounted_conjugate_update(prior, current, new_stats, 0.99))
    torch.linalg.cholesky(middle.V)
    torch.linalg.cholesky(middle.Psi)


def test_discounted_rl_default_anchors_to_random_initial_mean():
    torch.manual_seed(23)
    args = small_args(prior_mean_mode="initial")
    edge = BPC.MatrixNormalWishartEdge(3, 4, args, first_edge=True)
    initial = edge.posterior()
    prior = BPC.natural_to_mnw(edge._natural("prior"), edge.jitter)
    assert initial.M.abs().max().item() > 0.0
    torch.testing.assert_close(prior.M, initial.M)

    x, y = torch.randn(6, 3), torch.randn(6, 4)
    features = edge.features(x)
    expected = BPC.conjugate_candidate(edge._natural("prior"), BPC.sufficient_statistics(features.double(), y.double()))
    candidate = edge.posterior_candidate(x, y, None, "discounted", 0.1, 0.0)
    for expected_value, actual_value in zip(expected, candidate):
        torch.testing.assert_close(actual_value, expected_value)

    paper_edge = BPC.MatrixNormalWishartEdge(3, 4, small_args(prior_mean_mode="zero"), first_edge=True)
    paper_prior = BPC.natural_to_mnw(paper_edge._natural("prior"), paper_edge.jitter)
    torch.testing.assert_close(paper_prior.M, torch.zeros_like(paper_prior.M), atol=1e-12, rtol=0.0)
    assert paper_edge.posterior().M.abs().max().item() > 0.0


def set_edge_posterior(edge, parameters):
    natural = BPC.mnw_to_natural(parameters, jitter=1e-12)
    with torch.no_grad():
        for name, value in zip(BPC.NaturalParameters._fields, natural):
            getattr(edge, f"natural_{name}").copy_(value)
        edge._refresh_cache()


def test_finite_difference_free_energy_gradient_includes_nonzero_dv_term():
    torch.manual_seed(2)
    edge = BPC.MatrixNormalWishartEdge(3, 2, small_args(), first_edge=False)
    M = torch.tensor(
        [[0.4, -0.2, 0.3, 0.1], [-0.1, 0.5, 0.2, -0.2]], dtype=torch.float64
    )
    # Off-diagonal feature/bias covariance makes the uncertainty correction
    # nontrivial and catches omission or transposition of d_y D V x.
    V = torch.tensor(
        [[0.8, 0.1, -0.2, 0.15], [0.1, 0.7, 0.05, -0.1], [-0.2, 0.05, 0.9, 0.2], [0.15, -0.1, 0.2, 0.6]],
        dtype=torch.float64,
    )
    Psi = torch.tensor([[0.6, 0.12], [0.12, 0.5]], dtype=torch.float64)
    set_edge_posterior(edge, BPC.MNWParameters(M, V, Psi, torch.tensor(5.0, dtype=torch.float64)))

    previous = torch.tensor([[0.2, -0.35, 0.6], [-0.4, 0.1, 0.25]])
    state = torch.tensor([[0.7, -0.2], [0.15, 0.45]])
    analytic_gradient = edge.previous_state_gradient(previous, state)
    epsilon = 1e-3
    finite_difference = torch.empty_like(previous)
    for row in range(previous.shape[0]):
        for column in range(previous.shape[1]):
            plus, minus = previous.clone(), previous.clone()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            finite_difference[row, column] = (
                edge.expected_energy(plus, state).sum() - edge.expected_energy(minus, state).sum()
            ) / (2.0 * epsilon)
    torch.testing.assert_close(analytic_gradient, finite_difference, rtol=3e-3, atol=3e-4)

    x = edge.features(previous)
    residual = state - x @ edge.cached_M.T
    mean_only = (-(residual @ edge.cached_precision) @ edge.cached_M[:, :3]) * edge.feature_derivative(previous)
    correction = analytic_gradient - mean_only
    assert correction.abs().max().item() > 0.05


def test_latent_gradient_is_per_example_and_batch_invariant():
    torch.manual_seed(3)
    edge = BPC.MatrixNormalWishartEdge(3, 4, small_args(), first_edge=False)
    previous = torch.randn(1, 3)
    state = torch.randn(1, 4)
    reference = edge.previous_state_gradient(previous, state)
    # Other batch rows cannot rescale or otherwise change row zero's inference.
    batched_previous = torch.cat((previous, torch.randn(5, 3)), dim=0)
    batched_state = torch.cat((state, torch.randn(5, 4)), dim=0)
    actual = edge.previous_state_gradient(batched_previous, batched_state)[:1]
    torch.testing.assert_close(actual, reference)


def test_exact_sums_count_repeated_evidence_and_tempering_is_optional():
    torch.manual_seed(4)
    args = small_args()
    base = BPC.MatrixNormalWishartEdge(2, 3, args, first_edge=True)
    exact_once = BPC.MatrixNormalWishartEdge(2, 3, args, first_edge=True)
    exact_twice = BPC.MatrixNormalWishartEdge(2, 3, args, first_edge=True)
    exact_once.load_state_dict(base.state_dict())
    exact_twice.load_state_dict(base.state_dict())
    x, y = torch.randn(4, 2), torch.randn(4, 3)
    exact_once.update_from_activity(x, y, None, "paper_svi", 1.0, 0.99)
    exact_twice.update_from_activity(x.repeat(2, 1), y.repeat(2, 1), None, "paper_svi", 1.0, 0.99)
    # Exact BPC legitimately regards duplicate rows as twice the evidence.
    assert exact_twice.natural_xi - base.prior_xi == 2 * (exact_once.natural_xi - base.prior_xi)

    tempered_once = BPC.MatrixNormalWishartEdge(2, 3, args, first_edge=True)
    tempered_twice = BPC.MatrixNormalWishartEdge(2, 3, args, first_edge=True)
    tempered_once.load_state_dict(base.state_dict())
    tempered_twice.load_state_dict(base.state_dict())
    tempered_once.update_from_activity(x, y, 16.0, "paper_svi", 1.0, 0.99)
    tempered_twice.update_from_activity(x.repeat(2, 1), y.repeat(2, 1), 16.0, "paper_svi", 1.0, 0.99)
    for name in BPC.NaturalParameters._fields:
        torch.testing.assert_close(getattr(tempered_once, f"natural_{name}"), getattr(tempered_twice, f"natural_{name}"))


def test_trace_adds_current_eligibility_before_td_and_resets_afterward():
    trace = BPC.EligibilityTrace(2, 2, torch.device("cpu"))
    trace.value.copy_(torch.tensor([[2.0, -1.0], [1.0, 3.0]]))
    instantaneous = torch.tensor([[1.0, 4.0], [-2.0, 1.0]])
    trace.accumulate(instantaneous, 0.5)
    expected_trace = torch.tensor([[2.0, 3.5], [-1.5, 2.5]])
    torch.testing.assert_close(trace.value, expected_trace)
    td = torch.tensor([3.0, -2.0])
    expected_direction = (td[:, None] * expected_trace).mean(0)
    torch.testing.assert_close(trace.modulated_mean(td), expected_direction)
    trace.reset(torch.tensor([True, False]))
    torch.testing.assert_close(trace.value[0], torch.zeros(2))
    torch.testing.assert_close(trace.value[1], expected_trace[1])


def test_reward_sign_enters_terminal_energy_once_not_posterior_statistics():
    torch.manual_seed(5)
    head = BPC.BetaHead(4, 2)
    top = torch.randn(3, 4)
    action = torch.rand(3, 2).clamp(0.1, 0.9)
    td = torch.tensor([0.5, -1.0, 2.0])
    positive = BPC.actor_terminal_gradient(head, top, action, td)
    negative = BPC.actor_terminal_gradient(head, top, action, -td)
    torch.testing.assert_close(negative, -positive)
    zero = BPC.actor_terminal_gradient(head, top, action, torch.zeros_like(td), coefficient=10.0)
    amplified = BPC.actor_terminal_gradient(head, top, action, td, coefficient=10.0)
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    torch.testing.assert_close(amplified, 10.0 * positive, rtol=2e-5, atol=2e-6)

    # The conjugate API has no reward/TD argument: once settling has produced the
    # same activities, posterior statistics cannot acquire another sign or delta.
    assert "delta" not in BPC.MatrixNormalWishartEdge.update_from_activity.__code__.co_varnames


def test_weights_are_outside_nonlinearity():
    edge = BPC.MatrixNormalWishartEdge(2, 2, small_args(), first_edge=False)
    state = torch.tensor([[0.3, -0.8]])
    expected = edge.features(state) @ edge.cached_M.T
    torch.testing.assert_close(edge.predict(state), expected)


def test_actor_posterior_move_is_limited_by_final_deployed_policy_kl():
    torch.manual_seed(12)
    args = small_args(num_hidden_layers=1, max_update_kl=1e-7, kl_bisection_steps=18)
    stack = BPC.BPCStack(3, args)
    head = BPC.BetaHead(4, 2)
    observation = torch.randn(8, 3)
    behavior_state = stack.forward_states(observation)[-1]
    behavior_alpha, behavior_beta = head(behavior_state)
    strongly_settled = [100.0 * torch.randn(8, 4)]

    combined_kl, posterior_kl, scale, limited, proposal_kl = BPC.apply_actor_posterior_with_kl_limit(
        stack,
        head,
        observation,
        behavior_alpha,
        behavior_beta,
        lambda: stack.update_posteriors(
            observation,
            strongly_settled,
            None,
            "discounted",
            0.1,
            0.99,
        ),
        args.max_update_kl,
        args.posterior_guard_trials,
    )
    assert limited
    assert 0.0 <= scale < 1.0
    assert combined_kl.item() <= args.max_update_kl * 1.001
    assert proposal_kl.item() > args.max_update_kl
    torch.testing.assert_close(posterior_kl, combined_kl, rtol=2e-3, atol=1e-10)


def test_critic_posterior_move_is_limited_on_current_and_bootstrap_observations():
    torch.manual_seed(29)
    args = small_args(num_hidden_layers=2)
    stack = BPC.BPCStack(3, args)
    head = BPC.ValueHead(4)
    observations = torch.randn(16, 3)
    training_observations = observations[:8]
    strongly_settled = [50.0 * torch.randn(8, 4), 50.0 * torch.randn(8, 4)]
    rms, maximum, scale, limited, _, proposal_rms, proposal_max = BPC.apply_critic_posterior_with_value_limit(
        stack,
        head,
        observations,
        lambda: stack.update_posteriors(training_observations, strongly_settled, None, "discounted", 0.1, 0.99),
        rms_limit=1e-3,
        max_abs_limit=2e-3,
        corrective_trials=2,
    )
    assert limited
    assert 0.0 <= scale < 1.0
    assert rms.item() <= 1.001e-3
    assert maximum.item() <= 2.002e-3
    assert proposal_rms.item() >= rms.item()
    assert proposal_max.item() >= maximum.item()


def test_long_discounted_update_sequence_preserves_spd_and_finite_cache():
    torch.manual_seed(31)
    edge = BPC.MatrixNormalWishartEdge(4, 4, small_args(prior_mean_mode="initial"), first_edge=False)
    for step in range(512):
        previous = torch.randn(16, 4)
        state = torch.randn(16, 4)
        edge.update_from_activity(previous, state, None, "discounted", 0.1, 0.99)
        if step % 32 == 0 or step == 511:
            posterior = edge.posterior()
            torch.linalg.cholesky(posterior.V)
            torch.linalg.cholesky(posterior.Psi)
            assert torch.isfinite(edge.cached_M).all()
            assert torch.isfinite(edge.cached_V).all()
            assert torch.isfinite(edge.cached_precision).all()


def test_rl_prior_defaults_have_scale_matched_precision_and_uncertainty():
    args = BPC.Args(hidden_size=64, num_hidden_layers=1, compile=False)
    edge = BPC.MatrixNormalWishartEdge(64, 64, args, first_edge=False)
    posterior = edge.posterior()
    torch.testing.assert_close(
        posterior.nu * posterior.Psi,
        torch.eye(64, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    assert posterior.V.diag().mean().item() == pytest.approx(1e-3)
    # Paper settings remain directly expressible through Args overrides.
    paper = BPC.MatrixNormalWishartEdge(
        2,
        3,
        small_args(prior_column_covariance=10.0, prior_wishart_scale=1000.0),
        first_edge=True,
    ).posterior()
    assert paper.V.diag().mean().item() == pytest.approx(10.0)
    assert (paper.nu * paper.Psi).diag().mean().item() == pytest.approx(5000.0)

import importlib.util
import inspect
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_noe_fullsuffix_pgvec_v1.py"
)
SPEC = importlib.util.spec_from_file_location("sf_noe_fullsuffix_pgvec_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_suffix_residual_uses_every_available_step_and_cuts_resets():
    delta = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1)
    continuation = torch.tensor([1.0, 0.0, 1.0, 1.0]).reshape(4, 1)
    actual = MODULE.full_suffix_vector_residual(
        delta, continuation, gamma=0.5
    )
    expected = torch.tensor([2.0, 2.0, 5.0, 4.0]).reshape(4, 1, 1)
    torch.testing.assert_close(actual, expected)


def test_sf_credit_bootstraps_truncation_but_not_termination_and_cuts_both():
    gamma = 0.5
    phi = torch.ones(3, 2, 1)
    psi = torch.full_like(phi, 10.0)
    psi_next = torch.tensor(
        [[[7.0], [11.0]], [[13.0], [17.0]], [[19.0], [23.0]]]
    )
    terminations = torch.tensor(
        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    boundaries = torch.tensor(
        [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    transition_valids = torch.ones_like(boundaries)

    _, target = MODULE.full_suffix_sf_credit(
        phi,
        psi,
        psi_next,
        terminations,
        transition_valids,
        boundaries,
        gamma,
    )

    # At t=0 both environments cut the suffix. The truncation bootstraps its
    # final observation; the true termination does not.
    torch.testing.assert_close(target[0, 0], torch.tensor([4.5]))
    torch.testing.assert_close(target[0, 1], torch.tensor([1.0]))


def test_vector_plus_probe_residual_projects_to_exact_reward_suffix():
    torch.manual_seed(1)
    time, environments, dimensions = 7, 3, 4
    gamma = 0.93
    phi = torch.randn(time, environments, dimensions)
    psi = torch.randn(time, environments, dimensions)
    psi_next = torch.randn_like(psi)
    weight = torch.randn(dimensions)
    reward_error = torch.randn(time, environments)
    reward = phi @ weight + reward_error
    boundary = torch.zeros(time, environments)
    boundary[3, 1] = 1.0
    continuation = 1.0 - boundary
    bootstrap = torch.ones_like(boundary)
    delta_sf = phi + gamma * bootstrap.unsqueeze(-1) * psi_next - psi
    sf_residual = MODULE.full_suffix_vector_residual(
        delta_sf, continuation, gamma
    )
    residual_trace = MODULE.full_suffix_vector_residual(
        reward_error.unsqueeze(-1), continuation, gamma
    ).squeeze(-1)
    vector, projected = MODULE.project_policy_credit(
        sf_residual, residual_trace, weight
    )

    scalar_delta = reward + gamma * (psi_next @ weight) - psi @ weight
    expected = MODULE.full_suffix_vector_residual(
        scalar_delta.unsqueeze(-1), continuation, gamma
    ).squeeze(-1)
    assert vector.shape[-1] == dimensions + 1
    torch.testing.assert_close(projected, expected, atol=2e-5, rtol=2e-5)


def test_pgvec_redistribution_is_zero_sum_across_actuators():
    torch.manual_seed(2)
    logits = torch.randn(31, 5)
    standardized_action = torch.randn_like(logits)
    rho = MODULE.pgvec_rho(logits, standardized_action)
    torch.testing.assert_close(
        rho.sum(-1), torch.zeros(rho.shape[0]), atol=1e-6, rtol=1e-6
    )


def test_zero_router_recovers_joint_policy_gradient_at_behavior_policy():
    torch.manual_seed(4)
    logratio_dim = torch.zeros(13, 4, requires_grad=True)
    advantage = torch.randn(13)
    redistribution = torch.zeros_like(logratio_dim)
    routed_loss = MODULE.per_actuator_ppo_loss(
        logratio_dim,
        advantage,
        redistribution,
        clip_low=0.2,
        clip_high=0.28,
    )
    routed_gradient = torch.autograd.grad(
        routed_loss, logratio_dim, retain_graph=True
    )[0]
    joint_loss = -(advantage * logratio_dim.sum(-1).exp()).mean()
    joint_gradient = torch.autograd.grad(joint_loss, logratio_dim)[0]
    torch.testing.assert_close(routed_gradient, joint_gradient)


def test_marginal_clip_bounds_use_fisher_sqrt_dimension_scaling():
    low, high = MODULE.per_actuator_clip_bounds(0.2, 0.28, action_dim=4)
    assert low == 0.8**0.5
    assert high == 1.28**0.5


def test_cross_fitted_router_never_reads_its_destination_targets():
    torch.manual_seed(3)
    time, environments, dimensions, actions = 96, 8, 3, 2
    residual = torch.randn(time, environments, dimensions)
    standardized_action = torch.randn(time, environments, actions)
    coefficients = torch.randn(actions, dimensions)
    target = (
        (residual.unsqueeze(2) * coefficients.view(1, 1, actions, dimensions))
        .sum(-1)
        .mul(standardized_action)
        .sum(-1)
    )
    valid = torch.ones(time, environments, dtype=torch.bool)
    (
        rho,
        reliability,
        predictive_skill,
        split_agreement,
        _,
        _,
        active_folds,
    ) = MODULE.cross_fitted_pgvec(
        residual,
        standardized_action,
        target,
        valid,
        ridge=1e-5,
        min_fold_samples=32,
    )
    changed_target = target.clone()
    changed_target[:, ::2] += 100.0 * torch.randn_like(changed_target[:, ::2])
    changed_rho, _, _, _, _, _, _ = MODULE.cross_fitted_pgvec(
        residual,
        standardized_action,
        changed_target,
        valid,
        ridge=1e-5,
        min_fold_samples=32,
    )

    assert active_folds == 2
    assert reliability > 0.9
    assert predictive_skill > 0.9
    assert split_agreement > 0.9
    torch.testing.assert_close(rho[:, ::2], changed_rho[:, ::2])
    torch.testing.assert_close(
        rho.sum(-1), torch.zeros(time, environments), atol=2e-6, rtol=2e-6
    )


def test_actor_credit_standardization_is_functionally_independent_of_diagnostic_gae():
    policy_credit = torch.tensor([-3.0, 1.0, 2.0, 8.0])
    diagnostic_gae_a = torch.randn(4)
    diagnostic_gae_b = diagnostic_gae_a * -1000.0
    actual_a = MODULE.standardize_policy_credit(policy_credit)
    actual_b = MODULE.standardize_policy_credit(policy_credit)
    torch.testing.assert_close(actual_a, actual_b)
    assert not torch.equal(diagnostic_gae_a, diagnostic_gae_b)
    assert tuple(
        inspect.signature(MODULE.standardize_policy_credit).parameters
    ) == ("policy_credit",)


def test_actor_and_joint_control_share_full_suffix_credit_dataflow():
    source = SCRIPT.read_text()
    assert "pgvec: bool = True" in source
    assert "b_advantages = policy_adv.reshape(-1)" in source
    assert "b_policy_adv_normed = standardize_policy_credit(b_policy_adv)" in source
    assert "if args.pgvec:" in source
    assert "pg_loss = joint_ppo_loss(" in source
    assert "policy_adv = diagnostic_gae" not in source
    assert "last_sf = delta_sf + args.gamma * args.gae_lambda" not in source


def test_reward_covector_is_hard_lagged_until_targets_are_frozen():
    source = SCRIPT.read_text()
    lag = source.index("w_r_lagged = w_r")
    projection = source.index(
        "policy_vector_residual, policy_adv = project_policy_credit("
    )
    router = source.index(") = cross_fitted_pgvec(")
    solve_next = source.index(
        "w_r_next = solve_reward_probe(flat_phi, flat_rew, args.sf_ridge)"
    )
    install_next = source.index("w_r = w_r_next")
    assert lag < projection < router < solve_next < install_next
    assert "reward_resid = flat_rew - flat_phi @ w_r_lagged" in source
    assert "mc_ret - values" in source

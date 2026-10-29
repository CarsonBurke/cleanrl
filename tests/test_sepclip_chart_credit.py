"""Independent actual-shape chart contracts; CUDA execution only through mlq."""
import json

import pytest
import torch

from cleanrl.shared.runtime import configure_runtime
from scripts.audit_sepclip_chart_credit import actual_geometry, analyze_row, serial, transform_credit


@pytest.fixture(autouse=True)
def cuda_runtime():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    assert torch.cuda.is_available(), "Submit contracts through mlq; never CPU fallback"


def reference_fisher(alpha, beta):
    matrices = []
    for aa, bb in zip(alpha, beta):
        shapes = torch.stack((aa, bb)).detach().requires_grad_()
        matrices.append(torch.autograd.functional.hessian(
            lambda s: torch.lgamma(s[0])+torch.lgamma(s[1])-torch.lgamma(s.sum()), shapes))
    return torch.stack(matrices)


@pytest.mark.parametrize("alpha,beta", [(3., 2.), (1.0000001192092896, 20.), (100000., 2.), (1.5, 10000.)])
def test_chart_score_and_shape_jacobian_have_independent_references(alpha, beta):
    a = torch.tensor([alpha], dtype=torch.float64, device="cuda")
    b = torch.tensor([beta], dtype=torch.float64, device="cuda")
    geometry = actual_geometry(a, b)
    independent_fisher = reference_fisher(a, b)
    torch.testing.assert_close(geometry["shape_fisher"], independent_fisher, rtol=1e-12, atol=1e-15)
    coordinates = torch.stack((geometry["r"], geometry["ell"]), -1).detach().requires_grad_()
    def shapes(c):
        tau = c[..., 1].exp()
        q = c[..., 0].sigmoid()
        return 1+tau*q, 1+tau*(1-q)
    aa, bb = shapes(coordinates)
    native = torch.tensor([.37], dtype=torch.float64, device="cuda")
    # Direct density autograd never imports an old oracle or uses its chain.
    density = torch.distributions.Beta(aa, bb).log_prob(native).sum()
    direct = torch.autograd.grad(density, coordinates)[0]
    sa = native.log()-torch.digamma(a)+torch.digamma(a+b)
    sb = torch.log1p(-native)-torch.digamma(b)+torch.digamma(a+b)
    shape_score = torch.stack((sa, sb), -1)
    recorded_sigmoid = torch.tensor([[.21, .73]], dtype=torch.float64, device="cuda")
    converted = transform_credit(shape_score*recorded_sigmoid, recorded_sigmoid, geometry)
    torch.testing.assert_close(converted["chart_pg"], direct, rtol=2e-8, atol=2e-9)
    # Finite differences of independently defined shape coordinates.
    for h in (.005, .001):
        columns = []
        for i in range(2):
            shift = torch.zeros_like(coordinates); shift[..., i] = h
            plus = torch.stack(shapes(coordinates.detach()+shift), -1)
            minus = torch.stack(shapes(coordinates.detach()-shift), -1)
            columns.append((plus-minus)/(2*h))
        difference = torch.stack(columns, -1)
        torch.testing.assert_close(difference, geometry["jacobian"], rtol=3e-5, atol=3e-12)
    assert float(converted["duality_absolute_error"].max()) <= 1e-7*max(1., float(converted["shape_dual_energy"].max()))
    assert float(converted["natural_shape_direction_relative_error"].max()) < 1e-7


def test_fisher_quadratic_and_covector_pairing_are_invariant_with_cross_terms():
    a = torch.tensor([3., 1.0000001192092896, 100000.], dtype=torch.float64, device="cuda")
    b = torch.tensor([2., 20., 2.], dtype=torch.float64, device="cuda")
    geometry = actual_geometry(a, b)
    independent_fisher = reference_fisher(a, b)
    delta = torch.tensor([[.02, -.04], [.03, .01], [-.01, .03]], dtype=torch.float64, device="cuda")
    shape_delta = (geometry["jacobian"] @ delta.unsqueeze(-1)).squeeze(-1)
    chart_cost = torch.einsum("ni,nij,nj->n", delta, geometry["chart_fisher"], delta)
    shape_cost = torch.einsum("ni,nij,nj->n", shape_delta, independent_fisher, shape_delta)
    torch.testing.assert_close(chart_cost, shape_cost, rtol=2e-10, atol=1e-15)
    assert bool((geometry["chart_fisher"][:, 0, 1].abs() > 0).all())
    covector = torch.tensor([[.3, -.2], [.1, .5], [-.04, .3]], dtype=torch.float64, device="cuda")
    pulled = (geometry["jacobian"].mT @ covector.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close((pulled*delta).sum(-1), (covector*shape_delta).sum(-1), rtol=1e-12, atol=1e-12)


def test_shape_one_is_withheld_and_latent_precision_is_not_silently_clamped():
    row = dict(snapshot="fixture", snapshot_sha256="fixture-hash", global_step=7,
        selected_states=[dict(t=0, row=0, episode_age=0)],
        initial_policy=dict(logits=[[-30., 40., -30., -2.]], alpha=[[1., 41.]], beta=[[1., 1.126928]],
        ), oracle=dict(first_action_only=True, horizons=[dict(horizon=512, policy_logit_pg={"finite_raw": dict(
            mean=[[.1, .2, -.1, .3]], replicate_values=[[[.2, .3, -.2, .4]], [[0., .1, 0., .2]]])})]))
    result = analyze_row(row)
    assert result["usable_chart_mask"].tolist() == [[False, True]]
    assert result["latent_pre_add_fp32_softplus_excess"][0, 0, 0] > 0
    assert result["actual_rounded_excess"][0, 0, 0] == 0
    encoded = serial(result)
    assert encoded["geometry"]["q"][0][0] is None
    assert encoded["geometry"]["q"][0][1] > .99
    assert encoded["actual_shape_chart_r_ell"][0][0] == [None, None]
    assert encoded["latent_minus_actual_chart_r_ell"][0][0] == [None, None]
    assert encoded["latent_minus_actual_relative_excess_on_positive_actual"][0][0][0] is None
    json.dumps(encoded, allow_nan=False)
    # An extreme valid excess is preserved rather than capped at a hidden grid.
    geometry = actual_geometry(torch.tensor([1e12], device="cuda", dtype=torch.float64),
                               torch.tensor([2.], device="cuda", dtype=torch.float64))
    assert float(geometry["tau"][0]) > 1e11
    with pytest.raises(ValueError, match="no boundary clamp"):
        actual_geometry(torch.ones(1, device="cuda", dtype=torch.float64),
                        torch.tensor([2.], device="cuda", dtype=torch.float64))


def test_extreme_fisher_precision_failure_is_reported_without_regularization():
    row = dict(initial_policy=dict(logits=[[1e30, .5413248546]], alpha=[[1e30]], beta=[[2.]]),
        selected_states=[dict(t=0, row=0, episode_age=0)],
        oracle=dict(first_action_only=True, horizons=[dict(horizon=1, policy_logit_pg={"finite_raw": dict(
            mean=[[.1, .2]], replicate_values=[[[.2, .3]], [[0., .1]]])})]))
    result = analyze_row(row)
    assert result["actual_shape_interior"].tolist() == [[True]]
    assert result["usable_chart_mask"].tolist() == [[False]]
    assert result["numerical_fisher_withheld_coordinates"].tolist() == [[0, 0]]
    assert "withheld_reason" in result["horizons"][0]["fields"]["finite_raw"]


def test_missing_scope_and_state_identity_metadata_are_rejected():
    row = dict(initial_policy=dict(logits=[[1., 1.]], alpha=[[2.313261747]], beta=[[2.313261747]]),
        selected_states=[dict(t=0, row=0, episode_age=0)], oracle=dict(horizons=[]))
    with pytest.raises(ValueError, match="explicit first_action_only"):
        analyze_row(row)
    row["oracle"]["first_action_only"] = True
    row["selected_states"] = []
    with pytest.raises(ValueError, match="Selected state count"):
        analyze_row(row)
    row["selected_states"] = [dict(t=0, row=0, episode_age=0)]
    row["initial_policy"]["logits"] = [[float("inf"), 1.]]
    with pytest.raises(ValueError, match="Nonfinite recorded"):
        analyze_row(row)


def test_partial_withholding_keeps_all_state_denominator_and_rejects_false_mean():
    row = dict(initial_policy=dict(logits=[[-30., -30.], [1.854586542, .5413248546]],
        alpha=[[1.], [3.]], beta=[[1.], [2.]]),
        selected_states=[dict(t=i, row=0, episode_age=i) for i in range(2)],
        oracle=dict(first_action_only=True, horizons=[dict(horizon=64, policy_logit_pg={
            "finite_raw": dict(mean=[[.1, .1], [.2, .3]],
                replicate_values=[[[.1, .1], [.2, .3]], [[.1, .1], [.2, .3]]])})]))
    result = analyze_row(row)
    field = result["horizons"][0]["fields"]["finite_raw"]
    assert field["usable_state_action_count"] == 1
    assert field["total_state_action_count"] == 2
    assert field["mean_field_dual_energy_on_usable_coordinates"] == pytest.approx(
        float(field["mean_field"]["chart_dual_energy"][1, 0])/2)
    row["oracle"]["horizons"][0]["policy_logit_pg"]["finite_raw"]["mean"][1][0] = .25
    with pytest.raises(ValueError, match="mean does not agree"):
        analyze_row(row)
    row["oracle"]["horizons"][0]["policy_logit_pg"]["finite_raw"]["mean"][1][0] = .2
    row["initial_policy"]["alpha"][1][0] = 100.
    with pytest.raises(ValueError, match="Recorded shapes disagree"):
        analyze_row(row)


def test_near_shape_one_rounding_has_visible_relative_and_chart_discrepancies():
    row = dict(initial_policy=dict(logits=[[-16.2, .5413248546]],
        alpha=[[1.0000001192092896]], beta=[[2.]]),
        selected_states=[dict(t=0, row=0, episode_age=0)],
        oracle=dict(first_action_only=True, horizons=[dict(horizon=1, policy_logit_pg={
            "finite_raw": dict(mean=[[.1, .2]], replicate_values=[[[.1, .2]], [[.1, .2]]])})]))
    result = analyze_row(row)
    assert result["usable_chart_mask"].tolist() == [[True]]
    assert abs(float(result["latent_minus_actual_relative_excess_on_positive_actual"][0, 0, 0])) > .1
    assert abs(float(result["latent_minus_actual_chart_r_ell"][0, 0, 0])) > .1


def test_accepted_ulp_triplet_with_logit_fisher_underflow_is_withheld():
    # The replay tolerance conservatively accepts this one-ULP discrepancy,
    # but its hybrid logit metric cannot be inverted in FP64. No ridge allowed.
    row = dict(initial_policy=dict(logits=[[-700., .5413248546]],
        alpha=[[1.0000001192092896]], beta=[[2.]]),
        selected_states=[dict(t=0, row=0, episode_age=0)],
        oracle=dict(first_action_only=True, horizons=[dict(horizon=1, policy_logit_pg={
            "finite_raw": dict(mean=[[.1, .2]], replicate_values=[[[.1, .2]], [[.1, .2]]])})]))
    result = analyze_row(row)
    assert result["actual_shape_interior"].tolist() == [[True]]
    assert result["usable_chart_mask"].tolist() == [[False]]
    assert result["numerical_smooth_logit_fisher_withheld_coordinates"].tolist() == [[0, 0]]
    assert "withheld_reason" in result["horizons"][0]["fields"]["finite_raw"]

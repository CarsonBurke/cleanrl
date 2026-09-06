"""Behavioral checks for histogram geometry and lambda-return bias diagnostics."""

import math
from dataclasses import replace

import pytest

from cleanrl.shared.hl_gauss import HLGaussConfig
from scripts.hlgauss.diagnostics import analyze_support, lambda_return_fixed_point


def test_uniform_raw_interior_preserves_mean_and_lambda_fixed_point():
    config = HLGaussConfig(v_min=-10, v_max=10, num_bins=201, sigma_ratio=1.5, bin_type="centers")
    result = analyze_support(config, target_mean=0.37, target_std=0.25)

    # A Gaussian spanning 1.5 uniform cells has negligible periodic rounding
    # error here, and the support boundaries are more than 60 sigmas away.
    assert result["project_decode_value"] == pytest.approx(0.37, abs=1e-12)
    assert result["project_decode_derivative"] == pytest.approx(1.0, abs=1e-8)
    assert result["lambda_fixed_point_value"] == pytest.approx(0.37, abs=1e-11)
    assert result["lambda_fixed_point_advantage_bias"] == pytest.approx(0.0, abs=1e-12)
    assert result["lambda_fixed_point_converged"] == 1.0
    assert result["lambda_fixed_point_residual"] < 1e-12


def test_lambda_fixed_point_matches_affine_solution_not_one_step_amplification():
    gamma, trace, true_value = 0.97, 0.8, 5.0
    slope, offset = 0.9, 0.2
    bootstrap = gamma * (1 - trace) / (1 - gamma * trace)
    expected = (slope * (1 - bootstrap) * true_value + offset) / (1 - slope * bootstrap)
    result = lambda_return_fixed_point(lambda target: slope * target + offset, true_value, gamma, trace)

    assert result["lambda_fixed_point_value"] == pytest.approx(expected, abs=1e-9)
    assert result["lambda_fixed_point_value_bias"] == pytest.approx(expected - true_value, abs=1e-9)
    assert result["lambda_fixed_point_advantage_bias"] == pytest.approx((1 - bootstrap) * (true_value - expected), abs=1e-10)
    assert result["lambda_fixed_point_one_step_advantage_bias"] == pytest.approx(
        (1 - gamma) * (true_value - expected), abs=1e-10
    )
    one_step_solution = (slope * (1 - gamma) * true_value + offset) / (1 - slope * gamma)
    assert abs(result["lambda_fixed_point_value"] - one_step_solution) > 1.0
    assert result["lambda_fixed_point_converged"] == 1.0


def test_lambda_one_eliminates_bootstrap_and_nonconvergence_is_reported():
    direct = lambda_return_fixed_point(lambda target: 0.7 * target + 0.2, 4.0, gae_lambda=1.0)
    assert direct["lambda_fixed_point_value"] == pytest.approx(3.0)
    assert direct["lambda_fixed_point_advantage_bias"] == pytest.approx(1.0)

    unfinished = lambda_return_fixed_point(lambda target: target + 1.0, 4.0, max_iterations=1)
    assert unfinished["lambda_fixed_point_converged"] == 0.0
    assert unfinished["lambda_fixed_point_residual"] > 0.8


def test_broader_raw_gaussian_reduces_label_distinction():
    config = HLGaussConfig(v_min=-10, v_max=10, num_bins=201, sigma_ratio=1.0, bin_type="centers")
    narrow = analyze_support(config, target_mean=0.37, target_std=0.1)
    broad = analyze_support(replace(config, sigma_ratio=3.0), target_mean=0.37, target_std=0.1)

    assert broad["label_contrast_kl"] < narrow["label_contrast_kl"] / 4
    assert broad["ce_label_gradient_contrast_norm"] < narrow["ce_label_gradient_contrast_norm"]
    assert broad["target_location_fisher_information"] < narrow["target_location_fisher_information"] / 4
    assert broad["label_entropy"] > narrow["label_entropy"]


def test_fisher_matches_independent_gaussian_interval_score():
    config = HLGaussConfig(v_min=-10, v_max=10, num_bins=201, sigma_ratio=1.5, bin_type="centers")
    mean, sigma, width = 0.37, 0.15, 0.1
    expected = 0.0
    for i in range(201):
        lower = (-10.05 + i * width - mean) / sigma
        upper = (-10.05 + (i + 1) * width - mean) / sigma
        # Independent normal integrals in Python double precision. Boundaries
        # are so far out that normalization and its derivative are 1 and 0.
        if lower >= 0:
            mass = 0.5 * (math.erfc(lower / math.sqrt(2)) - math.erfc(upper / math.sqrt(2)))
        else:
            mass = 0.5 * (math.erfc(-upper / math.sqrt(2)) - math.erfc(-lower / math.sqrt(2)))
        score_mass = (math.exp(-lower * lower / 2) - math.exp(-upper * upper / 2)) / (sigma * math.sqrt(2 * math.pi))
        if mass > 1e-290:
            expected += (score_mass / math.sqrt(mass)) ** 2
    result = analyze_support(config, mean, 0.25)
    assert result["target_location_fisher_information"] == pytest.approx(expected, rel=1e-7)


def test_symlog_coordinate_bandwidth_is_not_raw_bandwidth():
    config = HLGaussConfig(v_min=-100, v_max=100, num_bins=101, sigma_ratio=0.75, bin_type="centers", transform="symlog")
    result = analyze_support(config, target_mean=3.0, target_std=0.2)
    sigma_coordinate = 0.75 * 2 * math.log1p(100) / 100
    assert result["smoothing_sigma_raw_local"] == pytest.approx(4 * sigma_coordinate)
    assert result["effective_sigma_over_target_std"] == pytest.approx(20 * sigma_coordinate)
    assert result["smoothing_sigma_raw_plus"] == pytest.approx(4 * math.expm1(sigma_coordinate))
    assert result["smoothing_sigma_raw_minus"] == pytest.approx(-4 * math.expm1(-sigma_coordinate))
    assert result["smoothing_sigma_raw_plus"] > result["smoothing_sigma_raw_minus"]


def test_symexp_placement_keeps_raw_gaussian_sigma_constant():
    config = HLGaussConfig(v_min=-50, v_max=50, num_bins=31, sigma_ratio=0.75, bin_type="symexp_centers")
    middle = analyze_support(config, target_mean=0.0, target_std=0.5)
    positive = analyze_support(config, target_mean=10.0, target_std=0.5)
    assert positive["smoothing_sigma_raw_local"] == pytest.approx(middle["smoothing_sigma_raw_local"])
    assert positive["smoothing_sigma_raw_plus"] == pytest.approx(positive["smoothing_sigma_raw_minus"])
    assert positive["local_raw_grid_spacing"] > 5 * middle["local_raw_grid_spacing"]


def test_outside_support_reports_clipping_and_lost_label_information():
    config = HLGaussConfig(v_min=-10, v_max=10, num_bins=101, sigma_ratio=1.0, bin_type="centers")
    result = analyze_support(config, target_mean=21.0, target_std=0.25, gamma=0.0)
    assert result["target_overflow"] == 1.0
    assert result["contrast_overflow_fraction"] == 1.0
    assert result["project_decode_derivative"] == 0.0
    assert result["label_contrast_kl"] == 0.0
    assert result["ce_label_gradient_contrast_norm"] == 0.0
    assert result["target_location_fisher_information"] == 0.0
    assert result["lambda_fixed_point_overflow"] == 1.0
    assert result["lambda_fixed_point_iteration_overflow_fraction"] == 1.0
    assert result["lambda_fixed_point_advantage_bias"] > 11.0


@pytest.mark.parametrize("gamma, trace", [(1.0, 0.95), (-0.1, 0.95), (0.99, 1.1), (0.99, math.nan)])
def test_noncontractive_or_invalid_discount_parameters_are_rejected(gamma, trace):
    with pytest.raises(ValueError):
        lambda_return_fixed_point(lambda value: value, 1.0, gamma, trace)

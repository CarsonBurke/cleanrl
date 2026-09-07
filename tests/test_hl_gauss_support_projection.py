"""Numerical contracts for the experimental Gaussian barycentric projector."""

import math

import numpy as np
import pytest
import torch

from scripts.hlgauss.support_projection import MeanPreservingSupport


@pytest.mark.parametrize("geometry", ["uniform", "asinh"])
@pytest.mark.parametrize("smoothing", ["fixed", "local", "twohot"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "bounds,center,scale",
    [
        ((-2e-9, 7e-9), 1e-9, 3e-10),
        ((-13.0, 79.0), 3.25, 0.38),
        ((-1e6, 1e6), 0.0, 0.3),
        ((2**30, 2**30 + 1024.0), 2**30 + 128.0, 32.0),
    ],
)
def test_probability_and_raw_mean_contract(geometry, smoothing, dtype, bounds, center, scale):
    lo, hi = bounds
    head = MeanPreservingSupport(
        v_min=lo, v_max=hi, num_bins=41, center=center, scale=scale, geometry=geometry, smoothing=smoothing
    )
    fractions = torch.tensor([0.0, 1e-7, 0.017, 0.23, 0.51, 0.899, 1.0 - 1e-7, 1.0], dtype=torch.float64)
    targets = (lo + (hi - lo) * fractions).to(dtype)
    labels = head.project(targets)
    clipped = targets.double().clamp(lo, hi)
    eps = torch.finfo(dtype).eps
    assert labels.dtype == dtype
    assert labels.shape == (*targets.shape, 41)
    assert torch.isfinite(labels).all()
    assert (labels >= 0).all()
    torch.testing.assert_close(labels.double().sum(-1), torch.ones_like(clipped), atol=4 * eps, rtol=0)
    # Center before accumulating: a huge origin must not hide mean bias.
    mean_error = ((head.support - clipped.unsqueeze(-1)) * labels.double()).sum(-1)
    torch.testing.assert_close(mean_error, torch.zeros_like(clipped), atol=8 * eps * (hi - lo), rtol=0)
    torch.testing.assert_close(
        head.probs_to_scalar(labels).double(),
        clipped,
        atol=8 * eps * max(abs(lo), abs(hi), hi - lo),
        rtol=0,
    )
    assert head.support[0].item() == lo
    assert head.support[-1].item() == hi


@pytest.mark.parametrize("smoothing", ["fixed", "local", "twohot"])
def test_endpoints_clipping_and_tiny_radius_use_local_barycentric_limit(smoothing):
    head = MeanPreservingSupport(v_min=-2.0, v_max=11.0, num_bins=17, center=0.25, scale=0.2, smoothing=smoothing)
    targets = torch.tensor([-math.inf, -9.0, -2.0, 11.0, 16.0, math.inf], dtype=torch.float64)
    expected = torch.zeros(6, 17, dtype=torch.float64)
    expected[:3, 0] = 1
    expected[3:, -1] = 1
    torch.testing.assert_close(head.project(targets), expected, atol=0, rtol=0)

    near = torch.stack(
        [
            torch.nextafter(head.support[0], head.support[-1]),
            torch.nextafter(head.support[-1], head.support[0]),
        ]
    )
    labels = head.project(near)
    assert torch.isfinite(labels).all()
    assert (labels >= 0).all()
    # These tiny symmetric intervals intersect only the endpoint bracket.
    assert torch.count_nonzero(labels[0, 2:]) == 0
    assert torch.count_nonzero(labels[1, :-2]) == 0
    torch.testing.assert_close(labels.sum(-1), torch.ones(2, dtype=torch.float64), atol=2e-16, rtol=0)
    torch.testing.assert_close(head.probs_to_scalar(labels), near, atol=2e-15, rtol=0)


@pytest.mark.parametrize("geometry", ["uniform", "asinh"])
@pytest.mark.parametrize("smoothing", ["fixed", "local", "twohot"])
@pytest.mark.parametrize("factor,shift", [(37.0, 1024.0), (0.003, -7.0), (-2.0, 19.0)])
def test_affine_units_preserve_labels_and_decode(geometry, smoothing, factor, shift):
    config = dict(v_min=-8.0, v_max=21.0, num_bins=33, center=1.5, scale=0.7, geometry=geometry, smoothing=smoothing)
    head = MeanPreservingSupport(**config)
    transformed_bounds = sorted([factor * config["v_min"] + shift, factor * config["v_max"] + shift])
    transformed = MeanPreservingSupport(
        **{
            **config,
            "v_min": transformed_bounds[0],
            "v_max": transformed_bounds[1],
            "center": factor * config["center"] + shift,
            "scale": abs(factor) * config["scale"],
        }
    )
    targets = torch.tensor([-10.0, -7.99, -0.2, 1.13, 5.7, 20.95, 23.0], dtype=torch.float64)
    expected = head.project(targets)
    actual = transformed.project(factor * targets + shift)
    if factor < 0:
        actual = actual.flip(-1)
    torch.testing.assert_close(actual, expected, atol=4e-12, rtol=4e-11)
    torch.testing.assert_close(
        transformed.probs_to_scalar(transformed.project(factor * targets + shift)),
        factor * head.probs_to_scalar(expected) + shift,
        atol=5e-12,
        rtol=2e-14,
    )


def _quadrature_labels(support, target, sigma):
    """Independent deterministic Gauss-Legendre integration, no normal CDF."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    radius = min(target - support[0], support[-1] - target)
    result = np.zeros_like(support)
    for i, (lo, hi) in enumerate(zip(support[:-1], support[1:])):
        lower, upper = max(lo, target - radius), min(hi, target + radius)
        if upper <= lower:
            continue
        points = (lower + upper) / 2 + (upper - lower) / 2 * nodes
        density_weights = weights * np.exp(-0.5 * ((points - target) / sigma) ** 2) * (upper - lower) / 2
        fraction = (points - lo) / (hi - lo)
        result[i] += np.dot(density_weights, 1 - fraction)
        result[i + 1] += np.dot(density_weights, fraction)
    return torch.from_numpy(result / result.sum())


@pytest.mark.parametrize("geometry", ["uniform", "asinh"])
@pytest.mark.parametrize("smoothing", ["fixed", "local"])
def test_gaussian_barycentric_labels_match_independent_quadrature(geometry, smoothing):
    head = MeanPreservingSupport(
        v_min=-3.0, v_max=7.0, num_bins=13, center=0.4, scale=0.6, geometry=geometry, smoothing=smoothing
    )
    target = 1.137
    support = head.support.numpy()
    bracket = np.searchsorted(support, target, side="right")
    sigma = 1.5 * head.scale if smoothing == "fixed" else 0.75 * (support[bracket] - support[bracket - 1])
    expected = _quadrature_labels(support, target, sigma)
    torch.testing.assert_close(
        head.project(torch.tensor(target, dtype=torch.float64)), expected, atol=2e-14, rtol=3e-12
    )


@pytest.mark.parametrize("scale", [1e4, 1e300])
def test_extremely_broad_gaussian_retains_symmetric_uniform_limit(scale):
    head = MeanPreservingSupport(v_min=-2.0, v_max=3.0, num_bins=11, center=0.0, scale=scale, geometry="uniform")
    target = torch.tensor(0.25, dtype=torch.float64)
    labels = head.project(target)
    expected = _quadrature_labels(head.support.numpy(), target.item(), 1.5 * scale)
    torch.testing.assert_close(labels, expected, atol=2e-14, rtol=2e-13)
    torch.testing.assert_close(head.probs_to_scalar(labels), target, atol=2e-15, rtol=0)


def test_subnormal_support_retains_gaussian_smoothing_not_only_its_mean():
    unit = math.ulp(0.0)
    head = MeanPreservingSupport(v_min=0.0, v_max=2 * unit, num_bins=3, center=0.0, scale=1.0, geometry="uniform")
    labels = head.project(torch.tensor(unit, dtype=torch.float64))
    # The Gaussian is uniform at this scale; integrating the linear basis
    # must retain its spread, not collapse to the mean bucket.
    torch.testing.assert_close(labels, torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(head.probs_to_scalar(labels), torch.tensor(unit, dtype=torch.float64), rtol=0, atol=0)


def test_twohot_is_exact_interpolation_and_decode_uses_raw_expectation():
    head = MeanPreservingSupport(v_min=-5.0, v_max=9.0, num_bins=9, center=0.5, scale=0.3, smoothing="twohot")
    targets = 0.7 * head.support[:-1] + 0.3 * head.support[1:]
    labels = head.project(targets)
    expected = torch.zeros(8, 9, dtype=torch.float64)
    expected[torch.arange(8), torch.arange(8)] = 0.7
    expected[torch.arange(8), torch.arange(1, 9)] = 0.3
    torch.testing.assert_close(labels, expected, atol=2e-15, rtol=0)
    probabilities = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3], dtype=torch.float64)
    expected_value = 0.2 * head.support[0] + 0.5 * head.support[4] + 0.3 * head.support[-1]
    torch.testing.assert_close(head.probs_to_scalar(probabilities), expected_value)
    torch.testing.assert_close(head.to_scalar(probabilities.log()), expected_value)


@pytest.mark.parametrize("smoothing", ["fixed", "local", "twohot"])
def test_scalar_empty_and_multidimensional_shapes(smoothing):
    head = MeanPreservingSupport(v_min=-1.0, v_max=2.0, num_bins=7, center=0.0, scale=0.4, smoothing=smoothing)
    for shape in [(), (0,), (2, 0), (2, 3)]:
        targets = torch.zeros(shape, dtype=torch.float32)
        labels = head.project(targets)
        assert labels.shape == (*shape, 7)
        assert head.probs_to_scalar(labels).shape == shape


@pytest.mark.parametrize(
    "invalid",
    [
        {"num_bins": True},
        {"num_bins": 1},
        {"num_bins": 3.5},
        {"v_min": 3.0},
        {"v_max": math.inf},
        {"center": math.nan},
        {"scale": 0.0},
        {"scale": -1.0},
        {"scale": math.inf},
        {"geometry": "symlog"},
        {"smoothing": "histogram"},
    ],
)
def test_invalid_configuration_is_rejected(invalid):
    config = dict(v_min=-1.0, v_max=2.0, num_bins=7, center=0.0, scale=0.4)
    with pytest.raises(ValueError):
        MeanPreservingSupport(**{**config, **invalid})

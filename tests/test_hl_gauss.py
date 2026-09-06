"""CPU numerical contracts for the configurable Gaussian histogram head."""

import math
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from cleanrl.shared.hl_gauss import HLGaussConfig, HLGaussSupport


def config(**changes):
    return replace(
        HLGaussConfig(
            num_bins=51,
            v_min=-10.0,
            v_max=20.0,
            sigma_ratio=0.75,
            transform="linear",
            bin_type="centers",
            decode="scalar",
        ),
        **changes,
    )


def reference_projection(head, targets):
    """Independent Python float64 Gaussian integrals, including small tails."""
    cfg = head.config
    transform = lambda value: math.copysign(math.log1p(abs(value)), value) if cfg.transform == "symlog" else value
    width = (transform(cfg.v_max) - transform(cfg.v_min)) / (cfg.num_bins if cfg.bin_type == "edges" else cfg.num_bins - 1)
    scale = math.sqrt(2) * cfg.sigma_ratio * width
    edges = head.coord_edges.double().tolist()
    rows = []
    for value in targets.flatten().tolist():
        value = transform(min(max(value, cfg.v_min), cfg.v_max))
        masses = []
        for left, right in zip(edges[:-1], edges[1:]):
            lower, upper = (left - value) / scale, (right - value) / scale
            if lower >= 1:
                mass = 0.5 * (math.erfc(lower) - math.erfc(upper))
            elif upper <= -1:
                mass = 0.5 * (math.erfc(-upper) - math.erfc(-lower))
            else:
                mass = 0.5 * (math.erf(upper) - math.erf(lower))
            masses.append(mass)
        total = math.fsum(masses)
        rows.append([mass / total for mass in masses])
    return torch.tensor(rows, dtype=torch.float64).reshape(*targets.shape, cfg.num_bins)


@pytest.mark.parametrize(
    "changes",
    [
        {"num_bins": 1},
        {"num_bins": 2.5},
        {"num_bins": True},
        {"v_min": float("nan")},
        {"v_max": float("inf")},
        {"v_min": 20.0},
        {"v_min": 21.0},
        {"sigma_ratio": 0.0},
        {"sigma_ratio": -1.0},
        {"sigma_ratio": float("inf")},
        {"sigma_ratio": float("nan")},
        {"transform": "log"},
        {"bin_type": "buckets"},
        {"decode": "mean"},
    ],
)
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_configuration_is_explicit_and_immutable():
    with pytest.raises(TypeError):
        HLGaussConfig()
    cfg = config()
    with pytest.raises(FrozenInstanceError):
        cfg.decode = "transformed"


@pytest.mark.parametrize("transform", ["linear", "symlog"])
@pytest.mark.parametrize("bin_type", ["centers", "edges"])
def test_geometry_uses_raw_bounds_and_uniform_coordinate_bins(transform, bin_type):
    cfg = config(num_bins=7, transform=transform, bin_type=bin_type)
    head = cfg.build().double()
    lo, hi = cfg.v_min, cfg.v_max
    if transform == "symlog":
        lo, hi = -math.log1p(-lo), math.log1p(hi)
    width = (hi - lo) / (cfg.num_bins if bin_type == "edges" else cfg.num_bins - 1)
    expected_edges = torch.linspace(
        lo if bin_type == "edges" else lo - width / 2,
        hi if bin_type == "edges" else hi + width / 2,
        cfg.num_bins + 1,
        dtype=torch.float64,
    )
    torch.testing.assert_close(head.coord_edges, expected_edges, atol=2e-6, rtol=2e-7)
    torch.testing.assert_close(head.coord_support, (expected_edges[:-1] + expected_edges[1:]) / 2, atol=2e-6, rtol=2e-7)
    expected_support = (
        head.coord_support.sign() * head.coord_support.abs().expm1() if transform == "symlog" else head.coord_support
    )
    torch.testing.assert_close(head.support, expected_support, atol=3e-6, rtol=3e-7)


@pytest.mark.parametrize("transform", ["linear", "symlog"])
@pytest.mark.parametrize("bin_type", ["centers", "edges"])
def test_projection_matches_float64_gaussian_distribution_integrals(transform, bin_type):
    # Construct in float64 rather than promoting already rounded buffers.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        head = config(transform=transform, bin_type=bin_type).build()
    finally:
        torch.set_default_dtype(previous_dtype)
    targets = torch.tensor([[-30.0, -10.0, -9.7], [-0.1, 0.0, 0.1], [8.0, 20.0, 30.0]], dtype=torch.float64)
    actual = head.project(targets)
    expected = reference_projection(head, targets)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=3e-12)
    torch.testing.assert_close(actual.sum(-1), torch.ones_like(targets), atol=1e-15, rtol=0)
    assert (actual >= 0).all()


def test_float32_tail_mass_survives_cdf_cancellation():
    head = config(num_bins=21, v_min=-10.0, v_max=10.0, sigma_ratio=1.0).build()
    target = torch.tensor(0.0)
    actual = head.project(target)
    expected = reference_projection(head, target)
    # The interval [7.5, 8.5] has representable mass, but float32 erf CDF
    # subtraction would round both endpoints to one and erase it.
    assert actual[18] > 0
    torch.testing.assert_close(actual.double(), expected, rtol=5e-6, atol=1e-30)
    torch.testing.assert_close(actual, actual.flip(-1), atol=0, rtol=0)


@pytest.mark.parametrize("sigma_ratio", [1e-5, 1e8])
def test_extreme_smoothing_stays_normalized(sigma_ratio):
    head = config(num_bins=11, v_min=-5.0, v_max=5.0, sigma_ratio=sigma_ratio).build()
    targets = torch.tensor([-5.0, -0.5, 0.0, 4.0, 5.0])
    labels = head.project(targets)
    assert torch.isfinite(labels).all()
    torch.testing.assert_close(labels.sum(-1), torch.ones_like(targets))
    torch.testing.assert_close(labels.double(), reference_projection(head, targets), atol=3e-8, rtol=3e-6)


@pytest.mark.parametrize("bin_type", ["centers", "edges"])
def test_raw_clipping_and_boundary_smoothing_are_not_mean_matching(bin_type):
    head = config(transform="symlog", bin_type=bin_type).build()
    boundary = torch.tensor([head.config.v_min, head.config.v_max])
    expected = head.project(boundary)
    torch.testing.assert_close(head.project(torch.tensor([-torch.inf, torch.inf])), expected, atol=0, rtol=0)
    torch.testing.assert_close(head.project(boundary * 100), expected, atol=0, rtol=0)
    decoded = head.probs_to_scalar(expected)
    assert decoded[0] > boundary[0]
    assert decoded[1] < boundary[1]


def test_noisy_positive_mixture_distinguishes_scalar_and_transformed_estimands():
    scalar = config(num_bins=201, v_min=-100.0, v_max=100.0, transform="symlog").build().double()
    transformed = replace(scalar.config, decode="transformed").build().double()
    # Expected labels of a noisy conditional target: 1 with p=.75, 30 with p=.25.
    labels = scalar.project(torch.tensor([1.0, 30.0], dtype=torch.float64))
    mixture = 0.75 * labels[0] + 0.25 * labels[1]
    raw_expectation = scalar.probs_to_scalar(mixture)
    coordinate_expectation = transformed.probs_to_scalar(mixture)
    torch.testing.assert_close(raw_expectation, (mixture * scalar.support).sum())
    expected_coordinate = (mixture * transformed.coord_support).sum()
    torch.testing.assert_close(coordinate_expectation, expected_coordinate.expm1())
    assert raw_expectation - coordinate_expectation > 4.0
    # Scalar decoding commutes with mixing; nonlinear inverse decoding does not.
    torch.testing.assert_close(
        raw_expectation, 0.75 * scalar.probs_to_scalar(labels[0]) + 0.25 * scalar.probs_to_scalar(labels[1])
    )
    assert not torch.isclose(
        coordinate_expectation, 0.75 * transformed.probs_to_scalar(labels[0]) + 0.25 * transformed.probs_to_scalar(labels[1])
    )
    logits = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    torch.testing.assert_close(scalar.to_scalar(logits), raw_expectation)
    torch.testing.assert_close(transformed.to_scalar(logits), coordinate_expectation)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3)])
def test_projection_decode_and_loss_preserve_scalar_and_empty_shapes(shape):
    head = config().build()
    targets = torch.zeros(shape)
    labels = head.project(targets)
    assert labels.shape == (*shape, head.config.num_bins)
    assert head.probs_to_scalar(labels).shape == shape
    logits = torch.zeros((*shape, head.config.num_bins))
    assert head.to_scalar(logits).shape == shape
    assert head.loss(logits, targets, reduction="none").shape == shape
    if targets.numel() == 0:
        assert head.loss(logits, targets, reduction="sum") == 0
        assert torch.isnan(head.loss(logits, targets))


def test_loss_gradient_matches_categorical_cross_entropy_and_detaches_targets():
    head = config(num_bins=7).build().double()
    targets = torch.tensor([[0.2, 3.0], [-4.0, 18.0]], dtype=torch.float64, requires_grad=True).t()
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn((2, 2, 7), dtype=torch.float64, generator=generator, requires_grad=True)
    labels = head.project(targets.detach())
    loss = head.loss(logits, targets)
    gradient, target_gradient = torch.autograd.grad(loss, (logits, targets), allow_unused=True)
    torch.testing.assert_close(gradient, (logits.softmax(-1) - labels) / targets.numel())
    assert target_gradient is None
    per_target = head.loss(logits, targets, reduction="none")
    torch.testing.assert_close(loss, per_target.mean())
    torch.testing.assert_close(head.loss(logits, targets, reduction="sum"), per_target.sum())
    # A real descent step must improve the same fixed-label objective.
    assert head.loss(logits.detach() - gradient, targets.detach()) < loss.detach()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_module_dtype_and_state_roundtrip_preserve_predictions(dtype):
    head = config(transform="symlog").build().to(dtype=dtype)
    targets = torch.tensor([-4.0, 0.0, 8.0], dtype=dtype)
    labels = head.project(targets)
    assert labels.dtype == dtype
    assert labels.device == targets.device
    assert head.to_scalar(labels).dtype == dtype
    assert not list(head.parameters())
    restored = head.config.build().to(dtype=dtype)
    restored.load_state_dict(head.state_dict())
    torch.testing.assert_close(restored.project(targets), labels, atol=0, rtol=0)
    torch.testing.assert_close(restored.probs_to_scalar(labels), head.probs_to_scalar(labels), atol=0, rtol=0)
    # Meta exercises real nn.Module device transfer without requiring a GPU.
    head.to("meta")
    assert all(buffer.device.type == "meta" for buffer in head.buffers())
    assert head.project(torch.empty((2,), device="meta", dtype=dtype)).device.type == "meta"


@pytest.mark.parametrize("transform", ["linear", "symlog"])
@pytest.mark.parametrize("bin_type", ["centers", "edges"])
def test_frozen_support_equivalence_when_geometry_and_decode_match(transform, bin_type):
    cfg = config(transform=transform, bin_type=bin_type, decode="transformed")
    head = cfg.build()
    lo, hi = cfg.v_min, cfg.v_max
    if transform == "symlog":
        lo, hi = -math.log1p(-lo), math.log1p(hi)
    legacy = HLGaussSupport(
        cfg.num_bins, lo, hi, cfg.sigma_ratio, "cpu", use_symlog=transform == "symlog", support_is_edges=bin_type == "edges"
    )
    targets = torch.tensor([-10.0, -2.0, 0.0, 5.0, 20.0])
    torch.testing.assert_close(head.project(targets), legacy.project(targets), atol=2e-6, rtol=2e-5)
    logits = torch.linspace(-2, 2, cfg.num_bins).repeat(5, 1)
    torch.testing.assert_close(head.to_scalar(logits), legacy.to_scalar(logits), atol=2e-6, rtol=2e-6)


def test_loss_rejects_accidental_target_broadcasting_and_invalid_reduction():
    head = config().build()
    with pytest.raises(ValueError):
        head.loss(torch.zeros(2, 51), torch.zeros(2, 1))
    with pytest.raises(ValueError):
        head.loss(torch.zeros(2, 50), torch.zeros(2))
    with pytest.raises(ValueError):
        head.loss(torch.zeros(2, 51), torch.zeros(2), reduction="median")
    with pytest.raises(ValueError):
        head.probs_to_scalar(torch.zeros(2, 50))


def test_small_symlog_ranges_retain_resolution():
    head = config(num_bins=21, v_min=-1e-8, v_max=1e-8, transform="symlog").build()
    target = torch.tensor([-5e-9, 5e-9])
    torch.testing.assert_close(head.support[[0, -1]], torch.tensor([-1e-8, 1e-8]), atol=1e-15, rtol=1e-6)
    actual = head.probs_to_scalar(head.project(target))
    torch.testing.assert_close(actual, target, atol=1e-13, rtol=1e-5)


def test_transformed_decode_and_projection_have_correct_derivatives_at_zero():
    head = config(num_bins=2, v_min=-2, v_max=2, transform="symlog", decode="transformed").build().double()
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    (derivative,) = torch.autograd.grad(head.to_scalar(logits), logits)
    torch.testing.assert_close(derivative, head.coord_support / 2)
    target = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    (derivative,) = torch.autograd.grad(head.project(target)[1], target)
    # symlog is C1 but not C2 at zero, so centered differences converge O(h).
    finite_difference = (head.project(target.detach() + 1e-7)[1] - head.project(target.detach() - 1e-7)[1]) / 2e-7
    assert derivative > 0
    torch.testing.assert_close(derivative, finite_difference, atol=1e-7, rtol=1e-6)


def test_symexp_center_cells_match_frozen_raw_gaussian_reference():
    from scripts.hlgauss.factorial import reference_class

    cfg = config(num_bins=31, v_min=-1500, v_max=1500, bin_type="symexp_centers", sigma_ratio=2)
    head = cfg.build()
    legacy = reference_class("symexp_grid_raw")(31, -math.log1p(1500), math.log1p(1500), 2, "cpu")
    targets = torch.tensor([-3000.0, -1499, -500, -5, 0, 5, 500, 1499, 3000])
    torch.testing.assert_close(head.project(targets), legacy.project(targets), atol=3e-7, rtol=1e-4)
    torch.testing.assert_close(
        head.probs_to_scalar(head.project(targets)), legacy.to_scalar(legacy.project(targets).log()), atol=5e-4, rtol=2e-6
    )
    with pytest.raises(ValueError):
        config(bin_type="symexp_centers", transform="symlog")

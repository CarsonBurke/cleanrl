"""Dreamer3 numerical contracts; run CUDA work only through mlq.

The references below use Python scalar arithmetic, not the implementation's
searchsorted or decoder. They establish FP32 tolerances, not bitwise JAX parity.
"""

import math
import struct

import pytest
import torch

from cleanrl.shared.two_hot import DreamerTwoHotSupport

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required")]


@pytest.fixture
def device():
    return torch.device("cuda")


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _dreamer_centers(num_bins, bound):
    """Independent heads.py mirrored-half construction in scalar arithmetic."""
    count = (num_bins + 1) // 2
    start = _f32(-math.log1p(bound))
    coordinates = [start] if count == 1 else [_f32(start * (1 - i / (count - 1))) for i in range(count)]
    half = [-_f32(math.expm1(abs(value))) if value else 0.0 for value in coordinates]
    mirror = half[:-1] if num_bins % 2 else half
    return half + [-value for value in reversed(mirror)]


def _dreamer_labels(centers, target):
    """The count-based outs.py oracle, with explicitly rounded FP32 weights."""
    target = _f32(target)
    below = max(0, min(len(centers) - 1, sum(center <= target for center in centers) - 1))
    above = max(0, min(len(centers) - 1, len(centers) - sum(center > target for center in centers)))
    dist_below = 1.0 if below == above else abs(_f32(centers[below] - target))
    dist_above = 1.0 if below == above else abs(_f32(centers[above] - target))
    total = _f32(dist_below + dist_above)
    labels = [0.0] * len(centers)
    labels[below] = _f32(dist_above / total)
    labels[above] = _f32(labels[above] + _f32(dist_below / total))
    return labels


@pytest.mark.parametrize(
    "num_bins,bound",
    [(2, 10.0), (4, 2500.0), (7, 2500.0), (254, 10.0), (255, math.expm1(20))],
)
def test_support_matches_independent_dreamer_half_construction(device, num_bins, bound):
    head = DreamerTwoHotSupport(num_bins, bound, device=device)
    expected = torch.tensor(_dreamer_centers(num_bins, bound), dtype=torch.float32, device=device)
    # CUDA linspace/expm1 need not round identically to scalar math or JAX.
    torch.testing.assert_close(head.support, expected, atol=2e-6, rtol=3e-6)
    torch.testing.assert_close(head.support, -head.support.flip(0), atol=0, rtol=0)
    assert head.support.dtype == torch.float32
    assert head.support.device == device or head.support.device == torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("spacing", ["symexp", "linear"])
@pytest.mark.parametrize("num_bins,bound", [(2, 10.0), (8, 2500.0), (255, math.expm1(20))])
def test_raw_barycentric_labels_preserve_knots_interiors_and_clipped_means(device, num_bins, bound, spacing):
    head = DreamerTwoHotSupport(num_bins, bound, device=device, spacing=spacing)
    centers = head.support.tolist()
    interiors = [0.37 * left + 0.63 * right for left, right in zip(centers[:-1], centers[1:]) if left < right]
    values = centers + interiors + [-math.inf, -2 * bound, 2 * bound, math.inf]
    targets = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True)
    labels = head.project(targets)
    expected = torch.tensor([_dreamer_labels(centers, value) for value in values], device=device)
    torch.testing.assert_close(labels, expected, atol=2e-7, rtol=2e-6)
    assert labels.dtype == torch.float32
    assert not labels.requires_grad
    assert labels.shape == (len(values), num_bins)
    assert torch.isfinite(labels).all()
    assert (labels >= 0).all()
    assert ((labels != 0).sum(-1) <= 2).all()
    torch.testing.assert_close(labels.sum(-1), torch.ones(len(values), device=device), atol=2e-7, rtol=0)
    torch.testing.assert_close(
        head.probs_to_scalar(labels), targets.detach().float().clamp(centers[0], centers[-1]), atol=3e-6, rtol=4e-6
    )
    endpoint_labels = torch.zeros(4, num_bins, device=device)
    endpoint_labels[:2, 0] = 1
    endpoint_labels[2:, -1] = 1
    torch.testing.assert_close(labels[-4:], endpoint_labels, atol=0, rtol=0)


def test_even_zero_uses_rightmost_duplicate_and_two_bins_have_no_zero(device):
    head = DreamerTwoHotSupport(8, 10.0, device=device)
    zero = torch.zeros((), device=device)
    expected = torch.zeros(8, device=device)
    expected[4] = 1
    torch.testing.assert_close(head.project(zero), expected, atol=0, rtol=0)
    assert head.support[3].item() == head.support[4].item() == 0
    torch.testing.assert_close(head.probs_to_scalar(expected), zero, atol=0, rtol=0)
    two = DreamerTwoHotSupport(2, 10.0, device=device)
    torch.testing.assert_close(two.project(zero), torch.full((2,), 0.5, device=device), atol=0, rtol=0)
    assert two.support[0].item() < 0 < two.support[1].item()


@pytest.mark.parametrize("num_bins", [2, 254, 255])
def test_mirrored_probabilities_cancel_exactly_at_dreamer_default_bounds(device, num_bins):
    head = DreamerTwoHotSupport(num_bins, device=device)
    left = torch.linspace(-7, 2, num_bins // 2, device=device)
    center = torch.tensor([0.73], device=device) if num_bins % 2 else left[:0]
    logits = torch.cat((left, center, left.flip(0)))
    probabilities = logits.softmax(-1)
    torch.testing.assert_close(head.probs_to_scalar(probabilities), torch.zeros((), device=device), atol=0, rtol=0)
    torch.testing.assert_close(head.to_scalar(logits), torch.zeros((), device=device), atol=0, rtol=0)
    torch.testing.assert_close(head.to_scalar(torch.zeros_like(logits)), torch.zeros((), device=device), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_asymmetric_distribution_decodes_raw_expectation_not_inverse_coordinate_mean(device, dtype):
    head = DreamerTwoHotSupport(7, 2500.0, device=device)
    probabilities = torch.tensor([0.15, 0.0, 0.25, 0.10, 0.0, 0.0, 0.50], device=device)
    centers = head.support.tolist()
    weights = probabilities.tolist()
    raw_mean = math.fsum(weight * center for weight, center in zip(weights, centers))
    coordinate_mean = math.fsum(
        weight * math.copysign(math.log1p(abs(center)), center) for weight, center in zip(weights, centers)
    )
    inverse_mean = math.copysign(math.expm1(abs(coordinate_mean)), coordinate_mean)
    assert abs(raw_mean - inverse_mean) > 0.25 * head.max_abs_value
    torch.testing.assert_close(
        head.probs_to_scalar(probabilities), torch.tensor(raw_mean, device=device), atol=2e-4, rtol=2e-6
    )
    logits = probabilities.log().to(dtype)
    # Quantized logits define a different distribution; softmax must be FP32.
    reference_weights = [math.exp(value) for value in logits.float().tolist()]
    total = math.fsum(reference_weights)
    expected = math.fsum(weight * center / total for weight, center in zip(reference_weights, centers))
    decoded = head.to_scalar(logits)
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, torch.tensor(expected, device=device), atol=3e-4, rtol=2e-6)


def test_loss_matches_count_oracle_and_only_differentiates_logits(device):
    head = DreamerTwoHotSupport(8, 10.0, device=device)
    centers = head.support.tolist()
    # These FP64 values round to exact FP32 knots before labels are constructed.
    values = [math.nextafter(centers[1], -math.inf), math.nextafter(centers[5], math.inf), 0.137, 30.0]
    targets = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True).reshape(2, 2)
    targets.retain_grad()
    logits = torch.linspace(-1.7, 2.3, 32, device=device).reshape(2, 2, 8).detach().requires_grad_()
    labels = torch.tensor([_dreamer_labels(centers, value) for value in values], device=device).reshape(2, 2, 8)
    expected_losses = []
    for row, label in zip(logits.detach().reshape(-1, 8).tolist(), labels.reshape(-1, 8).tolist()):
        log_normalizer = math.log(math.fsum(math.exp(value) for value in row))
        expected_losses.append(-math.fsum(weight * (value - log_normalizer) for weight, value in zip(label, row)))
    expected = torch.tensor(expected_losses, device=device).reshape(2, 2)
    per_target = head.loss(logits, targets, reduction="none")
    torch.testing.assert_close(per_target, expected, atol=3e-7, rtol=2e-6)
    torch.testing.assert_close(head.loss(logits, targets, reduction="sum"), expected.sum(), atol=5e-7, rtol=2e-6)
    loss = head.loss(logits, targets)
    torch.testing.assert_close(loss, expected.mean(), atol=3e-7, rtol=2e-6)
    loss.backward()
    assert targets.grad is None
    torch.testing.assert_close(logits.grad, (logits.detach().softmax(-1) - labels) / 4, atol=3e-8, rtol=2e-6)
    # A transposed batch exercises non-contiguous targets without changing labels.
    torch.testing.assert_close(head.project(targets.T), labels.transpose(0, 1), atol=2e-7, rtol=2e-6)


def test_scalar_and_empty_batches_preserve_consumer_shapes(device):
    head = DreamerTwoHotSupport(5, 10.0, device=device)
    for shape in [(), (0,), (2, 0)]:
        targets = torch.zeros(shape, device=device)
        labels = head.project(targets)
        assert labels.shape == (*shape, 5)
        assert head.probs_to_scalar(labels).shape == shape
        logits = torch.zeros((*shape, 5), device=device)
        assert head.loss(logits, targets, reduction="none").shape == shape


def test_invalid_configuration_and_shapes_raise_real_errors(device):
    for num_bins in [1, 0, 2.5, True]:
        with pytest.raises(ValueError, match="num_bins"):
            DreamerTwoHotSupport(num_bins, device=device)
    for bound in [0, -1, math.inf, math.nan, 1e40]:
        with pytest.raises(ValueError, match="max_abs_value"):
            DreamerTwoHotSupport(max_abs_value=bound, device=device)
    head = DreamerTwoHotSupport(5, device=device)
    for bad in [torch.zeros((), device=device), torch.zeros(2, 4, device=device)]:
        with pytest.raises(ValueError, match="final dimension"):
            head.probs_to_scalar(bad)
        with pytest.raises(ValueError, match="final dimension"):
            head.to_scalar(bad)
    with pytest.raises(ValueError, match="shape"):
        head.loss(torch.zeros(2, 5, device=device), torch.zeros(2, 1, device=device))
    with pytest.raises(ValueError, match="reduction"):
        head.loss(torch.zeros(2, 5, device=device), torch.zeros(2, device=device), reduction="batchmean")
    with pytest.raises(ValueError, match="spacing"):
        DreamerTwoHotSupport(5, device=device, spacing="unknown")


@pytest.mark.parametrize("bound", [2e38, torch.finfo(torch.float32).max, 1e-50])
def test_unsafe_fp32_interpolation_support_is_rejected(device, bound):
    # Finite Python endpoints can overflow interval widths or round to zero.
    # Accepting them silently creates zero-mass or degenerate labels.
    with pytest.raises(ValueError):
        DreamerTwoHotSupport(2, bound, device=device)


@pytest.mark.parametrize("num_bins", [2, 254, 255])
def test_linear_spacing_is_uniform_raw_not_symlog(device, num_bins):
    head = DreamerTwoHotSupport(num_bins, 20000.0, device=device, spacing="linear")
    expected = torch.linspace(-20000.0, 20000.0, num_bins, device=device)
    torch.testing.assert_close(head.support, expected, rtol=2e-6, atol=0.002)
    torch.testing.assert_close(
        head.support.diff(), torch.full_like(head.support[:-1], 40000 / (num_bins - 1)), rtol=2e-5, atol=0.002
    )
    torch.testing.assert_close(
        head.to_scalar(torch.zeros(3, num_bins, device=device)), torch.zeros(3, device=device), rtol=0, atol=0
    )

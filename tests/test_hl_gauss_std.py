"""Numerical contracts for the target-standardized HL-Gauss head."""

import math

import pytest
import torch

from cleanrl.shared.hl_gauss_std import StandardizedHistogram


def head(**kwargs):
    torch.manual_seed(0)
    return StandardizedHistogram(**kwargs)


def test_labels_integrate_to_one_without_renormalization():
    histogram = head(num_bins=101)
    targets = torch.linspace(-40.0, 40.0, 257, dtype=torch.float64)
    histogram.double()
    labels = histogram.project(targets)
    assert torch.allclose(labels.sum(-1), torch.ones_like(targets), atol=1e-12)
    assert (labels >= 0).all()


def test_label_mean_tracks_target_inside_the_support():
    histogram = head(num_bins=101, half_span=5.0, sigma_bins=0.75).double()
    targets = torch.linspace(-4.0, 4.0, 401, dtype=torch.float64)
    decoded = histogram.decode_probs(histogram.project(targets))
    # Quantization bias of a Gaussian on a uniform grid decays as
    # exp(-2 pi^2 (sigma/width)^2); at 0.75 it is far below a milli-bin.
    assert (decoded - targets).abs().max() < 1e-4 * histogram.bin_width


def test_out_of_range_targets_saturate_instead_of_producing_nans():
    histogram = head(num_bins=51, half_span=4.0)
    targets = torch.tensor([-1e4, -20.0, 0.0, 20.0, 1e4])
    labels = histogram.project(targets)
    assert torch.isfinite(labels).all()
    assert torch.allclose(labels.sum(-1), torch.ones(5), atol=1e-5)
    decoded = histogram.decode_probs(labels)
    assert decoded[0].item() == pytest.approx(-4.0, abs=1e-5)
    assert decoded[-1].item() == pytest.approx(4.0, abs=1e-5)


def test_float32_projection_matches_float64():
    single = head(num_bins=201, half_span=5.0, sigma_bins=0.75)
    double = head(num_bins=201, half_span=5.0, sigma_bins=0.75).double()
    targets = torch.linspace(-6.0, 6.0, 513)
    difference = (single.project(targets) - double.project(targets.double())).abs().max()
    assert difference < 1e-6


def test_observe_adopts_the_current_rollout_statistics_without_lag():
    histogram = head(num_bins=51)
    first = torch.randn(4096) * 0.4 + 3.0
    histogram.observe(first)
    assert histogram.mean.item() == pytest.approx(first.mean().item(), abs=1e-5)
    assert histogram.scale.item() == pytest.approx(first.std().item(), abs=1e-5)
    second = torch.randn(4096) * 0.9 + 5.0
    histogram.observe(second)
    # No smoothing: the support follows the drift immediately.
    assert histogram.mean.item() == pytest.approx(second.mean().item(), abs=1e-5)
    assert histogram.scale.item() == pytest.approx(second.std().item(), abs=1e-5)


def test_scale_floor_survives_a_constant_target_batch():
    histogram = head(num_bins=51, min_scale=1e-3)
    histogram.observe(torch.full((512,), 2.5))
    assert histogram.scale.item() == pytest.approx(1e-3)
    assert torch.isfinite(histogram.project(torch.full((8,), 2.5))).all()


def test_decode_is_exact_for_a_perfectly_fit_head():
    histogram = head(num_bins=101).double()
    histogram.observe(torch.randn(8192, dtype=torch.double) * 0.4 + 3.8)
    targets = torch.tensor([3.0, 3.5, 3.83, 4.2], dtype=torch.double)
    logits = histogram.project(targets).clamp_min(1e-300).log()
    assert (histogram.decode(logits) - targets).abs().max() < 1e-6


def test_cross_entropy_gradient_is_mse_gradient_divided_by_head_variance():
    """The calibration claim behind value_gradient_gain.

    Valid to first order in (target - value), so the displacement is a small
    fraction of sigma; that is the regime a converged critic operates in.
    """
    histogram = head(num_bins=101, half_span=5.0, sigma_bins=0.75).double()
    histogram.observe(torch.randn(8192, dtype=torch.double) * 0.4 + 3.8)
    value = torch.tensor([3.6], dtype=torch.double)
    target = value + 0.002 * histogram.scale
    logits = histogram.project(value).clamp_min(1e-300).log().requires_grad_(True)
    labels = histogram.project(target)
    cross_entropy = -(labels * logits.log_softmax(-1)).sum()
    (ce_grad,) = torch.autograd.grad(cross_entropy, logits)
    probs = logits.detach().softmax(-1)
    decoded = histogram.decode_probs(probs)
    # dV/dlogits for the same head, so both gradients live in one basis.
    dv = histogram.scale * probs * (histogram.centers - histogram.standardize(decoded))
    mse_grad = (decoded - target) * dv
    variance = histogram.standardized_variance(probs) * histogram.scale.square()
    live = probs > 1e-8
    matched = (ce_grad * variance)[live]
    reference = mse_grad[live]
    assert (matched - reference).abs().max() < 0.02 * reference.abs().max()
    assert histogram.value_gradient_gain(probs).item() == pytest.approx(variance.item(), rel=1e-6)
    assert histogram.converged_gain().item() == pytest.approx(variance.item(), rel=0.05)


def test_geometry_is_invariant_to_the_target_scale():
    histogram = head(num_bins=101, half_span=5.0)
    for centre, spread in ((0.0, 1.0), (3.83, 0.386), (1195.0, 218.0)):
        histogram.observe(torch.randn(4096) * spread + centre)
        targets = torch.randn(4096) * spread + centre
        labels = histogram.project(targets)
        overflow = (histogram.standardize(targets).abs() > histogram.half_span).float().mean()
        assert overflow.item() == 0.0
        # Two targets one target-std apart stay clearly distinguishable.
        pair = histogram.project(torch.tensor([centre, centre + spread]))
        symmetric_kl = (
            (pair[0] - pair[1]) * (pair[0].clamp_min(1e-30).log() - pair[1].clamp_min(1e-30).log())
        ).sum()
        assert symmetric_kl.item() > 10.0
        assert torch.allclose(labels.sum(-1), torch.ones(4096), atol=1e-5)


def test_rejects_degenerate_configuration():
    with pytest.raises(ValueError):
        StandardizedHistogram(num_bins=2)
    with pytest.raises(ValueError):
        StandardizedHistogram(num_bins=51, sigma_bins=0.0)
    with pytest.raises(ValueError):
        StandardizedHistogram(num_bins=51, min_scale=0.0)


def test_half_infinite_edges_are_finite_after_scaling():
    histogram = head(num_bins=31, half_span=3.0)
    assert math.isinf(histogram.edges[0].item()) and histogram.edges[0].item() < 0
    assert math.isinf(histogram.edges[-1].item()) and histogram.edges[-1].item() > 0
    assert torch.isfinite(histogram.edges[1:-1]).all()

"""Numerical CUDA projection tests; execute this module only through mlq.

These compare the frozen equations, labels, decoded moments, KL, and downstream
critic gradients. They are fixed-work correctness checks, not training runs.
"""

import math

import pytest
import torch

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport
from cleanrl.shared.hl_gauss_kernels import (
    moment_match_from_log_probs,
    project_moment_matched_fused,
)
from cleanrl.shared.runtime import configure_runtime


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def make_support(bins=51):
    configure_runtime()
    limit = math.log1p(20_000.0)
    return Dreamer3BucketHLGaussSupport(bins, -limit, limit, 0.75, "cuda")


def comparison_targets():
    wide = torch.linspace(-21_000.0, 21_000.0, 129, device="cuda")
    near_zero = torch.linspace(-1.0, 1.0, 65, device="cuda")
    coordinates = torch.linspace(-math.log1p(20_000), math.log1p(20_000), 129, device="cuda")
    logarithmic = coordinates.sign() * coordinates.abs().expm1()
    return torch.cat((wide, near_zero, logarithmic))


def assert_projection_close(support, reference, actual):
    assert actual.dtype == torch.float32
    assert actual.device.type == "cuda"
    assert torch.isfinite(actual).all()
    assert (actual >= 0).all()
    torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[..., 0]), rtol=0, atol=3e-7)
    torch.testing.assert_close(actual, reference, rtol=5e-5, atol=3e-6)
    # Large symexp buckets amplify small mass changes; also compare the actual
    # decoded training target, not only elementwise probability differences.
    torch.testing.assert_close(
        support.probs_to_scalar(actual), support.probs_to_scalar(reference),
        rtol=3e-6, atol=1e-2,
    )
    kl = (reference * (reference.clamp_min(1e-30).log() - actual.clamp_min(1e-30).log())).sum(-1)
    assert kl.max() <= 2e-6


@pytest.mark.parametrize("bins", [3, 51, 101])
def test_fused_labels_preserve_distribution_means_and_critic_gradients(bins):
    support = make_support(bins)
    targets = comparison_targets()
    with torch.no_grad():
        expected = support.project_moment_matched(targets)
        state_before = torch.cuda.get_rng_state()
        actual = project_moment_matched_fused(support, targets)
        assert torch.equal(state_before, torch.cuda.get_rng_state())
    assert_projection_close(support, expected, actual)

    generator = torch.Generator(device="cuda").manual_seed(1)
    logits = torch.randn(actual.shape, generator=generator, device="cuda", requires_grad=True)
    expected_loss = -(expected * logits.log_softmax(-1)).sum(-1).mean()
    actual_loss = -(actual * logits.log_softmax(-1)).sum(-1).mean()
    expected_grad, = torch.autograd.grad(expected_loss, logits)
    actual_grad, = torch.autograd.grad(actual_loss, logits)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-5, atol=3e-8)


@pytest.mark.parametrize("iterations,tilt_bound,cutoff", [(0, 1.0, 30.0), (8, 0.25, 12.0), (32, 2.0, 40.0)])
def test_nondefault_projection_parameters_match_frozen_equations(iterations, tilt_bound, cutoff):
    support = make_support()
    targets = comparison_targets()
    kwargs = dict(iterations=iterations, tilt_bound=tilt_bound, log_mass_cutoff=cutoff)
    with torch.no_grad():
        expected = support.project_moment_matched(targets, **kwargs)
        actual = project_moment_matched_fused(support, targets, **kwargs)
    assert_projection_close(support, expected, actual)


def test_noncontiguous_batches_and_support_preserve_layout_and_values():
    support = make_support()
    targets = torch.linspace(-100, 100, 63, device="cuda").reshape(7, 9)
    log_probs = support.project_log_probs(targets).permute(1, 0, 2)
    targets = targets.t()
    storage = torch.empty(2 * support.num_bins, device="cuda")
    strided_support = storage[::2]
    strided_support.copy_(support.support)
    assert not targets.is_contiguous() and not log_probs.is_contiguous()
    with torch.no_grad():
        expected = support.project_moment_matched(targets)
        actual = moment_match_from_log_probs(log_probs, targets, strided_support)
    assert actual.shape == (9, 7, support.num_bins)
    assert_projection_close(support, expected, actual)


def test_exact_endpoint_selection_nan_propagation_and_empty_batch():
    support = make_support()
    targets = torch.stack((support.support[0], support.support[-1]))
    targets = torch.cat((targets, torch.tensor([-torch.inf, torch.inf, torch.nan], device="cuda")))
    with torch.no_grad():
        expected = support.project_moment_matched(targets)
        actual = project_moment_matched_fused(support, targets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    empty = project_moment_matched_fused(support, torch.empty((0, 2), device="cuda"))
    assert empty.shape == (0, 2, support.num_bins)
    scalar = torch.tensor(3.0, device="cuda")
    assert_projection_close(support, support.project_moment_matched(scalar), project_moment_matched_fused(support, scalar))


def test_differentiable_targets_are_rejected_instead_of_silently_detached():
    support = make_support()
    targets = torch.zeros(4, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="constant target labels"):
        project_moment_matched_fused(support, targets)
    with torch.no_grad():
        actual = project_moment_matched_fused(support, targets)
    assert not actual.requires_grad
    fixed_targets = targets.detach()
    log_probs = support.project_log_probs(fixed_targets)
    with pytest.raises(ValueError, match="constant target labels"):
        moment_match_from_log_probs(log_probs.requires_grad_(), fixed_targets, support.support)
    with pytest.raises(ValueError, match="constant target labels"):
        moment_match_from_log_probs(log_probs.detach(), fixed_targets, support.support.clone().requires_grad_())


def test_shape_dtype_and_iteration_contracts_fail_explicitly():
    support = make_support()
    targets = torch.zeros(4, device="cuda")
    log_probs = support.project_log_probs(targets)
    with pytest.raises(ValueError, match="shape"):
        moment_match_from_log_probs(log_probs[:-1], targets, support.support)
    with pytest.raises(ValueError, match="float32"):
        moment_match_from_log_probs(log_probs.double(), targets, support.support)
    with pytest.raises(ValueError, match="iterations"):
        moment_match_from_log_probs(log_probs, targets, support.support, iterations=-1)
    with pytest.raises(ValueError, match="tilt_bound"):
        moment_match_from_log_probs(log_probs, targets, support.support, tilt_bound=0)


def test_compiled_fullgraph_projection_matches_compiled_frozen_reference():
    support = make_support()
    targets = torch.linspace(-20_100, 20_100, 39 * 16, device="cuda")
    frozen = torch.compile(support.project_moment_matched, fullgraph=True, mode="reduce-overhead")
    fused = torch.compile(lambda t: project_moment_matched_fused(support, t), fullgraph=True, mode="reduce-overhead")
    with torch.no_grad():
        torch.compiler.cudagraph_mark_step_begin()
        expected = frozen(targets).clone()
        torch.compiler.cudagraph_mark_step_begin()
        actual = fused(targets).clone()
    assert_projection_close(support, expected, actual)

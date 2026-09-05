"""Opt-in CUDA kernels for the existing Dreamer HL-Gauss target projection.

The stable Gaussian CDF/log-mass calculation stays in ``hl_gauss.py``. One
Triton program per target keeps all 32 exponential-tilt bisection iterations
on chip, instead of launching a succession of softmax/reduction kernels.
The support, cutoff, bounds, iteration count, endpoint labels, and symmetric
paired scalar expectation follow ``project_moment_matched`` unchanged.

This is an explicit target-label optimization, not a new projection objective.
Floating-point reduction order can differ; use the numerical and performance
comparisons in ``benchmark_mujoco_throughput.py`` before enabling it in a run.
Autograd through target/support construction is deliberately rejected instead
of being silently lost. Gradients of a loss using these constant labels still
flow normally into its predicted logits.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _paired_mean(probabilities, support, bins: tl.constexpr, block: tl.constexpr):
    """Match probs_to_scalar's mirrored pairs before reducing cancellation."""
    index = tl.arange(0, block)
    half: tl.constexpr = (bins - 1) // 2
    contributions = probabilities * support
    left_index = tl.maximum(half - 1 - index, 0)
    right_index = tl.minimum(half + 1 + index, bins - 1)
    left = tl.gather(contributions, left_index, axis=0)
    right = tl.gather(contributions, right_index, axis=0)
    paired_sum = tl.sum(tl.where(index < half, left + right, 0.0), axis=0)
    center = tl.sum(tl.where(index == half, contributions, 0.0), axis=0)
    return center + paired_sum


@triton.jit
def _softmax(logits):
    # Deliberately use libdevice exp and rounded division, not an approximate
    # exponential or reciprocal that changes the numerical problem for speed.
    numerator = libdevice.exp(logits - tl.max(logits, axis=0))
    return tl.div_rn(numerator, tl.sum(numerator, axis=0))


@triton.jit
def _moment_match_kernel(
    log_probs_ptr,
    targets_ptr,
    support_ptr,
    output_ptr,
    bins: tl.constexpr,
    block: tl.constexpr,
    iterations: tl.constexpr,
    tilt_bound: tl.constexpr,
    log_mass_cutoff: tl.constexpr,
):
    row = tl.program_id(axis=0)
    index = tl.arange(0, block)
    log_probs = tl.load(log_probs_ptr + row * bins + index, index < bins, other=-float("inf"))
    support = tl.load(support_ptr + index, index < bins, other=0.0)
    target = tl.load(targets_ptr + row)
    low_endpoint = tl.load(support_ptr)
    high_endpoint = tl.load(support_ptr + bins - 1)
    matched_target = tl.minimum(tl.maximum(target, low_endpoint), high_endpoint)
    maximum_log_prob = tl.max(log_probs, axis=0)
    has_nan_log_prob = tl.sum((log_probs != log_probs).to(tl.int32), axis=0) > 0
    log_probs = tl.where(log_probs >= maximum_log_prob - log_mass_cutoff, log_probs, -float("inf"))
    low = tl.full((), -tilt_bound, tl.float32)
    high = tl.full((), tilt_bound, tl.float32)
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        candidate = _softmax(log_probs + midpoint * support)
        mean = _paired_mean(candidate, support, bins, block)
        below = mean < matched_target
        low = tl.where(below, midpoint, low)
        high = tl.where(below, high, midpoint)
    tilt = 0.5 * (low + high)
    output = _softmax(log_probs + tilt * support)
    output = tl.where(has_nan_log_prob | (target != target), float("nan"), output)
    output = tl.where(target <= low_endpoint, (index == 0).to(tl.float32), output)
    output = tl.where(target >= high_endpoint, (index == bins - 1).to(tl.float32), output)
    tl.store(output_ptr + row * bins + index, output, index < bins)


def _validate_inputs(log_probs, targets, scalar_support, iterations, tilt_bound, log_mass_cutoff):
    if scalar_support.ndim != 1 or scalar_support.numel() < 3 or scalar_support.numel() % 2 != 1:
        raise ValueError("scalar_support must have an odd number of at least three bins")
    if log_probs.shape != targets.shape + (scalar_support.numel(),):
        raise ValueError("log_probs must have shape targets.shape + (number of bins,)")
    for name, tensor in (("log_probs", log_probs), ("targets", targets), ("scalar_support", scalar_support)):
        if tensor.device.type != "cuda" or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be a CUDA float32 tensor")
        if tensor.device != targets.device:
            raise ValueError("all projection inputs must be on the same CUDA device")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise ValueError("fused projection produces constant target labels; use torch.no_grad() or the differentiable reference")
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not math.isfinite(tilt_bound) or tilt_bound <= 0:
        raise ValueError("tilt_bound must be finite and positive")
    if not math.isfinite(log_mass_cutoff) or log_mass_cutoff < 0:
        raise ValueError("log_mass_cutoff must be finite and nonnegative")


def moment_match_from_log_probs(
    log_probs,
    targets,
    scalar_support,
    *,
    iterations=32,
    tilt_bound=1.0,
    log_mass_cutoff=30.0,
):
    """Fuse the existing tilt bisection for precomputed Gaussian log-masses.

    Inputs are float32 CUDA tensors. ``scalar_support`` is the existing odd,
    symmetric, increasing Dreamer support (including its exact central zero).
    No data-dependent host synchronization is added to inspect those values.
    Arbitrary target batch shapes and non-contiguous inputs are accepted.
    """
    _validate_inputs(log_probs, targets, scalar_support, iterations, tilt_bound, log_mass_cutoff)
    bins = scalar_support.numel()
    output = torch.empty(log_probs.shape, device=log_probs.device, dtype=log_probs.dtype)
    if targets.numel() == 0:
        return output
    _moment_match_kernel[(targets.numel(),)](
        log_probs.contiguous(), targets.contiguous(), scalar_support.contiguous(), output,
        bins=bins, block=triton.next_power_of_2(bins), iterations=iterations,
        tilt_bound=float(tilt_bound), log_mass_cutoff=float(log_mass_cutoff),
        num_warps=4 if bins > 128 else 1,
        enable_fp_fusion=False,
    )
    return output


def project_moment_matched_fused(
    hl_support,
    targets,
    *,
    iterations=32,
    tilt_bound=1.0,
    log_mass_cutoff=30.0,
):
    """Drop-in numerical candidate for constant Dreamer HL-Gauss target labels.

    Compile this callable around the existing support object to also fuse the
    stable PyTorch CDF preprocessing. The frozen support class is not patched.
    """
    if torch.is_grad_enabled() and targets.requires_grad:
        raise ValueError("fused projection produces constant target labels; use torch.no_grad() or the differentiable reference")
    log_probs = hl_support.project_log_probs(targets)
    return moment_match_from_log_probs(
        log_probs, targets, hl_support.support, iterations=iterations,
        tilt_bound=tilt_bound, log_mass_cutoff=log_mass_cutoff,
    )

"""Opt-in fused V-MPO temperature bisection, with the original 32 iterations.

One Triton program keeps the complete selected batch and geometric bracket on
chip. Selection (including cutoff ties), centering, initial bounds, downstream
weights, and policy gradients remain the caller's original implementation.
Reduction order can differ from PyTorch: numerical and throughput validation
are required before enabling this candidate. This is a constant-target solver,
not a differentiable temperature estimator.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def solve_log_temperature_reference(
    centered_advantages, selected, log_selected_count, log_eta_low,
    log_eta_high, epsilon_eta, iterations=32,
):
    """Original PyTorch recurrence, reusable for isolated compiled comparisons."""
    for _ in range(iterations):
        log_eta_mid = 0.5 * (log_eta_low + log_eta_high)
        eta_mid = log_eta_mid.exp()
        mid_logits = torch.where(selected, centered_advantages / eta_mid, -torch.inf)
        mid_log_weights = mid_logits - torch.logsumexp(mid_logits, dim=0)
        mid_weights = mid_log_weights.exp()
        safe_mid_log_weights = torch.where(selected, mid_log_weights, 0.0)
        mid_kl = (mid_weights * (safe_mid_log_weights + log_selected_count)).sum()
        log_eta_low = torch.where(mid_kl > epsilon_eta, log_eta_mid, log_eta_low)
        log_eta_high = torch.where(mid_kl > epsilon_eta, log_eta_high, log_eta_mid)
    return log_eta_high


@triton.jit
def _solve_log_temperature_kernel(
    advantages_ptr, selected_ptr, log_count_ptr, low_ptr, high_ptr,
    epsilon_arg, output_ptr,
    size: tl.constexpr, block: tl.constexpr, iterations: tl.constexpr,
    epsilon_is_tensor: tl.constexpr,
):
    index = tl.arange(0, block)
    valid = index < size
    advantages = tl.load(advantages_ptr + index, valid, other=0.0)
    selected = tl.load(selected_ptr + index, valid, other=False)
    log_count = tl.load(log_count_ptr)
    low = tl.load(low_ptr)
    high = tl.load(high_ptr)
    if epsilon_is_tensor:
        epsilon = tl.load(epsilon_arg)
    else:
        epsilon = epsilon_arg
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        eta = libdevice.exp(midpoint)
        logits = tl.where(selected, tl.div_rn(advantages, eta), -float("inf"))
        maximum = tl.max(logits, axis=0)
        # PyTorch logsumexp subtracts zero when the maximum is +/-inf.
        # Preserve its nonfinite behavior instead of adding a stabilizing clamp.
        shift = tl.where(tl.abs(maximum) == float("inf"), 0.0, maximum)
        exponentials = libdevice.exp(logits - shift)
        log_normalizer = libdevice.log(tl.sum(exponentials, axis=0)) + shift
        log_weights = logits - log_normalizer
        weights = libdevice.exp(log_weights)
        safe_log_weights = tl.where(selected, log_weights, 0.0)
        contributions = weights * (safe_log_weights + log_count)
        kl = tl.sum(tl.where(valid, contributions, 0.0), axis=0)
        above = kl > epsilon
        low = tl.where(above, midpoint, low)
        high = tl.where(above, high, midpoint)
    tl.store(output_ptr, high)


def solve_log_temperature(
    centered_advantages, selected, log_selected_count, log_eta_low,
    log_eta_high, epsilon_eta, iterations=32,
):
    """Return the reference bisection's final scalar upper log-temperature bound.

    Advantages are a nonempty float32 CUDA vector; ``selected`` is a matching
    bool vector. Count and bounds are scalar float32 CUDA tensors on the same
    device. ``epsilon_eta`` is a positive Python scalar or a scalar float32 CUDA
    tensor. Tensor values are never read back to validate data-dependent inputs.

    Call under ``torch.no_grad()`` when inputs require gradients. In particular,
    this solver must not silently detach a requested temperature gradient.
    Noncontiguous vectors are accepted and staged once before the single kernel.
    No sampling occurs and no RNG state changes.
    """
    if centered_advantages.ndim != 1 or centered_advantages.numel() == 0:
        raise ValueError("centered_advantages must be a nonempty vector")
    if selected.shape != centered_advantages.shape or selected.dtype != torch.bool:
        raise ValueError("selected must be a bool vector matching centered_advantages")
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    named = (
        ("centered_advantages", centered_advantages),
        ("log_selected_count", log_selected_count),
        ("log_eta_low", log_eta_low),
        ("log_eta_high", log_eta_high),
    )
    for name, value in named:
        if value.device.type != "cuda" or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a float32 CUDA tensor")
        if value.device != centered_advantages.device:
            raise ValueError("all inputs must share a CUDA device")
        if name != "centered_advantages" and value.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor")
        if torch.is_grad_enabled() and value.requires_grad:
            raise ValueError("temperature solver produces constant targets; use torch.no_grad() or the differentiable reference")
    if selected.device != centered_advantages.device:
        raise ValueError("all inputs must share a CUDA device")
    epsilon_is_tensor = isinstance(epsilon_eta, torch.Tensor)
    if epsilon_is_tensor:
        if (epsilon_eta.ndim != 0 or epsilon_eta.dtype != torch.float32
                or epsilon_eta.device != centered_advantages.device):
            raise ValueError("epsilon_eta must be a scalar float32 tensor on the same CUDA device")
        if torch.is_grad_enabled() and epsilon_eta.requires_grad:
            raise ValueError("temperature solver produces constant targets; use torch.no_grad() or the differentiable reference")
    elif not isinstance(epsilon_eta, (float, int)) or not math.isfinite(epsilon_eta) or epsilon_eta <= 0:
        raise ValueError("epsilon_eta must be finite and positive")
    size = centered_advantages.numel()
    output = torch.empty_like(log_eta_high)
    _solve_log_temperature_kernel[(1,)](
        centered_advantages.contiguous(), selected.contiguous(), log_selected_count,
        log_eta_low, log_eta_high, epsilon_eta, output,
        size=size, block=triton.next_power_of_2(size), iterations=iterations,
        epsilon_is_tensor=epsilon_is_tensor,
        num_warps=4 if size <= 1024 else 8 if size <= 8192 else 16,
        enable_fp_fusion=False,
    )
    return output

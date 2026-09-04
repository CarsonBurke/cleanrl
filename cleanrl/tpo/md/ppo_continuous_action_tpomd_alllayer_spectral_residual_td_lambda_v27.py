# TPO-MD ALL-LAYER SPECTRAL RESIDUAL TD(lambda) v27: graph-free predictive coding.
#
# Each actor-trunk layer predicts its one-step latent innovation and nine rFFT
# successor-innovation modes on a period-16 frequency grid.  A coherent frozen
# trunk/policy/predictor snapshot supplies SARSA bootstraps, while a vectorized 16-step
# forward-view TD(lambda) mixture grounds every mode in multiple realized transitions.
# Targets are detached: there is no BPTT, learned dynamics rollout, intrinsic reward, or
# probe-derived supervision.  The spectral modes preserve temporal phase and duration that
# a single delayed endpoint cannot identify.  H1 and spectral families receive equal weight,
# with local per-layer scales and the proven v25 trunk/predictor gradient budgets unchanged.
import copy
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from math import ceil, gcd, log
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport, symlog, symexp

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))
SPECTRAL_PERIOD = 16
SPECTRAL_NUM_FREQUENCIES = SPECTRAL_PERIOD // 2 + 1
SPECTRAL_NUM_SLOTS = 1 + 2 * SPECTRAL_NUM_FREQUENCIES
SPECTRAL_DIAGNOSTIC_STEPS = 64


@dataclass(frozen=True)
class SpectralTDLambdaTargets:
    """Detached packed H1 and complex successor-mode labels."""

    values: torch.Tensor
    masks: torch.Tensor


def spectral_frequencies(
    period: int = SPECTRAL_PERIOD,
    *,
    reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Angular frequencies of the real-FFT half spectrum."""
    if period < 2 or period % 2:
        raise ValueError("the spectral period must be positive and even")
    if reference is None:
        return (
            2.0
            * torch.pi
            * torch.arange(period // 2 + 1, dtype=torch.float32)
            / period
        )
    return (
        2.0
        * torch.pi
        * torch.arange(
            period // 2 + 1,
            dtype=reference.dtype,
            device=reference.device,
        )
        / period
    )


def spectral_component_mask(
    period: int = SPECTRAL_PERIOD,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Active [real,imag] components; DC and Nyquist have no imaginary part."""
    if period < 2 or period % 2:
        raise ValueError("the spectral period must be positive and even")
    active = torch.ones((period // 2 + 1, 2), dtype=torch.bool, device=device)
    active[0, 1] = False
    active[-1, 1] = False
    return active


def spectral_slot_mask(
    period: int = SPECTRAL_PERIOD,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Packed active mask for [h1,re0,im0,...,reN/2,imN/2]."""
    return torch.cat(
        (
            torch.ones(1, dtype=torch.bool, device=device),
            spectral_component_mask(period, device=device).reshape(-1),
        )
    )


def split_spectral_slots(
    packed: torch.Tensor,
    period: int = SPECTRAL_PERIOD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed [...,slot,latent] outputs into H1 and [...,freq,2,latent]."""
    expected_slots = 1 + 2 * (period // 2 + 1)
    if packed.ndim < 2 or packed.shape[-2] != expected_slots:
        raise ValueError(f"packed predictions need {expected_slots} spectral slots")
    modes = packed[..., 1:, :].reshape(
        *packed.shape[:-2], period // 2 + 1, 2, packed.shape[-1]
    )
    return packed[..., 0, :], modes


def pack_spectral_slots(h1: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    """Pack one-step residuals and complex modes without changing leading axes."""
    if modes.ndim != h1.ndim + 2 or modes.shape[:-3] != h1.shape[:-1]:
        raise ValueError("H1 and spectral mode leading dimensions do not match")
    if modes.shape[-2] != 2 or modes.shape[-1] != h1.shape[-1]:
        raise ValueError("spectral modes must end in [frequency,2,latent]")
    return torch.cat(
        (h1.unsqueeze(-2), modes.flatten(start_dim=-3, end_dim=-2)), dim=-2
    )


def mask_inactive_spectral_components(
    modes: torch.Tensor,
    period: int = SPECTRAL_PERIOD,
) -> torch.Tensor:
    if modes.ndim < 3 or modes.shape[-3:] != (
        period // 2 + 1,
        2,
        modes.shape[-1],
    ):
        raise ValueError("spectral modes must end in [frequency,2,latent]")
    shape = (1,) * (modes.ndim - 3) + (period // 2 + 1, 2, 1)
    active = spectral_component_mask(period, device=modes.device).reshape(shape)
    return torch.where(active, modes, torch.zeros_like(modes))


def rotate_complex_modes(
    modes: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Multiply packed real/imag modes by exp(-i*omega)."""
    if modes.ndim < 3 or modes.shape[-2] != 2:
        raise ValueError("complex modes must end in [frequency,2,latent]")
    if frequencies.ndim != 1 or modes.shape[-3] != frequencies.numel():
        raise ValueError("one angular frequency is required per complex mode")
    shape = (1,) * (modes.ndim - 3) + (frequencies.numel(), 1)
    cosine = frequencies.to(modes).cos().reshape(shape)
    sine = frequencies.to(modes).sin().reshape(shape)
    real = modes[..., 0, :]
    imag = modes[..., 1, :]
    rotated = torch.stack(
        (cosine * real + sine * imag, -sine * real + cosine * imag), dim=-2
    )
    return mask_inactive_spectral_components(rotated, 2 * (frequencies.numel() - 1))


def build_spectral_td_lambda_targets(
    innovations: torch.Tensor,
    current_predictions: torch.Tensor,
    edge_next_predictions: torch.Tensor,
    transition_terminations: torch.Tensor,
    transition_boundaries: torch.Tensor,
    transition_valids: torch.Tensor,
    tape_lambda: float,
    rho: float,
    period: int = SPECTRAL_PERIOD,
) -> SpectralTDLambdaTargets:
    """Build a detached truncated forward-view successor TD(lambda) target.

    G1 uses a frozen one-step SARSA bootstrap.  G2..GN replace that tail with
    successively longer realized innovation paths, stopping at terminals, time-limit
    cutoffs, rollout edges, or a missing next label.  The lambda mixture is
    (1-lambda) * sum_{n=1}^{N-1} lambda^(n-1) Gn + lambda^(N-1) GN.
    Its frozen-predictor fixed point is the Bellman successor-mode fixed point while
    requiring only N vectorized horizon iterations, never a temporal autograd graph.
    """
    if innovations.ndim != 4 or current_predictions.ndim != 5:
        raise ValueError(
            "innovations and predictions must be [T,N,L,D] and [T,N,L,S,D]"
        )
    if innovations.shape[:3] != current_predictions.shape[:3]:
        raise ValueError("innovation and prediction rows do not match")
    if innovations.shape[-1] != current_predictions.shape[-1]:
        raise ValueError("innovation and prediction latent dimensions do not match")
    expected_slots = 1 + 2 * (period // 2 + 1)
    if current_predictions.shape[-2] != expected_slots:
        raise ValueError("current predictions have the wrong spectral slot count")
    if edge_next_predictions.ndim != 4 or edge_next_predictions.shape[1:] != (
        innovations.shape[2],
        expected_slots,
        innovations.shape[3],
    ):
        raise ValueError("edge predictions must be compact [edge,L,S,D]")
    expected_mask = innovations.shape[:2]
    if any(
        mask.shape != expected_mask
        for mask in (
            transition_terminations,
            transition_boundaries,
            transition_valids,
        )
    ):
        raise ValueError("transition masks have the wrong shape")
    if not 0.0 <= tape_lambda <= 1.0:
        raise ValueError("tape_lambda must be in [0, 1]")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")

    valid = transition_valids.detach().bool()
    terminations = transition_terminations.detach().bool()
    boundaries = transition_boundaries.detach().bool()
    torch._assert_async(
        torch.all(~terminations | boundaries),
        "every true termination must also be a boundary",
    )
    safe_eta = torch.where(
        valid.unsqueeze(-1).unsqueeze(-1),
        innovations.detach(),
        torch.zeros_like(innovations),
    )
    _, frozen_modes = split_spectral_slots(current_predictions.detach(), period)
    _, edge_modes = split_spectral_slots(edge_next_predictions.detach(), period)
    frequencies = spectral_frequencies(period, reference=innovations)
    num_frequencies = frequencies.numel()
    injection = torch.stack(
        (
            safe_eta.unsqueeze(-2).expand(
                *safe_eta.shape[:-1], num_frequencies, safe_eta.shape[-1]
            ),
            torch.zeros(
                (*safe_eta.shape[:-1], num_frequencies, safe_eta.shape[-1]),
                dtype=safe_eta.dtype,
                device=safe_eta.device,
            ),
        ),
        dim=-2,
    )

    not_last = torch.ones_like(valid)
    not_last[-1] = False
    ordinary = valid & ~boundaries & not_last
    terminal = valid & terminations
    edge_cutoff = valid & ~terminations & (boundaries | ~not_last)
    edge_indices = edge_cutoff.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if edge_modes.shape[0] != edge_indices.shape[0]:
        raise ValueError("edge predictions must exactly match cutoff-row order")

    one_step_tail = torch.zeros_like(frozen_modes)
    one_step_tail[:-1] = torch.where(
        ordinary[:-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        frozen_modes[1:],
        torch.zeros_like(frozen_modes[1:]),
    )
    one_step_tail.reshape(-1, *one_step_tail.shape[2:]).index_copy_(
        0, edge_indices, edge_modes
    )
    g_one = injection + rho * rotate_complex_modes(one_step_tail, frequencies)
    g_one = torch.where(
        valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        g_one,
        torch.zeros_like(g_one),
    )
    g_one = torch.where(
        terminal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        injection,
        g_one,
    )

    lam = innovations.new_tensor(tape_lambda)
    previous = g_one
    continuation = ordinary[:-1] & valid[1:]
    continuation_full = torch.cat(
        (continuation, torch.zeros_like(valid[-1:])), dim=0
    ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # Algebraically identical to the explicit capped forward-view mixture, but it
    # avoids retaining a second full spectral accumulator (~755 MiB at defaults).
    # Y1=G1; Yh=eta+beta*((1-lambda)P^-next+lambda*Y{h-1}_next).
    for _ in range(2, period + 1):
        shifted = torch.zeros_like(previous)
        shifted[:-1] = torch.where(
            continuation.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            previous[1:],
            torch.zeros_like(previous[1:]),
        )
        td_tail = torch.lerp(one_step_tail, shifted, lam)
        extended = injection + rho * rotate_complex_modes(td_tail, frequencies)
        current = torch.where(continuation_full, extended, g_one)
        current = torch.where(
            valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            current,
            torch.zeros_like(current),
        )
        previous = current

    mixed_modes = previous
    del (
        previous,
        current,
        shifted,
        td_tail,
        extended,
        g_one,
        one_step_tail,
        injection,
        frozen_modes,
        edge_modes,
    )
    # No target tensor participates in autograd, so zeroing the structurally absent
    # endpoint quadratures in place avoids another full-size allocation.
    mixed_modes[..., 0, 1, :].zero_()
    mixed_modes[..., -1, 1, :].zero_()
    packed = pack_spectral_slots(safe_eta, mixed_modes).detach()
    packed_mask = (
        valid.unsqueeze(-1).unsqueeze(-1)
        & spectral_slot_mask(period, device=valid.device).reshape(1, 1, 1, -1)
    ).expand(*packed.shape[:-1])
    return SpectralTDLambdaTargets(values=packed, masks=packed_mask.detach())


@torch.no_grad()
def spectral_target_scales(
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    latent_rms: torch.Tensor,
    scale_floor_ratio: float,
    period: int = SPECTRAL_PERIOD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact full-rollout H1 and per-frequency RMS units without a fake prediction."""
    if target.ndim != 4 or target.shape[:-1] != mask.shape:
        raise ValueError("target and mask must be [B,L,S,D] and [B,L,S]")
    if scale_floor_ratio <= 0.0:
        raise ValueError("scale_floor_ratio must be positive")
    num_layers = target.shape[1]
    if latent_rms.shape != (num_layers,):
        raise ValueError("latent_rms must contain one scalar per layer")
    target_h1, target_modes = split_spectral_slots(target.detach(), period)
    row_valid = mask[..., 0].bool()
    h1_weight = row_valid.to(target.dtype).unsqueeze(-1)
    h1_denominator = (
        row_valid.to(target.dtype).sum(dim=0) * target.shape[-1]
    ).clamp_min(1.0)
    safe_h1 = torch.where(
        row_valid.unsqueeze(-1), target_h1, torch.zeros_like(target_h1)
    )
    h1_scale = (
        safe_h1.square().mul(h1_weight).sum(dim=(0, 2)) / h1_denominator
    ).sqrt()
    h1_scale = h1_scale.clamp_min(
        latent_rms.detach().to(target) * scale_floor_ratio
    ).clamp_min(torch.finfo(target.dtype).eps)
    del safe_h1

    component_active = spectral_component_mask(period, device=target.device)
    mode_scales = []
    for frequency in range(period // 2 + 1):
        frequency_active = component_active[frequency].reshape(1, 1, 2)
        frequency_valid = row_valid.unsqueeze(-1) & frequency_active
        frequency_weight = frequency_valid.to(target.dtype).unsqueeze(-1)
        denominator = (
            row_valid.to(target.dtype).sum(dim=0)
            * component_active[frequency].to(target.dtype).sum()
            * target.shape[-1]
        ).clamp_min(1.0)
        frequency_target = target_modes[:, :, frequency]
        safe_frequency = torch.where(
            frequency_valid.unsqueeze(-1),
            frequency_target,
            torch.zeros_like(frequency_target),
        )
        frequency_scale = (
            safe_frequency.square().mul(frequency_weight).sum(dim=(0, 2, 3))
            / denominator
        ).sqrt()
        mode_scales.append(
            frequency_scale.clamp_min(
                latent_rms.detach().to(target) * scale_floor_ratio
            ).clamp_min(torch.finfo(target.dtype).eps)
        )
    return h1_scale.detach(), torch.stack(mode_scales, dim=-1).detach()


def spectral_td_lambda_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    latent_rms: torch.Tensor,
    scale_floor_ratio: float,
    h1_scale: Optional[torch.Tensor] = None,
    mode_scale: Optional[torch.Tensor] = None,
    period: int = SPECTRAL_PERIOD,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Equal-family, equal-layer/frequency normalized SmoothL1 objective."""
    if prediction.shape != target.shape or prediction.shape[:-1] != mask.shape:
        raise ValueError("prediction, target, and mask shapes are inconsistent")
    if prediction.ndim != 4:
        raise ValueError("minibatch spectral tensors must be [B,L,S,D]")
    if scale_floor_ratio <= 0.0:
        raise ValueError("scale_floor_ratio must be positive")
    num_layers = prediction.shape[1]
    if latent_rms.shape != (num_layers,):
        raise ValueError("latent_rms must contain one scalar per layer")
    if h1_scale is None or mode_scale is None:
        computed_h1_scale, computed_mode_scale = spectral_target_scales(
            target,
            mask,
            latent_rms=latent_rms,
            scale_floor_ratio=scale_floor_ratio,
            period=period,
        )
        h1_scale = computed_h1_scale if h1_scale is None else h1_scale
        mode_scale = computed_mode_scale if mode_scale is None else mode_scale
    if h1_scale.shape != (num_layers,):
        raise ValueError("h1_scale must contain one scalar per layer")
    expected_scale = (num_layers, period // 2 + 1)
    if mode_scale.shape != expected_scale:
        raise ValueError("mode_scale must contain one scalar per layer/frequency")
    h1_scale = h1_scale.detach().to(target).clamp_min(torch.finfo(target.dtype).eps)
    mode_scale = mode_scale.detach().to(target).clamp_min(torch.finfo(target.dtype).eps)

    pred_h1, pred_modes = split_spectral_slots(prediction, period)
    target_h1, target_modes = split_spectral_slots(target.detach(), period)
    row_valid = mask[..., 0].bool()
    h1_weight = row_valid.to(prediction.dtype).unsqueeze(-1)
    h1_denominator = (
        row_valid.to(prediction.dtype).sum(dim=0) * prediction.shape[-1]
    ).clamp_min(1.0)
    h1_valid = row_valid.unsqueeze(-1)
    h1_error = F.smooth_l1_loss(
        torch.where(h1_valid, pred_h1, torch.zeros_like(pred_h1))
        / h1_scale.reshape(1, num_layers, 1),
        torch.where(h1_valid, target_h1, torch.zeros_like(target_h1))
        / h1_scale.reshape(1, num_layers, 1),
        reduction="none",
    )
    h1_losses = (h1_error * h1_weight).sum(dim=(0, 2)) / h1_denominator

    component_active = spectral_component_mask(period, device=prediction.device)
    mode_valid = row_valid.unsqueeze(-1).unsqueeze(-1) & component_active.reshape(
        1, 1, period // 2 + 1, 2
    )
    mode_weight = mode_valid.to(prediction.dtype).unsqueeze(-1)
    mode_denominator = (
        row_valid.to(prediction.dtype).sum(dim=0).unsqueeze(-1)
        * component_active.to(prediction.dtype).sum(dim=-1).reshape(1, -1)
        * prediction.shape[-1]
    ).clamp_min(1.0)
    safe_prediction = torch.where(
        mode_valid.unsqueeze(-1), pred_modes, torch.zeros_like(pred_modes)
    )
    safe_target = torch.where(
        mode_valid.unsqueeze(-1), target_modes, torch.zeros_like(target_modes)
    )
    mode_error = F.smooth_l1_loss(
        safe_prediction / mode_scale.reshape(1, num_layers, -1, 1, 1),
        safe_target / mode_scale.reshape(1, num_layers, -1, 1, 1),
        reduction="none",
    )
    mode_losses = (
        mode_error.mul(mode_weight).sum(dim=(0, 3, 4)) / mode_denominator
    )
    loss = 0.5 * (h1_losses.mean() + mode_losses.mean())
    return (
        loss,
        h1_scale.detach(),
        mode_scale.detach(),
        h1_losses.detach(),
        mode_losses.detach(),
    )


@torch.no_grad()
def build_m_step_bootstrapped_spectral_returns(
    innovations: torch.Tensor,
    bootstrap_modes: torch.Tensor,
    transition_boundaries: torch.Tensor,
    transition_valids: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    max_steps: int = 64,
    rho: float = 0.99,
    period: int = SPECTRAL_PERIOD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact realized M-step return plus a frozen mode bootstrap on full paths."""
    if innovations.ndim != 4:
        raise ValueError("innovations must be [T,N,L,D]")
    expected_mode_shape = (
        *innovations.shape[:3],
        period // 2 + 1,
        2,
        innovations.shape[-1],
    )
    if bootstrap_modes.shape != expected_mode_shape:
        raise ValueError("bootstrap modes must be [T,N,L,F,2,D]")
    if transition_boundaries.shape != innovations.shape[:2] or (
        transition_valids.shape != innovations.shape[:2]
    ):
        raise ValueError("transition masks have the wrong shape")
    if row_indices.ndim != 1 or row_indices.dtype != torch.long:
        raise ValueError("row_indices must be a rank-one long tensor")
    if max_steps < 1 or not 0.0 <= rho < 1.0:
        raise ValueError("max_steps and rho are invalid")
    time_dim, num_envs, num_layers, latent_dim = innovations.shape
    torch._assert_async(
        torch.all((row_indices >= 0) & (row_indices < time_dim * num_envs)),
        "diagnostic row indices must lie inside the rollout",
    )
    start_time = torch.div(row_indices, num_envs, rounding_mode="floor")
    env_index = row_indices.remainder(num_envs)
    frequencies = spectral_frequencies(period, reference=innovations)
    result = innovations.new_zeros(
        (row_indices.numel(), num_layers, frequencies.numel(), 2, latent_dim)
    )
    alive = torch.ones_like(row_indices, dtype=torch.bool)
    realized_steps = torch.zeros_like(row_indices)
    for delay in range(max_steps):
        time_index = start_time + delay
        in_bounds = time_index < time_dim
        safe_time = time_index.clamp_max(time_dim - 1)
        step_valid = alive & in_bounds & transition_valids[safe_time, env_index].bool()
        eta = innovations[safe_time, env_index].detach()
        phase = frequencies * delay
        real = phase.cos().reshape(1, 1, -1, 1) * eta.unsqueeze(-2)
        imag = -phase.sin().reshape(1, 1, -1, 1) * eta.unsqueeze(-2)
        contribution = (rho**delay) * torch.stack((real, imag), dim=-2)
        result.add_(
            torch.where(
                step_valid.reshape(-1, 1, 1, 1, 1),
                contribution,
                torch.zeros_like(contribution),
            )
        )
        realized_steps.add_(step_valid.to(realized_steps.dtype))
        alive = (
            step_valid
            & ~transition_boundaries[safe_time, env_index].bool()
            & (time_index + 1 < time_dim)
        )
    # Only complete M-step ordinary paths are diagnostic oracles.  Append the
    # frozen SARSA successor tail at (s_{t+M},a_{t+M}); partial/boundary rows are
    # deliberately excluded instead of comparing an infinite predictor to a prefix.
    full_path = alive
    tail_time = (start_time + max_steps).clamp_max(time_dim - 1)
    tail = bootstrap_modes.detach()[tail_time, env_index]
    phase = frequencies * max_steps
    cosine = phase.cos().reshape(1, 1, -1, 1)
    sine = phase.sin().reshape(1, 1, -1, 1)
    tail_real = cosine * tail[..., 0, :] + sine * tail[..., 1, :]
    tail_imag = -sine * tail[..., 0, :] + cosine * tail[..., 1, :]
    rotated_tail = torch.stack((tail_real, tail_imag), dim=-2)
    result.add_(
        torch.where(
            full_path.reshape(-1, 1, 1, 1, 1),
            (rho**max_steps) * rotated_tail,
            torch.zeros_like(rotated_tail),
        )
    )
    result = mask_inactive_spectral_components(result, period)
    return result.detach(), full_path.detach(), realized_steps.detach()


@torch.no_grad()
def complex_mode_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    row_mask: torch.Tensor,
    period: int = SPECTRAL_PERIOD,
) -> dict[str, torch.Tensor]:
    """Per-layer/frequency complex fit, scale, and quadrature statistics."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("complex tensors must match [B,L,F,2,D]")
    if row_mask.shape != prediction.shape[:1]:
        raise ValueError("row mask must select the diagnostic batch")
    active = spectral_component_mask(period, device=prediction.device).reshape(
        1, 1, period // 2 + 1, 2, 1
    )
    valid = row_mask.reshape(-1, 1, 1, 1, 1) & active
    safe_prediction = torch.where(valid, prediction.detach(), torch.zeros_like(prediction))
    safe_target = torch.where(valid, target.detach(), torch.zeros_like(target))
    reduce_dims = (0, 3, 4)
    error_energy = (safe_prediction - safe_target).square().sum(dim=reduce_dims)
    target_energy = safe_target.square().sum(dim=reduce_dims)
    prediction_energy = safe_prediction.square().sum(dim=reduce_dims)
    dot = (safe_prediction * safe_target).sum(dim=reduce_dims)
    denominator = (
        row_mask.to(prediction.dtype).sum()
        * active.to(prediction.dtype).sum(dim=(0, 1, 3, 4))
        * prediction.shape[-1]
    ).clamp_min(1.0)
    prediction_imag_energy = safe_prediction[..., 1, :].square().sum(dim=(0, 3))
    target_imag_energy = safe_target[..., 1, :].square().sum(dim=(0, 3))
    quadrature_denominator = (
        row_mask.to(prediction.dtype).sum() * prediction.shape[-1]
    ).clamp_min(1.0)
    return {
        "nmse": error_energy / target_energy.clamp_min(1e-12),
        "cosine": dot
        / (prediction_energy.sqrt() * target_energy.sqrt()).clamp_min(1e-12),
        "target_rms": (target_energy / denominator).sqrt(),
        "prediction_rms": (prediction_energy / denominator).sqrt(),
        "prediction_quadrature_rms": (
            prediction_imag_energy / quadrature_denominator
        ).sqrt(),
        "target_quadrature_rms": (
            target_imag_energy / quadrature_denominator
        ).sqrt(),
        "prediction_quadrature_fraction": prediction_imag_energy
        / prediction_energy.clamp_min(1e-12),
        "target_quadrature_fraction": target_imag_energy
        / target_energy.clamp_min(1e-12),
    }


@torch.no_grad()
def layer_vector_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    row_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-layer fit statistics for the explicit H1 innovation anchor."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("H1 tensors must match [B,L,D]")
    if row_mask.shape != prediction.shape[:1]:
        raise ValueError("row mask must select the H1 diagnostic batch")
    valid = row_mask.reshape(-1, 1, 1)
    safe_prediction = torch.where(valid, prediction.detach(), torch.zeros_like(prediction))
    safe_target = torch.where(valid, target.detach(), torch.zeros_like(target))
    error_energy = (safe_prediction - safe_target).square().sum(dim=(0, 2))
    target_energy = safe_target.square().sum(dim=(0, 2))
    prediction_energy = safe_prediction.square().sum(dim=(0, 2))
    dot = (safe_prediction * safe_target).sum(dim=(0, 2))
    denominator = (
        row_mask.to(prediction.dtype).sum() * prediction.shape[-1]
    ).clamp_min(1.0)
    return {
        "nmse": error_energy / target_energy.clamp_min(1e-12),
        "cosine": dot
        / (prediction_energy.sqrt() * target_energy.sqrt()).clamp_min(1e-12),
        "target_rms": (target_energy / denominator).sqrt(),
        "prediction_rms": (prediction_energy / denominator).sqrt(),
    }


def deterministic_diagnostic_rows(
    batch_size: int,
    count: int,
    iteration: int,
    device: torch.device,
) -> torch.Tensor:
    """Deterministically spread one bounded diagnostic minibatch over a rollout."""
    if batch_size < 1 or count < 1 or iteration < 1:
        raise ValueError("diagnostic row dimensions must be positive")
    count = min(count, batch_size)
    stride = max(1, batch_size // count)
    while gcd(stride, batch_size) != 1:
        stride += 1
    offset = (iteration - 1) % stride
    return (offset + torch.arange(count, device=device) * stride).remainder(batch_size)


def next_state_bootstrap_rows(
    transition_terminations: torch.Tensor,
    transition_boundaries: torch.Tensor,
    transition_valids: torch.Tensor,
) -> torch.Tensor:
    """Rows whose tail state has no stored behavior action."""
    if not (
        transition_terminations.shape
        == transition_boundaries.shape
        == transition_valids.shape
    ):
        raise ValueError("transition masks must have matching shapes")
    if transition_terminations.ndim != 2 or transition_terminations.shape[0] == 0:
        raise ValueError("transition masks must be non-empty [time, env] tensors")
    terminations = transition_terminations.detach().bool()
    boundaries = transition_boundaries.detach().bool()
    valids = transition_valids.detach().bool()
    rows = valids & boundaries & ~terminations
    rows[-1] |= valids[-1] & ~boundaries[-1]
    return rows


@torch.no_grad()
def build_sparse_frozen_edge_predictions(
    agent,
    next_actor_features: torch.Tensor,
    sampled_next_actions: torch.Tensor,
    edge_rows: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Evaluate compact frozen all-layer spectral predictions at cutoffs."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if next_actor_features.ndim != 4 or sampled_next_actions.ndim != 3:
        raise ValueError("features and actions must be [time,env,layer,channel] and [time,env,action]")
    if next_actor_features.shape[:2] != edge_rows.shape:
        raise ValueError("edge rows must match the feature table")
    if sampled_next_actions.shape[:2] != edge_rows.shape:
        raise ValueError("edge rows must match the action table")
    if next_actor_features.shape[-2:] != (
        agent.tape_num_layers,
        agent.tape_feature_dim,
    ):
        raise ValueError("next features have the wrong layer or latent dimension")
    if sampled_next_actions.shape[-1] != agent.tape_action_dim:
        raise ValueError("next actions have the wrong action dimension")
    if not (
        next_actor_features.device
        == sampled_next_actions.device
        == edge_rows.device
    ):
        raise ValueError("edge prediction inputs must share one device")

    flat_rows = edge_rows.detach().bool().reshape(-1)
    row_indices = flat_rows.nonzero(as_tuple=False).squeeze(-1)
    if row_indices.numel() == 0:
        return next_actor_features.new_zeros(
            0,
            agent.tape_num_layers,
            agent.tape_num_slots,
            agent.tape_feature_dim,
        )
    selected_features = next_actor_features.reshape(
        flat_rows.numel(), agent.tape_num_layers, agent.tape_feature_dim
    ).index_select(0, row_indices)
    selected_actions = sampled_next_actions.reshape(
        flat_rows.numel(), agent.tape_action_dim
    ).index_select(0, row_indices)
    prediction_chunks = [
        target_tape_predictor_forward(agent, feature_chunk, action_chunk).detach()
        for feature_chunk, action_chunk in zip(
            selected_features.split(batch_size),
            selected_actions.split(batch_size),
            strict=True,
        )
    ]
    return torch.cat(prediction_chunks).detach()


@torch.no_grad()
def latent_scale_and_participation_rank(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS and covariance participation rank expose latent gauge concentration."""
    if features.ndim < 2:
        raise ValueError("features must have a sample and latent dimension")
    flat = features.detach().reshape(-1, features.shape[-1]).float()
    scale = flat.square().mean().sqrt()
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, flat.shape[0])
    trace = covariance.diagonal().sum()
    participation = trace.square() / covariance.square().sum().clamp_min(1e-20)
    return scale, participation


@torch.no_grad()
def actor_head_weight_norm(agent) -> torch.Tensor:
    """Packed decoder norm catches compensating latent/decoder rescaling."""
    if agent.actor_dist == "gaussian":
        heads = (agent.actor_head, agent.actor_logvar_head)
    else:
        heads = (agent.actor_alpha_head, agent.actor_beta_head)
    return torch.cat(
        [parameter.detach().reshape(-1) for head in heads for parameter in head.parameters()]
    ).float().norm()


@torch.no_grad()
def target_encoder_functional_drift(
    agent, observations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute and relative feature drift before the rollout hard snapshot."""
    if observations.ndim != 2 or observations.shape[0] < 1:
        raise ValueError("drift observations must be a nonempty flat batch")
    live = agent.get_actor_feat(observations).detach()
    frozen = agent.target_actor_feature(observations).detach()
    absolute = (live - frozen).square().mean().sqrt()
    reference = frozen.square().mean().sqrt().clamp_min(1e-12)
    return absolute, absolute / reference



def summarize_episode_tail_risk(
    returns: Sequence[float], thresholds: Sequence[float] = (1500.0, 5000.0)
) -> dict[str, float]:
    """Deterministic rolling lower-tail statistics; never touches training RNG."""
    values = np.asarray(tuple(returns), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("episodic returns must be scalar")
    if values.size == 0:
        result = {
            "window_size": 0.0,
            "median": 0.0,
            "bottom_5pct_mean": 0.0,
            "cvar_5pct": 0.0,
            "below_half_window_median_count": 0.0,
            "below_half_window_median_fraction": 0.0,
        }
        for threshold in thresholds:
            tag = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
            result[f"below_{tag}_count"] = 0.0
            result[f"below_{tag}_fraction"] = 0.0
        return result
    tail_count = max(1, ceil(0.05 * values.size))
    lower_tail = np.partition(values, tail_count - 1)[:tail_count]
    median = float(np.median(values))
    below_half_median_count = int(np.count_nonzero(values < 0.5 * median))
    result = {
        "window_size": float(values.size),
        "median": median,
        "bottom_5pct_mean": float(lower_tail.mean()),
        "cvar_5pct": float(lower_tail.mean()),
        "below_half_window_median_count": float(below_half_median_count),
        "below_half_window_median_fraction": float(
            below_half_median_count / values.size
        ),
    }
    for threshold in thresholds:
        tag = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
        count = int(np.count_nonzero(values < threshold))
        result[f"below_{tag}_count"] = float(count)
        result[f"below_{tag}_fraction"] = float(count / values.size)
    return result


@torch.no_grad()
def clip_grad_norm_async_fail_loud_(parameters, max_norm, norm_type=2.0):
    """Clip exactly as PyTorch does and enqueue the finite check on-device."""
    total_norm = nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=False,
    )
    torch._assert_async(
        torch.isfinite(total_norm),
        "The total gradient norm is non-finite; refusing the optimizer step",
    )
    return total_norm


@torch.no_grad()
def synchronize_scalar_telemetry(statistics):
    """Materialize CUDA scalar telemetry with one packed device-to-host copy."""
    if not statistics:
        return {}
    names = tuple(statistics)
    scalars = tuple(statistics.values())
    if not all(torch.is_tensor(value) for value in scalars):
        raise TypeError("telemetry values must be tensors")
    if not all(value.numel() == 1 for value in scalars):
        raise ValueError("telemetry values must be scalar tensors")
    if len({value.device for value in scalars}) != 1:
        raise ValueError("telemetry values must share one device")
    host_values = torch.stack(
        [value.detach().reshape(()) for value in scalars]
    ).cpu().tolist()
    return dict(zip(names, host_values))


@torch.no_grad()
def retain_graph_output(output, *, compiled):
    """Detach and clone output whose CUDA-graph storage will be replayed."""
    if not torch.is_tensor(output):
        raise TypeError("retained graph output must be a tensor")
    return output.detach().clone() if compiled else output.detach()


@torch.no_grad()
def update_scalar_max_(maximum: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Asynchronously accumulate a detached scalar maximum on its device."""
    if maximum.numel() != 1 or value.numel() != 1:
        raise ValueError("max telemetry inputs must be scalar")
    maximum.copy_(torch.maximum(maximum, value.detach().to(maximum)))
    return maximum


def value_support_bounds(args):
    """Support bounds in the coordinate used for categorical bins."""
    return args.v_min, args.v_max


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    # SOFT-ADVANTAGE max-ent (gaussian only; --auto-entropy). Entropy enters the POLICY
    # ADVANTAGE only, NEVER the critic's target. The critic regresses to the RAW reward
    # return (fits the fixed support [v_min,v_max]); a SEPARATE soft-advantage GAE adds the
    # bootstrap bonus b_t = α·H_sq(s_{t+1}), estimated as the single-sample SQUASHED entropy
    # -logπ_sq(a|s), matching SAC's next_state_log_pi units. Rationale: forcing the bounded
    # categorical critic to learn the soft value both wastes capacity and overflows the
    # support (the softboot failure: edge_mass→0.9, expl_var→0). alpha is
    # auto-tuned by SAC's exact dual (target_entropy = -|A|; alpha_loss = -α(logπ + tgt)).
    # The explicit α·H actor bonus supplies the CURRENT-step entropy gradient (the soft
    # advantage's current-state entropy is action-independent => cancels in the PG term);
    # the soft advantage supplies FUTURE-entropy credit. Works WITH rankgauss: the soft
    # value reorders advantages and rankgauss preserves order/sign (magnitude is incidental).
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    # NOTE: target_kl epoch-stop would starve the (always-on) critic in this pure
    # variant; default None — the actor is leashed by tpo_kl_breaker instead.
    target_kl: Optional[float] = None

    # TPO mirror descent: probe-scored TPO target with MPO-style adaptive
    # temperature REPLACES the PPO surrogate. Probes run at EVERY rollout state.
    tpo_coef: float = 1.0        # weight of the TPO CE (the entire actor loss besides entropy); must be > 0
    tpo_eta: float = 6.0         # FIXED temperature, used only when tpo_adaptive_eta=False
    tpo_k: int = 8               # candidates per state (ALL probed, incl. the executed action as candidate 0)
    tpo_sigma_scale_coef: float = 1.0  # global score sigma = coef * EMA(one-step TD-residual RMS)
    tpo_eps: float = 0.03        # trust-region CAP / max KL per update (dyn-trust) OR fixed KL target (v1 mode)
    tpo_adaptive_eta: bool = True      # solve eta s.t. mean KL(p_old||q)=tpo_eps; False => fixed tpo_eta
    tpo_dyn_trust: bool = True   # one-sided KL cap on a fixed base temperature (v5 default). False => exact tpomd_v1 fixed-target dual
    tpo_eta_base: float = 1.0    # base temperature for the dynamic-cap path; natural KL at this eta is the signal-determined step (unused when tpo_dyn_trust=False)
    tpo_kl_breaker: float = 0.09 # actor circuit breaker: stop actor epochs when epoch-mean approx_kl exceeds (3x eps)

    # Union chassis: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # v149-aligned distributional critic support. Bounds are already symlog
    # coordinates for raw-return support [-20000, 20000].
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 0.5  # requested sharper HL-Gauss projection sigma

    # Raw-return ablation: keep observations as in the source, but do not divide
    # rewards by NormalizeReward's running discounted-return std and do not clip
    # raw rewards before GAE.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "rankgauss"

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "minibatch"

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    sigma_floor_bins: float = 2.0    # floor sigma(s) at this many bins (avoid /~0)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # Graph-free all-layer successor-innovation TD(lambda).  H1 is explicit; each
    # complex rFFT mode receives a 16-step forward-view target with rho discount.
    tape: bool = True
    tape_period: int = SPECTRAL_PERIOD
    tape_rho: float = 0.99
    tape_lambda: float = 0.95
    tape_slot_embed_dim: int = 16
    tape_predictor_hidden: int = 64
    tape_coef: float = 1.0
    tape_scale_floor_ratio: float = 1e-3
    tape_trunk_grad_clip: float = 0.025
    tape_predictor_grad_clip: float = 0.25
    tape_target_batch_size: int = 8192
    tape_diagnostic_rows: int = 1024

    # Non-learning tail-risk telemetry; configurable for Hopper/Walker thresholds.
    tail_risk_window: int = 512
    tail_risk_thresholds: tuple[float, ...] = (1500.0, 5000.0)

    # reduce-overhead enables CUDA graphs for every static neural forward. Raw MuJoCo
    # probes and all TPO control flow remain eager and behavior-equivalent to v13.
    compile: bool = False
    compile_mode: str = "reduce-overhead"

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, normalize_reward, clip_reward):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        clipped_obs_space = gym.spaces.Box(
            low=np.full(env.observation_space.shape, -10.0, dtype=env.observation_space.dtype),
            high=np.full(env.observation_space.shape, 10.0, dtype=env.observation_space.dtype),
            dtype=env.observation_space.dtype,
        )
        try:
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.clip(obs, -10, 10),
                observation_space=clipped_obs_space,
            )
        except TypeError:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def find_wrapper(env, wrapper_type):
    # Walk the .env wrapper chain looking for wrapper_type.
    cur = env
    while cur is not None:
        if isinstance(cur, wrapper_type):
            return cur
        cur = getattr(cur, "env", None)
    return None


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


class IndexedTransferBranch(nn.Module):
    def __init__(self, H, history_dim):
        super().__init__()
        if history_dim % H != 0:
            raise ValueError(f"history_dim={history_dim} must be divisible by H={H}")
        self.H = H
        self.history_slots = history_dim // H
        self.current_linear = layer_init(nn.Linear(H, H))
        self.act = ReLUSquared()
        self.out_linear = layer_init(nn.Linear(H, H))
        self.history_weight = nn.Parameter(torch.empty(self.history_slots, H))
        nn.init.normal_(self.history_weight, mean=0.0, std=np.sqrt(2.0 / (H + self.history_slots)))

    def forward(self, x, history):
        preact = self.current_linear(x)
        history = history.reshape(history.shape[0], self.history_slots, self.H)
        same_index_transfer = (history * self.history_weight.to(dtype=history.dtype).unsqueeze(0)).sum(dim=1)
        return self.out_linear(self.act(preact + same_index_transfer))


class ThinkBlock(nn.Module):
    """Bounded convex residual mix of x and x0, then parallel dense + soft MoE."""

    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.n_experts = n_experts

        self.in_proj = layer_init(nn.Linear(in_dim, H))
        # Bounded convex residual gate: g = sigmoid(resid_gate) per channel.
        # init +4 → g ≈ 0.982 → x_in ≈ x at start.
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        # Dense branch. RMSNorm without learnable affine (no free per-channel γ).
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = IndexedTransferBranch(H, in_dim)

        # Soft MoE branch (softmax over all experts, no top-K).
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([IndexedTransferBranch(H, in_dim) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in), cat_feats)        # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = torch.stack([e(m_in, cat_feats) for e in self.experts], dim=1)  # (B, E, H)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)          # (B, H)

        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            block_in_dim = H * (k + 1)
            self.blocks.append(ThinkBlock(block_in_dim, H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward_all(self, x):
        x0 = self.entry(x)
        features = [x0]
        for block in self.blocks:
            features.append(block(torch.cat(features, dim=-1), x0))
        output = self.out_proj(self.out_norm(torch.cat(features, dim=-1)))
        return torch.stack((*features, output), dim=-2)

    def forward(self, x):
        x0 = self.entry(x)
        features = [x0]
        for block in self.blocks:
            features.append(block(torch.cat(features, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(features, dim=-1)))


class SpectralResidualPredictor(nn.Module):
    """One weight-shared network over H1 and explicit complex Fourier slots."""

    def __init__(
        self,
        feature_dim,
        action_dim,
        period,
        embed_dim,
        hidden_dim,
    ):
        super().__init__()
        if min(feature_dim, action_dim, period, embed_dim, hidden_dim) < 1:
            raise ValueError("spectral predictor dimensions must be positive")
        if period % 2:
            raise ValueError("the spectral period must be even")
        self.feature_dim = feature_dim
        self.period = period
        self.num_frequencies = period // 2 + 1
        self.num_slots = 1 + 2 * self.num_frequencies
        self.slot_embedding = nn.Embedding(self.num_slots, embed_dim)
        self.source = layer_init(nn.Linear(feature_dim + action_dim, hidden_dim))
        self.conditioned = layer_init(nn.Linear(hidden_dim + embed_dim, hidden_dim))
        self.output = layer_init(nn.Linear(hidden_dim, feature_dim), std=0.01)
        self.activation = ReLUSquared()
        self.register_buffer(
            "slot_indices", torch.arange(self.num_slots), persistent=False
        )
        self.register_buffer(
            "active_slot_mask",
            spectral_slot_mask(period).reshape(1, self.num_slots, 1),
            persistent=False,
        )
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()

    def forward(self, actor_feature, action):
        if actor_feature.ndim != 2 or action.ndim != 2:
            raise ValueError("actor feature and action must be rank-two batches")
        if actor_feature.shape[0] != action.shape[0]:
            raise ValueError("actor feature and action batch sizes differ")
        base = self.activation(self.source(torch.cat((actor_feature, action), dim=-1)))
        slots = self.slot_embedding(self.slot_indices)
        base = base.unsqueeze(1).expand(-1, self.num_slots, -1)
        slots = slots.unsqueeze(0).expand(base.shape[0], -1, -1)
        conditioned = self.activation(self.conditioned(torch.cat((base, slots), dim=-1)))
        prediction = self.output(conditioned)
        return torch.where(
            self.active_slot_mask, prediction, torch.zeros_like(prediction)
        )


class AllLayerSpectralResidualPredictor(nn.Module):
    """Independent spectral successor-innovation predictor at every trunk depth."""

    def __init__(
        self,
        num_layers,
        feature_dim,
        action_dim,
        period,
        embed_dim,
        hidden_dim,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("the all-layer predictor needs at least one layer")
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.period = period
        self.num_slots = 1 + 2 * (period // 2 + 1)
        self.predictors = nn.ModuleList(
            [
                SpectralResidualPredictor(
                    feature_dim,
                    action_dim,
                    period,
                    embed_dim,
                    hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, actor_features, action):
        if actor_features.ndim != 3 or action.ndim != 2:
            raise ValueError(
                "all-layer features and actions must be [B,L,D] and [B,A]"
            )
        if actor_features.shape[0] != action.shape[0]:
            raise ValueError("all-layer feature and action batch sizes differ")
        if actor_features.shape[1:] != (self.num_layers, self.feature_dim):
            raise ValueError("all-layer feature channels do not match the predictor")
        predictions = [
            predictor(actor_features[:, layer], action)
            for layer, predictor in enumerate(self.predictors)
        ]
        return torch.stack(predictions, dim=1)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.tape_enabled = args.tape
        self.tape_period = args.tape_period
        self.tape_num_slots = 1 + 2 * (args.tape_period // 2 + 1)
        self.tape_feature_dim = H
        self.tape_num_layers = args.k_blocks + 2
        self.tape_layer_names = (
            "entry",
            *(f"block_{index + 1}" for index in range(args.k_blocks)),
            "output",
        )
        self.tape_action_dim = act_dim
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # v149 critic readout style, without MTP: biasless HL-Gauss value head.
        self.num_bins = args.num_bins
        self.critic_head = layer_init(nn.Linear(H, args.num_bins, bias=False), std=0.1)
        with torch.no_grad():
            self.critic_head.weight.zero_()
        # v24: action distribution. Both parameterizations are dreamer4-faithful;
        # the Gaussian path is tanh-squashed like SAC but uses log-variance, not log_std.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            # dreamer4 Gaussian: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # dreamer4 unimodal Beta: two concentration heads, alpha,beta = 1 + softplus.
            self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")
        # Isolate auxiliary initialization from the task RNG stream. This preserves
        # every base parameter and the post-Agent RNG state of TPO-MD v5 exactly.
        if self.tape_enabled:
            with torch.random.fork_rng(devices=[]):
                self.tape_predictor = AllLayerSpectralResidualPredictor(
                    self.tape_num_layers,
                    H,
                    act_dim,
                    args.tape_period,
                    args.tape_slot_embed_dim,
                    args.tape_predictor_hidden,
                )
                self.target_tape_predictor = copy.deepcopy(self.tape_predictor)
            # A coherent hard target bundle. Deepcopy consumes no RNG and every component
            # is synchronized together after collection, before labels or optimizer work.
            live_actor_trunk = self.trunk if self.share_backbone else self.actor_trunk
            self.target_actor_trunk = copy.deepcopy(live_actor_trunk)
            if self.actor_dist == "gaussian":
                self.target_actor_head = copy.deepcopy(self.actor_head)
                self.target_actor_logvar_head = copy.deepcopy(self.actor_logvar_head)
            else:
                self.target_actor_alpha_head = copy.deepcopy(self.actor_alpha_head)
                self.target_actor_beta_head = copy.deepcopy(self.actor_beta_head)
            for module in self.target_bundle_modules():
                module.requires_grad_(False)

    def _actor_dist(self, actor_feat):
        # Build the action distribution and the native-space transforms.
        # Returns (dist, to_action, log_det_fn) where:
        #   to_action(z): map a NATIVE sample z to the env action.
        #   log_det_fn(z): per-sample log|d action / d z| correction to SUBTRACT
        #                  from dist.log_prob(z) (0 where the map is volume-constant).
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
            # dreamer4 soft tanh-rescale bound on log-variance (smooth, no dead grad).
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            std = (0.5 * lv).exp()
            dist = Normal(mean, std)
            to_action = torch.tanh
            log_det_fn = lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
            return dist, to_action, log_det_fn
        # beta
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def _actor_dist_frozen_head(self, actor_feat):
        """Decode a latent in policy geometry without updating policy-head weights."""
        if self.actor_dist == "gaussian":
            mean = F.linear(
                actor_feat,
                self.actor_head.weight.detach(),
                None if self.actor_head.bias is None else self.actor_head.bias.detach(),
            )
            raw_lv = F.linear(
                actor_feat,
                self.actor_logvar_head.weight.detach(),
                None
                if self.actor_logvar_head.bias is None
                else self.actor_logvar_head.bias.detach(),
            )
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            return (
                Normal(mean, (0.5 * lv).exp()),
                torch.tanh,
                lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z)),
            )
        alpha_raw = F.linear(
            actor_feat,
            self.actor_alpha_head.weight.detach(),
            None
            if self.actor_alpha_head.bias is None
            else self.actor_alpha_head.bias.detach(),
        )
        beta_raw = F.linear(
            actor_feat,
            self.actor_beta_head.weight.detach(),
            None
            if self.actor_beta_head.bias is None
            else self.actor_beta_head.bias.detach(),
        )
        return (
            Beta(1.0 + F.softplus(alpha_raw), 1.0 + F.softplus(beta_raw)),
            lambda z: self.action_low + (self.action_high - self.action_low) * z,
            lambda z: 0.0,
        )

    def _trunks(self, x):
        # Return output-layer actor/critic features for the unchanged task path.
        if self.share_backbone:
            feature = self.trunk(x)
            return feature, feature
        return self.actor_trunk(x), self.critic_trunk(x)

    def _trunks_all(self, x):
        # Expose [entry, block_1..block_K, output] without a second actor pass.
        if self.share_backbone:
            actor_features = self.trunk.forward_all(x)
            return actor_features, actor_features[..., -1, :]
        return self.actor_trunk.forward_all(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns value LOGITS (B, num_bins); caller converts via support.
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

    def get_actor_feat(self, x):
        return self._trunks_all(x)[0]

    def get_action_and_value(
        self, x, z=None, candidate_zs=None, return_dist=False, return_actor_feat=False
    ):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (z-replay, generalized).
        # TPO extensions (both default-off => base behavior/graph/RNG untouched):
        #   candidate_zs (B, K, A): also return per-candidate logprobs (B, K) from
        #     the SAME dist (one trunk forward, consumes no RNG — log_prob only,
        #     evaluated AFTER the gaussian entropy rsample so the RNG order of the
        #     base computation is preserved).
        #   return_dist: also return (dist, to_action, log_det_fn) so the rollout
        #     can sample probe candidates from the already-constructed dist.
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_logits = self.critic_head(critic_feat)
        if self.actor_dist == "gaussian":
            # Reparameterized SQUASHED-entropy estimate H_sq = E_ε[-logπ_sq(tanh(μ+σε))].
            # Base-Normal H = dist.entropy() is monotone↑ in σ, so an entropy bonus rails σ
            # to the ceiling -> tanh saturates -> squashed H collapses, while the α-dual
            # (which targets squashed H) cranks α up: a runaway. The squashed H is BOUNDED
            # with an interior max in σ, so maximizing it settles σ at a finite optimum and
            # is consistent with the α target. Fresh rsample => gradient flows to μ,σ
            # (independent of the replayed z used for the PPO ratio).
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        out = (action, z, log_prob, entropy, value_logits)
        if candidate_zs is not None:
            # Evaluate as (K, B, A) so the dist's (B, A) batch shape broadcasts
            # over the K axis, then transpose back to (B, K).
            cz = candidate_zs.transpose(0, 1)
            candidate_log_probs = (dist.log_prob(cz) - log_det_fn(cz)).sum(-1).transpose(0, 1)
            out = out + (candidate_log_probs,)
        if return_dist:
            out = out + (dist, to_action, log_det_fn)
        if return_actor_feat:
            out = out + (actor_feat,)
        return out

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk). The two
        # distribution heads are clipped together as one actor group (2-way
        # decoupled clip; no separate std budget — gaussian's variance head and
        # both beta concentration heads sit in the same group).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())

    def tape_trunk_blocks(self):
        """Logical actor-trunk blocks used for global-clip delivery diagnostics."""
        if not self.tape_enabled:
            return []
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        blocks = [[*trunk.entry.parameters()]]
        blocks.extend([list(block.parameters()) for block in trunk.blocks])
        blocks.append([*trunk.out_proj.parameters()])
        return blocks

    def tape_trunk_parameters(self):
        return [parameter for block in self.tape_trunk_blocks() for parameter in block]

    def tape_predictor_parameters(self):
        return list(self.tape_predictor.parameters()) if self.tape_enabled else []

    def predict_tapes(self, actor_feature, action):
        return self.tape_predictor(actor_feature, action)

    def predict_target_tapes(self, actor_feature, action):
        return self.target_tape_predictor(actor_feature, action)

    def target_bundle_modules(self):
        if not self.tape_enabled:
            return ()
        modules = [self.target_actor_trunk, self.target_tape_predictor]
        if self.actor_dist == "gaussian":
            modules.extend((self.target_actor_head, self.target_actor_logvar_head))
        else:
            modules.extend((self.target_actor_alpha_head, self.target_actor_beta_head))
        return tuple(modules)

    def target_actor_feature(self, observations):
        if not self.tape_enabled:
            raise RuntimeError("the spectral TD(lambda) tape is disabled")
        return self.target_actor_trunk.forward_all(observations)

    def _target_actor_dist(self, actor_feature):
        if not self.tape_enabled:
            raise RuntimeError("the spectral TD(lambda) tape is disabled")
        if self.actor_dist == "gaussian":
            mean = self.target_actor_head(actor_feature)
            raw_lv = self.target_actor_logvar_head(actor_feature)
            logvar = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            return Normal(mean, (0.5 * logvar).exp()), torch.tanh
        alpha = 1.0 + F.softplus(self.target_actor_alpha_head(actor_feature))
        beta = 1.0 + F.softplus(self.target_actor_beta_head(actor_feature))
        return (
            Beta(alpha, beta),
            lambda sample: self.action_low
            + (self.action_high - self.action_low) * sample,
        )

    @torch.no_grad()
    def snapshot_tape_target(self):
        if not self.tape_enabled:
            return
        live_actor_trunk = self.trunk if self.share_backbone else self.actor_trunk
        self.target_actor_trunk.load_state_dict(live_actor_trunk.state_dict())
        self.target_tape_predictor.load_state_dict(self.tape_predictor.state_dict())
        if self.actor_dist == "gaussian":
            self.target_actor_head.load_state_dict(self.actor_head.state_dict())
            self.target_actor_logvar_head.load_state_dict(
                self.actor_logvar_head.state_dict()
            )
        else:
            self.target_actor_alpha_head.load_state_dict(
                self.actor_alpha_head.state_dict()
            )
            self.target_actor_beta_head.load_state_dict(
                self.actor_beta_head.state_dict()
            )

    @torch.no_grad()
    def target_snapshot_lag(self):
        if not self.tape_enabled:
            return next(self.parameters()).new_zeros(())
        live_actor_trunk = self.trunk if self.share_backbone else self.actor_trunk
        pairs = [
            (live_actor_trunk, self.target_actor_trunk),
            (self.tape_predictor, self.target_tape_predictor),
        ]
        if self.actor_dist == "gaussian":
            pairs.extend(
                (
                    (self.actor_head, self.target_actor_head),
                    (self.actor_logvar_head, self.target_actor_logvar_head),
                )
            )
        else:
            pairs.extend(
                (
                    (self.actor_alpha_head, self.target_actor_alpha_head),
                    (self.actor_beta_head, self.target_actor_beta_head),
                )
            )
        lag = next(self.parameters()).new_zeros(())
        for live, target in pairs:
            for live_parameter, target_parameter in zip(
                live.parameters(), target.parameters(), strict=True
            ):
                lag = torch.maximum(
                    lag, (live_parameter.detach() - target_parameter.detach()).abs().max()
                )
        return lag

    def tape_parameters(self):
        return self.tape_trunk_parameters() + self.tape_predictor_parameters()

    def task_parameters(self):
        predictor_ids = {
            id(parameter) for parameter in self.tape_predictor_parameters()
        }
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
            and id(parameter) not in predictor_ids
        ]


def policy_model_forward(agent, observations):
    """Static task path, exposing all actor representations only when enabled."""
    if agent.tape_enabled:
        actor_features, critic_feat = agent._trunks_all(observations)
        actor_output = actor_features[..., -1, :]
    else:
        actor_output, critic_feat = agent._trunks(observations)
        actor_features = actor_output
    value_logits = agent.critic_head(critic_feat)
    if agent.actor_dist == "gaussian":
        first = agent.actor_head(actor_output)
        raw_lv = agent.actor_logvar_head(actor_output)
        second = rescale(
            (raw_lv / (agent.logvar_max - agent.logvar_min)).tanh(),
            (-1.0, 1.0),
            (agent.logvar_min, agent.logvar_max),
        )
    else:
        first = 1.0 + F.softplus(agent.actor_alpha_head(actor_output))
        second = 1.0 + F.softplus(agent.actor_beta_head(actor_output))
    return actor_features, value_logits, first, second


def action_value_from_policy_outputs(
    agent,
    model_outputs,
    z=None,
    candidate_zs=None,
):
    """Apply v13's exact eager sampling/log-prob order to compiled model outputs."""
    actor_features, value_logits, first, second = model_outputs
    if agent.actor_dist == "gaussian":
        dist = Normal(first, (0.5 * second).exp())
        to_action = torch.tanh
        log_det_fn = lambda sample: 2.0 * (
            log(2.0) - sample - F.softplus(-2.0 * sample)
        )
    else:
        dist = Beta(first, second)
        to_action = lambda sample: agent.action_low + (
            agent.action_high - agent.action_low
        ) * sample
        log_det_fn = lambda sample: 0.0
    if z is None:
        z = dist.sample()
        if agent.actor_dist == "beta":
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    action = to_action(z)
    log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
    if agent.actor_dist == "gaussian":
        zr = dist.rsample()
        entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
    else:
        entropy = dist.entropy().sum(1)
    out = (action, z, log_prob, entropy, value_logits)
    if candidate_zs is not None:
        candidate_transposed = candidate_zs.transpose(0, 1)
        candidate_log_probs = (
            dist.log_prob(candidate_transposed) - log_det_fn(candidate_transposed)
        ).sum(-1).transpose(0, 1)
        out = out + (candidate_log_probs,)
    return out + (actor_features,), dist, to_action, log_det_fn


def value_forward(agent, observations):
    """Static value-logit wrapper used by transition and probe bootstraps."""
    return agent.get_value(observations)


def target_actor_feat_forward(agent, observations):
    """Static hard-snapshot encoder exposing [B,L,D]."""
    return agent.target_actor_feature(observations)


def tape_predictor_forward(agent, actor_feature, action):
    """Static live predictor path; gradients may flow into the source feature."""
    return agent.predict_tapes(actor_feature, action)


def tape_auxiliary_update_forward(
    agent,
    actor_feature,
    action,
    target,
    mask,
    latent_rms,
    h1_scale,
    mode_scale,
    *,
    scale_floor_ratio,
    period,
):
    """Single static predictor+loss graph retained until its immediate backward."""
    prediction = agent.predict_tapes(actor_feature, action)
    loss, _, _, h1_losses, mode_losses = spectral_td_lambda_loss(
        prediction,
        target,
        mask,
        latent_rms=latent_rms,
        scale_floor_ratio=scale_floor_ratio,
        h1_scale=h1_scale,
        mode_scale=mode_scale,
        period=period,
    )
    return loss, h1_losses, mode_losses


def target_tape_predictor_forward(agent, actor_feature, action):
    """Static rollout-frozen tail predictor path."""
    return agent.predict_target_tapes(actor_feature, action)


@torch.no_grad()
def frozen_policy_sample_action_isolated(
    agent,
    actor_features: torch.Tensor,
    *,
    seed: int,
    cpu_rng_state: Optional[torch.Tensor] = None,
    device_rng_state: Optional[torch.Tensor] = None,
):
    """Draw one unbiased frozen-policy action without advancing the main RNG.

    The returned states form a persistent auxiliary stream. A single Monte Carlo
    action is sufficient for an unbiased estimate of the frozen-policy action
    expectation in a tape tail.
    """
    if actor_features.ndim != 2 or actor_features.shape[-1] != agent.tape_feature_dim:
        raise ValueError("frozen policy actions require output features [B,D]")
    device = actor_features.device
    uses_cuda = device.type == "cuda"
    if cpu_rng_state is None:
        if device_rng_state is not None:
            raise ValueError("device RNG state requires a matching CPU RNG state")
        initial_cpu_generator = torch.Generator(device="cpu")
        initial_cpu_generator.manual_seed(seed)
        cpu_rng_state = initial_cpu_generator.get_state()
        if uses_cuda:
            initial_device_generator = torch.Generator(device=device)
            initial_device_generator.manual_seed(seed)
            device_rng_state = initial_device_generator.get_state()
    elif uses_cuda and device_rng_state is None:
        raise ValueError("CUDA sampling requires both persistent RNG states")
    elif not uses_cuda and device_rng_state is not None:
        raise ValueError("CPU sampling does not accept a CUDA RNG state")

    fork_devices = [device] if uses_cuda else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.set_rng_state(cpu_rng_state)
        if uses_cuda:
            torch.cuda.set_rng_state(device_rng_state, device)
        dist, to_action = agent._target_actor_dist(actor_features)
        native_sample = dist.sample()
        if agent.actor_dist == "beta":
            native_sample = native_sample.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        next_cpu_rng_state = torch.get_rng_state()
        next_device_rng_state = (
            torch.cuda.get_rng_state(device) if uses_cuda else None
        )
    return (
        to_action(native_sample).detach(),
        next_cpu_rng_state,
        next_device_rng_state,
    )


@torch.no_grad()
def build_sparse_frozen_next_action_table(
    agent,
    next_actor_features: torch.Tensor,
    bootstrap_rows: torch.Tensor,
    *,
    seed: int,
    cpu_rng_state: Optional[torch.Tensor] = None,
    device_rng_state: Optional[torch.Tensor] = None,
):
    """Sample edge actions from the coherent target policy's output layer only."""
    if (
        bootstrap_rows.ndim != 2
        or next_actor_features.shape[:2] != bootstrap_rows.shape
    ):
        raise ValueError("next features and bootstrap rows must share [time, env]")
    if next_actor_features.ndim != 4 or next_actor_features.shape[-2:] != (
        agent.tape_num_layers,
        agent.tape_feature_dim,
    ):
        raise ValueError("next features have the wrong layer or latent dimension")
    if next_actor_features.device != bootstrap_rows.device:
        raise ValueError("next features and bootstrap rows must share a device")

    flat_rows = bootstrap_rows.detach().bool().reshape(-1)
    row_indices = flat_rows.nonzero(as_tuple=False).squeeze(-1)
    full_actions = next_actor_features.new_zeros(
        (flat_rows.numel(), agent.tape_action_dim)
    )
    if row_indices.numel() == 0:
        return (
            full_actions.reshape(*bootstrap_rows.shape, agent.tape_action_dim),
            cpu_rng_state,
            device_rng_state,
        )

    flat_output_features = next_actor_features[..., -1, :].reshape(
        flat_rows.numel(), agent.tape_feature_dim
    )
    selected_actor_features = flat_output_features.index_select(0, row_indices)
    selected_actions, cpu_rng_state, device_rng_state = (
        frozen_policy_sample_action_isolated(
            agent,
            selected_actor_features,
            seed=seed,
            cpu_rng_state=cpu_rng_state,
            device_rng_state=device_rng_state,
        )
    )
    full_actions.index_copy_(0, row_indices, selected_actions)
    return (
        full_actions.reshape(*bootstrap_rows.shape, agent.tape_action_dim).detach(),
        cpu_rng_state,
        device_rng_state,
    )


def tpo_restricted_target(anchor_logp, score_signal, eta):
    """TPO-MD v5's anchored K-action mirror-descent target."""
    return torch.softmax(anchor_logp + score_signal / eta, dim=-1)


def tpo_reverse_kl(anchor_logp, score_signal, eta):
    """Batch mean KL(p_old || q_eta), the v5 one-sided-cap statistic."""
    p_old = anchor_logp.exp()
    log_q = F.log_softmax(anchor_logp + score_signal / eta, dim=-1)
    return (p_old * (anchor_logp - log_q)).sum(-1).mean()


@torch.no_grad()
def capture_gradients(parameters):
    """Clone the current (already clipped) gradients as a sparse parameter map."""
    return {
        parameter: parameter.grad.detach().clone()
        for parameter in parameters
        if parameter.grad is not None
    }


def tape_source_feature(actor_feature, *, trunk_active):
    """Root TAPE at the trunk only while policy updates are permitted."""
    return actor_feature if trunk_active else actor_feature.detach()


@torch.no_grad()
def tape_gradient_telemetry(parameters, auxiliary, actor, critic):
    """Pack four sparse maps once for end-of-rollout norm/cosine diagnostics."""
    parameters = list(parameters)
    if not parameters:
        raise ValueError("at least one parameter is required")

    def packed(mapping):
        return torch.cat(
            [
                torch.zeros_like(parameter).reshape(-1)
                if mapping.get(parameter) is None
                else mapping[parameter].detach().reshape(-1)
                for parameter in parameters
            ]
        ).float()

    auxiliary_flat = packed(auxiliary)
    actor_flat = packed(actor)
    critic_flat = packed(critic)
    task_flat = actor_flat + critic_flat

    def cosine(other):
        return (auxiliary_flat * other).sum() / (
            auxiliary_flat.norm() * other.norm()
        ).clamp_min(1e-20)

    return {
        "delivered_norm": auxiliary_flat.norm(),
        "actor_cosine": cosine(actor_flat),
        "critic_cosine": cosine(critic_flat),
        "task_cosine": cosine(task_flat),
    }


@torch.no_grad()
def capture_tape_gradient_groups(
    trunk_blocks,
    predictor_parameters,
    *,
    trunk_active,
    trunk_max_norm,
    predictor_max_norm,
):
    """Globally clip the trunk once, while recording block-local raw/delivered norms."""
    trunk_blocks = [list(block) for block in trunk_blocks]
    if not trunk_blocks or any(not block for block in trunk_blocks):
        raise ValueError("every logical trunk block must contain parameters")
    trunk_parameters = [
        parameter for block in trunk_blocks for parameter in block
    ]

    def block_norms():
        norms = []
        for block in trunk_blocks:
            squared_norms = [
                parameter.grad.detach().float().square().sum()
                for parameter in block
                if parameter.grad is not None
            ]
            norms.append(
                torch.stack(squared_norms).sum().sqrt()
                if squared_norms
                else trunk_parameters[0].new_zeros(())
            )
        return torch.stack(norms)

    if trunk_active:
        block_raw_norms = block_norms()
        trunk_norm = clip_grad_norm_async_fail_loud_(
            trunk_parameters,
            trunk_max_norm,
        )
        block_delivered_norms = block_norms()
        trunk_gradients = capture_gradients(trunk_parameters)
    else:
        trunk_norm = predictor_parameters[0].new_zeros(())
        block_raw_norms = predictor_parameters[0].new_zeros(len(trunk_blocks))
        block_delivered_norms = predictor_parameters[0].new_zeros(
            len(trunk_blocks)
        )
        trunk_gradients = {}
    predictor_norm = clip_grad_norm_async_fail_loud_(
        predictor_parameters,
        predictor_max_norm,
    )
    predictor_gradients = capture_gradients(predictor_parameters)
    return (
        trunk_norm,
        predictor_norm,
        block_raw_norms,
        block_delivered_norms,
        trunk_gradients,
        predictor_gradients,
    )


@torch.no_grad()
def merge_gradient_groups(parameters, *gradient_groups, validate_finite=True):
    """Sum independently clipped gradient groups for one shared optimizer step.

    Missing entries remain missing (not zero gradients), preserving Adam's exact lazy
    state/step behavior. Every provided tensor is checked on-device before installation.
    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError("at least one optimizer parameter is required")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("optimizer parameters must be unique")
    parameter_ids = {id(parameter) for parameter in parameters}
    merged = {}
    for group in gradient_groups:
        for parameter, gradient in group.items():
            if id(parameter) not in parameter_ids:
                raise ValueError("gradient group contains a foreign parameter")
            if gradient.shape != parameter.shape or gradient.device != parameter.device:
                raise ValueError("gradient must match its parameter shape and device")
            if validate_finite:
                torch._assert_async(
                    torch.isfinite(gradient).all(),
                    "A clipped gradient group is non-finite; refusing the shared Adam step",
                )
            if parameter in merged:
                merged[parameter].add_(gradient)
            else:
                merged[parameter] = gradient.detach().clone()
    for parameter in parameters:
        parameter.grad = merged.get(parameter)
    return merged


@torch.no_grad()
def apply_union_optimizer_step(
    parameters,
    optimizer,
    *,
    actor_gradients,
    critic_gradients,
    auxiliary_gradients,
    validate_finite=True,
):
    """Install three clipped groups and advance their coherent shared Adam moments."""
    parameters = list(parameters)
    merged = merge_gradient_groups(
        parameters,
        actor_gradients,
        critic_gradients,
        auxiliary_gradients,
        validate_finite=validate_finite,
    )
    if not merged:
        optimizer.zero_grad(set_to_none=True)
        return merged
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return merged


@torch.no_grad()
def apply_private_predictor_step(parameters, optimizer, gradients):
    """Advance the predictor's private Adam from its independently clipped gradient."""
    parameters = list(parameters)
    merged = merge_gradient_groups(parameters, gradients, validate_finite=True)
    if not merged:
        optimizer.zero_grad(set_to_none=True)
        return
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def shape_advantage(gae, sigma, u, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform. Works on a full
    batch or a single minibatch (sigma/u must be sliced to match gae)."""
    if args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_std":
        # Per-state-normalized, bounded, magnitude-preserving near 0 (THE FIX).
        return torch.tanh(gae / (args.tanh_kappa * sigma))
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "cdf_probit":
        centered = (2.0 * u - 1.0)
        c = args.cdf_probit_clamp
        return ((2.0 ** 0.5) * torch.erfinv(centered.clamp(-c, c).cpu())).to(device)
    elif args.adv_transform == "clip_z":
        # Winsorized z-score: ONE standardize + hard tail clip. tanh's hard cousin;
        # preserves bulk magnitude exactly (linear in [-c,c]) and is ~self-normalizing
        # (clipping a unit-var signal barely changes its std), so norm_adv ~ no-op.
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return gz.clamp(-args.clip_z_c, args.clip_z_c)
    elif args.adv_transform == "rankgauss":
        # Rank-Gaussian: the principled single op. Replaces each advantage by the
        # Gaussian quantile of its empirical rank -> self-normalizing (bounded,
        # zero-mean, ~unit-scale, fully outlier-immune; no kappa, no double z-score).
        # This is cdf_probit with the EMPIRICAL batch rank instead of the critic's
        # per-state CDF (distribution-free; the distribution was shown incidental).
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)  # 0..n-1
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        return ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
    elif args.adv_transform == "rankgauss_signed":
        # Sign-anchored rank-Gaussian: rank positives and negatives SEPARATELY so the
        # advantage's zero-crossing is preserved exactly. Plain rankgauss anchors the
        # sign flip at the batch MEDIAN, not 0, so on skewed GAE it can assign the
        # wrong sign to near-median samples — and PPO needs the sign right. Each sign
        # group is mapped to its half of the Gaussian by its within-group rank.
        c = args.cdf_probit_clamp
        out = torch.zeros_like(gae)
        for side in (gae > 0, gae < 0):
            if side.any():
                g = gae[side]
                r = g.argsort().argsort().to(torch.float32)
                half = (r + 0.5) / float(g.numel())                  # (0,1) in group
                uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)  # correct side
                ctr = (2.0 * uq - 1.0).clamp(-c, c)
                out[side] = ((2.0 ** 0.5) * torch.erfinv(ctr.cpu())).to(device)
        return out
    elif args.adv_transform == "rankgauss_temp":
        # Rank-Gaussian + tanh temperature: compress the Gaussian quantiles' extremes
        # harder (rank_tanh_kappa < inf). Probes the "more robust still" axis the kappa
        # sweep motivated (tanh_gae kappa=1 > kappa=2). Smaller kappa => harder.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        z = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
        return torch.tanh(z / args.rank_tanh_kappa)
    elif args.adv_transform == "rankgauss_signmag":
        # Sign-correct WITHOUT count distortion: take plain rankgauss's GLOBAL-rank
        # magnitude, then force the sign to match the raw advantage. Fixes the flaw in
        # rankgauss_signed (per-group half-Gaussian over-amplifies the minority sign by
        # COUNT); here magnitude still reflects global rank extremity and only the ~9%
        # near-zero "flips" get re-signed. Nonlinear (not a shift) => survives norm_adv.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c).cpu())).to(device).abs()
        return torch.sign(gae) * mag
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope == "batch"), \
        "norm_adv_scope=batch requires adv_transform_scope=batch"
    assert args.tpo_coef > 0.0, "TPO-MD is the entire policy update; tpo_coef must be > 0"
    assert args.tpo_k >= 2, "TPO needs at least two candidates per group"
    assert args.tpo_eps > 0.0, "tpo_eps must be positive"
    assert args.tpo_eta_base > 0.0, "tpo_eta_base must be positive"
    assert args.tpo_kl_breaker > 0.0, "tpo_kl_breaker must be positive"
    assert args.hidden >= 1, "hidden must be positive"
    assert args.k_blocks >= 1, "k_blocks must be positive"
    assert args.n_experts >= 1, "n_experts must be positive"
    assert args.num_envs >= 1 and args.num_steps >= 1
    assert args.num_minibatches >= 1
    assert args.batch_size % args.num_minibatches == 0
    assert args.tape_period == SPECTRAL_PERIOD, "v27 uses the fixed N=16 spectrum"
    assert 0.0 <= args.tape_rho < 1.0, "tape_rho must be in [0, 1)"
    assert 0.0 <= args.tape_lambda <= 1.0, "tape_lambda must be in [0, 1]"
    assert args.tape_slot_embed_dim >= 1
    assert args.tape_predictor_hidden >= 1
    assert args.tape_target_batch_size >= 1, "tape_target_batch_size must be positive"
    assert args.tape_coef >= 0.0
    assert args.tape_scale_floor_ratio > 0.0
    assert args.tape_trunk_grad_clip >= 0.0
    assert args.tape_predictor_grad_clip >= 0.0
    assert args.tape_diagnostic_rows >= 1
    assert args.tail_risk_window >= 1
    assert len(args.tail_risk_thresholds) >= 1
    assert args.separate_grad_clip, "TAPE union requires separate gradient groups"
    # Probe rewards are RAW physics rewards; the critic must live in the same units.
    assert not args.normalize_reward, "TPO probe scores require raw rewards"
    assert not args.clip_reward, "TPO probe scores require unclipped rewards"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Ampere+ TF32 tensor-core matmuls are substantially faster. ``high`` retains
    # full-size float32 outputs/accumulators; only last-bit eager-v13 numerics may differ.
    torch.set_float32_matmul_precision("high")

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                args.normalize_reward,
                args.clip_reward,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # TPO-MD probe machinery (always on: TPO-MD IS the policy update; every
    # (env, step) cell is probed — no state-frac mask).
    # Cache once: the raw physics env (walk the .env chain to the unwrapped
    # MujocoEnv) and the NormalizeObservation wrapper reference per env.
    probe_base_envs = [e.unwrapped for e in envs.envs]
    probe_obs_wrappers = [find_wrapper(e, gym.wrappers.NormalizeObservation) for e in envs.envs]
    assert all(w is not None for w in probe_obs_wrappers), "NormalizeObservation wrapper not found"
    probe_action_low = envs.single_action_space.low
    probe_action_high = envs.single_action_space.high
    # Persistent probe RNG stream: saved CPU+CUDA states restored inside
    # torch.random.fork_rng at every sampling site, so candidate sampling
    # never advances the MAIN RNG stream (the PPO trajectory of a tpo run
    # matches an unprobed run exactly).
    probe_cpu_rng_state = None
    probe_cuda_rng_state = None
    # Independent persistent stream for the one-sample frozen-policy TAPE tail.
    # fork_rng restores the main generator, and these states never alias probe state.
    tape_aux_cpu_rng_state = None
    tape_aux_cuda_rng_state = None
    td_rms_ema = None  # EMA (decay 0.99) of the one-step TD-residual RMS

    agent = Agent(envs, args).to(device)
    assert agent.tape_num_layers == args.k_blocks + 2
    assert len(agent.tape_layer_names) == agent.tape_num_layers
    # Task Adam owns the exact TPO model. Only the auxiliary trunk gradient joins its
    # moments; predictor parameters retain a private optimizer as in v13.
    task_params = agent.task_parameters()
    task_optimizer = optim.Adam(task_params, lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    tape_params = agent.tape_parameters()
    tape_trunk_blocks = agent.tape_trunk_blocks()
    tape_trunk_params = agent.tape_trunk_parameters()
    tape_predictor_params = agent.tape_predictor_parameters()
    if args.tape:
        assert len(tape_trunk_blocks) == agent.tape_num_layers
        assert len(agent.tape_predictor.predictors) == agent.tape_num_layers
        assert all(
            predictor.period == args.tape_period
            and predictor.num_slots == agent.tape_num_slots
            and predictor.feature_dim == agent.tape_feature_dim
            for predictor in agent.tape_predictor.predictors
        )
    assert {id(parameter) for parameter in tape_params} == {
        id(parameter)
        for parameter in (tape_trunk_params + tape_predictor_params)
    }
    predictor_optimizer = (
        optim.Adam(
            tape_predictor_params,
            lr=args.learning_rate,
            eps=1e-5,
        )
        if args.tape
        else None
    )

    def policy_rollout_fn(obs_):
        return policy_model_forward(agent, obs_)

    def policy_update_fn(obs_):
        return policy_model_forward(agent, obs_)

    def transition_value_fn(obs_):
        return value_forward(agent, obs_)

    def probe_value_fn(obs_):
        return value_forward(agent, obs_)

    def target_actor_feat_fn(obs_):
        return target_actor_feat_forward(agent, obs_)

    def tape_update_fn(
        actor_feature_,
        action_,
        target_,
        mask_,
        latent_rms_,
        h1_scale_,
        mode_scale_,
    ):
        return tape_auxiliary_update_forward(
            agent,
            actor_feature_,
            action_,
            target_,
            mask_,
            latent_rms_,
            h1_scale_,
            mode_scale_,
            scale_floor_ratio=args.tape_scale_floor_ratio,
            period=args.tape_period,
        )

    def tape_target_scale_fn(target_, mask_, latent_rms_):
        return spectral_target_scales(
            target_,
            mask_,
            latent_rms=latent_rms_,
            scale_floor_ratio=args.tape_scale_floor_ratio,
            period=args.tape_period,
        )

    def target_tape_fn(actor_feature_, action_):
        return target_tape_predictor_forward(agent, actor_feature_, action_)

    def project_value_targets_fn(targets_):
        return hl_support.project(targets_)

    if args.compile:
        policy_rollout_fn = torch.compile(
            policy_rollout_fn, mode=args.compile_mode, dynamic=False
        )
        policy_update_fn = torch.compile(
            policy_update_fn, mode=args.compile_mode, dynamic=False
        )
        transition_value_fn = torch.compile(
            transition_value_fn, mode=args.compile_mode, dynamic=False
        )
        probe_value_fn = torch.compile(
            probe_value_fn, mode=args.compile_mode, dynamic=False
        )
        target_actor_feat_fn = torch.compile(
            target_actor_feat_fn, mode=args.compile_mode, dynamic=False
        )
        tape_update_fn = torch.compile(
            tape_update_fn,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        tape_target_scale_fn = torch.compile(
            tape_target_scale_fn,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        target_tape_fn = torch.compile(
            target_tape_fn, mode=args.compile_mode, dynamic=False
        )
        project_value_targets_fn = torch.compile(
            project_value_targets_fn,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        print(f"compiled static agent and target paths ({args.compile_mode})")

    # Soft max-entropy temperature (gaussian only). log_alpha is learned so the
    # policy-only future-entropy bonus AND the actor entropy bonus self-tune to
    # hold the SQUASHED entropy at target_entropy via SAC's temperature dual.
    auto_alpha = args.auto_entropy and args.actor_dist == "gaussian"
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    support_min, support_max = value_support_bounds(args)
    hl_support = HLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )
    support = hl_support.support                       # (num_bins,) bin centers
    bin_width = hl_support.bin_width
    scalar_support = symexp(support) if args.value_symlog else support

    def value_logits_to_scalar(logits):
        return hl_support.to_expected_scalar(logits)

    scalar_bin_width = (
        (scalar_support[1:] - scalar_support[:-1]).abs().min()
        if args.value_symlog
        else bin_width
    )

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.ones((args.num_steps, args.num_envs)).to(device)
    next_transition_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_transition_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    # Preallocated probe storage: CPU numpy for the physics outputs, GPU
    # tensors for candidate z's/logprobs (written once per step, no per-env syncs).
    tpo_next_obs_np = np.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + envs.single_observation_space.shape, dtype=np.float32
    )
    tpo_rewards_np = np.zeros((args.num_steps, args.num_envs, args.tpo_k), dtype=np.float32)
    tpo_terms_np = np.zeros((args.num_steps, args.num_envs, args.tpo_k), dtype=np.float32)
    tpo_zs = torch.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + envs.single_action_space.shape
    ).to(device)
    tpo_logprobs = torch.zeros((args.num_steps, args.num_envs, args.tpo_k)).to(device)

    global_step = 0
    start_time = time.time()
    episode_return_window = deque(maxlen=args.tail_risk_window)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            task_optimizer.param_groups[0]["lr"] = lrnow
            if predictor_optimizer is not None:
                predictor_optimizer.param_groups[0]["lr"] = lrnow
        probe_seconds = 0.0

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                rollout_outputs = policy_rollout_fn(next_obs)
                (
                    action,
                    z,
                    logprob,
                    ent,
                    value_logits,
                    roll_actor_feat,
                ), roll_dist, roll_to_action, roll_log_det_fn = action_value_from_policy_outputs(
                    agent,
                    rollout_outputs,
                )
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits)
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            # --- TPO probe: K candidates per env, one raw physics step each (every state) ---
            probe_start = time.time()
            with torch.no_grad():
                # Candidate sampling rides the PERSISTENT probe RNG stream inside
                # fork_rng: restore probe state, sample, save state back. The main
                # stream is untouched.
                with torch.random.fork_rng(devices=[device]):
                    if probe_cpu_rng_state is None:
                        torch.manual_seed(args.seed + 1_000_003)
                    else:
                        torch.set_rng_state(probe_cpu_rng_state)
                        torch.cuda.set_rng_state(probe_cuda_rng_state, device)
                    cand_z = roll_dist.sample(torch.Size([args.tpo_k]))   # (K, N, A)
                    probe_cpu_rng_state = torch.get_rng_state()
                    probe_cuda_rng_state = torch.cuda.get_rng_state(device)
                cand_z = cand_z.permute(1, 0, 2).contiguous()             # (N, K, A)
                if args.actor_dist == "beta":
                    cand_z = cand_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                cand_z[:, 0] = z                                          # executed action = candidate 0
                cz = cand_z.transpose(0, 1)                               # (K, N, A)
                cand_logprob = (roll_dist.log_prob(cz) - roll_log_det_fn(cz)).sum(-1).transpose(0, 1)
                tpo_zs[step] = cand_z
                tpo_logprobs[step] = cand_logprob                         # (N, K)
                # One transfer for the whole candidate block (no per-env GPU syncs).
                cand_actions_np = roll_to_action(cand_z).cpu().numpy()
            cand_actions_np = np.clip(cand_actions_np, probe_action_low, probe_action_high)
            for env_i in range(args.num_envs):
                base_env = probe_base_envs[env_i]
                obs_rms = probe_obs_wrappers[env_i].obs_rms
                saved_qpos = base_env.data.qpos.copy()
                saved_qvel = base_env.data.qvel.copy()
                saved_warm = base_env.data.qacc_warmstart.copy()
                saved_time = base_env.data.time
                for cand_i in range(args.tpo_k):
                    # Direct-assign restore (NO mj_forward, NEVER MujocoEnv.set_state):
                    # mj_step recomputes forward dynamics itself; restoring
                    # qacc_warmstart keeps the solver warmstart bit-identical so the
                    # REAL env.step below matches an unprobed run exactly.
                    base_env.data.qpos[:] = saved_qpos
                    base_env.data.qvel[:] = saved_qvel
                    base_env.data.qacc_warmstart[:] = saved_warm
                    base_env.data.time = saved_time
                    probe_obs, probe_rew, probe_term, _, _ = base_env.step(cand_actions_np[env_i, cand_i])
                    # FROZEN wrapper stats (stepping the raw env never updates
                    # obs_rms): float64 math, cast float32, then clip [-10, 10].
                    norm_obs = ((probe_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)).astype(np.float32)
                    tpo_next_obs_np[step, env_i, cand_i] = np.clip(norm_obs, -10.0, 10.0)
                    tpo_rewards_np[step, env_i, cand_i] = probe_rew       # RAW reward (base is raw-return)
                    tpo_terms_np[step, env_i, cand_i] = float(probe_term)
                base_env.data.qpos[:] = saved_qpos
                base_env.data.qvel[:] = saved_qvel
                base_env.data.qacc_warmstart[:] = saved_warm
                base_env.data.time = saved_time
            probe_seconds += time.time() - probe_start

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                transition_next_obs = np.array(next_obs_np, copy=True)
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0
            else:
                transition_next_obs = next_obs_np
            transition_next_obs_t = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                next_transition_logits = transition_value_fn(transition_next_obs_t)
                next_transition_values[step] = value_logits_to_scalar(next_transition_logits)
            next_transition_obses[step] = transition_next_obs_t
            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(terminations, device=device, dtype=torch.float32)
            transition_boundaries[step] = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        episode_return_window.append(
                            float(np.asarray(info["episode"]["r"]).reshape(()))
                        )

        # Freeze the complete auxiliary teacher immediately after collection, before
        # constructing any rollout label and before any optimizer can mutate the live net.
        if args.tape:
            tape_snapshot_lag_before_capture = agent.target_snapshot_lag()
            drift_observations = obs.reshape(
                (-1,) + envs.single_observation_space.shape
            )[: min(256, args.batch_size)]
            (
                tape_encoder_drift_before_capture,
                tape_encoder_relative_drift_before_capture,
            ) = target_encoder_functional_drift(agent, drift_observations)
            agent.snapshot_tape_target()
            tape_snapshot_lag_at_capture = agent.target_snapshot_lag()

        with torch.no_grad():
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample a' ~ π(·|s_{t+1}) for each transition bootstrap entropy.
                # Use transition_next_obs, not rollout next_obs, so time-limit
                # truncations pair V(final_obs) with H(final_obs) rather than
                # accidentally reading entropy from the reset observation.
                _, _, next_transition_logprob, _, _ = agent.get_action_and_value(
                    next_transition_obses.reshape((-1,) + envs.single_observation_space.shape)
                )
                next_transition_logprob = next_transition_logprob.reshape(args.num_steps, args.num_envs)
                alpha_r = log_alpha.exp().detach()
                next_value_bonus = alpha_r * (-next_transition_logprob)
            else:
                next_value_bonus = None
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). Unbiased (state-dependent baseline); its large
            # magnitude is harmless — rankgauss at the optim step rank-normalizes it while
            # PRESERVING the entropy-induced reordering (the actual exploration signal).
            if auto_alpha:
                policy_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                    lambda_nonterminal = 1.0 - transition_boundaries[t]
                    delta = (
                        rewards[t]
                        + args.gamma
                        * (next_transition_values[t] + next_value_bonus[t])
                        * bootstrap_nonterminal
                        - values[t]
                    )
                    policy_adv[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                    )
            else:
                policy_adv = advantages
            # Critic target: Dreamer4-style scalar-return HL-Gauss. GAE computes
            # the scalar λ-return; the value encoder projects that scalar target
            # into a Gaussian-smoothed categorical distribution over fixed bins.
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            projected_targets = project_value_targets_fn(returns)
            # This full CUDA-resident table is indexed for all ten PPO epochs. A
            # compiled graph owns a reusable output buffer, so retain an independent
            # clone before any later compiled forward can replay its storage.
            target_probs = retain_graph_output(
                projected_targets,
                compiled=args.compile,
            )
            # Per-state return std sigma(s_t) in raw return units, matching the
            # GAE scale consumed by tanh_std.
            sigma = (value_probs * (scalar_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * scalar_bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            returns_coord = symlog(returns) if args.value_symlog else returns
            cdf_frac = ((returns_coord.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean()  # calib probe (uniform≈0.1)

            # --- TPO-MD target construction (frozen pre-update critic; q fixed across epochs) ---
            # Running one-step TD-residual RMS over EXECUTED transitions -> GLOBAL score sigma.
            td_resid = (
                rewards
                + args.gamma * next_transition_values * (1.0 - transition_terminations) * transition_valids
                - values
            )
            td_rms = td_resid.pow(2).mean().sqrt().item()
            td_rms_ema = td_rms if td_rms_ema is None else 0.99 * td_rms_ema + 0.01 * td_rms
            tpo_sigma_global = max(args.tpo_sigma_scale_coef * td_rms_ema, 1e-6)

            b_tpo_zs = tpo_zs.reshape((-1, args.tpo_k) + envs.single_action_space.shape)
            obs_dim = int(np.array(envs.single_observation_space.shape).prod())
            flat_probe_obs = torch.as_tensor(
                tpo_next_obs_np.reshape(-1, obs_dim), device=device
            )
            # Four static 65,536-row critic forwards at the defaults. Clone each graph
            # output immediately: later replays reuse its otherwise-ephemeral storage.
            probe_value_chunks = []
            for chunk in flat_probe_obs.split(65536):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                chunk_logits = probe_value_fn(chunk)
                chunk_logits = retain_graph_output(
                    chunk_logits,
                    compiled=args.compile,
                )
                probe_value_chunks.append(value_logits_to_scalar(chunk_logits))
            v_next = torch.cat(probe_value_chunks).reshape(
                args.batch_size, args.tpo_k
            )
            r_probe = torch.as_tensor(tpo_rewards_np.reshape(-1, args.tpo_k), device=device)
            term_probe = torch.as_tensor(tpo_terms_np.reshape(-1, args.tpo_k), device=device)
            # Oracle score: raw probe reward + bootstrapped frozen value.
            scores = r_probe + args.gamma * (1.0 - term_probe) * v_next      # (B, K)
            # Center per group, scale by the ONE GLOBAL sigma: cross-state advantage
            # MAGNITUDE survives (per-group z-scoring would erase it); no floor gating —
            # weak groups just contribute u ~= 0 naturally.
            u_scores = (
                (scores - scores.mean(dim=-1, keepdim=True)) / tpo_sigma_global
            ).clamp(-5.0, 5.0)
            group_std = scores.std(dim=-1, unbiased=False)                   # (B,) diagnostics only
            anchor_logp = F.log_softmax(tpo_logprobs.reshape(-1, args.tpo_k), dim=-1)

            def tpo_mean_kl(eta):
                # batch-mean KL(p_old || q(eta)); monotone DECREASING in eta
                # (eta -> inf => q -> p_old => KL -> 0).
                return tpo_reverse_kl(anchor_logp, u_scores, eta).item()

            # kl_base = the NATURAL (uncapped) step the SNR signal produces at the
            # fixed base temperature. In SNR units this is large under real signal
            # and -> 0 when candidates are within the critic's noise floor. Used by
            # the dynamic-cap path; also logged for diagnostics.
            tpo_kl_base = tpo_mean_kl(args.tpo_eta_base)
            tpo_cap_engaged = 0.0
            if u_scores.abs().max().item() < 1e-8:
                # Degenerate scores: target collapses to the anchor regardless of eta.
                # In dyn-trust eta_base is the natural choice (q ~= p_old anyway); in
                # v1 mode the original code returned 1.0 here.
                tpo_eta_solved = args.tpo_eta_base if args.tpo_dyn_trust else 1.0
            elif args.tpo_dyn_trust:
                # One-sided KL cap on the fixed base temperature. KL(eta) is monotone
                # DECREASING in eta, so we only ever RAISE eta above eta_base to pull
                # an over-large natural step DOWN to the cap; we never lower it. Thus
                # eta_solved >= eta_base ALWAYS, the step is bounded above by eps_cap,
                # and is free to shrink to ~0 when kl_base falls below the cap. No
                # lower floor — intentional (this is the late-training fix).
                if tpo_kl_base <= args.tpo_eps:
                    tpo_eta_solved = args.tpo_eta_base       # weak signal: natural step already within cap
                else:
                    tpo_cap_engaged = 1.0                    # strong signal: cap binds
                    log_lo, log_hi = float(np.log(args.tpo_eta_base)), float(np.log(1e4))
                    if tpo_mean_kl(float(np.exp(log_hi))) > args.tpo_eps:
                        tpo_eta_solved = float(np.exp(log_hi))  # even max temperature can't reach cap -> clamp
                    else:
                        # KL(eta_base) > eps and KL(1e4) <= eps -> root bracketed.
                        for _ in range(40):
                            log_mid = 0.5 * (log_lo + log_hi)
                            if tpo_mean_kl(float(np.exp(log_mid))) > args.tpo_eps:
                                log_lo = log_mid             # KL too big -> need larger eta
                            else:
                                log_hi = log_mid
                        tpo_eta_solved = float(np.exp(0.5 * (log_lo + log_hi)))
            elif args.tpo_adaptive_eta:
                # MPO-style dual: bisect log-eta so mean KL(p_old||q) = tpo_eps.
                log_lo, log_hi = float(np.log(1e-2)), float(np.log(1e4))
                if tpo_mean_kl(float(np.exp(log_lo))) < args.tpo_eps:
                    tpo_eta_solved = float(np.exp(log_lo))   # weak scores: even max-strength < eps
                elif tpo_mean_kl(float(np.exp(log_hi))) > args.tpo_eps:
                    tpo_eta_solved = float(np.exp(log_hi))   # huge scores: clamp at max temperature
                else:
                    for _ in range(40):
                        log_mid = 0.5 * (log_lo + log_hi)
                        if tpo_mean_kl(float(np.exp(log_mid))) > args.tpo_eps:
                            log_lo = log_mid                 # KL too big -> need larger eta
                        else:
                            log_hi = log_mid
                    tpo_eta_solved = float(np.exp(0.5 * (log_lo + log_hi)))
            else:
                tpo_eta_solved = args.tpo_eta
            b_tpo_q = tpo_restricted_target(anchor_logp, u_scores, tpo_eta_solved).detach()
            tpo_kl_achieved = tpo_mean_kl(tpo_eta_solved)
            log_q = b_tpo_q.clamp_min(1e-12).log()
            tpo_group_kl = (b_tpo_q * (log_q - anchor_logp)).sum(-1).mean().item()
            tpo_q_entropy = (-(b_tpo_q * log_q).sum(-1)).mean().item()
            tpo_score_std_mean = group_std.mean().item()
            tpo_score_std_p90 = group_std.quantile(0.9).item()
            if args.tape:
                flat_observations = obs.reshape(
                    (-1,) + envs.single_observation_space.shape
                )
                flat_next_observations = next_transition_obses.reshape(
                    (-1,) + envs.single_observation_space.shape
                )
                current_feature_chunks = []
                next_feature_chunks = []
                for current_chunk, next_chunk in zip(
                    flat_observations.split(args.tape_target_batch_size),
                    flat_next_observations.split(args.tape_target_batch_size),
                    strict=True,
                ):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    current_feature_chunks.append(
                        retain_graph_output(
                            target_actor_feat_fn(current_chunk), compiled=args.compile
                        )
                    )
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    next_feature_chunks.append(
                        retain_graph_output(
                            target_actor_feat_fn(next_chunk), compiled=args.compile
                        )
                    )
                frozen_actor_feats = torch.cat(current_feature_chunks).reshape(
                    args.num_steps,
                    args.num_envs,
                    agent.tape_num_layers,
                    agent.tape_feature_dim,
                )
                frozen_next_actor_feats = torch.cat(next_feature_chunks).reshape(
                    args.num_steps,
                    args.num_envs,
                    agent.tape_num_layers,
                    agent.tape_feature_dim,
                )
                del current_feature_chunks, next_feature_chunks
                valid_feature_rows = transition_valids.bool().unsqueeze(-1).unsqueeze(-1)
                tape_innovations = torch.where(
                    valid_feature_rows,
                    frozen_next_actor_feats - frozen_actor_feats,
                    torch.zeros_like(frozen_actor_feats),
                ).detach()

                # Truncations and the artificial rollout edge get exactly one target-
                # policy action from a persistent auxiliary RNG stream. Interior TD(lambda)
                # reads the frozen predictor table at the stored behavior (s_t, a_t).
                tape_next_state_rows = next_state_bootstrap_rows(
                    transition_terminations,
                    transition_boundaries,
                    transition_valids,
                )
                (
                    sampled_next_actions,
                    tape_aux_cpu_rng_state,
                    tape_aux_cuda_rng_state,
                ) = build_sparse_frozen_next_action_table(
                    agent,
                    frozen_next_actor_feats,
                    tape_next_state_rows,
                    seed=args.seed + 2_000_003,
                    cpu_rng_state=tape_aux_cpu_rng_state,
                    device_rng_state=tape_aux_cuda_rng_state,
                )
                current_prediction_chunks = []
                for current_feature_chunk, action_chunk in zip(
                    frozen_actor_feats.reshape(
                        -1, agent.tape_num_layers, agent.tape_feature_dim
                    ).split(args.tape_target_batch_size),
                    actions.reshape(
                        -1, agent.tape_action_dim
                    ).split(args.tape_target_batch_size),
                    strict=True,
                ):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    current_prediction_chunks.append(
                        retain_graph_output(
                            target_tape_fn(current_feature_chunk, action_chunk),
                            compiled=args.compile,
                        )
                    )
                target_current_predictions = torch.cat(
                    current_prediction_chunks
                ).reshape(
                    args.num_steps,
                    args.num_envs,
                    agent.tape_num_layers,
                    agent.tape_num_slots,
                    agent.tape_feature_dim,
                )
                del current_prediction_chunks
                target_edge_predictions = build_sparse_frozen_edge_predictions(
                    agent,
                    frozen_next_actor_feats,
                    sampled_next_actions,
                    tape_next_state_rows,
                    batch_size=args.tape_target_batch_size,
                )
                tape_sampled_action_rows = tape_next_state_rows
                tape_targets = build_spectral_td_lambda_targets(
                    tape_innovations,
                    target_current_predictions,
                    target_edge_predictions,
                    transition_terminations,
                    transition_boundaries,
                    transition_valids,
                    args.tape_lambda,
                    args.tape_rho,
                    args.tape_period,
                )
                latent_rms = frozen_actor_feats.detach().square().mean(dim=(0, 1, 3)).sqrt()
                flat_tape_targets = tape_targets.values.reshape(
                    args.batch_size,
                    agent.tape_num_layers,
                    agent.tape_num_slots,
                    agent.tape_feature_dim,
                )
                flat_tape_mask = tape_targets.masks.reshape(
                    args.batch_size,
                    agent.tape_num_layers,
                    agent.tape_num_slots,
                )
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                target_scale_outputs = tape_target_scale_fn(
                    flat_tape_targets, flat_tape_mask, latent_rms
                )
                tape_h1_scale = retain_graph_output(
                    target_scale_outputs[0], compiled=args.compile
                )
                tape_mode_scale = retain_graph_output(
                    target_scale_outputs[1], compiled=args.compile
                )
                del target_scale_outputs

                # One deterministic, bounded M=64 bootstrapped diagnostic per rollout.
                # It reuses the frozen pre-update predictor table and never allocates
                # a full return table or consumes policy/training RNG.
                tape_diagnostic_row_indices = deterministic_diagnostic_rows(
                    args.batch_size,
                    min(args.tape_diagnostic_rows, args.minibatch_size),
                    iteration,
                    device,
                )
                _, frozen_prediction_modes = split_spectral_slots(
                    target_current_predictions, args.tape_period
                )
                (
                    tape_m64_bootstrap_modes,
                    tape_m64_bootstrap_row_mask,
                    tape_m64_realized_steps,
                ) = build_m_step_bootstrapped_spectral_returns(
                    tape_innovations,
                    frozen_prediction_modes,
                    transition_boundaries,
                    transition_valids,
                    tape_diagnostic_row_indices,
                    max_steps=SPECTRAL_DIAGNOSTIC_STEPS,
                    rho=args.tape_rho,
                    period=args.tape_period,
                )
                flat_target_predictions = target_current_predictions.reshape(
                    args.batch_size,
                    agent.tape_num_layers,
                    agent.tape_num_slots,
                    agent.tape_feature_dim,
                )
                diagnostic_prediction = flat_target_predictions.index_select(
                    0, tape_diagnostic_row_indices
                )
                _, diagnostic_prediction_modes = split_spectral_slots(
                    diagnostic_prediction, args.tape_period
                )
                tape_m64_bootstrap_statistics = complex_mode_statistics(
                    diagnostic_prediction_modes,
                    tape_m64_bootstrap_modes,
                    tape_m64_bootstrap_row_mask,
                    args.tape_period,
                )
                tape_m64_bootstrap_coverage = (
                    tape_m64_bootstrap_row_mask.float().mean()
                )
                tape_m64_realized_steps_mean = tape_m64_realized_steps.float().mean()

                diagnostic_target = flat_tape_targets.index_select(
                    0, tape_diagnostic_row_indices
                )
                target_h1, target_modes = split_spectral_slots(
                    diagnostic_target, args.tape_period
                )
                prediction_h1, prediction_modes = split_spectral_slots(
                    diagnostic_prediction, args.tape_period
                )
                target_row_mask = transition_valids.bool().reshape(-1).index_select(
                    0, tape_diagnostic_row_indices
                )
                tape_h1_statistics = layer_vector_statistics(
                    prediction_h1, target_h1, target_row_mask
                )
                tape_mode_statistics = complex_mode_statistics(
                    prediction_modes, target_modes, target_row_mask, args.tape_period
                )
                tape_latent_scale, tape_latent_participation_rank = (
                    latent_scale_and_participation_rank(frozen_actor_feats)
                )
                tape_layer_latent_statistics = [
                    latent_scale_and_participation_rank(
                        frozen_actor_feats[..., layer_index, :]
                    )
                    for layer_index in range(agent.tape_num_layers)
                ]
                tape_layer_latent_scales = torch.stack(
                    [statistics[0] for statistics in tape_layer_latent_statistics]
                )
                tape_layer_participation_ranks = torch.stack(
                    [statistics[1] for statistics in tape_layer_latent_statistics]
                )
                tape_actor_head_norm = actor_head_weight_norm(agent)
                del (
                    target_current_predictions,
                    target_edge_predictions,
                    sampled_next_actions,
                    frozen_next_actor_feats,
                    flat_target_predictions,
                    diagnostic_prediction,
                    diagnostic_prediction_modes,
                    diagnostic_target,
                    prediction_h1,
                    prediction_modes,
                    target_h1,
                    target_modes,
                    frozen_prediction_modes,
                    frozen_actor_feats,
                    tape_innovations,
                    tape_m64_bootstrap_modes,
                    tape_diagnostic_row_indices,
                    tape_m64_bootstrap_row_mask,
                    tape_m64_realized_steps,
                )

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.num_bins)
        b_u = u.reshape(-1)
        if args.tape:
            b_tape_targets = flat_tape_targets
            b_tape_mask = flat_tape_mask
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean()

        b_inds = np.arange(args.batch_size)
        # v13 converted every float32 minibatch mean to Python before NumPy's
        # float64 average. Accumulate the same sequence asynchronously on CUDA.
        clipfrac_sum = torch.zeros((), dtype=torch.float64, device=device)
        clipfrac_count = 0
        epochs_completed = 0
        actor_epochs_completed = 0
        actor_active = True  # flipped off by the tpo_kl_breaker; critic runs all epochs regardless
        max_minibatch_approx_kl = torch.zeros((), device=device)
        max_epoch_approx_kl = torch.zeros((), device=device)
        if args.tape:
            tape_h1_losses = torch.zeros(agent.tape_num_layers, device=device)
            tape_mode_losses = torch.zeros(
                agent.tape_num_layers,
                SPECTRAL_NUM_FREQUENCIES,
                device=device,
            )
            tape_block_raw_norms = torch.zeros(
                agent.tape_num_layers, device=device
            )
            tape_block_delivered_norms = torch.zeros(
                agent.tape_num_layers, device=device
            )
            tape_grad_telemetry = {
                name: torch.zeros((), device=device)
                for name in (
                    "delivered_norm",
                    "actor_cosine",
                    "critic_cosine",
                    "task_cosine",
                )
            }
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # Keep NumPy's exact seeded permutation, paying one compact transfer per
            # epoch instead of implicit host-index conversion for every buffer read.
            epoch_inds = torch.as_tensor(b_inds, device=device)
            epoch_kl_sum = torch.zeros((), dtype=torch.float64, device=device)
            epoch_kl_count = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = epoch_inds[start:end]

                # Same trunk forward as the base; the candidate logprobs ride the
                # SAME dist (no second trunk pass, consumes no RNG).
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                update_outputs = policy_update_fn(b_obs[mb_inds])
                (
                    _,
                    _,
                    newlogprob,
                    entropy,
                    value_logits,
                    new_cand_logprobs,
                    mb_actor_feat,
                ), _, _, _ = action_value_from_policy_outputs(
                    agent,
                    update_outputs,
                    b_latent_zs[mb_inds],
                    b_tpo_zs[mb_inds],
                )

                with torch.no_grad():
                    # TELEMETRY ONLY: ratio / KL / clipfrac (and pg_loss below) never
                    # reach a backward — the actor update is the TPO CE alone.
                    logratio = newlogprob.detach() - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac_sum.add_(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    )
                    clipfrac_count += 1
                    epoch_kl_sum.add_(approx_kl)
                    epoch_kl_count += 1
                    update_scalar_max_(max_minibatch_approx_kl, approx_kl)

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # PMPO-style asymmetric pos/neg weighting: scale positive-advantage
                # samples by 2*alpha and negatives by 2*(1-alpha) (alpha=0.5 => identity).
                # alpha>0.5 emphasizes reinforcing good actions over suppressing bad ones.
                # Split on the SHAPED advantage's sign (pre-norm = the true advantage sign).
                if args.pos_neg_alpha != 0.5:
                    mb_advantages = mb_advantages * torch.where(
                        mb_raw_adv >= 0,
                        2.0 * args.pos_neg_alpha,
                        2.0 * (1.0 - args.pos_neg_alpha),
                    )

                # TELEMETRY-ONLY clipped surrogate (kept for cross-run comparability;
                # ratio is already detached, so no PG gradient can exist anywhere).
                with torch.no_grad():
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # TPO CE on the K-restricted softmax over ALL states (every state is
                # probed). Targets q are frozen (solved once post-rollout, detached).
                mb_logp_new = F.log_softmax(new_cand_logprobs, dim=-1)
                tpo_ce = (-(b_tpo_q[mb_inds] * mb_logp_new).sum(-1)).mean()
                # PURE mirror descent: the CE is the entire actor objective.
                actor_loss = args.tpo_coef * tpo_ce

                # HL-Gauss value loss: cross-entropy to the fixed scalar-return
                # projection target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

                if args.tape:
                    # Once the breaker fires, TAPE becomes a predictor-only
                    # objective. Detaching here prevents construction of an auxiliary
                    # backward path through the trunk while predictor learning continues.
                    tape_source = tape_source_feature(
                        mb_actor_feat,
                        trunk_active=actor_active,
                    )
                    (
                        tape_loss,
                        tape_h1_losses,
                        tape_mode_losses,
                    ) = tape_update_fn(
                        tape_source,
                        b_actions[mb_inds],
                        b_tape_targets[mb_inds],
                        b_tape_mask[mb_inds],
                        latent_rms,
                        tape_h1_scale,
                        tape_mode_scale,
                    )

                entropy_loss = entropy.mean()

                if auto_alpha:
                    # SAC's temperature dual (sac_continuous_action.py), on the
                    # SQUASHED log-prob: alpha_loss = (-α·(logπ + target_entropy)).mean().
                    # With target_entropy=-|A|, drives E[logπ_squashed] -> |A|,
                    # equivalently E[-logπ_squashed] -> -|A|.
                    # The SAME α weights the explicit CURRENT-step actor entropy bonus below
                    # (the soft return's current-state entropy is action-independent => zero
                    # in the PG term, so the bonus supplies the actual entropy gradient).
                    ent_coef_eff = log_alpha.exp().detach()
                    alpha_loss = (-log_alpha.exp() * (newlogprob.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    ent_coef_eff = args.ent_coef

                # Actor, critic, and auxiliary trunk are independently clipped; their
                # shared-trunk contributions meet only in task Adam. The predictor keeps
                # private moments and never enters the task optimizer.
                agent.zero_grad(set_to_none=True)
                (args.vf_coef * v_loss).backward(
                    retain_graph=actor_active
                )
                critic_gn = clip_grad_norm_async_fail_loud_(
                    critic_params, args.critic_grad_clip
                )
                value_grads = capture_gradients(critic_params)

                tape_trunk_grads = {}
                tape_predictor_grads = {}
                if args.tape:
                    agent.zero_grad(set_to_none=True)
                    (args.tape_coef * tape_loss).backward(
                        retain_graph=actor_active
                    )
                    (
                        tape_trunk_gn,
                        tape_predictor_gn,
                        tape_block_raw_norms,
                        tape_block_delivered_norms,
                        tape_trunk_grads,
                        tape_predictor_grads,
                    ) = capture_tape_gradient_groups(
                        tape_trunk_blocks,
                        tape_predictor_params,
                        trunk_active=actor_active,
                        trunk_max_norm=args.tape_trunk_grad_clip,
                        predictor_max_norm=args.tape_predictor_grad_clip,
                    )
                agent.zero_grad(set_to_none=True)
                actor_grads = {}
                if actor_active:
                    # Pure TPO CE remains the entire actor backward.
                    (actor_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = clip_grad_norm_async_fail_loud_(
                        actor_params, args.actor_grad_clip
                    )
                    actor_grads = capture_gradients(actor_params)
                else:
                    actor_gn = torch.zeros((), device=device)

                if args.tape and (
                    epoch + 1 == args.update_epochs
                    and start + args.minibatch_size >= args.batch_size
                ):
                    tape_grad_telemetry = tape_gradient_telemetry(
                        tape_trunk_params,
                        tape_trunk_grads,
                        actor_grads,
                        value_grads,
                    )

                apply_union_optimizer_step(
                    task_params,
                    task_optimizer,
                    actor_gradients=actor_grads,
                    critic_gradients=value_grads,
                    auxiliary_gradients=tape_trunk_grads,
                    # Each nonempty group was already checked once by its global clip.
                    validate_finite=False,
                )
                if args.tape:
                    apply_private_predictor_step(
                        tape_predictor_params,
                        predictor_optimizer,
                        tape_predictor_grads,
                    )

            epochs_completed = epoch + 1
            if actor_active:
                actor_epochs_completed = epoch + 1
                # Circuit breaker (NOT an epoch break): past 3x the per-update KL
                # budget the actor stops, but the critic keeps training all epochs.
                # One epoch-level control-flow synchronization replaces one sync per
                # minibatch while retaining v13's float64 mean and breaker boundary.
                epoch_mean_kl = epoch_kl_sum / epoch_kl_count
                update_scalar_max_(max_epoch_approx_kl, epoch_mean_kl)
                if epoch_mean_kl.item() > args.tpo_kl_breaker:
                    actor_active = False
            # target_kl (default None here) would also stop the critic; kept only as
            # an explicit opt-in escape hatch.
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        var_y = b_returns.var(correction=0)
        explained_var = torch.where(
            var_y == 0,
            torch.full_like(var_y, float("nan")),
            1.0 - (b_returns - b_values).var(correction=0) / var_y,
        )
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean()
        device_telemetry = {
            "losses/value_loss": v_loss,
            "losses/policy_loss": pg_loss,
            "losses/entropy": entropy_loss,
            "losses/old_approx_kl": old_approx_kl,
            "losses/approx_kl": approx_kl,
            "losses/max_minibatch_approx_kl": max_minibatch_approx_kl,
            "losses/max_epoch_approx_kl": max_epoch_approx_kl,
            "losses/clipfrac": clipfrac_sum / clipfrac_count,
            "losses/explained_variance": explained_var,
            "losses/actor_grad_norm": actor_gn,
            "losses/critic_grad_norm": critic_gn,
            "losses/tpo_ce": tpo_ce,
            "debug/returns_mean": b_returns.mean(),
            "debug/returns_std": b_returns.std(),
            "debug/returns_absmax": b_returns.abs().max(),
            "debug/target_edge_mass": edge_mass,
            "debug/distpg_corr_with_gae": adv_corr,
            "debug/distpg_sign_agree": adv_sign_agree,
            "debug/u_edge_frac": u_edge_frac,
            "debug/sigma_mean": b_sigma.mean(),
        }
        if auto_alpha:
            device_telemetry.update(
                {
                    "losses/alpha": log_alpha.exp(),
                    "debug/squashed_entropy": (-logprobs).mean(),
                    "debug/soft_bootstrap_bonus": next_value_bonus.mean(),
                    "debug/soft_adv_std_ratio": policy_adv.std()
                    / (advantages.std() + 1e-8),
                }
            )
        if args.tape:
            device_telemetry.update(
                {
                    "losses/tape_prediction": tape_loss,
                    "losses/tape_h1": tape_h1_losses.mean(),
                    "losses/tape_spectral": tape_mode_losses.mean(),
                    "losses/tape_grad_norm": tape_trunk_gn,
                    "losses/tape_trunk_grad_norm": tape_trunk_gn,
                    "losses/tape_predictor_grad_norm": tape_predictor_gn,
                    "tape/delivered_trunk_norm": tape_grad_telemetry["delivered_norm"],
                    "tape/actor_cosine": tape_grad_telemetry["actor_cosine"],
                    "tape/critic_cosine": tape_grad_telemetry["critic_cosine"],
                    "tape/task_cosine": tape_grad_telemetry["task_cosine"],
                    "tape/h1_scale": tape_h1_scale.mean(),
                    "tape/mode_scale": tape_mode_scale.mean(),
                    "tape/target_snapshot_lag_at_capture": tape_snapshot_lag_at_capture,
                    "tape/target_snapshot_lag_before_capture": tape_snapshot_lag_before_capture,
                    "tape/encoder_drift_before_capture": tape_encoder_drift_before_capture,
                    "tape/encoder_relative_drift_before_capture": tape_encoder_relative_drift_before_capture,
                    "tape/sampled_next_action_fraction": tape_sampled_action_rows.float().mean(),
                    "tape/m64_bootstrap_coverage": tape_m64_bootstrap_coverage,
                    "tape/m64_realized_steps": tape_m64_realized_steps_mean,
                    "tape/lambda": torch.as_tensor(args.tape_lambda, device=device),
                    "tape/rho": torch.as_tensor(args.tape_rho, device=device),
                    "tape/latent_scale": tape_latent_scale,
                    "tape/latent_participation_rank": tape_latent_participation_rank,
                    "tape/actor_head_weight_norm": tape_actor_head_norm,
                }
            )
            for layer_index, layer_name in enumerate(agent.tape_layer_names):
                device_telemetry.update(
                    {
                        f"tape_layers/{layer_name}/h1_loss": tape_h1_losses[layer_index],
                        f"tape_layers/{layer_name}/spectral_loss": tape_mode_losses[layer_index].mean(),
                        f"tape_layers/{layer_name}/h1_nmse": tape_h1_statistics["nmse"][layer_index],
                        f"tape_layers/{layer_name}/h1_cosine": tape_h1_statistics["cosine"][layer_index],
                        f"tape_layers/{layer_name}/h1_target_rms": tape_h1_statistics["target_rms"][layer_index],
                        f"tape_layers/{layer_name}/h1_prediction_rms": tape_h1_statistics["prediction_rms"][layer_index],
                        f"tape_layers/{layer_name}/h1_scale": tape_h1_scale[layer_index],
                        f"tape_layers/{layer_name}/mode_target_rms": tape_mode_statistics["target_rms"][layer_index].mean(),
                        f"tape_layers/{layer_name}/mode_prediction_rms": tape_mode_statistics["prediction_rms"][layer_index].mean(),
                        f"tape_layers/{layer_name}/mode_nmse": tape_mode_statistics["nmse"][layer_index].mean(),
                        f"tape_layers/{layer_name}/mode_cosine": tape_mode_statistics["cosine"][layer_index].mean(),
                        f"tape_layers/{layer_name}/mode_prediction_quadrature_rms": tape_mode_statistics["prediction_quadrature_rms"][layer_index].mean(),
                        f"tape_layers/{layer_name}/mode_target_quadrature_rms": tape_mode_statistics["target_quadrature_rms"][layer_index].mean(),
                        f"tape_layers/{layer_name}/m64_bootstrap_nmse": tape_m64_bootstrap_statistics["nmse"][layer_index].mean(),
                        f"tape_layers/{layer_name}/m64_bootstrap_cosine": tape_m64_bootstrap_statistics["cosine"][layer_index].mean(),
                        f"tape_layers/{layer_name}/latent_scale": tape_layer_latent_scales[layer_index],
                        f"tape_layers/{layer_name}/latent_participation_rank": tape_layer_participation_ranks[layer_index],
                        f"tape_blocks/{layer_name}/raw_aux_grad_norm": tape_block_raw_norms[layer_index],
                        f"tape_blocks/{layer_name}/delivered_aux_grad_norm": tape_block_delivered_norms[layer_index],
                    }
                )
                for frequency_index in range(SPECTRAL_NUM_FREQUENCIES):
                    prefix = f"tape_spectrum/{layer_name}/k{frequency_index}"
                    device_telemetry.update(
                        {
                            f"{prefix}_loss": tape_mode_losses[layer_index, frequency_index],
                            f"{prefix}_scale": tape_mode_scale[layer_index, frequency_index],
                            f"{prefix}_target_rms": tape_mode_statistics["target_rms"][layer_index, frequency_index],
                            f"{prefix}_prediction_rms": tape_mode_statistics["prediction_rms"][layer_index, frequency_index],
                            f"{prefix}_nmse": tape_mode_statistics["nmse"][layer_index, frequency_index],
                            f"{prefix}_cosine": tape_mode_statistics["cosine"][layer_index, frequency_index],
                            f"{prefix}_prediction_quadrature_rms": tape_mode_statistics["prediction_quadrature_rms"][layer_index, frequency_index],
                            f"{prefix}_target_quadrature_rms": tape_mode_statistics["target_quadrature_rms"][layer_index, frequency_index],
                            f"{prefix}_m64_bootstrap_nmse": tape_m64_bootstrap_statistics["nmse"][layer_index, frequency_index],
                            f"{prefix}_m64_bootstrap_cosine": tape_m64_bootstrap_statistics["cosine"][layer_index, frequency_index],
                            f"{prefix}_m64_bootstrap_target_rms": tape_m64_bootstrap_statistics["target_rms"][layer_index, frequency_index],
                            f"{prefix}_m64_bootstrap_prediction_rms": tape_m64_bootstrap_statistics["prediction_rms"][layer_index, frequency_index],
                        }
                    )
        host_telemetry = synchronize_scalar_telemetry(device_telemetry)
        writer.add_scalar(
            "charts/learning_rate", task_optimizer.param_groups[0]["lr"], global_step
        )
        for tag, value in host_telemetry.items():
            writer.add_scalar(tag, value, global_step)
        if auto_alpha:
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
        for tag, value in {
            "debug/epochs_completed": epochs_completed,
            "debug/actor_epochs_completed": actor_epochs_completed,
            "debug/tpo_eta_solved": tpo_eta_solved,
            "debug/tpo_kl_achieved": tpo_kl_achieved,
            "debug/tpo_kl_base": tpo_kl_base,
            "debug/tpo_cap_engaged": tpo_cap_engaged,
            "debug/tpo_group_kl": tpo_group_kl,
            "debug/tpo_score_std_mean": tpo_score_std_mean,
            "debug/tpo_score_std_p90": tpo_score_std_p90,
            "debug/tpo_sigma_global": tpo_sigma_global,
            "debug/tpo_q_entropy": tpo_q_entropy,
        }.items():
            writer.add_scalar(tag, value, global_step)
        if args.tape:
            writer.add_scalar(
                "tape_union/predictor_only_delivery",
                float(not actor_active),
                global_step,
            )
        tail_risk = summarize_episode_tail_risk(
            episode_return_window, args.tail_risk_thresholds
        )
        for name, value in tail_risk.items():
            writer.add_scalar(f"tail_risk/{name}", value, global_step)
        writer.add_scalar("charts/probe_sps_overhead", probe_seconds, global_step)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)
        del target_probs, b_target_probs
        if args.tape:
            # These names alias the same large rollout-resident target/mask storage.
            # Drop every reference before the next rollout constructs its replacement.
            del (
                tape_targets,
                flat_tape_targets,
                flat_tape_mask,
                b_tape_targets,
                b_tape_mask,
            )

    envs.close()
    writer.close()

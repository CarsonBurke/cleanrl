# SUCCESSOR-INNOVATION TD v10: action-conditioned predictive coding for PPO.
# =====================================================================================
# Let eta_t = hbar(s_{t+1}) - hbar(s_t) be an EMA-encoder innovation and
# Psi_g(s,a) its normalized discounted successor. For each gamma g:
#   y_g = (1-g) sum_{j=0}^{n-1} g^j eta_{t+j} + g^n sg[Psibar_g(s_{t+n},a_{t+n})].
# Prefixes (4,8,16,64) match gammas (.6,.9,.97,.99). True terminals keep their final
# innovation and stop; time-limit and rollout tails bootstrap from their actual final
# state under a frozen behavior-policy action. Detached RMS only preconditions losses,
# so changing scales never mix coordinate systems inside a TD target. Direct one-step,
# reward, frozen-policy, and frozen-critic anchors retain control semantics; iteration-
# scale EMA targets and block-local conflict projection keep the auxiliary subordinate.
# Atomic finite checks isolate private Adam and make every rejected batch task-only.
# =====================================================================================
import copy
import os
import random
import time
from dataclasses import dataclass
from math import isfinite, log
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

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def value_support_bounds(args):
    """Return critic support endpoints in the coordinate system used by bins."""
    return args.v_min, args.v_max


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


@dataclass(frozen=True)
class SuccessorTargets:
    """Fixed, stop-gradient rollout labels for the successor auxiliary."""

    values: torch.Tensor
    masks: torch.Tensor
    bootstrap_masks: torch.Tensor
    observed_mass: torch.Tensor


def next_state_bootstrap_rows(
    transition_terminations: torch.Tensor,
    transition_boundaries: torch.Tensor,
    transition_valids: torch.Tensor,
) -> torch.Tensor:
    """Rows needing a newly sampled next-state action: truncations and rollout edge."""
    if not (
        transition_terminations.shape
        == transition_boundaries.shape
        == transition_valids.shape
    ):
        raise ValueError("transition masks must have matching shapes")
    if transition_terminations.ndim != 2 or transition_terminations.shape[0] == 0:
        raise ValueError("transition masks must have non-empty [time, env] shape")
    terminations = transition_terminations.bool()
    boundaries = transition_boundaries.bool()
    valids = transition_valids.bool()
    rows = valids & boundaries & ~terminations
    rows[-1] |= valids[-1] & ~boundaries[-1]
    return rows


def masked_feature_rms(features: torch.Tensor, mask: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Per-feature RMS over valid rows; used only as a detached change of units."""
    if features.shape[:-1] != mask.shape:
        raise ValueError(f"feature/mask shape mismatch: {features.shape} vs {mask.shape}")
    weights = mask.to(dtype=features.dtype).unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    return (
        (features.detach().square() * weights)
        .sum(dim=tuple(range(features.ndim - 1)))
        .div(denom)
        .sqrt()
        .clamp_min(eps)
    )


def build_successor_targets(
    innovations: torch.Tensor,
    tail_predictions: torch.Tensor,
    next_state_tail_predictions: torch.Tensor,
    transition_terminations: torch.Tensor,
    transition_boundaries: torch.Tensor,
    transition_valids: torch.Tensor,
    gammas: Sequence[float],
    direct_horizons: Sequence[int],
) -> SuccessorTargets:
    """Build normalized successor targets with an observed prefix and one TD tail.

    A true terminal transition's observed final-state innovation is retained and then
    stops.  A truncation instead bootstraps at its final observation, and a source that
    reaches the artificial rollout edge bootstraps at the rollout's final next state.
    Both use a separately sampled frozen-policy action.  Thus every valid source has a
    complete target without ever crossing into an autoreset observation.
    """
    if innovations.ndim != 3:
        raise ValueError("innovations must have shape [time, env, feature]")
    if tail_predictions.ndim != 4:
        raise ValueError("tail_predictions must have shape [time, env, band, feature]")
    time_dim, num_envs, feature_dim = innovations.shape
    num_bands = len(gammas)
    if len(direct_horizons) != num_bands:
        raise ValueError("gammas and direct_horizons must have the same length")
    if tail_predictions.shape != (time_dim, num_envs, num_bands, feature_dim):
        raise ValueError("tail_predictions shape does not match innovations/bands")
    if next_state_tail_predictions.shape != tail_predictions.shape:
        raise ValueError("next_state_tail_predictions shape does not match tail_predictions")
    if transition_terminations.shape != (time_dim, num_envs):
        raise ValueError("transition_terminations has the wrong shape")
    if transition_boundaries.shape != (time_dim, num_envs):
        raise ValueError("transition_boundaries has the wrong shape")
    if transition_valids.shape != (time_dim, num_envs):
        raise ValueError("transition_valids has the wrong shape")
    if any(not 0.0 < gamma < 1.0 for gamma in gammas):
        raise ValueError("successor gammas must lie strictly between zero and one")
    if any(horizon < 1 for horizon in direct_horizons):
        raise ValueError("direct horizons must be positive")

    # Enforce semi-gradient TD even when a caller forgets a no_grad context.
    innovations = innovations.detach()
    tail_predictions = tail_predictions.detach()
    next_state_tail_predictions = next_state_tail_predictions.detach()
    terminations = transition_terminations.detach().bool()
    boundaries = transition_boundaries.detach().bool()
    valids = transition_valids.detach().bool()
    values = innovations.new_zeros((time_dim, num_envs, num_bands, feature_dim))
    bootstrap_masks = torch.zeros((time_dim, num_envs, num_bands), dtype=torch.bool, device=innovations.device)
    observed_mass = innovations.new_zeros((time_dim, num_envs, num_bands))

    for band, (gamma, horizon) in enumerate(zip(gammas, direct_horizons)):
        alive = torch.ones((time_dim, num_envs), dtype=torch.bool, device=innovations.device)
        discount = 1.0
        for offset in range(min(horizon, time_dim)):
            valid_len = time_dim - offset
            source_alive = alive[:valid_len]
            step_ok = source_alive & valids[offset:]
            weight = (1.0 - gamma) * discount
            values[:valid_len, :, band] += weight * innovations[offset:] * step_ok.unsqueeze(-1)
            observed_mass[:valid_len, :, band] += weight * step_ok

            # If a later final observation is unavailable, fall back before that
            # transition to the EMA tail at the known (state, action).  The invalid
            # transition itself remains masked as a source below.
            pre_step_fallback = source_alive & ~valids[offset:] & valids[:valid_len]
            values[:valid_len, :, band] += (
                discount
                * tail_predictions[offset:, :, band]
                * pre_step_fallback.unsqueeze(-1)
            )
            bootstrap_masks[:valid_len, :, band] |= pre_step_fallback

            next_discount = discount * gamma
            boundary_here = step_ok & boundaries[offset:]
            truncation_here = boundary_here & ~terminations[offset:]

            # Exactly one source reaches the artificial end of the rollout at each
            # offset. It has a real next observation but no stored next action.
            rollout_edge = torch.zeros_like(step_ok)
            rollout_edge[-1] = step_ok[-1] & ~boundaries[-1]
            next_state_bootstrap = truncation_here | rollout_edge
            values[:valid_len, :, band] += (
                next_discount
                * next_state_tail_predictions[offset:, :, band]
                * next_state_bootstrap.unsqueeze(-1)
            )
            bootstrap_masks[:valid_len, :, band] |= next_state_bootstrap

            # A complete n-step prefix inside the rollout uses the actually sampled
            # behavior action a_{t+n}; edge rows were handled just above.
            if offset + 1 == horizon and valid_len > 1:
                interior_bootstrap = step_ok[:-1] & ~boundaries[offset:-1]
                values[: valid_len - 1, :, band] += (
                    next_discount
                    * tail_predictions[offset + 1 :, :, band]
                    * interior_bootstrap.unsqueeze(-1)
                )
                bootstrap_masks[: valid_len - 1, :, band] |= interior_bootstrap

            next_alive = torch.zeros_like(alive)
            if offset + 1 < horizon:
                next_alive[:valid_len] = step_ok & ~boundaries[offset:]
            alive = next_alive
            discount = next_discount

    # Every band has a directly observed first innovation under the same mask.
    masks = valids.unsqueeze(-1).expand(time_dim, num_envs, num_bands).clone()
    return SuccessorTargets(values.detach(), masks, bootstrap_masks, observed_mass)


def masked_scaled_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_scale: float = 1e-3,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Band/feature-balanced Huber loss; RMS only preconditions the residual."""
    if prediction.shape != target.shape or prediction.shape[:-1] != mask.shape:
        raise ValueError("prediction, target, and mask shapes are inconsistent")
    if target.ndim < 3:
        raise ValueError("prediction must have leading rows, band, and feature dimensions")
    weights = mask.to(prediction.dtype).unsqueeze(-1)
    # One scale per (successor band, latent feature). This retains raw latent units
    # in the Bellman equation while preventing a high-variance feature from owning it.
    row_reduce_dims = tuple(range(target.ndim - 2))
    if scale is None:
        denom = weights.sum(dim=row_reduce_dims).clamp_min(1.0)
        scale = (
            (target.detach().square() * weights).sum(dim=row_reduce_dims) / denom
        ).sqrt().clamp_min(min_scale)
    else:
        if scale.shape != target.shape[-2:]:
            raise ValueError("provided scale must have [band, feature] shape")
        scale = scale.detach().to(device=target.device, dtype=target.dtype).clamp_min(min_scale)
    scale_view = scale.view(*([1] * (target.ndim - 2)), *scale.shape)
    normalized_error = F.smooth_l1_loss(
        prediction / scale_view,
        target / scale_view,
        reduction="none",
    ).mean(-1)
    loss = (normalized_error * mask.to(normalized_error.dtype)).sum() / mask.sum().clamp_min(1)
    return loss, scale.detach()


def store_critic_targets(
    probabilities: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep fixed critic labels resident; fp16 halves storage, CE still promotes to fp32."""
    if probabilities.shape[:-1] != mask.shape:
        raise ValueError("critic target probabilities and mask have inconsistent shapes")
    if not probabilities.is_floating_point():
        raise ValueError("critic target probabilities must be floating point")
    return (
        probabilities.detach().to(device=device, dtype=torch.float16),
        mask.detach().to(device=device, dtype=torch.bool),
    )


@torch.no_grad()
def capture_target_latent_tables(
    target_current_feature_fn,
    target_next_feature_fn,
    current_observations: torch.Tensor,
    next_observations: torch.Tensor,
    mark_compile_step,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize target features that must outlive later compiled graph replays.

    ``torch.compile(mode="reduce-overhead")`` may return views of reusable CUDA-graph
    output buffers. Distinct Python wrappers do not guarantee distinct storage because
    compile caches and CUDA-graph trees can share a pool. Both rollout-sized tables are
    retained while subsequent compiled functions run, so each output must become owned
    immediately, before any later graph invocation can overwrite its backing buffer.
    """
    mark_compile_step()
    current_latents = target_current_feature_fn(current_observations).clone()
    mark_compile_step()
    next_latents = target_next_feature_fn(next_observations).clone()
    if current_latents.ndim != 2 or next_latents.ndim != 2:
        raise ValueError("target feature functions must return rank-two latent tables")
    if current_latents.shape != next_latents.shape:
        raise ValueError("current and next target latent tables must have matching shapes")
    return current_latents, next_latents


def auxiliary_gradients_are_adam_safe(
    gradients: Sequence[Optional[torch.Tensor]],
    upstream_valid: bool | torch.Tensor = True,
) -> torch.Tensor:
    """Return one device-side validity bit for an atomic auxiliary transaction.

    Adam squares every gradient.  Elementwise finiteness is therefore insufficient:
    a finite FP32 value such as ``1e30`` would immediately overflow ``exp_avg_sq``.
    The accumulated squared norm is checked as well so many individually safe tensors
    cannot overflow the reduction.  Callers must treat a false result as a veto of the
    *whole* auxiliary transaction, never as permission to sanitize selected entries.
    """
    reference = next((gradient for gradient in gradients if gradient is not None), None)
    device = torch.device("cpu") if reference is None else reference.device
    valid = torch.as_tensor(upstream_valid, device=device)
    if valid.numel() != 1:
        raise ValueError("upstream validity must be scalar")
    if valid.dtype == torch.bool:
        valid = valid.reshape(())
    else:
        valid = torch.isfinite(valid).all() & valid.bool().all()
    present = [gradient.detach().float().reshape(-1) for gradient in gradients if gradient is not None]
    if not present:
        return valid
    # One packing op and one reduction avoid hundreds of tiny CUDA reductions over
    # ThinkTrunk parameter tensors.  A finite accumulated square proves both element
    # finiteness and Adam second-moment safety (NaN/Inf inputs cannot yield a finite sum).
    flat = torch.cat(present)
    return valid & torch.isfinite(flat.square().sum())


def finite_diagnostic(value: torch.Tensor) -> torch.Tensor:
    """Return a detached finite scalar while validity is logged separately."""
    if value.numel() != 1:
        raise ValueError("diagnostic value must be scalar")
    value = value.detach().float().reshape(())
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value))


def clip_auxiliary_gradients_fail_closed(
    gradients: Sequence[Optional[torch.Tensor]],
    max_norm: float,
    transaction_valid: bool | torch.Tensor = True,
) -> tuple[list[Optional[torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Clip a private-model gradient vector without ever evaluating NaN times zero."""
    if not isfinite(max_norm) or max_norm < 0.0:
        raise ValueError("auxiliary gradient cap must be finite and non-negative")
    reference = next((gradient for gradient in gradients if gradient is not None), None)
    device = torch.device("cpu") if reference is None else reference.device
    valid = auxiliary_gradients_are_adam_safe(gradients, transaction_valid)
    safe_gradients = [
        None
        if gradient is None
        else torch.where(valid, gradient.detach(), torch.zeros_like(gradient))
        for gradient in gradients
    ]
    square = torch.zeros((), device=device, dtype=torch.float32)
    for gradient in safe_gradients:
        if gradient is not None:
            square = square + gradient.float().square().sum()
    valid = valid & torch.isfinite(square)
    safe_square = torch.where(valid, square, torch.zeros_like(square))
    raw_norm = safe_square.sqrt()
    scale = torch.minimum(
        torch.ones((), device=device),
        torch.as_tensor(max_norm, device=device) / (raw_norm + 1e-6),
    )
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    clipped = [
        None if gradient is None else gradient * scale.to(gradient.dtype)
        for gradient in safe_gradients
    ]
    return clipped, raw_norm.detach(), valid


@torch.no_grad()
def _repair_optimizer_moments(
    optimizer: optim.Optimizer,
    parameters: Sequence[nn.Parameter],
) -> torch.Tensor:
    """Repair invalid private Adam moments and report whether repair was needed."""
    if not parameters:
        raise ValueError("at least one private parameter is required")
    reference = parameters[0]
    moments_valid = reference.new_ones((), dtype=torch.bool)
    for parameter in parameters:
        for name, value in optimizer.state.get(parameter, {}).items():
            # The scalar clock is not a proposal moment and must remain monotonic.
            if name == "step" or not torch.is_tensor(value):
                continue
            if not (torch.is_floating_point(value) or torch.is_complex(value)):
                continue
            finite = torch.isfinite(value)
            value_valid = finite.all()
            if value_valid.device != moments_valid.device:
                value_valid = value_valid.to(moments_valid.device)
            moments_valid = moments_valid & value_valid
            value.copy_(torch.where(finite, value, torch.zeros_like(value)))
    return moments_valid


@torch.no_grad()
def apply_private_auxiliary_optimizer_transaction(
    optimizer: optim.Optimizer,
    parameters: Sequence[nn.Parameter],
    gradients: Sequence[Optional[torch.Tensor]],
    transaction_valid: bool | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Advance private Adam atomically while containing every invalid auxiliary.

    Invalid transactions are represented to Adam as an all-zero gradient, allowing its
    clock and finite momentum decay to advance consistently.  The resulting parameter
    proposal is then rolled back exactly.  Invalid pre-existing or newly produced moments
    are repaired to zero, so a rejected batch cannot poison a later valid batch.
    """
    parameters = list(parameters)
    if not parameters or len(parameters) != len(gradients):
        raise ValueError("private parameters and gradients must be non-empty and align")
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        if (
            parameter.shape != gradient.shape
            or parameter.device != gradient.device
            or parameter.dtype != gradient.dtype
        ):
            raise ValueError("private gradients must match their parameters")

    reference = parameters[0]
    valid = auxiliary_gradients_are_adam_safe(gradients, transaction_valid)
    valid = valid & _repair_optimizer_moments(optimizer, parameters)
    parameter_before = [parameter.detach().clone() for parameter in parameters]
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, gradients):
        candidate = torch.zeros_like(parameter) if gradient is None else gradient.detach()
        parameter.grad = torch.where(valid, candidate, torch.zeros_like(candidate))
    optimizer.step()

    proposal_square = reference.new_zeros((), dtype=torch.float32)
    proposal_valid = reference.new_ones((), dtype=torch.bool)
    for parameter, before in zip(parameters, parameter_before):
        update = parameter.detach().float() - before.float()
        proposal_square = proposal_square + update.square().sum()
        proposal_valid = (
            proposal_valid
            & torch.isfinite(parameter).all()
            & torch.isfinite(update).all()
            & torch.isfinite(update.square()).all()
            & torch.isfinite(proposal_square)
        )
    valid = valid & proposal_valid & _repair_optimizer_moments(optimizer, parameters)
    for parameter, before in zip(parameters, parameter_before):
        parameter.copy_(torch.where(valid, parameter, before))
    optimizer.zero_grad(set_to_none=True)

    zero = reference.new_zeros(())
    return {
        "numeric_valid": valid.to(reference.dtype),
        "step_norm": torch.where(valid, proposal_square.sqrt(), zero).detach(),
    }


def install_gradient_transaction(
    parameters: Sequence[nn.Parameter],
    task_gradient_maps: Sequence[dict[nn.Parameter, torch.Tensor]],
    auxiliary_gradients: dict[nn.Parameter, torch.Tensor],
) -> None:
    """Install task plus already-vetted auxiliary gradients without touching task data.

    In particular, no ``nan_to_num`` is applied to PPO gradients: a task failure remains
    a loud task failure.  A globally vetoed auxiliary arrives as exact zero tensors, so
    adding it is bitwise identical to the corresponding task-only gradient.
    """
    for parameter in parameters:
        combined = None
        for gradient_map in (*task_gradient_maps, auxiliary_gradients):
            gradient = gradient_map.get(parameter)
            if gradient is None:
                continue
            if (
                gradient.shape != parameter.shape
                or gradient.device != parameter.device
                or gradient.dtype != parameter.dtype
            ):
                raise ValueError("installed gradients must match their parameters")
            combined = gradient.detach().clone() if combined is None else combined + gradient
        parameter.grad = combined


def project_and_cap_auxiliary_gradients(
    auxiliary: Sequence[Optional[torch.Tensor]],
    task: Sequence[Optional[torch.Tensor]],
    absolute_cap: float,
    task_ratio_cap: float,
    groups: Optional[Sequence[int]] = None,
    safety_tasks: Optional[Sequence[Sequence[Optional[torch.Tensor]]]] = None,
    transaction_valid: bool | torch.Tensor = True,
) -> tuple[list[Optional[torch.Tensor]], dict[str, torch.Tensor]]:
    """Block-local PCGrad, per-objective vetoes, and local/global trust budgets."""
    if len(auxiliary) != len(task):
        raise ValueError("auxiliary and task gradient lists must align")
    if groups is None:
        groups = [0] * len(auxiliary)
    if len(groups) != len(auxiliary):
        raise ValueError("gradient groups must align with the gradient lists")
    if safety_tasks is None:
        safety_tasks = ()
    if any(len(constraint) != len(auxiliary) for constraint in safety_tasks):
        raise ValueError("safety-task gradient lists must align")
    if (
        not isfinite(absolute_cap)
        or not isfinite(task_ratio_cap)
        or absolute_cap < 0.0
        or task_ratio_cap < 0.0
    ):
        raise ValueError("gradient caps must be finite and non-negative")
    device = next(
        (g.device for gradients in (auxiliary, task) for g in gradients if g is not None),
        torch.device("cpu"),
    )
    zero = torch.zeros((), device=device, dtype=torch.float32)
    # Training supplies one atomic auxiliary validity bit, while PPO task/safety
    # directions have already crossed ``clip_grad_norm_(error_if_nonfinite=True)``.
    # Standalone callers may omit it, in which case validate the auxiliary once here.
    standalone_validation = isinstance(transaction_valid, bool) and transaction_valid
    if standalone_validation:
        aux_valid = auxiliary_gradients_are_adam_safe(auxiliary).to(device)
        task_valid = auxiliary_gradients_are_adam_safe(task).to(device)
        safety_valid = torch.ones((), device=device, dtype=torch.bool)
        for constraint in safety_tasks:
            safety_valid = safety_valid & auxiliary_gradients_are_adam_safe(
                constraint
            ).to(device)
    else:
        aux_valid = torch.as_tensor(transaction_valid, device=device)
        if aux_valid.numel() != 1:
            raise ValueError("transaction validity must be scalar")
        aux_valid = torch.isfinite(aux_valid).all() & aux_valid.bool().all()
        task_valid = torch.ones((), device=device, dtype=torch.bool)
        safety_valid = torch.ones((), device=device, dtype=torch.bool)
    numeric_valid = aux_valid & task_valid & safety_valid

    aux_sq = zero.clone()
    task_sq = zero.clone()
    dot = zero.clone()
    conflict_flags = []
    veto_flags = []
    group_cosines = []
    group_records = []
    # Preserve module-level task directions rather than allowing a large positive
    # block to hide an anti-task update in another block. Flattening once per module
    # avoids hundreds of tiny CUDA reduction kernels over individual parameters.
    ordered_groups = list(dict.fromkeys(groups))
    all_active_indices = [
        index for index, gradient in enumerate(auxiliary) if gradient is not None
    ]
    all_active_sizes = [auxiliary[index].numel() for index in all_active_indices]
    if all_active_indices:
        raw_flat_auxiliary = torch.cat(
            [
                auxiliary[index].detach().float().reshape(-1)
                for index in all_active_indices
            ]
        )
        safe_flat_auxiliary = torch.where(
            numeric_valid,
            raw_flat_auxiliary,
            torch.zeros_like(raw_flat_auxiliary),
        )
        safe_auxiliary = dict(
            zip(all_active_indices, safe_flat_auxiliary.split(all_active_sizes))
        )
    else:
        safe_auxiliary = {}
    for group in ordered_groups:
        indices = [index for index, candidate in enumerate(groups) if candidate == group]
        active_indices = [index for index in indices if auxiliary[index] is not None]
        full_task_parts = [task[index].detach().float().reshape(-1) for index in indices if task[index] is not None]
        full_task = torch.cat(full_task_parts) if full_task_parts else torch.zeros(0, device=device)
        full_group_task_sq = full_task.square().sum()
        task_sq = task_sq + full_group_task_sq

        if active_indices:
            aux_parts = [safe_auxiliary[index] for index in active_indices]
            sizes = [part.numel() for part in aux_parts]
            flat_aux = torch.cat(aux_parts)
            flat_task = torch.cat(
                [
                    torch.zeros_like(auxiliary[index]).float().reshape(-1)
                    if task[index] is None
                    else task[index].detach().float().reshape(-1)
                    for index in active_indices
                ]
            )
        else:
            sizes = []
            flat_aux = torch.zeros(0, device=device)
            flat_task = torch.zeros(0, device=device)

        group_aux_sq = flat_aux.square().sum()
        overlap_task_sq = flat_task.square().sum()
        group_dot = (flat_aux * flat_task).sum()
        numeric_valid = (
            numeric_valid
            & torch.isfinite(group_aux_sq)
            & torch.isfinite(overlap_task_sq)
            & torch.isfinite(group_dot)
        )
        aux_sq = aux_sq + group_aux_sq
        dot = dot + group_dot
        conflict_tolerance = 64.0 * torch.finfo(flat_aux.dtype).eps * (
            group_aux_sq * overlap_task_sq
        ).clamp_min(0.0).sqrt()
        conflict = group_dot < -conflict_tolerance
        conflict_flags.append(conflict)
        group_cosines.append(
            group_dot / (group_aux_sq.sqrt() * overlap_task_sq.sqrt()).clamp_min(1e-20)
        )
        coeff = torch.where(
            conflict,
            group_dot / overlap_task_sq.clamp_min(1e-20),
            torch.zeros_like(group_dot),
        )
        projected_group = flat_aux - coeff * flat_task
        # Re-project the float32 residual.  Removing a large conflicting component can
        # suffer catastrophic cancellation; one correction recovers orthogonality to
        # the combined task direction without a host synchronization.
        residual_dot = (projected_group * flat_task).sum()
        residual_coeff = torch.where(
            conflict,
            residual_dot / overlap_task_sq.clamp_min(1e-20),
            torch.zeros_like(residual_dot),
        )
        projected_group = projected_group - residual_coeff * flat_task

        # The combined actor+critic direction can hide damage to either objective.
        # Veto a whole block if it remains anti-aligned with either one separately.
        veto = torch.zeros((), dtype=torch.bool, device=device)
        for constraint in safety_tasks:
            flat_constraint = torch.cat(
                [
                    torch.zeros_like(auxiliary[index]).float().reshape(-1)
                    if constraint[index] is None
                    else constraint[index].detach().float().reshape(-1)
                    for index in active_indices
                ]
            ) if active_indices else torch.zeros(0, device=device)
            constraint_dot = (projected_group * flat_constraint).sum()
            # PCGrad can subtract a large component and leave a tiny orthogonal
            # residual.  Its float32 roundoff scales with the *pre-projection* vector,
            # not only the residual.  Basing tolerance on the residual alone caused
            # ~10% false vetoes for mathematically aligned actor/critic objectives.
            projection_roundoff_sq = torch.maximum(
                projected_group.square().sum(), group_aux_sq
            )
            constraint_tolerance = 64.0 * torch.finfo(projected_group.dtype).eps * (
                projection_roundoff_sq * flat_constraint.square().sum()
            ).clamp_min(0.0).sqrt()
            numeric_valid = (
                numeric_valid
                & torch.isfinite(constraint_dot)
                & torch.isfinite(constraint_tolerance)
            )
            # Functional accumulation avoids aliasing retained scalar bookkeeping.
            veto = veto | (constraint_dot < -constraint_tolerance)
        veto_flags.append(veto)
        projected_group = projected_group * (~veto).to(projected_group.dtype)
        group_records.append(
            (active_indices, sizes, projected_group, full_group_task_sq.sqrt())
        )

    # A block with an active task direction receives at most the requested fraction
    # of that block's task norm. Completely dormant blocks are held fixed unless the
    # entire task trunk is dormant, in which case the global absolute budget applies.
    task_norm = task_sq.sqrt()
    task_active = task_norm > 0
    capped_group_records = []
    projected_sq = zero.clone()
    for active_indices, sizes, projected_group, group_task_norm in group_records:
        group_projected_norm = projected_group.square().sum().sqrt()
        numeric_valid = numeric_valid & torch.isfinite(group_projected_norm)
        group_limit = torch.where(
            task_active,
            task_ratio_cap * group_task_norm,
            torch.full((), float("inf"), device=device),
        )
        group_scale = torch.minimum(
            torch.ones((), device=device),
            group_limit / group_projected_norm.clamp_min(1e-20),
        )
        projected_group = projected_group * group_scale
        projected_sq = projected_sq + projected_group.square().sum()
        capped_group_records.append((active_indices, sizes, projected_group))

    projected_norm = projected_sq.sqrt()
    numeric_valid = numeric_valid & torch.isfinite(projected_norm)
    limit = torch.as_tensor(absolute_cap, device=device)
    scale = torch.minimum(torch.ones((), device=device), limit / projected_norm.clamp_min(1e-20))
    numeric_valid = numeric_valid & torch.isfinite(scale)
    delivered = [None] * len(auxiliary)
    delivered_groups = [record[2].mul(scale) for record in capped_group_records]
    if delivered_groups:
        delivered_group_sizes = [gradient.numel() for gradient in delivered_groups]
        flat_delivered = torch.cat(delivered_groups)
        numeric_valid = numeric_valid & torch.isfinite(flat_delivered).all()
        # One final vector selection makes containment transaction-wide even if an
        # intermediate projection/cap scalar overflowed after the initial aux check.
        flat_delivered = torch.where(
            numeric_valid, flat_delivered, torch.zeros_like(flat_delivered)
        )
        delivered_groups = flat_delivered.split(delivered_group_sizes)
    for (active_indices, sizes, _), delivered_group in zip(
        capped_group_records, delivered_groups
    ):
        for index, flat_gradient in zip(active_indices, delivered_group.split(sizes)):
            delivered[index] = flat_gradient.view_as(auxiliary[index]).to(auxiliary[index].dtype)
    raw_aux_norm = aux_sq.sqrt()
    cosine = dot / (raw_aux_norm * task_norm).clamp_min(1e-20)
    conflicts = torch.stack(conflict_flags) if conflict_flags else torch.zeros(1, device=device, dtype=torch.bool)
    vetoes = torch.stack(veto_flags) if veto_flags else torch.zeros(1, device=device, dtype=torch.bool)
    local_cosines = torch.stack(group_cosines) if group_cosines else torch.zeros(1, device=device)
    diagnostics = {
        "raw_norm": raw_aux_norm.detach(),
        "projected_norm": projected_norm.detach(),
        "delivered_norm": (projected_norm * scale).detach(),
        "task_norm": task_norm.detach(),
        "cosine": cosine.detach(),
        "conflict": conflicts.any().detach().to(torch.float32),
        "conflict_fraction": conflicts.float().mean().detach(),
        "objective_veto_fraction": vetoes.float().mean().detach(),
        "worst_group_cosine": local_cosines.min().detach(),
        "numeric_valid": numeric_valid.detach().to(torch.float32),
        "aux_numeric_valid": aux_valid.detach().to(torch.float32),
        "task_numeric_valid": (task_valid & safety_valid).detach().to(torch.float32),
    }
    diagnostics = {
        name: torch.where(
            torch.isfinite(value),
            value,
            torch.zeros_like(value),
        )
        for name, value in diagnostics.items()
    }
    return delivered, diagnostics


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, decay: float) -> None:
    """EMA parameters and copy buffers without ever constructing a target gradient."""
    if not 0.0 <= decay <= 1.0:
        raise ValueError("EMA decay must be in [0, 1]")
    target_params = dict(target.named_parameters())
    online_params = dict(online.named_parameters())
    if target_params.keys() != online_params.keys():
        raise ValueError("EMA modules do not have matching parameters")
    for name, target_param in target_params.items():
        target_param.lerp_(online_params[name].detach(), 1.0 - decay)
    target_buffers = dict(target.named_buffers())
    online_buffers = dict(online.named_buffers())
    if target_buffers.keys() != online_buffers.keys():
        raise ValueError("EMA modules do not have matching buffers")
    for name, target_buffer in target_buffers.items():
        target_buffer.copy_(online_buffers[name].detach())


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
    norm_adv: bool = True            # ppoadvnorm_batch base: standard PPO advantage standardization ON
    # --- Percentile advantage normalization (OFF in the ppoadvnorm_batch base) ---
    ret_percnorm: bool = False       # scale policy advantage by S = max(floor, P95-P5) of returns
    ret_perc_scope: str = "minibatch"  # "minibatch" = fresh per-mb P95-P5 of mb returns (NO EMA, like the
    #                                  old per-mb advnorm); "batch" = fresh WHOLE-ROLLOUT P95-P5, one S per
    #                                  update, NO EMA (the batch-vs-mb ablation); "ema" = v1's global EMA.
    ret_perc_rate: float = 0.01      # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
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
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it).
    target_kl: float = 0.03

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Dreamer3-style bucket critic: centers are symexp(linspace(v_min, v_max, num_bins)).
    # Defaults match v162's ±20k raw support, expressed in symlog coordinates.
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    # Keep observation normalization, but disable return/reward normalization for
    # this ablation. NormalizeReward tracks discounted returns internally and
    # divides rewards by their running std, so it must remain off here.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "v10"       # d3percnorm: identity -- NO rankgauss. DreamerV3 percentile
    #                                  norm (below) is the sole advantage scaler ("no advantage norm").

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "batch"    # ppoadvnorm_batch base: standardize ONCE over the whole rollout

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

    # v10 action-conditioned successor-innovation TD.  The (1-gamma) convention
    # equalizes infinite-horizon mass; direct prefixes make every target substantially
    # observed before a single EMA bootstrap is admitted at the tail.
    successor_td: bool = True
    successor_gammas: tuple[float, ...] = (0.60, 0.90, 0.97, 0.99)
    successor_direct_horizons: tuple[int, ...] = (4, 8, 16, 64)
    successor_coef: float = 1.0
    successor_direct_coef: float = 0.50
    successor_reward_coef: float = 0.25
    successor_value_coef: float = 0.20
    successor_critic_semantic_coef: float = 0.10
    successor_policy_coef: float = 0.05
    # Updated once after all PPO epochs: 0.95 => 13.5-iteration half-life.
    successor_ema_decay: float = 0.95
    innovation_scale_decay: float = 0.95
    successor_trunk_grad_clip: float = 0.03
    successor_predictor_grad_clip: float = 0.50
    successor_task_ratio: float = 0.25

    # Compile only pure, fixed-shape hot paths; target construction and gradient
    # projection intentionally stay eager and auditable.
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
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    # Plain pre-act MLP. Standard sqrt(2) init on both Linears.
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        ReLUSquared(),
        layer_init(nn.Linear(H, H)),
    )


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
        self.dense = _branch_body(H)

        # Soft MoE branch (softmax over all experts, no top-K).
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in))                   # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1) # (B, E, H)
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

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class SuccessorInnovationModel(nn.Module):
    """One inexpensive action-conditioned model with direct and successor readouts."""

    def __init__(self, hidden: int, act_dim: int, num_bands: int):
        super().__init__()
        self.hidden = hidden
        self.num_bands = num_bands
        self.feature_norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.action_proj = layer_init(nn.Linear(act_dim, hidden), std=1.0)
        self.context = nn.Sequential(
            layer_init(nn.Linear(2 * hidden, 2 * hidden)),
            ReLUSquared(),
            layer_init(nn.Linear(2 * hidden, hidden)),
            ReLUSquared(),
        )
        self.innovation_head = layer_init(nn.Linear(hidden, hidden), std=0.05)
        self.successor_head = layer_init(nn.Linear(hidden, num_bands * hidden), std=0.05)
        self.reward_head = layer_init(nn.Linear(hidden, 1), std=0.05)
        self.value_head = layer_init(nn.Linear(hidden, 1), std=0.05)

    def forward(self, feature: torch.Tensor, action: torch.Tensor):
        action_feature = torch.tanh(self.action_proj(action))
        context = self.context(torch.cat([self.feature_norm(feature), action_feature], dim=-1))
        innovation = self.innovation_head(context)
        successors = self.successor_head(context).view(-1, self.num_bands, self.hidden)
        reward = self.reward_head(context).squeeze(-1)
        value = self.value_head(context).squeeze(-1)
        return innovation, successors, reward, value


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # v162 critic: bias-free neutral MTP head. With symmetric symlog support,
        # zero logits decode to a zero raw value without a hidden prior.
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
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
        # Construct the online model last so the PPO base retains its initialization.
        self.num_successor_bands = len(args.successor_gammas)
        self.successor_model = SuccessorInnovationModel(H, act_dim, self.num_successor_bands)
        online_encoder = self.trunk if self.share_backbone else self.actor_trunk
        self.target_encoder = copy.deepcopy(online_encoder)
        self.target_successor_model = copy.deepcopy(self.successor_model)
        for module in (self.target_encoder, self.target_successor_model):
            module.requires_grad_(False)
            module.eval()
        self.register_buffer("innovation_scale", torch.ones(H))
        self.register_buffer("innovation_scale_initialized", torch.tensor(False))

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

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns value LOGITS (B, mtp, num_bins); horizon 0 is V(s_t).
        critic_feat = self.get_critic_feat(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_value_h0(self, x):
        """Only V(s_t) logits, avoiding the 6x full-rollout MTP output allocation."""
        critic_feat = self.get_critic_feat(x)
        return F.linear(critic_feat, self.critic_head.weight[: self.num_bins], None)

    def get_actor_feat(self, x):
        # Actor-side trunk feature used by the online successor model.
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        return trunk(x)

    def get_critic_feat(self, x):
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return trunk(x)

    @torch.no_grad()
    def sample_frozen_policy_action(self, x):
        """Sample an env-space action before PPO updates for successor bootstraps."""
        actor_feat = self.get_actor_feat(x)
        dist, to_action, _ = self._actor_dist(actor_feat)
        z = dist.sample()
        if self.actor_dist == "beta":
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        return to_action(z)

    @torch.no_grad()
    def get_target_feat(self, x):
        return self.target_encoder(x)

    def get_successor_predictions(self, feature, action):
        return self.successor_model(feature, action)

    @staticmethod
    def reconstruct_next_feature(feature, innovation_prediction):
        """Innovation predictions live in latent units; RMS never changes coordinates."""
        return feature + innovation_prediction

    @torch.no_grad()
    def get_target_successors(self, feature, action):
        return self.target_successor_model(feature, action)[1]

    @torch.no_grad()
    def update_successor_targets(self, decay: float):
        online_encoder = self.trunk if self.share_backbone else self.actor_trunk
        ema_update(self.target_encoder, online_encoder, decay)
        ema_update(self.target_successor_model, self.successor_model, decay)

    @torch.no_grad()
    def update_innovation_scale(self, batch_scale: torch.Tensor, decay: float):
        if batch_scale.shape != self.innovation_scale.shape:
            raise ValueError("innovation scale has the wrong shape")
        if not bool(self.innovation_scale_initialized):
            self.innovation_scale.copy_(batch_scale.detach())
            self.innovation_scale_initialized.fill_(True)
        else:
            self.innovation_scale.lerp_(batch_scale.detach(), 1.0 - decay)

    def frozen_actor_dist(self, feature):
        """Policy decoder as a differentiable-in-feature, read-only semantic probe."""
        if self.actor_dist == "gaussian":
            mean = F.linear(feature, self.actor_head.weight.detach(), self.actor_head.bias.detach())
            raw_lv = F.linear(
                feature,
                self.actor_logvar_head.weight.detach(),
                self.actor_logvar_head.bias.detach(),
            )
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            return Normal(mean, (0.5 * lv).exp())
        alpha = 1.0 + F.softplus(
            F.linear(feature, self.actor_alpha_head.weight.detach(), self.actor_alpha_head.bias.detach())
        )
        beta = 1.0 + F.softplus(
            F.linear(feature, self.actor_beta_head.weight.detach(), self.actor_beta_head.bias.detach())
        )
        return Beta(alpha, beta)

    def frozen_value_logits(self, feature):
        """Critic decoder with detached weights: gradients flow only to the feature."""
        logits = F.linear(feature, self.critic_head.weight.detach(), None)
        return logits.view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_action_and_value(self, x, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_logits = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)
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
        return action, z, log_prob, entropy, value_logits, actor_feat

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

    def successor_trunk_parameters(self):
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        return list(trunk.parameters())

    def successor_trunk_parameter_groups(self):
        """Structural group id for entry, each think block, and output projection."""
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        modules = [trunk.entry, *trunk.blocks, trunk.out_proj]
        group_by_parameter = {
            id(parameter): group
            for group, module in enumerate(modules)
            for parameter in module.parameters()
        }
        parameters = self.successor_trunk_parameters()
        if any(id(parameter) not in group_by_parameter for parameter in parameters):
            raise RuntimeError("successor trunk parameter is missing a structural group")
        return [group_by_parameter[id(parameter)] for parameter in parameters]

    def successor_predictor_parameters(self):
        return list(self.successor_model.parameters())

    def task_parameters(self):
        """PPO-owned parameters; private successor moments live in their own Adam."""
        private_ids = {id(parameter) for parameter in self.successor_model.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in private_ids
        ]

    def online_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


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
    assert args.norm_adv_scope in ("batch", "minibatch", "batch_retstd", "minibatch_retstd")
    assert len(args.successor_gammas) == len(args.successor_direct_horizons) > 0
    assert all(0.0 < gamma < 1.0 for gamma in args.successor_gammas)
    assert all(horizon >= 1 for horizon in args.successor_direct_horizons)
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope in ("batch", "batch_retstd")), \
        "norm_adv_scope=batch/batch_retstd requires adv_transform_scope=batch"
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

    agent = Agent(envs, args).to(device)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    successor_trunk_params = agent.successor_trunk_parameters()
    successor_trunk_groups = agent.successor_trunk_parameter_groups()
    successor_predictor_params = agent.successor_predictor_parameters()
    task_params = agent.task_parameters()
    # Adam is coordinate-local, so partitioning the private successor parameters is
    # update-equivalent on valid batches.  It additionally lets a rejected auxiliary
    # zero-advance/rollback its own moments without touching PPO's optimizer state.
    optimizer = optim.Adam(task_params, lr=args.learning_rate, eps=1e-5)
    successor_optimizer = (
        optim.Adam(successor_predictor_params, lr=args.learning_rate, eps=1e-5)
        if args.successor_td
        else None
    )

    def policy_rollout_fn(obs_):
        return agent.get_action_and_value(obs_)

    def policy_update_fn(obs_, z_):
        return agent.get_action_and_value(obs_, z_)

    def value_flat_fn(obs_):
        return agent.get_value_h0(obs_)

    # Separate wrappers reduce graph coupling, while capture_target_latent_tables below
    # establishes actual tensor ownership. Wrappers alone cannot guarantee disjoint
    # output storage when compile caches or CUDA-graph trees share a memory pool.
    def target_current_feat_fn(obs_):
        return agent.get_target_feat(obs_)

    def target_next_feat_fn(obs_):
        return agent.get_target_feat(obs_)

    if args.compile:
        policy_rollout_fn = torch.compile(policy_rollout_fn, mode=args.compile_mode, dynamic=False)
        policy_update_fn = torch.compile(policy_update_fn, mode=args.compile_mode, dynamic=False)
        value_flat_fn = torch.compile(value_flat_fn, mode=args.compile_mode, dynamic=False)
        target_current_feat_fn = torch.compile(
            target_current_feat_fn, mode=args.compile_mode, dynamic=False
        )
        target_next_feat_fn = torch.compile(
            target_next_feat_fn, mode=args.compile_mode, dynamic=False
        )
        print(f"compiled fixed-shape PPO and successor paths ({args.compile_mode})")

    def mark_compile_step():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()

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
    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
    )
    hl_support_cpu = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )
    support = hl_support.support                       # Dreamer3 raw bucket centers
    bin_width = hl_support.bin_width
    raw_support = support

    def value_logits_to_scalar(logits):
        return hl_support.to_scalar(logits)

    sigma_floor = args.sigma_floor_bins * bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    latents = torch.zeros((args.num_steps, args.num_envs, args.hidden)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            if successor_optimizer is not None:
                successor_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                mark_compile_step()
                action, z, logprob, ent, value_logits, actor_feat = policy_rollout_fn(next_obs)
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
                latents[step] = actor_feat
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obses[step] = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            mark_compile_step()
            next_transition_value_logits = value_flat_fn(
                next_obses.reshape((-1,) + envs.single_observation_space.shape)
            )
            next_transition_values = value_logits_to_scalar(next_transition_value_logits).reshape(
                args.num_steps, args.num_envs
            )
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample a' ~ π(·|s_T) for the bootstrap entropy (SAC's single-sample).
                mark_compile_step()
                _, _, boot_logprob, _, _, _ = policy_rollout_fn(next_obses[-1])
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
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
            # Batch-level percentile advantage normalization (scopes "ema" and "batch"). Both compute the
            # whole-rollout P5/P95 once and scale policy_adv by one S; "ema" smooths the percentiles with a
            # global EMA across iterations (v1), "batch" uses the FRESH per-rollout spread (no EMA -- the
            # batch-vs-mb ablation). scope=="minibatch" SKIPS this and scales fresh per-mb in the update loop,
            # leaving policy_adv RAW here. Divide-only; critic target `returns` stays RAW (valnorm=none).
            if args.ret_percnorm and args.ret_perc_scope in ("ema", "batch"):
                flat_ret = returns.reshape(-1)
                qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=device)
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                if args.ret_perc_scope == "ema":
                    if not ema_perc_inited:
                        ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
                    else:
                        r = args.ret_perc_rate
                        ema_ret_lo += r * (lo - ema_ret_lo)
                        ema_ret_hi += r * (hi - ema_ret_hi)
                    spread = ema_ret_hi - ema_ret_lo
                else:  # "batch": fresh whole-rollout percentile spread, no EMA
                    spread = hi - lo
                ret_perc_scale = max(args.ret_perc_floor, spread)
                policy_adv = policy_adv / ret_perc_scale
            # v162 critic target: scalar-return HL-Gauss MTP. Horizon 0 regresses
            # returns[t]; horizon h regresses returns[t+h] from the same features.
            # A future target is valid only when no reset boundary lies between
            # the source state and target state, and when it stays inside rollout.
            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros(
                (*returns.shape, mtp), dtype=torch.bool, device=returns.device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=returns.device
                )
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                return_mtp[:valid_len, :, h] = returns[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            # Project on CPU, then transfer once. Float16 label storage is ~192 MiB
            # at defaults instead of ~383 MiB; multiplication with float32 log-probs
            # promotes the CE arithmetic, and no minibatch incurs a host transfer.
            target_probs, return_mtp_mask = store_critic_targets(
                hl_support_cpu.project(return_mtp.detach().cpu()),
                return_mtp_mask,
                device,
            )
            if args.successor_td:
                flat_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                flat_next_obs = next_obses.reshape((-1,) + envs.single_observation_space.shape)
                target_latents, target_next_latents = capture_target_latent_tables(
                    target_current_feat_fn,
                    target_next_feat_fn,
                    flat_obs,
                    flat_next_obs,
                    mark_compile_step,
                )
                target_latents = target_latents.reshape(
                    args.num_steps, args.num_envs, args.hidden
                )
                target_next_latents = target_next_latents.reshape(
                    args.num_steps, args.num_envs, args.hidden
                )
                raw_innovations = target_next_latents - target_latents
                batch_innovation_scale = masked_feature_rms(raw_innovations, transition_valids.bool())
                agent.update_innovation_scale(batch_innovation_scale, args.innovation_scale_decay)
                innovation_scale = agent.innovation_scale.detach().clone()
                target_tail_predictions = agent.get_target_successors(
                    target_latents.reshape(-1, args.hidden), actions.reshape(-1, actions.shape[-1])
                ).reshape(
                    args.num_steps,
                    args.num_envs,
                    len(args.successor_gammas),
                    args.hidden,
                )
                # Interior n-step tails use stored behavior actions. Only truncation
                # endpoints and the artificial rollout edge lack one, so keep this
                # eager path sparse instead of running another 32k-row actor trunk.
                bootstrap_rows = next_state_bootstrap_rows(
                    transition_terminations,
                    transition_boundaries,
                    transition_valids,
                )
                flat_bootstrap_rows = bootstrap_rows.reshape(-1)
                bootstrap_indices = flat_bootstrap_rows.nonzero(as_tuple=False).squeeze(-1)
                flat_next_state_tail_predictions = target_tail_predictions.new_zeros(
                    args.batch_size,
                    len(args.successor_gammas),
                    args.hidden,
                )
                if bootstrap_indices.numel() > 0:
                    bootstrap_observations = flat_next_obs[bootstrap_indices]
                    next_state_actions = agent.sample_frozen_policy_action(
                        bootstrap_observations
                    )
                    flat_next_state_tail_predictions[bootstrap_indices] = (
                        agent.get_target_successors(
                            target_next_latents.reshape(-1, args.hidden)[bootstrap_indices],
                            next_state_actions,
                        )
                    )
                next_state_tail_predictions = flat_next_state_tail_predictions.reshape(
                    args.num_steps,
                    args.num_envs,
                    len(args.successor_gammas),
                    args.hidden,
                )
                successor_targets = build_successor_targets(
                    raw_innovations,
                    target_tail_predictions,
                    next_state_tail_predictions,
                    transition_terminations,
                    transition_boundaries,
                    transition_valids,
                    args.successor_gammas,
                    args.successor_direct_horizons,
                )
                reward_mean = rewards.mean()
                reward_scale = rewards.std(unbiased=False).clamp_min(0.1)
                normalized_rewards = (rewards - reward_mean) / reward_scale
                return_mean = returns.mean()
                return_scale = returns.std(unbiased=False).clamp_min(1.0)
                normalized_anchor_returns = (returns - return_mean) / return_scale
                latent_batch_std = latents.reshape(-1, args.hidden).std(dim=0).mean().item()
                _, successor_target_rms = masked_scaled_smooth_l1(
                    torch.zeros_like(successor_targets.values),
                    successor_targets.values,
                    successor_targets.masks,
                )
                successor_bootstrap_frac = successor_targets.bootstrap_masks.float().mean(dim=(0, 1))
                successor_observed_mass = successor_targets.observed_mass.mean(dim=(0, 1))
            # Per-state return std probe from the OLD rollout Z(s_t), decoded to
            # raw return units. The default rankgauss path does not consume this.
            sigma = (value_probs * (raw_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(sigma_floor)
            # CDF-rank u in Dreamer3 bucket order; intervals are uniform in symlog
            # coordinate even though raw bucket centers are exponentially spaced.
            cdf_frac = hl_support.cdf_fraction(returns)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)
        b_u = u.reshape(-1)
        if args.successor_td:
            b_successor_targets = successor_targets.values.reshape(
                -1, len(args.successor_gammas), args.hidden
            )
            b_successor_masks = successor_targets.masks.reshape(-1, len(args.successor_gammas))
            b_direct_innovations = raw_innovations.reshape(-1, args.hidden)
            b_target_next_latents = target_next_latents.reshape(-1, args.hidden)
            b_transition_valids = transition_valids.reshape(-1).bool()
            b_transition_terminations = transition_terminations.reshape(-1).bool()
            b_normalized_rewards = normalized_rewards.reshape(-1)
            b_normalized_anchor_returns = normalized_anchor_returns.reshape(-1)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            b_policy_adv_normed = b_policy_adv / b_returns.std().clamp(min=args.ret_perc_floor)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mark_compile_step()
                _, _, newlogprob, entropy, value_logits, mb_actor_feat = policy_update_fn(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_advantages = mb_advantages / b_returns[mb_inds].std().clamp(min=args.ret_perc_floor)
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds]
                    qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device)
                    lo, hi = torch.quantile(mb_ret, qs)
                    mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                    mb_advantages = mb_advantages / mb_perc_scale
                    ret_perc_scale = mb_perc_scale.item()

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

                # Asymmetric "clip-higher" when clip_coef_high is set: looser upper bound
                # gives positive-advantage actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # v162 HL-Gauss MTP value loss: per-horizon CE to scalar-return
                # targets, summed across valid horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds]
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(dtype=value_ce.dtype)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

                if args.successor_td:
                    (
                        direct_prediction,
                        successor_prediction,
                        reward_prediction,
                        return_prediction,
                    ) = agent.get_successor_predictions(mb_actor_feat, b_actions[mb_inds])
                    successor_loss, successor_loss_scales = masked_scaled_smooth_l1(
                        successor_prediction,
                        b_successor_targets[mb_inds],
                        b_successor_masks[mb_inds],
                        scale=successor_target_rms,
                    )
                    valid_mask = b_transition_valids[mb_inds]
                    valid_float = valid_mask.to(mb_actor_feat.dtype)
                    valid_denom = valid_float.sum().clamp_min(1.0)
                    direct_per_row = F.smooth_l1_loss(
                        direct_prediction / innovation_scale,
                        b_direct_innovations[mb_inds] / innovation_scale,
                        reduction="none",
                    ).mean(-1)
                    direct_loss = (direct_per_row * valid_float).sum() / valid_denom
                    reward_per_row = F.smooth_l1_loss(
                        reward_prediction,
                        b_normalized_rewards[mb_inds],
                        reduction="none",
                    )
                    # Reward is observed even when an environment omits its final
                    # observation, so it intentionally has no transition-valid mask.
                    reward_anchor_loss = reward_per_row.mean()
                    return_anchor_loss = F.smooth_l1_loss(
                        return_prediction,
                        b_normalized_anchor_returns[mb_inds],
                    )

                    # Decode the predicted next latent through read-only PPO heads.  Their
                    # detached weights preserve policy/value semantics while making it
                    # impossible for the auxiliary to rewrite either behavior decoder.
                    predicted_next_feature = agent.reconstruct_next_feature(
                        mb_actor_feat, direct_prediction
                    )
                    target_next_feature = b_target_next_latents[mb_inds]
                    semantic_mask = valid_mask & ~b_transition_terminations[mb_inds]
                    semantic_float = semantic_mask.to(mb_actor_feat.dtype)
                    semantic_denom = semantic_float.sum().clamp_min(1.0)
                    teacher_policy = agent.frozen_actor_dist(target_next_feature)
                    student_policy = agent.frozen_actor_dist(predicted_next_feature)
                    policy_per_row = torch.distributions.kl_divergence(
                        teacher_policy, student_policy
                    ).sum(-1) / b_actions.shape[-1]
                    policy_anchor_loss = (
                        policy_per_row * semantic_float
                    ).sum() / semantic_denom

                    teacher_value_logp = torch.log_softmax(
                        agent.frozen_value_logits(target_next_feature)[:, 0], dim=-1
                    )
                    student_value_logp = torch.log_softmax(
                        agent.frozen_value_logits(predicted_next_feature)[:, 0], dim=-1
                    )
                    teacher_value_prob = teacher_value_logp.detach().exp()
                    value_per_row = (
                        teacher_value_prob * (teacher_value_logp.detach() - student_value_logp)
                    ).sum(-1)
                    critic_semantic_loss = (
                        value_per_row * semantic_float
                    ).sum() / semantic_denom
                    successor_aux_loss = (
                        successor_loss
                        + args.successor_direct_coef * direct_loss
                        + args.successor_reward_coef * reward_anchor_loss
                        + args.successor_value_coef * return_anchor_loss
                        + args.successor_critic_semantic_coef * critic_semantic_loss
                        + args.successor_policy_coef * policy_anchor_loss
                    )
                    with torch.no_grad():
                        successor_weights = b_successor_masks[mb_inds].unsqueeze(-1)
                        successor_sse = (
                            (successor_prediction - b_successor_targets[mb_inds]).square()
                            * successor_weights
                        ).sum()
                        successor_mean = (
                            b_successor_targets[mb_inds] * successor_weights
                        ).sum() / (successor_weights.sum() * args.hidden).clamp_min(1)
                        successor_sst = (
                            (b_successor_targets[mb_inds] - successor_mean).square()
                            * successor_weights
                        ).sum()
                        successor_r2 = 1.0 - successor_sse / successor_sst.clamp_min(1e-8)
                else:
                    successor_aux_loss = None

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

                if args.separate_grad_clip:
                    # Compute the two PPO gradients first. Their clipped sum remains the
                    # exact task direction; task nonfinites abort instead of being hidden
                    # by the auxiliary fail-closed path.
                    agent.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(
                        critic_params,
                        args.critic_grad_clip,
                        error_if_nonfinite=True,
                    )
                    value_grads = {
                        parameter: parameter.grad.detach().clone()
                        for parameter in critic_params
                        if parameter.grad is not None
                    }
                    agent.zero_grad(set_to_none=True)
                    (pg_loss - ent_coef_eff * entropy_loss).backward(
                        retain_graph=args.successor_td
                    )
                    actor_gn = nn.utils.clip_grad_norm_(
                        actor_params,
                        args.actor_grad_clip,
                        error_if_nonfinite=True,
                    )
                    actor_grads = {
                        parameter: parameter.grad.detach().clone()
                        for parameter in actor_params
                        if parameter.grad is not None
                    }
                    safety_gradient_maps = (actor_grads, value_grads)
                else:
                    # The legacy global-clip mode still gets the same containment. It has
                    # one combined task half-space rather than separate actor/critic vetoes.
                    task_loss = (
                        pg_loss - ent_coef_eff * entropy_loss + args.vf_coef * v_loss
                    )
                    agent.zero_grad(set_to_none=True)
                    task_loss.backward(retain_graph=args.successor_td)
                    task_gn = nn.utils.clip_grad_norm_(
                        task_params,
                        args.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    actor_gn = critic_gn = task_gn
                    actor_grads = {
                        parameter: parameter.grad.detach().clone()
                        for parameter in task_params
                        if parameter.grad is not None
                    }
                    value_grads = {}
                    safety_gradient_maps = ()

                successor_grads = {}
                if args.successor_td:
                    task_trunk_grads = []
                    actor_trunk_grads = []
                    value_trunk_grads = []
                    for parameter in successor_trunk_params:
                        actor_grad = actor_grads.get(parameter)
                        value_grad = value_grads.get(parameter)
                        actor_trunk_grads.append(actor_grad)
                        value_trunk_grads.append(value_grad)
                        if actor_grad is None:
                            task_grad = value_grad
                        elif value_grad is None:
                            task_grad = actor_grad
                        else:
                            task_grad = actor_grad + value_grad
                        task_trunk_grads.append(task_grad)

                    # Backpropagation may materialize invalid values; containment begins
                    # before clipping or projection, so no invalid*0 expression is used.
                    agent.zero_grad(set_to_none=True)
                    (args.successor_coef * successor_aux_loss).backward()
                    raw_aux_trunk_grads = [
                        None if parameter.grad is None else parameter.grad.detach().clone()
                        for parameter in successor_trunk_params
                    ]
                    raw_predictor_grads = [
                        None if parameter.grad is None else parameter.grad.detach().clone()
                        for parameter in successor_predictor_params
                    ]
                    flat_auxiliary_gradients = torch.cat(
                        [
                            gradient.detach().float().reshape(-1)
                            for gradient in raw_aux_trunk_grads + raw_predictor_grads
                            if gradient is not None
                        ]
                    )
                    auxiliary_transaction_valid = auxiliary_gradients_are_adam_safe(
                        [flat_auxiliary_gradients],
                        torch.isfinite(successor_aux_loss.detach()),
                    )
                    delivered_trunk_grads, successor_grad_diagnostics = (
                        project_and_cap_auxiliary_gradients(
                            raw_aux_trunk_grads,
                            task_trunk_grads,
                            args.successor_trunk_grad_clip,
                            args.successor_task_ratio,
                            successor_trunk_groups,
                            tuple(
                                [gradient_map.get(parameter) for parameter in successor_trunk_params]
                                for gradient_map in safety_gradient_maps
                            ),
                            transaction_valid=auxiliary_transaction_valid,
                        )
                    )
                    (
                        clipped_predictor_grads,
                        successor_predictor_gn,
                        predictor_gradient_valid,
                    ) = clip_auxiliary_gradients_fail_closed(
                        raw_predictor_grads,
                        args.successor_predictor_grad_clip,
                        auxiliary_transaction_valid,
                    )
                    successor_grads = {
                        parameter: gradient
                        for parameter, gradient in zip(
                            successor_trunk_params, delivered_trunk_grads
                        )
                        if gradient is not None
                    }

                agent.zero_grad(set_to_none=True)
                install_gradient_transaction(
                    task_params,
                    (actor_grads, value_grads),
                    successor_grads,
                )
                optimizer.step()

                if args.successor_td:
                    assert successor_optimizer is not None
                    successor_private_diagnostics = (
                        apply_private_auxiliary_optimizer_transaction(
                            successor_optimizer,
                            successor_predictor_params,
                            clipped_predictor_grads,
                            successor_grad_diagnostics["numeric_valid"]
                            * predictor_gradient_valid.to(torch.float32),
                        )
                    )

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Targets label the whole rollout from one fixed snapshot and advance once
        # after all online PPO epochs. The decay is therefore iteration-scale, not an
        # accidental function of minibatch count or KL early stopping.
        if args.successor_td:
            agent.update_successor_targets(args.successor_ema_decay)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        # Promote resident fp16 labels before the large reduction; a fp16 mask
        # denominator would overflow above 65,504 valid horizon labels.
        edge_per_h = b_target_probs[..., 0].float() + b_target_probs[..., -1].float()
        edge_mask_f = b_target_mask.to(dtype=torch.float32)
        edge_mass = ((edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)).item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        if args.successor_td:
            with torch.no_grad():
                online_encoder = agent.trunk if agent.share_backbone else agent.actor_trunk
                ema_delta_sq = sum(
                    (
                        online_parameter.detach() - target_parameter.detach()
                    ).float().square().sum()
                    for online_parameter, target_parameter in zip(
                        online_encoder.parameters(), agent.target_encoder.parameters()
                    )
                )
                online_sq = sum(
                    parameter.detach().float().square().sum()
                    for parameter in online_encoder.parameters()
                )
                ema_relative_drift = (ema_delta_sq / online_sq.clamp_min(1e-20)).sqrt()
            writer.add_scalar("successor/loss", finite_diagnostic(successor_loss).item(), global_step)
            writer.add_scalar("successor/direct_loss", finite_diagnostic(direct_loss).item(), global_step)
            writer.add_scalar(
                "successor/reward_anchor_loss",
                finite_diagnostic(reward_anchor_loss).item(),
                global_step,
            )
            writer.add_scalar(
                "successor/value_anchor_loss",
                finite_diagnostic(return_anchor_loss).item(),
                global_step,
            )
            writer.add_scalar(
                "successor/critic_semantic_loss",
                finite_diagnostic(critic_semantic_loss).item(),
                global_step,
            )
            writer.add_scalar(
                "successor/policy_anchor_loss",
                finite_diagnostic(policy_anchor_loss).item(),
                global_step,
            )
            writer.add_scalar("successor/model_r2", finite_diagnostic(successor_r2).item(), global_step)
            writer.add_scalar(
                "successor/ema_relative_drift",
                finite_diagnostic(ema_relative_drift).item(),
                global_step,
            )
            writer.add_scalar(
                "successor/latent_batch_std",
                finite_diagnostic(torch.as_tensor(latent_batch_std, device=device)).item(),
                global_step,
            )
            writer.add_scalar(
                "successor/innovation_scale_mean",
                finite_diagnostic(innovation_scale.mean()).item(),
                global_step,
            )
            writer.add_scalar(
                "successor/innovation_scale_min",
                finite_diagnostic(innovation_scale.min()).item(),
                global_step,
            )
            writer.add_scalar(
                "successor/innovation_scale_max",
                finite_diagnostic(innovation_scale.max()).item(),
                global_step,
            )
            for name, value in successor_grad_diagnostics.items():
                writer.add_scalar(
                    f"successor/trunk_grad_{name}",
                    finite_diagnostic(value).item(),
                    global_step,
                )
            writer.add_scalar(
                "successor/predictor_grad_raw_norm",
                finite_diagnostic(successor_predictor_gn).item(),
                global_step,
            )
            for name, value in successor_private_diagnostics.items():
                writer.add_scalar(
                    f"successor/private_optimizer_{name}",
                    finite_diagnostic(value).item(),
                    global_step,
                )
            for band, gamma in enumerate(args.successor_gammas):
                tag = str(gamma).replace(".", "p")
                writer.add_scalar(
                    f"successor/band_{tag}_target_rms",
                    finite_diagnostic(successor_target_rms[band].mean()).item(),
                    global_step,
                )
                writer.add_scalar(
                    f"successor/band_{tag}_loss_scale",
                    finite_diagnostic(successor_loss_scales[band].mean()).item(),
                    global_step,
                )
                writer.add_scalar(
                    f"successor/band_{tag}_bootstrap_frac",
                    finite_diagnostic(successor_bootstrap_frac[band]).item(),
                    global_step,
                )
                writer.add_scalar(
                    f"successor/band_{tag}_observed_mass",
                    finite_diagnostic(successor_observed_mass[band]).item(),
                    global_step,
                )
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
            writer.add_scalar(
                "debug/soft_adv_std_ratio",
                (policy_adv.std() / (advantages.std() + 1e-8)).item(),
                global_step,
            )
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)
        writer.add_scalar("debug/u_edge_frac", u_edge_frac, global_step)
        writer.add_scalar("debug/sigma_mean", b_sigma.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

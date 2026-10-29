"""Bounded-code CUDA GAE, preserving the backwards FP32 recurrence.

A program owns one environment and loads 32 consecutive transitions at once.
Only that tile is unrolled: the horizon loop stays rolled, so compilation does
not trace thousands of PyTorch assignments. No associative scan or reordered
recurrence is used. Unsupported inputs retain the public PyTorch computation.
The standard (non-explicit) path also accepts a ``(T, N)`` lambda tensor; see
:func:`cleanrl.shared.ppo_loop.compute_gae` for its per-step semantics.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gae_kernel(
    rewards_ptr,
    values_ptr,
    terms_ptr,
    truncs_ptr,
    next_ptr,
    tail_ptr,
    lambdas_ptr,
    advantages_ptr,
    returns_ptr,
    steps: tl.constexpr,
    envs: tl.constexpr,
    gamma: tl.constexpr,
    decay: tl.constexpr,
    explicit: tl.constexpr,
    per_step: tl.constexpr,
    tile: tl.constexpr,
):
    env = tl.program_id(0)
    lane = tl.arange(0, tile)
    running = tl.full((tile,), 0.0, tl.float32)
    # Reverse-aligned tiles make padding occur before t=0, never at the tail.
    for block in range(tl.cdiv(steps, tile)):
        time = steps - (block + 1) * tile + lane
        valid = time >= 0
        offset = time * envs + env
        reward = tl.load(rewards_ptr + offset, valid, other=0.0)
        value = tl.load(values_ptr + offset, valid, other=0.0)
        term = tl.load(terms_ptr + offset, valid, other=0.0)
        trunc = tl.load(truncs_ptr + offset, valid, other=0.0)
        next_value = tl.load(next_ptr + offset, valid, other=0.0)
        bootstrap = 1.0 - term
        if explicit:
            delta = reward + (gamma * next_value) * bootstrap - value
        else:
            shifted = tl.load(values_ptr + offset + envs, valid & (time + 1 < steps), other=0.0)
            tail = tl.load(tail_ptr + env)
            shifted = tl.where(time + 1 == steps, tail, shifted)
            next_value = tl.where(trunc != 0.0, next_value, shifted)
            delta = reward + (gamma * bootstrap) * next_value - value
            trace = 1.0 - tl.maximum(term, trunc, propagate_nan=tl.PropagateNan.ALL)
            if per_step:
                lam = tl.load(lambdas_ptr + offset, valid, other=0.0)
                coefficient = (gamma * lam) * trace
            else:
                coefficient = decay * trace
        output = tl.full((tile,), 0.0, tl.float32)
        for index in tl.static_range(tile - 1, -1, -1):
            selected = tl.full((tile,), index, tl.int32)
            step_delta = tl.gather(delta, selected, axis=0)
            if explicit:
                boundary = tl.gather((term != 0.0) | (trunc != 0.0), selected, axis=0)
                running = tl.where(boundary, step_delta, step_delta + decay * running)
            else:
                step_coefficient = tl.gather(coefficient, selected, axis=0)
                running = step_delta + step_coefficient * running
            output = tl.where(lane == index, running, output)
        tl.store(advantages_ptr + offset, output, valid)
        tl.store(returns_ptr + offset, output + value, valid)


def _supported(rewards, tensors, gamma, gae_lambda):
    """Metadata-only guard: do not synchronize or detach differentiable inputs."""
    return (
        rewards.device.type == "cuda"
        and rewards.ndim == 2
        and rewards.shape[0] > 0
        and rewards.shape[1] > 0
        and isinstance(gamma, (float, int))
        and (isinstance(gae_lambda, (float, int)) or isinstance(gae_lambda, torch.Tensor))
        and all(
            tensor.device == rewards.device
            and tensor.dtype == torch.float32
            and tensor.layout == torch.strided
            and tensor.is_contiguous()
            and not (torch.is_grad_enabled() and tensor.requires_grad)
            for tensor in tensors
        )
    )


def _launch(rewards, values, terminations, truncations, next_values, tail_value, gamma, gae_lambda, explicit):
    advantages = torch.empty_like(rewards)
    returns = torch.empty_like(rewards)
    per_step = isinstance(gae_lambda, torch.Tensor)
    _gae_kernel[(rewards.shape[1],)](
        rewards,
        values,
        terminations,
        truncations,
        next_values,
        tail_value,
        # Unread unless per_step; reuse an input pointer as gae_next does for tail_ptr.
        gae_lambda if per_step else rewards,
        advantages,
        returns,
        rewards.shape[0],
        rewards.shape[1],
        gamma,
        0.0 if per_step else gamma * gae_lambda,
        explicit,
        per_step,
        32,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return advantages, returns


def gae_cuda_or_reference(
    rewards,
    values,
    terminations,
    truncations,
    truncation_bootstrap_values,
    tail_value,
    gamma,
    gae_lambda,
):
    from cleanrl.shared.ppo_loop import compute_gae

    matrices = (rewards, values, terminations, truncations, truncation_bootstrap_values)
    per_step = (gae_lambda,) if isinstance(gae_lambda, torch.Tensor) else ()
    if (
        _supported(rewards, (*matrices, *per_step, tail_value), gamma, gae_lambda)
        and all(tensor.shape == rewards.shape for tensor in (*matrices, *per_step))
        and tail_value.shape == rewards.shape[1:]
    ):
        return _launch(*matrices, tail_value, gamma, gae_lambda, False)
    return compute_gae(*matrices, tail_value, gamma, gae_lambda)


def gae_next_cuda_or_reference(
    rewards,
    values,
    terminations,
    truncations,
    next_values,
    gamma,
    gae_lambda,
):
    from cleanrl.shared.ppo_loop import compute_gae_from_next_values

    matrices = (rewards, values, terminations, truncations, next_values)
    if isinstance(gae_lambda, torch.Tensor):
        raise TypeError("explicit-next-values GAE takes a scalar lambda")
    if _supported(rewards, matrices, gamma, gae_lambda) and all(tensor.shape == rewards.shape for tensor in matrices):
        # The explicit kernel does not read tail_ptr; reuse an input pointer.
        return _launch(*matrices, next_values, gamma, gae_lambda, True)
    return compute_gae_from_next_values(*matrices, gamma, gae_lambda)

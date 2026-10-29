"""Reverse first-order linear recurrence y_t = x_t + a_t y_{t+1}, y_T = 0, over time-major (T, N).

The per-step-coefficient generalization of the GAE kernel in ``gae.py``: off-policy return
estimators (Retrace, tree-backup, V-trace) are this recurrence with importance-weighted traces
folded into ``a_t``. Same layout as ``gae.py``: one program per column, 32-step reverse-aligned
tiles, sequential FP32 recurrence (no reassociated scan, so no cumulative-product underflow).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _reverse_scan_kernel(inputs_ptr, coefficients_ptr, outputs_ptr, steps, columns, tile: tl.constexpr):
    column = tl.program_id(0)
    lane = tl.arange(0, tile)
    running = tl.full((tile,), 0.0, tl.float32)
    for block in range(tl.cdiv(steps, tile)):
        time = steps - (block + 1) * tile + lane
        valid = time >= 0
        offset = time * columns + column
        value = tl.load(inputs_ptr + offset, valid, other=0.0)
        coefficient = tl.load(coefficients_ptr + offset, valid, other=0.0)
        output = tl.full((tile,), 0.0, tl.float32)
        for index in tl.static_range(tile - 1, -1, -1):
            selected = tl.full((tile,), index, tl.int32)
            running = tl.gather(value, selected, axis=0) + tl.gather(coefficient, selected, axis=0) * running
            output = tl.where(lane == index, running, output)
        tl.store(outputs_ptr + offset, output, valid)


def reverse_linear_scan_reference(inputs, coefficients):
    outputs = torch.empty_like(inputs)
    running = torch.zeros_like(inputs[0])
    for step in range(inputs.shape[0] - 1, -1, -1):
        running = inputs[step] + coefficients[step] * running
        outputs[step] = running
    return outputs


def reverse_linear_scan(inputs, coefficients):
    """y_t = x_t + a_t y_{t+1} with y_T = 0; inputs and coefficients are (T, N)."""
    if inputs.shape != coefficients.shape or inputs.ndim != 2:
        raise ValueError("inputs and coefficients must share one (T, N) shape")
    supported = (inputs.device.type == "cuda" and coefficients.device == inputs.device
                 and inputs.dtype == coefficients.dtype == torch.float32
                 and not (torch.is_grad_enabled() and (inputs.requires_grad or coefficients.requires_grad)))
    if not supported:
        return reverse_linear_scan_reference(inputs, coefficients)
    inputs, coefficients = inputs.contiguous(), coefficients.contiguous()
    outputs = torch.empty_like(inputs)
    if inputs.numel():
        _reverse_scan_kernel[(inputs.shape[1],)](inputs, coefficients, outputs, inputs.shape[0], inputs.shape[1],
                                                 32, num_warps=1, enable_fp_fusion=False)
    return outputs

"""Hidden-matrix spectral strength with independently tuned AdamW head learning rate.

Hypotheses: three or eight Newton-Schulz steps may trade weak-direction noise
amplification against singular-direction equalization better than five steps.
The matrix_rms control normalizes Nesterov momentum without changing its spectrum;
adam_rms instead normalizes the standard Adam-preconditioned direction. All five
matrix families target the same .2 * sqrt(O * I) Frobenius norm, so neither RMS
control pays for unused Newton-Schulz work.

An independent head_lr_scale is available to every family, including AdamW: a
matrix optimizer must beat a baseline whose output head is fairly tuned, rather
than win by an accidental hidden/head learning-rate ratio. The final layer's
weight and bias updates use lr * head_lr_scale; only weight columns decay. Hidden
biases retain their original learning rate. Matrix families use ordinary AdamW
for biases and the output head; older families retain their moment/transport
rules, changing only the final layer's learning rate. Retired round_* families
are not advertised, but explicit legacy calls preserve their fresh-batch clock.

State remains linear (m/v/previous), without sample covariance or Fisher state.
Newton-Schulz Gram matrices are ephemeral algebra. Production operates in FP32
on CUDA; the same pure tensor functions support FP64 numerical oracles without
casting, host synchronization, or conditional algorithm fallbacks.
"""

import math

import torch

from cleanrl.plasticity import optimizer_proxy_model_v6 as base


_MATRIX_METHODS = ("polar", "matrix_rms", "polar_3", "polar_8", "adam_rms")
METHODS = (
    *(method for method in base.METHODS if not method.startswith("round_")),
    "polar_3",
    "polar_8",
    "adam_rms",
)


def matrix_direction(q, method):
    """Normalize [..., O, I] directions to Frobenius norm .2 * sqrt(O * I).

    Polar variants use three, five, or eight fixed polynomial Newton-Schulz
    steps on the smaller Gram side, with the v6 Muon shape/epsilon convention.
    Both RMS controls skip the polynomial entirely; adam_rms callers supply an
    already Adam-preconditioned q. Every method shares the final normalization,
    including epsilon, and maps zero input to zero in the input dtype/device.
    """
    if method not in _MATRIX_METHODS:
        raise ValueError(f"unknown matrix method: {method}")
    rows, columns = q.shape[-2:]
    direction = q
    if method in ("polar", "polar_3", "polar_8"):
        iterations = 3 if method == "polar_3" else 8 if method == "polar_8" else 5
        transposed = rows > columns
        direction = q.transpose(-2, -1) if transposed else q
        direction = direction / (direction.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(iterations):
            gram = direction @ direction.transpose(-2, -1)
            polynomial = -4.7750 * gram + 2.0315 * (gram @ gram)
            direction = 3.4445 * direction + polynomial @ direction
        if transposed:
            direction = direction.transpose(-2, -1)
    target_fro = 0.2 * math.sqrt(rows * columns)
    return direction * (target_fro / (direction.norm(dim=(-2, -1), keepdim=True) + 1e-7))


def gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, method):
    """Reuse v6 gradient math; matrix families use raw Adam gradients, no transport."""
    return base.gradients(
        weights, previous, x, target, actions, old_logprob, advantages, objective,
        "adamw" if method in _MATRIX_METHODS else method,
    )


def transition(
    weights, previous, m, v, step, grads, corrections, lr, beta1, beta2, decay, method, epochs: int, head_lr_scale
):
    """Return next weights/m/v/step without modifying inputs.

    Hyperparameters, including head_lr_scale, have shape [K, 1, 1]; step is a
    scalar device int64 tensor. The caller snapshots previous <- weights before
    installing next_weights. Matrix methods advance the update clock every pass.
    Their hidden Nesterov direction is beta1 * m_hat + (1-beta1) * g, except
    adam_rms, which uses m_hat / (sqrt(v_hat) + 1e-8) and stores full variance.
    Other matrix families update only hidden bias variance, preserving unused
    weight columns; scalar-output hidden layers retain v6's ordinary AdamW rule.

    The final layer always uses head_lr = lr * head_lr_scale, including its bias;
    hidden layers use lr unchanged. Older families delegate all state updates to
    v6, then recompute the head directly from returned moments and clock masses.
    Arithmetic order matches each original family's update when head_lr_scale
    is one; no already-rounded parameter delta is rescaled.
    """
    head_lr = lr * head_lr_scale
    if method not in _MATRIX_METHODS:
        next_weights, next_m, next_v, next_step = base.transition(
            weights, previous, m, v, step, grads, corrections, lr, beta1, beta2, decay, method, epochs
        )
        count = step // epochs + 1 if method in base.base._ROUND_METHODS else next_step
        zero_beta = beta1 == 0
        log_beta1 = torch.log(torch.where(zero_beta, torch.ones_like(beta1), beta1))
        mass = torch.where(zero_beta, torch.ones_like(beta1), -torch.expm1(log_beta1 * count))
        variance_mass = -torch.expm1(torch.log(beta2) * count)
        update = head_lr * (next_m[-1] / mass) / ((next_v[-1] / variance_mass).sqrt() + 1e-8)
        head = weights[-1]
        decayed = torch.cat((head[..., :-1] * (1 - head_lr * decay), head[..., -1:]), dim=-1)
        next_weights[-1] = decayed - update
        return next_weights, next_m, next_v, next_step

    next_step = step + 1
    zero_beta = beta1 == 0
    log_beta1 = torch.log(torch.where(zero_beta, torch.ones_like(beta1), beta1))
    mass = torch.where(zero_beta, torch.ones_like(beta1), -torch.expm1(log_beta1 * next_step))
    variance_mass = -torch.expm1(torch.log(beta2) * next_step)
    one_minus_beta1 = 1 - beta1
    one_minus_beta2 = 1 - beta2
    decay_factor = 1 - lr * decay
    head_decay_factor = 1 - head_lr * decay

    next_weights, next_m, next_v = [], [], []
    for index, (w, first, second, g) in enumerate(zip(weights, m, v, grads)):
        updated_m = beta1 * first + one_minus_beta1 * g
        corrected_m = updated_m / mass
        is_head = index == len(weights) - 1
        if is_head or w.shape[-2] == 1:
            updated_v = beta2 * second + one_minus_beta2 * g.square()
            direction = corrected_m / ((updated_v / variance_mass).sqrt() + 1e-8)
        elif method == "adam_rms":
            updated_v = beta2 * second + one_minus_beta2 * g.square()
            adam_direction = corrected_m / ((updated_v / variance_mass).sqrt() + 1e-8)
            weight_direction = matrix_direction(adam_direction[..., :-1], method)
            direction = torch.cat((weight_direction, adam_direction[..., -1:]), dim=-1)
        else:
            bias_v = beta2 * second[..., -1:] + one_minus_beta2 * g[..., -1:].square()
            updated_v = torch.cat((second[..., :-1], bias_v), dim=-1)
            q = beta1 * corrected_m[..., :-1] + one_minus_beta1 * g[..., :-1]
            weight_direction = matrix_direction(q, method)
            bias_direction = corrected_m[..., -1:] / ((bias_v / variance_mass).sqrt() + 1e-8)
            direction = torch.cat((weight_direction, bias_direction), dim=-1)
        layer_lr = head_lr if is_head else lr
        layer_decay_factor = head_decay_factor if is_head else decay_factor
        decayed = torch.cat((w[..., :-1] * layer_decay_factor, w[..., -1:]), dim=-1)
        next_weights.append(decayed - layer_lr * direction)
        next_m.append(updated_m)
        next_v.append(updated_v)
    return next_weights, next_m, next_v, next_step

"""Hidden-matrix direction geometry with bias-corrected Nesterov momentum.

Hypothesis: approximately equalizing singular directions improves on an exactly
RMS-matched momentum control; normalizing weak directions can also amplify noise.
Both target matrix RMS .2, while biases and the output head use ordinary AdamW.
This is a proxy experiment, not an exact reproduction of published Muon. State
remains linear (m/v/previous); Newton-Schulz Gram matrices are ephemeral algebraic
normalization, not sample covariance or Fisher estimates.
"""

import math

import torch

from cleanrl.plasticity import optimizer_proxy_model_v5 as base


_MATRIX_METHODS = ("polar", "matrix_rms")
METHODS = (*base.METHODS, *_MATRIX_METHODS)


def matrix_direction(q, method):
    """Normalize [..., O, I] directions to Frobenius norm .2 * sqrt(O * I).

    Both methods use the same final normalization, including epsilon; zero input
    stays zero. Polar uses five fixed polynomial Newton-Schulz steps, working on
    the smaller Gram side. Its cost is O(5 * (s*s*l + s*s*s)), s=min(O,I),
    l=max(O,I), versus O(O*I) for the RMS control. All work stays on the input
    device and in its dtype: production FP32, with FP64 available for reference
    checks. No SVD, host synchronization, or persistent matrix geometry state.
    """
    if method not in _MATRIX_METHODS:
        raise ValueError(f"unknown matrix method: {method}")
    rows, columns = q.shape[-2:]
    direction = q
    if method == "polar":
        transposed = rows > columns
        direction = q.transpose(-2, -1) if transposed else q
        direction = direction / (direction.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(5):
            gram = direction @ direction.transpose(-2, -1)
            polynomial = -4.7750 * gram + 2.0315 * (gram @ gram)
            direction = 3.4445 * direction + polynomial @ direction
        if transposed:
            direction = direction.transpose(-2, -1)
    target_fro = 0.2 * math.sqrt(rows * columns)
    return direction * (target_fro / (direction.norm(dim=(-2, -1), keepdim=True) + 1e-7))


def gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, method):
    """Use raw current gradients and zero transport for matrix methods."""
    return base.gradients(
        weights, previous, x, target, actions, old_logprob, advantages, objective,
        "adamw" if method in _MATRIX_METHODS else method,
    )


def transition(weights, previous, m, v, step, grads, corrections, lr, beta1, beta2, decay, method, epochs: int):
    """Return pure next state using the v5 API and unchanged older families.

    Matrix methods advance the update clock every pass, independently of epochs.
    Hyperparameters have shape [K, 1, 1]; step is a scalar device int64 tensor.
    Hidden direction q = beta1 * m_hat + (1-beta1) * g is normalized before the
    learning rate is applied; the stored moment is not the actual polar step.
    Hidden v updates only the bias column, preserving unused weight columns
    (initialized to zero). The final layer, and any scalar-output layer, use
    ordinary AdamW throughout. Only weight columns decay, on every update.
    The caller snapshots previous <- weights before installing next_weights.
    """
    if method not in _MATRIX_METHODS:
        return base.transition(
            weights, previous, m, v, step, grads, corrections, lr, beta1, beta2, decay, method, epochs
        )

    next_step = step + 1
    zero_beta = beta1 == 0
    log_beta1 = torch.log(torch.where(zero_beta, torch.ones_like(beta1), beta1))
    mass = torch.where(zero_beta, torch.ones_like(beta1), -torch.expm1(log_beta1 * next_step))
    variance_mass = -torch.expm1(torch.log(beta2) * next_step)
    one_minus_beta1 = 1 - beta1
    one_minus_beta2 = 1 - beta2
    decay_factor = 1 - lr * decay

    next_weights, next_m, next_v = [], [], []
    for index, (w, first, second, g) in enumerate(zip(weights, m, v, grads)):
        updated_m = beta1 * first + one_minus_beta1 * g
        corrected_m = updated_m / mass
        if index == len(weights) - 1 or w.shape[-2] == 1:
            updated_v = beta2 * second + one_minus_beta2 * g.square()
            direction = corrected_m / ((updated_v / variance_mass).sqrt() + 1e-8)
        else:
            bias_v = beta2 * second[..., -1:] + one_minus_beta2 * g[..., -1:].square()
            updated_v = torch.cat((second[..., :-1], bias_v), dim=-1)
            q = beta1 * corrected_m[..., :-1] + one_minus_beta1 * g[..., :-1]
            weight_direction = matrix_direction(q, method)
            bias_direction = corrected_m[..., -1:] / ((bias_v / variance_mass).sqrt() + 1e-8)
            direction = torch.cat((weight_direction, bias_direction), dim=-1)
        decayed = torch.cat((w[..., :-1] * decay_factor, w[..., -1:]), dim=-1)
        next_weights.append(decayed - lr * direction)
        next_m.append(updated_m)
        next_v.append(updated_v)
    return next_weights, next_m, next_v, next_step

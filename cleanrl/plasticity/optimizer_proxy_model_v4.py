"""Covariance-free, candidate-batched optimizer comparison for the v4 proxy.

Hypothesis: transporting momentum with the current Jacobian and an output-score
change captures useful gradient drift without a previous-model backward pass. Predictive
transport omits (J_current - J_previous)^T score_previous; full transport and the
MARS-style controls retain that term. Neither approximation implies PPO transfer.
The scalar-output, two-hidden-layer tanh reference architecture is unchanged;
batched VJPs never materialize per-example parameter Jacobians. Production uses
CUDA FP32, while pure dtype-preserving functions also support FP64 oracle checks.
"""

import math

import torch

from cleanrl.plasticity import network_bayes_stream_v2 as reference


METHODS = ("adamw", "predictive", "full", "mars_01", "mars_1")


def forward(weights, x):
    """Return predictions [K, B] for shared examples x [B, D]."""
    return reference.forward(weights, x)[2]


def _backward(weights, x, score, h1, h2):
    """Apply the exact VJP using cached activations and an already reduced score."""
    d3 = score.unsqueeze(-1)
    d2 = d3 * weights[2][..., :-1] * (1 - h2.square())
    d1 = (d2 @ weights[1][..., :-1]) * (1 - h1.square())
    return [
        torch.cat((d1.transpose(-1, -2) @ x, d1.sum(dim=1).unsqueeze(-1)), dim=-1),
        torch.cat((d2.transpose(-1, -2) @ h1, d2.sum(dim=1).unsqueeze(-1)), dim=-1),
        torch.cat((d3.transpose(-1, -2) @ h2, d3.sum(dim=1).unsqueeze(-1)), dim=-1),
    ]


def backward(weights, x, score):
    """Return three [K, O, I+1] VJPs; score [K, B] includes loss reduction."""
    h1, h2, _ = reference.forward(weights, x)
    return _backward(weights, x, score, h1, h2)


def _ppo_terms(prediction, actions, old_logprob, advantages):
    # Fixed scalar Gaussian std=.5, using the same log-density as Normal.log_prob.
    logprob = -(actions - prediction).square() / 0.5 - math.log(0.5) - math.log(math.sqrt(2 * math.pi))
    ratio = (logprob - old_logprob).exp()
    raw = -advantages * ratio
    clipped = -advantages * ratio.clamp(0.8, 1.2)
    return ratio, raw, clipped


def output_loss(prediction, target, actions, old_logprob, advantages, objective):
    """Per-candidate mean loss [K]; PPO rollout quantities remain frozen."""
    if objective == "regression":
        return 0.5 * (prediction - target).square().mean(dim=-1)
    if objective == "ppo":
        _, raw, clipped = _ppo_terms(prediction, actions, old_logprob, advantages)
        return torch.maximum(raw, clipped).mean(dim=-1)
    raise ValueError(f"unknown objective: {objective}")


def output_score(prediction, target, actions, old_logprob, advantages, objective):
    """Exact derivative of each candidate's mean loss with respect to its output."""
    batch_size = prediction.shape[-1]
    if objective == "regression":
        return (prediction - target) / batch_size
    if objective == "ppo":
        ratio, raw, clipped = _ppo_terms(prediction, actions, old_logprob, advantages)
        # clamp has derivative one at its endpoints; maximum splits exact ties
        # equally. Compare actual loss branches, including signed/zero advantages.
        clip_derivative = ((ratio >= 0.8) & (ratio <= 1.2)).to(prediction.dtype)
        branch_derivative = torch.where(
            raw > clipped,
            torch.ones_like(ratio),
            torch.where(raw < clipped, clip_derivative, 0.5 * (1 + clip_derivative)),
        )
        return -advantages * ratio * branch_derivative * ((actions - prediction) / 0.25) / batch_size
    raise ValueError(f"unknown objective: {objective}")


def gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, method):
    """Compute current gradients and same-batch transport before any state mutation."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    h1, h2, prediction = reference.forward(weights, x)
    score = output_score(prediction, target, actions, old_logprob, advantages, objective)
    grads = _backward(weights, x, score, h1, h2)
    if method == "adamw":
        return grads, [torch.zeros_like(g) for g in grads]

    old_h1, old_h2, old_prediction = reference.forward(previous, x)
    old_score = output_score(old_prediction, target, actions, old_logprob, advantages, objective)
    if method == "predictive":
        corrections = _backward(weights, x, score - old_score, h1, h2)
    else:
        old_grads = _backward(previous, x, old_score, old_h1, old_h2)
        corrections = [g - old_g for g, old_g in zip(grads, old_grads)]
    return grads, corrections


def transition(weights, previous, m, v, step, grads, corrections, lr, beta1, beta2, decay, method):
    """Pure AdamW/transport transition, with step counting completed updates.

    Hyperparameters have shape [K, 1, 1]; step is a scalar device int64 tensor.
    The caller snapshots previous <- weights before installing next_weights.
    beta1 lies in [0, 1), beta2 in (0, 1); bias columns never undergo decay.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    next_step = step + 1
    zero_beta = beta1 == 0
    log_beta1 = torch.log(torch.where(zero_beta, torch.ones_like(beta1), beta1))
    old_mass = torch.where(zero_beta, (step > 0).to(beta1.dtype), -torch.expm1(log_beta1 * step))
    mass = torch.where(zero_beta, torch.ones_like(beta1), -torch.expm1(log_beta1 * next_step))
    variance_mass = -torch.expm1(torch.log(beta2) * next_step)
    transport_mass = beta1 * old_mass
    one_minus_beta1 = 1 - beta1
    one_minus_beta2 = 1 - beta2
    decay_factor = 1 - lr * decay
    if method in ("mars_01", "mars_1"):
        gamma = 0.1 if method == "mars_01" else 1.0
        correction_scale = gamma * transport_mass / one_minus_beta1

    next_weights, next_m, next_v = [], [], []
    for w, first, second, g, correction in zip(weights, m, v, grads, corrections):
        if method in ("mars_01", "mars_1"):
            h = g + correction_scale * correction
            updated_m = beta1 * first + one_minus_beta1 * h
            updated_v = beta2 * second + one_minus_beta2 * h.square()
        else:
            updated_m = beta1 * first + one_minus_beta1 * g
            if method != "adamw":
                updated_m = updated_m + transport_mass * correction
            updated_v = beta2 * second + one_minus_beta2 * g.square()
        update = lr * (updated_m / mass) / ((updated_v / variance_mass).sqrt() + 1e-8)
        decayed = torch.cat((w[..., :-1] * decay_factor, w[..., -1:]), dim=-1)
        next_weights.append(decayed - update)
        next_m.append(updated_m)
        next_v.append(updated_v)
    return next_weights, next_m, next_v, next_step

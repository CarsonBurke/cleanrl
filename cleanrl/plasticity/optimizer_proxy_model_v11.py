"""SNR-shrunk momentum entering the per-layer direction: the M-SVAG factor before polar or per-layer RMS.

Key idea. RethinkD's structured-teacher diagnostic showed that optimizers with
the same in-distribution risk differ by a third off the training marginal, and
that the difference is carried by weight mass left on coordinates whose gradient
is noise (distractor inputs). Adam's per-coordinate whitening moves such
coordinates at signal scale (C3); polar moves them less; weight decay removes
them uniformly at an in-distribution cost for AdamW but not for polar. The
papers' principled version of "spend energy where the gradient is signal" is
Balles and Hennig's variance adaptation: gamma_i = mhat_i^2 / (mhat_i^2 + rho s_i)
with s_i = (vhat_i - mhat_i^2) / (1 - rho) the momentum-corrected gradient
variance and rho(beta1, t) = (1-b)(1+b^(t+1)) / ((1+b)(1-b^(t+1))) the variance
reduction of the EMA. It is scale-free, MSE-optimal shrinkage of a noisy
direction, and it was shown not to carry Adam's generalization harm.

Families. polar_svag and rms_svag apply gamma to the hidden layers' Nesterov
momentum q = beta1 mhat + (1-beta1) g before the v8 polar polynomial or the
per-layer Frobenius normalization (matrix_rms); a coordinate whose momentum is
mostly noise contributes less to the orthogonalized or normalized direction.
Hidden weights keep a full second moment for this (v8 matrix families keep only
the bias column). Head and bias updates are the v8 Adam rule. Under a
zero-variance gradient stream gamma is exactly one and each family reproduces
its parent bit-exactly on the weights (a relative dead zone of 8 ulp on the
excess variance absorbs the two moments' different rounding paths). beta1 = 0 has
rho = 1 and no variance estimate; its degenerate limit is gamma = 1, and the
plan excludes it from these families' grids.

Every v10 family passes through unchanged.
"""

import torch

from cleanrl.plasticity import optimizer_proxy_model_v8 as fast
from cleanrl.plasticity import optimizer_proxy_model_v10 as base


_SHRUNK = {"polar_svag": "polar", "rms_svag": "matrix_rms"}
METHODS = (*base.METHODS, *_SHRUNK)
TIER_KEYS = base.TIER_KEYS
SLOTS = base.SLOTS
initial_aux = base.initial_aux
consolidate = base.consolidate
is_tier = base.is_tier


def is_shrunk(method):
    return method in _SHRUNK


def gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, method):
    return base.gradients(weights, previous, x, target, actions, old_logprob, advantages, objective,
                          _SHRUNK.get(method, method))


def variance_reduction(beta1, count):
    """rho(beta1, t): variance of the bias-corrected EMA relative to one sample (Balles and Hennig eq. 21)."""
    power = torch.pow(beta1, count + 1)
    return (1 - beta1) * (1 + power) / ((1 + beta1) * (1 - power))


def shrinkage(corrected_m, corrected_v, beta1, count):
    """gamma = mhat^2 / (mhat^2 + rho s), s = max(vhat - mhat^2, 0) / (1 - rho); exactly one at zero variance."""
    finfo = torch.finfo(corrected_m.dtype)
    rho = variance_reduction(beta1, count)
    signal = corrected_m.square()
    # Relative dead zone: mhat^2 and vhat reach here by different rounding paths, so an
    # ulp-level excess must not move gamma off one under a zero-variance stream.
    excess = (corrected_v - signal - 8 * finfo.eps * signal).clamp(min=0)
    s = excess / (1 - rho).clamp(min=finfo.tiny)
    gamma = signal / (signal + rho * s + finfo.tiny)
    # beta1 = 0 has no variance estimate (rho = 1): the degenerate limit is no shrinkage.
    return torch.where(beta1 == 0, torch.ones_like(gamma), gamma)


def transition(weights, previous, m, v, aux, step, grads, corrections, hyper, method, epochs: int, block: int):
    """v10 transition for every v10 family; shrunk matrix families follow the v8 matrix update with gamma on q."""
    if method not in _SHRUNK:
        return base.transition(weights, previous, m, v, aux, step, grads, corrections, hyper, method, epochs, block)
    lr, beta1, beta2, decay = hyper["lr"], hyper["beta1"], hyper["beta2"], hyper["weight_decay"]
    head_lr = lr * hyper["head_lr_scale"]
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
        updated_v = beta2 * second + one_minus_beta2 * g.square()
        corrected_v = updated_v / variance_mass
        is_head = index == len(weights) - 1
        if is_head or w.shape[-2] == 1:
            direction = corrected_m / (corrected_v.sqrt() + 1e-8)
        else:
            gamma = shrinkage(corrected_m[..., :-1], corrected_v[..., :-1], beta1, next_step)
            q = beta1 * corrected_m[..., :-1] + one_minus_beta1 * g[..., :-1]
            weight_direction = fast.matrix_direction(gamma * q, _SHRUNK[method])
            bias_direction = corrected_m[..., -1:] / (corrected_v[..., -1:].sqrt() + 1e-8)
            direction = torch.cat((weight_direction, bias_direction), dim=-1)
        layer_lr = head_lr if is_head else lr
        layer_decay_factor = head_decay_factor if is_head else decay_factor
        decayed = torch.cat((w[..., :-1] * layer_decay_factor, w[..., -1:]), dim=-1)
        next_weights.append(decayed - layer_lr * direction)
        next_m.append(updated_m)
        next_v.append(updated_v)
    return next_weights, next_m, next_v, aux, next_step

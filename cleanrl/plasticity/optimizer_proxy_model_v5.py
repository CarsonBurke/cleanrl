"""Fresh-batch-clock momentum transport for the v5 optimizer proxy.

Hypothesis: repeated passes over one batch must not count as independent noisy
observations; only fresh batches advance EMA and variance history. Reuse transports
momentum but freezes variance, risking stale scaling as parameters move on that
batch. State stays linear in parameter count, with no covariance or oracle data;
gradients and same-batch transport reuse the v4 implementation.
"""

import torch

from cleanrl.plasticity import optimizer_proxy_model_v4 as base


_ROUND_METHODS = {
    "round_adamw": "adamw",
    "round_predictive": "predictive",
    "round_full": "full",
}
METHODS = (*base.METHODS, *_ROUND_METHODS)


def gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, method):
    """Compute unchanged v4 gradients and transport for the corresponding family."""
    return base.gradients(
        weights, previous, x, target, actions, old_logprob, advantages, objective, _ROUND_METHODS.get(method, method)
    )


def transition(weights, previous, m, v, step, grads, corrections, lr, beta1, beta2, decay, method, epochs: int):
    """Return pure next state; step counts completed parameter updates, not batches.

    epochs is a positive fixed number of passes per batch. Hyperparameters have
    shape [K, 1, 1], and step is a scalar device int64 tensor. The caller commits
    previous <- weights outside this transition, before installing next_weights.
    Round methods use count = floor(step / epochs) + 1 for bias correction;
    ordinary v4 methods retain their update clock. beta1 may be zero.
    """
    if method not in _ROUND_METHODS:
        return base.transition(weights, previous, m, v, step, grads, corrections, lr, beta1, beta2, decay, method)

    fresh = step % epochs == 0
    count = step // epochs + 1
    zero_beta = beta1 == 0
    log_beta1 = torch.log(torch.where(zero_beta, torch.ones_like(beta1), beta1))
    old_mass = torch.where(zero_beta, (count > 1).to(beta1.dtype), -torch.expm1(log_beta1 * (count - 1)))
    mass = torch.where(zero_beta, torch.ones_like(beta1), -torch.expm1(log_beta1 * count))
    variance_mass = -torch.expm1(torch.log(beta2) * count)
    if method != "round_adamw":
        transport_mass = torch.where(fresh, beta1 * old_mass, mass)
    one_minus_beta1 = 1 - beta1
    one_minus_beta2 = 1 - beta2
    decay_factor = 1 - lr * decay

    next_weights, next_m, next_v = [], [], []
    for w, first, second, g, correction in zip(weights, m, v, grads, corrections):
        updated_m = torch.where(fresh, beta1 * first + one_minus_beta1 * g, first)
        if method != "round_adamw":
            updated_m = updated_m + transport_mass * correction
        updated_v = torch.where(fresh, beta2 * second + one_minus_beta2 * g.square(), second)
        update = lr * (updated_m / mass) / ((updated_v / variance_mass).sqrt() + 1e-8)
        decayed = torch.cat((w[..., :-1] * decay_factor, w[..., -1:]), dim=-1)
        next_weights.append(decayed - update)
        next_m.append(updated_m)
        next_v.append(updated_v)
    return next_weights, next_m, next_v, step + 1

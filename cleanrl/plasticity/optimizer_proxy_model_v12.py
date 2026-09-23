"""Resistance-gated weight decay: decay everything, release each coordinate in proportion to the evidence that it pushes back.

Key idea. RethinkD section 13: in the structured regime most of the risk is weight
mass on irrelevant input columns, an oracle decay on exactly those columns lowers
in-distribution risk 26% (polar) to 36% (AdamW) and off-marginal risk 52% to 79%,
but the restoring force on an irrelevant coordinate is invisible in the gradient
stream (shrink agreement 0.50 even in a 2000-sample average). What IS visible is
resistance: under uniform decay a relevant coordinate's slow gradient average
opposes the pull most of the time and an irrelevant one's does not (agreement .20
to .27 versus .45 to .48 at decay .1). Use it or lose it: the prior is decay at
full strength, and each coordinate earns a release by resisting.

Rule. Per weight coordinate (bias columns never decay), from EMAs of the raw
gradient and its square with pole gate_beta,

    s_hat = s / (1 - beta^t),  q_hat = q / (1 - beta^t)
    z     = s_hat * sign(w) / sqrt(rho * (q_hat - s_hat^2) / (1 - rho))
    gate  = Phi(z) = (1 + erf(z / sqrt 2)) / 2
    w'    = parent step at decay 0  -  layer_lr * weight_decay * gate * w

rho(beta, n) is the n-update EMA's variance reduction (this module's `variance_reduction`), so z is
the slow mean's z-score relative to its own noise and gate is the probability that
the mean gradient does not oppose the pull. Pure-noise coordinates get gate 1/2 on
average (uniform decay at half strength, which the decay sweep covers), a
coordinate whose gradient consistently fights the decay gets gate near 0, and one
whose gradient agrees with shrinking gets gate near 1. In a dense regime every
coordinate resists at its equilibrium, so the rule is a weaker uniform decay and
cannot lose to the parent's decay sweep except through the gate's lag under drift.

Families: gate_adamw (parent adamw) and gate_polar (parent polar). At weight_decay
0 each reproduces its parent bit-exactly; under a zero-variance stream that says
shrink everywhere the gate is exactly one and the family equals its parent at the
same decay up to the order of two roundings. The EMAs live in aux slots 2 and 3
(unused by non-tier families). Every v11 family passes through unchanged.
"""

import math

import torch

from cleanrl.plasticity import optimizer_proxy_model_v11 as base


_GATED = {"gate_adamw": "adamw", "gate_polar": "polar"}
GATE_KEYS = ("gate_beta",)
MEAN_SLOT, POWER_SLOT = 2, 3
METHODS = (*base.METHODS, *_GATED)
TIER_KEYS = base.TIER_KEYS
SLOTS = base.SLOTS
initial_aux = base.initial_aux
consolidate = base.consolidate
is_tier = base.is_tier
is_shrunk = base.is_shrunk


def variance_reduction(beta, count):
    """rho(beta, n) = (1-b)(1+b^n) / ((1+b)(1-b^n)): variance of an EMA of n samples, corrected by 1 - b^n, relative to one sample.

    v11's version evaluates b^(n+1) for the same argument (one update too many); the gate corrects its EMAs by
    1 - beta^n with n the number of updates, so the noise it divides by must be the n-update reduction.
    """
    power = torch.pow(beta, count)
    return (1 - beta) * (1 + power) / ((1 + beta) * (1 - power))


def is_gated(method):
    return method in _GATED


def gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, method):
    return base.gradients(weights, previous, x, target, actions, old_logprob, advantages, objective,
                          _GATED.get(method, method))


def resistance_gate(corrected_s, corrected_q, weights, beta, count):
    """Phi(z), z = s_hat sign(w) / sqrt(rho (q_hat - s_hat^2) / (1 - rho)): exactly 1 or 0 at zero variance."""
    finfo = torch.finfo(corrected_s.dtype)
    rho = variance_reduction(beta, count)
    signal = corrected_s.square()
    excess = (corrected_q - signal - 8 * finfo.eps * signal).clamp(min=0)
    noise = (rho * excess / (1 - rho).clamp(min=finfo.tiny)).sqrt()
    z = corrected_s * torch.sign(weights) / (noise + finfo.tiny)
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


def transition(weights, previous, m, v, aux, step, grads, corrections, hyper, method, epochs: int, block: int):
    """v11 transition for every v11 family; gated families take the parent's step at decay 0 plus a gated pull."""
    if method not in _GATED:
        return base.transition(weights, previous, m, v, aux, step, grads, corrections, hyper, method, epochs, block)
    lr, decay, beta = hyper["lr"], hyper["weight_decay"], hyper["gate_beta"]
    head_lr = lr * hyper["head_lr_scale"]
    parent_hyper = {**hyper, "weight_decay": torch.zeros_like(decay)}
    next_weights, next_m, next_v, _, next_step = base.transition(
        weights, previous, m, v, aux, step, grads, corrections, parent_hyper, _GATED[method], epochs, block)
    mass = -torch.expm1(torch.log(beta) * next_step)
    one_minus_beta = 1 - beta
    gated_weights, next_aux = [], []
    for index, (w, state, g, stepped) in enumerate(zip(weights, aux, grads, next_weights)):
        s = beta * state[:, MEAN_SLOT] + one_minus_beta * g
        q = beta * state[:, POWER_SLOT] + one_minus_beta * g.square()
        gate = resistance_gate(s[..., :-1] / mass, q[..., :-1] / mass, w[..., :-1], beta, next_step)
        layer_lr = head_lr if index == len(weights) - 1 else lr
        pull = layer_lr * decay * gate * w[..., :-1]
        gated_weights.append(torch.cat((stepped[..., :-1] - pull, stepped[..., -1:]), dim=-1))
        next_aux.append(torch.stack((state[:, 0], state[:, 1], s, q, state[:, 4], state[:, 5]), dim=1))
    return gated_weights, next_m, next_v, next_aux, next_step

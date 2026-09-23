"""Prospective consolidation: a two-tier streaming optimizer with innovation-whitened gains.

Key idea. The deployed parameters phi are the learner's own prediction of its
future; a transient tier z (ordinary AdamW or polar, reset to phi every block)
is the noisy realized future. At each block boundary phi moves by a
per-parameter gain K on the block displacement delta = z - phi, and z resets to
the new phi. Under the local-level Kalman model the gain is optimal exactly when
successive innovations are uncorrelated, so K adapts multiplicatively on the
lag-1 autocorrelation rho of that parameter's own displacement sequence:
alternating displacements (jitter around a fixed point) shrink K toward kmin,
persistent displacements (drift, or a still-distant optimum) raise K toward 1.
Energy is the consolidated path length; it is spent only where displacement
evidence persists. Adam's within-block statistics never see this cross-block
structure when a whole block reuses one batch.

Hypothesis: block-level evidence about persistence lowers sustained held-out
excess risk without changing the fast tier's learning rate, and the per-parameter
dispersion of K (not its time-varying level) carries the gain. Controls: a
uniform-gain Lookahead (eta = 0) and a scalar-gain arm that adapts one K per
candidate from the mean rho, erasing spatial dispersion but keeping the profile.
Families with eta = 0 and k0 = 1 reproduce their fast tier exactly.

Optional velocity arm: phi additionally follows an EMA of its own consolidated
steps (constant-velocity prediction), and rho is measured on the innovation
relative to that prediction.

State per layer beyond the fast tier's m/v: phi, K, c (lag-1 innovation
covariance), nu (innovation power), previous innovation, velocity. All updates
are pure, candidate-batched, dtype-preserving, and free of host synchronization.
"""

import torch

from cleanrl.plasticity import optimizer_proxy_model_v8 as base


_TIER = {
    "tier_adamw": ("adamw", "param", False),
    "tier_polar": ("polar", "param", False),
    "tier_vel_adamw": ("adamw", "param", True),
    "look_adamw": ("adamw", "fixed", False),
    "scalar_adamw": ("adamw", "scalar", False),
}
METHODS = (*base.METHODS, *_TIER)
TIER_KEYS = ("tier_eta", "tier_gamma", "tier_kmin", "tier_k0", "tier_vgamma")
# phi, gain, lag-1 covariance, innovation power, previous innovation, velocity.
SLOTS = 6


def is_tier(method):
    return method in _TIER


def initial_aux(weights):
    """Per-layer [K, SLOTS, O, I+1] state: phi = weights, K = k0 is set by the caller."""
    return [torch.stack((w, torch.ones_like(w), *[torch.zeros_like(w)] * 4), dim=1) for w in weights]


def gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, method):
    """Gradients at the training point, which is the transient tier for tier families."""
    inner = _TIER[method][0] if method in _TIER else method
    return base.gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, inner)


def consolidate(z, aux, boundary, eta, gamma, kmin, vgamma, mode, velocity):
    """One masked consolidation of every layer; returns (weights, aux) lists.

    boundary is a scalar tensor in {0, 1}. At a boundary: innovation d = z - phi
    (minus the velocity prediction when enabled); c/nu EMAs advance; K <- clamp(
    K * exp(eta * rho), kmin, 1) with rho = c / nu; phi <- phi + K d + vel
    (vel = 0 without velocity); vel <- vgamma vel + (1-vgamma)(phi' - phi); z resets
    to phi'. Off a boundary every state passes through unchanged and z continues.
    mode 'fixed' skips adaptation; 'scalar' applies one K per candidate from mean rho.
    """
    keep = 1 - boundary
    innovations, powers, covariances = [], [], []
    for layer, state in zip(z, aux):
        phi, gain, c, nu, prev, vel = state.unbind(dim=1)
        d = layer - phi - (vel if velocity else 0)
        c_new = gamma * c + (1 - gamma) * d * prev
        nu_new = gamma * nu + (1 - gamma) * d.square()
        innovations.append(d)
        covariances.append(c_new)
        powers.append(nu_new)
    if mode == "scalar":
        total = sum(state.shape[-1] * state.shape[-2] for state in aux)
        mean_rho = sum((c / (nu + 1e-30)).sum(dim=(-1, -2), keepdim=True)
                       for c, nu in zip(covariances, powers)) / total
    next_weights, next_aux = [], []
    for layer, state, d, c_new, nu_new in zip(z, aux, innovations, covariances, powers):
        phi, gain, c, nu, prev, vel = state.unbind(dim=1)
        if mode == "fixed":
            gain_new = gain
        else:
            rho = mean_rho if mode == "scalar" else c_new / (nu_new + 1e-30)
            gain_new = (gain * torch.exp(eta * rho)).clamp(max=1.0)
            gain_new = torch.maximum(gain_new, kmin)
        # z - (1-K) d equals phi + K d + vel algebraically and reproduces z exactly at K = 1.
        phi_new = layer - (1 - gain_new) * d
        vel_new = vgamma * vel + (1 - vgamma) * (phi_new - phi) if velocity else vel
        blend = lambda new, old: boundary * new + keep * old  # noqa: E731
        next_weights.append(blend(phi_new, layer))
        next_aux.append(torch.stack((blend(phi_new, phi), blend(gain_new, gain), blend(c_new, c),
                                     blend(nu_new, nu), blend(d, prev), blend(vel_new, vel)), dim=1))
    return next_weights, next_aux


def transition(weights, previous, m, v, aux, step, grads, corrections, hyper, method, epochs: int, block: int):
    """Return next weights/m/v/aux/step without modifying inputs.

    hyper maps names to [K, 1, 1] tensors: lr, beta1, beta2, weight_decay,
    head_lr_scale, and for tier families tier_eta, tier_gamma, tier_kmin,
    tier_vgamma. Non-tier families pass aux through untouched. For tier families
    weights hold the transient tier during a block and equal phi at every block
    boundary, which is where the caller evaluates and prepares rollouts.
    """
    inner = _TIER[method][0] if method in _TIER else method
    fast = base.transition(weights, previous, m, v, step, grads, corrections, hyper["lr"], hyper["beta1"],
                           hyper["beta2"], hyper["weight_decay"], inner, epochs, hyper["head_lr_scale"])
    next_weights, next_m, next_v, next_step = fast
    if method not in _TIER:
        return next_weights, next_m, next_v, aux, next_step
    _, mode, velocity = _TIER[method]
    boundary = (next_step % block == 0).to(weights[0].dtype)
    next_weights, next_aux = consolidate(next_weights, aux, boundary, hyper["tier_eta"], hyper["tier_gamma"],
                                         hyper["tier_kmin"], hyper["tier_vgamma"], mode, velocity)
    return next_weights, next_m, next_v, next_aux, next_step

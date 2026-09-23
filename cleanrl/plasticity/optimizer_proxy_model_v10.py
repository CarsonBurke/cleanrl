"""Two-tier reset (Lookahead structure) on the polar fast tier, plus a smaller uniform gain.

Key idea. v9 measured the innovation-whitened per-parameter gain in all three
regimes: it never left K = 1 under the locked AdamW momentum (beta1 .995,
horizon 200 steps against 32-step blocks) or under sustained drift, and where
it did fire (bandit, beta1 .5) it raised risk. The only two-tier effect that
survived is structural: a uniform gain K < 1 on a hotter fast tier (uniform
Lookahead) lowered sustained risk 4.2% (iid_online) and 2.8% (drifting_reuse)
on AdamW, with K at the grid edge .25. v10 asks whether that structural effect
stacks with the polar fast tier, which is the frontier by a wide margin
(-21.8% / -8.4% / -6.0% versus AdamW), and extends the gain axis to .1.

Hypothesis: the two-tier reset reduces the stationary variance of the deployed
tier independent of the fast tier's preconditioning, so look_polar beats polar
at its own swept learning rate. Falsified if look_polar's selected risk is not
below polar's on the same lr axis in the regimes where look_adamw beat AdamW.

Every v9 family is passed through unchanged; look_polar reuses v9's fixed-gain
consolidation with the polar fast tier and reproduces polar exactly at K = 1.
"""

from cleanrl.plasticity import optimizer_proxy_model_v8 as fast
from cleanrl.plasticity import optimizer_proxy_model_v9 as base


_TIER = {**base._TIER, "look_polar": ("polar", "fixed", False)}
METHODS = (*base.METHODS, "look_polar")
TIER_KEYS = base.TIER_KEYS
SLOTS = base.SLOTS
initial_aux = base.initial_aux
consolidate = base.consolidate


def is_tier(method):
    return method in _TIER


def gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, method):
    inner = _TIER[method][0] if method in _TIER else method
    return fast.gradients(weights, previous, x, target, actions, old_logprob, advantages, objective, inner)


def transition(weights, previous, m, v, aux, step, grads, corrections, hyper, method, epochs: int, block: int):
    """v9 transition with the extended family table; see optimizer_proxy_model_v9.transition."""
    inner = _TIER[method][0] if method in _TIER else method
    next_weights, next_m, next_v, next_step = fast.transition(
        weights, previous, m, v, step, grads, corrections, hyper["lr"], hyper["beta1"], hyper["beta2"],
        hyper["weight_decay"], inner, epochs, hyper["head_lr_scale"])
    if method not in _TIER:
        return next_weights, next_m, next_v, aux, next_step
    _, mode, velocity = _TIER[method]
    boundary = (next_step % block == 0).to(weights[0].dtype)
    next_weights, next_aux = consolidate(next_weights, aux, boundary, hyper["tier_eta"], hyper["tier_gamma"],
                                         hyper["tier_kmin"], hyper["tier_vgamma"], mode, velocity)
    return next_weights, next_m, next_v, next_aux, next_step

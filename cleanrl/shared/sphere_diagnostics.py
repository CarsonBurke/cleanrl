"""Diagnostics and a radial-scale pin for the hypersphere trunks.

Two measured failure modes of the LeakyReluSq sphere trunk motivate this
module, and both are cheap to watch (or fix) once per PPO iteration:

1. ``LeakyReluSq`` is non-negative, so a fixed fraction of every branch
   output is an input-independent constant ("DC") direction. ``justnorm``
   only removes magnitude, never that shared direction, so the residual
   mixing spends step budget on a channel that carries no information.
   ``TrunkProbe.metrics`` reports it as ``branch_dc_frac`` / ``out_dc_frac``
   (~0.27 for LeakyReluSq at init, ~0.016 for SiTU-GLU).
2. ``lin2(f(lin1(u)))`` with a positively-homogeneous degree-2 ``f`` sits
   between two ``justnorm`` calls, which makes the loss exactly invariant to
   radially rescaling those matrices. Adam without weight decay therefore
   random-walks ``||W||`` upward along a flat direction while the effective
   functional step size decays like ``1/||W||``. ``w_norm_ratio`` watches
   the drift; ``WeightNormPin`` removes the null direction outright.

Trunk contract (satisfied by ``SiTUSphereTrunk`` and ``LReluSphereTrunk``):
``in_proj``, ``blocks`` (ModuleList), ``block_gates``/``skip_gates``
(ParameterList), ``width``, ``n_blocks``. Blocks are dispatched on attribute
presence, not class identity, so any ``lin1``/``act``/``lin2`` pair works
regardless of the activation it plugs in.
"""

import torch

from cleanrl.shared.host_actor import justnorm


def scale_invariant_weights(trunk):
    """Matrices whose radial scale the sphere geometry cancels, in order.

    ``in_proj.weight`` (its output is justnormed) plus, per block, the pair
    matrices ``lin1``/``lin2`` or the SiTU-GLU ``gate``/``up``/``down``.
    Biases are excluded: they are init-zero and are not part of the radial
    null direction the pin targets.
    """
    weights = [trunk.in_proj.weight]
    for block in trunk.blocks:
        if hasattr(block, "lin1"):
            weights.append(block.lin1.weight)
            weights.append(block.lin2.weight)
        else:
            weights.append(block.gate.weight)
            weights.append(block.up.weight)
            weights.append(block.down.weight)
    return weights


def _first_projection(block):
    """The block's stream-facing projection (pair ``lin1`` or SiTU ``gate``)."""
    return block.lin1 if hasattr(block, "lin1") else block.gate


def _dc_fraction(y):
    """Share of ``y``'s second moment carried by its mean direction."""
    return y.mean(0).square().sum() / y.square().sum(-1).mean()


class TrunkProbe:
    """Once-per-iteration health metrics for a sphere trunk.

    Everything is returned as 0-d device tensors so the caller can hand the
    dict straight to ``cleanrl.shared.ppo_loop.gather_metrics`` and pay a
    single host sync for the whole iteration.
    """

    def __init__(self, trunk):
        self.trunk = trunk
        self.weights = scale_invariant_weights(trunk)
        with torch.no_grad():
            self.init_norms = torch.stack(torch._foreach_norm(self.weights))

    @torch.no_grad()
    def metrics(self, obs):
        """Metrics for a ``(rows, in_dim)`` batch; see module docstring."""
        trunk = self.trunk
        ratios = torch.stack(torch._foreach_norm(self.weights)) / self.init_norms

        block_gate = torch.cat([g.reshape(-1) for g in trunk.block_gates]).sigmoid().mean()
        if len(trunk.skip_gates):
            skip_gate = torch.cat([g.reshape(-1) for g in trunk.skip_gates]).sigmoid().mean()
        else:
            skip_gate = torch.zeros((), device=obs.device, dtype=obs.dtype)

        block = trunk.blocks[0]
        stream = justnorm(trunk.in_proj(obs))
        preact = _first_projection(block)(stream)
        branch = justnorm(block(stream))

        out = trunk(obs)
        centered = out - out.mean(0)
        cov = centered.transpose(-2, -1) @ centered
        eigenvalues = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        erank = eigenvalues.sum().square() / eigenvalues.square().sum()

        return {
            "w_norm_ratio": ratios.mean(),
            "w_norm_ratio_max": ratios.max(),
            "block_gate": block_gate,
            "skip_gate": skip_gate,
            "preact_var": preact.var(unbiased=False),
            "branch_dc_frac": _dc_fraction(branch),
            "out_dc_frac": _dc_fraction(out),
            "out_erank": erank,
        }


class WeightNormPin:
    """Hold every scale-invariant trunk matrix at its initial Frobenius norm.

    ``lin2(f(lin1(u)))`` with a positively-homogeneous degree-2 ``f`` between
    two ``justnorm`` calls is exactly scale-invariant: scaling ``lin1`` by
    ``c`` scales the pair output by ``c^2`` and ``justnorm`` divides it back
    out, so the loss does not change at all along the radial direction. Adam
    without weight decay has no restoring force there, so ``||W||`` performs a
    random walk that drifts upward, and because the gradient of a
    scale-invariant map scales like ``1/||W||`` the functional step size decays
    as the norm grows -- a silent learning-rate collapse. Rescaling each matrix
    back to its initial norm after every optimizer step projects out exactly
    that null direction (it is a no-op on the function itself) and keeps the
    activation pinned at its design-point preactivation variance.
    """

    def __init__(self, trunk):
        self.weights = scale_invariant_weights(trunk)
        with torch.no_grad():
            self.init_norms = [n.clone() for n in torch._foreach_norm(self.weights)]

    @torch.no_grad()
    def apply(self):
        """Restore all pinned norms in place; a handful of kernels, no syncs."""
        norms = torch._foreach_norm(self.weights)
        torch._foreach_clamp_min_(norms, 1e-12)
        torch._foreach_mul_(self.weights, torch._foreach_div(self.init_norms, norms))

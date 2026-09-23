"""Muon with the Hyperball wrapper (arXiv:2606.16899, Algorithm 1 + eqs. 11-13).

Plain-EMA Muon momentum (no bias correction, no Nesterov), the Newton-Schulz
matrix sign as the direction, then Hyperball: normalize that direction, take a
step of absolute length eta * R, and project the WHOLE matrix back onto its
fixed initial Frobenius sphere. Because line 3 of Algorithm 1 divides by
||u||_F, every per-layer Muon gain cancels exactly -- s_mu = sqrt(max(1,
d_out/d_in)), Moonlight's 0.2*sqrt(max(d_in,d_out)) and the Newton-Schulz
output scale are all no-ops here, so no fan-in/fan-out rescaling is
implemented. The only surviving per-layer quantity is R = ||W_0||_F.

Weight decay is absent by construction, not by default: the radial projection
erases any purely radial shrinkage exactly, so decay would be a contradiction.
There is no second moment and no step clock; eta is a dimensionless relative
displacement (||W_{t+1} - W_t|| ~ eta * R, about eta radians of angular
motion per step), not an Adam learning rate.

Callers explicitly select hidden branch matrices; embeddings, readouts,
scalar residual gates and biases belong to ordinary Adam. Hyperball controls
matrix geometry, but does not make this PPO architecture exactly scale
invariant: fixed-gain RMS normalization and SiTU limit that interpretation.
"""

import math
from collections.abc import Callable
from typing import overload

import torch

from cleanrl.shared.gate_polar import polar_direction


def _update(params, grads, moments, radii, lr, beta1, ns_steps):
    for p, grad, m, radius in zip(params, grads, moments, radii):
        # Muon momentum, eq. (11): plain EMA, no bias correction, no Nesterov.
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        # msign(m) by Newton-Schulz; its Frobenius gain cancels below, but the
        # helper's Frobenius pre-normalization keeps the iteration in its basin.
        update = polar_direction(m, 1.0, iterations=ns_steps)
        update_norm = torch.linalg.vector_norm(update)
        # Only exact zeros are special: do not clamp small valid directions.
        safe_update_norm = torch.where(update_norm == 0, 1, update_norm)
        trial = p - lr * radius * (update / safe_update_norm)
        trial_norm = torch.linalg.vector_norm(trial)
        safe_trial_norm = torch.where(trial_norm == 0, 1, trial_norm)
        projected = radius * (trial / safe_trial_norm)
        active = (update_norm != 0) & (radius != 0) & (trial_norm != 0) & (lr != 0)
        p.copy_(torch.where(active, projected, p))


class MuonH(torch.optim.Optimizer):
    """Fixed-radius, whole-matrix Hyperball Muon without weight decay.

    ``params`` must explicitly contain only nonempty dense real matrices.
    Each group shares a device/dtype, and owns a mutable scalar device LR;
    use ``set_lr`` rather than replacing group LRs with Python floats.
    Compilation covers the tensor update of an entire group, including all
    Newton-Schulz matmuls. Changing LR does not specialize the graph;
    changing gradient presence/group shapes can require a new graph.
    ``compile=False`` is available for diagnostics.

    R is captured at construction/add_param_group, even before a gradient
    exists. A missing gradient leaves all state untouched; a present zero
    gradient still decays the momentum EMA (and may move a momentum-bearing
    matrix). An exactly zero momentum, zero LR or zero R leaves the matrix
    unchanged. R=0 describes the singleton {0}: the parameter cannot move.
    If a trial is exactly zero, its radial projection is undefined; retain
    the previous point instead of inventing a direction. State holds only
    the momentum buffer and the radius: without bias correction there is no
    step counter to restore. Nonfinite inputs propagate; no clipping or
    finite-value fallback is performed.
    """

    def __init__(self, params, lr=0.018, momentum=0.9, ns_steps=5, compile=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))
        self._update = torch.compile(_update, fullgraph=True) if compile else _update

    def add_param_group(self, param_group):
        group = dict(param_group)
        params = group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError("MuonH parameters must have deterministic ordering")
        else:
            params = list(params)
        if not params:
            raise ValueError("MuonH requires nonempty parameter groups")
        for p in params:
            if not isinstance(p, torch.Tensor):
                raise TypeError("MuonH parameters must be tensors")
            if p.ndim != 2 or p.numel() == 0 or p.layout != torch.strided or not p.is_floating_point():
                raise ValueError("MuonH requires nonempty dense real floating-point matrices")
            if p.dtype not in (torch.float32, torch.float64):
                raise ValueError("MuonH requires float32 or float64 optimizer parameters")
            if p.device != params[0].device or p.dtype != params[0].dtype:
                raise ValueError("MuonH parameters in one group must share device and dtype")
        if len({id(p) for p in params}) != len(params):
            raise ValueError("MuonH does not allow duplicate parameters")
        lr = group.get("lr", self.defaults["lr"])
        momentum = group.get("momentum", self.defaults["momentum"])
        ns_steps = group.get("ns_steps", self.defaults["ns_steps"])
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("MuonH learning rate must be finite and nonnegative")
        if not 0 <= momentum < 1:
            raise ValueError("MuonH momentum must lie in [0, 1)")
        if not isinstance(ns_steps, int) or isinstance(ns_steps, bool) or ns_steps < 1:
            raise ValueError("MuonH requires a positive integer Newton-Schulz step count")
        group["params"] = params
        group["lr"] = torch.tensor(lr, device=params[0].device, dtype=params[0].dtype)
        super().add_param_group(group)
        with torch.no_grad():
            for p in params:
                self.state[p]["radius"] = torch.linalg.vector_norm(p).detach()

    @torch.no_grad()
    def set_lr(self, value):
        """Set every group's LR in-place, preserving compiled graph inputs."""
        if not math.isfinite(value) or value < 0:
            raise ValueError("MuonH learning rate must be finite and nonnegative")
        for group in self.param_groups:
            group["lr"].fill_(value)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # The generic loader casts state tensors to the parameter dtype but
        # keeps the group LR as whatever the checkpoint's map_location gave.
        for group in self.param_groups:
            first = group["params"][0]
            group["lr"] = group["lr"].to(device=first.device, dtype=first.dtype)
            for p in group["params"]:
                if "radius" not in self.state[p]:
                    raise ValueError("MuonH checkpoint is missing its fixed initial radius")

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params, grads, moments, radii = [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.layout != torch.strided:
                    raise RuntimeError("MuonH does not support sparse gradients")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                params.append(p)
                grads.append(p.grad)
                moments.append(state["momentum_buffer"])
                radii.append(state["radius"])
            if params:
                self._update(params, grads, moments, radii,
                             group["lr"], group["momentum"], group["ns_steps"])
        return loss

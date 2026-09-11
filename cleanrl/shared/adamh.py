"""Adam with the Hyperball wrapper (arXiv:2606.16899, Algorithm 1).

Normalize the bias-corrected Adam direction, take a step of length eta * R,
then project the WHOLE matrix onto its fixed initial Frobenius sphere. This
is neither Adam followed by rescaling nor row-wise/tangential projection.
The modded-nanogpt 20260430_adamh donor supplies the Adam formulation, but
its current-norm radius is deliberately replaced by the paper's fixed R.

Callers explicitly select hidden branch matrices; embeddings, readouts,
scalar residual gates and biases belong to ordinary Adam. Hyperball controls
matrix geometry, but does not make this PPO architecture exactly scale
invariant: fixed-gain RMS normalization and SiTU limit that interpretation.
"""

import math
from collections.abc import Callable
from typing import overload

import torch


def _update(params, grads, moments, variances, steps, radii, lr, beta1, beta2, eps):
    for p, grad, m, v, step, radius in zip(params, grads, moments, variances, steps, radii):
        step.add_(1)
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        exponent = step.to(dtype=p.dtype)
        m_hat = m / (1 - beta1 ** exponent)
        v_hat = v / (1 - beta2 ** exponent)
        update = m_hat / (v_hat.sqrt() + eps)
        update_norm = torch.linalg.vector_norm(update)
        # Only exact zeros are special: do not clamp small valid directions.
        safe_update_norm = torch.where(update_norm == 0, 1, update_norm)
        trial = p - lr * radius * (update / safe_update_norm)
        trial_norm = torch.linalg.vector_norm(trial)
        safe_trial_norm = torch.where(trial_norm == 0, 1, trial_norm)
        projected = radius * (trial / safe_trial_norm)
        active = (update_norm != 0) & (radius != 0) & (trial_norm != 0) & (lr != 0)
        p.copy_(torch.where(active, projected, p))


class AdamH(torch.optim.Optimizer):
    """Fixed-radius, whole-matrix Hyperball Adam without weight decay.

    ``params`` must explicitly contain only nonempty dense real matrices.
    Each group shares a device/dtype, and owns a mutable scalar device LR;
    use ``set_lr`` rather than replacing group LRs with Python floats.
    Compilation covers the tensor update of an entire group. Changing LR
    does not specialize the graph; changing gradient presence/group shapes
    can require a new graph. ``compile=False`` is available for diagnostics.

    R is captured at construction/add_param_group, even before a gradient
    exists. A missing gradient leaves all state untouched; a present zero
    gradient advances Adam moments/time (and may move a momentum-bearing
    matrix). An exactly zero update or LR leaves the matrix unchanged.
    R=0 describes the singleton {0}: the parameter cannot move. If a trial
    is exactly zero, its radial projection is undefined; retain the previous
    point instead of inventing a direction. Moments still advance in both
    cases. Nonfinite inputs propagate, as in ordinary Adam; no clipping or
    finite-value fallback is performed.
    """

    def __init__(self, params, lr=0.018, betas=(0.9, 0.999), eps=1e-5, compile=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self._update = torch.compile(_update, fullgraph=True) if compile else _update

    def add_param_group(self, param_group):
        group = dict(param_group)
        params = group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError("AdamH parameters must have deterministic ordering")
        else:
            params = list(params)
        if not params:
            raise ValueError("AdamH requires nonempty parameter groups")
        for p in params:
            if not isinstance(p, torch.Tensor):
                raise TypeError("AdamH parameters must be tensors")
            if p.ndim != 2 or p.numel() == 0 or p.layout != torch.strided or not p.is_floating_point():
                raise ValueError("AdamH requires nonempty dense real floating-point matrices")
            if p.dtype not in (torch.float32, torch.float64):
                raise ValueError("AdamH requires float32 or float64 optimizer parameters")
            if p.device != params[0].device or p.dtype != params[0].dtype:
                raise ValueError("AdamH parameters in one group must share device and dtype")
        if len({id(p) for p in params}) != len(params):
            raise ValueError("AdamH does not allow duplicate parameters")
        lr = group.get("lr", self.defaults["lr"])
        betas = group.get("betas", self.defaults["betas"])
        eps = group.get("eps", self.defaults["eps"])
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("AdamH learning rate must be finite and nonnegative")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("AdamH betas must lie in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("AdamH epsilon must be finite and positive")
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
            raise ValueError("AdamH learning rate must be finite and nonnegative")
        for group in self.param_groups:
            group["lr"].fill_(value)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Optimizer's generic loader treats `step` specially and may leave it
        # on CPU after a map_location='cpu' checkpoint. No step may sync it.
        for group in self.param_groups:
            first = group["params"][0]
            group["lr"] = group["lr"].to(device=first.device, dtype=first.dtype)
            for p in group["params"]:
                state = self.state[p]
                if "radius" not in state:
                    raise ValueError("AdamH checkpoint is missing its fixed initial radius")
                if "step" in state:
                    state["step"] = state["step"].to(device=p.device, dtype=torch.int64)

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
            params, grads, moments, variances, steps, radii = [], [], [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.layout != torch.strided:
                    raise RuntimeError("AdamH does not support sparse gradients")
                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.zeros((), device=p.device, dtype=torch.int64)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                params.append(p)
                grads.append(p.grad)
                moments.append(state["exp_avg"])
                variances.append(state["exp_avg_sq"])
                steps.append(state["step"])
                radii.append(state["radius"])
            if params:
                beta1, beta2 = group["betas"]
                self._update(params, grads, moments, variances, steps, radii,
                             group["lr"], beta1, beta2, group["eps"])
        return loss

"""Resistance-gated polar optimizer for torch modules (plasticity proxy model v12, `gate_polar`).

Hidden weight matrices take the polar (Muon-style) direction of Nesterov momentum:
q = beta1 m_hat + (1 - beta1) g, five fixed Newton-Schulz steps on the smaller Gram
side, Frobenius norm polar_scale * sqrt(O * I). Every other parameter (biases, output
heads, scalars) takes the bias-corrected Adam direction. Weight decay is decoupled and
applied only to matrices, released per coordinate by the resistance gate

    s, q  : EMAs of the raw gradient and its square with pole gate_beta
    z     = s_hat sign(w) / sqrt(rho (q_hat - s_hat^2) / (1 - rho)),  rho = the EMA's variance reduction
    gate  = Phi(z),   w' = w - lr direction - lr weight_decay gate w

Pure-noise coordinates decay at half strength, coordinates whose slow gradient mean
resists the pull are released, coordinates whose gradient agrees with shrinking decay
at full strength. At weight_decay 0 the optimizer is exactly polar / Adam.

Cost. Two extra FP32 buffers per matrix (s, q) and one erf per coordinate per step,
comparable to Adam's second moment; the Newton-Schulz polynomial is ten small
matmuls per hidden matrix per step. Update arithmetic runs on the device without
host reads; the LR is a device scalar so annealing does not recompile.
"""

import math
from collections.abc import Callable, Iterable
from typing import overload

import torch


NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)


def polar_direction(q, scale, iterations=5):
    """[..., O, I] -> Frobenius norm scale * sqrt(O * I); zero maps to zero."""
    rows, columns = q.shape[-2:]
    transposed = rows > columns
    direction = q.transpose(-2, -1) if transposed else q
    direction = direction / (direction.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = NEWTON_SCHULZ
    for _ in range(iterations):
        gram = direction @ direction.transpose(-2, -1)
        direction = a * direction + (b * gram + c * (gram @ gram)) @ direction
    if transposed:
        direction = direction.transpose(-2, -1)
    return direction * (scale * math.sqrt(rows * columns) / (direction.norm(dim=(-2, -1), keepdim=True) + 1e-7))


def variance_reduction(beta, count):
    """Var of an n-update EMA of unit-variance noise, relative to one sample: (1-b)(1+b^n)/((1+b)(1-b^n))."""
    power = torch.pow(beta, count)
    return (1 - beta) * (1 + power) / ((1 + beta) * (1 - power))


def resistance_gate(corrected_s, corrected_q, weights, beta, count):
    """Phi(z): probability that the slow gradient mean does not oppose shrinking the weight."""
    finfo = torch.finfo(corrected_s.dtype)
    rho = variance_reduction(beta, count)
    signal = corrected_s.square()
    excess = (corrected_q - signal - 8 * finfo.eps * signal).clamp(min=0)
    noise = (rho * excess / (1 - rho).clamp(min=finfo.tiny)).sqrt()
    z = corrected_s * torch.sign(weights) / (noise + finfo.tiny)
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


def _update(params, grads, moments, variances, means, powers, step, lr, beta1, beta2, eps,
            weight_decay, gate_beta, polar, polar_scale):
    """One group's update after `step` has been advanced; every tensor is written in place."""
    count = step.to(dtype=params[0].dtype)
    momentum_mass = 1 - torch.pow(beta1, count)
    variance_mass = 1 - torch.pow(beta2, count)
    gate_mass = 1 - torch.pow(gate_beta, count)
    for p, g, m, v, s, q in zip(params, grads, moments, variances, means, powers):
        m.mul_(beta1).add_(g, alpha=1 - beta1)
        m_hat = m / momentum_mass
        if polar:
            direction = polar_direction(beta1 * m_hat + (1 - beta1) * g, polar_scale)
        else:
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            direction = m_hat / ((v / variance_mass).sqrt() + eps)
        stepped = p - lr * direction
        if s is not None:
            s.mul_(gate_beta).add_(g, alpha=1 - gate_beta)
            q.mul_(gate_beta).addcmul_(g, g, value=1 - gate_beta)
            gate = resistance_gate(s / gate_mass, q / gate_mass, p, gate_beta, count)
            stepped = stepped - (lr * weight_decay) * gate * p
        p.copy_(stepped)


class GatePolar(torch.optim.Optimizer):
    """Polar hidden matrices, Adam elsewhere, resistance-gated decoupled decay on matrices.

    Every group carries ``polar`` (matrix groups only: 2-D dense floating tensors)
    and ``lr_scale``; the group's LR is a device scalar ``base_lr * lr_scale`` that
    ``set_lr`` rewrites in place so compiled graphs keep their inputs. ``weight_decay``
    and ``gate_beta`` are per-group Python constants; decay touches only tensors
    with two or more dimensions. A missing gradient leaves that parameter's state
    untouched; the group clock advances whenever any parameter has a gradient.
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-5, weight_decay=0.0, gate_beta=0.9999,
                 polar=False, polar_scale=0.2, compile=True):
        defaults = dict(lr=lr, lr_scale=1.0, betas=betas, eps=eps, weight_decay=weight_decay,
                        gate_beta=gate_beta, polar=polar, polar_scale=polar_scale)
        self.base_lr = lr
        super().__init__(params, defaults)
        self._update = torch.compile(_update, fullgraph=True) if compile else _update

    def add_param_group(self, param_group):
        group = dict(param_group)
        params = group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError("GatePolar parameters must have deterministic ordering")
        else:
            params = list(params)
        if not params:
            raise ValueError("GatePolar requires nonempty parameter groups")
        for key, value in self.defaults.items():
            group.setdefault(key, value)
        polar = bool(group["polar"])
        for p in params:
            if not isinstance(p, torch.Tensor) or p.layout != torch.strided or not p.is_floating_point():
                raise TypeError("GatePolar parameters must be dense floating-point tensors")
            if p.numel() == 0 or p.dtype not in (torch.float32, torch.float64):
                raise ValueError("GatePolar requires nonempty float32 or float64 parameters")
            if polar and p.ndim != 2:
                raise ValueError("polar groups hold matrices only; route vectors and scalars to Adam groups")
            if p.device != params[0].device or p.dtype != params[0].dtype:
                raise ValueError("GatePolar parameters in one group must share device and dtype")
        if len({id(p) for p in params}) != len(params):
            raise ValueError("GatePolar does not allow duplicate parameters")
        lr, scale = group["lr"], group["lr_scale"]
        if not (math.isfinite(lr) and lr >= 0 and math.isfinite(scale) and scale >= 0):
            raise ValueError("GatePolar learning rate and lr_scale must be finite and nonnegative")
        betas = group["betas"]
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("GatePolar betas must lie in [0, 1)")
        if not (0 < group["gate_beta"] < 1):
            raise ValueError("GatePolar gate_beta must lie in (0, 1)")
        dtype = params[0].dtype
        if any(torch.tensor(beta, dtype=dtype) >= 1 for beta in (*betas, group["gate_beta"])):
            raise ValueError(f"betas and gate_beta must stay below one in {dtype}; slower poles need float64")
        if not math.isfinite(group["eps"]) or group["eps"] <= 0:
            raise ValueError("GatePolar epsilon must be finite and positive")
        if not math.isfinite(group["weight_decay"]) or group["weight_decay"] < 0:
            raise ValueError("GatePolar weight_decay must be finite and nonnegative")
        if not math.isfinite(group["polar_scale"]) or group["polar_scale"] <= 0:
            raise ValueError("GatePolar polar_scale must be finite and positive")
        group["params"] = params
        group["polar"] = polar
        group["lr"] = torch.tensor(lr * scale, device=params[0].device, dtype=params[0].dtype)
        group["step"] = torch.zeros((), device=params[0].device, dtype=torch.int64)
        super().add_param_group(group)

    @torch.no_grad()
    def set_lr(self, value):
        """Set the base LR; every group receives value * lr_scale in place."""
        if not math.isfinite(value) or value < 0:
            raise ValueError("GatePolar learning rate must be finite and nonnegative")
        self.base_lr = value
        for group in self.param_groups:
            group["lr"].fill_(value * group["lr_scale"])

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            first = group["params"][0]
            group["lr"] = torch.as_tensor(group["lr"], device=first.device, dtype=first.dtype)
            group["step"] = torch.as_tensor(group["step"], device=first.device, dtype=torch.int64)

    def _state(self, p, group):
        state = self.state[p]
        if not state:
            state["m"] = torch.zeros_like(p)
            if not group["polar"]:
                state["v"] = torch.zeros_like(p)
        if group["weight_decay"] > 0 and p.ndim >= 2 and "s" not in state:
            # Allocated on the first decayed step, so a decay raised from zero later still gets its EMAs.
            state["s"] = torch.zeros_like(p)
            state["q"] = torch.zeros_like(p)
        return state

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
            params, grads, moments, variances, means, powers = [], [], [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._state(p, group)
                params.append(p)
                grads.append(p.grad)
                moments.append(state["m"])
                variances.append(state.get("v"))
                means.append(state.get("s"))
                powers.append(state.get("q"))
            if not params:
                continue
            group["step"].add_(1)
            beta1, beta2 = group["betas"]
            self._update(params, grads, moments, variances, means, powers, group["step"], group["lr"],
                         beta1, beta2, group["eps"], group["weight_decay"], group["gate_beta"],
                         group["polar"], group["polar_scale"])
        return loss


def mlp_groups(*mlps: torch.nn.Sequential, head_lr_scale: float = 1.0) -> list[dict]:
    """Param groups for Linear stacks: hidden weights polar, hidden biases Adam, last Linear Adam at lr * head_lr_scale."""
    hidden_weights, hidden_biases, heads = [], [], []
    for mlp in mlps:
        linears = [layer for layer in mlp if isinstance(layer, torch.nn.Linear)]
        if not linears:
            raise ValueError("mlp_groups expects at least one Linear layer per stack")
        for layer in linears[:-1]:
            hidden_weights.append(layer.weight)
            if layer.bias is not None:
                hidden_biases.append(layer.bias)
        heads.extend(p for p in linears[-1].parameters())
    groups = [{"params": hidden_weights, "polar": True}]
    if hidden_biases:
        groups.append({"params": hidden_biases, "polar": False})
    groups.append({"params": heads, "polar": False, "lr_scale": head_lr_scale})
    return groups


def parameter_count(groups: Iterable[dict]) -> int:
    return sum(p.numel() for group in groups for p in group["params"])

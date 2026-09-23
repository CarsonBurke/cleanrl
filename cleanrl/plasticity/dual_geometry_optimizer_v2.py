"""Dual Geometry v2: isolate the metric, holding dual accumulation fixed.

The v1 stock experiment favored lifetime history, not faster forgetting; its
innovation denominator was indistinguishable from raw gradient RMS. Here all
controls solve the same cumulative linearized objective with a different diagonal
metric. A rowwise shared gradient scale preserves scalar loss-scale invariance;
coordinate power redistributes learning without changing that global unit.
No posterior, covariance matrix, support inference, betting ledger or ensembles.
This is a falsifiable geometry experiment, not a claimed novel optimizer theorem.
"""

import math
from typing import Any

import torch
from torch.optim import Optimizer


POWERS = {"shared": 0.0, "rms": 0.5, "variance": 1.0, "strong": 1.5}


def _step(weight, grad, *, anchor, rate, gradient_sum, square_sum, control):
    gradient_sum.add_(grad)
    square_sum.addcmul_(grad, grad)
    shared = square_sum.mean(dim=-1, keepdim=True) if square_sum.ndim else square_sum
    tiny = torch.finfo(weight.dtype).tiny
    root = shared.sqrt()
    relative = square_sum / shared.clamp_min(tiny)
    denominator = root * relative.pow(POWERS[control])
    displacement = gradient_sum / denominator.clamp_min(tiny)
    weight.copy_(anchor - rate * displacement)


class DualGeometryState:
    """CUDA FP32 gradient-only block-diagonal dual averaging.

    Each last-axis parameter row is a scale block. With Q=sum g² and shared
    q=mean(Q), D_i=sqrt(q)*(Q_i/q)^power. power=.5 recovers ordinary coordinate
    AdaGrad dual averaging exactly in real arithmetic. Positive powers can amplify
    rarely observed coordinates; cold-start and pure-noise controls are essential.
    The immutable anchor and rate are not warmup buffers; both history tensors are.
    """

    def __init__(self, weight, *, lr: float | torch.Tensor = 0.01, control="variance"):
        if weight.device.type != "cuda" or weight.dtype != torch.float32:
            raise ValueError("dual geometry requires CUDA float32 parameters")
        if control not in POWERS:
            raise ValueError(f"unknown control: {control}")
        self.rate = torch.as_tensor(lr, dtype=weight.dtype, device=weight.device).detach().clone()
        if torch.broadcast_shapes(self.rate.shape, weight.shape) != weight.shape:
            raise ValueError("lr must broadcast to the parameter shape")
        if not bool(torch.isfinite(self.rate).all()) or bool((self.rate <= 0).any()):
            raise ValueError("lr must be finite and positive")
        self.anchor = weight.detach().clone()
        self.gradient_sum = torch.zeros_like(weight)
        self.square_sum = torch.zeros_like(weight)
        self.control = control

    def buffers(self):
        return (self.gradient_sum, self.square_sum)

    @torch.no_grad()
    def step(self, weight, grad):
        _step(weight, grad, anchor=self.anchor, rate=self.rate,
              gradient_sum=self.gradient_sum, square_sum=self.square_sum, control=self.control)


class DualGeometry(Optimizer):
    """Torch gradient-only interface; parameter last-axis rows define metric blocks."""

    def __init__(self, params, *, lr=0.01, control="variance"):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        super().__init__(params, dict(lr=lr, control=control))
        for group in self.param_groups:
            for parameter in group["params"]:
                self._initialize(parameter, group)

    def _initialize(self, parameter, group):
        state = DualGeometryState(parameter, lr=group["lr"], control=group["control"])
        self.state[parameter].update({"anchor": state.anchor, "gradient_sum": state.gradient_sum,
                                      "square_sum": state.square_sum})

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.layout != torch.strided:
                    raise ValueError("sparse gradient layout is unsupported")
                if not self.state[parameter]:
                    self._initialize(parameter, group)
                _step(parameter, parameter.grad, **self.state[parameter],
                      rate=group["lr"], control=group["control"])
        return loss

"""Dual Tracking v1: re-solve accumulated linearized losses, do not bank steps.

A diagonal AdaGrad dual-average can erase a past displacement when later gradients
contradict it. Discounting provides a controlled history horizon for drift. The
mirror control takes the same local steps without this re-solving, while innovation
normalization charges predictable gradients less. These are optimization controls,
not posterior estimates, confidence measures, or claims of a novel regret theorem.
Hypothesis: weak distributed stock signal needs reversible learning, not an evidence
threshold or a lifetime maximum-gradient/earned-capital bottleneck.
"""

import math
from typing import Any

import torch
from torch.optim import Optimizer


def _setting(value, weight, *, name, positive):
    result = torch.as_tensor(value, dtype=weight.dtype, device=weight.device).detach().clone()
    if torch.broadcast_shapes(result.shape, weight.shape) != weight.shape:
        raise ValueError(f"{name} must broadcast to the parameter shape")
    bad = (result <= 0) if positive else (result < 0)
    if not bool(torch.isfinite(result).all()) or bool(bad.any()):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _step(weight, grad, *, anchor, rate, decay, gradient_sum, square_sum, mass, control):
    tiny = torch.finfo(weight.dtype).tiny
    if control == "innovation":
        surprise = grad - gradient_sum / mass.clamp_min(tiny)
    else:
        surprise = grad
    gradient_sum.mul_(decay).add_(grad)
    square_sum.mul_(decay).addcmul_(surprise, surprise)
    mass.mul_(decay).add_(1)
    denominator = square_sum.sqrt().clamp_min(tiny)
    if control == "mirror":
        weight.add_(-rate * grad / denominator)
    else:
        weight.copy_(anchor - rate * gradient_sum / denominator)


class DualTrackingState:
    """CUDA FP32 gradient-only state for compiled/captured streaming updates.

    memory=0 uses lifetime accumulators; positive memory H discounts by exp(-1/H)
    every observation, including zero gradients. The anchor never changes.
    All controls use the current gradient only AFTER its prediction was scored.
    lr is a parameter-scale hyperparameter; no gradient-scale epsilon is added.
    Numerical guards affect subnormal-scale arithmetic. Innovation Q can decline
    on a constant gradient and cause large steps: that is a measured risk, not a
    reason to silently clamp updates or fall back to another optimizer.
    """

    def __init__(self, weight, *, lr: float | torch.Tensor = 0.01,
                 memory: float | torch.Tensor = 0.0, control="dual"):
        if weight.device.type != "cuda" or weight.dtype != torch.float32:
            raise ValueError("dual tracking requires CUDA float32 parameters")
        if control not in {"dual", "mirror", "innovation"}:
            raise ValueError(f"unknown control: {control}")
        self.anchor = weight.detach().clone()
        self.rate = _setting(lr, weight, name="lr", positive=True)
        horizon = _setting(memory, weight, name="memory", positive=False)
        self.decay = torch.where(horizon > 0, (-1 / horizon.clamp_min(1e-30)).exp(), 1.0)
        self.gradient_sum = torch.zeros_like(weight)
        self.square_sum = torch.zeros_like(weight)
        self.mass = torch.zeros_like(weight)
        self.control = control

    def buffers(self):
        return (self.gradient_sum, self.square_sum, self.mass)

    @torch.no_grad()
    def step(self, weight, grad):
        _step(weight, grad, anchor=self.anchor, rate=self.rate, decay=self.decay,
              gradient_sum=self.gradient_sum, square_sum=self.square_sum,
              mass=self.mass, control=self.control)


class DualTracking(Optimizer):
    """Torch interface preserving the anchor and all history on checkpoint resume."""

    def __init__(self, params, *, lr=0.01, memory=0.0, control="dual"):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        super().__init__(params, dict(lr=lr, memory=memory, control=control))
        for group in self.param_groups:
            for parameter in group["params"]:
                self._initialize(parameter, group)

    def _initialize(self, parameter, group):
        state = DualTrackingState(parameter, lr=group["lr"], memory=group["memory"],
                                  control=group["control"])
        self.state[parameter].update({key: value for key, value in vars(state).items()
                                      if isinstance(value, torch.Tensor)})

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
                state = self.state[parameter]
                _step(parameter, parameter.grad, **{key: value for key, value in state.items()
                                                    if key != "rate"},
                      rate=group["lr"], control=group["control"])
        return loss

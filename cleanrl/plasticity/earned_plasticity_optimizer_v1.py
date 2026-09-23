"""Earned Plasticity v1: finance changes with their later gradient utility.

Coin-betting-inspired, not a posterior or a claimed new regret theorem. Previous
positions earn -g*(w-anchor); that capital funds subsequent positions. Drawdowns
shorten gradient history instead of waiting for a hyper-learning-rate controller.
Hypothesis: coherent gradients compound useful changes, while losing positions
lose both capital and stale directional history. Noise rejection and reopening
are empirical questions; this does not establish neural-training superiority.
"""

from typing import Any

import torch
from torch.optim import Optimizer


def _stake(value, weight):
    stake = torch.as_tensor(value, dtype=weight.dtype, device=weight.device).detach().clone()
    if torch.broadcast_shapes(stake.shape, weight.shape) != weight.shape:
        raise ValueError("initial_stake must broadcast to the parameter shape")
    if not bool(torch.isfinite(stake).all()) or bool((stake <= 0).any()):
        raise ValueError("initial_stake must be finite and positive")
    return stake


def _step(weight, grad, *, anchor, initial_stake, gradient_bound,
          gradient_sum, absolute_sum, wealth, control):
    tiny = torch.finfo(weight.dtype).tiny
    displacement = weight - anchor
    credited_position = (displacement.roll(1, dims=-1)
                         if control == "shuffled" and displacement.ndim else displacement)
    cost = grad * credited_position
    magnitude = grad.abs()
    next_bound = torch.maximum(gradient_bound, magnitude)
    capital = wealth + initial_stake * next_bound
    if control == "persistent":
        retention = 1.0
    else:
        retention = torch.where(
            capital > 0,
            capital / (capital + cost.clamp_min(0)).clamp_min(tiny),
            1.0,
        )
    gradient_sum.mul_(retention).add_(grad)
    absolute_sum.mul_(retention).add_(magnitude)
    if control != "fixed":
        wealth.sub_(cost).clamp_min_(0)
    else:
        wealth.zero_()
    gradient_bound.copy_(next_bound)
    fraction = gradient_sum / (absolute_sum + gradient_bound).clamp_min(tiny)
    position_budget = wealth / gradient_bound.clamp_min(tiny) + initial_stake
    weight.copy_(torch.where(magnitude > 0, anchor - fraction * position_budget, weight))


class EarnedPlasticityState:
    """Gradient-only O(parameters) state, suitable for torch.compile/CUDA graphs.

    initial_stake is a risk scale in parameter units, not a learning rate or a
    probability. It remains a hyperparameter. The lifetime maximum |gradient|
    defines the capital units; a large outlier can therefore slow future bets.
    The stake continuously subsidizes exploration and the ledger is floored at
    zero: positive wealth is NOT calibrated evidence or a self-financing guarantee.
    Outliers can exceed prior budgets. Numerical guards alter subnormal behavior.

    funded: earned capital plus drawdown-dependent history retention.
    persistent: earned capital, never forget signed/absolute gradient history.
    fixed: fixed initial capital, retaining the drawdown history rule.
    shuffled: deliberately assign previous-position credit to the wrong feature.

    No labels, features, curvature, clean targets or additional model copies.
    The four mutable buffers and the parameter must be restored after warmup.
    """

    def __init__(self, weight, *, initial_stake=1e-4, control="funded"):
        if weight.dtype != torch.float32 or weight.device.type != "cuda":
            raise ValueError("earned plasticity requires CUDA float32 parameters")
        if control not in {"funded", "persistent", "fixed", "shuffled"}:
            raise ValueError(f"unknown control: {control}")
        self.anchor = weight.detach().clone()
        self.initial_stake = _stake(initial_stake, weight)
        self.gradient_bound = torch.zeros_like(weight)
        self.gradient_sum = torch.zeros_like(weight)
        self.absolute_sum = torch.zeros_like(weight)
        self.wealth = torch.zeros_like(weight)
        self.control = control

    def buffers(self):
        return (self.gradient_bound, self.gradient_sum, self.absolute_sum, self.wealth)

    @torch.no_grad()
    def step(self, weight, grad):
        _step(weight, grad, anchor=self.anchor, initial_stake=self.initial_stake,
              gradient_bound=self.gradient_bound, gradient_sum=self.gradient_sum,
              absolute_sum=self.absolute_sum, wealth=self.wealth, control=self.control)


class EarnedPlasticity(Optimizer):
    """Standard gradient-only Torch optimizer interface; CUDA float32 parameters.

    state_dict includes the anchor, stake and all capital/history state. Gradients
    score the preupdate position. The current action cannot earn its own capital.
    Per-coordinate utility is a gradient attribution, not an exact counterfactual
    loss improvement in nonlinear or interacting models.
    """

    def __init__(self, params, *, initial_stake=1e-4, control="funded"):
        super().__init__(params, dict(initial_stake=initial_stake, control=control))
        for group in self.param_groups:
            for parameter in group["params"]:
                self._initialize(parameter, group)

    def _initialize(self, parameter, group):
        state = EarnedPlasticityState(parameter, initial_stake=group["initial_stake"],
                                     control=group["control"])
        self.state[parameter].update({key: value for key, value in vars(state).items()
                                      if isinstance(value, torch.Tensor)})

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        """Return the closure's result unchanged, or None without a closure."""
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
                      control=group["control"])
        return loss

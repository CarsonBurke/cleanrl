"""Predictive Readout v1: learn what to write, not merely how hard to write.

A coordinate memory estimates its residual target after removing other live
coordinates. A second, prequential regression learns how much OLD memory predicts
later observations. Live parameters follow that validated readout and can retract
rejected memory; closing a gate does not strand an old noisy parameter value.
All candidate writes have unit scale. No LR sweep, posterior, covariance matrix,
replay, or ensemble of models. This is a linear-stream mechanism experiment.
"""

import torch


CONTROLS = ("validated", "open", "shuffled")


class PredictiveReadoutState:
    """Half-squared, single-example linear-loss CUDA prototype, not a general NN API.

    grad=residual*x, curvature=x*x. A coordinate's partial target is h*w-g.
    Old memory z is scored before updating z: A+=z*(h*w-g), B+=h*z*z.
    The readout solves min_{a in [0,1]} .5*B*a*a-A*a. This bounded allocation is
    the model definition, not a numerical patch. B=0 means no validated write.
    The memory then becomes sum(h*w-g)/sum(h), a coordinate residual regression.
    These regressions interact through live residuals; no convergence claim is made.

    A unit candidate can have cross-coordinate interference. Exact current-example
    quadratic line minimization selects a fraction in [0,1], avoiding overshoot
    without a manually selected LR. sign(g)*sqrt(h) reconstructs x up to a common
    sign, sufficient for rank-one curvature. It is NOT valid for arbitrary neural
    diagonal Hessians. If residual is zero, the safe line fraction is zero.
    """

    def __init__(self, weight, *, control="validated"):
        if weight.device.type != "cuda" or weight.dtype != torch.float32:
            raise ValueError("predictive readout requires CUDA float32 parameters")
        if control not in CONTROLS:
            raise ValueError(f"unknown control: {control}")
        self.anchor = weight.detach().clone()
        self.memory = torch.zeros_like(weight)
        self.target_sum = torch.zeros_like(weight)
        self.curvature_sum = torch.zeros_like(weight)
        self.validation_target = torch.zeros_like(weight)
        self.validation_curvature = torch.zeros_like(weight)
        self.allocation = torch.zeros_like(weight)
        self.control = control

    def buffers(self):
        return (self.memory, self.target_sum, self.curvature_sum,
                self.validation_target, self.validation_curvature, self.allocation)

    @torch.no_grad()
    def step(self, weight, grad, curvature):
        tiny = torch.finfo(weight.dtype).tiny
        displacement = weight - self.anchor
        partial = curvature * displacement - grad
        predictor = self.memory.roll(1, dims=-1) if self.control == "shuffled" else self.memory
        self.validation_target.add_(predictor * partial)
        self.validation_curvature.add_(curvature * predictor.square())
        if self.control == "open":
            self.allocation.fill_(1)
        else:
            self.allocation.copy_((self.validation_target /
                                   self.validation_curvature.clamp_min(tiny)).clamp(0, 1))
        self.target_sum.add_(partial)
        self.curvature_sum.add_(curvature)
        self.memory.copy_(self.target_sum / self.curvature_sum.clamp_min(tiny))
        candidate = self.allocation * self.memory
        direction = candidate - displacement
        slope = (grad * direction).sum(-1, keepdim=True)
        signed_features = grad.sign() * curvature.sqrt()
        line_curvature = (signed_features * direction).sum(-1, keepdim=True).square()
        fraction = (-slope / line_curvature.clamp_min(tiny)).clamp(0, 1)
        # If the current example cannot observe this direction, permit the
        # independently validated readout; a zero residual alone does not grant it.
        invisible = ((curvature * direction.square()).sum(-1, keepdim=True) == 0)
        fraction = torch.where(invisible, 1.0, fraction)
        weight.add_(fraction * direction)

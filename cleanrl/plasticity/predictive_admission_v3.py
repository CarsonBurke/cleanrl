"""Predictive Admission v3: certify old memory before making a unit write.

The v2 readout fitted signal but admitted too many noise coordinates. This version
separates admission from amplitude: a self-normalized prequential score must support
a memory before its unshrunk fitted readout is eligible to write. Coordinate scores
handle sparse signal; one row-wide score can admit distributed weak structure.
There is one memory and one live model, not an ensemble. No learning-rate sweep.
The confidence boundary is a diagnostic heuristic, not a proven anytime p-value.
"""

import math

import torch


CONTROLS = ("hierarchical", "coordinate", "block", "unvalidated", "shuffled")


class PredictiveAdmissionState:
    """CUDA linear half-square prototype; grad=residual*x and curvature=x*x.

    Old coordinate memory z is a forecast. Its later partial-target product adds
    to A, its square to V, and its design curvature to B. Admit when
    A>sqrt(2*V*log(2*(dimension+1))); fit allocation A/B on[0,1]. A separate
    row-wide forecast uses the full rank-one quadratic, including cross terms.
    A certified coordinate overrides the row-wide allocation. Only admitted or
    previously written coordinates participate in the sample-Jacobian write.

    This is confidence-gated predictive optimization, not posterior inference.
    Repeated testing, serial dependence and changing forecasters preclude treating
    the boundary as an exact false-discovery guarantee. All controls test it on
    matched pure-noise streams. Lifetime evidence may delay changes of regime.
    """

    def __init__(self, weight, *, control="hierarchical"):
        if weight.device.type != "cuda" or weight.dtype != torch.float32:
            raise ValueError("predictive admission requires CUDA float32 parameters")
        if control not in CONTROLS:
            raise ValueError(f"unknown control: {control}")
        self.anchor = weight.detach().clone()
        self.memory = torch.zeros_like(weight)
        self.target_sum = torch.zeros_like(weight)
        self.curvature_sum = torch.zeros_like(weight)
        self.validation_target = torch.zeros_like(weight)
        self.validation_curvature = torch.zeros_like(weight)
        self.validation_square = torch.zeros_like(weight)
        self.allocation = torch.zeros_like(weight)
        shape = weight.shape[:-1] + (1,)
        self.block_target = torch.zeros(shape, device=weight.device)
        self.block_curvature = torch.zeros_like(self.block_target)
        self.block_square = torch.zeros_like(self.block_target)
        self.log_tests = math.log(2 * (weight.shape[-1] + 1))
        self.control = control

    def buffers(self):
        return (self.memory, self.target_sum, self.curvature_sum, self.validation_target,
                self.validation_curvature, self.validation_square, self.allocation,
                self.block_target, self.block_curvature, self.block_square)

    @torch.no_grad()
    def step(self, weight, grad, curvature):
        tiny = torch.finfo(weight.dtype).tiny
        displacement = weight - self.anchor
        partial = curvature * displacement - grad
        predictor = self.memory.roll(1, dims=-1) if self.control == "shuffled" else self.memory
        product = predictor * partial
        self.validation_target.add_(product)
        self.validation_curvature.add_(curvature * predictor.square())
        self.validation_square.addcmul_(product, product)
        signed_features = grad.sign() * curvature.sqrt()
        forecast = (signed_features * predictor).sum(-1, keepdim=True)
        block_product = (forecast * (signed_features * displacement).sum(-1, keepdim=True)
                         - (grad * predictor).sum(-1, keepdim=True))
        self.block_target.add_(block_product)
        self.block_curvature.addcmul_(forecast, forecast)
        self.block_square.addcmul_(block_product, block_product)
        coordinate_fit = (self.validation_target / self.validation_curvature.clamp_min(tiny)).clamp(0, 1)
        coordinate_admitted = self.validation_target > (2 * self.log_tests * self.validation_square).sqrt()
        block_fit = (self.block_target / self.block_curvature.clamp_min(tiny)).clamp(0, 1)
        block_admitted = self.block_target > (2 * self.log_tests * self.block_square).sqrt()
        block_readout = torch.where(block_admitted, block_fit, 0.0)
        if self.control == "unvalidated":
            self.allocation.copy_(coordinate_fit)
        elif self.control == "coordinate":
            self.allocation.copy_(torch.where(coordinate_admitted, coordinate_fit, 0.0))
        elif self.control == "block":
            self.allocation.copy_(block_readout.expand_as(weight))
        else:
            self.allocation.copy_(torch.where(coordinate_admitted, coordinate_fit, block_readout))
        self.target_sum.add_(partial)
        self.curvature_sum.add_(curvature)
        self.memory.copy_(self.target_sum / self.curvature_sum.clamp_min(tiny))
        desired = self.allocation * self.memory - displacement
        eligible = (self.allocation > 0) | (displacement != 0)
        writer = signed_features * eligible
        output_shift = (writer * desired).sum(-1, keepdim=True)
        feature_norm = (curvature * eligible).sum(-1, keepdim=True)
        direction = writer * (output_shift / feature_norm.clamp_min(tiny))
        slope = (grad * direction).sum(-1, keepdim=True)
        fraction = (-slope / output_shift.square().clamp_min(tiny)).clamp(0, 1)
        weight.add_(fraction * direction)

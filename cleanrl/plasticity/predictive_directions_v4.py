"""Predictive Directions v4: validate direction separately from write amplitude.

v3's stock row-wide score was polluted by early, wildly scaled forecasts (final
fitted amplitude about .004). Normalize the slow memory to a direction before
validation; fit its amplitude from later outcomes. Fast coordinate forecasts use
signs and a 128-participation horizon, so old noise does not own plasticity forever.
Certified directions permit full fitted writes. No global learning-rate sweep,
posterior, covariance matrix, replay or multiple trained models.
"""

import math

import torch


CONTROLS = ("hierarchical", "coordinate", "block", "unvalidated", "shuffled")


class PredictiveDirectionsState:
    """Linear single-example half-square CUDA prototype, not a general NN optimizer.

    Two optimizer memories share one live model: fast coordinate directions and a
    slow row-wide direction. Each forecast is scored on the next observation,
    before its memory changes. Coordinate evidence discounts by exp(-1/128) on
    participation; squared-score variance discounts by its square. Block evidence
    is cumulative to preserve weak distributed structure. The fitted amplitude
    is nonnegative but not capped at one; only actual sample-loss line search
    restricts the fraction of the full candidate write.

    Admission A>sqrt(2*V*log(2*(D+1))) is an explicit diagnostic heuristic, not a
    proven anytime test under changing forecasts, repeated tests or dependent data.
    Feature participation and exact line curvature use supplied h=x*x and g=r*x.
    """

    def __init__(self, weight, *, control="hierarchical"):
        if weight.device.type != "cuda" or weight.dtype != torch.float32:
            raise ValueError("predictive directions requires CUDA float32 parameters")
        if control not in CONTROLS:
            raise ValueError(f"unknown control: {control}")
        self.anchor = weight.detach().clone()
        self.memory = torch.zeros_like(weight)
        self.target_sum = torch.zeros_like(weight)
        self.curvature_sum = torch.zeros_like(weight)
        self.slow_memory = torch.zeros_like(weight)
        self.slow_target = torch.zeros_like(weight)
        self.slow_curvature = torch.zeros_like(weight)
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
        return (self.memory, self.target_sum, self.curvature_sum, self.slow_memory,
                self.slow_target, self.slow_curvature, self.validation_target,
                self.validation_curvature, self.validation_square, self.allocation,
                self.block_target, self.block_curvature, self.block_square)

    @torch.no_grad()
    def step(self, weight, grad, curvature):
        tiny = torch.finfo(weight.dtype).tiny
        displacement = weight - self.anchor
        partial = curvature * displacement - grad
        decay = torch.where(curvature > 0, math.exp(-1 / 128), 1.0)
        predictor = self.memory.sign()
        slow_scale = self.slow_memory.square().mean(-1, keepdim=True).sqrt().clamp_min(tiny)
        block_predictor = self.slow_memory / slow_scale
        if self.control == "shuffled":
            predictor = predictor.roll(1, dims=-1)
            block_predictor = block_predictor.roll(1, dims=-1)
        product = predictor * partial
        self.validation_target.mul_(decay).add_(product)
        self.validation_curvature.mul_(decay).add_(curvature * predictor.square())
        self.validation_square.mul_(decay.square()).addcmul_(product, product)
        signed_features = grad.sign() * curvature.sqrt()
        forecast = (signed_features * block_predictor).sum(-1, keepdim=True)
        block_product = (forecast * (signed_features * displacement).sum(-1, keepdim=True)
                         - (grad * block_predictor).sum(-1, keepdim=True))
        self.block_target.add_(block_product)
        self.block_curvature.addcmul_(forecast, forecast)
        self.block_square.addcmul_(block_product, block_product)
        coordinate_fit = (self.validation_target / self.validation_curvature.clamp_min(tiny)).clamp_min(0)
        coordinate_admitted = self.validation_target > (2 * self.log_tests * self.validation_square).sqrt()
        block_fit = (self.block_target / self.block_curvature.clamp_min(tiny)).clamp_min(0)
        block_admitted = self.block_target > (2 * self.log_tests * self.block_square).sqrt()
        self.target_sum.mul_(decay).add_(partial)
        self.curvature_sum.mul_(decay).add_(curvature)
        self.memory.copy_(self.target_sum / self.curvature_sum.clamp_min(tiny))
        self.slow_target.add_(partial)
        self.slow_curvature.add_(curvature)
        self.slow_memory.copy_(self.slow_target / self.slow_curvature.clamp_min(tiny))
        new_scale = self.slow_memory.square().mean(-1, keepdim=True).sqrt().clamp_min(tiny)
        coordinate_candidate = coordinate_fit * self.memory.sign()
        block_candidate = block_fit * self.slow_memory / new_scale
        if self.control == "unvalidated":
            self.allocation.copy_((self.validation_curvature > 0).float())
            candidate = coordinate_candidate
        elif self.control == "coordinate":
            self.allocation.copy_(coordinate_admitted.float())
            candidate = torch.where(coordinate_admitted, coordinate_candidate, 0.0)
        elif self.control == "block":
            self.allocation.copy_(block_admitted.expand_as(weight).float())
            candidate = torch.where(block_admitted, block_candidate, 0.0)
        else:
            self.allocation.copy_((coordinate_admitted | block_admitted).float())
            candidate = torch.where(coordinate_admitted, coordinate_candidate,
                                    torch.where(block_admitted, block_candidate, 0.0))
        eligible = (self.allocation > 0) | (displacement != 0)
        writer = signed_features * eligible
        output_shift = (writer * (candidate - displacement)).sum(-1, keepdim=True)
        feature_norm = (curvature * eligible).sum(-1, keepdim=True)
        direction = writer * (output_shift / feature_norm.clamp_min(tiny))
        slope = (grad * direction).sum(-1, keepdim=True)
        fraction = (-slope / output_shift.square().clamp_min(tiny)).clamp(0, 1)
        weight.add_(fraction * direction)

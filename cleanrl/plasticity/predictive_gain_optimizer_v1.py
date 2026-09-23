"""Predictive Gain v1: unit initial gain, causal credit, no global LR sweep.

Later gradients score the effect of previous updates through an eligibility trace.
Plasticity grows when that effect helped and falls when it hurt. All coordinates
start at gain one, with only a sample-curvature stability normalization. This is
an IDBD/AutoStep-inspired diagnostic, not a claimed novel algorithm. The sign-credit
arm tests whether magnitude-history normalization strands once-noisy coordinates.
No posterior, covariance fit, model mixture, or parameter-change annealing.
"""

import torch


CONTROLS = ("predictive", "sign", "frozen", "shuffled")


class PredictiveGainState:
    """Curvature-assisted CUDA prototype, NOT a gradient-only drop-in optimizer.

    step receives the current loss gradient and nonnegative diagonal curvature.
    For the benchmark half-squared loss these are residual*x and x*x. Eligibility
    is diagonal and omits cross-coordinate Hessian and shared-normalizer derivatives.
    meta_gain=.1 and normalization horizon10000 are fixed mechanism constants, not
    sweep dimensions. Current gradients credit previous eligibility, never the new
    update. Normalization permanently reduces only participating coordinates so
    absence does not erase a rare feature's unused plasticity.
    """

    def __init__(self, weight, *, control="predictive"):
        if weight.device.type != "cuda" or weight.dtype != torch.float32:
            raise ValueError("predictive gain requires CUDA float32 parameters")
        if control not in CONTROLS:
            raise ValueError(f"unknown control: {control}")
        self.log_gain = torch.zeros_like(weight)
        self.trace = torch.zeros_like(weight)
        self.credit_scale = torch.zeros_like(weight)
        self.control = control

    def buffers(self):
        return (self.log_gain, self.trace, self.credit_scale)

    @torch.no_grad()
    def step(self, weight, grad, curvature):
        tiny = torch.finfo(weight.dtype).tiny
        active = curvature > 0
        log_curvature = curvature.log()
        old_normalizer = torch.logsumexp(self.log_gain + log_curvature, dim=-1, keepdim=True).clamp_min(0)
        old_fraction = (self.log_gain + log_curvature - old_normalizer).exp()
        credited_trace = self.trace.roll(1, dims=-1) if self.control == "shuffled" else self.trace
        credit = -grad * credited_trace
        if self.control != "frozen":
            magnitude = credit.abs()
            self.credit_scale.copy_(torch.maximum(
                magnitude, self.credit_scale + old_fraction * (magnitude - self.credit_scale) / 10000))
            normalized = credit.sign() if self.control == "sign" else credit / self.credit_scale.clamp_min(tiny)
            self.log_gain.add_(0.1 * normalized)
        normalizer = torch.logsumexp(self.log_gain + log_curvature, dim=-1, keepdim=True).clamp_min(0)
        self.log_gain.sub_(torch.where(active, normalizer, 0.0))
        fraction = (self.log_gain + log_curvature).exp()
        update = -grad.sign() * (self.log_gain + grad.abs().log()).exp()
        weight.add_(update)
        if self.control != "frozen":
            self.trace.mul_(1 - fraction).add_(update)

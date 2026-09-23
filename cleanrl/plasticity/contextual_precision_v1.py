"""Contextual Precision v1: predictive calibration changes local plasticity.

Existing perceptrons only. Each parameter carries an affine conditional Gaussian
model of the observed scalar network residual, plus an adaptive local step.
Calibration is scored before seeing its target. Its mean separates correctable
bias from variation; only its conditional variance scales current gradients.

No weight posterior, hidden-neuron target, forecast direction, or exact delayed
credit claim. The local step trace is a diagonal Gauss-Newton/stop-statistic
approximation. A shared scalar normalizes simultaneous linearized output motion.
Eleven FP32 optimizer-state scalars per parameter; not an LLM-ready footprint.
"""
import math

import torch

CONTROLS = ("conditional", "unconditional", "no_precision", "fixed_plasticity", "uncentered_noise")
COEFFICIENTS = ("mean_bias", "mean_context", "logvar_bias", "logvar_context")
COUNTERS = ("observation_count", "nll_sum", "standardized_square_sum", "predicted_variance_sum", "gain_sum")


class ContextualPrecisionState:
    mean_bias: torch.Tensor
    mean_context: torch.Tensor
    logvar_bias: torch.Tensor
    logvar_context: torch.Tensor
    observation_count: torch.Tensor
    nll_sum: torch.Tensor
    standardized_square_sum: torch.Tensor
    predicted_variance_sum: torch.Tensor
    gain_sum: torch.Tensor

    def __init__(self, weight, *, control="conditional", reference=None,
                 calibration_step=0.1, meta_step=1.0):
        if weight.device.type != "cuda" or weight.dtype != torch.float32 or weight.ndim != 2:
            raise ValueError("expected CUDA FP32 (independent models, parameters)")
        if control not in CONTROLS:
            raise ValueError(f"unknown control {control}")
        self.control = control
        self.calibration_step = calibration_step
        self.meta_step = meta_step
        for name in COEFFICIENTS:
            setattr(self, name, torch.zeros_like(weight))
            setattr(self, name + "_squared_gradient", torch.zeros_like(weight))
        if reference is None:
            self.log_step = torch.full_like(weight, -math.log(weight.shape[1]))
        else:
            self.log_step = 2 * reference.to(weight).expand_as(weight).clone().log()
        self.log_step_squared_gradient = torch.zeros_like(weight)
        self.sensitivity = torch.zeros_like(weight)
        for name in COUNTERS:
            setattr(self, name, torch.zeros(weight.shape[0], dtype=torch.float64, device=weight.device))

    def buffers(self):
        names = (*COEFFICIENTS, *(name + "_squared_gradient" for name in COEFFICIENTS),
                 "log_step", "log_step_squared_gradient", "sensitivity", *COUNTERS)
        return {name: getattr(self, name) for name in names}

    def predictive(self, context):
        c = torch.zeros_like(context) if self.control == "unconditional" else context
        mean = (torch.zeros_like(c) if self.control == "uncentered_noise"
                else self.mean_bias + self.mean_context * c)
        log_variance = self.logvar_bias + self.logvar_context * c
        return c, mean, log_variance, log_variance.exp()

    def calibration_transition(self, context, residual, active):
        """Pure transition; no current outcome can change its own scored variance."""
        c, mean, log_variance, variance = self.predictive(context)
        innovation = residual[:, None] - mean
        standardized_square = innovation.square() / variance
        mean_score = torch.where(active, innovation / variance, 0.0)
        variance_score = torch.where(active, 0.5 * (standardized_square - 1), 0.0)
        scores = (mean_score, mean_score * c, variance_score, variance_score * c)
        updated = {}
        for name, score in zip(COEFFICIENTS, scores):
            if self.control == "uncentered_noise" and name.startswith("mean_"):
                updated[name] = getattr(self, name)
                updated[name + "_squared_gradient"] = getattr(self, name + "_squared_gradient")
            else:
                accumulation = getattr(self, name + "_squared_gradient") + score.square()
                updated[name] = (getattr(self, name) + self.calibration_step * score /
                                 (accumulation + 1e-12).sqrt())
                updated[name + "_squared_gradient"] = accumulation
        active_count = active.sum(1).clamp_min(1)
        nll = 0.5 * (math.log(2 * math.pi) + log_variance + standardized_square)
        statistics = (torch.where(active, nll.double(), 0.0).sum(1) / active_count,
                      torch.where(active, standardized_square.double(), 0.0).sum(1) / active_count,
                      torch.where(active, variance.double(), 0.0).sum(1) / active_count)
        return variance, updated, statistics

    def transition(self, weight, gradient, jacobian, context, residual):
        """Return independent new tensors before any caller mutates persistent state."""
        active = jacobian != 0
        variance, updated, statistics = self.calibration_transition(context, residual, active)
        precision = torch.ones_like(variance) if self.control == "no_precision" else variance.reciprocal()
        if self.control == "fixed_plasticity":
            log_step = self.log_step
            accumulated_credit = self.log_step_squared_gradient
        else:
            credit = gradient * self.sensitivity * precision
            accumulated_credit = self.log_step_squared_gradient + credit.square()
            log_step = self.log_step - self.meta_step * credit / (accumulated_credit + 1e-12).sqrt()
        scale = log_step.exp() * precision
        gain = scale / (1 + (scale * jacobian.square()).sum(1, keepdim=True))
        displacement = -gain * gradient
        contraction = 1 - gain * jacobian.square()
        # This includes the own-log-step derivative of the shared denominator.
        # Cross-coordinate transport and calibration-state dependence are omitted.
        sensitivity = contraction * (self.sensitivity + displacement)
        updated.update(log_step=log_step, log_step_squared_gradient=accumulated_credit,
                       sensitivity=sensitivity)
        count = active.any(1).double()
        updated["observation_count"] = self.observation_count + count
        for name, value in zip(COUNTERS[1:4], statistics):
            updated[name] = getattr(self, name) + value
        updated["gain_sum"] = self.gain_sum + torch.where(active, gain.double(), 0.0).sum(1) / active.sum(1).clamp_min(1)
        return weight + displacement, updated

    @torch.no_grad()
    def step(self, weight, gradient, jacobian, context, residual):
        updated_weight, updated = self.transition(weight, gradient, jacobian, context, residual)
        weight.copy_(updated_weight)
        for name, value in updated.items():
            getattr(self, name).copy_(value)

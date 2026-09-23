"""Curvature-transported evidence, v1.

Hypothesis: reconstruct a supported parameter displacement instead of accumulating
noise through a gradient gate. For diagonal quadratic observations, h*(w-anchor)-g
transports the gradient to one fixed reference. Its sum estimates the optimum;
a spike/slab posterior mean trades reconstruction against estimation uncertainty.

This is a curvature-assisted optimizer, NOT a gradient-only Adam replacement.
Off-diagonal Hessian terms make the transport approximate. Residual-based sandwich
variance is a plug-in estimate, not a calibrated posterior under adaptive data.
The prior explicitly expresses sparsity and scale; both must face dense controls.
"""

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


@torch.no_grad()
def _transport_step(weight, grad, curvature, residual_sq, *, anchor, information,
                    score, score_variance, inclusion, prior_variance,
                    prior_logodds, keep):
    transported = curvature * (weight - anchor) - grad
    score.mul_(keep).add_(transported)
    information.mul_(keep).add_(curvature)
    score_variance.mul_(keep * keep).add_(residual_sq * curvature)
    # b=sum(h*d-g), A=sum(h), Q=sum(residual^2*h) define the approximate
    # likelihood N(b/A, Q/A^2). Avoid dividing by never-observed A=0.
    q = score_variance.clamp_min(torch.finfo(weight.dtype).tiny)
    log_q = q.log()
    log_ratio = 2 * information.log() + torch.log(prior_variance) - log_q
    shrink = torch.sigmoid(log_ratio)
    # Log-space evidence avoids overflowing A^2*tau^2 for valid finite
    # curvature. Infinite positive evidence correctly saturates inclusion.
    evidence = torch.exp(2 * score.abs().log() - log_q
                         + torch.nn.functional.logsigmoid(log_ratio))
    log_bayes_factor = 0.5 * (evidence - torch.nn.functional.softplus(log_ratio))
    inclusion.copy_(torch.sigmoid(prior_logodds + log_bayes_factor))
    slab_mean = score / information.clamp_min(torch.finfo(weight.dtype).tiny) * shrink
    weight.copy_(anchor + inclusion * slab_mean)


class TransportState:
    """Streaming state for a tensor of independent output rows, (..., fan_in).

    `grad` is the gradient of HALF squared error, `curvature` its nonnegative
    diagonal Gauss-Newton curvature, and `residual_sq` the PREUPDATE squared
    residual per row. For a scalar linear output these are residual*x, x*x,
    and residual**2. No clean label, support, or target variance is consumed.
    Computational batching of independent learners is not observation batching.

    memory=0 retains all observations; otherwise old information receives factor
    1-memory per observation. Squared information weights decay twice as fast.
    The anchor is the initialization, not zero unless initialized at zero.
    """

    def __init__(self, weight, *, prior_scale=1.0, prior_density=None, memory=0.0):
        if not weight.is_cuda or weight.dtype != torch.float32:
            raise ValueError("TransportState requires CUDA FP32 master weights")
        if weight.ndim < 1 or weight.shape[-1] < 1:
            raise ValueError("weight must have a nonempty parameter axis")
        if not math.isfinite(memory) or not 0 <= memory < 1:
            raise ValueError("memory must be finite and in [0, 1)")
        density = 1.0 / weight.shape[-1] if prior_density is None else prior_density
        scale = torch.as_tensor(prior_scale, device=weight.device, dtype=weight.dtype)
        if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("prior_scale must be finite and positive")
        self.prior_variance = scale.square()
        self.density = torch.as_tensor(density, device=weight.device, dtype=weight.dtype)
        # Configuration checks happen once, outside the compiled update path.
        if not bool(torch.isfinite(self.prior_variance).all() & (self.prior_variance > 0).all()):
            raise ValueError("prior_scale must be finite and nonzero")
        if not bool(torch.isfinite(self.density).all() & (self.density > 0).all()
                    & (self.density <= 1).all()):
            raise ValueError("prior_density must lie in (0, 1]")
        if torch.broadcast_shapes(
                weight.shape, self.prior_variance.shape, self.density.shape) != weight.shape:
            raise ValueError("prior tensors must broadcast into the weight shape")
        self.prior_logodds = self.density.log() - torch.log1p(-self.density)
        self.keep = 1.0 - memory
        self.anchor = weight.detach().clone()
        self.information = torch.zeros_like(weight)
        self.score = torch.zeros_like(weight)
        self.score_variance = torch.zeros_like(weight)
        self.inclusion = torch.zeros_like(weight)

    def buffers(self):
        """All mutable state, for exact scratch warmup/capture restoration."""
        return (self.anchor, self.information, self.score, self.score_variance,
                self.inclusion)

    @torch.no_grad()
    def step(self, weight, grad, curvature, residual_sq):
        _transport_step(
            weight, grad, curvature, residual_sq, anchor=self.anchor,
            information=self.information, score=self.score,
            score_variance=self.score_variance, inclusion=self.inclusion,
            prior_variance=self.prior_variance, prior_logodds=self.prior_logodds,
            keep=self.keep)


class PredictiveTransport(Optimizer):
    """Torch optimizer interface for explicitly supplied local curvature.

    step(curvatures=[...], residual_squares=[...]) follows parameter-group order.
    A curvature tensor matches its parameter; residual_squares is broadcastable
    per output row. The caller must use the same loss scaling for gradient,
    curvature and residual. Unsupported/sparse gradients fail rather than silently
    falling back. FP32 state is deliberate: accumulating weak evidence in BF16
    would erase sub-ULP observations even if model matmuls use BF16.
    """

    def __init__(self, params: Iterable, *, prior_scale=1.0, prior_density=None,
                 memory=0.0):
        super().__init__(params, dict(prior_scale=prior_scale,
                                     prior_density=prior_density, memory=memory))
        for group in self.param_groups:
            for parameter in group["params"]:
                self._initialize(parameter, group)

    def _initialize(self, parameter, group):
        transport = TransportState(
            parameter, prior_scale=group["prior_scale"],
            prior_density=group["prior_density"], memory=group["memory"])
        self.state[parameter].update({
            "anchor": transport.anchor,
            "information": transport.information,
            "score": transport.score,
            "score_variance": transport.score_variance,
            "inclusion": transport.inclusion,
            "prior_variance": transport.prior_variance,
            "prior_logodds": transport.prior_logodds,
        })

    @torch.no_grad()
    def step(self, closure=None, *, curvatures, residual_squares):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = [(group, parameter) for group in self.param_groups
                      for parameter in group["params"]]
        curvatures, residual_squares = list(curvatures), list(residual_squares)
        if len(curvatures) != len(parameters) or len(residual_squares) != len(parameters):
            raise ValueError("one curvature and residual-square entry required per parameter")
        for (group, parameter), curvature, residual_sq in zip(
                parameters, curvatures, residual_squares):
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise ValueError("sparse gradient layout is unsupported")
            if not self.state[parameter]:
                self._initialize(parameter, group)
            _transport_step(parameter, parameter.grad, curvature, residual_sq,
                            **self.state[parameter], keep=1.0 - group["memory"])
        return loss

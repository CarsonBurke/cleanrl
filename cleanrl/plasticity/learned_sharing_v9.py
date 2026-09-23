"""Learned Sharing v9: learn a signed parameter-sharing pattern and its amplitude.

The learned contrast pools magnitudes, not signs: coordinate signs come from
observed regression evidence, and one signed amplitude is fitted on subsequent
observations. This can express both constant and mixed-sign shared structure.
Direction changes discount obsolete contrast evidence by squared directional
agreement, rather than by a fixed clock. unpooled and fixed_evidence isolate
these choices. Row-wide sufficient statistics use FP64 to avoid losing small
increments; parameters and featurewise learning remain CUDA FP32, batch one.
Curvature-assisted linear prototype; admission is not an anytime-valid test.
"""
import math

import torch


CONTROLS = ("structured", "renewed", "no_shared", "no_dense", "no_gate", "misaligned", "unpooled", "fixed_evidence")


class LearnedSharingState:
    def __init__(self, weight, *, control="structured"):
        if weight.device.type != "cuda" or weight.dtype != torch.float32:
            raise ValueError("learned sharing requires CUDA float32 parameters")
        if control not in CONTROLS:
            raise ValueError(f"unknown control: {control}")
        self.control = control
        self.anchor = weight.detach().clone()
        self.memory = torch.zeros_like(weight)
        self.allocation = torch.zeros_like(weight)
        self.common_weight = torch.zeros_like(weight[..., :1])
        self.previous_direction = torch.zeros_like(weight)
        # Regression numerator/curvature and independent prequential score/energy.
        self.fine = tuple(torch.zeros_like(weight) for _ in range(4))
        self.common = tuple(torch.zeros_like(self.common_weight, dtype=torch.float64) for _ in range(4))
        # For the old learned contrast, regression numerator is also its score.
        self.dense = tuple(torch.zeros_like(self.common_weight, dtype=torch.float64) for _ in range(3))
        self.log_tests = math.log(2 * (weight.shape[-1] + 2))

    def buffers(self):
        return {"memory": self.memory, "allocation": self.allocation,
                "common_weight": self.common_weight, "previous_direction": self.previous_direction,
                **{f"fine_{i}": value for i, value in enumerate(self.fine)},
                **{f"common_{i}": value for i, value in enumerate(self.common)},
                **{f"dense_{i}": value for i, value in enumerate(self.dense)}}

    def _accumulate(self, moments, xy, xx, predictor, decay):
        a, h, score, energy = moments
        product = predictor * xy
        score.mul_(decay).add_(product)
        energy.mul_(decay * decay).addcmul_(product, product)
        a.mul_(decay).add_(xy)
        h.mul_(decay).add_(xx)

    def _fit(self, moments):
        a, h, score, energy = moments
        estimate = a / h.clamp_min(torch.finfo(a.dtype).tiny)
        admitted = score > (2 * self.log_tests * energy).sqrt()
        if self.control == "no_gate":
            admitted = h > 0
        return estimate, admitted

    def _direction(self, memory):
        direction = memory if self.control == "no_shared" else memory - memory.mean(-1, keepdim=True)
        if self.control != "unpooled":
            direction = direction.sign()
            if self.control != "no_shared":
                direction = direction - direction.mean(-1, keepdim=True)
        norm = direction.square().mean(-1, keepdim=True).sqrt()
        return direction / norm.clamp_min(torch.finfo(memory.dtype).tiny)

    @torch.no_grad()
    def step(self, weight, grad, curvature):
        if self.control == "misaligned":
            grad = grad.roll(1, dims=-1)
            curvature = curvature.roll(1, dims=-1)
        tiny = torch.finfo(weight.dtype).tiny
        displacement = weight - self.anchor
        departures = displacement - self.common_weight
        x = grad.sign() * curvature.sqrt()
        residual_abs = (grad.square().sum(-1, keepdim=True) /
                        curvature.sum(-1, keepdim=True).clamp_min(tiny)).sqrt()
        s = x.sum(-1, keepdim=True)
        common_h = s.square()
        common_xy = common_h * self.common_weight - grad.sum(-1, keepdim=True)
        self._accumulate(self.common, common_xy, common_h, self.common[0].sign(), 1.0)

        # These products validate a direction formed before this observation.
        direction = self._direction(self.memory)
        if self.control != "fixed_evidence":
            disagreement = (direction - self.previous_direction).square().mean(-1, keepdim=True)
            retention = (1 - 0.5 * disagreement).clamp(0, 1).square()
            self.dense[0].mul_(retention)
            self.dense[1].mul_(retention)
            self.dense[2].mul_(retention.square())
        self.previous_direction.copy_(direction)
        z = (x * direction).sum(-1, keepdim=True)
        contrast_y = (x * departures).sum(-1, keepdim=True) - residual_abs
        dense_product = z * contrast_y
        self.dense[0].add_(dense_product)
        self.dense[1].add_(z.square())
        self.dense[2].addcmul_(dense_product, dense_product)

        decay = torch.where(curvature > 0, math.exp(-1 / 128), 1.0) if self.control == "renewed" else 1.0
        fine_xy = curvature * departures - grad
        self._accumulate(self.fine, fine_xy, curvature, self.memory.sign(), decay)
        fine_fit, fine_admitted = self._fit(self.fine)
        common_fit, common_admitted = self._fit(self.common)
        dense_fit = self.dense[0] / self.dense[1].clamp_min(tiny)
        dense_admitted = self.dense[0] > (2 * self.log_tests * self.dense[2]).sqrt()
        if self.control == "no_gate":
            dense_admitted = self.dense[1] > 0
        common_fit = common_fit.to(weight.dtype)
        dense_fit = dense_fit.to(weight.dtype)
        self.memory.copy_(fine_fit)
        if self.control == "no_shared":
            common_admitted = torch.zeros_like(common_admitted)
        if self.control == "no_dense":
            dense_admitted = torch.zeros_like(dense_admitted)
        self.common_weight.copy_(torch.where(common_admitted, common_fit, 0.0))
        fine_candidate = torch.where(fine_admitted, fine_fit, 0.0)
        contrast_candidate = dense_fit * self._direction(self.memory)
        candidate = self.common_weight + torch.where(dense_admitted, contrast_candidate, fine_candidate)
        self.allocation.copy_((common_admitted | dense_admitted | fine_admitted).float())
        weight.copy_(self.anchor + candidate)

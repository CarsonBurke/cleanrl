"""Structured Score v7: infer signal amplitudes directly, not through a stale gate.

One live linear model has a shared coefficient, coordinate departures, and a
prequentially validated learned contrast direction. The shared component pools
repeated structure without fixing its sign or amplitude. Direct partial-target
regression removes the earlier sign-forecast calibration tax. Full coefficient
writes happen after each observation; there is no batch accumulation or LR grid.
This is a curvature-assisted linear prototype, not a general neural optimizer.
"""
import math

import torch


CONTROLS = ("structured", "renewed", "no_shared", "no_dense", "no_gate", "misaligned")


class StructuredScoreState:
    def __init__(self, weight, *, control="structured"):
        if weight.device.type != "cuda" or weight.dtype != torch.float32:
            raise ValueError("structured score requires CUDA float32 parameters")
        if control not in CONTROLS:
            raise ValueError(f"unknown control: {control}")
        self.control = control
        self.anchor = weight.detach().clone()
        self.memory = torch.zeros_like(weight)
        self.allocation = torch.zeros_like(weight)
        self.common_weight = torch.zeros_like(weight[..., :1])
        # Each regression stores sum(x*y), sum(x*x), sum(y*y), sum(weights),
        # sum(weights**2), and sum(weights**2*x*x). Inactive samples do not count.
        self.fine = tuple(torch.zeros_like(weight) for _ in range(6))
        self.common = tuple(torch.zeros_like(self.common_weight) for _ in range(6))
        self.dense = tuple(torch.zeros_like(self.common_weight) for _ in range(6))
        self.log_tests = math.log(2 * (weight.shape[-1] + 2))

    def buffers(self):
        return {"memory": self.memory, "allocation": self.allocation,
                "common_weight": self.common_weight,
                **{f"fine_{i}": value for i, value in enumerate(self.fine)},
                **{f"common_{i}": value for i, value in enumerate(self.common)},
                **{f"dense_{i}": value for i, value in enumerate(self.dense)}}

    def _accumulate(self, moments, xy, xx, yy, active, decay):
        a, h, q, mass, mass2, h2 = moments
        a.mul_(decay).add_(xy)
        h.mul_(decay).add_(xx)
        q.mul_(decay).add_(torch.where(active, yy, 0.0))
        mass.mul_(decay).add_(active)
        mass2.mul_(decay * decay).add_(active)
        h2.mul_(decay * decay).add_(xx)

    def _fit(self, moments):
        a, h, q, mass, mass2, h2 = moments
        tiny = torch.finfo(a.dtype).tiny
        estimate = a / h.clamp_min(tiny)
        residual_power = (q - a * estimate).clamp_min(0)
        dof = (mass - mass2 / mass.clamp_min(tiny)).clamp_min(1)
        variance = residual_power / dof
        effective_dof = (mass.square() / mass2.clamp_min(tiny) - 1).clamp_min(1)
        # Finite-sample tail inflation prevents nearly interpolated two/three-
        # observation fits from looking certain. This remains a diagnostic
        # heuristic with changing partial targets, not an anytime-valid test.
        threshold = effective_dof * torch.expm1(2 * self.log_tests / effective_dof)
        admitted = (mass >= 3) & (a.square() > threshold * variance * h2)
        if self.control == "no_gate":
            admitted = mass >= 3
        return estimate, admitted

    def _direction(self, memory):
        direction = memory if self.control == "no_shared" else memory - memory.mean(-1, keepdim=True)
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
        # g=r*x identifies x up to the harmless common sign of r. Regressions
        # below depend on sign-invariant products. A zero residual supplies no
        # signed row direction, but still supplies coordinate curvature.
        x = grad.sign() * curvature.sqrt()
        residual_abs = (grad.square().sum(-1, keepdim=True) /
                        curvature.sum(-1, keepdim=True).clamp_min(tiny)).sqrt()
        s = x.sum(-1, keepdim=True)
        common_h = s.square()
        common_xy = common_h * self.common_weight - grad.sum(-1, keepdim=True)
        common_y = s * self.common_weight - residual_abs
        self._accumulate(self.common, common_xy, common_h, common_y.square(), common_h > 0, 1.0)

        # Score the OLD contrast on the next observed outcome, before changing
        # its coordinates. Its target excludes the live shared contribution.
        direction = self._direction(self.memory)
        z = (x * direction).sum(-1, keepdim=True)
        contrast_y = (x * departures).sum(-1, keepdim=True) - residual_abs
        self._accumulate(self.dense, z * contrast_y, z.square(), contrast_y.square(), z != 0, 1.0)

        active = curvature > 0
        decay = torch.where(active, math.exp(-1 / 128), 1.0) if self.control == "renewed" else 1.0
        fine_xy = curvature * departures - grad
        fine_y2 = curvature * departures.square() - 2 * departures * grad + residual_abs.square()
        self._accumulate(self.fine, fine_xy, curvature, fine_y2, active, decay)
        fine_fit, fine_admitted = self._fit(self.fine)
        common_fit, common_admitted = self._fit(self.common)
        dense_fit, dense_admitted = self._fit(self.dense)
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

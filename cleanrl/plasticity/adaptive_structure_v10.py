"""Adaptive Structure v10: validate both representation and evidence lifetime.

A pooled signed contrast can denoise repeated magnitudes, but stock coefficients
need not share magnitudes. Score pooled and continuous contrasts prequentially;
choose the larger conservative per-observation gain, never their average. A
fast-vs-slow predictive-loss CUSUM renews individual coordinate histories only
when recent estimates predict better. fixed_history and pooled_only isolate
these choices. All coefficients remain freely fitted, with batch-one updates.
Linear curvature-assisted research prototype; no calibrated anytime guarantees.
"""
import math

import torch


CONTROLS = ("adaptive", "fixed_history", "pooled_only", "unpooled", "renewed",
            "no_shared", "no_dense", "no_gate", "misaligned")


class AdaptiveStructureState:
    def __init__(self, weight, *, control="adaptive"):
        if weight.device.type != "cuda" or weight.dtype != torch.float32:
            raise ValueError("adaptive structure requires CUDA float32 parameters")
        if control not in CONTROLS:
            raise ValueError(f"unknown control: {control}")
        self.control = control
        self.anchor = weight.detach().clone()
        self.memory = torch.zeros_like(weight)
        self.allocation = torch.zeros_like(weight)
        self.common_weight = torch.zeros_like(weight[..., :1])
        self.fine = tuple(torch.zeros_like(weight) for _ in range(4))
        self.fast = tuple(torch.zeros_like(weight) for _ in range(4))
        self.change_score = torch.zeros_like(weight)
        self.common = tuple(torch.zeros_like(self.common_weight, dtype=torch.float64) for _ in range(4))
        self.contrasts = tuple(tuple(torch.zeros_like(self.common_weight, dtype=torch.float64)
                                    for _ in range(4)) for _ in range(2))
        self.previous_directions = tuple(torch.zeros_like(weight) for _ in range(2))
        self.error_power = torch.zeros_like(self.common_weight, dtype=torch.float64)
        self.observations = torch.zeros_like(self.error_power)
        self.pooled_choice = torch.zeros_like(self.common_weight, dtype=torch.bool)
        self.renewals = torch.zeros_like(weight)
        self.log_tests = math.log(2 * (weight.shape[-1] + 3))

    def buffers(self):
        return {"memory": self.memory, "allocation": self.allocation,
                "common_weight": self.common_weight, "change_score": self.change_score,
                "error_power": self.error_power, "observations": self.observations,
                "pooled_choice": self.pooled_choice, "renewals": self.renewals,
                **{f"fine_{i}": x for i, x in enumerate(self.fine)},
                **{f"fast_{i}": x for i, x in enumerate(self.fast)},
                **{f"common_{i}": x for i, x in enumerate(self.common)},
                **{f"direction_{k}": x for k, x in enumerate(self.previous_directions)},
                **{f"contrast_{k}_{i}": x for k, values in enumerate(self.contrasts)
                   for i, x in enumerate(values)}}

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

    def _direction(self, memory, pooled):
        direction = memory if self.control == "no_shared" else memory - memory.mean(-1, keepdim=True)
        if pooled:
            direction = direction.sign()
            if self.control != "no_shared":
                direction = direction - direction.mean(-1, keepdim=True)
        return direction / direction.square().mean(-1, keepdim=True).sqrt().clamp_min(torch.finfo(memory.dtype).tiny)

    @torch.no_grad()
    def step(self, weight, grad, curvature):
        if self.control == "misaligned":
            grad = grad.roll(1, dims=-1)
            curvature = curvature.roll(1, dims=-1)
        tiny = torch.finfo(weight.dtype).tiny
        departures = weight - self.anchor - self.common_weight
        x = grad.sign() * curvature.sqrt()
        residual_square = grad.square().sum(-1, keepdim=True) / curvature.sum(-1, keepdim=True).clamp_min(tiny)
        residual_abs = residual_square.sqrt()
        s = x.sum(-1, keepdim=True)
        common_h = s.square()
        self._accumulate(self.common, common_h * self.common_weight - grad.sum(-1, keepdim=True),
                         common_h, self.common[0].sign(), 1.0)

        qualities, contrast_fits, contrast_admitted = [], [], []
        contrast_y = (x * departures).sum(-1, keepdim=True) - residual_abs
        for index, moments in enumerate(self.contrasts):
            direction = self._direction(self.memory, pooled=index == 1)
            disagreement = (direction - self.previous_directions[index]).square().mean(-1, keepdim=True)
            retention = (1 - 0.5 * disagreement).clamp(0, 1).square()
            self.previous_directions[index].copy_(direction)
            z = (x * direction).sum(-1, keepdim=True)
            product = z * contrast_y
            a, h, energy, mass = moments
            a.mul_(retention).add_(product)
            h.mul_(retention).add_(z.square())
            energy.mul_(retention.square()).addcmul_(product, product)
            mass.mul_(retention).add_(1)
            margin = a - (2 * self.log_tests * energy).sqrt()
            qualities.append(margin.clamp_min(0).square() / (h * mass).clamp_min(tiny))
            contrast_fits.append((a / h.clamp_min(tiny)).to(weight.dtype))
            contrast_admitted.append(h > 0 if self.control == "no_gate" else margin > 0)

        fine_xy = curvature * departures - grad
        fast_fit = self.fast[0] / self.fast[1].clamp_min(tiny)
        slow_fit = self.fine[0] / self.fine[1].clamp_min(tiny)
        # Compare OLD fast/slow predictions on today's partial target. Their
        # squared-loss difference does not require access to a clean label.
        improvement = 2 * (fast_fit - slow_fit) * fine_xy - (fast_fit.square() - slow_fit.square()) * curvature
        scale = (self.error_power / self.observations.clamp_min(1)).to(weight.dtype).clamp_min(tiny)
        self.change_score.add_(torch.where(self.observations > 0, improvement / (2 * scale), 0.0)).clamp_min_(0)
        self.error_power.add_(residual_square)
        self.observations.add_(1)
        decay = torch.where(curvature > 0, math.exp(-1 / 128), 1.0)
        self._accumulate(self.fast, fine_xy, curvature, fast_fit.sign(), decay)
        self._accumulate(self.fine, fine_xy, curvature, self.memory.sign(),
                         decay if self.control == "renewed" else 1.0)
        if self.control not in ("fixed_history", "renewed"):
            reset = self.change_score > self.log_tests
            for slow, fast in zip(self.fine, self.fast):
                slow.copy_(torch.where(reset, fast, slow))
            self.change_score.masked_fill_(reset, 0)
            self.renewals.add_(reset)

        fine_fit, fine_admitted = self._fit(self.fine)
        common_fit, common_admitted = self._fit(self.common)
        self.memory.copy_(fine_fit)
        use_pool = qualities[1] > qualities[0]
        if self.control == "pooled_only":
            use_pool = torch.ones_like(use_pool)
        elif self.control == "unpooled":
            use_pool = torch.zeros_like(use_pool)
        self.pooled_choice.copy_(use_pool)
        dense_admitted = torch.where(use_pool, contrast_admitted[1], contrast_admitted[0])
        if self.control == "no_dense":
            dense_admitted = torch.zeros_like(dense_admitted)
        if self.control == "no_shared":
            common_admitted = torch.zeros_like(common_admitted)
        self.common_weight.copy_(torch.where(common_admitted, common_fit.to(weight.dtype), 0.0))
        raw_candidate = contrast_fits[0] * self._direction(self.memory, pooled=False)
        pooled_candidate = contrast_fits[1] * self._direction(self.memory, pooled=True)
        contrast_candidate = torch.where(use_pool, pooled_candidate, raw_candidate)
        candidate = self.common_weight + torch.where(dense_admitted, contrast_candidate,
                                                      torch.where(fine_admitted, fine_fit, 0.0))
        self.allocation.copy_((common_admitted | dense_admitted | fine_admitted).float())
        weight.copy_(self.anchor + candidate)

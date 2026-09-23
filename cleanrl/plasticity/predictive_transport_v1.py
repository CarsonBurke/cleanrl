"""Predictive Transport v1: forecast gradients; separately credit model changes.

Same-example gradients at current/previous parameters give a SARAH/STORM-like
parameter-motion correction. Learned sigmoid assimilation sets forecast memory;
there is no preset forgetting window or confidence gate. Local write multipliers
receive subsequent-loss credit, separate from gradient-forecast accuracy.

Credit anchors retain net displacements until the parameter participates again.
Their gradient product is the derivative of rescaling a CURRENT displacement,
not an exact hypergradient through its historical learning trajectory. This does
not solve arbitrary delayed credit, representation retrieval, or policy shift.

Input shape is (independent models, flattened parameters). The caller supplies
paired first-order gradients, not features, targets, support, or curvature.
Two model evaluations must use the same example and stochastic realization.
FP32 research implementation; not yet a memory/compute-efficient LLM optimizer.
"""
import math

import torch


CONTROLS = ("predictive", "fixed_trust", "no_transport", "fixed_write",
            "raw_gradient", "one_step_credit", "miscredited")


class PredictiveTransportState:
    def __init__(self, weight, *, control="predictive", reference=None, groups=None, meta_step=0.1):
        if weight.device.type != "cuda" or weight.dtype != torch.float32 or weight.ndim != 2:
            raise ValueError("expected CUDA FP32 (independent models, parameters)")
        if control not in CONTROLS:
            raise ValueError(f"unknown control: {control}")
        self.control = control
        self.meta_step = meta_step
        self.previous_weight = weight.detach().clone()
        self.credit_anchor = weight.detach().clone()
        self.forecast = torch.zeros_like(weight)
        self.previous_innovation = torch.zeros_like(weight)
        self.scale_power = torch.zeros_like(weight)
        self.scale_mass = torch.zeros_like(weight)
        self.trust_logit = torch.zeros_like(weight)
        self.write_log_gain = torch.zeros_like(weight)
        self.reference = (torch.full_like(weight, 1 / math.sqrt(weight.shape[-1])) if reference is None
                          else reference.to(weight).expand_as(weight).clone())
        sizes = (weight.shape[-1],) if groups is None else tuple(groups)
        if any(size <= 0 for size in sizes) or sum(sizes) != weight.shape[-1]:
            raise ValueError("groups must partition the flattened parameter vector")
        self.slices = []
        start = 0
        for size in sizes:
            self.slices.append(slice(start, start + size))
            start += size
        self.group_log_gain = torch.zeros((weight.shape[0], len(sizes)), device=weight.device)
        self.step_count = torch.zeros((), device=weight.device)

    def buffers(self):
        return {name: getattr(self, name) for name in (
            "previous_weight", "credit_anchor", "forecast", "previous_innovation",
            "scale_power", "scale_mass", "trust_logit", "write_log_gain",
            "group_log_gain", "step_count")}

    @torch.no_grad()
    def step(self, weight, gradient, previous_gradient):
        tiny = torch.finfo(weight.dtype).tiny
        active = (gradient != 0) | (previous_gradient != 0)
        old_trust = self.trust_logit.sigmoid()
        scale_gain = torch.where(active, old_trust, 0.0)
        self.scale_power.lerp_(gradient.square(), scale_gain)
        self.scale_mass.lerp_(torch.ones_like(self.scale_mass), scale_gain)
        second_moment = self.scale_power / self.scale_mass.clamp_min(tiny)
        scale = second_moment.sqrt().clamp_min(tiny)

        # Old forecast and old innovation are scored on a NEW example, both at
        # the previous parameters. Forecast error is not task-loss utility.
        forecast_error = previous_gradient - self.forecast
        old_innovation = self.previous_innovation
        if self.control == "miscredited":
            old_innovation = old_innovation.roll(1, dims=-1)
        if self.control != "fixed_trust":
            prediction_credit = forecast_error * old_innovation / second_moment.clamp_min(tiny)
            self.trust_logit.add_(self.meta_step * prediction_credit.tanh())
        trust = self.trust_logit.sigmoid()

        pending = weight - self.credit_anchor
        last_change = weight - self.previous_weight
        if self.control == "one_step_credit":
            pending = last_change
        elif self.control == "miscredited":
            pending = pending.roll(1, dims=-1)
            last_change = last_change.roll(1, dims=-1)
        if self.control != "fixed_write":
            # Positive gradient dot displacement means that shrinking the
            # CURRENT pending displacement would reduce this observation's loss.
            unit_gradient = gradient / scale
            local_credit = unit_gradient * (pending / self.reference)
            local_credit = torch.where(active, local_credit, 0.0)
            self.write_log_gain.sub_(0.5 * self.meta_step * local_credit.tanh())
            group_credits = []
            for region in self.slices:
                contribution = unit_gradient[:, region] * (last_change[:, region] / self.reference[:, region])
                count = active[:, region].sum(-1).clamp_min(1)
                group_credits.append(contribution.sum(-1) / count)
            self.group_log_gain.sub_(0.5 * self.meta_step * torch.stack(group_credits, -1).tanh())

        transported = self.forecast
        if self.control != "no_transport":
            transported = transported + gradient - previous_gradient
        innovation = gradient - transported
        self.forecast.copy_(transported + trust * innovation)
        self.previous_innovation.copy_(innovation)
        direction = gradient if self.control == "raw_gradient" else self.forecast
        self.credit_anchor.copy_(torch.where(active, weight, self.credit_anchor))
        self.previous_weight.copy_(weight)
        for index, region in enumerate(self.slices):
            gain = (self.write_log_gain[:, region] + self.group_log_gain[:, index:index + 1]).exp()
            change = -self.reference[:, region] * gain * (direction[:, region] / scale[:, region])
            weight[:, region].add_(change)
        self.step_count.add_(1)

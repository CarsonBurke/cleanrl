"""Contextual Belief v2: evidence controls plasticity, not a free log step.

Per-parameter optimizer metadata models scalar predictive error as calibrated
zero, persistent context-dependent bias, or a changed bias. Gaussian coefficient
uncertainty is about ERROR predictions, never a posterior over network weights.
A moment-matched mixture filter learns from raw residuals even when writes are
small. Prior expected bias energy / total error energy scales current gradients.

Networks remain FP32 and unchanged. Small belief covariances use FP64 to preserve
precision under long, often collinear streams. Cost:21 FP64 scalars/parameter,
plus seven FP64 counters/model. This is a research design, not an LLM-ready cost.
"""
import math

import torch
import torch.nn.functional as F

CONTROLS = ("belief", "unconditional", "no_change", "no_epistemic", "noise_only")
COUNTERS = ("observation_count", "nll_sum", "standardized_square_sum", "predicted_variance_sum",
            "gain_sum", "bias_probability_sum", "change_probability_sum")


class ContextualBeliefState:
    observation_count: torch.Tensor
    nll_sum: torch.Tensor
    standardized_square_sum: torch.Tensor
    predicted_variance_sum: torch.Tensor
    gain_sum: torch.Tensor
    bias_probability_sum: torch.Tensor
    change_probability_sum: torch.Tensor

    def __init__(self, weight, *, control="belief"):
        if weight.ndim != 2 or weight.dtype != torch.float32 or weight.device.type != "cuda":
            raise ValueError("expected CUDA FP32 (models, parameters)")
        if control not in CONTROLS:
            raise ValueError(f"unknown control {control}")
        self.control = control
        shape = weight.shape
        options = {"device": weight.device, "dtype": torch.float64}
        self.mean = torch.zeros((*shape, 3), **options)
        self.covariance = torch.eye(3, **options).expand(*shape, 3, 3).clone()
        self.log_variance = torch.zeros_like(self.mean)
        self.variance_accumulator = torch.zeros_like(self.mean)
        self.log_odds = torch.zeros(shape, **options)
        self.hazard_alpha = torch.ones(shape, **options)
        self.hazard_beta = torch.full(shape, 999.0, **options)
        for name in COUNTERS:
            setattr(self, name, torch.zeros(shape[0], **options))

    def buffers(self):
        return {name: getattr(self, name) for name in (
            "mean", "covariance", "log_variance", "variance_accumulator", "log_odds",
            "hazard_alpha", "hazard_beta", *COUNTERS)}

    def predictive(self, context):
        c = context.double()
        if self.control == "unconditional":
            c = torch.zeros_like(c)
        phi = torch.cat((torch.ones_like(c[..., :1]), c), -1)
        # Match prior predictive uncertainty across contexts and the context-free
        # control; otherwise adding features also increases initial plasticity.
        phi = phi / phi.square().sum(-1, keepdim=True).sqrt()
        mean = (self.mean * phi).sum(-1)
        projected_covariance = (self.covariance * phi[..., None, :]).sum(-1)
        uncertainty = (projected_covariance * phi).sum(-1)
        reset_uncertainty = phi.square().sum(-1)
        noise = (self.log_variance * phi).sum(-1).exp()
        if self.control == "no_change":
            hazard = torch.zeros_like(self.hazard_alpha)
            log_hazard = torch.full_like(hazard, -float("inf"))
        else:
            hazard = self.hazard_alpha / (self.hazard_alpha + self.hazard_beta)
            log_hazard = hazard.log()
        log_stay = torch.log1p(-hazard)
        log_zero = torch.logaddexp(log_stay + F.logsigmoid(-self.log_odds), log_hazard - math.log(2))
        log_bias = log_stay + F.logsigmoid(self.log_odds)
        log_reset = log_hazard - math.log(2)
        mean_prediction = log_bias.exp() * mean
        full_energy = log_bias.exp() * (mean.square() + uncertainty) + log_reset.exp() * reset_uncertainty
        predictive_variance = noise + full_energy - mean_prediction.square()
        energy = log_bias.exp() * mean.square() if self.control == "no_epistemic" else full_energy
        gain = 1 / (1 + noise) if self.control == "noise_only" else energy / (energy + noise)
        return {"phi": phi, "mean": mean, "projection": projected_covariance,
                "uncertainty": uncertainty, "reset_uncertainty": reset_uncertainty,
                "noise": noise, "log_zero": log_zero, "log_bias": log_bias, "log_reset": log_reset,
                "log_hazard": log_hazard, "prediction": mean_prediction,
                "predictive_variance": predictive_variance, "gain": gain}

    @staticmethod
    def posterior(mean, covariance, phi, error, noise):
        projection = (covariance * phi[..., None, :]).sum(-1)
        variance = noise + (phi * projection).sum(-1)
        kalman = projection / variance[..., None]
        updated_mean = mean + kalman * error[..., None]
        identity = torch.eye(3, dtype=mean.dtype, device=mean.device)
        residual_map = identity - kalman[..., :, None] * phi[..., None, :]
        # Joseph form is PSD by construction; no eigenvalue/variance clamp.
        transformed = residual_map @ covariance @ residual_map.transpose(-1, -2)
        updated_covariance = transformed + noise[..., None, None] * kalman[..., :, None] * kalman[..., None, :]
        return updated_mean, updated_covariance

    def belief_transition(self, context, residual, active):
        p = self.predictive(context)
        phi, noise = p["phi"], p["noise"]
        r = residual.double()[:, None]
        errors = torch.stack((r.expand_as(noise), r - p["mean"], r.expand_as(noise)), -1)
        variances = torch.stack((noise, noise + p["uncertainty"], noise + p["reset_uncertainty"]), -1)
        log_prior = torch.stack((p["log_zero"], p["log_bias"], p["log_reset"]), -1)
        log_joint = log_prior - 0.5 * (math.log(2 * math.pi) + variances.log() + errors.square() / variances)
        log_evidence = torch.logsumexp(log_joint, -1)
        responsibility = (log_joint - log_evidence[..., None]).exp()
        alternative_evidence = torch.logaddexp(log_joint[..., 1], log_joint[..., 2])
        persistent_weight = (log_joint[..., 1] - alternative_evidence).exp()
        changed_weight = (log_joint[..., 2] - alternative_evidence).exp()
        continuation_mean, continuation_covariance = self.posterior(
            self.mean, self.covariance, phi, r - p["mean"], noise)
        identity = torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(self.covariance)
        reset_mean, reset_covariance = self.posterior(torch.zeros_like(self.mean), identity, phi, r, noise)
        mean = persistent_weight[..., None] * continuation_mean + changed_weight[..., None] * reset_mean
        difference_old = continuation_mean - mean
        difference_new = reset_mean - mean
        covariance = (persistent_weight[..., None, None] *
                      (continuation_covariance + difference_old[..., :, None] * difference_old[..., None, :]) +
                      changed_weight[..., None, None] *
                      (reset_covariance + difference_new[..., :, None] * difference_new[..., None, :]))
        # Exact score of the current Gaussian-mixture predictive likelihood with
        # respect to log observation variance, holding prior beliefs fixed.
        variance_score = 0.5 * (responsibility * (noise[..., None] / variances) *
                                (errors.square() / variances - 1)).sum(-1)
        coefficient_score = torch.where(active[..., None], variance_score[..., None] * phi, 0.0)
        accumulator = self.variance_accumulator + coefficient_score.square()
        log_variance = self.log_variance + 0.1 * coefficient_score / (accumulator + 1e-12).sqrt()
        log_odds = alternative_evidence - log_joint[..., 0]
        changed_to_zero = (p["log_hazard"] - math.log(2) - p["log_zero"]).exp()
        change_probability = responsibility[..., 2] + responsibility[..., 0] * changed_to_zero
        updated = {
            "mean": torch.where(active[..., None], mean, self.mean),
            "covariance": torch.where(active[..., None, None], covariance, self.covariance),
            "log_variance": log_variance,
            "variance_accumulator": accumulator,
            "log_odds": torch.where(active, log_odds, self.log_odds),
            "hazard_alpha": self.hazard_alpha + torch.where(active, change_probability, 0.0),
            "hazard_beta": self.hazard_beta + torch.where(active, 1 - change_probability, 0.0),
        }
        if self.control == "no_change":
            updated["hazard_alpha"] = self.hazard_alpha
            updated["hazard_beta"] = self.hazard_beta
        diagnostics = (-log_evidence, (r - p["prediction"]).square() / p["predictive_variance"],
                       p["predictive_variance"], responsibility[..., 1] + responsibility[..., 2], change_probability)
        return p, updated, diagnostics

    def transition(self, weight, gradient, jacobian, context, residual):
        active = jacobian != 0
        prediction, updated, diagnostics = self.belief_transition(context, residual, active)
        signal_gain = prediction["gain"]
        # Average predictive confidence across active Jacobian directions: summing
        # it would turn redundant parameters into falsely stronger evidence.
        jacobian_energy = jacobian.double().square().sum(1, keepdim=True)
        denominator = torch.where(jacobian_energy > 0, jacobian_energy, 1.0)
        gain = signal_gain / denominator
        next_weight = weight - (gain * gradient.double()).to(weight.dtype)
        count = active.sum(1).clamp_min(1)
        updated["observation_count"] = self.observation_count + active.any(1).double()
        for name, value in zip(("nll_sum", "standardized_square_sum", "predicted_variance_sum",
                                "bias_probability_sum", "change_probability_sum"), diagnostics):
            updated[name] = getattr(self, name) + torch.where(active, value, 0.0).sum(1) / count
        updated["gain_sum"] = self.gain_sum + torch.where(active, gain, 0.0).sum(1) / count
        return next_weight, updated

    @torch.no_grad()
    def step(self, weight, gradient, jacobian, context, residual):
        next_weight, updated = self.transition(weight, gradient, jacobian, context, residual)
        weight.copy_(next_weight)
        for name, value in updated.items():
            getattr(self, name).copy_(value)

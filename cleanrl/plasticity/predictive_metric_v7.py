"""Predictive Metric v7: standalone evidence-controlled proximal updates.

Retain v4's target-function evidence; replace its unit parameter-space ridge
with empirical Jacobian curvature. Predictive risk chooses the output-step
fraction, not Adam. A confidence ceiling prevents tiny inclusion probabilities
from cancelling out of the metric. No target substitution or extra predictor.
This is a confidence-weighted diagonal Gauss-Newton/NLMS experiment, not a
solution to cumulative-evidence staleness or a general vector-loss optimizer.
"""
import torch

from cleanrl.plasticity.predictive_need_v4 import PredictiveNeedState

CONTROLS = ("metric", "local_need", "uncapped", "no_need", "v4_reference")


class PredictiveMetricState(PredictiveNeedState):
    def __init__(self, weight, *, control="metric"):
        if control not in CONTROLS[:-1]:
            raise ValueError(f"unknown metric control: {control}")
        # The frozen predecessor supplies only target-function statistics.
        # No inherited parameter transition is called.
        super().__init__(weight, control="predictive_need")
        self.metric_control = control
        self.mean_jacobian_square = torch.zeros_like(weight, dtype=torch.float64)

    def transition(self, weight, gradient, jacobian, feature, target, residual, prediction):
        stats = self.statistics(feature, prediction)
        count = self.observation_count + 1
        j = jacobian.double()
        # Current inputs/Jacobians are available before observing the target.
        # Including them defines curvature on newly encountered directions.
        curvature = self.mean_jacobian_square + (j.square() - self.mean_jacobian_square) / count[:, None]
        valid = stats["noise_ready"] & (self.m2_x > 0)
        participation = torch.where(valid, stats["participation"], 0.)
        metric = participation / torch.where(curvature > 0, curvature, 1.)
        energy = metric * j.square()
        total_energy = energy.sum(1, keepdim=True)
        active = j != 0
        confidence = torch.where(active, participation, 0.).amax(1, keepdim=True)
        latent_error_energy = torch.where(
            valid, stats["error_location"].square() + stats["mean_uncertainty"], 0.)
        noise = torch.where(valid, stats["noise"], 0.)
        # Pool moments, not optimal ratios: this is the risk-minimizing scalar
        # correction for the stated working mixture of local H1 models.
        pooled_error = (energy * latent_error_energy).sum(1, keepdim=True)
        pooled_total = (energy * (latent_error_energy + noise)).sum(1, keepdim=True)
        fraction = pooled_error / torch.where(pooled_total > 0, pooled_total, 1.)
        if self.metric_control == "local_need":
            fraction = stats["write_fraction"]
        elif self.metric_control == "uncapped":
            confidence = torch.ones_like(confidence)
        elif self.metric_control == "no_need":
            fraction = torch.ones_like(fraction)
        gain = confidence * fraction * metric / (1 + total_energy)
        next_weight = weight - (gain * gradient.double()).to(weight.dtype)

        observed = target.double()
        dx = feature.double() - self.mean_x
        dy = observed - self.mean_y
        mean_x = self.mean_x + dx / count[:, None]
        mean_y = self.mean_y + dy / count
        updated = {
            "observation_count": count, "mean_x": mean_x, "mean_y": mean_y,
            "m2_x": self.m2_x + dx * (feature.double() - mean_x),
            "cross_xy": self.cross_xy + dx * (observed - mean_y)[:, None],
            "m2_y": self.m2_y + dy * (observed - mean_y),
            "mean_jacobian_square": curvature,
        }
        active_count = active.sum(1).clamp_min(1)
        for name, value in (("gain_sum", gain), ("participation_sum", stats["participation"]),
                            ("log_bayes_factor_sum", stats["log_bf"])):
            updated[name] = getattr(self, name) + torch.where(active, value, 0.).sum(1) / active_count
        noise_active = active & stats["noise_ready"]
        updated["noise_observation_count"] = self.noise_observation_count + noise_active.any(1).double()
        updated["noise_variance_sum"] = self.noise_variance_sum + torch.where(
            noise_active, stats["noise"], 0.).sum(1) / active_count
        updated["prior_probability_sum"] = self.prior_probability_sum + stats["prior_probability"][:, 0]
        return next_weight, updated

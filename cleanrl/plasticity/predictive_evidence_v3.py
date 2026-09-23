"""Predictive Evidence v3: permit writes for a repeatable local target relation.

Compare a constant target predictor against a centered local-feature predictor
using a unit-information g-prior Bayes factor. Fit a shared inclusion rate by
composite empirical Bayes; unknown coefficients alone do not justify writing.
Pooled target moments estimate unexplained variance without a noise learning
rate. Ordinary gradients update the original weights, with bounded signal/noise
shrinkage and participation-weighted normalization. This first version tests
stationary univariate associations, not general nonlinear or drifting credit.
"""
import torch

CONTROLS = ("evidence", "fixed_prior", "no_evidence", "no_noise_shrinkage", "residual_statistics")
COUNTERS = ("observation_count", "noise_observation_count", "gain_sum", "participation_sum",
            "noise_variance_sum", "prior_probability_sum", "log_bayes_factor_sum")


def fit_inclusion_probability(log_bf):
    """Leftmost marginal-likelihood maximizer; correlated tests are composite evidence."""
    peak = log_bf.amax(1, keepdim=True)
    # Scaled derivative at zero: finite even for overwhelming positive evidence.
    left_derivative = ((log_bf - peak).exp() - (-peak).exp()).sum(1, keepdim=True)
    # logBF >= -.5*log1p(n), so this endpoint expression is also well scaled.
    right_inverse_sum = torch.expm1(-log_bf).sum(1, keepdim=True)
    low = torch.zeros_like(peak)
    high = torch.ones_like(peak)
    for _ in range(32):
        middle = (low + high) * .5
        posterior = torch.sigmoid(log_bf + torch.logit(middle))
        increasing = posterior.mean(1, keepdim=True) > middle
        low = torch.where(increasing, middle, low)
        high = torch.where(increasing, high, middle)
    interior = (low + high) * .5
    prior = torch.where(left_derivative <= 0, 0.,
                        torch.where(right_inverse_sum <= 0, 1., interior))
    return prior, torch.sigmoid(log_bf + torch.logit(prior))


class PredictiveEvidenceState:
    mean_x: torch.Tensor
    m2_x: torch.Tensor
    cross_xy: torch.Tensor
    mean_y: torch.Tensor
    m2_y: torch.Tensor
    observation_count: torch.Tensor
    noise_observation_count: torch.Tensor
    gain_sum: torch.Tensor
    participation_sum: torch.Tensor
    noise_variance_sum: torch.Tensor
    prior_probability_sum: torch.Tensor
    log_bayes_factor_sum: torch.Tensor

    def __init__(self, weight, *, control="evidence"):
        if control not in CONTROLS:
            raise ValueError(f"unknown control: {control}")
        self.control = control
        for name in ("mean_x", "m2_x", "cross_xy"):
            setattr(self, name, torch.zeros_like(weight, dtype=torch.float64))
        for name in ("mean_y", "m2_y", *COUNTERS):
            setattr(self, name, torch.zeros(weight.shape[0], device=weight.device, dtype=torch.float64))

    def buffers(self):
        return {name: value for name, value in vars(self).items() if isinstance(value, torch.Tensor)}

    def statistics(self, feature):
        x = feature.double()
        n = self.observation_count[:, None]
        yy = self.m2_y[:, None]
        safe_xx = torch.where(self.m2_x > 0, self.m2_x, 1.)
        safe_yy = torch.where(yy > 0, yy, 1.)
        identified = (n >= 2) & (self.m2_x > 0) & (yy > 0)
        correlation_square = torch.where(identified, self.cross_xy.square() / (safe_xx * safe_yy), 0.)
        shrink = n / (1 + n)
        log_bf = torch.where(identified,
                             -.5 * torch.log1p(n) - .5 * (n - 1) * torch.log1p(-shrink * correlation_square), 0.)
        slope = torch.where(identified, shrink * self.cross_xy / safe_xx, 0.)
        if self.control == "fixed_prior":
            prior = torch.full_like(n, .5)
            participation = log_bf.sigmoid()
        elif self.control == "no_evidence":
            prior = torch.ones_like(n)
            participation = torch.ones_like(log_bf)
        else:
            prior, participation = fit_inclusion_probability(log_bf)
        effect = slope * (x - self.mean_x)
        noise_ready = n > 3
        # Jeffreys-scale posterior mean exists only after four observations.
        divisor = torch.where(noise_ready, n - 3, 1.)
        baseline_noise = torch.where(noise_ready, yy / divisor, float("inf"))
        noise = baseline_noise * (1 - participation * shrink * correlation_square)
        effect_square = effect.square()
        total = effect_square + noise
        signal_fraction = effect_square / torch.where(total > 0, total, 1.)
        if self.control == "no_noise_shrinkage":
            signal_fraction = noise_ready.expand_as(effect).double()
        # Student locations remain defined even before posterior means exist.
        # Parameter writes wait until the noise variance expectation is finite.
        return {"log_bf": log_bf, "prior_probability": prior, "participation": participation,
                "slope_location": slope, "predictive_location": self.mean_y[:, None] + effect,
                "effect": effect, "noise": noise, "noise_ready": noise_ready,
                "signal_fraction": signal_fraction}

    def transition(self, weight, gradient, jacobian, feature, target, residual):
        prediction = self.statistics(feature)
        participation = prediction["participation"]
        denominator = 1 + (participation * jacobian.double().square()).sum(1, keepdim=True)
        gain = participation * prediction["signal_fraction"] / denominator
        next_weight = weight - (gain * gradient.double()).to(weight.dtype)

        observed = (residual if self.control == "residual_statistics" else target).double()
        count = self.observation_count + 1
        dx = feature.double() - self.mean_x
        dy = observed - self.mean_y
        mean_x = self.mean_x + dx / count[:, None]
        mean_y = self.mean_y + dy / count
        updated = {"observation_count": count, "mean_x": mean_x, "mean_y": mean_y,
                   "m2_x": self.m2_x + dx * (feature.double() - mean_x),
                   "cross_xy": self.cross_xy + dx * (observed - mean_y)[:, None],
                   "m2_y": self.m2_y + dy * (observed - mean_y)}
        active = jacobian != 0
        active_count = active.sum(1).clamp_min(1)
        for name, value in (("gain_sum", gain), ("participation_sum", participation),
                            ("log_bayes_factor_sum", prediction["log_bf"])):
            updated[name] = getattr(self, name) + torch.where(active, value, 0.).sum(1) / active_count
        noise_active = active & prediction["noise_ready"]
        updated["noise_observation_count"] = self.noise_observation_count + noise_active.any(1).double()
        updated["noise_variance_sum"] = self.noise_variance_sum + torch.where(noise_active, prediction["noise"], 0.).sum(1) / active_count
        updated["prior_probability_sum"] = self.prior_probability_sum + prediction["prior_probability"][:, 0]
        return next_weight, updated

    @torch.no_grad()
    def step(self, weight, gradient, jacobian, feature, target, residual):
        next_weight, updated = self.transition(weight, gradient, jacobian, feature, target, residual)
        weight.copy_(next_weight)
        for name, value in updated.items():
            getattr(self, name).copy_(value)

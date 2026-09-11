"""Factorized spike-and-slab assumed-density filtering, not exact joint Bayes.

Each coordinate has q(w_i) = (1-p_i) delta_0 + p_i N(m_i, v_i). Before
observing a label, an independent latent reset redraws each coefficient from
its original zero-mean spike/slab prior with probability ``hazard``. The
conditional active mixture is projected to one Gaussian by matching moments.
For conditioning, OTHER coordinates are replaced by a Gaussian with their
aggregate mean and variance. Their posterior correlations are discarded.

Unlike the plug-in ``bayes`` diagnostic, inclusion evidence integrates out
the active slab (including its log-determinant/Occam penalty), forecast
variance includes inclusion uncertainty, and slab means use the conditional
active residual. No clipping, confidence floor, diffusion, or noise fitting
is performed. Conditional means minimize squared error under the approximate
posterior, not under arbitrary data-generating processes. In one dimension
with no resets the spike/slab conjugate posterior is exact; in multiple
dimensions even the dense limit discards correlations after each observation.
"""

import math

import torch
import torch.nn.functional as F


class SparsePosterior:
    """CUDA FP32 filter batched over aligned configuration dictionaries.

    ``configs[k]`` requires ``prior`` (finite positive slab variance per
    coordinate), ``inclusion`` (0 < p0 <= 1), and ``hazard`` (0 <= h <= 1).
    There is no Cartesian expansion. Exactly dense inclusion is supported;
    an identically absent prior is excluded because its conditional slab is
    undefined and can never learn a relationship.

    Inputs are a finite CUDA vector x[D], scalar y, and strictly positive
    scalar or length-K observation_variance R, on the constructor device. R is supplied
    causally by the caller and is not estimated here. predict has no label
    argument and does not mutate state. update returns that same pre-label
    forecast before storing the posterior. Both return vectors of length K.
    Callers own compilation, CUDA graph capture, and input validation; there
    is no per-update synchronization, random draw, or autograd graph.
    """

    def __init__(self, input_dim: int, configs: list[dict], device):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("SparsePosterior requires CUDA")
        if input_dim <= 0 or not configs:
            raise ValueError("input_dim and the configuration count must be positive")
        priors = [float(c["prior"]) for c in configs]
        inclusions = [float(c["inclusion"]) for c in configs]
        hazards = [float(c["hazard"]) for c in configs]
        if any(not math.isfinite(v) or v <= 0 for v in priors):
            raise ValueError("prior must be a finite positive slab variance")
        if any(not 0 < p <= 1 for p in inclusions):
            raise ValueError("inclusion must lie in (0, 1]")
        if any(not 0 <= h <= 1 for h in hazards):
            raise ValueError("hazard must lie in [0, 1]")

        def column(values):
            return torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(-1)

        self.prior = column(priors)
        self.hazard = column(hazards)
        self.keep = column([1 - h for h in hazards])
        self.static = self.hazard == 0
        self.dense = column([p == 1 for p in inclusions]).bool()
        self._all_static = all(h == 0 for h in hazards)
        self._all_dense = all(p == 1 for p in inclusions)
        self.log_keep = column([math.log1p(-h) if h < 1 else -math.inf for h in hazards])
        self.log_reset_on = column([math.log(h) + math.log(p) if h else -math.inf
                                    for h, p in zip(hazards, inclusions)])
        self.log_reset_off = column([math.log(h) + math.log1p(-p) if h and p < 1 else -math.inf
                                     for h, p in zip(hazards, inclusions)])
        logits = [math.log(p) - math.log1p(-p) if p < 1 else math.inf for p in inclusions]
        self.log_odds = column(logits).expand(-1, input_dim).clone()
        self.slab_mean = torch.zeros((len(configs), input_dim), device=device, dtype=torch.float32)
        self.slab_variance = self.prior.expand_as(self.slab_mean).clone()

    def state_tensors(self) -> list[torch.Tensor]:
        """All mutable tensors, in stable order; restore with in-place copy_."""
        return [self.log_odds, self.slab_mean, self.slab_variance]

    def _next_prior(self):
        if self._all_static:
            return self.log_odds, self.log_odds.sigmoid(), self.slab_mean, self.slab_variance
        if self._all_dense:
            mean = self.keep * self.slab_mean
            variance = (self.keep * self.slab_variance + self.hazard * self.prior
                        + self.keep * self.hazard * self.slab_mean.square())
            return self.log_odds, torch.ones_like(mean), mean, variance

        # log(p') and log(1-p') are computed separately, never via 1-sigmoid.
        # This also keeps finite evidence recoverable when sigmoid rounds to 1.
        log_retained_on = self.log_keep + F.logsigmoid(self.log_odds)
        log_on = torch.logaddexp(log_retained_on, self.log_reset_on)
        log_off = torch.logaddexp(self.log_keep + F.logsigmoid(-self.log_odds), self.log_reset_off)
        odds = torch.where(self.static, self.log_odds, log_on - log_off)
        probability = odds.sigmoid()
        # a = (1-h)p/p': retained fraction conditional on the slab being active.
        retained = (log_retained_on - log_on).exp()
        retained = torch.where(self.dense, self.keep, retained)
        mean = retained * self.slab_mean
        variance = (retained * self.slab_variance + (1 - retained) * self.prior
                    + retained * (1 - retained) * self.slab_mean.square())
        mean = torch.where(self.static, self.slab_mean, mean)
        variance = torch.where(self.static, self.slab_variance, variance)
        return odds, probability, mean, variance

    @staticmethod
    def _forecast(x, observation_variance, probability, mean, variance):
        contribution = probability * mean * x
        own_variance = x.square() * (probability * variance + probability * (1 - probability) * mean.square())
        latent_variance = own_variance.sum(-1)
        return contribution, own_variance, latent_variance, contribution.sum(-1), observation_variance + latent_variance

    @torch.no_grad()
    def predict(self, x, observation_variance):
        """Return next-observation mean and total variance without mutation."""
        _, probability, mean, variance = self._next_prior()
        _, _, _, forecast, total_variance = self._forecast(x, observation_variance, probability, mean, variance)
        return forecast, total_variance

    @torch.no_grad()
    def update(self, x, y, observation_variance):
        """Return the pre-label forecast, then condition all coordinates in parallel."""
        odds, probability, mean, variance = self._next_prior()
        contribution, own_variance, latent_variance, forecast, total_variance = self._forecast(
            x, observation_variance, probability, mean, variance)
        residual_off = y - forecast.unsqueeze(-1) + contribution
        other_variance = observation_variance.reshape(-1, 1) + (latent_variance.unsqueeze(-1) - own_variance)
        c = mean * x
        d = variance * x.square()
        on_variance = other_variance + d
        residual_on = residual_off - c
        posterior_mean = mean + variance * x * residual_on / on_variance
        posterior_variance = variance * (other_variance / on_variance)
        # log N(r_off; c, R_other+d) / N(r_off; 0, R_other), without
        # subtracting two nearly equal squared residual likelihoods.
        log_bayes_factor = 0.5 * (
            (residual_off.square() * (d / other_variance) + 2 * residual_off * c - c.square()) / on_variance
            - torch.log1p(d / other_variance))
        observed = x != 0
        self.log_odds.copy_(torch.where(self.dense, odds, torch.where(observed, odds + log_bayes_factor, odds)))
        self.slab_mean.copy_(torch.where(observed, posterior_mean, mean))
        self.slab_variance.copy_(torch.where(observed, posterior_variance, variance))
        return forecast, total_variance

    @torch.no_grad()
    def mean_weights(self):
        """Next-prior coefficient mean [K,D], including the latent reset."""
        return self.keep * self.log_odds.sigmoid() * self.slab_mean

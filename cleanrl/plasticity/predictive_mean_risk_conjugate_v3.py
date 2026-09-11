"""Shared sparse experts, comparing density evidence with predictive-mean risk.

Hypothesis: density scoring can reject useful conditional means because its
variance model is wrong. The risk rows instead score squared-loss improvement
over zero, using one causal target-energy scale shared by every expert. This
is generalized exponential weighting, not posterior model probability. Static
and switching rows share exactly the same independently trained experts within
v3; the new scale estimator changes expert trajectories relative to v2.

For scale estimation, pre-label residuals (and, separately, targets around zero)
are treated as conditionally independent zero-mean Gaussian observations with
unknown variance and an inverse-Gamma(alpha0=2, beta0=1) prior. Its mean is one.
With weighted count n and squared-observation sum Q, the posterior parameters
are alpha=2+n/2 and beta=1+Q/2, giving mean variance (2+Q)/(2+n). This is also
the zero-mean Student-t posterior predictive variance under that scale model.
Only observed sufficient statistics are discounted by 1-noise_rate; the proper
prior is retained. Rate zero accumulates all evidence, while a positive rate
uses exponentially discounted evidence with finite effective memory. Cached
posterior means are used only on the next observation, with no numerical floor.

SparsePosterior remains a factorized assumed-density approximation and scoring
still uses Gaussian moments, not the exact Student-t scale mixture. Residual
scales include model error: they do not identify irreducible observation noise.
Neither objective guarantees sparse recovery, calibrated uncertainty, or
transfer to stocks or neural training. No clean targets, support information,
oracle noise, or evaluation data enter this model. Callers own compilation,
CUDA graph capture, and finite input validation; all update inputs must be FP32
CUDA tensors on the model device.
"""

import math

import torch

from cleanrl.plasticity.predictive_structure_filter_v1 import SparsePosterior


class PredictiveMeanRisk:
    """Four adaptive rows, one fixed-prior row, then nine individual experts."""

    output_names = (
        'likelihood_static', 'likelihood_switching',
        'mean_risk_static', 'mean_risk_switching', 'prior_mixture',
        'null', *(f'sparse_{i}' for i in range(8)),
    )

    def __init__(self, input_dim, device, hazard=1e-4, noise_rate=.001):
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError('input_dim must be a positive integer')
        hazard, noise_rate = float(hazard), float(noise_rate)
        if not math.isfinite(hazard) or not 0 <= hazard <= 1:
            raise ValueError('hazard must be finite and lie in [0, 1]')
        if not math.isfinite(noise_rate) or not 0 <= noise_rate < 1:
            raise ValueError('noise_rate must be finite and lie in [0, 1)')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('PredictiveMeanRisk requires CUDA')
        self.hazard, self.noise_rate = hazard, noise_rate
        self.configs = [dict(prior=p, inclusion=r, hazard=h)
                        for p in (.01, 1.) for r in (1 / input_dim, .1) for h in (0., 1e-4)]
        self.filter = SparsePosterior(input_dim, self.configs, device)
        self.log_prior = torch.tensor([.5, *([.5 / 8] * 8)], device=device, dtype=torch.float32).log()
        self.log_weights = self.log_prior.expand(4, -1).clone()
        self.count = torch.zeros((), device=device, dtype=torch.float32)
        self.residual_sum = torch.zeros(9, device=device, dtype=torch.float32)
        self.target_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.noise = torch.ones(9, device=device, dtype=torch.float32)
        self.energy = torch.ones((), device=device, dtype=torch.float32)
        self._null_mean = torch.zeros(1, device=device, dtype=torch.float32)
        self._log_keep = math.log1p(-hazard) if hazard < 1 else -math.inf
        self._log_hazard = math.log(hazard) if hazard else -math.inf

    def state_tensors(self) -> list[torch.Tensor]:
        """All mutable state, in stable order; restore in place with copy_."""
        return [*self.filter.state_tensors(), self.log_weights, self.count,
                self.residual_sum, self.target_sum, self.noise, self.energy]

    def _prior_log_weights(self):
        prior = self.log_weights - self.log_weights.logsumexp(-1, keepdim=True)
        if self.hazard:
            switched = torch.logaddexp(prior[1::2] + self._log_keep, self.log_prior + self._log_hazard)
            prior = torch.stack((prior[0], switched[0], prior[2], switched[1]))
        return prior

    @torch.no_grad()
    def aggregation_weights(self) -> torch.Tensor:
        """Next-observation weights [5,9], including the switching transition."""
        return torch.cat((self._prior_log_weights().exp(), self.log_prior.exp().unsqueeze(0)), 0)

    @torch.no_grad()
    def mean_weights(self) -> torch.Tensor:
        """Next-prior effective coefficients [14,D], including coordinate resets."""
        sparse = self.filter.mean_weights()
        experts = torch.cat((torch.zeros_like(sparse[:1]), sparse), 0)
        return torch.cat((self.aggregation_weights() @ experts, experts), 0)

    @torch.no_grad()
    def update(self, x, y) -> torch.Tensor:
        """Return all fourteen pre-label forecasts, then retain conditioned state."""
        prior_log = self._prior_log_weights()
        weights = torch.cat((prior_log.exp(), self.log_prior.exp().unsqueeze(0)), 0)
        # The filter returns pre-label moments while conditioning its own state.
        # R is the previous conjugate scale mean, not the current squared error.
        sparse_mean, sparse_variance = self.filter.update(x, y, self.noise[1:])
        mean = torch.cat((self._null_mean, sparse_mean))
        variance = torch.cat((self.noise[:1], sparse_variance))
        prediction = torch.cat((weights @ mean, mean))
        residual_squared = (y - mean).square()
        likelihood = -.5 * (math.log(2 * math.pi) + variance.log() + residual_squared / variance)
        # The omitted -y^2/(2*S) term is identical across experts. S is causal
        # and shared: no expert's predictive variance can affect this score.
        risk = (2 * y * mean - mean.square()) / (2 * self.energy)
        evidence = torch.stack((likelihood, likelihood, risk, risk))
        posterior = prior_log + evidence
        self.log_weights.copy_(posterior - posterior.logsumexp(-1, keepdim=True))
        self.count.mul_(1 - self.noise_rate).add_(1)
        self.residual_sum.mul_(1 - self.noise_rate).add_(residual_squared)
        self.target_sum.mul_(1 - self.noise_rate).add_(y.square())
        denominator = 2 + self.count
        self.noise.copy_((2 + self.residual_sum) / denominator)
        self.energy.copy_((2 + self.target_sum) / denominator)
        return prediction

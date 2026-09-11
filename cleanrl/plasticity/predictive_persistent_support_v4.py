"""Fixed-share aggregation of static-support, deterministically restarting experts.

One cohort keeps all support evidence indefinitely. Each configured period adds
two phase-staggered cohorts whose entire spike-and-slab posterior is reset on a
known schedule, never via independent per-coordinate forgetting. These are not
inferred changepoints: schedule boundaries can cause transients, and there is no
changepoint-optimality or Bayesian-global guarantee. The backend remains the
factorized assumed-density SparsePosterior approximation, not joint Bayes.

Retiring a cohort discards its aggregation evidence before fixed share assigns
its newborn replacement fresh prior mass. The primary birth-only row shares
only at scheduled births, preserving cumulative evidence between boundaries.
The switching control shares every observation; both train the same experts.
Gaussian integrated predictive scores use pre-label moments and causal residual
scales, with the retained IG(2, 1) prior from v3. Residual scales include model
error, not just observation noise.
No clean targets, support labels, or oracle noise enter learning. Callers own
compilation and CUDA graph capture; inputs must be finite FP32 CUDA tensors.
Only diagnostics() synchronizes with the host, at evaluation boundaries.
"""

import math

import torch

from cleanrl.plasticity.predictive_structure_filter_v1 import SparsePosterior


class PersistentSupport:
    """Birth-only and switching aggregates, fixed-prior mixture, null, cohorts."""

    def __init__(self, input_dim, device, periods=(8192, 16384, 32768, 65536),
                 share_rate=1e-4, noise_rate=.001):
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError('input_dim must be a positive integer')
        periods = tuple(periods)
        if any(isinstance(p, bool) or not isinstance(p, int) or p <= 0 for p in periods):
            raise ValueError('periods must contain positive integers')
        share_rate, noise_rate = float(share_rate), float(noise_rate)
        if not math.isfinite(share_rate) or not 0 < share_rate < 1:
            raise ValueError('share_rate must be finite and lie in (0, 1)')
        if not math.isfinite(noise_rate) or not 0 <= noise_rate < 1:
            raise ValueError('noise_rate must be finite and lie in [0, 1)')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('PersistentSupport requires CUDA')
        self.share_rate, self.noise_rate = share_rate, noise_rate
        schedules = [(0, 0), *((p, offset) for p in periods for offset in (0, p // 2))]
        self.configs = [dict(period=p, offset=offset, prior=1., inclusion=1 / input_dim, hazard=0.)
                        for p, offset in schedules]
        self.filter = SparsePosterior(input_dim, self.configs, device)
        cohorts = len(schedules)
        experts = cohorts + 1
        self.output_names = ('likelihood_birth_only', 'likelihood_switching', 'prior_mixture', 'null',
                             *(f'cohort_{i}' for i in range(cohorts)))
        self.periods = torch.tensor([p for p, _ in schedules], device=device, dtype=torch.int64)
        self.offsets = torch.tensor([offset for _, offset in schedules], device=device, dtype=torch.int64)
        # The persistent cohort never resets; its modulo denominator is still safe.
        self._safe_periods = torch.tensor([p or 1 for p, _ in schedules], device=device, dtype=torch.int64)
        self.log_prior = torch.tensor([.5, *([.5 / cohorts] * cohorts)], device=device, dtype=torch.float32).log()
        self.log_weights = self.log_prior.expand(2, -1).clone()
        self.count = torch.zeros(experts, device=device, dtype=torch.float32)
        self.residual_sum = torch.zeros_like(self.count)
        self.noise = torch.ones_like(self.count)
        self.observations = torch.zeros((), device=device, dtype=torch.int64)
        self._null_mean = torch.zeros(1, device=device, dtype=torch.float32)
        self._null_restart = torch.zeros(1, device=device, dtype=torch.bool)
        self._prior_log_odds = -math.log(input_dim - 1) if input_dim > 1 else math.inf
        self._always_share = torch.ones((), device=device, dtype=torch.bool)

    @property
    def energy(self):
        """The null expert's residual scale is exactly causal target energy."""
        return self.noise[0]

    def state_tensors(self) -> list[torch.Tensor]:
        """Every mutable array, including schedule time; restore via copy_."""
        return [*self.filter.state_tensors(), self.log_weights, self.count,
                self.residual_sum, self.noise, self.observations]

    def _restart_due(self):
        t = self.observations
        return ((self.periods > 0) & (t > 0) & (t >= self.offsets)
                & ((t - self.offsets).remainder(self._safe_periods) == 0))

    def _prior_log_weights(self, due):
        retired = torch.cat((self._null_restart, due))
        surviving = self.log_weights.masked_fill(retired, -math.inf)
        surviving = surviving - surviving.logsumexp(-1, keepdim=True)
        gamma = self.share_rate * torch.stack((due.any(), self._always_share)).to(self.log_weights.dtype)
        # A zero birth-only rate gives log(gamma)=-inf, exactly no refresh.
        return torch.logaddexp(surviving + torch.log1p(-gamma).unsqueeze(-1),
                               self.log_prior + gamma.log().unsqueeze(-1))

    @torch.no_grad()
    def aggregation_weights(self) -> torch.Tensor:
        """Pure next-forecast weights [3, experts], including retirement/share."""
        return torch.cat((self._prior_log_weights(self._restart_due()).exp(),
                          self.log_prior.exp().unsqueeze(0)), 0)

    @torch.no_grad()
    def mean_weights(self) -> torch.Tensor:
        """Pure next-forecast coefficients [outputs, D], including scheduled resets."""
        due = self._restart_due()
        sparse = self.filter.mean_weights().masked_fill(due.unsqueeze(-1), 0.)
        experts = torch.cat((torch.zeros_like(sparse[:1]), sparse), 0)
        weights = torch.cat((self._prior_log_weights(due).exp(), self.log_prior.exp().unsqueeze(0)), 0)
        return torch.cat((weights @ experts, experts), 0)

    @torch.no_grad()
    def update(self, x, y) -> torch.Tensor:
        """Reset due cohorts, forecast without this label, then condition state."""
        due = self._restart_due()
        prior_log = self._prior_log_weights(due)
        column = due.unsqueeze(-1)
        self.filter.log_odds.copy_(torch.where(column, self._prior_log_odds, self.filter.log_odds))
        self.filter.slab_mean.masked_fill_(column, 0.)
        self.filter.slab_variance.masked_fill_(column, 1.)
        retired = torch.cat((self._null_restart, due))
        self.count.masked_fill_(retired, 0.)
        self.residual_sum.masked_fill_(retired, 0.)
        self.noise.masked_fill_(retired, 1.)

        # The static backend returns pre-label integrated moments while storing
        # its conditioned posterior. All scales here precede the current label.
        sparse_mean, sparse_variance = self.filter.update(x, y, self.noise[1:])
        mean = torch.cat((self._null_mean, sparse_mean))
        variance = torch.cat((self.noise[:1], sparse_variance))
        weights = torch.cat((prior_log.exp(), self.log_prior.exp().unsqueeze(0)), 0)
        prediction = torch.cat((weights @ mean, mean))
        residual_squared = (y - mean).square()
        likelihood = -.5 * (math.log(2 * math.pi) + variance.log() + residual_squared / variance)
        posterior = prior_log + likelihood
        self.log_weights.copy_(posterior - posterior.logsumexp(-1, keepdim=True))
        self.count.mul_(1 - self.noise_rate).add_(1)
        self.residual_sum.mul_(1 - self.noise_rate).add_(residual_squared)
        self.noise.copy_((2 + self.residual_sum) / (2 + self.count))
        self.observations.add_(1)
        return prediction

    @torch.no_grad()
    def diagnostics(self):
        """JSON-safe next-forecast state; host transfers only at evaluation boundaries.

        Ages count labels retained at the next forecast (zero when a reset is
        due). Coefficient diagnostics likewise show the next forecast's prior,
        not the retiring cohort's posterior still stored before update().
        """
        due = self._restart_due()
        t = self.observations
        last = self.offsets + ((t - self.offsets) // self._safe_periods) * self.periods
        last = torch.where((self.periods > 0) & (t >= self.offsets), last, 0)
        inclusion = torch.where(due.unsqueeze(-1), self.configs[0]['inclusion'],
                                self.filter.log_odds[:, :2].sigmoid())
        slab_mean = self.filter.slab_mean[:, :2].masked_fill(due.unsqueeze(-1), 0.)
        return dict(observations=int(t.cpu()), cohort_ages=(t - last).cpu().tolist(),
                    next_restart=due.cpu().tolist(), inclusion=inclusion.cpu().tolist(),
                    slab_mean=slab_mean.cpu().tolist(),
                    effective_weights=(inclusion * slab_mean).cpu().tolist(),
                    aggregation_weights=self.aggregation_weights().cpu().tolist())

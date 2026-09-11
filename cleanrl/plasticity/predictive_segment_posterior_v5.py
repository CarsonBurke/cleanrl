"""Budgeted causal global-segment inference, not exact joint Bayes.

A global hazard splits continuing mass into null/new sparse segments before the
label. Static SparsePosterior states retain support within a segment. Gaussian
predictive evidence and past-only retained IG(2,1) residual scales follow v3;
residual scales include model error and are not oracle observation noise.
Posterior-mass truncation keeps B nonnull segments, then renormalizes with null.
The discarded probability is recorded, not redistributed as an evidence floor.
Within-segment factorization, Gaussian scale scoring, null-history pooling and
bounded history truncation are all explicit approximations. Biological analogy
is a hypothesis source only; this changes inference, not an optimizer.
"""

import math

import torch

from cleanrl.plasticity.predictive_structure_filter_v1 import SparsePosterior


class SegmentPosterior:
    """Fixed-shape CUDA filter; one output, with pure next-forecast coefficients."""

    output_names = ('segment_posterior',)

    def __init__(self, input_dim, device, budget=9, hazard=1e-4, noise_rate=.001):
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError('input_dim must be a positive integer')
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError('budget must be a positive integer')
        if not math.isfinite(hazard) or not 0 <= hazard <= 1:
            raise ValueError('hazard must lie in [0,1]')
        if not math.isfinite(noise_rate) or not 0 <= noise_rate < 1:
            raise ValueError('noise_rate must lie in [0,1)')
        self.budget, self.hazard, self.noise_rate = budget, hazard, noise_rate
        self.configs = [dict(prior=1., inclusion=1 / input_dim, hazard=0.) for _ in range(budget + 1)]
        self.filter = SparsePosterior(input_dim, self.configs, device)
        device = self.filter.slab_mean.device
        # Last filter slot is newborn workspace, reset before EVERY prediction.
        self._initial_odds = math.log(1 / input_dim) - math.log1p(-1 / input_dim) if input_dim > 1 else math.inf
        self._log_keep = math.log1p(-hazard) if hazard < 1 else -math.inf
        self._log_birth = math.log(.5 * hazard) if hazard else -math.inf
        self.log_weights = torch.full((budget + 1,), -math.inf, device=device)
        self.log_weights[:2] = math.log(.5)
        self.count = torch.zeros(budget + 2, device=device)
        self.residual_sum = torch.zeros_like(self.count)
        self.noise = torch.ones_like(self.count)
        self.birth = torch.zeros(budget + 1, device=device, dtype=torch.int64)
        self.observations = torch.zeros((), device=device, dtype=torch.int64)
        self.discarded_mass = torch.zeros((), device=device)
        self.cumulative_discarded_mass = torch.zeros((), device=device, dtype=torch.float64)
        self._zero = torch.zeros(1, device=device)
        self._birth_log = torch.tensor([self._log_birth], device=device)

    @property
    def energy(self):
        return self.noise[0]

    def state_tensors(self):
        return [*self.filter.state_tensors(), self.log_weights, self.count,
                self.residual_sum, self.noise, self.birth, self.observations,
                self.discarded_mass, self.cumulative_discarded_mass]

    def _prior_log_weights(self):
        normalized = self.log_weights - self.log_weights.logsumexp(0)
        null = torch.logaddexp(normalized[:1] + self._log_keep, self._birth_log)
        return torch.cat((null, normalized[1:] + self._log_keep, self._birth_log))

    @torch.no_grad()
    def aggregation_weights(self):
        """Null, B continuations, newborn: next-observation prior mass."""
        return self._prior_log_weights().exp()

    @torch.no_grad()
    def mean_weights(self):
        # Newborn/null means are zero, even when the workspace retains old data.
        coefficients = self.filter.mean_weights()[:self.budget]
        return (self.aggregation_weights()[1:-1] @ coefficients).unsqueeze(0)

    @torch.no_grad()
    def update(self, x, y):
        prior = self._prior_log_weights()
        self.filter.log_odds[-1].fill_(self._initial_odds)
        self.filter.slab_mean[-1].zero_()
        self.filter.slab_variance[-1].fill_(1.)
        self.count[-1].zero_()
        self.residual_sum[-1].zero_()
        self.noise[-1].fill_(1.)
        self.birth[-1].copy_(self.observations)
        # SparsePosterior returns causal moments while updating its own state.
        forecast, variance = self.filter.update(x, y, self.noise[1:])
        means = torch.cat((self._zero, forecast))
        variances = torch.cat((self.noise[:1], variance))
        prediction = (prior.exp() * means).sum().reshape(1)
        error = (y - means).square()
        scores = prior - .5 * (math.log(2 * math.pi) + variances.log() + error / variances)
        posterior = scores - scores.logsumexp(0)
        # Stable ordering gives older retained slots precedence on exact ties.
        order = torch.argsort(posterior[1:], descending=True, stable=True)
        keep = order[:self.budget]
        self.discarded_mass.copy_(posterior[1:].index_select(0, order[self.budget:]).exp().sum())
        self.cumulative_discarded_mass.add_(self.discarded_mass.double())
        retained = torch.cat((posterior[:1], posterior[1:].index_select(0, keep)))
        self.log_weights.copy_(retained - retained.logsumexp(0))
        self.count.mul_(1 - self.noise_rate).add_(1)
        self.residual_sum.mul_(1 - self.noise_rate).add_(error)
        self.noise.copy_((2 + self.residual_sum) / (2 + self.count))
        # Gather ALL segment sufficient statistics with the same posterior order.
        for tensor in self.filter.state_tensors():
            tensor[:self.budget].copy_(tensor.index_select(0, keep))
        for tensor in (self.count, self.residual_sum, self.noise):
            tensor[1:self.budget + 1].copy_(tensor[1:].index_select(0, keep))
        self.birth[:self.budget].copy_(self.birth.index_select(0, keep))
        self.observations.add_(1)
        return prediction

    @torch.no_grad()
    def trace(self):
        """Dense post-label diagnostics; slots are mass-ranked, not identities.

        Columns: retained mass, birth observation, IG count/Q/variance, inclusion
        at 0/1, conditional means at 0/1, effective coefficients at 0/1, summed
        inclusion, squared effective coefficient norm. Null is a separate row.
        """
        b = self.budget
        probability = self.filter.log_odds[:b].sigmoid()
        mean = self.filter.slab_mean[:b]
        effective = probability * mean
        ix = min(1, mean.shape[1] - 1)
        branches = torch.stack((self.log_weights[1:].exp(), self.birth[:b].float(),
                                self.count[1:b + 1], self.residual_sum[1:b + 1], self.noise[1:b + 1],
                                probability[:, 0], probability[:, ix], mean[:, 0], mean[:, ix],
                                effective[:, 0], effective[:, ix], probability.sum(-1),
                                effective.square().sum(-1)), -1)
        null = torch.stack((self.log_weights[0].exp(), self.observations.float() * 0,
                            self.count[0], self.residual_sum[0], self.noise[0],
                            *([self._zero[0]] * 8))).unsqueeze(0)
        return torch.cat((null, branches))

    @torch.no_grad()
    def diagnostics(self):
        return {'next_prior_mass': self.aggregation_weights().cpu().tolist(),
                'retained_branch_trace': self.trace().cpu().tolist(),
                'discarded_mass': float(self.discarded_mass),
                'cumulative_discarded_mass': float(self.cumulative_discarded_mass),
                'cumulative_discard_note': 'sum of local discarded probabilities, not a global error bound'}

"""Bayesian uncertainty over a fixed global-segment hazard, including no change.

Each child preserves the frozen v5 bounded-history approximation. Its evidence is
the Gaussian moment-mixture predictive density BEFORE posterior-mass truncation,
not an exact sparse posterior likelihood or a true market likelihood. A uniform
hyperprior is conditioned on these densities without fixed share or forgetting.
The brain analogy motivates a hypothesis only; it is not biological validation.
"""

import math

import torch

from cleanrl.plasticity.predictive_segment_posterior_v5 import SegmentPosterior


class SegmentEvidence(SegmentPosterior):
    """Frozen v5 transitions with the already-computed predictive density exposed."""

    def __init__(self, input_dim, device, budget=9, hazard=1e-4, noise_rate=.001):
        super().__init__(input_dim, device, budget, hazard, noise_rate)
        # No observation yet: zero log evidence is the multiplicative identity.
        self.predictive_log_prob = torch.zeros((), device=self.log_weights.device)

    def state_tensors(self):
        return [*super().state_tensors(), self.predictive_log_prob]

    @torch.no_grad()
    def update(self, x, y):
        # Keep the v5 kernel intact: only retain its one evidence normalizer.
        prior = self._prior_log_weights()
        self.filter.log_odds[-1].fill_(self._initial_odds)
        self.filter.slab_mean[-1].zero_()
        self.filter.slab_variance[-1].fill_(1.)
        self.count[-1].zero_()
        self.residual_sum[-1].zero_()
        self.noise[-1].fill_(1.)
        self.birth[-1].copy_(self.observations)
        forecast, variance = self.filter.update(x, y, self.noise[1:])
        means = torch.cat((self._zero, forecast))
        variances = torch.cat((self.noise[:1], variance))
        prediction = (prior.exp() * means).sum().reshape(1)
        error = (y - means).square()
        scores = prior - .5 * (math.log(2 * math.pi) + variances.log() + error / variances)
        self.predictive_log_prob.copy_(scores.logsumexp(0))
        posterior = scores - self.predictive_log_prob
        order = torch.argsort(posterior[1:], descending=True, stable=True)
        keep = order[:self.budget]
        self.discarded_mass.copy_(posterior[1:].index_select(0, order[self.budget:]).exp().sum())
        self.cumulative_discarded_mass.add_(self.discarded_mass.double())
        retained = torch.cat((posterior[:1], posterior[1:].index_select(0, keep)))
        self.log_weights.copy_(retained - retained.logsumexp(0))
        self.count.mul_(1 - self.noise_rate).add_(1)
        self.residual_sum.mul_(1 - self.noise_rate).add_(error)
        self.noise.copy_((2 + self.residual_sum) / (2 + self.count))
        for tensor in self.filter.state_tensors():
            tensor[:self.budget].copy_(tensor.index_select(0, keep))
        for tensor in (self.count, self.residual_sum, self.noise):
            tensor[1:self.budget + 1].copy_(tensor[1:].index_select(0, keep))
        self.birth[:self.budget].copy_(self.birth.index_select(0, keep))
        self.observations.add_(1)
        return prediction

    @torch.no_grad()
    def diagnostics(self):
        return {**super().diagnostics(), 'predictive_log_prob': float(self.predictive_log_prob)}


class HazardMixture:
    """Causal CUDA hazard model average plus each conditional child forecast."""

    output_names = ('hazard_mixture', 'hazard_0', 'hazard_1e-5', 'hazard_1e-4', 'hazard_1e-3')

    def __init__(self, input_dim, device, budget=9, hazards=(0., 1e-5, 1e-4, 1e-3), noise_rate=.001):
        self.hazards = tuple(hazards)
        if not self.hazards:
            raise ValueError('hazards must contain at least one hypothesis')
        self.segments = [SegmentEvidence(input_dim, device, budget, hazard, noise_rate)
                         for hazard in self.hazards]
        names = []
        for hazard in self.hazards:
            mantissa, exponent = format(hazard, '.15e').split('e')
            name = mantissa.rstrip('0').rstrip('.')
            names.append('hazard_' + (name if hazard == 0 else f'{name}e{int(exponent)}'))
        self.output_names = ('hazard_mixture', *names)
        device = self.segments[0].log_weights.device
        self.log_hazard_weights = torch.full((len(self.segments),), -math.log(len(self.segments)), device=device)
        self.predictions = torch.zeros(len(self.output_names), device=device)

    def state_tensors(self):
        return [*(tensor for segment in self.segments for tensor in segment.state_tensors()),
                self.log_hazard_weights, self.predictions]

    @torch.no_grad()
    def update(self, x, y):
        # Child updates return PRELABEL means; hyperweights still contain only
        # past evidence when these means are averaged for the current forecast.
        child_predictions = torch.cat([segment.update(x, y) for segment in self.segments])
        mixture = (self.log_hazard_weights.exp() * child_predictions).sum().reshape(1)
        self.predictions.copy_(torch.cat((mixture, child_predictions)))
        scores = self.log_hazard_weights + torch.stack([segment.predictive_log_prob for segment in self.segments])
        self.log_hazard_weights.copy_(scores - scores.logsumexp(0))
        return self.predictions

    @torch.no_grad()
    def mean_weights(self):
        child_means = torch.cat([segment.mean_weights() for segment in self.segments])
        mixture = (self.log_hazard_weights.exp() @ child_means).unsqueeze(0)
        return torch.cat((mixture, child_means))

    @torch.no_grad()
    def diagnostics(self):
        return {'hazards': list(self.hazards),
                'hazard_weights': self.log_hazard_weights.exp().cpu().tolist(),
                'predictive_log_prob': [float(segment.predictive_log_prob) for segment in self.segments],
                'discarded_mass': [float(segment.discarded_mass) for segment in self.segments],
                'children': [segment.diagnostics() for segment in self.segments],
                'evidence_note': 'Gaussian moment-mixture density before child truncation; approximate likelihood',
                'hyperprior_note': 'uniform hazard prior; sequential evidence updates without forgetting or fixed share'}

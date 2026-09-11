"""Exact singleton/null NIG inference, approximate compressed change histories.

The mutually exclusive support prior is specialized information, not a generic
optimizer. Bucket resampling preserves posterior mass, not the exact posterior
measure over histories. The frozen factorized v6 family hedges misspecification.
"""
import math

import torch

from cleanrl.plasticity.predictive_hazard_mixture_v6 import HazardMixture


HAZARDS = (0., 1e-5, 1e-4, 1e-3)


def bucket_resample(log_mass, birth, observations, uniforms, budget):
    """Fixed-shape categorical representatives and *unnormalized* bucket masses.

    Ages start at one after consuming the current label. Uniforms lie in [0,1).
    Empty slots select index zero but carry -inf log mass. No posterior tail is
    discarded. The caller must provision enough buckets for its full horizon.
    """
    age = observations + 1 - birth
    bucket = torch.floor(age.double().log() / math.log(1.5)).long()
    membership = bucket.unsqueeze(0) == torch.arange(budget, device=birth.device).unsqueeze(1)
    scores = torch.where(membership, log_mass.unsqueeze(0), -torch.inf)
    masses = scores.logsumexp(1)
    occupied = membership.logical_and(torch.isfinite(log_mass).unsqueeze(0)).any(1)
    # Empty buckets have no distribution; this identity avoids -inf - -inf,
    # without sanitizing NaN/Inf evidence in any occupied branch.
    denominator = torch.where(occupied, masses, torch.zeros_like(masses))
    probabilities = (scores - denominator.unsqueeze(1)).exp()
    cumulative = probabilities.cumsum(1)
    # Select by inverse CDF; last positive entry handles roundoff of sum(p).
    positive = membership & (probabilities > 0)
    indices = torch.arange(len(birth), device=birth.device).expand(budget, -1)
    last = torch.where(positive, indices, torch.zeros_like(indices)).amax(1)
    crosses = (cumulative > uniforms.unsqueeze(1)) & positive
    first = torch.where(crosses, indices, len(birth)).amin(1)
    chosen = torch.where(first < len(birth), first, last)
    return chosen, masses


class SingletonSegment:
    """Exact segment posterior under p(null)=1/2, p(k)=1/(2D).

    sigma²~IG(2,1), theta_k|sigma²~N(0,sigma²). FP64 sufficient
    statistics are label-updated only after predictive scoring. The top-mass
    variant intentionally discards history mass and reports that mass.
    """
    trace_columns = ('retained_mass', 'birth_observation', 'count', 'sum_y2',
                     'null_probability', 'support_0', 'support_1', 'mean_0', 'mean_1',
                     'effective_0', 'effective_1', 'support_mass', 'effective_squared_norm')

    def __init__(self, input_dim, device, budget=32, hazard=1e-4, compression='bucket', log_support_prior=None):
        if input_dim < 1 or budget < 1 or not 0 <= hazard < 1 or compression not in ('bucket', 'top_mass'):
            raise ValueError('invalid singleton segment contract')
        self.input_dim, self.budget, self.hazard, self.compression = input_dim, budget, hazard, compression
        kw = dict(device=device, dtype=torch.float64)
        self.n = torch.zeros(budget + 1, **kw)
        self.sum_y2 = torch.zeros_like(self.n)
        self.sum_x2 = torch.zeros((budget + 1, input_dim), **kw)
        self.sum_xy = torch.zeros_like(self.sum_x2)
        self.birth = torch.zeros(budget + 1, device=device, dtype=torch.int64)
        self.observations = torch.zeros((), device=device, dtype=torch.int64)
        self.log_weights = torch.full((budget,), -torch.inf, **kw)
        self.log_weights[0] = 0.
        self.predictive_log_prob = torch.zeros((), **kw)
        self.discarded_mass = torch.zeros((), **kw)
        self.cumulative_discarded_mass = torch.zeros((), **kw)
        self.mass_error = torch.zeros((), **kw)
        if log_support_prior is None:
            log_support_prior = torch.full((input_dim + 1,), -math.log(2 * input_dim), **kw)
            log_support_prior[0] = -math.log(2)
        self.log_support_prior = log_support_prior

    def state_tensors(self):
        return [self.n, self.sum_y2, self.sum_x2, self.sum_xy, self.birth, self.observations,
                self.log_weights, self.predictive_log_prob, self.discarded_mass,
                self.cumulative_discarded_mass, self.mass_error]

    def posterior(self):
        precision = 1 + self.sum_x2
        mean = self.sum_xy / precision
        alpha = 2 + self.n / 2
        beta_null = 1 + self.sum_y2 / 2
        reduction = self.sum_xy * mean / 2
        beta = torch.cat((beta_null.unsqueeze(1), beta_null.unsqueeze(1) - reduction), 1)
        log_bf = -.5 * precision.log() - alpha.unsqueeze(1) * torch.log1p(-reduction / beta_null.unsqueeze(1))
        score = torch.cat((torch.zeros_like(alpha).unsqueeze(1), log_bf), 1) + self.log_support_prior
        return score - score.logsumexp(1, keepdim=True), mean, precision, alpha, beta

    def _prior_log_weights(self):
        birth_mass = math.log(self.hazard) if self.hazard else -math.inf
        return torch.cat((self.log_weights + math.log1p(-self.hazard), self.log_weights.new_full((1,), birth_mass)))

    def predictive(self, x, y):
        log_support, mean, precision, alpha, beta = self.posterior()
        location = torch.cat((torch.zeros_like(alpha).unsqueeze(1), mean * x), 1)
        inflation = torch.cat((torch.ones_like(alpha).unsqueeze(1), 1 + x.square() / precision), 1)
        scaled_beta = beta * inflation
        log_terms = (log_support - .5 * scaled_beta.log()
                     - (alpha + .5).unsqueeze(1) * torch.log1p((y - location).square() / (2 * scaled_beta)))
        # Every support hypothesis shares alpha; keep its expensive normalizer
        # outside the support reduction and cancel df*scale² analytically.
        common = torch.lgamma(alpha + .5) - torch.lgamma(alpha) - .5 * math.log(2 * math.pi)
        return (log_support.exp() * location).sum(1), log_terms.logsumexp(1) + common

    @torch.no_grad()
    def mean_weights(self):
        # Newborn is a mathematical prior with zero mean, regardless of scratch.
        support, mean, _, _, _ = self.posterior()
        effective = support[:self.budget, 1:].exp() * mean[:self.budget]
        return ((1 - self.hazard) * self.log_weights.exp() @ effective).unsqueeze(0)

    @torch.no_grad()
    def update(self, x, y, uniforms=None):
        x, y = x.double(), y.double()
        for tensor in (self.n, self.sum_y2, self.sum_x2, self.sum_xy):
            tensor[-1].zero_()
        self.birth[-1].copy_(self.observations)
        means, densities = self.predictive(x, y)
        prior = self._prior_log_weights()
        forecast = (prior.exp() * means).sum().reshape(1)
        score = prior + densities
        self.predictive_log_prob.copy_(score.logsumexp(0))
        posterior = score - self.predictive_log_prob
        self.n.add_(1)
        self.sum_y2.add_(y.square())
        self.sum_x2.add_(x.square())
        self.sum_xy.add_(x * y)
        if self.compression == 'bucket':
            keep, retained = bucket_resample(posterior, self.birth, self.observations, uniforms, self.budget)
            self.discarded_mass.zero_()
        else:
            order = torch.argsort(posterior, descending=True, stable=True)
            keep = order[:self.budget]
            self.discarded_mass.copy_(posterior.index_select(0, order[self.budget:]).exp().sum())
            retained = posterior.index_select(0, keep)
            retained = retained - retained.logsumexp(0)
        self.mass_error.copy_(retained.exp().sum() - 1)
        self.cumulative_discarded_mass.add_(self.discarded_mass)
        self.log_weights.copy_(retained)
        for tensor in (self.n, self.sum_y2, self.sum_x2, self.sum_xy, self.birth):
            tensor[:self.budget].copy_(tensor.index_select(0, keep))
        self.observations.add_(1)
        return forecast

    @torch.no_grad()
    def trace(self):
        support, mean, _, _, _ = self.posterior()
        p = support[:self.budget].exp()
        m = mean[:self.budget]
        effective = p[:, 1:] * m
        second = min(1, self.input_dim - 1)
        return torch.stack((self.log_weights.exp(), self.birth[:self.budget].double(), self.n[:self.budget],
                            self.sum_y2[:self.budget], p[:, 0], p[:, 1], p[:, second + 1], m[:, 0], m[:, second],
                            effective[:, 0], effective[:, second], p[:, 1:].sum(1), effective.square().sum(1)), 1)

    def diagnostics(self):
        return {'hazard': self.hazard, 'compression': self.compression,
                'observations': int(self.observations), 'mass_error': float(self.mass_error),
                'discarded_mass': float(self.discarded_mass),
                'cumulative_discarded_mass': float(self.cumulative_discarded_mass),
                'predictive_log_prob': float(self.predictive_log_prob),
                'occupied_branches': int(torch.isfinite(self.log_weights).sum())}


class SingletonHazardMixture:
    output_names = ('singleton_hazard_mixture', 'singleton_hazard_0', 'singleton_hazard_1e-5',
                    'singleton_hazard_1e-4', 'singleton_hazard_1e-3')

    def __init__(self, input_dim, device, uniforms):
        if uniforms.ndim != 3 or uniforms.shape[1:] != (4, 32) or uniforms.dtype != torch.float64:
            raise ValueError('resampling tape must be FP64 [horizon,4,32]')
        if uniforms.device != torch.device(device):
            # torch.device("cuda") has no explicit index; compare concrete tensors.
            if uniforms.device != torch.empty(0, device=device).device:
                raise ValueError('resampling tape must be on model device')
        if len(uniforms) > 1.5 ** 32:
            raise ValueError('32 log-age buckets cannot cover tape horizon')
        if len(uniforms) < 1 or not bool((torch.isfinite(uniforms) & (uniforms >= 0) & (uniforms < 1)).all()):
            raise ValueError('resampling tape must contain finite uniforms in [0,1)')
        self.uniforms = uniforms
        first = SingletonSegment(input_dim, device, hazard=HAZARDS[0])
        self.segments = [first, *(SingletonSegment(input_dim, device, hazard=h,
                                                  log_support_prior=first.log_support_prior) for h in HAZARDS[1:])]
        self.log_hazard_weights = torch.full((4,), -math.log(4), device=device, dtype=torch.float64)
        self.index = torch.zeros((), device=device, dtype=torch.int64)
        self.uniforms_consumed = torch.zeros_like(self.index)
        self.predictive_log_prob = torch.zeros((), device=device, dtype=torch.float64)

    def state_tensors(self):
        return [*(t for segment in self.segments for t in segment.state_tensors()),
                self.log_hazard_weights, self.index, self.uniforms_consumed, self.predictive_log_prob]

    @torch.no_grad()
    def update(self, x, y):
        uniforms = self.uniforms.index_select(0, self.index.reshape(1)).squeeze(0)
        predictions = torch.cat([child.update(x, y, uniforms[i]) for i, child in enumerate(self.segments)])
        forecast = (self.log_hazard_weights.exp() * predictions).sum().reshape(1)
        score = self.log_hazard_weights + torch.stack([s.predictive_log_prob for s in self.segments])
        self.predictive_log_prob.copy_(score.logsumexp(0))
        self.log_hazard_weights.copy_(score - self.predictive_log_prob)
        self.index.add_(1)
        self.uniforms_consumed.add_(128)
        return torch.cat((forecast, predictions))

    def mean_weights(self):
        means = torch.cat([s.mean_weights() for s in self.segments])
        return torch.cat(((self.log_hazard_weights.exp() @ means).unsqueeze(0), means))

    def diagnostics(self):
        return {'hazard_weights': self.log_hazard_weights.exp().cpu().tolist(),
                'uniforms_consumed': int(self.uniforms_consumed),
                'children': [s.diagnostics() for s in self.segments]}


class StructuralMixture:
    output_names = ('structural_mixture', *SingletonHazardMixture.output_names, 'singleton_top_mass_1e-4',
                    *('v6_' + name for name in HazardMixture.output_names))

    def __init__(self, input_dim, device, uniforms):
        self.singleton = SingletonHazardMixture(input_dim, device, uniforms)
        self.top_mass = SingletonSegment(input_dim, device, compression='top_mass',
                                         log_support_prior=self.singleton.segments[0].log_support_prior)
        self.factorized = HazardMixture(input_dim, device, budget=32)
        self.log_structure_weights = torch.full((2,), -math.log(2), device=device, dtype=torch.float64)
        self.structure_log_prob = torch.zeros(2, device=device, dtype=torch.float64)
        self.predictive_log_prob = torch.zeros((), device=device, dtype=torch.float64)
        self.segments = [*self.singleton.segments, self.top_mass, *self.factorized.segments]

    def state_tensors(self):
        return [*(t for model in (self.singleton, self.top_mass, self.factorized) for t in model.state_tensors()),
                self.log_structure_weights, self.structure_log_prob, self.predictive_log_prob]

    @torch.no_grad()
    def update(self, x, y):
        # v6 exposes conditional densities, but not its pre-hyperupdate mixture
        # evidence. Snapshot just four prior weights before its frozen update.
        old_factorized = self.factorized.log_hazard_weights.double().clone()
        singleton = self.singleton.update(x, y)
        top = self.top_mass.update(x, y)
        factorized = self.factorized.update(x, y).double()
        prediction = (self.log_structure_weights.exp() * torch.stack((singleton[0], factorized[0]))).sum().reshape(1)
        factorized_density = (old_factorized + torch.stack([s.predictive_log_prob for s in self.factorized.segments]).double()).logsumexp(0)
        self.structure_log_prob.copy_(torch.stack((self.singleton.predictive_log_prob, factorized_density)))
        scores = self.log_structure_weights + self.structure_log_prob
        self.predictive_log_prob.copy_(scores.logsumexp(0))
        self.log_structure_weights.copy_(scores - self.predictive_log_prob)
        return torch.cat((prediction, singleton, top, factorized))

    def mean_weights(self):
        singleton = self.singleton.mean_weights()
        factorized = self.factorized.mean_weights().double()
        means = torch.stack((singleton[0], factorized[0]))
        return torch.cat(((self.log_structure_weights.exp() @ means).unsqueeze(0), singleton,
                          self.top_mass.mean_weights(), factorized))

    def diagnostics(self):
        return {'structure_weights': self.log_structure_weights.exp().cpu().tolist(),
                'structure_log_prob': self.structure_log_prob.cpu().tolist(),
                'singleton': self.singleton.diagnostics(), 'top_mass': self.top_mass.diagnostics(),
                'factorized': self.factorized.diagnostics()}

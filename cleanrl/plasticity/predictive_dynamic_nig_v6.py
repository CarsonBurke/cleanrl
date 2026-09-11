"""Conditional-scale dynamic regression with calibrated Student-t mixtures.

For each prior tau, initially w | sigma^2 ~ N(0, tau I sigma^2) and
sigma^2 ~ IG(2, 1). Before observation t, the specified transition is
w_t | w_(t-1), sigma^2 ~ N(w_(t-1), (delta^-1 - 1) P_(t-1) sigma^2).
Here P_(t-1) is the posterior geometry determined by P_0 and PAST features;
it does not depend on the current label. This is a data-dependent conditional
process covariance, NOT exact inference for a fixed process covariance Q.
Static experts use delta=1; dynamic experts use delta=2**(-1/halflife).

Prediction preserves the coefficient mean and inflates Pminus=P/delta.
With u=Pminus x, s=1+x'u, r=y-x'm, conditioning gives P=Pminus-uu'/s,
m=m+ur/s, alpha=alpha+1/2, beta=beta+r^2/(2s). The proper predictive law is
Student-t(2 alpha, x'm, scale^2=beta*s/alpha), evaluated BEFORE conditioning.
The mean and covariance equal exponentially weighted normal equations with
initial precision delta^t/tau after t observations: the initial regularizer
fades too. Beta instead accumulates innovation evidence for one global unknown
sigma^2; it is NOT an exponentially discounted residual sum. These identities
specify a conditional model, not an optimality claim or evidence of stock skill.

All experts share trajectories across three independent Bayesian mixtures.
Adaptive prior mass is 1/2 null, 1/4 static, 1/4 dynamic; each family is uniform.
Static-only and dynamic-only controls each assign 1/2 to their family and 1/2
to the null. The zero-mean null also learns sigma^2, with beta += y^2/2.
Forecasts use previous posterior probabilities, never current-label evidence.
There are no resets, mean decay, clamps, gates, or uncalibrated fallbacks.

CUDA only. Inputs are finite FP32 x[D], y[] on the model device. Callers disable
TF32 and own compilation/capture. Matrices and coefficients are FP32; scalar
scale/evidence arithmetic is FP64. Floating-point covariance drift must be
reported, not silently repaired. update has no host synchronization.
"""

import math

import torch

from cleanrl.plasticity.predictive_correlated_nig_v5 import student_t_log_prob


class DynamicNIG:
    """Shared static/dynamic experts and three causally weighted Bayes controls."""

    def __init__(self, input_dim, device, priors=(1e-7, 1e-6, 1e-5, 1e-4, 1e-3),
                 halflives=(65536, 262144)):
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError('input_dim must be a positive integer')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('DynamicNIG requires CUDA')
        priors = tuple(float(prior) for prior in priors)
        halflives = tuple(float(halflife) for halflife in halflives)
        for name, values in (('priors', priors), ('halflives', halflives)):
            if not values or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f'{name} must be nonempty, finite, and positive')
            if len({f'{value:g}' for value in values}) != len(values):
                raise ValueError(f'{name} must have distinct output names')
        self.input_dim, self.priors, self.halflives = input_dim, priors, halflives
        self.configs = [{'prior': prior, 'halflife': halflife}
                        for halflife in (0, *halflives) for prior in priors]
        self.output_names = ('adaptive_mixture', 'static_mixture', 'dynamic_mixture', 'zero',
                             *(f'static_{prior:g}' for prior in priors),
                             *(f'discount_{halflife:g}_{prior:g}'
                               for halflife in halflives for prior in priors))
        static_count = len(priors)
        dynamic_count = static_count * len(halflives)
        k = static_count + dynamic_count
        self.mean = torch.zeros((k, input_dim), device=device, dtype=torch.float32)
        self.cov = torch.zeros((k, input_dim, input_dim), device=device, dtype=torch.float32)
        prior_tensor = torch.tensor([cfg['prior'] for cfg in self.configs],
                                    device=device, dtype=torch.float32)
        self.cov.diagonal(dim1=-2, dim2=-1).copy_(prior_tensor[:, None])
        # expm1 retains tiny process fractions for long half-lives. Adding the
        # increment avoids rounding 1+fraction before multiplying each matrix.
        process_fraction = [0. if cfg['halflife'] == 0 else
                            math.expm1(math.log(2) / cfg['halflife']) for cfg in self.configs]
        self._process_fraction = torch.tensor(process_fraction, device=device, dtype=torch.float32)
        self.alpha = torch.tensor(2., device=device, dtype=torch.float64)
        self.beta = torch.ones(k + 1, device=device, dtype=torch.float64)
        self.log_all = torch.tensor([*([.25 / static_count] * static_count),
                                     *([.25 / dynamic_count] * dynamic_count), .5],
                                    device=device, dtype=torch.float64).log()
        self.log_static = torch.tensor([*([.5 / static_count] * static_count), .5],
                                       device=device, dtype=torch.float64).log()
        self.log_dynamic = torch.tensor([*([.5 / dynamic_count] * dynamic_count), .5],
                                        device=device, dtype=torch.float64).log()
        self.observations = torch.zeros((), device=device, dtype=torch.int64)
        self._zero = torch.zeros(1, device=device, dtype=torch.float32)

    def state_tensors(self) -> list[torch.Tensor]:
        """All mutable state, in stable order; restore in place for capture replay."""
        return [self.mean, self.cov, self.alpha, self.beta, self.log_all,
                self.log_static, self.log_dynamic, self.observations]

    @property
    def noise(self):
        """Posterior mean sigma^2 for all experts then null, excluding leverage."""
        return self.beta / (self.alpha - 1)

    @torch.no_grad()
    def diagnostics(self):
        """Host-only scales/probabilities; covariance PSD checks belong to caller."""
        return {'alpha': float(self.alpha.cpu()), 'beta': self.beta.cpu().tolist(),
                'noise': self.noise.cpu().tolist(),
                'noise_std': self.noise.sqrt().cpu().tolist(),
                'student_t_scale_squared': (self.beta / self.alpha).cpu().tolist(),
                'student_t_scale': (self.beta / self.alpha).sqrt().cpu().tolist(),
                'adaptive_probabilities': self.log_all.exp().cpu().tolist(),
                'static_probabilities': self.log_static.exp().cpu().tolist(),
                'dynamic_probabilities': self.log_dynamic.exp().cpu().tolist()}

    @torch.no_grad()
    def update(self, x, y) -> torch.Tensor:
        """Return owned FP32 pre-label forecasts, then transition and condition."""
        static_count = len(self.priors)
        # Mean-preserving transition. Geometry depends only on past features.
        self.cov.add_(self.cov * self._process_fraction[:, None, None])
        u = torch.matmul(self.cov, x)
        # Keep leverage below FP32 spacing at one in the proper evidence.
        s = (u * x).sum(-1).double() + 1
        mu = (self.mean * x).sum(-1)
        mu64 = mu.double()
        adaptive = (self.log_all[:-1].exp() * mu64).sum().float().reshape(1)
        static = (self.log_static[:-1].exp() * mu64[:static_count]).sum().float().reshape(1)
        dynamic = (self.log_dynamic[:-1].exp() * mu64[static_count:]).sum().float().reshape(1)
        prediction = torch.cat((adaptive, static, dynamic, self._zero, mu))
        locations = torch.cat((mu64, self._zero.double()))
        leverage = torch.cat((s, torch.ones_like(self.alpha).reshape(1)))
        residual = y.double() - locations
        residual_squared = residual.square()
        scores = student_t_log_prob(self.alpha, self.beta, residual_squared, leverage)
        posterior_all = self.log_all + scores
        posterior_static = self.log_static + torch.cat((scores[:static_count], scores[-1:]))
        posterior_dynamic = self.log_dynamic + scores[static_count:]
        self.log_all.copy_(posterior_all - posterior_all.logsumexp(-1))
        self.log_static.copy_(posterior_static - posterior_static.logsumexp(-1))
        self.log_dynamic.copy_(posterior_dynamic - posterior_dynamic.logsumexp(-1))
        rank = u * s.rsqrt().float()[:, None]
        self.cov.sub_(rank[:, :, None] * rank[:, None, :])
        self.mean.add_(u * (residual[:-1] / s).float()[:, None])
        self.beta.add_(.5 * residual_squared / leverage)
        self.alpha.add_(.5)
        self.observations.add_(1)
        return prediction

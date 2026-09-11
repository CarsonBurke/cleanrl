"""Dense conjugate regression and its diagonal assumed-density control.

Each fixed prior tau defines y | w,sigma^2,x ~ N(x'w,sigma^2),
w | sigma^2 ~ N(0,tau I sigma^2), and sigma^2 ~ IG(alpha=2,beta=1),
where IG has density proportional to v^(-alpha-1) exp(-beta/v).
After a prefix, w | sigma^2,data ~ N(m,P sigma^2). For the next x,
u=P x, s=1+x'u and r=y-x'm. Integrating both w and sigma^2 gives a
Student-t prediction with df=2 alpha, location=x'm and scale^2=beta*s/alpha.
Conditioning gives P'=P-uu'/s, m'=m+ur/s, beta'=beta+r^2/(2s),
alpha'=alpha+1/2. These are exact conjugate identities; dense covariance and
coefficient arithmetic here is FP32, not an exact-arithmetic implementation.

The diagonal control starts from the same prior, applies the same conditioning
identities to diag(p), then discards the new off-diagonals. Its next conditional
Gaussian has the retained diagonal covariance. This is an assumed-density
projection, NOT the exact posterior of dense regression. Each trajectory keeps
its own coefficients, covariance and inverse-Gamma beta.

A static Bayesian mixture uses dense experts only, with total prior mass 1/2
split equally, and a zero-mean unknown-variance null with mass 1/2. Its forecast
uses prefix posterior probabilities BEFORE the current Student-t score. There
is no forgetting, hazard, reset, clipping, variance floor, or robust influence.
The null is IG(2,1) updated by beta += y^2/2, irrespective of input features.

Scale arithmetic and proper Student-t evidence use FP64 to preserve weak
likelihood differences; matrix/coefficient arithmetic stays FP32. In particular
1 is added to the FP32 leverage contraction only AFTER promotion to FP64.
Long streams may accumulate covariance roundoff; callers must diagnose loss of
positive definiteness rather than silently repair it. No universal optimality
or real-stock correctness follows from conjugacy or the diagonal comparison.

CUDA only. Callers supply finite FP32 x[D], y[] on the model device, disable
TF32, and own compilation/capture. update never synchronizes with the host.
"""

import math

import torch


def student_t_log_prob(alpha, beta, residual_squared, leverage):
    """Proper log density for the NIG predictive (all arguments FP64 tensors).

    leverage is s=1+x'Px, not x'Px. The beta parameter is an IG rate,
    not the predictive variance; retaining the gamma normalization matters.
    """
    return (torch.lgamma(alpha + .5) - torch.lgamma(alpha)
            - .5 * (math.log(2 * math.pi) + beta.log() + leverage.log())
            - (alpha + .5) * torch.log1p(residual_squared / (2 * beta * leverage)))


class CorrelatedNIG:
    """Independent dense/diagonal priors, dense Bayes mixture, and zero forecast."""

    def __init__(self, input_dim, device, priors=(1e-7, 1e-6, 1e-5, 1e-4, 1e-3)):
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError('input_dim must be a positive integer')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('CorrelatedNIG requires CUDA')
        priors = tuple(float(prior) for prior in priors)
        if not priors or any(not math.isfinite(prior) or prior <= 0 for prior in priors):
            raise ValueError('priors must be nonempty, finite, and positive')
        names = tuple(f'{prior:g}' for prior in priors)
        if len(set(names)) != len(names):
            raise ValueError('priors must have distinct output names')
        self.input_dim, self.priors = input_dim, priors
        self.output_names = (*(f'dense_{name}' for name in names),
                             *(f'diagonal_{name}' for name in names),
                             'dense_bayes_mixture', 'zero')
        self.configs = {
            'priors': list(priors), 'alpha0': 2., 'beta0': 1.,
            'dense_prior_mass': .5, 'null_prior_mass': .5,
            'covariance_dtype': 'float32', 'scale_evidence_dtype': 'float64',
            'diagonal_projection': 'condition then discard off-diagonal covariance',
        }
        k = len(priors)
        self.dense_mean = torch.zeros((k, input_dim), device=device, dtype=torch.float32)
        self.diag_mean = torch.zeros_like(self.dense_mean)
        self.dense_cov = torch.zeros((k, input_dim, input_dim), device=device, dtype=torch.float32)
        prior_tensor = torch.tensor(priors, device=device, dtype=torch.float32)
        self.dense_cov.diagonal(dim1=-2, dim2=-1).copy_(prior_tensor[:, None])
        self.diag_cov = prior_tensor[:, None].expand(k, input_dim).clone()
        self.alpha = torch.tensor(2., device=device, dtype=torch.float64)
        self.beta = torch.ones(2 * k + 1, device=device, dtype=torch.float64)
        self.log_weights = torch.tensor([*([.5 / k] * k), .5], device=device, dtype=torch.float64).log()
        self.observations = torch.zeros((), device=device, dtype=torch.int64)
        self._zero = torch.zeros(1, device=device, dtype=torch.float32)

    def state_tensors(self) -> list[torch.Tensor]:
        """Complete mutable state in stable order; restore every tensor in place."""
        return [self.dense_mean, self.dense_cov, self.diag_mean, self.diag_cov,
                self.alpha, self.beta, self.log_weights, self.observations]

    @property
    def noise(self):
        """Posterior mean sigma^2 for dense, diagonal and null; excludes leverage."""
        return self.beta / (self.alpha - 1)

    @torch.no_grad()
    def diagnostics(self):
        """Host-only scale and effective-dimension reporting, never an update hook.

        beta/alpha is the Student-t scale squared before input leverage. The
        evaluator separately decides whether to pay for eigenvalue/Cholesky checks.
        """
        trace = self.dense_cov.diagonal(dim1=-2, dim2=-1).double().sum(-1).cpu()
        return {'alpha': float(self.alpha.cpu()), 'beta': self.beta.cpu().tolist(),
                'noise': self.noise.cpu().tolist(),
                'student_t_scale_squared': (self.beta / self.alpha).cpu().tolist(),
                'dense_effective_df': [self.input_dim - value / prior
                                       for value, prior in zip(trace.tolist(), self.priors)]}

    @torch.no_grad()
    def update(self, x, y) -> torch.Tensor:
        """Return newly allocated FP32 pre-label forecasts, then condition all state."""
        k = len(self.priors)
        dense_u = torch.matmul(self.dense_cov, x)
        diag_u = self.diag_cov * x
        # Promote before adding the observation-noise term: tiny priors must
        # not lose all leverage evidence to the FP32 spacing around one.
        dense_s = (dense_u * x).sum(-1).double() + 1
        diag_s = (diag_u * x).sum(-1).double() + 1
        dense_mu = (self.dense_mean * x).sum(-1)
        diag_mu = (self.diag_mean * x).sum(-1)
        mixture = (self.log_weights[:-1].exp() * dense_mu.double()).sum().float().reshape(1)
        prediction = torch.cat((dense_mu, diag_mu, mixture, self._zero))
        mean = torch.cat((dense_mu, diag_mu, self._zero)).double()
        leverage = torch.cat((dense_s, diag_s, torch.ones_like(self.alpha).reshape(1)))
        residual = y.double() - mean
        residual_squared = residual.square()
        # Only dense experts and the null participate in the Bayesian mixture.
        score_beta = torch.cat((self.beta[:k], self.beta[-1:]))
        score_residual = torch.cat((residual_squared[:k], residual_squared[-1:]))
        score_leverage = torch.cat((dense_s, leverage[-1:]))
        posterior_log = self.log_weights + student_t_log_prob(
            self.alpha, score_beta, score_residual, score_leverage)
        self.log_weights.copy_(posterior_log - posterior_log.logsumexp(-1))

        dense_rank = dense_u * dense_s.rsqrt().float()[:, None]
        self.dense_cov.sub_(dense_rank[:, :, None] * dense_rank[:, None, :])
        self.diag_cov.sub_(diag_u.square() * diag_s.reciprocal().float()[:, None])
        gain = (residual / leverage).float()
        self.dense_mean.add_(dense_u * gain[:k, None])
        self.diag_mean.add_(diag_u * gain[k:2 * k, None])
        self.beta.add_(.5 * residual_squared / leverage)
        self.alpha.add_(.5)
        self.observations.add_(1)
        return prediction

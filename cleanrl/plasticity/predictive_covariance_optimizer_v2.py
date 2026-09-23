"""Full-covariance quadratic evidence with a shared streaming design, v2.

The dense arm is conjugate Bayesian ridge regression with a normal/inverse-gamma
prior: delta | sigma² ~ N(0, sigma² * tau² I), alpha0=2, beta0=1. The fixed
initial weight is the anchor and delta is its learned displacement. Each distinct
prior scale owns one inverse information matrix, shared across target streams
and sparsity priors that observe the same feature sequence.

Coordinate inclusion uses a Gaussian Savage-Dickey approximation to the marginal
coefficient evidence under a dense nuisance prior. This is NOT an exact joint
spike/slab posterior (nor the exact Student-t marginal Bayes factor). Correlated
features can distribute support. Gating never feeds back into the dense ridge
mean, inverse covariance, or residual-variance estimate.
"""

import math

import torch


def _positive_divisor(value):
    """Protect positive subnormals only; invalid variances remain invalid."""
    tiny = torch.finfo(value.dtype).tiny
    return torch.where(value > 0, value.clamp_min(tiny), torch.nan)


class CovarianceState:
    """CUDA FP32 output weights with FP64 recursive least-squares state.

    ``weight`` has shape (configs, D). ``prior_scale`` and ``prior_density``
    must broadcast to (configs, 1), i.e. one scalar per configuration, not one
    per coefficient. The scale is the conditional prior standard deviation
    relative to the noise standard deviation. Density defaults to 1/D; density
    one returns the exact dense ridge mean for a valid posterior.

    Every call consumes one shared feature vector (D,) and noisy target vector
    (configs,). Callers must predict before calling ``step``. ``grad``,
    ``curvature``, and ``residual_sq`` are accepted only for the benchmark's
    existing call contract and are not used. No discounting is implemented.

    Construction deduplicates scales outside compilation. ``step`` has no
    data-dependent host synchronization; mutable buffers can be snapshotted
    and restored for compilation warmup and CUDA graph capture. Covariance
    storage is K*D*D, where K is the number of distinct scales, not configs.
    """

    @torch.no_grad()
    def __init__(self, weight, *, prior_scale=1.0, prior_density=None, memory=0.0):
        if not weight.is_cuda or weight.dtype != torch.float32:
            raise ValueError("CovarianceState requires CUDA FP32 master weights")
        if weight.ndim != 2 or min(weight.shape) < 1:
            raise ValueError("weight must have nonempty shape (configs, D)")
        if not math.isfinite(memory) or memory != 0:
            raise ValueError("CovarianceState requires memory=0 (no discounting)")

        configs, dim = weight.shape
        options = {"device": weight.device, "dtype": torch.float64}
        scale = torch.as_tensor(prior_scale, **options)
        density = torch.as_tensor(
            1.0 / dim if prior_density is None else prior_density, **options)
        if (torch.broadcast_shapes(scale.shape, (configs, 1)) != (configs, 1)
                or torch.broadcast_shapes(density.shape, (configs, 1)) != (configs, 1)):
            raise ValueError("prior tensors must broadcast to (configs, 1)")
        if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("prior_scale must be finite and positive")
        if not bool(torch.isfinite(density).all() & (density > 0).all()
                    & (density <= 1).all()):
            raise ValueError("prior_density must lie in (0, 1]")

        scales, self.scale_index = torch.unique(
            scale.expand(configs, 1).reshape(-1), sorted=True, return_inverse=True)
        prior_variance = scales.square()
        if not bool(torch.isfinite(prior_variance).all() & (prior_variance > 0).all()):
            raise ValueError("squared prior_scale must be finite and nonzero")
        self.log_prior_variance = prior_variance.log()[self.scale_index, None]
        density = density.expand(configs, 1)
        self.prior_logodds = density.log() - torch.log1p(-density)
        # Fixed hyperparameters and the fixed anchor are deliberately not buffers.
        self.anchor = weight.detach().to(dtype=torch.float64)
        self.inverse_covariance = torch.zeros((scales.numel(), dim, dim), **options)
        self.inverse_covariance.diagonal(dim1=-2, dim2=-1).copy_(prior_variance[:, None])
        self.mean = torch.zeros((configs, dim), **options)
        self.target_sum = torch.zeros_like(self.mean)
        self.target_square_sum = torch.zeros((configs, 1), **options)
        self.count = torch.zeros((), **options)
        self.noise_variance = torch.ones((configs, 1), **options)
        self.posterior_variance = prior_variance[self.scale_index, None].expand(configs, dim).clone()
        self.inclusion = density.expand(configs, dim).clone()

    def buffers(self):
        """All mutable tensors, without immutable priors, indices, or anchor."""
        return (self.inverse_covariance, self.mean, self.target_sum,
                self.target_square_sum, self.count, self.noise_variance,
                self.posterior_variance, self.inclusion)

    @torch.no_grad()
    def step(self, weight, grad, curvature, residual_sq, *, features, targets):
        x = features.to(dtype=torch.float64)
        y = targets.to(dtype=torch.float64)[:, None] - (self.anchor * x).sum(1, keepdim=True)
        # Only K matrix-vector products; never gather config-sized D*D matrices.
        u = torch.matmul(self.inverse_covariance, x)
        c = _positive_divisor(1.0 + (x * u).sum(1, keepdim=True))
        gain = (u / c)[self.scale_index]
        innovation = y - (self.mean * x).sum(1, keepdim=True)
        self.mean.addcmul_(gain, innovation)
        self.target_sum.addcmul_(y, x)
        self.target_square_sum.addcmul_(y, y)
        self.count.add_(1)

        # A symmetric outer product preserves symmetry without copying or
        # symmetrizing D*D matrices. addcmul_ broadcasts and updates in place.
        normalized_u = u / c.sqrt()
        self.inverse_covariance.addcmul_(
            normalized_u[:, :, None], normalized_u[:, None, :], value=-1)
        diagonal = self.inverse_covariance.diagonal(dim1=-2, dim2=-1)[self.scale_index]

        # E[sigma² | data] for alpha0=2, beta0=1. These sufficient statistics
        # refer to anchor-centered targets, not gated model residuals. FP64
        # reduces cancellation in sum(y²)-m·b; do not clamp a failed posterior.
        self.noise_variance.copy_(
            (2.0 + self.target_square_sum - (self.mean * self.target_sum).sum(1, keepdim=True))
            / (self.count + 2.0))
        self.posterior_variance.copy_(self.noise_variance * diagonal)
        variance = _positive_divisor(self.posterior_variance)
        log_bayes_factor = 0.5 * (
            diagonal.log() - self.log_prior_variance + self.mean.square() / variance)
        # Evidence already summarizes all data. Reuse the FIXED prior log odds,
        # never previous inclusion odds. Invalid variance propagates as NaN;
        # valid unbounded evidence may saturate sigmoid without clipping it.
        self.inclusion.copy_(torch.sigmoid(self.prior_logodds + log_bayes_factor))
        weight.copy_(self.anchor + self.inclusion * self.mean)

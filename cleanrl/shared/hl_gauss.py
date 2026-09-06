"""Histogram-Gaussian regression for categorical value heads.

New callers use explicit raw-bound HLGaussConfig -> HistogramGaussian, with
independent bin placement, integration transform, and expectation decoding.
The original learning/TD comparison is scripts/hlgauss/factorial.py. The
reward-normalized PPO correction is scripts/hlgauss/ppo_proxy_v3.py;
neither establishes MuJoCo policy returns. Legacy support classes below retain
frozen experiment behavior, including their coordinate-bound conventions.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


def symlog(x):
    return x.sign() * (x.abs() + 1.0).log()


def symexp(x):
    return x.sign() * (x.abs().exp() - 1.0)


@dataclass(frozen=True, kw_only=True)
class HLGaussConfig:
    """Raw-bound histogram configuration with historical proxy defaults.

    Bounds are always in raw scalar units, including for ``transform="symlog"``.
    Bounds remain required. Defaults retain the v2 proxy's 31-bin, sigma-2,
    raw-Gaussian, symexp-center configuration; they are NOT a PPO recommendation.
    That configuration regressed in the reward-normalized sphere PPO run.
    New experiments should explicitly compare sigma 0.5/0.75 on matched bins
    and bounds, and measure raw bandwidth, projection bias and clipped-PPO
    usefulness (benchmarks/hlgauss/hl_gauss_ppo_proxy_v3.json). Lower sigma need not
    remove nonuniform-bin mean bias. Historical defaults are retained so frozen
    versions that constructed this config implicitly do not silently change.
    ``sigma_ratio`` multiplies the bin width in the selected coordinate space.
    ``centers`` puts the outer centers on the bounds and extends edges by half
    a bin; ``edges`` puts the outer edges on the bounds.
    ``symexp_centers`` uses symexp-spaced raw centers, raw midpoint cells,
    and sigma_ratio times the mean raw cell width. It requires a linear
    transform: nonuniform placement and Gaussian integration are separate.

    Gaussian smoothing and truncation do not preserve target means. Under
    cross-entropy, a noisy target learns the conditional average of its
    projected labels. Scalar decoding takes the raw-support expectation of
    those labels; transformed decoding inverses their coordinate expectation.
    Neither is in general the conditional raw target mean, and the two
    estimands differ for nonlinear transforms.

    Historical v2 configuration (benchmarks/hlgauss/hl_gauss_proxy_v2.json)::

        head = HLGaussConfig(
            num_bins=31, v_min=-150.0, v_max=150.0, sigma_ratio=2.0,
            transform="linear", bin_type="symexp_centers", decode="scalar",
        ).build("cuda")

    Replace the example raw bounds with the task's return range. The selected
    proxy support was +/-3 times its known generating scale; it was NOT an
    automatic estimator, nor a universal range recommendation. ``centers`` is
    the simpler uniform-raw comparator with the same bins/sigma/decode.
    Nonuniform placement does not imply symlog Gaussian smoothing. Normalized
    scalar MSE remained the stronger average supervised baseline. The old TD
    utility cancelled constant critic bias and did not retain matched sigmas
    for this configuration; its gains cannot justify a universal default.
    """

    v_min: float
    v_max: float
    num_bins: int = 31
    sigma_ratio: float = 2.0
    transform: Literal["linear", "symlog"] = "linear"
    bin_type: Literal["centers", "edges", "symexp_centers"] = "symexp_centers"
    decode: Literal["scalar", "transformed"] = "scalar"

    def __post_init__(self):
        if isinstance(self.num_bins, bool) or not isinstance(self.num_bins, int) or self.num_bins < 2:
            raise ValueError("num_bins must be an integer >= 2")
        if not math.isfinite(self.v_min) or not math.isfinite(self.v_max) or self.v_min >= self.v_max:
            raise ValueError("v_min and v_max must be finite, ordered raw scalar bounds")
        if not math.isfinite(self.sigma_ratio) or self.sigma_ratio <= 0:
            raise ValueError("sigma_ratio must be finite and positive")
        if self.transform not in ("linear", "symlog"):
            raise ValueError("transform must be 'linear' or 'symlog'")
        if self.bin_type not in ("centers", "edges", "symexp_centers"):
            raise ValueError("bin_type must be 'centers', 'edges', or 'symexp_centers'")
        if self.bin_type == "symexp_centers" and self.transform != "linear":
            raise ValueError("symexp_centers integrates in raw space and requires transform='linear'")
        if self.decode not in ("scalar", "transformed"):
            raise ValueError("decode must be 'scalar' or 'transformed'")

    def build(self, device="cpu") -> "HistogramGaussian":
        """Build cached supports in the default floating dtype on ``device``."""
        return HistogramGaussian(self, device=device)


class HistogramGaussian(torch.nn.Module):
    """Gaussian histogram labels with independently selected scalar decoder.

    Supports and edges are registered buffers, so standard module ``to`` and
    state-dict operations apply. Inputs should share the module's device;
    ordinary PyTorch dtype promotion applies. Projection preserves all input
    dimensions and appends the bin dimension, including for scalar/empty input.

    These are truncated, normalized Gaussian labels, not mean-matched labels.
    In particular ``decode="scalar"`` describes the expectation of the learned
    histogram, not a guarantee of an unbiased target mean. See HLGaussConfig.
    """

    coord_support: torch.Tensor
    coord_edges: torch.Tensor
    support: torch.Tensor

    @staticmethod
    def _symlog(x):
        # Valid domains in both branches, log1p near zero, derivative 1 at zero.
        return torch.where(x >= 0, torch.log1p(x.clamp_min(0)), -torch.log1p((-x).clamp_min(0)))

    @staticmethod
    def _symexp(x):
        return torch.where(x >= 0, torch.expm1(x.clamp_min(0)), -torch.expm1((-x).clamp_min(0)))

    def __init__(self, config: HLGaussConfig, device="cpu"):
        super().__init__()
        self.config = config
        lo, hi = config.v_min, config.v_max
        if config.transform == "symlog":
            lo = math.copysign(math.log1p(abs(lo)), lo)
            hi = math.copysign(math.log1p(abs(hi)), hi)
        width = (hi - lo) / (config.num_bins if config.bin_type == "edges" else config.num_bins - 1)
        if config.bin_type == "symexp_centers":
            coord_lo = math.copysign(math.log1p(abs(lo)), lo)
            coord_hi = math.copysign(math.log1p(abs(hi)), hi)
            centers = self._symexp(torch.linspace(coord_lo, coord_hi, config.num_bins, device=device))
            edges = torch.cat(
                (
                    centers[:1] - (centers[1:2] - centers[:1]) * 0.5,
                    (centers[:-1] + centers[1:]) * 0.5,
                    centers[-1:] + (centers[-1:] - centers[-2:-1]) * 0.5,
                )
            )
            step = (coord_hi - coord_lo) / (config.num_bins - 1)
            inverse = lambda z: math.copysign(math.expm1(abs(z)), z)
            span = hi - lo + 0.5 * (hi - inverse(coord_hi - step) + inverse(coord_lo + step) - lo)
            width = span / config.num_bins
        elif config.bin_type == "edges":
            edges = torch.linspace(lo, hi, config.num_bins + 1, device=device)
            centers = (edges[:-1] + edges[1:]) * 0.5
        else:
            centers = torch.linspace(lo, hi, config.num_bins, device=device)
            edges = torch.linspace(lo - width * 0.5, hi + width * 0.5, config.num_bins + 1, device=device)
        self._sqrt_two_sigma = math.sqrt(2.0) * config.sigma_ratio * width
        self.register_buffer("coord_support", centers)
        self.register_buffer("coord_edges", edges)
        self.register_buffer("support", self._symexp(centers) if config.transform == "symlog" else centers.clone())

    def project(self, targets: torch.Tensor) -> torch.Tensor:
        """Integrate Gaussian bin masses after clipping in raw scalar space.

        Tail intervals use survival probabilities rather than subtracting CDFs
        near one. Mean-adjacent intervals use erf to avoid cancellation
        for a Gaussian much wider than a bin. Only ``loss`` detaches targets.
        """
        targets = targets.clamp(self.config.v_min, self.config.v_max)
        if self.config.transform == "symlog":
            targets = self._symlog(targets)
        z = (self.coord_edges - targets.unsqueeze(-1)) / self._sqrt_two_sigma
        lower, upper = z[..., :-1], z[..., 1:]
        # Evaluate shared edges once; reflection keeps both tails near zero.
        survival = torch.erfc(z.abs())
        tail_mass = (survival[..., 1:] - survival[..., :-1]).abs()
        cdf = torch.erf(z)
        central_mass = cdf[..., 1:] - cdf[..., :-1]
        mass = torch.where((lower <= 1) & (upper >= -1), central_mass, tail_mass)
        return mass / mass.sum(dim=-1, keepdim=True)

    def probs_to_scalar(self, probs: torch.Tensor) -> torch.Tensor:
        """Decode E[raw center] or inverse(E[coordinate center]), explicitly."""
        if probs.ndim == 0 or probs.shape[-1] != self.config.num_bins:
            raise ValueError("probabilities must have a final dimension of num_bins")
        if self.config.decode == "scalar":
            return (probs * self.support).sum(dim=-1)
        value = (probs * self.coord_support).sum(dim=-1)
        return self._symexp(value) if self.config.transform == "symlog" else value

    def to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax logits and decode using the configured expectation."""
        return self.probs_to_scalar(logits.softmax(dim=-1))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, reduction="mean") -> torch.Tensor:
        """Cross-entropy with detached Gaussian labels; no target gradients.

        ``none`` retains the target shape; ``sum`` and ``mean`` reduce it.
        Empty means follow PyTorch's convention and return NaN.
        """
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        if logits.ndim == 0 or logits.shape[-1] != self.config.num_bins or logits.shape[:-1] != targets.shape:
            raise ValueError("logits must have shape (*targets.shape, num_bins)")
        labels = self.project(targets.detach())
        values = -(labels * logits.log_softmax(dim=-1)).sum(dim=-1)
        if reduction == "none":
            return values
        return values.mean() if reduction == "mean" else values.sum()


class HLGaussSupport:
    """Discretized support with HL-Gauss projection for categorical value heads.

    Args:
        num_bins: Number of bins in the discrete support.
        v_min: Minimum value of the support range.
        v_max: Maximum value of the support range.
        sigma_ratio: Gaussian sigma as a fraction of bin width.
        device: Torch device.
        use_symlog: If True, apply symlog to targets before projection
                    and symexp after converting logits to scalar.
    """

    def __init__(
        self,
        num_bins,
        v_min,
        v_max,
        sigma_ratio,
        device,
        use_symlog=False,
        support_is_edges=False,
        clamp_targets=True,
        eps=1e-10,
    ):
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.support_is_edges = support_is_edges
        self.clamp_targets = clamp_targets
        self.eps = eps

        denom = num_bins if support_is_edges else num_bins - 1
        self.bin_width = (v_max - v_min) / denom
        self.sigma = sigma_ratio * self.bin_width
        if support_is_edges:
            self.edges = torch.linspace(v_min, v_max, num_bins + 1, device=device)
            self.support = (self.edges[:-1] + self.edges[1:]) / 2.0
        else:
            self.edges = None
            self.support = torch.linspace(v_min, v_max, num_bins, device=device)
        self.use_symlog = use_symlog

    def probs_to_scalar(self, probs):
        """Decode probabilities via inverse_transform(E[transformed bin])."""
        value = (probs * self.support).sum(dim=-1)
        if self.use_symlog:
            value = symexp(value)
        return value

    def to_scalar(self, logits):
        """Decode logits via inverse_transform(E[transformed bin]).

        This matches hl-gauss-pytorch's transform_from_logits semantics. For
        symlog supports this is symexp(E[z]), not E[symexp(z)].
        """
        probs = torch.softmax(logits, dim=-1)
        return self.probs_to_scalar(probs)

    def probs_to_expected_scalar(self, probs):
        """Decode probabilities as E[value(bin)] in scalar space."""
        scalar_support = symexp(self.support) if self.use_symlog else self.support
        return (probs * scalar_support).sum(dim=-1)

    def to_expected_scalar(self, logits):
        """Decode logits as E[value(bin)] in scalar space."""
        return self.probs_to_expected_scalar(torch.softmax(logits, dim=-1))

    def project(self, targets):
        """Project scalar targets onto HL-Gauss categorical distribution.

        For each target, compute:
            P(bin_i) = Phi((z_i + w/2 - target) / sigma)
                     - Phi((z_i - w/2 - target) / sigma)
        where Phi is the standard normal CDF, z_i are bin centers, w is bin width.
        """
        if self.use_symlog:
            targets = symlog(targets)
        if self.clamp_targets:
            targets = targets.clamp(self.v_min, self.v_max)
        targets = targets.unsqueeze(-1)

        if self.support_is_edges:
            assert self.edges is not None
            edges = self.edges.unsqueeze(0)
            cdf_evals = torch.erf((edges - targets) / (self.sigma * np.sqrt(2)))
            z = cdf_evals[..., -1:] - cdf_evals[..., :1]
            probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
            return probs / z.clamp(min=self.eps)

        support = self.support.unsqueeze(0)
        half_w = self.bin_width / 2.0
        upper = (support + half_w - targets) / self.sigma
        lower = (support - half_w - targets) / self.sigma
        probs = 0.5 * (torch.erf(upper / np.sqrt(2)) - torch.erf(lower / np.sqrt(2)))
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def project_to_logprobs(self, targets, eps=1e-20):
        """Project targets and return finite log-probabilities."""
        return self.project(targets).clamp_min(eps).log()


class Dreamer3BucketHLGaussSupport:
    """HL-Gauss labels on Dreamer3 symexp-spaced scalar buckets.

    Dreamer3's scalar buckets are raw-space centers generated by applying
    symexp to evenly spaced symlog coordinates. This support keeps those raw
    centers for scalar decoding, while assigning Gaussian histogram mass over
    the uniform symlog-coordinate intervals.
    """

    def __init__(self, num_bins, coord_min, coord_max, sigma_ratio, device, eps=1e-10):
        if num_bins % 2 != 1:
            raise ValueError("Dreamer3BucketHLGaussSupport expects an odd num_bins to keep one exact zero bucket")
        if not np.isclose(abs(coord_min), abs(coord_max)):
            raise ValueError("Dreamer3 bucket coordinates must be symmetric around zero")
        self.num_bins = num_bins
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.sigma_ratio = sigma_ratio
        self.eps = eps

        half = torch.linspace(coord_min, 0.0, (num_bins - 1) // 2 + 1, device=device)
        self.coord_support = torch.cat([half, -half[:-1].flip(0)])
        coord_step = (self.coord_support[1] - self.coord_support[0]).abs()
        self.coord_bin_width = coord_step
        self.sigma = sigma_ratio * coord_step

        self.coord_edges = torch.empty(num_bins + 1, device=device, dtype=self.coord_support.dtype)
        self.coord_edges[1:-1] = (self.coord_support[:-1] + self.coord_support[1:]) / 2.0
        self.coord_edges[0] = self.coord_support[0] - 0.5 * coord_step
        self.coord_edges[-1] = self.coord_support[-1] + 0.5 * coord_step

        self.support = symexp(self.coord_support)
        self.edges = symexp(self.coord_edges)
        raw_widths = self.support[1:] - self.support[:-1]
        self.bin_width = raw_widths.abs().min().item()
        self.low_endpoint = torch.zeros_like(self.support)
        self.low_endpoint[0] = 1.0
        self.high_endpoint = self.low_endpoint.flip(0)

    def probs_to_scalar(self, probs):
        """Decode as the expected raw scalar over symexp-spaced bucket centers."""
        n = probs.shape[-1]
        m = (n - 1) // 2
        p_left = probs[..., :m]
        p_zero = probs[..., m : m + 1]
        p_right = probs[..., m + 1 :]
        s_left = self.support[:m]
        s_zero = self.support[m : m + 1]
        s_right = self.support[m + 1 :]
        paired = (p_left * s_left).flip(-1) + p_right * s_right
        return (p_zero * s_zero).sum(dim=-1) + paired.sum(dim=-1)

    def to_scalar(self, logits):
        return self.probs_to_scalar(torch.softmax(logits, dim=-1))

    def project_log_probs(self, targets):
        """Return normalized Gaussian bin log-masses without tail underflow."""
        coord_targets = symlog(targets).clamp(self.coord_min, self.coord_max)
        lower = (
            self.coord_edges[:-1] - coord_targets.unsqueeze(-1)
        ) / self.sigma
        upper = (
            self.coord_edges[1:] - coord_targets.unsqueeze(-1)
        ) / self.sigma

        log_cdf_upper = torch.special.log_ndtr(upper)
        log_cdf_lower = torch.special.log_ndtr(lower)
        log_survival_lower = torch.special.log_ndtr(-lower)
        log_survival_upper = torch.special.log_ndtr(-upper)
        use_survival = lower >= 0.0
        log_large = torch.where(
            use_survival,
            log_survival_lower,
            log_cdf_upper,
        )
        log_small = torch.where(
            use_survival,
            log_survival_upper,
            log_cdf_lower,
        )
        log_masses = log_large + torch.log1p(
            -torch.exp(log_small - log_large)
        )
        return log_masses - torch.logsumexp(log_masses, dim=-1, keepdim=True)

    def project(self, targets):
        return self.project_log_probs(targets).exp()

    def project_moment_matched(
        self,
        targets,
        iterations=32,
        tilt_bound=1.0,
        log_mass_cutoff=30.0,
    ):
        """KL-project HL-Gauss labels to preserve their raw scalar mean.

        Gaussian smoothing in symlog coordinates otherwise shifts the decoded
        raw mean away from the target through symexp curvature. Negligible far
        tails are truncated before a minimum-KL exponential tilt enforces
        E[symexp(bucket)] = target without introducing remote support mass.
        """
        log_probs = self.project_log_probs(targets)
        maximum_log_prob = log_probs.max(dim=-1, keepdim=True).values
        log_probs = torch.where(
            log_probs >= maximum_log_prob - log_mass_cutoff,
            log_probs,
            -torch.inf,
        )
        matched_targets = targets.clamp(self.support[0], self.support[-1])
        low = torch.full_like(matched_targets, -tilt_bound)
        high = torch.full_like(matched_targets, tilt_bound)
        for _ in range(iterations):
            midpoint = 0.5 * (low + high)
            candidate = torch.softmax(
                log_probs + midpoint.unsqueeze(-1) * self.support,
                dim=-1,
            )
            candidate_mean = self.probs_to_scalar(candidate)
            low = torch.where(candidate_mean < matched_targets, midpoint, low)
            high = torch.where(candidate_mean < matched_targets, high, midpoint)

        tilt = 0.5 * (low + high)
        matched = torch.softmax(
            log_probs + tilt.unsqueeze(-1) * self.support,
            dim=-1,
        )
        matched = torch.where(
            (targets <= self.support[0]).unsqueeze(-1),
            self.low_endpoint,
            matched,
        )
        return torch.where(
            (targets >= self.support[-1]).unsqueeze(-1),
            self.high_endpoint,
            matched,
        )

    def cdf_fraction(self, targets):
        coord_targets = symlog(targets).clamp(self.coord_min, self.coord_max).unsqueeze(-1)
        lo = self.coord_edges[:-1]
        frac = (coord_targets - lo) / self.coord_bin_width.clamp_min(self.eps)
        return frac.clamp(0.0, 1.0)


class HLGaussCDFSupport:
    """Cumulative-CDF ("survival") variant of HLGaussSupport: K-1 sigmoid thresholds.

    The head emits K-1 threshold logits predicting survival probabilities
    s_k = P(Z > e_k) at the K-1 INTERIOR BIN EDGES e_k (midpoints between the
    K uniform bin centers spanning [v_min, v_max]). Training is per-threshold
    BCE against the smoothed survival labels of the HL-Gauss target,
    y_k = Phi((t - e_k)/sigma). The scalar decode is the exact tail-sum
    identity for a distribution on the centers:
        E[Z] = v_min + bin_width * sum_k s_k.
    (Edge thresholds, not centers: with thresholds at centers the tail-sum
    becomes a left-endpoint Riemann sum of the survival function and decodes
    with a systematic +bin_width/2 bias.)

    Why not softmax+CE: CE prices a probability-mass misplacement independently
    of how far it moves the expectation decode (eps mass at distance d costs
    ~eps nats but shifts E[Z] by eps*d). Per-threshold BCE decomposes the decode
    error exactly -- d loss / d logit_k = s_k - y_k while the decode error is
    bin_width * sum_k (s_k - y_k) -- so the loss geometry matches mean
    consumption while each threshold remains a proper scoring rule (the full
    CDF is calibrated at optimum). The implied survival curve is not forced
    monotone (no simplex constraint); harmless when only the tail-sum mean is
    read, but enforce monotonicity before reading quantiles from it.
    """

    def __init__(
        self,
        num_bins,
        v_min,
        v_max,
        sigma_ratio,
        device,
        use_symlog=False,
        clamp_targets=True,
    ):
        self.num_bins = num_bins
        self.num_thresholds = num_bins - 1
        self.v_min = v_min
        self.v_max = v_max
        self.bin_width = (v_max - v_min) / (num_bins - 1)
        self.sigma = sigma_ratio * self.bin_width
        self.use_symlog = use_symlog
        self.clamp_targets = clamp_targets
        # Thresholds at the K-1 interior bin edges (center midpoints), so
        # sum_k s_k in [0, K-1] maps the decode onto [v_min, v_max] and the
        # tail-sum identity is exact for mass on the bin centers.
        centers = torch.linspace(v_min, v_max, num_bins, device=device)
        self.thresholds = (centers[:-1] + centers[1:]) / 2.0

    def to_scalar(self, logits):
        """Tail-sum decode E[Z] = v_min + w * sum_k sigmoid(logit_k).

        Note: with use_symlog=True this is symexp(E[z]) (HLGaussSupport's
        convention), NOT the Jensen-correct E[symexp(z)] some callers compute
        externally via a scalar_support — don't swap one in for the other.
        """
        value = self.v_min + self.bin_width * torch.sigmoid(logits).sum(dim=-1)
        if self.use_symlog:
            value = symexp(value)
        return value

    def cdf_labels(self, targets):
        """Smoothed survival labels y_k = P(N(target, sigma) > e_k), shape (..., K-1).

        Uses the untruncated Gaussian survival, so perfect labels at a clamped
        target decode ~0.28*bin_width inward of v_min/v_max; renormalize over
        [v_min, v_max] if exact edge targets ever matter.
        """
        if self.use_symlog:
            targets = symlog(targets)
        if self.clamp_targets:
            targets = targets.clamp(self.v_min, self.v_max)
        z = (targets.unsqueeze(-1) - self.thresholds) / (self.sigma * np.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(z))

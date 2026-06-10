"""HL-Gauss categorical value head utilities.

Provides HLGaussSupport for discretized distributional value estimation,
with optional symlog/symexp scaling (DreamerV3-style).
"""

import numpy as np
import torch


def symlog(x):
    return x.sign() * (x.abs() + 1.0).log()


def symexp(x):
    return x.sign() * (x.abs().exp() - 1.0)


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

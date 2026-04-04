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

    def __init__(self, num_bins, v_min, v_max, sigma_ratio, device, use_symlog=False):
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.bin_width = (v_max - v_min) / (num_bins - 1)
        self.sigma = sigma_ratio * self.bin_width
        self.support = torch.linspace(v_min, v_max, num_bins, device=device)
        self.use_symlog = use_symlog

    def to_scalar(self, logits):
        """Convert logits to scalar value via E[z] = sum(softmax(logits) * support)."""
        probs = torch.softmax(logits, dim=-1)
        value = (probs * self.support).sum(dim=-1)
        if self.use_symlog:
            value = symexp(value)
        return value

    def project(self, targets):
        """Project scalar targets onto HL-Gauss categorical distribution.

        For each target, compute:
            P(bin_i) = Phi((z_i + w/2 - target) / sigma)
                     - Phi((z_i - w/2 - target) / sigma)
        where Phi is the standard normal CDF, z_i are bin centers, w is bin width.
        """
        if self.use_symlog:
            targets = symlog(targets)
        targets = targets.clamp(self.v_min, self.v_max)
        targets = targets.unsqueeze(-1)
        support = self.support.unsqueeze(0)
        half_w = self.bin_width / 2.0
        upper = (support + half_w - targets) / self.sigma
        lower = (support - half_w - targets) / self.sigma
        probs = 0.5 * (torch.erf(upper / np.sqrt(2)) - torch.erf(lower / np.sqrt(2)))
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

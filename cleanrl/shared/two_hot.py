"""Raw-space two-hot labels with Dreamer or uniform support geometry."""

import math

import torch


class DreamerTwoHotSupport(torch.nn.Module):
    """Raw-space barycentric labels on Dreamer or linear raw centers.

    ``max_abs_value`` is a raw bound, not a symlog coordinate. The default
    reproduces Dreamer3's [-20, 20] coordinate range. Even supports retain
    Dreamer3's duplicate central zeros; the two-bin case contains only the
    endpoints. ``spacing="linear"`` instead uses uniformly spaced raw centers,
    without symlog/symexp. All numerical operations use FP32.
    """

    support: torch.Tensor

    def __init__(self, num_bins=255, max_abs_value=math.expm1(20), device=None, *, spacing="symexp"):
        super().__init__()
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer of at least two")
        if not math.isfinite(max_abs_value) or not 0 < max_abs_value <= torch.finfo(torch.float32).max:
            raise ValueError("max_abs_value must be positive, finite, and representable in FP32")
        if spacing not in ("symexp", "linear"):
            raise ValueError("spacing must be symexp or linear")
        self.num_bins = num_bins
        self.max_abs_value = max_abs_value
        self.spacing = spacing
        if spacing == "symexp":
            coordinate_bound = math.log1p(max_abs_value)
            half_size = (num_bins + 1) // 2
            half = torch.linspace(-coordinate_bound, 0.0, half_size, dtype=torch.float32, device=device)
            half = half.sign() * half.abs().expm1()
        else:
            # Mirror explicitly for exact zero-mean initialization. Unlike
            # Dreamer's even support, a uniform even grid has no zero atom.
            half_size = (num_bins + 1) // 2
            endpoint = 0.0 if num_bins % 2 else -max_abs_value / (num_bins - 1)
            half = torch.linspace(-max_abs_value, endpoint, half_size, dtype=torch.float32, device=device)
        # Validate constructed FP32 endpoints, not just the Python bound:
        # expm1 can overflow, tiny bounds can underflow, and a two-bin interval
        # spans twice the endpoint magnitude. This sync occurs only at setup.
        endpoint = half[0]
        if not (torch.isfinite(endpoint) & (endpoint < 0) & (endpoint >= -torch.finfo(torch.float32).max / 2)).item():
            raise ValueError("max_abs_value must produce nonzero endpoints and finite FP32 interval widths")
        mirrored = -half[:-1].flip(0) if num_bins % 2 else -half.flip(0)
        self.register_buffer("support", torch.cat((half, mirrored)))

    def project(self, targets: torch.Tensor) -> torch.Tensor:
        """Return detached FP32 labels of shape ``(*targets.shape, num_bins)``.

        Right insertion matches Dreamer3's counts of centers <= and > target,
        including its choice of the second zero bin for an exact zero target.
        Coincident clipped indices each receive half mass, adding to one-hot.
        """
        targets = targets.detach().float().contiguous()
        insertion = torch.searchsorted(self.support, targets, right=True)
        below = (insertion - 1).clamp(0, self.num_bins - 1)
        above = insertion.clamp(0, self.num_bins - 1)
        equal = below == above
        dist_below = torch.where(equal, 1.0, (self.support[below] - targets).abs())
        dist_above = torch.where(equal, 1.0, (self.support[above] - targets).abs())
        total = dist_below + dist_above
        labels = targets.new_zeros((*targets.shape, self.num_bins))
        labels.scatter_add_(-1, below.unsqueeze(-1), (dist_above / total).unsqueeze(-1))
        labels.scatter_add_(-1, above.unsqueeze(-1), (dist_below / total).unsqueeze(-1))
        return labels

    def probs_to_scalar(self, probs: torch.Tensor) -> torch.Tensor:
        """Decode E[raw center], pairing mirrored products before summing."""
        if probs.ndim == 0 or probs.shape[-1] != self.num_bins:
            raise ValueError("probabilities must have a final dimension of num_bins")
        probs = probs.float()
        middle = self.num_bins // 2
        left = probs[..., :middle] * self.support[:middle]
        if self.num_bins % 2:
            center = (probs[..., middle : middle + 1] * self.support[middle : middle + 1]).sum(-1)
            right = probs[..., middle + 1 :] * self.support[middle + 1 :]
            return center + (left.flip(-1) + right).sum(-1)
        right = probs[..., middle:] * self.support[middle:]
        return (left.flip(-1) + right).sum(-1)

    def to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply FP32 softmax and decode the expected raw value."""
        if logits.ndim == 0 or logits.shape[-1] != self.num_bins:
            raise ValueError("logits must have a final dimension of num_bins")
        return self.probs_to_scalar(logits.float().softmax(dim=-1))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, reduction="mean") -> torch.Tensor:
        """Cross-entropy against scalar targets, with no target gradients."""
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        if logits.ndim == 0 or logits.shape[-1] != self.num_bins or logits.shape[:-1] != targets.shape:
            raise ValueError("logits must have shape (*targets.shape, num_bins)")
        labels = self.project(targets)
        values = -(labels * logits.float().log_softmax(dim=-1)).sum(dim=-1)
        if reduction == "none":
            return values
        return values.mean() if reduction == "mean" else values.sum()

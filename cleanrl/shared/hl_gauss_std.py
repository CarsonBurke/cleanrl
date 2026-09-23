"""HL-Gauss geometry defined in units of the value target's own scale.

The repo's earlier categorical critics fixed ``[v_min, v_max]`` in absolute
return units and then derived sigma from the bin width. Reward-normalized
HalfCheetah targets move from mean 0 to mean ~4 with a cross-state spread of
0.25-0.6 over a run, so every absolute support was either 20-260 target
standard deviations wide (a handful of live bins, sigma up to 19 target std)
or had to be guessed per environment. Farebrother et al. 2024 fix the support
to the known value range and tune ``sigma/bin_width``; the invariant that
actually matters is *resolution relative to the targets*, which an absolute
support does not provide when the targets move.

``StandardizedHistogram`` keeps the grid in standardized units: the head
predicts a distribution over ``z`` on ``[-half_span, half_span]`` and the raw
value is ``mean + scale * E[z]``, where ``mean``/``scale`` are this rollout's
own target statistics, frozen for the duration of the iteration. Bin width and
sigma are therefore a constant fraction of the target spread for the whole run.
There is deliberately no EMA: a rollout carries tens of thousands of targets,
so its mean and standard deviation are already precise and smoothing them only
delays the support's response to the drift it exists to follow.

Two further differences from ``HistogramGaussian``:

* The outer bins are half-infinite, so the labels integrate to one without
  renormalization and no target is ever clamped. Truncating a Gaussian and
  renormalizing moves the label mean; putting the tail in the edge bin does
  not (beyond representing that bin by its center).
* ``value_gradient_gain`` exposes the factor that makes the cross-entropy
  gradient in *value* units equal the MSE gradient. For a softmax head,
  ``dCE/dlogits = p - q`` and, to first order in ``target - value``,
  ``dCE/dvalue = dMSE/dvalue / Var_p(z)`` -- so a categorical critic silently
  runs at an effective value-space learning rate of ``1/Var_p``, which is
  1/0.0065 ~ 150x for a sharp head. ``vf_coef`` tuned for MSE does not carry
  over without this factor.
"""

import math

import torch
from torch import nn

SQRT2 = math.sqrt(2.0)


class StandardizedHistogram(nn.Module):
    """Gaussian histogram labels on a target-standardized, unclamped support."""

    centers: torch.Tensor
    edges: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(
        self,
        num_bins: int = 101,
        *,
        half_span: float = 5.0,
        sigma_bins: float = 0.75,
        min_scale: float = 1e-3,
        device=None,
    ):
        super().__init__()
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 3:
            raise ValueError("num_bins must be an integer >= 3")
        if not math.isfinite(half_span) or half_span <= 0:
            raise ValueError("half_span must be finite and positive")
        if not math.isfinite(sigma_bins) or sigma_bins <= 0:
            raise ValueError("sigma_bins must be finite and positive")
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive")
        self.num_bins = num_bins
        self.half_span = float(half_span)
        self.sigma_bins = float(sigma_bins)
        self.min_scale = float(min_scale)
        self.bin_width = 2.0 * half_span / (num_bins - 1)
        self.sigma = self.sigma_bins * self.bin_width
        centers = torch.linspace(-half_span, half_span, num_bins, device=device)
        # Half-infinite outer bins: labels sum to one with no clamp and no renormalization.
        interior = (centers[:-1] + centers[1:]) * 0.5
        edges = torch.cat(
            (
                torch.full((1,), -math.inf, device=device),
                interior,
                torch.full((1,), math.inf, device=device),
            )
        )
        self.register_buffer("centers", centers)
        self.register_buffer("edges", edges)
        self.register_buffer("mean", torch.zeros((), device=device))
        self.register_buffer("scale", torch.ones((), device=device))
        self._scaled_sigma = SQRT2 * self.sigma

    @torch.no_grad()
    def observe(self, targets: torch.Tensor) -> None:
        """Adopt this rollout's target location and scale as the support frame.

        No smoothing: one rollout already carries tens of thousands of targets,
        so the batch mean and standard deviation are precise, and any EMA only
        adds lag to the quantity the support exists to track. Callers reframe
        the previous rollout's decoded values so the change is exactly a
        change of readout convention, not a change of prediction.
        """
        self.mean.copy_(targets.mean())
        self.scale.copy_(targets.std().clamp_min(self.min_scale))

    def standardize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale

    def project(self, targets: torch.Tensor) -> torch.Tensor:
        """Gaussian bin masses for raw-unit targets; no clamping, no renormalization."""
        z = (targets - self.mean) / self.scale
        u = (self.edges - z.unsqueeze(-1)) / self._scaled_sigma
        lower, upper = u[..., :-1], u[..., 1:]
        # erf differences cancel in a shared tail; reflect to erfc there instead.
        survival = torch.erfc(u.abs())
        tail_mass = (survival[..., :-1] - survival[..., 1:]).abs()
        central_mass = torch.erf(upper) - torch.erf(lower)
        mass = torch.where((lower <= 1) & (upper >= -1), central_mass, tail_mass)
        return mass * 0.5

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Raw-unit expectation of the categorical head."""
        return self.mean + self.scale * (logits.softmax(dim=-1) * self.centers).sum(dim=-1)

    def decode_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return self.mean + self.scale * (probs * self.centers).sum(dim=-1)

    def standardized_variance(self, probs: torch.Tensor) -> torch.Tensor:
        """Var_p(z) in standardized units; the inverse CE-vs-MSE gradient gain.

        Centered form: E[z^2] - E[z]^2 cancels catastrophically in fp32 once the
        head is sharp and its mean sits near the edge of the support.
        """
        mean = (probs * self.centers).sum(dim=-1, keepdim=True)
        return (probs * (self.centers - mean).square()).sum(dim=-1).clamp_min(0.0)

    def value_gradient_gain(self, probs: torch.Tensor) -> torch.Tensor:
        """Per-sample CE multiplier whose value-space gradient matches 0.5*MSE.

        ``Var_p(z) * scale^2`` is the exact first-order factor at any stage of
        training. A constant built from the converged-head variance
        ``sigma^2 + bin_width^2/12`` is only right once the head is sharp: a
        near-uniform head over the whole support has ``Var_p ~ half_span^2/3``,
        which is 200x larger, so the constant over-corrects at initialization.
        """
        return self.standardized_variance(probs) * self.scale.square()

    def converged_gain(self) -> torch.Tensor:
        """The sharp-head limit of ``value_gradient_gain``, for diagnostics."""
        return self.scale.square() * (self.sigma**2 + self.bin_width**2 / 12.0)

    def extra_repr(self) -> str:
        return (
            f"num_bins={self.num_bins}, half_span={self.half_span}, "
            f"sigma_bins={self.sigma_bins}, sigma_z={self.sigma:.4g}"
        )

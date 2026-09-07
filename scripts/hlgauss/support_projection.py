"""Fixed raw supports for the CPU critic-proxy experiment.

These labels integrate a symmetrically truncated Gaussian against piecewise
linear (barycentric) basis functions. They are neither ordinary histogram
masses nor exponential-tilted HL-Gauss labels. Uniform geometry is therefore a
necessary projection control when comparing asinh placement with HL-Gauss.
The calibration median/standard deviation must be supplied independently;
this module never adapts its support to training targets.
"""

import math

import torch


class MeanPreservingSupport(torch.nn.Module):
    """Gaussian barycentric labels whose raw expectation is the clipped target.

    ``fixed`` uses sigma = 1.5 * calibration scale; ``local`` uses sigma =
    0.75 * the enclosing raw interval width; ``twohot`` omits smoothing.
    Asinh placement is uniform in asinh((value - center) / scale), including
    the exact raw endpoints. Integration and support construction use float64
    to avoid losing narrow intervals or shifted support coordinates. Public
    projection/decoding results retain the input dtype. Move the module to the
    input device with ``to(device)``; casting its buffers to a lower precision
    also reduces the precision of its support.
    """

    support: torch.Tensor
    _decode_support: torch.Tensor | None

    def __init__(
        self,
        *,
        v_min: float,
        v_max: float,
        num_bins: int,
        center: float,
        scale: float,
        geometry: str = "asinh",
        smoothing: str = "fixed",
    ):
        super().__init__()
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer >= 2")
        if not math.isfinite(v_min) or not math.isfinite(v_max) or not 0 < v_max - v_min < math.inf:
            raise ValueError("v_min and v_max must define a finite, positive raw span")
        if not math.isfinite(center):
            raise ValueError("center must be finite")
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(1.5 * scale):
            raise ValueError("scale must be finite and positive with representable Gaussian sigma")
        if geometry not in ("uniform", "asinh"):
            raise ValueError("geometry must be 'uniform' or 'asinh'")
        if smoothing not in ("fixed", "local", "twohot"):
            raise ValueError("smoothing must be 'fixed', 'local', or 'twohot'")
        if geometry == "uniform":
            support = torch.linspace(v_min, v_max, num_bins, dtype=torch.float64)
        else:
            lo, hi = (v_min - center) / scale, (v_max - center) / scale
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError("asinh bounds relative to center/scale must be finite")
            coordinates = torch.linspace(math.asinh(lo), math.asinh(hi), num_bins, dtype=torch.float64)
            support = center + scale * coordinates.sinh()
        support[0], support[-1] = v_min, v_max
        if not bool(torch.isfinite(support).all()) or not bool((support[1:] > support[:-1]).all()):
            raise ValueError("support centers must be finite and distinct in float64")
        self.register_buffer("support", support)
        # Rescale only subnormal supports: individual probability*center
        # products can underflow even when their final expectation is representable.
        self._decode_scale = max(abs(v_min), abs(v_max))
        self.register_buffer(
            "_decode_support",
            support / self._decode_scale if self._decode_scale < torch.finfo(torch.float64).tiny else None,
        )
        self.num_bins = num_bins
        self.center = center
        self.scale = scale
        self.geometry = geometry
        self.smoothing = smoothing

    @staticmethod
    def _normal_integrals(lower, upper):
        """Return normal interval probability and its centered first moment."""
        # Beyond 40 standard deviations even tail probabilities underflow in
        # float64. Saturation avoids inf-inf in the density difference below.
        lower, upper = lower.clamp(-40.0, 40.0), upper.clamp(-40.0, 40.0)
        a, b = lower / math.sqrt(2.0), upper / math.sqrt(2.0)
        central = 0.5 * (torch.erf(b) - torch.erf(a))
        tails = 0.5 * (torch.erfc(a.abs()) - torch.erfc(b.abs())).abs()
        probability = torch.where((a <= 1.0) & (b >= -1.0), central, tails)
        # expm1 keeps phi(lower)-phi(upper) accurate for close or reflected
        # endpoints; subtracting two nearly equal densities would lose it.
        delta = 0.5 * (upper - lower) * (upper + lower)
        density = torch.exp(-0.5 * torch.minimum(lower.square(), upper.square())) / math.sqrt(2.0 * math.pi)
        moment = density * (-torch.expm1(-delta.abs())) * delta.sign()
        return probability, moment

    def project(self, targets: torch.Tensor) -> torch.Tensor:
        """Preserve target shape and append bins, clipping only in raw units.

        Each interval contributes its Gaussian probability to its two adjacent
        centers with integrated barycentric weights. Symmetric truncation at
        radius min(target-lo, hi-target) preserves the Gaussian's raw mean;
        linear interpolation then preserves it again. Work is O(batch*bins),
        without an iterative moment correction or minimum-KL projection.
        """
        if not targets.is_floating_point():
            raise ValueError("targets must have a floating dtype")
        support = self.support.to(dtype=torch.float64)
        target = targets.to(dtype=torch.float64).clamp(support[0], support[-1])
        bracket = torch.searchsorted(support, target.contiguous(), right=True).clamp(1, self.num_bins - 1)
        width = support[bracket] - support[bracket - 1]
        fraction = ((target - support[bracket - 1]) / width).clamp(0.0, 1.0)
        twohot = target.new_zeros((*target.shape, self.num_bins))
        twohot.scatter_(-1, (bracket - 1).unsqueeze(-1), (1.0 - fraction).unsqueeze(-1))
        twohot.scatter_add_(-1, bracket.unsqueeze(-1), fraction.unsqueeze(-1))
        if self.smoothing == "twohot":
            return twohot.to(dtype=targets.dtype)

        radius = torch.minimum(target - support[0], support[-1] - target)
        # Offsets, rather than target +/- radius in absolute coordinates,
        # retain symmetric truncation on heavily shifted supports.
        left = support[:-1] - target.unsqueeze(-1)
        right = support[1:] - target.unsqueeze(-1)
        lower = torch.maximum(left, -radius.unsqueeze(-1))
        upper = torch.minimum(right, radius.unsqueeze(-1))
        active = upper > lower
        lower = torch.where(active, lower, 0.0)
        upper = torch.where(active, upper, 0.0)
        safe_radius = torch.where(radius > 0, radius, 1.0).unsqueeze(-1)
        sigma = (0.75 * width if self.smoothing == "local" else torch.full_like(target, 1.5 * self.scale)).unsqueeze(-1)

        # A near-uniform truncated Gaussian needs a dimensionless series:
        # otherwise r/sigma and the unnormalized probabilities can underflow.
        # Through q^4, the omitted relative term is below 2.1e-20 for q<1e-3.
        q = safe_radius / sigma
        nearly_uniform = q < 1e-3
        integration_sigma = torch.where(nearly_uniform, safe_radius, sigma)
        probability, moment = self._normal_integrals(lower / integration_sigma, upper / integration_sigma)
        moment = moment * integration_sigma
        upper_mass = (moment - left * probability) / (support[1:] - support[:-1])
        if bool(nearly_uniform.any()):
            x, y = lower / safe_radius, upper / safe_radius
            q2 = torch.where(nearly_uniform, q, 0.0).square()
            q4 = q2.square()
            # Factored power differences stay accurate for narrow intersections.
            dx, xy = y - x, x * y
            x2, y2 = x.square(), y.square()
            fourth = x2.square() + x2 * y2 + y2.square()
            p2 = x2 + xy + y2
            p4 = fourth + xy * (x2 + y2)
            uniform_probability = dx * (1.0 - q2 * p2 / 6.0 + q4 * p4 / 40.0)
            uniform_moment = dx * (x + y) * (0.5 - q2 * (x2 + y2) / 8.0 + q4 * fourth / 48.0)
            # Form dimensionless barycentric masses before returning to raw
            # units; raw first moments can underflow on subnormal supports.
            uniform_upper_mass = (uniform_moment - (left / safe_radius) * uniform_probability) / (
                (support[1:] - support[:-1]) / safe_radius
            )
            probability = torch.where(nearly_uniform, uniform_probability, probability)
            upper_mass = torch.where(nearly_uniform, uniform_upper_mass, upper_mass)
        # Exact integrals lie in [0, P]; enforce only their roundoff envelope.
        upper_mass = torch.minimum(upper_mass.clamp_min(0.0), probability)
        mass = target.new_zeros((*target.shape, self.num_bins))
        mass[..., :-1] = probability - upper_mass
        mass[..., 1:] += upper_mass
        total = mass.sum(dim=-1, keepdim=True)
        labels = mass / torch.where(total > 0, total, 1.0)
        # If truncation lies in one raw interval, integration is exactly twohot
        # for every sigma. This also gives the stable zero-radius endpoint limit.
        in_bracket = radius <= torch.minimum(target - support[bracket - 1], support[bracket] - target)
        return torch.where(in_bracket.unsqueeze(-1), twohot, labels).to(dtype=targets.dtype)

    def probs_to_scalar(self, probs: torch.Tensor) -> torch.Tensor:
        """Decode the raw support expectation, retaining the probability dtype."""
        if probs.ndim == 0 or probs.shape[-1] != self.num_bins:
            raise ValueError("probabilities must have a final dimension of num_bins")
        if not probs.is_floating_point():
            raise ValueError("probabilities must have a floating dtype")
        support = self.support if self._decode_support is None else self._decode_support
        result = (probs.to(dtype=torch.float64) * support.to(dtype=torch.float64)).sum(dim=-1)
        if self._decode_support is not None:
            result = result * self._decode_scale
        return result.to(dtype=probs.dtype)

    def to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax logits and decode their raw expectation."""
        return self.probs_to_scalar(logits.softmax(dim=-1))

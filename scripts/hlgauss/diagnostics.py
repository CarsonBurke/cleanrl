"""CPU float64 label geometry and constant-state lambda-return diagnostics.

These are deterministic properties of a histogram target/decoder, not a model
training experiment or an estimate of policy performance.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch.overrides import TorchFunctionMode

from cleanrl.shared.hl_gauss import HistogramGaussian, HLGaussConfig


class _Float64SupportConstruction(TorchFunctionMode):
    """Give the shared constructor's linspace factories an explicit dtype.

    The scoped dispatch mode avoids both promoting already-rounded float32
    geometry and mutating PyTorch's process-global default dtype. All other
    operations retain the shared implementation and ordinary dtype propagation.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.linspace:
            kwargs = {**kwargs, "dtype": torch.float64}
        return func(*args, **kwargs)


def _validate_discount(gamma: float, gae_lambda: float) -> None:
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError("gae_lambda must be finite and in [0, 1]")


def lambda_return_fixed_point(
    decode_project: Callable[[float], float],
    true_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    *,
    raw_bounds: tuple[float, float] | None = None,
    max_iterations: int = 10000,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Iterate V <- D(T_lambda(V)), starting at the constant state's true value.

    The state's deterministic reward is (1-gamma)*true_value. Write
    a = gamma*(1-lambda)/(1-gamma*lambda); then
    T_lambda(V) = (1-a)*true_value + a*V. ``decode_project`` is D.

    Returned ``lambda_fixed_point_value`` is the final iterate (even if not
    converged); ``value_bias`` is V-true_value; ``advantage_bias`` is the
    *infinite-horizon GAE* T_lambda(V)-V, not the one-step TD residual.
    ``one_step_advantage_bias`` is (1-gamma)*(true_value-V).
    All those keys carry the ``lambda_fixed_point_`` prefix.
    ``residual`` is abs(D(T_lambda(V))-V), evaluated at the returned V;
    ``converged`` is 1 iff residual <= tolerance*max(1, abs(true_value), abs(V)).
    ``iterations`` counts applied updates. ``overflow`` is 1 iff the final
    T_lambda(V) is outside raw_bounds. ``iteration_overflow_fraction`` is the
    fraction of update inputs outside those bounds (zero when bounds omitted).
    ``bootstrap_coefficient`` is a. All keys carry the same prefix and all
    values are Python floats. Overflow describes target clipping, not Gaussian
    tail truncation. Nonfinite iterates raise rather than yielding invalid JSON.
    """
    _validate_discount(gamma, gae_lambda)
    if not math.isfinite(true_value):
        raise ValueError("true_value must be finite")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if raw_bounds is not None and not (
        math.isfinite(raw_bounds[0]) and math.isfinite(raw_bounds[1]) and raw_bounds[0] < raw_bounds[1]
    ):
        raise ValueError("raw_bounds must be finite and ordered")
    denominator = 1.0 - gamma * gae_lambda
    reward_weight = (1.0 - gamma) / denominator
    bootstrap = gamma * (1.0 - gae_lambda) / denominator

    def target(value: float) -> float:
        return reward_weight * true_value + bootstrap * value

    def overflow(value: float) -> float:
        return float(raw_bounds is not None and not raw_bounds[0] <= value <= raw_bounds[1])

    value = float(true_value)
    overflow_count = 0.0
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        update_target = target(value)
        overflow_count += overflow(update_target)
        updated = float(decode_project(update_target))
        if not math.isfinite(updated):
            raise ValueError("decode_project produced a nonfinite fixed-point iterate")
        delta = abs(updated - value)
        value = updated
        if delta <= tolerance * max(1.0, abs(true_value), abs(value)):
            break
    final_target = target(value)
    residual = abs(float(decode_project(final_target)) - value)
    if not math.isfinite(residual):
        raise ValueError("decode_project produced a nonfinite fixed-point residual")
    return {
        "lambda_fixed_point_value": value,
        "lambda_fixed_point_value_bias": value - true_value,
        "lambda_fixed_point_advantage_bias": final_target - value,
        "lambda_fixed_point_one_step_advantage_bias": (1.0 - gamma) * (true_value - value),
        "lambda_fixed_point_converged": float(residual <= tolerance * max(1.0, abs(true_value), abs(value))),
        "lambda_fixed_point_residual": residual,
        "lambda_fixed_point_iterations": float(iteration),
        "lambda_fixed_point_overflow": overflow(final_target),
        "lambda_fixed_point_iteration_overflow_fraction": overflow_count / iteration,
        "lambda_fixed_point_bootstrap_coefficient": bootstrap,
    }


def analyze_support(
    config: HLGaussConfig,
    target_mean: float,
    target_std: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> dict[str, float]:
    """Measure label geometry at y=target_mean; target_std must be positive.

    Uses the shared HistogramGaussian projector/decoder on CPU in float64,
    without model training, file writes, or global Torch settings changes.
    No distributional integration over targets is implied: target_std sets a
    local contrast scale, while fixed-point analysis uses a deterministic state.

    Exact output definitions (raw units unless stated otherwise):
      smoothing_sigma_coordinate: the Gaussian's pre-truncation sigma in its
        integration coordinate, *not* necessarily the raw grid coordinate.
      smoothing_sigma_raw_local: sigma_coordinate / symlog'(clip(y)) for a
        symlog Gaussian, otherwise sigma_coordinate. This tangent bandwidth is
        not the standard deviation of a decoded/transformed Gaussian.
      smoothing_sigma_raw_minus/plus: distances from clip(y) to inverse-transform
        of coordinate(y) minus/plus sigma; these expose symlog asymmetry.
      effective_sigma_over_target_std: smoothing_sigma_raw_local / target_std.
      local_raw_grid_spacing: distance between raw centers bracketing clip(y);
        nearest end interval outside the centers, right interval at a center.
      project_decode_value/bias: D(y), D(y)-y, where D=decode(project(.)); includes
        clipping and truncation, with no assumption that narrower sigma helps.
      project_decode_derivative: [D(y+h)-D(y-h)]/(2h), a raw-target derivative.
      derivative_step_raw: h=max(1e-4*min(std, spacing, local_sigma),
        32*ulp(max(1, abs(y)))). At clipping/transform kinks this is a symmetric
        finite-difference diagnostic rather than a one-sided derivative.
      label_entropy: -sum p(y)*log(max(p(y),1e-300)), in nats; zero masses
        contribute 0 and representable sub-floor tails use the stated log floor.
      label_contrast_kl: half the sum of both directional KLs between p(y-std)
        and p(y+std). Logarithms use normalized max(p, 1e-300) masses, preventing
        numerical-zero tails from producing infinities; this is a floored KL.
      ce_label_gradient_contrast_norm: ||p(y+std)-p(y-std)||_2, the difference of
        categorical CE logit gradients for identical predictions.
      target_location_fisher_information: sum_i (dp_i/dy)^2/max(p_i(y),1e-300),
        with dp/dy the central difference at h. Units are inverse raw-value^2;
        squaring is performed after division by sqrt(mass) to preserve tails.
      target_overflow: 1 iff y is outside the configured raw clipping bounds.
      contrast_overflow_fraction: fraction of the two contrast targets outside
        those bounds. Boundary saturation therefore remains visible.
    Fixed-point fields are defined by ``lambda_return_fixed_point``. Every
    returned number is a finite Python float, suitable for strict JSON. Extreme
    settings whose arithmetic is nonfinite raise ValueError instead.
    """
    _validate_discount(gamma, gae_lambda)
    if not math.isfinite(target_mean) or not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("target_mean must be finite and target_std must be finite and positive")
    with _Float64SupportConstruction():
        head = HistogramGaussian(config, device="cpu")
    clipped_mean = min(max(target_mean, config.v_min), config.v_max)
    sigma_coordinate = head._sqrt_two_sigma / math.sqrt(2.0)
    coordinate_mean = clipped_mean
    if config.transform == "symlog":
        coordinate_mean = math.copysign(math.log1p(abs(clipped_mean)), clipped_mean)
    sigma_raw = sigma_coordinate * (1.0 + abs(clipped_mean)) if config.transform == "symlog" else sigma_coordinate

    def inverse(value: float) -> float:
        try:
            return math.copysign(math.expm1(abs(value)), value) if config.transform == "symlog" else value
        except OverflowError as error:
            raise ValueError("raw sigma offsets exceed finite float64 range") from error

    index = int(torch.searchsorted(head.support, torch.tensor(clipped_mean, dtype=torch.float64), right=True))
    index = min(max(index, 1), config.num_bins - 1)
    spacing = float(head.support[index] - head.support[index - 1])
    h = max(1e-4 * min(target_std, spacing, sigma_raw), 32 * math.ulp(max(1.0, abs(target_mean))))
    targets = torch.tensor(
        [target_mean, target_mean - h, target_mean + h, target_mean - target_std, target_mean + target_std],
        dtype=torch.float64,
        device="cpu",
    )
    with torch.no_grad():
        labels = head.project(targets)
        decoded = head.probs_to_scalar(labels)
        positive = labels.clamp_min(1e-300)
        positive = positive / positive.sum(dim=-1, keepdim=True)
        minus, plus = positive[3], positive[4]
        symmetric_kl = 0.5 * ((minus - plus) * (minus.log() - plus.log())).sum()
        derivative = (labels[2] - labels[1]) / (2.0 * h)
        fisher = (derivative / labels[0].clamp_min(1e-300).sqrt()).square().sum()

        def decode_project(value: float) -> float:
            return float(head.probs_to_scalar(head.project(torch.tensor(value, dtype=torch.float64, device="cpu"))))

        fixed_point = lambda_return_fixed_point(
            decode_project, target_mean, gamma, gae_lambda, raw_bounds=(config.v_min, config.v_max)
        )
    result = {
        "smoothing_sigma_coordinate": sigma_coordinate,
        "smoothing_sigma_raw_local": sigma_raw,
        "smoothing_sigma_raw_minus": clipped_mean - inverse(coordinate_mean - sigma_coordinate),
        "smoothing_sigma_raw_plus": inverse(coordinate_mean + sigma_coordinate) - clipped_mean,
        "effective_sigma_over_target_std": sigma_raw / target_std,
        "local_raw_grid_spacing": spacing,
        "project_decode_value": float(decoded[0]),
        "project_decode_bias": float(decoded[0]) - target_mean,
        "project_decode_derivative": float((decoded[2] - decoded[1]) / (2.0 * h)),
        "label_entropy": float(-(labels[0] * labels[0].clamp_min(1e-300).log()).sum()),
        "label_contrast_kl": float(symmetric_kl),
        "ce_label_gradient_contrast_norm": float(torch.linalg.vector_norm(labels[4] - labels[3])),
        "target_location_fisher_information": float(fisher),
        "derivative_step_raw": h,
        "target_overflow": float(not config.v_min <= target_mean <= config.v_max),
        "contrast_overflow_fraction": 0.5
        * sum(not config.v_min <= y <= config.v_max for y in (target_mean - target_std, target_mean + target_std)),
        **fixed_point,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("support diagnostics produced nonfinite arithmetic")
    return result

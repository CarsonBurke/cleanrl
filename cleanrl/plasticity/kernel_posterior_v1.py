"""A new fixed-function-basis model, not the original moving-network covariance model.

GaussianPosterior is exact Gaussian linear regression for hazard=0. For hazard>0,
its latent transition redraws all coefficients from N(0, prior*I) with probability
hazard; the resulting two-component mixture is moment-matched to one Gaussian
BEFORE conditioning. Conditioning that Gaussian is exact, but filtering the reset
mixture is not. Conditional squared-error optimality is only under these model
assumptions, not a claim of universally optimal prediction.

TangentFeatures freezes the v2 initialization and projects its exact output
Jacobian. Its model is f(x)=phi(x)'w with ZERO prior mean, not f(theta0,x)+phi(x)'w.
No label, fitted normalization, moving linearization, or data-dependent basis is
involved. The caller supplies causal positive observation variance, compiles the
hot methods, captures/resets graph state, and chunks feature extraction.
"""

import hashlib
import math

import torch

from cleanrl.plasticity import network_bayes_stream_v2 as v2


class GaussianPosterior:
    """Batched coefficient posteriors with aligned (not Cartesian) configurations.

    Each configuration supplies positive ``prior`` (per-feature variance) and
    ``hazard`` in [0,1]. x is [D], y is scalar, and observation_variance is
    strictly positive scalar or [K], all CUDA tensors. Forecasts have shape [K]. State
    uses FP32 and requires compatible FP32 inputs. P0 is diagonal, not a dense
    process-noise tensor. No variance clipping or confidence floor is applied.
    Storage and each update are O(K*D**2); predict is O(K*D**2) without allocating
    a full next-prior covariance. Floating-point precision remains a limitation
    of covariance-form conditioning, especially for nearly noiseless streams.
    """

    def __init__(self, input_dim, configs, device):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GaussianPosterior requires CUDA")
        if input_dim <= 0 or not configs:
            raise ValueError("input_dim and the configuration count must be positive")
        priors = [float(config["prior"]) for config in configs]
        hazards = [float(config["hazard"]) for config in configs]
        if any(not math.isfinite(p) or p <= 0 for p in priors):
            raise ValueError("prior must be finite and positive")
        if any(not math.isfinite(h) or not 0 <= h <= 1 for h in hazards):
            raise ValueError("hazard must be a finite probability")
        self.input_dim = input_dim
        self.prior = torch.tensor(priors, device=device, dtype=torch.float32)
        self.hazard = torch.tensor(hazards, device=device, dtype=torch.float32)
        self.retention = 1 - self.hazard
        self.reset_variance = self.hazard * self.prior
        self.mixture_variance = self.hazard * self.retention
        self.mean = torch.zeros(len(configs), input_dim, device=device, dtype=torch.float32)
        self.cov = torch.zeros(len(configs), input_dim, input_dim, device=device, dtype=torch.float32)
        self.cov.diagonal(dim1=-2, dim2=-1).copy_(self.prior[:, None])
        self.mutable = [self.mean, self.cov]

    def state_tensors(self):
        """Stable identities of ALL mutable tensors, for capture warmup/reset."""
        return self.mutable

    def mean_weights(self):
        """Return [K,D] NEXT-prior coefficients, including the redraw transition."""
        return self.retention[:, None] * self.mean

    def _forecast(self, x, observation_variance):
        old_prediction = (self.mean * x).sum(-1)
        prediction = self.retention * old_prediction
        # u = Pprior*x; no full Pprior is needed for a forecast.
        u = torch.matmul(self.cov, x) * self.retention[:, None]
        u = u + self.reset_variance[:, None] * x
        u = u + (self.mixture_variance * old_prediction)[:, None] * self.mean
        variance = observation_variance + (x * u).sum(-1)
        return prediction, variance, u

    @torch.no_grad()
    def predict(self, x, observation_variance):
        """Non-mutating pre-label forecast, using the same prior as update."""
        prediction, variance, _ = self._forecast(x, observation_variance)
        return prediction, variance

    @torch.no_grad()
    def update(self, x, y, observation_variance):
        """Return pre-label (mean, total variance), then condition on scalar y.

        m-=(1-h)m; P-=(1-h)P+h*P0+h*(1-h)*m*m'.
        S=R+x'P-x; m+=m-+P-x*(y-x'm-)/S; P+=P--(P-x)(P-x)'/S.
        The symmetric scaled outer product uses no full identity/Q allocation.
        """
        prediction, variance, u = self._forecast(x, observation_variance)
        self.cov.mul_(self.retention[:, None, None])
        self.cov.add_(self.mixture_variance[:, None, None]
                      * self.mean[:, :, None] * self.mean[:, None, :])
        self.cov.diagonal(dim1=-2, dim2=-1).add_(self.reset_variance[:, None])
        scaled_u = u / variance.sqrt()[:, None]
        self.cov.sub_(scaled_u[:, :, None] * scaled_u[:, None, :])
        self.mean.mul_(self.retention[:, None])
        self.mean.add_(u * ((y - prediction) / variance)[:, None])
        return prediction, variance


def _namespace_seed(seed, namespace):
    digest = hashlib.sha256(f"kernel_posterior_v1:{namespace}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


class TangentFeatures:
    """Frozen raw-plus-projected-tangent features, preserving every raw coordinate.

    phi(x) = cat(x/sqrt(input_dim), J_theta0(x) R/sqrt(projected_dim)).
    R is stored layerwise as [output, bias-augmented fan_in, projected_dim], with
    independent N(0,1/fan_in) entries. Bias counts in fan_in, as in the v2
    covariance row prior. Weight and direction draws have separate deterministic
    seed namespaces and do not consume the global generator. theta0 is drawn by
    the ORIGINAL v2 initializer, not trained. The prior function mean is zero;
    the frozen network's own output is deliberately NOT included.

    transform accepts FP32 CUDA [N,input_dim] and returns [N,output_dim]. It never
    constructs [N,total_parameter_count] Jacobians: layerwise contractions use
    O(N*hidden*projected_dim) working memory, and stored directions cost
    O(parameter_count*projected_dim). The caller controls N through chunking.
    """

    def __init__(self, input_dim, projected_dim=256, hidden=64, seed=1, device="cuda"):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("TangentFeatures requires CUDA")
        if min(input_dim, projected_dim, hidden) <= 0:
            raise ValueError("input_dim, projected_dim and hidden must be positive")
        self.input_dim = input_dim
        self.projected_dim = projected_dim
        self.hidden = hidden
        self.output_dim = input_dim + projected_dim
        weight_seed = _namespace_seed(seed, "weights")
        direction_seed = _namespace_seed(seed, "directions")
        weights_gen = torch.Generator(device=device).manual_seed(weight_seed)
        directions_gen = torch.Generator(device=device).manual_seed(direction_seed)
        self.weights = tuple(v2.init_weights(v2.Args(input_dim=input_dim, hidden=hidden), weights_gen, device))
        self.directions = tuple(
            torch.randn(*weight.shape, projected_dim, generator=directions_gen, device=device)
            / math.sqrt(weight.shape[-1]) for weight in self.weights
        )
        self.config = {
            "model": "fixed_raw_plus_projected_v2_tangent_v1",
            "input_dim": input_dim, "projected_dim": projected_dim, "hidden": hidden,
            "output_dim": self.output_dim, "seed": seed,
            "weight_seed": weight_seed, "direction_seed": direction_seed,
            "direction_variance": "1 / bias_augmented_fan_in",
            "raw_scale": 1 / math.sqrt(input_dim),
            "tangent_scale": 1 / math.sqrt(projected_dim),
            "prior_function_mean": 0.0,
        }

    @torch.no_grad()
    def transform(self, batch_x):
        h1, h2, _ = v2.forward(tuple(weight.unsqueeze(0) for weight in self.weights), batch_x)
        h1, h2 = h1[0], h2[0]
        sensitivity2 = self.weights[2][0, :-1] * (1 - h2.square())
        sensitivity1 = (sensitivity2 @ self.weights[1][:, :-1]) * (1 - h1.square())
        projected = h2 @ self.directions[2][0, :-1] + self.directions[2][0, -1]
        for inputs, sensitivity, direction in (
            (batch_x, sensitivity1, self.directions[0]),
            (h1, sensitivity2, self.directions[1]),
        ):
            # First contract incoming coordinates, then output sensitivities.
            # Intermediate is [N,hidden,projected_dim], not a parameter Jacobian.
            directional_input = torch.einsum("ni,oip->nop", inputs, direction[:, :-1])
            directional_input = directional_input + direction[:, -1]
            projected = projected + (sensitivity[:, :, None] * directional_input).sum(1)
        return torch.cat((batch_x / math.sqrt(self.input_dim),
                          projected / math.sqrt(self.projected_dim)), dim=-1)

    def metadata(self):
        """JSON-safe configuration and exact tensor fingerprint (host sync here only).

        Call outside compilation/capture. Includes shape/dtype and little-endian
        host tensor bytes of both initialization and random directions, in layer
        order, so the recorded basis is auditable rather than seed-only.
        """
        digest = hashlib.sha256()
        for name, tensors in (("weights", self.weights), ("directions", self.directions)):
            for layer, tensor in enumerate(tensors):
                digest.update(f"{name}:{layer}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
                digest.update(tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
        return {**self.config, "tensor_sha256": digest.hexdigest()}

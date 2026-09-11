"""Fixed-panel FP32 tanh learners with manual Adam and prior-history JS gates.

All configurations receive the same features, centered target and valid mask. The
categorical models change the head/loss, not the information set. Uniform two-hot
labels reuse the shared raw-space projector and preserve the target expectation.
No autograd, host reads, or dynamic sample selection occurs in ``step``.
"""

import math
from dataclasses import dataclass

import torch

from cleanrl.plasticity.panel_hd_gate import Net
from cleanrl.shared import runtime
from cleanrl.shared.two_hot import DreamerTwoHotSupport


FAMILIES = (
    "scalar_adam",
    "scalar_js",
    "categorical_ce",
    "categorical_mse",
    "categorical_ce_js",
)


@dataclass(frozen=True)
class Config:
    family: str
    lr: float


class Learner:
    """Independent configurations vectorized across a leading model dimension.

    ``prediction`` is owned storage overwritten with the *pre-update* forecasts.
    ``steps`` counts consumed bars; ``adam_steps`` counts nonempty bars, matching
    the scalar baseline's skip of entirely invalid cross-sections. Both clocks
    are scalar int64 CUDA tensors. Zero-valid bars do not advance optimizer or
    evidence state. All tensor state needed to restore capture/warmup is returned
    by ``state_tensors``; support and configuration tensors are immutable.
    """

    def __init__(
        self,
        input_dim: int,
        width: int,
        mu: float | torch.Tensor,
        configs: list[Config] | tuple[Config, ...],
        device: str | torch.device,
        bins: int = 33,
        seed: int = 1,
        kappa: float = 1.0,
        num_samples: int = 200,
    ):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CUDA required; there is no CPU learner fallback")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if min(input_dim, width, num_samples) < 1:
            raise ValueError("input_dim, width and num_samples must be positive")
        if not configs or any(c.family not in FAMILIES for c in configs):
            raise ValueError("configs must contain known families")
        if any(not math.isfinite(c.lr) or c.lr <= 0 for c in configs):
            raise ValueError("learning rates must be positive and finite")
        if not math.isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
        self.configs = tuple(configs)
        self.output_names = tuple(f"{c.family}_{c.lr:g}" for c in configs)
        if len(set(self.output_names)) != len(configs):
            raise ValueError("configurations must have distinct output names")
        self.kappa = kappa
        self.num_samples = num_samples
        self.bins = bins
        self.device = device
        count = len(configs)
        self.projector = DreamerTwoHotSupport(bins, 12.5, device, spacing="linear")
        raw_support = self.projector.support + 12.5
        self.projector.support.add_(12.5 - torch.as_tensor(mu, dtype=torch.float32, device=device))
        self.support = self.projector.support

        # The frozen baseline creates nn.Linear on CPU after seed=1 and then
        # transfers to CUDA. Reuse that exact initializer, without altering the
        # caller's RNG stream or executing a CPU forward/backward pass.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            baseline = Net(input_dim, width)
        self.weights = (
            baseline.l1.weight.detach().to(device).unsqueeze(0).repeat(count, 1, 1),
            baseline.l2.weight.detach().to(device).unsqueeze(0).repeat(count, 1, 1),
            torch.zeros((count, bins, width), dtype=torch.float32, device=device),
        )
        self.biases = (
            baseline.l1.bias.detach().to(device).unsqueeze(0).repeat(count, 1),
            baseline.l2.bias.detach().to(device).unsqueeze(0).repeat(count, 1),
            torch.zeros((count, bins), dtype=torch.float32, device=device),
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        category_weight = torch.randn((bins, width), generator=generator, device=device) * (1e-3 / math.sqrt(width))
        for index, config in enumerate(configs):
            if config.family.startswith("scalar"):
                self.weights[2][index, :1].copy_(baseline.l3.weight.detach())
                self.biases[2][index, :1].copy_(baseline.l3.bias.detach())
            else:
                self.weights[2][index].copy_(category_weight)
                self.biases[2][index].copy_(-raw_support)
        self.parameters = tuple(p for pair in zip(self.weights, self.biases) for p in pair)
        self.first_moments = tuple(torch.zeros_like(p) for p in self.parameters)
        self.second_moments = tuple(torch.zeros_like(p) for p in self.parameters)
        self.s1 = tuple(torch.zeros_like(p) for p in self.parameters)
        self.s2 = tuple(torch.zeros_like(p) for p in self.parameters)
        self.steps = torch.zeros((), dtype=torch.int64, device=device)
        self.adam_steps = torch.zeros((), dtype=torch.int64, device=device)
        self.prediction = torch.zeros((count, num_samples), dtype=torch.float32, device=device)
        self.scalar = torch.tensor([c.family.startswith("scalar") for c in configs], device=device)
        self.cross_entropy = torch.tensor([c.family in ("categorical_ce", "categorical_ce_js") for c in configs], device=device)
        self.js = torch.tensor([c.family in ("scalar_js", "categorical_ce_js") for c in configs], device=device)
        self.learning_rates = torch.tensor([c.lr for c in configs], dtype=torch.float32, device=device)
        self.scalar_coordinate = torch.arange(bins, device=device) == 0

    def state_tensors(self) -> tuple[torch.Tensor, ...]:
        """Stable, nonduplicated mutable storage order for capture and checkpoints."""
        return (
            *self.parameters,
            *self.first_moments,
            *self.second_moments,
            *self.s1,
            *self.s2,
            self.steps,
            self.adam_steps,
            self.prediction,
        )

    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Write forecasts, then consume one fixed-shape cross-sectional bar.

        For a categorical mean m=sum_j p_j s_j, dm/dlogit_j=p_j(s_j-m).
        Thus MSE uses 2(m-y)p_j(s_j-m), while CE uses p_j-twohot_j.
        Both divide by the valid sample count, not by N or the bin count.
        """
        w1, w2, w3 = self.weights
        b1, b2, b3 = self.biases
        h1 = torch.tanh(torch.matmul(x, w1.transpose(1, 2)) + b1[:, None, :])
        h2 = torch.tanh(torch.bmm(h1, w2.transpose(1, 2)) + b2[:, None, :])
        logits = torch.bmm(h2, w3.transpose(1, 2)) + b3[:, None, :]
        probabilities = logits.softmax(-1)
        category_mean = (probabilities * self.support).sum(-1)
        prediction = torch.where(self.scalar[:, None], logits[:, :, 0], category_mean)
        self.prediction.copy_(prediction)

        valid_count = mask.sum()
        active = valid_count > 0
        normalizer = valid_count.clamp_min(1).to(torch.float32)
        # Invalid labels have no defined value in the panel. Sanitize before the
        # projection/subtraction, then zero whole per-sample loss derivatives.
        target = torch.where(mask, y, 0.0)
        labels = self.projector.project(target)
        residual = prediction - target
        scalar_gradient = (2.0 * residual[:, :, None]) * self.scalar_coordinate
        mean_gradient = 2.0 * residual[:, :, None] * probabilities * (self.support - category_mean[:, :, None])
        category_gradient = torch.where(self.cross_entropy[:, None, None], probabilities - labels, mean_gradient)
        derivative = torch.where(self.scalar[:, None, None], scalar_gradient, category_gradient)
        dz3 = torch.where(mask[None, :, None], derivative / normalizer, 0.0)
        gw3 = torch.bmm(dz3.transpose(1, 2), h2)
        gb3 = dz3.sum(1)
        dz2 = torch.bmm(dz3, w3) * (1.0 - h2.square())
        gw2 = torch.bmm(dz2.transpose(1, 2), h1)
        gb2 = dz2.sum(1)
        dz1 = torch.bmm(dz2, w2) * (1.0 - h1.square())
        gw1 = torch.matmul(dz1.transpose(1, 2), x)
        gb1 = dz1.sum(1)
        gradients = (gw1, gb1, gw2, gb2, gw3, gb3)

        self.steps.add_(1)
        self.adam_steps.add_(active.to(torch.int64))
        # Scalar double precision matches Adam's Python bias corrections before
        # their cast into FP32 tensor arithmetic. Parameters/moments stay FP32.
        clock = self.adam_steps.clamp_min(1).to(torch.float64)
        correction1 = (1.0 - 0.9 ** clock).to(torch.float32)
        correction2 = (1.0 - 0.999 ** clock).to(torch.float32)
        for parameter, gradient, first, second, s1, s2 in zip(
            self.parameters, gradients, self.first_moments, self.second_moments, self.s1, self.s2
        ):
            shape = (-1,) + (1,) * (parameter.ndim - 1)
            first.copy_(torch.where(active, first * 0.9 + gradient * 0.1, first))
            second.copy_(torch.where(active, second * 0.999 + gradient.square() * 0.001, second))
            adam = (first / correction1) / ((second / correction2).sqrt() + 1e-8)
            # Previous-history positive-part gate. Replacing only an exact zero
            # denominator avoids 0/0 without perturbing any nonzero evidence.
            squared_sum = s1.square()
            denominator = torch.where(squared_sum == 0, 1.0, squared_sum)
            gate = torch.where(squared_sum == 0, 0.0, (1.0 - self.kappa * s2 / denominator).clamp_min(0.0))
            update = torch.where(self.js.view(shape), adam * gate, adam)
            parameter.sub_(torch.where(active, self.learning_rates.view(shape) * update, 0.0))
            s1.add_(gradient)
            s2.add_(gradient.square())
        return self.prediction

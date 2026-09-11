"""Four fixed-panel JS arms, delegating the frozen v1 learning rules.

The only new algorithm is categorical-mean MSE with v1's prior-history JS gate:
construct the MSE bank, then enable its owned ``js`` tensor once at setup. CE and
MSE categorical banks have exactly the same initialization, support and gate;
only their loss differs. Scientific configs retain the actual public family,
while group metadata records the constructor family and setup override.

Scalar capacity matching counts active parameters, not v1's inactive padded
head coordinates. Allocated parameter counts disclose that padding separately;
matching active capacity does not match FLOPs, optimizer storage or wall time.
No frozen implementation, global family registry or optimizer code is changed.
"""

import math

import torch

from cleanrl.plasticity import panel_distributional_model_v1 as frozen
from cleanrl.plasticity.panel_distributional_model_v1 import Config


FAMILIES = ("scalar_js", "scalar_budget_js", "categorical_mse_js", "categorical_ce_js")


def scalar_parameter_count(input_dim: int, width: int) -> int:
    return width * (input_dim + width + 3) + 1


def categorical_parameter_count(input_dim: int, width: int, bins: int) -> int:
    return input_dim * width + width * width + 2 * width + bins * (width + 1)


def capacity_matched_width(input_dim: int, width: int, bins: int) -> int:
    """Smallest positive scalar width meeting the categorical active budget."""
    if min(input_dim, width, bins) < 1:
        raise ValueError("input_dim, width and bins must be positive")
    budget = categorical_parameter_count(input_dim, width, bins)
    coefficient = input_dim + 3
    candidate = max(1, (math.isqrt(coefficient * coefficient + 4 * (budget - 1)) - coefficient) // 2)
    if scalar_parameter_count(input_dim, candidate) < budget:
        candidate += 1
    return candidate


class Learner:
    """Independent family banks with fixed CUDA scatter into owned forecasts.

    Config order and family subsets are supported without changing bank-local
    initialization. ``clocks`` exposes every consumed/optimizer clock pair;
    the first pair is also available through the generic runner aliases.
    ``state_tensors`` includes all bank state and the distinct gathered output,
    so graph warmup, capture and checkpoint restoration share one state order.
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
        self.configs = tuple(configs)
        if not self.configs or any(c.family not in FAMILIES for c in self.configs):
            raise ValueError("configs must contain known frontier families")
        self.output_names = tuple(f"{c.family}_{c.lr:g}" for c in self.configs)
        if len(set(self.output_names)) != len(self.configs):
            raise ValueError("configurations must have distinct output names")
        budget_width = capacity_matched_width(input_dim, width, bins)
        raw_families = ("scalar_js", "scalar_js", "categorical_mse", "categorical_ce_js")
        groups, indices, metadata = [], [], []
        self.widths = [0] * len(self.configs)
        self.effective_parameter_counts = [0] * len(self.configs)
        for family, raw_family in zip(FAMILIES, raw_families):
            positions = tuple(i for i, config in enumerate(self.configs) if config.family == family)
            if not positions:
                continue
            group_width = budget_width if family == "scalar_budget_js" else width
            public_configs = tuple(self.configs[i] for i in positions)
            bank = frozen.Learner(
                input_dim, group_width, mu,
                [Config(raw_family, config.lr) for config in public_configs], device,
                bins=bins, seed=seed, kappa=kappa, num_samples=num_samples,
            )
            if family == "categorical_mse_js":
                # Setup only: reuse the existing MSE derivative and existing JS
                # update without touching frozen code or any other owned bank.
                bank.js.fill_(True)
            bank.configs = public_configs
            bank.output_names = tuple(self.output_names[i] for i in positions)
            effective = (scalar_parameter_count(input_dim, group_width) if family.startswith("scalar")
                         else categorical_parameter_count(input_dim, group_width, bins))
            for position in positions:
                self.widths[position] = group_width
                self.effective_parameter_counts[position] = effective
            groups.append(bank)
            indices.append(torch.tensor(positions, dtype=torch.int64, device=bank.device))
            metadata.append({
                "family": family,
                "raw_family": raw_family,
                "js_enabled": True,
                "js_setup_override": family == "categorical_mse_js",
                "width": group_width,
                "config_indices": positions,
                "effective_parameters_per_config": effective,
                "allocated_parameters_per_config": sum(p[0].numel() for p in bank.parameters),
            })
        self.groups = tuple(groups)
        self.group_indices = tuple(indices)
        self.group_metadata = tuple(metadata)
        self.device = self.groups[0].device
        self.bins, self.num_samples, self.kappa = bins, num_samples, kappa
        self.support = self.groups[0].support
        self.clocks = tuple((bank.steps, bank.adam_steps) for bank in self.groups)
        self.steps, self.adam_steps = self.clocks[0]
        self.parameters = tuple(parameter for bank in self.groups for parameter in bank.parameters)
        self.allocated_parameter_count = sum(parameter.numel() for parameter in self.parameters)
        self.prediction = torch.zeros((len(self.configs), num_samples), dtype=torch.float32, device=self.device)

    def state_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*tuple(tensor for bank in self.groups for tensor in bank.state_tensors()), self.prediction)

    @torch.no_grad()
    def candidate_finite(self) -> torch.Tensor:
        """Checkpoint health of all mutable per-candidate state, in config order."""
        result = torch.empty(len(self.configs), dtype=torch.bool, device=self.device)
        for bank, indices in zip(self.groups, self.group_indices):
            finite = torch.stack([
                torch.isfinite(tensor).flatten(1).all(1)
                for tensor in bank.state_tensors() if tensor.ndim > 0
            ]).all(0)
            result.index_copy_(0, indices, finite)
        return result & torch.isfinite(self.prediction).all(1)

    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for bank, indices in zip(self.groups, self.group_indices):
            self.prediction.index_copy_(0, indices, bank.step(x, y, mask))
        return self.prediction

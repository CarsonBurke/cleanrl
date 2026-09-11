"""Frozen dense controls and independent four-expert conditional MSE learners.

All training is fixed-shape CUDA tensor algebra with manual backward. Expert
responsibilities are prediction-error credit, not a density/variance proxy.
"""

import math

import torch

from cleanrl.plasticity.panel_distributional_model_v1 import Config, Learner as DenseLearner
from cleanrl.plasticity.panel_hd_gate import Net
from cleanrl.shared import runtime


FAMILIES = (
    "scalar_adam",
    "scalar_js",
    "categorical_ce_js",
    "uniform_moe_js",
    "constant_moe_js",
    "state_moe_adam",
    "state_moe_js",
)
DENSE_FAMILIES = FAMILIES[:3]
EXPERTS = 4


class _AdamJS:
    """One batched parameter group sharing a configuration axis and bar clock."""

    def __init__(self, parameters, configs, kappa, device):
        self.parameters = tuple(parameters)
        self.first_moments = tuple(torch.zeros_like(p) for p in parameters)
        self.second_moments = tuple(torch.zeros_like(p) for p in parameters)
        self.s1 = tuple(torch.zeros_like(p) for p in parameters)
        self.s2 = tuple(torch.zeros_like(p) for p in parameters)
        self.learning_rates = torch.tensor([c.lr for c in configs], dtype=torch.float32, device=device)
        self.js = torch.tensor([c.family.endswith("_js") for c in configs], device=device)
        self.kappa = kappa

    def state_tensors(self):
        return (*self.parameters, *self.first_moments, *self.second_moments, *self.s1, *self.s2)

    def update(self, gradients, active, correction1, correction2):
        for parameter, gradient, first, second, s1, s2 in zip(
            self.parameters, gradients, self.first_moments, self.second_moments, self.s1, self.s2
        ):
            shape = (-1,) + (1,) * (parameter.ndim - 1)
            first.copy_(torch.where(active, first * 0.9 + gradient * 0.1, first))
            second.copy_(torch.where(active, second * 0.999 + gradient.square() * 0.001, second))
            adam = (first / correction1) / ((second / correction2).sqrt() + 1e-8)
            squared_sum = s1.square()
            denominator = torch.where(squared_sum == 0, 1.0, squared_sum)
            gate = torch.where(squared_sum == 0, 0.0, (1.0 - self.kappa * s2 / denominator).clamp_min(0.0))
            update = torch.where(self.js.view(shape), adam * gate, adam)
            parameter.sub_(torch.where(active, self.learning_rates.view(shape) * update, 0.0))
            s1.add_(gradient)
            s2.add_(gradient.square())


class Learner:
    """Seven families, preserving arbitrary caller order and owned predictions.

    Expert tensors are [MoE configurations, 4, output, input] (biases omit input).
    Only active routers are allocated: constant routing trains biases, state
    routing trains weights and biases, uniform routing has neither. CPU is used
    solely for the frozen nn.Linear initializer; there is no CPU training path.
    """

    def __init__(self, input_dim, width, mu, configs, device, bins=33, seed=1, kappa=1.0, num_samples=200):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CUDA required; there is no CPU learner fallback")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if min(input_dim, width, num_samples) < 1 or width % EXPERTS:
            raise ValueError("positive dimensions and width divisible by four required")
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
        self.device, self.num_samples, self.bins, self.kappa = device, num_samples, bins, kappa
        self.expert_width = width // EXPERTS
        dense_positions = [i for i, c in enumerate(configs) if c.family in DENSE_FAMILIES]
        moe_positions = [i for i, c in enumerate(configs) if c.family not in DENSE_FAMILIES]
        self.dense_indices = torch.tensor(dense_positions, dtype=torch.int64, device=device)
        self.moe_indices = torch.tensor(moe_positions, dtype=torch.int64, device=device)
        dense_configs = [configs[i] for i in dense_positions]
        self.moe_configs = tuple(configs[i] for i in moe_positions)
        self.dense = DenseLearner(input_dim, width, mu, dense_configs, device, bins, seed, kappa, num_samples) if dense_configs else None
        self.moe_count = len(moe_positions)
        bias_positions = [i for i, c in enumerate(self.moe_configs) if c.family != "uniform_moe_js"]
        weight_positions = [i for i, c in enumerate(self.moe_configs) if c.family.startswith("state_")]
        self.router_bias_indices = torch.tensor(bias_positions, dtype=torch.int64, device=device)
        self.router_weight_indices = torch.tensor(weight_positions, dtype=torch.int64, device=device)
        self.experts = None
        self.router_bias = None
        self.router_weight = None
        if self.moe_count:
            with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                initializers = []
                for expert in range(EXPERTS):
                    torch.random.default_generator.manual_seed(seed + expert * 1009)
                    initializers.append(Net(input_dim, self.expert_width))
            parameters = tuple(
                torch.stack([tuple(net.parameters())[j].detach() for net in initializers])
                .to(device).unsqueeze(0).repeat(self.moe_count, *([1] * (tuple(initializers[0].parameters())[j].ndim + 1)))
                for j in range(6)
            )
            self.experts = _AdamJS(parameters, self.moe_configs, kappa, device)
            if bias_positions:
                self.router_bias = _AdamJS(
                    (torch.zeros((len(bias_positions), EXPERTS), device=device),),
                    [self.moe_configs[i] for i in bias_positions], kappa, device,
                )
            if weight_positions:
                self.router_weight = _AdamJS(
                    (torch.zeros((len(weight_positions), EXPERTS, input_dim), device=device),),
                    [self.moe_configs[i] for i in weight_positions], kappa, device,
                )
        self.groups = tuple(g for g in (self.experts, self.router_bias, self.router_weight) if g is not None)
        self.parameters = (*(self.dense.parameters if self.dense is not None else ()), *(p for g in self.groups for p in g.parameters))
        self.moe_steps = torch.zeros((), dtype=torch.int64, device=device)
        self.moe_adam_steps = torch.zeros((), dtype=torch.int64, device=device)
        self.steps = self.dense.steps if self.dense is not None else self.moe_steps
        self.adam_steps = self.dense.adam_steps if self.dense is not None else self.moe_adam_steps
        self.clocks = ((self.dense.steps, self.dense.adam_steps) if self.dense is not None else ()) + (self.moe_steps, self.moe_adam_steps)
        self.prediction = torch.zeros((len(configs), num_samples), dtype=torch.float32, device=device)
        h = self.expert_width
        expert_count = EXPERTS * (input_dim * h + h * h + 3 * h + 1)
        self.effective_parameter_counts = [
            input_dim * width + width * width + 3 * width + 1 if c.family.startswith("scalar") else
            input_dim * width + width * width + 2 * width + bins * (width + 1) if c.family in DENSE_FAMILIES else
            expert_count + EXPERTS * (input_dim + 1) if c.family.startswith("state_") else
            expert_count + EXPERTS if c.family == "constant_moe_js" else expert_count
            for c in configs
        ]
        self.allocated_parameter_count = sum(p.numel() for p in self.parameters)

    def state_tensors(self):
        """Every mutable allocation exactly once, including both bar clocks."""
        return (
            *(self.dense.state_tensors() if self.dense is not None else ()),
            *(t for group in self.groups for t in group.state_tensors()),
            self.moe_steps, self.moe_adam_steps, self.prediction,
        )

    @torch.no_grad()
    def candidate_finite(self):
        """Host-boundary health reduction; never synchronize candidate data to CPU."""
        healthy = torch.ones(len(self.configs), dtype=torch.bool, device=self.device)
        if self.dense is not None:
            dense_health = torch.ones(len(self.dense.configs), dtype=torch.bool, device=self.device)
            for tensor in self.dense.state_tensors():
                if tensor.ndim:
                    dense_health.logical_and_(torch.isfinite(tensor).flatten(1).all(1))
            healthy.index_copy_(0, self.dense_indices, dense_health)
        for group, positions in (
            (self.experts, self.moe_indices),
            (self.router_bias, self.moe_indices[self.router_bias_indices]),
            (self.router_weight, self.moe_indices[self.router_weight_indices]),
        ):
            if group is not None:
                group_health = healthy.index_select(0, positions)
                for tensor in group.state_tensors():
                    group_health.logical_and_(torch.isfinite(tensor).flatten(1).all(1))
                healthy.index_copy_(0, positions, group_health)
        return healthy & torch.isfinite(self.prediction).all(1)

    @torch.no_grad()
    def step(self, x, y, mask):
        # Sanitize invalid features before *any* forward/GEMM: zero derivatives
        # alone cannot stop NaN activations contaminating parameter gradients.
        features = torch.where(mask[:, None], x, 0.0)
        target = torch.where(mask, y, 0.0)
        if self.dense is not None:
            self.prediction.index_copy_(0, self.dense_indices, self.dense.step(features, target, mask))
        valid_count = mask.sum()
        active = valid_count > 0
        self.moe_steps.add_(1)
        self.moe_adam_steps.add_(active.to(torch.int64))
        if self.experts is not None:
            m, e, n, h = self.moe_count, EXPERTS, self.num_samples, self.expert_width
            w1, b1, w2, b2, w3, b3 = self.experts.parameters
            h1 = torch.tanh(torch.matmul(features, w1.flatten(0, 1).transpose(1, 2)) + b1.flatten(0, 1)[:, None, :])
            h2 = torch.tanh(torch.bmm(h1, w2.flatten(0, 1).transpose(1, 2)) + b2.flatten(0, 1)[:, None, :])
            expert_output = (torch.bmm(h2, w3.flatten(0, 1).transpose(1, 2)) + b3.flatten(0, 1)[:, None, :]).view(m, e, n).transpose(1, 2)
            logits = torch.zeros((m, n, e), dtype=torch.float32, device=self.device)
            if self.router_bias is not None:
                logits.index_add_(0, self.router_bias_indices, self.router_bias.parameters[0][:, None, :].expand(-1, n, -1))
            if self.router_weight is not None:
                logits.index_add_(0, self.router_weight_indices, torch.matmul(features, self.router_weight.parameters[0].transpose(1, 2)))
            gate = logits.softmax(-1)
            prediction = (gate * expert_output).sum(-1)
            self.prediction.index_copy_(0, self.moe_indices, prediction)
            derivative = torch.where(mask[None, :], 2.0 * (prediction - target) / valid_count.clamp_min(1).to(torch.float32), 0.0)
            # Both credit paths use the same pre-update expert predictions.
            dlogit = derivative[:, :, None] * gate * (expert_output - prediction[:, :, None])
            dz3 = (derivative[:, :, None] * gate).transpose(1, 2).reshape(m * e, n, 1)
            gw3 = torch.bmm(dz3.transpose(1, 2), h2).view_as(w3)
            gb3 = dz3.sum(1).view_as(b3)
            dz2 = torch.bmm(dz3, w3.flatten(0, 1)) * (1.0 - h2.square())
            gw2 = torch.bmm(dz2.transpose(1, 2), h1).view_as(w2)
            gb2 = dz2.sum(1).view_as(b2)
            dz1 = torch.bmm(dz2, w2.flatten(0, 1)) * (1.0 - h1.square())
            gw1 = torch.matmul(dz1.transpose(1, 2), features).view_as(w1)
            gb1 = dz1.sum(1).view_as(b1)
            clock = self.moe_adam_steps.clamp_min(1).to(torch.float64)
            correction1 = (1.0 - 0.9 ** clock).to(torch.float32)
            correction2 = (1.0 - 0.999 ** clock).to(torch.float32)
            self.experts.update((gw1, gb1, gw2, gb2, gw3, gb3), active, correction1, correction2)
            if self.router_bias is not None:
                self.router_bias.update((dlogit.index_select(0, self.router_bias_indices).sum(1),), active, correction1, correction2)
            if self.router_weight is not None:
                weight_gradient = torch.matmul(dlogit.index_select(0, self.router_weight_indices).transpose(1, 2), features)
                self.router_weight.update((weight_gradient,), active, correction1, correction2)
        return self.prediction

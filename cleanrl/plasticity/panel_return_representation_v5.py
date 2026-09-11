"""Signed conditional-mean learners with shared volatility representation teaching.

The scalar linear return head is never a decoded volatility forecast. Auxiliary
mispairing permutes stocks within the CURRENT labeled bar, preserving common
state rather than providing an iid null. No features, forecasts, or permutation
selection depend on current labels. All online operations remain on CUDA.
"""

import math
from dataclasses import dataclass

import torch

from cleanrl.plasticity.panel_hd_gate import Net
from cleanrl.shared import runtime
from cleanrl.shared.two_hot import DreamerTwoHotSupport

GROUPS = ("old_adam", "latest_adam", "memory_adam", "memory_aux", "memory_aux_permuted")
LRS = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


@dataclass(frozen=True)
class Config:
    family: str
    lr: float
    beta2: float = .999
    aux_weight: float = 0.


def configurations():
    return tuple(
        Config(group, lr, beta2)
        for group in GROUPS[:3] for lr in LRS for beta2 in (.9, .99, .999)
    ) + tuple(
        Config(group, lr, aux_weight=weight)
        for group in GROUPS[3:] for lr in LRS for weight in (.1, 1., 10.)
    )


class _Bank:
    """One frame/head family, vectorized over independent Adam candidates."""

    def __init__(self, configs, template, category_weight, support, permutation, num_samples, device):
        self.configs = tuple(configs)
        self.auxiliary = configs[0].family in GROUPS[3:]
        self.permuted = configs[0].family == "memory_aux_permuted"
        self.permutation = permutation
        self.projector = support
        count = len(configs)
        memory = configs[0].family.startswith("memory_")
        w1 = template.l1.weight.detach().to(device)
        if memory:
            w1 = torch.cat((w1, w1.new_zeros((w1.shape[0], 24))), dim=1)
        base = (w1, template.l1.bias.detach().to(device),
                template.l2.weight.detach().to(device), template.l2.bias.detach().to(device),
                w1.new_zeros((1, w1.shape[0])), w1.new_zeros(1))
        if self.auxiliary:
            base += (category_weight, -support.support)
        self.parameters = tuple(p.unsqueeze(0).repeat((count,) + (1,) * p.ndim) for p in base)
        self.first_moments = tuple(torch.zeros_like(p) for p in self.parameters)
        self.second_moments = tuple(torch.zeros_like(p) for p in self.parameters)
        self.learning_rates = torch.tensor([c.lr for c in configs], device=device)
        # Bias corrections use the original Python beta2 precision, as Adam does.
        self.beta2 = torch.tensor([c.beta2 for c in configs], dtype=torch.float64, device=device)
        self.aux_weights = torch.tensor([c.aux_weight for c in configs], device=device)
        self.adam_steps = torch.zeros(count, dtype=torch.int64, device=device)
        self.auxiliary_steps = torch.zeros_like(self.adam_steps) if self.auxiliary else None
        self.prediction = torch.zeros((count, num_samples), device=device)
        self.healthy = torch.ones(count, dtype=torch.bool, device=device)

    def state_tensors(self):
        clocks = (self.adam_steps, self.auxiliary_steps) if self.auxiliary else (self.adam_steps,)
        return (*self.parameters, *self.first_moments, *self.second_moments,
                *clocks, self.prediction, self.healthy)

    def candidate_finite(self):
        finite = self.healthy.clone()
        for value in self.state_tensors():
            finite.logical_and_(torch.isfinite(value).reshape(len(self.configs), -1).all(1))
        return finite

    @torch.no_grad()
    def step(self, x, signed_target, vol_target, mask):
        w1, b1, w2, b2, wr, br = self.parameters[:6]
        h1 = torch.tanh(torch.matmul(x, w1.transpose(1, 2)) + b1[:, None])
        h2 = torch.tanh(torch.bmm(h1, w2.transpose(1, 2)) + b2[:, None])
        mean = (torch.bmm(h2, wr.transpose(1, 2)) + br[:, None]).squeeze(-1)
        self.prediction.copy_(mean)

        count = mask.sum()
        active = count > 0
        target = torch.where(mask, signed_target, 0.)
        dr = torch.where(mask[None], 2. * (mean - target) / count.clamp_min(1), 0.).unsqueeze(-1)
        gwr = torch.bmm(dr.transpose(1, 2), h2)
        gbr = dr.sum(1)
        dh2 = torch.bmm(dr, wr)
        auxiliary_gradients = ()
        if self.auxiliary:
            wa, ba = self.parameters[6:]
            logits = torch.bmm(h2, wa.transpose(1, 2)) + ba[:, None]
            if self.permuted:
                auxiliary_mask = mask & mask.index_select(0, self.permutation)
                auxiliary_target = vol_target.index_select(0, self.permutation)
            else:
                auxiliary_mask, auxiliary_target = mask, vol_target
            auxiliary_count = auxiliary_mask.sum()
            auxiliary_active = (auxiliary_count > 0) & (self.aux_weights > 0)
            safe_target = torch.where(auxiliary_mask, auxiliary_target, 0.)
            labels = self.projector.project(safe_target)
            da = torch.where(
                auxiliary_mask[None, :, None] & (self.aux_weights[:, None, None] > 0),
                (logits.softmax(-1) - labels) * self.aux_weights[:, None, None]
                / auxiliary_count.clamp_min(1), 0.,
            )
            auxiliary_gradients = (torch.bmm(da.transpose(1, 2), h2), da.sum(1))
            auxiliary_dh2 = torch.bmm(da, wa)
            dh2 = dh2 + torch.where(auxiliary_active[:, None, None], auxiliary_dh2, 0.)
            # The projector clips endpoints by design. Nonfinite observed labels
            # must still disqualify the candidate, even if clipping hid a NaN.
            self.healthy.logical_and_(~(self.aux_weights > 0) | torch.where(
                auxiliary_mask, torch.isfinite(auxiliary_target), True).all())
            self.auxiliary_steps.add_(auxiliary_active.to(torch.int64))

        dz2 = dh2 * (1. - h2.square())
        gw2, gb2 = torch.bmm(dz2.transpose(1, 2), h1), dz2.sum(1)
        dz1 = torch.bmm(dz2, w2) * (1. - h1.square())
        gw1, gb1 = torch.matmul(dz1.transpose(1, 2), x), dz1.sum(1)
        gradients = (gw1, gb1, gw2, gb2, gwr, gbr, *auxiliary_gradients)
        self.adam_steps.add_(active.to(torch.int64))
        for index, (parameter, gradient, first, second) in enumerate(zip(
                self.parameters, gradients, self.first_moments, self.second_moments)):
            shape = (-1,) + (1,) * (parameter.ndim - 1)
            clock = self.adam_steps if index < 6 else self.auxiliary_steps
            enabled = active if index < 6 else auxiliary_active.view(shape)
            time = clock.clamp_min(1).to(torch.float64)
            correction1 = (1. - .9 ** time).to(parameter.dtype).view(shape)
            correction2 = (1. - self.beta2 ** time).to(parameter.dtype).view(shape)
            beta2 = self.beta2.to(parameter.dtype).view(shape)
            one_minus_beta2 = (1. - self.beta2).to(parameter.dtype).view(shape)
            first.copy_(torch.where(enabled, first * .9 + gradient * .1, first))
            second.copy_(torch.where(enabled, second * beta2 + gradient.square() * one_minus_beta2, second))
            adam = (first / correction1) / ((second / correction2).sqrt() + 1e-8)
            parameter.sub_(torch.where(enabled, self.learning_rates.view(shape) * adam, 0.))
        self.healthy.logical_and_(self.candidate_finite())
        return self.prediction


class Learner:
    """Width-128 signed means; no padded baseline head or postgradient gate.

    CPU initialization reuses the frozen Net RNG sequence, but CPU model
    execution is forbidden. All banks match the original hidden weights and
    biases; memory columns are appended only after initialization. Auxiliary
    head parameters and their Adam state are additional, not compute matched.
    """

    def __init__(self, input_dim, width, configs, device, num_samples=200, seed=1, bins=33):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CUDA required; there is no CPU learner fallback")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if width != 128 or min(input_dim, num_samples) < 1:
            raise ValueError("width must be 128; input_dim and num_samples must be positive")
        self.configs = tuple(configs)
        if not self.configs or any(c.family not in GROUPS for c in self.configs):
            raise ValueError("configs must contain known families")
        for config in self.configs:
            if not math.isfinite(config.lr) or config.lr <= 0:
                raise ValueError("learning rates must be positive and finite")
            if not math.isfinite(config.beta2) or not 0 <= config.beta2 < 1:
                raise ValueError("beta2 must lie in [0, 1)")
            if not math.isfinite(config.aux_weight) or config.aux_weight < 0:
                raise ValueError("aux_weight must be finite and nonnegative")
            if config.family in GROUPS[:3] and config.aux_weight != 0:
                raise ValueError("baseline families cannot have an auxiliary loss")
        self.output_names = tuple(
            f"{c.family}_lr{c.lr:g}_beta2{c.beta2:g}_aux{c.aux_weight:g}" for c in self.configs)
        if len(set(self.output_names)) != len(self.configs):
            raise ValueError("configurations must have distinct output names")
        runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
        self.device, self.num_samples, self.bins = device, num_samples, bins
        self.projector = DreamerTwoHotSupport(bins, 12.5, device, spacing="linear")
        self.projector.support.add_(12.5)
        self.support = self.projector.support
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            template = Net(input_dim, width)
        generator = torch.Generator(device=device).manual_seed(seed)
        category_weight = torch.randn((bins, width), generator=generator, device=device) * (1e-3 / math.sqrt(width))
        # A dedicated CPU RNG fixes stock correspondence independently of CUDA
        # RNG implementations, auxiliary initialization, or candidate ordering.
        self.permutation = torch.randperm(num_samples, generator=torch.Generator(device="cpu").manual_seed(1),
                                          device="cpu").to(device)
        self.groups, self.indices, self.frame_indices = [], [], []
        for family in GROUPS:
            columns = [i for i, c in enumerate(self.configs) if c.family == family]
            if not columns:
                continue
            local = [self.configs[i] for i in columns]
            self.groups.append(_Bank(local, template, category_weight, self.projector,
                                     self.permutation, num_samples, device))
            self.indices.append(torch.tensor(columns, device=device))
            self.frame_indices.append(0 if family == "old_adam" else 1 if family == "latest_adam" else 2)
        self.prediction = torch.zeros((len(self.configs), num_samples), device=device)
        self.parameters = tuple(p for bank in self.groups for p in bank.parameters)
        self.costs = []
        for c in self.configs:
            dim = input_dim + (24 if c.family.startswith("memory_") else 0)
            auxiliary = bins if c.family in GROUPS[3:] else 0
            trunk = width * dim + width * width + 2 * width
            allocated = trunk + (1 + auxiliary) * (width + 1)
            active = trunk + (1 + (auxiliary if c.aux_weight > 0 else 0)) * (width + 1)
            self.costs.append({"family": c.family, "lr": c.lr, "beta2": c.beta2,
                               "aux_weight": c.aux_weight, "input_dim": dim,
                               "active_parameters": active, "allocated_parameters": allocated,
                               "optimizer_state_elements": 2 * allocated + 1 + bool(auxiliary),
                               "optimizer_state_bytes": 8 * allocated + 8 * (1 + bool(auxiliary)),
                               "auxiliary_head_parameters": auxiliary * (width + 1),
                               "dense_forward_macs_per_stock": width * dim + width * width + (1 + auxiliary) * width})

    def state_tensors(self):
        return (*tuple(t for bank in self.groups for t in bank.state_tensors()), self.prediction)

    @torch.no_grad()
    def candidate_finite(self):
        result = torch.ones(len(self.configs), dtype=torch.bool, device=self.device)
        for bank, indices in zip(self.groups, self.indices):
            result.index_copy_(0, indices, bank.candidate_finite())
        return result & torch.isfinite(self.prediction).all(1)

    @torch.no_grad()
    def step(self, frames_tuple, signed_target, vol_target, mask):
        """Return owned [candidate, stock] forecasts BEFORE consuming this bar."""
        for bank, indices, frame in zip(self.groups, self.indices, self.frame_indices):
            self.prediction.index_copy_(0, indices, bank.step(frames_tuple[frame], signed_target, vol_target, mask))
        return self.prediction

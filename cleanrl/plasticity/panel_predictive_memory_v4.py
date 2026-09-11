"""Fixed causal multiscale information and independent volatility-loss controls.

Memory is a hypothesis about useful sufficient-statistic approximations, not a
biological mechanism or a claim of conditional-mean optimality. Channel EWMAs
start at zero before bar zero and consume the Bank's zero-filled eight channels
through the forecast bar. They never consult label-validity masks. References
start at prefix mu and update only on CURRENT observed returns; a missing stock
holds its own state, while an empty cross-section holds the common state.
"""

from dataclasses import dataclass

import torch

from cleanrl.plasticity import panel_distributional_model_v1 as frozen

FRAMES = ("old", "latest", "memory")
GROUPS = tuple(f"{frame}_{head}" for frame in FRAMES for head in ("scalar_adam", "categorical_ce_js")) + (
    "memory_exp_mse", "memory_exp_qlike")
LRS = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
REFERENCE_NAMES = ("constant_mu",) + tuple(f"{kind}_ewma_{half}" for kind in ("own", "cross") for half in (16, 128, 1024)) + ("ridge_ewma_1",)


@dataclass(frozen=True)
class Config:
    family: str
    lr: float


def configurations(lrs=LRS):
    return tuple(Config(group, lr) for group in GROUPS for lr in lrs)


class CausalState:
    """Mutable feature/reference state; index must advance exactly once per bar."""

    def __init__(self, bank, observed_valid):
        self.bank = bank
        self.observed_valid = observed_valid
        self.memory = torch.zeros((3, bank.N, 8), device=bank.dev)
        self.memory_alpha = torch.tensor([1 - 2 ** (-1 / h) for h in (64, 256, 1024)], device=bank.dev)[:, None, None]
        self.reference_alpha = torch.tensor([1 - 2 ** (-1 / h) for h in (16, 128, 1024)], device=bank.dev)[:, None]
        self.own = torch.ones((3, bank.N), device=bank.dev) * bank.mu
        self.cross = torch.ones((3, 1), device=bank.dev) * bank.mu
        self.observed_steps = torch.zeros((), dtype=torch.int64, device=bank.dev)

    def state_tensors(self):
        return self.memory, self.own, self.cross, self.observed_steps

    def channels(self, t):
        b = self.bank
        index = t.reshape(1)
        z = b.zt.index_select(0, index).squeeze(0)
        own = z[1:]
        return torch.stack((own, own.abs(), b.vst.index_select(0, index).squeeze(0)[1:],
                            z[0].expand(b.N), z[0].abs().expand(b.N),
                            b.cst.index_select(0, index).squeeze(0).expand(b.N),
                            b.acst.index_select(0, index).squeeze(0).expand(b.N), own.square()), dim=1)

    @torch.no_grad()
    def observe(self, t):
        channels = self.channels(t)
        self.memory.add_(self.memory_alpha * (channels[None] - self.memory))
        valid = self.observed_valid.index_select(0, t.reshape(1)).squeeze(0)
        realized = channels[:, 0].square().clamp_max(25)
        self.own.add_(torch.where(valid[None], self.reference_alpha * (realized[None] - self.own), 0.0))
        count = valid.sum()
        common = torch.where(valid, realized, 0.0).sum() / count.clamp_min(1)
        self.cross.add_(torch.where(count > 0, self.reference_alpha * (common - self.cross), 0.0))
        self.observed_steps.add_(1)

    def frames(self, t):
        old = self.bank.feats(t)
        latest = self.bank.feats(t + 1)
        # Append AFTER the original bias: every old first-layer coordinate stays
        # at exactly its original index, including its bias feature coefficient.
        memory = torch.cat((latest, self.memory.permute(1, 0, 2).reshape(self.bank.N, 24)), dim=1)
        return old, latest, memory

    def references(self):
        b = self.bank
        return torch.cat((torch.zeros((1, b.N), device=b.dev), self.own - b.mu,
                          self.cross.expand(3, b.N) - b.mu), dim=0)


def exp_logit_gradient(mean, raw_target, qlike):
    """Unfloored derivatives of MSE and log(mean)+raw_target/mean."""
    return torch.where(qlike, 1.0 - raw_target / mean, 2.0 * (mean - raw_target) * mean)


class RidgeReference:
    """Fixed ridge1 pooled affine calibration, solved once at the prefix cut.

    Six centered EWMA features plus intercept; SSE + ||coefficient||² including
    intercept penalty. FP64 sufficient statistics and solve, no repair/floor.
    Prefix predictions are mu until the solve; only frozen suffix is evidence.
    """

    def __init__(self, device):
        self.gram = torch.zeros((7, 7), dtype=torch.float64, device=device)
        self.rhs = torch.zeros(7, dtype=torch.float64, device=device)
        self.coefficient = torch.zeros(7, dtype=torch.float64, device=device)

    def state_tensors(self):
        return self.gram, self.rhs, self.coefficient

    @torch.no_grad()
    def step(self, references, y, mask, fitting):
        x = torch.cat((references[1:].T.double(), torch.ones_like(y[:, None], dtype=torch.float64)), dim=1)
        prediction = (x @ self.coefficient).float()[None]
        fit_mask = mask & fitting
        masked_x = torch.where(fit_mask[:, None], x, 0.0)
        self.gram.add_(masked_x.T @ masked_x)
        self.rhs.add_(masked_x.T @ torch.where(fit_mask, y.double(), 0.0))
        return prediction

    @torch.no_grad()
    def fit(self):
        self.coefficient.copy_(torch.linalg.solve(self.gram + torch.eye(7, dtype=torch.float64, device=self.gram.device), self.rhs))
        if not bool(torch.isfinite(self.coefficient).all()):
            raise FloatingPointError("ridge1 coefficient solve is nonfinite")


class ExpLearner:
    """Scalar positive mean head; identical parameters for both Adam objectives."""

    def __init__(self, template, configs, mu):
        self.configs = tuple(configs)
        self.device = template.device
        count = len(configs)
        self.weights = tuple(p[:1, :1].clone().repeat(count, 1, 1) if i == 2 else p[:1].clone().repeat(count, 1, 1)
                             for i, p in enumerate(template.weights))
        self.biases = tuple(p[:1, :1].clone().repeat(count, 1) if i == 2 else p[:1].clone().repeat(count, 1)
                           for i, p in enumerate(template.biases))
        # Constant positive initialization prevents head/objective initialization
        # confounding; it is the same raw mu for EVERY exp-MSE/QLIKE candidate.
        self.weights[2].zero_()
        self.biases[2].copy_(torch.log(torch.as_tensor(mu, device=self.device)))
        self.mu = torch.as_tensor(mu, device=self.device)
        self.parameters = tuple(p for pair in zip(self.weights, self.biases) for p in pair)
        self.first_moments = tuple(torch.zeros_like(p) for p in self.parameters)
        self.second_moments = tuple(torch.zeros_like(p) for p in self.parameters)
        self.steps = torch.zeros((), dtype=torch.int64, device=self.device)
        self.adam_steps = torch.zeros_like(self.steps)
        self.prediction = torch.zeros((count, template.num_samples), device=self.device)
        self.learning_rates = torch.tensor([c.lr for c in configs], device=self.device)
        self.qlike = torch.tensor([c.family.endswith("qlike") for c in configs], device=self.device)[:, None]
        self.healthy = torch.ones(count, dtype=torch.bool, device=self.device)

    def state_tensors(self):
        return (*self.parameters, *self.first_moments, *self.second_moments,
                self.steps, self.adam_steps, self.prediction, self.healthy)

    @torch.no_grad()
    def step(self, x, y, mask):
        w1, w2, w3 = self.weights
        b1, b2, b3 = self.biases
        h1 = torch.tanh(torch.matmul(x, w1.transpose(1, 2)) + b1[:, None])
        h2 = torch.tanh(torch.bmm(h1, w2.transpose(1, 2)) + b2[:, None])
        mean = (torch.bmm(h2, w3.transpose(1, 2)) + b3[:, None]).squeeze(-1).exp()
        self.prediction.copy_(mean - self.mu)
        raw_target = torch.where(mask, y + self.mu, 0.0)
        derivative = exp_logit_gradient(mean, raw_target, self.qlike)
        self.healthy.logical_and_((torch.isfinite(mean) & (mean > 0)).all(1))
        self.healthy.logical_and_(torch.where(mask[None], torch.isfinite(derivative), True).all(1))
        count = mask.sum()
        active = count > 0
        dz3 = torch.where(mask[None], derivative / count.clamp_min(1), 0.0)[:, :, None]
        gw3, gb3 = torch.bmm(dz3.transpose(1, 2), h2), dz3.sum(1)
        dz2 = torch.bmm(dz3, w3) * (1 - h2.square())
        gw2, gb2 = torch.bmm(dz2.transpose(1, 2), h1), dz2.sum(1)
        dz1 = torch.bmm(dz2, w2) * (1 - h1.square())
        gradients = (torch.matmul(dz1.transpose(1, 2), x), dz1.sum(1), gw2, gb2, gw3, gb3)
        self.steps.add_(1)
        self.adam_steps.add_(active.to(torch.int64))
        clock = self.adam_steps.clamp_min(1).double()
        c1, c2 = (1 - .9 ** clock).float(), (1 - .999 ** clock).float()
        for p, g, m, v in zip(self.parameters, gradients, self.first_moments, self.second_moments):
            m.copy_(torch.where(active, .9 * m + .1 * g, m))
            v.copy_(torch.where(active, .999 * v + .001 * g.square(), v))
            update = (m / c1) / ((v / c2).sqrt() + 1e-8)
            p.sub_(torch.where(active, self.learning_rates.reshape((-1,) + (1,) * (p.ndim - 1)) * update, 0.0))
        return self.prediction


class Learner:
    """Six frozen banks plus one joint exp bank, independently tuned by group."""

    def __init__(self, input_dim, width, mu, configs, device, bins=33, seed=1, num_samples=200):
        self.configs = tuple(configs)
        if any(c.family not in GROUPS for c in configs):
            raise ValueError("unknown frame/head group")
        self.output_names = tuple(f"{c.family}_{c.lr:g}" for c in configs) + REFERENCE_NAMES
        self.groups, self.indices, self.frame_indices = [], [], []
        template = None
        for frame_index, frame in enumerate(FRAMES):
            columns = [i for i, c in enumerate(configs) if c.family in (f"{frame}_scalar_adam", f"{frame}_categorical_ce_js")]
            if not columns:
                continue
            local = [frozen.Config(configs[i].family[len(frame) + 1:], configs[i].lr) for i in columns]
            bank = frozen.Learner(input_dim, width, mu, local, device, bins=bins, seed=seed, num_samples=num_samples)
            if frame == "memory":
                # Construct with ORIGINAL dimension first, preserving downstream
                # RNG draws and all matching coefficients; new columns are zero.
                w1 = torch.cat((bank.weights[0], torch.zeros((len(local), width, 24), device=device)), dim=2)
                bank.weights = (w1, *bank.weights[1:])
                bank.parameters = tuple(p for pair in zip(bank.weights, bank.biases) for p in pair)
                for field in ("first_moments", "second_moments", "s1", "s2"):
                    setattr(bank, field, tuple(torch.zeros_like(p) for p in bank.parameters))
                template = bank
            self.groups.append(bank)
            self.indices.append(torch.tensor(columns, device=device))
            self.frame_indices.append(frame_index)
        columns = [i for i, c in enumerate(configs) if c.family in GROUPS[-2:]]
        if columns:
            if template is None:
                raise ValueError("exp controls require the memory frozen bank initializer")
            self.groups.append(ExpLearner(template, [configs[i] for i in columns], mu))
            self.indices.append(torch.tensor(columns, device=device))
            self.frame_indices.append(2)
        self.prediction = torch.zeros((len(self.output_names), num_samples), device=device)
        self.parameters = tuple(p for bank in self.groups for p in bank.parameters)
        self.costs = []
        for c in configs:
            dim = input_dim + (24 if c.family.startswith("memory_") else 0)
            head = bins if "categorical" in c.family else 1
            active = width * dim + width * width + 2 * width + head * (width + 1)
            allocated = active if "exp_" in c.family else width * dim + width * width + 2 * width + bins * (width + 1)
            self.costs.append({"family": c.family, "lr": c.lr, "input_dim": dim, "active_parameters": active,
                               "allocated_parameters": allocated, "dense_forward_macs_per_stock": width * dim + width * width + (head if "exp_" in c.family else bins) * width})

    def state_tensors(self):
        return (*tuple(t for bank in self.groups for t in bank.state_tensors()), self.prediction)

    @torch.no_grad()
    def candidate_finite(self):
        result = torch.ones(len(self.output_names), dtype=torch.bool, device=self.prediction.device)
        for bank, indices in zip(self.groups, self.indices):
            finite = torch.ones(len(bank.configs), dtype=torch.bool, device=self.prediction.device)
            for t in bank.state_tensors():
                if t.ndim and t.shape[0] == len(bank.configs):
                    finite.logical_and_(torch.isfinite(t).reshape(len(bank.configs), -1).all(1))
            if isinstance(bank, ExpLearner):
                finite.logical_and_(bank.healthy)
            result.index_copy_(0, indices, finite)
        return result & torch.isfinite(self.prediction).all(1)

    @torch.no_grad()
    def step(self, frames, y, mask, references):
        for bank, indices, frame in zip(self.groups, self.indices, self.frame_indices):
            self.prediction.index_copy_(0, indices, bank.step(frames[frame], y, mask))
        self.prediction[len(self.configs):].copy_(references)
        return self.prediction

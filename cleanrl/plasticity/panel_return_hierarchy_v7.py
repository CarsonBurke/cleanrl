"""Exact static Gaussian posterior means, not a claim about market truth.

Each distinct local precision owns one stock-state bank. Hierarchical global
priors reuse that bank (including independent-ridge overlaps). Schur information
is accumulated as positive increments; there is no covariance repair/fallback.
"""

import math
from dataclasses import dataclass

import torch

GROUPS = ("pooled_ridge", "independent_ridge", "hierarchical_ridge")
COARSE = tuple(10. ** i for i in range(8))
# 56 inclusive log-spaced points; none coincide with the eight integer powers.
RIDGE_PRIORS = tuple(sorted((*COARSE, *(10. ** (-2. + 12. * i / 55.) for i in range(56)))))


@dataclass(frozen=True)
class Config:
    family: str
    lambda_local: float
    lambda_global: float = 0.


def configurations():
    return tuple(Config(group, value) for group in GROUPS[:2] for value in RIDGE_PRIORS) + tuple(
        Config(GROUPS[2], local, global_) for local in COARSE for global_ in COARSE)


class Learner:
    """FP64 CUDA inference. step(x[N,D], y[N], mask[N]) returns pre-label means.

    All mutable tensors, including sticky numerical diagnostics, are exposed for
    CUDA graph snapshot/restore. solve_ex never synchronizes or hides solver info.
    """

    def __init__(self, num_stocks, features=13, configs=None, device="cuda"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA required; no CPU model fallback")
        if min(num_stocks, features) < 1:
            raise ValueError("positive stock and feature dimensions required")
        self.configs = tuple(configurations() if configs is None else configs)
        if not self.configs or len(set(self.configs)) != len(self.configs):
            raise ValueError("distinct configurations required")
        for c in self.configs:
            if c.family not in GROUPS or not math.isfinite(c.lambda_local) or c.lambda_local <= 0:
                raise ValueError("unknown family or invalid local precision")
            if (c.family == GROUPS[2] and (not math.isfinite(c.lambda_global) or c.lambda_global <= 0)
                    or c.family != GROUPS[2] and c.lambda_global != 0):
                raise ValueError("hierarchical global precision must be positive; controls have none")
        if any(not any(c.family == group for c in self.configs) for group in GROUPS):
            raise ValueError("all three structural families required")
        self.num_stocks, self.features = num_stocks, features
        kw = {"device": self.device, "dtype": torch.float64}
        self.eye = torch.eye(features, **kw)
        local_values = sorted({c.lambda_local for c in self.configs if c.family != GROUPS[0]})
        pooled_values = [c.lambda_local for c in self.configs if c.family == GROUPS[0]]
        hierarchy_values = sorted({c.lambda_local for c in self.configs if c.family == GROUPS[2]})
        self.local_values = tuple(local_values)
        self.local_precision = torch.tensor(local_values, **kw)
        self.pooled_precision = torch.tensor(pooled_values, **kw)
        self.hierarchy_precision = torch.tensor(hierarchy_values, **kw)
        self.global_precision = torch.tensor([c.lambda_global for c in self.configs if c.family == GROUPS[2]], **kw)
        self.columns = tuple(torch.tensor([i for i, c in enumerate(self.configs) if c.family == group], device=self.device)
                             for group in GROUPS)
        self.independent_local = torch.tensor([local_values.index(c.lambda_local) for c in self.configs
                                              if c.family == GROUPS[1]], device=self.device)
        self.hierarchy_local = torch.tensor([local_values.index(v) for v in hierarchy_values], device=self.device)
        self.hierarchy_config = torch.tensor([hierarchy_values.index(c.lambda_local) for c in self.configs
                                             if c.family == GROUPS[2]], device=self.device)
        self.P = (self.eye[None, None] / self.local_precision[:, None, None, None]).expand(
            -1, num_stocks, -1, -1).clone()
        self.m = torch.zeros((len(local_values), num_stocks, features), **kw)
        self.B = torch.zeros((len(hierarchy_values), features, features), **kw)
        self.q = torch.zeros((len(hierarchy_values), features), **kw)
        self.gram = torch.zeros((features, features), **kw)
        self.rhs = torch.zeros(features, **kw)
        self.stock_gram = torch.zeros((num_stocks, features, features), **kw)
        self.stock_rhs = torch.zeros((num_stocks, features), **kw)
        self.prediction = torch.zeros((len(self.configs), num_stocks), **kw)
        self.healthy = torch.ones(len(self.configs), dtype=torch.bool, device=self.device)
        self.solver_failures = torch.zeros(len(self.configs), dtype=torch.int64, device=self.device)
        self.steps = torch.zeros((), dtype=torch.int64, device=self.device)
        self.observations = torch.zeros(num_stocks, dtype=torch.int64, device=self.device)
        self.output_names = tuple(f"{c.family}_local{c.lambda_local:.17g}_global{c.lambda_global:.17g}" for c in self.configs)
        self.costs = {"local_state_banks": len(local_values), "hierarchical_schur_banks": len(hierarchy_values),
                      "pooled_shared_statistics": True, "local_covariance_elements": self.P.numel(),
                      "mutable_bytes": sum(t.numel() * t.element_size() for t in self.state_tensors()),
                      "likelihood": "unit Gaussian noise, static coefficients, FP64"}

    def state_tensors(self):
        return (self.P, self.m, self.B, self.q, self.gram, self.rhs, self.stock_gram, self.stock_rhs,
                self.prediction, self.healthy, self.solver_failures, self.steps, self.observations)

    def _solve(self, matrix, rhs, columns):
        solution, info = torch.linalg.solve_ex(matrix, rhs.unsqueeze(-1), check_errors=False)
        solution = solution.squeeze(-1)
        ok = (info == 0) & torch.isfinite(solution).all(-1)
        self.healthy.index_copy_(0, columns, self.healthy.index_select(0, columns) & ok)
        self.solver_failures.index_add_(0, columns, (info != 0).long())
        return solution

    @torch.no_grad()
    def predict(self, x):
        x = x.double()
        pooled = self._solve(self.gram[None] + self.pooled_precision[:, None, None] * self.eye,
                             self.rhs.expand(len(self.pooled_precision), -1), self.columns[0])
        global_ = self._solve(self.B.index_select(0, self.hierarchy_config)
                             + self.global_precision[:, None, None] * self.eye,
                             self.q.index_select(0, self.hierarchy_config), self.columns[2])
        local_mean = (self.m * x[None]).sum(-1)
        # P*x is reused for prediction and the subsequent sufficient-statistic update.
        u = torch.matmul(self.P, x[None, :, :, None]).squeeze(-1)
        hu = u.index_select(0, self.hierarchy_local) * self.hierarchy_precision[:, None, None]
        hierarchical = local_mean.index_select(0, self.hierarchy_local).index_select(0, self.hierarchy_config)
        hierarchical = hierarchical + (hu.index_select(0, self.hierarchy_config) * global_[:, None]).sum(-1)
        self.prediction.index_copy_(0, self.columns[0], pooled @ x.T)
        self.prediction.index_copy_(0, self.columns[1], local_mean.index_select(0, self.independent_local))
        self.prediction.index_copy_(0, self.columns[2], hierarchical)
        self.healthy.logical_and_(torch.isfinite(self.prediction).all(-1))
        return self.prediction, u, local_mean

    @torch.no_grad()
    def step(self, x, y, mask):
        prediction, u, local_mean = self.predict(x)
        x = x.double()
        # where removes missing labels/features before arithmetic: NaN*0 is not masking.
        u = torch.where(mask[None, :, None], u, 0.)
        xm = torch.where(mask[:, None], x, 0.)
        denominator = 1. + (u * xm[None]).sum(-1)
        error = torch.where(mask[None], y.double()[None] - local_mean, 0.)
        change_m = u * (error / denominator)[:, :, None]
        change_P = u[:, :, :, None] * (u / denominator[:, :, None])[:, :, None, :]
        hu = change_P.index_select(0, self.hierarchy_local).sum(1)
        self.B.add_(hu * self.hierarchy_precision[:, None, None].square())
        self.q.add_(change_m.index_select(0, self.hierarchy_local).sum(1) * self.hierarchy_precision[:, None])
        self.P.sub_(change_P)
        self.m.add_(change_m)
        gram_increment = xm[:, :, None] * xm[:, None, :]
        rhs_increment = xm * torch.where(mask, y.double(), 0.)[:, None]
        self.stock_gram.add_(gram_increment)
        self.stock_rhs.add_(rhs_increment)
        self.gram.add_(gram_increment.sum(0))
        self.rhs.add_(rhs_increment.sum(0))
        local_ok = (torch.isfinite(self.P).all(dim=(-1, -2, -3)) & torch.isfinite(self.m).all(dim=(-1, -2))
                    & torch.isfinite(denominator).all(-1) & (denominator > 0).all(-1))
        self.healthy.index_copy_(0, self.columns[1], self.healthy.index_select(0, self.columns[1])
                                 & local_ok.index_select(0, self.independent_local))
        hierarchy_ok = (local_ok.index_select(0, self.hierarchy_local) & torch.isfinite(self.B).all(dim=(-1, -2))
                        & torch.isfinite(self.q).all(-1)).index_select(0, self.hierarchy_config)
        self.healthy.index_copy_(0, self.columns[2], self.healthy.index_select(0, self.columns[2]) & hierarchy_ok)
        self.healthy.index_copy_(0, self.columns[0], self.healthy.index_select(0, self.columns[0])
                                 & torch.isfinite(self.gram).all() & torch.isfinite(self.rhs).all())
        self.steps.add_(1)
        self.observations.add_(mask.long())
        return prediction

    def candidate_finite(self):
        return self.healthy & torch.isfinite(self.prediction).all(-1)

    @torch.no_grad()
    def audit(self, config, rtol=1e-7, atol=1e-10):
        """Independent end-of-stream natural-statistic audit; never repairs state.

        The declared tolerance classifies disagreement, not solver eligibility.
        Batch Schur information uses lambda*A^-1*G rather than subtracting
        near-equal matrices. All audit solves and nonfinite values are reported.
        """
        if config not in self.configs:
            raise ValueError("audit configuration was not trained")
        infos = []

        def solve(matrix, rhs):
            value, info = torch.linalg.solve_ex(matrix, rhs, check_errors=False)
            infos.append(info.detach().cpu().reshape(-1).tolist())
            return value

        local = config.lambda_local
        if config.family == GROUPS[0]:
            natural = self.stock_gram.sum(0) + local * self.eye
            natural_rhs = self.stock_rhs.sum(0)
            batch = solve(natural, natural_rhs)
            recursive = solve(self.gram + local * self.eye, self.rhs)
            residual = natural @ recursive - natural_rhs
            scale = torch.linalg.vector_norm(natural @ recursive) + torch.linalg.vector_norm(natural_rhs)
        else:
            index = self.local_values.index(local)
            matrix = self.stock_gram + local * self.eye
            if config.family == GROUPS[1]:
                batch = solve(matrix, self.stock_rhs.unsqueeze(-1)).squeeze(-1)
                recursive = self.m[index]
                lhs = torch.matmul(matrix, recursive.unsqueeze(-1)).squeeze(-1)
                residual = lhs - self.stock_rhs
                scale = torch.linalg.vector_norm(lhs) + torch.linalg.vector_norm(self.stock_rhs)
            else:
                joined = torch.cat((self.stock_gram, self.stock_rhs.unsqueeze(-1),
                                    self.eye.expand(self.num_stocks, -1, -1)), dim=-1)
                natural_solution = solve(matrix, joined)
                batch_m = natural_solution[:, :, self.features]
                batch_inverse = natural_solution[:, :, self.features + 1:]
                batch_B = local * natural_solution[:, :, :self.features].sum(0)
                batch_q = local * batch_m.sum(0)
                batch_global = solve(batch_B + config.lambda_global * self.eye, batch_q)
                batch_local = batch_m + local * torch.matmul(batch_inverse, batch_global)
                hindex = sorted({c.lambda_local for c in self.configs if c.family == GROUPS[2]}).index(local)
                global_ = solve(self.B[hindex] + config.lambda_global * self.eye, self.q[hindex])
                local_ = self.m[index] + local * torch.matmul(self.P[index], global_)
                batch = torch.cat((batch_global[None], batch_local))
                recursive = torch.cat((global_[None], local_))
                data_lhs = torch.matmul(self.stock_gram, local_.unsqueeze(-1)).squeeze(-1)
                coupling = local * (local_ - global_)
                local_residual = data_lhs + coupling - self.stock_rhs
                global_residual = config.lambda_global * global_ - coupling.sum(0)
                residual = torch.cat((global_residual[None], local_residual))
                scale = (torch.linalg.vector_norm(data_lhs) + torch.linalg.vector_norm(coupling)
                         + torch.linalg.vector_norm(self.stock_rhs) + torch.linalg.vector_norm(config.lambda_global * global_))
        difference = recursive - batch
        finite = bool(torch.isfinite(batch).all() & torch.isfinite(recursive).all() & torch.isfinite(residual).all())
        solver_ok = all(code == 0 for call in infos for code in call)
        agreement = bool(torch.all(torch.abs(difference) <= atol + rtol * torch.abs(batch)))
        norm = float(torch.linalg.vector_norm(batch).cpu())
        error_norm = float(torch.linalg.vector_norm(difference).cpu())
        residual_norm = float(torch.linalg.vector_norm(residual).cpu())
        scale_value = float(scale.cpu())
        return {
            "status": "nonfinite_or_solver_failure" if not finite or not solver_ok else "agreement" if agreement else "disagreement",
            "method": "independent masked natural Gram/RHS batch solve; no recursive state used in reference",
            "rtol": rtol, "atol": atol, "solver_info_by_call": infos,
            "coefficient_max_abs_error": float(difference.abs().max().cpu()),
            "coefficient_relative_l2_error": error_norm / norm if norm > 0 else 0. if error_norm == 0 else math.inf,
            "normal_equation_residual_max_abs": float(residual.abs().max().cpu()),
            "normal_equation_relative_residual_l2": residual_norm / scale_value if scale_value > 0 else 0. if residual_norm == 0 else math.inf,
            "comparison_within_tolerance": agreement and finite and solver_ok,
            "recursive_coefficients": recursive.cpu().tolist(), "batch_coefficients": batch.cpu().tolist(),
            "observations_per_stock": self.observations.cpu().tolist(),
        }

"""Otto et al. ICLR 2021 Differentiable Trust Region Layers — paper-faithful core.

Reference: arXiv:2101.09207 + official boschresearch/trust-region-layers
  - Mean: closed-form Mahalanobis projection (Eq. 6 / mean_projection)
  - Cov (diag KL): precision interpolation (Eq. 11) with η from 1-D dual g(η)
    minimized as in DiagCovOnlyKLProjection.cpp (L-BFGS dual, here 1-D BFGS/Newton
    on the same dual + analytical dual gradient).
  - Backward through η: KKT / last_eta_grad as in official C++ (not stop-grad η).

Only the TRL math lives here. Training recipe (epochs, nets, etc.) is the caller's.
"""

from __future__ import annotations

import torch
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Mean projection (paper Eq. 6; official mean_projection)
# ---------------------------------------------------------------------------

def mean_projection(mean: torch.Tensor, old_mean: torch.Tensor, dist: torch.Tensor, eps: float) -> torch.Tensor:
    """Project mean so dist(mean_proj, old) ≤ eps, minimizing dist to unconstrained mean.

    `dist` must be the same mean-metric used in the constraint (for KL layer: mean part of
    reverse KL = 0.5 * Mahalanobis, as in official gaussian_kl + mean_projection).
    """
    eps = float(eps)
    mask = dist > eps
    omega = torch.zeros_like(dist)
    omega = torch.where(mask, (dist / eps).clamp_min(1e-12).sqrt() - 1.0, omega)
    omega = omega.abs().unsqueeze(-1)
    m = (mean + omega * old_mean) / (1.0 + omega + 1e-16)
    return torch.where(mask.unsqueeze(-1), m, mean)


def kl_mean_part(mean: torch.Tensor, old_mean: torch.Tensor, old_logstd: torch.Tensor) -> torch.Tensor:
    """Mean part of reverse KL(new||old) for diag Gaussian = 0.5 * maha (official)."""
    old_var = (2.0 * old_logstd).exp().clamp_min(1e-8)
    return 0.5 * ((mean - old_mean).pow(2) / old_var).sum(-1)


def kl_cov_part(logstd: torch.Tensor, old_logstd: torch.Tensor) -> torch.Tensor:
    """Cov part of reverse KL(new||old) for diag Gaussian."""
    var = (2.0 * logstd).exp().clamp_min(1e-12)
    old_var = (2.0 * old_logstd).exp().clamp_min(1e-12)
    return 0.5 * (torch.log(old_var / var) + var / old_var - 1.0).sum(-1)


def analytic_kl_diag(mean, logstd, old_mean, old_logstd):
    return kl_mean_part(mean, old_mean, old_logstd) + kl_cov_part(logstd, old_logstd)


# ---------------------------------------------------------------------------
# Dual g(η) — matches DiagCovOnlyKLProjection::dual (omega_offset=1)
# ---------------------------------------------------------------------------

def _dual_g_and_grad(eta: float, old_var: torch.Tensor, target_var: torch.Tensor, eps: float, omega_offset: float = 1.0):
    """Scalar dual g(η) and g'(η)=eps-KL for diag cov. old_var/target_var shape (A,)."""
    eta = max(float(eta), 0.0)
    old_prec = 1.0 / old_var.clamp_min(1e-12)
    target_prec = 1.0 / target_var.clamp_min(1e-12)
    new_prec = (eta * old_prec + target_prec) / (eta + omega_offset)
    new_var = 1.0 / new_prec.clamp_min(1e-12)
    old_logdet = old_var.clamp_min(1e-12).log().sum()
    new_logdet = new_var.clamp_min(1e-12).log().sum()
    dim = old_var.numel()
    # dual value (C++ DiagCovOnlyKLProjection::dual)
    dual = eta * eps - 0.5 * eta * old_logdet + 0.5 * (eta + omega_offset) * new_logdet
    # KL(new||old) cov-only and dual gradient g' = eps - kl
    trace_term = (old_prec * new_var).sum()
    kl = 0.5 * (old_logdet - dim - new_logdet + trace_term)
    grad = eps - kl
    return float(dual), float(grad), new_var, eta


def _solve_eta_lbfgs_1d(old_var: torch.Tensor, target_var: torch.Tensor, eps: float,
                       omega_offset: float = 1.0, max_eval: int = 100) -> tuple[float, bool]:
    """Minimize dual g(η), η≥0, with analytical dual gradient (paper/official L-BFGS dual).

    Official: nlopt LD_LBFGS on g(η), g'(η)=ε−KL. In 1-D with analytic g', damped Newton on
    g'=0 with dual-descent backtracking is the standard reduction of that BFGS solve.
    """
    _, grad0, _, _ = _dual_g_and_grad(0.0, old_var, target_var, eps, omega_offset)
    # At η=0: g'=ε−KL. If KL≤ε then g'≥0 and bound-constrained min is η=0.
    if grad0 >= -1e-12:
        return 0.0, True

    eta = 1.0
    succ = False
    for _ in range(max_eval):
        dual, grad, _, eta = _dual_g_and_grad(eta, old_var, target_var, eps, omega_offset)
        if abs(grad) < 1e-10:
            succ = True
            break
        eps_h = max(1e-6, 1e-4 * (abs(eta) + 1.0))
        _, grad_p, _, _ = _dual_g_and_grad(eta + eps_h, old_var, target_var, eps, omega_offset)
        hess = (grad_p - grad) / eps_h
        if abs(hess) < 1e-12:
            # Gradient descent on dual: η ← η − g'. When g'=ε−KL<0, η increases.
            step = grad  # so eta - step = eta - g' increases when g'<0
        else:
            # Newton root of g'=0: η ← η − g'/g''
            step = grad / hess
        eta_trial = max(0.0, eta - step)
        dual_trial, _, _, _ = _dual_g_and_grad(eta_trial, old_var, target_var, eps, omega_offset)
        t = 1.0
        while dual_trial > dual + 1e-12 and t > 1e-8:
            t *= 0.5
            eta_trial = max(0.0, eta - t * step)
            dual_trial, _, _, _ = _dual_g_and_grad(eta_trial, old_var, target_var, eps, omega_offset)
        if abs(eta_trial - eta) < 1e-14:
            succ = abs(grad) < 1e-6
            break
        eta = eta_trial
    else:
        _, grad, _, eta = _dual_g_and_grad(eta, old_var, target_var, eps, omega_offset)
        succ = abs(grad) < 1e-5 or grad >= -1e-5
    return float(eta), succ


def _last_eta_grad(eta: float, old_var: torch.Tensor, target_var: torch.Tensor,
                   projected_var: torch.Tensor, omega_offset: float = 1.0) -> torch.Tensor:
    """Official last_eta_grad: ∂η/∂Q_target factors (KKT)."""
    if eta <= 1e-12:
        return torch.zeros_like(old_var)
    old_prec = 1.0 / old_var.clamp_min(1e-12)
    target_prec = 1.0 / target_var.clamp_min(1e-12)
    # C++ uses (eta + omega_offset) not squared in last_eta_grad
    dQ_deta = (omega_offset * old_prec - target_prec) / (eta + omega_offset)
    f2_dQ = projected_var * (1.0 - old_prec * projected_var)
    denom = (f2_dQ * dQ_deta).sum()
    if float(denom.abs()) < 1e-10:
        return torch.zeros_like(old_var)
    c = -1.0 / denom
    return c * f2_dQ


class DiagCovKLProjectionFn(Function):
    """Official DiagCovOnlyKLProjection forward/backward for one diag covariance (A,).

    Identity short-circuit when KL_cov(target||old) ≤ eps (official BatchedDiagCovOnlyProjection).
    """

    @staticmethod
    def forward(ctx, target_var: torch.Tensor, old_var: torch.Tensor, eps: torch.Tensor):
        omega_offset = 1.0
        eps_f = float(eps.detach().reshape(()))
        old_v = old_var.detach()
        tgt_v = target_var.detach()

        # Identity path: already inside cov trust region (official)
        old_logdet = old_v.clamp_min(1e-12).log().sum()
        tgt_logdet = tgt_v.clamp_min(1e-12).log().sum()
        dim = old_v.numel()
        kl0 = 0.5 * (old_logdet - dim - tgt_logdet + (tgt_v / old_v.clamp_min(1e-12)).sum())
        if float(kl0) <= eps_f + 1e-12:
            ctx.eta = 0.0
            ctx.omega_offset = omega_offset
            ctx.identity = True
            ctx.save_for_backward(tgt_v, old_v, tgt_v.clone())
            return target_var  # identity — true autograd path

        eta, succ = _solve_eta_lbfgs_1d(old_v, tgt_v, eps_f, omega_offset, max_eval=100)
        if not succ:
            # Robust 1-D bisection on g'(η)=ε−KL=0 (monotone), not ad-hoc η hike
            lo, hi = 0.0, 1.0
            for _ in range(40):
                _, g_hi, _, _ = _dual_g_and_grad(hi, old_v, tgt_v, eps_f, omega_offset)
                if g_hi >= 0.0:
                    break
                hi *= 2.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                _, g_mid, _, _ = _dual_g_and_grad(mid, old_v, tgt_v, eps_f, omega_offset)
                if g_mid >= 0.0:
                    hi = mid
                else:
                    lo = mid
            eta = hi

        old_prec = 1.0 / old_v.clamp_min(1e-12)
        target_prec = 1.0 / tgt_v.clamp_min(1e-12)
        projected_prec = (eta * old_prec + target_prec) / (eta + omega_offset)
        projected_var = 1.0 / projected_prec.clamp_min(1e-12)

        ctx.eta = float(eta)
        ctx.omega_offset = omega_offset
        ctx.identity = False
        ctx.save_for_backward(tgt_v, old_v, projected_var)
        return projected_var.to(dtype=target_var.dtype, device=target_var.device)

    @staticmethod
    def backward(ctx, d_proj_var: torch.Tensor):
        if getattr(ctx, "identity", False):
            return d_proj_var, None, None
        target_var, old_var, projected_var = ctx.saved_tensors
        eta = ctx.eta
        omega_offset = ctx.omega_offset
        # C++: d_Q = -proj_var ⊙ d_cov ⊙ proj_var  (d_cov = ∂L/∂ projected_var)
        d_Q = -projected_var * d_proj_var * projected_var
        eo = omega_offset + eta
        old_prec = 1.0 / old_var.clamp_min(1e-12)
        target_prec = 1.0 / target_var.clamp_min(1e-12)
        # C++ backward uses eo_squared for dQ_deta
        dQ_deta = (omega_offset * old_prec - target_prec) / (eo * eo)
        d_eta = (d_Q * dQ_deta).sum()
        deta_dQ_target = _last_eta_grad(eta, old_var, target_var, projected_var, omega_offset)
        d_Q_target = d_eta * deta_dQ_target + d_Q / eo
        d_cov_target = -target_prec * d_Q_target * target_prec
        d_cov_target = torch.nan_to_num(d_cov_target, nan=0.0, posinf=0.0, neginf=0.0)
        return d_cov_target, None, None


def _solve_eta_batched(old_var: torch.Tensor, target_var: torch.Tensor, eps: float,
                       omega_offset: float = 1.0, max_eval: int = 25) -> torch.Tensor:
    """Vectorized dual η ≥ 0 for batch of diag covs. old/target_var: (B,A) → eta (B,).

    Bracket + bisection on g'(η)=ε−KL (monotone in η). Fully tensorized — O(1) Python,
    no per-row loop (was the SPS killer for contextual Beta TRL).
    """
    B, dim = old_var.shape
    device, dtype = old_var.device, old_var.dtype
    old_var = old_var.clamp_min(1e-12)
    target_var = target_var.clamp_min(1e-12)
    old_prec = 1.0 / old_var
    target_prec = 1.0 / target_var
    old_logdet = old_var.log().sum(-1)

    def dual_grad(eta: torch.Tensor) -> torch.Tensor:
        eo = (eta + omega_offset).unsqueeze(-1)
        new_prec = (eta.unsqueeze(-1) * old_prec + target_prec) / eo
        new_var = 1.0 / new_prec.clamp_min(1e-12)
        new_logdet = new_var.log().sum(-1)
        kl = 0.5 * (old_logdet - dim - new_logdet + (old_prec * new_var).sum(-1))
        return eps - kl

    g0 = dual_grad(torch.zeros(B, device=device, dtype=dtype))
    need = g0 < -1e-12
    if not need.any():
        return torch.zeros(B, device=device, dtype=dtype)

    # Bracket: expand hi until g'(hi) ≥ 0 on active rows
    lo = torch.zeros(B, device=device, dtype=dtype)
    hi = torch.ones(B, device=device, dtype=dtype)
    for _ in range(16):
        g_hi = dual_grad(hi)
        grow = need & (g_hi < 0)
        if not grow.any():
            break
        hi = torch.where(grow, hi * 2.0, hi)
    # Bisection
    for _ in range(max_eval):
        mid = 0.5 * (lo + hi)
        g_mid = dual_grad(mid)
        # g'≥0 ⇒ η large enough (KL≤eps); shrink hi
        geq = g_mid >= 0
        hi = torch.where(need & geq, mid, hi)
        lo = torch.where(need & ~geq, mid, lo)
    eta = torch.where(need, hi, torch.zeros_like(hi))
    return eta


class BatchedDiagCovKLProjectionFn(Function):
    """Batched (B,A) version of DiagCovOnlyKLProjection — no Python row loop.

    η=0 rows are true identity (autograd through target_var); active rows use KKT backward.
    """

    @staticmethod
    def forward(ctx, target_var: torch.Tensor, old_var: torch.Tensor, eps: torch.Tensor):
        omega_offset = 1.0
        eps_f = float(eps.detach().reshape(()))
        old_v = old_var.detach()
        tgt_v = target_var.detach()
        eta = _solve_eta_batched(old_v, tgt_v, eps_f, omega_offset, max_eval=40)  # (B,)
        inside = eta <= 1e-12

        old_prec = 1.0 / old_v.clamp_min(1e-12)
        target_prec = 1.0 / tgt_v.clamp_min(1e-12)
        eo = (eta + omega_offset).unsqueeze(-1)
        projected_prec = (eta.unsqueeze(-1) * old_prec + target_prec) / eo
        projected_var = 1.0 / projected_prec.clamp_min(1e-12)

        # Identity rows: return target_var (live graph). Active: projected buffer.
        out = torch.where(inside.unsqueeze(-1), target_var, projected_var.to(dtype=target_var.dtype))
        ctx.omega_offset = omega_offset
        ctx.eps_f = eps_f
        ctx.save_for_backward(tgt_v, old_v, projected_var, eta, inside)
        return out

    @staticmethod
    def backward(ctx, d_proj_var: torch.Tensor):
        target_var, old_var, projected_var, eta, inside = ctx.saved_tensors
        omega_offset = ctx.omega_offset
        # Identity rows: pass grad through
        d_out = d_proj_var.clone()
        active = ~inside
        if not active.any():
            return d_out, None, None

        # KKT backward only on active rows (vectorized)
        pv = projected_var
        d_Q = -pv * d_proj_var * pv
        eo = omega_offset + eta  # (B,)
        old_prec = 1.0 / old_var.clamp_min(1e-12)
        target_prec = 1.0 / target_var.clamp_min(1e-12)
        dQ_deta = (omega_offset * old_prec - target_prec) / (eo * eo).unsqueeze(-1)
        d_eta = (d_Q * dQ_deta).sum(-1)  # (B,)

        # last_eta_grad per row: c * f2_dQ with c = -1 / sum(f2_dQ * dQ_deta_unsquared form)
        # Official _last_eta_grad uses dQ_deta = (ω old_prec - tgt) / (η+ω)  [not squared]
        dQ_deta_kkt = (omega_offset * old_prec - target_prec) / eo.unsqueeze(-1)
        f2_dQ = pv * (1.0 - old_prec * pv)
        denom = (f2_dQ * dQ_deta_kkt).sum(-1).clamp(min=1e-10)  # avoid 0
        # where eta~0 denom unused
        c = -1.0 / denom
        deta_dQ_target = c.unsqueeze(-1) * f2_dQ  # (B,A)
        # zero for eta=0
        deta_dQ_target = torch.where((eta > 1e-12).unsqueeze(-1), deta_dQ_target, torch.zeros_like(deta_dQ_target))

        d_Q_target = d_eta.unsqueeze(-1) * deta_dQ_target + d_Q / eo.unsqueeze(-1)
        d_cov_target = -target_prec * d_Q_target * target_prec
        d_cov_target = torch.nan_to_num(d_cov_target, nan=0.0, posinf=0.0, neginf=0.0)

        # Mix: identity rows keep d_proj; active use KKT
        d_out = torch.where(inside.unsqueeze(-1), d_proj_var, d_cov_target)
        return d_out, None, None


def project_cov_kl_diag(logstd: torch.Tensor, old_logstd: torch.Tensor, eps_cov: float,
                        log_std_min: float = -5.0, log_std_max: float = 2.0,
                        contextual: bool = False):
    """Project diag log-std (batch B,A).

    Official KLProjectionLayer:
      - non-contextual: dual on row 0 only, expand (set_std path)
      - contextual: batched per-state dual (vectorized BatchedDiagCovOnlyProjection)
    """
    logstd = logstd.clamp(log_std_min, log_std_max)
    old_logstd = old_logstd.clamp(log_std_min, log_std_max)
    var = (2.0 * logstd).exp().clamp_min(1e-12)
    old_var = (2.0 * old_logstd).exp().clamp_min(1e-12)
    eps_t = torch.as_tensor(float(eps_cov), device=var.device, dtype=var.dtype)
    if not contextual:
        proj0 = DiagCovKLProjectionFn.apply(var[0], old_var[0], eps_t)
        var_p = proj0.unsqueeze(0).expand_as(var)
    else:
        var_p = BatchedDiagCovKLProjectionFn.apply(var, old_var, eps_t)
    logstd_p = (0.5 * var_p.log()).clamp(log_std_min, log_std_max)
    return logstd_p


def project_mean_kl(mean: torch.Tensor, old_mean: torch.Tensor, old_logstd: torch.Tensor, eps_mu: float):
    mean_part = kl_mean_part(mean, old_mean, old_logstd)
    return mean_projection(mean, old_mean, mean_part, eps_mu), mean_part.detach()


def project_policy_kl(mean, logstd, old_mean, old_logstd, eps_mu, eps_cov,
                      log_std_min=-5.0, log_std_max=2.0, contextual_std: bool = False):
    """Full paper KL projection (diag): mean closed form + cov dual.

    Matches official KLProjectionLayer._trust_region_projection.
    """
    mean_p, mean_part = project_mean_kl(mean, old_mean, old_logstd, eps_mu)
    logstd_p = project_cov_kl_diag(
        logstd, old_logstd, eps_cov, log_std_min, log_std_max, contextual=contextual_std
    )
    cov_part = kl_cov_part(logstd, old_logstd).detach()
    return mean_p, logstd_p, mean_part, cov_part


def trust_region_aux_loss(mean, logstd, mean_p, logstd_p, contextual_std: bool = False):
    """Paper §4.4 / official get_trust_region_loss for KL projection.

    - Projection target is detached (supervised signal only; no grad through layer).
    - Metric is reverse KL(π_θ || π̃) (same family as the projection).
    - Non-contextual std: only mean term (std is hard-set separately).
    """
    mean_t = mean_p.detach()
    logstd_t = logstd_p.detach()
    # KL mean part w.r.t. *projected* covariance (official maha uses std of q=target)
    mean_diff = kl_mean_part(mean, mean_t, logstd_t)
    if contextual_std:
        cov_diff = kl_cov_part(logstd, logstd_t)
        return (mean_diff + cov_diff).mean()
    return mean_diff.mean()


# ---------------------------------------------------------------------------
# Beta-policy TRL — paper KL layer via (mean, diag-var) moments of each Beta
# ---------------------------------------------------------------------------
#
# Otto et al. only define mean/cov projections for Gaussians. The paper-faithful
# map onto factorized Beta is:
#   1) encode Beta(α,β) by its mean μ=α/(α+β) and variance σ²=μ(1-μ)/(α+β+1)
#      (equiv. log-std), i.e. the same (mean, diag cov) factorization as the paper;
#   2) run the *exact* paper KL projection on those moments (mean Eq.6 + cov dual);
#   3) decode (μ̃, σ̃²) → Beta(α̃,β̃) with α=μ κ, β=(1-μ)κ, κ=μ(1-μ)/σ²−1;
#   4) aux = official get_trust_region_loss on the moment Gaussians (contextual).
#
# The rollout/surrogate stay on the Beta density (native z), not on a Gaussian.


def beta_to_mean_logstd(alpha: torch.Tensor, beta: torch.Tensor):
    """Beta → (μ, log σ) in native z-space (factorized, matches paper mean/cov split)."""
    a = alpha.clamp_min(1e-6)
    b = beta.clamp_min(1e-6)
    mu = a / (a + b)
    # Var[Z] = μ(1-μ)/(α+β+1)
    var = (mu * (1.0 - mu) / (a + b + 1.0)).clamp_min(1e-12)
    logstd = 0.5 * var.log()
    return mu, logstd


def mean_logstd_to_beta(mu: torch.Tensor, logstd: torch.Tensor, param_min: float = 1.0 + 1e-5):
    """(μ, log σ) → Beta(α,β) with α,β ≥ param_min (unimodal-friendly)."""
    mu = mu.clamp(1e-4, 1.0 - 1e-4)
    var = (2.0 * logstd).exp().clamp_min(1e-12)
    # Feasible Beta var is < μ(1-μ); also κ = α+β ≥ 2*param_min ⇒ var ≤ μ(1-μ)/(κ+1)
    kappa_min = 2.0 * param_min
    var_max = (mu * (1.0 - mu) / (kappa_min + 1.0)).clamp_min(1e-12)
    var = torch.minimum(var, var_max)
    kappa = (mu * (1.0 - mu) / var - 1.0).clamp_min(kappa_min)
    alpha = (mu * kappa).clamp_min(param_min)
    beta = ((1.0 - mu) * kappa).clamp_min(param_min)
    return alpha, beta


def beta_kl_reverse(alpha: torch.Tensor, beta: torch.Tensor,
                    old_alpha: torch.Tensor, old_beta: torch.Tensor) -> torch.Tensor:
    """Per-dim reverse KL(Beta(α,β) || Beta(α₀,β₀)) for logging / diagnostics."""
    a, b = alpha.clamp_min(1e-6), beta.clamp_min(1e-6)
    a0, b0 = old_alpha.clamp_min(1e-6), old_beta.clamp_min(1e-6)
    log_bq = torch.lgamma(a0) + torch.lgamma(b0) - torch.lgamma(a0 + b0)
    log_bp = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    return (
        log_bq - log_bp
        + (a - a0) * torch.digamma(a)
        + (b - b0) * torch.digamma(b)
        + (a0 - a + b0 - b) * torch.digamma(a + b)
    )


def project_policy_beta_kl(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    old_alpha: torch.Tensor,
    old_beta: torch.Tensor,
    eps_mu: float,
    eps_cov: float,
    log_std_min: float = -8.0,
    log_std_max: float = 2.0,
    param_min: float = 1.0 + 1e-5,
):
    """Paper KL TRL on Beta via moment (mean, diag-var) projection.

    Returns (alpha_p, beta_p, mean_part, cov_part, mean, logstd, mean_p, logstd_p)
    so the caller can run official get_trust_region_loss on the moment Gaussians.
    """
    mean, logstd = beta_to_mean_logstd(alpha, beta)
    old_mean, old_logstd = beta_to_mean_logstd(old_alpha, old_beta)
    # Both α,β are state-dependent → contextual_std=True (official branch)
    mean_p, logstd_p, mean_part, cov_part = project_policy_kl(
        mean, logstd, old_mean, old_logstd, eps_mu, eps_cov,
        log_std_min, log_std_max, contextual_std=True,
    )
    alpha_p, beta_p = mean_logstd_to_beta(mean_p, logstd_p, param_min=param_min)
    return alpha_p, beta_p, mean_part, cov_part, mean, logstd, mean_p, logstd_p


def trust_region_aux_loss_beta_moments(mean, logstd, mean_p, logstd_p):
    """Official get_trust_region_loss on Beta moment Gaussians (contextual)."""
    return trust_region_aux_loss(mean, logstd, mean_p, logstd_p, contextual_std=True)

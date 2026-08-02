"""Shared IDBD optimizer (Sutton 1992, unit-feature generalization)."""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class IDBD(Optimizer):
    """Parameter-wise Incremental Delta-Bar-Delta (Sutton 1992).

    Linear LMS (paper) uses features x; here each parameter is treated as a
    unit-feature weight and the backprop gradient g = ∂L/∂w plays the role of
    -δx. Updates (elementwise):

        β  ← β - θ · g · h
        α  ← exp(β)   (clamped)
        w  ← w - α · g
        h  ← [1 - α]₊ · h - α · g

    h tracks recent weight changes Δw = -α g. Matching gradient signs grow α.

    Diagnostics (see `pop_diagnostics`):
      - α spread / cap fraction: is meta-learning differentiating step-sizes?
      - meta_dot = E[-g·h]: >0 means successive grads agree → growing α (healthy)
      - h_abs: trace alive? near-0 means no memory / no meta signal
      - effective_step = E[α|g|]: actual weight-update magnitude
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        meta_lr: float = 0.05,
        max_alpha: float = 0.1,
        paper_beta_bounds: bool = False,
        eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr (initial alpha): {lr}")
        if meta_lr < 0.0:
            raise ValueError(f"Invalid meta_lr: {meta_lr}")
        if max_alpha <= 0.0:
            raise ValueError(f"Invalid max_alpha: {max_alpha}")
        defaults = dict(
            lr=lr, meta_lr=meta_lr, max_alpha=max_alpha, eps=eps,
            paper_beta_bounds=paper_beta_bounds,
        )
        super().__init__(params, defaults)
        self._reset_step_accum()

    def _reset_step_accum(self):
        self._acc = {
            "n": 0,
            "meta_dot": 0.0,  # sum of (-g*h); positive => grow α
            "meta_abs": 0.0,  # sum of |g*h|
            "h_abs": 0.0,
            "eff_step": 0.0,  # sum of α|g|
            "sign_agree": 0.0,  # sum of 1[sign(g)==sign(-h)] i.e. g opposes h? wait
            # sign agreement for *growing* α: sign(-g)==sign(h) i.e. g and h opposite
            # since Δw=-αg, h~Δw; same direction of weight updates means g same sign as previous g
            # meta_signal = -g*h; agree when -g*h > 0
            "grow_frac": 0.0,  # fraction of params with -g*h > 0 this step (sum of counts)
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            meta_lr = group["meta_lr"]
            max_alpha = group["max_alpha"]
            paper_beta_bounds = group.get("paper_beta_bounds", False)
            init_beta = math.log(group["lr"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("IDBD does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["beta"] = torch.full_like(p, init_beta)
                    state["h"] = torch.zeros_like(p)

                beta = state["beta"]
                h = state["h"]

                # Meta-learning signal *before* updating h (uses current memory)
                # meta_signal_i = -g_i h_i; E[meta_signal]>0 ⇒ α growing
                meta_signal = -(grad * h)
                n = grad.numel()
                self._acc["n"] += n
                self._acc["meta_dot"] += meta_signal.sum().item()
                self._acc["meta_abs"] += meta_signal.abs().sum().item()
                self._acc["h_abs"] += h.abs().sum().item()
                self._acc["grow_frac"] += (meta_signal > 0).sum().item()

                # Meta-update log step-sizes: β ← β - θ g h = β + θ · meta_signal
                d_beta = -meta_lr * grad * h
                if paper_beta_bounds:
                    d_beta.clamp_(-2.0, 2.0)
                beta.add_(d_beta)
                if paper_beta_bounds:
                    beta.clamp_(min=-10.0, max=math.log(max_alpha))
                else:
                    beta.clamp_(max=math.log(max_alpha))
                alpha = beta.exp()

                # Weight update: w ← w - α g  (true Δw = -α g)
                step = alpha * grad
                self._acc["eff_step"] += step.abs().sum().item()
                p.add_(step, alpha=-1.0)

                # Trace of recent updates: h ← [1-α]₊ h + Δw
                decay = (1.0 - alpha).clamp_(min=0.0)
                h.mul_(decay).sub_(step)

        return loss

    @torch.no_grad()
    def pop_diagnostics(self):
        """Snapshot α state + flush per-step meta accumulators (call once per PPO iter)."""
        alphas = []
        h_vals = []
        init_alphas = []
        max_alphas = []
        for group in self.param_groups:
            max_alpha = group["max_alpha"]
            init_a = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                if "beta" not in state:
                    continue
                a = state["beta"].exp().clamp(max=max_alpha).flatten()
                alphas.append(a)
                h_vals.append(state["h"].flatten())
                init_alphas.append(torch.full_like(a, init_a))
                max_alphas.append(torch.full_like(a, max_alpha))

        out = {
            "alpha_mean": 0.0,
            "alpha_median": 0.0,
            "alpha_std": 0.0,
            "alpha_p10": 0.0,
            "alpha_p90": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 0.0,
            "alpha_log_mean": 0.0,
            "frac_at_max": 0.0,
            "frac_above_init": 0.0,
            "frac_below_init": 0.0,
            "alpha_vs_init_ratio": 1.0,
            "h_abs_mean": 0.0,
            "meta_dot_mean": 0.0,
            "meta_abs_mean": 0.0,
            "grow_frac": 0.0,
            "eff_step_mean": 0.0,
        }
        if alphas:
            a = torch.cat(alphas)
            a0 = torch.cat(init_alphas)
            amax = torch.cat(max_alphas)
            hcat = torch.cat(h_vals)
            out["alpha_mean"] = a.mean().item()
            out["alpha_median"] = a.median().item()
            out["alpha_std"] = a.std(unbiased=False).item()
            out["alpha_p10"] = a.quantile(0.1).item()
            out["alpha_p90"] = a.quantile(0.9).item()
            out["alpha_min"] = a.min().item()
            out["alpha_max"] = a.max().item()
            out["alpha_log_mean"] = a.clamp_min(1e-12).log().mean().exp().item()  # geom mean
            out["frac_at_max"] = (a >= amax * 0.99).float().mean().item()
            out["frac_above_init"] = (a > a0 * 1.05).float().mean().item()
            out["frac_below_init"] = (a < a0 * 0.95).float().mean().item()
            out["alpha_vs_init_ratio"] = (a.mean() / (a0.mean() + 1e-12)).item()
            out["h_abs_mean"] = hcat.abs().mean().item()

        n = max(self._acc["n"], 1)
        out["meta_dot_mean"] = self._acc["meta_dot"] / n
        out["meta_abs_mean"] = self._acc["meta_abs"] / n
        out["grow_frac"] = self._acc["grow_frac"] / n
        out["eff_step_mean"] = self._acc["eff_step"] / n
        # h_abs from accum is average over steps; prefer current state above
        if out["h_abs_mean"] == 0.0 and self._acc["n"] > 0:
            out["h_abs_mean"] = self._acc["h_abs"] / n

        self._reset_step_accum()
        return out


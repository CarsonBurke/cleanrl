"""Streaming Bayesian state-space learner on the SPY stream (direction 1 of the finance rethink).

Model: y_t = w_t . x_t + e_t,  w_t = w_{t-1} + q_t,  e_t ~ N(0, R_t),  q_t ~ N(0, Q).
Exact posterior = Kalman filter. Plasticity is the posterior covariance: the gain
    k_t = P x_t / (x_t' P x_t + R_t)
is per-parameter and depends on the current state x_t (its leverage in the posterior metric),
and the correlated lagged features are decorrelated by P (diagonal Adam cannot do this).

Everything is vectorised over a grid of (prior scale, drift scale, robustness) so each config is
compared at its own best setting; every config sees the identical predict-before-update stream.
    R_t : running residual variance (EMA-free: cumulative until warmup, then a slow EMA is NOT used;
          instead a Student-t style graded influence for fat tails: weight (nu+1)/(nu+z^2)).
    Q   : drift per parameter, scalar q * I on a per-feature-variance scale (grid axis).
    P_0 : prior variance tau^2 I (grid axis); tau^2 = 0 disables learning (zero predictor).
Scores: relative MSE vs zero predictor over bars >= score_after, real and time-permuted target.

    .venv/bin/python cleanrl/plasticity/kalman_stream.py
"""
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro

sys.path.insert(0, "/home/marvin/Documents/repositories/cleanrl")
from cleanrl.plasticity.stock_stream import Args as StreamArgs, read_bars, build_stream  # noqa: E402


@dataclass
class Args:
    taus: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
    """prior variance of each coefficient (features are unit-scale)"""
    qs: tuple[float, ...] = (0.0, 1e-8, 1e-7, 1e-6, 1e-5)
    """per-step drift variance per coefficient (0 = stationary RLS with prior)"""
    nus: tuple[float, ...] = (0.0, 4.0)
    """Student-t df for graded influence (0 = Gaussian)"""
    score_after: int = 20000
    permuted: bool = True
    seed: int = 1
    raw_target: bool = False
    vol_feature: bool = False


def main():
    a = tyro.cli(Args)
    sa = StreamArgs()
    sa.raw_target, sa.vol_feature = a.raw_target, a.vol_feature
    feats, target = build_stream(read_bars(sa.bars), sa)
    dev = torch.device("cuda")
    X = torch.tensor(feats, device=dev, dtype=torch.float32)
    y = torch.tensor(target, device=dev, dtype=torch.float32)
    n, d = X.shape
    X = torch.cat([X, torch.ones(n, 1, device=dev)], 1)
    D = d + 1
    streams = [y]
    if a.permuted:
        g = torch.Generator(device=dev).manual_seed(a.seed)
        streams.append(y[torch.randperm(n, device=dev, generator=g)])
    cfgs = [(tau, q, nu, s) for tau in a.taus for q in a.qs for nu in a.nus for s in range(len(streams))]
    C = len(cfgs)
    tau = torch.tensor([c[0] for c in cfgs], device=dev)
    q = torch.tensor([c[1] for c in cfgs], device=dev)
    nu = torch.tensor([c[2] for c in cfgs], device=dev)
    sidx = torch.tensor([c[3] for c in cfgs], device=dev)
    Y = torch.stack(streams, 0)  # (S, n)

    w = torch.zeros(C, D, device=dev)
    P = tau.view(C, 1, 1) * torch.eye(D, device=dev).expand(C, D, D).clone()
    r_var = torch.ones(C, device=dev)   # observation noise estimate (cumulative mean of residual^2)
    r_cnt = torch.zeros(C, device=dev)
    se = torch.zeros(C, device=dev)
    ss = torch.zeros(C, device=dev)
    t0 = time.time()
    for t in range(n):
        x = X[t]                                        # (D,)
        yt = Y[sidx, t]                                 # (C,)
        pred = w @ x                                    # (C,)
        res = yt - pred
        if t >= a.score_after:
            se += res.square()
            ss += yt.square()
        # predict step: drift
        P.diagonal(dim1=-2, dim2=-1).add_(q.unsqueeze(1))
        Px = P @ x                                      # (C, D)
        s = (Px @ x) + r_var                            # innovation variance (C,)
        z2 = res.square() / s
        infl = torch.where(nu > 0, (nu + 1.0) / (nu + z2), torch.ones_like(z2))   # graded fat-tail influence
        k = Px / s.unsqueeze(1) * infl.unsqueeze(1)
        w += k * res.unsqueeze(1)
        P -= (k.unsqueeze(2) * Px.unsqueeze(1))          # (I - k x') P
        # observation-noise estimate: cumulative mean of squared residual (predict-before-update)
        r_cnt += 1
        r_var += (res.square() * infl - r_var) / r_cnt
        if t % 100000 == 0 and t > 0:
            print(f"  t={t} {time.time() - t0:.0f}s", file=sys.stderr)
    rel = (se / ss).cpu().numpy()
    print(f"{n} bars, {D} params, {C} configs, {time.time() - t0:.0f}s")
    print(f"{'tau':>8s} {'q':>8s} {'nu':>4s} | {'REAL':>8s} {'PERM':>8s} {'gap':>8s}")
    rows = {}
    for (tau_, q_, nu_, s_), r in zip(cfgs, rel):
        rows.setdefault((tau_, q_, nu_), {})[s_] = r
    for key, vals in rows.items():
        real, perm = vals.get(0, np.nan), vals.get(1, np.nan)
        print(f"{key[0]:8.0e} {key[1]:8.0e} {key[2]:4.0f} | {real:8.5f} {perm:8.5f} {perm - real:+8.5f}")


if __name__ == "__main__":
    main()

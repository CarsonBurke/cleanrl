"""High-dimensional cross-sectional vol panel: features assembled per bar from rolling history.

Per (bar t, stock i): 8 channels x L lags = own z, own |z|, own volume surprise, market z, market |z|,
cross-sectional mean z and |z|, own |z| squared  ->  F = 8 L features (+ bias). Target: next-bar squared
unit-vol return, centred on the fit-window mean. Most features are weak, correlated, or noise.

One streaming pass runs every learner on the identical predict-before-update stream and accumulates
the ridge Gram over the fit window (first 60% of time); a second pass scores hindsight ridge on the
test window (last 40%). Time-permuted target control runs alongside.

Learners (all pooled linear, one step per bar on the cross-section):
  adam        : LR grid
  adamw       : LR x weight-decay grid
Relative MSE vs the constant-mean predictor on the test window; lower is better.

    .venv/bin/python cleanrl/plasticity/panel_hd.py --lags 32
"""
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro

sys.path.insert(0, "/home/marvin/Documents/repositories/cleanrl")
from cleanrl.plasticity.panel_stream import Args as PanelArgs, build_panel, _ewma  # noqa: E402


@dataclass
class Args(PanelArgs):
    lags: int = 32
    lams: tuple[float, ...] = (1e2, 1e3, 1e4, 1e5, 1e6)
    lr_grid: tuple[float, ...] = (3e-5, 1e-4, 3e-4, 1e-3)
    wds: tuple[float, ...] = (0.0, 1e-3, 1e-2)


def series(a, close, volume):
    T, M = close.shape
    cov = np.isfinite(close[:, 1:]).mean(1)
    keep = cov >= a.min_coverage
    close, volume = close[keep], volume[keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.diff(np.log(close), axis=0, prepend=np.nan)
        ret[np.abs(ret) > 0.2] = np.nan
        scale = np.sqrt(_ewma(np.square(ret), a.vol_span))
        scale_lag = np.concatenate([np.full((1, M), np.nan), scale[:-1]], 0)
        z = np.clip(ret / np.maximum(scale_lag, 1e-6), -10, 10)
        lv = np.log1p(np.nan_to_num(volume, nan=0.0))
        vs = lv - _ewma(lv, a.vol_span)
    valid = np.isfinite(z[:, 1:])
    z = np.nan_to_num(z, nan=0.0).astype(np.float32)
    vs = np.nan_to_num(vs, nan=0.0).astype(np.float32)
    csz = np.where(valid, z[:, 1:], np.nan)
    with np.errstate(invalid="ignore"):
        cs = np.nan_to_num(np.nanmean(csz, 1), nan=0.0).astype(np.float32)
        acs = np.nan_to_num(np.nanmean(np.abs(csz), 1), nan=0.0).astype(np.float32)
    return z, vs, cs, acs, valid


class Bank:
    """Rolling-history feature bank on GPU: feats(t) -> (N, F) causal features for bar t."""

    def __init__(self, a):
        close, volume, _, _ = build_panel(a)
        z, vs, cs, acs, valid = series(a, close, volume)
        T, M = z.shape
        self.T, self.N, self.L = T, M - 1, a.lags
        dev = torch.device("cuda")
        self.dev = dev
        self.zt = torch.tensor(z, device=dev); self.vst = torch.tensor(vs, device=dev)
        self.cst = torch.tensor(cs, device=dev); self.acst = torch.tensor(acs, device=dev)
        vt = torch.tensor(valid, device=dev)
        y_all = torch.roll(self.zt[:, 1:], -1, 0).square().clamp_max(25.0)
        vmask = vt & torch.roll(vt, -1, 0)
        vmask[:self.L + 1] = False; vmask[-1] = False
        self.cut = int(0.6 * T)
        self.mu = y_all[:self.cut][vmask[:self.cut]].mean()
        self.y = torch.where(vmask, y_all - self.mu, torch.zeros_like(y_all))
        self.valid = vmask
        g = torch.Generator(device=dev).manual_seed(0)
        self.perm_t = torch.randperm(T, device=dev, generator=g)
        self.F = 8 * self.L + 1
        self.lag_idx = torch.arange(1, self.L + 1, device=dev)
        self.ones = torch.ones(self.N, 1, device=dev)

    def feats(self, t):
        idx = t - self.lag_idx                              # (L,)
        N, L = self.N, self.L
        own = self.zt[idx, 1:].T                            # (N, L)
        aown = own.abs()
        return torch.cat([own, aown, self.vst[idx, 1:].T,
                          self.zt[idx, 0].expand(N, L), self.zt[idx, 0].abs().expand(N, L),
                          self.cst[idx].expand(N, L), self.acst[idx].expand(N, L),
                          aown.square(), self.ones], 1)     # (N, F)

    def streams(self):
        return (("REAL", self.y, self.valid), ("PERM", self.y[self.perm_t], self.valid[self.perm_t]))


def main():
    a = tyro.cli(Args)
    bank = Bank(a)
    T, N, L, F, cut, dev = bank.T, bank.N, bank.L, bank.F, bank.cut, bank.dev
    feats, mu = bank.feats, bank.mu

    streams = bank.streams()
    cfgs = [(lr, wd) for lr in a.lr_grid for wd in a.wds]
    C = len(cfgs)
    lr = torch.tensor([c[0] for c in cfgs], device=dev).view(C, 1)
    wd = torch.tensor([c[1] for c in cfgs], device=dev).view(C, 1)
    print(f"panel {T} bars x {N} stocks, F={F}, fit window {cut} bars, test {T - cut} bars, mean target {mu.item():.4f}")
    t0 = time.time()
    for name, yy, vv in streams:
        G = torch.zeros(F, F, device=dev, dtype=torch.float64); c = torch.zeros(F, device=dev, dtype=torch.float64)
        w = torch.zeros(C, F, device=dev); m = torch.zeros_like(w); v = torch.zeros_like(w)
        se = torch.zeros(C, device=dev); ss = torch.zeros((), device=dev)
        for t in range(L + 1, T - 1):
            mk = vv[t]
            n_ok = mk.sum()
            if n_ok == 0:
                continue
            x = feats(t)[mk]                                # (n, F)
            yb = yy[t][mk]
            if t < cut:
                xd = x.double(); G += xd.T @ xd; c += xd.T @ yb.double()
            pred = w @ x.T                                  # (C, n)
            res = pred - yb
            if t >= cut:
                se += res.square().sum(1); ss += yb.square().sum()
            grad = (res @ x) / n_ok
            m.mul_(0.9).add_(grad, alpha=0.1); v.mul_(0.999).addcmul_(grad, grad, value=0.001)
            step = (m / (1 - 0.9 ** (t + 1))) / ((v / (1 - 0.999 ** (t + 1))).sqrt() + 1e-8)
            w -= lr * (step + wd * w)
        print(f"# {name} streaming ({time.time() - t0:.0f}s)")
        for (lr_, wd_), r in zip(cfgs, (se / ss).tolist()):
            print(f"  adamw lr={lr_:.0e} wd={wd_:.0e}: {r:.5f}")
        ws = {lam: torch.linalg.solve(G + lam * torch.eye(F, device=dev, dtype=torch.float64), c).float() for lam in a.lams}
        se_r = {lam: 0.0 for lam in a.lams}; ss_r = 0.0
        for t in range(cut, T - 1):
            mk = vv[t]
            if mk.sum() == 0:
                continue
            x = feats(t)[mk]; yb = yy[t][mk]
            ss_r += yb.square().sum().item()
            for lam in a.lams:
                se_r[lam] += (x @ ws[lam] - yb).square().sum().item()
        for lam in a.lams:
            print(f"  ridge lam={lam:.0e}: {se_r[lam] / ss_r:.5f}")
        if name == "REAL":
            wbest = ws[min(a.lams, key=lambda l: se_r[l])]
            names = ["own", "|own|", "vsurp", "mkt", "|mkt|", "cs", "|cs|", "own^2"]
            blocks = [wbest[k * L:(k + 1) * L].abs().sum().item() for k in range(8)]
            print("  ridge |coef| mass by channel: " + " ".join(f"{n}={b:.3f}" for n, b in zip(names, blocks)))


if __name__ == "__main__":
    main()

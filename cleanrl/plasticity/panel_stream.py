"""Cross-sectional 5-minute panel: N liquid US stocks, one sample per (bar, stock).

Each stock's next-bar return (in its own trailing-vol units) is predicted from causal state:
its own lagged returns, the market's (SPY) lagged returns, the cross-sectional mean lagged
return, and its own log-volume surprise. One pooled linear model across stocks.

This file only builds the panel and measures the CEILINGS the way spy_ceiling did:
  * hindsight ridge fit on the first 60% of time, scored out of sample on the last 40%
  * streaming Adam, one step per bar on the cross-section (batch = N stocks), LR grid
  * permuted-target controls for both
Relative MSE vs the zero predictor; lower is better.

    .venv/bin/python cleanrl/plasticity/panel_stream.py --n-stocks 200
"""
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro

sys.path.insert(0, "/home/marvin/Documents/repositories/cleanrl")
from cleanrl.plasticity.stock_stream import read_bars  # noqa: E402

DATA = "/home/marvin/Documents/repositories/trading_bot_0/long_data"


@dataclass
class Args:
    rank_offset: int = 0
    """skip this many of the most liquid names (illiquid slices have microstructure signal)"""
    target: str = "ret"
    """ret = next-bar unit-vol return (reference: zero); vol = next-bar squared unit-vol return
    (reference: constant mean of the fit window)"""
    n_stocks: int = 200
    own_lags: int = 6
    mkt_lags: int = 6
    cs_lags: int = 3
    vol_span: int = 200
    min_coverage: float = 0.9
    """keep bars where at least this fraction of stocks trade"""
    first_session_before: str = "2018-01-01"
    lr_grid: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
    cache: str = "/tmp/panel_cache.npz"


def _ewma(x, span):
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    acc = np.zeros(x.shape[1:], dtype=np.float64)
    for t in range(x.shape[0]):
        acc = (1 - alpha) * acc + alpha * np.nan_to_num(x[t], nan=0.0)
        out[t] = acc
    return out


def build_panel(a):
    if os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        if int(z["n_stocks"]) == a.n_stocks and int(z["rank_offset"]) == a.rank_offset:
            return z["close"], z["volume"], list(z["symbols"]), z["ts"]
    entries = json.load(open(f"{DATA}/universe.json"))["entries"]
    stocks = [e for e in entries if e["type"] not in ("ETF",) and not e["delisted"]
              and os.path.exists(f"{DATA}/bars/{e['symbol']}.300.bars")]
    stocks.sort(key=lambda e: -e["median_dollar_volume"])
    symbols = [e["symbol"] for e in stocks[a.rank_offset:a.rank_offset + a.n_stocks]]
    spy = read_bars(f"{DATA}/bars/SPY.300.bars")
    ts = spy["t"]
    pos = {int(t): i for i, t in enumerate(ts)}
    close = np.full((len(ts), len(symbols) + 1), np.nan, dtype=np.float32)
    volume = np.full_like(close, np.nan)
    close[:, 0] = spy["close"]; volume[:, 0] = spy["volume"]
    t0 = time.time()
    for j, sym in enumerate(symbols, start=1):
        bars = read_bars(f"{DATA}/bars/{sym}.300.bars")
        idx = np.fromiter((pos.get(int(t), -1) for t in bars["t"]), dtype=np.int64, count=len(bars))
        ok = idx >= 0
        close[idx[ok], j] = bars["close"][ok]
        volume[idx[ok], j] = bars["volume"][ok]
        if j % 50 == 0:
            print(f"  read {j}/{len(symbols)} {time.time() - t0:.0f}s", file=sys.stderr)
    np.savez(a.cache, close=close, volume=volume, symbols=np.array(symbols), ts=ts, n_stocks=a.n_stocks, rank_offset=a.rank_offset)
    return close, volume, symbols, ts


def build_samples(a, close, volume):
    T, M = close.shape                      # column 0 = SPY
    cov = np.isfinite(close[:, 1:]).mean(1)
    keep = cov >= a.min_coverage
    close, volume = close[keep], volume[keep]
    T = close.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.diff(np.log(close), axis=0, prepend=np.nan)          # (T, M)
        ret[np.abs(ret) > 0.2] = np.nan
        scale = np.sqrt(_ewma(np.square(ret), a.vol_span))            # trailing vol per stock, uses <= t
        scale_lag = np.concatenate([np.full((1, M), np.nan), scale[:-1]], 0)
        z = ret / np.maximum(scale_lag, 1e-6)                          # unit-vol return at t
        lv = np.log1p(np.nan_to_num(volume, nan=0.0))
        vsurp = lv - _ewma(lv, a.vol_span)
    z = np.clip(z, -10, 10).astype(np.float32)
    N = M - 1
    L = max(a.own_lags, a.mkt_lags, a.cs_lags)
    cs_mean = np.nanmean(z[:, 1:], axis=1)                              # cross-sectional mean at t
    feats = []
    for k in range(1, a.own_lags + 1):
        feats.append(np.roll(z[:, 1:], k, axis=0))                     # own lag k
    for k in range(1, a.mkt_lags + 1):
        feats.append(np.repeat(np.roll(z[:, :1], k, axis=0), N, axis=1))
    for k in range(1, a.cs_lags + 1):
        feats.append(np.repeat(np.roll(cs_mean, k)[:, None], N, axis=1))
    feats.append(np.roll(vsurp[:, 1:], 1, axis=0))
    X = np.stack(feats, -1)                                             # (T, N, F)
    y = np.roll(z[:, 1:], -1, axis=0)                                   # next-bar unit-vol return
    if a.target == "vol":
        y = np.minimum(np.square(y), 25.0)                              # next-bar squared unit-vol return
    valid = np.isfinite(X).all(-1) & np.isfinite(y)
    valid[:L + 1] = False; valid[-1] = False
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0).astype(np.float32)
    return X, y, valid


def main():
    a = tyro.cli(Args)
    close, volume, symbols, ts = build_panel(a)
    X, y, valid = build_samples(a, close, volume)
    T, N, F = X.shape
    print(f"panel: {T} bars x {N} stocks x {F} features; valid samples {valid.sum()} ({valid.mean():.2%})")
    dev = torch.device("cuda")
    Xt = torch.tensor(X, device=dev); yt = torch.tensor(y, device=dev); vt = torch.tensor(valid, device=dev)
    Xt = torch.cat([Xt, torch.ones(T, N, 1, device=dev)], -1)
    F1 = F + 1
    cut = int(0.6 * T)
    if a.target == "vol":
        mu = yt[:cut][vt[:cut]].mean()                                   # reference = constant mean predictor
        yt = torch.where(vt, yt - mu, torch.zeros_like(yt))
        print(f"vol target: fit-window mean {mu.item():.4f}; scores are relative to predicting it")
    g = torch.Generator(device=dev).manual_seed(0)
    perm_t = torch.randperm(T, device=dev, generator=g)
    yp = yt[perm_t]                                                    # time-permuted target (keeps cross-section)
    vp = vt[perm_t]

    def flat(Xs, ys, vs):
        m = vs.reshape(-1)
        return Xs.reshape(-1, F1)[m].double(), ys.reshape(-1)[m].double()

    print("# hindsight ridge, fit first 60% of time, scored on last 40%")
    for name, yy, vv in (("REAL", yt, vt), ("PERM", yp, vp)):
        Xa, ya = flat(Xt[:cut], yy[:cut], vv[:cut])
        Xb, yb = flat(Xt[cut:], yy[cut:], vv[cut:])
        G = Xa.T @ Xa; c = Xa.T @ ya
        for lam in (0.0, 1e3, 1e4, 1e5):
            w = torch.linalg.solve(G + lam * torch.eye(F1, device=dev, dtype=torch.float64), c)
            ins = ((ya - Xa @ w).square().mean() / ya.square().mean()).item()
            oos = ((yb - Xb @ w).square().mean() / yb.square().mean()).item()
            print(f"  {name} lam={lam:7.0f}  in {ins:.5f}  out {oos:.5f}")
        if name == "REAL":
            w = torch.linalg.solve(G + 1e3 * torch.eye(F1, device=dev, dtype=torch.float64), c)
            names = [f"own{k}" for k in range(1, a.own_lags + 1)] + [f"mkt{k}" for k in range(1, a.mkt_lags + 1)] \
                + [f"cs{k}" for k in range(1, a.cs_lags + 1)] + ["vsurp", "bias"]
            se = torch.sqrt(torch.diag(torch.linalg.inv(G)) * (ya - Xa @ w).square().mean())
            print("  coefficients (t-stat):", " ".join(f"{n}={wi / si:+.1f}" for n, wi, si in zip(names, w.tolist(), se.tolist())))

    print("# streaming Adam, pooled linear model, one step per bar on the cross-section, scored after 60% of time")
    Gr = len(a.lr_grid)
    lr = torch.tensor(a.lr_grid, device=dev).view(Gr, 1)
    for name, yy, vv in (("REAL", yt, vt), ("PERM", yp, vp)):
        w = torch.zeros(Gr, F1, device=dev); m = torch.zeros_like(w); v = torch.zeros_like(w)
        se = torch.zeros(Gr, device=dev); ss = torch.zeros(Gr, device=dev)
        for t in range(T):
            mask = vv[t].float()
            if mask.sum() == 0:
                continue
            pred = w @ Xt[t].T                                          # (Gr, N)
            res = (pred - yy[t]) * mask
            if t >= cut:
                se += res.square().sum(1); ss += (yy[t] * mask).square().sum()
            grad = (res @ Xt[t]) / mask.sum()                           # (Gr, F1)
            m.mul_(0.9).add_(grad, alpha=0.1); v.mul_(0.999).addcmul_(grad, grad, value=0.001)
            step = (m / (1 - 0.9 ** (t + 1))) / ((v / (1 - 0.999 ** (t + 1))).sqrt() + 1e-8)
            w -= lr * step
        rel = (se / ss).tolist()
        print(f"  {name} " + " ".join(f"lr={l:.0e}:{r:.5f}" for l, r in zip(a.lr_grid, rel)))


if __name__ == "__main__":
    main()

"""MLP on the high-dimensional vol panel: hindsight ceiling vs streaming Adam.

offline : features of every `sub`-th fit-window bar are materialised (fits in GPU memory); AdamW
          minibatch epochs; early stop on the 50-60% slice; fixed net scored on the last 40%.
stream  : fresh net, one Adam step per bar on that bar's cross-section, scored on the last 40%
          while learning. LR grid. Permuted-target control for the best stream LR.
Relative MSE vs the constant-mean predictor.

    .venv/bin/python cleanrl/plasticity/panel_hd_mlp.py --lags 32 --width 256
"""
import sys
import time
from dataclasses import dataclass

import torch
import tyro

sys.path.insert(0, "/home/marvin/Documents/repositories/cleanrl")
from cleanrl.plasticity.panel_hd import Args as HDArgs, Bank  # noqa: E402


@dataclass
class Args(HDArgs):
    width: int = 256
    sub: int = 5
    epochs: int = 8
    offline_lr: float = 3e-4
    weight_decay: float = 1e-4
    stream_lrs: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    seed: int = 1


def mlp(f, width, dev):
    return torch.nn.Sequential(torch.nn.Linear(f, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, 1)).to(dev)


def score_fixed(net, bank, yy, vv):
    se = 0.0; ss = 0.0
    with torch.no_grad():
        for t in range(bank.cut, bank.T - 1):
            mk = vv[t]
            if mk.sum() == 0:
                continue
            x = bank.feats(t)[mk]; yb = yy[t][mk]
            se += (net(x).squeeze(-1) - yb).square().sum().item(); ss += yb.square().sum().item()
    return se / ss


def main():
    a = tyro.cli(Args)
    torch.manual_seed(a.seed)
    bank = Bank(a)
    T, L, F, cut, dev = bank.T, bank.L, bank.F, bank.cut, bank.dev
    va = int(0.5 * T)
    y, vm = bank.y, bank.valid
    print(f"panel {T} bars x {bank.N} stocks, F={F}, width {a.width}")
    t0 = time.time()

    # ---- offline: materialise subsampled fit / val windows
    def gather(lo, hi):
        xs, ys = [], []
        for t in range(max(lo, L + 1), hi, a.sub):
            mk = vm[t]
            if mk.sum() == 0:
                continue
            xs.append(bank.feats(t)[mk]); ys.append(y[t][mk])
        return torch.cat(xs), torch.cat(ys)
    Xa, ya = gather(0, va); Xv, yv = gather(va, cut)
    print(f"offline fit samples {Xa.shape[0]}, val {Xv.shape[0]} ({time.time() - t0:.0f}s)")
    net = mlp(F, a.width, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.offline_lr, weight_decay=a.weight_decay)
    best_v, best_state, best_ep = 9.0, None, -1
    for ep in range(a.epochs):
        perm = torch.randperm(Xa.shape[0], device=dev)
        for i in range(0, Xa.shape[0], 8192):
            idx = perm[i:i + 8192]
            loss = (net(Xa[idx]).squeeze(-1) - ya[idx]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            v = sum((net(Xv[i:i + 262144]).squeeze(-1) - yv[i:i + 262144]).square().sum().item()
                    for i in range(0, Xv.shape[0], 262144)) / yv.square().sum().item()
        print(f"  offline epoch {ep}: val {v:.5f}")
        if v < best_v:
            best_v, best_ep = v, ep
            best_state = {k: p.detach().clone() for k, p in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
        del Xa, ya, Xv, yv
        torch.cuda.empty_cache()
        print(f"offline MLP ceiling: val {best_v:.5f} at epoch {best_ep}; TEST {score_fixed(net, bank, y, vm):.5f} ({time.time() - t0:.0f}s)")

    # ---- streaming Adam
    results = {}
    for name, yy, vv in bank.streams():
        for lr in a.stream_lrs:
            if name == "PERM" and lr != min(results, key=results.get):
                continue
            torch.manual_seed(a.seed)
            net = mlp(F, a.width, dev)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            se = 0.0; ss = 0.0
            for t in range(L + 1, T - 1):
                mk = vv[t]
                if mk.sum() == 0:
                    continue
                x = bank.feats(t)[mk]; yb = yy[t][mk]
                res = net(x).squeeze(-1) - yb
                if t >= cut:
                    se += res.square().sum().item(); ss += yb.square().sum().item()
                loss = res.square().mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            rel = se / ss
            if name == "REAL":
                results[lr] = rel
            print(f"  stream {name} lr={lr:.0e}: {rel:.5f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

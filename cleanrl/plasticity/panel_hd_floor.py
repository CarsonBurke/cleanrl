"""Noise-floor proxy for the vol panel: in-sample fit of an MLP on the WHOLE period (test window included).

Whatever a streaming learner cannot reach in-sample with hindsight and many epochs is irreducible noise (plus
architecture limits). Relative MSE on the test-window bars that were trained on; lower bound on any honest score.

    .venv/bin/python cleanrl/plasticity/panel_hd_floor.py --width 256 --epochs 30 --sub 3
"""
import sys
import time
from dataclasses import dataclass

import torch
import tyro

sys.path.insert(0, "/home/marvin/Documents/repositories/cleanrl")
from cleanrl.plasticity.panel_hd import Args as HDArgs, Bank  # noqa: E402
from cleanrl.plasticity.panel_hd_gate import Net  # noqa: E402


@dataclass
class Args(HDArgs):
    width: int = 256
    epochs: int = 30
    sub: int = 3
    lr: float = 1e-3
    batch: int = 8192
    seed: int = 1


def main():
    a = tyro.cli(Args)
    bank = Bank(a)
    T, L, cut, dev = bank.T, bank.L, bank.cut, bank.dev
    (_, y, vm), = bank.streams()[:1]
    t0 = time.time()
    xs, ys, istest = [], [], []
    for t in range(L + 1, T - 1, a.sub):
        mk = vm[t]
        if not bool(mk.any()):
            continue
        xs.append(bank.feats(t)[mk]); ys.append(y[t][mk]); istest.append(torch.full((int(mk.sum()),), t >= cut, device=dev))
    X = torch.cat(xs); Y = torch.cat(ys); IT = torch.cat(istest)
    del xs, ys
    print(f"materialised {X.shape[0]} samples ({IT.sum().item()} in test window) in {time.time() - t0:.0f}s", flush=True)
    torch.manual_seed(a.seed)
    net = Net(bank.F, a.width).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    N = X.shape[0]
    for ep in range(a.epochs):
        perm = torch.randperm(N, device=dev)
        for s in range(0, N, a.batch):
            idx = perm[s:s + a.batch]
            pred, _ = net(X[idx])
            loss = (pred - Y[idx]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            se = 0.0; ss = 0.0
            for s in range(0, N, 65536):
                pred, _ = net(X[s:s + 65536])
                m = IT[s:s + 65536]
                se += ((pred - Y[s:s + 65536]).square() * m).sum().item(); ss += (Y[s:s + 65536].square() * m).sum().item()
        print(f"  epoch {ep}: in-sample test-window relative mse {se / ss:.5f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

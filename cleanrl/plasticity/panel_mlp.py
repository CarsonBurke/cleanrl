"""Nonlinear ceiling vs streaming Adam on the cross-sectional vol panel (panel_stream.py samples).

offline : MLP fit with AdamW on the first 50% of time (minibatch, epochs), early-stopped on 50-60%,
          scored on the last 40%  -> what a nonlinear model CAN extract with hindsight
stream  : the same MLP trained online, one Adam step per bar on that bar's cross-section
          (batch ~ number of stocks), scored on the last 40% while learning -> what streaming Adam gets
Both relative to the constant-mean predictor. Time-permuted target control for the streaming run.

    .venv/bin/python cleanrl/plasticity/panel_mlp.py --target vol
"""
import sys
from dataclasses import dataclass

import torch
import tyro

sys.path.insert(0, "/home/marvin/Documents/repositories/cleanrl")
from cleanrl.plasticity.panel_stream import Args as PanelArgs, build_panel, build_samples  # noqa: E402


@dataclass
class Args(PanelArgs):
    width: int = 128
    epochs: int = 6
    offline_lr: float = 3e-4
    weight_decay: float = 1e-4
    stream_lrs: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    seed: int = 1


def mlp(f, width, dev):
    return torch.nn.Sequential(torch.nn.Linear(f, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, 1)).to(dev)


def main():
    a = tyro.cli(Args)
    torch.manual_seed(a.seed)
    close, volume, _, _ = build_panel(a)
    X, y, valid = build_samples(a, close, volume)
    T, N, F = X.shape
    dev = torch.device("cuda")
    Xt = torch.tensor(X, device=dev); yt = torch.tensor(y, device=dev); vt = torch.tensor(valid, device=dev)
    cut, va = int(0.6 * T), int(0.5 * T)
    if a.target == "vol":
        mu = yt[:cut][vt[:cut]].mean()
        yt = torch.where(vt, yt - mu, torch.zeros_like(yt))
    ref = yt[cut:][vt[cut:]].square().mean().item()
    print(f"panel {T} bars x {N} stocks x {F} features; test reference mse {ref:.4f}")

    # ---- offline ceiling
    Xa, ya = Xt[:va][vt[:va]], yt[:va][vt[:va]]
    Xv, yv = Xt[va:cut][vt[va:cut]], yt[va:cut][vt[va:cut]]
    Xb, yb = Xt[cut:][vt[cut:]], yt[cut:][vt[cut:]]
    net = mlp(F, a.width, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.offline_lr, weight_decay=a.weight_decay)
    best = (9.0, 9.0, -1)
    for ep in range(a.epochs):
        perm = torch.randperm(Xa.shape[0], device=dev)
        for i in range(0, Xa.shape[0], 8192):
            idx = perm[i:i + 8192]
            loss = (net(Xa[idx]).squeeze(-1) - ya[idx]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            v = sum((net(Xv[i:i + 262144]).squeeze(-1) - yv[i:i + 262144]).square().sum().item()
                    for i in range(0, Xv.shape[0], 262144)) / yv.square().sum().item()
            te = sum((net(Xb[i:i + 262144]).squeeze(-1) - yb[i:i + 262144]).square().sum().item()
                     for i in range(0, Xb.shape[0], 262144)) / yb.square().sum().item()
        print(f"  offline epoch {ep}: val {v:.5f}  test {te:.5f}")
        if v < best[0]:
            best = (v, te, ep)
    print(f"offline MLP ceiling: val {best[0]:.5f} at epoch {best[2]}, test {best[1]:.5f}")

    # ---- streaming Adam, one step per bar on the cross-section
    g = torch.Generator(device=dev).manual_seed(0)
    perm_t = torch.randperm(T, device=dev, generator=g)
    for name, yy, vv in (("REAL", yt, vt), ("PERM", yt[perm_t], vt[perm_t])):
        for lr in a.stream_lrs:
            torch.manual_seed(a.seed)
            net = mlp(F, a.width, dev)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            se = 0.0; ss = 0.0
            for t in range(T):
                m = vv[t]
                if not bool(m.any()):
                    continue
                pred = net(Xt[t][m]).squeeze(-1)
                res = pred - yy[t][m]
                if t >= cut:
                    se += res.square().sum().item(); ss += yy[t][m].square().sum().item()
                loss = res.square().mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            print(f"  stream {name} lr={lr:.0e}: {se / ss:.5f}")


if __name__ == "__main__":
    main()

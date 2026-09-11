"""Event-driven evidence descent (EVT) on the high-dimensional vol panel.

Not a gate on Adam. Each parameter runs its own sequential test on its gradient stream:
    S1 += g,  S2 += g^2,  t^2 = S1^2 / S2          (chi^2_1 under a driftless stream, <= n by construction)
and does nothing until t^2 >= kappa. Then it applies ONE step from the accumulated evidence,
    w -= lr * S1 / sqrt(S2)                        (|step| = t >= sqrt(kappa): Adam-normalised magnitude)
and resets S1 = S2 = 0. Evidence is consumed by steps, never forgotten by a horizon: a parameter with a
consistent gradient fires every few samples and tracks at the full rate; a noise-driven parameter fires only
on false alarms (rate set by kappa) and is otherwise frozen; a regime change only has to overturn the
evidence accumulated since the last step. No EMA, no beta, no per-sample work, three buffers per parameter,
batch size irrelevant (one bar's cross-section here; batch 1 is the same code).

Arms: evt | evt_shuffle (fire mask permuted across the tensor's parameters: firing rate kept, correspondence
broken) | adam (torch.optim.Adam reference at the same lr grid).

    .venv/bin/python cleanrl/plasticity/panel_hd_evt.py --arm evt --kappa 6 --width 1024 --stream-lrs 1e-3 3e-3 1e-2
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
    width: int = 1024
    arm: str = "evt"
    """evt | evt_shuffle | adam"""
    kappa: float = 6.0
    stream_lrs: tuple[float, ...] = (1e-3, 3e-3, 1e-2)
    permuted: bool = False
    seed: int = 1


def run(a, bank, yy, vv, lr):
    T, L, F, cut, dev = bank.T, bank.L, bank.F, bank.cut, bank.dev
    torch.manual_seed(a.seed)
    net = Net(F, a.width).to(dev)
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=lr) if a.arm == "adam" else None
    s1 = [torch.zeros_like(p) for p in params]
    s2 = [torch.zeros_like(p) for p in params]
    nb = 8
    se = [0.0] * nb; ss = [0.0] * nb
    fired = torch.zeros((), device=dev); fired_n = 0
    for t in range(L + 1, T - 1):
        mk = vv[t]
        if not bool(mk.any()):
            continue
        x = bank.feats(t)[mk]; yb = yy[t][mk]
        pred, _ = net(x)
        res = pred - yb
        if t >= cut:
            b = min((t - cut) * nb // (T - cut), nb - 1)
            se[b] += res.square().sum().item(); ss[b] += yb.square().sum().item()
        loss = res.square().mean()
        if opt is not None:
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            continue
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for p, g, a1, a2 in zip(params, grads, s1, s2):
                a1.add_(g); a2.addcmul_(g, g)
                t2 = a1.square() / a2.clamp_min(1e-30)
                fire = t2 >= a.kappa
                if a.arm == "evt_shuffle":
                    fire = fire.flatten()[torch.randperm(fire.numel(), device=dev)].view_as(fire)
                step = torch.where(fire, a1 / a2.clamp_min(1e-30).sqrt(), torch.zeros_like(a1))
                p.sub_(lr * step)
                a1.masked_fill_(fire, 0.0); a2.masked_fill_(fire, 0.0)
                fired += fire.sum(); fired_n += fire.numel()
    return sum(se) / sum(ss), (fired / max(fired_n, 1)).item(), [e / s for e, s in zip(se, ss)]


def main():
    a = tyro.cli(Args)
    bank = Bank(a)
    print(f"panel {bank.T} bars x {bank.N} stocks, F={bank.F}, width {a.width}, arm {a.arm}, kappa {a.kappa}")
    t0 = time.time()
    for name, yy, vv in bank.streams():
        if name == "PERM" and not a.permuted:
            continue
        for lr in a.stream_lrs:
            rel, fr, blocks = run(a, bank, yy, vv, lr)
            print(f"  {name} arm={a.arm} k={a.kappa:g} width={a.width} lr={lr:.0e}: {rel:.5f}   fire rate {fr:.4f}  ({time.time() - t0:.0f}s)")
            print("    blocks " + " ".join(f"{b:.4f}" for b in blocks))


if __name__ == "__main__":
    main()

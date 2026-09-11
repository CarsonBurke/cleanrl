"""Per-PARAMETER evidence-gated Adam on the high-dimensional vol panel (streaming MLP).

Baseline `none`: Adam (manual, identical to torch.optim.Adam), one step per bar on the cross-section.
Gate `js`: every parameter keeps plain sums of its per-bar gradient, S1 = sum g, S2 = sum g^2.
    Under pure noise E[S1^2] = S2; with a consistent mean E[S1^2] = n^2 mu^2 + S2. The positive-part
    James-Stein factor p = (1 - S2 / S1^2)^+ is 0 for a parameter whose gradient history is noise and
    -> 1 for one with consistent evidence. It multiplies the Adam step per parameter (post-optimizer, so
    1/sqrt(v) cannot renormalise it away). No EMA, no twin, no rate.
    `--discount d` (default 0 = plain sums) multiplies the sums by (1 - d) per bar.
Gate `js_shuffle`: p permuted across the parameters of each tensor (level kept, correspondence broken).

    .venv/bin/python cleanrl/plasticity/panel_hd_pp.py --gate js --width 256 --stream-lrs 1e-4 3e-4 1e-3
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
    gate: str = "js"
    """none | js | js_shuffle"""
    discount: float = 0.0
    """per-bar forgetting of the evidence sums: S1 *= (1-d), S2 *= (1-d)^2 keeps E[S1^2] = S2 under noise"""
    kappa: float = 1.0
    """gate = (1 - kappa * S2 / S1^2)^+ ; kappa=1 is James-Stein, larger = stricter null (t^2 must exceed kappa)"""
    cap: float = 0.0
    """>0: gate = min(t^2 / kappa, cap) with t^2 = S1^2/S2 -- linear in evidence, exceeds 1 (amplifies) above kappa"""
    step: str = "adam"
    """adam: m/sqrt(v) (beta 0.9/0.999) | cum: S1/sqrt(S2*N) -- the cumulative mean gradient over the cumulative
    rms (Adam-scale drift mu/sigma for a consistent parameter, ~1/sqrt(N) for a noise-driven one, no per-step noise)"""
    fast_decay: float = 0.0
    """>0: gate 'consol' -- w = w_slow + w_fast; the ungated Adam step lands in w_fast, which leaks toward 0 at this
    rate; each step a fraction gate_js of w_fast is consolidated into w_slow. Evidence-gated weight decay toward the
    consolidated value: noise-driven parameters carry bounded variance lr^2/(2 decay) instead of a random walk."""
    stream_lrs: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    permuted: bool = False
    seed: int = 1


def run(a, bank, yy, vv, lr):
    T, L, F, cut, dev = bank.T, bank.L, bank.F, bank.cut, bank.dev
    torch.manual_seed(a.seed)
    net = Net(F, a.width).to(dev)
    params = list(net.parameters())
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    s1 = [torch.zeros_like(p) for p in params]
    s2 = [torch.zeros_like(p) for p in params]
    ws = [p.detach().clone() for p in params] if a.fast_decay > 0 else None
    wf = [torch.zeros_like(p) for p in params] if a.fast_decay > 0 else None
    b1, b2, eps = 0.9, 0.999, 1e-8
    nb = 8
    se = [0.0] * nb; ss = [0.0] * nb
    lvl = torch.zeros((), device=dev); lvl_n = 0; k = 0
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
        grads = torch.autograd.grad(loss, params)
        k += 1
        with torch.no_grad():
            for i, (p, g, mm, vv_, a1, a2) in enumerate(zip(params, grads, m, v, s1, s2)):
                if a.step == "adam":
                    mm.mul_(b1).add_(g, alpha=1 - b1)
                    vv_.mul_(b2).addcmul_(g, g, value=1 - b2)
                if a.step == "cum":
                    step = (a1 + g) / ((a2 + g * g).clamp_min(1e-30) * k).sqrt()
                elif a.step == "snr":
                    # per-parameter smoothing horizon h = kappa N / t^2 so the smoothed gradient has SNR ~ sqrt(kappa):
                    # strong evidence -> short horizon (tracks), none -> cumulative mean (freezes as 1/sqrt(N))
                    t2p = (a1 + g).square() / (a2 + g * g).clamp_min(1e-30)
                    h = (a.kappa * k / t2p.clamp_min(1e-30)).clamp(2.0, float(k))
                    if a.gate == "hshuffle":
                        h = h.flatten()[torch.randperm(h.numel(), device=dev)].view_as(h)
                    mm.add_((g - mm) / h)
                    step = mm / ((a2 + g * g) / k).sqrt().clamp_min(1e-12)
                    lvl += (1.0 / h).sum(); lvl_n += h.numel()
                else:
                    step = (mm / (1 - b1 ** k)) / ((vv_ / (1 - b2 ** k)).sqrt() + eps)
                if a.fast_decay > 0:
                    t2 = a1.square() / a2.clamp_min(1e-30)
                    gate = (1.0 - a.kappa / t2.clamp_min(1e-30)).clamp_min(0.0)
                    if a.gate == "consol_shuffle":
                        gate = gate.flatten()[torch.randperm(gate.numel(), device=dev)].view_as(gate)
                    wf[i].mul_(1 - a.fast_decay).sub_(lr * step)
                    tau = gate * wf[i]
                    ws[i].add_(tau); wf[i].sub_(tau)
                    lvl += gate.sum(); lvl_n += gate.numel()
                    a1.add_(g); a2.addcmul_(g, g)
                    p.copy_(ws[i] + wf[i])
                    continue
                if a.gate not in ("none", "hshuffle"):
                    t2 = a1.square() / a2.clamp_min(1e-30)                 # read before adding this bar
                    gate = (t2 / a.kappa).clamp_max(a.cap) if a.cap > 0 else (1.0 - a.kappa / t2.clamp_min(1e-30)).clamp_min(0.0)
                    if a.gate == "js_shuffle":
                        gate = gate.flatten()[torch.randperm(gate.numel(), device=dev)].view_as(gate)
                    step = step * gate
                    lvl += gate.sum(); lvl_n += gate.numel()
                if a.discount > 0:
                    a1.mul_(1 - a.discount); a2.mul_((1 - a.discount) ** 2)
                a1.add_(g); a2.addcmul_(g, g)
                p.sub_(lr * step)
    return sum(se) / sum(ss), (lvl / max(lvl_n, 1)).item(), [e / s for e, s in zip(se, ss)]


def main():
    a = tyro.cli(Args)
    bank = Bank(a)
    print(f"panel {bank.T} bars x {bank.N} stocks, F={bank.F}, width {a.width}, gate {a.gate}, discount {a.discount}")
    t0 = time.time()
    for name, yy, vv in bank.streams():
        if name == "PERM" and not a.permuted:
            continue
        for lr in a.stream_lrs:
            rel, lm, blocks = run(a, bank, yy, vv, lr)
            print(f"  {name} gate={a.gate} width={a.width} lr={lr:.0e}: {rel:.5f}   gate level {lm:.3f}  ({time.time() - t0:.0f}s)")
            print("    blocks " + " ".join(f"{b:.4f}" for b in blocks))


if __name__ == "__main__":
    main()

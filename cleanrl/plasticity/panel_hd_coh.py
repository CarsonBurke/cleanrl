"""Per-(sample, parameter) coherence gates at optimizer-state cost, on the high-dimensional vol panel.

A dense layer's per-sample gradient is rank-1: g[t,i,j] = delta[t,i] * x[t,j]. Any per-(sample,parameter)
statistic that is a sum over samples of a product of a per-(sample,unit) and a per-(sample,input) factor is
ONE extra matmul, never an (n x params) tensor. The per-sample gradient MASS is such a statistic:
    L1[i,j] = sum_t |delta[t,i]| |x[t,j]|      (biases: sum_t |delta[t,i]|)
Sign-agreement gating of individual samples reduces to exactly (G, L1) -- 1[agree] = (1 + s*sgn(g_t))/2 gives
sum_t 1[agree] g_t = (G + s*L1)/2 -- so the per-sample information is the coherence |G| / L1 in [0, 1]:
the fraction of the gradient mass that points the same way. Under independent zero-mean per-sample gradients
(2/pi) N (S1/L1)^2 ~ chi^2_1 (Gaussian; heavy tails make it conservative), N = samples seen.

Arms (gate multiplies the Adam step post-optimizer; two buffers per parameter, no EMA, no tuned scale):
  none  : Adam
  js    : temporal evidence on batch gradients, t^2 = S1^2 / S2, gate (1 - kappa/t^2)^+     (panel_hd_pp)
  l1    : temporal evidence on per-sample mass, t^2 = (2/pi) N S1^2 / L1^2, same gate shape
  coh   : js gate x within-batch coherence gate on THIS bar's samples, c = |G| / L1_bar,
          t_c^2 = (2/pi) n c^2, gate (1 - kappa_c / t_c^2)^+   (state-dependent per step, no history)
  *_shuffle : gate permuted across the parameters of each tensor (level kept, correspondence broken)

    .venv/bin/python cleanrl/plasticity/panel_hd_coh.py --gate l1 --width 1024 --stream-lrs 1e-3 3e-3
"""
import math
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
    gate: str = "l1"
    """none | js | l1 | coh | js_shuffle | l1_shuffle | coh_shuffle"""
    kappa: float = 3.0
    kappa_c: float = 3.0
    stream_lrs: tuple[float, ...] = (1e-3, 3e-3)
    permuted: bool = False
    seed: int = 1
    clip: str = "none"
    """none | unit | sample -- winsorize the backprop error delta[t,i] at kappa_u * rms (per unit: own running
    rms over samples seen; per sample: the output residual only, i.e. a Huber-type loss control)"""
    kappa_u: float = 3.0


def grads_and_mass(net, x, h1, h2, res, clip=None):
    """Manual backward of mean(res^2) through Linear-tanh-Linear-tanh-Linear: batch gradients and
    per-sample gradient L1 masses, in parameter order (W1, b1, W2, b2, W3, b3).
    clip = (mode, kappa_u, [q3, q2, q1] running sums of delta^2 per unit, N) winsorizes delta per layer."""
    n = x.shape[0]
    d3 = (2.0 / n) * res.unsqueeze(1)                       # (n, 1)
    if clip is not None:
        mode, ku, q, N = clip
        def wins(d, qi):
            qi.add_(d.square().sum(0))
            if N > 0:
                tau = ku * (qi / (N + d.shape[0])).sqrt()
                d = d * (tau / d.abs().clamp_min(1e-30)).clamp_max(1.0)
            return d
        d3 = wins(d3, q[0])
    d2 = (d3 * net.l3.weight) * (1.0 - h2.square())         # (n, W)
    if clip is not None and mode == "unit":
        d2 = wins(d2, q[1])
    d1 = (d2 @ net.l2.weight) * (1.0 - h1.square())         # (n, W)
    if clip is not None and mode == "unit":
        d1 = wins(d1, q[2])
    a1, a2, a3 = d1.abs(), d2.abs(), d3.abs()
    grads = (d1.T @ x, d1.sum(0), d2.T @ h1, d2.sum(0), d3.T @ h2, d3.sum(0))
    mass = (a1.T @ x.abs(), a1.sum(0), a2.T @ h1.abs(), a2.sum(0), a3.T @ h2.abs(), a3.sum(0))
    return grads, mass


def run(a, bank, yy, vv, lr):
    T, L, F, cut, dev = bank.T, bank.L, bank.F, bank.cut, bank.dev
    torch.manual_seed(a.seed)
    net = Net(F, a.width).to(dev)
    params = [net.l1.weight, net.l1.bias, net.l2.weight, net.l2.bias, net.l3.weight, net.l3.bias]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    s1 = [torch.zeros_like(p) for p in params]
    s2 = [torch.zeros_like(p) for p in params]          # js: sum G^2 ; l1: sum per-sample |g|
    b1, b2, eps = 0.9, 0.999, 1e-8
    base = a.gate.replace("_shuffle", "")
    shuffle = a.gate.endswith("_shuffle")
    nb = 8
    se = [0.0] * nb; ss = [0.0] * nb
    lvl = torch.zeros((), device=dev); lvl_n = 0; k = 0; N = 0
    checked = False
    q = [torch.zeros(1, device=dev), torch.zeros(a.width, device=dev), torch.zeros(a.width, device=dev)]
    with torch.no_grad():
        for t in range(L + 1, T - 1):
            mk = vv[t]
            if not bool(mk.any()):
                continue
            x = bank.feats(t)[mk]; yb = yy[t][mk]
            n = x.shape[0]
            h1 = torch.tanh(net.l1(x)); h2 = torch.tanh(net.l2(h1))
            res = net.l3(h2).squeeze(-1) - yb
            if t >= cut:
                b = min((t - cut) * nb // (T - cut), nb - 1)
                se[b] += res.square().sum().item(); ss[b] += yb.square().sum().item()
            grads, mass = grads_and_mass(net, x, h1, h2, res)
            if not checked:   # one-time contract: manual backward == autograd
                with torch.enable_grad():
                    pred, _ = net(x)
                    ag = torch.autograd.grad((pred - yb).square().mean(), params)
                for g, r in zip(grads, ag):
                    assert torch.allclose(g, r, rtol=1e-3, atol=1e-6), (g - r).abs().max()
                checked = True
            if a.clip != "none":
                grads, mass = grads_and_mass(net, x, h1, h2, res, (a.clip, a.kappa_u, q, N))
            k += 1; N += n
            for p, g, ms, mm, vv_, a1, a2 in zip(params, grads, mass, m, v, s1, s2):
                mm.mul_(b1).add_(g, alpha=1 - b1)
                vv_.mul_(b2).addcmul_(g, g, value=1 - b2)
                step = (mm / (1 - b1 ** k)) / ((vv_ / (1 - b2 ** k)).sqrt() + eps)
                if base != "none":
                    if base == "l1":
                        t2 = (2.0 / math.pi) * N * a1.square() / a2.square().clamp_min(1e-30)
                    else:
                        t2 = a1.square() / a2.clamp_min(1e-30)
                    gate = (1.0 - a.kappa / t2.clamp_min(1e-30)).clamp_min(0.0)
                    if base == "coh":
                        tc2 = (2.0 / math.pi) * n * g.square() / ms.square().clamp_min(1e-30)
                        gate = gate * (1.0 - a.kappa_c / tc2.clamp_min(1e-30)).clamp_min(0.0)
                    if shuffle:
                        gate = gate.flatten()[torch.randperm(gate.numel(), device=dev)].view_as(gate)
                    step = step * gate
                    lvl += gate.sum(); lvl_n += gate.numel()
                    a1.add_(g)
                    if base == "l1":
                        a2.add_(ms)
                    else:
                        a2.addcmul_(g, g)
                p.sub_(lr * step)
    return sum(se) / sum(ss), (lvl / max(lvl_n, 1)).item(), [e / s for e, s in zip(se, ss)]


def main():
    a = tyro.cli(Args)
    bank = Bank(a)
    print(f"panel {bank.T} bars x {bank.N} stocks, F={bank.F}, width {a.width}, gate {a.gate}, kappa {a.kappa}, kappa_c {a.kappa_c}")
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

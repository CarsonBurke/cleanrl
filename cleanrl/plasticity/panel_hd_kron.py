"""Kronecker-factored cross-neuron posterior (kron_bayes_stream_v1) on the high-dimensional vol panel.

Same task, scoring, init, prior and R as panel_hd_ekf; the P x (r+n) posterior buffer is replaced by per-layer
factors Ahat_l (in+1)^2 (input second moments) and Ghat_l out^2 (sensitivity second moments / R), so a bar costs
O(n (in^2 + out^2)) per layer instead of O(n P (r+n)). One batched Kalman update per bar under the factored
posterior with factored Tikhonov damping (see kron_bayes_stream_v1):
    P J^T columns  (G_d^-1 s_i) (x) (A_d^-1 a_i),   J P J^T = sum_l (S G_d^-1 S^T) o (A_in A_d^-1 A_in^T),
    K = P J^T (R I + J P J^T)^-1,  theta <- theta - K res,   Ahat += A_in^T A_in,  Ghat += S^T S / R,  T += n.
`forget` decays the factors and T per bar (process noise). Ablation `--diag`: factor diagonals only.

    .venv/bin/python cleanrl/plasticity/panel_hd_kron.py --width 256 --prior 0.1 --forget 0
"""
import math
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
    prior: float = 0.1
    forget: float = 0.0
    noise_rate: float = 0.001
    diag: bool = False
    permuted: bool = False
    seed: int = 1


def solve(F, V, damp, diag):
    """V (F + damp I)^-1 for rows of V; diag keeps only F's diagonal."""
    if diag:
        return V / (torch.diagonal(F) + damp)
    return torch.linalg.solve(F + damp * torch.eye(F.shape[0], device=F.device), V.T).T


def run(a, bank, yy, vv):
    T, L, F, cut, dev = bank.T, bank.L, bank.F, bank.cut, bank.dev
    W = a.width
    g = torch.Generator(device=dev).manual_seed(a.seed)

    def layer(o, i, gain):
        q, _ = torch.linalg.qr(torch.randn(max(o, i), min(o, i), generator=g, device=dev))
        w = torch.zeros(o, i + 1, device=dev)
        w[:, :-1] = (q if o >= i else q.T) * gain
        return w
    weights = [layer(W, F, math.sqrt(2)), layer(W, W, math.sqrt(2)), layer(1, W, 1.0)]
    W1, W2, W3 = weights
    lam0 = [w.shape[1] / a.prior for w in weights]
    A = [torch.zeros(w.shape[1], w.shape[1], device=dev) for w in weights]
    G = [torch.zeros(w.shape[0], w.shape[0], device=dev) for w in weights]
    Tn = 0.0
    R = torch.zeros((), device=dev); R_init = False
    nb = 8
    se = [0.0] * nb; ss = [0.0] * nb; sev = 0.0; ssv = 0.0
    val_lo = int(0.4 * T)
    t0 = time.time()
    with torch.no_grad():
        for t in range(L + 1, T - 1):
            mk = vv[t]
            if not bool(mk.any()):
                continue
            x = bank.feats(t)[mk]; yb = yy[t][mk]
            n = x.shape[0]
            h1 = torch.tanh(x @ W1[:, :-1].T + W1[:, -1])
            h2 = torch.tanh(h1 @ W2[:, :-1].T + W2[:, -1])
            pred = h2 @ W3[0, :-1] + W3[0, -1]
            res = pred - yb
            if t >= cut:
                b = min((t - cut) * nb // (T - cut), nb - 1)
                se[b] += res.square().sum().item(); ss[b] += yb.square().sum().item()
            elif t >= val_lo:
                sev += res.square().sum().item(); ssv += yb.square().sum().item()
            d2 = W3[0, :-1] * (1.0 - h2.square())                  # (n, W)
            d1 = (d2 @ W2[:, :-1]) * (1.0 - h1.square())            # (n, W)
            ones = torch.ones(n, 1, device=dev)
            inputs = [torch.cat([x, ones], 1), torch.cat([h1, ones], 1), torch.cat([h2, ones], 1)]
            sens = [d1, d2, ones]
            if not R_init:
                R = yb.square().mean(); R_init = True
            if a.forget > 0:
                for Al, Gl in zip(A, G):
                    Al.mul_(1.0 - a.forget); Gl.mul_(1.0 - a.forget)
                Tn *= 1.0 - a.forget
            Tc = max(Tn, 1.0)
            C = R * torch.eye(n, device=dev)
            dirs = []
            for Al, Gl, l0, ai, si in zip(A, G, lam0, inputs, sens):
                Gn = Gl / Tc
                tr_g = torch.diagonal(Gn).mean(); tr_a = torch.diagonal(Al).mean()
                pi = ((tr_g + 1e-12) / (tr_a + 1e-12)).sqrt()
                rl = math.sqrt(l0)
                Ws = solve(Gn, si, pi * rl, a.diag)                  # (n, out)  = S G_d^-1
                Va = solve(Al, ai, rl / pi, a.diag)                  # (n, in+1) = A_in A_d^-1
                C = C + (si @ Ws.T) * (ai @ Va.T)
                dirs.append((Ws, Va))
            alpha = torch.linalg.solve(C, res)                       # (n,)
            for w, (Ws, Va) in zip(weights, dirs):
                w.sub_(torch.einsum("no,n,ni->oi", Ws, alpha, Va))
            for Al, Gl, ai, si in zip(A, G, inputs, sens):
                Al.add_(ai.T @ ai); Gl.add_((si.T @ si) / R)
            Tn += n
            R = R + a.noise_rate * (res.square().mean() - R)
            if t % 20000 == 0:
                print(f"    bar {t} ({time.time() - t0:.0f}s) R {R.item():.3f}", flush=True)
    return sum(se) / sum(ss), sev / max(ssv, 1e-12), [e / s for e, s in zip(se, ss)]


def main():
    a = tyro.cli(Args)
    bank = Bank(a)
    print(f"panel {bank.T} bars x {bank.N} stocks, F={bank.F}, width {a.width}, prior {a.prior}, forget {a.forget}, diag {a.diag}", flush=True)
    t0 = time.time()
    for name, yy, vv in bank.streams():
        if name == "PERM" and not a.permuted:
            continue
        rel, val, blocks = run(a, bank, yy, vv)
        print(f"  {name} kron width={a.width} prior={a.prior:g} forget={a.forget:g} diag={a.diag}: TEST {rel:.5f}  VAL(40-60%) {val:.5f}  ({time.time() - t0:.0f}s)")
        print("    blocks " + " ".join(f"{b:.4f}" for b in blocks))


if __name__ == "__main__":
    main()

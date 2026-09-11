"""Low-rank cross-neuron EKF (lowrank_bayes_stream_v1) on the high-dimensional vol panel.

No learning rate, no Adam. The dense MLP F-W-W-1 carries a posterior precision Lambda = diag(D) + M M^T over ALL
its parameters (M: P x (r + n) -- r retained directions plus this bar's n observations). Each bar's cross-section
is one batched Kalman update with the exact per-sample output Jacobian J (n x P):
    P J^T = D^-1 J^T - D^-1 M (I + M^T D^-1 M)^-1 M^T D^-1 J^T           (Woodbury, O(n P (r+n)))
    K = P J^T (R I + J P J^T)^-1,   theta <- theta - K res,   Lambda <- Lambda + J^T J / R
then [M | J^T/sqrt(R)] is truncated back to rank r by its Gram eigendecomposition, dropped energy folded into D.
Forgetting `forget` scales Lambda by (1 - forget) per bar. R is a causal EMA of the squared residual.
Every parameter's step on every sample is set by the sample's Jacobian through the joint posterior: per-parameter,
state-dependent, cross-neuron, streamable (batch 1 is the same code with n = 1).

Scoring as panel_hd_pp: relative MSE vs constant predictor on the last 40% of time, in 8 blocks; the 40-60% window
is reported separately for configuration selection.

    .venv/bin/python cleanrl/plasticity/panel_hd_ekf.py --width 256 --rank 128 --prior 0.1 --forget 0
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
    rank: int = 128
    prior: float = 0.1
    """prior weight variance = prior / fan_in (bias-augmented), as network_bayes_stream_v2"""
    forget: float = 0.0
    noise_rate: float = 0.001
    permuted: bool = False
    seed: int = 1


def run(a, bank, yy, vv):
    T, L, F, cut, dev = bank.T, bank.L, bank.F, bank.cut, bank.dev
    W = a.width
    g = torch.Generator(device=dev).manual_seed(a.seed)
    # bias-augmented layers, orthogonal init as Net (nn.Linear default would be uniform; keep it simple: orthogonal)
    def layer(o, i, gain):
        q, _ = torch.linalg.qr(torch.randn(max(o, i), min(o, i), generator=g, device=dev))
        w = torch.zeros(o, i + 1, device=dev)
        w[:, :-1] = (q if o >= i else q.T) * gain
        return w
    W1, W2, W3 = layer(W, F, math.sqrt(2)), layer(W, W, math.sqrt(2)), layer(1, W, 1.0)
    weights = [W1, W2, W3]
    sizes = [w.numel() for w in weights]
    P = sum(sizes)
    fan = torch.cat([torch.full((w.numel(),), float(w.shape[1]), device=dev) for w in weights])
    D = fan / a.prior                                    # diagonal precision
    r = a.rank
    nmax = int(vv.sum(1).max())
    M = torch.zeros(P, r + nmax, device=dev)
    S = torch.zeros(r + nmax, r + nmax, device=dev)      # M^T D^-1 M over live columns
    c = 0
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
            # exact output Jacobian per sample
            d2 = W3[0, :-1] * (1.0 - h2.square())                  # (n, W)
            d1 = (d2 @ W2[:, :-1]) * (1.0 - h1.square())            # (n, W)
            ones = torch.ones(n, 1, device=dev)
            J = torch.cat([(d1.unsqueeze(-1) * torch.cat([x, ones], 1).unsqueeze(1)).flatten(1),
                           (d2.unsqueeze(-1) * torch.cat([h1, ones], 1).unsqueeze(1)).flatten(1),
                           torch.cat([h2, ones], 1)], 1)             # (n, P)
            if not R_init:
                R = yb.square().mean(); R_init = True
            if a.forget > 0:
                D.mul_(1.0 - a.forget); M[:, :c].mul_(math.sqrt(1.0 - a.forget))
            Y = J / D                                                # (n, P) = J D^-1
            if c > 0:
                Mc = M[:, :c]
                YM = Y @ Mc                                          # (n, c)
                Z = torch.linalg.solve(S[:c, :c] + torch.eye(c, device=dev), YM.T)   # (c, n)
                PJt = Y.T - (Mc / D.unsqueeze(-1)) @ Z              # (P, n)
            else:
                PJt = Y.T
            C = J @ PJt + R * torch.eye(n, device=dev)               # (n, n)
            K = torch.linalg.solve(C, PJt.T).T                       # (P, n)
            delta = K @ res                                          # (P,)
            s = 0
            for w, sz in zip(weights, sizes):
                w.sub_(delta[s:s + sz].view_as(w)); s += sz
            # precision update: append n columns J^T / sqrt(R)
            B = J.T / R.sqrt()
            M[:, c:c + n] = B
            Bu = B / D.unsqueeze(-1)
            if c > 0:
                cross = M[:, :c].T @ Bu                              # (c, n)
                S[:c, c:c + n] = cross; S[c:c + n, :c] = cross.T
            S[c:c + n, c:c + n] = B.T @ Bu
            c += n
            if c > r:   # truncate to rank r
                Mc = M[:, :c]
                G = Mc.T @ Mc
                _, V = torch.linalg.eigh(G)
                keep, drop = V[:, -r:], V[:, :-r]
                Wr = Mc @ keep
                D.add_((Mc @ drop).square().sum(-1))
                M[:, :r] = Wr; M[:, r:c].zero_()
                S.zero_(); S[:r, :r] = Wr.T @ (Wr / D.unsqueeze(-1))
                c = r
            R = R + a.noise_rate * (res.square().mean() - R)
            if t % 20000 == 0:
                print(f"    bar {t} ({time.time() - t0:.0f}s) R {R.item():.3f}", flush=True)
    return sum(se) / sum(ss), sev / max(ssv, 1e-12), [e / s for e, s in zip(se, ss)]


def main():
    a = tyro.cli(Args)
    bank = Bank(a)
    print(f"panel {bank.T} bars x {bank.N} stocks, F={bank.F}, width {a.width}, rank {a.rank}, prior {a.prior}, forget {a.forget}", flush=True)
    t0 = time.time()
    for name, yy, vv in bank.streams():
        if name == "PERM" and not a.permuted:
            continue
        rel, val, blocks = run(a, bank, yy, vv)
        print(f"  {name} ekf width={a.width} rank={a.rank} prior={a.prior:g} forget={a.forget:g}: TEST {rel:.5f}  VAL(40-60%) {val:.5f}  ({time.time() - t0:.0f}s)")
        print("    blocks " + " ".join(f"{b:.4f}" for b in blocks))


if __name__ == "__main__":
    main()

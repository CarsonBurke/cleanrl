"""Low-rank cross-neuron Bayesian plasticity, v1: how much of the full-covariance gain survives rank r?

Family fact (network_bayes_stream_v2, 17-64-64-1, 65,536-sample batch-1 stream, paired seed 1): the full
5,377x5,377 EKF posterior beats Adam by 30% sustained clean error (0.121 vs 0.174); the neuron-block posterior
by 4%; erasing the update direction while keeping the conditioning is worse than Adam (0.201). The payload is
cross-neuron correlation, and full covariance is O(P^2). This file measures the rank curve.

Precision form: Lambda = diag(D) + M M^T with M = [W (rank r) | B (buffer of the last <T observations)].
Observation j (exact output Jacobian, bias-augmented) with innovation variance R adds column j/sqrt(R) to B:
Lambda <- Lambda + j j^T / R exactly. The gain uses the PRIOR posterior via Woodbury,
    P j = D^-1 j - D^-1 M (I + M^T D^-1 M)^-1 M^T D^-1 j,    theta <- theta - P j * res / (R + j^T P j),
with S = M^T D^-1 M maintained incrementally (one column per step). When B fills, [W | B] is truncated to rank r
by the eigendecomposition of its Gram matrix; the dropped directions' energy is folded into D as the diagonal of
the dropped rank-one terms (never discarded). Forgetting `forget` scales Lambda by (1 - forget) per sample
(multiplicative process noise; leaves S invariant). rank 0 is the diagonal Kalman learner.

Cost per sample O(P (r + T)) plus O(P (r+T)^2 / T) amortised truncation; memory O(P (r + T)) per config.
Same data, initialisation, validation selection and clean-error reporting as network_bayes_stream_v2 (adam and
full-covariance arms are imported from it).

    .venv/bin/python cleanrl/plasticity/lowrank_bayes_stream_v1.py --ranks 0 16 64 256 --methods adam network lowrank
"""
import math
import time
from dataclasses import dataclass

import torch
import tyro

from cleanrl.plasticity.network_bayes_stream_v2 import (Learner, draw_teacher, init_weights, sample_state,
                                                       teach)
from cleanrl.shared import runtime


@dataclass
class Args:
    seed: int = 1
    samples: int = 65536
    hidden: int = 64
    input_dim: int = 17
    noise: float = 1.0
    hetero: float = 0.0
    switch_at: float = 0.0
    methods: tuple[str, ...] = ("adam", "network", "lowrank")
    ranks: tuple[int, ...] = (0, 16, 64, 256)
    buffer: int = 64
    adam_lrs: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    prior_scales: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    forget: float = 0.0
    diffusion: float = 1e-5
    noise_rate: float = 0.001
    known_noise: bool = False
    validation: int = 2048
    test: int = 8192
    graph_steps: int = 16
    output: str = ""


class LowRank:
    def __init__(self, rank, grid, initial, a, xs, ys, clean, noise_var):
        self.a, self.rank, self.T = a, rank, a.buffer
        k, dev = len(grid), xs.device
        self.k, self.dev = k, dev
        self.weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
        self.scale = torch.tensor(grid, device=dev)
        self.P = sum(w[0].numel() for w in self.weights)
        prior_var = torch.cat([(self.scale / w.shape[-1])[:, None].expand(k, w[0].numel()) for w in self.weights], -1)
        self.D = 1.0 / prior_var                                   # (k, P) diagonal precision
        self.M = torch.zeros(k, self.P, rank + self.T, device=dev)  # [W | B]
        self.S = torch.zeros(k, rank + self.T, rank + self.T, device=dev)  # M^T D^-1 M
        self.c = 0                                                 # live columns
        self.noise = torch.ones(k, device=dev)
        self.error = torch.zeros(k, device=dev); self.null = torch.zeros((), device=dev)
        self.xs, self.ys, self.clean, self.noise_var = xs, ys, clean, noise_var
        self.steps = 0

    @torch.no_grad()
    def truncate(self):
        r, c = self.rank, self.c
        M = self.M[:, :, :c]
        if r == 0:
            self.D.add_(M.square().sum(-1))
        else:
            G = M.transpose(-1, -2) @ M                              # (k, c, c)
            evals, V = torch.linalg.eigh(G)                         # ascending
            keep, drop = V[:, :, -r:], V[:, :, :-r]
            W = M @ keep                                            # (k, P, r)
            dropped = M @ drop                                      # (k, P, c - r)
            self.D.add_(dropped.square().sum(-1))
            self.M[:, :, :r] = W
            self.M[:, :, r:].zero_()
            self.S.zero_()
            self.S[:, :r, :r] = W.transpose(-1, -2) @ (W / self.D.unsqueeze(-1))
        self.c = r

    @torch.no_grad()
    def update(self):
        i = self.steps
        x, y, target = self.xs[i], self.ys[i], self.clean[i]
        pred, inputs, sens = sample_state(self.weights, x)
        res = pred - y
        self.error.add_((pred - target).square()); self.null.add_(target.square())
        R = self.noise_var[i].expand(self.k) if self.a.known_noise else self.noise
        j = torch.cat([(s.unsqueeze(-1) * inp.unsqueeze(1)).flatten(1) for inp, s in zip(inputs, sens)], -1)
        if self.a.forget > 0:
            self.D.mul_(1.0 - self.a.forget); self.M.mul_(math.sqrt(1.0 - self.a.forget))
        u = j / self.D                                              # D^-1 j
        c = self.c
        if c > 0:
            M = self.M[:, :, :c]
            Mu = torch.einsum("kpc,kp->kc", M, u)
            A = self.S[:, :c, :c] + torch.eye(c, device=self.dev)
            z = torch.linalg.solve(A, Mu.unsqueeze(-1)).squeeze(-1)
            pj = u - torch.einsum("kpc,kc->kp", M, z) / self.D
        else:
            pj = u
        den = R + (j * pj).sum(-1)
        start = 0
        for w in self.weights:
            stop = start + w[0].numel()
            w.sub_(pj[:, start:stop].view_as(w) * (res / den)[:, None, None])
            start = stop
        # exact precision update: new column b = j / sqrt(R); extend S by one row/column
        b = j / R.sqrt().unsqueeze(-1)
        self.M[:, :, c] = b
        bu = b / self.D
        if c > 0:
            cross = torch.einsum("kpc,kp->kc", self.M[:, :, :c], bu)
            self.S[:, c, :c] = cross; self.S[:, :c, c] = cross
        self.S[:, c, c] = (b * bu).sum(-1)
        self.c = c + 1
        if self.c == self.rank + self.T:
            self.truncate()
        self.noise.lerp_(res.square(), self.a.noise_rate)
        self.steps += 1


@torch.no_grad()
def evaluate(weights, x, y):
    err = torch.zeros(weights[0].shape[0], device=x.device)
    for s in range(0, len(x), 1024):
        xb = x[s:s + 1024]
        h1 = torch.tanh(xb @ weights[0][..., :-1].transpose(-1, -2) + weights[0][..., -1].unsqueeze(1))
        h2 = torch.tanh(h1 @ weights[1][..., :-1].transpose(-1, -2) + weights[1][..., -1].unsqueeze(1))
        pred = (h2 @ weights[2][..., :-1].transpose(-1, -2) + weights[2][..., -1].unsqueeze(1)).squeeze(-1)
        err += (pred - y[s:s + 1024]).square().sum(-1)
    return err / len(x)


def report(name, learner, xv, yv, xt, yt, zero, elapsed, extra=""):
    val = evaluate(learner.weights, xv, yv)
    test = evaluate(learner.weights, xt, yt)
    sustained = learner.error / learner.null
    best = int(val.argmin())
    edge = " EDGE" if best in (0, len(val) - 1) else ""
    print(f"  {name:>14s}: sustained {sustained[best]:.6f}  endpoint test {test[best]:.6f} (zero {zero:.4f})"
          f"  selected #{best}{edge}  all sustained " + " ".join(f"{v:.4f}" for v in sustained.tolist())
          + f"  {elapsed:.0f}s {extra}", flush=True)


def main():
    a = tyro.cli(Args)
    runtime.configure_runtime()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    t1, t2 = draw_teacher(a, gen, dev), draw_teacher(a, gen, dev)
    initial = init_weights(a, gen, dev)
    xs = torch.randn(a.samples, a.input_dim, generator=gen, device=dev)
    direction = torch.randn(a.input_dim, generator=gen, device=dev); direction /= direction.norm()
    sigma = a.noise * (a.hetero * torch.tanh(xs @ direction)).exp()
    clean = teach(t1, xs)
    switch = int(a.samples * a.switch_at) if a.switch_at else a.samples
    if switch < a.samples:
        clean[switch:] = teach(t2, xs[switch:])
    ys = clean + sigma * torch.randn(a.samples, generator=gen, device=dev)
    xv = torch.randn(a.validation, a.input_dim, generator=gen, device=dev)
    xt = torch.randn(a.test, a.input_dim, generator=gen, device=dev)
    tv, tt = (t2, t2) if a.switch_at else (t1, t1)
    yv, yt = teach(tv, xv), teach(tt, xt)
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    zero = float(yt.square().mean())
    P = sum(w.numel() for w in initial)
    print(f"stream {a.samples} samples, P={P}, hetero {a.hetero}, switch {a.switch_at}, forget {a.forget}", flush=True)
    for method in a.methods:
        if method == "lowrank":
            for r in a.ranks:
                t0 = time.time()
                L = LowRank(r, a.prior_scales, initial, a, xs, ys, clean, sigma.square())
                for _ in range(a.samples):
                    L.update()
                torch.cuda.synchronize()
                report(f"lowrank r={r}", L, xv, yv, xt, yt, zero, time.time() - t0)
        else:
            grid = a.adam_lrs if method == "adam" else a.prior_scales
            t0 = time.time()
            L = Learner(method, grid, initial, a, xs, ys, clean, sigma.square())
            for _ in range(a.samples):
                L.update()
            torch.cuda.synchronize()
            L.null = L.null_error
            report(method, L, xv, yv, xt, yt, zero, time.time() - t0)


if __name__ == "__main__":
    main()

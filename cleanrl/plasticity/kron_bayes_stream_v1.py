"""Kronecker-factored cross-neuron Bayesian plasticity, v1: the low-rank EKF's payload at O(in^2 + out^2) per layer.

Family fact (lowrank_bayes_stream_v1): the cross-neuron posterior beats Adam by 30% on the dense 17-64-64-1 stream,
rank 16 of 5,377 recovers it, and the diagonal posterior is Adam. The remaining cost is the P x (r + T) buffer and
its truncation. This file replaces the posterior over each layer's weights by a Kronecker product,

    Lambda_l ~= lambda0 I + (1/T) Ghat_l (x) Ahat_l,   Ghat_l = sum_t s_t s_t^T / R_t,   Ahat_l = sum_t a_t a_t^T,

where a_t is the layer's (bias-augmented) input and s_t = dy/dpre_t its output sensitivity (exact output Jacobian
of the sample, j_l = s (x) a). The gain is the EKF gain under that posterior with factored Tikhonov damping,

    P_l j_l = (Ghat_l/T + pi sqrt(lambda0) I)^-1 s  (x)  (Ahat_l + sqrt(lambda0)/pi I)^-1 a,
    theta <- theta - P j * res / (R + j^T P j),

so a hidden unit's plasticity on this sample is its backprop error WHITENED against the other units' errors
(G^-1 s) times the input whitened against the other inputs (A^-1 a): per-perceptron, per-sample, no learning rate.
Ablations keep only the diagonal of one or both factors: kron_diag (per-parameter, no cross-neuron information),
kron_a (cross-input only), kron_g (cross-unit only).

Same data, initialisation, validation selection and clean-error reporting as network_bayes_stream_v2 / lowrank v1.

    .venv/bin/python cleanrl/plasticity/kron_bayes_stream_v1.py --methods adam kron kron_diag kron_a kron_g
"""
import math
import time
from dataclasses import dataclass

import torch
import tyro

from cleanrl.plasticity.lowrank_bayes_stream_v1 import LowRank, report
from cleanrl.plasticity.network_bayes_stream_v2 import Learner, draw_teacher, init_weights, sample_state, teach
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
    methods: tuple[str, ...] = ("adam", "kron", "kron_diag", "kron_a", "kron_g")
    ranks: tuple[int, ...] = (256,)
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
    offline_epochs: int = 60
    offline_batch: int = 64


class Kron:
    def __init__(self, mode, grid, initial, a, xs, ys, clean, noise_var):
        self.a, self.mode = a, mode
        k, dev = len(grid), xs.device
        self.k, self.dev = k, dev
        self.weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
        self.scale = torch.tensor(grid, device=dev)
        self.lam0 = [(w.shape[-1] / self.scale) for w in self.weights]          # prior precision per layer (k,)
        self.A = [torch.zeros(k, w.shape[-1], w.shape[-1], device=dev) for w in self.weights]
        self.G = [torch.zeros(k, w.shape[-2], w.shape[-2], device=dev) for w in self.weights]
        self.T = torch.zeros((), device=dev)
        self.noise = torch.ones(k, device=dev)
        self.error = torch.zeros(k, device=dev); self.null = torch.zeros((), device=dev)
        self.xs, self.ys, self.clean, self.noise_var = xs, ys, clean, noise_var
        self.steps = 0
        self.diag_a = mode in ("kron_diag", "kron_g")
        self.diag_g = mode in ("kron_diag", "kron_a")

    @staticmethod
    def _solve(F, v, damp, diag):
        """(F + damp I)^-1 v, per config; diag keeps only F's diagonal."""
        if diag:
            return v / (torch.diagonal(F, dim1=-2, dim2=-1) + damp.unsqueeze(-1))
        eye = torch.eye(F.shape[-1], device=F.device)
        return torch.linalg.solve(F + damp[:, None, None] * eye, v.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def update(self):
        i = self.steps
        x, y, target = self.xs[i], self.ys[i], self.clean[i]
        pred, inputs, sens = sample_state(self.weights, x)
        res = pred - y
        self.error.add_((pred - target).square()); self.null.add_(target.square())
        R = self.noise_var[i].expand(self.k) if self.a.known_noise else self.noise
        if self.a.forget > 0:
            for A, G in zip(self.A, self.G):
                A.mul_(1.0 - self.a.forget); G.mul_(1.0 - self.a.forget)
            self.T.mul_(1.0 - self.a.forget)
        T = self.T.clamp_min(1.0)
        quad = torch.zeros(self.k, device=self.dev)
        directions = []
        for A, G, lam0, inp, s in zip(self.A, self.G, self.lam0, inputs, sens):
            Gn = G / T
            tr_g = torch.diagonal(Gn, dim1=-2, dim2=-1).mean(-1)
            tr_a = torch.diagonal(A, dim1=-2, dim2=-1).mean(-1)
            pi = ((tr_g + 1e-12) / (tr_a + 1e-12)).sqrt()
            rl = lam0.sqrt()
            w = self._solve(Gn, s, pi * rl, self.diag_g)                     # (k, out)
            v = self._solve(A, inp, rl / pi, self.diag_a)                     # (k, in+1)
            quad += (s * w).sum(-1) * (inp * v).sum(-1)
            directions.append((w, v))
        den = R + quad
        gain = res / den
        for W, (w, v) in zip(self.weights, directions):
            W.sub_(torch.einsum("ko,ki->koi", w, v) * gain[:, None, None])
        # posterior precision accumulation (Kronecker factors)
        for A, G, inp, s in zip(self.A, self.G, inputs, sens):
            A.add_(torch.einsum("ki,kj->kij", inp, inp))
            G.add_(torch.einsum("ko,kp->kop", s, s) / R[:, None, None])
        self.T.add_(1.0)
        self.noise.lerp_(res.square(), self.a.noise_rate)
        self.steps += 1

def offline(a, initial, xs, ys, xv, yv, xt, yt, zero, switch, t0):
    """Hindsight ceiling: multi-epoch minibatch Adam on the post-switch stream, validation-selected LR and epoch."""
    from cleanrl.plasticity.lowrank_bayes_stream_v1 import evaluate
    k = len(a.adam_lrs)
    weights = [w.unsqueeze(0).repeat(k, 1, 1).requires_grad_() for w in initial]
    lrs = torch.tensor(a.adam_lrs, device=xs.device)[:, None, None]
    m = [torch.zeros_like(w) for w in weights]; v = [torch.zeros_like(w) for w in weights]
    step = 0
    xo, yo = xs[switch:] if switch < a.samples else xs, ys[switch:] if switch < a.samples else ys
    best_val, best_test, best_ep = None, None, -1
    for ep in range(a.offline_epochs):
        perm = torch.randperm(len(xo), device=xs.device)
        for s in range(0, len(xo), a.offline_batch):
            idx = perm[s:s + a.offline_batch]
            xb, yb = xo[idx], yo[idx]
            h1 = torch.tanh(xb @ weights[0][..., :-1].transpose(-1, -2) + weights[0][..., -1].unsqueeze(1))
            h2 = torch.tanh(h1 @ weights[1][..., :-1].transpose(-1, -2) + weights[1][..., -1].unsqueeze(1))
            pred = (h2 @ weights[2][..., :-1].transpose(-1, -2) + weights[2][..., -1].unsqueeze(1)).squeeze(-1)
            loss = (pred - yb).square().mean(-1).sum()
            grads = torch.autograd.grad(loss, weights)
            step += 1
            with torch.no_grad():
                for w, g, mm, vv in zip(weights, grads, m, v):
                    mm.mul_(0.9).add_(g, alpha=0.1); vv.mul_(0.999).addcmul_(g, g, value=0.001)
                    w.sub_(lrs * (mm / (1 - 0.9 ** step)) / ((vv / (1 - 0.999 ** step)).sqrt() + 1e-8))
        with torch.no_grad():
            val = evaluate([w.detach() for w in weights], xv, yv); test = evaluate([w.detach() for w in weights], xt, yt)
        if best_val is None or float(val.min()) < float(best_val.min()):
            best_val, best_test, best_ep = val.clone(), test.clone(), ep
    b = int(best_val.argmin()); edge = " EDGE" if b in (0, k - 1) else ""
    print(f"  {'offline':>14s}: endpoint test {best_test[b]:.6f} (zero {zero:.4f})  selected lr #{b}{edge} epoch {best_ep}"
          f"  all test " + " ".join(f"{v:.4f}" for v in best_test.tolist()) + f"  {time.time() - t0:.0f}s", flush=True)



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
    tv = t2 if a.switch_at else t1
    yv, yt = teach(tv, xv), teach(tv, xt)
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    zero = float(yt.square().mean())
    P = sum(w.numel() for w in initial)
    print(f"stream {a.samples} samples, P={P}, hetero {a.hetero}, switch {a.switch_at}, forget {a.forget}", flush=True)
    for method in a.methods:
        t0 = time.time()
        if method.startswith("kron"):
            L = Kron(method, a.prior_scales, initial, a, xs, ys, clean, sigma.square())
        elif method == "lowrank":
            for r in a.ranks:
                L = LowRank(r, a.prior_scales, initial, a, xs, ys, clean, sigma.square())
                for _ in range(a.samples):
                    L.update()
                torch.cuda.synchronize()
                report(f"lowrank r={r}", L, xv, yv, xt, yt, zero, time.time() - t0)
            continue
        elif method == "offline":
            offline(a, initial, xs, ys, xv, yv, xt, yt, zero, switch, t0)
            continue
        else:
            grid = a.adam_lrs if method == "adam" else a.prior_scales
            L = Learner(method, grid, initial, a, xs, ys, clean, sigma.square())
        for _ in range(a.samples):
            L.update()
        torch.cuda.synchronize()
        if not method.startswith("kron"):
            L.null = L.null_error
        report(method, L, xv, yv, xt, yt, zero, time.time() - t0)


if __name__ == "__main__":
    main()

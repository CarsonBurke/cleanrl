"""RethinkB: per-perceptron plasticity = the conditional SNR of the unit's own error signal.

DERIVATION. Adam already steps by lr * m / sqrt(v) ~ lr * sqrt(E[g]^2 / E[g^2]): the square
root of the TEMPORAL signal-to-noise ratio of each parameter's gradient. For weight w_ji the
gradient is delta_j * a_i, so E[g | state]^2 / E[g^2 | state] = E[delta_j | a]^2 / E[delta_j^2 | a]:
the input a_i factors out and every weight into unit j shares one number, the fraction of the
error signal delta_j that is PREDICTABLE FROM THE UNIT'S OWN INPUT a on this sample. That is
exactly "how predictable is the target from the state", per perceptron: the part of delta_j that
is linear in a is the part unit j's incoming weights can fix; the rest (noise, or error only
other units can fix) is noise to it. Per sample the Wiener gate is mu(a)^2 / (mu(a)^2 + s^2(a)),
mu = E[delta_j | a], s^2 = Var(delta_j | a).
GAIN NORMALISATION (`cond`): delta_j = r * d out/d z_j, so Var(delta_j | a) carries the unit's own
tanh gain and regressing raw delta_j (`condraw`) reads that gain as a noise field (hetero-0 loss).
`cond` regresses the error in output units, r, from the layer's input; all units of a layer then
share one gate (three state views: x, h1, h2).

ESTIMATION, self-calibrated without a twin, random past or oracle: two PREQUENTIAL linear
regressions per unit on its own input, predicted BEFORE the sample is used to train them:
  mu_hat = u . [a;1]                        (E[delta | a])
  l_hat  = w . [a;1]                        (log Var(delta | a); target log e^2 + 1.27, exact for
                                             Gaussian noise since E[log chi2_1] = -1.27)
A prequential prediction is independent of this sample's noise, so E[mu_hat * delta] = E[mu^2] is
an UNBIASED estimate of the predictable power while E[mu_hat^2] over-counts by the estimation
noise. rho = E[mu_hat delta] / E[mu_hat^2] is therefore the optimal shrinkage of the readout and
also its calibration: rho -> 0 when nothing is predictable (null cell), rho -> 1 when the readout
is right. The gate's numerator is the posterior E[mu^2 | mu_hat] = rho^2 mu_hat^2 + rho(1-rho)Q.
The same shrinkage applied to the log-variance deviation kills spurious heteroscedasticity in
the homoscedastic cell. The gate multiplies the gradient entering Adam's FIRST moment only
(v sees the raw gradient), so the absolute level survives Adam's normalisation.

MEASURED (see RethinkB.md): hetero 2 -0.073 vs Adam (t -5.4, shuffle +0.010), switch -0.025,
signal-scales neutral, null cell null, homoscedastic +0.065 (a loss). Arms: adam, oracle[_shuffle],
cond[_shuffle], condraw[_shuffle], param[_shuffle] (per-weight scalar state; null).

    .venv/bin/python cleanrl/plasticity/rethink/RethinkB.py --hetero 2 --methods adam,oracle,cond,cond_shuffle
"""
import argparse
import math
import time

import torch

D_IN, H, D_OUT = 17, 64, 1
DIMS = ((D_IN, H), (H, H), (H, D_OUT))
LOG_CHI2 = 1.2703628454614782  # -E[log chi2_1] = euler_gamma + log 2
METHODS = ("adam", "oracle", "oracle_shuffle", "cond", "cond_shuffle", "condraw", "condraw_shuffle", "param", "param_shuffle")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--samples", type=int, default=32768)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr-grid", type=float, nargs="+",
                   default=[2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2])
    p.add_argument("--hetero", type=float, default=2.0)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--tail-df", type=float, default=0.0)
    p.add_argument("--switch-at", type=float, default=0.0)
    p.add_argument("--signal-scales", action="store_true")
    p.add_argument("--null", action="store_true")
    p.add_argument("--tau", type=float, default=1000.0,
                   help="horizon (samples) of the prequential readouts = 1/(1 - Adam beta2); not per-task")
    p.add_argument("--diag", type=int, default=0, help="print readout diagnostics every N steps")
    p.add_argument("--wide", action="store_true",
                   help="cond readouts see the concatenated network state [x;h1;h2] instead of the unit's own input")
    p.add_argument("--gate", default="m", choices=("m", "pre", "post"),
                   help="m: gate the gradient entering Adam's m only; pre: gate level-normalised "
                        "to mean 1 entering m and v; post: gate multiplies the Adam step")
    p.add_argument("--no-shrink", action="store_true", help="ablation: rho = 1 (no self-calibration)")
    p.add_argument("--no-var", action="store_true", help="ablation: unconditional variance (rho_v = 0)")
    p.add_argument("--no-mean", action="store_true", help="ablation: numerator = unconditional E[mu^2]")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--test", type=int, default=4096)
    p.add_argument("--master-seed", type=int, default=1)
    return p.parse_args()


def init_mlp(S, G, gen, device):
    def layer(o, i, std):
        w = torch.randn(S, max(o, i), min(o, i), generator=gen, device=device)
        q, _ = torch.linalg.qr(w)
        w = (q if o > i else q.transpose(-1, -2)) * std
        return w.unsqueeze(1).expand(S, G, o, i).clone(), torch.zeros(S, G, o, device=device)
    W1, b1 = layer(H, D_IN, math.sqrt(2))
    W2, b2 = layer(H, H, math.sqrt(2))
    W3, b3 = layer(D_OUT, H, 1.0)
    return [W1, b1, W2, b2, W3, b3]


def teacher(S, gen, device):
    Ws = [torch.randn(S, H, D_IN, generator=gen, device=device) / math.sqrt(D_IN),
          torch.randn(S, H, H, generator=gen, device=device) / math.sqrt(H) * 1.5,
          torch.randn(S, D_OUT, H, generator=gen, device=device) / math.sqrt(H) * 3.0]
    v = torch.randn(S, D_IN, generator=gen, device=device)
    return Ws, v / v.norm(dim=-1, keepdim=True)


def teach(Ws, x):
    h = torch.tanh(torch.einsum("shi,sbi->sbh", Ws[0], x))
    h = torch.tanh(torch.einsum("shi,sbi->sbh", Ws[1], h))
    return torch.einsum("soh,sbh->sbo", Ws[2], h)


def forward(P, x):
    W1, b1, W2, b2, W3, b3 = P
    h1 = torch.tanh(torch.einsum("sgoi,sbi->sgbo", W1, x) + b1.unsqueeze(2))
    h2 = torch.tanh(torch.einsum("sgoi,sgbi->sgbo", W2, h1) + b2.unsqueeze(2))
    out = torch.einsum("sgoi,sgbi->sgbo", W3, h2) + b3.unsqueeze(2)
    return h1, h2, out


class UnitReadout:
    """Per-unit prequential regressions of the unit's error and its log-variance on the unit's input."""

    def __init__(self, S, G, n, o, tau, opts, device):
        z = lambda *s: torch.zeros(S, G, *s, device=device)
        self.U = z(o, n + 1)          # E[delta | a]
        self.Wl = z(o, n + 1)         # log Var(delta | a)
        self.P, self.Q = z(o), z(o)   # E[mu_hat delta], E[mu_hat^2]
        self.Lm = z(o)                # E[l_hat]
        self.Pv, self.Qv = z(o), z(o)  # E[dev (y - Lm)], E[dev^2]
        # NLMS moves the prediction for THIS input by the rate; n+1 regressors need (n+1)/tau
        # per sample for the regression to have the same horizon tau as the scalar statistics
        self.eta_reg, self.eta_lvl = min((n + 1) / tau, 1.0), 1.0 / tau
        self.opts, self.t = opts, 0

    def gate(self, A):  # A (S,G,B,n) -> c (S,G,B,o)
        S, G, B, n = A.shape
        self.At = torch.cat((A, torch.ones(S, G, B, 1, device=A.device)), -1)
        self.mu = torch.einsum("sgok,sgbk->sgbo", self.U, self.At)
        self.lh = torch.einsum("sgok,sgbk->sgbo", self.Wl, self.At)
        rho = torch.ones_like(self.P) if self.opts.no_shrink else (self.P / self.Q.clamp_min(1e-30)).clamp(0.0, 1.0)
        rho = rho.unsqueeze(2)
        Q = self.Q.unsqueeze(2)
        if self.opts.no_mean:
            num = (rho * Q).expand_as(self.mu)
        else:
            num = rho.square() * self.mu.square() + rho * (1.0 - rho) * Q
        self.dev = self.lh - self.Lm.unsqueeze(2)
        rhov = torch.zeros_like(self.Pv) if self.opts.no_var else (self.Pv / self.Qv.clamp_min(1e-30)).clamp(0.0, 1.0)
        rhov = torch.ones_like(rhov) if self.opts.no_shrink and not self.opts.no_var else rhov
        s2 = torch.exp((self.Lm.unsqueeze(2) + rhov.unsqueeze(2) * self.dev).clamp(-30.0, 30.0))
        return num / (num + s2).clamp_min(1e-30)

    def learn(self, D):  # D (S,G,B,o): the realised error signal of this sample
        warm = 1.0 / (self.t + 1)
        rate_reg, rate_lvl = max(self.eta_reg, warm), max(self.eta_lvl, warm)
        self.t += 1
        e = D - self.mu
        y = torch.log(e.square() + 1e-30) + LOG_CHI2
        a2 = self.At.square().sum(-1, keepdim=True)  # (S,G,B,1)
        # prequential calibration statistics (this sample's realisation vs its pre-update prediction)
        self.P.lerp_((self.mu * D).mean(2), rate_lvl)
        self.Q.lerp_(self.mu.square().mean(2), rate_lvl)
        self.Pv.lerp_((self.dev * (y - self.Lm.unsqueeze(2))).mean(2), rate_lvl)
        self.Qv.lerp_(self.dev.square().mean(2), rate_lvl)
        self.Lm.lerp_(self.lh.mean(2), rate_lvl)
        # normalised LMS on both regressions (per-sample rate, summed over the batch)
        self.U.add_(torch.einsum("sgbo,sgbk->sgok", e / a2, self.At), alpha=rate_reg)
        self.Wl.add_(torch.einsum("sgbo,sgbk->sgok", (y - self.lh) / a2, self.At), alpha=rate_reg)

    def stats(self):  # unit-averaged (S,G) diagnostics
        rho = (self.P / self.Q.clamp_min(1e-30)).clamp(0.0, 1.0)
        rhov = (self.Pv / self.Qv.clamp_min(1e-30)).clamp(0.0, 1.0)
        return dict(rho=rho.mean(-1), rhov=rhov.mean(-1), Q=self.Q.mean(-1), s2=self.Lm.exp().mean(-1))


class ParamReadout:
    """Per-WEIGHT version: w_ji's state is the scalar a_i; regress delta_j and log-var on (1, a_i)."""

    def __init__(self, S, G, n, o, tau, opts, device):
        z = lambda *s: torch.zeros(S, G, *s, device=device)
        self.U0, self.U1 = z(o, n), z(o, n)
        self.W0, self.W1 = z(o, n), z(o, n)
        self.P, self.Q, self.Lm, self.Pv, self.Qv = z(o, n), z(o, n), z(o, n), z(o, n), z(o, n)
        self.eta_reg, self.eta_lvl = min(2.0 / tau, 1.0), 1.0 / tau
        self.opts, self.t = opts, 0

    def gate(self, A):  # A (S,G,B,n) -> c (S,G,B,o,n)
        self.A = A.unsqueeze(3)                                  # (S,G,B,1,n)
        self.mu = self.U0.unsqueeze(2) + self.U1.unsqueeze(2) * self.A
        self.lh = self.W0.unsqueeze(2) + self.W1.unsqueeze(2) * self.A
        rho = torch.ones_like(self.P) if self.opts.no_shrink else (self.P / self.Q.clamp_min(1e-30)).clamp(0.0, 1.0)
        rho, Q = rho.unsqueeze(2), self.Q.unsqueeze(2)
        num = (rho * Q).expand_as(self.mu) if self.opts.no_mean else rho.square() * self.mu.square() + rho * (1.0 - rho) * Q
        self.dev = self.lh - self.Lm.unsqueeze(2)
        rhov = torch.zeros_like(self.Pv) if self.opts.no_var else (self.Pv / self.Qv.clamp_min(1e-30)).clamp(0.0, 1.0)
        rhov = torch.ones_like(rhov) if self.opts.no_shrink and not self.opts.no_var else rhov
        s2 = torch.exp((self.Lm.unsqueeze(2) + rhov.unsqueeze(2) * self.dev).clamp(-30.0, 30.0))
        return num / (num + s2).clamp_min(1e-30)

    def learn(self, D):  # D (S,G,B,o)
        warm = 1.0 / (self.t + 1)
        rate_reg, rate_lvl = max(self.eta_reg, warm), max(self.eta_lvl, warm)
        self.t += 1
        Dn = D.unsqueeze(-1)
        e = Dn - self.mu
        y = torch.log(e.square() + 1e-30) + LOG_CHI2
        a2 = 1.0 + self.A.square()
        self.P.lerp_((self.mu * Dn).mean(2), rate_lvl)
        self.Q.lerp_(self.mu.square().mean(2), rate_lvl)
        self.Pv.lerp_((self.dev * (y - self.Lm.unsqueeze(2))).mean(2), rate_lvl)
        self.Qv.lerp_(self.dev.square().mean(2), rate_lvl)
        self.Lm.lerp_(self.lh.mean(2), rate_lvl)
        ea, ya = e / a2, (y - self.lh) / a2
        self.U0.add_(ea.sum(2), alpha=rate_reg)
        self.U1.add_((ea * self.A).sum(2), alpha=rate_reg)
        self.W0.add_(ya.sum(2), alpha=rate_reg)
        self.W1.add_((ya * self.A).sum(2), alpha=rate_reg)


def run(a):
    dev = torch.device("cuda")
    methods = a.methods.split(",")
    S, G, B = a.seeds, len(a.lr_grid), a.batch
    lr = torch.tensor(a.lr_grid, device=dev).view(1, G, 1, 1)
    lr_b = lr.view(1, G, 1)
    steps = a.samples // B
    gen = torch.Generator(device=dev).manual_seed(a.master_seed)
    Ws, v = teacher(S, gen, dev)
    Ws2, _ = teacher(S, gen, dev)
    x_te = torch.randn(S, a.test, D_IN, generator=gen, device=dev)
    logsig_te = a.hetero * torch.tanh(torch.einsum("si,sbi->sb", v, x_te))
    amp_te = torch.exp(logsig_te) if a.signal_scales else torch.ones_like(logsig_te)
    if a.null:
        amp_te = torch.zeros_like(amp_te)
    y_te = teach(Ws2 if a.switch_at > 0 else Ws, x_te) * amp_te.unsqueeze(-1)
    P0 = init_mlp(S, G, gen, dev)
    xs = torch.randn(steps, S, B, D_IN, generator=gen, device=dev)
    logsig = a.hetero * torch.tanh(torch.einsum("si,tsbi->tsb", v, xs))
    sigma = a.noise * torch.exp(logsig)
    amp = torch.exp(logsig) if a.signal_scales else torch.ones_like(logsig)
    if a.null:
        amp = torch.zeros_like(amp)
    if a.tail_df > 0:
        zn = torch.randn(steps, S, B, generator=gen, device=dev)
        chi = torch.distributions.Chi2(a.tail_df).sample((steps, S, B)).to(dev)
        eps = zn / torch.sqrt(chi / a.tail_df) * math.sqrt((a.tail_df - 2) / a.tail_df)
    else:
        eps = torch.randn(steps, S, B, generator=gen, device=dev)
    switch = int(steps * a.switch_at) if a.switch_at > 0 else steps + 1
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    beta_lvl = (1 - 1.0 / a.tau) ** B

    results, levels = {}, {}
    for method in methods:
        t0 = time.time()
        base = method[:-len("_shuffle")] if method.endswith("_shuffle") else method
        shuffle = method.endswith("_shuffle")
        P = [p.clone() for p in P0]
        M = [torch.zeros_like(p) for p in P]
        V = [torch.zeros_like(p) for p in P]
        ro_n = [D_IN + H + H] * 3 if a.wide else [n for n, _ in DIMS]
        ros = None
        if base in ("cond", "condraw"):
            ros = [UnitReadout(S, G, n, 1 if base == "cond" else o, a.tau, a, dev) for n, (_, o) in zip(ro_n, DIMS)]
        if base == "param":  # weights: scalar-state readouts; biases: state-free (intercept-only) unit readouts
            ros = [(ParamReadout(S, G, n, o, a.tau, a, dev), UnitReadout(S, G, 0, o, a.tau, a, dev)) for n, o in DIMS]
        c_prev = [(torch.ones(S, G, B, o, n, device=dev), torch.ones(S, G, B, o, device=dev)) if base == "param"
                  else torch.ones(S, G, B, o, device=dev) for n, o in DIMS]
        c_lvl = torch.ones(S, G, device=dev)
        lvl_sum = torch.zeros(S, G, device=dev)
        lvl_sq = torch.zeros(S, G, device=dev)
        ones_c = [torch.ones(S, G, B, o, device=dev) for _, o in DIMS]
        with torch.no_grad():
            for t in range(steps):
                x = xs[t]
                y_clean = teach(Ws2 if t >= switch else Ws, x) * amp[t].unsqueeze(-1)
                y = y_clean + (sigma[t] * eps[t]).unsqueeze(-1)
                W1, b1, W2, b2, W3, b3 = P
                h1, h2, out = forward(P, x)
                r = out - y.unsqueeze(1)                                        # (S,G,B,1)
                d3 = r
                d2 = torch.einsum("sgbo,sgoi->sgbi", d3, W3) * (1 - h2 * h2)
                d1 = torch.einsum("sgbo,sgoi->sgbi", d2, W2) * (1 - h1 * h1)
                xg = x.unsqueeze(1).expand(S, G, B, D_IN)
                deltas, inputs = (d1, d2, d3), (xg, h1, h2)

                if base == "adam":
                    cs = ones_c
                elif base == "oracle":
                    b2t = (out - y_clean.unsqueeze(1)).square()                  # (S,G,B,1)
                    cw = b2t / (b2t + sigma[t].square().unsqueeze(1).unsqueeze(-1))
                    cs = [cw.expand(S, G, B, o) for _, o in DIMS]
                elif base in ("cond", "condraw"):
                    ro_in = [torch.cat((xg, h1, h2), -1)] * 3 if a.wide else inputs
                    cs = [ro.gate(inp) for ro, inp in zip(ros, ro_in)]
                    if base == "cond":
                        # every unit's error is delta_j = r * d out/d z_j: regress the error IN OUTPUT UNITS
                        # (delta_j / gain_j = r) so the unit's own gain does not masquerade as a noise field
                        cs = [c.expand(S, G, B, o) for c, (_, o) in zip(cs, DIMS)]
                        for ro in ros:
                            ro.learn(r)
                    else:
                        for ro, d in zip(ros, deltas):
                            ro.learn(d)
                elif base == "param":
                    cs = [(rw.gate(inp), rb.gate(inp[..., :0])) for (rw, rb), inp in zip(ros, inputs)]
                    for (rw, rb), d in zip(ros, deltas):
                        rw.learn(d)
                        rb.learn(d)
                else:
                    raise ValueError(method)
                if shuffle:
                    if B == 1:
                        cs, c_prev = c_prev, cs
                    else:
                        perm = torch.randperm(B, device=dev)
                        cs = [tuple(cc[:, :, perm] for cc in c) if isinstance(c, tuple) else c[:, :, perm] for c in cs]
                lvl = sum((c[0].mean((2, 3, 4)) if isinstance(c, tuple) else c.mean((2, 3))) for c in cs) / len(cs)
                lvl_sum += lvl
                lvl_sq += lvl.square()
                if a.diag and base in ("cond", "condraw") and (t + 1) % a.diag == 0:
                    b2t = (out - y_clean.unsqueeze(1)).square()
                    cw = (b2t / (b2t + sigma[t].square().unsqueeze(1).unsqueeze(-1))).mean((0, 2, 3))
                    line = f"# t={t + 1} oracle_gate {cw.mean().item():.3f}"
                    for li, (ro, c) in enumerate(zip(ros, cs)):
                        st = ro.stats()
                        line += (f" | L{li + 1} gate {c.mean((0, 2, 3)).mean().item():.3f} rho {st['rho'].mean().item():.2f}"
                                 f" rhov {st['rhov'].mean().item():.2f} Q {st['Q'].mean().item():.2e} s2 {st['s2'].mean().item():.2e}")
                    print(line, flush=True)
                if a.gate == "pre":
                    c_lvl = beta_lvl * c_lvl + (1 - beta_lvl) * lvl
                    norm = lambda c: (c / c_lvl.view(S, G, 1, 1, *([1] * (c.dim() - 4))).clamp_min(1e-12)).clamp_max(20.0)
                    cs = [tuple(norm(cc) for cc in c) if isinstance(c, tuple) else norm(c) for c in cs]

                bc1 = 1 - beta1 ** (t + 1)
                bc2 = 1 - beta2 ** (t + 1)
                for li, (d, inp, c) in enumerate(zip(deltas, inputs, cs)):
                    gW_raw = torch.einsum("sgbo,sgbi->sgoi", d, inp) / B
                    gb_raw = d.sum(2) / B
                    if a.gate == "post":
                        assert not isinstance(c, tuple)
                        gW, gb = gW_raw, gb_raw
                    elif isinstance(c, tuple):
                        gW = torch.einsum("sgboi,sgbo,sgbi->sgoi", c[0], d, inp) / B
                        gb = (c[1] * d).sum(2) / B
                    else:
                        cd = c * d
                        gW = torch.einsum("sgbo,sgbi->sgoi", cd, inp) / B
                        gb = cd.sum(2) / B
                    for k, (g, g_raw) in enumerate(((gW, gW_raw), (gb, gb_raw))):
                        idx = 2 * li + k
                        p, m, vv = P[idx], M[idx], V[idx]
                        m.mul_(beta1).add_(g, alpha=1 - beta1)
                        gv = g if a.gate == "pre" else g_raw
                        vv.mul_(beta2).addcmul_(gv, gv, value=1 - beta2)
                        step = (m / bc1) / ((vv / bc2).sqrt() + adam_eps)
                        if a.gate == "post":
                            assert B == 1
                            step = step * (c[:, :, 0].unsqueeze(-1) if k == 0 else c[:, :, 0])
                        p.sub_((lr if p.dim() == 4 else lr_b) * step)
            mse = torch.zeros(S, G, device=dev)
            for j in range(0, a.test, 512):  # chunked: the shared GPU has little headroom
                _, _, out_te = forward(P, x_te[:, j:j + 512])
                mse += (out_te - y_te[:, j:j + 512].unsqueeze(1)).square().sum((-1, -2))
            mse /= a.test * D_OUT
        results[method] = mse
        levels[method] = (lvl_sum / steps, (lvl_sq / steps - (lvl_sum / steps) ** 2).clamp_min(0).sqrt())
        print(f"# {method}: {time.time() - t0:.0f}s", flush=True)

    base_mse = (y_te.square().mean((-1, -2))).mean().item()
    print(f"# hetero={a.hetero} noise={a.noise} tail_df={a.tail_df} switch={a.switch_at} signal_scales={a.signal_scales} "
          f"null={a.null} batch={B} samples={a.samples} seeds={S} gate={a.gate} tau={a.tau} "
          f"shrink={not a.no_shrink} var={not a.no_var} mean={not a.no_mean}  zero-predictor mse {base_mse:.4f}")
    print(f"{'method':16s} {'best_lr':>8s} {'mse':>8s} {'sem':>7s}   {'c_mean':>6s} {'c_sd':>6s}  "
          + " ".join(f"{l:8.0e}" for l in a.lr_grid))
    best = {}
    for method in methods:
        mse = results[method]
        mean_over_seeds = mse.mean(0)
        gi = int(mean_over_seeds.argmin())
        edge = " EDGE" if gi in (0, G - 1) else ""
        best[method] = mse[:, gi]
        sem = mse[:, gi].std().item() / math.sqrt(S)
        lm, ls = levels[method]
        print(f"{method:16s} {a.lr_grid[gi]:8.0e} {mean_over_seeds[gi].item():8.4f} {sem:7.4f}   "
              f"{lm[:, gi].mean().item():6.3f} {ls[:, gi].mean().item():6.3f}  "
              + " ".join(f"{m:8.4f}" for m in mean_over_seeds.tolist()) + edge)
    if "adam" in best:
        print("# paired vs adam (best LR each):  mean diff  sem  t")
        for method in methods:
            if method == "adam":
                continue
            diff = best[method] - best["adam"]
            sem = diff.std().item() / math.sqrt(S)
            print(f"  {method:16s} {diff.mean().item():+8.4f} {sem:7.4f} {diff.mean().item() / max(sem, 1e-12):+6.2f}")


if __name__ == "__main__":
    run(parse())

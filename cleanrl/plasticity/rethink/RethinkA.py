"""RethinkA: per-perceptron plasticity = held-out predictability of the unit's own error from
the unit's own input, on the current sample.

Unit j (any layer) sees input vector a (its presynaptic activations) and receives the backprop
error d_j. Its incoming weights can only ever absorb the part of d_j that is LINEAR in a, so the
right question for the unit is "how much of d_j at THIS a is linearly predictable?". Two shadow
readouts per unit, both normalized-LMS with one global rate:
    mu_j(a)  = U_j . [a,1]              (predicted error at this state, read BEFORE it is updated
                                         with this sample, so it is a held-out forecast)
    s2_j(a)  = exp(V_j . [a,1])         (predicted held-out squared forecast error at this state)
    gate_j   = mu^2 / (mu^2 + s2)       (Wiener gain: explained / total, in [0,1])
mu^2 estimates b^2 restricted to what the unit can fix, s2 estimates sigma^2 + what it cannot fix.
The gate then either multiplies the unit's error before Adam (level-normalized by its running
mean so mean plasticity ~ 1) or multiplies the unit's Adam step (post).

Ablations via --form (prec = 1/s2 only, mu, r2, bprec = sbar2/(sbar2+s2)), --debias, --gate-layers.

MEASURED (see RethinkA.md): the numerator (linear forecast of the unit's error) is falsified -- at a first-order
stationary point E[d_j a] = 0 for every unit, so it decays to estimation noise; what survives is the per-unit
state-conditional precision 1/s2(a), which is null in homoscedastic/null cells, wins hetero 2 (-0.12 vs Adam,
shuffle +0.03) and switch (-0.07, shuffle +0.07), and loses signal-scales (+1.0, = its shuffle).

    .venv/bin/python cleanrl/plasticity/rethink/RethinkA.py --hetero 2 --readout-lr 0.02            # wiener form
    .venv/bin/python cleanrl/plasticity/rethink/RethinkA.py --hetero 2 --readout-lr 0.02 --form prec --norm unit \
        --methods adam,r2pre,r2pre_shuffle

Task generator copied from hetvar_stream.py (same seeds, same numbers).
"""
import argparse
import math
import time

import torch

D_IN, H, D_OUT = 17, 64, 1
METHODS = ("adam", "oracle_wiener", "r2post", "r2post_shuffle", "r2pre", "r2pre_shuffle")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--samples", type=int, default=32768)
    p.add_argument("--signal-scales", action="store_true")
    p.add_argument("--null", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr-grid", type=float, nargs="+",
                   default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    p.add_argument("--hetero", type=float, default=2.0)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--tail-df", type=float, default=0.0)
    p.add_argument("--switch-at", type=float, default=0.0)
    p.add_argument("--readout-lr", type=float, default=0.005, help="normalized-LMS rate of the readouts")
    p.add_argument("--cap", type=float, default=20.0, help="cap on level-normalized pre-Adam gate")
    p.add_argument("--norm", default="global", choices=("global", "unit"),
                   help="pre-Adam level normalizer: one running mean over all units or one per unit")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--form", default="wiener", choices=("wiener", "prec", "mu", "r2", "bprec"),
                   help="ablation: mu^2/(mu^2+s2) | precision 1/s2 only | mu^2 only | "
                        "r2 = 1 - E[e^2|a]/E[d^2|a] from two held-out log-variance readouts | "
                        "bprec = sbar2/(sbar2+s2), sbar2 the unit's running mean of s2 (bounded precision)")
    p.add_argument("--debias", action="store_true",
                   help="wiener: subtract the NLMS misadjustment variance eta/(2-eta)*s2 from mu^2 (positive part)")
    p.add_argument("--gate-layers", default="1,2,3", help="diagnostic: layers whose units are gated (others get 1)")
    p.add_argument("--test", type=int, default=4096)
    p.add_argument("--tag", default="")
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


def main():
    a = parse()
    dev = torch.device("cuda")
    methods = a.methods.split(",")
    S, G, B = a.seeds, len(a.lr_grid), a.batch
    lr = torch.tensor(a.lr_grid, device=dev).view(1, G, 1, 1)
    lr_b = lr.view(1, G, 1)
    steps = a.samples // B
    gen = torch.Generator(device=dev).manual_seed(1)
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
        z = torch.randn(steps, S, B, generator=gen, device=dev)
        chi = torch.distributions.Chi2(a.tail_df).sample((steps, S, B)).to(dev)
        eps = z / torch.sqrt(chi / a.tail_df) * math.sqrt((a.tail_df - 2) / a.tail_df)
    else:
        eps = torch.randn(steps, S, B, generator=gen, device=dev)
    switch = int(steps * a.switch_at) if a.switch_at > 0 else steps + 1
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    eta = a.readout_lr
    dims = ((D_IN, H), (H, H), (H, D_OUT))          # (inputs, units) per layer
    ones = torch.ones(S, G, B, 1, device=dev)
    gate_layers = [int(s) for s in a.gate_layers.split(",")]

    results, levels, corrs, lcorrs = {}, {}, {}, {}
    for method in methods:
        t0 = time.time()
        base = method[:-len("_shuffle")] if method.endswith("_shuffle") else method
        shuffle = method.endswith("_shuffle")
        P = [p.clone() for p in P0]
        M = [torch.zeros_like(p) for p in P]
        V = [torch.zeros_like(p) for p in P]
        # per-unit shadow readouts over [a, 1]: mean U and log-second-moment Vr
        U = [torch.zeros(S, G, o, i + 1, device=dev) for i, o in dims]
        Vr = [torch.zeros(S, G, o, i + 1, device=dev) for i, o in dims]
        Vt = [torch.zeros(S, G, o, i + 1, device=dev) for i, o in dims]
        lvl_sum = torch.zeros(S, G, device=dev)      # running sum of the raw gate (mean over units)
        lvl_sq = torch.zeros(S, G, device=dev)
        unit_sum = [torch.zeros(S, G, o, device=dev) for _, o in dims]
        prev = None
        # correlation of the (unit-mean) gate with the oracle gate, accumulated over the stream
        cg = torch.zeros(S, G, device=dev); co = torch.zeros(S, G, device=dev)
        cgo = torch.zeros(S, G, device=dev); cgg = torch.zeros(S, G, device=dev); coo = torch.zeros(S, G, device=dev)
        lcg = [torch.zeros(S, G, device=dev) for _ in range(3)]
        lcgo = [torch.zeros(S, G, device=dev) for _ in range(3)]
        s2_sum = [torch.zeros(S, G, o, device=dev) for _, o in dims]
        lcgg = [torch.zeros(S, G, device=dev) for _ in range(3)]
        for t in range(steps):
            x = xs[t]
            y_clean = teach(Ws2 if t >= switch else Ws, x) * amp[t].unsqueeze(-1)
            y = y_clean + (sigma[t] * eps[t]).unsqueeze(-1)
            W1, b1, W2, b2, W3, b3 = P
            h1, h2, out = forward(P, x)
            r = out - y.unsqueeze(1)                                          # (S,G,B,1)
            b_true = (out - y_clean.unsqueeze(1)).squeeze(-1)                 # (S,G,B)
            d3 = r
            d2 = torch.einsum("sgbo,sgoi->sgbi", d3, W3) * (1 - h2 * h2)
            d1 = torch.einsum("sgbo,sgoi->sgbi", d2, W2) * (1 - h1 * h1)
            xg = x.unsqueeze(1).expand(S, G, B, D_IN)
            deltas, inputs = (d1, d2, d3), (xg, h1, h2)
            with torch.no_grad():
                b2 = b_true.square()
                g_oracle = b2 / (b2 + sigma[t].square().unsqueeze(1))           # (S,G,B)
                if base == "adam":
                    gates = [torch.ones_like(d) for d in deltas]
                elif base == "oracle_wiener":
                    gates = [g_oracle.unsqueeze(-1).expand_as(d) for d in deltas]
                else:
                    gates = []
                    for k, (d, inp) in enumerate(zip(deltas, inputs)):
                        at = torch.cat((inp, ones), -1)                          # (S,G,B,i+1)
                        mu = torch.einsum("sgji,sgbi->sgbj", U[k], at)           # held-out forecast
                        s2 = torch.exp(torch.einsum("sgji,sgbi->sgbj", Vr[k], at))
                        e = d - mu
                        m2 = mu.square()
                        if a.form == "r2":
                            t2 = torch.exp(torch.einsum("sgji,sgbi->sgbj", Vt[k], at))   # E[d^2 | a]
                            gates.append((1.0 - s2 / t2).clamp(0.0, 1.0))
                        elif a.form == "bprec":
                            sbar = (s2_sum[k] / max(t * B, 1)).unsqueeze(2)
                            s2_sum[k] += s2.sum(2)
                            gates.append(sbar / (sbar + s2) if t > 0 else torch.full_like(s2, 0.5))
                        else:
                            num = (m2 - eta / (2.0 - eta) * s2).clamp_min(0.0) if a.debias else m2
                            gates.append(num / (m2 + s2) if a.form == "wiener" else (1.0 / s2 if a.form == "prec" else m2))
                        # normalized LMS on the mean; Gaussian-NLL step on the log variances
                        an = at / at.square().sum(-1, keepdim=True)
                        U[k].add_(eta * torch.einsum("sgbj,sgbi->sgji", e, an))
                        z2 = (e.square() / s2).clamp_max(16.0)
                        Vr[k].add_(eta * torch.einsum("sgbj,sgbi->sgji", z2 - 1.0, an))
                        if a.form == "r2":
                            z2 = (d.square() / t2).clamp_max(16.0)
                            Vt[k].add_(eta * torch.einsum("sgbj,sgbi->sgji", z2 - 1.0, an))
                    gates = [g if (k + 1) in gate_layers else torch.ones_like(g) for k, g in enumerate(gates)]
                gm = torch.cat([g.mean(-1) for g in gates], 0).view(3, S, G, B).mean(0)   # (S,G,B)
                cg += gm.mean(-1); co += g_oracle.mean(-1); cgo += (gm * g_oracle).mean(-1)
                cgg += gm.square().mean(-1); coo += g_oracle.square().mean(-1)
                lvl_sum += gm.mean(-1); lvl_sq += gm.square().mean(-1)
                for k in range(3):
                    unit_sum[k] += gates[k].sum(2)
                    lg = gates[k].mean(-1)
                    lcg[k] += lg.mean(-1); lcgo[k] += (lg * g_oracle).mean(-1); lcgg[k] += lg.square().mean(-1)
                if base == "r2pre":
                    if a.norm == "global":
                        level = (lvl_sum / (t + 1)).view(S, G, 1, 1)
                        cs = [(g / level.clamp_min(1e-8)).clamp_max(a.cap) for g in gates]
                    else:
                        cs = [(g / (u / (B * (t + 1))).unsqueeze(2).clamp_min(1e-8)).clamp_max(a.cap)
                              for g, u in zip(gates, unit_sum)]
                else:
                    cs = gates
                if shuffle:
                    if B == 1:
                        if prev is None:
                            prev = [torch.ones_like(c) for c in cs]
                        cs, prev = prev, cs
                    else:
                        perm = torch.randperm(B, device=dev)
                        cs = [c[:, :, perm] for c in cs]
                post = base == "r2post"
                grads = []
                for d, inp, c in zip(deltas, inputs, cs):
                    cd = d if post else c * d
                    grads.append(torch.einsum("sgbo,sgbi->sgoi", cd, inp) / B)
                    grads.append(cd.sum(2) / B)
                bc1 = 1 - beta1 ** (t + 1)
                bc2 = 1 - beta2 ** (t + 1)
                for k, (p, m, vv, g) in enumerate(zip(P, M, V, grads)):
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    vv.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    step = (m / bc1) / ((vv / bc2).sqrt() + adam_eps)
                    if post:
                        assert B == 1
                        gk = cs[k // 2].squeeze(2)                                  # (S,G,units)
                        step = step * (gk.unsqueeze(-1) if p.dim() == 4 else gk)
                    p.sub_((lr if p.dim() == 4 else lr_b) * step)
        with torch.no_grad():
            _, _, out_te = forward(P, x_te)
            mse = (out_te - y_te.unsqueeze(1)).square().mean((-1, -2))
        results[method] = mse
        n = steps
        levels[method] = (lvl_sum / n, (lvl_sq / n - (lvl_sum / n) ** 2).clamp_min(0).sqrt())
        cov = cgo / n - (cg / n) * (co / n)
        corrs[method] = cov / ((cgg / n - (cg / n) ** 2).clamp_min(1e-12).sqrt() * (coo / n - (co / n) ** 2).clamp_min(1e-12).sqrt())
        so = (coo / n - (co / n) ** 2).clamp_min(1e-12).sqrt()
        lcorrs[method] = [(lcgo[k] / n - (lcg[k] / n) * (co / n)) / ((lcgg[k] / n - (lcg[k] / n) ** 2).clamp_min(1e-12).sqrt() * so)
                          for k in range(3)]
        print(f"# {method} done in {time.time() - t0:.0f}s", flush=True)

    base_mse = (y_te.square().mean((-1, -2))).mean().item()
    print(f"# {a.tag} hetero={a.hetero} signal_scales={a.signal_scales} null={a.null} tail_df={a.tail_df} "
          f"switch={a.switch_at} batch={B} samples={a.samples} seeds={S} readout_lr={eta} norm={a.norm} "
          f"zero-predictor mse {base_mse:.4f}")
    print(f"{'method':16s} {'best_lr':>8s} {'mse':>8s} {'sem':>7s}  {'g_mean':>6s} {'g_sd':>6s} {'corr':>6s}  "
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
        lc = " ".join(f"{lcorrs[method][k][:, gi].mean().item():+5.2f}" for k in range(3))
        print(f"{method:16s} {a.lr_grid[gi]:8.0e} {mean_over_seeds[gi].item():8.4f} {sem:7.4f}  "
              f"{lm[:, gi].mean().item():6.3f} {ls[:, gi].mean().item():6.3f} {corrs[method][:, gi].mean().item():6.3f}  "
              + " ".join(f"{m:8.4f}" for m in mean_over_seeds.tolist()) + edge + f"   layer-corr {lc}")
    if "adam" in best:
        print("# paired vs adam (best LR each):  mean diff  sem  t")
        for method in methods:
            if method == "adam":
                continue
            diff = best[method] - best["adam"]
            sem = diff.std().item() / math.sqrt(S)
            print(f"  {method:16s} {diff.mean().item():+8.4f} {sem:7.4f} {diff.mean().item() / max(sem, 1e-12):+6.2f}")


if __name__ == "__main__":
    main()

"""Per-SAMPLE plasticity on a standard dense net: can a sample's own agreement with the
optimizer's first moment recover the Gauss-Markov precision weight?

Model: g_t = gbar + eps_t, Var(eps_t) = sigma_t^2 I. The update sum_t c_t g_t has maximal
expected second-order gain at c_t ~ 1/sigma_t^2 (Gauss-Markov). Gain over uniform is
mean(sigma^2) * mean(1/sigma^2) >= 1, equal to 1 iff homoscedastic -- so headroom exists
ONLY under heteroscedastic / heavy-tailed sample noise, and the homoscedastic cell of this
sweep must come out null for every arm.

Proposal (`agree`): with m the optimizer's causal first moment (before this step),
E[cos^2(g_t, m)] ~ |gbar|^2 / (|gbar|^2 + D sigma_t^2), so cos_t^2 / (1 - cos_t^2) is
proportional to 1/sigma_t^2: the sample's own agreement IS its precision weight, no tuned
scale. Sign-free, so a post-switch sample opposing a stale m is fully admitted. Per-sample
gradients of a Linear are rank one, so g_t . M = delta_t^T M x_t and |g_t| = |delta_t||x_t|
cost one forward pass: streamable at batch 1, batch-agnostic.

Arms: adam (uniform), oracle (true 1/sigma^2(x), upper bound), agree, agree_shuffle (same
weights, correspondence destroyed: previous step's c at B=1, permuted within batch else),
huber (classic residual-based robust weight, the known baseline). All weights are
normalized by a running per-sample EMA so mean plasticity is ~1 and no arm is a hidden LR.
Every arm at its own best LR from a bracketed grid; edge winners flagged. Seeds resample
teacher, stream, noise field and init. Metric: held-out MSE against the NOISE-FREE target.

    .venv/bin/python cleanrl/plasticity/sample_stream.py --hetero 2.0 --batch 1
    .venv/bin/python cleanrl/plasticity/sample_stream.py --hetero 0.0            # must be null
    .venv/bin/python cleanrl/plasticity/sample_stream.py --hetero 0 --tail-df 2  # heavy tails
"""
import argparse
import math

import torch

METHODS = ("adam", "oracle", "agree", "agree_shuffle", "huber",
           "hetvar", "hetvar_r2", "hetvar_t", "hetvar_ta", "hetvar_shuffle")
D_IN, H, D_OUT = 17, 64, 1


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--samples", type=int, default=65536)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr-grid", type=float, nargs="+",
                   default=[2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    p.add_argument("--hetero", type=float, default=2.0,
                   help="log-sigma half-range across input space (0 = homoscedastic)")
    p.add_argument("--noise", type=float, default=1.0, help="median noise sd")
    p.add_argument("--tail-df", type=float, default=0.0, help="Student-t df (0 = Gaussian)")
    p.add_argument("--switch-at", type=float, default=0.0, help="fraction; 0 = stationary")
    p.add_argument("--ema", type=float, default=0.999, help="per-sample EMA for c normalization")
    p.add_argument("--cap", type=float, default=20.0, help="cap on normalized c")
    p.add_argument("--huber-k", type=float, default=1.345)
    p.add_argument("--readout-lr", type=float, default=0.002,
                   help="per-SAMPLE normalized-LMS rate of the state readouts (summed over batch; "
                        "stable while batch * rate < 2)")
    p.add_argument("--readout-decay", type=float, default=0.0,
                   help="optional fixed shrinkage of the variance readout's non-bias coefficients, "
                        "per sample, in units of readout-lr (0 = twin calibration only)")
    p.add_argument("--nu", type=float, default=5.0,
                   help="Student-t df for hetvar_t: weight (nu+1)/(nu+z^2) on the standardized residual")
    p.add_argument("--var-floor", type=float, default=0.02,
                   help="floor on predicted variance, as a fraction of its running mean")
    p.add_argument("--test", type=int, default=4096)
    p.add_argument("--methods", default=",".join(METHODS))
    return p.parse_args()


def init_mlp(S, G, gen, device):
    def layer(o, i, std):
        # orthogonal init as layer_init in ppo_continuous_action.py (rows orthonormal
        # when o <= i, columns when o > i), scaled by std
        w = torch.randn(S, max(o, i), min(o, i), generator=gen, device=device)
        q, _ = torch.linalg.qr(w)                                   # (S, max, min)
        w = (q if o > i else q.transpose(-1, -2)) * std             # (S, o, i)
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


def teach(Ws, x):  # x (S,B,D_IN) -> (S,B,1)
    h = torch.tanh(torch.einsum("shi,sbi->sbh", Ws[0], x))
    h = torch.tanh(torch.einsum("shi,sbi->sbh", Ws[1], h))
    return torch.einsum("soh,sbh->sbo", Ws[2], h)


def forward(P, x):  # x (S,B,D_IN) shared over G
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
    y_te = teach(Ws2 if a.switch_at > 0 else Ws, x_te)  # noise-free, post-switch teacher
    P0 = init_mlp(S, G, gen, dev)
    xs = torch.randn(steps, S, B, D_IN, generator=gen, device=dev)
    # state-conditional noise field: log sigma = hetero * tanh(v . x)
    logsig = a.hetero * torch.tanh(torch.einsum("si,tsbi->tsb", v, xs))
    sigma = a.noise * torch.exp(logsig)
    if a.tail_df > 0:
        z = torch.randn(steps, S, B, generator=gen, device=dev)
        chi = torch.distributions.Chi2(a.tail_df).sample((steps, S, B)).to(dev)
        eps = z / torch.sqrt(chi / a.tail_df) * math.sqrt((a.tail_df - 2) / a.tail_df)
    else:
        eps = torch.randn(steps, S, B, generator=gen, device=dev)
    switch = int(steps * a.switch_at) if a.switch_at > 0 else steps + 1
    beta_step = a.ema ** B
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    results, levels = {}, {}
    for method in methods:
        P = [p.clone() for p in P0]
        M = [torch.zeros_like(p) for p in P]
        V = [torch.zeros_like(p) for p in P]
        c_ema = torch.ones(S, G, device=dev)
        c_prev = torch.ones(S, G, B, device=dev)
        r_scale = torch.ones(S, G, device=dev)
        lvl_sum = torch.zeros(S, G, device=dev)
        lvl_sq = torch.zeros(S, G, device=dev)
        # linear readouts from the sample's OWN last hidden state: E[r|x], log Var(r|x),
        # and a TWIN log-variance readout trained on MISPAIRED residuals (previous
        # sample's residual with this sample's state). The twin has zero true signal, so
        # the variance of its predicted deviation is the estimation-noise floor; the real
        # field is shrunk by the positive-part James-Stein factor 1 - V_twin/V_pred.
        u_mu = torch.zeros(S, G, H + 1, device=dev)
        u_lv = torch.zeros(S, G, H + 1, device=dev)
        u_tw = torch.zeros(S, G, H + 1, device=dev)
        v_pred = torch.zeros(S, G, device=dev)
        v_twin = torch.zeros(S, G, device=dev)
        var_ema = torch.ones(S, G, device=dev)
        ring = torch.ones(8192, S, G, device=dev)
        z2_ema = torch.ones(S, G, device=dev)
        z4_ema = 3.0 * torch.ones(S, G, device=dev)
        for t in range(steps):
            x = xs[t]
            y = teach(Ws2 if t >= switch else Ws, x) + (sigma[t] * eps[t]).unsqueeze(-1)
            W1, b1, W2, b2, W3, b3 = P
            h1, h2, out = forward(P, x)
            r = out - y.unsqueeze(1)                                   # (S,G,B,1)
            d3 = r
            d2 = torch.einsum("sgbo,sgoi->sgbi", d3, W3) * (1 - h2 * h2)
            d1 = torch.einsum("sgbo,sgoi->sgbi", d2, W2) * (1 - h1 * h1)
            xg = x.unsqueeze(1).expand(S, G, B, D_IN)
            deltas, inputs = (d1, d2, d3), (xg, h1, h2)

            with torch.no_grad():
                if method == "adam":
                    c_raw = torch.ones(S, G, B, device=dev)
                elif method == "oracle":
                    c_raw = (1.0 / sigma[t].square()).unsqueeze(1).expand(S, G, B)
                elif method in ("agree", "agree_shuffle"):
                    dot = torch.zeros(S, G, B, device=dev)
                    gsq = torch.zeros(S, G, B, device=dev)
                    msq = torch.zeros(S, G, device=dev)
                    for li, (d, inp) in enumerate(zip(deltas, inputs)):
                        Mw, Mb = M[2 * li], M[2 * li + 1]
                        dot += torch.einsum("sgbo,sgoi,sgbi->sgb", d, Mw, inp)
                        dot += torch.einsum("sgbo,sgo->sgb", d, Mb)
                        gsq += d.square().sum(-1) * (inp.square().sum(-1) + 1.0)
                        msq += Mw.square().sum((-1, -2)) + Mb.square().sum(-1)
                    cos2 = dot.square() / (gsq * msq.unsqueeze(-1)).clamp_min(1e-30)
                    cos2 = cos2.clamp(0.0, 1.0 - 1e-6)
                    c_raw = cos2 / (1.0 - cos2)
                    c_raw = torch.where(msq.unsqueeze(-1) > 0, c_raw, torch.ones_like(c_raw))
                elif method == "huber":
                    ar = r.squeeze(-1).abs()
                    r_scale = beta_step * r_scale + (1 - beta_step) * ar.mean(-1) * 1.4826
                    c_raw = (a.huber_k * r_scale.unsqueeze(-1) / ar.clamp_min(1e-12)).clamp_max(1.0)
                elif method.startswith("hetvar"):
                    phi = torch.cat((h2, torch.ones(S, G, B, 1, device=dev)), -1)   # (S,G,B,H+1)
                    rr = r.squeeze(-1)
                    p_mu = torch.einsum("sgbk,sgk->sgb", phi, u_mu)
                    dev_lv = torch.einsum("sgbk,sgk->sgb", phi[..., :-1], u_lv[..., :-1])
                    dev_tw = torch.einsum("sgbk,sgk->sgb", phi[..., :-1], u_tw[..., :-1])
                    v_pred = beta_step * v_pred + (1 - beta_step) * dev_lv.square().mean(-1)
                    v_twin = beta_step * v_twin + (1 - beta_step) * dev_tw.square().mean(-1)
                    shrink = (1.0 - v_twin / v_pred.clamp_min(1e-12)).clamp(0.0, 1.0)
                    p_lv = (u_lv[..., -1:] + shrink.unsqueeze(-1) * dev_lv).clamp(-12.0, 12.0)
                    centred = rr if method == "hetvar_r2" else rr - p_mu
                    var_hat = p_lv.exp()
                    var_ema = beta_step * var_ema + (1 - beta_step) * var_hat.mean(-1)
                    c_raw = 1.0 / var_hat.clamp_min(a.var_floor * var_ema.unsqueeze(-1))
                    if method in ("hetvar_t", "hetvar_ta"):
                        z2 = centred.square() / var_hat
                        nu = torch.full_like(z2, a.nu)
                        if method == "hetvar_ta":
                            # nu from the kurtosis of the standardized residual:
                            # kappa = 3(nu-2)/(nu-4)  =>  nu = (4 kappa - 6)/(kappa - 3); Gaussian -> inf
                            kappa = z4_ema / z2_ema.square().clamp_min(1e-12)
                            nu_hat = (4.0 * kappa - 6.0) / (kappa - 3.0).clamp_min(1e-3)
                            nu = torch.where(kappa > 3.05, nu_hat, torch.full_like(kappa, 1e6)).unsqueeze(-1)
                            z2c = z2.clamp_max(1e4)
                            z2_ema = beta_step * z2_ema + (1 - beta_step) * z2c.mean(-1)
                            z4_ema = beta_step * z4_ema + (1 - beta_step) * z2c.square().mean(-1)
                        c_raw = c_raw * (nu + 1.0) / (nu + z2)
                    # normalized LMS, summed over the batch (per-sample rate, batch-invariant)
                    phin = phi / phi.square().sum(-1, keepdim=True)
                    e_mu = p_mu - rr
                    c2 = centred.square()
                    raw_lv = torch.einsum("sgbk,sgk->sgb", phi, u_lv).clamp(-12.0, 12.0)
                    raw_tw = torch.einsum("sgbk,sgk->sgb", phi, u_tw).clamp(-12.0, 12.0)
                    e_lv = (1.0 - c2 / raw_lv.exp()).clamp(-20.0, 1.0)       # d/d(logvar) of Gaussian NLL
                    # mispaired residual^2 from a RANDOM past sample: a lag-1 pairing is only a
                    # null for iid streams; on autocorrelated data (time series, RL episodes)
                    # the previous sample's noise is predictable from this state and the twin
                    # would learn the real field and shrink it away
                    idx = torch.randint(0, ring.shape[0], (B,), device=dev)
                    mis = ring[idx].permute(1, 2, 0)                                 # (S,G,B)
                    e_tw = (1.0 - mis / raw_tw.exp()).clamp(-20.0, 1.0)
                    ring[(t * B + torch.arange(B, device=dev)) % ring.shape[0]] = c2.permute(2, 0, 1)
                    u_mu -= a.readout_lr * torch.einsum("sgb,sgbk->sgk", e_mu, phin)
                    u_lv -= a.readout_lr * torch.einsum("sgb,sgbk->sgk", e_lv, phin)
                    u_tw -= a.readout_lr * torch.einsum("sgb,sgbk->sgk", e_tw, phin)
                    if a.readout_decay > 0:
                        u_lv[..., :-1] *= (1.0 - a.readout_lr * a.readout_decay) ** B
                else:
                    raise ValueError(method)
                c_ema = beta_step * c_ema + (1 - beta_step) * c_raw.mean(-1)
                c = (c_raw / c_ema.unsqueeze(-1).clamp_min(1e-12)).clamp_max(a.cap)
                if method in ("agree_shuffle", "hetvar_shuffle"):
                    if B == 1:
                        c, c_prev = c_prev, c
                    else:
                        c = c[..., torch.randperm(B, device=dev)]
                if method == "adam":
                    c = c_raw
                lvl_sum += c.mean(-1)
                lvl_sq += c.square().mean(-1)

                grads = []
                for d, inp in zip(deltas, inputs):
                    cd = c.unsqueeze(-1) * d
                    grads.append(torch.einsum("sgbo,sgbi->sgoi", cd, inp) / B)
                    grads.append(cd.sum(2) / B)
                bc1 = 1 - beta1 ** (t + 1)
                bc2 = 1 - beta2 ** (t + 1)
                for p, m, vv, g in zip(P, M, V, grads):
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    vv.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    step = (m / bc1) / ((vv / bc2).sqrt() + adam_eps)
                    p.sub_((lr if p.dim() == 4 else lr_b) * step)
        with torch.no_grad():
            _, _, out_te = forward(P, x_te)
            mse = (out_te - y_te.unsqueeze(1)).square().mean((-1, -2))  # (S,G)
        results[method] = mse
        levels[method] = (lvl_sum / steps, (lvl_sq / steps - (lvl_sum / steps) ** 2).clamp_min(0).sqrt())

    base = (y_te.square().mean((-1, -2))).mean().item()
    print(f"# hetero={a.hetero} noise={a.noise} tail_df={a.tail_df} switch={a.switch_at} "
          f"batch={B} samples={a.samples} seeds={S}  zero-predictor mse {base:.4f}")
    print(f"{'method':14s} {'best_lr':>8s} {'mse':>8s} {'sem':>7s}   {'c_mean':>6s} {'c_sd':>6s}  "
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
        print(f"{method:14s} {a.lr_grid[gi]:8.0e} {mean_over_seeds[gi].item():8.4f} {sem:7.4f}   "
              f"{lm[:, gi].mean().item():6.3f} {ls[:, gi].mean().item():6.3f}  "
              + " ".join(f"{m:8.4f}" for m in mean_over_seeds.tolist()) + edge)
    if "adam" in best:
        print("# paired vs adam (best LR each):  mean diff  sem  t")
        for method in methods:
            if method == "adam":
                continue
            diff = best[method] - best["adam"]
            sem = diff.std().item() / math.sqrt(S)
            print(f"  {method:14s} {diff.mean().item():+8.4f} {sem:7.4f} {diff.mean().item() / max(sem, 1e-12):+6.2f}")


if __name__ == "__main__":
    main()

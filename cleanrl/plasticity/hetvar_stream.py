"""Per-SAMPLE plasticity on a standard dense net: each sample's own state decides the
precision of its gradient. Reference implementation of `hetvar` + the diagnostic that
certifies it (valid oracle bound, exact null cell, bracketed LR grids, paired seeds).

WHY PER-SAMPLE. g_t = gbar + eps_t with Var(eps_t) = sigma_t^2 I: the update sum_t c_t g_t
has maximal expected second-order gain at c_t ~ 1/sigma_t^2 (Gauss-Markov). The gain over
uniform is mean(sigma^2) * mean(1/sigma^2) >= 1, with equality iff homoscedastic, so the
homoscedastic cell MUST be null for every arm and heteroscedastic / heavy-tailed cells are
where any headroom lives.

WHY NOT GRADIENT AGREEMENT. For a scalar output g_t = r_t * grad f(x_t): noise and signal
are collinear within a sample, so cos(g_t, m) depends on x_t only and carries nothing about
sigma_t (measured: `agree` == its own shuffle). Only the residual magnitude relative to
what THIS state normally produces can carry precision.

THE MECHANISM (`hetvar_ta`, no tuned knobs):
  * two linear readouts from the sample's own last hidden state: E[r|x] and log Var(r|x)
    (Gaussian NLL, normalized LMS so batch * rate < 2 is stable and per-sample movement is
    batch-invariant);
  * a TWIN log-variance readout trained on MISPAIRED residuals (a random past sample's
    residual^2 with this state) has zero true signal, so the variance of its predicted
    deviation is the estimation-noise floor: the real field is shrunk by the positive-part
    James-Stein factor 1 - V_twin/V_pred. This is what makes the homoscedastic cell exactly
    null without a decay knob. The mispairing MUST be random-past, not lag-1: on
    autocorrelated streams (time series, RL episodes) the previous sample's noise is
    predictable from this state and a lag-1 twin learns the real field and shrinks it away;
  * heavy tails: weight (nu+1)/(nu+z^2) on the standardized residual, nu from the residual
    kurtosis kappa = 3(nu-2)/(nu-4) (Gaussian -> nu = inf, factor 1);
  * level: c = c_raw / EMA(c_raw), so mean plasticity is ~1 and no arm is a hidden LR.
The weight multiplies the sample's gradient BEFORE the optimizer (mean ~1 so Adam does not
cancel it; the per-sample variation changes the update direction, not just its size).

MEASURED (8 paired seeds, own best LR each, held-out MSE vs the noise-free target, batch 1
unless stated; oracle = true 1/sigma^2(x)):
  homoscedastic        oracle  0.000  hetvar -0.0001  hetvar_ta -0.0003  shuffle  0.000
  hetero 2             oracle -0.133  hetvar -0.131                       shuffle +0.025
  hetero 2 + t(2.5)    oracle -0.113  hetvar -0.120   hetvar_ta -0.153   shuffle +0.028
  teacher switch @50%  oracle -0.093  hetvar -0.091                       shuffle +0.042
  low noise (bias-led) oracle +0.039  hetvar -0.022   (1/sigma^2 is the WRONG target here;
                                                        total E[r^2|x] is right)
  batch 32             oracle -0.099  hetvar -0.115                       shuffle +0.039
  batch 512            oracle -0.103  hetvar -0.107   hetvar_ta -0.103   shuffle +0.091
Destroying the state->sample correspondence (shuffle) turns every gain into a loss.

    .venv/bin/python cleanrl/plasticity/hetvar_stream.py --hetero 2.0 --batch 1
    .venv/bin/python cleanrl/plasticity/hetvar_stream.py --hetero 0.0            # must be null
    .venv/bin/python cleanrl/plasticity/hetvar_stream.py --hetero 2 --tail-df 2.5
    .venv/bin/python cleanrl/plasticity/hetvar_stream.py --hetero 2 --batch 512 --samples 131072 \
        --lr-grid 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1
"""
import argparse
import math

import torch

METHODS = ("adam", "oracle", "oracle_snr", "oracle_wiener", "oracle_agree", "agree", "agree_shuffle", "huber",
           "hetvar", "hetvar_r2", "hetvar_t", "hetvar_ta", "hetvar_shuffle",
           "snr", "snr_shuffle", "wiener", "wiener_shuffle",
           "ntk", "ntk_shuffle", "ntk_wiener", "ntk_wiener_shuffle", "mom", "mom_shuffle",
           "bins", "bins_shuffle", "bins1", "bins1_shuffle")
HET = ("hetvar", "hetvar_r2", "hetvar_t", "hetvar_ta", "snr", "wiener", "ntk", "ntk_wiener", "mom")
PERPARAM = ("bins", "bins1")
AUTONOMOUS = ("adam", "wiener", "oracle_wiener", "oracle_agree", "ntk_wiener", "mom") + PERPARAM
D_IN, H, D_OUT = 17, 64, 1


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--samples", type=int, default=65536)
    p.add_argument("--signal-scales", action="store_true",
                   help="the noise-free target is multiplied by the same field as the noise sd "
                        "(signal ~ sigma(x), SNR constant; the SPY situation)")
    p.add_argument("--null", action="store_true",
                   help="noise-free target is 0: nothing learnable; MSE = absorbed noise")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr-grid", type=float, nargs="+",
                   default=[2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    p.add_argument("--hetero", type=float, default=2.0,
                   help="log-sigma half-range across input space (0 = homoscedastic)")
    p.add_argument("--noise", type=float, default=1.0, help="median noise sd")
    p.add_argument("--tail-df", type=float, default=0.0, help="Student-t df (0 = Gaussian)")
    p.add_argument("--switch-at", type=float, default=0.0, help="fraction; 0 = stationary")
    p.add_argument("--ntk-beta", type=float, default=0.99,
                   help="per-sample EMA of the gradient-kernel regression buffers (ntk arms)")
    p.add_argument("--bins", type=int, default=8, help="bins over a parameter's own input (bins arm)")
    p.add_argument("--bin-decay", type=float, default=0.0,
                   help="per-sample discount of the per-bin sums (0 = plain sums, horizonless)")
    p.add_argument("--gate-post", action="store_true",
                   help="per-param arms: Adam runs on the ungated gradient and the gate multiplies "
                        "the step (batch 1 only); avoids 1/sqrt(v) renormalizing a small gate away")
    p.add_argument("--ema", type=float, default=0.999, help="per-sample EMA for level/normalizers")
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
    p.add_argument("--ring", type=int, default=8192, help="past residuals the twin mispairs against")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--test", type=int, default=4096)
    return p.parse_args()


def init_mlp(S, G, gen, device):
    def layer(o, i, std):
        # orthogonal init as layer_init in ppo_continuous_action.py, scaled by std
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
    logsig_te = a.hetero * torch.tanh(torch.einsum("si,sbi->sb", v, x_te))
    amp_te = torch.exp(logsig_te) if a.signal_scales else torch.ones_like(logsig_te)
    if a.null:
        amp_te = torch.zeros_like(amp_te)
    y_te = teach(Ws2 if a.switch_at > 0 else Ws, x_te) * amp_te.unsqueeze(-1)  # noise-free, post-switch
    P0 = init_mlp(S, G, gen, dev)
    xs = torch.randn(steps, S, B, D_IN, generator=gen, device=dev)
    # state-conditional noise field: log sigma = hetero * tanh(v . x)
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
    beta_step = a.ema ** B
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    results, levels = {}, {}
    for method in methods:
        base = method[:-len("_shuffle")] if method.endswith("_shuffle") else method
        P = [p.clone() for p in P0]
        M = [torch.zeros_like(p) for p in P]
        V = [torch.zeros_like(p) for p in P]
        # per-PARAMETER target predictability from the parameter's own state. Weight w_ji:
        # state = its input a_i (binned), target = the error delta_j it is asked to fit. Bias:
        # state = the unit's own output. Sufficient statistics per bin are plain sums (n, S1, S2);
        # E[S1^2] = n^2 mu^2 + n var  =>  predictable fraction  p = (S1^2 - S2)^+ / (n S2),
        # 0 when the input value says nothing about the error, (n-1)/n when it fixes it.
        # No twin, no EMA: the calibration is analytic. Stats are read before the sample is added.
        K = a.bins if base == "bins" else 1
        qn = torch.erfinv(torch.arange(1, K, device=dev) / K * 2.0 - 1.0) * math.sqrt(2.0) if K > 1 else torch.zeros(0, device=dev)
        edges = (qn, torch.tanh(math.sqrt(2.0) * qn), torch.tanh(math.sqrt(2.0) * qn), 2.0 * qn)
        dims = ((D_IN, H), (H, H), (H, D_OUT))
        bn = [torch.zeros(S, G, i, K, device=dev) for i, _ in dims]
        bs1 = [torch.zeros(S, G, o, i, K, device=dev) for i, o in dims]
        bs2 = [torch.zeros(S, G, o, i, K, device=dev) for i, o in dims]
        un = [torch.zeros(S, G, o, K, device=dev) for _, o in dims]
        us1 = [torch.zeros(S, G, o, K, device=dev) for _, o in dims]
        us2 = [torch.zeros(S, G, o, K, device=dev) for _, o in dims]
        pp_prev = None
        c_ema = torch.ones(S, G, device=dev)
        c_prev = torch.ones(S, G, B, device=dev)
        r_scale = torch.ones(S, G, device=dev)
        lvl_sum = torch.zeros(S, G, device=dev)
        lvl_sq = torch.zeros(S, G, device=dev)
        u_mu = torch.zeros(S, G, H + 1, device=dev)
        u_lv = torch.zeros(S, G, H + 1, device=dev)
        u_tw = torch.zeros(S, G, H + 1, device=dev)
        v_pred = torch.zeros(S, G, device=dev)
        v_twin = torch.zeros(S, G, device=dev)
        var_ema = torch.ones(S, G, device=dev)
        ring = torch.ones(a.ring, S, G, device=dev)
        ring_r = torch.zeros(a.ring, S, G, device=dev)
        u_mt = torch.zeros(S, G, H + 1, device=dev)
        v_mu = torch.zeros(S, G, device=dev)
        v_mt = torch.zeros(S, G, device=dev)
        # ntk: Nadaraya-Watson estimate of E[r|x_t] under the model's own gradient kernel,
        # K(s,t) = grad f(x_s).grad f(x_t): numerator EMA(r_s grad f_s), denominator EMA(grad f_s),
        # twin EMA(eps_s r_s grad f_s) (random signs) = the estimator's noise floor. Buffers exclude
        # the current sample, so the estimate is held-out.
        Mr = [torch.zeros_like(p) for p in P]
        Mtw = [torch.zeros_like(p) for p in P]
        Mf = [torch.zeros_like(p) for p in P]
        beta_ntk = a.ntk_beta ** B
        z2_ema = torch.ones(S, G, device=dev)
        z4_ema = 3.0 * torch.ones(S, G, device=dev)
        for t in range(steps):
            x = xs[t]
            y_clean = teach(Ws2 if t >= switch else Ws, x) * amp[t].unsqueeze(-1)
            y = y_clean + (sigma[t] * eps[t]).unsqueeze(-1)
            W1, b1, W2, b2, W3, b3 = P
            h1, h2, out = forward(P, x)
            r = out - y.unsqueeze(1)                                   # (S,G,B,1)
            b_true = (out - y_clean.unsqueeze(1)).squeeze(-1)                 # learnable residual (S,G,B)
            d3 = r
            d2 = torch.einsum("sgbo,sgoi->sgbi", d3, W3) * (1 - h2 * h2)
            d1 = torch.einsum("sgbo,sgoi->sgbi", d2, W2) * (1 - h1 * h1)
            xg = x.unsqueeze(1).expand(S, G, B, D_IN)
            deltas, inputs = (d1, d2, d3), (xg, h1, h2)

            grads_pp = None
            with torch.no_grad():
                if base == "adam":
                    c_raw = torch.ones(S, G, B, device=dev)
                elif base == "oracle":
                    c_raw = (1.0 / sigma[t].square()).unsqueeze(1).expand(S, G, B)
                elif base == "oracle_snr":
                    # one-step optimum: c_t ~ |b_t| / sigma_t^2 (weight the learnable residual, not 1/var)
                    c_raw = b_true.abs() / sigma[t].square().unsqueeze(1)
                elif base == "oracle_wiener":
                    b2 = b_true.square()
                    c_raw = b2 / (b2 + sigma[t].square().unsqueeze(1))
                elif base == "oracle_agree":
                    # sign-agreement form: does the OBSERVED residual agree with the learnable one?
                    # noise sample (b=0) -> 0.5 on average; learnable -> 1. A coin-flip version of wiener.
                    c_raw = torch.sigmoid(r.squeeze(-1) * b_true / sigma[t].square().unsqueeze(1))
                elif base == "agree":
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
                elif base == "huber":
                    ar = r.squeeze(-1).abs()
                    r_scale = beta_step * r_scale + (1 - beta_step) * ar.mean(-1) * 1.4826
                    c_raw = (a.huber_k * r_scale.unsqueeze(-1) / ar.clamp_min(1e-12)).clamp_max(1.0)
                elif base in HET:
                    phi = torch.cat((h2, torch.ones(S, G, B, 1, device=dev)), -1)   # (S,G,B,H+1)
                    rr = r.squeeze(-1)
                    p_mu = torch.einsum("sgbk,sgk->sgb", phi, u_mu)
                    dev_mu = torch.einsum("sgbk,sgk->sgb", phi[..., :-1], u_mu[..., :-1])
                    dev_mt = torch.einsum("sgbk,sgk->sgb", phi[..., :-1], u_mt[..., :-1])
                    dev_lv = torch.einsum("sgbk,sgk->sgb", phi[..., :-1], u_lv[..., :-1])
                    dev_tw = torch.einsum("sgbk,sgk->sgb", phi[..., :-1], u_tw[..., :-1])
                    v_mu = beta_step * v_mu + (1 - beta_step) * dev_mu.square().mean(-1)
                    v_mt = beta_step * v_mt + (1 - beta_step) * dev_mt.square().mean(-1)
                    v_pred = beta_step * v_pred + (1 - beta_step) * dev_lv.square().mean(-1)
                    v_twin = beta_step * v_twin + (1 - beta_step) * dev_tw.square().mean(-1)
                    shrink_mu = (1.0 - v_mt / v_mu.clamp_min(1e-12)).clamp(0.0, 1.0)
                    shrink = (1.0 - v_twin / v_pred.clamp_min(1e-12)).clamp(0.0, 1.0)
                    # twin-shrunk estimate of the LEARNABLE residual at this state
                    b_hat = u_mu[..., -1:] + shrink_mu.unsqueeze(-1) * dev_mu
                    p_lv = (u_lv[..., -1:] + shrink.unsqueeze(-1) * dev_lv).clamp(-12.0, 12.0)
                    centred = rr if base == "hetvar_r2" else rr - p_mu
                    var_hat = p_lv.exp()
                    var_ema = beta_step * var_ema + (1 - beta_step) * var_hat.mean(-1)
                    prec = 1.0 / var_hat.clamp_min(a.var_floor * var_ema.unsqueeze(-1))
                    if base in ("ntk", "ntk_wiener", "mom"):
                        f3 = torch.ones_like(d3)
                        f2 = torch.einsum("sgbo,sgoi->sgbi", f3, W3) * (1 - h2 * h2)
                        f1 = torch.einsum("sgbo,sgoi->sgbi", f2, W2) * (1 - h1 * h1)
                        fds = (f1, f2, f3)
                        proj_r = torch.zeros(S, G, B, device=dev)
                        proj_tw = torch.zeros(S, G, B, device=dev)
                        denom = torch.zeros(S, G, B, device=dev)
                        for li, (f, inp) in enumerate(zip(fds, inputs)):
                            for buf, acc in ((Mr, proj_r), (Mtw, proj_tw), (Mf, denom)):
                                acc += torch.einsum("sgbo,sgoi,sgbi->sgb", f, buf[2 * li], inp)
                                acc += torch.einsum("sgbo,sgo->sgb", f, buf[2 * li + 1])
                        bc = 1.0 - beta_ntk ** (t + 1) if t > 0 else 1.0
                        denom = (denom / bc).clamp_min(0.25)      # last-layer bias contributes exactly 1
                        b_ntk = proj_r / bc / denom
                        b_flr = proj_tw / bc / denom
                        b2 = (b_ntk.square() - b_flr.square()).clamp_min(0.0)
                        if base == "ntk":
                            c_raw = b2.sqrt() * prec
                        elif base == "mom":
                            # does this sample's residual agree with what the accumulated gradient
                            # predicts for its state: sign(r_t) * <grad f_t, m>. Equivalent to the
                            # predicted change of THIS sample's loss under the momentum step.
                            c_raw = torch.sigmoid(rr * b_ntk * prec)
                        else:
                            c_raw = b2 / (b2 + var_hat)
                        sgn = torch.randint(0, 2, (S, G, B), device=dev).float() * 2.0 - 1.0
                        for li, (f, inp) in enumerate(zip(fds, inputs)):
                            for buf, wgt in ((Mr, rr), (Mtw, sgn * rr), (Mf, torch.ones_like(rr))):
                                wf = wgt.unsqueeze(-1) * f
                                buf[2 * li].mul_(beta_ntk).add_(torch.einsum("sgbo,sgbi->sgoi", wf, inp), alpha=(1 - beta_ntk) / B)
                                buf[2 * li + 1].mul_(beta_ntk).add_(wf.sum(2), alpha=(1 - beta_ntk) / B)
                    elif base == "snr":
                        c_raw = b_hat.abs() * prec                       # |b|/sigma^2
                    elif base == "wiener":
                        s2 = b_hat.square()
                        c_raw = s2 / (s2 + var_hat)                      # predictable fraction of E[r^2|x]
                    else:
                        c_raw = prec
                        if base in ("hetvar_t", "hetvar_ta"):
                            z2 = centred.square() / var_hat
                            nu = torch.full_like(z2, a.nu)
                            if base == "hetvar_ta":
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
                    idx = torch.randint(0, ring.shape[0], (B,), device=dev)
                    mis = ring[idx].permute(1, 2, 0)                         # random past residual^2
                    mis_r = ring_r[idx].permute(1, 2, 0)                     # random past residual
                    e_tw = (1.0 - mis / raw_tw.exp()).clamp(-20.0, 1.0)
                    e_mt = torch.einsum("sgbk,sgk->sgb", phi, u_mt) - mis_r
                    slot = (t * B + torch.arange(B, device=dev)) % ring.shape[0]
                    ring[slot] = c2.permute(2, 0, 1)
                    ring_r[slot] = rr.permute(2, 0, 1)
                    u_mu -= a.readout_lr * torch.einsum("sgb,sgbk->sgk", e_mu, phin)
                    u_mt -= a.readout_lr * torch.einsum("sgb,sgbk->sgk", e_mt, phin)
                    u_lv -= a.readout_lr * torch.einsum("sgb,sgbk->sgk", e_lv, phin)
                    u_tw -= a.readout_lr * torch.einsum("sgb,sgbk->sgk", e_tw, phin)
                    if a.readout_decay > 0:
                        u_lv[..., :-1] *= (1.0 - a.readout_lr * a.readout_decay) ** B
                elif base in PERPARAM:
                    if a.bin_decay > 0:
                        for buf in (bn, bs1, bs2, un, us1, us2):
                            for z in buf:
                                z.mul_(1.0 - a.bin_decay)
                    grads_pp = []
                    pp = []
                    c_raw = torch.zeros(S, G, B, device=dev)
                    states = (h1, h2, out)
                    for li, (d, inp) in enumerate(zip(deltas, inputs)):
                        gw = torch.zeros_like(P[2 * li])
                        gb = torch.zeros_like(P[2 * li + 1])
                        o, i = gw.shape[-2], gw.shape[-1]
                        for b in range(B):
                            ib = torch.bucketize(inp[:, :, b], edges[li]).unsqueeze(-1)          # (S,G,i,1)
                            ub = torch.bucketize(states[li][:, :, b], edges[li + 1]).unsqueeze(-1)  # (S,G,o,1)
                            ibw = ib.unsqueeze(2).expand(S, G, o, i, 1)
                            n_w = bn[li].gather(-1, ib).unsqueeze(2)                          # (S,G,1,i,1)
                            s1_w = bs1[li].gather(-1, ibw)
                            s2_w = bs2[li].gather(-1, ibw)
                            p_w = ((s1_w.square() - s2_w).clamp_min(0.0) / (n_w * s2_w + 1e-12)).squeeze(-1)
                            n_u = un[li].gather(-1, ub)
                            s1_u = us1[li].gather(-1, ub)
                            s2_u = us2[li].gather(-1, ub)
                            p_u = ((s1_u.square() - s2_u).clamp_min(0.0) / (n_u * s2_u + 1e-12)).squeeze(-1)
                            db = d[:, :, b]                                                    # (S,G,o)
                            # add the sample to its bins (after reading: held-out statistic)
                            bn[li].scatter_add_(-1, ib, torch.ones_like(ib, dtype=torch.float))
                            bs1[li].scatter_add_(-1, ibw, db.unsqueeze(-1).unsqueeze(-1).expand(S, G, o, i, 1))
                            bs2[li].scatter_add_(-1, ibw, db.square().unsqueeze(-1).unsqueeze(-1).expand(S, G, o, i, 1))
                            un[li].scatter_add_(-1, ub, torch.ones_like(ub, dtype=torch.float))
                            us1[li].scatter_add_(-1, ub, db.unsqueeze(-1))
                            us2[li].scatter_add_(-1, ub, db.square().unsqueeze(-1))
                            if method.endswith("_shuffle") and pp_prev is not None:
                                p_w, p_u, pp_prev[2 * li], pp_prev[2 * li + 1] = pp_prev[2 * li], pp_prev[2 * li + 1], p_w, p_u
                            pp.extend((p_w, p_u))
                            gw += p_w * db.unsqueeze(-1) * inp[:, :, b].unsqueeze(-2) / B
                            gb += p_u * db / B
                            c_raw[:, :, b] += (p_w.mean((-1, -2)) * (o * i) + p_u.mean(-1) * o) / (o * i + o) / 3
                        grads_pp.extend((gw, gb))
                    if method.endswith("_shuffle") and pp_prev is None:
                        pp_prev = pp
                else:
                    raise ValueError(method)
                c_ema = beta_step * c_ema + (1 - beta_step) * c_raw.mean(-1)
                if base in AUTONOMOUS:
                    c = c_raw
                else:
                    c = (c_raw / c_ema.unsqueeze(-1).clamp_min(1e-12)).clamp_max(a.cap)
                if method.endswith("_shuffle") and grads_pp is None:
                    if B == 1:
                        c, c_prev = c_prev, c
                    else:
                        c = c[..., torch.randperm(B, device=dev)]
                lvl_sum += c.mean(-1)
                lvl_sq += c.square().mean(-1)

                gate_post = None
                if grads_pp is not None and a.gate_post:
                    assert B == 1, "post-optimizer per-param gating is defined at batch 1"
                    gate_post = pp
                    grads_pp = None
                if grads_pp is not None:
                    grads = grads_pp
                else:
                    grads = []
                    for d, inp in zip(deltas, inputs):
                        cd = (c.unsqueeze(-1) * d) if gate_post is None else d
                        grads.append(torch.einsum("sgbo,sgbi->sgoi", cd, inp) / B)
                        grads.append(cd.sum(2) / B)
                bc1 = 1 - beta1 ** (t + 1)
                bc2 = 1 - beta2 ** (t + 1)
                for k, (p, m, vv, g) in enumerate(zip(P, M, V, grads)):
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    vv.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    step = (m / bc1) / ((vv / bc2).sqrt() + adam_eps)
                    if gate_post is not None:
                        step = step * gate_post[k]
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

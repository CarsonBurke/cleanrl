"""RethinkC: per-PERCEPTRON plasticity from the unit's own goal regression (report: RethinkC.md).

Each perceptron j sees, on the current sample, its input vector a (its state), its pre-activation
z_j and the error delta_j = dL/dz_j. Its own update -lr * delta_j * a is a stochastic linear
regression of a target on its state, so "how predictable is the target from the state" has an
exact per-unit reading. The regressed target is the unit's GOAL y_j = z_j - delta_j / mean(J^2)
(one Gauss-Newton step; literally the label for the output unit): unlike delta_j it does not move
when the unit learns, so horizonless plain sums stay valid. Two exact RLS readouts per unit from
[1, a] (one Gram inverse per layer):
    yhat_j(a)   the goal;  fixable power sig_j = mean((z_j - yhat_j)^2 - se^2), se^2 = OLS leverage
    s_j^2(a)    log-linear variance of the goal residual, James-Stein-shrunk (q slopes explain
                q s^2 under the null) so a homoscedastic stream collapses to a constant
    gate c_j(a) = sig_j / (sig_j + s_j^2(a))         (`pp_var`, the mechanism)
`pp` uses the per-state fixable power (z_j - yhat_j(a))^2 - se^2 instead of its mean (falsified:
it gates on estimation noise). The gate multiplies delta_j (row of W and bias) before Adam.
Arms: adam, oracle_wiener, pp_var, pp, pp1 (regress delta itself), pp_mu, pp_raw, pp_norm,
pp_post, each with a _shuffle control (previous sample's gates applied to this sample).

    .venv/bin/python cleanrl/plasticity/rethink/RethinkC.py --hetero 2
"""
import argparse
import math

import torch

D_IN, H, D_OUT = 17, 64, 1
LOG_CHI2_MEAN = -1.2703628454614782  # E[log chi2_1] = digamma(1/2) + log 2
BASES = ("adam", "oracle_wiener", "pp", "pp1", "pp_post", "pp_mu", "pp_var", "pp_raw", "pp_norm")
ALL = ("adam", "oracle_wiener", "pp_var", "pp_var_shuffle", "pp", "pp_shuffle")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--samples", type=int, default=32768)
    p.add_argument("--signal-scales", action="store_true")
    p.add_argument("--null", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr-grid", type=float, nargs="+",
                   default=[2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    p.add_argument("--hetero", type=float, default=2.0)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--tail-df", type=float, default=0.0)
    p.add_argument("--switch-at", type=float, default=0.0)
    p.add_argument("--forget", type=float, default=1.0,
                   help="RLS forgetting factor per sample (1 = plain sums, horizonless)")
    p.add_argument("--methods", default=",".join(ALL))
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
    z1 = torch.einsum("sgoi,sbi->sgbo", W1, x) + b1.unsqueeze(2)
    h1 = torch.tanh(z1)
    z2 = torch.einsum("sgoi,sgbi->sgbo", W2, h1) + b2.unsqueeze(2)
    h2 = torch.tanh(z2)
    out = torch.einsum("sgoi,sgbi->sgbo", W3, h2) + b3.unsqueeze(2)
    return h1, h2, out, (z1, z2, out)


class LayerReadout:
    """Exact (recursive) least squares of each perceptron's target on its own input [1, a].
    One Gram inverse per layer (features are shared by the layer's perceptrons); per
    perceptron only the cross sums. Two regressions per perceptron: the target (mean model)
    and log e^2 of its residual (log-variance model). Sums are plain (forget=1), float64.

    target="goal" (the mechanism): the regressed target is the unit's GOAL pre-activation
        y_j = z_j - delta_j (for the output unit this is literally the label). The fixable
        error at this state is mu = z_j - E[y_j | a], read against the unit's CURRENT
        pre-activation, so the unit's own learning progress never makes the readout stale.
    target="delta" (falsified first form): regress delta_j itself; mu = E[delta_j | a]."""

    def __init__(self, S, G, n_in, n_out, dev, forget, target):
        self.p, self.lam, self.target = n_in + 1, forget, target
        self.Pinv = torch.eye(self.p, device=dev, dtype=torch.float64).expand(S, G, self.p, self.p).clone()
        self.n = torch.zeros(S, G, 1, device=dev, dtype=torch.float64)
        z = lambda: torch.zeros(S, G, n_out, device=dev, dtype=torch.float64)
        self.b_mu, self.b_lv = [torch.zeros(S, G, n_out, self.p, device=dev, dtype=torch.float64) for _ in range(2)]
        self.sy_mu, self.sy2_mu, self.sy_lv, self.sy2_lv, self.sig_run, self.jsq = z(), z(), z(), z(), z(), z()

    @staticmethod
    def feats(a):
        return torch.cat([torch.ones_like(a[..., :1]), a], -1).double()

    def _fit(self, b, sy, sy2):
        # OLS identities: sum yhat^2 = w.b, sum yhat = sum y  =>  SS_exp, RSS from the sums alone
        w = torch.einsum("sgpq,sgoq->sgop", self.Pinv, b)
        fit = (w * b).sum(-1)
        ybar = sy / self.n.clamp_min(1)
        ss_exp = (fit - self.n * ybar.square()).clamp_min(0)
        rss = (sy2 - fit).clamp_min(0)
        dof = self.n - self.p
        s2 = rss / dof.clamp_min(1)
        # positive-part James-Stein on the fitted deviations: under the null each of the p-1
        # slopes explains s2 in expectation
        k = ((ss_exp - (self.p - 1) * s2) / ss_exp.clamp_min(1e-300)).clamp(0, 1)
        k = torch.where(dof > 0, k, torch.zeros_like(k))
        return w, ybar, k, s2, ss_exp

    def predict(self, phi, z, mode):
        """phi (S,G,B,p), z (S,G,B,n_out) current pre-activation -> gate (S,G,B,n_out), held-out
        prediction of the regressed target."""
        w_mu, ybar_mu, k_mu, s2_mu, ss_mu = self._fit(self.b_mu, self.sy_mu, self.sy2_mu)
        w_lv, ybar_lv, k_lv, _, _ = self._fit(self.b_lv, self.sy_lv, self.sy2_lv)
        yhat = torch.einsum("sgbp,sgop->sgbo", phi, w_mu)
        lv_hat = torch.einsum("sgbp,sgop->sgbo", phi, w_lv)
        if mode == "raw":
            k_lv = torch.ones_like(k_lv)
        lv_s = ybar_lv.unsqueeze(2) + k_lv.unsqueeze(2) * (lv_hat - ybar_lv.unsqueeze(2))
        s2 = torch.exp(lv_s - LOG_CHI2_MEAN)
        if self.target == "goal":
            mu = z.double() - yhat
            # estimation noise of the readout at this state: s2 * leverage (classical OLS se^2)
            lev = torch.einsum("sgbp,sgpq,sgbq->sgb", phi, self.Pinv, phi).unsqueeze(-1)
            se2 = s2_mu.unsqueeze(2) * lev
            sig = mu.square() if mode == "raw" else (mu.square() - se2).clamp_min(0)
        else:
            if mode == "raw":
                k_mu = torch.ones_like(k_mu)
            mu = ybar_mu.unsqueeze(2) + k_mu.unsqueeze(2) * (yhat - ybar_mu.unsqueeze(2))
            sig = mu.square()
        self.sig_run += sig.mean(2)
        if mode == "mu":  # ablation: no conditional variance, global residual variance
            s2 = s2_mu.unsqueeze(2).expand_as(s2)
        elif mode == "var":  # ablation: no conditional mean, average signal power
            sig = (self.sig_run / self.n.clamp_min(1)).unsqueeze(2).expand_as(sig)
        c = sig / (sig + s2).clamp_min(1e-300)
        c = torch.where((self.n > self.p).unsqueeze(2), c, torch.ones_like(c))
        return c.float(), yhat

    def update(self, phi, z, delta, jac, yhat):
        if self.target == "goal":
            # goal pre-activation = one Gauss-Newton step: z - delta / mean(J^2), J = d out / d z
            # (exactly the label for the output unit); the unit's own drift cannot stale it
            lam_b = self.lam ** phi.shape[2]
            self.jsq.mul_(lam_b).add_(jac.double().square().sum(2))
            y = z.double() - delta.double() / (self.jsq / (self.n * lam_b + phi.shape[2])).clamp_min(1e-300).unsqueeze(2)
        else:
            y = delta.double()
        e = y - yhat
        floor = 1e-8 * (self.sy2_mu / self.n.clamp_min(1)).unsqueeze(2)
        y_lv = torch.log(e.square() + floor + 1e-30)
        y_mu = y
        lam_b = self.lam ** phi.shape[2]
        for sums, yy in ((self.b_mu, y_mu), (self.b_lv, y_lv)):
            sums.mul_(lam_b).add_(torch.einsum("sgbp,sgbo->sgop", phi, yy))
        self.sy_mu.mul_(lam_b).add_(y_mu.sum(2))
        self.sy2_mu.mul_(lam_b).add_(y_mu.square().sum(2))
        self.sy_lv.mul_(lam_b).add_(y_lv.sum(2))
        self.sy2_lv.mul_(lam_b).add_(y_lv.square().sum(2))
        self.n.mul_(lam_b).add_(phi.shape[2])
        # Woodbury: (lam A + Phi^T Phi)^-1; Sherman-Morrison at B=1
        Pl = self.Pinv / self.lam if self.lam < 1 else self.Pinv
        PPhi = torch.einsum("sgpq,sgbq->sgpb", Pl, phi)              # (S,G,p,B)
        Kb = torch.einsum("sgbp,sgpc->sgbc", phi, PPhi)              # (S,G,B,B)
        if phi.shape[2] == 1:
            X = PPhi.transpose(-1, -2) / (1 + Kb)
        else:
            Kb = Kb + torch.eye(Kb.shape[-1], device=Kb.device, dtype=Kb.dtype)
            X = torch.linalg.solve(Kb, PPhi.transpose(-1, -2))       # (S,G,B,p)
        self.Pinv = Pl - torch.einsum("sgpb,sgbq->sgpq", PPhi, X)


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
    dims = ((D_IN, H), (H, H), (H, D_OUT))

    results, levels = {}, {}
    for method in methods:
        shuffle = method.endswith("_shuffle")
        base = method[:-len("_shuffle")] if shuffle else method
        assert base in BASES, method
        post = base == "pp_post"
        target = "delta" if base == "pp1" else "goal"
        mode = {"pp_mu": "mu", "pp_var": "var", "pp_raw": "raw"}.get(base, "full")
        P = [p.clone() for p in P0]
        M = [torch.zeros_like(p) for p in P]
        V = [torch.zeros_like(p) for p in P]
        ro = [LayerReadout(S, G, i, o, dev, a.forget, target) for i, o in dims] if base.startswith("pp") else None
        c_prev = [torch.ones(S, G, B, o, device=dev) for _, o in dims]
        lvl_sum = torch.zeros(S, G, device=dev)
        lvl_sq = torch.zeros(S, G, device=dev)
        c_run = [torch.zeros(S, G, o, device=dev) for _, o in dims]  # per-perceptron running sum of raw c
        with torch.no_grad():
            for t in range(steps):
                x = xs[t]
                y_clean = teach(Ws2 if t >= switch else Ws, x) * amp[t].unsqueeze(-1)
                y = y_clean + (sigma[t] * eps[t]).unsqueeze(-1)
                W1, b1, W2, b2, W3, b3 = P
                h1, h2, out, zs = forward(P, x)
                r = out - y.unsqueeze(1)
                d3 = r
                d2 = torch.einsum("sgbo,sgoi->sgbi", d3, W3) * (1 - h2 * h2)
                d1 = torch.einsum("sgbo,sgoi->sgbi", d2, W2) * (1 - h1 * h1)
                xg = x.unsqueeze(1).expand(S, G, B, D_IN)
                deltas, inputs = (d1, d2, d3), (xg, h1, h2)

                if base == "adam":
                    cs = [torch.ones(S, G, B, o, device=dev) for _, o in dims]
                elif base == "oracle_wiener":
                    b_true = (out - y_clean.unsqueeze(1)).squeeze(-1)
                    b2 = b_true.square()
                    c = (b2 / (b2 + sigma[t].square().unsqueeze(1))).unsqueeze(-1)
                    cs = [c.expand(S, G, B, o) for _, o in dims]
                else:
                    cs = []
                    for l, (d, inp, z) in enumerate(zip(deltas, inputs, zs)):
                        phi = LayerReadout.feats(inp)
                        c, yhat = ro[l].predict(phi, z, mode)
                        ro[l].update(phi, z, d, d / r, yhat)
                        cs.append(c)
                    if base == "pp_norm":  # level: c / own running mean, per perceptron (horizonless)
                        for l, c in enumerate(cs):
                            c_run[l] += c.mean(2)
                        cs = [(c / (cr / (t + 1)).unsqueeze(2).clamp_min(1e-3)).clamp_max(20) for c, cr in zip(cs, c_run)]
                if shuffle:
                    if B == 1:
                        cs, c_prev = c_prev, cs
                    else:
                        perm = torch.randperm(B, device=dev)
                        cs = [c[:, :, perm] for c in cs]
                lvl = torch.stack([c.mean((2, 3)) for c in cs], 0).mean(0)
                lvl_sum += lvl
                lvl_sq += lvl.square()

                grads, gates = [], []
                for c, d, inp in zip(cs, deltas, inputs):
                    cd = d if post else c * d
                    grads.append(torch.einsum("sgbo,sgbi->sgoi", cd, inp) / B)
                    grads.append(cd.sum(2) / B)
                    if post:
                        cm = c.mean(2)  # (S,G,o); at B=1 exactly this sample's gate
                        gates.append(cm.unsqueeze(-1))
                        gates.append(cm)
                bc1 = 1 - beta1 ** (t + 1)
                bc2 = 1 - beta2 ** (t + 1)
                for k, (p, m, vv, g) in enumerate(zip(P, M, V, grads)):
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    vv.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    step = (m / bc1) / ((vv / bc2).sqrt() + adam_eps)
                    if post:
                        step = step * gates[k]
                    p.sub_((lr if p.dim() == 4 else lr_b) * step)
            _, _, out_te, _ = forward(P, x_te)
            mse = (out_te - y_te.unsqueeze(1)).square().mean((-1, -2))
        results[method] = mse
        levels[method] = (lvl_sum / steps, (lvl_sq / steps - (lvl_sum / steps) ** 2).clamp_min(0).sqrt())

    base_mse = (y_te.square().mean((-1, -2))).mean().item()
    print(f"# {a.tag} hetero={a.hetero} noise={a.noise} tail_df={a.tail_df} switch={a.switch_at} "
          f"signal_scales={a.signal_scales} null={a.null} forget={a.forget} batch={B} "
          f"samples={a.samples} seeds={S}  zero-predictor mse {base_mse:.4f}")
    print(f"{'method':22s} {'best_lr':>8s} {'mse':>8s} {'sem':>7s}   {'c_mean':>6s} {'c_sd':>6s}  "
          + " ".join(f"{l:8.0e}" for l in a.lr_grid))
    best = {}
    for method in methods:
        mse = results[method]
        mean_over_seeds = mse.mean(0)
        gi = int(mean_over_seeds.nan_to_num(nan=float("inf")).argmin())
        edge = " EDGE" if gi in (0, G - 1) else ""
        best[method] = mse[:, gi]
        sem = mse[:, gi].std().item() / math.sqrt(S)
        lm, ls = levels[method]
        print(f"{method:22s} {a.lr_grid[gi]:8.0e} {mean_over_seeds[gi].item():8.4f} {sem:7.4f}   "
              f"{lm[:, gi].mean().item():6.3f} {ls[:, gi].mean().item():6.3f}  "
              + " ".join(f"{m:8.4f}" for m in mean_over_seeds.tolist()) + edge)
    if "adam" in best:
        print("# paired vs adam (best LR each):  mean diff  sem  t")
        for method in methods:
            if method == "adam":
                continue
            diff = best[method] - best["adam"]
            sem = diff.std().item() / math.sqrt(S)
            print(f"  {method:22s} {diff.mean().item():+8.4f} {sem:7.4f} {diff.mean().item() / max(sem, 1e-12):+6.2f}")


if __name__ == "__main__":
    main()

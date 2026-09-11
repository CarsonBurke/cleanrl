"""Streaming MLP on the high-dimensional vol panel with a state-conditional precision gate.

Baseline: pooled dense MLP F-256-256-1, one Adam step per bar on the cross-section (panel_hd_mlp).
Gate `prec`: every sample's squared error enters the loss weighted by c_t = 1/Var_hat(y | x_t),
    Var_hat from a log-variance regression on the model's OWN last hidden layer (features [h, 1]),
    fitted by exact horizonless RLS (no EMA, no rate), target log(res^2 + eps).
    Calibration: the predicted deviation from the running mean is shrunk by the prequential slope
    rho = S_xy / S_xx of realised log-res^2 on predicted deviation (plain sums; rho -> 0 when the
    field is noise, -> 1 when it is exact). Level: c is divided by its running mean (plain sums), so
    the gate is a reallocation across samples, not a learning-rate change; cap 20.
    Optional Student-t influence (nu+1)/(nu+z^2) with nu from the kurtosis of z (plain sums).
Gate `prec_shuffle`: identical c values permuted within the bar's cross-section (level kept,
    state->sample correspondence destroyed).
Everything else identical to the baseline; the gate only re-weights samples inside the loss.

    .venv/bin/python cleanrl/plasticity/panel_hd_gate.py --gate prec --stream-lrs 1e-4 3e-4
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
    gate: str = "prec"
    """none | prec | prec_shuffle"""
    student: bool = True
    cap: float = 20.0
    stream_lrs: tuple[float, ...] = (1e-4, 3e-4)
    permuted: bool = False
    seed: int = 1


class Net(torch.nn.Module):
    def __init__(self, f, width):
        super().__init__()
        self.l1 = torch.nn.Linear(f, width)
        self.l2 = torch.nn.Linear(width, width)
        self.l3 = torch.nn.Linear(width, 1)

    def forward(self, x):
        h = torch.tanh(self.l2(torch.tanh(self.l1(x))))
        return self.l3(h).squeeze(-1), h


def run(a, bank, yy, vv, lr):
    T, L, F, cut, dev = bank.T, bank.L, bank.F, bank.cut, bank.dev
    torch.manual_seed(a.seed)
    net = Net(F, a.width).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    H = a.width + 1
    u = torch.zeros(H, device=dev, dtype=torch.float64)           # log-variance readout
    P = torch.eye(H, device=dev, dtype=torch.float64) * 100.0     # RLS covariance (horizonless)
    s_xy = torch.zeros((), device=dev); s_xx = torch.zeros((), device=dev)
    s_lv = torch.zeros((), device=dev); n_lv = torch.zeros((), device=dev)
    s_c = torch.zeros((), device=dev); n_c = torch.zeros((), device=dev)
    s_z2 = torch.zeros((), device=dev); s_z4 = torch.zeros((), device=dev)
    se = 0.0; ss = 0.0; lvl_sum = 0.0; lvl_sq = 0.0; lvl_n = 0
    for t in range(L + 1, T - 1):
        mk = vv[t]
        n = int(mk.sum())
        if n == 0:
            continue
        x = bank.feats(t)[mk]; yb = yy[t][mk]
        pred, h = net(x)
        res = pred - yb
        if t >= cut:
            se += res.square().sum().item(); ss += yb.square().sum().item()
        if a.gate == "none":
            loss = res.square().mean()
        else:
            with torch.no_grad():
                phi = torch.cat([h, torch.ones(n, 1, device=dev)], 1).double()   # (n, H)
                raw = (phi @ u).float()                                # predicted log-variance
                mean_lv = s_lv / n_lv.clamp_min(1.0)
                rho = (s_xy / s_xx.clamp_min(1e-12)).clamp(0.0, 1.0)
                p_lv = (mean_lv + rho * (raw - mean_lv)).clamp(-12.0, 12.0)
                var_hat = p_lv.exp()
                c_raw = 1.0 / var_hat
                if a.student:
                    z2 = res.square() / var_hat
                    kappa = (s_z4 / n_c.clamp_min(1.0)) / (s_z2 / n_c.clamp_min(1.0)).square().clamp_min(1e-12)
                    nu = ((4.0 * kappa - 6.0) / (kappa - 3.0).clamp_min(1e-3)) if kappa > 3.05 else torch.tensor(1e6, device=dev)
                    c_raw = c_raw * (nu + 1.0) / (nu + z2)
                    z2c = z2.clamp_max(1e4)
                    s_z2 += z2c.sum(); s_z4 += z2c.square().sum()
                c = (c_raw / (s_c / n_c).clamp_min(1e-12) if n_c > 0 else torch.ones_like(c_raw)).clamp_max(a.cap)
                s_c += c_raw.sum(); n_c += n
                if a.gate == "prec_shuffle":
                    c = c[torch.randperm(n, device=dev)]
                lvl_sum += c.sum().item(); lvl_sq += c.square().sum().item(); lvl_n += n
                # readout update (after use: held-out), target log(res^2 + eps); prequential calibration sums
                tgt = (res.square() + 1e-6).log()
                dev_ = raw - mean_lv
                s_xy += (dev_ * (tgt - mean_lv)).sum(); s_xx += dev_.square().sum()
                s_lv += tgt.sum(); n_lv += n
                Pphi = P @ phi.T                                       # (H, n)
                S = phi @ Pphi + torch.eye(n, device=dev, dtype=torch.float64)
                K = torch.linalg.solve(S, Pphi.T).T                    # (H, n) gain
                u += K @ (tgt.double() - raw.double())
                P -= K @ Pphi.T
                P = 0.5 * (P + P.T)
            loss = (c * res.square()).sum() / c.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    lvl_m = lvl_sum / max(lvl_n, 1)
    lvl_sd = math.sqrt(max(lvl_sq / max(lvl_n, 1) - lvl_m ** 2, 0.0))
    return se / ss, lvl_m, lvl_sd


def main():
    a = tyro.cli(Args)
    bank = Bank(a)
    print(f"panel {bank.T} bars x {bank.N} stocks, F={bank.F}, width {a.width}, gate {a.gate}, student {a.student}")
    t0 = time.time()
    for name, yy, vv in bank.streams():
        if name == "PERM" and not a.permuted:
            continue
        for lr in a.stream_lrs:
            rel, lm, ls = run(a, bank, yy, vv, lr)
            print(f"  {name} gate={a.gate} lr={lr:.0e}: {rel:.5f}   level {lm:.3f} sd {ls:.3f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

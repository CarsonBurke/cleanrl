"""Theory toy 3: does the HL-Gauss cross-entropy resolve sub-bin action contrasts?

REPPO's head: 151 bins on [-50, 250] (width 2.0), smoothing sigma 0.75 * width = 1.5, Q ~ 100.
(a) Representation: decode(project(y)) - y over a fine grid (aliasing of the mean).
(b) Loss geometry: dCE/dm for a prediction that is the label shape centred at m, against (m - y) / s^2:
    the CE is a quadratic in the mean with curvature 1/s^2, i.e. an MSE with a fixed scale, no dead zone.
(c) Learning: same MLP trunk, same data (level sd 2.3 around 100, action contrast sd 0.22, iid target noise),
    inputs (s, eps) so nothing but the loss differs: CE/HL-Gauss vs MSE on standardized targets vs MSE with the
    contrast head scaled. Reported: corr of the predicted within-state contrast with the true one on fresh rows.
CPU only.
"""
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from cleanrl.shared.hl_gauss import HLGaussConfig  # noqa: E402

torch.set_num_threads(3)
torch.manual_seed(0)
hl = HLGaussConfig(v_min=-50.0, v_max=250.0, num_bins=151).build("cpu")

print("== (a) aliasing of the decoded mean, y in [99, 101] ==")
y = torch.linspace(99.0, 101.0, 2001, dtype=torch.float64).float()
labels = hl.project(y)
decoded = hl.to_scalar(labels.clamp_min(1e-30).log())
print(f"max |decode(project(y)) - y| = {(decoded - y).abs().max():.2e}; "
      f"sd of the error = {(decoded - y).std():.2e}  (action contrasts are ~0.03-0.22)")

print("\n== (b) CE gradient w.r.t. the predicted mean (prediction = label shape at m) ==")
target = torch.tensor(100.3)
for delta in (0.01, 0.05, 0.2, 1.0, 3.0, 6.0):
    m = (target + delta).clone().requires_grad_(True)
    # differentiable shape: Gaussian logits over bin centres with the label's sigma
    centers = torch.linspace(-50.0, 250.0, 151)
    for s_pred in (1.5, 2.5):
        logits = -0.5 * ((centers - m) / s_pred) ** 2
        ce = -(hl.project(target) * logits.log_softmax(-1)).sum()
        (grad,) = torch.autograd.grad(ce, m)
        print(f"m - y = {delta:5.2f}, predicted sd {s_pred}: dCE/dm = {grad.item():+.5f}   (m - y)/s^2 = "
              f"{delta / s_pred ** 2:+.5f}")

print("\n== (c) learning a 1%-variance action contrast: CE vs MSE, identical trunk and data ==")
n, d, sigma_a, level_sd, noise = 17, 6, 0.22, 2.3, 0.6
gen = torch.Generator().manual_seed(1)
Wu, Wg = torch.randn(32, n, generator=gen) / n ** 0.5, torch.randn(32, n, generator=gen) / n ** 0.5
cu, Cg = torch.randn(32, generator=gen), torch.randn(d, 32, generator=gen)


def truth(s, eps):
    u = torch.tanh(s @ Wu.T) @ cu
    g = torch.tanh(s @ Wg.T) @ Cg.T
    return u, (g * eps).sum(-1)


probe_s, probe_e = torch.randn(50000, n, generator=gen), torch.randn(50000, d, generator=gen)
pu, pc = truth(probe_s, probe_e)
u_scale, c_scale = level_sd / pu.std(), sigma_a / pc.std()


def sample(N):
    s, eps = torch.randn(N, n, generator=gen), torch.randn(N, d, generator=gen)
    u, c = truth(s, eps)
    return s, eps, 100.0 + u_scale * u, c_scale * c


def trunk(out):
    return nn.Sequential(nn.Linear(n + d, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, out))


def run(kind, iters=6):
    torch.manual_seed(0)
    net = trunk(151 if kind == "ce" else 1)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    def predict(s, eps):
        out = net(torch.cat((s, eps), -1))
        return hl.to_scalar(out) if kind == "ce" else 100.0 + level_sd * out.squeeze(-1)

    rows = []
    for it in range(iters):
        s, eps, lvl, con = sample(16384)
        yb = lvl + con + noise * torch.randn(16384, generator=gen)
        for _ in range(128):
            idx = torch.randint(0, 16384, (2048,))
            out = net(torch.cat((s[idx], eps[idx]), -1))
            if kind == "ce":
                loss = -(hl.project(yb[idx]) * out.log_softmax(-1)).sum(-1).mean()
            else:
                loss = (out.squeeze(-1) - (yb[idx] - 100.0) / level_sd).square().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
        with torch.no_grad():
            s, eps, lvl, con = sample(4096)
            eps2 = torch.randn(8, 4096, d, generator=gen)
            pred = torch.stack([predict(s, e) for e in eps2])
            true = torch.stack([truth(s, e)[1] * c_scale for e in eps2])
            pc_, tc_ = pred - pred.mean(0), true - true.mean(0)
            corr = torch.corrcoef(torch.stack((pc_.flatten(), tc_.flatten())))[0, 1].item()
            lvl_err = (pred.mean(0) - lvl).std().item()
        rows.append(f"it{it + 1}: corr {corr:+.2f} sd ratio {pc_.std().item() / tc_.std().item():.2f} lvl err {lvl_err:.2f}")
    print(f"{kind:4s} | " + " | ".join(rows[i] for i in (0, 2, iters - 1)))


for kind in ("ce", "mse"):
    run(kind)

"""Theory toy 2: on one on-policy batch, the pathwise gradient through a regression critic IS the
score-function gradient (compatible function approximation), and both obey the same SNR law.

Setup. s ~ N(0, I_n); policy a = mu(s) + sigma eps with mu(s) = W psi(s), psi = fixed random tanh features.
True Q(s, a) = V(s) + g(s).(a - mu(s)), g(s) = G0 psi(s); observed y = Q + nu * xi.
  SF      : G_sf = mean_i (y_i - b_i) eps_i psi_i^T / sigma                 (gradient of J w.r.t. W)
  critic  : Q_hat = v.phi(s) + (a - mu(s))^T Wc psi(s), joint OLS; pathwise G_pw = mean_i Wc psi_i psi_i^T
Claims checked: (1) cos(G_pw, G_sf) -> 1 and the two have the same cos with the true gradient;
(2) cos^2(G_hat, G) = |G|^2 / (|G|^2 + tr Sigma / N) with tr Sigma = (noise^2 + (d + 1) sigma_A^2-ish) m / sigma^2;
(3) min-norm single-sample interpolation gives fresh-action contrast corr
    sigma_A / (d sqrt(nu^2 / (d - 2) + sigma_A^2 / d)); K pooled SF samples give the formula in the last block.
CPU numpy only.
"""
import numpy as np

rng = np.random.default_rng(0)
n, d, m = 17, 6, 24
sigma = 0.5
Wf = rng.normal(size=(m, n)) / np.sqrt(n)


def psi(s):
    return np.tanh(s @ Wf.T)


G0 = rng.normal(size=(d, m))
# scale so the within-state sd sigma*|g| averages 0.22
probe = psi(rng.normal(size=(20000, n)))
G0 *= 0.22 / (sigma * np.sqrt(((probe @ G0.T) ** 2).sum(1).mean()))
vw = rng.normal(size=m)
vw *= 2.3 / (probe @ vw).std()
Sigma_psi = probe.T @ probe / len(probe)
G_true = G0 @ Sigma_psi  # d/dW E[Q(s, W psi)] = E[g psi^T]


def cos(a, b):
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def batch(N, nu, base_err):
    s = rng.normal(size=(N, n))
    p = psi(s)
    eps = rng.normal(size=(N, d))
    g = p @ G0.T
    v = p @ vw
    y = v + sigma * (g * eps).sum(1) + nu * rng.normal(size=N)
    b = v + base_err * rng.normal(size=N)  # baseline with independent level error
    return p, eps, y, b, g


def estimators(N, nu, base_err):
    p, eps, y, b, _ = batch(N, nu, base_err)
    G_sf = ((y - b)[:, None, None] * eps[:, :, None] * p[:, None, :]).mean(0) / sigma
    # joint OLS of y on [phi(s) = psi(s) (level), vec(sigma eps psi^T) (slope)]
    X = np.concatenate([p, (sigma * eps[:, :, None] * p[:, None, :]).reshape(N, -1)], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    Wc = coef[m:].reshape(d, m)
    G_pw = Wc @ (p.T @ p / N)
    return G_sf, G_pw


print("== (1,2) SF vs pathwise-through-OLS-critic; nu = target noise sd, level sd 2.3, within-state sd 0.22 ==")
print(f"{'N':>7} {'nu':>5} {'cos(pw,sf)':>11} {'cos(sf,true)':>13} {'cos(pw,true)':>13} {'pred cos(sf)':>13}")
for N in (1024, 4096, 16384, 65536):
    for nu in (0.0, 0.66):
        res = np.array([[cos(a, b), cos(a, G_true), cos(b, G_true)] for a, b in
                        (estimators(N, nu, 0.0) for _ in range(12))]).mean(0)
        # per-sample covariance trace of (y-b) eps psi^T / sigma: E[(nu^2 + (sigma g.eps)^2) |eps|^2 |psi|^2]/sigma^2
        p, eps, y, b, g = batch(50000, nu, 0.0)
        tr = (((y - b) ** 2) * (eps ** 2).sum(1) * (p ** 2).sum(1)).mean() / sigma ** 2 - (G_true ** 2).sum()
        pred = np.sqrt((G_true ** 2).sum() / ((G_true ** 2).sum() + tr / N))
        print(f"{N:7d} {nu:5.2f} {res[0]:11.3f} {res[1]:13.3f} {res[2]:13.3f} {pred:13.3f}")

print("\n== effect of a level (baseline) error on the SF estimator; N = 16384, nu = 0.66 ==")
for be in (0.0, 0.3, 0.9):
    c = np.mean([cos(estimators(16384, 0.66, be)[0], G_true) for _ in range(12)])
    print(f"baseline error sd {be:.1f}: cos(sf,true) = {c:.3f}")

print("\n== (3) per-state estimators of the action slope from K samples, evaluated on fresh actions ==")
print("corr of predicted contrast sigma*ghat.eps' with the true contrast; formulas vs Monte Carlo")
sigA = 0.22
for nu in (0.3, 0.6, 0.9):
    for K, kind in ((1, "minnorm"), (1, "sf"), (6, "sf"), (24, "sf"), (96, "sf")):
        S = 20000
        g = rng.normal(size=(S, d))
        g *= sigA / (sigma * np.linalg.norm(g, axis=1, keepdims=True))
        eps = rng.normal(size=(S, K, d))
        resid = sigma * (eps * g[:, None]).sum(-1) + nu * rng.normal(size=(S, K))
        if kind == "minnorm":  # interpolate the one residual with the minimum-norm slope
            ghat = eps[:, 0] * (resid[:, 0] / (sigma * (eps[:, 0] ** 2).sum(-1)))[:, None]
            # Var = nu^2 E[1/|eps|^2] + sigA^2/d = nu^2/(d-2) + sigA^2/d; Cov with truth = sigA^2/d
            pred = sigA / (d * np.sqrt(nu ** 2 / (d - 2) + sigA ** 2 / d))
        else:
            ghat = (eps * resid[..., None]).mean(1) / sigma
            pred = None
        e2 = rng.normal(size=(S, d))
        true_c, hat_c = sigma * (g * e2).sum(1), sigma * (ghat * e2).sum(1)
        mc = np.corrcoef(true_c, hat_c)[0, 1]
        if pred is None:  # Var(hat contrast) = sigA^2 + ((d+1) sigA^2 + d nu^2)/K  (derivation in the report)
            pred = sigA / np.sqrt(sigA ** 2 + ((d + 1) * sigA ** 2 + d * nu ** 2) / K)
        print(f"nu {nu:.1f} K {K:3d} {kind:8s} corr MC {mc:6.3f}  formula {pred:6.3f}  sd(hat contrast) {hat_c.std():.3f}")

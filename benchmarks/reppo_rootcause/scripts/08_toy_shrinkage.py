"""Red-team 08: what does an OPTIMAL (posterior-mean) action-slope estimator look like at the measured SNR?

Model per local slope cell (d = 6): g ~ N(0, tau^2 I), sigma_A^2 = d sigma^2 tau^2, K on-policy samples
y_k = sigma g.eps_k + n_k, n ~ N(0, nu^2), per-sample signal share S = sigma_A^2 / (sigma_A^2 + nu^2).
Posterior mean: g_hat = (E'E + kappa I)^-1 E'y / sigma, kappa = d nu^2 / sigma_A^2 = d (1 - S) / S.
Scored on fresh actions: corr with the truth, amplitude ratio sd(hat)/sd(true), calibration slope true-on-hat.
Identity for ANY posterior mean: Cov(hat, true) = Var(hat)  =>  slope = 1 and amplitude ratio = corr.
Contrasts: (i) unshrunk least squares / minimum-norm (kappa -> 0): amplitude > corr, slope < 1 (noise fitted);
           (ii) over-shrunk (posterior mean x 0.5): amplitude < corr, slope 2 (under-fit).
"""
import numpy as np

rng = np.random.default_rng(0)
d, cells, sig_a = 6, 20000, 1.0
print("S     K   | posterior mean: corr  amp  slope | closed form corr=sqrt(K/(K+kappa')) | min-norm/OLS: corr  amp  slope | half-shrunk: corr amp slope")
for S in (0.05, 0.06, 0.07, 0.24):
    nu2 = sig_a ** 2 * (1 - S) / S
    kappa = d * nu2 / sig_a ** 2
    for K in (1, 2, 4, 8, 16, 64):
        g = rng.normal(size=(cells, d)) * sig_a / np.sqrt(d)            # sigma absorbed: contrast = g.eps
        E = rng.normal(size=(cells, K, d))
        y = np.einsum("ckd,cd->ck", E, g) + rng.normal(size=(cells, K)) * np.sqrt(nu2)
        EtE = np.einsum("ckd,cke->cde", E, E); Ety = np.einsum("ckd,ck->cd", E, y)
        post = np.linalg.solve(EtE + kappa * np.eye(d), Ety[..., None])[..., 0]
        ols = np.einsum("cde,ce->cd", np.linalg.pinv(EtE, rcond=1e-10), Ety)
        ef = rng.normal(size=(cells, 8, d)); true = np.einsum("ckd,cd->ck", ef, g)
        out = []
        for est in (post, ols, 0.5 * post):
            hat = np.einsum("ckd,cd->ck", ef, est)
            c = (hat * true).sum() / np.sqrt((hat ** 2).sum() * (true ** 2).sum())
            out.append((c, hat.std() / true.std(), (hat * true).sum() / (hat ** 2).sum()))
        closed = np.sqrt(K / (K + kappa + d + 1))                       # 04's K_half = (d+1) + d nu^2/sigma_A^2
        print(f"{S:.2f} {K:3d}   | {out[0][0]:5.3f} {out[0][1]:5.3f} {out[0][2]:5.2f}   | {closed:5.3f}                           |"
              f" {out[1][0]:5.3f} {out[1][1]:6.2f} {out[1][2]:5.2f}    | {out[2][0]:5.3f} {out[2][1]:5.3f} {out[2][2]:4.2f}")
print("\nInverse: effective pooled samples per slope cell implied by a measured corr rho (K = K_half rho^2/(1-rho^2)),")
print("and by a measured amplitude ratio r under optimal shrinkage (r = rho):")
for S in (0.05, 0.06, 0.07, 0.24):
    k_half = (d + 1) + d * (1 - S) / S
    print(f"  S {S:.2f}  K_half {k_half:6.1f}   " + "  ".join(f"rho {r:.2f} -> K {k_half*r*r/(1-r*r):5.1f}" for r in (0.11, 0.15, 0.23, 0.29, 0.32)))

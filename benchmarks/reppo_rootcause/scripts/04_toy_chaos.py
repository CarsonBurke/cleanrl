"""Theory toy 4: action gradients of Q^pi under chaotic dynamics (exact, 1-D circle map).

x' = f(x) + a (mod 1), f(x) = x + w + (kappa / 2 pi) sin(2 pi x), a = sigma * eps (mean action 0), gamma < 1.
(kappa, w) = (3, 0.15) is chaotic under the noisy policy, (3, 0.31) contracting (Lyapunov exponent printed). V^pi is solved exactly on a grid, so
  Q^pi(x, a) = r(x) + gamma V(f(x) + a),  smoothed slope g(x) = d/da E_eps Q(x, a + sigma eps) at a = 0
are exact references.
(a) How rough is Q^pi in a at the scale of sigma: pointwise vs sigma-smoothed slope, linear-in-noise share.
(b) Per-sample variance of two unbiased estimators of g as the horizon n grows (exact V bootstrap at n):
    zeroth order (score function) eps_0 / sigma * (n-step return - V(x_0)) and first order (pathwise through the
    true simulator with reparameterized noise).
(c) Horizon for the SLOPE with an imperfect critic V_hat = V blurred at length l: exact bias of the n-step
    score-function target plus Monte-Carlo variance, MSE = bias^2 + var / K (K pooled samples).
    The level error of the same n-step target is printed next to it: the best n for the slope and for the
    level differ.
CPU numpy only.
"""
import numpy as np

M = 2048
x = np.arange(M) / M
freq = np.fft.fftfreq(M, d=1.0 / M)
W_ROT, KAPPA = 0.31, 3.0


def f(z):
    return z + W_ROT + KAPPA / (2 * np.pi) * np.sin(2 * np.pi * z)


def fprime(z):
    return 1.0 + KAPPA * np.cos(2 * np.pi * z)


def reward(z):
    return np.cos(2 * np.pi * z) + 0.5 * np.cos(4 * np.pi * z + 1.0)


def reward_prime(z):
    return -2 * np.pi * np.sin(2 * np.pi * z) - 2 * np.pi * np.sin(4 * np.pi * z + 1.0)


def blur(v, length):
    return np.fft.ifft(np.fft.fft(v) * np.exp(-0.5 * (2 * np.pi * freq * length) ** 2)).real


def deriv(v):
    return np.fft.ifft(np.fft.fft(v) * (2j * np.pi * freq)).real


def interp(v, z):
    pos = (z % 1.0) * M
    lo = np.floor(pos).astype(int) % M
    frac = pos - np.floor(pos)
    return (1 - frac) * v[lo] + frac * v[(lo + 1) % M]


def transition(sigma):
    """P[i, j] = Pr(x' in cell j | x_i) under a = sigma eps (wrapped Gaussian)."""
    centre = f(x)[:, None]
    dist = (x[None, :] - centre + 0.5) % 1.0 - 0.5
    P = sum(np.exp(-0.5 * ((dist + k) / sigma) ** 2) for k in (-1, 0, 1))
    return P / P.sum(1, keepdims=True)


def simulate(sigma, n_max, N, rng):
    x0 = rng.uniform(size=N)
    for _ in range(200):  # burn in to the stationary law
        x0 = (f(x0) + sigma * rng.normal(size=N)) % 1.0
    xs, eps = [x0], []
    for _ in range(n_max):
        e = rng.normal(size=N)
        eps.append(e)
        xs.append((f(xs[-1]) + sigma * e) % 1.0)
    return np.array(xs), np.array(eps)


def study(sigma, gamma, rng, w_rot, kappa, N=200000, horizons=(1, 2, 3, 5, 8, 12, 20, 40)):
    global W_ROT, KAPPA
    W_ROT, KAPPA = w_rot, kappa
    P = transition(sigma)
    V = np.linalg.solve(np.eye(M) - gamma * P, reward(x))
    xs, eps = simulate(sigma, max(horizons), N, rng)
    lyap = np.log(np.abs(fprime(xs[:-1]))).mean()
    g_grid = gamma * interp(blur(deriv(V), sigma), f(x))       # exact smoothed slope on the grid
    point = gamma * interp(deriv(V), f(x))                      # pointwise dQ/da at a = 0
    # within-state decomposition of Q over eps
    e = rng.normal(size=(64, M))
    q = gamma * interp(V, f(x)[None] + sigma * e)
    within = q - q.mean(0)
    linear_share = ((sigma * g_grid) ** 2).mean() / (within ** 2).mean()
    print(f"\n=== kappa {kappa}, w {w_rot}, sigma {sigma}, gamma {gamma}: Lyapunov {lyap:+.3f}/step (e-fold {1 / lyap:.1f} steps), horizon "
          f"{1 / (1 - gamma):.0f} steps, gamma*exp(lyap) = {gamma * np.exp(lyap):.2f} ===")
    print(f"(a) sd V {V.std():.3f} | within-state sd of Q {within.std():.4f} | rms smoothed slope {np.sqrt((g_grid ** 2).mean()):.3f}"
          f" | rms pointwise slope {np.sqrt((point ** 2).mean()):.3f} | linear-in-eps share of within-state var "
          f"{linear_share:.2f} | corr(pointwise, smoothed) {np.corrcoef(point, g_grid)[0, 1]:.2f}")
    g0 = interp(g_grid, xs[0])
    g2 = (g0 ** 2).mean()
    print("(b) per-sample variance / E[g^2] of unbiased slope estimators (exact V at the horizon)")
    print(f"    {'n':>3} {'zeroth (SF)':>12} {'first (pathwise sim)':>21} {'bias check SF':>14} {'bias check FO':>14}")
    disc_r = np.zeros(N)
    D = np.ones(N)             # d x_k / d a_0 (reparameterized noise held fixed)
    first = np.zeros(N)
    zeroth_var = {}
    for k in range(1, max(horizons) + 1):
        # after this block: disc_r = sum_{j<k} gamma^j r(x_j), first = sum_{1<=j<k} gamma^j r'(x_j) D_j
        disc_r += gamma ** (k - 1) * reward(xs[k - 1])
        if k > 1:
            first += gamma ** (k - 1) * reward_prime(xs[k - 1]) * D
            D = D * fprime(xs[k - 1])
        if k in horizons:
            sf = eps[0] / sigma * (disc_r + gamma ** k * interp(V, xs[k]) - interp(V, xs[0]))
            fo = first + gamma ** k * interp(deriv(V), xs[k]) * D
            zeroth_var[k] = (sf - g0).var()
            print(f"    {k:3d} {zeroth_var[k] / g2:12.2f} {(fo - g0).var() / g2:21.3g} "
                  f"{np.corrcoef(sf, g0)[0, 1] * sf.std() / g0.std():14.2f} {(fo * g0).mean() / g2:14.2f}")
    print("(c) slope MSE / E[g^2] of the n-step SF target with V_hat = blur(V, l); K = pooled samples per estimate")
    for length in (sigma, 4 * sigma, 16 * sigma):
        v_hat = blur(V, length)
        U = v_hat.copy()
        rows = []
        for n in range(1, max(horizons) + 1):
            if n in horizons:
                g_n = gamma * interp(blur(deriv(U), sigma), f(x))
                bias2 = ((interp(g_n, xs[0]) - g0) ** 2).mean()
                # level error of the same target, as a function of the state (expected n-step value - V)
                level = ((reward(x) + gamma * P @ U) - V).std()
                # variance: the n-step SF sample with V_hat as bootstrap and baseline
                ret = sum(gamma ** j * reward(xs[j]) for j in range(n)) + gamma ** n * interp(v_hat, xs[n])
                var = (eps[0] / sigma * (ret - interp(v_hat, xs[0]))).var()
                rows.append((n, bias2 / g2, var / g2, level))
            U = reward(x) + gamma * P @ U
        print(f"    V_hat blur l = {length / sigma:5.2f} sigma (level error sd {(v_hat - V).std():.3f}, sd V {V.std():.3f})")
        print("      n      : " + " ".join(f"{r[0]:7d}" for r in rows))
        print("      bias^2 : " + " ".join(f"{r[1]:7.3f}" for r in rows))
        print("      var    : " + " ".join(f"{r[2]:7.1f}" for r in rows))
        print("      lvl err: " + " ".join(f"{r[3]:7.3f}" for r in rows))
        for K in (16, 256, 4096):
            mse = [r[1] + r[2] / K for r in rows]
            best = rows[int(np.argmin(mse))][0]
            print(f"      K {K:5d}: best n = {best:2d}, MSE/E[g^2] at n=1 {mse[0]:.3f}, at best {min(mse):.3f}, "
                  f"at n={rows[-1][0]} {mse[-1]:.3f}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # chaotic (Lyapunov ~ +0.3/step) at two horizons and two noise scales, then a contracting control
    for sigma, gamma, w_rot, kappa in ((0.02, 0.9, 0.15, 3.0), (0.02, 0.97, 0.15, 3.0), (0.005, 0.9, 0.15, 3.0),
                                       (0.02, 0.9, 0.31, 3.0)):
        study(sigma, gamma, rng, w_rot, kappa)

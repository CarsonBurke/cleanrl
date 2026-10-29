"""Theory toy 1: what an on-policy regression critic learns about dQ/da (identifiability / leak).

Part A (closed form): two collinear features, GD from zero. y = v x + g sigma eps, a = k x + sigma eps,
model w_x x + w_a a. The identifying eigenvalue is ~ sigma^2 / (1 + k^2); until GD resolves it the action
coefficient is the ASSOCIATION k v / (1 + k^2), not the causal slope g.

Part B (exact LQ truth, MLP critic): s' = A s + B a, a = K s + sigma eps, quadratic + linear reward.
Q^pi is an exact quadratic (discounted Lyapunov), so dQ/da and the critic's own one-step backup
r + gamma V_hat(f(s, a)) are exact references. K = kappa * K_lqr: kappa < 1 is an under-gained feedback
(truth: push harder; association: effort marks bad states). Critic = MLP on (s, a) trained like REPPO
(16 passes of 2048-minibatches on each fresh 16384-row on-policy batch, warm-started across batches, policy
held fixed = best case), targets y = Q^pi(s, a) + nu * noise. Compared with the orthogonal parameterization
V(s) + g(s).eps (v63's linear part) on identical data.
Reported on fresh actions at held-out states: level corr, within-state contrast corr with truth, slope of
true directional derivative on the critic's, and the within-state corr of Q_hat with its own backup.
CPU only.
"""
import sys

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn

torch.set_num_threads(3)


def part_a():
    print("== Part A: collinear-feature GD, closed form ==")
    v, g = 2.3, 0.44
    for k, sigma in ((1.0, 0.5), (1.0, 0.2), (2.0, 0.2)):
        C = np.array([[1.0, k], [k, k * k + sigma ** 2]])
        c = np.array([v, k * v + g * sigma ** 2])
        lam, U = np.linalg.eigh(C)
        row = []
        for t in (1, 10, 100, 1000, 10000):
            w = U @ ((1 - np.exp(-lam * t)) / lam * (U.T @ c))
            row.append(f"t={t}: w_a={w[1]:+.3f}")
        print(f"k={k} sigma={sigma}: eig={lam[0]:.4f},{lam[1]:.3f}  true g={g}  early leak k v/(1+k^2)="
              f"{k * v / (1 + k * k):.3f} | " + "  ".join(row))


def make_system(seed, n=17, d=6, kappa=0.5, gamma=0.99, b_scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) / np.sqrt(n)
    A *= 0.97 / np.abs(np.linalg.eigvals(A)).max()
    B = b_scale * rng.normal(size=(n, d)) / np.sqrt(n)  # b_scale sets the within-state share of Var(Q)
    Qc = np.eye(n)
    Rc = 0.1 * np.eye(d)
    # discounted LQR: scale A, B by sqrt(gamma)
    P = scipy.linalg.solve_discrete_are(np.sqrt(gamma) * A, np.sqrt(gamma) * B, Qc, Rc)
    K_opt = -np.linalg.solve(Rc + gamma * B.T @ P @ B, gamma * B.T @ P @ A)
    K = kappa * K_opt
    c = rng.normal(size=n) * 2.0  # linear reward term (the "forward velocity" analogue)
    return dict(A=A, B=B, Qc=Qc, Rc=Rc, K=K, c=c, gamma=gamma, n=n, d=d)


def solve_value(sys_, sigma):
    """V^pi(s) = s^T P s + p^T s + p0 for a = K s + sigma eps; reward r = c.s - s^T Qc s - a^T Rc a."""
    A, B, K, gamma = sys_["A"], sys_["B"], sys_["K"], sys_["gamma"]
    Acl = A + B @ K
    assert np.abs(np.linalg.eigvals(Acl)).max() < 1.0, "closed loop unstable"
    M = -(sys_["Qc"] + K.T @ sys_["Rc"] @ K)
    P = scipy.linalg.solve_discrete_lyapunov(np.sqrt(gamma) * Acl.T, M)
    p = np.linalg.solve(np.eye(sys_["n"]) - gamma * Acl.T, sys_["c"])
    noise_const = sigma ** 2 * (-np.trace(sys_["Rc"]) + gamma * np.trace(B.T @ P @ B))
    return P, p, noise_const / (1 - gamma)


class Truth:
    def __init__(self, sys_, sigma, scale=1.0):
        self.sys, self.sigma = sys_, sigma
        self.P, self.p, self.p0 = solve_value(sys_, sigma)
        self.scale = scale
        self.proc = 0.0

    def reward(self, s, a):
        y = s @ self.sys["c"] - np.einsum("...i,ij,...j", s, self.sys["Qc"], s) - np.einsum(
            "...i,ij,...j", a, self.sys["Rc"], a)
        return self.scale * y

    def step(self, s, a):
        return s @ self.sys["A"].T + a @ self.sys["B"].T

    def value(self, s):
        return self.scale * (np.einsum("...i,ij,...j", s, self.P, s) + s @ self.p + self.p0)

    def q(self, s, a):
        return self.reward(s, a) + self.sys["gamma"] * self.value(self.step(s, a))

    def dqda(self, s, a):
        nxt = self.step(s, a)
        return self.scale * (-2 * a @ self.sys["Rc"] + self.sys["gamma"] * (2 * nxt @ self.P + self.p) @ self.sys["B"])


def rollout(truth, rng, envs=16, steps=1024, burn=200):
    """On-policy rows. Process noise truth.proc (sd) drives the state spread; it adds only a constant to Q^pi."""
    sys_, sigma = truth.sys, truth.sigma
    s = np.zeros((envs, sys_["n"]))
    S, E = [], []
    for t in range(burn + steps):
        eps = rng.normal(size=(envs, sys_["d"]))
        a = s @ sys_["K"].T + sigma * eps
        if t >= burn:
            S.append(s)
            E.append(eps)
        s = truth.step(s, a) + truth.proc * rng.normal(size=s.shape)
    return np.concatenate(S), np.concatenate(E)


def mlp(inp, width=256, out=1):
    return nn.Sequential(nn.Linear(inp, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, out))


def run_b(kappa, sigma, nu, e_sd=0.0, b_scale=1.0, proc=0.0, c_scale=1.0, drift=0.0, e_freq=1.5, seed=0, iters=16, passes=16, verbose=True):
    """e_sd: sd of a rough state-only target component e(s) (random Fourier features of the standardized state)
    standing in for the part of the value level the critic cannot resolve; it has NO action dependence.
    drift: per-iteration sd of a random-walk state-only level change (half linear, half rough in s), standing in
    for V^pi moving between policy iterations; also action-independent, so the true slope is stationary."""
    sys_ = make_system(seed, kappa=kappa, b_scale=b_scale)
    sys_["c"] = sys_["c"] * c_scale
    truth = Truth(sys_, sigma)
    truth.proc = proc
    rng = np.random.default_rng(seed + 1)
    S0, E0 = rollout(truth, rng)
    A0 = S0 @ sys_["K"].T + sigma * E0
    q0 = truth.q(S0, A0)
    truth.scale = 2.3 / truth.value(S0).std()  # between-state sd 2.3, like the oracle
    s_mean, s_std = S0.mean(0), S0.std(0)
    mu0 = S0 @ sys_["K"].T
    a_std = float(A0.std())
    q_mean = float(truth.q(S0, A0).mean())
    K_t = torch.tensor(sys_["K"], dtype=torch.float32)
    feat_rng = np.random.default_rng(seed + 7)
    omega = feat_rng.normal(size=(64, sys_["n"])) * e_freq / np.sqrt(sys_["n"])
    phase, amp = feat_rng.uniform(0, 2 * np.pi, 64), feat_rng.normal(size=64)

    def rough(s):
        return np.cos(((s - s_mean) / s_std) @ omega.T + phase) @ amp

    rough_scale = e_sd / rough(S0).std()
    drift_lin = np.zeros(sys_["n"])
    drift_feats = []  # (omega, phase, amplitude) of each iteration's rough increment

    def level_drift(s):
        z = (s - s_mean) / s_std
        out = z @ drift_lin
        for om, ph, am in drift_feats:
            out = out + am * np.cos(z @ om + ph)
        return out

    gt0 = truth.dqda(S0, mu0)
    align = (gt0 * mu0).sum() / np.sqrt((gt0 ** 2).sum() * (mu0 ** 2).sum())
    if verbose:
        print(f"  between-state sd {truth.value(S0).std():.2f}, within-state sd "
              f"{(truth.q(S0, A0) - truth.q(S0, mu0)).std():.3f}, sd(mu)/sigma {mu0.std() / sigma:.2f}, "
              f"cos(true dQ/da, mu(s)) {align:+.2f}")

    def tens(x):
        return torch.tensor(x, dtype=torch.float32)

    def norm_s(s):
        return (s - tens(s_mean)) / tens(s_std)

    torch.manual_seed(seed)
    qnet = mlp(sys_["n"] + sys_["d"])            # (s, a) coordinates
    vnet, gnet = mlp(sys_["n"]), mlp(sys_["n"], out=sys_["d"])  # orthogonal: V(s) + g(s).eps
    opt_q = torch.optim.Adam(qnet.parameters(), lr=3e-4)
    opt_o = torch.optim.Adam(list(vnet.parameters()) + list(gnet.parameters()), lr=3e-4)

    def q_hat(s, a):
        return qnet(torch.cat((norm_s(s), a / a_std), -1)).squeeze(-1) + q_mean

    def o_hat(s, a):
        eps = (a - s @ K_t.T) / sigma
        return vnet(norm_s(s)).squeeze(-1) + (gnet(norm_s(s)) * eps).sum(-1) * 0.1 + q_mean

    def evaluate(model, states=1024, fresh=8):
        with torch.no_grad():
            S, _ = rollout(truth, rng, envs=16, steps=states // 16)
            eps = rng.normal(size=(fresh,) + S.shape[:1] + (sys_["d"],))
            Sx = np.broadcast_to(S, (fresh,) + S.shape)
            Ax = Sx @ sys_["K"].T + sigma * eps
            qt = truth.q(Sx, Ax)
            qh = model(tens(Sx), tens(Ax)).numpy()
            nxt = truth.step(Sx, Ax)
            vh = 0.0
            for _ in range(8):  # V_hat(s') = E_a' Q_hat(s', a')
                an = nxt @ sys_["K"].T + sigma * rng.normal(size=Ax.shape)
                vh = vh + model(tens(nxt), tens(an)).numpy() / 8
            backup = truth.reward(Sx, Ax) + sys_["gamma"] * vh
            ct, ch, cb = (x - x.mean(0) for x in (qt, qh, backup))
            lvl = qt.mean(0) + rough_scale * rough(S) + level_drift(S)  # the level the critic is asked to fit
            level = np.corrcoef(lvl, qh.mean(0))[0, 1]
            level_err = (qh.mean(0) - lvl).std()
        # directional derivative at the mean action along random unit directions, h = 0.1 sigma-units * sigma
        Sg = tens(S)
        Ag = (Sg @ K_t.T).requires_grad_(True)
        (grad,) = torch.autograd.grad(model(Sg, Ag).sum(), Ag)
        gt = truth.dqda(S, S @ sys_["K"].T)
        gh = grad.numpy()
        slope = (gt * gh).sum() / (gh * gh).sum()  # regression (through 0) of true on critic's, all components
        gcos = (gt * gh).sum() / np.sqrt((gt * gt).sum() * (gh * gh).sum())
        return dict(level=level, level_err=level_err, true_sd=ct.std(), hat_sd=ch.std(),
                    contrast=np.corrcoef(ct.ravel(), ch.ravel())[0, 1],
                    self_cons=np.corrcoef(cb.ravel(), ch.ravel())[0, 1],
                    backup_true=np.corrcoef(cb.ravel(), ct.ravel())[0, 1], slope=slope, gcos=gcos)

    out = {}
    for it in range(iters):
        S, E = rollout(truth, rng)
        A = S @ sys_["K"].T + sigma * E
        if drift > 0:
            drift_lin = drift_lin + drift * np.sqrt(0.5 / sys_["n"]) * feat_rng.normal(size=sys_["n"])
            drift_feats.append((feat_rng.normal(size=sys_["n"]) * e_freq / np.sqrt(sys_["n"]),
                                feat_rng.uniform(0, 2 * np.pi), drift * feat_rng.choice([-1.0, 1.0])))
        y = truth.q(S, A) + rough_scale * rough(S) + level_drift(S) + nu * rng.normal(size=len(S))
        St, At, yt = tens(S), tens(A), tens(y)
        for _ in range(passes * len(S) // 2048):
            idx = torch.randint(0, len(S), (2048,))
            for model, opt in ((q_hat, opt_q), (o_hat, opt_o)):
                loss = (model(St[idx], At[idx]) - yt[idx]).square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        if it in (0, 3, 7, iters - 1):
            for name, model in (("Q(s,a)", q_hat), ("V+g.eps", o_hat)):
                r = evaluate(model)
                out[(it, name)] = r
                if verbose:
                    print(f"  it {it + 1:2d} {name:8s} level corr {r['level']:.3f} err sd {r['level_err']:.2f} | within sd "
                          f"true {r['true_sd']:.3f} hat {r['hat_sd']:.3f} | contrast corr {r['contrast']:+.3f} | "
                          f"true-on-critic slope {r['slope']:+.2f} cos {r['gcos']:+.2f} | "
                          f"self-consistency {r['self_cons']:+.3f} (backup vs truth {r['backup_true']:+.3f})")
    return out


if __name__ == "__main__":
    part_a()
    # (kappa, sigma, nu, e_sd, b_scale, proc, c_scale, drift)
    configs = [(0.5, 0.5, 0.0, 0.0, 1, 1, 1, 0.0), (0.5, 0.5, 0.6, 0.0, 1, 1, 1, 0.0), (0.5, 0.5, 0.6, 0.9, 1, 1, 1, 0.0),
               (1.5, 0.5, 0.6, 0.9, 1, 1, 1, 0.0), (0.5, 0.15, 0.6, 0.9, 1, 1, 1, 0.0),
               (0.5, 0.5, 0.6, 0.9, 1, 1, 1, 0.3), (0.5, 0.5, 0.6, 0.9, 1, 1, 1, 1.0), (0.5, 0.15, 0.6, 0.9, 1, 1, 1, 0.3)]
    if len(sys.argv) > 1:
        configs = [tuple(float(x) for x in arg.split(",")) for arg in sys.argv[1:]]
    for kappa, sigma, nu, e_sd, b_scale, proc, c_scale, drift in configs:
        print(f"\n== Part B: kappa {kappa} (gain / LQR gain), sigma {sigma}, iid target noise sd {nu}, rough state "
              f"component sd {e_sd}, B scale {b_scale}, process noise {proc}, linear reward scale {c_scale}, level drift/iter {drift} ==")
        run_b(kappa, sigma, nu, e_sd=e_sd, b_scale=b_scale, proc=proc, c_scale=c_scale, drift=drift)

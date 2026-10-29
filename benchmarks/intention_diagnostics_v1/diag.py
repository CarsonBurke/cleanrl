"""Offline diagnostics for a k-step "intention" design on the TD7 latent-Q delight v10 checkpoint.

Design under test: an actor picks g = a latent zs(s_{t+k}); a head pi(s, g) decodes actions; a
critic Q(s, g) values intentions. Stages (see README.md for results):

  C  mode spread: action spread among replay segments with near-identical (s_t, zs(s_{t+k})).
  A  off-reachable fraction: share of ||dQ/dzsa||^2 inside range(d zsa / da) at the actor mean.
  R  intention-ascent reliability: hindsight Q(s, g) and head pi(s, g, h) fit on replay, then
     realized/claimed value change of g* = g0 + eta * grad_g Q executed in MuJoCo from held-out
     replay states, against the same measurement for action-space ascent a* = mu + eta * grad_a Q.

HalfCheetah observations are qpos[1:] ++ qvel and the root x position does not enter dynamics or
reward (reward uses an x difference), so every replay state is restored exactly with qpos[0] = 0;
stage R verifies this against stored transitions before measuring anything.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cleanrl.td7.delight.td7_latentq_delight_reference_v10 import (  # noqa: E402
    Actor, Encoder, PreActionLatentTwinCritic,
)

HERE = Path(__file__).resolve().parent
CHECKPOINT = REPO / "runs/HalfCheetah-v4__td7_latentq_delight_reference_v10_latent_all__1__1791861435/checkpoint.pt"
NE, STEPS = 16, 1024
ROLLOUT = NE * STEPS
GAMMA = 0.99
SD, AD, ZD = 17, 6, 256
HMAX = 8
KS = (1, 4, 8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------------------------
class Checkpoint:
    def __init__(self, path, device):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["format"] == "td7_latentq_delight_reference_v10"
        models, agent = ck["agent"]["models"], ck["agent"]
        self.actor = Actor(SD, AD, ZD, 256)
        self.enc = Encoder(SD, AD, ZD, 256)
        self.critic = PreActionLatentTwinCritic(SD, AD, ZD, 256)
        self.actor.load_state_dict(models["actor"])
        self.enc.load_state_dict(models["fixed_encoder"])  # the encoder the actor and critic consume
        self.critic.load_state_dict(models["critic"])
        for m in (self.actor, self.enc, self.critic):
            m.to(device).eval().requires_grad_(False)
        self.storage = agent["storage"]
        self.global_step = int(ck["global_step"])
        self.device = device

    def zs(self, s):
        return self.enc.zs(s)

    def mu(self, s, zs=None):
        return self.actor(s, self.zs(s) if zs is None else zs)[0]

    def qp(self, s, a, zs=None, zsa=None):
        """Point head Q(s, a) (latent (a, 0), pre-action a), mean of twins, as the actor scores it."""
        zs = self.zs(s) if zs is None else zs
        zsa = self.enc.zsa(zs, a) if zsa is None else zsa
        return self.critic(s, torch.cat([a, torch.zeros_like(a)], 1), a, zsa, zs).mean(-1)

    def v(self, s):
        zs = self.zs(s)
        return self.qp(s, self.mu(s, zs), zs)


def chunked(fn, *xs, chunk=32768):
    outs = [fn(*(x[i: i + chunk] for x in xs)) for i in range(0, xs[0].shape[0], chunk)]
    return torch.cat(outs)


# ---------------------------------------------------------------------------------------------
# Replay window and hindsight segments
# ---------------------------------------------------------------------------------------------
class Window:
    """The most recent replay rollouts. Row = rollout*16384 + step*16 + env, so row+16 is the
    same env one step later (also across rollout boundaries); continuity is checked exactly."""

    def __init__(self, ck: Checkpoint, n_rollouts, holdout_rollouts):
        dev = ck.device
        w = ck.storage[-n_rollouts * ROLLOUT:].to(dev)
        self.n = n = w.shape[0]
        self.S, self.A, self.MU, self.SIG = w[:, :17], w[:, 17:23], w[:, 23:29], w[:, 29:35]
        self.NS, self.R = w[:, 47:64], w[:, 64]
        assert torch.all(w[:, 65] == 1), "HalfCheetah never terminates"
        idx = torch.arange(n, device=dev)
        cont = torch.zeros(n, dtype=torch.bool, device=dev)
        cont[:-NE] = (self.S[NE:] == self.NS[:-NE]).all(1)
        self.cont_fraction = float(cont[:-NE].float().mean())
        self.valid = {1: torch.ones(n, dtype=torch.bool, device=dev)}
        for h in range(2, HMAX + 1):
            ok = idx + NE * (h - 1) < n
            c = torch.zeros(n, dtype=torch.bool, device=dev)
            c[ok] = cont[(idx + NE * (h - 2))[ok]]
            self.valid[h] = self.valid[h - 1] & c
        self.holdout_start = n - holdout_rollouts * ROLLOUT
        self.train = {h: self.valid[h] & (idx + NE * (h - 1) < self.holdout_start) for h in self.valid}
        self.hold = {h: self.valid[h] & (idx >= self.holdout_start) for h in self.valid}
        self.env, self.time = idx % NE, idx // NE
        with torch.no_grad():
            self.ZS = chunked(ck.zs, self.S)
            self.ZSN = chunked(ck.zs, self.NS)
            self.VN = chunked(ck.v, self.NS)
        self.idx = idx

    def last_row(self, i, h):
        return i + NE * (h - 1)

    def g(self, i, h):
        return self.ZSN[self.last_row(i, h)]

    def y(self, i, h):
        """k-step return along the replay segment plus gamma^h V(s_{t+h}) from the checkpoint."""
        ret = torch.zeros(i.shape[0], device=i.device)
        for j in range(h):
            ret = ret + GAMMA ** j * self.R[i + NE * j]
        return ret + GAMMA ** h * self.VN[self.last_row(i, h)]

    def rows(self, mask):
        return self.idx[mask]


# ---------------------------------------------------------------------------------------------
# Stage C: mode spread
# ---------------------------------------------------------------------------------------------
def stage_c(ck, win, args, out):
    dev = ck.device
    rng = torch.Generator(device="cpu").manual_seed(0)
    base = win.rows(win.train[HMAX])  # same database/query rows for every k
    db = base
    q = base[torch.randperm(base.shape[0], generator=rng)[: args.c_queries].to(dev)]
    s_mu, s_sd = win.S[db].mean(0), win.S[db].std(0).clamp(min=1e-6)
    a_db = win.A[db]
    total_var = float(a_db.var(0).sum())
    mu_total_var = float(win.MU[db].var(0).sum())
    feats_s = (win.S - s_mu) / s_sd / SD ** 0.5
    res = {"database_rows": int(db.shape[0]), "queries": int(q.shape[0]), "action_total_var": total_var,
           "policy_mean_total_var": mu_total_var,
           "metric": "within-neighbourhood action variance (trace) / database action variance (trace)",
           "space": "per-dim z-scored s (block /sqrt(17)) ++ per-dim z-scored g=zs(s_{t+k}) (block /sqrt(256)); "
                    "k=0 is the s-only reference; neighbours from the same env within 10 steps are excluded",
           "by_k": {}}
    Kmax = 32
    for k in (0,) + KS:
        if k == 0:
            feat = feats_s
        else:
            g = win.g(win.idx.clamp(max=win.n - 1 - NE * (k - 1)), k)
            g_mu, g_sd = g[db].mean(0), g[db].std(0)
            g_sd = g_sd.clamp(min=1e-3 * float(g_sd.mean()))
            feat = torch.cat([feats_s, (g - g_mu) / g_sd / ZD ** 0.5], 1)
        F_db = feat[db]
        nbr = torch.empty(q.shape[0], Kmax, dtype=torch.long, device=dev)
        dist = torch.empty(q.shape[0], Kmax, device=dev)
        for c in range(0, q.shape[0], 256):
            qi = q[c: c + 256]
            d = torch.cdist(feat[qi], F_db)
            excl = (win.env[qi][:, None] == win.env[db][None]) & ((win.time[qi][:, None] - win.time[db][None]).abs() <= 10)
            d[excl] = float("inf")
            dd, jj = d.topk(Kmax, largest=False)
            nbr[c: c + 256], dist[c: c + 256] = db[jj], dd
        pair = torch.randint(0, db.shape[0], (20000, 2), generator=rng).to(dev)
        rand_d = float((F_db[pair[:, 0]] - F_db[pair[:, 1]]).norm(dim=1).median())
        rk = {"median_random_pair_distance": rand_d, "by_K": {}}
        for K in (4, 8, 32):
            an = win.A[nbr[:, :K]]
            spread = an.var(1, unbiased=True).sum(-1)
            mspread = win.MU[nbr[:, :K]].var(1, unbiased=True).sum(-1)
            radius = dist[:, K - 1]
            tight = radius <= radius.quantile(0.1)
            boot = spread[torch.randint(0, spread.shape[0], (2000, spread.shape[0]), generator=rng).to(dev)].mean(1)
            rk["by_K"][K] = {
                "radius_median": float(radius.median()),
                "radius_over_random_pair": float(radius.median()) / rand_d,
                "spread_ratio": float(spread.mean()) / total_var,
                "spread_ratio_ci95": [float(boot.quantile(0.025)) / total_var, float(boot.quantile(0.975)) / total_var],
                "spread_ratio_tightest_decile": float(spread[tight].mean()) / total_var,
                "tightest_decile_radius_median": float(radius[tight].median()),
                "policy_mean_spread_ratio": float(mspread.mean()) / total_var,
            }
        res["by_k"][k] = rk
        log(f"C k={k}: " + ", ".join(f"K{K} spread={v['spread_ratio']:.3f} tight={v['spread_ratio_tightest_decile']:.3f} "
                                       f"r/rand={v['radius_over_random_pair']:.2f}" for K, v in rk["by_K"].items()))
    out["C"] = res


# ---------------------------------------------------------------------------------------------
# Stage A: off-reachable fraction
# ---------------------------------------------------------------------------------------------
def model_jacobian(ck, zs, a):
    cols = []
    for i in range(AD):
        t = torch.zeros_like(a)
        t[:, i] = 1
        cols.append(ck.enc.zsa_jvp(zs, a, t)[1])
    return torch.stack(cols, -1)  # (B, 256, 6)


def critic_zsa_grad(ck, s, a, zs):
    zsa = ck.enc.zsa(zs, a).detach().requires_grad_(True)
    with torch.enable_grad():
        q = ck.qp(s, a, zs, zsa)
        return torch.autograd.grad(q.sum(), zsa)[0]


def range_fraction(J, v):
    Q, _ = torch.linalg.qr(J)
    p = torch.einsum("bdk,bd->bk", Q, v)
    return p.square().sum(-1) / v.square().sum(-1).clamp(min=1e-30)


def mean_ci(x, B=2000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def stage_a(ck, win, args, out):
    dev = ck.device
    g = torch.Generator(device="cpu").manual_seed(1)
    rows = win.idx[torch.randperm(win.n, generator=g)[: args.a_states].to(dev)]
    s = win.S[rows]
    zs = ck.zs(s)
    mu = ck.mu(s, zs)
    J = model_jacobian(ck, zs, mu)
    gz = critic_zsa_grad(ck, s, mu, zs)
    frac = range_fraction(J, gz)
    a_leaf = mu.clone().requires_grad_(True)
    with torch.enable_grad():
        ga_total = torch.autograd.grad(ck.qp(s, a_leaf, zs).sum(), a_leaf)[0]
    ga_zsa = torch.einsum("bdk,bd->bk", J, gz)
    sv = torch.linalg.svdvals(J)
    # Baselines: random isotropic directions and the replay zs-manifold principal subspaces.
    rand = torch.randn_like(gz)
    zn = win.ZSN[win.train[1]]
    zc = zn - zn.mean(0)
    _, _, Vh = torch.linalg.svd(zc[torch.randperm(zc.shape[0], generator=g)[:50000].to(dev)], full_matrices=False)
    pcs = {}
    for d in (6, 16, 32):
        P = Vh[:d]
        pcs[d] = mean_ci(((gz @ P.T).square().sum(-1) / gz.square().sum(-1)).cpu())
    zsa_rep = chunked(lambda s_, a_: ck.enc.zsa(ck.zs(s_), a_), win.S[win.hold[1]], win.A[win.hold[1]])
    tgt = win.ZSN[win.hold[1]]
    zsa_r2 = 1 - float((zsa_rep - tgt).square().sum() / (tgt - tgt.mean(0)).square().sum())
    res = {
        "states": int(rows.shape[0]),
        "fraction_dQdzsa_in_rangeJ": mean_ci(frac.cpu()),
        "fraction_median": float(frac.median()),
        "fraction_quantiles_10_50_90": [float(x) for x in frac.quantile(torch.tensor([0.1, 0.5, 0.9], device=dev))],
        "random_direction_fraction": mean_ci(range_fraction(J, rand).cpu()),
        "isotropic_expectation": AD / ZD,
        "fraction_in_zs_manifold_top_pcs": pcs,
        "J_singular_values_median": [float(x) for x in sv.median(0).values],
        "J_condition_median": float((sv[:, 0] / sv[:, -1]).median()),
        "dQda_zsa_path_norm_over_total": mean_ci((ga_zsa.norm(dim=1) / ga_total.norm(dim=1)).cpu()),
        "dQda_zsa_path_cosine_with_total": mean_ci(F.cosine_similarity(ga_zsa, ga_total, dim=1).cpu()),
        "zsa_predicts_next_zs_R2_holdout": zsa_r2,
        "actor_mean_saturated_fraction(|mu|>0.99)": float((mu.abs() > 0.99).float().mean()),
    }
    log(f"A: frac in range(J) = {res['fraction_dQdzsa_in_rangeJ']}, random = {res['random_direction_fraction'][0]:.4f}, "
        f"zsa-path/total |dQ/da| = {res['dQda_zsa_path_norm_over_total'][0]:.3f}")
    out["A"] = res


# ---------------------------------------------------------------------------------------------
# Fitted models for stage R
# ---------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, i, o, hid=512, depth=3):
        super().__init__()
        mods, d = [], i
        for _ in range(depth):
            mods += [nn.Linear(d, hid), nn.LayerNorm(hid), nn.SiLU()]
            d = hid
        mods.append(nn.Linear(d, o))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class Stats(nn.Module):
    def __init__(self, win, rows):
        super().__init__()
        self.register_buffer("s_mu", win.S[rows].mean(0))
        self.register_buffer("s_sd", win.S[rows].std(0).clamp(min=1e-6))
        self.register_buffer("z_mu", win.ZS[rows].mean(0))
        self.register_buffer("z_sd", win.ZS[rows].std(0).clamp(min=1e-3))
        d_sd = torch.ones(HMAX + 1, ZD, device=rows.device)
        for h in range(1, HMAX + 1):
            r = win.rows(win.train[h])
            d_sd[h] = (win.g(r, h) - win.ZS[r]).std(0).clamp(min=1e-3)
        self.register_buffer("d_sd", d_sd)

    def base(self, s, zs):
        return [(s - self.s_mu) / self.s_sd, (zs - self.z_mu) / self.z_sd]

    def goal(self, zs, g, h):
        return [(g - self.z_mu) / self.z_sd, (g - zs) / self.d_sd[h]]


class IntentionQ(nn.Module):
    def __init__(self, stats, h, y_mu, y_sd):
        super().__init__()
        self.stats, self.h = stats, h
        self.net = MLP(SD + 3 * ZD, 1)
        self.y_mu, self.y_sd = y_mu, y_sd

    def forward(self, s, zs, g):
        h = torch.full((s.shape[0],), self.h, device=s.device, dtype=torch.long)
        x = torch.cat(self.stats.base(s, zs) + self.stats.goal(zs, g, h), 1)
        return self.net(x).squeeze(-1) * self.y_sd + self.y_mu


class ActionQ(nn.Module):
    def __init__(self, stats, y_mu, y_sd):
        super().__init__()
        self.stats = stats
        self.net = MLP(SD + ZD + AD, 1)
        self.y_mu, self.y_sd = y_mu, y_sd

    def forward(self, s, zs, a):
        return self.net(torch.cat(self.stats.base(s, zs) + [a], 1)).squeeze(-1) * self.y_sd + self.y_mu


class Head(nn.Module):
    """pi(s, g, h): first action toward g reached h steps later. mse: point; mdn: K-Gaussian mixture."""

    def __init__(self, stats, kind, K=5):
        super().__init__()
        self.stats, self.kind, self.K = stats, kind, K
        self.net = MLP(SD + 3 * ZD + HMAX, AD if kind == "mse" else K * (1 + 2 * AD))

    def raw(self, s, zs, g, h):
        x = torch.cat(self.stats.base(s, zs) + self.stats.goal(zs, g, h) + [F.one_hot(h - 1, HMAX).float()], 1)
        return self.net(x)

    def mixture(self, out):
        K = self.K
        logits = out[:, :K]
        means = out[:, K: K + K * AD].view(-1, K, AD)
        log_std = out[:, K + K * AD:].view(-1, K, AD).clamp(-3.0, 1.0)
        return logits, means, log_std

    def loss(self, s, zs, g, h, a):
        out = self.raw(s, zs, g, h)
        if self.kind == "mse":
            return (out - a).square().sum(-1).mean()
        logits, means, log_std = self.mixture(out)
        lp = (-0.5 * ((a[:, None] - means) / log_std.exp()).square() - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        return -torch.logsumexp(F.log_softmax(logits, -1) + lp, -1).mean()

    def act(self, s, zs, g, h):
        out = self.raw(s, zs, g, h)
        if self.kind == "mse":
            return out.clamp(-1, 1)
        logits, means, _ = self.mixture(out)
        return means[torch.arange(out.shape[0], device=out.device), logits.argmax(-1)].clamp(-1, 1)


def fit(model, batch_fn, steps, lr=1e-3, wd=1e-4, name=""):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    for step in range(steps):
        loss = batch_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            log(f"fit {name} step {step} batch loss {float(loss):.4f}")
    model.eval()
    model.requires_grad_(False)
    return model


def sample(rows, n, gen):
    return rows[torch.randint(0, rows.shape[0], (n,), device=rows.device, generator=gen)]


def fit_models(ck, win, args):
    dev = ck.device
    gen = torch.Generator(device=dev).manual_seed(2)
    stats = Stats(win, win.rows(win.train[1])).to(dev)
    models, quality = {}, {}
    B = args.batch
    for k in KS:
        tr, ho = win.rows(win.train[k]), win.rows(win.hold[k])
        y_tr = win.y(tr, k)
        y_mu, y_sd = float(y_tr.mean()), float(y_tr.std())
        m = IntentionQ(stats, k, y_mu, y_sd).to(dev)

        def batch_fn(m=m, tr=tr, k=k, y_sd=y_sd):
            i = sample(tr, B, gen)
            return ((m(win.S[i], win.ZS[i], win.g(i, k)) - win.y(i, k)) / y_sd).square().mean()

        models[f"QI{k}"] = fit(m, batch_fn, args.fit_steps, name=f"QI{k}")
        with torch.no_grad():
            pred = m(win.S[ho], win.ZS[ho], win.g(ho, k))
            yh = win.y(ho, k)
            quality[f"QI{k}_holdout_R2"] = 1 - float((pred - yh).square().mean() / yh.var())
            quality[f"QI{k}_holdout_rmse"] = float((pred - yh).square().mean().sqrt())
            quality[f"QI{k}_y_sd"] = float(yh.std())
    tr, ho = win.rows(win.train[1]), win.rows(win.hold[1])
    y_tr = win.y(tr, 1)
    qa = ActionQ(stats, float(y_tr.mean()), float(y_tr.std())).to(dev)

    def batch_qa():
        i = sample(tr, B, gen)
        return ((qa(win.S[i], win.ZS[i], win.A[i]) - win.y(i, 1)) / qa.y_sd).square().mean()

    models["QA"] = fit(qa, batch_qa, args.fit_steps, name="QA")
    with torch.no_grad():
        yh = win.y(ho, 1)
        quality["QA_holdout_R2"] = 1 - float((qa(win.S[ho], win.ZS[ho], win.A[ho]) - yh).square().mean() / yh.var())
        quality["ckpt_Qp_vs_y1_holdout_R2"] = 1 - float((chunked(ck.qp, win.S[ho], win.A[ho]) - yh).square().mean() / yh.var())
    train_rows = {h: win.rows(win.train[h]) for h in range(1, HMAX + 1)}
    for kind in ("mse", "mdn"):
        head = Head(stats, kind).to(dev)

        def batch_head(head=head):
            hs = torch.randint(1, HMAX + 1, (B,), device=dev, generator=gen)
            i = torch.cat([sample(train_rows[h], int((hs == h).sum()), gen) for h in range(1, HMAX + 1)])
            hs = torch.sort(hs).values  # rows were drawn grouped by h in ascending order
            g = win.ZSN[i + NE * (hs - 1)]
            return head.loss(win.S[i], win.ZS[i], g, hs, win.A[i])

        models[f"head_{kind}"] = fit(head, batch_head, args.fit_steps, name=f"head_{kind}")
    with torch.no_grad():
        for k in KS:
            ho = win.rows(win.hold[k])
            hk = torch.full((ho.shape[0],), k, device=dev, dtype=torch.long)
            s, zs, g, a = win.S[ho], win.ZS[ho], win.g(ho, k), win.A[ho]
            avar = float(a.var(0).sum())
            mse_err = float((models["head_mse"].act(s, zs, g, hk) - a).square().sum(-1).mean())
            mdn = models["head_mdn"]
            out = mdn.raw(s, zs, g, hk)
            logits, means, _ = mdn.mixture(out)
            mode = means[torch.arange(ho.shape[0], device=dev), logits.argmax(-1)].clamp(-1, 1)
            best = (means.clamp(-1, 1) - a[:, None]).square().sum(-1).min(-1).values
            resid = models["head_mse"].act(s, zs, g, hk) - a
            gauss_nll = float(0.5 * (torch.log(2 * np.pi * resid.var(0)) + 1).sum())
            quality[f"head_k{k}"] = {
                "action_total_var": avar,
                "mse_head_err_over_var": mse_err / avar,
                "mdn_mode_err_over_var": float((mode - a).square().sum(-1).mean()) / avar,
                "mdn_best_of_K_err_over_var": float(best.mean()) / avar,
                "mdn_nll": float(mdn.loss(s, zs, g, hk, a)),
                "mse_head_global_gaussian_nll": gauss_nll,
                "mdn_mean_max_weight": float(F.softmax(logits, -1).max(-1).values.mean()),
            }
    log("fit quality: " + json.dumps(quality, default=float)[:1500])
    return models, quality


# ---------------------------------------------------------------------------------------------
# MuJoCo restoration and execution
# ---------------------------------------------------------------------------------------------
class Sim:
    def __init__(self, n):
        env = gym.make("HalfCheetah-v4").unwrapped
        self.model, self.fs, self.dt = env.model, env.frame_skip, env.dt
        self.datas = [mujoco.MjData(self.model) for _ in range(n)]
        self.n = n

    def set(self, obs):
        for d, o in zip(self.datas, obs):
            mujoco.mj_resetData(self.model, d)
            d.qpos[0] = 0.0
            d.qpos[1:] = o[:8]
            d.qvel[:] = o[8:]
            mujoco.mj_forward(self.model, d)

    def step(self, actions):
        obs = np.empty((self.n, SD), dtype=np.float32)
        rew = np.empty(self.n, dtype=np.float64)
        for i, (d, a) in enumerate(zip(self.datas, actions)):
            x0 = d.qpos[0]
            d.ctrl[:] = a
            mujoco.mj_step(self.model, d, nstep=self.fs)
            rew[i] = (d.qpos[0] - x0) / self.dt - 0.1 * float(np.square(a).sum())
            obs[i, :8] = d.qpos[1:]
            obs[i, 8:] = d.qvel
        return obs, rew


def verify_restoration(win, sim, n):
    g = torch.Generator(device="cpu").manual_seed(3)
    rows = win.rows(win.train[1])[torch.randperm(int(win.train[1].sum()), generator=g)[:n].to(win.S.device)]
    sim.set(win.S[rows].cpu().numpy())
    obs, rew = sim.step(win.A[rows].cpu().numpy().astype(np.float64))
    ns, r = win.NS[rows].cpu().numpy(), win.R[rows].cpu().numpy()
    return {"rows": n, "next_obs_max_abs_err": float(np.abs(obs - ns).max()),
            "next_obs_median_abs_err": float(np.median(np.abs(obs - ns))),
            "reward_max_abs_err": float(np.abs(rew - r).max())}


def rollout(ck, sim, obs0, first_k_policy, k, L):
    """first_k_policy(j, obs_t) for j < k, then the actor mean for L steps. Returns dict of
    G_cut = sum_{j<cut} gamma^j r_j + gamma^cut V(s_cut) for cut in {k, k+L}, zs(s_k)."""
    dev = ck.device
    sim.set(obs0)
    obs_t = torch.from_numpy(obs0).to(dev)
    ret = np.zeros(sim.n)
    out = {}
    for j in range(k + L):
        with torch.no_grad():
            a = first_k_policy(j, obs_t) if j < k else ck.mu(obs_t)
        obs, rew = sim.step(a.double().cpu().numpy())
        ret += GAMMA ** j * rew
        obs_t = torch.from_numpy(obs).to(dev)
        if j + 1 in (k, k + L):
            with torch.no_grad():
                out[f"G{j + 1 - k}"] = ret + GAMMA ** (j + 1) * ck.v(obs_t).double().cpu().numpy()
                if j + 1 == k:
                    out["zs_k"] = ck.zs(obs_t)
                    out["obs_k"] = obs_t
    return out


def ratio_stats(realized, claimed, B=2000, seed=0):
    realized, claimed = np.asarray(realized, np.float64), np.asarray(claimed, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(realized), (B, len(realized)))
    rm, cm = realized[idx].mean(1), claimed[idx].mean(1)
    slope = lambda r, c: (r * c).sum(-1) / (c * c).sum(-1)
    sb = slope(realized[idx], claimed[idx])
    return {"rho": float(realized.mean() / claimed.mean()),
            "rho_ci95": [float(np.percentile(rm / cm, 2.5)), float(np.percentile(rm / cm, 97.5))],
            "slope_through_origin": float(slope(realized, claimed)),
            "slope_ci95": [float(np.percentile(sb, 2.5)), float(np.percentile(sb, 97.5))],
            "claimed_mean": float(claimed.mean()), "claimed_ci95": [float(np.percentile(cm, 2.5)), float(np.percentile(cm, 97.5))],
            "realized_mean": float(realized.mean()), "realized_ci95": [float(np.percentile(rm, 2.5)), float(np.percentile(rm, 97.5))],
            "positive_realized_fraction": float((realized > 0).mean())}


def true_jacobian(ck, sim, obs0, mu, delta=1e-2):
    """Finite-difference d zs(step(s, a)) / da at a = mu from MuJoCo (one-sided inward at the bound)."""
    cols = []
    mu_np = mu.double().cpu().numpy()
    for i in range(AD):
        hi, lo = mu_np.copy(), mu_np.copy()
        hi[:, i] = np.minimum(mu_np[:, i] + delta, 1.0)
        lo[:, i] = np.maximum(mu_np[:, i] - delta, -1.0)
        sim.set(obs0)
        oh, _ = sim.step(hi)
        sim.set(obs0)
        ol, _ = sim.step(lo)
        with torch.no_grad():
            zh, zl = ck.zs(torch.from_numpy(oh).to(ck.device)), ck.zs(torch.from_numpy(ol).to(ck.device))
        width = torch.from_numpy(hi[:, i] - lo[:, i]).float().to(ck.device).clamp(min=1e-6)
        cols.append((zh - zl) / width[:, None])
    return torch.stack(cols, -1)


def stage_r(ck, win, args, out):
    dev = ck.device
    models, quality = fit_models(ck, win, args)
    torch.save({k: v.state_dict() for k, v in models.items()}, HERE / "results" / "models.pt")
    out["R_fit_quality"] = quality
    n = args.r_states
    sim = Sim(n)
    out["R_restoration_check"] = verify_restoration(win, sim, n)
    log("restoration:", out["R_restoration_check"])
    g = torch.Generator(device="cpu").manual_seed(4)
    hold = win.rows(win.hold[1])
    rows = hold[torch.randperm(hold.shape[0], generator=g)[:n].to(dev)]
    obs0 = win.S[rows].cpu().numpy()
    s = win.S[rows]
    zs = ck.zs(s)
    mu = ck.mu(s, zs)
    L = args.tail
    res = {"states": n, "tail_steps_L": L, "eta_action": args.eta_action, "eta_intention_over_Dk": args.eta_goal}

    # Actor reference: the greedy actor for HMAX + L steps; zs(s_k) of it is the actor's own g0.
    actor_ref = {k: rollout(ck, sim, obs0, lambda j, o: ck.mu(o), k, L) for k in KS}

    # Action-space ascent (k = 1), checkpoint critic and a from-scratch ActionQ fit like QI1.
    def action_ascent(qfn, name):
        a_leaf = mu.clone().requires_grad_(True)
        with torch.enable_grad():
            grad = torch.autograd.grad(qfn(a_leaf).sum(), a_leaf)[0]
        u = grad / grad.norm(dim=1, keepdim=True).clamp(min=1e-12)
        table = {}
        for eta in args.eta_action:
            a_star = (mu + eta * u).clamp(-1, 1)
            with torch.no_grad():
                claimed = (qfn(a_star) - qfn(mu)).double().cpu().numpy()
            r = rollout(ck, sim, obs0, lambda j, o, a_star=a_star: a_star, 1, L)
            table[eta] = {f"tail{c}": ratio_stats(r[f"G{c}"] - actor_ref[1][f"G{c}"], claimed) for c in (0, L)}
            log(f"R action[{name}] eta={eta}: rho(tail0)={table[eta]['tail0']['rho']:.3f} "
                f"{table[eta]['tail0']['rho_ci95']} rho(tail{L})={table[eta][f'tail{L}']['rho']:.3f} claimed={table[eta]['tail0']['claimed_mean']:.3f}")
        return table

    res["action_ckpt_critic"] = action_ascent(lambda a: ck.qp(s, a, zs), "ckpt")
    res["action_fitted_QA"] = action_ascent(lambda a: models["QA"](s, zs, a), "QA")

    # Intention ascent.
    D = {}
    for k in KS:
        tr = win.rows(win.train[k])
        D[k] = float((win.g(tr, k) - win.ZS[tr]).norm(dim=1).median())
    res["D_k_median_latent_displacement"] = D
    res["intention"] = {}
    for k in KS:
        qi = models[f"QI{k}"]
        g0 = actor_ref[k]["zs_k"]
        g_leaf = g0.clone().requires_grad_(True)
        with torch.enable_grad():
            grad = torch.autograd.grad(qi(s, zs, g_leaf).sum(), g_leaf)[0]
        u = grad / grad.norm(dim=1, keepdim=True).clamp(min=1e-12)
        res["intention"][k] = {}
        for kind in ("mse", "mdn"):
            head = models[f"head_{kind}"]

            def policy(g_target):
                def f(j, o):
                    h = torch.full((o.shape[0],), k - j, device=dev, dtype=torch.long)
                    return head.act(o, ck.zs(o), g_target, h)
                return f

            ref = rollout(ck, sim, obs0, policy(g0), k, L)
            base_err = (ref["zs_k"] - g0).norm(dim=1)
            table = {"eta0_execution_error_over_Dk": mean_ci((base_err / D[k]).cpu()),
                     "head_at_g0_minus_actor_G": {f"tail{c}": mean_ci(ref[f"G{c}"] - actor_ref[k][f"G{c}"]) for c in (0, L)},
                     "by_eta": {}}
            for eta in args.eta_goal:
                step = eta * D[k]
                g_star = g0 + step * u
                with torch.no_grad():
                    claimed = (qi(s, zs, g_star) - qi(s, zs, g0)).double().cpu().numpy()
                r = rollout(ck, sim, obs0, policy(g_star), k, L)
                err = (r["zs_k"] - g_star).norm(dim=1) / step
                moved = r["zs_k"] - ref["zs_k"]
                along = (moved * u).sum(1) / step
                ortho = (moved - (moved * u).sum(1, keepdim=True) * u).norm(dim=1) / step
                entry = {f"tail{c}": ratio_stats(r[f"G{c}"] - ref[f"G{c}"], claimed) for c in (0, L)}
                entry["execution_error_over_step"] = mean_ci(err.cpu())
                entry["execution_error_over_step_median"] = float(err.median())
                entry["tracking_along_step"] = mean_ci(along.cpu())
                entry["off_direction_motion_over_step"] = mean_ci(ortho.cpu())
                table["by_eta"][eta] = entry
                log(f"R intent k={k} {kind} eta={eta}: rho(tail0)={entry['tail0']['rho']:.3f} {entry['tail0']['rho_ci95']} "
                    f"rho(tail{L})={entry[f'tail{L}']['rho']:.3f} claimed={entry['tail0']['claimed_mean']:.3f} "
                    f"err/step={entry['execution_error_over_step'][0]:.2f} along={entry['tracking_along_step'][0]:.2f}")
            res["intention"][k][kind] = table

    # A for the fitted intention critic, with the model Jacobian and the MuJoCo Jacobian.
    J_model = model_jacobian(ck, zs, mu)
    J_true = true_jacobian(ck, sim, obs0, mu)
    g0 = actor_ref[1]["zs_k"]
    g_leaf = g0.clone().requires_grad_(True)
    with torch.enable_grad():
        gq = torch.autograd.grad(models["QI1"](s, zs, g_leaf).sum(), g_leaf)[0]
    gz = critic_zsa_grad(ck, s, mu, zs)
    Qm, _ = torch.linalg.qr(J_model)
    Qt, _ = torch.linalg.qr(J_true)
    cosines = torch.linalg.svdvals(Qm.transpose(1, 2) @ Qt)  # cosines of principal angles
    res["A_reachability"] = {
        "QI1_grad_in_range_J_model": mean_ci(range_fraction(J_model, gq).cpu()),
        "QI1_grad_in_range_J_mujoco": mean_ci(range_fraction(J_true, gq).cpu()),
        "ckpt_dQdzsa_in_range_J_mujoco": mean_ci(range_fraction(J_true, gz).cpu()),
        "ckpt_dQdzsa_in_range_J_model": mean_ci(range_fraction(J_model, gz).cpu()),
        "principal_angle_cosines_model_vs_mujoco_median": [float(x) for x in cosines.median(0).values],
        "J_mujoco_singular_values_median": [float(x) for x in torch.linalg.svdvals(J_true).median(0).values],
    }
    log("A reachability:", json.dumps(res["A_reachability"])[:800])
    out["R"] = res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="C,A,R")
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--window-rollouts", type=int, default=20)
    p.add_argument("--holdout-rollouts", type=int, default=2)
    p.add_argument("--c-queries", type=int, default=5000)
    p.add_argument("--a-states", type=int, default=20000)
    p.add_argument("--r-states", type=int, default=4000)
    p.add_argument("--fit-steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--tail", type=int, default=16)
    p.add_argument("--eta-action", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
    p.add_argument("--eta-goal", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--out", default="")
    args = p.parse_args()
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    (HERE / "results").mkdir(exist_ok=True)
    ck = Checkpoint(args.checkpoint, dev)
    win = Window(ck, args.window_rollouts, args.holdout_rollouts)
    out = {"checkpoint": str(Path(args.checkpoint).relative_to(REPO)), "checkpoint_global_step": ck.global_step,
           "window_rows": win.n, "window_first_global_step_approx": ck.global_step - win.n,
           "holdout_rows": win.n - win.holdout_start, "continuity_fraction": win.cont_fraction,
           "args": vars(args)}
    log(f"window rows={win.n} continuity={win.cont_fraction:.4f} train(h=8)={int(win.train[8].sum())}")
    stages = args.stages.split(",")
    with torch.no_grad():
        if "C" in stages:
            stage_c(ck, win, args, out)
        if "A" in stages:
            stage_a(ck, win, args, out)
    if "R" in stages:
        stage_r(ck, win, args, out)
    path = Path(args.out) if args.out else HERE / "results" / f"stages_{'_'.join(stages)}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    log("wrote", path)


if __name__ == "__main__":
    main()

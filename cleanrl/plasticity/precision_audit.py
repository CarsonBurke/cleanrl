"""What this conflict harness can and cannot resolve, and whether it can express
per-perceptron autonomy at all.

Two questions, both about the HARNESS rather than about any mechanism. They are
asked because the precision headline (4 seeds: precision 0.5982 vs adam 0.6258)
inverted at 8 seeds (precision 0.6530 vs adam 0.6201), i.e. a 4.4% "win" and a
5.3% "loss" were drawn from the same distribution.

# Q1  RESOLUTION

What effect size can this task resolve, and with how many seeds? Answered three
ways, all on the same data:

  * the seed-to-seed SD of `overall` per arm, and of the PAIRED difference;
  * the minimum detectable effect (MDE) at 95%/80% for n = 4..64;
  * a bootstrap of the ENTIRE reported procedure (sweep LRs, take each arm's
    best, report the gap) using two arms that are the same mechanism, so the
    true gap is exactly zero. That gives the false-positive rate of the
    published protocol directly, including the winner's-curse component of
    per-arm LR selection that an MDE calculation misses.

It also separates the two variance sources the published protocol conflates:
`novelty_stream.py` seeds only the weight INITIALISATION (one generator, one
teacher draw, one centre draw, one shared input stream; `configs` carries a
`"seed"` field that `run()` never reads). `--no-resample-task --share-stream`
reproduces that protocol here; the default resamples teachers, centres, stream,
noise and init.

# Q2  AUTONOMY

Does shift-12 centre dominance mean this task structurally cannot exhibit
per-unit autonomy?

The structural fact: in a fully-connected layer every unit's incoming row sees
the SAME input vector. The mechanism's rank-1 commitment direction is
`d_t = mean_batch(x)`, identical for every unit, so

    C_h = sum_t lambda * c_{h,t} * d_t d_t^T

with the DIRECTIONS `d_t` shared across all H units and only the scalar
commitments `c_{h,t}` per unit. So per-unit anisotropy can only ever be a
per-unit REWEIGHTING of one shared direction set. If that set is effectively
rank ~1 per region -- which is what a large shift buys, since
`|centre| >> |noise|/sqrt(batch)` -- then every unit damps the same one
direction and the per-unit content degenerates exactly into a per-unit SCALAR.
That is a prediction with three testable consequences, all measured here:

  1. cross-unit |cos| between most-damped directions -> 1 as shift grows;
  2. `precision` (per-unit) should not beat `precision_shared` (one layer-wide
     geometry), because the per-unit part carries nothing;
  3. `precision` should not beat `precision_scalar`, its own matched isotropic
     shadow, which reads the identical per-unit per-sample realized plasticity
     and applies it as a scalar with the rotation removed.

Reported alongside: effective rank (participation ratio) of the accumulated
direction set, |cos| of each unit's most-damped direction with the region
centres, eigenvalue anisotropy of the geometry, and the cross-unit dispersion of
realized plasticity -- which is the only number that says whether a per-unit
decision was made at all.

# ARMS

`adam`               plain Adam.
`precision`          per-unit geometry, step1 <- (I + lambda C_h)^-1 step1.
`precision_shared`   ONE layer-wide geometry (commitment averaged across units).
`precision_diag`     off-diagonals zeroed every step.
`precision_scalar`   matched isotropic shadow: identical geometry, identical
                     per-unit per-sample realized plasticity rho_{h,t}, applied
                     as a scalar on the UNPROJECTED step. Anisotropy removed,
                     nothing else changed.
`precision_rotate`   the complement: projected direction renormalised back to
                     the Adam step norm. Rotation only, no level change.
`adam_sched`         Adam whose global LR is multiplied step by step by a
                     supplied realized-plasticity trajectory.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro

GEOM_NONE, GEOM_FULL, GEOM_DIAG = 0, 1, 2
APPLY_NONE, APPLY_PROJ, APPLY_SCALAR, APPLY_ROTATE = 0, 1, 2, 3

# arm -> (geometry, application, layer-wide geometry, follows the LR schedule)
ARMS = {
    "adam": (GEOM_NONE, APPLY_NONE, False, False),
    "adam_sched": (GEOM_NONE, APPLY_NONE, False, True),
    "precision": (GEOM_FULL, APPLY_PROJ, False, False),
    "precision_shared": (GEOM_FULL, APPLY_PROJ, True, False),
    "precision_diag": (GEOM_DIAG, APPLY_PROJ, False, False),
    "precision_scalar": (GEOM_FULL, APPLY_SCALAR, False, False),
    "precision_rotate": (GEOM_FULL, APPLY_ROTATE, False, False),
}

# The CEILING arms. A per-unit gate cannot do better than knowing, for each unit
# on each sample, whether its own pending step reduces the loss on the whole
# region mixture. `oracle_unit` is given exactly that (non-causal, so it bounds
# every causal rule), and is shipped with the two controls that keep only the
# level: `oracle_mean` matches the mean gate with zero cross-unit dispersion,
# `oracle_shuffled` matches the mean AND the dispersion exactly and destroys
# only the state->unit correspondence. If the oracle ties its shuffled control,
# the task has NO per-unit headroom and no rule of this shape can ever win here.
ORACLE_NONE, ORACLE_UNIT, ORACLE_MEAN, ORACLE_SHUFFLED = 0, 1, 2, 3
ORACLE = {"oracle_unit": ORACLE_UNIT, "oracle_mean": ORACLE_MEAN,
          "oracle_shuffled": ORACLE_SHUFFLED}
for _name in ORACLE:
    ARMS[_name] = (GEOM_NONE, APPLY_NONE, False, False)

# two-sided 95% t quantiles (scipy is not installed in this venv)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 35: 2.030,
        40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984}
# 80%-power one-sided-at-2.5% companion: t95(df) + t80(df), t80 ~ 0.842 for large df
_T80 = {3: 1.061, 4: 1.000, 5: 0.941, 6: 0.906, 7: 0.896, 8: 0.889, 9: 0.883,
        10: 0.879, 15: 0.866, 20: 0.860, 30: 0.854, 60: 0.848, 100: 0.845}


def _lookup(table, df, default):
    if df in table:
        return table[df]
    for k in sorted(table):
        if k >= df:
            return table[k]
    return default


def t95(df):
    return float("inf") if df <= 0 else _lookup(_T95, df, 1.960)


def t80(df):
    return _lookup(_T80, df, 0.842)


def ci95(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(t95(v.size - 1) * v.std(ddof=1) / math.sqrt(v.size))


@dataclass
class Task:
    input_dim: int = 64
    hidden: int = 64
    regions: int = 8
    per_region_teacher: bool = True
    shift: float = 12.0
    samples_per_region: int = 4000
    noise_std: float = 0.5
    batch: int = 8
    eval_samples: int = 2048
    oracle_batch: int = 8
    """samples PER REGION the oracle arms judge each pending step against"""
    base_seed: int = 1
    shuffle: bool = False
    resample_task: bool = True
    """False reproduces novelty_stream.py: one teacher/centre draw for every seed."""
    share_stream: bool = False
    """True reproduces novelty_stream.py: one input/noise stream for every seed."""

    def steps(self):
        return self.regions * max(self.samples_per_region // self.batch, 1)

    def label(self):
        extra = "".join([" shared-teacher" if not self.per_region_teacher else "",
                         " shuffled" if self.shuffle else "",
                         " fixed-task" if not self.resample_task else "",
                         " shared-stream" if self.share_stream else ""])
        return (f"R{self.regions} shift{self.shift:g} B{self.batch} H{self.hidden} "
                f"D{self.input_dim} N{self.samples_per_region}{extra}")


def teacher_apply(x, t1, t2):
    """x (S,N,D), t1 (S,D,8), t2 (S,8) -> (S,N)"""
    return torch.einsum("sne,se->sn",
                        torch.tanh(torch.einsum("snd,sde->sne", x, t1)), t2)


def draw_task(task, n_seed, device, gen):
    d = task.input_dim
    draws = 1 if not task.resample_task else n_seed
    t1 = torch.randn((draws, task.regions, d, 8), device=device, generator=gen) / math.sqrt(d)
    t2 = torch.randn((draws, task.regions, 8), device=device, generator=gen) * 0.7
    if not task.per_region_teacher:
        t1 = t1[:, :1].expand(-1, task.regions, -1, -1)
        t2 = t2[:, :1].expand(-1, task.regions, -1)
    centres = torch.randn((draws, task.regions, d), device=device, generator=gen)
    centres = centres / centres.norm(dim=-1, keepdim=True) * task.shift
    if draws != n_seed:
        t1 = t1.expand(n_seed, -1, -1, -1)
        t2 = t2.expand(n_seed, -1, -1)
        centres = centres.expand(n_seed, -1, -1)
    init = torch.randn((n_seed, task.hidden, d), device=device, generator=gen) / math.sqrt(d)
    return t1.contiguous(), t2.contiguous(), centres.contiguous(), init


@torch.no_grad()
def evaluate(w1, w2, t1, t2, centres, seed_idx, task, device, gen, chunk=512):
    """Per-region error normalised by the zero-predictor, as in the harness."""
    n_seed, regions = centres.shape[0], centres.shape[1]
    scores = torch.zeros((regions, w1.shape[0]), device=device)
    for r in range(regions):
        num = torch.zeros((w1.shape[0],), device=device)
        den = torch.zeros((), device=device)
        seen = 0
        for start in range(0, task.eval_samples, chunk):
            n = min(chunk, task.eval_samples - start)
            x = centres[:, r].unsqueeze(1) + torch.randn(
                (n_seed, n, task.input_dim), device=device, generator=gen)
            clean = teacher_apply(x, t1[:, r], t2[:, r])
            xk, ck = x[seed_idx], clean[seed_idx]
            pred = (torch.tanh(torch.einsum("khd,knd->knh", w1, xk)) * w2).sum(-1)
            num += (pred - ck).square().sum(-1)
            den += clean.square().sum()
            seen += n
        scores[r] = (num / seen) / (den / (n_seed * seen)).clamp_min(1e-12)
    return scores


@torch.no_grad()
def geometry_report(ig, centres, seed_idx, geo_pos, configs, pairs=2000):
    """Anisotropy and cross-unit direction agreement of the learned geometry.

    ig (G,H,D,D) is (I + lambda C_h)^-1, so its SMALLEST eigenvalue is the
    direction the unit has committed to hardest.
    """
    n_geo, h, d, _ = ig.shape
    sym = 0.5 * (ig + ig.transpose(-1, -2))
    values, vectors = torch.linalg.eigh(sym.reshape(n_geo * h, d, d))
    values = values.view(n_geo, h, d)
    damped = vectors.view(n_geo, h, d, d)[..., 0]                  # (G,H,D)
    damped = damped / damped.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    anisotropy = (values[..., -1] / values[..., 0].clamp_min(1e-30))
    # participation ratio of the geometry's eigenvalues of C = ig^-1 - I
    curvature = (1.0 / values.clamp_min(1e-30) - 1.0).clamp_min(0.0)
    participation = (curvature.sum(-1).square()
                     / curvature.square().sum(-1).clamp_min(1e-30))
    gen = torch.Generator(device=ig.device).manual_seed(0)
    i = torch.randint(0, h, (pairs,), device=ig.device, generator=gen)
    j = torch.randint(0, h, (pairs,), device=ig.device, generator=gen)
    keep = i != j
    cos_units = (damped[:, i[keep]] * damped[:, j[keep]]).sum(-1).abs()   # (G,P)
    seeds_geo = seed_idx[torch.tensor(geo_pos, device=ig.device)]
    cen = centres[seeds_geo]                                       # (G,R,D)
    cen = cen / cen.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    cos_centre = torch.einsum("ghd,grd->ghr", damped, cen).abs().amax(-1)  # (G,H)
    return {"anisotropy": anisotropy.median(-1).values.cpu().numpy(),
            "participation": participation.median(-1).values.cpu().numpy(),
            "cos_units": cos_units.median(-1).values.cpu().numpy(),
            "cos_centre": cos_centre.median(-1).values.cpu().numpy()}


@torch.no_grad()
def separability_report(commit_region, commit_trace, pairs=2000):
    """How much per-unit anisotropy the commitment scalars can possibly carry.

    The rank-1 direction is layer-shared, so a unit's geometry is
    `C_h = sum_t lambda * c_ht * d_t d_t^T`. If the commitment matrix factorises,
    `c_ht = u_h * a_t`, then `C_h = u_h * C_shared` EXACTLY, and
    `(I + lambda u_h C_shared)^-1` is a one-parameter family: per-unit
    anisotropy is then provably nothing but a per-unit SCALAR applied to one
    shared geometry. So the fraction of `c_ht`'s energy captured by its best
    rank-1 approximation is an upper bound on the per-unit directional content,
    and `1 - that` is the entire budget available to it.
    """
    def rank1_energy(m):
        s = torch.linalg.svdvals(m.float())
        total = s.square().sum().clamp_min(1e-30)
        return float(s[0].square() / total)

    out = {"rank1_region": np.array([rank1_energy(m) for m in commit_region]),
           "rank1_trace": rank1_energy(commit_trace)}
    profile = commit_region / commit_region.sum(-1, keepdim=True).clamp_min(1e-30)
    profile = profile / profile.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    h = profile.shape[1]
    gen = torch.Generator(device=profile.device).manual_seed(0)
    i = torch.randint(0, h, (pairs,), device=profile.device, generator=gen)
    j = torch.randint(0, h, (pairs,), device=profile.device, generator=gen)
    keep = i != j
    cos_profile = (profile[:, i[keep]] * profile[:, j[keep]]).sum(-1)
    out["cos_profile"] = cos_profile.median(-1).values.cpu().numpy()
    total = commit_region.sum(-1)                                  # (G,H)
    out["commit_cv"] = (total.std(-1) / total.mean(-1).clamp_min(1e-30)).cpu().numpy()
    return out


@torch.no_grad()
def run(configs, task, device, schedule=None, diagnose=False):
    """One vectorised pass over every config. Keys: arm, lr, seed, lam, decay."""
    n_cfg = len(configs)
    d, h = task.input_dim, task.hidden
    n_seed = max(c["seed"] for c in configs) + 1
    gen = torch.Generator(device=device).manual_seed(task.base_seed)
    t1, t2, centres, init = draw_task(task, n_seed, device, gen)

    seed_idx = torch.tensor([c["seed"] for c in configs], device=device)
    lr = torch.tensor([c["lr"] for c in configs], device=device,
                      dtype=torch.float32).view(n_cfg, 1, 1)
    geom = np.array([ARMS[c["arm"]][0] for c in configs])
    apply_mode = np.array([ARMS[c["arm"]][1] for c in configs])
    sched_flag = torch.tensor([float(ARMS[c["arm"]][3]) for c in configs],
                              device=device).view(n_cfg, 1, 1)
    oracle_mode = np.array([ORACLE.get(c["arm"], ORACLE_NONE) for c in configs])
    any_oracle = bool((oracle_mode != ORACLE_NONE).any())
    o_unit = torch.tensor((oracle_mode == ORACLE_UNIT).astype(np.float32),
                          device=device).view(n_cfg, 1)
    o_mean = torch.tensor((oracle_mode == ORACLE_MEAN).astype(np.float32),
                          device=device).view(n_cfg, 1)
    o_shuf = torch.tensor((oracle_mode == ORACLE_SHUFFLED).astype(np.float32),
                          device=device).view(n_cfg, 1)

    w1 = init[seed_idx].clone()
    w2 = torch.zeros((n_cfg, 1, h), device=device)
    m1, v1 = torch.zeros_like(w1), torch.zeros_like(w1)
    m2, v2 = torch.zeros_like(w2), torch.zeros_like(w2)

    geo_pos = np.flatnonzero(geom != GEOM_NONE)
    n_geo = geo_pos.size
    ig = None
    if n_geo:
        geo_idx = torch.tensor(geo_pos, device=device)
        ig = torch.eye(d, device=device).expand(n_geo, h, d, d).clone()
        lam = torch.tensor([configs[i]["lam"] for i in geo_pos],
                           device=device).view(n_geo, 1, 1)
        decay = torch.tensor([configs[i]["decay"] for i in geo_pos], device=device)
        any_decay = bool(float(decay.max()) > 0.0)
        decay_v = decay.view(n_geo, 1, 1, 1)
        gmode, amode = geom[geo_pos], apply_mode[geo_pos]
        shared = np.array([ARMS[configs[i]["arm"]][2] for i in geo_pos])
        any_shared = bool(shared.any())
        m_shared = torch.tensor(shared.astype(np.float32), device=device).view(n_geo, 1, 1)
        any_diag = bool((gmode == GEOM_DIAG).any())
        if any_diag:
            is_diag = torch.tensor((gmode == GEOM_DIAG).astype(np.float32),
                                   device=device).view(n_geo, 1, 1, 1)
            eye = torch.eye(d, device=device)
            diag_mask = (is_diag * eye + (1.0 - is_diag)).view(n_geo, 1, d, d)
        m_proj = torch.tensor((amode == APPLY_PROJ).astype(np.float32),
                              device=device).view(n_geo, 1, 1)
        m_scalar = torch.tensor((amode == APPLY_SCALAR).astype(np.float32),
                                device=device).view(n_geo, 1, 1)
        m_rotate = torch.tensor((amode == APPLY_ROTATE).astype(np.float32),
                                device=device).view(n_geo, 1, 1)

    total_steps = task.steps()
    rho_sum = torch.zeros((n_cfg,), device=device)
    rho_spread_sum = torch.zeros((n_cfg,), device=device)
    norm_sum = torch.zeros((n_cfg,), device=device)
    adam_norm_sum = torch.zeros((n_cfg,), device=device)
    traj = torch.zeros((total_steps, n_cfg), device=device)
    direction_moment = torch.zeros((d, d), device=device) if diagnose else None
    commit_region = (torch.zeros((n_geo, h, task.regions), device=device)
                     if diagnose and n_geo else None)
    commit_trace = []
    snap_every = max(total_steps // 400, 1)
    sched = None
    if schedule is not None:
        sched = torch.as_tensor(schedule, device=device, dtype=torch.float32)
        if sched.numel() != total_steps:
            sched = torch.nn.functional.interpolate(
                sched.view(1, 1, -1), size=total_steps, mode="linear",
                align_corners=True).view(-1)

    steps_per_region = max(task.samples_per_region // task.batch, 1)
    stream_seeds = 1 if task.share_stream else n_seed
    rows_seed = torch.arange(n_seed, device=device)
    step_id = 0
    beta1, beta2 = 0.9, 0.999
    start = time.perf_counter()
    for region in range(task.regions):
        for _ in range(steps_per_region):
            if task.shuffle:
                pick = torch.randint(0, task.regions, (stream_seeds,), device=device,
                                     generator=gen)
            else:
                pick = torch.full((stream_seeds,), region, device=device, dtype=torch.long)
            src = rows_seed[:stream_seeds]
            centre = centres[src, pick]
            x = centre.unsqueeze(1) + torch.randn(
                (stream_seeds, task.batch, d), device=device, generator=gen)
            clean = teacher_apply(x, t1[src, pick], t2[src, pick])
            y = clean + task.noise_std * torch.randn(
                (stream_seeds, task.batch), device=device, generator=gen)
            if task.share_stream:
                xk = x[0].unsqueeze(0).expand(n_cfg, -1, -1)
                yk = y[0].unsqueeze(0).expand(n_cfg, -1)
            else:
                xk, yk = x[seed_idx], y[seed_idx]

            pre = torch.einsum("khd,kbd->kbh", w1, xk)
            act = torch.tanh(pre)
            residual = (act * w2).sum(-1) - yk
            g2 = torch.einsum("kb,kbh->kh", residual, act).view(n_cfg, 1, h) / task.batch
            back = residual.unsqueeze(2) * w2 * (1.0 - act.square())
            g1 = torch.einsum("kbh,kbd->khd", back, xk) / task.batch

            step_id += 1
            m1.mul_(beta1).add_(g1, alpha=1 - beta1)
            v1.mul_(beta2).addcmul_(g1, g1, value=1 - beta2)
            m2.mul_(beta1).add_(g2, alpha=1 - beta1)
            v2.mul_(beta2).addcmul_(g2, g2, value=1 - beta2)
            b1, b2 = 1 - beta1 ** step_id, 1 - beta2 ** step_id
            step1 = (m1 / b1) / ((v1 / b2).sqrt() + 1e-8)
            step2 = (m2 / b1) / ((v2 / b2).sqrt() + 1e-8)
            adam_norm_sum += step1.norm(dim=-1).mean(-1)

            ratio_full = torch.ones((n_cfg, h), device=device)
            if n_geo:
                if any_decay:
                    ig.mul_(1.0 - decay_v)
                    ig.diagonal(dim1=-2, dim2=-1).add_(decay_v.view(n_geo, 1, 1))
                s1 = step1[geo_idx]
                proj = torch.einsum("ghde,ghe->ghd", ig, s1)
                energy = (s1 * s1).sum(-1)
                ratio = torch.where(energy > 0,
                                    (proj * s1).sum(-1) / energy.clamp_min(1e-30),
                                    torch.ones_like(energy))
                scale = energy.sqrt() / proj.norm(dim=-1).clamp_min(1e-30)
                new = (m_proj * proj
                       + m_scalar * (s1 * ratio.unsqueeze(-1))
                       + m_rotate * (proj * scale.unsqueeze(-1)))
                step1 = step1.index_copy(0, geo_idx, new)
                ratio_full = ratio_full.index_copy(0, geo_idx, ratio)

                commitment = back[geo_idx].abs().mean(1).unsqueeze(-1)     # (G,H,1)
                if any_shared:
                    commitment = torch.where(
                        m_shared.bool(), commitment.mean(1, keepdim=True).expand_as(
                            commitment), commitment)
                direction = xk[geo_idx].mean(1)                            # (G,D)
                mapped = torch.einsum("ghde,ge->ghd", ig, direction)
                weight = lam * commitment
                denom = (1.0 + weight.squeeze(-1)
                         * torch.einsum("ghd,gd->gh", mapped, direction)).clamp_min(1e-12)
                coef = -(weight.squeeze(-1) / denom)
                ig.view(n_geo * h, d, d).baddbmm_(
                    (mapped * coef.unsqueeze(-1)).view(n_geo * h, d, 1),
                    mapped.view(n_geo * h, 1, d))
                if any_diag:
                    ig.mul_(diag_mask)
                if diagnose:
                    unit = direction[0] / direction[0].norm().clamp_min(1e-30)
                    direction_moment.addr_(unit, unit)
                    commit_region[:, :, region].add_(commitment.squeeze(-1))
                    if step_id % snap_every == 0:
                        commit_trace.append(commitment[0, :, 0].clone())

            if any_oracle:
                # the whole-mixture gradient this unit's step is judged against
                xm = torch.cat([centres[:, r].unsqueeze(1) + torch.randn(
                    (n_seed, task.oracle_batch, d), device=device, generator=gen)
                    for r in range(task.regions)], dim=1)
                tm = torch.cat([teacher_apply(
                    xm[:, r * task.oracle_batch:(r + 1) * task.oracle_batch],
                    t1[:, r], t2[:, r]) for r in range(task.regions)], dim=1)
                xmk, tmk = xm[seed_idx], tm[seed_idx]
                am = torch.tanh(torch.einsum("khd,kmd->kmh", w1, xmk))
                rm = (am * w2).sum(-1) - tmk
                bm = rm.unsqueeze(2) * w2 * (1.0 - am.square())
                g_all = torch.einsum("kmh,kmd->khd", bm, xmk) / xmk.shape[1]
                # first order: this step lowers the mixture loss iff <g_all, step> > 0
                gate = ((g_all * step1).sum(-1) > 0).float()             # (K,H)
                perm = torch.randperm(h, device=device, generator=gen)
                level = (o_unit * gate + o_mean * gate.mean(-1, keepdim=True)
                         + o_shuf * gate[:, perm]
                         + (1.0 - o_unit - o_mean - o_shuf))
                step1 = step1 * level.unsqueeze(-1)
                ratio_full = ratio_full * level

            traj[step_id - 1] = ratio_full.mean(-1)
            rho_sum += ratio_full.mean(-1)
            rho_spread_sum += ratio_full.std(-1)
            norm_sum += step1.norm(dim=-1).mean(-1)
            eff = lr if sched is None else lr * (
                1.0 + sched_flag * (sched[step_id - 1] - 1.0))
            w1 -= eff * step1
            w2 -= eff.view(n_cfg, 1, 1) * step2

    seconds = time.perf_counter() - start
    per_region = evaluate(w1, w2, t1, t2, centres, seed_idx, task, device, gen)
    result = {"acquisition": per_region[-1].cpu().numpy(),
              "retention": per_region[:-1].mean(0).cpu().numpy(),
              "overall": per_region.mean(0).cpu().numpy(),
              "rho": (rho_sum / total_steps).cpu().numpy(),
              "rho_spread": (rho_spread_sum / total_steps).cpu().numpy(),
              "norm_ratio": (norm_sum / adam_norm_sum.clamp_min(1e-30)).cpu().numpy(),
              "traj": traj.cpu().numpy(), "seconds": seconds}
    if diagnose and n_geo:
        result["geometry"] = geometry_report(ig, centres, seed_idx, geo_pos, configs)
        result["geo_pos"] = geo_pos
        values = torch.linalg.eigvalsh(direction_moment)
        result["dir_participation"] = float(
            values.clamp_min(0).sum().square() / values.clamp_min(0).square().sum())
        result["separability"] = separability_report(
            commit_region, torch.stack(commit_trace, dim=-1))
    return result


# ----------------------------------------------------------------------------- reporting

def grid(arms, lrs, seeds, lam=1.0, decay=0.0):
    return [{"arm": a, "lr": v, "seed": s, "lam": lam, "decay": decay}
            for a in arms for v in lrs for s in range(seeds)]


def matches(c, match):
    for k, v in match.items():
        if isinstance(v, float):
            if abs(c[k] - v) > 1e-12:
                return False
        elif c[k] != v:
            return False
    return True


def per_seed(configs, out, metric, **match):
    picks = sorted((c["seed"], i) for i, c in enumerate(configs) if matches(c, match))
    return np.array([out[metric][i] for _, i in picks])


def arm_lrs(configs, arm):
    return sorted({c["lr"] for c in configs if c["arm"] == arm})


def seeds_needed(sd, effect, limit=100000):
    """Paired seeds required to call an effect of size `effect` at 95%
    confidence and 80% power, given the paired seed-to-seed SD."""
    effect = abs(effect)
    if effect <= 0.0 or sd <= 0.0:
        return 0
    n = 3
    while n < limit:
        if (t95(n - 1) + t80(n - 1)) * sd / math.sqrt(n) <= effect:
            return n
        n += 1
    return limit


def best_lr(configs, out, arm, seed_subset=None, warn=True, floor=None, **extra):
    """LR minimising mean `overall`. Warns when the optimum sits on a grid
    boundary: `novelty_stream.py`'s published grid (3e-4..1e-2) has Adam's
    optimum BELOW its floor, so every arm there was scored on the over-stepped
    side of its own LR curve and a step-shrinking mechanism gets credit merely
    for descending toward the optimum Adam was never allowed to reach."""
    grid_lrs = [v for v in arm_lrs(configs, arm) if floor is None or v >= floor]
    best, chosen = None, None
    for v in grid_lrs:
        vals = per_seed(configs, out, "overall", arm=arm, lr=v, **extra)
        if seed_subset is not None:
            vals = vals[seed_subset]
        score = float(vals.mean())
        if best is None or score < best:
            best, chosen = score, v
    if warn and len(grid_lrs) > 1 and chosen in (grid_lrs[0], grid_lrs[-1]):
        side = "FLOOR" if chosen == grid_lrs[0] else "CEILING"
        print(f"    !! {arm}: best LR {chosen:.2e} is the grid {side}; "
              f"its optimum is not bracketed")
    return chosen, best


def report_arm(configs, out, arm, lr, seed_subset=None, **extra):
    row = {}
    for metric in ("acquisition", "retention", "overall", "rho", "rho_spread",
                   "norm_ratio"):
        vals = per_seed(configs, out, metric, arm=arm, lr=lr, **extra)
        row[metric] = vals if seed_subset is None else vals[seed_subset]
    return row


HEAD = (f"{'arm':>17} {'lr':>9} | {'acquire':>7} {'retain':>7} {'overall':>17} | "
        f"{'rho':>6} {'rho_sd':>6} {'|step|':>6}")


def print_row(name, lr, row):
    m, hw = ci95(row["overall"])
    print(f"{name:>17} {lr:>9.2e} | {row['acquisition'].mean():>7.4f} "
          f"{row['retention'].mean():>7.4f} {m:>7.4f} +-{hw:<8.4f} | "
          f"{row['rho'].mean():>6.3f} {row['rho_spread'].mean():>6.3f} "
          f"{row['norm_ratio'].mean():>6.3f}")


def print_paired(label, a, b, key="overall"):
    diff = a[key] - b[key]
    m, hw = ci95(diff)
    print(f"    {label:<48} {m:+.4f} +-{hw:.4f}  "
          f"{'SIGNIFICANT' if abs(m) > hw else 'inside noise'}")
    return m, hw


def mde_table(sd, label, ns=(4, 8, 16, 32, 64, 128, 256)):
    """Minimum detectable effect at 95% confidence and 80% power."""
    print(f"    {label}: SD {sd:.4f}")
    print(f"      {'n':>5} {'CI95 halfwidth':>15} {'MDE(80% power)':>15} "
          f"{'as % of 0.62':>13}")
    for n in ns:
        hw = t95(n - 1) * sd / math.sqrt(n)
        mde = (t95(n - 1) + t80(n - 1)) * sd / math.sqrt(n)
        print(f"      {n:>5} {hw:>15.4f} {mde:>15.4f} {100 * mde / 0.62:>12.1f}%")


# Measured: Adam's optimum on the shift-12 conflict task is 1.92e-4 (overall
# 0.4494), i.e. BELOW the published grid's 3e-4 floor, and `precision` at rho
# ~0.2 wants ~5x more. Every grid here therefore brackets both.
WIDE = tuple(1.2e-5 * 2.0 ** k for k in range(0, 15))             # 1.2e-5 .. 1.97e-1
# `novelty_stream.py --lr-grid 3e-4 1e-3 3e-3 1e-2`: its floor sits ABOVE Adam's
# optimum, so restricting the grid here to this value reproduces the regime the
# published comparison was actually made in.
PUBLISHED_FLOOR = 3.84e-4


# ----------------------------------------------------------------------------- Q1

def bootstrap_procedure(configs, out, arm_a, arm_b, ns, draws=4000, seed=0,
                        floor=None, tag=""):
    """Resample seeds, re-run the published protocol (per-arm best LR, report the
    gap) and report the distribution of the gap it produces. `floor` restricts
    the LR grid to the published window."""
    lrs_a = [v for v in arm_lrs(configs, arm_a) if floor is None or v >= floor]
    lrs_b = [v for v in arm_lrs(configs, arm_b) if floor is None or v >= floor]
    mat_a = np.stack([per_seed(configs, out, "overall", arm=arm_a, lr=v) for v in lrs_a])
    mat_b = np.stack([per_seed(configs, out, "overall", arm=arm_b, lr=v) for v in lrs_b])
    n_seed = mat_a.shape[1]
    rng = np.random.default_rng(seed)
    truth = mat_a.mean(1).min() - mat_b.mean(1).min()
    print(f"    bootstrap of `best LR per arm` gap, {arm_a} - {arm_b}{tag} "
          f"({len(lrs_a)} LRs, {n_seed} seeds, {draws} draws)")
    print(f"      full-sample gap {truth:+.4f}")
    print(f"      {'n seeds':>8} {'mean gap':>10} {'sd':>8} {'2.5%':>9} {'97.5%':>9} "
          f"{'P(gap<=-0.028)':>15} {'P(sign flip)':>13}")
    rows = {}
    for n in ns:
        idx = rng.integers(0, n_seed, size=(draws, n))
        gap = mat_a[:, idx].mean(2).min(0) - mat_b[:, idx].mean(2).min(0)
        rows[n] = gap
        flip = np.mean(np.sign(gap) != np.sign(truth)) if truth != 0 else float("nan")
        print(f"      {n:>8} {gap.mean():>+10.4f} {gap.std():>8.4f} "
              f"{np.quantile(gap, 0.025):>+9.4f} {np.quantile(gap, 0.975):>+9.4f} "
              f"{np.mean(gap <= -0.028):>15.3f} {flip:>13.3f}")
    return rows


def false_positive_rate(vals, sizes, effect=0.028, draws=4000, seed=1):
    """Split ONE arm's seeds into two disjoint halves of size n and ask how often
    the published protocol would have called a gap of `effect`. The true gap is
    exactly zero, so every hit is a false positive."""
    n_seed = vals.size
    rng = np.random.default_rng(seed)
    for n in sizes:
        if 2 * n > n_seed:
            continue
        gaps = np.empty(draws)
        for d in range(draws):
            perm = rng.permutation(n_seed)
            gaps[d] = vals[perm[:n]].mean() - vals[perm[n:2 * n]].mean()
        print(f"      n={n:<4} sd {gaps.std():.4f}  "
              f"P(|gap| >= {effect:.3f}) = {np.mean(np.abs(gaps) >= effect):.3f}  "
              f"P(|gap| >= 0.093) = {np.mean(np.abs(gaps) >= 0.093):.3f}")


def frontier_gap(configs, out, arm, lr, base="adam"):
    """LR-FREE comparison against the baseline's own (acquisition, retention)
    Pareto frontier, built per seed from that seed's baseline LR curve.

    A learning rate alone already trades acquisition against retention, so the
    only question that a step-size control cannot answer for a mechanism is
    whether it lies OUTSIDE that frontier. Two verdicts are reported:

      * dominated -- some single baseline LR on the same seed is better at BOTH
        acquisition and retention, which no step-size story can excuse;
      * retention penalty -- with acquisition matched by interpolation along the
        frontier, how much worse the mechanism's retention is.

    Adam's acquisition is NOT monotone in LR (it turns around once the run
    diverges), so the frontier must be taken as the non-dominated subset rather
    than as the raw LR-ordered curve.
    """
    lrs = arm_lrs(configs, base)
    acq = np.stack([per_seed(configs, out, "acquisition", arm=base, lr=v) for v in lrs])
    ret = np.stack([per_seed(configs, out, "retention", arm=base, lr=v) for v in lrs])
    m_acq = per_seed(configs, out, "acquisition", arm=arm, lr=lr)
    m_ret = per_seed(configs, out, "retention", arm=arm, lr=lr)
    gaps, covered, dominated = [], 0, 0
    for s in range(m_acq.size):
        a, r = acq[:, s], ret[:, s]
        if np.any((a <= m_acq[s]) & (r <= m_ret[s])):
            dominated += 1
        keep = [i for i in range(a.size)
                if not np.any((a <= a[i]) & (r <= r[i])
                              & ((a < a[i]) | (r < r[i])))]
        order = sorted(keep, key=lambda i: a[i])
        fa = np.array([a[i] for i in order])
        fr = np.array([r[i] for i in order])
        if fa.size < 2 or not (fa[0] <= m_acq[s] <= fa[-1]):
            continue
        covered += 1
        gaps.append(m_ret[s] - float(np.interp(m_acq[s], fa, fr)))
    n = m_acq.size
    print(f"    {arm} vs {base}'s own acquisition/retention frontier, per seed:")
    print(f"      strictly dominated by a single {base} LR on "
          f"{dominated}/{n} seeds ({100 * dominated / n:.0f}%)")
    if gaps:
        m, hw = ci95(np.array(gaps))
        print(f"      retention penalty at matched acquisition "
              f"({covered}/{n} seeds on-frontier): {m:+.4f} +-{hw:.4f}  "
              f"{'OFF the frontier' if abs(m) > hw else 'ON the frontier'}")
        return m, hw
    print(f"      acquisition lies OUTSIDE the frontier's range on every seed: "
          f"the mechanism is not even on the useful branch")
    return float("nan"), float("nan")


def exp_power(args, device):
    """Q1: what can this task resolve, and with how many seeds."""
    arms = ("adam", "precision", "precision_shared", "precision_scalar")
    for resample, share, tag in ((True, False, "resampled task (teachers/centres/stream/init)"),
                                 (False, True, "novelty_stream.py protocol (init only)")):
        task = Task(base_seed=args.base_seed, batch=args.batch, regions=args.regions,
                    shift=args.shift, hidden=args.hidden,
                    samples_per_region=args.samples, resample_task=resample,
                    share_stream=share)
        configs = grid(arms, WIDE, args.seeds)
        print(f"\n=== {tag}: {len(configs)} configs, {task.label()}, "
              f"{args.seeds} seeds ===")
        out = run(configs, task, device)
        print(f"  {out['seconds']:.1f}s")
        print(HEAD)
        rows = {}
        for arm in arms:
            v, _ = best_lr(configs, out, arm)
            rows[arm] = (v, report_arm(configs, out, arm, v))
            print_row(arm, v, rows[arm][1])
        print("  paired differences at each arm's own best LR:")
        print_paired("precision - adam", rows["precision"][1], rows["adam"][1])
        print_paired("precision - precision_shared", rows["precision"][1],
                     rows["precision_shared"][1])
        print_paired("precision - precision_scalar (anisotropy only)",
                     rows["precision"][1], rows["precision_scalar"][1])
        print_paired("precision_shared - adam", rows["precision_shared"][1],
                     rows["adam"][1])

        print("\n  --- the published LR window, same run, same seeds ---")
        print("  Restricting each arm's LR choice to >= 3.84e-4 reproduces the")
        print("  published grid's reachable range. Adam's optimum (1.92e-4) is")
        print("  outside it, so this is Adam scored on the over-stepped side.")
        window = {}
        for arm in arms:
            v, _ = best_lr(configs, out, arm, warn=False, floor=PUBLISHED_FLOOR)
            window[arm] = report_arm(configs, out, arm, v)
            print_row(arm + " (window)", v, window[arm])
        print_paired("precision - adam, published window", window["precision"],
                     window["adam"])
        print_paired("adam full grid - adam published window", rows["adam"][1],
                     window["adam"])

        print("\n  --- variance structure ---")
        for arm in arms:
            v = rows[arm][0]
            vals = per_seed(configs, out, "overall", arm=arm, lr=v)
            print(f"    {arm:>17} @ {v:.2e}: mean {vals.mean():.4f} "
                  f"SD {vals.std(ddof=1):.4f} min {vals.min():.4f} max {vals.max():.4f}")
        unpaired = np.mean([per_seed(configs, out, "overall", arm=a, lr=rows[a][0]
                                     ).std(ddof=1) for a in arms])
        mde_table(unpaired * math.sqrt(2.0), "UNPAIRED two-arm comparison")
        paired = (per_seed(configs, out, "overall", arm="precision", lr=rows["precision"][0])
                  - per_seed(configs, out, "overall", arm="adam", lr=rows["adam"][0]))
        mde_table(paired.std(ddof=1), "PAIRED precision-vs-adam (same seeds)")

        print("\n  --- false-positive rate of the published protocol ---")
        print("  Two arms drawn from the SAME mechanism at neighbouring LRs would")
        print("  confound a real LR effect, so the null is built from the identical")
        print("  arm compared against itself on disjoint seed halves instead.")
        for arm in ("adam", "precision"):
            v = rows[arm][0]
            print(f"    {arm} @ {v:.2e}, true gap exactly zero:")
            false_positive_rate(per_seed(configs, out, "overall", arm=arm, lr=v),
                                (4, 8, 16, args.seeds // 2))
        bootstrap_procedure(configs, out, "precision", "adam", (4, 8, 16, args.seeds))
        bootstrap_procedure(configs, out, "precision", "adam", (4, 8, 16, args.seeds),
                            floor=PUBLISHED_FLOOR, tag=" [PUBLISHED LR WINDOW]")
        bootstrap_procedure(configs, out, "precision", "precision_shared",
                            (4, 8, 16, args.seeds))

        print("\n  --- LR-free frontier position ---")
        for arm in ("precision", "precision_shared", "precision_scalar"):
            frontier_gap(configs, out, arm, rows[arm][0])

        print("\n  --- is the acquisition/retention tradeoff resolvable for ANY arm? ---")
        print("  Adam's own LR sweep is the reference frontier: if the tradeoff is")
        print("  real, acquisition must fall and retention must rise with LR, each")
        print("  significantly, and `overall` must have an interior optimum.")
        lrs = arm_lrs(configs, "adam")
        prev = None
        for v in lrs:
            row = report_arm(configs, out, "adam", v)
            a_m, a_hw = ci95(row["acquisition"])
            r_m, r_hw = ci95(row["retention"])
            o_m, o_hw = ci95(row["overall"])
            print(f"    adam {v:.2e}: acquire {a_m:.4f}+-{a_hw:.4f}  "
                  f"retain {r_m:.4f}+-{r_hw:.4f}  overall {o_m:.4f}+-{o_hw:.4f}")
            if prev is not None:
                print_paired(f"  step {prev[0]:.1e}->{v:.1e} acquisition",
                             row, prev[1], key="acquisition")
                print_paired(f"  step {prev[0]:.1e}->{v:.1e} retention",
                             row, prev[1], key="retention")
            prev = (v, row)


# ----------------------------------------------------------------------------- Q2

def exp_autonomy(args, device):
    """Q2: can this task express per-unit autonomy at all?"""
    print("\n=== STRUCTURAL CHECK: is the commitment direction shared across units? ===")
    print("  In `novelty_stream.py` the rank-1 direction is `x.mean(0)`, one vector")
    print("  per step for the whole layer, so C_h = sum_t lam*c_ht d_t d_t^T differs")
    print("  across units ONLY by the scalars c_ht. Verified below by construction:")
    print("  `precision_shared` replaces c_ht with its cross-unit mean; if the")
    print("  per-unit scalars carried anything, it must lose.")

    arms = ("adam", "precision", "precision_shared", "precision_scalar")
    print("\n=== shift sweep: centre dominance vs cross-unit direction agreement ===")
    print("  d_t = centre + mean of `batch` unit-variance draws, so")
    print("  E|centre| / E|noise part| = shift / sqrt(D/batch).")
    for shift in (0.0, 1.0, 3.0, 6.0, 12.0, 24.0):
        task = Task(base_seed=args.base_seed, batch=args.batch, regions=args.regions,
                    shift=shift, hidden=args.hidden, samples_per_region=args.samples)
        configs = grid(arms, WIDE, args.diag_seeds)
        out = run(configs, task, device, diagnose=True)
        geo = out["geometry"]
        pos = {p: k for k, p in enumerate(out["geo_pos"])}
        print(f"\n  --- shift {shift:g} "
              f"(centre/noise = {shift / math.sqrt(task.input_dim / task.batch):.2f}, "
              f"{out['seconds']:.1f}s) ---")
        print(HEAD)
        rows = {}
        for arm in arms:
            v, _ = best_lr(configs, out, arm)
            rows[arm] = report_arm(configs, out, arm, v)
            print_row(arm, v, rows[arm])
            picks = [pos[i] for i, c in enumerate(configs)
                     if c["arm"] == arm and abs(c["lr"] - v) < 1e-12 and i in pos]
            if picks:
                print(f"{'':>19} geometry: anisotropy "
                      f"{np.median(geo['anisotropy'][picks]):>9.1f}x  "
                      f"eff.rank(C) {np.median(geo['participation'][picks]):>5.2f}  "
                      f"cross-unit |cos| {np.median(geo['cos_units'][picks]):.3f}  "
                      f"|cos| with nearest centre "
                      f"{np.median(geo['cos_centre'][picks]):.3f}")
        print(f"{'':>19} input-direction effective rank over training: "
              f"{out['dir_participation']:.2f} of {task.input_dim}")
        sep = out["separability"]
        prec = [pos[i] for i, c in enumerate(configs)
                if c["arm"] == "precision" and i in pos]
        print(f"{'':>19} commitment matrix c_ht: rank-1 energy "
              f"{np.median(sep['rank1_region'][prec]):.4f} (unit x region), "
              f"{sep['rank1_trace']:.4f} (unit x step)  -> per-unit directional "
              f"budget {1 - np.median(sep['rank1_region'][prec]):.4f}")
        print(f"{'':>19} cross-unit cosine of region-commitment profiles "
              f"{np.median(sep['cos_profile'][prec]):.4f}, cross-unit CV of total "
              f"commitment {np.median(sep['commit_cv'][prec]):.3f}")
        print_paired("precision - adam", rows["precision"], rows["adam"])
        print_paired("precision - precision_shared (per-unit content)",
                     rows["precision"], rows["precision_shared"])
        print_paired("precision - precision_scalar (anisotropy content)",
                     rows["precision"], rows["precision_scalar"])

    print("\n=== batch sweep at shift 12: does averaging kill direction diversity? ===")
    for batch in (1, 8, 64):
        task = Task(base_seed=args.base_seed, batch=batch, regions=args.regions,
                    shift=args.shift, hidden=args.hidden,
                    samples_per_region=args.samples)
        configs = grid(("adam", "precision", "precision_shared"), WIDE, args.diag_seeds)
        out = run(configs, task, device, diagnose=True)
        geo, pos = out["geometry"], {p: k for k, p in enumerate(out["geo_pos"])}
        v, _ = best_lr(configs, out, "precision")
        picks = [pos[i] for i, c in enumerate(configs)
                 if c["arm"] == "precision" and abs(c["lr"] - v) < 1e-12 and i in pos]
        row = report_arm(configs, out, "precision", v)
        va, _ = best_lr(configs, out, "adam")
        print(f"\n  --- batch {batch} (centre/noise = "
              f"{task.shift / math.sqrt(task.input_dim / batch):.2f}, "
              f"{out['seconds']:.1f}s, {task.steps()} steps) ---")
        print(HEAD)
        print_row("adam", va, report_arm(configs, out, "adam", va))
        print_row("precision", v, row)
        print(f"{'':>19} cross-unit |cos| {np.median(geo['cos_units'][picks]):.3f}  "
              f"|cos| centre {np.median(geo['cos_centre'][picks]):.3f}  "
              f"eff.rank(C) {np.median(geo['participation'][picks]):.2f}  "
              f"input eff.rank {out['dir_participation']:.2f}")
        print_paired("precision - adam", row, report_arm(configs, out, "adam", va))

    print("\n=== what task property would be required ===")
    print("  Per-unit autonomy needs units to commit to DIFFERENT directions, and")
    print("  the direction is layer-shared, so the only route is that different")
    print("  units be the ones moving in different regions. Measured proxy:")
    print("  cross-unit dispersion of realized plasticity (rho_sd above) against")
    print("  the region count, at fixed total samples.")
    for regions in (2, 8, 32):
        task = Task(base_seed=args.base_seed, batch=args.batch, regions=regions,
                    shift=args.shift, hidden=args.hidden,
                    samples_per_region=32000 // regions)
        configs = grid(("adam", "precision", "precision_shared"), WIDE, args.diag_seeds)
        out = run(configs, task, device, diagnose=True)
        geo, pos = out["geometry"], {p: k for k, p in enumerate(out["geo_pos"])}
        v, _ = best_lr(configs, out, "precision")
        picks = [pos[i] for i, c in enumerate(configs)
                 if c["arm"] == "precision" and abs(c["lr"] - v) < 1e-12 and i in pos]
        row = report_arm(configs, out, "precision", v)
        vs, _ = best_lr(configs, out, "precision_shared")
        va, _ = best_lr(configs, out, "adam")
        print(f"\n  --- regions {regions} x {task.samples_per_region} samples "
              f"({out['seconds']:.1f}s) ---")
        print(HEAD)
        print_row("adam", va, report_arm(configs, out, "adam", va))
        print_row("precision", v, row)
        print_row("precision_shared", vs, report_arm(configs, out, "precision_shared", vs))
        print(f"{'':>19} cross-unit |cos| {np.median(geo['cos_units'][picks]):.3f}  "
              f"eff.rank(C) {np.median(geo['participation'][picks]):.2f}  "
              f"input eff.rank {out['dir_participation']:.2f}")
        print_paired("precision - adam", row, report_arm(configs, out, "adam", va))
        print_paired("precision - precision_shared",
                     row, report_arm(configs, out, "precision_shared", vs))


def exp_ceiling(args, device):
    """Does this task have ANY per-unit headroom? Oracle upper bound.

    `oracle_unit` is told, per unit per sample, whether its own pending step
    lowers the whole-mixture loss, and is allowed to keep or drop it. No causal
    rule reading the unit's state can beat that. Its two controls keep the level
    and destroy only the per-unit correspondence: `oracle_mean` (matched mean,
    zero dispersion) and `oracle_shuffled` (matched mean AND dispersion,
    correspondence permuted away).
    """
    arms = ("adam", "oracle_unit", "oracle_mean", "oracle_shuffled")
    settings = [("baseline shift 12", dict()),
                ("shift 3", dict(shift=3.0)),
                ("batch 1 streaming", dict(batch=1, samples_per_region=4000)),
                ("shared teacher (no conflict)", dict(per_region_teacher=False))]
    for label, kwargs in settings:
        base = dict(base_seed=args.base_seed, batch=args.batch, regions=args.regions,
                    shift=args.shift, hidden=args.hidden,
                    samples_per_region=args.samples)
        base.update(kwargs)
        task = Task(**base)
        configs = grid(arms, WIDE, args.seeds)
        out = run(configs, task, device)
        print(f"\n=== ceiling, {label}: {task.label()}, {args.seeds} seeds "
              f"({out['seconds']:.1f}s) ===")
        print(HEAD)
        rows = {}
        for arm in arms:
            v, _ = best_lr(configs, out, arm)
            rows[arm] = report_arm(configs, out, arm, v)
            print_row(arm, v, rows[arm])
        print_paired("oracle_unit - adam (total oracle value)",
                     rows["oracle_unit"], rows["adam"])
        print_paired("oracle_unit - oracle_mean (dispersion value)",
                     rows["oracle_unit"], rows["oracle_mean"])
        head, _ = print_paired("oracle_unit - oracle_shuffled (PER-UNIT HEADROOM)",
                               rows["oracle_unit"], rows["oracle_shuffled"])
        print_paired("oracle_shuffled - adam (level-only value)",
                     rows["oracle_shuffled"], rows["adam"])
        # what it costs to measure a mechanism that achieves the whole ceiling
        oracle_sd = (rows["oracle_unit"]["overall"]
                     - rows["oracle_shuffled"]["overall"]).std(ddof=1)
        mech_sd = (rows["oracle_unit"]["overall"] - rows["adam"]["overall"]).std(ddof=1)
        baseline = rows["adam"]["overall"].mean()
        print(f"    ceiling is {100 * abs(head) / baseline:.1f}% of the baseline "
              f"score ({baseline:.4f}); a mechanism at 100% of it needs "
              f"{seeds_needed(oracle_sd, head)} paired seeds against its own "
              f"shuffled control, {seeds_needed(mech_sd, head)} against Adam")


def exp_smoke(args, device):
    task = Task(base_seed=args.base_seed, samples_per_region=400, shift=args.shift)
    configs = grid(("adam", "precision", "precision_shared", "precision_diag",
                    "precision_scalar", "precision_rotate"), (1e-3,), 2)
    out = run(configs, task, device, diagnose=True)
    print(f"smoke: {len(configs)} configs, {out['seconds']:.2f}s, "
          f"{task.steps()} steps")
    print(HEAD)
    for arm in dict.fromkeys(c["arm"] for c in configs):
        print_row(arm, 1e-3, report_arm(configs, out, arm, 1e-3))
    print("geometry:", {k: np.round(v, 3).tolist() for k, v in out["geometry"].items()})
    print("input eff.rank:", round(out["dir_participation"], 3))


@dataclass
class Args:
    experiment: str = "power"
    """power | autonomy | ceiling | smoke"""
    seeds: int = 64
    diag_seeds: int = 8
    base_seed: int = 1
    regions: int = 8
    shift: float = 12.0
    batch: int = 8
    hidden: int = 64
    samples: int = 4000
    device: str = "cuda"


def main():
    args = tyro.cli(Args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    table = {"power": exp_power, "autonomy": exp_autonomy, "ceiling": exp_ceiling,
             "smoke": exp_smoke}
    for name in args.experiment.split(","):
        if name not in table:
            raise ValueError(f"experiment must be one of {tuple(table)}")
        table[name](args, device)


if __name__ == "__main__":
    main()

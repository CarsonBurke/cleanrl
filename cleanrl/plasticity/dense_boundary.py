"""Where does per-input plasticity stop paying? A sweep in USEFUL-INPUT DENSITY.

WHY THIS EXISTS. Every measurement of the mirror/softhinge family so far lives
in the sparse regime: 4096 inputs of which 1 predicts, or 1024 of which 4 do. A
rule whose whole job is to refuse junk columns has, by construction, nothing to
refuse when every column informs the target -- and MuJoCo HalfCheetah has 17
observation dimensions that essentially ALL inform the policy. A previous
mechanism in this family decayed monotonically with useful fraction and was
exactly inert (advantage 1.00x) at 100% density. This script asks whether
`softhinge_alloc` does the same, and -- the question that actually decides
whether it can be shipped into PPO -- whether it becomes HARMFUL there.

TASK. Ported from `cleanrl/plasticity/hidden_stream.py`: `input_dim` -> 64 tanh
hidden -> 1 scalar, teacher reads only the leading `useful` columns, Gaussian
label noise, batch 1 single-pass streaming, no replay. Test MSE is measured on
fresh samples against the NOISE-FREE teacher output, so absorbing noise is
punished rather than rewarded; it is reported divided by the zero-predictor's
score, so 1.0 means "no better than predicting nothing".

THE ONE DEVIATION FROM THE PORT, AND WHY IT IS REQUIRED. `hidden_stream` builds
the teacher as `tanh(x[:useful] @ randn(useful, 4))`, whose preactivation sd
grows as `sqrt(useful)`. Sweeping density with that teacher would sweep tanh
SATURATION at the same time: at `useful=1024` the preactivation sd is 32 and the
teacher degenerates into a sum of four sign functions. This script fixes the
preactivation sd at 2.0 for every density (`2/sqrt(useful)` scaling), which is
EXACTLY the reference teacher at `useful=4` and holds target difficulty constant
across the axis being swept. Without it the density axis is confounded.

METHODS. `adam` (level 1 everywhere), `mirror_alloc` (1 - FDP), `softhinge_alloc`
(`softplus(k(1 - z^2/t^2))/k`, k=24), `oracle` (told which columns are useful).
The `alloc` variants pool evidence over the hidden units that share an input,
renormalise the per-input level to mean one and cap it, so shut connections FUND
larger steps on confident ones. At 100% density the oracle mask is all ones, so
`oracle` IS `adam` up to nothing at all -- that is the arithmetic, not a bug, and
it is the control that says the density axis was actually reached.

PROTOCOL (the part that makes the numbers admissible).
* Every arm is scored at ITS OWN best learning rate over a grid wide enough to
  bracket its optimum. A winner sitting on a grid edge is reported VOID.
* Seeds are PAIRED: within a seed the teacher, the stream, the label noise, the
  Rademacher twin and the student init are bit-identical across arms, and all of
  them are resampled across seeds. The headline is the paired difference and its
  paired SD, never a pair of per-arm intervals.
* Seeds AND configs advance in one vectorized pass -- `w1` is
  `(seeds, configs, hidden, inputs)` -- so a whole density point is one kernel
  stream rather than 8 x 32 sequential runs.
"""

import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
import tyro

METHODS = ("adam", "mirror_alloc", "softhinge_alloc", "oracle")


@dataclass
class Args:
    input_dim: int = 1024
    useful: int = 4
    hidden: int = 64
    samples: int = 60000
    batch: int = 1
    """batch 1 is the native case: nothing in the rule needs a batch to estimate"""
    eval_steps: int = 4096
    eval_chunk: int = 512
    noise_std: float = 8.0
    seeds: int = 8
    lr_grid: tuple[float, ...] = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
                                  1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    """half-decade steps over five decades. The `alloc` arms multiply their
    nominal LR by a mean-one level capped at 128 -- measured level 55 on the
    useful columns at 0.4% density -- so their optimum sits one to two decades
    BELOW Adam's and a grid that starts at Adam's optimum would pin them at an
    edge. That exact error (a control held above its own optimum) has faked
    results in this family before, so every arm's winner must be interior."""
    alloc_cap: float = 128.0
    hinge_sharpness: float = 24.0
    refresh_every: int = 50
    sweep: str = "density"
    """`density` runs the full boundary sweep; `single` runs one setting"""
    density_grid: tuple[int, ...] = (4, 16, 64, 256, 1024)
    dense_noise_grid: tuple[float, ...] = (2.0, 8.0)
    """extra noise levels run at 100% density, to separate `no junk to reject`
    from `no noise to reject`"""
    seed: int = 1
    cuda: bool = True


def teacher_forward(x, useful, weights):
    """The reference teacher, with preactivation sd held at 2.0 for any `useful`."""
    first, second = weights
    return torch.tanh(x[..., :useful] @ first).mul(second).sum(-1)


def fdp_level(t_obs, t_null):
    """One minus the false-discovery proportion, per row. `(rows, width)` in/out."""
    width = t_obs.shape[-1]
    null_sorted = t_null.sort(dim=-1).values
    obs_sorted = t_obs.sort(dim=-1).values
    false_ge = width - torch.searchsorted(null_sorted, t_obs.contiguous(), right=False)
    total_ge = (width - torch.searchsorted(obs_sorted, t_obs.contiguous(),
                                           right=False)).clamp_min(1)
    return (1.0 - false_ge.float() / total_ge.float()).clamp_(0.0, 1.0)


def softhinge_level(obs_sq, null_sq, sharpness):
    """`softplus(k (1 - z^2/t^2)) / k`, z^2 the family-wise max of the null twin.

    `obs_sq`/`null_sq` are `(..., width)`; the family is the width axis, so a
    null coordinate lands at `softplus(-k)/k = 1.6e-12`, far below the
    `1/sqrt(width)` magnitude at which absorbed output noise starts to matter,
    while a certified one saturates at exactly 1 instead of being taxed forever.
    """
    z_sq = null_sq.max(dim=-1, keepdim=True).values
    return F.softplus(sharpness * (1.0 - z_sq / obs_sq.clamp_min(1e-30))) / sharpness


def allocate(level, cap):
    """Renormalise a per-input level to mean one and cap it: budget reallocation."""
    width = level.shape[-1]
    mean_one = level * (width / level.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    return mean_one.clamp_(0.0, cap)


def run_point(args, configs, device):
    """One (density, noise) point: all `configs` x all seeds in a single pass."""
    n_seed, n_cfg = args.seeds, len(configs)
    dim, hidden, useful = args.input_dim, args.hidden, args.useful
    gen = torch.Generator(device=device).manual_seed(args.seed)
    sign_gen = torch.Generator(device=device).manual_seed(args.seed + 1)

    # PAIRING: teacher and init are drawn per SEED and shared across configs, so
    # every arm inside a seed sees the identical problem and the identical start.
    teacher = (torch.randn((n_seed, useful, 4), device=device, generator=gen)
               * (2.0 / np.sqrt(useful)),
               torch.randn((n_seed, 1, 4), device=device, generator=gen) * 0.7)
    init = torch.randn((n_seed, 1, hidden, dim), device=device,
                       generator=gen) / np.sqrt(dim)

    lr = torch.tensor([c["lr"] for c in configs], device=device).view(1, n_cfg, 1, 1)
    kinds = [c["method"] for c in configs]
    def mask_of(name):
        return torch.tensor([k == name for k in kinds],
                            device=device).view(1, n_cfg, 1, 1)
    is_soft, is_mirror, is_oracle = (mask_of("softhinge_alloc"),
                                     mask_of("mirror_alloc"), mask_of("oracle"))
    oracle_level = torch.zeros((1, 1, 1, dim), device=device)
    oracle_level[..., :useful] = 1.0

    w1 = init.expand(n_seed, n_cfg, hidden, dim).contiguous()
    w2 = torch.zeros((n_seed, n_cfg, hidden), device=device)
    m1, v1 = torch.zeros_like(w1), torch.zeros_like(w1)
    m2, v2 = torch.zeros_like(w2), torch.zeros_like(w2)
    a1, q1, r1 = (torch.zeros_like(w1) for _ in range(3))
    level1 = torch.ones((n_seed, n_cfg, 1, dim), device=device)

    squared = torch.zeros((n_seed, n_cfg), device=device)
    trivial = torch.zeros((n_seed,), device=device)
    beta1, beta2 = 0.9, 0.999
    batch = args.batch
    steps = max(args.samples // batch, 1)
    for step in range(steps):
        x = torch.randn((n_seed, batch, dim), device=device, generator=gen)
        clean = teacher_forward(x, useful, teacher)                    # (S, B)
        y = clean + args.noise_std * torch.randn((n_seed, batch), device=device,
                                                 generator=gen)
        act = torch.tanh(torch.einsum("skhd,sbd->skbh", w1, x))        # (S, K, B, H)
        prediction = (act * w2.unsqueeze(2)).sum(-1)                   # (S, K, B)
        residual = prediction - y.unsqueeze(1)
        squared += (prediction.detach() - clean.unsqueeze(1)).square().mean(-1)
        trivial += clean.square().mean(-1)

        # manual backward of 0.5 * mean_b residual^2
        g2 = torch.einsum("skb,skbh->skh", residual, act) / batch
        back = residual.unsqueeze(-1) * w2[:, :, None, :] * (1.0 - act.square())
        g1 = torch.einsum("skbh,sbd->skhd", back, x) / batch

        # evidence comes from the RAW gradient; nothing below modifies it
        a1 += g1
        q1 += g1 * g1
        flips = torch.randint(0, 2, (n_seed, 1, 1, dim), device=device,
                              generator=sign_gen, dtype=torch.float32
                              ).mul_(2.0).sub_(1.0)
        r1.addcmul_(g1, flips)
        if step % args.refresh_every == 0:
            scale = q1.sqrt().clamp_min(1e-30)
            obs_pooled = (a1 / scale).square().sum(2)                  # (S, K, D)
            null_pooled = (r1 / scale).square().sum(2)
            hinge = allocate(softhinge_level(obs_pooled, null_pooled,
                                             args.hinge_sharpness), args.alloc_cap)
            fdp = allocate(fdp_level(obs_pooled.reshape(-1, dim),
                                     null_pooled.reshape(-1, dim)
                                     ).view(n_seed, n_cfg, dim), args.alloc_cap)
            level1 = torch.where(
                is_soft, hinge.unsqueeze(2),
                torch.where(is_mirror, fdp.unsqueeze(2),
                            torch.where(is_oracle, oracle_level,
                                        torch.ones_like(oracle_level))))

        m1.mul_(beta1).add_(g1, alpha=1 - beta1)
        v1.mul_(beta2).addcmul_(g1, g1, value=1 - beta2)
        m2.mul_(beta1).add_(g2, alpha=1 - beta1)
        v2.mul_(beta2).addcmul_(g2, g2, value=1 - beta2)
        bias1, bias2 = 1 - beta1 ** (step + 1), 1 - beta2 ** (step + 1)
        # the level scales the REALIZED Adam step, never the gradient
        w1 -= (lr * level1) * (m1 / bias1) / ((v1 / bias2).sqrt() + 1e-8)
        w2 -= lr.view(1, n_cfg, 1) * (m2 / bias1) / ((v2 / bias2).sqrt() + 1e-8)

    with torch.no_grad():
        squared_eval = torch.zeros((n_seed, n_cfg), device=device)
        variance = torch.zeros((n_seed,), device=device)
        done = 0
        while done < args.eval_steps:
            chunk = min(args.eval_chunk, args.eval_steps - done)
            x = torch.randn((n_seed, chunk, dim), device=device, generator=gen)
            clean = teacher_forward(x, useful, teacher)
            act = torch.tanh(torch.einsum("skhd,sbd->skbh", w1, x))
            prediction = (act * w2.unsqueeze(2)).sum(-1)
            squared_eval += (prediction - clean.unsqueeze(1)).square().sum(-1)
            variance += clean.square().sum(-1)
            done += chunk
    ratio = (squared_eval / variance.unsqueeze(1)).cpu().numpy()
    return {"ratio": ratio,
            "prequential": (squared / trivial.unsqueeze(1)).cpu().numpy(),
            "trivial": (variance / args.eval_steps).cpu().numpy(),
            "level_useful": level1[..., 0, :useful].mean(-1).cpu().numpy(),
            "level_junk": level1[..., 0, useful:].mean(-1).cpu().numpy()
            if useful < dim else np.zeros((args.seeds, len(configs))),
            "useful_w": w1[..., :useful].square().mean((2, 3)).sqrt().cpu().numpy(),
            "junk_w": w1[..., useful:].square().mean((2, 3)).sqrt().cpu().numpy()
            if useful < dim else np.zeros((args.seeds, len(configs)))}


def summarize(args, configs, out, methods):
    """Per-arm best LR over the seed mean, plus edge-of-grid detection."""
    ratio = out["ratio"]                                        # (seeds, configs)
    best = {}
    for method in methods:
        index = [i for i, c in enumerate(configs) if c["method"] == method]
        means = ratio[:, index].mean(0)
        pick = int(means.argmin())
        best[method] = {"lr": configs[index[pick]]["lr"],
                        "column": index[pick],
                        "mean": float(means[pick]),
                        "per_seed": ratio[:, index[pick]],
                        "grid": [(configs[i]["lr"], float(m))
                                 for i, m in zip(index, means)],
                        "edge": pick in (0, len(index) - 1),
                        "level_useful": float(out["level_useful"][:, index[pick]].mean()),
                        "level_junk": float(out["level_junk"][:, index[pick]].mean())}
    return best


def paired(best, treatment, control):
    diff = best[treatment]["per_seed"] - best[control]["per_seed"]
    n = len(diff)
    return {"mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
            "se": float(diff.std(ddof=1) / np.sqrt(n)),
            "t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n) + 1e-30))}


def report_point(label, args, configs, out, methods):
    best = summarize(args, configs, out, methods)
    print(f"\n=== {label} ===")
    print(f"{'method':>15s} {'best lr':>8s} {'test/zero':>10s} {'edge':>5s} "
          f"{'lvl use':>8s} {'lvl junk':>9s}")
    for method in methods:
        b = best[method]
        flag = "VOID" if b["edge"] else ""
        print(f"{method:>15s} {b['lr']:>8g} {b['mean']:>10.5f} {flag:>5s} "
              f"{b['level_useful']:>8.4f} {b['level_junk']:>9.2e}")
        print(f"{'':16s}grid " + " ".join(f"{lr:g}:{m:.4f}" for lr, m in b["grid"]))
    for control in ("mirror_alloc", "adam"):
        if control in methods and "softhinge_alloc" in methods:
            d = paired(best, "softhinge_alloc", control)
            print(f"  paired softhinge_alloc - {control:>12s}: "
                  f"{d['mean']:+.5f} (paired SD {d['sd']:.5f}, "
                  f"SE {d['se']:.5f}, t {d['t']:+.2f}, n={args.seeds})")
    return best


def main():
    args = tyro.cli(Args)
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda" if args.cuda else "cpu")
    methods = METHODS
    configs = [{"method": m, "lr": lr} for m in methods for lr in args.lr_grid]

    points = []
    if args.sweep == "single":
        points.append((args.useful, args.noise_std))
    else:
        for useful in args.density_grid:
            points.append((useful, args.noise_std))
        for noise in args.dense_noise_grid:
            if noise != args.noise_std:
                points.append((args.input_dim, noise))
    print(f"{args.input_dim} inputs -> {args.hidden} hidden, batch {args.batch}, "
          f"{args.samples} samples, {args.seeds} paired seeds, "
          f"cap {args.alloc_cap:g}, k {args.hinge_sharpness:g}")
    print(f"lr grid {args.lr_grid}")

    table = []
    for useful, noise in points:
        point_args = replace(args, useful=useful, noise_std=noise)
        started = time.perf_counter()
        out = run_point(point_args, configs, device)
        best = report_point(f"useful {useful}/{args.input_dim} "
                            f"({100.0 * useful / args.input_dim:.3g}%), noise {noise:g} "
                            f"[{time.perf_counter() - started:.0f}s]",
                            point_args, configs, out, methods)
        table.append((useful, noise, best))

    print("\n" + "=" * 118)
    print(f"{'useful':>10s} {'noise':>6s} | " + " ".join(
        f"{m[:13]:>17s}" for m in methods)
        + f" | {'sh - mirror':>20s} {'sh - adam':>20s}")
    for useful, noise, best in table:
        cells = " ".join(
            f"{best[m]['mean']:9.5f}@{best[m]['lr']:<6g}{'V' if best[m]['edge'] else ' '}"
            for m in methods)
        dm, da = paired(best, "softhinge_alloc", "mirror_alloc"), \
            paired(best, "softhinge_alloc", "adam")
        print(f"{useful:>4d}/{args.input_dim:<5d} {noise:>6g} | {cells} | "
              f"{dm['mean']:+8.5f}+-{dm['sd']:<7.5f} {da['mean']:+8.5f}+-{da['sd']:<7.5f}")
    print("V = winner sits on a grid edge -> VOID, widen the grid and rerun.")


if __name__ == "__main__":
    main()

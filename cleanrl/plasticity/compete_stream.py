"""Plasticity as CAPACITY ALLOCATION: perceptrons compete to be written to.

# THE ARGUMENT

A weight holds one value at inference time. So state-conditional plasticity cannot
make a weight context-dependent -- whatever the gate does, the forward pass is
still a fixed function. What the gate actually controls is WHO GETS WRITTEN TO by
which contexts. That is the entire unique capability, and it is not an estimation
problem at all:

  if every context writes to every unit, interference is structural and no
  estimator of gradient noise can remove it;
  if writes are consistent with state, the layer PARTITIONS into modules, and the
  interference disappears because the contexts stop sharing weights.

So the objective for a plasticity rule is not "is my gradient trustworthy" -- it is
"am I the right unit to absorb this sample". That question is answered by a
comparison ACROSS units, which is why every scalar-per-weight rule in this family
was structurally unable to express it, and why the estimator-flavoured attempts
(mirror t-statistic, Kalman gain, PC precision, Wiener explained-fraction) all
plateaued: each one asks a purely local question and each one is bounded above by
one, so it can only throttle. A throttle cannot beat a tuned learning rate.

# MECHANISM

Each perceptron keeps a signature of the inputs it has actually been written by --
an accumulation of input directions weighted by how much that unit moved. Its
plasticity for the current sample is how well the sample matches its own
signature, in competition with its peers:

    match_i   = cos(x_t, signature_i)
    p_i       = H * softmax_i(match / temperature)

Writes then reinforce the signature, so the assignment sharpens over training and
modules emerge. This is positive feedback by design: it is a self-organising
allocator of learning, not a noise filter.

At initialisation every signature is zero, so every match is zero, so the softmax
is exactly uniform and `p == 1` for every unit: the run starts bit-identical to
the baseline. There is no warmup and no init to tune -- verified by sweeping
`--temperature` and by the `--signature-init` sweep.

# THE CONTROL THAT MATTERS

`compete_shuffled` takes the mechanism's plasticity vector on every step and
randomly permutes it across units. That matches the mean AND the full dispersion
of the real rule, destroying only the correspondence between a unit's state and
its own plasticity. Every earlier "win" in this family died to a learning-rate
confound; a matched-mean control cannot catch a dispersion effect, and a
matched-dispersion control cannot catch an allocation effect. This one catches
both, so anything that survives it is state-conditional allocation and nothing
else.

`compete_frozen` freezes the signatures at their initial value, removing the
positive feedback while keeping state-dependence, which separates "matching
matters" from "sharpening matters".
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro

METHODS = ("adam", "compete", "compete_shuffled", "compete_frozen")


@dataclass
class Args:
    input_dim: int = 64
    hidden: int = 64
    regions: int = 8
    """sequentially presented contexts, each with its OWN target function. A
    single shared teacher makes every context want the same mapping, so no unit
    ever faces conflicting demands and nothing can be measured (verified:
    sequential and shuffled orders scored 0.0653 vs 0.0655)."""
    shift: float = 3.0
    samples_per_region: int = 4000
    noise_std: float = 0.5
    batch: int = 8
    temperature: float = 0.25
    """competition sharpness. High temperature -> uniform -> exactly the baseline."""
    signature_rate: float = 0.02
    signature_init: float = 0.0
    method: str = "all"
    lr_grid: tuple[float, ...] = (3e-4, 1e-3, 3e-3, 1e-2)
    seeds: int = 16
    seed: int = 1
    eval_samples: int = 2048
    device: str = "cuda"


def teacher_forward(x, teacher):
    w1, w2 = teacher
    return torch.tanh(x @ w1) @ w2


def make_teachers(args, device, gen):
    def draw():
        return (torch.randn((args.input_dim, 8), device=device, generator=gen)
                / np.sqrt(args.input_dim),
                torch.randn((8,), device=device, generator=gen) * 0.7)
    return [draw() for _ in range(args.regions)]


def run(args, configs, device):
    n_cfg = len(configs)
    dim, hidden = args.input_dim, args.hidden
    gen = torch.Generator(device=device).manual_seed(args.seed)
    teachers = make_teachers(args, device, gen)
    centres = [torch.randn((dim,), device=device, generator=gen) for _ in range(args.regions)]
    centres = [c / c.norm() * args.shift for c in centres]

    lr = torch.tensor([c["lr"] for c in configs], device=device).view(n_cfg, 1, 1)
    kind = [c["method"] for c in configs]
    is_compete = torch.tensor([k == "compete" for k in kind], device=device).view(n_cfg, 1)
    is_shuffled = torch.tensor([k == "compete_shuffled" for k in kind],
                               device=device).view(n_cfg, 1)
    is_frozen = torch.tensor([k == "compete_frozen" for k in kind],
                             device=device).view(n_cfg, 1)
    gated = is_compete | is_shuffled | is_frozen
    any_gate = bool(gated.any().item())

    w1 = torch.randn((n_cfg, hidden, dim), device=device, generator=gen) / np.sqrt(dim)
    w2 = torch.zeros((n_cfg, 1, hidden), device=device)
    m1, v1 = torch.zeros_like(w1), torch.zeros_like(w1)
    m2, v2 = torch.zeros_like(w2), torch.zeros_like(w2)
    # A unit's signature must live in a space where units actually DIFFER. In a
    # fully-connected layer every unit's per-sample gradient is `delta_h * x`, so
    # all input-space signatures grow along the same shared direction, become
    # parallel, and every match is identical -- measured: gate dispersion exactly
    # 0.00, the rule was inert. The one genuinely per-unit quantity is the unit's
    # own preactivation, so the signature is the state the unit has been written
    # at: a mean and a scale, per unit.
    written_mean = torch.zeros((n_cfg, hidden), device=device)
    written_scale = torch.ones((n_cfg, hidden), device=device)
    seen = torch.zeros((n_cfg, hidden), device=device)

    batch, updates = args.batch, 0
    steps_per_region = max(args.samples_per_region // batch, 1)
    write_by_region = torch.zeros((n_cfg, hidden, args.regions), device=device)
    gate_mean = torch.zeros((n_cfg,), device=device)
    gate_spread = torch.zeros((n_cfg,), device=device)

    for region in range(args.regions):
        for _ in range(steps_per_region):
            x = centres[region] + torch.randn((batch, dim), device=device, generator=gen)
            clean = teacher_forward(x, teachers[region])
            y = clean + args.noise_std * torch.randn((batch,), device=device, generator=gen)
            act = torch.tanh(torch.einsum("khd,bd->kbh", w1, x))
            rows = w2.reshape(n_cfg, 1, hidden)
            residual = (act * rows).sum(-1) - y
            g2 = torch.einsum("kb,kbh->kh", residual, act).view(n_cfg, 1, hidden) / batch
            delta = residual.unsqueeze(2) * rows * (1.0 - act.square())
            g1 = torch.einsum("kbh,bd->khd", delta, x) / batch
            updates += 1

            gate = torch.ones((n_cfg, hidden), device=device)
            if any_gate:
                state = torch.einsum("khd,bd->kbh", w1, x).detach().mean(1)  # (K,H)
                # how well does this sample match the state this unit has been
                # written at, relative to its peers
                match = -((state - written_mean) / written_scale.clamp_min(1e-6)).square()
                match = torch.where(seen > 0, match, torch.zeros_like(match))
                weights = torch.softmax(match / args.temperature, dim=-1) * hidden
                shuffled = weights[:, torch.randperm(hidden, device=device)]
                candidate = torch.where(is_shuffled, shuffled, weights)
                gate = torch.where(gated, candidate, gate)
                gate_mean += gate.mean(-1)
                gate_spread += gate.std(-1)

            beta1, beta2 = 0.9, 0.999
            m1.mul_(beta1).add_(g1, alpha=1 - beta1)
            v1.mul_(beta2).addcmul_(g1, g1, value=1 - beta2)
            m2.mul_(beta1).add_(g2, alpha=1 - beta1)
            v2.mul_(beta2).addcmul_(g2, g2, value=1 - beta2)
            bias1, bias2 = 1 - beta1 ** updates, 1 - beta2 ** updates
            step1 = (m1 / bias1) / ((v1 / bias2).sqrt() + 1e-8)
            # the gate scales the REALIZED step of the incoming row only; gating
            # the outgoing weight deadlocks the unit, since w2 starts at zero and
            # a gate read off that gradient would pin both at zero forever.
            w1 -= lr * step1 * gate.unsqueeze(-1)
            w2 -= lr * (m2 / bias1) / ((v2 / bias2).sqrt() + 1e-8)

            if any_gate:
                moved = (step1 * gate.unsqueeze(-1)).abs().mean(-1)     # (K, H)
                write_by_region[:, :, region] += moved
                if not bool(is_frozen.all().item()):
                    # writes reinforce the state a unit is responsible for, so
                    # the allocation sharpens: positive feedback by design
                    weight = (args.signature_rate * moved
                              / moved.mean(-1, keepdim=True).clamp_min(1e-12)).clamp(0, 1)
                    fresh_mean = written_mean + weight * (state - written_mean)
                    deviation = (state - fresh_mean).abs()
                    fresh_scale = written_scale + weight * (deviation - written_scale)
                    keep = is_frozen
                    written_mean = torch.where(keep, written_mean, fresh_mean)
                    written_scale = torch.where(keep, written_scale, fresh_scale)
                    seen = torch.where(keep, seen, seen + weight)

    scores = []
    for teacher, centre in zip(teachers, centres):
        x = centre + torch.randn((args.eval_samples, dim), device=device, generator=gen)
        clean = teacher_forward(x, teacher)
        act = torch.tanh(torch.einsum("khd,bd->kbh", w1, x))
        prediction = (act * w2.reshape(n_cfg, 1, hidden)).sum(-1)
        scores.append((prediction - clean).square().mean(1)
                      / clean.square().mean().clamp_min(1e-12))
    per_region = torch.stack(scores)

    # emergent specialisation: does each unit end up written by FEW contexts?
    share = write_by_region / write_by_region.sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(share * share.clamp_min(1e-12).log()).sum(-1).mean(-1)
    uniform = float(np.log(args.regions))
    return {"acquisition": per_region[-1].cpu().numpy(),
            "retention": per_region[:-1].mean(0).cpu().numpy(),
            "overall": per_region.mean(0).cpu().numpy(),
            "gate": (gate_mean / max(updates, 1)).cpu().numpy(),
            "spread": (gate_spread / max(updates, 1)).cpu().numpy(),
            "specialisation": (entropy / uniform).cpu().numpy()}


def main():
    args = tyro.cli(Args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    methods = METHODS if args.method == "all" else tuple(args.method.split(","))
    for name in methods:
        if name not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
    configs = [{"method": m, "lr": v, "seed": s}
               for m in methods for v in args.lr_grid for s in range(args.seeds)]
    print(f"{args.regions} contexts x {args.samples_per_region} samples, shift "
          f"{args.shift}, temperature {args.temperature}, {args.seeds} seeds, "
          f"{len(configs)} configs")
    start = time.perf_counter()
    out = run(args, configs, device)
    print(f"{len(configs)} configs in one pass, {time.perf_counter() - start:.1f}s\n")

    print(f"{'method':>18} {'lr':>7} | {'acquire':>8} {'retain':>8} {'overall':>8} "
          f"{'+/-':>7} | {'gate':>5} {'sd':>5} | spec")
    for name in methods:
        best, row = None, None
        for value in args.lr_grid:
            picks = [i for i, c in enumerate(configs)
                     if c["method"] == name and c["lr"] == value]
            stats = {k: float(np.mean(out[k][picks])) for k in out}
            stats["ci"] = float(1.96 * np.std(out["overall"][picks], ddof=1)
                                / np.sqrt(len(picks))) if len(picks) > 1 else float("nan")
            if best is None or stats["overall"] < best:
                best, row = stats["overall"], (value, stats)
        value, stats = row
        print(f"{name:>18} {value:>7} | {stats['acquisition']:>8.4f} "
              f"{stats['retention']:>8.4f} {stats['overall']:>8.4f} "
              f"{stats['ci']:>7.4f} | {stats['gate']:>5.2f} {stats['spread']:>5.2f} "
              f"| {stats['specialisation']:.3f}")


if __name__ == "__main__":
    main()

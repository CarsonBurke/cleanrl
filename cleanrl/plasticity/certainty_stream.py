"""Plasticity as CERTAINTY: each perceptron predicts its own teaching signal.

# PREMISE

A perceptron receives an error signal `delta` and has to decide how much of its
incoming weights to move. The decision it should make is: how much of this signal
do I actually understand? So it carries a predictor of its own `delta`, fitted from
ITS OWN STATE by proper heteroscedastic NLL, giving two things per sample:

    mu(state)     the part of the teaching signal it can predict
    sigma(state)  the part it must attribute to noise

and its plasticity is the fraction of the signal that is explained,

    certainty = mu^2 / (mu^2 + sigma^2)   in [0, 1]

Truly random signal -> mu = 0 -> certainty 0 -> no update, no matter how large the
gradient is. Fully explained signal -> certainty 1 -> the ordinary step. Nothing is
clipped, nothing is renormalised, and there is no free scale: the gate is a
probability, so it cannot masquerade as a learning-rate change (it can only ever
reduce the step, and its ceiling is exactly the baseline).

# WHY THIS AND NOT THE EARLIER MEMBERS OF THE FAMILY

This is the same ratio the family has circled for a long time, but estimated
CONDITIONALLY instead of marginally:

  mirror's statistic  t = |sum g| / sqrt(sum g^2) ~ sqrt(n) * mu / sigma
      the same ratio, estimated MARGINALLY over time. It works, and it is why the
      mirror family measured real selectivity -- but a marginal estimate is one
      scalar per weight, so it can never say "understood here, noise there".

  v8/v9 energy predictor    predicted E[delta^2] = mu^2 + sigma^2
      the second moment alone cannot separate signal from noise: measured 0.94x
      selectivity, i.e. WORSE than plain Adam.

  kalman / RLS gain         P phi^2 / (P phi^2 + R), also second-moment driven
      credit-blind for the same reason: shrinks a junk connection's variance as
      fast as a useful one. Measured 0.684, no better than Adam.

Note the algebra that killed the marginal version does NOT apply here. Marginally,
mu^2/(mu^2+sigma^2) = mu^2/E[delta^2] with both terms running statistics of one
stream, so the gate carries no information the second moment did not already have.
Conditioned on state, mu(s) and sigma(s) are separate functions and the identity
breaks -- which is exactly the degree of freedom the premise needs. The debiased
state-conditioned estimator already measured 13.60 vs 4.21 marginal (1.73x, 16
seeds) on the heteroscedastic regime stream in `noisy_stream_diagnostic.py`; this
file asks whether that survives inside a hidden layer of a network.

# WHAT IS PREDICTED, AND WHY NOT THE OTHER CANDIDATES

In predictive coding proper a unit predicts ACTIVITY -- the layer below, or its own
value -- and never weights. The three candidate targets are not equivalent:

  its own future weights   a weight is a deterministic function of its own past
                           gradients, so predicting it is a smoothed gradient.
                           That is momentum / IDBD, already measured inert here.
  the next layer's weights not locally available, and they move for reasons that
                           have nothing to do with this unit. Not well posed.
  its own teaching signal  locally available, and its predictability is exactly
                           the question "is this correction something I
                           understand, or noise?". This is PC's error unit, and
                           it is what this file predicts.

One deliberate divergence from textbook PC: PC precision-weights updates by
1/sigma^2, i.e. the second moment alone. That is the `certainty_var` arm below and
it is the same error v8/v9 made (measured 0.94x, worse than Adam). Weighting by the
explained FRACTION mu^2/(mu^2+sigma^2) uses the first moment as well, which is the
part that carries credit.

# ARMS

  adam                  control, LR swept
  certainty             mu and sigma are functions of the unit's own state
  certainty_marginal    mu and sigma are state-independent constants per unit.
                        This is the mirror/marginal case and isolates exactly the
                        STATE-CONDITIONAL content of the gate.
  certainty_var         gate = 1/(1+sigma^2), second moment only. Predicted to
                        fail; it is the v8/v9 error reproduced as a control.

The predictor is trained by supervised NLL on the signal the unit actually
received. It is NOT an outcome/meta objective -- every one-step-lookahead
meta-gradient tried in this family collapsed to inert (level pinned at 1.000).
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro

METHODS = ("adam", "certainty", "certainty_marginal", "certainty_var")


@dataclass
class Args:
    input_dim: int = 256
    hidden: int = 64
    useful: int = 8
    """input coordinates that carry signal; the rest are distractors"""
    samples: int = 60000
    batch: int = 8
    noise_std: float = 4.0
    """target noise in the QUIET regime"""
    noise_ratio: float = 8.0
    """noise multiplier in the LOUD regime. The regime must be visible IN THE
    UNIT'S OWN STATE or a state-conditional gate cannot beat a marginal one --
    measured: keying the regime to a single input coordinate gave loud/quiet
    certainty 0.084 vs 0.085, i.e. no discrimination, because a unit's
    preactivation is a random projection over `input_dim` dims and one
    coordinate contributes ~1/sqrt(input_dim) of its variance. So the regime is
    keyed to the magnitude of the TEACHER's own output: noise grows where the
    signal is large, which is ordinary heteroscedasticity and is visible to any
    unit that has learned signal features. With ratio 1.0 the stream is
    homoscedastic and the state-conditional arm should be INERT, not harmful."""
    loud_threshold: float = 0.8
    """|clean| above which the stream is in its loud regime"""
    predictor_lr: float = 1e-2
    logvar_init: float = 0.0
    """initial log noise estimate. There is deliberately NO warmup phase: the
    predictor observes `delta` whether or not the gate lets the weights move, so
    certainty is self-starting -- a unit that cannot yet predict its own signal
    refuses to move, fits its predictor from the signal anyway, and opens as soon
    as it can explain something. Sensitivity to this init is measured, not
    assumed: see the `--logvar-init` sweep."""
    method: str = "all"
    lr_grid: tuple[float, ...] = (3e-4, 1e-3, 3e-3, 1e-2)
    seeds: int = 8
    seed: int = 1
    eval_samples: int = 4096
    device: str = "cuda"


def teacher_forward(x, teacher):
    w1, w2 = teacher
    return torch.tanh(x @ w1) @ w2


def run(args, configs, device):
    n_cfg = len(configs)
    dim, hidden, useful = args.input_dim, args.hidden, args.useful
    gen = torch.Generator(device=device).manual_seed(args.seed)
    teacher = (torch.randn((useful, 4), device=device, generator=gen),
               torch.randn((4,), device=device, generator=gen) * 0.7)

    lr = torch.tensor([c["lr"] for c in configs], device=device).view(n_cfg, 1, 1)
    kind = [c["method"] for c in configs]
    is_state = torch.tensor([k == "certainty" for k in kind], device=device).view(n_cfg, 1)
    is_marginal = torch.tensor([k == "certainty_marginal" for k in kind],
                               device=device).view(n_cfg, 1)
    is_var = torch.tensor([k == "certainty_var" for k in kind], device=device).view(n_cfg, 1)
    gated = is_state | is_marginal | is_var
    any_gate = bool(gated.any().item())

    w1 = torch.randn((n_cfg, hidden, dim), device=device, generator=gen) / np.sqrt(dim)
    w2 = torch.zeros((n_cfg, 1, hidden), device=device)
    m1, v1 = torch.zeros_like(w1), torch.zeros_like(w1)
    m2, v2 = torch.zeros_like(w2), torch.zeros_like(w2)

    # each perceptron's predictor of its own teaching signal: two heads over its
    # own state. Zero init means mu = 0 and log sigma^2 = 0, i.e. it starts by
    # claiming the signal is pure noise -- hence the warmup.
    n_feat = 3
    head_mu = torch.zeros((n_cfg, hidden, n_feat), device=device)
    head_logvar = torch.zeros((n_cfg, hidden, n_feat), device=device)
    head_logvar[:, :, -1] = args.logvar_init

    batch = args.batch
    steps = max(args.samples // batch, 1)
    squared = torch.zeros((n_cfg,), device=device)
    trivial = torch.zeros((n_cfg,), device=device)
    scored = 0
    gate_sum = torch.zeros((n_cfg,), device=device)
    gate_sq = torch.zeros((n_cfg,), device=device)
    gate_spread = torch.zeros((n_cfg,), device=device)
    loud_gate = torch.zeros((n_cfg,), device=device)
    quiet_gate = torch.zeros((n_cfg,), device=device)
    loud_count = quiet_count = gate_count = 0

    for step in range(steps):
        x = torch.randn((batch, dim), device=device, generator=gen)
        clean = teacher_forward(x[:, :useful], teacher)
        loud = clean.abs() > args.loud_threshold      # regime visible in unit state
        scale = torch.where(loud, args.noise_std * args.noise_ratio, args.noise_std)
        y = clean + scale * torch.randn((batch,), device=device, generator=gen)

        pre = torch.einsum("khd,bd->kbh", w1, x)
        act = torch.tanh(pre)
        rows = w2.reshape(n_cfg, 1, hidden)
        prediction = (act * rows).sum(-1)
        residual = prediction - y
        if step >= steps // 2:
            squared += (prediction.detach() - clean).square().mean(1)
            trivial += clean.square().mean()
            scored += 1

        # the teaching signal this perceptron actually received
        delta = (residual.unsqueeze(2) * rows * (1.0 - act.square())).detach()  # (K,B,H)
        g2 = torch.einsum("kb,kbh->kh", residual, act).view(n_cfg, 1, hidden) / batch
        g1 = torch.einsum("kbh,bd->khd", delta, x) / batch

        gate = torch.ones((n_cfg, hidden), device=device)
        if any_gate:
            state = torch.stack([pre.detach(), pre.detach().abs(),
                                 torch.ones_like(pre)], dim=-1)          # (K,B,H,F)
            marginal_state = torch.zeros_like(state)
            marginal_state[..., -1] = 1.0
            # the marginal arm sees ONLY the bias feature, so its mu and sigma are
            # per-unit constants: the mirror case, with state removed and nothing
            # else changed.
            active = torch.where(is_marginal.view(n_cfg, 1, 1, 1), marginal_state, state)
            mu = (active * head_mu.unsqueeze(1)).sum(-1)                 # (K,B,H)
            logvar = (active * head_logvar.unsqueeze(1)).sum(-1).clamp(-8.0, 8.0)
            var = logvar.exp()

            # heteroscedastic NLL of the observed signal, per (sample, perceptron)
            error = delta - mu
            grad_mu = -(error / var)
            grad_logvar = 0.5 * (1.0 - error.square() / var)
            head_mu -= args.predictor_lr * torch.einsum(
                "kbh,kbhf->khf", grad_mu, active) / batch
            head_logvar -= args.predictor_lr * torch.einsum(
                "kbh,kbhf->khf", grad_logvar, active) / batch

            explained = mu.square()
            fraction = explained / (explained + var).clamp_min(1e-30)
            second_moment_only = 1.0 / (1.0 + var)
            candidate = torch.where(is_var.view(n_cfg, 1, 1),
                                    second_moment_only, fraction).mean(1)
            gate = torch.where(gated, candidate, gate)
            gate_sum += gate.mean(-1)
            gate_sq += gate.square().mean(-1)
            gate_spread += gate.std(-1)
            gate_count += 1
            if bool(loud.any()):
                loud_gate += torch.where(
                    is_var.view(n_cfg, 1, 1), second_moment_only, fraction
                )[:, loud].mean((1, 2))
                loud_count += 1
            if bool((~loud).any()):
                quiet_gate += torch.where(
                    is_var.view(n_cfg, 1, 1), second_moment_only, fraction
                )[:, ~loud].mean((1, 2))
                quiet_count += 1

        beta1, beta2 = 0.9, 0.999
        m1.mul_(beta1).add_(g1, alpha=1 - beta1)
        v1.mul_(beta2).addcmul_(g1, g1, value=1 - beta2)
        m2.mul_(beta1).add_(g2, alpha=1 - beta1)
        v2.mul_(beta2).addcmul_(g2, g2, value=1 - beta2)
        bias1, bias2 = 1 - beta1 ** (step + 1), 1 - beta2 ** (step + 1)
        # the gate scales the REALIZED step of the unit's incoming row only. It
        # must never touch the outgoing weight: w2 starts at zero, so a gate read
        # off that gradient would pin both at zero forever.
        w1 -= lr * ((m1 / bias1) / ((v1 / bias2).sqrt() + 1e-8)) * gate.unsqueeze(-1)
        w2 -= lr * (m2 / bias1) / ((v2 / bias2).sqrt() + 1e-8)

    with torch.no_grad():
        x = torch.randn((args.eval_samples, dim), device=device, generator=gen)
        clean = teacher_forward(x[:, :useful], teacher)
        act = torch.tanh(torch.einsum("khd,bd->kbh", w1, x))
        final = (act * w2.reshape(n_cfg, 1, hidden)).sum(-1)
        test = (final - clean).square().mean(1) / clean.square().mean().clamp_min(1e-12)

    denominator = max(gate_count, 1)
    return {"test": test.cpu().numpy(),
            "train": (squared / max(scored, 1) / (trivial / max(scored, 1))
                      ).cpu().numpy(),
            "gate": (gate_sum / denominator).cpu().numpy(),
            "gate_spread": (gate_spread / denominator).cpu().numpy(),
            "loud": (loud_gate / max(loud_count, 1)).cpu().numpy(),
            "quiet": (quiet_gate / max(quiet_count, 1)).cpu().numpy(),
            "useful_weight": w1[:, :, :useful].abs().mean((1, 2)).cpu().numpy(),
            "junk_weight": w1[:, :, useful:].abs().mean((1, 2)).cpu().numpy()}


def main():
    args = tyro.cli(Args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    methods = METHODS if args.method == "all" else tuple(args.method.split(","))
    for name in methods:
        if name not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
    configs = [{"method": m, "lr": v, "seed": s}
               for m in methods for v in args.lr_grid for s in range(args.seeds)]
    print(f"{args.input_dim} inputs ({args.useful} useful) -> {args.hidden} hidden, "
          f"noise {args.noise_std} x{args.noise_ratio} by regime, batch {args.batch}, "
          f"{args.samples} samples, {len(configs)} configs")
    start = time.perf_counter()
    out = run(args, configs, device)
    print(f"{len(configs)} configs in one pass, {time.perf_counter() - start:.1f}s\n")

    print(f"{'method':>19} {'lr':>7} | {'test/zero':>9} | {'gate':>6} {'sd':>5} "
          f"| {'loud':>6} {'quiet':>6} | {'|w| use':>7} {'|w| junk':>8}")
    for name in methods:
        best, row = None, None
        for value in args.lr_grid:
            picks = [i for i, c in enumerate(configs)
                     if c["method"] == name and c["lr"] == value]
            stats = {k: float(np.mean(out[k][picks])) for k in out}
            stats["ci"] = float(1.96 * np.std(out["test"][picks], ddof=1)
                                / np.sqrt(len(picks))) if len(picks) > 1 else float("nan")
            if best is None or stats["test"] < best:
                best, row = stats["test"], (value, stats)
        value, stats = row
        print(f"{name:>19} {value:>7} | {stats['test']:>9.4f} | {stats['gate']:>6.3f} "
              f"{stats['gate_spread']:>5.3f} | {stats['loud']:>6.3f} "
              f"{stats['quiet']:>6.3f} | {stats['useful_weight']:>7.4f} "
              f"{stats['junk_weight']:>8.5f}   +/-{stats['ci']:.4f}")


if __name__ == "__main__":
    main()

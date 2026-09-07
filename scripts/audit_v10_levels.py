"""Paired A/B/A measurement of the v10 batched ``PlasticityStepper.levels``.

Reuses ``scripts/audit_update_path.py``'s harness verbatim -- it already drives
the real ``Agent``/``ppo_loss``/``gate_supervision``/``PlasticityStepper``
against a stub env carrying HalfCheetah-v4 spaces -- with its module-level
trainer handle repointed at v10, so the classes under measurement are exactly
the ones the trainer uses (compiled loss and gate model included).

Two quantities, both interleaved A/B/A on a contended box:

* ``levels()`` per minibatch: v10's batched width-group publication against a
  local copy of v9's per-layer loop, on the SAME buffers.
* the whole update phase per minibatch (== ``timing/update_s`` / minibatches),
  by monkeypatching the stepper's ``levels`` between the two implementations.

Also reports the launch count of each variant and asserts the two produce
byte-identical ``lam``/``stats``, so a timing win can never be a silent
numerics change.
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.audit_update_path as audit

V10 = importlib.import_module("cleanrl.plasticity.ppo_continuous_action_sphere_sdplast_v10")
audit.V9 = V10  # the harness builds whatever this names; make it v10


def legacy_levels(stepper):
    """v9's ``levels`` body, verbatim, over the stepper's layers."""

    @torch.no_grad()
    def run():
        stepper.updates += 1
        if not stepper.enabled or stepper.updates <= stepper.args.snr_warmup:
            return
        span = float(np.log(stepper.args.lam_span))
        for layer in stepper.layers:
            level = stepper.args.snr_exponent * layer.snr.clamp_min(1e-30).log()
            level = level - level.mean()
            bounded = torch.tanh(stepper.args.lam_gain * level / span)
            bounded = bounded - bounded.mean()
            layer.lam.copy_(torch.exp(span * bounded))
            layer.stats[1] = layer.lam.mean()
            layer.stats[2] = layer.lam.std()

    return run


def fill_snr(stepper, seed=11):
    """Plant a wide, per-layer-distinct SNR spread in every layer's buffer."""
    with torch.no_grad():
        for index, layer in enumerate(stepper.layers):
            generator = torch.Generator(device=layer.snr.device).manual_seed(seed + index)
            draw = torch.rand(layer.snr.shape, device=layer.snr.device, generator=generator)
            layer.snr.copy_(((draw * 8.0 - 4.0) + 0.37 * index).exp())


def snapshot(stepper):
    return [(layer.lam.clone(), layer.stats.clone()) for layer in stepper.layers]


def identical(a, b):
    return all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a, b))


def launches(fn):
    """cudaLaunchKernel-class calls issued by one call of ``fn``."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch",
             "cudaMemcpyAsync", "cudaMemsetAsync")
    return sum(1 for e in prof.events() if e.name in names)


def aba(variants, sample, cycles):
    """Interleave every variant per cycle; return per-variant sample lists."""
    out = {name: [] for name in variants}
    for _ in range(cycles):
        for name, fn in variants.items():
            out[name].append(sample(fn))
    return out


def report(title, samples, scale, unit, per=1):
    audit.banner(title)
    print(f"{'variant':<28}{'median':>12}{'min':>12}{'samples':>9}")
    for name, values in samples.items():
        scaled = [v * scale / per for v in values]
        print(f"{name:<28}{statistics.median(scaled):>12.2f}{min(scaled):>12.2f}"
              f"{len(scaled):>9}  {unit}")
    names = list(samples)
    if len(names) == 2:
        pairs = [(a - b) * scale / per for a, b in zip(samples[names[0]], samples[names[1]])]
        print(f"paired {names[0]} - {names[1]}: median {statistics.median(pairs):.2f} {unit}"
              f"  min {min(pairs):.2f}  max {max(pairs):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--iters", type=int, default=6)
    cli = parser.parse_args()

    audit.configure_runtime()
    audit.report_load()
    # v10 compiles the gate model with ``compile_mode`` too; the audit harness
    # reads that from an audit-only ``gate_mode`` knob, so set it or the update
    # phase measured here is not the one v10 runs.
    args = audit.make_args({"num_envs": cli.num_envs, "num_steps": cli.num_steps})
    args.gate_mode = args.compile_mode
    harness = audit.Harness(args)
    stepper = harness.plasticity
    stepper.updates = args.snr_warmup + 1  # past warmup: measure the live path
    minibatches = args.update_epochs * args.num_minibatches
    print(f"plastic sites {len(stepper.layers)} | width groups "
          f"{[(len(g.layers), g.layers[0].lam.numel()) for g in stepper.groups]} | "
          f"batched {[g.batched for g in stepper.groups]}")

    fill_snr(stepper)
    legacy = legacy_levels(stepper)
    stepper.levels()
    batched_out = snapshot(stepper)
    legacy()
    legacy_out = snapshot(stepper)
    print(f"batched == per-layer loop, bytewise: {identical(batched_out, legacy_out)}")
    print(f"launches per call: batched {launches(stepper.levels)} | "
          f"per-layer loop {launches(legacy)}")

    samples = aba({"levels batched": stepper.levels, "levels per-layer loop": legacy},
                  lambda fn: audit.timed(fn, cli.iters, 50)[1], cli.cycles)
    report("levels() per call (min of 20-rep batches)", samples, 1e6, "us")
    audit.report_load()

    real = type(stepper).levels

    def span_of(patch, sink):
        type(stepper).levels = patch
        try:
            _, low, cpu = audit.cuda_event_span(harness.interval, cli.iters, 2)
            sink.append(cpu)
            return low
        finally:
            type(stepper).levels = real

    cpu_batched, cpu_loop = [], []
    spans = aba({"update batched": lambda: span_of(real, cpu_batched),
                 "update per-layer loop": lambda: span_of(lambda self: legacy(), cpu_loop)},
                lambda fn: fn(), cli.cycles)
    report("whole update phase (== timing/update_s) per minibatch", spans, 1e6, "us/mb",
           per=minibatches)
    report("whole update phase (== timing/update_s) per interval", spans, 1.0, "s")
    # CPU time in the region is the launch cost itself: immune to a neighbour's
    # GPU contention, which is what makes the wall-clock span so noisy here.
    report("cpu time inside the update phase, per minibatch",
           {"update batched": cpu_batched, "update per-layer loop": cpu_loop},
           1e6, "us/mb", per=minibatches)
    span = statistics.median(spans["update batched"])
    cpu = statistics.median(cpu_batched)
    print(f"cpu / span = {100 * cpu / span:.0f}% -> "
          f"{'CPU/launch-bound' if cpu > 0.8 * span else 'GPU-bound'}")
    audit.report_load()


if __name__ == "__main__":
    main()

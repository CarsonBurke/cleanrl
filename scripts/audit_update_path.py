"""Per-stage attribution for the v9 sdplast learner (update) phase.

Reconstructs the *exact* update phase of
``cleanrl/plasticity/ppo_continuous_action_sphere_sdplast_v9.py`` -- same net
(hypersphered SiTU-GLU trunks, width 64, 3 blocks, 20 plastic sites), same
batch (32768), same ``update_epochs``/``num_minibatches``, same fused Adam, the
same compiled loss (``reduce-overhead``) and gate supervision -- and attributes
GPU-timeline and CPU-launch time across:

    minibatch index/gather | loss forward | zero_grad+clear_probes | backward
    | gate forward | gate backward | clip_grad_norm x2 | levels | stash
    | optimizer.step | apply_levels | metric copy

Attribution uses a chain of ``cuda.Event``s: consecutive events measure the
GPU-timeline segment between them *including idle gaps*, so the segments sum to
the same quantity ``PhaseTimer.span("update")`` reports in production.
``perf_counter`` at the same boundaries gives the CPU launch cost, so a segment
with CPU >> GPU is launch-bound.

Also measures: kernel launches per minibatch (torch.profiler), whether inductor
cudagraphs actually engage, host-device syncs in the optimizer path (CUDA sync
debug counters), ``gather_metrics`` with the real 88-scalar payload, and
TensorBoard ``add_scalar`` at the real logging frequency.

Read-only with respect to the trainer: it imports v9 and drives its own copy.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

V9 = importlib.import_module("cleanrl.plasticity.ppo_continuous_action_sphere_sdplast_v9")

from cleanrl.shared.ppo_loop import device_minibatches, explained_variance, gather_metrics
from cleanrl.shared.runtime import configure_runtime


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


class StubEnvs:
    """Only the two spaces ``Agent.__init__`` reads."""

    def __init__(self, obs_dim, act_dim):
        import gymnasium as gym

        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float32)


class Harness:
    """One faithful v9 learner, driven a minibatch at a time."""

    def __init__(self, args, obs_dim=17, act_dim=6, device="cuda"):
        self.args = args
        self.device = torch.device(device)
        self.agent = V9.Agent(StubEnvs(obs_dim, act_dim), args).to(self.device)
        self.net_params = self.agent.network_parameters()
        self.gate_params = self.agent.gate_parameters()
        self.optimizer = torch.optim.Adam(
            [{"params": self.net_params, "lr": args.learning_rate},
             {"params": self.gate_params, "lr": args.gate_lr}],
            lr=args.learning_rate, eps=1e-5, fused=True)
        self.plasticity = V9.PlasticityStepper(self.agent.plastic_sites, args)
        self.gate_scale = float(args.gate_every)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return V9.ppo_loss(self.agent, observations, native, old_logprobs,
                               advantages, returns, old_values, args)

        def gate_model(observations):
            return V9.gate_supervision(self.agent, observations, args)

        self.loss_model, self.gate_model = loss_model, gate_model
        # ``gate_mode`` is an audit-only knob: v9 compiles the gate model with
        # inductor's default mode (no cudagraphs). None reproduces v9 exactly.
        gate_mode = getattr(args, "gate_mode", None)
        if args.compile:
            self.loss_model = torch.compile(loss_model, mode=args.compile_mode,
                                            fullgraph=True, dynamic=False)
            self.gate_model = torch.compile(gate_model, dynamic=False, mode=gate_mode)

        n, dev = args.batch_size, self.device
        gen = torch.Generator(device=dev).manual_seed(7)
        self.b_obs = torch.randn(n, obs_dim, device=dev, generator=gen)
        self.b_native = torch.rand(n, act_dim, device=dev, generator=gen).clamp(1e-6, 1 - 1e-6)
        self.b_logprobs = torch.randn(n, device=dev, generator=gen)
        self.b_advantages = torch.randn(n, device=dev, generator=gen)
        self.b_returns = torch.randn(n, device=dev, generator=gen)
        self.b_values = torch.randn(n, device=dev, generator=gen)
        self.shuffle_generator = torch.Generator(device=dev).manual_seed(args.seed)
        self.max_updates = args.update_epochs * (
            (args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        self.update_metrics = torch.empty((self.max_updates, 6), device=dev)
        self.gate_losses = torch.zeros(self.max_updates, device=dev)
        self.updates = 0
        self.skip_tail = False
        # When a list, every minibatch appends a float64 fingerprint of the 20
        # probe.grad tensors, taken between loss.backward() and the gate model:
        # exactly the window in which a cudagraph-pool overwrite would be
        # observable. Full clones are 168 MiB per minibatch, hence fingerprints.
        self.probe_trace = None

    # ---- the production update body, split at the boundaries we attribute ----

    def minibatch(self, indices, updates, mark, stages=None):
        """One faithful loop body. ``stages`` collects (name, cuda_event, t_cpu)."""
        args = self.args
        rec = _Recorder(stages)
        if mark:
            torch.compiler.cudagraph_mark_step_begin()
        rec.mark("mark_step")
        mb = (self.b_obs[indices], self.b_native[indices], self.b_logprobs[indices],
              self.b_advantages[indices], self.b_returns[indices], self.b_values[indices])
        rec.mark("index_gather")
        loss, metrics = self.loss_model(*mb)
        rec.mark("loss_forward")
        supervise = updates % args.gate_every == 0
        self.optimizer.zero_grad(set_to_none=True)
        self.plasticity.clear_probes()
        rec.mark("zero_grad")
        loss.backward()
        rec.mark("backward")
        if self.probe_trace is not None:
            self.probe_trace.append(self.probe_fingerprint())
        if supervise:
            gate_loss = self.gate_model(mb[0]) * self.gate_scale
            rec.mark("gate_forward")
            gate_loss.backward()
            self.gate_losses[updates].copy_(gate_loss.detach())
            rec.mark("gate_backward")
        else:
            self.gate_losses[updates].copy_(self.gate_losses[max(updates - 1, 0)])
            rec.mark("gate_carry")
        if not self.skip_tail:
            nn.utils.clip_grad_norm_(self.net_params, args.max_grad_norm)
            rec.mark("clip_net")
            if self.gate_params:
                nn.utils.clip_grad_norm_(self.gate_params, args.gate_clip)
            rec.mark("clip_gate")
            self.plasticity.levels()
            rec.mark("levels")
            self.plasticity.stash()
            rec.mark("stash")
            self.optimizer.step()
            rec.mark("opt_step")
            self.plasticity.apply_levels()
            rec.mark("apply_levels")
        self.update_metrics[updates].copy_(metrics)
        rec.mark("metric_copy")

    def probe_fingerprint(self):
        """Order-sensitive float64 signature of all 20 ``probe.grad`` tensors.

        Four independent reductions per site, in float64 so the reduction
        itself cannot lose a differing element to rounding: sum (catches any
        signed change that does not cancel), abs-sum and square-sum (catch
        sign-symmetric changes that cancel in the plain sum), and a strided
        sample weighted by position (catches a permutation or a shifted buffer,
        which every symmetric reduction is blind to).
        """
        parts = []
        for _, module in self.agent.plastic_sites:
            grad = module.probe.grad
            if grad is None:
                parts.append(torch.full((4,), float("nan"), dtype=torch.float64,
                                        device=self.device))
                continue
            flat = grad.detach().reshape(-1).double()
            strided = flat[::4097]
            weights = torch.arange(1, strided.numel() + 1, dtype=torch.float64,
                                   device=flat.device)
            parts.append(torch.stack((flat.sum(), flat.abs().sum(),
                                      flat.square().sum(), (strided * weights).sum())))
        return torch.stack(parts).reshape(-1)

    def interval(self, stages=None):
        """One production update phase: ``update_epochs`` x ``num_minibatches``."""
        args = self.args
        updates = 0
        for _ in range(args.update_epochs):
            for indices in device_minibatches(args.batch_size, args.minibatch_size,
                                              self.device, self.shuffle_generator):
                self.minibatch(indices, updates, args.compile, stages)
                updates += 1
        self.updates = updates
        return updates

    def metric_payload(self):
        last = self.update_metrics[self.updates - 1]
        return {
            "losses/policy_loss": last[0], "losses/value_loss": last[1],
            "losses/entropy": last[2], "losses/old_approx_kl": last[3],
            "losses/approx_kl": last[4],
            "losses/clipfrac": self.update_metrics[:self.updates, 5].mean(),
            "losses/explained_variance": explained_variance(self.b_values, self.b_returns),
            "losses/gate_nll": self.gate_losses[:self.updates].mean() / self.gate_scale,
            **{f"sdp/{name.replace('.', '_')}_{stat}": module.stats[index]
               for name, module in self.agent.plastic_sites
               for index, stat in enumerate(("w_std", "lam_mean", "lam_std", "snr"))},
        }


class _Recorder:
    """Event/CPU-clock chain. ``None`` sink => zero instrumentation cost."""

    __slots__ = ("sink",)

    def __init__(self, sink):
        self.sink = sink

    def mark(self, name):
        if self.sink is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.sink.append((name, event, time.perf_counter()))


def resolve(stages):
    """(name, gpu_us, cpu_us) per segment between consecutive marks."""
    torch.cuda.synchronize()
    out = []
    for (_, e0, t0), (name, e1, t1) in zip(stages, stages[1:]):
        out.append((name, e0.elapsed_time(e1) * 1e3, (t1 - t0) * 1e6))
    return out


def aggregate(rows):
    totals = {}
    for name, gpu, cpu in rows:
        g, c, n = totals.get(name, (0.0, 0.0, 0))
        totals[name] = (g + gpu, c + cpu, n + 1)
    return totals


def timed(fn, iters, warmup, reps=20):
    """Median/min per-call wall, plus CPU time per call.

    ``reps`` inner repetitions per sample amortize the trailing
    ``synchronize``, which under GPU contention from foreign processes costs
    milliseconds on its own and otherwise swamps sub-100us stages. Wall per
    call therefore includes launch cost plus the queue's steady-state drain
    rate, which is the quantity that matters inside the update loop.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples, cpu0 = [], cpu_seconds()
    for _ in range(iters):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) / reps)
    cpu = (cpu_seconds() - cpu0) / (iters * reps)
    return statistics.median(samples), min(samples), cpu


def cuda_event_span(fn, iters, warmup):
    """Exactly what ``PhaseTimer.span('update')`` measures in production."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    spans, cpu0 = [], cpu_seconds()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        spans.append(start.elapsed_time(end) / 1e3)
    cpu = (cpu_seconds() - cpu0) / iters
    return statistics.median(spans), min(spans), cpu


def banner(title):
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def make_args(overrides):
    args = V9.Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


def report_load():
    with open("/proc/loadavg") as handle:
        load = handle.read().split()[:3]
    free, total = (v / 2**20 for v in torch.cuda.mem_get_info())
    print(f"loadavg {' '.join(load)} | gpu free {free:.0f}/{total:.0f} MiB")


# --------------------------------------------------------------------------- #
#  measurements                                                               #
# --------------------------------------------------------------------------- #

def measure_stages(harness, iters, warmup):
    banner("per-stage attribution (GPU timeline / CPU launch), per minibatch")
    for _ in range(warmup):
        harness.interval()
    torch.cuda.synchronize()
    rows = []
    for _ in range(iters):
        stages = []
        harness.interval(stages)
        rows.extend(resolve(stages))
    totals = aggregate(rows)
    mbs = harness.args.update_epochs * harness.args.num_minibatches * iters
    order = ["mark_step", "index_gather", "loss_forward", "zero_grad", "backward",
             "gate_forward", "gate_backward", "gate_carry", "clip_net", "clip_gate",
             "levels", "stash", "opt_step", "apply_levels", "metric_copy"]
    gpu_sum = sum(v[0] for v in totals.values())
    print(f"{'stage':<16}{'gpu us/mb':>11}{'cpu us/mb':>11}{'gpu %':>8}{'calls':>8}")
    for name in order:
        if name not in totals:
            continue
        gpu, cpu, n = totals[name]
        print(f"{name:<16}{gpu / mbs:>11.1f}{cpu / mbs:>11.1f}"
              f"{100 * gpu / gpu_sum:>8.1f}{n:>8}")
    print(f"{'TOTAL':<16}{gpu_sum / mbs:>11.1f}"
          f"{sum(v[1] for v in totals.values()) / mbs:>11.1f}")
    print(f"per-interval gpu total: {gpu_sum / iters / 1e6:.4f} s "
          f"({harness.args.update_epochs * harness.args.num_minibatches} minibatches)")
    return totals, mbs


def measure_whole(harness, iters, warmup):
    banner("whole update phase (uninstrumented)")
    med, lo, cpu = cuda_event_span(harness.interval, iters, warmup)
    print(f"cuda-event span (== timing/update_s): median {med:.4f} s  min {lo:.4f} s")
    print(f"cpu time in the region:               {cpu:.4f} s "
          f"({100 * cpu / med:.0f}% of the span -> "
          f"{'CPU/launch-bound' if cpu > 0.8 * med else 'GPU-bound'})")
    return med, cpu


def measure_launches(harness):
    banner("kernel launches per minibatch + cudagraph engagement")
    from torch.profiler import ProfilerActivity, profile

    harness.interval()
    harness.interval()
    torch.cuda.synchronize()
    stages = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        harness.minibatch(next(iter(device_minibatches(
            harness.args.batch_size, harness.args.minibatch_size,
            harness.device, harness.shuffle_generator))), 0, harness.args.compile, stages)
        torch.cuda.synchronize()
    events = prof.events()
    kernels = [e for e in events if str(getattr(e, "device_type", "")).endswith("CUDA")
               and getattr(e, "device_index", -1) >= 0]
    launch = [e for e in events if e.name in ("cudaLaunchKernel", "cudaGraphLaunch",
                                              "cudaMemcpyAsync", "cudaMemsetAsync",
                                              "cudaStreamSynchronize",
                                              "cudaDeviceSynchronize",
                                              "cudaLaunchKernelExC")]
    counts = {}
    for e in launch:
        counts[e.name] = counts.get(e.name, 0) + 1
    ka = prof.key_averages()
    # ``kernels`` are leaf device events only, so summing their durations avoids
    # the double counting you get from key_averages (whose CPU-side annotation
    # rows, e.g. ``Optimizer.step#Adam.step``, also carry device time).
    device_total = sum(e.self_device_time_total for e in kernels)
    print(f"device-side kernel instances (profiler): {len(kernels)}")
    print(f"summed leaf device time: {device_total:.0f} us")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24}{n:>6} calls")
    gemm_time = sum(e.self_device_time_total for e in kernels
                    if any(k in e.name.lower() for k in ("gemm", "cutlass", "cublas",
                                                         "tensorop", "splitk")))
    print(f"GEMM/cuBLAS device time: {gemm_time:.0f} us "
          f"({100 * gemm_time / max(device_total, 1):.1f}% of leaf device time)")
    by_name = {}
    for e in kernels:
        acc = by_name.setdefault(e.name, [0.0, 0])
        acc[0] += e.self_device_time_total
        acc[1] += 1
    print("\n top device kernels (one minibatch):  us   count  name")
    for name, (total, n) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:16]:
        print(f"  {total:>9.1f} us  x{n:<5} {name[:76]}")
    tiny = [e for e in kernels if e.self_device_time_total < 6.0]
    print(f"\n kernels shorter than 6us: {len(tiny)} of {len(kernels)} "
          f"({sum(e.self_device_time_total for e in tiny):.0f} us total device time)")
    from torch._dynamo.utils import counters
    inductor = dict(counters.get("inductor", {}))
    print(f"\n inductor counters: {inductor}")
    try:
        import torch._inductor.cudagraph_trees as cgt
        container = cgt.get_container(torch.cuda.current_device())
        tree = container.tree_manager if container else None
        print(f" cudagraph_trees manager: {tree}")
        if tree is not None:
            print(f"   recorded graph nodes: {len(getattr(tree, 'roots', {}) or {})} roots, "
                  f"current={getattr(tree, 'current_node', None)}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f" cudagraph_trees introspection failed: {exc}")
    return counts, device_total


def measure_launch_cost(device, iters=30):
    """Per-launch CPU cost and the width-64 GEMM it is competing against."""
    banner("launch overhead vs width-64 GEMM work")
    scalar = torch.zeros(1, device=device)
    med, lo, cpu = timed(lambda: scalar.add_(1.0), iters, 200, reps=200)
    print(f"1-element add_ (pure launch): median {med * 1e6:>7.2f} us  "
          f"min {lo * 1e6:>7.2f} us  cpu {cpu * 1e6:>7.2f} us")
    a = torch.randn(32768, 64, device=device)
    w = torch.randn(64, 64, device=device)
    out = torch.empty(32768, 64, device=device)
    med, lo, cpu = timed(lambda: torch.mm(a, w, out=out), iters, 100, reps=100)
    print(f"mm (32768x64 @ 64x64):       median {med * 1e6:>7.2f} us  "
          f"min {lo * 1e6:>7.2f} us  cpu {cpu * 1e6:>7.2f} us")
    big = torch.randn(32768, 64, device=device)
    med, lo, cpu = timed(lambda: big.tanh_(), iters, 100, reps=100)
    print(f"tanh_ (32768x64 elementwise): median {med * 1e6:>7.2f} us  "
          f"min {lo * 1e6:>7.2f} us  cpu {cpu * 1e6:>7.2f} us")
    small = torch.randn(64, device=device)
    med, lo, cpu = timed(lambda: small.tanh_(), iters, 200, reps=200)
    print(f"tanh_ (64 elements):         median {med * 1e6:>7.2f} us  "
          f"min {lo * 1e6:>7.2f} us  cpu {cpu * 1e6:>7.2f} us")
    dest = torch.zeros(4, device=device)
    src = torch.zeros(64, device=device)
    med, lo, cpu = timed(lambda: dest.__setitem__(1, src.mean()), iters, 200, reps=200)
    print(f"stats[i] = t.mean() (setitem): median {med * 1e6:>7.2f} us  "
          f"min {lo * 1e6:>7.2f} us  cpu {cpu * 1e6:>7.2f} us")


def measure_syncs(harness):
    banner("host-device syncs in the update path")
    hits = []
    real = torch.cuda.synchronize

    class Detector:
        def __enter__(self):
            self.prev = getattr(torch.cuda, "_sync_debug_mode", None)
            torch.cuda.set_sync_debug_mode("warn")
            import warnings
            self.ctx = warnings.catch_warnings(record=True)
            self.log = self.ctx.__enter__()
            warnings.simplefilter("always")
            return self

        def __exit__(self, *exc):
            for w in self.log:
                hits.append(str(w.message)[:140])
            self.ctx.__exit__(*exc)
            torch.cuda.set_sync_debug_mode("default")
            return False

    harness.interval()
    torch.cuda.synchronize()
    with Detector():
        harness.interval()
    torch.cuda.synchronize()
    if hits:
        seen = {}
        for h in hits:
            seen[h] = seen.get(h, 0) + 1
        for h, n in seen.items():
            print(f"  x{n:<4} {h}")
    else:
        print("  no synchronizing call warned by torch.cuda sync debug mode "
              "across a full update phase")
    print(f"  torch.cuda.synchronize identity unchanged: {real is torch.cuda.synchronize}")
    return hits


def measure_logging(harness, iters):
    banner("end-of-interval metric gather + TensorBoard logging")
    harness.interval()
    torch.cuda.synchronize()
    payload = harness.metric_payload()
    print(f"payload scalars: {len(payload)} "
          f"({sum(1 for k in payload if k.startswith('sdp/'))} sdp/*)")
    med, lo, cpu = timed(lambda: gather_metrics(harness.metric_payload()), iters, 20)
    print(f"metric_payload+gather_metrics: median {med * 1e6:.0f} us  min {lo * 1e6:.0f} us "
          f"cpu {cpu * 1e6:.0f} us")
    med2, lo2, _ = timed(lambda: gather_metrics({"a": harness.update_metrics[0, 0]}), iters, 20)
    print(f"gather_metrics(1 scalar):      median {med2 * 1e6:.0f} us  min {lo2 * 1e6:.0f} us")
    flat = [v.detach().reshape(()) for v in payload.values()]
    med3, lo3, _ = timed(lambda: torch.stack(flat).cpu(), iters, 20)
    print(f"  fp32 stack+cpu (no float64 casts): median {med3 * 1e6:.0f} us min {lo3 * 1e6:.0f} us")
    stats = [m.stats for _, m in harness.agent.plastic_sites]
    med4, lo4, _ = timed(lambda: torch.cat(stats).cpu(), iters, 20)
    print(f"  torch.cat(20 stats buffers)+cpu:   median {med4 * 1e6:.0f} us min {lo4 * 1e6:.0f} us")

    logged = gather_metrics(payload)
    med5, lo5, cpu5 = timed(
        lambda: [np.isfinite(v) for v in logged.values()], iters, 20)
    print(f"finite check ({len(logged)} py floats):  median {med5 * 1e6:.0f} us")

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"tensorboard unavailable: {exc}")
        return
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="audit_tb_")
    writer = SummaryWriter(tmp)
    step = [0]

    def log_all():
        step[0] += 32768
        for name, value in logged.items():
            writer.add_scalar(name, value, step[0])
        writer.add_scalar("charts/learning_rate", 1e-3, step[0])
        writer.add_scalar("charts/SPS", 39617, step[0])
        writer.add_scalar("charts/interval_SPS", 39617.0, step[0])
        for phase in ("rollout", "env", "normalize_transfer", "gae", "update"):
            writer.add_scalar(f"timing/{phase}_s", 0.1, step[0])

    med6, lo6, cpu6 = timed(log_all, 20, 5, reps=5)
    n_scalars = len(logged) + 8
    print(f"add_scalar x{n_scalars}:            median {med6 * 1e6:.0f} us  "
          f"min {lo6 * 1e6:.0f} us  ({med6 * 1e6 / n_scalars:.1f} us/scalar)")
    writer.close()
    shutil.rmtree(tmp, ignore_errors=True)


def measure_alternatives(harness_factory, base_overrides, iters, warmup):
    banner("alternatives (identical math unless noted)")
    rows = []
    for label, overrides, note in [
        ("baseline reduce-overhead", {}, "production"),
        ("compile default mode", {"compile_mode": "default"}, "identical math, no cudagraphs"),
        ("compile=False", {"compile": False}, "identical math, eager"),
        ("max-autotune-no-cudagraphs", {"compile_mode": "max-autotune-no-cudagraphs"},
         "identical math"),
    ]:
        merged = dict(base_overrides)
        merged.update(overrides)
        args = make_args(merged)
        try:
            harness = harness_factory(args)
            med, lo, cpu = cuda_event_span(harness.interval, iters, warmup)
            rows.append((label, med, lo, cpu, note))
            del harness
        except Exception as exc:
            rows.append((label, float("nan"), float("nan"), float("nan"), f"FAILED: {exc}"))
        torch.cuda.empty_cache()
    print(f"{'variant':<28}{'span s':>10}{'min s':>10}{'cpu s':>10}  note")
    for label, med, lo, cpu, note in rows:
        print(f"{label:<28}{med:>10.4f}{lo:>10.4f}{cpu:>10.4f}  {note}")
    return rows


def measure_fused_plasticity(harness, iters):
    """Foreach-fused ``levels``/``apply_levels`` against the per-layer loops."""
    banner("fused plasticity bookkeeping (identical math, fewer launches)")
    p = harness.plasticity
    p.updates = harness.args.snr_warmup + 10  # past warmup: measure the live path
    med, lo, cpu = timed(p.levels, iters, 50)
    print(f"levels()      per-layer loop: median {med * 1e6:>8.1f} us  cpu {cpu * 1e6:>8.1f} us")
    med, lo, cpu = timed(p.stash, iters, 50)
    print(f"stash()       foreach_copy_ : median {med * 1e6:>8.1f} us  cpu {cpu * 1e6:>8.1f} us")
    med, lo, cpu = timed(p.apply_levels, iters, 50)
    print(f"apply_levels() per-param loop: median {med * 1e6:>8.1f} us  cpu {cpu * 1e6:>8.1f} us")

    args = harness.args
    layers = p.layers
    snrs = [l.snr for l in layers]
    lams = [l.lam for l in layers]
    span = float(np.log(args.lam_span))

    @torch.no_grad()
    def levels_foreach():
        level = torch._foreach_log(torch._foreach_clamp_min(snrs, 1e-30))
        if args.snr_exponent != 1.0:
            level = torch._foreach_mul(level, args.snr_exponent)
        means = [t.mean() for t in level]
        torch._foreach_sub_(level, means)
        torch._foreach_mul_(level, args.lam_gain / span)
        bounded = torch._foreach_tanh(level)
        torch._foreach_sub_(bounded, [t.mean() for t in bounded])
        torch._foreach_mul_(bounded, span)
        torch._foreach_copy_(lams, torch._foreach_exp(bounded))

    med, lo, cpu = timed(levels_foreach, iters, 50)
    print(f"levels()      foreach       : median {med * 1e6:>8.1f} us  cpu {cpu * 1e6:>8.1f} us"
          "   (stats writes omitted)")

    params = p.params
    snapshots = p.snapshots
    gains = [(l.lam - 1.0).unsqueeze(1) if is_m else (l.lam - 1.0)
             for l, _, is_m in p.plan]

    @torch.no_grad()
    def apply_foreach():
        torch._foreach_sub_(snapshots, params)
        torch._foreach_mul_(snapshots, gains)
        torch._foreach_sub_(params, snapshots)

    med, lo, cpu = timed(apply_foreach, iters, 50)
    print(f"apply_levels() foreach      : median {med * 1e6:>8.1f} us  cpu {cpu * 1e6:>8.1f} us")

    med, lo, cpu = timed(lambda: nn.utils.clip_grad_norm_(harness.net_params, 0.5), iters, 50)
    print(f"clip_grad_norm_(net,  {len(harness.net_params):>3} t): median {med * 1e6:>8.1f} us"
          f"  cpu {cpu * 1e6:>8.1f} us")
    med, lo, cpu = timed(lambda: nn.utils.clip_grad_norm_(harness.gate_params, 1.0), iters, 50)
    print(f"clip_grad_norm_(gate, {len(harness.gate_params):>3} t): median {med * 1e6:>8.1f} us"
          f"  cpu {cpu * 1e6:>8.1f} us")
    med, lo, cpu = timed(harness.optimizer.step, iters, 50)
    print(f"optimizer.step() fused Adam  : median {med * 1e6:>8.1f} us  cpu {cpu * 1e6:>8.1f} us")
    med, lo, cpu = timed(lambda: harness.optimizer.zero_grad(set_to_none=True), iters, 50)
    print(f"zero_grad(set_to_none=True)  : median {med * 1e6:>8.1f} us  cpu {cpu * 1e6:>8.1f} us")
    med, lo, cpu = timed(harness.plasticity.clear_probes, iters, 50)
    print(f"clear_probes()               : median {med * 1e6:>8.1f} us  cpu {cpu * 1e6:>8.1f} us")


def measure_duplicate_state_logits(harness, iters):
    """``state_logits`` is evaluated twice per site per minibatch.

    ``PlasticLinear.forward`` evaluates it under ``no_grad`` to build ``w``;
    ``gate_supervision`` evaluates it again (through ``predict_log_noise``) to
    build the NLL, with the SAME ``x`` and ``z``. This measures the size of
    that duplicate, plus the per-minibatch ``w.std()`` reductions that only the
    last minibatch's value is ever read from.
    """
    banner("redundant work inside the plastic sites")
    harness.interval()
    torch.cuda.synchronize()
    args = harness.args
    indices = next(iter(device_minibatches(args.batch_size, args.minibatch_size,
                                           harness.device, harness.shuffle_generator)))
    obs = harness.b_obs[indices]
    with torch.no_grad():
        pairs = harness.agent.site_activations(obs)
    sites = [module for _, module in harness.agent.plastic_sites]

    @torch.no_grad()
    def all_state_logits():
        for module, (x, z) in zip(sites, pairs):
            module.state_logits(x, z)

    @torch.no_grad()
    def all_w():
        for module, (x, z) in zip(sites, pairs):
            p_state = module.state_logits(x, z)
            w = torch.exp(module.p_ref - p_state).clamp(module.w_min, module.w_max)
            module.stats[0] = w.std()

    @torch.no_grad()
    def all_w_std_only():
        for module, (x, z) in zip(sites, pairs):
            module.stats[0] = z.std()

    fns = {"state_logits x20 (eager)": all_state_logits,
           "state_logits+w+w.std() x20 (eager)": all_w,
           "w.std() reduction x20 only": all_w_std_only}
    if args.compile:
        fns["state_logits x20 (compiled)"] = torch.compile(all_state_logits, dynamic=False)
        fns["state_logits+w+std x20 (compiled)"] = torch.compile(all_w, dynamic=False)
    for label, fn in fns.items():
        fn()
        torch.cuda.synchronize()
        med, lo, cpu = timed(fn, iters, 10, reps=10)
        print(f"{label:<38} median {med * 1e6:>8.1f} us  min {lo * 1e6:>8.1f} us  "
              f"cpu {cpu * 1e6:>8.1f} us")
    print(f"  -> per interval ({args.update_epochs * args.num_minibatches} minibatches), "
          "a duplicate state_logits pass costs 10x the compiled figure above")


def measure_gpu_work(harness, iters):
    """Split the two dominant stages into their real sub-costs."""
    banner("where the GPU work actually is")
    args = harness.args
    harness.interval()
    torch.cuda.synchronize()
    indices = next(iter(device_minibatches(args.batch_size, args.minibatch_size,
                                           harness.device, harness.shuffle_generator)))
    obs = harness.b_obs[indices]
    agent = harness.agent

    site_fn = agent.site_activations
    if args.compile:
        site_fn = torch.compile(agent.site_activations, dynamic=False)
    med, lo, cpu = timed(lambda: site_fn(obs), iters, 20, reps=10)
    print(f"site_activations (trunk re-walk, no_grad): median {med * 1e6:>8.1f} us "
          f"min {lo * 1e6:>8.1f} us")

    def loss_fwd():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        return harness.loss_model(obs, harness.b_native[indices],
                                  harness.b_logprobs[indices],
                                  harness.b_advantages[indices],
                                  harness.b_returns[indices], harness.b_values[indices])

    med, lo, cpu = timed(loss_fwd, iters, 20, reps=10)
    print(f"loss_model forward only:                   median {med * 1e6:>8.1f} us "
          f"min {lo * 1e6:>8.1f} us")
    med, lo, cpu = timed(lambda: harness.gate_model(obs), iters, 20, reps=10)
    print(f"gate_model forward only:                   median {med * 1e6:>8.1f} us "
          f"min {lo * 1e6:>8.1f} us")
    with torch.no_grad():
        med, lo, cpu = timed(lambda: agent.get_policy_and_value(obs), iters, 20, reps=10)
    print(f"eager no_grad policy+value fwd (ref):      median {med * 1e6:>8.1f} us "
          f"min {lo * 1e6:>8.1f} us")
    probe_bytes = sum(m.probe.numel() * 4 for _, m in agent.plastic_sites)
    print(f"probe parameters: {probe_bytes / 2**20:.0f} MiB across "
          f"{len(agent.plastic_sites)} sites (probe.grad materializes the same again)")
    print(f"peak allocated: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


def measure_linear_cse(harness_factory, base_overrides, iters, warmup):
    """``PlasticLinear.forward`` computes the SAME GEMM twice; is dropping it free?

    ``z = F.linear(xd, wd, bd)`` and ``y = F.linear(x, wd, bd)`` are numerically
    identical (``xd`` is ``x.detach()``, ``wd``/``bd`` are detached weights), so
    ``z = y.detach()`` is bit-identical and removes one 32768x64x64 GEMM per
    plastic site per forward. This measures whether inductor already CSEs it.
    """
    banner("PlasticLinear duplicate GEMM (z == y): does removing it help?")
    original = V9.PlasticLinear.forward

    def patched(self, x):
        if not torch.is_grad_enabled():
            return torch.nn.functional.linear(x, self.weight, self.bias)
        xd = x.detach()
        wd = self.weight.detach()
        bd = None if self.bias is None else self.bias.detach()
        y = torch.nn.functional.linear(x, wd, bd)
        z = y.detach()
        if self.use_weights:
            with torch.no_grad():
                p_state = self.state_logits(xd, z)
                w = torch.exp(self.p_ref - p_state).clamp(self.w_min, self.w_max)
        else:
            w = torch.ones_like(z)
        self.stats[0] = w.std()
        y = y + w * (torch.nn.functional.linear(xd, self.weight, self.bias) - z)
        return y + self.probe

    rows = []
    for label, fn in [("v9 (three linears)", original), ("z = y.detach()", patched)]:
        V9.PlasticLinear.forward = fn
        args = make_args(dict(base_overrides))
        harness = harness_factory(args)
        med, lo, cpu = cuda_event_span(harness.interval, iters, warmup)
        rows.append((label, med, lo, cpu))
        del harness
        torch.cuda.empty_cache()
    V9.PlasticLinear.forward = original
    print(f"{'variant':<24}{'span s':>10}{'min s':>10}{'cpu s':>10}")
    for label, med, lo, cpu in rows:
        print(f"{label:<24}{med:>10.4f}{lo:>10.4f}{cpu:>10.4f}")

    # bitwise equivalence of loss and every gradient, uncompiled to isolate math
    args = make_args(dict(base_overrides, compile=False))
    results = []
    for fn in (original, patched):
        V9.PlasticLinear.forward = fn
        torch.manual_seed(0)
        h = harness_factory(args)
        idx = torch.arange(args.minibatch_size, device=h.device)
        loss, metrics = h.loss_model(h.b_obs[idx], h.b_native[idx], h.b_logprobs[idx],
                                     h.b_advantages[idx], h.b_returns[idx], h.b_values[idx])
        h.optimizer.zero_grad(set_to_none=True)
        h.plasticity.clear_probes()
        loss.backward()
        grads = torch.cat([p.grad.reshape(-1) for p in h.net_params])
        probes = torch.cat([m.probe.grad.reshape(-1) for _, m in h.agent.plastic_sites])
        results.append((loss.detach().clone(), metrics.clone(), grads.clone(), probes.clone()))
        del h
        torch.cuda.empty_cache()
    V9.PlasticLinear.forward = original
    a, b = results
    print(f"bitwise identical: loss={torch.equal(a[0], b[0])} "
          f"metrics={torch.equal(a[1], b[1])} net_grads={torch.equal(a[2], b[2])} "
          f"probe_grads={torch.equal(a[3], b[3])}")
    print(f"max |dgrad|: {(a[2] - b[2]).abs().max().item():.3e}")


def measure_tail_graph(harness, iters, warmup):
    """How much of the eager optimizer tail is pure launch overhead.

    The tail (clip x2, levels, stash, optimizer.step, apply_levels) contains no
    host-device sync and no host-visible control flow, so it is CUDA-graph
    capturable as-is. Replaying it costs ONE launch, which isolates its true
    GPU work from the ~300 eager launches that currently issue it. Same math.
    """
    banner("eager optimizer tail vs a single CUDA-graph replay")
    args = harness.args
    p = harness.plasticity
    p.updates = args.snr_warmup + 10

    def tail():
        nn.utils.clip_grad_norm_(harness.net_params, args.max_grad_norm)
        nn.utils.clip_grad_norm_(harness.gate_params, args.gate_clip)
        p.levels()
        p.stash()
        harness.optimizer.step()
        p.apply_levels()

    # Grads must exist and be stable-addressed for capture: run one real
    # minibatch, then stop using set_to_none so the .grad buffers persist.
    harness.interval()
    torch.cuda.synchronize()
    med, lo, cpu = timed(tail, iters, 20, reps=20)
    print(f"eager tail:            median {med * 1e6:>8.1f} us  min {lo * 1e6:>8.1f} us "
          f"cpu {cpu * 1e6:>8.1f} us")
    graph = torch.cuda.CUDAGraph()
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                tail()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            tail()
    except Exception as exc:
        print(f"capture FAILED: {type(exc).__name__}: {exc}")
        return
    med2, lo2, cpu2 = timed(graph.replay, iters, 20, reps=20)
    print(f"graph replay (1 launch): median {med2 * 1e6:>8.1f} us  min {lo2 * 1e6:>8.1f} us "
          f"cpu {cpu2 * 1e6:>8.1f} us")
    print(f"=> launch overhead in the tail: {(med - med2) * 1e6:.0f} us/minibatch, "
          f"{(med - med2) * 1e6 * args.update_epochs * args.num_minibatches / 1e3:.1f} ms/interval")


def measure_skip_tail(harness, iters, warmup):
    """Upper bound on any tail optimization: delete the tail entirely."""
    banner("upper bound: update phase with the eager tail deleted (DIAGNOSTIC, wrong math)")
    harness.skip_tail = False
    full, full_lo, full_cpu = cuda_event_span(harness.interval, iters, warmup)
    harness.skip_tail = True
    cut, cut_lo, cut_cpu = cuda_event_span(harness.interval, iters, warmup)
    harness.skip_tail = False
    print(f"with tail:    span {full:.4f} s (min {full_lo:.4f})  cpu {full_cpu:.4f} s")
    print(f"without tail: span {cut:.4f} s (min {cut_lo:.4f})  cpu {cut_cpu:.4f} s")
    print(f"=> tail costs {full - cut:.4f} s/interval "
          f"({100 * (full - cut) / full:.1f}% of the update phase)")


def measure_gate_mode_equivalence(harness_factory, base_overrides, intervals=5):
    """Is compiling the gate model with cudagraphs bit-identical to v9?

    The hazard is real: ``probe.grad`` is produced inside the loss model's
    cudagraph pool and consumed by the gate model, so wrapping the gate model
    in a second cudagraph-managed region could read a buffer a later replay has
    overwritten. This compares the hazard's OWN channel directly -- a per-
    minibatch fingerprint of all 20 ``probe.grad`` tensors, taken in the window
    between ``loss.backward()`` and the gate model, plus a full exact compare
    of the final minibatch's raw ``probe.grad`` -- and then the downstream
    learner state that a corrupted read would perturb.
    """
    banner("gate reduce-overhead: bitwise equivalence against v9")
    states = []
    # Third arm: v9 mode again but UNTRACED, so we can prove the fingerprint
    # instrument (eager reads of a cudagraph-pool tensor between two graph
    # regions) does not itself perturb the learner it measures.
    for mode, trace in ((None, True), ("reduce-overhead", True), (None, False)):
        args = make_args(dict(base_overrides))
        args.gate_mode = mode
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        harness = harness_factory(args)
        harness.probe_trace = [] if trace else None
        for _ in range(intervals):
            harness.interval()
        torch.cuda.synchronize()
        states.append((
            torch.cat([p.detach().reshape(-1) for p in harness.agent.parameters()
                       if p.numel() < 10_000]).clone(),
            harness.update_metrics.clone(),
            harness.gate_losses.clone(),
            torch.cat([m.snr for _, m in harness.agent.plastic_sites]).clone(),
            torch.cat([m.lam for _, m in harness.agent.plastic_sites]).clone(),
            torch.cat([m.p_ref for _, m in harness.agent.plastic_sites]).clone(),
            torch.stack(harness.probe_trace).clone() if trace else None,
            torch.cat([m.probe.grad.detach().reshape(-1)
                       for _, m in harness.agent.plastic_sites]).clone(),
        ))
        print(f"  arm mode={mode!r} traced={trace}: "
              f"{len(harness.probe_trace) if trace else 0} minibatches traced, "
              f"final probe.grad elements {states[-1][7].numel()}")
        del harness
        torch.cuda.empty_cache()
    names = ("params", "update_metrics", "gate_losses", "snr", "lam", "p_ref",
             "probe_grad_trace", "probe_grad_final_raw")
    print("\n  v9-gate-default (traced) vs gate-reduce-overhead (traced):")
    for index, name in enumerate(names):
        a, b = states[0][index], states[1][index]
        if a is None or b is None:
            continue
        same = torch.equal(a, b)
        delta = (a - b).abs().max().item() if not same else 0.0
        rel = delta / max(a.abs().max().item(), 1e-30)
        print(f"    {name:<22} bitwise={same}  max|delta|={delta:.3e}  rel={rel:.3e}")
    trace_a, trace_b = states[0][6], states[1][6]
    print(f"    probe fingerprints compared: {trace_a.shape[0]} minibatches "
          f"x {trace_a.shape[1]} reductions")
    if not torch.equal(trace_a, trace_b):
        bad = (trace_a != trace_b).any(dim=1).nonzero().reshape(-1)
        print(f"    FIRST DIVERGING MINIBATCH: {bad[0].item()} of {trace_a.shape[0]}")
    if torch.isnan(trace_a).any():
        print("    WARNING: probe.grad was None in a traced minibatch -- vacuous there")
    print("\n  instrument control -- v9-gate-default traced vs UNTRACED "
          "(must be identical, else the fingerprint perturbs what it measures):")
    for index, name in enumerate(names):
        a, c = states[0][index], states[2][index]
        if a is None or c is None:
            continue
        same = torch.equal(a, c)
        delta = (a - c).abs().max().item() if not same else 0.0
        print(f"    {name:<22} bitwise={same}  max|delta|={delta:.3e}")


def measure_gate_modes(harness_factory, base_overrides, iters, warmup):
    banner("gate-supervision compile mode (identical math)")
    print(f"{'gate compile mode':<28}{'span s':>10}{'min s':>10}{'cpu s':>10}{'peak MiB':>10}")
    # A/B/A ordering: a drifting box shows up as disagreement between the two A
    # rows, so the B row is only credible when the A rows agree.
    for label, mode in [("default (v9 production)", None),
                        ("reduce-overhead", "reduce-overhead"),
                        ("default (repeat)", None)]:
        merged = dict(base_overrides)
        args = make_args(merged)
        args.gate_mode = mode
        try:
            harness = harness_factory(args)
            torch.cuda.reset_peak_memory_stats()
            med, lo, cpu = cuda_event_span(harness.interval, iters, warmup)
            peak = torch.cuda.max_memory_allocated() / 2**20
            print(f"{label:<28}{med:>10.4f}{lo:>10.4f}{cpu:>10.4f}{peak:>10.0f}")
            del harness
        except Exception as exc:
            print(f"{label:<28}FAILED: {type(exc).__name__}: {str(exc)[:70]}")
        torch.cuda.empty_cache()


def measure_minibatch_scaling(harness_factory, base_overrides, iters, warmup):
    banner("minibatch geometry (CHANGES MATH -- reported for reference only)")
    print(f"{'num_minibatches':<18}{'span s':>10}{'min s':>10}{'cpu s':>10}{'mb/interval':>13}")
    for nmb in (1, 2, 4):
        merged = dict(base_overrides)
        merged["num_minibatches"] = nmb
        args = make_args(merged)
        try:
            harness = harness_factory(args)
            med, lo, cpu = cuda_event_span(harness.interval, iters, warmup)
            print(f"{nmb:<18}{med:>10.4f}{lo:>10.4f}{cpu:>10.4f}"
                  f"{args.update_epochs * nmb:>13}")
            del harness
        except Exception as exc:
            print(f"{nmb:<18}FAILED: {exc}")
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--sections", default="stages,whole,launches,syncs,logging,fused",
                        help="comma list: stages,whole,launches,syncs,logging,fused,"
                             "alternatives,minibatch")
    cli = parser.parse_args()

    configure_runtime()
    report_load()
    base = {"num_envs": cli.num_envs, "num_steps": cli.num_steps}
    args = make_args(base)
    print(f"batch {args.batch_size} | minibatch {args.minibatch_size} | "
          f"epochs {args.update_epochs} | compile {args.compile}/{args.compile_mode} | "
          f"gate_every {args.gate_every}")

    sections = set(cli.sections.split(","))

    def factory(a):
        built = Harness(a)
        built.plasticity.updates = a.snr_warmup + 1
        return built

    harness = None
    if sections & {"stages", "whole", "launches", "syncs", "logging", "fused",
                   "tailgraph", "skiptail", "gpuwork", "dup"}:
        harness = Harness(args)
        # ``PlasticityStepper.levels`` early-returns for the first ``snr_warmup``
        # updates; a real run spends 10 iterations there out of thousands, so
        # arm past it or every measurement below understates ``levels``.
        harness.plasticity.updates = args.snr_warmup + 1
        print(f"plastic sites: {len(harness.agent.plastic_sites)} | "
              f"net tensors {len(harness.net_params)} | gate tensors {len(harness.gate_params)}")

    if "whole" in sections:
        measure_whole(harness, cli.iters, cli.warmup)
        report_load()
    if "stages" in sections:
        measure_stages(harness, cli.iters, cli.warmup)
        report_load()
    if "launchcost" in sections:
        measure_launch_cost(torch.device("cuda"))
    if "launches" in sections:
        measure_launches(harness)
    if "syncs" in sections:
        measure_syncs(harness)
    if "logging" in sections:
        measure_logging(harness, 200)
    if "fused" in sections:
        measure_fused_plasticity(harness, 200)
    if "gpuwork" in sections:
        measure_gpu_work(harness, 20)
    if "dup" in sections:
        measure_duplicate_state_logits(harness, 20)
    if "skiptail" in sections:
        measure_skip_tail(harness, cli.iters, cli.warmup)
    if "tailgraph" in sections:
        measure_tail_graph(harness, cli.iters, cli.warmup)
    if harness is not None and sections & {"alternatives", "minibatch", "gatemodes", "cse"}:
        del harness
        torch.cuda.empty_cache()
    if "cse" in sections:
        measure_linear_cse(factory, base, cli.iters, cli.warmup)
    if "gatemodes" in sections:
        measure_gate_modes(factory, base, cli.iters, cli.warmup)
    if "gatecheck" in sections:
        measure_gate_mode_equivalence(factory, base)
    if "alternatives" in sections:
        measure_alternatives(factory, base, cli.iters, cli.warmup)
    if "minibatch" in sections:
        measure_minibatch_scaling(factory, base, cli.iters, cli.warmup)
    report_load()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()

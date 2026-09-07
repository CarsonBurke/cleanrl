"""Whole-machine throughput audit: concurrency, affinity and startup cost.

Single-run latency is the wrong figure of merit on a box that runs several
trainers at once. What matters is aggregate environment steps per second and
steps per CPU core. This script answers three questions the per-link
microbenchmarks cannot:

- ``grid``: aggregate SPS over a (runs x threads) sweep, with repeats so the
  noise floor is measured rather than assumed. Workers are the *same* worker
  entry point as ``benchmark_rollout_scale.py``, so numbers are comparable.
- ``startup``: wall time from process launch to the first environment step of
  the real trainer, and how much of the first iteration is torch.compile,
  measured against a warm and a deliberately cold inductor/triton cache.
- ``affinity``: the grid again under CPU pinning plans, including plans that
  respect this box's two L3 domains (Zen 5 CCDs) and plans that avoid SMT
  siblings.

Every mode records ``/proc/loadavg`` alongside each result: a throughput number
taken on a contended box is not a measurement, it is a rumour.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCALE = REPO / "scripts" / "benchmark_rollout_scale.py"
DEFAULT_TRAINER = "cleanrl/plasticity/ppo_continuous_action_sphere_sdplast_v9.py"

# Zen 5 (9900X): 12 physical cores, SMT sibling of core N is logical N+12,
# and L3 is split into two 32MiB instances -- cores 0-5 and cores 6-11.
PHYSICAL = 12
CCDS = ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11))


def loadavg():
    return float(Path("/proc/loadavg").read_text().split()[0])


def chunk(items, parts):
    """Split ``items`` into ``parts`` near-equal contiguous groups."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    size, extra = divmod(len(items), parts)
    out, start = [], 0
    for index in range(parts):
        width = size + (1 if index < extra else 0)
        out.append(items[start:start + width])
        start += width
    return out


def affinity_plan(plan, runs, smt=False):
    """Return one logical-CPU list per run, or ``None`` for no pinning.

    ``phys`` shares physical cores 0-11 between every run. ``disjoint`` cuts
    those cores into contiguous equal blocks, which for some run counts
    straddles the CCD boundary. ``ccd`` refuses to straddle it: runs are dealt
    alternately to the two L3 domains and split the six cores inside one.
    """
    if plan == "none":
        return None
    if plan == "phys":
        cores = [list(range(PHYSICAL))] * runs
    elif plan == "disjoint":
        cores = chunk(list(range(PHYSICAL)), runs)
    elif plan == "ccd":
        assigned = [[], []]
        for index in range(runs):
            assigned[index % 2].append(index)
        cores = [None] * runs
        for domain, members in zip(CCDS, assigned):
            if not members:
                continue
            for member, block in zip(members, chunk(list(domain), len(members))):
                cores[member] = block
    else:
        raise ValueError(f"unknown plan {plan}")
    if smt:
        cores = [sorted(set(block) | {core + PHYSICAL for core in block}) for block in cores]
    if any(not block for block in cores):
        raise ValueError(f"plan {plan} cannot serve {runs} runs")
    return cores


class MeasurementFailed(RuntimeError):
    """A configuration could not be measured (typically VRAM exhaustion)."""


def run_point(args, runs, threads, plan="none", smt=False, extra_env=None):
    """Spawn ``runs`` rollout workers and aggregate their self-reported stats."""
    base = [sys.executable, str(SCALE), "--worker",
            "--env-id", args.env_id, "--num-envs", str(args.num_envs),
            "--threads", str(threads), "--width", str(args.width),
            "--n-blocks", str(args.n_blocks), "--seconds", str(args.seconds),
            "--warmup", str(args.warmup)]
    if args.host_graph:
        base.append("--host-graph")
    cores = affinity_plan(plan, runs, smt)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.update(extra_env or {})
    load_before = loadavg()
    processes = []
    for index in range(runs):
        command = list(base) + ["--seed", str(index + 1)]
        if cores is not None:
            command = ["taskset", "-c", ",".join(str(c) for c in cores[index])] + command
        processes.append(subprocess.Popen(command, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True,
                                          cwd=str(REPO), env=environment))
    results, failures = [], []
    for process in processes:
        out, err = process.communicate()
        line = next((l for l in out.splitlines() if l.startswith("{")), None)
        if process.returncode != 0 or line is None:
            failures.append(f"rc={process.returncode} {(err or out).strip()[-300:]}")
            continue
        results.append(json.loads(line))
    if failures:
        raise MeasurementFailed(f"{len(failures)}/{runs} workers failed; "
                                f"first: {failures[0]}")
    env_steps = sum(r["env_steps"] for r in results)
    wall = max(r["wall"] for r in results)
    cpu = sum(r["cpu"] for r in results)
    per_run = sorted(r["env_steps"] / r["wall"] for r in results)
    return {"runs": runs, "threads": threads, "plan": plan, "smt": smt,
            "aggregate_sps": env_steps / wall, "per_run_sps_min": per_run[0],
            "per_run_sps_max": per_run[-1], "cores": cpu / wall,
            "sps_per_core": env_steps / cpu, "load_before": load_before,
            "load_after": loadavg()}


def summarize(label, samples):
    aggregate = [s["aggregate_sps"] for s in samples]
    cores = [s["cores"] for s in samples]
    per_core = [s["sps_per_core"] for s in samples]
    best = max(aggregate)
    spread = (best - min(aggregate)) / best * 100 if best else 0.0
    print(f"{label:<34} agg={statistics.median(aggregate):>9.0f} "
          f"(min {min(aggregate):>9.0f} max {best:>9.0f} spread {spread:4.1f}%) "
          f"cores={statistics.median(cores):5.2f} "
          f"sps/core={statistics.median(per_core):>7.0f} "
          f"per_run={statistics.median([s['per_run_sps_min'] for s in samples]):>8.0f}.."
          f"{statistics.median([s['per_run_sps_max'] for s in samples]):>8.0f} "
          f"load={statistics.median([s['load_before'] for s in samples]):5.2f}",
          flush=True)
    return {"label": label, "median_aggregate_sps": statistics.median(aggregate),
            "min_aggregate_sps": min(aggregate), "max_aggregate_sps": best,
            "spread_pct": spread, "median_cores": statistics.median(cores),
            "median_sps_per_core": statistics.median(per_core),
            "samples": samples}


def sweep(args, points, header):
    """Measure ``points`` with repeats interleaved, not batched per point.

    On a box with other tenants the background load drifts over minutes. Taking
    every repeat of one configuration back-to-back bakes that drift into the
    comparison between configurations. Cycling through all configurations once
    per repeat spreads the drift evenly, so the *ranking* survives even when the
    absolute numbers do not.
    """
    print(header, flush=True)
    samples = {label: [] for label, _ in points}
    for cycle in range(args.repeats):
        for label, call in points:
            if label not in samples:
                continue
            try:
                samples[label].append(call())
            except (ValueError, MeasurementFailed) as error:
                print(f"UNMEASURABLE {label}: {error}", flush=True)
                samples.pop(label, None)
        print(f"# cycle {cycle + 1}/{args.repeats} done, load={loadavg():.2f}", flush=True)
    records = [summarize(label, samples[label]) for label, _ in points if samples.get(label)]
    best = max(records, key=lambda r: r["median_aggregate_sps"])
    print(f"\nbest aggregate: {best['label']} -> {best['median_aggregate_sps']:.0f} SPS "
          f"on {best['median_cores']:.2f} cores ({best['median_sps_per_core']:.0f} SPS/core)")
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1))
    return records


def mode_grid(args):
    points = []
    for runs in args.runs_grid:
        for threads in args.threads_grid:
            if threads > args.num_envs:
                continue
            label = f"runs={runs:<2} threads={threads:<2} plan={args.plan}{'+smt' if args.smt else ''}"
            points.append((label, lambda r=runs, t=threads: run_point(
                args, r, t, plan=args.plan, smt=args.smt)))
    return sweep(args, points,
                 f"# grid num_envs={args.num_envs} width={args.width} "
                 f"blocks={args.n_blocks} host_graph={args.host_graph} "
                 f"seconds={args.seconds} repeats={args.repeats}")


def mode_affinity(args):
    points = []
    for runs in args.runs_grid:
        for threads in args.threads_grid:
            for plan, smt in args.plans:
                label = f"runs={runs:<2} threads={threads:<2} plan={plan}{'+smt' if smt else ''}"
                points.append((label, lambda r=runs, t=threads, p=plan, s=smt: run_point(
                    args, r, t, plan=p, smt=s)))
    return sweep(args, points, f"# affinity runs={args.runs_grid} "
                               f"threads={args.threads_grid} repeats={args.repeats}")


def mode_malloc(args):
    """Does glibc arena behaviour or a different allocator move the needle?"""
    runs, threads = args.runs_grid[0], args.threads_grid[0]
    variants = {"baseline": {}, "arena_max_1": {"MALLOC_ARENA_MAX": "1"},
                "arena_max_2": {"MALLOC_ARENA_MAX": "2"},
                "trim_off": {"MALLOC_TRIM_THRESHOLD_": "-1", "MALLOC_MMAP_MAX_": "0"}}
    points = [(f"malloc={name}", lambda e=environment: run_point(
        args, runs, threads, plan=args.plan, extra_env=e))
        for name, environment in variants.items()]
    return sweep(args, points, f"# malloc runs={runs} threads={threads} "
                               f"repeats={args.repeats}")


def mode_spin(args):
    """Spinning physics workers trade peer CPU for their own wake latency.

    A spinning worker stays runnable, so it is dispatched the instant work
    arrives instead of queueing behind other runnable threads. That is free on
    an idle box and negative once the machine is oversubscribed, because the
    burnt cycles belong to a peer run. Only an aggregate sweep can price it.
    """
    points = []
    for runs in args.runs_grid:
        for threads in args.threads_grid:
            for spin in args.spin_grid:
                label = f"runs={runs:<2} threads={threads:<2} spin={spin:<5}"
                points.append((label, lambda r=runs, t=threads, s=spin: run_point(
                    args, r, t, plan=args.plan,
                    extra_env={"CLEANRL_ENV_SPIN": str(s)})))
    return sweep(args, points, f"# spin runs={args.runs_grid} "
                               f"threads={args.threads_grid} spin={args.spin_grid} "
                               f"repeats={args.repeats}")


STARTUP_PROBE = r"""
import json, os, runpy, sys, time
T0 = float(os.environ["AUDIT_T0"])
marks = {"interpreter": time.time() - T0}

import torch
marks["import_torch"] = time.time() - T0

sys.argv = json.loads(os.environ["AUDIT_ARGV"])
trainer = os.environ["AUDIT_TRAINER"]
import importlib.util
spec = importlib.util.spec_from_file_location("_trainer_probe", trainer)
module = importlib.util.module_from_spec(spec)
sys.modules["_trainer_probe"] = module
spec.loader.exec_module(module)
marks["import_trainer"] = time.time() - T0

original_env = module.make_training_env
first = {}

NUM_STEPS = int(os.environ["AUDIT_NUM_STEPS"])
bounds = []

# Differencing two separate launches cannot isolate a steady iteration here:
# compile plus cudagraph capture varies by more than ten seconds between
# launches, which swamps a sub-second iteration. Counting env steps inside one
# process and stamping every num_steps-th one removes that variance entirely.
class StepProbe:
    def __init__(self, inner):
        self._inner = inner
        self._count = 0
    def __getattr__(self, name):
        return getattr(self._inner, name)
    def step(self, action):
        if "first_step" not in first:
            first["first_step"] = time.time() - T0
            bounds.append(time.time() - T0)
        self._count += 1
        if self._count % NUM_STEPS == 0:
            bounds.append(time.time() - T0)
        return self._inner.step(action)

def probed_env(args, run_name):
    marks["pre_env"] = time.time() - T0
    envs = original_env(args, run_name)
    marks["post_env"] = time.time() - T0
    return StepProbe(envs)

module.make_training_env = probed_env
try:
    module.main()
finally:
    marks.update(first)
    marks["iteration_bounds"] = bounds
    # Exact cache accounting beats inferring it from wall time: dynamo keeps
    # hit/miss counters per cache, and reports where compile time went.
    try:
        from torch._dynamo.utils import counters
        marks["counters"] = {group: dict(values) for group, values in counters.items()
                             if values and group in ("inductor", "aot_autograd")}
    except Exception as error:
        marks["counters_error"] = repr(error)
    try:
        from torch._dynamo.utils import compile_times
        marks["compile_times"] = compile_times(repr="csv", aggregate=True)
    except Exception as error:
        marks["compile_times_error"] = repr(error)
    marks["exit"] = time.time() - T0
    sys.stderr.write("AUDITMARKS " + json.dumps(marks, default=str) + "\n")
    sys.stderr.flush()
"""


def launch_startups(args, iterations, compile_on, cold_cache, *, count=1,
                    compile_threads=None):
    """Launch ``count`` trainers *simultaneously* and time each one's phases.

    Solo compile time is the wrong operational number: runs are submitted with
    ``--max-parallel-runs``, so several processes enter their compile window at
    once, each forking ``compile_threads`` workers. Only a concurrent launch
    exposes that interaction.
    """
    horizon = 1000 * args.num_envs if args.num_envs > 1 else 0
    batch = args.num_envs * args.trainer_steps
    total = horizon + batch * iterations
    trainer = REPO / args.trainer
    argv = [str(trainer), "--env-id", args.env_id, "--num-envs", str(args.num_envs),
            "--num-steps", str(args.trainer_steps), "--total-timesteps", str(total),
            "--env-threads", str(args.env_threads),
            "--compile" if compile_on else "--no-compile"]
    base = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                AUDIT_TRAINER=str(trainer),
                AUDIT_NUM_STEPS=str(args.trainer_steps))
    if compile_threads is not None:
        base["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
    scratches, processes = [], []
    load_before = loadavg()
    start = time.time()
    try:
        for index in range(count):
            environment = dict(base, AUDIT_T0=repr(start))
            # Distinct seeds keep the runs from sharing a tensorboard run name.
            environment["AUDIT_ARGV"] = json.dumps(argv + ["--seed", str(index + 1)])
            if cold_cache:
                scratch = tempfile.mkdtemp(prefix="audit_cold_cache_")
                scratches.append(scratch)
                environment["TORCHINDUCTOR_CACHE_DIR"] = str(Path(scratch) / "inductor")
                environment["TRITON_CACHE_DIR"] = str(Path(scratch) / "triton")
            processes.append(subprocess.Popen(
                [sys.executable, "-c", STARTUP_PROBE], cwd=str(REPO), env=environment,
                text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        results = []
        for process in processes:
            _, err = process.communicate(timeout=args.startup_timeout)
            wall = time.time() - start
            line = next((l for l in err.splitlines() if l.startswith("AUDITMARKS")), None)
            if process.returncode != 0 or line is None:
                raise RuntimeError(f"trainer probe failed (rc={process.returncode}):\n"
                                   f"{err[-1500:]}")
            marks = json.loads(line.split(" ", 1)[1])
            marks.update({"wall": wall, "iterations": iterations, "compile": compile_on,
                          "cold_cache": cold_cache, "count": count,
                          "compile_threads": compile_threads, "load_before": load_before})
            results.append(marks)
    finally:
        for scratch in scratches:
            shutil.rmtree(scratch, ignore_errors=True)
    return results


def measure_startup(args, iterations, compile_on, cold_cache):
    return launch_startups(args, iterations, compile_on, cold_cache)[0]


def mode_trainer(args):
    """Aggregate throughput of the REAL trainer, not just the rollout chain.

    ``benchmark_rollout_scale`` measures the rollout chain alone, so its SPS is
    far above a trainer's: the trainer also pays for the upload, the batched
    statistics forward, GAE and the optimizer. Recommending a concurrency from
    the proxy alone would overstate wall clock, so this differences a
    1-iteration launch against a 3-iteration launch at the same concurrency.
    The gap is the steady iteration with every fixed cost removed.
    """
    print(f"# trainer concurrency num_envs={args.num_envs} "
          f"num_steps={args.trainer_steps} batch={args.num_envs * args.trainer_steps} "
          f"target_steps={args.target_steps}", flush=True)
    batch = args.num_envs * args.trainer_steps
    rows = []
    for count in args.launch_grid:
        for threads in args.threads_grid:
            args.env_threads = threads
            try:
                marks = launch_startups(args, args.trainer_iterations, True, False,
                                        count=count)
            except RuntimeError as error:
                print(f"UNMEASURABLE runs={count} threads={threads}: {error}", flush=True)
                continue
            # Iteration 1 carries first-touch faults and the tail of compile, so
            # steady state is the median of the later gaps within each process.
            steady_per_run = []
            for mark in marks:
                bounds = mark.get("iteration_bounds") or []
                gaps = [b - a for a, b in zip(bounds, bounds[1:])][1:]
                if gaps:
                    steady_per_run.append(statistics.median(gaps))
            if not steady_per_run:
                print(f"no iteration bounds for runs={count} threads={threads}", flush=True)
                continue
            slowest = max(steady_per_run)
            per_run_sps = batch / slowest
            aggregate = sum(batch / s for s in steady_per_run)
            minutes = args.target_steps / per_run_sps / 60.0
            rows.append({"runs": count, "threads": threads,
                         "steady_iteration_s": slowest, "per_run_sps": per_run_sps,
                         "aggregate_sps": aggregate, "minutes_for_target": minutes,
                         "iteration_spread_s": [round(s, 3) for s in sorted(steady_per_run)],
                         "load_before": marks[0]["load_before"]})
            print(f"runs={count:<2} threads={threads:<2} slowest_iter={slowest:6.3f}s "
                  f"per_run_SPS={per_run_sps:>8.0f} aggregate_SPS={aggregate:>9.0f} "
                  f"wall_{args.target_steps // 1000000}M={minutes:6.2f}min "
                  f"load={marks[0]['load_before']:5.2f}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return rows


def mode_compile(args):
    """Is per-run ``compile_threads`` starving concurrent launches?"""
    print(f"# compile-threads sweep num_envs={args.num_envs} "
          f"num_steps={args.trainer_steps} repeats={args.repeats}", flush=True)
    rows = []
    for cap in args.compile_threads_grid:
        for count in args.launch_grid:
            for cycle in range(args.repeats):
                marks = launch_startups(args, 1, True, False, count=count,
                                        compile_threads=cap)
                first = sorted(m["first_step"] for m in marks)
                exits = sorted(m["exit"] for m in marks)
                walls = sorted(m["wall"] for m in marks)
                rows.append({"cap": cap, "count": count, "cycle": cycle,
                             "first_step_max": first[-1], "exit_max": exits[-1],
                             "wall_max": walls[-1], "load_before": marks[0]["load_before"]})
                print(f"compile_threads={cap:<3} launches={count:<2} cycle={cycle} "
                      f"first_step={first[0]:6.2f}..{first[-1]:6.2f}s "
                      f"main_done={exits[0]:6.2f}..{exits[-1]:6.2f}s "
                      f"process_exit={walls[-1]:6.2f}s load={marks[0]['load_before']:5.2f}",
                      flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return rows


def mode_startup(args):
    print(f"# startup env={args.env_id} num_envs={args.num_envs} "
          f"num_steps={args.trainer_steps} (batch={args.num_envs * args.trainer_steps})",
          flush=True)
    records = []
    for compile_on in (True, False):
        for cold in ((False, True) if compile_on else (False,)):
            marks = {}
            for iterations in (1, 2):
                marks[iterations] = measure_startup(args, iterations, compile_on, cold)
            one, two = marks[1], marks[2]
            steady = two["wall"] - one["wall"]
            overhead = one["wall"] - steady
            tag = f"compile={compile_on} cold_cache={cold}"
            print(f"{tag:<34} launch->first_step={one['first_step']:6.2f}s "
                  f"(torch import {one['import_torch']:4.2f}s, trainer import "
                  f"{one['import_trainer'] - one['import_torch']:4.2f}s, env build "
                  f"{one['post_env'] - one['pre_env']:4.2f}s) "
                  f"iter1_total={one['wall']:6.2f}s iter2_total={two['wall']:6.2f}s "
                  f"steady_iter={steady:5.2f}s fixed_overhead={overhead:6.2f}s "
                  f"load={one['load_before']:.2f}", flush=True)
            records.append({"tag": tag, "one": one, "two": two,
                            "steady_iteration_s": steady, "fixed_overhead_s": overhead})
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grid", "affinity", "startup", "malloc",
                                        "compile", "spin", "trainer"])
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--host-graph", action="store_true")
    parser.add_argument("--runs-grid", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--threads-grid", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--plan", default="none", choices=["none", "phys", "disjoint", "ccd"])
    parser.add_argument("--smt", action="store_true", help="extend pinned sets with SMT siblings")
    parser.add_argument("--trainer-steps", type=int, default=2048)
    parser.add_argument("--env-threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--compile-threads-grid", type=int, nargs="+", default=[24, 8, 4])
    parser.add_argument("--launch-grid", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--spin-grid", type=int, nargs="+", default=[0, 5000])
    parser.add_argument("--target-steps", type=int, default=8000000)
    parser.add_argument("--trainer-iterations", type=int, default=4)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER,
                        help="repo-relative trainer script to launch")
    parser.add_argument("--plans", nargs="+", default=None,
                        help="affinity plans as name[+smt], e.g. none disjoint+smt")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    default_plans = ["none", "phys", "disjoint", "disjoint+smt", "ccd", "ccd+smt"]
    args.plans = [(name.removesuffix("+smt"), name.endswith("+smt"))
                  for name in (args.plans or default_plans)]
    {"grid": mode_grid, "affinity": mode_affinity, "startup": mode_startup,
     "malloc": mode_malloc, "compile": mode_compile,
     "spin": mode_spin, "trainer": mode_trainer}[args.mode](args)


if __name__ == "__main__":
    main()

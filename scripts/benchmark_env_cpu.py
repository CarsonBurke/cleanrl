"""Wall AND CPU cost of the native vector env step under a realistic duty cycle.

The rollout loop calls ``envs.step`` once per ~150us of surrounding Python
work, so the physics workers spend most of their life *between* batches. That
makes the idle policy, not the parallel speedup, the dominant term in a run's
CPU cost -- and it is invisible in a tight-loop benchmark. ``--gap-us``
emulates the policy/normalize/transfer work that separates two physics calls,
and the report includes process CPU per step so oversubscription waste is
measurable instead of inferred.

This is the measurement that condemned libgomp for this workload: at 4 threads
its default spin burned 1630us of CPU per step to back 614us of real work,
``OMP_WAIT_POLICY=passive`` cost more wall time than the physics it
parallelized, and intermediate ``GOMP_SPINCOUNT`` values were worse in *both*
dimensions. ``mujoco_batch.c`` now uses a parking pool instead; keep this
benchmark honest by always reading the ``loop_cpu``/``cores`` columns
alongside ``step_med``.
"""

from __future__ import annotations

import argparse
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def spin(microseconds):
    """Burn a fixed amount of single-threaded CPU, GIL held (like NumPy act)."""
    deadline = time.perf_counter() + microseconds * 1e-6
    while time.perf_counter() < deadline:
        pass


def measure(envs, action, iters, gap_us, warmup=200):
    for _ in range(warmup):
        envs.step(action)
    samples = []
    cpu_start, wall_start = cpu_seconds(), time.perf_counter()
    for _ in range(iters):
        start = time.perf_counter()
        envs.step(action)
        samples.append(time.perf_counter() - start)
        if gap_us:
            spin(gap_us)
    wall = time.perf_counter() - wall_start
    cpu = cpu_seconds() - cpu_start
    return {
        "step_median_us": statistics.median(samples) * 1e6,
        "step_min_us": min(samples) * 1e6,
        "loop_wall_us": wall / iters * 1e6,
        "loop_cpu_us": cpu / iters * 1e6,
        "cores": cpu / wall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--gap-us", type=float, default=250.0)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--backend", default="native")
    args = parser.parse_args()

    from cleanrl.shared.mujoco_env import make_mujoco_vector_env

    policy = os.environ.get("OMP_WAIT_POLICY", "<unset>")
    spincount = os.environ.get("GOMP_SPINCOUNT", "<unset>")
    print(f"env={args.env_id} num_envs={args.num_envs} gap={args.gap_us:.0f}us "
          f"OMP_WAIT_POLICY={policy} GOMP_SPINCOUNT={spincount}")
    print(f"  {'threads':<8s} {'step_med':>9s} {'step_min':>9s} {'loop_wall':>10s} "
          f"{'loop_cpu':>9s} {'cores':>6s}")
    for threads in args.threads:
        envs = make_mujoco_vector_env(args.env_id, args.num_envs, backend=args.backend,
                                      num_threads=threads)
        envs.reset(seed=1)
        action = np.zeros(envs.action_space.shape, dtype=np.float32)
        result = measure(envs, action, args.iters, args.gap_us)
        envs.close()
        print(f"  {threads:<8d} {result['step_median_us']:9.1f} {result['step_min_us']:9.1f} "
              f"{result['loop_wall_us']:10.1f} {result['loop_cpu_us']:9.1f} {result['cores']:6.2f}")


if __name__ == "__main__":
    main()

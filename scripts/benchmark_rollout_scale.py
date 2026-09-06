"""Aggregate rollout throughput across *concurrent* runs.

Single-process microbenchmarks cannot answer the question that matters on a
shared box: several trainers run at once, so what maximizes total environment
steps per second is throughput per CPU core, not latency per step. A physics
thread pool that spins while idle looks fast in isolation and starves its
peers in production; one that parks looks slow in isolation and may win by a
wide margin once three runs share twelve cores.

This spawns `--runs` independent worker processes, each driving the same
rollout chain the trainers use (native vector env + host SiTU-sphere policy +
vectorized normalization), and reports per-run SPS, aggregate SPS and total
CPU cores consumed. Sweep `--threads` and `CLEANRL_ENV_SPIN` to pick an
operating point from data.

Run it with the machine otherwise idle; contending jobs invalidate the result.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def worker(args):
    import numpy as np
    import torch
    from torch import nn

    from cleanrl.shared.host_actor import HostSiTUSphereActor, make_situ_sphere_trunk
    from cleanrl.shared.host_graph import HostGraphActor
    from cleanrl.shared.mujoco_env import make_mujoco_vector_env
    from cleanrl.shared.runtime import configure_runtime
    from cleanrl.shared.sampling import sample_beta_actions_host
    from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

    configure_runtime()
    envs = make_mujoco_vector_env(args.env_id, args.num_envs, backend="native",
                                  num_threads=args.threads)
    raw, _ = envs.reset(seed=args.seed)
    obs_dim, act_dim = raw.shape[1], envs.single_action_space.shape[0]
    trunk = make_situ_sphere_trunk(obs_dim, args.width, args.n_blocks).cuda()
    head = nn.Linear(args.width, 2 * act_dim).cuda()
    mirror = HostGraphActor if args.host_graph else HostSiTUSphereActor
    actor = mirror(nn.Sequential(trunk, head), args.num_envs)
    obs_norm = VectorObsNorm(args.num_envs, (obs_dim,))
    rew_norm = VectorRewardNorm(args.num_envs, 0.99)
    low = np.full(act_dim, -1.0, dtype=np.float32)
    high = np.full(act_dim, 1.0, dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    obs = obs_norm.normalize(raw)

    def loop(deadline):
        nonlocal obs
        steps = 0
        while time.perf_counter() < deadline:
            for _ in range(64):
                _, action = sample_beta_actions_host(actor(obs), low, high, rng)
                raw_obs, reward, terms, truncs, infos = envs.step(action)
                rew_norm.normalize(reward, terms)
                obs, _ = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
            steps += 64
        return steps

    loop(time.perf_counter() + args.warmup)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_start = usage.ru_utime + usage.ru_stime
    wall_start = time.perf_counter()
    steps = loop(wall_start + args.seconds)
    wall = time.perf_counter() - wall_start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu = usage.ru_utime + usage.ru_stime - cpu_start
    envs.close()
    print(json.dumps({"steps": steps, "wall": wall, "cpu": cpu,
                      "env_steps": steps * args.num_envs}), flush=True)


def spawn(args):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--env-id", args.env_id, "--num-envs", str(args.num_envs),
               "--threads", str(args.threads), "--width", str(args.width),
               "--n-blocks", str(args.n_blocks), "--seconds", str(args.seconds),
               "--warmup", str(args.warmup)]
    if args.host_graph:
        command.append("--host-graph")
    processes = []
    for index in range(args.runs):
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        processes.append(subprocess.Popen(
            command + ["--seed", str(index + 1)], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, cwd=str(REPO), env=environment))
    results = []
    for process in processes:
        out, _ = process.communicate()
        line = next((l for l in out.splitlines() if l.startswith("{")), None)
        if process.returncode != 0 or line is None:
            raise RuntimeError(f"worker failed (rc={process.returncode}): {out[-500:]}")
        results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--warmup", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--host-graph", action="store_true",
                        help="mirror the policy with the fused native HostGraphActor")
    args = parser.parse_args()

    if args.worker:
        worker(args)
        return

    results = spawn(args)
    env_steps = sum(r["env_steps"] for r in results)
    wall = max(r["wall"] for r in results)
    cpu = sum(r["cpu"] for r in results)
    per_run = [r["env_steps"] / r["wall"] for r in results]
    spin = os.environ.get("CLEANRL_ENV_SPIN", "<default>")
    mirror = "host_graph" if args.host_graph else "numpy_mirror"
    print(f"runs={args.runs} threads={args.threads} spin={spin} policy={mirror} "
          f"num_envs={args.num_envs}: per_run_SPS={min(per_run):.0f}..{max(per_run):.0f} "
          f"aggregate_SPS={env_steps / wall:.0f} cores={cpu / wall:.2f} "
          f"SPS_per_core={env_steps / cpu:.0f}")


if __name__ == "__main__":
    main()

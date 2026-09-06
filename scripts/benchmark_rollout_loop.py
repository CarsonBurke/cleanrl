"""Whole-rollout-loop throughput and CPU cost under thread/wait-policy settings.

Per-link microbenchmarks hide two effects that dominate the real loop:

* libgomp's idle workers spin between the ~2000 short physics calls of a
  rollout, so a run's CPU cost is roughly ``num_threads`` cores no matter how
  little physics it does. On a shared box that steals cores from every peer.
* the fixed per-call cost (NumPy dispatch, ctypes marshalling, Gymnasium
  bookkeeping) is far larger than the arithmetic, so anything that multiplies
  the number of calls per environment-step loses even when it adds parallelism.

This measures the assembled loop exactly as the trainers run it and reports
wall, CPU and cores, which is what has to improve for concurrent runs.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.host_actor import HostSiTUSphereActor, make_situ_sphere_trunk
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions_host
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def build(env_id, num_envs, threads, width, blocks, seed=1):
    envs = make_mujoco_vector_env(env_id, num_envs, backend="native", num_threads=threads)
    raw, _ = envs.reset(seed=seed)
    obs_dim = raw.shape[1]
    act_dim = envs.single_action_space.shape[0]
    trunk = make_situ_sphere_trunk(obs_dim, width, blocks).cuda()
    head = nn.Linear(width, 2 * act_dim).cuda()
    actor = HostSiTUSphereActor(nn.Sequential(trunk, head), num_envs)
    obs_norm = VectorObsNorm(num_envs, (obs_dim,))
    rew_norm = VectorRewardNorm(num_envs, 0.99)
    low = np.full(act_dim, -1.0, dtype=np.float32)
    high = np.full(act_dim, 1.0, dtype=np.float32)
    return envs, actor, obs_norm, rew_norm, low, high, obs_norm.normalize(raw)


def run_loop(state, steps, rng):
    envs, actor, obs_norm, rew_norm, low, high, obs = state
    for _ in range(steps):
        _, action = sample_beta_actions_host(actor(obs), low, high, rng)
        raw, reward, terms, truncs, infos = envs.step(action)
        rew_norm.normalize(reward, terms)
        obs, _ = obs_norm.normalize_step(raw, terms, truncs, infos)
    return obs


def measure(state, steps, num_envs, warmup=200):
    rng = np.random.default_rng(1)
    run_loop(state, warmup, rng)
    cpu_start, wall_start = cpu_seconds(), time.perf_counter()
    run_loop(state, steps, rng)
    wall = time.perf_counter() - wall_start
    cpu = cpu_seconds() - cpu_start
    return {
        "us_per_step": wall / steps * 1e6,
        "sps": num_envs * steps / wall,
        "cpu_us": cpu / steps * 1e6,
        "cores": cpu / wall,
        "sps_per_core": num_envs * steps / cpu,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    configure_runtime()
    import os

    print(f"env={args.env_id} num_envs={args.num_envs} width={args.width} "
          f"blocks={args.n_blocks} OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY', '<unset>')}")
    print(f"  {'threads':<8s} {'us/step':>9s} {'SPS':>9s} {'cpu_us':>9s} {'cores':>7s} {'SPS/core':>9s}")
    for threads in args.threads:
        state = build(args.env_id, args.num_envs, threads, args.width, args.n_blocks)
        result = measure(state, args.steps, args.num_envs)
        state[0].close()
        print(f"  {threads:<8d} {result['us_per_step']:9.1f} {result['sps']:9.0f} "
              f"{result['cpu_us']:9.1f} {result['cores']:7.2f} {result['sps_per_core']:9.0f}")


if __name__ == "__main__":
    main()

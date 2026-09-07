"""End-to-end per-step attribution for the *live* rollout chain (sphere policy).

``profile_rollout_step.py`` measures the plain two-layer MLP used by
``ppo_continuous_action.py``. The runs that are actually saturating the machine
use a SiTU-sphere trunk whose host mirror issues ~85 NumPy calls per step on
(num_envs, width) arrays, so it is dominated by per-call dispatch rather than
FLOPs. This script measures that real chain, link by link, in wall *and* CPU
time, and reports what a perfectly pipelined policy/physics overlap would cost.
"""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.host_actor import HostSiTUSphereActor, make_situ_sphere_trunk
from cleanrl.shared.host_graph import HostGraphActor
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions_host
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def timeit(fn, iters, warmup=100):
    for _ in range(warmup):
        fn()
    samples = []
    cpu_start = cpu_seconds()
    wall_start = time.perf_counter()
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    wall = time.perf_counter() - wall_start
    cpu = cpu_seconds() - cpu_start
    return statistics.median(samples) * 1e6, min(samples) * 1e6, cpu / iters * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--host-graph", action="store_true",
                        help="mirror the policy with the fused native HostGraphActor")
    parser.add_argument("--fused-beta", action="store_true",
                        help="fuse the Beta head's pre/post arithmetic into the kernel")
    args = parser.parse_args()

    configure_runtime()
    device = torch.device("cuda")
    envs = make_mujoco_vector_env(args.env_id, args.num_envs, backend="native",
                                 num_threads=args.threads)
    raw_obs, _ = envs.reset(seed=1)
    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    trunk = make_situ_sphere_trunk(obs_dim, args.width, args.n_blocks).to(device)
    head = nn.Linear(args.width, 2 * act_dim).to(device)
    actor = nn.Sequential(trunk, head)
    mirror = HostGraphActor if args.host_graph else HostSiTUSphereActor
    host_actor = mirror(actor, args.num_envs)

    low = np.full(act_dim, -1.0, dtype=np.float32)
    high = np.full(act_dim, 1.0, dtype=np.float32)
    sampler = np.random.default_rng(1)
    # Both arms are called through one extra Python frame, so the A/B stays
    # paired: what differs between them is only the head's own work.
    if args.fused_beta:
        head = make_beta_sampler(args.num_envs, act_dim, low, high)
        if not head.fused:
            raise SystemExit(f"--fused-beta requested but unavailable: {head.fallback_reason}")

        def sample(values):
            return head(values, sampler)
    else:
        def sample(values):
            return sample_beta_actions_host(values, low, high, sampler)

    obs_norm = VectorObsNorm(args.num_envs, obs_shape)
    rew_norm = VectorRewardNorm(args.num_envs, 0.99)
    transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                               fields={"observations": obs_shape, "native_actions": (act_dim,)})

    obs = obs_norm.normalize(raw_obs)
    action = np.zeros((args.num_envs, act_dim), dtype=np.float32)
    raw, rew, terms, truncs, infos = envs.step(action)

    results = {}
    results["policy_forward"] = timeit(lambda: host_actor(obs), args.iters)
    logits = host_actor(obs)
    results["beta_sample"] = timeit(lambda: sample(logits), args.iters)
    results["policy_chain"] = timeit(lambda: sample(host_actor(obs)), args.iters)
    results["env_step"] = timeit(lambda: envs.step(action), args.iters)
    results["normalize"] = timeit(
        lambda: (rew_norm.normalize(rew, terms), obs_norm.normalize_step(raw, terms, truncs, infos)),
        args.iters)
    native = np.zeros((args.num_envs, act_dim), dtype=np.float32)
    counter = {"i": 0}

    def push():
        step = counter["i"] % args.num_steps
        counter["i"] += 1
        transfer.push(step, rew, terms, truncs, observations=obs, native_actions=native)

    results["transfer_push"] = timeit(push, args.iters)

    print(f"env={args.env_id} num_envs={args.num_envs} width={args.width} "
          f"blocks={args.n_blocks} env_threads={args.threads} "
          f"beta={'fused' if args.fused_beta else 'numpy'} "
          f"loadavg={Path('/proc/loadavg').read_text().split(' ')[0]}")
    print(f"  {'link':<18s} {'median_us':>10s} {'min_us':>9s} {'cpu_us':>9s}")
    for name, (med, best, cpu) in results.items():
        print(f"  {name:<18s} {med:10.1f} {best:9.1f} {cpu:9.1f}")

    policy = results["policy_chain"][0]
    env_step = results["env_step"][0]
    normalize = results["normalize"][0] + results["transfer_push"][0]
    serial = policy + env_step + normalize
    pipelined = max(policy + normalize, env_step)
    print(f"\n  serial per-step        {serial:8.1f}us  -> {args.num_envs / serial * 1e6:9.0f} SPS")
    print(f"  ideal 2-group overlap  {pipelined:8.1f}us  -> "
          f"{args.num_envs / pipelined * 1e6:9.0f} SPS  ({serial / pipelined:.2f}x)")
    print(f"  physics-only floor     {env_step:8.1f}us  -> "
          f"{args.num_envs / env_step * 1e6:9.0f} SPS  ({serial / env_step:.2f}x)")
    transfer.close()
    envs.close()


if __name__ == "__main__":
    main()

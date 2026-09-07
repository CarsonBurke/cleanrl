"""Seeded native-environment snapshot A/B; fixed component work, not training.

Run through mlq. The baseline directory must contain the pre-change
cleanrl/shared/mujoco_env.py and mujoco_batch.c; it is loaded under a private
module name, not patched into the candidate. JSON and TensorBoard go to runs/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.mujoco_env import make_mujoco_vector_env


def assert_equal(left, right, path="output"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            if key != "t":  # Episode wall time is deliberately not reproducible.
                assert_equal(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f"{path}/{index}")
    elif isinstance(left, np.ndarray):
        assert left.shape == right.shape and left.dtype == right.dtype, path
        if left.dtype == object:
            for index, (a, b) in enumerate(zip(left.flat, right.flat)):
                assert_equal(a, b, f"{path}/{index}")
        else:
            assert left.tobytes() == right.tobytes(), path
    else:
        assert left == right, path


def parity(reference, candidate, seed, steps):
    assert_equal(reference.reset(seed=seed), candidate.reset(seed=seed))
    rng = np.random.default_rng(seed)
    boundaries = 0
    for step in range(steps):
        action = rng.normal(size=reference.action_space.shape).astype(np.float32)
        left, right = reference.step(action), candidate.step(action)
        assert_equal(left, right, f"step/{step}")
        boundaries += int(np.count_nonzero(left[2] | left[3]))
        if step == 7 and reference.num_envs > 1:
            assert_equal(reference.envs[1].reset(), candidate.envs[1].reset())
    for left, right in zip(reference.envs, candidate.envs):
        assert_equal(left.unwrapped.np_random.bit_generator.state,
                     right.unwrapped.np_random.bit_generator.state, "reset_rng")
        for attribute in ("episode_returns", "episode_lengths", "episode_count",
                          "return_queue", "length_queue", "_elapsed_steps"):
            assert_equal(np.asarray(left.get_wrapper_attr(attribute)),
                         np.asarray(right.get_wrapper_attr(attribute)), attribute)
    return {"bitwise_parity": True, "boundary_count": boundaries, "steps": steps}


def measure(env, actions, seed, gap_us):
    env.reset(seed=seed)
    for action in actions[:200]:
        env.step(action)
    env.reset(seed=seed)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    for action in actions:
        env.step(action)
        if gap_us:
            deadline = time.perf_counter() + gap_us * 1e-6
            while time.perf_counter() < deadline:
                pass
    return {"wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--env-ids", nargs="+", default=["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--parity-steps", type=int, default=1005)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--gap-us", type=float, default=0)
    parser.add_argument("--copy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-name", default=f"env_execution__{time.time_ns()}")
    args = parser.parse_args()
    if args.steps < 1000 or args.parity_steps < 1005 or args.repeats < 1 or args.gap_us < 0:
        parser.error("require steps>=1000, parity-steps>=1005, repeats>=1, gap-us>=0")
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be a single directory name")
    baseline_path = args.baseline_root.resolve() / "cleanrl/shared/mujoco_env.py"
    spec = importlib.util.spec_from_file_location("cleanrl_env_execution_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"config": vars(args) | {"baseline_root": str(args.baseline_root.resolve()),
                                     "CLEANRL_ENV_SPIN": os.environ.get("CLEANRL_ENV_SPIN")},
              "environments": {}}
    writer = SummaryWriter(str(run_dir))
    try:
        for env_id in args.env_ids:
            kwargs = dict(num_threads=args.num_threads, copy=args.copy)
            reference = baseline.make_mujoco_vector_env(env_id, args.num_envs, **kwargs)
            candidate = make_mujoco_vector_env(env_id, args.num_envs, **kwargs)
            result = report["environments"][env_id] = {}
            try:
                result.update(parity(reference, candidate, args.seed, args.parity_steps))
                # Original Gym is an independent oracle, not just the snapshot.
                oracle = make_mujoco_vector_env(env_id, args.num_envs, backend="sync", copy=args.copy)
                try:
                    parity(oracle, candidate, args.seed, args.parity_steps)
                    result["gym_bitwise_parity"] = True
                finally:
                    oracle.close()
                actions = np.random.default_rng(args.seed).normal(
                    size=(args.steps,) + reference.action_space.shape).astype(np.float32)
                samples = result["samples"] = {"baseline": [], "candidate": []}
                paths = [("baseline", reference), ("candidate", candidate)]
                for repeat in range(args.repeats):
                    for name, env in paths[::1 if repeat % 2 == 0 else -1]:
                        sample = measure(env, actions, args.seed, args.gap_us)
                        samples[name].append(sample)
                        for metric, value in sample.items():
                            writer.add_scalar(f"{env_id}/{name}/{metric}", value, repeat)
                    writer.flush()
                for name, rows in samples.items():
                    result[name] = {metric: statistics.median(row[metric] for row in rows)
                                    for metric in ("wall_seconds", "cpu_seconds")}
                result["wall_speedup"] = result["baseline"]["wall_seconds"] / result["candidate"]["wall_seconds"]
                result["cpu_speedup"] = result["baseline"]["cpu_seconds"] / result["candidate"]["cpu_seconds"]
                writer.add_scalar(f"{env_id}/wall_speedup", result["wall_speedup"], 0)
                writer.add_scalar(f"{env_id}/bitwise_parity", 1, 0)
            finally:
                reference.close()
                candidate.close()
                (run_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        writer.close()
    print(f"RESULT {run_dir / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()

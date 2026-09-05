"""Fixed-work MuJoCo/v30 performance measurements, not a training evaluation.

Submit exclusively through mlq, for example:
  mlq submit --name mujoco_throughput --max-parallel-runs 1 --cwd "$PWD" \
    --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
    .venv/bin/python scripts/benchmark_mujoco_throughput.py --num-envs 16 64

Reports repeated wall-clock distributions, CUDA-stream elapsed time, compilation
startup separately, seeded environment parity, and Beta sample/RNG identity.
JSON and TensorBoard results live together under runs/. Closed-loop rates include
policy inference, sampling, synchronous action readback, physics, normalization,
and observation upload, but exclude learning updates and imply no learning score.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport
from cleanrl.shared.hl_gauss_kernels import moment_match_from_log_probs, project_moment_matched_fused
from cleanrl.shared.cuda_update import CudaGraphUpdate
from cleanrl.shared.mujoco_env import NativeMujocoVectorEnv, ThreadedMujocoVectorEnv, make_mujoco_vector_env
from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm
from cleanrl.shared.vmpo_temperature import solve_log_temperature, solve_log_temperature_reference
from cleanrl_utils.reference_loss import load_reference_loss

REFERENCE = (
    "cleanrl.vmpo.ppo_continuous_action_iterthink_v24_beta_vmpo_v30_"
    "dreamer_bucket_moment_hlgauss_reward_norm"
)

# Frozen v30's metric stack order. Unaffected metrics retain the stringent
# captured-reference tolerance; fused-operation exceptions follow the separate
# label/temperature tests rather than a loose tolerance for the entire vector.
UPDATE_METRIC_NAMES = (
    "policy_loss", "value_loss", "temperature_loss", "mean_kl", "concentration_kl",
    "full_kl", "ess_fraction", "topk_threshold", "advantage_mean", "advantage_std",
    "temperature_kl", "eta_stationarity", "perplexity_fraction", "max_weight", "ess",
    "mean_kl_residual", "concentration_kl_residual", "value_rmse", "explained_variance",
    "target_outside_support", "target_edge_mass", "prediction_edge_mass",
    "policy_concentration", "policy_variance", "eta",
)
FUSED_METRIC_TOLERANCES = {
    "policy_loss": (3e-6, 3e-6), "value_loss": (2e-6, 2e-6),
    "temperature_loss": (3e-6, 3e-6), "ess_fraction": (2e-5, 2e-5),
    "temperature_kl": (3e-6, 0.0), "eta_stationarity": (3e-6, 0.0),
    "perplexity_fraction": (3e-6, 3e-6), "max_weight": (2e-7, 8e-5),
    "ess": (2e-5, 2e-5), "target_edge_mass": (3e-6, 5e-5),
    # exp(log_eta) amplifies the standalone solver's 2e-4 absolute log tolerance.
    "eta": (3e-6, 5e-4),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--backends", nargs="+", choices=["sync", "threaded", "native"], default=["sync", "native"])
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--thread-counts", nargs="+", type=int, help="Also sweep these native CPU thread counts with fixed-work raw/vector-normalized/legacy timing and parity; GPU compilation still runs once per N.")
    parser.add_argument("--env-steps", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--gpu-iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--exp-name", default="mujoco_throughput_v1")
    parser.add_argument("--env-only", action="store_true", help="Measure CPU physics/normalization without creating a model; still submit through mlq.")
    parser.add_argument("--profile", action="store_true", help="Also export 32-step CPU/CUDA closed-loop traces after timed repetitions.")
    parser.add_argument("--profile-update", action="store_true", help="Compile and compare the exact frozen v30 loss/backward/Adam with whole-update capture on fixed synthetic batches.")
    parser.add_argument("--fused-updates", action="store_true", help="Also evaluate experimental fused projection/temperature updates. These failed the initial optimizer parity gate and are not available in the training proxy.")
    parser.add_argument("--projection-only", action="store_true", help="Only benchmark/check the compiled frozen and opt-in fused value projector; still CUDA and mlq only.")
    parser.add_argument("--require-numerical-parity", action="store_true", help="Fail after saving evidence if documented projection/temperature/first-update tolerances or finite-value checks fail. This is a numerical prerequisite, not proof of equal learning scores.")
    parser.add_argument("--legacy-wrappers", action=argparse.BooleanOptionalAction, default=True, help="Also compare the original per-environment normalization wrapper stack.")
    args = parser.parse_args()
    for key in ("num_threads", "env_steps", "warmup_steps", "gpu_iterations", "repeats"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if any(n <= 0 for n in args.num_envs):
        parser.error("--num-envs entries must be positive")
    if args.thread_counts is not None:
        if any(n <= 0 for n in args.thread_counts):
            parser.error("--thread-counts entries must be positive")
        args.thread_counts = list(dict.fromkeys(args.thread_counts))
    if args.projection_only and args.env_only:
        parser.error("--projection-only requires CUDA; it cannot be combined with --env-only")
    if args.projection_only and args.thread_counts:
        parser.error("--thread-counts measures environments and cannot be combined with --projection-only")
    if args.fused_updates and (not args.profile_update or args.env_only or args.projection_only):
        parser.error("--fused-updates needs CUDA --profile-update and cannot be combined with --env-only or --projection-only")
    if args.require_numerical_parity and not (args.projection_only or (args.profile_update and not args.env_only)):
        parser.error("--require-numerical-parity needs --projection-only or CUDA --profile-update so candidate comparisons are actually measured")
    return args


def timing_summary(samples, units):
    median = statistics.median(samples)
    return {
        "seconds": samples,
        "median_seconds": median,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "units_per_second": units / median,
        "microseconds_per_unit": median * 1e6 / units,
        "units_per_repeat": units,
    }


def measure_cpu(fn, iterations, repeats, *, reset=None, units_per_call=1):
    samples = []
    for _ in range(repeats):
        if reset is not None:
            reset()
        started = time.perf_counter()
        for i in range(iterations):
            fn(i)
        samples.append(time.perf_counter() - started)
    return timing_summary(samples, iterations * units_per_call)


def measure_cuda(fn, iterations, repeats):
    """Synchronize at repetition boundaries; never add artificial per-call syncs."""
    for _ in range(10):
        torch.compiler.cudagraph_mark_step_begin()
        fn()
    torch.cuda.synchronize()
    wall, stream = [], []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        start.record()
        for _ in range(iterations):
            torch.compiler.cudagraph_mark_step_begin()
            fn()
        end.record()
        end.synchronize()
        wall.append(time.perf_counter() - started)
        stream.append(start.elapsed_time(end) / 1000)
    result = timing_summary(wall, iterations)
    result["cuda_stream_seconds"] = stream
    result["cuda_stream_microseconds_per_call"] = statistics.median(stream) * 1e6 / iterations
    return result


def make_env(args, n, backend, *, legacy=False):
    if legacy:
        baseline = importlib.import_module("cleanrl.ppo_continuous_action")
        factories = [baseline.make_env(args.env_id, i, False, "benchmark", 0.99) for i in range(n)]
        if backend == "sync":
            return gym.vector.SyncVectorEnv(factories)
        if backend == "threaded":
            return ThreadedMujocoVectorEnv(factories, num_threads=args.num_threads)
        return NativeMujocoVectorEnv(factories, env_id=args.env_id, num_threads=args.num_threads)
    return make_mujoco_vector_env(
        args.env_id, n, backend=backend, num_threads=args.num_threads,
    )


def compare_tree(reference, candidate, path="infos"):
    """Info parity includes final observations and episode stats, except wall time."""
    if isinstance(reference, dict):
        if set(reference) != set(candidate):
            raise AssertionError(f"{path}: different keys {set(reference) ^ set(candidate)}")
        for key in reference:
            if key == "t" and path.endswith("episode"):
                continue
            compare_tree(reference[key], candidate[key], f"{path}/{key}")
    elif reference is None:
        if candidate is not None:
            raise AssertionError(f"{path}: expected None")
    elif isinstance(reference, np.ndarray) and reference.dtype == object:
        if reference.shape != candidate.shape:
            raise AssertionError(f"{path}: shape mismatch")
        for i, (left, right) in enumerate(zip(reference.flat, candidate.flat)):
            compare_tree(left, right, f"{path}/{i}")
    else:
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise AssertionError(f"{path}: non-finite values")
        np.testing.assert_allclose(candidate, reference, rtol=1e-10, atol=1e-10, err_msg=path)


def environment_parity(args, n, backend, *, legacy=False):
    reference = make_env(args, n, "sync", legacy=legacy)
    candidate = None
    try:
        candidate = make_env(args, n, backend, legacy=legacy)
        obs_ref, info_ref = reference.reset(seed=args.seed)
        obs_fast, info_fast = candidate.reset(seed=args.seed)
        if not np.isfinite(obs_ref).all() or not np.isfinite(obs_fast).all():
            raise AssertionError("reset observations must be finite")
        np.testing.assert_array_equal(obs_ref, obs_fast)
        compare_tree(info_ref, info_fast)
        norms = [VectorObsNorm(n, reference.single_observation_space.shape) for _ in range(2)]
        rewards = [VectorRewardNorm(n, gamma=0.99) for _ in range(2)]
        for norm, obs in zip(norms, (obs_ref, obs_fast)):
            norm.normalize(obs)
        rng = np.random.default_rng(args.seed)
        steps = int(gym.spec(args.env_id).max_episode_steps) + 5
        observation_error = reward_error = 0.0
        terminations = truncations = 0
        for step in range(steps):
            # Exercise action clipping as well as same-step autoresets.
            action = rng.uniform(-2, 2, size=(n,) + reference.single_action_space.shape).astype(np.float32)
            left, right = reference.step(action), candidate.step(action)
            for label, a, b in zip(("obs", "reward", "terminated", "truncated"), left[:4], right[:4]):
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise AssertionError(f"{backend} step {step} {label}: non-finite values")
                np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-10, err_msg=f"{backend} step {step} {label}")
            compare_tree(left[4], right[4])
            observation_error = max(observation_error, float(np.max(np.abs(left[0] - right[0]))))
            reward_error = max(reward_error, float(np.max(np.abs(left[1] - right[1]))))
            terminations += int(np.sum(left[2]))
            truncations += int(np.sum(left[3]))
            if not legacy:
                normalized = [norm.normalize_step(data[0], data[2], data[3], data[4]) for norm, data in zip(norms, (left, right))]
                for a, b in zip(*normalized):
                    if not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise AssertionError("normalized observations must be finite")
                    np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-6)
                normalized_rewards = [norm.normalize(data[1], data[2]) for norm, data in zip(rewards, (left, right))]
                if not all(np.isfinite(reward).all() for reward in normalized_rewards):
                    raise AssertionError("normalized rewards must be finite")
                np.testing.assert_allclose(*normalized_rewards, rtol=1e-6, atol=1e-6)
        return {"passed": True, "steps": steps, "terminated_transitions": terminations,
                "truncated_transitions": truncations, "max_observation_abs_error": observation_error,
                "max_reward_abs_error": reward_error, "info_wall_time_excluded": True}
    finally:
        reference.close()
        if candidate is not None:
            candidate.close()


def benchmark_environment(args, n, backend, *, legacy=False):
    env = make_env(args, n, backend, legacy=legacy)
    try:
        rng = np.random.default_rng(args.seed)
        actions = rng.uniform(-1, 1, size=(args.env_steps + args.warmup_steps, n) + env.single_action_space.shape).astype(np.float32)
        env.reset(seed=args.seed)
        for action in actions[:args.warmup_steps]:
            env.step(action)
        result = {"raw": measure_cpu(
            lambda i: env.step(actions[i]), args.env_steps, args.repeats,
            reset=lambda: env.reset(seed=args.seed), units_per_call=n,
        )}
        if legacy:
            return {"wrapped": result["raw"]}
        obs_norm = reward_norm = None

        def reset():
            nonlocal obs_norm, reward_norm
            obs_norm = VectorObsNorm(n, env.single_observation_space.shape)
            reward_norm = VectorRewardNorm(n, gamma=0.99)
            obs_norm.normalize(env.reset(seed=args.seed)[0])

        def normalized_step(i):
            obs, reward, terms, truncs, infos = env.step(actions[i])
            reward_norm.normalize(reward, terms)
            obs_norm.normalize_step(obs, terms, truncs, infos)

        result["normalized"] = measure_cpu(
            normalized_step, args.env_steps, args.repeats, reset=reset, units_per_call=n,
        )
        reset()
        observations = rng.standard_normal((args.env_steps, n) + env.single_observation_space.shape)
        rewards = rng.standard_normal((args.env_steps, n))
        zeros = np.zeros(n, dtype=bool)

        def normalization_only(i):
            obs_norm.normalize_step(observations[i], zeros, zeros, {})
            reward_norm.normalize(rewards[i], zeros)

        result["normalization_only_no_boundaries"] = measure_cpu(
            normalization_only, args.env_steps, args.repeats, reset=reset, units_per_call=n,
        )
        return result
    finally:
        env.close()


def tensor_difference(reference, candidate, *, atol=None, rtol=None):
    difference = (reference - candidate).abs()
    reference_finite = bool(reference.isfinite().all().cpu())
    candidate_finite = bool(candidate.isfinite().all().cpu())
    result = {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_absolute": float(difference.max().cpu()),
        "mean_absolute": float(difference.mean().cpu()),
        "max_relative_denominator_floor_1e_6": float((difference / reference.abs().clamp_min(1e-6)).max().cpu()),
        "reference_finite": reference_finite,
        "candidate_finite": candidate_finite,
    }
    if atol is not None or rtol is not None:
        if atol is None or rtol is None:
            raise ValueError("provide both absolute and relative tolerances")
        result["tolerance"] = {"atol": atol, "rtol": rtol}
        result["checks"] = {
            "reference_finite": reference_finite,
            "candidate_finite": candidate_finite,
            "within_tolerance": reference_finite and candidate_finite and bool(
                torch.allclose(reference, candidate, atol=atol, rtol=rtol)
            ),
        }
    return result


def update_metric_comparisons(reference, candidate, *, fused):
    if reference.numel() != len(UPDATE_METRIC_NAMES) or candidate.shape != reference.shape:
        raise ValueError("frozen v30 metric layout changed; update the numerical gate explicitly")
    return {
        name: tensor_difference(
            reference[index], candidate[index],
            atol=FUSED_METRIC_TOLERANCES.get(name, (1e-6, 1e-6))[0] if fused else 1e-6,
            rtol=FUSED_METRIC_TOLERANCES.get(name, (1e-6, 1e-6))[1] if fused else 1e-6,
        )
        for index, name in enumerate(UPDATE_METRIC_NAMES)
    }


def numerical_checks(value, prefix="measurements"):
    """Yield explicit adoption checks; error-only diagnostics never set gates."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "checks":
                for name, passed in child.items():
                    yield f"{prefix}/{name}", passed is True
            else:
                yield from numerical_checks(child, f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from numerical_checks(child, f"{prefix}/{index}")


def numerical_failures(value, prefix="measurements"):
    return [path for path, passed in numerical_checks(value, prefix) if not passed]


def enforce_numerical_parity(args, report, save):
    checks = list(numerical_checks(report["measurements"]))
    failures = [path for path, passed in checks if not passed]
    if args.require_numerical_parity and not checks:
        failures.append("measurements/no_candidate_checks")
    report["numerical_parity"] = {
        "required": args.require_numerical_parity, "check_count": len(checks),
        "passed": not failures if checks or args.require_numerical_parity else None,
        "failures": failures,
    }
    # Save the complete measured errors, thresholds and failed names before
    # raising, so this job can safely gate an --after-success experiment chain.
    save()
    if args.require_numerical_parity and failures:
        raise AssertionError(f"Numerical parity gate failed ({len(failures)} checks): " + "; ".join(failures[:8]))


def json_compatible(value):
    if isinstance(value, dict):
        return {key: json_compatible(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_compatible(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def benchmark_thread_sweep(args, n, result, writer, prefix, save):
    """Measure CPU choices with identical seed/action work, never tune the learner."""
    sweep = result["thread_sweep"] = {
        "backend": "native",
        "workload": "Identical pre-generated actions and reset seeds at every thread count; full-horizon parity before timing.",
        "selection_rule": "Lowest repeated median wall time for each path; smaller thread count breaks exact ties. Training arguments are unchanged.",
        "by_num_threads": {},
        "selected_num_threads": {},
    }
    for threads in args.thread_counts:
        selected = copy.copy(args)
        selected.num_threads = threads
        print(f"N={n}: native thread sweep threads={threads}", flush=True)
        if threads == args.num_threads and "native" in result["environments"]:
            measured = dict(result["environments"]["native"])
        else:
            parity = environment_parity(selected, n, "native")
            measured = {"parity": parity, **benchmark_environment(selected, n, "native")}
        if args.legacy_wrappers:
            if threads == args.num_threads and "native" in result.get("legacy_environments", {}):
                measured["legacy"] = result["legacy_environments"]["native"]
            else:
                parity = environment_parity(selected, n, "native", legacy=True)
                measured["legacy"] = {"parity": parity, **benchmark_environment(selected, n, "native", legacy=True)}
        sweep["by_num_threads"][str(threads)] = measured
        write_scalars(writer, measured, f"{prefix}/thread_sweep/threads{threads}")
        writer.flush()
        save()

    paths = {"raw": lambda measured: measured["raw"],
             "vector_normalized": lambda measured: measured["normalized"]}
    if args.legacy_wrappers:
        paths["legacy_normalized"] = lambda measured: measured["legacy"]["wrapped"]
    for path, extract in paths.items():
        winner = min(args.thread_counts, key=lambda threads: (
            extract(sweep["by_num_threads"][str(threads)])["median_seconds"], threads,
        ))
        timing = extract(sweep["by_num_threads"][str(winner)])
        sweep["selected_num_threads"][path] = winner
        writer.add_scalar(f"{prefix}/thread_sweep/selected/{path}", winner, 0)
        print(f"N={n}: measured fastest {path} threads={winner}, {timing['units_per_second']:.0f} transitions/s", flush=True)
    writer.flush()
    save()


def benchmark_rollout_transfers(args, n, obs_shape, num_steps):
    """Equivalent v30 field movement, including its float64-to-float32 casts."""
    rng = np.random.default_rng(args.seed)
    shape = (num_steps, n)
    observations = rng.standard_normal(shape + obs_shape)
    rewards = rng.standard_normal(shape)
    terms = rng.random(shape) < 0.01
    truncs = rng.random(shape) < 0.02
    boundaries = terms | truncs
    device_observations = torch.empty(shape + obs_shape, device="cuda")
    device_transitions = torch.empty_like(device_observations)
    device_rewards = torch.empty(shape, device="cuda")
    device_terms = torch.empty_like(device_rewards)
    device_boundaries = torch.empty(shape, dtype=torch.bool, device="cuda")
    staging = RolloutTransfer(num_steps, n, obs_shape, "cuda", store_transition_observations=True)

    def immediate():
        for t in range(num_steps):
            device_rewards[t].copy_(torch.as_tensor(rewards[t], device="cuda", dtype=torch.float32))
            device_terms[t].copy_(torch.as_tensor(terms[t], device="cuda", dtype=torch.float32))
            device_boundaries[t].copy_(torch.as_tensor(boundaries[t], device="cuda", dtype=torch.bool))
            device_transitions[t].copy_(torch.as_tensor(observations[t], device="cuda", dtype=torch.float32))
            device_observations[t].copy_(torch.as_tensor(observations[t], device="cuda", dtype=torch.float32))

    def packed():
        for t in range(num_steps):
            staging.push(t, rewards[t], terms[t], truncs[t], observations[t])
            device_observations[t].copy_(staging.observation(observations[t]))
        return staging.upload()

    immediate()
    batch = packed()
    parity = {
        "rewards_equal": torch.equal(device_rewards, batch.rewards),
        "terminations_equal": torch.equal(device_terms, batch.terminations),
        "boundaries_equal": torch.equal(device_boundaries, (batch.terminations.bool() | batch.truncations.bool())),
        "transition_observations_equal": torch.equal(device_transitions, batch.transition_observations),
    }
    if not all(parity.values()):
        raise AssertionError(f"Packed rollout transfers changed data: {parity}")
    return {
        "num_steps_per_call": num_steps,
        "parity": parity,
        "v30_immediate": measure_cuda(immediate, args.gpu_iterations, args.repeats),
        "shared_packed": measure_cuda(packed, args.gpu_iterations, args.repeats),
    }


def benchmark_projection(args, support, targets, result):
    """Compare whole projection and isolated fused bisection, never imply parity."""
    projector = torch.compile(support.project_moment_matched, mode=args.compile_mode, dynamic=False, fullgraph=True)
    fused = torch.compile(lambda t: project_moment_matched_fused(support, t), mode=args.compile_mode, dynamic=False, fullgraph=True)
    kernel = torch.compile(lambda p, t: moment_match_from_log_probs(p, t, support.support), mode=args.compile_mode, dynamic=False, fullgraph=True)
    preprocessing = torch.compile(support.project_log_probs, mode=args.compile_mode, dynamic=False, fullgraph=True)
    result["kernel_sha256"] = hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("cleanrl/shared/hl_gauss_kernels.py").read_bytes()).hexdigest()
    result["numerical_contract"] = "Same32-step bisection, cutoff30, tilt bound1, FP32, symmetric paired expectation; libdevice exp, rounded division, contraction disabled. Numerical parity is measured, not presumed."
    result["startup_seconds"] = {}
    with torch.no_grad():
        for label, fn in (("frozen", projector), ("fused", fused), ("log_mass_preprocessing", preprocessing)):
            print(f"B={targets.numel()}: compiling {label} value projection", flush=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            torch.compiler.cudagraph_mark_step_begin()
            fn(targets)
            torch.cuda.synchronize()
            result["startup_seconds"][label] = time.perf_counter() - started
            result[label] = measure_cuda(lambda: fn(targets), args.gpu_iterations, args.repeats)
        torch.compiler.cudagraph_mark_step_begin()
        log_probs = preprocessing(targets).clone()
        torch.cuda.synchronize()
        started = time.perf_counter()
        torch.compiler.cudagraph_mark_step_begin()
        kernel(log_probs, targets)
        torch.cuda.synchronize()
        result["startup_seconds"]["bisection_only"] = time.perf_counter() - started
        result["bisection_only"] = measure_cuda(lambda: kernel(log_probs, targets), args.gpu_iterations, args.repeats)

        coordinate = torch.linspace(-1, 1, targets.numel(), device="cuda").reshape(targets.shape)
        suites = {
            "normalized_return_scale": targets,
            "linear_support_and_endpoints": coordinate * support.support[-1] * 1.1,
            "logarithmic_support": coordinate.sign() * (coordinate.abs() * support.coord_max).expm1(),
        }
        result["parity"] = {}
        for label, values in suites.items():
            torch.compiler.cudagraph_mark_step_begin()
            expected = projector(values).clone()
            torch.compiler.cudagraph_mark_step_begin()
            actual = fused(values).clone()
            expected_mean = support.probs_to_scalar(expected)
            actual_mean = support.probs_to_scalar(actual)
            kl = (expected * (expected.clamp_min(1e-30).log() - actual.clamp_min(1e-30).log())).sum(-1)
            max_absolute_kl = float(kl.abs().max().cpu())
            sum_error = float((actual.sum(-1) - 1).abs().max().cpu())
            comparison = {
                "probabilities": tensor_difference(expected, actual, atol=3e-6, rtol=5e-5),
                "decoded_means": tensor_difference(expected_mean, actual_mean, atol=1e-2, rtol=3e-6),
                "max_kl_reference_to_fused": float(kl.max().cpu()),
                "max_absolute_kl_reference_to_fused": max_absolute_kl,
                "max_probability_sum_error": sum_error,
                "checks": {
                    "probability_sum_error_within_3e_7": math.isfinite(sum_error) and sum_error <= 3e-7,
                    "absolute_kl_within_2e_6": math.isfinite(max_absolute_kl) and max_absolute_kl <= 2e-6,
                    "probabilities_nonnegative": bool((actual >= 0).all().cpu()),
                },
            }
            with torch.enable_grad():
                logits = torch.randn_like(expected, requires_grad=True)
                expected_loss = -(expected * logits.log_softmax(-1)).sum(-1).mean()
                actual_loss = -(actual * logits.log_softmax(-1)).sum(-1).mean()
                expected_grad, = torch.autograd.grad(expected_loss, logits)
                actual_grad, = torch.autograd.grad(actual_loss, logits)
            comparison["cross_entropy_logit_gradients"] = tensor_difference(expected_grad, actual_grad, atol=3e-8, rtol=5e-5)
            comparison["cross_entropy_loss"] = tensor_difference(expected_loss, actual_loss, atol=2e-6, rtol=2e-6)
            result["parity"][label] = comparison
    result["whole_projection_speedup"] = result["frozen"]["median_seconds"] / result["fused"]["median_seconds"]


def benchmark_temperature(args, advantages, result):
    """Measure the global eta solver separately from labels and neural networks."""
    def inputs_for(values):
        threshold = torch.sort(values).values[-int(values.numel() * 0.5)]
        selected = values >= threshold
        maximum = values.max()
        return (values - maximum, selected, selected.sum().float().log(),
                torch.full_like(threshold, float(np.log(1e-8))),
                ((maximum - threshold) / 0.01).clamp_min(1e-8).log(), 0.01)

    def decoded(inputs, log_eta):
        centered, selected, log_count = inputs[:3]
        logits = torch.where(selected, centered / log_eta.exp(), -torch.inf)
        log_weights = logits - torch.logsumexp(logits, dim=0)
        weights = log_weights.exp()
        kl = (weights * (torch.where(selected, log_weights, 0.0) + log_count)).sum()
        return weights, kl, weights.square().sum().reciprocal()

    frozen = torch.compile(solve_log_temperature_reference, mode=args.compile_mode, dynamic=False, fullgraph=True)
    fused = torch.compile(solve_log_temperature, mode=args.compile_mode, dynamic=False, fullgraph=True)
    result["kernel_sha256"] = hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("cleanrl/shared/vmpo_temperature.py").read_bytes()).hexdigest()
    result["startup_seconds"] = {}
    with torch.no_grad():
        inputs = inputs_for(advantages)
        for label, fn in (("frozen", frozen), ("fused", fused)):
            print(f"B={advantages.numel()}: compiling {label} temperature solver", flush=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            torch.compiler.cudagraph_mark_step_begin()
            fn(*inputs)
            torch.cuda.synchronize()
            result["startup_seconds"][label] = time.perf_counter() - started
            result[label] = measure_cuda(lambda: fn(*inputs), args.gpu_iterations, args.repeats)
        ties = advantages.clone()
        ties[::5] = advantages[1]
        suites = {"normal": advantages, "ties": ties, "flat": torch.zeros_like(advantages),
                  "small_scale": advantages * 1e-5, "large_scale": advantages * 1e6}
        result["parity"] = {}
        for label, values in suites.items():
            inputs = inputs_for(values)
            torch.compiler.cudagraph_mark_step_begin()
            expected_eta = frozen(*inputs).clone()
            torch.compiler.cudagraph_mark_step_begin()
            actual_eta = fused(*inputs).clone()
            expected_weights, expected_kl, expected_ess = decoded(inputs, expected_eta)
            actual_weights, actual_kl, actual_ess = decoded(inputs, actual_eta)
            result["parity"][label] = {
                "log_eta": tensor_difference(expected_eta, actual_eta, atol=2e-4, rtol=3e-5),
                "weights": tensor_difference(expected_weights, actual_weights, atol=2e-7, rtol=8e-5),
                "kl": tensor_difference(expected_kl, actual_kl, atol=3e-6, rtol=0),
                "ess": tensor_difference(expected_ess, actual_ess, atol=2e-5, rtol=2e-5),
                "checks": {"fused_kl_feasible_with_3e_6_tolerance": bool((actual_kl.isfinite() & (actual_kl <= inputs[-1] + 3e-6)).cpu())},
            }
    result["solver_speedup"] = result["frozen"]["median_seconds"] / result["fused"]["median_seconds"]


def benchmark_update(args, reference, model_args, agent, support, obs_shape, action_shape, result):
    """Extract the exact frozen loss without running its training __main__."""
    batch_size = model_args.batch_size
    observations = torch.randn((batch_size,) + obs_shape, device="cuda")
    old_alpha = torch.full((batch_size,) + action_shape, 1.7, device="cuda")
    old_beta = torch.full_like(old_alpha, 1.7)
    native_actions = Beta(old_alpha, old_beta, validate_args=False).sample()
    advantages = torch.randn(batch_size, device="cuda")
    targets = torch.randn(batch_size, device="cuda") * 20
    inputs = (observations, native_actions, old_alpha, old_beta, advantages, targets)
    result["workload"] = "Fixed synthetic batch; exact v30 forward/backward/fused Adam repeatedly updates a disposable model. This is not a training-score evaluation."
    result["variants"] = {}
    first_reference_parameters = first_reference_loss = first_reference_metrics = None
    fused_support = copy.copy(support)
    fused_support.project_moment_matched = functools.partial(project_moment_matched_fused, support)

    variants = [
        ("compiled_loss", support, False, False),
        ("captured_update", support, True, False),
    ]
    if args.fused_updates:
        variants.extend([
            ("fused_projection_compiled_loss", fused_support, False, False),
            ("fused_projection_temperature_captured_update", fused_support, True, True),
        ])
    for label, selected_support, capture, fused_temperature in variants:
        # Every variant gets the same untouched initialization and exact inputs.
        # Warmup updates from timing are subsequently identical in number too.
        update_agent = copy.deepcopy(agent)
        duals = torch.nn.Parameter(torch.tensor([model_args.initial_alpha_mean, model_args.initial_alpha_concentration], device="cuda"))
        parameters = [*update_agent.parameters(), duals]
        initial_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
        raw_loss, source_hash = load_reference_loss(reference, dict(
            args=model_args, agent=update_agent, duals=duals,
            hl_support=selected_support, autocast_dtype=torch.bfloat16,
        ), fused_temperature=fused_temperature)
        result["reference_sha256"] = source_hash
        optimizer = torch.optim.Adam(parameters, lr=model_args.learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=True)
        fused = selected_support is fused_support
        measured = result["variants"][label] = {
            "fused_projection": fused, "fused_temperature": fused_temperature,
            "checks": {},
        }

        def post_step():
            duals.clamp_(min=reference.DUAL_FLOOR)

        with torch.enable_grad():
            print(f"N={model_args.num_envs}: compiling {label} exact v30 update", flush=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            if capture:
                graph = CudaGraphUpdate(raw_loss, optimizer, inputs, modules=(update_agent,), post_step=post_step)
                after_capture = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
                measured["capture_preserves_initial_parameters_bitwise"] = torch.equal(initial_parameters, after_capture)
                measured["checks"]["capture_preserves_initial_parameters_bitwise"] = measured["capture_preserves_initial_parameters_bitwise"]
                if not measured["capture_preserves_initial_parameters_bitwise"]:
                    raise AssertionError("CUDA capture warmup changed the learner initialization")

                def update():
                    return graph(*inputs)
            else:
                loss_fn = torch.compile(raw_loss, mode=args.compile_mode, dynamic=False, fullgraph=True)

                def update():
                    loss, metrics = loss_fn(*inputs)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        post_step()
                    return loss.detach(), metrics.detach()

            torch.compiler.cudagraph_mark_step_begin()
            first_loss, first_metrics = update()
            torch.cuda.synchronize()
            measured["first_compiled_update_seconds"] = time.perf_counter() - started
            after_first = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
            measured["checks"].update({
                "first_update_parameters_finite": bool(after_first.isfinite().all().cpu()),
                "first_update_loss_finite": bool(first_loss.isfinite().all().cpu()),
                "first_update_metrics_finite": bool(first_metrics.isfinite().all().cpu()),
            })
            if first_reference_parameters is None:
                first_reference_parameters = after_first.clone()
                first_reference_loss = first_loss.clone()
                first_reference_metrics = first_metrics.clone()
            else:
                measured["first_update_parity"] = {
                    "parameters_and_duals": tensor_difference(first_reference_parameters, after_first, atol=2e-6 if fused else 1e-7, rtol=2e-5 if fused else 1e-6),
                    "loss": tensor_difference(first_reference_loss, first_loss, atol=3e-6 if fused else 1e-6, rtol=3e-6 if fused else 1e-6),
                    "metrics": tensor_difference(first_reference_metrics, first_metrics),
                    "per_metric": update_metric_comparisons(first_reference_metrics, first_metrics, fused=fused),
                }
            measured["forward_backward_adam"] = measure_cuda(update, args.gpu_iterations, args.repeats)
            measured["finite_parameters_after_measurement"] = bool(torch.stack([p.isfinite().all() for p in parameters]).all().cpu())
            measured["checks"]["parameters_finite_after_measurement"] = measured["finite_parameters_after_measurement"]
            if not measured["finite_parameters_after_measurement"]:
                raise FloatingPointError(f"Synthetic update benchmark {label} produced non-finite parameters")
        del update, optimizer, update_agent, parameters, raw_loss
        if capture:
            del graph
        else:
            del loss_fn

    baseline_seconds = result["variants"]["compiled_loss"]["forward_backward_adam"]["median_seconds"]
    for measured in result["variants"].values():
        measured["speedup_vs_compiled_loss"] = baseline_seconds / measured["forward_backward_adam"]["median_seconds"]
    if args.fused_updates:
        result["projection"] = {}
        benchmark_projection(args, support, targets, result["projection"])
        result["temperature"] = {}
        benchmark_temperature(args, advantages, result["temperature"])


def benchmark_gpu(args, n, writer, prefix, result, run_dir):
    reference = importlib.import_module(REFERENCE)
    env = make_env(args, n, "sync")
    try:
        model_args = reference.Args(num_envs=n)
        model_args.batch_size = model_args.num_steps * n
        model_args.topk_size = int(model_args.batch_size * model_args.topk_fraction)
        torch.manual_seed(args.seed)
        agent = reference.Agent(env, model_args).cuda().eval()
        target = copy.deepcopy(agent).requires_grad_(False)
        limit = float(np.log1p(model_args.value_support_limit))
        support = Dreamer3BucketHLGaussSupport(model_args.num_value_bins, -limit, limit, model_args.value_sigma_to_bin_ratio, torch.device("cuda"))
        obs_shape = env.single_observation_space.shape
        obs_cpu = np.random.default_rng(args.seed).standard_normal((n,) + obs_shape).astype(np.float32)
        obs = torch.as_tensor(obs_cpu, device="cuda")

        def rollout(x):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                alpha, beta = target.policy(x)
                logits = agent.value_logits(x)
            return alpha.float(), beta.float(), support.to_scalar(logits.float())

        def values(x):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = agent.value_logits(x)
            return support.to_scalar(logits.float())

        compiled_rollout = torch.compile(rollout, mode=args.compile_mode)
        compiled_values = torch.compile(values, mode=args.compile_mode)
        result["startup_seconds"] = {}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            alpha, beta, _ = compiled_rollout(obs)
            alpha, beta = alpha.clone(), beta.clone()
        torch.cuda.synchronize()
        result["startup_seconds"]["first_compiled_rollout"] = time.perf_counter() - started
        with torch.no_grad():
            result["compiled_rollout"] = measure_cuda(lambda: compiled_rollout(obs), args.gpu_iterations, args.repeats)
            result["compiled_rollout_with_readback"] = measure_cuda(
                lambda: compiled_rollout(obs)[0].cpu().numpy(), args.gpu_iterations, args.repeats,
            )

            # Both paths must consume exactly the same CUDA RNG stream.
            state = torch.cuda.get_rng_state()
            checked = Beta(alpha, beta, validate_args=True).sample()
            state_checked = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(state)
            unchecked = Beta(alpha, beta, validate_args=False).sample()
            state_unchecked = torch.cuda.get_rng_state()
            result["beta_parity"] = {"samples_bitwise_equal": torch.equal(checked, unchecked), "rng_state_equal": torch.equal(state_checked, state_unchecked)}
            torch.cuda.set_rng_state(state)
            shared_native, shared_action = sample_beta_actions(alpha, beta, agent.action_low, agent.action_high)
            checked_native = checked.clamp(reference.SAMPLE_EPS, 1 - reference.SAMPLE_EPS)
            checked_action = agent.action_low + checked_native * (agent.action_high - agent.action_low)
            result["beta_parity"].update({
                "shared_samples_bitwise_equal": torch.equal(checked_native, shared_native),
                "shared_actions_bitwise_equal": torch.equal(checked_action, shared_action),
                "shared_rng_state_equal": torch.equal(state_checked, torch.cuda.get_rng_state()),
            })
            if not all(result["beta_parity"].values()):
                raise AssertionError("Disabling Beta validation changed samples or RNG state")
            for validate in (True, False):
                label = "checked" if validate else "unchecked"
                result[f"beta_{label}"] = measure_cuda(
                    lambda: Beta(alpha, beta, validate_args=validate).sample(), args.gpu_iterations, args.repeats,
                )
            result["shared_sample_beta_actions"] = measure_cuda(
                lambda: sample_beta_actions(alpha, beta, agent.action_low, agent.action_high), args.gpu_iterations, args.repeats,
            )
            result["observation_h2d_allocate"] = measure_cuda(
                lambda: torch.as_tensor(obs_cpu, device="cuda"), args.gpu_iterations, args.repeats,
            )
            result["observation_h2d_reuse"] = measure_cuda(
                lambda: obs.copy_(torch.from_numpy(obs_cpu)), args.gpu_iterations, args.repeats,
            )
            actions = torch.zeros((n,) + env.single_action_space.shape, device="cuda")
            result["action_d2h_allocate"] = measure_cuda(lambda: actions.cpu().numpy(), args.gpu_iterations, args.repeats)
            result["rollout_transfers"] = benchmark_rollout_transfers(args, n, obs_shape, model_args.num_steps)

            trajectory = torch.randn((model_args.num_steps + 1, n) + obs_shape, device="cuda")
            transitions = trajectory[1:].reshape((-1,) + obs_shape)
            started = time.perf_counter()
            torch.compiler.cudagraph_mark_step_begin()
            compiled_values(transitions)
            torch.cuda.synchronize()
            result["startup_seconds"]["first_batched_critic"] = time.perf_counter() - started
            result["compiled_batched_next_critic"] = measure_cuda(
                lambda: compiled_values(transitions), args.gpu_iterations, args.repeats,
            )
            result["critic_batch_parity"] = {}
            for nonzero in (False, True):
                if nonzero:
                    torch.nn.init.normal_(agent.value_head.weight, std=0.03)
                torch.compiler.cudagraph_mark_step_begin()
                batched_values = compiled_values(transitions).clone().reshape(model_args.num_steps, n)
                step_values = torch.empty_like(batched_values)
                for t in range(model_args.num_steps):
                    torch.compiler.cudagraph_mark_step_begin()
                    step_values[t].copy_(compiled_rollout(trajectory[t + 1])[2])
                comparison = tensor_difference(batched_values, step_values)
                # No boundaries: this isolates batch-layout numerical differences.
                # Boundary correctness itself is covered by test_ppo_loop.py.
                reward = torch.randn_like(step_values)
                zeros = torch.zeros_like(step_values)
                gamma, lam = model_args.gamma, model_args.gae_lambda
                # Shared GAE gets actual next-state cached values. Its current
                # values are those same rollout values shifted by one step.
                torch.compiler.cudagraph_mark_step_begin()
                initial_value = compiled_rollout(trajectory[0])[2].clone()
                cached_current = torch.cat((initial_value.unsqueeze(0), step_values[:-1]), dim=0)
                baseline_advantage = torch.zeros_like(reward)
                running = torch.zeros(n, device="cuda")
                for t in range(model_args.num_steps - 1, -1, -1):
                    running = reward[t] + gamma * batched_values[t] - cached_current[t] + gamma * lam * running
                    baseline_advantage[t] = running
                shared_gae = get_gae_fn(compiled=True, mode=args.compile_mode)
                torch.compiler.cudagraph_mark_step_begin()
                shared_advantage, _ = shared_gae(reward, cached_current, zeros, zeros, zeros, step_values[-1], gamma, lam)
                comparison["shared_cached_gae_vs_reference"] = tensor_difference(baseline_advantage, shared_advantage)
                result["critic_batch_parity"]["nonzero_value_head" if nonzero else "zero_initialized_head"] = comparison

            # Restore the reference initialization for closed-loop measurements.
            agent.value_head.weight.zero_()
            result["closed_loop"] = {}
            for backend in args.backends:
                for validate in (True, False):
                    closed_env = make_env(args, n, backend)
                    try:
                        norm = reward_norm = next_obs = None

                        def reset():
                            nonlocal norm, reward_norm, next_obs
                            torch.manual_seed(args.seed)
                            norm = VectorObsNorm(n, obs_shape)
                            reward_norm = VectorRewardNorm(n, gamma=model_args.gamma)
                            raw, _ = closed_env.reset(seed=args.seed)
                            next_obs = torch.as_tensor(norm.normalize(raw), device="cuda")
                            torch.cuda.synchronize()

                        def step(_):
                            nonlocal next_obs
                            torch.compiler.cudagraph_mark_step_begin()
                            a, b, _ = compiled_rollout(next_obs)
                            if validate:
                                native_action = Beta(a, b, validate_args=True).sample().clamp(reference.SAMPLE_EPS, 1 - reference.SAMPLE_EPS)
                                scaled_action = agent.action_low + native_action * (agent.action_high - agent.action_low)
                            else:
                                _, scaled_action = sample_beta_actions(a, b, agent.action_low, agent.action_high)
                            action = scaled_action.cpu().numpy()
                            if not np.isfinite(action).all():
                                raise FloatingPointError("Non-finite sampled action")
                            raw, reward, terms, truncs, infos = closed_env.step(action)
                            reward_norm.normalize(reward, terms)
                            normalized, _ = norm.normalize_step(raw, terms, truncs, infos)
                            next_obs = torch.as_tensor(normalized, device="cuda")

                        reset()
                        for i in range(args.warmup_steps):
                            step(i)
                        label = f"{backend}/beta_{'checked' if validate else 'unchecked'}"
                        result["closed_loop"][label] = measure_cpu(step, args.env_steps, args.repeats, reset=reset, units_per_call=n)
                        torch.cuda.synchronize()
                        writer.add_scalar(f"{prefix}/closed_loop/{label}/transitions_per_second", result["closed_loop"][label]["units_per_second"], 0)
                        writer.flush()
                        if args.profile:
                            trace = run_dir / f"profile_n{n}_{backend}_{'checked' if validate else 'unchecked'}.json"
                            with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True,
                            ) as profiler:
                                for i in range(32):
                                    with torch.profiler.record_function(f"closed_loop/{label}"):
                                        step(i)
                            profiler.export_chrome_trace(str(trace))
                            result["closed_loop"][label]["profile_trace"] = str(trace)
                    finally:
                        closed_env.close()
            if args.profile_update:
                result["update"] = {}
                benchmark_update(args, reference, model_args, agent, support, obs_shape, env.single_action_space.shape, result["update"])
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        return result
    finally:
        env.close()


def write_scalars(writer, value, prefix):
    if isinstance(value, dict):
        for key, child in value.items():
            write_scalars(writer, child, f"{prefix}/{key}")
    elif isinstance(value, (int, float)):
        writer.add_scalar(prefix, value, 0)


def main():
    args = parse_args()
    configure_runtime()
    if not args.env_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; this benchmark never falls back to CPU models")
    if not args.env_only and not torch.cuda.is_bf16_supported():
        raise RuntimeError("v30 requires native CUDA BF16")
    # The shared harness parses this suffix as epoch seconds, as trainers do.
    run_dir = Path("runs") / f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir.mkdir(parents=True)
    report = {
        "kind": "fixed_work_performance_benchmark_not_training",
        "status": "running",
        "args": vars(args), "reference": REFERENCE,
        "reference_sha256": hashlib.sha256(Path(importlib.util.find_spec(REFERENCE).origin).read_bytes()).hexdigest(),
        "system": {"platform": platform.platform(), "torch": torch.__version__, "gymnasium": gym.__version__, "mujoco": mujoco.__version__},
        "measurements": {},
        "notes": ["CUDA-stream elapsed time includes launch starvation and is not summed kernel time.",
                  "Compilation and initialization are excluded from steady-state timings.",
                  "Closed-loop throughput excludes all learning updates and is not training SPS.",
                  "Environment timing includes all info construction and uses copy=True.",
                  "Parity tolerances: raw 1e-10 relative/absolute; float32 normalization 1e-6.",
                  "Numerical gates compare labels/means/KL/gradients, temperature weights/KL/ESS, and first-update parameters/loss/per-metric values. Thresholds are recorded per comparison.",
                  "Unfused captured updates use parameter atol1e-7/rtol1e-6 and loss/metric atol1e-6/rtol1e-6. Fused updates allow parameter atol2e-6/rtol2e-5 and loss atol3e-6/rtol3e-6; metric exceptions follow standalone solver tolerances.",
                  "These tolerances screen numerical regressions and do not prove equal learning scores. Critic batch-layout comparisons remain diagnostic and do not gate adoption.",
                  "Fixed sampled workloads do not measure learned-policy-dependent physics costs."],
    }
    if not args.env_only:
        report["system"]["gpu"] = torch.cuda.get_device_name()

    def save():
        (run_dir / "benchmark.json").write_text(json.dumps(json_compatible(report), indent=2, allow_nan=False) + "\n")

    with SummaryWriter(str(run_dir)) as writer:
        writer.add_text("benchmark/methodology", "\n".join(report["notes"]))
        writer.add_text("hyperparameters", json.dumps(vars(args), indent=2))
        save()
        try:
            for n in args.num_envs:
                prefix = f"benchmark/n{n}"
                result = report["measurements"][str(n)] = {"environments": {}}
                if args.projection_only:
                    limit = float(np.log1p(20_000.0))
                    support = Dreamer3BucketHLGaussSupport(51, -limit, limit, 0.75, "cuda")
                    torch.manual_seed(args.seed)
                    targets = torch.randn(39 * n, device="cuda") * 20
                    result["projection"] = {}
                    benchmark_projection(args, support, targets, result["projection"])
                    write_scalars(writer, result["projection"], f"{prefix}/projection")
                    writer.flush()
                    save()
                    enforce_numerical_parity(args, report, save)
                    continue
                for backend in args.backends:
                    print(f"N={n} backend={backend}: checking full-horizon parity", flush=True)
                    parity = environment_parity(args, n, backend)
                    measured = benchmark_environment(args, n, backend)
                    result["environments"][backend] = {"parity": parity, **measured}
                    write_scalars(writer, result["environments"][backend], f"{prefix}/environment/{backend}")
                    writer.flush()
                    save()
                    print(f"N={n} backend={backend}: raw {measured['raw']['units_per_second']:.0f}, normalized {measured['normalized']['units_per_second']:.0f} transitions/s", flush=True)
                if args.legacy_wrappers:
                    legacy_results = result["legacy_environments"] = {}
                    for backend in args.backends:
                        print(f"N={n} legacy wrappers backend={backend}: checking parity and timing", flush=True)
                        parity = environment_parity(args, n, backend, legacy=True)
                        measured = benchmark_environment(args, n, backend, legacy=True)
                        legacy_results[backend] = {"parity": parity, **measured}
                        write_scalars(writer, legacy_results[backend], f"{prefix}/legacy_environment/{backend}")
                        writer.flush()
                        save()
                        print(f"N={n} legacy wrappers backend={backend}: {measured['wrapped']['units_per_second']:.0f} transitions/s", flush=True)
                if args.thread_counts:
                    benchmark_thread_sweep(args, n, result, writer, prefix, save)
                if not args.env_only:
                    print(f"N={n}: measuring compiled CUDA v30 components and closed loop", flush=True)
                    result["gpu"] = {}
                    benchmark_gpu(args, n, writer, prefix, result["gpu"], run_dir)
                    write_scalars(writer, result["gpu"], f"{prefix}/gpu")
                    writer.flush()
                    save()
                    enforce_numerical_parity(args, report, save)
            report["status"] = "completed"
            writer.add_scalar("benchmark/complete", 1, 0)
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            writer.add_text("benchmark/error", f"{type(error).__name__}: {error}")
            raise
        finally:
            write_scalars(writer, report["measurements"], "benchmark/partial_results")
            writer.flush()
            save()
    print(f"RESULT {run_dir / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()

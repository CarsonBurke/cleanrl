"""Fixed-work v30 autocast-cache benchmark; not training or a score evaluation.

Queue exclusively through mlq with --max-parallel-runs 1. Tests exact compiled
BF16 outputs, unchanged Beta draws/RNG, and explicit master-update/refresh
semantics before measuring. Master parameters remain FP32; only Linear operands
and the reference's explicitly identified expert GEMM weights are cached.
"""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport
from cleanrl.shared.inference_cache import InferenceParameterCache, linear_parameter_names
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions
from scripts.benchmark_mujoco_throughput import (
    REFERENCE, json_compatible, measure_cuda, tensor_difference, write_scalars,
)


def selected_names(module, method):
    prefixes = ("trunk.", "policy_mlp.", "actor_alpha.", "actor_beta.") if method == "policy" else (
        "trunk.", "value_mlp.", "value_head.")
    # Keep the padded value head's original FP32 cast/pad lowering: caching
    # its BF16 weight changes Inductor's GEMM layout at N=64 and its rounding.
    names = [name for name in linear_parameter_names(module)
             if name.startswith(prefixes) and name != "value_head.weight"]
    # Explicit v30 contract: these tensors are only GEMM operands. Expert biases
    # are added OUTSIDE autocast GEMMs and must remain live FP32 tensors.
    names.extend(f"trunk.blocks.{i}.experts.{weight}"
                 for i in range(len(module.trunk.blocks)) for weight in ("weight1", "weight2"))
    return tuple(names)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--gpu-iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--exp-name", default="v30_inference_cache_v1")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cache-critic", action=argparse.BooleanOptionalAction, default=True,
                        help="Also cache selected critic operands; --no-cache-critic keeps the entire online critic on its original path.")
    args = parser.parse_args()
    if min(*args.num_envs, args.gpu_iterations, args.repeats) <= 0:
        parser.error("batch sizes, iterations and repeats must be positive")
    return args


def measure_case(args, n, result, save, run_dir):
    reference = importlib.import_module(REFERENCE)
    model_args = reference.Args(num_envs=n, seed=args.seed)
    torch.manual_seed(args.seed)
    # Only space metadata is needed; no simulation, warmup or training budget
    # is represented by these fixed observation batches.
    with gym.make(model_args.env_id) as env:
        spaces = SimpleNamespace(single_observation_space=env.observation_space,
                                 single_action_space=env.action_space)
        agent = reference.Agent(spaces, model_args).cuda()
    target = copy.deepcopy(agent).requires_grad_(False)
    limit = float(np.log1p(model_args.value_support_limit))
    support = Dreamer3BucketHLGaussSupport(model_args.num_value_bins, -limit, limit,
                                          model_args.value_sigma_to_bin_ratio, torch.device("cuda"))
    observations = torch.randn((n,) + spaces.single_observation_space.shape, device="cuda")
    master_ids = [{name: id(p) for name, p in model.named_parameters()} for model in (agent, target)]
    rng = torch.cuda.get_rng_state()
    policy_cache = InferenceParameterCache(target, selected_names(target, "policy"), method="policy")
    critic_cache = (InferenceParameterCache(agent, selected_names(agent, "value_logits"), method="value_logits")
                    if args.cache_critic else None)
    caches = (policy_cache, critic_cache) if critic_cache is not None else (policy_cache,)
    assert torch.equal(rng, torch.cuda.get_rng_state()), "cache construction consumed RNG"
    pointers = [{name: p.data_ptr() for name, p in cache.cached_parameters.items()}
                for cache in caches]
    result.update({
        "selection": {"policy": list(policy_cache.parameter_names),
                      "critic": list(critic_cache.parameter_names) if critic_cache is not None else []},
        "cache_critic": args.cache_critic,
        "cached_parameter_count": sum(len(cache.parameter_names) for cache in caches),
        "cached_bytes": sum(p.numel() * p.element_size() for cache in caches
                            for p in cache.cached_parameters.values()),
        "startup_seconds": {}, "parity": {}, "timing": {},
    })

    def original(obs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            alpha, beta = target.policy(obs)
            logits = agent.value_logits(obs)
        return alpha.float(), beta.float(), support.to_scalar(logits.float())

    def cached(obs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            alpha, beta = policy_cache(obs)
            logits = critic_cache(obs) if critic_cache is not None else agent.value_logits(obs)
        return alpha.float(), beta.float(), support.to_scalar(logits.float())

    variants = {"original": torch.compile(original, mode=args.compile_mode, fullgraph=True, dynamic=False),
                "cached": torch.compile(cached, mode=args.compile_mode, fullgraph=True, dynamic=False)}
    with torch.no_grad():
        for label, fn in variants.items():
            torch.cuda.synchronize()
            started = time.perf_counter()
            torch.compiler.cudagraph_mark_step_begin()
            fn(observations)
            torch.cuda.synchronize()
            result["startup_seconds"][label] = time.perf_counter() - started
            save()

        def check(label):
            before = torch.cuda.get_rng_state()
            torch.compiler.cudagraph_mark_step_begin()
            expected = tuple(t.clone() for t in variants["original"](observations))
            torch.compiler.cudagraph_mark_step_begin()
            actual = tuple(t.clone() for t in variants["cached"](observations))
            comparison = result["parity"][label] = {
                name: tensor_difference(a, b, atol=0.0, rtol=0.0)
                for name, a, b in zip(("alpha", "beta", "value"), expected, actual)
            }
            comparison["inference_rng_unchanged"] = torch.equal(before, torch.cuda.get_rng_state())
            native_a, action_a = sample_beta_actions(*expected[:2], target.action_low, target.action_high)
            after_sampling = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(before)
            native_b, action_b = sample_beta_actions(*actual[:2], target.action_low, target.action_high)
            comparison.update(native_actions_bitwise_equal=torch.equal(native_a, native_b),
                              scaled_actions_bitwise_equal=torch.equal(action_a, action_b),
                              sampling_rng_equal=torch.equal(after_sampling, torch.cuda.get_rng_state()))
            for model, ids in zip((agent, target), master_ids):
                assert {name: id(p) for name, p in model.named_parameters()} == ids
                assert all(p.dtype == torch.float32 and p.grad is None for p in model.parameters())
            for cache, addresses in zip(caches, pointers):
                assert {name: p.data_ptr() for name, p in cache.cached_parameters.items()} == addresses
            save()
            checks = [v for v in comparison.values() if isinstance(v, bool)]
            checks.extend(v["bitwise_equal"] and v["reference_finite"] and v["candidate_finite"]
                          for v in comparison.values() if isinstance(v, dict))
            if not all(checks):
                raise AssertionError(f"N={n} {label}: bitwise cache parity failed; see persisted JSON")

        check("zero_initialized_head")
        agent.value_head.weight.normal_(std=0.02)
        if critic_cache is not None:
            critic_cache.refresh()
        check("nonzero_value_head")
        # Deterministic master changes cover cached operands AND deliberately
        # uncached FP32 bias/gate state. This is not an optimizer/training run.
        for index in range(2):
            before = torch.cuda.get_rng_state()
            for model in (agent, target):
                model.trunk.entry.weight.add_(0.00390625)
                model.trunk.blocks[0].experts.weight1.mul_(0.9375)
                model.trunk.blocks[0].experts.bias1.add_(0.03125)
                model.trunk.blocks[0].resid_gate.sub_(0.125)
            agent.value_head.weight.mul_(0.875)
            policy_cache.refresh()
            if critic_cache is not None:
                critic_cache.refresh()
            assert torch.equal(before, torch.cuda.get_rng_state()), "refresh consumed RNG"
            check(f"master_update_and_refresh_{index + 1}")

        for label, fn in variants.items():
            result["timing"][label] = measure_cuda(lambda: fn(observations), args.gpu_iterations, args.repeats)
            save()
        for label, cache in (("policy_refresh", policy_cache), ("critic_refresh", critic_cache)):
            if cache is None:
                continue
            result["timing"][label] = measure_cuda(cache.refresh, args.gpu_iterations, args.repeats)
            save()

        def rollout(fn, refresh):
            if refresh:
                if critic_cache is not None:
                    critic_cache.refresh()
                # Refreshing all enabled caches every rollout is conservative;
                # actual v30 target refreshes much less frequently.
                policy_cache.refresh()
            for _ in range(model_args.num_steps):
                torch.compiler.cudagraph_mark_step_begin()
                fn(observations)

        for label, fn in variants.items():
            result["timing"][f"{label}_39_calls_with_refresh"] = measure_cuda(
                lambda: rollout(fn, label == "cached"), max(1, args.gpu_iterations // 39), args.repeats)
            save()
        timing = result["timing"]
        result["speedup_inference_only"] = timing["original"]["median_seconds"] / timing["cached"]["median_seconds"]
        result["speedup_39_calls_including_enabled_refreshes"] = (
            timing["original_39_calls_with_refresh"]["median_seconds"] /
            timing["cached_39_calls_with_refresh"]["median_seconds"])
        result["refresh_cadence_note"] = (
            "Measured 39-call candidate refreshes every enabled cache each rollout, conservatively. "
            "The estimate assumes target promotions only at the maximum age; "
            "v30's KL-triggered promotions can occur earlier. Neither includes learning or physics."
        )
        result["estimated_us_per_call_if_target_promoted_only_at_max_age"] = (
            timing["cached"]["microseconds_per_unit"]
            + timing.get("critic_refresh", {}).get("microseconds_per_unit", 0) / 39
            + timing["policy_refresh"]["microseconds_per_unit"] / (39 * model_args.target_update_period))
        if args.profile:
            for label, fn in variants.items():
                trace = run_dir / f"profile_n{n}_{label}.json"
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    for _ in range(32):
                        torch.compiler.cudagraph_mark_step_begin()
                        fn(observations)
                profiler.export_chrome_trace(str(trace))
                result.setdefault("traces", {})[label] = str(trace)
        result["passed"] = True
        save()


def main():
    args = parse_args()
    configure_runtime()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("queued CUDA BF16 execution required")
    run_dir = Path("runs") / f"HalfCheetah-v4__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=False)
    reference = importlib.import_module(REFERENCE)
    paths = [Path(__file__), Path(reference.__file__), Path("cleanrl/shared/inference_cache.py")]
    report = {"kind": "fixed_work_inference_cache_not_training", "status": "running", "args": vars(args),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              "measurements": {}}
    writer = SummaryWriter(str(run_dir))

    def save():
        (run_dir / "benchmark.json").write_text(json.dumps(json_compatible(report), indent=2, allow_nan=False) + "\n")
        writer.flush()

    try:
        writer.add_text("hyperparameters", json.dumps(vars(args), indent=2))
        save()
        for n in args.num_envs:
            result = report["measurements"][str(n)] = {}
            print(f"N={n}: checking exact compiled v30 inference cache", flush=True)
            measure_case(args, n, result, save, run_dir)
            write_scalars(writer, result, f"benchmark/n{n}")
            print(f"N={n}: bitwise parity passed; inference speedup {result['speedup_inference_only']:.3f}x; "
                  f"39 calls including refresh {result['speedup_39_calls_including_enabled_refreshes']:.3f}x", flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        writer.close()


if __name__ == "__main__":
    main()

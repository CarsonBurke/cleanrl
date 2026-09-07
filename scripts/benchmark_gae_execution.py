"""Compare cold compilation and steady GAE against a saved shared implementation.

Run through mlq. These are fixed-work, boundary-rich numerical fixtures, not
shortened training. Each implementation/shape runs in a fresh subprocess with
empty compiler caches; JSON and TensorBoard evidence are persisted under runs/.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.runtime import configure_runtime
from scripts.benchmark_mujoco_throughput import (
    json_compatible, tensor_difference, timing_summary, write_scalars,
)


TOLERANCES = {"atol": 1e-5, "rtol": 1e-5}
SHAPES = ((2048, 1), (2048, 16), (128, 16))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mode", default="reduce-overhead")
    parser.add_argument("--exp-name", default="gae_execution")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--envs", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--explicit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=("old", "current"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.iterations, args.repeats) <= 0:
        parser.error("iterations and repeats must be positive")
    args.baseline_root = args.baseline_root.resolve()
    if not (args.baseline_root / "cleanrl/shared/ppo_loop.py").is_file():
        parser.error("baseline-root must contain the saved cleanrl/shared/ppo_loop.py")
    return args


def load_baseline(root):
    spec = importlib.util.spec_from_file_location("saved_gae_reference", root / "cleanrl/shared/ppo_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(steps, envs, seed, explicit):
    generator = torch.Generator().manual_seed(seed)
    rewards, values, next_values = [torch.randn(steps, envs, generator=generator) for _ in range(3)]
    terms, truncs = torch.zeros_like(rewards), torch.zeros_like(rewards)
    terms[23::37] = 1
    truncs[17::53] = 1
    terms[0] = truncs[0] = 1
    # Include boundaries on each side of the bounded kernel's tile edges.
    terms[31::64, 0] = 1
    truncs[32::64, -1] = 1
    if envs > 1:
        truncs[-1, ::2] = 1
        terms[-1, 1::4] = 1
    tail = torch.randn(envs, generator=generator)
    matrices = [tensor.cuda() for tensor in (rewards, values, terms, truncs, next_values)]
    args = (*matrices, 0.99, 0.95) if explicit else (*matrices, tail.cuda(), 0.99, 0.95)
    counts = {
        "terminal_only": int(((terms == 1) & (truncs == 0)).sum()),
        "truncated_only": int(((terms == 0) & (truncs == 1)).sum()),
        "simultaneous": int(((terms == 1) & (truncs == 1)).sum()),
        "continuing": int(((terms == 0) & (truncs == 0)).sum()),
    }
    return args, counts


def measure(fn, iterations, repeats):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    wall, stream = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        wall.append(time.perf_counter() - started)
        stream.append(start.elapsed_time(end) / 1000)
    result = timing_summary(wall, iterations)
    result["cuda_stream_seconds"] = stream
    result["cuda_stream_microseconds_per_call"] = statistics.median(stream) * 1e6 / iterations
    return result


def save_json(path, value):
    path.write_text(json.dumps(json_compatible(value), indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def worker(args):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("queued CUDA execution required")
    baseline = load_baseline(args.baseline_root)
    inputs, boundaries = fixture(args.steps, args.envs, args.seed, args.explicit)
    reference = baseline.compute_gae_from_next_values if args.explicit else baseline.compute_gae
    expected = tuple(tensor.clone() for tensor in reference(*inputs))
    factory = baseline.get_gae_fn if args.implementation == "old" else get_gae_fn
    fn = factory(compiled=True, mode=args.mode, explicit_next_values=args.explicit)
    result = {"status": "running", "boundaries": boundaries, "parity": {},
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "steps": args.steps, "envs": args.envs, "explicit_next_values": args.explicit,
              "precision": "FP32/highest/TF32 disabled", "empty_compiler_caches": True}
    save_json(args.output, result)

    def replay():
        torch.compiler.cudagraph_mark_step_begin()
        return fn(*inputs)

    try:
        rng = torch.cuda.get_rng_state()
        torch.cuda.synchronize()
        started = time.perf_counter()
        actual = replay()
        torch.cuda.synchronize()
        result["cold_first_call_seconds"] = time.perf_counter() - started
        actual = tuple(tensor.clone() for tensor in actual)
        result["rng_unchanged"] = torch.equal(rng, torch.cuda.get_rng_state())
        for name, expected_tensor, actual_tensor in zip(("advantages", "returns"), expected, actual):
            result["parity"][name] = tensor_difference(expected_tensor, actual_tensor, **TOLERANCES)
        save_json(args.output, result)
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(actual_tensor, expected_tensor, **TOLERANCES)
        if not result["rng_unchanged"] or not all(bool(tensor.isfinite().all()) for tensor in actual):
            raise AssertionError("GAE RNG or finite-output gate failed")
        torch.save(tuple(tensor.cpu() for tensor in actual), args.output.with_suffix(".pt"))
        result["steady"] = measure(replay, args.iterations, args.repeats)
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(args.output, result)


def main(args):
    run_dir = Path("runs") / f"GAE__{args.exp_name}__{args.seed}__{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(str(run_dir))
    report = {"kind": "fixed_work_gae_not_training", "status": "running",
              "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "numerical_tolerances": TOLERANCES, "cases": {}, "sources": {}}
    sources = {"baseline": args.baseline_root / "cleanrl/shared/ppo_loop.py",
               "current_factory": Path("cleanrl/shared/ppo_loop.py"),
               "current_kernel": Path("cleanrl/shared/gae.py")}
    report["sources"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}

    def save():
        save_json(run_dir / "benchmark.json", report)
        write_scalars(writer, report["cases"], "benchmark/gae")
        writer.flush()

    try:
        save()
        for steps, envs in SHAPES:
            for explicit in (False, True):
                name = f"T{steps}_N{envs}_{'explicit' if explicit else 'standard'}"
                row = report["cases"][name] = {}
                for implementation in ("old", "current"):
                    output = run_dir / f"{name}_{implementation}.json"
                    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                               "--baseline-root", str(args.baseline_root), "--seed", str(args.seed),
                               "--steps", str(steps), "--envs", str(envs), "--mode", args.mode,
                               "--iterations", str(args.iterations), "--repeats", str(args.repeats),
                               "--implementation", implementation, "--output", str(output)]
                    if explicit:
                        command.append("--explicit")
                    # Fresh processes alone are not cold if disk caches survive.
                    with tempfile.TemporaryDirectory(prefix="gae-compile-") as cache:
                        environment = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=f"{cache}/inductor",
                                           TRITON_CACHE_DIR=f"{cache}/triton", TORCHINDUCTOR_FX_GRAPH_CACHE="0",
                                           TORCHINDUCTOR_AUTOGRAD_CACHE="0", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1")
                        completed = subprocess.run(command, env=environment, check=False)
                    row[implementation] = json.loads(output.read_text()) if output.exists() else {"status": "failed"}
                    save()
                    if completed.returncode:
                        raise RuntimeError(f"{name}/{implementation} failed with exit {completed.returncode}")
                old = torch.load(run_dir / f"{name}_old.pt", map_location="cpu", weights_only=True)
                current = torch.load(run_dir / f"{name}_current.pt", map_location="cpu", weights_only=True)
                row["old_compiled_vs_current"] = {
                    key: tensor_difference(before, after, **TOLERANCES)
                    for key, before, after in zip(("advantages", "returns"), old, current)
                }
                save()
                torch.testing.assert_close(current, old, **TOLERANCES)
                row["cold_speedup"] = row["old"]["cold_first_call_seconds"] / row["current"]["cold_first_call_seconds"]
                row["steady_speedup"] = row["old"]["steady"]["median_seconds"] / row["current"]["steady"]["median_seconds"]
                save()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        writer.close()
        print(run_dir / "benchmark.json", flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        main(arguments)

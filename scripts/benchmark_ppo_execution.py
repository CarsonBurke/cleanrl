"""Compare full seed-1 base PPO runs against a pre-change source snapshot.

Submit this entire command through mlq with --max-parallel-runs 1. Children
stay in its foreground process group. --runs 6 measures aggregate throughput,
not a rollout-only proxy. Every trainer runs the complete 8M-step budget.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._runs import RunScalars


ROOT = Path(__file__).resolve().parents[1]
STEPS = 8_000_000
BATCH = 16 * 2048
ITERATIONS = (STEPS - 16_000) // BATCH
FINAL_STEP = 16_000 + ITERATIONS * BATCH


def fingerprints(root):
    paths = [root / "cleanrl/ppo_continuous_action.py", *sorted((root / "cleanrl/shared").glob("*.py")),
             *sorted((root / "cleanrl/shared").glob("*.c"))]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_group(root, label, count, threads, spin, name, report_dir):
    processes, outputs = [], []
    started = time.perf_counter()
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CLEANRL_ENV_SPIN=str(spin))
    # A caller's source path must not defeat the explicit baseline/candidate cwd.
    environment.pop("PYTHONPATH", None)
    try:
        for index in range(count):
            experiment = f"{name}_{label}_{index}"
            command = [str(ROOT / ".venv/bin/python"), "-u", "-m", "cleanrl.ppo_continuous_action",
                       "--env-id", "HalfCheetah-v4", "--num-envs", "16", "--num-steps", "2048",
                       "--num-minibatches", "32", "--update-epochs", "10", "--seed", "1",
                       "--total-timesteps", str(STEPS), "--env-threads", str(threads),
                       "--compile", "--compile-mode", "reduce-overhead", "--exp-name", experiment]
            output = (report_dir / f"{label}_{index}.log").open("w")
            outputs.append(output)
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT)
            processes.append((experiment, process, command))
        while any(process.poll() is None for _, process, _ in processes):
            for experiment, process, _ in processes:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"{experiment} failed with exit {process.returncode}; see {report_dir}")
            time.sleep(0.5)
        elapsed = time.perf_counter() - started
        records = []
        for experiment, process, command in processes:
            if process.returncode:
                raise RuntimeError(f"{experiment} failed with exit {process.returncode}")
            directories = sorted((root / "runs").glob(f"HalfCheetah-v4__{experiment}__1__*"))
            if len(directories) != 1:
                raise RuntimeError(f"expected one unique run for {experiment}, got {directories}")
            run = RunScalars(directories[0])
            steps, interval = run.series("charts/interval_SPS")
            if len(steps) != ITERATIONS or int(steps[-1]) != FINAL_STEP:
                raise RuntimeError(f"incomplete run {experiment}: {len(steps)} iterations")
            _, returns = run.series("charts/episodic_return")
            if not returns.size or not np.isfinite(returns).all() or not np.isfinite(interval).all():
                raise RuntimeError(f"nonfinite or missing observations in {experiment}")
            phases = {}
            for tag in sorted(run.tags):
                if tag.startswith("timing/") and tag.endswith("_s"):
                    _, values = run.series(tag)
                    phases[tag] = {"total_seconds": float(values.sum()),
                                   "first_seconds": float(values[0]),
                                   "steady_mean_seconds": float(values[1:].mean()) if values.size > 1 else None}
            records.append({"run_dir": str(directories[0]), "command": command, "final_step": int(steps[-1]),
                            "final_100_return": float(returns[-100:].mean()),
                            "cumulative_sps": run.latest("charts/SPS"),
                            "steady_interval_sps": float(BATCH * (len(interval) - 1) / (BATCH / interval[1:]).sum()),
                            "phases": phases})
        return {"wall_seconds": elapsed, "aggregate_sps": count * FINAL_STEP / elapsed, "runs": records}
    finally:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for output in outputs:
            output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--runs", type=int, choices=[1, 3, 6], default=1)
    parser.add_argument("--env-threads", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--env-spin", type=int, default=5000)
    parser.add_argument("--exp-name", default="ppo_execution_v1")
    args = parser.parse_args()
    baseline = args.baseline_root.resolve()
    if not (baseline / "cleanrl/ppo_continuous_action.py").is_file():
        parser.error("--baseline-root must contain the pre-change trainer and shared modules")
    if args.env_spin < 0:
        parser.error("--env-spin must be nonnegative")
    stamp = int(time.time())
    name = f"{args.exp_name}_{stamp}"
    directory = ROOT / "runs" / f"HalfCheetah-v4__{name}__1__{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    report = {"kind": "complete_base_ppo_execution_comparison", "status": "running", "seed": 1,
              "budget": STEPS, "runs_per_group": args.runs, "env_threads": args.env_threads,
              "env_spin": args.env_spin, "sources": {"before": fingerprints(baseline), "after": fingerprints(ROOT)},
              "groups": {}}
    writer = SummaryWriter(str(directory))
    def save():
        (directory / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        writer.flush()
    try:
        save()
        for label, root in [("before", baseline), ("after", ROOT)]:
            print(f"Starting {label}: {args.runs} complete 8M runs", flush=True)
            group = run_group(root, label, args.runs, args.env_threads, args.env_spin, name, directory)
            report["groups"][label] = group
            writer.add_scalar(f"benchmark/{label}/aggregate_sps", group["aggregate_sps"], FINAL_STEP)
            save()
        report["whole_process_speedup"] = report["groups"]["before"]["wall_seconds"] / report["groups"]["after"]["wall_seconds"]
        writer.add_scalar("benchmark/whole_process_speedup", report["whole_process_speedup"], FINAL_STEP)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        writer.close()
    print(f"RESULT {directory / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Paired v8 controller/native-MuJoCo benchmark, not a training experiment.

Run exclusively through mlq (max-parallel-runs=1). The baseline directory needs
control.py and genome.py; its __init__.py is deliberately not executed because a
source snapshot need not contain the trainer imported by that initializer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.collective_control import control as current
from cleanrl.collective_control.genome import Genome
from cleanrl.shared.runtime import configure_runtime

SHAPES = {"proposal": (129, 1), "confirmation": (5, 4), "development": (1, 16)}
HORIZON = 1000
RESIDENTS = 16
INITIAL_NODES = 32
MAX_NODES = 128
ACTION_ATOL = 1e-6
ACTION_RTOL = 1e-5


def load_baseline(path: Path):
    """Create a real isolated package so relative imports share Genome identity."""
    name = "_collective_control_benchmark_baseline"
    spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(path.resolve())]
    sys.modules[name] = package
    return importlib.import_module(f"{name}.control"), importlib.import_module(f"{name}.genome")


def make_teams(seed, observation_dim, action_dim, authority, profile, team_count):
    rng = np.random.default_rng(seed)
    evolved = authority == "evolved"
    population = [Genome.random(rng, observation_dim, action_dim, INITIAL_NODES, evolved) for _ in range(RESIDENTS)]
    if profile == "heterogeneous":
        # Stress actual insertion/deletion and dangling node references, rather
        # than pretending every resident still has the initial 32-node shape.
        for resident, genome in enumerate(population):
            for _ in range(1 + resident % 8):
                genome.mutate(rng, observation_dim, action_dim, MAX_NODES, 2.0, 1.0, evolved)
    teams = [population]
    for _ in range(team_count - 1):
        parent, victim = rng.integers(RESIDENTS, size=2)
        candidate = population[parent].clone()
        candidate.mutate(rng, observation_dim, action_dim, MAX_NODES, 2.0, 0.02, evolved)
        team = list(population)
        team[victim] = candidate
        teams.append(team)
    return teams


def convert_teams(teams, genome_type):
    return [[genome_type.from_json(genome.to_json()) for genome in team] for team in teams]


def timed(fn, device):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize(device)
    return time.perf_counter() - started, value


def summary(samples):
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples)}


def parity(reference, actual, *, atol=ACTION_ATOL, rtol=ACTION_RTOL):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    difference = np.abs(reference - actual)
    return {
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(np.allclose(reference, actual, atol=atol, rtol=rtol)),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(actual).all()),
        "max_absolute_error": float(np.max(difference)),
        "mean_absolute_error": float(np.mean(difference)),
    }


def controller_benchmark(modules, teams, adapter_json, low, high, seeds, args, device):
    mapping = np.repeat(np.arange(len(teams[0])), len(seeds)).tolist()
    observations = np.random.default_rng(args.seed + 71).normal(
        size=(max(args.controller_steps, 32), len(mapping), len(adapter_json["mean"]))
    ).astype(np.float32)
    results = {name: {"setup": [], "first_action": [], "steady_step": []} for name in modules}
    traces = {}
    for repeat in range(args.repeats):
        # Alternate measurement order to reduce systematic thermal/order bias.
        names = list(modules) if repeat % 2 == 0 else list(reversed(modules))
        for name in names:
            module = modules[name]
            arm_teams = teams[0] if name == "baseline" else teams[1]

            def setup():
                controller = module.CollectiveController(
                    arm_teams, module.ObservationAdapter.from_json(adapter_json), low, high,
                    MAX_NODES, args.authority, device,
                )
                controller.reset(len(mapping), mapping)
                return controller

            elapsed, controller = timed(setup, device)
            results[name]["setup"].append(elapsed)
            elapsed, _ = timed(lambda: controller.action(observations[0]), device)
            results[name]["first_action"].append(elapsed)
            controller.reset(len(mapping), mapping)
            traces[name] = np.stack([controller.action(obs).copy() for obs in observations[:32]])
            for index in range(args.warmup):
                controller.action(observations[index % len(observations)])

            def steps():
                for observation in observations[:args.controller_steps]:
                    controller.action(observation)

            elapsed, _ = timed(steps, device)
            results[name]["steady_step"].append(elapsed / args.controller_steps)
            del controller
    metrics = {name: {key: summary(values) for key, values in result.items()} for name, result in results.items()}
    return {
        "timing": metrics,
        "speedup_baseline_over_current": {
            key: metrics["baseline"][key]["median_seconds"] / metrics["current"][key]["median_seconds"]
            for key in ("setup", "first_action", "steady_step")
        },
        "one_step_parity": parity(traces["baseline"][0], traces["current"][0]),
        "shared_observation_32_step_parity": parity(traces["baseline"], traces["current"]),
        "steady_step_includes": "public action: observation upload, recurrent computation, owned host action download",
    }


def evaluation_benchmark(baseline, team_versions, adapter_json, seeds, args, device):
    samples = {"baseline": [], "current": []}
    returns = {"baseline": [], "current": []}
    # Constructor is cheap; cold native-env/controller creation is inside the
    # first evaluate call and deliberately included in its timing.
    with current.TeamEvaluator(
        args.env_id, current.ObservationAdapter.from_json(adapter_json), MAX_NODES,
        args.authority, device, args.threads,
    ) as evaluator:
        for repeat in range(args.repeats):
            baseline_teams, current_teams = team_versions[repeat % 2]
            calls = {
                "baseline": lambda: baseline.evaluate_teams(
                    args.env_id, baseline_teams, baseline.ObservationAdapter.from_json(adapter_json),
                    seeds, HORIZON, MAX_NODES, args.authority, device, args.threads,
                ),
                "current": lambda: evaluator.evaluate(current_teams, seeds, HORIZON),
            }
            names = list(calls) if repeat % 2 == 0 else list(reversed(calls))
            for name in names:
                elapsed, result = timed(calls[name], device)
                samples[name].append(elapsed)
                returns[name].append(result.copy())
    pairs = []
    for index, (reference, actual) in enumerate(zip(returns["baseline"], returns["current"])):
        pairs.append({
            "call": index + 1,
            "genome_version": index % 2,
            "baseline_returns": reference.tolist(),
            "current_returns": actual.tolist(),
            # Tight return comparison is a diagnostic, NOT a correctness gate:
            # tiny action rounding differences can diverge in chaotic physics.
            "trajectory_parity_diagnostic": parity(reference, actual, atol=1e-3, rtol=1e-5),
            "baseline_mean": float(reference.mean()),
            "current_mean": float(actual.mean()),
        })
    return {
        "horizon": HORIZON,
        "timing": {name: summary(values) for name, values in samples.items()},
        "cold_speedup_baseline_over_current": samples["baseline"][0] / samples["current"][0],
        "reuse_speedup_baseline_over_current": (
            statistics.median(samples["baseline"][1:]) / statistics.median(samples["current"][1:])
            if args.repeats > 1 else None
        ),
        "paired_evaluations": pairs,
        "reuse_contract": "same evaluator and batch shape; alternate independently seeded genomes, reset identical episode seeds",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--stage", choices=("controller", "evaluation", "all"), default="all")
    parser.add_argument("--shapes", nargs="+", choices=tuple(SHAPES), default=list(SHAPES))
    parser.add_argument("--profiles", nargs="+", choices=("initial", "heterogeneous"), default=["initial", "heterogeneous"])
    parser.add_argument("--authority", choices=("uniform", "evolved"), default="uniform")
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--controller-steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Also write JSON to this file; stdout always contains the report")
    args = parser.parse_args()
    if min(args.repeats, args.controller_steps, args.threads) < 1 or args.warmup < 0:
        parser.error("repeats, controller-steps and threads must be positive; warmup must be nonnegative")
    for name in ("control.py", "genome.py"):
        if not (args.baseline_path / name).is_file():
            parser.error(f"baseline package is missing {name}")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA; run through mlq with --max-parallel-runs 1")
    configure_runtime()
    device = torch.device("cuda")
    baseline, baseline_genome = load_baseline(args.baseline_path)
    with gym.make(args.env_id) as env:
        observation_dim = int(np.prod(env.observation_space.shape))
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
    action_dim = len(low)
    adapter_json = current.ObservationAdapter(
        np.linspace(-0.2, 0.2, observation_dim, dtype=np.float32),
        np.linspace(0.5, 2.0, observation_dim, dtype=np.float32),
    ).to_json()
    report = {
        "scope": "Seeded synthetic v8 workload; controller and evaluation throughput only, no training or selection-quality claim",
        "config": {**vars(args), "baseline_path": str(args.baseline_path.resolve()), "output": str(args.output) if args.output else None},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
        "shapes": {"residents": RESIDENTS, "initial_nodes": INITIAL_NODES, "max_nodes": MAX_NODES, "horizon": HORIZON},
        "source_sha256": {
            arm: {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in ("control.py", "genome.py")}
            for arm, path in (("baseline", args.baseline_path), ("current", Path(current.__file__).parent))
        },
        "tolerance_policy": "Action allclose (atol=1e-6, rtol=1e-5) gates parity on shared observations. Full-horizon return allclose (atol=1e-3, rtol=1e-5) is diagnostic only: chaotic trajectory sensitivity is reported, not silently tolerated or treated as proof of equal selection.",
        "cases": [],
    }
    for profile in args.profiles:
        for shape in args.shapes:
            team_count, seed_count = SHAPES[shape]
            seeds = np.random.SeedSequence([args.seed, 307]).generate_state(seed_count).tolist()
            versions = []
            for version in range(2):
                teams = make_teams(args.seed + 1009 * version, observation_dim, action_dim, args.authority, profile, team_count)
                versions.append((convert_teams(teams, baseline_genome.Genome), teams))
            node_counts = [len(genome.nodes) for team in versions[0][1] for genome in team]
            payload = [[genome.to_json() for genome in team] for team in versions[0][1]]
            case = {
                "profile": profile, "shape": shape, "teams": team_count, "seeds": seeds,
                "node_count_min": min(node_counts), "node_count_max": max(node_counts),
                "genomes_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            }
            if args.stage in ("controller", "all"):
                case["controller"] = controller_benchmark(
                    {"baseline": baseline, "current": current}, versions[0], adapter_json, low, high, seeds, args, device,
                )
            if args.stage in ("evaluation", "all"):
                case["evaluation"] = evaluation_benchmark(baseline, versions, adapter_json, seeds, args, device)
            report["cases"].append(case)
    report["action_parity_passed"] = all(
        case["controller"][key]["allclose"]
        for case in report["cases"] if "controller" in case
        for key in ("one_step_parity", "shared_observation_32_step_parity")
    ) if args.stage != "evaluation" else None
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    if report["action_parity_passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

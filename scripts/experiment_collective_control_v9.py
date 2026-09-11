"""Full-horizon evolutionary control experiments; run exclusively through mlq.

The two arms match initial mutable-field counts to within two fields (0.05%)
and use the same transition target. Whole-generation overshoot is reported,
not hidden. Final test seeds never participate in search or dose calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.collective_control.control import CollectiveController, TeamEvaluator, calibrate_observations
from cleanrl.collective_control.evolve import Config, environment_dimensions, evaluate_checkpoint, load_checkpoint, seeds, train
from cleanrl.collective_control.genome import Genome
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *(root / "cleanrl/collective_control").glob("*.py")]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sem": float(values.std(ddof=1) / np.sqrt(values.size)),
            "min": float(values.min()), "max": float(values.max()), "values": values.tolist()}


def field_count(population):
    return sum(8 * len(genome.nodes) + len(genome.outputs) + int(genome.authority is not None) for genome in population)


def rollout_observations(config, adapter, population, episode_seeds, *, blind=False):
    """Actual fixed-horizon episodes, plus a deterministic observation probe."""
    env = make_mujoco_vector_env(config.env_id, len(episode_seeds), backend="native",
                                 num_threads=config.env_threads, copy=False)
    try:
        controller = CollectiveController(population, adapter, env.single_action_space.low,
                                          env.single_action_space.high, config.max_nodes,
                                          config.authority, torch.device(config.device))
        observation, _ = env.reset(seed=episode_seeds)
        controller.reset(len(episode_seeds))
        returns = np.zeros(len(episode_seeds), dtype=np.float64)
        active = np.ones(len(episode_seeds), dtype=bool)
        probe = []
        action_square_sum = 0.0
        action_count = 0
        for step in range(config.horizon):
            inputs = np.broadcast_to(adapter.mean, observation.shape) if blind else observation
            action = controller._action_buffer(inputs)
            action_square_sum += float(np.square(action[active].astype(np.float64)).sum())
            action_count += int(active.sum()) * action.shape[-1]
            # Consecutive observations retain recurrence; only one episode is
            # needed for the teacher-forced expression diagnostic.
            probe.append(np.asarray(observation[0], dtype=np.float32).copy())
            observation, reward, terminated, truncated, _ = env.step(action)
            np.add(returns, reward, out=returns, where=active)
            active &= ~np.logical_or(terminated, truncated)
            if not active.any():
                break
        return returns, np.asarray(probe), float(np.sqrt(action_square_sum / action_count))
    finally:
        env.close()


def calibrate(args):
    coherent = args.arm == "coherent"
    config = Config(env_threads=args.threads, proposal_episodes=1,
                    residents=1 if coherent else 16, initial_nodes=523 if coherent else 32,
                    max_nodes=2048 if coherent else 128)
    obs_dim, action_dim = environment_dimensions(config.env_id, config.env_threads)
    adapter = calibrate_observations(config.env_id, seeds(1, 11, 0, 4), 128, config.env_threads)
    rng = np.random.default_rng(1)
    population = [Genome.random(rng, obs_dim, action_dim, config.initial_nodes, False)
                  for _ in range(config.residents)]
    _, probe, _ = rollout_observations(config, adapter, population, seeds(1, 501, 0, 4))
    teams = [population]
    doses = [1.0, 2.0, 8.0, 32.0]
    metadata = []
    for dose in doses:
        # Paired victims and RNG starts across doses; no parent transplantation.
        for index in range(32):
            variation = np.random.default_rng(np.random.SeedSequence([9127, index]))
            victim = index % len(population)
            candidate = population[victim].clone()
            candidate.mutate(variation, obs_dim, action_dim, config.max_nodes, dose, 0.02, False)
            team = population.copy()
            team[victim] = candidate
            teams.append(team)
            metadata.append(dose)
    with TeamEvaluator(config.env_id, adapter, config.max_nodes, config.authority,
                       torch.device("cuda"), config.env_threads) as evaluator:
        # Independent training-pool seeds, not development or final test.
        returns = evaluator.evaluate(teams, seeds(1, 503, 0, 16), config.horizon)
    env = make_mujoco_vector_env(config.env_id, 1, backend="native", num_threads=config.env_threads, copy=False)
    try:
        controller = CollectiveController(teams, adapter, env.single_action_space.low,
                                          env.single_action_space.high, config.max_nodes,
                                          "uniform", torch.device("cuda"))
    finally:
        env.close()
    controller.reset(len(teams), list(range(len(teams))))
    square_differences = np.zeros(len(metadata), dtype=np.float64)
    maximum_differences = np.zeros(len(metadata), dtype=np.float64)
    for observation in probe:
        actions = controller._action_buffer(np.broadcast_to(observation, (len(teams), obs_dim)))
        difference = actions[1:].astype(np.float64) - actions[0]
        square_differences += np.square(difference).mean(axis=1)
        maximum_differences = np.maximum(maximum_differences, np.abs(difference).max(axis=1))
    rms = np.sqrt(square_differences / len(probe))
    gains = returns[1:] - returns[0]
    report = {"kind": "frozen_local_dose_calibration_not_training", "seed": 1, "arm": args.arm,
              "horizon": config.horizon, "episodes": 16, "probe_steps": len(probe),
              "initial_mutable_fields": field_count(population), "source_hashes": source_hashes(), "doses": {}}
    for dose in doses:
        mask = np.asarray(metadata) == dose
        means = gains[mask].mean(axis=1)
        report["doses"][str(dose)] = {
            "teacher_forced_team_action_rms": summarize(rms[mask]),
            "numerically_neutral_fraction": float(np.mean(maximum_differences[mask] <= 1e-7)),
            "paired_return_mean_gains": summarize(means),
            "positive_mean_fraction": float(np.mean(means > 0)),
        }
    return report


def run_arm(args):
    coherent = args.arm == "coherent"
    config = Config(run_dir=str(args.run_dir), residents=1 if coherent else 16,
                    initial_nodes=523 if coherent else 32, max_nodes=2048 if coherent else 128,
                    candidates=128, proposal_episodes=1, confirmation_episodes=4,
                    development_episodes=64, development_every=5, env_threads=args.threads,
                    total_transitions=args.transitions, generations=10000, seed=1,
                    mutation_events=args.mutation_events, transplant_probability=0.0)
    # Pure local variation in both arms isolates controller organization rather
    # than confounding it with a transplantation operator unavailable to one arm.
    manifest = {"arm": args.arm, "config": asdict(config), "source_hashes": source_hashes(),
                "initial_mutable_fields": config.residents * (8 * config.initial_nodes + 6),
                "matching": "4192 ensemble versus 4190 coherent fields; whole-generation transition overshoot reported"}
    manifest_path = args.run_dir.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run_dir = train(config)
    checkpoint = run_dir / "champion.json"
    result = evaluate_checkpoint(checkpoint, 64, 1000, None)
    saved_config, adapter, population, _ = load_checkpoint(checkpoint)
    test_seeds = seeds(config.seed + 100003, 307, 0, 64)
    blind_returns, _, blind_rms = rollout_observations(saved_config, adapter, population, test_seeds, blind=True)
    normal = np.asarray(result["returns"])
    result.update({"arm": args.arm, "initial_mutable_fields": manifest["initial_mutable_fields"],
                   "final_mutable_fields": field_count(population),
                   "no_observation": summarize(blind_returns),
                   "paired_observation_advantage": summarize(normal - blind_returns),
                   "blind_action_rms": blind_rms, "source_hashes": manifest["source_hashes"]})
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    result["final_training_metrics"] = metrics[-1]
    result["training_wall_seconds"] = metrics[-1]["elapsed_seconds"]
    if args.reference_checkpoint is not None:
        reference = evaluate_checkpoint(args.reference_checkpoint, 64, 1000, config.seed + 100003)
        result["reference"] = reference
        result["paired_reference_advantage"] = summarize(normal - np.asarray(reference["returns"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "train"])
    parser.add_argument("--arm", choices=["ensemble", "coherent"], default="ensemble")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--transitions", type=int, default=128_000_000)
    parser.add_argument("--mutation-events", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "train" and args.run_dir is None:
        parser.error("train requires --run-dir")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    report = calibrate(args) if args.mode == "calibrate" else run_arm(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "mean_return": report.get("mean_return")}))


if __name__ == "__main__":
    main()

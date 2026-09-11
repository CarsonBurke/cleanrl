"""V9 evolutionary collective control; no PPO, SGD, or prediction heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from scipy.stats import t as student_t
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.runtime import configure_runtime

from .control import ObservationAdapter, TeamEvaluator, calibrate_observations, evaluate
from .genome import Genome


SOURCE_CONTRACT = {
    "decoder": "typed-v9: kind0 observation channel; kind1 stable node ID; kind2 proposal channel; kind3 zero; kind4 half",
    "kind2": "resident's own previous normalized proposal, not the collective executed action",
    "missing_node": "normalized half",
}


@dataclass
class Config:
    env_id: str = "HalfCheetah-v4"
    run_dir: str = "runs/collective_control"
    residents: int = 16
    candidates: int = 128
    shortlist: int = 4
    initial_nodes: int = 32
    max_nodes: int = 128
    mutation_events: float = 2.0
    length_probability: float = 0.02
    transplant_probability: float = 0.1
    generations: int = 10000
    total_transitions: int = 0
    horizon: int = 1000
    proposal_horizon: int = 0
    confirmation_horizon: int = 0
    proposal_episodes: int = 2
    confirmation_episodes: int = 4
    validation_episodes: int = 16
    development_episodes: int = 64
    development_every: int = 1
    plateau_warmup_evaluations: int = 20
    plateau_patience: int = 20
    plateau_material_delta: float = 0.01
    plateau_decay: float = 0.8
    calibration_episodes: int = 4
    calibration_steps: int = 128
    authority: str = "uniform"
    seed: int = 1
    time_limit_seconds: int = 0
    device: str = "cuda"
    env_threads: int = 2


@dataclass
class Champion:
    generation: int
    mean_return: float
    returns: list[float]
    population: list[Genome]


@dataclass
class CullState:
    evaluations: int = 0
    ema: float | None = None
    raw_progress: float | None = None
    ema_progress: float | None = None
    stale_evaluations: int = 0

    def update(self, value: float, config: Config) -> bool:
        self.evaluations += 1
        self.ema = value if self.ema is None else config.plateau_decay * self.ema + (1 - config.plateau_decay) * value
        raw_progress = self.raw_progress is None or value > self.raw_progress + config.plateau_material_delta
        ema_progress = self.ema_progress is None or self.ema > self.ema_progress + config.plateau_material_delta
        if raw_progress:
            self.raw_progress = value
        if ema_progress:
            self.ema_progress = self.ema
        if raw_progress or ema_progress or self.evaluations <= config.plateau_warmup_evaluations:
            self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1
        return config.plateau_patience > 0 and self.stale_evaluations >= config.plateau_patience


def seeds(seed: int, namespace: int, generation: int, count: int) -> list[int]:
    sequence = np.random.SeedSequence([seed, namespace, generation])
    return [int(value) for value in sequence.generate_state(count, dtype=np.uint32)]


def paired_validation(incumbent: np.ndarray, candidate: np.ndarray, critical_value: float) -> dict[str, Any]:
    """One-sided paired Student-t bound; exact at finite n for iid normal differences.

    For nonnormal return differences this is an approximation, not a
    distribution-free guarantee. Selection must precede this independent sample.
    """
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(incumbent, dtype=np.float64)
    if differences.ndim != 1 or differences.size < 2 or not np.all(np.isfinite(differences)):
        raise ValueError("paired validation requires at least two finite return differences")
    mean = float(np.mean(differences))
    sem = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
    lower_bound = mean - critical_value * sem
    accepted = lower_bound > 0.0
    return {
        "paired_mean_gain": mean,
        "paired_sem": sem,
        "paired_lower_bound": lower_bound,
        "accepted": accepted,
        "rejection": None if accepted else "nonpositive_paired_lower_bound",
    }


def dump_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def population_json(population: list[Genome]) -> list[dict[str, Any]]:
    return [genome.to_json() for genome in population]


def save_state(
    run_dir: Path,
    config: Config,
    adapter: ObservationAdapter,
    generation: int,
    population: list[Genome],
    champion: Champion | None,
    cull_state: CullState,
    cumulative_evaluation_transitions: int,
    stop_reason: dict[str, Any] | None,
) -> None:
    state = {
        "schema": 3,
        "algorithm_version": "v9",
        "source_contract": SOURCE_CONTRACT,
        "config": asdict(config),
        "generation": generation,
        "population": population_json(population),
        "observation_adapter": adapter.to_json(),
        "return_contract": "sum of raw Gymnasium rewards until termination or the fixed horizon per episode",
        "transition_contract": "actual vector environment transitions, including inactive lanes stepped; excludes observation calibration",
        "cumulative_evaluation_transitions": cumulative_evaluation_transitions,
        "calibration_seeds": seeds(config.seed, 11, 0, config.calibration_episodes),
        "calibration_transition_upper_bound": config.calibration_episodes * config.calibration_steps,
        "development_seeds": seeds(config.seed, 31, 0, config.development_episodes),
        "cull_state": asdict(cull_state),
        "stop_reason": stop_reason,
        "champion": None
        if champion is None
        else {
            "generation": champion.generation,
            "mean_return": champion.mean_return,
            "returns": champion.returns,
            "population": population_json(champion.population),
        },
    }
    dump_json(run_dir / "latest.json", state)
    if champion is not None:
        dump_json(
            run_dir / "champion.json",
            {
                **state,
                "generation": champion.generation,
                "population": population_json(champion.population),
                "mean_return": champion.mean_return,
                "returns": champion.returns,
            },
        )


def load_checkpoint(path: Path) -> tuple[Config, ObservationAdapter, list[Genome], int]:
    value = json.loads(path.read_text())
    config = Config(**value["config"])
    adapter = ObservationAdapter.from_json(value["observation_adapter"])
    population = [Genome.from_json(item) for item in value["population"]]
    return config, adapter, population, int(value["generation"])


def checkpoint_semantics(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    legacy = value.get("schema", 1) < 3
    return {
        "checkpoint_schema": value.get("schema", 1),
        "evaluation_algorithm_version": "v9",
        "source_contract": SOURCE_CONTRACT,
        "legacy_reinterpreted": legacy,
        "semantics_note": "Legacy genome evaluated with corrected typed addressing; not exact legacy reproduction."
        if legacy else "Schema3 typed source addressing and resident-previous-proposal feedback.",
    }


def environment_dimensions(env_id: str, env_threads: int) -> tuple[int, int]:
    from cleanrl.shared.mujoco_env import make_mujoco_vector_env

    env = make_mujoco_vector_env(env_id, 1, backend="native", num_threads=env_threads, copy=False)
    observation_dim = int(np.prod(env.single_observation_space.shape))
    action_dim = int(np.prod(env.single_action_space.shape))
    env.close()
    return observation_dim, action_dim


def stage_horizon(config: Config, value: int) -> int:
    horizon = config.horizon if value == 0 else value
    if horizon < 1 or horizon > config.horizon:
        raise ValueError("stage horizons must be in 1..horizon, or zero for full horizon")
    return horizon


def validate_config(config: Config) -> None:
    for name in ("residents", "candidates", "shortlist", "proposal_episodes", "confirmation_episodes",
                 "development_episodes", "development_every", "horizon", "env_threads",
                 "calibration_episodes", "calibration_steps"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("generations", "total_transitions", "time_limit_seconds", "plateau_warmup_evaluations", "plateau_patience", "seed"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if config.validation_episodes < 2:
        raise ValueError("validation_episodes must be at least two for a paired Student-t bound")
    if config.shortlist > config.candidates:
        raise ValueError("shortlist cannot exceed candidates")
    if config.max_nodes < config.initial_nodes or config.initial_nodes < 1:
        raise ValueError("max_nodes must contain initial_nodes")
    for name in ("transplant_probability", "length_probability"):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for name in ("mutation_events", "plateau_material_delta"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 <= config.plateau_decay < 1:
        raise ValueError("plateau_decay must be in [0, 1)")
    if config.authority not in {"uniform", "evolved"}:
        raise ValueError("authority must be uniform or evolved")
    stage_horizon(config, config.proposal_horizon)
    stage_horizon(config, config.confirmation_horizon)


def train(config: Config) -> Path:
    validate_config(config)
    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this trainer requires a CUDA device; CPU fallback is disabled")
    proposal_horizon = stage_horizon(config, config.proposal_horizon)
    confirmation_horizon = stage_horizon(config, config.confirmation_horizon)
    # One candidate per resident is tested; Bonferroni controls the per-generation
    # family under the paired-test assumptions, not the entire adaptive run.
    validation_alpha = 0.05 / config.residents
    critical_value = float(student_t.ppf(1 - validation_alpha, config.validation_episodes - 1))
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(config.device)
    rng = np.random.default_rng(config.seed)
    observation_dim, action_dim = environment_dimensions(config.env_id, config.env_threads)
    calibration_seeds = seeds(config.seed, 11, 0, config.calibration_episodes)
    development_seeds = seeds(config.seed, 31, 0, config.development_episodes)
    adapter = calibrate_observations(config.env_id, calibration_seeds, config.calibration_steps, config.env_threads)
    population = [
        Genome.random(rng, observation_dim, action_dim, config.initial_nodes, config.authority == "evolved")
        for _ in range(config.residents)
    ]
    champion: Champion | None = None
    cull_state = CullState()
    started = time.monotonic()
    writer = SummaryWriter(log_dir=str(run_dir))
    with writer, TeamEvaluator(
        config.env_id, adapter, config.max_nodes, config.authority, device, config.env_threads
    ) as evaluator, (run_dir / "metrics.jsonl").open("w") as metrics:
        for generation in range(config.generations + 1):
            start_transitions = evaluator.evaluation_transitions
            row: dict[str, Any] = {"generation": generation, "accepted": 0, "decisions": []}
            if generation > 0:
                proposal_seeds = seeds(config.seed, 101, generation, config.proposal_episodes)
                proposal_teams = [population.copy()]
                proposal_metadata: list[tuple[int, Genome, str]] = []
                operators = {
                    name: {"proposed": 0, "accepted": 0, "proposal_gain_sum": 0.0,
                           "screening_gain_sum": 0.0, "paired_gain_sum": 0.0, "accepted_gain_sum": 0.0}
                    for name in ("mutation", "transplant")
                }
                for _ in range(config.candidates):
                    victim = int(rng.integers(config.residents))
                    parent = victim
                    operator = "mutation"
                    if config.residents > 1 and rng.random() < config.transplant_probability:
                        parent = int(rng.integers(config.residents - 1))
                        parent += parent >= victim
                        operator = "transplant"
                    candidate = population[parent].clone()
                    if operator == "mutation":
                        candidate.mutate(rng, observation_dim, action_dim, config.max_nodes, config.mutation_events,
                                         config.length_probability, config.authority == "evolved")
                    replaced = population.copy()
                    replaced[victim] = candidate
                    proposal_teams.append(replaced)
                    proposal_metadata.append((victim, candidate, operator))
                    operators[operator]["proposed"] += 1
                proposal_start = evaluator.evaluation_transitions
                proposal_returns = evaluator.evaluate(proposal_teams, proposal_seeds, proposal_horizon)
                proposal_transitions = evaluator.evaluation_transitions - proposal_start
                incumbent_mean = float(np.mean(proposal_returns[0]))
                proposals: list[list[tuple[float, Genome, str]]] = [[] for _ in population]
                for index, (victim, candidate, operator) in enumerate(proposal_metadata, start=1):
                    gain = float(np.mean(proposal_returns[index] - proposal_returns[0]))
                    proposals[victim].append((gain, candidate, operator))
                    operators[operator]["proposal_gain_sum"] += gain
                shortlists = [sorted(items, key=lambda item: item[0], reverse=True)[:config.shortlist] for items in proposals]
                confirmation_team_count = 0
                confirmation_transitions = 0
                validation_transitions = 0
                for victim_value in rng.permutation(config.residents):
                    victim = int(victim_value)
                    shortlist = shortlists[victim]
                    if not shortlist:
                        continue
                    # Each sequential decision gets independent screening and
                    # validation suites, even within this generation.
                    screening_seeds = seeds(config.seed, 1000 + 2 * victim, generation, config.confirmation_episodes)
                    validation_seeds = seeds(config.seed, 1001 + 2 * victim, generation, config.validation_episodes)
                    confirmation_teams = [population.copy()]
                    for _, candidate, _ in shortlist:
                        replaced = population.copy()
                        replaced[victim] = candidate
                        confirmation_teams.append(replaced)
                    confirmation_team_count += len(confirmation_teams)
                    before = evaluator.evaluation_transitions
                    confirmation_returns = evaluator.evaluate(confirmation_teams, screening_seeds, confirmation_horizon)
                    confirmation_transitions += evaluator.evaluation_transitions - before
                    screening_gains = np.mean(confirmation_returns[1:] - confirmation_returns[0], axis=1)
                    winner = int(np.argmax(screening_gains))
                    _, candidate, operator = shortlist[winner]
                    screening_gain = float(screening_gains[winner])
                    operators[operator]["screening_gain_sum"] += screening_gain
                    # The screen selects only. Independent full-horizon paired
                    # validation, never the winning screen, decides acceptance.
                    before = evaluator.evaluation_transitions
                    validation_returns = evaluator.evaluate(
                        [population.copy(), confirmation_teams[winner + 1]], validation_seeds, config.horizon
                    )
                    validation_transitions += evaluator.evaluation_transitions - before
                    decision = paired_validation(validation_returns[0], validation_returns[1], critical_value)
                    operators[operator]["paired_gain_sum"] += decision["paired_mean_gain"]
                    if decision["accepted"]:
                        population[victim] = candidate
                        row["accepted"] += 1
                        operators[operator]["accepted"] += 1
                        operators[operator]["accepted_gain_sum"] += decision["paired_mean_gain"]
                    row["decisions"].append({
                        "victim": victim, "operator": operator, "screening_gain": screening_gain,
                        "screening_seeds": screening_seeds, "validation_seeds": validation_seeds, **decision,
                    })
                row.update({
                    "proposal_seeds": proposal_seeds,
                    "proposal_horizon": proposal_horizon,
                    "confirmation_horizon": confirmation_horizon,
                    "validation_horizon": config.horizon,
                    "validation_alpha": validation_alpha,
                    "validation_critical_value": critical_value,
                    "proposal_mean_return": incumbent_mean,
                    "proposal_transitions": proposal_transitions,
                    "confirmation_transitions": confirmation_transitions,
                    "validation_transitions": validation_transitions,
                    "confirmation_team_count": confirmation_team_count,
                    "proposal_gains": [max((item[0] for item in items), default=0.0) for items in proposals],
                    "operators": operators,
                })
                writer.add_scalar("collective/proposal_return", incumbent_mean, generation)
                for operator, values in operators.items():
                    for name, value in values.items():
                        writer.add_scalar(f"operators/{operator}/{name}", value, generation)

            def boundary_reason() -> dict[str, Any] | None:
                if config.total_transitions and generation > 0 and evaluator.evaluation_transitions >= config.total_transitions:
                    return {"kind": "transition_budget", "requested_transitions": config.total_transitions}
                if config.time_limit_seconds and time.monotonic() - started >= config.time_limit_seconds:
                    return {"kind": "time_limit", "limit_seconds": config.time_limit_seconds}
                if generation == config.generations:
                    return {"kind": "generation_limit", "limit_generations": config.generations}
                return None

            stop_reason = boundary_reason()
            development_returns = None
            development_transitions = 0
            if generation % config.development_every == 0 or stop_reason is not None:
                before = evaluator.evaluation_transitions
                development_returns = evaluator.evaluate([population], development_seeds, config.horizon)[0]
                development_transitions = evaluator.evaluation_transitions - before
                development_mean = float(np.mean(development_returns))
                if champion is None or development_mean > champion.mean_return:
                    champion = Champion(generation, development_mean, development_returns.tolist(), [genome.clone() for genome in population])
                plateau = cull_state.update(development_mean, config)
                stop_reason = boundary_reason()
                if stop_reason is None and plateau:
                    stop_reason = {"kind": "development_plateau", "stale_evaluations": cull_state.stale_evaluations,
                                   "patience": config.plateau_patience, "material_delta": config.plateau_material_delta}
                writer.add_scalar("charts/episodic_return", development_mean, generation)
                writer.add_scalar("charts/episodic_return_std", float(np.std(development_returns)), generation)
                writer.add_scalar("collective/development_ema", cull_state.ema, generation)
                row["development_seeds"] = development_seeds
                row["development_returns"] = development_returns.tolist()
                print(json.dumps({"event": "development", "generation": generation,
                                  "mean_return": development_mean, "ema": cull_state.ema,
                                  "champion_mean_return": champion.mean_return,
                                  "cumulative_evaluation_transitions": evaluator.evaluation_transitions,
                                  "elapsed_seconds": time.monotonic() - started, "stop_reason": stop_reason}), flush=True)
            row.update({
                "evaluation_transition_budget": evaluator.evaluation_transitions - start_transitions,
                "cumulative_evaluation_transitions": evaluator.evaluation_transitions,
                "development_transitions": development_transitions,
                "development_mean_return": None if development_returns is None else float(np.mean(development_returns)),
                "champion_mean_return": None if champion is None else champion.mean_return,
                "cull_state": asdict(cull_state),
                "stop_reason": stop_reason,
                "elapsed_seconds": time.monotonic() - started,
            })
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            writer.add_scalar("collective/accepted_replacements", row["accepted"], generation)
            writer.add_scalar("collective/evaluation_transitions", evaluator.evaluation_transitions, generation)
            writer.add_scalar("collective/elapsed_seconds", row["elapsed_seconds"], generation)
            writer.flush()
            save_state(run_dir, config, adapter, generation, population, champion, cull_state,
                       evaluator.evaluation_transitions, stop_reason)
            if stop_reason is not None:
                if stop_reason["kind"] == "development_plateau":
                    # A completed, checkpointed research cull is a successful
                    # exit (0), not a failed attempt for mlq to retry.
                    print(json.dumps({"event": "AUTOCULL", "generation": generation,
                                      "cull_state": asdict(cull_state), "stop_reason": stop_reason}), flush=True)
                break
    return run_dir


def evaluate_checkpoint(checkpoint: Path, episodes: int, horizon: int, fresh_seed: int | None) -> dict[str, Any]:
    config, adapter, population, generation = load_checkpoint(checkpoint)
    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA; CPU fallback is disabled")
    if episodes < 1 or horizon < 1 or (fresh_seed is not None and fresh_seed < 0):
        raise ValueError("episodes and horizon must be positive, seed nonnegative")
    seed = config.seed + 100003 if fresh_seed is None else fresh_seed
    evaluation_seeds = seeds(seed, 307, 0, episodes)
    returns = evaluate(config.env_id, population, adapter, evaluation_seeds, horizon, config.max_nodes,
                       config.authority, torch.device(config.device), config.env_threads)
    return {"checkpoint": str(checkpoint), "generation": generation, "episodes": episodes, "horizon": horizon,
            "seeds": evaluation_seeds, "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "returns": returns.tolist(), **checkpoint_semantics(checkpoint)}


def diagnose_checkpoint(checkpoint: Path, episodes: int, horizon: int) -> dict[str, Any]:
    config, adapter, population, generation = load_checkpoint(checkpoint)
    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("diagnostics require CUDA; CPU fallback is disabled")
    if episodes < 1 or horizon < 1:
        raise ValueError("episodes and horizon must be positive")
    device = torch.device(config.device)
    diagnostic_seeds = seeds(config.seed, 401, 0, episodes)
    team = evaluate(config.env_id, population, adapter, diagnostic_seeds, horizon, config.max_nodes, config.authority, device, config.env_threads)
    standalone = [evaluate(config.env_id, [genome], adapter, diagnostic_seeds, horizon, config.max_nodes, "uniform", device, config.env_threads) for genome in population]
    leave_one_out = None
    if len(population) > 1:
        leave_one_out = []
        for victim in range(len(population)):
            reduced = [genome for index, genome in enumerate(population) if index != victim]
            reduced_returns = evaluate(config.env_id, reduced, adapter, diagnostic_seeds, horizon, config.max_nodes, config.authority, device, config.env_threads)
            leave_one_out.append(float(np.mean(team) - np.mean(reduced_returns)))
    return {
        "checkpoint": str(checkpoint), "generation": generation, "episodes": episodes, "horizon": horizon,
        "seeds": diagnostic_seeds,
        "team_mean_return": float(np.mean(team)),
        "standalone_mean_returns": [float(np.mean(values)) for values in standalone],
        "leave_one_out_gains": leave_one_out,
        "leave_one_out_unavailable_reason": "singleton population has no nonempty reduced team" if leave_one_out is None else None,
        "comparative_residents": None if leave_one_out is None else [index for index, gain in enumerate(leave_one_out) if gain > 0.0],
        **checkpoint_semantics(checkpoint),
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V9 evolutionary fixed-slot collective control")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    for field, field_value in Config.__dataclass_fields__.items():
        option = "--" + field.replace("_", "-")
        train_parser.add_argument(option, type=type(field_value.default), default=field_value.default)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--episodes", type=int, default=64)
    eval_parser.add_argument("--horizon", type=int, default=1000)
    eval_parser.add_argument("--seed", type=int)
    diag_parser = subparsers.add_parser("diagnose")
    diag_parser.add_argument("--checkpoint", type=Path, required=True)
    diag_parser.add_argument("--episodes", type=int, default=16)
    diag_parser.add_argument("--horizon", type=int, default=1000)
    return parser


def main() -> None:
    args = parser().parse_args()
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    if args.command == "train":
        run_dir = train(Config(**{key: value for key, value in vars(args).items() if key != "command"}))
        print(json.dumps({"run_dir": str(run_dir)}))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.episodes, args.horizon, args.seed), indent=2))
    else:
        print(json.dumps(diagnose_checkpoint(args.checkpoint, args.episodes, args.horizon), indent=2))


if __name__ == "__main__":
    main()

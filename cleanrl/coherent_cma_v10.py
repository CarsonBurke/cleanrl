"""Full-covariance CMA-ES over one compiled, observation-conditioned affine policy.

No gradients, critics, ensembles, reward normalization, or final-test selection.
Thousands of HalfCheetah return is a research target, not a guaranteed outcome.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import cma
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cleanrl.collective_control.control import ObservationAdapter, calibrate_observations
from cleanrl.collective_control.evolve import CullState, dump_json, seeds
from cleanrl.coherent_control.policy import AffineEvaluator
from cleanrl.shared.runtime import configure_runtime


POLICY_CONTRACT = {
    "parameter_layout": "C-order action-major W[action_dim,observation_dim], then b[action_dim]",
    "observation": "z=(observation-adapter.mean)/adapter.scale; fixed calibration; no tanh",
    "action": "clip((low+high)/2 + (high-low)/2 * (W @ z + b), low, high)",
    "blind": "replace observation with adapter.mean, leaving bias and action mapping unchanged",
    "return": "sum of raw Gymnasium rewards until first termination/truncation or horizon",
    "transitions": "actual stepped vector slots, including inactive lanes; calibration separate",
}


@dataclass
class Config:
    env_id: str = "HalfCheetah-v4"
    run_dir: str = "runs/coherent_cma_v10"
    sigma: float = 0.1
    population: int = 64
    generations: int = 10000
    total_transitions: int = 128_000_000
    horizon: int = 1000
    train_episodes: int = 2
    development_episodes: int = 64
    development_every: int = 10
    final_episodes: int = 128
    plateau_warmup_evaluations: int = 20
    plateau_patience: int = 30
    plateau_material_delta: float = 5.0
    plateau_decay: float = 0.8
    calibration_episodes: int = 4
    calibration_steps: int = 128
    seed: int = 1
    time_limit_seconds: int = 0
    device: str = "cuda"
    env_threads: int = 8


@dataclass
class Champion:
    generation: int
    source: str
    parameters: np.ndarray
    returns: np.ndarray

    @property
    def mean_return(self) -> float:
        return float(self.returns.mean())

    def to_json(self) -> dict[str, Any]:
        return {"generation": self.generation, "source": self.source,
                "parameters": self.parameters.tolist(), "returns": self.returns.tolist(),
                "mean_return": self.mean_return}


def validate_config(config: Config) -> None:
    for name in ("horizon", "train_episodes", "development_episodes", "development_every",
                 "final_episodes", "calibration_episodes", "calibration_steps", "env_threads"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("generations", "total_transitions", "seed", "time_limit_seconds",
                 "plateau_warmup_evaluations", "plateau_patience"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if config.population < 4:
        raise ValueError("population must be at least four for covariance adaptation")
    if not np.isfinite(config.sigma) or config.sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not 0 <= config.plateau_decay < 1:
        raise ValueError("plateau_decay must be in [0,1)")
    if not np.isfinite(config.plateau_material_delta) or config.plateau_material_delta < 0:
        raise ValueError("plateau_material_delta must be finite and nonnegative")
    if config.final_episodes < 2:
        raise ValueError("final_episodes must be at least two for paired SEM")
    if torch.device(config.device).type != "cuda":
        raise ValueError("coherent CMA requires CUDA; no CPU policy fallback")


def make_optimizer(config: Config, parameter_dim: int) -> cma.CMAEvolutionStrategy:
    # Explicitly retain pycma's active, full-covariance algorithm. No rank-one
    # winner-only acceptance gate: every population fitness reaches tell().
    rng = np.random.default_rng(config.seed)
    return cma.CMAEvolutionStrategy(np.zeros(parameter_dim), config.sigma, {
        # pycma treats seed=0 as time-seeded. Own the sampler instead, so every
        # accepted seed is reproducible and unrelated NumPy draws cannot move it.
        "popsize": config.population, "seed": np.nan,
        "randn": lambda *shape: rng.standard_normal(shape),
        "CMA_active": True, "CMA_diagonal": False, "verbose": -9,
        "verb_log": 0, "verb_disp": 0,
    })


def select_champion(champion: Champion | None, parameters: np.ndarray,
                    returns: np.ndarray, generation: int, sources: list[str]) -> Champion:
    index = int(np.argmax(returns.mean(axis=1)))
    if champion is None or float(returns[index].mean()) > champion.mean_return:
        return Champion(generation, sources[index], parameters[index].copy(), returns[index].copy())
    return champion


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def save_state(run_dir: Path, config: Config, adapter: ObservationAdapter, evaluator: AffineEvaluator,
               generation: int, mean: np.ndarray, champion: Champion, cull: CullState,
               stop_reason: dict[str, Any] | None, optimizer: cma.CMAEvolutionStrategy) -> None:
    state = {
        "schema": 1, "algorithm": "coherent_cma_v10", "config": asdict(config),
        "policy_contract": POLICY_CONTRACT, "observation_adapter": adapter.to_json(),
        "observation_dim": evaluator.observation_dim, "action_dim": evaluator.action_dim,
        "parameter_dim": evaluator.parameter_dim,
        "action_low": evaluator.action_low.tolist(), "action_high": evaluator.action_high.tolist(),
        "generation": generation, "parameters": mean.tolist(), "parameter_source": "cma_mean",
        "cumulative_evaluation_transitions": evaluator.evaluation_transitions,
        "calibration_seeds": seeds(config.seed, 11, 0, config.calibration_episodes),
        "calibration_transition_upper_bound": config.calibration_episodes * config.calibration_steps,
        "development_seeds": seeds(config.seed, 31, 0, config.development_episodes),
        "champion": champion.to_json(), "cull_state": asdict(cull), "stop_reason": stop_reason,
        "sigma": float(optimizer.sigma), "covariance_condition": float(optimizer.condition_number),
        "optimizer_state_saved": False,
    }
    dump_json(run_dir / "latest.json", state)
    dump_json(run_dir / "champion.json", {
        **state, "generation": champion.generation, "training_generation": generation,
        "parameters": champion.parameters.tolist(), "parameter_source": champion.source,
        "returns": champion.returns.tolist(), "mean_return": champion.mean_return,
    })


def load_checkpoint(path: Path) -> tuple[Config, ObservationAdapter, np.ndarray, dict[str, Any]]:
    state = json.loads(path.read_text())
    if state.get("schema") != 1 or state.get("algorithm") != "coherent_cma_v10":
        raise ValueError("not a coherent CMA v10 checkpoint")
    if state.get("policy_contract") != POLICY_CONTRACT:
        raise ValueError("checkpoint policy contract does not match this implementation")
    config = Config(**state["config"])
    validate_config(config)
    adapter = ObservationAdapter.from_json(state["observation_adapter"])
    parameters = np.asarray(state["parameters"], dtype=np.float64)
    obs_dim, act_dim = state["observation_dim"], state["action_dim"]
    if state["parameter_dim"] != act_dim * (obs_dim + 1) or parameters.shape != (state["parameter_dim"],):
        raise ValueError("checkpoint parameter layout is inconsistent")
    if (adapter.mean.shape != (obs_dim,) or adapter.scale.shape != (obs_dim,)
            or not np.isfinite(adapter.mean).all() or not np.isfinite(adapter.scale).all()
            or (adapter.scale <= 0).any() or not np.isfinite(parameters).all()):
        raise ValueError("checkpoint parameters or observation adapter are invalid")
    low, high = np.asarray(state["action_low"]), np.asarray(state["action_high"])
    if (low.shape != (act_dim,) or high.shape != (act_dim,) or not np.isfinite(low).all()
            or not np.isfinite(high).all() or not (low < high).all()):
        raise ValueError("checkpoint action bounds are invalid")
    return config, adapter, parameters, state


def run_evolution(config: Config, evaluator: AffineEvaluator, adapter: ObservationAdapter,
                  run_dir: Path, writer: SummaryWriter) -> tuple[Champion, dict[str, Any]]:
    """Run ask/tell with generation-boundary stops and mandatory final development.

    The evaluator boundary is injectable for orchestration tests; production
    always supplies the compiled CUDA AffineEvaluator. Never consumes final seeds.
    """
    optimizer = make_optimizer(config, evaluator.parameter_dim)
    development_seeds = seeds(config.seed, 31, 0, config.development_episodes)
    cull = CullState()
    champion = None
    started = time.monotonic()
    with (run_dir / "metrics.jsonl").open("w") as metrics:
        for generation in range(config.generations + 1):
            before = evaluator.evaluation_transitions
            train_returns = None
            best_candidate = None
            train_seeds = None
            if generation:
                population = optimizer.ask()
                parameters = np.asarray(population, dtype=np.float64)
                train_seeds = seeds(config.seed, 101, generation, config.train_episodes)
                train_returns = evaluator.evaluate(parameters, train_seeds, config.horizon)
                if not np.isfinite(train_returns).all():
                    raise FloatingPointError("nonfinite raw training returns; refusing invalid CMA fitness")
                fitness = -train_returns.mean(axis=1)
                best_candidate = parameters[int(np.argmin(fitness))].copy()
                optimizer.tell(population, fitness.tolist())
            train_transitions = evaluator.evaluation_transitions - before

            def boundary_reason() -> dict[str, Any] | None:
                numerical_stop = _plain(dict(optimizer.stop()))
                if numerical_stop:
                    return {"kind": "cma_stop", "library_reasons": numerical_stop}
                if config.total_transitions and evaluator.evaluation_transitions >= config.total_transitions:
                    return {"kind": "transition_budget", "requested_transitions": config.total_transitions}
                if config.time_limit_seconds and time.monotonic() - started >= config.time_limit_seconds:
                    return {"kind": "time_limit", "limit_seconds": config.time_limit_seconds}
                if generation == config.generations:
                    return {"kind": "generation_limit", "limit_generations": config.generations}
                return None

            stop_reason = boundary_reason()
            development_returns = None
            development_sources = []
            before_dev = evaluator.evaluation_transitions
            if generation % config.development_every == 0 or stop_reason is not None:
                candidates = [np.asarray(optimizer.mean).copy()]
                development_sources = ["cma_mean"]
                if best_candidate is not None:
                    candidates.append(best_candidate)
                    development_sources.append("population_best")
                candidate_parameters = np.asarray(candidates)
                development_returns = evaluator.evaluate(candidate_parameters, development_seeds, config.horizon)
                if not np.isfinite(development_returns).all():
                    raise FloatingPointError("nonfinite raw development returns")
                champion = select_champion(champion, candidate_parameters, development_returns,
                                           generation, development_sources)
                development_mean = float(development_returns.mean(axis=1).max())
                plateau = cull.update(development_mean, config)
                stop_reason = boundary_reason()
                if stop_reason is None and plateau:
                    stop_reason = {"kind": "development_plateau", "stale_evaluations": cull.stale_evaluations,
                                   "patience": config.plateau_patience,
                                   "material_delta": config.plateau_material_delta}
                print(json.dumps({"event": "development", "generation": generation,
                                  "mean_returns": development_returns.mean(axis=1).tolist(),
                                  "sources": development_sources, "ema": cull.ema,
                                  "champion_mean_return": champion.mean_return,
                                  "cumulative_evaluation_transitions": evaluator.evaluation_transitions,
                                  "elapsed_seconds": time.monotonic() - started, "stop_reason": stop_reason}), flush=True)
                writer.add_scalar("charts/episodic_return", development_mean, generation)
                writer.add_scalar("cma/development_ema", cull.ema, generation)
                for source, values in zip(development_sources, development_returns):
                    writer.add_scalar(f"development/{source}_return", float(values.mean()), generation)
                    writer.add_scalar(f"development/{source}_std", float(values.std()), generation)
            row = {
                "generation": generation, "cumulative_evaluation_transitions": evaluator.evaluation_transitions,
                "train_transitions": train_transitions,
                "development_transitions": evaluator.evaluation_transitions - before_dev,
                "train_seeds": train_seeds,
                "train_return": None if train_returns is None else float(train_returns.mean()),
                "train_std": None if train_returns is None else float(train_returns.std()),
                "train_best_return": None if train_returns is None else float(train_returns.mean(axis=1).max()),
                "development_seeds": None if development_returns is None else development_seeds,
                "development_sources": development_sources,
                "development_returns": None if development_returns is None else development_returns.tolist(),
                "champion_mean_return": champion.mean_return,
                "sigma": float(optimizer.sigma), "covariance_condition": float(optimizer.condition_number),
                "elapsed_seconds": time.monotonic() - started,
                "cull_state": asdict(cull), "stop_reason": stop_reason,
            }
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            for name in ("cumulative_evaluation_transitions", "train_return", "train_std", "train_best_return",
                         "champion_mean_return", "sigma", "covariance_condition", "elapsed_seconds"):
                if row[name] is not None:
                    writer.add_scalar(f"cma/{name}", row[name], generation)
            writer.add_scalar("cma/stale_evaluations", cull.stale_evaluations, generation)
            writer.add_scalar("cma/stopped", int(stop_reason is not None), generation)
            if stop_reason is not None:
                writer.add_text("cma/stop_reason", json.dumps(stop_reason), generation)
            writer.flush()
            save_state(run_dir, config, adapter, evaluator, generation, np.asarray(optimizer.mean),
                       champion, cull, stop_reason, optimizer)
            if stop_reason is not None:
                if stop_reason["kind"] == "development_plateau":
                    print(json.dumps({"event": "AUTOCULL", "generation": generation,
                                      "cull_state": asdict(cull), "stop_reason": stop_reason}), flush=True)
                return champion, {"generation": generation, "stop_reason": stop_reason,
                                  "cumulative_evaluation_transitions": evaluator.evaluation_transitions,
                                  "transition_budget_overshoot": max(0, evaluator.evaluation_transitions - config.total_transitions)
                                  if config.total_transitions else 0,
                                  "elapsed_seconds": time.monotonic() - started}
    raise RuntimeError("evolution ended without a generation-boundary stop")


def evaluate_final(evaluator: AffineEvaluator, parameters: np.ndarray, seed: int,
                   episodes: int, horizon: int) -> dict[str, Any]:
    if episodes < 2 or horizon < 1:
        raise ValueError("final evaluation needs at least two episodes and a positive horizon")
    final_seeds = seeds(seed, 701, 0, episodes)
    before = evaluator.evaluation_transitions
    returns = evaluator.evaluate(parameters[None, :], final_seeds, horizon)[0]
    blind_returns = evaluator.evaluate(parameters[None, :], final_seeds, horizon, blind=True)[0]
    if not np.isfinite(returns).all() or not np.isfinite(blind_returns).all():
        raise FloatingPointError("nonfinite final returns")
    differences = returns - blind_returns
    return {
        "evaluation_role": "heldout only; never used for champion or hyperparameter selection",
        "seed": seed, "seed_namespace": 701, "seeds": final_seeds, "episodes": episodes, "horizon": horizon,
        "returns": returns.tolist(), "mean_return": float(returns.mean()), "std_return": float(returns.std()),
        "sem_return": float(returns.std(ddof=1) / np.sqrt(episodes)),
        "blind_returns": blind_returns.tolist(), "blind_mean_return": float(blind_returns.mean()),
        "blind_std_return": float(blind_returns.std()), "paired_gains": differences.tolist(),
        "blind_paired_gain": float(differences.mean()),
        "blind_paired_sem": float(differences.std(ddof=1) / np.sqrt(episodes)),
        "final_evaluation_transitions": evaluator.evaluation_transitions - before,
    }


def write_manifest(run_dir: Path, config: Config) -> None:
    root = Path(__file__).resolve().parent.parent
    paths = [Path(__file__).resolve(), root / "cleanrl/coherent_control/policy.py",
             root / "cleanrl/collective_control/control.py", root / "cleanrl/collective_control/evolve.py",
             root / "cleanrl/shared/runtime.py", root / "cleanrl/shared/mujoco_env.py",
             root / "cleanrl/shared/rollout_graph.py"]
    dump_json(run_dir / "manifest.json", {
        "algorithm": "coherent_cma_v10", "config": asdict(config), "policy_contract": POLICY_CONTRACT,
        "cma_version": cma.__version__, "numpy_version": np.__version__, "torch_version": torch.__version__,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "selection": "fixed development suite compares current CMA mean and current train-best candidate; strict improvement",
        "optimizer": "pycma active full covariance; zero mean; ask all candidates, tell all negative mean raw returns",
        "calibration_seeds": seeds(config.seed, 11, 0, config.calibration_episodes),
        "calibration_transition_upper_bound": config.calibration_episodes * config.calibration_steps,
        "development_seeds": seeds(config.seed, 31, 0, config.development_episodes),
        "final_seeds": seeds(config.seed, 701, 0, config.final_episodes),
        "limitations": ["No return threshold is guaranteed.", "Development scores are adaptively selected, not heldout estimates.",
                        "latest.json stores the mean, not a resumable optimizer state.",
                        "Calibration helper reports no actual count; its step upper bound is recorded separately."],
    })


def train(config: Config) -> Path:
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(run_dir, config)
    adapter = calibrate_observations(config.env_id, seeds(config.seed, 11, 0, config.calibration_episodes),
                                     config.calibration_steps, config.env_threads)
    with SummaryWriter(str(run_dir / "tensorboard")) as writer, AffineEvaluator(
        config.env_id, adapter, torch.device(config.device), config.env_threads
    ) as evaluator:
        champion, summary = run_evolution(config, evaluator, adapter, run_dir, writer)
        # Champion and stop checkpoint are frozen before the test suite is touched.
        _, _, saved_parameters, _ = load_checkpoint(run_dir / "champion.json")
        result = evaluate_final(evaluator, saved_parameters, config.seed, config.final_episodes, config.horizon)
        result.update({"algorithm": "coherent_cma_v10", "config": asdict(config),
                       "checkpoint": str(run_dir / "champion.json"), "champion": champion.to_json(),
                       "training": summary, "total_evaluation_transitions": evaluator.evaluation_transitions,
                       "calibration_transition_upper_bound": config.calibration_episodes * config.calibration_steps})
        dump_json(run_dir / "final_result.json", result)
        writer.add_scalar("final/return", result["mean_return"], summary["generation"])
        writer.add_scalar("final/blind_paired_gain", result["blind_paired_gain"], summary["generation"])
        writer.add_scalar("final/blind_paired_sem", result["blind_paired_sem"], summary["generation"])
        print(json.dumps({"event": "final", "result": str(run_dir / "final_result.json"),
                          "mean_return": result["mean_return"], "blind_paired_gain": result["blind_paired_gain"],
                          "blind_paired_sem": result["blind_paired_sem"]}), flush=True)
    return run_dir


def evaluate_checkpoint(checkpoint: Path, episodes: int = 128, horizon: int | None = None,
                        fresh_seed: int | None = None) -> dict[str, Any]:
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    config, adapter, parameters, state = load_checkpoint(checkpoint)
    with AffineEvaluator(config.env_id, adapter, torch.device(config.device), config.env_threads) as evaluator:
        if (evaluator.parameter_dim != state["parameter_dim"]
                or evaluator.observation_dim != state["observation_dim"] or evaluator.action_dim != state["action_dim"]
                or not np.array_equal(evaluator.action_low, np.asarray(state["action_low"], dtype=np.float32))
                or not np.array_equal(evaluator.action_high, np.asarray(state["action_high"], dtype=np.float32))):
            raise ValueError("environment dimensions or action bounds differ from checkpoint")
        result = evaluate_final(evaluator, parameters, config.seed if fresh_seed is None else fresh_seed,
                                episodes, config.horizon if horizon is None else horizon)
    return {**result, "checkpoint": str(checkpoint), "generation": state["generation"],
            "parameter_source": state["parameter_source"], "policy_contract": POLICY_CONTRACT}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    for name, field in Config.__dataclass_fields__.items():
        training.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--episodes", type=int, default=128)
    evaluation.add_argument("--horizon", type=int)
    evaluation.add_argument("--seed", type=int)
    evaluation.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        train(Config(**{name: value for name, value in vars(args).items() if name != "command"}))
    else:
        result = evaluate_checkpoint(args.checkpoint, args.episodes, args.horizon, args.seed)
        if args.output is not None:
            dump_json(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

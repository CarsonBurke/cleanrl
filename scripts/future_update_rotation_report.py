"""Compare future-credit update learners using existing harness scalars.

No training, model loading, or queue-state inference. Single-seed episode
statistics are descriptive, not confidence intervals over independent runs.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _runs import RETURN_TAG, RunScalars, find_runs, run_timestamp


METRICS = {
    "losses/exact_beta_kl_max", "losses/exact_beta_kl_last",
    "critic/decoded_mse", "charts/interval_SPS", "timing/update_s",
    "timing/rollout_s", "timing/env_s", "losses/entropy",
    "timing/controller_s", "rotation/preprojection_norm_error",
    "rotation/preprojection_cosine", "meta/rollout_reward", "meta/objective",
    "meta/advantage", "meta/score_norm", "meta/updates", "meta/windows",
    "meta/mean_first", "meta/mean_second", "meta/mean_readout",
}


def controller_evidence(run_dir, fixed):
    """Join outcomes to the action that generated them, not the next action.

    Boundary records contain the completed objective but the NEW window's
    angles. Retain the start record until all subsequent outcomes arrive.
    """
    records_path = run_dir / "metrics.jsonl"
    if not records_path.exists():
        return None
    start, rewards, windows = None, [], []
    groups = ("first", "second", "readout")
    horizon = fixed["meta_horizon"]
    scale, std = fixed["meta_reward_scale"], fixed["meta_std"]
    with records_path.open() as source:
        for line in source:
            if not line.endswith("\n"):
                break  # A live writer may not have finished its last record.
            row = json.loads(line)
            if start is None:
                start = row
                continue
            rewards.append(row["meta/rollout_reward"])
            if row["meta/windows"] == start["meta/windows"]:
                continue
            if row["meta/windows"] != start["meta/windows"] + 1 or len(rewards) != horizon:
                raise ValueError(f"Missing or reordered controller outcomes in {records_path}")
            credit = int(row["meta/credit_horizon"])
            objective = float(np.mean(rewards[:credit]))
            baseline = start["meta/baseline"]
            advantage = (objective - baseline) / scale
            if not np.isclose(objective, row["meta/objective"], rtol=1e-5, atol=1e-6):
                raise ValueError("Credited objective does not match its subsequent rollout rewards")
            if not np.isclose(advantage, row["meta/advantage"], rtol=1e-4, atol=1e-6):
                raise ValueError("Completed advantage does not use its pre-action baseline")
            residual = np.array([start[f"meta/angle_{g}"] - start[f"meta/mean_{g}"]
                                 for g in groups])
            windows.append({
                "step": row["step"], "objective": objective, "advantage": advantage,
                "residual": residual, "score_gradient": advantage * residual / std**2,
                "objective_error": abs(objective - row["meta/objective"]),
                "advantage_error": abs(advantage - row["meta/advantage"]),
            })
            start, rewards = row, []
    result = {
        "completed_windows": len(windows),
        "pairing": "Each start-window action is joined to its subsequent rollout outcomes; boundary-row new actions are never paired with old objectives.",
        "statistical_limit": "Score means and residual correlations are descriptive, nonstationary single-trajectory diagnostics, not independent-seed significance tests.",
        "partitions": {},
    }
    if not windows:
        return result
    result["max_objective_alignment_error"] = max(w["objective_error"] for w in windows)
    result["max_advantage_alignment_error"] = max(w["advantage_error"] for w in windows)
    for label, selected in (
        ("all", windows),
        ("first_half_of_training", [w for w in windows if w["step"] <= fixed["total_timesteps"] / 2]),
    ):
        if not selected:
            continue
        advantages = np.array([w["advantage"] for w in selected])
        residuals = np.stack([w["residual"] for w in selected])
        gradients = np.stack([w["score_gradient"] for w in selected])
        partition = {
            "windows": len(selected), "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()), "groups": {},
        }
        for index, group in enumerate(groups):
            correlation = None
            if len(selected) > 1 and residuals[:, index].std() > 0 and advantages.std() > 0:
                correlation = float(np.corrcoef(residuals[:, index], advantages)[0, 1])
            partition["groups"][group] = {
                "angle_noise_advantage_correlation": correlation,
                "score_gradient_mean": float(gradients[:, index].mean()),
                "score_gradient_std": float(gradients[:, index].std()),
            }
        result["partitions"][label] = partition
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("runs/future_update_rotation_v21_experiment.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/future_update_rotation_v21_results.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "selection": "Latest timestamp of each exact experiment name and seed, never best score.",
        "limits": "Single-seed training returns. No independent evaluation or statistical superiority claim. Logged extent is not queue completion.",
        "window_halfwidth_steps": 250_000,
        "analysis": manifest.get("analysis"),
        "contract_verification": manifest.get("contract_job"),
        "runs": {},
    }
    for arm in manifest["arms"]:
        name = arm["name"]
        pattern = f"__{name}__{manifest['fixed']['seed']}__"
        paths = find_runs([Path("runs")], [pattern], env_filter=manifest["fixed"]["environment"])
        entry = {"job_id": arm.get("job_id"), "queue_state": arm.get("queue_state"),
                 "meta_mode": arm["meta_mode"], "credit_horizon": arm["meta_credit_horizon"]}
        report["runs"][name] = entry
        if not paths:
            entry["available"] = False
            continue
        path = max(paths, key=lambda candidate: run_timestamp(candidate) or 0)
        scalars = RunScalars(path, tags={RETURN_TAG, *METRICS})
        steps, returns = scalars.series(RETURN_TAG)
        entry.update(available=bool(steps.size), path=str(path))
        if not steps.size:
            continue
        entry.update(last_episode_step=int(steps[-1]), episodes=int(returns.size),
                     last_100_mean=float(returns[-100:].mean()),
                     last_100_count=int(min(100, returns.size)),
                     last_100_episode_std=float(returns[-100:].std()),
                     all_training_episode_mean=float(returns.mean()))
        entry["matched_step_returns"] = {}
        for target in (1_000_000, 2_000_000, 5_000_000, 10_000_000,
                       20_000_000, 30_000_000, 40_000_000, 50_000_000):
            selected = (steps >= target - 250_000) & (steps <= target + 250_000)
            entry["matched_step_returns"][str(target)] = None if not selected.any() else {
                "mean": float(returns[selected].mean()), "episodes": int(selected.sum()),
                "partial_window": bool(steps[-1] < target + 250_000),
            }
        entry["metrics"] = {}
        for tag in sorted(METRICS):
            metric_steps, values = scalars.series(tag)
            finite = values[np.isfinite(values)]
            if finite.size:
                entry["metrics"][tag] = {
                    "last_step": int(metric_steps[-1]),
                    "latest": float(values[-1]) if np.isfinite(values[-1]) else None,
                    "mean": float(finite.mean()), "max": float(finite.max()),
                    "nonfinite_count": int(values.size - finite.size),
                }
        entry["controller_evidence"] = controller_evidence(path, manifest["fixed"])
        entry["learning_checkpoint_present"] = (path / "final_learning_checkpoint.pt").is_file()
        print(f"{name:40s} step={steps[-1]:>9,d} last100={entry['last_100_mean']:9.2f} "
              f"episodes={returns.size:>6d} queue={entry['queue_state']}")
    historical = manifest.get("historical_reference")
    identity = next((run for run in report["runs"].values()
                     if run["meta_mode"] == "none" and run.get("available")), None)
    if historical and identity:
        tags = {RETURN_TAG, "critic/decoded_mse", "grad/actor_postclip_norm",
                "losses/exact_beta_kl_max"}
        reference = RunScalars(Path(historical["run"]), tags=tags)
        current = RunScalars(Path(identity["path"]), tags=tags)
        fidelity = {}
        for tag in sorted(tags):
            old_steps, old_values = reference.series(tag)
            new_steps, new_values = current.series(tag)
            count = min(old_values.size, new_values.size)
            aligned = bool(np.array_equal(old_steps[:count], new_steps[:count]))
            fidelity[tag] = {
                "compared_prefix_records": int(count), "steps_aligned": aligned,
                "max_absolute_error": float(np.max(np.abs(old_values[:count] - new_values[:count])))
                if count and aligned else None,
            }
        report["identity_fidelity_against_historical_reference"] = fidelity
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

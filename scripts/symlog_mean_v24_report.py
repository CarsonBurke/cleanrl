"""Report exact-name seed-1 critic ablations and a manifest-selected reference.

Reads TensorBoard scalars and checkpoint existence only. Queue states come solely
from the manifest; missing or partial logs never establish a job outcome.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _runs import RETURN_TAG, RunScalars, find_runs, fmt_step, run_timestamp


HISTORICAL_EXP_NAME = "residual_stiglu_ngpt_scaled_noadv_novalueclip_50M_v7"
MATCHED_STEPS = (1_000_000, 2_000_000, 5_000_000, 10_000_000,
                 20_000_000, 30_000_000, 40_000_000, 50_000_000)
WINDOW_HALF_WIDTH = 250_000
METRICS = {
    "losses/policy_loss", "losses/value_loss", "losses/entropy",
    "losses/explained_variance", "losses/old_approx_kl", "losses/approx_kl",
    "losses/clipfrac", "critic/decoded_mse", "critic/preupdate_mse",
    "critic/postupdate_ev", "critic/coordinate_min", "critic/coordinate_max",
    "critic/coordinate_abs_max", "critic/coordinate_finite_fraction",
    "critic/decoded_finite_fraction", "grad/total_preclip_norm",
    "grad/total_postclip_norm", "grad/actor_postclip_norm", "grad/critic_postclip_norm",
    "charts/SPS", "charts/interval_SPS", "charts/learning_rate",
    "timing/update_s", "timing/rollout_s", "timing/env_s", "timing/gae_s",
    "critic/target_fraction", "critic/target_count", "critic/target_mean",
    "critic/target_std", "critic/target_finite_fraction",
    "popart/mean", "popart/std", "popart/raw_preservation_max_error",
    "critic/normalized_mse",
}


def sample_summary(steps, values):
    """Do not hide nonfinite observations by averaging only surviving samples."""
    count = int(values.size)
    nonfinite_count = int(np.count_nonzero(~np.isfinite(values)))
    valid = count > 0 and nonfinite_count == 0
    return {
        "count": count,
        "nonfinite_count": nonfinite_count,
        "mean": float(values.mean()) if valid else None,
        "std": float(values.std()) if valid else None,
        "first_step": int(steps[0]) if count else None,
        "last_step": int(steps[-1]) if count else None,
    }


def matched_returns(steps, values):
    points = {}
    for target in MATCHED_STEPS:
        lo, hi = target - WINDOW_HALF_WIDTH, target + WINDOW_HALF_WIDTH
        selected = (steps >= lo) & (steps <= hi)
        points[str(target)] = {
            **sample_summary(steps[selected], values[selected]),
            "window_start": lo,
            "window_end": hi,
            "partial_window": bool(not steps.size or steps[0] > lo or steps[-1] < hi),
        }
    return points


def select_run(runs_dir, exp_name, environment, seed):
    candidates = find_runs(
        [runs_dir], [f"__{exp_name}__{seed}__"], env_filter=environment,
    )
    exact = []
    for path in candidates:
        parts = path.name.rsplit("__", 2)
        timestamp = run_timestamp(path)
        if (len(parts) == 3 and parts[0] == f"{environment}__{exp_name}"
                and parts[1] == str(seed) and timestamp is not None and np.isfinite(timestamp)):
            exact.append((timestamp, path.name, path))
    return max(exact)[2] if exact else None


def summarize_run(arm, runs_dir, environment, seed):
    exp_name = arm["exp_name"]
    if arm.get("run_dir"):
        path = Path(arm["run_dir"])
        parts = path.name.rsplit("__", 2)
        if (len(parts) != 3 or parts[0] != f"{environment}__{exp_name}"
                or parts[1] != str(seed) or run_timestamp(path) is None):
            raise ValueError(f"Explicit run does not match the requested experiment: {path}")
        if not path.is_dir():
            path = None
    else:
        path = select_run(runs_dir, exp_name, environment, seed)
    empty_steps = np.array([], dtype=np.int64)
    empty_values = np.array([], dtype=np.float64)
    entry = {
        **arm,
        "job": arm.get("job"),
        "state": arm.get("state"),
        "available": False,
        "log_status": "missing_run" if path is None else "no_return_scalars",
        "path": str(path) if path is not None else None,
        "run_timestamp": run_timestamp(path) if path is not None else None,
        "episodes": 0,
        "nonfinite_returns": 0,
        "last_episode_step": None,
        "last_logged_step": None,
        "last_100": {**sample_summary(empty_steps, empty_values), "partial_window": True},
        "matched_step_returns": matched_returns(empty_steps, empty_values),
        "metrics_not_matched": {},
        "checkpoints": {
            filename: {
                "path": str(path / filename) if path is not None else None,
                "exists": bool(path is not None and (path / filename).is_file()),
            }
            for filename in ("final_learning_checkpoint.pt", f"{exp_name}.cleanrl_model")
        },
    }
    if path is None:
        return entry
    try:
        scalars = RunScalars(path, tags={RETURN_TAG, *METRICS})
    except Exception as error:
        entry["log_status"] = "read_error"
        entry["read_error"] = {"type": type(error).__name__, "message": str(error)}
        return entry
    steps, returns = scalars.series(RETURN_TAG)
    if steps.size:
        entry.update(
            available=True,
            log_status="returns_present",
            episodes=int(returns.size),
            nonfinite_returns=int(np.count_nonzero(~np.isfinite(returns))),
            last_episode_step=int(steps[-1]),
            last_logged_step=int(steps[-1]),
            last_100={
                **sample_summary(steps[-100:], returns[-100:]),
                "partial_window": bool(returns.size < 100),
            },
            matched_step_returns=matched_returns(steps, returns),
        )
    for tag in sorted(METRICS):
        metric_steps, values = scalars.series(tag)
        if not values.size:
            continue
        last_step = int(metric_steps[-1])
        entry["last_logged_step"] = max(entry["last_logged_step"] or 0, last_step)
        entry["metrics_not_matched"][tag] = {
            "latest_step": last_step,
            "latest": float(values[-1]) if np.isfinite(values[-1]) else None,
            "count": int(values.size),
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
            "last_100": {
                **sample_summary(metric_steps[-100:], values[-100:]),
                "partial_window": bool(values.size < 100),
            },
        }
    return entry


def print_table(report):
    print(f"{'arm':34s} {'job':>6s} {'queue':14s} {'step':>7s} {'last100(n)':>15s} "
          + " ".join(f"{fmt_step(step):>8s}" for step in MATCHED_STEPS) + " checkpoint")
    reference = report["historical_reference"]
    for name, entry in [*report["runs"].items(), (reference.get("name", "reference"), reference)]:
        last = entry["last_100"]
        score = "-" if last["mean"] is None else f"{last['mean']:.1f}"
        score = f"{score}({last['count']})" + ("*" if last["partial_window"] else "")
        points = []
        for target in MATCHED_STEPS:
            point = entry["matched_step_returns"][str(target)]
            value = "-" if point["mean"] is None else f"{point['mean']:.0f}"
            points.append(f"{value + ('*' if point['partial_window'] and point['count'] else ''):>8s}")
        step = "-" if entry["last_episode_step"] is None else fmt_step(entry["last_episode_step"])
        job = "-" if entry["job"] is None else str(entry["job"])
        state = "unknown" if entry["state"] is None else str(entry["state"])
        checkpoint = "yes" if any(item["exists"] for item in entry["checkpoints"].values()) else "no"
        print(f"{name:34s} {job:>6s} {state:14s} {step:>7s} {score:>15s} "
              + " ".join(points) + f" {checkpoint}")
    print("* partial window; - missing/nonfinite score. Counts, log availability, and metric steps are in JSON.")
    print("Queue states are copied from the manifest, never inferred from logs or checkpoints.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("runs/symlog_mean_v24_experiment.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/symlog_mean_v24_results.json"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    config = manifest.get("config", {})
    environment, seed = config.get("env_id", "HalfCheetah-v4"), config.get("seed", 1)
    reference = {
        "name": "historical_v7", "exp_name": HISTORICAL_EXP_NAME,
        **manifest.get("reference", {}),
    }
    if seed != 1:
        raise ValueError("This matched experiment report requires seed 1")
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "environment": environment,
        "seed": seed,
        "selection": "Explicit run_dir when supplied; otherwise latest trailing timestamp for each exact environment, experiment name, and seed. Never best score or furthest log.",
        "queue_state_source": "Manifest job/state fields; the reporter does not query or infer job state.",
        "return_tag": RETURN_TAG,
        "return_count_unit": "training episodes",
        "window_halfwidth_steps": WINDOW_HALF_WIDTH,
        "matched_steps": list(MATCHED_STEPS),
        "limits": "Single-seed training returns, not independent evaluation. No statistical superiority claim. Losses in different critic coordinates are not directly comparable. Latest metric summaries are not matched horizons. Nonfinite observations invalidate their window mean. A 50M run generally has a partial 50M +/-250k window.",
        "config": config,
        "contract_verification": manifest.get("contract_job"),
        "monte_carlo_contract_verification": manifest.get("mc_contract_job"),
        "design": manifest.get("design"),
        "analysis": manifest.get("analysis"),
        "runs": {
            arm.get("name", arm["exp_name"]): summarize_run(arm, args.runs_dir, environment, seed)
            for arm in manifest["arms"]
        },
        "historical_reference": summarize_run(reference, args.runs_dir, environment, seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print_table(report)
    print(args.output)


if __name__ == "__main__":
    main()

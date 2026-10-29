"""Describe paired-run learning evidence; never control jobs or run networks."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _runs import RunScalars, find_runs

JOBS = {12016: "td7_latentq_paired_lambda_coverage_v24",
        12018: "td7_latentq_paired_lambda_center_v25",
        12022: "td7_latentq_paired_replay_coverage_v26"}


def series(run, tag):
    steps, values = run.series(tag)
    # Multiple optimizer logs share the same interaction step. For fitting
    # diagnostics there is one record/burst; last record resolves any repeats.
    return {int(s): float(v) for s, v in zip(steps, values) if np.isfinite(v)}


def compare(run, old_tag, new_tag):
    old, new = series(run, old_tag), series(run, new_tag)
    steps = sorted(old.keys() & new.keys())
    if not steps:
        return None
    a, b = np.array([old[s] for s in steps]), np.array([new[s] for s in steps])
    result = {"bursts": len(steps), "first_step": steps[0], "last_step": steps[-1],
              "old_mean_mse": float(a.mean()), "new_mean_mse": float(b.mean()),
              "improved_fraction": float((b < a).mean())}
    positive = a > 0
    result["median_burst_ratio"] = float(np.median(b[positive] / a[positive])) if positive.any() else None
    result["pooled_ratio"] = float(b.sum() / a.sum()) if a.sum() > 0 else None
    tail = min(20, len(a))
    result["last20_pooled_ratio"] = float(b[-tail:].sum() / a[-tail:].sum()) if a[-tail:].sum() > 0 else None
    return result


def summarize():
    results = []
    for job_id, name in JOBS.items():
        job = json.loads(subprocess.check_output(["mlq", "show", str(job_id), "--json"]))
        row = {key: job.get(key) for key in ("id", "name", "state", "stateReason", "maxParallelRuns", "timeLimitMs")}
        paths = find_runs([ROOT / "runs"], [name])
        if not paths:
            results.append(row)
            continue
        run = RunScalars(paths[-1])
        row["run"] = paths[-1].name
        row["counts"] = {}
        for tag in ("charts/SPS", "charts/ordinary_transitions", "charts/branch_transitions", "charts/branch_useful_transitions", "charts/optimizer_steps"):
            values = series(run, tag)
            if values:
                step = max(values)
                row["counts"][tag] = {"step": step, "value": values[step]}
        row["returns"] = {}
        for tag in ("charts/episodic_return", "eval/episodic_return"):
            steps, values = run.series(tag)
            if not len(steps):
                continue
            points = {}
            for target in (100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 8_000_000):
                partial_budget_tail = target == 8_000_000 and row["counts"].get("charts/SPS", {}).get("step", 0) >= target - 20_480
                if steps[-1] < target and not partial_budget_tail:
                    continue
                mask = (steps >= target - 100_000) & (steps <= target)
                if mask.any():
                    points[str(target)] = {"mean": float(values[mask].mean()), "samples": int(mask.sum()), "partial": bool(steps[-1] < target)}
            row["returns"][tag] = {"last_step": int(steps[-1]), "last20_mean": float(values[-20:].mean()), "trailing100k": points}
        row["fit_effects"] = {}
        for split in ("fit", "holdout"):
            for channel in ("odd", "even"):
                for clock in ("initial", "final"):
                    prefix = f"paired/postfit/{split}/clock_matrix/{channel}"
                    row["fit_effects"][f"{split}/{channel}/{clock}"] = compare(run, f"{prefix}_old_{clock}_mse", f"{prefix}_new_{clock}_mse")
        row["crossed_label_effects"] = {}
        for offset in (0, 8, 16, 31):
            for clock in ("initial", "latest"):
                (row["fit_effects"] if clock == "initial" else row["crossed_label_effects"])[f"branch_point/{offset}/{clock}"] = compare(run, f"branch_direct/prefit/offset_{offset:02d}/{clock}/point_mse", f"branch_direct/postfit/offset_{offset:02d}/{clock}/point_mse")
        for tag in ("actor/kl", "charts/sigma_mean", "losses/alpha"):
            values = series(run, tag)
            if values:
                row[tag] = {"last_step": max(values), "last_value": values[max(values)]}
        results.append(row)
    document = {"generated": datetime.now(timezone.utc).isoformat(),
                "limitations": ["Single seed; correlated bursts and eight temporally dependent heldout anchors per burst; lane0 main states still train SALE and actor.",
                                 "Same-label fitting effects are not accuracy against a true value oracle.",
                                 "Collected interaction counts can precede completed fitting/SPS logs.",
                                 "Pooled MSE ratios weight high-error bursts more; median ratios weight bursts equally.",
                                 "Continuation initial/latest comparisons also change labels; only paired clock matrix holds both clocks explicitly."],
                "jobs": results}
    (OUT / "paired_diagnostics.json").write_text(json.dumps(document, indent=2) + "\n")
    for row in results:
        metrics = row.get("returns", {}).get("charts/episodic_return", {})
        print(row["id"], row["state"], metrics.get("last_step"), round(metrics.get("last20_mean", float("nan"))), metrics.get("trailing100k", {})) if metrics else print(row["id"], row["state"])
    return document


if __name__ == "__main__":
    summarize()

"""Regenerate matched-step INTACT objective evidence from harness scalars.

This reads logs only; it never launches training or infers queue state from them.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _runs import RETURN_TAG, RunScalars, find_runs


VARIANTS = {
    "v5 PPO control": "jepa_intact_model_control_v5_none",
    "v6 real-return correction": "jepa_intact_real_corrected_gradient_v6_corrected",
    "v7 imagined H1 gradient": "jepa_intact_one_step_model_gradient_v7_model",
    "v8 factual goals H8": "intact_factual_goals_v8_h8_weighted",
    "v8 factual goals H1": "intact_factual_goals_v8_h1_weighted",
    "v9 action quotient WML": "intact_quotient_wml_v9_h8_weighted",
}
METRICS = {
    "selection/empirical_kl", "selection/ess_fraction",
    "goals/factual_to_sampled_law_kl", "goals/shuffled_action_nll_gap",
    "goals/prefit_nll", "goals/mixture_entropy", "goals/mean_component_std",
    "goals/factual_norm", "goals/sampled_norm", "goals/prescribed_norm",
    "losses/weighted_action_nll", "losses/explained_variance",
    "world/local_nll", "world/goal_nll", "world/latent_std",
    "drift/kl_all", "charts/interval_SPS", "policy/entropy", "policy/concentration",
}


def main():
    output = Path("docs/intact-objectives-results")
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": "HalfCheetah-v4", "seed": 1,
        "selection": "furthest logged return step per named variant, never highest return",
        "window_steps": 50_000, "smoothing_episodes": 20,
        "limits": "Single-seed training returns; no independent evaluation or statistical superiority claim. Log extent does not establish job completion.",
        "runs": {},
    }
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for label, pattern in VARIANTS.items():
        candidates = find_runs([Path("runs")], [pattern], env_filter="HalfCheetah-v4")
        selected = None
        for path in candidates:
            if path.name.rsplit("__", 2)[-2] != "1":
                continue
            run = RunScalars(path, tags={RETURN_TAG, *METRICS})
            if selected is None or run.max_step() > selected.max_step():
                selected = run
        if selected is None or selected.max_step() == 0:
            report["runs"][label] = {"available": False}
            continue
        steps, values = selected.series(RETURN_TAG)
        count = min(20, values.size)
        points = {}
        for target in (1_000_000, 2_000_000, 8_000_000):
            mask = (steps >= target - 50_000) & (steps <= target + 50_000)
            points[str(target)] = None if not mask.any() else {
                "mean": float(values[mask].mean()), "episodes": int(mask.sum()),
                "partial_window": bool(steps[-1] < target + 50_000),
            }
        latest_metrics = {}
        for tag in sorted(METRICS):
            metric_steps, metric_values = selected.series(tag)
            if metric_values.size and np.isfinite(metric_values[-1]):
                latest_metrics[tag] = {"step": int(metric_steps[-1]), "value": float(metric_values[-1])}
        report["runs"][label] = {
            "available": True, "path": str(selected.run_dir),
            "last_episode_step": int(steps[-1]),
            "last_20_mean": float(values[-count:].mean()), "tail_episodes": count,
            "matched_step_returns": points, "latest_metrics_not_matched": latest_metrics,
        }
        smooth = np.convolve(values, np.ones(count) / count, mode="valid")
        ax.plot(steps[count - 1:] / 1e6, smooth, label=label, linewidth=1.5)
    ax.set(xlabel="Environment transitions (millions)", ylabel="Training episode return (20-episode mean)",
           title="INTACT: real-return grounding versus imagined control")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output.with_suffix(".json"))


if __name__ == "__main__":
    main()

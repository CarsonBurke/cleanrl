"""Print the throughput and budget tables of 06_compute.md from compute_mine_runs.py's JSON.

Usage:
  .venv/bin/python benchmarks/reppo_rootcause/scripts/compute_tables.py DATA.json timing [SUBSTR...]
  .venv/bin/python benchmarks/reppo_rootcause/scripts/compute_tables.py DATA.json budget [SUBSTR...]
  .venv/bin/python benchmarks/reppo_rootcause/scripts/compute_tables.py DATA.json tags SUBSTR
"""
from __future__ import annotations

import json
import sys

BUDGET_ARGS = ("num_envs", "num_steps", "minibatch_size", "num_minibatches", "update_epochs", "critic_updates",
               "actor_updates", "target_refresh", "aux_horizon", "critic_hidden_dim", "critic_width",
               "matmul_precision", "bf16", "learning_rate", "gve_lambda")
CRITIC_TAGS = ("losses/critic_ce", "losses/q_mse", "losses/aux_loss", "diagnostics/advantage_corr_heldout",
               "diagnostics/advantage_std_ratio", "diagnostics/dqda_cos_consecutive",
               "diagnostics/bellman_rms_ratio", "losses/explained_variance")


def fmt(value, digits=0):
    return "-" if value is None else f"{value:.{digits}f}"


def select(records, needles):
    return [r for r in records if "error" not in r and (not needles or any(n in r["run"] for n in needles))]


def short(run):
    return run.split("__")[1]


def timing(records):
    print("run | max_step | SPS_last | SPS>1M | concurrent | per-interval phase medians (s)")
    for record in records:
        metrics = record.get("metrics", {})
        phases = {tag[len("timing/"):-2]: entry for tag, entry in metrics.items()
                  if tag.startswith("timing/") and tag.endswith("_s") and tag != "timing/warmup_s"}
        steps = next((entry.get("steps_per_log") for entry in phases.values()), None)
        cells = " ".join(f"{name}={entry['median']:.3f}" for name, entry in sorted(phases.items()))
        sps = metrics.get("charts/SPS", {}).get("last")
        print(f"{short(record['run'])} | {record.get('max_step', 0)/1e6:.2f}M | {fmt(sps)} | "
              f"{fmt(record.get('sps_after_1M'))} | {record.get('concurrent', '?')} | "
              f"steps/log={fmt(steps)} {cells}")


def budget(records):
    print("run | args | max_step | R@500k R@1M R@2M R@4M R@6M R@8M | last20 | SPS | critic metrics @1M/4M/8M")
    for record in records:
        args = record.get("args", {})
        shown = " ".join(f"{key}={args[key]}" for key in BUDGET_ARGS if key in args)
        returns = record.get("returns", {})
        cells = " ".join(fmt(returns.get(str(step))) for step in (500_000, 1_000_000, 2_000_000, 4_000_000,
                                                                 6_000_000, 8_000_000))
        metrics = record.get("metrics", {})
        critic = "; ".join(
            f"{tag.split('/')[1]}=" + "/".join(fmt(metrics[tag].get(str(step)), 3)
                                               for step in (1_000_000, 4_000_000, 8_000_000))
            for tag in CRITIC_TAGS if tag in metrics)
        print(f"{short(record['run'])} | {shown} | {record.get('max_step', 0)/1e6:.2f}M | {cells} | "
              f"{fmt(record.get('return_last20'))} | {fmt(metrics.get('charts/SPS', {}).get('last'))} | {critic}")


def main():
    records = json.load(open(sys.argv[1]))
    # Runs whose wall-clock spans overlap: shared-box throughput is not single-run throughput.
    spans = [(r["wall_start"], r["wall_end"]) for r in records if "wall_start" in r]
    for record in records:
        if "wall_start" in record:
            middle = 0.5 * (record["wall_start"] + record["wall_end"])
            record["concurrent"] = sum(start <= middle <= end for start, end in spans)
    mode, needles = sys.argv[2], sys.argv[3:]
    chosen = select(records, needles)
    if mode == "timing":
        timing(chosen)
    elif mode == "budget":
        budget(chosen)
    else:
        for record in chosen:
            print(short(record["run"]), sorted(record.get("metrics", {})))
            print({key: value for key, value in record.get("args", {}).items()})


if __name__ == "__main__":
    main()

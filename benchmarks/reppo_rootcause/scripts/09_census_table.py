"""Flip census, stage 1: one row per REPPO HalfCheetah run with the return trajectory, a trajectory fingerprint
(bit-identical runs share it), and the design features. Output: data/09/census_raw.tsv."""
import hashlib
import importlib
import re
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
common = importlib.import_module("09_common")

GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
FEATURES = ["num_envs", "num_steps", "gve_lambda", "kl_bound", "ent_target_mult", "ent_start", "temperature_source",
            "aux_horizon", "aux_loss_coef", "reward_loss_coef", "matmul_precision", "critic_hidden_dim",
            "critic_updates", "actor_updates", "minibatch_size", "num_bins", "buffer_iterations", "buffer_size",
            "num_heads", "pg_weight", "exact_targets", "init_snapshot", "resume", "learning_rate", "actor_min_std"]


def fingerprint(run, upto):
    steps, values = run.series(common.RETURN)
    mask = steps <= upto
    if mask.sum() < 20:
        return "short"
    return hashlib.md5(np.round(values[mask], 2).tobytes()).hexdigest()[:8]


def main():
    rows = []
    for run in common.load_runs():
        version = re.search(r"_v(\d+)", run.exp)
        steps, values = run.series(common.RETURN)
        row = {"exp": run.exp, "stamp": run.stamp, "version": int(version.group(1)) if version else -1,
               "last_step": run.last_step, "first_step": int(steps[0]),
               "fp500k": fingerprint(run, 500_000), "fp1M": fingerprint(run, 1_000_000),
               "fp2M": fingerprint(run, 2_000_000),
               "max_ret": float(np.max(np.convolve(values, np.ones(16) / 16, "valid"))) if len(values) >= 16 else np.nan,
               "last20": float(values[-20:].mean())}
        for mark in GRID:
            row[f"r{mark:g}M"] = run.at(common.RETURN, int(mark * 1e6))
        for feature in FEATURES:
            row[feature] = run.hyper.get(feature, "")
        rows.append(row)
    columns = list(rows[0])
    common.OUT.mkdir(parents=True, exist_ok=True)
    with open(common.OUT / "census_raw.tsv", "w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(f"{row[c]:.0f}" if isinstance(row[c], float) else str(row[c]) for c in columns) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()

"""Leading indicators of the trap. Cohort: distinct from-scratch tanh-Gaussian trajectories, 16 x 1024, single
actor, version >= 23, classified F (trapped) or U (upright) by 09_census.py. For each iteration and scalar,
report the class medians and the AUC (probability that a random F run has a larger value than a random U run;
0.5 = no separation, 0 or 1 = perfect). Output: stdout (tee to data/09/early_cohort.txt)."""
import csv
import importlib
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
common = importlib.import_module("09_common")

TAGS = ["return_median", "return_iqr", "losses/aux_loss", "losses/q_mse", "diagnostics/dqda_norm", "policy/std",
        "policy/entropy", "policy/mean_abs_action", "policy/saturation", "policy/kl_violation", "policy/kl_multiplier",
        "policy/temperature", "policy/q_pi", "grad/actor_preclip_norm", "grad/critic_preclip_norm",
        "diagnostics/advantage_corr_heldout", "diagnostics/advantage_std_ratio"]
ITERATIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 24, 30]


def auc(positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        return float("nan")
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * ties)


def value(run, tag, iteration, batch):
    if tag.startswith("return"):
        steps, values = run.series(common.RETURN)
        mask = (steps > (iteration - 1) * batch) & (steps <= iteration * batch)
        if mask.sum() < 4:
            return float("nan")
        return float(np.median(values[mask]) if tag == "return_median" else np.subtract(*np.percentile(values[mask], [75, 25])))
    _, values = run.series(tag)
    return float(values[iteration - 1]) if len(values) >= iteration else float("nan")


def main():
    census = list(csv.DictReader(open(common.OUT / "census.tsv"), delimiter="\t"))
    runs = {run.exp: run for run in common.load_runs()}
    cohort = [row for row in census if row["dist"] == "tanh" and row["envs_x_steps"] == "16x1024"
              and row["heads"] == "1" and int(row["version"]) >= 23 and row["label"] in "FU"]
    groups = {label: [runs[row["exp"]] for row in cohort if row["label"] == label] for label in "FU"}
    print(f"cohort: F = {len(groups['F'])}, U = {len(groups['U'])}")
    print("F:", ", ".join(run.exp.replace("reppo_", "") for run in groups["F"]))
    print("U:", ", ".join(run.exp.replace("reppo_", "") for run in groups["U"]))
    print("\niteration (x 16384 steps):".ljust(30) + " ".join(f"{i:7d}" for i in ITERATIONS))
    for tag in TAGS:
        table = {label: [[value(run, tag, i, 16384) for i in ITERATIONS] for run in groups[label]] for label in "FU"}
        table = {label: np.asarray(rows, dtype=float) for label, rows in table.items()}
        if np.isnan(table["F"]).all() or np.isnan(table["U"]).all():
            continue
        print(tag)
        for label in "FU":
            print(f"  median {label}".ljust(30) + " ".join(f"{v:7.3g}" for v in np.nanmedian(table[label], axis=0)))
        scores = []
        for column in range(len(ITERATIONS)):
            f, u = table["F"][:, column], table["U"][:, column]
            scores.append(auc(f[np.isfinite(f)], u[np.isfinite(u)]))
        print("  AUC (F > U)".ljust(30) + " ".join(f"{v:7.2f}" for v in scores))


if __name__ == "__main__":
    main()

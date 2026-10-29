"""Event 2 across runs: is the regime change tied to return level (gait speed) or to step count, and does it track
exploration noise? For every REPPO run that reaches a rolling-median return of 6k, report at the first crossing of
each return level L (6k, 8k, 10k, 11k): the step, and over the following 0.5M steps the crash share (episodes below
0.7 x the window median), pre-tanh std, q_mse, q_mse / target_mean^2, KL multiplier, actor pre-clip gradient norm,
held-out advantage correlation and temperature. Also the crash-onset step (first 0.5M bin with crash share >= 2%).
Output: data/09/regime_cohort.tsv and a stdout digest."""
import importlib
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
common = importlib.import_module("09_common")

LEVELS = [6000, 8000, 10000, 11000]
TAGS = {"std": "policy/std", "qmse": "losses/q_mse", "tgt": "losses/target_mean", "klmul": "policy/kl_multiplier",
        "agrad": "grad/actor_preclip_norm", "corr": "diagnostics/advantage_corr_heldout", "temp": "policy/temperature",
        "sat": "policy/saturation", "absmu": "policy/mean_abs_action", "ent": "policy/entropy"}


def crash_share(values):
    return float((values < 0.7 * np.median(values)).mean()) if len(values) >= 20 else float("nan")


def main():
    rows = []
    for run in common.load_runs():
        steps, values = run.series(common.RETURN)
        if len(values) < 300:
            continue
        rolling = np.median(sliding_window_view(values, 100), axis=1)
        if rolling.max() < 6000:
            continue
        row = {"exp": run.exp.replace("reppo_", ""), "first_M": steps[0] / 1e6, "last_M": steps[-1] / 1e6}
        onset = float("nan")
        for lo in np.arange(np.floor(steps[0] / 5e5) * 5e5, steps[-1], 5e5):
            mask = (steps >= lo) & (steps < lo + 5e5)
            if mask.sum() >= 100 and np.median(values[mask]) > 3000 and crash_share(values[mask]) >= 0.02:
                onset = lo
                break
        row["onset_M"] = onset / 1e6
        if np.isfinite(onset):
            mask = (steps >= onset) & (steps < onset + 5e5)
            row["onset_ret"] = float(np.median(values[mask]))
            row["onset_std"] = run.window(TAGS["std"], onset, onset + 5e5)
        else:
            row["onset_ret"] = row["onset_std"] = float("nan")
        for level in LEVELS:
            index = int(np.argmax(rolling > level))
            key = f"{level // 1000}k"
            if rolling[index] <= level:
                for name in ["step", "crash", *TAGS, "relq"]:
                    row[f"{name}@{key}"] = float("nan")
                continue
            start = float(steps[index + 99])
            mask = (steps >= start) & (steps < start + 5e5)
            row[f"step@{key}"] = start / 1e6
            row[f"crash@{key}"] = crash_share(values[mask])
            for name, tag in TAGS.items():
                row[f"{name}@{key}"] = run.window(tag, start, start + 5e5)
            row[f"relq@{key}"] = row[f"qmse@{key}"] / row[f"tgt@{key}"] ** 2 * 1e4
        rows.append(row)
    columns = list(rows[0])
    with open(common.OUT / "regime_cohort.tsv", "w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c]) for c in columns) + "\n")
    show = ["onset_M", "onset_ret", "step@6k", "crash@6k", "std@6k", "qmse@6k", "step@8k", "crash@8k", "std@8k", "qmse@8k",
            "klmul@8k", "agrad@8k", "corr@8k", "step@10k", "crash@10k", "std@10k", "qmse@10k", "klmul@10k", "agrad@10k"]
    print("exp".ljust(40) + "".join(c.rjust(10) for c in show))
    for row in rows:
        print(row["exp"][:39].ljust(40) + "".join(f"{row[c]:10.3g}" for c in show))


if __name__ == "__main__":
    main()

"""Event 2 alignment: for from-scratch tanh-Gaussian REPPO runs that reach 9k, locate the first 0.25M bin where
each marker crosses a threshold (actor pre-clip grad norm > 2, KL multiplier > 0.5, q_mse > 0.5, crash share
>= 3% in a 0.5M window with median > 5k) and report the step and the median return there. If the change is tied
to gait speed, return-at-crossing is tighter across runs than step-at-crossing (coefficient of variation).
Also: crash share and noise statistics by return band, pooled over runs. Output: stdout."""
import csv
import importlib
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
common = importlib.import_module("09_common")

MARKERS = [("agrad>2", "grad/actor_preclip_norm", 2.0), ("klmul>0.5", "policy/kl_multiplier", 0.5),
           ("qmse>0.5", "losses/q_mse", 0.5)]
BANDS = [(3000, 5000), (5000, 7000), (7000, 9000), (9000, 10500), (10500, 12000), (12000, 14500)]


def median_return(run, lo, hi):
    return run.window(common.RETURN, lo, hi, np.median)


def main():
    census = {row["exp"]: row for row in csv.DictReader(open(common.OUT / "census.tsv"), delimiter="\t")}
    runs = [run for run in common.load_runs() if run.exp in census and census[run.exp]["dist"] == "tanh"
            and census[run.exp]["heads"] == "1" and census[run.exp]["label"] in "UR"
            and census[run.exp]["envs_x_steps"] == "16x1024" and run.last_step >= 6e6]
    runs = [run for run in runs if median_return(run, run.last_step - 5e5, run.last_step + 1) >= 9000]
    print(f"{len(runs)} runs")
    header = "exp".ljust(34) + "".join(f"{name + ' M':>12s}{'ret':>7s}" for name, _, _ in MARKERS) + f"{'crash>=3% M':>12s}{'ret':>7s}{'std':>6s}"
    print(header)
    collected = {name: [] for name, _, _ in MARKERS}
    collected["crash"] = []
    for run in runs:
        cells = []
        for name, tag, threshold in MARKERS:
            hit = None
            for lo in np.arange(1e6, run.last_step, 2.5e5):
                if run.window(tag, lo, lo + 2.5e5) > threshold:
                    hit = lo
                    break
            if hit is None:
                cells.append(f"{'-':>12s}{'-':>7s}")
            else:
                ret = median_return(run, hit, hit + 2.5e5)
                collected[name].append((hit / 1e6, ret))
                cells.append(f"{hit / 1e6:12.2f}{ret:7.0f}")
        steps, values = run.series(common.RETURN)
        hit = None
        for lo in np.arange(1e6, run.last_step - 5e5, 2.5e5):
            mask = (steps >= lo) & (steps < lo + 5e5)
            window = values[mask]
            if len(window) > 100 and np.median(window) > 5000 and (window < 0.7 * np.median(window)).mean() >= 0.03:
                hit = lo
                break
        if hit is None:
            cells.append(f"{'-':>12s}{'-':>7s}{'-':>6s}")
        else:
            ret = median_return(run, hit, hit + 5e5)
            collected["crash"].append((hit / 1e6, ret))
            cells.append(f"{hit / 1e6:12.2f}{ret:7.0f}{run.window('policy/std', hit, hit + 5e5):6.2f}")
        print(run.exp.replace("reppo_", "")[:33].ljust(34) + "".join(cells))
    print("\ncoefficient of variation across runs (sd / mean) of step-at-crossing vs return-at-crossing")
    for name, pairs in collected.items():
        pairs = np.asarray(pairs)
        if len(pairs) < 4:
            continue
        print(f"  {name:10s} n={len(pairs):2d}  step: median {np.median(pairs[:, 0]):5.2f}M, CV {pairs[:, 0].std() / pairs[:, 0].mean():.2f}"
              f"   return: median {np.median(pairs[:, 1]):6.0f}, CV {pairs[:, 1].std() / pairs[:, 1].mean():.2f}")

    print("\npooled by return band (0.5M windows of every run above; median over windows [IQR])")
    print(f"{'band':>14s} {'windows':>8s} {'runs':>5s} {'crash share':>20s} {'std':>6s} {'q_mse':>7s} {'q_mse/tgt^2 e4':>15s} {'agrad':>6s} {'klmul':>6s} {'corr':>6s}")
    for lo_band, hi_band in BANDS:
        records, members = [], set()
        for run in runs:
            steps, values = run.series(common.RETURN)
            for lo in np.arange(5e5, run.last_step - 5e5, 5e5):
                mask = (steps >= lo) & (steps < lo + 5e5)
                window = values[mask]
                if len(window) < 100 or not lo_band <= np.median(window) < hi_band:
                    continue
                members.add(run.exp)
                qmse, target = run.window("losses/q_mse", lo, lo + 5e5), run.window("losses/target_mean", lo, lo + 5e5)
                records.append(((window < 0.7 * np.median(window)).mean(), run.window("policy/std", lo, lo + 5e5), qmse,
                                qmse / target ** 2 * 1e4, run.window("grad/actor_preclip_norm", lo, lo + 5e5),
                                run.window("policy/kl_multiplier", lo, lo + 5e5),
                                run.window("diagnostics/advantage_corr_heldout", lo, lo + 5e5)))
        if not records:
            continue
        records = np.asarray(records, dtype=float)
        q25, q50, q75 = np.nanpercentile(records[:, 0], [25, 50, 75])
        rest = np.nanmedian(records[:, 1:], axis=0)
        print(f"{lo_band:6d}-{hi_band:<7d} {len(records):8d} {len(members):5d} {q50:8.3f} [{q25:.3f}, {q75:.3f}] "
              + " ".join(f"{v:6.3g}" for v in rest[:2]) + f" {rest[2]:15.2f} " + " ".join(f"{v:6.3g}" for v in rest[3:]))


if __name__ == "__main__":
    main()

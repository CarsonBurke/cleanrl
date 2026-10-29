"""Flip census. One row per DISTINCT from-scratch REPPO trajectory (bit-identical runs up to 300k steps are one
draw), classified from the per-episode return record, then cross-tabulated against design features.

Posture classes (from the return record; validated against snapshot posture in 09_obsnorm_posture.py):
  F  trapped: never leaves the back-running curve (rolling 48-episode median never > 2800) and ends with the
     back-running signature (median 1.4k-2.7k, IQR < 120).
  R  recovered: shows the signature (a 100k window from 0.6M on with median 1.0k-2.3k and IQR < 120) for at least
     one window before the rolling median first exceeds 2800.
  M  mixed at the end: left the curve but >= 10% of the last 200 episodes sit in the back-running band
     (1.2k-2.8k) while the p90 episode is above 3.5k, i.e. upright and inverted episodes coexist.
  U  upright: left the curve, no signature window, unimodal at the end.
  X  not classifiable: shorter than 1M steps, or never learns / collapses outside the band.
Output: data/09/census.tsv and stdout tables."""
import hashlib
import importlib
import re
import sys
from collections import defaultdict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
common = importlib.import_module("09_common")

BETA = {3, 18, 19, 20, 21, 47, 48, 57}
CLIPPED = {22}


def quantile_window(steps, values, lo, hi):
    mask = (steps >= lo) & (steps < hi)
    if mask.sum() < 20:
        return None
    q25, q50, q75 = np.percentile(values[mask], [25, 50, 75])
    return q50, q75 - q25


def classify(steps, values):
    last = int(steps[-1])
    escape = float("nan")
    if len(values) >= 48:
        median = np.median(sliding_window_view(values, 48), axis=1)
        index = int(np.argmax(median > 2800))
        if median[index] > 2800:
            escape = float(steps[index + 47])
    tail = values[-48:]
    final, final_iqr = float(np.median(tail)), float(np.subtract(*np.percentile(tail, [75, 25])))
    signature = 0
    limit = escape if np.isfinite(escape) else last
    for lo in range(600_000, int(limit), 100_000):
        window = quantile_window(steps, values, lo, lo + 100_000)
        if window and 1000 <= window[0] <= 2300 and window[1] < 120:
            signature += 1
    late = values[-200:]
    low_fraction = float(((late > 1200) & (late < 2800)).mean()) if np.percentile(late, 90) > 3500 else 0.0
    if last < 1_000_000:
        label = "X"
    elif not np.isfinite(escape):
        label = "F" if 1400 <= final <= 2700 and final_iqr < 120 and signature >= 1 else "X"
    elif low_fraction >= 0.10:
        label = "M"
    elif signature >= 1:
        label = "R"
    else:
        label = "U"
    return label, escape, final, final_iqr, signature, low_fraction


def features(run):
    hyper = run.hyper
    version = re.search(r"_v(\d+)", run.exp)
    version = int(version.group(1)) if version else -1
    name = run.exp
    precision = hyper.get("matmul_precision", "")
    critic = ("scalar" if re.search(r"scalar_v14|scalarq_v59", name) else
              "distq" if "distq" in name else "duel" if "duel" in name else
              "latentq" if "latentq" in name else "hlgauss")
    replay = (float(hyper.get("buffer_iterations") or 1) > 1 or float(hyper.get("buffer_rollouts") or 1) > 1
              or bool(hyper.get("buffer_size")) or bool(hyper.get("replay_capacity")) or bool(hyper.get("replay_rows"))
              or float(hyper.get("window_size") or 1) > 1)
    return {
        "version": version,
        "dist": "beta" if version in BETA else "clipgauss" if version in CLIPPED else "tanh",
        "lam": hyper.get("gve_lambda", "1step" if version == 5 else ""),
        "kl_bound": hyper.get("kl_bound", ""),
        "ent_mult": hyper.get("ent_target_mult", ""),
        "temp_src": hyper.get("temperature_source", "dual"),
        "aux_h": hyper.get("aux_horizon", "1" if float(hyper.get("aux_loss_coef", 0)) > 0 else "0"),
        "aux_coef": hyper.get("aux_loss_coef", ""),
        "precision": "tf32" if precision == "high" else "fp32",
        "critic": critic,
        "cw": hyper.get("critic_hidden_dim", ""),
        "cu": hyper.get("critic_updates", ""),
        "envs_x_steps": f"{hyper.get('num_envs')}x{hyper.get('num_steps')}",
        "replay": int(replay),
        "heads": hyper.get("num_heads", "1"),
        "pg_w": hyper.get("pg_weight", "0"),
        "lr": hyper.get("learning_rate", ""),
    }


def table(rows, key, title, minimum=1):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row["label"])
    print(f"\n-- by {title} (classifiable trajectories; trapped = F, ever-on-curve = F+R, mixed = M)")
    print(f"{'value':18s} {'n':>3s} {'F':>3s} {'R':>3s} {'M':>3s} {'U':>3s}  {'P(F)':>5s} {'95% Wilson':>13s}  {'P(F+R+M)':>8s} {'95% Wilson':>13s}")
    for value, labels in sorted(groups.items(), key=lambda item: -len(item[1])):
        n = len(labels)
        if n < minimum:
            continue
        counts = {c: labels.count(c) for c in "FRMU"}
        lo, hi = common.wilson(counts["F"], n)
        any_count = counts["F"] + counts["R"] + counts["M"]
        lo2, hi2 = common.wilson(any_count, n)
        print(f"{str(value):18s} {n:3d} {counts['F']:3d} {counts['R']:3d} {counts['M']:3d} {counts['U']:3d}  "
              f"{counts['F'] / n:5.2f} [{lo:4.2f}, {hi:4.2f}]  {any_count / n:8.2f} [{lo2:4.2f}, {hi2:4.2f}]")


def fisher(rows, key, a, b, positive=("F",)):
    from scipy.stats import fisher_exact
    cells = []
    for value in (a, b):
        labels = [row["label"] for row in rows if row[key] in value]
        hits = sum(label in positive for label in labels)
        cells.append((hits, len(labels) - hits))
    p = fisher_exact(cells)[1]
    print(f"Fisher exact, {key} {a} vs {b}, outcome {'+'.join(positive)}: {cells[0][0]}/{sum(cells[0])} vs "
          f"{cells[1][0]}/{sum(cells[1])}, two-sided p = {p:.3f}")


def main():
    rows, seen = [], {}
    for run in sorted(common.load_runs(), key=lambda r: r.stamp):
        steps, values = run.series(common.RETURN)
        forked = run.hyper.get("init_snapshot", "") not in ("", "None") or run.hyper.get("resume", "") not in ("", "None")
        if steps[0] > 300_000 or forked:
            continue
        early = steps <= 300_000
        key = hashlib.md5(np.round(values[early], 2).tobytes()).hexdigest()[:8] if run.last_step >= 300_000 else run.name
        if key in seen:
            seen[key]["duplicates"].append(run.exp)
            # Keep the longest member of an identical-start group as its record.
            if run.last_step <= seen[key]["last_step"]:
                continue
            duplicates = seen[key]["duplicates"] + [seen[key]["exp"]]
            rows.remove(seen[key])
        else:
            duplicates = []
        label, escape, final, final_iqr, signature, low_fraction = classify(steps, values)
        row = {"exp": run.exp, "last_step": run.last_step, "label": label, "escape_M": escape / 1e6,
               "final_median": final, "final_iqr": final_iqr, "signature_windows": signature,
               "low_fraction": low_fraction, "r0.3M": run.at(common.RETURN, 300_000),
               "r0.5M": run.at(common.RETURN, 500_000), "r1M": run.at(common.RETURN, 1_000_000),
               "r2M": run.at(common.RETURN, 2_000_000), "r4M": run.at(common.RETURN, 4_000_000),
               "r8M": run.at(common.RETURN, 8_000_000), **features(run), "duplicates": duplicates}
        seen[key] = row
        rows.append(row)
    for row in rows:
        row["duplicates"] = ",".join(sorted(set(row["duplicates"]) - {row["exp"]}))
    columns = list(rows[0])
    with open(common.OUT / "census.tsv", "w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(f"{row[c]:.3g}" if isinstance(row[c], float) else str(row[c]) for c in columns) + "\n")
    print(f"{len(rows)} distinct from-scratch trajectories")
    for label in "FRMUX":
        members = [row["exp"].replace("reppo_", "") for row in rows if row["label"] == label]
        print(f"\n{label} ({len(members)}): " + ", ".join(members))
    usable = [row for row in rows if row["label"] != "X"]
    print(f"\nclassifiable: {len(usable)}")
    table(usable, "dist", "actor distribution")
    tanh = [row for row in usable if row["dist"] == "tanh"]
    print()
    fisher(usable, "dist", ("tanh",), ("beta", "clipgauss"))
    fisher(usable, "dist", ("tanh",), ("beta", "clipgauss"), positive=("F", "R", "M"))
    fisher(tanh, "lam", ("0.95", "0.99"), ("0.8", "0.5", "1step"))
    fisher(tanh, "lam", ("0.95", "0.99"), ("0.8", "0.5", "1step"), positive=("F", "R"))
    fisher(tanh, "replay", (0,), (1,))
    fisher(tanh, "precision", ("fp32",), ("tf32",))
    fisher(tanh, "aux_h", ("1",), ("8",))
    print("\n===== tanh-Gaussian trajectories only =====")
    for key, title in (("lam", "gve_lambda"), ("kl_bound", "KL bound"), ("ent_mult", "entropy target mult"),
                       ("temp_src", "temperature source"), ("aux_h", "aux horizon"), ("precision", "precision"),
                       ("critic", "critic type"), ("cw", "critic width"), ("cu", "critic updates"),
                       ("envs_x_steps", "rollout shape"), ("replay", "replay / window"), ("heads", "actor heads"),
                       ("lr", "learning rate")):
        table(tanh, key, title)
    for row in tanh:
        row["era"] = ("v1-v6" if row["version"] <= 6 else "v7-v17" if row["version"] <= 17 else
                      "v23-v31" if row["version"] <= 31 else "v32-v44" if row["version"] <= 44 else "v45-v63")
    table(tanh, "era", "version era")
    modern = [row for row in tanh if row["version"] >= 27 and row["envs_x_steps"] == "16x1024" and row["heads"] == "1"]
    print("\n===== tanh, v27+, 16x1024, single actor =====")
    for key, title in (("lam", "gve_lambda"), ("precision", "precision"), ("aux_h", "aux horizon"),
                       ("replay", "replay / window"), ("temp_src", "temperature source"), ("ent_mult", "entropy target mult")):
        table(modern, key, title)


if __name__ == "__main__":
    main()

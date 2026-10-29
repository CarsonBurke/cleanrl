"""Q2: across all logged REPPO runs, which interventions moved the critic-quality metrics, and do those metrics
predict return? Reads data/03_runs/*.npz (from 03_dump_reppo_runs.py). Writes data/03_q2_table.tsv."""
import collections
import glob
import os
from pathlib import Path

import numpy as np
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data"
SKIP = {"exp_name", "total_timesteps", "snapshot_steps", "checkpoint_steps", "resume", "num_iterations", "snapshot_at",
        "checkpoint_at", "eval_interval", "compile", "compile_mode", "env_threads", "env_backend", "profile_iteration",
        "non_blocking_transfers", "iterations", "track", "wandb_project_name", "wandb_entity", "capture_video", "cuda",
        "torch_deterministic", "matmul_precision", "bf16", "diagnostic_rows", "diagnostic_samples", "batch_size",
        "fork_iterations", "save_snapshots", "snapshot_dir"}
TAGS = {"ret": "charts/episodic_return", "corr": "diagnostics/advantage_corr_heldout", "ratio": "diagnostics/advantage_std_ratio",
        "bell": "diagnostics/bellman_rms_heldout", "qmse": "losses/q_mse", "dqda": "diagnostics/dqda_norm",
        "dcos": "diagnostics/dqda_cos_consecutive", "std": "policy/std", "kl": "policy/kl", "viol": "policy/kl_violation",
        "temp": "policy/temperature", "ent": "policy/entropy", "sat": "policy/saturation", "clip": "losses/support_clip_frac"}
AT = [1_000_000, 4_000_000, 8_000_000]


def window(z, tag, at, half=250_000):
    if f"s|{tag}" not in z.files:
        return np.nan
    s, v = z[f"s|{tag}"], z[f"v|{tag}"]
    m = (s >= at - half) & (s <= at + half) & np.isfinite(v)
    if m.sum() < 3 or s.max() < at - half / 2:
        return np.nan
    return float(v[m].mean())


def parse_hyper(text):
    out = {}
    for line in str(text).splitlines()[2:]:
        parts = line.strip().strip("|").split("|")
        if len(parts) >= 2:
            out[parts[0]] = "|".join(parts[1:])
    return out


rows = []
for f in sorted(glob.glob(str(DATA / "03_runs/*.npz"))):
    z = np.load(f)
    name = os.path.basename(f)[len("HalfCheetah-v4__"):-4]
    exp, ts = name.rsplit("__", 2)[0], name.rsplit("__", 1)[1]
    if "s|charts/episodic_return" not in z.files or len(z["s|charts/episodic_return"]) == 0:
        continue
    row = {"exp": exp, "ts": ts, "hyper": parse_hyper(z["hyper"]), "last_step": int(z["s|charts/episodic_return"].max()),
           "final20": float(z["v|charts/episodic_return"][-20:].mean())}
    for key, tag in TAGS.items():
        for at in AT:
            row[f"{key}@{at // 1_000_000}M"] = window(z, tag, at)
    rows.append(row)

# config delta vs the modal value of each hyperparameter
values = collections.defaultdict(collections.Counter)
for r in rows:
    for k, v in r["hyper"].items():
        values[k][v] += 1
mode = {k: c.most_common(1)[0][0] for k, c in values.items()}
for r in rows:
    delta = [f"{k}={v}" for k, v in r["hyper"].items() if k not in SKIP and v != mode[k]]
    r["delta"] = ",".join(delta) if delta else "-"

cols = ["exp", "last_step", "final20"] + [f"{k}@{a}M" for k in TAGS for a in (1, 4, 8)] + ["delta"]
with open(DATA / "03_q2_table.tsv", "w") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
print(f"{len(rows)} runs with returns; table -> data/03_q2_table.tsv")

arr = lambda key: np.array([r[key] for r in rows], dtype=float)  # noqa: E731
print("\n== distribution of critic-quality metrics across runs (n = runs with the metric at that step)")
for key in ("corr", "ratio", "bell", "qmse", "dqda", "dcos"):
    for at in (1, 4, 8):
        v = arr(f"{key}@{at}M")
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            q = np.percentile(v, [0, 10, 25, 50, 75, 90, 100])
            print(f"  {key:5s}@{at}M n={len(v):3d}: min {q[0]:+.3f} p10 {q[1]:+.3f} p25 {q[2]:+.3f} median {q[3]:+.3f} p75 {q[4]:+.3f} p90 {q[5]:+.3f} max {q[6]:+.3f}")

print("\n== Spearman rank correlation of metric@T with return@T and with return@8M across runs")
print("   (healthy = return@T above the flipped ~2.3k trap: >3000 @1M, >5000 @4M, >6000 @8M)")
floors = {1: 3000, 4: 5000, 8: 6000}
for key in ("corr", "ratio", "bell", "qmse", "dqda", "dcos", "std", "kl", "viol", "temp", "ent", "sat", "clip"):
    line = f"  {key:5s}"
    for at in (1, 4, 8):
        m, ret, ret8 = arr(f"{key}@{at}M"), arr(f"ret@{at}M"), arr("ret@8M")
        for label, mask in (("all", np.isfinite(m) & np.isfinite(ret)), ("healthy", np.isfinite(m) & np.isfinite(ret) & (ret > floors[at]))):
            if mask.sum() >= 8:
                rho, p = stats.spearmanr(m[mask], ret[mask])
                line += f" | @{at}M {label} n={mask.sum():3d} rho={rho:+.2f} p={p:.3f}"
        mask = np.isfinite(m) & np.isfinite(ret8) & (ret8 > floors[8])
        if at < 8 and mask.sum() >= 8:
            rho, p = stats.spearmanr(m[mask], ret8[mask])
            line += f" | @{at}M->ret8M healthy n={mask.sum()} rho={rho:+.2f} p={p:.3f}"
    print(line)

print("\n== extremes: runs with the highest / lowest held-out advantage corr @4M (need >= 3.75M steps)")
have = [r for r in rows if np.isfinite(r["corr@4M"])]
have.sort(key=lambda r: r["corr@4M"])
fmt = lambda r: (f"  {r['exp'][:44]:44s} corr {r['corr@1M']:+.3f}/{r['corr@4M']:+.3f}/{r['corr@8M']:+.3f} ratio {r['ratio@1M']:.2f}/{r['ratio@4M']:.2f}/{r['ratio@8M']:.2f} "  # noqa: E731
                 f"qmse {r['qmse@4M']:.2f} ret {r['ret@1M']:.0f}/{r['ret@4M']:.0f}/{r['ret@8M']:.0f} | {r['delta'][:90]}")
for r in have[:12]:
    print(fmt(r))
print("  ...")
for r in have[-14:]:
    print(fmt(r))
print("\n== extremes of advantage_std_ratio @4M")
have.sort(key=lambda r: r["ratio@4M"])
for r in have[:6] + have[-10:]:
    print(fmt(r))
print("\n== runs logging bellman_rms_heldout")
for r in rows:
    if np.isfinite(r["bell@1M"]) or np.isfinite(r["bell@4M"]):
        print(f"  {r['exp'][:44]:44s} bell {r['bell@1M']:.3f}/{r['bell@4M']:.3f}/{r['bell@8M']:.3f} corr {r['corr@4M']:+.3f}/{r['corr@8M']:+.3f} ret {r['ret@4M']:.0f}/{r['ret@8M']:.0f} | {r['delta'][:80]}")

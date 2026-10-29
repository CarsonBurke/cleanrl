"""Q3: co-evolution of return, critic fit, critic ranking, policy scale and trust-region state in the reference runs.
Per 1M-step bin means from data/03_runs/*.npz."""
import sys
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data/03_runs"
RUNS = {
    "v27_h8 (=v29 h8 reward)": "reppo_latentroll_v27_h8_50M__1__1790378891",
    "v29 parent lam0.8 (v25)": "reppo_diagnose_v29_parent_lam08_16M__1__1790481423",
    "v29 h8 noreward": "reppo_diagnose_v29_h8_noreward_16M__1__1790482787",
    "v36 fp32": "reppo_fast_v36_fp32__1__1790621840",
    "v55 dual (landscape ref)": "reppo_landscape_ref_v55dual__1__1790711607",
    "v60 base": "reppo_noisefork_v60_base__1__1790713932",
    "v61 exact l095": "reppo_exacttarget_v61_l095__1__1790714718",
    "v62 control": "reppo_exactfork_v62_control__1__1790715176",
    "v62 exact l08": "reppo_exactfork_v62_l08__1__1790715176",
    "v62 exact l095": "reppo_exactfork_v62_l095__1__1790715176",
}
COLS = [("ret", "charts/episodic_return", "{:7.0f}"), ("det", "eval/deterministic_return", "{:7.0f}"),
        ("qmse", "losses/q_mse", "{:6.2f}"), ("ce", "losses/critic_ce", "{:6.3f}"), ("tgt", "losses/target_mean", "{:6.1f}"),
        ("corr", "diagnostics/advantage_corr_heldout", "{:+6.3f}"), ("ratio", "diagnostics/advantage_std_ratio", "{:6.3f}"),
        ("bell", "diagnostics/bellman_rms_heldout", "{:6.2f}"), ("dqda", "diagnostics/dqda_norm", "{:6.2f}"),
        ("dcos", "diagnostics/dqda_cos_consecutive", "{:6.2f}"), ("std", "policy/std", "{:6.3f}"),
        ("ent", "policy/entropy", "{:6.2f}"), ("temp", "policy/temperature", "{:7.4f}"), ("kl", "policy/kl", "{:7.4f}"),
        ("viol", "policy/kl_violation", "{:6.3f}"), ("klmul", "policy/kl_multiplier", "{:7.3f}"), ("sat", "policy/saturation", "{:6.3f}"),
        ("|a|", "policy/mean_abs_action", "{:6.3f}"), ("gA", "grad/actor_preclip_norm", "{:6.2f}"), ("gC", "grad/critic_preclip_norm", "{:6.2f}")]


def bins(z, tag, edges):
    if f"s|{tag}" not in z.files:
        return [np.nan] * (len(edges) - 1)
    s, v = z[f"s|{tag}"], z[f"v|{tag}"]
    return [float(np.nanmean(v[(s >= a) & (s < b)])) if ((s >= a) & (s < b)).any() else np.nan for a, b in zip(edges[:-1], edges[1:])]


out = {}
for label, name in RUNS.items():
    z = np.load(DATA / f"HalfCheetah-v4__{name}.npz")
    last = int(z["s|charts/episodic_return"].max())
    edges = np.arange(0, min(last, 16_000_000) + 1, 1_000_000)
    table = {key: bins(z, tag, edges) for key, tag, _ in COLS}
    out[label] = table
    present = [(k, f) for k, _, f in COLS if np.isfinite(table[k]).any()]
    print(f"\n=== {label}: {name} (last step {last / 1e6:.2f}M)")
    print(" M-bin  " + " ".join(f"{k:>7s}" for k, _ in present) + "   dRet/M")
    ret = table["ret"]
    for i in range(len(edges) - 1):
        slope = ret[i] - ret[i - 1] if i else np.nan
        print(f" {i:2d}-{i + 1:<2d}  " + " ".join(f.format(table[k][i]).rjust(7) for k, f in present) + f"  {slope:+7.0f}")
    # within-run co-movement on 250k bins, detrended by first differences
    fine = np.arange(500_000, min(last, 16_000_000) + 1, 250_000)
    r = np.array(bins(z, "charts/episodic_return", fine))
    line = "  corr of 250k-bin first differences with d(return) [same bin | return one bin later]:"
    for key, tag, _ in COLS[2:]:
        m = np.array(bins(z, tag, fine))
        if np.isfinite(m).sum() > 10:
            dm, dr = np.diff(m), np.diff(r)
            ok = np.isfinite(dm) & np.isfinite(dr)
            same = np.corrcoef(dm[ok], dr[ok])[0, 1]
            ok2 = np.isfinite(dm[:-1]) & np.isfinite(dr[1:])
            lead = np.corrcoef(dm[:-1][ok2], dr[1:][ok2])[0, 1]
            line += f" {key} {same:+.2f}|{lead:+.2f};"
    print(line)

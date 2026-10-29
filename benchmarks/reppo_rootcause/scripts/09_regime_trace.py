"""Binned trace of one run through the 3-4M regime change: return distribution (median, p10, share of episodes
below 0.85 x median = stumbles), critic and actor scalars. Usage: 09_regime_trace.py <exp-substring> [lo_M hi_M bin_M]"""
import importlib
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
common = importlib.import_module("09_common")

TAGS = [("q_mse", "losses/q_mse"), ("tgt", "losses/target_mean"), ("corr", "diagnostics/advantage_corr_heldout"),
        ("ratio", "diagnostics/advantage_std_ratio"), ("dqda", "diagnostics/dqda_norm"),
        ("klmul", "policy/kl_multiplier"), ("agrad", "grad/actor_preclip_norm"), ("cgrad", "grad/critic_preclip_norm"),
        ("std", "policy/std"), ("temp", "policy/temperature"), ("ent", "policy/entropy"),
        ("sat", "policy/saturation"), ("|mu|", "policy/mean_abs_action"), ("viol", "policy/kl_violation"),
        ("aux", "losses/aux_loss"), ("rew", "losses/reward_loss"), ("ce", "losses/critic_ce"),
        ("clip", "losses/support_clip_frac"), ("det", "eval/deterministic_return"),
        ("detsat", "eval/deterministic_saturation")]


def trace(run, lo, hi, width):
    steps, values = run.series(common.RETURN)
    print(f"# {run.exp}")
    print("M".rjust(5) + "".join(name.rjust(8) for name in ["med", "p10", "min", "stumb", "iqr"] + [n for n, _ in TAGS]))
    for start in np.arange(lo, hi, width):
        a, b = start * 1e6, (start + width) * 1e6
        mask = (steps >= a) & (steps < b)
        if mask.sum() < 10:
            continue
        v = values[mask]
        median = np.median(v)
        cells = [median, np.percentile(v, 10), v.min(), (v < 0.85 * median).mean(), np.subtract(*np.percentile(v, [75, 25]))]
        cells += [run.window(tag, a, b) for _, tag in TAGS]
        print(f"{start:5.2f}" + "".join(f"{c:8.3g}" for c in cells))


if __name__ == "__main__":
    runs = common.load_runs()
    pattern = sys.argv[1]
    lo, hi, width = (float(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (0.0, 8.0, 0.5)
    for run in runs:
        if pattern in run.exp:
            trace(run, lo, hi, width)

"""Early-phase scalars, iteration by iteration, for matched trajectories of the same algorithm
(B upright = v60 base, A upright = v27 H8, C flipped = v61 control, D flipped = v43 FP32) and the v13 r1/r2 pair.
Usage: 09_early_pairs.py [max_iteration] [stride]. Output: stdout (tee to data/09/early_pairs.txt)."""
import importlib
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
common = importlib.import_module("09_common")

GROUPS = {
    "reference": [("B_up", "reppo_noisefork_v60_base"), ("A_up", "reppo_latentroll_v27_h8_50M"),
                  ("C_flip", "reppo_exacttarget_v61_control"), ("D_flip", "reppo_entanneal_v43_fp32_s50_f1p5"),
                  ("l095_up", "reppo_exacttarget_v61_l095"), ("l08_flip", "reppo_exacttarget_v61_l08")],
    "v13": [("r2_up", "reppo_distq_hidden_v13_r2"), ("r1_flip", "reppo_distq_hidden_v13_r1")],
}
TAGS = ["policy/std", "policy/entropy", "policy/temperature", "policy/saturation", "policy/mean_abs_action",
        "policy/kl", "policy/kl_violation", "policy/kl_multiplier", "policy/q_pi", "losses/target_mean",
        "losses/q_mse", "losses/critic_ce", "losses/aux_loss", "losses/support_clip_frac", "diagnostics/dqda_norm",
        "diagnostics/advantage_corr_heldout", "diagnostics/advantage_std_ratio", "grad/actor_preclip_norm",
        "grad/critic_preclip_norm", "eval/deterministic_return"]


def main():
    last = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    runs = {run.exp: run for run in common.load_runs()}
    for group, members in GROUPS.items():
        members = [(label, runs[name]) for label, name in members if name in runs]
        batch = int(members[0][1].hyper["batch_size"])
        picks = list(range(0, last, stride))
        print(f"\n=== {group}: " + ", ".join(f"{label}={run.exp}" for label, run in members))
        print("iteration:".ljust(34) + " ".join(f"{i + 1:7d}" for i in picks))
        print("median episodic return")
        for label, run in members:
            steps, values = run.series(common.RETURN)
            cells = []
            for i in picks:
                mask = (steps > i * batch) & (steps <= (i + stride) * batch)
                cells.append(f"{np.median(values[mask]):7.0f}" if mask.any() else "      -")
            print(f"  {label}".ljust(34) + " ".join(cells))
        print("episodic return IQR")
        for label, run in members:
            steps, values = run.series(common.RETURN)
            cells = []
            for i in picks:
                mask = (steps > i * batch) & (steps <= (i + stride) * batch)
                cells.append(f"{np.subtract(*np.percentile(values[mask], [75, 25])):7.0f}" if mask.sum() > 3 else "      -")
            print(f"  {label}".ljust(34) + " ".join(cells))
        for tag in TAGS:
            if not any(tag in run.tags for _, run in members):
                continue
            print(tag)
            for label, run in members:
                _, values = run.series(tag)
                if tag.startswith("eval"):
                    cells = [f"{v:7.0f}" for v in values[:12]]
                else:
                    cells = [f"{values[i:i + stride].mean():7.3g}" if i < len(values) else "      -" for i in picks]
                print(f"  {label}".ljust(34) + " ".join(cells))


if __name__ == "__main__":
    main()

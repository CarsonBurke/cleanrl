"""Red-team 08: matched-step scalars (return, KL per iteration, entropy/std, qprop diagnostics) for selected runs.

Usage: 08_mine_runs.py <exp_name> [<exp_name> ...]   (exact exp names; first matching run dir each)
Window: +-50k steps around each target (same convention as score_runs.py --at).
"""
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

STEPS = [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]
TAGS = [
    "charts/episodic_return", "eval/deterministic_return", "losses/approx_kl", "losses/clipfrac", "losses/entropy",
    "losses/explained_variance", "policy/kl", "policy/kl_violation", "policy/entropy", "policy/std",
    "policy/temperature", "qprop/control_corr", "qprop/fullq_corr", "qprop/eta", "qprop/control_std_ratio",
    "q/explained_variance", "diagnostics/advantage_corr_heldout", "diagnostics/advantage_std_ratio",
    "probe/slope_beta", "probe/real_advantage_std", "probe/critic_contrast_std", "grad/actor_preclip_norm",
]


def window(acc, tag, step, half=50_000):
    ev = acc.Scalars(tag)
    s = np.array([e.step for e in ev]); v = np.array([e.value for e in ev])
    m = (s >= step - half) & (s <= step + half)
    return float(np.nanmean(v[m])) if m.any() else float("nan")


for name in sys.argv[1:]:
    dirs = sorted(Path("runs").glob(f"HalfCheetah-v4__{name}__1__*"))
    if not dirs:
        print(f"## {name}: NOT FOUND"); continue
    d = max(dirs, key=lambda p: sum(f.stat().st_size for f in p.glob("events*")))
    acc = EventAccumulator(str(d), size_guidance={"scalars": 0}); acc.Reload()
    have = set(acc.Tags()["scalars"])
    print(f"## {name}  ({d.name})")
    print("tag".ljust(38) + "".join(f"{s/1e6:>10.2f}M" for s in STEPS))
    for t in TAGS:
        if t in have:
            print(t.ljust(38) + "".join(f"{window(acc, t, s):>11.4g}" for s in STEPS))

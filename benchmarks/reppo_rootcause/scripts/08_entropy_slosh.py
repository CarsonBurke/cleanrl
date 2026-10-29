"""Red-team 08: per-log-point fluctuation of the squashed entropy / temperature / std in REPPO reference runs.

Asks whether the entropy dual moves the policy's noise scale back and forth between iterations by amounts whose
first-order return effect is comparable to or larger than the net progress per iteration.
"""
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for name in sys.argv[1:]:
    d = sorted(Path("runs").glob(f"HalfCheetah-v4__{name}__1__*"))[-1]
    acc = EventAccumulator(str(d), size_guidance={"scalars": 0}); acc.Reload()
    print(f"## {name}")
    for tag in ("policy/entropy", "policy/std", "policy/temperature", "policy/kl_multiplier", "policy/q_pi"):
        if tag not in acc.Tags()["scalars"]:
            continue
        ev = acc.Scalars(tag); s = np.array([e.step for e in ev], float); v = np.array([e.value for e in ev], float)
        gap = np.median(np.diff(s))
        out = []
        for lo, hi in ((0.8e6, 1.5e6), (3.5e6, 4.5e6), (7e6, 8e6)):
            m = (s >= lo) & (s < hi); x = v[m]
            if m.sum() < 8:
                out.append("    -"); continue
            dx = np.diff(x)
            out.append(f"[{lo/1e6:.1f}-{hi/1e6:.1f}M] mean {x.mean():+.4f} sd {x.std():.4f} sd(diff) {dx.std():.4f} "
                       f"lag1(diff) {np.corrcoef(dx[:-1], dx[1:])[0, 1]:+.2f}")
        print(f"  {tag:22s} log gap {gap:.0f} steps | " + " | ".join(out))

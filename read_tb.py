#!/usr/bin/env python3
"""Read TensorBoard logs for quick experiment comparison.

Usage:
    python read_tb.py [pattern] [env_filter]

Examples:
    python read_tb.py space_v8              # All space_v8 runs
    python read_tb.py ace_v6 HalfCheetah    # ACE v6 on HalfCheetah only
    python read_tb.py space                 # All SPACE runs
    python read_tb.py                       # 10 most recent runs
"""
import sys, os
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

pattern = sys.argv[1] if len(sys.argv) > 1 else ""
env_filter = sys.argv[2] if len(sys.argv) > 2 else ""
metric = "charts/episodic_return"

runs_dir = "runs"
dirs = sorted(os.listdir(runs_dir), key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)), reverse=True)

if pattern:
    dirs = [d for d in dirs if pattern.lower() in d.lower()]
if env_filter:
    dirs = [d for d in dirs if env_filter.lower() in d.lower()]

if not dirs:
    print("No matching runs found.")
    sys.exit(0)

for d in dirs[:10]:
    path = os.path.join(runs_dir, d)
    try:
        ea = EventAccumulator(path)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if metric not in tags:
            continue
        events = ea.Scalars(metric)
        if not events:
            continue

        by_step = defaultdict(list)
        for e in events:
            by_step[e.step].append(e.value)

        steps = sorted(by_step.keys())
        last_step = steps[-1]
        last_vals = by_step[last_step]
        last_avg = sum(last_vals) / len(last_vals)

        peak_step, peak_avg = 0, 0
        for s in steps:
            avg = sum(by_step[s]) / len(by_step[s])
            if avg > peak_avg:
                peak_avg = avg
                peak_step = s

        tail_steps = steps[-5:]
        avgs_str = ", ".join(str(int(sum(by_step[s])/len(by_step[s]))) for s in tail_steps)
        steps_str = str([s for s in tail_steps])

        print(f"\n--- {d} ---")
        print(f"  Final: step={last_step}, avg={last_avg:.1f} (n={len(last_vals)})")
        print(f"  Peak:  step={peak_step}, avg={peak_avg:.1f}")
        print(f"  Last 5: [{avgs_str}] at steps {steps_str}")
    except Exception as e:
        print(f"  Error reading {d}: {e}")

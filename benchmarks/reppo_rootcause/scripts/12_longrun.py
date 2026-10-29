"""12: long-horizon stochastic / deterministic returns and noise of REPPO 50M runs (data/03_runs dumps)."""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
common = import_module("09_common")
runs = {run.exp: run for run in common.load_runs()}
MARKS = [4, 8, 12, 16, 20, 30, 40, 49.5]
for exp in sys.argv[1:]:
    run = runs[exp]
    print(exp)
    for tag in ("charts/episodic_return", "eval/deterministic_return", "policy/std", "losses/q_mse", "probe/slope_beta"):
        row = [run.window(tag, (m - 0.5) * 1e6, (m + 0.5) * 1e6) for m in MARKS]
        print(f"  {tag:28s}", " ".join(f"{v:9.3f}" if abs(v) < 100 else f"{v:9.0f}" for v in row))
print("marks (M):", MARKS)

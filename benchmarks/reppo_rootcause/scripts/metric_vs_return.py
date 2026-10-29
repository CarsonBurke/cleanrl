"""Does advantage_corr_heldout predict performance across REPPO runs? Reads the dump_runs.py TSV."""
import csv
import sys

import numpy as np
from scipy.stats import spearmanr

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
seen = set()
for step, ret in (("1M", "@1000k"), ("2M", "@2000k"), ("4M", "@4000k"), ("8M", "@8000k")):
    pts = []
    for r in rows:
        try:
            c, s, y = float(r[f"advantage_corr_heldout@{step}"]), float(r[f"advantage_std_ratio@{step}"]), float(r[ret])
        except (ValueError, KeyError, TypeError):
            continue
        key = (round(c, 5), round(y))  # bit-identical trajectories logged under several names count once
        if (step, key) in seen:
            continue
        seen.add((step, key))
        pts.append((c, s, y))
    a = np.array(pts)
    up = a[a[:, 2] > 3000] if step != "1M" else a[a[:, 2] > 2000]
    for name, x in (("all", a), ("upright", up)):
        print(f"{step} {name:8s} n={len(x):3d}  spearman(corr, return)={spearmanr(x[:,0], x[:,2])[0]:+.2f}  "
              f"spearman(std_ratio, return)={spearmanr(x[:,1], x[:,2])[0]:+.2f}  "
              f"median corr={np.median(x[:,0]):.3f} ratio={np.median(x[:,1]):.3f} slope=corr/ratio={np.median(x[:,0]/x[:,1]):.2f}")

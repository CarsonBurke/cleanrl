"""Dump every scalar series of every runs/HalfCheetah-v4__reppo_* tfevents run to one npz per run (data/03_runs/),
plus the hyperparameter text table. Read-only on runs/."""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "benchmarks/reppo_rootcause/data/03_runs"


def dump(run):
    target = OUT / f"{run.name}.npz"
    if target.exists():
        return run.name, -1
    if not list(run.glob("events.out.tfevents.*")):
        return run.name, 0
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0, "tensors": 0})
    acc.Reload()
    arrays = {}
    for tag in acc.Tags()["scalars"]:
        events = acc.Scalars(tag)
        arrays[f"s|{tag}"] = np.array([e.step for e in events], dtype=np.int64)
        arrays[f"v|{tag}"] = np.array([e.value for e in events], dtype=np.float32)
    hyper = ""
    for tag in acc.Tags().get("tensors", []):
        if "hyperparameters" in tag:
            hyper = acc.Tensors(tag)[0].tensor_proto.string_val[0].decode()
    np.savez_compressed(target, hyper=np.array(hyper), **arrays)
    return run.name, len(arrays) // 2


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in (REPO / "runs").glob("HalfCheetah-v4__reppo_*") if p.is_dir())
    with ProcessPoolExecutor(3) as pool:
        for name, count in pool.map(dump, runs):
            if count == 0:
                print("EMPTY", name, flush=True)
    print("done", len(runs))

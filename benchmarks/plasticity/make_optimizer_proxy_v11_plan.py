"""v11 plan: a structured-teacher off-marginal regime plus SNR-shrunk matrix families on lean per-family grids.

Every family (adamw, polar, matrix_rms, polar_svag, rms_svag) sweeps the same axes in every
regime: the 13-point v8 lr axis x weight decay {0, .03, .1, .3} x beta1 {parent lock, .9, .99},
with beta2 and head_lr_scale fixed to the parent's v8-locked confirmation configuration for the
regime (structured_ood inherits the iid_online locks: same batch, horizon, noise and drift). The
shrunk families' parents are polar and matrix_rms. Weight decay is swept for every arm because
RethinkD showed it is the cheapest energy prior and that its in-distribution cost differs by
direction; the beta1 axis exists because the shrinkage factor's strength is set by beta1.
"""
import itertools
import json
from pathlib import Path

lean = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v9_lean_plan.json").read_text())
confirm = json.loads(Path("benchmarks/plasticity/optimizer_proxy_confirmation_plan.json").read_text())
locked = {case["scenario"]["name"]: {name: entry["config"] for name, entry in case["locked"].items()} for case in confirm["cases"]}
locked["structured_ood"] = locked["iid_online"]
lrs = lean["grid"]["shared_v8"]["lr"]
decays = [0.0, 0.03, 0.1, 0.3]
parents = {"adamw": "adamw", "polar": "polar", "matrix_rms": "matrix_rms", "polar_svag": "polar", "rms_svag": "matrix_rms"}
structured = {"name": "structured_ood", "objective": "regression", "samples": 65536, "batch_size": 1, "epochs": 1,
              "drift": False, "block": 32, "relevant": 4, "distractor_scale": 3.0, "relevant_scale": 1.5}


def sweep(parent):
    beta1s = sorted({parent["beta1"], 0.9, 0.99})
    return [{**parent, "lr": lr, "weight_decay": decay, "beta1": beta1}
            for lr, decay, beta1 in itertools.product(lrs, decays, beta1s)]


families = list(parents)
scenarios = [*lean["suite"], structured]
plan = {**lean, "suite": scenarios, "families": families, "scenario_families": {s["name"]: families for s in scenarios},
        "family_configurations": {family: {s["name"]: sweep(locked[s["name"]][parent]) for s in scenarios}
                                  for family, parent in parents.items()},
        "grid": {"shared_v8": lean["grid"]["shared_v8"], "lean_axes": {"lr": lrs, "weight_decay": decays,
                                                                       "beta1": "parent lock, 0.9, 0.99"}},
        "selection": "Every family fixes beta2/head_lr_scale to its parent's v8-locked confirmation configuration per regime "
                     "(structured_ood inherits iid_online) and sweeps 13 lrs x decay {0,.03,.1,.3} x beta1 {lock,.9,.99}. "
                     "Selection by noisy in-distribution validation only; off-marginal sets are never seen before locking. One family per job.",
        "heldout": lean["heldout"] + " structured_ood additionally scores the locked trajectories on distractor-scaled (x3, "
                   "teacher-invariant) and relevant-scaled (x1.5) test inputs; those scores never rerank configurations.",
        "promotion": "structured_ood: a shrunk family must have >=20% lower sustained distractor risk than its parent on the same grid "
                     "with in-distribution sustained risk within 5% of the parent's; the parent comparison is polar for polar_svag and "
                     "matrix_rms for rms_svag, and the AdamW reference is reported. A structured_ood win is extended to the three "
                     "protocol regimes before any promotion; a regression there kills it. No RL promotion from proxy wins.",
        "state": "Shrunk families keep a full hidden second moment (v8 matrix families keep the bias column only); no aux slots.",
        "inherited_from": "benchmarks/plasticity/optimizer_proxy_v9_lean_plan.json"}
Path("benchmarks/plasticity/optimizer_proxy_v11_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
print({f: {s: len(c) for s, c in v.items()} for f, v in plan["family_configurations"].items()})

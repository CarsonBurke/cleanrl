"""v12 plan: resistance-gated decay families against their parents on the same lean grids, all four regimes.

Parents (adamw, polar) sweep the 13-point v8 lr axis x weight decay {0, .03, .1, .3, 1} at the regime's
v8-locked beta1/beta2/head_lr_scale (structured_ood inherits iid_online). Gated families (gate_adamw,
gate_polar) sweep the same lr axis x weight decay {.03, .1, .3, 1} x gate_beta {.99, .999}. Decay 1 is
on both grids because the oracle ceiling (RethinkD 13.2) was still rising at that value; the parents'
uniform decay 1 is the control that separates "stronger decay" from "gated decay".
"""
import itertools
import json
from pathlib import Path

v11 = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v11_plan.json").read_text())
confirm = json.loads(Path("benchmarks/plasticity/optimizer_proxy_confirmation_plan.json").read_text())
locked = {case["scenario"]["name"]: {name: entry["config"] for name, entry in case["locked"].items()} for case in confirm["cases"]}
locked["structured_ood"] = locked["iid_online"]
lrs = v11["grid"]["shared_v8"]["lr"]
parent_decays = [0.0, 0.03, 0.1, 0.3, 1.0]
gated_decays = [0.03, 0.1, 0.3, 1.0]
gate_betas = [0.99, 0.999]
parents = {"adamw": "adamw", "polar": "polar", "gate_adamw": "adamw", "gate_polar": "polar"}


def sweep(family, parent):
    if family.startswith("gate_"):
        return [{**parent, "lr": lr, "weight_decay": decay, "gate_beta": beta}
                for lr, decay, beta in itertools.product(lrs, gated_decays, gate_betas)]
    return [{**parent, "lr": lr, "weight_decay": decay} for lr, decay in itertools.product(lrs, parent_decays)]


families = list(parents)
scenarios = v11["suite"]
plan = {**v11, "families": families, "scenario_families": {s["name"]: families for s in scenarios},
        "family_configurations": {family: {s["name"]: sweep(family, locked[s["name"]][parent]) for s in scenarios}
                                  for family, parent in parents.items()},
        "grid": {"shared_v8": v11["grid"]["shared_v8"],
                 "lean_axes": {"lr": lrs, "parent_weight_decay": parent_decays, "gated_weight_decay": gated_decays,
                               "gate_beta": gate_betas, "beta1": "regime lock"}},
        "selection": "Parents sweep 13 lrs x decay {0,.03,.1,.3,1}; gated families sweep 13 lrs x decay {.03,.1,.3,1} x "
                     "gate_beta {.99,.999}; beta1/beta2/head_lr_scale are the regime's v8 locks (structured_ood inherits "
                     "iid_online). Selection by noisy in-distribution validation only. One family per job.",
        "promotion": "RethinkD section 14. structured_ood: a gated family FIRES if its selected sustained in-distribution "
                     "risk is >= 20% below its parent's on this plan, or within 5% of it with >= 20% lower sustained "
                     "distractor risk. Protocol regimes (iid_online, drifting_reuse, clipped_bandit): a gated family must be "
                     "within 5% of its parent's selected risk in every regime; a larger regression kills promotion. "
                     "Promotion of gate_polar over polar requires the structured fire and no regression. No RL promotion from proxy wins.",
        "state": "Gated families keep the raw-gradient mean and power EMAs (pole gate_beta) in aux slots 2 and 3; no tier state.",
        "inherited_from": "benchmarks/plasticity/optimizer_proxy_v11_plan.json"}
Path("benchmarks/plasticity/optimizer_proxy_v12_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
print({f: {s: len(c) for s, c in v.items()} for f, v in plan["family_configurations"].items()})

"""Lean v9 plan: tier families inherit their fast tier's v8-locked betas/decay/head per regime and sweep lr x tier axes."""
import itertools
import json
from pathlib import Path

v9 = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v9_plan.json").read_text())
confirm = json.loads(Path("benchmarks/plasticity/optimizer_proxy_confirmation_plan.json").read_text())
locked = {case["scenario"]["name"]: {name: entry["config"] for name, entry in case["locked"].items()} for case in confirm["cases"]}
lrs = v9["grid"]["shared_v8"]["lr"]
tier = {"tier_eta": [0.3, 1.0], "tier_gamma": [0.9, 0.98], "tier_kmin": [0.03], "tier_k0": [1.0], "tier_vgamma": [0.9]}
look = {"tier_eta": [0.0], "tier_gamma": [0.9], "tier_kmin": [0.03], "tier_k0": [0.25, 0.5, 0.75], "tier_vgamma": [0.9]}
base = {"look_adamw": "adamw", "scalar_adamw": "adamw", "tier_adamw": "adamw", "tier_polar": "polar", "tier_vel_adamw": "adamw"}


def sweep(fixed, axes):
    keys = ["lr", *axes]
    return [{**{k: v for k, v in fixed.items() if k != "lr"}, **dict(zip(keys, values))}
            for values in itertools.product(lrs, *axes.values())]


family_configurations = {}
for family, parent in base.items():
    family_configurations[family] = {scenario: sweep(locked[scenario][parent], look if family == "look_adamw" else tier)
                                     for scenario in locked}
plan = {**v9, "family_configurations": family_configurations,
        "selection": "adamw/polar reuse the 2036 v8 configurations. Tier families fix beta1/beta2/weight_decay/head_lr_scale to their fast tier's v8-locked confirmation configuration for the regime and sweep the full 13-point lr axis times tier axes (52 configs; look_adamw 39). One family per job.",
        "grid": {**v9["grid"], "lean_tier_axes": tier, "lean_look_axes": look, "inherited_from": "benchmarks/plasticity/optimizer_proxy_confirmation_plan.json"},
        "execution": {**v9["execution"], "experiment_time_limit": "20m", "one_family_per_job": True}}
Path("benchmarks/plasticity/optimizer_proxy_v9_lean_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
# Reference plan: the fast tiers themselves on the same 13-point lr axis at their locked betas, so every lean tier
# family has a same-grid reference (the full-grid adamw/polar selections differ from the v8 lock in the reuse regimes).
reference = {**plan, "families": ["adamw", "polar"],
             "family_configurations": {parent: {scenario: sweep(locked[scenario][parent], {}) for scenario in locked}
                                       for parent in ("adamw", "polar")},
             "selection": "adamw/polar on the lean 13-point lr axis at their v8-locked betas/decay/head per regime; same-grid references for the lean tier families."}
Path("benchmarks/plasticity/optimizer_proxy_v9_lean_reference_plan.json").write_text(json.dumps(reference, indent=1) + "\n")
print({f: {s: len(c) for s, c in v.items()} for f, v in reference["family_configurations"].items()})
print({f: {s: len(c) for s, c in v.items()} for f, v in family_configurations.items()})
print(family_configurations["tier_polar"]["iid_online"][0])

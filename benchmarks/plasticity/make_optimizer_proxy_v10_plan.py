"""v10 plan: uniform-gain two-tier reset on both fast tiers, lean per-scenario grids (13 lrs x k0 axis)."""
import itertools
import json
from pathlib import Path

lean = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v9_lean_plan.json").read_text())
confirm = json.loads(Path("benchmarks/plasticity/optimizer_proxy_confirmation_plan.json").read_text())
locked = {case["scenario"]["name"]: {name: entry["config"] for name, entry in case["locked"].items()} for case in confirm["cases"]}
lrs = lean["grid"]["shared_v8"]["lr"]
gains = [0.1, 0.25, 0.5, 0.75]
fixed = {"tier_eta": 0.0, "tier_gamma": 0.9, "tier_kmin": 0.03, "tier_vgamma": 0.9}


def sweep(parent):
    return [{**{k: v for k, v in parent.items() if k != "lr"}, **fixed, "lr": lr, "tier_k0": k0}
            for lr, k0 in itertools.product(lrs, gains)]


families = {"look_adamw": "adamw", "look_polar": "polar"}
plan = {**lean, "families": list(families),
        "family_configurations": {family: {scenario: sweep(locked[scenario][parent]) for scenario in locked}
                                  for family, parent in families.items()},
        "selection": "look_adamw/look_polar fix betas/decay/head to their fast tier's v8-locked config per regime and sweep 13 lrs x k0 {.1,.25,.5,.75} (52 configs). References: v9 lean adamw/polar runs on the same lr axis (iid_online 6352/6353 full grid, bandit 6371/6372, drifting 6373/6374).",
        "grid": {"shared_v8": lean["grid"]["shared_v8"], "look_axes": {**fixed, "tier_k0": gains}},
        "promotion": "look_polar must beat polar on the same lr axis by more than run-to-run selection noise in every regime where look_adamw beats adamw; a gain at the k0 = .1 edge triggers an extension, not a promotion."}
Path("benchmarks/plasticity/optimizer_proxy_v10_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
print({f: {s: len(c) for s, c in v.items()} for f, v in plan["family_configurations"].items()})

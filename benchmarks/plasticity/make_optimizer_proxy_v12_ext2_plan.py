"""v12 gate-pole extension 2: gate_beta {.99999} for the gated families, structured_ood only.

Both gated families selected gate_beta .999, the slowest pole on the v12 grid. The gate's discrimination
is a z-score whose magnitude grows with the square root of the averaging window, so the pole is the axis
that sets how far a resisting coordinate is released; lag (window 2k to 10k of a 65k-sample horizon) is
the cost. Same lr and decay axes as the v12 plan; parents are not rerun (their v12 results stand).
"""
import itertools
import json
from pathlib import Path

v12 = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v12_plan.json").read_text())
gate_betas = [0.99999]
families = ["gate_adamw", "gate_polar"]
scenario = next(s for s in v12["suite"] if s["name"] == "structured_ood")


def extend(configs):
    seen = {(c["lr"], c["weight_decay"]) for c in configs}
    base = {k: v for k, v in configs[0].items() if k not in ("lr", "weight_decay", "gate_beta")}
    return [{**base, "lr": lr, "weight_decay": decay, "gate_beta": beta}
            for (lr, decay), beta in itertools.product(sorted(seen), gate_betas)]


plan = {**v12, "suite": [scenario], "families": families, "scenario_families": {"structured_ood": families},
        "family_configurations": {f: {"structured_ood": extend(v12["family_configurations"][f]["structured_ood"])} for f in families},
        "grid": {**v12["grid"], "lean_axes": {**v12["grid"]["lean_axes"], "gate_beta": gate_betas}},
        "selection": v12["selection"] + " Extension 2: gate_beta {.99999} only; compared against the v12 parents.",
        "inherited_from": "benchmarks/plasticity/optimizer_proxy_v12_plan.json"}
Path("benchmarks/plasticity/optimizer_proxy_v12_ext2_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
print({f: len(c["structured_ood"]) for f, c in plan["family_configurations"].items()})

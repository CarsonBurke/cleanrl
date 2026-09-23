"""v12 union grid for gate_polar: one selection over every axis value tried, in all four regimes.

The structured_ood extensions (poles .9995-.99999, decays 3 and 10) each selected within their own plan;
this plan holds the union (13 lrs x decay {.03, .1, .3, 1, 3, 10} x gate_beta {.99, .999, .9999, .99999},
312 configurations) so the reported number is one selection, and so the dense-regime no-regression check
sees the same axes the structured winner came from. Parents' v12 results stand as the comparison.
"""
import itertools
import json
from pathlib import Path

v12 = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v12_plan.json").read_text())
decays = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
gate_betas = [0.99, 0.999, 0.9999, 0.99999]
families = ["gate_polar"]


def union(configs):
    lrs = sorted({c["lr"] for c in configs})
    base = {k: v for k, v in configs[0].items() if k not in ("lr", "weight_decay", "gate_beta")}
    return [{**base, "lr": lr, "weight_decay": decay, "gate_beta": beta}
            for lr, decay, beta in itertools.product(lrs, decays, gate_betas)]


plan = {**v12, "families": families, "scenario_families": {s["name"]: families for s in v12["suite"]},
        "family_configurations": {f: {s["name"]: union(v12["family_configurations"][f][s["name"]]) for s in v12["suite"]} for f in families},
        "grid": {**v12["grid"], "lean_axes": {**v12["grid"]["lean_axes"], "gated_weight_decay": decays, "gate_beta": gate_betas}},
        "selection": v12["selection"] + " Union plan: gate_polar over decay {.03,.1,.3,1,3,10} x gate_beta {.99,.999,.9999,.99999}.",
        "inherited_from": "benchmarks/plasticity/optimizer_proxy_v12_plan.json"}
Path("benchmarks/plasticity/optimizer_proxy_v12_full_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
print({f: {s: len(c) for s, c in v.items()} for f, v in plan["family_configurations"].items()})

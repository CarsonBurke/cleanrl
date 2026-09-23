"""v11 look extension: gain axis below .1 after job 6438 selected the k0 = .1 edge (RethinkD section 11)."""
import itertools
import json
from pathlib import Path

plan = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v11_plan.json").read_text())
confirm = json.loads(Path("benchmarks/plasticity/optimizer_proxy_confirmation_plan.json").read_text())
lock = next(c for c in confirm["cases"] if c["scenario"]["name"] == "iid_online")["locked"]["polar"]["config"]
lrs = plan["grid"]["shared_v8"]["lr"]
fixed = {"tier_eta": 0.0, "tier_gamma": 0.9, "tier_kmin": 0.03, "tier_vgamma": 0.9}
configs = [{**lock, **fixed, "lr": lr, "weight_decay": decay, "tier_k0": k0}
           for lr, decay, k0 in itertools.product(lrs, [0.0, 0.03, 0.1], [0.02, 0.05, 0.1])]
look = {**plan, "families": ["look_polar"], "scenario_families": {"structured_ood": ["look_polar"]},
        "family_configurations": {"look_polar": {"structured_ood": configs}},
        "selection": "look_polar fixes betas/head to polar's iid_online v8 lock and sweeps 13 lrs x decay {0,.03,.1} x k0 {.02,.05,.1} (117 configs), extending the v11 look plan whose selection sat at k0 = .1 on structured_ood; noisy in-distribution validation only.",
        "promotion": "RethinkD section 11: fire if selected sustained distractor risk is >= 20% below polar's v11 structured_ood value (0.543) with in-distribution risk within 5% of 0.099; kill if within 10% of polar off-marginal.",
        "inherited_from": "benchmarks/plasticity/optimizer_proxy_v11_plan.json"}
Path("benchmarks/plasticity/optimizer_proxy_v11_look_ext_plan.json").write_text(json.dumps(look, indent=1) + "\n")
print(len(configs))

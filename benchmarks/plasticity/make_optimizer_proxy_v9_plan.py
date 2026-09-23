"""Write the v9 plan: shared v8 configurations for adamw/polar, tier grids for new families."""
import itertools
import json
from pathlib import Path

v8 = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v8_plan.json").read_text())
base_axes = {"lr": v8["grid"]["lr"], "beta1": [0, 0.5, 0.9, 0.95, 0.99], "beta2": [0.9, 0.99, 0.999],
             "weight_decay": [0, 0.1], "head_lr_scale": [0.5, 1.0, 2.0]}
tier_axes = {"tier_eta": [0.3, 1.0], "tier_gamma": [0.9, 0.98], "tier_kmin": [0.03], "tier_k0": [1.0],
             "tier_vgamma": [0.9]}
look_axes = {"tier_eta": [0.0], "tier_gamma": [0.9], "tier_kmin": [0.03], "tier_k0": [0.25, 0.5, 0.75],
             "tier_vgamma": [0.9]}


def product(*axes_groups):
    axes = {k: v for group in axes_groups for k, v in group.items()}
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


tier = product(base_axes, tier_axes)
look = product(base_axes, look_axes)
plan = {
    "goal": "Test prospective consolidation (two-tier, innovation-whitened per-parameter gains) against tuned AdamW and the retained polar frontier, with uniform-gain and scalar-gain controls.",
    "diagnosis": "Adam's first/second moments summarize within-block gradient statistics; when a block reuses one batch (8 epochs) or is a single sample, cross-block persistence of realized displacements is unobserved. The consolidation gain reads it per parameter from lag-1 innovation autocorrelation.",
    "suite": [{**s, "block": 32} if s["name"] == "iid_online" else {**s, "block": s["epochs"]} for s in v8["suite"]],
    "model": v8["model"], "moving_task": v8["moving_task"], "bandit": v8["bandit"],
    "families": ["adamw", "polar", "look_adamw", "scalar_adamw", "tier_adamw", "tier_polar", "tier_vel_adamw"],
    "equations": {
        "tier": "block of `block` fast steps (AdamW or polar) from phi; at boundary d = z - phi (- vel); c = g c + (1-g) d d_prev; nu = g nu + (1-g) d^2; K = clamp(K exp(eta c/nu), kmin, 1); phi' = z - (1-K) d; vel' = vg vel + (1-vg)(phi'-phi) (vel arm only); z <- phi'.",
        "look_adamw": "eta = 0: uniform Lookahead with alpha = k0 over the same blocks (control for the two-tier structure without adaptation).",
        "scalar_adamw": "one K per candidate adapted from the parameter-mean of c/nu (control: time profile kept, spatial dispersion erased).",
        "adamw": v8["equations"]["adamw"], "polar": "retained v8 five-step polar; hidden matrices Newton-Schulz, biases/head AdamW.",
    },
    "grid": {"shared_v8": v8["grid"], "tier": {**base_axes, **tier_axes}, "look": {**base_axes, **look_axes}},
    "configurations": v8["configurations"],
    "family_configurations": {"look_adamw": look, "scalar_adamw": tier, "tier_adamw": tier, "tier_polar": tier,
                              "tier_vel_adamw": tier},
    "selection": "Per-family explicit configurations; adamw/polar reuse the 2036 v8 configurations exactly. Noisy validation locks; clean risk diagnostic only; test never reranks.",
    "heldout": v8["heldout"], "diagnostics": v8["diagnostics"] + " Tier families log per-candidate gain mean/std per checkpoint.",
    "promotion": v8["promotion"] + " Additionally the adaptive arms must beat both look_adamw and scalar_adamw at their own tuned configurations, or the gain is a level/structure effect.",
    "execution": {**v8["execution"], "experiment_time_limit": "120m"},
    "scenario_families": {},
    "state": "Tier families add six per-parameter slots (phi, K, c, nu, previous innovation, velocity); no covariance or Fisher state.",
}
Path("benchmarks/plasticity/optimizer_proxy_v9_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
print({k: len(v) for k, v in plan["family_configurations"].items()}, len(plan["configurations"]))

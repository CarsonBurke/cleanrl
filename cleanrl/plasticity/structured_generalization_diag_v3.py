"""Structured-teacher generalization diagnostic v3: the ceiling and the firing condition of a gated weight decay.

Why. In the structured_ood regime most of the in-distribution risk is weight
mass on the 13 irrelevant input columns (v11: about 55% for polar, 90% for
AdamW, from distractor / in-distribution risk ratios), and uniform weight decay
removes it only at the price it charges the 4 relevant columns: polar's lock
with decay 0.1 has the same fit as decay 0 and 34% less distractor risk (v2),
i.e. the removal gain and the signal cost cancel. Per-coordinate gradient
VARIANCE cannot tell the columns apart (v2: gamma uniform). What can, in
principle, is the DIRECTION of the gradient mean relative to the weight: on an
irrelevant column the population gradient is a restoring force, it always
points the weight toward zero, while on a relevant column it points toward the
teacher's value, which for a student initialised smaller than the (rescaled)
teacher mostly means away from zero. A decay applied only where the slow
gradient average agrees with shrinking ("consistency-gated decay") would remove
irrelevant mass at full strength and leave relevant columns alone.

Two questions, answered before anything is built:

  ceiling   oracle-gated decay (applied to the distractor columns of the first
            layer only, decoupled, factor 1 - lr * decay after each step) versus
            uniform decay: what a PERFECT gate buys. If the oracle buys nothing,
            the direction is dead regardless of how it is gated.
  detect    fraction of first-layer coordinates whose slow gradient average
            (the learner's own momentum, and two-pole filters of it with second
            poles .99 and .999) or slow realised displacement says "move toward
            zero", on relevant versus distractor columns, at the locks without
            decay. If distractor and relevant columns agree at the same rate, the
            gate cannot be read from the stream.

Candidates per method (adamw, polar): iid_online v8 locks at lr x {1/2, 1, 2}
with uniform decay {0, .03, .1, .3} plus oracle decay {.1, .3, 1} (uniform
decay 0), batched through the v8 RoundLearner compile + CUDA-graph path; the
oracle factor is applied in place to the graph's weight buffer between replays.
Regime: batch 1, 65536 samples, unit-variance target noise, structured teacher
reading the first 4 of 17 inputs, three held-out sets as in v1/v2.

Pre-registered rules (RethinkD.md section 13): see `decide_ceiling` and
`decide_detect`. Build the gated rule only if BOTH fire.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import optimizer_proxy_eval_v8 as eval8
from cleanrl.plasticity.covariance_sketch_eval_v3 import save_json
from cleanrl.shared import runtime


@dataclass
class Args:
    seed: int = 1
    hidden: int = 64
    input_dim: int = 17
    relevant: int = 4
    samples: int = 65536
    held_out: int = 8192
    checkpoints: int = 16
    distractor_scale: float = 3.0
    relevant_scale: float = 1.5
    plan: str = "benchmarks/plasticity/optimizer_proxy_confirmation_plan.json"
    output: str = ""


SCENARIO = {"name": "iid_online", "objective": "regression", "samples": 65536, "batch_size": 1,
            "epochs": 1, "drift": False}
METHODS = ("adamw", "polar")
LR_MULTIPLIERS = (0.5, 1.0, 2.0)
UNIFORM_DECAYS = (0.0, 0.03, 0.1, 0.3)
ORACLE_DECAYS = (0.1, 0.3, 1.0)
FILTER_BETAS = (0.99, 0.999)
KEYS = ("lr", "beta1", "beta2", "weight_decay", "head_lr_scale")
GRADIENT_SIGNALS = ("m_lock", *(f"m_slow_{b:g}" for b in FILTER_BETAS))
DISPLACEMENT_SIGNALS = tuple(f"d_slow_{b:g}" for b in FILTER_BETAS)


def locked_configs(plan_path, scenario_name=SCENARIO["name"]):
    plan = json.loads(Path(plan_path).read_text())
    case = next(c for c in plan["cases"] if c["scenario"]["name"] == scenario_name)
    return {name: dict(case["locked"][name]["config"]) for name in METHODS}


def candidate_grid(locks):
    """{method: [(name, config, oracle_decay), ...]}: locks at lr x multipliers; uniform decay swept, or oracle decay."""
    grid = {}
    for method in METHODS:
        rows = []
        for multiplier in LR_MULTIPLIERS:
            base = {**locks[method], "lr": locks[method]["lr"] * multiplier}
            for decay in UNIFORM_DECAYS:
                config = {**base, "weight_decay": decay}
                rows.append((f"{method}_lr{multiplier:g}_wd{decay:g}", {key: config[key] for key in KEYS}, 0.0))
            for decay in ORACLE_DECAYS:
                config = {**base, "weight_decay": 0.0}
                rows.append((f"{method}_lr{multiplier:g}_oracle{decay:g}", {key: config[key] for key in KEYS}, decay))
        grid[method] = rows
    return grid


def structured_teacher(args, gen):
    """Teacher reading only the first `relevant` inputs, pre-activation variance preserved."""
    teacher = reference.draw_teacher(args, gen, "cuda")
    teacher[0][:, args.relevant:] = 0
    teacher[0][:, :args.relevant] *= math.sqrt(args.input_dim / args.relevant)
    return teacher


def held_out_sets(args, teacher, gen):
    """Three (x, clean target) pairs on one shared draw; distractor scaling leaves the target unchanged."""
    x = torch.randn(args.held_out, args.input_dim, generator=gen, device="cuda")
    distractor = x.clone()
    distractor[:, args.relevant:] *= args.distractor_scale
    relevant = x.clone()
    relevant[:, :args.relevant] *= args.relevant_scale
    return {"in_dist": (x, reference.teach(teacher, x)),
            "distractor": (distractor, reference.teach(teacher, distractor)),
            "relevant": (relevant, reference.teach(teacher, relevant))}


def clean_risk(weights, x, y):
    return (eval8.model.forward(weights, x) - y).square().mean(-1)


def column_energy(first_layer, relevant, input_dim):
    """Squared weight mass of the first layer on relevant versus distractor input columns (bias excluded)."""
    return (first_layer[..., :relevant].square().sum(dim=(-1, -2)),
            first_layer[..., relevant:input_dim].square().sum(dim=(-1, -2)))


def oracle_factor(lr, oracle_decays):
    """Per-candidate decoupled shrink factor 1 - lr * decay, shaped like the learner's hyper tensors ([K, 1, 1])."""
    return 1 - lr * torch.as_tensor(oracle_decays, dtype=lr.dtype, device=lr.device).view(-1, 1, 1)


def apply_oracle_decay(first_layer, factor, relevant, input_dim):
    """Shrink the first layer's distractor columns in place (relevant columns and bias untouched)."""
    first_layer[..., relevant:input_dim].mul_(factor)


def update_filter(state, value, beta):
    """state <- beta * state + (1 - beta) * value, in place."""
    state.mul_(beta).add_(value, alpha=1 - beta)


def gate_agreement(signal, weights, relevant, input_dim, toward_zero_sign):
    """Fraction of first-layer coordinates whose signal says 'shrink': sign(signal * w) == toward_zero_sign.

    Gradient-type signals (momentum) say shrink when signal * w > 0 (the step -signal
    points toward zero): toward_zero_sign = +1. Displacement-type signals say shrink
    when signal * w < 0: toward_zero_sign = -1. Returns (relevant, distractor) means.
    """
    shrink = (signal[..., :input_dim] * weights[..., :input_dim]) * toward_zero_sign > 0
    return (shrink[..., :relevant].float().mean(dim=(-1, -2)),
            shrink[..., relevant:input_dim].float().mean(dim=(-1, -2)))


def decide_ceiling(summary, method, fire_margin=0.2, match=0.05, kill_margin=0.1):
    """What a perfect gate buys, per method.

    u = the uniform-decay candidate (decay 0 included) with the best in-distribution risk.
    FIRE if some oracle candidate has in-distribution risk >= fire_margin below u's, or
    in-distribution risk within `match` of u's and distractor risk >= fire_margin below u's.
    KILL only if at least one oracle candidate is fit-matched, no oracle candidate is
    >= kill_margin better than u in distribution, and every fit-matched oracle candidate
    has distractor risk within kill_margin of u's. Otherwise ambiguous. Nonfinite or zero risks are excluded.
    """
    finite = {name: s for name, s in summary.items()
              if s["method"] == method and s["valid"]
              and all(math.isfinite(s[key]) and s[key] > 0 for key in ("in_dist", "distractor", "relevant"))}
    uniform = {name: s for name, s in finite.items() if s["oracle_decay"] == 0}
    oracle = {name: s for name, s in finite.items() if s["oracle_decay"] > 0}
    if not uniform or not oracle:
        return {"verdict": "ambiguous", "reason": "missing uniform or oracle candidates"}
    u_name = min(uniform, key=lambda name: uniform[name]["in_dist"])
    u = uniform[u_name]
    rows = []
    for name, o in oracle.items():
        rows.append({"oracle": name, "fit_gain": 1 - o["in_dist"] / u["in_dist"],
                     "distractor_gain": 1 - o["distractor"] / u["distractor"],
                     "fit_matched": abs(math.log(o["in_dist"] / u["in_dist"])) <= math.log1p(match)})
    fires = [r for r in rows if r["fit_gain"] >= fire_margin or (r["fit_matched"] and r["distractor_gain"] >= fire_margin)]
    matched = [r for r in rows if r["fit_matched"]]
    killed = (bool(matched) and all(r["fit_gain"] < kill_margin for r in rows)
              and all(abs(r["distractor_gain"]) <= kill_margin for r in matched))
    verdict = "fire" if fires else "kill" if killed else "ambiguous"
    return {"verdict": verdict, "best_uniform": u_name, "fires": fires, "rows": rows}


def decide_detect(margins, fire_margin=0.25, kill_margin=0.10):
    """Can the gate be read from the stream? `margins` = {signal: distractor agreement - relevant agreement}.

    FIRE if some signal's margin >= fire_margin; KILL if every signal's margin <= kill_margin; else ambiguous.
    """
    finite = {key: value for key, value in margins.items() if math.isfinite(value)}
    if not finite:
        return {"verdict": "ambiguous", "reason": "no finite margins"}
    best = max(finite, key=finite.get)
    verdict = "fire" if finite[best] >= fire_margin else "kill" if all(v <= kill_margin for v in finite.values()) else "ambiguous"
    return {"verdict": verdict, "best_signal": best, "best_margin": finite[best], "margins": finite}


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    runtime.configure_runtime()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    scenario = {**SCENARIO, "samples": args.samples}
    if args.checkpoints <= 0 or args.samples % args.checkpoints:
        raise ValueError("samples must be a positive multiple of checkpoints")
    root = Path(args.output or f"runs/StructuredGeneralizationDiag__v3__{args.seed}__{time.time_ns()}")
    root.mkdir(parents=True)
    generator = eval8.generator
    initial = reference.init_weights(args, generator(args.seed, 1), "cuda")
    teacher = structured_teacher(args, generator(args.seed, 2))
    train_gen = generator(args.seed, 3)
    n = args.samples
    xs = torch.randn(n, args.input_dim, generator=train_gen, device="cuda")
    ys_noise = torch.randn(n, generator=train_gen, device="cuda")
    action_noise = torch.randn(n, generator=train_gen, device="cuda")
    targets = reference.teach(teacher, xs) + ys_noise
    held = held_out_sets(args, teacher, generator(args.seed, 7))  # never the protocol's validation (4) or test (5)
    torch.testing.assert_close(held["distractor"][1], held["in_dist"][1], rtol=0, atol=0)
    grid = candidate_grid(locked_configs(args.plan))
    log_rounds = n // args.checkpoints
    init_relevant, init_distractor = column_energy(initial[0].unsqueeze(0), args.relevant, args.input_dim)
    result = {"args": asdict(args), "scenario": scenario, "candidates": {}, "rows": {}, "seconds": {},
              "init_first_layer_energy": {"relevant": float(init_relevant[0]), "distractor": float(init_distractor[0])}}
    graph = learner = None
    for method, rows in grid.items():
        del graph, learner
        torch.cuda.empty_cache()
        torch.compiler.reset()
        learner = eval8.RoundLearner(initial, [config for _, config, _ in rows], scenario, method)
        learner.x.copy_(xs[:1])
        learner.target.copy_(targets[:1])
        learner.action_noise.copy_(action_noise[:1])
        learner.reward_noise.copy_(ys_noise[:1])
        started = time.perf_counter()
        graph = learner.capture()
        torch.cuda.synchronize()
        startup = time.perf_counter() - started
        factor = oracle_factor(learner.hyper[0], [oracle for _, _, oracle in rows])
        first_weights, first_previous, first_m = learner.weights[0], learner.previous[0], learner.m[0]
        filters = {**{f"m_slow_{b:g}": (torch.zeros_like(first_m), b) for b in FILTER_BETAS},
                   **{f"d_slow_{b:g}": (torch.zeros_like(first_m), b) for b in FILTER_BETAS}}
        curves = []
        started = time.perf_counter()
        for t in range(n):
            learner.x.copy_(xs[t:t + 1])
            learner.target.copy_(targets[t:t + 1])
            learner.action_noise.copy_(action_noise[t:t + 1])
            learner.reward_noise.copy_(ys_noise[t:t + 1])
            graph.replay()
            apply_oracle_decay(first_weights, factor, args.relevant, args.input_dim)
            displacement = first_weights - first_previous
            for key, (state, beta) in filters.items():
                update_filter(state, first_m if key.startswith("m_") else displacement, beta)
            if (t + 1) % log_rounds:
                continue
            risks = {key: clean_risk(learner.weights, x, y) for key, (x, y) in held.items()}
            for value in risks.values():
                learner.valid.logical_and_(torch.isfinite(value))
            relevant_energy, distractor_energy = column_energy(first_weights, args.relevant, args.input_dim)
            signals = {"m_lock": first_m, **{key: state for key, (state, _) in filters.items()}}
            agreement = {}
            for key, signal in signals.items():
                sign = 1.0 if key in GRADIENT_SIGNALS else -1.0
                rel, dis = gate_agreement(signal, first_weights, args.relevant, args.input_dim, sign)
                agreement[key] = {"relevant": rel.tolist(), "distractor": dis.tolist()}
            curves.append({"step": t + 1, **{f"risk_{key}": value.tolist() for key, value in risks.items()},
                           "gate_agreement": agreement,
                           "first_layer_relevant_energy": relevant_energy.tolist(),
                           "first_layer_distractor_energy": distractor_energy.tolist(),
                           "valid": learner.valid.tolist()})
        torch.cuda.synchronize()
        result["seconds"][method] = {"startup": startup, "training": time.perf_counter() - started}
        result["rows"][method] = curves
        second_half = curves[len(curves) // 2:]
        for index, (name, config, oracle) in enumerate(rows):
            valid = curves[-1]["valid"][index]
            sustained = {key: (sum(c[f"risk_{key}"][index] for c in curves) / len(curves) if valid else float("nan"))
                         for key in held}
            last = curves[-1]
            agreement = {key: {side: sum(c["gate_agreement"][key][side][index] for c in second_half) / len(second_half)
                               for side in ("relevant", "distractor")}
                         for key in (*GRADIENT_SIGNALS, *DISPLACEMENT_SIGNALS)}
            result["candidates"][name] = {
                "method": method, "config": config, "oracle_decay": oracle, "valid": valid, **sustained,
                "endpoint_in_dist": last["risk_in_dist"][index],
                "second_half_distractor": sum(c["risk_distractor"][index] for c in second_half) / len(second_half),
                "gate_agreement_second_half": agreement,
                "first_layer_relevant_energy": last["first_layer_relevant_energy"][index],
                "first_layer_distractor_energy": last["first_layer_distractor_energy"][index]}
    summary = result["candidates"]
    result["decision"] = {}
    for method in grid:
        lock = summary[f"{method}_lr1_wd0"]
        margins = {key: value["distractor"] - value["relevant"] for key, value in lock["gate_agreement_second_half"].items()}
        result["decision"][method] = {"ceiling": decide_ceiling(summary, method),
                                      "detect": decide_detect(margins if lock["valid"] else {})}
    save_json(root / "results.json", result)
    print(json.dumps({"root": str(root), "seconds": result["seconds"]}))
    for method, decision in result["decision"].items():
        print(json.dumps({"method": method, "ceiling": decision["ceiling"]["verdict"],
                          "best_uniform": decision["ceiling"].get("best_uniform"),
                          "detect": decision["detect"]["verdict"], "margins": decision["detect"].get("margins")}))
    for name, s in summary.items():
        gate = s["gate_agreement_second_half"]
        print(json.dumps({"candidate": name, "in_dist": round(s["in_dist"], 5), "distractor": round(s["distractor"], 5),
                          "relevant": round(s["relevant"], 5), "end_in_dist": round(s["endpoint_in_dist"], 5),
                          "w1_rel": round(s["first_layer_relevant_energy"], 3),
                          "w1_dis": round(s["first_layer_distractor_energy"], 3),
                          "gate": {key: (round(v["relevant"], 3), round(v["distractor"], 3)) for key, v in gate.items()}}))


if __name__ == "__main__":
    main()

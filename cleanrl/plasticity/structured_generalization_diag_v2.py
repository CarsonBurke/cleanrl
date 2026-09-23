"""Structured-teacher generalization diagnostic v2: fit-matched decision rule plus the SNR-shrinkage firing condition.

Changes from v1 (independent review of the v1 run file). The kill span is computed
among fit-matched candidates only (in-distribution risk within `match` of each
other), because the distractor/in-distribution ratio moves with the fit even
under the null; the single-candidate and zero-risk cases are guarded; the init
column energies are recorded; the decay axis is {0, .03, .1, .3}; and at every
checkpoint the AdamW learner's own bias-corrected moments give the Balles-Hennig
variance-adaptation factor gamma = m^2 / (m^2 + rho s) per coordinate, summarized
on the first layer's relevant versus distractor columns. gamma near 1 on the
distractor columns means an SNR-shrinkage rule has nothing to shrink there and
is dead before it is built; gamma well below 1 on distractors and near 1 on
relevant columns is its firing condition. Polar keeps no hidden-weight second
moment, so gamma is reported for AdamW candidates only.

Question. The v8 proxy scores in-distribution clean excess risk against a dense
random teacher of the student's own architecture. In that harness no optimizer
can show "better generalization through abstraction": every mechanism reduces
to statistical efficiency (bias decay in iid_online, tracking under drift; see
rethink/RethinkD.md). This diagnostic gives the teacher structure, only the
first `relevant` of `input_dim` inputs reach it (the other columns of its first
layer are zero, the kept columns rescaled so pre-activation variance is
unchanged), and scores every candidate on three fresh held-out sets:

  in_dist     x ~ N(0, I)                         the protocol's own measurement
  distractor  irrelevant inputs scaled by 3       teacher output unchanged: invariance
  relevant    relevant inputs scaled by 1.5       extrapolation on the true signal

plus the brain-energy papers' energy, path length sum_t |w_t - w_{t-1}|_1, the
distance from init, and the first layer's relevant/distractor column energy.

Candidates: the iid_online v8 locks for adamw and polar at lr x {1/2, 1, 2} and
weight decay {0, 0.1} (weight decay is the cheapest energy prior), batched per
method through the v8 RoundLearner compile + CUDA-graph path. Regime: batch 1,
65536 samples, unit-variance target noise, 16 right-endpoint checkpoints.

Pre-registered rule (RethinkD.md sections 6 and 9): see `decide`.
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
WEIGHT_DECAYS = (0.0, 0.03, 0.1, 0.3)
KEYS = ("lr", "beta1", "beta2", "weight_decay", "head_lr_scale")


def locked_configs(plan_path, scenario_name=SCENARIO["name"]):
    plan = json.loads(Path(plan_path).read_text())
    case = next(c for c in plan["cases"] if c["scenario"]["name"] == scenario_name)
    return {name: dict(case["locked"][name]["config"]) for name in METHODS}


def candidate_grid(locks):
    """{method: [(name, config), ...]}: locks at lr x multipliers, weight decay swept, all else fixed."""
    grid = {}
    for method in METHODS:
        rows = []
        for multiplier in LR_MULTIPLIERS:
            for decay in WEIGHT_DECAYS:
                config = {**locks[method], "lr": locks[method]["lr"] * multiplier, "weight_decay": decay}
                rows.append((f"{method}_lr{multiplier:g}_wd{decay:g}", {key: config[key] for key in KEYS}))
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


def accumulate_path(path, weights, previous):
    """path[l][k] += |w_l - w_l^prev|_1 for every layer; pure elementwise-sum energy."""
    for total, current, before in zip(path, weights, previous):
        total.add_((current - before).abs().sum(dim=(-1, -2)))


def clean_risk(weights, x, y):
    return (eval8.model.forward(weights, x) - y).square().mean(-1)


def column_energy(first_layer, relevant, input_dim):
    """Squared weight mass of the first layer on relevant versus distractor input columns (bias excluded)."""
    return (first_layer[..., :relevant].square().sum(dim=(-1, -2)),
            first_layer[..., relevant:input_dim].square().sum(dim=(-1, -2)))


def decide(summary, fire_margin=0.2, kill_span=0.1, match=0.05):
    """Pre-registered rule on per-candidate sustained risks.

    FIRE: some candidate has distractor risk >= fire_margin below another's with
    equal or worse in-distribution risk. KILL: at least one fit-matched pair
    (in-distribution risks within `match`) exists and every such pair has
    distractor risks within kill_span. Otherwise ambiguous. Candidates with a
    nonfinite or zero risk are excluded from every comparison.
    """
    finite = {name: s for name, s in summary.items()
              if all(math.isfinite(s[key]) and s[key] > 0 for key in ("in_dist", "distractor", "relevant"))}
    fires, matched = [], []
    for a, sa in finite.items():
        for b, sb in finite.items():
            if a == b:
                continue
            if sa["in_dist"] >= sb["in_dist"] and sa["distractor"] <= (1 - fire_margin) * sb["distractor"]:
                fires.append({"better_ood": a, "worse_ood": b,
                              "distractor_ratio": sa["distractor"] / sb["distractor"],
                              "in_dist_ratio": sa["in_dist"] / sb["in_dist"]})
            if a < b and abs(math.log(sa["in_dist"] / sb["in_dist"])) <= math.log1p(match):
                matched.append({"a": a, "b": b, "distractor_log_ratio": math.log(sa["distractor"] / sb["distractor"])})
    span = max((abs(p["distractor_log_ratio"]) for p in matched), default=float("nan"))
    verdict = "fire" if fires else "kill" if matched and span <= math.log1p(kill_span) else "ambiguous"
    return {"verdict": verdict, "fires": fires, "fit_matched_pairs": matched,
            "fit_matched_distractor_log_span": span, "finite_candidates": len(finite)}


def svag_gamma(m, v, beta1, beta2, count):
    """Balles-Hennig factor gamma = mhat^2 / (mhat^2 + rho s) from Adam moments after `count` updates."""
    mhat = m / (1 - beta1 ** count)
    vhat = v / (1 - beta2 ** count)
    rho = (1 - beta1) * (1 + beta1 ** (count + 1)) / ((1 + beta1) * (1 - beta1 ** (count + 1)))
    s = (vhat - mhat.square()).clamp(min=0) / (1 - rho)
    return mhat.square() / (mhat.square() + rho * s + 1e-30)


def gamma_summary(learner, relevant, input_dim, count):
    """Mean gamma on first-layer relevant / distractor columns and per layer, per candidate."""
    beta1, beta2 = learner.hyper[1], learner.hyper[2]
    gammas = [svag_gamma(m, v, beta1, beta2, count) for m, v in zip(learner.m, learner.v)]
    first = gammas[0]
    return {"first_layer_relevant": first[..., :relevant].mean(dim=(-1, -2)).tolist(),
            "first_layer_distractor": first[..., relevant:input_dim].mean(dim=(-1, -2)).tolist(),
            "per_layer": [g.mean(dim=(-1, -2)).tolist() for g in gammas]}


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
    root = Path(args.output or f"runs/StructuredGeneralizationDiag__v2__{args.seed}__{time.time_ns()}")
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
        names = [name for name, _ in rows]
        learner = eval8.RoundLearner(initial, [config for _, config in rows], scenario, method)
        learner.x.copy_(xs[:1])
        learner.target.copy_(targets[:1])
        learner.action_noise.copy_(action_noise[:1])
        learner.reward_noise.copy_(ys_noise[:1])
        started = time.perf_counter()
        graph = learner.capture()
        torch.cuda.synchronize()
        startup = time.perf_counter() - started
        path = [torch.zeros(len(rows), device="cuda") for _ in learner.weights]
        curves = []
        started = time.perf_counter()
        for t in range(n):
            learner.x.copy_(xs[t:t + 1])
            learner.target.copy_(targets[t:t + 1])
            learner.action_noise.copy_(action_noise[t:t + 1])
            learner.reward_noise.copy_(ys_noise[t:t + 1])
            graph.replay()
            accumulate_path(path, learner.weights, learner.previous)
            if (t + 1) % log_rounds:
                continue
            risks = {key: clean_risk(learner.weights, x, y) for key, (x, y) in held.items()}
            for value in risks.values():
                learner.valid.logical_and_(torch.isfinite(value))
            relevant_energy, distractor_energy = column_energy(learner.weights[0], args.relevant, args.input_dim)
            distance = sum((w - w0.unsqueeze(0)).square().sum(dim=(-1, -2)) for w, w0 in zip(learner.weights, initial))
            gamma = gamma_summary(learner, args.relevant, args.input_dim, t + 1) if method == "adamw" else None
            curves.append({"step": t + 1, **{f"risk_{key}": value.tolist() for key, value in risks.items()},
                           "svag_gamma": gamma,
                           "path_l1": torch.stack(path).sum(0).tolist(),
                           "path_l1_per_layer": [p.tolist() for p in path],
                           "distance_from_init_sq": distance.tolist(),
                           "first_layer_relevant_energy": relevant_energy.tolist(),
                           "first_layer_distractor_energy": distractor_energy.tolist(),
                           "valid": learner.valid.tolist()})
        torch.cuda.synchronize()
        result["seconds"][method] = {"startup": startup, "training": time.perf_counter() - started}
        result["rows"][method] = curves
        for index, (name, config) in enumerate(rows):
            valid = curves[-1]["valid"][index]
            sustained = {key: (sum(c[f"risk_{key}"][index] for c in curves) / len(curves) if valid else float("nan"))
                         for key in held}
            last = curves[-1]
            gamma_curve = [c["svag_gamma"] for c in curves if c["svag_gamma"] is not None]
            gamma_mean = ({key: sum(g[key][index] for g in gamma_curve) / len(gamma_curve)
                           for key in ("first_layer_relevant", "first_layer_distractor")}
                          if gamma_curve else None)
            result["candidates"][name] = {
                "method": method, "config": config, "valid": valid, **sustained, "svag_gamma_mean": gamma_mean,
                "endpoint_in_dist": last["risk_in_dist"][index],
                "path_l1": last["path_l1"][index], "distance_from_init_sq": last["distance_from_init_sq"][index],
                "first_layer_relevant_energy": last["first_layer_relevant_energy"][index],
                "first_layer_distractor_energy": last["first_layer_distractor_energy"][index]}
    summary = result["candidates"]
    result["decision"] = decide(summary)
    for method in grid:
        own = {name: s for name, s in summary.items() if s["method"] == method and s["valid"]}
        best = min(own, key=lambda name: own[name]["in_dist"]) if own else None
        result["decision"][f"best_in_dist_{method}"] = best
    save_json(root / "results.json", result)
    print(json.dumps({"root": str(root), "seconds": result["seconds"], "decision": result["decision"]}))
    for name, s in summary.items():
        print(json.dumps({"candidate": name, "in_dist": round(s["in_dist"], 5), "distractor": round(s["distractor"], 5),
                          "relevant": round(s["relevant"], 5),
                          "ood_ratio": round(s["distractor"] / s["in_dist"], 3) if s["valid"] else None,
                          "gamma_rel": round(s["svag_gamma_mean"]["first_layer_relevant"], 3) if s["svag_gamma_mean"] else None,
                          "gamma_dis": round(s["svag_gamma_mean"]["first_layer_distractor"], 3) if s["svag_gamma_mean"] else None,
                          "path_l1": round(s["path_l1"], 1), "dist2": round(s["distance_from_init_sq"], 3),
                          "w1_rel": round(s["first_layer_relevant_energy"], 3),
                          "w1_dis": round(s["first_layer_distractor_energy"], 3)}))


if __name__ == "__main__":
    main()

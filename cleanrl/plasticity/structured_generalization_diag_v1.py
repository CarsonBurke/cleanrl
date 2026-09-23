"""Structured-teacher generalization diagnostic: does the optimizer change WHAT is learned, or only how fast?

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

Pre-registered rule (RethinkD.md section 6). Fire if some candidate's sustained
distractor risk is >= 20% below another's while its in-distribution risk is
equal or worse (a generalization difference not explained by fit). Kill if the
distractor/in-distribution ratio spans < 10% across all finite candidates (the
optimizer only sets the fit; the out-of-distribution penalty is a property of
the fit alone). Between: report, no build.
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
WEIGHT_DECAYS = (0.0, 0.1)
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


def decide(summary, fire_margin=0.2, kill_span=0.1):
    """Apply the pre-registered rule to per-candidate sustained risks."""
    finite = {name: s for name, s in summary.items()
              if all(math.isfinite(s[key]) for key in ("in_dist", "distractor", "relevant"))}
    fires = []
    for a, sa in finite.items():
        for b, sb in finite.items():
            if a != b and sa["in_dist"] >= sb["in_dist"] and sa["distractor"] <= (1 - fire_margin) * sb["distractor"]:
                fires.append({"better_ood": a, "worse_ood": b,
                              "distractor_ratio": sa["distractor"] / sb["distractor"],
                              "in_dist_ratio": sa["in_dist"] / sb["in_dist"]})
    ratios = [s["distractor"] / s["in_dist"] for s in finite.values()]
    span = max(ratios) / min(ratios) - 1 if ratios else float("nan")
    verdict = "fire" if fires else "kill" if span < kill_span else "ambiguous"
    return {"verdict": verdict, "fires": fires, "distractor_over_in_dist_span": span, "finite_candidates": len(finite)}


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    runtime.configure_runtime()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    scenario = {**SCENARIO, "samples": args.samples}
    root = Path(args.output or f"runs/StructuredGeneralizationDiag__v1__{args.seed}__{time.time_ns()}")
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
    result = {"args": asdict(args), "scenario": scenario, "candidates": {}, "rows": {}, "seconds": {}}
    for method, rows in grid.items():
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
            curves.append({"step": t + 1, **{f"risk_{key}": value.tolist() for key, value in risks.items()},
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
            result["candidates"][name] = {
                "method": method, "config": config, "valid": valid, **sustained,
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
                          "path_l1": round(s["path_l1"], 1), "dist2": round(s["distance_from_init_sq"], 3),
                          "w1_rel": round(s["first_layer_relevant_energy"], 3),
                          "w1_dis": round(s["first_layer_distractor_energy"], 3)}))


if __name__ == "__main__":
    main()

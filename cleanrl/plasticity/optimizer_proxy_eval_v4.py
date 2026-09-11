"""Discriminative optimizer proxies: minibatch/reuse/drift and signed policy loss.

Hypothesis: missing Jacobian drift and raw-gradient scaling misrank transported
momentum outside B=1 regression. Tune AdamW equally; choose on noisy validation,
lock all choices before test scoring, and retain negative results. Not an RL
benchmark or a guarantee that synthetic rankings transfer to MuJoCo.
"""

import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import optimizer_proxy_model_v4 as model
from cleanrl.plasticity.covariance_sketch_eval_v3 import save_json, tensor_hash
from cleanrl.shared import runtime


@dataclass
class Args:
    scenario: str = "drifting_reuse"
    seed: int = 1
    hidden: int = 64
    input_dim: int = 17
    validation: int = 2048
    test: int = 8192
    output: str = ""
    plan: str = "benchmarks/plasticity/optimizer_proxy_v4_plan.json"


def generator(seed, namespace):
    return torch.Generator(device="cuda").manual_seed(seed + namespace * 100003)


def task_at(teachers, x, progress, drift):
    if not drift:
        return x, reference.teach(teachers[0], x)
    position = min(progress * 8, 8.0)
    index = min(int(position), 7)
    fraction = position - index
    shift = 0.5 * math.sin(2 * math.pi * progress * 2)
    shifted = x + shift
    y0 = reference.teach(teachers[index], shifted)
    y1 = reference.teach(teachers[index + 1], shifted)
    return shifted, torch.lerp(y0, y1, fraction)


def training_task(teachers, xs, batch_size, drift):
    """Generate exogenous contexts/targets once, not once per optimizer family."""
    if not drift:
        return xs, reference.teach(teachers[0], xs)
    count = len(xs)
    rounds = count // batch_size
    progress = torch.arange(count, device=xs.device).div(batch_size, rounding_mode="floor") / rounds
    shifted = xs + (0.5 * torch.sin(4 * math.pi * progress))[:, None]
    target = torch.empty(count, device=xs.device)
    for segment in range(8):
        start, stop = segment * count // 8, (segment + 1) * count // 8
        x = shifted[start:stop]
        fraction = progress[start:stop] * 8 - segment
        target[start:stop] = torch.lerp(reference.teach(teachers[segment], x),
                                        reference.teach(teachers[segment + 1], x), fraction)
    return shifted, target


class RoundLearner:
    """One captured fresh-data round; policy data are frozen across its epochs."""

    def __init__(self, initial, grid, scenario, method):
        self.method, self.scenario = method, scenario
        self.weights = [w.unsqueeze(0).repeat(len(grid), 1, 1) for w in initial]
        self.previous = [w.clone() for w in self.weights]
        self.m = [torch.zeros_like(w) for w in self.weights]
        self.v = [torch.zeros_like(w) for w in self.weights]
        self.step = torch.zeros((), dtype=torch.int64, device="cuda")
        self.valid = torch.ones(len(grid), dtype=torch.bool, device="cuda")
        self.hyper = [torch.tensor([g[key] for g in grid], device="cuda")[:, None, None]
                      for key in ("lr", "beta1", "beta2", "weight_decay")]
        b, d = scenario["batch_size"], initial[0].shape[-1] - 1
        self.x = torch.zeros(b, d, device="cuda")
        self.target = torch.zeros(b, device="cuda")
        self.action_noise = torch.zeros(b, device="cuda")
        self.reward_noise = torch.zeros(b, device="cuda")
        self.actions = torch.zeros(len(grid), b, device="cuda")
        self.old_logprob = torch.zeros_like(self.actions)
        self.advantages = torch.zeros_like(self.actions)
        self.mutable = [*self.weights, *self.previous, *self.m, *self.v, self.step,
                        self.valid, self.actions, self.old_logprob, self.advantages]

    def prepare(self):
        if self.scenario["objective"] == "ppo":
            mu = model.forward(self.weights, self.x)
            actions = mu + 0.5 * self.action_noise
            rewards = -(actions - self.target).square() + self.reward_noise
            return actions, (-0.5 * self.action_noise.square()
                             - math.log(0.5 * math.sqrt(2 * math.pi))).expand_as(actions), rewards - rewards.mean(-1, keepdim=True)
        return self.actions, self.old_logprob, self.advantages

    def compute(self):
        grads, corrections = model.gradients(
            self.weights, self.previous, self.x, self.target, self.actions,
            self.old_logprob, self.advantages, self.scenario["objective"], self.method)
        lr, beta1, beta2, decay = self.hyper
        return model.transition(self.weights, self.previous, self.m, self.v,
                                self.step, grads, corrections, lr, beta1, beta2, decay, self.method)

    def commit(self, following):
        weights, m, v, step = following
        for previous, current in zip(self.previous, self.weights):
            previous.copy_(current)
        for destination, source in zip([*self.weights, *self.m, *self.v, self.step], [*weights, *m, *v, step]):
            destination.copy_(source)
        for tensor in [*self.weights, *self.m, *self.v]:
            self.valid.logical_and_(torch.isfinite(tensor).all(dim=(-1, -2)))

    def round(self, prepare, compute):
        for destination, source in zip((self.actions, self.old_logprob, self.advantages), prepare()):
            destination.copy_(source)
        for _ in range(self.scenario["epochs"]):
            self.commit(compute())

    def restore(self, state):
        for destination, source in zip(self.mutable, state):
            destination.copy_(source)

    def capture(self):
        before = [t.clone() for t in self.mutable]
        prepare = torch.compile(self.prepare, fullgraph=True, options={"triton.cudagraphs": False})
        compute = torch.compile(self.compute, fullgraph=True, options={"triton.cudagraphs": False})
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.round(prepare, compute)
                self.restore(before)
                self.round(prepare, compute)
            torch.cuda.current_stream().wait_stream(stream)
            self.restore(before)
            self.round(self.prepare, self.compute)
            eager = [t.clone() for t in self.mutable]
            self.restore(before)
            self.round(prepare, compute)
            torch.cuda.synchronize()
            torch.testing.assert_close(self.valid, eager[-4], rtol=0, atol=0)
            valid = self.valid
            for actual, expected in zip(self.mutable, eager):
                if actual.ndim and actual.shape[0] == len(valid):
                    actual, expected = actual[valid], expected[valid]
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-5, equal_nan=True)
            self.restore(before)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.round(prepare, compute)
            self.restore(before)
            graph.replay()
            first = [t.clone() for t in self.mutable]
            self.restore(before)
            graph.replay()
            torch.cuda.synchronize()
            for actual, expected in zip(self.mutable, first):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            return graph
        finally:
            self.restore(before)


@torch.no_grad()
def observation_loss(weights, x, clean, noise, policy_noise, objective):
    output = model.forward(weights, x)
    if objective == "regression":
        observed = (output - (clean + noise)).square().mean(-1)
    else:
        # Negative realized reward from fresh on-policy validation actions.
        observed = ((output + 0.5 * policy_noise - clean).square() - noise).mean(-1)
    return observed, (output - clean).square().mean(-1)


def edge_flags(chosen, grid_axes):
    flags = {}
    for key, values in grid_axes.items():
        if not isinstance(values, list):
            continue
        value = chosen[key]
        # Zero is the physical no-momentum/no-decay endpoint, not a missing region.
        flags[key] = value == max(values) or (value == min(values) and value != 0)
    return flags


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    plan = json.loads(Path(args.plan).read_text())
    scenario = next(s for s in plan["suite"] if s["name"] == args.scenario)
    runtime.configure_runtime()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    initial = reference.init_weights(args, generator(args.seed, 1), "cuda")
    teacher_gen = generator(args.seed, 2)
    teachers = [reference.draw_teacher(args, teacher_gen, "cuda") for _ in range(9)]
    n, batch = scenario["samples"], scenario["batch_size"]
    rounds, log_rounds = n // batch, (n // batch) // 16
    train_gen = generator(args.seed, 3)
    xs = torch.randn(n, args.input_dim, generator=train_gen, device="cuda")
    ys_noise = torch.randn(n, generator=train_gen, device="cuda")
    action_noise = torch.randn(n, generator=train_gen, device="cuda")
    xs, train_targets = training_task(teachers, xs, batch, scenario["drift"])
    if scenario["objective"] == "regression":
        train_targets = train_targets + ys_noise
    validation_gen = generator(args.seed, 4)
    validation = [(torch.randn(args.validation, args.input_dim, generator=validation_gen, device="cuda"),
                   torch.randn(args.validation, generator=validation_gen, device="cuda"),
                   torch.randn(args.validation, generator=validation_gen, device="cuda")) for _ in range(16)]
    keys = ("lr", "beta1", "beta2", "weight_decay")
    grid = [dict(zip(keys, values)) for values in itertools.product(*(plan["grid"][key] for key in keys))]
    root = Path(args.output or f"runs/OptimizerProxy__v4_{args.scenario}__{args.seed}__{time.time_ns()}")
    root.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(model.__file__), Path(reference.__file__), Path(args.plan), Path(runtime.__file__)]
    hashes = {str(p.resolve().relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = {"args": asdict(args), "scenario": scenario, "grid": grid, "status": "training",
              "source_sha256": hashes, "initial_sha256": [tensor_hash(w) for w in initial],
              "train_data_sha256": [tensor_hash(t) for t in (xs, train_targets, ys_noise, action_noise)],
              "protocol": plan, "methods": {}, "selections": {},
              "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "device": torch.cuda.get_device_name(), "precision": "FP32 TF32 off"}}
    writer = SummaryWriter(str(root))
    writer.add_text("protocol", json.dumps(plan))
    save_json(root / "results.json", result)
    try:
        for method in plan["families"]:
            learner = RoundLearner(initial, grid, scenario, method)
            # Nontrivial first real batch for the capture audit; capture restores all state.
            learner.x.copy_(xs[:batch])
            learner.target.copy_(train_targets[:batch])
            learner.action_noise.copy_(action_noise[:batch])
            learner.reward_noise.copy_(ys_noise[:batch])
            started = time.perf_counter()
            graph = learner.capture()
            torch.cuda.synchronize()
            row = {"startup_seconds": time.perf_counter() - started, "curves": [], "status": "training"}
            result["methods"][method] = row
            folder = root / "checkpoints" / method
            folder.mkdir(parents=True)
            scores = torch.zeros(len(grid), device="cuda")
            started = time.perf_counter()
            for round_index in range(rounds):
                offset = round_index * batch
                learner.x.copy_(xs[offset:offset + batch])
                learner.target.copy_(train_targets[offset:offset + batch])
                learner.action_noise.copy_(action_noise[offset:offset + batch])
                learner.reward_noise.copy_(ys_noise[offset:offset + batch])
                graph.replay()
                if (round_index + 1) % log_rounds:
                    continue
                checkpoint_index = (round_index + 1) // log_rounds - 1
                progress = (round_index + 1) / rounds
                vx, noise, pnoise = validation[checkpoint_index]
                vx, vy = task_at(teachers, vx, progress, scenario["drift"])
                observed, clean = observation_loss(learner.weights, vx, vy, noise, pnoise, scenario["objective"])
                learner.valid.logical_and_(torch.isfinite(observed))
                scores.add_(observed / 16)
                fit = model.output_loss(model.forward(learner.weights, learner.x), learner.target,
                                        learner.actions, learner.old_logprob, learner.advantages,
                                        scenario["objective"])
                step = (round_index + 1) * batch
                checkpoint = folder / f"{step}.pt"
                torch.save({"weights": [w.cpu() for w in learner.weights],
                            "previous": [w.cpu() for w in learner.previous],
                            "m": [m.cpu() for m in learner.m],
                            "v": [v.cpu() for v in learner.v], "optimizer_step": learner.step.cpu(),
                            "batch": {key: getattr(learner, key).cpu() for key in
                                      ("x", "target", "actions", "old_logprob", "advantages")},
                            "progress": progress}, checkpoint)
                curve = {"step": step, "progress": progress, "validation": observed.cpu().tolist(),
                         "clean_diagnostic": clean.cpu().tolist(), "same_batch_objective": fit.cpu().tolist(),
                         "valid": learner.valid.cpu().tolist(), "checkpoint": str(checkpoint.relative_to(root))}
                row["curves"].append(curve)
                finite = learner.valid & torch.isfinite(scores)
                best = int(scores.masked_fill(~finite, float("inf")).argmin().item())
                writer.add_scalar(f"validation/{method}", float(observed[best]), step)
                writer.add_scalar(f"diagnostic_clean/{method}", float(clean[best]), step)
                writer.add_scalar(f"finite_candidates/{method}", int(finite.sum()), step)
                writer.flush()
                save_json(root / "results.json", result)
                print(json.dumps({"method": method, "step": step, "best_running_validation": float(scores[best]),
                                  "best_config": grid[best], "finite": int(finite.sum())}), flush=True)
            torch.cuda.synchronize()
            row["training_seconds"] = time.perf_counter() - started
            row["status"] = "completed"
            row["candidate_validation_scores"] = scores.cpu().tolist()
            valid = learner.valid & torch.isfinite(scores)
            for name in (["adamw", "adam"] if method == "adamw" else [method]):
                eligible = valid.clone()
                if name == "adam":
                    eligible &= torch.tensor([g["weight_decay"] == 0 for g in grid], device="cuda")
                if not bool(eligible.any()):
                    result["selections"][name] = {"status": "failed_all_candidates", "source_method": method}
                    continue
                index = int(scores.masked_fill(~eligible, float("inf")).argmin().item())
                choice = {"status": "selected", "source_method": method, "index": index, "config": grid[index],
                          "validation": float(scores[index]), "edge_flags": edge_flags(grid[index], plan["grid"])}
                with (root / f"selection_{name}.json").open("x") as file:
                    json.dump(choice, file, indent=2)
                result["selections"][name] = choice
            save_json(root / "results.json", result)
            del graph, learner
            torch.cuda.empty_cache()
        # All selections are locked before any held-out test draw or score.
        result["status"] = "locked_test_scoring"
        save_json(root / "results.json", result)
        test_gen = generator(args.seed, 5)
        test_draws = [(torch.randn(args.test, args.input_dim, generator=test_gen, device="cuda"),
                       torch.randn(args.test, generator=test_gen, device="cuda"),
                       torch.randn(args.test, generator=test_gen, device="cuda")) for _ in range(16)]
        for name, choice in result["selections"].items():
            if choice["status"] != "selected":
                continue
            locked = json.loads((root / f"selection_{name}.json").read_text())
            assert locked == choice
            test_curve = []
            for index, curve in enumerate(result["methods"][choice["source_method"]]["curves"]):
                checkpoint = torch.load(root / curve["checkpoint"], map_location="cpu", weights_only=True)
                weights = [w[choice["index"]:choice["index"] + 1].cuda() for w in checkpoint["weights"]]
                tx, noise, pnoise = test_draws[index]
                tx, target = task_at(teachers, tx, curve["progress"], scenario["drift"])
                observed, clean = observation_loss(weights, tx, target, noise, pnoise, scenario["objective"])
                if not bool(torch.isfinite(observed).all() & torch.isfinite(clean).all()):
                    raise FloatingPointError(f"nonfinite held-out score: {name}")
                selected = slice(choice["index"], choice["index"] + 1)
                previous = [w[selected].cuda() for w in checkpoint["previous"]]
                prediction = model.forward(previous, tx)
                # Diagnostic oracle only: deployment half-MSE or expected negative
                # bandit reward gradient, not a learner input or selection score.
                factor = 2.0 if scenario["objective"] == "ppo" else 1.0
                oracle = model.backward(previous, tx, factor * (prediction - target) / args.test)
                data = checkpoint["batch"]
                batch_args = [data[key][selected].cuda() for key in ("actions", "old_logprob", "advantages")]
                raw_score = model.output_score(model.forward(previous, data["x"].cuda()),
                                               data["target"].cuda(), *batch_args, scenario["objective"])
                raw = model.backward(previous, data["x"].cuda(), raw_score)
                beta = choice["config"]["beta1"]
                mass = 1.0 if beta == 0 else -math.expm1(math.log(beta) * int(checkpoint["optimizer_step"]))
                moment = [m[selected].cuda() / mass for m in checkpoint["m"]]
                truth = torch.cat([g.flatten() for g in oracle])
                diagnostics = {}
                for label, tensors in (("raw", raw), ("moment", moment)):
                    vector = torch.cat([g.flatten() for g in tensors])
                    diagnostics[label] = {
                        "relative_squared_error": float((vector - truth).square().sum() / truth.square().sum().clamp_min(1e-30)),
                        "cosine": float((vector * truth).sum() / (vector.norm() * truth.norm()).clamp_min(1e-30)),
                    }
                test_curve.append({"step": curve["step"], "observed_loss": float(observed[0]),
                                   "clean_excess_risk": float(clean[0]), "deployment_gradient_diagnostic": diagnostics})
            choice["test_curve"] = test_curve
            choice["test_sustained_risk"] = sum(c["clean_excess_risk"] for c in test_curve) / len(test_curve)
            choice["test_endpoint_risk"] = test_curve[-1]["clean_excess_risk"]
            print(json.dumps({"selection": name, **choice}), flush=True)
        for path, digest in hashes.items():
            if hashlib.sha256((project / path).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"source changed during experiment: {path}")
        result["status"] = "completed"
        save_json(root / "results.json", result)
    finally:
        writer.close()
    print(f"RESULTS {root / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

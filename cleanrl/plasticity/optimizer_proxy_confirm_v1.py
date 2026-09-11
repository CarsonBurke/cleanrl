"""Locked optimizer confirmation on fresh proxy realizations, with no retuning.

Reuse v8 learners and task definitions. A prewritten plan fixes each method's
configuration before fresh train/test draws. New RNG namespaces under seed 1
prevent repeated development draws from serving as untouched confirmation.
No seed-level significance or MuJoCo claim follows from these synthetic tasks.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import optimizer_proxy_eval_v8 as protocol
from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity.covariance_sketch_eval_v3 import save_json, tensor_hash
from cleanrl.shared import runtime


@dataclass
class Args:
    plan: str = "benchmarks/plasticity/optimizer_proxy_confirmation_plan.json"
    seed: int = 1
    hidden: int = 64
    input_dim: int = 17
    test: int = 8192
    output: str = ""


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    plan_path = Path(args.plan)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    runtime.configure_runtime()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    namespace = plan["namespace"]
    root = Path(args.output or f"runs/OptimizerProxyConfirm__v1__{args.seed}__{time.time_ns()}")
    root.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(protocol.__file__), Path(protocol.optimizer.__file__),
               Path(protocol.optimizer.base.__file__), Path(protocol.optimizer.base.base.__file__),
               Path(protocol.model.__file__), Path(reference.__file__), Path(runtime.__file__),
               Path(protocol.reporting.__file__), plan_path]
    hashes = {str(p.resolve().relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = {"status": "training", "args": asdict(args), "locked_plan": plan,
              "source_sha256": hashes, "cases": {}}
    writer = SummaryWriter(str(root))
    writer.add_text("locked_plan", plan_bytes.decode())
    save_json(root / "results.json", result)
    try:
        for case in plan["cases"]:
            scenario = case["scenario"]
            name = scenario["name"]
            # Each scenario uses the same underlying fresh teacher/initialization,
            # just as development did; methods within each case share every draw.
            initial = reference.init_weights(args, protocol.generator(args.seed, namespace + 1), "cuda")
            teacher_gen = protocol.generator(args.seed, namespace + 2)
            teachers = [reference.draw_teacher(args, teacher_gen, "cuda") for _ in range(9)]
            train_gen = protocol.generator(args.seed, namespace + 3)
            count, batch = scenario["samples"], scenario["batch_size"]
            rounds = count // batch
            xs = torch.randn(count, args.input_dim, generator=train_gen, device="cuda")
            noise = torch.randn(count, generator=train_gen, device="cuda")
            pnoise = torch.randn(count, generator=train_gen, device="cuda")
            xs, targets = protocol.training_task(teachers, xs, batch, scenario["drift"])
            if scenario["objective"] == "regression":
                targets = targets + noise
            row = {"scenario": scenario, "initial_sha256": [tensor_hash(w) for w in initial],
                   "train_sha256": [tensor_hash(t) for t in (xs, targets, noise, pnoise)], "methods": {}}
            result["cases"][name] = row
            for method, trial in case["locked"].items():
                algorithm, config = trial["method"], trial["config"]
                torch.compiler.reset()
                learner = protocol.RoundLearner(initial, [config], scenario, algorithm)
                learner.x.copy_(xs[:batch])
                learner.target.copy_(targets[:batch])
                learner.action_noise.copy_(pnoise[:batch])
                learner.reward_noise.copy_(noise[:batch])
                started = time.perf_counter()
                graph = learner.capture()
                torch.cuda.synchronize()
                arm = {"algorithm": algorithm, "config": config, "startup_seconds": time.perf_counter() - started,
                       "checkpoints": [], "status": "training"}
                row["methods"][method] = arm
                started = time.perf_counter()
                for r in range(rounds):
                    start = r * batch
                    learner.x.copy_(xs[start:start+batch])
                    learner.target.copy_(targets[start:start+batch])
                    learner.action_noise.copy_(pnoise[start:start+batch])
                    learner.reward_noise.copy_(noise[start:start+batch])
                    graph.replay()
                    if (r+1) % (rounds//16):
                        continue
                    if not bool(learner.valid[0]):
                        raise FloatingPointError(f"locked candidate failed: {name}/{method}")
                    step = (r+1)*batch
                    checkpoint = root / "checkpoints" / name / method / f"{step}.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"weights": [w.cpu() for w in learner.weights],
                                "config": config, "step": step}, checkpoint)
                    arm["checkpoints"].append({"step": step, "progress": (r+1)/rounds,
                                               "path": str(checkpoint.relative_to(root))})
                    writer.add_scalar(f"progress/{name}/{method}", step, step)
                torch.cuda.synchronize()
                arm["training_seconds"] = time.perf_counter() - started
                arm["status"] = "completed"
                arm["optimizer_steps"] = int(learner.step)
                arm["fresh_samples"] = count
                save_json(root / "results.json", result)
                print(json.dumps({"case": name, "method": method, "trained_samples": count}), flush=True)
                del graph, learner
                torch.cuda.empty_cache()
            # Locked before starting; test draws cannot select an optimizer or HP.
            test_gen = protocol.generator(args.seed, namespace + 5)
            draws = [(torch.randn(args.test, args.input_dim, generator=test_gen, device="cuda"),
                      torch.randn(args.test, generator=test_gen, device="cuda"),
                      torch.randn(args.test, generator=test_gen, device="cuda")) for _ in range(16)]
            for method, arm in row["methods"].items():
                curve = []
                for index, checkpoint in enumerate(arm["checkpoints"]):
                    data = torch.load(root / checkpoint["path"], map_location="cpu", weights_only=True)
                    weights = [w.cuda() for w in data["weights"]]
                    x, test_noise, policy_noise = draws[index]
                    x, target = protocol.task_at(teachers, x, checkpoint["progress"], scenario["drift"])
                    observed, clean = protocol.observation_loss(weights, x, target, test_noise, policy_noise,
                                                               scenario["objective"])
                    if not bool(torch.isfinite(observed).all() & torch.isfinite(clean).all()):
                        raise FloatingPointError("nonfinite locked test score")
                    curve.append({"step": checkpoint["step"], "observed_loss": float(observed[0]),
                                  "clean_excess_risk": float(clean[0])})
                    writer.add_scalar(f"heldout/{name}/{method}", float(clean[0]), checkpoint["step"])
                arm["test_curve"] = curve
                arm["test_sustained_risk"] = sum(c["clean_excess_risk"] for c in curve)/len(curve)
                arm["test_endpoint_risk"] = curve[-1]["clean_excess_risk"]
            control = row["methods"]["adamw"]["test_sustained_risk"]
            for arm in row["methods"].values():
                arm["relative_to_adamw_pct"] = 100*(arm["test_sustained_risk"]/control-1)
            print(json.dumps({"case": name, "scores": {method: {key: arm[key] for key in
                  ("test_sustained_risk", "test_endpoint_risk", "relative_to_adamw_pct", "training_seconds")}
                  for method,arm in row["methods"].items()}}), flush=True)
            writer.flush()
            save_json(root / "results.json", result)
        for path, expected in hashes.items():
            if hashlib.sha256((project / path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"source changed during confirmation: {path}")
        result["status"] = "completed"
        save_json(root / "results.json", result)
    finally:
        writer.close()
    print(f"RESULTS {root / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

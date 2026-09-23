"""Optimizer proxies with per-family grids and auxiliary consolidation state.

Hypothesis: two-tier prospective consolidation with innovation-whitened gains
lowers sustained held-out excess risk against independently tuned AdamW and the
retained polar frontier, and its uniform-gain and scalar-gain controls do not.
Same three regimes, model, noisy-validation locking and held-out scoring as v8;
families may carry extra hyperparameter axes and their own configuration lists.
Tier families deploy the consolidated tier at every block boundary, which is
where validation, checkpoints and bandit rollouts occur. Not an RL benchmark.
"""

import hashlib
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
from cleanrl.plasticity import optimizer_proxy_model_v9 as optimizer
from cleanrl.plasticity import covariance_sketch_eval_v3 as reporting
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
    plan: str = "benchmarks/plasticity/optimizer_proxy_v9_plan.json"
    families: str = ""  # comma-separated override of the plan's family list (one family per job keeps runs short)


HYPER_KEYS = ("lr", "beta1", "beta2", "weight_decay", "head_lr_scale", *optimizer.TIER_KEYS)


def family_grid(plan, family, scenario_name):
    """Explicit per-family configurations (optionally per scenario), defaulting to the shared list."""
    configs = plan.get("family_configurations", {}).get(family, plan["configurations"])
    if isinstance(configs, dict):
        configs = configs[scenario_name]
    return [{"head_lr_scale": 1.0, **config} for config in configs]


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
        self.aux = optimizer.initial_aux(self.weights)
        self.step = torch.zeros((), dtype=torch.int64, device="cuda")
        self.valid = torch.ones(len(grid), dtype=torch.bool, device="cuda")
        self.hyper = {key: torch.tensor([g[key] for g in grid], device="cuda", dtype=torch.float32)[:, None, None]
                      for key in HYPER_KEYS if all(key in g for g in grid)}
        for key in ("lr", "beta1", "beta2", "weight_decay", "head_lr_scale"):
            assert key in self.hyper, f"missing hyperparameter axis: {key}"
        if optimizer.is_tier(method):
            for key in optimizer.TIER_KEYS:
                assert key in self.hyper, f"missing tier axis: {key}"
            for state in self.aux:
                state[:, 1].copy_(self.hyper["tier_k0"].expand_as(state[:, 1]))
        self.block = scenario.get("block", scenario["epochs"])
        # Deployment (validation, checkpoints, bandit rollouts) must land on block boundaries.
        assert scenario["epochs"] % self.block == 0 or self.block % scenario["epochs"] == 0
        if scenario["objective"] == "ppo":
            assert scenario["epochs"] % self.block == 0, "bandit rollouts must start at a consolidated tier"
        b, d = scenario["batch_size"], initial[0].shape[-1] - 1
        self.x = torch.zeros(b, d, device="cuda")
        self.target = torch.zeros(b, device="cuda")
        self.action_noise = torch.zeros(b, device="cuda")
        self.reward_noise = torch.zeros(b, device="cuda")
        self.actions = torch.zeros(len(grid), b, device="cuda")
        self.old_logprob = torch.zeros_like(self.actions)
        self.advantages = torch.zeros_like(self.actions)
        self.mutable = [*self.weights, *self.previous, *self.m, *self.v, *self.aux, self.step,
                        self.valid, self.actions, self.old_logprob, self.advantages,
                        self.x, self.target, self.action_noise, self.reward_noise]

    def prepare(self):
        if self.scenario["objective"] == "ppo":
            mu = model.forward(self.weights, self.x)
            actions = mu + 0.5 * self.action_noise
            rewards = -(actions - self.target).square() + self.reward_noise
            return actions, (-0.5 * self.action_noise.square()
                             - math.log(0.5 * math.sqrt(2 * math.pi))).expand_as(actions), rewards - rewards.mean(-1, keepdim=True)
        return self.actions, self.old_logprob, self.advantages

    def compute(self):
        grads, corrections = optimizer.gradients(
            self.weights, self.previous, self.x, self.target, self.actions,
            self.old_logprob, self.advantages, self.scenario["objective"], self.method)
        return optimizer.transition(self.weights, self.previous, self.m, self.v, self.aux,
                                    self.step, grads, corrections, self.hyper,
                                    self.method, self.scenario["epochs"], self.block)

    def commit(self, following):
        weights, m, v, aux, step = following
        for previous, current in zip(self.previous, self.weights):
            previous.copy_(current)
        for destination, source in zip([*self.weights, *self.m, *self.v, *self.aux, self.step],
                                       [*weights, *m, *v, *aux, step]):
            destination.copy_(source)
        for tensor in [*self.weights, *self.m, *self.v]:
            self.valid.logical_and_(torch.isfinite(tensor).all(dim=(-1, -2)))
        for tensor in self.aux:
            self.valid.logical_and_(torch.isfinite(tensor).all(dim=(-1, -2, -3)))

    def gain_summary(self):
        """Per-candidate mean and dispersion of the consolidation gain over all parameters."""
        gains = torch.cat([state[:, 1].flatten(1) for state in self.aux], dim=1)
        return gains.mean(dim=1), gains.std(dim=1)

    def round(self, prepare, compute):
        for destination, source in zip((self.actions, self.old_logprob, self.advantages), prepare()):
            destination.copy_(source)
        for _ in range(self.scenario["epochs"]):
            self.commit(compute())

    def restore(self, state):
        for destination, source in zip(self.mutable, state):
            destination.copy_(source)

    @staticmethod
    def audit_equal(actual_state, expected_state):
        for actual, expected in zip(actual_state, expected_state):
            torch.testing.assert_close(torch.isfinite(actual), torch.isfinite(expected), rtol=0, atol=0)
            tolerance = {"rtol": 3e-3, "atol": 3e-5} if actual.is_floating_point() else {"rtol": 0, "atol": 0}
            torch.testing.assert_close(actual, expected, **tolerance, equal_nan=True)

    @staticmethod
    def audit_transition(before_state, actual_state, expected_state):
        """Audit same-state local updates, accounting only for FP32 input rounding."""
        for index, (before, actual, expected) in enumerate(zip(before_state, actual_state, expected_state, strict=True)):
            context = f"transition tensor {index}"
            assert actual.shape == expected.shape == before.shape, f"{context}: shape mismatch"
            assert actual.dtype == expected.dtype == before.dtype, f"{context}: dtype mismatch"
            finite = torch.isfinite(expected)
            torch.testing.assert_close(torch.isfinite(actual), finite, rtol=0, atol=0,
                                       msg=f"{context}: finite-mask mismatch")
            if not actual.is_floating_point():
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=context)
                continue
            # Preserve strict NaN/Inf equality, including infinity signs. These
            # candidates still retain the ordinary learner validity mask.
            torch.testing.assert_close(actual[~finite], expected[~finite], rtol=0, atol=0,
                                       equal_nan=True, msg=f"{context}: nonfinite-state mismatch")
            finite_before = torch.isfinite(before)
            # A nonfinite input has no defined local delta; any finite result
            # must therefore agree exactly rather than receive an infinite bound.
            recovered = finite & ~finite_before
            torch.testing.assert_close(actual[recovered], expected[recovered], rtol=0, atol=0,
                                       msg=f"{context}: finite recovery mismatch")
            compared = finite & finite_before
            actual_update = actual[compared] - before[compared]
            expected_update = expected[compared] - before[compared]
            error = (actual_update - expected_update).abs()
            allowance = (3e-5 + 3e-3 * expected_update.abs()
                         + 4 * torch.finfo(before.dtype).eps * before[compared].abs())
            if not bool((error <= allowance).all()):
                maximum = float((error / allowance).max())
                raise AssertionError(f"{context}: max normalized error {maximum:.9g} exceeds 1")

    def capture(self):
        before = [t.clone() for t in self.mutable]
        try:
            prepare = torch.compile(self.prepare, fullgraph=True, options={"triton.cudagraphs": False})
            compute = torch.compile(self.compute, fullgraph=True, options={"triton.cudagraphs": False})
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.round(prepare, compute)
                self.restore(before)
                self.round(prepare, compute)
            torch.cuda.current_stream().wait_stream(stream)
            self.restore(before)

            # Audit preparation separately; both paths see exactly the same state.
            eager_prepared = [t.clone() for t in self.prepare()]
            self.restore(before)
            compiled_prepared = prepare()
            self.audit_equal(compiled_prepared, eager_prepared)
            for destination, source in zip((self.actions, self.old_logprob, self.advantages), compiled_prepared):
                destination.copy_(source)

            # Each local transition starts from one common state. Carry the compiled
            # trajectory forward, not an independently drifting eager trajectory.
            for _ in range(self.scenario["epochs"]):
                epoch_before = [t.clone() for t in self.mutable]
                self.commit(self.compute())
                eager = [t.clone() for t in self.mutable]
                self.restore(epoch_before)
                self.commit(compute())
                self.audit_transition(epoch_before, self.mutable, eager)
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
            # A compile/audit/capture failure can leave side-stream work pending.
            # Drain it before rollback so it cannot overwrite the restored state.
            try:
                torch.cuda.synchronize()
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
        if len(values) < 2:
            continue
        flags[key] = value == max(values) or (value == min(values) and value != 0)
    return flags


def retain_selected_checkpoints(root, result, generated_checkpoints):
    """Prune full grids only after every locked trajectory has been saved."""
    curve_updates = []
    test_updates = []
    for method, row in result["methods"].items():
        selections = [(name, choice) for name, choice in result["selections"].items()
                      if choice["status"] == "selected" and choice["source_method"] == method]
        for index, curve in enumerate(row["curves"]):
            retained = {}
            if selections:
                checkpoint = torch.load(root / curve["checkpoint"], map_location="cpu", weights_only=True)
                for name, choice in selections:
                    selected = slice(choice["index"], choice["index"] + 1)
                    state = {key: [tensor[selected].clone() for tensor in checkpoint[key]]
                             for key in ("weights", "previous", "m")}
                    state.update({"optimizer_step": checkpoint["optimizer_step"],
                                  "progress": checkpoint["progress"], "source_method": method,
                                  "candidate_index": choice["index"], "config": choice["config"],
                                  "batch": {key: tensor if key in ("x", "target") else tensor[selected].clone()
                                            for key, tensor in checkpoint["batch"].items()}})
                    path = root / "checkpoints_selected" / name / f"{curve['step']}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as file:
                        torch.save(state, file)
                    retained[name] = str(path.relative_to(root))
                    test_updates.append((choice["test_curve"][index], retained[name]))
                del checkpoint, state
            curve_updates.append((curve, retained))
    # No full-grid artifact is touched if any selected write above fails.
    for curve, retained in curve_updates:
        del curve["checkpoint"]
        curve["selected_checkpoints"] = retained
    for curve, path in test_updates:
        curve["checkpoint"] = path
    result["checkpoint_retention"] = ("locked selections only (weights, previous, first moment); second moments and "
                                      "tier state are not checkpointed; full candidate curves and invalid masks retained")
    save_json(root / "results.json", result)
    for path in generated_checkpoints:
        path.unlink()


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
    block = scenario.get("block", scenario["epochs"])
    if (log_rounds * scenario["epochs"]) % block:
        raise ValueError(f"checkpoint spacing {log_rounds} rounds does not align with block {block}")
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
    families = (args.families.split(",") if args.families else
                plan.get("scenario_families", {}).get(args.scenario, plan["families"]))
    grids = {family: family_grid(plan, family, args.scenario) for family in families}
    grid_axes = {family: {key: sorted({config[key] for config in grids[family]})
                          for key in HYPER_KEYS if all(key in config for config in grids[family])}
                 for family in families}
    tag = f"_{args.families.replace(',', '+')}" if args.families else ""
    root = Path(args.output or f"runs/OptimizerProxy__v9_{args.scenario}{tag}__{args.seed}__{time.time_ns()}")
    root.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(optimizer.__file__), Path(optimizer.base.__file__),
               Path(optimizer.base.base.__file__), Path(optimizer.base.base.base.__file__),
               Path(optimizer.base.base.base.base.__file__), Path(model.__file__), Path(reference.__file__),
               Path(reporting.__file__), Path(args.plan), Path(runtime.__file__)]
    hashes = {str(p.resolve().relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = {"args": asdict(args), "scenario": scenario, "grids": grids, "status": "training",
              "source_sha256": hashes, "initial_sha256": [tensor_hash(w) for w in initial],
              "train_data_sha256": [tensor_hash(t) for t in (xs, train_targets, ys_noise, action_noise)],
              "protocol": plan, "families": families, "methods": {}, "selections": {},
              "tier_semantics": "tier families: weights are the transient tier within a block and the consolidated tier at block boundaries; aux slots are phi, gain, lag-1 innovation covariance, innovation power, previous innovation, velocity",
              "parameter_groups": {
                  "hidden_weights": {"learning_rate": "lr", "weight_decay": "weight_decay",
                                     "direction": "family-specific; excludes final bias column"},
                  "hidden_biases": {"learning_rate": "lr", "weight_decay": 0,
                                    "direction": "ordinary AdamW for matrix families; temporal-family state otherwise"},
                  "head_weights": {"learning_rate": "lr * head_lr_scale", "weight_decay": "weight_decay",
                                   "direction": "Adam-preconditioned; temporal-family state retained"},
                  "head_bias": {"learning_rate": "lr * head_lr_scale", "weight_decay": 0,
                                "direction": "Adam-preconditioned; temporal-family state retained"},
              },
              "deployment_gradient_diagnostic_semantics": {
                  "raw": "last-update minibatch gradient evaluated at previous weights",
                  "moment": "bias-corrected EMA state; not the applied polar/spectral step",
                  "applied_step": "(previous weights - current weights) / group learning rate; final layer including bias uses lr * head_lr_scale, hidden layers use lr; includes weight-column decay",
                  "tier_caveat": "for tier families `previous` is the last transient-tier state before the boundary step, so raw/oracle are evaluated at z and applied_step includes the consolidation jump (1-K)d; interpret only clean_excess_risk for those families",
              },
              "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "device": torch.cuda.get_device_name(), "precision": "FP32 TF32 off"}}
    generated_checkpoints = []
    writer = SummaryWriter(str(root))
    writer.add_text("protocol", json.dumps(plan))
    save_json(root / "results.json", result)
    try:
        for method in families:
            # Release prior-family Dynamo guards without clearing the Inductor disk cache.
            torch.compiler.reset()
            grid = grids[method]
            learner = RoundLearner(initial, grid, scenario, method)
            # Nontrivial first real batch for the capture audit; capture restores all state.
            learner.x.copy_(xs[:batch])
            learner.target.copy_(train_targets[:batch])
            learner.action_noise.copy_(action_noise[:batch])
            learner.reward_noise.copy_(ys_noise[:batch])
            started = time.perf_counter()
            graph = learner.capture()
            torch.cuda.synchronize()
            row = {"startup_seconds": time.perf_counter() - started,
                   "startup_scope": "compile, conditioned audit, capture and replay; excludes compiler reset and learner construction",
                   "curves": [], "status": "training"}
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
                            "m": [m.cpu() for m in learner.m], "optimizer_step": learner.step.cpu(),
                            "batch": {key: getattr(learner, key).cpu() for key in
                                      ("x", "target", "actions", "old_logprob", "advantages")},
                            "progress": progress, "grid": grid}, checkpoint)
                generated_checkpoints.append(checkpoint)
                curve = {"step": step, "progress": progress, "validation": observed.cpu().tolist(),
                         "clean_diagnostic": clean.cpu().tolist(), "same_batch_objective": fit.cpu().tolist(),
                         "valid": learner.valid.cpu().tolist(), "checkpoint": str(checkpoint.relative_to(root))}
                if optimizer.is_tier(method):
                    gain_mean, gain_std = learner.gain_summary()
                    curve["gain_mean"] = gain_mean.cpu().tolist()
                    curve["gain_std"] = gain_std.cpu().tolist()
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
            row["stream_counts"] = {"samples": (round_index + 1) * batch,
                                    "fresh_batches": round_index + 1,
                                    "optimizer_updates": int(learner.step),
                                    "validation_checkpoints": len(row["curves"]),
                                    "validation_samples": len(row["curves"]) * args.validation}
            if row["stream_counts"]["samples"] != n or int(learner.step) != rounds * scenario["epochs"]:
                raise RuntimeError(f"incomplete training stream: {method}")
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
                          "validation": float(scores[index]), "edge_flags": edge_flags(grid[index], grid_axes[method])}
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
                count = int(checkpoint["optimizer_step"])
                if choice["source_method"].startswith("round_"):
                    count = (count + scenario["epochs"] - 1) // scenario["epochs"]
                mass = 1.0 if beta == 0 else -math.expm1(math.log(beta) * count)
                # EMA state is not the applied polar/spectral direction. Measure
                # the actual parameter descent separately, including weight decay.
                moment = [m[selected].cuda() / mass for m in checkpoint["m"]]
                applied_step = [
                    (old - current) / (choice["config"]["lr"]
                                       * (choice["config"]["head_lr_scale"] if layer == len(weights) - 1 else 1.0))
                    for layer, (old, current) in enumerate(zip(previous, weights))
                ]
                truth = torch.cat([g.flatten() for g in oracle])
                diagnostics = {}
                for label, tensors in (("raw", raw), ("moment", moment), ("applied_step", applied_step)):
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
            choice["test_stream_counts"] = {"checkpoints": len(test_curve), "samples": len(test_curve) * args.test}
            print(json.dumps({"selection": name, **choice}), flush=True)
        after_hashes = {str(p.resolve().relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sources}
        result["source_sha256_after"] = after_hashes
        for path, digest in hashes.items():
            if after_hashes[path] != digest:
                save_json(root / "results.json", result)
                raise RuntimeError(f"source changed during experiment: {path}")
        retain_selected_checkpoints(root, result, generated_checkpoints)
        result["status"] = "completed"
        save_json(root / "results.json", result)
    finally:
        writer.close()
    print(f"RESULTS {root / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

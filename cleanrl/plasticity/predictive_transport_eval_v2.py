"""Covariance-free predictive momentum transport on the paired dense stream.

Every family runs its complete joint learning-rate/beta1 grid. Clean validation
locks each choice before a global held-out test barrier; failed candidates remain
visible and cannot be rescued or replaced using held-out results. The primary
implicit arm applies a per-example rank-one curvature action without covariance
memory. Robust uses pseudo-Huber rather than the squared-error training objective.
After all held-out scoring, fixed-selected replay timing measures performance only.
"""

import hashlib
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import predictive_transport_stream_v2 as transport
from cleanrl.plasticity.covariance_sketch_eval_v3 import (
    draw_data,
    phase_at,
    require_finite,
    save_json,
    score_selected,
    tensor_hash,
    validation_selection,
)
from cleanrl.shared import runtime


METHODS = ("adam", "predictive", "tangent", "implicit", "robust")


@dataclass
class Args:
    seed: int = 1
    samples: int = 65536
    hidden: int = 64
    input_dim: int = 17
    noise: float = 1.0
    hetero: float = 1.0
    switch_at: float = 0.5
    methods: tuple[str, ...] = METHODS
    adam_lrs: tuple[float, ...] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    betas: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999)
    validation: int = 2048
    test: int = 8192
    log_every: int = 4096
    graph_steps: int = 16
    output: str = ""


def validate_args(a):
    if min(a.samples, a.hidden, a.input_dim, a.validation, a.test, a.log_every) <= 0:
        raise ValueError("sample counts, dimensions and cadences must be positive")
    if a.graph_steps != 16:
        raise ValueError("graph_steps must be 16 for this protocol")
    if any(not math.isfinite(v) for v in (a.noise, a.hetero, a.switch_at)):
        raise ValueError("noise, hetero and switch_at must be finite")
    if a.noise < 0 or a.hetero < 0 or not 0 <= a.switch_at < 1:
        raise ValueError("invalid noise, heterogeneity or switch fraction")
    switch = int(a.samples * a.switch_at) if a.switch_at else a.samples
    if switch <= 0 or any(n % a.graph_steps for n in (a.samples, a.log_every, switch)):
        raise ValueError("samples, log_every and nonzero teacher switch must align with graph_steps")
    if not a.methods or len(set(a.methods)) != len(a.methods) or set(a.methods) - set(METHODS):
        raise ValueError("methods must be nonempty, unique and covariance-free")
    if not {"adam", "implicit"} <= set(a.methods):
        raise ValueError("every experiment requires adam control and primary implicit")
    if not a.adam_lrs or any(not math.isfinite(v) or v <= 0 for v in a.adam_lrs):
        raise ValueError("learning rates must be finite and positive")
    if not a.betas or any(not math.isfinite(v) or not 0 <= v < 1 for v in a.betas):
        raise ValueError("beta1 grid must be finite and in [0, 1)")
    if list(a.adam_lrs) != sorted(set(a.adam_lrs)) or list(a.betas) != sorted(set(a.betas)):
        raise ValueError("learning-rate and beta1 grids must be strictly increasing")


def arm_configs(a):
    grid = [{"learning_rate": lr, "beta1": beta} for lr in a.adam_lrs for beta in a.betas]
    return [{"name": method, "method": method, "grid": grid} for method in a.methods]


def joint_selection(curves, grid, samples):
    """Select a complete-stream finite candidate; preserve joint-index identity.

    State failures are historical, not endpoint-only. Validation nonfiniteness is
    independently disqualifying even when a caller's state mask says it is valid.
    No training-risk threshold or held-out value participates in eligibility.
    """
    if not grid:
        raise ValueError("selection requires a nonempty joint grid")
    reasons = [None] * len(grid)
    previous = 0
    for curve in curves:
        step = curve["step"]
        values, valid = curve["validation"], curve["candidate_valid"]
        if not previous < step <= samples or len(values) != len(grid) or len(valid) != len(grid):
            raise ValueError("validation checkpoints must be increasing and match the joint grid")
        failures = curve.get("candidate_failures", [None] * len(grid))
        if len(failures) != len(grid):
            raise ValueError("failure records must match the joint grid")
        for index, (value, state_valid) in enumerate(zip(values, valid)):
            if reasons[index] is None:
                if not state_valid:
                    reasons[index] = failures[index] or {"step": step, "reason": "nonfinite optimizer state"}
                elif value is None or not math.isfinite(value):
                    reasons[index] = {"step": step, "reason": "nonfinite clean validation"}
        previous = step
    if previous != samples:
        raise ValueError("selection requires the complete stream")
    eligible = [i for i, reason in enumerate(reasons) if reason is None]
    scores = [None] * len(grid)
    if eligible:
        compact = [{"step": c["step"], "validation": [c["validation"][i] for i in eligible]} for c in curves]
        decision = validation_selection(compact, eligible, samples)
        best = decision["chosen"]
        for index, score in zip(eligible, decision["validation_sustained_grid"]):
            scores[index] = score
        chosen = dict(grid[best])
        lrs = sorted({g["learning_rate"] for g in grid})
        betas = sorted({g["beta1"] for g in grid})
        edges = {"learning_rate": {"lower": chosen["learning_rate"] == lrs[0],
                                    "upper": chosen["learning_rate"] == lrs[-1]},
                 "beta1": {"lower": chosen["beta1"] == betas[0], "upper": chosen["beta1"] == betas[-1]}}
    else:
        best = chosen = edges = None
    return {"criterion": "duration_weighted_sustained_clean_validation", "samples": samples,
            "grid": grid, "validation_sustained_grid": scores,
            "candidate_valid": [reason is None for reason in reasons], "candidate_failures": reasons,
            "chosen_index": best, "chosen": chosen, "edge_flags": edges,
            "status": "selected" if eligible else "failed_all_candidates"}


def candidate_state(learner, count):
    """Reduce independently over candidate axes; shared counters cannot be masked."""
    per_candidate, shared = [], []
    for tensor in learner.mutable:
        if tensor.ndim == 0:
            shared.append(torch.isfinite(tensor))
        elif tensor.shape[0] == count:
            per_candidate.append(torch.isfinite(tensor).reshape(count, -1).all(-1))
        else:
            raise RuntimeError("mutable tensor lacks a candidate axis or scalar counter shape")
    if shared and not bool(torch.stack(shared).all().item()):
        raise FloatingPointError("nonfinite shared learner scalar counter")
    mask = torch.stack(per_candidate).all(0)
    # A state can become finite again through overwrites; transient failures are permanent.
    mask &= learner.finite_candidates
    return mask.cpu().tolist()


def phase_means(curves, value):
    totals, durations = {}, {}
    previous = 0
    for curve in curves:
        duration = curve["step"] - previous
        phase = str(curve["teacher_phase"])
        totals[phase] = totals.get(phase, 0.0) + value(curve) * duration
        durations[phase] = durations.get(phase, 0) + duration
        previous = curve["step"]
    return {phase: {"samples": durations[phase], "mse": total / durations[phase]}
            for phase, total in totals.items()}


@torch.no_grad()
def train_arm(a, config, data, root, writer, result):
    name, grid = config["name"], config["grid"]
    count = len(grid)
    row = result["methods"][name]
    checkpoint_dir = root / "checkpoints" / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.compiler.reset()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    learner = graph = None
    try:
        learner = transport.Learner(name, [(g["learning_rate"], g["beta1"]) for g in grid],
                                    data["initial"], a, data["xs"], data["ys"], data["clean"], data["noise_var"])
        if learner.capture_steps != a.graph_steps:
            raise RuntimeError("learner capture cadence differs from the protocol")
        row["capture_steps"] = learner.capture_steps
        row["resources"].update(
            forward_evaluations_per_step=learner.forward_evaluations_per_step,
            jacobian_evaluations_per_step=learner.jacobian_evaluations_per_step,
        )
        row["resources"]["mutable_tensor_bytes"] = sum(t.numel() * t.element_size() for t in learner.mutable)
        row["status"] = "capturing"
        save_json(root / "results.json", result)
        graph, parity = learner.capture()
        torch.cuda.synchronize()
        row["timing"]["startup_seconds"] = time.perf_counter() - started
        row["resources"]["startup_candidate_forward_evaluations"] = (
            count * learner.capture_update_calls * learner.forward_evaluations_per_step)
        row["resources"]["startup_candidate_jacobian_evaluations"] = (
            count * learner.capture_update_calls * learner.jacobian_evaluations_per_step)
        row["resources"]["startup_compute_scope"] = "explicit eager parity audit, compiled warmup, capture and replay; excludes compiler/autotuner internals"
        row["graph_parity_max_abs"] = parity
        row["capture_failed_candidates"] = learner.capture_failed_candidates
        require_finite(parity, "capture parity")
        if int(learner.index.item()) != 0 or float(learner.steps.item()) != 0:
            raise RuntimeError("capture did not restore the stream position")
        row["status"] = "training"
        started = time.perf_counter()
        previous = 0
        previous_error = torch.zeros(count, device=data["xs"].device)
        previous_zero = 0.0
        valid = [not failed for failed in learner.capture_failed_candidates]
        failures = [None if eligible else {"step": 0, "reason": "nonfinite optimizer state during capture audit"}
                    for eligible in valid]
        for step in range(learner.capture_steps, a.samples + 1, learner.capture_steps):
            graph.replay()
            if step % a.log_every and step != a.samples and step != data["switch"]:
                continue
            phase = phase_at(step, data["switch"])
            validation = reference.evaluate(learner.weights, data["xv"], data["yv"][phase]).cpu().tolist()
            state_valid = candidate_state(learner, count)
            for index in range(count):
                reason = ("nonfinite optimizer state (including historical failures)" if not state_valid[index]
                          else "nonfinite clean validation" if not math.isfinite(validation[index]) else None)
                if valid[index] and reason is not None:
                    valid[index] = False
                    failures[index] = {"step": step, "reason": reason}
            duration = step - previous
            error = learner.error.detach().clone()
            zero = float(learner.null_error)
            curve = {"step": step, "interval_samples": duration, "teacher_phase": phase,
                     "validation": [v if math.isfinite(v) else None for v in validation],
                     "candidate_state_finite": state_valid, "candidate_valid": list(valid),
                     "candidate_failures": list(failures),
                     "online_clean_mse": reference.finite_json((error / step).cpu().tolist()),
                     "interval_online_clean_mse": reference.finite_json(((error - previous_error) / duration).cpu().tolist()),
                     "online_zero_mse": zero / step, "interval_online_zero_mse": (zero - previous_zero) / duration,
                     "diagnostics": learner.diagnostics()}
            row["curves"].append(curve)
            row["resources"].update(consumed_samples=int(learner.index.item()), learner_steps=float(learner.steps.item()),
                                    candidate_samples=count * step,
                                    candidate_forward_evaluations=count * step * learner.forward_evaluations_per_step,
                                    candidate_jacobian_evaluations=count * step * learner.jacobian_evaluations_per_step,
                                    candidate_validation_forward_evaluations=count * a.validation * len(row["curves"]))
            if row["resources"]["consumed_samples"] != step or row["resources"]["learner_steps"] != step:
                raise RuntimeError("CUDA replay consumed a different number of samples than reported")
            checkpoint = checkpoint_dir / f"{step}.pt"
            torch.save({"step": step, "weights": [w.detach().cpu() for w in learner.weights],
                        "grid": grid, "candidate_valid": list(valid), "candidate_failures": list(failures)}, checkpoint)
            curve["checkpoint"] = str(checkpoint.relative_to(root))
            previous, previous_error, previous_zero = step, error, zero
            for index, candidate in enumerate(grid):
                tag = f"{name}/lr_{candidate['learning_rate']:g}/beta1_{candidate['beta1']:g}"
                writer.add_scalar(f"candidate_valid/{tag}", int(valid[index]), step)
                for key in ("validation", "online_clean_mse", "interval_online_clean_mse"):
                    if curve[key][index] is not None:
                        writer.add_scalar(f"{key}/{tag}", curve[key][index], step)
            row["timing"]["training_seconds"] = time.perf_counter() - started
            save_json(root / "results.json", result)
            writer.flush()
            print(json.dumps({"arm": name, "step": step, "validation": curve["validation"],
                              "candidate_valid": valid, "candidate_failures": failures}, allow_nan=False), flush=True)
        torch.cuda.synchronize()
        row["timing"]["training_seconds"] = time.perf_counter() - started
        row["timing"]["training_scope"] = "full joint-grid graph replay, clean validation, diagnostics, CPU transfers, checkpoints, TensorBoard, JSON and progress reporting; excludes startup and held-out scoring"
        row["timing"]["aggregate_candidate_samples_per_second"] = count * a.samples / row["timing"]["training_seconds"]
        row["selection"] = joint_selection(row["curves"], grid, a.samples)
        row["failed_candidates"] = [{"index": i, **grid[i], **reason}
                                    for i, reason in enumerate(row["selection"]["candidate_failures"]) if reason is not None]
        row["status"] = row["selection"]["status"]
        selection_path = root / f"selection_{name}.json"
        with selection_path.open("x") as file:
            file.write(json.dumps(row["selection"], indent=2, allow_nan=False) + "\n")
        row["selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        if row["status"] != "selected":
            row["failure"] = {"type": "AllCandidatesFailed", "message": "all joint-grid candidates had nonfinite validation or state"}
            return
        best = row["selection"]["chosen_index"]
        row["selected_learning_rate"] = grid[best]["learning_rate"]
        row["selected_beta1"] = grid[best]["beta1"]
        row["edge_flags"] = row["selection"]["edge_flags"]
        row["validation_by_phase"] = phase_means(row["curves"], lambda c: c["validation"][best])
        row["online_clean_by_phase"] = phase_means(row["curves"], lambda c: c["interval_online_clean_mse"][best])
    finally:
        row["resources"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        row["resources"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        del graph, learner


def verify_sources(project, expected):
    for relative, digest in expected.items():
        if hashlib.sha256((project / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"implementation source changed during experiment: {relative}")


def verify_replay_data(data):
    """Hash the actual retained learning inputs, without drawing a second task."""
    inputs = {"xs": data["xs"], "ys": data["ys"], "clean": data["clean"],
              "noise_variance": data["noise_var"],
              **{f"initial_{i}": value for i, value in enumerate(data["initial"])}}
    hashes = {name: tensor_hash(value) for name, value in inputs.items()}
    if any(digest != data["hashes"][name] for name, digest in hashes.items()):
        raise RuntimeError("retained replay inputs differ from the original paired task")
    return hashes


@torch.no_grad()
def time_selected_replay(a, name, data, root, row, project, source_hashes):
    """Benchmark one locked candidate only after every arm's test scoring ends.

    Capture audits and compiles the single-candidate shape outside timing. The
    measured interval contains only full-stream graph replay, including the
    learner's usual online telemetry and any host launch gaps; no validation,
    checkpoint transfer, reporting, or held-out evaluation occurs in it.
    """
    path = root / f"selection_{name}.json"
    selection_bytes = path.read_bytes()
    if hashlib.sha256(selection_bytes).hexdigest() != row["selection_sha256"]:
        raise RuntimeError("persisted selection changed before performance replay")
    selection = json.loads(selection_bytes)
    chosen = selection["chosen"]
    if selection["samples"] != a.samples or chosen != row["selection"]["chosen"]:
        raise RuntimeError("performance replay differs from the locked candidate or stream length")
    verify_sources(project, source_hashes)
    input_hashes = verify_replay_data(data)
    benchmark = row["fixed_selected_replay"] = {
        "status": "capturing", "candidate_count": 1,
        "chosen": chosen, "chosen_index": selection["chosen_index"],
        "selection_sha256": row["selection_sha256"], "data_hashes": input_hashes,
        "source_sha256": dict(source_hashes),
        "selection_policy": "performance only; choice is immutable and no score participates in reselection",
        "scope": "CUDA-event elapsed full-stream graph replay from initial state, K=1; includes optimizer online telemetry and host graph-submission gaps; excludes construction, compilation, capture/audit, input hashing, validation, test scoring, checkpoints and report IO",
        "comparison_limit": "single-candidate compiled shape is benchmarked independently; not the joint-grid training/report throughput or a second accuracy estimate",
    }
    torch.compiler.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    learner = graph = None
    started = time.perf_counter()
    try:
        learner = transport.Learner(name, [(chosen["learning_rate"], chosen["beta1"])],
                                    data["initial"], a, data["xs"], data["ys"], data["clean"], data["noise_var"])
        if learner.capture_steps != a.graph_steps:
            raise RuntimeError("performance replay capture cadence differs from protocol")
        graph, parity = learner.capture()
        torch.cuda.synchronize()
        benchmark["startup_seconds"] = time.perf_counter() - started
        benchmark["graph_parity_max_abs"] = parity
        benchmark["capture_failed_candidates"] = learner.capture_failed_candidates
        benchmark["startup_candidate_forward_evaluations"] = learner.capture_update_calls * learner.forward_evaluations_per_step
        benchmark["startup_candidate_jacobian_evaluations"] = learner.capture_update_calls * learner.jacobian_evaluations_per_step
        benchmark["startup_compute_scope"] = "explicit eager parity audit, compiled warmup, capture and replay; excludes compiler/autotuner internals"
        require_finite(parity, "fixed-selected capture parity")
        if any(learner.capture_failed_candidates) or not all(candidate_state(learner, 1)):
            raise FloatingPointError("fixed-selected replay capture produced nonfinite state")
        if int(learner.index.item()) != 0 or float(learner.steps.item()) != 0:
            raise RuntimeError("performance capture did not restore initial stream position")
        if any(not torch.equal(weight, initial.unsqueeze(0)) for weight, initial in zip(learner.weights, data["initial"])):
            raise RuntimeError("performance capture did not restore the paired initial weights")
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        begin.record()
        for _ in range(0, a.samples, learner.capture_steps):
            graph.replay()
        end.record()
        end.synchronize()
        seconds = begin.elapsed_time(end) / 1000.0
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError("invalid CUDA event duration for full-stream replay")
        consumed, steps = int(learner.index.item()), float(learner.steps.item())
        if consumed != a.samples or steps != a.samples:
            raise RuntimeError("performance replay consumed a different number of samples than reported")
        if not all(candidate_state(learner, 1)):
            raise FloatingPointError("fixed-selected full-stream replay produced nonfinite state")
        benchmark.update(
            status="completed", samples=a.samples, consumed_samples=consumed, learner_steps=steps,
            capture_steps=learner.capture_steps, graph_replays=a.samples // learner.capture_steps,
            replay_seconds=seconds, samples_per_second=a.samples / seconds,
            forward_evaluations_per_step=learner.forward_evaluations_per_step,
            jacobian_evaluations_per_step=learner.jacobian_evaluations_per_step,
            candidate_forward_evaluations=a.samples * learner.forward_evaluations_per_step,
            candidate_jacobian_evaluations=a.samples * learner.jacobian_evaluations_per_step,
            mutable_tensor_bytes=sum(t.numel() * t.element_size() for t in learner.mutable),
        )
        if verify_replay_data(data) != input_hashes:
            raise RuntimeError("performance replay mutated the retained paired input data")
        verify_sources(project, source_hashes)
        if path.read_bytes() != selection_bytes:
            raise RuntimeError("selection lock changed during performance replay")
        benchmark["source_and_data_equality_verified"] = True
    finally:
        benchmark["benchmark_wall_seconds"] = time.perf_counter() - started
        benchmark["peak_allocated_bytes_including_capture"] = torch.cuda.max_memory_allocated()
        benchmark["peak_reserved_bytes_including_capture"] = torch.cuda.max_memory_reserved()
        del graph, learner


def main():
    a = tyro.cli(Args)
    validate_args(a)
    root = Path(a.output) if a.output else Path("runs") / f"DenseStream__predictive_transport_v2__{a.seed}__{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "results.json").exists() or any(root.glob("selection_*.json")):
        raise FileExistsError(f"refusing to overwrite an existing experiment at {root}")
    configs = arm_configs(a)
    # no_grad wrappers live in torch contextlib, not in the actual helper source.
    helpers = (draw_data, phase_at, require_finite, save_json, score_selected, tensor_hash, validation_selection)
    source_paths = sorted({Path(__file__).resolve(), Path(transport.__file__).resolve(),
                           Path(reference.__file__).resolve(), Path(runtime.__file__).resolve(),
                           *(Path(inspect.unwrap(helper).__code__.co_filename).resolve() for helper in helpers)})
    project = Path(__file__).resolve().parents[2]
    result = {
        "args": asdict(a), "run_dir": str(root), "status": "initializing", "primary_method": "implicit",
        "protocol": "single paired task seed; full streams; independent joint LR/beta1 clean-validation selection; exclusive locks before global test barrier; no test-based rescue; no cross-seed inference",
        "information_fairness": {
            "optimizer_inputs": "same current x and noisy y; ordinary parameter/momentum history only",
            "clean_labels": "metrics and validation selection only, never optimizer updates",
            "noise_variance": "shared data-generation artifact, never optimizer input",
            "predictive": "C=(f-f_previous)*J_current; independent of the current label; only previous prediction is evaluated",
            "tangent": "C=<J_current,theta-theta_previous>*J_current with a global all-layer inner product; no previous forward or Jacobian",
            "implicit": "same predictive m/v; frozen-current-J proximal quadratic step delta=u-lr*D*J*<J,u>/(1+lr*<J,D*J>), u=-lr*D*mhat; per-example rank-one curvature action, not covariance memory",
            "robust": "pseudo-Huber loss sqrt(1+e^2)-1, scale 1; psi=e/sqrt(1+e^2), g=psi(e)*J, C=[psi(f-y)-psi(f_previous-y)]*J; label-dependent correction; objective differs from squared error and can have a different finite-capacity optimum even though selection/reporting use clean MSE",
            "state": "weights, owned previous pre-update weights, Adam m/v and linear/scalar telemetry; no covariance, sketches, Fisher or outer-product state",
            "adam": "beta2=.999, epsilon=1e-8; raw observed gradient squared updates v for every family, including the raw pseudo-Huber gradient in robust",
            "transport": "m=beta1*m+(1-beta1)*g+beta1*(1-beta1**(t-1))*C with stable expm1 moment masses; previous weights retain an owned snapshot across separately captured commits",
            "compute": "per-candidate forward and exact output-Jacobian evaluations counted separately: one current forward and Jacobian for every method; predictive/implicit/robust add only a previous forward; validation/test are forward-only examples; not compute-matched",
            "claims": "single paired seed; no calibrated uncertainty, global-optimality, cross-seed significance or objective-equivalence claim for robust",
        },
        "source_sha256": {str(p.resolve().relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "data_hashes": {}, "timing": {}, "resources": {},
        "methods": {c["name"]: {**c, "status": "pending", "curves": [], "selection": None,
                                  "failed_candidates": [], "timing": {}, "resources": {"candidate_count": len(c["grid"])}}
                    for c in configs},
    }
    writer = SummaryWriter(str(root))
    writer.add_text("hyperparameters", json.dumps(asdict(a), indent=2))
    writer.add_text("protocol", result["protocol"])
    writer.add_text("information_fairness", json.dumps(result["information_fairness"], indent=2))
    writer.flush()
    save_json(root / "results.json", result)
    started = time.perf_counter()
    failed = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU or eager fallback")
        result["resources"] = {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                               "cuda": torch.version.cuda, "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                               "precision": "unchanged reference task generation; all CUDA model calculations and learning tensors FP32 in every arm; TF32 disabled",
                               "execution": "compiled CUDA graphs; no fallback; graph_steps=16",
                               "memory_values_are_measured": True}
        generated = time.perf_counter()
        data = draw_data(a)
        torch.cuda.synchronize()
        result["timing"]["generation_and_hashing_seconds"] = time.perf_counter() - generated
        result["data_hashes"] = data["hashes"]
        result["resources"]["parameters"] = sum(w.numel() for w in data["initial"])
        result["status"] = "training_and_selecting"
        save_json(root / "results.json", result)
        for config in configs:
            row = result["methods"][config["name"]]
            try:
                train_arm(a, config, data, root, writer, result)
                failed |= row["status"] != "selected"
            except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
                row.update(status="unavailable" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed_nonfinite",
                           failure={"type": type(error).__name__, "message": str(error)})
                failed = True
            except Exception as error:
                row.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                save_json(root / "results.json", result)
                writer.flush()
            torch.cuda.empty_cache()
        # No held-out risk, including a zero predictor, has been computed above.
        result["status"] = "testing_locked_selections"
        result["selection_barrier"] = {name: {"status": row["status"], "sha256": row.get("selection_sha256")}
                                       for name, row in result["methods"].items()}
        # Freeze the complete family manifest before the first held-out score.
        barrier_path = root / "selection_barrier.json"
        with barrier_path.open("x") as file:
            file.write(json.dumps(result["selection_barrier"], indent=2, allow_nan=False) + "\n")
        barrier_bytes = barrier_path.read_bytes()
        result["selection_barrier_sha256"] = hashlib.sha256(barrier_bytes).hexdigest()
        for name, locked in result["selection_barrier"].items():
            if locked["sha256"] is not None:
                if hashlib.sha256((root / f"selection_{name}.json").read_bytes()).hexdigest() != locked["sha256"]:
                    raise RuntimeError("selection changed before the global test barrier")
        verify_sources(project, result["source_sha256"])
        save_json(root / "results.json", result)
        for name, row in result["methods"].items():
            if row["status"] != "selected":
                continue
            scored = time.perf_counter()
            try:
                path = root / f"selection_{name}.json"
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["selection_sha256"]:
                    raise RuntimeError("persisted selection changed before test scoring")
                row.update(score_selected(path, row["curves"], root, data["xt"], data["test_targets"], data["switch"]))
                torch.cuda.synchronize()
                row["test_by_phase"] = phase_means(row["test_curve"], lambda c: c["mse"])
                row["resources"]["selected_test_forward_evaluations"] = a.test * len(row["test_curve"])
                row["status"] = "completed"
                for point in row["test_curve"]:
                    writer.add_scalar(f"test/{name}/clean_mse", point["mse"], point["step"])
                for parameter, edges in row["edge_flags"].items():
                    for side, value in edges.items():
                        writer.add_scalar(f"selection/{name}/{parameter}_{side}_edge", int(value), a.samples)
            except FloatingPointError as error:
                # Keep the locked candidate and its eligibility intact. Test cannot rerank.
                row.update(status="failed_test_nonfinite", failure={"type": type(error).__name__, "message": str(error)})
                failed = True
            except Exception as error:
                row.update(status="failed_test", failure={"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                row["timing"]["test_seconds"] = time.perf_counter() - scored
                save_json(root / "results.json", result)
                writer.flush()
        # All held-out reporting is complete before any performance-only rerun.
        result["timing"]["generation_training_selection_test_seconds"] = time.perf_counter() - started
        result["status"] = "timing_fixed_selected_replays"
        save_json(root / "results.json", result)
        for name, row in result["methods"].items():
            if row["status"] not in ("completed", "failed_test_nonfinite"):
                continue
            try:
                time_selected_replay(a, name, data, root, row, project, result["source_sha256"])
                replay = row["fixed_selected_replay"]
                writer.add_scalar(f"performance/{name}/fixed_selected_samples_per_second", replay["samples_per_second"], a.samples)
            except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
                row.setdefault("fixed_selected_replay", {}).update(
                    status="failed", failure={"type": type(error).__name__, "message": str(error)})
                failed = True
            except Exception as error:
                row.setdefault("fixed_selected_replay", {}).update(
                    status="failed", failure={"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                save_json(root / "results.json", result)
                writer.flush()
            torch.cuda.empty_cache()
        if barrier_path.read_bytes() != barrier_bytes:
            raise RuntimeError("global selection barrier changed during held-out scoring or performance replay")
        for name, locked in result["selection_barrier"].items():
            if locked["sha256"] is not None:
                if hashlib.sha256((root / f"selection_{name}.json").read_bytes()).hexdigest() != locked["sha256"]:
                    raise RuntimeError("selection changed after the global test barrier")
        verify_sources(project, result["source_sha256"])
        result["status"] = "failed" if failed else "completed"
    except BaseException as error:
        result["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["timing"]["total_seconds"] = time.perf_counter() - started
        save_json(root / "results.json", result)
        writer.add_text("status", result["status"])
        writer.close()
        print(f"RESULTS {root / 'results.json'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

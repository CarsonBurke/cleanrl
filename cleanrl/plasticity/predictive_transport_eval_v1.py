"""Covariance-free predictive momentum transport on the paired dense stream.

Every family runs its complete joint learning-rate/beta1 grid. Clean validation
locks each choice before a global held-out test barrier; failed candidates remain
visible and cannot be rescued or replaced using held-out results.
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
from cleanrl.plasticity import predictive_transport_stream_v1 as transport
from cleanrl.plasticity.covariance_sketch_eval_v3 import (
    draw_data,
    phase_at,
    require_finite,
    save_json,
    score_selected,
    validation_selection,
)
from cleanrl.shared import runtime


METHODS = ("adam", "full", "predictive", "uncertainty", "shared", "history")


@dataclass
class Args(reference.Args):
    hetero: float = 1.0
    switch_at: float = 0.5
    methods: tuple[str, ...] = METHODS
    betas: tuple[float, ...] = (0.9, 0.99, 0.999)
    noise_rate: float = 0.001
    """Past-only per-row stability EMA update rate for the history ablation."""
    graph_steps: int = 16
    # Inherited only for the reference Args shape, never optimizer inputs or CLI choices.
    prior_scales: tyro.conf.Suppress[tuple[float, ...]] = ()
    diffusion: tyro.conf.Suppress[float] = 0.0
    known_noise: tyro.conf.Suppress[bool] = False


def validate_args(a):
    if min(a.samples, a.hidden, a.input_dim, a.validation, a.test, a.log_every) <= 0:
        raise ValueError("sample counts, dimensions and cadences must be positive")
    if a.graph_steps != 16:
        raise ValueError("graph_steps must be 16 for this protocol")
    if any(not math.isfinite(v) for v in (a.noise, a.hetero, a.switch_at, a.noise_rate)):
        raise ValueError("noise, hetero, switch_at and noise_rate must be finite")
    if a.noise < 0 or a.hetero < 0 or not 0 <= a.switch_at < 1 or not 0 < a.noise_rate <= 1:
        raise ValueError("invalid noise, heterogeneity, switch fraction or history rate")
    switch = int(a.samples * a.switch_at) if a.switch_at else a.samples
    if switch <= 0 or any(n % a.graph_steps for n in (a.samples, a.log_every, switch)):
        raise ValueError("samples, log_every and nonzero teacher switch must align with graph_steps")
    if not a.methods or len(set(a.methods)) != len(a.methods) or set(a.methods) - set(METHODS):
        raise ValueError("methods must be nonempty, unique and covariance-free")
    if not {"adam", "uncertainty"} <= set(a.methods):
        raise ValueError("every experiment requires adam control and primary uncertainty")
    if not a.adam_lrs or any(not math.isfinite(v) or v <= 0 for v in a.adam_lrs):
        raise ValueError("learning rates must be finite and positive")
    if not a.betas or any(not math.isfinite(v) or not 0 <= v < 1 for v in a.betas):
        raise ValueError("beta1 grid must be finite and in [0, 1)")
    if list(a.adam_lrs) != sorted(set(a.adam_lrs)) or list(a.betas) != sorted(set(a.betas)):
        raise ValueError("learning-rate and beta1 grids must be strictly increasing")
    if a.prior_scales or a.diffusion != 0 or a.known_noise:
        raise ValueError("covariance and privileged-noise configuration is not supported")


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
        row["resources"]["mutable_tensor_bytes"] = sum(t.numel() * t.element_size() for t in learner.mutable)
        row["status"] = "capturing"
        save_json(root / "results.json", result)
        graph, parity = learner.capture()
        torch.cuda.synchronize()
        row["timing"]["startup_seconds"] = time.perf_counter() - started
        row["resources"]["startup_candidate_gradient_forward_evaluations"] = (
            count * learner.capture_update_calls * (1 if name == "adam" else 2))
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
                                    candidate_gradient_forward_evaluations=count * step * (1 if name == "adam" else 2),
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


def main():
    a = tyro.cli(Args)
    validate_args(a)
    root = Path(a.output) if a.output else Path("runs") / f"DenseStream__predictive_transport_v1__{a.seed}__{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "results.json").exists() or any(root.glob("selection_*.json")):
        raise FileExistsError(f"refusing to overwrite an existing experiment at {root}")
    configs = arm_configs(a)
    source_paths = [Path(__file__), Path(transport.__file__), Path(reference.__file__), Path(runtime.__file__),
                    Path(draw_data.__code__.co_filename)]
    project = Path(__file__).resolve().parents[2]
    result = {
        "args": asdict(a), "run_dir": str(root), "status": "initializing", "primary_method": "uncertainty",
        "protocol": "single paired task seed; full streams; independent joint LR/beta1 clean-validation selection; exclusive locks before global test barrier; no test-based rescue; no cross-seed inference",
        "information_fairness": {
            "optimizer_inputs": "same current x and noisy y; ordinary parameter/momentum history only",
            "clean_labels": "metrics and validation selection only, never optimizer updates",
            "noise_variance": "shared data-generation artifact, never optimizer input",
            "stability_proxy": "deterministic per-row Jacobian stability conditioned on current x; not calibrated aleatoric or posterior uncertainty",
            "full": "same current noisy label in current and previous gradients",
            "predictive": "prediction-change transport independent of current label; q=1",
            "shared": "current q averaged within each layer, then applied equally to layer rows",
            "history": "past-only per-row q EMA initialized to one, updated after use",
            "state": "weights, previous pre-update weights, Adam m/v, row/scalar telemetry; no covariance, sketches, Fisher or outer-product state",
            "adam": "beta2=.999, epsilon=1e-8; raw observed g squared updates v for every family",
            "transport": "C added to biased m with beta1*(1-beta1**(t-1)); previous weights retain an owned pre-update snapshot across compiled mutations",
            "uncertainty": "C=(f-f_previous)*J_current times row q=||J_current||^2/(||J_current||^2+||J_current-J_previous||^2), both norms zero gives q=1",
            "compute": "training counts sample_state forward+Jacobian evaluations per candidate; validation/test counts forward examples separately; extra previous-weight evaluation is not equal compute",
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
                               "precision": "reference task generation; CUDA FP32 weights/moments and FP64 forward/Jacobian teaching signals in every arm; TF32 disabled",
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

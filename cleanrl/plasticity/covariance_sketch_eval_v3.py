"""Paired dense-stream covariance-sketch experiment with locked validation selection.

Every grid consumes the complete shared noisy stream. All arms finish selection
before any held-out test risk is scored. The scalar control preserves instantaneous
linearized gain, not autonomous per-neuron gating or matched future trajectories.
Run on CUDA through mlq; compiled/captured execution has no eager fallback.
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import covariance_sketch_stream_v3 as sketch
from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.shared import runtime


@dataclass
class Args(reference.Args):
    buffer: int = 64
    ranks: tuple[int, ...] = (16, 64, 256)
    scalar_rank: int = 64
    methods: tuple[str, ...] = ("adam", "unit", "network", "diagonal", "sketch", "sketch_scalar")
    graph_steps: int = 16


def validate_cadence(a, capture_steps):
    if capture_steps <= 0:
        raise ValueError("capture_steps must be positive")
    switch = int(a.samples * a.switch_at) if a.switch_at else a.samples
    if a.samples % capture_steps or a.log_every % capture_steps or switch % capture_steps:
        raise ValueError("samples, log_every and teacher switch must be divisible by every arm's capture_steps")
    if a.switch_at and switch == 0:
        raise ValueError("a nonzero teacher switch must follow at least one sample")


def validate_args(a):
    # Reuse reference numeric/grid checks without pretending its method registry
    # contains the new covariance representations.
    reference.validate_args(replace(a, methods=("network",)))
    if not a.methods or len(set(a.methods)) != len(a.methods):
        raise ValueError("methods must be nonempty and unique")
    if set(a.methods) - {"adam", "unit", "network", "diagonal", "sketch", "sketch_scalar"}:
        raise ValueError("unknown method")
    if "network" not in a.methods:
        raise ValueError("every experiment must include the full network reference")
    if a.buffer <= 0:
        raise ValueError("buffer must be positive")
    if not a.ranks or len(set(a.ranks)) != len(a.ranks) or any(r <= 0 for r in a.ranks):
        raise ValueError("ranks must be nonempty, unique and positive")
    if a.scalar_rank <= 0:
        raise ValueError("scalar_rank must be positive")
    if any(not math.isfinite(v) for v in (a.noise, a.hetero, a.diffusion, a.noise_rate, a.switch_at)):
        raise ValueError("noise, hetero, diffusion, noise_rate and switch_at must be finite")
    validate_cadence(a, a.graph_steps)
    validate_cadence(a, a.buffer)


def arm_configs(a):
    configs = []
    for method in a.methods:
        ranks = a.ranks if method == "sketch" else ((a.scalar_rank,) if method == "sketch_scalar" else (0,))
        for rank in ranks:
            configs.append({
                "name": f"{method}_r{rank}" if rank else method,
                "method": method,
                "rank": rank,
                "grid": list(a.adam_lrs if method == "adam" else a.prior_scales),
                "parameter": "learning_rate" if method == "adam" else "prior_scale",
            })
    return configs


def phase_at(step, switch):
    """A checkpoint at the boundary has consumed only the first teacher's labels."""
    return int(step > switch)


def validation_selection(curves, grid, samples):
    """Right-endpoint clean validation, weighted by actual interval duration.

    A nonfinite candidate fails the arm, rather than disappearing from ranking.
    Selection cannot use online or held-out test errors: neither is an input.
    """
    scores = [0.0] * len(grid)
    previous = 0
    for curve in curves:
        step, values = curve["step"], curve["validation"]
        if not previous < step <= samples or len(values) != len(grid):
            raise ValueError("validation checkpoints must be increasing and match the grid")
        if any(not math.isfinite(value) for value in values):
            raise FloatingPointError("nonfinite validation risk; the complete arm failed")
        for index, value in enumerate(values):
            scores[index] += value * ((step - previous) / samples)
        previous = step
    if not grid or previous != samples:
        raise ValueError("selection requires the complete stream and a nonempty grid")
    if any(not math.isfinite(value) for value in scores):
        raise FloatingPointError("nonfinite sustained validation risk")
    best = min(range(len(grid)), key=scores.__getitem__)
    return {"criterion": "duration_weighted_sustained_clean_validation",
            "samples": samples, "grid": list(grid), "validation_sustained_grid": scores,
            "chosen_index": best, "chosen": grid[best], "edge": best in (0, len(grid) - 1)}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(reference.finite_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_finite(value, label):
    """Fail explicitly before JSON null conversion can hide a numerical failure."""
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"nonfinite {label}")


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def draw_data(a):
    # Preserve the frozen reference's generation order and runtime policy exactly.
    runtime.configure_runtime()
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(a.seed)
    t1, t2 = reference.draw_teacher(a, gen, device), reference.draw_teacher(a, gen, device)
    initial = reference.init_weights(a, gen, device)
    xs = torch.randn(a.samples, a.input_dim, generator=gen, device=device)
    direction = torch.randn(a.input_dim, generator=gen, device=device)
    direction /= direction.norm()
    sigma = a.noise * (a.hetero * torch.tanh(xs @ direction)).exp()
    clean = reference.teach(t1, xs)
    switch = int(a.samples * a.switch_at) if a.switch_at else a.samples
    if switch < a.samples:
        clean[switch:] = reference.teach(t2, xs[switch:])
    ys = clean + sigma * torch.randn(a.samples, generator=gen, device=device)
    xv = torch.randn(a.validation, a.input_dim, generator=gen, device=device)
    xt = torch.randn(a.test, a.input_dim, generator=gen, device=device)
    yv = [reference.teach(t1, xv), reference.teach(t2, xv)]
    test_targets = [reference.teach(t1, xt), reference.teach(t2, xt)]
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    hashes = {name: tensor_hash(value) for name, value in {
        "xs": xs, "ys": ys, "clean": clean, "noise_variance": sigma.square(),
        "noise_direction": direction, "validation_x": xv, "test_x": xt,
        **{f"initial_{i}": w for i, w in enumerate(initial)},
        **{f"teacher_{p}_{i}": w for p, teacher in enumerate((t1, t2)) for i, w in enumerate(teacher)},
        **{f"validation_y_{i}": y for i, y in enumerate(yv)},
        **{f"test_y_{i}": y for i, y in enumerate(test_targets)},
    }.items()}
    return {"initial": initial, "xs": xs, "ys": ys, "clean": clean,
            "noise_var": sigma.square(), "xv": xv, "yv": yv, "xt": xt,
            "test_targets": test_targets, "switch": switch, "hashes": hashes}


@torch.no_grad()
def train_arm(a, config, data, root, writer, result):
    name, method, grid = config["name"], config["method"], config["grid"]
    row = result["methods"][name]
    checkpoint_dir = root / "checkpoints" / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    learner = graph = None
    # Arms own disjoint compiled graphs. Release prior Dynamo guards instead of
    # exhausting the shared no_grad wrapper's specialization cache across ranks.
    torch.compiler.reset()
    try:
        arguments = (grid, data["initial"], a, data["xs"], data["ys"], data["clean"], data["noise_var"])
        learner = (reference.Learner(method, *arguments) if method in ("adam", "unit", "network")
                   else sketch.SketchLearner(method, config["rank"], *arguments))
        validate_cadence(a, learner.capture_steps)
        row["capture_steps"] = learner.capture_steps
        row["execution"] = ("compiled CUDA graph" if method in ("adam", "unit", "network") else
                            "compiled CUDA update graph; compiled eigensolver compression outside graph after each buffer; no eager fallback")
        row["status"] = "capturing"
        save_json(root / "results.json", result)
        graph, parity = learner.capture()
        if hasattr(graph, "compression_execution"):
            row["compression_execution"] = graph.compression_execution
        torch.cuda.synchronize()
        row["timing"]["startup_seconds"] = time.perf_counter() - started
        row["graph_parity_max_abs"] = parity
        require_finite(parity, "capture parity")
        if int(learner.index.item()) != 0 or float(learner.steps.item()) != 0:
            raise RuntimeError("capture did not restore the stream position")
        row["status"] = "training"
        started = time.perf_counter()
        previous = 0
        for step in range(learner.capture_steps, a.samples + 1, learner.capture_steps):
            graph.replay()
            if step % a.log_every and step != a.samples and step != data["switch"]:
                continue
            phase = phase_at(step, data["switch"])
            curve = {"step": step, "interval_samples": step - previous, "teacher_phase": phase,
                     "validation": reference.evaluate(learner.weights, data["xv"], data["yv"][phase]).cpu().tolist(),
                     "online_clean_mse": (learner.error / step).cpu().tolist(),
                     "online_zero_mse": float(learner.null_error / step),
                     "gain": (learner.gain_sum / step).cpu().tolist(),
                     "unit_gain_sd": (learner.unit_variance / step).sqrt().cpu().tolist(),
                     "noise_estimate": learner.noise.cpu().tolist()}
            row["curves"].append(curve)
            row["resources"]["consumed_samples"] = int(learner.index.item())
            row["resources"]["learner_steps"] = float(learner.steps.item())
            if row["resources"]["consumed_samples"] != step or row["resources"]["learner_steps"] != step:
                raise RuntimeError("CUDA replay consumed a different number of samples than reported")
            require_finite(curve, f"{name}.curve")
            # Check the actual posterior, not just its observed directions. Reduce
            # on device and synchronize once per checkpoint, never per sample.
            if not bool(torch.stack([torch.isfinite(t).all() for t in learner.mutable]).all().item()):
                raise FloatingPointError(f"nonfinite mutable learner state in {name} at sample {step}")
            checkpoint = checkpoint_dir / f"{step}.pt"
            torch.save({"step": step, "weights": [w.detach().cpu() for w in learner.weights]}, checkpoint)
            curve["checkpoint"] = str(checkpoint.relative_to(root))
            previous = step
            for index, value in enumerate(grid):
                tag = f"{name}/{value:g}"
                for key in ("validation", "online_clean_mse", "gain", "unit_gain_sd", "noise_estimate"):
                    writer.add_scalar(f"{key}/{tag}", curve[key][index], step)
            row["timing"]["training_seconds"] = time.perf_counter() - started
            save_json(root / "results.json", result)
            writer.flush()
            print(json.dumps({"arm": name, **curve}, allow_nan=False), flush=True)
        torch.cuda.synchronize()
        row["timing"]["training_seconds"] = time.perf_counter() - started
        row["timing"]["aggregate_samples_per_second"] = a.samples * len(grid) / row["timing"]["training_seconds"]
        row["resources"]["mutable_tensor_bytes"] = sum(t.numel() * t.element_size() for t in learner.mutable)
        row["selection"] = validation_selection(row["curves"], grid, a.samples)
        row["edge"] = row["selection"]["edge"]
        selection_path = root / f"selection_{name}.json"
        # Exclusive creation prevents an accidental rerun from replacing a lock.
        with selection_path.open("x") as file:
            file.write(json.dumps(row["selection"], indent=2, allow_nan=False) + "\n")
        row["selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        row["status"] = "selected"
        save_json(root / "results.json", result)
        if method in ("adam", "unit", "network"):
            row["geometry_probe"] = reference.geometry_probe(learner, data["xv"][:128])
        elif hasattr(learner, "geometry"):
            row["geometry_probe"] = learner.geometry(data["xv"][:128])
        else:
            row["geometry_probe"] = {"status": "not_implemented", "claim": "no geometry evidence available"}
        require_finite(row["geometry_probe"], f"{name}.geometry")
    finally:
        row["resources"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        row["resources"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        del graph, learner


@torch.no_grad()
def score_selected(selection_path, curves, root, xt, test_targets, switch):
    """Read a locked decision; held-out scores cannot rewrite or rerank it."""
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    best = selection["chosen_index"]
    test_curve = []
    previous = 0
    sustained = 0.0
    for curve in curves:
        checkpoint = torch.load(root / curve["checkpoint"], map_location="cpu", weights_only=True)
        if checkpoint["step"] != curve["step"]:
            raise RuntimeError("checkpoint sample count does not match its curve")
        chosen = [w[best:best + 1].to(xt.device) for w in checkpoint["weights"]]
        phase = phase_at(curve["step"], switch)
        error = float(reference.evaluate(chosen, xt, test_targets[phase])[0])
        zero = float(test_targets[phase].square().mean())
        test_curve.append({"step": curve["step"], "teacher_phase": phase, "mse": error,
                           "zero_mse": zero, "over_zero": error / zero if zero > 0 else None})
        require_finite(test_curve[-1], "held_out_test")
        sustained += error * ((curve["step"] - previous) / selection["samples"])
        previous = curve["step"]
    if previous != selection["samples"]:
        raise ValueError("held-out reporting requires checkpoints over the complete stream")
    if selection_path.read_bytes() != selection_bytes:
        raise RuntimeError("selection lock changed during test evaluation")
    require_finite(sustained, "held_out_test_sustained")
    return {"test_curve": test_curve, "test_mse": test_curve[-1]["mse"],
            "test_sustained_mse": sustained, "test_over_zero": test_curve[-1]["over_zero"]}


def main():
    a = tyro.cli(Args)
    validate_args(a)
    root = Path(a.output) if a.output else Path("runs") / f"DenseStream__covariance_sketch_v3__{a.seed}__{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "results.json").exists() or any(root.glob("selection_*.json")):
        raise FileExistsError(f"refusing to overwrite an existing experiment at {root}")
    configs = arm_configs(a)
    result = {
        "args": asdict(a), "run_dir": str(root), "status": "initializing",
        "protocol": "single paired task seed; complete streams; duration-weighted validation per arm; all decisions locked before untouched test curves; no cross-seed inference",
        "unit_gain_semantics": "signed per-neuron incoming-row contribution to the mean update's linearized output change; joint posterior coordination, not autonomous scalar gates",
        "scalar_semantics": "same posterior conditioning rule; applied direction j*(j'Pj)/(j'j) erases orientation and preserves instantaneous gain; subsequent trajectories evolve independently",
        "geometry_semantics": "frozen validation inputs; directional probes cannot certify positive definiteness in unprobed directions",
        "source_sha256": {str(Path(module.__file__).relative_to(Path(__file__).resolve().parents[2])):
                          hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for module in (reference, sketch, runtime)},
        "data_hashes": {}, "timing": {}, "resources": {},
        "methods": {c["name"]: {**c, "status": "pending", "curves": [], "selection": None,
                                  "edge": None, "timing": {}, "resources": {}} for c in configs},
    }
    result["source_sha256"]["cleanrl/plasticity/covariance_sketch_eval_v3.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    writer = SummaryWriter(str(root))
    writer.add_text("hyperparameters", json.dumps(asdict(a), indent=2))
    writer.add_text("protocol", result["protocol"])
    writer.flush()
    save_json(root / "results.json", result)
    started = time.perf_counter()
    failed = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU or eager fallback")
        result["resources"] = {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                               "cuda": torch.version.cuda, "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                               "precision": "reference runtime task generation, then float32 highest precision without TF32",
                               "execution": "compiled CUDA graphs; no fallback; no per-sample host scalar extraction",
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
            except torch.cuda.OutOfMemoryError as error:
                # An unavailable exact reference remains a visible missing arm,
                # never silently replaced with a smaller model or grid.
                row.update(status="unavailable", failure={"type": type(error).__name__, "message": str(error)})
                failed = True
            except FloatingPointError as error:
                row.update(status="failed_nonfinite", failure={"type": type(error).__name__, "message": str(error)})
                failed = True
            except Exception as error:
                row.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
                failed = True
                raise
            finally:
                save_json(root / "results.json", result)
                writer.flush()
            torch.cuda.empty_cache()
        # Deliberate global barrier: no test scoring (including zero predictors)
        # above this point. Failed arms have no test-based replacement selection.
        result["status"] = "testing_locked_selections"
        result["selection_barrier"] = {name: row.get("selection_sha256") for name, row in result["methods"].items()}
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
                row["timing"]["test_seconds"] = time.perf_counter() - scored
                row["status"] = "completed"
                for point in row["test_curve"]:
                    writer.add_scalar(f"test/{name}/clean_mse", point["mse"], point["step"])
                writer.add_scalar(f"selection/{name}/edge", int(row["edge"]), a.samples)
            except FloatingPointError as error:
                row.update(status="failed_nonfinite", failure={"type": type(error).__name__, "message": str(error)})
                failed = True
            except Exception as error:
                row.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
                failed = True
                raise
            finally:
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

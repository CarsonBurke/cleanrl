"""Paired full-stream IEKF experiment; validation locks precede all held-out scoring.

Run with CUDA through mlq. The primary arm is fixed to iterated4, not chosen by
held-out results. Refinement reuses one observation and conditions covariance once.
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

from cleanrl.plasticity import covariance_sketch_eval_v3 as protocol
from cleanrl.plasticity import iterated_bayes_stream_v4 as iterated
from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.shared import runtime


@dataclass
class Args(reference.Args):
    hetero: float = 1.0
    switch_at: float = 0.5
    methods: tuple[str, ...] = ("adam", "network", "iterated")
    iterations: tuple[int, ...] = (2, 4)


def validate_args(a):
    reference.validate_args(replace(a, methods=("network",)))
    if len(a.methods) != 3 or set(a.methods) != {"adam", "network", "iterated"}:
        raise ValueError("methods must include exactly adam, network and iterated")
    if (not a.iterations or len(set(a.iterations)) != len(a.iterations)
            or any(type(n) is not int or n <= 0 for n in a.iterations)):
        raise ValueError("iterations must be nonempty, positive integers and unique")
    if 4 not in a.iterations:
        raise ValueError("iterations must include the fixed primary iterated4")
    if any(not math.isfinite(v) for v in (a.noise, a.hetero, a.diffusion, a.noise_rate, a.switch_at)):
        raise ValueError("noise, hetero, diffusion, noise_rate and switch_at must be finite")
    protocol.validate_cadence(a, a.graph_steps)
    protocol.validate_cadence(a, 1)


def arm_configs(a):
    return [{"name": f"iterated{n}" if method == "iterated" else method,
             "method": method, "iterations": n,
             "grid": list(a.adam_lrs if method == "adam" else a.prior_scales),
             "parameter": "learning_rate" if method == "adam" else "prior_scale"}
            for method in a.methods
            for n in (a.iterations if method == "iterated" else (1 if method == "network" else 0,))]


@torch.no_grad()
def train_arm(a, config, data, root, writer, result):
    name, method, grid = config["name"], config["method"], config["grid"]
    row = result["methods"][name]
    checkpoint_dir = root / "checkpoints" / name
    checkpoint_dir.mkdir(parents=True)
    # The previous arm has disposed its graph/learner. Disk kernel caches survive
    # this reset, but shared no_grad wrapper specialization guards do not.
    torch.compiler.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    learner = graph = None
    stage = "startup_seconds"
    try:
        arguments = (grid, data["initial"], a, data["xs"], data["ys"], data["clean"], data["noise_var"])
        learner = (iterated.IteratedLearner(config["iterations"], *arguments) if method == "iterated"
                   else reference.Learner(method, *arguments))
        protocol.validate_cadence(a, learner.capture_steps)
        if method != "adam" and learner.capture_steps != 1:
            raise RuntimeError("full covariance arms require capture_steps=1")
        row["capture_steps"] = learner.capture_steps
        row["resources"].update(grid_candidates=len(grid),
                                mutable_tensor_bytes=sum(t.numel() * t.element_size() for t in learner.mutable))
        row["status"] = "capturing"
        protocol.save_json(root / "results.json", result)
        graph, parity = learner.capture()
        torch.cuda.synchronize()
        row["graph_parity_max_abs"] = parity
        protocol.require_finite(parity, "capture parity")
        if int(learner.index.item()) != 0 or float(learner.steps.item()) != 0:
            raise RuntimeError("capture did not restore the stream position")
        row["timing"][stage] = time.perf_counter() - started
        stage, started = "training_seconds", time.perf_counter()
        row["status"] = "training"
        previous = 0
        for step in range(learner.capture_steps, a.samples + 1, learner.capture_steps):
            graph.replay()
            if step % a.log_every and step != a.samples and step != data["switch"]:
                continue
            phase = protocol.phase_at(step, data["switch"])
            curve = {"step": step, "interval_samples": step - previous, "teacher_phase": phase,
                     "validation": reference.evaluate(learner.weights, data["xv"], data["yv"][phase]).cpu().tolist(),
                     "online_clean_mse": (learner.error / step).cpu().tolist(),
                     "online_zero_mse": float(learner.null_error / step)}
            if method != "adam":
                curve.update(final_linearization_leverage=(learner.gain_sum / step).cpu().tolist(),
                             final_linearization_row_sd=(learner.unit_variance / step).sqrt().cpu().tolist(),
                             noise_estimate=learner.noise.cpu().tolist())
            row["curves"].append(curve)
            row["resources"].update(consumed_samples=int(learner.index.item()),
                                    learner_steps=float(learner.steps.item()),
                                    candidate_sample_updates=int(learner.index.item()) * len(grid))
            if row["resources"]["consumed_samples"] != step or row["resources"]["learner_steps"] != step:
                raise RuntimeError("CUDA replay sample count or learner clock differs from the reported step")
            protocol.require_finite(curve, f"{name}.curve")
            if not bool(torch.stack([torch.isfinite(t).all() for t in learner.mutable]).all().item()):
                raise FloatingPointError(f"nonfinite mutable learner state in {name} at sample {step}")
            checkpoint = checkpoint_dir / f"{step}.pt"
            torch.save({"step": step, "weights": [w.detach().cpu() for w in learner.weights]}, checkpoint)
            curve["checkpoint"] = str(checkpoint.relative_to(root))
            for key, values in curve.items():
                if isinstance(values, list):
                    for index, value in enumerate(grid):
                        writer.add_scalar(f"{key}/{name}/{value:g}", values[index], step)
            previous = step
            row["timing"][stage] = time.perf_counter() - started
            protocol.save_json(root / "results.json", result)
            writer.flush()
            print(json.dumps({"arm": name, **curve}, allow_nan=False), flush=True)
        torch.cuda.synchronize()
        row["timing"][stage] = time.perf_counter() - started
        row["timing"]["aggregate_samples_per_second"] = a.samples * len(grid) / row["timing"][stage]
        stage, started = "selection_seconds", time.perf_counter()
        row["selection"] = protocol.validation_selection(row["curves"], grid, a.samples)
        row["edge"] = row["selection"]["edge"]
        row[f"selected_{config['parameter']}"] = row["selection"]["chosen"]
        selection_path = root / f"selection_{name}.json"
        with selection_path.open("x") as file:
            file.write(json.dumps(row["selection"], indent=2, allow_nan=False) + "\n")
        row["selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        row["status"] = "selected"
    finally:
        row["timing"][stage] = time.perf_counter() - started
        row["resources"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if learner is not None:
            row["resources"].update(consumed_samples=int(learner.index.item()),
                                    learner_steps=float(learner.steps.item()),
                                    candidate_sample_updates=int(learner.index.item()) * len(grid))
        del graph, learner


def main():
    a = tyro.cli(Args)
    validate_args(a)
    root = Path(a.output) if a.output else Path("runs") / f"DenseStream__iterated_bayes_v4__{a.seed}__{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"refusing to mix experiment outputs in nonempty directory {root}")
    configs = arm_configs(a)
    sources = [Path(module.__file__) for module in (reference, iterated, protocol, runtime)] + [Path(__file__)]
    result = {
        "args": asdict(a), "run_dir": str(root), "status": "initializing", "primary_arm": "iterated4",
        "protocol": "single paired task seed; complete shared streams and initialization; independent duration-weighted clean validation selection per arm; every decision locked before any held-out scoring; no cross-seed inference",
        "gain_semantics": "signed per-incoming-row FINAL-linearization leverage, averaged across consumed samples; row_sd is the square root of time-mean across-row variance; neither measures the total nonlinear output change from refinement; Adam has no leverage metric",
        "execution": "CUDA FP32 highest precision without TF32; compiled CUDA graphs with no fallback; full covariance capture_steps=1; compiler guards reset between arms",
        "source_sha256": {str(path.resolve().relative_to(Path(__file__).resolve().parents[2])):
                          hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "data_hashes": {}, "timing": {}, "resources": {},
        "methods": {c["name"]: {**c, "status": "pending", "curves": [], "selection": None,
                                  "edge": None, "timing": {}, "resources": {}} for c in configs},
    }
    writer = SummaryWriter(str(root))
    writer.add_text("hyperparameters", json.dumps(asdict(a), indent=2))
    writer.add_text("protocol", result["protocol"])
    writer.add_text("gain_semantics", result["gain_semantics"])
    protocol.save_json(root / "results.json", result)
    started = time.perf_counter()
    failed = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU or eager fallback")
        result["resources"] = {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                               "cuda": torch.version.cuda, "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                               "memory_values_are_measured": True}
        generated = time.perf_counter()
        data = protocol.draw_data(a)
        torch.cuda.synchronize()
        result["timing"]["generation_and_hashing_seconds"] = time.perf_counter() - generated
        result["data_hashes"] = data["hashes"]
        result["resources"]["parameters_per_candidate"] = sum(w.numel() for w in data["initial"])
        result["status"] = "training_and_selecting"
        for config in configs:
            row = result["methods"][config["name"]]
            try:
                train_arm(a, config, data, root, writer, result)
            except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
                row.update(status="failed_nonfinite" if isinstance(error, FloatingPointError) else "unavailable",
                           failure={"type": type(error).__name__, "message": str(error)})
                failed = True
            except BaseException as error:
                row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           failure={"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                protocol.save_json(root / "results.json", result)
                writer.flush()
            torch.cuda.empty_cache()
        # Global barrier: even zero-predictor test risk stays untouched until all
        # arms have selected or failed. Failed candidates never disappear from a grid.
        result["status"] = "testing_locked_selections"
        result["selection_barrier"] = {name: row.get("selection_sha256") for name, row in result["methods"].items()}
        protocol.save_json(root / "results.json", result)
        for name, row in result["methods"].items():
            if row["status"] != "selected":
                continue
            scored = time.perf_counter()
            try:
                path = root / f"selection_{name}.json"
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["selection_sha256"]:
                    raise RuntimeError("persisted selection changed before test scoring")
                row.update(protocol.score_selected(path, row["curves"], root, data["xt"], data["test_targets"], data["switch"]))
                row["status"] = "completed"
                for point in row["test_curve"]:
                    writer.add_scalar(f"test/{name}/clean_mse", point["mse"], point["step"])
                writer.add_scalar(f"selection/{name}/edge", int(row["edge"]), a.samples)
                writer.add_scalar(f"selection/{name}/{row['parameter']}", row["selection"]["chosen"], a.samples)
            except FloatingPointError as error:
                row.update(status="failed_nonfinite", failure={"type": type(error).__name__, "message": str(error)})
                failed = True
            except BaseException as error:
                row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           failure={"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                row["timing"]["test_seconds"] = time.perf_counter() - scored
                protocol.save_json(root / "results.json", result)
                writer.flush()
        primary = result["methods"]["iterated4"]
        result["primary_comparison"] = {}
        for baseline in ("adam", "network"):
            control = result["methods"][baseline]
            complete = primary["status"] == control["status"] == "completed"
            result["primary_comparison"][baseline] = {"status": "completed" if complete else "unavailable"}
            if complete:
                risk, reference_risk = primary["test_sustained_mse"], control["test_sustained_mse"]
                result["primary_comparison"][baseline].update(
                    sustained_mse_difference=risk - reference_risk,
                    relative_reduction=1 - risk / reference_risk if reference_risk > 0 else None)
        protocol.require_finite(result["primary_comparison"], "primary comparison")
        result["status"] = "failed" if failed else "completed"
    except BaseException as error:
        result["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["timing"]["total_seconds"] = time.perf_counter() - started
        protocol.save_json(root / "results.json", result)
        writer.add_text("status", result["status"])
        writer.close()
        print(f"RESULTS {root / 'results.json'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

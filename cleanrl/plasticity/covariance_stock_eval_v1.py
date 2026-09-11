"""Full-width frozen-v2 covariance with a complete-stream maximum horizon.

Run only through mlq. No bars are read and no preprocessing occurs at import.
The random-sign view is a conditional-mean diagnostic, NOT a finance test.

Resource pruning is ON by default; --no-autocull requests fixed-horizon evidence.
The shared proxy policy checks smoothed training progress after warmup, gives
scheduled regimes fresh patience, and stops the entire job when all candidates
stagnate. Partial artifacts are explicitly pruned, never full-run comparisons.
Submit with --max-attempts 1: intentional pruning exits75 rather than success,
so mlq after-success descendants cannot start. No external watcher is required.
"""

import gc
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import network_bayes_stream_v2 as bayes
from cleanrl.plasticity import stock_stream
from cleanrl.shared.autocull import ProxyCull, ProxyPruned, prune_proxy, PRUNED_EXIT_CODE
from cleanrl.shared import runtime


@dataclass
class Args:
    bars: str = stock_stream.Args.bars
    seed: int = 1
    methods: tuple[str, ...] = ("adam", "covariance")
    views: tuple[str, ...] = ("real", "random_sign", "planted")
    covariance_diffusions: tuple[float, ...] = (1e-5, 0.0)
    adam_lrs: tuple[float, ...] = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
                                    1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    cold_start: int = 2000
    log_every: int = 8192
    trace_points: int = 1024
    autocull: bool = True
    """Default resource guard; --no-autocull explicitly requests full-horizon evidence.
    Pruned runs save partial results and exit75; submit with --max-attempts 1."""
    output: str = "runs"


class MeasuredLearner(bayes.Learner):
    """Only add pre-update observation; the frozen initializer/update are inherited."""

    def __init__(self, method, grid, initial, a, xs, ys):
        if method not in ("network", "adam"):
            raise ValueError("stock evaluation supports full covariance and Adam only")
        super().__init__(method, grid, initial, a, xs, ys, ys, torch.ones_like(ys))
        self.capture_steps = 1
        self.predictions = torch.zeros((len(xs), len(grid)), device=xs.device)
        self.mutable.append(self.predictions)

    @torch.no_grad()
    def update(self):
        ix = self.index.reshape(1)
        x = self.xs.index_select(0, ix).squeeze(0)
        prediction = bayes.sample_state(self.weights, x)[0]
        self.predictions.index_copy_(0, ix, prediction.unsqueeze(0))
        # No posterior, noise-estimation, optimizer, or initialization changes.
        super().update()


def phase_ranges(samples, cold_start):
    prefix = samples // 4
    switch = prefix + (samples - prefix) // 2
    if not 0 < cold_start < prefix < switch < samples:
        raise ValueError("need a nonempty cold start, first-quarter prefix, and both suffix phases")
    return {"cold_start": (0, cold_start), "selection_prefix": (cold_start, prefix),
            "prefix_all": (0, prefix), "suffix_positive": (prefix, switch),
            "suffix_reversed": (switch, samples), "suffix_all": (prefix, samples)}


def metric_sums(target, prediction):
    """Float64 offline reductions of actual FP32 prequential outputs, per arm."""
    y = np.asarray(target, dtype=np.float64).reshape(-1, 1)
    p = np.asarray(prediction, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    if p.shape[0] != y.shape[0] or not len(y):
        raise ValueError("nonempty target and prediction lengths must match")
    target2 = float(np.square(y).sum())
    prediction2 = np.square(p).sum(axis=0)
    cross = (y * p).sum(axis=0)
    error2 = np.square(y - p).sum(axis=0)
    denominator = target2 if target2 > 0 else math.nan
    return [{"count": len(y), "target_squared_sum": target2,
             "prediction_squared_sum": float(pp), "target_prediction_sum": float(yp),
             "error_squared_sum": float(ee), "error_ratio": float(ee / denominator),
             "prediction_energy_ratio": float(pp / denominator),
             "signed_cross_term_ratio": float(-2 * yp / denominator),
             "decomposition_residual": float((ee - target2 - pp + 2 * yp) / denominator)}
            for pp, yp, ee in zip(prediction2, cross, error2)]


def select_prefix(predictions, target, grid, cold_start, endpoint, consumed):
    """A lock is valid only at the exact endpoint, before any suffix update."""
    if consumed != endpoint or len(predictions) != endpoint or len(target) != endpoint:
        raise ValueError("selection must occur exactly at the first-quarter endpoint, before suffix")
    if not 0 <= cold_start < endpoint:
        raise ValueError("invalid selection scoring interval")
    candidates = metric_sums(target[cold_start:endpoint], predictions[cold_start:endpoint])
    if len(candidates) != len(grid):
        raise ValueError("one prediction column is required per candidate")
    scores = [row["error_ratio"] for row in candidates]
    finite = [i for i, value in enumerate(scores) if math.isfinite(value)]
    if not finite:
        raise RuntimeError("no finite prefix candidate; cannot lock a hyperparameter")
    winner = min(finite, key=lambda i: scores[i])
    return {"selected_index": winner, "selected_lr": grid[winner],
            "selection_start_inclusive": cold_start, "selection_end_exclusive": endpoint,
            "optimizer_updates_at_lock": consumed, "suffix_observations_used": 0,
            "criterion": "minimum prequential squared error / zero-predictor squared error",
            "tie_break": "first (smallest) learning rate", "candidates": [
                {"lr": lr, **metrics} for lr, metrics in zip(grid, candidates)]}


def build_views(features, target, prefix, switch, seed):
    # Stock helper's final return-channel slot. It is two bars back in its
    # ACTUAL implementation, despite the helper's one-bar feature_name label.
    column = (stock_stream.Args.lags - 1) * len(stock_stream.CHANNELS)
    source = features[:, column].astype(np.float64)
    mean, std = float(source[:prefix].mean()), float(source[:prefix].std())
    if not math.isfinite(std) or std <= 0:
        raise ValueError("prefix lagged-return feature has no finite variation")
    z = ((source - mean) / std).astype(np.float32)
    signal = np.float32(0.03) * z
    signal[switch:] *= -1
    signs = np.random.default_rng(seed).integers(0, 2, size=len(target), dtype=np.int8) * 2 - 1
    return {"real": target, "random_sign": target * signs,
            "planted": target + signal}, signal, {
                "column": column, "actual_feature_bars_before_target": 2,
                "prefix_mean": mean, "prefix_std_ddof0": std,
                "standardization_end_exclusive": prefix, "amplitude": 0.03,
                "sign_switch_index": switch,
                "standardization_note": "fixed fit on full selection prefix; no suffix information; prefix calibration is not an untouched evaluation"}


def save_result(root, result):
    temporary = root / "results.json.tmp"
    temporary.write_text(json.dumps(bayes.finite_json(result), indent=2, allow_nan=False))
    temporary.replace(root / "results.json")


def run_arm(name, method, grid, diffusion, initial, xs, target, args, phases,
            root, writer, result, trace_indices, lock_real=False):
    device = xs.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    a = bayes.Args(seed=args.seed, samples=len(xs), input_dim=xs.shape[1], hidden=64,
                   diffusion=diffusion, graph_steps=1, known_noise=False)
    ys = torch.as_tensor(target, device=device)
    learner = MeasuredLearner(method, grid, initial, a, xs, ys)
    before_capture = time.perf_counter()
    graph, agreement = learner.capture()
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - before_capture
    prefix = phases["prefix_all"][1]
    boundaries = sorted(set([*range(args.log_every, len(xs), args.log_every),
                             phases["cold_start"][1], prefix,
                             phases["suffix_positive"][1], len(xs)]))
    previous = 0
    replay_seconds = 0.0
    arm = {"method": method, "grid": list(grid), "diffusion": diffusion,
           "prior_scale": 1.0 if method == "network" else None,
           "status": "running", "capture_steps": learner.capture_steps,
           "capture_seconds": capture_seconds, "graph_max_absolute_error": agreement,
           "selection": "fixed ex ante prior1" if method == "network" else "real-prefix LR lock",
           "methodology": ("original frozen v2 prior1 diffusion1e-5" if method == "network" and diffusion == 1e-5
                           else "zero-diffusion diagnostic control" if method == "network" else "Adam control")}
    result["arms"][name] = arm
    guard = ProxyCull(len(grid), {"error_ratio": 1e-4, **(
        {"prediction_energy_ratio": 1e-4} if name.endswith("random_sign") else {})})
    save_result(root, result)
    for end in boundaries:
        start_time = time.perf_counter()
        for _ in range(previous, end):
            graph.replay()
        torch.cuda.synchronize()
        replay_seconds += time.perf_counter() - start_time
        window = learner.predictions[previous:end].cpu().numpy()
        interval_metrics = metric_sums(target[previous:end], window)
        for index, metrics in enumerate(interval_metrics):
            for key in ("error_ratio", "prediction_energy_ratio", "signed_cross_term_ratio"):
                writer.add_scalar(f"{name}/candidate_{index}/{key}", metrics[key], end)
        consumed = int(learner.index.item())
        if consumed != end or int(learner.steps.item()) != end:
            raise RuntimeError("optimizer count does not match chronological stream position")
        if end == prefix:
            prefix_predictions = learner.predictions[:prefix].cpu().numpy()
            arm["prefix_candidates"] = [
                {"scale": value, **metrics} for value, metrics in zip(
                    grid, metric_sums(target[args.cold_start:prefix], prefix_predictions[args.cold_start:]))]
            if lock_real:
                result["adam_real_prefix_lock"] = select_prefix(
                    prefix_predictions, target[:prefix], grid, args.cold_start, prefix, consumed)
                arm["locked_index"] = result["adam_real_prefix_lock"]["selected_index"]
            save_result(root, result)  # Durable lock BEFORE the first suffix replay.
        arm["optimizer_updates_per_candidate"] = consumed
        arm["total_optimizer_updates"] = consumed * len(grid)
        arm["replay_seconds"] = replay_seconds
        writer.add_scalar(f"{name}/updates_per_candidate", consumed, end)
        writer.add_scalar(f"{name}/updates_per_second", end / replay_seconds, end)
        writer.flush()
        if args.autocull:
            switch = phases["suffix_positive"][1]
            after_switch = name.endswith("planted") and previous >= switch
            decision = guard.observe(
                end, {metric: [row[metric] for row in interval_metrics] for metric in guard.metrics},
                phase="reversed" if after_switch else "stationary",
                phase_start=switch if after_switch else 0)
            if name.endswith("planted") and end == switch:
                decision = None  # The new regime is entitled to its own warmup.
            arm["autocull_state"] = guard.state_dict()
            save_result(root, result)
            if decision:
                arm.update(status="pruned", pruning=decision)
                # Unconsumed preallocated zeros are NOT predictions or evidence.
                partial = learner.predictions[:end].cpu().numpy()
                np.save(root / f"{name}_predictions.npy", partial)
                arm["prediction_artifact"] = f"{name}_predictions.npy"
                arm["phase_metrics"] = {
                    phase: metric_sums(target[start:min(stop, end)], partial[start:min(stop, end)])
                    for phase, (start, stop) in phases.items() if start < end}
                arm["phase_observed_ranges"] = {
                    phase: [start, min(stop, end)] for phase, (start, stop) in phases.items() if start < end}
                torch.save({"step": end, "weights": [w.cpu() for w in learner.weights],
                            "checkpoint_scope": "analysis only; covariance and optimizer moments omitted; not resumable",
                            "noise": learner.noise.cpu(), "autocull": guard.state_dict()},
                           root / f"{name}_pruned_checkpoint.pt")
                arm["wall_seconds"] = time.perf_counter() - started
                result.update(status="pruned", pruning={**decision, "arm": name})
                save_result(root, result)
                prune_proxy(root, name, decision)
        previous = end
    predictions = learner.predictions.cpu().numpy().copy()
    arm["phase_metrics"] = {phase: metric_sums(target[start:end], predictions[start:end])
                            for phase, (start, end) in phases.items()}
    np.save(root / f"{name}_predictions.npy", predictions)
    arm["prediction_artifact"] = f"{name}_predictions.npy"
    arm["trace"] = {"indices": trace_indices.tolist(), "target": target[trace_indices].tolist(),
                    "prediction": predictions[trace_indices].tolist()}
    arm["wall_seconds"] = time.perf_counter() - started
    arm["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    arm["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    arm["all_predictions_finite"] = bool(np.isfinite(predictions).all())
    arm["status"] = "completed" if arm["all_predictions_finite"] else "completed_nonfinite"
    save_result(root, result)  # Completed evidence survives any later arm failure.
    print(f"ARM {name} updates={len(xs)} replay_seconds={replay_seconds:.1f} status={arm['status']}", flush=True)
    del graph, learner, ys
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def paired_response(real, planted, signal, phases):
    response = planted - real
    return {"interpretation": "paired planted-minus-real prediction response against added signal; not total true clean MSE",
            "phase_metrics": {phase: metric_sums(signal[start:end], response[start:end])
                              for phase, (start, end) in phases.items()}}


def main():
    args = tyro.cli(Args)
    if args.seed != 1:
        raise ValueError("this predeclared evaluation uses seed1 only")
    for name, values, allowed in (
            ("methods", args.methods, {"adam", "covariance"}),
            ("views", args.views, {"real", "random_sign", "planted"}),
            ("covariance_diffusions", args.covariance_diffusions, {0.0, 1e-5})):
        if not values or len(set(values)) != len(values) or set(values) - allowed:
            raise ValueError(f"invalid or duplicate {name}: {values}")
    if "adam" in args.methods and "real" not in args.views:
        raise ValueError("Adam requires the real view for its prefix LR lock; covariance-only views can run independently")
    if min(args.log_every, args.trace_points) <= 0:
        raise ValueError("logging cadence and trace count must be positive")
    if (len(args.adam_lrs) < 3 or list(args.adam_lrs) != sorted(set(args.adam_lrs))
            or any(not math.isfinite(x) or x <= 0 for x in args.adam_lrs)):
        raise ValueError("Adam grid must have at least three strictly increasing positive finite rates")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU learner fallback")
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    root = Path(args.output) / f"SPY__covariance_stock_v1__1__{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(root))
    result = {"args": asdict(args), "run_dir": str(root), "status": "preprocessing",
              "arms": {}, "paired_responses": {}}
    save_result(root, result)
    try:
        # Complete original default stream: no projection, row truncation, raw
        # target constant fit, optional constant vol feature, or helper changes.
        helper_args = stock_stream.Args(seed=1, steps=0)
        bars = stock_stream.read_bars(args.bars)
        features, target = stock_stream.build_stream(bars, helper_args)
        # Separate queued views must describe the same fixed stream, even if
        # the external bars file or helper changes between their start times.
        result["data_sha256"] = {
            "features": hashlib.sha256(memoryview(features).cast("B")).hexdigest(),
            "target": hashlib.sha256(memoryview(target).cast("B")).hexdigest()}
        result["source_sha256"] = {
            str(Path(module.__file__).name): hashlib.sha256(
                Path(module.__file__).read_bytes()).hexdigest()
            for module in (stock_stream, bayes)}
        phases = phase_ranges(len(target), args.cold_start)
        prefix, switch = phases["prefix_all"][1], phases["suffix_positive"][1]
        views, signal, planting = build_views(features, target, prefix, switch, args.seed)
        device = torch.device("cuda")
        xs = torch.as_tensor(features, device=device)
        a = bayes.Args(seed=1, hidden=64, input_dim=features.shape[1])
        initial = bayes.init_weights(a, torch.Generator(device=device).manual_seed(1), device)
        with torch.no_grad():
            frozen_weights = [w.unsqueeze(0) for w in initial]
            initial_predictions = np.concatenate([
                bayes.forward(frozen_weights, chunk)[2].squeeze(0).cpu().numpy()
                for chunk in xs.split(4096)])
        np.save(root / "frozen_initial_predictions.npy", initial_predictions)
        result["frozen_initial_model"] = {
            "interpretation": "v2 starts with nonzero output; trained null prediction energy includes inherited initial output, not only newly absorbed noise",
            "prediction_artifact": "frozen_initial_predictions.npy",
            "phase_metrics": {
                view: {phase: metric_sums(y[start:end], initial_predictions[start:end])
                       for phase, (start, end) in phases.items()}
                for view, y in views.items()}}
        for view, phase_metrics in result["frozen_initial_model"]["phase_metrics"].items():
            for phase, rows in phase_metrics.items():
                for key in ("error_ratio", "prediction_energy_ratio", "signed_cross_term_ratio"):
                    writer.add_scalar(f"frozen_initial/{view}/{phase}/{key}", rows[0][key], phases[phase][1])
        writer.flush()
        parameters = sum(w.numel() for w in initial)
        p_bytes = parameters * parameters * 4
        trace_indices = np.unique(np.concatenate([
            np.linspace(0, len(target) - 1, min(args.trace_points, len(target)), dtype=np.int64),
            np.array([args.cold_start - 1, args.cold_start, prefix - 1, prefix, switch - 1, switch])]))
        result.update({"status": "running", "bars": len(bars), "samples": len(target),
                       "input_dim": features.shape[1], "hidden": 64, "parameters": parameters,
                       "phases": phases, "planting": planting, "stock_helper_args": asdict(helper_args),
                       "protocol": {
                           "views": "identical features and frozen-v2 initial weights for every arm",
                           "selection": "Adam selected on real chronological first-quarter prequential score excluding separately reported cold start; durable lock before suffix; no null or planted-specific tuning",
                           "adam_grid_suffix": "all grid rows continue until horizon or explicit prune; only the prefix-locked row is the selected comparator",
                           "covariance": "prior1 fixed ex ante; diffusion1e-5 original plus diffusion0 control; up to six serial arms",
                           "random_sign": "independent per-observation Rademacher signs times original targets, preserving conditional magnitude/volatility; zero conditional-mean diagnostic, not a financial significance test",
                           "suffix": "continued online learning, not frozen-weight evaluation; no suffix hyperparameter tuning",
                           "autocull": "default-on phase-local smoothed progress with warmup and patience; all candidates must stagnate; training errors and null prediction energy only; --no-autocull requests fixed-horizon evidence",
                           "precision": "FP32 covariance and learner arithmetic; TF32 disabled; compiled single-update CUDA graphs",
                           "alignment_disclosure": "helper row t features bars t..t+31, target ret[t+33]; newest return is two bars old, contrary to helper docs/feature_name; preserved unchanged",
                           "normalization_disclosure": "helper target centering and volatility use intervening bar t+32 (pre-target but newer than final input t+31); all features retain causal trailing channel scaling and clipping; helper omits final otherwise possible feature window; default raw_target=False avoids retrospective cold-start constant scaling; vol_feature=False avoids its identically-zero scaled-target feature",
                           "limitations": "one market and one seed; no financial significance, global optimality, or true clean market MSE claims"},
                       "resources": {"full_covariance_bytes_per_arm": p_bytes,
                                     "capture_snapshot_pair_bytes": 2 * p_bytes,
                                     "minimum_covariance_traffic_bytes_per_arm": 3 * len(target) * p_bytes,
                                     "six_arm_minimum_covariance_traffic_bytes": 18 * len(target) * p_bytes,
                                     "planning": "O(P^2) per observation; P plus two audit snapshots and compiler/graph workspaces; serial covariance arms, never six resident posteriors; measured capture/replay times and CUDA peaks recorded per arm",
                                     "runtime_estimate": "traffic / measured sustainable bandwidth is only an optimistic floor; actual runtime is measured, not inferred from a shortened learning run"}})
        np.save(root / "planted_signal.npy", signal)
        writer.add_text("protocol", json.dumps(result["protocol"], indent=2))
        save_result(root, result)
        print(f"PLAN samples={len(target)} inputs={features.shape[1]} hidden=64 parameters={parameters} P_GiB={p_bytes / 2**30:.3f} six_arm_min_traffic_PB={18 * len(target) * p_bytes / 1e15:.3f}", flush=True)
        if "adam" in args.methods:
            real_adam = run_arm("adam_real_grid", "adam", args.adam_lrs, 0.0, initial, xs,
                                views["real"], args, phases, root, writer, result, trace_indices, lock_real=True)
            choice = result["adam_real_prefix_lock"]
            selected_real = real_adam[:, choice["selected_index"]:choice["selected_index"] + 1]
            for view in ("random_sign", "planted"):
                if view not in args.views:
                    continue
                pred = run_arm(f"adam_{view}", "adam", (choice["selected_lr"],), 0.0, initial, xs,
                               views[view], args, phases, root, writer, result, trace_indices)
                if view == "planted":
                    result["paired_responses"]["adam"] = paired_response(selected_real, pred, signal, phases)
                    save_result(root, result)
            del real_adam, selected_real
        if "covariance" in args.methods:
            for diffusion in args.covariance_diffusions:
                label = "original" if diffusion == 1e-5 else "zero_diffusion"
                real = None
                for view in ("real", "random_sign", "planted"):
                    if view not in args.views:
                        continue
                    pred = run_arm(f"covariance_{label}_{view}", "network", (1.0,), diffusion, initial, xs,
                                   views[view], args, phases, root, writer, result, trace_indices)
                    if view == "real":
                        real = pred
                    elif view == "planted":
                        result["paired_responses"][f"covariance_{label}"] = (
                            paired_response(real, pred, signal, phases) if real is not None else
                            {"status": "requires_matching_real_prediction_artifact",
                             "reason": "real arm excluded by explicit view filter; no substitute clean MSE reported"})
                        save_result(root, result)
                del real
        result["status"] = "completed"
        save_result(root, result)
        print(f"RESULTS {root / 'results.json'}", flush=True)
    except ProxyPruned:
        # Intentional prune is not success: dependent experiments must not start.
        raise SystemExit(PRUNED_EXIT_CODE)
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save_result(root, result)
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()

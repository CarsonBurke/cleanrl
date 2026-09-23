"""Sequential CUDA-only scale-normalized renewable predictive-direction benchmark, without model averaging.

Every learner receives only observed gradients. Clean synthetic targets and
support are evaluator-only. Configurations are selected using first-half
observed loss and frozen before the second half is consumed. Old candidate memory
is validated on later outcomes; accepted memory can enter or leave in unit jumps. All controls
complete the full horizon, including underperforming ones.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity.noisy_stream_diagnostic import Args as SparseArgs, Stream, clean_risk
from cleanrl.plasticity.stock_stream import Args as StockArgs, read_bars, build_stream, build_targets
from cleanrl.plasticity.predictive_writer_v6 import PredictiveWriterState


CHUNK = 100
LR_GRID = tuple(float(value) for value in np.logspace(-8, -1, 15))
CONTROLS = ("unit", "raw_line", "zero_retirement", "recheck", "coordinate", "block", "shuffled")
SEGMENTS = ("initial", "moved", "returned")


@dataclass
class Args:
    task: Literal["sparse", "stock"]
    seed: Literal[1] = 1
    signal_inputs: int = 1
    """4096 selects the dense synthetic signal; one is the sparse signal."""
    switch_at: float = 0.0
    """Nonzero enables the full 100000-observation support-switch experiment."""
    switch_back: float = 0.0
    switch_to: int = 2
    bars: str = StockArgs().bars
    output_dir: str = "runs"
    log_every: int | None = None
    """Defaults to 10000 synthetic observations or 100000 stock observations."""


def configurations(streams):
    configs, groups = [], {}
    for method in ("sgd", "adamw", *CONTROLS):
        start = len(configs)
        if method in ("sgd", "adamw"):
            settings = [{"lr": lr} for lr in LR_GRID]
        else:
            settings = [{}]
        for stream, info in enumerate(streams):
            for setting in settings:
                configs.append({"id": len(configs), "method": method, "stream": stream,
                                **info, **setting})
        groups[method] = slice(start, len(configs))
    return configs, groups


def named_buffers(state):
    buffers = state.buffers()
    if isinstance(buffers, dict):
        return buffers
    return {f"buffer_{index}": value for index, value in enumerate(buffers)}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Runner:
    def __init__(self, args, dim, streams, total, score_after, configs, groups):
        self.args, self.dim, self.streams, self.total = args, dim, streams, total
        self.configs, self.groups = configs, groups
        self.sparse = args.task == "sparse"
        device = torch.device("cuda")
        n, stream_count = len(configs), len(streams)
        self.weight = torch.zeros((n, dim), device=device)
        adam = groups["adamw"]
        self.m = torch.zeros_like(self.weight[adam])
        self.v = torch.zeros_like(self.m)
        self.step_count = torch.zeros((), device=device)
        self.count = torch.zeros(2, dtype=torch.float64, device=device)
        self.error = torch.zeros((2, n), dtype=torch.float64, device=device)
        self.excess = torch.zeros_like(self.error)
        self.prediction_power = torch.zeros_like(self.error)
        self.trivial = torch.zeros((2, stream_count), dtype=torch.float64, device=device)
        self.clean_error = torch.zeros_like(self.error)
        self.clean_segment_error = torch.zeros((3, n), dtype=torch.float64, device=device)
        self.clean_segment_count = torch.zeros(3, dtype=torch.float64, device=device)
        self.ever_nonfinite = torch.zeros(n, dtype=torch.bool, device=device)
        self.first_nonfinite = torch.full((n,), -1.0, device=device)
        self.stream_index = torch.tensor([cfg["stream"] for cfg in configs], device=device)
        self.lr = torch.tensor([cfg.get("lr", 0.0) for cfg in configs], device=device)[:, None]
        self.controllers = {}
        self.controller_buffers = {}
        self.previous_weight = torch.zeros_like(self.weight)
        self.update_energy = torch.zeros((3, n), dtype=torch.float64, device=device)
        self.signal_update_energy = torch.zeros_like(self.update_energy)
        self.noise_update_energy = torch.zeros_like(self.update_energy)
        self.large_useful_signal_updates = torch.zeros_like(self.update_energy)
        self.large_noise_updates = torch.zeros_like(self.update_energy)
        self.peak_signal_update = torch.zeros((3, n), device=device)
        self.peak_noise_update = torch.zeros_like(self.peak_signal_update)
        self.peak_update = torch.zeros_like(self.peak_signal_update)
        initial_support = torch.arange(dim, device=device) < args.signal_inputs
        moved_support = ((torch.arange(dim, device=device) >= args.switch_to) &
                         (torch.arange(dim, device=device) < args.switch_to + args.signal_inputs))
        for control in CONTROLS:
            rows = groups[control]
            state = PredictiveWriterState(self.weight[rows], control=control)
            self.controllers[control] = state
            self.controller_buffers[control] = named_buffers(state)
        self.states = [self.weight, self.m, self.v, self.step_count, self.count,
                       self.error, self.excess, self.prediction_power, self.trivial,
                       self.clean_error, self.clean_segment_error, self.clean_segment_count,
                       self.ever_nonfinite, self.first_nonfinite]
        self.states.extend((self.previous_weight, self.update_energy, self.signal_update_energy,
                            self.noise_update_energy, self.large_useful_signal_updates,
                            self.large_noise_updates, self.peak_signal_update,
                            self.peak_noise_update, self.peak_update))
        self.states.extend(value for buffers in self.controller_buffers.values()
                           for value in buffers.values())
        self.inputs = (torch.zeros((CHUNK, dim), device=device),
                       torch.zeros((CHUNK, stream_count), device=device),
                       torch.zeros((CHUNK, stream_count), device=device))
        self.segment_endpoint_risks = {}
        self.stock_intervals = []
        self.interval_previous = (0, 0.0, np.zeros(n), np.zeros(n), np.zeros(stream_count))
        epsilon = 1e-5 if self.sparse else 1e-8
        weight_decay = SparseArgs().weight_decay if self.sparse else StockArgs().weight_decay
        decay_coordinates = torch.ones((1, dim), device=device)
        if not self.sparse:
            decay_coordinates[:, -1] = 0
        midpoint = total // 2
        switch = int(total * args.switch_at) if args.switch_at else total
        back = int(total * args.switch_back) if args.switch_back else total

        def step(x, target_rows, clean_rows):
            self.previous_weight.copy_(self.weight)
            t = self.step_count
            segment = (torch.stack((t < switch, (t >= switch) & (t < back), t >= back))
                       if self.sparse else torch.tensor([True, False, False], device=device))
            prediction = (self.weight * x).sum(1)
            targets = target_rows[self.stream_index]
            residual = prediction - targets
            bad = ~torch.isfinite(prediction)
            self.first_nonfinite.copy_(torch.where(bad & ~self.ever_nonfinite, t + 1,
                                                   self.first_nonfinite))
            self.ever_nonfinite.logical_or_(bad)
            scored = t >= score_after
            phase = torch.stack((scored & (t < midpoint), scored & (t >= midpoint)))
            self.count.add_(phase.double())
            prediction64, targets64 = prediction.double(), targets.double()
            # Shared y^2 cancels before accumulation: weak stock effects need not
            # survive subtraction of two large, nearly equal accumulated losses.
            excess = prediction64.square() - 2.0 * prediction64 * targets64
            self.excess.add_(torch.where(phase[:, None], excess[None, :], 0.0))
            self.error.add_(torch.where(phase[:, None],
                                       (prediction64 - targets64).square()[None, :], 0.0))
            self.trivial.add_(torch.where(phase[:, None], target_rows.double().square()[None, :], 0.0))
            self.prediction_power.add_(torch.where(phase[:, None], prediction64.square()[None, :], 0.0))
            if self.sparse:
                clean_squared = (prediction64 - clean_rows[self.stream_index].double()).square()
                self.clean_error.add_(torch.where(phase[:, None], clean_squared[None, :], 0.0))
                self.clean_segment_count.add_(segment.double())
                self.clean_segment_error.add_(torch.where(segment[:, None], clean_squared[None, :], 0.0))
            self.step_count.add_(1)
            grad = residual[:, None] * x
            for control, state in self.controllers.items():
                rows = groups[control]
                # Optimizers never receive targets, clean labels, or support.
                state.step(self.weight[rows], grad[rows], x.square()[None, :])
            sgd = groups["sgd"]
            self.weight[sgd].sub_(self.lr[sgd] * grad[sgd])
            g = grad[adam]
            self.m.mul_(0.9).add_(g, alpha=0.1)
            self.v.mul_(0.999).addcmul_(g, g, value=0.001)
            update = ((self.m / (1.0 - 0.9 ** self.step_count)) /
                      ((self.v / (1.0 - 0.999 ** self.step_count)).sqrt() + epsilon))
            if self.sparse:
                self.weight[adam].mul_(1.0 - self.lr[adam] * weight_decay)
            self.weight[adam].sub_(self.lr[adam] * update)
            if not self.sparse:
                # Match original stock ordering and its undecayed intercept.
                self.weight[adam].sub_(self.lr[adam] * weight_decay *
                                       self.weight[adam] * decay_coordinates)

            delta = self.weight - self.previous_weight
            squared = delta.double().square()
            absolute = delta.abs()
            if self.sparse:
                support = torch.where(segment[1], moved_support, initial_support)
                signal = support[None, :] & (self.stream_index[:, None] == 0)
                noise = ~signal
                useful = ((self.weight - 1).square() < (self.previous_weight - 1).square()) & signal
                signal_energy = (squared * signal).sum(1)
                noise_energy = (squared * noise).sum(1)
                peak_signal = torch.where(signal, absolute, 0.0).max(1).values
                peak_noise = torch.where(noise, absolute, 0.0).max(1).values
                large_signal = ((absolute >= 0.1) & useful).sum(1).double()
                large_noise = ((absolute >= 0.1) & noise).sum(1).double()
                self.signal_update_energy.add_(torch.where(segment[:, None], signal_energy[None, :], 0.0))
                self.noise_update_energy.add_(torch.where(segment[:, None], noise_energy[None, :], 0.0))
                self.large_useful_signal_updates.add_(torch.where(segment[:, None], large_signal[None, :], 0.0))
                self.large_noise_updates.add_(torch.where(segment[:, None], large_noise[None, :], 0.0))
                self.peak_signal_update.copy_(torch.maximum(self.peak_signal_update, torch.where(segment[:, None], peak_signal[None, :], 0.0)))
                self.peak_noise_update.copy_(torch.maximum(self.peak_noise_update, torch.where(segment[:, None], peak_noise[None, :], 0.0)))
            self.update_energy.add_(torch.where(segment[:, None], squared.sum(1)[None, :], 0.0))
            self.peak_update.copy_(torch.maximum(self.peak_update, torch.where(segment[:, None], absolute.max(1).values[None, :], 0.0)))

        self.compiled = torch.compile(step, fullgraph=True, dynamic=False,
                                      options={"triton.cudagraphs": False})

    @torch.no_grad()
    def capture(self):
        snapshots = [value.clone() for value in self.states]
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        self.graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        try:
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    self.compiled(*(value[0] for value in self.inputs))
            capture_stream.synchronize()
            with torch.cuda.graph(self.graph, stream=capture_stream):
                for offset in range(CHUNK):
                    self.compiled(*(value[offset] for value in self.inputs))
            capture_stream.synchronize()
        finally:
            capture_stream.synchronize()
            for value, saved in zip(self.states, snapshots):
                value.copy_(saved)
            torch.cuda.synchronize()
        self.compile_seconds = time.perf_counter() - started
        self.capture_restored = all(torch.equal(value, saved)
                                    for value, saved in zip(self.states, snapshots))
        if not self.capture_restored:
            raise RuntimeError("CUDA warmup/capture failed to restore mutable state")

    @torch.no_grad()
    def advance(self, x, targets, clean=None):
        length = x.shape[0]
        self.inputs[0][:length].copy_(x)
        self.inputs[1][:length].copy_(targets)
        if clean is not None:
            self.inputs[2][:length].copy_(clean)
        if length == CHUNK:
            self.graph.replay()
        else:
            for offset in range(length):
                self.compiled(*(value[offset] for value in self.inputs))

    def record_interval(self, observed):
        if self.sparse:
            return
        previous_end, previous_count, previous_excess, previous_power, previous_zero = self.interval_previous
        count = float(self.count.sum().item())
        excess = self.excess.sum(0).cpu().numpy()
        power = self.prediction_power.sum(0).cpu().numpy()
        zero = self.trivial.sum(0).cpu().numpy()
        self.stock_intervals.append({
            "start_observation_inclusive": max(previous_end, 2000),
            "end_observation_exclusive": observed,
            "scored_count": count - previous_count,
            "configs": [{"id": cfg["id"], "stream": cfg["stream"], "method": cfg["method"],
                         "excess_sum": float(excess[i] - previous_excess[i]),
                         "prediction_power_sum": float(power[i] - previous_power[i]),
                         "zero_power_sum": float(zero[cfg["stream"]] - previous_zero[cfg["stream"]])}
                        for i, cfg in enumerate(self.configs) if cfg["alpha"] == 0.0]})
        self.interval_previous = (observed, count, excess.copy(), power.copy(), zero.copy())

    @torch.no_grad()
    def results(self, observed, elapsed, final_support=None):
        counts = self.count.cpu().tolist()
        error, excess, trivial, power, clean = (value.cpu().numpy() for value in
            (self.error, self.excess, self.trivial, self.prediction_power, self.clean_error))
        weights = self.weight.cpu().numpy()
        finite_weights = np.isfinite(weights).all(axis=1)
        finite_state = finite_weights.copy()
        adam = self.groups["adamw"]
        finite_state[adam] &= (np.isfinite(self.m.cpu().numpy()).all(axis=1) &
                               np.isfinite(self.v.cpu().numpy()).all(axis=1))
        controller_stats = {}
        for control, state in self.controllers.items():
            rows = self.groups[control]
            size = rows.stop - rows.start
            for value in self.controller_buffers[control].values():
                finite = torch.isfinite(value)
                if value.ndim and value.shape[0] == size:
                    finite_state[rows] &= finite.reshape(size, -1).all(1).cpu().numpy()
                else:
                    finite_state[rows] &= bool(finite.all().item())
            statistics = {}
            for label, value in (("memory", state.memory), ("allocation", state.allocation),
                                 ("validation_curvature", state.validation_curvature)):
                value = value.detach().double().reshape(size, -1)
                for stat, values in (("mean", value.mean(1)),
                                     ("std", value.std(1, correction=0)),
                                     ("min", value.min(1).values),
                                     ("max", value.max(1).values)):
                    statistics[f"{label}_{stat}"] = values.cpu().tolist()
            for local, index in enumerate(range(rows.start, rows.stop)):
                controller_stats[index] = {key: values[local] for key, values in statistics.items()}
        ever_bad = self.ever_nonfinite.cpu().tolist()
        first_bad = self.first_nonfinite.cpu().tolist()
        segment_counts = self.clean_segment_count.cpu().tolist()
        segment_error = self.clean_segment_error.cpu().numpy()
        risk_metrics = {}
        if self.sparse:
            assert final_support is not None
            probabilities = torch.full((self.dim,), 0.01, device=self.weight.device)
            for stream in range(len(self.streams)):
                indices = [index for index, cfg in enumerate(self.configs) if cfg["stream"] == stream]
                support = final_support if stream == 0 else torch.zeros_like(final_support)
                w = self.weight[indices]
                measured = clean_risk(w, support, probabilities)
                count_signal = support.sum().clamp_min(1)
                count_junk = (1 - support).sum().clamp_min(1)
                measured["signal_coefficient_mean"] = (w * support).sum(1) / count_signal
                measured["signal_coefficient_rmse"] = ((w - 1).square() * support).sum(1).div(count_signal).sqrt()
                measured["distractor_coefficient_rms"] = (w.square() * (1 - support)).sum(1).div(count_junk).sqrt()
                measured["coefficient_l2_error"] = (w - support).square().sum(1).sqrt()
                if self.args.switch_at:
                    historical = torch.zeros_like(support)
                    historical[:self.args.signal_inputs] = 1
                    historical[self.args.switch_to:self.args.switch_to + self.args.signal_inputs] = 1
                    stale = historical * (1 - support) if stream == 0 else torch.zeros_like(support)
                    measured["stale_coefficient_abs_mean"] = (w.abs() * stale).sum(1) / stale.sum().clamp_min(1)
                cpu_metrics = {name: value.cpu().tolist() for name, value in measured.items()}
                for local, index in enumerate(indices):
                    risk_metrics[index] = {name: values[local] for name, values in cpu_metrics.items()}
        changes = {name: getattr(self, name).cpu().numpy() for name in (
            "update_energy", "signal_update_energy", "noise_update_energy", "peak_update",
            "peak_signal_update", "peak_noise_update", "large_useful_signal_updates", "large_noise_updates")}
        rows = []
        for index, config in enumerate(self.configs):
            stream = config["stream"]
            row = {**config, "finite_weights": bool(finite_weights[index]),
                   "finite_optimizer_state": bool(finite_state[index]),
                   "finite_predictions_entire_pass": not ever_bad[index],
                   "first_nonfinite_prediction_step": int(first_bad[index]),
                   "weight_rms": float(np.sqrt(np.mean(weights[index].astype(np.float64) ** 2))),
                   "weight_l2": float(np.linalg.norm(weights[index].astype(np.float64))),
                   **risk_metrics.get(index, {}), **controller_stats.get(index, {})}
            row["parameter_changes"] = {f"{segment}_{name}": float(values[j, index])
                                        for name, values in changes.items()
                                        for j, segment in enumerate(SEGMENTS)}
            row.update(score_metrics(error[:, index], excess[:, index], trivial[:, stream],
                                     power[:, index], clean[:, index] if self.sparse else None, counts))
            if self.sparse:
                row["segment_prequential_clean_mse"] = {
                    name: float(segment_error[j, index] / segment_counts[j]) if segment_counts[j] else None
                    for j, name in enumerate(SEGMENTS)}
            else:
                row["bias"] = float(weights[index, -1])
            rows.append(row)
        return {"observations": observed, "total_observations": self.total,
                "reported_epoch_seconds": time.time(), "stock_intervals": list(self.stock_intervals),
                "support_segment_observation_counts": segment_counts if self.sparse else None,
                "segment_endpoint_exact_clean_risk": dict(self.segment_endpoint_risks),
                "phase_scored_counts": counts, "compile_capture_seconds": self.compile_seconds,
                "capture_mutable_state_restored": self.capture_restored,
                "pass_wall_seconds_including_reporting": elapsed,
                "observations_per_second": observed / elapsed,
                "config_observations_per_second": observed * len(rows) / elapsed,
                "finite_state_config_count": int(finite_state.sum()), "configs": rows}


def score_metrics(error, excess, trivial, power, clean, counts):
    out = {}
    for name, indices in (("first_half", (0,)), ("second_half", (1,)), ("all", (0, 1))):
        count = sum(counts[i] for i in indices)
        zero = float(sum(trivial[i] for i in indices))
        loss = float(sum(error[i] for i in indices))
        delta = float(sum(excess[i] for i in indices))
        out[f"{name}_mse"] = loss / count if count else None
        out[f"{name}_zero_mse"] = zero / count if count else None
        out[f"{name}_mse_ratio"] = 1 + delta / zero if zero else None
        out[f"{name}_relative_improvement_over_zero"] = -delta / zero if zero else None
        out[f"{name}_excess_mse"] = delta / count if count else None
        out[f"{name}_prediction_power"] = float(sum(power[i] for i in indices)) / count if count else None
        if clean is not None:
            out[f"{name}_clean_mse"] = float(sum(clean[i] for i in indices)) / count if count else None
    return out


def best_row(rows, key):
    finite = [row for row in rows if row.get(key) is not None and math.isfinite(row[key])
              and row["finite_optimizer_state"] and row["finite_predictions_entire_pass"]]
    return min(finite, key=lambda row: row[key]) if finite else None


def matched_config(rows, selected, target_stream):
    if selected is None:
        return None
    keys = ("method", "lr")
    return next(row for row in rows if row["stream"] == target_stream
                and all(row.get(key) == selected.get(key) for key in keys))


def comparisons(result, task, selection):
    rows, output = result["configs"], []
    for stream, method in sorted({(row["stream"], row["method"]) for row in rows}):
        candidates = [row for row in rows if row["stream"] == stream and row["method"] == method]
        selected_id = selection.get(f"{stream}:{method}")
        selected = next((row for row in candidates if row["id"] == selected_id), None)
        exploratory = {"all_noisy_mse_best": best_row(candidates, "all_mse")}
        if task == "sparse":
            exploratory.update(final_exact_clean_risk_best=best_row(candidates, "exact_mse"),
                               second_half_clean_mse_best=best_row(candidates, "second_half_clean_mse"))
        else:
            exploratory["second_half_noisy_mse_best"] = best_row(candidates, "second_half_mse")
        # Null refusal/power is judged at the hyperparameters selected on its
        # signal/real stream, not an independently selected near-zero null LR.
        signal_stream = 0 if task == "sparse" else stream - stream % 5
        signal_id = selection.get(f"{signal_stream}:{method}")
        signal_selected = next((row for row in rows if row["id"] == signal_id), None)
        output.append({"stream": stream, "method": method,
                       "first_half_observed_mse_selected": selected,
                       "signal_selected_config_id": signal_id,
                       "matched_signal_selected_config": matched_config(rows, signal_selected, stream),
                       "null_interpretation": "Use matched_signal_selected_config for refusal/power; independent null selection is not refusal evidence.",
                       "exploratory_retrospective_envelopes": exploratory})
    return output


def log_result(writer, result, step):
    for key in ("reported_epoch_seconds", "compile_capture_seconds", "pass_wall_seconds_including_reporting",
                "observations_per_second", "config_observations_per_second", "finite_state_config_count"):
        writer.add_scalar(f"runtime/{key}", result[key], step)
    for row in result["configs"]:
        tag = f"{row['method']}/stream_{row['stream']}/config_{row['id']}"
        for key, value in row.items():
            if key not in ("id", "stream") and isinstance(value, (int, float, bool)) and math.isfinite(value):
                writer.add_scalar(f"{tag}/{key}", value, step)
            elif isinstance(value, dict):
                for name, scalar in value.items():
                    if scalar is not None and math.isfinite(scalar):
                        writer.add_scalar(f"{tag}/{key}/{name}", scalar, step)
    writer.add_text("selected_comparisons", json.dumps(json_safe(result["comparisons"])), step)
    writer.flush()


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    start_epoch = time.time()
    if args.seed != 1:
        raise ValueError("This protocol uses seed 1 only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU or eager fallback")
    if args.log_every is None:
        args.log_every = 10000 if args.task == "sparse" else 100000
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if not 1 <= args.signal_inputs <= 4096:
        raise ValueError("--signal-inputs must lie in [1, 4096]")
    if args.switch_at and not 0 < args.switch_at < 1:
        raise ValueError("--switch-at must lie in (0, 1)")
    if args.switch_back and not 0 < args.switch_at < args.switch_back < 1:
        raise ValueError("--switch-back requires an earlier --switch-at")
    if args.switch_at and not 0 <= args.switch_to <= 4096 - args.signal_inputs:
        raise ValueError("Moved signal support must fit within 4096 inputs")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    prepared = time.perf_counter()
    features = targets = stream = None
    inject_column = None
    if args.task == "sparse":
        total = 100000 if args.switch_at else 20000
        source_args = SparseArgs(steps=total, input_dim=4096, signal_inputs=args.signal_inputs,
                                seeds=1, seed=args.seed, feature_prob=0.01,
                                target_noise_std=math.sqrt(5), spike_prob=0.01,
                                switch_at=args.switch_at, switch_back=args.switch_back,
                                switch_to=args.switch_to)
        stream = Stream(source_args, device, False)
        streams = [{"kind": "signal", "alpha": 1.0}, {"kind": "pure_noise", "alpha": 0.0}]
        dim, score_after = 4096, 0
    else:
        if args.switch_at or args.switch_back or args.signal_inputs != 1:
            raise ValueError("Synthetic signal/switch options cannot modify the stock protocol")
        source_args = StockArgs(bars=args.bars, seed=args.seed, center=True, steps=0,
                                score_after=2000, null_streams=4, inject_alpha=(0.0, 0.03, 0.1))
        bars = read_bars(source_args.bars)
        features_np, target_np = build_stream(bars, source_args)
        rng = np.random.default_rng(args.seed)
        orders = [rng.permutation(len(target_np)) for _ in range(4)]
        target_rows, inject_column = build_targets(target_np, features_np, source_args, orders)
        features = torch.cat((torch.as_tensor(features_np, device=device),
                              torch.ones((len(target_np), 1), device=device)), dim=1)
        targets = torch.as_tensor(target_rows.T.copy(), device=device)
        streams = [{"alpha": alpha, "kind": "real" if kind == 0 else f"shuffled_{kind}"}
                   for alpha in source_args.inject_alpha for kind in range(5)]
        total, dim, score_after = len(target_np), features.shape[1], source_args.score_after
        if total // 2 <= score_after:
            raise ValueError("Stock file has no scored first half after the 2000-sample warmup")
        del bars, features_np, target_np, orders, target_rows
    data_seconds = time.perf_counter() - prepared
    configs, groups = configurations(streams)
    stamp = f"{time.time():.6f}"
    directory = Path(args.output_dir) / f"{args.task}__predictive_writer_v6__{args.seed}__{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    source_files = ("predictive_writer_benchmark_v6.py", "predictive_writer_v6.py",
                    "noisy_stream_diagnostic.py", "stock_stream.py")
    metadata = {"arguments": vars(args), "source_arguments": asdict(source_args), "seed": args.seed,
                "started_epoch_seconds": start_epoch, "run_stamp": stamp,
                "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                "dimensions_including_intercept": dim, "total_observations": total,
                "score_after": score_after, "midpoint": total // 2,
                "stock_injected_feature_column": inject_column,
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in source_files},
                "data_preparation_seconds": data_seconds, "configs": configs, "chunk_steps": CHUNK,
                "run_label": "predictive_writer_v6",
                "optimizer_interface": "Single-example linear half-square step(weight,grad,h=x*x). Rank-one curvature comes from sign(g)*sqrt(h); not a general NN diagonal-Hessian API. No clean targets or support enter the optimizer.",
                "readout_rule": "Fast memory forecasts coordinate signs; slow memory forecasts a row-wide RMS-normalized direction. Both are scored on later partial targets before updating memory. Readout amplitudes A/B are fitted from those later outcomes, nonnegative and not capped at1. A>sqrt(2*V*log(2*(D+1))) admits the direction. Only admitted or previously written coordinates enter the observed-Jacobian unit-candidate write.",
                "coordinate_horizon": "128 participating observations, fixed before this run; memory, A and B decay by exp(-1/128), while V decays by its square. This changes evidence lifetime, not the amplitude of an admitted write.",
                "block_horizon": "Lifetime direction and evidence; RMS normalization keeps early raw forecast magnitude out of calibration units.",
                "control_semantics": {"unit": "Write the full projected predictive target; half-boundary retirement", "raw_line": "Same memory/admission but raw noisy-loss line search can veto or truncate the write", "zero_retirement": "Unit writes, retain admitted coordinates for any positive covariance", "recheck": "Unit writes, require full discovery evidence at all times", "coordinate": "Unit writes from coordinate evidence only", "block": "Unit writes from block evidence only", "shuffled": "Unit writes with wrong-coordinate validation forecasts"},
                "admission_retirement": "Discovery uses full evidence boundary; retirement uses half, except explicit controls. This is a prespecified hysteresis policy, not a false-discovery guarantee.",
                "writer_objective": "Match the validated candidate prediction, NOT today's noisy observed label. Unit writes may intentionally increase current realized loss. Current gradients still train memories and prequential validation; no clean labels enter the optimizer.",
                "controller_grid": "None: fixed unit-candidate mechanism, fixed evidence lifetime and dimension-based admission boundary; no LR/threshold/horizon sweep.",
                "statistical_caution": "Admission is a diagnostic heuristic, not an exact anytime test under repeated tests, dependent data and changing forecasters. allocation_mean now means fraction admitted, not fitted amplitude.",
                "change_metrics": "Actual per-update parameter energy and peak change, by signal regime. Synthetic large useful signal changes exceed .1 coefficient units and reduce that signal coordinate squared error. Noise changes use nonsignal coordinates, including every coordinate in the matched pure-noise stream.",
                "primary_objective": "High-gain batch-one signal-selective learning plus original stock targets alpha0; conservative tiny-LR MSE improvements alone are not success.",
                "selection_protocol": "Each method/stream selected by first-half observed noisy MSE; JSON frozen before second-half targets; updates continue with unchanged grids.",
                "null_protocol": "Primary null power uses exactly the configuration selected on the corresponding signal/real stream, never an independently tuned null winner.",
                "limitations": ["One seed and linear models; no universal optimizer-superiority claim.",
                    "Coordinate residual regressions interact through live residuals; no convergence theorem or neural-network superiority established.",
                    "Readout amplitude is nonnegative, with no unit cap. Fast evidence can reopen; slow block evidence may remain stale. Full regime and null tests decide whether this works.",
                    "Repeated research on this stream is exploratory, not an untouched confirmatory holdout.",
                    "Synthetic sparse/dense targets are uncentered; dense signal has a large predictable mean.",
                    "New mechanisms have one fixed configuration each; SGD/AdamW retain fifteen-rate baseline envelopes.",
                    "Switch midpoint selection precedes the moved-support regime; deliberate distribution shift is not a stationary holdout.",
                    "Retrospective envelopes inspect evaluation metrics and are exploratory, not selected-method performance.",
                    "Whole-file stock permutations destroy serial dependence and are not trading backtests; no clean market target exists.",
                    "All controls run to completion without culling. Divergence is reported, not hidden.",
                    "Baseline task-specific epsilon/decay ordering matches sources; stock intercept is not decayed."]}
    writer = SummaryWriter(str(directory))
    writer.add_text("protocol", json.dumps(metadata, indent=2))
    print(f"run={directory} observations={total} dim={dim} configs={len(configs)} CUDA graph={CHUNK}", flush=True)
    runner = Runner(args, dim, streams, total, score_after, configs, groups)
    selection = {}
    try:
        runner.capture()
        print(f"compile/capture={runner.compile_seconds:.3f}s; mutable state restored={runner.capture_restored}", flush=True)
        started = time.perf_counter()
        observed, next_log, midpoint = 0, args.log_every, total // 2
        boundaries = {midpoint, total}
        segment_ends = {}
        if args.task == "sparse":
            ends = ([int(total * args.switch_at)] if args.switch_at else [])
            if args.switch_back:
                ends.append(int(total * args.switch_back))
            ends.append(total)
            segment_ends = dict(zip(ends, SEGMENTS))
            boundaries.update(ends)
        # Initial metadata also records actual initialized finite/controller state.
        initial = runner.results(0, max(time.perf_counter() - started, 1e-12),
                                 stream.initial if stream is not None else None)
        initial["comparisons"] = comparisons(initial, args.task, selection)
        save_json(directory / "metadata.json", {"metadata": metadata, **initial})
        log_result(writer, initial, 0)
        result = initial
        while observed < total:
            boundary = min(value for value in boundaries if value > observed)
            length = min(CHUNK, boundary - observed)
            if args.task == "sparse":
                assert stream is not None
                x, noisy, clean, _, _ = stream.draw(length, observed + 1)
                clean_rows = torch.cat((clean, torch.zeros_like(clean)), dim=1)
                target_rows = torch.cat((noisy, noisy - clean), dim=1)
                runner.advance(x[:, 0, :], target_rows, clean_rows)
            else:
                assert features is not None and targets is not None
                runner.advance(features[observed:observed + length], targets[observed:observed + length])
            observed += length
            if observed >= next_log or observed in boundaries:
                torch.cuda.synchronize()
                final_support = (stream.support(torch.tensor([observed], device=device))[0]
                                 if stream is not None else None)
                runner.record_interval(observed)
                result = runner.results(observed, time.perf_counter() - started, final_support)
                if observed in segment_ends:
                    runner.segment_endpoint_risks[segment_ends[observed]] = {
                        "observations": observed,
                        "config_exact_clean_risk": {str(row["id"]): row["exact_mse"] for row in result["configs"]}}
                    result["segment_endpoint_exact_clean_risk"] = dict(runner.segment_endpoint_risks)
                if observed == midpoint:
                    for target_index in range(len(streams)):
                        for method in groups:
                            candidates = [row for row in result["configs"]
                                          if row["stream"] == target_index and row["method"] == method]
                            best = best_row(candidates, "first_half_mse")
                            selection[f"{target_index}:{method}"] = None if best is None else best["id"]
                    save_json(directory / "frozen_first_half_selection.json",
                              {"metadata": metadata, "frozen_epoch_seconds": time.time(),
                               "selection": selection, **result})
                result["frozen_selection"] = selection.copy()
                result["comparisons"] = comparisons(result, args.task, selection)
                save_json(directory / "partial.json", {"metadata": metadata, **result})
                log_result(writer, result, observed)
                print(f"progress {observed}/{total} elapsed={result['pass_wall_seconds_including_reporting']:.1f}s "
                      f"finite={result['finite_state_config_count']}/{len(configs)}", flush=True)
                next_log = (observed // args.log_every + 1) * args.log_every
        result["finished_epoch_seconds"] = time.time()
        save_json(directory / "results.json", {"metadata": metadata, **result})
        # Save learned parameters and useful controller continuation state only;
        # capture buffers, metric accumulators and input blocks are not checkpoints.
        torch.save({"weight": runner.weight.cpu(), "configs": configs,
                    "controllers": {control: {name: value.cpu() for name, value in buffers.items()}
                                    for control, buffers in runner.controller_buffers.items()},
                    "adam_m": runner.m.cpu(), "adam_v": runner.v.cpu(), "observations": total,
                    "frozen_selection": selection, "metadata": metadata},
                   directory / "final_state.pt")
        writer.add_scalar("runtime/finished_epoch_seconds", result["finished_epoch_seconds"], total)
        print(f"saved {directory / 'results.json'}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

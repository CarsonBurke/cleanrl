"""Sequential CUDA-only Predictive Need v4 benchmark.

Candidates receive ordinary gradients, analytic prediction Jacobians, bounded
presynaptic features, direct observed targets, and actual pre-update predictions.
Association inclusion p retains v3's stationary composite empirical Bayes; local
need uses q=(b^2+U)/(b^2+U+R1), with current prediction error location b,
conditional latent-mean uncertainty U, and H1 conditional noise variance R1.
Controls are predictive_need, signal_strength (v3's Rmix), no_epistemic,
fixed_prior, and residual_statistics. This is local inference, not a global
teacher or network-weight confidence; cumulative statistics do not solve drift.
Clean targets and support are evaluator-only. First-half observed-loss selections
are frozen before second-half evaluation; nulls use their signal stream's selected
hyperparameters. All controls complete the full horizon. No network layers are
added. Stock functionality is inherited, not qualified.
Persistent controller state costs 24 bytes/parameter plus 72 bytes/model.
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

from cleanrl.shared.runtime import configure_runtime

from cleanrl.plasticity.noisy_stream_diagnostic import Args as SparseArgs, Stream, bernoulli_power
from cleanrl.plasticity.stock_stream import Args as StockArgs, read_bars, build_stream, build_targets
from cleanrl.plasticity.predictive_need_v4 import CONTROLS, PredictiveNeedState
from cleanrl.plasticity.predictive_transport_benchmark_v2 import (
    best_row, comparisons, log_result, matched_config, named_buffers, save_json, score_metrics,
)


CHUNK = 100
LR_GRID = tuple(float(value) for value in np.logspace(-8, -1, 15))
SEGMENTS = ("initial", "moved", "returned")


@dataclass
class Args:
    task: Literal["sparse", "stock"]
    seed: Literal[1] = 1
    signal_amplitude: float = 1.0
    signal_pattern: Literal["constant", "alternating"] = "constant"
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
        self.signal_values = torch.full((dim,), args.signal_amplitude, device=device)
        if args.signal_pattern == "alternating":
            self.signal_values *= torch.where(torch.arange(dim, device=device) % 2 == 0, 1.0, -1.0)
        initial_support = torch.arange(dim, device=device) < args.signal_inputs
        moved_support = ((torch.arange(dim, device=device) >= args.switch_to) &
                         (torch.arange(dim, device=device) < args.switch_to + args.signal_inputs))
        for control in CONTROLS:
            rows = groups[control]
            state = PredictiveNeedState(self.weight[rows], control=control)
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
            feature = x.tanh()
            if not self.sparse:
                # The bias feature is an explicit intercept, not tanh(1).
                feature[-1] = 1.0
            for control, state in self.controllers.items():
                rows = groups[control]
                candidate_weight = self.previous_weight[rows].clone()
                jacobian = x.expand_as(candidate_weight)
                state.step(candidate_weight, grad[rows], jacobian,
                           feature.expand_as(candidate_weight), targets[rows], residual[rows],
                           prediction[rows])
                self.weight[rows].copy_(candidate_weight)
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
                truth = support * self.signal_values
                useful = ((self.weight - truth).square() < (self.previous_weight - truth).square()) & signal
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

        kernel = torch.compile(step, fullgraph=True, dynamic=False,
                               options={"triton.cudagraphs": False})

        def invoke(fn, x, targets, clean):
            # Keep this snapshot outside the fused mutation graph. Inductor
            # otherwise schedules the first controller's snapshot after the
            # model write, even through clone(), corrupting the next example.
            self.previous_weight.copy_(self.weight)
            fn(x, targets, clean)

        self.compiled = lambda x, targets, clean: invoke(kernel, x, targets, clean)
        self.eager_step = lambda x, targets, clean: invoke(step, x, targets, clean)

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
            counts_by_model = state.observation_count.detach().cpu().tolist()
            for label, accumulator in (
                ("mean_effective_gain", state.gain_sum),
                ("mean_participation_probability", state.participation_sum),
                ("mean_prior_probability", state.prior_probability_sum),
                ("mean_log_bayes_factor", state.log_bayes_factor_sum),
            ):
                statistics[label] = [value / count if count > 0 else None
                                     for value, count in zip(accumulator.detach().cpu().tolist(),
                                                             counts_by_model)]
            statistics["mean_unexplained_target_variance"] = [
                value / count if count > 0 else None
                for value, count in zip(state.noise_variance_sum.detach().cpu().tolist(),
                                        state.noise_observation_count.detach().cpu().tolist())]
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
                truth = support * self.signal_values
                signal_error = (w - truth) * support
                junk = w * (1 - support)
                variance = (truth.double().square() * probabilities.double() * (1 - probabilities.double())).sum()
                risk = bernoulli_power(w - truth, probabilities)
                measured = {"exact_mse": risk,
                            "exact_trivial": bernoulli_power(truth, probabilities).expand(len(indices)),
                            "exact_mean_predictor_mse": variance.expand(len(indices)),
                            "signal_reconstruction_mse": bernoulli_power(signal_error, probabilities),
                            "distractor_leakage_mse": bernoulli_power(junk, probabilities),
                            "signal_distractor_cross": 2 * (signal_error.double() * probabilities.double()).sum(-1) * (junk.double() * probabilities.double()).sum(-1),
                            "clean_R2": 1 - risk / variance if stream == 0 else torch.full_like(risk, float("nan"))}
                count_signal = support.sum().clamp_min(1)
                count_junk = (1 - support).sum().clamp_min(1)
                measured["signal_coefficient_mean"] = (w * support).sum(1) / count_signal
                measured["signal_coefficient_rmse"] = ((w - truth).square() * support).sum(1).div(count_signal).sqrt()
                measured["distractor_coefficient_rms"] = (w.square() * (1 - support)).sum(1).div(count_junk).sqrt()
                measured["coefficient_l2_error"] = (w - truth).square().sum(1).sqrt()
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
                "completed_full_horizon": observed == self.total,
                "reported_epoch_seconds": time.time(), "stock_intervals": list(self.stock_intervals),
                "support_segment_observation_counts": segment_counts if self.sparse else None,
                "segment_endpoint_exact_clean_risk": dict(self.segment_endpoint_risks),
                "phase_scored_counts": counts, "compile_capture_seconds": self.compile_seconds,
                "capture_mutable_state_restored": self.capture_restored,
                "pass_wall_seconds_including_reporting": elapsed,
                "observations_per_second": observed / elapsed,
                "config_observations_per_second": observed * len(rows) / elapsed,
                "finite_state_config_count": int(finite_state.sum()), "configs": rows}


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    start_epoch = time.time()
    if args.seed != 1:
        raise ValueError("This protocol uses seed 1 only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU or eager fallback")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    if args.log_every is None:
        args.log_every = 10000 if args.task == "sparse" else 100000
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if not math.isfinite(args.signal_amplitude) or args.signal_amplitude == 0:
        raise ValueError("--signal-amplitude must be finite and nonzero")
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
        if args.switch_at or args.switch_back or args.signal_inputs != 1 or args.signal_amplitude != 1 or args.signal_pattern != "constant":
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
    directory = Path(args.output_dir) / f"{args.task}__predictive_need_v4__{args.seed}__{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    source_files = ("predictive_need_benchmark_v4.py", "predictive_need_v4.py",
                    "predictive_transport_benchmark_v2.py", "noisy_stream_diagnostic.py", "stock_stream.py")
    metadata = {"arguments": vars(args), "source_arguments": asdict(source_args), "seed": args.seed,
                "started_epoch_seconds": start_epoch, "run_stamp": stamp,
                "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                "dimensions_including_intercept": dim, "total_observations": total,
                "score_after": score_after, "midpoint": total // 2,
                "stock_injected_feature_column": inject_column,
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in source_files},
                "data_preparation_seconds": data_seconds, "configs": configs, "chunk_steps": CHUNK,
                "run_label": "predictive_need_v4",
                "optimizer_interface": "step(weight, gradient, jacobian, feature, target, residual, prediction). Gradient is residual times Jacobian. Observed targets and actual pre-update network predictions are passed directly, never reconstructed from target plus residual. No clean labels, support, hidden targets, Hessian, paired gradients, or extra network layers.",
                "evidence_model": "Unchanged v3 Welford moments and stationary centered univariate input-target association: H0 has an unknown constant target mean; H1 adds a slope on the bounded presynaptic feature. Common flat intercept and Jeffreys scale prior; Zellner slope g-prior with g=n. Local Gaussian regression and stationary observations underlie the conditional interpretation. Shared composite empirical-Bayes inclusion p describes association, not network weights. All statistics and gains precede the current observed target.",
                "feature": "One target-free association feature: tanh(presynaptic activity), with stock bias feature exactly one. The actual pre-update prediction enters need, not association moments. All examples update evidence, including feature-zero examples; zero Jacobians still make zero writes. Observed targets remain direct.",
                "local_moments": "Let n=count, Mxx and Myy be centered Welford sums of squares, Cxy their cross sum, and tau=n/(n+1). For identified associations (n>=2, Mxx>0, Myy>0): r2=Cxy^2/(Mxx*Myy), slope=tau*Cxy/Mxx, effect=slope*(feature-mean_x), location=mean_y+effect. No network weight is inferred or replaced.",
                "conditional_variance": "R0=Myy/(n-3), R1=R0*(1-tau*r2). The proper variance expectation requires n>3 and Myy>0; otherwise noise is infinity and gain is zero. All need controls use H1 conditional R1. Only signal_strength retains v3's inclusion-mixture Rmix=R0*(1-p*tau*r2).",
                "mean_uncertainty": "H1 conditional latent target mean variance U=R1*(1/n+tau*(feature-mean_x)^2/Mxx), including intercept and slope leverage uncertainty, not network-weight uncertainty. Undefined mean second moments or feature contrast yield infinite mean_uncertainty and zero need fraction.",
                "plasticity": "For predictive_need, b=actual_current_prediction-location, M=b^2+U, q=M/(M+R1), only for identified and variance-ready associations. Under a correctly specified local H1, let latent current error B have E[B]=b and Var(B)=U, with zero-mean observation noise epsilon uncorrelated with B and E[epsilon^2]=R1. Then q minimizes E[(B-q*(B-epsilon))^2]; this is not global optimality for multiple, interacting, or correlated signals. With D=1+sum_j p_j*Jacobian_j^2, gain_i=p_i*q_i/D and delta_i=-gain_i*ordinary_gradient_i. The linearized output gain cannot exceed max(q). Ordinary network weights remain the predictor; inferred locations never replace the target or gradient direction.",
                "control_semantics": {
                    "predictive_need": "Full pre-observation association evidence, shared composite empirical-Bayes inclusion p, and current prediction-need fraction (b^2+U)/(b^2+U+R1)",
                    "signal_strength": "Exact v3 evidence control: q=effect^2/(effect^2+Rmix), with Rmix=R0*(1-p*tau*r2); retain the same inclusion p and Jacobian geometry",
                    "no_epistemic": "Exclude U from need: q=b^2/(b^2+R1); retain association evidence, empirical-Bayes inclusion, and conditional R1",
                    "fixed_prior": "Fix inclusion prior at one half, so p=sigmoid(logBF); otherwise retain full current prediction need with U and R1",
                    "residual_statistics": "Observe the learner's direct residual instead of target in sufficient-statistic updates. Its location estimates error, not target, so b=location, NOT prediction-location; otherwise retain full need with U and R1"},
                "initialization": "Zero cumulative FP64 sufficient statistics and diagnostics defined in predictive_need_v4.py. No candidate learning-rate argument, reference scale, or candidate sweep; network weights retain original zero initialization.",
                "controller_grid": "No LR grid for candidates; five causal controls. Baselines retain fifteen rates. No retrospective selection across controls.",
                "prediction_registry": "runs/predictive_need_v4_preregistration.json",
                "prediction_registry_sha256": hashlib.sha256(Path("runs/predictive_need_v4_preregistration.json").read_bytes()).hexdigest(),
                "cost": "Three FP64 sufficient-statistic scalars per parameter (24 bytes/P) plus nine FP64 statistics/diagnostic scalars per model (72 bytes/model). Network weights remain FP32. One analytic prediction/Jacobian evaluation per observation. No covariance matrix, free log step, gradient accumulation, model averaging, or replay.",
                "evidence_metric_caution": "Effective gain, participation, and log Bayes factor average over active Jacobians per example, with zero for no active parameters, then divide by observation_count. Prior probability averages per example. Mean unexplained target variance uses H1 conditional R1 except signal_strength's Rmix, averages over active parameters only when variance-ready (n>3 and Myy>0), and divides by noise_observation_count. Residual-statistics variance refers to observed errors instead of targets. Empty divisors report null. No persistent uncertainty counter is added. Variance includes other signal and is not known irreducible task noise. These diagnostics are not evidence of model-learning success by themselves.",
                "change_metrics": "Actual per-update parameter energy and peak change, by signal regime. Synthetic large useful signal changes exceed .1 coefficient units and reduce that signal coordinate squared error. Noise changes use nonsignal coordinates, including every coordinate in the matched pure-noise stream.",
                "primary_objective": "First qualify high-gain batch-one current-error-sensitive, signal-selective learning over the full 20000-observation stationary sparse signal versus matched pure-noise horizon. Conservative tiny-LR MSE improvements alone are not success. Only after stationary qualification consider the full 100000-observation support-switch experiment; cumulative statistics remain stationary. Stock execution is inherited functionality, not a qualification claim.",
                "selection_protocol": "Each method/stream selected by first-half observed noisy MSE; JSON frozen before second-half targets; updates continue with unchanged grids.",
                "null_protocol": "Primary null power uses exactly the configuration selected on the corresponding signal/real stream, never an independently tuned null winner.",
                "limitations": ["One seed; repeated development streams are not an untouched confirmatory holdout.",
                    "Requires scalar-output prediction Jacobians and local activations, not a standard gradient-only optimizer.",
                    "Stationary univariate association only: parity, conditional interactions, drift, nonlinear networks, RL, and LLMs are not qualified.",
                    "The shared inclusion-rate likelihood is composite empirical Bayes under dependent feature evidence, not a jointly exact posterior or an FDR guarantee.",
                    "Inclusion describes repeatable local input-target association, not confidence in a network weight. U is conditional uncertainty of a latent local target mean, not network-weight uncertainty.",
                    "A local univariate predictor is not a calibrated global teacher. The need objective requires correctly specified local H1 assumptions and is not exact global optimality for correlated, interacting, or multiple signals. Conditional unexplained variance includes other signal and can absorb misspecification; no target denoising or replacement gradient direction is used.",
                    "The unit-information g=n prior changes with n: this is not a fixed-prior sequential posterior or a parameter-free claim.",
                    "Constant features cannot earn association evidence, including the output bias.",
                    "Cumulative stationary sufficient statistics can be stale under drift. The linearized gain bound does not guarantee arbitrary nonlinear finite-step safety or parameterization invariance.",
                    "Synthetic clean labels and support appear only in evaluation. Stock has no clean target.",
                    "Baselines retain original epsilon/decay ordering and first-half LR selection.",
                    "All controls run full horizons; divergence is recorded, not hidden."]}
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
                x, noisy, clean, _, support_rows = stream.draw(length, observed + 1)
                if args.signal_amplitude != 1 or args.signal_pattern != "constant":
                    noise = noisy - clean
                    clean = (x[:, 0, :] * support_rows * runner.signal_values).sum(1, keepdim=True)
                    noisy = clean + noise
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

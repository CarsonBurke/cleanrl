"""Full-covariance v2 linear/quadratic benchmark; no estimator sees clean labels.

Queue the module through mlq, with --task sparse or --task stock. Sparse sweeps
are exploratory. Stock freezes each method's hyperparameter choice at the
chronological midpoint; its online weights continue to learn in the second half.
This compares a full-covariance estimator on the v1 data/scoring protocol, not a
general nonlinear optimizer or Muon test. Geometry, the noise estimator, and
the inclusion approximation change together: this is not a one-factor ablation.
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
from cleanrl.plasticity.predictive_covariance_optimizer_v2 import CovarianceState


CHUNK = 100
LR_GRID = tuple(float(x) for x in np.logspace(-8, -1, 15))
PRIOR_SCALES = (0.01, 0.1, 1.0, 10.0)


@dataclass
class Args:
    task: Literal["sparse", "stock"]
    seed: Literal[1] = 1
    steps: int | None = None
    """Sparse only: defaults to 20000, or 100000 with support switching."""
    input_dim: int = 4096
    signal_inputs: int = 1
    feature_prob: float = 0.01
    noise_variance: float = 5.0
    spike_prob: float = 0.01
    switch_support: bool = False
    switch_at: float = 0.0
    """Fraction at which support moves; nonzero enables switching."""
    switch_back: float = 0.0
    switch_to: int = 2
    bars: str = StockArgs().bars
    output_dir: str = "runs"
    log_every: int = 10000


def arguments():
    return tyro.cli(Args)


def configurations(streams, dim, sparse):
    """Contiguous method slices; transport rows are stream-major for mixing."""
    configs, groups = [], {}
    densities = tuple(dict.fromkeys((1.0 / dim, 0.01, 0.1, 1.0)))
    methods = ("transport", "sgd", "adamw") + (("oracle_sgd", "oracle_adam") if sparse else ())
    for method in methods:
        start = len(configs)
        for stream, target in enumerate(streams):
            settings = ([{"prior_scale": scale, "prior_density": density}
                         for scale in PRIOR_SCALES for density in densities]
                        if method == "transport" else [{"lr": lr} for lr in LR_GRID])
            for setting in settings:
                configs.append({"id": len(configs), "method": method, "stream": stream,
                                **target, **setting, "reference_only": method.startswith("oracle")})
        groups[method] = slice(start, len(configs))
    return configs, groups, len(PRIOR_SCALES) * len(densities)


def tensor_buffers(state):
    buffers = state.buffers()
    return list(buffers.values()) if isinstance(buffers, dict) else list(buffers)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Runner:
    def __init__(self, args, dim, streams, total, score_after, configs, groups, candidates):
        self.args, self.dim, self.streams, self.total = args, dim, streams, total
        self.score_after, self.configs, self.groups = score_after, configs, groups
        self.candidates = candidates
        self.sparse = args.task == "sparse"
        self.device = torch.device("cuda")
        n, s = len(configs), len(streams)
        self.weight = torch.zeros((n, dim), device=self.device)
        self.m = torch.zeros_like(self.weight)
        self.v = torch.zeros_like(self.weight)
        self.step_count = torch.zeros((), device=self.device)
        self.count = torch.zeros(2, dtype=torch.float64, device=self.device)
        self.error = torch.zeros((2, n), dtype=torch.float64, device=self.device)
        self.trivial = torch.zeros_like(self.error)
        self.prediction_power = torch.zeros_like(self.error)
        self.clean_error = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.clean_segment_error = torch.zeros((3, n), dtype=torch.float64, device=self.device)
        self.clean_segment_count = torch.zeros(3, dtype=torch.float64, device=self.device)
        self.ever_nonfinite = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.first_nonfinite = torch.full((n,), -1.0, device=self.device)
        self.mix_error = torch.zeros((2, s), dtype=torch.float64, device=self.device)
        self.mix_trivial = torch.zeros_like(self.mix_error)
        self.mix_power = torch.zeros_like(self.mix_error)
        self.mix_clean_error = torch.zeros(s, dtype=torch.float64, device=self.device)
        self.mix_clean_segment_error = torch.zeros((3, s), dtype=torch.float64, device=self.device)
        self.mix_log_weight = torch.zeros((s, candidates), dtype=torch.float64, device=self.device)
        self.mix_target_square = torch.ones(s, dtype=torch.float64, device=self.device)
        self.mix_count = torch.ones((), device=self.device)
        self.mix_failed = torch.zeros(s, dtype=torch.int64, device=self.device)
        self.mix_excluded = torch.zeros(s, dtype=torch.int64, device=self.device)
        self.stream_index = torch.tensor([c["stream"] for c in configs], device=self.device)
        self.lr = torch.tensor([c.get("lr", 0.0) for c in configs], device=self.device)[:, None]
        transport = groups["transport"]
        scales = torch.tensor([c["prior_scale"] for c in configs[transport]], device=self.device)[:, None]
        density = torch.tensor([c["prior_density"] for c in configs[transport]], device=self.device)[:, None]
        self.transport = CovarianceState(self.weight[transport], prior_scale=scales,
                                        prior_density=density, memory=0.0)
        self.transport_buffers = tensor_buffers(self.transport)
        self.states = [self.weight, self.m, self.v, self.step_count, self.count,
                       self.error, self.trivial, self.prediction_power, self.clean_error,
                       self.ever_nonfinite, self.first_nonfinite, self.mix_error,
                       self.mix_trivial, self.mix_power, self.mix_clean_error,
                       self.mix_log_weight, self.mix_target_square, self.mix_count,
                       self.mix_failed, self.mix_excluded, self.clean_segment_error,
                       self.clean_segment_count, self.mix_clean_segment_error, *self.transport_buffers]
        self.inputs = (torch.zeros((CHUNK, dim), device=self.device),
                       torch.zeros((CHUNK, s), device=self.device),
                       torch.zeros((CHUNK, s), device=self.device),
                       torch.zeros((CHUNK, dim), device=self.device))
        # Source uses eps=1e-5 on the sparse diagnostic and 1e-8 on stocks.
        epsilon = 1e-5 if self.sparse else 1e-8
        midpoint = total // 2
        weight_decay = SparseArgs().weight_decay if self.sparse else StockArgs().weight_decay
        decay_coordinates = torch.ones((1, dim), device=self.device)
        if not self.sparse:
            decay_coordinates[:, -1] = 0  # original stock bias is not decayed

        def step(x, target_rows, clean_rows, support):
            t = self.step_count
            # Only explicitly labelled oracle references inspect support. Clear stale
            # coefficients/moments before predicting, as in the source diagnostic.
            if self.sparse:
                for method in ("oracle_sgd", "oracle_adam"):
                    rows = groups[method]
                    mask = support[None, :] * (self.stream_index[rows] == 0)[:, None]
                    self.weight[rows].mul_(mask)
                    self.m[rows].mul_(mask)
                    self.v[rows].mul_(mask)
            prediction = (self.weight * x).sum(1)
            targets = target_rows[self.stream_index]
            residual = prediction - targets
            bad = ~torch.isfinite(prediction)
            self.first_nonfinite.copy_(torch.where(bad & ~self.ever_nonfinite, t + 1,
                                                   self.first_nonfinite))
            self.ever_nonfinite.logical_or_(bad)
            scored = t >= score_after
            phase = torch.stack((scored & (t < midpoint), scored & (t >= midpoint))).double()
            self.count.add_(phase)
            # where, not 0*NaN: divergence after midpoint must not contaminate
            # first-half scores and retrospectively change the frozen selection.
            squared = residual.double().square()
            self.error.add_(torch.where(phase[:, None].bool(), squared[None, :], 0.0))
            self.trivial.add_(phase[:, None] * targets.double().square()[None, :])
            self.prediction_power.add_(torch.where(phase[:, None].bool(),
                                                   prediction.double().square()[None, :], 0.0))
            if self.sparse:
                clean_squared = (prediction.double() - clean_rows[self.stream_index]).square()
                self.clean_error.add_(clean_squared)
                switch = int(total * args.switch_at) if args.switch_support else total
                back = int(total * args.switch_back) if args.switch_back else total
                segment = torch.stack((t < switch, (t >= switch) & (t < back), t >= back))
                self.clean_segment_count.add_(segment.double())
                self.clean_segment_error.add_(torch.where(segment[:, None], clean_squared[None, :], 0.0))

            # Prediction-only prequential aggregation. Weights depend on previous
            # observed noisy losses, never on clean labels or the current target.
            # The Gaussian likelihood scale is the past target second moment;
            # this is a causal heuristic, not a calibrated posterior/regret claim.
            candidates_prediction = prediction[transport].reshape(s, candidates)
            valid = torch.isfinite(candidates_prediction) & torch.isfinite(self.mix_log_weight)
            masked_log = torch.where(valid, self.mix_log_weight, -torch.inf)
            any_valid = valid.any(1)
            safe_log = torch.where(any_valid[:, None], masked_log, torch.zeros_like(masked_log))
            probability = safe_log.softmax(1)
            safe_prediction = torch.where(valid, candidates_prediction, 0.0).double()
            mix_prediction = torch.where(any_valid, (probability * safe_prediction).sum(1), torch.nan)
            self.mix_failed.add_((~any_valid).long())
            self.mix_excluded.add_((~valid).sum(1))
            mix_residual = mix_prediction - target_rows
            self.mix_error.add_(torch.where(phase[:, None].bool(),
                                             mix_residual.double().square()[None, :], 0.0))
            self.mix_trivial.add_(phase[:, None] * target_rows.double().square()[None, :])
            self.mix_power.add_(torch.where(phase[:, None].bool(),
                                             mix_prediction.double().square()[None, :], 0.0))
            if self.sparse:
                self.mix_clean_error.add_((mix_prediction.double() - clean_rows).square())
                self.mix_clean_segment_error.add_(torch.where(
                    segment[:, None], (mix_prediction.double() - clean_rows).square()[None, :], 0.0))
            variance = self.mix_target_square / self.mix_count
            # The common y² cancels in normalized likelihood weights. Removing
            # it, and accumulating in FP64, preserves weak excess-loss evidence.
            excess_loss = safe_prediction.square() - 2.0 * target_rows.double()[:, None] * safe_prediction
            log_next = masked_log - 0.5 * excess_loss / variance[:, None]
            maximum = log_next.max(1, keepdim=True).values
            self.mix_log_weight.copy_(torch.where(any_valid[:, None],
                                                   log_next - maximum, masked_log))
            self.mix_target_square.add_(target_rows.double().square())
            self.mix_count.add_(1)

            self.step_count.add_(1)
            grad = residual[:, None] * x
            # Only observed features/noisy targets reach the covariance estimator.
            self.transport.step(self.weight[transport], grad[transport],
                                x.square()[None, :], residual[transport, None].square(),
                                features=x, targets=targets[transport])
            for method, rows in groups.items():
                if method == "transport":
                    continue
                g = grad[rows]
                if method.startswith("oracle"):
                    g = g * support[None, :] * (self.stream_index[rows] == 0)[:, None]
                if method in ("adamw", "oracle_adam"):
                    self.m[rows].mul_(0.9).add_(g, alpha=0.1)
                    self.v[rows].mul_(0.999).addcmul_(g, g, value=0.001)
                    update = ((self.m[rows] / (1.0 - 0.9 ** self.step_count)) /
                              ((self.v[rows] / (1.0 - 0.999 ** self.step_count)).sqrt() + epsilon))
                else:
                    update = g
                if method == "adamw" and self.sparse:
                    self.weight[rows].mul_(1.0 - self.lr[rows] * weight_decay)
                self.weight[rows].sub_(self.lr[rows] * update)
                if method == "adamw" and not self.sparse:
                    # Stock source shrinks after updating; sparse source before.
                    self.weight[rows].sub_(self.lr[rows] * weight_decay *
                                           self.weight[rows] * decay_coordinates)

        self.compiled = torch.compile(step, fullgraph=True, dynamic=False,
                                      options={"triton.cudagraphs": False})

    @torch.no_grad()
    def capture(self):
        snapshots = [state.clone() for state in self.states]
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
            for state, saved in zip(self.states, snapshots):
                state.copy_(saved)
            torch.cuda.synchronize()
        self.compile_seconds = time.perf_counter() - started

    @torch.no_grad()
    def advance(self, x, targets, clean=None, support=None):
        length = x.shape[0]
        self.inputs[0][:length].copy_(x)
        self.inputs[1][:length].copy_(targets)
        if clean is not None:
            self.inputs[2][:length].copy_(clean)
        if support is not None:
            self.inputs[3][:length].copy_(support)
        if length == CHUNK:
            self.graph.replay()
        else:
            for offset in range(length):
                self.compiled(*(value[offset] for value in self.inputs))

    @torch.no_grad()
    def results(self, observed, elapsed, final_support=None):
        counts = self.count.cpu().tolist()
        error, trivial, power = (tensor.cpu().numpy() for tensor in
                                  (self.error, self.trivial, self.prediction_power))
        weights = self.weight.cpu().numpy()
        finite_weights = np.isfinite(weights).all(axis=1)
        finite_state = finite_weights & np.isfinite(self.m.cpu().numpy()).all(axis=1) & np.isfinite(self.v.cpu().numpy()).all(axis=1)
        transport_slice = self.groups["transport"]
        # Full covariance buffers are shared by unique prior scale, not by
        # configuration. Conservatively require every shared/internal buffer to
        # be finite instead of interpreting its leading dimension as configs.
        for value in self.transport_buffers:
            finite_state[transport_slice] &= bool(torch.isfinite(value).all().item())
        ever_bad = self.ever_nonfinite.cpu().tolist()
        first_bad = self.first_nonfinite.cpu().tolist()
        clean_error = self.clean_error.cpu().numpy()
        segment_counts = self.clean_segment_count.cpu().tolist()
        segment_error = self.clean_segment_error.cpu().numpy()
        mix_segment_error = self.mix_clean_segment_error.cpu().numpy()
        risk_metrics = {}
        if self.sparse:
            probabilities = torch.full((self.dim,), self.args.feature_prob, device=self.device)
            for stream in range(len(self.streams)):
                indices = [i for i, cfg in enumerate(self.configs) if cfg["stream"] == stream]
                support = final_support if stream == 0 else torch.zeros_like(final_support)
                index = torch.tensor(indices, device=self.device)
                w = self.weight[index]
                measured = clean_risk(w, support, probabilities)
                count_signal = support.sum().clamp_min(1)
                count_junk = (1 - support).sum().clamp_min(1)
                measured["signal_coefficient_mean"] = (w * support).sum(1) / count_signal
                measured["signal_coefficient_rmse"] = ((w - 1).square() * support).sum(1).div(count_signal).sqrt()
                measured["distractor_coefficient_rms"] = (w.square() * (1 - support)).sum(1).div(count_junk).sqrt()
                measured["coefficient_l2_error"] = (w - support).square().sum(1).sqrt()
                if self.args.switch_support:
                    historical = torch.zeros_like(support)
                    historical[:self.args.signal_inputs] = 1
                    historical[self.args.switch_to:self.args.switch_to + self.args.signal_inputs] = 1
                    stale = historical * (1 - support) if stream == 0 else torch.zeros_like(support)
                    measured["stale_coefficient_abs_mean"] = (w.abs() * stale).sum(1) / stale.sum().clamp_min(1)
                cpu_metrics = {name: tensor.cpu().tolist() for name, tensor in measured.items()}
                for local, config_id in enumerate(indices):
                    risk_metrics[config_id] = {name: values[local] for name, values in cpu_metrics.items()}
        rows = []
        for i, config in enumerate(self.configs):
            row = {**config, "finite_weights": bool(finite_weights[i]),
                   "finite_optimizer_state": bool(finite_state[i]),
                   "finite_predictions_entire_pass": not ever_bad[i],
                   "first_nonfinite_prediction_step": int(first_bad[i]),
                   "weight_rms": float(np.sqrt(np.mean(weights[i].astype(np.float64) ** 2))),
                   "weight_l2": float(np.linalg.norm(weights[i].astype(np.float64))),
                   **risk_metrics.get(i, {})}
            row.update(score_metrics(error[:, i], trivial[:, i], power[:, i], counts))
            if self.sparse:
                row["prequential_clean_mse"] = float(clean_error[i] / observed)
                row["segment_prequential_clean_mse"] = {
                    name: float(segment_error[j, i] / segment_counts[j]) if segment_counts[j] else None
                    for j, name in enumerate(("initial", "moved", "returned"))}
            if not self.sparse:
                row["bias"] = float(weights[i, -1])
            rows.append(row)
        mix_error, mix_trivial, mix_power = (tensor.cpu().numpy() for tensor in
                                              (self.mix_error, self.mix_trivial, self.mix_power))
        mix_clean = self.mix_clean_error.cpu().numpy()
        failed, excluded = self.mix_failed.cpu().tolist(), self.mix_excluded.cpu().tolist()
        mixtures = []
        mixture_log = self.mix_log_weight.cpu()
        for stream, target in enumerate(self.streams):
            row = {"method": "causal_transport_mixture", "stream": stream, **target,
                   "all_candidates_invalid_steps": failed[stream],
                   "invalid_candidate_observations": excluded[stream],
                   "finite_predictions_entire_pass": failed[stream] == 0,
                   "final_candidate_probabilities": mixture_log[stream].softmax(0).tolist(),
                   **score_metrics(mix_error[:, stream], mix_trivial[:, stream], mix_power[:, stream], counts)}
            if self.sparse:
                row["prequential_clean_mse"] = float(mix_clean[stream] / observed)
                row["segment_prequential_clean_mse"] = {
                    name: float(mix_segment_error[j, stream] / segment_counts[j]) if segment_counts[j] else None
                    for j, name in enumerate(("initial", "moved", "returned"))}
                candidate_weights = self.weight[transport_slice].reshape(
                    len(self.streams), self.candidates, self.dim)[stream]
                valid = torch.isfinite(self.mix_log_weight[stream])
                probability = self.mix_log_weight[stream].softmax(0).float()
                mixture_weight = (probability[:, None] * torch.where(
                    valid[:, None], candidate_weights, 0.0)).sum(0, keepdim=True)
                support = final_support if stream == 0 else torch.zeros_like(final_support)
                measured = clean_risk(mixture_weight, support, probabilities)
                row.update({name: tensor.item() for name, tensor in measured.items()})
                row["signal_coefficient_mean"] = (
                    (mixture_weight * support).sum() / support.sum().clamp_min(1)).item()
            mixtures.append(row)
        return {"observations": observed, "total_observations": self.total,
                "support_segment_observation_counts": segment_counts if self.sparse else None,
                "phase_scored_counts": counts, "compile_capture_seconds": self.compile_seconds,
                "pass_wall_seconds_including_reporting": elapsed,
                "observations_per_second": observed / elapsed,
                "config_observations_per_second": observed * len(rows) / elapsed,
                "configs": rows, "mixtures": mixtures}


def score_metrics(error, trivial, power, counts):
    out = {}
    for name, indices in (("first_half", [0]), ("second_half", [1]), ("all", [0, 1])):
        count = sum(counts[i] for i in indices)
        denominator = float(sum(trivial[i] for i in indices))
        loss = float(sum(error[i] for i in indices))
        out[f"{name}_mse"] = loss / count if count else None
        out[f"{name}_trivial_mse"] = denominator / count if count else None
        out[f"{name}_mse_ratio"] = loss / denominator if denominator else None
        out[f"{name}_prediction_power"] = float(sum(power[i] for i in indices)) / count if count else None
    return out


def best_row(rows, key):
    finite = [row for row in rows if row.get(key) is not None and math.isfinite(row[key])
              and row["finite_optimizer_state"] and row["finite_predictions_entire_pass"]]
    return min(finite, key=lambda row: row[key]) if finite else None


def comparisons(result, task, selection):
    rows, output = result["configs"], []
    keys = sorted({(row["stream"], row["method"]) for row in rows})
    for stream, method in keys:
        candidates = [row for row in rows if row["stream"] == stream and row["method"] == method]
        if task == "stock":
            selected_id = selection.get(f"{stream}:{method}")
            selected = next((row for row in candidates if row["id"] == selected_id), None)
            exploratory = best_row(candidates, "all_mse")
            output.append({"stream": stream, "method": method, "selection_config_id": selected_id,
                           "selected_second_half": selected,
                           "exploratory_all_horizon_best": exploratory})
        else:
            output.append({"stream": stream, "method": method,
                           "exploratory_final_risk_best": best_row(candidates, "exact_mse"),
                           "exploratory_prequential_best": best_row(candidates, "prequential_clean_mse")})
    return output


def log_result(writer, result, step):
    writer.add_scalar("runtime/observations_per_second", result["observations_per_second"], step)
    keys = ("all_mse_ratio", "second_half_mse_ratio", "prequential_clean_mse", "exact_mse",
            "distractor_leakage_mse", "signal_coefficient_rmse", "all_prediction_power")
    for row in result["configs"] + result["mixtures"]:
        identity = str(row["id"]) if "id" in row else str(row["stream"])
        tag = f"{row['method']}/stream_{row['stream']}/config_{identity}"
        for key in keys:
            value = row.get(key)
            if value is not None and math.isfinite(value):
                writer.add_scalar(f"{tag}/{key}", value, step)
        if "finite_optimizer_state" in row:
            writer.add_scalar(f"{tag}/finite_optimizer_state", int(row["finite_optimizer_state"]), step)
    writer.flush()


def print_comparisons(result, task):
    for item in result["comparisons"]:
        if task == "stock":
            selected = item["selected_second_half"]
            exploratory = item["exploratory_all_horizon_best"]
            report = None if selected is None else selected["second_half_mse_ratio"]
            best = None if exploratory is None else exploratory["all_mse_ratio"]
            print(f"stream={item['stream']:2} {item['method']:12} frozen-choice second={report} exploratory-all={best}", flush=True)
        else:
            row = item["exploratory_final_risk_best"]
            if row is None:
                print(f"stream={item['stream']} {item['method']}: no finite risk", flush=True)
            else:
                print(f"stream={item['stream']} {item['method']:12} exploratory config={row['id']} "
                      f"risk={row['exact_mse']:.6g} prequential={row['prequential_clean_mse']:.6g} "
                      f"signal={row['signal_coefficient_mean']:.5g} leakage={row['distractor_leakage_mse']:.6g}", flush=True)
    for row in result["mixtures"]:
        key = "prequential_clean_mse" if task == "sparse" else "second_half_mse_ratio"
        print(f"stream={row['stream']:2} causal-mixture {key}={row[key]} "
              f"invalid-steps={row['all_candidates_invalid_steps']}", flush=True)


@torch.no_grad()
def main():
    args = arguments()
    if args.switch_at:
        args.switch_support = True
    elif args.switch_support:
        args.switch_at = 0.5
    if args.switch_back and not args.switch_support:
        raise ValueError("--switch-back requires --switch-at or --switch-support")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this runner has no CPU/eager fallback")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    source_args = None
    data_seconds_start = time.perf_counter()
    features = targets = stream = None
    inject_column = None
    if args.task == "sparse":
        total = args.steps if args.steps is not None else (100000 if args.switch_support else 20000)
        if total <= 0 or args.input_dim <= 0 or not 0 < args.feature_prob < 1:
            raise ValueError("positive steps/dimension and feature probability in (0,1) required")
        if args.noise_variance < 0 or not 0 <= args.spike_prob <= 1:
            raise ValueError("invalid Gaussian variance or spike probability")
        if not 1 <= args.signal_inputs <= args.input_dim:
            raise ValueError("--signal-inputs must lie in [1, input-dim]")
        if args.switch_support:
            if not 0 < args.switch_at < 1 or not 0 <= args.switch_to <= args.input_dim - args.signal_inputs:
                raise ValueError("invalid switch location or feature")
            if args.switch_back and not args.switch_at < args.switch_back < 1:
                raise ValueError("switch-back must follow switch-at and precede the end")
        source_args = SparseArgs(steps=total, input_dim=args.input_dim, seeds=1, seed=args.seed,
                                 feature_prob=args.feature_prob, target_noise_std=math.sqrt(args.noise_variance),
                                 spike_prob=args.spike_prob,
                                 signal_inputs=args.signal_inputs,
                                 switch_at=args.switch_at if args.switch_support else 0.0,
                                 switch_back=args.switch_back if args.switch_support else 0.0,
                                 switch_to=args.switch_to)
        stream = Stream(source_args, device, False)
        streams = [{"kind": "signal", "alpha": 1.0}, {"kind": "pure_noise", "alpha": 0.0}]
        dim, score_after = args.input_dim, 0
    else:
        if args.steps is not None or args.switch_support:
            raise ValueError("Stock always uses the entire file; sparse-only options cannot truncate it")
        source_args = StockArgs(bars=args.bars, seed=args.seed, center=True, steps=0,
                                score_after=2000, null_streams=4, inject_alpha=(0.0, 0.03, 0.1))
        bars = read_bars(source_args.bars)
        features_np, target_np = build_stream(bars, source_args)
        rng = np.random.default_rng(args.seed)
        orders = [rng.permutation(len(target_np)) for _ in range(4)]
        target_rows, inject_column = build_targets(target_np, features_np, source_args, orders)
        # The source has a learned intercept outside its feature matrix. Include
        # it as a constant coordinate, with no baseline weight decay on bias.
        features = torch.cat((torch.as_tensor(features_np, device=device),
                              torch.ones((len(target_np), 1), device=device)), dim=1)
        targets = torch.as_tensor(target_rows.T.copy(), device=device)
        streams = [{"alpha": alpha, "kind": "real" if kind == 0 else f"shuffled_{kind}"}
                   for alpha in source_args.inject_alpha for kind in range(5)]
        total, dim, score_after = len(target_np), features.shape[1], source_args.score_after
        if total // 2 <= score_after:
            raise ValueError("File lacks a scored first half after source score_after=2000")
        del bars, features_np, target_np, orders, target_rows
    data_seconds = time.perf_counter() - data_seconds_start
    configs, groups, candidates = configurations(streams, dim, args.task == "sparse")
    stamp = f"{time.time():.6f}"
    directory = Path(args.output_dir) / f"{args.task}__predictive_covariance_v2__{args.seed}__{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {"arguments": vars(args), "source_arguments": asdict(source_args), "seed": args.seed,
                "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                "dimensions_including_intercept": dim, "total_observations": total,
                "score_after": score_after, "midpoint": total // 2,
                "stock_injected_feature_column": inject_column,
                "source_sha256": {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("predictive_covariance_benchmark_v2.py", "predictive_covariance_optimizer_v2.py")},
                "data_preparation_seconds": data_seconds, "configs": configs,
                "memory": 0.0, "chunk_steps": CHUNK,
                "estimator": "predictive_covariance_v2",
                "method_key_mapping": {"transport": "full_covariance_linear_quadratic_estimator",
                                       "causal_transport_mixture": "causal_covariance_candidate_mixture"},
                "covariance_storage": "FP64 inverse covariance shared by distinct prior_scale; FP32 prediction weights",
                "finite_state_protocol": "all covariance configs require every shared/internal covariance-state buffer finite",
                "selection_protocol": "sparse exploratory only" if args.task == "sparse" else
                "choose each method/stream by scored first-half MSE at midpoint, report chronological second half; online updates continue",
                "limitations": ["One seed, linear models only; no Muon or universal optimizer superiority claim.",
                                "Full-covariance rank-one Bayesian linear regression; dense prior_density=1 is conjugate ridge conditional on model assumptions.",
                                "Sparse inclusion is approximate, not an exact joint spike-and-slab posterior or a nonlinear general optimizer.",
                                "Unlike v1, prior_scale is relative to noise standard deviation; NIG noise estimation and inclusion evidence also change, so this is not an isolated geometry ablation.",
                                "memory=0 accumulates all evidence and can impede adaptation after support changes.",
                                "Sparse final-risk/prequential grid winners are retrospective exploratory selections.",
                                "Stock shuffled controls permute the entire target, destroy serial structure, and are not a live-trading backtest.",
                                "Stock alpha controls are injected before shuffling; no clean market target is known.",
                                "Mixture is a causal Gaussian-likelihood heuristic using past target second moment, not calibrated Bayes.",
                                "Mixture permanently excludes nonfinite candidates; all-invalid experts fail the mixture with NaN scores.",
                                "Oracle sparse references know support (including empty null support) and are not achievable competitors.",
                                "Primitive SGD/AdamW match source epsilon and task-specific decay ordering (sparse before, stock after update)."]}
    save_json(directory / "metadata.json", metadata)
    writer = SummaryWriter(str(directory))
    writer.add_text("protocol", json.dumps(metadata, indent=2))
    print(f"run={directory} observations={total} dim={dim} configs={len(configs)} CUDA graph={CHUNK}", flush=True)
    runner = Runner(args, dim, streams, total, score_after, configs, groups, candidates)
    selection = {}
    try:
        runner.capture()
        print(f"compile/capture={runner.compile_seconds:.3f}s; all mutable state restored", flush=True)
        started = time.perf_counter()
        observed, next_log = 0, args.log_every
        midpoint = total // 2
        result = None
        while observed < total:
            # Never cross midpoint: selection and its artifact are frozen before
            # any second-half target can be ingested, even when not 100-aligned.
            boundary = midpoint if observed < midpoint else total
            length = min(CHUNK, boundary - observed)
            if args.task == "sparse":
                x, noisy_target, clean, _, support = stream.draw(length, observed + 1)
                clean_rows = torch.cat((clean, torch.zeros_like(clean)), dim=1)
                target_rows = torch.cat((noisy_target, noisy_target - clean), dim=1)
                runner.advance(x[:, 0, :], target_rows, clean_rows, support)
            else:
                runner.advance(features[observed:observed + length], targets[observed:observed + length])
            observed += length
            if observed >= next_log or observed in (midpoint, total):
                torch.cuda.synchronize()
                final_support = (stream.support(torch.tensor([observed], device=device))[0]
                                 if stream is not None else None)
                result = runner.results(observed, time.perf_counter() - started, final_support)
                if observed == midpoint and args.task == "stock":
                    for target_index in range(len(streams)):
                        for method in groups:
                            candidates_rows = [row for row in result["configs"]
                                               if row["stream"] == target_index and row["method"] == method]
                            best = best_row(candidates_rows, "first_half_mse")
                            selection[f"{target_index}:{method}"] = None if best is None else best["id"]
                    save_json(directory / "frozen_first_half_selection.json",
                              {"observations": observed, "selection": selection, "first_half_results": result})
                result["frozen_selection"] = selection.copy()
                result["comparisons"] = comparisons(result, args.task, selection)
                save_json(directory / "partial.json", result)
                log_result(writer, result, observed)
                print(f"progress {observed}/{total} elapsed={result['pass_wall_seconds_including_reporting']:.1f}s "
                      f"finite={sum(row['finite_optimizer_state'] for row in result['configs'])}/{len(configs)}", flush=True)
                print_comparisons(result, args.task)
                next_log = (observed // args.log_every + 1) * args.log_every
        save_json(directory / "results.json", {"metadata": metadata, **result})
        torch.save({"weight": runner.weight.cpu(), "configs": configs,
                    "transport_buffers": [value.cpu() for value in runner.transport_buffers]},
                   directory / "final_coefficients.pt")
        writer.add_scalar("runtime/compile_capture_seconds", runner.compile_seconds, total)
        writer.add_text("final_comparisons", json.dumps(json_safe(result["comparisons"]), indent=2), total)
        print(f"saved {directory / 'results.json'}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

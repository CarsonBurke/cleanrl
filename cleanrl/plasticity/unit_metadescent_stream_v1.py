"""Causal contextual meta-descent on a genuine single-example dense stream.

Hypothesis, NOT a measured result or novelty claim: a neuron's local state can
predict whether enlarging its incoming Adam row step reduces the NEXT actual
stream loss. This is a truncated online hypergradient method, not a noise mask.
For L_t = (prediction_t - noisy_y_t)^2 / 2 and applied step Delta_{t-1,i},

  c_i(s) = softplus(theta_i . phi_i(s) + inverse_softplus(1))
  h_{t,i} = <grad_i L_t, Delta_{t-1,i}> * dlog(c_{t-1,i}) * phi_{t-1,i}.

Delta includes its negative descent sign. h is the exact direct derivative
through the PREVIOUS applied step with its Adam direction, features and history
held fixed. We omit older weight paths, Hessian transport, Adam-state derivatives,
representation derivatives, and dependence of the current loss on earlier meta
updates. No unjustified long eligibility trace is introduced. Gate t is computed
BEFORE using y_t to update theta, so y_t cannot condition its own multiplier;
theta's update first changes gate t+1. The current noisy label still determines
the ordinary Adam direction. All upstream gradients remain completely ungated.

Bias-corrected RMSProp on h removes the old O(base_lr) meta-gradient attenuation:
h / sqrt(EMA(h^2)/(1-beta^n)) is invariant to a fixed positive rescaling of the
hypergradient history. Exact-zero coordinates use a unit denominator, not an
absolute epsilon that silently dominates tiny hypergradients. There is no gate
clamp, mean-one budget, regularizer, or clean-label supervision. Softplus starts
at identity with nonzero derivative; every network layer has nonzero weights.
This scaling does NOT make Adam trajectories invariant to base learning rate.

Arms (all independently trained and jointly base-LR/meta-LR tuned):
* adam: ordinary Adam, no gate learner.
* unit: one contextual readout per incoming neuron row, including output/bias.
* bias: one learned scalar per unit, no current-state features.
* shared: one readout of mean local features and ONE multiplier for all rows.
* shuffle: unit readouts, but fresh independent nonidentity cyclic routing within
  each hidden layer on every sample. At each step this exactly preserves that
  arm's layer-wise multiset of source gates, not the unit arm's divergent gates
  or any unit's temporal marginal. It breaks own-unit attachment, not shared
  sample information. Stored inverse routing assigns next-sample credit to the
  source that actually controlled each destination. The singleton output cannot
  be shuffled. Routing uses independent pre-generated randomness, never labels,
  future states, or future gates; random routing noise is a control limitation.

Local features: intercept, activation, squared activation, tanh(dy/dz), and four
fixed random projections of bias-free incoming activations followed by tanh.
The output activation feature is tanh(prediction). Hidden activation features are
already tanh. Input context is fan-in normalized; readouts/projections are not
trained through the task network. These small contextual readouts can underfit;
iid next-example credit can be noisy or uninformative even after normalization.

Default: paired seed1, 17-64-64-1, full65536 samples, midpoint teacher switch.
All arms see identical initialization/examples. Entire grids run full horizon;
nonfinite candidates are disqualified, not rescued by clipping. Duration-weighted
current-teacher validation checkpoints choose BOTH learning rates before ANY
test evaluation. Test is separate and evaluated only for locked candidates.
Recovery means the first recovery_samples after the switch, not endpoint-only
performance. Frozen-readout validation probes distinguish true within-unit state
variation from realized temporal drift and random routing variation. Seed1 has
no cross-seed confidence interval or claim of optimality; grid edges are flagged.

CUDA-only, FP32 optimizer statistics and full-precision matmuls. The actual update is
fullgraph-compiled and CUDA-graphed. Warmup/capture/replay is rolled back for EVERY
mutated tensor, including the stream index, routing, eligibility and diagnostics.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity.unit_bayes_stream_v1 import draw_teacher, evaluate, forward, init_weights, sample_state, teach
from cleanrl.shared import runtime


IDENTITY_LOGIT = math.log(math.expm1(1.0))
METHODS = ("adam", "unit", "bias", "shared", "shuffle")


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
    learning_rates: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    meta_rates: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    meta_beta: float = 0.99
    context_rank: int = 4
    validation: int = 2048
    test: int = 8192
    probe_samples: int = 128
    log_every: int = 4096
    recovery_samples: int = 4096
    graph_steps: int = 16
    output: str = ""


def validate_args(a):
    if min(a.samples, a.hidden, a.input_dim, a.validation, a.test, a.probe_samples,
           a.log_every, a.recovery_samples, a.graph_steps, a.context_rank) <= 0:
        raise ValueError("dimensions, counts and cadences must be positive")
    if a.samples < max(2, a.graph_steps):
        raise ValueError("capture needs at least two samples and one complete graph")
    if a.samples % a.graph_steps or a.log_every % a.graph_steps or a.recovery_samples % a.graph_steps:
        raise ValueError("samples, log_every and recovery_samples must align with graph_steps")
    if not 0 <= a.switch_at < 1 or not math.isfinite(a.noise) or not math.isfinite(a.hetero):
        raise ValueError("require finite noise/hetero and 0 <= switch_at < 1")
    if min(a.noise, a.hetero) < 0 or not 0 < a.meta_beta < 1:
        raise ValueError("noise/hetero must be nonnegative and 0 < meta_beta < 1")
    if a.switch_at:
        switch = int(a.samples * a.switch_at)
        if not 0 < switch < a.samples or switch % a.graph_steps:
            raise ValueError("teacher switch must fall inside the horizon and align with graph_steps")
        if switch + a.recovery_samples > a.samples:
            raise ValueError("the complete recovery window must fit after the teacher switch")
    if not a.methods or len(a.methods) != len(set(a.methods)) or set(a.methods) - set(METHODS):
        raise ValueError("methods must be nonempty, unique, and recognized")
    if "shuffle" in a.methods and a.hidden < 2:
        raise ValueError("shuffle needs at least two units in each hidden layer")
    for grid in (a.learning_rates, a.meta_rates):
        if len(grid) < 3 or any(not math.isfinite(v) or v <= 0 for v in grid):
            raise ValueError("each LR grid needs at least three finite positive values")
        if list(grid) != sorted(set(grid)):
            raise ValueError("LR grids must be strictly increasing")


def local_features(prediction, inputs, sensitivities, projections):
    """Accept sample (config, ...) or probe (config, example, ...) state."""
    activations = (inputs[1][..., :-1], inputs[2][..., :-1], torch.tanh(prediction).unsqueeze(-1))
    features = []
    for activation, inp, sensitivity, projection in zip(activations, inputs, sensitivities, projections):
        context = torch.tanh(inp[..., :-1] @ projection)
        context = context.unsqueeze(-2).expand(*activation.shape, projection.shape[-1])
        features.append(torch.cat((torch.ones_like(activation).unsqueeze(-1), activation.unsqueeze(-1),
                                   activation.square().unsqueeze(-1), torch.tanh(sensitivity).unsqueeze(-1),
                                   context), -1))
    return torch.cat(features, -2)


def gate_value_and_log_derivative(logit):
    """No gates below identity are forcibly rescued; softplus is positive in exact arithmetic."""
    gate = F.softplus(logit)
    # sigmoid(z) = 1-exp(-softplus(z)); expm1 preserves the ratio near zero.
    # The exact underflow limit is 1, without changing the applied gate.
    denominator = torch.where(gate > 0, gate, torch.ones_like(gate))
    derivative = torch.where(gate > 0, -torch.expm1(-gate) / denominator, torch.ones_like(gate))
    return gate, derivative


class Learner:
    """Optimizer receives only x and observed y: clean targets cannot enter its API."""

    def __init__(self, method, grid, initial, projections, a, xs, ys, shifts):
        self.method, self.a = method, a
        self.xs, self.ys, self.shifts = xs, ys, shifts
        self.projections = projections
        k, device = len(grid), xs.device
        self.weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
        self.m = [torch.zeros_like(w) for w in self.weights]
        self.v = [torch.zeros_like(w) for w in self.weights]
        self.lr = torch.tensor([row[0] for row in grid], device=device)[:, None, None]
        self.meta_lr = torch.tensor([row[1] for row in grid], device=device)[:, None, None]
        self.units = 2 * a.hidden + 1
        self.index = torch.zeros((), dtype=torch.int64, device=device)
        self.steps = torch.zeros((), device=device)
        self.gate_sum = torch.zeros(k, self.units, device=device)
        self.gate_square = torch.zeros_like(self.gate_sum)
        self.cross_unit_variance = torch.zeros(k, device=device)
        self.mutable = [*self.weights, *self.m, *self.v, self.index, self.steps,
                        self.gate_sum, self.gate_square, self.cross_unit_variance]
        if method != "adam":
            owners = 1 if method == "shared" else self.units
            features = 1 if method == "bias" else 4 + a.context_rank
            self.theta = torch.zeros(k, owners, features, device=device)
            self.meta_v = torch.zeros_like(self.theta)
            self.previous_features = torch.zeros_like(self.theta)
            self.previous_log_derivative = torch.zeros(k, owners, device=device)
            self.previous_step = [torch.zeros_like(w) for w in self.weights]
            self.previous_inverse = torch.arange(self.units, device=device)
            self.local_index = torch.arange(a.hidden, device=device)
            self.output_index = torch.tensor([2 * a.hidden], device=device)
            self.mutable += [self.theta, self.meta_v, self.previous_features, self.previous_log_derivative,
                             *self.previous_step, self.previous_inverse]

    def features(self, prediction, inputs, sensitivities):
        if self.method == "bias":
            return torch.ones(*prediction.shape, self.units, 1, device=prediction.device)
        phi = local_features(prediction, inputs, sensitivities, self.projections)
        return phi.mean(-2, keepdim=True) if self.method == "shared" else phi

    @torch.no_grad()
    def update(self):
        ix = self.index.reshape(1)
        x = self.xs.index_select(0, ix).squeeze(0)
        prediction, inputs, sensitivities = sample_state(self.weights, x)
        self.steps.add_(1)
        y = self.ys.index_select(0, ix).squeeze(0)
        residual = prediction - y
        gradients = [residual[:, None, None] * j.unsqueeze(-1) * inp.unsqueeze(1)
                     for inp, j in zip(inputs, sensitivities)]
        if self.method == "adam":
            gate = torch.ones_like(self.gate_sum)
        else:
            # Gate inputs contain no label/residual, and theta is not changed
            # by the current meta-credit until this gate has been computed.
            phi = self.features(prediction, inputs, sensitivities)
            source_gate, log_derivative = gate_value_and_log_derivative(
                (self.theta * phi).sum(-1) + IDENTITY_LOGIT)
            credit = torch.cat([(g * previous).sum(-1)
                                for g, previous in zip(gradients, self.previous_step)], -1)
            if self.method == "shuffle":
                credit = credit.index_select(-1, self.previous_inverse)
                shifts = self.shifts.index_select(0, ix).squeeze(0)
                order = torch.cat(((self.local_index + shifts[0]) % self.a.hidden,
                                   self.a.hidden + (self.local_index + shifts[1]) % self.a.hidden,
                                   self.output_index))
                inverse = torch.cat(((self.local_index - shifts[0]) % self.a.hidden,
                                     self.a.hidden + (self.local_index - shifts[1]) % self.a.hidden,
                                     self.output_index))
                gate = source_gate.index_select(-1, order)
                self.previous_inverse.copy_(inverse)
            else:
                gate = source_gate.expand(-1, self.units)
                if self.method == "shared":
                    credit = credit.sum(-1, keepdim=True)
            hypergradient = (credit * self.previous_log_derivative).unsqueeze(-1) * self.previous_features
            self.meta_v.lerp_(hypergradient.square(), 1 - self.a.meta_beta)
            meta_count = torch.maximum(self.steps - 1, torch.ones_like(self.steps))
            rms = (self.meta_v / (1 - self.a.meta_beta ** meta_count)).sqrt()
            denominator = torch.where(rms > 0, rms, torch.ones_like(rms))
            self.theta.sub_(self.meta_lr * hypergradient / denominator)
            self.previous_features.copy_(phi)
            self.previous_log_derivative.copy_(log_derivative)
        bc1, bc2 = 1 - 0.9 ** self.steps, 1 - 0.999 ** self.steps
        start = 0
        for layer, (w, m, v, g) in enumerate(zip(self.weights, self.m, self.v, gradients)):
            m.lerp_(g, 0.1)
            v.lerp_(g.square(), 0.001)
            stop = start + w.shape[1]
            step = -self.lr * (m / bc1) / ((v / bc2).sqrt() + 1e-8)
            applied = gate[:, start:stop, None] * step
            w.add_(applied)
            if self.method != "adam":
                self.previous_step[layer].copy_(applied)
            start = stop
        self.gate_sum.add_(gate)
        self.gate_square.add_(gate.square())
        self.cross_unit_variance.add_(gate.var(-1, unbiased=False))
        self.index.add_(1)

    def snapshot(self):
        return [tensor.clone() for tensor in self.mutable]

    def restore(self, snapshot):
        for destination, source in zip(self.mutable, snapshot):
            destination.copy_(source)

    @torch.no_grad()
    def capture(self):
        """Compile/capture the actual update and prove replay parity, then restore."""
        initial = self.snapshot()
        for _ in range(self.a.graph_steps):
            self.update()
        expected = self.snapshot()
        self.restore(initial)
        compiled = torch.compile(self.update, fullgraph=True, mode="max-autotune-no-cudagraphs")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            compiled()
            compiled()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.restore(initial)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(self.a.graph_steps):
                compiled()
        self.restore(initial)
        graph.replay()
        torch.cuda.synchronize()
        max_error = 0.0
        try:
            for actual, reference in zip(self.mutable, expected):
                torch.testing.assert_close(actual, reference, rtol=3e-3, atol=3e-5)
                max_error = max(max_error, float((actual - reference).abs().max()))
        finally:
            self.restore(initial)
        return graph, max_error

    @torch.no_grad()
    def diagnostics(self, x):
        mean = self.gate_sum / self.steps
        temporal = (self.gate_square / self.steps - mean.square()).clamp_min(0).sqrt().mean(-1)
        result = {"realized_gate_mean": mean.mean(-1), "realized_within_unit_temporal_sd": temporal,
                  "realized_cross_unit_rms_sd": (self.cross_unit_variance / self.steps).sqrt()}
        if self.method == "adam":
            result.update(frozen_within_unit_state_sd=torch.zeros_like(temporal),
                          frozen_cross_unit_sd=torch.zeros_like(temporal), theta_rms=torch.zeros_like(temporal))
        else:
            h1, h2, prediction = forward(self.weights, x)
            one = torch.ones_like(prediction).unsqueeze(-1)
            inputs = (torch.cat((x.unsqueeze(0).expand(prediction.shape[0], -1, -1), one), -1),
                      torch.cat((h1, one), -1), torch.cat((h2, one), -1))
            j2 = self.weights[2][:, None, 0, :-1] * (1 - h2.square())
            j1 = torch.einsum("kbo,koi->kbi", j2, self.weights[1][..., :-1]) * (1 - h1.square())
            phi = self.features(prediction, inputs, (j1, j2, one))
            gate, _ = gate_value_and_log_derivative((self.theta.unsqueeze(1) * phi).sum(-1) + IDENTITY_LOGIT)
            gate = gate.expand(-1, -1, self.units)
            # For shuffle these are SOURCE gates (equivalently any fixed routing),
            # never independently shuffled probes that mistake routing noise for state dependence.
            result.update(frozen_within_unit_state_sd=gate.std(1, unbiased=False).mean(-1),
                          frozen_cross_unit_sd=gate.std(-1, unbiased=False).mean(-1),
                          theta_rms=self.theta.square().mean((-1, -2)).sqrt())
        return {name: value.cpu().tolist() for name, value in result.items()}


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_samples(a, switch):
    checkpoints = set(range(a.log_every, a.samples + 1, a.log_every)) | {a.samples, switch}
    if switch < a.samples:
        checkpoints.add(switch + a.recovery_samples)
        offset = a.graph_steps
        while offset < a.recovery_samples:
            checkpoints.add(switch + offset)
            offset *= 2
    return sorted(checkpoints)


@torch.no_grad()
def main():
    a = tyro.cli(Args)
    validate_args(a)
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("This optimizer requires compiled CUDA execution; no CPU fallback")
    device = torch.device("cuda")

    def generator(offset):
        return torch.Generator(device=device).manual_seed(a.seed + offset)

    # Independent namespaces keep changes to split size/routing/features from
    # changing the teacher, training stream, noise or initial model.
    teachers = [draw_teacher(a, generator(0), device), draw_teacher(a, generator(1), device)]
    initial = init_weights(a, generator(2), device)
    xs = torch.randn(a.samples, a.input_dim, generator=generator(3), device=device)
    direction = torch.randn(a.input_dim, generator=generator(4), device=device)
    direction /= direction.norm()
    sigma = a.noise * (a.hetero * torch.tanh(xs @ direction)).exp()
    switch = int(a.samples * a.switch_at) if a.switch_at else a.samples
    clean = teach(teachers[0], xs)
    if switch < a.samples:
        clean[switch:] = teach(teachers[1], xs[switch:])
    ys = clean + sigma * torch.randn(a.samples, generator=generator(5), device=device)
    if not torch.isfinite(ys).all().item():
        raise ValueError("Generated observations overflow float32")
    del clean, sigma
    projection_generator = generator(6)
    projections = [torch.randn(w.shape[-1] - 1, a.context_rank, generator=projection_generator, device=device)
                   / math.sqrt(w.shape[-1] - 1) for w in initial]
    shifts = torch.randint(1, max(2, a.hidden), (a.samples, 2), generator=generator(7), device=device)
    xv = torch.randn(a.validation, a.input_dim, generator=generator(8), device=device)
    yv = [teach(teacher, xv) for teacher in teachers]
    checkpoints = checkpoint_samples(a, switch)
    intervals = [stop - start for start, stop in zip([0, *checkpoints[:-1]], checkpoints)]
    root = Path(a.output or "runs") / f"DenseStream__unit_metadescent_v1__{a.seed}__{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(root))
    writer.add_text("hyperparameters", json.dumps(asdict(a), indent=2))
    result = {"args": asdict(a), "run_dir": str(root), "status": "training",
              "protocol": "paired seed1 default; full horizons; per-arm joint validation selection; no cross-seed inference",
              "sustained_metric": "duration-weighted right-endpoint current-teacher held-out MSE, not exact loss integral",
              "hypergradient": "one-step direct only; frozen Adam direction/state, representation and older weight paths",
              "meta_scaling": "bias-corrected coordinate RMSProp; exact-zero denominator=1; no gate budget or clamp",
              "shuffle_marginals": "exact same-arm per-step hidden-layer gate multisets; not temporal or cross-arm marginals",
              "frozen_probe": "validation inputs; fixed network/readout; shuffle source gates, not random routing noise",
              "dtype": "FP32 optimizer/readout and matmuls; TF32 disabled for normalized-hypergradient fidelity", "device": torch.cuda.get_device_name(),
              "checkpoints": checkpoints, "actual_switch_sample": switch if a.switch_at else None, "methods": {}}
    saved_snapshots = {}
    try:
        for method in a.methods:
            grid = [(lr, meta) for lr in a.learning_rates
                    for meta in ((0.0,) if method == "adam" else a.meta_rates)]
            learner = Learner(method, grid, initial, projections, a, xs, ys, shifts)
            setup_start = time.perf_counter()
            graph, parity = learner.capture()
            startup = time.perf_counter() - setup_start
            curves, snapshots = [], []
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            update_seconds, completed = 0.0, 0
            for stop in checkpoints:
                begin.record()
                for _ in range((stop - completed) // a.graph_steps):
                    graph.replay()
                end.record()
                segment = int(stop > switch)
                validation = evaluate(learner.weights, xv, yv[segment]).cpu().tolist()
                update_seconds += begin.elapsed_time(end) / 1000
                diagnostics = learner.diagnostics(xv[:a.probe_samples])
                row = {"step": stop, "validation": validation, **diagnostics}
                curves.append(row)
                snapshots.append([w.cpu().clone() for w in learner.weights])
                completed = stop
                for index, (lr, meta) in enumerate(grid):
                    tag = f"{method}/lr_{lr:g}/meta_{meta:g}"
                    writer.add_scalar(f"validation/{tag}", validation[index], stop)
                    for name, values in diagnostics.items():
                        writer.add_scalar(f"{name}/{tag}", values[index], stop)
                print(json.dumps(finite_json({"method": method, **row}), allow_nan=False), flush=True)
                result["active"] = {"method": method, "completed_samples": stop, "curves": curves}
                save_json(root / "results.json", result)
                writer.flush()
            scores = [sum(row["validation"][index] * duration for row, duration in zip(curves, intervals)) / a.samples
                      for index in range(len(grid))]
            eligible = [index for index, score in enumerate(scores) if math.isfinite(score)
                        and all(math.isfinite(row["validation"][index]) for row in curves)
                        and all(math.isfinite(values[index]) for name, values in curves[-1].items()
                                if name not in ("step", "validation"))]
            best = min(eligible, key=scores.__getitem__) if eligible else None
            chosen = grid[best] if best is not None else None
            entry = {"grid": [{"lr": lr, "meta_lr": meta} for lr, meta in grid], "chosen_index": best,
                     "chosen": None if chosen is None else {"lr": chosen[0], "meta_lr": chosen[1]},
                     "lr_edge": chosen is None or chosen[0] in (a.learning_rates[0], a.learning_rates[-1]),
                     "meta_lr_edge": method != "adam" and (chosen is None or chosen[1] in (a.meta_rates[0], a.meta_rates[-1])),
                     "validation_sustained_grid": scores, "invalid_indices": [i for i in range(len(grid)) if i not in eligible],
                     "curves": curves, "startup_seconds": startup, "graph_update_seconds": update_seconds,
                     "stream_samples_per_second": a.samples / update_seconds,
                     "candidate_samples_per_second": a.samples * len(grid) / update_seconds,
                     "graph_parity_max_abs": parity}
            result["methods"][method] = entry
            save_json(root / f"selection_{method}.json", entry)
            saved_snapshots[method] = [] if best is None else [[w[best:best + 1].clone() for w in snapshot]
                                                            for snapshot in snapshots]
            del graph, learner, snapshots
        # ALL arms' choices are locked on disk before even generating test inputs.
        result.pop("active", None)
        result["status"] = "all_hyperparameters_locked"
        save_json(root / "results.json", result)
        xt = torch.randn(a.test, a.input_dim, generator=generator(9), device=device)
        yt = [teach(teacher, xt) for teacher in teachers]
        result["zero_test_mse_by_segment"] = [float(y.square().mean()) for y in yt]
        for method in a.methods:
            entry = result["methods"][method]
            curve = []
            for stop, snapshot in zip(checkpoints, saved_snapshots[method]):
                weights = [w.to(device) for w in snapshot]
                mse = float(evaluate(weights, xt, yt[int(stop > switch)])[0])
                curve.append({"step": stop, "mse": mse})
                writer.add_scalar(f"test/{method}/clean_mse", mse, stop)
            entry["test_curve"] = curve
            if curve:
                if any(not math.isfinite(row["mse"]) for row in curve):
                    result["status"] = "nonfinite_selected_test"
                    save_json(root / "results.json", result)
                    raise RuntimeError(f"{method} selected candidate has nonfinite untouched-test predictions")
                entry["test_sustained_mse"] = sum(row["mse"] * dt for row, dt in zip(curve, intervals)) / a.samples
                entry["test_endpoint_mse"] = curve[-1]["mse"]
                if switch < a.samples:
                    recovery = [(row, dt) for row, dt in zip(curve, intervals)
                                if switch < row["step"] <= switch + a.recovery_samples]
                    post = [(row, dt) for row, dt in zip(curve, intervals) if row["step"] > switch]
                    entry["test_recovery_mse"] = sum(row["mse"] * dt for row, dt in recovery) / a.recovery_samples
                    entry["test_recovery_endpoint_mse"] = recovery[-1][0]["mse"]
                    entry["test_post_switch_sustained_mse"] = sum(row["mse"] * dt for row, dt in post) / (a.samples - switch)
                best = entry["chosen_index"]
                entry["selected_diagnostics"] = [{name: values[best] for name, values in row.items()
                                                   if name not in ("step", "validation")} | {"step": row["step"]}
                                                  for row in entry["curves"]]
            print("RESULT " + json.dumps(finite_json({"method": method, **{name: value for name, value in entry.items()
                                                                           if name not in ("curves", "grid", "selected_diagnostics")}}),
                                         allow_nan=False), flush=True)
        baseline = result["methods"].get("adam", {}).get("test_sustained_mse")
        if baseline is not None:
            for entry in result["methods"].values():
                if "test_sustained_mse" in entry:
                    entry["paired_test_sustained_difference_vs_adam"] = entry["test_sustained_mse"] - baseline
        failed = [name for name, entry in result["methods"].items() if entry["chosen_index"] is None]
        result["status"] = "no_finite_validation_candidate" if failed else "completed"
        save_json(root / "results.json", result)
        if failed:
            raise RuntimeError(f"No finite validation candidate for {failed}; complete-grid evidence saved")
        print(f"RESULTS {root / 'results.json'}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

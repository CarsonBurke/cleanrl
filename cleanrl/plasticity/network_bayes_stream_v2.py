"""Full-network Bayesian plasticity, v2: retain cross-neuron posterior covariance.

The exact bias-augmented output Jacobian j conditions one FP32 covariance P over
all weights: P <- P + diag(Q), k = Pj / (R + j'Pj), theta <- theta - k * residual,
P <- P - (Pj)(Pj)' / (R + j'Pj). This is a full extended Kalman linearization,
not an optimality guarantee for a nonlinear network. The hypothesis is that
discarding cross-neuron correlations repeatedly double-counts redundant features.

Dense 17-64-64-1 regression, paired seed 1, full 65,536-sample streams. Adam and
the neuron-block EKF are comparison arms. network_scalar retains full covariance
conditioning but applies positive scalar leverage along j, erasing the update's
directional preconditioning. Each arm trains independently; scalar's subsequent
Jacobians can differ. Cross-block credit j_block' (Pj)_block / denominator can be
negative even for positive-definite P; logged row dispersion is signed credit,
not a distribution of positive neuron uncertainties.

Validation selects prior scale or Adam LR; untouched test reports selected
trajectories. All arms share examples, initialization and held-out splits.
Ordinary arms use a causal residual-square EMA for R, an innovation-scale
approximation including model error. Known conditional noise is privileged,
not a bound. No clean target enters an optimizer. CUDA-only compiled updates
and explicit CUDA graphs restore warmup state before consuming research data.
Full covariance and its matrix products use FP32, with TF32 disabled.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared import runtime


@dataclass
class Args:
    seed: int = 1
    samples: int = 65536
    hidden: int = 64
    input_dim: int = 17
    noise: float = 1.0
    hetero: float = 0.0
    switch_at: float = 0.0
    methods: tuple[str, ...] = ("adam", "unit", "network", "network_scalar")
    adam_lrs: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    prior_scales: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    diffusion: float = 1e-5
    """Per-sample isotropic process variance, as a fraction of initial row variance."""
    noise_rate: float = 0.001
    known_noise: bool = False
    """Privileged known conditional noise, reference only; off by default."""
    validation: int = 2048
    test: int = 8192
    log_every: int = 4096
    graph_steps: int = 16
    output: str = ""


def validate_args(a):
    if min(a.samples, a.hidden, a.input_dim, a.validation, a.test, a.log_every, a.graph_steps) <= 0:
        raise ValueError("sample counts, dimensions and cadences must be positive")
    if a.samples % a.graph_steps or a.log_every % a.graph_steps:
        raise ValueError("samples and log_every must be divisible by graph_steps")
    if not 0 <= a.switch_at < 1 or a.noise < 0 or a.hetero < 0:
        raise ValueError("require 0 <= switch_at < 1 and nonnegative noise/hetero")
    if a.switch_at and int(a.samples * a.switch_at) % a.graph_steps:
        raise ValueError("the teacher switch must be aligned with graph_steps")
    if not 0 < a.noise_rate <= 1 or a.diffusion < 0:
        raise ValueError("require 0 < noise_rate <= 1 and nonnegative diffusion")
    if not a.methods or len(set(a.methods)) != len(a.methods):
        raise ValueError("methods must be nonempty and unique")
    if set(a.methods) - {"adam", "unit", "network", "network_scalar"}:
        raise ValueError("unknown method")
    for grid in (a.adam_lrs, a.prior_scales):
        if len(grid) < 3 or any(not math.isfinite(v) or v <= 0 for v in grid):
            raise ValueError("each grid needs at least three finite positive entries")
        if list(grid) != sorted(set(grid)):
            raise ValueError("grids must be strictly increasing")
    if a.known_noise and a.noise == 0:
        raise ValueError("known_noise requires positive observation noise")


def forward(weights, x):
    """Bias is the final incoming coordinate; configs are independent leading rows."""
    h1 = torch.tanh(x @ weights[0][..., :-1].transpose(-1, -2) + weights[0][..., -1].unsqueeze(1))
    h2 = torch.tanh(h1 @ weights[1][..., :-1].transpose(-1, -2) + weights[1][..., -1].unsqueeze(1))
    y = h2 @ weights[2][..., :-1].transpose(-1, -2) + weights[2][..., -1].unsqueeze(1)
    return h1, h2, y.squeeze(-1)


def sample_state(weights, x):
    """Exact output Jacobian, not loss gradient; independent of target noise."""
    h1 = torch.tanh(torch.einsum("koi,i->ko", weights[0][..., :-1], x) + weights[0][..., -1])
    h2 = torch.tanh(torch.einsum("koi,ki->ko", weights[1][..., :-1], h1) + weights[1][..., -1])
    out = (weights[2][:, 0, :-1] * h2).sum(-1) + weights[2][:, 0, -1]
    j2 = weights[2][:, 0, :-1] * (1 - h2.square())
    j1 = torch.einsum("ko,koi->ki", j2, weights[1][..., :-1]) * (1 - h1.square())
    one = torch.ones_like(out).unsqueeze(-1)
    inputs = (torch.cat((x.expand(weights[0].shape[0], -1), one), -1),
              torch.cat((h1, one), -1), torch.cat((h2, one), -1))
    return out, inputs, (j1, j2, one)


def init_weights(a, gen, device):
    out = []
    for o, i, gain in ((a.hidden, a.input_dim, math.sqrt(2)),
                       (a.hidden, a.hidden, math.sqrt(2)), (1, a.hidden, 1.0)):
        draw = torch.randn(max(o, i), min(o, i), generator=gen, device=device)
        q, _ = torch.linalg.qr(draw)
        w = torch.zeros(o, i + 1, device=device)
        w[:, :-1] = (q if o >= i else q.T) * gain
        out.append(w)
    return out


def draw_teacher(a, gen, device):
    return [torch.randn(o, i, generator=gen, device=device) * gain / math.sqrt(i)
            for o, i, gain in ((a.hidden, a.input_dim, 1.0),
                                (a.hidden, a.hidden, 1.5), (1, a.hidden, 3.0))]


def teach(weights, x):
    return (torch.tanh(torch.tanh(x @ weights[0].T) @ weights[1].T) @ weights[2].T).squeeze(-1)


class Learner:
    def __init__(self, method, grid, initial, a, xs, ys, clean, noise_var):
        self.method, self.a = method, a
        # Full P is bandwidth-bound: capture one update rather than retaining
        # graph_steps sets of large compiler buffers in a private graph pool.
        self.capture_steps = 1 if method in ("network", "network_scalar") else a.graph_steps
        k, device = len(grid), xs.device
        self.weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
        self.scale = torch.tensor(grid, device=device)
        self.index = torch.zeros((), dtype=torch.int64, device=device)
        self.steps = torch.zeros((), device=device)
        self.error = torch.zeros(k, device=device)
        self.null_error = torch.zeros((), device=device)
        self.noise = torch.ones(k, device=device)
        self.gain_sum = torch.zeros(k, device=device)
        self.unit_variance = torch.zeros(k, device=device)
        self.xs, self.ys, self.clean, self.noise_var = xs, ys, clean, noise_var
        self.mutable = [*self.weights, self.index, self.steps, self.error,
                        self.null_error, self.noise, self.gain_sum, self.unit_variance]
        if method == "adam":
            self.m = [torch.zeros_like(w) for w in self.weights]
            self.v = [torch.zeros_like(w) for w in self.weights]
            self.mutable += self.m + self.v
        elif method in ("network", "network_scalar"):
            # Concatenation order is layer, output row, incoming coordinate (bias last).
            # Match the block-unit prior exactly without allocating a dense Q or eye.
            prior = torch.cat([
                (self.scale / w.shape[-1])[:, None].expand(k, w[0].numel())
                for w in self.weights], -1)
            self.cov = torch.zeros(k, prior.shape[-1], prior.shape[-1],
                                   device=device, dtype=torch.float32)
            self.cov.diagonal(dim1=-2, dim2=-1).copy_(prior)
            self.process = a.diffusion * prior
            self.mutable.append(self.cov)
        else:
            self.cov, self.process = [], []
            for w in self.weights:
                d = w.shape[-1]
                # Same fan-in-normalized prior and process noise as unit v1.
                p = (self.scale / d).view(k, 1, 1, 1) * torch.eye(d, device=device).view(1, 1, d, d)
                self.cov.append(p.expand(k, w.shape[1], d, d).clone())
                self.process.append(a.diffusion * p)
            self.mutable += self.cov

    def network_update(self, inputs, sensitivities, residual, observation):
        """Full EKF conditioning; no off-block projection or dense process matrix."""
        assert isinstance(self.cov, torch.Tensor) and isinstance(self.process, torch.Tensor)
        jacobian = torch.cat([
            (j.unsqueeze(-1) * inp.unsqueeze(1)).flatten(1)
            for inp, j in zip(inputs, sensitivities)], -1)
        self.cov.diagonal(dim1=-2, dim2=-1).add_(self.process)
        pj = torch.bmm(self.cov, jacobian.unsqueeze(-1)).squeeze(-1)
        leverage = (jacobian * pj).sum(-1)
        denominator = observation + leverage
        if self.method == "network_scalar":
            # Positive for PSD P; do not abs/clamp away a broken posterior.
            # The output bias makes ||j||² strictly positive.
            applied = jacobian * (leverage / jacobian.square().sum(-1)).unsqueeze(-1)
        else:
            applied = pj
        credit, start = [], 0
        for w in self.weights:
            stop = start + w[0].numel()
            direction = applied[:, start:stop].view_as(w)
            block_j = jacobian[:, start:stop].view_as(w)
            credit.append((block_j * direction).sum(-1) / denominator.unsqueeze(-1))
            w.sub_(direction * (residual / denominator)[:, None, None])
            start = stop
        # Scalar control changes only the mean update, never posterior conditioning.
        # Compilation can fuse this rank-one subtraction into the in-place write.
        self.cov.sub_(pj.unsqueeze(-1) * pj.unsqueeze(-2) / denominator[:, None, None])
        return torch.cat(credit, -1)

    @torch.no_grad()
    def update(self):
        # index_select is graph-safe and avoids an implicit host extraction of index.
        ix = self.index.reshape(1)
        x = self.xs.index_select(0, ix).squeeze(0)
        y = self.ys.index_select(0, ix).squeeze(0)
        target = self.clean.index_select(0, ix).squeeze(0)
        prediction, inputs, sensitivities = sample_state(self.weights, x)
        residual = prediction - y
        self.error.add_((prediction - target).square())
        self.null_error.add_(target.square())
        self.steps.add_(1)
        if self.method == "adam":
            bc1, bc2 = 1 - 0.9 ** self.steps, 1 - 0.999 ** self.steps
            for w, m, v, inp, j in zip(self.weights, self.m, self.v, inputs, sensitivities):
                g = residual[:, None, None] * j.unsqueeze(-1) * inp.unsqueeze(1)
                m.lerp_(g, 0.1)
                v.lerp_(g.square(), 0.001)
                w.sub_(self.scale[:, None, None] * (m / bc1) / ((v / bc2).sqrt() + 1e-8))
        else:
            observation = (self.noise_var.index_select(0, ix).squeeze(0)
                           if self.a.known_noise else self.noise)
            if self.method in ("network", "network_scalar"):
                gains = self.network_update(inputs, sensitivities, residual, observation)
            else:
                mapped, uncertainty = [], []
                for p, q, inp, j in zip(self.cov, self.process, inputs, sensitivities):
                    p.add_(q)
                    px = torch.einsum("koij,kj->koi", p, inp)
                    mapped.append(px)
                    uncertainty.append(j.square() * (px * inp.unsqueeze(1)).sum(-1))
                # Preserve unit v1's denominator for the comparison arm.
                denominator = observation + sum(u.sum(-1) for u in uncertainty) + 1e-8
                gains = torch.cat([u / denominator.unsqueeze(-1) for u in uncertainty], -1)
                for w, p, px, j in zip(self.weights, self.cov, mapped, sensitivities):
                    w.sub_(residual[:, None, None] * j.unsqueeze(-1) * px / denominator[:, None, None])
                    reduction = (j.square() / denominator.unsqueeze(-1))[:, :, None, None]
                    p.sub_(reduction * px.unsqueeze(-1) * px.unsqueeze(-2))
            self.gain_sum.add_(gains.sum(-1))
            self.unit_variance.add_(gains.var(-1, unbiased=False))
            # Prediction uses history only. Current residual updates the NEXT scale.
            self.noise.lerp_(residual.square(), self.a.noise_rate)
        self.index.add_(1)

    def snapshot(self):
        return [t.clone() for t in self.mutable]

    def restore(self, values):
        for t, v in zip(self.mutable, values):
            t.copy_(v)

    def capture(self):
        """Audit graph replay against eager before any research samples are consumed."""
        state = self.snapshot()
        for _ in range(self.capture_steps):
            self.update()
        expected = self.snapshot()
        self.restore(state)
        compiled = torch.compile(self.update, fullgraph=True, mode="max-autotune-no-cudagraphs")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            compiled()
            compiled()
        torch.cuda.current_stream().wait_stream(stream)
        self.restore(state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(self.capture_steps):
                compiled()
        self.restore(state)
        graph.replay()
        torch.cuda.synchronize()
        # Compiled FP32 reductions may reorder accumulation, not semantics.
        max_error = 0.0
        for actual, reference in zip(self.mutable, expected):
            torch.testing.assert_close(actual, reference, rtol=3e-3, atol=3e-5)
            max_error = max(max_error, float((actual - reference).abs().max()))
        self.restore(state)
        return graph, max_error


@torch.no_grad()
def evaluate(weights, x, y):
    # Fixed-size chunks bound peak evaluation memory regardless of the grid width.
    error = torch.zeros(weights[0].shape[0], device=x.device)
    for start in range(0, len(x), 512):
        out = forward(weights, x[start:start + 512])[2]
        error += (out - y[start:start + 512]).square().sum(-1)
    return error / len(x)


@torch.no_grad()
def geometry_probe(learner, x):
    """Frozen next-prediction geometry; full probes allocate only K x examples x P.

    Relative directional variance is one for isotropic covariance. Full-network
    posterior trace is reported separately from next-prediction trace (P + Q).
    These diagnostics do not establish positive definiteness in unprobed directions.
    """
    if learner.method == "adam":
        return []
    if learner.method in ("network", "network_scalar"):
        h1, h2, prediction = forward(learner.weights, x)
        one = torch.ones_like(prediction).unsqueeze(-1)
        inputs = (torch.cat((x.unsqueeze(0).expand(prediction.shape[0], -1, -1), one), -1),
                  torch.cat((h1, one), -1), torch.cat((h2, one), -1))
        j2 = learner.weights[2][:, None, 0, :-1] * (1 - h2.square())
        j1 = torch.einsum("kbo,koi->kbi", j2, learner.weights[1][..., :-1]) * (1 - h1.square())
        jacobian = torch.cat([(j.unsqueeze(-1) * inp.unsqueeze(-2)).flatten(2)
                              for inp, j in zip(inputs, (j1, j2, one))], -1)
        pj = torch.bmm(learner.cov, jacobian.transpose(1, 2)).transpose(1, 2)
        pj.add_(learner.process.unsqueeze(1) * jacobian)
        actual = (jacobian * pj).sum(-1)
        diagonal = learner.cov.diagonal(dim1=-2, dim2=-1)
        trace = diagonal.sum(-1)
        predicted_trace = trace + learner.process.sum(-1)
        isotropic = predicted_trace[:, None] / diagonal.shape[-1] * jacobian.square().sum(-1)
        relative = actual / isotropic
        return {
            "posterior_trace": trace.cpu().tolist(),
            "prediction_trace": predicted_trace.cpu().tolist(),
            "minimum_posterior_diagonal": diagonal.amin(-1).cpu().tolist(),
            "state_geometry_sd": relative.std(1, unbiased=False).cpu().tolist(),
            "minimum_directional_variance": actual.amin(1).cpu().tolist(),
            "maximum_directional_variance": actual.amax(1).cpu().tolist(),
        }
    h1, h2, _ = forward(learner.weights, x)
    k = learner.weights[0].shape[0]
    one = torch.ones(k, len(x), 1, device=x.device)
    inputs = (torch.cat((x.unsqueeze(0).expand(k, -1, -1), one), -1),
              torch.cat((h1, one), -1), torch.cat((h2, one), -1))
    result = []
    for covariance, process, inp in zip(learner.cov, learner.process, inputs):
        p = covariance + process
        px = torch.einsum("koij,kbj->kboi", p, inp)
        actual = (px * inp.unsqueeze(2)).sum(-1)
        isotropic = (p.diagonal(dim1=-2, dim2=-1).mean(-1).unsqueeze(1)
                     * inp.square().sum(-1, keepdim=True))
        relative = actual / isotropic.clamp_min(1e-30)
        result.append({
            "state_geometry_sd": relative.std(1, unbiased=False).mean(-1).cpu().tolist(),
            "unit_geometry_sd": relative.std(2, unbiased=False).mean(-1).cpu().tolist(),
            "minimum_directional_variance": actual.amin((1, 2)).cpu().tolist(),
        })
    return result


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def main():
    a = tyro.cli(Args)
    validate_args(a)
    runtime.configure_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(a.seed)
    t1, t2 = draw_teacher(a, gen, device), draw_teacher(a, gen, device)
    initial = init_weights(a, gen, device)
    xs = torch.randn(a.samples, a.input_dim, generator=gen, device=device)
    direction = torch.randn(a.input_dim, generator=gen, device=device)
    direction /= direction.norm()
    sigma = a.noise * (a.hetero * torch.tanh(xs @ direction)).exp()
    clean = teach(t1, xs)
    switch = int(a.samples * a.switch_at) if a.switch_at else a.samples
    if switch < a.samples:
        clean[switch:] = teach(t2, xs[switch:])
    ys = clean + sigma * torch.randn(a.samples, generator=gen, device=device)
    xv = torch.randn(a.validation, a.input_dim, generator=gen, device=device)
    xt = torch.randn(a.test, a.input_dim, generator=gen, device=device)
    yv = [teach(t1, xv), teach(t2, xv)]
    test_targets = [teach(t1, xt), teach(t2, xt)]
    yt = test_targets[1 if a.switch_at else 0]
    # Task generation stays bit-for-bit on v1's runtime policy. Disable TF32 only
    # after drawing the shared data, for full posterior products and all learners.
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    root = Path(a.output or "runs") / f"DenseStream__network_bayes_v2__{a.seed}__{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(root))
    writer.add_text("hyperparameters", json.dumps(asdict(a), indent=2))
    result = {"args": asdict(a),
              "protocol": "single paired task seed; sustained validation selects; untouched test reports; no cross-seed inference",
              "run_dir": str(root), "zero_test_mse": float(yt.square().mean()), "methods": {}}
    parameters = sum(w.numel() for w in initial)
    covariance_bytes = len(a.prior_scales) * parameters * parameters * 4
    result["resources"] = {
        "parameters": parameters,
        "minimum_covariance_traffic_bytes_per_full_arm": 3 * a.samples * covariance_bytes,
        "full_covariance_grid_bytes": covariance_bytes,
        "full_covariance_snapshot_pair_bytes": 2 * covariance_bytes,
        "full_covariance_graph_steps": 1,
        "estimated_full_arm_memory_budget_gib": 16,
        "memory_estimate_is_not_measurement": True,
        "cost": "O(grid * parameters^2) per sample; full FP32, no low-rank/CPU fallback",
        "precision": "v1 task generation; FP32 learner/evaluation products without TF32",
    }
    for method in a.methods:
        grid = a.adam_lrs if method == "adam" else a.prior_scales
        torch.cuda.reset_peak_memory_stats()
        learner = Learner(method, grid, initial, a, xs, ys, clean, sigma.square())
        started = time.perf_counter()
        graph, parity = learner.capture()
        startup = time.perf_counter() - started
        torch.cuda.synchronize()
        started = time.perf_counter()
        curves, snapshots, intervals = [], [], []
        previous_step = 0
        for step in range(a.graph_steps, a.samples + 1, a.graph_steps):
            for _ in range(a.graph_steps // learner.capture_steps):
                graph.replay()
            if step % a.log_every == 0 or step == a.samples or step == switch:
                current = 1 if step > switch else 0
                validation = evaluate(learner.weights, xv, yv[current]).cpu().tolist()
                online = (learner.error / step).cpu().tolist()
                gain = (learner.gain_sum / step).cpu().tolist()
                dispersion = (learner.unit_variance / step).sqrt().cpu().tolist()
                curves.append({"step": step, "validation": validation, "online_clean_mse": online,
                               "gain": gain, "unit_gain_sd": dispersion})
                snapshots.append([w.clone() for w in learner.weights])
                intervals.append(step - previous_step)
                previous_step = step
                for n, value in enumerate(grid):
                    tag = f"{method}/{value:g}"
                    writer.add_scalar(f"validation/{tag}", validation[n], step)
                    writer.add_scalar(f"online_clean/{tag}", online[n], step)
                    writer.add_scalar(f"plasticity_gain/{tag}", gain[n], step)
                    writer.add_scalar(f"unit_gain_sd/{tag}", dispersion[n], step)
                print(json.dumps(finite_json({"method": method, **curves[-1]})), flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        val = [sum(c["validation"][n] * duration for c, duration in zip(curves, intervals)) / a.samples
               for n in range(len(grid))]
        candidates = [n for n, value in enumerate(val) if math.isfinite(value)]
        best = min(candidates, key=val.__getitem__) if candidates else None
        # Persist the decision before observing any test errors, including endpoints.
        (root / f"selection_{method}.json").write_text(json.dumps(finite_json({
            "validation_sustained_grid": val, "chosen_index": best}), allow_nan=False) + "\n")
        test_curve = []
        if best is not None:
            for checkpoint, params in zip(curves, snapshots):
                target = test_targets[1 if checkpoint["step"] > switch else 0]
                chosen = [w[best:best + 1] for w in params]
                error = float(evaluate(chosen, xt, target)[0])
                test_curve.append({"step": checkpoint["step"], "mse": error})
                writer.add_scalar(f"test/{method}/clean_mse", error, checkpoint["step"])
        test_mse = test_curve[-1]["mse"] if test_curve else None
        test_sustained = (sum(c["mse"] * duration for c, duration in zip(test_curve, intervals)) / a.samples
                          if test_curve else None)
        row = {"grid": list(grid), "parameter": "learning_rate" if method == "adam" else "prior_scale",
               "chosen": None if best is None else grid[best],
               "edge": best is None or best in (0, len(grid) - 1),
               "validation_sustained_grid": val, "test_mse": test_mse,
               "test_sustained_mse": test_sustained, "test_curve": test_curve,
               "test_over_zero": None if test_mse is None else test_mse / result["zero_test_mse"],
               "startup_seconds": startup, "training_seconds": elapsed,
               "aggregate_samples_per_second": a.samples * len(grid) / elapsed,
               "graph_parity_max_abs": parity, "curves": curves}
        row["unit_gain_semantics"] = "signed per-row contribution to the mean update's linearized output change"
        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        row["geometry_probe"] = geometry_probe(learner, xv[:128])
        result["methods"][method] = finite_json(row)
        (root / "results.json").write_text(json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n")
        print("RESULT " + json.dumps(finite_json({"method": method, **{k: v for k, v in row.items() if k != "curves"}})), flush=True)
        del graph, learner
    writer.close()
    print(f"RESULTS {root / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

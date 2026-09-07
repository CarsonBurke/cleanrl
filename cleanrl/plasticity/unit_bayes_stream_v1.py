"""Neuron-block Bayesian plasticity, v1: state-conditioned updates, not weight masks.

Each perceptron owns a covariance P over its incoming weights (bias included).
On THIS sample its sensitivity j=dy/dz and input x select P x; uncertainty
u=j² x'P x determines how much of the output innovation the neuron absorbs.
The denominator R + sum(u) accounts for every block's uncertainty, so simultaneous
updates do not each spend the whole residual. P is conditioned after each sample.
This is a block-diagonal extended Kalman approximation, not a new Kalman algorithm.
Off-block posterior correlations and second derivatives of the network are omitted.
Hypothesis: uncertainty in a neuron's CURRENT receptive-field direction is a better
plasticity signal than the marginal energy/significance of its past gradients.

Dense 17-64-64-1 regression, paired seed 1, full 65,536-sample streams. Adam is
LR-swept; Bayesian arms sweep their prior scale (not falsely called Adam's LR).
Controls replace neuron covariances by their layer mean, cyclically permute their
attachment, or remove their directional component. These are independently trained
ablations, not counterfactual trajectories. Known-noise arms are privileged references,
NOT bounds. Default observation variance is a causal EMA of previous squared residuals;
it includes model error, so it is explicitly an innovation-scale approximation.

Validation chooses hyperparameters; untouched test reports them. Online clean-target
error and current-teacher validation curves prevent endpoint-only switch claims.
All configs share examples, initialization and held-out splits. No clean target enters
an ordinary optimizer. CUDA-only compiled steps and explicit CUDA graphs; graph warmup
is rolled back, including counters. FP32 covariance updates are retained for positive-
definiteness fidelity; TF32 accelerates matmuls, with parity checked before training.
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
    methods: tuple[str, ...] = ("adam", "unit", "shared", "shuffle", "scalar")
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
    if set(a.methods) - {"adam", "unit", "shared", "shuffle", "scalar"}:
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
        else:
            self.cov, self.process = [], []
            for w in self.weights:
                d = w.shape[-1]
                # Prior variance is fan-in normalized: each row's initial functional
                # uncertainty is comparable across layers, rather than width times scale.
                p = (self.scale / d).view(k, 1, 1, 1) * torch.eye(d, device=device).view(1, 1, d, d)
                self.cov.append(p.expand(k, w.shape[1], d, d).clone())
                self.process.append(a.diffusion * p)
            self.mutable += self.cov

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
            mapped, uncertainty, actual_p = [], [], []
            for p, q, inp, j in zip(self.cov, self.process, inputs, sensitivities):
                p.add_(q)
                if self.method == "shared":
                    applied = p.mean(1, keepdim=True).expand_as(p)
                elif self.method == "shuffle":
                    applied = torch.roll(p, 1, dims=1)
                else:
                    applied = p
                px = torch.einsum("koij,kj->koi", applied, inp)
                leverage = (px * inp.unsqueeze(1)).sum(-1)
                if self.method == "scalar":
                    px = (leverage / inp.square().sum(-1, keepdim=True)).unsqueeze(-1) * inp.unsqueeze(1)
                mapped.append(px)
                uncertainty.append(j.square() * leverage)
                actual_p.append(applied)
            observation = (self.noise_var.index_select(0, ix).squeeze(0)
                           if self.a.known_noise else self.noise)
            total = sum(u.sum(-1) for u in uncertainty)
            denominator = observation + total + 1e-8
            gains = torch.cat([u / denominator.unsqueeze(-1) for u in uncertainty], -1)
            self.gain_sum.add_(gains.sum(-1))
            self.unit_variance.add_(gains.var(-1, unbiased=False))
            for w, p, applied, px, inp, j in zip(
                    self.weights, self.cov, actual_p, mapped, inputs, sensitivities):
                w.sub_(residual[:, None, None] * j.unsqueeze(-1) * px / denominator[:, None, None])
                # Scalar control changes ONLY the mean update, not its covariance
                # model; the covariance is still conditioned with its real P x.
                true_px = (torch.einsum("koij,kj->koi", applied, inp)
                           if self.method == "scalar" else px)
                reduction = (j.square() / denominator.unsqueeze(-1))[:, :, None, None]
                posterior = applied - reduction * true_px.unsqueeze(-1) * true_px.unsqueeze(-2)
                # For the shuffle control, conditioned blocks return to their owners.
                if self.method == "shuffle":
                    posterior = torch.roll(posterior, -1, dims=1)
                p.copy_(posterior)
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
        for _ in range(self.a.graph_steps):
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
            for _ in range(self.a.graph_steps):
                compiled()
        self.restore(state)
        graph.replay()
        torch.cuda.synchronize()
        # TF32 arithmetic and compiled reductions may reorder accumulation, not semantics.
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
    yt = teach(t2 if a.switch_at else t1, xt)
    root = Path(a.output or f"runs/DenseStream__unit_bayes_v1__{a.seed}__{time.time_ns()}")
    root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(root))
    writer.add_text("hyperparameters", json.dumps(asdict(a), indent=2))
    result = {"args": asdict(a), "protocol": "single paired task seed; validation-selected test error; no cross-seed inference",
              "run_dir": str(root), "zero_test_mse": float(yt.square().mean()), "methods": {}}
    for method in a.methods:
        grid = a.adam_lrs if method == "adam" else a.prior_scales
        learner = Learner(method, grid, initial, a, xs, ys, clean, sigma.square())
        started = time.perf_counter()
        graph, parity = learner.capture()
        startup = time.perf_counter() - started
        torch.cuda.synchronize()
        started = time.perf_counter()
        curves = []
        for step in range(a.graph_steps, a.samples + 1, a.graph_steps):
            graph.replay()
            if step % a.log_every == 0 or step == a.samples or step == switch:
                current = 1 if step > switch else 0
                validation = evaluate(learner.weights, xv, yv[current]).cpu().tolist()
                online = (learner.error / step).cpu().tolist()
                gain = (learner.gain_sum / step).cpu().tolist()
                dispersion = (learner.unit_variance / step).sqrt().cpu().tolist()
                curves.append({"step": step, "validation": validation, "online_clean_mse": online,
                               "gain": gain, "unit_gain_sd": dispersion})
                for n, value in enumerate(grid):
                    tag = f"{method}/{value:g}"
                    writer.add_scalar(f"validation/{tag}", validation[n], step)
                    writer.add_scalar(f"online_clean/{tag}", online[n], step)
                    writer.add_scalar(f"plasticity_gain/{tag}", gain[n], step)
                    writer.add_scalar(f"unit_gain_sd/{tag}", dispersion[n], step)
                print(json.dumps(finite_json({"method": method, **curves[-1]})), flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        val = evaluate(learner.weights, xv, yv[1 if a.switch_at else 0])
        finite = torch.isfinite(val)
        if not bool(finite.any()):
            best = None
            test_mse = None
        else:
            best = int(torch.where(finite, val, torch.inf).argmin())
            chosen = [w[best:best + 1] for w in learner.weights]
            test_mse = float(evaluate(chosen, xt, yt)[0])
        row = {"grid": list(grid), "parameter": "learning_rate" if method == "adam" else "prior_scale",
               "chosen": None if best is None else grid[best],
               "edge": best is None or best in (0, len(grid) - 1),
               "validation_mse": val.cpu().tolist(), "test_mse": test_mse,
               "test_over_zero": None if test_mse is None else test_mse / result["zero_test_mse"],
               "startup_seconds": startup, "training_seconds": elapsed,
               "aggregate_samples_per_second": a.samples * len(grid) / elapsed,
               "graph_parity_max_abs": parity, "curves": curves}
        result["methods"][method] = finite_json(row)
        (root / "results.json").write_text(json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n")
        print("RESULT " + json.dumps(finite_json({"method": method, **{k: v for k, v in row.items() if k != "curves"}})), flush=True)
        del graph, learner
    writer.close()
    print(f"RESULTS {root / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

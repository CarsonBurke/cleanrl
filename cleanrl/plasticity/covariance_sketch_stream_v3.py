"""Covariance-sketch EKF with exact buffered rank-one conditioning.

The posterior representation is P = diag(D) - U U.T. Adding diagonal process
variance and appending Pj / sqrt(R + j.T Pj) is exactly the dense EKF recurrence
between compressions, not an exact Bayesian posterior for a nonlinear network.
Compression retains the leading modes of U.T diag(D)^-1 U. Dropping covariance
downdates restores uncertainty in PSD order; it never folds lost correlations
into diagonal certainty. D is unchanged by compression.

The scalar control conditions the same posterior at its own current Jacobian,
but erases the mean-update orientation while preserving instantaneous leverage.
Signed per-neuron contributions describe joint posterior coordination, not
independent autonomous scalar gates. Clean targets are diagnostics only.

Execution uses one compiled CUDA update graph replayed buffer times, followed
by separately compiled compression. The eigensolver is deliberately outside
CUDA capture (torch.linalg.eigh may synchronize); there is no eager fallback.
"""

import math
from dataclasses import dataclass

import torch

from cleanrl.plasticity.network_bayes_stream_v2 import Args as ReferenceArgs
from cleanrl.plasticity.network_bayes_stream_v2 import forward, sample_state
from cleanrl.shared import runtime


@dataclass
class Args(ReferenceArgs):
    buffer: int = 64


class _Macrocycle:
    """Replay exactly one full buffer; compression consumes no observations."""

    def __init__(self, step_graph, compression, steps):
        self.step_graph = step_graph
        self.compression = compression
        self.steps = steps
        self.compression_execution = "compiled_outside_cuda_graph"

    def replay(self):
        for _ in range(self.steps):
            self.step_graph.replay()
        self.compression()


class SketchLearner:
    def __init__(self, method, rank, grid, initial, a, xs, ys, clean, noise_var):
        if method not in ("sketch", "sketch_scalar", "diagonal"):
            raise ValueError(f"unknown covariance-sketch method: {method}")
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        if not isinstance(a.buffer, int) or a.buffer <= 0:
            raise ValueError("buffer must be a positive integer")
        if not grid or any(not math.isfinite(v) or v <= 0 for v in grid):
            raise ValueError("prior grid must contain finite positive values")
        if a.diffusion < 0 or not math.isfinite(a.diffusion) or not 0 < a.noise_rate <= 1:
            raise ValueError("require finite nonnegative diffusion and 0 < noise_rate <= 1")
        tensors = [*initial, xs, ys, clean, noise_var]
        if xs.device.type != "cuda" or any(t.device != xs.device for t in tensors):
            raise ValueError("all learner tensors must be on the same CUDA device")
        if any(t.dtype != torch.float32 for t in tensors):
            raise ValueError("covariance-sketch recurrence requires float32 tensors")
        if xs.ndim != 2 or any(t.shape != (len(xs),) for t in (ys, clean, noise_var)):
            raise ValueError("stream inputs must be N x input_dim and targets/noise length N")
        if len(xs) < a.buffer or len(xs) % a.buffer:
            raise ValueError("stream length must be a positive multiple of buffer")
        runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
        self.method, self.a = method, a
        self.rank = 0 if method == "diagonal" else rank
        self.capture_steps = a.buffer
        k, device = len(grid), xs.device
        self.weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
        self.scale = torch.tensor(grid, device=device, dtype=torch.float32)
        self.index = torch.zeros((), dtype=torch.int64, device=device)
        self.offset = torch.zeros((), dtype=torch.int64, device=device)
        self.steps = torch.zeros((), device=device)
        self.error = torch.zeros(k, device=device)
        self.null_error = torch.zeros((), device=device)
        self.noise = torch.ones(k, device=device)
        self.gain_sum = torch.zeros(k, device=device)
        self.unit_variance = torch.zeros(k, device=device)
        self.xs, self.ys, self.clean, self.noise_var = xs, ys, clean, noise_var
        prior = torch.cat([
            (self.scale / w.shape[-1])[:, None].expand(k, w[0].numel())
            for w in self.weights], -1)
        self.D = prior.clone()
        self.process = a.diffusion * prior
        self.U = torch.zeros(k, prior.shape[-1], 0 if method == "diagonal" else rank + a.buffer,
                             device=device, dtype=torch.float32)
        self.mutable = [*self.weights, self.index, self.offset, self.steps, self.error,
                        self.null_error, self.noise, self.gain_sum, self.unit_variance,
                        self.D, self.U]

    def posterior_product(self, jacobian):
        """Apply the represented covariance without allocating a dense matrix."""
        pj = self.D * jacobian
        if self.method != "diagonal":
            coordinates = torch.bmm(self.U.transpose(1, 2), jacobian.unsqueeze(-1))
            pj = pj - torch.bmm(self.U, coordinates).squeeze(-1)
        return pj

    @torch.no_grad()
    def update(self):
        """Consume one observation; caller compresses after capture_steps updates."""
        ix = self.index.reshape(1)
        x = self.xs.index_select(0, ix).squeeze(0)
        y = self.ys.index_select(0, ix).squeeze(0)
        target = self.clean.index_select(0, ix).squeeze(0)
        prediction, inputs, sensitivities = sample_state(self.weights, x)
        residual = prediction - y
        jacobian = torch.cat([
            (j.unsqueeze(-1) * inp.unsqueeze(1)).flatten(1)
            for inp, j in zip(inputs, sensitivities)], -1)
        self.D.add_(self.process)
        pj = self.posterior_product(jacobian)
        leverage = (jacobian * pj).sum(-1)
        observation = (self.noise_var.index_select(0, ix).squeeze(0)
                       if self.a.known_noise else self.noise)
        denominator = observation + leverage
        if self.method == "sketch_scalar":
            # The output-bias Jacobian is one, so the norm is strictly positive.
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
        if self.method == "diagonal":
            self.D.sub_(pj.square() / denominator.unsqueeze(-1))
        else:
            column = (pj / denominator.sqrt().unsqueeze(-1)).unsqueeze(-1)
            self.U.index_copy_(2, (self.rank + self.offset).reshape(1), column)
        gains = torch.cat(credit, -1)
        self.gain_sum.add_(gains.sum(-1))
        self.unit_variance.add_(gains.var(-1, unbiased=False))
        self.error.add_((prediction - target).square())
        self.null_error.add_(target.square())
        # Current residual sets only the NEXT observation's inferred noise scale.
        self.noise.lerp_(residual.square(), self.a.noise_rate)
        self.index.add_(1)
        self.steps.add_(1)
        self.offset.add_(1)

    @torch.no_grad()
    def compress(self):
        """Prior-whitened truncation; discarding PSD downdates increases P."""
        if self.method != "diagonal":
            if self.rank:
                gram = torch.bmm(self.U.transpose(1, 2), self.U / self.D.unsqueeze(-1))
                # Symmetrization only removes GEMM roundoff; eigenvalues and
                # posterior leverage are never clipped to conceal PSD failures.
                _, vectors = torch.linalg.eigh((gram + gram.transpose(1, 2)) * 0.5)
                retained = torch.bmm(self.U, vectors[:, :, -self.rank:])
                self.U[:, :, :self.rank].copy_(retained)
            self.U[:, :, self.rank:].zero_()
        self.offset.zero_()

    def snapshot(self):
        return [t.clone() for t in self.mutable]

    @torch.no_grad()
    def restore(self, values):
        if len(values) != len(self.mutable):
            raise ValueError("snapshot does not match learner state")
        for tensor, value in zip(self.mutable, values):
            tensor.copy_(value)

    @torch.no_grad()
    def capture(self):
        """Audit compiled replay and restore every mutable tensor on success/failure.

        Only setup/audit uses eager execution. Production replay is always the
        captured compiled step plus compiled compression, without eager fallback.
        Factor bases can rotate in degenerate eigenspaces, so post-compression
        parity compares covariance actions on the union of the factor spans.
        """
        if self.offset.item() != 0 or self.index.item() + self.capture_steps > len(self.xs):
            raise ValueError("capture requires a buffer boundary with a full macrocycle remaining")
        state = self.snapshot()
        stream = torch.cuda.Stream(device=self.xs.device)
        try:
            for _ in range(self.capture_steps):
                self.update()
            before_compression = self.snapshot()
            self.compress()
            expected = self.snapshot()
            self.restore(state)
            compiled_step = torch.compile(self.update, fullgraph=True, mode="max-autotune-no-cudagraphs")
            compiled_compress = torch.compile(self.compress, fullgraph=True, mode="max-autotune-no-cudagraphs")
            stream.wait_stream(torch.cuda.current_stream(self.xs.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    compiled_step()
                    self.restore(before_compression)
                    compiled_compress()
                    self.restore(state)
            torch.cuda.current_stream(self.xs.device).wait_stream(stream)
            torch.cuda.synchronize(self.xs.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                compiled_step()
            self.restore(state)
            for _ in range(self.capture_steps):
                graph.replay()
            torch.cuda.synchronize(self.xs.device)
            max_error = 0.0
            for actual, reference in zip(self.mutable, before_compression):
                torch.testing.assert_close(actual, reference, rtol=3e-3, atol=3e-5)
                if actual.numel():
                    max_error = max(max_error, float((actual - reference).abs().max()))
            compiled_compress()
            torch.cuda.synchronize(self.xs.device)
            for actual, reference in zip(self.mutable, expected):
                if actual is self.U:
                    # D was compared separately. Equality on this spanning set
                    # establishes equality of the represented low-rank operator.
                    basis = torch.cat((actual, reference), -1)
                    actual = torch.bmm(actual, torch.bmm(actual.transpose(1, 2), basis))
                    reference = torch.bmm(reference, torch.bmm(reference.transpose(1, 2), basis))
                torch.testing.assert_close(actual, reference, rtol=3e-3, atol=3e-5)
                if actual.numel():
                    max_error = max(max_error, float((actual - reference).abs().max()))
            return _Macrocycle(graph, compiled_compress, self.capture_steps), max_error
        finally:
            torch.cuda.current_stream(self.xs.device).wait_stream(stream)
            self.restore(state)

    @torch.no_grad()
    def geometry(self, x=None):
        """Frozen next-state geometry; probes do not certify PSD in every direction."""
        if x is None:
            x = self.xs[:min(128, len(self.xs))]
        h1, h2, prediction = forward(self.weights, x)
        one = torch.ones_like(prediction).unsqueeze(-1)
        inputs = (torch.cat((x.unsqueeze(0).expand(prediction.shape[0], -1, -1), one), -1),
                  torch.cat((h1, one), -1), torch.cat((h2, one), -1))
        j2 = self.weights[2][:, None, 0, :-1] * (1 - h2.square())
        j1 = torch.einsum("kbo,koi->kbi", j2, self.weights[1][..., :-1]) * (1 - h1.square())
        jacobian = torch.cat([(j.unsqueeze(-1) * inp.unsqueeze(-2)).flatten(2)
                              for inp, j in zip(inputs, (j1, j2, one))], -1)
        pj = (self.D + self.process).unsqueeze(1) * jacobian
        diagonal = self.D
        if self.method != "diagonal":
            projected = torch.bmm(jacobian, self.U)
            pj = pj - torch.bmm(projected, self.U.transpose(1, 2))
            diagonal = diagonal - self.U.square().sum(-1)
        actual = (jacobian * pj).sum(-1)
        trace = diagonal.sum(-1)
        predicted_trace = trace + self.process.sum(-1)
        isotropic = predicted_trace[:, None] / self.D.shape[-1] * jacobian.square().sum(-1)
        relative = actual / isotropic
        downdate_trace = self.U.square().sum((1, 2))
        if self.method == "diagonal":
            minimum_whitened = maximum_whitened = torch.zeros_like(trace)
        else:
            gram = torch.bmm(self.U.transpose(1, 2), self.U / self.D.unsqueeze(-1))
            eigenvalues = torch.linalg.eigvalsh((gram + gram.transpose(1, 2)) * 0.5)
            minimum_whitened, maximum_whitened = eigenvalues[:, 0], eigenvalues[:, -1]
        return {
            "posterior_trace": trace.cpu().tolist(),
            "prediction_trace": predicted_trace.cpu().tolist(),
            "base_diagonal_trace": self.D.sum(-1).cpu().tolist(),
            "covariance_downdate_trace": downdate_trace.cpu().tolist(),
            "minimum_whitened_eigenvalue": minimum_whitened.cpu().tolist(),
            "maximum_whitened_eigenvalue": maximum_whitened.cpu().tolist(),
            "minimum_posterior_diagonal": diagonal.amin(-1).cpu().tolist(),
            "state_geometry_sd": relative.std(1, unbiased=False).cpu().tolist(),
            "minimum_directional_variance": actual.amin(1).cpu().tolist(),
            "maximum_directional_variance": actual.amax(1).cpu().tolist(),
        }

"""Candidate-owned FP32 Triton execution of the frozen v2 transport equations.

One CTA retains an entire three-layer network and its Adam state across a dynamic
sample loop. No covariance, approximate matmul, or eager production path exists.
The v2 eager implementation is instantiated only for capture's local audit.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from cleanrl.plasticity import predictive_transport_stream_v2 as reference


METHODS = reference.METHODS


@triton.jit
def _sum_matrix(value):
    return tl.sum(tl.sum(value, axis=1), axis=0)


@triton.jit
def _finite_matrix(value):
    return _sum_matrix((tl.abs(value) < float("inf")).to(tl.int32)) == value.numel


@triton.jit
def _finite_vector(value):
    return tl.sum((tl.abs(value) < float("inf")).to(tl.int32), axis=0) == value.numel


@triton.jit
def _forward(w0, w1, w2, x, I: tl.constexpr, H1: tl.constexpr, H2: tl.constexpr,
             R0: tl.constexpr, C0: tl.constexpr, R1: tl.constexpr,
             C1: tl.constexpr, C2: tl.constexpr):
    c0 = tl.arange(0, C0)
    c1 = tl.arange(0, C1)
    c2 = tl.arange(0, C2)
    h1 = libdevice.tanh(
        tl.sum(tl.where(c0[None, :] < I, w0 * x[None, :], 0.0), axis=1)
        + tl.sum(tl.where(c0[None, :] == I, w0, 0.0), axis=1))
    h1 = tl.where(tl.arange(0, R0) < H1, h1, 0.0)
    h1_columns = tl.gather(h1, tl.minimum(c1, R0 - 1), axis=0)
    h1_columns = tl.where(c1 < H1, h1_columns, 0.0)
    h2 = libdevice.tanh(
        tl.sum(tl.where(c1[None, :] < H1, w1 * h1_columns[None, :], 0.0), axis=1)
        + tl.sum(tl.where(c1[None, :] == H1, w1, 0.0), axis=1))
    h2 = tl.where(tl.arange(0, R1) < H2, h2, 0.0)
    h2_columns = tl.gather(h2, tl.minimum(c2, R1 - 1), axis=0)
    prediction = (tl.sum(tl.where(c2 < H2, w2 * h2_columns, 0.0), axis=0)
                  + tl.sum(tl.where(c2 == H2, w2, 0.0), axis=0))
    return prediction, h1, h2


@triton.jit
def _moment_update(m, v, jacobian, mask, score, coefficient, beta, lr,
                   old_mass, mass, variance_mass):
    # Zero padding explicitly: zero times a nonfinite coefficient is not zero.
    gradient = tl.where(mask, score * jacobian, 0.0)
    correction = tl.where(mask, coefficient * jacobian, 0.0)
    next_m = beta * m + (1.0 - beta) * gradient + (beta * old_mass) * correction
    next_v = 0.999 * v + 0.001 * (gradient * gradient)
    d = tl.div_rn(1.0, libdevice.sqrt(tl.div_rn(next_v, variance_mass)) + 1.0e-8)
    u = (-lr * tl.div_rn(next_m, mass)) * d
    return next_m, next_v, u, d, correction * correction


@triton.jit
def _transport_kernel(
    W0, W1, W2, P0, P1, P2, M0, M1, M2, V0, V1, V2,
    LR, BETA, XS, YS, CLEAN, INDICES, STEPS, ERROR, NULL_ERROR,
    CORRECTION_NORM, FINITE,
    I: tl.constexpr, H1: tl.constexpr, H2: tl.constexpr,
    R0: tl.constexpr, C0: tl.constexpr, R1: tl.constexpr,
    C1: tl.constexpr, C2: tl.constexpr,
    METHOD: tl.constexpr, N_STEPS: tl.constexpr,
):
    candidate = tl.program_id(0)
    r0, c0 = tl.arange(0, R0), tl.arange(0, C0)
    r1, c1 = tl.arange(0, R1), tl.arange(0, C1)
    c2 = tl.arange(0, C2)
    mask0 = (r0[:, None] < H1) & (c0[None, :] <= I)
    mask1 = (r1[:, None] < H2) & (c1[None, :] <= H1)
    mask2 = c2 <= H2
    offset0 = candidate * H1 * (I + 1) + r0[:, None] * (I + 1) + c0[None, :]
    offset1 = candidate * H2 * (H1 + 1) + r1[:, None] * (H1 + 1) + c1[None, :]
    offset2 = candidate * (H2 + 1) + c2
    w0 = tl.load(W0 + offset0, mask0, other=0.0)
    w1 = tl.load(W1 + offset1, mask1, other=0.0)
    w2 = tl.load(W2 + offset2, mask2, other=0.0)
    m0 = tl.load(M0 + offset0, mask0, other=0.0)
    m1 = tl.load(M1 + offset1, mask1, other=0.0)
    m2 = tl.load(M2 + offset2, mask2, other=0.0)
    v0 = tl.load(V0 + offset0, mask0, other=0.0)
    v1 = tl.load(V1 + offset1, mask1, other=0.0)
    v2 = tl.load(V2 + offset2, mask2, other=0.0)
    if METHOD != "adam":
        p0 = tl.load(P0 + offset0, mask0, other=0.0)
        p1 = tl.load(P1 + offset1, mask1, other=0.0)
        p2 = tl.load(P2 + offset2, mask2, other=0.0)
    lr = tl.load(LR + candidate)
    beta = tl.load(BETA + candidate)
    log_beta = libdevice.log(tl.where(beta == 0.0, 1.0, beta))
    ix = tl.load(INDICES + candidate)
    error = tl.load(ERROR + candidate)
    correction_norm = tl.load(CORRECTION_NORM + candidate)
    finite = tl.load(FINITE + candidate)
    # Other CTAs neither read nor write reporting scalars. Their optimizer clock
    # is their own index, so there is no cross-CTA read-after-write index race.
    null_error = tl.full((), 0.0, tl.float32)
    if candidate == 0:
        null_error = tl.load(NULL_ERROR)

    for _ in range(N_STEPS):
        x = tl.load(XS + ix * I + c0, c0 < I, other=0.0)
        y = tl.load(YS + ix)
        target = tl.load(CLEAN + ix)
        prediction, h1, h2 = _forward(w0, w1, w2, x, I, H1, H2, R0, C0, R1, C1, C2)
        out_weights = tl.gather(w2, tl.arange(0, R1), axis=0)
        sensitivity2 = tl.where(r1 < H2, out_weights * (1.0 - h2 * h2), 0.0)
        sensitivity1_columns = tl.sum(
            tl.where(c1[None, :] < H1, sensitivity2[:, None] * w1, 0.0), axis=0)
        sensitivity1 = tl.gather(sensitivity1_columns, r0, axis=0) * (1.0 - h1 * h1)
        sensitivity1 = tl.where(r0 < H1, sensitivity1, 0.0)
        input0 = tl.where(c0 == I, 1.0, x)
        input1 = tl.gather(h1, tl.minimum(c1, R0 - 1), axis=0)
        input1 = tl.where(c1 == H1, 1.0, tl.where(c1 < H1, input1, 0.0))
        input2 = tl.gather(h2, tl.minimum(c2, R1 - 1), axis=0)
        input2 = tl.where(c2 == H2, 1.0, tl.where(c2 < H2, input2, 0.0))
        j0 = tl.where(mask0, sensitivity1[:, None] * input0[None, :], 0.0)
        j1 = tl.where(mask1, sensitivity2[:, None] * input1[None, :], 0.0)
        j2 = input2
        residual = prediction - y
        if METHOD == "robust":
            score = tl.div_rn(residual, libdevice.hypot(residual, 1.0))
        else:
            score = residual
        if METHOD == "adam":
            coefficient = tl.full((), 0.0, tl.float32)
        elif METHOD == "tangent":
            coefficient = (_sum_matrix(j0 * (w0 - p0)) + _sum_matrix(j1 * (w1 - p1))
                           + tl.sum(j2 * (w2 - p2), axis=0))
        else:
            previous_prediction, _, _ = _forward(p0, p1, p2, x, I, H1, H2, R0, C0, R1, C1, C2)
            if METHOD == "robust":
                previous_residual = previous_prediction - y
                coefficient = score - tl.div_rn(previous_residual, libdevice.hypot(previous_residual, 1.0))
            else:
                coefficient = prediction - previous_prediction
        clean_residual = prediction - target
        error += clean_residual * clean_residual
        if candidate == 0:
            null_error += target * target
        t = (ix + 1).to(tl.float32)
        old_mass = tl.where(beta == 0.0, (t > 1.0).to(tl.float32), -libdevice.expm1(log_beta * (t - 1.0)))
        mass = tl.where(beta == 0.0, 1.0, -libdevice.expm1(log_beta * t))
        variance_mass = -libdevice.expm1(-0.0010005003335835344 * t)
        m0, v0, u0, d0, energy0 = _moment_update(
            m0, v0, j0, mask0, score, coefficient, beta, lr, old_mass, mass, variance_mass)
        m1, v1, u1, d1, energy1 = _moment_update(
            m1, v1, j1, mask1, score, coefficient, beta, lr, old_mass, mass, variance_mass)
        m2, v2, u2, d2, energy2 = _moment_update(
            m2, v2, j2, mask2, score, coefficient, beta, lr, old_mass, mass, variance_mass)
        correction_energy = _sum_matrix(energy0) + _sum_matrix(energy1) + tl.sum(energy2, axis=0)
        correction_norm += libdevice.sqrt(correction_energy)
        if METHOD == "implicit":
            h = _sum_matrix(j0 * u0) + _sum_matrix(j1 * u1) + tl.sum(j2 * u2, axis=0)
            ell = (_sum_matrix((j0 * j0) * d0) + _sum_matrix((j1 * j1) * d1)
                   + tl.sum((j2 * j2) * d2, axis=0))
            scale = tl.div_rn(h, 1.0 + lr * ell)
            u0 = u0 - ((lr * d0) * j0) * scale
            u1 = u1 - ((lr * d1) * j1) * scale
            u2 = u2 - ((lr * d2) * j2) * scale
        # Check each transition before a later sample can overwrite evidence.
        finite = (finite & _finite_matrix(m0) & _finite_matrix(v0)
                  & _finite_matrix(m1) & _finite_matrix(v1)
                  & _finite_vector(m2) & _finite_vector(v2))
        if METHOD != "adam":
            finite = finite & _finite_matrix(p0) & _finite_matrix(p1) & _finite_vector(p2)
            # These SSA values are pre-update current weights, not next weights.
            p0, p1, p2 = w0, w1, w2
        w0 = tl.where(mask0, w0 + u0, 0.0)
        w1 = tl.where(mask1, w1 + u1, 0.0)
        w2 = tl.where(mask2, w2 + u2, 0.0)
        finite = (finite & _finite_matrix(w0) & _finite_matrix(w1) & _finite_vector(w2)
                  & (tl.abs(error) < float("inf")) & (tl.abs(correction_norm) < float("inf")))
        ix += 1

    # Store only after the complete sample loop. No global intermediate buffers.
    tl.store(W0 + offset0, w0, mask0)
    tl.store(W1 + offset1, w1, mask1)
    tl.store(W2 + offset2, w2, mask2)
    tl.store(M0 + offset0, m0, mask0)
    tl.store(M1 + offset1, m1, mask1)
    tl.store(M2 + offset2, m2, mask2)
    tl.store(V0 + offset0, v0, mask0)
    tl.store(V1 + offset1, v1, mask1)
    tl.store(V2 + offset2, v2, mask2)
    if METHOD != "adam":
        tl.store(P0 + offset0, p0, mask0)
        tl.store(P1 + offset1, p1, mask1)
        tl.store(P2 + offset2, p2, mask2)
    tl.store(INDICES + candidate, ix)
    tl.store(ERROR + candidate, error)
    tl.store(CORRECTION_NORM + candidate, correction_norm)
    tl.store(FINITE + candidate, finite)
    if candidate == 0:
        tl.store(STEPS, ix.to(tl.float32))
        tl.store(NULL_ERROR, null_error)


class Learner:
    def __init__(self, method, grid, initial, a, xs, ys, clean, noise_var):
        if method not in METHODS:
            raise ValueError(f"unknown transport method: {method}")
        if not grid or any(not math.isfinite(lr) or lr <= 0 or not math.isfinite(beta)
                           or not 0 <= beta < 1 - 2**-25 for lr, beta in grid):
            raise ValueError("grid requires positive finite learning rates and beta1 in [0, 1) after FP32 rounding")
        if xs.device.type != "cuda" or xs.dtype != torch.float32:
            raise ValueError("predictive transport requires CUDA FP32 inputs")
        if any(t.device != xs.device or t.dtype != torch.float32 for t in [*initial, ys, clean]):
            raise ValueError("weights and observations must share the CUDA FP32 input device")
        if a.graph_steps <= 0:
            raise ValueError("graph_steps must be positive")
        if (len(initial) != 3 or xs.ndim != 2 or ys.shape != (xs.shape[0],)
                or clean.shape != ys.shape or any(w.ndim != 2 for w in initial)):
            raise ValueError("expected three dense weight matrices and scalar stream labels")
        h1, i = initial[0].shape[0], xs.shape[1]
        h2 = initial[1].shape[0]
        if (min(i, h1, h2) < 1 or initial[0].shape != (h1, i + 1)
                or initial[1].shape != (h2, h1 + 1) or initial[2].shape != (1, h2 + 1)):
            raise ValueError("weights must describe an input-hidden-hidden-1 network with final-column biases")
        if not all(t.is_contiguous() for t in (xs, ys, clean)):
            raise ValueError("fused transport requires contiguous stream tensors")
        self.method, self.a, self.grid = method, a, list(grid)
        self.capture_steps = a.graph_steps
        self.capture_update_calls = 5 * self.capture_steps + 2
        self.forward_evaluations_per_step = 1 if method in ("adam", "tangent") else 2
        self.jacobian_evaluations_per_step = 1
        self.xs, self.ys, self.clean = xs, ys, clean
        self.capture_failed_candidates = [False] * len(grid)
        k, device = len(grid), xs.device
        self.lr = torch.tensor([lr for lr, _ in grid], dtype=torch.float32, device=device)[:, None, None]
        self.beta = torch.tensor([beta for _, beta in grid], dtype=torch.float32, device=device)[:, None, None]
        self.weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
        self.previous_weights = [] if method == "adam" else [w.clone() for w in self.weights]
        self.m = [torch.zeros_like(w) for w in self.weights]
        self.v = [torch.zeros_like(w) for w in self.weights]
        self.candidate_indices = torch.zeros(k, dtype=torch.int64, device=device)
        self.index = self.candidate_indices[0]
        self.steps = torch.zeros((), dtype=torch.float32, device=device)
        self.error = torch.zeros(k, dtype=torch.float32, device=device)
        self.null_error = torch.zeros((), dtype=torch.float32, device=device)
        self.correction_norm_sum = torch.zeros(k, dtype=torch.float32, device=device)
        self.finite_candidates = torch.ones(k, dtype=torch.bool, device=device)
        # Include the owned vector, not its redundant scalar alias, in snapshots.
        self.mutable = [*self.weights, *self.previous_weights, *self.m, *self.v,
                        self.candidate_indices, self.steps, self.error, self.null_error,
                        self.correction_norm_sum, self.finite_candidates]
        self._dimensions = dict(I=i, H1=h1, H2=h2, R0=triton.next_power_of_2(h1),
                                C0=triton.next_power_of_2(i + 1), R1=triton.next_power_of_2(h2),
                                C1=triton.next_power_of_2(h1 + 1), C2=triton.next_power_of_2(h2 + 1))
        # Large networks need enough threads to keep the four persistent arrays
        # plus live Jacobians and increments in registers rather than local spills.
        padded = self._dimensions["R0"] * self._dimensions["C0"] + self._dimensions["R1"] * self._dimensions["C1"]
        self._num_warps = 16 if padded > 4096 else 4

    def _launch(self, n_steps):
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError("fused sample count must be a positive integer")
        previous = self.previous_weights if self.previous_weights else self.weights
        _transport_kernel[(len(self.grid),)](
            *self.weights, *previous, *self.m, *self.v, self.lr, self.beta,
            self.xs, self.ys, self.clean, self.candidate_indices, self.steps,
            self.error, self.null_error, self.correction_norm_sum, self.finite_candidates,
            **self._dimensions, METHOD=self.method, N_STEPS=n_steps,
            num_warps=self._num_warps, enable_fp_fusion=False)

    @torch.no_grad()
    def update(self):
        self._launch(1)

    def snapshot(self):
        return [tensor.clone() for tensor in self.mutable]

    @torch.no_grad()
    def restore(self, values):
        if len(values) != len(self.mutable):
            raise ValueError("snapshot does not match learner state")
        for tensor, value in zip(self.mutable, values):
            tensor.copy_(value)

    @torch.no_grad()
    def _copy_to_reference(self, oracle):
        for name in ("weights", "previous_weights", "m", "v"):
            for destination, source in zip(getattr(oracle, name), getattr(self, name)):
                destination.copy_(source)
        for name in ("index", "steps", "error", "null_error", "correction_norm_sum", "finite_candidates"):
            getattr(oracle, name).copy_(getattr(self, name))

    def _reference_snapshot(self, oracle):
        state = []
        for name in ("weights", "previous_weights", "m", "v"):
            state.extend(tensor.clone() for tensor in getattr(oracle, name))
        state.extend((oracle.index.expand_as(self.candidate_indices).clone(), oracle.steps.clone(),
                      oracle.error.clone(), oracle.null_error.clone(), oracle.correction_norm_sum.clone(),
                      oracle.finite_candidates.clone()))
        return state

    def _assert_state(self, expected, *, exact, excluded=None):
        # Validity disagreement is a numerical defect, never a reason to silently
        # discard a candidate before checking its finite numerical transition.
        torch.testing.assert_close(self.finite_candidates, expected[-1], rtol=0, atol=0)
        surviving = self.finite_candidates & expected[-1]
        if excluded is not None:
            surviving = surviving & ~excluded
        max_error = 0.0
        for actual, wanted in zip(self.mutable, expected):
            if actual.ndim:
                actual, wanted = actual[surviving], wanted[surviving]
            torch.testing.assert_close(actual, wanted, rtol=0 if exact else 3e-3,
                                       atol=0 if exact else 3e-5)
            if not exact and actual.dtype != torch.bool and actual.numel():
                max_error = max(max_error, float((actual - wanted).abs().max()))
        return surviving, max_error

    def capture(self):
        """Audit eager local steps, fused ordering, then exact production replay.

        The oracle is reset from the production trajectory before every local
        step. A complete fused block must then agree with those single-step
        kernels within the same strict tolerance, and graph replay must match
        that exact fused block bit-for-bit. All entry state is always restored;
        failure metadata is changed only after a successful complete audit.
        """
        state = self.snapshot()
        previous_failures = list(self.capture_failed_candidates)
        success = False
        oracle = None
        try:
            if not bool(torch.all(self.candidate_indices == self.index)):
                raise ValueError("capture's shared-stream eager audit requires equal candidate indices")
            if int(self.index) + self.capture_steps > self.xs.shape[0]:
                raise ValueError("not enough retained samples for capture audit")
            oracle = reference.Learner(self.method, self.grid, [w[0] for w in self.weights],
                                       self.a, self.xs, self.ys, self.clean, None)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            try:
                with torch.cuda.stream(stream):
                    self._launch(1)
                    self.restore(state)
                    self._launch(1)
            finally:
                torch.cuda.current_stream().wait_stream(stream)
            self.restore(state)
            max_error = 0.0
            audit_failed = torch.zeros_like(self.finite_candidates)
            for _ in range(self.capture_steps):
                self._copy_to_reference(oracle)
                oracle.update()
                eager = self._reference_snapshot(oracle)
                self._launch(1)
                surviving, error = self._assert_state(eager, exact=False)
                audit_failed.logical_or_(~surviving)
                max_error = max(max_error, error)
            singles = self.snapshot()
            # The reference does not persist into capture or production replay.
            del oracle
            oracle = None
            self.restore(state)
            self._launch(self.capture_steps)
            surviving, error = self._assert_state(singles, exact=False)
            audit_failed.logical_or_(~surviving)
            max_error = max(max_error, error)
            expected = self.snapshot()
            self.restore(state)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._launch(self.capture_steps)
            self.restore(state)
            graph.replay()
            torch.cuda.synchronize()
            surviving, _ = self._assert_state(expected, exact=True, excluded=audit_failed)
            self.capture_failed_candidates = (~surviving).cpu().tolist()
            success = True
            return graph, max_error
        finally:
            self.restore(state)
            if not success:
                self.capture_failed_candidates = previous_failures
            del oracle

    def diagnostics(self):
        """Checkpoint-only synchronization; candidate-major telemetry, invalid -> null."""
        count = max(float(self.steps), 1.0)

        def values(tensor):
            return [float(value) if math.isfinite(value) else None for value in tensor.detach().cpu().tolist()]

        return {"correction_l2_mean": values(self.correction_norm_sum / count),
                "finite_candidates": self.finite_candidates.cpu().tolist()}

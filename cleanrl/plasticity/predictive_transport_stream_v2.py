"""FP32, matrix-free transport of Adam's existing first-moment mass.

Hypothesis: current-J transport avoids a second backward pass; a frozen-current-J
proximal step can stabilize that transported momentum without covariance memory.
Tangent transport uses J dot (theta - theta_previous), with no previous forward.
Implicit uses delta = u - lr D J (J dot u) / (1 + lr J dot D J), u = -lr D mhat.
Robust uses scale-one pseudo-Huber scores: its transport depends on the observed
label and it optimizes a different objective, not finite-capacity squared error.
Only noisy observations enter learning. Clean targets serve reporting alone.
All learning tensors and model calculations are CUDA FP32; persistent optimizer
storage is linear in parameter count, with no previous Jacobian or outer product.
"""

import math

import torch

from cleanrl.plasticity.network_bayes_stream_v2 import sample_state


METHODS = ("adam", "predictive", "tangent", "implicit", "robust")


def sample_prediction(weights, x):
    """Previous network's scalar output only, with no sensitivities or Jacobians."""
    h1 = torch.tanh(torch.einsum("koi,i->ko", weights[0][..., :-1], x) + weights[0][..., -1])
    h2 = torch.tanh(torch.einsum("koi,ki->ko", weights[1][..., :-1], h1) + weights[1][..., -1])
    return (weights[2][:, 0, :-1] * h2).sum(-1) + weights[2][:, 0, -1]


def robust_score(residual):
    """Derivative of sqrt(1 + residual**2) - 1, stable for large finite residuals."""
    return residual / torch.hypot(residual, torch.ones_like(residual))


def moment_masses(beta, steps):
    """Stable old/current first-moment and current second-moment masses; t >= 1."""
    zero_beta = beta == 0
    log_beta = torch.log(torch.where(zero_beta, torch.ones_like(beta), beta))
    old_mass = torch.where(zero_beta, (steps > 1).to(beta.dtype),
                           -torch.expm1(log_beta * (steps - 1)))
    mass = torch.where(zero_beta, torch.ones_like(beta), -torch.expm1(log_beta * steps))
    variance_mass = -torch.expm1(math.log(0.999) * steps)
    return old_mass, mass, variance_mass


def transported_moments(m, v, gradient, correction, beta, old_mass):
    """Pure EMA transition: transport biased mass; v sees raw observed gradient only."""
    next_m = beta * m + (1 - beta) * gradient + beta * old_mass * correction
    next_v = 0.999 * v + 0.001 * gradient.square()
    return next_m, next_v


def implicit_step(jacobians, unconstrained, preconditioners, lr):
    """Pure frozen-J proximal correction with one global contraction across layers.

    Inputs are candidate-major layer tensors. lr has shape (candidates, 1, 1).
    No outer product is formed; preconditioners are Adam's diagonal D.
    """
    h = sum((j * u).sum((-1, -2)) for j, u in zip(jacobians, unconstrained))
    ell = sum((j.square() * d).sum((-1, -2)) for j, d in zip(jacobians, preconditioners))
    scale = h[:, None, None] / (1 + lr * ell[:, None, None])
    return [u - lr * d * j * scale for j, u, d in zip(jacobians, unconstrained, preconditioners)]


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
        self.method, self.a, self.grid = method, a, list(grid)
        self.capture_steps = a.graph_steps
        self.capture_update_calls = 4 * self.capture_steps + 2
        self.forward_evaluations_per_step = 1 if method in ("adam", "tangent") else 2
        self.jacobian_evaluations_per_step = 1
        self.xs, self.ys, self.clean = xs, ys, clean
        # Neither known noise variance nor noise smoothing rate is learner information.
        self.capture_failed_candidates = [False] * len(grid)
        k, device = len(grid), xs.device
        self.lr = torch.tensor([lr for lr, _ in grid], dtype=torch.float32, device=device)[:, None, None]
        self.beta = torch.tensor([beta for _, beta in grid], dtype=torch.float32, device=device)[:, None, None]
        self.weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
        self.previous_weights = [] if method == "adam" else [w.clone() for w in self.weights]
        self.m = [torch.zeros_like(w) for w in self.weights]
        self.v = [torch.zeros_like(w) for w in self.weights]
        self.index = torch.zeros((), dtype=torch.int64, device=device)
        self.steps = torch.zeros((), dtype=torch.float32, device=device)
        self.error = torch.zeros(k, dtype=torch.float32, device=device)
        self.null_error = torch.zeros((), dtype=torch.float32, device=device)
        self.correction_norm_sum = torch.zeros(k, dtype=torch.float32, device=device)
        self.finite_candidates = torch.ones(k, dtype=torch.bool, device=device)
        # Every nonscalar state tensor is candidate-major. The capture audit relies on it.
        self.mutable = [*self.weights, *self.previous_weights, *self.m, *self.v,
                        self.index, self.steps, self.error, self.null_error,
                        self.correction_norm_sum, self.finite_candidates]

    @torch.no_grad()
    def _compute_update(self):
        ix = self.index.reshape(1)
        x = self.xs.index_select(0, ix).squeeze(0)
        y = self.ys.index_select(0, ix).squeeze(0)
        target = self.clean.index_select(0, ix).squeeze(0)
        prediction, inputs, sensitivities = sample_state(self.weights, x)
        jacobians = [j.unsqueeze(-1) * inp.unsqueeze(1) for inp, j in zip(inputs, sensitivities)]
        residual = prediction - y
        score = robust_score(residual) if self.method == "robust" else residual
        if self.method == "adam":
            coefficient = torch.zeros_like(prediction)
        elif self.method == "tangent":
            coefficient = sum((j * (w - previous)).sum((-1, -2))
                              for j, w, previous in zip(jacobians, self.weights, self.previous_weights))
        else:
            previous_prediction = sample_prediction(self.previous_weights, x)
            coefficient = (score - robust_score(previous_prediction - y)
                           if self.method == "robust" else prediction - previous_prediction)
        self.error.add_((prediction - target).square())
        self.null_error.add_(target.square())
        self.steps.add_(1)
        old_mass, mass, variance_mass = moment_masses(self.beta, self.steps)
        correction_energy = torch.zeros_like(prediction)
        unconstrained, preconditioners = [], []
        # All old/current model evaluations precede mutation; weights remain read-only here.
        for m, v, jacobian in zip(self.m, self.v, jacobians):
            gradient = score[:, None, None] * jacobian
            correction = coefficient[:, None, None] * jacobian
            next_m, next_v = transported_moments(m, v, gradient, correction, self.beta, old_mass)
            m.copy_(next_m)
            v.copy_(next_v)
            preconditioner = 1 / ((next_v / variance_mass).sqrt() + 1e-8)
            unconstrained.append(-self.lr * (next_m / mass) * preconditioner)
            if self.method == "implicit":
                preconditioners.append(preconditioner)
            correction_energy = correction_energy + correction.square().sum((-1, -2))
            self.finite_candidates.logical_and_(torch.isfinite(next_m).all((-1, -2))
                                                & torch.isfinite(next_v).all((-1, -2)))
        self.correction_norm_sum.add_(correction_energy.sqrt())
        increments = (implicit_step(jacobians, unconstrained, preconditioners, self.lr)
                      if self.method == "implicit" else unconstrained)
        next_weights = [weight + increment for weight, increment in zip(self.weights, increments)]
        for next_weight in next_weights:
            self.finite_candidates.logical_and_(torch.isfinite(next_weight).all((-1, -2)))
        for previous in self.previous_weights:
            self.finite_candidates.logical_and_(torch.isfinite(previous).all((-1, -2)))
        self.finite_candidates.logical_and_(torch.isfinite(self.error)
                                            & torch.isfinite(self.correction_norm_sum))
        self.index.add_(1)
        return next_weights

    @torch.no_grad()
    def _commit_weights(self, next_weights):
        # Captured copies outside compiled compute preserve distinct old/new lifetimes.
        # Clones within one compiled mutating graph can be eliminated by Inductor.
        for previous, current in zip(self.previous_weights, self.weights):
            previous.copy_(current)
        for current, following in zip(self.weights, next_weights):
            current.copy_(following)

    @torch.no_grad()
    def update(self):
        self._commit_weights(self._compute_update())

    def snapshot(self):
        return [tensor.clone() for tensor in self.mutable]

    @torch.no_grad()
    def restore(self, values):
        if len(values) != len(self.mutable):
            raise ValueError("snapshot does not match learner state")
        for tensor, value in zip(self.mutable, values):
            tensor.copy_(value)

    def capture(self):
        """Audit same-state eager/compiled transitions, then require exact graph replay.

        High-LR trajectories amplify rounding differences: local transitions, not
        independently evolved trajectories, are the strict numerical comparison.
        All mutable buffers are restored on success and every exception path.
        """
        state = self.snapshot()
        try:
            compute = torch.compile(self._compute_update, fullgraph=True, options={
                "max_autotune": True,
                "triton.cudagraphs": False,
                "emulate_precision_casts": True,
                "eager_numerics.division_rounding": True,
                "eager_numerics.use_pytorch_libdevice": True,
            })

            def compiled():
                self._commit_weights(compute())

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            try:
                with torch.cuda.stream(stream):
                    compiled()
                    self.restore(state)
                    compiled()
            finally:
                # Even a failed warmup must finish before rollback on the caller stream.
                torch.cuda.current_stream().wait_stream(stream)
            self.restore(state)
            max_error = 0.0
            audit_failed = torch.zeros_like(self.finite_candidates)
            for _ in range(self.capture_steps):
                before = self.snapshot()
                self.update()
                eager = self.snapshot()
                self.restore(before)
                compiled()
                torch.cuda.synchronize()
                # A compiler-only failure is a parity defect, not an ineligible trial.
                torch.testing.assert_close(self.finite_candidates, eager[-1], rtol=0, atol=0)
                surviving = self.finite_candidates & eager[-1]
                audit_failed.logical_or_(~surviving)
                for actual, wanted in zip(self.mutable, eager):
                    if actual.ndim:
                        actual, wanted = actual[surviving], wanted[surviving]
                    torch.testing.assert_close(actual, wanted, rtol=3e-3, atol=3e-5)
                    if actual.dtype != torch.bool and actual.numel():
                        max_error = max(max_error, float((actual - wanted).abs().max()))
            expected = self.snapshot()
            self.restore(state)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(self.capture_steps):
                    compiled()
            self.restore(state)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(self.finite_candidates, expected[-1], rtol=0, atol=0)
            surviving = self.finite_candidates & expected[-1] & ~audit_failed
            for actual, wanted in zip(self.mutable, expected):
                if actual.ndim:
                    actual, wanted = actual[surviving], wanted[surviving]
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            self.capture_failed_candidates = (~surviving).cpu().tolist()
            return graph, max_error
        finally:
            self.restore(state)

    def diagnostics(self):
        """Checkpoint-only synchronization; candidate-major telemetry, invalid -> null."""
        count = max(float(self.steps), 1.0)

        def values(tensor):
            return [float(v) if math.isfinite(v) else None for v in tensor.detach().cpu().tolist()]

        return {"correction_l2_mean": values(self.correction_norm_sum / count),
                "finite_candidates": self.finite_candidates.cpu().tolist()}

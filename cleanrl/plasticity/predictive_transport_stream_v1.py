"""Matrix-free predictive transport of Adam's biased first-moment mass.

The Jacobian stability q is a deterministic, current-input-conditioned proxy,
not a calibrated posterior or aleatoric uncertainty. Only noisy observations
enter learning; clean labels are used solely for prequential error reporting.
Persistent optimizer storage is linear in parameter count: previous weights,
Adam m/v, and per-row/scalar telemetry. No covariance state is constructed.
Weights and moment buffers are FP32. Forward/Jacobian differences use FP64 on
CUDA: near-zero sensitivities make the ratio ill-conditioned in FP32.
"""

import math

import torch

from cleanrl.plasticity.network_bayes_stream_v2 import sample_state


METHODS = ("adam", "full", "predictive", "uncertainty", "shared", "history")


def transport_terms(method, prediction, previous_prediction, y, jacobian, previous_jacobian, history):
    """Return applied correction and current/applied row stability, before EMA mutation."""
    difference = jacobian - previous_jacobian
    energy = jacobian.square().sum(-1)
    difference_energy = difference.square().sum(-1)
    denominator = energy + difference_energy
    q = torch.where(denominator == 0, torch.ones_like(denominator),
                    energy / torch.where(denominator == 0, torch.ones_like(denominator), denominator))
    q = q.to(history.dtype)
    if method == "uncertainty":
        applied = q
    elif method == "shared":
        applied = q.mean(-1, keepdim=True).expand_as(q)
    elif method == "history":
        applied = history
    else:
        applied = torch.ones_like(q)
    if method == "full":
        correction = ((prediction - y)[:, None, None] * jacobian
                      - (previous_prediction - y)[:, None, None] * previous_jacobian)
    else:
        correction = (prediction - previous_prediction)[:, None, None] * jacobian
    return correction * applied.unsqueeze(-1), q, applied, energy, difference_energy


def transported_adam_update(weight, m, v, gradient, correction, lr, beta, steps):
    """Transport the existing biased mass, not a fictitious unit-mass momentum."""
    # Avoid cancellation in 1-beta**t at long averaging horizons.
    log_beta = torch.log(torch.where(beta == 0, torch.ones_like(beta), beta))
    old_mass = -torch.expm1(log_beta * (steps - 1))
    mass = torch.where(beta == 0, torch.ones_like(beta), -torch.expm1(log_beta * steps))
    m.mul_(beta).add_((1 - beta) * gradient + beta * old_mass * correction)
    v.mul_(0.999).add_(0.001 * gradient.square())
    variance_mass = -torch.expm1(math.log(0.999) * steps)
    weight.sub_(lr * (m / mass) / ((v / variance_mass).sqrt() + 1e-8))


class Learner:
    def __init__(self, method, grid, initial, a, xs, ys, clean, noise_var):
        if method not in METHODS:
            raise ValueError(f"unknown transport method: {method}")
        if not grid or any(not math.isfinite(lr) or lr <= 0 or not math.isfinite(beta)
                           or not 0 <= beta < 1 for lr, beta in grid):
            raise ValueError("grid requires positive finite learning rates and 0 <= beta1 < 1")
        if xs.device.type != "cuda" or xs.dtype != torch.float32:
            raise ValueError("predictive transport requires CUDA FP32 inputs")
        if any(t.device != xs.device or t.dtype != torch.float32 for t in [*initial, ys, clean]):
            raise ValueError("weights and observations must share the CUDA FP32 input device")
        if a.graph_steps <= 0 or not 0 < a.noise_rate <= 1:
            raise ValueError("graph_steps must be positive and 0 < noise_rate <= 1")
        self.method, self.a, self.grid = method, a, list(grid)
        self.capture_steps = a.graph_steps
        self.capture_update_calls = 4 * self.capture_steps + 2
        self.xs, self.ys, self.clean = xs, ys, clean
        # noise_var deliberately is not retained: it is not optimizer information.
        self.capture_failed_candidates = [False] * len(grid)
        k, device = len(grid), xs.device
        self.lr = torch.tensor([lr for lr, _ in grid], device=device)[:, None, None]
        self.beta = torch.tensor([beta for _, beta in grid], device=device)[:, None, None]
        self.weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
        self.previous_weights = [w.clone() for w in self.weights]
        self.m = [torch.zeros_like(w) for w in self.weights]
        self.v = [torch.zeros_like(w) for w in self.weights]
        self.q_current = [torch.ones_like(w[..., 0]) for w in self.weights]
        self.q_history = [torch.ones_like(q) for q in self.q_current]
        self.q_applied = [torch.ones_like(q) for q in self.q_current]
        self.q_state_difference_sum = [torch.zeros_like(q) for q in self.q_current]
        self.index = torch.zeros((), dtype=torch.int64, device=device)
        self.steps = torch.zeros((), device=device)
        self.error = torch.zeros(k, device=device)
        self.null_error = torch.zeros((), device=device)
        self.correction_norm_sum = torch.zeros(k, device=device)
        self.jacobian_difference_ratio_sum = torch.zeros(k, device=device)
        self.finite_candidates = torch.ones(k, dtype=torch.bool, device=device)
        self.mutable = [*self.weights, *self.previous_weights, *self.m, *self.v,
                        *self.q_current, *self.q_history, *self.q_applied,
                        *self.q_state_difference_sum, self.index, self.steps, self.error,
                        self.null_error, self.correction_norm_sum,
                        self.jacobian_difference_ratio_sum, self.finite_candidates]

    @torch.no_grad()
    def _compute_update(self):
        ix = self.index.reshape(1)
        x = self.xs.index_select(0, ix).squeeze(0).double()
        y = self.ys.index_select(0, ix).squeeze(0)
        target = self.clean.index_select(0, ix).squeeze(0)
        prediction, inputs, sensitivities = sample_state([w.double() for w in self.weights], x)
        jacobians = [j.unsqueeze(-1) * inp.unsqueeze(1) for inp, j in zip(inputs, sensitivities)]
        if self.method != "adam":
            previous_prediction, previous_inputs, previous_sensitivities = sample_state(
                [w.double() for w in self.previous_weights], x)
            previous_jacobians = [j.unsqueeze(-1) * inp.unsqueeze(1)
                                  for inp, j in zip(previous_inputs, previous_sensitivities)]
        self.error.add_((prediction - target).square())
        self.null_error.add_(target.square())
        self.steps.add_(1)
        corrections = []
        correction_energy = torch.zeros_like(prediction)
        current_energy = torch.zeros_like(prediction)
        difference_energy = torch.zeros_like(prediction)
        # Finish every current/previous Jacobian and correction before ANY weight changes.
        for layer, jacobian in enumerate(jacobians):
            if self.method == "adam":
                correction = torch.zeros_like(jacobian)
            else:
                correction, q, applied, energy, delta_energy = transport_terms(
                    self.method, prediction, previous_prediction, y, jacobian,
                    previous_jacobians[layer], self.q_history[layer])
                self.q_current[layer].copy_(q)
                self.q_applied[layer].copy_(applied)
                self.q_state_difference_sum[layer].add_((q - self.q_history[layer]).square())
                self.finite_candidates.logical_and_(
                    torch.isfinite(self.q_history[layer]).all(-1)
                    & torch.isfinite(self.q_state_difference_sum[layer]).all(-1)
                    & torch.isfinite(applied).all(-1)
                    & torch.isfinite(self.previous_weights[layer]).all((-1, -2)))
                correction_energy = correction_energy + correction.square().sum((-1, -2))
                current_energy = current_energy + energy.sum(-1)
                difference_energy = difference_energy + delta_energy.sum(-1)
            corrections.append(correction)
        self.correction_norm_sum.add_(correction_energy.sqrt())
        # This label-free ratio is also the noise coefficient energy ||delta J||²/||J||².
        # With J=0 and delta J!=0 it is undefined/infinite, never silently clamped.
        both_zero = (current_energy == 0) & (difference_energy == 0)
        ratio = difference_energy / torch.where(both_zero, torch.ones_like(current_energy), current_energy)
        self.jacobian_difference_ratio_sum.add_(ratio)
        next_weights = []
        for weight, m, v, jacobian, correction in zip(self.weights, self.m, self.v, jacobians, corrections):
            gradient = (prediction - y)[:, None, None] * jacobian
            next_weight = weight.clone()
            transported_adam_update(next_weight, m, v, gradient, correction, self.lr, self.beta, self.steps)
            next_weights.append(next_weight)
            valid = (torch.isfinite(next_weight).all((-1, -2)) & torch.isfinite(m).all((-1, -2))
                     & torch.isfinite(v).all((-1, -2)))
            self.finite_candidates.logical_and_(valid)
        self.finite_candidates.logical_and_(torch.isfinite(self.error)
                                            & torch.isfinite(self.correction_norm_sum)
                                            & torch.isfinite(self.jacobian_difference_ratio_sum))
        self.index.add_(1)
        return next_weights

    @torch.no_grad()
    def _commit_weights(self, next_weights):
        # Separate captured CUDA copies preserve old/new weight lifetimes.
        # Inductor eliminated even explicit clones when both generations were
        # mutated inside one compiled graph, making transport spuriously zero.
        for previous, current in zip(self.previous_weights, self.weights):
            previous.copy_(current)
        for current, following in zip(self.weights, next_weights):
            current.copy_(following)
        # Likewise, keep the prior history alive until its applied value and
        # correction have been materialized by the compiled transition.
        if self.method != "adam":
            for history, current in zip(self.q_history, self.q_current):
                history.lerp_(current, self.a.noise_rate)

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
        """Audit each compiled transition locally, then require exact graph replay.

        Long high-LR trajectories amplify harmless rounding differences. Compare
        eager and compiled updates from the SAME state at every audit sample,
        rather than conflating trajectory sensitivity with a wrong transition.
        """
        state = self.snapshot()
        try:
            # q can resolve tiny Jacobian differences near a zero sensitivity.
            # Match eager rounding/libdevice and forbid fused multiply-add
            # contraction rather than relaxing the transition audit.
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
            with torch.cuda.stream(stream):
                compiled()
                self.restore(state)
                compiled()
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
            surviving = self.finite_candidates & expected[-1] & ~audit_failed
            self.capture_failed_candidates = (~surviving).cpu().tolist()
            for actual, wanted in zip(self.mutable, expected):
                if actual.ndim:
                    actual, wanted = actual[surviving], wanted[surviving]
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            return graph, max_error
        finally:
            self.restore(state)

    def diagnostics(self):
        """Checkpoint-only host synchronization; candidate-major JSON, invalid -> null."""
        count = max(float(self.steps), 1.0)

        def values(tensor):
            return [float(v) if math.isfinite(v) else None for v in tensor.detach().cpu().tolist()]

        layers = []
        for q, history, applied, difference in zip(self.q_current, self.q_history, self.q_applied,
                                                    self.q_state_difference_sum):
            layers.append({
                "q_current_mean": values(q.mean(-1)),
                "q_current_row_std": values(q.std(-1, unbiased=False)),
                "q_history_mean": values(history.mean(-1)),
                "q_history_row_std": values(history.std(-1, unbiased=False)),
                "q_state_minus_history_rms": values((difference.mean(-1) / count).sqrt()),
                "q_applied_mean": values(applied.mean(-1)),
            })
        return {"layers": layers, "correction_l2_mean": values(self.correction_norm_sum / count),
                "jacobian_difference_energy_ratio_mean": values(self.jacobian_difference_ratio_sum / count),
                "finite_candidates": self.finite_candidates.cpu().tolist(),
                "q_interpretation": "deterministic Jacobian stability, not calibrated uncertainty"}

"""Iterated full-network EKF: refine one observation, condition the prior once.

For prior mean theta0 and predicted covariance C = P + diag(Q), a local
linearization at theta has innovation f(theta) - y - j @ (theta - theta0).
Its Gaussian posterior mean is theta0 - Cj * innovation / (R + j @ Cj).
Every refinement uses the same C, theta0, observation, and historical R.
Only the final linearization conditions C. Repeated refinements therefore do
not falsely count the same observation as independent evidence.

The frozen v2 outer update retains pre-update clean scoring and updates the
inferred observation variance only after assimilation. No clean label enters
this recurrence. Unstable nonlinear refinements are exposed, not clamped.
"""

import torch

from cleanrl.plasticity import network_bayes_stream_v2 as reference


class IteratedLearner(reference.Learner):
    def __init__(self, iterations, grid, initial, a, xs, ys, clean, noise_var):
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self.iterations = iterations
        super().__init__("network", grid, initial, a, xs, ys, clean, noise_var)

    def network_update(self, inputs, sensitivities, residual, observation):
        """Return signed row leverage of the final linearization, as in v2."""
        if self.iterations == 1:
            return super().network_update(inputs, sensitivities, residual, observation)

        # cat owns its storage: subsequent in-place weight updates cannot move
        # the fixed prior mean. Configurations share the actual sampled input.
        theta0 = torch.cat([w.flatten(1) for w in self.weights], -1)
        x = inputs[0][0, :-1]
        prediction0 = (self.weights[-1][:, 0, :-1] * inputs[-1][:, :-1]).sum(-1) + self.weights[-1][:, 0, -1]
        self.cov.diagonal(dim1=-2, dim2=-1).add_(self.process)

        for iteration in range(self.iterations):
            if iteration:
                prediction, inputs, sensitivities = reference.sample_state(self.weights, x)
            jacobian = torch.cat([
                (j.unsqueeze(-1) * inp.unsqueeze(1)).flatten(1)
                for inp, j in zip(inputs, sensitivities)], -1)
            pj = torch.bmm(self.cov, jacobian.unsqueeze(-1)).squeeze(-1)
            denominator = observation + (jacobian * pj).sum(-1)
            # Retain the original residual rather than reconstructing y with
            # another rounded subtraction. The first linearization has delta=0.
            innovation = residual if iteration == 0 else (
                residual + (prediction - prediction0) - (jacobian * (theta - theta0)).sum(-1))
            theta = theta0 - pj * (innovation / denominator).unsqueeze(-1)
            start = 0
            for w in self.weights:
                stop = start + w[0].numel()
                w.copy_(theta[:, start:stop].view_as(w))
                start = stop

        credit, start = [], 0
        for w in self.weights:
            stop = start + w[0].numel()
            block_j = jacobian[:, start:stop].view_as(w)
            direction = pj[:, start:stop].view_as(w)
            credit.append((block_j * direction).sum(-1) / denominator.unsqueeze(-1))
            start = stop
        self.cov.sub_(pj.unsqueeze(-1) * pj.unsqueeze(-2) / denominator[:, None, None])
        return torch.cat(credit, -1)

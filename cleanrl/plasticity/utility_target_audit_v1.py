"""CUDA float64 counterexamples and chain-rule checks for finite-update targets.

This module performs no training and initializes no CUDA state until run_audits().
The fixtures witness correctness/failure cases; they do not restrict the utility
contract to quadratic losses, stateless networks, or independent parameter blocks.
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call


Parameters = tuple[Tensor, ...]


def _audit_log_gain(
    loss: Callable[[Parameters], Tensor], theta: Parameters, delta: Parameters
) -> dict:
    """Hold the originating update fixed; differentiate only its applied gain."""
    theta = tuple(p.detach() for p in theta)
    delta = tuple(d.detach() for d in delta)
    baseline = loss(theta).detach()
    endpoint = tuple((p + d).requires_grad_() for p, d in zip(theta, delta))
    endpoint_loss = loss(endpoint)
    gradient = torch.autograd.grad(endpoint_loss, endpoint)
    projected = torch.stack([-(g * d).sum() for g, d in zip(gradient, delta)])

    def utility(log_gain: Tensor) -> Tensor:
        return baseline - loss(
            tuple(p + gain.exp() * d for p, d, gain in zip(theta, delta, log_gain))
        )

    log_gain = baseline.new_zeros(len(theta), requires_grad=True)
    finite_utility = utility(log_gain)
    autodiff = torch.autograd.grad(finite_utility, log_gain)[0]
    epsilon = 1e-5
    perturbations = torch.eye(len(theta), device=baseline.device, dtype=baseline.dtype) * epsilon
    with torch.no_grad():
        central = torch.stack(
            [(utility(step) - utility(-step)) / (2 * epsilon) for step in perturbations]
        )
    assert torch.isfinite(torch.cat((projected, autodiff, central))).all().item()
    assert torch.isfinite(torch.stack((baseline, endpoint_loss, finite_utility))).all().item()
    torch.testing.assert_close(projected, autodiff, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(projected, central, rtol=2e-7, atol=2e-9)
    torch.testing.assert_close(finite_utility, baseline - endpoint_loss, rtol=1e-12, atol=1e-12)
    return {
        "old_loss": baseline.item(),
        "new_loss": endpoint_loss.item(),
        "finite_utility": finite_utility.item(),
        "negative_endpoint_gradient_dot_fixed_delta": projected.detach().tolist(),
        "autograd_log_gain_derivative": autodiff.detach().tolist(),
        "central_log_gain_derivative": central.tolist(),
        "autograd_max_absolute_error": (projected - autodiff).abs().max().item(),
        "central_max_absolute_error": (projected - central).abs().max().item(),
        "central_step": epsilon,
    }


class _MaskedBufferedModel(nn.Module):
    """Explicit dropout realization plus a genuinely mutating running buffer."""

    def __init__(self, tensor: Callable[..., Tensor]) -> None:
        super().__init__()
        self.weight = nn.Parameter(tensor([[0.6, -0.3], [0.4, 0.8]]))
        self.readout = nn.Parameter(tensor([0.9, -0.5]))
        self.register_buffer("running_mean", tensor([0.15, -0.2]))

    def forward(self, example: Tensor, dropout_mask: Tensor) -> Tensor:
        activation = example @ self.weight
        hidden = torch.tanh(activation - self.running_mean)
        self.running_mean.lerp_(activation.detach().mean(dim=0), 0.2)
        return (hidden * dropout_mask / 0.5) @ self.readout


@torch.enable_grad()
def run_audits() -> dict:
    """Return JSON-safe evidence, or raise AssertionError; invoke only via mlq."""
    assert torch.cuda.is_available(), "Utility target audits require CUDA; no CPU fallback."

    def tensor(value) -> Tensor:
        return torch.tensor(value, device="cuda", dtype=torch.float64)

    theta = (tensor([0.4, 0.7]), tensor([0.8, 0.5]))
    delta = (tensor([-0.2, -0.15]), tensor([-0.3, -0.1]))
    target = tensor(-0.3)

    def coupled_loss(parameters: Parameters) -> Tensor:
        left, right = parameters
        prediction = (left * right).sum()
        return 0.5 * (prediction - target).square() + 0.05 * (
            left.pow(4).sum() + right.pow(4).sum()
        )

    nonlinear = _audit_log_gain(coupled_loss, theta, delta)
    with torch.no_grad():
        baseline = coupled_loss(theta)

        def gain_utility(gain: float) -> Tensor:
            return baseline - coupled_loss(tuple(p + gain * d for p, d in zip(theta, delta)))

        # Three exact finite evaluations determine a quadratic, not the true curve.
        nodes = (0.0, 1.0, 2.0)
        node_utilities = torch.stack([gain_utility(gain) for gain in nodes])

        def quadratic(gain: float) -> Tensor:
            return (
                node_utilities[0] * (gain - 1) * (gain - 2) / 2
                - node_utilities[1] * gain * (gain - 2)
                + node_utilities[2] * gain * (gain - 1) / 2
            )

        at_nodes = torch.stack([quadratic(gain) for gain in nodes])
        torch.testing.assert_close(at_nodes, node_utilities, rtol=0, atol=1e-14)
        probe_gains = (0.5, 1.5, 3.0)
        true_utilities = torch.stack([gain_utility(gain) for gain in probe_gains])
        interpolated = torch.stack([quadratic(gain) for gain in probe_gains])
        interpolation_errors = (true_utilities - interpolated).abs()
        assert interpolation_errors[:2].min().item() > 1e-4
        assert interpolation_errors[2].item() > 1e-4

        left_only = baseline - coupled_loss((theta[0] + delta[0], theta[1]))
        right_only = baseline - coupled_loss((theta[0], theta[1] + delta[1]))
        joint = gain_utility(1.0)
        interaction = joint - left_only - right_only
        assert abs(interaction.item()) > 1e-3
        # Marginals depend on which other block was already applied.
        right_after_left = joint - left_only
        left_after_right = joint - right_only
        torch.testing.assert_close(right_after_left - right_only, interaction)
        torch.testing.assert_close(left_after_right - left_only, interaction)

    model = _MaskedBufferedModel(tensor)
    parameters = tuple(model.parameters())
    parameter_names = tuple(name for name, _ in model.named_parameters())
    state = {name: value.detach().clone() for name, value in model.named_buffers()}
    example, label = tensor([[1.2, -0.7]]), tensor([-0.4])
    shared_mask, other_mask = tensor([[1.0, 0.0]]), tensor([[0.0, 1.0]])
    update = (tensor([[-0.05, 0.03], [0.02, -0.04]]), tensor([-0.08, 0.06]))

    def evaluate(parameters: Parameters, mask: Tensor, buffer_state: dict) -> tuple[Tensor, dict]:
        # functional_call redirects mutation into these private buffers. It does
        # not itself copy buffers or guarantee matched stochastic realizations.
        branch_buffers = {name: value.clone() for name, value in buffer_state.items()}
        prediction = functional_call(
            model,
            (dict(zip(parameter_names, parameters)), branch_buffers),
            (example, mask),
            strict=True,
        )
        return 0.5 * (prediction - label).square().mean(), branch_buffers

    def stochastic_loss(parameters: Parameters) -> Tensor:
        return evaluate(parameters, shared_mask, state)[0]

    stochastic = _audit_log_gain(stochastic_loss, parameters, update)
    with torch.no_grad():
        old_loss, next_state = evaluate(parameters, shared_mask, state)
        zero_update_parameters = tuple(p + torch.zeros_like(p) for p in parameters)
        matched_zero_loss, _ = evaluate(zero_update_parameters, shared_mask, state)
        other_mask_loss, _ = evaluate(zero_update_parameters, other_mask, state)
        advanced_buffer_loss, _ = evaluate(zero_update_parameters, shared_mask, next_state)
        zero_utility = old_loss - matched_zero_loss
        mask_artifact = old_loss - other_mask_loss
        buffer_artifact = old_loss - advanced_buffer_loss
        assert zero_utility.item() == 0.0
        assert abs(mask_artifact.item()) > 1e-4
        assert abs(buffer_artifact.item()) > 1e-4
        assert (next_state["running_mean"] - state["running_mean"]).abs().max().item() > 0
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        shifted = tuple(p + d for p, d in zip(parameters, update))
        # Reverse the branch evaluation order; private buffer mutation must not
        # change either branch or consume a different dropout realization.
        new_first = stochastic_loss(shifted)
        old_second = stochastic_loss(parameters)
        torch.testing.assert_close(old_second, old_loss, rtol=0, atol=0)
        torch.testing.assert_close(
            old_second - new_first, tensor(stochastic["finite_utility"]), rtol=0, atol=0
        )

    return {
        "passed": True,
        "device": str(theta[0].device),
        "dtype": str(theta[0].dtype),
        "contract": {
            "utility": "U = loss(theta, next_example) - loss(theta + delta, next_example)",
            "log_gain_derivative_at_zero": "dU/dell_b = -grad_b loss(theta + delta) dot delta_b",
            "held_fixed": "Originating delta, example, target, stochastic realization, initial buffer state.",
            "scope": "Any differentiable scalar loss and disjoint parameter blocks; no quadratic or block-additivity assumption.",
            "excluded": "Historical credit through intervening updates, optimizer performance, and predictive learnability are not tested here.",
            "state_semantics": "Each branch starts from identical buffers; differentiable within-forward state dependence belongs to the loss closure. Never advance one branch from the other's resulting buffers.",
        },
        "nonlinear_two_block": nonlinear,
        "quadratic_counterexample": {
            "gain_parameterization": "theta + gain * fixed_delta (gain, not log gain)",
            "fitted_gains": list(nodes),
            "fitted_utilities": node_utilities.tolist(),
            "fit_max_absolute_error": (at_nodes - node_utilities).abs().max().item(),
            "probe_gains": list(probe_gains),
            "true_utilities": true_utilities.tolist(),
            "quadratic_utilities": interpolated.tolist(),
            "absolute_errors": interpolation_errors.tolist(),
            "conclusion": "Exact finite interpolation at three gains is neither exact between nodes nor globally exact.",
        },
        "block_interactions": {
            "left_only_utility": left_only.item(),
            "right_only_utility": right_only.item(),
            "joint_utility": joint.item(),
            "joint_minus_sum_of_individual_utilities": interaction.item(),
            "right_marginal_after_left": right_after_left.item(),
            "left_marginal_after_right": left_after_right.item(),
            "conclusion": "Global finite utility is a joint target, not a per-block label. Block derivatives are sensitivities, not additive finite-utility allocations.",
        },
        "stochastic_buffered_closure": {
            **stochastic,
            "shared_mask_zero_update_utility": zero_utility.item(),
            "different_mask_zero_update_spurious_utility": mask_artifact.item(),
            "advanced_buffer_zero_update_spurious_utility": buffer_artifact.item(),
            "buffer_update_max_absolute_size": (next_state["running_mean"] - state["running_mean"]).abs().max().item(),
            "source_buffers_unchanged": True,
            "branch_order_invariant": True,
            "randomness_scope": "Two explicit Bernoulli-dropout realizations; a counterexample, not an estimate of stochastic variance or expected utility.",
        },
        "cost": {
            "finite_target": "Two matched forward loss evaluations; one extra if the updated-parameter loss is already available.",
            "all_block_derivatives": "One reverse-mode gradient at theta + delta, then one dot product per block; no Hessian or per-block backward pass.",
            "replay_storage": "Preserve theta and fixed delta (or equivalent replay), example/target, randomness, and initial buffers; private mutable buffers per branch.",
            "audit_only_overhead": "Independent gain autograd, 2*block_count central-difference forwards, gain interpolation probes, and paired state/randomness counterexamples. These are audit checks, not a proposed online training workload.",
            "two_block_interaction": "Four forward evaluations at neither/left/right/both applied updates, reusable with the joint finite-utility evaluations.",
        },
    }

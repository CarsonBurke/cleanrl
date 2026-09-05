"""Queue fused temperature numerical tests through mlq; no training is run."""

import math

import pytest
import torch
from torch.distributions import Beta

from cleanrl.shared.vmpo_temperature import solve_log_temperature


def reference_log_temperature(centered, selected, log_count, low, high, epsilon, iterations=32):
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        eta = midpoint.exp()
        logits = torch.where(selected, centered / eta, -torch.inf)
        log_weights = logits - torch.logsumexp(logits, dim=0)
        weights = log_weights.exp()
        safe_log_weights = torch.where(selected, log_weights, 0.0)
        kl = (weights * (safe_log_weights + log_count)).sum()
        low = torch.where(kl > epsilon, midpoint, low)
        high = torch.where(kl > epsilon, high, midpoint)
    return high


def solver_inputs(advantages, epsilon=0.01, topk_fraction=0.5):
    threshold = torch.sort(advantages).values[-max(1, int(advantages.numel() * topk_fraction))]
    selected = advantages >= threshold
    maximum = advantages.max()
    centered = advantages - maximum
    log_count = selected.sum().to(torch.float32).log()
    low = torch.full_like(threshold, math.log(1e-8))
    high = ((maximum - threshold) / epsilon).clamp_min(1e-8).log()
    return centered, selected, log_count, low, high, epsilon


def weights_and_metrics(centered, selected, log_count, log_eta):
    logits = torch.where(selected, centered / log_eta.exp(), -torch.inf)
    log_weights = logits - torch.logsumexp(logits, dim=0)
    weights = log_weights.exp()
    kl = (weights * (torch.where(selected, log_weights, 0.0) + log_count)).sum()
    ess = weights.square().sum().reciprocal()
    return weights, kl, ess


def requires_cuda(function):
    function = pytest.mark.cuda(function)
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")(function)


@requires_cuda
@pytest.mark.parametrize("size,scale,epsilon", [(39 * 16, 1.0, 0.01), (39 * 64, 1.0, 0.01),
                                               (127, 1e-5, 0.01), (4097, 1e6, 0.1)])
def test_solver_preserves_weights_kl_ess_rng_and_policy_gradients(size, scale, epsilon):
    generator = torch.Generator(device="cuda").manual_seed(1)
    advantages = torch.randn(size, device="cuda", generator=generator) * scale
    # Repeated advantages exercise inclusive selection at ties.
    advantages[::5] = advantages[1]
    inputs = solver_inputs(advantages, epsilon)
    with torch.no_grad():
        expected = reference_log_temperature(*inputs)
        rng_before = torch.cuda.get_rng_state()
        actual = solve_log_temperature(*inputs)
        assert torch.equal(rng_before, torch.cuda.get_rng_state())
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-4)
        weights_a, kl_a, ess_a = weights_and_metrics(*inputs[:3], expected)
        weights_b, kl_b, ess_b = weights_and_metrics(*inputs[:3], actual)
    torch.testing.assert_close(weights_b, weights_a, rtol=8e-5, atol=2e-7)
    torch.testing.assert_close(kl_b, kl_a, rtol=0, atol=3e-6)
    torch.testing.assert_close(ess_b, ess_a, rtol=2e-5, atol=2e-5)
    assert kl_b <= epsilon + 3e-6
    torch.testing.assert_close(weights_b.sum(), torch.ones((), device="cuda"), rtol=0, atol=1e-6)
    assert (weights_b[~inputs[1]] == 0).all()

    alpha_logits = torch.randn((size, 6), device="cuda", generator=generator, requires_grad=True)
    beta_logits = torch.randn((size, 6), device="cuda", generator=generator, requires_grad=True)
    actions = torch.rand((size, 6), device="cuda", generator=generator).clamp(1e-6, 1 - 1e-6)
    distribution = Beta(torch.nn.functional.softplus(alpha_logits) + 1,
                        torch.nn.functional.softplus(beta_logits) + 1, validate_args=False)
    log_prob = distribution.log_prob(actions).sum(-1)
    expected_loss = -(weights_a * log_prob).sum()
    actual_loss = -(weights_b * log_prob).sum()
    expected_grads = torch.autograd.grad(expected_loss, (alpha_logits, beta_logits), retain_graph=True)
    actual_grads = torch.autograd.grad(actual_loss, (alpha_logits, beta_logits))
    torch.testing.assert_close(actual_loss, expected_loss, rtol=3e-6, atol=3e-6)
    for expected_grad, actual_grad in zip(expected_grads, actual_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=8e-5, atol=2e-7)


@requires_cuda
@pytest.mark.parametrize("advantages", [[1.0], [2.0] * 33, [1.0] * 21 + [0.0] * 20])
def test_singleton_and_flat_selected_ties(advantages):
    inputs = solver_inputs(torch.tensor(advantages, device="cuda"))
    with torch.no_grad():
        expected = reference_log_temperature(*inputs)
        actual = solve_log_temperature(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("iterations", [0, 1, 8, 32])
def test_iteration_count_noncontiguous_vectors_and_tensor_epsilon(iterations):
    advantages = torch.linspace(-2, 2, 37, device="cuda")
    centered, selected, count, low, high, epsilon = solver_inputs(advantages)
    centered_storage = torch.empty(74, device="cuda")
    selected_storage = torch.empty(74, device="cuda", dtype=torch.bool)
    centered_storage[::2] = centered
    selected_storage[::2] = selected
    inputs = (centered_storage[::2], selected_storage[::2], count, low, high,
              torch.tensor(epsilon, device="cuda"))
    with torch.no_grad():
        expected = reference_log_temperature(*inputs, iterations=iterations)
        actual = solve_log_temperature(*inputs, iterations=iterations)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-4)


@requires_cuda
def test_solver_compiles_fullgraph_and_replays_without_stale_output():
    inputs = solver_inputs(torch.linspace(-3, 3, 39 * 16, device="cuda"))
    compiled = torch.compile(solve_log_temperature, fullgraph=True, mode="reduce-overhead")
    with torch.no_grad():
        for scale in (1.0, 2.0, 0.5):
            current = (inputs[0] * scale, *inputs[1:])
            expected = reference_log_temperature(*current)
            torch.compiler.cudagraph_mark_step_begin()
            actual = compiled(*current).clone()
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-4)


@requires_cuda
def test_requested_temperature_gradients_are_rejected():
    inputs = list(solver_inputs(torch.linspace(-1, 1, 32, device="cuda")))
    inputs[0].requires_grad_()
    with pytest.raises(ValueError, match="constant targets"):
        solve_log_temperature(*inputs)
    with torch.no_grad():
        assert not solve_log_temperature(*inputs).requires_grad


@requires_cuda
@pytest.mark.parametrize("case", ["empty_selection", "masked_nan", "selected_nan", "nan_bound"])
def test_nonfinite_and_empty_selection_follow_reference_decisions(case):
    centered = torch.tensor([0.0, -1.0, -2.0], device="cuda")
    selected = torch.tensor([True, False, True], device="cuda")
    count = torch.tensor(math.log(2.0), device="cuda")
    low = torch.tensor(-18.0, device="cuda")
    high = torch.tensor(2.0, device="cuda")
    if case == "empty_selection":
        selected.zero_()
        count.fill_(-torch.inf)
    elif case == "masked_nan":
        centered[1] = torch.nan
    elif case == "selected_nan":
        centered[0] = torch.nan
    else:
        high.fill_(torch.nan)
    inputs = centered, selected, count, low, high, 0.01
    with torch.no_grad():
        expected = reference_log_temperature(*inputs)
        actual = solve_log_temperature(*inputs)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-4, equal_nan=True)


def test_cpu_inputs_and_invalid_shapes_are_rejected_without_gpu_work():
    scalar = torch.tensor(0.0)
    with pytest.raises(ValueError, match="float32 CUDA"):
        solve_log_temperature(torch.ones(3), torch.ones(3, dtype=torch.bool), scalar, scalar, scalar, 0.01)
    with pytest.raises(ValueError, match="nonempty"):
        solve_log_temperature(torch.empty(0), torch.empty(0, dtype=torch.bool), scalar, scalar, scalar, 0.01)

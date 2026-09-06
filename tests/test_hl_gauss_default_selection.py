"""Numerical invariants for candidate rules, not assertions about winning scores."""

import math

import pytest
import torch

from scripts.hlgauss.default_selection import Policy


@pytest.mark.parametrize("policy", [Policy(bins=101, auto_sigma=True), Policy(width_in_std=1.0, sigma=0.75)])
def test_return_unit_changes_preserve_relative_resolution_and_smoothing(policy):
    original = policy.candidate(50.0, 0.38)
    rescaled = policy.candidate(5000.0, 38.0)
    assert original.bins == rescaled.bins
    assert original.sigma == rescaled.sigma
    original_head = original.config().build().double()
    rescaled_head = rescaled.config().build().double()
    targets = torch.tensor([-0.2, 0.1, 3.8], dtype=torch.float64)
    torch.testing.assert_close(original_head.project(targets), rescaled_head.project(100 * targets), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        original_head.probs_to_scalar(original_head.project(targets)),
        rescaled_head.probs_to_scalar(rescaled_head.project(100 * targets)) / 100,
        atol=2e-6,
        rtol=2e-5,
    )


def test_automatic_resolution_respects_capacity_without_changing_support():
    policy = Policy(width_in_std=1.0, max_bins=255)
    resolved = policy.candidate(50.0, 0.001)
    assert resolved.bins == 255
    head = resolved.config().build()
    torch.testing.assert_close(head.support[[0, -1]], torch.tensor([-50.0, 50.0]))


@pytest.mark.parametrize("target_std", [1.0, 0.01])
def test_selected_sigma_satisfies_interior_quantization_bias_budget(target_std):
    resolved = Policy(bins=101, auto_sigma=True).candidate(50.0, target_std)
    width = 2 * resolved.bound / (resolved.bins - 1)
    q = math.exp(-2 * math.pi**2 * resolved.sigma**2)
    bound = width / math.pi * q / (1 - q)
    assert bound <= 0.005 * target_std
    head = resolved.config().build().double()
    targets = torch.linspace(-0.5, 0.5, 257, dtype=torch.float64)
    actual = head.probs_to_scalar(head.project(targets))
    assert float((actual - targets).abs().max()) <= bound + 1e-12
    if resolved.sigma > 0.5:
        previous_q = math.exp(-2 * math.pi**2 * 0.5**2)
        assert width / math.pi * previous_q / (1 - previous_q) > 0.005 * target_std

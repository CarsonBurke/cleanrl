"""The continuous MaxRL weights must be a STRICT generalisation of the released ones.

The whole claim of cleanrl/shared/continuous_maclaurin.py is that replacing MaxRL's
failure INDICATOR with continuous failure MASS is the right generalisation. The only
way that claim is worth anything is if, fed binary rewards, it reproduces the authors'
own estimator element for element -- so that is what most of this file checks, against
the reference implementation loaded from the MaxRL repo by path.
"""
import importlib.util
import pathlib

import pytest
import torch

from cleanrl.shared.continuous_maclaurin import (
    continuous_maclaurin_weights, failure_mass_symmetric_means,
)

REFERENCE = pathlib.Path("/home/marvin/Documents/repositories/maxrl/verl/trainer/ppo/maclaurin.py")


def _reference():
    spec = importlib.util.spec_from_file_location("maclaurin_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="MaxRL reference checkout absent")

GROUP_SIZES = [2, 3, 4, 8, 16]
ORDERS = [1, 2, 3, 4, 8, 16, 64]


@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("order", ORDERS)
def test_binary_rewards_reproduce_the_released_estimator(group_size, order):
    """Element for element, for every achievable success count."""
    reference = _reference()
    for successes in range(group_size + 1):
        rewards = torch.zeros(group_size, dtype=torch.float64)
        rewards[:successes] = 1.0
        mine = continuous_maclaurin_weights(rewards, order)

        w_succ, w_fail = reference.maclaurin_weights(
            torch.tensor([successes], dtype=torch.float64), group_size, order
        )
        expected = torch.where(rewards > 0.5, w_succ[0], w_fail[0])
        assert torch.allclose(mine, expected, rtol=1e-12, atol=1e-14), (
            f"N={group_size} T={order} C={successes}: {mine} != {expected}"
        )


@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_binary_reduction_holds_batched_and_shuffled(group_size):
    """The reduction must not depend on successes being contiguous or on batch shape."""
    reference = _reference()
    generator = torch.Generator().manual_seed(11)
    rewards = (torch.rand((32, group_size), generator=generator) < 0.5).double()
    mine = continuous_maclaurin_weights(rewards, order=8)
    counts = rewards.sum(-1)
    w_succ, w_fail = reference.maclaurin_weights(counts, group_size, 8)
    expected = torch.where(rewards > 0.5, w_succ[:, None], w_fail[:, None])
    assert torch.allclose(mine, expected, rtol=1e-12, atol=1e-14)


# --- properties of the continuous extension itself --------------------------

def test_order_one_is_reinforce():
    """T=1 truncates the series at its first term: w_i = r_i / N, no failure coupling."""
    rewards = torch.rand(4, 8, dtype=torch.float64)
    assert torch.allclose(
        continuous_maclaurin_weights(rewards, 1), rewards / 8, rtol=1e-12, atol=1e-14
    )


def test_symmetric_means_exclude_self_and_average_over_subsets():
    """e_m^(-i) is the MEAN over m-subsets of the others -- check against brute force."""
    import itertools
    q = torch.tensor([0.1, 0.4, 0.7, 0.9], dtype=torch.float64)
    means = failure_mass_symmetric_means(q, order=3)
    for i in range(4):
        others = [q[j] for j in range(4) if j != i]
        for m in range(4):
            subsets = list(itertools.combinations(others, m))
            brute = sum(torch.prod(torch.stack(s)) if s else torch.tensor(1.0, dtype=torch.float64)
                        for s in subsets) / len(subsets)
            assert means[i, m].item() == pytest.approx(brute.item(), rel=1e-12)


def test_degree_zero_is_one_and_high_degrees_vanish():
    q = torch.rand(5, 3, dtype=torch.float64)
    means = failure_mass_symmetric_means(q, order=6)
    assert torch.allclose(means[..., 0], torch.ones_like(means[..., 0]))
    # Only N-1 = 2 others exist, so degrees 3+ average over an empty family: 0.
    assert torch.all(means[..., 3:] == 0.0)


def test_weights_are_continuous_in_the_reward():
    """A strict generalisation must interpolate, not just agree at the binary corners."""
    group = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    low = continuous_maclaurin_weights(group, 8)
    group_mid = group.clone()
    group_mid[0, 1] = 0.5
    mid = continuous_maclaurin_weights(group_mid, 8)
    group_high = group.clone()
    group_high[0, 1] = 1.0
    high = continuous_maclaurin_weights(group_high, 8)
    # Raising rollout 1's reward moves its own weight monotonically upward...
    assert low[0, 1] < mid[0, 1] < high[0, 1]
    # ...and strictly between the two binary endpoints it is a genuine interior value.
    assert not torch.allclose(mid, low) and not torch.allclose(mid, high)


def test_a_uniformly_failing_group_still_produces_a_bounded_signal():
    """All-fail is where the deflation recurrence would have lost precision."""
    rewards = torch.zeros(1, 8, dtype=torch.float64)
    weights = continuous_maclaurin_weights(rewards, 8)
    assert torch.isfinite(weights).all()
    # Every rollout failed equally, so every weight is identical: a pure baseline that
    # contributes nothing once the scores are centred.
    assert torch.allclose(weights, weights[0, 0].expand_as(weights))


def test_rewards_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError, match=r"rewards in \[0, 1\]"):
        continuous_maclaurin_weights(torch.tensor([[0.0, 1.5]]), 4)
    with pytest.raises(ValueError, match="order must be"):
        continuous_maclaurin_weights(torch.tensor([[0.0, 1.0]]), 0)

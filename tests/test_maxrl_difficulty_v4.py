import pytest
import torch

from cleanrl.maxrl.ppo_continuous_action_maxrl_difficulty_v4 import (
    Args, difficulty_weights, success_probability, truncated_ml_weight, validate_args,
)


def test_truncated_weight_limits():
    p = torch.tensor([0.0, 1e-9, 0.01, 0.5, 1.0], dtype=torch.float64)
    torch.testing.assert_close(truncated_ml_weight(p, 1.0), torch.ones_like(p))
    w = truncated_ml_weight(p, 8.0)
    assert w[0].item() == 8.0 and w[1].item() == pytest.approx(8.0, rel=1e-6)
    assert w[2].item() == pytest.approx((1 - 0.99 ** 8) / 0.01)
    assert w[3].item() == pytest.approx((1 - 0.5 ** 8) / 0.5)
    assert w[4].item() == pytest.approx(1.0, rel=1e-5)
    # Large T approaches maximum likelihood's 1/p away from p = 0.
    assert truncated_ml_weight(torch.tensor([0.2]), 1e4).item() == pytest.approx(5.0)


def test_success_probability_is_monotone_in_value():
    values = torch.linspace(-2, 2, 101)
    returns = values + 0.3 * torch.randn(101, generator=torch.Generator().manual_seed(0))
    p, tau, _ = success_probability(values, returns, 0.9)
    assert (p.diff() >= 0).all() and 0 < p.mean() < 1


def test_unit_truncation_is_the_control():
    values, returns = torch.randn(512), torch.randn(512)
    weights, stats = difficulty_weights(values, returns, Args(maxrl_truncation=1.0))
    torch.testing.assert_close(weights, torch.ones(512))
    weights, _ = difficulty_weights(values, returns, Args(maxrl_truncation=8.0))
    assert weights.mean().item() == pytest.approx(1.0) and weights.max() <= 8.0


@pytest.mark.parametrize("override", [{"success_quantile": 1.0}, {"maxrl_truncation": 0.5}])
def test_rejects_invalid_args(override):
    with pytest.raises(ValueError):
        validate_args(Args(num_envs=2, num_steps=8, num_minibatches=2, **override))

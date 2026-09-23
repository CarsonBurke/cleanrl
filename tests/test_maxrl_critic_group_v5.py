import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.maxrl.ppo_continuous_action_maxrl_critic_group_v5 import (
    Agent, Args, higher_order_coefficients, order_weights, ppo_loss, validate_args,
)


def omega(n, t, k):
    return sum(math.comb(n - k, m) / math.comb(n - 1, m) for m in range(t)) / n


def test_order_weights_limits():
    n = 8
    assert torch.allclose(order_weights(n, 1), torch.zeros(n))  # T = 1 is REINFORCE: no excess
    full = order_weights(n, n) + 1.0 / n
    torch.testing.assert_close(full, torch.tensor([1.0 / k for k in range(1, n + 1)]))  # paper: 1/K


def test_coefficients_match_threshold_integral():
    torch.manual_seed(0)
    n, t = 6, 3
    scores = torch.randn(n, 5, dtype=torch.float64)
    coefficients = higher_order_coefficients(scores, order_weights(n, t).double())
    # Brute force: integrate per-threshold excess weights over a fine grid of thresholds.
    for b in range(5):
        q = scores[:, b]
        grid = torch.linspace(q.min().item(), q.max().item(), 200_001, dtype=torch.float64)[:-1]
        dtau = (q.max() - q.min()) / 200_000
        above = (q.unsqueeze(1) > grid.unsqueeze(0))
        k = above.sum(0)
        excess = torch.tensor([omega(n, t, int(x)) - 1.0 / n for x in k])
        raw = (above.double() * excess).sum(1) * dtau
        torch.testing.assert_close(coefficients[:, b], raw - raw.mean(), atol=2e-4, rtol=0)


def test_ppo_loss_runs_and_t1_adds_nothing():
    envs = SimpleNamespace(single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
                           single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float64))
    torch.manual_seed(0)
    agent = Agent(envs)
    n, b = 4, 32
    observations = torch.randn(b, 17)
    native = torch.rand(b, 6).clamp(0.01, 0.99)
    group = torch.rand(n, b, 6).clamp(0.01, 0.99)
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        old = agent.action_logprob(alpha, beta, native)
        group_old = agent.action_logprob(alpha, beta, group)
    for t, expect_zero in ((1, True), (4, False)):
        args = Args(group_size=n, maxrl_truncation=t)
        loss, metrics = ppo_loss(agent, observations, native, old, torch.randn(b), values.flatten(),
                                 values.flatten(), group, group_old, order_weights(n, t), args)
        assert torch.isfinite(loss) and metrics.shape == (11,)
        assert (metrics[6].abs().item() == 0.0) == expect_zero
    scales, advantages = [], torch.randn(b)
    for form in ("maxrl", "linear"):
        args = Args(group_size=n, maxrl_truncation=4, order_form=form)
        _, metrics = ppo_loss(agent, observations, native, old, advantages, values.flatten(),
                              values.flatten(), group, group_old, order_weights(n, 4), args)
        scales.append(metrics[9].item())
    assert scales[0] > 0 and scales[0] == pytest.approx(scales[1], rel=1e-5)


@pytest.mark.parametrize("override", [{"group_size": 1}, {"maxrl_truncation": 0}, {"maxrl_truncation": 17}, {"order_form": "exp"}])
def test_rejects_invalid(override):
    with pytest.raises(ValueError):
        validate_args(Args(num_envs=2, num_steps=8, num_minibatches=2, **override))

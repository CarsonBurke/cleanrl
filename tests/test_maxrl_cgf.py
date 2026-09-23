import math

import pytest
import torch

from cleanrl.maxrl.ppo_continuous_action_maxrl_cgf_v1 import (
    Args, linex_loss, maxrl_advantages, posterior_diagnostics, validate_args,
)


@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_linex_minimizer_is_soft_value(beta):
    targets = torch.randn(4096, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 0.7 + 0.3
    psi = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([psi], line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-14, tolerance_change=1e-16)

    def closure():
        optimizer.zero_grad()
        loss = linex_loss(psi, targets, beta).mean()
        loss.backward()
        return loss

    for _ in range(5):
        optimizer.step(closure)
    expected = (torch.logsumexp(beta * targets, 0) - math.log(targets.numel())) / beta
    assert psi.item() == pytest.approx(expected.item(), abs=1e-9)
    # At the soft value the posterior weights are normalized per state: E[e^{b(G - psi)}] = 1.
    assert torch.exp(beta * (targets - psi.detach())).mean().item() == pytest.approx(1.0, abs=1e-9)
    assert maxrl_advantages(targets - psi.detach(), beta).mean().item() == pytest.approx(0.0, abs=1e-9)


def test_small_beta_recovers_ppo():
    advantages = torch.linspace(-3, 3, 101, dtype=torch.float64)
    torch.testing.assert_close(maxrl_advantages(advantages, 1e-6), advantages, rtol=0, atol=1e-5)
    torch.testing.assert_close(linex_loss(advantages, torch.zeros_like(advantages), 1e-4),
                               0.5 * advantages.square(), rtol=0, atol=1e-3)


def test_uniform_posterior_diagnostics():
    stats = posterior_diagnostics(torch.zeros(1000), torch.tensor(2.0))
    assert stats["maxrl/log_w_mean"].item() == pytest.approx(0.0, abs=1e-6)
    assert stats["maxrl/ess_frac"].item() == pytest.approx(1.0)
    assert stats["maxrl/top10_mass"].item() == pytest.approx(0.1)


@pytest.mark.parametrize("override", [{"maxrl_kappa": 0.0}, {"critic_objective": "huber"}, {"advantage_transform": "tanh"}])
def test_rejects_invalid_maxrl_args(override):
    args = Args(num_envs=2, num_steps=8, num_minibatches=2)
    for key, value in override.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        validate_args(args)


def test_diagnostics_survive_heavy_tails():
    advantages = torch.zeros(1000)
    advantages[0] = 60.0  # e^{bA} overflows fp32 squares; log-space statistics must not
    stats = posterior_diagnostics(advantages, torch.tensor(2.0))
    assert all(torch.isfinite(value) for value in stats.values())
    assert stats["maxrl/ess_frac"].item() == pytest.approx(1e-3, rel=1e-3)
    assert stats["maxrl/top10_mass"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("critic_objective", ["linex", "mse"])
def test_ppo_loss_runs_with_temperature(critic_objective):
    import gymnasium as gym
    import numpy as np
    from types import SimpleNamespace

    from cleanrl.maxrl.ppo_continuous_action_maxrl_cgf_v1 import Agent, ppo_loss

    envs = SimpleNamespace(single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
                           single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float64))
    agent = Agent(envs)
    args = Args(critic_objective=critic_objective)
    n = 32
    observations = torch.randn(n, 17)
    native = torch.rand(n, 6).clamp(0.01, 0.99)
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    loss, metrics = ppo_loss(agent, observations, native, old_logprobs, torch.randn(n),
                             values.flatten() + torch.randn(n), values.flatten(), torch.tensor(1.5), args)
    loss.backward()
    assert torch.isfinite(loss) and metrics.shape == (6,)

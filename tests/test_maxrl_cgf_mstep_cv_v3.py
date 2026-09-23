from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.maxrl.ppo_continuous_action_maxrl_cgf_mstep_cv_v3 import (
    Agent, Args, mstep_loss, posterior_weights, validate_args,
)


def make_agent():
    envs = SimpleNamespace(single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
                           single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float64))
    return Agent(envs)


def test_posterior_weights_have_unit_mean_and_survive_tails():
    advantages = torch.randn(4096)
    advantages[0] = 80.0
    weights = posterior_weights(advantages, torch.tensor(3.0))
    assert torch.isfinite(weights).all()
    assert weights.mean().item() == pytest.approx(1.0, rel=1e-5)


def test_mstep_at_old_policy_has_zero_kl_and_trains_dual():
    torch.manual_seed(0)
    agent = make_agent()
    args = Args()
    n = 64
    observations = torch.randn(n, 17)
    native = torch.rand(n, 6).clamp(0.01, 0.99)
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
    log_multipliers = torch.nn.Parameter(torch.zeros(2))
    loss, metrics = mstep_loss(agent, observations, native, alpha, beta, torch.ones(n),
                               values.flatten() + 0.1, values.flatten(), torch.tensor(1.0),
                               log_multipliers, args)
    loss.backward()
    assert torch.isfinite(loss) and metrics.shape == (7,)
    torch.testing.assert_close(metrics[3:5], torch.zeros(2), atol=1e-6, rtol=0)
    # KL below budget: dual descent must shrink the multipliers (positive gradient on log).
    assert (log_multipliers.grad > 0).all()


def test_decoupled_mean_step_keeps_concentration():
    agent = make_agent()
    args = Args()
    observations = torch.randn(8, 17)
    native = torch.rand(8, 6).clamp(0.01, 0.99)
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
    # Pretend the old policy had a different concentration: the mean-step KL must still vanish
    # when the means agree, so concentration changes are charged only to eps_conc.
    loss, metrics = mstep_loss(agent, observations, native, alpha * 2.0, beta * 2.0, torch.ones(8),
                               values.flatten(), values.flatten(), torch.tensor(1.0),
                               torch.nn.Parameter(torch.zeros(2)), args)
    assert metrics[3].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics[4].item() > 0


def test_rejects_nonpositive_budgets():
    with pytest.raises(ValueError):
        validate_args(Args(num_envs=2, num_steps=8, num_minibatches=2, eps_conc=0.0))


def test_beta_kl_matches_torch():
    from torch.distributions import Beta, kl_divergence

    from cleanrl.maxrl.ppo_continuous_action_maxrl_cgf_mstep_cv_v3 import beta_kl

    params = torch.rand(4, 50, dtype=torch.float64) * 20 + 0.2
    expected = kl_divergence(Beta(params[0], params[1]), Beta(params[2], params[3]))
    torch.testing.assert_close(beta_kl(*params), expected)


def test_mstep_loss_compiles():
    agent = make_agent()
    args = Args()
    observations = torch.randn(16, 17)
    native = torch.rand(16, 6).clamp(0.01, 0.99)
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
    compiled = torch.compile(mstep_loss, fullgraph=True)
    loss, _ = compiled(agent, observations, native, alpha, beta, torch.ones(16), values.flatten(),
                       values.flatten(), torch.tensor(1.0), torch.nn.Parameter(torch.zeros(2)), args)
    assert torch.isfinite(loss)


def test_control_variate_matches_mstep_objective_in_expectation():
    """E_old[(w-1) log pi] - KL(old||pi) equals E_old[w log pi] up to the constant H(old)."""
    from torch.distributions import Beta

    from cleanrl.maxrl.ppo_continuous_action_maxrl_cgf_mstep_cv_v3 import beta_kl

    torch.manual_seed(0)
    old = Beta(torch.tensor(3.0, dtype=torch.float64), torch.tensor(5.0, dtype=torch.float64))
    new = Beta(torch.tensor(4.0, dtype=torch.float64), torch.tensor(4.5, dtype=torch.float64))
    actions = old.sample((400_000,))
    weights = torch.exp(2.0 * actions)
    weights = weights / weights.mean()
    mstep = (weights * new.log_prob(actions)).mean()
    control = ((weights - 1.0) * new.log_prob(actions)).mean() - beta_kl(
        old.concentration1, old.concentration0, new.concentration1, new.concentration0)
    assert (mstep - (control - old.entropy())).abs().item() < 5e-3

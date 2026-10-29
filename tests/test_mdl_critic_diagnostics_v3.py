"""Critic-posterior diagnostics (mdl v3): training identical to v2, instruments exact."""

import types

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.mdl import ppo_continuous_action_mdl_critic_diagnostics_v3 as v3
from cleanrl.mdl import ppo_continuous_action_mdl_critic_posterior_v2 as v2

OBS, ACT, ROWS = 5, 3, 64


def envs():
    return types.SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
    )


def make_agent(module, posterior, log_sigma2=-12.0):
    torch.manual_seed(0)
    agent = module.Agent(envs(), posterior, init_log_sigma2=log_sigma2)
    agent.normalize_matrices()
    return agent


def args_for(module, posterior):
    args = module.Args(posterior=posterior, norm_adv=False, clip_vloss=False, clip_coef_upper=0.2)
    args.batch_size = ROWS
    return args


@pytest.mark.parametrize("posterior", ["none", "mdl", "svd"])
def test_training_loss_is_v2(posterior):
    old, new = make_agent(v2, posterior, -4.0), make_agent(v3, posterior, -4.0)
    torch.manual_seed(1)
    obs = torch.randn(ROWS, OBS)
    native = torch.rand(ROWS, ACT).clamp(1e-3, 1 - 1e-3)
    data = (obs, native, torch.randn(ROWS), torch.randn(ROWS), torch.randn(ROWS), torch.zeros(ROWS))
    extra = () if posterior == "none" else (torch.tensor(0.3),)
    torch.manual_seed(2)
    want = v2.ppo_loss(old, *data, args_for(v2, posterior), *extra)
    torch.manual_seed(2)
    got = v3.ppo_loss(new, *data, args_for(v3, posterior), *extra)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=0)


def test_critic_error_decomposes_mse():
    values, returns = torch.randn(100), torch.randn(100) + 0.7
    parts = v3.critic_error("x", values, returns)
    torch.testing.assert_close(parts["x_bias"].square() + parts["x_residual_var"],
                               (values - returns).square().mean())


def test_gauss_newton_diagonal_matches_per_sample_gradients():
    agent = make_agent(v3, "mdl")
    obs = torch.randn(16, OBS)
    got = v3.gauss_newton_diagonal(agent, obs)
    matrices = agent.critic.matrices()
    want = [torch.zeros_like(m) for m in matrices]
    for row in obs:
        grads = torch.autograd.grad(agent.critic(row[None]).sum(), matrices)
        for total, grad in zip(want, grads):
            total += grad.square() / obs.shape[0]
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-10)


def test_gauss_newton_predicts_the_noise_the_loss_sees():
    # With returns equal to the mean critic, E (v_eps - R)^2 = sum sigma^2 F to second order.
    agent = make_agent(v3, "mdl", log_sigma2=-14.0)
    obs = torch.randn(32, OBS)
    fisher = v3.gauss_newton_diagonal(agent, obs)
    predicted = sum((s.exp() * f).sum() for s, f in zip(agent.critic.log_sigma2, fisher))
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        mean = agent.critic(obs).flatten()
        repeated = obs.repeat(4000, 1)
        draws = agent.lrt_value(repeated, v3.lrt_linear_from(generator)).view(4000, -1)
    measured = (draws - mean).square().mean()
    torch.testing.assert_close(measured, predicted, rtol=0.05, atol=0)


def test_monte_carlo_pull_matches_gauss_newton_at_zero_residual():
    # With R = the mean critic the residual-curvature term vanishes: the unbiased LRT gradient
    # of the noisy value loss with respect to log sigma^2 must equal the Gauss-Newton pull.
    agent = make_agent(v3, "mdl", log_sigma2=-14.0)
    obs = torch.randn(32, OBS).repeat(500, 1)
    with torch.no_grad():
        returns = agent.critic(obs).flatten()
    out = v3.sigma_equilibrium(agent, obs, returns, 1e-6, 0.5, torch.Generator().manual_seed(0), 8,
                               torch.optim.Adam(agent.parameters()))
    torch.testing.assert_close(out["equilibrium/mc_over_gn_pull"], torch.tensor(1.0), rtol=0.05, atol=0)


def test_ensemble_metrics_remove_finite_draw_noise():
    agent = make_agent(v3, "mdl", log_sigma2=-9.0)
    obs = torch.randn(4096, OBS)
    with torch.no_grad():
        decoded = agent.critic(obs).flatten()
    returns = decoded + 0.1 * torch.randn(4096)
    few, many = (v3.posterior_diagnostics("p", agent, obs, returns, decoded, torch.Generator().manual_seed(0), d)
                 for d in (8, 64))
    torch.testing.assert_close(few["p_ensemble_mse"], many["p_ensemble_mse"], rtol=0.1, atol=0)
    torch.testing.assert_close(few["p_noisy_mse"] - few["p_ensemble_mse"], few["p_noise_std"].square(),
                               rtol=0.1, atol=0)


def test_mdl_fixed_point_balances_the_forces():
    agent = make_agent(v3, "mdl")
    obs = torch.randn(ROWS, OBS)
    kl_weight, vf_coef = 1e-4, 0.5
    fisher = v3.gauss_newton_diagonal(agent, obs)
    with torch.no_grad():
        for _ in range(50):  # s_p^2 depends weakly on sigma: iterate to the joint fixed point
            for mean, s, f in zip(agent.critic.matrices(), agent.critic.log_sigma2, fisher):
                prior = (mean.square() + s.exp()).mean()
                s.copy_(-torch.log(1.0 / prior + vf_coef * f / kl_weight))
    with torch.no_grad():
        returns = agent.critic(obs).flatten()
    out = v3.sigma_equilibrium(agent, obs, returns, kl_weight, vf_coef, torch.Generator().manual_seed(0), 2,
                               torch.optim.Adam(agent.parameters()))
    assert abs(float(out["equilibrium/log_sigma2_gap_median"])) < 1e-4
    assert abs(float(out["equilibrium/log_pull_over_push_median"])) < 1e-3
    assert float(out["equilibrium/kl_pushes_up_fraction"]) == 1.0


def test_diagnostics_leave_the_training_stream_alone():
    agent = make_agent(v3, "mdl", log_sigma2=-6.0)
    obs, returns = torch.randn(ROWS, OBS), torch.randn(ROWS)
    state = torch.get_rng_state()
    with torch.no_grad():
        decoded = agent.get_value(obs).flatten()
    out = v3.posterior_diagnostics("p", agent, obs, returns, decoded, torch.Generator().manual_seed(0), 4)
    v3.sigma_equilibrium(agent, obs, returns, 1e-5, 0.5, torch.Generator().manual_seed(1), 2,
                         torch.optim.Adam(agent.parameters()))
    assert torch.equal(torch.get_rng_state(), state)
    assert float(out["p_noise_std"]) > 0
    assert all(p.grad is None for p in agent.parameters())


def test_validate_rejects_bad_diagnostic_settings():
    for kwargs in ({"diagnostic_draws": 1}, {"checkpoint_fractions": (0.0,)}):
        with pytest.raises(ValueError):
            v3.validate_args(v3.Args(**kwargs))

"""Critic MDL with natural-gradient precisions (mdl v4): control fidelity, the precision step."""

import types

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.awr import ppo_continuous_action_ppo_sepclip_control_v1 as control
from cleanrl.mdl import ppo_continuous_action_mdl_critic_natgrad_v4 as v4

OBS, ACT, ROWS = 5, 3, 64


def envs():
    return types.SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
    )


def make_agent(posterior, log_sigma2=-12.0):
    torch.manual_seed(0)
    agent = v4.Agent(envs(), posterior, init_log_sigma2=log_sigma2)
    agent.normalize_matrices()
    return agent


def args_for(module, posterior="none"):
    args = module.Args(norm_adv=False, clip_vloss=False, clip_coef_upper=0.2)
    if hasattr(args, "posterior"):
        args.posterior = posterior
    args.batch_size = ROWS
    return args


def data():
    torch.manual_seed(1)
    obs = torch.randn(ROWS, OBS)
    native = torch.rand(ROWS, ACT).clamp(1e-3, 1 - 1e-3)
    return obs, native, torch.randn(ROWS), torch.randn(ROWS), torch.randn(ROWS), torch.zeros(ROWS)


def test_none_is_the_control():
    torch.manual_seed(0)
    reference = control.Agent(envs())
    reference.normalize_matrices()
    agent = make_agent("none")
    assert reference.state_dict().keys() == agent.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(agent.state_dict()[key], value, rtol=0, atol=0)
    want = control.ppo_loss(reference, *data(), args_for(control))
    got = v4.ppo_loss(agent, *data(), args_for(v4))
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=0)


def test_precisions_are_buffers_outside_the_optimiser():
    agent = make_agent("mdl")
    names = {name for name, _ in agent.named_buffers()}
    assert {f"critic.log_sigma2_{i}" for i in range(7)} <= names
    assert all(not name.startswith("critic.log_sigma2") for name, _ in agent.named_parameters())
    assert agent.actor.log_sigma2 is None


def test_vanishing_noise_recovers_the_control_loss():
    exact, noisy = make_agent("none"), make_agent("mdl", log_sigma2=-60.0)
    noisy.load_state_dict({**noisy.state_dict(), **exact.state_dict()})
    want = v4.ppo_loss(exact, *data(), args_for(v4))[0]
    got = v4.ppo_loss(noisy, *data(), args_for(v4, "mdl"))[0]
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)


def test_full_step_reaches_the_gauss_newton_fixed_point():
    agent = make_agent("mdl")
    obs, returns = torch.randn(ROWS, OBS), torch.randn(ROWS)
    kl_weight, vf_coef = torch.tensor(1e-5), 0.5
    for _ in range(30):  # s_p^2 contains sigma^2: repeated full steps converge to the joint fixed point
        v4.natural_precision_step(agent, obs, kl_weight, vf_coef, 1.0)
    out = v4.sigma_equilibrium(agent, obs, returns, kl_weight, vf_coef, torch.Generator().manual_seed(0), 2)
    assert float(out["equilibrium/log_sigma2_gap_abs_mean"]) < 1e-4
    # At the fixed point, every weight's data pull balances the KL push.
    assert abs(float(out["equilibrium/log_pull_over_push_median"])) < 1e-3


def test_partial_step_interpolates_precision():
    agent = make_agent("mdl")
    obs = torch.randn(ROWS, OBS)
    kl_weight, vf_coef = torch.tensor(1e-5), 0.5
    before = [(-s).exp().clone() for s in agent.critic.log_sigma2]
    targets = [v4.precision_target(m, s, f, kl_weight, vf_coef) for m, s, f in
               zip(agent.critic.matrices(), agent.critic.log_sigma2, v4.gauss_newton_diagonal(agent, obs))]
    v4.natural_precision_step(agent, obs, kl_weight, vf_coef, 0.25)
    for lam0, target, s in zip(before, targets, agent.critic.log_sigma2):
        torch.testing.assert_close((-s).exp(), 0.75 * lam0 + 0.25 * target, rtol=1e-5, atol=0)


def test_kl_weight_sets_the_noise_scale():
    # More description cost per nat (larger kl_weight) admits more noise; none at all recovers the means.
    obs = torch.randn(ROWS, OBS)
    medians = []
    for kl_weight in (1e-7, 1e-5, 1e-3):
        agent = make_agent("mdl")
        v4.natural_precision_step(agent, obs, torch.tensor(kl_weight), 0.5, 1.0)
        medians.append(torch.cat([s.flatten() for s in agent.critic.log_sigma2]).median())
    assert medians[0] < medians[1] < medians[2]


def test_monte_carlo_pull_matches_gauss_newton_at_zero_residual():
    agent = make_agent("mdl", log_sigma2=-14.0)
    obs = torch.randn(32, OBS).repeat(500, 1)
    with torch.no_grad():
        returns = agent.critic(obs).flatten()
    out = v4.sigma_equilibrium(agent, obs, returns, torch.tensor(1e-6), 0.5,
                               torch.Generator().manual_seed(0), 8)
    torch.testing.assert_close(out["equilibrium/mc_over_gn_pull"], torch.tensor(1.0), rtol=0.05, atol=0)


def test_step_and_diagnostics_leave_rng_and_grads_alone():
    agent = make_agent("mdl", log_sigma2=-6.0)
    obs, returns = torch.randn(ROWS, OBS), torch.randn(ROWS)
    state = torch.get_rng_state()
    v4.natural_precision_step(agent, obs, torch.tensor(1e-5), 0.5, 1.0)
    v4.sigma_equilibrium(agent, obs, returns, torch.tensor(1e-5), 0.5, torch.Generator().manual_seed(0), 2)
    assert torch.equal(torch.get_rng_state(), state)
    assert all(p.grad is None for p in agent.parameters())


def test_validate_args():
    for kwargs in ({"posterior": "svd"}, {"posterior": "mdl", "kl_scale": 0.0},
                   {"posterior": "mdl", "precision_step": 0.0}, {"posterior": "mdl", "precision_step": 1.5},
                   {"posterior": "mdl", "clip_vloss": True}):
        with pytest.raises(ValueError):
            v4.validate_args(v4.Args(**kwargs))
    v4.validate_args(v4.Args(posterior="mdl", clip_vloss=False))

"""Predictive-mean critic MDL (mdl v6): v5's posterior contracts, plus the values GAE uses."""

import copy
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.awr import ppo_continuous_action_ppo_sepclip_control_v1 as control
from cleanrl.mdl import ppo_continuous_action_mdl_critic_natgrad_v4 as v4
from cleanrl.mdl import ppo_continuous_action_mdl_critic_predictive_v6 as v6

OBS, ACT, ROWS = 5, 3, 64


def envs():
    return types.SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
    )


def make_agent(posterior, log_sigma2=-12.0):
    torch.manual_seed(0)
    agent = v6.Agent(envs(), posterior, init_log_sigma2=log_sigma2)
    agent.normalize_matrices()
    return agent


def args_for(module, posterior="none", prior="sequential"):
    args = module.Args(norm_adv=False, clip_vloss=False, clip_coef_upper=0.2)
    if hasattr(args, "posterior"):
        args.posterior, args.prior = posterior, prior
    args.batch_size = ROWS
    return args


def data():
    torch.manual_seed(1)
    obs = torch.randn(ROWS, OBS)
    native = torch.rand(ROWS, ACT).clamp(1e-3, 1 - 1e-3)
    return obs, native, torch.randn(ROWS), torch.randn(ROWS), torch.randn(ROWS), torch.zeros(ROWS)


def flat(tensors):
    return torch.cat([t.flatten() for t in tensors])


def test_none_is_the_control():
    torch.manual_seed(0)
    reference = control.Agent(envs())
    reference.normalize_matrices()
    agent = make_agent("none")
    assert reference.state_dict().keys() == agent.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(agent.state_dict()[key], value, rtol=0, atol=0)
    want = control.ppo_loss(reference, *data(), args_for(control))
    got = v6.ppo_loss(agent, *data(), args_for(v6), torch.tensor(0.3))
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=0)


def test_posterior_and_prior_are_buffers_outside_the_optimiser():
    agent = make_agent("mdl")
    names = {name for name, _ in agent.named_buffers()}
    for kind in ("log_sigma2", "prior_precision", "prior_mean"):
        assert {f"critic.{kind}_{i}" for i in range(7)} <= names
    assert not any("sigma2" in name or "prior" in name for name, _ in agent.named_parameters())
    assert agent.actor.log_sigma2 is None


@pytest.mark.parametrize("prior", v6.PRIORS)
def test_vanishing_noise_at_the_prior_mean_recovers_the_control_loss(prior):
    exact, noisy = make_agent("none"), make_agent("mdl", log_sigma2=-60.0)
    noisy.load_state_dict({**noisy.state_dict(), **exact.state_dict()})
    with torch.no_grad():
        for mean, prior_mean in zip(noisy.critic.matrices(), noisy.critic.prior_mean):
            prior_mean.copy_(mean)  # nothing moved since the prior: the anchor is zero
    want = v6.ppo_loss(exact, *data(), args_for(v6))[0]
    got = v6.ppo_loss(noisy, *data(), args_for(v6, "mdl", prior), torch.tensor(0.3))[0]
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)


def test_sequential_loss_adds_the_kl_mean_term():
    agent = make_agent("mdl", log_sigma2=-8.0)
    torch.manual_seed(3)
    with torch.no_grad():
        for precision, prior_mean in zip(agent.critic.prior_precision, agent.critic.prior_mean):
            precision.uniform_(1.0, 50.0)
            prior_mean.normal_(0.0, 0.1)
    kl_weight = torch.tensor(0.02)
    torch.manual_seed(2)
    sequential = v6.ppo_loss(agent, *data(), args_for(v6, "mdl", "sequential"), kl_weight)[0]
    torch.manual_seed(2)  # same LRT draw
    empirical = v6.ppo_loss(agent, *data(), args_for(v6, "mdl", "empirical"), kl_weight)[0]
    want = 0.5 * sum((p * (m - pm).square()).sum() for m, p, pm in
                     zip(agent.critic.matrices(), agent.critic.prior_precision, agent.critic.prior_mean))
    torch.testing.assert_close(sequential - empirical, kl_weight * want)
    torch.testing.assert_close(agent.anchor(), want)


def test_gaussian_kl_matches_torch():
    torch.manual_seed(0)
    mean, log_sigma2 = torch.randn(6, 7), torch.randn(6, 7) - 3.0
    precision, prior_mean = torch.rand(6, 7) + 0.5, torch.randn(6, 7)
    want = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean, (0.5 * log_sigma2).exp()),
        torch.distributions.Normal(prior_mean, precision.rsqrt())).sum()
    torch.testing.assert_close(v6.gaussian_kl(mean, log_sigma2, precision, prior_mean), want)


def test_empirical_prior_is_v4s_profiled_code():
    agent = make_agent("mdl", log_sigma2=-5.0)
    v6.set_empirical_prior(agent)
    want = sum(v4.mdl_kl(m, s) for m, s in zip(agent.critic.matrices(), agent.critic.log_sigma2))
    torch.testing.assert_close(agent.description_length(), want)
    want = 0.5 * sum(p[0, 0] * m.square().sum() for m, p in zip(agent.critic.matrices(), agent.critic.prior_precision))
    torch.testing.assert_close(agent.anchor(), want)


def test_sequential_prior_tempers_the_previous_posterior():
    agent = make_agent("mdl", log_sigma2=-6.0)
    torch.manual_seed(4)
    with torch.no_grad():
        for s in agent.critic.log_sigma2:
            s.uniform_(-10.0, -4.0)
    means, log_sigma2 = agent.critic.matrices(), [s.clone() for s in agent.critic.log_sigma2]
    gamma = 0.7
    v6.set_sequential_prior(agent, gamma)
    for mean, s, precision, prior_mean in zip(means, log_sigma2, agent.critic.prior_precision,
                                              agent.critic.prior_mean):
        base = 1.0 / (mean.square() + s.exp()).mean()
        want = gamma * (-s).exp() + (1 - gamma) * base
        torch.testing.assert_close(precision, want)
        torch.testing.assert_close(prior_mean, gamma * (-s).exp() * mean / want)
    # Without forgetting the prior is the posterior itself: the update starts at zero description cost.
    v6.set_sequential_prior(agent, 1.0 - 1e-9)
    assert abs(float(agent.description_length())) < 1e-2


def test_curvature_is_per_sample_gradients_at_a_weight_sample():
    agent = make_agent("mdl", log_sigma2=-3.0)
    obs = torch.randn(16, OBS)
    got = v6.gauss_newton_under_posterior(agent, obs, torch.Generator().manual_seed(7), draws=1)
    generator = torch.Generator().manual_seed(7)  # replay the same weight sample
    sample = [(m + (0.5 * s).exp() * torch.randn(m.shape, generator=generator)).detach().requires_grad_()
              for m, s in zip(agent.critic.matrices(), agent.critic.log_sigma2)]
    dense = lambda x, w: torch.nn.functional.linear(x, w)
    want = [torch.zeros_like(m) for m in sample]
    for row in obs:
        grads = torch.autograd.grad(agent.critic.forward_with(row[None], tuple(sample), dense).sum(), sample)
        for total, grad in zip(want, grads):
            total += grad.square() / obs.shape[0]
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-10)


def test_curvature_matches_the_exact_sigma_derivative_at_zero_residual():
    # Price: d E_q[1/2 (v - R)^2] / d sigma^2 = 1/2 E_q[F] + residual curvature, which vanishes as R -> v.
    agent = make_agent("mdl", log_sigma2=-14.0)
    obs = torch.randn(32, OBS).repeat(500, 1)
    with torch.no_grad():
        returns = agent.critic(obs).flatten()
    out = v6.sigma_equilibrium(agent, obs, returns, torch.tensor(1e-6), 0.5, torch.Generator().manual_seed(0), 8)
    torch.testing.assert_close(out["equilibrium/mc_over_gn_pull"], torch.tensor(1.0), rtol=0.05, atol=0)


def precision_targets(agent, obs, kl_weight, vf_coef, generator, draws):
    fisher = v6.gauss_newton_under_posterior(agent, obs, generator, draws)
    return [p + vf_coef * f / kl_weight for p, f in zip(agent.critic.prior_precision, fisher)]


def test_first_iteration_is_a_full_step_then_damped():
    agent = make_agent("mdl", log_sigma2=-6.0)
    obs = torch.randn(ROWS, OBS)
    kl_weight, vf_coef = torch.tensor(1e-5), 0.5
    v6.set_empirical_prior(agent)
    replay = copy.deepcopy(agent)
    generator = torch.Generator().manual_seed(9)
    first = precision_targets(replay, obs, kl_weight, vf_coef, generator, 3)
    with torch.no_grad():
        for s, target in zip(replay.critic.log_sigma2, first):
            s.copy_(-target.log())
    second = precision_targets(replay, obs, kl_weight, vf_coef, generator, 3)
    v6.natural_precision_step(agent, obs, kl_weight, vf_coef, torch.Generator().manual_seed(9), 3, 2, 0.25)
    for lam1, target, s in zip(first, second, agent.critic.log_sigma2):
        torch.testing.assert_close((-s).exp(), 0.75 * lam1 + 0.25 * target, rtol=1e-5, atol=0)


def converged_empirical(obs, kl_weight, vf_coef, draws=2, iterations=60):
    agent = make_agent("mdl")
    v6.natural_precision_step(agent, obs, kl_weight, vf_coef, torch.Generator().manual_seed(0), draws, iterations,
                              0.5, v6.set_empirical_prior)
    return agent


@pytest.mark.parametrize("prior", v6.PRIORS)
def test_iterations_reach_the_self_consistent_fixed_point(prior):
    # F_q is a Monte-Carlo estimate: at the fixed point a fresh estimate scatters around the current
    # sigma (more so for noisy weights, less with more draws) but is centred on it.
    obs, returns = torch.randn(ROWS, OBS), torch.randn(ROWS)
    kl_weight, vf_coef = torch.tensor(1e-6), 0.5
    scatter = []
    for draws in (2, 16):
        if prior == "sequential":  # a realistic previous posterior, after its means moved
            agent = converged_empirical(obs, kl_weight, vf_coef)
            with torch.no_grad():
                for mean in agent.critic.matrices():
                    mean.add_(0.05 * torch.randn_like(mean))
            agent.normalize_matrices()
            v6.set_sequential_prior(agent, 0.9)
            refresh = None
        else:
            agent, refresh = make_agent("mdl"), v6.set_empirical_prior
        v6.natural_precision_step(agent, obs, kl_weight, vf_coef, torch.Generator().manual_seed(0), draws, 60, 0.5,
                                  refresh)
        out = v6.sigma_equilibrium(agent, obs, returns, kl_weight, vf_coef, torch.Generator().manual_seed(1), draws)
        assert abs(float(out["equilibrium/log_sigma2_gap_median"])) < 0.05
        assert abs(float(out["equilibrium/log_pull_over_push_median"])) < 0.05
        scatter.append(float(out["equilibrium/log_sigma2_gap_abs_mean"]))
    assert scatter[1] < 0.1 and scatter[1] < scatter[0] / 2


def test_default_schedule_reaches_the_fixed_point_from_the_initialisation():
    # The first rollout starts from init_log_sigma2; the default schedule must not carry it over.
    obs = torch.randn(ROWS, OBS)
    kl_weight, vf_coef = torch.tensor(1e-4), 0.5
    args = v6.Args(posterior="mdl")
    agent = make_agent("mdl", log_sigma2=args.init_log_sigma2)
    v6.fit_posterior(agent, obs, kl_weight, args, torch.Generator().manual_seed(0), "empirical", cold=True)
    reference = converged_empirical(obs, kl_weight, vf_coef, draws=16, iterations=100)
    gap = flat(agent.critic.log_sigma2) - flat(reference.critic.log_sigma2)
    assert abs(float(gap.median())) < 0.1


def test_fit_posterior_priors():
    obs = torch.randn(ROWS, OBS)
    kl_weight, args = torch.tensor(1e-5), v6.Args(posterior="mdl", forgetting=0.8)
    agent = converged_empirical(obs, kl_weight, args.vf_coef)
    previous = copy.deepcopy(agent)
    v6.fit_posterior(agent, obs, kl_weight, args, torch.Generator().manual_seed(1), "sequential")
    v6.set_sequential_prior(previous, args.forgetting)  # the prior comes from the previous posterior
    for got, want in zip(agent.critic.prior_precision + agent.critic.prior_mean,
                         previous.critic.prior_precision + previous.critic.prior_mean):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    # A refit keeps the prior and moves only sigma.
    prior = [t.clone() for t in agent.critic.prior_precision + agent.critic.prior_mean]
    with torch.no_grad():
        for mean in agent.critic.matrices():
            mean.add_(0.05 * torch.randn_like(mean))
    agent.normalize_matrices()
    before = flat(agent.critic.log_sigma2).clone()
    v6.fit_posterior(agent, obs, kl_weight, args, torch.Generator().manual_seed(2), "kept")
    for got, want in zip(agent.critic.prior_precision + agent.critic.prior_mean, prior):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert not torch.equal(flat(agent.critic.log_sigma2), before)


def test_compiled_loss_sees_in_place_prior_and_weight_updates():
    # Inductor draws its own random numbers: with negligible noise the compiled and eager losses agree.
    agent = make_agent("mdl", log_sigma2=-60.0)
    args = args_for(v6, "mdl", "sequential")
    kl_weight = torch.tensor(0.01)
    compiled = torch.compile(lambda *inputs: v6.ppo_loss(agent, *inputs, args, kl_weight)[0], fullgraph=True,
                             dynamic=False)
    for step in range(3):
        with torch.no_grad():
            kl_weight.fill_(0.01 * (step + 1))
            for precision, prior_mean in zip(agent.critic.prior_precision, agent.critic.prior_mean):
                precision.uniform_(1.0, 50.0)
                prior_mean.normal_(0.0, 0.1)
        torch.manual_seed(step)
        got = compiled(*data())
        torch.manual_seed(step)
        want = v6.ppo_loss(agent, *data(), args, kl_weight)[0]
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_kl_weight_sets_the_noise_scale():
    obs = torch.randn(ROWS, OBS)
    medians = []
    for kl_weight in (1e-7, 1e-5, 1e-3):
        agent = make_agent("mdl")
        v6.natural_precision_step(agent, obs, torch.tensor(kl_weight), 0.5, torch.Generator().manual_seed(0),
                                  2, 4, 0.5, v6.set_empirical_prior)
        medians.append(flat(agent.critic.log_sigma2).median())
    assert medians[0] < medians[1] < medians[2]


def test_sequential_prior_accumulates_precision_over_the_stream():
    # A stream of identical rollouts accumulates precision to the 1/(1 - gamma) effective sample count.
    agent = make_agent("mdl")
    obs = torch.randn(ROWS, OBS)
    kl_weight, vf_coef, gamma = torch.tensor(1e-5), 0.5, 0.75
    v6.natural_precision_step(agent, obs, kl_weight, vf_coef, torch.Generator().manual_seed(0), 2, 30, 1.0,
                              v6.set_empirical_prior)
    single = flat(agent.critic.log_sigma2).clone()
    for _ in range(40):
        v6.set_sequential_prior(agent, gamma)
        v6.natural_precision_step(agent, obs, kl_weight, vf_coef, torch.Generator().manual_seed(0), 2, 1, 1.0)
    # Steady state of lambda = gamma lambda + (1 - gamma)/s_p^2 + G:  lambda = 1/s_p^2 + G / (1 - gamma).
    data = [vf_coef * f / kl_weight for f in
            v6.gauss_newton_under_posterior(agent, obs, torch.Generator().manual_seed(0), 2)]
    base = [v6.base_precision(m, s) for m, s in zip(agent.critic.matrices(), agent.critic.log_sigma2)]
    torch.testing.assert_close(flat((-s).exp() for s in agent.critic.log_sigma2),
                               flat(b + g / (1 - gamma) for b, g in zip(base, data)), rtol=5e-3, atol=0)
    # Each rollout's data then counts 1/(1 - gamma) times relative to the one-rollout posterior's.
    assert flat(agent.critic.log_sigma2).median() < single.median() - 1.0


def test_step_and_diagnostics_leave_rng_and_grads_alone():
    agent = make_agent("mdl", log_sigma2=-6.0)
    obs, returns = torch.randn(ROWS, OBS), torch.randn(ROWS)
    state = torch.get_rng_state()
    v6.set_sequential_prior(agent, 0.9)
    v6.natural_precision_step(agent, obs, torch.tensor(1e-5), 0.5, torch.Generator().manual_seed(0), 2, 2, 0.5)
    v6.sigma_equilibrium(agent, obs, returns, torch.tensor(1e-5), 0.5, torch.Generator().manual_seed(0), 2)
    v6.posterior_telemetry(agent, torch.tensor(1e-5), True)
    assert torch.equal(torch.get_rng_state(), state)
    assert all(p.grad is None for p in agent.parameters())


def test_validate_args():
    for kwargs in ({"posterior": "svd"}, {"posterior": "mdl", "kl_scale": 0.0},
                   {"posterior": "mdl", "precision_damping": 0.0}, {"posterior": "mdl", "precision_damping": 1.5},
                   {"posterior": "mdl", "precision_draws": 0}, {"posterior": "mdl", "precision_iterations": 0},
                   {"posterior": "mdl", "prior": "flat"}, {"posterior": "mdl", "forgetting": 1.0},
                   {"posterior": "mdl", "clip_vloss": True}, {"posterior": "mdl", "predictive_draws": 0}):
        with pytest.raises(ValueError):
            v6.validate_args(v6.Args(**kwargs))
    v6.validate_args(v6.Args(posterior="mdl", clip_vloss=False))


def test_predictive_value_is_the_mean_of_the_sampled_critics():
    agent = make_agent("mdl", log_sigma2=-4.0)
    agent.resample_predictive(torch.Generator().manual_seed(3))
    x = torch.randn(ROWS, OBS)
    dense = lambda inputs, weight: torch.nn.functional.linear(inputs, weight)
    critic = agent.critic
    want = torch.stack([
        critic.forward_with(x, tuple(m + (0.5 * s).exp() * n[k] for m, s, n in
                                     zip(critic.matrices(), critic.log_sigma2, critic.predictive_noise)), dense)
        for k in range(critic.predictive_noise[0].shape[0])]).mean(0)
    torch.testing.assert_close(agent.get_value(x), want, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(agent.get_policy_and_value(x)[2], want, rtol=1e-5, atol=1e-6)


def test_predictive_value_without_noise_is_the_critic_at_its_means():
    agent = make_agent("mdl", log_sigma2=-60.0)
    agent.resample_predictive(torch.Generator().manual_seed(3))
    x = torch.randn(ROWS, OBS)
    torch.testing.assert_close(agent.get_value(x), agent.critic(x), rtol=1e-5, atol=1e-6)


def test_predictive_value_is_what_the_noisy_loss_fits():
    # The loss's LRT critic and the weight-sample ensemble share E_q[v]; the critic at its means does not.
    agent = make_agent("mdl", log_sigma2=-3.0)
    x = torch.randn(32, OBS)
    with torch.no_grad():
        lrt = torch.stack([agent.lrt_value(x, v6.lrt_linear_from(torch.Generator().manual_seed(k)))
                           for k in range(2000)]).mean(0)
        predictive = []
        for k in range(125):
            agent.resample_predictive(torch.Generator().manual_seed(1000 + k))
            predictive.append(agent.get_value(x))
        predictive = torch.stack(predictive).mean(0)
        at_means = agent.critic(x)
    error = (predictive - lrt).abs().mean()
    assert error < 0.2 * (at_means - lrt).abs().mean()


def test_resampling_uses_only_its_generator():
    agent = make_agent("mdl")
    state = torch.get_rng_state()
    agent.resample_predictive(torch.Generator().manual_seed(0))
    first = [n.clone() for n in agent.critic.predictive_noise]
    agent.resample_predictive(torch.Generator().manual_seed(0))
    assert torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(a, b) for a, b in zip(first, agent.critic.predictive_noise))
    assert first[0].shape == (v6.Args().predictive_draws,) + tuple(agent.critic.matrices()[0].shape)

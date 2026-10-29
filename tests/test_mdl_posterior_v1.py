"""MDL / sparse-VD weight posteriors (mdl v1): control fidelity, host parity, KLs, replay."""

import math
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.awr import ppo_continuous_action_ppo_sepclip_control_v1 as control
from cleanrl.mdl import ppo_continuous_action_mdl_posterior_v1 as v1

OBS, ACT, ENVS = 5, 3, 4


def envs():
    return types.SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
    )


def make_agent(posterior, log_sigma2=-12.0, seed=0):
    torch.manual_seed(seed)
    agent = v1.Agent(envs(), posterior, init_log_sigma2=log_sigma2)
    agent.normalize_matrices()
    return agent


def test_none_is_the_control_agent():
    torch.manual_seed(0)
    reference = control.Agent(envs())
    agent = make_agent("none")
    reference.normalize_matrices()
    assert reference.state_dict().keys() == agent.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(agent.state_dict()[key], value, rtol=0, atol=0)
    x = torch.randn(64, OBS)
    for got, want in zip(agent.get_policy_and_value(x), reference.get_policy_and_value(x)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_functional_forward_matches_module():
    agent = make_agent("mdl")
    x = torch.randn(64, OBS)
    for trunk in (agent.actor, agent.critic):
        torch.testing.assert_close(trunk.forward_with(x, trunk.matrices(), v1.dense_linear), trunk(x))


def test_per_env_host_mirror_matches_device_forward():
    agent = make_agent("svd", log_sigma2=-4.0)  # material noise: samples differ per env
    noise = agent.sample_actor_noise(ENVS)
    with torch.no_grad():
        agent.actor.head[0].bias.normal_()
        agent.actor.readout_gain.uniform_(0.5, 1.5)
        samples = agent.actor_samples(noise)
        mirror = v1.PerEnvHostMirror(agent.actor, ENVS)
        mirror.refresh(samples)
        obs = np.random.default_rng(0).standard_normal((ENVS, OBS)).astype(np.float32)
        want = agent.actor.forward_with(torch.from_numpy(obs)[:, None], samples, v1.env_linear)[:, 0]
    np.testing.assert_allclose(mirror(obs), want.numpy(), rtol=1e-5, atol=1e-5)
    assert not np.allclose(want[0].numpy(), want[1].numpy())


def test_mdl_kl_profiles_the_optimal_gaussian_prior():
    torch.manual_seed(0)
    mean = torch.randn(7, 9, dtype=torch.float64, requires_grad=True)
    log_sigma2 = (torch.randn(7, 9, dtype=torch.float64) - 3.0).requires_grad_()
    s2 = (mean.square() + log_sigma2.exp()).mean()
    explicit = 0.5 * (log_sigma2.exp() / s2 + mean.square() / s2 - 1 + s2.log() - log_sigma2).sum()
    profiled = v1.mdl_kl(mean, log_sigma2)
    torch.testing.assert_close(profiled, explicit)
    # Envelope theorem: differentiating through the optimal s^2 equals holding it fixed.
    fixed = 0.5 * (log_sigma2.exp() / s2.detach() + mean.square() / s2.detach() - 1
                   + s2.detach().log() - log_sigma2).sum()
    for got, want in zip(torch.autograd.grad(profiled, (mean, log_sigma2)),
                         torch.autograd.grad(fixed, (mean, log_sigma2))):
        torch.testing.assert_close(got, want)


def test_svd_kl_matches_exact_log_uniform_kl_up_to_its_constant():
    # Exact: -KL(N(t, a t^2) || log-uniform) = 0.5 log a - E_{e~N(1, a)} log|e| + C.
    log_alphas = torch.linspace(-6.0, 6.0, 25, dtype=torch.float64)
    # Dense midpoint rule over z ~ N(0, 1); log|1 + s z| has an integrable singularity.
    z = np.linspace(-12.0, 12.0, 4_000_001)
    z = 0.5 * (z[1:] + z[:-1])
    density = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) * (z[1] - z[0])
    exact = []
    for la in log_alphas.tolist():
        expected_log = (density * np.log(np.abs(1.0 + math.exp(0.5 * la) * z))).sum()
        exact.append(-(0.5 * la - expected_log))
    exact = torch.tensor(exact, dtype=torch.float64)
    mean = torch.ones_like(log_alphas)
    approx = torch.stack([v1.svd_kl(m.view(1), (la + m.square().log()).view(1))
                          for m, la in zip(mean, log_alphas)])
    offset = (approx - exact).mean()
    assert (approx - exact - offset).abs().max() < 0.02
    assert torch.all(approx[1:] < approx[:-1])  # more noise, fewer bits
    assert float(v1.svd_kl(torch.ones(1), torch.tensor([20.0]))) < 1e-4


def test_lrt_moments_match_explicit_weight_sampling():
    torch.manual_seed(0)
    mean, variance, x = torch.randn(3, 4), torch.rand(3, 4) * 0.5, torch.randn(1, 4)
    draws = torch.cat([v1.lrt_linear(x.expand(200_000, 4), (mean, variance))])
    torch.testing.assert_close(draws.mean(0), (x @ mean.T)[0], atol=1e-2, rtol=0)
    torch.testing.assert_close(draws.var(0), (x.square() @ variance.T)[0], atol=2e-2, rtol=2e-2)


def test_replayed_noise_gives_unit_first_ratio():
    agent = make_agent("mdl", log_sigma2=-4.0)
    args = v1.Args(posterior="mdl", norm_adv=False, clip_vloss=False, clip_coef_upper=0.2)
    args.batch_size = ENVS * 32
    noise = agent.sample_actor_noise(ENVS)
    obs = torch.randn(ENVS, 32, OBS)
    with torch.no_grad():
        alpha, beta = agent.sampled_policy(obs, noise)
        native = torch.distributions.Beta(alpha, beta).sample().clamp(1e-4, 1 - 1e-4)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    adv, ret, old_values = torch.randn(ENVS, 32), torch.randn(ENVS, 32), torch.zeros(ENVS, 32)
    scale, variance = adv.std(), (ret - old_values).square().mean()
    loss, metrics = v1.posterior_ppo_loss(agent, obs, native, old_logprobs, adv, ret, old_values, noise,
                                          scale, variance, args)
    assert abs(float(metrics[3])) < 1e-6 and float(metrics[5]) == 0.0
    # The ELBO reaches the posterior variances through both the replayed samples and the KL.
    loss.backward()
    assert all(p.grad is not None for p in agent.actor.log_sigma2)


def test_kl_pushes_noise_up_when_data_is_silent():
    agent = make_agent("mdl")
    for kl in (v1.mdl_kl, v1.svd_kl):
        log_sigma2 = agent.actor.log_sigma2[0]
        (grad,) = torch.autograd.grad(kl(agent.actor.matrices()[0].detach(), log_sigma2), (log_sigma2,))
        assert torch.all(grad < 0)


def test_group_layout_round_trips_and_keeps_segments_contiguous_in_time():
    steps, segment = 32, 8
    step_major = torch.arange(steps * ENVS, dtype=torch.float32).view(steps, ENVS)  # value = t*E + e
    grouped = v1.steps_to_groups(step_major, segment)
    assert grouped.shape == (ENVS * steps // segment, segment) and grouped.is_contiguous()
    for g in range(grouped.shape[0]):
        env, seg = divmod(g, steps // segment)
        times = seg * segment + torch.arange(segment)
        torch.testing.assert_close(grouped[g], times.float() * ENVS + env)
    back = v1.groups_to_steps(grouped, ENVS)
    assert back.is_contiguous()
    torch.testing.assert_close(back, step_major)
    features = torch.randn(steps, ENVS, OBS)
    torch.testing.assert_close(v1.steps_to_groups(features, segment)[5],
                               features[(5 % 4) * segment:(5 % 4 + 1) * segment, 5 // 4])


def test_group_minibatches_partition_each_group():
    generator = torch.Generator().manual_seed(0)
    batches = v1.group_minibatches(ENVS, 32, 4, "cpu", generator)
    assert len(batches) == 4 and all(b.shape == (ENVS, 8) for b in batches)
    covered = torch.cat(batches, dim=1).sort(dim=1).values
    assert torch.equal(covered, torch.arange(32).expand(ENVS, 32))


def test_svd_decoding_prunes_noisy_weights():
    agent = make_agent("svd")
    with torch.no_grad():
        agent.critic.log_sigma2[0].fill_(10.0)  # log alpha >> 3 for every weight in gate1
    decoded = agent.decoded_matrices(agent.critic)
    assert torch.count_nonzero(decoded[0]) == 0
    torch.testing.assert_close(decoded[1], agent.critic.matrices()[1])


def test_validate_rejects_uneven_env_split():
    args = v1.Args(posterior="svd", num_steps=1024, noise_segment=48)
    with pytest.raises(ValueError):
        v1.validate_args(args)

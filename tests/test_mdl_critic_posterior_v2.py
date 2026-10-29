"""Critic-only MDL / sparse-VD posteriors (mdl v2): control fidelity, ELBO wiring, decoding."""

import types

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.awr import ppo_continuous_action_ppo_sepclip_control_v1 as control
from cleanrl.mdl import ppo_continuous_action_mdl_critic_posterior_v2 as v2
from cleanrl.mdl import ppo_continuous_action_mdl_posterior_v1 as v1

OBS, ACT, ROWS = 5, 3, 64


def envs():
    return types.SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
    )


def make_agent(posterior, log_sigma2=-12.0):
    torch.manual_seed(0)
    agent = v2.Agent(envs(), posterior, init_log_sigma2=log_sigma2)
    agent.normalize_matrices()
    return agent


def batch(agent):
    torch.manual_seed(1)
    obs = torch.randn(ROWS, OBS)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(obs)
        native = torch.distributions.Beta(alpha, beta).sample().clamp(1e-4, 1 - 1e-4)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    return obs, native, old_logprobs, torch.randn(ROWS), torch.randn(ROWS), torch.zeros(ROWS)


def args_for(posterior, **kw):
    args = v2.Args(posterior=posterior, norm_adv=False, clip_vloss=False, clip_coef_upper=0.2, **kw)
    args.batch_size = ROWS
    return args


def test_none_is_the_control():
    torch.manual_seed(0)
    reference = control.Agent(envs())
    reference.normalize_matrices()
    agent = make_agent("none")
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(agent.state_dict()[key], value, rtol=0, atol=0)
    assert reference.state_dict().keys() == agent.state_dict().keys()
    data = batch(agent)
    want_loss, want_metrics = control.ppo_loss(reference, *data, args_for("none"))
    got_loss, got_metrics = v2.ppo_loss(agent, *data, args_for("none"))
    torch.testing.assert_close(got_loss, want_loss, rtol=0, atol=0)
    torch.testing.assert_close(got_metrics, want_metrics, rtol=0, atol=0)


@pytest.mark.parametrize("posterior", ["mdl", "svd"])
def test_posterior_lives_on_the_critic_only(posterior):
    agent = make_agent(posterior, log_sigma2=-4.0)
    assert agent.actor.log_sigma2 is None
    data = batch(agent)
    loss, metrics = v2.ppo_loss(agent, *data, args_for(posterior), torch.tensor(0.5))
    assert metrics.shape == (7,)
    torch.testing.assert_close(metrics[6], agent.description_length().detach())
    assert abs(float(metrics[3])) < 1e-6  # the actor is deterministic: first ratio is one
    loss.backward()
    assert all(p.grad is not None and torch.all(p.grad != 0) for p in agent.critic.log_sigma2)


@pytest.mark.parametrize("posterior", ["mdl", "svd"])
def test_kl_term_is_vf_coef_s2_kl_over_n(posterior):
    agent = make_agent(posterior, log_sigma2=-4.0)
    data = batch(agent)
    s2 = torch.tensor(0.37)
    torch.manual_seed(2)
    with_kl, _ = v2.ppo_loss(agent, *data, args_for(posterior, kl_scale=1.0), s2)
    torch.manual_seed(2)  # same LRT draw
    without, _ = v2.ppo_loss(agent, *data, args_for(posterior, kl_scale=0.0), s2)
    want = 0.5 * s2 * agent.description_length() / ROWS
    torch.testing.assert_close(with_kl - without, want)


def test_kls_match_v1():
    torch.manual_seed(0)
    mean, log_sigma2 = torch.randn(6, 7), torch.randn(6, 7) - 3.0
    for name in ("mdl_kl", "svd_kl"):
        torch.testing.assert_close(getattr(v2, name)(mean, log_sigma2), getattr(v1, name)(mean, log_sigma2))


def test_decoded_critic():
    mdl = make_agent("mdl")
    x = torch.randn(ROWS, OBS)
    torch.testing.assert_close(mdl.get_value(x), mdl.critic(x))
    svd = make_agent("svd")
    with torch.no_grad():
        svd.critic.log_sigma2[0].fill_(10.0)
    decoded = svd.decoded_matrices()
    assert torch.count_nonzero(decoded[0]) == 0
    want = svd.critic.forward_with(x, decoded, v2.dense_linear)
    torch.testing.assert_close(svd.get_value(x), want)
    assert not torch.allclose(svd.get_value(x), svd.critic(x))

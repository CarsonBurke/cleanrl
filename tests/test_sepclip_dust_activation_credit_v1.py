"""Numerical estimator contracts; CUDA checks must execute through mlq."""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.dust.ppo_continuous_action_sepclip_activation_credit_v1 import (
    Agent, Args, ActivationCredit, actor_example_loss, critic_example_loss,
    dust_forward, ppo_loss, validate_args,
)


def fixture_agent():
    torch.manual_seed(1)
    env = SimpleNamespace(single_action_space=gym.spaces.Box(-1., 1., (6,), dtype=np.float32),
                          single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32))
    agent = Agent(env).cuda()
    agent.normalize_matrices()
    obs = torch.randn(512, 17, device='cuda')
    native = .1 + .8 * torch.rand(512, 6, device='cuda')
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(obs)
        old = agent.action_logprob(alpha, beta, native)
    fields = (obs, native, old, torch.randn(512, device='cuda'),
              torch.randn(512, device='cuda'), values.flatten())
    return agent, fields


def test_clean_forward_and_pointwise_objective_match_control():
    agent, fields = fixture_agent()
    args = Args(norm_adv=False, clip_vloss=False, ent_coef=.01)
    credit = ActivationCredit(agent, Args(compile=False))
    for index, network in enumerate((agent.actor, agent.critic)):
        widths = [layer.out_features for layer in credit.layers[index]] + [network.readout_gain.numel()]
        noises = tuple(fields[0].new_zeros((512, w)) for w in widths)
        output, inputs = dust_forward(network, fields[0], noises, credit.masks[0]*0, .05)
        torch.testing.assert_close(output, network(fields[0]))
        assert len(inputs) == 8
    policy = actor_example_loss(agent, agent.actor(fields[0]), *fields[1:4], args)
    value = critic_example_loss(agent.critic(fields[0]), *fields[4:], args)
    loss, _ = ppo_loss(agent, *fields, args)
    torch.testing.assert_close((policy+value).mean(), loss)


@pytest.mark.parametrize('norm_adv,clip_vloss', [(False, False), (True, True)])
def test_estimator_agrees_with_autograd_and_updates_every_parameter(norm_adv, clip_vloss):
    agent, fields = fixture_agent()
    args = Args(compile=False, dust_draws=4096, dust_pairs_per_chunk=16,
                dust_sigma=.002, norm_adv=norm_adv, clip_vloss=clip_vloss)
    loss, _ = ppo_loss(agent, *fields, args)
    loss.backward()
    expected = {name: p.grad.clone() for name, p in agent.named_parameters()}
    agent.zero_grad(set_to_none=True)
    agent.requires_grad_(False)
    ActivationCredit(agent, args).gradients(*fields)
    for network_name in ('actor', 'critic'):
        names = [n for n, _ in agent.named_parameters() if n.startswith(network_name+'.')]
        exact = torch.cat([expected[n].flatten() for n in names])
        estimated = torch.cat([p.grad.flatten() for n, p in agent.named_parameters() if n in names])
        assert torch.nn.functional.cosine_similarity(exact, estimated, dim=0) > .97
        assert (exact-estimated).norm() / exact.norm() < .25
    before = {n: p.clone() for n, p in agent.named_parameters()}
    optimizer = torch.optim.Adam(agent.parameters(), lr=.0024, eps=1e-5, fused=True)
    optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in agent.parameters())
    assert all(not torch.equal(before[n], p) for n, p in agent.named_parameters())


def test_compiled_scorer_reuses_shapes_across_sites():
    agent, fields = fixture_agent()
    agent.requires_grad_(False)
    args = Args(dust_pairs_per_chunk=2)
    eager = ActivationCredit(agent, Args(compile=False, dust_pairs_per_chunk=2))
    compiled = ActivationCredit(agent, args)
    for index in range(2):
        widths = [layer.out_features for layer in eager.layers[index]] + [(agent.actor, agent.critic)[index].readout_gain.numel()]
        noises = tuple(torch.randn(2, 512, w, device='cuda') for w in widths)
        for site in range(8):
            inputs = (fields[0], noises, eager.masks[site], *fields[1:])
            with torch.no_grad():
                actual = compiled.scorers[index](*inputs).clone()
                expected = eager.scorers[index](*inputs)
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)

    # Exercise actual permanent-buffer strides and all production site calls.
    compiled.gradients(*fields)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in agent.parameters())


@pytest.mark.parametrize('overrides', [dict(dust_draws=3), dict(dust_draws=0),
                                       dict(dust_pairs_per_chunk=3), dict(dust_sigma=0),
                                       dict(dust_sigma=float('nan'))])
def test_invalid_estimator_config(overrides):
    with pytest.raises(ValueError):
        validate_args(Args(**overrides))

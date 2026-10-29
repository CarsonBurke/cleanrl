"""Clean-hub credit contracts; run CUDA work exclusively through mlq."""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.dust.ppo_continuous_action_sepclip_clean_hub_v2 import (
    Agent, Args, ActivationCredit, actor_example_loss, actor_output_error,
    critic_example_loss, critic_output_error, clean_cache,
    first_suffix_readout, second_suffix_readout, credit_diagnostics, ppo_loss,
)


def fixture_agent(batch=512):
    torch.manual_seed(1)
    env = SimpleNamespace(single_action_space=gym.spaces.Box(-1., 1., (6,), dtype=np.float32),
                          single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32))
    agent = Agent(env).cuda()
    agent.normalize_matrices()
    obs = torch.randn(batch, 17, device='cuda')
    native = .1+.8*torch.rand(batch, 6, device='cuda')
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(obs)
        old = agent.action_logprob(alpha, beta, native)
    # Ratios include both active and inactive clean PPO clipping branches.
    target_ratios = torch.tensor([.5, .9, 1.1, 1.5], device='cuda').repeat((batch+3)//4)[:batch]
    old = old-target_ratios.log()
    advantages = torch.randn(batch, device='cuda')
    returns = values.flatten()+torch.randn(batch, device='cuda')
    return agent, (obs, native, old, advantages, returns, values.flatten())


@pytest.mark.parametrize('entropy_coefficient', [0., .03])
def test_analytic_actor_output_credit_matches_autograd_and_clean_clip_mask(entropy_coefficient):
    agent, fields = fixture_agent()
    args = Args(norm_adv=False, ent_coef=entropy_coefficient)
    output = agent.actor(fields[0]).detach().requires_grad_(True)
    pointwise = actor_example_loss(agent, output, *fields[1:4], args)
    expected = torch.autograd.grad(pointwise.sum(), output)[0]
    actual = actor_output_error(agent, output.detach(), *fields[1:4], args)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    if entropy_coefficient == 0:
        alpha, beta = (torch.nn.functional.softplus(output)+1).chunk(2, -1)
        ratio = (agent.action_logprob(alpha, beta, fields[1])-fields[2]).exp()
        inactive = ((fields[3]>0)&(ratio>1+args.clip_coef_upper)) | ((fields[3]<0)&(ratio<1-args.clip_coef))
        assert inactive.any()
        assert torch.count_nonzero(actual[inactive]) == 0


@pytest.mark.parametrize('clip_vloss', [False, True])
def test_analytic_critic_credit_matches_autograd_with_clipped_and_unclipped_targets(clip_vloss):
    args = Args(clip_vloss=clip_vloss)
    values = torch.tensor([-.8, -.1, 0., .1, .8, 1.], device='cuda').reshape(-1, 1).requires_grad_(True)
    returns = torch.tensor([-1., 1., -1., -1., 1., 0.], device='cuda')
    old = torch.zeros_like(returns)
    expected = torch.autograd.grad(critic_example_loss(values, returns, old, args).sum(), values)[0]
    actual = critic_output_error(values.detach(), returns, old, args)
    torch.testing.assert_close(actual, expected)


def test_cached_suffix_banks_match_site_replay_and_clean_network():
    agent, fields = fixture_agent(64)
    mask = torch.eye(3, device='cuda')
    for network in (agent.actor, agent.critic):
        cache = clean_cache(network, fields[0])
        torch.testing.assert_close(network.readout_gain*cache[-1], network(fields[0]))
        noise = torch.randn(4, 64, 64, device='cuda')
        for site in range(6):
            width = (network.first[0].gate.out_features if site%3<2 else 64)
            bank = (first_suffix_readout, second_suffix_readout)[site//3]
            layer = (network.first[0], network.second[0])[site//3]
            selected = (layer.gate, layer.up, layer.down)[site%3]
            hook = selected.register_forward_hook(lambda module, inputs, output: output+.05*noise[..., :width])
            try:
                expected = network(fields[0].unsqueeze(0).expand(4, -1, -1))
            finally:
                hook.remove()
            readout = bank(network, cache, noise, mask[site%3], .05)
            actual = network.readout_gain*network.head[0](readout)
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('norm_adv', [False, True])
def test_estimates_converge_and_exact_readout_has_no_sampling_noise(norm_adv):
    agent, fields = fixture_agent()
    args = Args(compile=False, dust_draws=4096, dust_pairs_per_chunk=16,
                dust_sigma=.002, norm_adv=norm_adv, clip_vloss=False)
    expected = torch.autograd.grad(ppo_loss(agent, *fields, args)[0], tuple(agent.parameters()))
    exact = {id(p): g for p, g in zip(agent.parameters(), expected)}
    agent.requires_grad_(False)
    ActivationCredit(agent, args).gradients(*fields)
    for network in (agent.actor, agent.critic):
        for parameter in (*network.head.parameters(), network.readout_gain):
            torch.testing.assert_close(parameter.grad, exact[id(parameter)], atol=2e-6, rtol=5e-5)
        reference = torch.cat([exact[id(p)].flatten() for p in network.parameters()])
        estimate = torch.cat([p.grad.flatten() for p in network.parameters()])
        assert torch.nn.functional.cosine_similarity(reference, estimate, dim=0) > .97
        assert (reference-estimate).norm()/reference.norm() < .25
    before = {id(p):p.clone() for p in agent.parameters()}
    torch.optim.Adam(agent.parameters(), lr=.0024, eps=1e-5, fused=True).step()
    assert all(not torch.equal(before[id(p)], p) for p in agent.parameters())


@pytest.mark.parametrize('draws', [16, 64])
def test_compiled_production_estimator_and_read_only_diagnostics(draws):
    agent, fields = fixture_agent()
    agent.requires_grad_(False)
    args = Args(dust_draws=draws, dust_pairs_per_chunk=8, norm_adv=False, clip_vloss=False)
    eager = ActivationCredit(agent, Args(compile=False, dust_draws=draws,
                                        dust_pairs_per_chunk=8, norm_adv=False, clip_vloss=False))
    compiled = ActivationCredit(agent, args)
    # Check every bank/site using exactly the scorer inputs and strides from training.
    for index, network in enumerate((agent.actor, agent.critic)):
        cache = clean_cache(network, fields[0])
        target = torch.randn(512, 64, device='cuda')
        noise = torch.randn(8, 512, 64, device='cuda')
        for site in range(6):
            inputs = (cache, noise, compiled.masks[site%3], target)
            with torch.no_grad():
                actual = compiled.scorers[index][site//3](*inputs).clone()
                expected = eager.scorers[index][site//3](*inputs)
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
    with torch.no_grad():
        parameters_before = [p.clone() for p in agent.parameters()]
        compiled.gradients(*fields)
        estimates = [p.grad.clone() for p in agent.parameters()]
    random_state = torch.cuda.get_rng_state().clone()
    estimator_random_state = compiled.generator.get_state().clone()
    diagnostics = credit_diagnostics(agent, fields, args)
    torch.testing.assert_close(torch.cuda.get_rng_state(), random_state, atol=0, rtol=0)
    torch.testing.assert_close(compiled.generator.get_state(), estimator_random_state, atol=0, rtol=0)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert all(not p.requires_grad for p in agent.parameters())
    for p, before, gradient in zip(agent.parameters(), parameters_before, estimates):
        torch.testing.assert_close(p, before, atol=0, rtol=0)
        torch.testing.assert_close(p.grad, gradient, atol=0, rtol=0)


def test_output_credit_at_exact_policy_boundaries_and_value_max_tie():
    agent, fields = fixture_agent(8)
    args = Args(norm_adv=False, clip_coef=.25, clip_coef_upper=.25)
    output = agent.actor(fields[0]).detach().double().requires_grad_(True)
    native = fields[1].double()
    alpha, beta = (torch.nn.functional.softplus(output)+1).chunk(2, -1)
    ratios = torch.tensor([.75, 1.25, .75, 1.25, .5, 1.5, 1., 1.], device='cuda', dtype=torch.float64)
    old = (agent.action_logprob(alpha, beta, native)-ratios.log()).detach()
    advantages = torch.tensor([1., 1., -1., -1., 1., -1., 1., -1.], device='cuda', dtype=torch.float64)
    expected = torch.autograd.grad(actor_example_loss(agent, output, native, old, advantages, args).sum(), output)[0]
    actual = actor_output_error(agent, output.detach(), native, old, advantages, args)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    values = torch.tensor([[.75], [.25], [-.75], [-.25]], device='cuda', dtype=torch.float64, requires_grad=True)
    returns = torch.tensor([.5, .5, -.5, -.5], device='cuda', dtype=torch.float64)
    old_values = torch.zeros_like(returns)
    expected = torch.autograd.grad(critic_example_loss(values, returns, old_values, args).sum(), values)[0]
    actual = critic_output_error(values.detach(), returns, old_values, args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

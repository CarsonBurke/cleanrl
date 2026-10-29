"""Full-population local-credit contracts; all GPU execution through mlq."""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch
from cleanrl.dust.ppo_continuous_action_sepclip_orthogonal_hub_v4 import (
    Agent, Args, ActivationCredit, ppo_loss, signed_hadamard_basis,
    orthogonal_probe_chunk, credit_diagnostics, validate_args,
)
from cleanrl.shared.timing import PhaseTimer


@pytest.mark.parametrize('width', [43, 64])
def test_full_basis_covariance_and_exact_linear_hub_credit(width):
    torch.manual_seed(1)
    basis = signed_hadamard_basis('cuda')
    signs = torch.empty(19, 64, device='cuda').bernoulli_(.5).mul_(2).sub_(1)
    noise = torch.cat([orthogonal_probe_chunk(basis, signs, start, 8) for start in range(0, 64, 8)])
    directions = noise[..., :width]
    covariance = directions.permute(1, 2, 0) @ directions.permute(1, 0, 2)/64
    torch.testing.assert_close(covariance, torch.eye(width, device='cuda').expand(19, -1, -1), atol=0, rtol=0)
    inputs = torch.randn(19, width, device='cuda')
    weights = torch.randn(7, width, device='cuda')
    targets = torch.randn(19, 7, device='cuda')
    positive = torch.nn.functional.linear(inputs+.05*directions, weights)
    negative = torch.nn.functional.linear(inputs-.05*directions, weights)
    score = ((positive-negative)*targets).sum(-1)/.1
    estimate = (score.unsqueeze(-1)*directions).mean(0)
    torch.testing.assert_close(estimate, targets @ weights, atol=3e-5, rtol=3e-5)


def test_production_basis_gradient_alignment_chunk_invariance_and_compiled_integration():
    torch.manual_seed(1)
    env = SimpleNamespace(single_action_space=gym.spaces.Box(-1., 1., (6,), dtype=np.float32),
                          single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32))
    agent = Agent(env).cuda()
    agent.normalize_matrices()
    args = Args(norm_adv=False, clip_vloss=False)
    obs = torch.randn(512, 17, device='cuda')
    native = .1+.8*torch.rand(512, 6, device='cuda')
    with torch.no_grad():
        alpha, beta, value = agent.get_policy_and_value(obs)
        ratios = torch.tensor([.5, .9, 1.1, 1.5], device='cuda').repeat(128)
        old = agent.action_logprob(alpha, beta, native)-ratios.log()
    fields = (obs, native, old, torch.randn(512, device='cuda'),
              value.flatten()+torch.randn(512, device='cuda'), value.flatten())
    expected = torch.autograd.grad(ppo_loss(agent, *fields, args)[0], tuple(agent.parameters()))
    exact = {id(p):g for p,g in zip(agent.parameters(), expected)}
    agent.requires_grad_(False)
    eager = ActivationCredit(agent, Args(compile=False, norm_adv=False, clip_vloss=False))
    eager.gradients(*fields)
    eager_estimates = [p.grad.clone() for p in agent.parameters()]
    for network in (agent.actor, agent.critic):
        gradient = torch.cat([p.grad.flatten() for p in network.parameters()])
        reference = torch.cat([exact[id(p)].flatten() for p in network.parameters()])
        assert torch.nn.functional.cosine_similarity(gradient, reference, dim=0) > .995
        assert (gradient-reference).norm()/reference.norm() < .05
    compiled = ActivationCredit(agent, Args(dust_pairs_per_chunk=16, norm_adv=False, clip_vloss=False))
    timer = PhaseTimer()
    with timer.span('update'):
        compiled.gradients(*fields)
        for p, previous in zip(agent.parameters(), eager_estimates):
            torch.testing.assert_close(p.grad, previous, atol=2e-6, rtol=2e-4)
        diagnostic = credit_diagnostics(agent, fields, args)
        before = [p.clone() for p in agent.parameters()]
        torch.optim.Adam(agent.parameters(), lr=.0024, eps=1e-5, fused=True).step()
        agent.normalize_matrices()
    assert timer.summary()['update']['calls'] == 1
    assert all(torch.isfinite(value) for value in diagnostic.values())
    assert all(not torch.equal(p, previous) for p,previous in zip(agent.parameters(), before))
    assert all(not p.requires_grad for p in agent.parameters())


def test_rejects_incomplete_population():
    with pytest.raises(ValueError, match='complete orthogonal'):
        validate_args(Args(dust_draws=64))

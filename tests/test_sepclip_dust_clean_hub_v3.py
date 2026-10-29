"""Exercise actual learner/diagnostic/timer integration through mlq."""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.dust.ppo_continuous_action_sepclip_clean_hub_v3 import (
    Agent, Args, ActivationCredit, credit_diagnostics, ppo_loss,
)
from cleanrl.shared.timing import PhaseTimer


@pytest.mark.parametrize('draws', [16, 64])
def test_compiled_credit_optimizer_and_diagnostics_share_update_span(draws):
    torch.manual_seed(1)
    env = SimpleNamespace(single_action_space=gym.spaces.Box(-1., 1., (6,), dtype=np.float32),
                          single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32))
    agent = Agent(env).cuda()
    agent.normalize_matrices()
    agent.requires_grad_(False)
    args = Args(dust_draws=draws, norm_adv=False, clip_vloss=False)
    obs = torch.randn(512, 17, device='cuda')
    native = .1+.8*torch.rand(512, 6, device='cuda')
    alpha, beta, values = agent.get_policy_and_value(obs)
    old = agent.action_logprob(alpha, beta, native)
    fields = (obs, native, old, torch.randn(512, device='cuda'),
              values.flatten()+torch.randn(512, device='cuda'), values.flatten())
    credit = ActivationCredit(agent, args)
    optimizer = torch.optim.Adam(agent.parameters(), lr=.0024, eps=1e-5, fused=True)
    timer = PhaseTimer()
    before = [p.clone() for p in agent.parameters()]
    with timer.span('update'):
        optimizer.zero_grad(set_to_none=True)
        credit.gradients(*fields)
        gradient_before = [p.grad.clone() for p in agent.parameters()]
        diagnostics = credit_diagnostics(agent, fields, args)
        for p, gradient in zip(agent.parameters(), gradient_before):
            torch.testing.assert_close(p.grad, gradient, atol=0, rtol=0)
        torch.nn.utils.clip_grad_norm_(tuple(agent.actor.parameters()), .5, foreach=True)
        torch.nn.utils.clip_grad_norm_(tuple(agent.critic.parameters()), .5, foreach=True)
        optimizer.step()
        agent.normalize_matrices()
    assert timer.summary()['update']['calls'] == 1
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert all(not torch.equal(p, old) for p, old in zip(agent.parameters(), before))
    # Subsequent compiled credit must see current parameters and remain
    # compatible with the frozen training state restored by diagnostics.
    with timer.span('update'):
        optimizer.zero_grad(set_to_none=True)
        credit.gradients(*fields)
        assert torch.isfinite(ppo_loss(agent, *fields, args)[0])
        assert all(not p.requires_grad for p in agent.parameters())
    assert timer.summary()['update']['calls'] == 2

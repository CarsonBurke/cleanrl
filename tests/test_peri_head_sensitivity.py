"""Actor-head step isolation and post-update continuous-policy KL contracts."""

import copy
from types import SimpleNamespace
from typing import cast

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_peri_head_sensitivity_v21 as trainer
from cleanrl.shared.rollout_graph import graph_compile
from test_ppo_normres_twohot import device

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _agent(device):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
    )
    with torch.device(device):
        return trainer.Agent(spaces)


def _objective(agent, observations):
    alpha, beta, value = agent.get_policy_and_value(observations)
    distribution = torch.distributions.Beta(alpha, beta, validate_args=False)
    return -distribution.log_prob(torch.full_like(alpha, 0.8)).mean() + (value - 0.7).square().mean()


def test_head_multiplier_changes_only_head_step_and_survives_annealing(device):
    agent = _agent(device)
    control = copy.deepcopy(agent)
    initial = {name: value.detach().clone() for name, value in agent.named_parameters()}
    optimizer = trainer.make_optimizer(agent, trainer.Args(actor_head_lr_scale=0.1))
    reference = torch.optim.Adam(control.parameters(), lr=9.6e-3, eps=1e-5, fused=True)
    observations = torch.randn(32, 17, device=device)
    # The schedule must preserve the group multiplier, not anneal only group 0.
    trainer.set_learning_rate(optimizer, 0.0024)
    reference.param_groups[0]["lr"] = 0.0024
    _objective(agent, observations).backward()
    _objective(control, observations).backward()
    optimizer.step()
    reference.step()
    for name, value in agent.named_parameters():
        baseline = dict(control.named_parameters())[name]
        expected_scale = 0.1 if name.startswith("actor.1.") else 1.0
        torch.testing.assert_close(value - initial[name], expected_scale * (baseline - initial[name]),
                                   rtol=2e-4, atol=2e-7, msg=name)


def test_unit_head_multiplier_matches_original_adam_across_steps(device):
    agent = _agent(device)
    control = copy.deepcopy(agent)
    optimizer = trainer.make_optimizer(agent, trainer.Args(actor_head_lr_scale=1.0))
    reference = torch.optim.Adam(control.parameters(), lr=9.6e-3, eps=1e-5, fused=True)
    for lr in (0.0096, 0.0048, 0.001):
        observations = torch.randn(32, 17, device=device)
        trainer.set_learning_rate(optimizer, lr)
        reference.param_groups[0]["lr"] = lr
        for model, opt in ((agent, optimizer), (control, reference)):
            opt.zero_grad(set_to_none=True)
            _objective(model, observations).backward()
            opt.step()
        for actual, expected in zip(agent.parameters(), control.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


def test_compiled_post_update_kl_detects_change_with_known_beta_integral(device):
    agent = _agent(device)
    observations = torch.randn(32, 17, device=device)
    old_alpha = torch.full((32, 6), 2.0, device=device)
    old_beta = torch.full((32, 6), 3.0, device=device)
    targets = torch.zeros(32, device=device)

    def statistics(obs, alpha, beta, values):
        return trainer.post_update_statistics(agent, obs, alpha, beta, values)

    compiled = graph_compile(statistics)
    head = cast(torch.nn.Linear, agent.actor[-1])
    assert head.bias is not None
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.expm1(torch.cat((old_alpha[0] - 1, old_beta[0] - 1))).log())
        torch.compiler.cudagraph_mark_step_begin()
        before = compiled(observations, old_alpha, old_beta, targets).clone()
        # Swap Beta(2,3) to Beta(3,2). Per-coordinate KL is exactly
        # E_old[log((1-x)/x)] = psi(3)-psi(2) = 1/2; six coordinates give 3.
        head.bias.copy_(torch.expm1(torch.cat((old_beta[0] - 1, old_alpha[0] - 1))).log())
        torch.compiler.cudagraph_mark_step_begin()
        after = compiled(observations, old_alpha, old_beta, targets).clone()
    torch.testing.assert_close(before[:2], torch.zeros_like(before[:2]), rtol=0, atol=2e-5)
    torch.testing.assert_close(after[:2], torch.full_like(after[:2], 3.0), rtol=2e-5, atol=2e-5)

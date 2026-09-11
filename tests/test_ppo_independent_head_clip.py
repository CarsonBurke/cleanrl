"""Excluded readouts must not consume or receive the trunk clipping budget."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_normres_indclip_headscope_v5 import (
    clipping_parameters,
    clip_gradients,
)

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("clip_heads", [False, True])
def test_head_updates_and_independent_trunk_budgets(clip_heads):
    agent = SimpleNamespace(
        actor=nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1)).cuda(),
        critic=nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1)).cuda(),
    )
    networks = (agent.actor, agent.critic)
    originals = []
    for network in networks:
        for index, layer in enumerate(network):
            for parameter in layer.parameters():
                parameter.grad = torch.full_like(parameter, 1000.0 if index else 2.0)
        originals.append([p.detach().clone() for p in network[1].parameters()])
    actor, critic = clipping_parameters(agent, clip_heads)
    clip_gradients(actor, critic, 0.5)
    for network in networks:
        norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in network[0].parameters()]))
        if clip_heads:
            assert norm < 0.01
        else:
            torch.testing.assert_close(norm, torch.tensor(0.5, device="cuda"))
    # Heads remain trainable in both modes; no clipping means the full raw SGD step.
    for network, before in zip(networks, originals):
        optimizer = torch.optim.SGD(network.parameters(), lr=0.001)
        optimizer.step()
        for parameter, old in zip(network[1].parameters(), before):
            displacement = old - parameter
            if clip_heads:
                assert torch.all((displacement > 0) & (displacement < 0.001))
            else:
                torch.testing.assert_close(displacement, torch.ones_like(displacement))

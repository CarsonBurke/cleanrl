"""CUDA contracts for the likelihood-ratio actor under explicit Beta KL control."""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence

from cleanrl.ppo_continuous_action_tangent_joint_graph_v9 import TangentPolicyOptimizer
from test_ppo_normres_twohot import device


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    torch._dynamo.reset()
    yield device
    torch._dynamo.reset()


def data(device, dtype):
    # Fit streams give a symmetric, unambiguous direction toward larger actions.
    # Held streams have rare positive-tail actions: the first has larger positive
    # advantage-weighted gain, but clipping removes that gain selectively.
    observations = torch.ones((8 * 8, 1), device=device, dtype=dtype)
    actions = observations.new_tensor([0.8, 0.2]).repeat(4)[:, None, None].expand(8, 8, 1).clone()
    actions[:, 6:, 0] = observations.new_tensor([1.0 - 1e-6, 1.0 - 1e-3]).repeat(4)[:, None]
    advantages = observations.new_tensor([1.0, -1.0]).repeat(4)[:, None].expand(8, 8).clone()
    return observations, actions.flatten(0, 1), advantages.flatten()


def observed_gains(old_logits, new_logits, native, advantages):
    old_alpha, old_beta = (F.softplus(old_logits.double()) + 1).chunk(2, -1)
    alpha, beta = (F.softplus(new_logits.double()) + 1).chunk(2, -1)
    old, new = Beta(old_alpha, old_beta), Beta(alpha, beta)
    changes = torch.expm1(new.log_prob(native.double()).sum(-1) - old.log_prob(native.double()).sum(-1))
    fit = advantages.view(8, 8)[:, :6].flatten()
    normalized = (advantages.double() - fit.double().mean()) / (fit.double().std() + 1e-8)
    gains = (changes * normalized).view(8, 8).mean(0)
    clipped = torch.minimum(changes * normalized, changes.clamp(-0.2, 0.2) * normalized).view(8, 8).mean(0)
    kl = kl_divergence(old, new).sum(-1).view(8, 8).mean(0)
    return gains, clipped, kl


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled_cold"])
def test_kl_feasible_improvement_is_not_vetoed_by_pessimistic_clipping(device, monkeypatch, compiled):
    dtype = torch.float32 if compiled else torch.float64
    actor = nn.Linear(1, 2, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        actor.weight.zero_()
    rejected_actor = deepcopy(actor)
    optimizer = TangentPolicyOptimizer(actor, 8, 8, kl_budget=0.01, compile=compiled)
    rejected_optimizer = TangentPolicyOptimizer(rejected_actor, 8, 8, kl_budget=0.01, compile=False)
    observations, native, advantages = data(device, dtype)
    old_logits = actor(observations).detach()
    origin = actor.weight.detach().clone()
    monkeypatch.setattr(torch.distributions.kl, "_KL_MEMOIZE", {})

    result = optimizer.step(observations, native, advantages, iteration=0)
    gain, clipped, kl = observed_gains(old_logits, actor(observations).detach(), native, advantages)
    assert result["policy_model/accepted"].item() == 1
    assert gain[:6].mean().item() > 0 and gain[6:].mean().item() > 0
    assert clipped[6:].mean().item() < 0
    assert kl[:6].mean().item() <= 0.01 + 2e-6
    assert kl[6:].mean().item() <= 0.01 + 2e-6
    torch.testing.assert_close(result["policy_model/candidate_heldout_gain"].double(), gain[6:].mean(),
                               rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(result["policy_model/candidate_heldout_clipped_gain"].double(), clipped[6:].mean(),
                               rtol=3e-4, atol=3e-6)
    assert result["policy_model/candidate_heldout_clipping_penalty"].item() > 0

    # Removing clipping does NOT remove the held-out gate or feed its labels
    # back into the proposed direction/length.
    opposing = advantages.view(8, 8).clone()
    opposing[:, 6:] *= -1
    rejected = rejected_optimizer.step(observations, native, opposing.flatten(), iteration=0)
    assert rejected["policy_model/accepted"].item() == 0
    assert rejected["policy_model/candidate_heldout_gain"].item() < 0
    torch.testing.assert_close(rejected_actor.weight, origin, rtol=0, atol=0)
    for name in ("candidate_fit_gain", "candidate_fit_kl", "linear_predicted_gain"):
        torch.testing.assert_close(result["policy_model/" + name], rejected["policy_model/" + name],
                                   rtol=5e-4, atol=3e-6)

    if compiled:
        # Warmup happened above; a new call must reuse graphs without device-to-
        # host control flow, including the added paired objective diagnostics.
        with torch.no_grad():
            actor.weight.copy_(origin)
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            with torch._dynamo.config.patch(error_on_recompile=True):
                second = optimizer.step(observations, native, opposing.flatten(), iteration=0)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        assert second["policy_model/accepted"].item() == 0
        torch.testing.assert_close(actor.weight, origin, rtol=0, atol=0)

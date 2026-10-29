"""CUDA contracts for sample-calibrated head steps; execute through mlq."""
import copy

import pytest
import torch
from torch.nn import functional as F

from cleanrl.ppo_continuous_action_sepclip_sample_fisher_head_v9 import (
    FisherHead, beta_logit_fisher, sample_head_quadratic,
)


def make_actor():
    actor = torch.nn.Module()
    actor.head = torch.nn.Sequential(torch.nn.Linear(5, 6, device='cuda'))
    actor.readout_gain = torch.nn.Parameter(torch.ones(6, device='cuda'))
    with torch.no_grad():
        actor.head[0].weight.copy_(F.normalize(actor.head[0].weight, dim=1))
    return actor


def test_sample_quadratic_matches_independent_logit_kl_hessian():
    torch.manual_seed(22)
    features = torch.randn(32, 5, device='cuda', dtype=torch.float64)
    logits = torch.randn(32, 6, device='cuda', dtype=torch.float64)
    direction = torch.randn(3, 2, 6, device='cuda', dtype=torch.float64)
    rows = torch.cat((direction[:, 0], direction[:, 1]), 0)
    displacement = F.linear(features, rows[:, :-1], rows[:, -1])
    a, b = (1+F.softplus(logits)).chunk(2, -1)
    old = torch.distributions.Beta(a, b)

    def kl(t):
        aa, bb = (1+F.softplus(logits+t*displacement)).chunk(2, -1)
        return torch.distributions.kl_divergence(old, torch.distributions.Beta(aa, bb)).sum(-1).mean()

    curvature = torch.autograd.functional.hessian(kl, torch.zeros((), device='cuda', dtype=torch.float64))
    torch.testing.assert_close(sample_head_quadratic(direction, features, logits),
                               .5*curvature, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('normalization', ['sample', 'kronecker'])
@torch.no_grad()
def test_compiled_curvature_calibration_checkpoint_and_ownership(normalization):
    torch.manual_seed(23)
    actor = make_actor()
    features = torch.randn(256, 5, device='cuda')
    # Feature-dependent concentration intentionally violates KFAC independence.
    logits = 2*features[:, :1].expand(-1, 6).clone()
    optimizer = FisherHead(actor, target_kl=1e-4, normalization=normalization)
    optimizer.refresh(features, logits)
    resumed_actor = copy.deepcopy(actor)
    resumed = FisherHead(resumed_actor, normalization=normalization)
    resumed.load_state_dict(optimizer.state_dict())
    for radius in (1e-4, 3e-5):
        optimizer.radius.fill_(radius)
        resumed.radius.fill_(radius)
        wg, bg = torch.randn_like(actor.head[0].weight), torch.randn_like(actor.head[0].bias)
        current_features = features * 1.2
        current_logits = actor.readout_gain * actor.head(current_features)
        old_a, old_b = (1+F.softplus(current_logits)).chunk(2, -1)
        for model, opt in ((actor, optimizer), (resumed_actor, resumed)):
            model.head[0].weight.grad, model.head[0].bias.grad = wg.clone(), bg.clone()
            opt.step(current_features, current_logits)
        for p, q in zip(actor.parameters(), resumed_actor.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        torch.testing.assert_close(actor.head[0].weight.grad, wg, atol=0, rtol=0)
        torch.testing.assert_close(actor.head[0].bias.grad, bg, atol=0, rtol=0)
        index = 1 if normalization == 'sample' else 0
        torch.testing.assert_close(optimizer.head_metrics[index], optimizer.radius, rtol=2e-5, atol=1e-9)
        new_logits = actor.readout_gain * actor.head(current_features)
        new_a, new_b = (1+F.softplus(new_logits)).chunk(2, -1)
        exact = torch.distributions.kl_divergence(torch.distributions.Beta(old_a, old_b),
                    torch.distributions.Beta(new_a, new_b)).sum(-1).mean()
        torch.testing.assert_close(optimizer.head_metrics[2], exact, rtol=.005, atol=5e-7)
        if normalization == 'sample':
            torch.testing.assert_close(exact, optimizer.radius, rtol=.08, atol=5e-7)
    before = copy.deepcopy(actor.state_dict())
    actor.head[0].weight.grad.zero_()
    actor.head[0].bias.grad.zero_()
    optimizer.step(current_features, actor.readout_gain*actor.head(current_features))
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.testing.assert_close(optimizer.head_metrics, torch.zeros_like(optimizer.head_metrics), rtol=0, atol=0)


@torch.no_grad()
def test_kronecker_control_preserves_v8_update_and_rejects_mode_mismatch():
    from cleanrl.ppo_continuous_action_sepclip_fisher_head_v8 import FisherHead as Original
    torch.manual_seed(24)
    actor = make_actor()
    original_actor = copy.deepcopy(actor)
    current = FisherHead(actor, normalization='kronecker')
    original = Original(original_actor)
    features = torch.randn(128, 5, device='cuda')
    logits = actor.readout_gain*actor.head(features)
    current.refresh(features, logits)
    original.refresh(features, logits)
    for model in (actor, original_actor):
        model.head[0].weight.grad = torch.ones_like(model.head[0].weight)
        model.head[0].bias.grad = torch.ones_like(model.head[0].bias)
    current.step(features, logits)
    original.step()
    for p, q in zip(actor.parameters(), original_actor.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-6, atol=2e-6)
    with pytest.raises(ValueError, match='normalization checkpoint'):
        FisherHead(copy.deepcopy(actor), normalization='sample').load_state_dict(current.state_dict())

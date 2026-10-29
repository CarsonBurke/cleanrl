"""CUDA policy geometry and continuation contracts, executed through mlq."""
import copy

import pytest
import torch
from torch.nn import functional as F

from cleanrl.ppo_continuous_action_sepclip_policy_budget_anvil_v11 import (
    MatrixOptimizer, PolicyCalibratedMatrices, ResidualMLP, policy_displacement_quadratic,
)


@pytest.fixture(autouse=True)
def independent_compile_cache():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def make_optimizer(actor,rate=.0025):
    matrices = [layer.weight for stage in (actor.first,actor.second)
                for layer in (stage[0].gate,stage[0].up,stage[0].down)]
    return PolicyCalibratedMatrices(MatrixOptimizer(matrices,lr=rate,total_steps=20,spectral_power=.5),
                                   [1,1,0,1,1,0])


def test_policy_quadratic_matches_independent_beta_kl_hessian():
    torch.manual_seed(27)
    logits = torch.randn(32,6,device='cuda',dtype=torch.float64)
    displacement = torch.randn_like(logits)
    a,b = (1+F.softplus(logits)).chunk(2,-1)
    old = torch.distributions.Beta(a,b)
    def kl(t):
        aa,bb = (1+F.softplus(logits+t*displacement)).chunk(2,-1)
        return torch.distributions.kl_divergence(old,torch.distributions.Beta(aa,bb)).sum(-1).mean()
    curvature = torch.autograd.functional.hessian(kl,torch.zeros((),device='cuda',dtype=torch.float64))
    torch.testing.assert_close(policy_displacement_quadratic(logits,logits+displacement),
                               .5*curvature,rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize('rate',[.0025,.05])
def test_policy_budget_gradient_ownership_and_checkpoint_continuation(rate):
    torch.manual_seed(28)
    actor = ResidualMLP(17,12,.1).cuda()
    # Match Agent's initial unit-matrix manifold, including rejection's old weights.
    with torch.no_grad():
        for stage in (actor.first,actor.second):
            for layer,axis in ((stage[0].gate,1),(stage[0].up,1),(stage[0].down,0)):
                layer.weight.copy_(F.normalize(layer.weight,dim=axis))
        actor.head[0].weight.copy_(F.normalize(actor.head[0].weight,dim=1))
    optimizer = make_optimizer(actor,rate)
    observations = torch.randn(128,17,device='cuda')
    logits_fn = torch.compile(actor,fullgraph=True,options={'triton.cudagraphs':False})
    optimizer.begin_rollout(4)
    actor(observations).square().mean().backward()
    optimizer.step(logits_fn,observations)
    if rate == .0025:
        assert optimizer.metrics[4] == 1
        assert optimizer.metrics[2] > 0
    restored_actor = copy.deepcopy(actor)
    restored = make_optimizer(restored_actor,rate)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_logits = torch.compile(restored_actor,fullgraph=True,options={'triton.cudagraphs':False})
    for step in range(2):
        if step==1:
            optimizer.begin_rollout(4)
            restored.begin_rollout(4)
        for model in (actor,restored_actor):
            model.zero_grad(set_to_none=True)
            model(observations).square().mean().backward()
        gradients = [p.grad.clone() for p in optimizer.parameters]
        before_logits = actor(observations).detach().clone()
        optimizer.step(logits_fn,observations)
        restored.step(restored_logits,observations)
        for p,q,g,axis in zip(optimizer.parameters,restored.parameters,gradients,optimizer.axes):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
            torch.testing.assert_close(p.norm(dim=axis),torch.ones_like(p.norm(dim=axis)),rtol=2e-6,atol=2e-6)
        actual_q = policy_displacement_quadratic(before_logits,actor(observations).detach())
        torch.testing.assert_close(actual_q,optimizer.metrics[2],rtol=2e-5,atol=1e-9)
        assert actual_q <= optimizer.metrics[0]*1.02001+1e-10
        assert 0 <= optimizer.metrics[3] <= 1
        for real,shadow in zip(optimizer.actual.stacks,optimizer.shadow.stacks):
            torch.testing.assert_close(real[2]['velocity'],shadow[2]['velocity'],rtol=1e-6,atol=1e-8)
    state = optimizer.state_dict()
    state['axes'] = tuple(reversed(state['axes']))
    with pytest.raises(ValueError,match='normalization axes'):
        restored.load_state_dict(state)

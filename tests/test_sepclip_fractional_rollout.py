"""CUDA spectral-shape/state contracts; execute through mlq."""
import copy

import pytest
import torch

from cleanrl.ppo_continuous_action_sepclip_fractional_rollout_anvil_v10 import (
    MatrixOptimizer, spectral_reweight,
)


@pytest.fixture(autouse=True)
def independent_compile_cache():
    # Independent powers/shapes otherwise exceed Dynamo's per-code graph limit.
    # One trainer process has one fixed power; do not alter its compiler limits.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize('tall', [False, True])
def test_positive_spectral_weighting_and_norm_match(tall):
    singulars = torch.tensor([4., 1., .01], device='cuda')
    matrix = torch.cat((torch.diag(singulars), torch.zeros(3,2,device='cuda')), -1)
    reference = torch.cat((torch.eye(3,device='cuda'), torch.zeros(3,2,device='cuda')), -1)
    if tall:
        matrix, reference = matrix.T, reference.T
    assert spectral_reweight(matrix, reference, 0) is reference
    result = spectral_reweight(matrix, reference, .5)
    expected = torch.cat((torch.diag(singulars.sqrt()), torch.zeros(3,2,device='cuda')), -1)
    expected = expected * reference.norm()/expected.norm()
    torch.testing.assert_close(result, expected.T if tall else expected)
    torch.testing.assert_close(result.norm(), reference.norm())
    zeros = torch.zeros_like(matrix)
    torch.testing.assert_close(spectral_reweight(zeros, zeros, .5), zeros, rtol=0, atol=0)


@pytest.mark.parametrize('power', [0., .5, 1.])
@torch.no_grad()
def test_compiled_checkpoint_continuation_and_gradient_ownership(power):
    torch.manual_seed(25)
    weights = [torch.nn.Parameter(torch.randn(*shape,device='cuda')) for shape in [(43,17),(43,64),(64,43)]]
    optimizer = MatrixOptimizer(weights, spectral_power=power, total_steps=20)
    optimizer.begin_rollout(4)
    for p in weights:
        p.grad = torch.randn_like(p)
    optimizer.step()
    cloned = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    restored = MatrixOptimizer(cloned, spectral_power=power, total_steps=20)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for step, rate in enumerate([.0025,.001,.0005]):
        if step == 1:
            optimizer.begin_rollout(4)
            restored.begin_rollout(4)
        optimizer.set_lr(rate)
        restored.set_lr(rate)
        gradients = [torch.randn_like(p) for p in weights]
        for p,q,g in zip(weights,cloned,gradients):
            p.grad,q.grad = g.clone(),g.clone()
        optimizer.step()
        restored.step()
        for p,q,g in zip(weights,cloned,gradients):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        for a,b in zip(optimizer.stacks,restored.stacks):
            for key in ('velocity','second'):
                torch.testing.assert_close(a[2][key],b[2][key],rtol=0,atol=0)


@pytest.mark.parametrize('compiled', [False, True])
@torch.no_grad()
def test_power_zero_preserves_frozen_v7_control(compiled):
    from cleanrl.ppo_continuous_action_sepclip_rollout_clock_anvil_v7 import MatrixOptimizer as Original
    torch.manual_seed(26)
    weights = [torch.nn.Parameter(torch.randn(*shape,device='cuda')) for shape in [(43,17),(43,64),(64,43)]]
    clones = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    current = MatrixOptimizer(weights,spectral_power=0,total_steps=20,compile=compiled)
    original = Original(clones,total_steps=20,compile=compiled)
    for step in range(10):
        if step%5 == 0:
            current.begin_rollout(5)
            original.begin_rollout(5)
        for p,q in zip(weights,clones):
            gradient = torch.randn_like(p)
            p.grad,q.grad = gradient.clone(),gradient.clone()
        current.step()
        original.step()
        for p,q in zip(weights,clones):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        for a,b in zip(current.stacks,original.stacks):
            for key in ('velocity','second'):
                torch.testing.assert_close(a[2][key],b[2][key],rtol=1e-6 if compiled else 0,atol=1e-9 if compiled else 0)

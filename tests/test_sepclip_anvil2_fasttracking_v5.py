"""Fasttracking contracts; run through mlq on CUDA."""
import copy

import pytest
import torch

from cleanrl.ppo_continuous_action_sepclip_anvil2_fasttracking_v5 import MatrixOptimizer


def test_fast_tracking_retains_slow_rail_engagement():
    for total in (1194, 4870):
        for progress in (0, 239, 240, 513, 514, 1144, 1194):
            actual = MatrixOptimizer.rail_schedule(progress * total / 1194, total)
            assert actual == pytest.approx((.85, .85, .4385 if progress >= 514 else 1.))


def test_compiled_continuation_across_slow_rail_engagement():
    torch.manual_seed(1)
    weights = [torch.nn.Parameter(torch.randn(*shape, device='cuda'))
               for shape in ((43, 17), (43, 17), (64, 43))]
    opt = MatrixOptimizer(weights, total_steps=8)
    for _ in range(3):
        for p in weights:
            p.grad = torch.randn_like(p)
        opt.step()
    clones = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    restored = MatrixOptimizer(clones, total_steps=8)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for step in range(3, 8):
        saved = []
        for p, q in zip(weights, clones):
            gradient = torch.randn_like(p)
            saved.append(gradient.clone())
            p.grad, q.grad = gradient.clone(), gradient.clone()
        opt.set_lr(.0025 * (8 - step) / 8)
        restored.set_lr(.0025 * (8 - step) / 8)
        opt.step()
        restored.step()
        for p, q, gradient in zip(weights, clones, saved):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            torch.testing.assert_close(p.grad, gradient, rtol=0, atol=0)
        for left, right in zip(opt.stacks, restored.stacks):
            for key in ('velocity', 'second'):
                torch.testing.assert_close(left[2][key], right[2][key], rtol=0, atol=0)
        assert opt.param_groups[0]['steps'] == restored.param_groups[0]['steps'] == step + 1

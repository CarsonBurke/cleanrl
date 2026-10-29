"""PPO rollout-clock state contracts; all GPU checks must run through mlq."""
import copy

import pytest
import torch

from cleanrl.ppo_continuous_action_sepclip_rollout_clock_anvil_v7 import MatrixOptimizer


def test_slow_rail_counts_independent_rollouts_and_ignores_later_epochs():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(43, 64, device='cuda'))
    opt = MatrixOptimizer([p], total_steps=30)
    expected = torch.zeros_like(p)
    for rollout in range(3):
        initial = torch.randn_like(p)
        opt.begin_rollout(10)
        expected = .98**10 * expected + (1-.98**10) * initial
        for epoch in range(10):
            p.grad = initial.clone() if epoch == 0 else torch.randn_like(p) * 100
            opt.step()
            torch.testing.assert_close(opt.state[p]['velocity'][1, 0], expected, rtol=3e-6, atol=5e-7)
    assert opt.param_groups[0]['rollouts'] == 3
    assert opt.param_groups[0]['steps'] == 30


@pytest.mark.parametrize('compiled', [False, True])
def test_epoch_clock_control_matches_frozen_v4(compiled):
    from cleanrl.ppo_continuous_action_sepclip_anvil2_v4 import MatrixOptimizer as Original
    torch.manual_seed(1)
    shapes = [(43, 17), (43, 64), (64, 43)]
    weights = [torch.nn.Parameter(torch.randn(*shape, device='cuda')) for shape in shapes]
    clones = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    new = MatrixOptimizer(weights, slow_clock='epoch', total_steps=10, compile=compiled)
    old = Original(clones, total_steps=10, compile=compiled)
    maximum_differences = {name: {'absolute': 0., 'relative': 0.}
                           for name in ('parameters', 'velocity', 'second')}

    def track(name, actual, expected):
        difference = (actual-expected).abs()
        maximum_differences[name]['absolute'] = max(maximum_differences[name]['absolute'], float(difference.max()))
        relative = difference/expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
        maximum_differences[name]['relative'] = max(maximum_differences[name]['relative'], float(relative.max()))

    for step in range(10):
        for p, q in zip(weights, clones):
            gradient = torch.randn_like(p)
            p.grad, q.grad = gradient.clone(), gradient.clone()
        new.step()
        old.step()
        for p, q in zip(weights, clones):
            track('parameters', p, q)
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        for a, b in zip(new.stacks, old.stacks):
            for key in ('velocity', 'second'):
                track(key, a[2][key], b[2][key])
                # The extra clock scalar changes compilation/fusion. Cross-version
                # FP32 moments permit rounding; eager and parameters remain exact.
                torch.testing.assert_close(a[2][key], b[2][key],
                                           atol=1e-9 if compiled else 0,
                                           rtol=1e-6 if compiled else 0)
    print(f'epoch_clock_parity compiled={compiled}: {maximum_differences}')


@pytest.mark.parametrize('pending', [False, True])
def test_compiled_checkpoint_preserves_rollout_phase_and_banks(pending):
    torch.manual_seed(1)
    weights = [torch.nn.Parameter(torch.randn(*shape, device='cuda'))
               for shape in [(43, 17), (43, 17), (43, 64), (64, 43)]]
    opt = MatrixOptimizer(weights, total_steps=8)
    opt.begin_rollout(4)
    for _ in range(2):
        for p in weights:
            p.grad = torch.randn_like(p)
        opt.step()
    if pending:
        opt.begin_rollout(4)
    clones = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    resumed = MatrixOptimizer(clones, total_steps=8)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert resumed.param_groups[0]['rollout_pending'] == pending
    for rate in (.0025, .001):
        opt.set_lr(rate)
        resumed.set_lr(rate)
        gradients = []
        for p, q in zip(weights, clones):
            g = torch.randn_like(p)
            gradients.append(g.clone())
            p.grad, q.grad = g.clone(), g.clone()
        opt.step()
        resumed.step()
        for p, q, g in zip(weights, clones, gradients):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
            torch.testing.assert_close(p.grad, g, atol=0, rtol=0)
        for a, b in zip(opt.stacks, resumed.stacks):
            assert b[2] is resumed.state[b[1][0]]
            for key in ('velocity', 'second'):
                torch.testing.assert_close(a[2][key], b[2][key], atol=0, rtol=0)

"""Long-axis equalizer and checkpoint continuation contracts; run via mlq."""
import copy

import pytest
import torch

from cleanrl.ppo_continuous_action_sepclip_longaxis_normuon_v3 import PolarExpressNorMuon, polar_express


@pytest.mark.parametrize("shape", [(43, 17), (43, 64), (64, 43)])
def test_axis_and_norm_match_reference(shape):
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(*shape, device="cuda"))
    opt = PolarExpressNorMuon([p], compile=False)
    momentum = torch.zeros(1, *shape, device="cuda")
    dim = -1 if shape[0] >= shape[1] else -2
    second = torch.zeros((1, shape[0], 1) if dim == -1 else (1, 1, shape[1]), device="cuda")
    for _ in range(4):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        before = p.detach().clone()
        # Match the pre-bf16 arithmetic: algebraically equivalent FP32 sums
        # can cross a bf16 rounding boundary. The oracle below independently
        # checks the equalizer axis, state, and restored Frobenius norm.
        momentum.lerp_(grad[None], 1 - .95)
        polar = polar_express(grad[None].lerp(momentum, .95)).float()
        second = .95 * second + .05 * polar.square().mean(dim, keepdim=True)
        adapted = polar * second.clamp_min(1e-10).rsqrt()
        expected = -.005 * max(1., shape[0] / shape[1]) ** .5 * adapted * (polar.norm() / adapted.norm())
        opt.step()
        torch.testing.assert_close(p - before, expected[0], rtol=1e-4, atol=2e-7)
        torch.testing.assert_close(opt.state[p]["second"], second, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)


def test_compiled_checkpoint_continuation_preserves_all_banks():
    torch.manual_seed(1)
    weights = [torch.nn.Parameter(torch.randn(*shape, device="cuda"))
               for shape in [(43, 17), (43, 17), (43, 64), (64, 43)]]
    original = PolarExpressNorMuon(weights)
    for _ in range(4):
        for p in weights:
            p.grad = torch.randn_like(p)
        original.step()
    original.set_lr(.0017)
    clones = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    restored = PolarExpressNorMuon(clones)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    torch.testing.assert_close(restored.param_groups[0]["lr"], original.param_groups[0]["lr"], atol=0, rtol=0)
    for group, ps, state, _ in restored.stacks:
        assert group is restored.param_groups[0]
        assert state is restored.state[ps[0]]
        assert group["lr"].is_cuda
    for lr in (.0017, .0008):
        original.set_lr(lr)
        restored.set_lr(lr)
        for p, clone in zip(weights, clones):
            grad = torch.randn_like(p)
            p.grad, clone.grad = grad.clone(), grad.clone()
        original.step()
        restored.step()
        for p, clone in zip(weights, clones):
            torch.testing.assert_close(p, clone, atol=0, rtol=0)
        for old, new in zip(original.stacks, restored.stacks):
            for key in ("momentum_buffer", "second"):
                torch.testing.assert_close(old[2][key], new[2][key], atol=0, rtol=0)


def test_incompatible_row_history_is_rejected():
    p = torch.nn.Parameter(torch.randn(43, 64, device="cuda"))
    opt = PolarExpressNorMuon([p])
    saved = copy.deepcopy(opt.state_dict())
    saved["state"][0]["second"] = torch.zeros(1, 43, 1, device="cuda")
    with pytest.raises(ValueError, match="variance lanes"):
        opt.load_state_dict(saved)

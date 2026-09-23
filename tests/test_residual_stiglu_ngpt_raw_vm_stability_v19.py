"""Reference value clipping and gain smoothing must preserve intended gradients."""
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_raw_vm_stability_v19 import (
    ValueGainEMA, categorical_value_terms,
)
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.two_hot import DreamerTwoHotSupport


def test_compiled_value_gate_freezes_improvement_beyond_clip_not_worsening_or_overshoot():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    histogram = DreamerTwoHotSupport(5, 2, device="cuda", spacing="linear")
    # Means: improvement beyond boundary, worsening, overshoot, inside boundary.
    means = torch.tensor([0.6, -0.6, 1.9, 0.1], device="cuda")
    probabilities = torch.full((4, 5), 1e-6, device="cuda")
    probabilities[:, 0] = (2-means)/4
    probabilities[:, -1] = (2+means)/4
    probabilities /= probabilities.sum(-1, keepdim=True)
    logits = probabilities.log().requires_grad_()
    targets = torch.ones(4, device="cuda")
    old_values = torch.zeros_like(targets)
    labels = histogram.project(targets).requires_grad_()
    loss_fn = torch.compile(categorical_value_terms, fullgraph=True,
                            options={"triton.cudagraphs": False})
    loss, _, gain, _, clipfrac = loss_fn(logits, labels, histogram, targets, old_values, True)
    loss.backward()
    expected = gain * (logits.detach().softmax(-1)-labels.detach())/4
    expected[0].zero_()
    torch.testing.assert_close(logits.grad, expected)
    torch.testing.assert_close(clipfrac, torch.tensor(0.25, device="cuda"))
    assert labels.grad is None


def test_compiled_gain_ema_initialization_history_and_detached_backward():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    histogram = DreamerTwoHotSupport(5, 2, device="cuda", spacing="linear")
    ema = ValueGainEMA(0.5).cuda()
    labels = torch.tensor([[0.1,0.2,0.4,0.2,0.1]],device="cuda",requires_grad=True)
    targets = torch.zeros(1,device="cuda")
    logits = torch.tensor([[0.0,0.1,0.2,0.3,0.4]],device="cuda",requires_grad=True)
    loss_fn = torch.compile(categorical_value_terms, fullgraph=True,
                            options={"triton.cudagraphs": False})
    first_loss, _, first_gain, raw_first, _ = loss_fn(logits,labels,histogram,targets,targets,False,0.2,ema)
    torch.testing.assert_close(first_gain,raw_first)
    histogram.support.mul_(2)
    second_loss, _, second_gain, raw_second, _ = loss_fn(logits,labels,histogram,targets,targets,False,0.2,ema)
    torch.testing.assert_close(raw_second,4*raw_first)
    torch.testing.assert_close(second_gain,0.5*(first_gain+raw_second))
    # Advancing EMA must not mutate the multiplier saved by the earlier backward.
    first_loss.backward()
    torch.testing.assert_close(logits.grad,first_gain*(logits.detach().softmax(-1)-labels.detach()))
    logits.grad = None
    second_loss.backward()
    torch.testing.assert_close(logits.grad,second_gain*(logits.detach().softmax(-1)-labels.detach()))
    assert labels.grad is None
    assert not first_gain.requires_grad and not second_gain.requires_grad
    restored = ValueGainEMA(0.5).cuda()
    restored.load_state_dict(ema.state_dict())
    next_gain = restored.update(raw_first)
    torch.testing.assert_close(next_gain,0.5*(second_gain+raw_first))

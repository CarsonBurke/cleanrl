"""Gradient-clipping ablation must preserve raw updates when disabled."""
import pytest
import torch
from torch import nn

from cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_normres_clip_toggle_v6 import clip_gradients

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("enabled", [True, False])
def test_clipping_toggle_preserves_unclipped_update(enabled):
    actor = nn.Parameter(torch.zeros(2, device="cuda"))
    critic = nn.Parameter(torch.zeros(2, device="cuda"))
    actor.grad = torch.tensor([3.0, 4.0], device="cuda")
    critic.grad = torch.tensor([0.0, 12.0], device="cuda")
    norm_a, norm_c = clip_gradients((actor,), (critic,), 0.5, enabled=enabled)
    torch.testing.assert_close(norm_a, torch.tensor(5.0, device="cuda"))
    torch.testing.assert_close(norm_c, torch.tensor(12.0, device="cuda"))
    torch.optim.SGD((actor, critic), lr=1.0).step()
    if enabled:
        torch.testing.assert_close(actor, torch.tensor([-0.3, -0.4], device="cuda"))
        torch.testing.assert_close(critic, torch.tensor([0.0, -0.5], device="cuda"))
    else:
        torch.testing.assert_close(actor, torch.tensor([-3.0, -4.0], device="cuda"), rtol=0, atol=0)
        torch.testing.assert_close(critic, torch.tensor([0.0, -12.0], device="cuda"), rtol=0, atol=0)

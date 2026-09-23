"""Clipping scope must not silently couple the actor and critic gradients."""
import pytest
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_raw_variance_clip_v18 import apply_gradient_clipping


@pytest.mark.parametrize("mode", ["global", "actor", "none"])
def test_clipping_scope_preserves_unselected_gradients_and_adam_update(mode):
    actor = torch.nn.Parameter(torch.zeros(2, device="cuda"))
    critic = torch.nn.Parameter(torch.zeros(2, device="cuda"))
    original_actor = torch.tensor([3.0, 4.0], device="cuda")
    original_critic = torch.tensor([0.0, 1200.0], device="cuda")
    actor.grad = original_actor.clone()
    critic.grad = original_critic.clone()
    parameters = (actor, critic)
    optimizer = torch.optim.Adam(parameters, lr=0.0024, eps=1e-5, fused=True)
    expected_actor = original_actor.clone()
    expected_critic = original_critic.clone()
    if mode == "global":
        scale = 0.5 / (torch.sqrt(original_actor.square().sum() + original_critic.square().sum()) + 1e-6)
        expected_actor *= scale
        expected_critic *= scale
    elif mode == "actor":
        expected_actor *= 0.5 / (original_actor.norm() + 1e-6)
    apply_gradient_clipping(parameters, (actor,), 0.5, mode)
    torch.testing.assert_close(actor.grad, expected_actor, rtol=0, atol=0)
    torch.testing.assert_close(critic.grad, expected_critic, rtol=0, atol=0)
    optimizer.step()
    # First-step Adam makes the effect of clipping and epsilon observable.
    torch.testing.assert_close(actor, -0.0024 * expected_actor / (expected_actor.abs() + 1e-5))
    torch.testing.assert_close(critic, -0.0024 * expected_critic / (expected_critic.abs() + 1e-5))

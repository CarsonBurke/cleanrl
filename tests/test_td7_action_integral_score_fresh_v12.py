"""Numerical likelihood-surrogate tests; no networks or GPU work."""
import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl/td7/latentq/td7_latentq_action_integral_score_fresh_v12.py"
SPEC = importlib.util.spec_from_file_location("action_integral_score_v12", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ppo_clips_only_improving_ratios_for_each_advantage_sign():
    ratios = torch.tensor([1.5, 0.5, 1.5, 0.5], dtype=torch.float64)
    logprob = ratios.log().requires_grad_()
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    loss = MODULE.ppo_surrogate_loss(logprob, torch.zeros_like(logprob), advantage, 0.2)
    torch.testing.assert_close(loss, torch.tensor((-1.2 - 0.5 + 1.5 + 0.8) / 4, dtype=torch.float64))
    loss.backward()
    # Excess improvement plateaus. Deteriorating ratios retain their gradient.
    torch.testing.assert_close(logprob.grad, torch.tensor([0.0, -0.125, 0.375, 0.0], dtype=torch.float64))


def test_gaussian_likelihood_uses_raw_draw_when_executed_action_saturates():
    mu = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[0.5]], dtype=torch.float64).log()
    raw_action = torch.tensor([[1.4]], dtype=torch.float64)
    executed_action = raw_action.clamp(-1, 1)
    raw_logprob = MODULE.gaussian_log_prob(mu, log_std, raw_action)
    clipped_logprob = MODULE.gaussian_log_prob(mu, log_std, executed_action)
    assert raw_logprob.item() < clipped_logprob.item()
    torch.testing.assert_close(executed_action, torch.ones_like(executed_action))
    raw_gradient = torch.autograd.grad(raw_logprob.sum(), mu)[0]
    torch.testing.assert_close(raw_gradient, torch.tensor([[3.6]], dtype=torch.float64))


def test_unit_behavior_ratio_recovers_normalized_score_gradient():
    logprob = torch.tensor([-0.5, -1.5], dtype=torch.float64, requires_grad=True)
    old_logprob = logprob.detach().clone()
    advantage = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    loss = MODULE.ppo_surrogate_loss(logprob, old_logprob, advantage, 0.2)
    torch.testing.assert_close(loss, torch.zeros((), dtype=torch.float64))
    loss.backward()
    torch.testing.assert_close(logprob.grad, -advantage / 2)

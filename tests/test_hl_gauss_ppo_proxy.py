"""Guard the calibrated proxy against plausible false-winner mechanisms."""

import pytest
import torch

from scripts.hlgauss.ppo_proxy_v3 import (
    CASES,
    EnsembleCritic,
    lambda_advantages,
    make_mdp,
    ppo_gradients,
    update_critics,
)


def test_mdp_values_satisfy_bellman_equation_before_and_after_shift():
    for phase in (0, 1):
        _, transition, reward, truth, advantage = make_mdp(CASES[3], phase)
        torch.testing.assert_close(transition.sum(-1), torch.ones(2, 96))
        action_values = reward + CASES[3].gamma * (transition @ truth)
        torch.testing.assert_close(action_values.mean(0), truth)
        torch.testing.assert_close(action_values - truth, advantage, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(make_mdp(CASES[3], 0)[3], make_mdp(CASES[3], 1)[3])


def test_all_lambda_targets_keep_constant_critic_bias_in_raw_advantages():
    gamma, trace, error, horizon = 0.99, 0.95, -0.14, 64
    truth = torch.full((1, horizon + 1, 2), 3.8, dtype=torch.float64)
    rewards = torch.full((horizon, 2), (1 - gamma) * 3.8, dtype=torch.float64)
    advantages = lambda_advantages(rewards, truth + error, gamma, trace)
    remaining = torch.arange(horizon, 0, -1, dtype=torch.float64)
    expected = -(1 - gamma) * error * (1 - (gamma * trace) ** remaining) / (1 - gamma * trace)
    torch.testing.assert_close(advantages[0, :, 0], expected)
    torch.testing.assert_close(advantages[0, :, 1], expected)
    torch.testing.assert_close(
        lambda_advantages(rewards, truth, gamma, trace), torch.zeros_like(advantages), atol=1e-12, rtol=0
    )


def test_clipped_ppo_gradient_matches_autograd_for_each_critic():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(40, 6, generator=generator, dtype=torch.float64)
    actions = torch.randint(2, (40,), generator=generator).double()
    advantages = torch.randn(2, 40, generator=generator, dtype=torch.float64)
    actual = ppo_gradients(x, actions, advantages)
    direction = torch.tensor([0.8, -0.5, 0.3, 0.2, -0.2, 0.1], dtype=torch.float64)
    for model in range(2):
        for probe, amplitude in enumerate((0.0, 0.6, -0.6)):
            theta = (amplitude * direction).requires_grad_()
            probability = (x @ theta).sigmoid()
            ratio = 2 * torch.where(actions.bool(), probability, 1 - probability)
            loss = torch.maximum(-advantages[model] * ratio, -advantages[model] * ratio.clamp(0.8, 1.2)).mean()
            (reference,) = torch.autograd.grad(loss, theta)
            torch.testing.assert_close(actual[model, probe], reference)


def test_clipped_ppo_detects_bias_that_exact_action_differences_cancel():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(12, 6, generator=generator).repeat_interleave(2, 0)
    actions = torch.tensor([0.0, 1.0] * 12)
    advantage = ((2 * actions - 1) * 0.05).unsqueeze(0)
    shifted = advantage + 0.1
    reference = ppo_gradients(x, actions, advantage)
    actual = ppo_gradients(x, actions, shifted)
    # Balanced actions cancel constant baselines on-policy, but clipping does not.
    torch.testing.assert_close(actual[:, 0], reference[:, 0], atol=1e-8, rtol=1e-6)
    assert (actual[:, 1:] - reference[:, 1:]).square().sum() > 1e-5
    torch.testing.assert_close(
        ppo_gradients(x, actions, shifted, normalize=True),
        ppo_gradients(x, actions, advantage, normalize=True),
        atol=1e-7,
        rtol=1e-5,
    )


@pytest.mark.parametrize("architecture", ["tanh", "sphere"])
def test_critic_bank_optimizer_and_clipping_are_independent(architecture):
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(12, 6, generator=generator)
    labels = torch.randn(2, 12, 3, generator=generator).softmax(-1)
    unchanged = EnsembleCritic(2, 3, architecture, True)
    poisoned = EnsembleCritic(2, 3, architecture, True)
    original_optimizer = torch.optim.Adam(unchanged.parameters(), lr=0.01, eps=1e-5)
    poisoned_optimizer = torch.optim.Adam(poisoned.parameters(), lr=0.01, eps=1e-5)
    different_labels = labels.clone()
    different_labels[1] = torch.tensor([1.0, 0.0, 0.0])
    for _ in range(3):
        update_critics(unchanged, original_optimizer, x, labels, None, None)
        update_critics(poisoned, poisoned_optimizer, x, different_labels, None, None)
    torch.testing.assert_close(unchanged(x)[0], poisoned(x)[0], atol=0, rtol=0)
    assert not torch.allclose(unchanged(x)[1], poisoned(x)[1])

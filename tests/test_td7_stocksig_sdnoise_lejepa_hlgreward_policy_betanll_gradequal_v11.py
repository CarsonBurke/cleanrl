import torch
import torch.nn.functional as F

from cleanrl.td7.lesale.td7_lesale_v1 import (
    IsometricOutcomeTokens,
    equalized_outcome_trunk_scales,
)


def _tokens():
    torch.manual_seed(21)
    return IsometricOutcomeTokens(
        32,
        6,
        64,
        51,
        -40.0,
        40.0,
        0.75,
        True,
        1e-5,
        30.0,
        True,
    )


def test_equalized_scales_split_one_representation_gradient_budget():
    representation_norm = torch.tensor(12.0)
    raw_reward_norm = torch.tensor(3.0)
    raw_policy_norm = torch.tensor(8.0)
    scales = equalized_outcome_trunk_scales(
        representation_norm, raw_reward_norm, raw_policy_norm
    )
    assert torch.equal(scales, torch.tensor([2.0, 0.75]))
    assert torch.equal(scales[0] * raw_reward_norm, representation_norm / 2.0)
    assert torch.equal(scales[1] * raw_policy_norm, representation_norm / 2.0)


def test_trunk_scales_change_only_upstream_gradients_not_forward_or_head_gradients():
    tokens = _tokens()
    transition = torch.randn(16, 32, requires_grad=True)
    action = torch.rand(16, 6) * 2.0 - 1.0
    reward = torch.randn(16, 1)
    policy_target = torch.rand(16, 12) * 2.0 - 1.0

    baseline_outputs = tokens(None, action, reward, policy_target, transition)
    baseline_policy_loss = tokens.policy_moment_beta_nll(
        baseline_outputs[2], baseline_outputs[3]
    )[0].mean()
    baseline_transition_grad, baseline_head_grad = torch.autograd.grad(
        baseline_policy_loss,
        (transition, tokens.policy_predictor[-1].weight),
    )

    tokens.set_trunk_scales(torch.tensor([0.25, 3.0]))
    scaled_outputs = tokens(None, action, reward, policy_target, transition)
    scaled_policy_loss = tokens.policy_moment_beta_nll(
        scaled_outputs[2], scaled_outputs[3]
    )[0].mean()
    scaled_transition_grad, scaled_head_grad = torch.autograd.grad(
        scaled_policy_loss,
        (transition, tokens.policy_predictor[-1].weight),
    )

    for baseline, scaled in zip(baseline_outputs, scaled_outputs):
        assert torch.equal(baseline, scaled)
    assert torch.allclose(scaled_transition_grad, 3.0 * baseline_transition_grad)
    assert torch.equal(scaled_head_grad, baseline_head_grad)


def test_reward_and_policy_scales_are_independent():
    tokens = _tokens()
    with torch.no_grad():
        tokens.reward_readout[-1].weight.normal_(std=1e-2)
    transition = torch.randn(8, 32, requires_grad=True)
    action = torch.rand(8, 6) * 2.0 - 1.0
    reward = torch.randn(8, 1)
    policy_target = torch.rand(8, 12) * 2.0 - 1.0

    tokens.set_trunk_scales(torch.tensor([0.2, 4.0]))
    reward_logits, reward_target, policy_raw, policy_semantic = tokens(
        None, action, reward, policy_target, transition
    )[:4]
    reward_loss = -(reward_target * F.log_softmax(reward_logits, -1)).sum(-1).mean()
    policy_loss = tokens.policy_moment_beta_nll(
        policy_raw, policy_semantic
    )[0].mean()
    reward_grad = torch.autograd.grad(reward_loss, transition, retain_graph=True)[0]
    policy_grad = torch.autograd.grad(policy_loss, transition)[0]

    tokens.set_trunk_scales(torch.ones(2))
    reward_logits, reward_target, policy_raw, policy_semantic = tokens(
        None, action, reward, policy_target, transition
    )[:4]
    baseline_reward = -(reward_target * F.log_softmax(reward_logits, -1)).sum(-1).mean()
    baseline_policy = tokens.policy_moment_beta_nll(
        policy_raw, policy_semantic
    )[0].mean()
    baseline_reward_grad = torch.autograd.grad(
        baseline_reward, transition, retain_graph=True
    )[0]
    baseline_policy_grad = torch.autograd.grad(baseline_policy, transition)[0]
    assert torch.allclose(reward_grad, 0.2 * baseline_reward_grad)
    assert torch.allclose(policy_grad, 4.0 * baseline_policy_grad)

import torch
import torch.nn.functional as F

from cleanrl.td7_lesale_v1 import IsometricOutcomeTokens


def _tokens():
    torch.manual_seed(6)
    return IsometricOutcomeTokens(
        latent_dim=32,
        action_dim=6,
        token_dim=64,
        reward_num_bins=51,
        reward_raw_min=-40.0,
        reward_raw_max=40.0,
        reward_sigma_ratio=0.75,
    )


def test_target_encoders_are_exact_information_preserving_isometries():
    tokens = _tokens()
    reward_weight = tokens.reward_tokenizer.weight.detach()
    policy_weight = tokens.policy_tokenizer.weight.detach()
    assert torch.allclose(reward_weight.T @ reward_weight, torch.eye(51), atol=1e-6)
    assert torch.allclose(policy_weight.T @ policy_weight, torch.eye(12), atol=1e-6)

    reward_distribution = tokens.project_reward(torch.randn(16, 1))
    reward_token = tokens.reward_tokenizer(reward_distribution)
    assert torch.allclose(
        tokens.decode_reward_token(reward_token), reward_distribution, atol=1e-6
    )
    policy_moments = torch.rand(16, 12) * 2.0 - 1.0
    policy_token = tokens.policy_tokenizer(policy_moments)
    assert torch.allclose(
        tokens.decode_policy_token(policy_token), policy_moments, atol=1e-6
    )


def test_attached_latent_mse_shapes_both_predictor_and_target_encoder():
    tokens = _tokens()
    batch = 16
    transition = torch.randn(batch, 32, requires_grad=True)
    action = torch.rand(batch, 6) * 2.0 - 1.0
    reward = torch.randn(batch, 1)
    policy_moments = torch.rand(batch, 12) * 2.0 - 1.0
    outputs = tokens(transition, action, reward, policy_moments, transition)
    reward_prediction, reward_target, policy_prediction, policy_target = outputs[:4]
    assert reward_prediction.shape == reward_target.shape == (batch, 64)
    assert policy_prediction.shape == policy_target.shape == (batch, 64)
    assert not hasattr(tokens, "reward_decoder")
    assert not hasattr(tokens, "policy_decoder")

    loss = F.mse_loss(reward_prediction, reward_target) + F.mse_loss(
        policy_prediction, policy_target
    )
    loss.backward()
    assert float(transition.grad.abs().sum()) > 0.0
    assert float(tokens.reward_predictor[-1].weight.grad.abs().sum()) > 0.0
    assert float(tokens.policy_predictor[-1].weight.grad.abs().sum()) > 0.0
    assert (
        float(
            tokens.reward_tokenizer.parametrizations.weight.original.grad.abs().sum()
        )
        > 0.0
    )
    assert (
        float(
            tokens.policy_tokenizer.parametrizations.weight.original.grad.abs().sum()
        )
        > 0.0
    )


def test_target_encoder_optimizer_step_preserves_isometry():
    tokens = _tokens()
    optimizer = torch.optim.AdamW(tokens.parameters(), lr=3e-4, weight_decay=1e-3)
    transition = torch.randn(32, 32)
    action = torch.rand(32, 6) * 2.0 - 1.0
    reward = torch.randn(32, 1)
    policy_moments = torch.rand(32, 12) * 2.0 - 1.0
    outputs = tokens(transition, action, reward, policy_moments, transition)
    loss = F.mse_loss(outputs[0], outputs[1]) + F.mse_loss(outputs[2], outputs[3])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    reward_weight = tokens.reward_tokenizer.weight.detach()
    policy_weight = tokens.policy_tokenizer.weight.detach()
    assert torch.allclose(reward_weight.T @ reward_weight, torch.eye(51), atol=1e-5)
    assert torch.allclose(policy_weight.T @ policy_weight, torch.eye(12), atol=1e-5)


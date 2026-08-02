import torch
import torch.nn.functional as F

from cleanrl.td7_lesale_v1 import FullSIGReg, LatentOutcomeTokens


def _tokens():
    torch.manual_seed(5)
    return LatentOutcomeTokens(
        latent_dim=32,
        action_dim=6,
        token_dim=64,
        reward_num_bins=51,
        reward_raw_min=-40.0,
        reward_raw_max=40.0,
        reward_sigma_ratio=0.75,
        reward_prior_floor=1e-20,
    )


def test_outcome_predictions_and_attached_targets_are_64d_latents():
    tokens = _tokens()
    batch = 8
    transition = torch.randn(batch, 32, requires_grad=True)
    action = torch.randn(batch, 6, requires_grad=True)
    reward = torch.randn(batch, 1)
    policy_moments = torch.rand(batch, 12) * 2.0 - 1.0
    outputs = tokens(transition, action, reward, policy_moments, transition)
    reward_prediction, reward_target, policy_prediction, policy_target = outputs[:4]
    assert reward_prediction.shape == reward_target.shape == (batch, 64)
    assert policy_prediction.shape == policy_target.shape == (batch, 64)

    latent_loss = F.mse_loss(reward_prediction, reward_target)
    latent_loss.backward(retain_graph=True)
    assert float(transition.grad.abs().sum()) > 0.0
    assert float(tokens.reward_tokenizer[-1].weight.grad.abs().sum()) > 0.0


def test_semantic_decoders_train_only_target_tokens_not_world_transition():
    tokens = _tokens()
    # The calibrated zero prior initially blocks target-encoder reward gradients; emulate the
    # decoder after its first update so both semantic anchors exercise their complete paths.
    torch.nn.init.xavier_uniform_(tokens.reward_decoder[-1].weight)
    batch = 8
    transition = torch.randn(batch, 32, requires_grad=True)
    action = torch.randn(batch, 6, requires_grad=True)
    reward = torch.randn(batch, 1)
    policy_moments = torch.rand(batch, 12) * 2.0 - 1.0
    outputs = tokens(transition, action, reward, policy_moments, transition)
    reward_logits, reward_distribution = outputs[4], outputs[5]
    policy_reconstruction, policy_target = outputs[6], outputs[7]
    semantic_loss = -(
        reward_distribution * F.log_softmax(reward_logits, dim=-1)
    ).sum(-1).mean() + F.mse_loss(policy_reconstruction, policy_target)
    transition_grad = torch.autograd.grad(
        semantic_loss, transition, retain_graph=True, allow_unused=True
    )[0]
    assert transition_grad is None or torch.count_nonzero(transition_grad) == 0
    target_grads = torch.autograd.grad(
        semantic_loss,
        [tokens.reward_tokenizer[-1].weight, tokens.policy_tokenizer[-1].weight],
    )
    assert all(float(gradient.abs().sum()) > 0.0 for gradient in target_grads)


def test_weighted_fullsig_matches_explicit_valid_subset():
    torch.manual_seed(7)
    sigreg = FullSIGReg(64, 128, 17, torch.device("cpu"), seed=8)
    sigreg.resample_directions()
    tokens = torch.randn(1, 16, 64)
    valid = torch.cat([torch.ones(12, 1), torch.zeros(4, 1)])
    weighted = sigreg(tokens, valid)
    explicit = sigreg(tokens[:, :12])
    assert torch.allclose(weighted, explicit, atol=1e-6)


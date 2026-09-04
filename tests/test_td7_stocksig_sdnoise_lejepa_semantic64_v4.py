import math

import numpy as np
import torch
import torch.nn.functional as F

from cleanrl.td7.lesale.td7_lesale_v1 import (
    Args,
    EncoderLossCore,
    SemanticOutcomeTokens,
    StockEncoder,
    UniformLAPBuffer,
)


def _tokens():
    torch.manual_seed(1)
    return SemanticOutcomeTokens(
        latent_dim=32,
        action_dim=6,
        token_dim=64,
        reward_num_bins=51,
        reward_raw_min=-40.0,
        reward_raw_max=40.0,
        reward_sigma_ratio=0.75,
        reward_prior_floor=1e-20,
    )


def test_reward_support_uses_symlog_raw_range_and_normalized_hlgauss_labels():
    tokens = _tokens()
    assert torch.allclose(
        tokens.reward_support[[0, -1]],
        torch.tensor([-math.log(41.0), math.log(41.0)]),
    )
    targets = torch.tensor([[-40.0], [-5.0], [0.0], [5.0], [40.0]])
    probs = tokens.project_reward(targets)
    assert probs.shape == (5, 51)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(probs[0], probs[-1].flip(0), atol=1e-6)
    assert torch.allclose(probs[1], probs[-2].flip(0), atol=1e-6)
    assert int(probs[2].argmax()) == 25


def test_zero_reward_prior_decodes_to_zero():
    tokens = _tokens()
    logits = tokens.reward_predictor[-1].bias.unsqueeze(0)
    assert torch.allclose(tokens.reward_to_scalar(logits), torch.zeros(1), atol=1e-6)


def test_semantic_tokens_have_64d_bottlenecks_and_receive_direct_gradients():
    tokens = _tokens()
    # v215-style zero-prior initialization intentionally blocks reward-feature gradients on the
    # first update; emulate the decoder after it has taken a learning step.
    torch.nn.init.xavier_uniform_(tokens.reward_predictor[-1].weight)
    batch = 8
    transition = torch.randn(batch, 32, requires_grad=True)
    action = torch.randn(batch, 6, requires_grad=True)
    reward = torch.randn(batch, 1)
    policy_target = torch.rand(batch, 12) * 2.0 - 1.0
    (
        reward_logits,
        reward_probs,
        policy_prediction,
        returned_policy_target,
        reward_token,
        policy_token,
    ) = tokens(transition, action, reward, policy_target, transition)
    assert tokens.reward_query.shape == (64,)
    assert tokens.policy_query.shape == (64,)
    assert reward_logits.shape == (batch, 51)
    assert reward_probs.shape == (batch, 51)
    assert policy_prediction.shape == (batch, 12)
    assert reward_token.shape == (batch, 64)
    assert policy_token.shape == (batch, 64)
    assert returned_policy_target is policy_target

    reward_loss = -(reward_probs * F.log_softmax(reward_logits, dim=-1)).sum(-1).mean()
    policy_loss = F.mse_loss(policy_prediction, policy_target)
    (reward_loss + policy_loss).backward()
    assert float(transition.grad.abs().sum()) > 0.0
    assert float(action.grad.abs().sum()) > 0.0


def test_semantic_policy_loss_masks_terminal_successors():
    torch.manual_seed(2)
    encoder = StockEncoder(state_dim=5, action_dim=2, zs_dim=8, hdim=16)
    tokens = SemanticOutcomeTokens(8, 2, 64, 51, -40.0, 40.0, 0.75, 1e-20)
    args = Args(
        zs_dim=8,
        hidden_dim=16,
        prediction_from_lap=False,
        outcome_from_transition=True,
        semantic_outcome_tokens=True,
        sd_noise=True,
    )
    core = EncoderLossCore(encoder, None, None, args, outcome_tokens=tokens)
    state = torch.randn(2, 5)
    action = torch.randn(2, 2)
    next_state = torch.randn(2, 5)
    reward = torch.randn(2, 1)
    policy_target = torch.randn(2, 4)
    policy_valid = torch.tensor([[1.0], [0.0]])

    with torch.no_grad():
        transition = encoder.zsa(encoder.zs(state), action)
        policy_prediction = tokens(
            encoder.zs(state), action, reward, policy_target, transition
        )[2]
        expected = F.mse_loss(policy_prediction[0], policy_target[0])
    outputs = core(
        state,
        action,
        next_state,
        state,
        action,
        next_state,
        enc_reward=reward,
        policy_mean_target=policy_target,
        policy_valid_target=policy_valid,
    )
    assert torch.allclose(outputs[15], expected)
    assert torch.allclose(outputs[20], policy_target[0].std(unbiased=False))


def test_policy_validity_is_distinct_from_td_bootstrap_on_terminal_timelimit():
    buffer = UniformLAPBuffer(
        state_dim=3,
        action_dim=2,
        device=torch.device("cpu"),
        max_size=4,
        batch_size=1,
        max_action=1.0,
    )
    buffer.add(
        np.zeros(3, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        reward=1.0,
        done=0.0,
        episode_boundary=True,
        successor_policy_valid=False,
    )
    assert float(buffer.not_done[0, 0]) == 1.0
    assert float(buffer.successor_policy_valid[0, 0]) == 0.0
    *_, sampled_policy_valid = buffer.sample_uniform_with_reward()
    assert float(sampled_policy_valid[0, 0]) == 0.0

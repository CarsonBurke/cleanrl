import torch
import torch.nn.functional as F

from cleanrl.td7.lesale.td7_lesale_v1 import (
    Args,
    EncoderLossCore,
    IsometricOutcomeTokens,
    StockEncoder,
)


def _tokens(latent_dim=32, action_dim=6):
    torch.manual_seed(7)
    return IsometricOutcomeTokens(
        latent_dim=latent_dim,
        action_dim=action_dim,
        token_dim=64,
        reward_num_bins=51,
        reward_raw_min=-40.0,
        reward_raw_max=40.0,
        reward_sigma_ratio=0.75,
        policy_beta_nll=True,
        policy_beta_nll_eps=1e-5,
    )


def test_policy_moment_beta_nll_matches_dreamer4_parameterization():
    tokens = _tokens()
    raw_parameters = torch.randn(8, 24)
    policy_moments = torch.linspace(-1.0, 1.0, 96).reshape(8, 12)
    element_nll, targets, alpha, beta = tokens.policy_moment_beta_nll(
        raw_parameters, policy_moments
    )
    expected_alpha = F.softplus(raw_parameters[:, :12]) + 1.0
    expected_beta = F.softplus(raw_parameters[:, 12:]) + 1.0
    expected_targets = (0.5 * (policy_moments + 1.0)).clamp(
        1e-5, 1.0 - 1e-5
    )
    expected_nll = -torch.distributions.Beta(
        expected_alpha, expected_beta
    ).log_prob(expected_targets)
    assert torch.allclose(alpha, expected_alpha)
    assert torch.allclose(beta, expected_beta)
    assert torch.allclose(targets, expected_targets)
    assert torch.allclose(element_nll, expected_nll, atol=1e-6)
    assert not hasattr(tokens, "policy_tokenizer")


def test_policy_beta_nll_keeps_existing_policy_moment_target_semantics():
    tokens = _tokens()
    transition = torch.randn(16, 32, requires_grad=True)
    action = torch.rand(16, 6) * 2.0 - 1.0
    reward = torch.randn(16, 1)
    policy_moments = torch.rand(16, 12) * 2.0 - 1.0
    outputs = tokens(None, action, reward, policy_moments, transition)
    reward_prediction, reward_target, policy_parameters, policy_target = outputs[:4]
    policy_nll = tokens.policy_moment_beta_nll(
        policy_parameters, policy_target
    )[0].mean()
    loss = F.mse_loss(reward_prediction, reward_target) + policy_nll
    loss.backward()
    assert policy_parameters.shape == (16, 24)
    assert policy_target is policy_moments
    assert float(transition.grad.abs().sum()) > 0.0
    assert float(tokens.policy_predictor[-1].weight.grad.abs().sum()) > 0.0


def test_encoder_core_masks_policy_moment_beta_nll_and_compiles_fullgraph():
    torch.manual_seed(8)
    encoder = StockEncoder(state_dim=5, action_dim=2, zs_dim=8, hdim=16)
    tokens = _tokens(latent_dim=8, action_dim=2)
    args = Args(
        zs_dim=8,
        hidden_dim=16,
        prediction_from_lap=False,
        outcome_from_transition=True,
        isometric_outcome_tokens=True,
        policy_beta_nll=True,
        sd_noise=True,
    )
    core = EncoderLossCore(encoder, None, None, args, outcome_tokens=tokens)
    state = torch.randn(2, 5)
    action = torch.rand(2, 2) * 2.0 - 1.0
    next_state = torch.randn(2, 5)
    reward = torch.randn(2, 1)
    policy_moments = torch.rand(2, 4) * 2.0 - 1.0
    policy_valid = torch.tensor([[1.0], [0.0]])
    inputs = (
        state,
        action,
        next_state,
        state,
        action,
        next_state,
        reward,
        policy_moments,
        policy_valid,
    )
    with torch.no_grad():
        transition = encoder.zsa(encoder.zs(state), action)
        policy_parameters = tokens(
            None, action, reward, policy_moments, transition
        )[2]
        expected = tokens.policy_moment_beta_nll(
            policy_parameters[:1], policy_moments[:1]
        )[0].mean()
    outputs = core(*inputs)
    assert torch.allclose(outputs[15], expected)
    expected_total = (
        outputs[1]
        + args.subsig_coef * outputs[2]
        + args.outcome_token_coef * (outputs[13] + outputs[15])
    )
    assert torch.allclose(outputs[0], expected_total)
    compiled = torch.compile(core, backend="eager", fullgraph=True)
    compiled_outputs = compiled(*inputs)
    assert torch.allclose(compiled_outputs[0], outputs[0])


def test_reward_branch_initialization_matches_v6_despite_policy_head_replacement():
    torch.manual_seed(6602)
    v6 = IsometricOutcomeTokens(256, 6, 64, 51, -40.0, 40.0, 0.75)
    torch.manual_seed(6602)
    v7 = IsometricOutcomeTokens(
        256, 6, 64, 51, -40.0, 40.0, 0.75, policy_beta_nll=True
    )
    assert torch.equal(v6.reward_tokenizer.weight, v7.reward_tokenizer.weight)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            v6.reward_predictor.state_dict().values(),
            v7.reward_predictor.state_dict().values(),
        )
    )
    assert torch.equal(v6.reward_action_proj.weight, v7.reward_action_proj.weight)

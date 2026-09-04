import numpy as np
import torch

from cleanrl.td7.lesale.td7_lesale_v1 import (
    Args,
    EncoderLossCore,
    IsometricOutcomeTokens,
    StockEncoder,
)


def _tokens(latent_dim=32, action_dim=6):
    torch.manual_seed(9)
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
        policy_beta_max_precision=30.0,
    )


def test_bounded_beta_preserves_v7_initial_distribution_and_caps_confidence():
    tokens = _tokens()
    initial_raw = torch.zeros(3, 24)
    alpha, beta = tokens.policy_moment_beta(initial_raw)
    initial_shape = 1.0 + np.log(2.0)
    assert torch.allclose(alpha, torch.full_like(alpha, initial_shape))
    assert torch.allclose(beta, torch.full_like(beta, initial_shape))

    extreme_raw = torch.cat(
        (torch.full((3, 12), 1e3), torch.full((3, 12), 1e3)), dim=-1
    )
    alpha, beta = tokens.policy_moment_beta(extreme_raw)
    assert torch.all(alpha >= 1.0)
    assert torch.all(beta >= 1.0)
    assert torch.all(alpha + beta <= 32.0 + 1e-6)


def test_bounded_beta_separates_location_and_precision():
    tokens = _tokens()
    raw = torch.randn(8, 24)
    alpha, beta = tokens.policy_moment_beta(raw)
    expected_mean = torch.sigmoid(raw[:, :12])
    expected_initial_precision = 2.0 * np.log(2.0)
    expected_offset = np.log(
        (expected_initial_precision / 30.0)
        / (1.0 - expected_initial_precision / 30.0)
    )
    expected_precision = 30.0 * torch.sigmoid(raw[:, 12:] + expected_offset)
    assert torch.allclose(alpha, 1.0 + expected_mean * expected_precision)
    assert torch.allclose(beta, 1.0 + (1.0 - expected_mean) * expected_precision)


def test_encoder_uses_independent_policy_beta_coefficient_and_compiles():
    torch.manual_seed(10)
    encoder = StockEncoder(state_dim=5, action_dim=2, zs_dim=8, hdim=16)
    tokens = _tokens(latent_dim=8, action_dim=2)
    args = Args(
        zs_dim=8,
        hidden_dim=16,
        prediction_from_lap=False,
        outcome_from_transition=True,
        isometric_outcome_tokens=True,
        policy_beta_nll=True,
        policy_beta_nll_coef=0.05,
        policy_beta_max_precision=30.0,
        outcome_token_coef=0.5,
        sd_noise=True,
    )
    core = EncoderLossCore(encoder, None, None, args, outcome_tokens=tokens)
    inputs = (
        torch.randn(3, 5),
        torch.rand(3, 2) * 2.0 - 1.0,
        torch.randn(3, 5),
        torch.randn(3, 5),
        torch.rand(3, 2) * 2.0 - 1.0,
        torch.randn(3, 5),
        torch.randn(3, 1),
        torch.rand(3, 4) * 2.0 - 1.0,
        torch.ones(3, 1),
    )
    outputs = core(*inputs)
    expected_total = (
        outputs[1]
        + args.subsig_coef * outputs[2]
        + args.outcome_token_coef * outputs[13]
        + args.policy_beta_nll_coef * outputs[15]
    )
    assert torch.allclose(outputs[0], expected_total)

    compiled = torch.compile(core, backend="eager", fullgraph=True)
    compiled_outputs = compiled(*inputs)
    assert torch.allclose(compiled_outputs[0], outputs[0])

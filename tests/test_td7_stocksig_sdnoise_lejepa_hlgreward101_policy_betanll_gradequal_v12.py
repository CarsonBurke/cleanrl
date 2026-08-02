import math

import torch

from cleanrl.shared.hl_gauss import HLGaussSupport
from cleanrl.td7_lesale_v1 import IsometricOutcomeTokens, symlog


def _tokens():
    torch.manual_seed(31)
    return IsometricOutcomeTokens(
        latent_dim=32,
        action_dim=6,
        token_dim=64,
        reward_num_bins=101,
        reward_raw_min=-40.0,
        reward_raw_max=40.0,
        reward_sigma_ratio=0.5,
        policy_beta_nll=True,
        policy_beta_nll_eps=1e-5,
        policy_beta_max_precision=30.0,
        reward_hlgauss_ce=True,
    )


def test_direct_hlgauss_can_have_more_classes_than_the_outcome_feature_width():
    tokens = _tokens()
    transition = torch.randn(8, 32)
    action = torch.rand(8, 6) * 2.0 - 1.0
    reward = torch.linspace(-40.0, 40.0, 8).unsqueeze(-1)
    policy_target = torch.rand(8, 12) * 2.0 - 1.0

    reward_logits, reward_target = tokens(
        None, action, reward, policy_target, transition
    )[:2]

    assert reward_logits.shape == reward_target.shape == (8, 101)
    assert tokens.reward_readout[-1].in_features == 64
    assert tokens.reward_readout[-1].out_features == 101
    assert torch.allclose(reward_target.sum(dim=-1), torch.ones(8))


def test_reward_resolution_changes_only_direct_support_and_readout_initialization():
    torch.manual_seed(31)
    baseline = IsometricOutcomeTokens(
        32, 6, 64, 51, -40.0, 40.0, 0.75, True, 1e-5, 30.0, True
    )
    treatment = _tokens()

    assert all(
        torch.equal(baseline_value, treatment_value)
        for baseline_value, treatment_value in zip(
            baseline.policy_predictor.state_dict().values(),
            treatment.policy_predictor.state_dict().values(),
        )
    )
    assert all(
        torch.equal(baseline_value, treatment_value)
        for baseline_value, treatment_value in zip(
            baseline.reward_predictor.state_dict().values(),
            treatment.reward_predictor.state_dict().values(),
        )
    )
    assert all(
        torch.equal(baseline_value, treatment_value)
        for baseline_value, treatment_value in zip(
            baseline.reward_action_proj.state_dict().values(),
            treatment.reward_action_proj.state_dict().values(),
        )
    )


def test_reward_support_symlogs_raw_plus_minus_40_and_uses_half_bin_sigma():
    tokens = _tokens()
    raw_bounds = torch.tensor([-40.0, 40.0])

    assert torch.allclose(tokens.reward_support[[0, -1]], symlog(raw_bounds))
    assert math.isclose(tokens.reward_bin_width, 2.0 * math.log1p(40.0) / 100.0)
    assert math.isclose(tokens.reward_sigma, 0.5 * tokens.reward_bin_width)
    assert tokens.reward_scalar_support[0].item() == -40.0
    assert tokens.reward_scalar_support[-1].item() == 40.0


def test_symlog_projection_matches_shared_hlgauss_center_convention():
    tokens = _tokens()
    shared = HLGaussSupport(
        num_bins=101,
        v_min=-math.log1p(40.0),
        v_max=math.log1p(40.0),
        sigma_ratio=0.5,
        device=torch.device("cpu"),
        use_symlog=True,
    )
    rewards = torch.linspace(-45.0, 45.0, 257)

    assert torch.equal(tokens.reward_support, shared.support)
    assert torch.allclose(
        tokens.project_reward(rewards), shared.project(rewards), atol=3e-6, rtol=1e-6
    )

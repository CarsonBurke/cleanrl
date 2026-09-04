import math

import torch

from cleanrl.shared.hl_gauss import HLGaussSupport
from cleanrl.td7.lesale.td7_lesale_v1 import IsometricOutcomeTokens


def test_raw_hlgauss_uses_uniform_raw_reward_centers_and_raw_sigma():
    tokens = IsometricOutcomeTokens(
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
        reward_hlgauss_symlog=False,
    )

    assert torch.equal(tokens.reward_support, torch.linspace(-40.0, 40.0, 101))
    assert torch.equal(tokens.reward_scalar_support, tokens.reward_support)
    assert math.isclose(tokens.reward_bin_width, 0.8)
    assert math.isclose(tokens.reward_sigma, 0.4)

    rewards = torch.tensor([[-40.0], [-1.0], [0.0], [1.0], [40.0]])
    targets = tokens.project_reward(rewards)
    decoded = targets @ tokens.reward_scalar_support
    assert torch.allclose(targets.sum(dim=-1), torch.ones(5))
    assert torch.allclose(decoded, rewards.squeeze(-1), atol=0.25)

    shared = HLGaussSupport(
        num_bins=101,
        v_min=-40.0,
        v_max=40.0,
        sigma_ratio=0.5,
        device=torch.device("cpu"),
    )
    dense_rewards = torch.linspace(-45.0, 45.0, 257)
    assert torch.equal(tokens.reward_support, shared.support)
    assert torch.allclose(tokens.project_reward(dense_rewards), shared.project(dense_rewards))

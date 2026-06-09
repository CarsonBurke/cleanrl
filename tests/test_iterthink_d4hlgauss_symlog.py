from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_v1 import Args as D4Args
from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_v1 import (
    Agent as D4SymlogAgent,
    Args as D4SymlogArgs,
    value_support_bounds,
)
from cleanrl.shared.hl_gauss import HLGaussSupport, symlog


def test_d4_symlog_only_preserves_d4_value_defaults_except_symlog():
    base = D4Args()
    args = D4SymlogArgs()

    assert args.num_bins == base.num_bins == 511
    assert args.v_min == base.v_min == -10.0
    assert args.v_max == base.v_max == 10.0
    assert args.value_sigma_to_bin_ratio == base.value_sigma_to_bin_ratio == 2.0
    assert args.critic_init_tau == base.critic_init_tau == 0.5
    assert base.value_symlog is False
    assert args.value_symlog is True


def test_d4_symlog_support_uses_transformed_bounds():
    args = SimpleNamespace(v_min=-10.0, v_max=10.0, value_symlog=True)
    support_min, support_max = value_support_bounds(args)
    expected = symlog(torch.tensor([args.v_min, args.v_max])).tolist()

    assert support_min == expected[0]
    assert support_max == expected[1]

    support = HLGaussSupport(
        num_bins=511,
        v_min=support_min,
        v_max=support_max,
        sigma_ratio=D4SymlogArgs.value_sigma_to_bin_ratio,
        device=torch.device("cpu"),
        use_symlog=True,
        support_is_edges=True,
    )
    probs = support.project(torch.tensor([-10.0, 0.0, 10.0]))

    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert support.support.min().item() > -3.0
    assert support.support.max().item() < 3.0


def test_d4_symlog_critic_bias_is_in_transformed_coordinate():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    args = D4SymlogArgs(hidden=16, k_blocks=1, n_experts=1, num_bins=31)
    agent = D4SymlogAgent(DummyVecEnv(), args)
    support_min, support_max = value_support_bounds(args)
    edge_width = (support_max - support_min) / args.num_bins
    centers = torch.linspace(
        support_min + 0.5 * edge_width,
        support_max - 0.5 * edge_width,
        args.num_bins,
    )
    expected_bias = -0.5 * (centers / args.critic_init_tau) ** 2

    assert torch.allclose(agent.critic_head.bias, expected_bias, atol=1e-6)

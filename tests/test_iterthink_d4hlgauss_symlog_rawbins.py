import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawbins_v1 import (
    Agent,
    Args,
    RawSpacedSymlogHLGaussSupport,
)


def test_rawbins_support_is_linear_in_raw_space_not_symlog_space():
    args = Args()
    support = RawSpacedSymlogHLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )

    raw_widths = support.edges[1:] - support.edges[:-1]
    coord_widths = support.coord_edges[1:] - support.coord_edges[:-1]

    assert torch.allclose(raw_widths, torch.full_like(raw_widths, support.bin_width), atol=1e-6)
    assert not torch.allclose(coord_widths, torch.full_like(coord_widths, coord_widths.mean()))
    assert support.support.min().item() > args.v_min
    assert support.support.max().item() < args.v_max


def test_rawbins_symlog_projection_normalizes_and_decodes_raw_expectation():
    args = Args(num_bins=101)
    support = RawSpacedSymlogHLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )

    targets = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    probs = support.project(targets)

    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(targets), atol=1e-5)
    assert probs[0].argmax().item() == 0
    assert probs[-1].argmax().item() == args.num_bins - 1

    logits = torch.full((1, support.num_bins), -100.0)
    raw_center_idx = support.support.abs().argmin()
    logits[0, raw_center_idx] = 100.0

    assert torch.allclose(support.to_scalar(logits), support.support[raw_center_idx].reshape(1))


def test_rawbins_critic_bias_matches_exact_symlog_gaussian_bin_mass():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    args = Args(hidden=16, k_blocks=1, n_experts=1, num_bins=31)
    agent = Agent(DummyVecEnv(), args)
    support = RawSpacedSymlogHLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )

    expected_bias = support.gaussian_prior_logits(args.critic_init_tau)

    assert torch.allclose(agent.critic_head.bias, expected_bias, atol=1e-6)

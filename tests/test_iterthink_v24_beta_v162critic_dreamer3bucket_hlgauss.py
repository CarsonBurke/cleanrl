import torch

from cleanrl.iterthink.critic_variants.ppo_continuous_action_iterthink_v24_beta_v162critic_dreamer3bucket_hlgauss_mtp_v1 import (
    Args,
)
from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport, symlog, symexp


def test_dreamer3_bucket_support_matches_symexp_twohot_centers():
    args = Args()
    assert args.value_sigma_to_bin_ratio == 0.75
    assert torch.allclose(torch.tensor(args.v_min), symlog(torch.tensor(-20000.0)))
    assert torch.allclose(torch.tensor(args.v_max), symlog(torch.tensor(20000.0)))
    support = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )

    half = torch.linspace(args.v_min, 0.0, (args.num_bins - 1) // 2 + 1)
    expected_coord = torch.cat([half, -half[:-1].flip(0)])

    assert torch.allclose(support.coord_support, expected_coord)
    assert torch.allclose(support.support, symexp(expected_coord))
    assert torch.allclose(
        support.support[[0, -1]],
        torch.tensor([-20000.0, 20000.0]),
        rtol=1e-5,
        atol=1e-2,
    )
    assert support.support[args.num_bins // 2].item() == 0.0
    assert torch.allclose(support.support, -support.support.flip(0))


def test_dreamer3_bucket_hlgauss_projection_normalizes_and_zero_decodes_to_zero():
    args = Args(num_bins=511)
    support = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )

    targets = torch.tensor([-20000.0, -10.0, 0.0, 10.0, 20000.0])
    probs = support.project(targets)
    zero_logits = torch.zeros(3, args.critic_mtp_horizon, args.num_bins)

    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(targets), atol=1e-5)
    assert probs[2].argmax().item() == args.num_bins // 2
    assert support.project(torch.tensor(0.0)).shape == (args.num_bins,)
    assert torch.allclose(support.to_scalar(zero_logits), torch.zeros(3, args.critic_mtp_horizon), atol=1e-5)
    assert support.project(torch.zeros(2, 4, args.critic_mtp_horizon)).shape == (
        2,
        4,
        args.critic_mtp_horizon,
        args.num_bins,
    )


def test_dreamer3_bucket_moment_matching_removes_symlog_jensen_bias():
    coord_limit = symlog(torch.tensor(20_000.0)).item()
    support = Dreamer3BucketHLGaussSupport(
        51,
        -coord_limit,
        coord_limit,
        0.75,
        torch.device("cpu"),
    )
    targets = torch.tensor(
        [-20_000.0, -1_000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1_000.0, 20_000.0]
    )

    base_probs = support.project(targets)
    matched_probs = support.project_moment_matched(targets)
    base_values = support.probs_to_scalar(base_probs)
    matched_values = support.probs_to_scalar(matched_probs)

    assert (base_values[1:-1] - targets[1:-1]).abs().max() > 1.0
    assert not torch.any(
        (base_probs[1:-1] == 0.0) & (matched_probs[1:-1] > 0.0)
    )
    assert torch.allclose(
        matched_probs.sum(dim=-1),
        torch.ones_like(targets),
        atol=1e-6,
    )
    assert torch.allclose(matched_values, targets, rtol=1e-5, atol=1e-3)


def test_dreamer3_bucket_projection_clamps_to_raw_support_centers():
    args = Args()
    support = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )

    high = torch.tensor(1e9)
    low = torch.tensor(-1e9)
    assert torch.allclose(support.project(high), support.project(torch.tensor(20000.0)))
    assert torch.allclose(support.project(low), support.project(torch.tensor(-20000.0)))
    assert torch.allclose(support.cdf_fraction(high), support.cdf_fraction(torch.tensor(20000.0)))
    assert torch.allclose(support.cdf_fraction(low), support.cdf_fraction(torch.tensor(-20000.0)))

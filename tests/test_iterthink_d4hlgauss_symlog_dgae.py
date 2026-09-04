from types import SimpleNamespace

import torch

from cleanrl.iterthink.v24_d4hlgauss.dg.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dgae_v1 import (
    Args,
    distributional_raw_mean,
    value_support_bounds,
)
from cleanrl.shared.hl_gauss import HLGaussSupport, symexp, symlog


def exact_matched_quantile_direction(current_probs, next_probs, support, gamma):
    current_cdf = current_probs.cumsum(dim=0)
    next_cdf = next_probs.cumsum(dim=0)
    boundaries = torch.cat(
        [
            torch.zeros(1, dtype=support.dtype),
            current_cdf,
            next_cdf,
            torch.ones(1, dtype=support.dtype),
        ]
    ).unique(sorted=True)

    total = torch.zeros((), dtype=support.dtype)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if hi <= lo:
            continue
        q = (lo + hi) / 2.0
        current_idx = torch.searchsorted(current_cdf, q, right=False).clamp(max=support.numel() - 1)
        next_idx = torch.searchsorted(next_cdf, q, right=False).clamp(max=support.numel() - 1)
        total = total + (hi - lo) * (gamma * support[next_idx] - support[current_idx])
    return total


def test_dgae_defaults_preserve_symlog_hlgauss_setup():
    args = Args()

    assert args.dgae_policy_adv is True
    assert args.value_symlog is True
    assert args.v_min == -10.0
    assert args.v_max == 10.0
    assert args.value_sigma_to_bin_ratio == 2.0
    assert args.adv_transform == "rankgauss"
    assert args.norm_adv is True


def test_directional_quantile_metric_collapses_to_raw_mean_difference():
    support = torch.tensor([-3.0, -0.5, 0.25, 2.0, 5.0])
    current_probs = torch.tensor([0.05, 0.45, 0.10, 0.30, 0.10])
    next_probs = torch.tensor([0.30, 0.05, 0.20, 0.05, 0.40])
    gamma = 0.99

    quantile_metric = exact_matched_quantile_direction(current_probs, next_probs, support, gamma)
    mean_metric = gamma * distributional_raw_mean(next_probs, support) - distributional_raw_mean(current_probs, support)

    assert torch.allclose(quantile_metric, mean_metric, atol=1e-6)


def test_dgae_uses_raw_distribution_mean_not_symlog_scalar_decode():
    args = SimpleNamespace(v_min=-10.0, v_max=10.0, value_symlog=True)
    support_min, support_max = value_support_bounds(args)
    support = HLGaussSupport(
        num_bins=7,
        v_min=support_min,
        v_max=support_max,
        sigma_ratio=2.0,
        device=torch.device("cpu"),
        use_symlog=True,
        support_is_edges=True,
    )
    probs = torch.tensor([[0.02, 0.04, 0.08, 0.10, 0.16, 0.25, 0.35]])
    logits = probs.log()
    raw_support = symexp(support.support)

    raw_mean = distributional_raw_mean(probs, raw_support)
    scalar_decode = support.to_scalar(logits)

    assert torch.allclose(raw_mean, (probs * raw_support).sum(dim=-1), atol=1e-6)
    assert not torch.allclose(raw_mean, scalar_decode, atol=1e-3)
    assert support.v_min == symlog(torch.tensor(-10.0)).item()
    assert support.v_max == symlog(torch.tensor(10.0)).item()

"""Unit tests for the AWR ADVANTAGE-WEIGHT variant (advweight_awr_v1).

AWR (Peng et al. 2019) gives the closed-form return-maximizing weight w=exp(A/beta)
as the trust-region optimum. We apply it as a non-negative, mean-1 per-sample
multiplier on the PPO surrogate (commutes with the clip max). These tests pin the
defaults, the softmax/mean-1 form, the ESS auto-tune (monotone in beta), the CRR
clamp + renorm, the fixed-beta fallback, and that w>=0 commutes with the clip max.
"""
import numpy as np
import torch

from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_advweight_awr_v1 import (
    Args,
    awr_weights,
)


def test_awr_defaults():
    args = Args()
    # The base advantage estimator is STANDARD GAE, identity transform, standardized;
    # the AWR weight is then the ONLY shaping (clean A/B vs the rankgauss baseline).
    assert args.adv_transform == "v10"
    assert args.norm_adv is True
    assert args.awr is True
    assert args.awr_beta > 0.0
    assert 0.0 < args.awr_target_ess < 1.0
    assert args.awr_wmax > 1.0
    assert args.total_timesteps == 8000000 and args.env_id == "HalfCheetah-v4"
    assert args.value_symlog is True


def test_weight_is_mean_one_and_nonnegative():
    torch.manual_seed(0)
    adv = torch.randn(2048) * 3.0
    w, beta, ess = awr_weights(adv, beta=1.0, target_ess=0.0, wmax=0.0)
    assert (w >= 0).all()
    assert abs(w.mean().item() - 1.0) < 1e-4          # softmax*n => mean exactly 1
    assert 0.0 < ess <= 1.0


def test_ess_autotune_hits_target():
    torch.manual_seed(1)
    adv = torch.randn(1024) * 2.0
    for target in (0.3, 0.5, 0.8):
        w, beta, ess = awr_weights(adv, beta=1.0, target_ess=target, wmax=0.0)
        assert abs(ess - target) < 0.03               # bisection converges to target ESS
        assert beta > 0.0


def test_ess_is_monotone_increasing_in_beta():
    # Larger beta => flatter softmax => higher effective sample size.
    torch.manual_seed(2)
    adv = torch.randn(512) * 2.0
    esss = [awr_weights(adv, beta=b, target_ess=0.0, wmax=0.0)[2] for b in (0.25, 0.5, 1.0, 4.0, 16.0)]
    assert all(esss[i] < esss[i + 1] for i in range(len(esss) - 1))


def test_higher_advantage_gets_higher_weight():
    # AWR up-weights large (positive) advantages: w is monotone in A.
    adv = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    w, _, _ = awr_weights(adv, beta=1.0, target_ess=0.0, wmax=0.0)
    assert torch.all(w[1:] > w[:-1])                  # strictly increasing with A


def test_crr_clamp_caps_amplification():
    # A heavy outlier would get a huge exp weight; the CRR clamp caps it. softmax
    # gives mean 1 first, so clamping only REDUCES => max<=wmax and mean<=1 (never
    # re-amplified above the cap).
    adv = torch.tensor([0.0] * 31 + [50.0])           # one extreme advantage
    w, _, _ = awr_weights(adv, beta=1.0, target_ess=0.0, wmax=5.0)
    assert w.max().item() <= 5.0 + 1e-5
    assert w.mean().item() <= 1.0 + 1e-5
    # without the cap the same outlier would blow up far past wmax
    w_unc, _, _ = awr_weights(adv, beta=1.0, target_ess=0.0, wmax=0.0)
    assert w_unc.max().item() > 5.0


def test_clamp_is_noop_for_standardized_advantages():
    # For realistic ~unit-std advantages the wmax=20 cap rarely binds => mean stays 1.
    torch.manual_seed(7)
    adv = torch.randn(2048)
    w, _, _ = awr_weights(adv, beta=1.0, target_ess=0.0, wmax=20.0)
    assert abs(w.mean().item() - 1.0) < 1e-3


def test_fixed_beta_used_when_target_ess_zero():
    # target_ess<=0 => the passed beta is used verbatim (no autotune).
    adv = torch.randn(256)
    w, beta, _ = awr_weights(adv, beta=2.0, target_ess=0.0, wmax=0.0)
    assert beta == 2.0


def test_weight_commutes_with_clip_max():
    # max(w*l1, w*l2) == w*max(l1,l2) for w>=0, so weighting the clipped surrogate is
    # identical to scaling the advantage inside the clip.
    torch.manual_seed(3)
    l1, l2 = torch.randn(64), torch.randn(64)
    w, _, _ = awr_weights(torch.randn(64), beta=1.0, target_ess=0.0, wmax=0.0)
    lhs = torch.maximum(w * l1, w * l2)
    rhs = w * torch.maximum(l1, l2)
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_large_beta_approaches_uniform():
    # beta -> inf => softmax flat => w -> all ones (no reweighting; AWR off).
    adv = torch.randn(128) * 2.0
    w, _, ess = awr_weights(adv, beta=1e6, target_ess=0.0, wmax=0.0)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-3)
    assert ess > 0.99

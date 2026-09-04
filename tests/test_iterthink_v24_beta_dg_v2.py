import torch
from torch.distributions import Beta

from cleanrl.iterthink.dg.ppo_continuous_action_iterthink_v24_beta_dg_v2 import (
    Args,
    delight_gate,
    dg_raw_surprisal,
)


def test_default_surprisal_is_mode_ref():
    args = Args()
    assert args.dg_surprisal == "mode_ref"
    assert args.dg_eta == 1.0 and args.dg_clip == 10.0
    assert args.dg_surprisal_norm is False and args.dg_whiten_chi is False and args.dg_renorm is False


def test_mode_ref_surprisal_is_logpeak_minus_logprob():
    args = Args()  # mode_ref
    logprob = torch.randn(64)
    logp_peak = logprob + torch.rand(64)  # peak density is the max => logp_peak >= logprob
    entropy = torch.rand(64)
    ell = dg_raw_surprisal(logprob, entropy, args, logp_peak)
    assert torch.allclose(ell, logp_peak - logprob)


def test_mode_ref_is_nonnegative_for_real_beta():
    # The whole point: for bounded Beta the mode-referenced surprisal is >= 0 everywhere,
    # unlike raw -logp which is negative most of the time once the policy sharpens.
    torch.manual_seed(0)
    for a, b in [(1.7, 1.7), (8.0, 3.0), (20.0, 20.0)]:
        A = torch.full((4096, 6), a)
        B = torch.full((4096, 6), b)
        d = Beta(A, B)
        acts = d.sample()
        logprob = d.log_prob(acts).sum(-1)
        mode = ((A - 1.0) / (A + B - 2.0)).clamp(1e-4, 1 - 1e-4)
        logp_peak = d.log_prob(mode).sum(-1)
        args = Args()
        ell = dg_raw_surprisal(logprob, None, args, logp_peak)
        assert torch.all(ell >= -1e-4), (a, b, float(ell.min()))
        # raw -logp would be negative for most samples here -> contrast
        raw = -logprob
        if a >= 8.0:
            assert (raw < 0).float().mean() > 0.5  # raw_clip would invert the gate


def test_mode_ref_keeps_sign_agreement_one():
    # ell >= 0 => sign(chi) == sign(U): the gate tracks breakthrough/blunder, not the inverse.
    torch.manual_seed(1)
    A = torch.full((2048, 6), 12.0)
    B = torch.full((2048, 6), 5.0)
    d = Beta(A, B)
    acts = d.sample()
    logprob = d.log_prob(acts).sum(-1)
    mode = ((A - 1.0) / (A + B - 2.0)).clamp(1e-4, 1 - 1e-4)
    logp_peak = d.log_prob(mode).sum(-1)
    args = Args()
    ell = dg_raw_surprisal(logprob, None, args, logp_peak)
    adv = torch.randn(2048)
    gate, surprisal, chi = delight_gate(adv, ell, args)
    sign_agree = ((chi > 0) == (adv > 0)).float().mean()
    # near 1.0 (ties only where ell hits exactly 0 at the mode); contrast v1 raw_clip ~0.3
    assert sign_agree > 0.97
    # positive-advantage tail actions get gate > 0.5, negative get < 0.5
    assert torch.all(gate[(adv > 0) & (ell > 1e-3)] > 0.5)
    assert torch.all(gate[(adv < 0) & (ell > 1e-3)] < 0.5)


def test_zero_advantage_gives_neutral_gate():
    args = Args()
    ell = torch.rand(32) + 0.5  # strictly positive surprisal
    gate, _, chi = delight_gate(torch.zeros(32), ell, args)
    assert torch.allclose(chi, torch.zeros(32))
    assert torch.allclose(gate, torch.full((32,), 0.5))

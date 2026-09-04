import torch

from cleanrl.iterthink.dg.ppo_continuous_action_iterthink_v24_beta_dg_v5 import (
    Args,
    delight_gate,
    dg_critic_tail_surprisal,
    dg_raw_surprisal,
)


def _gaussian_probs(support, mu, sigma):
    z = (support[None, :] - mu[:, None]) / sigma[:, None]
    return torch.softmax(-0.5 * z * z, dim=-1)


def test_default_surprisal_is_hybrid():
    args = Args()
    assert args.dg_surprisal == "hybrid"
    assert args.dg_eta == 1.0


def test_hybrid_is_average_of_policy_and_critic():
    args = Args()  # hybrid
    cdf = torch.rand(512)
    crit = torch.rand(512)
    out = dg_raw_surprisal(torch.randn(512), torch.randn(512), args,
                           logp_peak=None, cdf_ell=cdf, critic_ell=crit)
    assert torch.allclose(out, 0.5 * (cdf + crit))


def test_hybrid_nonnegative_and_preserves_sign():
    # both channels >= 0 => hybrid >= 0 => sign(chi) = sign(U) (proposition faithfulness)
    cdf = torch.rand(2048) * 3.0
    crit = torch.rand(2048) * 3.0
    out = dg_raw_surprisal(torch.randn(2048), torch.randn(2048), Args(),
                           logp_peak=None, cdf_ell=cdf, critic_ell=crit)
    assert torch.all(out >= -1e-6)
    adv = torch.randn(2048)
    _, _, chi = delight_gate(adv, out, Args())
    assert ((chi > 0) == (adv > 0)).float().mean() > 0.99


def test_hybrid_scale_is_about_one_when_both_calibrated():
    # critic calibrated => E[ell_V] ~ 1; policy cdf_tail ~ 1; average ~ 1 => eta=1 responsive.
    torch.manual_seed(0)
    support = torch.linspace(-10, 10, 511)
    mu = torch.randn(20000) * 3
    sig = torch.full((20000,), 1.0)
    g = mu + sig * torch.randn(20000)
    tp = _gaussian_probs(support, g, torch.full((20000,), 0.4))
    vp = _gaussian_probs(support, mu, sig)
    crit = dg_critic_tail_surprisal(tp, vp, support)
    cdf = torch.empty(20000).exponential_(1.0)  # E~1 surrogate for policy tail
    out = dg_raw_surprisal(torch.randn(20000), torch.randn(20000), Args(),
                           logp_peak=None, cdf_ell=cdf, critic_ell=crit)
    assert 0.7 < out.mean() < 1.3, out.mean().item()


def test_gate_centered_for_typical_surprisal():
    # E[ell]~1 typical samples with mean-zero advantage => gate hovers near 0.5 (graded band).
    cdf = torch.empty(4096).exponential_(1.0)
    crit = torch.empty(4096).exponential_(1.0)
    out = dg_raw_surprisal(torch.randn(4096), torch.randn(4096), Args(),
                           logp_peak=None, cdf_ell=cdf, critic_ell=crit)
    g, _, _ = delight_gate(torch.randn(4096) * 0.1, out, Args())
    assert 0.4 < g.mean() < 0.6

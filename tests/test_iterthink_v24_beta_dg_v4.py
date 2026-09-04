import torch

from cleanrl.iterthink.dg.ppo_continuous_action_iterthink_v24_beta_dg_v4 import (
    Args,
    delight_gate,
    dg_critic_tail_surprisal,
    dg_raw_surprisal,
)


def _gaussian_probs(support, mu, sigma):
    # soft categorical over `support` approximating N(mu, sigma) (HL-Gauss-like)
    z = (support[None, :] - mu[:, None]) / sigma[:, None]
    logp = -0.5 * z * z
    return torch.softmax(logp, dim=-1)


def test_default_surprisal_is_critic():
    args = Args()
    assert args.dg_surprisal == "critic"
    assert args.dg_eta == 1.0


def test_critic_surprisal_nonnegative_and_routes():
    support = torch.linspace(-10, 10, 511)
    mu = torch.randn(256) * 2
    tp = _gaussian_probs(support, mu, torch.full((256,), 1.0))
    vp = _gaussian_probs(support, mu + torch.randn(256) * 0.5, torch.full((256,), 1.0))
    ell = dg_critic_tail_surprisal(tp, vp, support)
    assert torch.all(ell >= -1e-4)
    args = Args()  # critic
    out = dg_raw_surprisal(torch.randn(256), torch.randn(256), args,
                           logp_peak=torch.randn(256), cdf_ell=None, critic_ell=ell)
    assert torch.equal(out, ell)


def test_critic_surprisal_zero_when_return_at_predicted_median():
    # If the realized return G equals the critic's predicted mean/median, F_V(G)=0.5 => ell=0.
    support = torch.linspace(-10, 10, 511)
    mu = torch.zeros(64)  # predicted (and realized) centered at 0
    tp = _gaussian_probs(support, mu, torch.full((64,), 0.8))     # realized return dist ~ centered
    vp = _gaussian_probs(support, mu, torch.full((64,), 0.8))     # predicted == realized
    ell = dg_critic_tail_surprisal(tp, vp, support)
    assert torch.all(ell < 0.15)  # near zero (discretization noise)


def test_critic_surprisal_large_when_outcome_in_predicted_tail():
    # Critic confidently predicted ~0 but the return came in far out => high surprisal.
    support = torch.linspace(-10, 10, 511)
    tp = _gaussian_probs(support, torch.full((64,), 6.0), torch.full((64,), 0.3))  # return ~ +6
    vp = _gaussian_probs(support, torch.full((64,), 0.0), torch.full((64,), 1.0))  # predicted ~ 0
    ell = dg_critic_tail_surprisal(tp, vp, support)
    assert torch.all(ell > 3.0)


def test_critic_surprisal_calibrated_scale_is_about_one():
    # When the critic is calibrated (predicted dist == return dist), F_V(G) ~ Uniform over draws
    # => E[ell_V] ~ 1, matching the policy cdf_tail scale (so eta=1 stays responsive).
    torch.manual_seed(0)
    support = torch.linspace(-10, 10, 511)
    mu = torch.randn(20000) * 3
    sig = torch.full((20000,), 1.0)
    # realized return drawn from the SAME predicted distribution (calibrated)
    g = mu + sig * torch.randn(20000)
    tp = _gaussian_probs(support, g, torch.full((20000,), 0.4))   # tight soft label around G
    vp = _gaussian_probs(support, mu, sig)                        # predicted dist
    ell = dg_critic_tail_surprisal(tp, vp, support)
    assert 0.7 < ell.mean() < 1.3, ell.mean().item()


def test_critic_gate_preserves_sign_agreement():
    support = torch.linspace(-10, 10, 511)
    mu = torch.randn(1024) * 2
    tp = _gaussian_probs(support, mu + torch.randn(1024), torch.full((1024,), 0.5))
    vp = _gaussian_probs(support, mu, torch.full((1024,), 1.0))
    ell = dg_critic_tail_surprisal(tp, vp, support)
    adv = torch.randn(1024)
    _, _, chi = delight_gate(adv, ell, Args())
    assert ((chi > 0) == (adv > 0)).float().mean() > 0.97

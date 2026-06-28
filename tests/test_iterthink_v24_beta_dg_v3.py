import torch
from torch.distributions import Beta

from cleanrl.ppo_continuous_action_iterthink_v24_beta_dg_v3 import (
    Args,
    beta_cdf,
    delight_gate,
    dg_cdf_tail_surprisal,
    dg_raw_surprisal,
)


def test_default_surprisal_is_cdf_tail():
    args = Args()
    assert args.dg_surprisal == "cdf_tail"
    assert args.dg_eta == 1.0 and args.dg_clip == 10.0


def test_beta_cdf_matches_monte_carlo():
    torch.manual_seed(0)
    for a, b in [(1.7, 1.7), (8.0, 3.0), (2.0, 40.0), (20.0, 20.0)]:
        A, B = torch.tensor(a), torch.tensor(b)
        s = Beta(A, B).sample((200000,))
        xs = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        mc = torch.stack([(s <= x).float().mean() for x in xs])
        an = beta_cdf(xs, A.expand(5), B.expand(5))
        assert (mc - an).abs().max() < 5e-3, (a, b, (mc - an).abs().max().item())


def test_cdf_tail_is_nonnegative_and_zero_at_median():
    A = torch.full((1, 6), 8.0)
    B = torch.full((1, 6), 3.0)
    d = Beta(A, B)
    # an action exactly at each dim's median => F=0.5 => min(F,1-F)=0.5 => -log(2*0.5)=0
    median = beta_cdf_inverse_median(A, B)
    ell = dg_cdf_tail_surprisal(median, d)
    assert torch.all(ell >= -1e-4)
    assert torch.allclose(ell, torch.zeros_like(ell), atol=2e-2)
    # random draws are strictly >= 0
    acts = d.sample((4096,) + ()).reshape(-1, 6) if False else d.expand((4096, 6)).sample()
    ell2 = dg_cdf_tail_surprisal(acts.clamp(1e-6, 1 - 1e-6), Beta(A.expand(4096, 6), B.expand(4096, 6)))
    assert torch.all(ell2 >= -1e-4)


def beta_cdf_inverse_median(A, B):
    # crude bisection for the per-dim median (F=0.5), shape like A
    lo = torch.zeros_like(A)
    hi = torch.ones_like(A)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f = beta_cdf(mid, A, B)
        hi = torch.where(f > 0.5, mid, hi)
        lo = torch.where(f > 0.5, lo, mid)
    return 0.5 * (lo + hi)


def test_cdf_tail_keeps_sign_agreement_one():
    torch.manual_seed(1)
    A = torch.full((2048, 6), 12.0)
    B = torch.full((2048, 6), 5.0)
    d = Beta(A, B)
    acts = d.sample().clamp(1e-6, 1 - 1e-6)
    ell = dg_cdf_tail_surprisal(acts, d)
    args = Args()
    adv = torch.randn(2048)
    _, _, chi = delight_gate(adv, ell, args)
    sign_agree = ((chi > 0) == (adv > 0)).float().mean()
    assert sign_agree > 0.97


def test_cdf_tail_is_concentration_invariant():
    # The headline property: a fixed-QUANTILE action gets ~the same surprisal regardless of
    # how sharp the Beta is. Compare the 90th-pct action across concentrations.
    vals = []
    for kappa in [3.0, 20.0, 200.0]:
        A = torch.tensor([[kappa]]); B = torch.tensor([[kappa]])
        q90 = beta_cdf_inverse_quantile(A, B, 0.9)
        ell = dg_cdf_tail_surprisal(q90, Beta(A, B))
        vals.append(ell.item())
    # all close to -log(2*0.1) = 1.609, and to each other
    target = -torch.log(torch.tensor(2 * 0.1)).item()
    for v in vals:
        assert abs(v - target) < 0.05, (vals, target)
    assert max(vals) - min(vals) < 0.05


def beta_cdf_inverse_quantile(A, B, q):
    lo = torch.zeros_like(A); hi = torch.ones_like(A)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f = beta_cdf(mid, A, B)
        hi = torch.where(f > q, mid, hi)
        lo = torch.where(f > q, lo, mid)
    return 0.5 * (lo + hi)


def test_cdf_tail_is_mass_symmetric_unlike_mode_ref():
    # equal-probability tails (5th vs 95th pct) of a skewed Beta get EQUAL cdf_tail surprisal,
    # whereas mode_ref over-penalizes the long tail.
    A = torch.tensor([[8.0]]); B = torch.tensor([[3.0]])
    d = Beta(A, B)
    p05 = beta_cdf_inverse_quantile(A, B, 0.05)
    p95 = beta_cdf_inverse_quantile(A, B, 0.95)
    e05 = dg_cdf_tail_surprisal(p05, d).item()
    e95 = dg_cdf_tail_surprisal(p95, d).item()
    assert abs(e05 - e95) < 0.05  # mass-symmetric
    # mode_ref is asymmetric: logp(mode)-logp(a) differs for the two equal-mass tails
    mode = ((A - 1) / (A + B - 2)).clamp(1e-6, 1 - 1e-6)
    lp_mode = d.log_prob(mode)
    m05 = (lp_mode - d.log_prob(p05)).item()
    m95 = (lp_mode - d.log_prob(p95)).item()
    assert abs(m05 - m95) > 0.3 * max(m05, m95)  # clearly skewed


def test_dg_raw_surprisal_routes_cdf_tail():
    args = Args()  # cdf_tail
    cdf_ell = torch.rand(16)
    out = dg_raw_surprisal(torch.randn(16), torch.randn(16), args, logp_peak=torch.randn(16), cdf_ell=cdf_ell)
    assert torch.equal(out, cdf_ell)

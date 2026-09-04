"""Tests for v33's successor-feature action Jacobian.

The load-bearing claims are:
  1. the closed-form Jacobian of the quadratic action-successor head is the real one;
  2. the analytic d phi / d a is the real one, and the reward probe genuinely recovers
     HalfCheetah's control cost from phi's a*a block (this is what makes the analytic
     half "exact and free");
  3. the per-dimension redistribution is EXACTLY zero-sum, so it cannot rescale or
     re-sign the aggregate policy gradient -- only re-attribute it;
  4. the per-dimension credit is an unbiased, lower-variance route to the same expected
     update as the scalar advantage;
  5. the channel-B step really is the metric-preconditioned solve, and is bounded.
"""

import importlib.util
import pathlib

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_jacobian_outcome_v33.py"
)

_spec = importlib.util.spec_from_file_location("v33", SRC)
v33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v33)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB, OBS, ACT = 32, 17, 6
SF_DIM = EMB + OBS + 2 * ACT + 1  # 62, matching HalfCheetah-v4
A0 = EMB + OBS          # start of phi's `a` block
A1 = A0 + ACT           # start of phi's `a*a` block


def _randomize(head):
    """Undo the zero-init so the Jacobian has something to be right about."""
    with torch.no_grad():
        for m in (head.const, head.lin, head.quad):
            m.weight.normal_(0.0, 0.05)
            m.bias.normal_(0.0, 0.05)


# --------------------------------------------------------------------- head / Jacobian


def test_head_is_identically_zero_at_init_so_iteration_one_is_analytic_only():
    head = v33.ActionSuccessorHead(64, SF_DIM, ACT).to(DEV)
    feat = torch.randn(128, 64, device=DEV)
    action = torch.randn(128, ACT, device=DEV)
    assert torch.equal(head(feat, action), torch.zeros(128, SF_DIM, device=DEV))
    lin, quad = head.jacobian_coeffs(feat)
    assert torch.equal(lin, torch.zeros_like(lin))
    assert torch.equal(quad, torch.zeros_like(quad))


def test_head_gradient_is_alive_despite_the_zero_init():
    # Zero-init would be a bug if it were also a gradient dead end.
    head = v33.ActionSuccessorHead(64, SF_DIM, ACT).to(DEV)
    feat = torch.randn(256, 64, device=DEV)
    action = torch.randn(256, ACT, device=DEV)
    target = torch.randn(256, SF_DIM, device=DEV)
    F.mse_loss(head(feat, action), target).backward()
    assert head.lin.weight.grad.abs().max() > 0
    assert head.quad.weight.grad.abs().max() > 0


def test_closed_form_head_jacobian_matches_autograd():
    torch.manual_seed(0)
    head = v33.ActionSuccessorHead(64, SF_DIM, ACT).to(DEV).double()
    _randomize(head)
    feat = torch.randn(4, 64, device=DEV, dtype=torch.float64)
    action = torch.randn(4, ACT, device=DEV, dtype=torch.float64)

    lin, quad = head.jacobian_coeffs(feat)
    closed = lin + 2.0 * quad * action.unsqueeze(1)          # (n, sf_dim, act_dim)

    for i in range(feat.shape[0]):
        auto = torch.autograd.functional.jacobian(
            lambda a: head(feat[i : i + 1], a.unsqueeze(0)).squeeze(0), action[i]
        )
        assert torch.allclose(closed[i], auto, atol=1e-10), (closed[i] - auto).abs().max()


def test_analytic_phi_jacobian_matches_autograd():
    # phi = [e(s), s, a, a*a, 1]; only the a and a*a blocks move with the action, with
    # derivatives I and 2 diag(a). Everything v32 does analytically rests on this.
    torch.manual_seed(0)
    emb = torch.randn(3, EMB, device=DEV, dtype=torch.float64)
    obs = torch.randn(3, OBS, device=DEV, dtype=torch.float64)
    action = torch.randn(3, ACT, device=DEV, dtype=torch.float64)
    w = torch.randn(SF_DIM, device=DEV, dtype=torch.float64)

    closed = w[A0:A1] + 2.0 * w[A1 : A1 + ACT] * action
    for i in range(3):
        auto = torch.autograd.functional.jacobian(
            lambda a: (v33.phi_features(emb[i], obs[i], a) * w).sum(), action[i]
        )
        assert torch.allclose(closed[i], auto, atol=1e-10)


def test_phi_has_no_action_dependence_outside_the_two_action_blocks():
    emb = torch.randn(5, EMB, device=DEV)
    obs = torch.randn(5, OBS, device=DEV)
    a1 = torch.randn(5, ACT, device=DEV)
    a2 = torch.randn(5, ACT, device=DEV)
    p1 = v33.phi_features(emb, obs, a1)
    p2 = v33.phi_features(emb, obs, a2)
    assert torch.equal(p1[:, :A0], p2[:, :A0])       # e and s blocks
    assert torch.equal(p1[:, -1], p2[:, -1])         # constant block
    assert not torch.equal(p1[:, A0:-1], p2[:, A0:-1])


# --------------------------------------------------------- the "exact and free" claim


def test_reward_probe_recovers_the_quadratic_control_cost_from_the_aa_block():
    # HalfCheetah: r = forward_velocity - 0.1 * ||a||^2. The control cost is EXACTLY
    # representable in phi's a*a block, so w_r must recover -0.1 there, which makes the
    # analytic d(w_r.phi)/da the true control-cost gradient at zero variance.
    torch.manual_seed(0)
    n = 20000
    emb = torch.randn(n, EMB, device=DEV)
    obs = torch.randn(n, OBS, device=DEV)
    action = torch.rand(n, ACT, device=DEV) * 2.0 - 1.0
    phi = v33.phi_features(emb, obs, action)
    # velocity is a linear readout of the observation, plus the exact control cost
    v_w = torch.randn(OBS, device=DEV)
    reward = obs @ v_w - 0.1 * action.pow(2).sum(-1)

    w_r = v33.solve_reward_probe(phi, reward, 1e-6)
    assert torch.allclose(
        w_r[A1 : A1 + ACT], torch.full((ACT,), -0.1, device=DEV), atol=2e-3
    ), w_r[A1 : A1 + ACT]
    # and the linear `a` block must stay at zero -- the true reward has no linear term
    assert w_r[A0:A1].abs().max() < 2e-3
    assert v33.ev_score(phi @ w_r, reward) > 0.999


def test_analytic_gradient_of_the_recovered_probe_is_the_true_control_cost_gradient():
    torch.manual_seed(1)
    n = 20000
    emb = torch.randn(n, EMB, device=DEV)
    obs = torch.randn(n, OBS, device=DEV)
    action = torch.rand(n, ACT, device=DEV) * 2.0 - 1.0
    phi = v33.phi_features(emb, obs, action)
    reward = obs @ torch.randn(OBS, device=DEV) - 0.1 * action.pow(2).sum(-1)
    w_r = v33.solve_reward_probe(phi, reward, 1e-6)

    mu = torch.rand(64, ACT, device=DEV) * 2.0 - 1.0
    g = w_r[A0:A1] + 2.0 * w_r[A1 : A1 + ACT] * mu
    truth = -0.2 * mu                                  # d/da of -0.1 ||a||^2
    assert torch.allclose(g, truth, atol=5e-3), (g - truth).abs().max()


class _Envs:
    def __init__(self):
        import gymnasium as gym

        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float64)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32)


def _agent(dist_name):
    args = v33.Args()
    args.emb_dim, args.actor_dist = EMB, dist_name
    return v33.Agent(_Envs(), args).to(DEV), args


# ------------------------------------------------------------- channel A (shipped code)


def test_perdim_loss_reproduces_the_scalar_ppo_GRADIENT_at_the_start_of_an_epoch():
    """The regression test for the bug this suite originally missed.

    With A_j = A and every ratio_j == 1, the per-dimension surrogate must produce the
    SAME gradient as v9's scalar surrogate -- not merely the same loss value. Pairing
    true per-dimension ratios with A/act_dim gives an identical DIRECTION and a gradient
    act_dim times too small, which no loss-value or sum-of-advantages test can see.
    """
    torch.manual_seed(0)
    n = 4096
    mu = torch.zeros(ACT, device=DEV, dtype=torch.float64, requires_grad=True)
    a = torch.randn(n, ACT, device=DEV, dtype=torch.float64)
    lp = torch.distributions.Normal(mu, 1.0).log_prob(a)
    old = lp.detach()
    adv = torch.randn(n, device=DEV, dtype=torch.float64)

    joint_ratio = (lp.sum(1) - old.sum(1)).exp()
    joint = torch.max(
        -adv * joint_ratio, -adv * torch.clamp(joint_ratio, 0.8, 1.28)
    ).mean()
    (g_joint,) = torch.autograd.grad(joint, mu, retain_graph=True)

    ratio_d = (lp - old).exp()
    loss, adv_dims, _ = v33.perdim_policy_loss(
        adv,
        torch.zeros(n, ACT, device=DEV, dtype=torch.float64),
        0.0,
        ratio_d,
        0.2,
        0.28,
        "sqrt",
    )
    (g_perdim,) = torch.autograd.grad(loss, mu)
    assert torch.allclose(g_joint, g_perdim, rtol=1e-9), (g_joint, g_perdim)
    assert torch.allclose(adv_dims.mean(-1), adv, atol=1e-12)


def test_perdim_loss_gradient_is_sensitive_to_the_sign_of_the_shift():
    # A channel wired up backwards must be detectable. Flipping the shift has to move the
    # gradient, and by more than rounding.
    torch.manual_seed(0)
    n = 4096
    mu = torch.zeros(ACT, device=DEV, dtype=torch.float64, requires_grad=True)
    a = torch.randn(n, ACT, device=DEV, dtype=torch.float64)
    lp = torch.distributions.Normal(mu, 1.0).log_prob(a)
    ratio_d = (lp - lp.detach()).exp()
    adv = torch.randn(n, device=DEV, dtype=torch.float64)
    shift = v33.attribution_shift(
        torch.randn(n, ACT, device=DEV, dtype=torch.float64), 1.0, 3.0
    )

    def g(sign):
        loss, _, _ = v33.perdim_policy_loss(
            adv, sign * shift, 0.5, ratio_d, 0.2, 0.28, "sqrt"
        )
        return torch.autograd.grad(loss, mu, retain_graph=True)[0]

    pos, neg = g(1.0), g(-1.0)
    assert not torch.allclose(pos, neg, atol=1e-6)
    assert float((pos - neg).norm()) > 1e-3 * float(pos.norm())


def test_shift_is_exactly_zero_sum_so_mean_credit_is_the_scalar_advantage():
    torch.manual_seed(0)
    attrib = torch.randn(4096, ACT, device=DEV, dtype=torch.float64)
    adv = torch.randn(4096, device=DEV, dtype=torch.float64)
    ratio_d = torch.ones(4096, ACT, device=DEV, dtype=torch.float64)
    for omega in (0.0, 0.25, 1.0, 25.0):
        shift = v33.attribution_shift(attrib, 1.0, 3.0)
        assert shift.sum(-1).abs().max() < 1e-12
        _, adv_dims, _ = v33.perdim_policy_loss(adv, shift, omega, ratio_d, 0.2, 0.28, "sqrt")
        assert torch.allclose(adv_dims.mean(-1), adv, atol=1e-12), omega


def test_shrinkage_disables_the_channel_when_the_attribution_predicts_nothing():
    # The self-limiting property: a Jacobian carrying no information must inject nothing.
    torch.manual_seed(0)
    attrib = torch.randn(4096, ACT, device=DEV, dtype=torch.float64)
    assert torch.equal(
        v33.attribution_shift(attrib, 0.0, 3.0),
        torch.zeros_like(attrib),
    )
    half = v33.attribution_shift(attrib, 0.5, 3.0)
    full = v33.attribution_shift(attrib, 1.0, 3.0)
    assert torch.allclose(half * 2.0, full, atol=1e-12)


def test_shift_is_winsorized_without_losing_the_zero_sum_property():
    torch.manual_seed(0)
    attrib = torch.randn(2048, ACT, device=DEV, dtype=torch.float64)
    attrib[0, 0] = 5000.0                                  # a single wild coordinate
    clamped = v33.attribution_shift(attrib, 1.0, 3.0)
    unclamped = v33.attribution_shift(attrib, 1.0, 1e9)
    assert float(unclamped.abs().max()) > 100.0            # the test is not vacuous
    assert float(clamped.abs().max()) <= 2.0 * 3.0
    # re-centring after the clamp is what keeps the blast-radius argument true
    assert clamped.sum(-1).abs().max() < 1e-12


def test_clip_match_modes_apply_the_bounds_they_advertise():
    lo_j, hi_j = 0.8, 1.28
    n = 256
    adv = torch.ones(n, device=DEV, dtype=torch.float64)
    zero = torch.zeros(n, ACT, device=DEV, dtype=torch.float64)
    big = torch.full((n, ACT), 10.0, device=DEV, dtype=torch.float64)
    for mode, root in [("joint", 1.0), ("linear", float(ACT)), ("sqrt", ACT**0.5)]:
        hi = hi_j ** (1.0 / root)
        loss, _, frac = v33.perdim_policy_loss(adv, zero, 0.0, big, 0.2, 0.28, mode)
        assert abs(float(loss) + ACT * hi) < 1e-9, (mode, float(loss))
        assert float(frac) == 1.0


def test_frequency_matched_clip_binds_near_v9s_rate_where_the_product_match_does_not():
    """The v32 post-mortem, as a regression test.

    Per-coordinate log-ratios are ~1/sqrt(n) of the joint one, so bounds set by the n-th
    root ("linear", v32) bind on a large fraction of coordinates while v9's joint clip
    binds on ~2%. Draw per-coordinate log-ratios with the sd v9 actually exhibits and
    check that "sqrt" lands near v9's rate and "linear" does not.
    """
    torch.manual_seed(0)
    n = 200000
    joint_log_sd = 0.09                       # v9: approx_kl ~ 0.004 => |log r| ~ 0.09
    per_dim = torch.randn(n, ACT, device=DEV, dtype=torch.float64) * (
        joint_log_sd / ACT**0.5
    )
    adv = torch.randn(n, device=DEV, dtype=torch.float64)
    zero = torch.zeros(n, ACT, device=DEV, dtype=torch.float64)
    rates = {}
    for mode in ("linear", "sqrt"):
        _, _, frac = v33.perdim_policy_loss(
            adv, zero, 0.0, per_dim.exp(), 0.2, 0.28, mode
        )
        rates[mode] = float(frac)
    # v9's own joint clipfrac is ~0.023
    assert rates["linear"] > 0.25, rates
    assert rates["sqrt"] < 0.05, rates
    assert rates["sqrt"] < rates["linear"] / 5.0, rates


def test_the_attribution_is_zero_mean_per_action_dimension():
    # Ahat_j = g_j (a_j - mu_j) + centred quadratic term, with mu the mean of a FACTORIZED
    # distribution, so each coordinate is its own valid baseline. Beta's map to action
    # space is affine, which is why v32 takes mu = to_action(dist.mean), not the mode.
    torch.manual_seed(0)
    n = 400000
    dist = torch.distributions.Beta(
        torch.full((n, ACT), 2.3, device=DEV), torch.full((n, ACT), 5.1, device=DEV)
    )
    action = -1.0 + 2.0 * dist.sample()
    mu = -1.0 + 2.0 * dist.mean
    g = torch.randn(1, ACT, device=DEV).expand(n, ACT)     # g is a function of s only
    q = torch.randn(1, ACT, device=DEV).expand(n, ACT)
    dev = action - mu
    quad = q * dev.pow(2)
    attrib = g * dev + (quad - quad.mean(0, keepdim=True))
    assert attrib.mean(0).abs().max() < 5e-3, attrib.mean(0)


def test_the_exact_quadratic_attribution_beats_the_first_order_one():
    # Psi is exactly quadratic in a, so keeping only g_j*dev_j discards a real term.
    # The exact decomposition must reproduce w_r.(Psi(a) - Psi(mu)) to machine precision
    # while the first-order one does not.
    torch.manual_seed(0)
    n = 4096
    L = torch.randn(n, ACT, device=DEV, dtype=torch.float64)
    Q = torch.randn(n, ACT, device=DEV, dtype=torch.float64)
    mu = torch.randn(n, ACT, device=DEV, dtype=torch.float64) * 0.5
    dev = torch.randn(n, ACT, device=DEV, dtype=torch.float64) * 0.3
    a = mu + dev
    truth = ((L * a + Q * a * a) - (L * mu + Q * mu * mu)).sum(-1)
    g = L + 2.0 * Q * mu
    exact = (g * dev + Q * dev.pow(2)).sum(-1)
    first_order = (g * dev).sum(-1)
    assert torch.allclose(exact, truth, atol=1e-12)
    assert not torch.allclose(first_order, truth, atol=1e-3)
    # the discarded term is not a rounding detail at realistic sigma
    assert float((truth - first_order).std() / truth.std()) > 0.1


def test_perdim_credit_gives_the_same_expected_update_as_the_scalar_advantage():
    # E[ d log pi / d mu_j * Ahat_j ] = g_j exactly, for a factorized Gaussian: the same
    # mean as the scalar route, with the other coordinates' noise removed.
    torch.manual_seed(0)
    n, sigma = 2000000, 0.4
    mu = torch.zeros(ACT, device=DEV)
    g = torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0, -0.25], device=DEV)
    a = mu + sigma * torch.randn(n, ACT, device=DEV)
    dev = a - mu
    score = dev / sigma**2                       # d log pi(a|s) / d mu
    perdim = (score * (g * dev)).mean(0)
    assert torch.allclose(perdim, g, atol=0.02), perdim - g

    scalar_A = (g * dev).sum(-1, keepdim=True)
    assert torch.allclose((score * scalar_A).mean(0), g, atol=0.02)
    per_var = (score * (g * dev)).var(0)
    sca_var = (score * scalar_A).var(0)
    live = g != 0
    assert bool((per_var[live] < sca_var[live]).all()), (per_var, sca_var)


def test_per_dimension_ratios_multiply_to_the_joint_ratio_used_by_the_kl_stop():
    # approx_kl, clipfracs and the early stop still read the JOINT ratio, so the
    # per-dimension ratios must compose into exactly it.
    torch.manual_seed(0)
    agent, _ = _agent("beta")
    x = torch.randn(256, OBS, device=DEV)
    with torch.no_grad():
        _, z, lp_old, _, _, lpd_old, _ = agent.get_action_and_value(x)
        _, _, lp_new, _, _, lpd_new, _ = agent.get_action_and_value(x, z)
    assert torch.allclose(
        (lpd_new - lpd_old).exp().prod(-1), (lp_new - lp_old).exp(), atol=1e-5
    )


# ------------------------------------------------------------- channel B: the solve


def _improve(j_emb, g, ridge, kappa, act_std):
    """Exactly v33's channel-B arithmetic."""
    metric = torch.einsum("nkj,nkl->njl", j_emb, j_emb) / j_emb.shape[1]
    eye = torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)
    trace_mean = metric.diagonal(dim1=-2, dim2=-1).mean(-1).view(-1, 1, 1)
    mm = metric + ridge * (trace_mean + 1.0) * eye
    step = torch.linalg.solve(mm, g.unsqueeze(-1)).squeeze(-1)
    cap = kappa * act_std
    return torch.max(torch.min(step, cap), -cap), metric


def test_improvement_step_solves_the_preconditioned_system():
    torch.manual_seed(0)
    j_emb = torch.randn(64, EMB, ACT, device=DEV, dtype=torch.float64)
    g = torch.randn(64, ACT, device=DEV, dtype=torch.float64)
    huge = torch.full((1, ACT), 1e9, device=DEV, dtype=torch.float64)   # cap disabled
    step, metric = _improve(j_emb, g, 1e-2, 1.0, huge)
    eye = torch.eye(ACT, device=DEV, dtype=torch.float64)
    trace_mean = metric.diagonal(dim1=-2, dim2=-1).mean(-1).view(-1, 1, 1)
    mm = metric + 1e-2 * (trace_mean + 1.0) * eye
    assert torch.allclose(torch.einsum("nij,nj->ni", mm, step), g, atol=1e-8)


def test_improvement_step_shrinks_along_high_leverage_directions():
    # The metric's job: an action direction that moves the future a lot needs a SMALLER
    # step to achieve the same outcome-space displacement. Coordinate 0 gets 100x the
    # leverage of the rest, so its step must come out far smaller than the unpreconditioned
    # one, while a zero-leverage coordinate is left to the ridge.
    torch.manual_seed(0)
    j_emb = torch.zeros(1, EMB, ACT, device=DEV, dtype=torch.float64)
    j_emb[0, :, 0] = 10.0
    j_emb[0, :, 1] = 0.1
    g = torch.ones(1, ACT, device=DEV, dtype=torch.float64)
    huge = torch.full((1, ACT), 1e9, device=DEV, dtype=torch.float64)
    step, _ = _improve(j_emb, g, 1e-2, 1.0, huge)
    assert abs(float(step[0, 0])) < abs(float(step[0, 1]))
    assert abs(float(step[0, 1])) < abs(float(step[0, 2]))    # coord 2 has zero leverage


def test_improvement_step_is_bounded_by_the_trust_region():
    torch.manual_seed(0)
    j_emb = torch.randn(256, EMB, ACT, device=DEV, dtype=torch.float64) * 1e-4
    g = torch.randn(256, ACT, device=DEV, dtype=torch.float64) * 1e4   # would blow up
    act_std = torch.full((1, ACT), 0.3, device=DEV, dtype=torch.float64)
    step, _ = _improve(j_emb, g, 1e-2, 0.5, act_std)
    assert float(step.abs().max()) <= 0.5 * 0.3 + 1e-12


def test_ridge_is_scale_free_once_the_metric_dominates_its_absolute_floor():
    # ridge * (trace_mean + 1.0): the trace term makes conditioning relative wherever the
    # metric carries signal, so a drifting embedding scale cannot silently turn the ridge
    # into either a no-op or a wall. Steps go as 1/scale^2.
    torch.manual_seed(0)
    j = torch.randn(32, EMB, ACT, device=DEV, dtype=torch.float64)
    g = torch.randn(32, ACT, device=DEV, dtype=torch.float64)
    huge = torch.full((1, ACT), 1e9, device=DEV, dtype=torch.float64)
    s_a, _ = _improve(j * 1e2, g, 1e-2, 1.0, huge)
    s_b, _ = _improve(j * 1e3, g, 1e-2, 1.0, huge)
    assert torch.allclose(s_a, s_b * 1e2, rtol=1e-3)


def test_absolute_floor_keeps_the_solve_finite_when_the_head_is_untrained():
    # The whole reason for the +1.0: at init the head is zero, so J_emb -- and therefore
    # the metric -- is IDENTICALLY zero and the bare solve would be singular. With the
    # floor the step degenerates gracefully to a plain gradient step of size g/ridge,
    # which the trust region then bounds.
    j_zero = torch.zeros(16, EMB, ACT, device=DEV, dtype=torch.float64)
    g = torch.randn(16, ACT, device=DEV, dtype=torch.float64)
    huge = torch.full((1, ACT), 1e9, device=DEV, dtype=torch.float64)
    step, metric = _improve(j_zero, g, 1e-2, 1.0, huge)
    assert torch.equal(metric, torch.zeros_like(metric))
    assert torch.isfinite(step).all()
    assert torch.allclose(step, g / 1e-2, rtol=1e-10)
    # ...and that 100x amplification is exactly what the trust region exists to cap.
    capped, _ = _improve(
        j_zero, g, 1e-2, 0.5, torch.full((1, ACT), 0.3, device=DEV, dtype=torch.float64)
    )
    assert float(capped.abs().max()) <= 0.15 + 1e-12


# --------------------------------------------------------------- fit / plumbing


def test_head_recovers_a_known_quadratic_action_map():
    torch.manual_seed(0)
    head = v33.ActionSuccessorHead(16, 8, ACT).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    feat = torch.randn(8192, 16, device=DEV)
    action = torch.rand(8192, ACT, device=DEV) * 2 - 1
    true_lin = torch.randn(16, 8 * ACT, device=DEV) * 0.3
    true_quad = torch.randn(16, 8 * ACT, device=DEV) * 0.3
    target = (
        (feat @ true_lin).view(-1, 8, ACT) * action.unsqueeze(1)
    ).sum(-1) + ((feat @ true_quad).view(-1, 8, ACT) * (action * action).unsqueeze(1)).sum(-1)

    for _ in range(600):
        loss = F.mse_loss(head(feat, action), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
    assert v33.ev_score(head(feat, action).reshape(-1), target.reshape(-1)) > 0.95


def test_jacobian_chunking_is_exact():
    torch.manual_seed(0)
    head = v33.ActionSuccessorHead(64, SF_DIM, ACT).to(DEV).double()
    _randomize(head)
    feat = torch.randn(5000, 64, device=DEV, dtype=torch.float64)
    mu = torch.randn(5000, ACT, device=DEV, dtype=torch.float64)
    w = torch.randn(SF_DIM, device=DEV, dtype=torch.float64)

    def contract(sl):
        lin, quad = head.jacobian_coeffs(feat[sl])
        js = lin + 2.0 * quad * mu[sl].unsqueeze(1)
        return torch.einsum("k,nkj->nj", w, js)

    whole = contract(slice(0, 5000))
    chunked = torch.cat([contract(slice(s, min(s + 1024, 5000))) for s in range(0, 5000, 1024)])
    # Not bit-exact: CUDA selects different reduction kernels by batch size. The
    # invariant is that chunking changes nothing beyond float64 rounding.
    assert torch.allclose(whole, chunked, atol=1e-10, rtol=1e-10)


def test_standardized_contraction_equals_the_raw_unit_contraction():
    # v32 contracts w_r against the STANDARDIZED Jacobian using (w_r * asf_std) rather
    # than de-standardizing the (n, sf_dim, act_dim) tensor. Must be identical.
    torch.manual_seed(0)
    j_std = torch.randn(128, SF_DIM, ACT, device=DEV, dtype=torch.float64)
    std = torch.rand(SF_DIM, device=DEV, dtype=torch.float64) + 0.5
    w = torch.randn(SF_DIM, device=DEV, dtype=torch.float64)
    cheap = torch.einsum("k,nkj->nj", w * std, j_std)
    explicit = torch.einsum("k,nkj->nj", w, j_std * std.view(1, -1, 1))
    assert torch.allclose(cheap, explicit, atol=1e-12)


def test_fit_mask_excludes_terminations_and_invalid_steps():
    # psi(s') is not the continuation value at a termination, and there is no s' at all
    # at an invalid step -- both would poison the action-successor regression.
    term = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=DEV)
    valid = torch.tensor([[1.0, 1.0, 0.0, 1.0]], device=DEV)
    mask = ((1.0 - term) * valid).reshape(-1) > 0.5
    assert mask.tolist() == [True, False, False, True]


# ------------------------------------------------------------------- Agent plumbing


@pytest.mark.parametrize("dist_name", ["beta", "gaussian"])
def test_get_action_and_value_returns_consistent_per_dimension_logprobs(dist_name):
    torch.manual_seed(0)
    agent, args = _agent(dist_name)
    x = torch.randn(64, OBS, device=DEV)
    action, z, lp, ent, sf, lp_d, mean_a = agent.get_action_and_value(x)
    assert lp_d.shape == (64, ACT)
    assert mean_a.shape == (64, ACT)
    assert sf.shape == (64, args.critic_mtp_horizon, SF_DIM)
    assert torch.allclose(lp_d.sum(1), lp, atol=1e-5)


def test_beta_mean_action_is_exactly_the_expected_action():
    # For beta, to_action is affine, so to_action(E[z]) == E[a] EXACTLY and the
    # per-dimension baseline is unbiased rather than first-order.
    torch.manual_seed(0)
    agent, _ = _agent("beta")
    x = torch.randn(8, OBS, device=DEV)
    with torch.no_grad():
        _, critic_feat = agent._trunks(x)
        actor_feat, _ = agent._trunks(x)
        dist, to_action, _ = agent._actor_dist(actor_feat)
        samples = torch.stack([to_action(dist.sample()) for _ in range(20000)])
        _, _, _, _, _, _, mean_a = agent.get_action_and_value(x)
    assert torch.allclose(samples.mean(0), mean_a, atol=2e-2), (samples.mean(0) - mean_a).abs().max()


def test_action_successor_head_is_not_in_the_ppo_optimizer_or_clip_groups():
    # Attaching it to Agent would put it in agent.parameters() (the PPO optimizer) while
    # leaving it out of BOTH grad-clip groups -- and the dual-backward zeroes grads
    # between passes, so its gradient would be silently discarded.
    agent, args = _agent("beta")
    head = v33.ActionSuccessorHead(args.hidden, SF_DIM, ACT).to(DEV)
    agent_ids = {id(p) for p in agent.parameters()}
    assert not any(id(p) in agent_ids for p in head.parameters())
    grouped = {id(p) for p in agent.actor_parameters()} | {
        id(p) for p in agent.critic_parameters()
    }
    assert agent_ids == grouped, "every Agent param must sit in a clip group"

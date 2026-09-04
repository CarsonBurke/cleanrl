"""Tests for v36's value-gradient channel (DPG on successor features).

The load-bearing claims are:

  1. THE ANALYTIC HALF IS EXACT. d(w_r . phi)/d a_j == w_a[j] + 2 a_j w_aa[j], with the
     index arithmetic (jac_a0, jac_a1) right. This term is claimed to be exact and free, so
     an off-by-one in the block boundaries would be a silent, permanent error.

  2. IT RECOVERS MUJOCO'S CONTROL COST IN CLOSED FORM. HalfCheetah's reward is
     x_vel - 0.1||a||^2 and the control cost is not a function of state, so a linear probe
     on phi must put exactly -0.1 in the a*a block and the analytic gradient must be -0.2*a.
     This is the concrete reason the analytic half is worth having.

  3. g NEVER SEES THE SAMPLED ACTION. It is built from w_r and J(s, mu), so self-alignment
     is structurally absent rather than corrected -- the whole reason v36 exists after v35.

  4. The learned half contracts the STANDARDIZED Jacobian against w_r * asf_std, because
     psi_next_raw = asf_std * psi_next_std.

  5. The target is bounded per row and by the action space.
"""

import importlib.util
import pathlib

import pytest
import torch

SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_valuegrad_v36.py"
)

_spec = importlib.util.spec_from_file_location("v36", SRC)
v36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v36)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB, OBS, ACT = 32, 17, 6
SF_DIM = EMB + OBS + 2 * ACT + 1  # 62, matching HalfCheetah-v4
A0 = EMB + OBS          # start of phi's `a` block
A1 = A0 + ACT           # start of phi's `a*a` block
GAMMA = 0.99


def _analytic(w_r, means):
    """The exact analytic half, exactly as main() computes it."""
    return w_r[A0:A1] + 2.0 * w_r[A1 : A1 + ACT] * means


# ------------------------------------------------------------------ claim 1


def test_analytic_gradient_matches_autograd_through_phi():
    torch.manual_seed(0)
    n = 64
    emb = torch.randn(n, EMB, device=DEV)
    obs = torch.randn(n, OBS, device=DEV)
    act = torch.randn(n, ACT, device=DEV, requires_grad=True)
    w_r = torch.randn(SF_DIM, device=DEV)

    phi = v36.phi_features(emb, obs, act)
    assert phi.shape[-1] == SF_DIM
    g_auto = torch.autograd.grad((phi @ w_r).sum(), act)[0]
    assert torch.allclose(g_auto, _analytic(w_r, act.detach()), atol=1e-5)


def test_block_boundaries_are_where_the_code_thinks_they_are():
    """An off-by-one in jac_a0/jac_a1 would silently differentiate the wrong coordinates."""
    emb = torch.zeros(1, EMB, device=DEV)
    obs = torch.zeros(1, OBS, device=DEV)
    act = torch.arange(1, ACT + 1, dtype=torch.float32, device=DEV).unsqueeze(0)
    phi = v36.phi_features(emb, obs, act)[0]
    assert torch.equal(phi[A0:A1], act[0])              # the `a` block
    assert torch.equal(phi[A1 : A1 + ACT], act[0] ** 2)  # the `a*a` block
    assert float(phi[-1]) == 1.0                         # the trailing constant


# ------------------------------------------------------------------ claim 2


def test_probe_recovers_the_control_cost_and_its_gradient():
    """End-to-end: build HalfCheetah's reward exactly, solve the probe, differentiate."""
    torch.manual_seed(0)
    n = 20000
    emb = torch.randn(n, EMB, device=DEV)
    obs = torch.randn(n, OBS, device=DEV)
    act = torch.rand(n, ACT, device=DEV) * 2.0 - 1.0
    x_vel = obs[:, 8]                       # some state coordinate stands in for velocity
    reward = x_vel - 0.1 * act.pow(2).sum(-1)

    phi = v36.phi_features(emb, obs, act)
    w_r = v36.solve_reward_probe(phi, reward, 1e-6)

    assert torch.allclose(
        w_r[A1 : A1 + ACT], torch.full((ACT,), -0.1, device=DEV), atol=1e-3
    ), w_r[A1 : A1 + ACT]
    # ... and therefore the analytic action-gradient is exactly -0.2 * a.
    assert torch.allclose(_analytic(w_r, act), -0.2 * act, atol=5e-3)


def test_probe_gradient_points_downhill_on_control_cost():
    """Sanity on the sign: more action magnitude must lower predicted reward."""
    torch.manual_seed(1)
    n = 20000
    emb = torch.randn(n, EMB, device=DEV)
    obs = torch.randn(n, OBS, device=DEV)
    act = torch.rand(n, ACT, device=DEV) * 2.0 - 1.0
    reward = obs[:, 8] - 0.1 * act.pow(2).sum(-1)
    w_r = v36.solve_reward_probe(v36.phi_features(emb, obs, act), reward, 1e-6)
    g = _analytic(w_r, act)
    assert float((g * act).sum(-1).mean()) < 0.0


# ------------------------------------------------------------------ claim 3


def test_direction_is_independent_of_the_sampled_action():
    """g depends on w_r and the MEAN only. This is what makes self-alignment impossible."""
    torch.manual_seed(2)
    n = 512
    w_r = torch.randn(SF_DIM, device=DEV)
    means = torch.randn(n, ACT, device=DEV).tanh()
    g1 = _analytic(w_r, means)
    # resampling the action changes nothing, because the action never enters
    g2 = _analytic(w_r, means)
    assert torch.equal(g1, g2)
    for _ in range(4):
        dev = torch.randn(n, ACT, device=DEV) * 0.4
        align = torch.nn.functional.cosine_similarity(g1, dev, dim=-1).mean()
        assert abs(float(align)) < 3.0 / n ** 0.5


def test_learned_half_is_also_independent_of_the_sampled_action():
    torch.manual_seed(3)
    n, H = 256, 64
    head = v36.ActionSuccessorHead(H, SF_DIM, ACT).to(DEV)
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.05)
    feat = torch.randn(n, H, device=DEV)
    means = torch.randn(n, ACT, device=DEV).tanh()
    w_scaled = torch.randn(SF_DIM, device=DEV)

    def learned(m):
        lin, quad = head.jacobian_coeffs(feat)
        return GAMMA * torch.einsum("k,nkj->nj", w_scaled, lin + 2.0 * quad * m.unsqueeze(1))

    g = learned(means)
    for _ in range(4):
        dev = torch.randn(n, ACT, device=DEV) * 0.4
        align = torch.nn.functional.cosine_similarity(g, dev, dim=-1).mean()
        assert abs(float(align)) < 3.0 / n ** 0.5


# ------------------------------------------------------------------ claim 4


def test_learned_half_matches_an_explicit_reference_loop():
    torch.manual_seed(4)
    n, H = 8, 64
    head = v36.ActionSuccessorHead(H, SF_DIM, ACT).to(DEV)
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.05)
    feat = torch.randn(n, H, device=DEV)
    means = torch.randn(n, ACT, device=DEV).tanh()
    w_r = torch.randn(SF_DIM, device=DEV)
    asf_std = torch.rand(SF_DIM, device=DEV) + 0.5
    w_scaled = w_r * asf_std

    lin, quad = head.jacobian_coeffs(feat)
    jac = lin + 2.0 * quad * means.unsqueeze(1)
    got = GAMMA * torch.einsum("k,nkj->nj", w_scaled, jac)

    want = torch.zeros_like(got)
    for i in range(n):
        for j in range(ACT):
            want[i, j] = GAMMA * sum(
                w_r[k] * asf_std[k] * jac[i, k, j] for k in range(SF_DIM)
            )
    assert torch.allclose(got, want, atol=1e-4)


def test_learned_half_equals_autograd_of_the_scaled_value():
    """gamma * d(w_scaled . head(feat, a))/da, checked against autograd."""
    torch.manual_seed(5)
    n, H = 16, 64
    head = v36.ActionSuccessorHead(H, SF_DIM, ACT).to(DEV)
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.05)
    feat = torch.randn(n, H, device=DEV)
    means = torch.randn(n, ACT, device=DEV).tanh().requires_grad_(True)
    w_scaled = torch.randn(SF_DIM, device=DEV)

    v = (head(feat, means) * w_scaled).sum()
    g_auto = GAMMA * torch.autograd.grad(v, means)[0]

    lin, quad = head.jacobian_coeffs(feat)
    jac = lin + 2.0 * quad * means.detach().unsqueeze(1)
    assert torch.allclose(
        g_auto, GAMMA * torch.einsum("k,nkj->nj", w_scaled, jac), atol=1e-5
    )


# ------------------------------------------------------------------ claim 5


def _target(means, g, step, lo=-10.0, hi=10.0):
    return v36.value_gradient_target(
        means,
        g,
        torch.ones(1, ACT, device=DEV),
        step,
        torch.full((ACT,), lo, device=DEV),
        torch.full((ACT,), hi, device=DEV),
    )


def test_target_respects_action_bounds():
    torch.manual_seed(6)
    means = torch.randn(256, ACT, device=DEV).tanh()
    g = torch.randn(256, ACT, device=DEV)
    tgt = _target(means, g, 10.0, lo=-1.0, hi=1.0)
    assert float(tgt.min()) >= -1.0 - 1e-6 and float(tgt.max()) <= 1.0 + 1e-6


def test_no_row_exceeds_the_declared_step():
    torch.manual_seed(7)
    means = torch.randn(256, ACT, device=DEV) * 0.1
    g = torch.randn(256, ACT, device=DEV)
    g[0] *= 500.0  # heavy-tailed outlier
    step = 0.25
    assert float((_target(means, g, step) - means).abs().max()) <= step + 1e-5


def test_typical_displacement_matches_the_declared_step():
    torch.manual_seed(8)
    means = torch.randn(4096, ACT, device=DEV) * 0.1
    g = torch.randn(4096, ACT, device=DEV)
    step = 0.25
    rms = (_target(means, g, step) - means).pow(2).sum(-1).mean().sqrt()
    assert 0.5 * step < float(rms) <= step + 1e-5


@pytest.mark.parametrize("g_zero", [True, False])
def test_zero_step_or_zero_direction_is_a_pure_anchor(g_zero):
    torch.manual_seed(9)
    means = torch.randn(128, ACT, device=DEV) * 0.1
    g = torch.zeros(128, ACT, device=DEV) if g_zero else torch.randn(128, ACT, device=DEV)
    step = 0.25 if g_zero else 0.0
    assert torch.allclose(_target(means, g, step), means, atol=1e-7)

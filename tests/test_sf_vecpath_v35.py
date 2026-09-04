"""Tests for v35's vector-pathwise channel.

The load-bearing claims are:

  1. THE SELF-ALIGNMENT GUARD. u is built from sf_residual, whose state coordinates contain
     gamma*psi(s_{t+1}), and s_{t+1} depends on a_t. Without correction the expected step
     contains gamma^2 J^T diag(asf_std/res_std) J dev, a PSD form, so E[g.dev] >= 0
     regardless of reward -- the policy would be pulled toward whatever it just sampled.
     A shuffle control CANNOT catch this, because shuffling destroys the self-alignment too.
     vector_pathwise_debias must remove it.

  2. The action, a*a and constant blocks of the vector advantage can never drive the mean.

  3. mode="perp" strips BOTH reward directions: (w_r * asf_std), which is what makes the
     ascended functional carry no return, and (w_r * res_std), which is what makes the arm
     the true complement of what adv_vector = sf_residual @ w_r discards. They are not
     parallel, so projecting out only one leaves most of the advantage's variance behind.

  4. The contraction is scale-free: rescaling any phi coordinate changes nothing. phi mixes
     a whitened latent, raw observations, actions and actions squared, so a contraction
     that is not scale-free is a vote by whichever block carries the largest units.

  5. The quadratic head's closed-form Jacobian is the real derivative.

  6. The target is bounded per row and by the action space, so the regression cannot walk
     the mean off the action range across PPO epochs.
"""

import importlib.util
import pathlib

import pytest
import torch

SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vecpath_v35.py"
)

_spec = importlib.util.spec_from_file_location("v35", SRC)
v35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v35)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB, OBS, ACT = 32, 17, 6
SF_DIM = EMB + OBS + 2 * ACT + 1  # 62, matching HalfCheetah-v4
A0 = EMB + OBS  # end of the state blocks == start of phi's `a` block
N = 4096
GAMMA = 0.99


def _world(seed=0, n=N, contaminate=True, surprise=1.0):
    """Synthesise a batch whose contamination structure matches the real rollout.

    sf_residual's state block = (surprise independent of the action)
                              + gamma * boot * asf_std * (J . dev)
    which is exactly the term vector_pathwise_debias exists to remove.

    ONE generator, drawn sequentially. Separate generators seeded with adjacent integers
    produce correlated streams, which shows up here as a spurious self-alignment floor and
    would make the debias look imperfect when it is exact.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(DEV)
    jac = rnd(n, SF_DIM, ACT) * 0.3
    dev = rnd(n, ACT) * 0.4
    means = (rnd(n, ACT) * 0.5).tanh()
    w_r = rnd(SF_DIM)
    asf_std = (torch.rand(SF_DIM, generator=g).to(DEV) + 0.5) * 2.0
    boot = (torch.rand(n, 1, generator=g).to(DEV) > 0.02).float()
    res = rnd(n, SF_DIM) * surprise
    if contaminate:
        res = res + GAMMA * boot * asf_std * torch.einsum("nkj,nj->nk", jac, dev)
    return dict(res=res, jac=jac, dev=dev, means=means, w_r=w_r, asf_std=asf_std, boot=boot)


def _pipeline(w, mode="perp", debias=True):
    """Mirror of the (chunked) main-loop pipeline, unchunked."""
    u, res_std = v35.vector_pathwise_outcome_direction(w["res"], A0)
    if debias:
        u = v35.vector_pathwise_debias(
            u, w["jac"], w["dev"], w["asf_std"], res_std, w["boot"], GAMMA, A0
        )
    if mode == "perp":
        basis = v35.vector_pathwise_reward_basis(w["w_r"], w["asf_std"], res_std, A0)
        u = u - (u @ basis.T) @ basis
    return GAMMA * torch.einsum("nk,nkj->nj", u, w["jac"]), u, res_std


def _self_align(g, dev):
    return float(torch.nn.functional.cosine_similarity(g, dev, dim=-1).mean())


# ------------------------------------------------------------ claim 1 (the big one)


@pytest.mark.parametrize("mode", ["perp", "full"])
def test_undebiased_direction_chases_its_own_action_noise(mode):
    """The failure mode must actually be present, or the next test proves nothing."""
    w = _world(10)
    g, _, _ = _pipeline(w, mode, debias=False)
    # Measured at +0.55 to +0.92 depending on how much independent surprise there is --
    # far from a rounding error.
    assert _self_align(g, w["dev"]) > 0.3


@pytest.mark.parametrize("mode", ["perp", "full"])
def test_debias_removes_the_self_alignment(mode):
    """After debiasing, the step must carry no systematic pull toward the sampled action."""
    w = _world(10)
    before = _self_align(*(_pipeline(w, mode, debias=False)[0], w["dev"]))
    align = _self_align(_pipeline(w, mode, debias=True)[0], w["dev"])
    # Null sd of a mean cosine over N rows is about 0.41/sqrt(N); at N=4096 that is 0.0064,
    # so anything under ~0.01 is indistinguishable from no bias at this sample size.
    assert abs(align) < 0.01, f"residual self-alignment {align:+.4f}"
    assert abs(align) < 0.05 * abs(before), f"{align:+.4f} vs {before:+.4f}"


def test_debias_is_exact_when_the_residual_is_pure_contamination():
    """With no independent surprise, debiasing must annihilate the direction entirely."""
    w = _world(11, surprise=0.0)
    g, u, _ = _pipeline(w, "full", debias=True)
    assert u.abs().max() < 1e-3
    assert g.abs().max() < 1e-3


def test_debias_preserves_signal_orthogonal_to_the_action():
    """It must remove the action's own effect and NOTHING else."""
    clean = _world(12, contaminate=False)
    dirty = _world(12, contaminate=True)  # identical except for the contamination
    g_clean, _, _ = _pipeline(clean, "full", debias=False)
    g_dirty, _, _ = _pipeline(dirty, "full", debias=True)
    # res_std differs slightly between the two (the contamination adds variance), so this
    # is a direction test, not an equality test.
    cos = torch.nn.functional.cosine_similarity(g_clean, g_dirty, dim=-1).mean()
    assert float(cos) > 0.95


def test_shuffle_control_cannot_detect_the_bias():
    """Documents WHY the debias is needed rather than left to the control arm.

    Permuting g across the batch drives self-alignment to zero whether or not the bias is
    present, so treatment-vs-shuffle would attribute a self-imitation effect to the vector.
    """
    w = _world(13)
    g, _, _ = _pipeline(w, "perp", debias=False)
    perm = torch.randperm(g.shape[0], device=DEV)
    assert _self_align(g, w["dev"]) > 0.02
    assert abs(_self_align(g[perm], w["dev"])) < 0.01


# ------------------------------------------------------------ claim 2


@pytest.mark.parametrize("mode", ["perp", "full"])
def test_non_state_blocks_cannot_drive_the_mean(mode):
    """Perturbing the a / a*a / constant coordinates arbitrarily must change nothing.

    Stated as an invariance rather than as "zero the state block and expect zero": the
    latter drives res_std to the floor and tests the guard rather than the intent.
    """
    w = _world(14)
    base, _, _ = _pipeline(w, mode)
    w2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in w.items()}
    w2["res"][:, A0:] += 50.0 * torch.randn_like(w2["res"][:, A0:])
    perturbed, _, _ = _pipeline(w2, mode)
    assert torch.allclose(base, perturbed, atol=1e-5)


def test_input_is_not_mutated():
    """Aliasing sf_residual would corrupt the critic target computed from it."""
    w = _world(15)
    before = w["res"].clone()
    _pipeline(w, "perp")
    assert torch.equal(w["res"], before)


def test_debias_keeps_the_non_state_blocks_zero():
    w = _world(16)
    _, u, _ = _pipeline(w, "full")
    assert u[:, A0:].abs().max() == 0.0


# ------------------------------------------------------------ claim 3


def test_perp_strips_both_reward_directions():
    w = _world(17)
    _, u, res_std = _pipeline(w, "perp")
    for vec in ((w["w_r"] * w["asf_std"]).clone(), (w["w_r"] * res_std).clone()):
        vec[A0:] = 0.0
        proj = u @ (vec / vec.norm())
        assert proj.abs().max() < 1e-3, f"surviving component {proj.abs().max():.2e}"


def test_the_two_reward_directions_are_not_parallel():
    """If they were, the 2-dim projection would be gratuitous. They are not."""
    w = _world(18)
    _, _, res_std = _pipeline(w, "full")
    b = v35.vector_pathwise_reward_basis(w["w_r"], w["asf_std"], res_std, A0)
    assert b.shape[0] == 2
    assert abs(float(b[0] @ b[1])) < 1e-5          # orthonormal
    assert torch.allclose(b.norm(dim=-1), torch.ones(2, device=DEV), atol=1e-5)
    assert b[:, A0:].abs().max() == 0.0            # confined to the state blocks


def test_full_mode_retains_the_reward_direction():
    """Control on the perp test: without projection the component is emphatically there."""
    w = _world(17)
    _, u, _ = _pipeline(w, "full")
    vec = (w["w_r"] * w["asf_std"]).clone()
    vec[A0:] = 0.0
    assert float((u @ (vec / vec.norm())).abs().mean()) > 0.05


def test_reward_basis_degenerates_gracefully():
    """Uniform explained variance makes the two directions coincide; must not divide by 0."""
    w = _world(19)
    res_std = w["asf_std"] * 0.5  # exactly parallel => rank 1
    b = v35.vector_pathwise_reward_basis(w["w_r"], w["asf_std"], res_std, A0)
    assert b.shape[0] == 1
    assert abs(float(b[0].norm()) - 1.0) < 1e-5


# ------------------------------------------------------------ claim 4


@pytest.mark.parametrize("k", [0, 5, EMB + 3])
def test_pipeline_is_invariant_to_rescaling_a_phi_coordinate(k):
    """Rescale phi_k by c: sf_residual_k, asf_std_k and res_std_k all scale by c, while
    psi_std -- and therefore the head's Jacobian -- is unchanged."""
    w = _world(20)
    base, _, _ = _pipeline(w, "perp")
    c = 7.5
    w2 = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in w.items()}
    w2["res"][:, k] *= c
    w2["asf_std"][k] *= c
    w2["w_r"][k] /= c  # so that w_r . phi is still the same reward
    scaled, _, _ = _pipeline(w2, "perp")
    assert torch.allclose(base, scaled, rtol=1e-3, atol=1e-5)


# ------------------------------------------------------------ claim 5


def test_head_jacobian_matches_autograd():
    torch.manual_seed(0)
    head = v35.ActionSuccessorHead(64, SF_DIM, ACT).to(DEV)
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.1)
    feat = torch.randn(8, 64, device=DEV)
    act = torch.randn(8, ACT, device=DEV, requires_grad=True)
    out = head(feat, act)
    lin, quad = head.jacobian_coeffs(feat)
    analytic = lin + 2.0 * quad * act.detach().unsqueeze(1)
    for k in range(0, SF_DIM, 13):
        g = torch.autograd.grad(out[:, k].sum(), act, retain_graph=True)[0]
        assert torch.allclose(g, analytic[:, k, :], atol=1e-5)


def test_contraction_matches_an_explicit_reference_loop():
    """Guards the einsum against a transposed-index bug, which would be silent."""
    w = _world(21, n=8)
    got, u, _ = _pipeline(w, "full")
    want = torch.zeros_like(got)
    for n in range(8):
        for j in range(ACT):
            want[n, j] = GAMMA * sum(u[n, k] * w["jac"][n, k, j] for k in range(SF_DIM))
    assert torch.allclose(got, want, atol=1e-4)


# ------------------------------------------------------------ claim 6


def _target(w, g, step, lo=-1.0, hi=10.0):
    low = torch.full((ACT,), lo, device=DEV)
    high = torch.full((ACT,), abs(hi), device=DEV)
    if hi > 5:  # bounds deliberately inactive
        low = torch.full((ACT,), -10.0, device=DEV)
    return v35.vector_pathwise_target(
        w["means"], g, torch.ones(1, ACT, device=DEV), step, low, high
    )


def test_target_respects_action_bounds():
    w = _world(22)
    g, _, _ = _pipeline(w, "full")
    low = torch.full((ACT,), -1.0, device=DEV)
    high = torch.full((ACT,), 1.0, device=DEV)
    tgt = v35.vector_pathwise_target(
        w["means"], g, torch.ones(1, ACT, device=DEV), 10.0, low, high
    )
    assert (tgt >= low - 1e-6).all() and (tgt <= high + 1e-6).all()


def test_no_row_exceeds_the_declared_step():
    """The per-row clamp: without it an outlier row is bounded only by the action box."""
    w = _world(23)
    g, _, _ = _pipeline(w, "full")
    g[0] *= 500.0  # a heavy-tailed outlier
    step = 0.25
    tgt = _target(w, g, step)
    assert float((tgt - w["means"]).abs().max()) <= step + 1e-5


def test_typical_displacement_matches_the_declared_step():
    w = _world(24)
    g, _, _ = _pipeline(w, "full")
    step = 0.25
    rms = (_target(w, g, step) - w["means"]).pow(2).sum(-1).mean().sqrt()
    assert 0.5 * step < float(rms) <= step + 1e-5


def test_zero_step_is_a_pure_anchor():
    """The `anchor` control arm leaves the target exactly at the current mean."""
    w = _world(25)
    g, _, _ = _pipeline(w, "full")
    assert torch.allclose(_target(w, g, 0.0), w["means"], atol=1e-7)


def test_zero_direction_is_a_pure_anchor():
    """As does a zeroed g, which is how the anchor control is actually implemented."""
    w = _world(26)
    g, _, _ = _pipeline(w, "full")
    assert torch.allclose(_target(w, torch.zeros_like(g), 0.25), w["means"], atol=1e-7)

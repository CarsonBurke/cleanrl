import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load():
    path = ROOT / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v31_anchored_minvariance.py"
    spec = importlib.util.spec_from_file_location("sf_vlam_v31_anchored_minvariance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()

GAMMA = 0.99
LAM = 0.95


def _fit(regressor, psi, trace, targets, epochs=60, lr=3e-3, seed=0, use_covector=True):
    optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
    return MODULE.fit_anchored_innovation_regressor(
        regressor,
        optimizer,
        psi,
        trace,
        targets,
        epochs,
        512,
        10.0,
        torch.Generator().manual_seed(seed),
        use_covector=use_covector,
    )


def _reference_scalar_gae(rewards, values, next_values, terminations, boundaries, valids):
    """cleanrl's scalar GAE, written out independently of the vector recursion."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        boot = (1.0 - terminations[t]) * valids[t]
        cont = 1.0 - boundaries[t]
        delta = rewards[t] + GAMMA * boot * next_values[t] - values[t]
        running = delta + GAMMA * LAM * cont * running
        advantages[t] = running
    return advantages


def test_reward_coordinate_innovation_is_exactly_the_scalar_gae():
    """The load-bearing claim of the anchor.

    With phi_0 = r, coordinate 0 of the vector TD(lambda) residual must be the scalar GAE
    advantage BIT FOR BIT -- not an approximation of it. Includes a truncation (term=0,
    valid=1, boundary=1) and a termination, the two cases where the bootstrap and trace
    masks disagree, because those are the only places a vector recursion can silently
    diverge from the scalar one.
    """
    torch.manual_seed(0)
    steps, envs, emb = 12, 3, 5
    rewards = torch.randn(steps, envs)
    embeddings = torch.randn(steps, envs, emb)
    phi = torch.cat([rewards.unsqueeze(-1), (1.0 - GAMMA) * embeddings], dim=-1)
    psi_cur = torch.randn(steps, envs, emb + 1)
    psi_next = torch.randn(steps, envs, emb + 1)

    terminations = torch.zeros(steps, envs)
    boundaries = torch.zeros(steps, envs)
    valids = torch.ones(steps, envs)
    terminations[4, 0] = 1.0  # a real termination
    boundaries[4, 0] = 1.0
    boundaries[7, 1] = 1.0    # a truncation: bootstrap through it, cut the trace
    valids[9, 2] = 0.0

    residual = MODULE.successor_lambda_residual(
        phi,
        psi_cur,
        psi_next,
        terminations,
        boundaries,
        valids,
        GAMMA,
        torch.full((emb + 1,), LAM),
    )
    reference = _reference_scalar_gae(
        rewards, psi_cur[..., 0], psi_next[..., 0], terminations, boundaries, valids
    )
    torch.testing.assert_close(residual[..., 0], reference, atol=0.0, rtol=0.0)


def test_folded_covector_reproduces_the_anchor_exactly():
    """The anchor survives being folded into the standardized contraction.

    The training loop forms the credit as ONE dot product against E / s, with the anchor
    encoded as s_0 in coordinate 0. That must contract back to exactly E_0 for any
    standardizer, and the correction must contribute only through coordinates 1..d-1.
    """
    torch.manual_seed(1)
    steps, envs, dim = 6, 4, 7
    trace = torch.randn(steps, envs, dim)
    readout_std = torch.rand(dim) + 0.5
    correction = torch.randn(steps, envs, dim - 1)

    direction = torch.zeros(steps, envs, dim)
    direction[..., 0] = readout_std[0]
    direction[..., 1:] = correction
    standardized_trace = trace / readout_std
    credit = (direction * standardized_trace).sum(-1)

    torch.testing.assert_close(credit - trace[..., 0], (correction * standardized_trace[..., 1:]).sum(-1))
    # With no correction the credit is bit-identically the scalar GAE.
    bare = torch.zeros_like(direction)
    bare[..., 0] = readout_std[0]
    torch.testing.assert_close((bare * standardized_trace).sum(-1), trace[..., 0])


def test_covector_head_is_negligible_at_init_so_the_floor_is_the_scalar_gae():
    """std=0.01 output init: iteration 1's correction must be a small perturbation.

    This is what makes the 9,905 scalar-GAE control the FLOOR of this variant rather than
    something it has to relearn from a random covector. At the deployed shape
    (sf_dim=33, hidden=128) the measured ratio is ~0.07 -- the correction adds well under
    1% of the anchor's variance, so iteration 1 is scalar GAE for all practical purposes.
    The bound is loose enough not to be a seed anecdote and tight enough to catch an init
    change that would put a random covector in front of the actor.
    """
    torch.manual_seed(2)
    dim = 33
    regressor = MODULE.AnchoredInnovationRegressor(dim, 128, dim - 1)
    psi = torch.randn(512, dim)
    trace = torch.randn(512, dim)
    with torch.no_grad():
        correction = (regressor.covector(psi) * trace[:, 1:]).sum(-1)
    anchor = trace[:, 0]
    ratio = float(correction.square().mean().sqrt()) / float(
        anchor.square().mean().sqrt()
    )
    assert ratio < 0.12, ratio


def test_covector_spans_only_the_embedding_block():
    """Coordinate 0 is withheld by construction, not by convention.

    If u could read the reward innovation it would just re-derive a shrunk copy of the
    anchor, and readout/innovation_delta_explained_variance would stop answering the one
    question this design asks: does the LATENT innovation carry advantage the reward does
    not? A dimension mismatch is the only thing that makes that structural.
    """
    dim = 33
    regressor = MODULE.AnchoredInnovationRegressor(dim, 64, dim - 1)
    covector = regressor.covector(torch.randn(16, dim))
    assert covector.shape == (16, dim - 1)
    _, from_forward = regressor(torch.randn(16, dim))
    assert from_forward.shape == (16, dim - 1)


def test_covector_is_consistent_between_the_two_entry_points():
    torch.manual_seed(3)
    regressor = MODULE.AnchoredInnovationRegressor(8, 32, 7)
    psi = torch.randn(64, 8)
    _, covector = regressor(psi)
    torch.testing.assert_close(covector, regressor.covector(psi))


def test_recovers_the_innovation_covector_through_a_nonlinear_state_baseline():
    """G - E_0 = b(psi) + w.E[1:] with a NONLINEAR state baseline.

    The point of the partially-linear fit is that b absorbs the state so u is identified
    only by within-state covariation between the innovation and the return. A linear
    baseline would make this pass trivially; the tanh/square baseline is what makes it a
    real test of the partialling-out.
    """
    torch.manual_seed(4)
    dim = 7
    weight = torch.tensor([1.0, -2.0, 0.5, 0.0, 1.5, -0.75])
    psi = torch.randn(4096, dim)
    trace = torch.randn(4096, dim - 1)
    baseline = 3.0 * torch.tanh(psi[:, 0]) - psi[:, 1].square()
    targets = baseline + trace @ weight

    regressor = MODULE.AnchoredInnovationRegressor(dim, 64, dim - 1)
    _fit(regressor, psi, trace, targets)
    covector = regressor.covector(psi)
    cosine = torch.nn.functional.cosine_similarity(covector.mean(0), weight, dim=0)
    assert float(cosine) > 0.98, float(cosine)
    torch.testing.assert_close(covector.mean(0), weight, atol=0.2, rtol=0.2)


def test_correction_scale_is_set_by_the_residual_not_by_the_return_level():
    """THE reason this file exists -- the defect that killed v29.

    v29 built the correction as residual_std * grad_psi g: a STATE-level regression,
    differentiated, then rescaled by a RETURN-scale constant, then added to an
    ADVANTAGE-scale anchor. Nothing in it tied the correction's magnitude to the thing it
    corrects, and in the run it hit 274x the anchor while explaining under 0.18 of the
    residual.

    Here the state level is made enormous (40x tanh) next to a small true innovation term.
    Least squares must set the correction's scale from the innovation term ALONE, because
    b eats the level. So the fitted credit must land near the true innovation RMS and be a
    small fraction of the return's own RMS -- the opposite of what a rescaled gradient did.
    """
    torch.manual_seed(5)
    dim = 7
    weight = torch.tensor([0.4, -0.3, 0.0, 0.2, -0.5, 0.1])
    psi = torch.randn(8192, dim)
    # E is NOT independent of the state in the loop: E[E_t | s_t] is the critic's TD bias,
    # which is state-varying. So give the innovation a state-driven mean. If the level and
    # the innovation were orthogonal in population, u could not absorb the level no matter
    # how badly b fit, and this test would assume away the entire confound it exists for.
    state_bias = torch.stack(
        [torch.tanh(psi[:, 0]), psi[:, 2], torch.zeros_like(psi[:, 0])], dim=-1
    )
    trace = torch.randn(8192, dim - 1)
    trace[:, :3] = trace[:, :3] + state_bias
    innovation_term = trace @ weight
    level = 40.0 * torch.tanh(psi[:, 0]) + 15.0 * psi[:, 2]
    targets = level + innovation_term

    truth_rms = float(innovation_term.square().mean().sqrt())
    target_rms = float(targets.square().mean().sqrt())
    assert truth_rms < 0.05 * target_rms  # the level really does dominate the target

    regressor = MODULE.AnchoredInnovationRegressor(dim, 64, dim - 1)
    _fit(regressor, psi, trace, targets, epochs=80)
    with torch.no_grad():
        credit = (regressor.covector(psi) * trace).sum(-1)
    credit_rms = float(credit.square().mean().sqrt())
    assert 0.5 * truth_rms < credit_rms < 2.0 * truth_rms, (credit_rms, truth_rms)
    assert credit_rms < 0.15 * target_rms, (credit_rms, target_rms)
    # And u is the true covector, not a level-absorbing surrogate that happens to be small.
    cosine = torch.nn.functional.cosine_similarity(
        regressor.covector(psi).mean(0), weight, dim=0
    )
    assert float(cosine) > 0.9, float(cosine)


def test_correction_collapses_to_zero_when_the_innovation_is_noise():
    """The property a unit-norm ordinal covector cannot have, at any scale.

    When the innovation explains nothing, least squares sends u to zero, so the credit
    collapses back to the exact scalar GAE -- the control that currently wins. v23's
    ordinal field normalized to unit length instead, so it always injected a full-strength
    direction whether or not one existed.
    """
    torch.manual_seed(6)
    dim = 7
    psi = torch.randn(4096, dim)
    trace = torch.randn(4096, dim - 1)
    targets = 3.0 * torch.tanh(psi[:, 0]) - psi[:, 1].square()  # no innovation term

    regressor = MODULE.AnchoredInnovationRegressor(dim, 64, dim - 1)
    _fit(regressor, psi, trace, targets)
    with torch.no_grad():
        baseline, covector = regressor(psi)
        credit = (covector * trace).sum(-1)
    assert float(credit.square().mean().sqrt()) < 0.15 * float(
        targets.square().mean().sqrt()
    )
    baseline_ev = MODULE.explained_variance_1d(baseline, targets)
    full_ev = MODULE.explained_variance_1d(baseline + credit, targets)
    assert full_ev - baseline_ev < 0.05, (baseline_ev, full_ev)


def test_innovation_delta_explained_variance_fires_when_it_should():
    """The logged falsifier must move in BOTH directions, or it certifies nothing.

    Paired with the test above, which pins it near zero on a pure-noise innovation.
    """
    torch.manual_seed(7)
    dim = 7
    weight = torch.tensor([1.0, -1.0, 0.0, 2.0, 0.5, 0.0])
    psi = torch.randn(4096, dim)
    trace = torch.randn(4096, dim - 1)
    targets = torch.tanh(psi[:, 0]) + trace @ weight

    regressor = MODULE.AnchoredInnovationRegressor(dim, 64, dim - 1)
    _fit(regressor, psi, trace, targets)
    with torch.no_grad():
        baseline, covector = regressor(psi)
    baseline_ev = MODULE.explained_variance_1d(baseline, targets)
    full_ev = MODULE.explained_variance_1d(
        baseline + (covector * trace).sum(-1), targets
    )
    assert full_ev - baseline_ev > 0.5, (baseline_ev, full_ev)


def test_a_constant_psi_fit_blows_the_correction_up_which_is_why_it_is_skipped():
    """Iteration 1: critic_head is zero-init and bias-free, so every fit row of psi is
    identical, AND psi_0 is 0 so the target keeps its full return-scale level.

    With psi constant, b is a single scalar: it can absorb the target's MEAN but not its
    cross-sectional level. Least squares then hands that level to u, because the
    innovation is state-correlated in the loop. The result is a correction many times the
    anchor it is supposed to correct -- and policy_credit_rms is divide-only, so it does
    not clip, it DILUTES the exact anchor down to a few percent of the PG direction. That
    is the v29 pathology reappearing at iteration 2, which is why the fit is skipped.

    Asserted here rather than argued: the same data with a varying psi gives a correction
    at the true scale, the constant psi gives one several times larger, and the guard
    predicate separates the two cases.
    """
    torch.manual_seed(8)
    dim = 7
    weight = torch.tensor([0.4, -0.3, 0.0, 0.2, -0.5, 0.1])
    psi = torch.randn(4096, dim)
    state_bias = torch.stack(
        [torch.tanh(psi[:, 0]), psi[:, 2], torch.zeros_like(psi[:, 0])], dim=-1
    )
    trace = torch.randn(4096, dim - 1)
    trace[:, :3] = trace[:, :3] + state_bias
    truth_rms = float((trace @ weight).square().mean().sqrt())
    targets = 40.0 * torch.tanh(psi[:, 0]) + 15.0 * psi[:, 2] + trace @ weight

    constant_psi = torch.zeros_like(psi)  # what iteration 1 actually presents
    degenerate = MODULE.AnchoredInnovationRegressor(dim, 64, dim - 1)
    _fit(degenerate, constant_psi, trace, targets, epochs=80)
    with torch.no_grad():
        blown_rms = float(
            (degenerate.covector(constant_psi) * trace).sum(-1).square().mean().sqrt()
        )
    assert blown_rms > 3.0 * truth_rms, (blown_rms, truth_rms)

    # The guard predicate: fires on the degenerate input, silent on the real one.
    assert float(constant_psi.std(0).max()) < 1e-6
    assert float(psi.std(0).max()) > 1e-6


def test_fit_is_a_no_op_on_an_empty_gate():
    regressor = MODULE.AnchoredInnovationRegressor(7, 16, 6)
    before = [p.clone() for p in regressor.parameters()]
    loss, grad_norm, steps = _fit(
        regressor, torch.zeros(0, 7), torch.zeros(0, 6), torch.zeros(0), epochs=5
    )
    assert steps == 0 and loss != loss and grad_norm != grad_norm  # NaN, not a crash
    for parameter, original in zip(regressor.parameters(), before):
        torch.testing.assert_close(parameter, original)


def test_fit_gate_falls_back_instead_of_deadlocking():
    """An empty mc_window gate must never leave the fit set empty.

    Mirrors the fallback chain in the training loop: mc_mask -> mc_complete -> everything.
    """
    complete = torch.zeros(64, dtype=torch.bool)
    complete[:10] = True

    def select(mc_mask, mc_complete):
        gate = mc_mask
        if not bool(gate.any()):
            gate = mc_complete
        if not bool(gate.any()):
            gate = torch.ones_like(mc_mask)
        return gate

    populated = torch.zeros(64, dtype=torch.bool)
    populated[20:40] = True
    assert int(select(populated, complete).sum()) == 20
    empty = torch.zeros(64, dtype=torch.bool)
    assert int(select(empty, complete).sum()) == 10
    assert int(select(empty, empty).sum()) == 64


def test_critic_coordinate_weights_reserve_the_reward_share():
    for dim in (33, 9):
        for coefficient in (0.5, 0.25):
            weights = torch.full((dim,), (1.0 - coefficient) / (dim - 1))
            weights[0] = coefficient
            assert abs(float(weights.sum()) - 1.0) < 1e-6
            assert float(weights[0]) == coefficient
            assert float(weights[1:].std()) < 1e-8


def test_return_residual_correlation_uses_the_cross_env_baseline():
    """The falsifier must see ADVANTAGE, not the level of the return."""
    torch.manual_seed(9)
    returns = torch.randn(20, 8)
    residual = returns - returns.mean(1, keepdim=True)
    valid = torch.ones(20, 8, dtype=torch.bool)
    assert abs(MODULE.credit_return_correlation(residual, returns, valid) - 1.0) < 1e-5
    shifted = returns + torch.arange(20.0).unsqueeze(1)
    assert abs(MODULE.credit_return_correlation(residual, shifted, valid) - 1.0) < 1e-5
    assert abs(
        MODULE.credit_return_correlation(17.0 * residual, returns, valid) - 1.0
    ) < 1e-5


def test_baseline_only_control_withholds_the_innovation_term():
    """use_covector=False must fit b ALONE, or the incremental R^2 is not incremental.

    Read out of the joint model instead, b's R^2 falls exactly when u wrongly absorbs
    state level -- so readout_ev - baseline_ev would peak in the failure case. The control
    has to be a model that never saw the innovation.
    """
    torch.manual_seed(10)
    dim = 7
    weight = torch.tensor([1.0, -1.0, 0.0, 2.0, 0.5, 0.0])
    psi = torch.randn(4096, dim)
    trace = torch.randn(4096, dim - 1)
    targets = torch.tanh(psi[:, 0]) + trace @ weight

    control = MODULE.AnchoredInnovationRegressor(dim, 64, dim - 1)
    before = control.covector_head.weight.clone()
    _fit(control, psi, trace, targets, use_covector=False)
    # u got no gradient at all: it is untouched at its ~0 init.
    torch.testing.assert_close(control.covector_head.weight, before)

    joint = MODULE.AnchoredInnovationRegressor(dim, 64, dim - 1)
    _fit(joint, psi, trace, targets)
    with torch.no_grad():
        control_ev = MODULE.explained_variance_1d(control(psi)[0], targets)
        joint_baseline, joint_covector = joint(psi)
        joint_ev = MODULE.explained_variance_1d(
            joint_baseline + (joint_covector * trace).sum(-1), targets
        )
    # The control explains only the state term; the joint model explains that plus the
    # innovation, so the honest delta is large and positive here.
    assert joint_ev - control_ev > 0.5, (control_ev, joint_ev)


def test_regressor_inputs_are_detached_at_the_module_boundary():
    """The regressor must never be a path from the return target back into the trunk.

    Every call site is inside torch.no_grad() today, so this is defence in depth against a
    later edit that moves one out.
    """
    dim = 7
    regressor = MODULE.AnchoredInnovationRegressor(dim, 16, dim - 1)
    psi = torch.randn(32, dim, requires_grad=True)
    baseline, covector = regressor(psi)
    (baseline.sum() + covector.sum()).backward()
    assert psi.grad is None

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
GUARD = 20.0


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V8 = _load("tpo_intra_beta_noadvnorm_v8", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_noadvnorm_v8.py")
V7 = _load("tpo_intra_beta_klctrl_v7_ref", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_klctrl_v7.py")

_ARGS = V8.Args()
SOLVER = dict(eta_min=_ARGS.eta_min, eta_max=_ARGS.eta_max, iters=_ARGS.eta_solver_iters)
CTRL = dict(
    gain=_ARGS.kl_budget_gain,
    ratio_clip=_ARGS.kl_ratio_clip,
    bounds=_ARGS.kl_budget_bounds,
)


def _advantages(n=4096, seed=0, mean=0.0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g, dtype=torch.float64) * scale + mean)


# --- the theorem the whole ablation rests on ---------------------------------


def test_the_dual_solve_makes_the_std_division_exactly_a_no_op():
    """KL(c*u, c*eta) = KL(u, eta), so rescaling utilities cannot change the target.

    This is why "no advantage normalisation" reduces to "no centring": the solver
    absorbs any positive rescaling by moving eta the same factor, leaving the
    target weights exp(u/eta) pointwise unchanged.
    """
    u = _advantages(seed=4)
    u = u - u.mean()
    eta_ref, kl_ref = V8.tpo_solve_eta(u, 0.08, **SOLVER)
    for c in (0.05, 0.5, 3.0, 15.0):
        assert SOLVER["eta_min"] < c * eta_ref < SOLVER["eta_max"], (c, eta_ref)
        eta_scaled, kl_scaled = V8.tpo_solve_eta(c * u, 0.08, **SOLVER)
        assert math.isclose(eta_scaled, c * eta_ref, rel_tol=1e-6), (c, eta_ref, eta_scaled)
        assert math.isclose(kl_scaled, kl_ref, rel_tol=1e-9), (c, kl_ref, kl_scaled)
        torch.testing.assert_close(
            (c * u / eta_scaled).exp(), (u / eta_ref).exp(), rtol=1e-6, atol=1e-9
        )


def test_the_no_op_holds_only_while_the_bracket_still_contains_the_solution():
    """Why v8 keeps the (provably inert) division by std instead of dropping it.

    The eta bracket and the eta floor are absolute numbers. Rescaling the utilities
    moves the solution proportionally but leaves the bracket where it is, so a large
    enough rescaling walks the solution out of the bracket and the solver saturates
    -- at which point the equivalence above fails and the trust region is silently
    wrong. Keeping the division pins the units, so v8's bracket, floor and eta all
    mean exactly what they meant in v7.
    """
    u = _advantages(seed=9)
    u = u - u.mean()
    eta_ref, _ = V8.tpo_solve_eta(u, 0.08, **SOLVER)
    for c, bound in ((0.01, SOLVER["eta_min"]), (100.0, SOLVER["eta_max"])):
        assert not SOLVER["eta_min"] < c * eta_ref < SOLVER["eta_max"]
        eta_scaled, _ = V8.tpo_solve_eta(c * u, 0.08, **SOLVER)
        assert math.isclose(eta_scaled, bound, rel_tol=1e-6), (c, eta_scaled)
        assert not math.isclose(eta_scaled, c * eta_ref, rel_tol=1e-3)


def test_a_shift_is_the_one_thing_the_solver_does_not_absorb():
    """KL is shift-invariant, so eta does not move -- but the target ratios all do.

    The complement of the test above: this is the entire behavioural content of
    dropping the mean subtraction.
    """
    u = _advantages(seed=5)
    u = u - u.mean()
    eta_ref, kl_ref = V8.tpo_solve_eta(u, 0.08, **SOLVER)
    for shift in (-1.5, 0.4, 2.0):
        eta_shift, kl_shift = V8.tpo_solve_eta(u + shift, 0.08, **SOLVER)
        assert math.isclose(eta_shift, eta_ref, rel_tol=1e-9), (shift, eta_ref, eta_shift)
        assert math.isclose(kl_shift, kl_ref, rel_tol=1e-9)
        # ...and every target ratio is scaled by exactly exp(shift / eta).
        got = ((u + shift) / eta_shift).exp() / (u / eta_ref).exp()
        torch.testing.assert_close(
            got, torch.full_like(got, math.exp(shift / eta_ref)), rtol=1e-6, atol=0
        )
        assert not math.isclose(math.exp(shift / eta_ref), 1.0, rel_tol=1e-3), "shift must bite"


# --- the utility transform ----------------------------------------------------


def test_utility_keeps_the_level_and_normalises_the_spread():
    advantages = _advantages(mean=0.7, scale=4.0, seed=1)
    u = V8.tpo_utility(advantages, utility_clip=None)
    expected_mean = float(advantages.mean() / advantages.std(unbiased=False))
    assert math.isclose(float(u.mean()), expected_mean, rel_tol=1e-9)
    assert math.isclose(float(u.std(unbiased=False)), 1.0, rel_tol=1e-9)
    # The level is the point: it must be materially non-zero for this batch.
    assert abs(expected_mean) > 0.1


def test_utility_differs_from_v7_by_exactly_the_batch_mean():
    advantages = _advantages(mean=-1.3, scale=2.5, seed=2)
    delta = V8.tpo_utility(advantages, None) - V7.tpo_utility(advantages, None)
    expected = float(advantages.mean() / advantages.std(unbiased=False))
    torch.testing.assert_close(delta, torch.full_like(delta, expected), rtol=1e-9, atol=1e-9)


def test_utility_is_identical_to_v7_on_an_already_centred_batch():
    """No centring is a no-op when there is nothing to centre, clip included."""
    advantages = _advantages(seed=3, scale=3.0)
    advantages = advantages - advantages.mean()
    for clip in (None, 3.0, 0.5):
        torch.testing.assert_close(
            V8.tpo_utility(advantages, clip), V7.tpo_utility(advantages, clip), rtol=1e-6, atol=1e-7
        )


def test_a_spreadless_batch_is_left_alone_rather_than_pushed_by_its_level():
    """The level of a zero-variance batch is indistinguishable from critic bias."""
    utility = V8.tpo_utility(torch.full((256,), 2.5), utility_clip=3.0)
    torch.testing.assert_close(utility, torch.zeros(256))
    eta, kl = V8.tpo_solve_eta(utility, 0.08, **SOLVER)
    assert eta == SOLVER["eta_max"] and kl == 0.0
    logratio = torch.zeros(256, requires_grad=True)
    V8.tpo_intra_loss(logratio, utility, eta, GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad, torch.zeros(256))


def test_the_clip_binds_asymmetrically_once_the_batch_is_not_centred():
    """A positive-mean batch loses more of its upper tail than its lower one."""
    advantages = _advantages(mean=1.0, seed=6)
    unclipped = V8.tpo_utility(advantages, None)
    clipped = V8.tpo_utility(advantages, 3.0)
    hit_high = int(((unclipped > 3.0)).sum())
    hit_low = int(((unclipped < -3.0)).sum())
    assert hit_high > hit_low, (hit_high, hit_low)
    assert float(clipped.max()) <= 3.0 and float(clipped.min()) >= -3.0
    # The level survives the clip, which is what keeps the ablation meaningful.
    assert float(clipped.mean()) > 0.5


# --- the uniform push, and its containment ------------------------------------


def test_a_positive_mean_batch_asks_for_every_density_to_rise():
    advantages = _advantages(mean=1.2, seed=7)
    utility = V8.tpo_utility(advantages, 3.0)
    eta, _ = V8.tpo_solve_eta(utility, 0.08, **SOLVER)
    target_ratios = (utility / eta).exp()
    assert float(target_ratios.mean()) > 1.0
    # v7 cannot express this: whitening forces the mean utility to zero.
    v7_ratios = (V7.tpo_utility(advantages, 3.0) / eta).exp()
    assert float(target_ratios.mean()) > float(v7_ratios.mean())
    # The gradient at r = 1 is (1 - w), so a majority of samples are pushed up.
    assert float((target_ratios > 1.0).double().mean()) > 0.5


def test_the_eta_floor_bounds_the_uniform_push_but_the_bound_is_not_containment():
    """The floor caps the push at max_target_ratio -- which is 10x, not safety.

    This is a characterisation test, not a guarantee: it records that the stated
    containment argument (|mean(u)| <= max|u| <= utility_clip) is true and useless,
    because a uniform 10x density push *is* the entropy collapse it is supposed to
    exclude. The real containment is the realized-KL controller.
    """
    floor = V8.eta_floor(3.0, 10.0, 0.05)
    assert math.isclose(floor, 3.0 / math.log(10.0), rel_tol=1e-12)
    advantages = _advantages(mean=5.0, seed=8)  # ~45 sigma beyond anything measured
    utility = V8.tpo_utility(advantages, 3.0)
    eta, _ = V8.tpo_solve_eta(utility, 1.0, **{**SOLVER, "eta_min": floor})
    assert eta >= floor
    assert float((utility / eta).exp().max()) <= 10.0 + 1e-9
    # The bound holds and is nearly saturated: every action is pushed up ~10x.
    assert 9.0 < float((utility.mean() / eta).exp()) <= 10.0


def test_the_level_eats_the_clip_budget_and_saturates_the_ranking_signal():
    """The silent failure mode: at a large level the update loses its ranking.

    Unreachable in practice -- measured mean(A)/std(A) has std 0.111 and max 0.41
    over 400 simulated rollouts -- but it degrades gracefully into nonsense rather
    than erroring, so it is pinned here and surfaced by tpo/utility_std.
    """
    spreads = []
    for level in (0.0, 1.0, 3.0, 5.0):
        u = V8.tpo_utility(_advantages(mean=level, seed=20), 3.0)
        spreads += [float(u.std(unbiased=False))]
    assert spreads[0] > 0.99, spreads
    assert spreads == sorted(spreads, reverse=True), spreads
    assert spreads[2] < 0.7 and spreads[3] < 0.1, spreads
    # v7 is immune: centring means the level never consumes the clip budget.
    v7_spreads = [
        float(V7.tpo_utility(_advantages(mean=level, seed=20), 3.0).std(unbiased=False))
        for level in (0.0, 5.0)
    ]
    assert math.isclose(v7_spreads[0], v7_spreads[1], rel_tol=1e-12)


def test_v8_is_v7_plus_a_constant_only_while_the_clip_does_not_bind():
    """The clip is not shift-equivariant, so the clean relation has a domain.

    Left untested this would quietly become false exactly where the ablation is
    most active, since a level pushes samples over the clip on one side only.
    """
    advantages = _advantages(mean=0.8, scale=2.0, seed=21)
    expected = float(advantages.mean() / advantages.std(unbiased=False))
    unclipped = V8.tpo_utility(advantages, None) - V7.tpo_utility(advantages, None)
    torch.testing.assert_close(unclipped, torch.full_like(unclipped, expected), rtol=1e-9, atol=1e-9)
    clipped = V8.tpo_utility(advantages, 3.0) - V7.tpo_utility(advantages, 3.0)
    differing = int((~torch.isclose(clipped, torch.full_like(clipped, expected), atol=1e-6)).sum())
    assert differing > 0, "the clip must bind for this batch, or the test proves nothing"
    assert differing < advantages.numel() // 10, differing


def test_eta_floor_is_unchanged_from_v7():
    for clip, cap in ((3.0, 10.0), (3.0, 2.0), (1.0, 50.0), (None, 10.0)):
        assert V8.eta_floor(clip, cap, 0.05) == V7.eta_floor(clip, cap, 0.05)


# --- controller anti-windup ---------------------------------------------------


def test_controller_matches_v7_exactly_while_eta_is_unsaturated():
    for realized in (0.005, 0.03, 0.2):
        assert V8.kl_budget_update(0.08, realized, 0.03, **CTRL) == V7.kl_budget_update(
            0.08, realized, 0.03, **CTRL
        )


def test_a_floored_eta_refuses_the_demand_it_cannot_serve():
    """At the floor eta cannot go lower, so asking for more movement is windup."""
    budget = 0.08
    frozen = V8.kl_budget_update(budget, 0.001, 0.03, **CTRL, eta_at_floor=True)
    assert frozen == budget
    # v7 would have integrated upward here, which is the bug being fixed.
    assert V7.kl_budget_update(budget, 0.001, 0.03, **CTRL) > budget
    # The other direction is still actionable and must still act.
    assert V8.kl_budget_update(budget, 0.3, 0.03, **CTRL, eta_at_floor=True) < budget


def test_a_ceilinged_eta_refuses_the_opposite_demand():
    budget = 0.08
    assert V8.kl_budget_update(budget, 0.3, 0.03, **CTRL, eta_at_ceiling=True) == budget
    assert V8.kl_budget_update(budget, 0.001, 0.03, **CTRL, eta_at_ceiling=True) > budget


def test_anti_windup_does_not_fire_when_the_controller_is_disabled():
    assert V8.kl_budget_update(0.08, 0.5, None, **CTRL, eta_at_floor=True) == 0.08


def test_windup_is_actually_prevented_over_a_saturated_run():
    """Reproduces the v7 kl3 failure: 200 iterations pinned at the eta floor."""
    v7_budget = v8_budget = 0.08
    for _ in range(200):
        v7_budget = V7.kl_budget_update(v7_budget, 0.02, 0.03, **CTRL)
        v8_budget = V8.kl_budget_update(v8_budget, 0.02, 0.03, **CTRL, eta_at_floor=True)
    assert math.isclose(v7_budget, CTRL["bounds"][1]), v7_budget  # slammed into the bound
    assert v8_budget == 0.08


# --- everything else must be v7 verbatim --------------------------------------


def test_loss_solver_and_target_kl_are_identical_to_v7():
    torch.manual_seed(0)
    logratio = torch.randn(256) * 0.6
    utility = torch.randn(256) * 1.5
    for eta in (0.7, 2.0, 4.0):
        torch.testing.assert_close(
            V8.tpo_intra_loss(logratio, utility, eta, GUARD),
            V7.tpo_intra_loss(logratio, utility, eta, GUARD),
        )
        assert V8.tpo_target_kl(utility.double(), eta) == V7.tpo_target_kl(utility.double(), eta)
    assert V8.tpo_solve_eta(utility.double(), 0.08, **SOLVER) == V7.tpo_solve_eta(
        utility.double(), 0.08, **SOLVER
    )


def _envs_stub():
    import gymnasium as gym

    class Stub:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    return Stub()


def test_agent_is_bit_identical_to_v7_under_the_same_seed():
    obs = torch.randn(16, 5)
    torch.manual_seed(7)
    a8 = V8.Agent(_envs_stub())
    torch.manual_seed(7)
    a7 = V7.Agent(_envs_stub())
    p8 = dict(a8.named_parameters())
    p7 = dict(a7.named_parameters())
    assert p8.keys() == p7.keys() and len(p8) > 0
    for name in p8:
        torch.testing.assert_close(p8[name], p7[name])
    d8, d7 = a8._dist(obs), a7._dist(obs)
    torch.testing.assert_close(d8.concentration1, d7.concentration1)
    torch.testing.assert_close(d8.concentration0, d7.concentration0)
    torch.testing.assert_close(a8.get_value(obs), a7.get_value(obs))


def test_the_args_surface_is_v7_plus_only_the_anti_windup_switch():
    """Every default must be v7's, or the run is not a one-variable ablation."""
    added = set(V8.Args.__dataclass_fields__) - set(V7.Args.__dataclass_fields__)
    assert added == {"kl_anti_windup"}, added
    assert not set(V7.Args.__dataclass_fields__) - set(V8.Args.__dataclass_fields__)
    for name, field in V8.Args.__dataclass_fields__.items():
        if name in ("exp_name", "kl_anti_windup"):
            continue
        assert field.default == V7.Args.__dataclass_fields__[name].default, name
    # Off by default, so the controller behaves exactly as it did in the v7 kl3 run.
    assert V8.Args().kl_anti_windup is False

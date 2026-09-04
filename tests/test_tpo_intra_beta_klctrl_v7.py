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


V7 = _load("tpo_intra_beta_klctrl_v7", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_klctrl_v7.py")
V2 = _load("tpo_intra_beta_v2_ref", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_v2.py")

# Read the bracket from Args rather than restating it, so a changed default cannot
# silently cap the trust region in real runs while the tests keep passing.
DEFAULTS = V7.Args()
SOLVER = dict(eta_min=DEFAULTS.eta_min, eta_max=DEFAULTS.eta_max, iters=DEFAULTS.eta_solver_iters)
CTRL = dict(
    gain=DEFAULTS.kl_budget_gain,
    ratio_clip=DEFAULTS.kl_ratio_clip,
    bounds=DEFAULTS.kl_budget_bounds,
)


def _utilities(n=8192, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float64)


def _envs_stub():
    import gymnasium as gym

    class Stub:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    return Stub()


# --- KL(eta) and the dual, re-verified against an independent construction ----


def test_target_kl_equals_a_brute_force_discrete_kl():
    """Build q_i ∝ p_i exp(u_i/eta) by hand and sum q log(q/p) directly."""
    for n, eta in [(5, 0.3), (17, 1.0), (64, 2.0), (64, 7.0)]:
        u = _utilities(n=n, seed=n)
        p = torch.full((n,), 1.0 / n, dtype=torch.float64)
        unnormalised = p * (u / eta).exp()
        q = unnormalised / unnormalised.sum()
        brute = float((q * (q / p).log()).sum())
        assert math.isclose(brute, V7.tpo_target_kl(u, eta), rel_tol=1e-12), (n, eta)


def test_target_kl_is_strictly_decreasing_including_adversarial_batches():
    batches = {
        "gaussian": _utilities(1024),
        "one_outlier": torch.cat([_utilities(1023), torch.tensor([300.0], dtype=torch.float64)]),
        "two_valued": torch.tensor([-1.0, 1.0] * 256, dtype=torch.float64),
        "clip_saturated": torch.tensor([-3.0, 3.0] * 256, dtype=torch.float64),
        "bimodal": torch.cat([torch.zeros(900, dtype=torch.float64), torch.ones(124, dtype=torch.float64)]),
        "singleton": torch.tensor([1.7], dtype=torch.float64),
    }
    grid = np.exp(np.linspace(math.log(0.05), math.log(50.0), 400))
    for name, u in batches.items():
        kls = [V7.tpo_target_kl(u, float(e)) for e in grid]
        # Flat regions are legitimate; a real increase beyond float64 noise is not.
        worst = max(b - a for a, b in zip(kls, kls[1:]))
        assert worst < 1e-12, f"{name} breaks monotonicity by {worst:e}"


def test_solved_eta_is_the_stationary_point_of_the_convex_dual():
    u = _utilities()
    eps = 0.125
    eta, _ = V7.tpo_solve_eta(u, eps, **SOLVER)

    def dual(e):
        return e * eps + e * float(torch.logsumexp(u / e, 0) - math.log(u.numel()))

    h = 1e-5 * eta
    assert abs((dual(eta + h) - dual(eta - h)) / (2.0 * h)) < 1e-6
    assert dual(eta) < min(dual(eta * 0.8), dual(eta * 1.25))


def test_solved_eta_hits_the_requested_budget_and_the_returned_kl_matches_it():
    u = _utilities()
    for budget in (0.02, 0.0625, 0.125, 0.25):
        eta, kl = V7.tpo_solve_eta(u, budget, **SOLVER)
        assert SOLVER["eta_min"] < eta < SOLVER["eta_max"]
        assert math.isclose(kl, budget, rel_tol=1e-6)
        # The reported KL must be the KL *at the returned eta*, not at a neighbour.
        assert math.isclose(kl, V7.tpo_target_kl(u, eta), rel_tol=1e-12)


def test_gaussian_utilities_recover_the_equivalent_fixed_eta():
    u = _utilities(n=400_000, seed=5)
    u = (u - u.mean()) / u.std(unbiased=False)
    for eta_ref in (1.5, 2.0, 4.0):
        eta, _ = V7.tpo_solve_eta(u, 1.0 / (2.0 * eta_ref**2), **SOLVER)
        assert math.isclose(eta, eta_ref, rel_tol=0.03), (eta_ref, eta)


def test_solved_eta_moves_with_the_shape_of_the_utility_distribution():
    g = torch.Generator().manual_seed(11)
    gaussian = torch.randn(200_000, generator=g, dtype=torch.float64)
    skewed = torch.distributions.Exponential(1.0).sample((200_000,)).double()
    clipped = [V7.tpo_utility(raw, utility_clip=3.0).double() for raw in (gaussian, skewed)]
    etas, kls = zip(*[V7.tpo_solve_eta(u, 0.125, **SOLVER) for u in clipped])
    assert all(math.isclose(kl, 0.125, rel_tol=1e-6) for kl in kls)
    assert abs(etas[0] - etas[1]) / etas[0] > 0.05, etas
    fixed = [V7.tpo_target_kl(u, 2.0) for u in clipped]
    assert abs(fixed[0] - fixed[1]) / fixed[0] > 0.05, fixed


# --- the eta floor, which is the fix for the pooled-KL/per-sample-ratio gap ----


def test_eta_floor_caps_the_largest_demanded_probability_change():
    for clip, cap in [(3.0, 10.0), (3.0, 4.482), (2.0, 20.0), (5.0, 10.0)]:
        floor = V7.eta_floor(clip, cap, eta_min=0.05)
        assert math.isclose(math.exp(clip / floor), cap, rel_tol=1e-9), (clip, cap, floor)


def test_eta_floor_is_inactive_without_a_utility_clip_or_a_finite_cap():
    assert V7.eta_floor(None, 10.0, eta_min=0.05) == 0.05
    assert V7.eta_floor(3.0, float("inf"), eta_min=0.05) == 0.05
    # A cap of 1 or below would demand an infinite eta and is rejected.
    for bad in (1.0, 0.5, 0.0):
        try:
            V7.eta_floor(3.0, bad, eta_min=0.05)
        except ValueError:
            continue
        raise AssertionError(f"max_target_ratio={bad} was accepted")


def test_the_floor_actually_binds_on_the_heavy_tailed_case_it_exists_for():
    """A single huge advantage compresses the whitened bulk and pulls eta down."""
    g = torch.Generator().manual_seed(4)
    bulk = torch.randn(2047, generator=g)
    pathological = V7.tpo_utility(torch.cat([bulk, torch.tensor([300.0])]), utility_clip=3.0)
    benign = V7.tpo_utility(bulk, utility_clip=3.0)
    # The outlier really does collapse the post-clip scale.
    assert float(pathological.std(unbiased=False)) < 0.5 * float(benign.std(unbiased=False))

    unfloored, _ = V7.tpo_solve_eta(pathological, 0.25, eta_min=0.05, eta_max=50.0, iters=40)
    assert math.exp(3.0 / unfloored) > 50.0, "the hazard should be reproducible without the floor"
    floor = V7.eta_floor(3.0, 10.0, eta_min=0.05)
    floored, _ = V7.tpo_solve_eta(pathological, 0.25, eta_min=floor, eta_max=50.0, iters=40)
    assert math.isclose(floored, floor, rel_tol=1e-6)
    assert math.exp(3.0 / floored) <= 10.0 + 1e-6
    # ...and leaves the benign case alone, so it is not just a global clamp.
    assert V7.tpo_solve_eta(benign, 0.25, eta_min=floor, eta_max=50.0, iters=40)[0] > floor * 1.05


# --- bracket saturation is defined, reported, and no longer misdescribed ------


def test_budget_below_the_ceiling_kl_saturates_and_does_not_silently_pass():
    """At eta_max the returned KL is ABOVE the budget: a violation, not an approach."""
    u = _utilities()
    ceiling_kl = V7.tpo_target_kl(u, SOLVER["eta_max"])
    eta, kl = V7.tpo_solve_eta(u, ceiling_kl * 0.5, **SOLVER)
    assert math.isclose(eta, SOLVER["eta_max"], rel_tol=1e-6)
    assert kl > ceiling_kl * 0.5, "callers must be able to detect the overshoot"


def test_non_positive_or_inverted_inputs_are_rejected():
    u = _utilities(n=64)
    for budget in (0.0, -0.1):
        try:
            V7.tpo_solve_eta(u, budget, **SOLVER)
        except ValueError:
            continue
        raise AssertionError(f"budget={budget} was accepted")
    try:
        V7.tpo_solve_eta(u, 0.125, eta_min=10.0, eta_max=1.0, iters=40)
    except ValueError:
        pass
    else:
        raise AssertionError("an inverted bracket was accepted")


def test_constant_nonzero_utilities_do_not_drive_eta_to_the_aggressive_bracket():
    """KL is identically 0 for any constant batch, not just an all-zero one."""
    constant = torch.full((256,), 1.7, dtype=torch.float64)
    assert abs(V7.tpo_target_kl(constant, 0.05)) < 1e-12
    eta, kl = V7.tpo_solve_eta(constant, 0.125, **SOLVER)
    assert eta == SOLVER["eta_max"] and kl == 0.0


def test_variance_free_batch_leaves_the_actor_exactly_neutral():
    utility = V7.tpo_utility(torch.full((256,), 2.5), utility_clip=3.0)
    torch.testing.assert_close(utility, torch.zeros(256))
    eta, _ = V7.tpo_solve_eta(utility, 0.125, **SOLVER)
    logratio = torch.zeros(256, requires_grad=True)
    V7.tpo_intra_loss(logratio, utility, eta, GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad, torch.zeros(256))


def test_non_finite_utilities_are_rejected():
    u = _utilities(n=64)
    u[7] = float("nan")
    try:
        V7.tpo_solve_eta(u, 0.125, **SOLVER)
    except ValueError:
        return
    raise AssertionError("the solver accepted non-finite utilities")


def test_solver_neither_mutates_its_input_nor_returns_graph_bearing_values():
    u = _utilities(n=256).float().requires_grad_(True)
    before = u.detach().clone()
    eta, kl = V7.tpo_solve_eta(u, 0.125, **SOLVER)
    torch.testing.assert_close(u.detach(), before)
    # Plain floats, so nothing can leak into a later backward pass.
    assert type(eta) is float and type(kl) is float


def test_solver_reduces_in_float64_even_for_float32_input():
    """Production passes an fp32 CUDA tensor; the reduction must not follow it down.

    Comparing against the *exact* float64 promotion of the same fp32 values isolates
    the reduction dtype from input precision: if the solver upcast were removed the
    left side would reduce in fp32 and the two would no longer agree bit-for-bit.
    """
    u32 = _utilities(n=4096).float()
    assert V7.tpo_solve_eta(u32, 0.125, **SOLVER) == V7.tpo_solve_eta(u32.double(), 0.125, **SOLVER)
    assert V7.tpo_target_kl(u32.double(), 2.0) != V7.tpo_target_kl(_utilities(n=4096), 2.0), (
        "fp32 rounding must be visible at all, or the comparison above proves nothing"
    )


# --- the realized-KL controller ----------------------------------------------


def test_controller_moves_the_budget_against_the_error_and_is_a_no_op_on_target():
    target = 0.03
    on_target = V7.kl_budget_update(0.08, target, target, **CTRL)
    assert math.isclose(on_target, 0.08, rel_tol=1e-12)
    assert V7.kl_budget_update(0.08, 2.0 * target, target, **CTRL) < 0.08
    assert V7.kl_budget_update(0.08, 0.5 * target, target, **CTRL) > 0.08


def test_controller_converges_to_the_budget_that_delivers_the_target():
    """Closed loop against a fixed realisability factor must reach a fixed point."""
    target, efficiency, budget = 0.03, 0.42, 0.5
    for _ in range(200):
        budget = V7.kl_budget_update(budget, budget * efficiency, target, **CTRL)
    assert math.isclose(budget, target / efficiency, rel_tol=1e-3), budget
    # From far below as well, so it is not a one-sided walk into a bound.
    budget = 1e-3
    for _ in range(200):
        budget = V7.kl_budget_update(budget, budget * efficiency, target, **CTRL)
    assert math.isclose(budget, target / efficiency, rel_tol=1e-3), budget


def test_controller_is_scale_free_in_log_space():
    """The same relative error produces the same relative correction."""
    ratios = [
        V7.kl_budget_update(b, 2.0 * 0.03, 0.03, **CTRL) / b for b in (0.003, 0.03, 0.3)
    ]
    assert max(ratios) - min(ratios) < 1e-12


def test_one_anomalous_iteration_cannot_slam_the_budget_into_a_bound():
    budget = 0.08
    worst = V7.kl_budget_update(budget, 1e6, 0.03, **CTRL)
    bound = CTRL["ratio_clip"] ** -CTRL["gain"]
    assert math.isclose(worst / budget, bound, rel_tol=1e-9)
    assert worst > CTRL["bounds"][0] * 10.0, "a single spike must be recoverable"
    # A realized KL of exactly zero must not produce -inf / nan.
    assert math.isfinite(V7.kl_budget_update(budget, 0.0, 0.03, **CTRL))


def test_controller_respects_its_hard_bounds():
    lo, hi = CTRL["bounds"]
    for _ in range(500):
        lo_walk = V7.kl_budget_update(lo, 1e3, 0.03, **CTRL)
        hi_walk = V7.kl_budget_update(hi, 1e-9, 0.03, **CTRL)
    assert lo_walk >= lo and hi_walk <= hi


def test_disabling_the_controller_recovers_v6_open_loop_behaviour():
    for realized in (1e-6, 0.03, 10.0):
        assert V7.kl_budget_update(0.08, realized, None, **CTRL) == 0.08


# --- everything downstream of the temperature must still be v2 verbatim ------


def test_objective_and_whitening_are_identical_to_v2():
    torch.manual_seed(0)
    logratio = torch.randn(256) * 0.6
    utility = torch.randn(256) * 1.5
    for eta in (1.3, 2.0, 4.0):
        torch.testing.assert_close(
            V7.tpo_intra_loss(logratio, utility, eta, GUARD),
            V2.tpo_intra_loss(logratio, utility, eta, GUARD),
        )
    advantages = torch.randn(512) * 7.0
    torch.testing.assert_close(V7.tpo_utility(advantages, 3.0), V2.tpo_utility(advantages, 3.0))


def test_agent_parameters_are_bit_identical_to_v2_under_the_same_seed():
    """Compare actual parameters and outputs.

    A previous version of this test iterated a Beta distribution's __dict__, which
    holds no tensors, so its assertion loop ran zero times and changes to the actor
    head width, init std, and even the alpha/beta offset all passed silently.
    """
    obs = torch.randn(16, 5)
    torch.manual_seed(7)
    a7 = V7.Agent(_envs_stub())
    torch.manual_seed(7)
    a2 = V2.Agent(_envs_stub())

    p7 = dict(a7.named_parameters())
    p2 = dict(a2.named_parameters())
    assert set(p7) == set(p2) and len(p7) > 0
    for name in p7:
        torch.testing.assert_close(p7[name], p2[name], msg=lambda m, n=name: f"{n}: {m}")

    d7, d2 = a7._dist(obs), a2._dist(obs)
    torch.testing.assert_close(d7.concentration1, d2.concentration1)
    torch.testing.assert_close(d7.concentration0, d2.concentration0)
    assert torch.all(d7.concentration1 >= 1.0) and torch.all(d7.concentration0 >= 1.0)
    torch.testing.assert_close(a7.get_value(obs), a2.get_value(obs))


def test_end_to_end_gradient_is_ratio_minus_target_times_dlogpi():
    torch.manual_seed(0)
    agent = V7.Agent(_envs_stub())
    obs = torch.randn(8, 5)
    with torch.no_grad():
        _, z, old_logprob, *_ = agent.get_action_and_value(obs)
        agent.actor_alpha.weight.add_(0.05)

    utility = V7.tpo_utility(torch.linspace(-3.0, 4.0, 8), utility_clip=3.0)
    eta, _ = V7.tpo_solve_eta(utility, 0.08, **SOLVER)
    _, _, logprob, *_ = agent.get_action_and_value(obs, z)
    logratio = logprob - old_logprob
    V7.tpo_intra_loss(logratio, utility, eta, GUARD).sum().backward()
    got = agent.actor_alpha.weight.grad.clone()

    agent.zero_grad(set_to_none=True)
    coefficient = (logratio.detach().exp() - (utility / eta).exp()).detach()
    _, _, logprob2, *_ = agent.get_action_and_value(obs, z)
    (coefficient * logprob2).sum().backward()
    torch.testing.assert_close(got, agent.actor_alpha.weight.grad, rtol=1e-4, atol=1e-6)


def test_args_expose_the_intended_knobs_and_no_stale_ones():
    fields = V7.Args.__dataclass_fields__
    for gone in ("eta", "utility_scope"):
        assert gone not in fields
    for present in ("realized_kl_target", "kl_budget", "kl_budget_gain", "max_target_ratio"):
        assert present in fields
    # The defaults the solver actually runs with must stay sane; a raised eta_min
    # or a lowered eta_max would silently cap the trust region in every run.
    assert DEFAULTS.eta_min <= 0.1 and DEFAULTS.eta_max >= 10.0
    assert DEFAULTS.utility_clip is not None
    floor = V7.eta_floor(DEFAULTS.utility_clip, DEFAULTS.max_target_ratio, DEFAULTS.eta_min)
    assert 1.0 < floor < 2.0, f"default floor {floor} should sit below fixed eta=2"
    assert math.exp(DEFAULTS.utility_clip / floor) <= DEFAULTS.max_target_ratio + 1e-9

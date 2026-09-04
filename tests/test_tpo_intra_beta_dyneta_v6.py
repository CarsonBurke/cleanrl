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


V6 = _load("tpo_intra_beta_dyneta_v6", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_dyneta_v6.py")
V2 = _load("tpo_intra_beta_v2_ref", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_v2.py")

SOLVER = dict(eta_min=0.05, eta_max=50.0, iters=40)


def _utilities(n=8192, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float64)


# --- KL(eta) is the quantity the solver claims it is -------------------------


def test_target_kl_is_strictly_decreasing_in_eta():
    u = _utilities()
    kls = [V6.tpo_target_kl(u, eta) for eta in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
    assert all(a > b for a, b in zip(kls, kls[1:])), kls
    assert kls[-1] > 0.0


def test_target_kl_matches_the_analytic_gaussian_value():
    """For u ~ N(0,1) the tilted target has KL exactly 1/(2 eta^2)."""
    u = _utilities(n=400_000, seed=3)
    u = (u - u.mean()) / u.std(unbiased=False)
    for eta in (1.0, 2.0, 4.0):
        got = V6.tpo_target_kl(u, eta)
        assert math.isclose(got, 1.0 / (2.0 * eta**2), rel_tol=0.03), (eta, got)


def test_target_kl_is_zero_for_a_variance_free_batch():
    assert V6.tpo_target_kl(torch.zeros(64, dtype=torch.float64), 2.0) == 0.0


def test_target_kl_is_permutation_invariant():
    u = _utilities(n=512)
    torch.testing.assert_close(
        V6.tpo_target_kl(u, 1.7), V6.tpo_target_kl(u[torch.randperm(512)], 1.7), rtol=1e-12, atol=0
    )


def test_target_kl_survives_utilities_that_would_overflow_exp():
    """u/eta reaches 60 at the aggressive end of the bracket; logsumexp must hold."""
    u = torch.tensor([-3.0, 0.0, 3.0] * 32, dtype=torch.float64)
    kl = V6.tpo_target_kl(u, 0.05)
    assert math.isfinite(kl) and kl > 0.0
    # The weights collapse onto the third of samples at the clip, so KL -> log 3.
    assert math.isclose(kl, math.log(3.0), rel_tol=1e-6)


# --- the solver actually solves the dual -------------------------------------


def test_solved_eta_hits_the_requested_budget():
    u = _utilities()
    for budget in (0.02, 0.0625, 0.125, 0.25, 0.5):
        eta, kl = V6.tpo_solve_eta(u, budget, **SOLVER)
        assert SOLVER["eta_min"] < eta < SOLVER["eta_max"]
        assert math.isclose(kl, budget, rel_tol=1e-6), (budget, eta, kl)


def test_solved_eta_is_the_stationary_point_of_the_convex_dual():
    """g(eta) = eta*eps + eta*log E[exp(u/eta)] must have g'(eta*) = 0."""
    u = _utilities()
    eps = 0.125
    eta, _ = V6.tpo_solve_eta(u, eps, **SOLVER)

    def dual(e):
        log_partition = torch.logsumexp(u / e, 0) - math.log(u.numel())
        return e * eps + e * float(log_partition)

    h = 1e-5 * eta
    slope = (dual(eta + h) - dual(eta - h)) / (2.0 * h)
    assert abs(slope) < 1e-6, slope
    # ...and it is a minimum, not just a stationary point.
    assert dual(eta) < min(dual(eta * 0.8), dual(eta * 1.25))


def test_gaussian_utilities_recover_the_equivalent_fixed_eta():
    """The default budget 0.125 must reproduce eta=2, so v6 is comparable to v2."""
    u = _utilities(n=400_000, seed=5)
    u = (u - u.mean()) / u.std(unbiased=False)
    eta, _ = V6.tpo_solve_eta(u, 0.125, **SOLVER)
    assert math.isclose(eta, 2.0, rel_tol=0.03), eta


def test_solved_eta_moves_with_the_shape_of_the_utility_distribution():
    """The motivating claim: unit variance does not pin down the trust region.

    Two batches are whitened by the same code and clipped by the same rule, and
    differ only in the shape of the raw distribution. A fixed eta gives them
    materially different trust regions; the solver gives them different etas and
    the requested KL.

    Note the clip is a second, independent source of drift: whitening guarantees
    unit variance only *before* clipping, and clipping a skewed batch truncates one
    tail harder than the other, so the post-clip moments are not unit either.
    """
    g = torch.Generator().manual_seed(11)
    gaussian = torch.randn(200_000, generator=g, dtype=torch.float64)
    skewed = torch.distributions.Exponential(1.0).sample((200_000,)).double()

    # The whitening contract itself is exact; it is the clip that reshapes.
    for raw in (gaussian, skewed):
        unclipped = V6.tpo_utility(raw, utility_clip=None)
        assert abs(float(unclipped.mean())) < 1e-5
        assert math.isclose(float(unclipped.std(unbiased=False)), 1.0, rel_tol=1e-5)
    clipped = [V6.tpo_utility(raw, utility_clip=3.0).double() for raw in (gaussian, skewed)]
    assert not math.isclose(
        float(clipped[1].std(unbiased=False)), float(clipped[0].std(unbiased=False)), rel_tol=0.02
    ), "the clip should already have broken the shared scale"

    etas, kls = zip(*[V6.tpo_solve_eta(u, 0.125, **SOLVER) for u in clipped])
    assert all(math.isclose(kl, 0.125, rel_tol=1e-6) for kl in kls), kls
    assert abs(etas[0] - etas[1]) / etas[0] > 0.05, etas
    fixed_kls = [V6.tpo_target_kl(u, 2.0) for u in clipped]
    assert abs(fixed_kls[0] - fixed_kls[1]) / fixed_kls[0] > 0.05, fixed_kls


# --- bracket saturation is a defined outcome, not an error -------------------


def test_unreachable_budget_saturates_at_the_aggressive_bracket():
    """Clipped utilities cap the expressible KL; the solver must not diverge."""
    u = _utilities()
    ceiling = V6.tpo_target_kl(u.clamp(-3.0, 3.0), SOLVER["eta_min"])
    eta, kl = V6.tpo_solve_eta(u.clamp(-3.0, 3.0), ceiling * 2.0, **SOLVER)
    assert math.isclose(eta, SOLVER["eta_min"], rel_tol=1e-6)
    assert math.isfinite(kl) and kl < ceiling * 2.0


def test_variance_free_batch_saturates_at_the_neutral_bracket():
    utility = V6.tpo_utility(torch.full((256,), 2.5), utility_clip=3.0)
    torch.testing.assert_close(utility, torch.zeros(256))
    eta, kl = V6.tpo_solve_eta(utility, 0.125, **SOLVER)
    assert eta == SOLVER["eta_max"] and kl == 0.0
    # Whatever eta comes back, a signal-free rollout must not move the actor.
    logratio = torch.zeros(256, requires_grad=True)
    V6.tpo_intra_loss(logratio, utility, eta, GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad, torch.zeros(256))


def test_non_finite_utilities_are_rejected_rather_than_silently_solved():
    u = _utilities(n=64)
    u[7] = float("nan")
    try:
        V6.tpo_solve_eta(u, 0.125, **SOLVER)
    except ValueError:
        return
    raise AssertionError("the solver accepted non-finite utilities")


def test_solver_does_not_mutate_or_backpropagate_into_its_input():
    u = _utilities(n=256).float().requires_grad_(True)
    before = u.detach().clone()
    V6.tpo_solve_eta(u, 0.125, **SOLVER)
    torch.testing.assert_close(u.detach(), before)
    assert u.grad is None


# --- everything else must be v2 verbatim -------------------------------------


def test_objective_and_whitening_are_identical_to_v2():
    torch.manual_seed(0)
    logratio = torch.randn(256) * 0.6
    utility = torch.randn(256) * 1.5
    for eta in (0.7, 2.0, 4.0):
        torch.testing.assert_close(
            V6.tpo_intra_loss(logratio, utility, eta, GUARD),
            V2.tpo_intra_loss(logratio, utility, eta, GUARD),
        )
    advantages = torch.randn(512) * 7.0
    torch.testing.assert_close(V6.tpo_utility(advantages, 3.0), V2.tpo_utility(advantages, 3.0))


def _envs_stub():
    import gymnasium as gym

    class Stub:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    return Stub()


def test_agent_is_bit_identical_to_v2_under_the_same_seed():
    obs = torch.randn(16, 5)
    torch.manual_seed(7)
    a6 = V6.Agent(_envs_stub())
    torch.manual_seed(7)
    a2 = V2.Agent(_envs_stub())
    for m6, m2 in zip(a6._dist(obs).__dict__.values(), a2._dist(obs).__dict__.values()):
        if torch.is_tensor(m6):
            torch.testing.assert_close(m6, m2)
    torch.testing.assert_close(a6.get_value(obs), a2.get_value(obs))


def test_end_to_end_gradient_is_ratio_minus_target_times_dlogpi():
    torch.manual_seed(0)
    agent = V6.Agent(_envs_stub())
    obs = torch.randn(8, 5)
    with torch.no_grad():
        _, z, old_logprob, *_ = agent.get_action_and_value(obs)
        agent.actor_alpha.weight.add_(0.05)

    utility = V6.tpo_utility(torch.linspace(-3.0, 4.0, 8), utility_clip=3.0)
    eta, _ = V6.tpo_solve_eta(utility, 0.125, **SOLVER)
    _, _, logprob, *_ = agent.get_action_and_value(obs, z)
    logratio = logprob - old_logprob
    V6.tpo_intra_loss(logratio, utility, eta, GUARD).sum().backward()
    got = agent.actor_alpha.weight.grad.clone()

    agent.zero_grad(set_to_none=True)
    coefficient = (logratio.detach().exp() - (utility / eta).exp()).detach()
    _, _, logprob2, *_ = agent.get_action_and_value(obs, z)
    (coefficient * logprob2).sum().backward()
    torch.testing.assert_close(got, agent.actor_alpha.weight.grad, rtol=1e-4, atol=1e-6)


def test_the_fixed_eta_knob_is_gone_from_the_args():
    fields = V6.Args.__dataclass_fields__
    assert "eta" not in fields, "v6 must not expose a fixed temperature"
    assert "utility_scope" not in fields, "batch scope is the only correct scope here"
    for name in ("kl_budget", "eta_min", "eta_max", "eta_solver_iters"):
        assert name in fields
    assert fields["kl_budget"].default == 0.125

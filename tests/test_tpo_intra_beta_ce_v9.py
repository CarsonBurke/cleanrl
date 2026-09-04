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


V9 = _load("tpo_intra_beta_ce_v9", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_ce_v9.py")
V7 = _load("tpo_intra_beta_klctrl_v7_ref", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_klctrl_v7.py")

_ARGS = V9.Args()
SOLVER = dict(eta_min=_ARGS.eta_min, eta_max=_ARGS.eta_max, iters=_ARGS.eta_solver_iters)
CTRL = dict(
    gain=_ARGS.kl_budget_gain,
    ratio_clip=_ARGS.kl_ratio_clip,
    bounds=_ARGS.kl_budget_bounds,
)


def _utilities(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, generator=g, dtype=torch.float64)
    return ((u - u.mean()) / u.std(unbiased=False)).clamp(-3.0, 3.0)


# --- the target is the paper's -----------------------------------------------


def test_group_target_is_the_papers_no_anchor_target():
    """softmax(skill/eta) over the group, which is tpo_target_no_anchor verbatim."""
    u = _utilities()
    q = V9.tpo_group_target(u, 1.7)
    torch.testing.assert_close(q, torch.softmax(u / 1.7, 0))
    assert math.isclose(float(q.sum()), 1.0, rel_tol=1e-12)
    assert float(q.min()) > 0.0


def test_group_target_equals_the_anchored_form_under_the_pooled_measure():
    """The paper anchors on log_softmax(log pi_old) over the group.

    Samples here are drawn from pi_old, so their empirical distribution under
    pi_old is uniform and the anchor is a constant that the softmax removes. Any
    *non*-uniform anchor would give a different target, so this is a statement
    about the sampling scheme, not an identity of softmax.
    """
    u = _utilities()
    uniform_anchor = torch.log_softmax(torch.zeros_like(u), 0)
    anchored = torch.softmax(uniform_anchor + u / 2.0, 0)
    torch.testing.assert_close(anchored, V9.tpo_group_target(u, 2.0))
    skewed_anchor = torch.log_softmax(torch.linspace(-1.0, 1.0, u.numel(), dtype=u.dtype), 0)
    skewed = torch.softmax(skewed_anchor + u / 2.0, 0)
    assert not torch.allclose(skewed, anchored)


def test_the_target_is_detached_and_cannot_be_trained_through():
    u = _utilities().float().requires_grad_(True)
    q = V9.tpo_group_target(u, 2.0)
    assert not q.requires_grad and q.grad_fn is None
    # Not vacuous: the same computation without the detach does carry a graph.
    assert torch.softmax(u / 2.0, 0).grad_fn is not None


def test_no_sample_can_demand_more_than_all_the_mass_however_small_eta_is():
    """The property v7's exp(u/eta) lacked, and needed an eta floor to fake."""
    u = _utilities()
    for eta in (2.0, 0.5, 0.05, 0.001):
        q = V9.tpo_group_target(u, eta)
        assert float(q.max()) <= 1.0 and torch.isfinite(q).all()
        assert math.isclose(float(q.sum()), 1.0, rel_tol=1e-9)
    # v7's unnormalised target over the same range is not merely larger, it diverges.
    assert float((u / 0.001).exp().max()) > 1e100


# --- the loss and its gradient ------------------------------------------------


def test_gradient_of_the_loss_is_exactly_p_minus_q():
    u = _utilities(n=256, seed=1)
    logratio = (torch.randn(256, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 0.4)
    logratio.requires_grad_(True)
    q = V9.tpo_group_target(u, 2.0)
    V9.tpo_cross_entropy_loss(logratio, q).backward()
    p = torch.softmax(logratio.detach(), 0)
    torch.testing.assert_close(logratio.grad, p - q, rtol=1e-10, atol=1e-14)


def test_the_gradient_coefficients_sum_to_zero():
    """No net push on the group's total density, so no built-in entropy pressure.

    v7's (r - w) has no such constraint: a positive-mean utility batch pushes every
    sample's density up, which a normalised density can only grant by sharpening.
    """
    u = _utilities(n=256, seed=3)
    logratio = torch.zeros(256, dtype=torch.float64, requires_grad=True)
    V9.tpo_cross_entropy_loss(logratio, V9.tpo_group_target(u, 2.0)).backward()
    assert abs(float(logratio.grad.sum())) < 1e-12
    v7_logratio = torch.zeros(256, dtype=torch.float64, requires_grad=True)
    V7.tpo_intra_loss(v7_logratio, u + 0.8, 2.0, GUARD).mean().backward()
    assert abs(float(v7_logratio.grad.sum())) > 1e-3


def test_the_loss_is_invariant_to_a_constant_shift_of_the_log_ratios():
    """The free constant that makes the target reachable.

    v7 demands r_i = exp(u_i/eta) exactly; v9 demands it only up to a common
    factor, and that factor is precisely the part no policy could have delivered.
    """
    u = _utilities(n=128, seed=4)
    logratio = torch.randn(128, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    q = V9.tpo_group_target(u, 2.0)
    base = float(V9.tpo_cross_entropy_loss(logratio, q))
    for c in (-4.0, 0.3, 7.0):
        shifted = float(V9.tpo_cross_entropy_loss(logratio + c, q))
        assert math.isclose(shifted, base, rel_tol=1e-12), (c, base, shifted)
    v7_losses = [float(V7.tpo_intra_loss(logratio + c, u, 2.0, GUARD).mean()) for c in (0.0, 0.3)]
    assert not math.isclose(*v7_losses, rel_tol=1e-6), "v7 must not share the invariance"


def test_the_fixed_point_is_the_tilted_ratio_up_to_any_constant():
    u = _utilities(n=128, seed=6)
    eta = 1.9
    q = V9.tpo_group_target(u, eta)
    for c in (-3.0, 0.0, 2.5):
        logratio = (u / eta + c).clone().requires_grad_(True)
        V9.tpo_cross_entropy_loss(logratio, q).backward()
        assert float(logratio.grad.abs().max()) < 1e-15, (c, float(logratio.grad.abs().max()))


def test_the_loss_needs_no_overflow_guard_at_any_log_ratio():
    """log_softmax subtracts its own max, so nothing is ever exponentiated raw."""
    u = _utilities(n=64, seed=7)
    q = V9.tpo_group_target(u, 2.0)
    for magnitude in (50.0, 1e3, 1e6):
        logratio = torch.linspace(-magnitude, magnitude, 64, dtype=torch.float64).requires_grad_(True)
        loss = V9.tpo_cross_entropy_loss(logratio, q)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logratio.grad).all(), magnitude
        assert float(logratio.grad.abs().max()) <= 1.0 + 1e-12
    # The same log-ratios overflow the unguarded form v7 had to linearise.
    assert not torch.isfinite(torch.tensor(1e6, dtype=torch.float64).exp())


def test_a_uniform_target_leaves_an_unmoved_policy_alone():
    u = torch.zeros(64, dtype=torch.float64)
    logratio = torch.zeros(64, dtype=torch.float64, requires_grad=True)
    V9.tpo_cross_entropy_loss(logratio, V9.tpo_group_target(u, 2.0)).backward()
    torch.testing.assert_close(logratio.grad, torch.zeros(64, dtype=torch.float64))


# --- the trust region and the objective are now one quantity ------------------


def test_the_solved_target_kl_is_the_loss_at_the_start_of_the_update():
    """KL(q ‖ p) with p uniform is exactly what tpo_target_kl returns.

    So the dual is solving for the initial value of the objective it is paired
    with, rather than for a related-but-different joint KL.
    """
    u = _utilities(n=1024, seed=8)
    for eta in (0.9, 2.0, 5.0):
        q = V9.tpo_group_target(u, eta)
        logratio = torch.zeros_like(u)
        cross_entropy = float(V9.tpo_cross_entropy_loss(logratio, q))
        group_kl = cross_entropy + float((q * q.log()).sum())
        assert math.isclose(group_kl, V9.tpo_target_kl(u, eta), rel_tol=1e-9), eta


def test_fitting_the_target_drives_the_group_kl_down():
    u = _utilities(n=256, seed=9)
    eta, _ = V9.tpo_solve_eta(u, 0.08, **SOLVER)
    q = V9.tpo_group_target(u, eta)
    logratio = torch.zeros_like(u).requires_grad_(True)
    optimiser = torch.optim.SGD([logratio], lr=200.0)

    def group_kl():
        return float((q * (q.log() - torch.log_softmax(logratio.detach(), 0))).sum())

    before = group_kl()
    for _ in range(50):
        optimiser.zero_grad()
        V9.tpo_cross_entropy_loss(logratio, q).backward()
        optimiser.step()
    after = group_kl()
    assert math.isclose(before, V9.tpo_target_kl(u, eta), rel_tol=1e-9)
    assert after < before * 0.5, (before, after)


# --- controller anti-windup ---------------------------------------------------


def test_controller_matches_v7_exactly_while_eta_is_unsaturated():
    for realized in (0.005, 0.03, 0.2):
        assert V9.kl_budget_update(0.08, realized, 0.03, **CTRL) == V7.kl_budget_update(
            0.08, realized, 0.03, **CTRL
        )


def test_saturated_eta_refuses_the_demand_it_cannot_serve():
    budget = 0.08
    assert V9.kl_budget_update(budget, 0.001, 0.03, **CTRL, eta_at_floor=True) == budget
    assert V7.kl_budget_update(budget, 0.001, 0.03, **CTRL) > budget
    assert V9.kl_budget_update(budget, 0.3, 0.03, **CTRL, eta_at_floor=True) < budget
    assert V9.kl_budget_update(budget, 0.3, 0.03, **CTRL, eta_at_ceiling=True) == budget


def test_windup_is_actually_prevented_over_a_saturated_run():
    v7_budget = v9_budget = 0.08
    for _ in range(200):
        v7_budget = V7.kl_budget_update(v7_budget, 0.02, 0.03, **CTRL)
        v9_budget = V9.kl_budget_update(v9_budget, 0.02, 0.03, **CTRL, eta_at_floor=True)
    assert math.isclose(v7_budget, CTRL["bounds"][1]), v7_budget
    assert v9_budget == 0.08


# --- everything outside the loss must be v7 verbatim --------------------------


def test_solver_utility_and_floor_logic_are_identical_to_v7():
    advantages = torch.randn(512, generator=torch.Generator().manual_seed(11)) * 7.0
    torch.testing.assert_close(V9.tpo_utility(advantages, 3.0), V7.tpo_utility(advantages, 3.0))
    u = _utilities(n=1024, seed=12)
    assert V9.tpo_solve_eta(u, 0.08, **SOLVER) == V7.tpo_solve_eta(u, 0.08, **SOLVER)
    assert V9.eta_floor(3.0, 10.0, 0.05) == V7.eta_floor(3.0, 10.0, 0.05)


def test_the_eta_floor_is_disabled_so_the_controller_can_reach_its_target():
    """v9 realises less approx_kl per nat of budget, so it needs eta headroom.

    v7's floor exists to bound exp(u/eta); under a normalised target that bound is
    structural, so retaining the floor would only pin eta and strand realized KL
    below its target -- which is the one variable being held equal.
    """
    assert V9.Args().max_target_ratio == float("inf")
    assert V9.eta_floor(3.0, V9.Args().max_target_ratio, 0.05) == 0.05
    assert V7.eta_floor(3.0, V7.Args().max_target_ratio, 0.05) > 1.3
    # And nothing blows up down there, which is what earned the floor's removal.
    u = _utilities(n=256, seed=13)
    u[0], u[1] = 3.0, -3.0  # the clip is what bounds the demand, so exercise it
    q = V9.tpo_group_target(u, 0.05)
    assert torch.isfinite(q).all() and math.isclose(float(q.sum()), 1.0, rel_tol=1e-9)
    logratio = torch.zeros_like(u).requires_grad_(True)
    V9.tpo_cross_entropy_loss(logratio, q).backward()
    assert float(logratio.grad.abs().max()) <= 1.0
    # v7's target at the same eta demands a 1e26-fold density change from a single
    # sample, which is why it needed the floor that v9 can afford to drop.
    assert float((u / 0.05).exp().max()) > 1e25
    v7_logratio = torch.zeros_like(u).requires_grad_(True)
    V7.tpo_intra_loss(v7_logratio, u, 0.05, GUARD).mean().backward()
    assert float(v7_logratio.grad.abs().max()) > 1e22


def test_the_ratio_matching_loss_is_gone():
    assert not hasattr(V9, "tpo_intra_loss"), "v9 must not keep the unnormalised loss"


def _envs_stub():
    import gymnasium as gym

    class Stub:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    return Stub()


def test_agent_is_bit_identical_to_v7_under_the_same_seed():
    obs = torch.randn(16, 5)
    torch.manual_seed(7)
    a9 = V9.Agent(_envs_stub())
    torch.manual_seed(7)
    a7 = V7.Agent(_envs_stub())
    p9 = dict(a9.named_parameters())
    p7 = dict(a7.named_parameters())
    assert p9.keys() == p7.keys() and len(p9) > 0
    for name in p9:
        torch.testing.assert_close(p9[name], p7[name])
    d9, d7 = a9._dist(obs), a7._dist(obs)
    torch.testing.assert_close(d9.concentration1, d7.concentration1)
    torch.testing.assert_close(d9.concentration0, d7.concentration0)
    torch.testing.assert_close(a9.get_value(obs), a7.get_value(obs))


def test_end_to_end_gradient_flows_as_p_minus_q_times_dlogpi():
    torch.manual_seed(0)
    agent = V9.Agent(_envs_stub())
    obs = torch.randn(16, 5)
    with torch.no_grad():
        _, z, old_logprob, *_ = agent.get_action_and_value(obs)
        agent.actor_alpha.weight.add_(0.05)

    utility = V9.tpo_utility(torch.linspace(-3.0, 4.0, 16), utility_clip=3.0)
    eta, _ = V9.tpo_solve_eta(utility, 0.08, **SOLVER)
    target = V9.tpo_group_target(utility, eta)

    _, _, logprob, *_ = agent.get_action_and_value(obs, z)
    V9.tpo_cross_entropy_loss(logprob - old_logprob, target).backward()
    got = agent.actor_alpha.weight.grad.clone()

    agent.zero_grad(set_to_none=True)
    _, _, logprob2, *_ = agent.get_action_and_value(obs, z)
    coefficient = (torch.softmax(logprob2.detach() - old_logprob, 0) - target).detach()
    (coefficient * logprob2).sum().backward()
    torch.testing.assert_close(got, agent.actor_alpha.weight.grad, rtol=1e-4, atol=1e-7)
    assert float(got.abs().sum()) > 0.0


def test_the_args_differ_from_v7_in_exactly_two_declared_places():
    """Anything else changing silently would break the controlled comparison."""
    added = set(V9.Args.__dataclass_fields__) - set(V7.Args.__dataclass_fields__)
    assert added == {"kl_anti_windup"}, added
    assert not set(V7.Args.__dataclass_fields__) - set(V9.Args.__dataclass_fields__)
    changed = {
        name
        for name, field in V9.Args.__dataclass_fields__.items()
        if name not in added
        and name != "exp_name"
        and field.default != V7.Args.__dataclass_fields__[name].default
    }
    assert changed == {"max_target_ratio"}, changed
    assert V9.Args().kl_anti_windup is False

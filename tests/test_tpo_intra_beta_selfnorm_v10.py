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


V10 = _load("tpo_intra_beta_selfnorm_v10", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_selfnorm_v10.py")
V7 = _load("tpo_intra_beta_klctrl_v7_ref", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_klctrl_v7.py")

_ARGS = V10.Args()
SOLVER = dict(eta_min=_ARGS.eta_min, eta_max=_ARGS.eta_max, iters=_ARGS.eta_solver_iters)


def _utilities(n=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, generator=g, dtype=torch.float64)
    return ((u - u.mean()) / u.std(unbiased=False)).clamp(-3.0, 3.0)


def _target(utility, eta, log_partition):
    return (utility / eta - log_partition).exp()


# --- the defining property: the target is now a density ratio -----------------


def test_the_target_ratios_average_to_exactly_one():
    """E_pi_old[q/pi_old] = 1 is not a nicety, it is what makes q a distribution."""
    for seed, eta in ((0, 2.3), (1, 0.7), (2, 8.0)):
        u = _utilities(seed=seed)
        w = _target(u, eta, V10.tpo_log_partition(u, eta))
        assert math.isclose(float(w.mean()), 1.0, rel_tol=1e-12), (seed, eta, float(w.mean()))


def test_v7s_target_does_not_average_to_one_and_the_gap_is_the_jensen_term():
    u = _utilities(seed=3)
    for eta in (1.5, 2.33, 4.0):
        v7_mean = float((u / eta).exp().mean())
        assert v7_mean > 1.0, eta
        # exp(logZ) is exactly that mean, so the correction removes exactly it.
        assert math.isclose(math.exp(V10.tpo_log_partition(u, eta)), v7_mean, rel_tol=1e-12)
    # At the eta this codebase runs, the inflation is the measured ~7%.
    assert 1.05 < float((u / 2.33).exp().mean()) < 1.10


def test_the_target_is_n_times_the_softmax_ie_the_reps_importance_weights():
    """The correction is the self-normalised estimator, not an arbitrary rescale."""
    u = _utilities(n=512, seed=4)
    eta = 1.9
    w = _target(u, eta, V10.tpo_log_partition(u, eta))
    torch.testing.assert_close(w, u.numel() * torch.softmax(u / eta, 0), rtol=1e-12, atol=0)


def test_the_correction_is_a_pure_rescale_that_preserves_every_ratio():
    """(A) alone: the ranking the target encodes is untouched, only its level."""
    u = _utilities(n=256, seed=5)
    eta = 2.1
    w10 = _target(u, eta, V10.tpo_log_partition(u, eta))
    w7 = (u / eta).exp()
    quotient = w10 / w7
    torch.testing.assert_close(quotient, torch.full_like(quotient, float(quotient[0])), rtol=1e-12, atol=0)
    assert float(quotient[0]) < 1.0, "the correction must shrink, not inflate"


# --- what that does to the gradient -------------------------------------------


def test_the_coefficients_sum_to_zero_at_the_start_of_the_update():
    """The property v9 had identically; v10 has it only here, which is the point."""
    u = _utilities(n=512, seed=6)
    eta = 2.2
    logratio = torch.zeros_like(u).requires_grad_(True)
    V10.tpo_intra_loss(logratio, u, eta, GUARD, V10.tpo_log_partition(u, eta)).sum().backward()
    assert abs(float(logratio.grad.sum())) < 1e-9
    # v7's sum is the coherent sharpening drive, N*(1 - E[w]), and is not zero.
    v7_logratio = torch.zeros_like(u).requires_grad_(True)
    V7.tpo_intra_loss(v7_logratio, u, eta, GUARD).sum().backward()
    assert float(v7_logratio.grad.sum()) < -0.02 * u.numel(), float(v7_logratio.grad.sum())


def test_the_coefficients_are_free_to_sum_to_anything_once_the_policy_moves():
    """Unlike v9, where p - q sums to zero at every point, killing the drive."""
    u = _utilities(n=512, seed=7)
    eta = 2.2
    log_partition = V10.tpo_log_partition(u, eta)
    sums = []
    for shift in (-0.3, 0.0, 0.3):
        logratio = torch.full_like(u, shift).requires_grad_(True)
        V10.tpo_intra_loss(logratio, u, eta, GUARD, log_partition).sum().backward()
        sums += [float(logratio.grad.sum())]
    assert sums[0] < sums[1] < sums[2]
    assert abs(sums[0]) > 1.0 and abs(sums[2]) > 1.0, sums


def test_the_fixed_point_moved_by_exactly_the_partition():
    u = _utilities(n=256, seed=8)
    eta = 1.7
    log_partition = V10.tpo_log_partition(u, eta)
    logratio = (u / eta - log_partition).clone().requires_grad_(True)
    V10.tpo_intra_loss(logratio, u, eta, GUARD, log_partition).sum().backward()
    assert float(logratio.grad.abs().max()) < 1e-9
    # v7's fixed point is a different policy, higher by exactly exp(log_partition).
    v7_fixed = (u / eta).clone().requires_grad_(True)
    V7.tpo_intra_loss(v7_fixed, u, eta, GUARD).sum().backward()
    assert float(v7_fixed.grad.abs().max()) < 1e-9
    assert math.isclose(float((u / eta).mean() - (u / eta - log_partition).mean()), log_partition)


def test_an_average_utility_sample_is_now_asked_to_go_down():
    """The mass handed to above-average samples has to come from somewhere."""
    u = _utilities(n=1024, seed=9)
    eta = 2.2
    log_partition = V10.tpo_log_partition(u, eta)
    at_mean = _target(torch.zeros(1, dtype=torch.float64), eta, log_partition)
    assert float(at_mean) < 1.0
    assert math.isclose(float(at_mean), math.exp(-log_partition), rel_tol=1e-12)
    # And the split about 1.0 is no longer 50/50 by utility sign.
    w = _target(u, eta, log_partition)
    assert float((w > 1.0).double().mean()) < 0.5


def test_zero_signal_still_leaves_the_actor_exactly_neutral():
    """A constant utility has partition u/eta, so every target ratio is exactly 1."""
    u = torch.zeros(256, dtype=torch.float64)
    eta, _ = V10.tpo_solve_eta(u, 0.08, **SOLVER)
    log_partition = V10.tpo_log_partition(u, eta)
    logratio = torch.zeros(256, dtype=torch.float64, requires_grad=True)
    V10.tpo_intra_loss(logratio, u, eta, GUARD, log_partition).sum().backward()
    torch.testing.assert_close(logratio.grad, torch.zeros(256, dtype=torch.float64))
    constant = torch.full((256,), 2.5, dtype=torch.float64)
    torch.testing.assert_close(
        _target(constant, 2.0, V10.tpo_log_partition(constant, 2.0)),
        torch.ones(256, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


# --- the partition itself ------------------------------------------------------


def test_the_partition_cannot_move_the_solved_eta():
    """It is a constant shift, and KL is a property of the normalised target."""
    u = _utilities(seed=10)
    assert V10.tpo_solve_eta(u, 0.08, **SOLVER) == V7.tpo_solve_eta(u, 0.08, **SOLVER)


def test_the_partition_is_stable_where_a_naive_mean_of_exp_would_not_be():
    u = torch.tensor([-3.0, 0.0, 3.0] * 64, dtype=torch.float64)
    eta = 0.003  # u/eta reaches 1000, past float64's exp overflow at ~709
    got = V10.tpo_log_partition(u, eta)
    assert math.isfinite(got)
    # Dominated by the top third: mean(exp(u/eta)) -> exp(1000)/3.
    assert math.isclose(got, 1000.0 - math.log(3.0), rel_tol=1e-12)
    assert not math.isfinite(float((u / eta).exp().mean()))


def test_the_partition_is_computed_in_float64_regardless_of_input_dtype():
    u = _utilities(n=512, seed=11)
    assert V10.tpo_log_partition(u.float(), 2.0) == V10.tpo_log_partition(u.float().double(), 2.0)


def test_the_partition_does_not_mutate_or_backpropagate_into_its_input():
    u = _utilities(n=128, seed=12).float().requires_grad_(True)
    before = u.detach().clone()
    V10.tpo_log_partition(u, 2.0)
    torch.testing.assert_close(u.detach(), before)
    assert u.grad is None


def test_a_zero_partition_recovers_v7_exactly():
    """The one-line diff, pinned: v10 with the correction disabled *is* v7."""
    torch.manual_seed(0)
    logratio = torch.randn(256, dtype=torch.float64) * 0.6
    utility = torch.randn(256, dtype=torch.float64) * 1.5
    for eta in (0.7, 2.0, 4.0):
        torch.testing.assert_close(
            V10.tpo_intra_loss(logratio, utility, eta, GUARD, 0.0),
            V7.tpo_intra_loss(logratio, utility, eta, GUARD),
        )


# --- everything else must be v7 verbatim --------------------------------------


def test_the_guard_still_protects_only_exp_and_never_the_restoring_term():
    u = _utilities(n=64, seed=13)
    eta = 2.0
    log_partition = V10.tpo_log_partition(u, eta)
    logratio = torch.full((64,), -60.0, dtype=torch.float64, requires_grad=True)
    V10.tpo_intra_loss(logratio, u, eta, GUARD, log_partition).sum().backward()
    # A heavily suppressed sample keeps its full restoring -w gradient.
    torch.testing.assert_close(
        logratio.grad, -_target(u, eta, log_partition), rtol=1e-9, atol=1e-12
    )
    high = torch.full((64,), 40.0, dtype=torch.float64, requires_grad=True)
    V10.tpo_intra_loss(high, u, eta, GUARD, log_partition).sum().backward()
    assert torch.isfinite(high.grad).all() and float(high.grad.min()) > 0.0


def test_utility_solver_and_floor_are_identical_to_v7():
    advantages = torch.randn(512, generator=torch.Generator().manual_seed(14)) * 7.0
    torch.testing.assert_close(V10.tpo_utility(advantages, 3.0), V7.tpo_utility(advantages, 3.0))
    assert V10.eta_floor(3.0, 10.0, 0.05) == V7.eta_floor(3.0, 10.0, 0.05)
    for realized in (0.005, 0.03, 0.2):
        assert V10.kl_budget_update(0.08, realized, 0.03, 0.3, 3.0, (1e-4, 1.0)) == (
            V7.kl_budget_update(0.08, realized, 0.03, 0.3, 3.0, (1e-4, 1.0))
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
    a10 = V10.Agent(_envs_stub())
    torch.manual_seed(7)
    a7 = V7.Agent(_envs_stub())
    p10 = dict(a10.named_parameters())
    p7 = dict(a7.named_parameters())
    assert p10.keys() == p7.keys() and len(p10) > 0
    for name in p10:
        torch.testing.assert_close(p10[name], p7[name])
    d10, d7 = a10._dist(obs), a7._dist(obs)
    torch.testing.assert_close(d10.concentration1, d7.concentration1)
    torch.testing.assert_close(d10.concentration0, d7.concentration0)
    torch.testing.assert_close(a10.get_value(obs), a7.get_value(obs))


def test_end_to_end_gradient_is_ratio_minus_normalised_target_times_dlogpi():
    torch.manual_seed(0)
    agent = V10.Agent(_envs_stub())
    obs = torch.randn(8, 5)
    with torch.no_grad():
        _, z, old_logprob, *_ = agent.get_action_and_value(obs)
        agent.actor_alpha.weight.add_(0.05)

    utility = V10.tpo_utility(torch.linspace(-3.0, 4.0, 8), utility_clip=3.0)
    eta, _ = V10.tpo_solve_eta(utility, 0.08, **SOLVER)
    log_partition = V10.tpo_log_partition(utility, eta)

    _, _, logprob, *_ = agent.get_action_and_value(obs, z)
    logratio = logprob - old_logprob
    V10.tpo_intra_loss(logratio, utility, eta, GUARD, log_partition).sum().backward()
    got = agent.actor_alpha.weight.grad.clone()

    agent.zero_grad(set_to_none=True)
    coefficient = (logratio.detach().exp() - (utility / eta - log_partition).exp()).detach()
    _, _, logprob2, *_ = agent.get_action_and_value(obs, z)
    (coefficient * logprob2).sum().backward()
    torch.testing.assert_close(got, agent.actor_alpha.weight.grad, rtol=1e-4, atol=1e-6)
    assert float(got.abs().sum()) > 0.0


def test_the_args_surface_is_v7_verbatim():
    assert V10.Args.__dataclass_fields__.keys() == V7.Args.__dataclass_fields__.keys()
    for name, field in V10.Args.__dataclass_fields__.items():
        if name == "exp_name":
            continue
        assert field.default == V7.Args.__dataclass_fields__[name].default, name

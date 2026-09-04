"""Tests for ppo_continuous_action_tpo_selfnorm_sfvlam_v11.

v11 is v23's latent-successor critic chassis with v10's self-normalised TPO actor.
The actor functions are copied verbatim from v10, so these tests concentrate on the
properties the MERGE relies on and that v10's own suite could not cover:

  * v23's policy credit is deliberately uncentred and rms-scaled, so the target must
    be invariant to both -- otherwise feeding it to a whitening utility would change
    the algorithm rather than only the clip's position.
  * v23 freezes the policy on iteration 1 by zeroing the preference direction. TPO
    has to reproduce that freeze exactly, not approximately.
  * v23's target_kl was its trust region and is now only a leash.
"""

import importlib.util
import math
import pathlib

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PATH = _ROOT / "cleanrl" / "tpo" / "ppo_continuous_action_tpo_selfnorm_sfvlam_v11.py"
_V10 = _ROOT / "cleanrl" / "tpo" / "ppo_continuous_action_tpo_intra_beta_selfnorm_v10.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = _load(_PATH, "tpo_sfvlam_v11")
SOURCE = _PATH.read_text()


def _credit(n=4096, seed=0, dtype=torch.float32):
    """A stand-in for v23's covector-dot-trace credit: uncentred and heavy-tailed."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(n, generator=g, dtype=torch.float64)
    # Skewed and offset, like a preference direction dotted into a TD trace.
    return (raw.exp() - 1.2).to(dtype)


# --------------------------------------------------------------------------------
# The defining v10 property, on v23's actual utility source
# --------------------------------------------------------------------------------


def test_the_target_ratios_average_to_exactly_one_on_uncentred_credit():
    u = m.tpo_utility(_credit(), 3.0)
    eta, _ = m.tpo_solve_eta(u, 0.08, 1.3029, 50.0, 40)
    logz = m.tpo_log_partition(u, eta)
    mean_ratio = float((u.double() / eta - logz).exp().mean())
    assert abs(mean_ratio - 1.0) < 1e-12, mean_ratio


def test_the_unnormalised_target_that_v7_used_does_not_average_to_one():
    """Non-vacuity for the test above: the correction is doing real work here."""
    u = m.tpo_utility(_credit(), 3.0)
    eta, _ = m.tpo_solve_eta(u, 0.08, 1.3029, 50.0, 40)
    naive = float((u.double() / eta).exp().mean())
    assert naive > 1.02, naive


# --------------------------------------------------------------------------------
# The header's invariance claim: v23 does not centre or whiten its credit
# --------------------------------------------------------------------------------


def _ratios(credit, clip=3.0, budget=0.08):
    u = m.tpo_utility(credit, clip)
    eta, _ = m.tpo_solve_eta(u, budget, 1.3029, 50.0, 40)
    logz = m.tpo_log_partition(u, eta)
    return (u.double() / eta - logz).exp(), eta


def test_a_shift_in_the_raw_credit_is_exactly_absorbed():
    """v23 never centres its credit; whitening must therefore be a no-op on the target.

    Exact in float64. In the float32 the training loop actually runs, subtracting a
    mean of the same magnitude as the shift costs precision, so the residual is
    cancellation error and nothing else -- both tolerances are asserted so a genuine
    regression cannot hide behind "it's just float32".
    """
    exact_a, exact_eta_a = _ratios(_credit(dtype=torch.float64))
    exact_b, exact_eta_b = _ratios(_credit(dtype=torch.float64) + 7.5)
    assert exact_eta_a == exact_eta_b
    assert float((exact_a - exact_b).abs().max()) < 1e-12

    a, eta_a = _ratios(_credit())
    b, eta_b = _ratios(_credit() + 7.5)
    assert abs(eta_a - eta_b) < 1e-6
    assert float((a - b).abs().max()) < 1e-5


def test_a_rescale_of_the_raw_credit_is_exactly_absorbed():
    """v23 divides by a rollout rms; the whitening plus dual solve absorbs that."""
    for dtype, eta_tol, ratio_tol in (
        (torch.float64, 0.0, 1e-12),
        (torch.float32, 1e-6, 1e-5),
    ):
        base = _credit(dtype=dtype)
        a, eta_a = _ratios(base)
        for c in (0.5, 2.0, 25.0):
            b, eta_b = _ratios(base * c)
            # KL(c*u, c*eta) = KL(u, eta), so the whitened utilities coincide...
            assert abs(eta_b / eta_a - 1.0) <= eta_tol, (dtype, c, eta_a, eta_b)
            # ...and therefore so does the target the actor is fitted to.
            assert float((a - b).abs().max()) <= ratio_tol, (dtype, c)


def test_the_invariance_is_a_property_of_whitening_not_of_the_solver():
    """Feeding raw uncentred credit straight to the solver is NOT scale-invariant.

    This is why v11 whitens before solving even though v23 does not: the eta bracket
    is an absolute constant, so an unwhitened utility walks the solution out of it.
    """
    base = _credit()
    solved = []
    for c in (1.0, 100.0):
        eta, _ = m.tpo_solve_eta(base * c, 0.08, 1.3029, 50.0, 40)
        solved.append(eta)
    # Both saturate at the bracket ends rather than scaling, so the KL is not held.
    assert solved[1] >= 50.0 * (1 - 1e-6) or solved[0] <= 1.3029 * (1 + 1e-6), solved


# --------------------------------------------------------------------------------
# v23's iteration-1 policy freeze must survive the actor swap
# --------------------------------------------------------------------------------


def test_the_warmup_iteration_leaves_the_policy_exactly_untouched():
    """v23 zeroes the preference direction on iteration 1 on purpose."""
    zero_credit = torch.zeros(4096)
    u = m.tpo_utility(zero_credit, 3.0)
    assert float(u.abs().max()) == 0.0
    eta, kl = m.tpo_solve_eta(u, 0.08, 1.3029, 50.0, 40)
    assert eta == 50.0 and kl == 0.0, (eta, kl)
    logz = m.tpo_log_partition(u, eta)
    assert abs(logz) < 1e-15, logz

    logratio = torch.zeros(4096, requires_grad=True)
    loss = m.tpo_intra_loss(logratio, u, eta, 20.0, logz).mean()
    loss.backward()
    assert abs(float(loss.detach())) < 1e-12, float(loss.detach())
    assert float(logratio.grad.abs().max()) < 1e-12, float(logratio.grad.abs().max())


def test_the_warmup_freeze_is_not_vacuous():
    """The same call with real credit does move the policy, so the freeze means something."""
    u = m.tpo_utility(_credit(), 3.0)
    eta, _ = m.tpo_solve_eta(u, 0.08, 1.3029, 50.0, 40)
    logz = m.tpo_log_partition(u, eta)
    logratio = torch.zeros(4096, requires_grad=True)
    m.tpo_intra_loss(logratio, u, eta, 20.0, logz).mean().backward()
    assert float(logratio.grad.abs().max()) > 1e-4


# --------------------------------------------------------------------------------
# Loss identities
# --------------------------------------------------------------------------------


def test_the_gradient_is_exactly_ratio_minus_target():
    u = m.tpo_utility(_credit(512, seed=3), 3.0)
    eta, _ = m.tpo_solve_eta(u, 0.08, 1.3029, 50.0, 40)
    logz = m.tpo_log_partition(u, eta)
    g = torch.Generator().manual_seed(11)
    logratio = (0.3 * torch.randn(512, generator=g)).requires_grad_(True)
    m.tpo_intra_loss(logratio, u, eta, 20.0, logz).sum().backward()
    expected = logratio.detach().exp() - (u / eta - logz).exp()
    assert torch.allclose(logratio.grad, expected, atol=1e-5), (
        float((logratio.grad - expected).abs().max())
    )


def test_the_fixed_point_is_the_target_and_the_loss_is_zero_there():
    u = m.tpo_utility(_credit(512, seed=4), 3.0)
    eta, _ = m.tpo_solve_eta(u, 0.08, 1.3029, 50.0, 40)
    logz = m.tpo_log_partition(u, eta)
    logratio = (u / eta - logz).clone().detach().requires_grad_(True)
    loss = m.tpo_intra_loss(logratio, u, eta, 20.0, logz)
    loss.sum().backward()
    assert float(loss.detach().abs().max()) < 1e-6
    assert float(logratio.grad.abs().max()) < 1e-5


def test_a_hugely_suppressed_sample_keeps_its_full_restoring_gradient():
    """The linear -w*logratio term is never guarded, unlike exp."""
    u = m.tpo_utility(_credit(256, seed=5), 3.0)
    eta, _ = m.tpo_solve_eta(u, 0.08, 1.3029, 50.0, 40)
    logz = m.tpo_log_partition(u, eta)
    logratio = torch.full((256,), -60.0, requires_grad=True)
    m.tpo_intra_loss(logratio, u, eta, 20.0, logz).sum().backward()
    target = (u / eta - logz).exp()
    assert torch.allclose(logratio.grad, -target, atol=1e-6)


def test_above_the_guard_the_gradient_saturates_rather_than_exploding():
    u = m.tpo_utility(_credit(256, seed=6), 3.0)
    eta, _ = m.tpo_solve_eta(u, 0.08, 1.3029, 50.0, 40)
    logz = m.tpo_log_partition(u, eta)
    logratio = torch.full((256,), 40.0, requires_grad=True)
    m.tpo_intra_loss(logratio, u, eta, 20.0, logz).sum().backward()
    assert torch.isfinite(logratio.grad).all()
    expected = math.exp(20.0)
    assert abs(float(logratio.grad.max()) / expected - 1.0) < 1e-3


# --------------------------------------------------------------------------------
# Dual solve and controller
# --------------------------------------------------------------------------------


def test_the_solved_eta_reproduces_the_requested_budget():
    u = m.tpo_utility(_credit(), 3.0)
    for budget in (0.01, 0.05, 0.2):
        eta, kl = m.tpo_solve_eta(u, budget, 1.3029, 50.0, 60)
        assert abs(kl - budget) / budget < 1e-3, (budget, kl)


def test_the_eta_floor_caps_the_per_sample_target_ratio():
    floor = m.eta_floor(3.0, 10.0, 0.05)
    assert abs(floor - 3.0 / math.log(10.0)) < 1e-12
    # At the floor the most extreme surviving sample asks for at most the cap.
    assert math.exp(3.0 / floor) <= 10.0 + 1e-9


def test_the_controller_moves_the_budget_against_the_realized_error():
    up = m.kl_budget_update(0.08, 0.01, 0.03, 0.3, 3.0, (1e-4, 1.0))
    down = m.kl_budget_update(0.08, 0.09, 0.03, 0.3, 3.0, (1e-4, 1.0))
    assert up > 0.08 > down
    # One iteration cannot slam a bound: influence is capped at ratio_clip**gain.
    assert up / 0.08 <= 3.0**0.3 + 1e-9
    assert 0.08 / down <= 3.0**0.3 + 1e-9


def test_the_controller_respects_its_bounds():
    assert m.kl_budget_update(1e-4, 10.0, 0.03, 0.3, 3.0, (1e-4, 1.0)) == 1e-4
    assert m.kl_budget_update(1.0, 1e-9, 0.03, 0.3, 3.0, (1e-4, 1.0)) == 1.0


# --------------------------------------------------------------------------------
# Merge-level wiring, asserted against the source
# --------------------------------------------------------------------------------


def test_the_actor_functions_are_byte_identical_to_v10s():
    """v11 is meant to differ from v10 in the critic only."""
    import ast

    v10_src = _V10.read_text()
    names = {
        "tpo_utility",
        "tpo_target_kl",
        "tpo_solve_eta",
        "eta_floor",
        "kl_budget_update",
        "tpo_log_partition",
        "tpo_intra_loss",
    }

    def bodies(text):
        tree = ast.parse(text)
        out = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                stripped = ast.parse(ast.unparse(node))
                fn = stripped.body[0]
                # Drop the docstring; prose was reflowed for the merged file.
                if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(
                    fn.body[0].value, ast.Constant
                ):
                    fn.body = fn.body[1:]
                out[node.name] = ast.unparse(fn.body)
        return out

    mine, theirs = bodies(SOURCE), bodies(v10_src)
    assert set(mine) == names, sorted(names - set(mine))
    for name in sorted(names):
        assert mine[name] == theirs[name], name


def test_the_policy_loss_has_no_ratio_clip_left():
    assert "clip_coef" not in SOURCE
    assert "pg_loss1" not in SOURCE and "pg_loss2" not in SOURCE
    assert "pg_loss = tpo_intra_loss(" in SOURCE


def test_the_target_is_built_once_per_rollout_not_per_minibatch():
    """Per-minibatch normalisation would silently turn this into v9's group share."""
    solve = SOURCE.index("eta, solved_kl = tpo_solve_eta(")
    partition = SOURCE.index("log_partition = tpo_log_partition(b_utilities, eta)")
    epoch = SOURCE.index("for epoch in range(args.update_epochs):")
    assert solve < partition < epoch
    assert "tpo_log_partition(b_utilities, eta)" in SOURCE
    # The utility comes from the rollout-scope credit, not a minibatch slice.
    assert "b_utilities = tpo_utility(b_policy_credit, args.utility_clip)" in SOURCE


def test_the_solve_precedes_the_partition_which_depends_on_it():
    """log_partition is a shift in log-ratio space and must not feed back into eta."""
    import ast

    tree = ast.parse(SOURCE)
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"tpo_solve_eta", "tpo_log_partition"}
    ]
    assert calls == ["tpo_solve_eta", "tpo_log_partition"], calls


def test_the_leash_is_looser_than_the_controller_target():
    """A leash at the controller's own target would cut every update and hide its error."""
    args = m.Args()
    assert args.target_kl > args.realized_kl_target
    assert args.target_kl >= 3.0 * args.realized_kl_target


def test_the_controller_skips_the_frozen_warmup_iteration():
    update = SOURCE.index("kl_budget = kl_budget_update(")
    # The nearest preceding guard, not the earlier one that gates policy_credit itself.
    guard = SOURCE.rindex("if preference_ready:", 0, update)
    between = SOURCE[guard + len("if preference_ready:") : update]
    # Only comments and whitespace may sit between the guard and the update, so the
    # update is unconditionally inside it.
    assert all(
        not line.strip() or line.strip().startswith("#") for line in between.splitlines()
    ), between


def test_the_realized_kl_estimate_survives_a_truncated_final_epoch():
    assert "min(args.num_minibatches, len(approx_kls))" in SOURCE


def test_the_critic_chassis_is_untouched():
    """The successor-feature critic is the whole point of the merge."""
    for marker in (
        "class VectorPreferenceField",
        "class LeJepaSSL",
        "successor_lambda_residual",
        "args.vf_coef * v_loss",
        "critic_grad_clip",
    ):
        assert marker in SOURCE, marker


def test_the_defaults_are_the_benchmark_configuration():
    args = m.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1
    assert args.actor_dist == "beta"
    assert args.ent_coef == 0.0
    assert not args.auto_entropy


# --------------------------------------------------------------------------------
# Numerics at the shipped defaults
# --------------------------------------------------------------------------------


def test_no_overflow_or_underflow_anywhere_in_the_bracket():
    u = m.tpo_utility(_credit(32768, seed=9), 3.0)
    for eta in (m.eta_floor(3.0, 10.0, 0.05), 2.5, 50.0):
        logz = m.tpo_log_partition(u, eta)
        w = (u / eta - logz).exp()
        assert torch.isfinite(w).all()
        # exp is never asked for more than the cap, and never underflows to zero.
        assert float(w.max()) <= 10.0 + 1e-6, (eta, float(w.max()))
        assert float(w.min()) > 0.0, eta


def test_the_partition_is_computed_in_float64():
    u = m.tpo_utility(_credit(32768, seed=10), 3.0)
    eta = 1.3029
    exact = float(
        torch.logsumexp(u.double() / eta, 0) - math.log(u.numel())
    )
    assert abs(m.tpo_log_partition(u, eta) - exact) < 1e-12


def test_the_partition_is_stable_where_a_naive_mean_of_exp_would_not_be():
    u = torch.tensor([900.0, 0.0, -900.0])
    assert math.isfinite(m.tpo_log_partition(u, 1.0))
    assert not math.isfinite(float((u.double() / 1.0).exp().mean()))

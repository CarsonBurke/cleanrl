"""Tests for ppo_continuous_action_tpo_intra_beta_trajcenter_v12.

v12 is v10 with exactly one change: the utility subtracts each env-column's own
advantage mean instead of the rollout's. The whole claim is that this is a UTILITY
TRANSFORMATION and not a change of group, so these tests split into three parts:

  * the change does what it says (per-column centring, correct fraction of variance
    removed, and demonstrably different from global centring when the columns
    differ -- and demonstrably identical when they do not);
  * the change is aligned with the samples it labels, which is the one way to
    corrupt a run silently rather than loudly;
  * nothing else moved -- eta and the log-partition are still batch-wide, still
    solved once per rollout, the target still averages to exactly 1, and
    traj_center=False is v10 bit-for-bit.

Every assertion is paired with a control that shows it could have failed.
"""

import ast
import importlib.util
import math
import pathlib

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PATH = _ROOT / "cleanrl" / "tpo" / "ppo_continuous_action_tpo_intra_beta_trajcenter_v12.py"
_V10 = _ROOT / "cleanrl" / "tpo" / "ppo_continuous_action_tpo_intra_beta_selfnorm_v10.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = _load(_PATH, "tpo_trajcenter_v12")
v10 = _load(_V10, "tpo_selfnorm_v10")
SOURCE = _PATH.read_text()

# The shipped configuration: 128 steps x 16 envs = 2048, clip 3, eta floor 3/log(10).
STEPS, ENVS = 128, 16
CLIP = 3.0
FLOOR = 3.0 / math.log(10.0)
BUDGET = 0.08


def _rollout(offsets=None, seed=0, steps=STEPS, envs=ENVS, dtype=torch.float64):
    """A (num_steps, num_envs) advantage block with a per-column offset.

    ``offsets`` stands in for the per-env critic bias v12 exists to remove; None
    means iid columns with no differential shift.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(steps, envs, generator=g, dtype=torch.float64)
    if offsets is not None:
        a = a + torch.as_tensor(offsets, dtype=torch.float64).view(1, envs)
    return a.to(dtype)


def _target(u, budget=BUDGET):
    eta, _ = m.tpo_solve_eta(u, budget, FLOOR, m.Args().eta_max, 40)
    logz = m.tpo_log_partition(u, eta)
    return (u.double() / eta - logz).exp(), eta, logz


# --------------------------------------------------------------------------------
# The change itself
# --------------------------------------------------------------------------------


def test_every_column_of_the_centred_utility_has_zero_mean():
    """The defining property, read back in the original 2-D layout."""
    a = _rollout(offsets=[2.0 * i - 15.0 for i in range(ENVS)])
    # No clip, so the linear identity is not truncated.
    u = m.tpo_utility_trajectory(a, None, True).view(STEPS, ENVS)
    assert float(u.mean(dim=0).abs().max()) < 1e-12, float(u.mean(dim=0).abs().max())


def test_globally_centred_columns_do_not_have_zero_mean_control():
    """Control for the test above: v10's utility fails it badly on the same input."""
    a = _rollout(offsets=[2.0 * i - 15.0 for i in range(ENVS)])
    u = v10.tpo_utility(a.reshape(-1), None).view(STEPS, ENVS)
    assert float(u.mean(dim=0).abs().max()) > 0.5, float(u.mean(dim=0).abs().max())


def test_per_trajectory_centring_demonstrably_changes_the_target():
    """With differential offsets, v12's target is a different object from v10's."""
    a = _rollout(offsets=[3.0 * math.sin(i) for i in range(ENVS)], seed=1)
    w12, eta12, _ = _target(m.tpo_utility_trajectory(a, CLIP, True))
    w10, eta10, _ = _target(v10.tpo_utility(a.reshape(-1), CLIP))
    assert float((w12 - w10).abs().max()) > 0.25, float((w12 - w10).abs().max())
    # And the change is not a relabelling: the ranking of samples actually moves.
    assert not torch.equal(w12.argsort(), w10.argsort())
    assert eta12 > 0 and eta10 > 0


def test_it_provably_cannot_change_anything_when_the_column_means_agree():
    """Identical per-column means => per-column centring == global centring.

    Constructed by explicitly removing each column's mean first, so the two centrings
    differ by the zero vector. This is the boundary case the header warns about: if
    the live tpo/traj_mean_spread is ~0, the run is exactly this.
    """
    a = _rollout(seed=2)
    a = a - a.mean(dim=0, keepdim=True)  # all column means now exactly 0
    assert float(a.mean(dim=0).abs().max()) < 1e-13
    u12 = m.tpo_utility_trajectory(a, CLIP, True)
    u10 = v10.tpo_utility(a.reshape(-1), CLIP)
    assert float((u12 - u10).abs().max()) < 1e-12, float((u12 - u10).abs().max())
    w12, eta12, z12 = _target(u12)
    w10, eta10, z10 = _target(u10)
    assert eta12 == eta10 and z12 == z10
    assert float((w12 - w10).abs().max()) < 1e-12


def test_a_common_shift_across_all_columns_is_absorbed_by_both_centrings():
    """Global level is not what v12 removes; only the differential part is.

    Non-discriminating by construction -- v10 passes it too -- and that is the point:
    it pins down which half of the header's motivation is load-bearing.
    """
    a = _rollout(seed=3)
    for shift in (0.0, 7.5, -100.0):
        u = m.tpo_utility_trajectory(a + shift, CLIP, True)
        base = m.tpo_utility_trajectory(a, CLIP, True)
        assert float((u - base).abs().max()) < 1e-12, shift


def test_v10s_unit_variance_whitening_survives_the_new_centring():
    """Not a v12-specific property: v10 has it too, and v12 must not lose it.

    The eta bracket, utility_clip and the max_target_ratio floor are all calibrated
    to Var(u) = 1, so this is the invariant the new centring could most easily break
    by dividing by the wrong std.
    """
    a = _rollout(offsets=[5.0 * i for i in range(ENVS)], seed=4)
    pre_clip = m.tpo_utility_trajectory(a, None, True)
    assert abs(float(pre_clip.mean())) < 1e-12
    assert abs(float(pre_clip.std(unbiased=False)) - 1.0) < 1e-12


def test_the_divisor_is_batch_wide_not_per_column():
    """A per-column std would equalise column scales; the batch-wide one must not."""
    scales = torch.tensor([0.1] * (ENVS // 2) + [10.0] * (ENVS // 2), dtype=torch.float64)
    a = _rollout(seed=5) * scales.view(1, ENVS)
    u = m.tpo_utility_trajectory(a, None, True).view(STEPS, ENVS)
    col_std = u.std(dim=0, unbiased=False)
    # Per-column whitening would give every column std 1; batch-wide preserves the
    # 100x scale difference the rollout actually had.
    assert float(col_std.max() / col_std.min()) > 50.0, float(col_std.max() / col_std.min())


# --------------------------------------------------------------------------------
# Alignment: the silent-corruption failure mode
# --------------------------------------------------------------------------------


def test_utilities_stay_aligned_with_the_samples_they_label():
    """Utility k must describe advantage k of `advantages.reshape(-1)`.

    Tagged with a unique per-cell value so any transpose, column-major flatten, or
    off-by-one in the centring shows up as a mismatch rather than as a plausible
    number. Utility is a strictly increasing function of (A - column mean) within a
    column, so the check is: recover the column index from the flat position and
    confirm the utility matches that column's own centring.
    """
    tags = torch.arange(STEPS * ENVS, dtype=torch.float64).view(STEPS, ENVS)
    a = tags / 100.0
    u = m.tpo_utility_trajectory(a, None, True)
    flat = a.reshape(-1)
    assert flat.shape == u.shape
    # Reconstruct expected utility directly from the flat index, independently of
    # any 2-D op: row-major flatten means index k lives in column k % ENVS.
    cols = torch.arange(STEPS * ENVS) % ENVS
    col_means = a.mean(dim=0)
    expected = flat - col_means[cols]
    expected = expected / expected.std(unbiased=False)
    assert float((u - expected).abs().max()) < 1e-10, float((u - expected).abs().max())


def test_the_alignment_check_catches_a_transposed_flatten_control():
    """Control: the same check applied to a column-major flatten must fail."""
    tags = torch.arange(STEPS * ENVS, dtype=torch.float64).view(STEPS, ENVS)
    a = tags / 100.0
    wrong = (a - a.mean(dim=0, keepdim=True)).t().reshape(-1)  # column-major
    wrong = wrong / wrong.std(unbiased=False)
    cols = torch.arange(STEPS * ENVS) % ENVS
    expected = a.reshape(-1) - a.mean(dim=0)[cols]
    expected = expected / expected.std(unbiased=False)
    assert float((wrong - expected).abs().max()) > 1.0


def test_the_training_loop_builds_the_utility_from_the_unflattened_advantages():
    """Centring after the reshape would centre 128-sample stripes across all envs."""
    assert "b_utilities = tpo_utility_trajectory(advantages, args.utility_clip, args.traj_center)" in SOURCE
    # And nothing reintroduces a pre-flattened advantage vector to centre by mistake.
    assert "b_advantages = advantages.reshape(-1)" not in SOURCE


def test_the_rest_of_the_batch_is_flattened_the_same_row_major_way():
    """b_obs/b_zs/b_logprobs must use the identical (T, N) -> row-major reshape."""
    for marker in (
        "b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)",
        "b_zs = zs.reshape((-1,) + envs.single_action_space.shape)",
        "b_logprobs = logprobs.reshape(-1)",
        "b_returns = returns.reshape(-1)",
        "b_values = values.reshape(-1)",
    ):
        assert marker in SOURCE, marker
    assert ".t()" not in SOURCE and ".transpose(" not in SOURCE and ".permute(" not in SOURCE


# --------------------------------------------------------------------------------
# The control flag is an exact v10, not an approximate one
# --------------------------------------------------------------------------------


def test_traj_center_false_reproduces_v10s_utility_bit_for_bit():
    for seed, offsets in ((6, None), (7, [1.5 * i for i in range(ENVS)])):
        for dtype in (torch.float32, torch.float64):
            a = _rollout(offsets=offsets, seed=seed, dtype=dtype)
            mine = m.tpo_utility_trajectory(a, CLIP, False)
            theirs = v10.tpo_utility(a.reshape(-1), CLIP)
            assert torch.equal(mine, theirs), (seed, dtype)


def test_traj_center_true_is_not_bit_identical_to_v10_control():
    """Non-vacuity for the test above."""
    a = _rollout(offsets=[1.5 * i for i in range(ENVS)], seed=7)
    assert not torch.equal(
        m.tpo_utility_trajectory(a, CLIP, True), v10.tpo_utility(a.reshape(-1), CLIP)
    )


def test_tpo_utility_itself_is_unchanged_from_v10():
    """v12 must not have edited the shared path the control flag routes through."""
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
        out = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                fn = ast.parse(ast.unparse(node)).body[0]
                if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(
                    fn.body[0].value, ast.Constant
                ):
                    fn.body = fn.body[1:]
                out[node.name] = ast.unparse(fn.body)
        return out

    mine, theirs = bodies(SOURCE), bodies(_V10.read_text())
    assert set(mine) == names, sorted(names - set(mine))
    for name in sorted(names):
        assert mine[name] == theirs[name], name


def test_the_defaults_are_the_v12_configuration():
    args = m.Args()
    assert args.traj_center is True
    assert args.env_id == "HalfCheetah-v4"
    assert args.seed == 1
    assert args.num_steps == STEPS and args.num_envs == ENVS
    assert args.utility_clip == CLIP
    assert args.num_minibatches == 32 and args.update_epochs == 10
    assert args.ent_coef == 0.0 and args.max_grad_norm == 0.5


# --------------------------------------------------------------------------------
# The invariants: same group, one solve, one partition, fixed target
# --------------------------------------------------------------------------------


def test_v10s_self_normalisation_still_holds_on_the_new_utilities():
    a = _rollout(offsets=[4.0 * math.cos(i) for i in range(ENVS)], seed=8)
    w, _, _ = _target(m.tpo_utility_trajectory(a, CLIP, True))
    assert abs(float(w.mean()) - 1.0) < 1e-12, float(w.mean())


def test_the_unnormalised_target_would_not_average_to_one_control():
    a = _rollout(offsets=[4.0 * math.cos(i) for i in range(ENVS)], seed=8)
    u = m.tpo_utility_trajectory(a, CLIP, True)
    eta, _ = m.tpo_solve_eta(u, BUDGET, FLOOR, 50.0, 40)
    assert float((u.double() / eta).exp().mean()) > 1.02


def test_v10s_dual_solve_still_hits_the_budget_on_the_new_utilities():
    u = m.tpo_utility_trajectory(_rollout(offsets=range(ENVS), seed=9), CLIP, True)
    for budget in (0.01, 0.05, 0.2):
        eta, kl = m.tpo_solve_eta(u, budget, FLOOR, 50.0, 60)
        assert abs(kl - budget) / budget < 1e-3, (budget, kl)


def test_the_eta_solve_sees_the_whole_flattened_rollout_not_a_column():
    """Batch-wide group. A per-column solve would be a different algorithm."""
    tree = ast.parse(SOURCE)
    solves = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tpo_solve_eta"
    ]
    assert len(solves) == 1, len(solves)
    assert isinstance(solves[0].args[0], ast.Name) and solves[0].args[0].id == "b_utilities"


def test_the_partition_is_batch_wide_and_computed_once_after_the_solve():
    tree = ast.parse(SOURCE)
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"tpo_solve_eta", "tpo_log_partition"}
    ]
    assert calls == ["tpo_solve_eta", "tpo_log_partition"], calls
    assert "log_partition = tpo_log_partition(b_utilities, eta)" in SOURCE


def test_nothing_that_defines_the_target_lives_inside_the_epoch_loop():
    """v9 recomputed the normalisation per minibatch and collapsed (1028 vs 3331)."""
    tree = ast.parse(SOURCE)
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.If))
    epoch_loop = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "epoch"
    )
    forbidden = {
        "tpo_solve_eta",
        "tpo_log_partition",
        "tpo_utility",
        "tpo_utility_trajectory",
        "trajectory_mean_spread",
    }
    inner = {
        n.func.id
        for n in ast.walk(epoch_loop)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert not (inner & forbidden), sorted(inner & forbidden)
    # Non-vacuity: the loop body is not empty of calls we could have caught.
    assert "tpo_intra_loss" in inner

    # ...and the ordering: utility, solve, partition all precede the epoch loop.
    order = [
        SOURCE.index("b_utilities = tpo_utility_trajectory("),
        SOURCE.index("eta, solved_kl = tpo_solve_eta("),
        SOURCE.index("log_partition = tpo_log_partition(b_utilities, eta)"),
        SOURCE.index("for epoch in range(args.update_epochs):"),
    ]
    assert order == sorted(order), order


def test_the_target_a_minibatch_sees_is_a_slice_of_the_fixed_one():
    """mb_utilities must be an index into b_utilities, never a recomputation."""
    assert "mb_utilities = b_utilities[mb_inds]" in SOURCE
    assert "tpo_utility" not in SOURCE.split("for epoch in range(args.update_epochs):")[1]


# --------------------------------------------------------------------------------
# The diagnostic
# --------------------------------------------------------------------------------


def test_the_spread_equals_the_fraction_of_variance_centring_removes():
    """Law of total variance, asserted as the identity the header claims."""
    a = _rollout(offsets=[2.0 * i - 15.0 for i in range(ENVS)], seed=10)
    spread = m.trajectory_mean_spread(a)
    within = float((a - a.mean(dim=0, keepdim=True)).std(unbiased=False))
    total = float(a.std(unbiased=False))
    assert abs(spread**2 - (1.0 - (within / total) ** 2)) < 1e-10


def test_the_spread_is_near_zero_exactly_when_the_variant_is_a_no_op():
    a = _rollout(seed=11)
    a = a - a.mean(dim=0, keepdim=True)
    assert m.trajectory_mean_spread(a) < 1e-12
    # ...and large when the columns really are differentially shifted.
    # ...and it saturates towards its ceiling when the columns really are
    # differentially shifted. The ceiling is exactly 1: the quantity is the square
    # root of a variance *fraction*, so tpo/traj_mean_spread_z cannot exceed
    # sqrt(num_steps) = 11.3 however biased the critic is.
    shifted = a + 5.0 * torch.arange(ENVS, dtype=torch.float64).view(1, ENVS)
    assert 0.95 < m.trajectory_mean_spread(shifted) <= 1.0


def test_the_iid_null_spread_sits_at_one_over_sqrt_num_steps():
    """The number the live log must be compared against to mean anything."""
    null = m.trajectory_spread_null(STEPS, ENVS)
    zs = [m.trajectory_mean_spread(_rollout(seed=100 + i)) / null for i in range(200)]
    mean_z = sum(zs) / len(zs)
    # SE of the mean over 200 rollouts is ~0.013, so this band has real content: the
    # 3% error from dropping (C-1)/C at 16 envs is ~2 SE and the 22% error at 2 envs
    # is enormous. Verified: the naive 1/sqrt(T) null gives mean_z 0.96 and fails.
    assert 0.96 < mean_z < 1.04, mean_z
    # And the null itself is the (C-1)/C form, not 1/sqrt(num_steps).
    assert abs(null - math.sqrt((ENVS - 1) / (ENVS * STEPS))) < 1e-15
    assert null > 1.0 / math.sqrt(STEPS) * 0.9 and null != 1.0 / math.sqrt(STEPS)


def test_the_spread_is_degenerate_safe():
    assert m.trajectory_mean_spread(torch.zeros(STEPS, ENVS)) == 0.0
    assert m.trajectory_mean_spread(torch.full((STEPS, ENVS), 3.0)) == 0.0


def test_the_diagnostics_are_logged_and_the_z_form_uses_the_corrected_null():
    assert 'writer.add_scalar("tpo/traj_mean_spread", traj_spread, global_step)' in SOURCE
    assert 'writer.add_scalar("tpo/traj_mean_spread_z", traj_spread / spread_null, global_step)' in SOURCE
    assert "spread_null = trajectory_spread_null(args.num_steps, args.num_envs)" in SOURCE
    # The naive null must not be what is shipped.
    assert "math.sqrt(args.num_steps)" not in SOURCE
    # Computed unconditionally, so the --no-traj-center control reports it too.
    assert "traj_spread = trajectory_mean_spread(advantages)" in SOURCE
    assert 'writer.add_scalar(\n                "tpo/utility_clip_frac",' in SOURCE


def test_the_null_is_computed_once_outside_the_iteration_loop():
    """A per-iteration constant would be pure waste and could drift with args."""
    setup = SOURCE.index("spread_null = trajectory_spread_null(")
    loop = SOURCE.index("for iteration in range(1, args.num_iterations + 1):")
    assert setup < loop


def test_the_null_rejects_configurations_where_it_is_undefined():
    for bad in ((STEPS, 1), (STEPS, 0), (0, ENVS)):
        try:
            m.trajectory_spread_null(*bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_the_clip_fraction_diagnostic_measures_the_amplification_it_claims():
    """Re-whitening after removing between-column variance makes the clip bind more."""
    a = _rollout(offsets=[1.6 * i - 12.0 for i in range(ENVS)], seed=15)
    spread = m.trajectory_mean_spread(a)
    assert spread > 0.6, spread  # a regime where the effect should be visible

    def frac(u):
        return float((u.abs() >= CLIP * (1 - 1e-6)).double().mean())

    f12 = frac(m.tpo_utility_trajectory(a, CLIP, True))
    f10 = frac(v10.tpo_utility(a.reshape(-1), CLIP))
    assert f12 > f10, (f12, f10)
    # The within-column signal is amplified by 1/sqrt(1 - spread^2); check the scale
    # factor directly, since that is the mechanism the header names.
    within = float((a - a.mean(dim=0, keepdim=True)).std(unbiased=False))
    total = float(a.std(unbiased=False))
    assert abs(total / within - 1.0 / math.sqrt(1.0 - spread**2)) < 1e-9


def test_a_non_finite_rollout_does_not_log_a_clean_zero_spread():
    """0.0 is the 'nothing to remove' reading and must not come from a broken rollout."""
    a = _rollout(seed=16)
    a[3, 2] = float("nan")
    assert math.isnan(m.trajectory_mean_spread(a))
    # Control: the finite version of the same rollout reports a real number.
    assert m.trajectory_mean_spread(_rollout(seed=16)) > 0.0


# --------------------------------------------------------------------------------
# Numerical safety at the shipped defaults
# --------------------------------------------------------------------------------


def test_no_overflow_or_underflow_across_the_eta_bracket():
    a = _rollout(offsets=[6.0 * i for i in range(ENVS)], seed=12, dtype=torch.float32)
    u = m.tpo_utility_trajectory(a, CLIP, True)
    assert torch.isfinite(u).all()
    for eta in (FLOOR, 2.5, 50.0):
        logz = m.tpo_log_partition(u, eta)
        w = (u.double() / eta - logz).exp()
        assert torch.isfinite(w).all(), eta
        assert float(w.max()) <= m.Args().max_target_ratio + 1e-6, (eta, float(w.max()))
        assert float(w.min()) > 0.0, eta


def test_a_degenerate_rollout_makes_the_actor_exactly_neutral():
    """Zero advantage variance must give zero utility, not NaN."""
    for a in (torch.zeros(STEPS, ENVS), torch.full((STEPS, ENVS), -2.0)):
        u = m.tpo_utility_trajectory(a, CLIP, True)
        assert torch.isfinite(u).all()
        assert float(u.abs().max()) == 0.0
        eta, kl = m.tpo_solve_eta(u, BUDGET, FLOOR, 50.0, 40)
        assert eta == 50.0 and kl == 0.0
        assert abs(m.tpo_log_partition(u, eta)) < 1e-15


def test_a_single_huge_outlier_stays_finite_and_near_the_cap():
    """The max_target_ratio floor bounds exp(u_max/eta), not the self-normalised w.

    v10's header asserts w_max = exp(u_max/eta - logZ) <= exp(u_max/eta) "since the
    floor is still a valid bound". That needs logZ >= 0, which Jensen gives only when
    the POST-CLIP utility mean is >= 0. An outlier that the clip truncates hard on the
    positive side leaves a slightly negative mean, logZ goes negative, and the cap is
    exceeded by exp(-logZ). This is inherited from v10, not introduced here -- the
    control below shows v10 overshoots by MORE on the same rollout -- so the test
    asserts the exact mechanism and bounds the overshoot rather than pretending the
    documented cap is tight.
    """
    a = _rollout(seed=13, dtype=torch.float32)
    a[0, 0] = 3e4
    u = m.tpo_utility_trajectory(a, CLIP, True)
    assert torch.isfinite(u).all() and float(u.abs().max()) <= CLIP
    w, eta, logz = _target(u)
    assert torch.isfinite(w).all() and float(w.min()) > 0.0
    assert eta >= FLOOR * (1 - 1e-9)
    cap = m.Args().max_target_ratio
    # The overshoot is exactly the self-normalisation factor, and it is small.
    assert abs(float(w.max()) - cap * math.exp(-logz)) < 1e-6 * cap
    assert float(w.max()) < 1.05 * cap, float(w.max())

    # Control: v10's global centring overshoots the same cap by more, so this is a
    # property of the clip plus self-normalisation, not of per-trajectory centring.
    u10 = v10.tpo_utility(a.reshape(-1), CLIP)
    w10, _, _ = _target(u10)
    assert float(w10.max()) > float(w.max()) > cap


def test_the_gradient_through_the_loss_is_still_ratio_minus_target():
    """The loss path is untouched, but assert it end-to-end on v12's utilities."""
    a = _rollout(offsets=[2.0 * i for i in range(ENVS)], seed=14, dtype=torch.float32)
    u = m.tpo_utility_trajectory(a, CLIP, True)
    _, eta, logz = _target(u)
    g = torch.Generator().manual_seed(21)
    logratio = (0.3 * torch.randn(u.numel(), generator=g)).requires_grad_(True)
    m.tpo_intra_loss(logratio, u, eta, 20.0, logz).sum().backward()
    expected = logratio.detach().exp() - (u / eta - logz).exp()
    assert torch.allclose(logratio.grad, expected, atol=1e-5)


def test_a_one_step_rollout_is_rejected_rather_than_silently_freezing_the_actor():
    """num_steps=1 centres every column to exactly 0 -- a no-op run with no error.

    The control below shows how quiet the failure would be: constant utilities make
    the solver return eta_max with KL 0 and every target ratio exactly 1, so the run
    trains forever with nothing to report.
    """
    a = _rollout(seed=17, steps=1)
    try:
        m.tpo_utility_trajectory(a, CLIP, True)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a 1-step rollout under traj_center")
    # It is still a legal v10 rollout, so the control flag must not reject it.
    assert torch.equal(
        m.tpo_utility_trajectory(a, CLIP, False), v10.tpo_utility(a.reshape(-1), CLIP)
    )
    # Control for the docstring's claim about how silent the failure would have been.
    degenerate = (a - a.mean(dim=0, keepdim=True)).reshape(-1)
    assert float(degenerate.abs().max()) == 0.0
    eta, kl = m.tpo_solve_eta(degenerate, BUDGET, FLOOR, 50.0, 40)
    assert eta == 50.0 and kl == 0.0


def test_a_shape_that_is_not_a_rollout_is_rejected_rather_than_silently_centred():
    """Passing an already-flattened batch would centre nothing and look fine."""
    for bad in (torch.randn(STEPS * ENVS), torch.randn(2, STEPS, ENVS)):
        try:
            m.tpo_utility_trajectory(bad, CLIP, True)
        except ValueError:
            continue
        raise AssertionError(f"accepted shape {tuple(bad.shape)}")


def test_the_header_documents_the_episode_boundary_limitation():
    """The caveat is load-bearing for anyone reading the result; it must be stated."""
    head = SOURCE.split("import math")[0]
    assert "Episode boundaries" in head
    assert "dones" in head

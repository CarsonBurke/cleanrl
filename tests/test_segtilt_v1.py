"""Contract tests for segment-tilted PPO.

Load-bearing claims: the multiplier depends only on a SEGMENT's rank, has mean exactly 1
so it cannot rescale the update, is identically 1 at T=1 (making PPO a member rather than
a control), and keeps every timestep inside a segment on the same weight.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "cleanrl" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sg = _load("segtilt", "ppo_continuous_action_segtilt_v1.py")
base = _load("ppo_baseline", "ppo_continuous_action.py")

STEPS, ENVS, L = 1024, 16, 64


def advantages(seed=0, steps=STEPS, envs=ENVS, heavy=False):
    torch.manual_seed(seed)
    x = torch.randn(steps, envs, dtype=torch.float64)
    if heavy:  # the kurtosis-12-150 regime the runs actually log
        chi = torch.distributions.Chi2(torch.tensor(3.0, dtype=torch.float64)).sample((steps, envs))
        x = x / (chi / 3).sqrt()
    return x


# ---------------------------------------------------------------- PPO is the T=1 member

def test_order_one_multiplier_is_identically_one():
    m = sg.segment_multiplier(advantages(), L, 1.0, 1.0)
    assert torch.equal(m, torch.ones_like(m))


def test_the_trainer_skips_ranking_entirely_at_order_one():
    """Not merely equal to 1 -- not computed at all, so T=1 cannot differ from baseline
    PPO by so much as a rounding mode."""
    source = (_ROOT / "cleanrl" / "ppo_continuous_action_segtilt_v1.py").read_text()
    assert "if args.tilt_order <= 1.0:" in source
    assert "multiplier = None" in source


def test_order_one_leaves_advantages_bit_identical():
    a = advantages()
    assert torch.equal(a * sg.segment_multiplier(a, L, 1.0, 1.0), a)


# ------------------------------------------------------- the tilt cannot rescale anything

@pytest.mark.parametrize("order", [1.0, 2.0, 4.0, 8.0])
@pytest.mark.parametrize("heavy", [False, True])
def test_multiplier_has_mean_exactly_one(order, heavy):
    """A tilt must move gradient BETWEEN segments, never change the total. This is the
    guard against the defect that turned an earlier sweep into a learning-rate sweep."""
    m = sg.segment_multiplier(advantages(heavy=heavy), L, order, 1.0)
    assert float(m.mean()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("order", [2.0, 4.0])
def test_multiplier_is_invariant_to_the_advantage_scale_and_offset(order):
    """Ranks are all that enter, so a drifting reward-normaliser scale cannot reach the
    policy through this term."""
    a = advantages()
    reference = sg.segment_multiplier(a, L, order, 1.0)
    for factor, shift in ((0.01, 0.0), (250.0, 0.0), (1.0, 17.0), (3.0, -4.0)):
        assert torch.allclose(sg.segment_multiplier(a * factor + shift, L, order, 1.0),
                              reference, atol=1e-12)


@pytest.mark.parametrize("order", [2.0, 4.0, 8.0])
def test_multiplier_is_bounded_by_the_order(order):
    m = sg.segment_multiplier(advantages(), L, order, 1.0)
    assert float(m.min()) >= 0.0
    assert float(m.max()) <= order * 1.02


# --------------------------------------------------------------- it is a segment statistic

@pytest.mark.parametrize("order", [2.0, 4.0])
def test_every_timestep_in_a_segment_shares_one_weight(order):
    """The whole point: the graded unit is the generation, not the timestep."""
    m = sg.segment_multiplier(advantages(), L, order, 1.0)
    grouped = m.view(STEPS // L, L, ENVS)
    assert float((grouped - grouped[:, :1, :]).abs().max()) == 0.0


@pytest.mark.parametrize("order", [2.0, 4.0])
def test_weight_is_monotone_in_the_segment_score(order):
    a = advantages()
    m = sg.segment_multiplier(a, L, order, 1.0)
    scores = a.view(STEPS // L, L, ENVS).mean(dim=1).flatten()
    weights = m.view(STEPS // L, L, ENVS)[:, 0, :].flatten()
    ordering = scores.argsort()
    assert torch.all(weights[ordering].diff() >= -1e-12)


def test_reordering_timesteps_inside_a_segment_cannot_change_its_grade():
    """The grade is the segment's outcome, so it must not depend on the order of events
    within it -- otherwise it is a timestep statistic wearing a segment's name."""
    a = advantages()
    shuffled = a.clone()
    torch.manual_seed(3)
    for seg in range(STEPS // L):
        block = shuffled[seg * L:(seg + 1) * L]
        shuffled[seg * L:(seg + 1) * L] = block[torch.randperm(L)]
    assert torch.allclose(sg.segment_multiplier(a, L, 4.0, 1.0),
                          sg.segment_multiplier(shuffled, L, 4.0, 1.0), atol=1e-12)


def test_segments_do_not_bleed_across_envs():
    """`view(nseg, L, envs)` must group along time within an env. If the reshape were
    wrong, one env's advantages would set another's weight."""
    a = torch.zeros(STEPS, ENVS, dtype=torch.float64)
    a[:, 0] = 100.0          # env 0 is uniformly excellent
    m = sg.segment_multiplier(a, L, 4.0, 1.0)
    assert float(m[:, 0].min()) > float(m[:, 1:].max())


# ---------------------------------------------------------------- the falsification arm

@pytest.mark.parametrize("order", [2.0, 4.0])
def test_lower_side_tilts_toward_the_worst_segments(order):
    a = advantages()
    scores = a.view(STEPS // L, L, ENVS).mean(dim=1).flatten()
    upper = sg.segment_multiplier(a, L, order, 1.0).view(STEPS // L, L, ENVS)[:, 0, :].flatten()
    lower = sg.segment_multiplier(a, L, order, -1.0).view(STEPS // L, L, ENVS)[:, 0, :].flatten()
    best, worst = int(scores.argmax()), int(scores.argmin())
    assert upper[best] > upper[worst] and lower[worst] > lower[best]
    # Same family, opposite direction: the ranks are exactly reversed.
    assert torch.allclose(upper.sort().values, lower.sort().values, atol=1e-12)


def test_both_sides_coincide_at_order_one():
    a = advantages()
    assert torch.equal(sg.segment_multiplier(a, L, 1.0, 1.0),
                       sg.segment_multiplier(a, L, 1.0, -1.0))


# ----------------------------------------------------- effective sample size stays sane

@pytest.mark.parametrize("order,floor", [(2.0, 0.70), (4.0, 0.40), (8.0, 0.20)])
def test_effective_sample_size_of_the_tilt_stays_high(order, floor):
    """The number that condemned bestof_v2: its per-timestep weights had an effective
    sample size of ~42 out of 32768 at the kurtosis these runs log. Here the multiplier is
    a bounded mean-1 function of a segment's RANK, so its ESS is a property of T alone and
    cannot collapse with the advantage distribution -- checked on a t(3) law too.
    """
    for heavy in (False, True):
        m = sg.segment_multiplier(advantages(heavy=heavy), L, order, 1.0).view(
            STEPS // L, L, ENVS)[:, 0, :].flatten()
        ess = float(m.sum() ** 2 / (m.numel() * (m ** 2).sum()))
        assert ess >= floor, (order, heavy, ess)


def test_segment_length_sets_how_many_generations_are_graded():
    for length, expected in ((16, 1024), (64, 256), (256, 64)):
        m = sg.segment_multiplier(advantages(), length, 4.0, 1.0)
        distinct = m.view(STEPS // length, length, ENVS)[:, 0, :].numel()
        assert distinct == expected


# ------------------------------------------------------------------------ argument guard

def test_validate_args_rejects_a_segment_length_that_does_not_divide_the_rollout():
    for bad in (0, -8, 100, 1025):
        args = sg.Args()
        args.segment_len = bad
        with pytest.raises(ValueError, match="segment_len"):
            sg.validate_args(args)


def test_validate_args_rejects_bad_order_and_side():
    args = sg.Args(); args.tilt_order = 0.5
    with pytest.raises(ValueError, match="tilt_order"):
        sg.validate_args(args)
    args = sg.Args(); args.tilt_side = "sideways"
    with pytest.raises(ValueError, match="tilt_side"):
        sg.validate_args(args)


def test_validate_args_accepts_the_whole_sweep():
    for order in (1.0, 2.0, 4.0):
        for length in (16, 64, 256):
            for side in ("upper", "lower"):
                args = sg.Args()
                args.tilt_order, args.segment_len, args.tilt_side = order, length, side
                assert sg.validate_args(args).segment_len == length


def test_only_the_tilt_arguments_differ_from_the_baseline():
    """Nothing else moves: no hyperparameter drifted while forking the file."""
    added = {"tilt_order", "segment_len", "tilt_side"}
    mine, theirs = vars(sg.Args()), vars(base.Args())
    assert set(mine) - set(theirs) == added
    for name, value in theirs.items():
        if name == "exp_name":      # derived from the filename, necessarily different
            continue
        assert mine[name] == value, name

"""Contracts for the structured-generalization diagnostic v2: teacher, held-out sets, grid, path energy, fit-matched rule, gamma."""

import math

import pytest
import torch

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import structured_generalization_diag_v2 as diag

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def args(**overrides):
    return diag.Args(**overrides)


def test_structured_teacher_ignores_distractors_and_keeps_preactivation_scale():
    a = args()
    gen = torch.Generator(device="cuda").manual_seed(2)
    teacher = diag.structured_teacher(a, gen)
    assert torch.count_nonzero(teacher[0][:, a.relevant:]) == 0
    x = torch.randn(4096, a.input_dim, generator=gen, device="cuda")
    scaled = x.clone()
    scaled[:, a.relevant:] *= 7.0
    torch.testing.assert_close(reference.teach(teacher, scaled), reference.teach(teacher, x), rtol=0, atol=0)
    # A dense teacher from the same generator state has unit pre-activation variance; structure keeps it.
    preactivation = (x @ teacher[0].T).var().item()
    assert 0.8 < preactivation < 1.25


def test_held_out_sets_share_draw_and_only_relevant_changes_target():
    a = args(held_out=2048)
    teacher = diag.structured_teacher(a, torch.Generator(device="cuda").manual_seed(2))
    held = diag.held_out_sets(a, teacher, torch.Generator(device="cuda").manual_seed(7))
    x, y = held["in_dist"]
    dx, dy = held["distractor"]
    rx, ry = held["relevant"]
    torch.testing.assert_close(dx[:, :a.relevant], x[:, :a.relevant], rtol=0, atol=0)
    torch.testing.assert_close(dx[:, a.relevant:], x[:, a.relevant:] * a.distractor_scale, rtol=0, atol=0)
    torch.testing.assert_close(dy, y, rtol=0, atol=0)
    torch.testing.assert_close(rx[:, a.relevant:], x[:, a.relevant:], rtol=0, atol=0)
    assert not torch.allclose(ry, y)


def test_candidate_grid_reproduces_locks_at_unit_multiplier_without_decay():
    locks = diag.locked_configs(args().plan)
    grid = diag.candidate_grid(locks)
    assert set(grid) == set(diag.METHODS)
    for method, rows in grid.items():
        assert len(rows) == len(diag.LR_MULTIPLIERS) * len(diag.WEIGHT_DECAYS)
        names = [name for name, _ in rows]
        assert len(set(names)) == len(names)
        lock = dict(next(config for name, config in rows if name == f"{method}_lr1_wd0"))
        assert lock == {key: locks[method][key] for key in diag.KEYS}
        for _, config in rows:
            assert {key: config[key] for key in ("beta1", "beta2", "head_lr_scale")} == \
                {key: locks[method][key] for key in ("beta1", "beta2", "head_lr_scale")}


def test_accumulate_path_is_the_l1_step_length_per_candidate():
    gen = torch.Generator(device="cuda").manual_seed(5)
    weights = [torch.randn(3, o, i + 1, generator=gen, device="cuda") for o, i in [(5, 3), (5, 5), (1, 5)]]
    previous = [w + torch.randn_like(w) * 0.1 for w in weights]
    path = [torch.full((3,), 2.0, device="cuda") for _ in weights]
    diag.accumulate_path(path, weights, previous)
    for total, w, p in zip(path, weights, previous):
        torch.testing.assert_close(total, 2.0 + (w - p).abs().sum(dim=(-1, -2)))


def test_decision_rule_fires_kills_and_reports_ambiguity():
    base = {"relevant": 1.0}
    fire = {"a": {"in_dist": 0.20, "distractor": 0.40, **base}, "b": {"in_dist": 0.19, "distractor": 0.60, **base}}
    assert diag.decide(fire)["verdict"] == "fire"
    # Fit-matched pair with equal distractor risk kills; an unmatched third candidate cannot rescue it.
    kill = {"a": {"in_dist": 0.20, "distractor": 0.40, **base}, "b": {"in_dist": 0.204, "distractor": 0.41, **base},
            "c": {"in_dist": 0.10, "distractor": 0.30, **base}}
    out = diag.decide(kill)
    assert out["verdict"] == "kill" and len(out["fit_matched_pairs"]) == 1
    # No fit-matched pair at all: ambiguous, never kill, whatever the raw ratios do.
    unmatched = {"a": {"in_dist": 0.20, "distractor": 0.40, **base}, "b": {"in_dist": 0.10, "distractor": 0.21, **base}}
    assert diag.decide(unmatched)["verdict"] == "ambiguous"
    # A single finite candidate is ambiguous, not kill; NaN and zero risks are excluded.
    single = {"a": {"in_dist": 0.20, "distractor": 0.40, **base}, "c": {"in_dist": math.nan, "distractor": 0.0, **base},
              "d": {"in_dist": 0.0, "distractor": 0.3, **base}}
    out = diag.decide(single)
    assert out["verdict"] == "ambiguous" and out["finite_candidates"] == 1


def test_svag_gamma_is_one_without_variance_and_shrinks_with_relative_variance():
    gen = torch.Generator(device="cuda").manual_seed(9)
    m = torch.randn(2, 4, 5, generator=gen, device="cuda")
    beta1, beta2 = torch.full((2, 1, 1), 0.9, device="cuda"), torch.full((2, 1, 1), 0.99, device="cuda")
    count = 50
    exact = m.square() * (1 - beta2 ** count) / (1 - beta1 ** count) ** 2  # vhat == mhat^2: zero variance
    torch.testing.assert_close(diag.svag_gamma(m, exact, beta1, beta2, count), torch.ones_like(m), rtol=1e-5, atol=1e-6)
    noisy = exact + 10.0
    gamma = diag.svag_gamma(m, noisy, beta1, beta2, count)
    assert bool((gamma < 1).all()) and bool((gamma > 0).all())
    # Balles-Hennig closed form with rho(b, t) and s = (vhat - mhat^2) / (1 - rho).
    mhat = m / (1 - beta1 ** count)
    rho = (1 - beta1) * (1 + beta1 ** (count + 1)) / ((1 + beta1) * (1 - beta1 ** (count + 1)))
    s = 10.0 / (1 - beta2 ** count) / (1 - rho)
    torch.testing.assert_close(gamma, mhat.square() / (mhat.square() + rho * s), rtol=1e-5, atol=1e-7)

"""Contracts for the structured-generalization diagnostic: teacher structure, held-out construction, grid, path energy, rule."""

import math

import pytest
import torch

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import structured_generalization_diag_v1 as diag

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
    kill = {"a": {"in_dist": 0.20, "distractor": 0.40, **base}, "b": {"in_dist": 0.10, "distractor": 0.21, **base}}
    assert diag.decide(kill)["verdict"] == "kill"
    ambiguous = {"a": {"in_dist": 0.20, "distractor": 0.40, **base}, "b": {"in_dist": 0.10, "distractor": 0.30, **base}}
    assert diag.decide(ambiguous)["verdict"] == "ambiguous"
    with_nan = {**fire, "c": {"in_dist": math.nan, "distractor": 0.0, **base}}
    assert diag.decide(with_nan)["finite_candidates"] == 2

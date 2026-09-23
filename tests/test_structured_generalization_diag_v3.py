"""Contracts for the structured-generalization diagnostic v3: teacher, held-out sets, grid, oracle decay, gate agreement, filters, rules."""

import math

import pytest
import torch

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import structured_generalization_diag_v3 as diag

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


def test_candidate_grid_has_uniform_and_oracle_rows_on_the_locks():
    locks = diag.locked_configs(args().plan)
    grid = diag.candidate_grid(locks)
    assert set(grid) == set(diag.METHODS)
    for method, rows in grid.items():
        assert len(rows) == len(diag.LR_MULTIPLIERS) * (len(diag.UNIFORM_DECAYS) + len(diag.ORACLE_DECAYS))
        names = [name for name, _, _ in rows]
        assert len(set(names)) == len(names)
        lock = next(config for name, config, _ in rows if name == f"{method}_lr1_wd0")
        assert lock == {key: locks[method][key] for key in diag.KEYS}
        for name, config, oracle in rows:
            assert {key: config[key] for key in ("beta1", "beta2", "head_lr_scale")} == \
                {key: locks[method][key] for key in ("beta1", "beta2", "head_lr_scale")}
            if "oracle" in name:
                assert config["weight_decay"] == 0 and oracle in diag.ORACLE_DECAYS
            else:
                assert oracle == 0 and config["weight_decay"] in diag.UNIFORM_DECAYS
        assert sum(oracle > 0 for _, _, oracle in rows) == len(diag.LR_MULTIPLIERS) * len(diag.ORACLE_DECAYS)


def test_oracle_decay_shrinks_only_distractor_columns_by_the_decoupled_factor():
    gen = torch.Generator(device="cuda").manual_seed(5)
    relevant, input_dim = 4, 17
    weights = torch.randn(3, 8, input_dim + 1, generator=gen, device="cuda")
    before = weights.clone()
    lr = torch.tensor([1e-3, 2e-3, 5e-4], device="cuda").view(-1, 1, 1)
    factor = diag.oracle_factor(lr, [0.0, 0.3, 1.0])
    torch.testing.assert_close(factor.flatten(), torch.tensor([1.0, 1 - 2e-3 * 0.3, 1 - 5e-4], device="cuda"))
    diag.apply_oracle_decay(weights, factor, relevant, input_dim)
    torch.testing.assert_close(weights[..., :relevant], before[..., :relevant], rtol=0, atol=0)
    torch.testing.assert_close(weights[..., input_dim:], before[..., input_dim:], rtol=0, atol=0)
    torch.testing.assert_close(weights[..., relevant:input_dim], before[..., relevant:input_dim] * factor)
    torch.testing.assert_close(weights[0], before[0], rtol=0, atol=0)


def test_gate_agreement_reads_shrink_direction_by_signal_type():
    relevant, input_dim = 4, 17
    weights = torch.rand(2, 8, input_dim + 1, device="cuda") + 0.1
    signal = torch.ones_like(weights)
    signal[..., relevant:input_dim] = -1.0
    rel, dis = diag.gate_agreement(signal, weights, relevant, input_dim, 1.0)  # gradient-type: shrink where s * w > 0
    torch.testing.assert_close(rel, torch.ones(2, device="cuda"))
    torch.testing.assert_close(dis, torch.zeros(2, device="cuda"))
    rel, dis = diag.gate_agreement(signal, weights, relevant, input_dim, -1.0)  # displacement-type: shrink where s * w < 0
    torch.testing.assert_close(rel, torch.zeros(2, device="cuda"))
    torch.testing.assert_close(dis, torch.ones(2, device="cuda"))
    # The bias column never counts.
    signal[..., input_dim] = -1.0
    rel_again, _ = diag.gate_agreement(signal, weights, relevant, input_dim, 1.0)
    torch.testing.assert_close(rel_again, torch.ones(2, device="cuda"))


def test_update_filter_matches_the_closed_form_response_to_a_constant():
    state = torch.zeros(3, 4, device="cuda")
    value = torch.full((3, 4), 2.0, device="cuda")
    for _ in range(25):
        diag.update_filter(state, value, 0.9)
    torch.testing.assert_close(state, value * (1 - 0.9 ** 25))


def test_ceiling_rule_fires_kills_and_reports_ambiguity():
    def row(method, in_dist, distractor, oracle=0.0, valid=True):
        return {"method": method, "valid": valid, "oracle_decay": oracle, "in_dist": in_dist,
                "distractor": distractor, "relevant": 1.0}
    fire_fit = {"u0": row("polar", 0.10, 0.50), "u1": row("polar", 0.11, 0.40), "o": row("polar", 0.07, 0.30, 0.3)}
    assert diag.decide_ceiling(fire_fit, "polar")["verdict"] == "fire"
    fire_ood = {"u0": row("polar", 0.10, 0.50), "o": row("polar", 0.103, 0.35, 0.3)}
    assert diag.decide_ceiling(fire_ood, "polar")["verdict"] == "fire"
    kill = {"u0": row("polar", 0.10, 0.50), "o1": row("polar", 0.101, 0.48, 0.1), "o2": row("polar", 0.13, 0.20, 1.0)}
    out = diag.decide_ceiling(kill, "polar")
    assert out["verdict"] == "kill" and out["best_uniform"] == "u0"
    ambiguous = {"u0": row("polar", 0.10, 0.50), "o": row("polar", 0.108, 0.40, 0.3)}  # unmatched, sub-bar fit gain
    assert diag.decide_ceiling(ambiguous, "polar")["verdict"] == "ambiguous"
    other_method = {"u0": row("adamw", 0.10, 0.50), "o": row("adamw", 0.05, 0.10, 0.3), "p": row("polar", 0.1, 0.5)}
    assert diag.decide_ceiling(other_method, "polar")["verdict"] == "ambiguous"
    invalid = {"u0": row("polar", 0.10, 0.50), "o": row("polar", math.nan, 0.0, 0.3, valid=False)}
    assert diag.decide_ceiling(invalid, "polar")["verdict"] == "ambiguous"


def test_detect_rule_thresholds_the_best_margin():
    assert diag.decide_detect({"a": 0.05, "b": 0.30})["verdict"] == "fire"
    assert diag.decide_detect({"a": 0.05, "b": 0.08})["verdict"] == "kill"
    out = diag.decide_detect({"a": 0.05, "b": 0.15, "c": math.nan})
    assert out["verdict"] == "ambiguous" and out["best_signal"] == "b" and "c" not in out["margins"]
    assert diag.decide_detect({})["verdict"] == "ambiguous"

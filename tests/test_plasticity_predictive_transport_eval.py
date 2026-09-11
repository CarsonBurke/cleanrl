"""Joint selection and candidate isolation contracts; no learner/CUDA execution."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cleanrl.plasticity import predictive_transport_eval_v1 as experiment


def test_joint_selection_preserves_tuple_identity_after_failed_candidate_removal():
    grid = experiment.arm_configs(replace(experiment.Args(), adam_lrs=(0.001, 0.01), betas=(0.0, 0.9, 0.99)))[0]["grid"]
    curves = [
        {"step": 2, "validation": [0.0, 9.0, 8.0, 0.0, 1.0, 10.0],
         "candidate_valid": [False, True, True, False, True, True]},
        {"step": 8, "validation": [0.0, 0.0, 8.0, 0.0, 1.0, 10.0],
         "candidate_valid": [False, True, True, False, True, True]},
    ]
    selection = experiment.joint_selection(curves, grid, 8)
    # Candidate 1 wins at the endpoint but loses duration-weighted validation.
    assert selection["chosen_index"] == 4
    assert selection["chosen"] == {"learning_rate": 0.01, "beta1": 0.9}
    assert selection["validation_sustained_grid"] == [None, 2.25, 8.0, None, 1.0, 10.0]
    assert selection["edge_flags"] == {"learning_rate": {"lower": False, "upper": True},
                                       "beta1": {"lower": False, "upper": False}}
    # Lock serialization cannot turn a pair into a numeric surrogate index.
    assert json.loads(json.dumps(selection))["chosen"] == grid[4]


def test_historical_failures_remain_disqualified_even_after_finite_recovery():
    grid = [{"learning_rate": 0.01, "beta1": beta} for beta in (0.0, 0.9, 0.99)]
    failure = {"step": 2, "reason": "nonfinite moment"}
    curves = [
        {"step": 2, "validation": [0.0, float("nan"), 1e100],
         "candidate_valid": [False, True, True], "candidate_failures": [failure, None, None]},
        {"step": 4, "validation": [0.0, 0.0, 1e100], "candidate_valid": [True, True, True]},
    ]
    selection = experiment.joint_selection(curves, grid, 4)
    assert selection["candidate_valid"] == [False, False, True]
    assert selection["candidate_failures"][0] == failure
    assert selection["candidate_failures"][1] == {"step": 2, "reason": "nonfinite clean validation"}
    # Very bad but finite validation is still eligible; there is no performance cull.
    assert selection["chosen_index"] == 2
    assert selection["validation_sustained_grid"] == [None, None, 1e100]
    assert selection["edge_flags"]["learning_rate"] == {"lower": True, "upper": True}
    assert selection["edge_flags"]["beta1"] == {"lower": False, "upper": True}


def test_all_invalid_is_explicit_failed_decision_with_null_scores():
    grid = [{"learning_rate": 0.01, "beta1": 0.9}, {"learning_rate": 0.1, "beta1": 0.99}]
    curves = [{"step": 4, "validation": [None, float("inf")], "candidate_valid": [True, True]}]
    decision = experiment.joint_selection(curves, grid, 4)
    assert decision["status"] == "failed_all_candidates"
    assert decision["chosen"] is None and decision["chosen_index"] is None
    assert decision["edge_flags"] is None
    assert decision["validation_sustained_grid"] == [None, None]
    assert all(reason["reason"] == "nonfinite clean validation" for reason in decision["candidate_failures"])
    json.dumps(decision, allow_nan=False)


@pytest.mark.parametrize("curves", [
    [{"step": 3, "validation": [1.0], "candidate_valid": [True]}],
    [{"step": 4, "validation": [1.0], "candidate_valid": []}],
    [{"step": 4, "validation": [1.0], "candidate_valid": [True], "candidate_failures": []}],
])
def test_joint_selection_rejects_incomplete_stream_and_mismatched_failure_axes(curves):
    with pytest.raises(ValueError):
        experiment.joint_selection(curves, [{"learning_rate": 0.01, "beta1": 0.9}], 4)


def test_candidate_state_isolation_and_transient_failure_history():
    learner = SimpleNamespace(
        mutable=[torch.ones(3, 2, 2), torch.zeros(3), torch.tensor(16), torch.tensor(16.0)],
        finite_candidates=torch.tensor([True, True, False]),
    )
    learner.mutable[0][1, 0, 0] = float("nan")
    assert experiment.candidate_state(learner, 3) == [True, False, False]
    # Shared state corruption is not attributable to one grid member.
    learner.mutable[-1].fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="shared"):
        experiment.candidate_state(learner, 3)


@pytest.mark.parametrize("updates", [
    {"methods": ("adam", "predictive")},
    {"methods": ("uncertainty", "full")},
    {"methods": ("adam", "uncertainty", "network")},
    {"methods": ("adam", "uncertainty", "adam")},
    {"betas": (0.9, 1.0)},
    {"betas": (-0.1, 0.9)},
    {"betas": (0.9, float("nan"))},
    {"betas": (0.99, 0.9)},
    {"betas": (0.9, 0.9)},
    {"betas": ()},
    {"adam_lrs": (0.01, 0.001)},
    {"adam_lrs": (0.0, 0.01)},
    {"adam_lrs": (0.01, float("inf"))},
    {"graph_steps": 8},
    {"samples": 65535},
    {"log_every": 4097},
    {"switch_at": 0.000001},
    {"switch_at": 17 / 65536},
    {"noise_rate": 0.0},
    {"hetero": float("inf")},
    {"prior_scales": (0.1,)},
    {"diffusion": 0.001},
    {"known_noise": True},
])
def test_invalid_optimizer_protocol_rejected_without_cuda(updates):
    with pytest.raises(ValueError):
        experiment.validate_args(replace(experiment.Args(), **updates))


def test_beta_zero_and_stationary_protocol_remain_available_for_transport_limits():
    args = replace(experiment.Args(), betas=(0.0, 0.9), switch_at=0.0, hetero=0.0)
    experiment.validate_args(args)
    grid = experiment.arm_configs(args)[0]["grid"]
    assert {candidate["beta1"] for candidate in grid} == {0.0, 0.9}


def test_phase_reporting_uses_interval_duration_not_checkpoint_count():
    curves = [
        {"step": 2, "teacher_phase": 0, "risk": 1.0},
        {"step": 8, "teacher_phase": 0, "risk": 5.0},
        {"step": 10, "teacher_phase": 1, "risk": 9.0},
    ]
    assert experiment.phase_means(curves, lambda c: c["risk"]) == {
        "0": {"samples": 8, "mse": 4.0}, "1": {"samples": 2, "mse": 9.0}}

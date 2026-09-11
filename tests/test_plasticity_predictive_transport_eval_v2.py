"""Validation-only selection, invalid-state history and immutable source contracts."""

import hashlib
import json
from dataclasses import replace

import pytest

from cleanrl.plasticity import predictive_transport_eval_v2 as experiment


def test_selection_preserves_joint_identity_and_disqualifies_historical_failure():
    grid = [{"learning_rate": lr, "beta1": beta} for lr in (0.001, 0.01) for beta in (0.0, 0.9)]
    failure = {"step": 2, "reason": "nonfinite state"}
    curves = [
        {"step": 2, "validation": [4.0, 2.0, 0.1, 3.0], "candidate_valid": [True, True, False, True],
         "candidate_failures": [None, None, failure, None]},
        {"step": 8, "validation": [3.0, 1.0, 0.1, 2.0], "candidate_valid": [True, True, True, True],
         "candidate_failures": [None, None, None, None]},
    ]
    decision = experiment.joint_selection(curves, grid, 8)
    assert decision["chosen"] == grid[1]
    assert decision["chosen_index"] == 1
    assert decision["candidate_failures"][2] == failure
    assert decision["candidate_valid"] == [True, True, False, True]
    json.dumps(decision, allow_nan=False)


def test_selection_weights_durations_instead_of_checkpoint_count():
    grid = [{"learning_rate": lr, "beta1": 0.9} for lr in (0.001, 0.01)]
    curves = [
        {"step": 2, "validation": [0.0, 4.0], "candidate_valid": [True, True], "candidate_failures": [None, None]},
        {"step": 10, "validation": [4.0, 2.0], "candidate_valid": [True, True], "candidate_failures": [None, None]},
    ]
    assert experiment.joint_selection(curves, grid, 10)["chosen"] == grid[1]
    with pytest.raises(ValueError):
        experiment.joint_selection(curves, grid, 12)


@pytest.mark.parametrize("updates", [
    {"methods": ("implicit",)}, {"methods": ("adam", "robust")},
    {"methods": ("adam", "implicit", "uncertainty")},
    {"betas": (0.9, 1.0)}, {"adam_lrs": (0.01, 0.001)},
    {"hetero": float("nan")}, {"samples": 17}, {"switch_at": 0.5003},
])
def test_invalid_experiment_protocol_rejected_before_cuda(updates):
    with pytest.raises(ValueError):
        experiment.validate_args(replace(experiment.Args(), **updates))


def test_source_verification_detects_changed_implementation(tmp_path):
    source = tmp_path / "optimizer.py"
    source.write_text("a = 1\n")
    hashes = {"optimizer.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    experiment.verify_sources(tmp_path, hashes)
    source.write_text("a = 2\n")
    with pytest.raises(RuntimeError, match="source changed"):
        experiment.verify_sources(tmp_path, hashes)

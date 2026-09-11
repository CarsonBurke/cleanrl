"""CPU-only configuration boundaries; inherited selection helpers are tested in v3."""

from dataclasses import replace

import pytest

from cleanrl.plasticity import iterated_bayes_eval_v4 as experiment


@pytest.mark.parametrize("iterations", [(), (0, 4), (-1, 4), (2, 4, 4), (2, 3), (1.5, 4), (True, 4)])
def test_invalid_iterations_cannot_drop_primary_or_alias_arm_names(iterations):
    with pytest.raises(ValueError, match="iterations"):
        experiment.validate_args(replace(experiment.Args(), iterations=iterations))


@pytest.mark.parametrize("methods", [
    ("iterated",),
    ("adam", "iterated"),
    ("network", "iterated"),
    ("adam", "network", "network"),
    ("adam", "network", "unit", "iterated"),
])
def test_comparison_cannot_omit_a_control_or_add_an_unimplemented_family(methods):
    with pytest.raises(ValueError, match="methods"):
        experiment.validate_args(replace(experiment.Args(), methods=methods))


@pytest.mark.parametrize("updates", [
    {"samples": 65},
    {"log_every": 15},
    {"samples": 64, "switch_at": 0.3},
    {"samples": 64, "switch_at": 0.001},
    {"graph_steps": 0},
])
def test_partial_replay_or_truncated_teacher_switch_is_rejected(updates):
    with pytest.raises(ValueError):
        experiment.validate_args(replace(experiment.Args(), **updates))


@pytest.mark.parametrize("switch_at", [0.0, 0.5])
def test_full_stream_can_end_between_logs_and_switch_between_logs(switch_at):
    # Samples 48 (teacher boundary), 64 (log) and 96 (final) are legal
    # checkpoints. Requiring log divisibility would reject this complete stream.
    experiment.validate_args(replace(experiment.Args(), samples=96, log_every=64,
                                     switch_at=switch_at, iterations=(1, 2, 4)))


@pytest.mark.parametrize("updates", [
    {"noise": float("inf")},
    {"hetero": float("nan")},
    {"diffusion": float("inf")},
])
def test_nonfinite_stream_or_process_configuration_fails_before_cuda(updates):
    with pytest.raises(ValueError, match="finite"):
        experiment.validate_args(replace(experiment.Args(), **updates))

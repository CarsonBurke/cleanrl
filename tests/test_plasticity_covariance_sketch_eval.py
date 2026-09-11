"""CPU-only reporting contracts; no learner execution or CUDA allocation."""

import json
from dataclasses import replace

import pytest
import torch

from cleanrl.plasticity import covariance_sketch_eval_v3 as experiment


def test_sustained_selection_resists_endpoint_trap_and_weights_duration():
    curves = [
        {"step": 2, "validation": [9.0, 1.0]},
        {"step": 9, "validation": [0.0, 4.0]},
        {"step": 10, "validation": [5.0, 0.0]},
    ]
    selection = experiment.validation_selection(curves, [0.01, 0.1], 10)
    assert selection["validation_sustained_grid"] == pytest.approx([2.3, 3.0])
    assert selection["chosen_index"] == 0
    assert min(range(2), key=curves[-1]["validation"].__getitem__) == 1
    # Equal checkpoint weighting would also choose the wrong candidate.
    assert sum(c["validation"][0] for c in curves) > sum(c["validation"][1] for c in curves)


@pytest.mark.parametrize("curves", [
    [{"step": 4, "validation": [1.0, float("nan")]}],
    [{"step": 4, "validation": [float("inf"), 1.0]}],
])
def test_nonfinite_candidate_is_failure_not_silent_grid_culling(curves):
    with pytest.raises(FloatingPointError, match="nonfinite"):
        experiment.validation_selection(curves, [0.01, 0.1], 4)


@pytest.mark.parametrize("curves", [
    [{"step": 3, "validation": [1.0, 2.0]}],
    [{"step": 4, "validation": [1.0, 2.0]}, {"step": 4, "validation": [1.0, 2.0]}],
    [{"step": 4, "validation": [1.0]}],
])
def test_selection_rejects_partial_stream_duplicate_steps_and_wrong_grid(curves):
    with pytest.raises(ValueError):
        experiment.validation_selection(curves, [0.01, 0.1], 4)


def test_held_out_curves_route_teachers_and_cannot_change_locked_selection(tmp_path):
    curves = [{"step": 4, "validation": [0.0, 10.0], "checkpoint": "4.pt"},
              {"step": 8, "validation": [0.0, 10.0], "checkpoint": "8.pt"}]
    selection = experiment.validation_selection(curves, [0.01, 0.1], 8)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    locked = path.read_bytes()
    # Two actual constant networks predict 1 and 9. Candidate 1 would win the
    # second teacher's test but cannot replace validation's locked candidate 0.
    weights = [torch.zeros(2, 1, 2), torch.zeros(2, 1, 2),
               torch.tensor([[[0.0, 1.0]], [[0.0, 9.0]]])]
    for curve in curves:
        torch.save({"step": curve["step"], "weights": weights}, tmp_path / curve["checkpoint"])
    x = torch.zeros(3, 1)
    targets = [torch.ones(3), torch.full((3,), 9.0)]
    result = experiment.score_selected(path, curves, tmp_path, x, targets, switch=4)
    assert [point["teacher_phase"] for point in result["test_curve"]] == [0, 1]
    assert [point["mse"] for point in result["test_curve"]] == pytest.approx([0.0, 64.0])
    assert result["test_sustained_mse"] == pytest.approx(32.0)
    assert path.read_bytes() == locked
    # A different untouched test reverses its preferred hyperparameter without
    # modifying the persisted decision or the validation source curves.
    changed = experiment.score_selected(path, curves, tmp_path, x, targets[::-1], switch=4)
    assert [point["mse"] for point in changed["test_curve"]] == pytest.approx([64.0, 0.0])
    assert path.read_bytes() == locked
    assert experiment.validation_selection(curves, [0.01, 0.1], 8) == selection
    stationary = experiment.score_selected(path, curves, tmp_path, x, targets, switch=8)
    assert [point["teacher_phase"] for point in stationary["test_curve"]] == [0, 0]
    assert stationary["test_sustained_mse"] == 0.0


def test_checkpoint_sample_mismatch_is_rejected(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"chosen_index": 0, "samples": 4}))
    torch.save({"step": 3, "weights": []}, tmp_path / "4.pt")
    with pytest.raises(RuntimeError, match="sample count"):
        experiment.score_selected(path, [{"step": 4, "checkpoint": "4.pt"}], tmp_path,
                                  torch.zeros(1, 1), [torch.ones(1), torch.ones(1)], switch=4)


@pytest.mark.parametrize("updates", [
    {"samples": 65520},
    {"log_every": 4112},
    {"switch_at": 16 / 65536},
    {"switch_at": 1 / 65536},
    {"switch_at": 0.000001},
    {"buffer": 0},
    {"graph_steps": 0},
])
def test_invalid_capture_or_switch_cadence_rejected_without_cuda(updates):
    with pytest.raises(ValueError):
        experiment.validate_args(replace(experiment.Args(), **updates))


@pytest.mark.parametrize("updates", [
    {"methods": ("adam", "sketch")},
    {"methods": ("network", "unsupported")},
    {"methods": ("network", "network")},
    {"ranks": (16, 16)},
    {"ranks": (0,)},
    {"ranks": ()},
    {"scalar_rank": -1},
    {"prior_scales": (0.1, 0.01, 1.0)},
    {"adam_lrs": (0.001, 0.01, float("inf"))},
    {"noise": float("nan")},
    {"hetero": float("inf")},
    {"diffusion": float("nan")},
    {"noise_rate": 0.0},
    {"known_noise": True, "noise": 0.0},
])
def test_genuine_configuration_errors_rejected_without_cuda(updates):
    with pytest.raises(ValueError):
        experiment.validate_args(replace(experiment.Args(), **updates))


def test_actual_capture_cadence_is_checked_even_if_args_cadences_fit():
    with pytest.raises(ValueError, match="capture_steps"):
        experiment.validate_cadence(experiment.Args(), 3)


def test_nonfinite_diagnostic_is_persistable_but_never_success(tmp_path):
    result = {"status": "training", "curves": [{"validation": [1.0, float("nan")]}]}
    with pytest.raises(FloatingPointError) as error:
        experiment.require_finite(result["curves"], "curves")
    result.update(status="failed_nonfinite", failure=str(error.value))
    experiment.save_json(tmp_path / "results.json", result)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["status"] == "failed_nonfinite"
    assert saved["curves"][0]["validation"] == [1.0, None]
    assert "nonfinite" in saved["failure"]

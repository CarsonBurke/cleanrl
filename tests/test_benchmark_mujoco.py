"""Small CPU tests of reporting/gates; no models, physics, or CUDA execution."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "benchmark_mujoco_throughput", Path(__file__).parents[1] / "scripts/benchmark_mujoco_throughput.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_recursive_gate_checks_lists_tuples_and_rejects_nonboolean_truthy_values():
    measurements = {
        "projection": {"checks": {"labels": True, "means": False}},
        "updates": [{"checks": {"finite": True}}, {"checks": {"metrics": float("nan")}}],
        "tuple": ({"checks": {"not_a_boolean": 1}},),
    }
    assert benchmark.numerical_failures(measurements) == [
        "measurements/projection/means", "measurements/updates/1/metrics",
        "measurements/tuple/0/not_a_boolean",
    ]


@pytest.mark.parametrize("reference,candidate", [
    ([0.0], [float("nan")]), ([float("nan")], [0.0]),
    ([float("inf")], [float("inf")]), ([-float("inf")], [-float("inf")]),
])
def test_nonfinite_tensors_never_pass_even_when_allclose_accepts_equal_infinities(reference, candidate):
    comparison = benchmark.tensor_difference(torch.tensor(reference), torch.tensor(candidate), atol=1e-6, rtol=1e-6)
    assert comparison["checks"]["within_tolerance"] is False
    assert benchmark.numerical_failures(comparison)


def test_error_only_diagnostics_do_not_override_adoption_tolerances():
    expected = torch.ones(len(benchmark.UPDATE_METRIC_NAMES))
    actual = expected.clone()
    actual[-1] += 3e-4  # Permitted exp(log_eta) error for the fused solver only.
    aggregate = benchmark.tensor_difference(expected, actual)
    assert "checks" not in aggregate
    assert benchmark.numerical_failures(aggregate) == []
    fused = benchmark.update_metric_comparisons(expected, actual, fused=True)
    assert benchmark.numerical_failures(fused) == []
    strict = benchmark.update_metric_comparisons(expected, actual, fused=False)
    assert "measurements/eta/within_tolerance" in benchmark.numerical_failures(strict)
    actual[3] += 1e-5  # Unaffected mean-KL metric must stay stringent when fused.
    fused = benchmark.update_metric_comparisons(expected, actual, fused=True)
    assert "measurements/mean_kl/within_tolerance" in benchmark.numerical_failures(fused)


def test_critic_batch_diagnostics_are_not_an_adoption_gate():
    error = benchmark.tensor_difference(torch.tensor([0.0]), torch.tensor([100.0]))
    assert not error["bitwise_equal"]
    assert benchmark.numerical_failures({"critic_batch_parity": error}) == []


def test_gate_persists_named_failures_before_raising_and_serializes_nonfinite_evidence():
    report = {"measurements": {"projection": {"max_error": float("nan"), "checks": {"labels": False}}}}
    saved = []

    def save():
        saved.append(json.loads(json.dumps(benchmark.json_compatible(report), allow_nan=False)))

    with pytest.raises(AssertionError, match="projection/labels"):
        benchmark.enforce_numerical_parity(SimpleNamespace(require_numerical_parity=True), report, save)
    assert len(saved) == 1
    assert saved[0]["measurements"]["projection"]["max_error"] == "nan"
    assert saved[0]["numerical_parity"] == {
        "required": True, "check_count": 1, "passed": False,
        "failures": ["measurements/projection/labels"],
    }


def test_optional_gate_reports_failures_without_raising():
    report = {"measurements": {"checks": {"labels": False}}}
    saved = []
    benchmark.enforce_numerical_parity(SimpleNamespace(require_numerical_parity=False), report, lambda: saved.append(True))
    assert saved == [True]
    assert report["numerical_parity"]["passed"] is False


def test_required_gate_rejects_absent_candidate_checks():
    report = {"measurements": {}}
    with pytest.raises(AssertionError, match="no_candidate_checks"):
        benchmark.enforce_numerical_parity(SimpleNamespace(require_numerical_parity=True), report, lambda: None)


def test_json_serialization_handles_nested_nonfinite_tuples():
    raw = {"values": (float("inf"), [float("nan"), -float("inf"), 2.0])}
    assert json.loads(json.dumps(benchmark.json_compatible(raw), allow_nan=False)) == {
        "values": ["inf", ["nan", "-inf", 2.0]],
    }


def test_required_gate_cli_needs_a_candidate_benchmark(monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark", "--env-only", "--require-numerical-parity"])
    with pytest.raises(SystemExit) as error:
        benchmark.parse_args()
    assert error.value.code == 2
    monkeypatch.setattr("sys.argv", ["benchmark", "--profile-update", "--require-numerical-parity"])
    assert benchmark.parse_args().require_numerical_parity


def test_experimental_fused_updates_require_explicit_opt_in(monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark", "--profile-update"])
    assert not benchmark.parse_args().fused_updates
    monkeypatch.setattr("sys.argv", ["benchmark", "--profile-update", "--fused-updates"])
    assert benchmark.parse_args().fused_updates


@pytest.mark.parametrize("extra", [[], ["--env-only"], ["--projection-only"]])
def test_fused_update_cli_rejects_ignored_combinations(monkeypatch, extra):
    monkeypatch.setattr("sys.argv", ["benchmark", "--fused-updates", *extra])
    with pytest.raises(SystemExit) as error:
        benchmark.parse_args()
    assert error.value.code == 2


def test_thread_sweep_keeps_order_and_deduplicates_counts(monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark", "--env-only", "--thread-counts", "1", "2", "2", "4"])
    assert benchmark.parse_args().thread_counts == [1, 2, 4]

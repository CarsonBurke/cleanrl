"""Score-hook reporting tests use fake logs and mocked read-only queue queries."""

import importlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("score_runs")


class FakeScalars:
    def window_mean(self, tag, step, window):
        return 15.0 if tag == "charts/episodic_return" else 1000.0


def existing_run(scorer, monkeypatch):
    directory = Path("runs/HalfCheetah-v4__available__1__100")
    result = scorer.RunResult(np.array([10.0, 20.0]), 15.0, 15.0, 9.8, 1_000_000, FakeScalars())
    monkeypatch.setattr(scorer, "find_runs", lambda *args: [directory])
    monkeypatch.setattr(scorer, "load_returns", lambda *args: result)
    return directory, result


def test_partial_comparison_names_missing_patterns_and_empty_logs(scorer, monkeypatch, capsys):
    available, result = existing_run(scorer, monkeypatch)
    empty = Path("runs/HalfCheetah-v4__empty__1__100")
    monkeypatch.setattr(scorer, "find_runs", lambda *args: [available, empty])
    monkeypatch.setattr(scorer, "load_returns", lambda path, *args: result if path == available else None)
    scorer.main(["available", "missing", "empty", "--env", "HalfCheetah-v4", "--at", "1M,2M",
                 "--metrics", "charts/SPS"])
    output = capsys.readouterr().out
    assert "No run directories matched requested pattern 'missing'" in output
    assert f"Run '{empty}' has no charts/episodic_return samples" in output
    assert "1 runs across 1 env(s)" in output
    assert "available" in output and "15.0" in output
    assert "@1M" in output and "@2M" in output and "SPS@2M" in output
    assert "-- = run never reached this step" in output


def test_all_missing_patterns_are_each_reported(scorer, monkeypatch, capsys):
    monkeypatch.setattr(scorer, "find_runs", lambda *args: [])
    with pytest.raises(SystemExit) as failure:
        scorer.main(["first", "second"])
    assert failure.value.code == 1
    output = capsys.readouterr().out
    assert "requested pattern 'first'" in output
    assert "requested pattern 'second'" in output
    assert "No runs found matching" in output


def test_jobs_are_queried_once_and_scores_do_not_override_failed_state(scorer, monkeypatch, capsys):
    existing_run(scorer, monkeypatch)
    calls = []
    jobs = {
        7: {"id": 7, "name": "baseline", "state": "succeeded"},
        8: {"id": 8, "name": "candidate", "state": "skipped",
            "stateReason": "prerequisite job 6 failed"},
    }
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(jobs[int(command[2])]), "")
    monkeypatch.setattr(scorer.subprocess, "run", run)
    scorer.main(["available", "--jobs", "7", "8", "7"])
    output = capsys.readouterr().out
    assert [command for command, _ in calls] == [["mlq", "show", "7", "--json"], ["mlq", "show", "8", "--json"]]
    assert all(options == dict(capture_output=True, text=True, check=True, timeout=30) for _, options in calls)
    assert "job 7 [baseline]: succeeded" in output
    assert "job 8 [candidate]: skipped — prerequisite job 6 failed" in output
    assert "Jobs not reported as succeeded: 8" in output
    assert "Scores do not establish job completion" in output
    assert "1 runs across 1 env(s)" in output


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(1, ["mlq"], stderr="daemon unavailable"),
    FileNotFoundError("mlq executable missing"),
    subprocess.TimeoutExpired(["mlq"], timeout=30),
])
def test_queue_query_failure_exits_before_scoring(scorer, monkeypatch, capsys, failure):
    def run(*args, **kwargs):
        raise failure
    def unexpected_scan(*args):
        raise AssertionError("must not infer a job state from score data after a query failure")
    monkeypatch.setattr(scorer.subprocess, "run", run)
    monkeypatch.setattr(scorer, "find_runs", unexpected_scan)
    with pytest.raises(SystemExit) as result:
        scorer.main(["available", "--jobs", "7"])
    assert result.value.code == 1
    output = capsys.readouterr()
    assert "Could not query mlq job 7:" in output.err
    assert "runs across" not in output.out


@pytest.mark.parametrize("payload", ["not JSON", "[]", '{"id": 8, "state": "succeeded"}',
                                      '{"id": 7}', '{"id": 7, "state": "failed", "stateReason": 5}'])
def test_malformed_job_response_is_an_explicit_error(scorer, monkeypatch, payload):
    monkeypatch.setattr(scorer.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, payload, ""))
    with pytest.raises(RuntimeError, match="Could not query mlq job 7"):
        scorer.print_job_states([7])


def test_nonpositive_job_id_does_not_query_queue(scorer, monkeypatch):
    def unexpected_query(*args, **kwargs):
        raise AssertionError("invalid job IDs must be rejected before calling mlq")
    monkeypatch.setattr(scorer.subprocess, "run", unexpected_query)
    with pytest.raises(SystemExit) as result:
        scorer.main(["available", "--jobs", "0"])
    assert result.value.code == 2

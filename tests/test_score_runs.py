"""Score reporting, exact TensorBoard histories, and bounded log retention."""

import importlib
import json
from pathlib import Path
import subprocess

import tracemalloc

import numpy as np
import pytest
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("score_runs")


def write_events(directory, events):
    writer = EventFileWriter(str(directory))
    try:
        for event in events:
            writer.add_event(event)
    finally:
        writer.close()


def scalar_event(step, **values):
    return Event(wall_time=float(step), step=step, summary=Summary(value=[
        Summary.Value(tag=tag, simple_value=value) for tag, value in values.items()
    ]))


def existing_run(scorer, monkeypatch, tmp_path):
    directory = tmp_path / "HalfCheetah-v4__available__1__100"
    write_events(directory, [
        scalar_event(500_000, **{scorer.TAG: 10.0}),
        scalar_event(1_000_000, **{scorer.TAG: 20.0}),
    ])
    result = scorer.load_returns(directory)
    monkeypatch.setattr(scorer, "find_runs", lambda *args: [directory])
    return directory, result


def test_partial_comparison_names_missing_patterns_and_empty_logs(scorer, monkeypatch, capsys, tmp_path):
    available, result = existing_run(scorer, monkeypatch, tmp_path)
    empty = tmp_path / "HalfCheetah-v4__empty__1__100"
    write_events(empty, [])
    monkeypatch.setattr(scorer, "find_runs", lambda *args: [available, empty])
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


def test_jobs_are_queried_once_and_scores_do_not_override_failed_state(scorer, monkeypatch, capsys, tmp_path):
    existing_run(scorer, monkeypatch, tmp_path)
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


def test_selected_scalars_preserve_restart_and_window_semantics(scorer, tmp_path):
    directory = tmp_path / "HalfCheetah-v4__restart__1__100"
    write_events(directory, [
        scalar_event(10, **{scorer.TAG: 1.0, "unused": 99.0}),
        scalar_event(20, **{scorer.TAG: 200.0}),
        Event(wall_time=30, step=20, session_log=SessionLog(status=SessionLog.START)),
        scalar_event(20, **{scorer.TAG: 3.0}),
        scalar_event(40, **{scorer.TAG: 5.0}),
    ])
    selected = scorer.RunScalars(directory, tags={scorer.TAG})
    full = scorer.RunScalars(directory)
    assert selected.tags == {scorer.TAG}
    for actual, expected in zip(selected.series(scorer.TAG), full.series(scorer.TAG)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(selected.series(scorer.TAG)[1], [1, 3, 5])
    assert selected.window_stats(scorer.TAG, 20, 10) == full.window_stats(scorer.TAG, 20, 10)
    assert selected.value_near(scorer.TAG, 30) == 3.0


def test_scoring_many_runs_does_not_retain_their_event_histories(scorer, tmp_path, monkeypatch, capsys):
    # Reusing a frozen log makes retained-memory growth deterministic without
    # a large fixture on disk. Each matching entry is still scored independently.
    directory = tmp_path / "HalfCheetah-v4__memory__1__100"
    write_events(directory, [
        scalar_event(step, **{scorer.TAG: float(step), "unused": float(step)})
        for step in range(4000)
    ])
    tracemalloc.start()
    try:
        monkeypatch.setattr(scorer, "find_runs", lambda *args: [directory])
        scorer.main(["memory", "--at", "2k", "--at-window", "100"])
        _, single_peak = tracemalloc.get_traced_memory()
        capsys.readouterr()
        tracemalloc.reset_peak()
        monkeypatch.setattr(scorer, "find_runs", lambda *args: [directory] * 24)
        scorer.main(["memory", "--at", "2k", "--at-window", "100"])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert "24 runs across 1 env(s)" in capsys.readouterr().out
    growth = peak - single_peak
    assert growth < 8 * 1024 * 1024, f"additional runs retained {growth / 1024**2:.1f} MiB"


def test_metric_cells_use_reloaded_coverage_after_a_restart(scorer, tmp_path, monkeypatch, capsys):
    directory = tmp_path / "HalfCheetah-v4__restart__1__100"
    write_events(directory, [
        scalar_event(1_000_000, **{scorer.TAG: 100.0, "charts/SPS": 1000.0}),
    ])
    original_load = scorer.load_returns

    def load_then_restart(path, last_n, **kwargs):
        result = original_load(path, last_n, **kwargs)
        write_events(directory, [
            Event(step=0, wall_time=2_000_000, session_log=SessionLog(status=SessionLog.START)),
            scalar_event(100_000, **{scorer.TAG: 10.0, "charts/SPS": 100.0}),
        ])
        return result

    monkeypatch.setattr(scorer, "load_returns", load_then_restart)
    scorer.main(["restart", "--runs-dir", str(tmp_path), "--metrics", "charts/SPS"])
    row = next(line for line in capsys.readouterr().out.splitlines() if line.strip().startswith("1  restart"))
    assert row.split()[-1] == "--"

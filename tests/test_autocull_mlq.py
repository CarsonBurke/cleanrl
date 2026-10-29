"""Queue supervisor failures must be visible and never counted as cancellations."""
import json
import subprocess

import pytest

from scripts import autocull


def test_status_and_cancel_use_checked_public_cli(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = json.dumps({"jobs": [{"id": 9}]}) if "status" in command else "cancelled 9\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(autocull.subprocess, "run", run)
    assert autocull.mlq_jobs() == [{"id": 9}]
    assert autocull.mlq_cancel(9, force=True) == "cancelled 9"
    assert calls[0][0] == ["mlq", "status", "--json"]
    assert calls[1][0] == ["mlq", "cancel", "9", "--force"]
    assert all(kwargs["check"] for _, kwargs in calls)


def test_failed_cancel_propagates_instead_of_reporting_success(monkeypatch):
    def failed(command, **kwargs):
        assert kwargs["check"]
        raise subprocess.CalledProcessError(1, command, stderr="cannot reach mlqd")

    monkeypatch.setattr(autocull.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError, match="exit status 1"):
        autocull.mlq_cancel(9, force=False)


@pytest.mark.parametrize("watch", [False, True])
def test_cli_failure_preserves_stderr_and_only_watch_retries(monkeypatch, capsys, watch):
    args = ["autocull"] + (["--watch", "1"] if watch else [])
    monkeypatch.setattr(autocull.sys, "argv", args)

    def failed(*args):
        raise subprocess.CalledProcessError(1, ["mlq", "status", "--json"],
                                            stderr="cannot reach mlqd at /custom/runtime/mlqd.sock")

    def stop_after_retry(seconds):
        raise StopIteration("test stops the retry loop")

    monkeypatch.setattr(autocull, "sweep", failed)
    monkeypatch.setattr(autocull.time, "sleep", stop_after_retry)
    with pytest.raises(StopIteration if watch else subprocess.CalledProcessError):
        autocull.main()
    stderr = capsys.readouterr().err
    assert "cannot reach mlqd at /custom/runtime/mlqd.sock" in stderr
    assert ("retrying" if watch else "exiting") in stderr


@pytest.mark.parametrize("flags,expected", [([], 3_000_000),
                                            (["--min-steps", "2M"], 2_000_000),
                                            (["--collapse-min-steps", "200k"], 200_000)])
def test_cli_collapse_grace_inherits_performance_grace_unless_overridden(monkeypatch, flags, expected):
    monkeypatch.setattr(autocull.sys, "argv", ["autocull", *flags])
    captured = []
    monkeypatch.setattr(autocull, "sweep", lambda args, *rest: captured.append(args))
    autocull.main()
    assert captured[0].collapse_min_steps == expected
    assert captured[0].min_steps == (2_000_000 if "2M" in flags else 3_000_000)

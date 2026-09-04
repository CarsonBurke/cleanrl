import os
from pathlib import Path

import pytest

from cleanrl_utils.checkpointing import (
    CheckpointCadence,
    atomic_hardlink,
    atomic_save,
    atomic_write_bytes,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_checkpoint_cadence_waits_for_ten_minutes_and_deduplicates_terminal_state():
    clock = FakeClock()
    cadence = CheckpointCadence(clock=clock)

    assert not cadence.periodic_due()
    clock.now = 599.999
    assert not cadence.periodic_due()
    clock.now = 600.0
    assert cadence.periodic_due()

    cadence.record_commit(12)
    assert not cadence.needs_terminal_commit(12)
    assert cadence.needs_terminal_commit(13)
    clock.now = 1199.999
    assert not cadence.periodic_due()
    clock.now = 1200.0
    assert cadence.periodic_due()


def test_checkpoint_cadence_rejects_intervals_outside_five_to_ten_minutes():
    for interval in (299.999, 600.001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="between 300 and 600 seconds"):
            CheckpointCadence(interval_seconds=interval)


def test_atomic_save_keeps_committed_checkpoint_when_serialization_fails(tmp_path: Path):
    checkpoint = tmp_path / "latest.model"
    checkpoint.write_bytes(b"committed")

    def fail_after_partial_write(staging_path: Path) -> None:
        staging_path.write_bytes(b"partial")
        raise RuntimeError("serialization failed")

    with pytest.raises(RuntimeError, match="serialization failed"):
        atomic_save(checkpoint, fail_after_partial_write)

    assert checkpoint.read_bytes() == b"committed"
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_rolling_checkpoint_retention_and_final_alias_do_not_copy_payload(tmp_path: Path):
    latest = tmp_path / "offline-latest.model"
    final = tmp_path / "offline-final.model"

    atomic_write_bytes(latest, b"first")
    atomic_write_bytes(latest, b"second")
    assert list(tmp_path.iterdir()) == [latest]

    atomic_hardlink(latest, final)
    assert latest.read_bytes() == final.read_bytes() == b"second"
    assert os.stat(latest).st_ino == os.stat(final).st_ino

    atomic_write_bytes(latest, b"third")
    assert latest.read_bytes() == b"third"
    assert final.read_bytes() == b"second"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "offline-final.model",
        "offline-latest.model",
    ]

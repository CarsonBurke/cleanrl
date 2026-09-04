import os
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MIN_CHECKPOINT_INTERVAL_SECONDS = 300.0
MAX_CHECKPOINT_INTERVAL_SECONDS = 600.0
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 600.0
_UNCOMMITTED = object()


@dataclass
class CheckpointCadence:
    """Wall-clock cadence for checkpoints committed at safe training boundaries."""

    interval_seconds: float = DEFAULT_CHECKPOINT_INTERVAL_SECONDS
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    last_committed_state: Any = field(default=_UNCOMMITTED, init=False)
    _next_checkpoint_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.interval_seconds)
            or not MIN_CHECKPOINT_INTERVAL_SECONDS
            <= self.interval_seconds
            <= MAX_CHECKPOINT_INTERVAL_SECONDS
        ):
            raise ValueError(
                "checkpoint interval must be finite and between "
                f"{MIN_CHECKPOINT_INTERVAL_SECONDS:g} and "
                f"{MAX_CHECKPOINT_INTERVAL_SECONDS:g} seconds"
            )
        self._next_checkpoint_at = self.clock() + self.interval_seconds

    def periodic_due(self) -> bool:
        return self.clock() >= self._next_checkpoint_at

    def record_commit(self, state: Any) -> None:
        self.last_committed_state = state
        self._next_checkpoint_at = self.clock() + self.interval_seconds

    def needs_terminal_commit(self, state: Any) -> bool:
        return self.last_committed_state is _UNCOMMITTED or self.last_committed_state != state

def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_save(path: str | os.PathLike[str], save: Callable[[Path], None]) -> None:
    """Write with a same-directory staging file and atomically replace ``path``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    staging_path = Path(staging_name)
    try:
        save(staging_path)
        with staging_path.open("rb") as staging_file:
            os.fsync(staging_file.fileno())
        os.replace(staging_path, destination)
        _fsync_directory(destination.parent)
    finally:
        staging_path.unlink(missing_ok=True)


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> None:
    atomic_save(path, lambda staging_path: staging_path.write_bytes(payload))


def atomic_hardlink(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    """Atomically publish an immutable alias without copying or reserializing it."""

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    staging_path = Path(staging_name)
    staging_path.unlink()
    try:
        os.link(source_path, staging_path)
        os.replace(staging_path, destination_path)
        _fsync_directory(destination_path.parent)
    finally:
        staging_path.unlink(missing_ok=True)

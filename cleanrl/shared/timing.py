"""Zero-sync phase timing for rollout/update-loop breakdowns.

STANDARD: new versions MUST report per-phase time (at minimum ``env``,
``rollout`` and ``update`` totals per logging interval) instead of a single
SPS scalar, so the next optimization target is chosen empirically. Usage::

    timer = PhaseTimer()
    for iteration in ...:
        with timer.span("env", use_cuda=False):
            ... vector env steps ...
        with timer.span("rollout"):
            ... policy forwards ...
        with timer.span("update"):
            ... loss + optimizer ...
        if should_log:
            stats = timer.summary()   # the ONLY synchronization point
            ... log stats["env"]["total_s"], etc. ...
            timer.reset()

Contracts:
- No synchronization happens inside :meth:`span`. CPU phases use
  ``perf_counter``; CUDA phases record ``torch.cuda.Event`` pairs whose
  elapsed time is resolved lazily in :meth:`summary` after one
  ``torch.cuda.synchronize``. Never call ``summary`` (or ``.item()`` /
  ``.cpu()`` on metric tensors) inside the optimizer path.
- ``summary`` accumulates into running totals; call :meth:`reset` after
  logging to start the next window.
- Spans must not interleave (no span inside another span); sequential reuse
  of a name accumulates. :meth:`start` / :meth:`stop` expose the same span
  for regions that do not align with a block (e.g. around a loop).
"""

import time
from contextlib import contextmanager

import torch


class PhaseTimer:
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self._pending_cuda = []  # (name, start_event, end_event)
        self._event_pool = {}  # device -> completed event pairs reusable next interval
        self._open = None

    def start(self, name, use_cuda=True):
        if self._open is not None:
            raise RuntimeError("PhaseTimer spans must not interleave")
        if use_cuda and torch.cuda.is_available():
            device = torch.cuda.current_device()
            pool = self._event_pool.setdefault(device, [])
            if pool:
                start, end = pool.pop()
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._open = ("cuda", name, (device, start, end))
        else:
            self._open = ("cpu", name, time.perf_counter())

    def stop(self):
        if self._open is None:
            raise RuntimeError("PhaseTimer.stop without a matching start")
        kind, name, start = self._open
        self._open = None
        if kind == "cuda":
            device, start, end = start
            end.record()
            self._pending_cuda.append((name, device, start, end))
        else:
            self._add(name, time.perf_counter() - start)

    @contextmanager
    def span(self, name, use_cuda=True):
        self.start(name, use_cuda=use_cuda)
        try:
            yield self
        finally:
            self.stop()

    def _add(self, name, seconds):
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self):
        """Resolve events once per participating device, then recycle pairs."""
        if self._pending_cuda:
            for device in {entry[1] for entry in self._pending_cuda}:
                torch.cuda.synchronize(device)
            for name, device, start, end in self._pending_cuda:
                self._add(name, start.elapsed_time(end) / 1000.0)
                self._event_pool[device].append((start, end))
            self._pending_cuda.clear()
        return {
            name: {"total_s": total, "calls": self.counts[name]}
            for name, total in self.totals.items()
        }

    def reset(self):
        self.totals.clear()
        self.counts.clear()
        self._pending_cuda.clear()

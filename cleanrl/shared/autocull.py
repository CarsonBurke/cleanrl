"""Conservative, framework-agnostic early-cull decision hook.

Call at coherent evaluation boundaries with already-smoothed, comparable metrics.
The hook never terminates work; the training framework owns pruning, checkpointing,
and process exit semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal, Mapping, cast
import json
from pathlib import Path
from statistics import median

__all__ = ["AutoCullHook", "CullDecision", "MetricSpec", "ProxyCull", "ProxyPruned"]

Mode = Literal["max", "min"]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    mode: Mode
    min_delta: float

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        if (
            isinstance(self.min_delta, bool)
            or not isinstance(self.min_delta, (int, float))
            or not isfinite(self.min_delta)
            or self.min_delta < 0
        ):
            raise ValueError("min_delta must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CullDecision:
    should_cull: bool
    evaluations: int
    stale_evaluations: int
    improved_metrics: tuple[str, ...]
    reason: str | None = None


class AutoCullHook:
    """Cull after no configured signal materially improves for a full window."""

    _STATE_VERSION = 1

    def __init__(
        self,
        metrics: Mapping[str, MetricSpec],
        *,
        warmup_evaluations: int,
        patience_evaluations: int,
    ) -> None:
        if not metrics or any(
            not isinstance(name, str) or not name for name in metrics
        ):
            raise ValueError("at least one nonempty string metric name is required")
        if (
            isinstance(warmup_evaluations, bool)
            or not isinstance(warmup_evaluations, int)
            or warmup_evaluations < 1
        ):
            raise ValueError("warmup_evaluations must be an integer of at least 1")
        if (
            isinstance(patience_evaluations, bool)
            or not isinstance(patience_evaluations, int)
            or patience_evaluations < 2
        ):
            raise ValueError("patience_evaluations must be an integer of at least 2")

        self.metrics = dict(metrics)
        self.warmup_evaluations = warmup_evaluations
        self.patience_evaluations = patience_evaluations
        self._evaluations = 0
        self._best: dict[str, float] = {}
        self._stale_evaluations = 0
        self._culled = False

    def __call__(self, **metrics: float | None) -> CullDecision:
        return self.update(metrics)

    def update(self, metrics: Mapping[str, float | None]) -> CullDecision:
        observed: dict[str, float] = {}
        missing: list[str] = []
        for name in self.metrics:
            value = metrics.get(name)
            if value is None:
                missing.append(name)
                continue
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"metric {name!r} must be finite")
            observed[name] = numeric

        if missing:
            raise ValueError(
                "evaluation is missing configured metrics: " + ", ".join(missing)
            )
        if self._culled:
            return self._decision(())

        self._evaluations += 1
        if self._evaluations <= self.warmup_evaluations:
            # Anchor patience at the end of warmup, not at an early noisy extreme.
            self._best.update(observed)
            return self._decision(())

        improved: list[str] = []
        for name, value in observed.items():
            best = self._best.get(name)
            if best is None or self._improved(value, best, self.metrics[name]):
                self._best[name] = value
                improved.append(name)

        if improved:
            self._stale_evaluations = 0
        else:
            self._stale_evaluations += 1
            self._culled = self._stale_evaluations >= self.patience_evaluations

        return self._decision(tuple(improved))

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self._STATE_VERSION,
            "signature": self._signature(),
            "evaluations": self._evaluations,
            "best": dict(self._best),
            "stale_evaluations": self._stale_evaluations,
            "culled": self._culled,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("version") != self._STATE_VERSION:
            raise ValueError("unsupported autocull state version")
        if state.get("signature") != self._signature():
            raise ValueError("autocull state does not match this hook configuration")

        try:
            evaluations = state["evaluations"]
            stale = state["stale_evaluations"]
            raw_best = state["best"]
            culled = state["culled"]
        except KeyError as error:
            raise ValueError(f"autocull state is missing {error.args[0]!r}") from error

        if (
            isinstance(evaluations, bool)
            or not isinstance(evaluations, int)
            or isinstance(stale, bool)
            or not isinstance(stale, int)
            or not isinstance(raw_best, Mapping)
            or not isinstance(culled, bool)
            or evaluations < 0
            or stale < 0
            or stale > max(0, evaluations - self.warmup_evaluations)
            or (culled and stale != self.patience_evaluations)
            or (not culled and stale >= self.patience_evaluations)
        ):
            raise ValueError("invalid autocull state")

        best: dict[str, float] = {}
        for name, value in raw_best.items():
            if name not in self.metrics:
                raise ValueError(f"unknown metric in autocull state: {name!r}")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"non-finite metric in autocull state: {name!r}")
            best[str(name)] = numeric

        expected_metrics = set(self.metrics) if evaluations else set()
        if set(best) != expected_metrics:
            raise ValueError("autocull state has incomplete metric baselines")

        self._evaluations = evaluations
        self._best = best
        self._stale_evaluations = stale
        self._culled = culled

    @staticmethod
    def _improved(value: float, best: float, spec: MetricSpec) -> bool:
        delta = value - best if spec.mode == "max" else best - value
        return delta > spec.min_delta

    def _decision(self, improved: tuple[str, ...]) -> CullDecision:
        reason = None
        if self._culled:
            reason = (
                "no configured metric materially improved for "
                f"{self._stale_evaluations} evaluation(s) after "
                f"{self.warmup_evaluations} warmup evaluation(s)"
            )
        return CullDecision(
            should_cull=self._culled,
            evaluations=self._evaluations,
            stale_evaluations=self._stale_evaluations,
            improved_metrics=improved,
            reason=reason,
        )

    def _signature(self) -> dict[str, object]:
        return {
            "metrics": {
                name: {"mode": spec.mode, "min_delta": spec.min_delta}
                for name, spec in self.metrics.items()
            },
            "warmup_evaluations": self.warmup_evaluations,
            "patience_evaluations": self.patience_evaluations,
        }


PRUNED_EXIT_CODE = 75


class ProxyPruned(Exception):
    """Intentional resource stop; CLI exits 75, never reports training success."""

    def __init__(self, record: dict):
        self.record = record
        super().__init__(record["reason"])


class ProxyCull:
    """Per-candidate, phase-local proxy progress guard.

    Uses the portable AutoCullHook from the queue-ml-jobs skill. All signals
    are losses to minimize, smoothed in observation-time (not log-call count).
    An arm is pruned only when EVERY candidate is stale; improvement in ANY
    configured metric protects its candidate. Plateaus are resource decisions,
    not claims that the attained score is scientifically uninteresting.

    Defaults: median of three intervals, then 8192-observation EMA half-life,
    16384 observation warmup, then
    three consecutive stale evaluations; each evaluation requires another
    4096 observations. min_delta is explicit in each task's normalized units.
    Scheduled regime changes reset history before the new regime is judged.
    Nonfinite candidates are disqualified individually, not silently reported
    as successful, and cannot protect a dead grid.
    """

    def __init__(self, candidates: int, metrics: Mapping[str, float]):
        if candidates < 1:
            raise ValueError("at least one proxy candidate required")
        self.metrics = {name: MetricSpec("min", delta) for name, delta in metrics.items()}
        if not self.metrics:
            raise ValueError("at least one proxy metric required")
        self.candidates = candidates
        self.phase = None
        self.last_step = 0
        self.phase_start = 0
        self.last_evaluation = 0
        self.smoothed: list[dict[str, float]] = [{} for _ in range(candidates)]
        self.recent = [{name: [] for name in self.metrics} for _ in range(candidates)]
        self.invalid = [False] * candidates
        self.hooks = self._hooks()

    def _hooks(self):
        return [AutoCullHook(self.metrics, warmup_evaluations=1, patience_evaluations=3)
                for _ in range(self.candidates)]

    def observe(self, step: int, metrics: Mapping[str, list[float]], *, phase: str = "stationary",
                phase_start: int = 0) -> dict | None:
        if step <= self.last_step or not 0 <= phase_start < step:
            raise ValueError("proxy steps must increase and follow the phase start")
        if set(metrics) != set(self.metrics) or any(len(v) != self.candidates for v in metrics.values()):
            raise ValueError("proxy metrics must cover every configured candidate and metric")
        changed = phase != self.phase
        if changed:
            self.phase, self.phase_start = phase, phase_start
            self.hooks = self._hooks()
            self.smoothed = [{} for _ in range(self.candidates)]
            self.recent = [{name: [] for name in self.metrics} for _ in range(self.candidates)]
            self.invalid = [False] * self.candidates
            self.last_evaluation = phase_start
        start = max(self.last_step, self.phase_start)
        decay = .5 ** ((step - start) / 8192)
        self.last_step = step
        for i in range(self.candidates):
            values = {name: float(rows[i]) for name, rows in metrics.items()}
            if not all(isfinite(value) for value in values.values()):
                self.invalid[i] = True
            if self.invalid[i]:
                continue
            for name, value in values.items():
                window = self.recent[i][name]
                window.append(value)
                if len(window) > 3:
                    del window[0]
                value = median(window)
                old = self.smoothed[i].get(name)
                self.smoothed[i][name] = value if old is None else decay * old + (1 - decay) * value
        if all(self.invalid):
            return self._record(step, "all proxy candidates became nonfinite", [])
        if step - self.phase_start < 16384 or step - self.last_evaluation < 4096:
            return None
        self.last_evaluation = step
        decisions = []
        for i, hook in enumerate(self.hooks):
            if self.invalid[i]:
                decisions.append(None)
                continue
            state = hook.state_dict()
            best = cast(dict[str, float], state["best"])
            # A stale candidate is still being trained while another survives.
            # It may recover; do not let the base hook's terminal latch hide it.
            if state["culled"] and any(
                    best[name] - self.smoothed[i][name] > spec.min_delta
                    for name, spec in self.metrics.items()):
                hook = AutoCullHook(self.metrics, warmup_evaluations=1, patience_evaluations=3)
                self.hooks[i] = hook
            decisions.append(hook.update(self.smoothed[i]))
        if all(decision is None or decision.should_cull for decision in decisions):
            return self._record(step, "all proxy candidates plateaued after warmup and full patience",
                                [None if d is None else asdict(d) for d in decisions])
        return None

    def _record(self, step, reason, decisions):
        return {"status": "pruned", "reason": reason, "step": step, "phase": self.phase,
                "decisions": decisions, "policy_state": self.state_dict(),
                "exit_code": PRUNED_EXIT_CODE,
                "queue_contract": "submit with --max-attempts 1; mlq records exit75 as failed, artifact status is pruned; after-success children must not run"}

    def state_dict(self):
        return {"version": 1, "metrics": {name: asdict(spec) for name, spec in self.metrics.items()},
                "candidates": self.candidates, "phase": self.phase, "phase_start": self.phase_start,
                "last_step": self.last_step, "last_evaluation": self.last_evaluation,
                "smoothed": [dict(values) for values in self.smoothed], "invalid": self.invalid.copy(),
                "recent": [{name: list(values) for name, values in candidate.items()}
                           for candidate in self.recent], "median_window_intervals": 3,
                "smoothing_halflife_observations": 8192, "warmup_observations": 16384,
                "evaluation_stride_observations": 4096, "patience_evaluations": 3,
                "hooks": [hook.state_dict() for hook in self.hooks]}


def prune_proxy(root: Path, arm: str, record: dict) -> None:
    """Persist the decision before unwinding; callers save partial model/results."""
    record = {**record, "arm": arm}
    path = root / "autocull.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print("AUTOCULL " + json.dumps(record, allow_nan=False), flush=True)
    raise ProxyPruned(record)
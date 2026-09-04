"""Supervise live mlq jobs from their TensorBoard metrics and cull the losers.

    python scripts/autocull.py                                  # dry-run verdicts, exit 0
    python scripts/autocull.py --ref ppoadvnorm_batch_v1 --yes   # actually cancel
    python scripts/autocull.py --ref ppoadvnorm_batch_v1 --yes --watch 600

Why this lives here and not in mlq: the queue is machine-wide and domain-agnostic
(other repos submit to the same daemon), so it must not learn what
`charts/episodic_return` means. It already exposes everything a policy layer
needs -- `mlq status --json` for job state and argv, `mlq cancel` for control --
so the metric knowledge stays next to the metrics, reusing `_runs.py`.

Safety model (this thing kills jobs, so it is deliberately timid):
  * dry-run unless --yes; every decision is appended to runs/.autocull.jsonl.
  * only jobs whose cwd is THIS repo are considered; other repos are invisible.
  * only envs listed in --calibrated-envs get performance verdicts at all; every threshold
    here was measured on HalfCheetah-v4 and they do NOT transfer (see PLATEAU CALIBRATION).
  * the two verdicts that can cancel under the default --enforce health are `plateau`, which
    requires flatness AND being far behind AND >= --plateau-min-steps (1.5M), and the health
    checks nan/stall. `dead` and `behind` are reported but require --enforce all. NOTE this
    means culling CAN happen before --min-steps (3M): plateau has its own earlier grace.
  * --protect keeps named runs off the block, but it DEFAULTS TO EMPTY: only an explicit
    --ref is auto-protected. Baselines and controls are not protected unless you name them.
  * a run at-or-above the reference is never culled, however flat it is: plateauing at the
    top is success, not failure. The reference itself is never judged (self-comparison).

Verdicts, in precedence order:
  nan      non-finite return or value loss -> the run is destroyed, not slow.
  stall    no tfevents write for --stall secs while mlq says "running" -> hung.
  dead     inside [--dead-steps, --dead-until] (600k..1M), below --dead-ratio (0.35) of the
           gain the REFERENCE made from the same starting line. REPORTED ONLY by default:
           the rule is falsified as a health check (see DEAD IS NOT SEPARABLE below).
  behind   matched-step progress below --min-ratio x reference for --strikes consecutive
           checkpoints (--at). The strike requirement is what makes this robust: one bad
           checkpoint is variance, two is a trend. Requires --enforce all.
  plateau  flat AND far behind: normalized slope over --plateau-slope-window (on a faster
           --plateau-ema-halflife EMA) below --plateau-eps, AND progress below
           --plateau-floor x the SAME-ENV reference. Both required. Has its own, earlier
           --plateau-min-steps grace and is evaluated BEFORE the --min-steps gate, since a
           long flat far below the bar is a different object from a run that is still rising.

THE YARDSTICK IS PROGRESS, NOT VALUE RATIO. Comparisons use
    (run - origin) / (ref - origin),   origin = the reference's first EMA sample
because episodic return has no meaningful zero and starts NEGATIVE on every MuJoCo task, so
`run / ref` is unusable exactly when culling pays most. See progress_ratio() for the measured
justification; it agrees with the old value ratio to <=0.02 wherever that was valid, so the
table below still calibrates the thresholds.

CALIBRATION -- why the performance defaults are so lenient (measured, this repo,
HalfCheetah-v4 seed 1, the Ent-PPO line):
  variant                          @2M     @4M      @8M     vs ref @2M
  ppoadvnorm_batch_v1 (ref)       5012    6716     8455        1.00x
  entppo_v1 a=0.3                 4808    7958     9939        0.96x  <- a tight early
                                                                       gate would cull
                                                                       this; v2 a=0.3
                                                                       b=0.3 below is the
                                                                       actual winner
  entppo_v2 a=0.3 b=0.3           5443    8207    10269        1.09x
  entppo_v1 a=0.1                 3942      --       --        0.79x  (culled)
The eventual +18% winner was BEHIND the reference at 2M. Entropy-regularized and
exploration-heavy methods trade early return for a higher asymptote, so a tight
early matched-step gate is a winner-killer: --min-steps defaults to 3M and
--min-ratio to 0.8 for exactly this reason. Tighten only with seeds to back it.

AN EARLY PERFORMANCE GATE WAS TESTED AND REJECTED. Hypothesis: catastrophic early deficits
are separable from late bloomers, so a ~0.5x gate at ~500k could reclaim GPU slots ~2.5M steps
sooner. Measured (dead_calib.py) as progress at the run's own step, against how each run
turned out -- "corpse" = hand-killed as not learning, "learner" = kept improving:
  run                                    kind      400k   500k   600k   800k     1M     2M
  dblockactor_v2_dblock                  corpse    0.18   0.20   0.24      -      -      -
  dblockactor_v1_dblock                  corpse    0.12      -      -      -      -      -
  tpomd_spreadtemp_v28_r2                learner   0.81   1.02   1.24   1.41   1.45      -
  dblockvalue_v1_e2e                     learner   0.78   0.86   0.89   0.93   0.95   0.98
  entppo_v2 a=0.3 b=0.0                  learner   0.78   0.85   0.92   1.04   1.09   1.16
  dblockvalue_v1_dblock                  learner   0.71   0.78   0.88   0.92   0.88   0.87
  entppo_v2 a=0.3 b=0.3                  learner   0.60   0.63   0.69   0.78   0.80   0.96
  entppo_v2 a=0.6 b=0.3                  learner   0.41   0.39   0.51   0.64   0.70   0.81
  worst learner / best corpse            margin   +0.23  +0.19  +0.27
The ranges OVERLAP at every early gate (a 0.39 learner sits BELOW a 0.24 corpse's neighbours,
and a06_b03 recovers 0.39 -> 0.81), so no gate catches the corpses without killing learners:
an early PERFORMANCE gate is rejected. An earlier draft concluded that "a run that never
leaves the starting line" IS separable and that --dead-steps 600k / --dead-ratio 0.35 had the
widest measured margin. The next section falsifies that on a wider sample, which is why
`dead` is no longer enforced by default.
Note the corpses' own EMA slopes at 600k are +398%/1M and +155%/1M -- POSITIVE, because
norm_slope divides by max(|mean|,1) and their mean return is near zero. A slope conjunction
therefore adds nothing early; it is only meaningful once returns are far from zero, which is
why `plateau` is gated behind --plateau-min-steps (1.5M).

DEAD IS NOT SEPARABLE -- why `dead` is reported but not enforced by default. The table above
is n=2 on the corpse side and n=6 on the learner side, and a wider sweep breaks it outright:
`sf_pgvec_v1` sits at progress 0.239 / 0.234 / 0.250 / 0.261 / 0.270 at 600k..1M and finishes
at 7778 = 0.84x of the best curve on disk. Its in-window minimum (0.234) is within 0.003 of
the only measured corpse's 600k value (0.231), so no threshold on progress-level tells the
two classes apart, and slope cannot help: near-zero returns make norm_slope report +398%/1M
for corpses. The verdict is kept because it is informative in dry-run and in the decision
log, and it is bounded to [--dead-steps, --dead-until] so it can no longer act as an
unbounded performance gate; enforcing it needs --enforce all and a better rule.

ENFORCEMENT: --yes enables action, --enforce chooses which verdicts may act. Default "health"
is nan/stall/plateau. `dead` and `behind` are reported only -- both are performance judgments
whose separation from late bloomers is not established here (see above and CALIBRATION).

PLATEAU CALIBRATION (measured, this repo; reproduce with `--audit <patterns>`):
The first version of this rule did the OPPOSITE of its job: it kept localteacher_v1_8M,
which burned 1.5M steps flat (3670 -> 3659, 2.0M -> 3.5M) at 0.70x, and flagged
emarms_k2_8M, which finished at 8386 = 0.91x of the best curve on disk. Four defects, fixed:
  1. --ref defaulted to None, and judge() enforces only nan/stall without a reference, so the
     DEFAULT invocation could not detect a plateau at all. A same-env reference is now
     auto-selected from finished, full-length curves when --ref is absent.
  2. The slope was fit over 2M, which straddles the earlier rise: the flat run read +13.0%/1M.
     The plateau test now has its own shorter window (--plateau-slope-window).
  3. eps and the EMA were mismatched. On the 400k level EMA the flat run reads +2.9%/1M and
     the champion +2.5%/1M -- no separation at any eps. On the 150k slope EMA the same flat
     run reads -1.1%/1M, so eps 0.02 both catches it and spares dblockvalue_v1_dblock
     (+4.8%/1M at rel 0.73), which this file's own table labels a learner.
  4. --plateau-floor 0.95 condemns every healthy run's final third, when curves flatten with
     the LR anneal.
Floor headroom, measured against the auto reference (final rel, and the minimum rel after
1.5M): hopsd_v27_sdgate 0.885 / 0.898, emarms_k2_8M 0.906 / 0.909, advcond_k2p0_a4c4_8m
0.836 / 0.834, advcond_v5_a4c4_mb128 0.913 / 0.788. So the floor's real margin to the worst
observed near-winner is 0.038, not the 0.15 an earlier draft of this note claimed.
Slope half-life, first detection on the labelled plateau (250k grid, `--audit`):
  400k -> 3.5M | 250k -> 3.2M | 150k -> 3.2M
i.e. 150k buys nothing over 250k; an earlier draft claimed 3.00M/2.75M, which does not
reproduce. 150k is kept as the default because it is the setting the floor/eps pair above was
measured with. The level tests keep --ema-halflife 400k.
WHOLE-DISK VALIDATION (`--audit <env>` over every run dir on disk, 250k grid, counting only
verdicts that would actually CANCEL under the default --enforce):
  HalfCheetah-v4  849 judged: 666 never, 183 plateau (615 dirs not evaluable)
  Walker2d-v4       6 judged: 0 cancelled     Hopper-v4  6 judged: 0 cancelled
Before the --calibrated-envs gate those last two were 6-of-6 and 5-of-6 cancelled, including
both envs' PPO baselines. On HalfCheetah the cancellation frontier sits where it should: all
25 top finishers (finals 9703..10966) survive, the highest-finishing run that would be
cancelled is hopsd_v2 at 6732 = 0.73x, and sf_pgvec_v1 (0.84x, the slow starter that
falsified `dead`) survives with its dead verdict reported but not enforced.
A note on `--audit` output: "never" means judged-and-survived; a run with no return samples
or no same-env reference prints "not evaluated" instead, because "survived 0 checkpoints"
reads like evidence and is not.

CROSS-ENV COMPARISON IS A WINNER-KILLER, and this tool used to do it: a single --ref was
applied to every live job regardless of task, so a Hopper arm was judged "DEAD at 0.11x" of
a HalfCheetah bar it could never reach at any quality (HalfCheetah tops ~9k, Hopper ~3.5k).
References are now per env, must be finished, must be full-length, and must span at least
--ref-min-span of the judged run; otherwise the run gets health-only enforcement.

EMA detail that matters: `charts/episodic_return` is sampled irregularly (per
episode, 16 envs), so a fixed-span EMA silently weights fast phases differently
from slow ones. Decay here is per-sample 0.5 ** (dstep / halflife), i.e. defined
in STEP space, which makes EMAs comparable across runs and across time.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runs import (  # noqa: E402
    RETURN_TAG,
    find_runs,
    fmt_step,
    last_active,
    load_run,
    parse_step,
    parse_steps,
    run_timestamp,
)

VALUE_LOSS_TAG = "losses/value_loss"
DECISION_LOG = "runs/.autocull.jsonl"


# --------------------------------------------------------------------------- #
# mlq interface (public CLI only)
# --------------------------------------------------------------------------- #
def mlq_jobs() -> list[dict]:
    out = subprocess.run(
        ["mlq", "status", "--json"], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out).get("jobs", [])


def mlq_cancel(job_id: int, force: bool) -> str:
    cmd = ["mlq", "cancel", str(job_id)] + (["--force"] if force else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def exp_name_of(job: dict) -> str:
    """Recover --exp-name from argv; fall back to the mlq job name."""
    args = job.get("args") or []
    for i, a in enumerate(args):
        if a == "--exp-name" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--exp-name="):
            return a.split("=", 1)[1]
    return job.get("name", "")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ema_series(steps: np.ndarray, vals: np.ndarray, halflife: int) -> np.ndarray:
    """Step-aware EMA: per-sample decay 0.5 ** (dstep / halflife).

    Irregular sampling is the norm here (episodes end when they end), so decay
    is defined in step space rather than per sample.
    """
    if vals.size == 0:
        return vals
    out = np.empty_like(vals, dtype=np.float64)
    out[0] = vals[0]
    d = np.diff(steps).astype(np.float64)
    alpha = 1.0 - np.power(0.5, np.maximum(d, 0.0) / float(halflife))
    for i in range(1, vals.size):
        out[i] = out[i - 1] + alpha[i - 1] * (vals[i] - out[i - 1])
    return out


def ema_at(steps: np.ndarray, ema: np.ndarray, step: int) -> float | None:
    """EMA value at-or-before `step`; None if the run never got there."""
    if ema.size == 0 or steps[0] > step:
        return None
    idx = int(np.searchsorted(steps, step, side="right")) - 1
    return float(ema[max(idx, 0)])


def norm_slope(steps: np.ndarray, ema: np.ndarray, window: int) -> float | None:
    """Least-squares EMA slope over the trailing `window` steps, per 1M steps,
    normalized by the window's mean level -> a dimensionless growth rate."""
    if ema.size < 8:
        return None
    lo = steps[-1] - window
    mask = steps >= lo
    if mask.sum() < 8:
        return None
    x = steps[mask].astype(np.float64)
    y = ema[mask]
    if x[-1] - x[0] < window * 0.5:  # not enough span to call a trend
        return None
    slope = float(np.polyfit(x, y, 1)[0]) * 1e6
    return slope / max(abs(float(np.mean(y))), 1.0)


class RunView:
    """Return-curve view of one run dir: EMA, matched-step lookups, health."""

    def __init__(self, run_dir: Path, halflife: int):
        self.run_dir = run_dir
        # Run dirs are `{env_id}__{exp}__{seed}__{ts}`. The env is load-bearing for judging:
        # returns are not comparable across tasks (HalfCheetah tops ~9k, Hopper ~3.5k), so a
        # cross-env reference is a winner-killer. Measured: a Hopper arm was called DEAD at
        # "0.11x" of a HalfCheetah bar it could never reach at any quality.
        self.env = run_dir.name.split("__")[0]
        r = load_run(run_dir)
        self.steps, self.vals = r.series(RETURN_TAG)
        self.ema = ema_series(self.steps, self.vals, halflife)
        self.max_step = int(self.steps[-1]) if self.steps.size else 0
        # The curve's own starting line, used as the common origin for progress ratios.
        # First EMA sample, i.e. the mean of the first episodes to finish across the vec-env.
        self.origin = float(self.ema[0]) if self.ema.size else 0.0
        self.age = time.time() - last_active(run_dir)
        vl = r.latest(VALUE_LOSS_TAG)
        self.broken = (
            (self.vals.size > 0 and not np.isfinite(self.vals[-1]))
            or (vl is not None and not np.isfinite(vl))
        )
        self.halflife = halflife
        self._alt: dict[int, np.ndarray] = {}

    def truncated(self, step: int) -> "RunView":
        """This curve as a live sweep would have seen it at `step`.

        The point of --audit: a verdict is a claim about a DECISION MADE IN TIME, so judging
        only a finished curve's last point cannot answer "would this rule have killed the
        winner while it ran?". Shares the parsed arrays; slicing is a view, not a reload.
        """
        m = self.steps <= step
        t = object.__new__(RunView)
        t.run_dir, t.env, t.halflife, t._alt = self.run_dir, self.env, self.halflife, {}
        t.steps, t.vals, t.ema = self.steps[m], self.vals[m], self.ema[m]
        t.max_step = int(t.steps[-1]) if t.steps.size else 0
        t.origin = float(t.ema[0]) if t.ema.size else 0.0
        t.age = 0.0          # a historical slice cannot be "stalled"
        t.broken = False     # nan-ness is judged on the live curve, not a replay
        return t

    def at(self, step: int) -> float | None:
        return ema_at(self.steps, self.ema, step)

    def slope(self, window: int, halflife: int | None = None) -> float | None:
        """Normalized trailing slope, optionally on a shorter-memory EMA.

        The plateau test wants a FASTER EMA than the level tests: with the 400k half-life
        used for `dead`/`behind`, the average still carries the pre-plateau rise, so a curve
        flat from 2.0M read +28.8%/1M at 2.5M and only crossed the flatness threshold at
        3.5M. Measured on this repo's labelled set, a 150k half-life for the slope alone
        moves detection to 2.75M with zero winners flagged at any checkpoint (7 winners,
        137 checkpoints). The level tests keep 400k so the calibrated dead/behind thresholds
        stay valid.
        """
        if halflife is None or halflife == self.halflife:
            return norm_slope(self.steps, self.ema, window)
        if halflife not in self._alt:
            self._alt[halflife] = ema_series(self.steps, self.vals, halflife)
        return norm_slope(self.steps, self._alt[halflife], window)


def resolve_run(exp: str, runs_dir: Path, halflife: int, since: float | None = None) -> RunView | None:
    """Run dir for an exp-name, or None when it cannot be attributed unambiguously.

    Two ways the naive "most recently active match" is a cancel hazard:

    * RELAUNCH. A job is `state==running` for the whole of imports/env construction
      before its SummaryWriter dir exists. In that window the only match is the
      PREVIOUS run of the same exp-name, whose `age` is hours, so the stall rule
      cancels the new job for its predecessor's staleness. `since` (the job's
      createdAt) fixes it: a dir stamped before the job existed cannot be its output.
    * MULTI-SEED. Dirs are `{env}__{exp}__{seed}__{ts}`, so `__{exp}__` matches every
      seed and `max(..., key=last_active)` would attach one seed's curve to all of
      them -- exactly the sweep the CALIBRATION note asks for. If more than one
      candidate survives the `since` filter we refuse to judge instead of guessing.
    """
    matched = find_runs([runs_dir], [f"__{exp}__"])
    if not matched:
        matched = find_runs([runs_dir], [exp])
    if since is not None:
        matched = [d for d in matched if (run_timestamp(d) or 0.0) >= since]
        if len(matched) != 1:
            return None
    if not matched:
        return None
    return RunView(max(matched, key=last_active), halflife)


def auto_reference(runs_dir: Path, halflife: int, env: str, min_steps: int,
                   scan: int, min_age: float) -> RunView | None:
    """Strongest FINISHED, FULL-LENGTH curve of `env` as the bar, when --ref was not given.

    Without this, `--ref` defaulting to None made the whole supervisor structurally
    incapable of its main job: `judge()` returns "only stall/nan enforced" whenever there is
    no reference, so a plateaued or hopelessly-behind run could never be caught by the
    default invocation. It reaped corpses and nothing else.

    Three filters, each measured to matter:

    * FINISHED. `age >= min_age` (the stall threshold). Without it a run whose tfevents was
      written 0.7 s ago was eligible, so a live arm's early peak could become the standard
      that judges its own siblings.
    * FULL LENGTH. Candidates are restricted to the longest tier before ranking by height.
      Ranking by `(final_ema, max_step)` does NOT do this -- length breaks only exact ties in
      final EMA, so a 3.2M arm peaking at 8214 outranked an 8.0M curve at 8188 and became
      the bar. Every comparison then clamps to `min(step, 3.2M)` against a value nothing
      else reaches until ~6M, which PLATEAU'd `entppo_v2_declkl_ppoadvnorm_a03_b00` at 3.0M
      -- a run that finishes at 9896, rank 3 of 93.
    * STRONGEST of that tier: the highest final EMA. A weak bar cannot cull anything, so
      erring toward the strongest full-length curve is the conservative direction only for
      false negatives; the length filter is what protects against false positives.

    Bounded scan: `runs/` accumulates thousands of dirs over a campaign and each RunView
    parses tfevents, so an exhaustive scan takes minutes and made this unusable in a
    --watch loop. Candidates are the `scan` most recently active dirs. NOTE this window is
    load-bearing: launching more than `scan` fresh arms evicts the historical champion from
    the candidate set and the bar silently becomes the best of the new sweep, so --ref-scan
    is exposed on the CLI.
    """
    matched = find_runs([runs_dir], [f"{env}__"])
    matched.sort(key=last_active, reverse=True)
    cands: list[RunView] = []
    for d in matched[:scan]:
        try:
            rv = RunView(d, halflife)
        except Exception:
            continue  # unreadable/partial dir: not a yardstick
        if rv.broken or rv.max_step < min_steps or rv.steps.size < 8 or rv.age < min_age:
            continue
        cands.append(rv)
    if not cands:
        return None
    longest = max(c.max_step for c in cands)
    tier = [c for c in cands if c.max_step >= 0.95 * longest]
    return max(tier, key=lambda r: r.at(r.max_step) or -1e30)



class ReferencePicker:
    """Per-env bar to judge against, resolved once per env and reused.

    Three rules, all learned from measured misjudgments:

    * SAME ENV ONLY. An explicit --ref is honoured only for runs of its own env; a run whose
      env has no reference gets health-only enforcement rather than a comparison against an
      unreachable number. (A Hopper arm was once called "DEAD at 0.11x" of a HalfCheetah bar.)
    * A RUN IS NEVER ITS OWN BAR in auto mode. Comparing a curve to itself reads 1.00x and
      makes the best live arm unjudgeable, which is safe, but reporting it as a comparison
      would be a lie. The per-env bar is cached WITHOUT any run-specific exclusion so every
      run of an env is judged against the same number; self-reference is dropped at the end.
    * THE BAR MUST BE AT LEAST AS LONG AS THE RUN. `ema_at` returns the value at-or-before
      the query step, so past a short bar's end every comparison silently freezes at its last
      value and the ratio becomes nonsense. Measured: with a 2.1M bar, an 8.0M run read
      "2.31x" and `localteacher_v1_8M` -- the run this whole rewrite exists to catch -- read
      1.04x and escaped. Refusing to judge is the honest outcome: a yardstick shorter than
      the thing being measured cannot measure it.
    """

    def __init__(self, a, runs_dir: Path):
        self.a = a
        self.runs_dir = runs_dir
        self.explicit = resolve_run(a.ref, runs_dir, a.ema_halflife) if a.ref else None
        if a.ref and self.explicit is None:
            print(f"! reference {a.ref!r} not found under {runs_dir}", file=sys.stderr)
        self._cache: dict[str, RunView | None] = {}

    def for_run(self, run: RunView) -> tuple[RunView | None, str]:
        """(bar, why-not) -- `why-not` is a printable reason when the bar is None."""
        if self.explicit is not None:
            if self.explicit.env != run.env:
                return None, f"--ref is {self.explicit.env}, run is {run.env}"
            ref = self.explicit
        else:
            if run.env not in self._cache:
                self._cache[run.env] = auto_reference(
                    self.runs_dir, self.a.ema_halflife, run.env, self.a.min_steps,
                    self.a.ref_scan, self.a.ref_min_age,
                )
            ref = self._cache[run.env]
            if ref is None:
                return None, f"no finished {run.env} curve reaching {fmt_step(self.a.min_steps)}"
            if ref.run_dir == run.run_dir:
                return None, "this run IS the reference"
        if ref.max_step < run.max_step * self.a.ref_min_span:
            return None, (
                f"reference {ref.run_dir.name.split('__')[1]} stops at "
                f"{fmt_step(ref.max_step)}, short of this run's {fmt_step(run.max_step)}"
            )
        return ref, ""


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def progress_ratio(mine: float, theirs: float, origin: float) -> float | None:
    """Fraction of the reference's OWN improvement that this run has achieved.

        (mine - origin) / (theirs - origin),   origin = the reference's first EMA sample

    Why not the obvious `mine / theirs`: episodic return is not a ratio scale. It has no
    meaningful zero, and on every MuJoCo task it STARTS NEGATIVE and crosses zero mid-run,
    which makes a value ratio meaningless exactly when culling is most valuable. Measured on
    this repo's HalfCheetah reference (progress_metric.py):
        @100k ref_ema=-219   @200k -162   @300k -60   @400k +131   @500k +323
    so a `theirs <= 0` guard makes the tool INERT before ~400k, and where returns straddle
    zero the ordering inverts: at 500k the eventual 1.23x WINNER (entppo_v1_a03) scores value
    ratio 0.33 while an eventual 0.87x loser (dblockvalue_v1_dblock) scores 0.63, and two runs
    score NEGATIVE ratios. Progress ratio orders the same three 0.61 / 0.78 / 0.39.

    It is a drop-in: where the value ratio IS valid the two agree to <=0.02 across every run
    measured (@2M/4M/8M: 0.88/0.89, 1.17/1.16, 1.23/1.22, 0.80/0.81, 0.70/0.71, 0.54/0.55,
    0.86/0.87, 0.88/0.88, 0.87/0.87, 0.98/0.98, 1.27/1.26, 1.09/1.08), so the CALIBRATION
    table above and every threshold below carry over unchanged.

    Returns None when the reference has not yet improved on its own starting line (denominator
    <= 0): a yardstick of zero length cannot rank anything, so we abstain instead of guessing.
    """
    denom = theirs - origin
    if denom <= 0:
        return None
    return (mine - origin) / denom


def judge(run: RunView, ref: RunView | None, a) -> tuple[str, str]:
    """Return (verdict, reason). verdict "keep" means leave the job alone."""
    if run.broken:
        return "nan", "non-finite episodic_return/value_loss"
    if run.age > a.stall:
        return "stall", f"no tfevents write for {run.age / 60:.0f}m (limit {a.stall / 60:.0f}m)"
    if ref is None:
        if run.max_step < a.min_steps:
            return "keep", f"grace: {fmt_step(run.max_step)} < {fmt_step(a.min_steps)}"
        return "keep", "no reference curve; only stall/nan enforced"
    # PERFORMANCE VERDICTS REQUIRE A CALIBRATED ENV. Every threshold in this file was measured
    # on HalfCheetah-v4. Exporting them to another task is a winner-killer, because the
    # inter-arm spread relative to the leader differs per env: replaying judge() over every
    # run on disk with these defaults culled 6 of 6 judgeable Walker2d runs (including
    # `ppo_baseline_walker`, the control, and `pmpo_magpos_rank_walker_v2`, the best
    # non-reference arm) and 5 of 6 Hopper runs (including its #1 and #2 finishers), against
    # 65 of 93 HalfCheetah runs left alone. Per-env REFERENCES were not enough; the
    # thresholds themselves do not transfer. Envs outside --calibrated-envs get health-only
    # enforcement until someone measures a table for them.
    if a.calibrated_envs and run.env not in a.calibrated_envs:
        return "keep", (
            f"{run.env} has no calibrated thresholds "
            f"(--calibrated-envs {','.join(a.calibrated_envs)}); only stall/nan enforced"
        )
    origin = ref.origin

    # DEAD: reported only, not enforced by default. The intended reading is "the run has
    # barely moved off the starting line long after the reference left it", but the claim
    # that this is separable from slow starters is FALSIFIED: sf_pgvec_v1 sat at 0.234-0.270
    # across this whole window and finished at 0.84x, within 0.003 of the only measured
    # corpse's 600k value. See DEAD IS NOT SEPARABLE in the module docstring.
    #
    # BOUNDED to [--dead-steps, --dead-until]. Without the upper bound this rule is an
    # unbounded performance gate -- precisely the thing the table below says was TESTED AND
    # REJECTED -- because `max_step >= dead_steps` stays true forever and the ratio is
    # recomputed every sweep. Measured consequence: it fired at 1.8M-2.5M on runs climbing at
    # +30% to +108%/1M (hindsight_opsd_betanll_walker_vmpostack dipped to 0.32 at 2.0M and
    # recovered to 0.74 by 3.8M; advcond_v6_emarms_Hopper_8M was condemned from 1.8M through
    # 3.2M while its slope rose to +108%/1M). The calibration table only covers 400k-1M, so
    # the rule may only speak there. Past --dead-until, `plateau` and `behind` take over, and
    # both require evidence that this rule lacks: flatness, or repeated strikes.
    if a.dead_steps <= run.max_step <= a.dead_until:
        cp = min(run.max_step, ref.max_step)
        mine, theirs = run.at(cp), ref.at(cp)
        if mine is not None and theirs is not None:
            pr = progress_ratio(mine, theirs, origin)
            if pr is not None and pr < a.dead_ratio:
                return "dead", (
                    f"progress {pr:.2f} < {a.dead_ratio:.2f} of the reference's own gain at "
                    f"{fmt_step(cp)} ({mine:.0f} vs {theirs:.0f}, origin {origin:.0f})"
                )

    # PLATEAU. Rewritten after this rule was measured doing the exact opposite of its job:
    # it kept localteacher_v1_8M (flat 3670->3659 from 2.0M to 3.5M, 0.70x ref) while
    # flagging emarms_k2_8M, which finished at 8386 = 0.91x of the best curve on disk. Three
    # causes, all fixed here:
    #   1. The slope was fit over a 2M window, which for a run at 3.5M straddles
    #      the earlier RISE and averaged a 1.5M-long flat into "+13.0%/1M". The plateau test
    #      uses its own, shorter window so it measures the recent past only.
    #   2. `--plateau-eps 0.02` cannot separate anything: measured, the 1.5M-flat run reads
    #      +2.9%/1M and the eventual champion reads +2.5%/1M at 8M. Slope alone is not the
    #      discriminator -- the LEVEL is (0.70x vs 0.97x), so eps only has to mean "not
    #      visibly climbing" and the floor does the judging.
    #   3. `--plateau-floor 0.95` condemned any run whose curve flattens as its LR anneal
    #      ends -- i.e. every healthy run's final third, including a 0.91x near-winner.
    # This test runs BEFORE the `--min-steps` grace and uses its own, earlier
    # `--plateau-min-steps`: AGENTS.md's winner-killer warning is about culling runs that are
    # still RISING, and a run flat for a full window AND far below the bar is a different
    # object. Without this ordering `--plateau-min-steps` was unreachable dead code.
    slope = run.slope(a.plateau_slope_window, a.plateau_ema_halflife)
    here = run.at(run.max_step)
    ref_here = ref.at(min(run.max_step, ref.max_step))
    rel = None if (here is None or ref_here is None) else progress_ratio(here, ref_here, origin)
    if (
        slope is not None
        and rel is not None
        and run.max_step >= a.plateau_min_steps
        and slope < a.plateau_eps
        and rel < a.plateau_floor
    ):
        return "plateau", (
            f"slope {slope * 100:+.1f}%/1M < {a.plateau_eps * 100:.1f}% over the last "
            f"{fmt_step(a.plateau_slope_window)} and {here:.0f} = {rel:.2f}x ref progress "
            f"(< {a.plateau_floor:.2f}) at {fmt_step(run.max_step)}"
        )

    if run.max_step < a.min_steps:
        return "keep", (
            f"grace: {fmt_step(run.max_step)} < {fmt_step(a.min_steps)}"
            + (f", {rel:.2f}x ref" if rel is not None else "")
            + (f", slope {slope * 100:+.1f}%/1M" if slope is not None else "")
        )

    strikes: list[str] = []
    for cp in a.at:
        if run.max_step < cp:
            break
        mine, theirs = run.at(cp), ref.at(cp)
        ratio = None if (mine is None or theirs is None) else progress_ratio(mine, theirs, origin)
        if ratio is None:
            strikes.clear()  # a skipped checkpoint is not evidence; adjacency required
            continue
        if ratio < a.min_ratio:
            strikes.append(f"@{fmt_step(cp)} {mine:.0f}/{theirs:.0f}={ratio:.2f}")
        else:
            strikes.clear()  # a recovery breaks the streak; only trends kill
        if len(strikes) >= a.strikes:
            return "behind", f"{len(strikes)} consecutive checkpoints < {a.min_ratio:.2f}x ref: " + ", ".join(strikes)

    # (plateau is evaluated ABOVE, before the --min-steps grace gate)
    return "keep", (
        f"{fmt_step(run.max_step)} ema={here:.0f}"
        + (f" ({rel:.2f}x ref)" if rel is not None else "")
        + (f" slope={slope * 100:+.1f}%/1M" if slope is not None else "")
    )


def enforceable(verdict: str, a) -> bool:
    """Whether `verdict` may cancel under the current --enforce.

    Shared by sweep() and audit() so the safety validation and the actor can never disagree
    about which rules actually kill jobs -- the audit's whole value is that its "never culled"
    means "would not have been cancelled".
    """
    return verdict in ("nan", "stall", "plateau") or a.enforce == "all"


def log_decision(root: Path, record: dict) -> None:
    path = root / DECISION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def prior_cancels(root: Path) -> dict[int, float]:
    """job id -> epoch secs of the last cancel we issued (for --force escalation)."""
    path = root / DECISION_LOG
    out: dict[int, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        # Per-record guard, not just json.loads: the log is append-only and re-read every
        # iteration, so one malformed record (schema change, partial write, interleaved
        # writer) killed the supervisor on this and every subsequent start. int()/float()
        # were outside the old try.
        try:
            rec = json.loads(line)
            if rec.get("action") == "cancel":
                out[int(rec["job"])] = float(rec["t"])
        except (ValueError, TypeError, KeyError):
            continue
    return out


def audit(a, root: Path) -> None:
    """Replay each named run through judge() at EVERY checkpoint it passed through.

    This is the check that the original plateau rule failed and `--replay` could not catch:
    --replay judges a finished curve at its final point, but a supervisor decides while the
    run is alive, so the winner-killer question is "at any step, would this config have
    cancelled a run that went on to win?". Report the FIRST non-keep verdict and the number
    of checkpoints survived. A winner that reports anything but "never" is a rejected config.
    """
    runs_dir = root / "runs"
    picker = ReferencePicker(a, runs_dir)
    a.stall = float("inf")
    print(f"audit: every {fmt_step(a.audit_every)} from {fmt_step(a.audit_from)}, "
          f"ref={a.ref or 'auto (best same-env curve)'}, "
          f"plateau={a.plateau_eps * 100:.0f}%/{fmt_step(a.plateau_slope_window)}"
          f"@<{a.plateau_floor:.2f}x from {fmt_step(a.plateau_min_steps)}, "
          f"dead={a.dead_ratio}@{fmt_step(a.dead_steps)}")
    for pat in a.audit:
        seen: dict[str, Path] = {}
        for d in find_runs([runs_dir], [f"__{pat}__"]) + find_runs([runs_dir], [pat]):
            seen[d.name] = d
        if not seen:
            print(f"  ?      {pat}: no run dir")
            continue
        for d in sorted(seen.values(), key=lambda p: p.name):
            run = RunView(d, a.ema_halflife)
            # A dir with no return samples cannot be judged OR reported: `run.at()` is None
            # and formatting it raised TypeError mid-report, aborting every pattern after it
            # -- on the CLEAN branch, i.e. exactly while an operator confirms a config safe.
            if run.steps.size == 0:
                print(f"  ----   {d.name}: no episodic_return samples; not evaluated")
                continue
            ref, why = picker.for_run(run)
            if ref is None:
                print(f"  ----   {d.name}: not evaluated ({why})")
                continue
            first, reported, n = None, None, 0
            for cp in range(a.audit_from, run.max_step + 1, a.audit_every):
                t = run.truncated(cp)
                if t.steps.size < 8:
                    continue
                n += 1
                verdict, reason = judge(t, ref, a)
                if verdict == "keep":
                    continue
                # Separate "would have been CANCELLED" from "would have been printed". The
                # winner-killer question is about cancellation, and `dead`/`behind` are
                # reported-only under the default --enforce, so folding them in here would
                # overstate the danger and hide which rule actually acts.
                if enforceable(verdict, a) and first is None:
                    first = (cp, verdict, reason)
                elif not enforceable(verdict, a) and reported is None:
                    reported = (cp, verdict)
            if first:
                cp, verdict, reason = first
                print(f"  {verdict.upper():6s} {d.name}\n"
                      f"         first @{fmt_step(cp)} of {n} checkpoints: {reason}")
            elif n == 0:
                # "never culled across 0 checkpoints" reads like evidence and is not.
                print(f"  ----   {d.name}: not evaluated (0 checkpoints at or after "
                      f"{fmt_step(a.audit_from)}; run reached {fmt_step(run.max_step)})")
            else:
                note = ""
                if reported is not None:
                    note = (f"; {reported[1]} reported (not enforced) from "
                            f"{fmt_step(reported[0])}")
                print(f"  never  {d.name}\n"
                      f"         survived {n} checkpoints to {fmt_step(run.max_step)} "
                      f"(final {run.at(run.max_step):.0f}, "
                      f"ref {ref.at(min(run.max_step, ref.max_step)):.0f}){note}")


def replay(a, root: Path) -> None:
    """Print the verdict this policy would give named runs, live or finished.

    Threshold calibration tool: `--replay <winner>` answers "would these settings have killed
    the run that actually won?" without touching the queue. The stall rule is disabled here
    because a finished run is stale by definition.

    EVERY matching dir is judged, not just the most recently active one. `resolve_run` collapses
    to one dir because the sweep path must attach exactly one curve to one job; here a pattern
    is a human's query and quietly dropping its other matches hides precisely the runs you are
    calibrating against (a seed sweep, or `dblockvalue_v1` matching both the _dblock and _e2e
    arms -- the loser of which is the interesting one).
    """
    runs_dir = root / "runs"
    picker = ReferencePicker(a, runs_dir)
    a.stall = float("inf")
    print(f"replay: ref={a.ref or 'auto (best same-env curve)'}"
          f" min_steps={fmt_step(a.min_steps)} min_ratio={a.min_ratio} strikes={a.strikes}"
          f" dead={a.dead_ratio}@{fmt_step(a.dead_steps)}"
          f" plateau={a.plateau_eps * 100:.0f}%/{fmt_step(a.plateau_slope_window)}"
          f"@<{a.plateau_floor:.2f}x from {fmt_step(a.plateau_min_steps)}")
    for exp in a.replay:
        # UNION, not `a or b`: `__{exp}__` is the precise form (whole variant field) but an
        # `or` lets it SHADOW the substring form, so `--replay dblockactor_v2` would report
        # three aborted 0-step dirs whose variant happens to end in that text and silently
        # drop dblockactor_v2_dblock -- the run the query was about.
        seen: dict[str, Path] = {}
        for d in find_runs([runs_dir], [f"__{exp}__"]) + find_runs([runs_dir], [exp]):
            seen[d.name] = d
        matched = list(seen.values())
        if not matched:
            print(f"  ?     {exp}: no run dir")
            continue
        for d in sorted(matched, key=lambda p: p.name):
            run = RunView(d, a.ema_halflife)
            ref, why = picker.for_run(run)
            verdict, reason = judge(run, ref, a)
            reason += (f"  [ref {ref.run_dir.name.split('__')[1]}]" if ref is not None
                       else f"  [no ref: {why}]")
            print(f"  {'keep ' if verdict == 'keep' else verdict.upper():5s} {d.name}\n"
                  f"        {reason}")


def sweep(a, root: Path) -> int:
    runs_dir = root / "runs"
    picker = ReferencePicker(a, runs_dir)
    issued = prior_cancels(root)
    culled = 0
    rows = []
    for job in mlq_jobs():
        if job.get("state") != "running":
            continue
        cwd = job.get("cwd")
        if not cwd or Path(cwd).resolve() != root:
            continue  # another repo's job, or unattributable: not ours to judge
        exp = exp_name_of(job)
        if a.only and not any(p in exp for p in a.only):
            continue
        if any(p in exp for p in a.protect):
            rows.append((job["id"], exp, "keep", "protected"))
            continue
        created = job.get("createdAt")
        run = resolve_run(exp, runs_dir, a.ema_halflife,
                          since=float(created) / 1000.0 if created else None)
        if run is None:
            rows.append((job["id"], exp, "keep", "no run dir newer than the job, or >1 seed matched"))
            continue
        ref, why = picker.for_run(run)
        verdict, reason = judge(run, ref, a)
        reason += (f"  [ref {ref.run_dir.name.split('__')[1]} @{fmt_step(ref.max_step)}]"
                   if ref is not None else f"  [no ref: {why}]")
        rows.append((job["id"], exp, verdict, reason))
        if verdict == "keep":
            continue
        prev = issued.get(int(job["id"]))
        force = prev is not None and (time.time() - prev) > a.force_after
        # Enforcement gate. "health" = nan/stall/plateau. Two deliberate placements:
        #
        # `plateau` IS enforced: it requires BOTH flatness over a 1M window (on the fast slope
        # EMA) AND sitting under 0.75x of a same-env, at-least-as-long reference, from 1.5M
        # steps on. Replaying judge() over every run on disk leaves all 25 top HalfCheetah
        # finishers (finals 9703-10966) untouched and still catches the 1.5M-step flat that
        # motivated this rewrite.
        #
        # `dead` is NOT enforced, despite reading like a health check. Its premise -- "below
        # 0.35x of the reference's gain early means never-learning" -- is falsified by
        # `sf_pgvec_v1`, which sat at progress 0.239 / 0.234 / 0.250 / 0.261 / 0.270 across
        # the whole 600k-1M window and finished at 7778 = 0.84x, i.e. a slow starter, not a
        # corpse. Worse, its in-window minimum (0.234) is within 0.003 of the only measured
        # corpse's 600k value (0.231), so NO threshold on progress-level separates the two
        # classes, and the corpse side of the table is n=2. Slope cannot rescue it either:
        # near-zero returns make norm_slope report +398%/1M for corpses. So `dead` is
        # demoted to a reported-only verdict (visible in dry-run and in .autocull.jsonl,
        # actionable under --enforce all) until someone measures a rule that actually
        # separates. `behind` stays opt-in for the same reason: no flatness requirement.
        actionable = a.yes and enforceable(verdict, a)
        if actionable:
            msg = mlq_cancel(int(job["id"]), force)
            culled += 1
            log_decision(root, {
                "t": time.time(), "action": "cancel", "job": job["id"], "exp": exp,
                "verdict": verdict, "reason": reason, "force": force,
                "step": run.max_step, "mlq": msg,
            })
        else:
            log_decision(root, {
                "t": time.time(), "action": "would-cancel", "job": job["id"],
                "exp": exp, "verdict": verdict, "reason": reason, "step": run.max_step,
                "suppressed_by": None if a.yes else "dry-run", "enforce": a.enforce,
            })

    label = f"enforce={a.enforce}" if a.yes else "DRY-RUN (pass --yes to enforce)"
    ref_note = f"ref={a.ref}" if a.ref else "ref=auto (best same-env curve)"
    print(f"autocull: {len(rows)} live job(s), {ref_note}, {label}")
    for jid, exp, verdict, reason in rows:
        mark = "keep " if verdict == "keep" else f"{verdict.upper():5s}"
        print(f"  [{jid}] {mark} {exp}\n           {reason}")
    if a.json:
        print(json.dumps([
            {"job": j, "exp": e, "verdict": v, "reason": r} for j, e, v, r in rows
        ]))
    return culled


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", default=None, help="variant substring whose curve is the bar to beat")
    p.add_argument("--protect", default="", help="comma-separated substrings never culled")
    p.add_argument("--only", default="",
                   help="comma-separated substrings; supervise only matching experiment names")
    p.add_argument("--at", default="3M,4M,6M", help="matched-step checkpoints")
    p.add_argument("--min-ratio", type=float, default=0.8,
                   help="cull below this fraction of the reference's own gain (see progress_ratio)")
    p.add_argument("--strikes", type=int, default=2, help="consecutive failing checkpoints required")
    p.add_argument("--dead-ratio", type=float, default=0.35,
                   help="never-learning check: cull below this fraction of the reference's own gain")
    p.add_argument("--dead-steps", default="600k", help="grace period before the dead check")
    p.add_argument("--min-steps", default="3M", help="grace period before any performance verdict")
    p.add_argument("--ema-halflife", default="400k", help="EMA half-life in STEPS")
    p.add_argument("--plateau-slope-window", default="1M",
                   help="trailing window for the PLATEAU slope test. Must be short enough not to "
                        "straddle an earlier rise: at 2M this read +13%%/1M on a curve that had "
                        "been flat for 1.5M steps.")
    p.add_argument("--plateau-min-steps", default="1500k",
                   help="grace before the plateau verdict. Earlier than --min-steps on purpose: "
                        "the winner-killer risk is culling runs that are still RISING, and this "
                        "verdict additionally requires flatness.")
    p.add_argument("--plateau-ema-halflife", default="150k",
                   help="EMA half-life for the plateau SLOPE only (level tests keep "
                        "--ema-halflife). Measured: 400k/250k/150k catch the labelled plateau "
                        "at 3.50M/3.00M/2.75M with zero winners flagged at any setting.")
    p.add_argument("--plateau-eps", type=float, default=0.02,
                   help="slope floor (fraction of level per 1M steps), paired with the faster "
                        "--plateau-ema-halflife. At the 400k level half-life the labelled "
                        "plateau read +2.9%%/1M and the champion +2.5%%/1M, so 0.02 separated "
                        "nothing; on the 150k slope EMA the same plateau reads -1.1%%/1M, "
                        "which restores the margin AND spares dblockvalue_v1_dblock "
                        "(+4.8%%/1M at rel 0.73), a run this file's own table labels a learner.")
    p.add_argument("--plateau-floor", type=float, default=0.75,
                   help="plateau only culls below this x ref progress. 0.95 condemned a 0.94x "
                        "near-winner whose curve merely flattened with its LR anneal.")
    p.add_argument("--calibrated-envs", default="HalfCheetah-v4",
                   help="comma-separated env_ids whose thresholds have a measured table here. "
                        "Runs of any other env get health-only enforcement, because these "
                        "constants do NOT transfer: with them, 6 of 6 Walker2d and 5 of 6 "
                        "Hopper runs on disk are culled, including both envs' PPO baselines. "
                        "Empty string disables the gate (measure first).")
    p.add_argument("--ref-scan", type=int, default=60,
                   help="how many most-recently-active dirs per env are reference candidates. "
                        "Load-bearing: launching more than this many fresh arms evicts the "
                        "historical champion from the candidate set.")
    p.add_argument("--ref-min-age", type=float, default=900.0,
                   help="a reference's tfevents must be at least this stale (secs), i.e. the "
                        "run is over. Deliberately NOT --stall: --replay/--audit set --stall "
                        "to infinity to judge finished curves, which would reject every "
                        "candidate as 'still live'.")
    p.add_argument("--ref-min-span", type=float, default=0.95,
                   help="a reference must reach this fraction of the judged run's steps. Past "
                        "a short bar's end, ema_at freezes at its last value and ratios become "
                        "nonsense (a 2.1M bar made an 8M run read 2.31x and made the plateaued "
                        "run read 1.04x and escape).")
    p.add_argument("--dead-until", default="1M",
                   help="upper bound of the dead check's window. Its calibration table only "
                        "covers 400k-1M; unbounded, it became an early-performance gate that "
                        "culled runs climbing at +30..+108%%/1M at 1.8M-2.5M.")
    p.add_argument("--stall", type=float, default=900.0, help="seconds without tfevents writes = hung")
    p.add_argument("--force-after", type=float, default=600.0, help="escalate to --force this long after a graceful cancel")
    p.add_argument("--watch", type=float, default=0.0, help="loop forever, sleeping this many seconds")
    p.add_argument("--yes", action="store_true", help="actually cancel (default: dry-run)")
    p.add_argument("--enforce", choices=("health", "all"), default="health",
                   help="which verdicts may cancel when --yes: "
                        "health=nan/stall/dead/plateau (default), all=+behind")
    p.add_argument("--json", action="store_true")
    p.add_argument("--replay", default="", help="comma-separated run patterns: print verdicts and exit (queue untouched)")
    p.add_argument("--audit", default="",
                   help="comma-separated run patterns: replay each through judge() at every "
                        "checkpoint and report the first cull. The winner-killer test; a "
                        "known winner must report 'never'. Queue untouched.")
    p.add_argument("--audit-from", default="1M", help="first checkpoint --audit evaluates")
    p.add_argument("--audit-every", default="250k", help="checkpoint spacing for --audit")
    a = p.parse_args()

    a.at = parse_steps(a.at)
    a.min_steps = parse_step(a.min_steps)
    a.dead_steps = parse_step(a.dead_steps)
    a.ema_halflife = parse_step(a.ema_halflife)
    a.plateau_slope_window = parse_step(a.plateau_slope_window)
    a.plateau_min_steps = parse_step(a.plateau_min_steps)
    a.plateau_ema_halflife = parse_step(a.plateau_ema_halflife)
    a.protect = [s for s in a.protect.split(",") if s.strip()]
    a.only = [s for s in a.only.split(",") if s.strip()]
    if a.ref:
        a.protect.append(a.ref)
    a.replay = [s for s in a.replay.split(",") if s.strip()]
    a.audit = [s for s in a.audit.split(",") if s.strip()]
    a.audit_from = parse_step(a.audit_from)
    a.audit_every = parse_step(a.audit_every)
    a.dead_until = parse_step(a.dead_until)
    a.calibrated_envs = [s.strip() for s in a.calibrated_envs.split(",") if s.strip()]

    root = Path(__file__).resolve().parent.parent
    if a.replay:
        replay(a, root)
        return
    if a.audit:
        audit(a, root)
        return
    while True:
        try:
            sweep(a, root)
        except subprocess.CalledProcessError as e:
            print(f"autocull: mlq unavailable ({e}); retrying", file=sys.stderr)
        except Exception:
            # A supervisor that dies silently is worse than none, because the operator
            # believes plateaued arms are still being reaped. Measured killers of the old
            # loop, none of them a CalledProcessError: `mlq` absent from PATH
            # (FileNotFoundError), `mlq` printing a non-JSON banner (JSONDecodeError), and a
            # run dir archived between the glob and the stat (FileNotFoundError inside
            # last_active). Without --watch this still exits non-zero via the raise below.
            traceback.print_exc()
            print("autocull: sweep failed; continuing", file=sys.stderr)
            if not a.watch:
                raise
        if not a.watch:
            return
        sys.stdout.flush()
        time.sleep(a.watch)


if __name__ == "__main__":
    main()

"""Opt-in collapse rejection must distinguish instability from slow learning."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.autocull import RunView, enforceable, judge


def policy(**overrides):
    values = dict(stall=900, min_steps=3_000_000, calibrated_envs=["HalfCheetah-v4"],
                  collapse_ratio=0.6, collapse_drop=2000, collapse_ema_halflife=100_000,
                  enforce="health")
    values.update(overrides)
    return SimpleNamespace(**values)


def curve(values, env="HalfCheetah-v4"):
    steps = np.arange(1, len(values) + 1, dtype=np.int64) * 1000
    run = object.__new__(RunView)
    run.steps, run.vals = steps, np.asarray(values, dtype=float)
    run.max_step, run.env = int(steps[-1]), env
    run.broken, run.age = False, 0
    return run


def test_severe_collapse_remains_disqualifying_after_recovery():
    run = curve(np.r_[np.linspace(0, 8000, 3500), np.full(500, 500), np.full(1000, 9000)])
    verdict, _ = judge(run, None, policy())
    assert verdict == "collapse"
    assert enforceable(verdict, policy())
    assert judge(run, None, policy(collapse_ratio=0))[0] == "keep"
    run.env = "Walker2d-v4"
    assert judge(run, None, policy())[0] == "keep"


@pytest.mark.parametrize("values", [
    np.linspace(0, 2000, 5000),
    np.r_[np.linspace(0, 8000, 3500), -1000, np.full(1499, 8000)],
    np.r_[np.linspace(0, 8000, 3500), np.full(1500, 6000)],
    np.r_[np.linspace(0, 3000, 3500), np.full(1500, 1700)],
    np.r_[np.linspace(0, 8000, 1000), np.full(500, 500), np.full(3500, 9000)],
])
def test_slow_start_noise_small_drawdown_and_pregrace_dip_survive(values):
    assert judge(curve(values), None, policy())[0] == "keep"

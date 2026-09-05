"""Scheduling contract tests use deterministic fake physics, without ML work."""

from concurrent.futures import TimeoutError
from threading import Event

import numpy as np
import pytest

from cleanrl.shared.async_env import AsyncEnvStepper


class BlockingEnv:
    def __init__(self):
        self.entered = Event()
        self.release = Event()
        self.closed = False
        self.steps = 0
        self.failure = None

    def reset(self, *, seed=None):
        self.steps = 0
        return seed

    def step(self, actions):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test failed to release environment")
        if self.failure:
            raise self.failure
        self.steps += 1
        return actions.copy(), self.steps

    def close(self):
        self.closed = True


def test_async_step_snapshots_actions_and_prevents_reentry():
    env = BlockingEnv()
    with AsyncEnvStepper(env) as pipeline:
        assert pipeline.reset(seed=3) == 3
        actions = np.arange(6, dtype=np.float32).reshape(2, 3)
        expected = actions.copy()
        pipeline.step_async(actions)
        try:
            assert env.entered.wait(timeout=5)
            actions[:] = -1
            with pytest.raises(RuntimeError, match="pending"):
                pipeline.step_async(actions)
            with pytest.raises(RuntimeError, match="pending"):
                pipeline.reset()
            with pytest.raises(TimeoutError):
                pipeline.step_wait(timeout=0)
            with pytest.raises(RuntimeError, match="pending"):
                pipeline.step_async(actions)
        finally:
            env.release.set()
        observed, index = pipeline.step_wait()
        np.testing.assert_array_equal(observed, expected)
        assert index == 1
        with pytest.raises(RuntimeError, match="no environment step"):
            pipeline.step_wait()
        observed, index = pipeline.step(actions)
        np.testing.assert_array_equal(observed, actions)
        assert index == 2
    assert env.closed
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.reset()


@pytest.mark.parametrize("error", [ValueError("physics failed"), TimeoutError("physics failed")])
def test_async_worker_failure_propagates_and_closes(error):
    env = BlockingEnv()
    env.failure = error
    env.release.set()
    pipeline = AsyncEnvStepper(env)
    pipeline.step_async(np.zeros((2, 3)))
    with pytest.raises(type(error), match="physics failed"):
        pipeline.step_wait()
    with pytest.raises(RuntimeError, match="no environment step"):
        pipeline.step_wait()
    pipeline.close()
    pipeline.close()
    assert env.closed


def test_close_drains_outstanding_work_even_on_worker_failure():
    env = BlockingEnv()
    env.failure = ValueError("physics failed")
    env.release.set()
    pipeline = AsyncEnvStepper(env)
    pipeline.step_async(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="physics failed"):
        pipeline.close()
    assert env.closed

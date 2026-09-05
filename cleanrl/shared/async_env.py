"""Ordered CPU stepping that overlaps independent rollout work with physics.

This schedules one complete vector step on a persistent worker. It does not
pipeline policy versions, reorder transitions, or change environment RNGs.
After ``step_async(actions)``, callers can enqueue GPU rollout-buffer writes or
do host bookkeeping, then obtain the ordinary step tuple from ``step_wait()``.
Next-action inference still depends on that tuple. Native MuJoCo releases the
GIL; a Python-only environment may see no benefit from this wrapper.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np


class AsyncEnvStepper:
    """Wrap an environment with one in-flight, action-owning vector step.

    Submit/wait/reset/close must be called from one controlling thread. Do not
    inspect or mutate the wrapped environment while a step is pending. Returned
    observations have the wrapped environment's usual lifetime: snapshot them
    before the next submission if that environment reuses its output buffers.
    The action snapshot is owned here and cannot be reused until the worker
    finishes; callers may mutate their original actions immediately.

    Environments that require stepping on their creating thread (for example,
    some render contexts) are not supported. Use this for headless CPU physics.
    """

    def __init__(self, env):
        self.env = env
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-step")
        self._future = None
        self._actions = None
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _check_idle(self):
        if self._closed:
            raise RuntimeError("environment stepper is closed")
        if self._future is not None:
            raise RuntimeError("wait for the pending environment step first")

    def reset(self, *args, **kwargs):
        self._check_idle()
        return self.env.reset(*args, **kwargs)

    def step_async(self, actions):
        self._check_idle()
        actions = np.asarray(actions)
        if self._actions is None or self._actions.shape != actions.shape or self._actions.dtype != actions.dtype:
            self._actions = np.empty_like(actions)
        np.copyto(self._actions, actions)
        self._future = self._executor.submit(self.env.step, self._actions)

    def step_wait(self, timeout=None):
        if self._future is None:
            raise RuntimeError("no environment step is pending")
        try:
            result = self._future.result(timeout=timeout)
        except TimeoutError as error:
            # A wait timeout does not relinquish the worker's action buffer.
            # A TimeoutError raised *by* env.step is a completed failure.
            if self._future.done() and self._future.exception() is error:
                self._future = None
            raise
        except Exception:
            self._future = None
            raise
        self._future = None
        return result

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self, *args, **kwargs):
        if self._closed:
            return
        self._closed = True
        try:
            if self._future is not None:
                self.step_wait()
        finally:
            self._executor.shutdown(wait=True)
            self.env.close(*args, **kwargs)

    def __enter__(self):
        self._check_idle()
        return self

    def __exit__(self, *exc_info):
        self.close()

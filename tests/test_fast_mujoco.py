"""Launcher substitution and cleanup without running training or physics."""

import gymnasium as gym
import pytest

from cleanrl_utils import fast_mujoco


def test_scoped_constructor_restored_after_failure(monkeypatch):
    original = gym.vector.SyncVectorEnv
    calls = []

    def native(env_fns, **kwargs):
        calls.append((env_fns, kwargs))
        return "native instance"

    monkeypatch.setattr(fast_mujoco, "NativeMujocoVectorEnv", native)
    factories = [lambda: None]
    with pytest.raises(RuntimeError, match="trainer failed"):
        with fast_mujoco.vector_backend("native", 3):
            assert gym.vector.SyncVectorEnv(factories, copy=False) == "native instance"
            raise RuntimeError("trainer failed")
    assert gym.vector.SyncVectorEnv is original
    assert calls == [(factories, {"num_threads": 3, "copy": False})]


def test_unsupported_custom_spaces_fail_explicitly():
    with fast_mujoco.vector_backend("native", 1):
        with pytest.raises(ValueError, match="custom vector spaces"):
            gym.vector.SyncVectorEnv([], observation_space=object())


def test_unknown_backend_and_invalid_threads_leave_gym_untouched():
    original = gym.vector.SyncVectorEnv
    for backend, threads in [("other", 1), ("native", 0)]:
        with pytest.raises(ValueError):
            with fast_mujoco.vector_backend(backend, threads):
                pytest.fail("invalid configuration accepted")
        assert gym.vector.SyncVectorEnv is original


def test_sync_preserves_constructor_options(monkeypatch):
    calls = []

    def original(*args):
        calls.append(args)
        return "reference"

    monkeypatch.setattr(gym.vector, "SyncVectorEnv", original)
    with fast_mujoco.vector_backend("sync", 1):
        assert gym.vector.SyncVectorEnv is original
        assert gym.vector.SyncVectorEnv([], "obs", "act", False) == "reference"
    assert calls == [([], "obs", "act", False)]
    assert gym.vector.SyncVectorEnv is original

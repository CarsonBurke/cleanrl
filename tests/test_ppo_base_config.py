"""CPU-only execution configuration checks; no model or simulator construction."""

import pytest

from cleanrl import ppo_continuous_action as ppo


@pytest.mark.parametrize("env_id, version, expected", [
    ("HalfCheetah-v4", "0.29.1", "native"),
    ("Hopper-v4", "0.29.1", "native"),
    ("Walker2d-v4", "0.29.1", "native"),
    ("HalfCheetah-v5", "0.29.1", "sync"),
    ("dm_control/cartpole-balance-v0", "0.29.1", "sync"),
    ("Hopper-v4", "1.0.0", "sync"),
])
def test_auto_backend_uses_only_supported_native_tasks(monkeypatch, env_id, version, expected):
    monkeypatch.setattr(ppo.gym, "__version__", version)
    calls = []
    monkeypatch.setattr(ppo, "make_mujoco_vector_env", lambda *a, **kw: calls.append((a, kw)))
    ppo.make_training_env(ppo.Args(env_id=env_id), "test")
    assert calls[0][1]["backend"] == expected
    assert calls[0][1]["num_threads"] == 1  # no four-worker overhead for one env


def test_explicit_backend_render_and_thread_options_are_forwarded(monkeypatch):
    calls = []
    monkeypatch.setattr(ppo, "make_mujoco_vector_env", lambda *a, **kw: calls.append((a, kw)))
    ppo.make_training_env(ppo.Args(num_envs=16, env_threads=3, env_backend="sync", capture_video=True), "video")
    assert calls == [(("HalfCheetah-v4", 16), dict(
        backend="sync", num_threads=3, capture_video=True, run_name="video",
    ))]


@pytest.mark.parametrize("kwargs", [
    {"num_envs": 0}, {"num_steps": 0}, {"num_minibatches": 0},
    {"update_epochs": 0}, {"env_threads": 0}, {"env_backend": "unknown"},
    {"num_steps": 8, "num_minibatches": 32},
    {"num_steps": 32, "num_minibatches": 32}, {"cuda": False},
])
def test_invalid_configuration_fails_before_cuda_or_physics(kwargs):
    with pytest.raises(ValueError):
        ppo.validate_args(ppo.Args(**kwargs))


def test_valid_configuration_preserves_batch_and_optimizer_defaults():
    args = ppo.validate_args(ppo.Args())
    assert (args.batch_size, args.minibatch_size) == (2048, 64)
    assert (args.update_epochs, args.learning_rate, args.max_grad_norm) == (10, 3e-4, 0.5)

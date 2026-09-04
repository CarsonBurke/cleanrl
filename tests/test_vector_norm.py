"""Parity + perf tests for the shared vectorized normalization standard.

- Math/behavior must match the legacy per-env gym wrapper stack exactly
  (independent per-env stats, terminated-only reward returns, +-10 clip,
  final-before-reset ordering on autoreset boundaries).
- Perf test prints wrapper-stack vs vectorized throughput; it asserts parity,
  not timing (timing asserts are flaky on shared machines).
"""

import time

import gymnasium as gym
import numpy as np
import pytest

from cleanrl.shared.vector_norm import (
    VectorObsNorm,
    VectorRewardNorm,
    make_raw_continuous_env,
)

OBS_SHAPE = (17,)
N_ENVS = 4
GAMMA = 0.99


def _legacy_thunk(seed):
    def thunk():
        env = gym.make("HalfCheetah-v4")
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=GAMMA)
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        return env

    return thunk


def test_obs_math_matches_gym_running_mean_std():
    rng = np.random.default_rng(0)
    n, dim = 5, 7
    vec = VectorObsNorm(n, (dim,))
    rms = [gym.wrappers.normalize.RunningMeanStd(shape=(dim,)) for _ in range(n)]
    for _ in range(50):
        batch = rng.normal(size=(n, dim))
        out = vec.normalize(batch, out_dtype=np.float64)
        for i in range(n):
            rms[i].update(batch[i : i + 1])
            expected = np.clip(
                (batch[i] - rms[i].mean) / np.sqrt(rms[i].var + 1e-8), -10, 10
            )
            np.testing.assert_allclose(out[i], expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(vec.means[i], rms[i].mean, rtol=0, atol=1e-12)
            np.testing.assert_allclose(vec.variances[i], rms[i].var, rtol=0, atol=1e-12)
            assert vec.counts[i] == pytest.approx(rms[i].count)


def test_obs_subset_rows_match_sequential_wrapper_order():
    rng = np.random.default_rng(1)
    n, dim = 4, 5
    vec = VectorObsNorm(n, (dim,))
    rms = [gym.wrappers.normalize.RunningMeanStd(shape=(dim,)) for _ in range(n)]
    # Full-batch update, then a boundary second pass on rows 1 and 3 --
    # the autoreset order normalize_step() must reproduce.
    batch = rng.normal(size=(n, dim))
    vec.normalize(batch, out_dtype=np.float64)
    for i in range(n):
        rms[i].update(batch[i : i + 1])
    resets = rng.normal(size=(2, dim))
    out = vec.normalize(resets, rows=np.array([1, 3]), out_dtype=np.float64)
    for k, i in enumerate([1, 3]):
        rms[i].update(resets[k : k + 1])
        expected = np.clip(
            (resets[k] - rms[i].mean) / np.sqrt(rms[i].var + 1e-8), -10, 10
        )
        np.testing.assert_allclose(out[k], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(vec.means[i], rms[i].mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(vec.variances[i], rms[i].var, rtol=0, atol=1e-12)


def test_reward_math_matches_gym_normalize_reward():
    rng = np.random.default_rng(2)
    n = 6
    vec = VectorRewardNorm(n, gamma=GAMMA)
    wraps = [
        gym.wrappers.NormalizeReward(gym.make("CartPole-v1"), gamma=GAMMA)
        for _ in range(n)
    ]
    for _ in range(100):
        raw = rng.normal(size=n)
        terms = rng.random(size=n) < 0.05
        out = vec.normalize(raw, terms, out_dtype=np.float64)
        for i in range(n):
            w = wraps[i]
            # Reproduce the wrapper scalar update for env i.
            w.returns = w.returns * GAMMA * (1.0 - float(terms[i])) + raw[i]
            w.return_rms.update(np.array([w.returns], dtype=np.float64))
            expected = np.clip(raw[i] / np.sqrt(w.return_rms.var + 1e-8), -10, 10)
            assert out[i] == pytest.approx(expected, abs=1e-12)
            assert vec.returns[i] == pytest.approx(float(np.asarray(w.returns)))
            assert vec.means[i] == pytest.approx(float(w.return_rms.mean))
            assert vec.variances[i] == pytest.approx(float(w.return_rms.var))


def test_boundary_ordering_matches_wrapper_sequence():
    rng = np.random.default_rng(3)
    n, dim = 4, 6
    vec = VectorObsNorm(n, (dim,))
    rms = [gym.wrappers.normalize.RunningMeanStd(shape=(dim,)) for _ in range(n)]
    raw_next = rng.normal(size=(n, dim))
    finals = rng.normal(size=(n, dim))
    boundaries = np.array([1, 3])
    infos = {
        "final_observation": [None, finals[1], None, finals[3]],
        "_final_observation": np.array([False, True, False, True]),
    }
    next_obs, trans_obs = vec.normalize_step(
        raw_next,
        np.array([False, True, False, False]),
        np.array([False, False, False, True]),
        infos,
    )
    # Expected wrapper order: transition batch (finals at boundaries) for all
    # rows, then reset batch for boundary rows.
    transition = raw_next.copy()
    transition[boundaries] = finals[boundaries]
    for i in range(n):
        rms[i].update(transition[i : i + 1])
    expected_trans = np.clip(
        (transition - np.array([r.mean for r in rms]))
        / np.sqrt(np.array([r.var for r in rms]) + 1e-8),
        -10,
        10,
    )
    np.testing.assert_allclose(trans_obs, expected_trans.astype(np.float32), atol=1e-6)
    for k, i in enumerate(boundaries):
        rms[i].update(raw_next[i : i + 1])
    expected_next = expected_trans.copy()
    for i in boundaries:
        expected_next[i] = np.clip(
            (raw_next[i] - rms[i].mean) / np.sqrt(rms[i].var + 1e-8), -10, 10
        )
    np.testing.assert_allclose(next_obs, expected_next.astype(np.float32), atol=1e-6)
    for i in range(n):
        np.testing.assert_allclose(vec.means[i], rms[i].mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(vec.variances[i], rms[i].var, rtol=0, atol=1e-12)


def _rollout_equal(steps=200, num_envs=16, seed=1):
    legacy = gym.vector.SyncVectorEnv([_legacy_thunk(seed) for _ in range(num_envs)])
    raw = gym.vector.SyncVectorEnv(
        [make_raw_continuous_env("HalfCheetah-v4", i, False, "test") for i in range(num_envs)]
    )
    obs_norm = VectorObsNorm(num_envs, (17,))
    rew_norm = VectorRewardNorm(num_envs, gamma=GAMMA)
    try:
        legacy_obs, _ = legacy.reset(seed=seed)
        raw_obs, _ = raw.reset(seed=seed)
        next_obs = obs_norm.normalize(raw_obs)
        np.testing.assert_allclose(next_obs, legacy_obs, atol=1e-6)
        rng = np.random.default_rng(seed)
        t_legacy, t_vec = 0.0, 0.0
        for _ in range(steps):
            actions = rng.uniform(-1.0, 1.0, size=(num_envs, 6)).astype(np.float32)
            t = time.perf_counter()
            l_obs, l_rew, _, _, _ = legacy.step(actions)
            t_legacy += time.perf_counter() - t
            t = time.perf_counter()
            r_obs, r_rew, terms, truncs, infos = raw.step(actions)
            v_rew = rew_norm.normalize(r_rew, terms)
            v_obs, _ = obs_norm.normalize_step(r_obs, terms, truncs, infos)
            t_vec += time.perf_counter() - t
            np.testing.assert_allclose(v_obs, l_obs, atol=1e-6)
            np.testing.assert_allclose(v_rew, l_rew, atol=1e-6)
        return t_legacy, t_vec
    finally:
        legacy.close()
        raw.close()


def test_same_step_autoreset_assumption():
    """Guard the ordering contract: SyncVectorEnv must return reset (not
    final) obs at boundaries, with the final obs stashed in infos."""
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.wrappers.TimeLimit(gym.make("CartPole-v1"), max_episode_steps=5),
            lambda: gym.wrappers.TimeLimit(gym.make("CartPole-v1"), max_episode_steps=5),
        ]
    )
    try:
        envs.reset(seed=0)
        saw_boundary = False
        for _ in range(8):
            obs, _, terms, truncs, infos = envs.step(np.zeros(2, dtype=int))
            if np.any(np.logical_or(terms, truncs)):
                saw_boundary = True
                assert infos.get("final_observation") is not None
                for i in np.flatnonzero(np.logical_or(terms, truncs)):
                    # Same-step autoreset: live obs differs from stashed final.
                    assert not np.array_equal(obs[i], infos["final_observation"][i])
        assert saw_boundary
    finally:
        envs.close()


def test_halfcheetah_parity_and_report_speedup(capsys):
    t_legacy, t_vec = _rollout_equal()
    with capsys.disabled():
        print(
            f"\n[vector_norm] legacy wrappers: {t_legacy:.3f}s, "
            f"vectorized: {t_vec:.3f}s, speedup: {t_legacy / t_vec:.2f}x"
        )

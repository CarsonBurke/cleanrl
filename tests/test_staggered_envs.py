"""Tests for the phase-staggered env standard (exact v9/v25 semantics)."""

from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from cleanrl.shared.staggered_envs import (
    compute_phase_offsets,
    episode_horizon,
    run_phase_warmup,
)
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


def test_phase_offsets_even_spacing_and_seed_isolation():
    offsets = compute_phase_offsets(8, 1000, seed=1)
    assert sorted(offsets.tolist()) == [0, 125, 250, 375, 500, 625, 750, 875]
    # Same seed reproduces; another seed permutes (same multiset).
    np.testing.assert_array_equal(compute_phase_offsets(8, 1000, seed=1), offsets)
    other = compute_phase_offsets(8, 1000, seed=2)
    assert sorted(other.tolist()) == sorted(offsets.tolist())
    assert not np.array_equal(other, offsets)
    # Global np.random stream is untouched by the isolated RNG.
    np.random.seed(123)
    before = np.random.rand()
    compute_phase_offsets(64, 1000, seed=999)
    after = np.random.rand()
    np.random.seed(123)
    assert np.random.rand() == before
    assert np.random.rand() == after


def test_episode_horizon_reads_spec():
    assert episode_horizon("HalfCheetah-v4") == 1000
    with mock.patch(
        "cleanrl.shared.staggered_envs.gym.spec",
        return_value=SimpleNamespace(max_episode_steps=None),
    ):
        with pytest.raises(ValueError, match="finite episode horizon"):
            episode_horizon("FakeEnv-v0")


class _FakeVecEnvs:
    """Counter envs with immediate autoreset; tracks ages and reset calls."""

    def __init__(self, num_envs, horizon, obs_dim=3):
        self.num_envs = num_envs
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.ages = np.zeros(num_envs, dtype=np.int64)
        self.single_resets = []

    def reset(self, seed=None):
        self.ages[:] = 0
        return np.zeros((self.num_envs, self.obs_dim)), {}

    def step(self, actions):
        self.ages += 1
        done = self.ages >= self.horizon
        obs = np.tile(self.ages[:, None], (1, self.obs_dim)).astype(np.float64)
        finals = [None] * self.num_envs
        for i in np.flatnonzero(done):
            finals[i] = obs[i].copy()
            self.ages[i] = 0
        obs = np.tile(self.ages[:, None], (1, self.obs_dim)).astype(np.float64)
        infos = {"final_observation": finals, "_final_observation": done.copy()}
        return obs, np.ones(self.num_envs), done.copy(), np.zeros(self.num_envs, bool), infos

    def single_reset(self, index):
        self.single_resets.append(index)
        self.ages[index] = 0
        return np.zeros(self.obs_dim)


def test_warmup_establishes_phases_fake_envs():
    num_envs, horizon, seed = 4, 10, 7
    envs = _FakeVecEnvs(num_envs, horizon)
    obs_norm = VectorObsNorm(num_envs, (3,))
    rew_norm = VectorRewardNorm(num_envs, gamma=0.99)
    offsets = compute_phase_offsets(num_envs, horizon, seed)
    result = run_phase_warmup(
        envs,
        obs_norm=obs_norm,
        act_fn=lambda obs: np.zeros((num_envs, 1)),
        horizon=horizon,
        phase_offsets=offsets,
        seed=seed,
        rew_norm=rew_norm,
        single_reset=envs.single_reset,
    )
    assert result.transitions == num_envs * horizon
    # Final ages are exactly the planned offsets (scheduled reset at
    # horizon - offset leaves age == offset; no natural completion first).
    assert sorted(envs.ages.tolist()) == sorted(offsets.tolist())
    # Every env is reset exactly once (offset 0 lands on the final step).
    assert sorted(envs.single_resets) == [0, 1, 2, 3]
    np.testing.assert_array_equal(result.suppress_mask, offsets != 0)
    np.testing.assert_array_equal(result.phase_offsets, offsets)
    assert np.all(np.isfinite(result.next_obs))
    # Stat burn-in: initial reset + every step + scheduled reset for every
    # env. Offset-0 envs additionally truncate exactly on their scheduled
    # step, so they also take the autoreset boundary-row update (same-step
    # autoreset in gym 0.29: reset obs), matching v25 update order.
    expected = 1 + horizon + 1 + (offsets == 0).astype(np.float64)
    np.testing.assert_allclose(obs_norm.counts - 1e-4, expected, rtol=0, atol=1e-9)
    # Reward return stats burned in over the full warmup.
    assert np.all(rew_norm.counts - 1e-4 == horizon)


def test_warmup_halfcheetah_integration(capsys):
    import time

    num_envs, seed = 4, 1
    from cleanrl.shared.vector_norm import make_raw_continuous_env

    envs = gym.vector.SyncVectorEnv(
        [make_raw_continuous_env("HalfCheetah-v4", i, False, "test") for i in range(num_envs)]
    )
    try:
        horizon = episode_horizon("HalfCheetah-v4")
        obs_norm = VectorObsNorm(num_envs, (17,))
        rew_norm = VectorRewardNorm(num_envs, gamma=0.99)
        offsets = compute_phase_offsets(num_envs, horizon, seed)
        rng = np.random.default_rng(seed)
        t = time.perf_counter()
        result = run_phase_warmup(
            envs,
            obs_norm=obs_norm,
            act_fn=lambda obs: rng.uniform(-1, 1, size=(num_envs, 6)).astype(np.float32),
            horizon=horizon,
            phase_offsets=offsets,
            seed=seed,
            rew_norm=rew_norm,
        )
        elapsed = time.perf_counter() - t
        assert result.transitions == num_envs * horizon
        assert result.next_obs.shape == (num_envs, 17)
        assert np.all(np.isfinite(result.next_obs))
        with capsys.disabled():
            print(
                f"\n[staggered_envs] {result.transitions} warmup transitions "
                f"in {elapsed:.2f}s ({result.transitions / elapsed:.0f} TPS)"
            )
    finally:
        envs.close()
